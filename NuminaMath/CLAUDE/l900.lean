import Mathlib

namespace NUMINAMATH_CALUDE_instantaneous_velocity_at_2s_l900_90030

-- Define the distance function
def S (t : ℝ) : ℝ := 2 * (1 - t)^2

-- Define the instantaneous velocity function (derivative of S)
def v (t : ℝ) : ℝ := -4 * (1 - t)

-- Theorem statement
theorem instantaneous_velocity_at_2s :
  v 2 = 4 := by sorry

end NUMINAMATH_CALUDE_instantaneous_velocity_at_2s_l900_90030


namespace NUMINAMATH_CALUDE_notebook_calculation_sara_sister_notebooks_l900_90040

theorem notebook_calculation : ℕ → ℕ
  | initial =>
    let ordered := initial + (initial * 3 / 2)
    let after_loss := ordered - 2
    let after_sale := after_loss - (after_loss / 4)
    let final := after_sale - 3
    final

theorem sara_sister_notebooks :
  notebook_calculation 4 = 3 := by sorry

end NUMINAMATH_CALUDE_notebook_calculation_sara_sister_notebooks_l900_90040


namespace NUMINAMATH_CALUDE_min_rubles_to_win_l900_90068

/-- Represents the state of the game --/
structure GameState :=
  (score : ℕ)
  (rubles_spent : ℕ)

/-- The rules of the game --/
def apply_rule (state : GameState) (coin : ℕ) : GameState :=
  match coin with
  | 1 => ⟨state.score + 1, state.rubles_spent + 1⟩
  | 2 => ⟨state.score * 2, state.rubles_spent + 2⟩
  | _ => state

/-- Check if the game is won --/
def is_won (state : GameState) : Bool :=
  state.score = 50

/-- Check if the game is lost --/
def is_lost (state : GameState) : Bool :=
  state.score > 50

/-- The main theorem to prove --/
theorem min_rubles_to_win :
  ∃ (sequence : List ℕ),
    let final_state := sequence.foldl apply_rule ⟨0, 0⟩
    is_won final_state ∧
    final_state.rubles_spent = 11 ∧
    (∀ (other_sequence : List ℕ),
      let other_final_state := other_sequence.foldl apply_rule ⟨0, 0⟩
      is_won other_final_state →
      other_final_state.rubles_spent ≥ 11) :=
by sorry

end NUMINAMATH_CALUDE_min_rubles_to_win_l900_90068


namespace NUMINAMATH_CALUDE_root_sum_squares_l900_90073

theorem root_sum_squares (a b c : ℝ) : 
  (a^3 - 20*a^2 + 18*a - 7 = 0) →
  (b^3 - 20*b^2 + 18*b - 7 = 0) →
  (c^3 - 20*c^2 + 18*c - 7 = 0) →
  (a+b)^2 + (b+c)^2 + (c+a)^2 = 764 := by
sorry

end NUMINAMATH_CALUDE_root_sum_squares_l900_90073


namespace NUMINAMATH_CALUDE_triangle_formation_l900_90065

def triangle_inequality (a b c : ℝ) : Prop :=
  a + b > c ∧ b + c > a ∧ c + a > b

def can_form_triangle (sides : List ℝ) : Prop :=
  sides.length = 3 ∧ 
  let a := sides[0]!
  let b := sides[1]!
  let c := sides[2]!
  triangle_inequality a b c

theorem triangle_formation :
  ¬(can_form_triangle [1, 2, 4]) ∧
  ¬(can_form_triangle [2, 3, 6]) ∧
  ¬(can_form_triangle [12, 5, 6]) ∧
  can_form_triangle [8, 6, 4] :=
by sorry

end NUMINAMATH_CALUDE_triangle_formation_l900_90065


namespace NUMINAMATH_CALUDE_dice_probability_l900_90081

/-- The number of sides on each die -/
def numSides : ℕ := 6

/-- The number of dice rolled -/
def numDice : ℕ := 5

/-- The total number of possible outcomes when rolling the dice -/
def totalOutcomes : ℕ := numSides ^ numDice

/-- The number of outcomes where all dice show the same number -/
def allSameOutcomes : ℕ := numSides

/-- The number of possible sequences (e.g., 1-2-3-4-5, 2-3-4-5-6) -/
def numSequences : ℕ := 2

/-- The number of ways to arrange each sequence -/
def sequenceArrangements : ℕ := Nat.factorial numDice

/-- The probability of rolling five fair 6-sided dice where they don't all show
    the same number and the numbers do not form a sequence -/
theorem dice_probability : 
  (totalOutcomes - allSameOutcomes - numSequences * sequenceArrangements) / totalOutcomes = 7530 / 7776 := by
  sorry

end NUMINAMATH_CALUDE_dice_probability_l900_90081


namespace NUMINAMATH_CALUDE_max_bananas_is_7_l900_90053

def budget : ℕ := 10
def single_banana_cost : ℕ := 2
def bundle_4_cost : ℕ := 6
def bundle_6_cost : ℕ := 8

def max_bananas (b s b4 b6 : ℕ) : ℕ := 
  sorry

theorem max_bananas_is_7 : 
  max_bananas budget single_banana_cost bundle_4_cost bundle_6_cost = 7 := by
  sorry

end NUMINAMATH_CALUDE_max_bananas_is_7_l900_90053


namespace NUMINAMATH_CALUDE_f_derivative_l900_90023

-- Define the function f
def f (x : ℝ) : ℝ := (2*x - 1) * (x^2 + 3)

-- State the theorem
theorem f_derivative :
  deriv f = fun x => 6*x^2 - 2*x + 6 :=
by sorry

end NUMINAMATH_CALUDE_f_derivative_l900_90023


namespace NUMINAMATH_CALUDE_farm_animal_difference_l900_90060

/-- Represents the number of horses and cows on a farm before and after a transaction --/
structure FarmAnimals where
  initial_horses : ℕ
  initial_cows : ℕ
  final_horses : ℕ
  final_cows : ℕ

/-- The conditions of the farm animal problem --/
def farm_conditions (farm : FarmAnimals) : Prop :=
  farm.initial_horses = 6 * farm.initial_cows ∧
  farm.final_horses = farm.initial_horses - 15 ∧
  farm.final_cows = farm.initial_cows + 15 ∧
  farm.final_horses = 3 * farm.final_cows

theorem farm_animal_difference (farm : FarmAnimals) 
  (h : farm_conditions farm) : farm.final_horses - farm.final_cows = 70 := by
  sorry

end NUMINAMATH_CALUDE_farm_animal_difference_l900_90060


namespace NUMINAMATH_CALUDE_gini_coefficient_change_l900_90087

/-- Represents a region in the country -/
structure Region where
  population : ℕ
  ppc : ℝ → ℝ
  maxKits : ℝ

/-- Calculates the Gini coefficient given two regions -/
def giniCoefficient (r1 r2 : Region) : ℝ :=
  sorry

/-- Calculates the new Gini coefficient after collaboration -/
def newGiniCoefficient (r1 r2 : Region) (compensation : ℝ) : ℝ :=
  sorry

theorem gini_coefficient_change
  (north : Region)
  (south : Region)
  (h1 : north.population = 24)
  (h2 : south.population = 6)
  (h3 : north.ppc = fun x => 13.5 - 9 * x)
  (h4 : south.ppc = fun x => 24 - 1.5 * x^2)
  (h5 : north.maxKits = 18)
  (h6 : south.maxKits = 12)
  (setPrice : ℝ)
  (h7 : setPrice = 6000)
  (compensation : ℝ)
  (h8 : compensation = 109983) :
  (giniCoefficient north south = 0.2) ∧
  (newGiniCoefficient north south compensation = 0.199) :=
sorry

end NUMINAMATH_CALUDE_gini_coefficient_change_l900_90087


namespace NUMINAMATH_CALUDE_sqrt_D_always_odd_l900_90035

theorem sqrt_D_always_odd (x : ℤ) : 
  let a : ℤ := x
  let b : ℤ := x + 1
  let c : ℤ := a * b
  let D : ℤ := a^2 + b^2 + c^2
  ∃ (k : ℤ), D = (2 * k + 1)^2 := by sorry

end NUMINAMATH_CALUDE_sqrt_D_always_odd_l900_90035


namespace NUMINAMATH_CALUDE_b_power_a_equals_negative_one_l900_90034

theorem b_power_a_equals_negative_one (a b : ℝ) : 
  (a - 5)^2 + |2*b + 2| = 0 → b^a = -1 := by sorry

end NUMINAMATH_CALUDE_b_power_a_equals_negative_one_l900_90034


namespace NUMINAMATH_CALUDE_max_value_abc_l900_90075

theorem max_value_abc (a b c : ℝ) (h1 : 0 ≤ a) (h2 : 0 ≤ b) (h3 : 0 ≤ c) (h4 : a + b + c = 1) :
  ∃ (max : ℝ), max = Real.sqrt 3 / 18 ∧ 
  ∀ (x y z : ℝ), 0 ≤ x ∧ 0 ≤ y ∧ 0 ≤ z ∧ x + y + z = 1 → 
  x^2 * (y - z) + y^2 * (z - x) + z^2 * (x - y) ≤ max :=
by sorry

end NUMINAMATH_CALUDE_max_value_abc_l900_90075


namespace NUMINAMATH_CALUDE_square_difference_equality_l900_90080

theorem square_difference_equality : 1013^2 - 987^2 - 1007^2 + 993^2 = 24000 := by
  sorry

end NUMINAMATH_CALUDE_square_difference_equality_l900_90080


namespace NUMINAMATH_CALUDE_hyperbola_equilateral_triangle_l900_90046

-- Define the hyperbola
def hyperbola (x y : ℝ) : Prop := x * y = 1

-- Define the two branches of the hyperbola
def branch1 (x y : ℝ) : Prop := hyperbola x y ∧ x > 0
def branch2 (x y : ℝ) : Prop := hyperbola x y ∧ x < 0

-- Define an equilateral triangle
def is_equilateral_triangle (P Q R : ℝ × ℝ) : Prop :=
  let (px, py) := P
  let (qx, qy) := Q
  let (rx, ry) := R
  (px - qx)^2 + (py - qy)^2 = (qx - rx)^2 + (qy - ry)^2 ∧
  (qx - rx)^2 + (qy - ry)^2 = (rx - px)^2 + (ry - py)^2

-- Main theorem
theorem hyperbola_equilateral_triangle :
  ∀ (Q R : ℝ × ℝ),
  hyperbola (-1) (-1) →
  branch2 (-1) (-1) →
  branch1 Q.1 Q.2 →
  branch1 R.1 R.2 →
  is_equilateral_triangle (-1, -1) Q R →
  (Q = (2 - Real.sqrt 3, 2 + Real.sqrt 3) ∧ R = (2 + Real.sqrt 3, 2 - Real.sqrt 3)) ∨
  (Q = (2 + Real.sqrt 3, 2 - Real.sqrt 3) ∧ R = (2 - Real.sqrt 3, 2 + Real.sqrt 3)) :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_equilateral_triangle_l900_90046


namespace NUMINAMATH_CALUDE_last_score_is_84_l900_90097

def scores : List ℕ := [68, 75, 78, 84, 85, 90]

def is_valid_last_score (s : ℕ) : Prop :=
  s ∈ scores ∧
  ∀ subset : List ℕ, subset.length < 6 → subset ⊆ scores →
  (subset.sum + s) % (subset.length + 1) = 0

theorem last_score_is_84 :
  ∀ s ∈ scores, is_valid_last_score s ↔ s = 84 := by sorry

end NUMINAMATH_CALUDE_last_score_is_84_l900_90097


namespace NUMINAMATH_CALUDE_roots_quadratic_equation_l900_90007

theorem roots_quadratic_equation (α β : ℝ) : 
  (α^2 + 2*α - 2021 = 0) → 
  (β^2 + 2*β - 2021 = 0) → 
  (α^2 + 3*α + β = 2019) := by
sorry

end NUMINAMATH_CALUDE_roots_quadratic_equation_l900_90007


namespace NUMINAMATH_CALUDE_syrup_volume_proof_l900_90006

/-- Calculates the final volume of syrup in cups -/
def final_syrup_volume (original_volume : ℚ) (reduction_factor : ℚ) (added_sugar : ℚ) (cups_per_quart : ℚ) : ℚ :=
  original_volume * cups_per_quart * reduction_factor + added_sugar

theorem syrup_volume_proof (original_volume : ℚ) (reduction_factor : ℚ) (added_sugar : ℚ) (cups_per_quart : ℚ)
  (h1 : original_volume = 6)
  (h2 : reduction_factor = 1 / 12)
  (h3 : added_sugar = 1)
  (h4 : cups_per_quart = 4) :
  final_syrup_volume original_volume reduction_factor added_sugar cups_per_quart = 3 := by
  sorry

#eval final_syrup_volume 6 (1/12) 1 4

end NUMINAMATH_CALUDE_syrup_volume_proof_l900_90006


namespace NUMINAMATH_CALUDE_pears_left_theorem_l900_90022

/-- The number of pears Keith and Mike are left with after Keith gives away some pears -/
def pears_left (keith_picked : ℕ) (mike_picked : ℕ) (keith_gave_away : ℕ) : ℕ :=
  (keith_picked - keith_gave_away) + mike_picked

/-- Theorem stating that Keith and Mike are left with 13 pears -/
theorem pears_left_theorem : pears_left 47 12 46 = 13 := by
  sorry

end NUMINAMATH_CALUDE_pears_left_theorem_l900_90022


namespace NUMINAMATH_CALUDE_percentage_relation_l900_90003

theorem percentage_relation (A B x : ℝ) (h1 : A > 0) (h2 : B > 0) (h3 : A = (x / 100) * B) : 
  x = 100 * (A / B) := by
sorry

end NUMINAMATH_CALUDE_percentage_relation_l900_90003


namespace NUMINAMATH_CALUDE_marathon_yards_remainder_l900_90047

/-- Represents the distance of a marathon in miles and yards -/
structure MarathonDistance where
  miles : ℕ
  yards : ℕ

/-- Represents a total distance in miles and yards -/
structure TotalDistance where
  miles : ℕ
  yards : ℕ

def marathon : MarathonDistance :=
  { miles := 26, yards := 385 }

def yards_per_mile : ℕ := 1760

def num_marathons : ℕ := 15

/-- Calculates the total distance for a given number of marathons -/
def total_distance (n : ℕ) (d : MarathonDistance) : TotalDistance :=
  { miles := n * d.miles,
    yards := n * d.yards }

/-- Converts excess yards to miles and updates the TotalDistance -/
def normalize_distance (d : TotalDistance) : TotalDistance :=
  { miles := d.miles + d.yards / yards_per_mile,
    yards := d.yards % yards_per_mile }

theorem marathon_yards_remainder :
  (normalize_distance (total_distance num_marathons marathon)).yards = 495 := by
  sorry

end NUMINAMATH_CALUDE_marathon_yards_remainder_l900_90047


namespace NUMINAMATH_CALUDE_iris_spending_l900_90028

/-- Calculates the total amount spent by Iris on clothes, including discount and tax --/
def total_spent (jacket_price : ℚ) (jacket_quantity : ℕ)
                (shorts_price : ℚ) (shorts_quantity : ℕ)
                (pants_price : ℚ) (pants_quantity : ℕ)
                (tops_price : ℚ) (tops_quantity : ℕ)
                (skirts_price : ℚ) (skirts_quantity : ℕ)
                (discount_rate : ℚ) (tax_rate : ℚ) : ℚ :=
  sorry

/-- The theorem stating that Iris spent $230.16 on clothes --/
theorem iris_spending : 
  total_spent 15 3 10 2 18 4 7 6 12 5 (10/100) (7/100) = 230.16 := by
  sorry

end NUMINAMATH_CALUDE_iris_spending_l900_90028


namespace NUMINAMATH_CALUDE_initial_concentration_proof_l900_90071

/-- Proves that the initial concentration of an acidic liquid is 40% given the problem conditions --/
theorem initial_concentration_proof (initial_volume : ℝ) (water_removed : ℝ) (final_concentration : ℝ) :
  initial_volume = 12 →
  water_removed = 4 →
  final_concentration = 60 →
  (initial_volume - water_removed) * final_concentration / 100 = initial_volume * 40 / 100 := by
  sorry

#check initial_concentration_proof

end NUMINAMATH_CALUDE_initial_concentration_proof_l900_90071


namespace NUMINAMATH_CALUDE_excellent_round_probability_l900_90085

/-- Represents the result of a single dart throw -/
inductive DartThrow
| Miss  : DartThrow  -- Didn't land in 8th ring or higher
| Hit   : DartThrow  -- Landed in 8th ring or higher

/-- Represents a round of 3 dart throws -/
def Round := (DartThrow × DartThrow × DartThrow)

/-- Determines if a round is excellent (at least 2 hits) -/
def is_excellent (r : Round) : Bool :=
  match r with
  | (DartThrow.Hit, DartThrow.Hit, _) => true
  | (DartThrow.Hit, _, DartThrow.Hit) => true
  | (_, DartThrow.Hit, DartThrow.Hit) => true
  | _ => false

/-- The total number of rounds in the experiment -/
def total_rounds : Nat := 20

/-- The number of excellent rounds observed -/
def excellent_rounds : Nat := 12

/-- Theorem: The probability of an excellent round is 0.6 -/
theorem excellent_round_probability :
  (excellent_rounds : ℚ) / total_rounds = 0.6 := by sorry

end NUMINAMATH_CALUDE_excellent_round_probability_l900_90085


namespace NUMINAMATH_CALUDE_sqrt_nine_factorial_over_108_l900_90093

theorem sqrt_nine_factorial_over_108 : 
  Real.sqrt (Nat.factorial 9 / 108) = 8 * Real.sqrt 35 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_nine_factorial_over_108_l900_90093


namespace NUMINAMATH_CALUDE_fib_10_calls_l900_90038

def FIB : ℕ → ℕ
  | 0 => 1
  | 1 => 1
  | n+2 => FIB (n+1) + FIB n

def count_calls : ℕ → ℕ
  | 0 => 0
  | 1 => 0
  | n+2 => count_calls (n+1) + count_calls n + 2

theorem fib_10_calls : count_calls 10 = 176 := by
  sorry

end NUMINAMATH_CALUDE_fib_10_calls_l900_90038


namespace NUMINAMATH_CALUDE_f_max_property_l900_90010

def f_properties (f : ℚ → ℚ) : Prop :=
  (f 0 = 0) ∧
  (∀ α : ℚ, α ≠ 0 → f α > 0) ∧
  (∀ α β : ℚ, f (α * β) = f α * f β) ∧
  (∀ α β : ℚ, f (α + β) ≤ f α + f β) ∧
  (∀ m : ℤ, f m ≤ 1989)

theorem f_max_property (f : ℚ → ℚ) (h : f_properties f) :
  ∀ α β : ℚ, f α ≠ f β → f (α + β) = max (f α) (f β) := by
  sorry

end NUMINAMATH_CALUDE_f_max_property_l900_90010


namespace NUMINAMATH_CALUDE_light_bulbs_remaining_l900_90037

theorem light_bulbs_remaining (initial : Nat) (used : Nat) : 
  initial = 40 → used = 16 → (initial - used) / 2 = 12 :=
by
  sorry

end NUMINAMATH_CALUDE_light_bulbs_remaining_l900_90037


namespace NUMINAMATH_CALUDE_normal_distribution_estimate_l900_90074

/-- Represents the normal distribution function -/
noncomputable def normal_dist (μ σ : ℝ) (x : ℝ) : ℝ := sorry

/-- Represents the cumulative distribution function of a normal distribution -/
noncomputable def normal_cdf (μ σ : ℝ) (x : ℝ) : ℝ := sorry

/-- The probability that a value from N(μ, σ²) falls within μ ± σ -/
axiom normal_prob_within_1sigma (μ σ : ℝ) : 
  normal_cdf μ σ (μ + σ) - normal_cdf μ σ (μ - σ) = 0.6826

/-- The probability that a value from N(μ, σ²) falls within μ ± 2σ -/
axiom normal_prob_within_2sigma (μ σ : ℝ) : 
  normal_cdf μ σ (μ + 2*σ) - normal_cdf μ σ (μ - 2*σ) = 0.9544

/-- The theorem to be proved -/
theorem normal_distribution_estimate : 
  let μ : ℝ := 70
  let σ : ℝ := 5
  let sample_size : ℕ := 100000
  let lower_bound : ℝ := 75
  let upper_bound : ℝ := 80
  ⌊(normal_cdf μ σ upper_bound - normal_cdf μ σ lower_bound) * sample_size⌋ = 13590 := by sorry

end NUMINAMATH_CALUDE_normal_distribution_estimate_l900_90074


namespace NUMINAMATH_CALUDE_profit_percentage_previous_year_l900_90027

/-- Prove that the profit percentage in the previous year was 10% -/
theorem profit_percentage_previous_year
  (R : ℝ) -- Revenue in the previous year
  (h1 : R > 0) -- Assume positive revenue
  (h2 : 0.8 * R = revenue_2009) -- Revenue in 2009 was 80% of previous year
  (h3 : 0.13 * revenue_2009 = profit_2009) -- Profit in 2009 was 13% of 2009 revenue
  (h4 : profit_2009 = 1.04 * profit_previous) -- Profit in 2009 was 104% of previous year's profit
  (h5 : profit_previous = P / 100 * R) -- Definition of profit percentage
  : P = 10 := by sorry

end NUMINAMATH_CALUDE_profit_percentage_previous_year_l900_90027


namespace NUMINAMATH_CALUDE_texas_passengers_on_l900_90064

/-- Represents the number of passengers at different stages of the flight --/
structure PassengerCount where
  initial : ℕ
  texas_off : ℕ
  texas_on : ℕ
  nc_off : ℕ
  nc_on : ℕ
  crew : ℕ
  final : ℕ

/-- Theorem stating that given the flight conditions, 24 passengers got on in Texas --/
theorem texas_passengers_on (p : PassengerCount) 
  (h1 : p.initial = 124)
  (h2 : p.texas_off = 58)
  (h3 : p.nc_off = 47)
  (h4 : p.nc_on = 14)
  (h5 : p.crew = 10)
  (h6 : p.final = 67)
  (h7 : p.final = p.initial - p.texas_off + p.texas_on - p.nc_off + p.nc_on + p.crew) :
  p.texas_on = 24 := by
  sorry

end NUMINAMATH_CALUDE_texas_passengers_on_l900_90064


namespace NUMINAMATH_CALUDE_simplify_sqrt_seven_simplify_sqrt_fraction_simplify_sqrt_sum_simplify_sqrt_expression_l900_90086

-- Problem 1
theorem simplify_sqrt_seven : 2 * Real.sqrt 7 - 6 * Real.sqrt 7 = -4 * Real.sqrt 7 := by sorry

-- Problem 2
theorem simplify_sqrt_fraction : Real.sqrt (2/3) / Real.sqrt (8/27) = 3/2 := by sorry

-- Problem 3
theorem simplify_sqrt_sum : Real.sqrt 18 + Real.sqrt 98 - Real.sqrt 27 = 10 * Real.sqrt 2 - 3 * Real.sqrt 3 := by sorry

-- Problem 4
theorem simplify_sqrt_expression : 
  (Real.sqrt 0.5 + Real.sqrt 6) - (Real.sqrt (1/8) - Real.sqrt 24) = Real.sqrt 2 / 4 + 3 * Real.sqrt 6 := by sorry

end NUMINAMATH_CALUDE_simplify_sqrt_seven_simplify_sqrt_fraction_simplify_sqrt_sum_simplify_sqrt_expression_l900_90086


namespace NUMINAMATH_CALUDE_binomial_coefficient_ratio_l900_90070

theorem binomial_coefficient_ratio (n k : ℕ) : 
  (n.choose k : ℚ) / (n.choose (k + 1) : ℚ) = 3 / 4 ∧ 
  (n.choose (k + 1) : ℚ) / (n.choose (k + 2) : ℚ) = 4 / 5 → 
  n + k = 55 := by sorry

end NUMINAMATH_CALUDE_binomial_coefficient_ratio_l900_90070


namespace NUMINAMATH_CALUDE_square_sum_value_l900_90090

theorem square_sum_value (x y : ℝ) (h1 : x + y = 10) (h2 : x * y = 15) : x^2 + y^2 = 70 := by
  sorry

end NUMINAMATH_CALUDE_square_sum_value_l900_90090


namespace NUMINAMATH_CALUDE_marble_weight_difference_is_8_l900_90096

/-- Calculates the difference in weight between red and yellow marbles -/
def marble_weight_difference (total_marbles : ℕ) (yellow_marbles : ℕ) (blue_red_ratio : ℚ) 
  (yellow_weight : ℝ) (red_yellow_weight_ratio : ℝ) : ℝ :=
  let remaining_marbles := total_marbles - yellow_marbles
  let red_marbles := (remaining_marbles : ℝ) * (1 / (1 + blue_red_ratio)) * (blue_red_ratio / (1 + blue_red_ratio))⁻¹
  let red_weight := yellow_weight * red_yellow_weight_ratio
  red_weight - yellow_weight

theorem marble_weight_difference_is_8 :
  marble_weight_difference 19 5 (3/4) 8 2 = 8 := by
  sorry

end NUMINAMATH_CALUDE_marble_weight_difference_is_8_l900_90096


namespace NUMINAMATH_CALUDE_x_in_terms_of_y_l900_90092

theorem x_in_terms_of_y (x y : ℝ) (h : x / (x - 3) = (y^2 + 3*y + 1) / (y^2 + 3*y - 4)) :
  x = (3*y^2 + 9*y + 3) / 5 := by
sorry

end NUMINAMATH_CALUDE_x_in_terms_of_y_l900_90092


namespace NUMINAMATH_CALUDE_S_equals_T_l900_90020

def S : Set ℤ := {x | ∃ n : ℕ, x = 3 * n + 1}
def T : Set ℤ := {x | ∃ k : ℤ, x = 3 * k - 2}

theorem S_equals_T : S = T := by sorry

end NUMINAMATH_CALUDE_S_equals_T_l900_90020


namespace NUMINAMATH_CALUDE_milk_water_ratio_after_addition_l900_90067

/-- Calculates the final ratio of milk to water after adding water to a mixture -/
theorem milk_water_ratio_after_addition
  (initial_volume : ℚ)
  (initial_milk_ratio : ℚ)
  (initial_water_ratio : ℚ)
  (added_water : ℚ)
  (h1 : initial_volume = 45)
  (h2 : initial_milk_ratio = 4)
  (h3 : initial_water_ratio = 1)
  (h4 : added_water = 21) :
  let initial_milk := initial_volume * initial_milk_ratio / (initial_milk_ratio + initial_water_ratio)
  let initial_water := initial_volume * initial_water_ratio / (initial_milk_ratio + initial_water_ratio)
  let final_water := initial_water + added_water
  let final_milk_ratio := initial_milk / final_water
  let final_water_ratio := final_water / final_water
  (final_milk_ratio : ℚ) / (final_water_ratio : ℚ) = 6 / 5 := by
  sorry

end NUMINAMATH_CALUDE_milk_water_ratio_after_addition_l900_90067


namespace NUMINAMATH_CALUDE_intercepted_line_equation_l900_90056

/-- A line passing through a point and intercepted by two parallel lines -/
structure InterceptedLine where
  -- The slope of the line
  k : ℝ
  -- The point that the line passes through
  p : ℝ × ℝ
  -- The two parallel lines represented by their y-intercepts
  l1 : ℝ
  l2 : ℝ
  -- Conditions
  passes_through : p.2 = k * p.1 + 1 - 2 * k
  parallel_lines : l1 = -4/3 * p.1 - 1/3 ∧ l2 = -4/3 * p.1 - 2
  segment_length : (((6*k - 4)/(4 + 3*k) - (6*k - 9)/(4 + 3*k))^2 + 
                    ((4 - 9*k)/(4 + 3*k) - (4 - 14*k)/(4 + 3*k))^2) = 2

/-- Theorem: The equation of the intercepted line is either x + 7y - 9 = 0 or 7x - y - 13 = 0 -/
theorem intercepted_line_equation (l : InterceptedLine) 
  (h : l.p = (2, 1)) : 
  (l.k = -1/7 ∧ l.p.2 = -1/7 * l.p.1 + 1 + 2/7) ∨
  (l.k = 7 ∧ l.p.2 = 7 * l.p.1 + 1 - 14) :=
sorry

end NUMINAMATH_CALUDE_intercepted_line_equation_l900_90056


namespace NUMINAMATH_CALUDE_outfits_count_l900_90089

/-- The number of shirts available -/
def num_shirts : ℕ := 8

/-- The number of ties available -/
def num_ties : ℕ := 7

/-- Calculate the number of outfits possible -/
def num_outfits : ℕ := num_shirts * (num_ties + 1)

/-- Theorem stating that the number of outfits is 64 -/
theorem outfits_count : num_outfits = 64 := by
  sorry

end NUMINAMATH_CALUDE_outfits_count_l900_90089


namespace NUMINAMATH_CALUDE_factorial_prime_factorization_l900_90082

def factorial (n : ℕ) : ℕ := (List.range n).foldl (· * ·) 1

theorem factorial_prime_factorization :
  ∃ (i k m p q : ℕ+),
    factorial 12 = 2^(i.val) * 3^(k.val) * 5^(m.val) * 7^(p.val) * 11^(q.val) ∧
    i.val + k.val + m.val + p.val + q.val = 28 := by
  sorry

end NUMINAMATH_CALUDE_factorial_prime_factorization_l900_90082


namespace NUMINAMATH_CALUDE_apple_lemon_ratio_l900_90079

theorem apple_lemon_ratio (apples oranges lemons : ℕ) 
  (h1 : apples * 4 = oranges * 1) 
  (h2 : oranges * 2 = lemons * 5) : 
  apples * 8 = lemons * 5 := by
sorry

end NUMINAMATH_CALUDE_apple_lemon_ratio_l900_90079


namespace NUMINAMATH_CALUDE_circle_line_intersection_chord_length_l900_90001

/-- A circle in the xy-plane -/
structure Circle where
  equation : ℝ → ℝ → ℝ → Prop

/-- A line in the xy-plane -/
structure Line where
  equation : ℝ → ℝ → Prop

/-- The length of a chord formed by the intersection of a circle and a line -/
def chordLength (c : Circle) (l : Line) : ℝ := sorry

theorem circle_line_intersection_chord_length (a : ℝ) :
  let c : Circle := { equation := fun x y z => x^2 + y^2 + 2*x - 2*y + z = 0 }
  let l : Line := { equation := fun x y => x + y + 2 = 0 }
  chordLength c l = 4 → a = -4 := by sorry

end NUMINAMATH_CALUDE_circle_line_intersection_chord_length_l900_90001


namespace NUMINAMATH_CALUDE_freshmen_liberal_arts_percentage_l900_90048

theorem freshmen_liberal_arts_percentage 
  (total_students : ℝ) 
  (freshmen_percentage : ℝ) 
  (liberal_arts_freshmen_percentage : ℝ) 
  (psychology_majors_percentage : ℝ) 
  (freshmen_psychology_liberal_arts_percentage : ℝ) 
  (h1 : freshmen_percentage = 0.5)
  (h2 : psychology_majors_percentage = 0.2)
  (h3 : freshmen_psychology_liberal_arts_percentage = 0.04)
  (h4 : freshmen_psychology_liberal_arts_percentage * total_students = 
        psychology_majors_percentage * liberal_arts_freshmen_percentage * freshmen_percentage * total_students) :
  liberal_arts_freshmen_percentage = 0.4 := by
sorry

end NUMINAMATH_CALUDE_freshmen_liberal_arts_percentage_l900_90048


namespace NUMINAMATH_CALUDE_exercise_minutes_proof_l900_90062

def javier_minutes : ℕ := 50
def javier_days : ℕ := 10

def sanda_minutes_1 : ℕ := 90
def sanda_days_1 : ℕ := 3
def sanda_minutes_2 : ℕ := 75
def sanda_days_2 : ℕ := 2
def sanda_minutes_3 : ℕ := 45
def sanda_days_3 : ℕ := 4

def total_exercise_minutes : ℕ := 1100

theorem exercise_minutes_proof :
  (javier_minutes * javier_days) +
  (sanda_minutes_1 * sanda_days_1) +
  (sanda_minutes_2 * sanda_days_2) +
  (sanda_minutes_3 * sanda_days_3) = total_exercise_minutes :=
by sorry

end NUMINAMATH_CALUDE_exercise_minutes_proof_l900_90062


namespace NUMINAMATH_CALUDE_opposite_face_is_blue_l900_90078

/-- Represents the colors of the squares --/
inductive Color
  | R | B | O | Y | G | W

/-- Represents a square with colors on both sides --/
structure Square where
  front : Color
  back : Color

/-- Represents the cube formed by folding the squares --/
structure Cube where
  squares : List Square
  white_face : Color
  opposite_face : Color

/-- Axiom: The cube is formed by folding six hinged squares --/
axiom cube_formation (c : Cube) : c.squares.length = 6

/-- Axiom: The white face exists in the cube --/
axiom white_face_exists (c : Cube) : c.white_face = Color.W

/-- Theorem: The face opposite to the white face is blue --/
theorem opposite_face_is_blue (c : Cube) : c.opposite_face = Color.B := by
  sorry

end NUMINAMATH_CALUDE_opposite_face_is_blue_l900_90078


namespace NUMINAMATH_CALUDE_max_children_count_max_children_is_26_l900_90031

def initial_apples : ℕ := 55
def initial_cookies : ℕ := 114
def initial_chocolates : ℕ := 83

def remaining_apples : ℕ := 3
def remaining_cookies : ℕ := 10
def remaining_chocolates : ℕ := 5

def distributed_apples : ℕ := initial_apples - remaining_apples
def distributed_cookies : ℕ := initial_cookies - remaining_cookies
def distributed_chocolates : ℕ := initial_chocolates - remaining_chocolates

theorem max_children_count : ℕ → Prop :=
  fun n =>
    n > 0 ∧
    distributed_apples % n = 0 ∧
    distributed_cookies % n = 0 ∧
    distributed_chocolates % n = 0 ∧
    ∀ m : ℕ, m > n →
      (distributed_apples % m ≠ 0 ∨
       distributed_cookies % m ≠ 0 ∨
       distributed_chocolates % m ≠ 0)

theorem max_children_is_26 : max_children_count 26 := by sorry

end NUMINAMATH_CALUDE_max_children_count_max_children_is_26_l900_90031


namespace NUMINAMATH_CALUDE_plastic_rings_weight_l900_90032

/-- The weight of the orange ring in ounces -/
def orange_weight : ℝ := 0.08333333333333333

/-- The weight of the purple ring in ounces -/
def purple_weight : ℝ := 0.3333333333333333

/-- The weight of the white ring in ounces -/
def white_weight : ℝ := 0.4166666666666667

/-- The total weight of all rings in ounces -/
def total_weight : ℝ := orange_weight + purple_weight + white_weight

theorem plastic_rings_weight : total_weight = 0.8333333333333333 := by
  sorry

end NUMINAMATH_CALUDE_plastic_rings_weight_l900_90032


namespace NUMINAMATH_CALUDE_multiple_properties_l900_90016

theorem multiple_properties (a b : ℤ) 
  (h1 : ∃ k : ℤ, a = 5 * k)
  (h2 : ∃ m : ℤ, a = 2 * m + 1)
  (h3 : ∃ n : ℤ, b = 10 * n) :
  (∃ p : ℤ, b = 5 * p) ∧ (∃ q : ℤ, a - b = 5 * q) := by
sorry

end NUMINAMATH_CALUDE_multiple_properties_l900_90016


namespace NUMINAMATH_CALUDE_divisibility_criterion_l900_90000

theorem divisibility_criterion (p : ℕ) (hp : Nat.Prime p) :
  (∀ x y : ℕ, x > 0 → y > 0 → p ∣ (x + y)^19 - x^19 - y^19) ↔ p = 2 ∨ p = 3 ∨ p = 7 ∨ p = 19 :=
sorry

end NUMINAMATH_CALUDE_divisibility_criterion_l900_90000


namespace NUMINAMATH_CALUDE_inequality_system_solution_set_l900_90026

theorem inequality_system_solution_set : 
  {x : ℝ | (5 - 2*x ≤ 1) ∧ (x - 4 < 0)} = {x : ℝ | 2 ≤ x ∧ x < 4} := by
  sorry

end NUMINAMATH_CALUDE_inequality_system_solution_set_l900_90026


namespace NUMINAMATH_CALUDE_yolanda_scoring_l900_90013

/-- Yolanda's basketball scoring problem -/
theorem yolanda_scoring (total_points : ℕ) (num_games : ℕ) (avg_free_throws : ℕ) (avg_two_pointers : ℕ) 
  (h1 : total_points = 345)
  (h2 : num_games = 15)
  (h3 : avg_free_throws = 4)
  (h4 : avg_two_pointers = 5) :
  (total_points / num_games - (avg_free_throws * 1 + avg_two_pointers * 2)) / 3 = 3 := by
  sorry

end NUMINAMATH_CALUDE_yolanda_scoring_l900_90013


namespace NUMINAMATH_CALUDE_sequence_general_term_l900_90017

theorem sequence_general_term (n : ℕ) (S : ℕ → ℤ) (a : ℕ → ℤ) 
  (h : ∀ k, S k = 2 * k^2 - 3 * k) : 
  a n = 4 * n - 5 := by
  sorry

end NUMINAMATH_CALUDE_sequence_general_term_l900_90017


namespace NUMINAMATH_CALUDE_partnership_gain_is_18000_l900_90050

/-- Represents the annual gain of a partnership given the investments and one partner's share. -/
def partnership_annual_gain (x : ℚ) (a_share : ℚ) : ℚ :=
  let a_invest_time : ℚ := x * 12
  let b_invest_time : ℚ := 2 * x * 6
  let c_invest_time : ℚ := 3 * x * 4
  let total_invest_time : ℚ := a_invest_time + b_invest_time + c_invest_time
  (total_invest_time / a_invest_time) * a_share

/-- 
Given:
- A invests x at the beginning
- B invests 2x after 6 months
- C invests 3x after 8 months
- A's share of the gain is 6000

Prove that the total annual gain of the partnership is 18000.
-/
theorem partnership_gain_is_18000 (x : ℚ) (h : x > 0) :
  partnership_annual_gain x 6000 = 18000 := by
  sorry

end NUMINAMATH_CALUDE_partnership_gain_is_18000_l900_90050


namespace NUMINAMATH_CALUDE_pen_price_calculation_l900_90021

theorem pen_price_calculation (num_pens : ℕ) (num_pencils : ℕ) (total_cost : ℚ) (pencil_price : ℚ) :
  num_pens = 30 →
  num_pencils = 75 →
  total_cost = 690 →
  pencil_price = 2 →
  (total_cost - num_pencils * pencil_price) / num_pens = 18 := by
sorry

end NUMINAMATH_CALUDE_pen_price_calculation_l900_90021


namespace NUMINAMATH_CALUDE_sin_equality_n_512_l900_90055

theorem sin_equality_n_512 (n : ℤ) :
  -100 ≤ n ∧ n ≤ 100 ∧ Real.sin (n * π / 180) = Real.sin (512 * π / 180) → n = 28 :=
by sorry

end NUMINAMATH_CALUDE_sin_equality_n_512_l900_90055


namespace NUMINAMATH_CALUDE_circles_cover_quadrilateral_l900_90002

-- Define a convex quadrilateral
def ConvexQuadrilateral (A B C D : Real × Real) : Prop :=
  -- Add conditions for convexity
  sorry

-- Define a circle with diameter as a side of the quadrilateral
def CircleOnSide (A B : Real × Real) : Set (Real × Real) :=
  {P | ∃ (t : Real), 0 ≤ t ∧ t ≤ 1 ∧ P = (1 - t) • A + t • B ∧ 
    (P.1 - A.1)^2 + (P.2 - A.2)^2 ≤ ((B.1 - A.1)^2 + (B.2 - A.2)^2) / 4}

-- Define the union of four circles on the sides of the quadrilateral
def UnionOfCircles (A B C D : Real × Real) : Set (Real × Real) :=
  CircleOnSide A B ∪ CircleOnSide B C ∪ CircleOnSide C D ∪ CircleOnSide D A

-- Define the interior of the quadrilateral
def QuadrilateralInterior (A B C D : Real × Real) : Set (Real × Real) :=
  -- Add definition for the interior of the quadrilateral
  sorry

-- Theorem statement
theorem circles_cover_quadrilateral (A B C D : Real × Real) :
  ConvexQuadrilateral A B C D →
  QuadrilateralInterior A B C D ⊆ UnionOfCircles A B C D :=
sorry

end NUMINAMATH_CALUDE_circles_cover_quadrilateral_l900_90002


namespace NUMINAMATH_CALUDE_solve_equation_l900_90058

theorem solve_equation (x : ℝ) : 
  Real.sqrt ((3 / x) + 3) = 5 / 3 → x = -27 / 2 := by
sorry

end NUMINAMATH_CALUDE_solve_equation_l900_90058


namespace NUMINAMATH_CALUDE_overall_profit_calculation_l900_90099

/-- Calculates the overall profit from selling a refrigerator and a mobile phone -/
theorem overall_profit_calculation (refrigerator_cost mobile_cost : ℕ)
  (refrigerator_loss_percent mobile_profit_percent : ℚ)
  (h1 : refrigerator_cost = 15000)
  (h2 : mobile_cost = 8000)
  (h3 : refrigerator_loss_percent = 4 / 100)
  (h4 : mobile_profit_percent = 10 / 100) :
  (refrigerator_cost * (1 - refrigerator_loss_percent) +
   mobile_cost * (1 + mobile_profit_percent) -
   (refrigerator_cost + mobile_cost)).floor = 200 :=
by sorry

end NUMINAMATH_CALUDE_overall_profit_calculation_l900_90099


namespace NUMINAMATH_CALUDE_mias_gift_spending_l900_90005

theorem mias_gift_spending (total_spending : ℕ) (num_siblings : ℕ) (parent_gift : ℕ) (num_parents : ℕ) 
  (h1 : total_spending = 150)
  (h2 : num_siblings = 3)
  (h3 : parent_gift = 30)
  (h4 : num_parents = 2) :
  (total_spending - num_parents * parent_gift) / num_siblings = 30 := by
  sorry

end NUMINAMATH_CALUDE_mias_gift_spending_l900_90005


namespace NUMINAMATH_CALUDE_least_integer_square_36_more_than_triple_l900_90044

theorem least_integer_square_36_more_than_triple (x : ℤ) :
  (x^2 = 3*x + 36) → (x ≥ -6) :=
by sorry

end NUMINAMATH_CALUDE_least_integer_square_36_more_than_triple_l900_90044


namespace NUMINAMATH_CALUDE_sum_of_powers_l900_90076

theorem sum_of_powers (x y : ℝ) (h1 : (x + y)^2 = 7) (h2 : (x - y)^2 = 3) :
  (x^2 + y^2 = 5) ∧ (x^4 + y^4 = 23) ∧ (x^6 + y^6 = 110) := by
  sorry

end NUMINAMATH_CALUDE_sum_of_powers_l900_90076


namespace NUMINAMATH_CALUDE_product_trailing_zeros_l900_90012

/-- The number of trailing zeros in a natural number -/
def trailingZeros (n : ℕ) : ℕ := sorry

/-- The product of 45 and 320 -/
def product : ℕ := 45 * 320

/-- Theorem: The number of trailing zeros in the product 45 × 320 is 2 -/
theorem product_trailing_zeros : trailingZeros product = 2 := by sorry

end NUMINAMATH_CALUDE_product_trailing_zeros_l900_90012


namespace NUMINAMATH_CALUDE_viewers_scientific_notation_equality_l900_90029

-- Define the number of viewers
def viewers : ℕ := 16300000000

-- Define the scientific notation representation
def scientific_notation : ℝ := 1.63 * (10 ^ 10)

-- Theorem to prove the equality
theorem viewers_scientific_notation_equality :
  (viewers : ℝ) = scientific_notation := by sorry

end NUMINAMATH_CALUDE_viewers_scientific_notation_equality_l900_90029


namespace NUMINAMATH_CALUDE_sams_calculation_l900_90091

theorem sams_calculation (x y : ℝ) : 
  x + 2 * 2 + y = x * 2 + 2 + y → x + y = 4 := by
  sorry

end NUMINAMATH_CALUDE_sams_calculation_l900_90091


namespace NUMINAMATH_CALUDE_point_on_translated_line_l900_90094

/-- The original line -/
def original_line (x : ℝ) : ℝ := x

/-- The translated line -/
def translated_line (x : ℝ) : ℝ := x + 2

/-- Theorem stating that (2, 4) lies on the translated line -/
theorem point_on_translated_line : translated_line 2 = 4 := by sorry

end NUMINAMATH_CALUDE_point_on_translated_line_l900_90094


namespace NUMINAMATH_CALUDE_intersection_line_canonical_form_l900_90088

/-- Given two planes in 3D space, prove that their intersection forms a line with specific canonical equations. -/
theorem intersection_line_canonical_form (x y z : ℝ) :
  (2 * x - 3 * y - 2 * z + 6 = 0) →
  (x - 3 * y + z + 3 = 0) →
  ∃ (t : ℝ), x = 9 * t - 3 ∧ y = 4 * t ∧ z = 3 * t :=
by sorry

end NUMINAMATH_CALUDE_intersection_line_canonical_form_l900_90088


namespace NUMINAMATH_CALUDE_largest_t_value_l900_90072

theorem largest_t_value (t : ℚ) : 
  (13 * t^2 - 34 * t + 12) / (3 * t - 2) + 5 * t = 6 * t - 1 →
  t ≤ 5/2 :=
by sorry

end NUMINAMATH_CALUDE_largest_t_value_l900_90072


namespace NUMINAMATH_CALUDE_probability_4H_before_3T_is_4_57_l900_90025

/-- The probability of encountering 4 heads before 3 consecutive tails in fair coin flips -/
def probability_4H_before_3T : ℚ :=
  4 / 57

/-- Theorem stating that the probability of encountering 4 heads before 3 consecutive tails
    in fair coin flips is equal to 4/57 -/
theorem probability_4H_before_3T_is_4_57 :
  probability_4H_before_3T = 4 / 57 := by
  sorry

end NUMINAMATH_CALUDE_probability_4H_before_3T_is_4_57_l900_90025


namespace NUMINAMATH_CALUDE_eccentricity_range_l900_90052

/-- Represents an ellipse with semi-major axis a and semi-minor axis b -/
structure Ellipse where
  a : ℝ
  b : ℝ
  h_pos : 0 < b ∧ b < a

/-- Represents a point on the coordinate plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- Defines the property of a point P being inside the ellipse -/
def inside_ellipse (P : Point) (E : Ellipse) : Prop :=
  (P.x^2 / E.a^2) + (P.y^2 / E.b^2) < 1

/-- Defines the condition that the dot product of vectors PF₁ and PF₂ is zero -/
def orthogonal_foci (P : Point) (E : Ellipse) : Prop :=
  ∃ (F₁ F₂ : Point), (P.x - F₁.x) * (P.x - F₂.x) + (P.y - F₁.y) * (P.y - F₂.y) = 0

/-- Theorem stating the range of eccentricity for the ellipse -/
theorem eccentricity_range (E : Ellipse) 
  (h : ∀ P : Point, orthogonal_foci P E → inside_ellipse P E) :
  ∃ e : ℝ, 0 < e ∧ e < Real.sqrt 2 / 2 ∧ e^2 = (E.a^2 - E.b^2) / E.a^2 := by
  sorry

end NUMINAMATH_CALUDE_eccentricity_range_l900_90052


namespace NUMINAMATH_CALUDE_min_sum_factors_2400_l900_90019

/-- The minimum sum of two positive integer factors of 2400 -/
theorem min_sum_factors_2400 : ∀ a b : ℕ+, a * b = 2400 → (∀ c d : ℕ+, c * d = 2400 → a + b ≤ c + d) → a + b = 98 := by
  sorry

end NUMINAMATH_CALUDE_min_sum_factors_2400_l900_90019


namespace NUMINAMATH_CALUDE_product_a4b4_equals_negative_six_l900_90041

theorem product_a4b4_equals_negative_six
  (a₁ a₂ a₃ a₄ b₁ b₂ b₃ b₄ : ℝ)
  (eq1 : a₁ * b₁ + a₂ * b₃ = 1)
  (eq2 : a₁ * b₂ + a₂ * b₄ = 0)
  (eq3 : a₃ * b₁ + a₄ * b₃ = 0)
  (eq4 : a₃ * b₂ + a₄ * b₄ = 1)
  (eq5 : a₂ * b₃ = 7) :
  a₄ * b₄ = -6 := by
  sorry

end NUMINAMATH_CALUDE_product_a4b4_equals_negative_six_l900_90041


namespace NUMINAMATH_CALUDE_factory_production_constraints_l900_90059

/-- Given a factory producing two products A and B, this theorem states the constraint
conditions for maximizing the total monthly profit. -/
theorem factory_production_constraints
  (a₁ a₂ b₁ b₂ c₁ c₂ d₁ d₂ : ℝ)
  (x y : ℝ) -- Monthly production of products A and B in kg
  (h_pos_a₁ : a₁ > 0) (h_pos_a₂ : a₂ > 0)
  (h_pos_b₁ : b₁ > 0) (h_pos_b₂ : b₂ > 0)
  (h_pos_c₁ : c₁ > 0) (h_pos_c₂ : c₂ > 0)
  (h_pos_d₁ : d₁ > 0) (h_pos_d₂ : d₂ > 0) :
  (∃ z : ℝ, z = d₁ * x + d₂ * y ∧ -- Total monthly profit
    a₁ * x + a₂ * y ≤ c₁ ∧       -- Constraint on raw material A
    b₁ * x + b₂ * y ≤ c₂ ∧       -- Constraint on raw material B
    x ≥ 0 ∧ y ≥ 0) →             -- Non-negative production constraints
  (a₁ * x + a₂ * y ≤ c₁ ∧
   b₁ * x + b₂ * y ≤ c₂ ∧
   x ≥ 0 ∧ y ≥ 0) :=
by sorry

end NUMINAMATH_CALUDE_factory_production_constraints_l900_90059


namespace NUMINAMATH_CALUDE_evaluate_expression_l900_90004

theorem evaluate_expression (a b : ℤ) (h1 : a = 5) (h2 : b = 3) :
  (a^2 + b)^2 - (a^2 - b)^2 = 300 := by
  sorry

end NUMINAMATH_CALUDE_evaluate_expression_l900_90004


namespace NUMINAMATH_CALUDE_fraction_equality_l900_90084

theorem fraction_equality (a b : ℝ) (h : (4*a + 3*b) / (4*a - 3*b) = 4) : a / b = 5/4 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_l900_90084


namespace NUMINAMATH_CALUDE_function_properties_l900_90095

noncomputable def f (a b x : ℝ) : ℝ := 6 * Real.log x - a * x^2 - 7 * x + b

theorem function_properties (a b : ℝ) :
  (∀ x, x > 0 → HasDerivAt (f a b) ((6 / x) - 2 * a * x - 7) x) →
  HasDerivAt (f a b) 0 2 →
  (a = -1 ∧
   (∀ x, 0 < x ∧ x < 3/2 → HasDerivAt (f a b) ((x - 2) * (2 * x - 3) / x) x ∧ ((x - 2) * (2 * x - 3) / x > 0)) ∧
   (∀ x, 3/2 < x ∧ x < 2 → HasDerivAt (f a b) ((x - 2) * (2 * x - 3) / x) x ∧ ((x - 2) * (2 * x - 3) / x < 0)) ∧
   (∀ x, x > 2 → HasDerivAt (f a b) ((x - 2) * (2 * x - 3) / x) x ∧ ((x - 2) * (2 * x - 3) / x > 0)) ∧
   (33/4 - 6 * Real.log (3/2) < b ∧ b < 10 - 6 * Real.log 2)) :=
by sorry

end NUMINAMATH_CALUDE_function_properties_l900_90095


namespace NUMINAMATH_CALUDE_sean_houses_bought_l900_90024

theorem sean_houses_bought (initial_houses : ℕ) (traded_houses : ℕ) (final_houses : ℕ) 
  (h1 : initial_houses = 27)
  (h2 : traded_houses = 8)
  (h3 : final_houses = 31) :
  final_houses - (initial_houses - traded_houses) = 12 :=
by
  sorry

end NUMINAMATH_CALUDE_sean_houses_bought_l900_90024


namespace NUMINAMATH_CALUDE_rectangle_dimension_change_l900_90063

theorem rectangle_dimension_change (L B : ℝ) (p : ℝ) 
  (h1 : L > 0) (h2 : B > 0) (h3 : p > 0) :
  (L * (1 + p)) * (B * 0.75) = L * B * 1.05 → p = 0.4 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_dimension_change_l900_90063


namespace NUMINAMATH_CALUDE_mango_profit_percentage_l900_90043

theorem mango_profit_percentage 
  (total_crates : ℕ) 
  (total_cost : ℝ) 
  (lost_crates : ℕ) 
  (selling_price : ℝ) : 
  total_crates = 10 → 
  total_cost = 160 → 
  lost_crates = 2 → 
  selling_price = 25 → 
  ((total_crates - lost_crates) * selling_price - total_cost) / total_cost * 100 = 25 := by
  sorry

end NUMINAMATH_CALUDE_mango_profit_percentage_l900_90043


namespace NUMINAMATH_CALUDE_geometric_sequence_k_value_l900_90069

/-- Given a geometric sequence {a_n} with a₂ = 3, a₃ = 9, and a_k = 243, prove that k = 6 -/
theorem geometric_sequence_k_value (a : ℕ → ℝ) (k : ℕ) :
  (∀ n : ℕ, a (n + 1) / a n = a 3 / a 2) →  -- geometric sequence condition
  a 2 = 3 →
  a 3 = 9 →
  a k = 243 →
  k = 6 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_k_value_l900_90069


namespace NUMINAMATH_CALUDE_sharon_oranges_l900_90054

theorem sharon_oranges (janet_oranges total_oranges : ℕ) 
  (h1 : janet_oranges = 9) 
  (h2 : total_oranges = 16) : 
  total_oranges - janet_oranges = 7 := by
  sorry

end NUMINAMATH_CALUDE_sharon_oranges_l900_90054


namespace NUMINAMATH_CALUDE_base_n_multiple_of_11_l900_90049

theorem base_n_multiple_of_11 : 
  ∀ n : ℕ, 2 ≤ n ∧ n ≤ 100 → 
  ¬(11 ∣ (7 + 4*n + 6*n^2 + 3*n^3 + 4*n^4 + 3*n^5)) := by
sorry

end NUMINAMATH_CALUDE_base_n_multiple_of_11_l900_90049


namespace NUMINAMATH_CALUDE_largest_absolute_value_l900_90033

theorem largest_absolute_value : 
  let S : Finset ℤ := {4, -5, 0, -1}
  ∀ x ∈ S, |(-5 : ℤ)| ≥ |x| := by
  sorry

end NUMINAMATH_CALUDE_largest_absolute_value_l900_90033


namespace NUMINAMATH_CALUDE_product_of_multiples_l900_90057

def smallest_two_digit_multiple_of_5 : ℕ := 10

def smallest_three_digit_multiple_of_7 : ℕ := 105

theorem product_of_multiples : 
  smallest_two_digit_multiple_of_5 * smallest_three_digit_multiple_of_7 = 1050 := by
  sorry

end NUMINAMATH_CALUDE_product_of_multiples_l900_90057


namespace NUMINAMATH_CALUDE_william_final_napkins_l900_90098

def initial_napkins : ℕ := 15
def olivia_napkins : ℕ := 10
def amelia_napkins : ℕ := 2 * olivia_napkins

theorem william_final_napkins :
  initial_napkins + olivia_napkins + amelia_napkins = 45 :=
by sorry

end NUMINAMATH_CALUDE_william_final_napkins_l900_90098


namespace NUMINAMATH_CALUDE_one_sided_limits_arctg_reciprocal_l900_90077

noncomputable def f (x : ℝ) : ℝ := Real.arctan (1 / (x - 1))

theorem one_sided_limits_arctg_reciprocal :
  (∀ ε > 0, ∃ δ > 0, ∀ x > 1, |x - 1| < δ → |f x - π/2| < ε) ∧
  (∀ ε > 0, ∃ δ > 0, ∀ x < 1, |x - 1| < δ → |f x + π/2| < ε) :=
sorry

end NUMINAMATH_CALUDE_one_sided_limits_arctg_reciprocal_l900_90077


namespace NUMINAMATH_CALUDE_work_completion_time_l900_90011

/-- The time it takes for A to finish the remaining work after B has worked for 10 days -/
def remaining_time_for_A (a_time b_time b_work_days : ℚ) : ℚ :=
  (1 - b_work_days / b_time) / (1 / a_time)

theorem work_completion_time :
  remaining_time_for_A 9 15 10 = 3 := by sorry

end NUMINAMATH_CALUDE_work_completion_time_l900_90011


namespace NUMINAMATH_CALUDE_line_through_point_equal_intercepts_l900_90066

-- Define a line by its equation ax + by + c = 0
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

-- Define a point in 2D space
structure Point where
  x : ℝ
  y : ℝ

-- Function to check if a point lies on a line
def pointOnLine (p : Point) (l : Line) : Prop :=
  l.a * p.x + l.b * p.y + l.c = 0

-- Function to check if a line has equal intercepts on x and y axes
def equalIntercepts (l : Line) : Prop :=
  l.a ≠ 0 ∧ l.b ≠ 0 ∧ (-l.c / l.a = -l.c / l.b)

-- The main theorem
theorem line_through_point_equal_intercepts :
  ∃ (l₁ l₂ : Line),
    (pointOnLine ⟨3, -2⟩ l₁) ∧
    (pointOnLine ⟨3, -2⟩ l₂) ∧
    (equalIntercepts l₁ ∨ (l₁.a = 0 ∧ l₁.b = 0)) ∧
    (equalIntercepts l₂ ∨ (l₂.a = 0 ∧ l₂.b = 0)) ∧
    ((l₁.a = 2 ∧ l₁.b = 3 ∧ l₁.c = 0) ∨ (l₂.a = 1 ∧ l₂.b = 1 ∧ l₂.c = -1)) :=
  sorry

end NUMINAMATH_CALUDE_line_through_point_equal_intercepts_l900_90066


namespace NUMINAMATH_CALUDE_sum_of_two_numbers_l900_90042

theorem sum_of_two_numbers : ∃ (a b : ℤ), 
  (a = |(-10)| + 1) ∧ 
  (b = -(2) - 1) ∧ 
  (a + b = 8) := by
  sorry

end NUMINAMATH_CALUDE_sum_of_two_numbers_l900_90042


namespace NUMINAMATH_CALUDE_arithmetic_sequence_problem_l900_90083

/-- An arithmetic sequence -/
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

/-- Given an arithmetic sequence where a₃ + a₅ = 10, prove that a₄ = 5 -/
theorem arithmetic_sequence_problem (a : ℕ → ℝ) 
  (h_arith : arithmetic_sequence a) 
  (h_sum : a 3 + a 5 = 10) : 
  a 4 = 5 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_problem_l900_90083


namespace NUMINAMATH_CALUDE_total_repair_cost_is_4850_l900_90008

-- Define the repair costs
def engine_labor_rate : ℕ := 75
def engine_labor_hours : ℕ := 16
def engine_part_cost : ℕ := 1200

def brake_labor_rate : ℕ := 85
def brake_labor_hours : ℕ := 10
def brake_part_cost : ℕ := 800

def tire_labor_rate : ℕ := 50
def tire_labor_hours : ℕ := 4
def tire_part_cost : ℕ := 600

-- Define the total cost function
def total_repair_cost : ℕ :=
  (engine_labor_rate * engine_labor_hours + engine_part_cost) +
  (brake_labor_rate * brake_labor_hours + brake_part_cost) +
  (tire_labor_rate * tire_labor_hours + tire_part_cost)

-- Theorem statement
theorem total_repair_cost_is_4850 : total_repair_cost = 4850 := by
  sorry

end NUMINAMATH_CALUDE_total_repair_cost_is_4850_l900_90008


namespace NUMINAMATH_CALUDE_monthly_fee_plan_a_correct_l900_90051

/-- The monthly fee for Plan A in a cell phone company's text-messaging plans. -/
def monthly_fee_plan_a : ℝ := 9

/-- The cost per text message for Plan A. -/
def cost_per_text_plan_a : ℝ := 0.25

/-- The cost per text message for Plan B. -/
def cost_per_text_plan_b : ℝ := 0.40

/-- The number of text messages at which both plans cost the same. -/
def equal_cost_messages : ℕ := 60

/-- Theorem stating that the monthly fee for Plan A is correct. -/
theorem monthly_fee_plan_a_correct :
  monthly_fee_plan_a = 
    equal_cost_messages * (cost_per_text_plan_b - cost_per_text_plan_a) :=
by sorry

end NUMINAMATH_CALUDE_monthly_fee_plan_a_correct_l900_90051


namespace NUMINAMATH_CALUDE_base3_even_iff_sum_even_base10_multiple_of_7_iff_sum_congruent_l900_90015

/- Define a function to convert a list of digits to a number in base 3 -/
def toBase3 (digits : List Nat) : Nat :=
  digits.foldl (fun acc d => 3 * acc + d) 0

/- Define a function to check if a number is even -/
def isEven (n : Nat) : Bool :=
  n % 2 = 0

/- Define a function to sum the digits of a list -/
def sumDigits (digits : List Nat) : Nat :=
  digits.sum

/- Theorem for base 3 even numbers -/
theorem base3_even_iff_sum_even (digits : List Nat) :
  isEven (toBase3 digits) ↔ isEven (sumDigits digits) := by
  sorry

/- Define a function to convert a list of digits to a number in base 10 -/
def toBase10 (digits : List Nat) : Nat :=
  digits.foldl (fun acc d => 10 * acc + d) 0

/- Define a function to check if a number is divisible by 7 -/
def isDivisibleBy7 (n : Nat) : Bool :=
  n % 7 = 0

/- Define a function to compute the sum of digits multiplied by powers of 10 mod 7 -/
def sumDigitsPowersOf10Mod7 (digits : List Nat) : Nat :=
  (List.range digits.length).zip digits
  |> List.foldl (fun acc (i, d) => (acc + d * (10^i % 7)) % 7) 0

/- Theorem for base 10 multiples of 7 -/
theorem base10_multiple_of_7_iff_sum_congruent (digits : List Nat) :
  isDivisibleBy7 (toBase10 digits) ↔ sumDigitsPowersOf10Mod7 digits = 0 := by
  sorry

end NUMINAMATH_CALUDE_base3_even_iff_sum_even_base10_multiple_of_7_iff_sum_congruent_l900_90015


namespace NUMINAMATH_CALUDE_r_power_sum_l900_90061

theorem r_power_sum (r : ℝ) (h : (r + 1/r)^2 = 5) : r^4 + 1/r^4 = 7 := by
  sorry

end NUMINAMATH_CALUDE_r_power_sum_l900_90061


namespace NUMINAMATH_CALUDE_candy_mixture_cost_l900_90018

theorem candy_mixture_cost (first_candy_weight : ℝ) (second_candy_weight : ℝ) 
  (second_candy_price : ℝ) (mixture_price : ℝ) :
  first_candy_weight = 20 →
  second_candy_weight = 80 →
  second_candy_price = 5 →
  mixture_price = 6 →
  first_candy_weight + second_candy_weight = 100 →
  ∃ (first_candy_price : ℝ),
    first_candy_price * first_candy_weight + 
    second_candy_price * second_candy_weight = 
    mixture_price * (first_candy_weight + second_candy_weight) ∧
    first_candy_price = 10 := by
  sorry


end NUMINAMATH_CALUDE_candy_mixture_cost_l900_90018


namespace NUMINAMATH_CALUDE_unique_x_value_l900_90045

def A (x : ℝ) : Set ℝ := {2, 0, x}
def B (x : ℝ) : Set ℝ := {2, x^2}

theorem unique_x_value : 
  ∀ x : ℝ, (A x ∩ B x = B x) → x = 1 := by
  sorry

end NUMINAMATH_CALUDE_unique_x_value_l900_90045


namespace NUMINAMATH_CALUDE_perfect_square_binomial_l900_90014

theorem perfect_square_binomial (c : ℚ) : 
  (∃ t u : ℚ, ∀ x : ℚ, c * x^2 + (45/2) * x + 1 = (t * x + u)^2) → 
  c = 2025/16 := by
sorry

end NUMINAMATH_CALUDE_perfect_square_binomial_l900_90014


namespace NUMINAMATH_CALUDE_invalid_inequality_l900_90009

theorem invalid_inequality (a b : ℝ) (h1 : a < b) (h2 : b < 0) : ¬(1 / (a - b) > 1 / a) := by
  sorry

end NUMINAMATH_CALUDE_invalid_inequality_l900_90009


namespace NUMINAMATH_CALUDE_price_first_box_is_two_l900_90039

/-- The price of each movie in the first box -/
def price_first_box : ℝ := 2

/-- The number of movies bought from the first box -/
def num_first_box : ℕ := 10

/-- The number of movies bought from the second box -/
def num_second_box : ℕ := 5

/-- The price of each movie in the second box -/
def price_second_box : ℝ := 5

/-- The average price of all DVDs bought -/
def average_price : ℝ := 3

/-- The total number of movies bought -/
def total_movies : ℕ := num_first_box + num_second_box

theorem price_first_box_is_two :
  price_first_box * num_first_box + price_second_box * num_second_box = average_price * total_movies :=
by sorry

end NUMINAMATH_CALUDE_price_first_box_is_two_l900_90039


namespace NUMINAMATH_CALUDE_lewis_earnings_l900_90036

/-- Lewis's earnings during harvest season --/
theorem lewis_earnings (weekly_earnings weekly_rent : ℕ) (harvest_weeks : ℕ) : 
  weekly_earnings = 403 → 
  weekly_rent = 49 → 
  harvest_weeks = 233 → 
  (weekly_earnings * harvest_weeks) - (weekly_rent * harvest_weeks) = 82482 := by
  sorry

end NUMINAMATH_CALUDE_lewis_earnings_l900_90036
