import Mathlib

namespace NUMINAMATH_CALUDE_g_at_negative_two_l3051_305127

/-- The function g(x) = 2x^2 + 3x + 1 -/
def g (x : ℝ) : ℝ := 2 * x^2 + 3 * x + 1

/-- Theorem: g(-2) = 3 -/
theorem g_at_negative_two : g (-2) = 3 := by
  sorry

end NUMINAMATH_CALUDE_g_at_negative_two_l3051_305127


namespace NUMINAMATH_CALUDE_smaller_integer_is_49_l3051_305110

theorem smaller_integer_is_49 (m n : ℕ) : 
  10 ≤ m ∧ m < 100 ∧  -- m is a 2-digit positive integer
  10 ≤ n ∧ n < 100 ∧  -- n is a 2-digit positive integer
  ∃ k : ℕ, n = 25 * k ∧ -- n is a multiple of 25
  m < n ∧  -- n is larger than m
  (m + n) / 2 = m + n / 100  -- their average equals the decimal number
  → m = 49 := by sorry

end NUMINAMATH_CALUDE_smaller_integer_is_49_l3051_305110


namespace NUMINAMATH_CALUDE_union_of_M_and_N_l3051_305152

def M : Set ℝ := {x : ℝ | x^2 - x - 12 = 0}
def N : Set ℝ := {x : ℝ | x^2 + 3*x = 0}

theorem union_of_M_and_N : M ∪ N = {-3, 0, 4} := by
  sorry

end NUMINAMATH_CALUDE_union_of_M_and_N_l3051_305152


namespace NUMINAMATH_CALUDE_sequence_property_l3051_305157

-- Define the sequence type
def Sequence := ℕ+ → ℝ

-- Define the property of the sequence
def HasProperty (a : Sequence) : Prop :=
  ∀ n : ℕ+, a n * a (n + 2) = (a (n + 1))^2

-- State the theorem
theorem sequence_property (a : Sequence) 
  (h1 : HasProperty a) 
  (h2 : a 7 = 16) 
  (h3 : a 3 * a 5 = 4) : 
  a 3 = 1 := by
  sorry

end NUMINAMATH_CALUDE_sequence_property_l3051_305157


namespace NUMINAMATH_CALUDE_seashells_given_to_jason_l3051_305147

theorem seashells_given_to_jason (initial_seashells : ℕ) (remaining_seashells : ℕ) 
  (h1 : initial_seashells = 66) (h2 : remaining_seashells = 14) : 
  initial_seashells - remaining_seashells = 52 := by
  sorry

end NUMINAMATH_CALUDE_seashells_given_to_jason_l3051_305147


namespace NUMINAMATH_CALUDE_savings_goal_theorem_l3051_305104

/-- Calculates the amount to save per paycheck given the total savings goal,
    number of months, and number of paychecks per month. -/
def amount_per_paycheck (total_savings : ℚ) (num_months : ℕ) (paychecks_per_month : ℕ) : ℚ :=
  total_savings / (num_months * paychecks_per_month)

/-- Proves that saving $100 per paycheck for 15 months with 2 paychecks per month
    results in a total savings of $3000. -/
theorem savings_goal_theorem :
  amount_per_paycheck 3000 15 2 = 100 := by
  sorry

#eval amount_per_paycheck 3000 15 2

end NUMINAMATH_CALUDE_savings_goal_theorem_l3051_305104


namespace NUMINAMATH_CALUDE_marble_arrangement_theorem_l3051_305115

def total_arrangements (n : ℕ) : ℕ := Nat.factorial n

def restricted_arrangements (n : ℕ) : ℕ := 
  Nat.factorial 3 * Nat.factorial 2 * Nat.factorial 3

def valid_arrangements (n : ℕ) : ℕ := 
  total_arrangements n - restricted_arrangements n

theorem marble_arrangement_theorem : 
  valid_arrangements 5 = 48 := by sorry

end NUMINAMATH_CALUDE_marble_arrangement_theorem_l3051_305115


namespace NUMINAMATH_CALUDE_min_value_ab_l3051_305189

theorem min_value_ab (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : a + b + 3 = a * b) :
  a * b ≥ 9 ∧ ∃ (a₀ b₀ : ℝ), a₀ > 0 ∧ b₀ > 0 ∧ a₀ + b₀ + 3 = a₀ * b₀ ∧ a₀ * b₀ = 9 :=
sorry

end NUMINAMATH_CALUDE_min_value_ab_l3051_305189


namespace NUMINAMATH_CALUDE_prob_four_ones_in_five_rolls_l3051_305130

/-- The probability of rolling a 1 on a fair six-sided die -/
def prob_one : ℚ := 1 / 6

/-- The probability of not rolling a 1 on a fair six-sided die -/
def prob_not_one : ℚ := 5 / 6

/-- The number of rolls -/
def num_rolls : ℕ := 5

/-- The number of times we want to roll a 1 -/
def target_ones : ℕ := 4

/-- The probability of rolling exactly four 1s in five rolls of a fair six-sided die -/
theorem prob_four_ones_in_five_rolls : 
  (num_rolls.choose target_ones : ℚ) * prob_one ^ target_ones * prob_not_one ^ (num_rolls - target_ones) = 25 / 7776 := by
  sorry

end NUMINAMATH_CALUDE_prob_four_ones_in_five_rolls_l3051_305130


namespace NUMINAMATH_CALUDE_tuesday_tips_calculation_l3051_305145

def hourly_wage : ℕ := 10
def monday_hours : ℕ := 7
def monday_tips : ℕ := 18
def tuesday_hours : ℕ := 5
def wednesday_hours : ℕ := 7
def wednesday_tips : ℕ := 20
def total_earnings : ℕ := 240

theorem tuesday_tips_calculation :
  ∃ tuesday_tips : ℕ,
    hourly_wage * (monday_hours + tuesday_hours + wednesday_hours) +
    monday_tips + tuesday_tips + wednesday_tips = total_earnings ∧
    tuesday_tips = 12 := by
  sorry

end NUMINAMATH_CALUDE_tuesday_tips_calculation_l3051_305145


namespace NUMINAMATH_CALUDE_inequality_proof_l3051_305129

theorem inequality_proof (a b c d : ℝ) 
  (h_pos_a : a > 0) (h_pos_b : b > 0) (h_pos_c : c > 0) (h_pos_d : d > 0)
  (h_prod : a * b * c * d = 1) : 
  a^5 + b^5 + c^5 + d^5 ≥ a + b + c + d := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l3051_305129


namespace NUMINAMATH_CALUDE_gary_chicken_multiple_l3051_305102

/-- The multiple of chickens Gary has now compared to the start -/
def chicken_multiple (initial_chickens : ℕ) (eggs_per_day : ℕ) (total_eggs_per_week : ℕ) : ℕ :=
  (total_eggs_per_week / (eggs_per_day * 7)) / initial_chickens

/-- Proof that Gary's chicken multiple is 8 -/
theorem gary_chicken_multiple :
  chicken_multiple 4 6 1344 = 8 := by
  sorry

end NUMINAMATH_CALUDE_gary_chicken_multiple_l3051_305102


namespace NUMINAMATH_CALUDE_solution_and_minimum_value_l3051_305111

def A (a : ℕ) : Set ℝ := {x : ℝ | |x - 2| < a}

theorem solution_and_minimum_value (a : ℕ) (h1 : a > 0) 
  (h2 : (3/2 : ℝ) ∈ A a) (h3 : (1/2 : ℝ) ∉ A a) :
  (a = 1) ∧ 
  (∀ x : ℝ, |x + a| + |x - 2| ≥ 3) ∧ 
  (∃ x : ℝ, |x + a| + |x - 2| = 3) := by
sorry

end NUMINAMATH_CALUDE_solution_and_minimum_value_l3051_305111


namespace NUMINAMATH_CALUDE_f_composition_value_l3051_305153

noncomputable def f (x : ℝ) : ℝ :=
  if x ≤ 0 then Real.sin (5 * Real.pi * x / 2)
  else 1/6 - Real.log x / Real.log 3

theorem f_composition_value : f (f (3 * Real.sqrt 3)) = Real.sqrt 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_f_composition_value_l3051_305153


namespace NUMINAMATH_CALUDE_expected_voters_for_candidate_A_prove_expected_voters_for_candidate_A_l3051_305187

theorem expected_voters_for_candidate_A : ℝ → Prop :=
  fun x => 
    -- Define the percentage of Democrats
    let percent_democrats : ℝ := 0.60
    -- Define the percentage of Republicans
    let percent_republicans : ℝ := 1 - percent_democrats
    -- Define the percentage of Democrats voting for A
    let percent_democrats_for_A : ℝ := 0.85
    -- Define the percentage of Republicans voting for A
    let percent_republicans_for_A : ℝ := 0.20
    -- Calculate the total percentage of voters for A
    let total_percent_for_A : ℝ := 
      percent_democrats * percent_democrats_for_A + 
      percent_republicans * percent_republicans_for_A
    -- The theorem statement
    x = total_percent_for_A * 100

-- The proof of the theorem
theorem prove_expected_voters_for_candidate_A : 
  expected_voters_for_candidate_A 59 := by
  sorry

end NUMINAMATH_CALUDE_expected_voters_for_candidate_A_prove_expected_voters_for_candidate_A_l3051_305187


namespace NUMINAMATH_CALUDE_carolyn_final_marbles_l3051_305105

/-- Represents the number of marbles Carolyn has after sharing -/
def marbles_after_sharing (initial_marbles shared_marbles : ℕ) : ℕ :=
  initial_marbles - shared_marbles

/-- Theorem stating that Carolyn ends up with 5 marbles -/
theorem carolyn_final_marbles :
  marbles_after_sharing 47 42 = 5 := by
  sorry

end NUMINAMATH_CALUDE_carolyn_final_marbles_l3051_305105


namespace NUMINAMATH_CALUDE_abs_k_less_than_abs_b_l3051_305184

/-- Given a linear function y = kx + b, prove that |k| < |b| under certain conditions --/
theorem abs_k_less_than_abs_b (k b : ℝ) : 
  (∀ x y, y = k * x + b) →  -- The function is of the form y = kx + b
  (b > 0) →  -- The y-intercept is positive
  (0 < k + b) →  -- The point (1, k+b) is above the x-axis
  (k + b < b) →  -- The point (1, k+b) is below b
  |k| < |b| := by
sorry


end NUMINAMATH_CALUDE_abs_k_less_than_abs_b_l3051_305184


namespace NUMINAMATH_CALUDE_remaining_distance_to_nyc_l3051_305176

/-- Richard's journey from Cincinnati to New York City -/
def richards_journey (total_distance first_day second_day third_day : ℕ) : Prop :=
  let distance_walked := first_day + second_day + third_day
  total_distance - distance_walked = 36

theorem remaining_distance_to_nyc :
  richards_journey 70 20 4 10 := by sorry

end NUMINAMATH_CALUDE_remaining_distance_to_nyc_l3051_305176


namespace NUMINAMATH_CALUDE_y_value_proof_l3051_305154

theorem y_value_proof (y : ℝ) (h : (40 : ℝ) / 80 = Real.sqrt (y / 80)) : y = 20 := by
  sorry

end NUMINAMATH_CALUDE_y_value_proof_l3051_305154


namespace NUMINAMATH_CALUDE_base7_to_decimal_correct_l3051_305172

/-- Converts a base 7 digit to its decimal (base 10) value -/
def base7ToDecimal (d : ℕ) : ℕ := d

/-- Represents the number 23456 in base 7 as a list of its digits -/
def base7Number : List ℕ := [2, 3, 4, 5, 6]

/-- Converts a list of base 7 digits to its decimal (base 10) equivalent -/
def convertBase7ToDecimal (digits : List ℕ) : ℕ :=
  digits.enum.foldl (fun acc (i, d) => acc + (base7ToDecimal d) * 7^i) 0

theorem base7_to_decimal_correct :
  convertBase7ToDecimal base7Number = 6068 := by sorry

end NUMINAMATH_CALUDE_base7_to_decimal_correct_l3051_305172


namespace NUMINAMATH_CALUDE_special_polygon_properties_l3051_305183

/-- A polygon where the sum of interior angles is more than three times the sum of exterior angles by 180° --/
structure SpecialPolygon where
  n : ℕ
  interior_sum : ℝ
  exterior_sum : ℝ
  h : interior_sum = 3 * exterior_sum + 180

theorem special_polygon_properties (p : SpecialPolygon) :
  p.n = 9 ∧ p.n - 3 = 6 := by sorry


end NUMINAMATH_CALUDE_special_polygon_properties_l3051_305183


namespace NUMINAMATH_CALUDE_chords_from_eight_points_l3051_305194

/-- The number of chords that can be drawn from n points on a circle's circumference -/
def num_chords (n : ℕ) : ℕ := n.choose 2

/-- Theorem: The number of chords from 8 points on a circle is 28 -/
theorem chords_from_eight_points : num_chords 8 = 28 := by
  sorry

end NUMINAMATH_CALUDE_chords_from_eight_points_l3051_305194


namespace NUMINAMATH_CALUDE_fraction_equality_l3051_305165

theorem fraction_equality (m n r t : ℚ) 
  (h1 : m / n = 5 / 2) 
  (h2 : r / t = 8 / 5) : 
  (2 * m * r - 3 * n * t) / (5 * n * t - 4 * m * r) = -5 / 11 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_l3051_305165


namespace NUMINAMATH_CALUDE_model_y_completion_time_l3051_305136

/-- The time (in minutes) it takes for a Model Y computer to complete the task -/
def model_y_time : ℝ := 30

/-- The time (in minutes) it takes for a Model X computer to complete the task -/
def model_x_time : ℝ := 60

/-- The number of Model X computers used -/
def num_model_x : ℝ := 20

/-- The time (in minutes) it takes for both models working together to complete the task -/
def total_time : ℝ := 1

theorem model_y_completion_time :
  (num_model_x / model_x_time + num_model_x / model_y_time) * total_time = 1 :=
by sorry

end NUMINAMATH_CALUDE_model_y_completion_time_l3051_305136


namespace NUMINAMATH_CALUDE_function_properties_l3051_305146

def SymmetricAboutOrigin (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

def DecreasingOn (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x y, a ≤ x ∧ x < y ∧ y ≤ b → f y < f x

def HasMinValueOn (f : ℝ → ℝ) (v : ℝ) (a b : ℝ) : Prop :=
  (∀ x, a ≤ x ∧ x ≤ b → v ≤ f x) ∧ (∃ x, a ≤ x ∧ x ≤ b ∧ f x = v)

def HasMaxValueOn (f : ℝ → ℝ) (v : ℝ) (a b : ℝ) : Prop :=
  (∀ x, a ≤ x ∧ x ≤ b → f x ≤ v) ∧ (∃ x, a ≤ x ∧ x ≤ b ∧ f x = v)

theorem function_properties (f : ℝ → ℝ) :
  SymmetricAboutOrigin f →
  DecreasingOn f 1 5 →
  HasMinValueOn f 3 1 5 →
  DecreasingOn f (-5) (-1) ∧ HasMaxValueOn f (-3) (-5) (-1) :=
by sorry

end NUMINAMATH_CALUDE_function_properties_l3051_305146


namespace NUMINAMATH_CALUDE_ginkgo_field_length_l3051_305162

/-- The length of a field with evenly spaced trees -/
def field_length (num_trees : ℕ) (interval : ℕ) : ℕ :=
  (num_trees - 1) * interval

/-- Theorem: The length of a field with 10 ginkgo trees planted at 10-meter intervals, 
    including trees at both ends, is 90 meters. -/
theorem ginkgo_field_length : field_length 10 10 = 90 := by
  sorry

end NUMINAMATH_CALUDE_ginkgo_field_length_l3051_305162


namespace NUMINAMATH_CALUDE_sally_received_quarters_l3051_305122

/-- The number of quarters Sally initially had -/
def initial_quarters : ℕ := 760

/-- The number of quarters Sally now has -/
def final_quarters : ℕ := 1178

/-- The number of quarters Sally received -/
def received_quarters : ℕ := final_quarters - initial_quarters

theorem sally_received_quarters : received_quarters = 418 := by
  sorry

end NUMINAMATH_CALUDE_sally_received_quarters_l3051_305122


namespace NUMINAMATH_CALUDE_ellipse_slope_theorem_l3051_305138

/-- Given an ellipse with specific properties, prove the slope of a line passing through a point on the ellipse --/
theorem ellipse_slope_theorem (F₁ PF : ℝ) (k₂ : ℝ) :
  F₁ = (6/5) * Real.sqrt 5 →
  PF = (4/5) * Real.sqrt 5 →
  ∃ (k : ℝ), k = (3/2) * k₂ ∧ (k = (3 * Real.sqrt 5) / 10 ∨ k = -(3 * Real.sqrt 5) / 10) := by
  sorry

end NUMINAMATH_CALUDE_ellipse_slope_theorem_l3051_305138


namespace NUMINAMATH_CALUDE_largest_cube_in_sphere_l3051_305142

theorem largest_cube_in_sphere (a b c : ℝ) (ha : a = 22) (hb : b = 2) (hc : c = 10) :
  let cuboid_diagonal := Real.sqrt (a^2 + b^2 + c^2)
  let cube_side := Real.sqrt ((a^2 + b^2 + c^2) / 3)
  cube_side = 14 :=
sorry

end NUMINAMATH_CALUDE_largest_cube_in_sphere_l3051_305142


namespace NUMINAMATH_CALUDE_quadratic_root_in_interval_l3051_305158

theorem quadratic_root_in_interval
  (a b c : ℝ)
  (h_two_roots : ∃ x y : ℝ, x ≠ y ∧ a * x^2 + b * x + c = 0 ∧ a * y^2 + b * y + c = 0)
  (h_inequality : |a * (b - c)| > |b^2 - a * c| + |c^2 - a * b|) :
  ∃ α : ℝ, 0 < α ∧ α < 2 ∧ a * α^2 + b * α + c = 0 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_root_in_interval_l3051_305158


namespace NUMINAMATH_CALUDE_corn_farmer_profit_l3051_305151

/-- Calculates the profit for a corn farmer given specific conditions. -/
theorem corn_farmer_profit : 
  let seeds_per_ear : ℕ := 4
  let price_per_ear : ℚ := 1/10
  let seeds_per_bag : ℕ := 100
  let price_per_bag : ℚ := 1/2
  let ears_sold : ℕ := 500
  let total_seeds : ℕ := seeds_per_ear * ears_sold
  let bags_needed : ℕ := (total_seeds + seeds_per_bag - 1) / seeds_per_bag
  let total_cost : ℚ := bags_needed * price_per_bag
  let total_revenue : ℚ := ears_sold * price_per_ear
  let profit : ℚ := total_revenue - total_cost
  profit = 40 := by sorry

end NUMINAMATH_CALUDE_corn_farmer_profit_l3051_305151


namespace NUMINAMATH_CALUDE_soap_box_dimension_proof_l3051_305119

/-- Represents the dimensions of a rectangular box -/
structure BoxDimensions where
  length : ℝ
  width : ℝ
  height : ℝ

/-- Calculates the volume of a box given its dimensions -/
def boxVolume (d : BoxDimensions) : ℝ :=
  d.length * d.width * d.height

theorem soap_box_dimension_proof 
  (carton : BoxDimensions)
  (soap : BoxDimensions)
  (h1 : carton.length = 25)
  (h2 : carton.width = 48)
  (h3 : carton.height = 60)
  (h4 : soap.width = 6)
  (h5 : soap.height = 5)
  (h6 : (300 : ℝ) * boxVolume soap = boxVolume carton) :
  soap.length = 8 := by
sorry

end NUMINAMATH_CALUDE_soap_box_dimension_proof_l3051_305119


namespace NUMINAMATH_CALUDE_chameleons_cannot_be_same_color_l3051_305116

/-- Represents the color of a chameleon -/
inductive Color
  | Blue
  | White
  | Red

/-- Represents the state of chameleons on the island -/
structure ChameleonState where
  blue : Nat
  white : Nat
  red : Nat

/-- The initial state of chameleons -/
def initialState : ChameleonState :=
  { blue := 800, white := 220, red := 1003 }

/-- The total number of chameleons -/
def totalChameleons : Nat := 2023

/-- Calculates the invariant Q for a given state -/
def calculateQ (state : ChameleonState) : Int :=
  state.blue - state.white

/-- Represents a meeting between two chameleons of different colors -/
def meetingTransformation (state : ChameleonState) (c1 c2 : Color) : ChameleonState :=
  match c1, c2 with
  | Color.Blue, Color.White => { state with blue := state.blue - 1, white := state.white - 1, red := state.red + 2 }
  | Color.Blue, Color.Red => { state with blue := state.blue - 1, white := state.white + 2, red := state.red - 1 }
  | Color.White, Color.Red => { state with blue := state.blue + 2, white := state.white - 1, red := state.red - 1 }
  | Color.White, Color.Blue => { state with blue := state.blue - 1, white := state.white - 1, red := state.red + 2 }
  | Color.Red, Color.Blue => { state with blue := state.blue - 1, white := state.white + 2, red := state.red - 1 }
  | Color.Red, Color.White => { state with blue := state.blue + 2, white := state.white - 1, red := state.red - 1 }
  | _, _ => state  -- No change if same color

/-- Theorem: It is impossible for all chameleons to become the same color -/
theorem chameleons_cannot_be_same_color :
  ∀ (finalState : ChameleonState),
  (finalState.blue + finalState.white + finalState.red = totalChameleons) →
  (finalState.blue = totalChameleons ∨ finalState.white = totalChameleons ∨ finalState.red = totalChameleons) →
  False := by
  sorry


end NUMINAMATH_CALUDE_chameleons_cannot_be_same_color_l3051_305116


namespace NUMINAMATH_CALUDE_solution_is_five_binomial_coefficient_identity_l3051_305113

-- Define A_x
def A (x : ℕ) : ℕ := x * (x - 1) * (x - 2)

-- Part 1: Prove that the solution to 3A_x^3 = 2A_{x+1}^2 + 6A_x^2 is x = 5
theorem solution_is_five : ∃ (x : ℕ), x > 3 ∧ 3 * (A x)^3 = 2 * (A (x + 1))^2 + 6 * (A x)^2 ∧ x = 5 := by
  sorry

-- Part 2: Prove that kC_n^k = nC_{n-1}^{k-1}
theorem binomial_coefficient_identity (n k : ℕ) (h : k ≤ n) : 
  k * Nat.choose n k = n * Nat.choose (n - 1) (k - 1) := by
  sorry

end NUMINAMATH_CALUDE_solution_is_five_binomial_coefficient_identity_l3051_305113


namespace NUMINAMATH_CALUDE_tangent_line_equations_l3051_305161

/-- The function for which we're finding the tangent line -/
def f (x : ℝ) : ℝ := x^3 - 2*x

/-- The derivative of f -/
def f' (x : ℝ) : ℝ := 3*x^2 - 2

/-- The point through which the tangent line passes -/
def P : ℝ × ℝ := (2, 4)

/-- Theorem: The equations of the tangent lines to y = x³ - 2x passing through (2,4) -/
theorem tangent_line_equations :
  ∃ (m : ℝ), (f m = m^3 - 2*m) ∧
             ((4 - f m = (f' m) * (2 - m)) ∧
              (∀ x, f' m * (x - m) + f m = 10*x - 16 ∨
                    f' m * (x - m) + f m = x + 2)) :=
sorry

end NUMINAMATH_CALUDE_tangent_line_equations_l3051_305161


namespace NUMINAMATH_CALUDE_tangent_circle_equation_l3051_305141

-- Define the given circle C
def circle_C (x y : ℝ) : Prop := (x - 2)^2 + (y + 1)^2 = 4

-- Define the point of tangency
def tangent_point : ℝ × ℝ := (4, -1)

-- Define the radius of the new circle
def new_radius : ℝ := 1

-- Define the possible equations of the new circle
def new_circle_1 (x y : ℝ) : Prop := (x - 5)^2 + (y + 1)^2 = 1
def new_circle_2 (x y : ℝ) : Prop := (x - 3)^2 + (y + 1)^2 = 1

-- Theorem statement
theorem tangent_circle_equation : 
  ∃ (x y : ℝ), (circle_C x y ∧ (x, y) = tangent_point) → 
  (new_circle_1 x y ∨ new_circle_2 x y) :=
sorry

end NUMINAMATH_CALUDE_tangent_circle_equation_l3051_305141


namespace NUMINAMATH_CALUDE_candy_cost_450_l3051_305198

/-- The cost of buying a specified number of chocolate candies. -/
def candy_cost (total_candies : ℕ) (candies_per_box : ℕ) (cost_per_box : ℕ) : ℕ :=
  (total_candies / candies_per_box) * cost_per_box

/-- Theorem: The cost of buying 450 chocolate candies is $120, given that a box of 30 candies costs $8. -/
theorem candy_cost_450 : candy_cost 450 30 8 = 120 := by
  sorry

end NUMINAMATH_CALUDE_candy_cost_450_l3051_305198


namespace NUMINAMATH_CALUDE_lcm_520_693_l3051_305140

theorem lcm_520_693 : Nat.lcm 520 693 = 360360 := by
  sorry

end NUMINAMATH_CALUDE_lcm_520_693_l3051_305140


namespace NUMINAMATH_CALUDE_trigonometric_identity_l3051_305144

theorem trigonometric_identity (c : ℝ) (h : c = 2 * Real.pi / 9) :
  (Real.sin (2 * c) * Real.sin (5 * c) * Real.sin (8 * c) * Real.sin (11 * c) * Real.sin (14 * c)) /
  (Real.sin c * Real.sin (3 * c) * Real.sin (4 * c) * Real.sin (7 * c) * Real.sin (8 * c)) =
  Real.sin (80 * Real.pi / 180) :=
by sorry

end NUMINAMATH_CALUDE_trigonometric_identity_l3051_305144


namespace NUMINAMATH_CALUDE_comic_book_collections_l3051_305199

/-- Kymbrea's initial comic book collection -/
def kymbrea_initial : ℕ := 50

/-- LaShawn's initial comic book collection -/
def lashawn_initial : ℕ := 20

/-- Kymbrea's monthly comic book addition -/
def kymbrea_monthly : ℕ := 3

/-- LaShawn's monthly comic book addition -/
def lashawn_monthly : ℕ := 5

/-- The number of months after which LaShawn's collection is three times Kymbrea's -/
def months : ℕ := 33

theorem comic_book_collections : 
  lashawn_initial + lashawn_monthly * months = 3 * (kymbrea_initial + kymbrea_monthly * months) :=
by sorry

end NUMINAMATH_CALUDE_comic_book_collections_l3051_305199


namespace NUMINAMATH_CALUDE_age_ratio_theorem_l3051_305148

/-- Represents the ages of John and Mary -/
structure Ages where
  john : ℕ
  mary : ℕ

/-- The conditions of the problem -/
def problem_conditions (a : Ages) : Prop :=
  (a.john - 5 = 2 * (a.mary - 5)) ∧ 
  (a.john - 12 = 3 * (a.mary - 12))

/-- The ratio condition we're looking for -/
def ratio_condition (a : Ages) (years : ℕ) : Prop :=
  3 * (a.mary + years) = 2 * (a.john + years)

/-- The main theorem -/
theorem age_ratio_theorem (a : Ages) :
  problem_conditions a → ∃ years : ℕ, years = 9 ∧ ratio_condition a years := by
  sorry


end NUMINAMATH_CALUDE_age_ratio_theorem_l3051_305148


namespace NUMINAMATH_CALUDE_inequality_proof_l3051_305179

theorem inequality_proof (a b : ℝ) : 
  |a + b| / (1 + |a + b|) ≤ |a| / (1 + |a|) + |b| / (1 + |b|) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l3051_305179


namespace NUMINAMATH_CALUDE_square_implies_four_right_angles_but_not_conversely_l3051_305163

-- Define a quadrilateral
structure Quadrilateral :=
  (A B C D : Point)

-- Define a square
def is_square (q : Quadrilateral) : Prop :=
  -- A square has four equal sides and four right angles
  sorry

-- Define a quadrilateral with four right angles
def has_four_right_angles (q : Quadrilateral) : Prop :=
  -- A quadrilateral has four right angles
  sorry

-- Theorem statement
theorem square_implies_four_right_angles_but_not_conversely :
  (∀ q : Quadrilateral, is_square q → has_four_right_angles q) ∧
  (∃ q : Quadrilateral, has_four_right_angles q ∧ ¬is_square q) :=
sorry

end NUMINAMATH_CALUDE_square_implies_four_right_angles_but_not_conversely_l3051_305163


namespace NUMINAMATH_CALUDE_walkway_problem_l3051_305164

/-- Represents the scenario of a person walking on a moving walkway -/
structure WalkwayScenario where
  length : ℝ
  time_with : ℝ
  time_against : ℝ

/-- Calculates the time taken to walk when the walkway is stopped -/
def time_when_stopped (scenario : WalkwayScenario) : ℝ :=
  -- The actual calculation is not provided here
  sorry

/-- Theorem stating the correct time when the walkway is stopped -/
theorem walkway_problem (scenario : WalkwayScenario) 
  (h1 : scenario.length = 80)
  (h2 : scenario.time_with = 40)
  (h3 : scenario.time_against = 120) :
  time_when_stopped scenario = 60 := by
  sorry

end NUMINAMATH_CALUDE_walkway_problem_l3051_305164


namespace NUMINAMATH_CALUDE_surface_area_increase_after_removal_l3051_305150

/-- Represents a rectangular solid with length, width, and height -/
structure RectangularSolid where
  length : ℝ
  width : ℝ
  height : ℝ

/-- Calculates the surface area of a rectangular solid -/
def surfaceArea (solid : RectangularSolid) : ℝ :=
  2 * (solid.length * solid.width + solid.length * solid.height + solid.width * solid.height)

/-- Represents the change in surface area after removal of a smaller prism -/
def surfaceAreaChange (larger : RectangularSolid) (smaller : RectangularSolid) : ℝ :=
  (smaller.length * smaller.width + smaller.length * smaller.height + smaller.width * smaller.height) * 2 -
  smaller.length * smaller.width

theorem surface_area_increase_after_removal :
  let larger := RectangularSolid.mk 5 3 2
  let smaller := RectangularSolid.mk 2 1 1
  surfaceAreaChange larger smaller = 4 := by
  sorry


end NUMINAMATH_CALUDE_surface_area_increase_after_removal_l3051_305150


namespace NUMINAMATH_CALUDE_convex_nonagon_diagonals_l3051_305177

/-- The number of distinct diagonals in a convex nonagon -/
def nonagon_diagonals : ℕ := 27

/-- A convex nonagon has 27 distinct diagonals -/
theorem convex_nonagon_diagonals : 
  ∀ (n : ℕ), n = 9 → nonagon_diagonals = n * (n - 3) / 2 :=
by sorry

end NUMINAMATH_CALUDE_convex_nonagon_diagonals_l3051_305177


namespace NUMINAMATH_CALUDE_root_equation_q_value_l3051_305132

theorem root_equation_q_value (a b m p q : ℝ) : 
  (a^2 - m*a + (3/2) = 0) →
  (b^2 - m*b + (3/2) = 0) →
  ((a + 1/b)^2 - p*(a + 1/b) + q = 0) →
  ((b + 1/a)^2 - p*(b + 1/a) + q = 0) →
  q = 19/6 := by
sorry

end NUMINAMATH_CALUDE_root_equation_q_value_l3051_305132


namespace NUMINAMATH_CALUDE_square_field_diagonal_l3051_305174

theorem square_field_diagonal (area : ℝ) (diagonal : ℝ) : 
  area = 128 → diagonal = 16 := by
  sorry

end NUMINAMATH_CALUDE_square_field_diagonal_l3051_305174


namespace NUMINAMATH_CALUDE_arithmetic_progression_1980_l3051_305155

/-- An arithmetic progression of natural numbers. -/
structure ArithProgression where
  first : ℕ
  diff : ℕ

/-- Check if a natural number belongs to an arithmetic progression. -/
def belongsTo (n : ℕ) (ap : ArithProgression) : Prop :=
  ∃ k : ℕ, n = ap.first + k * ap.diff

/-- The main theorem statement. -/
theorem arithmetic_progression_1980 (P₁ P₂ P₃ : ArithProgression) :
  (∀ n : ℕ, n ≤ 8 → belongsTo n P₁ ∨ belongsTo n P₂ ∨ belongsTo n P₃) →
  belongsTo 1980 P₁ ∨ belongsTo 1980 P₂ ∨ belongsTo 1980 P₃ := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_progression_1980_l3051_305155


namespace NUMINAMATH_CALUDE_christen_peeled_22_l3051_305171

/-- Represents the potato peeling scenario -/
structure PotatoPeeling where
  initial_potatoes : ℕ
  homer_rate : ℕ
  christen_rate : ℕ
  time_before_christen : ℕ

/-- Calculates the number of potatoes Christen peeled -/
def potatoes_peeled_by_christen (scenario : PotatoPeeling) : ℕ :=
  sorry

/-- Theorem stating that Christen peeled 22 potatoes -/
theorem christen_peeled_22 (scenario : PotatoPeeling) 
  (h1 : scenario.initial_potatoes = 60)
  (h2 : scenario.homer_rate = 4)
  (h3 : scenario.christen_rate = 6)
  (h4 : scenario.time_before_christen = 6) :
  potatoes_peeled_by_christen scenario = 22 := by
  sorry

end NUMINAMATH_CALUDE_christen_peeled_22_l3051_305171


namespace NUMINAMATH_CALUDE_first_number_in_ratio_l3051_305114

theorem first_number_in_ratio (a b : ℕ) : 
  a ≠ 0 → b ≠ 0 → 
  (a : ℚ) / b = 3 / 4 → 
  Nat.lcm a b = 84 → 
  a = 63 :=
by sorry

end NUMINAMATH_CALUDE_first_number_in_ratio_l3051_305114


namespace NUMINAMATH_CALUDE_complex_equation_solution_l3051_305197

theorem complex_equation_solution (a : ℝ) : 
  (Complex.I + a) * (1 - a * Complex.I) = 2 → a = 1 := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l3051_305197


namespace NUMINAMATH_CALUDE_bar_charts_cannot_show_change_line_charts_can_show_change_bar_charts_show_change_is_false_l3051_305186

-- Define the types of charts
inductive Chart
| Bar
| Line

-- Define the capability of showing increase or decrease
def can_show_change (c : Chart) : Prop :=
  match c with
  | Chart.Line => true
  | Chart.Bar => false

-- Theorem stating that bar charts cannot show change
theorem bar_charts_cannot_show_change :
  ¬(can_show_change Chart.Bar) :=
by
  sorry

-- Theorem stating that line charts can show change
theorem line_charts_can_show_change :
  can_show_change Chart.Line :=
by
  sorry

-- Main theorem proving the original statement is false
theorem bar_charts_show_change_is_false :
  ¬(∀ (c : Chart), can_show_change c) :=
by
  sorry

end NUMINAMATH_CALUDE_bar_charts_cannot_show_change_line_charts_can_show_change_bar_charts_show_change_is_false_l3051_305186


namespace NUMINAMATH_CALUDE_derivative_f_l3051_305190

noncomputable def f (x : ℝ) : ℝ := (Real.sinh x) / (2 * (Real.cosh x)^2) + (1/2) * Real.arctan (Real.sinh x)

theorem derivative_f (x : ℝ) : 
  deriv f x = 1 / (Real.cosh x)^3 := by sorry

end NUMINAMATH_CALUDE_derivative_f_l3051_305190


namespace NUMINAMATH_CALUDE_bowling_team_weight_l3051_305118

theorem bowling_team_weight (x : ℝ) : 
  let initial_players : ℕ := 7
  let initial_avg_weight : ℝ := 94
  let new_players : ℕ := 2
  let known_new_player_weight : ℝ := 60
  let new_avg_weight : ℝ := 92
  (initial_players * initial_avg_weight + x + known_new_player_weight) / 
    (initial_players + new_players) = new_avg_weight → x = 110 :=
by sorry

end NUMINAMATH_CALUDE_bowling_team_weight_l3051_305118


namespace NUMINAMATH_CALUDE_triangle_ratio_l3051_305126

-- Define the triangle ABC
structure Triangle :=
  (A B C : ℝ)  -- Angles
  (a b c : ℝ)  -- Side lengths
  (area : ℝ)   -- Area

-- Define the properties of the triangle
def triangle_properties (t : Triangle) : Prop :=
  t.area = 8 ∧ t.a = 5 ∧ Real.tan t.B = -4/3

-- Define the theorem
theorem triangle_ratio (t : Triangle) (h : triangle_properties t) :
  (t.a + t.b + t.c) / (Real.sin t.A + Real.sin t.B + Real.sin t.C) = 5 * Real.sqrt 65 / 4 :=
sorry

end NUMINAMATH_CALUDE_triangle_ratio_l3051_305126


namespace NUMINAMATH_CALUDE_segment_length_ratio_l3051_305107

/-- Given points P, Q, R, and S on a line segment PQ, where PQ = 4PS and PQ = 8QR,
    the length of segment RS is 5/8 of the length of PQ. -/
theorem segment_length_ratio (P Q R S : Real) 
  (h1 : P ≤ R) (h2 : R ≤ S) (h3 : S ≤ Q)  -- Points order on the line
  (h4 : Q - P = 4 * (S - P))  -- PQ = 4PS
  (h5 : Q - P = 8 * (Q - R))  -- PQ = 8QR
  : S - R = 5/8 * (Q - P) := by
  sorry

end NUMINAMATH_CALUDE_segment_length_ratio_l3051_305107


namespace NUMINAMATH_CALUDE_weight_replacement_l3051_305181

/-- Given 5 people, if replacing one person with a new person weighing 70 kg
    increases the average weight by 4 kg, then the replaced person weighed 50 kg. -/
theorem weight_replacement (initial_count : ℕ) (weight_increase : ℝ) (new_weight : ℝ) :
  initial_count = 5 →
  weight_increase = 4 →
  new_weight = 70 →
  (initial_count : ℝ) * weight_increase = new_weight - 50 := by
  sorry

end NUMINAMATH_CALUDE_weight_replacement_l3051_305181


namespace NUMINAMATH_CALUDE_students_not_enrolled_in_languages_l3051_305195

/-- Given a class with the following properties:
  * There are 150 students in total
  * 61 students are taking French
  * 32 students are taking German
  * 45 students are taking Spanish
  * 15 students are taking both French and German
  * 12 students are taking both French and Spanish
  * 10 students are taking both German and Spanish
  * 5 students are taking all three languages
  This theorem proves that the number of students not enrolled in any
  of these language courses is 44. -/
theorem students_not_enrolled_in_languages (total : ℕ) (french : ℕ) (german : ℕ) (spanish : ℕ)
  (french_and_german : ℕ) (french_and_spanish : ℕ) (german_and_spanish : ℕ) (all_three : ℕ)
  (h_total : total = 150)
  (h_french : french = 61)
  (h_german : german = 32)
  (h_spanish : spanish = 45)
  (h_french_and_german : french_and_german = 15)
  (h_french_and_spanish : french_and_spanish = 12)
  (h_german_and_spanish : german_and_spanish = 10)
  (h_all_three : all_three = 5) :
  total - (french + german + spanish - french_and_german - french_and_spanish - german_and_spanish + all_three) = 44 := by
  sorry

end NUMINAMATH_CALUDE_students_not_enrolled_in_languages_l3051_305195


namespace NUMINAMATH_CALUDE_new_profit_percentage_l3051_305124

/-- Given the initial and new manufacturing costs, and the initial profit percentage,
    calculate the new profit percentage of the selling price. -/
theorem new_profit_percentage
  (initial_cost : ℝ)
  (new_cost : ℝ)
  (initial_profit_percentage : ℝ)
  (h_initial_cost : initial_cost = 70)
  (h_new_cost : new_cost = 50)
  (h_initial_profit_percentage : initial_profit_percentage = 30)
  : (1 - new_cost / (initial_cost / (1 - initial_profit_percentage / 100))) * 100 = 50 := by
  sorry

#check new_profit_percentage

end NUMINAMATH_CALUDE_new_profit_percentage_l3051_305124


namespace NUMINAMATH_CALUDE_remaining_volume_cube_with_hole_l3051_305133

/-- The remaining volume of a cube after drilling a cylindrical hole -/
theorem remaining_volume_cube_with_hole (cube_side : Real) (hole_radius : Real) (hole_height : Real) :
  cube_side = 6 →
  hole_radius = 3 →
  hole_height = 4 →
  cube_side ^ 3 - π * hole_radius ^ 2 * hole_height = 216 - 36 * π := by
  sorry

#check remaining_volume_cube_with_hole

end NUMINAMATH_CALUDE_remaining_volume_cube_with_hole_l3051_305133


namespace NUMINAMATH_CALUDE_polynomial_division_theorem_l3051_305117

theorem polynomial_division_theorem (a b c : ℚ) : 
  (∀ x, (17 * x^2 - 3 * x + 4) - (a * x^2 + b * x + c) = (5 * x + 6) * (2 * x + 1)) →
  a - b - c = 29 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_division_theorem_l3051_305117


namespace NUMINAMATH_CALUDE_profit_percent_calculation_l3051_305178

theorem profit_percent_calculation (selling_price cost_price profit : ℝ) :
  cost_price = 0.75 * selling_price →
  profit = selling_price - cost_price →
  (profit / cost_price) * 100 = 33.33333333333333 := by
  sorry

end NUMINAMATH_CALUDE_profit_percent_calculation_l3051_305178


namespace NUMINAMATH_CALUDE_field_length_is_96_l3051_305182

/-- Proves that the length of a rectangular field is 96 meters given specific conditions -/
theorem field_length_is_96 (w : ℝ) (l : ℝ) : 
  l = 2 * w →                   -- length is double the width
  64 = (1 / 72) * (l * w) →     -- area of pond (8^2) is 1/72 of field area
  l = 96 := by
sorry

end NUMINAMATH_CALUDE_field_length_is_96_l3051_305182


namespace NUMINAMATH_CALUDE_parallel_line_through_point_desired_line_equation_l3051_305109

/-- A line in 2D space represented by the equation ax + by + c = 0 -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Check if two lines are parallel -/
def parallel (l1 l2 : Line) : Prop :=
  l1.a * l2.b = l1.b * l2.a

/-- Check if a point (x, y) lies on a line -/
def on_line (x y : ℝ) (l : Line) : Prop :=
  l.a * x + l.b * y + l.c = 0

theorem parallel_line_through_point (P : ℝ × ℝ) (l : Line) :
  ∃ (l' : Line), parallel l' l ∧ on_line P.1 P.2 l' :=
by sorry

theorem desired_line_equation (P : ℝ × ℝ) (l l' : Line) :
  P = (-1, 3) →
  l = Line.mk 1 (-2) 3 →
  parallel l' l →
  on_line P.1 P.2 l' →
  l' = Line.mk 1 (-2) 7 :=
by sorry

end NUMINAMATH_CALUDE_parallel_line_through_point_desired_line_equation_l3051_305109


namespace NUMINAMATH_CALUDE_union_of_sets_l3051_305135

theorem union_of_sets (a : ℝ) : 
  let A : Set ℝ := {a^2 + 1, 2*a}
  let B : Set ℝ := {a + 1, 0}
  (A ∩ B).Nonempty → A ∪ B = {0, 1} := by
  sorry

end NUMINAMATH_CALUDE_union_of_sets_l3051_305135


namespace NUMINAMATH_CALUDE_num_solutions_eq_53_l3051_305166

/-- The number of solutions in positive integers for the equation 2x + 3y = 317 -/
def num_solutions : ℕ :=
  (Finset.filter (fun p : ℕ × ℕ => 2 * p.1 + 3 * p.2 = 317 ∧ p.1 > 0 ∧ p.2 > 0)
    (Finset.product (Finset.range 318) (Finset.range 318))).card

/-- Theorem stating that the number of solutions is 53 -/
theorem num_solutions_eq_53 : num_solutions = 53 := by
  sorry

end NUMINAMATH_CALUDE_num_solutions_eq_53_l3051_305166


namespace NUMINAMATH_CALUDE_winnie_balloon_distribution_l3051_305175

theorem winnie_balloon_distribution (total_balloons : ℕ) (num_friends : ℕ) 
  (h1 : total_balloons = 272) (h2 : num_friends = 5) :
  total_balloons % num_friends = 2 := by
  sorry

end NUMINAMATH_CALUDE_winnie_balloon_distribution_l3051_305175


namespace NUMINAMATH_CALUDE_parabola_line_slope_l3051_305170

-- Define the parabola
def parabola (x y : ℝ) : Prop := y^2 = 4*x

-- Define the focus of the parabola
def focus : ℝ × ℝ := (1, 0)

-- Define a point on the parabola
def on_parabola (P : ℝ × ℝ) : Prop := parabola P.1 P.2

-- Define a point in the first quadrant
def in_first_quadrant (P : ℝ × ℝ) : Prop := P.1 > 0 ∧ P.2 > 0

-- Define the vector relationship
def vector_relation (P Q : ℝ × ℝ) : Prop :=
  3 * (focus.1 - P.1) = Q.1 - focus.1 ∧
  3 * (focus.2 - P.2) = Q.2 - focus.2

-- Main theorem
theorem parabola_line_slope (P Q : ℝ × ℝ) :
  on_parabola P →
  on_parabola Q →
  in_first_quadrant Q →
  vector_relation P Q →
  (Q.2 - P.2) / (Q.1 - P.1) = Real.sqrt 3 :=
sorry

end NUMINAMATH_CALUDE_parabola_line_slope_l3051_305170


namespace NUMINAMATH_CALUDE_correct_result_l3051_305185

def add_subtract_round (a b c : ℕ) : ℕ :=
  let result := a + b - c
  let remainder := result % 5
  if remainder < 3 then result - remainder else result + (5 - remainder)

theorem correct_result : add_subtract_round 82 56 15 = 125 := by
  sorry

end NUMINAMATH_CALUDE_correct_result_l3051_305185


namespace NUMINAMATH_CALUDE_bird_legs_count_l3051_305169

theorem bird_legs_count (num_birds : ℕ) (legs_per_bird : ℕ) (h1 : num_birds = 5) (h2 : legs_per_bird = 2) :
  num_birds * legs_per_bird = 10 := by
  sorry

end NUMINAMATH_CALUDE_bird_legs_count_l3051_305169


namespace NUMINAMATH_CALUDE_correct_purchase_combinations_l3051_305108

/-- The number of oreo flavors -/
def oreo_flavors : ℕ := 7

/-- The number of milk flavors -/
def milk_flavors : ℕ := 4

/-- The total number of product flavors -/
def total_flavors : ℕ := oreo_flavors + milk_flavors

/-- The total number of products they purchase -/
def total_products : ℕ := 4

/-- The number of ways Alpha and Beta could have left the store with 4 products collectively -/
def purchase_combinations : ℕ := sorry

theorem correct_purchase_combinations :
  purchase_combinations = 4054 := by sorry

end NUMINAMATH_CALUDE_correct_purchase_combinations_l3051_305108


namespace NUMINAMATH_CALUDE_sports_conference_games_l3051_305128

/-- Calculates the total number of games in a sports conference season --/
def total_games (total_teams : ℕ) (teams_per_division : ℕ) (intra_division_games : ℕ) (inter_division_games : ℕ) : ℕ :=
  let games_per_team := (teams_per_division - 1) * intra_division_games + teams_per_division * inter_division_games
  (total_teams * games_per_team) / 2

theorem sports_conference_games : 
  total_games 12 6 3 2 = 162 := by sorry

end NUMINAMATH_CALUDE_sports_conference_games_l3051_305128


namespace NUMINAMATH_CALUDE_right_prism_cross_section_type_l3051_305112

/-- Represents a right prism -/
structure RightPrism where
  -- Add necessary fields here
  mk :: -- Constructor

/-- Represents a cross-section of a prism -/
inductive CrossSection
  | GeneralTrapezoid
  | IsoscelesTrapezoid
  | Other

/-- Function to determine the type of cross-section through the centers of base faces -/
def crossSectionThroughCenters (prism : RightPrism) : CrossSection :=
  sorry

/-- Theorem stating that the cross-section through the centers of base faces
    of a right prism is either a general trapezoid or an isosceles trapezoid -/
theorem right_prism_cross_section_type (prism : RightPrism) :
  (crossSectionThroughCenters prism = CrossSection.GeneralTrapezoid) ∨
  (crossSectionThroughCenters prism = CrossSection.IsoscelesTrapezoid) :=
by
  sorry

end NUMINAMATH_CALUDE_right_prism_cross_section_type_l3051_305112


namespace NUMINAMATH_CALUDE_average_price_is_86_l3051_305173

def prices : List ℝ := [82, 86, 90, 85, 87, 85, 86, 82, 90, 87, 85, 86, 82, 86, 87, 90]

theorem average_price_is_86 : 
  (prices.sum / prices.length : ℝ) = 86 := by sorry

end NUMINAMATH_CALUDE_average_price_is_86_l3051_305173


namespace NUMINAMATH_CALUDE_problem_statement_l3051_305120

def C : Set ℕ := {x | ∃ s t : ℕ, x = 1999 * s + 2000 * t}

theorem problem_statement :
  (3994001 ∉ C) ∧
  (∀ n : ℕ, 0 ≤ n ∧ n ≤ 3994001 ∧ n ∉ C → (3994001 - n) ∈ C) := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l3051_305120


namespace NUMINAMATH_CALUDE_three_digit_rotations_divisibility_l3051_305125

/-- Represents a three-digit number -/
structure ThreeDigitNumber where
  hundreds : Nat
  tens : Nat
  ones : Nat
  h_hundreds : hundreds ≥ 1 ∧ hundreds ≤ 9
  h_tens : tens ≥ 0 ∧ tens ≤ 9
  h_ones : ones ≥ 0 ∧ ones ≤ 9

/-- Converts a ThreeDigitNumber to its numeric value -/
def ThreeDigitNumber.toNat (n : ThreeDigitNumber) : Nat :=
  100 * n.hundreds + 10 * n.tens + n.ones

/-- Rotates the digits of a ThreeDigitNumber once to the left -/
def ThreeDigitNumber.rotateLeft (n : ThreeDigitNumber) : ThreeDigitNumber where
  hundreds := n.tens
  tens := n.ones
  ones := n.hundreds
  h_hundreds := by sorry
  h_tens := by sorry
  h_ones := by sorry

/-- Rotates the digits of a ThreeDigitNumber twice to the left -/
def ThreeDigitNumber.rotateLeftTwice (n : ThreeDigitNumber) : ThreeDigitNumber where
  hundreds := n.ones
  tens := n.hundreds
  ones := n.tens
  h_hundreds := by sorry
  h_tens := by sorry
  h_ones := by sorry

theorem three_digit_rotations_divisibility (n : ThreeDigitNumber) :
  27 ∣ n.toNat → 27 ∣ (n.rotateLeft).toNat ∧ 27 ∣ (n.rotateLeftTwice).toNat := by
  sorry

end NUMINAMATH_CALUDE_three_digit_rotations_divisibility_l3051_305125


namespace NUMINAMATH_CALUDE_complex_modulus_problem_l3051_305167

theorem complex_modulus_problem (z : ℂ) (h : z * (1 + Complex.I) = 2 - Complex.I) : 
  Complex.abs z = Real.sqrt 10 / 2 := by
  sorry

end NUMINAMATH_CALUDE_complex_modulus_problem_l3051_305167


namespace NUMINAMATH_CALUDE_expression_simplification_l3051_305100

theorem expression_simplification :
  let a := 3
  let b := 4
  let c := 5
  let d := 6
  (Real.sqrt (a + b + c + d) / 3) + ((a * b + 10) / 4) = 5.5 + Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l3051_305100


namespace NUMINAMATH_CALUDE_red_sector_overlap_l3051_305134

theorem red_sector_overlap (n : ℕ) (red_sectors : ℕ) (h1 : n = 1965) (h2 : red_sectors = 200) :
  ∃ (positions : Finset ℕ), 
    (Finset.card positions ≥ 60) ∧ 
    (∀ p ∈ positions, p < n) ∧
    (∀ p ∈ positions, (red_sectors * red_sectors - n * 20) / n ≤ red_sectors - 
      (red_sectors * red_sectors - (n - p) * red_sectors) / n) :=
sorry

end NUMINAMATH_CALUDE_red_sector_overlap_l3051_305134


namespace NUMINAMATH_CALUDE_log_equation_solution_l3051_305103

theorem log_equation_solution (x y : ℝ) (h : x > 0 ∧ y > 0 ∧ x - 2*y > 0) :
  2 * Real.log (x - 2*y) = Real.log x + Real.log y → x / y = 4 := by
sorry

end NUMINAMATH_CALUDE_log_equation_solution_l3051_305103


namespace NUMINAMATH_CALUDE_dot_product_equals_one_l3051_305180

def a : ℝ × ℝ := (1, -1)
def b : ℝ × ℝ := (-1, 2)

theorem dot_product_equals_one :
  (2 • a + b) • a = 1 := by sorry

end NUMINAMATH_CALUDE_dot_product_equals_one_l3051_305180


namespace NUMINAMATH_CALUDE_inequality_proof_l3051_305159

theorem inequality_proof (a b : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (h : a > b) :
  1 / (a * b^2) > 1 / (a^2 * b) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l3051_305159


namespace NUMINAMATH_CALUDE_wax_needed_proof_l3051_305168

/-- Given an amount of wax and a required amount, calculate the additional wax needed -/
def additional_wax_needed (current_amount required_amount : ℕ) : ℕ :=
  required_amount - current_amount

/-- Theorem stating that 17 grams of additional wax are needed -/
theorem wax_needed_proof (current_amount required_amount : ℕ) 
  (h1 : current_amount = 557)
  (h2 : required_amount = 574) :
  additional_wax_needed current_amount required_amount = 17 := by
  sorry

end NUMINAMATH_CALUDE_wax_needed_proof_l3051_305168


namespace NUMINAMATH_CALUDE_factorial_500_trailing_zeroes_l3051_305121

/-- The number of trailing zeroes in n! -/
def trailingZeroes (n : ℕ) : ℕ :=
  (n / 5) + (n / 25) + (n / 125)

/-- Theorem: 500! has 124 trailing zeroes -/
theorem factorial_500_trailing_zeroes :
  trailingZeroes 500 = 124 := by
  sorry

end NUMINAMATH_CALUDE_factorial_500_trailing_zeroes_l3051_305121


namespace NUMINAMATH_CALUDE_younger_person_age_l3051_305196

theorem younger_person_age (elder_age younger_age : ℕ) : 
  elder_age = younger_age + 20 →
  elder_age = 32 →
  elder_age - 7 = 5 * (younger_age - 7) →
  younger_age = 12 := by
sorry

end NUMINAMATH_CALUDE_younger_person_age_l3051_305196


namespace NUMINAMATH_CALUDE_geometric_sequence_properties_l3051_305156

/-- Given a geometric sequence with common ratio q > 0 and T_n as the product of the first n terms,
    if T_7 > T_6 > T_8, then 0 < q < 1 and T_13 > 1 > T_14 -/
theorem geometric_sequence_properties (q : ℝ) (T : ℕ → ℝ) 
    (h_q_pos : q > 0)
    (h_T : ∀ n, T n = (T 1) * q^(n-1))
    (h_ineq : T 7 > T 6 ∧ T 6 > T 8) :
    (0 < q ∧ q < 1) ∧ (T 13 > 1 ∧ 1 > T 14) := by
  sorry


end NUMINAMATH_CALUDE_geometric_sequence_properties_l3051_305156


namespace NUMINAMATH_CALUDE_min_value_theorem_min_value_achievable_l3051_305193

theorem min_value_theorem (x : ℝ) : 
  (x^2 + 9) / Real.sqrt (x^2 + 3) ≥ 2 * Real.sqrt 6 := by
  sorry

theorem min_value_achievable : 
  ∃ x : ℝ, (x^2 + 9) / Real.sqrt (x^2 + 3) = 2 * Real.sqrt 6 := by
  sorry

end NUMINAMATH_CALUDE_min_value_theorem_min_value_achievable_l3051_305193


namespace NUMINAMATH_CALUDE_odds_to_probability_losing_l3051_305106

-- Define the odds of winning
def odds_winning : ℚ := 5 / 6

-- Define the probability of losing
def prob_losing : ℚ := 6 / 11

-- Theorem statement
theorem odds_to_probability_losing : 
  odds_winning = 5 / 6 → prob_losing = 6 / 11 := by
  sorry

end NUMINAMATH_CALUDE_odds_to_probability_losing_l3051_305106


namespace NUMINAMATH_CALUDE_average_temperature_l3051_305123

def temperatures : List ℝ := [55, 59, 60, 57, 64]

theorem average_temperature : 
  (temperatures.sum / temperatures.length : ℝ) = 59.0 := by
  sorry

end NUMINAMATH_CALUDE_average_temperature_l3051_305123


namespace NUMINAMATH_CALUDE_romeo_chocolate_profit_l3051_305143

/-- Calculates the profit for Romeo's chocolate business -/
theorem romeo_chocolate_profit :
  let total_revenue : ℕ := 340
  let chocolate_cost : ℕ := 175
  let packaging_cost : ℕ := 60
  let advertising_cost : ℕ := 20
  let total_cost : ℕ := chocolate_cost + packaging_cost + advertising_cost
  let profit : ℕ := total_revenue - total_cost
  profit = 85 := by sorry

end NUMINAMATH_CALUDE_romeo_chocolate_profit_l3051_305143


namespace NUMINAMATH_CALUDE_suji_age_problem_l3051_305139

theorem suji_age_problem (abi_age suji_age : ℕ) : 
  (abi_age : ℚ) / suji_age = 5 / 4 →
  (abi_age + 3 : ℚ) / (suji_age + 3) = 11 / 9 →
  suji_age = 24 := by
sorry

end NUMINAMATH_CALUDE_suji_age_problem_l3051_305139


namespace NUMINAMATH_CALUDE_robert_reading_capacity_l3051_305101

/-- The number of books Robert can read given his reading speed, book length, and available time -/
def books_read (pages_per_hour : ℕ) (pages_per_book : ℕ) (available_hours : ℕ) : ℕ :=
  (pages_per_hour * available_hours) / pages_per_book

/-- Theorem stating that Robert can read 2 books in 8 hours -/
theorem robert_reading_capacity :
  books_read 100 400 8 = 2 := by
  sorry

#eval books_read 100 400 8

end NUMINAMATH_CALUDE_robert_reading_capacity_l3051_305101


namespace NUMINAMATH_CALUDE_volume_is_one_sixth_l3051_305160

-- Define the region
def region (x y z : ℝ) : Prop :=
  abs x + abs y + abs z ≤ 1 ∧ abs x + abs y + abs (z - 1) ≤ 1

-- Define the volume of the region
noncomputable def volume_of_region : ℝ := sorry

-- Theorem statement
theorem volume_is_one_sixth : volume_of_region = 1/6 := by sorry

end NUMINAMATH_CALUDE_volume_is_one_sixth_l3051_305160


namespace NUMINAMATH_CALUDE_ralph_received_eight_cards_l3051_305131

/-- The number of cards Ralph's father gave him -/
def cards_from_father (initial_cards final_cards : ℕ) : ℕ :=
  final_cards - initial_cards

/-- Proof that Ralph's father gave him 8 cards -/
theorem ralph_received_eight_cards :
  let initial_cards : ℕ := 4
  let final_cards : ℕ := 12
  cards_from_father initial_cards final_cards = 8 := by
  sorry

end NUMINAMATH_CALUDE_ralph_received_eight_cards_l3051_305131


namespace NUMINAMATH_CALUDE_sphere_surface_area_of_circumscribed_cube_l3051_305192

theorem sphere_surface_area_of_circumscribed_cube (R : ℝ) :
  let cube_edge_1 : ℝ := 2
  let cube_edge_2 : ℝ := 3
  let cube_edge_3 : ℝ := 1
  let cube_diagonal : ℝ := (cube_edge_1^2 + cube_edge_2^2 + cube_edge_3^2).sqrt
  R = cube_diagonal / 2 →
  4 * Real.pi * R^2 = 14 * Real.pi :=
by sorry

end NUMINAMATH_CALUDE_sphere_surface_area_of_circumscribed_cube_l3051_305192


namespace NUMINAMATH_CALUDE_triangle_angle_measure_l3051_305191

theorem triangle_angle_measure (D E F : ℝ) : 
  D = 88 →
  E = 4 * F + 20 →
  D + E + F = 180 →
  F = 14.4 := by
sorry

end NUMINAMATH_CALUDE_triangle_angle_measure_l3051_305191


namespace NUMINAMATH_CALUDE_residue_negative_811_mod_24_l3051_305137

theorem residue_negative_811_mod_24 : Int.mod (-811) 24 = 5 := by
  sorry

end NUMINAMATH_CALUDE_residue_negative_811_mod_24_l3051_305137


namespace NUMINAMATH_CALUDE_infinite_series_sum_l3051_305188

/-- The sum of the infinite series ∑_{k=1}^∞ (k^2 / 3^k) is equal to 7/8 -/
theorem infinite_series_sum : 
  ∑' k : ℕ+, (k : ℝ)^2 / 3^(k : ℝ) = 7/8 := by sorry

end NUMINAMATH_CALUDE_infinite_series_sum_l3051_305188


namespace NUMINAMATH_CALUDE_pete_age_triple_son_l3051_305149

/-- 
Given:
- Pete's current age is 35
- Pete's son's current age is 9

Prove that in 4 years, Pete will be exactly three times older than his son.
-/
theorem pete_age_triple_son (pete_age : ℕ) (son_age : ℕ) : 
  pete_age = 35 → son_age = 9 → 
  ∃ (years : ℕ), years = 4 ∧ pete_age + years = 3 * (son_age + years) :=
by sorry

end NUMINAMATH_CALUDE_pete_age_triple_son_l3051_305149
