import Mathlib

namespace NUMINAMATH_CALUDE_carl_personal_share_l560_56096

/-- Carl's car accident costs and insurance coverage -/
structure AccidentCost where
  propertyDamage : ℝ
  medicalBills : ℝ
  insuranceCoverage : ℝ

/-- Calculate Carl's personal share of the accident costs -/
def calculatePersonalShare (cost : AccidentCost) : ℝ :=
  (cost.propertyDamage + cost.medicalBills) * (1 - cost.insuranceCoverage)

/-- Theorem stating that Carl's personal share is $22,000 -/
theorem carl_personal_share :
  let cost : AccidentCost := {
    propertyDamage := 40000,
    medicalBills := 70000,
    insuranceCoverage := 0.8
  }
  calculatePersonalShare cost = 22000 := by
  sorry


end NUMINAMATH_CALUDE_carl_personal_share_l560_56096


namespace NUMINAMATH_CALUDE_corridor_lights_l560_56099

/-- The number of ways to choose k non-adjacent items from n consecutive items -/
def nonAdjacentChoices (n : ℕ) (k : ℕ) : ℕ :=
  Nat.choose (n - k + 1) k

/-- Theorem: There are 20 ways to choose 3 non-adjacent positions from 8 consecutive positions -/
theorem corridor_lights : nonAdjacentChoices 8 3 = 20 := by
  sorry

#eval nonAdjacentChoices 8 3

end NUMINAMATH_CALUDE_corridor_lights_l560_56099


namespace NUMINAMATH_CALUDE_debt_payment_additional_amount_l560_56067

theorem debt_payment_additional_amount 
  (total_installments : ℕ)
  (first_payment_count : ℕ)
  (remaining_payment_count : ℕ)
  (first_payment_amount : ℚ)
  (average_payment : ℚ)
  (h1 : total_installments = 52)
  (h2 : first_payment_count = 12)
  (h3 : remaining_payment_count = total_installments - first_payment_count)
  (h4 : first_payment_amount = 410)
  (h5 : average_payment = 460) :
  let additional_amount := (total_installments * average_payment - 
    first_payment_count * first_payment_amount) / remaining_payment_count - 
    first_payment_amount
  additional_amount = 65 := by sorry

end NUMINAMATH_CALUDE_debt_payment_additional_amount_l560_56067


namespace NUMINAMATH_CALUDE_simple_interest_rate_calculation_l560_56053

theorem simple_interest_rate_calculation (P A T : ℝ) (h1 : P = 750) (h2 : A = 950) (h3 : T = 5) :
  let SI := A - P
  let R := (SI * 100) / (P * T)
  R = (200 * 100) / (750 * 5) := by sorry

end NUMINAMATH_CALUDE_simple_interest_rate_calculation_l560_56053


namespace NUMINAMATH_CALUDE_product_sequence_sum_l560_56061

theorem product_sequence_sum (a b : ℕ) (h1 : a / 3 = 16) (h2 : b = a - 1) : a + b = 95 := by
  sorry

end NUMINAMATH_CALUDE_product_sequence_sum_l560_56061


namespace NUMINAMATH_CALUDE_candy_distribution_proof_l560_56035

def distribute_candies (total_candies : ℕ) (num_boxes : ℕ) : ℕ :=
  sorry

theorem candy_distribution_proof :
  distribute_candies 10 5 = 7 :=
by sorry

end NUMINAMATH_CALUDE_candy_distribution_proof_l560_56035


namespace NUMINAMATH_CALUDE_henry_earnings_l560_56069

/-- Henry's lawn mowing earnings calculation -/
theorem henry_earnings (rate : ℕ) (total_lawns : ℕ) (forgotten_lawns : ℕ) :
  rate = 5 → total_lawns = 12 → forgotten_lawns = 7 →
  (total_lawns - forgotten_lawns) * rate = 25 :=
by sorry

end NUMINAMATH_CALUDE_henry_earnings_l560_56069


namespace NUMINAMATH_CALUDE_expression_evaluation_l560_56014

theorem expression_evaluation :
  let x : ℝ := 2
  let y : ℝ := -3
  let z : ℝ := 1
  x^2 + y^2 - z^2 + 2*x*y + 3*z = 0 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l560_56014


namespace NUMINAMATH_CALUDE_triangle_problem_l560_56085

theorem triangle_problem (A B C : Real) (a b c : Real) : 
  -- Given conditions
  (a = 2) →
  (b = Real.sqrt 6) →
  (B = 60 * π / 180) →
  -- Triangle properties
  (A + B + C = π) →
  (a / Real.sin A = b / Real.sin B) →
  (b / Real.sin B = c / Real.sin C) →
  -- Conclusions
  (A = 45 * π / 180 ∧ 
   C = 75 * π / 180 ∧ 
   c = 1 + Real.sqrt 3) :=
by sorry

end NUMINAMATH_CALUDE_triangle_problem_l560_56085


namespace NUMINAMATH_CALUDE_complex_roots_magnitude_l560_56092

theorem complex_roots_magnitude (p : ℝ) (x₁ x₂ : ℂ) : 
  (∀ x : ℂ, x^2 + p*x + 4 = 0 ↔ x = x₁ ∨ x = x₂) →
  x₁ ≠ x₂ →
  Complex.abs x₁ = 2 := by
  sorry

end NUMINAMATH_CALUDE_complex_roots_magnitude_l560_56092


namespace NUMINAMATH_CALUDE_triangle_probability_theorem_l560_56043

noncomputable def triangle_probability (XY : ℝ) (angle_XYZ : ℝ) : ℝ :=
  (Real.sqrt 3 - Real.sqrt 2) / Real.sqrt 3

theorem triangle_probability_theorem (XY : ℝ) (angle_XYZ : ℝ) :
  XY = 12 →
  angle_XYZ = π / 6 →
  triangle_probability XY angle_XYZ = (Real.sqrt 3 - Real.sqrt 2) / Real.sqrt 3 :=
by sorry

end NUMINAMATH_CALUDE_triangle_probability_theorem_l560_56043


namespace NUMINAMATH_CALUDE_triangle_probability_l560_56073

def stick_lengths : List ℕ := [3, 4, 6, 8, 10, 12, 15, 18]

def can_form_triangle (a b c : ℕ) : Prop :=
  a + b > c ∧ a + c > b ∧ b + c > a

def valid_triangle_combinations : List (ℕ × ℕ × ℕ) :=
  [(4, 6, 8), (6, 8, 10), (8, 10, 12), (10, 12, 15)]

def total_combinations : ℕ := Nat.choose 8 3

theorem triangle_probability :
  (List.length valid_triangle_combinations : ℚ) / total_combinations = 1 / 14 := by
  sorry

end NUMINAMATH_CALUDE_triangle_probability_l560_56073


namespace NUMINAMATH_CALUDE_super_vcd_cost_price_l560_56078

theorem super_vcd_cost_price (x : ℝ) : 
  x * (1 + 0.4) * 0.9 - 50 = x + 340 → x = 1500 := by sorry

end NUMINAMATH_CALUDE_super_vcd_cost_price_l560_56078


namespace NUMINAMATH_CALUDE_cupcake_ratio_l560_56098

theorem cupcake_ratio (total : ℕ) (gluten_free : ℕ) (vegan : ℕ) (non_vegan_gluten : ℕ) :
  total = 80 →
  gluten_free = total / 2 →
  vegan = 24 →
  non_vegan_gluten = 28 →
  (gluten_free - non_vegan_gluten) / vegan = 1 / 2 := by
sorry

end NUMINAMATH_CALUDE_cupcake_ratio_l560_56098


namespace NUMINAMATH_CALUDE_arden_cricket_club_members_l560_56059

/-- The cost of a pair of gloves in dollars -/
def glove_cost : ℕ := 6

/-- The additional cost of a cap compared to a pair of gloves in dollars -/
def cap_additional_cost : ℕ := 8

/-- The total expenditure of the club in dollars -/
def total_expenditure : ℕ := 4140

/-- The number of gloves and caps each member needs -/
def items_per_member : ℕ := 2

theorem arden_cricket_club_members :
  ∃ (n : ℕ), n * (items_per_member * (glove_cost + (glove_cost + cap_additional_cost))) = total_expenditure ∧
  n = 103 := by
  sorry

end NUMINAMATH_CALUDE_arden_cricket_club_members_l560_56059


namespace NUMINAMATH_CALUDE_f_of_2_eq_neg_2_l560_56081

-- Define the function f
def f (x : ℝ) : ℝ := x^2 - 3*x

-- State the theorem
theorem f_of_2_eq_neg_2 : f 2 = -2 := by
  sorry

end NUMINAMATH_CALUDE_f_of_2_eq_neg_2_l560_56081


namespace NUMINAMATH_CALUDE_sum_even_implies_diff_even_l560_56027

theorem sum_even_implies_diff_even (a b : ℤ) : 
  Even (a + b) → Even (a - b) := by
  sorry

end NUMINAMATH_CALUDE_sum_even_implies_diff_even_l560_56027


namespace NUMINAMATH_CALUDE_train_length_is_415_l560_56044

/-- Represents the problem of calculating a train's length -/
def TrainProblem (speed : ℝ) (tunnelLength : ℝ) (time : ℝ) : Prop :=
  let speedMPS := speed * 1000 / 3600
  let totalDistance := speedMPS * time
  totalDistance = tunnelLength + 415

/-- Theorem stating that given the conditions, the train length is 415 meters -/
theorem train_length_is_415 :
  TrainProblem 63 285 40 := by
  sorry

#check train_length_is_415

end NUMINAMATH_CALUDE_train_length_is_415_l560_56044


namespace NUMINAMATH_CALUDE_segment_ratio_l560_56050

/-- Given a line segment GH with points E and F on it, where GE is 3 times EH and GF is 4 times FH,
    prove that EF is 1/20 of GH. -/
theorem segment_ratio (G E F H : Real) (GH EF : Real) : 
  E ∈ Set.Icc G H → 
  F ∈ Set.Icc G H → 
  G - E = 3 * (H - E) → 
  G - F = 4 * (H - F) → 
  GH = G - H → 
  EF = E - F → 
  EF = (1 / 20) * GH := by
sorry

end NUMINAMATH_CALUDE_segment_ratio_l560_56050


namespace NUMINAMATH_CALUDE_min_sum_polygons_l560_56040

theorem min_sum_polygons (m n : ℕ) : 
  m ≥ 1 → n ≥ 3 → 
  (180 * m * n - 360 * m) % 8 = 0 → 
  ∀ (m' n' : ℕ), m' ≥ 1 → n' ≥ 3 → (180 * m' * n' - 360 * m') % 8 = 0 → 
  m + n ≤ m' + n' :=
by sorry

end NUMINAMATH_CALUDE_min_sum_polygons_l560_56040


namespace NUMINAMATH_CALUDE_unpainted_area_triangular_board_l560_56064

/-- The area of the unpainted region on a triangular board that intersects with a rectangular board -/
theorem unpainted_area_triangular_board (base height width intersection_angle : ℝ) 
  (h_base : base = 8)
  (h_height : height = 10)
  (h_width : width = 5)
  (h_angle : intersection_angle = 45) :
  base * height / 2 - width * height = 50 := by
  sorry

end NUMINAMATH_CALUDE_unpainted_area_triangular_board_l560_56064


namespace NUMINAMATH_CALUDE_hanyoung_weight_l560_56052

theorem hanyoung_weight (hanyoung joohyung : ℝ) 
  (h1 : hanyoung = joohyung - 4)
  (h2 : hanyoung + joohyung = 88) : 
  hanyoung = 42 := by
sorry

end NUMINAMATH_CALUDE_hanyoung_weight_l560_56052


namespace NUMINAMATH_CALUDE_root_exists_in_interval_l560_56034

def f (x : ℝ) := x^2 + 3*x - 5

theorem root_exists_in_interval :
  ∃ x : ℝ, 1.1 < x ∧ x < 1.2 ∧ f x = 0 :=
by
  have h1 : f 1.1 < 0 := by sorry
  have h2 : f 1.2 > 0 := by sorry
  sorry

#check root_exists_in_interval

end NUMINAMATH_CALUDE_root_exists_in_interval_l560_56034


namespace NUMINAMATH_CALUDE_pizza_toppings_l560_56007

theorem pizza_toppings (total_slices : ℕ) (pepperoni_slices : ℕ) (mushroom_slices : ℕ) 
  (h1 : total_slices = 24)
  (h2 : pepperoni_slices = 15)
  (h3 : mushroom_slices = 16)
  (h4 : ∀ (slice : ℕ), slice < total_slices → (slice < pepperoni_slices ∨ slice < mushroom_slices)) :
  pepperoni_slices + mushroom_slices - total_slices = 7 :=
by sorry

end NUMINAMATH_CALUDE_pizza_toppings_l560_56007


namespace NUMINAMATH_CALUDE_intersection_of_specific_sets_l560_56008

theorem intersection_of_specific_sets :
  let A : Set ℤ := {1, 2, -3}
  let B : Set ℤ := {1, -4, 5}
  A ∩ B = {1} := by
sorry

end NUMINAMATH_CALUDE_intersection_of_specific_sets_l560_56008


namespace NUMINAMATH_CALUDE_apple_cost_l560_56042

/-- The cost of an apple and an orange given two price combinations -/
theorem apple_cost (apple orange : ℝ) 
  (h1 : 6 * apple + 3 * orange = 1.77)
  (h2 : 2 * apple + 5 * orange = 1.27) :
  apple = 0.21 := by
sorry

end NUMINAMATH_CALUDE_apple_cost_l560_56042


namespace NUMINAMATH_CALUDE_quadratic_two_distinct_roots_l560_56033

/-- A quadratic function f(x) = kx^2 - 4x - 2 -/
def f (k : ℝ) (x : ℝ) : ℝ := k * x^2 - 4 * x - 2

/-- The discriminant of the quadratic function f(x) = kx^2 - 4x - 2 -/
def discriminant (k : ℝ) : ℝ := 16 + 8 * k

theorem quadratic_two_distinct_roots (k : ℝ) :
  (∃ x y : ℝ, x ≠ y ∧ f k x = 0 ∧ f k y = 0) ↔ k > -2 ∧ k ≠ 0 :=
sorry

end NUMINAMATH_CALUDE_quadratic_two_distinct_roots_l560_56033


namespace NUMINAMATH_CALUDE_rug_area_theorem_l560_56025

/-- Given three overlapping rugs, calculates their combined area -/
def combined_rug_area (total_floor_area two_layer_area three_layer_area : ℝ) : ℝ :=
  let one_layer_area := total_floor_area - two_layer_area - three_layer_area
  one_layer_area + 2 * two_layer_area + 3 * three_layer_area

/-- Theorem stating that the combined area of three rugs is 200 square meters
    given the specified overlapping conditions -/
theorem rug_area_theorem :
  combined_rug_area 138 24 19 = 200 := by
  sorry


end NUMINAMATH_CALUDE_rug_area_theorem_l560_56025


namespace NUMINAMATH_CALUDE_werewolf_identity_l560_56017

-- Define the inhabitants
inductive Inhabitant : Type
| A : Inhabitant
| B : Inhabitant
| C : Inhabitant

-- Define the possible states
inductive State
| Knight : State
| Liar : State
| Werewolf : State

def is_knight (i : Inhabitant) (state : Inhabitant → State) : Prop :=
  state i = State.Knight

def is_liar (i : Inhabitant) (state : Inhabitant → State) : Prop :=
  state i = State.Liar

def is_werewolf (i : Inhabitant) (state : Inhabitant → State) : Prop :=
  state i = State.Werewolf

-- A's statement: At least one of us is a knight
def A_statement (state : Inhabitant → State) : Prop :=
  ∃ i : Inhabitant, is_knight i state

-- B's statement: At least one of us is a liar
def B_statement (state : Inhabitant → State) : Prop :=
  ∃ i : Inhabitant, is_liar i state

-- Theorem to prove
theorem werewolf_identity (state : Inhabitant → State) :
  -- At least one is a werewolf
  (∃ i : Inhabitant, is_werewolf i state) →
  -- None are both knight and werewolf
  (∀ i : Inhabitant, ¬(is_knight i state ∧ is_werewolf i state)) →
  -- A's statement is true if A is a knight, false if A is a liar
  ((is_knight Inhabitant.A state → A_statement state) ∧
   (is_liar Inhabitant.A state → ¬A_statement state)) →
  -- B's statement is true if B is a knight, false if B is a liar
  ((is_knight Inhabitant.B state → B_statement state) ∧
   (is_liar Inhabitant.B state → ¬B_statement state)) →
  -- C is the werewolf
  is_werewolf Inhabitant.C state :=
by sorry

end NUMINAMATH_CALUDE_werewolf_identity_l560_56017


namespace NUMINAMATH_CALUDE_largest_n_unique_k_l560_56005

theorem largest_n_unique_k : ∃ (n : ℕ), n > 0 ∧ n ≤ 27 ∧
  (∃! (k : ℤ), (9:ℚ)/17 < (n:ℚ)/(n+k) ∧ (n:ℚ)/(n+k) < 8/15) ∧
  (∀ (m : ℕ), m > 27 → ¬(∃! (k : ℤ), (9:ℚ)/17 < (m:ℚ)/(m+k) ∧ (m:ℚ)/(m+k) < 8/15)) :=
by sorry

end NUMINAMATH_CALUDE_largest_n_unique_k_l560_56005


namespace NUMINAMATH_CALUDE_square_area_ratio_after_doubling_l560_56084

theorem square_area_ratio_after_doubling (s : ℝ) (h : s > 0) :
  (s^2) / ((2*s)^2) = 1/4 := by
sorry

end NUMINAMATH_CALUDE_square_area_ratio_after_doubling_l560_56084


namespace NUMINAMATH_CALUDE_no_multiple_of_five_l560_56015

theorem no_multiple_of_five : ∀ C : ℕ, C < 10 → ¬(∃ k : ℕ, 200 + 10 * C + 4 = 5 * k) := by
  sorry

end NUMINAMATH_CALUDE_no_multiple_of_five_l560_56015


namespace NUMINAMATH_CALUDE_percentage_increase_l560_56026

theorem percentage_increase (original_earnings new_earnings : ℝ) 
  (h1 : original_earnings = 55)
  (h2 : new_earnings = 60) :
  ((new_earnings - original_earnings) / original_earnings) * 100 = 9.09 := by
  sorry

end NUMINAMATH_CALUDE_percentage_increase_l560_56026


namespace NUMINAMATH_CALUDE_annulus_area_l560_56055

/-- The area of an annulus formed by two concentric circles -/
theorem annulus_area (R r t : ℝ) (h1 : R > r) (h2 : R^2 = r^2 + t^2) : 
  π * R^2 - π * r^2 = π * t^2 := by sorry

end NUMINAMATH_CALUDE_annulus_area_l560_56055


namespace NUMINAMATH_CALUDE_same_side_of_line_l560_56047

/-- Given a line x + y = a, if the origin (0, 0) and the point (1, 1) are on the same side of this line, then a < 0 or a > 2. -/
theorem same_side_of_line (a : ℝ) : 
  (0 + 0 - a) * (1 + 1 - a) > 0 → a < 0 ∨ a > 2 := by
  sorry

end NUMINAMATH_CALUDE_same_side_of_line_l560_56047


namespace NUMINAMATH_CALUDE_intersection_equality_condition_l560_56062

theorem intersection_equality_condition (M N P : Set α) :
  (M = N → M ∩ P = N ∩ P) ∧
  ∃ M N P : Set ℕ, (M ∩ P = N ∩ P) ∧ (M ≠ N) := by
  sorry

end NUMINAMATH_CALUDE_intersection_equality_condition_l560_56062


namespace NUMINAMATH_CALUDE_exponential_system_solution_l560_56019

theorem exponential_system_solution (x y : ℝ) : 
  (4 : ℝ)^x = 256^(y + 1) → (27 : ℝ)^y = 3^(x - 2) → x = -4 ∧ y = -2 := by
  sorry

end NUMINAMATH_CALUDE_exponential_system_solution_l560_56019


namespace NUMINAMATH_CALUDE_min_value_expression_min_value_achieved_l560_56039

theorem min_value_expression (x : ℝ) : (x + 2) * (x + 3) * (x + 4) * (x + 5) + 2024 ≥ 2023 :=
sorry

theorem min_value_achieved : ∃ x : ℝ, (x + 2) * (x + 3) * (x + 4) * (x + 5) + 2024 = 2023 :=
sorry

end NUMINAMATH_CALUDE_min_value_expression_min_value_achieved_l560_56039


namespace NUMINAMATH_CALUDE_dice_product_div_eight_prob_l560_56091

/-- Represents a standard 6-sided die --/
def Die : Type := Fin 6

/-- The probability space of rolling 8 dice --/
def DiceRoll : Type := Fin 8 → Die

/-- A function that determines if a number is divisible by 8 --/
def divisible_by_eight (n : ℕ) : Prop := n % 8 = 0

/-- The product of the numbers shown on the dice --/
def dice_product (roll : DiceRoll) : ℕ :=
  (List.range 8).foldl (λ acc i => acc * (roll i).val.succ) 1

/-- The event that the product of the dice roll is divisible by 8 --/
def event_divisible_by_eight (roll : DiceRoll) : Prop :=
  divisible_by_eight (dice_product roll)

/-- The probability measure on the dice roll space --/
axiom prob : (DiceRoll → Prop) → ℚ

/-- The probability of the event is well-defined --/
axiom prob_well_defined : ∀ (E : DiceRoll → Prop), 0 ≤ prob E ∧ prob E ≤ 1

theorem dice_product_div_eight_prob :
  prob event_divisible_by_eight = 199 / 256 := by
  sorry

end NUMINAMATH_CALUDE_dice_product_div_eight_prob_l560_56091


namespace NUMINAMATH_CALUDE_floor_equation_solution_l560_56010

theorem floor_equation_solution (x : ℝ) : 
  ⌊⌊3*x⌋ - 1/3⌋ = ⌊x + 3⌋ ↔ 4/3 ≤ x ∧ x < 5/3 :=
sorry

end NUMINAMATH_CALUDE_floor_equation_solution_l560_56010


namespace NUMINAMATH_CALUDE_no_threefold_decreasing_number_l560_56048

theorem no_threefold_decreasing_number : ¬∃ (a b c : ℕ), 
  (1 ≤ a ∧ a ≤ 9) ∧ 
  (0 ≤ b ∧ b ≤ 9) ∧ 
  (0 ≤ c ∧ c ≤ 9) ∧ 
  (100 * a + 10 * b + c = 3 * (100 * b + 10 * c + a)) := by
  sorry

end NUMINAMATH_CALUDE_no_threefold_decreasing_number_l560_56048


namespace NUMINAMATH_CALUDE_discount_reduction_l560_56075

theorem discount_reduction (original_price : ℝ) (h : original_price > 0) :
  let first_discount := 0.3
  let second_discount := 0.2
  let remaining_after_first := 1 - first_discount
  let remaining_after_second := 1 - second_discount
  let final_price := original_price * remaining_after_first * remaining_after_second
  (original_price - final_price) / original_price = 0.44 :=
by
  sorry

end NUMINAMATH_CALUDE_discount_reduction_l560_56075


namespace NUMINAMATH_CALUDE_intersection_of_sets_l560_56070

theorem intersection_of_sets : 
  let A : Set ℤ := {1, 2, 3}
  let B : Set ℤ := {-2, 2}
  A ∩ B = {2} := by
sorry

end NUMINAMATH_CALUDE_intersection_of_sets_l560_56070


namespace NUMINAMATH_CALUDE_video_game_points_sum_l560_56032

theorem video_game_points_sum : 
  let paul_points : ℕ := 3103
  let cousin_points : ℕ := 2713
  paul_points + cousin_points = 5816 :=
by sorry

end NUMINAMATH_CALUDE_video_game_points_sum_l560_56032


namespace NUMINAMATH_CALUDE_quadratic_negative_root_l560_56072

theorem quadratic_negative_root (a : ℝ) :
  (∃ x : ℝ, x < 0 ∧ a * x^2 + 4 * x + 1 = 0) ↔ a ≤ 4 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_negative_root_l560_56072


namespace NUMINAMATH_CALUDE_quadratic_equation_roots_l560_56060

theorem quadratic_equation_roots (k : ℝ) :
  let f : ℝ → ℝ := λ x => x^2 + (2*k + 1)*x + k^2 + 1
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ f x₁ = 0 ∧ f x₂ = 0) →
  (k > 3/4 ∧ (∀ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ f x₁ = 0 ∧ f x₂ = 0 ∧ x₁ * x₂ = 5 → k = 2)) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_roots_l560_56060


namespace NUMINAMATH_CALUDE_last_year_ticket_cost_l560_56021

/-- 
Proves that the ticket cost last year was $85, given that this year's cost 
is $102 and represents a 20% increase from last year.
-/
theorem last_year_ticket_cost : 
  ∀ (last_year_cost : ℝ), 
  (last_year_cost + 0.2 * last_year_cost = 102) → 
  last_year_cost = 85 := by
sorry

end NUMINAMATH_CALUDE_last_year_ticket_cost_l560_56021


namespace NUMINAMATH_CALUDE_line_relationships_l560_56023

/-- Two lines in the 2D plane -/
structure TwoLines where
  l1 : ℝ → ℝ → ℝ → ℝ  -- (2a+1)x+(a+2)y+3=0
  l2 : ℝ → ℝ → ℝ → ℝ  -- (a-1)x-2y+2=0

/-- Definition of parallel lines -/
def parallel (lines : TwoLines) (a : ℝ) : Prop :=
  ∀ x y, lines.l1 a x y = 0 ↔ lines.l2 a x y = 0

/-- Definition of perpendicular lines -/
def perpendicular (lines : TwoLines) (a : ℝ) : Prop :=
  ∀ x1 y1 x2 y2, lines.l1 a x1 y1 = 0 ∧ lines.l2 a x2 y2 = 0 →
    (x2 - x1) * ((2 * a + 1) * (x2 - x1) + (a + 2) * (y2 - y1)) +
    (y2 - y1) * ((a - 1) * (x2 - x1) - 2 * (y2 - y1)) = 0

/-- The main theorem -/
theorem line_relationships (lines : TwoLines) :
  (∀ a, parallel lines a ↔ a = 0) ∧
  (∀ a, perpendicular lines a ↔ a = -1 ∨ a = 5/2) := by
  sorry

#check line_relationships

end NUMINAMATH_CALUDE_line_relationships_l560_56023


namespace NUMINAMATH_CALUDE_admission_cutoff_score_l560_56090

theorem admission_cutoff_score (
  admitted_fraction : Real)
  (admitted_avg_diff : Real)
  (not_admitted_avg_diff : Real)
  (overall_avg : Real)
  (h1 : admitted_fraction = 2 / 5)
  (h2 : admitted_avg_diff = 15)
  (h3 : not_admitted_avg_diff = -20)
  (h4 : overall_avg = 90) :
  let cutoff_score := 
    (overall_avg - admitted_fraction * admitted_avg_diff - (1 - admitted_fraction) * not_admitted_avg_diff) /
    (admitted_fraction + (1 - admitted_fraction))
  cutoff_score = 96 := by
  sorry

end NUMINAMATH_CALUDE_admission_cutoff_score_l560_56090


namespace NUMINAMATH_CALUDE_intersection_sum_l560_56012

/-- Given two equations and their intersection points, prove the sum of x-coordinates is 0 and the sum of y-coordinates is 3 -/
theorem intersection_sum (x₁ x₂ x₃ y₁ y₂ y₃ : ℝ) : 
  (y₁ = x₁^3 - 3*x₁ + 2) →
  (y₂ = x₂^3 - 3*x₂ + 2) →
  (y₃ = x₃^3 - 3*x₃ + 2) →
  (2*x₁ + 3*y₁ = 3) →
  (2*x₂ + 3*y₂ = 3) →
  (2*x₃ + 3*y₃ = 3) →
  (x₁ + x₂ + x₃ = 0 ∧ y₁ + y₂ + y₃ = 3) :=
by sorry

end NUMINAMATH_CALUDE_intersection_sum_l560_56012


namespace NUMINAMATH_CALUDE_soup_can_price_l560_56006

/-- Calculates the normal price of a can of soup given a "buy 1 get one free" offer -/
theorem soup_can_price (total_cans : ℕ) (total_paid : ℚ) : 
  total_cans > 0 → total_paid > 0 → (total_paid / (total_cans / 2 : ℚ) = 0.60) := by
  sorry

end NUMINAMATH_CALUDE_soup_can_price_l560_56006


namespace NUMINAMATH_CALUDE_remainder_problem_l560_56076

theorem remainder_problem (d : ℕ) (h1 : d = 170) (h2 : d ∣ 690) (h3 : d ∣ 875) 
  (h4 : 875 % d = 25) (h5 : ∀ k : ℕ, k > d → ¬(k ∣ 690 ∧ k ∣ 875)) : 
  690 % d = 10 := by
  sorry

end NUMINAMATH_CALUDE_remainder_problem_l560_56076


namespace NUMINAMATH_CALUDE_count_words_beginning_ending_with_A_l560_56041

/-- The number of letters in the alphabet --/
def alphabet_size : ℕ := 26

/-- The length of the words we're considering --/
def word_length : ℕ := 5

/-- The number of variable positions in the word --/
def variable_positions : ℕ := word_length - 2

/-- The number of five-letter words beginning and ending with 'A' --/
def words_beginning_ending_with_A : ℕ := alphabet_size ^ variable_positions

theorem count_words_beginning_ending_with_A :
  words_beginning_ending_with_A = 17576 :=
sorry

end NUMINAMATH_CALUDE_count_words_beginning_ending_with_A_l560_56041


namespace NUMINAMATH_CALUDE_area_FJGH_area_FJGH_proof_l560_56029

/-- Represents a parallelogram EFGH with point J on side EH -/
structure Parallelogram where
  /-- Length of side EH -/
  eh : ℝ
  /-- Length of JH -/
  jh : ℝ
  /-- Height of the parallelogram from FG to EH -/
  height : ℝ
  /-- Condition that EH = 12 -/
  eh_eq : eh = 12
  /-- Condition that JH = 8 -/
  jh_eq : jh = 8
  /-- Condition that the height is 10 -/
  height_eq : height = 10

/-- The area of region FJGH in the parallelogram is 100 -/
theorem area_FJGH (p : Parallelogram) : ℝ :=
  100

#check area_FJGH

/-- Proof of the theorem -/
theorem area_FJGH_proof (p : Parallelogram) : area_FJGH p = 100 := by
  sorry

end NUMINAMATH_CALUDE_area_FJGH_area_FJGH_proof_l560_56029


namespace NUMINAMATH_CALUDE_prime_pair_divisibility_l560_56087

theorem prime_pair_divisibility (p q : ℕ) : 
  Nat.Prime p → Nat.Prime q →
  (p^p + q^q + 1 ≡ 0 [MOD pq] ↔ (p = 2 ∧ q = 5) ∨ (p = 5 ∧ q = 2)) := by
  sorry

end NUMINAMATH_CALUDE_prime_pair_divisibility_l560_56087


namespace NUMINAMATH_CALUDE_jack_weight_jack_weight_proof_l560_56093

/-- Jack and Anna's see-saw problem -/
theorem jack_weight (anna_weight : ℕ) (num_rocks : ℕ) (rock_weight : ℕ) : ℕ :=
  let total_rock_weight := num_rocks * rock_weight
  let jack_weight := anna_weight - total_rock_weight
  jack_weight

/-- Proof of Jack's weight -/
theorem jack_weight_proof :
  jack_weight 40 5 4 = 20 := by
  sorry

end NUMINAMATH_CALUDE_jack_weight_jack_weight_proof_l560_56093


namespace NUMINAMATH_CALUDE_second_section_has_180_cars_l560_56046

-- Define the given information
def section_g_rows : ℕ := 15
def section_g_cars_per_row : ℕ := 10
def second_section_rows : ℕ := 20
def nate_cars_per_minute : ℕ := 11
def nate_search_time : ℕ := 30

-- Define the total number of cars Nate walked past
def total_cars_walked : ℕ := nate_cars_per_minute * nate_search_time

-- Define the number of cars in Section G
def section_g_cars : ℕ := section_g_rows * section_g_cars_per_row

-- Define the number of cars in the second section
def second_section_cars : ℕ := total_cars_walked - section_g_cars

-- Theorem to prove
theorem second_section_has_180_cars :
  second_section_cars = 180 :=
sorry

end NUMINAMATH_CALUDE_second_section_has_180_cars_l560_56046


namespace NUMINAMATH_CALUDE_relation_between_x_and_y_l560_56003

theorem relation_between_x_and_y (p : ℝ) :
  let x : ℝ := 3 + 3^p
  let y : ℝ := 3 + 3^(-p)
  y = (3*x - 8) / (x - 3) :=
by sorry

end NUMINAMATH_CALUDE_relation_between_x_and_y_l560_56003


namespace NUMINAMATH_CALUDE_smallest_multiple_of_45_and_75_not_20_l560_56001

theorem smallest_multiple_of_45_and_75_not_20 :
  ∃ n : ℕ, n > 0 ∧ 45 ∣ n ∧ 75 ∣ n ∧ ¬(20 ∣ n) ∧
  ∀ m : ℕ, m > 0 ∧ 45 ∣ m ∧ 75 ∣ m ∧ ¬(20 ∣ m) → n ≤ m :=
by sorry

end NUMINAMATH_CALUDE_smallest_multiple_of_45_and_75_not_20_l560_56001


namespace NUMINAMATH_CALUDE_g_negative_in_range_l560_56097

def f (a x : ℝ) : ℝ := x^3 + 3*a*x - 1

def g (a x : ℝ) : ℝ := (3*x^2 + 3*a) - a*x - 5

theorem g_negative_in_range :
  ∀ x : ℝ, -2/3 < x → x < 1 →
    ∀ a : ℝ, -1 ≤ a → a ≤ 1 →
      g a x < 0 :=
by sorry

end NUMINAMATH_CALUDE_g_negative_in_range_l560_56097


namespace NUMINAMATH_CALUDE_snow_leopard_arrangement_l560_56071

theorem snow_leopard_arrangement (n : ℕ) (k : ℕ) (h1 : n = 9) (h2 : k = 3) :
  (k.factorial) * ((n - k).factorial) = 4320 := by
  sorry

end NUMINAMATH_CALUDE_snow_leopard_arrangement_l560_56071


namespace NUMINAMATH_CALUDE_sqrt_sqrt_81_l560_56011

theorem sqrt_sqrt_81 : Real.sqrt (Real.sqrt 81) = 3 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_sqrt_81_l560_56011


namespace NUMINAMATH_CALUDE_heather_emily_weight_difference_l560_56058

/-- Given the weights of Heather and Emily, prove that Heather is 78 pounds heavier than Emily. -/
theorem heather_emily_weight_difference :
  let heather_weight : ℕ := 87
  let emily_weight : ℕ := 9
  heather_weight - emily_weight = 78 := by sorry

end NUMINAMATH_CALUDE_heather_emily_weight_difference_l560_56058


namespace NUMINAMATH_CALUDE_largest_number_with_digits_4_1_sum_14_l560_56028

def is_valid_number (n : ℕ) : Prop :=
  ∀ d : ℕ, d ∈ n.digits 10 → d = 4 ∨ d = 1

def sum_of_digits (n : ℕ) : ℕ :=
  (n.digits 10).sum

theorem largest_number_with_digits_4_1_sum_14 :
  ∀ n : ℕ, is_valid_number n ∧ sum_of_digits n = 14 → n ≤ 4411 :=
by sorry

end NUMINAMATH_CALUDE_largest_number_with_digits_4_1_sum_14_l560_56028


namespace NUMINAMATH_CALUDE_find_y_value_l560_56018

theorem find_y_value (x y z : ℤ) 
  (eq1 : x + y + z = 355)
  (eq2 : x - y = 200)
  (eq3 : x + z = 500) :
  y = -145 := by
  sorry

end NUMINAMATH_CALUDE_find_y_value_l560_56018


namespace NUMINAMATH_CALUDE_dave_baseball_cards_pages_l560_56080

/-- The number of pages needed to organize baseball cards in a binder -/
def pages_needed (cards_per_page new_cards old_cards : ℕ) : ℕ :=
  (new_cards + old_cards + cards_per_page - 1) / cards_per_page

/-- Proof that Dave needs 2 pages to organize his baseball cards -/
theorem dave_baseball_cards_pages :
  pages_needed 8 3 13 = 2 := by
  sorry

end NUMINAMATH_CALUDE_dave_baseball_cards_pages_l560_56080


namespace NUMINAMATH_CALUDE_marys_income_percentage_l560_56004

theorem marys_income_percentage (juan tim mary : ℝ) 
  (h1 : tim = juan * 0.7)
  (h2 : mary = tim * 1.6) :
  mary = juan * 1.12 := by
  sorry

end NUMINAMATH_CALUDE_marys_income_percentage_l560_56004


namespace NUMINAMATH_CALUDE_part_one_part_two_l560_56063

-- Define the sets A and B
def A : Set ℝ := {x | (x - 3) / (2 * x - 1) ≤ 0}
def B (a : ℝ) : Set ℝ := {x | x^2 - a < 0}

-- Part (I)
theorem part_one :
  (A ∩ B 4 = {x | 1/2 < x ∧ x < 2}) ∧
  (A ∪ B 4 = {x | -2 < x ∧ x ≤ 3}) := by sorry

-- Part (II)
theorem part_two :
  (∀ a, B a ∩ (Set.univ \ A) = B a) →
  {a | a ≤ 1/4} = Set.Iic (1/4) := by sorry

end NUMINAMATH_CALUDE_part_one_part_two_l560_56063


namespace NUMINAMATH_CALUDE_max_k_logarithm_inequality_l560_56077

theorem max_k_logarithm_inequality (x₀ x₁ x₂ x₃ : ℝ) 
  (h₀ : x₀ > x₁) (h₁ : x₁ > x₂) (h₂ : x₂ > x₃) (h₃ : x₃ > 0) :
  let log_base (b a : ℝ) := Real.log a / Real.log b
  9 * log_base (x₀ / x₃) 1993 ≤ 
    log_base (x₀ / x₁) 1993 + log_base (x₁ / x₂) 1993 + log_base (x₂ / x₃) 1993 ∧
  ∀ k > 9, ∃ x₀' x₁' x₂' x₃' : ℝ, x₀' > x₁' ∧ x₁' > x₂' ∧ x₂' > x₃' ∧ x₃' > 0 ∧
    k * log_base (x₀' / x₃') 1993 > 
      log_base (x₀' / x₁') 1993 + log_base (x₁' / x₂') 1993 + log_base (x₂' / x₃') 1993 :=
by sorry

end NUMINAMATH_CALUDE_max_k_logarithm_inequality_l560_56077


namespace NUMINAMATH_CALUDE_final_lives_calculation_l560_56086

def calculate_final_lives (initial_lives lives_lost gain_factor : ℕ) : ℕ :=
  initial_lives - lives_lost + gain_factor * lives_lost

theorem final_lives_calculation (initial_lives lives_lost gain_factor : ℕ) :
  calculate_final_lives initial_lives lives_lost gain_factor =
  initial_lives - lives_lost + gain_factor * lives_lost :=
by
  sorry

-- Example usage
example : calculate_final_lives 75 28 3 = 131 :=
by
  sorry

end NUMINAMATH_CALUDE_final_lives_calculation_l560_56086


namespace NUMINAMATH_CALUDE_completing_square_equivalence_l560_56038

theorem completing_square_equivalence :
  ∀ x : ℝ, x^2 - 2*x = 9 ↔ (x - 1)^2 = 10 :=
by sorry

end NUMINAMATH_CALUDE_completing_square_equivalence_l560_56038


namespace NUMINAMATH_CALUDE_outfit_combinations_l560_56065

theorem outfit_combinations (shirts : ℕ) (pants : ℕ) (ties : ℕ) (belts : ℕ) 
  (h_shirts : shirts = 8)
  (h_pants : pants = 5)
  (h_ties : ties = 4)
  (h_belts : belts = 2) :
  shirts * pants * (ties + 1) * (belts + 1) = 600 := by
  sorry

end NUMINAMATH_CALUDE_outfit_combinations_l560_56065


namespace NUMINAMATH_CALUDE_x_coordinate_difference_at_y_20_l560_56020

/-- A line in a 2D coordinate system --/
structure Line where
  slope : ℚ
  y_intercept : ℚ

def Line.through_points (x1 y1 x2 y2 : ℚ) : Line where
  slope := (y2 - y1) / (x2 - x1)
  y_intercept := y1 - ((y2 - y1) / (x2 - x1)) * x1

def Line.x_at_y (l : Line) (y : ℚ) : ℚ :=
  (y - l.y_intercept) / l.slope

theorem x_coordinate_difference_at_y_20 :
  let l := Line.through_points 0 6 3 0
  let m := Line.through_points 0 3 8 0
  let x_l := l.x_at_y 20
  let x_m := m.x_at_y 20
  |x_l - x_m| = 115 / 3 := by
  sorry

end NUMINAMATH_CALUDE_x_coordinate_difference_at_y_20_l560_56020


namespace NUMINAMATH_CALUDE_games_purchased_l560_56066

theorem games_purchased (initial_amount : ℕ) (spent_amount : ℕ) (game_cost : ℕ) : 
  initial_amount = 104 → spent_amount = 41 → game_cost = 9 →
  (initial_amount - spent_amount) / game_cost = 7 := by
  sorry

end NUMINAMATH_CALUDE_games_purchased_l560_56066


namespace NUMINAMATH_CALUDE_map_to_actual_distance_l560_56082

/-- Given a map scale and a road length on the map, calculate the actual road length in kilometers. -/
theorem map_to_actual_distance (scale : ℚ) (map_length : ℚ) (actual_length : ℚ) : 
  scale = 1 / 50000 →
  map_length = 15 →
  actual_length = 7.5 →
  scale * actual_length = map_length := by
  sorry

#check map_to_actual_distance

end NUMINAMATH_CALUDE_map_to_actual_distance_l560_56082


namespace NUMINAMATH_CALUDE_green_balloons_count_l560_56000

theorem green_balloons_count (total : ℕ) (red : ℕ) (green : ℕ) : 
  total = 17 → red = 8 → total = red + green → green = 9 := by
  sorry

end NUMINAMATH_CALUDE_green_balloons_count_l560_56000


namespace NUMINAMATH_CALUDE_double_plus_five_difference_l560_56049

theorem double_plus_five_difference (x : ℝ) (h : x = 4) : 2 * x + 5 - x / 2 = 11 := by
  sorry

end NUMINAMATH_CALUDE_double_plus_five_difference_l560_56049


namespace NUMINAMATH_CALUDE_square_area_not_covered_by_circle_l560_56083

theorem square_area_not_covered_by_circle (d : ℝ) (h : d = 8) :
  let r := d / 2
  let square_area := d^2
  let circle_area := π * r^2
  square_area - circle_area = 64 - 16 * π := by
  sorry

end NUMINAMATH_CALUDE_square_area_not_covered_by_circle_l560_56083


namespace NUMINAMATH_CALUDE_total_degrees_theorem_l560_56068

/-- Represents the budget allocation percentages for different sectors -/
structure BudgetAllocation where
  microphotonics : Float
  homeElectronics : Float
  foodAdditives : Float
  geneticallyModifiedMicroorganisms : Float
  industrialLubricants : Float
  artificialIntelligence : Float
  nanotechnology : Float

/-- Calculates the degrees in a circle graph for a given percentage -/
def percentageToDegrees (percentage : Float) : Float :=
  percentage * 3.6

/-- Calculates the total degrees for basic astrophysics, artificial intelligence, and nanotechnology -/
def totalDegrees (allocation : BudgetAllocation) : Float :=
  let basicAstrophysics := 100 - (allocation.microphotonics + allocation.homeElectronics + 
    allocation.foodAdditives + allocation.geneticallyModifiedMicroorganisms + 
    allocation.industrialLubricants + allocation.artificialIntelligence + allocation.nanotechnology)
  percentageToDegrees basicAstrophysics + 
  percentageToDegrees allocation.artificialIntelligence + 
  percentageToDegrees allocation.nanotechnology

/-- Theorem: The total degrees for basic astrophysics, artificial intelligence, and nanotechnology is 117.36 -/
theorem total_degrees_theorem (allocation : BudgetAllocation) 
  (h1 : allocation.microphotonics = 12.3)
  (h2 : allocation.homeElectronics = 17.8)
  (h3 : allocation.foodAdditives = 9.4)
  (h4 : allocation.geneticallyModifiedMicroorganisms = 21.7)
  (h5 : allocation.industrialLubricants = 6.2)
  (h6 : allocation.artificialIntelligence = 4.1)
  (h7 : allocation.nanotechnology = 5.3) :
  totalDegrees allocation = 117.36 := by
  sorry

end NUMINAMATH_CALUDE_total_degrees_theorem_l560_56068


namespace NUMINAMATH_CALUDE_tourist_meeting_time_l560_56036

/-- Represents a tourist -/
structure Tourist where
  name : String

/-- Represents a meeting between two tourists -/
structure Meeting where
  tourist1 : Tourist
  tourist2 : Tourist
  time : ℕ  -- Time in hours after noon

/-- The problem setup -/
def tourist_problem (vitya pasha katya masha : Tourist) : Prop :=
  ∃ (vitya_masha vitya_katya pasha_masha pasha_katya : Meeting),
    -- Meetings
    vitya_masha.tourist1 = vitya ∧ vitya_masha.tourist2 = masha ∧ vitya_masha.time = 0 ∧
    vitya_katya.tourist1 = vitya ∧ vitya_katya.tourist2 = katya ∧ vitya_katya.time = 2 ∧
    pasha_masha.tourist1 = pasha ∧ pasha_masha.tourist2 = masha ∧ pasha_masha.time = 3 ∧
    -- Vitya and Pasha travel at the same speed from A to B
    (vitya_masha.time - vitya_katya.time = pasha_masha.time - pasha_katya.time) ∧
    -- Katya and Masha travel at the same speed from B to A
    (vitya_masha.time - pasha_masha.time = vitya_katya.time - pasha_katya.time) →
    pasha_katya.time = 5

theorem tourist_meeting_time (vitya pasha katya masha : Tourist) :
  tourist_problem vitya pasha katya masha := by
  sorry

#check tourist_meeting_time

end NUMINAMATH_CALUDE_tourist_meeting_time_l560_56036


namespace NUMINAMATH_CALUDE_height_tiles_count_l560_56088

def shower_tiles (num_walls : ℕ) (width_tiles : ℕ) (total_tiles : ℕ) : ℕ :=
  total_tiles / (num_walls * width_tiles)

theorem height_tiles_count : shower_tiles 3 8 480 = 20 := by
  sorry

end NUMINAMATH_CALUDE_height_tiles_count_l560_56088


namespace NUMINAMATH_CALUDE_cross_product_perpendicular_l560_56095

def v1 : ℝ × ℝ × ℝ := (4, 3, -5)
def v2 : ℝ × ℝ × ℝ := (2, -1, 4)

def cross_product (a b : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  let (a₁, a₂, a₃) := a
  let (b₁, b₂, b₃) := b
  (a₂ * b₃ - a₃ * b₂, a₃ * b₁ - a₁ * b₃, a₁ * b₂ - a₂ * b₁)

def dot_product (a b : ℝ × ℝ × ℝ) : ℝ :=
  let (a₁, a₂, a₃) := a
  let (b₁, b₂, b₃) := b
  a₁ * b₁ + a₂ * b₂ + a₃ * b₃

theorem cross_product_perpendicular :
  let result := cross_product v1 v2
  result = (7, -26, -10) ∧
  dot_product v1 result = 0 ∧
  dot_product v2 result = 0 := by
  sorry

end NUMINAMATH_CALUDE_cross_product_perpendicular_l560_56095


namespace NUMINAMATH_CALUDE_min_value_problem_l560_56031

theorem min_value_problem (x y a : ℝ) (hx : x > 0) (hy : y > 0) (ha : a > 0)
  (h1 : 2 * x + y = 1)
  (h2 : ∀ x' y' : ℝ, x' > 0 → y' > 0 → 2 * x' + y' = 1 → a / x' + 1 / y' ≥ 9)
  (h3 : ∃ x' y' : ℝ, x' > 0 ∧ y' > 0 ∧ 2 * x' + y' = 1 ∧ a / x' + 1 / y' = 9) :
  a = 2 := by
sorry

end NUMINAMATH_CALUDE_min_value_problem_l560_56031


namespace NUMINAMATH_CALUDE_bryden_receives_correct_amount_l560_56074

/-- The amount Bryden receives for selling state quarters -/
def bryden_receive (num_quarters : ℕ) (face_value : ℚ) (collector_offer_percent : ℕ) : ℚ :=
  num_quarters * face_value * (collector_offer_percent : ℚ) / 100

/-- Theorem stating that Bryden receives $31.25 for selling five state quarters -/
theorem bryden_receives_correct_amount :
  bryden_receive 5 (1/4) 2500 = 125/4 :=
sorry

end NUMINAMATH_CALUDE_bryden_receives_correct_amount_l560_56074


namespace NUMINAMATH_CALUDE_starting_lineup_combinations_l560_56056

def team_size : ℕ := 12
def offensive_linemen : ℕ := 4
def positions : ℕ := 5

theorem starting_lineup_combinations :
  (offensive_linemen) *
  (team_size - 1) *
  (team_size - 2) *
  (team_size - 3) *
  (team_size - 4) = 31680 :=
by sorry

end NUMINAMATH_CALUDE_starting_lineup_combinations_l560_56056


namespace NUMINAMATH_CALUDE_new_man_weight_l560_56054

/-- Given a group of 8 men, if replacing a 60 kg man with a new man increases the average weight by 1 kg, then the new man weighs 68 kg. -/
theorem new_man_weight (initial_average : ℝ) : 
  (8 * initial_average + 68 = 8 * (initial_average + 1) + 60) → 68 = 68 := by
  sorry

end NUMINAMATH_CALUDE_new_man_weight_l560_56054


namespace NUMINAMATH_CALUDE_black_pens_count_l560_56057

theorem black_pens_count (green_pens red_pens : ℕ) 
  (prob_neither_red_nor_green : ℚ) :
  green_pens = 5 →
  red_pens = 7 →
  prob_neither_red_nor_green = 1/3 →
  ∃ (total_pens black_pens : ℕ),
    total_pens = green_pens + red_pens + black_pens ∧
    (black_pens : ℚ) / total_pens = prob_neither_red_nor_green ∧
    black_pens = 6 :=
by sorry

end NUMINAMATH_CALUDE_black_pens_count_l560_56057


namespace NUMINAMATH_CALUDE_host_horse_speed_calculation_l560_56009

/-- The daily travel distance of the guest's horse in li -/
def guest_horse_speed : ℚ := 300

/-- The fraction of the day that passes before the host realizes the guest left without clothes -/
def realization_time : ℚ := 1/3

/-- The fraction of the day that has passed when the host returns home -/
def return_time : ℚ := 3/4

/-- The daily travel distance of the host's horse in li -/
def host_horse_speed : ℚ := 780

theorem host_horse_speed_calculation :
  let catch_up_time : ℚ := return_time - realization_time
  let guest_travel_time : ℚ := realization_time + catch_up_time
  2 * guest_horse_speed * guest_travel_time = host_horse_speed * catch_up_time :=
by sorry

end NUMINAMATH_CALUDE_host_horse_speed_calculation_l560_56009


namespace NUMINAMATH_CALUDE_expression_simplification_l560_56051

theorem expression_simplification (a b : ℝ) (h1 : a = Real.sqrt 3 - 3) (h2 : b = 3) :
  1 - (a - b) / (a + 2*b) / ((a^2 - b^2) / (a^2 + 4*a*b + 4*b^2)) = -Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l560_56051


namespace NUMINAMATH_CALUDE_juice_theorem_l560_56037

def juice_problem (tom_initial jerry_initial : ℚ) 
  (drink_fraction transfer_fraction : ℚ) (final_transfer : ℚ) : Prop :=
  let tom_after_drinking := tom_initial * (1 - drink_fraction)
  let jerry_after_drinking := jerry_initial * (1 - drink_fraction)
  let jerry_transfer := jerry_after_drinking * transfer_fraction
  let tom_before_final := tom_after_drinking + jerry_transfer
  let jerry_before_final := jerry_after_drinking - jerry_transfer
  let tom_final := tom_before_final - final_transfer
  let jerry_final := jerry_before_final + final_transfer
  (jerry_initial = 2 * tom_initial) ∧
  (tom_final = jerry_final + 4) ∧
  (tom_initial + jerry_initial - (tom_final + jerry_final) = 80)

theorem juice_theorem : 
  juice_problem 40 80 (2/3) (1/4) 5 := by sorry

end NUMINAMATH_CALUDE_juice_theorem_l560_56037


namespace NUMINAMATH_CALUDE_expression_evaluation_l560_56089

theorem expression_evaluation (a b c : ℚ) 
  (h1 : c = a + 8)
  (h2 : b = a + 4)
  (h3 : a = 5)
  (h4 : a + 2 ≠ 0)
  (h5 : b - 3 ≠ 0)
  (h6 : c + 7 ≠ 0) :
  (a + 3) / (a + 2) * (b - 2) / (b - 3) * (c + 10) / (c + 7) = 23 / 15 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l560_56089


namespace NUMINAMATH_CALUDE_vertical_angles_equal_parallel_lines_corresponding_angles_equal_l560_56016

-- Define the concept of an angle
def Angle : Type := ℝ

-- Define the concept of a line
def Line : Type := Unit

-- Define vertical angles
def are_vertical (a b : Angle) : Prop := sorry

-- Define parallel lines
def are_parallel (l1 l2 : Line) : Prop := sorry

-- Define corresponding angles
def are_corresponding (a b : Angle) (l1 l2 : Line) : Prop := sorry

-- Theorem: Vertical angles are equal
theorem vertical_angles_equal (a b : Angle) : 
  are_vertical a b → a = b := by sorry

-- Theorem: If two lines are parallel, then corresponding angles are equal
theorem parallel_lines_corresponding_angles_equal (a b : Angle) (l1 l2 : Line) :
  are_parallel l1 l2 → are_corresponding a b l1 l2 → a = b := by sorry

end NUMINAMATH_CALUDE_vertical_angles_equal_parallel_lines_corresponding_angles_equal_l560_56016


namespace NUMINAMATH_CALUDE_expression_value_l560_56002

theorem expression_value (a b c : ℝ) (h : a^2 + b = b^2 + c ∧ b^2 + c = c^2 + a) :
  a * (a^2 - b^2) + b * (b^2 - c^2) + c * (c^2 - a^2) = 0 := by
  sorry

end NUMINAMATH_CALUDE_expression_value_l560_56002


namespace NUMINAMATH_CALUDE_rectangular_prism_diagonal_l560_56022

theorem rectangular_prism_diagonal (a b c : ℝ) (ha : a = 12) (hb : b = 15) (hc : c = 8) :
  Real.sqrt (a^2 + b^2 + c^2) = Real.sqrt 433 := by
  sorry

end NUMINAMATH_CALUDE_rectangular_prism_diagonal_l560_56022


namespace NUMINAMATH_CALUDE_number_in_scientific_notation_l560_56094

/-- Scientific notation representation of a positive real number -/
structure ScientificNotation where
  coefficient : ℝ
  exponent : ℤ
  coeff_range : 1 ≤ coefficient ∧ coefficient < 10

/-- Function to convert a positive real number to scientific notation -/
def toScientificNotation (x : ℝ) : ScientificNotation :=
  sorry

theorem number_in_scientific_notation :
  toScientificNotation 36600 = ScientificNotation.mk 3.66 4 (by sorry) :=
sorry

end NUMINAMATH_CALUDE_number_in_scientific_notation_l560_56094


namespace NUMINAMATH_CALUDE_select_representatives_l560_56024

theorem select_representatives (boys girls total reps : ℕ) 
  (h1 : boys = 6)
  (h2 : girls = 4)
  (h3 : total = boys + girls)
  (h4 : reps = 3) :
  (Nat.choose total reps) - (Nat.choose boys reps) = 100 := by
  sorry

end NUMINAMATH_CALUDE_select_representatives_l560_56024


namespace NUMINAMATH_CALUDE_noah_age_after_10_years_l560_56079

def joe_age : ℕ := 6
def noah_age : ℕ := 2 * joe_age
def years_passed : ℕ := 10

theorem noah_age_after_10_years :
  noah_age + years_passed = 22 := by
  sorry

end NUMINAMATH_CALUDE_noah_age_after_10_years_l560_56079


namespace NUMINAMATH_CALUDE_parallel_line_equation_l560_56030

/-- Given a point A and a line L, this theorem proves that the equation
    4x + y - 14 = 0 represents the line passing through A and parallel to L. -/
theorem parallel_line_equation (A : ℝ × ℝ) (L : Set (ℝ × ℝ)) : 
  A.1 = 3 ∧ A.2 = 2 →
  L = {(x, y) | 4 * x + y - 2 = 0} →
  {(x, y) | 4 * x + y - 14 = 0} = 
    {(x, y) | ∃ (t : ℝ), x = A.1 + t ∧ y = A.2 - 4 * t} :=
by sorry

end NUMINAMATH_CALUDE_parallel_line_equation_l560_56030


namespace NUMINAMATH_CALUDE_sequence_zero_at_201_l560_56013

/-- Sequence defined by the given recurrence relation -/
def a : ℕ → ℚ
  | 0 => 134
  | 1 => 150
  | (k + 2) => a k - (k + 1) / a (k + 1)

/-- The smallest positive integer n for which a_n = 0 -/
def n : ℕ := 201

/-- Theorem stating that a_n = 0 and n is the smallest such positive integer -/
theorem sequence_zero_at_201 :
  a n = 0 ∧ ∀ m : ℕ, m > 0 ∧ m < n → a m ≠ 0 := by sorry

end NUMINAMATH_CALUDE_sequence_zero_at_201_l560_56013


namespace NUMINAMATH_CALUDE_largest_solution_of_equation_l560_56045

theorem largest_solution_of_equation (x : ℝ) : 
  (6 * (12 * x^2 + 12 * x + 11) = x * (12 * x - 44)) → x ≤ -1 :=
by sorry

end NUMINAMATH_CALUDE_largest_solution_of_equation_l560_56045
