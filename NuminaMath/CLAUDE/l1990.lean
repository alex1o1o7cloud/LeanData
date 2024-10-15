import Mathlib

namespace NUMINAMATH_CALUDE_inequality_solution_l1990_199044

theorem inequality_solution (x : ℝ) :
  x ≠ -3 ∧ x ≠ 4 →
  ((x - 3) / (x + 3) > (2 * x - 1) / (x - 4) ↔
   (x > -6 - 3 * Real.sqrt 17 ∧ x < -6 + 3 * Real.sqrt 17) ∨
   (x > -3 ∧ x < 4)) :=
by sorry

end NUMINAMATH_CALUDE_inequality_solution_l1990_199044


namespace NUMINAMATH_CALUDE_cubic_sum_problem_l1990_199011

theorem cubic_sum_problem (x y : ℝ) (h1 : x + y = 5) (h2 : x^2 + y^2 = 17) : x^3 + y^3 = 65 := by
  sorry

end NUMINAMATH_CALUDE_cubic_sum_problem_l1990_199011


namespace NUMINAMATH_CALUDE_tree_height_when_boy_is_36_inches_l1990_199088

/-- Calculates the final height of a tree given initial heights and growth rates -/
def final_tree_height (initial_tree_height : ℝ) (initial_boy_height : ℝ) (final_boy_height : ℝ) : ℝ :=
  initial_tree_height + 2 * (final_boy_height - initial_boy_height)

/-- Proves that the tree will be 40 inches tall when the boy is 36 inches tall -/
theorem tree_height_when_boy_is_36_inches :
  final_tree_height 16 24 36 = 40 := by
  sorry

end NUMINAMATH_CALUDE_tree_height_when_boy_is_36_inches_l1990_199088


namespace NUMINAMATH_CALUDE_remove_six_maximizes_probability_l1990_199018

def original_list : List Int := [-1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]

def remove_number (list : List Int) (n : Int) : List Int :=
  list.filter (· ≠ n)

def count_pairs_sum_12 (list : List Int) : Nat :=
  (list.filterMap (λ x => 
    if x < 12 ∧ list.contains (12 - x) ∧ x ≠ 12 - x
    then some (min x (12 - x))
    else none
  )).dedup.length

theorem remove_six_maximizes_probability : 
  ∀ n ∈ original_list, n ≠ 6 → 
    count_pairs_sum_12 (remove_number original_list 6) ≥ 
    count_pairs_sum_12 (remove_number original_list n) :=
by sorry

end NUMINAMATH_CALUDE_remove_six_maximizes_probability_l1990_199018


namespace NUMINAMATH_CALUDE_larger_tart_flour_usage_l1990_199085

theorem larger_tart_flour_usage
  (small_tarts : ℕ)
  (large_tarts : ℕ)
  (small_flour : ℚ)
  (h1 : small_tarts = 50)
  (h2 : large_tarts = 25)
  (h3 : small_flour = 1 / 8)
  (h4 : small_tarts * small_flour = large_tarts * large_flour) :
  large_flour = 1 / 4 :=
by
  sorry

#check larger_tart_flour_usage

end NUMINAMATH_CALUDE_larger_tart_flour_usage_l1990_199085


namespace NUMINAMATH_CALUDE_filling_pipe_time_calculation_l1990_199080

/-- The time it takes to fill the tank when both pipes are open -/
def both_pipes_time : ℝ := 180

/-- The time it takes for the emptying pipe to empty the tank -/
def emptying_pipe_time : ℝ := 45

/-- The time it takes for the filling pipe to fill the tank -/
def filling_pipe_time : ℝ := 36

theorem filling_pipe_time_calculation :
  (1 / filling_pipe_time) - (1 / emptying_pipe_time) = (1 / both_pipes_time) :=
sorry

end NUMINAMATH_CALUDE_filling_pipe_time_calculation_l1990_199080


namespace NUMINAMATH_CALUDE_ratio_chain_l1990_199032

theorem ratio_chain (a b c d e : ℚ) 
  (h1 : a / b = 3 / 4)
  (h2 : b / c = 7 / 9)
  (h3 : c / d = 5 / 7)
  (h4 : d / e = 11 / 13) :
  a / e = 165 / 468 := by
  sorry

end NUMINAMATH_CALUDE_ratio_chain_l1990_199032


namespace NUMINAMATH_CALUDE_prove_callys_colored_shirts_l1990_199012

/-- The number of colored shirts Cally washed -/
def callys_colored_shirts : ℕ := 5

theorem prove_callys_colored_shirts :
  let callys_other_clothes : ℕ := 10 + 7 + 6 -- white shirts + shorts + pants
  let dannys_clothes : ℕ := 6 + 8 + 10 + 6 -- white shirts + colored shirts + shorts + pants
  let total_clothes : ℕ := 58
  callys_colored_shirts = total_clothes - (callys_other_clothes + dannys_clothes) :=
by sorry

end NUMINAMATH_CALUDE_prove_callys_colored_shirts_l1990_199012


namespace NUMINAMATH_CALUDE_distribute_four_balls_four_boxes_l1990_199037

/-- The number of ways to distribute indistinguishable balls into distinguishable boxes -/
def distribute_balls (n : ℕ) (k : ℕ) : ℕ := Nat.choose (n + k - 1) (k - 1)

/-- Theorem: There are 35 ways to distribute 4 indistinguishable balls into 4 distinguishable boxes -/
theorem distribute_four_balls_four_boxes : distribute_balls 4 4 = 35 := by
  sorry

end NUMINAMATH_CALUDE_distribute_four_balls_four_boxes_l1990_199037


namespace NUMINAMATH_CALUDE_license_plate_combinations_l1990_199014

def letter_count : ℕ := 26
def digit_count : ℕ := 10
def letter_positions : ℕ := 4
def digit_positions : ℕ := 2

theorem license_plate_combinations : 
  (Nat.choose letter_count 2 * 2 * Nat.choose letter_positions 2 * digit_count ^ digit_positions) = 390000 := by
  sorry

end NUMINAMATH_CALUDE_license_plate_combinations_l1990_199014


namespace NUMINAMATH_CALUDE_expenditure_is_negative_l1990_199000

/-- Represents the recording of a monetary transaction -/
inductive MonetaryRecord
| Income (amount : ℤ)
| Expenditure (amount : ℤ)

/-- Converts a MonetaryRecord to its signed integer representation -/
def toSignedAmount (record : MonetaryRecord) : ℤ :=
  match record with
  | MonetaryRecord.Income a => a
  | MonetaryRecord.Expenditure a => -a

theorem expenditure_is_negative (income_amount expenditure_amount : ℤ) 
  (h : toSignedAmount (MonetaryRecord.Income income_amount) = income_amount) :
  toSignedAmount (MonetaryRecord.Expenditure expenditure_amount) = -expenditure_amount := by
  sorry

end NUMINAMATH_CALUDE_expenditure_is_negative_l1990_199000


namespace NUMINAMATH_CALUDE_inequality_proof_l1990_199047

theorem inequality_proof (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a ≠ b) :
  (2 * a * b) / (a + b) < Real.sqrt (a * b) ∧ Real.sqrt (a * b) < (a + b) / 2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l1990_199047


namespace NUMINAMATH_CALUDE_charlie_share_l1990_199082

def distribute_money (total : ℕ) (ratio1 ratio2 ratio3 : ℕ) (deduct1 deduct2 deduct3 : ℕ) : ℕ × ℕ × ℕ :=
  sorry

theorem charlie_share :
  let (alice, bond, charlie) := distribute_money 1105 11 18 24 10 20 15
  charlie = 495 := by sorry

end NUMINAMATH_CALUDE_charlie_share_l1990_199082


namespace NUMINAMATH_CALUDE_potato_cooking_time_l1990_199079

def cooking_problem (total_potatoes cooked_potatoes time_per_potato : ℕ) : Prop :=
  let remaining_potatoes := total_potatoes - cooked_potatoes
  remaining_potatoes * time_per_potato = 45

theorem potato_cooking_time :
  cooking_problem 16 7 5 := by
  sorry

end NUMINAMATH_CALUDE_potato_cooking_time_l1990_199079


namespace NUMINAMATH_CALUDE_final_cell_population_l1990_199048

/-- Represents the cell population growth over time -/
def cell_population (initial_cells : ℕ) (split_factor : ℕ) (days : ℕ) : ℕ :=
  initial_cells * split_factor ^ (days / 3)

/-- Theorem: Given the conditions, the final cell population after 9 days is 18 -/
theorem final_cell_population :
  cell_population 2 3 9 = 18 := by
  sorry

#eval cell_population 2 3 9

end NUMINAMATH_CALUDE_final_cell_population_l1990_199048


namespace NUMINAMATH_CALUDE_final_load_is_30600_l1990_199039

def initial_load : ℝ := 50000

def first_unload_percent : ℝ := 0.1
def second_unload_percent : ℝ := 0.2
def third_unload_percent : ℝ := 0.15

def remaining_after_first (load : ℝ) : ℝ :=
  load * (1 - first_unload_percent)

def remaining_after_second (load : ℝ) : ℝ :=
  load * (1 - second_unload_percent)

def remaining_after_third (load : ℝ) : ℝ :=
  load * (1 - third_unload_percent)

theorem final_load_is_30600 :
  remaining_after_third (remaining_after_second (remaining_after_first initial_load)) = 30600 := by
  sorry

end NUMINAMATH_CALUDE_final_load_is_30600_l1990_199039


namespace NUMINAMATH_CALUDE_captain_selection_count_l1990_199093

/-- The number of ways to choose k items from n items without regard to order -/
def choose (n k : ℕ) : ℕ := Nat.choose n k

/-- The number of people in the team -/
def team_size : ℕ := 15

/-- The number of captains to be chosen -/
def captain_count : ℕ := 4

/-- Theorem: The number of ways to choose 4 captains from a team of 15 people is 1365 -/
theorem captain_selection_count : choose team_size captain_count = 1365 := by
  sorry

end NUMINAMATH_CALUDE_captain_selection_count_l1990_199093


namespace NUMINAMATH_CALUDE_base8_to_base10_3206_l1990_199089

/-- Converts a base 8 number to base 10 --/
def base8_to_base10 (digits : List Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * (8 ^ i)) 0

/-- The base 8 representation of the number --/
def base8_num : List Nat := [6, 0, 2, 3]

/-- Theorem stating that the base 10 representation of 3206₈ is 1670 --/
theorem base8_to_base10_3206 : base8_to_base10 base8_num = 1670 := by
  sorry

end NUMINAMATH_CALUDE_base8_to_base10_3206_l1990_199089


namespace NUMINAMATH_CALUDE_variance_invariant_under_translation_l1990_199036

def variance (data : List ℝ) : ℝ := sorry

theorem variance_invariant_under_translation (data : List ℝ) (c : ℝ) :
  variance data = variance (data.map (λ x => x - c)) := by sorry

end NUMINAMATH_CALUDE_variance_invariant_under_translation_l1990_199036


namespace NUMINAMATH_CALUDE_fraction_comparison_l1990_199033

theorem fraction_comparison (a b c d : ℝ) 
  (h1 : a > b) (h2 : b > 0) (h3 : c < d) (h4 : d < 0) : 
  b / (a - c) < a / (b - d) := by
  sorry

end NUMINAMATH_CALUDE_fraction_comparison_l1990_199033


namespace NUMINAMATH_CALUDE_negation_of_existential_inequality_l1990_199071

theorem negation_of_existential_inequality :
  ¬(∃ x : ℝ, x^2 - 2*x + 1 < 0) ↔ (∀ x : ℝ, x^2 - 2*x + 1 ≥ 0) := by sorry

end NUMINAMATH_CALUDE_negation_of_existential_inequality_l1990_199071


namespace NUMINAMATH_CALUDE_triangle_angle_A_l1990_199027

theorem triangle_angle_A (A B C : ℝ) (a b : ℝ) (angleB : ℝ) : 
  a = Real.sqrt 3 →
  b = Real.sqrt 2 →
  angleB = π / 4 →
  (∃ (angleA : ℝ), (angleA = π / 3 ∨ angleA = 2 * π / 3) ∧ 
    a / Real.sin angleA = b / Real.sin angleB) :=
by sorry

end NUMINAMATH_CALUDE_triangle_angle_A_l1990_199027


namespace NUMINAMATH_CALUDE_total_profit_calculation_l1990_199042

/-- Represents a partner's investment information -/
structure PartnerInvestment where
  initial : ℝ
  monthly : ℝ

/-- Calculates the total annual investment for a partner -/
def annualInvestment (p : PartnerInvestment) : ℝ :=
  p.initial + 12 * p.monthly

/-- Represents the investment information for all partners -/
structure Investments where
  a : PartnerInvestment
  b : PartnerInvestment
  c : PartnerInvestment

/-- Calculates the total investment for all partners -/
def totalInvestment (inv : Investments) : ℝ :=
  annualInvestment inv.a + annualInvestment inv.b + annualInvestment inv.c

/-- The main theorem stating the total profit given the conditions -/
theorem total_profit_calculation (inv : Investments) 
    (h1 : inv.a = { initial := 45000, monthly := 1500 })
    (h2 : inv.b = { initial := 63000, monthly := 2100 })
    (h3 : inv.c = { initial := 72000, monthly := 2400 })
    (h4 : (annualInvestment inv.c / totalInvestment inv) * 60000 = 24000) :
    60000 = (totalInvestment inv * 24000) / (annualInvestment inv.c) := by
  sorry

end NUMINAMATH_CALUDE_total_profit_calculation_l1990_199042


namespace NUMINAMATH_CALUDE_sufficient_condition_transitivity_l1990_199008

theorem sufficient_condition_transitivity (p q r : Prop) :
  (p → q) → (q → r) → (p → r) := by
  sorry

end NUMINAMATH_CALUDE_sufficient_condition_transitivity_l1990_199008


namespace NUMINAMATH_CALUDE_probability_is_three_fifths_l1990_199004

-- Define the set S
def S : Finset ℤ := {-3, 0, 0, 4, 7, 8}

-- Define the function to check if a pair of integers has a product of 0
def productIsZero (x y : ℤ) : Bool :=
  x * y = 0

-- Define the probability calculation function
def probabilityOfZeroProduct (s : Finset ℤ) : ℚ :=
  let totalPairs := (s.card.choose 2 : ℚ)
  let zeroPairs := (s.filter (· = 0)).card * (s.filter (· ≠ 0)).card +
                   (if (s.filter (· = 0)).card ≥ 2 then 1 else 0)
  zeroPairs / totalPairs

-- State the theorem
theorem probability_is_three_fifths :
  probabilityOfZeroProduct S = 3/5 := by sorry

end NUMINAMATH_CALUDE_probability_is_three_fifths_l1990_199004


namespace NUMINAMATH_CALUDE_course_selection_combinations_l1990_199061

/-- The number of available courses -/
def num_courses : ℕ := 4

/-- The number of courses student A chooses -/
def courses_A : ℕ := 2

/-- The number of courses students B and C each choose -/
def courses_BC : ℕ := 3

/-- The total number of different possible combinations -/
def total_combinations : ℕ := Nat.choose num_courses courses_A * (Nat.choose num_courses courses_BC)^2

theorem course_selection_combinations :
  total_combinations = 96 :=
by sorry

end NUMINAMATH_CALUDE_course_selection_combinations_l1990_199061


namespace NUMINAMATH_CALUDE_second_wing_floors_is_seven_l1990_199067

/-- A hotel with two wings -/
structure Hotel where
  total_rooms : ℕ
  wing1_floors : ℕ
  wing1_halls_per_floor : ℕ
  wing1_rooms_per_hall : ℕ
  wing2_halls_per_floor : ℕ
  wing2_rooms_per_hall : ℕ

/-- Calculate the number of floors in the second wing -/
def second_wing_floors (h : Hotel) : ℕ :=
  let wing1_rooms := h.wing1_floors * h.wing1_halls_per_floor * h.wing1_rooms_per_hall
  let wing2_rooms := h.total_rooms - wing1_rooms
  let rooms_per_floor_wing2 := h.wing2_halls_per_floor * h.wing2_rooms_per_hall
  wing2_rooms / rooms_per_floor_wing2

/-- The theorem stating that the number of floors in the second wing is 7 -/
theorem second_wing_floors_is_seven (h : Hotel) 
    (h_total : h.total_rooms = 4248)
    (h_wing1_floors : h.wing1_floors = 9)
    (h_wing1_halls : h.wing1_halls_per_floor = 6)
    (h_wing1_rooms : h.wing1_rooms_per_hall = 32)
    (h_wing2_halls : h.wing2_halls_per_floor = 9)
    (h_wing2_rooms : h.wing2_rooms_per_hall = 40) : 
  second_wing_floors h = 7 := by
  sorry

#eval second_wing_floors {
  total_rooms := 4248,
  wing1_floors := 9,
  wing1_halls_per_floor := 6,
  wing1_rooms_per_hall := 32,
  wing2_halls_per_floor := 9,
  wing2_rooms_per_hall := 40
}

end NUMINAMATH_CALUDE_second_wing_floors_is_seven_l1990_199067


namespace NUMINAMATH_CALUDE_orange_cost_proof_l1990_199050

/-- Given that 5 dozen oranges cost $39.00, prove that 8 dozen oranges at the same rate cost $62.40 -/
theorem orange_cost_proof (cost_five_dozen : ℝ) (h1 : cost_five_dozen = 39) :
  let cost_per_dozen : ℝ := cost_five_dozen / 5
  let cost_eight_dozen : ℝ := 8 * cost_per_dozen
  cost_eight_dozen = 62.4 := by
  sorry

end NUMINAMATH_CALUDE_orange_cost_proof_l1990_199050


namespace NUMINAMATH_CALUDE_no_points_above_diagonal_l1990_199086

-- Define the triangle
def triangle : Set (ℝ × ℝ) :=
  {p | ∃ (t₁ t₂ : ℝ), 0 ≤ t₁ ∧ 0 ≤ t₂ ∧ t₁ + t₂ ≤ 1 ∧
    p = (4 * t₁ + 4 * t₂, 10 * t₂)}

-- Theorem statement
theorem no_points_above_diagonal (a b : ℝ) :
  (a, b) ∈ triangle → a - b ≤ 0 := by sorry

end NUMINAMATH_CALUDE_no_points_above_diagonal_l1990_199086


namespace NUMINAMATH_CALUDE_third_number_from_lcm_hcf_l1990_199095

/-- Given three positive integers with known LCM and HCF, prove the third number -/
theorem third_number_from_lcm_hcf (A B C : ℕ+) : 
  A = 36 → B = 44 → Nat.lcm A (Nat.lcm B C) = 792 → Nat.gcd A (Nat.gcd B C) = 12 → C = 6 := by
  sorry

end NUMINAMATH_CALUDE_third_number_from_lcm_hcf_l1990_199095


namespace NUMINAMATH_CALUDE_page_number_added_twice_l1990_199025

theorem page_number_added_twice (n : ℕ) : 
  (n * (n + 1) / 2 ≤ 1986) ∧ 
  ((n + 1) * (n + 2) / 2 > 1986) →
  1986 - (n * (n + 1) / 2) = 33 := by
sorry

end NUMINAMATH_CALUDE_page_number_added_twice_l1990_199025


namespace NUMINAMATH_CALUDE_alices_preferred_number_l1990_199045

def is_between (n : ℕ) (a b : ℕ) : Prop := a ≤ n ∧ n ≤ b

def sum_of_digits (n : ℕ) : ℕ :=
  let digits := n.digits 10
  digits.sum

theorem alices_preferred_number :
  ∃! n : ℕ,
    is_between n 100 200 ∧
    11 ∣ n ∧
    ¬(2 ∣ n) ∧
    3 ∣ sum_of_digits n ∧
    n = 165 :=
by sorry

end NUMINAMATH_CALUDE_alices_preferred_number_l1990_199045


namespace NUMINAMATH_CALUDE_sugar_water_concentration_l1990_199010

theorem sugar_water_concentration (a : ℝ) : 
  (100 * 0.4 + a * 0.2) / (100 + a) = 0.25 → a = 300 := by
  sorry

end NUMINAMATH_CALUDE_sugar_water_concentration_l1990_199010


namespace NUMINAMATH_CALUDE_tan_alpha_two_l1990_199007

theorem tan_alpha_two (α : Real) (h : Real.tan α = 2) :
  (Real.tan (α + π/4) = -3) ∧
  ((Real.sin (2*α)) / (Real.sin α ^ 2 * Real.cos α - Real.cos α ^ 2 - 1) = 1) := by
  sorry

end NUMINAMATH_CALUDE_tan_alpha_two_l1990_199007


namespace NUMINAMATH_CALUDE_evaluate_expression_l1990_199083

theorem evaluate_expression : 
  |7 - (8^2) * (3 - 12)| - |(5^3) - (Real.sqrt 11)^4| = 579 := by
  sorry

end NUMINAMATH_CALUDE_evaluate_expression_l1990_199083


namespace NUMINAMATH_CALUDE_problem_solution_l1990_199059

theorem problem_solution : (((3^1 : ℝ) + 2 + 6^2 + 3)⁻¹ * 6) = 3/22 := by sorry

end NUMINAMATH_CALUDE_problem_solution_l1990_199059


namespace NUMINAMATH_CALUDE_tylenol_dosage_l1990_199075

/-- Represents the dosage schedule and total amount of medication taken -/
structure DosageInfo where
  interval : ℕ  -- Time interval between doses in hours
  duration : ℕ  -- Total duration of medication in hours
  tablets_per_dose : ℕ  -- Number of tablets taken per dose
  total_grams : ℕ  -- Total amount of medication taken in grams

/-- Calculates the milligrams per tablet given dosage information -/
def milligrams_per_tablet (info : DosageInfo) : ℕ :=
  let total_milligrams := info.total_grams * 1000
  let num_doses := info.duration / info.interval
  let milligrams_per_dose := total_milligrams / num_doses
  milligrams_per_dose / info.tablets_per_dose

/-- Theorem stating that under the given conditions, each tablet contains 500 milligrams -/
theorem tylenol_dosage (info : DosageInfo) 
  (h1 : info.interval = 4)
  (h2 : info.duration = 12)
  (h3 : info.tablets_per_dose = 2)
  (h4 : info.total_grams = 3) :
  milligrams_per_tablet info = 500 := by
  sorry

end NUMINAMATH_CALUDE_tylenol_dosage_l1990_199075


namespace NUMINAMATH_CALUDE_initial_number_of_girls_l1990_199006

theorem initial_number_of_girls (initial_boys : ℕ) (boys_dropout : ℕ) (girls_dropout : ℕ) (remaining_students : ℕ) : 
  initial_boys = 14 →
  boys_dropout = 4 →
  girls_dropout = 3 →
  remaining_students = 17 →
  initial_boys - boys_dropout + (initial_girls - girls_dropout) = remaining_students →
  initial_girls = 10 :=
by
  sorry

#check initial_number_of_girls

end NUMINAMATH_CALUDE_initial_number_of_girls_l1990_199006


namespace NUMINAMATH_CALUDE_cos_120_degrees_l1990_199029

theorem cos_120_degrees : Real.cos (120 * π / 180) = -1/2 := by
  sorry

end NUMINAMATH_CALUDE_cos_120_degrees_l1990_199029


namespace NUMINAMATH_CALUDE_sports_suits_cost_price_l1990_199034

/-- The cost price of one set of type A sports suits -/
def cost_A : ℝ := 180

/-- The cost price of one set of type B sports suits -/
def cost_B : ℝ := 200

/-- The total cost of purchasing one set of each type -/
def total_cost : ℝ := 380

/-- The amount spent on type A sports suits -/
def amount_A : ℝ := 8100

/-- The amount spent on type B sports suits -/
def amount_B : ℝ := 9000

theorem sports_suits_cost_price :
  cost_A + cost_B = total_cost ∧
  amount_A / cost_A = amount_B / cost_B :=
by sorry

end NUMINAMATH_CALUDE_sports_suits_cost_price_l1990_199034


namespace NUMINAMATH_CALUDE_fraction_equality_l1990_199053

theorem fraction_equality : (4 + 5) / (7 + 5) = 3 / 4 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_l1990_199053


namespace NUMINAMATH_CALUDE_distance_to_origin_of_complex_fraction_l1990_199009

theorem distance_to_origin_of_complex_fraction : 
  let z : ℂ := 2 / (1 + Complex.I)
  Complex.abs z = Real.sqrt 2 := by
sorry

end NUMINAMATH_CALUDE_distance_to_origin_of_complex_fraction_l1990_199009


namespace NUMINAMATH_CALUDE_prob_shortest_diagonal_icosagon_l1990_199016

/-- A regular polygon with n sides -/
structure RegularPolygon where
  n : ℕ
  h : n ≥ 3

/-- The number of diagonals in a regular polygon -/
def num_diagonals (p : RegularPolygon) : ℕ := p.n * (p.n - 3) / 2

/-- The number of shortest diagonals in a regular polygon -/
def num_shortest_diagonals (p : RegularPolygon) : ℕ := p.n / 2

/-- An icosagon is a regular polygon with 20 sides -/
def icosagon : RegularPolygon where
  n := 20
  h := by norm_num

/-- The probability of selecting a shortest diagonal in an icosagon -/
def prob_shortest_diagonal (p : RegularPolygon) : ℚ :=
  (num_shortest_diagonals p : ℚ) / (num_diagonals p : ℚ)

theorem prob_shortest_diagonal_icosagon :
  prob_shortest_diagonal icosagon = 1 / 17 := by
  sorry

end NUMINAMATH_CALUDE_prob_shortest_diagonal_icosagon_l1990_199016


namespace NUMINAMATH_CALUDE_equal_intercept_line_equation_l1990_199019

-- Define a line passing through (2,5) with equal intercepts
structure EqualInterceptLine where
  -- The slope of the line
  slope : ℝ
  -- The y-intercept of the line
  y_intercept : ℝ
  -- Condition: Line passes through (2,5)
  point_condition : 5 = slope * 2 + y_intercept
  -- Condition: Equal intercepts on both axes
  equal_intercepts : y_intercept = slope * y_intercept

-- Theorem statement
theorem equal_intercept_line_equation (l : EqualInterceptLine) :
  (l.slope = 5/2 ∧ l.y_intercept = 0) ∨ (l.slope = -1 ∧ l.y_intercept = 7) :=
sorry

end NUMINAMATH_CALUDE_equal_intercept_line_equation_l1990_199019


namespace NUMINAMATH_CALUDE_consecutive_odd_integers_sum_l1990_199054

theorem consecutive_odd_integers_sum (x : ℤ) : 
  (∃ y : ℤ, y = x + 2 ∧ Odd x ∧ Odd y ∧ y = 5 * x - 2) → x + (x + 2) = 4 := by
  sorry

end NUMINAMATH_CALUDE_consecutive_odd_integers_sum_l1990_199054


namespace NUMINAMATH_CALUDE_enjoy_both_activities_l1990_199057

theorem enjoy_both_activities (total : ℕ) (reading : ℕ) (movies : ℕ) (neither : ℕ)
  (h1 : total = 50)
  (h2 : reading = 22)
  (h3 : movies = 20)
  (h4 : neither = 15) :
  total - neither - (reading + movies - (total - neither)) = 7 := by
  sorry

end NUMINAMATH_CALUDE_enjoy_both_activities_l1990_199057


namespace NUMINAMATH_CALUDE_partition_with_equal_product_l1990_199021

def numbers : List Nat := [2, 3, 12, 14, 15, 20, 21]

theorem partition_with_equal_product :
  ∃ (s₁ s₂ : List Nat),
    s₁ ∪ s₂ = numbers ∧
    s₁ ∩ s₂ = [] ∧
    s₁ ≠ [] ∧
    s₂ ≠ [] ∧
    (s₁.prod = 2520 ∧ s₂.prod = 2520) :=
  sorry

end NUMINAMATH_CALUDE_partition_with_equal_product_l1990_199021


namespace NUMINAMATH_CALUDE_office_distance_l1990_199094

/-- The distance to the office in kilometers -/
def distance : ℝ := sorry

/-- The time it takes to reach the office on time in hours -/
def on_time : ℝ := sorry

/-- Condition 1: At 10 kmph, the person arrives 10 minutes late -/
axiom condition_1 : distance = 10 * (on_time + 1/6)

/-- Condition 2: At 15 kmph, the person arrives 10 minutes early -/
axiom condition_2 : distance = 15 * (on_time - 1/6)

/-- Theorem: The distance to the office is 10 kilometers -/
theorem office_distance : distance = 10 := by sorry

end NUMINAMATH_CALUDE_office_distance_l1990_199094


namespace NUMINAMATH_CALUDE_sum_interior_angles_regular_polygon_l1990_199051

/-- For a regular polygon with exterior angles of 40 degrees, 
    the sum of interior angles is 1260 degrees. -/
theorem sum_interior_angles_regular_polygon : 
  ∀ n : ℕ, 
  n > 2 → 
  360 / n = 40 → 
  (n - 2) * 180 = 1260 :=
by
  sorry

end NUMINAMATH_CALUDE_sum_interior_angles_regular_polygon_l1990_199051


namespace NUMINAMATH_CALUDE_number_of_boys_l1990_199077

/-- Given conditions about men, women, and boys with their earnings, prove the number of boys --/
theorem number_of_boys (men women boys : ℕ) (total_earnings men_wage : ℕ) : 
  men = 5 →
  men = women →
  women = boys →
  total_earnings = 150 →
  men_wage = 10 →
  boys = 10 := by
  sorry

end NUMINAMATH_CALUDE_number_of_boys_l1990_199077


namespace NUMINAMATH_CALUDE_ali_final_money_l1990_199003

-- Define the initial state of Ali's wallet
def initial_wallet : ℚ :=
  7 * 5 + 1 * 10 + 3 * 20 + 1 * 50 + 8 * 1 + 10 * (1/4)

-- Morning transaction
def morning_transaction (wallet : ℚ) : ℚ :=
  wallet - (50 + 20 + 5) + (3 + 8 * (1/4) + 10 * (1/10))

-- Coffee shop transaction
def coffee_transaction (wallet : ℚ) : ℚ :=
  wallet - (15/4)

-- Afternoon transaction
def afternoon_transaction (wallet : ℚ) : ℚ :=
  wallet + 42

-- Evening transaction
def evening_transaction (wallet : ℚ) : ℚ :=
  wallet - (45/4)

-- Final wallet state after all transactions
def final_wallet : ℚ :=
  evening_transaction (afternoon_transaction (coffee_transaction (morning_transaction initial_wallet)))

-- Theorem statement
theorem ali_final_money :
  final_wallet = 247/2 := by sorry

end NUMINAMATH_CALUDE_ali_final_money_l1990_199003


namespace NUMINAMATH_CALUDE_sufficient_condition_not_necessary_condition_sufficient_but_not_necessary_l1990_199046

/-- The distance from point (2,1) to the line x=a is 3 -/
def distance_condition (a : ℝ) : Prop := |a - 2| = 3

/-- a=5 is a sufficient condition -/
theorem sufficient_condition : distance_condition 5 := by sorry

/-- a=5 is not a necessary condition -/
theorem not_necessary_condition : ∃ x, x ≠ 5 ∧ distance_condition x := by sorry

/-- a=5 is a sufficient but not necessary condition -/
theorem sufficient_but_not_necessary : 
  (distance_condition 5) ∧ (∃ x, x ≠ 5 ∧ distance_condition x) := by sorry

end NUMINAMATH_CALUDE_sufficient_condition_not_necessary_condition_sufficient_but_not_necessary_l1990_199046


namespace NUMINAMATH_CALUDE_complement_intersection_theorem_l1990_199076

def U : Set Nat := {2, 3, 4, 5, 6}
def A : Set Nat := {2, 5, 6}
def B : Set Nat := {3, 5}

theorem complement_intersection_theorem :
  (U \ A) ∩ B = {3} := by sorry

end NUMINAMATH_CALUDE_complement_intersection_theorem_l1990_199076


namespace NUMINAMATH_CALUDE_jones_elementary_population_l1990_199098

theorem jones_elementary_population :
  ∀ (total_students : ℕ) (boys_percentage : ℚ) (sample_size : ℕ),
    boys_percentage = 60 / 100 →
    sample_size = 90 →
    (sample_size : ℚ) / boys_percentage = total_students →
    total_students = 150 := by
  sorry

end NUMINAMATH_CALUDE_jones_elementary_population_l1990_199098


namespace NUMINAMATH_CALUDE_sum_of_roots_l1990_199043

-- Define the quadratic equation
def quadratic (x p q : ℝ) : Prop := x^2 - 2*p*x + q = 0

-- Define the theorem
theorem sum_of_roots (p q : ℝ) (hp : p ≠ 0) (hq : q ≠ 0) :
  (∃ x y : ℝ, quadratic x p q ∧ quadratic y p q ∧ x ≠ y) →
  x + y = 2 :=
by sorry

end NUMINAMATH_CALUDE_sum_of_roots_l1990_199043


namespace NUMINAMATH_CALUDE_ant_problem_l1990_199038

/-- Represents the position of an ant on a square path -/
structure AntPosition where
  side : ℕ  -- 0: bottom, 1: right, 2: top, 3: left
  distance : ℝ  -- distance from the start of the side

/-- Represents a square path -/
structure SquarePath where
  sideLength : ℝ

/-- Represents the state of the three ants -/
structure AntState where
  mu : AntPosition
  ra : AntPosition
  vey : AntPosition

/-- Checks if the ants are aligned on a straight line -/
def areAntsAligned (state : AntState) (paths : List SquarePath) : Prop :=
  sorry

/-- Updates the positions of the ants based on the distance they've traveled -/
def updateAntPositions (initialState : AntState) (paths : List SquarePath) (distance : ℝ) : AntState :=
  sorry

theorem ant_problem (a : ℝ) :
  let paths := [⟨a⟩, ⟨a + 2⟩, ⟨a + 4⟩]
  let initialState : AntState := {
    mu := { side := 0, distance := 0 },
    ra := { side := 0, distance := 0 },
    vey := { side := 0, distance := 0 }
  }
  let finalState := updateAntPositions initialState paths ((a + 4) / 2)
  finalState.mu.side = 1 ∧
  finalState.mu.distance = 0 ∧
  finalState.ra.side = 1 ∧
  finalState.vey.side = 1 ∧
  areAntsAligned finalState paths →
  a = 4 :=
sorry

end NUMINAMATH_CALUDE_ant_problem_l1990_199038


namespace NUMINAMATH_CALUDE_devices_delivered_l1990_199072

/-- Represents the properties of the energy-saving devices delivery -/
structure DeviceDelivery where
  totalWeight : ℕ
  lightestThreeWeight : ℕ
  heaviestThreeWeight : ℕ
  allWeightsDifferent : Bool

/-- The number of devices in the delivery -/
def numDevices (d : DeviceDelivery) : ℕ := sorry

/-- Theorem stating that given the specific conditions, the number of devices is 10 -/
theorem devices_delivered (d : DeviceDelivery) 
  (h1 : d.totalWeight = 120)
  (h2 : d.lightestThreeWeight = 31)
  (h3 : d.heaviestThreeWeight = 41)
  (h4 : d.allWeightsDifferent = true) :
  numDevices d = 10 := by sorry

end NUMINAMATH_CALUDE_devices_delivered_l1990_199072


namespace NUMINAMATH_CALUDE_max_value_sqrt_sum_l1990_199056

theorem max_value_sqrt_sum (x y z : ℝ) (h1 : 0 ≤ x) (h2 : 0 ≤ y) (h3 : 0 ≤ z) (h4 : x + y + z = 15) :
  Real.sqrt (3 * x + 1) + Real.sqrt (3 * y + 1) + Real.sqrt (3 * z + 1) ≤ Real.sqrt 48 ∧
  ∃ (a b c : ℝ), 0 ≤ a ∧ 0 ≤ b ∧ 0 ≤ c ∧ a + b + c = 15 ∧
    Real.sqrt (3 * a + 1) + Real.sqrt (3 * b + 1) + Real.sqrt (3 * c + 1) = Real.sqrt 48 :=
by sorry

end NUMINAMATH_CALUDE_max_value_sqrt_sum_l1990_199056


namespace NUMINAMATH_CALUDE_angle_A_is_pi_over_three_max_area_when_a_is_four_l1990_199026

-- Define the triangle ABC
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

-- Define the given condition
def condition (t : Triangle) : Prop :=
  (t.a + t.b) * (Real.sin t.A - Real.sin t.B) = t.c * (Real.sin t.C - Real.sin t.B)

-- Theorem for part 1
theorem angle_A_is_pi_over_three (t : Triangle) (h : condition t) : t.A = π / 3 := by
  sorry

-- Theorem for part 2
theorem max_area_when_a_is_four (t : Triangle) (h1 : condition t) (h2 : t.a = 4) :
  ∃ (S : ℝ), S = 4 * Real.sqrt 3 ∧ ∀ (S' : ℝ), S' = 1/2 * t.b * t.c * Real.sin t.A → S' ≤ S := by
  sorry

end NUMINAMATH_CALUDE_angle_A_is_pi_over_three_max_area_when_a_is_four_l1990_199026


namespace NUMINAMATH_CALUDE_otimes_inequality_implies_a_range_l1990_199040

-- Define the ⊗ operation
def otimes (x y : ℝ) := x * (1 - y)

-- State the theorem
theorem otimes_inequality_implies_a_range (a : ℝ) :
  (∀ x : ℝ, otimes (x - a) (x + a) < 1) →
  -1/2 < a ∧ a < 3/2 :=
by sorry

end NUMINAMATH_CALUDE_otimes_inequality_implies_a_range_l1990_199040


namespace NUMINAMATH_CALUDE_probability_james_and_david_chosen_l1990_199078

def total_workers : ℕ := 22
def workers_to_choose : ℕ := 4

theorem probability_james_and_david_chosen :
  (Nat.choose (total_workers - 2) (workers_to_choose - 2)) / 
  (Nat.choose total_workers workers_to_choose) = 2 / 231 := by
  sorry

end NUMINAMATH_CALUDE_probability_james_and_david_chosen_l1990_199078


namespace NUMINAMATH_CALUDE_x_over_y_equals_four_l1990_199058

theorem x_over_y_equals_four (x y : ℝ) (h1 : y ≠ 0) (h2 : 2 * x - y = 1.75 * x) : x / y = 4 := by
  sorry

end NUMINAMATH_CALUDE_x_over_y_equals_four_l1990_199058


namespace NUMINAMATH_CALUDE_factoring_expression_l1990_199017

theorem factoring_expression (x y : ℝ) : 3*x*(x+3) + y*(x+3) = (x+3)*(3*x+y) := by
  sorry

end NUMINAMATH_CALUDE_factoring_expression_l1990_199017


namespace NUMINAMATH_CALUDE_cheese_grating_time_is_five_l1990_199063

/-- The time in minutes it takes to grate cheese for one omelet --/
def cheese_grating_time (
  total_time : ℕ)
  (num_omelets : ℕ)
  (pepper_chop_time : ℕ)
  (onion_chop_time : ℕ)
  (omelet_cook_time : ℕ)
  (num_peppers : ℕ)
  (num_onions : ℕ) : ℕ :=
  total_time - 
  (num_peppers * pepper_chop_time + 
   num_onions * onion_chop_time + 
   num_omelets * omelet_cook_time)

theorem cheese_grating_time_is_five :
  cheese_grating_time 50 5 3 4 5 4 2 = 5 := by
  sorry

end NUMINAMATH_CALUDE_cheese_grating_time_is_five_l1990_199063


namespace NUMINAMATH_CALUDE_deck_size_proof_l1990_199020

theorem deck_size_proof (r b : ℕ) : 
  (r : ℚ) / (r + b : ℚ) = 1/4 →
  ((r + 6 : ℚ) / (r + b + 6 : ℚ) = 1/3) →
  r + b = 48 := by
  sorry

end NUMINAMATH_CALUDE_deck_size_proof_l1990_199020


namespace NUMINAMATH_CALUDE_nonagon_prism_edges_l1990_199066

/-- A prism is a three-dimensional shape with two identical ends (bases) and flat sides. -/
structure Prism where
  base : Nat  -- number of sides in the base shape

/-- The number of edges in a prism. -/
def Prism.edges (p : Prism) : Nat :=
  3 * p.base

theorem nonagon_prism_edges :
  ∀ p : Prism, p.base = 9 → p.edges = 27 :=
by
  sorry

end NUMINAMATH_CALUDE_nonagon_prism_edges_l1990_199066


namespace NUMINAMATH_CALUDE_trigonometric_equation_solution_l1990_199002

theorem trigonometric_equation_solution :
  ∀ x : ℝ, 5.22 * (Real.sin x)^2 - 2 * Real.sin x * Real.cos x = 3 * (Real.cos x)^2 ↔
  (∃ k : ℤ, x = Real.arctan 0.973 + k * Real.pi) ∨
  (∃ k : ℤ, x = Real.arctan (-0.59) + k * Real.pi) :=
by sorry

end NUMINAMATH_CALUDE_trigonometric_equation_solution_l1990_199002


namespace NUMINAMATH_CALUDE_series_sum_l1990_199092

/-- The positive real solution to x^3 + (2/5)x - 1 = 0 -/
noncomputable def r : ℝ :=
  Real.sqrt (Real.sqrt (1 + 2/5))

/-- The sum of the series r^2 + 2r^5 + 3r^8 + 4r^11 + ... -/
noncomputable def S : ℝ :=
  ∑' n, (n + 1) * r^(3*n + 2)

theorem series_sum : 
  r > 0 ∧ r^3 + 2/5 * r - 1 = 0 → S = 25/4 :=
by
  sorry

#check series_sum

end NUMINAMATH_CALUDE_series_sum_l1990_199092


namespace NUMINAMATH_CALUDE_logarithmic_inequality_l1990_199084

theorem logarithmic_inequality (a b c : ℝ) 
  (ha : a > 0) (hb : b > 0) (hc : c > 0) 
  (hab : a ≠ b) (hbc : b ≠ c) (hca : c ≠ a) : 
  (Real.log a)/((a-b)*(a-c)) + (Real.log b)/((b-c)*(b-a)) + (Real.log c)/((c-a)*(c-b)) < 0 := by
  sorry

end NUMINAMATH_CALUDE_logarithmic_inequality_l1990_199084


namespace NUMINAMATH_CALUDE_supplementary_angle_measures_l1990_199097

theorem supplementary_angle_measures :
  ∃ (possible_measures : Finset ℕ),
    (∀ A ∈ possible_measures,
      ∃ (B : ℕ) (k : ℕ),
        A > 0 ∧ B > 0 ∧ k > 0 ∧
        A + B = 180 ∧
        A = k * B) ∧
    (∀ A : ℕ,
      (∃ (B : ℕ) (k : ℕ),
        A > 0 ∧ B > 0 ∧ k > 0 ∧
        A + B = 180 ∧
        A = k * B) →
      A ∈ possible_measures) ∧
    Finset.card possible_measures = 17 :=
by sorry

end NUMINAMATH_CALUDE_supplementary_angle_measures_l1990_199097


namespace NUMINAMATH_CALUDE_download_speed_calculation_l1990_199070

theorem download_speed_calculation (total_size : ℝ) (downloaded : ℝ) (remaining_time : ℝ)
  (h1 : total_size = 880)
  (h2 : downloaded = 310)
  (h3 : remaining_time = 190) :
  (total_size - downloaded) / remaining_time = 3 := by
  sorry

end NUMINAMATH_CALUDE_download_speed_calculation_l1990_199070


namespace NUMINAMATH_CALUDE_johns_age_l1990_199023

theorem johns_age (john_age dad_age : ℕ) : 
  dad_age = john_age + 30 →
  john_age + dad_age = 80 →
  john_age = 25 := by
sorry

end NUMINAMATH_CALUDE_johns_age_l1990_199023


namespace NUMINAMATH_CALUDE_circle_condition_l1990_199090

-- Define the equation
def circle_equation (a x y : ℝ) : Prop :=
  a^2 * x^2 + (a + 2) * y^2 + 2 * a * x + a = 0

-- Define what it means for an equation to represent a circle
def represents_circle (a : ℝ) : Prop :=
  a^2 = a + 2 ∧ a ≠ 0

-- Theorem statement
theorem circle_condition (a : ℝ) :
  represents_circle a ↔ a = -1 :=
sorry

end NUMINAMATH_CALUDE_circle_condition_l1990_199090


namespace NUMINAMATH_CALUDE_stream_speed_l1990_199031

/-- Proves that given a boat with a speed of 22 km/hr in still water,
    traveling 216 km downstream in 8 hours, the speed of the stream is 5 km/hr. -/
theorem stream_speed (boat_speed : ℝ) (distance : ℝ) (time : ℝ) (stream_speed : ℝ) :
  boat_speed = 22 →
  distance = 216 →
  time = 8 →
  distance = (boat_speed + stream_speed) * time →
  stream_speed = 5 := by
sorry

end NUMINAMATH_CALUDE_stream_speed_l1990_199031


namespace NUMINAMATH_CALUDE_three_fifths_of_twelve_times_ten_minus_twenty_l1990_199005

theorem three_fifths_of_twelve_times_ten_minus_twenty : 
  (3 : ℚ) / 5 * ((12 * 10) - 20) = 60 := by
  sorry

end NUMINAMATH_CALUDE_three_fifths_of_twelve_times_ten_minus_twenty_l1990_199005


namespace NUMINAMATH_CALUDE_rose_price_l1990_199028

/-- The price of roses given Hanna's budget and distribution to friends -/
theorem rose_price (total_budget : ℚ) (jenna_fraction : ℚ) (imma_fraction : ℚ) (friends_roses : ℕ) : 
  total_budget = 300 →
  jenna_fraction = 1/3 →
  imma_fraction = 1/2 →
  friends_roses = 125 →
  total_budget / ((friends_roses : ℚ) / (jenna_fraction + imma_fraction)) = 2 :=
by sorry

end NUMINAMATH_CALUDE_rose_price_l1990_199028


namespace NUMINAMATH_CALUDE_simplify_expression_l1990_199069

theorem simplify_expression : (((3 + 4 + 5 + 6) / 3) + ((3 * 4 + 9) / 4)) = 45 / 4 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l1990_199069


namespace NUMINAMATH_CALUDE_solution_set_equality_l1990_199096

/-- The solution set of the inequality (x^2 - 2x - 3)(x^2 - 4x + 4) < 0 -/
def SolutionSet : Set ℝ :=
  {x | (x^2 - 2*x - 3) * (x^2 - 4*x + 4) < 0}

/-- The set {x | -1 < x < 3 and x ≠ 2} -/
def TargetSet : Set ℝ :=
  {x | -1 < x ∧ x < 3 ∧ x ≠ 2}

theorem solution_set_equality : SolutionSet = TargetSet := by
  sorry

end NUMINAMATH_CALUDE_solution_set_equality_l1990_199096


namespace NUMINAMATH_CALUDE_monica_money_exchange_l1990_199049

def exchange_rate : ℚ := 8 / 5

theorem monica_money_exchange (x : ℚ) : 
  (exchange_rate * x - 40 = x) → x = 200 := by
  sorry

end NUMINAMATH_CALUDE_monica_money_exchange_l1990_199049


namespace NUMINAMATH_CALUDE_black_area_after_transformations_l1990_199001

/-- The fraction of black area remaining after one transformation -/
def remaining_fraction : ℚ := 2 / 3

/-- The number of transformations -/
def num_transformations : ℕ := 6

/-- The theorem stating the fraction of black area remaining after six transformations -/
theorem black_area_after_transformations :
  remaining_fraction ^ num_transformations = 64 / 729 := by
  sorry

end NUMINAMATH_CALUDE_black_area_after_transformations_l1990_199001


namespace NUMINAMATH_CALUDE_stratified_sample_over_45_l1990_199074

/-- Represents the number of employees in a stratified sample from a given population -/
def stratified_sample (total_population : ℕ) (group_size : ℕ) (sample_size : ℕ) : ℕ :=
  (group_size * sample_size) / total_population

/-- Proves that the number of employees over 45 in the stratified sample is 10 -/
theorem stratified_sample_over_45 :
  stratified_sample 200 80 25 = 10 := by
  sorry

end NUMINAMATH_CALUDE_stratified_sample_over_45_l1990_199074


namespace NUMINAMATH_CALUDE_simplify_and_evaluate_l1990_199035

theorem simplify_and_evaluate (a : ℝ) (h : a = Real.sqrt 3 + 1) :
  (1 - a / (a + 1)) / ((a^2 - 2*a + 1) / (a^2 - 1)) = Real.sqrt 3 / 3 := by
  sorry

end NUMINAMATH_CALUDE_simplify_and_evaluate_l1990_199035


namespace NUMINAMATH_CALUDE_specific_tetrahedron_volume_l1990_199087

/-- Represents a tetrahedron with vertices P, Q, R, and S -/
structure Tetrahedron where
  PQ : ℝ
  PR : ℝ
  PS : ℝ
  QR : ℝ
  QS : ℝ
  RS : ℝ

/-- Calculates the volume of a tetrahedron -/
def tetrahedronVolume (t : Tetrahedron) : ℝ :=
  sorry

/-- Theorem stating that the volume of the specific tetrahedron is 20/3 -/
theorem specific_tetrahedron_volume :
  let t : Tetrahedron := {
    PQ := 6,
    PR := 5,
    PS := 5,
    QR := 5,
    QS := 4,
    RS := (10/3) * Real.sqrt 3
  }
  tetrahedronVolume t = 20/3 := by sorry

end NUMINAMATH_CALUDE_specific_tetrahedron_volume_l1990_199087


namespace NUMINAMATH_CALUDE_distance_between_centers_l1990_199064

/-- Given an isosceles triangle with circumradius R and inradius r,
    the distance d between the centers of the circumcircle and incircle
    is given by d = √(R(R-2r)). -/
theorem distance_between_centers (R r : ℝ) (h : R > 0 ∧ r > 0) :
  ∃ (d : ℝ), d = Real.sqrt (R * (R - 2 * r)) := by
  sorry

end NUMINAMATH_CALUDE_distance_between_centers_l1990_199064


namespace NUMINAMATH_CALUDE_square_side_length_l1990_199052

theorem square_side_length (area : ℚ) (side : ℚ) 
  (h1 : area = 9/16) (h2 : side^2 = area) : side = 3/4 := by
  sorry

end NUMINAMATH_CALUDE_square_side_length_l1990_199052


namespace NUMINAMATH_CALUDE_simplify_radicals_l1990_199024

theorem simplify_radicals : Real.sqrt 18 - Real.sqrt 8 = Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_simplify_radicals_l1990_199024


namespace NUMINAMATH_CALUDE_largest_decimal_l1990_199055

theorem largest_decimal : 
  let a := 0.9123
  let b := 0.9912
  let c := 0.9191
  let d := 0.9301
  let e := 0.9091
  b > a ∧ b > c ∧ b > d ∧ b > e := by
  sorry

end NUMINAMATH_CALUDE_largest_decimal_l1990_199055


namespace NUMINAMATH_CALUDE_attached_pyramids_volume_l1990_199041

/-- A solid formed by two attached pyramids -/
structure AttachedPyramids where
  /-- Length of each edge in the square-based pyramid -/
  base_edge_length : ℝ
  /-- Total length of all edges in the resulting solid -/
  total_edge_length : ℝ

/-- The volume of the attached pyramids solid -/
noncomputable def volume (ap : AttachedPyramids) : ℝ :=
  2 * Real.sqrt 2

/-- Theorem stating the volume of the attached pyramids solid -/
theorem attached_pyramids_volume (ap : AttachedPyramids) 
  (h1 : ap.base_edge_length = 2)
  (h2 : ap.total_edge_length = 18) :
  volume ap = 2 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_attached_pyramids_volume_l1990_199041


namespace NUMINAMATH_CALUDE_dividend_calculation_l1990_199091

theorem dividend_calculation (divisor quotient remainder : ℕ) 
  (h_divisor : divisor = 19)
  (h_quotient : quotient = 7)
  (h_remainder : remainder = 6) :
  divisor * quotient + remainder = 139 := by
sorry

end NUMINAMATH_CALUDE_dividend_calculation_l1990_199091


namespace NUMINAMATH_CALUDE_distinct_prime_factors_of_30_factorial_l1990_199022

theorem distinct_prime_factors_of_30_factorial :
  (∀ p : ℕ, p.Prime → p ≤ 30 → p ∣ Nat.factorial 30) ∧
  (∃ S : Finset ℕ, (∀ p ∈ S, p.Prime ∧ p ≤ 30) ∧ 
                   (∀ p : ℕ, p.Prime → p ≤ 30 → p ∈ S) ∧ 
                   S.card = 10) :=
by sorry

end NUMINAMATH_CALUDE_distinct_prime_factors_of_30_factorial_l1990_199022


namespace NUMINAMATH_CALUDE_complex_sum_theorem_l1990_199065

theorem complex_sum_theorem (a c d e f : ℝ) : 
  e = -a - c → (a + 2*I) + (c + d*I) + (e + f*I) = 2*I → d + f = 0 := by
  sorry

end NUMINAMATH_CALUDE_complex_sum_theorem_l1990_199065


namespace NUMINAMATH_CALUDE_total_pears_picked_l1990_199073

theorem total_pears_picked (sara_pears tim_pears : ℕ) 
  (h1 : sara_pears = 6) 
  (h2 : tim_pears = 5) : 
  sara_pears + tim_pears = 11 := by
  sorry

end NUMINAMATH_CALUDE_total_pears_picked_l1990_199073


namespace NUMINAMATH_CALUDE_point_in_second_quadrant_implies_m_range_l1990_199015

/-- A point in the second quadrant has a negative x-coordinate and a positive y-coordinate. -/
def is_in_second_quadrant (x y : ℝ) : Prop := x < 0 ∧ y > 0

/-- The theorem states that if the point P(m-3, m-2) is in the second quadrant,
    then m is strictly between 2 and 3. -/
theorem point_in_second_quadrant_implies_m_range
  (m : ℝ)
  (h : is_in_second_quadrant (m - 3) (m - 2)) :
  2 < m ∧ m < 3 :=
sorry

end NUMINAMATH_CALUDE_point_in_second_quadrant_implies_m_range_l1990_199015


namespace NUMINAMATH_CALUDE_polyhedron_parity_l1990_199060

-- Define a polyhedron structure
structure Polyhedron where
  vertices : Set (ℕ × ℕ × ℕ)
  edges : Set (Set (ℕ × ℕ × ℕ))
  faces : Set (Set (ℕ × ℕ × ℕ))
  -- Add necessary conditions for a valid polyhedron

-- Function to count faces with odd number of sides
def count_odd_faces (p : Polyhedron) : ℕ := sorry

-- Function to count vertices with odd degree
def count_odd_degree_vertices (p : Polyhedron) : ℕ := sorry

-- Theorem statement
theorem polyhedron_parity (p : Polyhedron) : 
  Even (count_odd_faces p) ∧ Even (count_odd_degree_vertices p) := by sorry

end NUMINAMATH_CALUDE_polyhedron_parity_l1990_199060


namespace NUMINAMATH_CALUDE_pet_store_dogs_l1990_199030

theorem pet_store_dogs (initial_dogs : ℕ) (sunday_dogs : ℕ) (monday_dogs : ℕ) (final_dogs : ℕ)
  (h1 : initial_dogs = 2)
  (h2 : monday_dogs = 3)
  (h3 : final_dogs = 10)
  (h4 : initial_dogs + sunday_dogs + monday_dogs = final_dogs) :
  sunday_dogs = 5 := by
  sorry

end NUMINAMATH_CALUDE_pet_store_dogs_l1990_199030


namespace NUMINAMATH_CALUDE_scoops_per_carton_is_ten_l1990_199081

/-- Represents the number of scoops in each carton of ice cream -/
def scoops_per_carton : ℕ := sorry

/-- The total number of cartons -/
def total_cartons : ℕ := 3

/-- The number of scoops Ethan wants -/
def ethan_scoops : ℕ := 2

/-- The number of people who want 2 scoops of chocolate -/
def chocolate_lovers : ℕ := 3

/-- The number of scoops Olivia wants -/
def olivia_scoops : ℕ := 2

/-- The number of scoops Shannon wants (twice as much as Olivia) -/
def shannon_scoops : ℕ := 2 * olivia_scoops

/-- The number of scoops left after everyone has taken their scoops -/
def scoops_left : ℕ := 16

/-- The total number of scoops taken -/
def total_scoops_taken : ℕ := 
  ethan_scoops + (chocolate_lovers * 2) + olivia_scoops + shannon_scoops

/-- Theorem stating that the number of scoops per carton is 10 -/
theorem scoops_per_carton_is_ten : scoops_per_carton = 10 := by
  sorry

end NUMINAMATH_CALUDE_scoops_per_carton_is_ten_l1990_199081


namespace NUMINAMATH_CALUDE_toms_weekly_income_l1990_199062

/-- Tom's crab fishing business --/
def crab_business (num_buckets : ℕ) (crabs_per_bucket : ℕ) (price_per_crab : ℕ) (days_per_week : ℕ) : ℕ :=
  num_buckets * crabs_per_bucket * price_per_crab * days_per_week

/-- Tom's weekly income from selling crabs --/
theorem toms_weekly_income :
  crab_business 8 12 5 7 = 3360 := by
  sorry

#eval crab_business 8 12 5 7

end NUMINAMATH_CALUDE_toms_weekly_income_l1990_199062


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_l1990_199013

-- Define the quadratic function
def f (a : ℝ) (x : ℝ) : ℝ := 2 * x^2 - 3 * x + a

-- Define the solution set condition
def is_solution_set (a m : ℝ) : Prop :=
  ∀ x, f a x < 0 ↔ m < x ∧ x < 1

-- Theorem statement
theorem quadratic_inequality_solution (a m : ℝ) 
  (h : is_solution_set a m) : m = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_l1990_199013


namespace NUMINAMATH_CALUDE_common_root_quadratic_equations_l1990_199068

theorem common_root_quadratic_equations (b : ℤ) :
  (∃ x : ℝ, 2 * x^2 + (3 * b - 1) * x - 3 = 0 ∧ 6 * x^2 - (2 * b - 3) * x - 1 = 0) ↔ b = 2 :=
by sorry

end NUMINAMATH_CALUDE_common_root_quadratic_equations_l1990_199068


namespace NUMINAMATH_CALUDE_number_division_property_l1990_199099

theorem number_division_property : ∃ x : ℝ, x / 5 = 80 + x / 6 := by
  sorry

end NUMINAMATH_CALUDE_number_division_property_l1990_199099
