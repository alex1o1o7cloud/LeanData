import Mathlib

namespace NUMINAMATH_CALUDE_unique_score_100_l630_63043

/-- Represents a competition score -/
structure CompetitionScore where
  total : Nat
  correct : Nat
  wrong : Nat
  score : Nat
  h1 : total = 25
  h2 : score = 25 + 5 * correct - wrong
  h3 : total = correct + wrong

/-- Checks if a given score uniquely determines the number of correct and wrong answers -/
def isUniquelyDetermined (s : Nat) : Prop :=
  ∃! cs : CompetitionScore, cs.score = s

theorem unique_score_100 :
  isUniquelyDetermined 100 ∧
  ∀ s, 95 < s ∧ s < 100 → ¬isUniquelyDetermined s :=
sorry

end NUMINAMATH_CALUDE_unique_score_100_l630_63043


namespace NUMINAMATH_CALUDE_coins_missing_fraction_l630_63089

-- Define the initial number of coins
variable (x : ℚ)

-- Define the fractions based on the problem conditions
def lost_fraction : ℚ := 1 / 3
def found_fraction : ℚ := 5 / 6
def spent_fraction : ℚ := 1 / 4

-- Define the fraction of coins still missing
def missing_fraction : ℚ := 
  spent_fraction + (lost_fraction - lost_fraction * found_fraction)

-- Theorem to prove
theorem coins_missing_fraction : missing_fraction = 11 / 36 := by
  sorry

end NUMINAMATH_CALUDE_coins_missing_fraction_l630_63089


namespace NUMINAMATH_CALUDE_opposite_numbers_absolute_value_l630_63038

theorem opposite_numbers_absolute_value (a b : ℝ) : 
  a + b = 0 → |a - 2014 + b| = 2014 := by sorry

end NUMINAMATH_CALUDE_opposite_numbers_absolute_value_l630_63038


namespace NUMINAMATH_CALUDE_amy_video_files_amy_video_files_proof_l630_63060

theorem amy_video_files : ℕ → Prop :=
  fun initial_video_files =>
    let initial_music_files : ℕ := 4
    let deleted_files : ℕ := 23
    let remaining_files : ℕ := 2
    initial_music_files + initial_video_files - deleted_files = remaining_files →
    initial_video_files = 21

-- Proof
theorem amy_video_files_proof : amy_video_files 21 := by
  sorry

end NUMINAMATH_CALUDE_amy_video_files_amy_video_files_proof_l630_63060


namespace NUMINAMATH_CALUDE_quadratic_minimum_value_l630_63024

/-- A quadratic function that takes values 6, 5, and 5 for three consecutive natural numbers. -/
def QuadraticFunction (f : ℝ → ℝ) : Prop :=
  ∃ n : ℕ, f n = 6 ∧ f (n + 1) = 5 ∧ f (n + 2) = 5

/-- The theorem stating that the minimum value of the quadratic function is 39/8. -/
theorem quadratic_minimum_value (f : ℝ → ℝ) (h : QuadraticFunction f) :
  ∃ x : ℝ, f x = 39/8 ∧ ∀ y : ℝ, f y ≥ 39/8 :=
by
  sorry

end NUMINAMATH_CALUDE_quadratic_minimum_value_l630_63024


namespace NUMINAMATH_CALUDE_line_passes_through_fixed_point_l630_63095

/-- The line equation passing through a fixed point -/
def line_equation (k x y : ℝ) : Prop :=
  (k + 1) * x - (2 * k - 1) * y + 3 * k = 0

/-- Theorem stating that the line passes through (-1, 1) for all k -/
theorem line_passes_through_fixed_point :
  ∀ k : ℝ, line_equation k (-1) 1 := by
  sorry

end NUMINAMATH_CALUDE_line_passes_through_fixed_point_l630_63095


namespace NUMINAMATH_CALUDE_salt_solution_concentration_l630_63050

/-- Given a mixture of pure water and salt solution, prove the original salt solution concentration -/
theorem salt_solution_concentration 
  (pure_water_volume : ℝ) 
  (salt_solution_volume : ℝ) 
  (final_mixture_concentration : ℝ) 
  (h1 : pure_water_volume = 1)
  (h2 : salt_solution_volume = 0.5)
  (h3 : final_mixture_concentration = 15) :
  let total_volume := pure_water_volume + salt_solution_volume
  let salt_amount := final_mixture_concentration / 100 * total_volume
  salt_amount / salt_solution_volume * 100 = 45 := by
  sorry

end NUMINAMATH_CALUDE_salt_solution_concentration_l630_63050


namespace NUMINAMATH_CALUDE_not_divisible_by_100_l630_63071

theorem not_divisible_by_100 : ∀ n : ℕ, ¬(100 ∣ (n^2 + 6*n + 2019)) := by
  sorry

end NUMINAMATH_CALUDE_not_divisible_by_100_l630_63071


namespace NUMINAMATH_CALUDE_input_is_only_input_statement_l630_63099

-- Define the possible statement types
inductive StatementType
| Output
| Input
| Conditional
| Termination

-- Define the statements
def PRINT : StatementType := StatementType.Output
def INPUT : StatementType := StatementType.Input
def IF : StatementType := StatementType.Conditional
def END : StatementType := StatementType.Termination

-- Theorem: INPUT is the only input statement among the given options
theorem input_is_only_input_statement :
  (PRINT = StatementType.Input → False) ∧
  (INPUT = StatementType.Input) ∧
  (IF = StatementType.Input → False) ∧
  (END = StatementType.Input → False) :=
by sorry

end NUMINAMATH_CALUDE_input_is_only_input_statement_l630_63099


namespace NUMINAMATH_CALUDE_discount_profit_calculation_l630_63030

-- Define the discount percentage
def discount : ℝ := 0.05

-- Define the profit percentage without discount
def profit_without_discount : ℝ := 0.29

-- Define the function to calculate profit percentage with discount
def profit_with_discount (d : ℝ) (p : ℝ) : ℝ :=
  (1 - d) * (1 + p) - 1

-- Theorem statement
theorem discount_profit_calculation :
  abs (profit_with_discount discount profit_without_discount - 0.2255) < 0.0001 := by
  sorry

end NUMINAMATH_CALUDE_discount_profit_calculation_l630_63030


namespace NUMINAMATH_CALUDE_pharmacy_service_l630_63078

/-- The number of customers served by three workers in a day -/
def customers_served (regular_hours work_rate reduced_hours : ℕ) : ℕ :=
  work_rate * (2 * regular_hours + reduced_hours)

/-- Theorem: Given the specific conditions, the total number of customers served is 154 -/
theorem pharmacy_service : customers_served 8 7 6 = 154 := by
  sorry

end NUMINAMATH_CALUDE_pharmacy_service_l630_63078


namespace NUMINAMATH_CALUDE_smallest_sum_of_reciprocals_l630_63009

theorem smallest_sum_of_reciprocals (x y : ℕ+) :
  x ≠ y →
  (1 : ℚ) / x + (1 : ℚ) / y = (1 : ℚ) / 18 →
  ∀ a b : ℕ+, a ≠ b → (1 : ℚ) / a + (1 : ℚ) / b = (1 : ℚ) / 18 →
  (x : ℕ) + (y : ℕ) ≤ (a : ℕ) + (b : ℕ) →
  (x : ℕ) + (y : ℕ) = 75 :=
by sorry

end NUMINAMATH_CALUDE_smallest_sum_of_reciprocals_l630_63009


namespace NUMINAMATH_CALUDE_function_identity_l630_63074

def StrictlyIncreasing (f : ℕ → ℕ) : Prop :=
  ∀ m n, m < n → f m < f n

theorem function_identity (f : ℕ → ℕ) 
  (h_increasing : StrictlyIncreasing f)
  (h_two : f 2 = 2)
  (h_coprime : ∀ m n, Nat.Coprime m n → f (m * n) = f m * f n) :
  ∀ n, f n = n :=
sorry

end NUMINAMATH_CALUDE_function_identity_l630_63074


namespace NUMINAMATH_CALUDE_f_eval_one_l630_63055

/-- The polynomial g(x) -/
def g (a : ℝ) (x : ℝ) : ℝ := x^3 + a*x^2 + x + 20

/-- The polynomial f(x) -/
def f (b c : ℝ) (x : ℝ) : ℝ := x^4 + x^3 + b*x^2 + 200*x + c

/-- Theorem stating that f(1) = -28417 given the conditions -/
theorem f_eval_one (a b c : ℝ) :
  (∃ x y z : ℝ, x ≠ y ∧ y ≠ z ∧ x ≠ z ∧ g a x = 0 ∧ g a y = 0 ∧ g a z = 0) →
  (∀ x : ℝ, g a x = 0 → f b c x = 0) →
  f b c 1 = -28417 := by
  sorry

end NUMINAMATH_CALUDE_f_eval_one_l630_63055


namespace NUMINAMATH_CALUDE_female_students_count_l630_63087

theorem female_students_count (total_students sample_size male_sample : ℕ) 
  (h1 : total_students = 2000)
  (h2 : sample_size = 200)
  (h3 : male_sample = 103) : 
  (total_students : ℚ) * ((sample_size - male_sample) : ℚ) / (sample_size : ℚ) = 970 := by
  sorry

end NUMINAMATH_CALUDE_female_students_count_l630_63087


namespace NUMINAMATH_CALUDE_no_valid_triples_l630_63079

theorem no_valid_triples : 
  ¬∃ (a b c : ℤ) (x : ℚ), 
    a < 0 ∧ 
    b^2 - 4*a*c = 5 ∧ 
    a * x^2 + b * x + c > 0 := by
  sorry

end NUMINAMATH_CALUDE_no_valid_triples_l630_63079


namespace NUMINAMATH_CALUDE_expression_evaluation_l630_63097

theorem expression_evaluation (a b : ℚ) (h1 : a = 1/2) (h2 : b = -2) :
  (2*a + b)^2 - (2*a - b)*(a + b) - 2*(a - 2*b)*(a + 2*b) = 37 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l630_63097


namespace NUMINAMATH_CALUDE_triangle_max_area_l630_63065

/-- Given a triangle ABC with sides a, b, c opposite to angles A, B, C respectively,
    prove that the maximum area of the triangle is (3√3)/4 when b = √3 and (2a-c)cos B = √3 cos C -/
theorem triangle_max_area (a b c : ℝ) (A B C : ℝ) :
  b = Real.sqrt 3 →
  (2 * a - c) * Real.cos B = Real.sqrt 3 * Real.cos C →
  (∃ (S : ℝ), S = (1/2) * b * c * Real.sin A ∧ 
    ∀ (S' : ℝ), S' = (1/2) * b * c * Real.sin A → S' ≤ S) →
  (3 * Real.sqrt 3) / 4 = (1/2) * b * c * Real.sin A :=
by sorry

end NUMINAMATH_CALUDE_triangle_max_area_l630_63065


namespace NUMINAMATH_CALUDE_inequality_of_powers_l630_63058

theorem inequality_of_powers (m n : ℕ) : (5 + 3 * Real.sqrt 2) ^ m ≠ (3 + 5 * Real.sqrt 2) ^ n := by
  sorry

end NUMINAMATH_CALUDE_inequality_of_powers_l630_63058


namespace NUMINAMATH_CALUDE_arithmetic_sequence_common_difference_l630_63027

-- Define an arithmetic sequence
def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

-- State the theorem
theorem arithmetic_sequence_common_difference 
  (a : ℕ → ℝ) (h : is_arithmetic_sequence a) 
  (h2015_2013 : a 2015 = a 2013 + 6) : 
  ∃ d : ℝ, (∀ n : ℕ, a (n + 1) = a n + d) ∧ d = 3 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_common_difference_l630_63027


namespace NUMINAMATH_CALUDE_fraction_denominator_problem_l630_63069

theorem fraction_denominator_problem (n d : ℤ) : 
  d = n - 4 ∧ n + 6 = 3 * d → d = 5 := by
  sorry

end NUMINAMATH_CALUDE_fraction_denominator_problem_l630_63069


namespace NUMINAMATH_CALUDE_expression_simplification_l630_63039

theorem expression_simplification (p q r : ℝ) (hp : p ≠ 0) (hq : q ≠ 0) (hr : r ≠ 0) :
  (p / q + q / r + r / p - 1) * (p + q + r) +
  (p / q + q / r - r / p + 1) * (p + q - r) +
  (p / q - q / r + r / p + 1) * (p - q + r) +
  (-p / q + q / r + r / p + 1) * (-p + q + r) =
  4 * (p^2 / q + q^2 / r + r^2 / p) :=
by sorry

end NUMINAMATH_CALUDE_expression_simplification_l630_63039


namespace NUMINAMATH_CALUDE_square_side_length_l630_63001

theorem square_side_length (area : ℚ) (h : area = 9/16) : 
  ∃ (side : ℚ), side * side = area ∧ side = 3/4 := by
  sorry

end NUMINAMATH_CALUDE_square_side_length_l630_63001


namespace NUMINAMATH_CALUDE_initial_pineapples_l630_63091

/-- Proves that the initial number of pineapples in the store was 86 -/
theorem initial_pineapples (sold : ℕ) (rotten : ℕ) (fresh : ℕ) 
  (h1 : sold = 48) 
  (h2 : rotten = 9) 
  (h3 : fresh = 29) : 
  sold + rotten + fresh = 86 := by
  sorry

end NUMINAMATH_CALUDE_initial_pineapples_l630_63091


namespace NUMINAMATH_CALUDE_second_batch_weight_is_100_l630_63042

-- Define the initial stock
def initial_stock : ℝ := 400

-- Define the percentage of decaf in initial stock
def initial_decaf_percent : ℝ := 0.20

-- Define the percentage of decaf in the second batch
def second_batch_decaf_percent : ℝ := 0.50

-- Define the final percentage of decaf in total stock
def final_decaf_percent : ℝ := 0.26

-- Define the weight of the second batch as a variable
variable (second_batch_weight : ℝ)

-- Theorem statement
theorem second_batch_weight_is_100 :
  (initial_stock * initial_decaf_percent + second_batch_weight * second_batch_decaf_percent) / 
  (initial_stock + second_batch_weight) = final_decaf_percent →
  second_batch_weight = 100 := by
  sorry

end NUMINAMATH_CALUDE_second_batch_weight_is_100_l630_63042


namespace NUMINAMATH_CALUDE_polygon_with_40_degree_exterior_angles_has_9_sides_l630_63029

/-- A polygon with exterior angles of 40° has 9 sides -/
theorem polygon_with_40_degree_exterior_angles_has_9_sides :
  ∀ (n : ℕ), 
  (n > 2) →
  (360 / n = 40) →
  n = 9 := by
sorry

end NUMINAMATH_CALUDE_polygon_with_40_degree_exterior_angles_has_9_sides_l630_63029


namespace NUMINAMATH_CALUDE_gear_rotations_l630_63005

/-- Represents a gear with a given number of teeth -/
structure Gear where
  teeth : ℕ

/-- Represents a system of two engaged gears -/
structure GearSystem where
  gearA : Gear
  gearB : Gear

/-- Checks if the rotations of two gears are valid (i.e., they mesh properly) -/
def validRotations (gs : GearSystem) (rotA : ℕ) (rotB : ℕ) : Prop :=
  rotA * gs.gearA.teeth = rotB * gs.gearB.teeth

/-- Checks if the given rotations are the smallest possible -/
def smallestRotations (gs : GearSystem) (rotA : ℕ) (rotB : ℕ) : Prop :=
  ∀ (a b : ℕ), validRotations gs a b → (rotA ≤ a ∧ rotB ≤ b)

/-- The main theorem to prove -/
theorem gear_rotations (gs : GearSystem) (h1 : gs.gearA.teeth = 12) (h2 : gs.gearB.teeth = 54) :
  smallestRotations gs 9 2 :=
sorry

end NUMINAMATH_CALUDE_gear_rotations_l630_63005


namespace NUMINAMATH_CALUDE_remainder_problem_l630_63067

theorem remainder_problem (N : ℤ) : ∃ k : ℤ, N = 35 * k + 25 → ∃ m : ℤ, N = 15 * m + 10 := by
  sorry

end NUMINAMATH_CALUDE_remainder_problem_l630_63067


namespace NUMINAMATH_CALUDE_max_gcd_consecutive_terms_l630_63057

def a (n : ℕ) : ℕ := n.factorial + n

theorem max_gcd_consecutive_terms :
  ∃ (k : ℕ), k ≥ 2 ∧ 
  (∀ (n : ℕ), Nat.gcd (a n) (a (n + 1)) ≤ k) ∧
  (∃ (m : ℕ), Nat.gcd (a m) (a (m + 1)) = k) :=
sorry

end NUMINAMATH_CALUDE_max_gcd_consecutive_terms_l630_63057


namespace NUMINAMATH_CALUDE_hockey_league_face_count_l630_63022

/-- The number of times each team faces other teams in a hockey league -/
def faceCount (n : ℕ) (total_games : ℕ) : ℕ :=
  total_games / (n * (n - 1) / 2)

/-- Theorem: In a hockey league with 19 teams and 1710 total games, each team faces others 5 times -/
theorem hockey_league_face_count :
  faceCount 19 1710 = 5 := by
  sorry

end NUMINAMATH_CALUDE_hockey_league_face_count_l630_63022


namespace NUMINAMATH_CALUDE_zongzi_price_proof_l630_63073

-- Define the unit price of type B zongzi
def unit_price_B : ℝ := 4

-- Define the conditions
def amount_A : ℝ := 1200
def amount_B : ℝ := 800
def quantity_difference : ℕ := 50

-- Theorem statement
theorem zongzi_price_proof :
  -- Conditions
  (amount_A = (2 * unit_price_B) * ((amount_B / unit_price_B) - quantity_difference)) ∧
  (amount_B = unit_price_B * (amount_B / unit_price_B)) →
  -- Conclusion
  unit_price_B = 4 := by
  sorry


end NUMINAMATH_CALUDE_zongzi_price_proof_l630_63073


namespace NUMINAMATH_CALUDE_islander_liar_count_l630_63004

/-- Represents the type of islander: knight or liar -/
inductive IslanderType
| Knight
| Liar

/-- Represents a group of islanders making a statement -/
structure IslanderGroup where
  count : Nat
  statement : Nat

/-- The main theorem to prove -/
theorem islander_liar_count 
  (total_islanders : Nat)
  (group1 group2 group3 : IslanderGroup)
  (h1 : total_islanders = 19)
  (h2 : group1.count = 3 ∧ group1.statement = 3)
  (h3 : group2.count = 6 ∧ group2.statement = 6)
  (h4 : group3.count = 9 ∧ group3.statement = 9)
  (h5 : group1.count + group2.count + group3.count = total_islanders) :
  ∃ (liar_count : Nat), (liar_count = 9 ∨ liar_count = 18 ∨ liar_count = 19) ∧
    (∀ (x : Nat), x ≠ 9 ∧ x ≠ 18 ∧ x ≠ 19 → x ≠ liar_count) :=
by sorry

end NUMINAMATH_CALUDE_islander_liar_count_l630_63004


namespace NUMINAMATH_CALUDE_final_ratio_theorem_l630_63014

/-- Represents the final amount ratio between two players -/
structure FinalRatio where
  player1 : ℕ
  player2 : ℕ

/-- Represents a game with three players -/
structure Game where
  initialAmount : ℕ
  finalRatioAS : FinalRatio
  sGain : ℕ

theorem final_ratio_theorem (g : Game) 
  (h1 : g.initialAmount = 70)
  (h2 : g.finalRatioAS = FinalRatio.mk 1 2)
  (h3 : g.sGain = 50) :
  ∃ (finalRatioSB : FinalRatio), 
    finalRatioSB.player1 = 4 ∧ 
    finalRatioSB.player2 = 1 := by
  sorry

end NUMINAMATH_CALUDE_final_ratio_theorem_l630_63014


namespace NUMINAMATH_CALUDE_cube_volume_from_surface_area_l630_63010

theorem cube_volume_from_surface_area (surface_area : ℝ) (volume : ℝ) : 
  surface_area = 1734 → volume = (((surface_area / 6) ^ (1/2 : ℝ)) ^ 3) → volume = 4913 := by
  sorry

end NUMINAMATH_CALUDE_cube_volume_from_surface_area_l630_63010


namespace NUMINAMATH_CALUDE_binary_product_theorem_l630_63012

/-- Converts a binary number (represented as a list of bits) to a natural number. -/
def binaryToNat (bits : List Bool) : Nat :=
  bits.reverse.enum.foldl (fun acc (i, b) => acc + if b then 2^i else 0) 0

/-- Converts a natural number to its binary representation. -/
def natToBinary (n : Nat) : List Bool :=
  if n = 0 then [false] else
  let rec toBinary (m : Nat) : List Bool :=
    if m = 0 then [] else (m % 2 = 1) :: toBinary (m / 2)
  toBinary n

theorem binary_product_theorem :
  let a := [true, false, true, true]  -- 1101₂
  let b := [true, true, true]         -- 111₂
  let c := [true, true, true, true, false, false, true]  -- 1001111₂
  binaryToNat a * binaryToNat b = binaryToNat c := by
  sorry

end NUMINAMATH_CALUDE_binary_product_theorem_l630_63012


namespace NUMINAMATH_CALUDE_cost_price_calculation_l630_63082

-- Define the selling price
def selling_price : ℚ := 715

-- Define the profit percentage
def profit_percentage : ℚ := 10 / 100

-- Define the cost price
def cost_price : ℚ := 650

-- Theorem to prove
theorem cost_price_calculation :
  cost_price = selling_price / (1 + profit_percentage) :=
sorry

end NUMINAMATH_CALUDE_cost_price_calculation_l630_63082


namespace NUMINAMATH_CALUDE_tan_sum_special_angles_l630_63033

theorem tan_sum_special_angles : 
  Real.tan (π / 12) + Real.tan (5 * π / 12) = 4 * Real.sqrt 3 / 3 := by
  sorry

end NUMINAMATH_CALUDE_tan_sum_special_angles_l630_63033


namespace NUMINAMATH_CALUDE_grade_assignment_count_l630_63077

theorem grade_assignment_count : (4 : ℕ) ^ 15 = 1073741824 := by
  sorry

end NUMINAMATH_CALUDE_grade_assignment_count_l630_63077


namespace NUMINAMATH_CALUDE_turquoise_tile_cost_l630_63017

/-- Proves that the cost of each turquoise tile is $13 given the problem conditions -/
theorem turquoise_tile_cost :
  ∀ (total_area : ℝ) (tiles_per_sqft : ℝ) (purple_cost : ℝ) (savings : ℝ),
    total_area = 96 →
    tiles_per_sqft = 4 →
    purple_cost = 11 →
    savings = 768 →
    ∃ (turquoise_cost : ℝ),
      turquoise_cost = 13 ∧
      (total_area * tiles_per_sqft) * turquoise_cost - (total_area * tiles_per_sqft) * purple_cost = savings :=
by
  sorry


end NUMINAMATH_CALUDE_turquoise_tile_cost_l630_63017


namespace NUMINAMATH_CALUDE_sin_cos_tan_order_l630_63059

theorem sin_cos_tan_order :
  ∃ (a b c : ℝ), a = Real.sin 2 ∧ b = Real.cos 2 ∧ c = Real.tan 2 ∧ c < b ∧ b < a := by
  sorry

end NUMINAMATH_CALUDE_sin_cos_tan_order_l630_63059


namespace NUMINAMATH_CALUDE_remaining_soup_feeds_twenty_adults_l630_63008

/-- Represents the number of adults a can of soup can feed -/
def adults_per_can : ℕ := 4

/-- Represents the number of children a can of soup can feed -/
def children_per_can : ℕ := 7

/-- Represents the total number of cans of soup -/
def total_cans : ℕ := 10

/-- Represents the number of children fed -/
def children_fed : ℕ := 35

/-- Calculates the number of adults that can be fed with the remaining soup -/
def adults_fed_with_remaining_soup : ℕ := 
  adults_per_can * (total_cans - (children_fed / children_per_can))

theorem remaining_soup_feeds_twenty_adults : 
  adults_fed_with_remaining_soup = 20 := by sorry

end NUMINAMATH_CALUDE_remaining_soup_feeds_twenty_adults_l630_63008


namespace NUMINAMATH_CALUDE_triangle_intersection_coord_diff_l630_63026

/-- Triangle ABC with vertices A(0,10), B(3,-1), C(9,-1) -/
structure Triangle :=
  (A : ℝ × ℝ)
  (B : ℝ × ℝ)
  (C : ℝ × ℝ)

/-- Point R on line AC and S on line BC -/
structure IntersectionPoints (T : Triangle) :=
  (R : ℝ × ℝ)
  (S : ℝ × ℝ)

/-- The area of triangle RSC -/
def areaRSC (T : Triangle) (I : IntersectionPoints T) : ℝ := sorry

/-- The positive difference between x and y coordinates of R -/
def coordDiffR (I : IntersectionPoints T) : ℝ :=
  |I.R.1 - I.R.2|

theorem triangle_intersection_coord_diff 
  (T : Triangle) 
  (hA : T.A = (0, 10)) 
  (hB : T.B = (3, -1)) 
  (hC : T.C = (9, -1)) 
  (I : IntersectionPoints T) 
  (hvert : I.R.1 = I.S.1) -- R and S on the same vertical line
  (harea : areaRSC T I = 20) :
  coordDiffR I = 50/9 := by sorry

end NUMINAMATH_CALUDE_triangle_intersection_coord_diff_l630_63026


namespace NUMINAMATH_CALUDE_symmetry_about_center_three_zeros_existence_l630_63092

-- Define the function f(x)
def f (a b x : ℝ) : ℝ := x^3 - a*x^2 + b*x + 1

-- Theorem for symmetry (Option B)
theorem symmetry_about_center (b : ℝ) :
  ∀ x : ℝ, f 0 b x + f 0 b (-x) = 2 :=
sorry

-- Theorem for three zeros (Option C)
theorem three_zeros_existence (a : ℝ) (h : a > -4) :
  ∃ x y z : ℝ, x ≠ y ∧ y ≠ z ∧ x ≠ z ∧
  f a (a^2/4) x = 0 ∧ f a (a^2/4) y = 0 ∧ f a (a^2/4) z = 0 :=
sorry

end NUMINAMATH_CALUDE_symmetry_about_center_three_zeros_existence_l630_63092


namespace NUMINAMATH_CALUDE_village_children_average_l630_63006

/-- Given a village with families and children, calculates the average number of children in families with children -/
def average_children_in_families_with_children (total_families : ℕ) (total_average : ℚ) (childless_families : ℕ) : ℚ :=
  let total_children := total_families * total_average
  let families_with_children := total_families - childless_families
  total_children / families_with_children

/-- Proves that in a village with 12 families, an average of 3 children per family, and 3 childless families, 
    the average number of children in families with children is 4.0 -/
theorem village_children_average : average_children_in_families_with_children 12 3 3 = 4 := by
  sorry

end NUMINAMATH_CALUDE_village_children_average_l630_63006


namespace NUMINAMATH_CALUDE_initial_hno3_concentration_l630_63062

/-- Proves that the initial concentration of HNO3 is 35% given the problem conditions -/
theorem initial_hno3_concentration
  (initial_volume : ℝ)
  (pure_hno3_added : ℝ)
  (final_concentration : ℝ)
  (h1 : initial_volume = 60)
  (h2 : pure_hno3_added = 18)
  (h3 : final_concentration = 50)
  : ∃ (initial_concentration : ℝ),
    initial_concentration = 35 ∧
    (initial_concentration / 100) * initial_volume + pure_hno3_added =
    (final_concentration / 100) * (initial_volume + pure_hno3_added) :=
by sorry

end NUMINAMATH_CALUDE_initial_hno3_concentration_l630_63062


namespace NUMINAMATH_CALUDE_triangle_area_l630_63048

/-- Given a triangle with perimeter 36 and inradius 2.5, its area is 45 -/
theorem triangle_area (p : ℝ) (r : ℝ) (A : ℝ) 
  (h1 : p = 36) -- perimeter is 36
  (h2 : r = 2.5) -- inradius is 2.5
  (h3 : A = r * (p / 2)) -- area formula: A = r * s, where s is semiperimeter (p / 2)
  : A = 45 := by
  sorry

end NUMINAMATH_CALUDE_triangle_area_l630_63048


namespace NUMINAMATH_CALUDE_amandas_family_painting_l630_63035

/-- The number of walls each person should paint in Amanda's family house painting problem -/
theorem amandas_family_painting (
  total_rooms : ℕ)
  (rooms_with_four_walls : ℕ)
  (rooms_with_five_walls : ℕ)
  (family_size : ℕ)
  (h1 : total_rooms = rooms_with_four_walls + rooms_with_five_walls)
  (h2 : total_rooms = 9)
  (h3 : rooms_with_four_walls = 5)
  (h4 : rooms_with_five_walls = 4)
  (h5 : family_size = 5)
  : (4 * rooms_with_four_walls + 5 * rooms_with_five_walls) / family_size = 8 := by
  sorry

end NUMINAMATH_CALUDE_amandas_family_painting_l630_63035


namespace NUMINAMATH_CALUDE_chord_length_implies_a_value_l630_63064

theorem chord_length_implies_a_value (a : ℝ) :
  (∃ (x y : ℝ), (a * x + y + 1 = 0) ∧ (x^2 + y^2 - 2*a*x + a = 0)) →
  (∃ (x₁ y₁ x₂ y₂ : ℝ), 
    (a * x₁ + y₁ + 1 = 0) ∧ (x₁^2 + y₁^2 - 2*a*x₁ + a = 0) ∧
    (a * x₂ + y₂ + 1 = 0) ∧ (x₂^2 + y₂^2 - 2*a*x₂ + a = 0) ∧
    ((x₁ - x₂)^2 + (y₁ - y₂)^2 = 4)) →
  a = -2 :=
sorry


end NUMINAMATH_CALUDE_chord_length_implies_a_value_l630_63064


namespace NUMINAMATH_CALUDE_monica_reading_ratio_l630_63081

/-- The number of books Monica read last year -/
def last_year : ℕ := 16

/-- The number of books Monica will read next year -/
def next_year : ℕ := 69

/-- The number of books Monica read this year -/
def this_year : ℕ := last_year * 2

theorem monica_reading_ratio :
  (this_year / last_year : ℚ) = 2 := by
  sorry

end NUMINAMATH_CALUDE_monica_reading_ratio_l630_63081


namespace NUMINAMATH_CALUDE_E_Z_eq_l630_63034

variable (p : ℝ)

-- Assumption that p is a probability
axiom h_p_prob : 0 < p ∧ p < 1

-- Definition of the probability mass function for Y
def P_Y (k : ℕ) : ℝ := p * (1 - p) ^ (k - 1)

-- Definition of the probability mass function for Z
def P_Z (k : ℕ) : ℝ := 
  if k ≥ 2 then p * (1 - p) ^ (k - 1) + (1 - p) * p ^ (k - 1) else 0

-- Expected value of Y
axiom E_Y : ∑' k, k * P_Y p k = 1 / p

-- Theorem to prove
theorem E_Z_eq : ∑' k, k * P_Z p k = 1 / (p * (1 - p)) - 1 := by sorry

end NUMINAMATH_CALUDE_E_Z_eq_l630_63034


namespace NUMINAMATH_CALUDE_y_intercept_of_parallel_line_l630_63045

/-- A line in the plane can be represented by its slope and a point it passes through. -/
structure Line where
  slope : ℝ
  point : ℝ × ℝ

/-- The y-intercept of a line is the y-coordinate of the point where the line intersects the y-axis. -/
def y_intercept (l : Line) : ℝ :=
  l.point.2 - l.slope * l.point.1

/-- Two lines are parallel if and only if they have the same slope. -/
def parallel (l1 l2 : Line) : Prop :=
  l1.slope = l2.slope

theorem y_intercept_of_parallel_line (b : Line) :
  parallel b { slope := 3, point := (0, 2) } →
  b.point = (5, 10) →
  y_intercept b = -5 := by
  sorry

#check y_intercept_of_parallel_line

end NUMINAMATH_CALUDE_y_intercept_of_parallel_line_l630_63045


namespace NUMINAMATH_CALUDE_probability_black_ball_l630_63019

/-- The probability of drawing a black ball from a bag of colored balls. -/
theorem probability_black_ball (total : ℕ) (red : ℕ) (white : ℕ) (black : ℕ) :
  total = red + white + black →
  total = 6 →
  red = 1 →
  white = 2 →
  black = 3 →
  (black : ℚ) / total = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_probability_black_ball_l630_63019


namespace NUMINAMATH_CALUDE_data_tape_cost_calculation_l630_63054

/-- The cost of mounting a data tape for a computer program run. -/
def data_tape_cost : ℝ := 5.35

/-- The operating-system overhead cost per run. -/
def os_overhead_cost : ℝ := 1.07

/-- The cost of computer time per millisecond. -/
def computer_time_cost_per_ms : ℝ := 0.023

/-- The total cost for one run of the program. -/
def total_cost : ℝ := 40.92

/-- The duration of the computer program run in seconds. -/
def run_duration_seconds : ℝ := 1.5

theorem data_tape_cost_calculation :
  data_tape_cost = total_cost - (os_overhead_cost + computer_time_cost_per_ms * (run_duration_seconds * 1000)) :=
by sorry

end NUMINAMATH_CALUDE_data_tape_cost_calculation_l630_63054


namespace NUMINAMATH_CALUDE_exponential_function_implies_a_eq_three_l630_63080

/-- A function f: ℝ → ℝ is exponential if there exist constants b > 0, b ≠ 1, and c such that f(x) = c * b^x for all x ∈ ℝ. -/
def IsExponentialFunction (f : ℝ → ℝ) : Prop :=
  ∃ (b c : ℝ), b > 0 ∧ b ≠ 1 ∧ ∀ x, f x = c * b^x

/-- If f(x) = (a-2) * a^x is an exponential function, then a = 3. -/
theorem exponential_function_implies_a_eq_three (a : ℝ) :
  IsExponentialFunction (fun x ↦ (a - 2) * a^x) → a = 3 := by
  sorry

end NUMINAMATH_CALUDE_exponential_function_implies_a_eq_three_l630_63080


namespace NUMINAMATH_CALUDE_symmetry_sum_l630_63076

/-- Two points are symmetric with respect to the origin if their coordinates are negatives of each other -/
def symmetric_wrt_origin (A B : ℝ × ℝ) : Prop :=
  A.1 = -B.1 ∧ A.2 = -B.2

theorem symmetry_sum (a b : ℝ) : 
  symmetric_wrt_origin (a, 2) (4, b) → a + b = -6 := by
  sorry

end NUMINAMATH_CALUDE_symmetry_sum_l630_63076


namespace NUMINAMATH_CALUDE_train_speed_l630_63032

/-- The speed of a train given its length and time to cross a fixed point -/
theorem train_speed (length : ℝ) (time : ℝ) (h1 : length = 600) (h2 : time = 25) :
  length / time = 24 := by
  sorry

end NUMINAMATH_CALUDE_train_speed_l630_63032


namespace NUMINAMATH_CALUDE_solution_set_when_a_is_1_range_of_a_given_condition_l630_63061

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := |x - 2| + |3*x + a|

-- Part 1
theorem solution_set_when_a_is_1 :
  {x : ℝ | f 1 x ≥ 5} = {x : ℝ | x ≤ -1 ∨ x ≥ 1} := by sorry

-- Part 2
theorem range_of_a_given_condition (a : ℝ) :
  (∃ x₀ : ℝ, f a x₀ + 2*|x₀ - 2| < 3) → -9 < a ∧ a < -3 := by sorry

end NUMINAMATH_CALUDE_solution_set_when_a_is_1_range_of_a_given_condition_l630_63061


namespace NUMINAMATH_CALUDE_john_feeds_twice_daily_l630_63021

/-- Represents the scenario of John feeding his horses -/
structure HorseFeeding where
  num_horses : ℕ
  food_per_feeding : ℕ
  days : ℕ
  bags_bought : ℕ
  bag_weight : ℕ

/-- Calculates the number of feedings per horse per day -/
def feedings_per_horse_per_day (hf : HorseFeeding) : ℚ :=
  let total_food := hf.bags_bought * hf.bag_weight
  let food_per_day := total_food / hf.days
  let feedings_per_day := food_per_day / hf.food_per_feeding
  feedings_per_day / hf.num_horses

/-- Theorem stating that John feeds each horse twice a day -/
theorem john_feeds_twice_daily : 
  ∀ (hf : HorseFeeding), 
    hf.num_horses = 25 → 
    hf.food_per_feeding = 20 → 
    hf.days = 60 → 
    hf.bags_bought = 60 → 
    hf.bag_weight = 1000 → 
    feedings_per_horse_per_day hf = 2 := by
  sorry


end NUMINAMATH_CALUDE_john_feeds_twice_daily_l630_63021


namespace NUMINAMATH_CALUDE_mod_23_equivalence_l630_63000

theorem mod_23_equivalence : ∃! n : ℤ, 0 ≤ n ∧ n < 23 ∧ 57846 ≡ n [ZMOD 23] ∧ n = 1 := by
  sorry

end NUMINAMATH_CALUDE_mod_23_equivalence_l630_63000


namespace NUMINAMATH_CALUDE_parabola_intersection_l630_63084

/-- Two parabolas intersect at exactly two points -/
theorem parabola_intersection :
  let f (x : ℝ) := 3 * x^2 - 12 * x - 5
  let g (x : ℝ) := x^2 - 2 * x + 3
  ∃! (s : Set (ℝ × ℝ)), s = {(1, -14), (4, -5)} ∧ 
    ∀ (x y : ℝ), (x, y) ∈ s ↔ f x = g x ∧ y = f x := by
  sorry

end NUMINAMATH_CALUDE_parabola_intersection_l630_63084


namespace NUMINAMATH_CALUDE_circle_triangle_count_l630_63047

def points : ℕ := 9

def total_combinations : ℕ := Nat.choose points 3

def consecutive_triangles : ℕ := points

def valid_triangles : ℕ := total_combinations - consecutive_triangles

theorem circle_triangle_count :
  valid_triangles = 75 := by sorry

end NUMINAMATH_CALUDE_circle_triangle_count_l630_63047


namespace NUMINAMATH_CALUDE_factorization_of_2a_squared_minus_8_l630_63083

theorem factorization_of_2a_squared_minus_8 (a : ℝ) : 2 * a^2 - 8 = 2 * (a + 2) * (a - 2) := by
  sorry

end NUMINAMATH_CALUDE_factorization_of_2a_squared_minus_8_l630_63083


namespace NUMINAMATH_CALUDE_marble_selection_ways_l630_63056

def total_marbles : ℕ := 20
def red_marbles : ℕ := 3
def green_marbles : ℕ := 3
def blue_marbles : ℕ := 2
def special_marbles : ℕ := red_marbles + green_marbles + blue_marbles
def other_marbles : ℕ := total_marbles - special_marbles
def chosen_marbles : ℕ := 5
def chosen_special : ℕ := 2

theorem marble_selection_ways :
  (Nat.choose red_marbles 2 +
   Nat.choose green_marbles 2 +
   Nat.choose blue_marbles 2 +
   Nat.choose red_marbles 1 * Nat.choose green_marbles 1 +
   Nat.choose red_marbles 1 * Nat.choose blue_marbles 1 +
   Nat.choose green_marbles 1 * Nat.choose blue_marbles 1) *
  Nat.choose other_marbles (chosen_marbles - chosen_special) = 6160 := by
  sorry

end NUMINAMATH_CALUDE_marble_selection_ways_l630_63056


namespace NUMINAMATH_CALUDE_job_completion_time_equivalence_l630_63053

/-- Represents the number of days required to complete a job -/
def days_to_complete (num_men : ℕ) (man_days : ℕ) : ℚ :=
  man_days / num_men

theorem job_completion_time_equivalence :
  let initial_men : ℕ := 30
  let initial_days : ℕ := 8
  let new_men : ℕ := 40
  let man_days : ℕ := initial_men * initial_days
  days_to_complete new_men man_days = 6 := by
  sorry

end NUMINAMATH_CALUDE_job_completion_time_equivalence_l630_63053


namespace NUMINAMATH_CALUDE_infinite_pairs_exist_l630_63066

theorem infinite_pairs_exist (m : ℕ+) :
  ∃ f : ℕ → ℕ × ℕ,
    Function.Injective f ∧
    ∀ n : ℕ,
      let (x, y) := f n
      Nat.gcd x y = 1 ∧
      y ∣ (x^2 + m) ∧
      x ∣ (y^2 + m) :=
by
  sorry

end NUMINAMATH_CALUDE_infinite_pairs_exist_l630_63066


namespace NUMINAMATH_CALUDE_vertical_angles_equal_l630_63088

/-- Two lines in a plane -/
structure Line :=
  (slope : ℝ)
  (intercept : ℝ)

/-- An angle formed by two lines -/
structure Angle :=
  (line1 : Line)
  (line2 : Line)

/-- Two lines intersect -/
def intersect (l1 l2 : Line) : Prop :=
  l1.slope ≠ l2.slope

/-- Vertical angles are formed when two lines intersect -/
def vertical_angles (a1 a2 : Angle) : Prop :=
  ∃ (l1 l2 : Line), intersect l1 l2 ∧ 
    ((a1.line1 = l1 ∧ a1.line2 = l2 ∧ a2.line1 = l2 ∧ a2.line2 = l1) ∨
     (a1.line1 = l2 ∧ a1.line2 = l1 ∧ a2.line1 = l1 ∧ a2.line2 = l2))

/-- Angles are equal -/
def angle_equal (a1 a2 : Angle) : Prop :=
  sorry  -- Definition of angle equality

/-- Theorem: Vertical angles are equal -/
theorem vertical_angles_equal (a1 a2 : Angle) : 
  vertical_angles a1 a2 → angle_equal a1 a2 := by
  sorry

end NUMINAMATH_CALUDE_vertical_angles_equal_l630_63088


namespace NUMINAMATH_CALUDE_arithmetic_square_root_of_sqrt_81_l630_63075

theorem arithmetic_square_root_of_sqrt_81 : Real.sqrt (Real.sqrt 81) = 3 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_square_root_of_sqrt_81_l630_63075


namespace NUMINAMATH_CALUDE_division_relation_l630_63013

theorem division_relation (D : ℝ) (h : D > 0) :
  let d := D / 35
  let q := D / 5
  q = D / 5 ∧ q = 7 * d := by sorry

end NUMINAMATH_CALUDE_division_relation_l630_63013


namespace NUMINAMATH_CALUDE_obtuse_triangle_consecutive_sides_l630_63096

theorem obtuse_triangle_consecutive_sides :
  ∀ (a b c : ℕ), 
    (a < b) → 
    (b < c) → 
    (c = a + 2) → 
    (a^2 + b^2 < c^2) → 
    (a = 2 ∧ b = 3 ∧ c = 4) :=
by sorry

end NUMINAMATH_CALUDE_obtuse_triangle_consecutive_sides_l630_63096


namespace NUMINAMATH_CALUDE_set_closure_under_difference_l630_63046

theorem set_closure_under_difference (A : Set ℝ) 
  (h1 : ∀ a ∈ A, (2 * a) ∈ A) 
  (h2 : ∀ a b, a ∈ A → b ∈ A → (a + b) ∈ A) : 
  ∀ x y, x ∈ A → y ∈ A → (x - y) ∈ A := by
  sorry

end NUMINAMATH_CALUDE_set_closure_under_difference_l630_63046


namespace NUMINAMATH_CALUDE_circle_center_l630_63086

/-- The equation of a circle in the form (x - h)² + (y - k)² = r² 
    where (h, k) is the center and r is the radius -/
def CircleEquation (h k r : ℝ) (x y : ℝ) : Prop :=
  (x - h)^2 + (y - k)^2 = r^2

/-- The given equation of the circle -/
def GivenEquation (x y : ℝ) : Prop :=
  x^2 - 8*x + y^2 - 4*y = 16

theorem circle_center : 
  ∃ (r : ℝ), ∀ (x y : ℝ), GivenEquation x y ↔ CircleEquation 4 2 r x y :=
sorry

end NUMINAMATH_CALUDE_circle_center_l630_63086


namespace NUMINAMATH_CALUDE_imaginary_part_of_i_times_one_plus_i_l630_63049

theorem imaginary_part_of_i_times_one_plus_i : 
  Complex.im (Complex.I * (1 + Complex.I)) = 1 := by sorry

end NUMINAMATH_CALUDE_imaginary_part_of_i_times_one_plus_i_l630_63049


namespace NUMINAMATH_CALUDE_square_difference_formula_l630_63011

theorem square_difference_formula (x y : ℚ) 
  (sum_eq : x + y = 8/15)
  (diff_eq : x - y = 2/15) :
  x^2 - y^2 = 16/225 := by
sorry

end NUMINAMATH_CALUDE_square_difference_formula_l630_63011


namespace NUMINAMATH_CALUDE_average_of_expressions_l630_63070

theorem average_of_expressions (x : ℚ) : 
  (1/3 : ℚ) * ((2*x + 8) + (5*x + 3) + (3*x + 9)) = 3*x + 2 → x = -14 := by
sorry

end NUMINAMATH_CALUDE_average_of_expressions_l630_63070


namespace NUMINAMATH_CALUDE_rod_and_rope_problem_l630_63072

theorem rod_and_rope_problem (x y : ℝ) : 
  (x - y = 5 ∧ y - (1/2) * x = 5) ↔ 
  (x > y ∧ x - y = 5 ∧ y > (1/2) * x ∧ y - (1/2) * x = 5) :=
sorry

end NUMINAMATH_CALUDE_rod_and_rope_problem_l630_63072


namespace NUMINAMATH_CALUDE_cube_sum_equality_l630_63025

theorem cube_sum_equality (a b c : ℕ+) (h : a^3 + b^3 + c^3 = 3*a*b*c) : a = b ∧ b = c := by
  sorry

end NUMINAMATH_CALUDE_cube_sum_equality_l630_63025


namespace NUMINAMATH_CALUDE_principal_amount_proof_l630_63031

/-- Proves that given specific interest conditions, the principal amount is 6400 --/
theorem principal_amount_proof (rate : ℝ) (time : ℝ) (difference : ℝ) : rate = 0.05 → time = 2 → difference = 16 → 
  ∃ (principal : ℝ), principal * ((1 + rate)^time - 1 - rate * time) = difference ∧ principal = 6400 := by
  sorry

end NUMINAMATH_CALUDE_principal_amount_proof_l630_63031


namespace NUMINAMATH_CALUDE_max_time_between_happy_moments_l630_63041

/-- A happy moment on a 24-hour digital clock --/
structure HappyMoment where
  hours : Fin 24
  minutes : Fin 60
  is_happy : (hours = 6 * minutes) ∨ (minutes = 6 * hours)

/-- The time difference between two happy moments in minutes --/
def time_difference (h1 h2 : HappyMoment) : ℕ :=
  let total_minutes1 := h1.hours * 60 + h1.minutes
  let total_minutes2 := h2.hours * 60 + h2.minutes
  if total_minutes2 ≥ total_minutes1 then
    total_minutes2 - total_minutes1
  else
    (24 * 60) - (total_minutes1 - total_minutes2)

/-- Theorem stating the maximum time difference between consecutive happy moments --/
theorem max_time_between_happy_moments :
  ∃ (max : ℕ), max = 361 ∧
  ∀ (h1 h2 : HappyMoment), time_difference h1 h2 ≤ max :=
sorry

end NUMINAMATH_CALUDE_max_time_between_happy_moments_l630_63041


namespace NUMINAMATH_CALUDE_sum_of_absolute_coefficients_l630_63094

-- Define the polynomial coefficients
variable (a₀ a₁ a₂ a₃ a₄ a₅ a₆ : ℝ)

-- Define the polynomial equation
def polynomial_equation (x : ℝ) : Prop :=
  (2 - x)^6 = a₀ + a₁*x + a₂*x^2 + a₃*x^3 + a₄*x^4 + a₅*x^5 + a₆*x^6

-- State the theorem
theorem sum_of_absolute_coefficients :
  (∀ x, polynomial_equation a₀ a₁ a₂ a₃ a₄ a₅ a₆ x) →
  |a₁| + |a₂| + |a₃| + |a₄| + |a₅| + |a₆| = 665 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_absolute_coefficients_l630_63094


namespace NUMINAMATH_CALUDE_area_above_line_ratio_l630_63093

/-- Given two positive real numbers a and b, where a > b > 1/2 * a,
    and two squares with side lengths a and b placed next to each other,
    with the larger square having its lower left corner at (0,0) and
    the smaller square having its lower left corner at (a,0),
    if the area above the line passing through (0,a) and (a+b,0) in both squares is 2013,
    and (a,b) is the unique pair maximizing a+b,
    then a/b = ∛5√3. -/
theorem area_above_line_ratio (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a > b)
  (hba : b > (1/2) * a) (harea : (a^3 / (2*(a+b))) + (a*b/2) = 2013)
  (hmax : ∀ (x y : ℝ), x > 0 → y > 0 → x > y → y > (1/2) * x →
    (x^3 / (2*(x+y))) + (x*y/2) = 2013 → x + y ≤ a + b) :
  a / b = (3 : ℝ)^(1/5) :=
sorry

end NUMINAMATH_CALUDE_area_above_line_ratio_l630_63093


namespace NUMINAMATH_CALUDE_candidate_vote_percentage_l630_63052

theorem candidate_vote_percentage
  (total_votes : ℕ)
  (invalid_percentage : ℚ)
  (candidate_valid_votes : ℕ)
  (h1 : total_votes = 560000)
  (h2 : invalid_percentage = 15 / 100)
  (h3 : candidate_valid_votes = 357000) :
  (candidate_valid_votes : ℚ) / ((1 - invalid_percentage) * total_votes) = 75 / 100 := by
  sorry

end NUMINAMATH_CALUDE_candidate_vote_percentage_l630_63052


namespace NUMINAMATH_CALUDE_tube_length_doubles_pressure_l630_63007

/-- The length of a vertical tube required to double the pressure at the bottom of a water-filled barrel -/
theorem tube_length_doubles_pressure 
  (h₁ : ℝ) -- Initial height of water in the barrel
  (m : ℝ) -- Mass of water in the barrel
  (a : ℝ) -- Cross-sectional area of the tube
  (ρ : ℝ) -- Density of water
  (g : ℝ) -- Acceleration due to gravity
  (h₁_val : h₁ = 1.5) -- Given height of the barrel
  (m_val : m = 1000) -- Given mass of water
  (a_val : a = 1e-4) -- Given cross-sectional area (1 cm² = 1e-4 m²)
  (ρ_val : ρ = 1000) -- Given density of water
  : ∃ (h₂ : ℝ), h₂ = h₁ ∧ ρ * g * (h₁ + h₂) = 2 * (ρ * g * h₁) :=
by sorry

end NUMINAMATH_CALUDE_tube_length_doubles_pressure_l630_63007


namespace NUMINAMATH_CALUDE_complex_condition_implies_a_value_l630_63015

theorem complex_condition_implies_a_value (a : ℝ) :
  (((a : ℂ) + Complex.I) * (2 * Complex.I)).re > 0 → a = -1 := by
  sorry

end NUMINAMATH_CALUDE_complex_condition_implies_a_value_l630_63015


namespace NUMINAMATH_CALUDE_function_difference_l630_63051

-- Define the functions f and g
def f (x : ℝ) : ℝ := 4 * x^2 - 3 * x + 5
def g (k : ℝ) (x : ℝ) : ℝ := x^2 + k * x - 8

-- State the theorem
theorem function_difference (k : ℝ) : 
  f 5 - g k 5 = 20 → k = 53 / 5 := by
  sorry

end NUMINAMATH_CALUDE_function_difference_l630_63051


namespace NUMINAMATH_CALUDE_square_side_length_l630_63036

theorem square_side_length (A : ℝ) (s : ℝ) (h : A = 9) (h₁ : A = s^2) :
  s = Real.sqrt 9 := by
  sorry

end NUMINAMATH_CALUDE_square_side_length_l630_63036


namespace NUMINAMATH_CALUDE_unique_number_l630_63016

def is_two_digit (n : ℕ) : Prop := 10 ≤ n ∧ n ≤ 99

def is_odd (n : ℕ) : Prop := ∃ k : ℕ, n = 2*k + 1

def is_multiple_of_9 (n : ℕ) : Prop := ∃ k : ℕ, n = 9*k

def digit_product (n : ℕ) : ℕ := (n / 10) * (n % 10)

def is_perfect_square (n : ℕ) : Prop := ∃ k : ℕ, n = k^2

theorem unique_number : 
  ∃! n : ℕ, is_two_digit n ∧ 
            is_odd n ∧ 
            is_multiple_of_9 n ∧ 
            is_perfect_square (digit_product n) ∧
            n = 99 := by
  sorry

end NUMINAMATH_CALUDE_unique_number_l630_63016


namespace NUMINAMATH_CALUDE_hyperbola_equation_l630_63037

/-- Represents a hyperbola with center at the origin -/
structure Hyperbola where
  /-- Distance from center to focus -/
  c : ℝ
  /-- Ratio of b to a in the standard equation -/
  b_over_a : ℝ

/-- The standard equation of a hyperbola -/
def standard_equation (h : Hyperbola) (x y : ℝ) : Prop :=
  ∃ (a b : ℝ), a > 0 ∧ b > 0 ∧
    h.b_over_a = b / a ∧
    h.c^2 = a^2 + b^2 ∧
    x^2 / a^2 - y^2 / b^2 = 1

theorem hyperbola_equation (h : Hyperbola) (h_focus : h.c = 10) (h_asymptote : h.b_over_a = 4/3) :
  standard_equation h x y ↔ x^2 / 36 - y^2 / 64 = 1 :=
sorry

end NUMINAMATH_CALUDE_hyperbola_equation_l630_63037


namespace NUMINAMATH_CALUDE_f_has_max_and_min_iff_m_in_range_l630_63018

/-- The function f with parameter m -/
def f (m : ℝ) (x : ℝ) : ℝ := x^3 + m*x^2 + (m + 6)*x + 1

/-- The derivative of f with respect to x -/
def f' (m : ℝ) (x : ℝ) : ℝ := 3*x^2 + 2*m*x + (m + 6)

/-- The discriminant of f' -/
def discriminant (m : ℝ) : ℝ := (2*m)^2 - 4*3*(m + 6)

theorem f_has_max_and_min_iff_m_in_range (m : ℝ) :
  (∃ (a b : ℝ), ∀ x, f m x ≤ f m a ∧ f m x ≥ f m b) ↔ 
  m < -3 ∨ m > 6 :=
sorry

end NUMINAMATH_CALUDE_f_has_max_and_min_iff_m_in_range_l630_63018


namespace NUMINAMATH_CALUDE_candle_burning_time_l630_63044

-- Define the original length of the candle
def original_length : ℝ := 12

-- Define the rate of decrease in length per minute
def rate_of_decrease : ℝ := 0.08

-- Define the function for remaining length after x minutes
def remaining_length (x : ℝ) : ℝ := original_length - rate_of_decrease * x

-- Theorem statement
theorem candle_burning_time :
  ∃ (max_time : ℝ), max_time = 150 ∧ remaining_length max_time = 0 :=
sorry

end NUMINAMATH_CALUDE_candle_burning_time_l630_63044


namespace NUMINAMATH_CALUDE_intersection_and_subset_condition_l630_63028

def A : Set ℝ := {x | ∃ y, y = Real.sqrt (6 + 5*x - x^2)}
def B (m : ℝ) : Set ℝ := {x | (x - 1 + m) * (x - 1 - m) ≤ 0}

theorem intersection_and_subset_condition :
  (∃ m : ℝ, m = 3 ∧ A ∩ B m = {x | -1 ≤ x ∧ x ≤ 4}) ∧
  (∀ m : ℝ, m > 0 → (A ⊆ B m → m ≥ 5)) := by sorry

end NUMINAMATH_CALUDE_intersection_and_subset_condition_l630_63028


namespace NUMINAMATH_CALUDE_hyperbola_eccentricity_l630_63090

/-- The eccentricity of a hyperbola with equation x^2/a^2 - y^2/b^2 = 1 (a > 0, b > 0)
    is √2, given that one of its asymptotes is parallel to the line x - y + 3 = 0. -/
theorem hyperbola_eccentricity (a b : ℝ) (ha : a > 0) (hb : b > 0) 
    (h_asymptote : ∃ (k : ℝ), b / a = k ∧ 1 = k) : 
    Real.sqrt ((a^2 + b^2) / a^2) = Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_hyperbola_eccentricity_l630_63090


namespace NUMINAMATH_CALUDE_exist_same_color_parallel_triangle_l630_63002

/-- Represents a color --/
inductive Color
  | Red
  | Green
  | Blue

/-- Represents a point in the triangle --/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents the coloring of vertices --/
def Coloring := Point → Color

/-- Represents the large equilateral triangle --/
structure LargeTriangle where
  side_length : ℝ
  division_count : ℕ
  coloring : Coloring

/-- Checks if three points form a triangle parallel to the original triangle --/
def is_parallel_triangle (p1 p2 p3 : Point) : Prop := sorry

/-- Main theorem --/
theorem exist_same_color_parallel_triangle (T : LargeTriangle) 
  (h1 : T.division_count = 3000) -- 9000000 small triangles means 3000 divisions per side
  (h2 : T.side_length > 0) :
  ∃ (c : Color) (p1 p2 p3 : Point),
    T.coloring p1 = c ∧
    T.coloring p2 = c ∧
    T.coloring p3 = c ∧
    is_parallel_triangle p1 p2 p3 :=
  sorry

end NUMINAMATH_CALUDE_exist_same_color_parallel_triangle_l630_63002


namespace NUMINAMATH_CALUDE_complex_equation_solution_l630_63068

theorem complex_equation_solution (z : ℂ) : z * (1 + Complex.I) = 2 * Complex.I ^ 2018 → z = -1 + Complex.I := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l630_63068


namespace NUMINAMATH_CALUDE_sqrt_nine_factorial_over_126_l630_63085

theorem sqrt_nine_factorial_over_126 : 
  Real.sqrt (Nat.factorial 9 / 126) = 8 * Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_nine_factorial_over_126_l630_63085


namespace NUMINAMATH_CALUDE_function_inequality_l630_63063

theorem function_inequality (f : ℝ → ℝ) 
  (h1 : ∀ x, f (2 - x) = f x) 
  (h2 : ∀ x ≥ 1, f x = Real.log x) : 
  f (1/2) < f 2 ∧ f 2 < f (1/3) := by
  sorry

end NUMINAMATH_CALUDE_function_inequality_l630_63063


namespace NUMINAMATH_CALUDE_trigonometric_equation_consequences_l630_63023

open Real

theorem trigonometric_equation_consequences (α : ℝ) 
  (h : sin (π - α) * cos (2*π - α) / (tan (π - α) * sin (π/2 + α) * cos (π/2 - α)) = 1/2) : 
  (cos α - 2*sin α) / (3*cos α + sin α) = 5 ∧ 
  1 - 2*sin α*cos α + cos α^2 = 2/5 := by
sorry

end NUMINAMATH_CALUDE_trigonometric_equation_consequences_l630_63023


namespace NUMINAMATH_CALUDE_test_total_points_l630_63003

theorem test_total_points (total_questions : ℕ) (four_point_questions : ℕ) 
  (h1 : total_questions = 40)
  (h2 : four_point_questions = 10) :
  let two_point_questions := total_questions - four_point_questions
  total_questions * 2 + four_point_questions * 2 = 100 :=
by sorry

end NUMINAMATH_CALUDE_test_total_points_l630_63003


namespace NUMINAMATH_CALUDE_mono_increasing_and_even_shift_implies_l630_63020

/-- A function that is monotonically increasing on [1,+∞) and f(x+1) is even -/
def MonoIncreasingAndEvenShift (f : ℝ → ℝ) : Prop :=
  (∀ x y, 1 ≤ x ∧ x ≤ y → f x ≤ f y) ∧
  (∀ x, f (x + 1) = f (-x + 1))

/-- Theorem: If f is monotonically increasing on [1,+∞) and f(x+1) is even, then f(-2) > f(2) -/
theorem mono_increasing_and_even_shift_implies (f : ℝ → ℝ) 
  (h : MonoIncreasingAndEvenShift f) : f (-2) > f 2 := by
  sorry

end NUMINAMATH_CALUDE_mono_increasing_and_even_shift_implies_l630_63020


namespace NUMINAMATH_CALUDE_soccer_team_age_mode_l630_63040

def player_ages : List ℕ := [17, 17, 18, 18, 16, 18, 17, 15, 18, 18, 17, 16, 18, 17, 18, 14]

def mode (l : List ℕ) : ℕ :=
  l.foldl (fun acc x => if l.count x > l.count acc then x else acc) 0

theorem soccer_team_age_mode :
  mode player_ages = 18 := by
  sorry

end NUMINAMATH_CALUDE_soccer_team_age_mode_l630_63040


namespace NUMINAMATH_CALUDE_min_team_a_size_l630_63098

theorem min_team_a_size : ∃ (a : ℕ), a > 0 ∧ 
  (∃ (b : ℕ), b > 0 ∧ b + 90 = 2 * (a - 90)) ∧
  (∃ (k : ℕ), a + k = 6 * (b - k)) ∧
  (∀ (a' : ℕ), a' > 0 → 
    (∃ (b' : ℕ), b' > 0 ∧ b' + 90 = 2 * (a' - 90)) →
    (∃ (k' : ℕ), a' + k' = 6 * (b' - k')) →
    a ≤ a') ∧
  a = 153 := by
sorry

end NUMINAMATH_CALUDE_min_team_a_size_l630_63098
