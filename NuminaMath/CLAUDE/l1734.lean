import Mathlib

namespace NUMINAMATH_CALUDE_gcd_power_minus_identity_gcd_power_minus_identity_general_l1734_173422

theorem gcd_power_minus_identity (a : ℕ) (h : a ≥ 2) : 
  13530 ∣ a^41 - a :=
sorry

/- More general version for any natural number n -/
theorem gcd_power_minus_identity_general (n : ℕ) (a : ℕ) (h : a ≥ 2) : 
  ∃ k : ℕ, k ∣ a^n - a :=
sorry

end NUMINAMATH_CALUDE_gcd_power_minus_identity_gcd_power_minus_identity_general_l1734_173422


namespace NUMINAMATH_CALUDE_nine_b_value_l1734_173404

theorem nine_b_value (a b : ℚ) (h1 : 8 * a + 3 * b = 0) (h2 : b - 3 = a) : 9 * b = 216 / 11 := by
  sorry

end NUMINAMATH_CALUDE_nine_b_value_l1734_173404


namespace NUMINAMATH_CALUDE_binomial_expansion_sum_l1734_173463

theorem binomial_expansion_sum (m : ℝ) : 
  (∃ a₀ a₁ a₂ a₃ a₄ a₅ a₆ : ℝ, 
    (∀ x : ℝ, (1 + m * x)^6 = a₀ + a₁ * x + a₂ * x^2 + a₃ * x^3 + a₄ * x^4 + a₅ * x^5 + a₆ * x^6) ∧
    (a₀ + a₁ + a₂ + a₃ + a₄ + a₅ + a₆ = 64)) → 
  m = 1 ∨ m = -3 := by
sorry

end NUMINAMATH_CALUDE_binomial_expansion_sum_l1734_173463


namespace NUMINAMATH_CALUDE_function_not_satisfying_condition_l1734_173427

theorem function_not_satisfying_condition :
  ∃ f : ℝ → ℝ, (∀ x, f x = x + 1) ∧ (∃ x, f (2 * x) ≠ 2 * f x) := by
  sorry

end NUMINAMATH_CALUDE_function_not_satisfying_condition_l1734_173427


namespace NUMINAMATH_CALUDE_longest_side_of_triangle_l1734_173479

theorem longest_side_of_triangle (y : ℝ) : 
  10 + (y + 6) + (3*y + 2) = 45 →
  max 10 (max (y + 6) (3*y + 2)) = 22.25 := by
sorry

end NUMINAMATH_CALUDE_longest_side_of_triangle_l1734_173479


namespace NUMINAMATH_CALUDE_binary_encodes_to_032239_l1734_173492

/-- Represents a mapping from characters to digits -/
def EncodeMap := Char → Nat

/-- The encoding scheme based on "MONITOR KEYBOARD" -/
def monitorKeyboardEncode : EncodeMap :=
  fun c => match c with
  | 'M' => 0
  | 'O' => 1
  | 'N' => 2
  | 'I' => 3
  | 'T' => 4
  | 'R' => 6
  | 'K' => 7
  | 'E' => 8
  | 'Y' => 9
  | 'B' => 0
  | 'A' => 2
  | 'D' => 4
  | _ => 0  -- Default case, should not be reached for valid inputs

/-- Encodes a string to a list of digits using the given encoding map -/
def encodeString (encode : EncodeMap) (s : String) : List Nat :=
  s.data.map encode

/-- Converts a list of digits to a natural number -/
def digitsToNat (digits : List Nat) : Nat :=
  digits.foldl (fun acc d => acc * 10 + d) 0

/-- The main theorem: BINARY encodes to 032239 -/
theorem binary_encodes_to_032239 :
  digitsToNat (encodeString monitorKeyboardEncode "BINARY") = 032239 := by
  sorry


end NUMINAMATH_CALUDE_binary_encodes_to_032239_l1734_173492


namespace NUMINAMATH_CALUDE_set_intersection_problem_l1734_173426

theorem set_intersection_problem (M N : Set ℕ) : 
  M = {1, 2, 3} → N = {2, 3, 4} → M ∩ N = {2, 3} := by
  sorry

end NUMINAMATH_CALUDE_set_intersection_problem_l1734_173426


namespace NUMINAMATH_CALUDE_sum_reciprocal_inequality_l1734_173421

theorem sum_reciprocal_inequality (a b c : ℝ) 
  (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_pos_c : 0 < c)
  (h_sum : a + b + c ≤ 3) : 
  1 / (a + 1) + 1 / (b + 1) + 1 / (c + 1) ≥ 3 / 2 := by
sorry

end NUMINAMATH_CALUDE_sum_reciprocal_inequality_l1734_173421


namespace NUMINAMATH_CALUDE_choose_four_from_thirteen_l1734_173481

theorem choose_four_from_thirteen : Nat.choose 13 4 = 715 := by sorry

end NUMINAMATH_CALUDE_choose_four_from_thirteen_l1734_173481


namespace NUMINAMATH_CALUDE_reunion_attendance_l1734_173420

/-- The number of people attending a family reunion. -/
def n : ℕ := sorry

/-- The age of the youngest person at the reunion. -/
def youngest_age : ℕ := sorry

/-- The age of the oldest person at the reunion. -/
def oldest_age : ℕ := sorry

/-- The sum of ages of all people at the reunion. -/
def total_age_sum : ℕ := sorry

/-- The average age of members excluding the oldest person is 18 years old. -/
axiom avg_without_oldest : (total_age_sum - oldest_age) / (n - 1) = 18

/-- The average age of members excluding the youngest person is 20 years old. -/
axiom avg_without_youngest : (total_age_sum - youngest_age) / (n - 1) = 20

/-- The age difference between the oldest and youngest person is 40 years. -/
axiom age_difference : oldest_age - youngest_age = 40

/-- The number of people attending the reunion is 21. -/
theorem reunion_attendance : n = 21 := by sorry

end NUMINAMATH_CALUDE_reunion_attendance_l1734_173420


namespace NUMINAMATH_CALUDE_lilies_per_centerpiece_is_six_l1734_173465

/-- Calculates the number of lilies per centerpiece given the following conditions:
  * There are 6 centerpieces
  * Each centerpiece uses 8 roses
  * Each centerpiece uses twice as many orchids as roses
  * The total budget is $2700
  * Each flower costs $15
-/
def lilies_per_centerpiece (num_centerpieces : ℕ) (roses_per_centerpiece : ℕ) 
  (orchid_ratio : ℕ) (total_budget : ℕ) (flower_cost : ℕ) : ℕ :=
  let total_roses := num_centerpieces * roses_per_centerpiece
  let total_orchids := num_centerpieces * roses_per_centerpiece * orchid_ratio
  let rose_orchid_cost := (total_roses + total_orchids) * flower_cost
  let remaining_budget := total_budget - rose_orchid_cost
  let total_lilies := remaining_budget / flower_cost
  total_lilies / num_centerpieces

/-- Theorem stating that given the specific conditions, the number of lilies per centerpiece is 6 -/
theorem lilies_per_centerpiece_is_six :
  lilies_per_centerpiece 6 8 2 2700 15 = 6 := by
  sorry

end NUMINAMATH_CALUDE_lilies_per_centerpiece_is_six_l1734_173465


namespace NUMINAMATH_CALUDE_min_matchsticks_removal_theorem_l1734_173455

/-- Represents a configuration of matchsticks forming triangles -/
structure MatchstickConfiguration where
  total_matchsticks : ℕ
  total_triangles : ℕ

/-- Represents the minimum number of matchsticks to remove -/
def min_matchsticks_to_remove (config : MatchstickConfiguration) : ℕ := sorry

/-- The theorem to be proved -/
theorem min_matchsticks_removal_theorem (config : MatchstickConfiguration) 
  (h1 : config.total_matchsticks = 42)
  (h2 : config.total_triangles = 38) :
  min_matchsticks_to_remove config ≥ 12 := by sorry

end NUMINAMATH_CALUDE_min_matchsticks_removal_theorem_l1734_173455


namespace NUMINAMATH_CALUDE_watch_sale_gain_percentage_l1734_173467

/-- Calculates the selling price given the cost price and loss percentage -/
def sellingPriceWithLoss (costPrice : ℚ) (lossPercentage : ℚ) : ℚ :=
  costPrice * (1 - lossPercentage / 100)

/-- Calculates the gain percentage given the cost price and selling price -/
def gainPercentage (costPrice : ℚ) (sellingPrice : ℚ) : ℚ :=
  (sellingPrice - costPrice) / costPrice * 100

theorem watch_sale_gain_percentage 
  (costPrice : ℚ) 
  (lossPercentage : ℚ) 
  (additionalAmount : ℚ) : 
  costPrice = 3000 →
  lossPercentage = 10 →
  additionalAmount = 540 →
  gainPercentage costPrice (sellingPriceWithLoss costPrice lossPercentage + additionalAmount) = 8 := by
  sorry

end NUMINAMATH_CALUDE_watch_sale_gain_percentage_l1734_173467


namespace NUMINAMATH_CALUDE_inequality_implies_max_a_l1734_173431

theorem inequality_implies_max_a (a : ℝ) : 
  (∀ x : ℝ, x^2 - 1 ≥ a * |x - 1|) → a ≤ -2 :=
by sorry

end NUMINAMATH_CALUDE_inequality_implies_max_a_l1734_173431


namespace NUMINAMATH_CALUDE_five_cubic_yards_to_cubic_feet_l1734_173458

/-- Conversion factor from yards to feet -/
def yards_to_feet : ℝ := 3

/-- The number of cubic yards we want to convert -/
def cubic_yards : ℝ := 5

/-- The theorem states that 5 cubic yards is equal to 135 cubic feet -/
theorem five_cubic_yards_to_cubic_feet : 
  cubic_yards * (yards_to_feet ^ 3) = 135 := by
  sorry

end NUMINAMATH_CALUDE_five_cubic_yards_to_cubic_feet_l1734_173458


namespace NUMINAMATH_CALUDE_unique_sums_count_l1734_173408

def bagA : Finset ℕ := {2, 3, 4}
def bagB : Finset ℕ := {3, 4, 5}

def possibleSums : Finset ℕ := (bagA.product bagB).image (fun p => p.1 + p.2)

theorem unique_sums_count : possibleSums.card = 5 := by sorry

end NUMINAMATH_CALUDE_unique_sums_count_l1734_173408


namespace NUMINAMATH_CALUDE_invitations_per_package_l1734_173430

theorem invitations_per_package 
  (total_packs : ℕ) 
  (total_invitations : ℕ) 
  (h1 : total_packs = 5)
  (h2 : total_invitations = 45) :
  total_invitations / total_packs = 9 :=
by sorry

end NUMINAMATH_CALUDE_invitations_per_package_l1734_173430


namespace NUMINAMATH_CALUDE_irene_worked_50_hours_l1734_173471

/-- Calculates the total hours worked given the regular hours, overtime hours, regular pay, overtime pay rate, and total income. -/
def total_hours_worked (regular_hours : ℕ) (regular_pay : ℕ) (overtime_rate : ℕ) (total_income : ℕ) : ℕ :=
  regular_hours + (total_income - regular_pay) / overtime_rate

/-- Proves that given the problem conditions, Irene worked 50 hours. -/
theorem irene_worked_50_hours (regular_hours : ℕ) (regular_pay : ℕ) (overtime_rate : ℕ) (total_income : ℕ)
  (h1 : regular_hours = 40)
  (h2 : regular_pay = 500)
  (h3 : overtime_rate = 20)
  (h4 : total_income = 700) :
  total_hours_worked regular_hours regular_pay overtime_rate total_income = 50 := by
  sorry

#eval total_hours_worked 40 500 20 700

end NUMINAMATH_CALUDE_irene_worked_50_hours_l1734_173471


namespace NUMINAMATH_CALUDE_product_mod_seventeen_l1734_173456

theorem product_mod_seventeen :
  (3001 * 3002 * 3003 * 3004 * 3005) % 17 = 14 := by
  sorry

end NUMINAMATH_CALUDE_product_mod_seventeen_l1734_173456


namespace NUMINAMATH_CALUDE_simplify_expression_l1734_173447

theorem simplify_expression : (3 + 1) * (3^2 + 1) * (3^4 + 1) * (3^8 + 1) = (1 / 2) * (3^16 - 1) := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l1734_173447


namespace NUMINAMATH_CALUDE_derivative_x_over_one_minus_cos_l1734_173440

/-- The derivative of x / (1 - cos x) is (1 - cos x - x * sin x) / (1 - cos x)^2 -/
theorem derivative_x_over_one_minus_cos (x : ℝ) :
  deriv (fun x => x / (1 - Real.cos x)) x = (1 - Real.cos x - x * Real.sin x) / (1 - Real.cos x)^2 :=
sorry

end NUMINAMATH_CALUDE_derivative_x_over_one_minus_cos_l1734_173440


namespace NUMINAMATH_CALUDE_magical_points_on_quadratic_unique_magical_point_condition_l1734_173453

/-- Definition of a magical point -/
def is_magical_point (x y : ℝ) : Prop := y = 2 * x

/-- The quadratic function y = x^2 - x - 4 -/
def quadratic_function (x : ℝ) : ℝ := x^2 - x - 4

/-- The generalized quadratic function y = tx^2 + (t-2)x - 4 -/
def generalized_quadratic_function (t x : ℝ) : ℝ := t * x^2 + (t - 2) * x - 4

theorem magical_points_on_quadratic :
  ∀ x y : ℝ, is_magical_point x y ∧ y = quadratic_function x ↔ (x = -1 ∧ y = -2) ∨ (x = 4 ∧ y = 8) :=
sorry

theorem unique_magical_point_condition :
  ∀ t : ℝ, t ≠ 0 →
  (∃! x y : ℝ, is_magical_point x y ∧ y = generalized_quadratic_function t x) ↔ t = -4 :=
sorry

end NUMINAMATH_CALUDE_magical_points_on_quadratic_unique_magical_point_condition_l1734_173453


namespace NUMINAMATH_CALUDE_shyne_plants_l1734_173468

/-- The number of eggplants that can be grown from one seed packet -/
def eggplants_per_packet : ℕ := 14

/-- The number of sunflowers that can be grown from one seed packet -/
def sunflowers_per_packet : ℕ := 10

/-- The number of eggplant seed packets Shyne bought -/
def eggplant_packets : ℕ := 4

/-- The number of sunflower seed packets Shyne bought -/
def sunflower_packets : ℕ := 6

/-- The total number of plants Shyne can grow -/
def total_plants : ℕ := eggplants_per_packet * eggplant_packets + sunflowers_per_packet * sunflower_packets

theorem shyne_plants : total_plants = 116 := by
  sorry

end NUMINAMATH_CALUDE_shyne_plants_l1734_173468


namespace NUMINAMATH_CALUDE_hair_cut_total_l1734_173480

theorem hair_cut_total (first_cut second_cut : ℝ) 
  (h1 : first_cut = 0.375)
  (h2 : second_cut = 0.5) : 
  first_cut + second_cut = 0.875 := by
  sorry

end NUMINAMATH_CALUDE_hair_cut_total_l1734_173480


namespace NUMINAMATH_CALUDE_trigonometric_identity_l1734_173437

theorem trigonometric_identity (A B C : Real) 
  (h : A + B + C = Real.pi) : 
  (Real.sin A + Real.sin B - Real.sin C) / (Real.sin A + Real.sin B + Real.sin C) = 
  Real.tan (A/2) * Real.tan (B/2) := by
  sorry

end NUMINAMATH_CALUDE_trigonometric_identity_l1734_173437


namespace NUMINAMATH_CALUDE_water_transfer_height_l1734_173493

/-- Represents a rectangular tank with given dimensions -/
structure Tank where
  length : ℝ
  width : ℝ
  height : ℝ

/-- Calculates the volume of water in a tank given the water height -/
def waterVolume (t : Tank) (waterHeight : ℝ) : ℝ :=
  t.length * t.width * waterHeight

/-- Calculates the base area of a tank -/
def baseArea (t : Tank) : ℝ :=
  t.length * t.width

/-- Represents the problem setup -/
structure ProblemSetup where
  tankA : Tank
  tankB : Tank
  waterHeightB : ℝ

/-- The main theorem to prove -/
theorem water_transfer_height (setup : ProblemSetup) 
  (h1 : setup.tankA = { length := 4, width := 3, height := 5 })
  (h2 : setup.tankB = { length := 4, width := 2, height := 8 })
  (h3 : setup.waterHeightB = 1.5) :
  (waterVolume setup.tankB setup.waterHeightB) / (baseArea setup.tankA) = 1 := by
  sorry

end NUMINAMATH_CALUDE_water_transfer_height_l1734_173493


namespace NUMINAMATH_CALUDE_valid_param_iff_l1734_173438

/-- A structure representing a vector parameterization of a line -/
structure VectorParam where
  x₀ : ℝ
  y₀ : ℝ
  a : ℝ
  b : ℝ

/-- The line equation y = 2x + 6 -/
def line_equation (x y : ℝ) : Prop := y = 2 * x + 6

/-- Predicate to check if a vector parameterization is valid for the line y = 2x + 6 -/
def is_valid_param (p : VectorParam) : Prop :=
  line_equation p.x₀ p.y₀ ∧ p.b = 2 * p.a

/-- Theorem stating the condition for a valid vector parameterization -/
theorem valid_param_iff (p : VectorParam) :
  is_valid_param p ↔
    (∀ t : ℝ, line_equation (p.x₀ + t * p.a) (p.y₀ + t * p.b)) :=
by sorry

end NUMINAMATH_CALUDE_valid_param_iff_l1734_173438


namespace NUMINAMATH_CALUDE_greatest_possible_award_l1734_173407

/-- The greatest possible individual award in a prize distribution problem --/
theorem greatest_possible_award (total_prize : ℝ) (num_winners : ℕ) (min_award : ℝ)
  (h1 : total_prize = 2000)
  (h2 : num_winners = 50)
  (h3 : min_award = 25)
  (h4 : (3 / 4 : ℝ) * total_prize = (2 / 5 : ℝ) * (num_winners : ℝ) * (greatest_award : ℝ)) :
  greatest_award = 775 := by
  sorry

end NUMINAMATH_CALUDE_greatest_possible_award_l1734_173407


namespace NUMINAMATH_CALUDE_floor_sum_equals_negative_one_l1734_173464

theorem floor_sum_equals_negative_one : ⌊(18.7 : ℝ)⌋ + ⌊(-18.7 : ℝ)⌋ = -1 := by
  sorry

end NUMINAMATH_CALUDE_floor_sum_equals_negative_one_l1734_173464


namespace NUMINAMATH_CALUDE_marbles_choice_count_l1734_173462

/-- The number of ways to choose marbles under specific conditions -/
def choose_marbles (total : ℕ) (red green blue : ℕ) (choose : ℕ) : ℕ :=
  let other := total - (red + green + blue)
  let color_pairs := (red * green + red * blue + green * blue)
  let remaining_choices := Nat.choose (other + red - 1 + green - 1 + blue - 1) (choose - 2)
  color_pairs * remaining_choices

/-- Theorem stating the number of ways to choose marbles under given conditions -/
theorem marbles_choice_count :
  choose_marbles 15 2 2 2 5 = 495 := by sorry

end NUMINAMATH_CALUDE_marbles_choice_count_l1734_173462


namespace NUMINAMATH_CALUDE_bonus_remainder_l1734_173403

theorem bonus_remainder (P : ℕ) (h : P % 5 = 2) : (3 * P) % 5 = 1 := by
  sorry

end NUMINAMATH_CALUDE_bonus_remainder_l1734_173403


namespace NUMINAMATH_CALUDE_laurens_mail_problem_l1734_173451

/-- Lauren's mail sending problem -/
theorem laurens_mail_problem (monday tuesday wednesday thursday : ℕ) :
  monday = 65 ∧
  tuesday > monday ∧
  wednesday = tuesday - 5 ∧
  thursday = wednesday + 15 ∧
  monday + tuesday + wednesday + thursday = 295 →
  tuesday - monday = 10 := by
sorry

end NUMINAMATH_CALUDE_laurens_mail_problem_l1734_173451


namespace NUMINAMATH_CALUDE_min_value_sum_squared_ratios_l1734_173460

theorem min_value_sum_squared_ratios (x y z : ℝ) 
  (hx : x > 0) (hy : y > 0) (hz : z > 0) (hsum : x + y + z = 9) :
  (x^2 + y^2) / (x + y) + (x^2 + z^2) / (x + z) + (y^2 + z^2) / (y + z) ≥ 9 := by
  sorry

end NUMINAMATH_CALUDE_min_value_sum_squared_ratios_l1734_173460


namespace NUMINAMATH_CALUDE_tangerine_tree_count_prove_tangerine_tree_count_l1734_173457

theorem tangerine_tree_count : ℕ → ℕ → ℕ → Prop :=
  fun pear_trees apple_trees tangerine_trees =>
    (pear_trees = 56) →
    (pear_trees = apple_trees + 18) →
    (tangerine_trees = apple_trees - 12) →
    (tangerine_trees = 26)

-- Proof
theorem prove_tangerine_tree_count :
  ∃ (pear_trees apple_trees tangerine_trees : ℕ),
    tangerine_tree_count pear_trees apple_trees tangerine_trees :=
by
  sorry

end NUMINAMATH_CALUDE_tangerine_tree_count_prove_tangerine_tree_count_l1734_173457


namespace NUMINAMATH_CALUDE_garden_scale_drawing_l1734_173478

/-- Represents the length in feet given a scale drawing measurement -/
def actualLength (scale : ℝ) (drawingLength : ℝ) : ℝ :=
  scale * drawingLength

theorem garden_scale_drawing :
  let scale : ℝ := 500  -- 1 inch represents 500 feet
  let drawingLength : ℝ := 6.5  -- length in the drawing is 6.5 inches
  actualLength scale drawingLength = 3250 := by
  sorry

end NUMINAMATH_CALUDE_garden_scale_drawing_l1734_173478


namespace NUMINAMATH_CALUDE_student_weight_l1734_173445

/-- Prove that the student's present weight is 71 kilograms -/
theorem student_weight (student_weight sister_weight : ℝ) 
  (h1 : student_weight - 5 = 2 * sister_weight)
  (h2 : student_weight + sister_weight = 104) :
  student_weight = 71 := by
  sorry

end NUMINAMATH_CALUDE_student_weight_l1734_173445


namespace NUMINAMATH_CALUDE_all_positive_integers_are_valid_l1734_173466

-- Define a coloring of the infinite grid
def Coloring := ℤ → ℤ → Bool

-- Define a rectangle on the grid
structure Rectangle where
  x : ℤ
  y : ℤ
  width : ℕ+
  height : ℕ+

-- Count the number of red cells in a rectangle
def countRedCells (c : Coloring) (r : Rectangle) : ℕ :=
  sorry

-- Define the property that all n-cell rectangles have an odd number of red cells
def validColoring (n : ℕ+) (c : Coloring) : Prop :=
  ∀ r : Rectangle, r.width * r.height = n → Odd (countRedCells c r)

-- The main theorem
theorem all_positive_integers_are_valid :
  ∀ n : ℕ+, ∃ c : Coloring, validColoring n c :=
sorry

end NUMINAMATH_CALUDE_all_positive_integers_are_valid_l1734_173466


namespace NUMINAMATH_CALUDE_number_of_shoppers_l1734_173499

theorem number_of_shoppers (isabella sam giselle : ℕ) (shoppers : ℕ) : 
  isabella = sam + 45 →
  isabella = giselle + 15 →
  giselle = 120 →
  (isabella + sam + giselle) / shoppers = 115 →
  shoppers = 3 := by
sorry

end NUMINAMATH_CALUDE_number_of_shoppers_l1734_173499


namespace NUMINAMATH_CALUDE_shaded_area_between_circles_l1734_173488

theorem shaded_area_between_circles (r₁ r₂ r₃ R : ℝ) (h₁ : r₁ = 4) (h₂ : r₂ = 5) (h₃ : r₃ = 2)
  (h_external : R = (r₁ + r₂ + r₁ + r₂) / 2)
  (h_tangent : r₁ + r₂ = R - r₁ - r₂) :
  π * R^2 - π * r₁^2 - π * r₂^2 - π * r₃^2 = 36 * π :=
by sorry

end NUMINAMATH_CALUDE_shaded_area_between_circles_l1734_173488


namespace NUMINAMATH_CALUDE_time_to_fill_cistern_l1734_173473

/-- Given a cistern that can be partially filled by a pipe, this theorem proves
    the time required to fill the entire cistern. -/
theorem time_to_fill_cistern (partial_fill_time : ℝ) (partial_fill_fraction : ℝ) 
  (h1 : partial_fill_time = 7)
  (h2 : partial_fill_fraction = 1 / 11) : 
  partial_fill_time / partial_fill_fraction = 77 := by
  sorry

end NUMINAMATH_CALUDE_time_to_fill_cistern_l1734_173473


namespace NUMINAMATH_CALUDE_tablet_savings_l1734_173414

/-- The savings when buying a tablet in cash versus installment -/
theorem tablet_savings : 
  let cash_price : ℕ := 450
  let down_payment : ℕ := 100
  let first_four_months : ℕ := 4 * 40
  let next_four_months : ℕ := 4 * 35
  let last_four_months : ℕ := 4 * 30
  let total_installment : ℕ := down_payment + first_four_months + next_four_months + last_four_months
  total_installment - cash_price = 70 := by
  sorry

end NUMINAMATH_CALUDE_tablet_savings_l1734_173414


namespace NUMINAMATH_CALUDE_train_length_l1734_173417

/-- Given a train that crosses a platform in 50 seconds and a signal pole in 42 seconds,
    with the platform length being 38.0952380952381 meters, prove that the length of the train is 200 meters. -/
theorem train_length (platform_crossing_time : ℝ) (pole_crossing_time : ℝ) (platform_length : ℝ)
    (h1 : platform_crossing_time = 50)
    (h2 : pole_crossing_time = 42)
    (h3 : platform_length = 38.0952380952381) :
    ∃ train_length : ℝ, train_length = 200 := by
  sorry

end NUMINAMATH_CALUDE_train_length_l1734_173417


namespace NUMINAMATH_CALUDE_polar_midpoint_specific_case_l1734_173474

/-- The midpoint of a line segment in polar coordinates --/
def polar_midpoint (r1 : ℝ) (θ1 : ℝ) (r2 : ℝ) (θ2 : ℝ) : ℝ × ℝ :=
  sorry

/-- Theorem: The midpoint of a line segment with endpoints (10, π/4) and (10, 3π/4) in polar coordinates is (5√2, π/2) --/
theorem polar_midpoint_specific_case :
  let (r, θ) := polar_midpoint 10 (π/4) 10 (3*π/4)
  r = 5 * Real.sqrt 2 ∧ θ = π/2 := by sorry

end NUMINAMATH_CALUDE_polar_midpoint_specific_case_l1734_173474


namespace NUMINAMATH_CALUDE_smallest_gcd_bc_l1734_173415

theorem smallest_gcd_bc (a b c : ℕ+) 
  (hab : Nat.gcd a b = 360)
  (hac : Nat.gcd a c = 1170)
  (hb : 5 ∣ b)
  (hc : 13 ∣ c) :
  ∃ (k : ℕ+), Nat.gcd b c = k ∧ 
  ∀ (m : ℕ+), Nat.gcd b c ≤ m → k ≤ m :=
by sorry

end NUMINAMATH_CALUDE_smallest_gcd_bc_l1734_173415


namespace NUMINAMATH_CALUDE_plum_pies_count_l1734_173494

theorem plum_pies_count (total : ℕ) (ratio_r : ℕ) (ratio_p : ℕ) (ratio_m : ℕ) 
  (h_total : total = 30)
  (h_ratio : ratio_r = 2 ∧ ratio_p = 5 ∧ ratio_m = 3) :
  (total * ratio_m) / (ratio_r + ratio_p + ratio_m) = 9 := by
sorry

end NUMINAMATH_CALUDE_plum_pies_count_l1734_173494


namespace NUMINAMATH_CALUDE_unique_perfect_square_sum_l1734_173484

def is_perfect_square (n : ℕ) : Prop := ∃ m : ℕ, n = m ^ 2

def distinct_perfect_square_sum (a b c : ℕ) : Prop :=
  is_perfect_square a ∧ is_perfect_square b ∧ is_perfect_square c ∧
  a ≠ b ∧ b ≠ c ∧ a ≠ c ∧
  a > 0 ∧ b > 0 ∧ c > 0 ∧
  a + b + c = 100

theorem unique_perfect_square_sum :
  ∃! (s : Finset (Finset ℕ)), s.card = 1 ∧
    ∀ t ∈ s, t.card = 3 ∧
      (∃ a b c, {a, b, c} = t ∧ distinct_perfect_square_sum a b c) :=
sorry

end NUMINAMATH_CALUDE_unique_perfect_square_sum_l1734_173484


namespace NUMINAMATH_CALUDE_linear_equation_implies_a_value_l1734_173433

/-- Given that (a-2)x^(|a|-1) + 3y = 1 is a linear equation in x and y, prove that a = -2 --/
theorem linear_equation_implies_a_value (a : ℝ) : 
  (∀ x y : ℝ, ∃ k m : ℝ, (a - 2) * x^(|a| - 1) + 3 * y = k * x + m * y + 1) → 
  a = -2 :=
by sorry

end NUMINAMATH_CALUDE_linear_equation_implies_a_value_l1734_173433


namespace NUMINAMATH_CALUDE_equal_points_per_round_l1734_173472

-- Define the total points and number of rounds
def total_points : ℕ := 300
def num_rounds : ℕ := 5

-- Define the points per round
def points_per_round : ℕ := total_points / num_rounds

-- Theorem to prove
theorem equal_points_per_round :
  (total_points = num_rounds * points_per_round) ∧ (points_per_round = 60) := by
  sorry

end NUMINAMATH_CALUDE_equal_points_per_round_l1734_173472


namespace NUMINAMATH_CALUDE_find_a_and_m_l1734_173450

-- Define the sets A, B, and C
def A : Set ℝ := {x | x^2 - 3*x + 2 = 0}
def B (a : ℝ) : Set ℝ := {x | x^2 - a*x + (a-1) = 0}
def C (m : ℝ) : Set ℝ := {x | x^2 - m*x + 2 = 0}

-- State the theorem
theorem find_a_and_m :
  ∃ (a m : ℝ),
    (A ∪ B a = A) ∧
    (A ∩ B a = C m) ∧
    (a = 3) ∧
    (m = 3) := by
  sorry

end NUMINAMATH_CALUDE_find_a_and_m_l1734_173450


namespace NUMINAMATH_CALUDE_total_cost_is_21_l1734_173482

/-- The cost of a single carnation in dollars -/
def single_carnation_cost : ℚ := 0.50

/-- The cost of a dozen carnations in dollars -/
def dozen_carnation_cost : ℚ := 4.00

/-- The number of teachers Georgia is sending carnations to -/
def number_of_teachers : ℕ := 5

/-- The number of friends Georgia is buying carnations for -/
def number_of_friends : ℕ := 14

/-- The total cost of Georgia's carnation purchases -/
def total_cost : ℚ := dozen_carnation_cost * number_of_teachers + 
  single_carnation_cost * (number_of_friends % 12)

/-- Theorem stating that the total cost is $21.00 -/
theorem total_cost_is_21 : total_cost = 21 := by sorry

end NUMINAMATH_CALUDE_total_cost_is_21_l1734_173482


namespace NUMINAMATH_CALUDE_parabola_transformation_l1734_173495

def original_parabola (x : ℝ) : ℝ := 2 * (x - 3)^2 + 2

def shift_left (f : ℝ → ℝ) (k : ℝ) : ℝ → ℝ := λ x => f (x + k)

def shift_up (f : ℝ → ℝ) (k : ℝ) : ℝ → ℝ := λ x => f x + k

def transformed_parabola : ℝ → ℝ :=
  shift_up (shift_left original_parabola 3) 2

theorem parabola_transformation :
  ∀ x : ℝ, transformed_parabola x = 2 * x^2 + 4 := by
  sorry

end NUMINAMATH_CALUDE_parabola_transformation_l1734_173495


namespace NUMINAMATH_CALUDE_a_2017_equals_2_l1734_173483

-- Define the sequence S_n
def S (n : ℕ) : ℕ := 2 * n - 1

-- Define the sequence a_n
def a (n : ℕ) : ℕ := S n - S (n - 1)

-- Theorem statement
theorem a_2017_equals_2 : a 2017 = 2 := by
  sorry

end NUMINAMATH_CALUDE_a_2017_equals_2_l1734_173483


namespace NUMINAMATH_CALUDE_three_numbers_sum_l1734_173496

theorem three_numbers_sum (a b c m : ℕ) : 
  a + b + c = 2015 →
  a + b = m + 1 →
  b + c = m + 2011 →
  c + a = m + 2012 →
  m = 2 := by
sorry

end NUMINAMATH_CALUDE_three_numbers_sum_l1734_173496


namespace NUMINAMATH_CALUDE_cab_speed_ratio_l1734_173425

/-- Proves that the ratio of a cab's current speed to its usual speed is 5:6 -/
theorem cab_speed_ratio : 
  ∀ (usual_time current_time usual_speed current_speed : ℝ),
  usual_time = 25 →
  current_time = usual_time + 5 →
  usual_speed * usual_time = current_speed * current_time →
  current_speed / usual_speed = 5 / 6 := by
  sorry

end NUMINAMATH_CALUDE_cab_speed_ratio_l1734_173425


namespace NUMINAMATH_CALUDE_min_value_theorem_l1734_173419

theorem min_value_theorem (a b : ℝ) (ha : a > 0) (hb : b > 0)
  (hmin : ∀ x, |x + a| + |x - b| ≥ 4) : 
  (a + b = 4) ∧ (∀ a' b' : ℝ, a' > 0 → b' > 0 → (1/4) * a'^2 + (1/9) * b'^2 ≥ 16/13) := by
  sorry

end NUMINAMATH_CALUDE_min_value_theorem_l1734_173419


namespace NUMINAMATH_CALUDE_congruence_solution_l1734_173443

theorem congruence_solution (n : ℤ) : 
  (13 * n) % 47 = 8 % 47 → n % 47 = 29 % 47 := by
sorry

end NUMINAMATH_CALUDE_congruence_solution_l1734_173443


namespace NUMINAMATH_CALUDE_abs_2x_minus_1_lt_15_x_squared_plus_6x_minus_16_lt_0_abs_2x_plus_1_gt_13_x_squared_minus_2x_gt_0_l1734_173446

-- Question 1
theorem abs_2x_minus_1_lt_15 (x : ℝ) : 
  |2*x - 1| < 15 ↔ -7 < x ∧ x < 8 := by sorry

-- Question 2
theorem x_squared_plus_6x_minus_16_lt_0 (x : ℝ) : 
  x^2 + 6*x - 16 < 0 ↔ -8 < x ∧ x < 2 := by sorry

-- Question 3
theorem abs_2x_plus_1_gt_13 (x : ℝ) : 
  |2*x + 1| > 13 ↔ x < -7 ∨ x > 6 := by sorry

-- Question 4
theorem x_squared_minus_2x_gt_0 (x : ℝ) : 
  x^2 - 2*x > 0 ↔ x < 0 ∨ x > 2 := by sorry

end NUMINAMATH_CALUDE_abs_2x_minus_1_lt_15_x_squared_plus_6x_minus_16_lt_0_abs_2x_plus_1_gt_13_x_squared_minus_2x_gt_0_l1734_173446


namespace NUMINAMATH_CALUDE_symmetric_point_theorem_l1734_173418

/-- A point in a 2D plane. -/
structure Point where
  x : ℝ
  y : ℝ

/-- The symmetric point of a given point with respect to the origin. -/
def symmetricPoint (p : Point) : Point :=
  { x := -p.x, y := -p.y }

/-- Theorem: The symmetric point of (3, -4) with respect to the origin is (-3, 4). -/
theorem symmetric_point_theorem :
  let p : Point := { x := 3, y := -4 }
  symmetricPoint p = { x := -3, y := 4 } := by
  sorry


end NUMINAMATH_CALUDE_symmetric_point_theorem_l1734_173418


namespace NUMINAMATH_CALUDE_carla_fish_count_l1734_173411

/-- Given that Carla, Kyle, and Tasha caught a total of 36 fish, Kyle caught 14 fish,
    and Kyle and Tasha caught the same number of fish, prove that Carla caught 8 fish. -/
theorem carla_fish_count (total : ℕ) (kyle_fish : ℕ) (h1 : total = 36) (h2 : kyle_fish = 14)
    (h3 : ∃ (tasha_fish : ℕ), tasha_fish = kyle_fish ∧ total = kyle_fish + tasha_fish + (total - kyle_fish - tasha_fish)) :
  total - kyle_fish - kyle_fish = 8 := by
  sorry

end NUMINAMATH_CALUDE_carla_fish_count_l1734_173411


namespace NUMINAMATH_CALUDE_chick_count_product_l1734_173491

/-- Represents the state of chicks in a nest for a given week -/
structure ChickState :=
  (open_beak : ℕ)
  (growing_feathers : ℕ)

/-- The chick lifecycle in the nest -/
def chick_lifecycle : Prop :=
  ∃ (last_week this_week : ChickState),
    last_week.open_beak = 20 ∧
    last_week.growing_feathers = 14 ∧
    this_week.open_beak = 15 ∧
    this_week.growing_feathers = 11

/-- The theorem to be proved -/
theorem chick_count_product :
  chick_lifecycle →
  ∃ (two_weeks_ago next_week : ℕ),
    two_weeks_ago = 11 ∧
    next_week = 15 ∧
    two_weeks_ago * next_week = 165 :=
by
  sorry


end NUMINAMATH_CALUDE_chick_count_product_l1734_173491


namespace NUMINAMATH_CALUDE_barbell_cost_l1734_173454

theorem barbell_cost (number_of_barbells : ℕ) (amount_paid : ℕ) (change_received : ℕ) :
  number_of_barbells = 3 ∧ amount_paid = 850 ∧ change_received = 40 →
  (amount_paid - change_received) / number_of_barbells = 270 :=
by
  sorry

end NUMINAMATH_CALUDE_barbell_cost_l1734_173454


namespace NUMINAMATH_CALUDE_ice_cream_volume_l1734_173449

/-- The volume of ice cream in a cone with a cylinder on top -/
theorem ice_cream_volume (cone_height : ℝ) (cone_radius : ℝ) (cylinder_height : ℝ) : 
  cone_height = 12 → 
  cone_radius = 3 → 
  cylinder_height = 2 → 
  (1/3 * π * cone_radius^2 * cone_height) + (π * cone_radius^2 * cylinder_height) = 54 * π := by
  sorry


end NUMINAMATH_CALUDE_ice_cream_volume_l1734_173449


namespace NUMINAMATH_CALUDE_polynomial_decomposition_l1734_173434

theorem polynomial_decomposition (x : ℝ) :
  x^3 - 2*x^2 + 3*x + 5 = 11 + 7*(x - 2) + 4*(x - 2)^2 + (x - 2)^3 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_decomposition_l1734_173434


namespace NUMINAMATH_CALUDE_peaches_after_seven_days_l1734_173400

def peaches_after_days (initial_total : ℕ) (initial_ripe : ℕ) (days : ℕ) : ℕ × ℕ :=
  sorry

theorem peaches_after_seven_days :
  let initial_total := 18
  let initial_ripe := 4
  let ripen_pattern (d : ℕ) := d + 1
  let eat_pattern (d : ℕ) := d
  let (ripe, unripe) := peaches_after_days initial_total initial_ripe 7
  ripe = 0 ∧ unripe = 0 :=
sorry

end NUMINAMATH_CALUDE_peaches_after_seven_days_l1734_173400


namespace NUMINAMATH_CALUDE_smallest_a_for_divisibility_l1734_173470

theorem smallest_a_for_divisibility : 
  (∃ (a : ℕ), a > 0 ∧ 
    (∃ (n : ℕ), n > 0 ∧ Odd n ∧ 
      (2001 ∣ 55^n + a * 32^n))) ∧ 
  (∀ (a : ℕ), a > 0 → 
    (∃ (n : ℕ), n > 0 ∧ Odd n ∧ 
      (2001 ∣ 55^n + a * 32^n)) → 
    a ≥ 436) := by
  sorry

end NUMINAMATH_CALUDE_smallest_a_for_divisibility_l1734_173470


namespace NUMINAMATH_CALUDE_five_spheres_max_regions_l1734_173423

/-- The maximum number of regions into which n spheres can divide three-dimensional space -/
def max_regions (n : ℕ) : ℕ :=
  match n with
  | 0 => 1
  | n + 1 => max_regions n + 2 + n + n * (n + 1) / 2

/-- The maximum number of regions into which five spheres can divide three-dimensional space is 47 -/
theorem five_spheres_max_regions :
  max_regions 5 = 47 := by sorry

end NUMINAMATH_CALUDE_five_spheres_max_regions_l1734_173423


namespace NUMINAMATH_CALUDE_first_negative_term_position_l1734_173410

def arithmeticSequence (a₁ : ℤ) (d : ℤ) (n : ℕ) : ℤ := a₁ + (n - 1 : ℤ) * d

theorem first_negative_term_position
  (a₁ : ℤ)
  (d : ℤ)
  (h₁ : a₁ = 1031)
  (h₂ : d = -3) :
  (∀ k < 345, arithmeticSequence a₁ d k ≥ 0) ∧
  arithmeticSequence a₁ d 345 < 0 :=
sorry

end NUMINAMATH_CALUDE_first_negative_term_position_l1734_173410


namespace NUMINAMATH_CALUDE_f_monotone_range_l1734_173486

/-- The function f(x) defined as x^2 + a|x-1| -/
def f (a : ℝ) (x : ℝ) : ℝ := x^2 + a * abs (x - 1)

/-- The theorem stating the range of 'a' for which f is monotonically increasing on [0, +∞) -/
theorem f_monotone_range (a : ℝ) :
  (∀ x y, 0 ≤ x ∧ x ≤ y → f a x ≤ f a y) ↔ -2 ≤ a ∧ a ≤ 0 := by
  sorry

end NUMINAMATH_CALUDE_f_monotone_range_l1734_173486


namespace NUMINAMATH_CALUDE_hunter_score_theorem_l1734_173436

def math_test_scores (grant_score john_score hunter_score : ℕ) : Prop :=
  (grant_score = 100) ∧
  (grant_score = john_score + 10) ∧
  (john_score = 2 * hunter_score)

theorem hunter_score_theorem :
  ∀ grant_score john_score hunter_score : ℕ,
  math_test_scores grant_score john_score hunter_score →
  hunter_score = 45 :=
by
  sorry

end NUMINAMATH_CALUDE_hunter_score_theorem_l1734_173436


namespace NUMINAMATH_CALUDE_sin_seventeen_pi_quarters_l1734_173490

theorem sin_seventeen_pi_quarters : Real.sin (17 * Real.pi / 4) = Real.sqrt 2 / 2 := by
  sorry

end NUMINAMATH_CALUDE_sin_seventeen_pi_quarters_l1734_173490


namespace NUMINAMATH_CALUDE_new_ratio_after_refill_l1734_173412

def initial_ratio_a : ℚ := 7
def initial_ratio_b : ℚ := 5
def initial_volume_a : ℚ := 21
def volume_drawn : ℚ := 9

theorem new_ratio_after_refill :
  let total_volume := initial_volume_a * (initial_ratio_a + initial_ratio_b) / initial_ratio_a
  let removed_a := volume_drawn * initial_ratio_a / (initial_ratio_a + initial_ratio_b)
  let removed_b := volume_drawn * initial_ratio_b / (initial_ratio_a + initial_ratio_b)
  let remaining_a := initial_volume_a - removed_a
  let remaining_b := total_volume - initial_volume_a - removed_b
  let new_b := remaining_b + volume_drawn
  (remaining_a : ℚ) / new_b = 21 / 27 :=
sorry

end NUMINAMATH_CALUDE_new_ratio_after_refill_l1734_173412


namespace NUMINAMATH_CALUDE_complement_A_intersect_B_l1734_173442

def I : Finset Nat := {1,2,3,4,5,6}
def A : Finset Nat := {1,3,5}
def B : Finset Nat := {2,3,6}

theorem complement_A_intersect_B : 
  (I \ A) ∩ B = {2,6} := by sorry

end NUMINAMATH_CALUDE_complement_A_intersect_B_l1734_173442


namespace NUMINAMATH_CALUDE_profit_growth_rate_l1734_173459

/-- The average monthly growth rate of profit from March to May -/
def average_growth_rate : ℝ := 0.2

/-- The profit in March -/
def march_profit : ℝ := 5000

/-- The profit in May -/
def may_profit : ℝ := 7200

/-- The number of months between March and May -/
def months_between : ℕ := 2

theorem profit_growth_rate :
  march_profit * (1 + average_growth_rate) ^ months_between = may_profit :=
sorry

end NUMINAMATH_CALUDE_profit_growth_rate_l1734_173459


namespace NUMINAMATH_CALUDE_image_of_one_two_l1734_173441

def f (p : ℝ × ℝ) : ℝ × ℝ :=
  (2 * p.1 - p.2, p.1 - 2 * p.2)

theorem image_of_one_two :
  f (1, 2) = (0, -3) := by sorry

end NUMINAMATH_CALUDE_image_of_one_two_l1734_173441


namespace NUMINAMATH_CALUDE_complex_equality_implication_l1734_173428

theorem complex_equality_implication (x y : ℝ) : 
  (Complex.I * x + 2 = y - Complex.I) → (x - y = -3) := by
  sorry

end NUMINAMATH_CALUDE_complex_equality_implication_l1734_173428


namespace NUMINAMATH_CALUDE_angle_C_is_pi_over_three_sum_a_b_range_l1734_173409

-- Define a triangle ABC
structure Triangle where
  A : Real
  B : Real
  C : Real
  a : Real
  b : Real
  c : Real

-- Define the condition given in the problem
def satisfiesCondition (t : Triangle) : Prop :=
  (t.a + t.c) * (Real.sin t.A - Real.sin t.C) = Real.sin t.B * (t.a - t.b)

-- Theorem for part I
theorem angle_C_is_pi_over_three (t : Triangle) 
  (h : satisfiesCondition t) : t.C = π / 3 := by sorry

-- Theorem for part II
theorem sum_a_b_range (t : Triangle) 
  (h1 : satisfiesCondition t) 
  (h2 : t.c = 2) : 
  2 < t.a + t.b ∧ t.a + t.b ≤ 4 := by sorry

end NUMINAMATH_CALUDE_angle_C_is_pi_over_three_sum_a_b_range_l1734_173409


namespace NUMINAMATH_CALUDE_dinosaur_weight_theorem_l1734_173429

def regular_dinosaur_weight : ℕ := 800
def number_of_regular_dinosaurs : ℕ := 5
def barney_weight_difference : ℕ := 1500

def total_weight : ℕ :=
  (regular_dinosaur_weight * number_of_regular_dinosaurs) +
  (regular_dinosaur_weight * number_of_regular_dinosaurs + barney_weight_difference)

theorem dinosaur_weight_theorem :
  total_weight = 9500 :=
by sorry

end NUMINAMATH_CALUDE_dinosaur_weight_theorem_l1734_173429


namespace NUMINAMATH_CALUDE_tan_seven_pi_fourths_l1734_173487

theorem tan_seven_pi_fourths : Real.tan (7 * π / 4) = -1 := by sorry

end NUMINAMATH_CALUDE_tan_seven_pi_fourths_l1734_173487


namespace NUMINAMATH_CALUDE_equation_solution_inequality_solution_l1734_173435

-- Definition of permutation
def A (n : ℕ) (m : ℕ) : ℕ := Nat.factorial n / Nat.factorial (n - m)

-- Theorem for the equation
theorem equation_solution :
  ∃! x : ℕ, 3 * A 8 x = 4 * A 9 (x - 1) ∧ x = 6 :=
sorry

-- Theorem for the inequality
theorem inequality_solution :
  ∀ x : ℕ, x ≥ 4 ↔ A (x - 2) 2 + x ≥ 2 :=
sorry

end NUMINAMATH_CALUDE_equation_solution_inequality_solution_l1734_173435


namespace NUMINAMATH_CALUDE_sector_perimeter_l1734_173406

theorem sector_perimeter (θ : Real) (r : Real) (h1 : θ = 54) (h2 : r = 20) :
  let l := (θ / 360) * (2 * Real.pi * r)
  l + 2 * r = 6 * Real.pi + 40 := by
  sorry

end NUMINAMATH_CALUDE_sector_perimeter_l1734_173406


namespace NUMINAMATH_CALUDE_power_of_two_minus_one_as_power_l1734_173497

theorem power_of_two_minus_one_as_power (n : ℕ) : 
  (∃ (a k : ℕ), k ≥ 2 ∧ 2^n - 1 = a^k) ↔ n = 0 ∨ n = 1 := by
  sorry

end NUMINAMATH_CALUDE_power_of_two_minus_one_as_power_l1734_173497


namespace NUMINAMATH_CALUDE_sam_pennies_washing_l1734_173469

/-- The number of pennies Sam got for washing clothes -/
def pennies_from_washing (total_cents : ℕ) (num_quarters : ℕ) : ℕ :=
  total_cents - (num_quarters * 25)

/-- Theorem stating that Sam got 9 pennies for washing clothes -/
theorem sam_pennies_washing : 
  pennies_from_washing 184 7 = 9 := by sorry

end NUMINAMATH_CALUDE_sam_pennies_washing_l1734_173469


namespace NUMINAMATH_CALUDE_darcys_walking_speed_l1734_173402

/-- Proves that Darcy's walking speed is 3 miles per hour given the problem conditions -/
theorem darcys_walking_speed 
  (distance_to_work : ℝ) 
  (train_speed : ℝ) 
  (additional_train_time : ℝ) 
  (time_difference : ℝ) 
  (h1 : distance_to_work = 1.5)
  (h2 : train_speed = 20)
  (h3 : additional_train_time = 23.5 / 60)
  (h4 : time_difference = 2 / 60)
  (h5 : distance_to_work / train_speed + additional_train_time + time_difference = distance_to_work / 3) :
  3 = 3 := by
  sorry

#check darcys_walking_speed

end NUMINAMATH_CALUDE_darcys_walking_speed_l1734_173402


namespace NUMINAMATH_CALUDE_bridge_length_l1734_173444

/-- The length of a bridge given train parameters and crossing time -/
theorem bridge_length
  (train_length : ℝ)
  (train_speed_kmh : ℝ)
  (crossing_time_s : ℝ)
  (h1 : train_length = 130)
  (h2 : train_speed_kmh = 45)
  (h3 : crossing_time_s = 30) :
  train_speed_kmh * (1000 / 3600) * crossing_time_s - train_length = 245 :=
by sorry

end NUMINAMATH_CALUDE_bridge_length_l1734_173444


namespace NUMINAMATH_CALUDE_potato_sale_revenue_l1734_173477

-- Define the given constants
def total_weight : ℕ := 6500
def damaged_weight : ℕ := 150
def bag_weight : ℕ := 50
def price_per_bag : ℕ := 72

-- Define the theorem
theorem potato_sale_revenue : 
  (((total_weight - damaged_weight) / bag_weight) * price_per_bag = 9144) := by
  sorry


end NUMINAMATH_CALUDE_potato_sale_revenue_l1734_173477


namespace NUMINAMATH_CALUDE_twenty_first_term_is_4641_l1734_173432

/-- The nth term of the sequence is the sum of n consecutive integers starting from n(n-1)/2 + 1 -/
def sequence_term (n : ℕ) : ℕ :=
  let start := n * (n - 1) / 2 + 1
  (n * (2 * start + n - 1)) / 2

theorem twenty_first_term_is_4641 : sequence_term 21 = 4641 := by sorry

end NUMINAMATH_CALUDE_twenty_first_term_is_4641_l1734_173432


namespace NUMINAMATH_CALUDE_tan_600_l1734_173485

-- Define the tangent function (simplified for this example)
noncomputable def tan (x : ℝ) : ℝ := sorry

-- State the periodicity of tangent
axiom tan_periodic (x : ℝ) : tan (x + 180) = tan x

-- State the value of tan 60°
axiom tan_60 : tan 60 = Real.sqrt 3

-- Theorem to prove
theorem tan_600 : tan 600 = Real.sqrt 3 := by sorry

end NUMINAMATH_CALUDE_tan_600_l1734_173485


namespace NUMINAMATH_CALUDE_bee_legs_count_l1734_173498

/-- Given 8 bees with a total of 48 legs, prove that each bee has 6 legs. -/
theorem bee_legs_count :
  let total_bees : ℕ := 8
  let total_legs : ℕ := 48
  total_legs / total_bees = 6 := by sorry

end NUMINAMATH_CALUDE_bee_legs_count_l1734_173498


namespace NUMINAMATH_CALUDE_lailas_test_scores_l1734_173448

theorem lailas_test_scores (first_four_score last_score : ℕ) : 
  (0 ≤ first_four_score ∧ first_four_score ≤ 100) →
  (0 ≤ last_score ∧ last_score ≤ 100) →
  (last_score > first_four_score) →
  ((4 * first_four_score + last_score) / 5 = 82) →
  (∃ possible_scores : Finset ℕ, 
    possible_scores.card = 4 ∧
    last_score ∈ possible_scores ∧
    ∀ s, s ∈ possible_scores → 
      (0 ≤ s ∧ s ≤ 100) ∧
      (∃ x : ℕ, (0 ≤ x ∧ x ≤ 100) ∧ 
                (s > x) ∧ 
                ((4 * x + s) / 5 = 82))) :=
by sorry

end NUMINAMATH_CALUDE_lailas_test_scores_l1734_173448


namespace NUMINAMATH_CALUDE_max_silver_tokens_l1734_173424

/-- Represents the state of Alex's tokens -/
structure TokenState where
  red : ℕ
  blue : ℕ
  silver : ℕ

/-- Represents an exchange booth -/
structure Booth where
  redIn : ℕ
  blueIn : ℕ
  redOut : ℕ
  blueOut : ℕ
  silverOut : ℕ

/-- Checks if an exchange is possible at a given booth -/
def canExchange (state : TokenState) (booth : Booth) : Prop :=
  state.red ≥ booth.redIn ∧ state.blue ≥ booth.blueIn

/-- Performs an exchange at a given booth -/
def exchange (state : TokenState) (booth : Booth) : TokenState :=
  { red := state.red - booth.redIn + booth.redOut,
    blue := state.blue - booth.blueIn + booth.blueOut,
    silver := state.silver + booth.silverOut }

/-- The theorem to be proved -/
theorem max_silver_tokens : ∃ (finalState : TokenState),
  let initialState : TokenState := { red := 90, blue := 65, silver := 0 }
  let booth1 : Booth := { redIn := 3, blueIn := 0, redOut := 0, blueOut := 2, silverOut := 1 }
  let booth2 : Booth := { redIn := 0, blueIn := 4, redOut := 2, blueOut := 0, silverOut := 1 }
  (∀ state, (canExchange state booth1 ∨ canExchange state booth2) → 
    (finalState.silver ≥ state.silver)) ∧
  (¬ canExchange finalState booth1 ∧ ¬ canExchange finalState booth2) ∧
  finalState.silver = 67 :=
sorry

end NUMINAMATH_CALUDE_max_silver_tokens_l1734_173424


namespace NUMINAMATH_CALUDE_max_digits_product_l1734_173476

theorem max_digits_product (a b : ℕ) : 
  1000 ≤ a ∧ a < 10000 → 10000 ≤ b ∧ b < 100000 → 
  a * b < 1000000000 :=
sorry

end NUMINAMATH_CALUDE_max_digits_product_l1734_173476


namespace NUMINAMATH_CALUDE_book_cost_problem_l1734_173489

theorem book_cost_problem (total_cost : ℝ) (loss_percent : ℝ) (gain_percent : ℝ) :
  total_cost = 360 ∧ loss_percent = 0.15 ∧ gain_percent = 0.19 →
  ∃ (cost_loss cost_gain : ℝ),
    cost_loss + cost_gain = total_cost ∧
    cost_loss * (1 - loss_percent) = cost_gain * (1 + gain_percent) ∧
    cost_loss = 210 := by
  sorry

end NUMINAMATH_CALUDE_book_cost_problem_l1734_173489


namespace NUMINAMATH_CALUDE_metallic_sheet_length_l1734_173401

/-- Given a rectangular metallic sheet from which squares are cut at corners to form a box,
    this theorem proves the length of the original sheet. -/
theorem metallic_sheet_length
  (square_side : ℝ)
  (sheet_width : ℝ)
  (box_volume : ℝ)
  (h_square : square_side = 6)
  (h_width : sheet_width = 36)
  (h_volume : box_volume = 5184)
  (h_box : box_volume = (sheet_length - 2 * square_side) * (sheet_width - 2 * square_side) * square_side) :
  sheet_length = 48 :=
by
  sorry


end NUMINAMATH_CALUDE_metallic_sheet_length_l1734_173401


namespace NUMINAMATH_CALUDE_unique_solution_l1734_173475

/-- Represents a number of the form 13xy4.5z -/
def SpecialNumber (x y z : ℕ) : ℚ :=
  13000 + 100 * x + 10 * y + 4 + 0.5 + 0.01 * z

theorem unique_solution :
  ∃! (x y z : ℕ),
    x < 10 ∧ y < 10 ∧ z < 10 ∧
    (∃ (k : ℕ), SpecialNumber x y z = k * 792) ∧
    (45 + z) % 8 = 0 ∧
    (1 + 3 + x + y + 4 + 5 + z) % 9 = 0 ∧
    (1 - 3 + x - y + 4 - 5 + z) % 11 = 0 ∧
    SpecialNumber x y z = 13804.56 :=
by
  sorry

#eval SpecialNumber 8 0 6  -- Should output 13804.56

end NUMINAMATH_CALUDE_unique_solution_l1734_173475


namespace NUMINAMATH_CALUDE_marble_problem_l1734_173416

theorem marble_problem (r b : ℕ) : 
  ((r - 3 : ℚ) / (r + b - 3) = 1 / 10) →
  ((r : ℚ) / (r + b - 3) = 1 / 4) →
  r + b = 13 := by
  sorry

end NUMINAMATH_CALUDE_marble_problem_l1734_173416


namespace NUMINAMATH_CALUDE_square_difference_l1734_173405

theorem square_difference (x y : ℝ) 
  (h1 : x + y = 12) 
  (h2 : 3 * x + y = 18) : 
  x^2 - y^2 = -72 := by
sorry

end NUMINAMATH_CALUDE_square_difference_l1734_173405


namespace NUMINAMATH_CALUDE_train_crossing_time_l1734_173413

/-- The time taken for a train to cross a platform of equal length -/
theorem train_crossing_time (train_length : ℝ) (train_speed_kmh : ℝ) : 
  train_length = 1500 →
  train_speed_kmh = 180 →
  (2 * train_length) / (train_speed_kmh * 1000 / 3600) = 60 := by
  sorry

#check train_crossing_time

end NUMINAMATH_CALUDE_train_crossing_time_l1734_173413


namespace NUMINAMATH_CALUDE_ralph_weekly_tv_hours_l1734_173452

/-- Represents Ralph's TV watching habits for a week -/
structure TVWatchingHabits where
  weekday_hours : ℕ
  weekend_hours : ℕ
  weekdays : ℕ
  weekend_days : ℕ

/-- Calculates the total hours of TV watched in a week -/
def total_weekly_hours (habits : TVWatchingHabits) : ℕ :=
  habits.weekday_hours * habits.weekdays + habits.weekend_hours * habits.weekend_days

/-- Theorem stating that Ralph watches 32 hours of TV in a week -/
theorem ralph_weekly_tv_hours :
  let habits : TVWatchingHabits := {
    weekday_hours := 4,
    weekend_hours := 6,
    weekdays := 5,
    weekend_days := 2
  }
  total_weekly_hours habits = 32 := by
  sorry

end NUMINAMATH_CALUDE_ralph_weekly_tv_hours_l1734_173452


namespace NUMINAMATH_CALUDE_math_team_combinations_l1734_173461

/-- The number of ways to choose k elements from n elements -/
def choose (n k : ℕ) : ℕ := Nat.choose n k

/-- The number of girls in the math club -/
def num_girls : ℕ := 4

/-- The number of boys in the math club -/
def num_boys : ℕ := 6

/-- The number of girls to be chosen for the team -/
def girls_in_team : ℕ := 3

/-- The number of boys to be chosen for the team -/
def boys_in_team : ℕ := 2

theorem math_team_combinations :
  (choose num_girls girls_in_team) * (choose num_boys boys_in_team) = 60 := by
  sorry

end NUMINAMATH_CALUDE_math_team_combinations_l1734_173461


namespace NUMINAMATH_CALUDE_soup_cans_bought_soup_cans_received_johns_soup_cans_l1734_173439

theorem soup_cans_bought (normal_price : ℝ) (total_paid : ℝ) : ℝ :=
  total_paid / normal_price

theorem soup_cans_received (cans_bought : ℝ) : ℝ :=
  2 * cans_bought

theorem johns_soup_cans (normal_price : ℝ) (total_paid : ℝ) : 
  soup_cans_received (soup_cans_bought normal_price total_paid) = 30 :=
by
  -- Assuming normal_price = 0.60 and total_paid = 9
  have h1 : normal_price = 0.60 := by sorry
  have h2 : total_paid = 9 := by sorry
  
  -- Calculate the number of cans bought
  have cans_bought : ℝ := soup_cans_bought normal_price total_paid
  
  -- Calculate the total number of cans received
  have total_cans : ℝ := soup_cans_received cans_bought
  
  -- Prove that the total number of cans is 30
  sorry

end NUMINAMATH_CALUDE_soup_cans_bought_soup_cans_received_johns_soup_cans_l1734_173439
