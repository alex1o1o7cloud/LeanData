import Mathlib

namespace NUMINAMATH_CALUDE_doughnuts_remaining_l2612_261297

/-- The number of doughnuts in a dozen -/
def dozen : ℕ := 12

/-- The number of dozens of doughnuts initially in the box -/
def initial_dozens : ℕ := 2

/-- The number of doughnuts eaten by the family -/
def eaten_doughnuts : ℕ := 8

/-- The number of doughnuts left in the box -/
def doughnuts_left : ℕ := initial_dozens * dozen - eaten_doughnuts

theorem doughnuts_remaining : doughnuts_left = 16 := by
  sorry

end NUMINAMATH_CALUDE_doughnuts_remaining_l2612_261297


namespace NUMINAMATH_CALUDE_inequality_solution_range_l2612_261240

theorem inequality_solution_range (k : ℝ) : 
  (∀ x : ℝ, |x - 2| - |x - 5| - k > 0) → k < -3 :=
by sorry

end NUMINAMATH_CALUDE_inequality_solution_range_l2612_261240


namespace NUMINAMATH_CALUDE_pentagon_shaded_probability_l2612_261237

/-- A regular pentagon game board with shaded regions -/
structure PentagonBoard where
  /-- The total number of regions formed by the diagonals -/
  total_regions : ℕ
  /-- The number of shaded regions -/
  shaded_regions : ℕ
  /-- Proof that the number of shaded regions is less than or equal to the total regions -/
  h_shaded_le_total : shaded_regions ≤ total_regions

/-- The probability of landing in a shaded region -/
def shaded_probability (board : PentagonBoard) : ℚ :=
  board.shaded_regions / board.total_regions

/-- Theorem stating the probability of landing in a shaded region for the specific game board -/
theorem pentagon_shaded_probability :
  ∃ (board : PentagonBoard),
    board.total_regions = 10 ∧
    board.shaded_regions = 3 ∧
    shaded_probability board = 3 / 10 := by
  sorry

end NUMINAMATH_CALUDE_pentagon_shaded_probability_l2612_261237


namespace NUMINAMATH_CALUDE_domain_of_f_l2612_261207

-- Define the function f
def f (x : ℝ) : ℝ := (x - 5) ^ (1/4) + (x - 6) ^ (1/5) + (x - 7) ^ (1/2)

-- State the theorem about the domain of f
theorem domain_of_f :
  {x : ℝ | ∃ y, f x = y} = {x : ℝ | x ≥ 7} :=
by
  sorry

-- Note: The proof is omitted as per instructions

end NUMINAMATH_CALUDE_domain_of_f_l2612_261207


namespace NUMINAMATH_CALUDE_presidency_meeting_ways_l2612_261285

def num_schools : Nat := 4
def members_per_school : Nat := 6
def host_representatives : Nat := 3
def non_host_representatives : Nat := 2
def seniors_per_school : Nat := 3

theorem presidency_meeting_ways :
  let choose_host := num_schools
  let host_rep_ways := Nat.choose members_per_school host_representatives
  let non_host_school_ways := Nat.choose seniors_per_school 1 * Nat.choose (members_per_school - seniors_per_school) 1
  let non_host_schools_ways := non_host_school_ways ^ (num_schools - 1)
  choose_host * host_rep_ways * non_host_schools_ways = 58320 := by
  sorry

end NUMINAMATH_CALUDE_presidency_meeting_ways_l2612_261285


namespace NUMINAMATH_CALUDE_green_toads_in_shrublands_l2612_261244

/-- Represents the different types of toads -/
inductive ToadType
| Green
| Brown
| Blue
| Red

/-- Represents the different habitats -/
inductive Habitat
| Wetlands
| Forests
| Grasslands
| Marshlands
| Shrublands

/-- The population ratio of toads -/
def populationRatio : ToadType → ℕ
| ToadType.Green => 1
| ToadType.Brown => 25
| ToadType.Blue => 10
| ToadType.Red => 20

/-- The proportion of brown toads that are spotted -/
def spottedBrownProportion : ℚ := 1/4

/-- The proportion of blue toads that are striped -/
def stripedBlueProportion : ℚ := 1/3

/-- The proportion of red toads with star pattern -/
def starPatternRedProportion : ℚ := 1/2

/-- The density of specific toad types in each habitat -/
def specificToadDensity : Habitat → ℚ
| Habitat.Wetlands => 60  -- spotted brown toads
| Habitat.Forests => 45   -- camouflaged blue toads
| Habitat.Grasslands => 100  -- star pattern red toads
| Habitat.Marshlands => 120  -- plain brown toads
| Habitat.Shrublands => 35   -- striped blue toads

/-- Theorem: The number of green toads per acre in Shrublands is 10.5 -/
theorem green_toads_in_shrublands :
  let totalBlueToads : ℚ := specificToadDensity Habitat.Shrublands / stripedBlueProportion
  let greenToads : ℚ := totalBlueToads / populationRatio ToadType.Blue
  greenToads = 10.5 := by sorry

end NUMINAMATH_CALUDE_green_toads_in_shrublands_l2612_261244


namespace NUMINAMATH_CALUDE_profit_at_35_selling_price_for_600_profit_no_900_profit_l2612_261280

/-- Represents the daily sales and profit model for a product in a shopping mall. -/
structure SalesModel where
  purchase_price : ℝ
  min_selling_price : ℝ
  max_selling_price : ℝ
  sales_volume : ℝ → ℝ
  profit : ℝ → ℝ

/-- The specific sales model for the given problem. -/
def mall_model : SalesModel :=
  { purchase_price := 30
    min_selling_price := 30
    max_selling_price := 55
    sales_volume := fun x => -2 * x + 140
    profit := fun x => (x - 30) * (-2 * x + 140) }

/-- Theorem 1: The daily profit when the selling price is 35 yuan is 350 yuan. -/
theorem profit_at_35 (model : SalesModel := mall_model) :
    model.profit 35 = 350 := by sorry

/-- Theorem 2: The selling price that yields a daily profit of 600 yuan is 40 yuan. -/
theorem selling_price_for_600_profit (model : SalesModel := mall_model) :
    ∃ x, model.min_selling_price ≤ x ∧ x ≤ model.max_selling_price ∧ model.profit x = 600 ∧ x = 40 := by sorry

/-- Theorem 3: There is no selling price within the given range that can yield a daily profit of 900 yuan. -/
theorem no_900_profit (model : SalesModel := mall_model) :
    ¬∃ x, model.min_selling_price ≤ x ∧ x ≤ model.max_selling_price ∧ model.profit x = 900 := by sorry

end NUMINAMATH_CALUDE_profit_at_35_selling_price_for_600_profit_no_900_profit_l2612_261280


namespace NUMINAMATH_CALUDE_rectangle_dimensions_l2612_261221

theorem rectangle_dimensions (vertical_side : ℝ) (square_side : ℝ) (horizontal_side : ℝ) : 
  vertical_side = 28 →
  square_side = 10 →
  (vertical_side - square_side) ^ 2 + (horizontal_side - square_side) ^ 2 = vertical_side ^ 2 →
  horizontal_side = 45 :=
by sorry

end NUMINAMATH_CALUDE_rectangle_dimensions_l2612_261221


namespace NUMINAMATH_CALUDE_shopkeeper_mango_profit_l2612_261204

/-- Calculates the profit percentage given the cost price and selling price -/
def profit_percent (cost_price selling_price : ℚ) : ℚ :=
  ((selling_price - cost_price) / cost_price) * 100

/-- Theorem: A shopkeeper who buys mangoes at 6 for 1 rupee and sells them at 3 for 1 rupee makes a 100% profit -/
theorem shopkeeper_mango_profit :
  let cost_price : ℚ := 1 / 6  -- Cost price per mango
  let selling_price : ℚ := 1 / 3  -- Selling price per mango
  profit_percent cost_price selling_price = 100 := by
sorry

end NUMINAMATH_CALUDE_shopkeeper_mango_profit_l2612_261204


namespace NUMINAMATH_CALUDE_arithmetic_sequence_13th_term_l2612_261225

def arithmetic_sequence (a : ℕ → ℤ) (d : ℤ) : Prop :=
  ∀ n : ℕ, a (n + 1) = a n + d

def geometric_sequence (a b c : ℤ) : Prop :=
  b * b = a * c

theorem arithmetic_sequence_13th_term
  (a : ℕ → ℤ)
  (d : ℤ)
  (h1 : d ≠ 0)
  (h2 : arithmetic_sequence a d)
  (h3 : geometric_sequence (a 9) (a 1) (a 5))
  (h4 : a 1 + 3 * a 5 + a 9 = 20) :
  a 13 = 28 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_13th_term_l2612_261225


namespace NUMINAMATH_CALUDE_power_product_equality_l2612_261292

theorem power_product_equality : 2^4 * 3^2 * 5^2 * 7^2 = 176400 := by
  sorry

end NUMINAMATH_CALUDE_power_product_equality_l2612_261292


namespace NUMINAMATH_CALUDE_two_valid_permutations_l2612_261246

def S : Finset ℕ := Finset.range 2022

def is_valid_permutation (A : Fin 2022 → ℕ) : Prop :=
  Function.Injective A ∧ (∀ i, A i ∈ S) ∧
  (∀ n m : Fin 2022, (A n + A m) % (Nat.gcd n.val m.val) = 0)

theorem two_valid_permutations :
  ∃! (p : Finset (Fin 2022 → ℕ)), p.card = 2 ∧ ∀ A ∈ p, is_valid_permutation A :=
sorry

end NUMINAMATH_CALUDE_two_valid_permutations_l2612_261246


namespace NUMINAMATH_CALUDE_omega_double_omega_8n_plus_5_omega_2_pow_n_minus_1_l2612_261223

-- Define a function to represent the binary expansion of a non-negative integer
def binaryExpansion (n : ℕ) : List (Fin 2) := sorry

-- Define the ω function
def ω (n : ℕ) : ℕ := (binaryExpansion n).sum

-- Theorem 1
theorem omega_double (n : ℕ) : ω (2 * n) = ω n := by sorry

-- Theorem 2
theorem omega_8n_plus_5 (n : ℕ) : ω (8 * n + 5) = ω (4 * n + 3) := by sorry

-- Theorem 3
theorem omega_2_pow_n_minus_1 (n : ℕ) : ω (2^n - 1) = n := by sorry

end NUMINAMATH_CALUDE_omega_double_omega_8n_plus_5_omega_2_pow_n_minus_1_l2612_261223


namespace NUMINAMATH_CALUDE_representative_selection_count_l2612_261217

def total_students : ℕ := 10
def female_students : ℕ := 4
def male_students : ℕ := 6
def representatives : ℕ := 3

theorem representative_selection_count : 
  (Nat.choose female_students 1 * Nat.choose male_students 2) + 
  (Nat.choose female_students 2 * Nat.choose male_students 1) + 
  (Nat.choose female_students 3) = 100 := by
  sorry

end NUMINAMATH_CALUDE_representative_selection_count_l2612_261217


namespace NUMINAMATH_CALUDE_basketball_games_left_to_play_l2612_261293

theorem basketball_games_left_to_play 
  (games_played : ℕ) 
  (win_percentage : ℚ) 
  (additional_losses : ℕ) 
  (final_win_percentage : ℚ) :
  games_played = 40 →
  win_percentage = 70 / 100 →
  additional_losses = 8 →
  final_win_percentage = 60 / 100 →
  ∃ (games_left : ℕ), games_left = 7 ∧ 
    (games_played * win_percentage).floor + (games_played + games_left - (games_played * win_percentage).floor - additional_losses) = 
    (final_win_percentage * (games_played + games_left)).floor :=
by sorry

end NUMINAMATH_CALUDE_basketball_games_left_to_play_l2612_261293


namespace NUMINAMATH_CALUDE_probability_shaded_is_two_fifths_l2612_261227

/-- A structure representing the triangle selection scenario -/
structure TriangleSelection where
  total_triangles : ℕ
  shaded_triangles : ℕ
  shaded_triangles_le_total : shaded_triangles ≤ total_triangles

/-- The probability of selecting a triangle with a shaded part -/
def probability_shaded (ts : TriangleSelection) : ℚ :=
  ts.shaded_triangles / ts.total_triangles

/-- Theorem stating that the probability of selecting a shaded triangle is 2/5 -/
theorem probability_shaded_is_two_fifths (ts : TriangleSelection) 
    (h1 : ts.total_triangles = 5)
    (h2 : ts.shaded_triangles = 2) : 
  probability_shaded ts = 2 / 5 := by
  sorry

end NUMINAMATH_CALUDE_probability_shaded_is_two_fifths_l2612_261227


namespace NUMINAMATH_CALUDE_negation_of_proposition_l2612_261256

open Real

theorem negation_of_proposition (p : ∀ x : ℝ, x > 0 → (x + 1) * Real.exp x > 1) :
  ∃ x₀ : ℝ, x₀ > 0 ∧ (x₀ + 1) * Real.exp x₀ ≤ 1 :=
by sorry

end NUMINAMATH_CALUDE_negation_of_proposition_l2612_261256


namespace NUMINAMATH_CALUDE_max_x_value_l2612_261220

theorem max_x_value (x y z : ℝ) (sum_eq : x + y + z = 7) (prod_sum_eq : x*y + x*z + y*z = 10) :
  x ≤ 3 ∧ ∃ (y' z' : ℝ), x = 3 ∧ y' + z' = 4 ∧ 3*y' + 3*z' + y'*z' = 10 :=
by sorry

end NUMINAMATH_CALUDE_max_x_value_l2612_261220


namespace NUMINAMATH_CALUDE_distinct_arrangements_apples_l2612_261233

def word_length : ℕ := 6
def repeated_letter_count : ℕ := 2
def single_letter_count : ℕ := 1
def number_of_single_letters : ℕ := 4

theorem distinct_arrangements_apples :
  (word_length.factorial) / (repeated_letter_count.factorial * (single_letter_count.factorial ^ number_of_single_letters)) = 360 := by
  sorry

end NUMINAMATH_CALUDE_distinct_arrangements_apples_l2612_261233


namespace NUMINAMATH_CALUDE_work_completion_days_l2612_261232

theorem work_completion_days (total_men : ℕ) (absent_men : ℕ) (reduced_days : ℕ) 
  (h1 : total_men = 60)
  (h2 : absent_men = 10)
  (h3 : reduced_days = 60) :
  let remaining_men := total_men - absent_men
  let original_days := (remaining_men * reduced_days) / total_men
  original_days = 50 := by
  sorry

end NUMINAMATH_CALUDE_work_completion_days_l2612_261232


namespace NUMINAMATH_CALUDE_sum_of_roots_for_non_intersecting_lines_l2612_261289

/-- Parabola defined by y = x^2 + 4 -/
def P : ℝ → ℝ := λ x ↦ x^2 + 4

/-- Point Q with coordinates (10, 6) -/
def Q : ℝ × ℝ := (10, 6)

/-- Theorem stating that the sum of roots of m^2 - 40m + 8 = 0 is 40 -/
theorem sum_of_roots_for_non_intersecting_lines (P : ℝ → ℝ) (Q : ℝ × ℝ) :
  P = (λ x ↦ x^2 + 4) →
  Q = (10, 6) →
  let r := (40 - Real.sqrt 1568) / 2
  let s := (40 + Real.sqrt 1568) / 2
  r + s = 40 := by sorry

end NUMINAMATH_CALUDE_sum_of_roots_for_non_intersecting_lines_l2612_261289


namespace NUMINAMATH_CALUDE_jason_shampoo_time_l2612_261210

theorem jason_shampoo_time :
  ∀ (J : ℝ),
  J > 0 →
  (1 / J + 1 / 6 = 1 / 2) →
  J = 3 :=
by sorry

end NUMINAMATH_CALUDE_jason_shampoo_time_l2612_261210


namespace NUMINAMATH_CALUDE_arithmetic_error_correction_l2612_261231

theorem arithmetic_error_correction : ∃! x : ℝ, 3 * x - 4 = x / 3 + 4 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_error_correction_l2612_261231


namespace NUMINAMATH_CALUDE_newton_interpolation_polynomial_l2612_261291

/-- The interpolation polynomial -/
def P (x : ℝ) : ℝ := 2 * x^2 - 5 * x + 3

/-- The given points -/
def x₀ : ℝ := 2
def x₁ : ℝ := 4
def x₂ : ℝ := 5

/-- The given function values -/
def y₀ : ℝ := 1
def y₁ : ℝ := 15
def y₂ : ℝ := 28

theorem newton_interpolation_polynomial :
  P x₀ = y₀ ∧ P x₁ = y₁ ∧ P x₂ = y₂ ∧
  ∀ Q : ℝ → ℝ, (Q x₀ = y₀ ∧ Q x₁ = y₁ ∧ Q x₂ = y₂) →
  (∃ a b c : ℝ, ∀ x, Q x = a * x^2 + b * x + c) →
  (∀ x, Q x = P x) :=
sorry

end NUMINAMATH_CALUDE_newton_interpolation_polynomial_l2612_261291


namespace NUMINAMATH_CALUDE_geometric_sequence_implies_b_eq_4_b_eq_4_not_sufficient_geometric_sequence_sufficient_not_necessary_l2612_261253

/-- A geometric sequence with first term 1, fifth term 16, and middle terms a, b, c -/
def is_geometric_sequence (a b c : ℝ) : Prop :=
  ∃ q : ℝ, q ≠ 0 ∧ a = q ∧ b = q^2 ∧ c = q^3 ∧ 16 = q^4

/-- The statement that b = 4 is a necessary condition for the geometric sequence -/
theorem geometric_sequence_implies_b_eq_4 (a b c : ℝ) :
  is_geometric_sequence a b c → b = 4 :=
sorry

/-- The statement that b = 4 is not a sufficient condition for the geometric sequence -/
theorem b_eq_4_not_sufficient (a b c : ℝ) :
  b = 4 → ¬(∀ a c : ℝ, is_geometric_sequence a b c) :=
sorry

/-- The main theorem stating that the geometric sequence condition is sufficient but not necessary for b = 4 -/
theorem geometric_sequence_sufficient_not_necessary :
  (∃ a b c : ℝ, is_geometric_sequence a b c ∧ b = 4) ∧
  (∃ b : ℝ, b = 4 ∧ ¬(∀ a c : ℝ, is_geometric_sequence a b c)) :=
sorry

end NUMINAMATH_CALUDE_geometric_sequence_implies_b_eq_4_b_eq_4_not_sufficient_geometric_sequence_sufficient_not_necessary_l2612_261253


namespace NUMINAMATH_CALUDE_f_is_increasing_l2612_261249

def f (x : ℝ) := 2 * x + 1

theorem f_is_increasing : Monotone f := by sorry

end NUMINAMATH_CALUDE_f_is_increasing_l2612_261249


namespace NUMINAMATH_CALUDE_twentieth_digit_sum_one_thirteenth_one_eleventh_l2612_261278

/-- The decimal representation of a rational number -/
def decimalRepresentation (q : ℚ) : ℕ → ℕ := sorry

/-- The sum of decimal representations of two rational numbers -/
def sumDecimalRepresentations (q₁ q₂ : ℚ) : ℕ → ℕ := sorry

/-- The nth digit after the decimal point in a decimal representation -/
def nthDigitAfterDecimal (f : ℕ → ℕ) (n : ℕ) : ℕ := sorry

theorem twentieth_digit_sum_one_thirteenth_one_eleventh :
  nthDigitAfterDecimal (sumDecimalRepresentations (1/13) (1/11)) 20 = 6 := by sorry

end NUMINAMATH_CALUDE_twentieth_digit_sum_one_thirteenth_one_eleventh_l2612_261278


namespace NUMINAMATH_CALUDE_shooting_probability_l2612_261209

/-- The probability of scoring less than 9 in a shooting practice -/
def prob_less_than_9 (prob_10 prob_9 prob_8 : ℝ) : Prop :=
  prob_10 = 0.24 ∧ prob_9 = 0.28 ∧ prob_8 = 0.19 →
  1 - (prob_10 + prob_9) = 0.29

theorem shooting_probability : 
  ∃ (prob_10 prob_9 prob_8 : ℝ), prob_less_than_9 prob_10 prob_9 prob_8 := by
  sorry

end NUMINAMATH_CALUDE_shooting_probability_l2612_261209


namespace NUMINAMATH_CALUDE_smallest_side_of_triangle_l2612_261258

/-- Given a triangle ABC with sides a, b, and c satisfying b^2 + c^2 ≥ 5a^2, 
    BC is the smallest side of the triangle. -/
theorem smallest_side_of_triangle (a b c : ℝ) (h_triangle : a > 0 ∧ b > 0 ∧ c > 0) 
    (h_condition : b^2 + c^2 ≥ 5*a^2) : 
    c ≤ a ∧ c ≤ b := by
  sorry

end NUMINAMATH_CALUDE_smallest_side_of_triangle_l2612_261258


namespace NUMINAMATH_CALUDE_game_result_l2612_261284

def g (n : ℕ) : ℕ :=
  if n % 6 = 0 then 8
  else if n % 3 = 0 then 4
  else if n % 2 = 0 then 3
  else 1

def cora_rolls : List ℕ := [5, 4, 3, 6, 2, 1]
def dana_rolls : List ℕ := [6, 3, 4, 3, 5, 3]

def total_points (rolls : List ℕ) : ℕ :=
  rolls.map g |>.sum

theorem game_result :
  (total_points cora_rolls) * (total_points dana_rolls) = 480 := by
  sorry

end NUMINAMATH_CALUDE_game_result_l2612_261284


namespace NUMINAMATH_CALUDE_brown_gumdrops_after_replacement_l2612_261275

/-- Theorem about the number of brown gumdrops after replacement in a jar --/
theorem brown_gumdrops_after_replacement (total : ℕ) (green blue brown red yellow : ℕ) :
  total = 200 →
  green = 40 →
  blue = 50 →
  brown = 60 →
  red = 20 →
  yellow = 30 →
  (brown + (red / 3 : ℕ)) = 67 := by
  sorry

#check brown_gumdrops_after_replacement

end NUMINAMATH_CALUDE_brown_gumdrops_after_replacement_l2612_261275


namespace NUMINAMATH_CALUDE_dividend_calculation_l2612_261206

theorem dividend_calculation (dividend divisor : ℕ) : 
  (dividend / divisor = 4) → 
  (dividend % divisor = 3) → 
  (dividend + divisor + 4 + 3 = 100) → 
  dividend = 75 := by
sorry

end NUMINAMATH_CALUDE_dividend_calculation_l2612_261206


namespace NUMINAMATH_CALUDE_lottery_tickets_bought_l2612_261201

theorem lottery_tickets_bought (total_won : ℕ) (winning_number_value : ℕ) (winning_numbers_per_ticket : ℕ) : 
  total_won = 300 →
  winning_number_value = 20 →
  winning_numbers_per_ticket = 5 →
  (total_won / winning_number_value) / winning_numbers_per_ticket = 3 :=
by sorry

end NUMINAMATH_CALUDE_lottery_tickets_bought_l2612_261201


namespace NUMINAMATH_CALUDE_inequality_proof_l2612_261267

theorem inequality_proof (x y m : ℝ) (h1 : x > y) (h2 : m > 0) : x - y > 0 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l2612_261267


namespace NUMINAMATH_CALUDE_rosys_age_l2612_261279

/-- Proves Rosy's current age given the conditions of the problem -/
theorem rosys_age :
  ∀ (rosy_age : ℕ),
  (∃ (david_age : ℕ),
    david_age = rosy_age + 18 ∧
    david_age + 6 = 2 * (rosy_age + 6)) →
  rosy_age = 12 :=
by
  sorry

end NUMINAMATH_CALUDE_rosys_age_l2612_261279


namespace NUMINAMATH_CALUDE_final_amount_calculation_l2612_261257

def initial_amount : ℕ := 5
def spent_amount : ℕ := 2
def allowance : ℕ := 26

theorem final_amount_calculation :
  initial_amount - spent_amount + allowance = 29 := by
  sorry

end NUMINAMATH_CALUDE_final_amount_calculation_l2612_261257


namespace NUMINAMATH_CALUDE_horner_method_v2_l2612_261260

def horner_polynomial (x : ℝ) : ℝ := x^5 + 5*x^4 + 10*x^3 + 10*x^2 + 5*x + 1

def horner_v0 : ℝ := 1

def horner_v1 (x : ℝ) : ℝ := horner_v0 * x + 5

def horner_v2 (x : ℝ) : ℝ := horner_v1 x * x + 10

theorem horner_method_v2 :
  horner_v2 2 = 24 :=
by sorry

end NUMINAMATH_CALUDE_horner_method_v2_l2612_261260


namespace NUMINAMATH_CALUDE_point_same_side_as_origin_l2612_261235

def same_side_as_origin (x y : ℝ) : Prop :=
  (3 * x + 2 * y + 5) * (3 * 0 + 2 * 0 + 5) > 0

theorem point_same_side_as_origin :
  same_side_as_origin (-3) 4 := by sorry

end NUMINAMATH_CALUDE_point_same_side_as_origin_l2612_261235


namespace NUMINAMATH_CALUDE_point_coordinates_l2612_261242

/-- A point in the Cartesian coordinate system -/
structure Point where
  x : ℝ
  y : ℝ

/-- Checks if a point is in the fourth quadrant -/
def is_fourth_quadrant (p : Point) : Prop :=
  p.x > 0 ∧ p.y < 0

/-- Distance of a point to the x-axis -/
def distance_to_x_axis (p : Point) : ℝ :=
  |p.y|

/-- Distance of a point to the y-axis -/
def distance_to_y_axis (p : Point) : ℝ :=
  |p.x|

theorem point_coordinates (P : Point) 
  (h1 : is_fourth_quadrant P)
  (h2 : distance_to_x_axis P = 2)
  (h3 : distance_to_y_axis P = 5) :
  P.x = 5 ∧ P.y = -2 := by
  sorry

end NUMINAMATH_CALUDE_point_coordinates_l2612_261242


namespace NUMINAMATH_CALUDE_expression_simplification_l2612_261213

theorem expression_simplification 
  (p q r x : ℝ) 
  (h_distinct : p ≠ q ∧ q ≠ r ∧ p ≠ r) : 
  ((x + p)^4) / ((p - q)*(p - r)) + 
  ((x + q)^4) / ((q - p)*(q - r)) + 
  ((x + r)^4) / ((r - p)*(r - q)) = 
  p + q + r + 4*x := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l2612_261213


namespace NUMINAMATH_CALUDE_ball_probability_l2612_261268

theorem ball_probability (x : ℕ) : 
  (6 : ℝ) / ((6 : ℝ) + x) = (3 : ℝ) / 10 → x = 14 := by
  sorry

end NUMINAMATH_CALUDE_ball_probability_l2612_261268


namespace NUMINAMATH_CALUDE_birthday_savings_growth_l2612_261272

/-- Calculates the final amount in a bank account after one year, given an initial amount and an annual interest rate. -/
def final_amount (initial_amount : ℝ) (interest_rate : ℝ) : ℝ :=
  initial_amount * (1 + interest_rate)

/-- Theorem: Given an initial amount of $90 and an annual interest rate of 10%, 
    the final amount after 1 year with no withdrawals is $99. -/
theorem birthday_savings_growth : final_amount 90 0.1 = 99 := by
  sorry

end NUMINAMATH_CALUDE_birthday_savings_growth_l2612_261272


namespace NUMINAMATH_CALUDE_cubic_curve_tangent_line_bc_product_l2612_261202

/-- Given a cubic curve y = x³ + bx + c passing through (1, 2) with tangent line y = x + 1 at that point, 
    the product bc equals -6. -/
theorem cubic_curve_tangent_line_bc_product (b c : ℝ) : 
  (1^3 + b*1 + c = 2) →   -- Point (1, 2) is on the curve
  (3*1^2 + b = 1) →       -- Derivative at x = 1 is 1 (from tangent line y = x + 1)
  b * c = -6 := by sorry

end NUMINAMATH_CALUDE_cubic_curve_tangent_line_bc_product_l2612_261202


namespace NUMINAMATH_CALUDE_largest_divisor_is_24_l2612_261236

/-- The set of all integer tuples (a, b, c, d, e, f) satisfying a^2 + b^2 + c^2 + d^2 + e^2 = f^2 -/
def S : Set (ℤ × ℤ × ℤ × ℤ × ℤ × ℤ) :=
  {t | let (a, b, c, d, e, f) := t
       a^2 + b^2 + c^2 + d^2 + e^2 = f^2}

/-- The property that k divides the product of all elements in a tuple -/
def DividesTuple (k : ℤ) (t : ℤ × ℤ × ℤ × ℤ × ℤ × ℤ) : Prop :=
  let (a, b, c, d, e, f) := t
  k ∣ (a * b * c * d * e * f)

theorem largest_divisor_is_24 :
  ∃ (k : ℤ), k = 24 ∧ (∀ t ∈ S, DividesTuple k t) ∧
  (∀ m : ℤ, (∀ t ∈ S, DividesTuple m t) → m ≤ k) :=
sorry

end NUMINAMATH_CALUDE_largest_divisor_is_24_l2612_261236


namespace NUMINAMATH_CALUDE_log_problem_l2612_261265

-- Define the logarithm function for base 3
noncomputable def log3 (y : ℝ) : ℝ := Real.log y / Real.log 3

-- Define the logarithm function for base 9
noncomputable def log9 (y : ℝ) : ℝ := Real.log y / Real.log 9

theorem log_problem (x : ℝ) (h : log3 (x + 1) = 4) : log9 x = 2 := by
  sorry

end NUMINAMATH_CALUDE_log_problem_l2612_261265


namespace NUMINAMATH_CALUDE_modulo_thirteen_equivalence_l2612_261245

theorem modulo_thirteen_equivalence : 
  ∃! n : ℤ, 0 ≤ n ∧ n < 13 ∧ 52801 ≡ n [ZMOD 13] ∧ n = 8 := by
  sorry

end NUMINAMATH_CALUDE_modulo_thirteen_equivalence_l2612_261245


namespace NUMINAMATH_CALUDE_twice_x_minus_three_l2612_261211

/-- The algebraic expression for "twice x minus 3" is equal to 2x - 3. -/
theorem twice_x_minus_three (x : ℝ) : 2 * x - 3 = 2 * x - 3 := by
  sorry

end NUMINAMATH_CALUDE_twice_x_minus_three_l2612_261211


namespace NUMINAMATH_CALUDE_fruit_salad_cherries_l2612_261200

theorem fruit_salad_cherries (b r g c : ℕ) : 
  b + r + g + c = 580 →
  r = 2 * b →
  g = 3 * c →
  c = 3 * r →
  c = 129 := by
sorry

end NUMINAMATH_CALUDE_fruit_salad_cherries_l2612_261200


namespace NUMINAMATH_CALUDE_ratio_from_mean_ratio_l2612_261259

theorem ratio_from_mean_ratio (a b : ℝ) (h : a > 0 ∧ b > 0) :
  (a + b) / 2 / Real.sqrt (a * b) = 25 / 24 →
  a / b = 16 / 9 ∨ a / b = 9 / 16 := by
sorry

end NUMINAMATH_CALUDE_ratio_from_mean_ratio_l2612_261259


namespace NUMINAMATH_CALUDE_max_sum_of_remaining_pairs_l2612_261255

/-- Given a set of four distinct real numbers, this function returns the list of their six pairwise sums. -/
def pairwiseSums (a b c d : ℝ) : List ℝ :=
  [a + b, a + c, a + d, b + c, b + d, c + d]

/-- This theorem states that given four distinct real numbers whose pairwise sums include 210, 360, 330, and 300,
    the maximum possible sum of the remaining two pairwise sums is 870. -/
theorem max_sum_of_remaining_pairs (a b c d : ℝ) :
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d →
  (∃ (l : List ℝ), l = pairwiseSums a b c d ∧ 
    (210 ∈ l) ∧ (360 ∈ l) ∧ (330 ∈ l) ∧ (300 ∈ l)) →
  (∃ (x y : ℝ), x ∈ pairwiseSums a b c d ∧ 
                y ∈ pairwiseSums a b c d ∧ 
                x ≠ 210 ∧ x ≠ 360 ∧ x ≠ 330 ∧ x ≠ 300 ∧
                y ≠ 210 ∧ y ≠ 360 ∧ y ≠ 330 ∧ y ≠ 300 ∧
                x + y ≤ 870) :=
by sorry


end NUMINAMATH_CALUDE_max_sum_of_remaining_pairs_l2612_261255


namespace NUMINAMATH_CALUDE_grid_value_theorem_l2612_261295

/-- Represents a 7x2 grid of rational numbers -/
def Grid := Fin 7 → Fin 2 → ℚ

/-- The main column forms an arithmetic sequence -/
def is_main_column_arithmetic (g : Grid) : Prop :=
  ∃ d : ℚ, ∀ i : Fin 6, g (i + 1) 0 - g i 0 = d

/-- The first two rows form arithmetic sequences -/
def are_first_two_rows_arithmetic (g : Grid) : Prop :=
  ∃ d₁ d₂ : ℚ, (g 0 1 - g 0 0 = d₁) ∧ (g 1 1 - g 1 0 = d₂)

/-- The grid satisfies the given conditions -/
def satisfies_conditions (g : Grid) : Prop :=
  (g 0 0 = -9) ∧ (g 3 0 = 56) ∧ (g 6 1 = 16) ∧
  is_main_column_arithmetic g ∧
  are_first_two_rows_arithmetic g

theorem grid_value_theorem (g : Grid) (h : satisfies_conditions g) : g 4 1 = -851/3 := by
  sorry

end NUMINAMATH_CALUDE_grid_value_theorem_l2612_261295


namespace NUMINAMATH_CALUDE_sum_equals_quadratic_l2612_261212

open BigOperators

/-- The sequence a_n defined as 4n - 3 -/
def a (n : ℕ) : ℤ := 4 * n - 3

/-- The sum of the first n terms of the sequence -/
def S (n : ℕ) : ℤ := ∑ i in Finset.range n, a (i + 1)

/-- The theorem stating the equality between the sum and the quadratic expression -/
theorem sum_equals_quadratic (a b c : ℤ) : 
  (∀ n : ℕ, n > 0 → S n = 2 * a * n^2 + b * n + c) →
  a - b + c = 2 := by
  sorry


end NUMINAMATH_CALUDE_sum_equals_quadratic_l2612_261212


namespace NUMINAMATH_CALUDE_roots_sum_reciprocal_l2612_261222

theorem roots_sum_reciprocal (x₁ x₂ : ℝ) : 
  (5 * x₁^2 - 3 * x₁ - 2 = 0) → 
  (5 * x₂^2 - 3 * x₂ - 2 = 0) → 
  x₁ ≠ x₂ →
  (1 / x₁ + 1 / x₂ = -3 / 2) :=
by sorry

end NUMINAMATH_CALUDE_roots_sum_reciprocal_l2612_261222


namespace NUMINAMATH_CALUDE_rectangle_length_l2612_261290

/-- Proves that a rectangle with perimeter to width ratio of 5:1 and area 150 has length 15 -/
theorem rectangle_length (w l : ℝ) (h1 : w > 0) (h2 : l > 0) : 
  (2 * l + 2 * w) / w = 5 → l * w = 150 → l = 15 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_length_l2612_261290


namespace NUMINAMATH_CALUDE_initial_average_marks_l2612_261277

theorem initial_average_marks
  (n : ℕ)  -- number of students
  (correct_avg : ℚ)  -- correct average after fixing the error
  (wrong_mark : ℚ)  -- wrongly noted mark
  (right_mark : ℚ)  -- correct mark
  (h1 : n = 30)  -- there are 30 students
  (h2 : correct_avg = 98)  -- correct average is 98
  (h3 : wrong_mark = 70)  -- wrongly noted mark is 70
  (h4 : right_mark = 10)  -- correct mark is 10
  : (n * correct_avg + (right_mark - wrong_mark)) / n = 100 :=
by sorry

end NUMINAMATH_CALUDE_initial_average_marks_l2612_261277


namespace NUMINAMATH_CALUDE_range_of_a_l2612_261203

theorem range_of_a (a : ℝ) : 
  (∀ x > 0, Real.log x - a * x ≤ 2 * a^2 - 3) → a ≥ 1 := by
  sorry

end NUMINAMATH_CALUDE_range_of_a_l2612_261203


namespace NUMINAMATH_CALUDE_gum_sharing_proof_l2612_261247

/-- The number of people sharing gum equally -/
def num_people (john_gum cole_gum aubrey_gum pieces_per_person : ℕ) : ℕ :=
  (john_gum + cole_gum + aubrey_gum) / pieces_per_person

/-- Proof that 3 people are sharing the gum -/
theorem gum_sharing_proof :
  num_people 54 45 0 33 = 3 := by
  sorry

end NUMINAMATH_CALUDE_gum_sharing_proof_l2612_261247


namespace NUMINAMATH_CALUDE_polynomial_coefficient_B_l2612_261266

theorem polynomial_coefficient_B (A C D : ℤ) : 
  ∃ (r₁ r₂ r₃ r₄ r₅ r₆ : ℕ+), 
    (r₁ : ℤ) + r₂ + r₃ + r₄ + r₅ + r₆ = 10 →
    ∀ (z : ℂ), z^6 - 10*z^5 + A*z^4 + (-108)*z^3 + C*z^2 + D*z + 16 = 
      (z - r₁) * (z - r₂) * (z - r₃) * (z - r₄) * (z - r₅) * (z - r₆) :=
by sorry

end NUMINAMATH_CALUDE_polynomial_coefficient_B_l2612_261266


namespace NUMINAMATH_CALUDE_cos_180_degrees_l2612_261283

theorem cos_180_degrees : Real.cos (Real.pi) = -1 := by
  sorry

end NUMINAMATH_CALUDE_cos_180_degrees_l2612_261283


namespace NUMINAMATH_CALUDE_polynomial_coefficient_equality_l2612_261224

theorem polynomial_coefficient_equality (x : ℝ) : 
  ∃ (s : ℝ), (3*x^3 - 2*x^2 + x + 6)*(2*x^3 + s*x^2 + 3*x + 5) = 
              6*x^6 + s*x^5 + 5*x^4 + 17*x^3 + 10*x^2 + 33*x + 30 ∧ 
              s = 4 :=
by
  sorry

end NUMINAMATH_CALUDE_polynomial_coefficient_equality_l2612_261224


namespace NUMINAMATH_CALUDE_ninas_allowance_l2612_261250

theorem ninas_allowance (game_cost : ℝ) (tax_rate : ℝ) (savings_rate : ℝ) (weeks : ℕ) :
  game_cost = 50 →
  tax_rate = 0.1 →
  savings_rate = 0.5 →
  weeks = 11 →
  ∃ (allowance : ℝ),
    allowance * savings_rate * weeks = game_cost * (1 + tax_rate) ∧
    allowance = 10 := by
  sorry

end NUMINAMATH_CALUDE_ninas_allowance_l2612_261250


namespace NUMINAMATH_CALUDE_k_value_theorem_l2612_261299

theorem k_value_theorem (x y z k : ℝ) 
  (h1 : (y + z) / x = k)
  (h2 : (z + x) / y = k)
  (h3 : (x + y) / z = k)
  (h4 : x ≠ 0)
  (h5 : y ≠ 0)
  (h6 : z ≠ 0) :
  k = 2 ∨ k = -1 := by
sorry

end NUMINAMATH_CALUDE_k_value_theorem_l2612_261299


namespace NUMINAMATH_CALUDE_gwen_total_books_l2612_261238

/-- The number of books on each shelf -/
def books_per_shelf : ℕ := 9

/-- The number of shelves for mystery books -/
def mystery_shelves : ℕ := 3

/-- The number of shelves for picture books -/
def picture_shelves : ℕ := 5

/-- The total number of books Gwen has -/
def total_books : ℕ := books_per_shelf * (mystery_shelves + picture_shelves)

theorem gwen_total_books : total_books = 72 := by sorry

end NUMINAMATH_CALUDE_gwen_total_books_l2612_261238


namespace NUMINAMATH_CALUDE_negation_of_all_linear_functions_are_monotonic_l2612_261261

-- Define the type of functions from real numbers to real numbers
def RealFunction := ℝ → ℝ

-- Define what it means for a function to be linear
def IsLinear (f : RealFunction) : Prop := ∀ x y : ℝ, ∀ c : ℝ, f (c * x + y) = c * f x + f y

-- Define what it means for a function to be monotonic
def IsMonotonic (f : RealFunction) : Prop := ∀ x y : ℝ, x ≤ y → f x ≤ f y

-- State the theorem
theorem negation_of_all_linear_functions_are_monotonic :
  (¬ ∀ f : RealFunction, IsLinear f → IsMonotonic f) ↔
  (∃ f : RealFunction, IsLinear f ∧ ¬IsMonotonic f) :=
sorry

end NUMINAMATH_CALUDE_negation_of_all_linear_functions_are_monotonic_l2612_261261


namespace NUMINAMATH_CALUDE_harris_feeds_one_carrot_per_day_l2612_261264

/-- Represents the number of carrots Harris feeds his dog per day -/
def carrots_per_day : ℚ :=
  let carrots_per_bag : ℕ := 5
  let cost_per_bag : ℚ := 2
  let annual_spend : ℚ := 146
  let days_per_year : ℕ := 365
  (annual_spend / days_per_year.cast) / cost_per_bag * carrots_per_bag

/-- Proves that Harris feeds his dog 1 carrot per day -/
theorem harris_feeds_one_carrot_per_day : 
  carrots_per_day = 1 := by sorry

end NUMINAMATH_CALUDE_harris_feeds_one_carrot_per_day_l2612_261264


namespace NUMINAMATH_CALUDE_bowtie_equation_solution_l2612_261251

-- Define the operation ⊗
noncomputable def bowtie (a b : ℝ) : ℝ :=
  a + Real.sqrt (b + Real.sqrt (b + Real.sqrt (b + Real.sqrt b)))

-- Theorem statement
theorem bowtie_equation_solution :
  ∃ y : ℝ, bowtie 5 y = 12 → y = 42 :=
by sorry

end NUMINAMATH_CALUDE_bowtie_equation_solution_l2612_261251


namespace NUMINAMATH_CALUDE_matrix19_sum_nonzero_l2612_261282

def Matrix19 := Fin 19 → Fin 19 → Int

def isValidMatrix (A : Matrix19) : Prop :=
  ∀ i j, A i j = 1 ∨ A i j = -1

def rowProduct (A : Matrix19) (i : Fin 19) : Int :=
  (Finset.univ.prod fun j => A i j)

def colProduct (A : Matrix19) (j : Fin 19) : Int :=
  (Finset.univ.prod fun i => A i j)

theorem matrix19_sum_nonzero (A : Matrix19) (h : isValidMatrix A) :
  (Finset.univ.sum fun i => rowProduct A i) + (Finset.univ.sum fun j => colProduct A j) ≠ 0 := by
  sorry

end NUMINAMATH_CALUDE_matrix19_sum_nonzero_l2612_261282


namespace NUMINAMATH_CALUDE_sin_sum_product_l2612_261271

theorem sin_sum_product (x : ℝ) : 
  Real.sin (7 * x) + Real.sin (9 * x) = 2 * Real.sin (8 * x) * Real.cos x := by
  sorry

end NUMINAMATH_CALUDE_sin_sum_product_l2612_261271


namespace NUMINAMATH_CALUDE_max_routes_in_network_l2612_261216

/-- A bus route network -/
structure BusNetwork where
  stops : Nat
  routes : Nat
  stops_per_route : Nat
  route_intersection : Nat

/-- The condition that any two routes either have no common stops or have exactly one common stop -/
def valid_intersection (network : BusNetwork) : Prop :=
  network.route_intersection = 0 ∨ network.route_intersection = 1

/-- The maximum number of routes possible given the constraints -/
def max_routes (network : BusNetwork) : Prop :=
  network.routes ≤ (network.stops * 4) / 3 ∧
  network.routes = 12

/-- Theorem stating the maximum number of routes in the given network -/
theorem max_routes_in_network (network : BusNetwork) 
  (h1 : network.stops = 9)
  (h2 : network.stops_per_route = 3)
  (h3 : valid_intersection network) :
  max_routes network :=
sorry

end NUMINAMATH_CALUDE_max_routes_in_network_l2612_261216


namespace NUMINAMATH_CALUDE_sum_reciprocal_inequality_root_inequality_l2612_261287

-- Problem 1
theorem sum_reciprocal_inequality (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  (a + b + c) * (1/a + 1/b + 1/c) ≥ 9 := by sorry

-- Problem 2
theorem root_inequality (x y : ℝ) (hx : x > 0) (hy : y > 0) :
  Real.sqrt (x^2 + y^2) > (x^3 + y^3)^(1/3) := by sorry

end NUMINAMATH_CALUDE_sum_reciprocal_inequality_root_inequality_l2612_261287


namespace NUMINAMATH_CALUDE_fractional_equation_solution_l2612_261286

theorem fractional_equation_solution :
  ∃! x : ℝ, x ≠ 1 ∧ x ≠ -1 ∧ (1 / (x - 1) + 1 = 2 / (x^2 - 1)) ∧ x = -2 := by
  sorry

end NUMINAMATH_CALUDE_fractional_equation_solution_l2612_261286


namespace NUMINAMATH_CALUDE_all_black_after_two_rotations_l2612_261239

/-- Represents a 4x4 grid where each cell can be either black or white -/
def Grid := Fin 4 → Fin 4 → Bool

/-- Returns true if the given position is on a diagonal of a 4x4 grid -/
def isDiagonal (row col : Fin 4) : Bool :=
  row = col ∨ row + col = 3

/-- Initial grid configuration with black diagonals -/
def initialGrid : Grid :=
  fun row col => isDiagonal row col

/-- Rotates a position 90 degrees clockwise in a 4x4 grid -/
def rotate (row col : Fin 4) : Fin 4 × Fin 4 :=
  (col, 3 - row)

/-- Applies the transformation rule after rotation -/
def transform (g : Grid) : Grid :=
  fun row col =>
    let (oldRow, oldCol) := rotate row col
    g row col ∨ initialGrid oldRow oldCol

/-- Applies two consecutive 90° rotations and transformations -/
def finalGrid : Grid :=
  transform (transform initialGrid)

/-- Theorem stating that all squares in the final grid are black -/
theorem all_black_after_two_rotations :
  ∀ row col, finalGrid row col = true := by sorry

end NUMINAMATH_CALUDE_all_black_after_two_rotations_l2612_261239


namespace NUMINAMATH_CALUDE_percent_less_than_l2612_261252

theorem percent_less_than (P Q : ℝ) (h : P < Q) :
  (Q - P) / Q * 100 = 100 * (Q - P) / Q :=
by sorry

end NUMINAMATH_CALUDE_percent_less_than_l2612_261252


namespace NUMINAMATH_CALUDE_ruels_usable_stamps_l2612_261205

/-- The number of usable stamps Ruel has -/
def usable_stamps : ℕ :=
  let books_10 := 4
  let stamps_per_book_10 := 10
  let books_15 := 6
  let stamps_per_book_15 := 15
  let books_25 := 3
  let stamps_per_book_25 := 25
  let books_30 := 2
  let stamps_per_book_30 := 30
  let damaged_25 := 5
  let damaged_30 := 3
  let total_stamps := books_10 * stamps_per_book_10 +
                      books_15 * stamps_per_book_15 +
                      books_25 * stamps_per_book_25 +
                      books_30 * stamps_per_book_30
  let total_damaged := damaged_25 + damaged_30
  total_stamps - total_damaged

theorem ruels_usable_stamps :
  usable_stamps = 257 := by
  sorry

end NUMINAMATH_CALUDE_ruels_usable_stamps_l2612_261205


namespace NUMINAMATH_CALUDE_system_solution_l2612_261274

theorem system_solution : ∃ (x y : ℝ), 
  x^2 + y * Real.sqrt (x * y) = 105 ∧
  y^2 + x * Real.sqrt (x * y) = 70 ∧
  x = 9 ∧ y = 4 := by
  sorry

end NUMINAMATH_CALUDE_system_solution_l2612_261274


namespace NUMINAMATH_CALUDE_polygon_sides_l2612_261230

theorem polygon_sides (n : ℕ) (h : (n - 2) * 180 = 360 * 3) : n = 8 := by
  sorry

end NUMINAMATH_CALUDE_polygon_sides_l2612_261230


namespace NUMINAMATH_CALUDE_proportional_relationship_l2612_261298

theorem proportional_relationship (x y z : ℝ) (k₁ k₂ : ℝ) :
  (∃ k₁ > 0, x = k₁ * y^3) →
  (∃ k₂ > 0, y = k₂ / z^2) →
  (x = 8 ∧ z = 16) →
  (z = 64 → x = 1/256) :=
by sorry

end NUMINAMATH_CALUDE_proportional_relationship_l2612_261298


namespace NUMINAMATH_CALUDE_unique_two_digit_reverse_ratio_l2612_261296

def is_two_digit (n : ℕ) : Prop := 10 ≤ n ∧ n < 100

def reverse_digits (n : ℕ) : ℕ :=
  (n % 10) * 10 + (n / 10)

theorem unique_two_digit_reverse_ratio :
  ∃! n : ℕ, is_two_digit n ∧ (n : ℚ) / (reverse_digits n : ℚ) = 7 / 4 :=
by
  -- The proof would go here
  sorry

end NUMINAMATH_CALUDE_unique_two_digit_reverse_ratio_l2612_261296


namespace NUMINAMATH_CALUDE_friend_riding_area_l2612_261294

/-- Given a rectangular riding area of width 2 and length 3, 
    prove that another area 4 times larger is 24 square blocks. -/
theorem friend_riding_area (width : ℕ) (length : ℕ) (multiplier : ℕ) : 
  width = 2 → length = 3 → multiplier = 4 → 
  (width * length * multiplier : ℕ) = 24 := by
  sorry

end NUMINAMATH_CALUDE_friend_riding_area_l2612_261294


namespace NUMINAMATH_CALUDE_y_squared_plus_7y_plus_12_range_l2612_261218

theorem y_squared_plus_7y_plus_12_range (y : ℝ) (h : y^2 - 7*y + 12 < 0) :
  42 < y^2 + 7*y + 12 ∧ y^2 + 7*y + 12 < 56 := by
  sorry

end NUMINAMATH_CALUDE_y_squared_plus_7y_plus_12_range_l2612_261218


namespace NUMINAMATH_CALUDE_consecutive_interior_equal_parallel_false_l2612_261208

-- Define the concept of lines
variable (Line : Type)

-- Define the concept of angles
variable (Angle : Type)

-- Define what it means for lines to be parallel
variable (parallel : Line → Line → Prop)

-- Define what it means for angles to be consecutive interior angles
variable (consecutive_interior : Angle → Angle → Line → Line → Prop)

-- Define what it means for angles to be equal
variable (angle_equal : Angle → Angle → Prop)

-- Statement to be proven false
theorem consecutive_interior_equal_parallel_false :
  ¬(∀ (l1 l2 : Line) (a1 a2 : Angle), 
    consecutive_interior a1 a2 l1 l2 → angle_equal a1 a2 → parallel l1 l2) :=
sorry

end NUMINAMATH_CALUDE_consecutive_interior_equal_parallel_false_l2612_261208


namespace NUMINAMATH_CALUDE_population_size_l2612_261215

/-- Given a population with specific birth and death rates, prove the initial population size. -/
theorem population_size (P : ℝ) 
  (birth_rate : ℝ) (death_rate : ℝ) (net_growth_rate : ℝ)
  (h1 : birth_rate = 32)
  (h2 : death_rate = 11)
  (h3 : net_growth_rate = 2.1)
  (h4 : (birth_rate - death_rate) / P * 100 = net_growth_rate) :
  P = 1000 := by
  sorry

end NUMINAMATH_CALUDE_population_size_l2612_261215


namespace NUMINAMATH_CALUDE_part1_part2_l2612_261288

-- Define the function f
def f (a x : ℝ) : ℝ := 2 * a * x^2 - (a^2 + 4) * x + 2 * a

-- Part 1
theorem part1 (a : ℝ) : 
  (∀ x, f a x > 0 ↔ -4 < x ∧ x < -1/4) → (a = -8 ∨ a = -1/2) :=
sorry

-- Part 2
theorem part2 (a : ℝ) (h : a > 0) :
  (∀ x, f a x ≤ 0 ↔ 
    ((0 < a ∧ a < 2 → a/2 ≤ x ∧ x ≤ 2/a) ∧
     (a > 2 → 2/a ≤ x ∧ x ≤ a/2) ∧
     (a = 2 → x = 1))) :=
sorry

end NUMINAMATH_CALUDE_part1_part2_l2612_261288


namespace NUMINAMATH_CALUDE_quadratic_real_roots_l2612_261219

/-- The quadratic equation kx^2 - 6x + 9 = 0 has real roots if and only if k ≤ 1 and k ≠ 0 -/
theorem quadratic_real_roots (k : ℝ) : 
  (∃ x : ℝ, k * x^2 - 6 * x + 9 = 0) ↔ (k ≤ 1 ∧ k ≠ 0) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_real_roots_l2612_261219


namespace NUMINAMATH_CALUDE_jonah_aquarium_fish_count_l2612_261241

/-- Calculates the final number of fish in Jonah's aquarium after a series of events. -/
def final_fish_count (initial : ℕ) (added : ℕ) (eaten : ℕ) (returned : ℕ) (exchanged : ℕ) : ℕ :=
  initial + added - eaten - returned + exchanged

/-- Theorem stating that given the initial conditions and series of events, 
    the final number of fish in Jonah's aquarium is 11. -/
theorem jonah_aquarium_fish_count : 
  final_fish_count 14 2 6 2 3 = 11 := by sorry

end NUMINAMATH_CALUDE_jonah_aquarium_fish_count_l2612_261241


namespace NUMINAMATH_CALUDE_exists_divisible_pair_l2612_261270

def u : ℕ → ℤ
  | 0 => 0
  | 1 => 0
  | (n + 2) => u (n + 1) + u n + 1

theorem exists_divisible_pair :
  ∃ n : ℕ, n ≥ 1 ∧ (2011^2012 ∣ u n) ∧ (2011^2012 ∣ u (n + 1)) := by
  sorry

end NUMINAMATH_CALUDE_exists_divisible_pair_l2612_261270


namespace NUMINAMATH_CALUDE_meaningful_fraction_l2612_261262

theorem meaningful_fraction (x : ℝ) : 
  (∃ y : ℝ, y = 1 / (x - 3)) ↔ x ≠ 3 := by sorry

end NUMINAMATH_CALUDE_meaningful_fraction_l2612_261262


namespace NUMINAMATH_CALUDE_cos_270_degrees_l2612_261234

theorem cos_270_degrees (h : ∀ θ, Real.cos (360 - θ) = Real.cos θ) : 
  Real.cos 270 = 0 := by
sorry

end NUMINAMATH_CALUDE_cos_270_degrees_l2612_261234


namespace NUMINAMATH_CALUDE_order_of_x_l2612_261226

theorem order_of_x (x₁ x₂ x₃ x₄ x₅ a₁ a₂ a₃ a₄ a₅ : ℝ)
  (eq1 : x₁ + x₂ + x₃ = a₁)
  (eq2 : x₂ + x₃ + x₄ = a₂)
  (eq3 : x₃ + x₄ + x₅ = a₃)
  (eq4 : x₄ + x₅ + x₁ = a₄)
  (eq5 : x₅ + x₁ + x₂ = a₅)
  (h : a₁ > a₂ ∧ a₂ > a₃ ∧ a₃ > a₄ ∧ a₄ > a₅) :
  x₃ > x₁ ∧ x₁ > x₄ ∧ x₄ > x₂ ∧ x₂ > x₅ :=
by sorry


end NUMINAMATH_CALUDE_order_of_x_l2612_261226


namespace NUMINAMATH_CALUDE_abcd_inequality_l2612_261263

theorem abcd_inequality (a b c d : ℝ) 
  (ha : 0 < a ∧ a < 1) 
  (hb : 0 < b ∧ b < 1) 
  (hc : 0 < c ∧ c < 1) 
  (hd : 0 < d ∧ d < 1) 
  (h_prod : a * b * c * d = (1 - a) * (1 - b) * (1 - c) * (1 - d)) : 
  (a + b + c + d) - (a + c) * (b + d) ≥ 1 := by
  sorry

end NUMINAMATH_CALUDE_abcd_inequality_l2612_261263


namespace NUMINAMATH_CALUDE_binomial_congruence_l2612_261281

theorem binomial_congruence (p : ℕ) (k : ℕ) (hp : Nat.Prime p) (hk : 1 ≤ k ∧ k ≤ p - 1) :
  (Nat.choose (p - 1) k) ≡ ((-1 : ℤ) ^ k) [ZMOD p] := by
  sorry

end NUMINAMATH_CALUDE_binomial_congruence_l2612_261281


namespace NUMINAMATH_CALUDE_acid_mixing_problem_l2612_261214

/-- Represents the acid mixing problem -/
theorem acid_mixing_problem 
  (volume_first : ℝ) 
  (percentage_second : ℝ) 
  (volume_final : ℝ) 
  (percentage_final : ℝ) 
  (h1 : volume_first = 4)
  (h2 : percentage_second = 75)
  (h3 : volume_final = 20)
  (h4 : percentage_final = 72) :
  ∃ (percentage_first : ℝ),
    percentage_first = 60 ∧
    volume_first * (percentage_first / 100) + 
    (volume_final - volume_first) * (percentage_second / 100) = 
    volume_final * (percentage_final / 100) :=
sorry

end NUMINAMATH_CALUDE_acid_mixing_problem_l2612_261214


namespace NUMINAMATH_CALUDE_arithmetic_sequence_common_difference_l2612_261248

/-- An arithmetic sequence with specific conditions -/
def ArithmeticSequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_common_difference
  (a : ℕ → ℝ)
  (h_arith : ArithmeticSequence a)
  (h_sum1 : a 3 + a 6 = 11)
  (h_sum2 : a 5 + a 8 = 39) :
  ∃ d : ℝ, d = 7 ∧ ∀ n : ℕ, a (n + 1) = a n + d :=
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_common_difference_l2612_261248


namespace NUMINAMATH_CALUDE_student_b_score_l2612_261273

-- Define the scoring function
def calculateScore (totalQuestions : ℕ) (correctResponses : ℕ) : ℕ :=
  let incorrectResponses := totalQuestions - correctResponses
  correctResponses - 2 * incorrectResponses

-- Theorem statement
theorem student_b_score :
  calculateScore 100 91 = 73 := by
  sorry

end NUMINAMATH_CALUDE_student_b_score_l2612_261273


namespace NUMINAMATH_CALUDE_tangent_line_equation_l2612_261229

/-- The circle C with equation x^2 + y^2 = 10 -/
def C : Set (ℝ × ℝ) := {p | p.1^2 + p.2^2 = 10}

/-- The point P(1, 3) -/
def P : ℝ × ℝ := (1, 3)

/-- The equation of a line in the form ax + by + c = 0 -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- A line is tangent to a circle if it intersects the circle at exactly one point -/
def IsTangentTo (l : Line) (s : Set (ℝ × ℝ)) : Prop :=
  ∃! p, p ∈ s ∧ l.a * p.1 + l.b * p.2 + l.c = 0

theorem tangent_line_equation :
  IsTangentTo (Line.mk 1 3 (-10)) C ∧ (Line.mk 1 3 (-10)).a * P.1 + (Line.mk 1 3 (-10)).b * P.2 + (Line.mk 1 3 (-10)).c = 0 :=
by sorry

end NUMINAMATH_CALUDE_tangent_line_equation_l2612_261229


namespace NUMINAMATH_CALUDE_trapezoid_solution_l2612_261254

/-- Represents a trapezoid with specific properties -/
structure Trapezoid where
  a : ℝ  -- Length of the shorter parallel side
  h : ℝ  -- Height of the trapezoid
  area : ℝ -- Area of the trapezoid

/-- Properties of the trapezoid -/
def trapezoid_properties (t : Trapezoid) : Prop :=
  t.h = (2 * t.a + 3) / 2 ∧
  t.area = t.a^2 + 3 * t.a + 9 / 4 ∧
  t.area = 2 * t.a^2 - 7.75

/-- Theorem stating the solution to the trapezoid problem -/
theorem trapezoid_solution (t : Trapezoid) (h : trapezoid_properties t) :
  t.a = 5 ∧ t.a + 3 = 8 ∧ t.h = 6.5 := by
  sorry


end NUMINAMATH_CALUDE_trapezoid_solution_l2612_261254


namespace NUMINAMATH_CALUDE_max_groups_is_nine_l2612_261243

/-- Represents the number of singers for each voice type -/
structure ChoirComposition :=
  (sopranos : ℕ)
  (altos : ℕ)
  (tenors : ℕ)
  (basses : ℕ)

/-- Represents the ratio of voice types required in each group -/
structure GroupRatio :=
  (soprano_ratio : ℕ)
  (alto_ratio : ℕ)
  (tenor_ratio : ℕ)
  (bass_ratio : ℕ)

/-- Function to calculate the maximum number of complete groups -/
def maxCompleteGroups (choir : ChoirComposition) (ratio : GroupRatio) : ℕ :=
  min (choir.sopranos / ratio.soprano_ratio)
      (min (choir.altos / ratio.alto_ratio)
           (min (choir.tenors / ratio.tenor_ratio)
                (choir.basses / ratio.bass_ratio)))

/-- Theorem stating that the maximum number of complete groups is 9 -/
theorem max_groups_is_nine :
  let choir := ChoirComposition.mk 10 15 12 18
  let ratio := GroupRatio.mk 1 1 1 2
  maxCompleteGroups choir ratio = 9 :=
by
  sorry

#check max_groups_is_nine

end NUMINAMATH_CALUDE_max_groups_is_nine_l2612_261243


namespace NUMINAMATH_CALUDE_marcus_second_goal_value_l2612_261269

def team_total_points : ℕ := 70
def marcus_3point_goals : ℕ := 5
def marcus_unknown_goals : ℕ := 10
def marcus_percentage : ℚ := 1/2

theorem marcus_second_goal_value :
  ∃ (second_goal_value : ℕ),
    (marcus_3point_goals * 3 + marcus_unknown_goals * second_goal_value : ℚ) = 
      (marcus_percentage * team_total_points) ∧
    second_goal_value = 2 := by
  sorry

end NUMINAMATH_CALUDE_marcus_second_goal_value_l2612_261269


namespace NUMINAMATH_CALUDE_constant_avg_speed_not_imply_uniform_motion_l2612_261228

/-- A snail's motion over a time interval -/
structure SnailMotion where
  /-- The time interval in minutes -/
  interval : ℝ
  /-- The distance traveled in meters -/
  distance : ℝ
  /-- The average speed in meters per minute -/
  avg_speed : ℝ
  /-- Condition: The average speed is constant -/
  constant_avg_speed : avg_speed = distance / interval

/-- Definition of uniform motion -/
def is_uniform_motion (motion : SnailMotion) : Prop :=
  ∀ t : ℝ, 0 ≤ t → t ≤ motion.interval →
    motion.distance * (t / motion.interval) = motion.avg_speed * t

/-- Theorem: Constant average speed does not imply uniform motion -/
theorem constant_avg_speed_not_imply_uniform_motion :
  ∃ (motion : SnailMotion), ¬(is_uniform_motion motion) :=
sorry

end NUMINAMATH_CALUDE_constant_avg_speed_not_imply_uniform_motion_l2612_261228


namespace NUMINAMATH_CALUDE_remainder_theorem_l2612_261276

theorem remainder_theorem (d : ℕ) (r : ℕ) : d > 1 →
  (1059 % d = r ∧ 1417 % d = r ∧ 2312 % d = r) →
  d - r = 15 := by
sorry

end NUMINAMATH_CALUDE_remainder_theorem_l2612_261276
