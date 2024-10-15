import Mathlib

namespace NUMINAMATH_CALUDE_roots_cubic_sum_l4042_404271

theorem roots_cubic_sum (p q : ℝ) : 
  (p^2 - 5*p + 3 = 0) → (q^2 - 5*q + 3 = 0) → (p + q)^3 = 125 := by
  sorry

end NUMINAMATH_CALUDE_roots_cubic_sum_l4042_404271


namespace NUMINAMATH_CALUDE_product_49_sum_14_l4042_404260

theorem product_49_sum_14 (a b c d : ℤ) : 
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d →
  a * b * c * d = 49 →
  a + b + c + d = 14 :=
by sorry

end NUMINAMATH_CALUDE_product_49_sum_14_l4042_404260


namespace NUMINAMATH_CALUDE_desk_length_l4042_404268

/-- Given a rectangular desk with width 9 cm and perimeter 46 cm, prove its length is 14 cm. -/
theorem desk_length (width : ℝ) (perimeter : ℝ) (length : ℝ) : 
  width = 9 → perimeter = 46 → 2 * (length + width) = perimeter → length = 14 := by
  sorry

end NUMINAMATH_CALUDE_desk_length_l4042_404268


namespace NUMINAMATH_CALUDE_river_speed_l4042_404257

/-- Proves that the speed of the river is 1.2 kmph given the conditions of the rowing problem -/
theorem river_speed (still_water_speed : ℝ) (total_time : ℝ) (total_distance : ℝ) :
  still_water_speed = 10 →
  total_time = 1 →
  total_distance = 9.856 →
  ∃ (river_speed : ℝ),
    river_speed = 1.2 ∧
    total_distance = (still_water_speed - river_speed) * (total_time / 2) +
                     (still_water_speed + river_speed) * (total_time / 2) :=
by
  sorry

end NUMINAMATH_CALUDE_river_speed_l4042_404257


namespace NUMINAMATH_CALUDE_garcia_fourth_quarter_shots_l4042_404267

/-- Represents the number of shots taken and made in a basketball game --/
structure GameStats :=
  (shots_taken : ℕ)
  (shots_made : ℕ)

/-- Calculates the shooting accuracy as a rational number --/
def accuracy (stats : GameStats) : ℚ :=
  stats.shots_made / stats.shots_taken

theorem garcia_fourth_quarter_shots 
  (first_two_quarters : GameStats)
  (third_quarter : GameStats)
  (fourth_quarter : GameStats)
  (h1 : first_two_quarters.shots_taken = 20)
  (h2 : first_two_quarters.shots_made = 12)
  (h3 : third_quarter.shots_taken = 10)
  (h4 : accuracy third_quarter = (1/2) * accuracy first_two_quarters)
  (h5 : accuracy fourth_quarter = (4/3) * accuracy third_quarter)
  (h6 : accuracy (GameStats.mk 
    (first_two_quarters.shots_taken + third_quarter.shots_taken + fourth_quarter.shots_taken)
    (first_two_quarters.shots_made + third_quarter.shots_made + fourth_quarter.shots_made)) = 46/100)
  : fourth_quarter.shots_made = 8 := by
  sorry

end NUMINAMATH_CALUDE_garcia_fourth_quarter_shots_l4042_404267


namespace NUMINAMATH_CALUDE_largest_angle_is_120_degrees_l4042_404249

-- Define the sequence a_n
def a (n : ℕ) : ℕ := n^2 - (n-1)^2

-- Define the triangle sides
def side_a : ℕ := a 2
def side_b : ℕ := a 3
def side_c : ℕ := a 4

-- State the theorem
theorem largest_angle_is_120_degrees :
  let angle := Real.arccos ((side_a^2 + side_b^2 - side_c^2) / (2 * side_a * side_b))
  angle = 2 * Real.pi / 3 := by sorry

end NUMINAMATH_CALUDE_largest_angle_is_120_degrees_l4042_404249


namespace NUMINAMATH_CALUDE_sphere_volume_l4042_404253

theorem sphere_volume (r : ℝ) (d V : ℝ) (h : d = (16 / 9 * V) ^ (1 / 3)) (h_r : r = 1 / 3) : V = 1 / 6 := by
  sorry

end NUMINAMATH_CALUDE_sphere_volume_l4042_404253


namespace NUMINAMATH_CALUDE_subtracting_and_dividing_l4042_404251

theorem subtracting_and_dividing (x : ℝ) : x = 32 → (x - 6) / 13 = 2 := by
  sorry

end NUMINAMATH_CALUDE_subtracting_and_dividing_l4042_404251


namespace NUMINAMATH_CALUDE_lcm_1230_924_l4042_404229

theorem lcm_1230_924 : Nat.lcm 1230 924 = 189420 := by
  sorry

end NUMINAMATH_CALUDE_lcm_1230_924_l4042_404229


namespace NUMINAMATH_CALUDE_b_job_fraction_l4042_404250

/-- The fraction of the job that B completes when A and B work together to finish a job -/
theorem b_job_fraction (a_time b_time : ℝ) (a_solo_time : ℝ) : 
  a_time = 6 →
  b_time = 3 →
  a_solo_time = 1 →
  (25 : ℝ) / 54 = 
    ((1 - a_solo_time / a_time) * (1 / b_time) * 
     (1 - a_solo_time / a_time) / ((1 / a_time) + (1 / b_time))) :=
by sorry

end NUMINAMATH_CALUDE_b_job_fraction_l4042_404250


namespace NUMINAMATH_CALUDE_roots_expression_l4042_404222

theorem roots_expression (a b : ℝ) (α β γ δ : ℝ) 
  (hα : α^2 - a*α - 1 = 0)
  (hβ : β^2 - a*β - 1 = 0)
  (hγ : γ^2 - b*γ - 1 = 0)
  (hδ : δ^2 - b*δ - 1 = 0) :
  (α - γ)^2 * (β - γ)^2 * (α + δ)^2 * (β + δ)^2 = (b^2 - a^2)^2 := by
  sorry

end NUMINAMATH_CALUDE_roots_expression_l4042_404222


namespace NUMINAMATH_CALUDE_min_board_size_is_77_l4042_404277

/-- A domino placement on a square board. -/
structure DominoPlacement where
  n : ℕ  -- Size of the square board
  dominoes : ℕ  -- Number of dominoes placed

/-- Checks if the domino placement is valid. -/
def is_valid_placement (p : DominoPlacement) : Prop :=
  p.dominoes * 2 = 2008 ∧  -- Total area covered by dominoes
  (p.n + 1)^2 ≥ p.dominoes * 6  -- Extended board can fit dominoes with shadows

/-- The minimum board size for a valid domino placement. -/
def min_board_size : ℕ := 77

/-- Theorem stating that 77 is the minimum board size for a valid domino placement. -/
theorem min_board_size_is_77 :
  ∀ p : DominoPlacement, is_valid_placement p → p.n ≥ min_board_size :=
by sorry

end NUMINAMATH_CALUDE_min_board_size_is_77_l4042_404277


namespace NUMINAMATH_CALUDE_factor_expression_l4042_404231

theorem factor_expression (a : ℝ) : 58 * a^2 + 174 * a = 58 * a * (a + 3) := by
  sorry

end NUMINAMATH_CALUDE_factor_expression_l4042_404231


namespace NUMINAMATH_CALUDE_expression_simplification_l4042_404226

theorem expression_simplification (m : ℝ) (h : m = Real.sqrt 3 + 1) :
  (1 - 1 / m) / ((m^2 - 2*m + 1) / m) = Real.sqrt 3 / 3 := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l4042_404226


namespace NUMINAMATH_CALUDE_original_number_proof_l4042_404284

theorem original_number_proof (x y : ℝ) : 
  10 * x + 22 * y = 780 → 
  y = 34 → 
  x + y = 37.2 :=
by sorry

end NUMINAMATH_CALUDE_original_number_proof_l4042_404284


namespace NUMINAMATH_CALUDE_smallest_n_remainder_l4042_404234

theorem smallest_n_remainder (n : ℕ) : 
  (n > 0 ∧ ∀ m : ℕ, m > 0 → m < n → (3 * m + 45) % 1060 ≠ 16) →
  (3 * n + 45) % 1060 = 16 →
  (18 * n + 17) % 1920 = 1043 := by
sorry

end NUMINAMATH_CALUDE_smallest_n_remainder_l4042_404234


namespace NUMINAMATH_CALUDE_sum_of_special_numbers_l4042_404216

/-- A function that counts the number of divisors of a natural number -/
def count_divisors (n : ℕ) : ℕ := sorry

/-- A function that checks if a number ends with 5 zeros -/
def ends_with_five_zeros (n : ℕ) : Prop := sorry

/-- The set of natural numbers that end with 5 zeros and have 42 divisors -/
def special_numbers : Set ℕ :=
  {n : ℕ | ends_with_five_zeros n ∧ count_divisors n = 42}

theorem sum_of_special_numbers :
  ∃ (a b : ℕ), a ∈ special_numbers ∧ b ∈ special_numbers ∧ a ≠ b ∧ a + b = 700000 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_special_numbers_l4042_404216


namespace NUMINAMATH_CALUDE_shortest_tangent_is_30_l4042_404246

/-- Two circles in a 2D plane --/
structure TwoCircles where
  c1 : (ℝ × ℝ) → Prop
  c2 : (ℝ × ℝ) → Prop

/-- The given circles from the problem --/
def problem_circles : TwoCircles :=
  { c1 := λ (x, y) => (x - 12)^2 + y^2 = 25,
    c2 := λ (x, y) => (x + 18)^2 + y^2 = 64 }

/-- The length of the shortest line segment tangent to both circles --/
def shortest_tangent_length (circles : TwoCircles) : ℝ :=
  sorry

/-- Theorem stating that the shortest tangent length for the given circles is 30 --/
theorem shortest_tangent_is_30 :
  shortest_tangent_length problem_circles = 30 :=
sorry

end NUMINAMATH_CALUDE_shortest_tangent_is_30_l4042_404246


namespace NUMINAMATH_CALUDE_candy_distribution_theorem_l4042_404247

/-- Represents the number of candy bars of each type --/
structure CandyBars where
  chocolate : ℕ
  caramel : ℕ
  nougat : ℕ

/-- Represents the ratio of candy bars in a bag --/
structure BagRatio where
  chocolate : ℕ
  caramel : ℕ
  nougat : ℕ

/-- Checks if the given ratio is valid for the total number of candy bars --/
def isValidRatio (total : CandyBars) (ratio : BagRatio) (bags : ℕ) : Prop :=
  total.chocolate = ratio.chocolate * bags ∧
  total.caramel = ratio.caramel * bags ∧
  total.nougat = ratio.nougat * bags

/-- The main theorem to be proved --/
theorem candy_distribution_theorem (total : CandyBars) 
  (h1 : total.chocolate = 12) 
  (h2 : total.caramel = 18) 
  (h3 : total.nougat = 15) :
  ∃ (ratio : BagRatio) (bags : ℕ), 
    bags = 5 ∧ 
    ratio.chocolate = 2 ∧ 
    ratio.caramel = 3 ∧ 
    ratio.nougat = 3 ∧
    isValidRatio total ratio bags ∧
    ∀ (other_ratio : BagRatio) (other_bags : ℕ), 
      isValidRatio total other_ratio other_bags → other_bags ≤ bags :=
by
  sorry

end NUMINAMATH_CALUDE_candy_distribution_theorem_l4042_404247


namespace NUMINAMATH_CALUDE_right_triangle_hypotenuse_and_perimeter_l4042_404245

theorem right_triangle_hypotenuse_and_perimeter : 
  ∀ (a b h : ℝ), 
    a = 24 → 
    b = 25 → 
    h^2 = a^2 + b^2 → 
    h = Real.sqrt 1201 ∧ 
    a + b + h = 49 + Real.sqrt 1201 := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_hypotenuse_and_perimeter_l4042_404245


namespace NUMINAMATH_CALUDE_total_ad_cost_is_66000_l4042_404225

/-- Represents an advertisement with its duration and cost per minute -/
structure Advertisement where
  duration : ℕ
  costPerMinute : ℕ

/-- Calculates the total cost of an advertisement -/
def adCost (ad : Advertisement) : ℕ := ad.duration * ad.costPerMinute

/-- The list of advertisements shown during the race -/
def raceAds : List Advertisement := [
  ⟨2, 3500⟩,
  ⟨3, 4500⟩,
  ⟨3, 3000⟩,
  ⟨2, 4000⟩,
  ⟨5, 5500⟩
]

/-- The theorem stating that the total cost of advertisements is $66000 -/
theorem total_ad_cost_is_66000 :
  (raceAds.map adCost).sum = 66000 := by sorry

end NUMINAMATH_CALUDE_total_ad_cost_is_66000_l4042_404225


namespace NUMINAMATH_CALUDE_dinitrogen_monoxide_weight_is_44_02_l4042_404263

/-- The atomic weight of nitrogen in g/mol -/
def nitrogen_weight : ℝ := 14.01

/-- The atomic weight of oxygen in g/mol -/
def oxygen_weight : ℝ := 16.00

/-- The number of nitrogen atoms in dinitrogen monoxide -/
def nitrogen_count : ℕ := 2

/-- The number of oxygen atoms in dinitrogen monoxide -/
def oxygen_count : ℕ := 1

/-- The molecular weight of dinitrogen monoxide (N2O) in g/mol -/
def dinitrogen_monoxide_weight : ℝ :=
  nitrogen_count * nitrogen_weight + oxygen_count * oxygen_weight

/-- Theorem stating that the molecular weight of dinitrogen monoxide is 44.02 g/mol -/
theorem dinitrogen_monoxide_weight_is_44_02 :
  dinitrogen_monoxide_weight = 44.02 := by
  sorry

end NUMINAMATH_CALUDE_dinitrogen_monoxide_weight_is_44_02_l4042_404263


namespace NUMINAMATH_CALUDE_polynomial_divisibility_l4042_404295

theorem polynomial_divisibility (n : ℕ) (h : n > 1) :
  ∃ Q : Polynomial ℂ, x^(4*n+3) + x^(4*n+1) + x^(4*n-2) + x^8 = (x^2 + 1) * Q := by
  sorry

#check polynomial_divisibility

end NUMINAMATH_CALUDE_polynomial_divisibility_l4042_404295


namespace NUMINAMATH_CALUDE_john_sublet_count_l4042_404258

/-- The number of people John sublets his apartment to -/
def num_subletters : ℕ := by sorry

/-- Monthly payment per subletter in dollars -/
def subletter_payment : ℕ := 400

/-- John's monthly rent in dollars -/
def john_rent : ℕ := 900

/-- John's annual profit in dollars -/
def annual_profit : ℕ := 3600

/-- Number of months in a year -/
def months_per_year : ℕ := 12

theorem john_sublet_count : 
  num_subletters * subletter_payment * months_per_year - john_rent * months_per_year = annual_profit → 
  num_subletters = 3 := by sorry

end NUMINAMATH_CALUDE_john_sublet_count_l4042_404258


namespace NUMINAMATH_CALUDE_min_value_reciprocal_sum_l4042_404293

theorem min_value_reciprocal_sum (a b : ℝ) (ha : a > 0) (hb : b > 0) (h_sum : a + 3*b = 2) :
  (1/a + 3/b) ≥ 8 ∧ ∃ (a₀ b₀ : ℝ), a₀ > 0 ∧ b₀ > 0 ∧ a₀ + 3*b₀ = 2 ∧ 1/a₀ + 3/b₀ = 8 :=
sorry

end NUMINAMATH_CALUDE_min_value_reciprocal_sum_l4042_404293


namespace NUMINAMATH_CALUDE_least_number_satisfying_conditions_l4042_404291

def is_divisible_by_11 (n : ℕ) : Prop := n % 11 = 0

def leaves_remainder_2 (n : ℕ) (d : ℕ) : Prop := n % d = 2

def satisfies_conditions (n : ℕ) : Prop :=
  is_divisible_by_11 n ∧ 
  (∀ d : ℕ, 3 ≤ d → d ≤ 7 → leaves_remainder_2 n d)

theorem least_number_satisfying_conditions : 
  satisfies_conditions 3782 ∧ 
  ∀ m : ℕ, m < 3782 → ¬(satisfies_conditions m) :=
sorry

end NUMINAMATH_CALUDE_least_number_satisfying_conditions_l4042_404291


namespace NUMINAMATH_CALUDE_quadratic_increasing_l4042_404282

/-- Given a quadratic function y = (x - 1)^2 + 2, prove that y is increasing when x > 1 -/
theorem quadratic_increasing (x : ℝ) : 
  let y : ℝ → ℝ := λ x ↦ (x - 1)^2 + 2
  x > 1 → ∀ h > 0, y (x + h) > y x :=
by sorry

end NUMINAMATH_CALUDE_quadratic_increasing_l4042_404282


namespace NUMINAMATH_CALUDE_metallic_sheet_dimension_l4042_404219

/-- Given a rectangular metallic sheet with one dimension of 36 meters,
    where a square of 8 meters is cut from each corner to form an open box,
    if the volume of the resulting box is 5760 cubic meters,
    then the length of the other dimension of the metallic sheet is 52 meters. -/
theorem metallic_sheet_dimension (sheet_width : ℝ) (cut_size : ℝ) (box_volume : ℝ) :
  sheet_width = 36 →
  cut_size = 8 →
  box_volume = 5760 →
  (sheet_width - 2 * cut_size) * (52 - 2 * cut_size) * cut_size = box_volume :=
by sorry

end NUMINAMATH_CALUDE_metallic_sheet_dimension_l4042_404219


namespace NUMINAMATH_CALUDE_no_tangent_lines_l4042_404243

/-- Represents a circle in a 2D plane --/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- Represents the configuration of two circles --/
structure TwoCircles where
  circle1 : Circle
  circle2 : Circle
  center_distance : ℝ

/-- Counts the number of tangent lines between two circles --/
def count_tangent_lines (tc : TwoCircles) : ℕ := sorry

/-- The specific configuration given in the problem --/
def problem_config : TwoCircles :=
  { circle1 := { center := (0, 0), radius := 4 }
  , circle2 := { center := (3, 0), radius := 6 }
  , center_distance := 3 }

/-- Theorem stating that the number of tangent lines is zero for the given configuration --/
theorem no_tangent_lines : count_tangent_lines problem_config = 0 := by sorry

end NUMINAMATH_CALUDE_no_tangent_lines_l4042_404243


namespace NUMINAMATH_CALUDE_triangle_max_area_l4042_404261

/-- Given a triangle ABC with area S and sides a, b, c, 
    if 4S = a² - (b - c)² and b + c = 4, 
    then the maximum value of S is 2 -/
theorem triangle_max_area (a b c S : ℝ) : 
  4 * S = a^2 - (b - c)^2 → b + c = 4 → S ≤ 2 ∧ ∃ b c, b + c = 4 ∧ S = 2 := by
  sorry

end NUMINAMATH_CALUDE_triangle_max_area_l4042_404261


namespace NUMINAMATH_CALUDE_cube_sum_theorem_l4042_404299

theorem cube_sum_theorem (x y : ℝ) (h1 : x + y = 5) (h2 : x^2 + y^2 = 14) : 
  x^3 + y^3 = 85/2 := by
sorry

end NUMINAMATH_CALUDE_cube_sum_theorem_l4042_404299


namespace NUMINAMATH_CALUDE_heracles_age_l4042_404275

/-- Proves that Heracles' current age is 10 years, given the conditions stated in the problem. -/
theorem heracles_age : ∃ (H : ℕ), 
  (∀ (A : ℕ), A = H + 7 → A + 3 = 2 * H) → H = 10 := by
  sorry

end NUMINAMATH_CALUDE_heracles_age_l4042_404275


namespace NUMINAMATH_CALUDE_meal_combinations_l4042_404286

def fruit_count : ℕ := 3
def salad_count : ℕ := 4
def dessert_count : ℕ := 5

def choose (n k : ℕ) : ℕ := (Nat.factorial n) / (Nat.factorial k * Nat.factorial (n - k))

theorem meal_combinations :
  fruit_count * choose salad_count 2 * dessert_count = 90 := by
  sorry

end NUMINAMATH_CALUDE_meal_combinations_l4042_404286


namespace NUMINAMATH_CALUDE_plain_lemonade_sales_l4042_404259

/-- The number of glasses of plain lemonade sold -/
def plain_lemonade_glasses : ℕ := 36

/-- The price of plain lemonade in dollars -/
def plain_lemonade_price : ℚ := 3/4

/-- The total revenue from strawberry lemonade in dollars -/
def strawberry_revenue : ℕ := 16

/-- The revenue difference between plain and strawberry lemonade in dollars -/
def revenue_difference : ℕ := 11

theorem plain_lemonade_sales :
  plain_lemonade_glasses * plain_lemonade_price = 
    (strawberry_revenue + revenue_difference : ℚ) := by sorry

end NUMINAMATH_CALUDE_plain_lemonade_sales_l4042_404259


namespace NUMINAMATH_CALUDE_unique_k_solution_l4042_404264

/-- The function f(x) = x^2 - 2x + 1 -/
def f (x : ℝ) : ℝ := x^2 - 2*x + 1

/-- The theorem stating that k = 2 is the only solution -/
theorem unique_k_solution :
  ∃! k : ℝ, ∀ x : ℝ, f (x + k) = x^2 + 2*x + 1 ∧ k = 2 := by
  sorry

end NUMINAMATH_CALUDE_unique_k_solution_l4042_404264


namespace NUMINAMATH_CALUDE_no_solution_to_system_l4042_404209

theorem no_solution_to_system :
  ¬∃ (x y z : ℝ), (x^2 - 2*y + 2 = 0) ∧ (y^2 - 4*z + 3 = 0) ∧ (z^2 + 4*x + 4 = 0) := by
sorry

end NUMINAMATH_CALUDE_no_solution_to_system_l4042_404209


namespace NUMINAMATH_CALUDE_leahs_coins_value_l4042_404202

theorem leahs_coins_value :
  ∀ (p n : ℕ),
  p + n = 18 →
  n + 2 = p →
  5 * n + p = 50 :=
by
  sorry

end NUMINAMATH_CALUDE_leahs_coins_value_l4042_404202


namespace NUMINAMATH_CALUDE_S_is_finite_l4042_404228

/-- Euler's totient function -/
def phi (n : ℕ) : ℕ := sorry

/-- Number of positive divisors function -/
def tau (n : ℕ) : ℕ := sorry

/-- The set of positive integers satisfying the inequality -/
def S : Set ℕ := {n : ℕ | n > 0 ∧ phi n * tau n ≥ Real.sqrt (n^3 / 3)}

/-- Theorem stating that S is finite -/
theorem S_is_finite : Set.Finite S := by sorry

end NUMINAMATH_CALUDE_S_is_finite_l4042_404228


namespace NUMINAMATH_CALUDE_quadratic_inequality_l4042_404297

theorem quadratic_inequality (x : ℝ) : x^2 - 6*x + 5 > 0 ↔ x < 1 ∨ x > 5 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_l4042_404297


namespace NUMINAMATH_CALUDE_percentage_calculation_l4042_404206

theorem percentage_calculation (initial_amount : ℝ) : 
  initial_amount = 1200 →
  (((initial_amount * 0.60) * 0.30) * 2) / 3 = 144 := by
sorry

end NUMINAMATH_CALUDE_percentage_calculation_l4042_404206


namespace NUMINAMATH_CALUDE_set_operations_l4042_404269

def A : Set ℝ := {x | -4 ≤ x ∧ x ≤ -2}
def B : Set ℝ := {x | x + 3 ≥ 0}
def U : Set ℝ := {x | x ≤ -1}

theorem set_operations :
  (A ∩ B = {x | -3 ≤ x ∧ x ≤ -2}) ∧
  (A ∪ B = {x | x ≥ -4}) ∧
  (U \ (A ∩ B) = {x | x < -3 ∨ (-2 < x ∧ x ≤ -1)}) :=
by sorry

end NUMINAMATH_CALUDE_set_operations_l4042_404269


namespace NUMINAMATH_CALUDE_polynomial_equality_l4042_404238

/-- Given a polynomial q(x) satisfying the equation
    q(x) + (2x^6 + 4x^4 + 5x^3 + 11x) = (10x^4 + 30x^3 + 40x^2 + 8x + 3),
    prove that q(x) = -2x^6 + 6x^4 + 25x^3 + 40x^2 - 3x + 3 -/
theorem polynomial_equality (q : ℝ → ℝ) :
  (∀ x, q x + (2 * x^6 + 4 * x^4 + 5 * x^3 + 11 * x) = 
       (10 * x^4 + 30 * x^3 + 40 * x^2 + 8 * x + 3)) →
  (∀ x, q x = -2 * x^6 + 6 * x^4 + 25 * x^3 + 40 * x^2 - 3 * x + 3) := by
sorry

end NUMINAMATH_CALUDE_polynomial_equality_l4042_404238


namespace NUMINAMATH_CALUDE_prob_eight_odd_rolls_l4042_404240

/-- A fair twelve-sided die -/
def TwelveSidedDie : Finset ℕ := Finset.range 12

/-- The set of odd numbers on a twelve-sided die -/
def OddNumbers : Finset ℕ := TwelveSidedDie.filter (λ x => x % 2 = 1)

/-- The probability of rolling an odd number with a twelve-sided die -/
def ProbOdd : ℚ := (OddNumbers.card : ℚ) / (TwelveSidedDie.card : ℚ)

/-- The number of consecutive rolls -/
def NumRolls : ℕ := 8

theorem prob_eight_odd_rolls :
  ProbOdd ^ NumRolls = 1 / 256 := by sorry

end NUMINAMATH_CALUDE_prob_eight_odd_rolls_l4042_404240


namespace NUMINAMATH_CALUDE_cos_2α_value_l4042_404239

theorem cos_2α_value (α : Real) (h1 : 0 < α ∧ α < π/2) (h2 : Real.sin (α - π/4) = 1/4) : 
  Real.cos (2*α) = -(Real.sqrt 15)/8 := by
sorry

end NUMINAMATH_CALUDE_cos_2α_value_l4042_404239


namespace NUMINAMATH_CALUDE_smallest_product_is_zero_l4042_404223

def S : Set ℤ := {-8, -4, 0, 2, 5}

theorem smallest_product_is_zero :
  ∃ (a b : ℤ), a ∈ S ∧ b ∈ S ∧ a * b = 0 ∧ 
  ∀ (x y : ℤ), x ∈ S → y ∈ S → a * b ≤ x * y :=
by sorry

end NUMINAMATH_CALUDE_smallest_product_is_zero_l4042_404223


namespace NUMINAMATH_CALUDE_composite_function_value_l4042_404274

-- Define the functions f and g
def f (c : ℝ) (x : ℝ) : ℝ := 5 * x + c
def g (c : ℝ) (x : ℝ) : ℝ := c * x + 3

-- State the theorem
theorem composite_function_value (c d : ℝ) :
  (∀ x, f c (g c x) = 15 * x + d) → d = 18 := by
  sorry

end NUMINAMATH_CALUDE_composite_function_value_l4042_404274


namespace NUMINAMATH_CALUDE_difference_of_squares_l4042_404270

theorem difference_of_squares (x y : ℝ) (h1 : x + y = 20) (h2 : x - y = 10) :
  x^2 - y^2 = 200 := by
  sorry

end NUMINAMATH_CALUDE_difference_of_squares_l4042_404270


namespace NUMINAMATH_CALUDE_park_area_l4042_404227

/-- The area of a rectangular park with sides in ratio 3:2 and fencing cost $150 at 60 ps per meter --/
theorem park_area (length width : ℝ) (area perimeter cost_per_meter total_cost : ℝ) : 
  length / width = 3 / 2 →
  area = length * width →
  perimeter = 2 * (length + width) →
  cost_per_meter = 0.60 →
  total_cost = 150 →
  total_cost = perimeter * cost_per_meter →
  area = 3750 :=
by sorry

end NUMINAMATH_CALUDE_park_area_l4042_404227


namespace NUMINAMATH_CALUDE_hyperbola_iff_m_negative_l4042_404248

/-- A conic section represented by the equation x^2 + my^2 = m -/
structure ConicSection (m : ℝ) where
  x : ℝ
  y : ℝ
  eq : x^2 + m * y^2 = m

/-- Predicate to determine if a conic section is a hyperbola -/
def IsHyperbola (m : ℝ) : Prop :=
  ∃ (a b : ℝ), a ≠ 0 ∧ b ≠ 0 ∧ ∀ (x y : ℝ), x^2 / a^2 - y^2 / b^2 = 1 ↔ x^2 + m * y^2 = m

/-- Theorem stating that the equation x^2 + my^2 = m represents a hyperbola if and only if m < 0 -/
theorem hyperbola_iff_m_negative (m : ℝ) : IsHyperbola m ↔ m < 0 := by
  sorry

end NUMINAMATH_CALUDE_hyperbola_iff_m_negative_l4042_404248


namespace NUMINAMATH_CALUDE_parking_tickets_l4042_404287

theorem parking_tickets (total : ℕ) (alan : ℕ) (marcy : ℕ) 
  (h1 : total = 150)
  (h2 : alan = 26)
  (h3 : marcy = 5 * alan)
  (h4 : total = alan + marcy) :
  total - marcy = 104 := by
  sorry

end NUMINAMATH_CALUDE_parking_tickets_l4042_404287


namespace NUMINAMATH_CALUDE_system_one_solution_system_two_solution_l4042_404254

-- System (1)
theorem system_one_solution (x y : ℚ) : 
  (3 * x - 6 * y = 4 ∧ x + 5 * y = 6) ↔ (x = 8/3 ∧ y = 2/3) := by sorry

-- System (2)
theorem system_two_solution (x y : ℚ) :
  (x/4 + y/3 = 3 ∧ 3*(x-4) - 2*(y-1) = -1) ↔ (x = 6 ∧ y = 9/2) := by sorry

end NUMINAMATH_CALUDE_system_one_solution_system_two_solution_l4042_404254


namespace NUMINAMATH_CALUDE_madeline_water_intake_l4042_404292

structure WaterBottle where
  capacity : ℕ

structure Activity where
  name : String
  goal : ℕ
  bottle : WaterBottle
  refills : ℕ

def total_intake (activities : List Activity) : ℕ :=
  activities.foldl (λ acc activity => acc + activity.bottle.capacity * (activity.refills + 1)) 0

def madeline_water_plan : List Activity :=
  [{ name := "Morning yoga", goal := 15, bottle := { capacity := 8 }, refills := 1 },
   { name := "Work", goal := 35, bottle := { capacity := 12 }, refills := 2 },
   { name := "Afternoon jog", goal := 20, bottle := { capacity := 16 }, refills := 1 },
   { name := "Evening leisure", goal := 30, bottle := { capacity := 8 }, refills := 1 },
   { name := "Evening leisure", goal := 30, bottle := { capacity := 16 }, refills := 1 }]

theorem madeline_water_intake :
  total_intake madeline_water_plan = 132 := by
  sorry

end NUMINAMATH_CALUDE_madeline_water_intake_l4042_404292


namespace NUMINAMATH_CALUDE_l_shape_area_l4042_404200

theorem l_shape_area (a : ℝ) (h1 : a > 0) (h2 : 5 * a^2 = 4 * ((a + 3)^2 - a^2)) :
  (a + 3)^2 - a^2 = 45 := by
  sorry

end NUMINAMATH_CALUDE_l_shape_area_l4042_404200


namespace NUMINAMATH_CALUDE_orange_juice_fraction_is_467_2400_l4042_404211

/-- Represents a pitcher with a specific volume and content --/
structure Pitcher :=
  (volume : ℚ)
  (content : ℚ)

/-- Calculates the fraction of orange juice in the final mixture --/
def orangeJuiceFraction (p1 p2 p3 : Pitcher) : ℚ :=
  let totalVolume := p1.volume + p2.volume + p3.volume
  let orangeJuiceVolume := p1.content + p2.content
  orangeJuiceVolume / totalVolume

/-- Theorem stating the fraction of orange juice in the final mixture --/
theorem orange_juice_fraction_is_467_2400 :
  let p1 : Pitcher := ⟨800, 800 * (1/4)⟩
  let p2 : Pitcher := ⟨800, 800 * (1/3)⟩
  let p3 : Pitcher := ⟨800, 0⟩  -- Third pitcher doesn't contribute to orange juice
  orangeJuiceFraction p1 p2 p3 = 467 / 2400 := by
  sorry

#eval orangeJuiceFraction ⟨800, 800 * (1/4)⟩ ⟨800, 800 * (1/3)⟩ ⟨800, 0⟩

end NUMINAMATH_CALUDE_orange_juice_fraction_is_467_2400_l4042_404211


namespace NUMINAMATH_CALUDE_max_cone_bound_for_f_l4042_404221

/-- A function f: ℝ → ℝ is cone-bottomed if there exists a constant M > 0
    such that |f(x)| ≥ M|x| for all x ∈ ℝ -/
def ConeBounded (f : ℝ → ℝ) : Prop :=
  ∃ M : ℝ, M > 0 ∧ ∀ x : ℝ, |f x| ≥ M * |x|

/-- The function f(x) = x^2 + 1 -/
def f (x : ℝ) : ℝ := x^2 + 1

theorem max_cone_bound_for_f :
  (ConeBounded f) ∧ (∀ M : ℝ, (∀ x : ℝ, |f x| ≥ M * |x|) → M ≤ 2) ∧
  (∃ x : ℝ, |f x| = 2 * |x|) := by
  sorry


end NUMINAMATH_CALUDE_max_cone_bound_for_f_l4042_404221


namespace NUMINAMATH_CALUDE_lowest_cost_l4042_404252

variable (x y z a b c : ℝ)

/-- The painting areas of the three rooms satisfy x < y < z -/
axiom area_order : x < y ∧ y < z

/-- The painting costs of the three colors satisfy a < b < c -/
axiom cost_order : a < b ∧ b < c

/-- The total cost function for a painting scheme -/
def total_cost (p q r : ℝ) : ℝ := p*x + q*y + r*z

/-- The theorem stating that az + by + cx is the lowest total cost -/
theorem lowest_cost : 
  ∀ (p q r : ℝ), (p = a ∧ q = b ∧ r = c) ∨ (p = a ∧ q = c ∧ r = b) ∨ 
                  (p = b ∧ q = a ∧ r = c) ∨ (p = b ∧ q = c ∧ r = a) ∨ 
                  (p = c ∧ q = a ∧ r = b) ∨ (p = c ∧ q = b ∧ r = a) →
                  total_cost a b c ≤ total_cost p q r :=
by sorry

end NUMINAMATH_CALUDE_lowest_cost_l4042_404252


namespace NUMINAMATH_CALUDE_range_of_m_l4042_404215

-- Define the sets A and B
def A : Set ℝ := {x | x ≤ 1}
def B (m : ℝ) : Set ℝ := {x | x ≤ m}

-- State the theorem
theorem range_of_m (m : ℝ) : B m ⊆ A → m ≤ 1 := by
  sorry

end NUMINAMATH_CALUDE_range_of_m_l4042_404215


namespace NUMINAMATH_CALUDE_total_pictures_l4042_404241

/-- The number of pictures drawn by each person and their total -/
def picture_problem (randy peter quincy susan thomas : ℕ) : Prop :=
  randy = 5 ∧
  peter = randy + 3 ∧
  quincy = peter + 20 ∧
  susan = 2 * quincy - 7 ∧
  thomas = randy ^ 3 ∧
  randy + peter + quincy + susan + thomas = 215

/-- Proof that the total number of pictures drawn is 215 -/
theorem total_pictures : ∃ randy peter quincy susan thomas : ℕ, 
  picture_problem randy peter quincy susan thomas := by
  sorry

end NUMINAMATH_CALUDE_total_pictures_l4042_404241


namespace NUMINAMATH_CALUDE_ice_cream_consumption_l4042_404220

theorem ice_cream_consumption (friday_consumption : Real) (total_consumption : Real)
  (h1 : friday_consumption = 3.25)
  (h2 : total_consumption = 3.5) :
  total_consumption - friday_consumption = 0.25 := by
sorry

end NUMINAMATH_CALUDE_ice_cream_consumption_l4042_404220


namespace NUMINAMATH_CALUDE_solve_equation_l4042_404235

theorem solve_equation (n m x : ℚ) 
  (h1 : (7 : ℚ) / 8 = n / 96)
  (h2 : (7 : ℚ) / 8 = (m + n) / 112)
  (h3 : (7 : ℚ) / 8 = (x - m) / 144) : 
  x = 140 := by sorry

end NUMINAMATH_CALUDE_solve_equation_l4042_404235


namespace NUMINAMATH_CALUDE_power_of_two_equation_l4042_404203

theorem power_of_two_equation (m : ℕ) : 
  2^2002 - 2^2000 - 2^1999 + 2^1998 = m * 2^1998 → m = 11 := by
  sorry

end NUMINAMATH_CALUDE_power_of_two_equation_l4042_404203


namespace NUMINAMATH_CALUDE_sqrt_sum_powers_of_five_l4042_404294

theorem sqrt_sum_powers_of_five : 
  Real.sqrt (5^3 + 5^4 + 5^5) = 5 * Real.sqrt 155 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_sum_powers_of_five_l4042_404294


namespace NUMINAMATH_CALUDE_shelbys_rainy_drive_time_l4042_404208

theorem shelbys_rainy_drive_time 
  (speed_no_rain : ℝ) 
  (speed_rain : ℝ) 
  (total_distance : ℝ) 
  (total_time : ℝ) 
  (h1 : speed_no_rain = 30) 
  (h2 : speed_rain = 20) 
  (h3 : total_distance = 24) 
  (h4 : total_time = 50) : 
  ∃ (rain_time : ℝ), 
    rain_time = 3 ∧ 
    (speed_no_rain / 60) * (total_time - rain_time) + (speed_rain / 60) * rain_time = total_distance :=
by
  sorry


end NUMINAMATH_CALUDE_shelbys_rainy_drive_time_l4042_404208


namespace NUMINAMATH_CALUDE_notebook_purchase_solution_l4042_404207

/-- Notebook types -/
inductive NotebookType
| A
| B
| C

/-- Represents the notebook purchase problem -/
structure NotebookPurchase where
  totalNotebooks : ℕ
  priceA : ℕ
  priceB : ℕ
  priceC : ℕ
  totalCostI : ℕ
  totalCostII : ℕ

/-- Represents the solution for part I -/
structure SolutionI where
  numA : ℕ
  numB : ℕ

/-- Represents the solution for part II -/
structure SolutionII where
  numA : ℕ

/-- The given notebook purchase problem -/
def problem : NotebookPurchase :=
  { totalNotebooks := 30
  , priceA := 11
  , priceB := 9
  , priceC := 6
  , totalCostI := 288
  , totalCostII := 188
  }

/-- Checks if the solution for part I is correct -/
def checkSolutionI (p : NotebookPurchase) (s : SolutionI) : Prop :=
  s.numA + s.numB = p.totalNotebooks ∧
  s.numA * p.priceA + s.numB * p.priceB = p.totalCostI

/-- Checks if the solution for part II is correct -/
def checkSolutionII (p : NotebookPurchase) (s : SolutionII) : Prop :=
  ∃ (numB numC : ℕ), 
    s.numA + numB + numC = p.totalNotebooks ∧
    s.numA * p.priceA + numB * p.priceB + numC * p.priceC = p.totalCostII

/-- The main theorem to prove -/
theorem notebook_purchase_solution :
  checkSolutionI problem { numA := 9, numB := 21 } ∧
  checkSolutionII problem { numA := 1 } :=
sorry


end NUMINAMATH_CALUDE_notebook_purchase_solution_l4042_404207


namespace NUMINAMATH_CALUDE_ball_ratio_proof_l4042_404201

/-- Proves that the ratio of blue balls to red balls is 16:5 given the initial conditions --/
theorem ball_ratio_proof (initial_red : ℕ) (lost_red : ℕ) (yellow : ℕ) (total : ℕ) :
  initial_red = 16 →
  lost_red = 6 →
  yellow = 32 →
  total = 74 →
  ∃ (blue : ℕ), blue * 5 = (initial_red - lost_red) * 16 ∧ 
                blue + (initial_red - lost_red) + yellow = total :=
by sorry

end NUMINAMATH_CALUDE_ball_ratio_proof_l4042_404201


namespace NUMINAMATH_CALUDE_g_of_negative_two_l4042_404298

-- Define the function g
def g (x : ℝ) : ℝ := 5 * x + 2

-- Theorem statement
theorem g_of_negative_two : g (-2) = -8 := by
  sorry

end NUMINAMATH_CALUDE_g_of_negative_two_l4042_404298


namespace NUMINAMATH_CALUDE_train_length_l4042_404289

/-- The length of a train given its speed and time to cross a pole -/
theorem train_length (speed : ℝ) (time : ℝ) : 
  speed = 50 → time = 9 → ∃ length : ℝ, 
  (length ≥ 124.5 ∧ length ≤ 125.5) ∧ length = speed * 1000 / 3600 * time :=
sorry

end NUMINAMATH_CALUDE_train_length_l4042_404289


namespace NUMINAMATH_CALUDE_path_cost_calculation_l4042_404205

/-- Calculates the total cost of constructing a path around a rectangular field -/
def path_construction_cost (field_length field_width path_width cost_per_sqm : ℝ) : ℝ :=
  let outer_length := field_length + 2 * path_width
  let outer_width := field_width + 2 * path_width
  let total_area := outer_length * outer_width
  let field_area := field_length * field_width
  let path_area := total_area - field_area
  path_area * cost_per_sqm

/-- Theorem stating the total cost of constructing the path -/
theorem path_cost_calculation :
  path_construction_cost 75 55 2.5 10 = 6750 := by
  sorry

end NUMINAMATH_CALUDE_path_cost_calculation_l4042_404205


namespace NUMINAMATH_CALUDE_sum_of_cubes_l4042_404262

theorem sum_of_cubes (x y : ℝ) (h1 : x + y = 12) (h2 : x * y = 20) :
  x^3 + y^3 = 1008 := by sorry

end NUMINAMATH_CALUDE_sum_of_cubes_l4042_404262


namespace NUMINAMATH_CALUDE_largest_rectangle_area_l4042_404288

/-- Represents a rectangular area within a square grid -/
structure Rectangle where
  width : Nat
  height : Nat

/-- Represents a square grid -/
structure Grid where
  size : Nat
  center : Nat × Nat

/-- Checks if a rectangle contains the center of a grid -/
def containsCenter (r : Rectangle) (g : Grid) : Prop :=
  ∃ (x y : Nat), x ≥ 1 ∧ x ≤ r.width ∧ y ≥ 1 ∧ y ≤ r.height ∧ 
    (x + (g.size - r.width) / 2, y + (g.size - r.height) / 2) = g.center

/-- Checks if a rectangle fits within a grid -/
def fitsInGrid (r : Rectangle) (g : Grid) : Prop :=
  r.width ≤ g.size ∧ r.height ≤ g.size

/-- The area of a rectangle -/
def area (r : Rectangle) : Nat :=
  r.width * r.height

/-- The theorem to be proved -/
theorem largest_rectangle_area (g : Grid) (r : Rectangle) : 
  g.size = 11 → 
  g.center = (6, 6) → 
  fitsInGrid r g → 
  ¬containsCenter r g → 
  area r ≤ 55 := by
  sorry

end NUMINAMATH_CALUDE_largest_rectangle_area_l4042_404288


namespace NUMINAMATH_CALUDE_square_perimeter_sum_l4042_404214

theorem square_perimeter_sum (a b : ℝ) (h1 : a + b = 130) (h2 : a - b = 42) :
  4 * (Real.sqrt a.toNNReal + Real.sqrt b.toNNReal) = 4 * (Real.sqrt 86 + 2 * Real.sqrt 11) :=
by sorry

end NUMINAMATH_CALUDE_square_perimeter_sum_l4042_404214


namespace NUMINAMATH_CALUDE_circle_radii_sum_l4042_404276

theorem circle_radii_sum (r₁ r₂ : ℝ) : 
  (∃ (a : ℝ), (2 - a)^2 + (5 - a)^2 = a^2 ∧ 
               r₁ = a ∧ 
               (∃ (b : ℝ), b^2 - 14*b + 29 = 0 ∧ r₂ = b)) →
  r₁ + r₂ = 14 := by
sorry


end NUMINAMATH_CALUDE_circle_radii_sum_l4042_404276


namespace NUMINAMATH_CALUDE_sin_pi_minus_alpha_l4042_404255

theorem sin_pi_minus_alpha (α : Real) (h1 : 0 < α ∧ α < π / 2) 
  (h2 : Real.sin (α + π / 3) = 3 / 5) : 
  Real.sin (π - α) = (3 + 4 * Real.sqrt 3) / 10 := by
  sorry

end NUMINAMATH_CALUDE_sin_pi_minus_alpha_l4042_404255


namespace NUMINAMATH_CALUDE_hawks_score_l4042_404266

/-- Given the total score and winning margin in a basketball game, 
    calculate the score of the losing team. -/
theorem hawks_score (total_score winning_margin : ℕ) : 
  total_score = 58 → winning_margin = 12 → 
  (total_score - winning_margin) / 2 = 23 := by
  sorry

#check hawks_score

end NUMINAMATH_CALUDE_hawks_score_l4042_404266


namespace NUMINAMATH_CALUDE_mickey_horses_per_week_l4042_404204

/-- The number of horses Minnie mounts per day -/
def minnie_horses_per_day : ℕ := 7 + 3

/-- The number of horses Mickey mounts per day -/
def mickey_horses_per_day : ℕ := 2 * minnie_horses_per_day - 6

/-- The number of days in a week -/
def days_in_week : ℕ := 7

/-- Theorem: Mickey mounts 98 horses per week -/
theorem mickey_horses_per_week : mickey_horses_per_day * days_in_week = 98 := by
  sorry

end NUMINAMATH_CALUDE_mickey_horses_per_week_l4042_404204


namespace NUMINAMATH_CALUDE_average_of_remaining_numbers_l4042_404285

theorem average_of_remaining_numbers
  (total : ℝ)
  (group1 : ℝ)
  (group2 : ℝ)
  (h1 : total = 6 * 6.40)
  (h2 : group1 = 2 * 6.2)
  (h3 : group2 = 2 * 6.1) :
  (total - group1 - group2) / 2 = 6.9 := by
  sorry

end NUMINAMATH_CALUDE_average_of_remaining_numbers_l4042_404285


namespace NUMINAMATH_CALUDE_absolute_value_plus_reciprocal_zero_l4042_404279

theorem absolute_value_plus_reciprocal_zero (x : ℝ) :
  x ≠ 0 ∧ |x| + 1/x = 0 → x = -1 :=
by
  sorry

end NUMINAMATH_CALUDE_absolute_value_plus_reciprocal_zero_l4042_404279


namespace NUMINAMATH_CALUDE_sine_shift_overlap_l4042_404283

/-- The smallest positive value of ω that makes the sine function overlap with its shifted version -/
theorem sine_shift_overlap : ∃ ω : ℝ, ω > 0 ∧ 
  (∀ x : ℝ, Real.sin (ω * x + π / 3) = Real.sin (ω * (x - π / 3) + π / 3)) ∧
  (∀ ω' : ℝ, ω' > 0 → 
    (∀ x : ℝ, Real.sin (ω' * x + π / 3) = Real.sin (ω' * (x - π / 3) + π / 3)) → 
    ω ≤ ω') ∧
  ω = 2 * π := by
sorry

end NUMINAMATH_CALUDE_sine_shift_overlap_l4042_404283


namespace NUMINAMATH_CALUDE_solve_exponential_equation_l4042_404296

theorem solve_exponential_equation : 
  ∃ x : ℝ, (125 : ℝ)^4 = 5^x ∧ x = 12 := by
  sorry

end NUMINAMATH_CALUDE_solve_exponential_equation_l4042_404296


namespace NUMINAMATH_CALUDE_bus_travel_fraction_l4042_404217

theorem bus_travel_fraction (total_distance : ℝ) 
  (h1 : total_distance = 105.00000000000003)
  (h2 : (1 : ℝ) / 5 * total_distance + 14 + (2 : ℝ) / 3 * total_distance = total_distance) :
  (total_distance - ((1 : ℝ) / 5 * total_distance + 14)) / total_distance = 2 / 3 := by
  sorry

end NUMINAMATH_CALUDE_bus_travel_fraction_l4042_404217


namespace NUMINAMATH_CALUDE_draw_probability_modified_deck_l4042_404237

/-- A modified deck of cards -/
structure ModifiedDeck :=
  (total_cards : ℕ)
  (ranks : ℕ)
  (suits : ℕ)
  (cards_per_suit : ℕ)
  (cards_per_rank : ℕ)

/-- The probability of drawing a specific sequence of cards -/
def draw_probability (deck : ModifiedDeck) (heart_cards : ℕ) (spade_cards : ℕ) (king_cards : ℕ) : ℚ :=
  (heart_cards * spade_cards * king_cards : ℚ) / 
  (deck.total_cards * (deck.total_cards - 1) * (deck.total_cards - 2))

/-- The main theorem -/
theorem draw_probability_modified_deck :
  let deck := ModifiedDeck.mk 104 26 4 26 8
  draw_probability deck 26 26 8 = 169 / 34102 := by sorry

end NUMINAMATH_CALUDE_draw_probability_modified_deck_l4042_404237


namespace NUMINAMATH_CALUDE_theo_cookie_days_l4042_404273

/-- The number of days Theo eats cookies each month -/
def days_per_month (cookies_per_time cookies_per_day total_cookies months : ℕ) : ℕ :=
  (total_cookies / months) / (cookies_per_time * cookies_per_day)

/-- Theorem stating that Theo eats cookies for 20 days each month -/
theorem theo_cookie_days : days_per_month 13 3 2340 3 = 20 := by
  sorry

end NUMINAMATH_CALUDE_theo_cookie_days_l4042_404273


namespace NUMINAMATH_CALUDE_white_ring_weight_l4042_404212

/-- Given the weights of three plastic rings (orange, purple, and white) and their total weight,
    this theorem proves that the weight of the white ring is 0.42 ounces. -/
theorem white_ring_weight
  (orange_weight : ℝ)
  (purple_weight : ℝ)
  (total_weight : ℝ)
  (h1 : orange_weight = 0.08)
  (h2 : purple_weight = 0.33)
  (h3 : total_weight = 0.83)
  : total_weight - (orange_weight + purple_weight) = 0.42 := by
  sorry

#eval (0.83 : ℝ) - ((0.08 : ℝ) + (0.33 : ℝ))

end NUMINAMATH_CALUDE_white_ring_weight_l4042_404212


namespace NUMINAMATH_CALUDE_two_white_balls_probability_l4042_404230

/-- The probability of drawing two white balls sequentially without replacement
    from a box containing 7 white balls and 8 black balls is 1/5. -/
theorem two_white_balls_probability
  (white_balls : Nat)
  (black_balls : Nat)
  (h_white : white_balls = 7)
  (h_black : black_balls = 8) :
  (white_balls / (white_balls + black_balls)) *
  ((white_balls - 1) / (white_balls + black_balls - 1)) =
  1 / 5 := by
  sorry

end NUMINAMATH_CALUDE_two_white_balls_probability_l4042_404230


namespace NUMINAMATH_CALUDE_expression_simplification_l4042_404210

theorem expression_simplification (x : ℝ) : (2*x + 1)*(2*x - 1) - x*(4*x - 1) = x - 1 := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l4042_404210


namespace NUMINAMATH_CALUDE_number_of_observations_l4042_404224

theorem number_of_observations (original_mean corrected_mean wrong_value correct_value : ℝ) 
  (h1 : original_mean = 36)
  (h2 : wrong_value = 23)
  (h3 : correct_value = 48)
  (h4 : corrected_mean = 36.5) :
  ∃ n : ℕ, (n : ℝ) * original_mean + (correct_value - wrong_value) = n * corrected_mean ∧ n = 50 := by
  sorry

end NUMINAMATH_CALUDE_number_of_observations_l4042_404224


namespace NUMINAMATH_CALUDE_binomial_12_4_l4042_404290

theorem binomial_12_4 : Nat.choose 12 4 = 495 := by
  sorry

end NUMINAMATH_CALUDE_binomial_12_4_l4042_404290


namespace NUMINAMATH_CALUDE_square_difference_1002_1000_l4042_404213

theorem square_difference_1002_1000 : 1002^2 - 1000^2 = 4004 := by
  sorry

end NUMINAMATH_CALUDE_square_difference_1002_1000_l4042_404213


namespace NUMINAMATH_CALUDE_bullseye_value_l4042_404281

/-- 
Given a dart game with the following conditions:
- Three darts are thrown
- One dart is a bullseye worth B points
- One dart completely misses (0 points)
- One dart is worth half the bullseye points
- The total score is 75 points

Prove that the bullseye is worth 50 points
-/
theorem bullseye_value (B : ℝ) 
  (total_score : B + 0 + B/2 = 75) : 
  B = 50 := by
  sorry

end NUMINAMATH_CALUDE_bullseye_value_l4042_404281


namespace NUMINAMATH_CALUDE_subtraction_base_8_to_10_l4042_404233

def base_8_to_10 (n : ℕ) : ℕ :=
  let digits := n.digits 8
  (List.range digits.length).foldl (λ acc i => acc + digits[i]! * (8 ^ i)) 0

theorem subtraction_base_8_to_10 :
  base_8_to_10 (4725 - 2367) = 1246 :=
sorry

end NUMINAMATH_CALUDE_subtraction_base_8_to_10_l4042_404233


namespace NUMINAMATH_CALUDE_hypotenuse_of_special_triangle_l4042_404265

-- Define a right triangle
structure RightTriangle where
  leg1 : ℝ
  leg2 : ℝ
  hypotenuse : ℝ
  angle_opposite_leg1 : ℝ
  is_right_triangle : leg1^2 + leg2^2 = hypotenuse^2

-- Theorem statement
theorem hypotenuse_of_special_triangle 
  (triangle : RightTriangle)
  (h1 : triangle.leg1 = 15)
  (h2 : triangle.angle_opposite_leg1 = 30 * π / 180) :
  triangle.hypotenuse = 30 :=
sorry

end NUMINAMATH_CALUDE_hypotenuse_of_special_triangle_l4042_404265


namespace NUMINAMATH_CALUDE_quadratic_factorization_l4042_404242

theorem quadratic_factorization (b k : ℝ) : 
  (∀ x, x^2 + b*x + 5 = (x - 2)^2 + k) → (b = -4 ∧ k = 1) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_factorization_l4042_404242


namespace NUMINAMATH_CALUDE_dance_steps_time_l4042_404244

def time_step1 : ℕ := 30

def time_step2 (t1 : ℕ) : ℕ := t1 / 2

def time_step3 (t1 t2 : ℕ) : ℕ := t1 + t2

def total_time (t1 t2 t3 : ℕ) : ℕ := t1 + t2 + t3

theorem dance_steps_time :
  total_time time_step1 (time_step2 time_step1) (time_step3 time_step1 (time_step2 time_step1)) = 90 :=
by sorry

end NUMINAMATH_CALUDE_dance_steps_time_l4042_404244


namespace NUMINAMATH_CALUDE_prime_divisors_and_totient_l4042_404272

theorem prime_divisors_and_totient (a b c t q : ℕ) (k n : ℕ) 
  (hk : k = c^t) 
  (hn : n = a^k - b^k) 
  (hq : ∃ (p : List ℕ), (∀ x ∈ p, Nat.Prime x) ∧ (List.length p ≥ q) ∧ (∀ x ∈ p, x ∣ k)) :
  (∃ (p : List ℕ), (∀ x ∈ p, Nat.Prime x) ∧ (List.length p ≥ q * t) ∧ (∀ x ∈ p, x ∣ n)) ∧
  (∃ m : ℕ, Nat.totient n = m * 2^(t/2)) := by
  sorry

end NUMINAMATH_CALUDE_prime_divisors_and_totient_l4042_404272


namespace NUMINAMATH_CALUDE_apples_for_juice_apples_for_juice_proof_l4042_404256

/-- Calculates the amount of apples used for fruit juice given the harvest and sales information -/
theorem apples_for_juice (total_harvest : ℕ) (restaurant_amount : ℕ) (bag_size : ℕ) 
  (total_sales : ℕ) (price_per_bag : ℕ) : ℕ :=
  let bags_sold := total_sales / price_per_bag
  let apples_sold := bags_sold * bag_size
  total_harvest - (restaurant_amount + apples_sold)

/-- Proves that 90 kg of apples were used for fruit juice given the specific values -/
theorem apples_for_juice_proof : 
  apples_for_juice 405 60 5 408 8 = 90 := by
  sorry

end NUMINAMATH_CALUDE_apples_for_juice_apples_for_juice_proof_l4042_404256


namespace NUMINAMATH_CALUDE_tv_series_seasons_l4042_404218

theorem tv_series_seasons (episodes_per_season : ℕ) (episodes_per_day : ℕ) (days_to_finish : ℕ) : 
  episodes_per_season = 20 →
  episodes_per_day = 2 →
  days_to_finish = 30 →
  (episodes_per_day * days_to_finish) / episodes_per_season = 3 := by
sorry

end NUMINAMATH_CALUDE_tv_series_seasons_l4042_404218


namespace NUMINAMATH_CALUDE_swimming_pool_count_l4042_404278

theorem swimming_pool_count (total : ℕ) (garage : ℕ) (both : ℕ) (neither : ℕ) : 
  total = 70 → garage = 50 → both = 35 → neither = 15 → 
  ∃ pool : ℕ, pool = 40 ∧ total = garage + pool - both + neither :=
by sorry

end NUMINAMATH_CALUDE_swimming_pool_count_l4042_404278


namespace NUMINAMATH_CALUDE_jan_extra_miles_l4042_404236

theorem jan_extra_miles (t s : ℝ) 
  (ian_distance : ℝ → ℝ → ℝ)
  (han_distance : ℝ → ℝ → ℝ)
  (jan_distance : ℝ → ℝ → ℝ)
  (h1 : ian_distance t s = s * t)
  (h2 : han_distance t s = (s + 10) * (t + 2))
  (h3 : han_distance t s = ian_distance t s + 100)
  (h4 : jan_distance t s = (s + 15) * (t + 3)) :
  jan_distance t s - ian_distance t s = 165 := by
sorry

end NUMINAMATH_CALUDE_jan_extra_miles_l4042_404236


namespace NUMINAMATH_CALUDE_fourth_bell_interval_l4042_404280

theorem fourth_bell_interval 
  (bell1 bell2 bell3 : ℕ) 
  (h1 : bell1 = 5)
  (h2 : bell2 = 8)
  (h3 : bell3 = 11)
  (h4 : ∃ bell4 : ℕ, Nat.lcm (Nat.lcm (Nat.lcm bell1 bell2) bell3) bell4 = 1320) :
  ∃ bell4 : ℕ, bell4 = 1320 ∧ 
    Nat.lcm (Nat.lcm (Nat.lcm bell1 bell2) bell3) bell4 = 1320 :=
by sorry

end NUMINAMATH_CALUDE_fourth_bell_interval_l4042_404280


namespace NUMINAMATH_CALUDE_expression_evaluation_l4042_404232

theorem expression_evaluation (x y : ℝ) (h1 : x > y) (h2 : y > 0) :
  (x^(2*y) * y^(2*x)) / (y^(2*y) * x^(2*x)) = (x/y)^(2*(y-x)) := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l4042_404232
