import Mathlib

namespace NUMINAMATH_CALUDE_same_color_probability_l1361_136130

/-- The probability of drawing two balls of the same color with replacement -/
theorem same_color_probability (green red blue : ℕ) (h_green : green = 8) (h_red : red = 6) (h_blue : blue = 4) :
  let total := green + red + blue
  (green / total) ^ 2 + (red / total) ^ 2 + (blue / total) ^ 2 = 29 / 81 :=
by sorry

end NUMINAMATH_CALUDE_same_color_probability_l1361_136130


namespace NUMINAMATH_CALUDE_sum_26_35_in_base7_l1361_136183

/-- Converts a natural number from base 10 to base 7 -/
def toBase7 (n : ℕ) : ℕ := sorry

/-- Theorem: The sum of 26 and 35 in base 10, when converted to base 7, equals 85 -/
theorem sum_26_35_in_base7 : toBase7 (26 + 35) = 85 := by sorry

end NUMINAMATH_CALUDE_sum_26_35_in_base7_l1361_136183


namespace NUMINAMATH_CALUDE_smith_cycling_time_comparison_l1361_136172

/-- Proves that the time taken for the second trip is 3/4 of the time taken for the first trip -/
theorem smith_cycling_time_comparison 
  (first_distance : ℝ) 
  (second_distance : ℝ) 
  (speed_multiplier : ℝ) 
  (h1 : first_distance = 90) 
  (h2 : second_distance = 270) 
  (h3 : speed_multiplier = 4) 
  (v : ℝ) 
  (hv : v > 0) : 
  (second_distance / (speed_multiplier * v)) / (first_distance / v) = 3/4 :=
by sorry

end NUMINAMATH_CALUDE_smith_cycling_time_comparison_l1361_136172


namespace NUMINAMATH_CALUDE_almas_test_score_l1361_136102

/-- Proves that Alma's test score is 45 given the specified conditions. -/
theorem almas_test_score (alma_age melina_age carlos_age alma_score carlos_score : ℕ) : 
  alma_age + melina_age + carlos_age = 3 * alma_score →
  melina_age = 3 * alma_age →
  carlos_age = 4 * alma_age →
  melina_age = 60 →
  carlos_score = 2 * alma_score + 15 →
  carlos_score - alma_score = melina_age →
  alma_score = 45 := by
  sorry

#check almas_test_score

end NUMINAMATH_CALUDE_almas_test_score_l1361_136102


namespace NUMINAMATH_CALUDE_prob_at_least_one_of_three_l1361_136121

/-- The probability that at least one of three independent events occurs, 
    given that each event has a probability of 1/3. -/
theorem prob_at_least_one_of_three (p : ℝ) (h_p : p = 1 / 3) :
  1 - (1 - p)^3 = 19 / 27 := by
  sorry

end NUMINAMATH_CALUDE_prob_at_least_one_of_three_l1361_136121


namespace NUMINAMATH_CALUDE_vanessa_savings_time_l1361_136163

def dress_cost : ℕ := 120
def initial_savings : ℕ := 25
def weekly_allowance : ℕ := 30
def arcade_expense : ℕ := 15
def snack_expense : ℕ := 5

def weekly_savings : ℕ := weekly_allowance - arcade_expense - snack_expense

theorem vanessa_savings_time : 
  ∃ (weeks : ℕ), 
    weeks * weekly_savings + initial_savings ≥ dress_cost ∧ 
    (weeks - 1) * weekly_savings + initial_savings < dress_cost ∧
    weeks = 10 := by
  sorry

end NUMINAMATH_CALUDE_vanessa_savings_time_l1361_136163


namespace NUMINAMATH_CALUDE_learning_time_difference_l1361_136117

def hours_english : ℕ := 6
def hours_chinese : ℕ := 2
def hours_spanish : ℕ := 3
def hours_french : ℕ := 1

theorem learning_time_difference : 
  (hours_english + hours_chinese) - (hours_spanish + hours_french) = 4 := by
  sorry

end NUMINAMATH_CALUDE_learning_time_difference_l1361_136117


namespace NUMINAMATH_CALUDE_total_tickets_bought_l1361_136150

/-- Represents the cost of an adult ticket in dollars -/
def adult_ticket_cost : ℚ := 5.5

/-- Represents the cost of a child ticket in dollars -/
def child_ticket_cost : ℚ := 3.5

/-- Represents the total cost of all tickets bought in dollars -/
def total_cost : ℚ := 83.5

/-- Represents the number of children's tickets bought -/
def num_child_tickets : ℕ := 16

/-- Theorem stating that the total number of tickets bought is 21 -/
theorem total_tickets_bought : ℕ := by
  sorry

end NUMINAMATH_CALUDE_total_tickets_bought_l1361_136150


namespace NUMINAMATH_CALUDE_equation_equivalence_l1361_136118

theorem equation_equivalence : ∀ x : ℝ, (x = 3) ↔ (x - 3 = 0) := by
  sorry

end NUMINAMATH_CALUDE_equation_equivalence_l1361_136118


namespace NUMINAMATH_CALUDE_marcus_pebbles_l1361_136156

theorem marcus_pebbles (P : ℕ) : 
  P / 2 + 30 = 39 → P = 18 := by
  sorry

end NUMINAMATH_CALUDE_marcus_pebbles_l1361_136156


namespace NUMINAMATH_CALUDE_cupcake_flour_requirement_l1361_136144

-- Define the given quantities
def total_flour : ℝ := 6
def flour_for_cakes : ℝ := 4
def flour_per_cake : ℝ := 0.5
def flour_for_cupcakes : ℝ := 2
def price_per_cake : ℝ := 2.5
def price_per_cupcake : ℝ := 1
def total_earnings : ℝ := 30

-- Define the theorem
theorem cupcake_flour_requirement :
  ∃ (flour_per_cupcake : ℝ),
    flour_per_cupcake * (flour_for_cupcakes / flour_per_cupcake) = 
      total_earnings - (flour_for_cakes / flour_per_cake) * price_per_cake ∧
    flour_per_cupcake = 0.2 := by
  sorry

end NUMINAMATH_CALUDE_cupcake_flour_requirement_l1361_136144


namespace NUMINAMATH_CALUDE_inequality_equivalence_l1361_136129

theorem inequality_equivalence (x : ℝ) : 
  |2*x - 1| < |x| + 1 ↔ 0 < x ∧ x < 2 := by sorry

end NUMINAMATH_CALUDE_inequality_equivalence_l1361_136129


namespace NUMINAMATH_CALUDE_inequality_proof_l1361_136119

theorem inequality_proof (a b : ℝ) (h : 1 / a < 1 / b ∧ 1 / b < 0) :
  (a + b < a * b) ∧ (b / a + a / b > 2) := by sorry

end NUMINAMATH_CALUDE_inequality_proof_l1361_136119


namespace NUMINAMATH_CALUDE_online_store_prices_l1361_136188

/-- Represents the pricing structure for an online store --/
structure StorePricing where
  flatFee : ℝ
  commissionRate : ℝ

/-- Calculates the final price for a given store --/
def calculateFinalPrice (costPrice profit : ℝ) (store : StorePricing) : ℝ :=
  let sellingPrice := costPrice + profit
  sellingPrice + store.flatFee + store.commissionRate * sellingPrice

theorem online_store_prices (costPrice : ℝ) (profitRate : ℝ) 
    (storeA storeB storeC : StorePricing) : 
    costPrice = 18 ∧ 
    profitRate = 0.2 ∧
    storeA = { flatFee := 0, commissionRate := 0.2 } ∧
    storeB = { flatFee := 5, commissionRate := 0.1 } ∧
    storeC = { flatFee := 0, commissionRate := 0.15 } →
    let profit := profitRate * costPrice
    calculateFinalPrice costPrice profit storeA = 25.92 ∧
    calculateFinalPrice costPrice profit storeB = 28.76 ∧
    calculateFinalPrice costPrice profit storeC = 24.84 := by
  sorry

end NUMINAMATH_CALUDE_online_store_prices_l1361_136188


namespace NUMINAMATH_CALUDE_quadrilateral_area_theorem_l1361_136132

noncomputable def quadrilateral_area (P Q A B : ℝ × ℝ) : ℝ :=
  sorry

theorem quadrilateral_area_theorem (P Q A B : ℝ × ℝ) :
  let d := 3 -- distance between P and Q
  let r1 := Real.sqrt 3 -- radius of circle centered at P
  let r2 := 3 -- radius of circle centered at Q
  dist P Q = d ∧
  dist P A = r1 ∧
  dist Q A = r2 ∧
  dist P B = r1 ∧
  dist Q B = r2
  →
  quadrilateral_area P Q A B = (3 * Real.sqrt 5) / 2 := by
  sorry

end NUMINAMATH_CALUDE_quadrilateral_area_theorem_l1361_136132


namespace NUMINAMATH_CALUDE_tic_tac_toe_tie_games_l1361_136125

theorem tic_tac_toe_tie_games 
  (amy_wins : ℚ) 
  (lily_wins : ℚ) 
  (h1 : amy_wins = 5 / 12) 
  (h2 : lily_wins = 1 / 4) : 
  1 - (amy_wins + lily_wins) = 1 / 3 := by
sorry

end NUMINAMATH_CALUDE_tic_tac_toe_tie_games_l1361_136125


namespace NUMINAMATH_CALUDE_women_half_of_total_l1361_136116

/-- Represents the number of bones in different types of skeletons -/
structure BoneCount where
  woman : ℕ
  man : ℕ
  child : ℕ

/-- Represents the count of different types of skeletons -/
structure SkeletonCount where
  women : ℕ
  men : ℕ
  children : ℕ

theorem women_half_of_total (bc : BoneCount) (sc : SkeletonCount) : 
  bc.woman = 20 →
  bc.man = bc.woman + 5 →
  bc.child = bc.woman / 2 →
  sc.men = sc.children →
  sc.women + sc.men + sc.children = 20 →
  bc.woman * sc.women + bc.man * sc.men + bc.child * sc.children = 375 →
  2 * sc.women = sc.women + sc.men + sc.children := by
  sorry

#check women_half_of_total

end NUMINAMATH_CALUDE_women_half_of_total_l1361_136116


namespace NUMINAMATH_CALUDE_fraction_division_problem_solution_l1361_136165

theorem fraction_division (a b c d : ℚ) (hb : b ≠ 0) (hd : d ≠ 0) :
  (a / b) / (c / d) = (a * d) / (b * c) := by sorry

theorem problem_solution :
  (5 : ℚ) / 6 / ((11 : ℚ) / 12) = 10 / 11 := by sorry

end NUMINAMATH_CALUDE_fraction_division_problem_solution_l1361_136165


namespace NUMINAMATH_CALUDE_middle_angle_range_l1361_136128

theorem middle_angle_range (α β γ : Real) : 
  (0 ≤ α) → (0 ≤ β) → (0 ≤ γ) →  -- angles are non-negative
  (α + β + γ = 180) →             -- sum of angles in a triangle
  (α ≤ β) → (β ≤ γ) →             -- β is the middle angle
  (0 < β) ∧ (β < 90) :=           -- conclusion
by sorry

end NUMINAMATH_CALUDE_middle_angle_range_l1361_136128


namespace NUMINAMATH_CALUDE_inscribed_square_side_length_l1361_136122

/-- Given a right triangle PQR with legs of length 9 and 12, prove that a square inscribed
    with one side on the hypotenuse and vertices on the other two sides has side length 45/8 -/
theorem inscribed_square_side_length (P Q R : ℝ × ℝ) 
  (right_angle_P : (Q.1 - P.1) * (R.1 - P.1) + (Q.2 - P.2) * (R.2 - P.2) = 0)
  (leg_PQ : (Q.1 - P.1)^2 + (Q.2 - P.2)^2 = 9^2)
  (leg_PR : (R.1 - P.1)^2 + (R.2 - P.2)^2 = 12^2)
  (square : ℝ × ℝ → ℝ × ℝ → ℝ × ℝ → ℝ × ℝ → Prop)
  (inscribed : ∃ (A B C D : ℝ × ℝ), square A B C D ∧ 
    (A.1 - Q.1) * (R.1 - Q.1) + (A.2 - Q.2) * (R.2 - Q.2) = 0 ∧
    (∃ t : ℝ, 0 < t ∧ t < 1 ∧ B = (t * Q.1 + (1 - t) * R.1, t * Q.2 + (1 - t) * R.2)) ∧
    (∃ u : ℝ, 0 < u ∧ u < 1 ∧ D = (u * P.1 + (1 - u) * Q.1, u * P.2 + (1 - u) * Q.2)) ∧
    (∃ v : ℝ, 0 < v ∧ v < 1 ∧ C = (v * P.1 + (1 - v) * R.1, v * P.2 + (1 - v) * R.2)))
  : ∃ (A B C D : ℝ × ℝ), square A B C D ∧ 
    (B.1 - A.1)^2 + (B.2 - A.2)^2 = (45/8)^2 := by
  sorry

end NUMINAMATH_CALUDE_inscribed_square_side_length_l1361_136122


namespace NUMINAMATH_CALUDE_myfavorite_sum_l1361_136159

def letters : Finset Char := {'m', 'y', 'f', 'a', 'v', 'o', 'r', 'i', 't', 'e'}
def digits : Finset Nat := {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}

theorem myfavorite_sum (f : Char → Nat) 
  (h1 : Function.Bijective f)
  (h2 : ∀ c ∈ letters, f c ∈ digits) :
  (letters.sum fun c => f c) = 45 := by
  sorry

end NUMINAMATH_CALUDE_myfavorite_sum_l1361_136159


namespace NUMINAMATH_CALUDE_apps_deleted_l1361_136133

theorem apps_deleted (initial_apps new_apps final_apps : ℕ) :
  initial_apps = 10 →
  new_apps = 11 →
  final_apps = 4 →
  initial_apps + new_apps - final_apps = 17 :=
by
  sorry

end NUMINAMATH_CALUDE_apps_deleted_l1361_136133


namespace NUMINAMATH_CALUDE_dan_bought_18_stickers_l1361_136147

/-- The number of stickers Dan bought -/
def stickers_bought (initial_stickers : ℕ) : ℕ := 18

theorem dan_bought_18_stickers (initial_stickers : ℕ) :
  let cindy_remaining := initial_stickers - 15
  let dan_total := initial_stickers + stickers_bought initial_stickers
  dan_total = cindy_remaining + 33 :=
by
  sorry

end NUMINAMATH_CALUDE_dan_bought_18_stickers_l1361_136147


namespace NUMINAMATH_CALUDE_dime_count_proof_l1361_136154

/-- Represents the number of cents in a dollar -/
def cents_per_dollar : ℕ := 100

/-- Represents the value of a quarter in cents -/
def quarter_value : ℕ := 25

/-- Represents the value of a dime in cents -/
def dime_value : ℕ := 10

/-- Represents the value of a nickel in cents -/
def nickel_value : ℕ := 5

/-- Calculates the number of dimes given the total amount, number of quarters, and number of nickels -/
def calculate_dimes (total_amount : ℕ) (num_quarters : ℕ) (num_nickels : ℕ) : ℕ :=
  (total_amount * cents_per_dollar - (num_quarters * quarter_value + num_nickels * nickel_value)) / dime_value

theorem dime_count_proof (total_amount : ℕ) (num_quarters : ℕ) (num_nickels : ℕ) 
  (h1 : total_amount = 4)
  (h2 : num_quarters = 10)
  (h3 : num_nickels = 6) :
  calculate_dimes total_amount num_quarters num_nickels = 12 := by
  sorry

end NUMINAMATH_CALUDE_dime_count_proof_l1361_136154


namespace NUMINAMATH_CALUDE_product_of_y_coordinates_l1361_136180

/-- Theorem: Product of y-coordinates for point Q -/
theorem product_of_y_coordinates (y₁ y₂ : ℝ) : 
  (((4 - (-2))^2 + (y₁ - (-3))^2 = 7^2) ∧
   ((4 - (-2))^2 + (y₂ - (-3))^2 = 7^2)) →
  y₁ * y₂ = -4 := by
sorry

end NUMINAMATH_CALUDE_product_of_y_coordinates_l1361_136180


namespace NUMINAMATH_CALUDE_perfect_square_equation_l1361_136193

theorem perfect_square_equation (a b c : ℤ) 
  (h : (a - 5)^2 + (b - 12)^2 - (c - 13)^2 = a^2 + b^2 - c^2) : 
  ∃ k : ℤ, a^2 + b^2 - c^2 = k^2 := by
  sorry

end NUMINAMATH_CALUDE_perfect_square_equation_l1361_136193


namespace NUMINAMATH_CALUDE_algebraic_expression_value_l1361_136187

theorem algebraic_expression_value (a b : ℝ) (h : a - 2*b = 2) : 2*a - 4*b + 1 = 5 := by
  sorry

end NUMINAMATH_CALUDE_algebraic_expression_value_l1361_136187


namespace NUMINAMATH_CALUDE_baba_yaga_powder_division_l1361_136104

/-- Represents the weight measurement system with a possible consistent error --/
structure ScaleSystem where
  total_shown : ℤ
  part1_shown : ℤ
  part2_shown : ℤ
  error : ℤ

/-- The actual weights of the two parts of the powder --/
def actual_weights (s : ScaleSystem) : ℤ × ℤ :=
  (s.part1_shown - s.error, s.part2_shown - s.error)

/-- Theorem stating the correct weights given the scale measurements --/
theorem baba_yaga_powder_division (s : ScaleSystem) 
  (h1 : s.total_shown = 6)
  (h2 : s.part1_shown = 3)
  (h3 : s.part2_shown = 2)
  (h4 : s.total_shown = s.part1_shown + s.part2_shown - s.error) :
  actual_weights s = (4, 3) := by
  sorry


end NUMINAMATH_CALUDE_baba_yaga_powder_division_l1361_136104


namespace NUMINAMATH_CALUDE_reciprocal_square_roots_l1361_136164

theorem reciprocal_square_roots (a b c d : ℂ) : 
  (a^4 - a^2 - 5 = 0 ∧ b^4 - b^2 - 5 = 0 ∧ c^4 - c^2 - 5 = 0 ∧ d^4 - d^2 - 5 = 0) →
  (5 * (1/a)^4 + (1/a)^2 - 1 = 0 ∧ 5 * (1/b)^4 + (1/b)^2 - 1 = 0 ∧
   5 * (1/c)^4 + (1/c)^2 - 1 = 0 ∧ 5 * (1/d)^4 + (1/d)^2 - 1 = 0) :=
by sorry

end NUMINAMATH_CALUDE_reciprocal_square_roots_l1361_136164


namespace NUMINAMATH_CALUDE_not_perfect_square_l1361_136177

theorem not_perfect_square (n : ℕ) : ∀ m : ℕ, 4 * n^2 + 4 * n + 4 ≠ m^2 := by
  sorry

end NUMINAMATH_CALUDE_not_perfect_square_l1361_136177


namespace NUMINAMATH_CALUDE_zinc_copper_mixture_weight_l1361_136185

theorem zinc_copper_mixture_weight 
  (zinc_weight : Real) 
  (zinc_copper_ratio : Real) 
  (h1 : zinc_weight = 28.8) 
  (h2 : zinc_copper_ratio = 9 / 11) : 
  zinc_weight + (zinc_weight * (1 / zinc_copper_ratio)) = 64 := by
  sorry

end NUMINAMATH_CALUDE_zinc_copper_mixture_weight_l1361_136185


namespace NUMINAMATH_CALUDE_toys_per_day_l1361_136111

def total_weekly_production : ℕ := 5505
def working_days_per_week : ℕ := 5

theorem toys_per_day :
  total_weekly_production / working_days_per_week = 1101 :=
by
  sorry

end NUMINAMATH_CALUDE_toys_per_day_l1361_136111


namespace NUMINAMATH_CALUDE_intersection_of_M_and_N_l1361_136181

def M : Set ℝ := {x | (x - 1)^2 < 4}
def N : Set ℝ := {-1, 0, 1, 2, 3}

theorem intersection_of_M_and_N : M ∩ N = {0, 1, 2} := by sorry

end NUMINAMATH_CALUDE_intersection_of_M_and_N_l1361_136181


namespace NUMINAMATH_CALUDE_winning_percentage_correct_l1361_136149

/-- Represents the percentage of votes secured by the winning candidate -/
def winning_percentage : ℝ := 70

/-- Represents the total number of valid votes -/
def total_votes : ℕ := 450

/-- Represents the majority of votes by which the winning candidate won -/
def vote_majority : ℕ := 180

/-- Theorem stating that the winning percentage is correct given the conditions -/
theorem winning_percentage_correct :
  (winning_percentage / 100 * total_votes : ℝ) -
  ((100 - winning_percentage) / 100 * total_votes : ℝ) = vote_majority :=
sorry

end NUMINAMATH_CALUDE_winning_percentage_correct_l1361_136149


namespace NUMINAMATH_CALUDE_basis_linear_independence_l1361_136191

-- Define a 2D vector space
variable {V : Type*} [AddCommGroup V] [Module ℝ V]

-- Define the property of being a basis for a plane
def IsBasisForPlane (e₁ e₂ : V) : Prop :=
  ∀ v : V, ∃ (m n : ℝ), v = m • e₁ + n • e₂

-- Define the property of vectors being not collinear
def NotCollinear (e₁ e₂ : V) : Prop :=
  ∀ (k : ℝ), k • e₁ ≠ e₂

-- The main theorem
theorem basis_linear_independence
  (e₁ e₂ : V)
  (h_basis : IsBasisForPlane e₁ e₂)
  (h_not_collinear : NotCollinear e₁ e₂) :
  ∀ (m n : ℝ), m • e₁ + n • e₂ = 0 → m = 0 ∧ n = 0 :=
by sorry

end NUMINAMATH_CALUDE_basis_linear_independence_l1361_136191


namespace NUMINAMATH_CALUDE_smallest_divisor_partition_l1361_136120

/-- A function that returns the sum of divisors of a positive integer -/
def sumOfDivisors (n : ℕ+) : ℕ := sorry

/-- A function that checks if the divisors of a number can be partitioned into three sets with equal sums -/
def canPartitionDivisors (n : ℕ+) : Prop := sorry

/-- The theorem stating that 120 is the smallest positive integer with the required property -/
theorem smallest_divisor_partition :
  (∀ m : ℕ+, m < 120 → ¬(canPartitionDivisors m)) ∧ 
  (canPartitionDivisors 120) := by sorry

end NUMINAMATH_CALUDE_smallest_divisor_partition_l1361_136120


namespace NUMINAMATH_CALUDE_pauls_fishing_theorem_l1361_136196

/-- Calculates the number of fish caught given a fishing rate and total time -/
def fish_caught (rate : ℚ) (period : ℚ) (total_time : ℚ) : ℚ :=
  (total_time / period) * rate

theorem pauls_fishing_theorem :
  let rate : ℚ := 5 / 2  -- 5 fish per 2 hours
  let total_time : ℚ := 12
  fish_caught rate 2 total_time = 30 := by
sorry

end NUMINAMATH_CALUDE_pauls_fishing_theorem_l1361_136196


namespace NUMINAMATH_CALUDE_chameleon_color_change_l1361_136166

theorem chameleon_color_change (total : ℕ) (blue_initial red_initial : ℕ) 
  (blue_final red_final : ℕ) (changed : ℕ) : 
  total = 140 →
  total = blue_initial + red_initial →
  total = blue_final + red_final →
  blue_initial = 5 * blue_final →
  red_final = 3 * red_initial →
  changed = blue_initial - blue_final →
  changed = 80 := by
sorry

end NUMINAMATH_CALUDE_chameleon_color_change_l1361_136166


namespace NUMINAMATH_CALUDE_equal_shaded_areas_condition_l1361_136107

/-- Given a circle with radius s and an angle φ, where 0 < φ < π/4,
    this theorem states the necessary and sufficient condition for
    the equality of two specific areas related to the circle. --/
theorem equal_shaded_areas_condition (s : ℝ) (φ : ℝ) 
    (h1 : 0 < φ) (h2 : φ < π/4) (h3 : s > 0) :
  let sector_area := φ * s^2 / 2
  let triangle_area := s^2 * Real.tan φ / 2
  sector_area = triangle_area ↔ Real.tan φ = 3 * φ :=
sorry

end NUMINAMATH_CALUDE_equal_shaded_areas_condition_l1361_136107


namespace NUMINAMATH_CALUDE_trig_simplification_l1361_136139

theorem trig_simplification :
  (1 + Real.cos (20 * π / 180)) / (2 * Real.sin (20 * π / 180)) -
  Real.sin (10 * π / 180) * ((1 / Real.tan (5 * π / 180)) - Real.tan (5 * π / 180)) =
  Real.sqrt 3 / 2 := by sorry

end NUMINAMATH_CALUDE_trig_simplification_l1361_136139


namespace NUMINAMATH_CALUDE_monotonic_increasing_not_implies_positive_derivative_l1361_136145

theorem monotonic_increasing_not_implies_positive_derivative :
  ∃ (f : ℝ → ℝ) (a b : ℝ), a < b ∧
    (∀ x y, a < x ∧ x < y ∧ y < b → f x ≤ f y) ∧
    ¬(∀ x, a < x ∧ x < b → (deriv f x) > 0) :=
by sorry

end NUMINAMATH_CALUDE_monotonic_increasing_not_implies_positive_derivative_l1361_136145


namespace NUMINAMATH_CALUDE_square_area_12m_l1361_136113

/-- The area of a square with side length 12 meters is 144 square meters. -/
theorem square_area_12m : 
  let side_length : ℝ := 12
  let area : ℝ := side_length ^ 2
  area = 144 := by sorry

end NUMINAMATH_CALUDE_square_area_12m_l1361_136113


namespace NUMINAMATH_CALUDE_two_common_tangents_l1361_136126

-- Define the circles
def C1 (x y : ℝ) : Prop := (x - 5)^2 + (y - 3)^2 = 9
def C2 (x y : ℝ) : Prop := x^2 + y^2 - 4*x + 2*y - 9 = 0

-- Define the number of common tangent lines
def num_common_tangents : ℕ := 2

-- Theorem statement
theorem two_common_tangents :
  num_common_tangents = 2 :=
sorry

end NUMINAMATH_CALUDE_two_common_tangents_l1361_136126


namespace NUMINAMATH_CALUDE_trihedral_angle_obtuse_angles_l1361_136101

/-- Represents a trihedral angle -/
structure TrihedralAngle where
  AOB : ℝ
  BOC : ℝ
  COA : ℝ
  α : ℝ
  β : ℝ
  γ : ℝ

/-- Theorem: If all plane angles of a trihedral angle are obtuse, then all dihedral angles are obtuse -/
theorem trihedral_angle_obtuse_angles (t : TrihedralAngle)
  (h_AOB : t.AOB > π / 2)
  (h_BOC : t.BOC > π / 2)
  (h_COA : t.COA > π / 2) :
  t.α > π / 2 ∧ t.β > π / 2 ∧ t.γ > π / 2 := by
  sorry

end NUMINAMATH_CALUDE_trihedral_angle_obtuse_angles_l1361_136101


namespace NUMINAMATH_CALUDE_gcd_204_85_l1361_136158

theorem gcd_204_85 : Nat.gcd 204 85 = 17 := by
  sorry

end NUMINAMATH_CALUDE_gcd_204_85_l1361_136158


namespace NUMINAMATH_CALUDE_max_min_f_on_interval_l1361_136186

def f (x : ℝ) := x^3 - 3*x + 1

theorem max_min_f_on_interval :
  ∃ (max min : ℝ),
    (∀ x ∈ Set.Icc (-3) 0, f x ≤ max) ∧
    (∃ x ∈ Set.Icc (-3) 0, f x = max) ∧
    (∀ x ∈ Set.Icc (-3) 0, min ≤ f x) ∧
    (∃ x ∈ Set.Icc (-3) 0, f x = min) ∧
    max = 3 ∧ min = -17 := by
  sorry

end NUMINAMATH_CALUDE_max_min_f_on_interval_l1361_136186


namespace NUMINAMATH_CALUDE_joggers_meet_time_l1361_136179

def lap_times : List Nat := [3, 5, 9, 10]

def start_time : Nat := 7 * 60  -- 7:00 AM in minutes since midnight

theorem joggers_meet_time (lcm_result : Nat) 
  (h1 : lcm_result = Nat.lcm (Nat.lcm (Nat.lcm 3 5) 9) 10)
  (h2 : ∀ t ∈ lap_times, lcm_result % t = 0)
  (h3 : ∀ m : Nat, (∀ t ∈ lap_times, m % t = 0) → m ≥ lcm_result) :
  (start_time + lcm_result) % (24 * 60) = 8 * 60 + 30 := by sorry

end NUMINAMATH_CALUDE_joggers_meet_time_l1361_136179


namespace NUMINAMATH_CALUDE_k_value_l1361_136112

/-- The function f(x) = 4x^2 + 3x + 5 -/
def f (x : ℝ) : ℝ := 4 * x^2 + 3 * x + 5

/-- The function g(x) = x^2 + kx - 7 with parameter k -/
def g (k : ℝ) (x : ℝ) : ℝ := x^2 + k * x - 7

/-- Theorem stating that if f(5) - g(5) = 20, then k = 82/5 -/
theorem k_value (k : ℝ) : f 5 - g k 5 = 20 → k = 82 / 5 := by
  sorry

end NUMINAMATH_CALUDE_k_value_l1361_136112


namespace NUMINAMATH_CALUDE_gcd_problem_l1361_136115

theorem gcd_problem (b : ℤ) (h : ∃ k : ℤ, b = 887 * (2 * k + 1)) :
  Int.gcd (3 * b^2 + 47 * b + 91) (b + 17) = 3 := by
  sorry

end NUMINAMATH_CALUDE_gcd_problem_l1361_136115


namespace NUMINAMATH_CALUDE_prob_at_least_three_cured_value_l1361_136194

-- Define the probability of success for the drug
def drug_success_rate : ℝ := 0.9

-- Define the number of patients
def num_patients : ℕ := 4

-- Define the minimum number of successes we're interested in
def min_successes : ℕ := 3

-- Define the probability of at least 3 out of 4 patients being cured
def prob_at_least_three_cured : ℝ :=
  1 - (Nat.choose num_patients 0 * drug_success_rate^0 * (1 - drug_success_rate)^4 +
       Nat.choose num_patients 1 * drug_success_rate^1 * (1 - drug_success_rate)^3 +
       Nat.choose num_patients 2 * drug_success_rate^2 * (1 - drug_success_rate)^2)

-- Theorem statement
theorem prob_at_least_three_cured_value :
  prob_at_least_three_cured = 0.9477 := by
  sorry

end NUMINAMATH_CALUDE_prob_at_least_three_cured_value_l1361_136194


namespace NUMINAMATH_CALUDE_polynomial_rational_difference_l1361_136160

theorem polynomial_rational_difference (f : ℝ → ℝ) :
  (∃ a b c : ℝ, ∀ x, f x = a * x^2 + b * x + c) →
  (∀ x y : ℝ, ∃ q : ℚ, x - y = q → ∃ r : ℚ, f x - f y = r) →
  ∃ b : ℚ, ∃ c : ℝ, ∀ x, f x = b * x + c :=
sorry

end NUMINAMATH_CALUDE_polynomial_rational_difference_l1361_136160


namespace NUMINAMATH_CALUDE_shaded_area_theorem_l1361_136100

/-- Represents a point on the grid -/
structure GridPoint where
  x : ℚ
  y : ℚ

/-- Represents a line segment on the grid -/
inductive GridSegment
  | Horizontal : GridPoint → GridPoint → GridSegment
  | Vertical : GridPoint → GridPoint → GridSegment
  | Diagonal : GridPoint → GridPoint → GridSegment
  | Midpoint : GridPoint → GridPoint → GridSegment

/-- Represents a shaded area on the grid -/
structure ShadedArea where
  boundary : List GridSegment

/-- The main theorem statement -/
theorem shaded_area_theorem (grid : List (List ℚ)) (shaded_areas : List ShadedArea) :
  (List.length shaded_areas = 2015) →
  (∀ area ∈ shaded_areas, ∀ segment ∈ area.boundary,
    match segment with
    | GridSegment.Horizontal p1 p2 => p1.y = p2.y ∧ (p2.x - p1.x).den = 1
    | GridSegment.Vertical p1 p2 => p1.x = p2.x ∧ (p2.y - p1.y).den = 1
    | GridSegment.Diagonal p1 p2 => (p2.x - p1.x).num = (p2.x - p1.x).den ∧ (p2.y - p1.y).num = (p2.y - p1.y).den
    | GridSegment.Midpoint p1 p2 => (p2.x - p1.x).num = 1 ∧ (p2.x - p1.x).den = 2 ∧ (p2.y - p1.y).num = 1 ∧ (p2.y - p1.y).den = 2
  ) →
  (∃ total_area : ℚ, total_area = 95/2) :=
by sorry

end NUMINAMATH_CALUDE_shaded_area_theorem_l1361_136100


namespace NUMINAMATH_CALUDE_factorization_left_to_right_l1361_136176

theorem factorization_left_to_right (x : ℝ) : x^2 - 4 = (x + 2) * (x - 2) := by
  sorry

end NUMINAMATH_CALUDE_factorization_left_to_right_l1361_136176


namespace NUMINAMATH_CALUDE_intercept_sum_l1361_136170

theorem intercept_sum (m : ℕ) (x_0 y_0 : ℕ) : m = 17 →
  (2 * x_0) % m = 3 →
  (5 * y_0) % m = m - 3 →
  x_0 < m →
  y_0 < m →
  x_0 + y_0 = 22 := by
  sorry

end NUMINAMATH_CALUDE_intercept_sum_l1361_136170


namespace NUMINAMATH_CALUDE_field_trip_adults_l1361_136151

theorem field_trip_adults (van_capacity : ℕ) (num_students : ℕ) (num_vans : ℕ) : 
  van_capacity = 8 → num_students = 22 → num_vans = 3 → 
  (num_vans * van_capacity - num_students : ℕ) = 2 := by
  sorry

end NUMINAMATH_CALUDE_field_trip_adults_l1361_136151


namespace NUMINAMATH_CALUDE_f_even_when_a_zero_f_minimum_when_a_between_neg_one_and_one_l1361_136103

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := x^2 + 2 * abs (x - a)

-- Statement 1: When a = 0, f is an even function
theorem f_even_when_a_zero :
  ∀ x : ℝ, f 0 x = f 0 (-x) := by sorry

-- Statement 2: When -1 < a < 1, f achieves a minimum value of a^2
theorem f_minimum_when_a_between_neg_one_and_one :
  ∀ a : ℝ, -1 < a → a < 1 → ∀ x : ℝ, f a x ≥ a^2 := by sorry

end NUMINAMATH_CALUDE_f_even_when_a_zero_f_minimum_when_a_between_neg_one_and_one_l1361_136103


namespace NUMINAMATH_CALUDE_sets_intersection_empty_l1361_136189

-- Define the sets A, B, and C
def A : Set (ℝ × ℝ) := {p | p.2^2 - p.1 - 1 = 0}
def B : Set (ℝ × ℝ) := {p | 4*p.1^2 + 2*p.1 - 2*p.2 + 5 = 0}
def C (k b : ℕ) : Set (ℝ × ℝ) := {p | p.2 = k * p.1 + b}

-- State the theorem
theorem sets_intersection_empty :
  ∃! k b : ℕ, (A ∪ B) ∩ C k b = ∅ ∧ k = 1 ∧ b = 2 := by sorry

end NUMINAMATH_CALUDE_sets_intersection_empty_l1361_136189


namespace NUMINAMATH_CALUDE_largest_divisor_of_composite_l1361_136152

/-- A number is composite if it has a proper divisor -/
def IsComposite (n : ℕ) : Prop :=
  ∃ m : ℕ, 1 < m ∧ m < n ∧ n % m = 0

/-- The largest integer that always divides n + n^4 - n^3 for composite n > 6 -/
def LargestDivisor : ℕ := 6

theorem largest_divisor_of_composite (n : ℕ) (h1 : IsComposite n) (h2 : n > 6) :
  (∀ d : ℕ, d > LargestDivisor → ∃ m : ℕ, IsComposite m ∧ m > 6 ∧ ¬(d ∣ (m + m^4 - m^3))) ∧
  (LargestDivisor ∣ (n + n^4 - n^3)) :=
sorry

end NUMINAMATH_CALUDE_largest_divisor_of_composite_l1361_136152


namespace NUMINAMATH_CALUDE_partial_multiplication_reconstruction_l1361_136106

/-- Represents a partially visible digit (0-9 or unknown) -/
inductive PartialDigit
  | Known (n : Fin 10)
  | Unknown

/-- Represents a partially visible number -/
def PartialNumber := List PartialDigit

/-- Represents a multiplication step in the written method -/
structure MultiplicationStep where
  multiplicand : PartialNumber
  multiplier : PartialNumber
  partialProducts : List PartialNumber
  result : PartialNumber

/-- Check if a number matches a partial number -/
def matchesPartial (n : ℕ) (pn : PartialNumber) : Prop := sorry

/-- The main theorem to prove -/
theorem partial_multiplication_reconstruction 
  (step : MultiplicationStep)
  (h1 : step.multiplicand.length = 3)
  (h2 : step.multiplier.length = 3)
  (h3 : matchesPartial 56576 step.result)
  : ∃ (a b : ℕ), 
    a * b = 56500 ∧ 
    matchesPartial a step.multiplicand ∧ 
    matchesPartial b step.multiplier :=
sorry

end NUMINAMATH_CALUDE_partial_multiplication_reconstruction_l1361_136106


namespace NUMINAMATH_CALUDE_thirteen_fifth_power_mod_seven_l1361_136141

theorem thirteen_fifth_power_mod_seven : (13^5 : ℤ) ≡ 6 [ZMOD 7] := by
  sorry

end NUMINAMATH_CALUDE_thirteen_fifth_power_mod_seven_l1361_136141


namespace NUMINAMATH_CALUDE_canoe_rental_cost_l1361_136146

/-- Represents the daily rental cost and quantities for canoes and kayaks --/
structure RentalInfo where
  canoe_cost : ℝ
  kayak_cost : ℝ
  canoe_count : ℕ
  kayak_count : ℕ

/-- Calculates the total revenue from canoe and kayak rentals --/
def total_revenue (r : RentalInfo) : ℝ :=
  r.canoe_cost * r.canoe_count + r.kayak_cost * r.kayak_count

/-- Theorem stating the canoe rental cost given the problem conditions --/
theorem canoe_rental_cost :
  ∀ (r : RentalInfo),
    r.kayak_cost = 15 →
    r.canoe_count = (3 * r.kayak_count) / 2 →
    total_revenue r = 288 →
    r.canoe_count = r.kayak_count + 4 →
    r.canoe_cost = 14 := by
  sorry

end NUMINAMATH_CALUDE_canoe_rental_cost_l1361_136146


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l1361_136178

-- Define the sets A and B
def A : Set ℝ := {x | x > 2 ∨ x < -1}
def B : Set ℝ := {x | (x + 1) * (4 - x) < 4}

-- State the theorem
theorem intersection_of_A_and_B :
  A ∩ B = {x : ℝ | x > 3 ∨ x < -1} := by sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l1361_136178


namespace NUMINAMATH_CALUDE_actual_average_height_after_correction_actual_average_height_is_184_cm_l1361_136173

/-- The actual average height of boys in a class after correcting measurement errors -/
theorem actual_average_height_after_correction (num_boys : ℕ) 
  (initial_avg : ℝ) (wrong_heights : Fin 4 → ℝ) (correct_heights : Fin 4 → ℝ) : ℝ :=
  let inch_to_cm : ℝ := 2.54
  let total_initial_height : ℝ := num_boys * initial_avg
  let height_difference : ℝ := (wrong_heights 0 - correct_heights 0) + 
                                (wrong_heights 1 - correct_heights 1) + 
                                (wrong_heights 2 - correct_heights 2) + 
                                (wrong_heights 3 * inch_to_cm - correct_heights 3 * inch_to_cm)
  let corrected_total_height : ℝ := total_initial_height - height_difference
  let actual_avg : ℝ := corrected_total_height / num_boys
  actual_avg

/-- The actual average height of boys in the class is 184.00 cm (rounded to two decimal places) -/
theorem actual_average_height_is_184_cm : 
  ∃ (ε : ℝ), ε > 0 ∧ ε < 0.005 ∧ 
  |actual_average_height_after_correction 75 185 
    (λ i => [170, 195, 160, 70][i]) 
    (λ i => [140, 165, 190, 64][i]) - 184| < ε :=
by
  sorry

end NUMINAMATH_CALUDE_actual_average_height_after_correction_actual_average_height_is_184_cm_l1361_136173


namespace NUMINAMATH_CALUDE_f_neg_one_eq_three_l1361_136140

/-- Given a function f(x) = x^2 - 2x, prove that f(-1) = 3 -/
theorem f_neg_one_eq_three (f : ℝ → ℝ) (h : ∀ x, f x = x^2 - 2*x) : f (-1) = 3 := by
  sorry

end NUMINAMATH_CALUDE_f_neg_one_eq_three_l1361_136140


namespace NUMINAMATH_CALUDE_dove_flag_dimensions_l1361_136175

/-- Represents the shape of a dove on a square grid -/
structure DoveShape where
  area : ℝ
  perimeter_type : List String
  grid_type : String

/-- Represents the dimensions of a rectangular flag -/
structure FlagDimensions where
  length : ℝ
  height : ℝ

/-- Theorem: Given a dove shape with area 192 cm² on a square grid, 
    the flag dimensions are 24 cm × 16 cm -/
theorem dove_flag_dimensions 
  (dove : DoveShape) 
  (h1 : dove.area = 192) 
  (h2 : dove.perimeter_type = ["quarter-circle", "straight line"])
  (h3 : dove.grid_type = "square") :
  ∃ (flag : FlagDimensions), flag.length = 24 ∧ flag.height = 16 :=
by sorry

end NUMINAMATH_CALUDE_dove_flag_dimensions_l1361_136175


namespace NUMINAMATH_CALUDE_bag_probabilities_l1361_136136

/-- Definition of the bag of balls -/
structure Bag where
  total : ℕ
  red : ℕ
  yellow : ℕ

/-- Initial bag configuration -/
def initialBag : Bag := ⟨20, 5, 15⟩

/-- Probability of picking a ball of a certain color -/
def probability (bag : Bag) (color : ℕ) : ℚ :=
  color / bag.total

/-- Add balls to the bag -/
def addBalls (bag : Bag) (redAdd : ℕ) (yellowAdd : ℕ) : Bag :=
  ⟨bag.total + redAdd + yellowAdd, bag.red + redAdd, bag.yellow + yellowAdd⟩

theorem bag_probabilities (bag : Bag := initialBag) :
  (probability bag bag.yellow > probability bag bag.red) ∧
  (probability bag bag.red = 1/4) ∧
  (probability (addBalls bag 40 0) (bag.red + 40) = 3/4) ∧
  (probability (addBalls bag 14 4) (bag.red + 14) = 
   probability (addBalls bag 14 4) (bag.yellow + 4)) :=
by sorry

end NUMINAMATH_CALUDE_bag_probabilities_l1361_136136


namespace NUMINAMATH_CALUDE_waiter_tables_l1361_136161

theorem waiter_tables (people_per_table : ℕ) (total_customers : ℕ) (h1 : people_per_table = 9) (h2 : total_customers = 63) :
  total_customers / people_per_table = 7 := by
  sorry

end NUMINAMATH_CALUDE_waiter_tables_l1361_136161


namespace NUMINAMATH_CALUDE_triangular_array_coins_l1361_136123

-- Define the sum of the first N natural numbers
def triangular_sum (N : ℕ) : ℕ := N * (N + 1) / 2

-- Define the sum of digits function
def sum_of_digits (n : ℕ) : ℕ :=
  if n < 10 then n else (n % 10) + sum_of_digits (n / 10)

theorem triangular_array_coins :
  ∃ N : ℕ, triangular_sum N = 5050 ∧ sum_of_digits N = 1 :=
sorry

end NUMINAMATH_CALUDE_triangular_array_coins_l1361_136123


namespace NUMINAMATH_CALUDE_sum_of_digits_of_B_is_7_l1361_136162

/-- The sum of digits of a natural number -/
def sumOfDigits (n : ℕ) : ℕ := sorry

/-- The property that the sum of digits of a number is congruent to the number itself modulo 9 -/
axiom sum_of_digits_mod_9 (n : ℕ) : sumOfDigits n ≡ n [ZMOD 9]

/-- A is the sum of digits of 4444^444 -/
def A : ℕ := sumOfDigits (4444^444)

/-- B is the sum of digits of A -/
def B : ℕ := sumOfDigits A

/-- The main theorem: the sum of digits of B is 7 -/
theorem sum_of_digits_of_B_is_7 : sumOfDigits B = 7 := by sorry

end NUMINAMATH_CALUDE_sum_of_digits_of_B_is_7_l1361_136162


namespace NUMINAMATH_CALUDE_lidia_apps_to_buy_l1361_136138

-- Define the given conditions
def average_app_cost : ℕ := 4
def total_budget : ℕ := 66
def remaining_money : ℕ := 6

-- Define the number of apps to buy
def apps_to_buy : ℕ := (total_budget - remaining_money) / average_app_cost

-- Theorem statement
theorem lidia_apps_to_buy : apps_to_buy = 15 := by
  sorry

end NUMINAMATH_CALUDE_lidia_apps_to_buy_l1361_136138


namespace NUMINAMATH_CALUDE_complex_fraction_equality_l1361_136114

theorem complex_fraction_equality (a b : ℂ) (ha : a ≠ 0) (hb : b ≠ 0) 
  (hab : a + b ≠ 0) (h : a^3 + a^2*b + a*b^2 + b^3 = 0) : 
  (a^12 + b^12) / (a + b)^12 = 2/81 := by
sorry

end NUMINAMATH_CALUDE_complex_fraction_equality_l1361_136114


namespace NUMINAMATH_CALUDE_gildas_marbles_l1361_136190

theorem gildas_marbles (initial_marbles : ℝ) (initial_marbles_pos : initial_marbles > 0) :
  let remaining_after_pedro := initial_marbles * (1 - 0.25)
  let remaining_after_ebony := remaining_after_pedro * (1 - 0.15)
  let remaining_after_jimmy := remaining_after_ebony * (1 - 0.30)
  (remaining_after_jimmy / initial_marbles) * 100 = 44.625 := by
sorry

end NUMINAMATH_CALUDE_gildas_marbles_l1361_136190


namespace NUMINAMATH_CALUDE_percentage_goldfish_special_food_l1361_136131

-- Define the parameters
def total_goldfish : ℕ := 50
def food_per_goldfish : ℚ := 3/2
def special_food_cost : ℚ := 3
def total_special_food_cost : ℚ := 45

-- Define the theorem
theorem percentage_goldfish_special_food :
  (((total_special_food_cost / special_food_cost) / food_per_goldfish) / total_goldfish) * 100 = 20 := by
  sorry

end NUMINAMATH_CALUDE_percentage_goldfish_special_food_l1361_136131


namespace NUMINAMATH_CALUDE_floor_of_e_l1361_136105

theorem floor_of_e : ⌊Real.exp 1⌋ = 2 := by
  sorry

end NUMINAMATH_CALUDE_floor_of_e_l1361_136105


namespace NUMINAMATH_CALUDE_zero_existence_l1361_136184

open Real

-- Define the differential equation
def is_solution (y : ℝ → ℝ) : Prop :=
  ∀ x, (x^2 + 9) * (deriv^[2] y x) + (x^2 + 4) * y x = 0

-- Define the theorem
theorem zero_existence (y : ℝ → ℝ) 
  (h_sol : is_solution y) 
  (h_init1 : y 0 = 0) 
  (h_init2 : deriv y 0 = 1) :
  ∃ x ∈ Set.Icc (Real.sqrt (63/53) * π) (3*π/2), y x = 0 := by
  sorry

end NUMINAMATH_CALUDE_zero_existence_l1361_136184


namespace NUMINAMATH_CALUDE_cosine_roots_of_equation_l1361_136135

theorem cosine_roots_of_equation : 
  let f (t : ℝ) := 32 * t^5 - 40 * t^3 + 10 * t - Real.sqrt 3
  (f (Real.cos (6 * π / 180)) = 0) →
  (f (Real.cos (78 * π / 180)) = 0) ∧
  (f (Real.cos (150 * π / 180)) = 0) ∧
  (f (Real.cos (222 * π / 180)) = 0) ∧
  (f (Real.cos (294 * π / 180)) = 0) :=
by sorry

end NUMINAMATH_CALUDE_cosine_roots_of_equation_l1361_136135


namespace NUMINAMATH_CALUDE_min_value_theorem_l1361_136167

theorem min_value_theorem (x y : ℝ) (h1 : x > 0) (h2 : y > 0) (h3 : 2 * x + y = 2) :
  (2 / x) + (1 / y) ≥ 9 / 2 := by
sorry

end NUMINAMATH_CALUDE_min_value_theorem_l1361_136167


namespace NUMINAMATH_CALUDE_litter_patrol_pickup_l1361_136198

/-- The number of glass bottles picked up by the Litter Patrol -/
def glass_bottles : ℕ := 10

/-- The number of aluminum cans picked up by the Litter Patrol -/
def aluminum_cans : ℕ := 8

/-- The total number of pieces of litter picked up by the Litter Patrol -/
def total_litter : ℕ := glass_bottles + aluminum_cans

theorem litter_patrol_pickup :
  total_litter = 18 := by sorry

end NUMINAMATH_CALUDE_litter_patrol_pickup_l1361_136198


namespace NUMINAMATH_CALUDE_triangle_problem_l1361_136157

theorem triangle_problem (A B C : Real) (a b c : Real) :
  B = 2 * C →
  c = 2 →
  a = 1 →
  b = Real.sqrt 6 ∧
  Real.sin (2 * B - π / 3) = (7 * Real.sqrt 3 - Real.sqrt 15) / 16 := by
  sorry

end NUMINAMATH_CALUDE_triangle_problem_l1361_136157


namespace NUMINAMATH_CALUDE_journey_speed_problem_l1361_136124

/-- Proves that given a journey of 3 km, if traveling at speed v km/hr results in arriving 7 minutes late, 
    and traveling at 12 km/hr results in arriving 8 minutes early, then v = 6 km/hr. -/
theorem journey_speed_problem (v : ℝ) : 
  (3 / v - 3 / 12 = 15 / 60) → v = 6 := by
  sorry

end NUMINAMATH_CALUDE_journey_speed_problem_l1361_136124


namespace NUMINAMATH_CALUDE_arithmetic_progression_sum_l1361_136168

theorem arithmetic_progression_sum (a b : ℝ) : 
  0 < a ∧ 0 < b ∧ 
  4 < a ∧ a < b ∧ b < 16 ∧
  (b - a = a - 4) ∧ 
  (16 - b = b - a) ∧
  (b - a ≠ a - 4) →
  a + b = 20 := by sorry

end NUMINAMATH_CALUDE_arithmetic_progression_sum_l1361_136168


namespace NUMINAMATH_CALUDE_ten_object_rotation_l1361_136148

/-- Represents a circular arrangement of n objects -/
def CircularArrangement (n : ℕ) := Fin n

/-- The operation of switching two objects in the arrangement -/
def switch (arr : CircularArrangement n) (i j : Fin n) : CircularArrangement n :=
  sorry

/-- Checks if the arrangement is rotated one position clockwise -/
def isRotatedOneStep (original rotated : CircularArrangement n) : Prop :=
  sorry

/-- The minimum number of switches required to rotate the arrangement one step -/
def minSwitches (n : ℕ) : ℕ :=
  sorry

theorem ten_object_rotation (arr : CircularArrangement 10) :
  ∃ (switches : List (Fin 10 × Fin 10)),
    switches.length = 9 ∧
    isRotatedOneStep arr (switches.foldl (λ a (i, j) => switch a i j) arr) :=
  sorry

end NUMINAMATH_CALUDE_ten_object_rotation_l1361_136148


namespace NUMINAMATH_CALUDE_infinite_series_sum_l1361_136182

/-- The sum of the infinite series ∑(1/(n(n+3))) for n from 1 to infinity is equal to 11/18. -/
theorem infinite_series_sum : ∑' (n : ℕ), 1 / (n * (n + 3 : ℝ)) = 11 / 18 := by sorry

end NUMINAMATH_CALUDE_infinite_series_sum_l1361_136182


namespace NUMINAMATH_CALUDE_units_digit_sum_of_powers_l1361_136137

theorem units_digit_sum_of_powers : ∃ n : ℕ, n < 10 ∧ (35^87 + 93^53) % 10 = n ∧ n = 8 := by
  sorry

end NUMINAMATH_CALUDE_units_digit_sum_of_powers_l1361_136137


namespace NUMINAMATH_CALUDE_inequality_relation_to_line_l1361_136142

-- Define the line equation
def line_equation (x y a : ℝ) : Prop := x + (a - 1) * y + 3 = 0

-- Define the inequality
def inequality (x y a : ℝ) : Prop := x + (a - 1) * y + 3 > 0

-- Theorem statement
theorem inequality_relation_to_line :
  ∀ (a : ℝ), 
    (a > 1 → ∀ (x y : ℝ), inequality x y a → ¬(line_equation x y a)) ∧
    (a < 1 → ∀ (x y : ℝ), ¬(inequality x y a) → line_equation x y a) :=
by sorry

end NUMINAMATH_CALUDE_inequality_relation_to_line_l1361_136142


namespace NUMINAMATH_CALUDE_problem_solution_l1361_136195

theorem problem_solution :
  ∀ (x y : ℕ), 
    y > 3 → 
    x^2 + y^4 = 2*((x-6)^2 + (y+1)^2) → 
    x^2 + y^4 = 1994 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l1361_136195


namespace NUMINAMATH_CALUDE_not_cylinder_if_triangle_front_view_l1361_136192

/-- A type representing geometric bodies -/
inductive GeometricBody
  | Cylinder
  | Cone
  | Tetrahedron
  | TriangularPrism

/-- A type representing possible front views -/
inductive FrontView
  | Triangle
  | Rectangle
  | Circle

/-- A function that returns the front view of a geometric body -/
def frontView (body : GeometricBody) : FrontView :=
  match body with
  | GeometricBody.Cylinder => FrontView.Rectangle
  | GeometricBody.Cone => FrontView.Triangle
  | GeometricBody.Tetrahedron => FrontView.Triangle
  | GeometricBody.TriangularPrism => FrontView.Triangle

/-- Theorem: If a geometric body has a triangle as its front view, it cannot be a cylinder -/
theorem not_cylinder_if_triangle_front_view (body : GeometricBody) :
  frontView body = FrontView.Triangle → body ≠ GeometricBody.Cylinder :=
by
  sorry

end NUMINAMATH_CALUDE_not_cylinder_if_triangle_front_view_l1361_136192


namespace NUMINAMATH_CALUDE_tim_balloons_l1361_136197

theorem tim_balloons (dan_balloons : ℝ) (ratio : ℝ) : 
  dan_balloons = 29.0 → 
  ratio = 7.0 → 
  ⌊dan_balloons / ratio⌋ = 4 := by
sorry

end NUMINAMATH_CALUDE_tim_balloons_l1361_136197


namespace NUMINAMATH_CALUDE_gcd_problems_l1361_136110

theorem gcd_problems :
  (Nat.gcd 840 1764 = 84) ∧ (Nat.gcd 153 119 = 17) := by
  sorry

end NUMINAMATH_CALUDE_gcd_problems_l1361_136110


namespace NUMINAMATH_CALUDE_students_per_computer_l1361_136134

theorem students_per_computer :
  let initial_students : ℕ := 82
  let additional_students : ℕ := 16
  let total_students : ℕ := initial_students + additional_students
  let computers_after_increase : ℕ := 49
  let students_per_computer : ℚ := initial_students / (total_students / computers_after_increase : ℚ)
  students_per_computer = 2 := by
sorry

end NUMINAMATH_CALUDE_students_per_computer_l1361_136134


namespace NUMINAMATH_CALUDE_tan_product_special_angles_l1361_136155

theorem tan_product_special_angles : 
  Real.tan (π / 8) * Real.tan (3 * π / 8) * Real.tan (5 * π / 8) = 2 * Real.sqrt 7 := by
  sorry

end NUMINAMATH_CALUDE_tan_product_special_angles_l1361_136155


namespace NUMINAMATH_CALUDE_composite_sum_l1361_136109

theorem composite_sum (a b c d : ℕ) (h1 : c > b) (h2 : a + b + c + d = a * b - c * d) :
  ∃ (x y : ℕ), x > 1 ∧ y > 1 ∧ a + c = x * y :=
sorry

end NUMINAMATH_CALUDE_composite_sum_l1361_136109


namespace NUMINAMATH_CALUDE_height_estimate_l1361_136171

/-- Given a survey of 1500 first-year high school students' heights:
    - The height range [160cm, 170cm] is divided into two groups of 5cm each
    - 'a' is the height of the histogram rectangle for [160cm, 165cm]
    - 'b' is the height of the histogram rectangle for [165cm, 170cm]
    - 1 unit of height in the histogram corresponds to 1500 students
    Then, the estimated number of students with heights in [160cm, 170cm] is 7500(a+b) -/
theorem height_estimate (a b : ℝ) : ℝ :=
  let total_students : ℕ := 1500
  let group_width : ℝ := 5
  let scale : ℝ := 1500
  7500 * (a + b)

#check height_estimate

end NUMINAMATH_CALUDE_height_estimate_l1361_136171


namespace NUMINAMATH_CALUDE_fraction_to_decimal_l1361_136169

theorem fraction_to_decimal : (13 : ℚ) / 200 = (52 : ℚ) / 100 := by sorry

end NUMINAMATH_CALUDE_fraction_to_decimal_l1361_136169


namespace NUMINAMATH_CALUDE_rectangle_width_length_ratio_l1361_136143

/-- Given a rectangle with width w, length 10, and perimeter 30, 
    prove that the ratio of width to length is 1:2 -/
theorem rectangle_width_length_ratio 
  (w : ℝ) 
  (h1 : w > 0)
  (h2 : 2 * w + 2 * 10 = 30) : 
  w / 10 = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_width_length_ratio_l1361_136143


namespace NUMINAMATH_CALUDE_expression_simplification_l1361_136174

theorem expression_simplification (x y z : ℝ) 
  (h_pos : 0 < z ∧ z < y ∧ y < x) : 
  (x^z * y^x * z^y) / (z^z * y^y * x^x) = (x/z)^(z-y) := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l1361_136174


namespace NUMINAMATH_CALUDE_imaginary_part_of_z_l1361_136153

theorem imaginary_part_of_z (z : ℂ) (h : z + (3 - 4*I) = 1) : z.im = 4 := by
  sorry

end NUMINAMATH_CALUDE_imaginary_part_of_z_l1361_136153


namespace NUMINAMATH_CALUDE_megan_earnings_after_discount_l1361_136199

/-- Calculates Megan's earnings from selling necklaces at a garage sale with a discount --/
theorem megan_earnings_after_discount :
  let bead_necklaces : ℕ := 7
  let bead_price : ℕ := 5
  let gem_necklaces : ℕ := 3
  let gem_price : ℕ := 15
  let discount_rate : ℚ := 1/5  -- 20% as a rational number
  
  let total_before_discount := bead_necklaces * bead_price + gem_necklaces * gem_price
  let discount_amount := (total_before_discount : ℚ) * discount_rate
  let earnings_after_discount := (total_before_discount : ℚ) - discount_amount
  
  earnings_after_discount = 64 := by sorry

end NUMINAMATH_CALUDE_megan_earnings_after_discount_l1361_136199


namespace NUMINAMATH_CALUDE_hyperbola_eccentricity_l1361_136108

/-- Given a hyperbola with equation x²/a² - y²/2 = 1 where a > √2, 
    if the angle between its asymptotes is π/3, 
    then its eccentricity is 2√3/3 -/
theorem hyperbola_eccentricity (a : ℝ) (h1 : a > Real.sqrt 2) :
  let angle_between_asymptotes := π / 3
  let slope_of_asymptote := Real.sqrt 2 / a
  let eccentricity := Real.sqrt (a^2 + 2) / a
  (angle_between_asymptotes = π / 3 ∧ 
   slope_of_asymptote = Real.tan (π / 6)) →
  eccentricity = 2 * Real.sqrt 3 / 3 := by
  sorry

end NUMINAMATH_CALUDE_hyperbola_eccentricity_l1361_136108


namespace NUMINAMATH_CALUDE_factor_tree_value_l1361_136127

theorem factor_tree_value : ∀ (A B C D E : ℕ),
  A = B * C →
  B = 3 * D →
  D = 3 * 2 →
  C = 5 * E →
  E = 5 * 2 →
  A = 900 := by
  sorry

end NUMINAMATH_CALUDE_factor_tree_value_l1361_136127
