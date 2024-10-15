import Mathlib

namespace NUMINAMATH_CALUDE_quadratic_root_theorem_l3201_320173

/-- Represents a quadratic equation ax^2 + bx + c = 0 -/
structure QuadraticEquation where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Checks if three real numbers form a geometric sequence -/
def isGeometricSequence (a b c : ℝ) : Prop :=
  ∃ k : ℝ, b = k * a ∧ c = k * b

/-- Theorem: The root of a specific quadratic equation -/
theorem quadratic_root_theorem (p q r : ℝ) (h1 : isGeometricSequence p q r)
    (h2 : p ≤ q ∧ q ≤ r ∧ r ≤ 0) (h3 : ∃! x : ℝ, p * x^2 + q * x + r = 0) :
    ∃ x : ℝ, p * x^2 + q * x + r = 0 ∧ x = -1 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_root_theorem_l3201_320173


namespace NUMINAMATH_CALUDE_interval_and_sum_l3201_320157

theorem interval_and_sum : 
  ∃ (m M : ℝ), 
    (∀ x : ℝ, x > 0 ∧ 2 * |x^2 - 9| ≤ 9 * |x| ↔ m ≤ x ∧ x ≤ M) ∧
    m = 3/2 ∧ 
    M = 6 ∧
    10 * m + M = 21 := by
  sorry

end NUMINAMATH_CALUDE_interval_and_sum_l3201_320157


namespace NUMINAMATH_CALUDE_negative_division_subtraction_l3201_320170

theorem negative_division_subtraction : (-96) / (-24) - 3 = 1 := by
  sorry

end NUMINAMATH_CALUDE_negative_division_subtraction_l3201_320170


namespace NUMINAMATH_CALUDE_cubic_tangent_max_value_l3201_320174

/-- A cubic function with parameters a and b -/
def f (a b x : ℝ) : ℝ := x^3 + a*x^2 + b*x

/-- The derivative of f with respect to x -/
def f' (a b x : ℝ) : ℝ := 3*x^2 + 2*a*x + b

theorem cubic_tangent_max_value (a b m : ℝ) (hm : m ≠ 0) :
  (f a b m = 0) →                          -- f(x) is zero at x = m
  (f' a b m = 0) →                         -- f'(x) is zero at x = m
  (∀ x, f a b x ≤ (1/2)) →                 -- maximum value of f(x) is 1/2
  (∃ x, f a b x = (1/2)) →                 -- f(x) achieves the maximum value 1/2
  m = (3/2) := by sorry

end NUMINAMATH_CALUDE_cubic_tangent_max_value_l3201_320174


namespace NUMINAMATH_CALUDE_infinite_sum_not_diff_powers_l3201_320113

theorem infinite_sum_not_diff_powers (n : ℕ) (hn : n > 1) :
  ∃ S : Set ℕ, (Set.Infinite S) ∧
    (∀ k ∈ S, ∃ a b : ℕ, k = a^n + b^n) ∧
    (∀ k ∈ S, ∀ c d : ℕ, k ≠ c^n - d^n) :=
sorry

end NUMINAMATH_CALUDE_infinite_sum_not_diff_powers_l3201_320113


namespace NUMINAMATH_CALUDE_quadratic_root_implies_k_l3201_320183

theorem quadratic_root_implies_k (k : ℝ) : 
  (∃ x : ℝ, x^2 + k*x - 3 = 0) ∧ (1^2 + k*1 - 3 = 0) → k = 2 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_root_implies_k_l3201_320183


namespace NUMINAMATH_CALUDE_path_length_calculation_l3201_320136

/-- Represents the scale of a map in feet per inch -/
def map_scale : ℝ := 500

/-- Represents the length of the path on the map in inches -/
def path_length_on_map : ℝ := 3.5

/-- Calculates the actual length of the path in feet -/
def actual_path_length : ℝ := map_scale * path_length_on_map

theorem path_length_calculation :
  actual_path_length = 1750 := by sorry

end NUMINAMATH_CALUDE_path_length_calculation_l3201_320136


namespace NUMINAMATH_CALUDE_rectangle_square_overlap_ratio_l3201_320145

/-- Given a rectangle ABCD and a square EFGH, if the rectangle shares 40% of its area with the square,
    and the square shares 25% of its area with the rectangle, then the ratio of the rectangle's length
    to its width is 10. -/
theorem rectangle_square_overlap_ratio (AB AD s : ℝ) (h1 : AB > 0) (h2 : AD > 0) (h3 : s > 0) : 
  (0.4 * AB * AD = 0.25 * s^2) → (AD = s / 4) → AB / AD = 10 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_square_overlap_ratio_l3201_320145


namespace NUMINAMATH_CALUDE_exponential_models_for_rapid_change_l3201_320111

/-- Represents an exponential function model -/
structure ExponentialModel where
  -- Add necessary fields here
  rapidChange : Bool
  largeChangeInShortTime : Bool

/-- Represents a practical problem with rapid changes and large amounts of change in short periods -/
structure RapidChangeProblem where
  -- Add necessary fields here
  hasRapidChange : Bool
  hasLargeChangeInShortTime : Bool

/-- States that exponential models are generally used for rapid change problems -/
theorem exponential_models_for_rapid_change 
  (model : ExponentialModel) 
  (problem : RapidChangeProblem) : 
  model.rapidChange ∧ model.largeChangeInShortTime → 
  problem.hasRapidChange ∧ problem.hasLargeChangeInShortTime →
  (∃ (usage : Bool), usage = true) :=
by
  sorry

#check exponential_models_for_rapid_change

end NUMINAMATH_CALUDE_exponential_models_for_rapid_change_l3201_320111


namespace NUMINAMATH_CALUDE_symmetric_point_proof_l3201_320164

/-- A point in a 2D plane represented by its x and y coordinates. -/
structure Point where
  x : ℝ
  y : ℝ

/-- The origin point (0, 0) in a 2D plane. -/
def origin : Point := ⟨0, 0⟩

/-- Determines if two points are symmetric with respect to the origin. -/
def isSymmetricToOrigin (p1 p2 : Point) : Prop :=
  p1.x = -p2.x ∧ p1.y = -p2.y

/-- The given point (3, -1). -/
def givenPoint : Point := ⟨3, -1⟩

/-- The point to be proven symmetric to the given point. -/
def symmetricPoint : Point := ⟨-3, 1⟩

/-- Theorem stating that the symmetricPoint is symmetric to the givenPoint with respect to the origin. -/
theorem symmetric_point_proof : isSymmetricToOrigin givenPoint symmetricPoint := by
  sorry

end NUMINAMATH_CALUDE_symmetric_point_proof_l3201_320164


namespace NUMINAMATH_CALUDE_telescope_visual_range_increase_l3201_320112

theorem telescope_visual_range_increase (original_range new_range : ℝ) 
  (h1 : original_range = 80)
  (h2 : new_range = 150) :
  (new_range - original_range) / original_range * 100 = 87.5 := by
  sorry

end NUMINAMATH_CALUDE_telescope_visual_range_increase_l3201_320112


namespace NUMINAMATH_CALUDE_money_equalization_l3201_320167

theorem money_equalization (xiaoli_money xiaogang_money : ℕ) : 
  xiaoli_money = 18 → xiaogang_money = 24 → 
  (xiaogang_money - (xiaoli_money + xiaogang_money) / 2) = 3 := by
  sorry

end NUMINAMATH_CALUDE_money_equalization_l3201_320167


namespace NUMINAMATH_CALUDE_sum_simplification_l3201_320133

theorem sum_simplification : -2^3 + (-2)^4 + 2^2 - 2^3 = 4 := by
  sorry

end NUMINAMATH_CALUDE_sum_simplification_l3201_320133


namespace NUMINAMATH_CALUDE_cube_edge_length_l3201_320124

/-- Given the cost of paint, coverage per quart, and total cost to paint a cube,
    prove that the edge length of the cube is 10 feet. -/
theorem cube_edge_length
  (paint_cost_per_quart : ℝ)
  (coverage_per_quart : ℝ)
  (total_cost : ℝ)
  (h1 : paint_cost_per_quart = 3.2)
  (h2 : coverage_per_quart = 60)
  (h3 : total_cost = 32)
  : ∃ (edge_length : ℝ), edge_length = 10 ∧ 6 * edge_length^2 = total_cost / paint_cost_per_quart * coverage_per_quart :=
by sorry

end NUMINAMATH_CALUDE_cube_edge_length_l3201_320124


namespace NUMINAMATH_CALUDE_olivias_paper_pieces_l3201_320150

/-- The number of paper pieces Olivia used -/
def pieces_used : ℕ := 56

/-- The number of paper pieces Olivia has left -/
def pieces_left : ℕ := 25

/-- The initial number of paper pieces Olivia had -/
def initial_pieces : ℕ := pieces_used + pieces_left

theorem olivias_paper_pieces : initial_pieces = 81 := by
  sorry

end NUMINAMATH_CALUDE_olivias_paper_pieces_l3201_320150


namespace NUMINAMATH_CALUDE_grandmother_age_is_132_l3201_320154

-- Define the ages as natural numbers
def mason_age : ℕ := 20
def sydney_age : ℕ := 3 * mason_age
def father_age : ℕ := sydney_age + 6
def grandmother_age : ℕ := 2 * father_age

-- Theorem to prove
theorem grandmother_age_is_132 : grandmother_age = 132 := by
  sorry


end NUMINAMATH_CALUDE_grandmother_age_is_132_l3201_320154


namespace NUMINAMATH_CALUDE_rohan_salary_rohan_salary_proof_l3201_320175

/-- Rohan's monthly salary calculation --/
theorem rohan_salary (food_percent : ℝ) (rent_percent : ℝ) (entertainment_percent : ℝ) 
  (conveyance_percent : ℝ) (savings : ℝ) : ℝ :=
  let total_expenses_percent : ℝ := food_percent + rent_percent + entertainment_percent + conveyance_percent
  let savings_percent : ℝ := 1 - total_expenses_percent
  savings / savings_percent

/-- Proof of Rohan's monthly salary --/
theorem rohan_salary_proof :
  rohan_salary 0.4 0.2 0.1 0.1 2500 = 12500 := by
  sorry

end NUMINAMATH_CALUDE_rohan_salary_rohan_salary_proof_l3201_320175


namespace NUMINAMATH_CALUDE_exponent_division_equality_l3201_320152

theorem exponent_division_equality (a : ℝ) : a^6 / (-a)^2 = a^4 := by
  sorry

end NUMINAMATH_CALUDE_exponent_division_equality_l3201_320152


namespace NUMINAMATH_CALUDE_strengthened_erdos_mordell_inequality_l3201_320143

theorem strengthened_erdos_mordell_inequality (a b c : ℝ) (h : 0 < a ∧ 0 < b ∧ 0 < c) :
  let s := (a + b + c) / 2
  let area := Real.sqrt (s * (s - a) * (s - b) * (s - c))
  a^2 + b^2 + c^2 ≥ 4 * Real.sqrt 3 * area + (a - b)^2 + (b - c)^2 + (c - a)^2 := by
sorry

end NUMINAMATH_CALUDE_strengthened_erdos_mordell_inequality_l3201_320143


namespace NUMINAMATH_CALUDE_expression_evaluation_l3201_320125

theorem expression_evaluation : 
  let a : ℝ := 2 * Real.sin (π / 4) + (1 / 2)⁻¹
  ((a^2 - 4) / a) / ((4 * a - 4) / a - a) + 2 / (a - 2) = -1 - Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l3201_320125


namespace NUMINAMATH_CALUDE_fraction_simplification_l3201_320135

theorem fraction_simplification (a b x : ℝ) :
  (Real.sqrt (a^2 + x^2) - (x^2 - b*a^2) / Real.sqrt (a^2 + x^2) + b) / (a^2 + x^2 + b^2) =
  (1 + b) / Real.sqrt ((a^2 + x^2) * (a^2 + x^2 + b^2)) :=
by sorry

end NUMINAMATH_CALUDE_fraction_simplification_l3201_320135


namespace NUMINAMATH_CALUDE_find_M_l3201_320121

theorem find_M : ∃ M : ℕ+, (36 ^ 2 : ℕ) * (75 ^ 2) = (30 ^ 2) * (M.val ^ 2) ∧ M.val = 90 := by
  sorry

end NUMINAMATH_CALUDE_find_M_l3201_320121


namespace NUMINAMATH_CALUDE_compound_interest_calculation_l3201_320129

/-- Given a principal amount that yields $50 in simple interest over 2 years at 5% per annum,
    the compound interest for the same principal, rate, and time is $51.25. -/
theorem compound_interest_calculation (P : ℝ) : 
  (P * 0.05 * 2 = 50) →  -- Simple interest condition
  (P * (1 + 0.05)^2 - P = 51.25) :=  -- Compound interest calculation
by
  sorry


end NUMINAMATH_CALUDE_compound_interest_calculation_l3201_320129


namespace NUMINAMATH_CALUDE_sequence_inequality_l3201_320196

theorem sequence_inequality (a : ℕ → ℝ) (h_nonneg : ∀ n, a n ≥ 0)
  (h_ineq : ∀ m n, a (m + n) ≤ a m + a n) :
  ∀ m n, m > 0 → n ≥ m → a n ≤ m * a 1 + (n / m - 1) * a m :=
by sorry

end NUMINAMATH_CALUDE_sequence_inequality_l3201_320196


namespace NUMINAMATH_CALUDE_smallest_integer_with_remainders_l3201_320161

theorem smallest_integer_with_remainders : ∃ b : ℕ, 
  b > 0 ∧ 
  b % 9 = 5 ∧ 
  b % 11 = 7 ∧
  ∀ c : ℕ, c > 0 ∧ c % 9 = 5 ∧ c % 11 = 7 → b ≤ c :=
by sorry

end NUMINAMATH_CALUDE_smallest_integer_with_remainders_l3201_320161


namespace NUMINAMATH_CALUDE_angle_b_in_axisymmetric_triangle_l3201_320139

-- Define an axisymmetric triangle
structure AxisymmetricTriangle :=
  (A B C : ℝ)
  (axisymmetric : True)  -- This is a placeholder for the axisymmetric property
  (sum_of_angles : A + B + C = 180)

-- Theorem statement
theorem angle_b_in_axisymmetric_triangle 
  (triangle : AxisymmetricTriangle) 
  (angle_a_value : triangle.A = 70) :
  triangle.B = 70 ∨ triangle.B = 55 := by
  sorry

end NUMINAMATH_CALUDE_angle_b_in_axisymmetric_triangle_l3201_320139


namespace NUMINAMATH_CALUDE_aarons_brothers_l3201_320194

theorem aarons_brothers (bennett_brothers : ℕ) (h1 : bennett_brothers = 6) 
  (h2 : bennett_brothers = 2 * aaron_brothers - 2) : aaron_brothers = 4 := by
  sorry

end NUMINAMATH_CALUDE_aarons_brothers_l3201_320194


namespace NUMINAMATH_CALUDE_robin_gum_packages_l3201_320188

theorem robin_gum_packages (pieces_per_package : ℕ) (total_pieces : ℕ) (h1 : pieces_per_package = 15) (h2 : total_pieces = 135) :
  total_pieces / pieces_per_package = 9 := by
  sorry

end NUMINAMATH_CALUDE_robin_gum_packages_l3201_320188


namespace NUMINAMATH_CALUDE_no_solutions_in_interval_l3201_320118

theorem no_solutions_in_interval (x : Real) : 
  x ∈ Set.Icc 0 (2 * Real.pi) → 
  (1 / Real.sin x + 1 / Real.cos x ≠ 4) :=
by sorry

end NUMINAMATH_CALUDE_no_solutions_in_interval_l3201_320118


namespace NUMINAMATH_CALUDE_polygon_coloring_l3201_320140

/-- Given a regular 103-sided polygon with 79 red vertices and 24 blue vertices,
    A is the number of pairs of adjacent red vertices and
    B is the number of pairs of adjacent blue vertices. -/
theorem polygon_coloring (A B : ℕ) :
  (∀ i : ℕ, 0 ≤ i ∧ i ≤ 23 → (A = 55 + i ∧ B = i)) ∧
  (B = 14 →
    (Nat.choose 23 10 * Nat.choose 78 9) / 14 =
      (Nat.choose 23 9 * Nat.choose 78 9) / 10) :=
by sorry

end NUMINAMATH_CALUDE_polygon_coloring_l3201_320140


namespace NUMINAMATH_CALUDE_D_2021_2022_2023_odd_l3201_320190

def D : ℕ → ℕ
  | 0 => 1
  | 1 => 1
  | 2 => 1
  | n + 3 => D (n + 2) + D (n + 1)

theorem D_2021_2022_2023_odd :
  Odd (D 2021) ∧ Odd (D 2022) ∧ Odd (D 2023) := by
  sorry

end NUMINAMATH_CALUDE_D_2021_2022_2023_odd_l3201_320190


namespace NUMINAMATH_CALUDE_alpha_plus_beta_eq_115_l3201_320155

theorem alpha_plus_beta_eq_115 :
  ∃ (α β : ℝ), (∀ x : ℝ, (x - α) / (x + β) = (x^2 - 116*x + 2783) / (x^2 + 99*x - 4080)) →
  α + β = 115 := by
  sorry

end NUMINAMATH_CALUDE_alpha_plus_beta_eq_115_l3201_320155


namespace NUMINAMATH_CALUDE_max_gold_marbles_is_66_l3201_320151

/-- Represents the number of marbles of each color --/
structure MarbleCount where
  red : ℕ
  blue : ℕ
  gold : ℕ

/-- Represents an exchange of marbles --/
inductive Exchange
  | RedToGold : Exchange
  | BlueToGold : Exchange

/-- Applies an exchange to a MarbleCount --/
def applyExchange (mc : MarbleCount) (e : Exchange) : MarbleCount :=
  match e with
  | Exchange.RedToGold => 
      if mc.red ≥ 3 then ⟨mc.red - 3, mc.blue + 2, mc.gold + 1⟩ else mc
  | Exchange.BlueToGold => 
      if mc.blue ≥ 4 then ⟨mc.red + 1, mc.blue - 4, mc.gold + 1⟩ else mc

/-- Checks if any exchange is possible --/
def canExchange (mc : MarbleCount) : Prop :=
  mc.red ≥ 3 ∨ mc.blue ≥ 4

/-- The maximum number of gold marbles obtainable --/
def maxGoldMarbles : ℕ := 66

/-- The theorem to be proved --/
theorem max_gold_marbles_is_66 :
  ∀ (exchanges : List Exchange),
    let finalCount := (exchanges.foldl applyExchange ⟨80, 60, 0⟩)
    ¬(canExchange finalCount) →
    finalCount.gold = maxGoldMarbles :=
  sorry

end NUMINAMATH_CALUDE_max_gold_marbles_is_66_l3201_320151


namespace NUMINAMATH_CALUDE_cube_split_sequence_l3201_320171

theorem cube_split_sequence (n : ℕ) : ∃ (k : ℕ), 
  2019 = n^2 - (n - 1) + 2 * k ∧ 
  0 ≤ k ∧ 
  k < n ∧ 
  n = 45 := by
  sorry

end NUMINAMATH_CALUDE_cube_split_sequence_l3201_320171


namespace NUMINAMATH_CALUDE_range_of_a_l3201_320162

theorem range_of_a (a : ℝ) : 
  (¬ ∀ x ∈ Set.Icc 1 2, x^2 - a ≥ 0) → a > 1 := by
  sorry

end NUMINAMATH_CALUDE_range_of_a_l3201_320162


namespace NUMINAMATH_CALUDE_profit_for_two_yuan_reduction_selling_price_for_770_profit_no_price_for_880_profit_l3201_320100

/-- Represents the supermarket beverage pricing and sales model -/
structure BeverageModel where
  cost_price : ℝ
  initial_price : ℝ
  initial_sales : ℝ
  price_sensitivity : ℝ

/-- Calculates the profit for a given price reduction -/
def profit (model : BeverageModel) (price_reduction : ℝ) : ℝ :=
  let new_price := model.initial_price - price_reduction
  let new_sales := model.initial_sales + model.price_sensitivity * price_reduction
  (new_price - model.cost_price) * new_sales

/-- Theorem: The profit with a 2 yuan price reduction is 800 yuan -/
theorem profit_for_two_yuan_reduction (model : BeverageModel) 
  (h1 : model.cost_price = 48)
  (h2 : model.initial_price = 60)
  (h3 : model.initial_sales = 60)
  (h4 : model.price_sensitivity = 10) :
  profit model 2 = 800 := by sorry

/-- Theorem: To achieve a profit of 770 yuan, the selling price should be 55 yuan -/
theorem selling_price_for_770_profit (model : BeverageModel) 
  (h1 : model.cost_price = 48)
  (h2 : model.initial_price = 60)
  (h3 : model.initial_sales = 60)
  (h4 : model.price_sensitivity = 10) :
  ∃ (price_reduction : ℝ), profit model price_reduction = 770 ∧ 
  model.initial_price - price_reduction = 55 := by sorry

/-- Theorem: There is no selling price that can achieve a profit of 880 yuan -/
theorem no_price_for_880_profit (model : BeverageModel) 
  (h1 : model.cost_price = 48)
  (h2 : model.initial_price = 60)
  (h3 : model.initial_sales = 60)
  (h4 : model.price_sensitivity = 10) :
  ¬∃ (price_reduction : ℝ), profit model price_reduction = 880 := by sorry

end NUMINAMATH_CALUDE_profit_for_two_yuan_reduction_selling_price_for_770_profit_no_price_for_880_profit_l3201_320100


namespace NUMINAMATH_CALUDE_new_person_weight_l3201_320116

/-- Proves that the weight of a new person is 380 kg given the conditions of the problem -/
theorem new_person_weight (initial_count : ℕ) (replaced_weight : ℝ) (average_increase : ℝ) :
  initial_count = 20 →
  replaced_weight = 80 →
  average_increase = 15 →
  (initial_count : ℝ) * average_increase + replaced_weight = 380 :=
by sorry

end NUMINAMATH_CALUDE_new_person_weight_l3201_320116


namespace NUMINAMATH_CALUDE_sqrt_sum_simplification_l3201_320146

theorem sqrt_sum_simplification : 
  Real.sqrt 75 - 9 * Real.sqrt (1/3) + Real.sqrt 48 = 6 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_sum_simplification_l3201_320146


namespace NUMINAMATH_CALUDE_parallel_planes_transitive_perpendicular_planes_from_perpendicular_lines_parallel_line_plane_from_perpendicular_planes_l3201_320187

-- Define the types for planes and lines
variable (Plane Line : Type)

-- Define the relations
variable (parallel_planes : Plane → Plane → Prop)
variable (perpendicular_plane_line : Plane → Line → Prop)
variable (perpendicular_planes : Plane → Plane → Prop)
variable (perpendicular_lines : Line → Line → Prop)
variable (line_in_plane : Line → Plane → Prop)
variable (parallel_line_plane : Line → Plane → Prop)

-- Axioms for the relations
axiom parallel_planes_trans {a b c : Plane} : 
  parallel_planes a b → parallel_planes b c → parallel_planes a c

axiom perpendicular_planes_of_perpendicular_lines {a b : Plane} {m n : Line} :
  perpendicular_plane_line a m → perpendicular_plane_line b n → 
  perpendicular_lines m n → perpendicular_planes a b

axiom parallel_line_plane_of_perpendicular_planes {a b : Plane} {m : Line} :
  perpendicular_planes a b → perpendicular_plane_line b m → 
  ¬line_in_plane m a → parallel_line_plane m a

-- Theorems to prove
theorem parallel_planes_transitive {a b c : Plane} :
  parallel_planes a b → parallel_planes b c → parallel_planes a c :=
sorry

theorem perpendicular_planes_from_perpendicular_lines {a b : Plane} {m n : Line} :
  perpendicular_plane_line a m → perpendicular_plane_line b n → 
  perpendicular_lines m n → perpendicular_planes a b :=
sorry

theorem parallel_line_plane_from_perpendicular_planes {a b : Plane} {m : Line} :
  perpendicular_planes a b → perpendicular_plane_line b m → 
  ¬line_in_plane m a → parallel_line_plane m a :=
sorry

end NUMINAMATH_CALUDE_parallel_planes_transitive_perpendicular_planes_from_perpendicular_lines_parallel_line_plane_from_perpendicular_planes_l3201_320187


namespace NUMINAMATH_CALUDE_three_digit_factorial_sum_l3201_320176

def factorial (n : Nat) : Nat :=
  match n with
  | 0 => 1
  | n + 1 => (n + 1) * factorial n

def sum_of_digit_factorials (n : Nat) : Nat :=
  let hundreds := n / 100
  let tens := (n / 10) % 10
  let units := n % 10
  factorial hundreds + factorial tens + factorial units

theorem three_digit_factorial_sum :
  ∃ (n : Nat), 100 ≤ n ∧ n < 1000 ∧ n / 100 = 2 ∧ n = sum_of_digit_factorials n :=
by
  sorry

end NUMINAMATH_CALUDE_three_digit_factorial_sum_l3201_320176


namespace NUMINAMATH_CALUDE_f_max_value_l3201_320160

/-- The quadratic function f(x) = -3x^2 + 15x + 9 -/
def f (x : ℝ) : ℝ := -3 * x^2 + 15 * x + 9

/-- The maximum value of f(x) is 111/4 -/
theorem f_max_value : ∃ (M : ℝ), M = 111/4 ∧ ∀ (x : ℝ), f x ≤ M := by
  sorry

end NUMINAMATH_CALUDE_f_max_value_l3201_320160


namespace NUMINAMATH_CALUDE_store_purchase_price_l3201_320138

theorem store_purchase_price (tax_rate : Real) (discount : Real) (cody_payment : Real) : 
  tax_rate = 0.05 → discount = 8 → cody_payment = 17 → 
  ∃ (original_price : Real), 
    (original_price * (1 + tax_rate) - discount) / 2 = cody_payment ∧ 
    original_price = 40 := by
  sorry

end NUMINAMATH_CALUDE_store_purchase_price_l3201_320138


namespace NUMINAMATH_CALUDE_complex_fraction_simplification_l3201_320102

theorem complex_fraction_simplification : (1 + 2*Complex.I) / (2 - Complex.I) = Complex.I := by
  sorry

end NUMINAMATH_CALUDE_complex_fraction_simplification_l3201_320102


namespace NUMINAMATH_CALUDE_min_distance_to_origin_l3201_320185

/-- The minimum distance from a point on the line 5x + 12y = 60 to the origin (0, 0) is 60/13 -/
theorem min_distance_to_origin (x y : ℝ) : 
  5 * x + 12 * y = 60 → 
  (∃ (d : ℝ), d = 60 / 13 ∧ 
    ∀ (p : ℝ × ℝ), p.1 * 5 + p.2 * 12 = 60 → 
      d ≤ Real.sqrt (p.1^2 + p.2^2)) := by
  sorry

end NUMINAMATH_CALUDE_min_distance_to_origin_l3201_320185


namespace NUMINAMATH_CALUDE_inner_square_side_length_l3201_320192

/-- A square with side length 2 -/
structure Square :=
  (A B C D : ℝ × ℝ)
  (is_square : A = (0, 2) ∧ B = (2, 2) ∧ C = (2, 0) ∧ D = (0, 0))

/-- A smaller square inside the main square -/
structure InnerSquare (outer : Square) :=
  (P Q R S : ℝ × ℝ)
  (P_midpoint : P = (1, 2))
  (S_on_BC : S.1 = 2)
  (is_square : (P.1 - S.1)^2 + (P.2 - S.2)^2 = (Q.1 - R.1)^2 + (Q.2 - R.2)^2)

/-- The theorem to be proved -/
theorem inner_square_side_length (outer : Square) (inner : InnerSquare outer) :
  Real.sqrt ((inner.P.1 - inner.S.1)^2 + (inner.P.2 - inner.S.2)^2) = 1 := by
  sorry

end NUMINAMATH_CALUDE_inner_square_side_length_l3201_320192


namespace NUMINAMATH_CALUDE_polynomial_functional_equation_l3201_320186

/-- A polynomial satisfies P(x+1) = P(x) + 2x + 1 for all x if and only if it is of the form x^2 + c for some constant c. -/
theorem polynomial_functional_equation (P : ℝ → ℝ) :
  (∀ x, P (x + 1) = P x + 2 * x + 1) ↔
  (∃ c, ∀ x, P x = x^2 + c) :=
sorry

end NUMINAMATH_CALUDE_polynomial_functional_equation_l3201_320186


namespace NUMINAMATH_CALUDE_retail_price_calculation_l3201_320105

/-- The retail price of a machine given wholesale price, discount rate, and profit rate -/
theorem retail_price_calculation (W D R : ℚ) (h1 : W = 126) (h2 : D = 0.10) (h3 : R = 0.20) :
  ∃ P : ℚ, (1 - D) * P = W + R * W :=
by
  sorry

end NUMINAMATH_CALUDE_retail_price_calculation_l3201_320105


namespace NUMINAMATH_CALUDE_mapping_A_to_B_l3201_320156

def A : Finset ℕ := {1, 2, 3, 4, 5}
def B : Finset ℕ := {0, 3, 8, 15, 24}

def f (x : ℕ) : ℕ := x^2 - 1

theorem mapping_A_to_B :
  ∀ x ∈ A, f x ∈ B :=
by sorry

end NUMINAMATH_CALUDE_mapping_A_to_B_l3201_320156


namespace NUMINAMATH_CALUDE_parabola_and_line_properties_l3201_320158

-- Define the parabola
def parabola (p : ℝ) (x y : ℝ) : Prop := y^2 = 2*p*x ∧ p > 0

-- Define the intersecting line
def intersecting_line (x y : ℝ) : Prop := ∃ t, x = t*y + 4

-- Define the tangent line
def tangent_line (k m x y : ℝ) : Prop := y = k*x + m

-- Define the circle condition
def circle_condition (x₀ k m r : ℝ) : Prop :=
  ∃ x y, tangent_line k m x y ∧
  (2*m^2 - r)*(x₀ - r) + 2*k*m*x₀ + 2*m^2 = 0

theorem parabola_and_line_properties :
  ∀ p : ℝ,
  (∃ x₁ y₁ x₂ y₂ : ℝ,
    parabola p x₁ y₁ ∧ parabola p x₂ y₂ ∧
    intersecting_line x₁ y₁ ∧ intersecting_line x₂ y₂ ∧
    y₁ * y₂ = -8) →
  p = 1 ∧
  (∀ k m r : ℝ,
    (∃ x y, parabola 1 x y ∧ tangent_line k m x y) →
    (∀ x₀, circle_condition x₀ k m r → x₀ = -1/2)) :=
by sorry

end NUMINAMATH_CALUDE_parabola_and_line_properties_l3201_320158


namespace NUMINAMATH_CALUDE_final_cost_is_33_08_l3201_320117

/-- The cost of a single deck in dollars -/
def deck_cost : ℚ := 7

/-- The discount rate as a decimal -/
def discount_rate : ℚ := 0.1

/-- The sales tax rate as a decimal -/
def sales_tax_rate : ℚ := 0.05

/-- The number of decks Frank bought -/
def frank_decks : ℕ := 3

/-- The number of decks Frank's friend bought -/
def friend_decks : ℕ := 2

/-- The total cost before discount and tax -/
def total_cost : ℚ := deck_cost * (frank_decks + friend_decks)

/-- The discounted cost -/
def discounted_cost : ℚ := total_cost * (1 - discount_rate)

/-- The final cost including tax -/
def final_cost : ℚ := discounted_cost * (1 + sales_tax_rate)

/-- Theorem stating the final cost is $33.08 -/
theorem final_cost_is_33_08 : 
  ∃ (ε : ℚ), abs (final_cost - 33.08) < ε ∧ ε = 0.005 := by
  sorry

end NUMINAMATH_CALUDE_final_cost_is_33_08_l3201_320117


namespace NUMINAMATH_CALUDE_largest_divisor_five_consecutive_integers_l3201_320101

theorem largest_divisor_five_consecutive_integers :
  ∀ n : ℤ, ∃ m : ℤ, m > 60 ∧ ¬(m ∣ (n * (n+1) * (n+2) * (n+3) * (n+4))) ∧
  ∀ k : ℤ, k ≤ 60 → k ∣ (n * (n+1) * (n+2) * (n+3) * (n+4)) :=
by sorry

end NUMINAMATH_CALUDE_largest_divisor_five_consecutive_integers_l3201_320101


namespace NUMINAMATH_CALUDE_smallest_inverse_domain_l3201_320114

def g (x : ℝ) : ℝ := (x - 3)^2 - 1

theorem smallest_inverse_domain (d : ℝ) :
  (∀ x y, x ≥ d → y ≥ d → g x = g y → x = y) ↔ d ≥ 3 :=
sorry

end NUMINAMATH_CALUDE_smallest_inverse_domain_l3201_320114


namespace NUMINAMATH_CALUDE_axis_of_symmetry_is_x_equals_one_l3201_320142

/-- Represents a parabola of the form y = a(x - h)^2 + k --/
structure Parabola where
  a : ℝ
  h : ℝ
  k : ℝ

/-- The given parabola y = -2(x-1)^2 + 3 --/
def givenParabola : Parabola :=
  { a := -2
  , h := 1
  , k := 3 }

/-- The axis of symmetry of a parabola --/
def axisOfSymmetry (p : Parabola) : ℝ := p.h

theorem axis_of_symmetry_is_x_equals_one :
  axisOfSymmetry givenParabola = 1 := by sorry

end NUMINAMATH_CALUDE_axis_of_symmetry_is_x_equals_one_l3201_320142


namespace NUMINAMATH_CALUDE_abc_sum_l3201_320165

theorem abc_sum (a b c : ℕ+) 
  (eq1 : a * b + c = 55)
  (eq2 : b * c + a = 55)
  (eq3 : a * c + b = 55) :
  a + b + c = 40 := by
  sorry

end NUMINAMATH_CALUDE_abc_sum_l3201_320165


namespace NUMINAMATH_CALUDE_inscribed_circle_radius_l3201_320153

theorem inscribed_circle_radius (a b c : ℝ) (ha : a = 5) (hb : b = 10) (hc : c = 20) :
  let r := (1 / a + 1 / b + 1 / c + 2 * Real.sqrt (1 / (a * b) + 1 / (a * c) + 1 / (b * c)))⁻¹
  r = 20 * (7 - Real.sqrt 10) / 39 :=
by sorry

end NUMINAMATH_CALUDE_inscribed_circle_radius_l3201_320153


namespace NUMINAMATH_CALUDE_triangle_inequality_l3201_320195

theorem triangle_inequality (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0)
  (triangle_cond : a + b > c ∧ b + c > a ∧ c + a > b) :
  |a/b + b/c + c/a - b/a - c/b - a/c| < 1 := by
sorry

end NUMINAMATH_CALUDE_triangle_inequality_l3201_320195


namespace NUMINAMATH_CALUDE_stewart_farm_sheep_count_l3201_320115

/-- Proves that the number of sheep is 32 given the farm conditions --/
theorem stewart_farm_sheep_count :
  ∀ (sheep horses : ℕ),
  (sheep : ℚ) / (horses : ℚ) = 4 / 7 →
  horses * 230 = 12880 →
  sheep = 32 := by
sorry

end NUMINAMATH_CALUDE_stewart_farm_sheep_count_l3201_320115


namespace NUMINAMATH_CALUDE_quadratic_equation_solution_l3201_320134

theorem quadratic_equation_solution (k : ℝ) : 
  (∀ x : ℝ, (k - 1) * x^2 + 3 * x + k^2 - 1 = 0 ↔ x = 0) ∧ 
  (k - 1 ≠ 0) → 
  k = -1 := by
sorry

end NUMINAMATH_CALUDE_quadratic_equation_solution_l3201_320134


namespace NUMINAMATH_CALUDE_theresa_kayla_ratio_l3201_320147

/-- The number of chocolate bars Theresa bought -/
def theresa_chocolate : ℕ := 12

/-- The number of soda cans Theresa bought -/
def theresa_soda : ℕ := 18

/-- The total number of items Kayla bought -/
def kayla_total : ℕ := 15

/-- The ratio of items Theresa bought to items Kayla bought -/
def item_ratio : ℚ := (theresa_chocolate + theresa_soda : ℚ) / kayla_total

theorem theresa_kayla_ratio : item_ratio = 2 := by sorry

end NUMINAMATH_CALUDE_theresa_kayla_ratio_l3201_320147


namespace NUMINAMATH_CALUDE_x_squared_minus_y_squared_l3201_320169

theorem x_squared_minus_y_squared (x y : ℚ) 
  (h1 : x + y = 8 / 15) 
  (h2 : x - y = 1 / 45) : 
  x^2 - y^2 = 8 / 675 := by
sorry

end NUMINAMATH_CALUDE_x_squared_minus_y_squared_l3201_320169


namespace NUMINAMATH_CALUDE_mechanic_hourly_rate_l3201_320182

/-- The mechanic's hourly rate calculation -/
theorem mechanic_hourly_rate :
  let hours_per_day : ℕ := 8
  let days_worked : ℕ := 14
  let parts_cost : ℕ := 2500
  let total_cost : ℕ := 9220
  let total_hours : ℕ := hours_per_day * days_worked
  let labor_cost : ℕ := total_cost - parts_cost
  labor_cost / total_hours = 60 := by sorry

end NUMINAMATH_CALUDE_mechanic_hourly_rate_l3201_320182


namespace NUMINAMATH_CALUDE_quadratic_roots_property_l3201_320127

theorem quadratic_roots_property (r s : ℝ) : 
  (∃ α β : ℝ, α + β = 10 ∧ α^2 - β^2 = 8 ∧ α^2 + r*α + s = 0 ∧ β^2 + r*β + s = 0) → 
  r = -10 := by
sorry

end NUMINAMATH_CALUDE_quadratic_roots_property_l3201_320127


namespace NUMINAMATH_CALUDE_right_triangle_from_trig_equality_l3201_320189

theorem right_triangle_from_trig_equality (α β : Real) (h : 0 < α ∧ 0 < β ∧ α + β < Real.pi) :
  (Real.cos α + Real.cos β = Real.sin α + Real.sin β) → ∃ γ : Real, α + β + γ = Real.pi ∧ γ = Real.pi / 2 :=
by
  sorry

end NUMINAMATH_CALUDE_right_triangle_from_trig_equality_l3201_320189


namespace NUMINAMATH_CALUDE_opposite_of_negative_fraction_l3201_320109

theorem opposite_of_negative_fraction (n : ℕ) (hn : n ≠ 0) :
  -(-(1 : ℚ) / n) = 1 / n :=
by sorry

end NUMINAMATH_CALUDE_opposite_of_negative_fraction_l3201_320109


namespace NUMINAMATH_CALUDE_earnings_difference_l3201_320159

/-- Mateo's hourly rate in dollars -/
def mateo_hourly_rate : ℕ := 20

/-- Sydney's daily rate in dollars -/
def sydney_daily_rate : ℕ := 400

/-- Number of hours in a week -/
def hours_per_week : ℕ := 24 * 7

/-- Number of days in a week -/
def days_per_week : ℕ := 7

/-- Mateo's total earnings for one week in dollars -/
def mateo_earnings : ℕ := mateo_hourly_rate * hours_per_week

/-- Sydney's total earnings for one week in dollars -/
def sydney_earnings : ℕ := sydney_daily_rate * days_per_week

theorem earnings_difference : mateo_earnings - sydney_earnings = 560 := by
  sorry

end NUMINAMATH_CALUDE_earnings_difference_l3201_320159


namespace NUMINAMATH_CALUDE_square_cut_perimeter_l3201_320198

/-- Given a square with side length 2a and a line y = 2x/3 cutting through it,
    the perimeter of one piece divided by a is 6 + (2√13 + 3√2)/3 -/
theorem square_cut_perimeter (a : ℝ) (a_pos : a > 0) :
  let square := {(x, y) | -a ≤ x ∧ x ≤ a ∧ -a ≤ y ∧ y ≤ a}
  let line := {(x, y) | y = (2/3) * x}
  let piece := {p ∈ square | p.2 ≤ (2/3) * p.1 ∨ (p.1 = a ∧ p.2 ≥ (2/3) * a) ∨ (p.1 = -a ∧ p.2 ≤ -(2/3) * a)}
  let perimeter := Real.sqrt ((2*a)^2 + ((4*a)/3)^2) + (4*a)/3 + 2*a + a * Real.sqrt 2
  perimeter / a = 6 + (2 * Real.sqrt 13 + 3 * Real.sqrt 2) / 3 := by
  sorry

end NUMINAMATH_CALUDE_square_cut_perimeter_l3201_320198


namespace NUMINAMATH_CALUDE_fraction_zero_implies_x_equals_five_l3201_320128

theorem fraction_zero_implies_x_equals_five (x : ℝ) : 
  (x^2 - 25) / (x + 5) = 0 ∧ x + 5 ≠ 0 → x = 5 := by
  sorry

end NUMINAMATH_CALUDE_fraction_zero_implies_x_equals_five_l3201_320128


namespace NUMINAMATH_CALUDE_sum_first_seven_primes_mod_eighth_prime_l3201_320191

def first_seven_primes : List Nat := [2, 3, 5, 7, 11, 13, 17]
def eighth_prime : Nat := 19

theorem sum_first_seven_primes_mod_eighth_prime : 
  (first_seven_primes.sum) % eighth_prime = 1 := by
  sorry

end NUMINAMATH_CALUDE_sum_first_seven_primes_mod_eighth_prime_l3201_320191


namespace NUMINAMATH_CALUDE_complex_number_problem_l3201_320132

theorem complex_number_problem (a b c : ℂ) 
  (h_a_real : a.im = 0)
  (h_sum : a + b + c = 4)
  (h_prod_sum : a * b + b * c + c * a = 6)
  (h_prod : a * b * c = 4) :
  a = 2 := by
sorry

end NUMINAMATH_CALUDE_complex_number_problem_l3201_320132


namespace NUMINAMATH_CALUDE_expression_evaluation_l3201_320108

theorem expression_evaluation :
  ∀ x y : ℝ,
  (abs x = 2) →
  (y = 1) →
  (x * y < 0) →
  3 * x^2 * y - 2 * x^2 - (x * y)^2 - 3 * x^2 * y - 4 * (x * y)^2 = -18 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l3201_320108


namespace NUMINAMATH_CALUDE_complex_on_imaginary_axis_l3201_320126

theorem complex_on_imaginary_axis (a : ℝ) : 
  (Complex.I * (a^2 - a - 2) : ℂ).re = 0 → a = 0 ∨ a = 2 :=
by sorry

end NUMINAMATH_CALUDE_complex_on_imaginary_axis_l3201_320126


namespace NUMINAMATH_CALUDE_house_rent_fraction_l3201_320110

theorem house_rent_fraction (salary : ℝ) 
  (food_fraction : ℝ) (conveyance_fraction : ℝ) (left_amount : ℝ) (food_conveyance_expense : ℝ)
  (h1 : food_fraction = 3/10)
  (h2 : conveyance_fraction = 1/8)
  (h3 : left_amount = 1400)
  (h4 : food_conveyance_expense = 3400)
  (h5 : food_fraction * salary + conveyance_fraction * salary = food_conveyance_expense)
  (h6 : salary - (food_fraction * salary + conveyance_fraction * salary + left_amount) = 
        salary * (1 - food_fraction - conveyance_fraction - left_amount / salary)) :
  1 - food_fraction - conveyance_fraction - left_amount / salary = 2/5 := by
sorry

end NUMINAMATH_CALUDE_house_rent_fraction_l3201_320110


namespace NUMINAMATH_CALUDE_ancient_chinese_gold_tax_l3201_320149

theorem ancient_chinese_gold_tax (x : ℚ) : 
  x > 0 ∧ 
  x/2 + x/2 * 1/3 + x/3 * 1/4 + x/4 * 1/5 + x/5 * 1/6 = 1 → 
  x/5 * 1/6 = 1/25 := by
  sorry

end NUMINAMATH_CALUDE_ancient_chinese_gold_tax_l3201_320149


namespace NUMINAMATH_CALUDE_commute_time_difference_l3201_320131

/-- Given a set of 5 commuting times (a, b, 8, 9, 10) with an average of 9 and a variance of 2, prove that |a-b| = 4 -/
theorem commute_time_difference (a b : ℝ) 
  (h_mean : (a + b + 8 + 9 + 10) / 5 = 9)
  (h_variance : ((a - 9)^2 + (b - 9)^2 + (8 - 9)^2 + (9 - 9)^2 + (10 - 9)^2) / 5 = 2) :
  |a - b| = 4 := by
sorry

end NUMINAMATH_CALUDE_commute_time_difference_l3201_320131


namespace NUMINAMATH_CALUDE_sum_base_6_100_equals_666_l3201_320168

def base_6_to_10 (n : ℕ) : ℕ := sorry

def sum_base_6 (n : ℕ) : ℕ := sorry

theorem sum_base_6_100_equals_666 :
  sum_base_6 (base_6_to_10 100) = 666 := by sorry

end NUMINAMATH_CALUDE_sum_base_6_100_equals_666_l3201_320168


namespace NUMINAMATH_CALUDE_p_range_l3201_320103

-- Define the function p(x)
def p (x : ℝ) : ℝ := x^4 + 6*x^2 + 9

-- State the theorem
theorem p_range :
  ∀ y : ℝ, (∃ x : ℝ, x ≥ 0 ∧ p x = y) ↔ y ≥ 9 :=
by sorry

end NUMINAMATH_CALUDE_p_range_l3201_320103


namespace NUMINAMATH_CALUDE_min_value_theorem_l3201_320181

theorem min_value_theorem (x y z : ℝ) (hx : x ≥ 0) (hy : y ≥ 0) (hz : z ≥ 0) :
  let A := Real.sqrt (x + 3) + Real.sqrt (y + 7) + Real.sqrt (z + 12)
  let B := Real.sqrt (x + 2) + Real.sqrt (y + 2) + Real.sqrt (z + 2)
  A^2 - B^2 ≥ 36 := by
  sorry

end NUMINAMATH_CALUDE_min_value_theorem_l3201_320181


namespace NUMINAMATH_CALUDE_range_of_m_for_line_intersecting_semicircle_l3201_320120

/-- A line intersecting a semicircle at exactly two points -/
structure LineIntersectingSemicircle where
  m : ℝ
  intersects_twice : ∃ (x₁ y₁ x₂ y₂ : ℝ),
    x₁ + y₁ = m ∧ y₁ = Real.sqrt (9 - x₁^2) ∧ y₁ ≥ 0 ∧
    x₂ + y₂ = m ∧ y₂ = Real.sqrt (9 - x₂^2) ∧ y₂ ≥ 0 ∧
    (x₁ ≠ x₂ ∨ y₁ ≠ y₂)

/-- The range of m values for lines intersecting the semicircle at exactly two points -/
theorem range_of_m_for_line_intersecting_semicircle (l : LineIntersectingSemicircle) :
  l.m ≥ 3 ∧ l.m < 3 * Real.sqrt 2 :=
sorry

end NUMINAMATH_CALUDE_range_of_m_for_line_intersecting_semicircle_l3201_320120


namespace NUMINAMATH_CALUDE_exchange_point_configuration_exists_multiple_configurations_exist_l3201_320163

/-- A planar graph representing a city map -/
structure CityMap where
  -- The number of edges (roads) in the map
  num_edges : ℕ
  -- The initial number of vertices (exchange points)
  initial_vertices : ℕ
  -- The number of faces in the planar graph (city parts)
  num_faces : ℕ
  -- Euler's formula for planar graphs
  euler_formula : num_faces = num_edges - initial_vertices + 2

/-- The configuration of exchange points in the city -/
structure ExchangePointConfig where
  -- The total number of exchange points after adding new ones
  total_points : ℕ
  -- The number of points in each face
  points_per_face : ℕ
  -- Condition that each face has exactly two points
  two_points_per_face : points_per_face = 2
  -- The total number of points is consistent with the number of faces
  total_points_condition : total_points = num_faces * points_per_face

/-- Theorem stating that it's possible to add three exchange points to satisfy the conditions -/
theorem exchange_point_configuration_exists (m : CityMap) (h : m.initial_vertices = 1) :
  ∃ (config : ExchangePointConfig), config.total_points = m.initial_vertices + 3 :=
sorry

/-- Theorem stating that multiple valid configurations exist -/
theorem multiple_configurations_exist (m : CityMap) (h : m.initial_vertices = 1) :
  ∃ (config1 config2 config3 config4 : ExchangePointConfig),
    config1 ≠ config2 ∧ config1 ≠ config3 ∧ config1 ≠ config4 ∧
    config2 ≠ config3 ∧ config2 ≠ config4 ∧ config3 ≠ config4 ∧
    (∀ c ∈ [config1, config2, config3, config4], c.total_points = m.initial_vertices + 3) :=
sorry

end NUMINAMATH_CALUDE_exchange_point_configuration_exists_multiple_configurations_exist_l3201_320163


namespace NUMINAMATH_CALUDE_mean_median_difference_l3201_320130

/-- Represents the score distribution in the math competition -/
structure ScoreDistribution where
  score72 : Float
  score84 : Float
  score86 : Float
  score92 : Float
  score98 : Float
  sum_to_one : score72 + score84 + score86 + score92 + score98 = 1

/-- Calculates the median score given the score distribution -/
def median (d : ScoreDistribution) : Float :=
  86

/-- Calculates the mean score given the score distribution -/
def mean (d : ScoreDistribution) : Float :=
  72 * d.score72 + 84 * d.score84 + 86 * d.score86 + 92 * d.score92 + 98 * d.score98

/-- The main theorem stating the difference between mean and median -/
theorem mean_median_difference (d : ScoreDistribution) 
  (h1 : d.score72 = 0.15)
  (h2 : d.score84 = 0.30)
  (h3 : d.score86 = 0.25)
  (h4 : d.score92 = 0.10) :
  mean d - median d = 0.3 := by
  sorry

#check mean_median_difference

end NUMINAMATH_CALUDE_mean_median_difference_l3201_320130


namespace NUMINAMATH_CALUDE_projection_theorem_l3201_320106

def vector_projection (u v : Fin 2 → ℚ) : Fin 2 → ℚ :=
  let dot_product := (u 0 * v 0 + u 1 * v 1)
  let norm_squared := (v 0 * v 0 + v 1 * v 1)
  fun i => (dot_product / norm_squared) * v i

def linear_transformation (v : Fin 2 → ℚ) : Fin 2 → ℚ :=
  vector_projection v (fun i => if i = 0 then 2 else -3)

theorem projection_theorem :
  let v : Fin 2 → ℚ := fun i => if i = 0 then 3 else -1
  let result := linear_transformation v
  result 0 = 18/13 ∧ result 1 = -27/13 := by
  sorry

end NUMINAMATH_CALUDE_projection_theorem_l3201_320106


namespace NUMINAMATH_CALUDE_line_equations_l3201_320122

/-- Given a line passing through (-b, c) that cuts a triangular region with area U from the second quadrant,
    this theorem states the equations of the inclined line and the horizontal line passing through its y-intercept. -/
theorem line_equations (b c U : ℝ) (h_b : b > 0) (h_c : c > 0) (h_U : U > 0) :
  ∃ (m k : ℝ),
    (∀ x y, y = m * x + k ↔ 2 * U * x - b^2 * y + 2 * U * b + c * b^2 = 0) ∧
    (k = 2 * U / b + c) := by
  sorry

end NUMINAMATH_CALUDE_line_equations_l3201_320122


namespace NUMINAMATH_CALUDE_inverse_proportion_problem_inverse_proportion_comparison_l3201_320137

theorem inverse_proportion_problem (k : ℝ) (h_k : k ≠ 0) :
  (∀ x : ℝ, x > 0 → x ≤ 1 → k / x > 3 * x) ↔ k = 3 :=
by sorry

theorem inverse_proportion_comparison (m : ℝ) (h_m : m ≠ 0) :
  (∀ x : ℝ, x > 0 → x ≤ 1 → 3 / x > m * x) ↔ (m < 0 ∨ (0 < m ∧ m < 3)) :=
by sorry

end NUMINAMATH_CALUDE_inverse_proportion_problem_inverse_proportion_comparison_l3201_320137


namespace NUMINAMATH_CALUDE_rectangle_to_square_l3201_320148

/-- Given a rectangle with perimeter 50 cm, prove that decreasing its length by 4 cm
    and increasing its width by 3 cm results in a square with side 12 cm and equal area. -/
theorem rectangle_to_square (L W : ℝ) : 
  L > 0 ∧ W > 0 ∧                    -- Length and width are positive
  2 * L + 2 * W = 50 ∧               -- Perimeter of original rectangle is 50 cm
  L * W = (L - 4) * (W + 3) →        -- Area remains constant after transformation
  L = 16 ∧ W = 9 ∧                   -- Original rectangle dimensions
  L - 4 = 12 ∧ W + 3 = 12            -- New shape is a square with side 12 cm
  := by sorry

end NUMINAMATH_CALUDE_rectangle_to_square_l3201_320148


namespace NUMINAMATH_CALUDE_fraction_simplification_l3201_320144

theorem fraction_simplification :
  (1 : ℝ) / (1 + Real.sqrt 3) * (1 / (1 - Real.sqrt 3)) = -(1 / 2) := by
  sorry

end NUMINAMATH_CALUDE_fraction_simplification_l3201_320144


namespace NUMINAMATH_CALUDE_symmetric_probability_l3201_320199

/-- Represents a standard die with faces labeled 1 to 6 -/
def StandardDie : Type := Fin 6

/-- The number of dice being rolled -/
def numDice : Nat := 9

/-- The sum we are comparing to -/
def targetSum : Nat := 14

/-- The sum we want to prove has the same probability as the target sum -/
def symmetricSum : Nat := 49

/-- Function to calculate the probability of a specific sum occurring when rolling n dice -/
noncomputable def probabilityOfSum (n : Nat) (sum : Nat) : ℚ := sorry

theorem symmetric_probability :
  probabilityOfSum numDice targetSum = probabilityOfSum numDice symmetricSum := by sorry

end NUMINAMATH_CALUDE_symmetric_probability_l3201_320199


namespace NUMINAMATH_CALUDE_max_value_expression_l3201_320180

theorem max_value_expression (x y z : ℝ) 
  (nonneg_x : x ≥ 0) (nonneg_y : y ≥ 0) (nonneg_z : z ≥ 0)
  (sum_eq_3 : x + y + z = 3)
  (x_ge_y : x ≥ y) (y_ge_z : y ≥ z) :
  (x^2 - x*y + y^2) * (x^2 - x*z + z^2) * (y^2 - y*z + z^2) ≤ 12 := by
  sorry

end NUMINAMATH_CALUDE_max_value_expression_l3201_320180


namespace NUMINAMATH_CALUDE_equation_represents_hyperbola_l3201_320166

/-- The equation x^2 - 16y^2 - 10x + 4y + 36 = 0 represents a hyperbola. -/
theorem equation_represents_hyperbola :
  ∃ (a b h k : ℝ) (A B : ℝ),
    A > 0 ∧ B > 0 ∧
    ∀ x y : ℝ,
      x^2 - 16*y^2 - 10*x + 4*y + 36 = 0 ↔
      ((x - h)^2 / A - (y - k)^2 / B = 1 ∨ (x - h)^2 / A - (y - k)^2 / B = -1) :=
by sorry

end NUMINAMATH_CALUDE_equation_represents_hyperbola_l3201_320166


namespace NUMINAMATH_CALUDE_quadratic_ratio_l3201_320193

/-- 
Given a quadratic expression x^2 + 1440x + 1600, which can be written in the form (x + d)^2 + e,
prove that e/d = -718.
-/
theorem quadratic_ratio (d e : ℝ) : 
  (∀ x, x^2 + 1440*x + 1600 = (x + d)^2 + e) → e/d = -718 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_ratio_l3201_320193


namespace NUMINAMATH_CALUDE_batsman_average_17th_inning_l3201_320104

def batsman_average (total_innings : ℕ) (last_inning_score : ℕ) (average_increase : ℚ) : ℚ :=
  (total_innings - 1 : ℚ) * (average_increase + last_inning_score / total_innings) + last_inning_score / total_innings

theorem batsman_average_17th_inning :
  batsman_average 17 92 3 = 44 := by sorry

end NUMINAMATH_CALUDE_batsman_average_17th_inning_l3201_320104


namespace NUMINAMATH_CALUDE_seven_b_value_l3201_320178

theorem seven_b_value (a b : ℚ) (h1 : 8 * a + 3 * b = 0) (h2 : b - 3 = a) : 7 * b = 168 / 11 := by
  sorry

end NUMINAMATH_CALUDE_seven_b_value_l3201_320178


namespace NUMINAMATH_CALUDE_triangle_exterior_angle_l3201_320177

theorem triangle_exterior_angle (A B C : Real) (h1 : A + B + C = 180) 
  (h2 : A = B) (h3 : A = 40 ∨ B = 40 ∨ C = 40) : 
  180 - C = 80 ∨ 180 - C = 140 := by
  sorry

end NUMINAMATH_CALUDE_triangle_exterior_angle_l3201_320177


namespace NUMINAMATH_CALUDE_union_complement_equals_less_than_three_l3201_320197

open Set

-- Define the sets A and B
def A : Set ℝ := {x | 1 < x ∧ x < 3}
def B : Set ℝ := {x | x > 2}

-- State the theorem
theorem union_complement_equals_less_than_three :
  A ∪ (univ \ B) = {x : ℝ | x < 3} := by sorry

end NUMINAMATH_CALUDE_union_complement_equals_less_than_three_l3201_320197


namespace NUMINAMATH_CALUDE_equation_simplification_l3201_320172

theorem equation_simplification (x : ℝ) : 
  (x / 0.3 = 1 + (1.2 - 0.3 * x) / 0.2) ↔ (10 * x / 3 = 1 + (12 - 3 * x) / 2) :=
by sorry

end NUMINAMATH_CALUDE_equation_simplification_l3201_320172


namespace NUMINAMATH_CALUDE_cube_root_of_3375_l3201_320179

theorem cube_root_of_3375 (x : ℝ) (h1 : x > 0) (h2 : x^3 = 3375) : x = 15 := by
  sorry

end NUMINAMATH_CALUDE_cube_root_of_3375_l3201_320179


namespace NUMINAMATH_CALUDE_ones_digit_of_14_power_power_of_4_cycle_exponent_even_ones_digit_14_power_14_7_power_7_l3201_320141

theorem ones_digit_of_14_power (n : ℕ) : (14^n) % 10 = (4^n) % 10 := by sorry

theorem power_of_4_cycle : ∀ n : ℕ, (4^n) % 10 = (4^(n % 2 + 1)) % 10 := by sorry

theorem exponent_even : (14 * (7^7)) % 2 = 0 := by sorry

theorem ones_digit_14_power_14_7_power_7 : (14^(14 * (7^7))) % 10 = 4 := by sorry

end NUMINAMATH_CALUDE_ones_digit_of_14_power_power_of_4_cycle_exponent_even_ones_digit_14_power_14_7_power_7_l3201_320141


namespace NUMINAMATH_CALUDE_yellow_shirt_pairs_l3201_320107

/-- Given a math contest with blue and yellow shirted students, this theorem proves
    the number of pairs where both students wear yellow shirts. -/
theorem yellow_shirt_pairs
  (total_students : ℕ)
  (blue_students : ℕ)
  (yellow_students : ℕ)
  (total_pairs : ℕ)
  (blue_blue_pairs : ℕ)
  (h1 : total_students = blue_students + yellow_students)
  (h2 : total_students = 150)
  (h3 : blue_students = 65)
  (h4 : yellow_students = 85)
  (h5 : total_pairs = 75)
  (h6 : blue_blue_pairs = 30) :
  ∃ (yellow_yellow_pairs : ℕ), yellow_yellow_pairs = 40 ∧
  yellow_yellow_pairs + blue_blue_pairs + (total_students - 2 * blue_blue_pairs - 2 * yellow_yellow_pairs) / 2 = total_pairs :=
by sorry

end NUMINAMATH_CALUDE_yellow_shirt_pairs_l3201_320107


namespace NUMINAMATH_CALUDE_ellipse_properties_and_max_area_l3201_320123

noncomputable def ellipse_equation (a b : ℝ) (x y : ℝ) : Prop :=
  x^2 / a^2 + y^2 / b^2 = 1

noncomputable def eccentricity (c a : ℝ) : ℝ :=
  c / a

noncomputable def distance (x y : ℝ) : ℝ :=
  Real.sqrt (x^2 + y^2)

theorem ellipse_properties_and_max_area 
  (a b c : ℝ) 
  (h1 : a > b) (h2 : b > 0) (h3 : c > 0)
  (h4 : eccentricity c a = Real.sqrt 3 / 2)
  (h5 : ellipse_equation a b c (b^2/a))
  (h6 : distance c (b^2/a) = Real.sqrt 13 / 2) :
  (∃ (x y : ℝ), ellipse_equation 2 1 x y) ∧
  (∃ (S : ℝ), S = 4 ∧ 
    ∀ (m : ℝ), abs m < Real.sqrt 2 → 
      2 * Real.sqrt (m^2 * (8 - 4 * m^2)) ≤ S) := by
sorry

end NUMINAMATH_CALUDE_ellipse_properties_and_max_area_l3201_320123


namespace NUMINAMATH_CALUDE_modular_inverse_of_5_mod_31_l3201_320119

theorem modular_inverse_of_5_mod_31 : ∃ x : ℕ, x ≤ 30 ∧ (5 * x) % 31 = 1 :=
by
  use 25
  sorry

end NUMINAMATH_CALUDE_modular_inverse_of_5_mod_31_l3201_320119


namespace NUMINAMATH_CALUDE_quadratic_inequality_equivalence_l3201_320184

theorem quadratic_inequality_equivalence (x : ℝ) :
  x^2 - 9*x + 14 < 0 ↔ 2 < x ∧ x < 7 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_equivalence_l3201_320184
