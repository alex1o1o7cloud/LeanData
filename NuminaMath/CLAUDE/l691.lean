import Mathlib

namespace NUMINAMATH_CALUDE_exists_m_divisible_by_1988_l691_69157

def f (x : ℕ) : ℕ := 3 * x + 2

def iterate (n : ℕ) (f : ℕ → ℕ) (x : ℕ) : ℕ :=
  match n with
  | 0 => x
  | n + 1 => f (iterate n f x)

theorem exists_m_divisible_by_1988 :
  ∃ m : ℕ, (1988 : ℕ) ∣ (iterate 100 f m) := by
  sorry

end NUMINAMATH_CALUDE_exists_m_divisible_by_1988_l691_69157


namespace NUMINAMATH_CALUDE_simplify_expression_l691_69105

theorem simplify_expression (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) :
  4 * x^4 * y^2 / (-2 * x * y) = -2 * x^3 * y :=
by sorry

end NUMINAMATH_CALUDE_simplify_expression_l691_69105


namespace NUMINAMATH_CALUDE_sin_cos_symmetry_l691_69152

open Real

theorem sin_cos_symmetry :
  ∃ (k : ℤ), (∀ x : ℝ, sin (2 * x - π / 6) = sin (π / 2 - (2 * x - π / 6))) ∧
             (∀ x : ℝ, cos (x - π / 3) = cos (π - (x - π / 3))) ∧
  ¬ ∃ (c : ℝ), (∀ x : ℝ, sin (2 * (x + c) - π / 6) = -sin (2 * (x - c) - π / 6)) ∧
                (∀ x : ℝ, cos ((x + c) - π / 3) = cos ((x - c) - π / 3)) :=
by sorry

end NUMINAMATH_CALUDE_sin_cos_symmetry_l691_69152


namespace NUMINAMATH_CALUDE_always_real_roots_roots_difference_condition_l691_69117

-- Define the quadratic equation
def quadratic_equation (m x : ℝ) : ℝ := m * x^2 - (3*m - 1) * x + (2*m - 2)

-- Theorem 1: The equation always has real roots
theorem always_real_roots (m : ℝ) : 
  ∃ x : ℝ, quadratic_equation m x = 0 :=
sorry

-- Theorem 2: If the difference between the roots is 2, then m = 1 or m = -1/3
theorem roots_difference_condition (m : ℝ) :
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ 
    quadratic_equation m x₁ = 0 ∧ 
    quadratic_equation m x₂ = 0 ∧ 
    |x₁ - x₂| = 2) →
  (m = 1 ∨ m = -1/3) :=
sorry

end NUMINAMATH_CALUDE_always_real_roots_roots_difference_condition_l691_69117


namespace NUMINAMATH_CALUDE_inequality_proof_l691_69173

theorem inequality_proof (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : 1/a + 1/b = 1) :
  ∀ n : ℕ, (a + b)^n - a^n - b^n ≥ 2^(2*n) - 2^(n+1) :=
by sorry

end NUMINAMATH_CALUDE_inequality_proof_l691_69173


namespace NUMINAMATH_CALUDE_largest_n_satisfying_conditions_l691_69183

theorem largest_n_satisfying_conditions : 
  ∃ (m : ℤ), 365^2 = (m+1)^3 - m^3 ∧
  ∃ (a : ℤ), 2*365 + 111 = a^2 ∧
  ∀ (n : ℤ), n > 365 → 
    (∀ (m : ℤ), n^2 ≠ (m+1)^3 - m^3 ∨ 
    ∀ (a : ℤ), 2*n + 111 ≠ a^2) :=
by sorry

end NUMINAMATH_CALUDE_largest_n_satisfying_conditions_l691_69183


namespace NUMINAMATH_CALUDE_astronomers_use_analogical_reasoning_l691_69169

/-- Represents a celestial body in the solar system -/
structure CelestialBody where
  name : String
  hasLife : Bool

/-- Represents a type of reasoning -/
inductive ReasoningType
  | Inductive
  | Analogical
  | Deductive
  | ProofByContradiction

/-- Determines if two celestial bodies are similar -/
def areSimilar (a b : CelestialBody) : Bool := sorry

/-- Represents the astronomers' reasoning process -/
def astronomersReasoning (earth mars : CelestialBody) : ReasoningType :=
  if areSimilar earth mars ∧ earth.hasLife then
    ReasoningType.Analogical
  else
    sorry

/-- Theorem stating that the astronomers' reasoning is analogical -/
theorem astronomers_use_analogical_reasoning (earth mars : CelestialBody) 
  (h1 : areSimilar earth mars = true)
  (h2 : earth.hasLife = true) :
  astronomersReasoning earth mars = ReasoningType.Analogical := by
  sorry

end NUMINAMATH_CALUDE_astronomers_use_analogical_reasoning_l691_69169


namespace NUMINAMATH_CALUDE_inclination_angle_sqrt3x_plus_y_minus2_l691_69100

/-- The inclination angle of a line given by the equation √3x + y - 2 = 0 is 120°. -/
theorem inclination_angle_sqrt3x_plus_y_minus2 :
  let line : ℝ → ℝ → Prop := λ x y ↦ Real.sqrt 3 * x + y - 2 = 0
  ∃ α : ℝ, α = 120 * (π / 180) ∧ 
    ∀ x y : ℝ, line x y → Real.tan α = -Real.sqrt 3 :=
by sorry

end NUMINAMATH_CALUDE_inclination_angle_sqrt3x_plus_y_minus2_l691_69100


namespace NUMINAMATH_CALUDE_sum_of_ages_sum_of_ages_proof_l691_69140

theorem sum_of_ages : ℕ → ℕ → Prop :=
  fun john_age father_age =>
    (john_age = 15) →
    (father_age = 2 * john_age + 32) →
    (john_age + father_age = 77)

-- Proof
theorem sum_of_ages_proof : sum_of_ages 15 62 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_ages_sum_of_ages_proof_l691_69140


namespace NUMINAMATH_CALUDE_total_pumpkins_sold_l691_69153

/-- Represents the price of a jumbo pumpkin in dollars -/
def jumbo_price : ℚ := 9

/-- Represents the price of a regular pumpkin in dollars -/
def regular_price : ℚ := 4

/-- Represents the total amount collected in dollars -/
def total_collected : ℚ := 395

/-- Represents the number of regular pumpkins sold -/
def regular_sold : ℕ := 65

/-- Theorem stating that the total number of pumpkins sold is 80 -/
theorem total_pumpkins_sold : 
  ∃ (jumbo_sold : ℕ), 
    (jumbo_price * jumbo_sold + regular_price * regular_sold = total_collected) ∧
    (jumbo_sold + regular_sold = 80) := by
  sorry

end NUMINAMATH_CALUDE_total_pumpkins_sold_l691_69153


namespace NUMINAMATH_CALUDE_circle_area_with_diameter_10_l691_69111

theorem circle_area_with_diameter_10 (π : ℝ) :
  let diameter : ℝ := 10
  let radius : ℝ := diameter / 2
  let area : ℝ := π * radius ^ 2
  area = 25 * π :=
by sorry

end NUMINAMATH_CALUDE_circle_area_with_diameter_10_l691_69111


namespace NUMINAMATH_CALUDE_x_intercept_of_parallel_lines_l691_69175

/-- Given two parallel lines, prove that the x-intercept of one line is -1 -/
theorem x_intercept_of_parallel_lines (m : ℝ) :
  (∀ x y, y + m * (x + 1) = 0 ↔ m * y - (2 * m + 1) * x = 1) →
  ∃ x, x + m * (x + 1) = 0 ∧ x = -1 :=
by sorry

end NUMINAMATH_CALUDE_x_intercept_of_parallel_lines_l691_69175


namespace NUMINAMATH_CALUDE_parallel_vectors_k_value_l691_69150

def vector_parallel (a b : ℝ × ℝ) : Prop :=
  ∃ (t : ℝ), t ≠ 0 ∧ a.1 = t * b.1 ∧ a.2 = t * b.2

theorem parallel_vectors_k_value (k : ℝ) :
  vector_parallel (1, k) (2, 2) → k = 1 := by
  sorry

end NUMINAMATH_CALUDE_parallel_vectors_k_value_l691_69150


namespace NUMINAMATH_CALUDE_computer_table_cost_price_l691_69197

/-- Proves that the cost price of a computer table is 6672 when the selling price is 8340 and the markup is 25% -/
theorem computer_table_cost_price (selling_price : ℝ) (markup_percentage : ℝ) 
  (h1 : selling_price = 8340)
  (h2 : markup_percentage = 25) : 
  selling_price / (1 + markup_percentage / 100) = 6672 := by
  sorry

end NUMINAMATH_CALUDE_computer_table_cost_price_l691_69197


namespace NUMINAMATH_CALUDE_kindergarten_craft_problem_l691_69177

theorem kindergarten_craft_problem :
  ∃ (scissors glue_sticks crayons : ℕ),
    scissors + glue_sticks + crayons = 26 ∧
    2 * scissors + 3 * glue_sticks + 4 * crayons = 24 := by
  sorry

end NUMINAMATH_CALUDE_kindergarten_craft_problem_l691_69177


namespace NUMINAMATH_CALUDE_cube_sum_sqrt_l691_69185

theorem cube_sum_sqrt : Real.sqrt (4^3 + 4^3 + 4^3) = 8 * Real.sqrt 3 := by sorry

end NUMINAMATH_CALUDE_cube_sum_sqrt_l691_69185


namespace NUMINAMATH_CALUDE_f_positive_m_range_l691_69108

-- Define the function f
def f (x : ℝ) : ℝ := |2*x - 1| - |x + 2|

-- Theorem for the solution set of f(x) > 0
theorem f_positive (x : ℝ) : f x > 0 ↔ x < -1/3 ∨ x > 3 := by sorry

-- Theorem for the range of m
theorem m_range (m : ℝ) : 
  (∃ x₀ : ℝ, f x₀ + 2*m^2 < 4*m) ↔ -1/2 < m ∧ m < 5/2 := by sorry

end NUMINAMATH_CALUDE_f_positive_m_range_l691_69108


namespace NUMINAMATH_CALUDE_bus_stop_time_l691_69181

/-- Proves that a bus with given speeds stops for 4 minutes per hour -/
theorem bus_stop_time (speed_without_stops : ℝ) (speed_with_stops : ℝ) 
  (h1 : speed_without_stops = 90) 
  (h2 : speed_with_stops = 84) : 
  (speed_without_stops - speed_with_stops) / speed_without_stops * 60 = 4 := by
  sorry

end NUMINAMATH_CALUDE_bus_stop_time_l691_69181


namespace NUMINAMATH_CALUDE_largest_integer_less_than_150_over_7_l691_69124

theorem largest_integer_less_than_150_over_7 : 
  (∀ n : ℤ, 7 * n < 150 → n ≤ 21) ∧ (7 * 21 < 150) := by sorry

end NUMINAMATH_CALUDE_largest_integer_less_than_150_over_7_l691_69124


namespace NUMINAMATH_CALUDE_trig_identity_l691_69184

theorem trig_identity (α : Real) (h : Real.sin α + Real.sin α ^ 2 = 1) :
  Real.cos α ^ 2 + Real.cos α ^ 4 = 1 := by
  sorry

end NUMINAMATH_CALUDE_trig_identity_l691_69184


namespace NUMINAMATH_CALUDE_root_product_equality_l691_69118

theorem root_product_equality (p q : ℝ) (α β γ δ : ℂ) 
  (h1 : α^2 + p*α + 1 = 0)
  (h2 : β^2 + p*β + 1 = 0)
  (h3 : γ^2 + q*γ + 1 = 0)
  (h4 : δ^2 + q*δ + 1 = 0) :
  (α - γ)*(β - γ)*(α + δ)*(β + δ) = q^2 - p^2 := by
  sorry


end NUMINAMATH_CALUDE_root_product_equality_l691_69118


namespace NUMINAMATH_CALUDE_coin_flip_expected_value_is_two_thirds_l691_69121

def coin_flip_expected_value : ℚ :=
  let p_heads : ℚ := 1/2
  let p_tails : ℚ := 1/3
  let p_edge : ℚ := 1/6
  let win_heads : ℚ := 1
  let win_tails : ℚ := 3
  let win_edge : ℚ := -5
  p_heads * win_heads + p_tails * win_tails + p_edge * win_edge

theorem coin_flip_expected_value_is_two_thirds :
  coin_flip_expected_value = 2/3 := by
  sorry

end NUMINAMATH_CALUDE_coin_flip_expected_value_is_two_thirds_l691_69121


namespace NUMINAMATH_CALUDE_log_equation_solution_l691_69159

theorem log_equation_solution (p q : ℝ) (hp : p > 0) (hq : q > 0) (hq1 : q ≠ 1) :
  Real.log p + Real.log (q^2) = Real.log (p + q^2) ↔ p = q^2 / (q^2 - 1) :=
by sorry

end NUMINAMATH_CALUDE_log_equation_solution_l691_69159


namespace NUMINAMATH_CALUDE_vlad_score_in_competition_l691_69103

/-- A video game competition between two players -/
structure VideoGameCompetition where
  rounds : ℕ
  points_per_win : ℕ
  taro_score : ℕ

/-- Calculate Vlad's score in the video game competition -/
def vlad_score (game : VideoGameCompetition) : ℕ :=
  game.rounds * game.points_per_win - game.taro_score

/-- Theorem stating Vlad's score in the specific competition described in the problem -/
theorem vlad_score_in_competition :
  let game : VideoGameCompetition := {
    rounds := 30,
    points_per_win := 5,
    taro_score := 3 * (30 * 5) / 5 - 4
  }
  vlad_score game = 64 := by sorry

end NUMINAMATH_CALUDE_vlad_score_in_competition_l691_69103


namespace NUMINAMATH_CALUDE_equipment_production_theorem_l691_69182

theorem equipment_production_theorem
  (total_production : ℕ)
  (sample_size : ℕ)
  (sample_from_A : ℕ)
  (h1 : total_production = 4800)
  (h2 : sample_size = 80)
  (h3 : sample_from_A = 50)
  : (total_production - (sample_from_A * (total_production / sample_size))) = 1800 :=
by sorry

end NUMINAMATH_CALUDE_equipment_production_theorem_l691_69182


namespace NUMINAMATH_CALUDE_sqrt_two_is_quadratic_radical_l691_69113

-- Define what a quadratic radical is
def is_quadratic_radical (x : ℝ) : Prop :=
  ∃ (y : ℝ), x = Real.sqrt y ∧ ¬ (∃ (n : ℤ), x = n)

-- Theorem statement
theorem sqrt_two_is_quadratic_radical : 
  is_quadratic_radical (Real.sqrt 2) :=
sorry

end NUMINAMATH_CALUDE_sqrt_two_is_quadratic_radical_l691_69113


namespace NUMINAMATH_CALUDE_tire_change_problem_l691_69112

theorem tire_change_problem (total_cars : ℕ) (tires_per_car : ℕ) (half_change_cars : ℕ) (tires_left : ℕ) : 
  total_cars = 10 →
  tires_per_car = 4 →
  half_change_cars = 2 →
  tires_left = 20 →
  ∃ (no_change_cars : ℕ), 
    no_change_cars = total_cars - (half_change_cars + (total_cars * tires_per_car - tires_left - half_change_cars * (tires_per_car / 2)) / tires_per_car) ∧
    no_change_cars = 4 :=
by sorry

end NUMINAMATH_CALUDE_tire_change_problem_l691_69112


namespace NUMINAMATH_CALUDE_fibonacci_polynomial_property_l691_69187

def fibonacci : ℕ → ℕ
  | 0 => 1
  | 1 => 1
  | (n + 2) => fibonacci (n + 1) + fibonacci n

theorem fibonacci_polynomial_property (n : ℕ) (P : ℕ → ℕ) :
  (∀ k ∈ Finset.range (n + 1), P (k + n + 2) = fibonacci (k + n + 2)) →
  P (2 * n + 3) = fibonacci (2 * n + 3) - 1 := by
  sorry

end NUMINAMATH_CALUDE_fibonacci_polynomial_property_l691_69187


namespace NUMINAMATH_CALUDE_pet_store_cages_l691_69158

/-- Given a pet store scenario with puppies and cages, calculate the number of cages used. -/
theorem pet_store_cages (initial_puppies sold_puppies puppies_per_cage : ℕ) 
  (h1 : initial_puppies = 13)
  (h2 : sold_puppies = 7)
  (h3 : puppies_per_cage = 2)
  : (initial_puppies - sold_puppies) / puppies_per_cage = 3 := by
  sorry

end NUMINAMATH_CALUDE_pet_store_cages_l691_69158


namespace NUMINAMATH_CALUDE_min_additional_coins_l691_69160

/-- The number of friends Alex has -/
def num_friends : ℕ := 12

/-- The initial number of coins Alex has -/
def initial_coins : ℕ := 63

/-- The sum of the first n natural numbers -/
def sum_first_n (n : ℕ) : ℕ := n * (n + 1) / 2

/-- The minimum number of additional coins needed -/
def additional_coins_needed : ℕ := sum_first_n num_friends - initial_coins

theorem min_additional_coins :
  additional_coins_needed = 15 :=
sorry

end NUMINAMATH_CALUDE_min_additional_coins_l691_69160


namespace NUMINAMATH_CALUDE_red_balls_count_l691_69134

theorem red_balls_count (total_balls : ℕ) (red_balls : ℕ) (white_balls : ℕ) :
  total_balls = red_balls + white_balls →
  white_balls = 4 →
  (red_balls : ℚ) / total_balls = 6 / 10 →
  red_balls = 6 := by
  sorry

end NUMINAMATH_CALUDE_red_balls_count_l691_69134


namespace NUMINAMATH_CALUDE_amber_amethyst_ratio_l691_69148

/-- Given a necklace with 40 beads, 7 amethyst beads, and 19 turquoise beads,
    prove that the ratio of amber beads to amethyst beads is 2:1. -/
theorem amber_amethyst_ratio (total : ℕ) (amethyst : ℕ) (turquoise : ℕ) 
  (h1 : total = 40)
  (h2 : amethyst = 7)
  (h3 : turquoise = 19) :
  (total - amethyst - turquoise) / amethyst = 2 := by
  sorry

end NUMINAMATH_CALUDE_amber_amethyst_ratio_l691_69148


namespace NUMINAMATH_CALUDE_tan_X_equals_four_l691_69133

-- Define the triangle XYZ
structure Triangle (X Y Z : ℝ) where
  -- Angle Y is 90°
  right_angle : Y = 90
  -- Length of side YZ
  yz_length : Z - Y = 4
  -- Length of side XZ
  xz_length : Z - X = Real.sqrt 17

-- Theorem statement
theorem tan_X_equals_four {X Y Z : ℝ} (t : Triangle X Y Z) : Real.tan X = 4 := by
  sorry

end NUMINAMATH_CALUDE_tan_X_equals_four_l691_69133


namespace NUMINAMATH_CALUDE_felix_axe_sharpening_cost_l691_69138

/-- Calculates the total cost of axe sharpening given the number of trees chopped,
    trees per sharpening, and cost per sharpening. -/
def axeSharpeningCost (treesChopped : ℕ) (treesPerSharpening : ℕ) (costPerSharpening : ℕ) : ℕ :=
  ((treesChopped - 1) / treesPerSharpening + 1) * costPerSharpening

/-- Proves that given the conditions, the total cost of axe sharpening is $35. -/
theorem felix_axe_sharpening_cost :
  ∀ (treesChopped : ℕ),
    treesChopped ≥ 91 →
    axeSharpeningCost treesChopped 13 5 = 35 := by
  sorry

end NUMINAMATH_CALUDE_felix_axe_sharpening_cost_l691_69138


namespace NUMINAMATH_CALUDE_total_shoes_l691_69164

def scott_shoes : ℕ := 7

def anthony_shoes : ℕ := 3 * scott_shoes

def jim_shoes : ℕ := anthony_shoes - 2

def melissa_shoes : ℕ := jim_shoes / 2

def tim_shoes : ℕ := (anthony_shoes + melissa_shoes) / 2

theorem total_shoes : scott_shoes + anthony_shoes + jim_shoes + melissa_shoes + tim_shoes = 71 := by
  sorry

end NUMINAMATH_CALUDE_total_shoes_l691_69164


namespace NUMINAMATH_CALUDE_cricket_game_initial_overs_l691_69142

/-- Prove that the number of overs played initially is 10, given the conditions of the cricket game. -/
theorem cricket_game_initial_overs (target : ℝ) (initial_rate : ℝ) (required_rate : ℝ) (remaining_overs : ℝ) :
  target = 282 →
  initial_rate = 3.2 →
  required_rate = 6.25 →
  remaining_overs = 40 →
  ∃ (initial_overs : ℝ), initial_overs = 10 ∧ 
    target = initial_rate * initial_overs + required_rate * remaining_overs :=
by
  sorry

end NUMINAMATH_CALUDE_cricket_game_initial_overs_l691_69142


namespace NUMINAMATH_CALUDE_inequality_holds_iff_a_in_range_l691_69170

theorem inequality_holds_iff_a_in_range :
  ∀ a : ℝ, (∀ x : ℝ, x ≤ 1 → (1 + 2^x + 4^x * a) / (a^2 - a + 1) > 0) ↔ a > -3/4 :=
by sorry

end NUMINAMATH_CALUDE_inequality_holds_iff_a_in_range_l691_69170


namespace NUMINAMATH_CALUDE_parabola_vertex_l691_69102

/-- The parabola defined by y = -x^2 + 3 has its vertex at (0, 3) -/
theorem parabola_vertex (x y : ℝ) : 
  y = -x^2 + 3 → (0, 3) = (x, y) ∨ ∃ z, y < -z^2 + 3 := by
  sorry

end NUMINAMATH_CALUDE_parabola_vertex_l691_69102


namespace NUMINAMATH_CALUDE_pure_imaginary_complex_number_l691_69151

theorem pure_imaginary_complex_number (a : ℝ) : 
  (∃ b : ℝ, a - (10 : ℂ) / (3 - Complex.I) = b * Complex.I) → a = 3 := by
  sorry

end NUMINAMATH_CALUDE_pure_imaginary_complex_number_l691_69151


namespace NUMINAMATH_CALUDE_mode_most_relevant_for_restocking_l691_69116

/-- Represents a shoe size -/
def ShoeSize := ℕ

/-- Represents the inventory of shoes -/
def Inventory := List ShoeSize

/-- A statistical measure for shoe sizes -/
class StatisticalMeasure where
  measure : Inventory → ℝ

/-- Variance of shoe sizes -/
def variance : StatisticalMeasure := sorry

/-- Mode of shoe sizes -/
def mode : StatisticalMeasure := sorry

/-- Median of shoe sizes -/
def median : StatisticalMeasure := sorry

/-- Mean of shoe sizes -/
def mean : StatisticalMeasure := sorry

/-- Relevance of a statistical measure for restocking -/
def relevance (m : StatisticalMeasure) : ℝ := sorry

/-- The shoe store -/
structure ShoeStore where
  inventory : Inventory

/-- Theorem: Mode is the most relevant statistical measure for restocking -/
theorem mode_most_relevant_for_restocking (store : ShoeStore) :
  ∀ m : StatisticalMeasure, m ≠ mode → relevance mode > relevance m :=
sorry

end NUMINAMATH_CALUDE_mode_most_relevant_for_restocking_l691_69116


namespace NUMINAMATH_CALUDE_cos_angle_F₁PF₂_l691_69127

-- Define the ellipse and hyperbola
def is_on_ellipse (x y : ℝ) : Prop := x^2/6 + y^2/2 = 1
def is_on_hyperbola (x y : ℝ) : Prop := x^2/3 - y^2 = 1

-- Define the common foci
def F₁ : ℝ × ℝ := (-2, 0)
def F₂ : ℝ × ℝ := (2, 0)

-- Define the common point P
structure CommonPoint where
  x : ℝ
  y : ℝ
  on_ellipse : is_on_ellipse x y
  on_hyperbola : is_on_hyperbola x y

-- Theorem statement
theorem cos_angle_F₁PF₂ (P : CommonPoint) : 
  let PF₁ := (F₁.1 - P.x, F₁.2 - P.y)
  let PF₂ := (F₂.1 - P.x, F₂.2 - P.y)
  let dot_product := PF₁.1 * PF₂.1 + PF₁.2 * PF₂.2
  let magnitude_PF₁ := Real.sqrt (PF₁.1^2 + PF₁.2^2)
  let magnitude_PF₂ := Real.sqrt (PF₂.1^2 + PF₂.2^2)
  dot_product / (magnitude_PF₁ * magnitude_PF₂) = 1/3 :=
by sorry

end NUMINAMATH_CALUDE_cos_angle_F₁PF₂_l691_69127


namespace NUMINAMATH_CALUDE_exponent_multiplication_l691_69199

theorem exponent_multiplication (a : ℝ) : a^3 * a^3 = a^6 := by
  sorry

end NUMINAMATH_CALUDE_exponent_multiplication_l691_69199


namespace NUMINAMATH_CALUDE_cube_root_8000_simplification_l691_69110

theorem cube_root_8000_simplification :
  ∃ (a b : ℕ+), (a : ℝ) * (b : ℝ)^(1/3) = 8000^(1/3) ∧
                a = 20 ∧ b = 1 ∧
                ∀ (c d : ℕ+), (c : ℝ) * (d : ℝ)^(1/3) = 8000^(1/3) → d ≥ b :=
by sorry

end NUMINAMATH_CALUDE_cube_root_8000_simplification_l691_69110


namespace NUMINAMATH_CALUDE_A₁_Aₒ₂_independent_l691_69131

/-- A bag containing black and white balls -/
structure Bag where
  black : ℕ
  white : ℕ

/-- An event in the probability space of drawing balls from the bag -/
structure Event (bag : Bag) where
  prob : ℝ
  nonneg : 0 ≤ prob
  le_one : prob ≤ 1

/-- Drawing a ball from the bag with replacement -/
def draw (bag : Bag) : Event bag := sorry

/-- The event of drawing a black ball -/
def black_ball (bag : Bag) : Event bag := sorry

/-- The event of drawing a white ball -/
def white_ball (bag : Bag) : Event bag := sorry

/-- The probability of an event -/
def P (bag : Bag) (e : Event bag) : ℝ := e.prob

/-- Two events are independent if the probability of their intersection
    is equal to the product of their individual probabilities -/
def independent (bag : Bag) (e1 e2 : Event bag) : Prop :=
  P bag (draw bag) = P bag e1 * P bag e2

/-- A₁: The event of drawing a black ball on the first draw -/
def A₁ (bag : Bag) : Event bag := black_ball bag

/-- A₂: The event of drawing a black ball on the second draw -/
def A₂ (bag : Bag) : Event bag := black_ball bag

/-- Aₒ₂: The complement of A₂ (drawing a white ball on the second draw) -/
def Aₒ₂ (bag : Bag) : Event bag := white_ball bag

/-- Theorem: A₁ and Aₒ₂ are independent events when drawing with replacement -/
theorem A₁_Aₒ₂_independent (bag : Bag) : independent bag (A₁ bag) (Aₒ₂ bag) := by
  sorry

end NUMINAMATH_CALUDE_A₁_Aₒ₂_independent_l691_69131


namespace NUMINAMATH_CALUDE_baker_cakes_sold_l691_69145

theorem baker_cakes_sold (initial_cakes : ℕ) (additional_cakes : ℕ) (remaining_cakes : ℕ) : 
  initial_cakes = 110 →
  additional_cakes = 76 →
  remaining_cakes = 111 →
  initial_cakes + additional_cakes - remaining_cakes = 75 := by
sorry

end NUMINAMATH_CALUDE_baker_cakes_sold_l691_69145


namespace NUMINAMATH_CALUDE_line_translation_down_5_l691_69129

/-- Represents a line in 2D space -/
structure Line where
  slope : ℚ
  intercept : ℚ

/-- Translates a line vertically by a given amount -/
def translateLine (l : Line) (dy : ℚ) : Line :=
  { slope := l.slope, intercept := l.intercept + dy }

theorem line_translation_down_5 :
  let original_line := { slope := -1/2, intercept := 2 : Line }
  let translated_line := translateLine original_line (-5)
  translated_line = { slope := -1/2, intercept := -3 : Line } := by
  sorry

end NUMINAMATH_CALUDE_line_translation_down_5_l691_69129


namespace NUMINAMATH_CALUDE_integral_sin4_cos4_3x_l691_69161

theorem integral_sin4_cos4_3x (x : ℝ) : 
  ∫ x in (0 : ℝ)..(2 * Real.pi), (Real.sin (3 * x))^4 * (Real.cos (3 * x))^4 = (3 * Real.pi) / 64 := by
  sorry

end NUMINAMATH_CALUDE_integral_sin4_cos4_3x_l691_69161


namespace NUMINAMATH_CALUDE_inequality_proof_l691_69107

theorem inequality_proof (x y : ℝ) (hx : x > 0) (hy : y > 0) (hxy : x > y) :
  2 * x + 1 / (x^2 - 2*x*y + y^2) ≥ 2 * y + 3 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l691_69107


namespace NUMINAMATH_CALUDE_existence_of_solution_specific_solution_valid_l691_69189

theorem existence_of_solution :
  ∃ (n m : ℝ), n ≠ 0 ∧ m ≠ 0 ∧ (n * 5^n)^n = m * 5^9 :=
by sorry

theorem specific_solution_valid :
  let n : ℝ := 3
  let m : ℝ := 27
  (n * 5^n)^n = m * 5^9 :=
by sorry

end NUMINAMATH_CALUDE_existence_of_solution_specific_solution_valid_l691_69189


namespace NUMINAMATH_CALUDE_isosceles_trapezoid_side_length_l691_69147

/-- An isosceles trapezoid with given properties -/
structure IsoscelesTrapezoid where
  base1 : ℝ
  base2 : ℝ
  area : ℝ
  side : ℝ

/-- The theorem stating the relationship between the trapezoid's properties -/
theorem isosceles_trapezoid_side_length 
  (t : IsoscelesTrapezoid) 
  (h1 : t.base1 = 6) 
  (h2 : t.base2 = 12) 
  (h3 : t.area = 36) : 
  t.side = 5 := by
  sorry

#check isosceles_trapezoid_side_length

end NUMINAMATH_CALUDE_isosceles_trapezoid_side_length_l691_69147


namespace NUMINAMATH_CALUDE_problem_solution_l691_69162

theorem problem_solution : (-1)^2022 + |(-2)^3 + (-3)^2| - (-1/4 + 1/6) * (-24) = 0 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l691_69162


namespace NUMINAMATH_CALUDE_number_operations_l691_69166

theorem number_operations (n y : ℝ) : ((2 * n + y) / 2) - n = y / 2 := by
  sorry

end NUMINAMATH_CALUDE_number_operations_l691_69166


namespace NUMINAMATH_CALUDE_binary_1101101000_to_octal_1550_l691_69130

def binary_to_decimal (b : List Bool) : Nat :=
  b.foldl (fun acc x => 2 * acc + if x then 1 else 0) 0

def decimal_to_octal (n : Nat) : List Nat :=
  if n < 8 then [n]
  else (n % 8) :: decimal_to_octal (n / 8)

theorem binary_1101101000_to_octal_1550 :
  let binary : List Bool := [false, false, false, true, false, true, true, false, true, true]
  let octal : List Nat := [0, 5, 5, 1]
  decimal_to_octal (binary_to_decimal binary) = octal.reverse := by
  sorry

end NUMINAMATH_CALUDE_binary_1101101000_to_octal_1550_l691_69130


namespace NUMINAMATH_CALUDE_workshop_average_salary_l691_69126

/-- Proves that the average salary of all workers in a workshop is 8000, given the specified conditions. -/
theorem workshop_average_salary 
  (total_workers : ℕ) 
  (num_technicians : ℕ) 
  (avg_salary_technicians : ℕ) 
  (avg_salary_rest : ℕ) 
  (h1 : total_workers = 49)
  (h2 : num_technicians = 7)
  (h3 : avg_salary_technicians = 20000)
  (h4 : avg_salary_rest = 6000) :
  (num_technicians * avg_salary_technicians + (total_workers - num_technicians) * avg_salary_rest) / total_workers = 8000 := by
  sorry

#check workshop_average_salary

end NUMINAMATH_CALUDE_workshop_average_salary_l691_69126


namespace NUMINAMATH_CALUDE_fourth_rectangle_area_l691_69135

theorem fourth_rectangle_area (a b c : ℕ) (h1 : a + b + c = 350) : 
  ∃ d : ℕ, d = 300 - (a + b + c) ∧ d = 50 := by
  sorry

#check fourth_rectangle_area

end NUMINAMATH_CALUDE_fourth_rectangle_area_l691_69135


namespace NUMINAMATH_CALUDE_total_highlighters_l691_69109

theorem total_highlighters (pink : ℕ) (yellow : ℕ) (blue : ℕ)
  (h_pink : pink = 4)
  (h_yellow : yellow = 2)
  (h_blue : blue = 5) :
  pink + yellow + blue = 11 := by
  sorry

end NUMINAMATH_CALUDE_total_highlighters_l691_69109


namespace NUMINAMATH_CALUDE_max_cables_equals_max_edges_l691_69119

/-- Represents a bipartite graph with two sets of nodes -/
structure BipartiteGraph where
  setA : Nat
  setB : Nat

/-- Calculates the maximum number of edges in a bipartite graph -/
def maxEdges (g : BipartiteGraph) : Nat :=
  g.setA * g.setB

/-- Represents the company's computer network -/
def companyNetwork : BipartiteGraph :=
  { setA := 16, setB := 12 }

/-- The maximum number of cables needed is equal to the maximum number of edges in the bipartite graph -/
theorem max_cables_equals_max_edges :
  maxEdges companyNetwork = 192 := by
  sorry

#eval maxEdges companyNetwork

end NUMINAMATH_CALUDE_max_cables_equals_max_edges_l691_69119


namespace NUMINAMATH_CALUDE_right_triangle_division_l691_69190

/-- In a right triangle divided by lines parallel to the legs through a point on the hypotenuse,
    if the areas of the two smaller triangles are m and n times the area of the square respectively,
    then n = 1/(4m). -/
theorem right_triangle_division (m n : ℝ) : m > 0 → n > 0 → n = 1 / (4 * m) := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_division_l691_69190


namespace NUMINAMATH_CALUDE_curve_inequality_l691_69128

/-- Given real numbers a, b, c satisfying certain conditions, 
    prove an inequality for points on a specific curve. -/
theorem curve_inequality (a b c : ℝ) 
  (h1 : b^2 - a*c < 0) 
  (h2 : ∀ x y : ℝ, x > 0 → y > 0 → 
    a * (Real.log x)^2 + 2*b*(Real.log x * Real.log y) + c * (Real.log y)^2 = 1 → 
    (x = 10 ∧ y = 1/10) ∨ 
    (-1 / Real.sqrt (a*c - b^2) ≤ Real.log (x*y) ∧ 
     Real.log (x*y) ≤ 1 / Real.sqrt (a*c - b^2))) : 
  ∀ x y : ℝ, x > 0 → y > 0 → 
    a * (Real.log x)^2 + 2*b*(Real.log x * Real.log y) + c * (Real.log y)^2 = 1 → 
    -1 / Real.sqrt (a*c - b^2) ≤ Real.log (x*y) ∧ 
    Real.log (x*y) ≤ 1 / Real.sqrt (a*c - b^2) := by
  sorry

end NUMINAMATH_CALUDE_curve_inequality_l691_69128


namespace NUMINAMATH_CALUDE_coefficient_x_squared_in_binomial_expansion_l691_69180

theorem coefficient_x_squared_in_binomial_expansion :
  let n : ℕ := 5
  let a : ℕ := 1
  let b : ℕ := 2
  let r : ℕ := 2
  (n.choose r) * b^r = 40 := by
  sorry

end NUMINAMATH_CALUDE_coefficient_x_squared_in_binomial_expansion_l691_69180


namespace NUMINAMATH_CALUDE_function_inequality_function_inequality_bounded_l691_69136

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := 
  a * Real.sin x - (1/2) * Real.cos (2*x) + a - 3/a + 1/2

theorem function_inequality (a : ℝ) (h : a ≠ 0) :
  (∀ x, f a x ≤ 0) ↔ (0 < a ∧ a ≤ 1) :=
sorry

theorem function_inequality_bounded (a : ℝ) (h : a ≥ 2) :
  (∃ x, f a x ≤ 0) ↔ a ≥ 3 :=
sorry

end NUMINAMATH_CALUDE_function_inequality_function_inequality_bounded_l691_69136


namespace NUMINAMATH_CALUDE_gcd_of_162_180_450_l691_69120

theorem gcd_of_162_180_450 : Nat.gcd 162 (Nat.gcd 180 450) = 18 := by sorry

end NUMINAMATH_CALUDE_gcd_of_162_180_450_l691_69120


namespace NUMINAMATH_CALUDE_sum_of_squares_theorem_l691_69198

theorem sum_of_squares_theorem (a b c x y z : ℝ) 
  (h1 : x / a + y / b + z / c = 1)
  (h2 : a / x + b / y + c / z = 3) :
  x^2 / a^2 + y^2 / b^2 + z^2 / c^2 = 1/3 := by sorry

end NUMINAMATH_CALUDE_sum_of_squares_theorem_l691_69198


namespace NUMINAMATH_CALUDE_two_part_problem_solution_count_l691_69192

theorem two_part_problem_solution_count 
  (part1_methods : ℕ) 
  (part2_methods : ℕ) 
  (h1 : part1_methods = 2) 
  (h2 : part2_methods = 3) : 
  part1_methods * part2_methods = 6 := by
sorry

end NUMINAMATH_CALUDE_two_part_problem_solution_count_l691_69192


namespace NUMINAMATH_CALUDE_postcard_price_calculation_bernie_postcard_problem_l691_69132

theorem postcard_price_calculation (initial_postcards : Nat) 
  (sold_postcards : Nat) (price_per_sold : Nat) (final_total : Nat) : Nat :=
  let total_earned := sold_postcards * price_per_sold
  let remaining_original := initial_postcards - sold_postcards
  let new_postcards := final_total - remaining_original
  total_earned / new_postcards

theorem bernie_postcard_problem : 
  postcard_price_calculation 18 9 15 36 = 5 := by
  sorry

end NUMINAMATH_CALUDE_postcard_price_calculation_bernie_postcard_problem_l691_69132


namespace NUMINAMATH_CALUDE_common_external_tangent_y_intercept_l691_69122

/-- Given two circles with centers (1,3) and (15,8) and radii 3 and 10 respectively,
    this theorem proves that the y-intercept of their common external tangent
    with positive slope is 518/1197. -/
theorem common_external_tangent_y_intercept :
  let c1 : ℝ × ℝ := (1, 3)
  let r1 : ℝ := 3
  let c2 : ℝ × ℝ := (15, 8)
  let r2 : ℝ := 10
  let m : ℝ := (8 - 3) / (15 - 1)  -- slope of line connecting centers
  let tan_2theta : ℝ := (2 * m) / (1 - m^2)  -- tangent of double angle
  let m_tangent : ℝ := Real.sqrt (tan_2theta / (1 + tan_2theta))  -- slope of tangent line
  let x_intercept : ℝ := -(3 - m * 1) / m  -- x-intercept of line connecting centers
  ∃ b : ℝ, b = m_tangent * (-x_intercept) ∧ b = 518 / 1197 :=
by sorry

end NUMINAMATH_CALUDE_common_external_tangent_y_intercept_l691_69122


namespace NUMINAMATH_CALUDE_arcsin_equation_solution_l691_69125

theorem arcsin_equation_solution :
  ∃ x : ℝ, x = Real.sqrt 102 / 51 ∧ 
    Real.arcsin x + Real.arcsin (3 * x) = π / 4 ∧
    -1 < x ∧ x < 1 ∧ -1 < 3 * x ∧ 3 * x < 1 :=
by sorry

end NUMINAMATH_CALUDE_arcsin_equation_solution_l691_69125


namespace NUMINAMATH_CALUDE_arithmetic_sequence_common_difference_l691_69191

/-- An arithmetic sequence with sum S_n for the first n terms -/
structure ArithmeticSequence where
  a : ℕ → ℝ  -- The sequence
  S : ℕ → ℝ  -- Sum function
  is_arithmetic : ∀ n, a (n + 1) - a n = a (n + 2) - a (n + 1)
  sum_formula : ∀ n, S n = n * (a 1 + a n) / 2

/-- The common difference of an arithmetic sequence -/
def common_difference (seq : ArithmeticSequence) : ℝ :=
  seq.a 2 - seq.a 1

theorem arithmetic_sequence_common_difference 
  (seq : ArithmeticSequence)
  (h1 : seq.S 4 = 3 * seq.S 2)
  (h2 : seq.a 7 = 15) :
  common_difference seq = 2 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_common_difference_l691_69191


namespace NUMINAMATH_CALUDE_league_games_l691_69141

theorem league_games (n : ℕ) (h : n = 10) : 
  (n * (n - 1)) / 2 = 45 := by
  sorry

end NUMINAMATH_CALUDE_league_games_l691_69141


namespace NUMINAMATH_CALUDE_sine_cosine_parity_l691_69106

def is_odd_function (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

def is_even_function (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = f x

theorem sine_cosine_parity (sine cosine : ℝ → ℝ) 
  (h1 : ∀ x, sine (-x) = -(sine x)) 
  (h2 : ∀ x, cosine (-x) = cosine x) : 
  is_odd_function sine ∧ is_even_function cosine := by
  sorry

end NUMINAMATH_CALUDE_sine_cosine_parity_l691_69106


namespace NUMINAMATH_CALUDE_multiplicative_inverse_301_mod_401_l691_69168

theorem multiplicative_inverse_301_mod_401 : ∃ x : ℤ, 0 ≤ x ∧ x < 401 ∧ (301 * x) % 401 = 1 :=
  by
  use 397
  sorry

end NUMINAMATH_CALUDE_multiplicative_inverse_301_mod_401_l691_69168


namespace NUMINAMATH_CALUDE_all_dice_even_probability_l691_69115

/-- The probability of a single standard six-sided die showing an even number -/
def prob_single_even : ℚ := 1 / 2

/-- The number of dice being tossed simultaneously -/
def num_dice : ℕ := 5

/-- The probability of all dice showing an even number -/
def prob_all_even : ℚ := (prob_single_even) ^ num_dice

theorem all_dice_even_probability :
  prob_all_even = 1 / 32 := by
  sorry

end NUMINAMATH_CALUDE_all_dice_even_probability_l691_69115


namespace NUMINAMATH_CALUDE_polygon_contains_integer_different_points_l691_69163

/-- A polygon on the coordinate plane. -/
structure Polygon where
  -- We don't need to define the full structure of a polygon,
  -- just its existence and area property
  area : ℝ

/-- Two points are integer-different if their coordinate differences are integers. -/
def integer_different (p₁ p₂ : ℝ × ℝ) : Prop :=
  ∃ (m n : ℤ), p₁.1 - p₂.1 = m ∧ p₁.2 - p₂.2 = n

/-- Main theorem: If a polygon has area greater than 1, it contains two integer-different points. -/
theorem polygon_contains_integer_different_points (P : Polygon) (h : P.area > 1) :
  ∃ (p₁ p₂ : ℝ × ℝ), p₁ ≠ p₂ ∧ integer_different p₁ p₂ := by
  sorry

end NUMINAMATH_CALUDE_polygon_contains_integer_different_points_l691_69163


namespace NUMINAMATH_CALUDE_qin_jiushao_v_1_l691_69188

def f (x : ℝ) : ℝ := 3 * x^4 + 2 * x^2 + x + 4

def nested_f (x : ℝ) : ℝ := ((3 * x + 0) * x + 2) * x + 1 * x + 4

def v_1 (x : ℝ) : ℝ := 3 * x + 0

theorem qin_jiushao_v_1 : v_1 10 = 30 := by sorry

end NUMINAMATH_CALUDE_qin_jiushao_v_1_l691_69188


namespace NUMINAMATH_CALUDE_scientific_notation_equivalence_l691_69104

theorem scientific_notation_equivalence : 
  (361000000 : ℝ) = 3.61 * (10 : ℝ) ^ 8 := by sorry

end NUMINAMATH_CALUDE_scientific_notation_equivalence_l691_69104


namespace NUMINAMATH_CALUDE_hash_nested_20_l691_69139

-- Define the # operation
def hash (N : ℝ) : ℝ := 0.5 * N + 2

-- State the theorem
theorem hash_nested_20 : hash (hash (hash (hash 20))) = 5 := by sorry

end NUMINAMATH_CALUDE_hash_nested_20_l691_69139


namespace NUMINAMATH_CALUDE_zero_meetings_on_circular_track_l691_69193

/-- Represents the number of meetings between two people moving on a circular track. -/
def number_of_meetings (circumference : ℝ) (speed_forward : ℝ) (speed_backward : ℝ) : ℕ :=
  -- The actual calculation is not implemented, as we only need the statement
  sorry

/-- Theorem stating that the number of meetings is 0 under the given conditions. -/
theorem zero_meetings_on_circular_track :
  let circumference : ℝ := 270
  let speed_forward : ℝ := 6
  let speed_backward : ℝ := 3
  number_of_meetings circumference speed_forward speed_backward = 0 := by
  sorry

end NUMINAMATH_CALUDE_zero_meetings_on_circular_track_l691_69193


namespace NUMINAMATH_CALUDE_walking_delay_bus_miss_time_l691_69123

/-- Given a usual walking time and a reduced speed factor, calculates the delay in reaching the destination. -/
theorem walking_delay (usual_time : ℝ) (speed_factor : ℝ) : 
  usual_time > 0 → 
  speed_factor > 0 → 
  speed_factor < 1 → 
  (usual_time / speed_factor) - usual_time = usual_time * (1 / speed_factor - 1) :=
by sorry

/-- Proves that walking at 4/5 of the usual speed, with a usual time of 24 minutes, results in a 6-minute delay. -/
theorem bus_miss_time (usual_time : ℝ) (h1 : usual_time = 24) : 
  (usual_time / (4/5)) - usual_time = 6 :=
by sorry

end NUMINAMATH_CALUDE_walking_delay_bus_miss_time_l691_69123


namespace NUMINAMATH_CALUDE_basketball_weight_proof_l691_69154

/-- The weight of one basketball in pounds -/
def basketball_weight : ℝ := 20.83333

/-- The weight of one bicycle in pounds -/
def bicycle_weight : ℝ := 37.5

/-- The weight of one skateboard in pounds -/
def skateboard_weight : ℝ := 15

theorem basketball_weight_proof :
  (9 * basketball_weight = 5 * bicycle_weight) ∧
  (2 * bicycle_weight + 3 * skateboard_weight = 120) ∧
  (skateboard_weight = 15) :=
by sorry

end NUMINAMATH_CALUDE_basketball_weight_proof_l691_69154


namespace NUMINAMATH_CALUDE_quadratic_equation_solution_l691_69167

theorem quadratic_equation_solution : 
  let f : ℝ → ℝ := fun x ↦ 2 * x^2 - 2
  ∃ x₁ x₂ : ℝ, x₁ = 1 ∧ x₂ = -1 ∧ f x₁ = 0 ∧ f x₂ = 0 ∧ 
  ∀ x : ℝ, f x = 0 → x = x₁ ∨ x = x₂ := by
sorry

end NUMINAMATH_CALUDE_quadratic_equation_solution_l691_69167


namespace NUMINAMATH_CALUDE_stating_least_possible_area_l691_69149

/-- Represents the length of a side of a square in centimeters. -/
def SideLength : ℝ := 5

/-- The lower bound of the actual side length when measured to the nearest centimeter. -/
def LowerBound : ℝ := SideLength - 0.5

/-- Calculates the area of a square given its side length. -/
def SquareArea (side : ℝ) : ℝ := side * side

/-- 
Theorem stating that the least possible area of a square with sides measured as 5 cm 
to the nearest centimeter is 20.25 cm².
-/
theorem least_possible_area :
  SquareArea LowerBound = 20.25 := by sorry

end NUMINAMATH_CALUDE_stating_least_possible_area_l691_69149


namespace NUMINAMATH_CALUDE_total_money_found_l691_69165

-- Define the value of each coin type in cents
def quarter_value : ℕ := 25
def dime_value : ℕ := 10
def nickel_value : ℕ := 5
def penny_value : ℕ := 1

-- Define the number of each coin type found
def quarters_found : ℕ := 10
def dimes_found : ℕ := 3
def nickels_found : ℕ := 3
def pennies_found : ℕ := 5

-- Theorem to prove
theorem total_money_found :
  (quarters_found * quarter_value +
   dimes_found * dime_value +
   nickels_found * nickel_value +
   pennies_found * penny_value) = 300 := by
  sorry

end NUMINAMATH_CALUDE_total_money_found_l691_69165


namespace NUMINAMATH_CALUDE_simplify_complex_expression_l691_69178

-- Define the complex number i
def i : ℂ := Complex.I

-- Theorem statement
theorem simplify_complex_expression : i * (1 - i)^2 = 2 := by
  sorry

end NUMINAMATH_CALUDE_simplify_complex_expression_l691_69178


namespace NUMINAMATH_CALUDE_largest_inscribed_circle_radius_l691_69176

/-- The radius of the quarter circle -/
def R : ℝ := 12

/-- The radius of the largest inscribed circle -/
def r : ℝ := 3

/-- Theorem stating that r is the radius of the largest inscribed circle -/
theorem largest_inscribed_circle_radius : 
  (R - r)^2 - r^2 = (R/2 + r)^2 - (R/2 - r)^2 := by sorry

end NUMINAMATH_CALUDE_largest_inscribed_circle_radius_l691_69176


namespace NUMINAMATH_CALUDE_hadassah_additional_paintings_l691_69144

/-- Calculates the number of additional paintings given initial and total painting information -/
def additional_paintings (initial_paintings : ℕ) (initial_time : ℕ) (total_time : ℕ) : ℕ :=
  let painting_rate := initial_paintings / initial_time
  let additional_time := total_time - initial_time
  painting_rate * additional_time

/-- Proves that Hadassah painted 20 additional paintings -/
theorem hadassah_additional_paintings :
  additional_paintings 12 6 16 = 20 := by
  sorry

end NUMINAMATH_CALUDE_hadassah_additional_paintings_l691_69144


namespace NUMINAMATH_CALUDE_probability_even_rolls_l691_69196

def is_even (n : ℕ) : Bool := n % 2 = 0

def count_even (n : ℕ) : ℕ := (List.range n).filter is_even |>.length

theorem probability_even_rolls (die1 : ℕ) (die2 : ℕ) 
  (h1 : die1 = 6) (h2 : die2 = 7) : 
  (count_even die1 : ℚ) / die1 * (count_even die2 : ℚ) / die2 = 3 / 14 := by
  sorry

end NUMINAMATH_CALUDE_probability_even_rolls_l691_69196


namespace NUMINAMATH_CALUDE_sqrt_nested_expression_l691_69143

theorem sqrt_nested_expression : Real.sqrt (25 * Real.sqrt (15 * Real.sqrt 9)) = 5 * Real.sqrt 15 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_nested_expression_l691_69143


namespace NUMINAMATH_CALUDE_trees_in_yard_l691_69156

/-- The number of trees planted along a yard with given specifications -/
def number_of_trees (yard_length : ℕ) (tree_distance : ℕ) : ℕ :=
  (yard_length / tree_distance) + 1

/-- Theorem stating the number of trees planted along the yard -/
theorem trees_in_yard :
  number_of_trees 273 21 = 14 := by
  sorry

end NUMINAMATH_CALUDE_trees_in_yard_l691_69156


namespace NUMINAMATH_CALUDE_bridge_extension_l691_69195

theorem bridge_extension (river_width bridge_length : ℕ) 
  (hw : river_width = 487) 
  (hb : bridge_length = 295) : 
  river_width - bridge_length = 192 := by
sorry

end NUMINAMATH_CALUDE_bridge_extension_l691_69195


namespace NUMINAMATH_CALUDE_algebraic_multiplication_l691_69137

theorem algebraic_multiplication (x y : ℝ) : 
  6 * x * y^2 * (-1/2 * x^3 * y^3) = -3 * x^4 * y^5 := by
  sorry

end NUMINAMATH_CALUDE_algebraic_multiplication_l691_69137


namespace NUMINAMATH_CALUDE_even_function_value_l691_69146

def is_even (f : ℝ → ℝ) : Prop := ∀ x, f x = f (-x)

theorem even_function_value (f : ℝ → ℝ) 
  (h_even : is_even f)
  (h_periodic : ∀ x, f (x + 2) * f x = 4)
  (h_positive : ∀ x, f x > 0) :
  f 2017 = 2 := by sorry

end NUMINAMATH_CALUDE_even_function_value_l691_69146


namespace NUMINAMATH_CALUDE_ellipse_intersection_midpoint_l691_69186

-- Define the ellipse C
def ellipse (a b : ℝ) (x y : ℝ) : Prop :=
  x^2 / a^2 + y^2 / b^2 = 1

-- Define the line L
def line (x y : ℝ) : Prop :=
  y = 3/4 * (x - 3)

theorem ellipse_intersection_midpoint :
  ∀ a b : ℝ,
  a > b ∧ b > 0 →
  ellipse a b 0 4 →
  (a^2 - b^2) / a^2 = (3/5)^2 →
  ∃ x1 x2 y1 y2 : ℝ,
    ellipse a b x1 y1 ∧
    ellipse a b x2 y2 ∧
    line x1 y1 ∧
    line x2 y2 ∧
    (x1 + x2) / 2 = 1 ∧
    (y1 + y2) / 2 = -9/4 :=
by sorry

end NUMINAMATH_CALUDE_ellipse_intersection_midpoint_l691_69186


namespace NUMINAMATH_CALUDE_landscape_breadth_l691_69114

theorem landscape_breadth (length width : ℝ) (playground_area : ℝ) : 
  width = 8 * length →
  playground_area = 3200 →
  playground_area = (1 / 9) * (length * width) →
  width = 480 := by
sorry

end NUMINAMATH_CALUDE_landscape_breadth_l691_69114


namespace NUMINAMATH_CALUDE_duck_pond_problem_l691_69172

theorem duck_pond_problem (small_pond : ℕ) (large_pond : ℕ) 
  (green_small : ℚ) (green_large : ℚ) (total_green : ℚ) :
  large_pond = 50 →
  green_small = 1/5 →
  green_large = 3/25 →
  total_green = 3/20 →
  green_small * small_pond.cast + green_large * large_pond.cast = 
    total_green * (small_pond.cast + large_pond.cast) →
  small_pond = 30 := by
sorry

end NUMINAMATH_CALUDE_duck_pond_problem_l691_69172


namespace NUMINAMATH_CALUDE_smallest_solution_of_equation_l691_69174

theorem smallest_solution_of_equation :
  ∃ x : ℝ, x^4 - 54*x^2 + 441 = 0 ∧
  (∀ y : ℝ, y^4 - 54*y^2 + 441 = 0 → x ≤ y) ∧
  x = -Real.sqrt 33 := by
  sorry

end NUMINAMATH_CALUDE_smallest_solution_of_equation_l691_69174


namespace NUMINAMATH_CALUDE_abs_rational_nonnegative_l691_69171

theorem abs_rational_nonnegative (x : ℚ) : 0 ≤ |x| := by
  sorry

end NUMINAMATH_CALUDE_abs_rational_nonnegative_l691_69171


namespace NUMINAMATH_CALUDE_number_problem_l691_69155

theorem number_problem : ∃ x : ℝ, x = 25 ∧ (2/5) * x + 22 = (80/100) * 40 := by
  sorry

end NUMINAMATH_CALUDE_number_problem_l691_69155


namespace NUMINAMATH_CALUDE_decreasing_then_increasing_possible_increasing_then_decreasing_impossible_l691_69194

/-- Definition of our sequence based on original positive numbers -/
def sequence_a (original : List ℝ) : ℕ → ℝ :=
  λ n => (original.map (λ x => x ^ n)).sum

/-- Theorem stating the possibility of decreasing then increasing sequence -/
theorem decreasing_then_increasing_possible :
  ∃ original : List ℝ, 
    (∀ x ∈ original, x > 0) ∧
    (sequence_a original 1 > sequence_a original 2) ∧
    (sequence_a original 2 > sequence_a original 3) ∧
    (sequence_a original 3 > sequence_a original 4) ∧
    (sequence_a original 4 > sequence_a original 5) ∧
    (∀ n ≥ 5, sequence_a original n < sequence_a original (n + 1)) :=
sorry

/-- Theorem stating the impossibility of increasing then decreasing sequence -/
theorem increasing_then_decreasing_impossible :
  ¬∃ original : List ℝ, 
    (∀ x ∈ original, x > 0) ∧
    (sequence_a original 1 < sequence_a original 2) ∧
    (sequence_a original 2 < sequence_a original 3) ∧
    (sequence_a original 3 < sequence_a original 4) ∧
    (sequence_a original 4 < sequence_a original 5) ∧
    (∀ n ≥ 5, sequence_a original n > sequence_a original (n + 1)) :=
sorry

end NUMINAMATH_CALUDE_decreasing_then_increasing_possible_increasing_then_decreasing_impossible_l691_69194


namespace NUMINAMATH_CALUDE_andy_candy_canes_l691_69101

/-- The number of candy canes Andy got from his parents -/
def parents_candy : ℕ := sorry

/-- The number of candy canes Andy got from teachers -/
def teachers_candy : ℕ := 3 * 4

/-- The ratio of candy canes to cavities -/
def candy_to_cavity_ratio : ℕ := 4

/-- The number of cavities Andy got -/
def cavities : ℕ := 16

/-- The fraction of additional candy canes Andy buys -/
def bought_candy_fraction : ℚ := 1 / 7

theorem andy_candy_canes :
  parents_candy = 44 ∧
  parents_candy + teachers_candy + (parents_candy + teachers_candy : ℚ) * bought_candy_fraction = cavities * candy_to_cavity_ratio := by sorry

end NUMINAMATH_CALUDE_andy_candy_canes_l691_69101


namespace NUMINAMATH_CALUDE_range_of_a_l691_69179

open Set Real

noncomputable def f (x : ℝ) : ℝ := 4 * x / (3 * x^2 + 3)

noncomputable def g (a : ℝ) (x : ℝ) : ℝ := (1/2) * x^2 - log x - a

def I : Set ℝ := Ioo 0 2
def J : Set ℝ := Icc 1 2

theorem range_of_a :
  {a : ℝ | ∀ x₁ ∈ I, ∃ x₂ ∈ J, f x₁ = g a x₂} = Icc (1/2) (4/3 - log 2) := by sorry

end NUMINAMATH_CALUDE_range_of_a_l691_69179
