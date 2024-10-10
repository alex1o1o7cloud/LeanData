import Mathlib

namespace vector_dot_product_cos_2x_l3072_307243

theorem vector_dot_product_cos_2x (x : ℝ) : 
  let a := (Real.sqrt 3 * Real.sin x, Real.cos x)
  let b := (Real.cos x, -Real.cos x)
  x ∈ Set.Ioo (7 * Real.pi / 12) (5 * Real.pi / 6) →
  a.1 * b.1 + a.2 * b.2 = -5/4 →
  Real.cos (2 * x) = (3 - Real.sqrt 21) / 8 := by
    sorry

end vector_dot_product_cos_2x_l3072_307243


namespace discount_percent_calculation_l3072_307229

theorem discount_percent_calculation (MP : ℝ) (CP : ℝ) (h1 : CP = 0.64 * MP) (h2 : 34.375 = (CP * 1.34375 - CP) / CP * 100) :
  (MP - CP * 1.34375) / MP * 100 = 14 := by
  sorry

end discount_percent_calculation_l3072_307229


namespace collinear_vectors_imply_fixed_point_l3072_307273

/-- Two 2D vectors are collinear if one is a scalar multiple of the other -/
def collinear (v w : ℝ × ℝ) : Prop :=
  ∃ t : ℝ, v.1 = t * w.1 ∧ v.2 = t * w.2

/-- A point (x, y) is on a line y = mx + c if y = mx + c -/
def on_line (m c : ℝ) (p : ℝ × ℝ) : Prop :=
  p.2 = m * p.1 + c

theorem collinear_vectors_imply_fixed_point (k b : ℝ) :
  collinear (k + 2, 1) (-b, 1) →
  on_line k b (1, -2) :=
by sorry

end collinear_vectors_imply_fixed_point_l3072_307273


namespace cubic_factorization_l3072_307290

theorem cubic_factorization (k : ℕ) (hk : k ≥ 2) :
  let n : ℕ := 16 * k^3 + 12 * k^2 + 3 * k - 126
  let factor1 : ℕ := n + 4 * k + 1
  let factor2 : ℕ := (n - 4 * k - 1)^2 + (4 * k + 1) * n
  (n^3 + 4 * n + 505 = factor1 * factor2) ∧
  (factor1 > Real.sqrt n) ∧
  (factor2 > Real.sqrt n) :=
by sorry

end cubic_factorization_l3072_307290


namespace card_ratio_proof_l3072_307265

theorem card_ratio_proof (total_cards baseball_cards : ℕ) 
  (h1 : total_cards = 125)
  (h2 : baseball_cards = 95) : 
  (baseball_cards : ℚ) / (total_cards - baseball_cards) = 19 / 6 := by
  sorry

end card_ratio_proof_l3072_307265


namespace sum_greater_product_iff_one_l3072_307296

theorem sum_greater_product_iff_one (a b : ℕ) (ha : a > 0) (hb : b > 0) :
  a + b > a * b ↔ a = 1 ∨ b = 1 := by
  sorry

end sum_greater_product_iff_one_l3072_307296


namespace cheapest_for_second_caterer_l3072_307257

-- Define the pricing functions for both caterers
def first_caterer (x : ℕ) : ℕ := 150 + 18 * x

def second_caterer (x : ℕ) : ℕ :=
  if x ≤ 30 then 250 + 15 * x
  else 400 + 10 * x

-- Define a function to compare the prices
def second_cheaper (x : ℕ) : Prop :=
  second_caterer x < first_caterer x

-- Theorem statement
theorem cheapest_for_second_caterer :
  ∀ n : ℕ, n < 32 → ¬(second_cheaper n) ∧ second_cheaper 32 :=
sorry

end cheapest_for_second_caterer_l3072_307257


namespace imaginary_part_of_z_l3072_307211

theorem imaginary_part_of_z (i : ℂ) (h : i^2 = -1) : 
  let z : ℂ := 2 / (-1 + i)
  Complex.im z = -1 := by
sorry

end imaginary_part_of_z_l3072_307211


namespace tops_count_l3072_307247

-- Define the number of marbles for each person
def dennis_marbles : ℕ := 70
def kurt_marbles : ℕ := dennis_marbles - 45
def laurie_marbles : ℕ := kurt_marbles + 12
def jessica_marbles : ℕ := laurie_marbles + 25

-- Define the number of tops for each person
def laurie_tops : ℕ := laurie_marbles * 2
def kurt_tops : ℕ := kurt_marbles - 3
def dennis_tops : ℕ := dennis_marbles + 8
def jessica_tops : ℕ := jessica_marbles - 10

theorem tops_count :
  laurie_tops = 74 ∧
  kurt_tops = 22 ∧
  dennis_tops = 78 ∧
  jessica_tops = 52 := by sorry

end tops_count_l3072_307247


namespace correct_multiplication_result_l3072_307277

theorem correct_multiplication_result (a : ℕ) : 
  (153 * a ≠ 102325 ∧ 153 * a < 102357 ∧ 102357 - 153 * a < 153) → 
  153 * a = 102357 :=
by sorry

end correct_multiplication_result_l3072_307277


namespace modular_inverse_17_mod_19_l3072_307260

theorem modular_inverse_17_mod_19 :
  ∃ x : ℕ, x ≤ 18 ∧ (17 * x) % 19 = 1 :=
by
  -- The proof goes here
  sorry

end modular_inverse_17_mod_19_l3072_307260


namespace sum_of_powers_l3072_307259

theorem sum_of_powers (x : ℂ) (h1 : x^7 = 1) (h2 : x ≠ 1) :
  x^2 / (x - 1) + x^4 / (x^2 - 1) + x^6 / (x^3 - 1) = 1 := by
  sorry

end sum_of_powers_l3072_307259


namespace exponential_function_property_l3072_307298

theorem exponential_function_property (a : ℝ) (ha : a > 0 ∧ a ≠ 1) :
  ∀ x y : ℝ, (fun x => a^x) (x + y) = (fun x => a^x) x * (fun x => a^x) y :=
by sorry

end exponential_function_property_l3072_307298


namespace extreme_value_difference_l3072_307221

noncomputable def f (a b x : ℝ) : ℝ := x^3 + 3*a*x^2 + 3*b*x

theorem extreme_value_difference (a b : ℝ) :
  (∃ x, x = 2 ∧ (deriv (f a b)) x = 0) →
  (deriv (f a b)) 1 = -3 →
  ∃ max min, (∀ x, f a b x ≤ max) ∧ 
              (∀ x, f a b x ≥ min) ∧ 
              max - min = 4 :=
by sorry

end extreme_value_difference_l3072_307221


namespace part_one_part_two_l3072_307230

-- Define the function f
def f (m : ℝ) (x : ℝ) : ℝ := |x + m| + |2 * x + 1|

-- Part I: Prove that when m = -1, f(x) ≤ 3 is equivalent to -1 ≤ x ≤ 1
theorem part_one : 
  ∀ x : ℝ, f (-1) x ≤ 3 ↔ -1 ≤ x ∧ x ≤ 1 := by sorry

-- Part II: Prove that the minimum value of f(x) is |m - 1/2|
theorem part_two (m : ℝ) : 
  ∃ x : ℝ, ∀ y : ℝ, f m x ≤ f m y ∧ f m x = |m - 1/2| := by sorry

end part_one_part_two_l3072_307230


namespace characterization_of_C_l3072_307261

def A : Set ℝ := {x | x^2 - 6*x + 8 = 0}
def B (m : ℝ) : Set ℝ := {x | m*x - 4 = 0}
def C : Set ℝ := {m | B m ∩ A = B m}

theorem characterization_of_C : C = {0, 1, 2} := by sorry

end characterization_of_C_l3072_307261


namespace ellipse_dimensions_l3072_307241

/-- Given an ellipse with semi-major axis a and semi-minor axis b,
    where the intersection of lines AB and CF is at (3a, 16),
    prove that a = 5 and b = 4 -/
theorem ellipse_dimensions (a b : ℝ) (h1 : a > b) (h2 : b > 0) :
  (∃ c : ℝ, a^2 = b^2 + c^2) →
  (∃ x y : ℝ, x = 3*a ∧ y = 16 ∧ x/(-a) + y/b = 1 ∧ x/c + y/(-b) = 1) →
  a = 5 ∧ b = 4 := by sorry

end ellipse_dimensions_l3072_307241


namespace expected_outcome_is_negative_two_thirds_l3072_307266

/-- Represents the sides of the die --/
inductive DieSide
| A
| B
| C

/-- The probability of rolling each side of the die --/
def probability (side : DieSide) : ℚ :=
  match side with
  | DieSide.A => 1/3
  | DieSide.B => 1/2
  | DieSide.C => 1/6

/-- The monetary outcome of rolling each side of the die --/
def monetaryOutcome (side : DieSide) : ℚ :=
  match side with
  | DieSide.A => 2
  | DieSide.B => -4
  | DieSide.C => 6

/-- The expected monetary outcome of rolling the die --/
def expectedOutcome : ℚ :=
  (probability DieSide.A * monetaryOutcome DieSide.A) +
  (probability DieSide.B * monetaryOutcome DieSide.B) +
  (probability DieSide.C * monetaryOutcome DieSide.C)

/-- Theorem stating that the expected monetary outcome is -2/3 --/
theorem expected_outcome_is_negative_two_thirds :
  expectedOutcome = -2/3 := by
  sorry

end expected_outcome_is_negative_two_thirds_l3072_307266


namespace total_wage_calculation_l3072_307281

/-- Represents the number of days it takes for a worker to complete the job alone -/
structure WorkerSpeed :=
  (days : ℕ)

/-- Calculates the daily work rate of a worker -/
def dailyRate (w : WorkerSpeed) : ℚ :=
  1 / w.days

/-- Represents the wage distribution between two workers -/
structure WageDistribution :=
  (worker_a : ℚ)
  (total : ℚ)

theorem total_wage_calculation 
  (speed_a : WorkerSpeed)
  (speed_b : WorkerSpeed)
  (wage_dist : WageDistribution)
  (h1 : speed_a.days = 10)
  (h2 : speed_b.days = 15)
  (h3 : wage_dist.worker_a = 1980)
  : wage_dist.total = 3300 :=
sorry

end total_wage_calculation_l3072_307281


namespace circle_trajectory_l3072_307299

-- Define the circle equation as a function of m, x, and y
def circle_equation (m x y : ℝ) : Prop :=
  x^2 + y^2 - (4*m + 2)*x - 2*m*y + 4*m^2 + 4*m + 1 = 0

-- Define the trajectory equation
def trajectory_equation (x y : ℝ) : Prop :=
  x - 2*y - 1 = 0

-- State the theorem
theorem circle_trajectory :
  ∀ m x y : ℝ, x ≠ 1 →
  (∃ m, circle_equation m x y) ↔ trajectory_equation x y :=
sorry

end circle_trajectory_l3072_307299


namespace total_sheets_required_l3072_307297

/-- The number of letters in the English alphabet -/
def alphabet_size : ℕ := 26

/-- The number of times each letter needs to be written -/
def writing_times : ℕ := 3

/-- The number of sheets needed for one writing of a letter -/
def sheets_per_writing : ℕ := 1

/-- Theorem: The total number of sheets required to write each letter of the English alphabet
    three times (uppercase, lowercase, and cursive script) is 78. -/
theorem total_sheets_required :
  alphabet_size * writing_times * sheets_per_writing = 78 := by sorry

end total_sheets_required_l3072_307297


namespace speed_conversion_l3072_307233

/-- Conversion factor from meters per second to kilometers per hour -/
def mps_to_kmph : ℝ := 3.6

/-- Given speed in meters per second -/
def given_speed : ℝ := 20

/-- Theorem: Converting 20 mps to kmph results in 72 kmph -/
theorem speed_conversion :
  given_speed * mps_to_kmph = 72 := by sorry

end speed_conversion_l3072_307233


namespace fixed_point_unique_l3072_307215

/-- The line l passes through the point (x, y) for all real values of m -/
def passes_through (x y : ℝ) : Prop :=
  ∀ m : ℝ, (2 + m) * x + (1 - 2*m) * y + (4 - 3*m) = 0

/-- The point M is the unique point that the line l passes through for all m -/
theorem fixed_point_unique :
  ∃! p : ℝ × ℝ, passes_through p.1 p.2 ∧ p = (-1, -2) :=
sorry

end fixed_point_unique_l3072_307215


namespace player_b_wins_l3072_307210

/-- Represents a chessboard --/
def Chessboard := Fin 8 → Fin 8 → Option Bool

/-- Represents a position on the chessboard --/
def Position := Fin 8 × Fin 8

/-- Checks if a bishop can be captured at a given position --/
def canBeCaptured (board : Chessboard) (pos : Position) : Prop :=
  sorry

/-- Represents a valid move in the game --/
def ValidMove (board : Chessboard) (pos : Position) : Prop :=
  board pos.1 pos.2 = none ∧ ¬canBeCaptured board pos

/-- Represents the state of the game --/
structure GameState where
  board : Chessboard
  playerATurn : Bool

/-- Represents a strategy for a player --/
def Strategy := GameState → Position

/-- Checks if a strategy is winning for Player B --/
def isWinningStrategyForB (s : Strategy) : Prop :=
  sorry

/-- The main theorem stating that Player B has a winning strategy --/
theorem player_b_wins : ∃ s : Strategy, isWinningStrategyForB s :=
  sorry

end player_b_wins_l3072_307210


namespace sum_of_fifty_eights_l3072_307227

theorem sum_of_fifty_eights : (List.replicate 50 8).sum = 400 := by
  sorry

end sum_of_fifty_eights_l3072_307227


namespace total_rent_is_435_l3072_307280

/-- Represents the rent calculation for a pasture shared by multiple parties -/
structure PastureRent where
  a_horses : ℕ
  a_months : ℕ
  b_horses : ℕ
  b_months : ℕ
  c_horses : ℕ
  c_months : ℕ
  b_payment : ℕ

/-- Calculates the total rent for the pasture -/
def calculate_total_rent (pr : PastureRent) : ℕ :=
  let total_horse_months := pr.a_horses * pr.a_months + pr.b_horses * pr.b_months + pr.c_horses * pr.c_months
  let b_horse_months := pr.b_horses * pr.b_months
  (pr.b_payment * total_horse_months) / b_horse_months

/-- Theorem stating that the total rent for the given conditions is 435 -/
theorem total_rent_is_435 (pr : PastureRent) 
  (h1 : pr.a_horses = 12) (h2 : pr.a_months = 8)
  (h3 : pr.b_horses = 16) (h4 : pr.b_months = 9)
  (h5 : pr.c_horses = 18) (h6 : pr.c_months = 6)
  (h7 : pr.b_payment = 180) : 
  calculate_total_rent pr = 435 := by
  sorry

end total_rent_is_435_l3072_307280


namespace f_bounds_f_inequality_solution_set_l3072_307217

def f (x : ℝ) : ℝ := |x - 2| - |x - 5|

theorem f_bounds : ∀ x : ℝ, -3 ≤ f x ∧ f x ≤ 3 := by sorry

theorem f_inequality_solution_set :
  {x : ℝ | f x ≥ x^2 - 8*x + 15} = {x : ℝ | 5 - Real.sqrt 3 ≤ x ∧ x ≤ 6} := by sorry

end f_bounds_f_inequality_solution_set_l3072_307217


namespace arthurs_dinner_cost_l3072_307288

def dinner_cost (appetizer steak wine_glass dessert : ℚ) (wine_glasses : ℕ) (discount_percent tip_percent : ℚ) : ℚ :=
  let full_cost := appetizer + steak + (wine_glass * wine_glasses) + dessert
  let discount := steak * discount_percent
  let discounted_cost := full_cost - discount
  let tip := full_cost * tip_percent
  discounted_cost + tip

theorem arthurs_dinner_cost :
  dinner_cost 8 20 3 6 2 (1/2) (1/5) = 38 := by
  sorry

end arthurs_dinner_cost_l3072_307288


namespace product_of_primes_l3072_307284

theorem product_of_primes (a b c d : ℕ) : 
  Prime a ∧ Prime b ∧ Prime c ∧ Prime d ∧  -- a, b, c, d are prime
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d ∧  -- a, b, c, d are distinct
  a + c = d ∧  -- condition (i)
  a * (a + b + c + d) = c * (d - b) ∧  -- condition (ii)
  1 + b * c + d = b * d  -- condition (iii)
  → a * b * c * d = 2002 := by
sorry

end product_of_primes_l3072_307284


namespace problem_solution_l3072_307289

theorem problem_solution (α β : Real) 
  (h1 : α ∈ Set.Ioo 0 (Real.pi / 2))
  (h2 : Real.cos (2 * α) = 4 / 5)
  (h3 : β ∈ Set.Ioo (Real.pi / 2) Real.pi)
  (h4 : 5 * Real.sin (2 * α + β) = Real.sin β) : 
  (Real.sin α + Real.cos α = 2 * Real.sqrt 10 / 5) ∧ 
  (β = 3 * Real.pi / 4) := by
sorry


end problem_solution_l3072_307289


namespace C_share_of_profit_l3072_307271

def investment_A : ℕ := 24000
def investment_B : ℕ := 32000
def investment_C : ℕ := 36000
def total_profit : ℕ := 92000

theorem C_share_of_profit :
  (investment_C : ℚ) / (investment_A + investment_B + investment_C) * total_profit = 36000 := by
  sorry

end C_share_of_profit_l3072_307271


namespace f_of_3_eq_5_l3072_307204

/-- The function f defined on ℝ -/
def f : ℝ → ℝ := fun x ↦ 2 * x - 1

/-- Theorem: f(3) = 5 -/
theorem f_of_3_eq_5 : f 3 = 5 := by sorry

end f_of_3_eq_5_l3072_307204


namespace trigonometric_identities_l3072_307269

theorem trigonometric_identities (α : Real) 
  (h : Real.sin α + Real.cos α = Real.sqrt 3 / 3) : 
  (Real.sin α)^4 + (Real.cos α)^4 = 7/9 ∧ 
  Real.tan α / (1 + (Real.tan α)^2) = -1/3 := by
  sorry

end trigonometric_identities_l3072_307269


namespace smallest_marble_count_l3072_307214

/-- Represents the number of marbles of each color in the urn -/
structure MarbleCount where
  red : ℕ
  white : ℕ
  blue : ℕ
  green : ℕ
  yellow : ℕ

/-- Calculates the total number of marbles in the urn -/
def totalMarbles (mc : MarbleCount) : ℕ :=
  mc.red + mc.white + mc.blue + mc.green + mc.yellow

/-- Represents the probability of drawing a specific combination of marbles -/
def drawProbability (mc : MarbleCount) (r w b g y : ℕ) : ℚ :=
  (mc.red.choose r * mc.white.choose w * mc.blue.choose b * mc.green.choose g * mc.yellow.choose y : ℚ) /
  (totalMarbles mc).choose 4

/-- Checks if the four specified events are equally likely -/
def eventsEquallyLikely (mc : MarbleCount) : Prop :=
  drawProbability mc 4 0 0 0 0 = drawProbability mc 3 1 0 0 0 ∧
  drawProbability mc 4 0 0 0 0 = drawProbability mc 1 1 1 0 1 ∧
  drawProbability mc 4 0 0 0 0 = drawProbability mc 1 1 1 1 0

/-- The main theorem stating the smallest number of marbles satisfying the conditions -/
theorem smallest_marble_count : ∃ (mc : MarbleCount), 
  eventsEquallyLikely mc ∧ 
  totalMarbles mc = 11 ∧ 
  (∀ (mc' : MarbleCount), eventsEquallyLikely mc' → totalMarbles mc' ≥ totalMarbles mc) :=
sorry

end smallest_marble_count_l3072_307214


namespace distance_between_roots_l3072_307212

/-- The distance between the roots of x^2 - 2x - 3 = 0 is 4 -/
theorem distance_between_roots : ∃ x₁ x₂ : ℝ, 
  x₁^2 - 2*x₁ - 3 = 0 ∧ 
  x₂^2 - 2*x₂ - 3 = 0 ∧ 
  x₁ ≠ x₂ ∧
  |x₁ - x₂| = 4 :=
by sorry

end distance_between_roots_l3072_307212


namespace trig_simplification_l3072_307262

theorem trig_simplification (α : ℝ) : 
  (1 + Real.sin (4 * α) - Real.cos (4 * α)) / (1 + Real.sin (4 * α) + Real.cos (4 * α)) = Real.tan (2 * α) := by
  sorry

end trig_simplification_l3072_307262


namespace exists_continuous_surjective_non_monotonic_l3072_307213

/-- A continuous function from ℝ to ℝ with full range that is not monotonic -/
theorem exists_continuous_surjective_non_monotonic :
  ∃ f : ℝ → ℝ, Continuous f ∧ Function.Surjective f ∧ ¬Monotone f := by
  sorry

end exists_continuous_surjective_non_monotonic_l3072_307213


namespace solution_of_equation_l3072_307263

theorem solution_of_equation (x : ℝ) : (1 / (3 * x) = 2 / (x + 5)) ↔ x = 1 :=
by sorry

end solution_of_equation_l3072_307263


namespace probability_identical_value_l3072_307236

/-- Represents the colors that can be used to paint a cube face -/
inductive Color
| Red
| Blue

/-- Represents a cube with painted faces -/
def Cube := Fin 6 → Color

/-- Checks if two cubes are identical after rotation -/
def identical_after_rotation (cube1 cube2 : Cube) : Prop := sorry

/-- The set of all possible cube paintings -/
def all_cubes : Set Cube := sorry

/-- The set of pairs of cubes that are identical after rotation -/
def identical_pairs : Set (Cube × Cube) := sorry

/-- The probability of two independently painted cubes being identical after rotation -/
def probability_identical : ℚ := sorry

theorem probability_identical_value :
  probability_identical = 459 / 4096 := by sorry

end probability_identical_value_l3072_307236


namespace quadratic_roots_l3072_307270

theorem quadratic_roots : ∃ (x₁ x₂ : ℝ), x₁ = 2 ∧ x₂ = 3 ∧ 
  (∀ x : ℝ, x^2 - 5*x + 6 = 0 ↔ (x = x₁ ∨ x = x₂)) := by
  sorry

end quadratic_roots_l3072_307270


namespace min_value_x_plus_y_l3072_307231

theorem min_value_x_plus_y (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : x^2 + 3*x*y - 2 = 0) :
  ∃ (m : ℝ), m = 4/3 ∧ ∀ (a b : ℝ), a > 0 → b > 0 → a^2 + 3*a*b - 2 = 0 → x + y ≤ a + b :=
sorry

end min_value_x_plus_y_l3072_307231


namespace reliability_comparison_l3072_307250

/-- Probability of a 3-member system making a correct decision -/
def prob_3_correct (p : ℝ) : ℝ := 3 * p^2 * (1 - p) + p^3

/-- Probability of a 5-member system making a correct decision -/
def prob_5_correct (p : ℝ) : ℝ := 10 * p^3 * (1 - p)^2 + 5 * p^4 * (1 - p) + p^5

/-- A 5-member system is more reliable than a 3-member system -/
def more_reliable (p : ℝ) : Prop := prob_5_correct p > prob_3_correct p

theorem reliability_comparison (p : ℝ) (h1 : 0 < p) (h2 : p < 1) :
  more_reliable p ↔ p > (1/2 : ℝ) := by sorry

end reliability_comparison_l3072_307250


namespace circle_angle_constraint_l3072_307283

-- Define the circle C
def C (x y : ℝ) : Prop := (x - 6)^2 + (y - 8)^2 = 1

-- Define points A and B
def A (m : ℝ) : ℝ × ℝ := (-m, 0)
def B (m : ℝ) : ℝ × ℝ := (m, 0)

-- Define the angle APB
def angle_APB (m : ℝ) (P : ℝ × ℝ) : ℝ := sorry

-- Main theorem
theorem circle_angle_constraint (m : ℝ) :
  m > 0 →
  (∀ P : ℝ × ℝ, C P.1 P.2 → angle_APB m P < 90) →
  9 < m ∧ m < 11 :=
sorry

end circle_angle_constraint_l3072_307283


namespace inequality_proof_l3072_307206

theorem inequality_proof (x : ℝ) (h : x > 0) : Real.exp (2 * x^3) - 2*x > 2*(x+1)*Real.log x := by
  sorry

end inequality_proof_l3072_307206


namespace sqrt_x_minus_one_real_l3072_307254

theorem sqrt_x_minus_one_real (x : ℝ) : (∃ y : ℝ, y ^ 2 = x - 1) ↔ x ≥ 1 := by sorry

end sqrt_x_minus_one_real_l3072_307254


namespace binomial_coefficient_equality_l3072_307225

theorem binomial_coefficient_equality (n : ℕ) : 
  (Nat.choose 18 (3*n + 6) = Nat.choose 18 (4*n - 2)) ↔ n = 2 :=
sorry

end binomial_coefficient_equality_l3072_307225


namespace shaded_area_semicircles_l3072_307252

/-- The area of the shaded region formed by semicircles in a given pattern -/
theorem shaded_area_semicircles (diameter : Real) (pattern_length_feet : Real) : 
  diameter = 3 →
  pattern_length_feet = 1.5 →
  (pattern_length_feet * 12 / diameter) * (π * (diameter / 2)^2 / 2) = 13.5 * π := by
  sorry

end shaded_area_semicircles_l3072_307252


namespace triangle_inequalities_triangle_equality_condition_l3072_307253

/-- Triangle properties -/
structure Triangle :=
  (a b c : ℝ)
  (r R : ℝ)
  (h_a h_b h_c : ℝ)
  (β_a β_b β_c : ℝ)
  (m_a m_b m_c : ℝ)
  (r_a r_b r_c : ℝ)
  (p : ℝ)

/-- Main theorem -/
theorem triangle_inequalities (t : Triangle) :
  (9 * t.r ≤ t.h_a + t.h_b + t.h_c) ∧
  (t.h_a + t.h_b + t.h_c ≤ t.β_a + t.β_b + t.β_c) ∧
  (t.β_a + t.β_b + t.β_c ≤ t.m_a + t.m_b + t.m_c) ∧
  (t.m_a + t.m_b + t.m_c ≤ 9/2 * t.R) ∧
  (t.β_a + t.β_b + t.β_c ≤ Real.sqrt (t.r_a * t.r_b) + Real.sqrt (t.r_b * t.r_c) + Real.sqrt (t.r_c * t.r_a)) ∧
  (Real.sqrt (t.r_a * t.r_b) + Real.sqrt (t.r_b * t.r_c) + Real.sqrt (t.r_c * t.r_a) ≤ t.p * Real.sqrt 3) ∧
  (t.p * Real.sqrt 3 ≤ t.r_a + t.r_b + t.r_c) ∧
  (t.r_a + t.r_b + t.r_c = t.r + 4 * t.R) ∧
  (27 * t.r^2 ≤ t.h_a^2 + t.h_b^2 + t.h_c^2) ∧
  (t.h_a^2 + t.h_b^2 + t.h_c^2 ≤ t.β_a^2 + t.β_b^2 + t.β_c^2) ∧
  (t.β_a^2 + t.β_b^2 + t.β_c^2 ≤ t.p^2) ∧
  (t.p^2 ≤ t.m_a^2 + t.m_b^2 + t.m_c^2) ∧
  (t.m_a^2 + t.m_b^2 + t.m_c^2 = 3/4 * (t.a^2 + t.b^2 + t.c^2)) ∧
  (3/4 * (t.a^2 + t.b^2 + t.c^2) ≤ 27/4 * t.R^2) ∧
  (1/t.r = 1/t.r_a + 1/t.r_b + 1/t.r_c) ∧
  (1/t.r = 1/t.h_a + 1/t.h_b + 1/t.h_c) ∧
  (1/t.h_a + 1/t.h_b + 1/t.h_c ≥ 1/t.β_a + 1/t.β_b + 1/t.β_c) ∧
  (1/t.β_a + 1/t.β_b + 1/t.β_c ≥ 1/t.m_a + 1/t.m_b + 1/t.m_c) ∧
  (1/t.m_a + 1/t.m_b + 1/t.m_c ≥ 2/t.R) :=
sorry

/-- Equality condition -/
theorem triangle_equality_condition (t : Triangle) :
  (9 * t.r = t.h_a + t.h_b + t.h_c) ∧
  (t.h_a + t.h_b + t.h_c = t.β_a + t.β_b + t.β_c) ∧
  (t.β_a + t.β_b + t.β_c = t.m_a + t.m_b + t.m_c) ∧
  (t.m_a + t.m_b + t.m_c = 9/2 * t.R) ∧
  (t.β_a + t.β_b + t.β_c = Real.sqrt (t.r_a * t.r_b) + Real.sqrt (t.r_b * t.r_c) + Real.sqrt (t.r_c * t.r_a)) ∧
  (Real.sqrt (t.r_a * t.r_b) + Real.sqrt (t.r_b * t.r_c) + Real.sqrt (t.r_c * t.r_a) = t.p * Real.sqrt 3) ∧
  (t.p * Real.sqrt 3 = t.r_a + t.r_b + t.r_c) ∧
  (27 * t.r^2 = t.h_a^2 + t.h_b^2 + t.h_c^2) ∧
  (t.h_a^2 + t.h_b^2 + t.h_c^2 = t.β_a^2 + t.β_b^2 + t.β_c^2) ∧
  (t.β_a^2 + t.β_b^2 + t.β_c^2 = t.p^2) ∧
  (t.p^2 = t.m_a^2 + t.m_b^2 + t.m_c^2) ∧
  (3/4 * (t.a^2 + t.b^2 + t.c^2) = 27/4 * t.R^2) ∧
  (1/t.h_a + 1/t.h_b + 1/t.h_c = 1/t.β_a + 1/t.β_b + 1/t.β_c) ∧
  (1/t.β_a + 1/t.β_b + 1/t.β_c = 1/t.m_a + 1/t.m_b + 1/t.m_c) ∧
  (1/t.m_a + 1/t.m_b + 1/t.m_c = 2/t.R) ↔
  (t.a = t.b ∧ t.b = t.c) :=
sorry

end triangle_inequalities_triangle_equality_condition_l3072_307253


namespace city_distance_proof_l3072_307223

/-- Calculates the actual distance between two cities given the map distance and scale. -/
def actual_distance (map_distance : ℝ) (scale : ℝ) : ℝ :=
  map_distance * scale

/-- Proves that the actual distance between two cities is 2400 km given the map conditions. -/
theorem city_distance_proof (map_distance : ℝ) (scale : ℝ) 
  (h1 : map_distance = 120)
  (h2 : scale = 20) : 
  actual_distance map_distance scale = 2400 := by
  sorry

#check city_distance_proof

end city_distance_proof_l3072_307223


namespace intersection_equals_open_closed_interval_l3072_307228

def A : Set ℝ := {x | x^2 - 4*x + 3 < 0}

def B : Set ℝ := {x | ∃ y, y = Real.log (x - 2)}

def C_R_B : Set ℝ := (Set.univ : Set ℝ) \ B

theorem intersection_equals_open_closed_interval : (C_R_B ∩ A) = {x | 1 < x ∧ x ≤ 2} := by sorry

end intersection_equals_open_closed_interval_l3072_307228


namespace log3_one_third_l3072_307287

-- Define the logarithm function for base 3
noncomputable def log3 (x : ℝ) := Real.log x / Real.log 3

-- State the theorem
theorem log3_one_third : log3 (1/3) = -1 := by
  sorry

end log3_one_third_l3072_307287


namespace fifth_term_zero_l3072_307251

/-- An arithmetic sequence {a_n} -/
def arithmeticSequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n - d

/-- The sequence is decreasing -/
def decreasingSequence (a : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, a (n + 1) < a n

theorem fifth_term_zero
  (a : ℕ → ℝ)
  (h_arith : arithmeticSequence a)
  (h_decr : decreasingSequence a)
  (h_eq : a 1 ^ 2 = a 9 ^ 2) :
  a 5 = 0 :=
sorry

end fifth_term_zero_l3072_307251


namespace isosceles_not_unique_l3072_307232

/-- Represents a triangle with side lengths a, b, c and angles A, B, C --/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- Predicate to check if a triangle is isosceles --/
def IsIsosceles (t : Triangle) : Prop :=
  t.a = t.b ∨ t.b = t.c ∨ t.c = t.a

/-- Predicate to check if two triangles are non-congruent --/
def AreNonCongruent (t1 t2 : Triangle) : Prop :=
  t1.a ≠ t2.a ∨ t1.b ≠ t2.b ∨ t1.c ≠ t2.c

/-- Theorem stating that a base angle and opposite side do not uniquely determine an isosceles triangle --/
theorem isosceles_not_unique (θ : ℝ) (s : ℝ) :
  ∃ (t1 t2 : Triangle), IsIsosceles t1 ∧ IsIsosceles t2 ∧ 
  AreNonCongruent t1 t2 ∧
  ((t1.A = θ ∧ t1.a = s) ∨ (t1.B = θ ∧ t1.b = s) ∨ (t1.C = θ ∧ t1.c = s)) ∧
  ((t2.A = θ ∧ t2.a = s) ∨ (t2.B = θ ∧ t2.b = s) ∨ (t2.C = θ ∧ t2.c = s)) :=
sorry

end isosceles_not_unique_l3072_307232


namespace select_students_l3072_307240

theorem select_students (n_boys : ℕ) (n_girls : ℕ) (n_select : ℕ) : 
  n_boys = 4 → n_girls = 2 → n_select = 4 →
  (Nat.choose n_boys (n_select - 1) * Nat.choose n_girls 1 + 
   Nat.choose n_boys (n_select - 2) * Nat.choose n_girls 2) = 14 := by
sorry

end select_students_l3072_307240


namespace arithmetic_sequence_length_l3072_307255

theorem arithmetic_sequence_length : 
  ∀ (a d last : ℕ), 
  a = 3 → d = 3 → last = 198 → 
  ∃ n : ℕ, n = 66 ∧ last = a + (n - 1) * d :=
by sorry

end arithmetic_sequence_length_l3072_307255


namespace average_score_calculation_l3072_307279

theorem average_score_calculation (total_students : ℕ) (male_students : ℕ) (female_students : ℕ)
  (male_avg_score : ℝ) (female_avg_score : ℝ) :
  male_students = (0.4 : ℝ) * total_students →
  female_students = total_students - male_students →
  male_avg_score = 75 →
  female_avg_score = 80 →
  (male_avg_score * male_students + female_avg_score * female_students) / total_students = 78 :=
by
  sorry

#check average_score_calculation

end average_score_calculation_l3072_307279


namespace square_root_sum_l3072_307201

theorem square_root_sum (a b : ℕ+) : 
  (Real.sqrt (7 + a / b) = 7 * Real.sqrt (a / b)) → a + b = 55 := by
  sorry

end square_root_sum_l3072_307201


namespace second_car_rate_l3072_307245

/-- Given two cars starting at the same point, with the first car traveling at 50 mph,
    and after 3 hours the distance between them is 30 miles,
    prove that the rate of the second car is 40 mph. -/
theorem second_car_rate (v : ℝ) : 
  v > 0 →  -- The rate of the second car is positive
  50 * 3 - v * 3 = 30 →  -- After 3 hours, the distance between the cars is 30 miles
  v = 40 := by
sorry

end second_car_rate_l3072_307245


namespace work_completion_l3072_307239

theorem work_completion (days_group1 : ℕ) (men_group2 : ℕ) (days_group2 : ℕ) :
  days_group1 = 18 →
  men_group2 = 27 →
  days_group2 = 24 →
  ∃ men_group1 : ℕ, men_group1 * days_group1 = men_group2 * days_group2 ∧ men_group1 = 36 := by
  sorry

end work_completion_l3072_307239


namespace prob_D_is_one_fourth_l3072_307276

/-- A spinner with four regions -/
structure Spinner :=
  (probA : ℚ)
  (probB : ℚ)
  (probC : ℚ)
  (probD : ℚ)

/-- The properties of our specific spinner -/
def spinner : Spinner :=
  { probA := 1/4
  , probB := 1/3
  , probC := 1/6
  , probD := 1/4 }

/-- The sum of probabilities in a spinner must equal 1 -/
axiom probability_sum (s : Spinner) : s.probA + s.probB + s.probC + s.probD = 1

/-- Theorem: Given the probabilities of A, B, and C, the probability of D is 1/4 -/
theorem prob_D_is_one_fourth :
  spinner.probA = 1/4 → spinner.probB = 1/3 → spinner.probC = 1/6 →
  spinner.probD = 1/4 := by
  sorry

end prob_D_is_one_fourth_l3072_307276


namespace sum_of_max_and_min_is_two_l3072_307208

-- Define the function f
def f (x : ℝ) : ℝ := |x - 3| + |x - 5| - |2*x - 8|

-- Define the domain
def domain : Set ℝ := { x | 3 ≤ x ∧ x ≤ 10 }

-- State the theorem
theorem sum_of_max_and_min_is_two :
  ∃ (max min : ℝ), 
    (∀ x ∈ domain, f x ≤ max) ∧
    (∃ x ∈ domain, f x = max) ∧
    (∀ x ∈ domain, min ≤ f x) ∧
    (∃ x ∈ domain, f x = min) ∧
    max + min = 2 := by
  sorry

end sum_of_max_and_min_is_two_l3072_307208


namespace initial_geese_count_l3072_307258

/-- Given that 28 geese flew away and 23 geese remain in a field,
    prove that there were initially 51 geese in the field. -/
theorem initial_geese_count (flew_away : ℕ) (remaining : ℕ) : 
  flew_away = 28 → remaining = 23 → flew_away + remaining = 51 := by
  sorry

end initial_geese_count_l3072_307258


namespace john_travel_time_l3072_307275

/-- Proves that given a distance of 24 km and a normal travel time of 44 minutes,
    if a speed of 40 kmph results in arriving 8 minutes early,
    then a speed of 30 kmph will result in arriving 4 minutes late. -/
theorem john_travel_time (distance : ℝ) (normal_time : ℝ) (early_speed : ℝ) (late_speed : ℝ) :
  distance = 24 →
  normal_time = 44 / 60 →
  early_speed = 40 →
  late_speed = 30 →
  distance / early_speed = normal_time - 8 / 60 →
  distance / late_speed = normal_time + 4 / 60 :=
by sorry

end john_travel_time_l3072_307275


namespace exponential_inequality_l3072_307220

theorem exponential_inequality (a b : ℝ) (h1 : a > b) (h2 : b > 0) :
  (1/2 : ℝ)^a < (1/2 : ℝ)^b := by
  sorry

end exponential_inequality_l3072_307220


namespace multiply_powers_of_a_l3072_307286

theorem multiply_powers_of_a (a : ℝ) : -2 * a^3 * (3 * a^2) = -6 * a^5 := by
  sorry

end multiply_powers_of_a_l3072_307286


namespace opinion_change_range_l3072_307291

def initial_yes : ℝ := 40
def initial_no : ℝ := 60
def final_yes : ℝ := 60
def final_no : ℝ := 40

theorem opinion_change_range :
  let min_change := |final_yes - initial_yes|
  let max_change := min initial_yes initial_no + min final_yes final_no
  max_change - min_change = 40 := by sorry

end opinion_change_range_l3072_307291


namespace treys_chores_l3072_307218

theorem treys_chores (clean_house_tasks : ℕ) (shower_tasks : ℕ) (dinner_tasks : ℕ) 
  (total_time_hours : ℕ) (h1 : clean_house_tasks = 7) (h2 : shower_tasks = 1) 
  (h3 : dinner_tasks = 4) (h4 : total_time_hours = 2) : 
  (total_time_hours * 60) / (clean_house_tasks + shower_tasks + dinner_tasks) = 10 := by
  sorry

end treys_chores_l3072_307218


namespace inequality_proof_l3072_307285

theorem inequality_proof (x y : ℝ) (p q : ℕ) 
  (hx : x > 0) (hy : y > 0) (hxy : x ≠ y) :
  (x^(-(p:ℝ)/q) - y^((p:ℝ)/q) * x^(-(2*p:ℝ)/q)) / 
  (x^((1-2*p:ℝ)/q) - y^((1:ℝ)/q) * x^(-(2*p:ℝ)/q)) > 
  p * (x*y)^((p-1:ℝ)/(2*q)) :=
sorry

end inequality_proof_l3072_307285


namespace parallel_planes_from_skew_lines_l3072_307219

-- Define the types for lines and planes in 3D space
variable (Line Plane : Type)

-- Define the parallel relation between lines and planes
variable (parallel : Line → Plane → Prop)

-- Define the parallel relation between planes
variable (planeParallel : Plane → Plane → Prop)

-- Define the skew relation between lines
variable (skew : Line → Line → Prop)

-- Theorem statement
theorem parallel_planes_from_skew_lines 
  (m n : Line) (α β : Plane) 
  (h_skew : skew m n) 
  (h_m_α : parallel m α) (h_n_α : parallel n α) 
  (h_m_β : parallel m β) (h_n_β : parallel n β) : 
  planeParallel α β :=
sorry

end parallel_planes_from_skew_lines_l3072_307219


namespace middle_marble_radius_l3072_307244

/-- Given a sequence of five marbles with radii forming a geometric sequence,
    where the smallest radius is 8 and the largest radius is 18,
    prove that the middle (third) marble has a radius of 12. -/
theorem middle_marble_radius 
  (r : Fin 5 → ℝ)  -- r is a function mapping the index of each marble to its radius
  (h_geom_seq : ∀ i j k, i < j → j < k → r j ^ 2 = r i * r k)  -- geometric sequence condition
  (h_smallest : r 0 = 8)  -- radius of the smallest marble
  (h_largest : r 4 = 18)  -- radius of the largest marble
  : r 2 = 12 := by  -- radius of the middle (third) marble
sorry


end middle_marble_radius_l3072_307244


namespace vector_c_value_l3072_307222

def a : ℝ × ℝ := (2, -1)
def b : ℝ × ℝ := (-3, 5)

theorem vector_c_value (c : ℝ × ℝ) : 
  3 • a + (4 • b - a) + 2 • c = (0, 0) → c = (4, -9) := by
  sorry

end vector_c_value_l3072_307222


namespace speed_of_sound_calculation_l3072_307292

/-- The speed of sound in meters per second -/
def speed_of_sound : ℝ := 330

/-- The time between hearing the first and second blast in seconds -/
def time_between_blasts : ℝ := 30 * 60 + 24

/-- The time between the occurrence of the first and second blast in seconds -/
def time_between_blast_occurrences : ℝ := 30 * 60

/-- The distance from the blast site when hearing the second blast in meters -/
def distance_at_second_blast : ℝ := 7920

/-- Theorem stating that the speed of sound is 330 m/s given the problem conditions -/
theorem speed_of_sound_calculation :
  speed_of_sound = distance_at_second_blast / (time_between_blasts - time_between_blast_occurrences) :=
by sorry

end speed_of_sound_calculation_l3072_307292


namespace function_maximum_implies_a_range_l3072_307216

/-- Given a function f(x) = 4x³ - 3x with a maximum in the interval (a, a+2), prove that a is in the range (-5/2, -1]. -/
theorem function_maximum_implies_a_range 
  (f : ℝ → ℝ) 
  (a : ℝ) 
  (h₁ : ∀ x, f x = 4 * x^3 - 3 * x)
  (h₂ : ∃ x₀ ∈ Set.Ioo a (a + 2), ∀ x ∈ Set.Ioo a (a + 2), f x ≤ f x₀) :
  a ∈ Set.Ioc (-5/2) (-1) :=
sorry

end function_maximum_implies_a_range_l3072_307216


namespace fast_food_cost_l3072_307278

theorem fast_food_cost (burger shake cola : ℝ) : 
  (3 * burger + 7 * shake + cola = 120) →
  (4 * burger + 10 * shake + cola = 160.5) →
  (burger + shake + cola = 39) :=
by sorry

end fast_food_cost_l3072_307278


namespace f_composition_value_l3072_307238

noncomputable def f (x : ℝ) : ℝ :=
  if x < 0 then 2^(x + 2) else x^3

theorem f_composition_value : f (f (-1)) = 8 := by sorry

end f_composition_value_l3072_307238


namespace product_of_specific_primes_l3072_307248

def smallest_one_digit_primes : List Nat := [2, 3]
def largest_two_digit_prime : Nat := 97

theorem product_of_specific_primes :
  (smallest_one_digit_primes.prod * largest_two_digit_prime) = 582 := by
  sorry

end product_of_specific_primes_l3072_307248


namespace solution_equation1_solution_equation2_l3072_307209

-- Define the equations
def equation1 (x : ℝ) : Prop := 2 * x^2 = 8 * x
def equation2 (y : ℝ) : Prop := y^2 - 10 * y - 1 = 0

-- Theorem for the first equation
theorem solution_equation1 : 
  (∃ x : ℝ, equation1 x) ∧ 
  (∀ x : ℝ, equation1 x ↔ (x = 0 ∨ x = 4)) :=
sorry

-- Theorem for the second equation
theorem solution_equation2 : 
  (∃ y : ℝ, equation2 y) ∧ 
  (∀ y : ℝ, equation2 y ↔ (y = 5 + Real.sqrt 26 ∨ y = 5 - Real.sqrt 26)) :=
sorry

end solution_equation1_solution_equation2_l3072_307209


namespace intersection_slope_l3072_307246

/-- Given two lines m and n that intersect at (-4, 0), prove that the slope of line n is -9/4 -/
theorem intersection_slope (k : ℚ) : 
  (∀ x y, y = 2 * x + 8 → y = k * x - 9 → x = -4 ∧ y = 0) → 
  k = -9/4 := by
  sorry

end intersection_slope_l3072_307246


namespace toms_flying_robots_l3072_307282

theorem toms_flying_robots (michael_robots : ℕ) (tom_robots : ℕ) : 
  michael_robots = 12 →
  michael_robots = 4 * tom_robots →
  tom_robots = 3 := by
sorry

end toms_flying_robots_l3072_307282


namespace perimeter_after_adding_tiles_l3072_307224

/-- A figure composed of square tiles -/
structure TiledFigure where
  tiles : ℕ
  perimeter : ℕ

/-- Adds tiles to a figure, each sharing at least one side with the original figure -/
def add_tiles (figure : TiledFigure) (new_tiles : ℕ) : TiledFigure :=
  { tiles := figure.tiles + new_tiles,
    perimeter := figure.perimeter + 2 * new_tiles }

theorem perimeter_after_adding_tiles (initial_figure : TiledFigure) :
  initial_figure.tiles = 10 →
  initial_figure.perimeter = 16 →
  (add_tiles initial_figure 4).perimeter = 20 := by
  sorry

end perimeter_after_adding_tiles_l3072_307224


namespace fiftieth_term_is_448_l3072_307295

/-- Checks if a natural number contains the digit 4 --/
def containsDigitFour (n : ℕ) : Bool :=
  n.repr.any (· = '4')

/-- The sequence of positive multiples of 4 that contain at least one digit 4 --/
def specialSequence : ℕ → ℕ
  | 0 => 4  -- The first term is always 4
  | n + 1 => 
      let next := specialSequence n + 4
      if containsDigitFour next then next
      else specialSequence (n + 1)

/-- The 50th term of the special sequence is 448 --/
theorem fiftieth_term_is_448 : specialSequence 49 = 448 := by
  sorry

#eval specialSequence 49  -- This should output 448

end fiftieth_term_is_448_l3072_307295


namespace min_value_theorem_l3072_307274

theorem min_value_theorem (x y : ℝ) (h : x * y > 0) :
  ∃ m : ℝ, m = 4 - 2 * Real.sqrt 2 ∧
    ∀ z : ℝ, z = y / (x + y) + 2 * x / (2 * x + y) → z ≥ m := by
  sorry

end min_value_theorem_l3072_307274


namespace words_per_page_l3072_307226

theorem words_per_page (total_pages : Nat) (max_words_per_page : Nat) (total_words_mod : Nat) :
  total_pages = 136 →
  max_words_per_page = 100 →
  total_words_mod = 184 →
  ∃ (words_per_page : Nat),
    words_per_page ≤ max_words_per_page ∧
    (total_pages * words_per_page) % 203 = total_words_mod ∧
    words_per_page = 73 := by
  sorry

end words_per_page_l3072_307226


namespace tan_alpha_eq_one_l3072_307242

theorem tan_alpha_eq_one (α : Real) 
  (h : (Real.sin α + Real.cos α) / (2 * Real.sin α - Real.cos α) = 2) : 
  Real.tan α = 1 := by
  sorry

end tan_alpha_eq_one_l3072_307242


namespace room_width_calculation_l3072_307207

/-- Given a room with specified dimensions and paving costs, calculate its width -/
theorem room_width_calculation (length : ℝ) (cost_per_sqm : ℝ) (total_cost : ℝ) 
  (h1 : length = 5.5)
  (h2 : cost_per_sqm = 600)
  (h3 : total_cost = 12375) :
  total_cost / cost_per_sqm / length = 3.75 := by
  sorry

#check room_width_calculation

end room_width_calculation_l3072_307207


namespace real_part_of_z_l3072_307256

theorem real_part_of_z (z : ℂ) (h1 : Complex.abs z = 1) (h2 : Complex.abs (z - 1.45) = 1.05) :
  z.re = 20 / 29 := by
  sorry

end real_part_of_z_l3072_307256


namespace solution_pair_l3072_307264

theorem solution_pair : ∃ (x y : ℝ), 
  (2 * x + 3 * y = (7 - x) + (7 - y)) ∧ 
  (x - 2 * y = (x - 3) + (y - 3)) ∧ 
  x = 2 ∧ y = 2 := by
  sorry

end solution_pair_l3072_307264


namespace allStarSeatingArrangements_l3072_307272

def factorial (n : ℕ) : ℕ := (List.range n).foldl (·*·+1) 1

-- Define the number of All-Stars for each team
def cubs : ℕ := 4
def redSox : ℕ := 3
def yankees : ℕ := 2

-- Define the total number of All-Stars
def totalAllStars : ℕ := cubs + redSox + yankees

-- Define the number of team blocks (excluding the fixed block)
def remainingTeamBlocks : ℕ := 2

theorem allStarSeatingArrangements :
  factorial remainingTeamBlocks * factorial cubs * factorial redSox * factorial yankees = 576 := by
  sorry

end allStarSeatingArrangements_l3072_307272


namespace measure_union_ge_sum_measures_l3072_307294

open MeasureTheory Set

-- Define the algebra structure
variable {α : Type*} [MeasurableSpace α]

-- Define the measure
variable (μ : Measure α)

-- Define the sequence of sets
variable (A : ℕ → Set α)

-- State the theorem
theorem measure_union_ge_sum_measures
  (h_algebra : ∀ n, MeasurableSet (A n))
  (h_disjoint : Pairwise (Disjoint on A))
  (h_union : MeasurableSet (⋃ n, A n)) :
  μ (⋃ n, A n) ≥ ∑' n, μ (A n) :=
sorry

end measure_union_ge_sum_measures_l3072_307294


namespace erica_ride_time_l3072_307205

/-- The time Dave can ride the merry-go-round in minutes -/
def dave_time : ℝ := 10

/-- The factor by which Chuck can ride longer than Dave -/
def chuck_factor : ℝ := 5

/-- The percentage longer Erica can stay compared to Chuck -/
def erica_percentage : ℝ := 0.3

/-- The time Chuck can ride the merry-go-round in minutes -/
def chuck_time : ℝ := dave_time * chuck_factor

/-- The time Erica can ride the merry-go-round in minutes -/
def erica_time : ℝ := chuck_time * (1 + erica_percentage)

theorem erica_ride_time : erica_time = 65 := by
  sorry

end erica_ride_time_l3072_307205


namespace parallelogram_properties_l3072_307249

/-- Represents a parallelogram with given dimensions -/
structure Parallelogram where
  base : ℝ
  height : ℝ
  total_side : ℝ

/-- Calculate the area of a parallelogram -/
def area (p : Parallelogram) : ℝ := p.base * p.height

/-- Calculate the slant height of a parallelogram -/
def slant_height (p : Parallelogram) : ℝ := p.total_side - p.height

theorem parallelogram_properties (p : Parallelogram) 
  (h_base : p.base = 20)
  (h_height : p.height = 6)
  (h_total_side : p.total_side = 9) :
  area p = 120 ∧ slant_height p = 3 := by
  sorry


end parallelogram_properties_l3072_307249


namespace complex_modulus_problem_l3072_307237

theorem complex_modulus_problem (i : ℂ) (h : i * i = -1) :
  Complex.abs (1 / (1 - i) + i) = Real.sqrt 10 / 2 := by
  sorry

end complex_modulus_problem_l3072_307237


namespace five_balls_four_boxes_l3072_307234

/-- The number of ways to distribute indistinguishable balls into indistinguishable boxes -/
def distribute_balls (n : ℕ) (k : ℕ) : ℕ := sorry

/-- Theorem: There are 6 ways to distribute 5 indistinguishable balls into 4 indistinguishable boxes -/
theorem five_balls_four_boxes : distribute_balls 5 4 = 6 := by sorry

end five_balls_four_boxes_l3072_307234


namespace fence_cost_per_foot_l3072_307203

theorem fence_cost_per_foot 
  (area : ℝ) 
  (total_cost : ℝ) 
  (h1 : area = 81) 
  (h2 : total_cost = 2088) : 
  (total_cost / (4 * Real.sqrt area)) = 58 := by
  sorry

end fence_cost_per_foot_l3072_307203


namespace student_height_correction_l3072_307268

theorem student_height_correction (n : ℕ) (initial_avg : ℝ) (incorrect_height : ℝ) (actual_avg : ℝ) :
  n = 20 →
  initial_avg = 175 →
  incorrect_height = 151 →
  actual_avg = 174.25 →
  ∃ (actual_height : ℝ), 
    actual_height = 166 ∧
    n * initial_avg = (n - 1) * actual_avg + incorrect_height ∧
    n * actual_avg = (n - 1) * actual_avg + actual_height :=
by sorry

end student_height_correction_l3072_307268


namespace complex_power_difference_abs_l3072_307202

def i : ℂ := Complex.I

theorem complex_power_difference_abs : 
  Complex.abs ((2 + i)^18 - (2 - i)^18) = 19531250 := by sorry

end complex_power_difference_abs_l3072_307202


namespace power_function_through_point_value_l3072_307200

-- Define a power function
def isPowerFunction (f : ℝ → ℝ) : Prop :=
  ∃ a b : ℝ, ∀ x : ℝ, f x = a * x ^ b

theorem power_function_through_point_value :
  ∀ f : ℝ → ℝ,
  isPowerFunction f →
  f 2 = 8 →
  f 3 = 27 := by
sorry

end power_function_through_point_value_l3072_307200


namespace cell_phone_customers_l3072_307267

theorem cell_phone_customers (total : ℕ) (us_customers : ℕ) 
  (h1 : total = 7422) 
  (h2 : us_customers = 723) : 
  total - us_customers = 6699 := by
  sorry

end cell_phone_customers_l3072_307267


namespace age_difference_l3072_307235

/-- The difference in total ages of (A,B) and (B,C) given C is 10 years younger than A -/
theorem age_difference (A B C : ℕ) (h : C = A - 10) : A + B - (B + C) = 10 := by
  sorry

end age_difference_l3072_307235


namespace sock_profit_percentage_l3072_307293

/-- Calculates the percentage profit on 4 pairs of socks given the following conditions:
  * 9 pairs of socks were bought
  * Each pair costs $2
  * $0.2 profit is made on 5 pairs
  * Total profit is $3
-/
theorem sock_profit_percentage 
  (total_pairs : Nat) 
  (cost_per_pair : ℚ) 
  (profit_on_five : ℚ) 
  (total_profit : ℚ) 
  (h1 : total_pairs = 9)
  (h2 : cost_per_pair = 2)
  (h3 : profit_on_five = 5 * (1 / 5))
  (h4 : total_profit = 3) :
  let remaining_pairs := total_pairs - 5
  let remaining_profit := total_profit - profit_on_five
  let remaining_cost := remaining_pairs * cost_per_pair
  (remaining_profit / remaining_cost) * 100 = 25 := by
sorry

end sock_profit_percentage_l3072_307293
