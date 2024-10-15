import Mathlib

namespace NUMINAMATH_CALUDE_wind_speed_calculation_l3253_325307

/-- Wind speed calculation for a helicopter flight --/
theorem wind_speed_calculation 
  (s v : ℝ) 
  (h_positive_s : 0 < s)
  (h_positive_v : 0 < v)
  (h_v_greater_s : s < v) :
  ∃ (x y vB : ℝ),
    x + y = 2 ∧                 -- Total flight time is 2 hours
    v + vB = s / x ∧            -- Speed from A to B (with wind)
    v - vB = s / y ∧            -- Speed from B to A (against wind)
    vB = Real.sqrt (v * (v - s)) -- Wind speed formula
  := by sorry

end NUMINAMATH_CALUDE_wind_speed_calculation_l3253_325307


namespace NUMINAMATH_CALUDE_square_has_most_symmetry_l3253_325335

-- Define the types of figures
inductive Figure
  | EquilateralTriangle
  | NonSquareRhombus
  | NonSquareRectangle
  | IsoscelesTrapezoid
  | Square

-- Function to get the number of lines of symmetry for each figure
def linesOfSymmetry (f : Figure) : ℕ :=
  match f with
  | Figure.EquilateralTriangle => 3
  | Figure.NonSquareRhombus => 2
  | Figure.NonSquareRectangle => 2
  | Figure.IsoscelesTrapezoid => 1
  | Figure.Square => 4

-- Theorem stating that the square has the greatest number of lines of symmetry
theorem square_has_most_symmetry :
  ∀ f : Figure, linesOfSymmetry Figure.Square ≥ linesOfSymmetry f :=
by
  sorry


end NUMINAMATH_CALUDE_square_has_most_symmetry_l3253_325335


namespace NUMINAMATH_CALUDE_sugar_packs_theorem_l3253_325390

/-- Given the total amount of sugar, weight per pack, and leftover sugar, 
    calculate the number of packs. -/
def calculate_packs (total_sugar : ℕ) (weight_per_pack : ℕ) (leftover_sugar : ℕ) : ℕ :=
  (total_sugar - leftover_sugar) / weight_per_pack

/-- Theorem stating that given the specific conditions, 
    the number of packs is 12. -/
theorem sugar_packs_theorem (total_sugar weight_per_pack leftover_sugar : ℕ) 
  (h1 : total_sugar = 3020)
  (h2 : weight_per_pack = 250)
  (h3 : leftover_sugar = 20) :
  calculate_packs total_sugar weight_per_pack leftover_sugar = 12 := by
  sorry

#eval calculate_packs 3020 250 20

end NUMINAMATH_CALUDE_sugar_packs_theorem_l3253_325390


namespace NUMINAMATH_CALUDE_corn_kernel_weight_theorem_l3253_325329

/-- Calculates the total weight of corn kernels after shucking and accounting for losses -/
def corn_kernel_weight (
  ears_per_stalk : ℕ)
  (total_stalks : ℕ)
  (bad_ear_percentage : ℚ)
  (kernel_distribution : List (ℚ × ℕ))
  (kernel_weight : ℚ)
  (lost_kernel_percentage : ℚ) : ℚ :=
  let total_ears := ears_per_stalk * total_stalks
  let good_ears := total_ears - (bad_ear_percentage * total_ears).floor
  let total_kernels := (kernel_distribution.map (fun (p, k) => 
    ((p * good_ears).floor * k))).sum
  let kernels_after_loss := total_kernels - 
    (lost_kernel_percentage * total_kernels).floor
  kernels_after_loss * kernel_weight

/-- The total weight of corn kernels is approximately 18527.9 grams -/
theorem corn_kernel_weight_theorem : 
  ∃ (ε : ℚ), ε > 0 ∧ ε < 0.1 ∧ 
  |corn_kernel_weight 4 108 (1/5) 
    [(3/5, 500), (3/10, 600), (1/10, 700)] (1/10) (3/200) - 18527.9| < ε :=
sorry

end NUMINAMATH_CALUDE_corn_kernel_weight_theorem_l3253_325329


namespace NUMINAMATH_CALUDE_train_length_l3253_325354

/-- The length of a train given its speed and time to cross a fixed point. -/
theorem train_length (speed : ℝ) (time : ℝ) : 
  speed = 72 * (5 / 18) → time = 16 → speed * time = 320 := by
  sorry

#check train_length

end NUMINAMATH_CALUDE_train_length_l3253_325354


namespace NUMINAMATH_CALUDE_normal_distribution_symmetry_l3253_325311

/-- A random variable following a normal distribution -/
structure NormalRV where
  μ : ℝ
  σ : ℝ
  hσ : σ > 0

/-- Probability function for a normal random variable -/
noncomputable def P (ξ : NormalRV) (x : ℝ) : ℝ := sorry

theorem normal_distribution_symmetry 
  (ξ : NormalRV) 
  (h1 : ξ.μ = 2) 
  (h2 : P ξ 4 = 0.84) : 
  P ξ 0 = 0.16 := by sorry

end NUMINAMATH_CALUDE_normal_distribution_symmetry_l3253_325311


namespace NUMINAMATH_CALUDE_system_solution_exists_and_unique_l3253_325350

theorem system_solution_exists_and_unique :
  ∃! (x y z : ℝ), 
    8 * (x^3 + y^3 + z^3) = 73 ∧
    2 * (x^2 + y^2 + z^2) = 3 * (x*y + y*z + z*x) ∧
    x * y * z = 1 ∧
    x = 1 ∧ y = 2 ∧ z = (1/2 : ℝ) := by
  sorry

end NUMINAMATH_CALUDE_system_solution_exists_and_unique_l3253_325350


namespace NUMINAMATH_CALUDE_max_value_m_l3253_325340

theorem max_value_m (a b m : ℝ) (ha : a > 0) (hb : b > 0) :
  (∀ a b, (3/a + 1/b ≥ m/(a + 3*b))) → m ≤ 12 :=
by sorry

end NUMINAMATH_CALUDE_max_value_m_l3253_325340


namespace NUMINAMATH_CALUDE_polynomial_factorization_l3253_325344

theorem polynomial_factorization (x : ℝ) : 
  x^6 - 4*x^4 + 6*x^2 - 4 = (x-1)*(x+1)*(x-Real.sqrt 2)^2*(x+Real.sqrt 2)^2 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_factorization_l3253_325344


namespace NUMINAMATH_CALUDE_consecutive_color_draw_probability_l3253_325304

def num_tan_chips : ℕ := 4
def num_pink_chips : ℕ := 4
def num_violet_chips : ℕ := 3
def total_chips : ℕ := num_tan_chips + num_pink_chips + num_violet_chips

theorem consecutive_color_draw_probability :
  (2 * (num_tan_chips.factorial * num_pink_chips.factorial * num_violet_chips.factorial)) / 
  total_chips.factorial = 1 / 5760 := by sorry

end NUMINAMATH_CALUDE_consecutive_color_draw_probability_l3253_325304


namespace NUMINAMATH_CALUDE_average_percentage_decrease_l3253_325323

theorem average_percentage_decrease (initial_price final_price : ℝ) 
  (h1 : initial_price = 800)
  (h2 : final_price = 578)
  (h3 : final_price = initial_price * (1 - x)^2)
  : x = 0.15 := by
  sorry

end NUMINAMATH_CALUDE_average_percentage_decrease_l3253_325323


namespace NUMINAMATH_CALUDE_inscribed_cube_diagonal_l3253_325366

/-- The diagonal length of a cube inscribed in a sphere of radius R is 2R -/
theorem inscribed_cube_diagonal (R : ℝ) (R_pos : R > 0) :
  ∃ (cube : Set (Fin 3 → ℝ)), 
    (∀ p ∈ cube, ‖p‖ = R) ∧ 
    (∃ (d : Fin 3 → ℝ), d ∈ cube ∧ ‖d‖ = 2*R) :=
sorry

end NUMINAMATH_CALUDE_inscribed_cube_diagonal_l3253_325366


namespace NUMINAMATH_CALUDE_escalator_travel_time_l3253_325339

-- Define the escalator properties and person's walking speed
def escalator_speed : ℝ := 11
def escalator_length : ℝ := 126
def person_speed : ℝ := 3

-- Theorem statement
theorem escalator_travel_time :
  let combined_speed := escalator_speed + person_speed
  let time := escalator_length / combined_speed
  time = 9 := by sorry

end NUMINAMATH_CALUDE_escalator_travel_time_l3253_325339


namespace NUMINAMATH_CALUDE_geometric_sequence_a7_l3253_325361

/-- A geometric sequence with positive terms -/
def GeometricSequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, r > 0 ∧ ∀ n : ℕ, a (n + 1) = r * a n

theorem geometric_sequence_a7 (a : ℕ → ℝ) :
  GeometricSequence a →
  (∀ n : ℕ, a n > 0) →
  a 3 * a 11 = 16 →
  a 7 = 4 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_a7_l3253_325361


namespace NUMINAMATH_CALUDE_paint_cost_per_liter_l3253_325332

/-- Calculates the cost of paint per liter given the costs of materials and profit --/
theorem paint_cost_per_liter 
  (brush_cost : ℚ) 
  (canvas_cost_multiplier : ℚ) 
  (min_paint_liters : ℚ) 
  (selling_price : ℚ) 
  (profit : ℚ) 
  (h1 : brush_cost = 20)
  (h2 : canvas_cost_multiplier = 3)
  (h3 : min_paint_liters = 5)
  (h4 : selling_price = 200)
  (h5 : profit = 80) :
  (selling_price - profit - (brush_cost + canvas_cost_multiplier * brush_cost)) / min_paint_liters = 8 := by
  sorry

end NUMINAMATH_CALUDE_paint_cost_per_liter_l3253_325332


namespace NUMINAMATH_CALUDE_cheryl_material_usage_l3253_325322

theorem cheryl_material_usage
  (material1_bought : ℚ)
  (material2_bought : ℚ)
  (material3_bought : ℚ)
  (material1_left : ℚ)
  (material2_left : ℚ)
  (material3_left : ℚ)
  (h1 : material1_bought = 4/9)
  (h2 : material2_bought = 2/3)
  (h3 : material3_bought = 5/6)
  (h4 : material1_left = 8/18)
  (h5 : material2_left = 3/9)
  (h6 : material3_left = 2/12) :
  (material1_bought - material1_left) + (material2_bought - material2_left) + (material3_bought - material3_left) = 1 := by
  sorry

#check cheryl_material_usage

end NUMINAMATH_CALUDE_cheryl_material_usage_l3253_325322


namespace NUMINAMATH_CALUDE_james_chocolate_sales_l3253_325388

/-- Calculates the number of chocolate bars James sold this week -/
def chocolate_bars_sold_this_week (total : ℕ) (sold_last_week : ℕ) (to_sell : ℕ) : ℕ :=
  total - (sold_last_week + to_sell)

/-- Proves that James sold 2 chocolate bars this week -/
theorem james_chocolate_sales : chocolate_bars_sold_this_week 18 5 6 = 2 := by
  sorry

end NUMINAMATH_CALUDE_james_chocolate_sales_l3253_325388


namespace NUMINAMATH_CALUDE_correct_sampling_methods_l3253_325352

/-- Represents different types of sampling methods -/
inductive SamplingMethod
  | SimpleRandom
  | Systematic
  | Stratified

/-- Represents a sampling technique -/
structure SamplingTechnique where
  method : SamplingMethod
  description : String

/-- Represents the survey conducted by the school -/
structure Survey where
  totalStudents : Nat
  technique1 : SamplingTechnique
  technique2 : SamplingTechnique

/-- The actual survey conducted by the school -/
def schoolSurvey : Survey :=
  { totalStudents := 200,
    technique1 := 
      { method := SamplingMethod.SimpleRandom,
        description := "Random selection of 20 students by the student council" },
    technique2 := 
      { method := SamplingMethod.Systematic,
        description := "Students numbered from 001 to 200, those with last digit 2 are selected" }
  }

/-- Theorem stating that the sampling methods are correctly identified -/
theorem correct_sampling_methods :
  schoolSurvey.technique1.method = SamplingMethod.SimpleRandom ∧
  schoolSurvey.technique2.method = SamplingMethod.Systematic :=
by sorry


end NUMINAMATH_CALUDE_correct_sampling_methods_l3253_325352


namespace NUMINAMATH_CALUDE_pirate_treasure_distribution_l3253_325317

def coin_distribution (x : ℕ) : ℕ × ℕ := 
  (x * (x + 1) / 2, x)

theorem pirate_treasure_distribution :
  ∃ x : ℕ, 
    let (bob_coins, sam_coins) := coin_distribution x
    bob_coins = 3 * sam_coins ∧ 
    bob_coins + sam_coins = 20 := by
  sorry

end NUMINAMATH_CALUDE_pirate_treasure_distribution_l3253_325317


namespace NUMINAMATH_CALUDE_number_exceeding_fraction_l3253_325305

theorem number_exceeding_fraction (x : ℚ) : 
  x = (3/8) * x + 35 → x = 56 := by
  sorry

end NUMINAMATH_CALUDE_number_exceeding_fraction_l3253_325305


namespace NUMINAMATH_CALUDE_possible_values_of_a_l3253_325355

theorem possible_values_of_a : 
  ∀ (a b c : ℤ), 
  (∀ x : ℝ, (x - a) * (x - 8) + 1 = (x + b) * (x + c)) → 
  (a = 6 ∨ a = 10) := by
  sorry

end NUMINAMATH_CALUDE_possible_values_of_a_l3253_325355


namespace NUMINAMATH_CALUDE_max_value_of_f_one_l3253_325389

/-- Given a function f(x) = x^2 + abx + a + 2b where f(0) = 4, 
    the maximum value of f(1) is 7. -/
theorem max_value_of_f_one (a b : ℝ) : 
  let f : ℝ → ℝ := λ x ↦ x^2 + a*b*x + a + 2*b
  (f 0 = 4) → (∀ y : ℝ, f 1 ≤ 7) ∧ (∃ y : ℝ, f 1 = 7) := by
  sorry

end NUMINAMATH_CALUDE_max_value_of_f_one_l3253_325389


namespace NUMINAMATH_CALUDE_train_length_calculation_train_length_proof_l3253_325338

/-- Calculates the length of a train given its speed, the time it takes to cross a bridge, and the length of the bridge. -/
theorem train_length_calculation (train_speed : ℝ) (crossing_time : ℝ) (bridge_length : ℝ) : 
  train_speed = 45 * (1000 / 3600) →
  crossing_time = 30 →
  bridge_length = 240 →
  (train_speed * crossing_time) - bridge_length = 135 :=
by
  sorry

/-- Proves that a train traveling at 45 km/hr that crosses a 240-meter bridge in 30 seconds has a length of 135 meters. -/
theorem train_length_proof : 
  ∃ (train_speed crossing_time bridge_length : ℝ),
    train_speed = 45 * (1000 / 3600) ∧
    crossing_time = 30 ∧
    bridge_length = 240 ∧
    (train_speed * crossing_time) - bridge_length = 135 :=
by
  sorry

end NUMINAMATH_CALUDE_train_length_calculation_train_length_proof_l3253_325338


namespace NUMINAMATH_CALUDE_crayons_added_l3253_325353

theorem crayons_added (initial : ℕ) (final : ℕ) (added : ℕ) : 
  initial = 7 → final = 10 → initial + added = final → added = 3 := by
  sorry

end NUMINAMATH_CALUDE_crayons_added_l3253_325353


namespace NUMINAMATH_CALUDE_function_property_implies_f3_values_l3253_325333

-- Define the function type
def FunctionType := ℝ → ℝ

-- Define the property that the function must satisfy
def SatisfiesProperty (f : FunctionType) : Prop :=
  ∀ x y : ℝ, f (x * f y - x) = x * y - f x

-- State the theorem
theorem function_property_implies_f3_values (f : FunctionType) 
  (h : SatisfiesProperty f) : 
  ∃ (a b : ℝ), (∀ z : ℝ, f 3 = z → (z = a ∨ z = b)) ∧ a + b = 0 := by
  sorry

end NUMINAMATH_CALUDE_function_property_implies_f3_values_l3253_325333


namespace NUMINAMATH_CALUDE_ab_is_perfect_cube_l3253_325347

theorem ab_is_perfect_cube (a b : ℕ+) (h1 : b < a) 
  (h2 : ∃ k : ℕ, k * (a * b * (a - b)) = a^3 + b^3 + a*b) : 
  ∃ n : ℕ, (a * b : ℕ) = n^3 := by
sorry

end NUMINAMATH_CALUDE_ab_is_perfect_cube_l3253_325347


namespace NUMINAMATH_CALUDE_train_passing_platform_l3253_325369

/-- Calculates the time taken for a train to pass a platform -/
theorem train_passing_platform 
  (train_length : ℝ) 
  (platform_length : ℝ) 
  (pole_passing_time : ℝ) 
  (h1 : train_length = 500)
  (h2 : platform_length = 500)
  (h3 : pole_passing_time = 50) :
  (train_length + platform_length) / (train_length / pole_passing_time) = 100 :=
by sorry

end NUMINAMATH_CALUDE_train_passing_platform_l3253_325369


namespace NUMINAMATH_CALUDE_south_american_countries_visited_l3253_325313

/-- Proves the number of South American countries visited given the conditions --/
theorem south_american_countries_visited
  (total : ℕ)
  (europe : ℕ)
  (asia : ℕ)
  (h1 : total = 42)
  (h2 : europe = 20)
  (h3 : asia = 6)
  (h4 : 2 * asia = total - europe - asia) :
  total - europe - asia = 8 :=
by sorry

end NUMINAMATH_CALUDE_south_american_countries_visited_l3253_325313


namespace NUMINAMATH_CALUDE_max_value_of_f_l3253_325364

def f (m : ℝ) (x : ℝ) : ℝ := -x^3 + 3*x^2 + m

theorem max_value_of_f (m : ℝ) :
  (∀ x ∈ Set.Icc (-2) 2, f m x ≥ 1) →
  (∃ x ∈ Set.Icc (-2) 2, f m x = 1) →
  (∃ x ∈ Set.Icc (-2) 2, f m x = 21) ∧
  (∀ x ∈ Set.Icc (-2) 2, f m x ≤ 21) := by
sorry

end NUMINAMATH_CALUDE_max_value_of_f_l3253_325364


namespace NUMINAMATH_CALUDE_diophantine_equation_solution_l3253_325393

theorem diophantine_equation_solution : 
  ∀ a b : ℤ, a > 0 ∧ b > 0 → (1 : ℚ) / a - (1 : ℚ) / b = (1 : ℚ) / 37 → (a = 38 ∧ b = 1332) :=
by
  sorry

end NUMINAMATH_CALUDE_diophantine_equation_solution_l3253_325393


namespace NUMINAMATH_CALUDE_irreducible_fraction_l3253_325372

theorem irreducible_fraction (n : ℕ+) : Nat.gcd (21 * n + 4) (14 * n + 3) = 1 := by
  sorry

end NUMINAMATH_CALUDE_irreducible_fraction_l3253_325372


namespace NUMINAMATH_CALUDE_spherical_coordinate_conversion_l3253_325386

def standardSphericalCoordinates (ρ θ φ : Real) : Prop :=
  ρ > 0 ∧ 0 ≤ θ ∧ θ < 2 * Real.pi ∧ 0 ≤ φ ∧ φ ≤ Real.pi

theorem spherical_coordinate_conversion :
  ∃ (ρ θ φ : Real),
    standardSphericalCoordinates ρ θ φ ∧
    ρ = 4 ∧
    θ = 5 * Real.pi / 4 ∧
    φ = Real.pi / 5 ∧
    (ρ, θ, φ) = (4, Real.pi / 4, 9 * Real.pi / 5) :=
sorry

end NUMINAMATH_CALUDE_spherical_coordinate_conversion_l3253_325386


namespace NUMINAMATH_CALUDE_line_relationship_l3253_325312

-- Define the concept of lines in 3D space
variable (Line : Type)

-- Define the relationships between lines
variable (are_skew : Line → Line → Prop)
variable (are_parallel : Line → Line → Prop)
variable (are_intersecting : Line → Line → Prop)

-- Theorem statement
theorem line_relationship (a b c : Line)
  (h1 : are_skew a b)
  (h2 : are_parallel a c) :
  are_skew b c ∨ are_intersecting b c :=
sorry

end NUMINAMATH_CALUDE_line_relationship_l3253_325312


namespace NUMINAMATH_CALUDE_absolute_value_non_positive_l3253_325397

theorem absolute_value_non_positive (y : ℚ) : 
  |4 * y - 6| ≤ 0 ↔ y = 3/2 := by sorry

end NUMINAMATH_CALUDE_absolute_value_non_positive_l3253_325397


namespace NUMINAMATH_CALUDE_no_real_roots_l3253_325394

theorem no_real_roots (c : ℝ) (h : c > 1) : ∀ x : ℝ, x^2 + 2*x + c ≠ 0 := by
  sorry

end NUMINAMATH_CALUDE_no_real_roots_l3253_325394


namespace NUMINAMATH_CALUDE_sufficient_necessary_equivalence_l3253_325330

theorem sufficient_necessary_equivalence (A B : Prop) :
  (A → B) ↔ (¬B → ¬A) :=
sorry

end NUMINAMATH_CALUDE_sufficient_necessary_equivalence_l3253_325330


namespace NUMINAMATH_CALUDE_monotonic_quadratic_function_condition_l3253_325345

/-- A function f(x) = x^2 + 2(a - 1)x + 2 is monotonic on the interval [-1, 2] if and only if a ∈ (-∞, -1] ∪ [2, +∞) -/
theorem monotonic_quadratic_function_condition (a : ℝ) :
  (∀ x ∈ Set.Icc (-1 : ℝ) 2, Monotone (fun x => x^2 + 2*(a - 1)*x + 2)) ↔
  a ∈ Set.Iic (-1 : ℝ) ∪ Set.Ici 2 :=
sorry

end NUMINAMATH_CALUDE_monotonic_quadratic_function_condition_l3253_325345


namespace NUMINAMATH_CALUDE_max_value_x_plus_reciprocal_l3253_325314

theorem max_value_x_plus_reciprocal (x : ℝ) (h : 9 = x^3 + 1/x^3) :
  ∃ (y : ℝ), y = x + 1/x ∧ y ≤ 3 ∧ y = 3 := by
  sorry

end NUMINAMATH_CALUDE_max_value_x_plus_reciprocal_l3253_325314


namespace NUMINAMATH_CALUDE_game_d_higher_prob_l3253_325319

def coin_prob_tails : ℚ := 3/4
def coin_prob_heads : ℚ := 1/4

def game_c_win_prob : ℚ := 2 * (coin_prob_heads * coin_prob_tails)

def game_d_win_prob : ℚ := 
  3 * (coin_prob_tails^2 * coin_prob_heads) + coin_prob_tails^3

theorem game_d_higher_prob : 
  game_d_win_prob - game_c_win_prob = 15/32 :=
sorry

end NUMINAMATH_CALUDE_game_d_higher_prob_l3253_325319


namespace NUMINAMATH_CALUDE_chocolate_bar_breaks_l3253_325399

/-- Represents a chocolate bar with grooves -/
structure ChocolateBar where
  longitudinal_grooves : Nat
  transverse_grooves : Nat

/-- Calculates the minimum number of breaks required to separate the chocolate bar into pieces with no grooves -/
def min_breaks (bar : ChocolateBar) : Nat :=
  4

/-- Theorem stating that a chocolate bar with 2 longitudinal grooves and 3 transverse grooves requires 4 breaks -/
theorem chocolate_bar_breaks (bar : ChocolateBar) 
  (h1 : bar.longitudinal_grooves = 2) 
  (h2 : bar.transverse_grooves = 3) : 
  min_breaks bar = 4 := by
  sorry

end NUMINAMATH_CALUDE_chocolate_bar_breaks_l3253_325399


namespace NUMINAMATH_CALUDE_trigonometric_identity_l3253_325306

theorem trigonometric_identity (a b c : Real) :
  (Real.sin (a - b)) / (Real.sin a * Real.sin b) +
  (Real.sin (b - c)) / (Real.sin b * Real.sin c) +
  (Real.sin (c - a)) / (Real.sin c * Real.sin a) = 0 := by
  sorry

end NUMINAMATH_CALUDE_trigonometric_identity_l3253_325306


namespace NUMINAMATH_CALUDE_sqrt_equation_solutions_l3253_325342

theorem sqrt_equation_solutions (x : ℝ) :
  (Real.sqrt (5 * x - 4) + 12 / Real.sqrt (5 * x - 4) = 8) ↔ (x = 8 ∨ x = 8/5) :=
by sorry

end NUMINAMATH_CALUDE_sqrt_equation_solutions_l3253_325342


namespace NUMINAMATH_CALUDE_school_committee_formation_l3253_325351

theorem school_committee_formation (n_students : ℕ) (n_teachers : ℕ) (committee_size : ℕ) : 
  n_students = 11 → n_teachers = 3 → committee_size = 8 →
  (Nat.choose (n_students + n_teachers) committee_size) - (Nat.choose n_students committee_size) = 2838 :=
by sorry

end NUMINAMATH_CALUDE_school_committee_formation_l3253_325351


namespace NUMINAMATH_CALUDE_max_parts_quadratic_trinomials_l3253_325315

/-- The maximum number of parts into which the coordinate plane can be divided by n quadratic trinomials -/
def max_parts (n : ℕ) : ℕ := n^2 + 1

/-- Theorem: The maximum number of parts into which the coordinate plane can be divided by n quadratic trinomials is n^2 + 1 -/
theorem max_parts_quadratic_trinomials (n : ℕ) :
  max_parts n = n^2 + 1 := by
  sorry

end NUMINAMATH_CALUDE_max_parts_quadratic_trinomials_l3253_325315


namespace NUMINAMATH_CALUDE_arithmetic_simplification_l3253_325391

theorem arithmetic_simplification :
  (0.25 * 4 - (5/6 + 1/12) * 6/5 = 1/10) ∧
  ((5/12 - 5/16) * 4/5 + 2/3 - 3/4 = 0) := by sorry

end NUMINAMATH_CALUDE_arithmetic_simplification_l3253_325391


namespace NUMINAMATH_CALUDE_smallest_X_proof_l3253_325302

/-- A function that checks if a positive integer only contains 0s and 1s as digits -/
def onlyZerosAndOnes (n : ℕ) : Prop := sorry

/-- The smallest positive integer X such that there exists a T satisfying the conditions -/
def smallestX : ℕ := 74

theorem smallest_X_proof :
  ∀ T : ℕ,
  T > 0 →
  onlyZerosAndOnes T →
  (∃ X : ℕ, T = 15 * X) →
  ∃ X : ℕ, T = 15 * X ∧ X ≥ smallestX :=
sorry

end NUMINAMATH_CALUDE_smallest_X_proof_l3253_325302


namespace NUMINAMATH_CALUDE_mystery_number_addition_l3253_325368

theorem mystery_number_addition (mystery_number certain_number : ℕ) : 
  mystery_number = 47 → 
  mystery_number + certain_number = 92 → 
  certain_number = 45 := by
sorry

end NUMINAMATH_CALUDE_mystery_number_addition_l3253_325368


namespace NUMINAMATH_CALUDE_vector_sum_magnitude_l3253_325310

variable {E : Type*} [NormedAddCommGroup E] [InnerProductSpace ℝ E]

theorem vector_sum_magnitude (a b : E) 
  (ha : ‖a‖ = 1) 
  (hb : ‖b‖ = 1) 
  (hab : ‖a - b‖ = 1) : 
  ‖a + b‖ = Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_vector_sum_magnitude_l3253_325310


namespace NUMINAMATH_CALUDE_points_bound_l3253_325346

/-- A structure representing a set of points on a line with colored circles -/
structure ColoredCircles where
  k : ℕ  -- Number of points
  n : ℕ  -- Number of colors
  points : Fin k → ℝ  -- Function mapping point indices to their positions on the line
  circle_color : Fin k → Fin k → Fin n  -- Function assigning a color to each circle

/-- Predicate to check if two circles are mutually tangent -/
def mutually_tangent (cc : ColoredCircles) (i j m l : Fin cc.k) : Prop :=
  (cc.points i < cc.points m) ∧ (cc.points m < cc.points j) ∧ (cc.points j < cc.points l)

/-- Axiom: Mutually tangent circles have different colors -/
axiom different_colors (cc : ColoredCircles) :
  ∀ (i j m l : Fin cc.k), mutually_tangent cc i j m l →
    cc.circle_color i j ≠ cc.circle_color m l

/-- Theorem: The number of points is at most 2^n -/
theorem points_bound (cc : ColoredCircles) : cc.k ≤ 2^cc.n := by
  sorry

end NUMINAMATH_CALUDE_points_bound_l3253_325346


namespace NUMINAMATH_CALUDE_evaluate_expression_l3253_325308

theorem evaluate_expression : (0.5^4 - 0.25^2) / (0.1^2) = 0 := by
  sorry

end NUMINAMATH_CALUDE_evaluate_expression_l3253_325308


namespace NUMINAMATH_CALUDE_min_value_of_expression_l3253_325375

theorem min_value_of_expression (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : 1/a + 1/b = 1) :
  ∃ (min : ℝ), min = 4 ∧ ∀ (x : ℝ), x = 1/(a-1) + 4/(b-1) → x ≥ min :=
sorry

end NUMINAMATH_CALUDE_min_value_of_expression_l3253_325375


namespace NUMINAMATH_CALUDE_gravel_path_rate_l3253_325316

/-- Calculates the rate per square meter for gravelling a path around a rectangular plot. -/
theorem gravel_path_rate (length width path_width total_cost : ℝ) 
  (h1 : length = 110)
  (h2 : width = 65)
  (h3 : path_width = 2.5)
  (h4 : total_cost = 680) : 
  total_cost / ((length * width) - ((length - 2 * path_width) * (width - 2 * path_width))) = 0.8 := by
  sorry

end NUMINAMATH_CALUDE_gravel_path_rate_l3253_325316


namespace NUMINAMATH_CALUDE_total_limes_and_plums_l3253_325356

theorem total_limes_and_plums (L M P : ℕ) (hL : L = 25) (hM : M = 32) (hP : P = 12) :
  L + M + P = 69 := by
  sorry

end NUMINAMATH_CALUDE_total_limes_and_plums_l3253_325356


namespace NUMINAMATH_CALUDE_decimal_to_fraction_l3253_325392

theorem decimal_to_fraction :
  (2.35 : ℚ) = 47 / 20 := by sorry

end NUMINAMATH_CALUDE_decimal_to_fraction_l3253_325392


namespace NUMINAMATH_CALUDE_exists_number_with_three_prime_factors_l3253_325370

def M (n : ℕ) : Set ℕ := {m | n ≤ m ∧ m ≤ n + 9}

def has_at_least_three_prime_factors (k : ℕ) : Prop :=
  ∃ (p q r : ℕ), Nat.Prime p ∧ Nat.Prime q ∧ Nat.Prime r ∧ 
  p ≠ q ∧ p ≠ r ∧ q ≠ r ∧ k % p = 0 ∧ k % q = 0 ∧ k % r = 0

theorem exists_number_with_three_prime_factors (n : ℕ) (h : n ≥ 93) :
  ∃ k ∈ M n, has_at_least_three_prime_factors k := by
  sorry

end NUMINAMATH_CALUDE_exists_number_with_three_prime_factors_l3253_325370


namespace NUMINAMATH_CALUDE_coffee_mug_price_l3253_325378

/-- The cost of a personalized coffee mug -/
def coffee_mug_cost : ℕ := sorry

/-- The price of a bracelet -/
def bracelet_price : ℕ := 15

/-- The price of a gold heart necklace -/
def necklace_price : ℕ := 10

/-- The number of bracelets bought -/
def num_bracelets : ℕ := 3

/-- The number of gold heart necklaces bought -/
def num_necklaces : ℕ := 2

/-- The amount paid with -/
def amount_paid : ℕ := 100

/-- The change received -/
def change_received : ℕ := 15

theorem coffee_mug_price : coffee_mug_cost = 20 := by
  sorry

end NUMINAMATH_CALUDE_coffee_mug_price_l3253_325378


namespace NUMINAMATH_CALUDE_quadratic_function_properties_l3253_325371

-- Define the quadratic function f(x)
def f (b c x : ℝ) : ℝ := x^2 + b*x + c

-- State the theorem
theorem quadratic_function_properties :
  ∀ b c : ℝ,
  (f b c 1 = 0) →
  (f b c 3 = 0) →
  (b = -4 ∧ c = 3) ∧
  (∀ x y : ℝ, 2 < x → x < y → f (-4) 3 x < f (-4) 3 y) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_function_properties_l3253_325371


namespace NUMINAMATH_CALUDE_consecutive_integers_product_360_l3253_325300

theorem consecutive_integers_product_360 :
  ∃ (n m : ℤ), 
    n * (n + 1) = 360 ∧ 
    (m - 1) * m * (m + 1) = 360 ∧ 
    n + (n + 1) + (m - 1) + m + (m + 1) = 55 := by
  sorry

end NUMINAMATH_CALUDE_consecutive_integers_product_360_l3253_325300


namespace NUMINAMATH_CALUDE_smallest_n_for_sqrt_difference_l3253_325334

theorem smallest_n_for_sqrt_difference : ∃ n : ℕ+, 
  (n = 626 ∧ 
   ∀ m : ℕ+, m < n → Real.sqrt m.val - Real.sqrt (m.val - 1) ≥ 0.02) ∧
  Real.sqrt n.val - Real.sqrt (n.val - 1) < 0.02 := by
  sorry

end NUMINAMATH_CALUDE_smallest_n_for_sqrt_difference_l3253_325334


namespace NUMINAMATH_CALUDE_triangle_side_length_l3253_325336

theorem triangle_side_length (a b c : ℝ) (A : ℝ) :
  a = Real.sqrt 5 →
  c = 2 →
  Real.cos A = 2/3 →
  b = 3 :=
by sorry

end NUMINAMATH_CALUDE_triangle_side_length_l3253_325336


namespace NUMINAMATH_CALUDE_number_fraction_problem_l3253_325341

theorem number_fraction_problem (x : ℝ) : (1/3 : ℝ) * (1/4 : ℝ) * x = 15 → (3/10 : ℝ) * x = 54 := by
  sorry

end NUMINAMATH_CALUDE_number_fraction_problem_l3253_325341


namespace NUMINAMATH_CALUDE_set_A_equals_interval_rep_l3253_325326

-- Define the set A
def A : Set ℝ := {x : ℝ | 0 ≤ x ∧ x < 5 ∨ x > 10}

-- Define the interval representation
def intervalRep : Set ℝ := Set.Ici 0 ∩ Set.Iio 5 ∪ Set.Ioi 10

-- Theorem statement
theorem set_A_equals_interval_rep : A = intervalRep := by sorry

end NUMINAMATH_CALUDE_set_A_equals_interval_rep_l3253_325326


namespace NUMINAMATH_CALUDE_prob_at_least_one_boy_and_girl_l3253_325320

/-- The probability of having a boy or a girl -/
def child_probability : ℚ := 1 / 2

/-- The number of children in the family -/
def family_size : ℕ := 4

/-- The probability of having at least one boy and one girl in a family with four children -/
theorem prob_at_least_one_boy_and_girl : 
  (1 : ℚ) - (child_probability ^ family_size + child_probability ^ family_size) = 7 / 8 := by
  sorry

end NUMINAMATH_CALUDE_prob_at_least_one_boy_and_girl_l3253_325320


namespace NUMINAMATH_CALUDE_finleys_age_l3253_325385

/-- Proves that Finley's age is 55 years old given the conditions in the problem --/
theorem finleys_age (jill roger finley : ℕ) : 
  jill = 20 → 
  roger = 2 * jill + 5 → 
  (roger + 15) - (jill + 15) = finley - 30 → 
  finley = 55 := by sorry

end NUMINAMATH_CALUDE_finleys_age_l3253_325385


namespace NUMINAMATH_CALUDE_student_congress_sample_size_l3253_325324

/-- Given a school with classes and students, prove the sample size for a "Student Congress" -/
theorem student_congress_sample_size 
  (num_classes : ℕ) 
  (students_per_class : ℕ) 
  (selected_students : ℕ) 
  (h1 : num_classes = 40)
  (h2 : students_per_class = 50)
  (h3 : selected_students = 150) :
  selected_students = 150 := by
  sorry

end NUMINAMATH_CALUDE_student_congress_sample_size_l3253_325324


namespace NUMINAMATH_CALUDE_root_sum_and_square_l3253_325331

theorem root_sum_and_square (α β : ℝ) : 
  (α^2 - α - 2006 = 0) → 
  (β^2 - β - 2006 = 0) → 
  (α + β = 1) →
  α + β^2 = 2007 := by
sorry

end NUMINAMATH_CALUDE_root_sum_and_square_l3253_325331


namespace NUMINAMATH_CALUDE_inequality_system_solution_set_l3253_325360

theorem inequality_system_solution_set :
  ∀ x : ℝ, (x < -3 ∧ x < 2) ↔ x < -3 := by sorry

end NUMINAMATH_CALUDE_inequality_system_solution_set_l3253_325360


namespace NUMINAMATH_CALUDE_time_after_1875_minutes_l3253_325367

/-- Represents time of day in 24-hour format -/
structure Time where
  hours : Nat
  minutes : Nat
  deriving Repr

/-- Adds minutes to a given time -/
def addMinutes (t : Time) (m : Nat) : Time :=
  sorry

theorem time_after_1875_minutes : 
  let start_time := Time.mk 15 15  -- 3:15 p.m.
  let end_time := Time.mk 10 30    -- 10:30 a.m.
  addMinutes start_time 1875 = end_time :=
by sorry

end NUMINAMATH_CALUDE_time_after_1875_minutes_l3253_325367


namespace NUMINAMATH_CALUDE_mary_speed_calculation_l3253_325387

/-- Mary's running speed in miles per hour -/
def mary_speed : ℝ := sorry

/-- Jimmy's running speed in miles per hour -/
def jimmy_speed : ℝ := 4

/-- Time elapsed in hours -/
def time : ℝ := 1

/-- Distance between Mary and Jimmy after 1 hour in miles -/
def distance : ℝ := 9

theorem mary_speed_calculation :
  mary_speed = 5 :=
by
  sorry

end NUMINAMATH_CALUDE_mary_speed_calculation_l3253_325387


namespace NUMINAMATH_CALUDE_isosceles_right_triangle_area_l3253_325377

/-- Given an isosceles right triangle with hypotenuse 6√2, prove its area is 18 -/
theorem isosceles_right_triangle_area (h : ℝ) (is_hypotenuse : h = 6 * Real.sqrt 2) :
  let a : ℝ := h / Real.sqrt 2
  let area : ℝ := (1 / 2) * a ^ 2
  area = 18 := by
  sorry

end NUMINAMATH_CALUDE_isosceles_right_triangle_area_l3253_325377


namespace NUMINAMATH_CALUDE_root_product_theorem_l3253_325348

theorem root_product_theorem (x₁ x₂ x₃ : ℝ) : 
  (Real.sqrt 2025 * x₁^3 - 4050 * x₁^2 + 4 = 0) →
  (Real.sqrt 2025 * x₂^3 - 4050 * x₂^2 + 4 = 0) →
  (Real.sqrt 2025 * x₃^3 - 4050 * x₃^2 + 4 = 0) →
  x₁ < x₂ → x₂ < x₃ →
  x₂ * (x₁ + x₃) = 90 := by
sorry

end NUMINAMATH_CALUDE_root_product_theorem_l3253_325348


namespace NUMINAMATH_CALUDE_crocus_bulb_cost_l3253_325376

theorem crocus_bulb_cost (total_space : ℕ) (daffodil_cost : ℚ) (total_budget : ℚ) (crocus_count : ℕ) :
  total_space = 55 →
  daffodil_cost = 65/100 →
  total_budget = 2915/100 →
  crocus_count = 22 →
  ∃ (crocus_cost : ℚ), crocus_cost = 35/100 ∧
    crocus_count * crocus_cost + (total_space - crocus_count) * daffodil_cost = total_budget :=
by sorry

end NUMINAMATH_CALUDE_crocus_bulb_cost_l3253_325376


namespace NUMINAMATH_CALUDE_segment_length_l3253_325327

/-- Given three points A, B, and C on a line, with AB = 4 and BC = 3,
    the length of AC is either 7 or 1. -/
theorem segment_length (A B C : ℝ) : 
  (B - A = 4) → (C - B = 3 ∨ B - C = 3) → (C - A = 7 ∨ C - A = 1) := by
  sorry

end NUMINAMATH_CALUDE_segment_length_l3253_325327


namespace NUMINAMATH_CALUDE_inequality_equivalence_l3253_325309

theorem inequality_equivalence (a : ℝ) : 
  (∀ x y : ℝ, x > 0 → y > 0 → Real.sqrt x + Real.sqrt y ≤ a * Real.sqrt (x + y)) ↔ 
  a ≥ Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_equivalence_l3253_325309


namespace NUMINAMATH_CALUDE_max_cuttable_strings_l3253_325349

/-- Represents a volleyball net as a graph --/
structure VolleyballNet where
  rows : Nat
  cols : Nat

/-- Calculates the number of nodes in the net --/
def VolleyballNet.nodeCount (net : VolleyballNet) : Nat :=
  (net.rows + 1) * (net.cols + 1)

/-- Calculates the total number of strings in the net --/
def VolleyballNet.stringCount (net : VolleyballNet) : Nat :=
  net.rows * (net.cols + 1) + (net.rows + 1) * net.cols

/-- Theorem: Maximum number of cuttable strings in a 10x100 volleyball net --/
theorem max_cuttable_strings (net : VolleyballNet) 
  (h_rows : net.rows = 10) (h_cols : net.cols = 100) : 
  net.stringCount - (net.nodeCount - 1) = 1000 := by
  sorry

end NUMINAMATH_CALUDE_max_cuttable_strings_l3253_325349


namespace NUMINAMATH_CALUDE_student_lecture_assignment_l3253_325357

/-- The number of ways to assign students to lectures -/
def assignment_count (num_students : ℕ) (num_lectures : ℕ) : ℕ :=
  num_lectures ^ num_students

/-- Theorem: The number of ways to assign 5 students to 3 lectures is 243 -/
theorem student_lecture_assignment :
  assignment_count 5 3 = 243 := by
  sorry

end NUMINAMATH_CALUDE_student_lecture_assignment_l3253_325357


namespace NUMINAMATH_CALUDE_swim_club_members_l3253_325359

theorem swim_club_members : 
  ∀ (total_members passed_members not_passed_members : ℕ),
  passed_members = (30 * total_members) / 100 →
  not_passed_members = 70 →
  not_passed_members = total_members - passed_members →
  total_members = 100 := by
sorry

end NUMINAMATH_CALUDE_swim_club_members_l3253_325359


namespace NUMINAMATH_CALUDE_sum_of_roots_equation_l3253_325382

theorem sum_of_roots_equation (x : ℝ) : 
  (3 = (x^3 - 3*x^2 - 12*x) / (x + 3)) → 
  (∃ y : ℝ, (3 = (y^3 - 3*y^2 - 12*y) / (y + 3)) ∧ (x + y = 6)) :=
by sorry

end NUMINAMATH_CALUDE_sum_of_roots_equation_l3253_325382


namespace NUMINAMATH_CALUDE_expression_evaluation_l3253_325379

theorem expression_evaluation (x y : ℤ) (hx : x = -2) (hy : y = -3) :
  (x - 3*y)^2 + (x - 2*y)*(x + 2*y) - x*(2*x - 5*y) - y = 42 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l3253_325379


namespace NUMINAMATH_CALUDE_school_expansion_theorem_l3253_325303

/-- Calculates the total number of students after adding classes to a school -/
def total_students_after_adding_classes 
  (initial_classes : ℕ) 
  (students_per_class : ℕ) 
  (added_classes : ℕ) : ℕ :=
  (initial_classes + added_classes) * students_per_class

/-- Theorem: A school with 15 initial classes of 20 students each, 
    after adding 5 more classes, will have 400 students in total -/
theorem school_expansion_theorem : 
  total_students_after_adding_classes 15 20 5 = 400 := by
  sorry

end NUMINAMATH_CALUDE_school_expansion_theorem_l3253_325303


namespace NUMINAMATH_CALUDE_contrapositive_equivalence_l3253_325396

theorem contrapositive_equivalence (x : ℝ) :
  (¬(x^2 < 1) → ¬(-1 < x ∧ x < 1)) ↔ ((x ≥ 1 ∨ x ≤ -1) → x^2 ≥ 1) :=
by sorry

end NUMINAMATH_CALUDE_contrapositive_equivalence_l3253_325396


namespace NUMINAMATH_CALUDE_triangle_angle_leq_60_l3253_325395

/-- Theorem: In any triangle, at least one angle is less than or equal to 60 degrees. -/
theorem triangle_angle_leq_60 (A B C : ℝ) (h_triangle : A + B + C = 180) :
  A ≤ 60 ∨ B ≤ 60 ∨ C ≤ 60 := by
  sorry

end NUMINAMATH_CALUDE_triangle_angle_leq_60_l3253_325395


namespace NUMINAMATH_CALUDE_favorite_sports_survey_l3253_325328

/-- Given a survey of students' favorite sports, prove the number of students who like chess or basketball. -/
theorem favorite_sports_survey (total_students : ℕ) 
  (basketball_percent chess_percent soccer_percent badminton_percent : ℚ) :
  basketball_percent = 40/100 →
  chess_percent = 10/100 →
  soccer_percent = 28/100 →
  badminton_percent = 22/100 →
  basketball_percent + chess_percent + soccer_percent + badminton_percent = 1 →
  total_students = 250 →
  ⌊(basketball_percent + chess_percent) * total_students⌋ = 125 := by
  sorry


end NUMINAMATH_CALUDE_favorite_sports_survey_l3253_325328


namespace NUMINAMATH_CALUDE_parabola_intersection_theorem_l3253_325343

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- The parabola y^2 = 4x -/
def Parabola : Set Point :=
  {p : Point | p.y^2 = 4 * p.x}

/-- Distance squared between two points -/
def distanceSquared (p q : Point) : ℝ :=
  (p.x - q.x)^2 + (p.y - q.y)^2

/-- A line in parametric form -/
structure Line where
  a : ℝ  -- x-intercept
  α : ℝ  -- angle of inclination

/-- Get points where a line intersects the parabola -/
def lineParabolaIntersection (l : Line) : Set Point :=
  {p : Point | p ∈ Parabola ∧ ∃ t : ℝ, p.x = l.a + t * Real.cos l.α ∧ p.y = t * Real.sin l.α}

/-- The theorem to be proved -/
theorem parabola_intersection_theorem (a : ℝ) :
  (∀ l : Line, l.a = a →
    let M : Point := ⟨a, 0⟩
    let intersections := lineParabolaIntersection l
    ∃ k : ℝ, ∀ P Q : Point, P ∈ intersections → Q ∈ intersections → P ≠ Q →
      1 / distanceSquared P M + 1 / distanceSquared Q M = k) →
  a = 1 := by
  sorry

end NUMINAMATH_CALUDE_parabola_intersection_theorem_l3253_325343


namespace NUMINAMATH_CALUDE_simplify_expression_1_simplify_expression_2_simplify_expression_3_l3253_325380

-- Problem 1
theorem simplify_expression_1 (x : ℝ) : (1 : ℝ) * (2 * x^2)^3 - x^2 * x^4 = 7 * x^6 := by sorry

-- Problem 2
theorem simplify_expression_2 (a b : ℝ) : (a + b)^2 - b * (2 * a + b) = a^2 := by sorry

-- Problem 3
theorem simplify_expression_3 (x : ℝ) : (x + 1) * (x - 1) - x^2 = -1 := by sorry

end NUMINAMATH_CALUDE_simplify_expression_1_simplify_expression_2_simplify_expression_3_l3253_325380


namespace NUMINAMATH_CALUDE_same_month_same_gender_exists_l3253_325362

/-- Represents a student with their gender and birth month. -/
structure Student where
  gender : Bool  -- True for girl, False for boy
  birthMonth : Fin 12

/-- Theorem: In a class of 25 students, there must be at least two girls
    or two boys born in the same month. -/
theorem same_month_same_gender_exists (students : Finset Student)
    (h_count : students.card = 25) :
    (∃ (m : Fin 12), 2 ≤ (students.filter (fun s => s.gender ∧ s.birthMonth = m)).card) ∨
    (∃ (m : Fin 12), 2 ≤ (students.filter (fun s => ¬s.gender ∧ s.birthMonth = m)).card) :=
  sorry


end NUMINAMATH_CALUDE_same_month_same_gender_exists_l3253_325362


namespace NUMINAMATH_CALUDE_quadratic_roots_condition_l3253_325363

-- Define the quadratic equation
def quadratic (m : ℝ) (x : ℝ) : ℝ := (m + 3) * x^2 - 4 * m * x + 2 * m - 1

-- Define the condition for roots having opposite signs
def opposite_signs (x₁ x₂ : ℝ) : Prop := x₁ * x₂ < 0

-- Define the condition for the absolute value of the negative root being greater than the positive root
def negative_root_greater (x₁ x₂ : ℝ) : Prop := 
  (x₁ < 0 ∧ x₂ > 0 ∧ abs x₁ > x₂) ∨ (x₂ < 0 ∧ x₁ > 0 ∧ abs x₂ > x₁)

-- The main theorem
theorem quadratic_roots_condition (m : ℝ) :
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ 
    quadratic m x₁ = 0 ∧ 
    quadratic m x₂ = 0 ∧ 
    opposite_signs x₁ x₂ ∧ 
    negative_root_greater x₁ x₂) →
  -3 < m ∧ m < 0 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_roots_condition_l3253_325363


namespace NUMINAMATH_CALUDE_fence_painting_fraction_l3253_325384

theorem fence_painting_fraction (total_time minutes : ℚ) (fraction : ℚ) :
  total_time = 60 →
  minutes = 15 →
  fraction = minutes / total_time →
  fraction = 1 / 4 := by
sorry

end NUMINAMATH_CALUDE_fence_painting_fraction_l3253_325384


namespace NUMINAMATH_CALUDE_length_of_side_b_area_of_triangle_ABC_l3253_325325

-- Define the triangle ABC
def triangle_ABC (a b c : ℝ) (A B C : ℝ) : Prop :=
  -- Add conditions for a valid triangle if necessary
  true

-- Given conditions
axiom side_a : ℝ
axiom side_a_value : side_a = 3 * Real.sqrt 3

axiom side_c : ℝ
axiom side_c_value : side_c = 2

axiom angle_B : ℝ
axiom angle_B_value : angle_B = 150 * Real.pi / 180

-- Theorem for the length of side b
theorem length_of_side_b (a b c : ℝ) (A B C : ℝ) 
  (h : triangle_ABC a b c A B C)
  (ha : a = side_a) (hc : c = side_c) (hB : B = angle_B) :
  b = 7 := by sorry

-- Theorem for the area of triangle ABC
theorem area_of_triangle_ABC (a b c : ℝ) (A B C : ℝ) 
  (h : triangle_ABC a b c A B C)
  (ha : a = side_a) (hc : c = side_c) (hB : B = angle_B) :
  (1/2) * a * c * Real.sin B = (3 * Real.sqrt 3) / 2 := by sorry

end NUMINAMATH_CALUDE_length_of_side_b_area_of_triangle_ABC_l3253_325325


namespace NUMINAMATH_CALUDE_ellipse_triangle_perimeter_l3253_325358

/-- Given an ellipse with equation x²/4 + y²/2 = 1, prove that the perimeter of the triangle
    formed by any point on the ellipse and its foci is 4 + 2√2. -/
theorem ellipse_triangle_perimeter (x y : ℝ) (P : ℝ × ℝ) (F₁ F₂ : ℝ × ℝ) :
  x^2 / 4 + y^2 / 2 = 1 →
  P = (x, y) →
  F₁.1^2 / 4 + F₁.2^2 / 2 = 1 →
  F₂.1^2 / 4 + F₂.2^2 / 2 = 1 →
  ∃ c : ℝ, c^2 = 2 ∧ 
    dist P F₁ + dist P F₂ + dist F₁ F₂ = 4 + 2 * Real.sqrt 2 :=
by sorry

end NUMINAMATH_CALUDE_ellipse_triangle_perimeter_l3253_325358


namespace NUMINAMATH_CALUDE_circle_equation_l3253_325383

/-- A circle in the 2D plane -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- Check if a point lies on a circle -/
def pointOnCircle (c : Circle) (p : ℝ × ℝ) : Prop :=
  let (x, y) := p
  let (h, k) := c.center
  (x - h)^2 + (y - k)^2 = c.radius^2

/-- Check if a circle is tangent to the y-axis -/
def tangentToYAxis (c : Circle) : Prop :=
  c.center.1 = c.radius

/-- Check if a point lies on the line x - 3y = 0 -/
def pointOnLine (p : ℝ × ℝ) : Prop :=
  p.1 - 3 * p.2 = 0

theorem circle_equation (c : Circle) :
  tangentToYAxis c ∧ 
  pointOnLine c.center ∧ 
  pointOnCircle c (6, 1) →
  c.center = (3, 1) ∧ c.radius = 3 :=
by sorry

#check circle_equation

end NUMINAMATH_CALUDE_circle_equation_l3253_325383


namespace NUMINAMATH_CALUDE_marks_initial_fries_l3253_325301

/-- Given that Sally had 14 fries initially, Mark gave her one-third of his fries,
    and Sally ended up with 26 fries, prove that Mark initially had 36 fries. -/
theorem marks_initial_fries (sally_initial : ℕ) (sally_final : ℕ) (mark_fraction : ℚ) :
  sally_initial = 14 →
  sally_final = 26 →
  mark_fraction = 1 / 3 →
  ∃ (mark_initial : ℕ), 
    mark_initial = 36 ∧
    sally_final = sally_initial + mark_fraction * mark_initial :=
by sorry

end NUMINAMATH_CALUDE_marks_initial_fries_l3253_325301


namespace NUMINAMATH_CALUDE_triangle_area_l3253_325381

theorem triangle_area (a b : ℝ) (C : ℝ) (h1 : a = 3 * Real.sqrt 2) (h2 : b = 2 * Real.sqrt 3) (h3 : Real.cos C = 1/3) :
  (1/2) * a * b * Real.sin C = 4 * Real.sqrt 3 := by
sorry

end NUMINAMATH_CALUDE_triangle_area_l3253_325381


namespace NUMINAMATH_CALUDE_wage_payment_problem_l3253_325398

/-- Given a sum of money that can pay A's wages for 20 days and both A and B's wages for 12 days,
    prove that it can pay B's wages for 30 days. -/
theorem wage_payment_problem (total_sum : ℝ) (wage_A wage_B : ℝ) 
  (h1 : total_sum = 20 * wage_A)
  (h2 : total_sum = 12 * (wage_A + wage_B)) :
  total_sum = 30 * wage_B :=
by sorry

end NUMINAMATH_CALUDE_wage_payment_problem_l3253_325398


namespace NUMINAMATH_CALUDE_circle_E_and_tangents_l3253_325374

-- Define the circle E
def circle_E (x y : ℝ) : Prop := (x - 5)^2 + (y - 1)^2 = 25

-- Define points A, B, C, and P
def point_A : ℝ × ℝ := (0, 1)
def point_B : ℝ × ℝ := (1, 4)
def point_C : ℝ × ℝ := (10, 1)
def point_P : ℝ × ℝ := (10, 11)

-- Define lines l1 and l2
def line_l1 (x y : ℝ) : Prop := x - 5*y - 5 = 0
def line_l2 (x y : ℝ) : Prop := x - 2*y - 8 = 0

-- Define tangent lines
def tangent_line1 (x : ℝ) : Prop := x = 10
def tangent_line2 (x y : ℝ) : Prop := 3*x - 4*y + 14 = 0

theorem circle_E_and_tangents :
  (∀ x y : ℝ, line_l1 x y ∧ line_l2 x y → (x, y) = point_C) →
  circle_E point_A.1 point_A.2 →
  circle_E point_B.1 point_B.2 →
  circle_E point_C.1 point_C.2 →
  (∀ x y : ℝ, circle_E x y ∧ (tangent_line1 x ∨ tangent_line2 x y) →
    ((x - point_P.1)^2 + (y - point_P.2)^2) * 25 = ((x - 5)^2 + (y - 1)^2) * ((point_P.1 - 5)^2 + (point_P.2 - 1)^2)) :=
by sorry

end NUMINAMATH_CALUDE_circle_E_and_tangents_l3253_325374


namespace NUMINAMATH_CALUDE_negation_equivalence_l3253_325365

theorem negation_equivalence (m : ℤ) :
  (¬ ∃ x : ℤ, x^2 + 2*x + m < 0) ↔ (∀ x : ℤ, x^2 + 2*x + m ≥ 0) :=
by sorry

end NUMINAMATH_CALUDE_negation_equivalence_l3253_325365


namespace NUMINAMATH_CALUDE_brick_length_is_125_l3253_325337

/-- Represents the dimensions of a rectangular object in centimeters -/
structure Dimensions where
  length : ℝ
  width : ℝ
  height : ℝ

/-- Calculates the volume of a rectangular object given its dimensions -/
def volume (d : Dimensions) : ℝ :=
  d.length * d.width * d.height

/-- The dimensions of the wall in centimeters -/
def wall_dimensions : Dimensions :=
  { length := 800, width := 600, height := 22.5 }

/-- The partial dimensions of a brick in centimeters -/
def brick_partial_dimensions (x : ℝ) : Dimensions :=
  { length := x, width := 11.25, height := 6 }

/-- The number of bricks needed to build the wall -/
def number_of_bricks : ℕ := 1280

/-- Theorem stating that the length of each brick is 125 cm -/
theorem brick_length_is_125 : 
  ∃ x : ℝ, x = 125 ∧ 
  volume wall_dimensions = (number_of_bricks : ℝ) * volume (brick_partial_dimensions x) :=
sorry

end NUMINAMATH_CALUDE_brick_length_is_125_l3253_325337


namespace NUMINAMATH_CALUDE_distance_vertical_line_l3253_325373

/-- The distance between two points on a vertical line with y-coordinates differing by 2 is 2. -/
theorem distance_vertical_line (a : ℝ) : 
  Real.sqrt (((-3) - (-3))^2 + ((2 - a) - (-a))^2) = 2 := by
  sorry

end NUMINAMATH_CALUDE_distance_vertical_line_l3253_325373


namespace NUMINAMATH_CALUDE_square_difference_l3253_325318

/-- A configuration of four squares with specific side length differences -/
structure SquareConfiguration where
  small : ℝ
  third : ℝ
  second : ℝ
  largest : ℝ
  third_diff : third = small + 13
  second_diff : second = third + 5
  largest_diff : largest = second + 11

/-- The theorem stating that the difference between the largest and smallest square's side lengths is 29 -/
theorem square_difference (config : SquareConfiguration) : config.largest - config.small = 29 :=
  sorry

end NUMINAMATH_CALUDE_square_difference_l3253_325318


namespace NUMINAMATH_CALUDE_composite_polynomial_l3253_325321

theorem composite_polynomial (n : ℕ) : 
  ∃ (a b : ℕ), a > 1 ∧ b > 1 ∧ n^3 + 9*n^2 + 27*n + 35 = a * b :=
by sorry

end NUMINAMATH_CALUDE_composite_polynomial_l3253_325321
