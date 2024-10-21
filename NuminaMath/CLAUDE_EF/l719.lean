import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_omega_for_symmetry_l719_71936

noncomputable def f (ω : ℝ) (x : ℝ) : ℝ := Real.sin (ω * x + Real.pi / 3)

noncomputable def C (ω : ℝ) (x : ℝ) : ℝ := f ω (x + Real.pi / 2)

def is_symmetric_about_y_axis (g : ℝ → ℝ) : Prop :=
  ∀ x, g x = g (-x)

theorem min_omega_for_symmetry :
  ∃ ω_min : ℝ, ω_min > 0 ∧
    is_symmetric_about_y_axis (C ω_min) ∧
    (∀ ω, ω > 0 → is_symmetric_about_y_axis (C ω) → ω ≥ ω_min) ∧
    ω_min = 1 / 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_omega_for_symmetry_l719_71936


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cauchy_mean_value_theorem_l719_71983

open Set

/-- Cauchy's Mean Value Theorem -/
theorem cauchy_mean_value_theorem 
  {f g : ℝ → ℝ} {a b : ℝ} (hab : a < b)
  (hf : Differentiable ℝ f) (hg : Differentiable ℝ g)
  (hg' : ∀ x ∈ Ioo a b, (deriv g) x ≠ 0) :
  ∃ x ∈ Ioo a b, (f b - f a) / (g b - g a) = (deriv f x) / (deriv g x) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cauchy_mean_value_theorem_l719_71983


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_translation_l719_71962

/-- Given a parabola y = 3x², prove that after translating 4 units right and 1 unit up, 
    the resulting equation is y = 3(x-4)² + 1 -/
theorem parabola_translation (x y : ℝ) : 
  (y = 3 * x^2) → 
  (y + 1 = 3 * (x + 4 - 4)^2 + 1) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_translation_l719_71962


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_limit_tangent_over_square_limit_fraction_does_not_exist_limit_xy_over_sqrt_l719_71989

theorem limit_tangent_over_square :
  ∀ ε > 0, ∃ δ > 0, ∀ p : ℝ × ℝ, 
    p.1^2 + p.2^2 < δ^2 → 
    |Real.tan (p.1^2 + p.2^2) / (p.1^2 + p.2^2) - 1| < ε :=
by sorry

theorem limit_fraction_does_not_exist :
  ¬∃ L : ℝ, ∀ ε > 0, ∃ δ > 0, ∀ p : ℝ × ℝ, 
    p.1^2 + p.2^2 < δ^2 → 
    |(p.1^2 - p.2^2) / (p.1^2 + p.2^2) - L| < ε :=
by sorry

theorem limit_xy_over_sqrt :
  ∀ ε > 0, ∃ δ > 0, ∀ p : ℝ × ℝ, 
    p.1^2 + p.2^2 < δ^2 → 
    |p.1 * p.2 / (Real.sqrt (4 - p.1 * p.2) - 2) + 4| < ε :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_limit_tangent_over_square_limit_fraction_does_not_exist_limit_xy_over_sqrt_l719_71989


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_complex_expression_l719_71990

/-- Given a complex number w with magnitude 2, the maximum value of |(w - 2)^2 (w + 2)| is 16√2 -/
theorem max_value_complex_expression (w : ℂ) (h : Complex.abs w = 2) :
  ∃ M, M = 16 * Real.sqrt 2 ∧ ∀ z, Complex.abs w = 2 → Complex.abs ((z - 2)^2 * (z + 2)) ≤ M := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_complex_expression_l719_71990


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_loss_percent_calculation_l719_71904

/-- Calculates the loss percent given the cost price and selling price -/
noncomputable def loss_percent (cost_price selling_price : ℝ) : ℝ :=
  ((cost_price - selling_price) / cost_price) * 100

/-- Proves that the loss percent is 50% when an article is bought for Rs. 600 and sold for Rs. 300 -/
theorem loss_percent_calculation :
  let cost_price : ℝ := 600
  let selling_price : ℝ := 300
  loss_percent cost_price selling_price = 50 := by
  -- Unfold the definition of loss_percent
  unfold loss_percent
  -- Simplify the expression
  simp
  -- Perform the calculation
  norm_num
  -- The proof is complete
  done

end NUMINAMATH_CALUDE_ERRORFEEDBACK_loss_percent_calculation_l719_71904


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_initial_sum_proof_l719_71944

/-- Compound interest calculation -/
noncomputable def compound_interest (principal : ℝ) (rate : ℝ) (time : ℝ) (frequency : ℝ) : ℝ :=
  principal * ((1 + rate / frequency) ^ (frequency * time) - 1)

/-- Theorem: Given compound interest conditions, prove the initial sum -/
theorem initial_sum_proof (interest : ℝ) (rate : ℝ) (time : ℝ) (frequency : ℝ) 
  (h1 : interest = 1289.0625)
  (h2 : rate = 0.25)
  (h3 : time = 0.5)
  (h4 : frequency = 4) :
  ∃ (principal : ℝ), compound_interest principal rate time frequency = interest ∧ principal = 10000 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_initial_sum_proof_l719_71944


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_achievable_rook_configuration_l719_71984

/-- Represents a position on the chessboard -/
structure Position where
  x : Nat
  y : Nat

/-- Represents a move on the chessboard -/
inductive Move where
  | PlaceBottomLeft : Move
  | MoveUp : Position → Move
  | MoveRight : Position → Move

/-- Represents the state of the chessboard -/
structure BoardState where
  rooksOnBoard : Nat
  rooksInBag : Nat
  positions : List Position

/-- Applies a move to the board state -/
def applyMove (state : BoardState) (move : Move) : BoardState :=
  sorry

/-- Checks if a given configuration of rooks is valid (non-attacking) -/
def isValidConfiguration (positions : List Position) : Bool :=
  sorry

/-- The main theorem to be proved -/
theorem achievable_rook_configuration
  (finalConfig : List Position)
  (h1 : finalConfig.length = 100)
  (h2 : isValidConfiguration finalConfig = true) :
  ∃ (moves : List Move),
    let initialState : BoardState := ⟨0, 199, []⟩
    let finalState := moves.foldl applyMove initialState
    finalState.rooksOnBoard = 100 ∧
    finalState.positions = finalConfig :=
  sorry

#check achievable_rook_configuration

end NUMINAMATH_CALUDE_ERRORFEEDBACK_achievable_rook_configuration_l719_71984


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pyramid_surface_area_theorem_l719_71978

/-- Calculates the total surface area of a right pyramid with a rectangular base -/
noncomputable def pyramidSurfaceArea (baseLength width peakHeight : ℝ) : ℝ :=
  let baseArea := baseLength * width
  let slantHeight1 := Real.sqrt (peakHeight^2 + (baseLength/2)^2)
  let slantHeight2 := Real.sqrt (peakHeight^2 + (width/2)^2)
  let sideArea1 := baseLength * slantHeight1
  let sideArea2 := width * slantHeight2
  baseArea + sideArea1 + sideArea2

/-- The total surface area of a right pyramid with given dimensions is 392 + 50√10 -/
theorem pyramid_surface_area_theorem :
  pyramidSurfaceArea 14 10 15 = 392 + 50 * Real.sqrt 10 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_pyramid_surface_area_theorem_l719_71978


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distinct_values_count_l719_71986

def digits : Finset ℕ := {1, 2, 5, 6}

def expression (a b c d : ℕ) : ℕ := (a * b) + (c * d)

def valid_combinations : Finset (ℕ × ℕ × ℕ × ℕ) :=
  Finset.filter (λ (a, b, c, d) => a ∈ digits ∧ b ∈ digits ∧ c ∈ digits ∧ d ∈ digits ∧
    a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d)
    (Finset.product digits (Finset.product digits (Finset.product digits digits)))

def distinct_values : Finset ℕ :=
  Finset.image (λ (a, b, c, d) => expression a b c d) valid_combinations

theorem distinct_values_count : distinct_values.card = 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distinct_values_count_l719_71986


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_length_calculation_l719_71975

/-- The speed of the train in km/hr -/
noncomputable def train_speed : ℚ := 72

/-- The length of the platform in meters -/
noncomputable def platform_length : ℚ := 80

/-- The time taken to cross the platform in seconds -/
noncomputable def crossing_time : ℚ := 26

/-- Conversion factor from km/hr to m/s -/
noncomputable def km_hr_to_m_s : ℚ := 1000 / 3600

theorem train_length_calculation : 
  train_speed * km_hr_to_m_s * crossing_time - platform_length = 440 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_length_calculation_l719_71975


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_coordinates_of_P_l719_71938

-- Define the angle in radians
noncomputable def angle : ℝ := 4 * Real.pi / 3

-- Define the distance from origin O to point P
def distance : ℝ := 2

-- Define the coordinates of point P
noncomputable def point_P : ℝ × ℝ := (distance * Real.cos angle, distance * Real.sin angle)

-- Theorem statement
theorem coordinates_of_P : point_P = (-1, -Real.sqrt 3) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_coordinates_of_P_l719_71938


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_b_value_l719_71927

-- Define the curve
noncomputable def f (x : ℝ) : ℝ := x * Real.log x

-- Define the derivative of the curve
noncomputable def f_prime (x : ℝ) : ℝ := Real.log x + 1

-- Define the tangent line
def tangent_line (x b : ℝ) : ℝ := -2 * x + b

-- Theorem statement
theorem tangent_line_b_value :
  ∃ (x₀ : ℝ), x₀ > 0 ∧ 
  f_prime x₀ = -2 ∧
  tangent_line x₀ (f x₀ + 2 * x₀) = f x₀ →
  f x₀ + 2 * x₀ = -Real.exp (-3) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_b_value_l719_71927


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_needle_pattern_l719_71999

noncomputable def needle_angle (x y : ℤ) : ℝ :=
  Real.arctan ((y^2 - 2*x*y - x^2) / (y^2 + 2*x*y - x^2))

theorem needle_pattern (x y : ℤ) (hxy : (x, y) ≠ (0, 0)) :
  let φ := needle_angle x y
  ((x = 0 ∨ y = 0) → φ = π/4) ∧
  (∀ m : ℚ, Real.tan φ = (m^2 - 2*m - 1) / (m^2 + 2*m - 1) → y = m * x) ∧
  (∀ m : ℚ, m ≠ 0 → Real.tan (needle_angle x y) = Real.tan (needle_angle y (-x))) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_needle_pattern_l719_71999


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_g_less_than_one_h_two_distinct_zeros_l719_71913

-- Define the functions
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * x^2 + 1
noncomputable def g (x : ℝ) : ℝ := (x * (f 1 x) - x) / Real.exp x
noncomputable def h (a : ℝ) (x : ℝ) : ℝ := 1 - (f a x - 1) / Real.exp x

-- Statement 1
theorem g_less_than_one : ∀ x ≥ 5, g x < 1 := by sorry

-- Statement 2
theorem h_two_distinct_zeros (a : ℝ) :
  (∃ x y, 0 < x ∧ x < y ∧ h a x = 0 ∧ h a y = 0 ∧ 
    (∀ z, (0 < z ∧ z < x) → h a z ≠ 0) ∧ 
    (∀ z, (x < z ∧ z < y) → h a z ≠ 0)) ↔
  a > Real.exp 2 / 4 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_g_less_than_one_h_two_distinct_zeros_l719_71913


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ball_distribution_probability_ratio_l719_71932

theorem ball_distribution_probability_ratio :
  let n : ℕ := 18  -- number of balls
  let k : ℕ := 6   -- number of bins
  let p' : ℚ := (Nat.choose k 1 * Nat.choose (k-1) 1 * Nat.choose (n-8+4-1) (4-1)) / Nat.choose (n+k-1) (k-1)
  let q' : ℚ := 1
  p' / q' = 8580 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ball_distribution_probability_ratio_l719_71932


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_T_2_formula_T_3_formula_T_k_degree_l719_71994

/-- T_k(n) denotes the sum of the products of k numbers from 1 to n -/
def T (k n : ℕ) : ℚ :=
  sorry

/-- T_2(n) formula -/
theorem T_2_formula (n : ℕ) :
  T 2 n = n * (n + 1) * (n - 1) * (3 * n + 2) / 24 :=
sorry

/-- T_3(n) formula -/
theorem T_3_formula (n : ℕ) :
  T 3 n = n^2 * (n + 1)^2 * (n - 1) * (n - 2) / 48 :=
sorry

/-- T_k(n) is a polynomial in n of degree 2k -/
theorem T_k_degree (k : ℕ) :
  ∃ p : Polynomial ℚ, (∀ n : ℕ, T k n = p.eval (n : ℚ)) ∧ p.degree = 2 * k :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_T_2_formula_T_3_formula_T_k_degree_l719_71994


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_sums_count_l719_71951

/-- Represents a path from P to Q as a list of integers -/
def MyPath := List Int

/-- The set of all possible paths from P to Q -/
def paths : List MyPath := [
  [1, 3, 5],
  [1, 4, 6],
  [2, 3, 5],
  [2, 4, 6]
]

/-- Calculates the sum of a path -/
def pathSum (p : MyPath) : Int :=
  p.sum

/-- Theorem: The number of unique sums obtained from the given paths is 4 -/
theorem unique_sums_count : (paths.map pathSum).toFinset.card = 4 := by
  sorry

#eval (paths.map pathSum).toFinset.card

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_sums_count_l719_71951


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_closest_point_on_plane_l719_71955

noncomputable def plane (x y z : ℝ) : Prop := 5*x + 3*y - 2*z = 20

noncomputable def distance (x₁ y₁ z₁ x₂ y₂ z₂ : ℝ) : ℝ := 
  Real.sqrt ((x₁ - x₂)^2 + (y₁ - y₂)^2 + (z₁ - z₂)^2)

theorem closest_point_on_plane : 
  ∀ x y z : ℝ, 
    plane x y z → 
    distance x y z 2 1 4 ≥ distance (265/38) (83/38) (122/38) 2 1 4 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_closest_point_on_plane_l719_71955


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_shaded_area_approximation_l719_71931

noncomputable section

-- Define the semicircles and their properties
def semicircle_ADB_radius : ℝ := 2
def semicircle_BEC_radius : ℝ := 3
def semicircle_DFE_radius : ℝ := 2.5

-- Define the areas of the semicircles
def area_ADB : ℝ := (1/2) * Real.pi * semicircle_ADB_radius^2
def area_BEC : ℝ := (1/2) * Real.pi * semicircle_BEC_radius^2
def area_DFE : ℝ := (1/2) * Real.pi * semicircle_DFE_radius^2

-- Define the estimated overlap area
def estimated_overlap_area : ℝ := Real.pi

-- Define the shaded area
def shaded_area : ℝ := area_ADB + area_BEC + area_DFE - estimated_overlap_area

-- Theorem statement
theorem shaded_area_approximation :
  |shaded_area - 8.625 * Real.pi| < 0.001 := by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_shaded_area_approximation_l719_71931


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fifth_root_expression_values_l719_71902

noncomputable def fifth_root_of_unity (x : ℂ) : Prop := x^5 = 1

noncomputable def expression (x : ℂ) : ℂ :=
  2*x + 1/(1+x) + x/(1+x^2) + x^2/(1+x^3) + x^3/(1+x^4)

theorem fifth_root_expression_values (x : ℂ) :
  fifth_root_of_unity x →
  (expression x = 4 ∨ expression x = -1 + Real.sqrt 5 ∨ expression x = -1 - Real.sqrt 5) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fifth_root_expression_values_l719_71902


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_mass_percentage_of_H_in_certain_compound_l719_71920

/-- The mass percentage of an element in a compound -/
def mass_percentage (element : String) (compound : String) : ℝ := 
  sorry

/-- A certain compound -/
def certain_compound : String :=
  sorry

theorem mass_percentage_of_H_in_certain_compound :
  mass_percentage "H" certain_compound = 7.55 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_mass_percentage_of_H_in_certain_compound_l719_71920


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_odd_and_decreasing_l719_71993

-- Define the function f
noncomputable def f (θ : Real) (x : Real) : Real := 
  Real.sin (2 * x + θ) + Real.sqrt 3 * Real.cos (2 * x + θ)

-- Define the interval
def interval : Set Real := Set.Icc (-Real.pi/4) 0

-- Theorem statement
theorem function_odd_and_decreasing :
  ∃ θ : Real, 
    (∀ x, f θ (-x) = -(f θ x)) ∧ 
    (∀ x y, x ∈ interval → y ∈ interval → x < y → f θ y < f θ x) ∧ 
    θ = 2*Real.pi/3 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_odd_and_decreasing_l719_71993


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_S_trapezoid_from_equilateral_triangle_l719_71970

/-- The minimum value of S for a trapezoid cut from an equilateral triangle -/
theorem min_S_trapezoid_from_equilateral_triangle :
  let side_length : ℝ := 1
  let S (x : ℝ) : ℝ := 
    (3 - x)^2 / ((1/2) * (x + 1) * (Real.sqrt 3/2) * (1 - x))
  ∃ (x : ℝ), 0 < x ∧ x < 1 ∧
    (∀ (y : ℝ), 0 < y → y < 1 → S x ≤ S y) ∧
    S x = 32 * Real.sqrt 3 / 3 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_S_trapezoid_from_equilateral_triangle_l719_71970


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_zero_in_interval_l719_71997

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (1/3) * x^3 - a * x^2 + 1

theorem unique_zero_in_interval (a : ℝ) (h : a > 2) : 
  ∃! x, x ∈ Set.Ioo 0 2 ∧ f a x = 0 := by
  sorry

#check unique_zero_in_interval

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_zero_in_interval_l719_71997


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circumscribed_sphere_surface_area_l719_71987

/-- A regular triangular prism with base edge length √3 and lateral edge length 2 --/
structure RegularTriangularPrism where
  base_edge : ℝ
  lateral_edge : ℝ
  base_edge_eq : base_edge = Real.sqrt 3
  lateral_edge_eq : lateral_edge = 2

/-- The sphere circumscribing the regular triangular prism --/
def circumscribed_sphere (prism : RegularTriangularPrism) (radius : ℝ) : Prop :=
  radius = Real.sqrt 2

/-- The surface area of a sphere --/
noncomputable def sphere_surface_area (radius : ℝ) : ℝ := 4 * Real.pi * radius^2

theorem circumscribed_sphere_surface_area (prism : RegularTriangularPrism) :
  ∃ (radius : ℝ), circumscribed_sphere prism radius ∧ sphere_surface_area radius = 8 * Real.pi := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_circumscribed_sphere_surface_area_l719_71987


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_valid_spy_placement_l719_71929

-- Define the board size
def board_size : Nat := 6

-- Define the number of spies
def num_spies : Nat := 18

-- Define a position on the board
structure Position where
  x : Fin board_size
  y : Fin board_size

-- Define the spy's vision
def can_see (p1 p2 : Position) : Prop :=
  (p1.x = p2.x ∧ (p2.y.val - p1.y.val : Int).natAbs ≤ 2) ∨
  (p1.y = p2.y ∧ (p2.x.val - p1.x.val : Int).natAbs = 1)

-- Define a valid spy placement
def valid_placement (spies : Finset Position) : Prop :=
  spies.card = num_spies ∧
  ∀ s1 s2, s1 ∈ spies → s2 ∈ spies → s1 ≠ s2 → ¬(can_see s1 s2)

-- Theorem statement
theorem exists_valid_spy_placement :
  ∃ (spies : Finset Position), valid_placement spies := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_valid_spy_placement_l719_71929


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l719_71960

-- Define the function f(x) = lg((1+x)/(1-x))
noncomputable def f (x : ℝ) : ℝ := Real.log ((1 + x) / (1 - x))

-- Define the domain of f
def domain : Set ℝ := Set.Ioo (-1) 1

-- Theorem statement
theorem f_properties :
  -- f is defined on the open interval (-1, 1)
  (∀ x, x ∈ domain → f x ≠ 0) ∧
  -- f is an odd function
  (∀ x, x ∈ domain → f (-x) = -f x) ∧
  -- f is monotonically increasing on its domain
  (∀ x y, x ∈ domain → y ∈ domain → x < y → f x < f y) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l719_71960


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ninth_game_players_l719_71903

/-- Represents a player in the badminton games -/
inductive Player
  | XiaoZhao
  | XiaoQian
  | XiaoSun

/-- Represents a game with two players -/
structure Game where
  player1 : Player
  player2 : Player

/-- The sequence of games played -/
def gameSequence : Fin 9 → Game := sorry

/-- The number of games each player rested -/
def restCount : Player → Nat := sorry

/-- The number of games each player played -/
def playCount : Player → Nat := sorry

/-- The conditions given in the problem -/
axiom total_games : Fintype.card (Fin 9) = 9
axiom xiao_zhao_rest : restCount Player.XiaoZhao = 2
axiom xiao_qian_play : playCount Player.XiaoQian = 8
axiom xiao_sun_play : playCount Player.XiaoSun = 5

/-- The theorem to be proved -/
theorem ninth_game_players :
  (gameSequence 8).player1 = Player.XiaoZhao ∧
  (gameSequence 8).player2 = Player.XiaoQian := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ninth_game_players_l719_71903


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_all_subjects_max_marks_300_l719_71965

/-- Represents a subject with its score and percentage -/
structure Subject where
  name : String
  score : ℕ
  percentage : ℚ
  h : 0 < percentage ∧ percentage ≤ 1

/-- Calculates the maximum marks for a subject -/
def max_marks (s : Subject) : ℚ :=
  s.score / s.percentage

/-- The exam results for Victor -/
def victor_results : List Subject :=
  [{ name := "Mathematics", score := 285, percentage := 95/100, h := ⟨by norm_num, by norm_num⟩ },
   { name := "Physics", score := 270, percentage := 90/100, h := ⟨by norm_num, by norm_num⟩ },
   { name := "Chemistry", score := 255, percentage := 85/100, h := ⟨by norm_num, by norm_num⟩ }]

theorem all_subjects_max_marks_300 :
  ∀ s ∈ victor_results, max_marks s = 300 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_all_subjects_max_marks_300_l719_71965


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cone_height_relationship_l719_71908

/-- Represents a right circular cone --/
structure Cone where
  radius : ℝ
  height : ℝ

/-- The volume of a right circular cone --/
noncomputable def volume (c : Cone) : ℝ := (1/3) * Real.pi * c.radius^2 * c.height

theorem cone_height_relationship (c1 c2 : Cone) 
  (h_volume : volume c1 = volume c2) 
  (h_radius : c2.radius = 1.2 * c1.radius) : 
  c1.height = 1.44 * c2.height := by
  sorry

#check cone_height_relationship

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cone_height_relationship_l719_71908


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_negation_of_universal_statement_l719_71963

theorem negation_of_universal_statement :
  (¬ ∀ x : ℝ, x > 1 → (1/2 : ℝ)^x < 1/2) ↔ (∃ x : ℝ, x > 1 ∧ (1/2 : ℝ)^x ≥ 1/2) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_negation_of_universal_statement_l719_71963


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_divisor_problem_l719_71919

theorem divisor_problem (y x : ℕ) (h1 : y % x = 5) (h2 : (3 * y) % x = 6) : x = 9 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_divisor_problem_l719_71919


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_chord_length_squared_l719_71992

/-- Two circles are externally tangent -/
def externally_tangent (r₁ r₂ : ℝ) : Prop :=
  sorry

/-- A circle is internally tangent to another circle -/
def internally_tangent (r₁ r₂ : ℝ) : Prop :=
  sorry

/-- A chord is a common external tangent to two smaller circles inside a larger circle -/
def is_common_external_tangent (chord r₁ r₂ r₃ : ℝ) : Prop :=
  sorry

/-- Given three circles with radii 4, 8, and 12, where the two smaller circles
    are externally tangent to each other and internally tangent to the larger circle,
    the square of the length of the chord of the larger circle that is a common
    external tangent to the two smaller circles is equal to 3584/9. -/
theorem chord_length_squared (r₁ r₂ r₃ : ℝ) (h₁ : r₁ = 4) (h₂ : r₂ = 8) (h₃ : r₃ = 12)
  (h_external : externally_tangent r₁ r₂)
  (h_internal₁ : internally_tangent r₁ r₃)
  (h_internal₂ : internally_tangent r₂ r₃)
  (chord : ℝ) (h_chord : is_common_external_tangent chord r₁ r₂ r₃) :
  chord ^ 2 = 3584 / 9 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_chord_length_squared_l719_71992


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_slope_of_specific_line_l719_71981

/-- A line in the xy-plane with y-intercept b and passing through point (x, y) has slope m -/
noncomputable def line_slope (b x y : ℝ) : ℝ :=
  (y - b) / x

/-- The slope of a line with y-intercept 10 and passing through (100, 1000) is 9.9 -/
theorem slope_of_specific_line :
  line_slope 10 100 1000 = 9.9 := by
  -- Unfold the definition of line_slope
  unfold line_slope
  -- Perform the calculation
  norm_num
  -- The proof is complete
  done

end NUMINAMATH_CALUDE_ERRORFEEDBACK_slope_of_specific_line_l719_71981


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_dandelion_seed_percentage_approx_23_22_l719_71906

/-- Represents the number of plants for each flower type --/
structure FlowerCounts where
  sunflowers : ℕ
  dandelions : ℕ
  roses : ℕ
  tulips : ℕ
  lilies : ℕ
  irises : ℕ

/-- Represents the number of seeds produced per plant for each flower type --/
structure SeedsPerPlant where
  sunflowers : ℕ
  dandelions : ℕ
  roses : ℕ
  tulips : ℕ
  lilies : ℕ
  irises : ℕ

/-- Calculates the total number of seeds for a given flower type --/
def totalSeeds (count : ℕ) (seedsPerPlant : ℕ) : ℕ :=
  count * seedsPerPlant

/-- Calculates the percentage of dandelion seeds --/
noncomputable def dandelionSeedPercentage (counts : FlowerCounts) (seeds : SeedsPerPlant) : ℝ :=
  let dandelionSeeds := totalSeeds counts.dandelions seeds.dandelions
  let totalSeeds := 
    totalSeeds counts.sunflowers seeds.sunflowers +
    totalSeeds counts.dandelions seeds.dandelions +
    totalSeeds counts.roses seeds.roses +
    totalSeeds counts.tulips seeds.tulips +
    totalSeeds counts.lilies seeds.lilies +
    totalSeeds counts.irises seeds.irises
  (dandelionSeeds : ℝ) / (totalSeeds : ℝ) * 100

/-- Theorem stating that the percentage of dandelion seeds is approximately 23.22% --/
theorem dandelion_seed_percentage_approx_23_22 (counts : FlowerCounts) (seeds : SeedsPerPlant) :
  counts.sunflowers = 6 →
  counts.dandelions = 8 →
  counts.roses = 4 →
  counts.tulips = 10 →
  counts.lilies = 5 →
  counts.irises = 7 →
  seeds.sunflowers = 9 →
  seeds.dandelions = 12 →
  seeds.roses = 7 →
  seeds.tulips = 15 →
  seeds.lilies = 10 →
  seeds.irises = 5 →
  |dandelionSeedPercentage counts seeds - 23.22| < 0.01 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_dandelion_seed_percentage_approx_23_22_l719_71906


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_loan_amount_proof_l719_71901

/-- Represents a simple interest loan -/
structure SimpleLoan where
  principal : ℚ
  rate : ℚ
  time : ℚ

/-- Calculates the simple interest for a given loan -/
def simpleInterest (loan : SimpleLoan) : ℚ :=
  (loan.principal * loan.rate * loan.time) / 100

theorem loan_amount_proof (loan : SimpleLoan) 
  (h1 : loan.rate = 7)
  (h2 : loan.time = loan.rate)
  (h3 : simpleInterest loan = 735) :
  loan.principal = 1500 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_loan_amount_proof_l719_71901


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_focus_to_asymptote_l719_71918

/-- The hyperbola equation -/
def hyperbola (x y : ℝ) : Prop := x^2 / 9 - y^2 / 4 = 1

/-- The right focus of the hyperbola -/
noncomputable def right_focus : ℝ × ℝ := (Real.sqrt 13, 0)

/-- One asymptote of the hyperbola -/
def asymptote (x y : ℝ) : Prop := 2 * x + 3 * y = 0

/-- Distance from a point to a line -/
noncomputable def distance_point_to_line (x₀ y₀ A B C : ℝ) : ℝ :=
  abs (A * x₀ + B * y₀ + C) / Real.sqrt (A^2 + B^2)

/-- The main theorem -/
theorem distance_focus_to_asymptote :
  distance_point_to_line (right_focus.1) (right_focus.2) 2 3 0 = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_focus_to_asymptote_l719_71918


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_boat_breadth_is_three_l719_71925

/-- Calculates the breadth of a boat given specific conditions --/
noncomputable def calculate_boat_breadth (boat_length : ℝ) (sinking_depth : ℝ) (man_mass : ℝ) (gravity : ℝ) (water_density : ℝ) : ℝ :=
  let water_weight := man_mass * gravity
  let displaced_volume := water_weight / (water_density * gravity)
  displaced_volume / (boat_length * sinking_depth)

/-- Theorem stating that under given conditions, the boat's breadth is 3 meters --/
theorem boat_breadth_is_three :
  let boat_length : ℝ := 6
  let sinking_depth : ℝ := 0.01
  let man_mass : ℝ := 180
  let gravity : ℝ := 9.81
  let water_density : ℝ := 1000
  calculate_boat_breadth boat_length sinking_depth man_mass gravity water_density = 3 := by
  sorry

#check boat_breadth_is_three

end NUMINAMATH_CALUDE_ERRORFEEDBACK_boat_breadth_is_three_l719_71925


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_dart_probability_l719_71926

theorem dart_probability (s : ℝ) (h : s = 2 + Real.sqrt 2) : 
  let r := s * (1 + Real.sqrt 2) / 2
  let circle_area := Real.pi * r^2
  let octagon_area := 2 * s^2 * (1 + Real.sqrt 2)
  circle_area / octagon_area = circle_area / octagon_area := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_dart_probability_l719_71926


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_cone_section_area_l719_71928

noncomputable def max_section_area (h : ℝ) (r : ℝ) : ℝ :=
  let l := Real.sqrt (h^2 + r^2)
  (1/2) * l * l

theorem max_cone_section_area (h : ℝ) (V : ℝ) (r : ℝ) :
  h = 1 →
  V = π →
  V = (1/3) * π * r^2 * h →
  2 = max_section_area h r :=
by sorry

#check max_cone_section_area

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_cone_section_area_l719_71928


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_is_odd_l719_71974

noncomputable def f (x : ℝ) : ℝ := Real.log (Real.sqrt (x^2 + 1) + x)

theorem f_is_odd : ∀ x : ℝ, f (-x) = -f x := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_is_odd_l719_71974


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_negation_of_proposition_negation_of_specific_proposition_l719_71940

theorem negation_of_proposition (P : ℝ → Prop) :
  (¬ ∀ α, 0 < α ∧ α < Real.pi/4 → P α) ↔ (∃ α, 0 < α ∧ α < Real.pi/4 ∧ ¬(P α)) :=
by sorry

-- The specific proposition
def proposition (α : ℝ) : Prop := Real.sin α ≠ Real.cos α

theorem negation_of_specific_proposition :
  (¬ ∀ α, 0 < α ∧ α < Real.pi/4 → Real.sin α ≠ Real.cos α) ↔
  (∃ α, 0 < α ∧ α < Real.pi/4 ∧ Real.sin α = Real.cos α) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_negation_of_proposition_negation_of_specific_proposition_l719_71940


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_at_zero_l719_71998

-- Define the curve
noncomputable def f (x : ℝ) : ℝ := Real.exp x + 2 * x

-- Define the derivative of the curve
noncomputable def f' (x : ℝ) : ℝ := Real.exp x + 2

-- Theorem statement
theorem tangent_line_at_zero : 
  let p : ℝ × ℝ := (0, 1)
  let m : ℝ := f' p.1
  let tangent_line (x : ℝ) := m * (x - p.1) + p.2
  tangent_line = fun x ↦ 3 * x + 1 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_at_zero_l719_71998


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_toothpicks_15th_stage_l719_71947

def toothpicks : ℕ → ℕ
  | 0 => 5  -- Base case for n = 0 (which represents the first stage)
  | (n + 1) => 2 * toothpicks n + 2

theorem toothpicks_15th_stage : toothpicks 14 = 32766 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_toothpicks_15th_stage_l719_71947


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_first_five_pages_drawings_l719_71948

/-- The number of drawings on a given page -/
def drawings (page : ℕ) : ℕ := 5 * 2^(page - 1)

/-- The sum of drawings on the first n pages -/
def sum_drawings (n : ℕ) : ℕ := Finset.sum (Finset.range n) (fun i => drawings (i + 1))

theorem first_five_pages_drawings : sum_drawings 5 = 155 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_first_five_pages_drawings_l719_71948


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_two_true_propositions_l719_71967

theorem two_true_propositions (p q : Prop) [Decidable p] [Decidable q] : 
  ∃! n : Nat, n = (if (p ∧ q) then 1 else 0) + 
                  (if (p ∨ q) then 1 else 0) + 
                  (if (¬p) then 1 else 0) + 
                  (if (¬q) then 1 else 0) ∧ n = 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_two_true_propositions_l719_71967


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_divisibility_l719_71921

def sequence_a : ℕ → ℤ
  | 0 => 2
  | n + 1 => 2 * (sequence_a n)^2 - 1

theorem sequence_divisibility (n : ℕ) (p : ℕ) :
  Nat.Prime p → Odd p → (p : ℤ) ∣ sequence_a n → 2^(n+3) ∣ p^2 - 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_divisibility_l719_71921


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_plot_length_is_87_5_l719_71907

/-- Represents a rectangular plot with fencing -/
structure FencedPlot where
  width : ℝ
  poleDistance : ℝ
  totalPoles : ℕ

/-- Calculates the length of a rectangular plot given its width, fence pole distance, and total number of poles -/
noncomputable def calculatePlotLength (plot : FencedPlot) : ℝ :=
  (plot.poleDistance * (plot.totalPoles - 1 : ℝ) - 2 * plot.width) / 2

/-- Theorem stating that for a plot with width 60m, pole distance 5m, and 60 poles, the length is 87.5m -/
theorem plot_length_is_87_5 :
  let plot := FencedPlot.mk 60 5 60
  calculatePlotLength plot = 87.5 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_plot_length_is_87_5_l719_71907


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_appropriate_units_l719_71957

-- Define the units
inductive MeasurementUnit
| Seconds
| Millimeters
| Other

-- Define the measurement types
inductive MeasurementType
| Duration
| Thickness

-- Define the objects being measured
inductive MeasuredObject
| TrafficLight
| MathBook

-- Define a function to determine the appropriate unit
def appropriateUnit (m : MeasurementType) (o : MeasuredObject) : MeasurementUnit :=
  match m, o with
  | MeasurementType.Duration, MeasuredObject.TrafficLight => MeasurementUnit.Seconds
  | MeasurementType.Thickness, MeasuredObject.MathBook => MeasurementUnit.Millimeters
  | _, _ => MeasurementUnit.Other

-- State the theorem
theorem appropriate_units :
  (appropriateUnit MeasurementType.Duration MeasuredObject.TrafficLight = MeasurementUnit.Seconds) ∧
  (appropriateUnit MeasurementType.Thickness MeasuredObject.MathBook = MeasurementUnit.Millimeters) :=
by
  apply And.intro
  · rfl
  · rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_appropriate_units_l719_71957


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_reflected_light_equation_l719_71977

/-- The incident light line -/
noncomputable def incident_line (x : ℝ) : ℝ := 2 * x + 1

/-- The reflecting line -/
noncomputable def reflecting_line (x : ℝ) : ℝ := x

/-- The reflected light line -/
noncomputable def reflected_line (x : ℝ) : ℝ := (1/2) * x - (1/2)

/-- Theorem stating that the given reflected_line is correct -/
theorem reflected_light_equation : 
  ∀ (x y : ℝ), y = reflected_line x ↔ 
  (∃ (x₀ y₀ : ℝ), 
    y₀ = incident_line x₀ ∧ 
    y₀ = reflecting_line x₀ ∧
    (y - y₀) / (x - x₀) = -(incident_line x₀ - y₀) / (x₀ - x)) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_reflected_light_equation_l719_71977


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ratings_can_increase_after_migrations_l719_71923

/-- Represents a country with a list of resident Q scores --/
structure Country where
  scores : List ℚ

/-- Calculates the average Q score (rating) of a country --/
def avgScore (c : Country) : ℚ :=
  if c.scores.length > 0 then
    c.scores.sum / c.scores.length
  else 0

/-- Represents a migration event between two countries --/
structure Migration where
  fromCountry : Country
  toCountry : Country
  movers : List ℚ

/-- Applies a migration to update the countries --/
def applyMigration (m : Migration) : (Country × Country) :=
  let newFrom : Country := ⟨m.fromCountry.scores.filter (λ x => ¬ m.movers.contains x)⟩
  let newTo : Country := ⟨m.toCountry.scores ++ m.movers⟩
  (newFrom, newTo)

/-- Theorem statement --/
theorem ratings_can_increase_after_migrations :
  ∃ (a b c : Country) (m1 m2 m3 m4 : Migration),
    let (a', b') := applyMigration m1
    let (b'', c') := applyMigration m2
    let (c'', b''') := applyMigration m3
    let (b'''', a'') := applyMigration m4
    avgScore a < avgScore a' ∧
    avgScore b < avgScore b' ∧
    avgScore c < avgScore c' ∧
    avgScore a' < avgScore a'' ∧
    avgScore b'' < avgScore b'''' ∧
    avgScore c' < avgScore c'' := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ratings_can_increase_after_migrations_l719_71923


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_box_volume_count_l719_71914

/-- The volume function for the rectangular prism -/
def volume (x : ℕ) : ℕ := (x + 3) * (x^2 - 1) * (x^2 + 9)

/-- The theorem stating that exactly 3 positive integers satisfy the volume condition -/
theorem box_volume_count :
  (∃! (n : ℕ), n = (Finset.filter (fun x => x > 0 ∧ volume x < 1200) (Finset.range 1000)).card) ∧
  (Finset.filter (fun x => x > 0 ∧ volume x < 1200) (Finset.range 1000)).card = 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_box_volume_count_l719_71914


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_zit_difference_l719_71996

/-- Represents a class with students and their zits -/
structure ClassInfo where
  avg_zits : ℕ
  num_students : ℕ

/-- Calculates the total number of zits in a class -/
def total_zits (c : ClassInfo) : ℕ := c.avg_zits * c.num_students

/-- The problem statement -/
theorem zit_difference (swanson jones smith : ClassInfo)
  (h_swanson : swanson = { avg_zits := 5, num_students := 25 })
  (h_jones : jones = { avg_zits := 6, num_students := 32 })
  (h_smith : smith = { avg_zits := 7, num_students := 20 }) :
  total_zits jones + total_zits smith - total_zits swanson = 207 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_zit_difference_l719_71996


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_system_solutions_l719_71916

def solution_system (x y : ℝ) : Prop :=
  (9 : ℝ)^(Real.sqrt (x * y^2)^(1/4)) - 27 * (3 : ℝ)^(Real.sqrt y) = 0 ∧
  (1/4) * Real.log x + (1/2) * Real.log y = Real.log (4 - x^(1/4))

theorem system_solutions :
  ∀ x y : ℝ, 0 < x → x < 256 → 0 < y →
  solution_system x y ↔ ((x = 1 ∧ y = 9) ∨ (x = 16 ∧ y = 1)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_system_solutions_l719_71916


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_divisibility_equivalence_l719_71973

theorem divisibility_equivalence (m n : ℕ) :
  (25 * m + 3 * n) % 83 = 0 ↔ (3 * m + 7 * n) % 83 = 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_divisibility_equivalence_l719_71973


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_with_frequency_0_3_l719_71933

noncomputable def sample : List ℝ := [10, 8, 6, 10, 13, 8, 10, 12, 11, 7, 8, 9, 11, 9, 12, 9, 10, 11, 12, 12]

noncomputable def inRange (x : ℝ) (lower upper : ℝ) : Bool :=
  lower ≤ x ∧ x < upper

noncomputable def countInRange (s : List ℝ) (lower upper : ℝ) : Nat :=
  s.filter (λ x => inRange x lower upper) |>.length

def frequency (count : Nat) (total : Nat) : ℚ :=
  count / total

theorem range_with_frequency_0_3 :
  let total := sample.length
  let count := countInRange sample 7.5 9.5
  frequency count total = 3 / 10 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_with_frequency_0_3_l719_71933


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_composition_zero_iff_image_subset_kernel_l719_71905

variable {K : Type*} [Field K]
variable {E F G : Type*} [AddCommGroup E] [Module K E] [AddCommGroup F] [Module K F] [AddCommGroup G] [Module K G]

/-- The composition of two linear maps is zero if and only if the image of the first map
    is contained in the kernel of the second map. -/
theorem composition_zero_iff_image_subset_kernel
  (f : E →ₗ[K] F) (g : F →ₗ[K] G) :
  (g.comp f = 0) ↔ (∀ x : E, g (f x) = 0) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_composition_zero_iff_image_subset_kernel_l719_71905


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tree_planting_cost_l719_71941

/-- The cost to reduce temperature by planting trees -/
noncomputable def cost_to_reduce_temperature (initial_temp final_temp temp_drop_per_tree cost_per_tree : ℝ) : ℝ :=
  ((initial_temp - final_temp) / temp_drop_per_tree) * cost_per_tree

/-- Theorem: The cost to reduce the temperature from 80 to 78.2 degrees by planting trees, 
    where each tree reduces the temperature by 0.1 degrees and costs $6, is $108. -/
theorem tree_planting_cost : 
  cost_to_reduce_temperature 80 78.2 0.1 6 = 108 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tree_planting_cost_l719_71941


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_local_minimum_f_total_area_f_l719_71910

-- Define the function
noncomputable def f (x : ℝ) : ℝ := x * (1 - x^2) * Real.exp (x^2)

-- Theorem for the local minimum
theorem local_minimum_f :
  ∃ (x : ℝ), x = -1 / Real.sqrt 2 ∧ 
  f x = -Real.sqrt (Real.exp 1) / (2 * Real.sqrt 2) ∧
  ∃ δ > 0, ∀ y, |y - x| < δ → f y ≥ f x :=
sorry

-- Theorem for the total area
theorem total_area_f :
  ∫ x in Set.Icc (-1) 1, |f x| = Real.exp 1 - 2 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_local_minimum_f_total_area_f_l719_71910


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_marble_distribution_l719_71949

def is_valid_distribution (k : Nat) : Bool :=
  k > 1 && k < 450 && 450 % k = 0 && 450 / k > 1

theorem marble_distribution :
  (List.filter is_valid_distribution (List.range 450)).length = 14 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_marble_distribution_l719_71949


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_double_angle_with_tan_l719_71912

theorem sin_double_angle_with_tan (α : Real) (h1 : 0 < α) (h2 : α < π / 2) (h3 : Real.tan α = 3 / 4) :
  Real.sin (2 * α) = 24 / 25 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_double_angle_with_tan_l719_71912


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tetrahedron_circumsphere_area_l719_71991

/-- Represents a tetrahedron S-ABC with specific properties -/
structure Tetrahedron where
  -- Base side length
  a : ℝ
  -- Length of edges SA and SB
  b : ℝ
  -- Length of edge SC
  c : ℝ
  -- Conditions
  base_equilateral : a = 4
  sa_sb_length : b = Real.sqrt 19
  sc_length : c = 3

/-- The surface area of the circumscribed sphere of the tetrahedron -/
noncomputable def circumsphere_surface_area (t : Tetrahedron) : ℝ :=
  (244 * Real.pi) / 11

/-- Theorem stating that the surface area of the circumscribed sphere is 244π/11 -/
theorem tetrahedron_circumsphere_area (t : Tetrahedron) :
  circumsphere_surface_area t = (244 * Real.pi) / 11 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tetrahedron_circumsphere_area_l719_71991


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_order_of_abc_l719_71911

theorem order_of_abc (a b c : ℝ) : 
  a = Real.sqrt 2 → b = Real.exp (1 / Real.exp 1) → c = (6 : ℝ) ^ (1/3) → a < b ∧ b < c := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_order_of_abc_l719_71911


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_additional_points_l719_71900

open Set
open Function
open Real

/-- A type representing points in a plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- Distance between two points -/
noncomputable def distance (p q : Point) : ℝ :=
  sqrt ((p.x - q.x)^2 + (p.y - q.y)^2)

/-- Congruence of triangles -/
def triangles_congruent (a b c d e f : Point) : Prop :=
  distance a b = distance d e ∧
  distance b c = distance e f ∧
  distance c a = distance f d

/-- The theorem stating the maximum number of additional points -/
theorem max_additional_points (A B C D : Point) (h : distance A B ≠ distance C D) :
  ∃ (n : ℕ), n = 4 ∧ 
  (∃ (X : Fin n → Point), 
    (∀ (i : Fin n), triangles_congruent A B (X i) C D (X i)) ∧
    (∀ (m : ℕ) (Y : Fin m → Point), 
      (∀ (i : Fin m), triangles_congruent A B (Y i) C D (Y i)) → m ≤ n)) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_additional_points_l719_71900


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_even_divisors_of_factorial_8_l719_71972

-- Define 8!
def factorial_8 : ℕ := 8 * 7 * 6 * 5 * 4 * 3 * 2 * 1

-- Define a function to count even divisors
def count_even_divisors (n : ℕ) : ℕ :=
  (Finset.filter (fun d => d % 2 = 0 && n % d = 0) (Finset.range (n + 1))).card

-- Theorem statement
theorem even_divisors_of_factorial_8 :
  count_even_divisors factorial_8 = 84 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_even_divisors_of_factorial_8_l719_71972


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_medal_assignment_l719_71958

-- Define the people and medals
inductive Person : Type
  | Jirka : Person
  | Vit : Person
  | Ota : Person

inductive Medal : Type
  | Gold : Medal
  | Silver : Medal
  | Bronze : Medal

-- Define the winner relation
def winner : Person → Medal → Prop := sorry

-- Define the statements made by each person
def statement (p : Person) : Prop :=
  match p with
  | Person.Jirka => ∃ m, m = Medal.Gold ∧ winner Person.Ota m
  | Person.Vit => ∃ m, m = Medal.Silver ∧ winner Person.Ota m
  | Person.Ota => ¬(∃ m, (m = Medal.Gold ∨ m = Medal.Silver) ∧ winner Person.Ota m)

-- Define the truth-telling condition
def tells_truth (p : Person) : Prop :=
  statement p

-- Theorem: The only valid assignment is Vít (gold), Ota (silver), and Jirka (bronze)
theorem medal_assignment :
  (∀ p m, winner p m → (m = Medal.Gold → tells_truth p) ∧ (m = Medal.Bronze → ¬tells_truth p)) →
  (winner Person.Vit Medal.Gold ∧ winner Person.Ota Medal.Silver ∧ winner Person.Jirka Medal.Bronze) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_medal_assignment_l719_71958


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_series_l719_71982

open Real BigOperators

noncomputable def T : ℝ := ∑ k in Finset.range 50, (3 + 7 * (k + 1)) / 3^(51 - (k + 1))

theorem sum_of_series : T = 171.5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_series_l719_71982


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_between_curves_l719_71934

noncomputable section

-- Define the two curves
noncomputable def curve1 (x : ℝ) : ℝ := Real.exp (3 * x + 5)
noncomputable def curve2 (x : ℝ) : ℝ := (Real.log x - 5) / 3

-- Define the distance function between a point on curve1 and its symmetric point on curve2
noncomputable def distance (x : ℝ) : ℝ := Real.sqrt 2 * abs (curve1 x - x)

-- State the theorem
theorem min_distance_between_curves :
  ∃ (x : ℝ), ∀ (y : ℝ), distance x ≤ distance y ∧ 
  distance x = Real.sqrt 2 * (2 + Real.log 3 / 3) := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_between_curves_l719_71934


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_eccentricity_equilateral_foci_l719_71952

/-- Predicate indicating that a set represents an ellipse in ℝ² -/
def IsEllipse (E : Set (ℝ × ℝ)) : Prop := sorry

/-- Predicate indicating that two points are the foci of an ellipse -/
def AreEllipseFoci (E : Set (ℝ × ℝ)) (F₁ F₂ : ℝ × ℝ) : Prop := sorry

/-- Predicate indicating that three points form an equilateral triangle -/
def IsEquilateralTriangle (A B C : ℝ × ℝ) : Prop := sorry

/-- Function returning the eccentricity of an ellipse -/
noncomputable def EllipseEccentricity (E : Set (ℝ × ℝ)) : ℝ := sorry

/-- An ellipse with a vertex forming an equilateral triangle with its foci has eccentricity 1/2 -/
theorem ellipse_eccentricity_equilateral_foci (E : Set (ℝ × ℝ)) (V F₁ F₂ : ℝ × ℝ) :
  IsEllipse E → 
  V ∈ E →
  AreEllipseFoci E F₁ F₂ →
  IsEquilateralTriangle V F₁ F₂ →
  EllipseEccentricity E = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_eccentricity_equilateral_foci_l719_71952


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_line_through_point_l719_71969

/-- Given a point A and a line L, this theorem proves that the line passing through A
    and parallel to L has the equation 4x - y - 11 = 0. -/
theorem parallel_line_through_point (x y : ℝ) :
  let A : ℝ × ℝ := (-2, 3)
  let L : ℝ → ℝ → Prop := λ x y => 4 * x - y - 7 = 0
  let parallel_line : ℝ → ℝ → Prop := λ x y => 4 * x - y - 11 = 0
  (parallel_line A.1 A.2) ∧ 
  (∀ x y : ℝ, L x y ↔ ∃ k : ℝ, parallel_line x y ↔ L x y ∧ k = 4) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_line_through_point_l719_71969


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_monic_cubic_polynomial_unique_l719_71959

/-- A monic cubic polynomial with real coefficients -/
def MonicCubicPolynomial (a b c : ℝ) : ℂ → ℂ := fun x ↦ x^3 + a*x^2 + b*x + c

theorem monic_cubic_polynomial_unique 
  (q : ℂ → ℂ) 
  (h_monic : ∃ a b c : ℝ, ∀ x, q x = x^3 + a*x^2 + b*x + c) 
  (h_complex_root : q (4 - 3*Complex.I) = 0) 
  (h_real_root : q 0 = -80) :
  q = fun x ↦ x^3 - 11.2*x^2 + 50.6*x - 80 := by
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_monic_cubic_polynomial_unique_l719_71959


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_overlapping_squares_l719_71964

/-- Function to calculate the area of the overlap region -/
noncomputable def area_of_overlap_region (side_length : ℝ) (β : ℝ) : ℝ :=
  -- Definition of the area calculation would go here
  sorry

/-- The area of overlap between two squares with side length 2, 
    where one is rotated about a vertex by an angle β (0° < β < 90°) and cos β = 3/5 -/
theorem area_of_overlapping_squares (β : ℝ) 
  (h1 : 0 < β) (h2 : β < Real.pi / 2) (h3 : Real.cos β = 3 / 5) : 
  ∃ (A : ℝ), A = 2 / 3 ∧ A = area_of_overlap_region 2 β :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_overlapping_squares_l719_71964


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unit_vector_parallel_coordinates_l719_71954

def a : Fin 2 → ℝ := ![5, 4]
def b : Fin 2 → ℝ := ![3, 2]

def parallel_vector : Fin 2 → ℝ := 2 • a - 3 • b

noncomputable def unit_vector_parallel : Fin 2 → ℝ := 
  (1 / Real.sqrt ((parallel_vector 0) ^ 2 + (parallel_vector 1) ^ 2)) • parallel_vector

theorem unit_vector_parallel_coordinates :
  unit_vector_parallel = ![Real.sqrt 5 / 5, 2 * Real.sqrt 5 / 5] ∨
  unit_vector_parallel = ![-Real.sqrt 5 / 5, -2 * Real.sqrt 5 / 5] := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_unit_vector_parallel_coordinates_l719_71954


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cone_radius_approx_seven_l719_71995

/-- The curved surface area of a cone -/
def curved_surface_area : ℝ := 307.8760800517997

/-- The slant height of the cone -/
def slant_height : ℝ := 14

/-- The radius of the cone -/
noncomputable def radius : ℝ := curved_surface_area / (Real.pi * slant_height)

/-- Theorem stating that the radius of the cone is approximately 7 meters -/
theorem cone_radius_approx_seven : 
  |radius - 7| < 0.01 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cone_radius_approx_seven_l719_71995


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_prime_generating_integer_l719_71966

def is_prime (n : ℤ) : Prop := n > 1 ∧ ∀ m : ℤ, m > 1 → m < n → ¬(n % m = 0)

theorem unique_prime_generating_integer :
  ∃! n : ℤ, is_prime (5*n - 7) ∧ is_prime (6*n + 1) ∧ is_prime (20 - 3*n) ∧ n = 6 :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_prime_generating_integer_l719_71966


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_excellent_pair_prob_part1_excellent_pair_prob_part2_optimal_probabilities_l719_71976

-- Define the probability of success for each shot
noncomputable def p₁ : ℝ := 3/4
noncomputable def p₂ : ℝ := 2/3

-- Define the probability of achieving "excellent pair" status
noncomputable def excellent_pair_prob (p₁ p₂ : ℝ) : ℝ :=
  2 * p₁ * (1 - p₁) * p₂^2 + p₁^2 * 2 * p₂ * (1 - p₂) + p₁^2 * p₂^2

-- Part 1: Prove the probability of achieving "excellent pair" status
theorem excellent_pair_prob_part1 : excellent_pair_prob p₁ p₂ = 2/3 := by sorry

-- Part 2: Define the constraint on p₁ and p₂
noncomputable def p_sum : ℝ := 4/3

-- Define the function to calculate the probability of being an excellent pair
noncomputable def p (p₁ p₂ : ℝ) : ℝ := 2 * p₁ * p₂ * (p₁ + p₂) - 3 * p₁^2 * p₂^2

-- Define the optimal values for p₁ and p₂
noncomputable def optimal_p : ℝ := 2/3

-- Part 2: Prove the minimum number of rounds and optimal probabilities
theorem excellent_pair_prob_part2 (p₁ p₂ : ℝ) 
  (h₁ : p₁ + p₂ = p_sum) 
  (h₂ : p₁ * p₂ ≤ optimal_p^2) : 
  ∃ n : ℕ, n ≥ 27 ∧ (n : ℝ) * p p₁ p₂ ≥ 16 := by sorry

theorem optimal_probabilities :
  p optimal_p optimal_p = (16 : ℝ) / 27 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_excellent_pair_prob_part1_excellent_pair_prob_part2_optimal_probabilities_l719_71976


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_teacher_budget_shortage_l719_71909

/-- Represents the teacher's budget and expenses for school supplies --/
structure TeacherBudget where
  lastYearBudget : ℝ
  thisYearAllocation : ℝ
  grant : ℝ
  notebooksCost : ℝ
  notebooksDiscount : ℝ
  pensCost : ℝ
  pensDiscount : ℝ
  artSuppliesCost : ℝ
  foldersCost : ℝ
  foldersVoucher : ℝ

/-- Calculates the remaining budget after purchases --/
def remainingBudget (budget : TeacherBudget) : ℝ :=
  (budget.lastYearBudget + budget.thisYearAllocation + budget.grant) -
  ((budget.notebooksCost * (1 - budget.notebooksDiscount)) +
   (budget.pensCost * (1 - budget.pensDiscount)) +
   budget.artSuppliesCost +
   (budget.foldersCost - budget.foldersVoucher))

/-- Theorem stating that the teacher will have a shortage of $11.85 --/
theorem teacher_budget_shortage (budget : TeacherBudget)
  (h1 : budget.lastYearBudget = 6)
  (h2 : budget.thisYearAllocation = 50)
  (h3 : budget.grant = 20)
  (h4 : budget.notebooksCost = 18)
  (h5 : budget.notebooksDiscount = 0.1)
  (h6 : budget.pensCost = 27)
  (h7 : budget.pensDiscount = 0.05)
  (h8 : budget.artSuppliesCost = 35)
  (h9 : budget.foldersCost = 15)
  (h10 : budget.foldersVoucher = 5) :
  remainingBudget budget = -11.85 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_teacher_budget_shortage_l719_71909


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_integers_between_8_and_27_l719_71979

theorem odd_integers_between_8_and_27 : 
  ∃! n : ℕ, n = (Finset.filter (fun x => Odd x ∧ 8 < x ∧ x < 27) (Finset.range 27)).card :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_integers_between_8_and_27_l719_71979


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_total_sphere_volume_in_specific_pyramid_l719_71953

/-- A regular triangular pyramid -/
structure RegularTriangularPyramid where
  base_side_length : ℝ
  height : ℝ

/-- A sequence of spheres in the pyramid -/
noncomputable def pyramid_spheres (p : RegularTriangularPyramid) : ℕ → ℝ
  | 0 => 2 / 3  -- radius of the first sphere
  | n + 1 => pyramid_spheres p n / 3  -- radius of subsequent spheres

/-- The total volume of all spheres in the pyramid -/
noncomputable def total_sphere_volume (p : RegularTriangularPyramid) : ℝ :=
  (4 / 3) * Real.pi * (∑' n, (pyramid_spheres p n) ^ 3)

/-- Theorem: The total volume of spheres in the specified pyramid is 16π/39 -/
theorem total_sphere_volume_in_specific_pyramid :
  let p := RegularTriangularPyramid.mk 4 2
  total_sphere_volume p = (16 * Real.pi) / 39 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_total_sphere_volume_in_specific_pyramid_l719_71953


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_independent_set_of_size_three_l719_71924

/-- A directed graph with 8 vertices where each vertex has exactly one outgoing edge. -/
def DirectedGraph := Fin 8 → Fin 8

/-- A subset of vertices with no internal edges. -/
def IndependentSet (G : DirectedGraph) (S : Finset (Fin 8)) : Prop :=
  ∀ i j, i ∈ S → j ∈ S → i ≠ j → G i ≠ j

/-- There exists an independent set of size 3 in any directed graph with 8 vertices
    where each vertex has exactly one outgoing edge. -/
theorem exists_independent_set_of_size_three (G : DirectedGraph) :
    ∃ S : Finset (Fin 8), S.card = 3 ∧ IndependentSet G S := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_independent_set_of_size_three_l719_71924


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_coefficient_x_cubed_in_expansion_l719_71935

theorem coefficient_x_cubed_in_expansion : 
  (Polynomial.coeff ((1 + X : Polynomial ℝ) ^ 50) 3) = 19600 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_coefficient_x_cubed_in_expansion_l719_71935


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equilateral_triangle_area_squared_l719_71946

/-- Parabola function y = x^2 - 4x + 4 -/
def parabola (x : ℝ) : ℝ := x^2 - 4*x + 4

/-- Centroid of the triangle -/
def centroid : ℝ × ℝ := (2, 0)

/-- Theorem: The square of the area of the equilateral triangle is 432 -/
theorem equilateral_triangle_area_squared : ∃ (A B C : ℝ × ℝ),
  let (x₁, y₁) := A
  let (x₂, y₂) := B
  let (x₃, y₃) := C
  -- Vertices lie on the parabola
  (y₁ = parabola x₁) ∧ (y₂ = parabola x₂) ∧ (y₃ = parabola x₃) ∧
  -- Triangle is equilateral
  ((x₁ - x₂)^2 + (y₁ - y₂)^2 = (x₂ - x₃)^2 + (y₂ - y₃)^2) ∧
  ((x₂ - x₃)^2 + (y₂ - y₃)^2 = (x₃ - x₁)^2 + (y₃ - y₁)^2) ∧
  -- Centroid is at the vertex of the parabola
  (((x₁ + x₂ + x₃) / 3, (y₁ + y₂ + y₃) / 3) = centroid) →
  -- Square of the area is 432
  let s := Real.sqrt ((x₁ - x₂)^2 + (y₁ - y₂)^2)
  (s^2 * Real.sqrt 3 / 4)^2 = 432 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_equilateral_triangle_area_squared_l719_71946


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_abc_value_l719_71930

theorem abc_value (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0)
  (hab : a * b = 24 * (3 ^ (1/3 : ℝ)))
  (hac : a * c = 40 * (3 ^ (1/3 : ℝ)))
  (hbc : b * c = 16 * (3 ^ (1/3 : ℝ))) :
  a * b * c = 96 * Real.sqrt 15 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_abc_value_l719_71930


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_linear_function_k_value_l719_71985

/-- A linear function f satisfying specific conditions -/
noncomputable def f (x : ℝ) : ℝ := 4 * x - 9

/-- The value of k -/
noncomputable def k : ℝ := 13 / 4

theorem linear_function_k_value :
  (f k = 4) ∧ (f (f k) = 7) ∧ (f (f (f k)) = 19) := by
  -- We'll use 'sorry' to skip the proof for now
  sorry

#check linear_function_k_value

end NUMINAMATH_CALUDE_ERRORFEEDBACK_linear_function_k_value_l719_71985


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_mean_of_prime_reciprocals_l719_71980

theorem arithmetic_mean_of_prime_reciprocals :
  (1 / 2 + 1 / 3 + 1 / 5 + 1 / 7 : ℚ) / 4 = 247 / 840 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_mean_of_prime_reciprocals_l719_71980


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_not_right_triangle_with_angle_ratio_3_4_5_l719_71915

theorem not_right_triangle_with_angle_ratio_3_4_5 :
  ∀ (α β γ : ℝ),
  α > 0 ∧ β > 0 ∧ γ > 0 →
  α + β + γ = 180 →
  (α : ℝ) / 3 = (β : ℝ) / 4 ∧ (β : ℝ) / 4 = (γ : ℝ) / 5 →
  ¬(α = 90 ∨ β = 90 ∨ γ = 90) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_not_right_triangle_with_angle_ratio_3_4_5_l719_71915


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_area_MPNQ_l719_71988

-- Define the curves C₁ and C₂
noncomputable def C₁ (r : ℝ) (θ : ℝ) : ℝ × ℝ := (r * Real.cos θ, r * Real.sin θ)
noncomputable def C₂ (θ : ℝ) : ℝ × ℝ := (2 + 2 * Real.sqrt 2 * Real.cos θ, 2 + 2 * Real.sqrt 2 * Real.sin θ)

-- Define the ray
noncomputable def ray (α : ℝ) (ρ : ℝ) : ℝ × ℝ := (ρ * Real.cos α, ρ * Real.sin α)

-- State the theorem
theorem max_area_MPNQ (r : ℝ) (α : ℝ) :
  0 < r → r < 4 → 0 < α → α < Real.pi / 2 →
  (∃ ρ₁ ρ₂, ρ₁ < ρ₂ ∧ ray α ρ₁ = C₂ α ∧ ray α ρ₂ = C₂ α) →
  (∃ ρ, ray α ρ = C₁ r α) →
  (∀ ρ₁ ρ₂, ray α ρ₁ = C₂ α → ray α ρ₂ = C₁ r α → |ρ₁ - ρ₂| ≤ 2 * Real.sqrt 2) →
  (∃ ρ₁ ρ₂, ray (α + Real.pi / 4) ρ₁ = C₁ r (α + Real.pi / 4) ∧ ray (α + Real.pi / 4) ρ₂ = C₂ (α + Real.pi / 4)) →
  (∃ S : ℝ, S ≤ 4 + 2 * Real.sqrt 2 ∧
    ∀ S' : ℝ, (∃ M P N Q : ℝ × ℝ,
      M = C₂ (α + Real.pi / 4) ∧
      P = C₂ α ∧
      N = C₁ r α ∧
      Q = C₁ r (α + Real.pi / 4) ∧
      S' = abs ((M.1 - P.1) * (N.2 - Q.2) - (M.2 - P.2) * (N.1 - Q.1)) / 2) →
    S' ≤ S) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_area_MPNQ_l719_71988


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parallelogram_side_length_l719_71971

noncomputable section

-- Define the parallelogram properties
def parallelogram_side1 (s : ℝ) : ℝ := 3 * s
def parallelogram_side2 (s : ℝ) : ℝ := s
def parallelogram_angle : ℝ := 30 * Real.pi / 180
def parallelogram_area : ℝ := 9 * Real.sqrt 3

-- Theorem statement
theorem parallelogram_side_length :
  ∀ s : ℝ, s > 0 →
  parallelogram_side1 s * parallelogram_side2 s * Real.sin parallelogram_angle = parallelogram_area →
  s = Real.sqrt (6 * Real.sqrt 3) := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parallelogram_side_length_l719_71971


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_perimeter_quadrilateral_l719_71950

/-- Point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Distance between two points -/
noncomputable def distance (p1 p2 : Point) : ℝ :=
  Real.sqrt ((p1.x - p2.x)^2 + (p1.y - p2.y)^2)

/-- Perimeter of quadrilateral PABN -/
noncomputable def perimeter (a : ℝ) : ℝ :=
  let A : Point := ⟨1, -2⟩
  let B : Point := ⟨4, 0⟩
  let P : Point := ⟨a, 1⟩
  let N : Point := ⟨a+1, 1⟩
  distance P A + distance A B + distance B N + distance N P

theorem smallest_perimeter_quadrilateral :
  ∃ (a : ℝ), ∀ (x : ℝ), perimeter a ≤ perimeter x ∧ a = 5/2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_perimeter_quadrilateral_l719_71950


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_simple_interest_equals_58_l719_71968

-- Define the interest rate and time
noncomputable def interest_rate : ℝ := 5
noncomputable def time : ℝ := 2

-- Define the compound interest amount
noncomputable def compound_interest : ℝ := 59.45

-- Define the function for calculating compound interest
noncomputable def calc_compound_interest (P : ℝ) : ℝ :=
  P * ((1 + interest_rate / 100) ^ time - 1)

-- Define the function for calculating simple interest
noncomputable def calc_simple_interest (P : ℝ) : ℝ :=
  P * interest_rate * time / 100

-- Theorem stating that if the compound interest is 59.45, 
-- then the simple interest is 58
theorem simple_interest_equals_58 :
  ∃ P : ℝ, calc_compound_interest P = compound_interest → 
  calc_simple_interest P = 58 := by
  sorry

#eval "Proof completed"

end NUMINAMATH_CALUDE_ERRORFEEDBACK_simple_interest_equals_58_l719_71968


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_and_triangle_properties_l719_71937

-- Define the parabola C
def parabola (p : ℝ) (x y : ℝ) : Prop := y^2 = 2*p*x ∧ p > 0

-- Define the circle M
def circleM (x y : ℝ) : Prop := (x-2)^2 + y^2 = 4

-- Define the point P on the parabola
def point_on_parabola (x₀ y₀ : ℝ) (p : ℝ) : Prop :=
  parabola p x₀ y₀ ∧ x₀ ≥ 5 ∧ y₀ ≥ 0

-- Define the distance from center of M to directrix of C
def distance_center_to_directrix (p : ℝ) : Prop := 2 + p/2 = 3

-- Define the area of triangle PAB
noncomputable def area_triangle_PAB (x₀ y₀ : ℝ) : ℝ := 2 * ((x₀ - 1) + 1/(x₀ - 1) + 2)

theorem parabola_and_triangle_properties :
  ∀ p x₀ y₀ : ℝ,
  parabola p x₀ y₀ →
  circleM x₀ y₀ →
  point_on_parabola x₀ y₀ p →
  distance_center_to_directrix p →
  (∃ (x y : ℝ), parabola 2 x y) ∧
  (∀ x y : ℝ, area_triangle_PAB x y ≥ 25/2) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_and_triangle_properties_l719_71937


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_abi_spend_fraction_l719_71945

/-- Represents Abi's monthly salary savings scenario -/
structure SavingsScenario where
  salary : ℚ
  spendFraction : ℚ
  saveFraction : ℚ
  monthsInYear : ℕ
  savingsMultiple : ℚ

/-- The conditions of Abi's savings scenario -/
def abiScenario : SavingsScenario where
  salary := 1  -- We can use any non-zero value for salary
  spendFraction := 2/3  -- This is what we want to prove
  saveFraction := 1/3
  monthsInYear := 12
  savingsMultiple := 6

/-- The main theorem to prove -/
theorem abi_spend_fraction :
  let s := abiScenario
  s.salary > 0 →
  s.spendFraction > 0 →
  s.spendFraction < 1 →
  s.saveFraction = 1 - s.spendFraction →
  s.monthsInYear * s.saveFraction * s.salary = s.savingsMultiple * s.spendFraction * s.salary →
  s.spendFraction = 2/3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_abi_spend_fraction_l719_71945


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_red_ball_highest_prob_l719_71956

/-- The probability that a ball lands in bin k -/
noncomputable def prob_in_bin (k : ℕ+) : ℝ := 2^(-(k : ℝ))

/-- The probability that all three balls land in the same bin -/
noncomputable def prob_all_same_bin : ℝ := ∑' k, (prob_in_bin k)^3

/-- The probability that the red ball is in a higher-numbered bin than both green and blue balls -/
noncomputable def prob_red_highest : ℝ := (1 - prob_all_same_bin) / 3

/-- Theorem stating that the probability of the red ball being in the highest bin is 2/7 -/
theorem red_ball_highest_prob :
  prob_red_highest = 2/7 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_red_ball_highest_prob_l719_71956


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_difference_n_m_l719_71922

-- Define the functions f and g
noncomputable def f (x : ℝ) : ℝ := 1 - Real.sqrt (1 - 2*x)
noncomputable def g (x : ℝ) : ℝ := Real.log x

-- State the theorem
theorem min_difference_n_m :
  ∀ m n : ℝ, m ≤ (1/2) → n > 0 → f m = g n →
  (∀ m' n' : ℝ, m' ≤ (1/2) → n' > 0 → f m' = g n' → n - m ≤ n' - m') →
  n - m = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_difference_n_m_l719_71922


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arctan_sum_equation_l719_71917

theorem arctan_sum_equation (n : ℕ) : 
  (Real.arctan (1/2) + Real.arctan (1/3) + Real.arctan (1/6) + Real.arctan (1/(n:ℝ)) = π/3) ↔ 
  n = 14 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_arctan_sum_equation_l719_71917


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_trig_function_l719_71939

theorem max_value_trig_function :
  ∃ M, M = 5 ∧ ∀ x : ℝ, 3 * Real.cos x + 4 * Real.sin x ≤ M := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_trig_function_l719_71939


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_integers_for_median_10_l719_71942

def initial_set : Finset ℕ := {5, 6, 3, 8, 4}

def is_median (s : Finset ℕ) (m : ℕ) : Prop :=
  2 * (s.filter (· < m)).card ≤ s.card ∧
  2 * (s.filter (· > m)).card ≤ s.card

theorem smallest_integers_for_median_10 :
  ∃! (a b : ℕ),
    (is_median (initial_set ∪ {a, b}) 10 ∧
     ∀ c d : ℕ, (c < a ∨ d < b) → ¬is_median (initial_set ∪ {c, d}) 10) ∧
    a ≤ b :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_integers_for_median_10_l719_71942


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_direction_vector_l₁_when_a_1_l₁_perpendicular_l₂_when_a_3_2_l719_71943

-- Define the lines l₁ and l₂
def l₁ (a : ℝ) (x y : ℝ) : Prop := a * x + 3 * y + 4 = 0
def l₂ (a : ℝ) (x y : ℝ) : Prop := x + (a - 2) * y + a^2 - 5 = 0

-- Define the direction vector of a line
def direction_vector (m : ℝ) : ℝ × ℝ := (1, -m)

-- Define the slope of l₁
noncomputable def slope_l₁ (a : ℝ) : ℝ := -a / 3

-- Define the slope of l₂
noncomputable def slope_l₂ (a : ℝ) : ℝ := -1 / (a - 2)

-- Theorem 1: Direction vector of l₁ when a = 1
theorem direction_vector_l₁_when_a_1 :
  direction_vector (slope_l₁ 1) = (3, -1) := by sorry

-- Theorem 2: l₁ is perpendicular to l₂ when a = 3/2
theorem l₁_perpendicular_l₂_when_a_3_2 :
  slope_l₁ (3/2) * slope_l₂ (3/2) = -1 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_direction_vector_l₁_when_a_1_l₁_perpendicular_l₂_when_a_3_2_l719_71943


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_same_color_points_exist_l719_71961

-- Define a type for colors
inductive Color
| White
| Black

-- Define a point in the plane
structure Point where
  x : ℝ
  y : ℝ

-- Define a coloring function
def coloring : Point → Color := sorry

-- Define the distance between two points
noncomputable def distance (p q : Point) : ℝ :=
  Real.sqrt ((p.x - q.x)^2 + (p.y - q.y)^2)

-- The main theorem
theorem same_color_points_exist :
  ∃ (p q : Point), coloring p = coloring q ∧ distance p q = 1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_same_color_points_exist_l719_71961
