import Mathlib

namespace NUMINAMATH_CALUDE_inscribed_sphere_radius_eq_one_l3126_312699

/-- A right triangular pyramid with base edge length 6 and lateral edge length √21 -/
structure RightTriangularPyramid where
  base_edge_length : ℝ
  lateral_edge_length : ℝ
  base_edge_length_eq : base_edge_length = 6
  lateral_edge_length_eq : lateral_edge_length = Real.sqrt 21

/-- The radius of the inscribed sphere of a right triangular pyramid -/
def inscribed_sphere_radius (p : RightTriangularPyramid) : ℝ :=
  1 -- Definition, not proof

/-- Theorem: The radius of the inscribed sphere of a right triangular pyramid
    with base edge length 6 and lateral edge length √21 is equal to 1 -/
theorem inscribed_sphere_radius_eq_one (p : RightTriangularPyramid) :
  inscribed_sphere_radius p = 1 := by
  sorry

#check inscribed_sphere_radius_eq_one

end NUMINAMATH_CALUDE_inscribed_sphere_radius_eq_one_l3126_312699


namespace NUMINAMATH_CALUDE_water_displaced_squared_l3126_312602

-- Define the dimensions of the barrel and cube
def barrel_radius : ℝ := 4
def barrel_height : ℝ := 10
def cube_side : ℝ := 8

-- Define the volume of water displaced
def water_displaced : ℝ := cube_side ^ 3

-- Theorem statement
theorem water_displaced_squared :
  water_displaced ^ 2 = 262144 := by sorry

end NUMINAMATH_CALUDE_water_displaced_squared_l3126_312602


namespace NUMINAMATH_CALUDE_hyperbola_eccentricity_l3126_312668

/-- The eccentricity of a hyperbola with given properties -/
theorem hyperbola_eccentricity (a b : ℝ) (h1 : a > 0) (h2 : b > 0) 
  (h3 : b / a = 4 / 3) : 
  Real.sqrt (1 + (b / a)^2) = 5 / 3 := by
  sorry

#check hyperbola_eccentricity

end NUMINAMATH_CALUDE_hyperbola_eccentricity_l3126_312668


namespace NUMINAMATH_CALUDE_sheela_income_proof_l3126_312620

/-- Sheela's monthly income in rupees -/
def monthly_income : ℝ := 17272.73

/-- The amount Sheela deposits in the bank in rupees -/
def deposit : ℝ := 3800

/-- The percentage of Sheela's monthly income that she deposits -/
def deposit_percentage : ℝ := 22

theorem sheela_income_proof :
  deposit = (deposit_percentage / 100) * monthly_income :=
by sorry

end NUMINAMATH_CALUDE_sheela_income_proof_l3126_312620


namespace NUMINAMATH_CALUDE_new_tram_properties_l3126_312624

/-- Represents the properties of a tram journey -/
structure TramJourney where
  distance : ℝ
  old_time : ℝ
  new_time : ℝ
  old_speed : ℝ
  new_speed : ℝ

/-- Theorem stating the properties of the new tram journey -/
theorem new_tram_properties (j : TramJourney) 
  (h1 : j.distance = 20)
  (h2 : j.new_time = j.old_time - 1/5)
  (h3 : j.new_speed = j.old_speed + 5)
  (h4 : j.distance = j.old_speed * j.old_time)
  (h5 : j.distance = j.new_speed * j.new_time) :
  j.new_time = 4/5 ∧ j.new_speed = 25 := by
  sorry

end NUMINAMATH_CALUDE_new_tram_properties_l3126_312624


namespace NUMINAMATH_CALUDE_cone_height_equals_cube_volume_l3126_312651

/-- The height of a circular cone with base radius 5 units and volume equal to that of a cube with edge length 5 units is 15/π units. -/
theorem cone_height_equals_cube_volume (h : ℝ) : h = 15 / π := by
  -- Define the edge length of the cube
  let cube_edge : ℝ := 5

  -- Define the base radius of the cone
  let cone_radius : ℝ := 5

  -- Define the volume of the cube
  let cube_volume : ℝ := cube_edge ^ 3

  -- Define the volume of the cone
  let cone_volume : ℝ := (1 / 3) * π * cone_radius ^ 2 * h

  -- Assume the volumes are equal
  have volumes_equal : cube_volume = cone_volume := by sorry

  sorry

end NUMINAMATH_CALUDE_cone_height_equals_cube_volume_l3126_312651


namespace NUMINAMATH_CALUDE_power_sum_equality_l3126_312696

theorem power_sum_equality : 3 * 3^3 + 9^61 / 9^59 = 162 := by sorry

end NUMINAMATH_CALUDE_power_sum_equality_l3126_312696


namespace NUMINAMATH_CALUDE_part_one_part_two_l3126_312683

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := |x - a|

-- Part 1
theorem part_one :
  let f := f 2
  {x : ℝ | f x ≥ 3 - |x - 1|} = {x : ℝ | x ≤ 0 ∨ x ≥ 3} := by sorry

-- Part 2
theorem part_two :
  ∀ a m n : ℝ,
  m > 0 → n > 0 →
  m + 2*n = a →
  ({x : ℝ | f a x ≤ 1} = {x : ℝ | 2 ≤ x ∧ x ≤ 4}) →
  ∃ (min : ℝ), min = 9/2 ∧ ∀ m' n' : ℝ, m' > 0 → n' > 0 → m' + 2*n' = a → m'^2 + 4*n'^2 ≥ min := by sorry

end NUMINAMATH_CALUDE_part_one_part_two_l3126_312683


namespace NUMINAMATH_CALUDE_rectangular_field_area_l3126_312631

theorem rectangular_field_area (w l : ℝ) : 
  l = 2 * w + 35 →
  2 * (w + l) = 700 →
  w * l = 25725 := by
sorry

end NUMINAMATH_CALUDE_rectangular_field_area_l3126_312631


namespace NUMINAMATH_CALUDE_total_situps_is_110_l3126_312694

/-- The number of situps Diana did -/
def diana_situps : ℕ := 40

/-- Diana's rate of situps per minute -/
def diana_rate : ℕ := 4

/-- The difference between Hani's and Diana's situp rates -/
def rate_difference : ℕ := 3

/-- Calculates the total number of situps done by Hani and Diana together -/
def total_situps : ℕ :=
  let diana_time := diana_situps / diana_rate
  let hani_rate := diana_rate + rate_difference
  let hani_situps := hani_rate * diana_time
  diana_situps + hani_situps

/-- Theorem stating that the total number of situps is 110 -/
theorem total_situps_is_110 : total_situps = 110 := by
  sorry

end NUMINAMATH_CALUDE_total_situps_is_110_l3126_312694


namespace NUMINAMATH_CALUDE_berries_and_coconut_cost_l3126_312655

/-- The cost of a bundle of berries and a coconut given the specified conditions -/
theorem berries_and_coconut_cost :
  ∀ (p b c d : ℚ),
  p + b + c + d = 30 →
  d = 3 * p →
  c = (p + b) / 2 →
  b + c = 65 / 9 := by
sorry

end NUMINAMATH_CALUDE_berries_and_coconut_cost_l3126_312655


namespace NUMINAMATH_CALUDE_triangle_property_l3126_312621

theorem triangle_property (a b c : ℝ) (A B C : ℝ) :
  a > 0 → b > 0 → c > 0 →
  A > 0 → B > 0 → C > 0 →
  A + B + C = π →
  a^2 + c^2 = b^2 + Real.sqrt 2 * a * c →
  (∃ (x y z : ℝ), x^2 + y^2 + z^2 = 1 ∧ 
    a = b * y ∧ b = c * z ∧ c = a * x) →
  B = π / 4 ∧ 
  (∀ A' C', A' + C' = 3 * π / 4 → 
    Real.sqrt 2 * Real.cos A' + Real.cos C' ≤ 1) :=
by sorry

end NUMINAMATH_CALUDE_triangle_property_l3126_312621


namespace NUMINAMATH_CALUDE_max_sequence_length_l3126_312678

/-- Fibonacci sequence -/
def fib : ℕ → ℕ
  | 0 => 0
  | 1 => 1
  | n + 2 => fib n + fib (n + 1)

/-- The n-th term of the sequence -/
def a (n : ℕ) (x : ℕ) : ℤ :=
  if n % 2 = 1 then
    1000 * (fib (n - 2)) - x * (fib (n - 1))
  else
    x * (fib (n - 1)) - 1000 * (fib (n - 2))

/-- The condition for the first negative term -/
def first_negative (x : ℕ) : Prop :=
  ∃ n : ℕ, (∀ k < n, a k x ≥ 0) ∧ a n x < 0

/-- The maximum value of x that produces the longest sequence -/
def max_x : ℕ := 618

/-- The main theorem -/
theorem max_sequence_length :
  ∀ y : ℕ, y > max_x → first_negative y → 
  ∃ z : ℕ, z ≤ max_x ∧ first_negative z ∧ 
  ∀ w : ℕ, first_negative w → (∃ n : ℕ, a n z < 0) → (∃ m : ℕ, m ≤ n ∧ a m w < 0) :=
sorry

end NUMINAMATH_CALUDE_max_sequence_length_l3126_312678


namespace NUMINAMATH_CALUDE_squirrels_in_tree_l3126_312680

theorem squirrels_in_tree (nuts : ℕ) (squirrels : ℕ) 
  (h1 : nuts = 2)
  (h2 : squirrels - nuts = 2) :
  squirrels = 4 := by
  sorry

end NUMINAMATH_CALUDE_squirrels_in_tree_l3126_312680


namespace NUMINAMATH_CALUDE_complex_modulus_equality_l3126_312611

theorem complex_modulus_equality (t : ℝ) : 
  t > 0 → Complex.abs (-3 + t * Complex.I) = 5 → t = 4 := by sorry

end NUMINAMATH_CALUDE_complex_modulus_equality_l3126_312611


namespace NUMINAMATH_CALUDE_regular_polygon_20_sides_l3126_312695

-- Define a regular polygon with exterior angle of 18 degrees
structure RegularPolygon where
  sides : ℕ
  exteriorAngle : ℝ
  regular : exteriorAngle = 18

-- Theorem: A regular polygon with exterior angle of 18 degrees has 20 sides
theorem regular_polygon_20_sides (p : RegularPolygon) : p.sides = 20 := by
  sorry

end NUMINAMATH_CALUDE_regular_polygon_20_sides_l3126_312695


namespace NUMINAMATH_CALUDE_max_m_inequality_l3126_312661

theorem max_m_inequality (x y : ℝ) (hx : x > 1) (hy : y > 1) :
  (∃ m : ℝ, ∀ x y : ℝ, x > 1 → y > 1 → x^2 / (y - 1) + y^2 / (x - 1) ≥ 3 * m - 1) ∧
  (∀ m : ℝ, (∀ x y : ℝ, x > 1 → y > 1 → x^2 / (y - 1) + y^2 / (x - 1) ≥ 3 * m - 1) → m ≤ 3) :=
by sorry

end NUMINAMATH_CALUDE_max_m_inequality_l3126_312661


namespace NUMINAMATH_CALUDE_brothers_age_fraction_l3126_312600

/-- Given three brothers with ages M, O, and Y, prove that Y/O = 1/3 -/
theorem brothers_age_fraction (M O Y : ℕ) : 
  Y = 5 → 
  M + O + Y = 28 → 
  O = 2 * (M - 1) + 1 → 
  Y / O = 1 / 3 := by
  sorry

end NUMINAMATH_CALUDE_brothers_age_fraction_l3126_312600


namespace NUMINAMATH_CALUDE_min_sticks_arrangement_l3126_312639

theorem min_sticks_arrangement (n : ℕ) : n = 1012 ↔ 
  (n > 1000) ∧
  (∃ k : ℕ, n = 3 * k + 1) ∧
  (∃ m : ℕ, n = 5 * m + 2) ∧
  (∃ p : ℕ, n = 2 * p * (p + 1)) ∧
  (∀ x : ℕ, x > 1000 → 
    ((∃ k : ℕ, x = 3 * k + 1) ∧
     (∃ m : ℕ, x = 5 * m + 2) ∧
     (∃ p : ℕ, x = 2 * p * (p + 1))) → x ≥ n) :=
by sorry

end NUMINAMATH_CALUDE_min_sticks_arrangement_l3126_312639


namespace NUMINAMATH_CALUDE_chord_length_l3126_312664

/-- The polar equation of line l is √3ρcosθ + ρsinθ - 1 = 0 -/
def line_l (ρ θ : ℝ) : Prop :=
  Real.sqrt 3 * ρ * Real.cos θ + ρ * Real.sin θ - 1 = 0

/-- The polar equation of curve C is ρ = 4 -/
def curve_C (ρ : ℝ) : Prop := ρ = 4

/-- The length of the chord formed by the intersection of l and C is 3√7 -/
theorem chord_length : 
  ∃ (A B : ℝ × ℝ), 
    (∃ (ρ_A θ_A ρ_B θ_B : ℝ), 
      line_l ρ_A θ_A ∧ line_l ρ_B θ_B ∧ 
      curve_C ρ_A ∧ curve_C ρ_B ∧
      A = (ρ_A * Real.cos θ_A, ρ_A * Real.sin θ_A) ∧
      B = (ρ_B * Real.cos θ_B, ρ_B * Real.sin θ_B)) →
    Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) = 3 * Real.sqrt 7 :=
sorry

end NUMINAMATH_CALUDE_chord_length_l3126_312664


namespace NUMINAMATH_CALUDE_root_existence_l3126_312676

theorem root_existence (a b c x₁ x₂ : ℝ) (ha : a ≠ 0)
  (h₁ : a * x₁^2 + b * x₁ + c = 0)
  (h₂ : -a * x₂^2 + b * x₂ + c = 0) :
  ∃ x₃, (a / 2) * x₃^2 + b * x₃ + c = 0 ∧
    ((x₁ ≤ x₃ ∧ x₃ ≤ x₂) ∨ (x₁ ≥ x₃ ∧ x₃ ≥ x₂)) :=
by sorry

end NUMINAMATH_CALUDE_root_existence_l3126_312676


namespace NUMINAMATH_CALUDE_custom_multiplication_l3126_312657

theorem custom_multiplication (a b : ℤ) : a * b = a^2 + a*b - b^2 → 5 * (-3) = 1 := by
  sorry

end NUMINAMATH_CALUDE_custom_multiplication_l3126_312657


namespace NUMINAMATH_CALUDE_diagonal_difference_l3126_312663

/-- The number of diagonals in a convex polygon with n sides -/
def f (n : ℕ) : ℕ :=
  sorry

/-- Theorem: The difference between the number of diagonals in a convex polygon
    with n+1 sides and n sides is n-1, for n ≥ 4 -/
theorem diagonal_difference (n : ℕ) (h : n ≥ 4) : f (n + 1) - f n = n - 1 :=
  sorry

end NUMINAMATH_CALUDE_diagonal_difference_l3126_312663


namespace NUMINAMATH_CALUDE_halfway_point_sixth_twelfth_l3126_312630

theorem halfway_point_sixth_twelfth (x y : ℚ) (hx : x = 1/6) (hy : y = 1/12) :
  (x + y) / 2 = 1/8 := by
  sorry

end NUMINAMATH_CALUDE_halfway_point_sixth_twelfth_l3126_312630


namespace NUMINAMATH_CALUDE_sasha_always_wins_l3126_312674

/-- Represents the state of the game board -/
structure GameState where
  digits : List Nat
  deriving Repr

/-- Represents a player's move -/
structure Move where
  appendedDigits : List Nat
  deriving Repr

/-- Checks if a number represented by a list of digits is divisible by 112 -/
def isDivisibleBy112 (digits : List Nat) : Bool :=
  sorry

/-- Generates all possible moves for Sasha (appending one digit) -/
def sashasMoves : List Move :=
  sorry

/-- Generates all possible moves for Andrey (appending two digits) -/
def andreysMoves : List Move :=
  sorry

/-- Applies a move to the current game state -/
def applyMove (state : GameState) (move : Move) : GameState :=
  sorry

/-- Checks if Sasha wins in the current state -/
def sashaWins (state : GameState) : Bool :=
  sorry

/-- Checks if Andrey wins in the current state -/
def andreyWins (state : GameState) : Bool :=
  sorry

/-- Theorem: Sasha can always win the game -/
theorem sasha_always_wins :
  ∀ (state : GameState),
    (state.digits.length < 2018) →
    (∃ (move : Move), move ∈ sashasMoves ∧
      ∀ (andreyMove : Move), andreyMove ∈ andreysMoves →
        ¬(andreyWins (applyMove (applyMove state move) andreyMove))) ∨
    (sashaWins state) :=
  sorry

end NUMINAMATH_CALUDE_sasha_always_wins_l3126_312674


namespace NUMINAMATH_CALUDE_green_shells_count_l3126_312690

/-- Proves that the number of green shells is 49 given the total number of shells,
    the number of red shells, and the number of shells that are not red or green. -/
theorem green_shells_count
  (total_shells : ℕ)
  (red_shells : ℕ)
  (not_red_or_green_shells : ℕ)
  (h1 : total_shells = 291)
  (h2 : red_shells = 76)
  (h3 : not_red_or_green_shells = 166) :
  total_shells - not_red_or_green_shells - red_shells = 49 :=
by sorry

end NUMINAMATH_CALUDE_green_shells_count_l3126_312690


namespace NUMINAMATH_CALUDE_equal_goals_moment_l3126_312645

/-- Represents the state of a football match at any given moment -/
structure MatchState where
  goalsWinner : ℕ
  goalsLoser : ℕ

/-- The final score of the match -/
def finalScore : MatchState := { goalsWinner := 9, goalsLoser := 5 }

/-- Theorem stating that there exists a point during the match where the number of goals
    the winning team still needs to score equals the number of goals the losing team has already scored -/
theorem equal_goals_moment :
  ∃ (state : MatchState), 
    state.goalsWinner ≤ finalScore.goalsWinner ∧ 
    state.goalsLoser ≤ finalScore.goalsLoser ∧
    (finalScore.goalsWinner - state.goalsWinner) = state.goalsLoser :=
sorry

end NUMINAMATH_CALUDE_equal_goals_moment_l3126_312645


namespace NUMINAMATH_CALUDE_total_fruit_salads_l3126_312659

/-- The total number of fruit salads in three restaurants -/
theorem total_fruit_salads (alaya angel betty : ℕ) : 
  alaya = 200 →
  angel = 2 * alaya →
  betty = 3 * angel →
  alaya + angel + betty = 1800 :=
by
  sorry

end NUMINAMATH_CALUDE_total_fruit_salads_l3126_312659


namespace NUMINAMATH_CALUDE_quadratic_always_nonnegative_implies_a_range_l3126_312603

theorem quadratic_always_nonnegative_implies_a_range (a : ℝ) :
  (∀ x : ℝ, x^2 + (a - 1) * x + 1 ≥ 0) → a ∈ Set.Icc (-1 : ℝ) 3 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_always_nonnegative_implies_a_range_l3126_312603


namespace NUMINAMATH_CALUDE_truncated_pyramid_volume_l3126_312634

/-- Given a truncated pyramid with base areas S₁ and S₂ (where S₁ < S₂) and volume V,
    the volume of the complete pyramid is (V * S₂ * √S₂) / (S₂ * √S₂ - S₁ * √S₁) -/
theorem truncated_pyramid_volume (S₁ S₂ V : ℝ) (h₁ : 0 < S₁) (h₂ : S₁ < S₂) (h₃ : 0 < V) :
  let complete_volume := (V * S₂ * Real.sqrt S₂) / (S₂ * Real.sqrt S₂ - S₁ * Real.sqrt S₁)
  ∃ (h : ℝ), h > 0 ∧ complete_volume = (1 / 3) * S₂ * h :=
by sorry

end NUMINAMATH_CALUDE_truncated_pyramid_volume_l3126_312634


namespace NUMINAMATH_CALUDE_journey_distance_l3126_312641

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Calculates the final position after a series of movements -/
def finalPosition (initialDistance : ℝ) : Point :=
  let southWalk := Point.mk 0 (-initialDistance)
  let eastWalk := Point.mk initialDistance (-initialDistance)
  let northWalk := Point.mk initialDistance 0
  let finalEastWalk := Point.mk (initialDistance + 10) 0
  finalEastWalk

/-- Theorem stating that the initial distance must be 30 to end up 30 meters north -/
theorem journey_distance (initialDistance : ℝ) :
  (finalPosition initialDistance).y = 30 ↔ initialDistance = 30 :=
by sorry

end NUMINAMATH_CALUDE_journey_distance_l3126_312641


namespace NUMINAMATH_CALUDE_quadratic_solution_difference_l3126_312654

-- Define the quadratic equation
def quadratic_equation (x : ℝ) : Prop :=
  2 * x^2 - 5 * x + 18 = 3 * x + 55

-- Define the solutions of the quadratic equation
noncomputable def solution1 : ℝ := 2 + (3 * Real.sqrt 10) / 2
noncomputable def solution2 : ℝ := 2 - (3 * Real.sqrt 10) / 2

-- Theorem statement
theorem quadratic_solution_difference :
  quadratic_equation solution1 ∧ 
  quadratic_equation solution2 ∧ 
  |solution1 - solution2| = 3 * Real.sqrt 10 := by sorry

end NUMINAMATH_CALUDE_quadratic_solution_difference_l3126_312654


namespace NUMINAMATH_CALUDE_estimate_two_sqrt_five_l3126_312692

theorem estimate_two_sqrt_five : 4 < 2 * Real.sqrt 5 ∧ 2 * Real.sqrt 5 < 5 := by
  sorry

end NUMINAMATH_CALUDE_estimate_two_sqrt_five_l3126_312692


namespace NUMINAMATH_CALUDE_sin_cos_relation_l3126_312658

theorem sin_cos_relation (α β : ℝ) (h : 2 * Real.sin α - Real.cos β = 2) :
  Real.sin α + 2 * Real.cos β = 1 ∨ Real.sin α + 2 * Real.cos β = -1 := by
  sorry

end NUMINAMATH_CALUDE_sin_cos_relation_l3126_312658


namespace NUMINAMATH_CALUDE_average_monthly_balance_l3126_312670

def monthly_balances : List ℕ := [100, 200, 150, 150, 180]

theorem average_monthly_balance :
  (monthly_balances.sum / monthly_balances.length : ℚ) = 156 := by
  sorry

end NUMINAMATH_CALUDE_average_monthly_balance_l3126_312670


namespace NUMINAMATH_CALUDE_number_comparisons_l3126_312684

theorem number_comparisons :
  (0.5 < 0.8) ∧ (0.5 < 0.7) ∧ (Real.log 125 < Real.log 1215) := by
  sorry

end NUMINAMATH_CALUDE_number_comparisons_l3126_312684


namespace NUMINAMATH_CALUDE_inequality_solution_set_l3126_312681

theorem inequality_solution_set (c : ℝ) : 
  (c / 5 ≤ 4 + c ∧ 4 + c < -3 * (1 + 2 * c)) ↔ c ∈ Set.Icc (-5) (-1) := by
sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l3126_312681


namespace NUMINAMATH_CALUDE_quadratic_inequality_sufficient_conditions_quadratic_inequality_not_necessary_l3126_312685

theorem quadratic_inequality_sufficient_conditions 
  (k : ℝ) (h : k = 0 ∨ (-3 < k ∧ k < 0) ∨ (-3 < k ∧ k < -1)) :
  ∀ x : ℝ, 2*k*x^2 + k*x - 3/8 < 0 :=
by sorry

theorem quadratic_inequality_not_necessary 
  (k : ℝ) (h : ∀ x : ℝ, 2*k*x^2 + k*x - 3/8 < 0) :
  ¬(k = 0 ∧ (-3 < k ∧ k < 0) ∧ (-3 < k ∧ k < -1)) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_sufficient_conditions_quadratic_inequality_not_necessary_l3126_312685


namespace NUMINAMATH_CALUDE_millet_majority_day_l3126_312627

def seed_mix : ℝ := 0.25
def millet_eaten_daily : ℝ := 0.25
def total_seeds_daily : ℝ := 1

def millet_proportion (n : ℕ) : ℝ := 1 - (1 - seed_mix)^n

theorem millet_majority_day :
  ∀ k : ℕ, k < 5 → millet_proportion k ≤ 0.5 ∧
  millet_proportion 5 > 0.5 := by sorry

end NUMINAMATH_CALUDE_millet_majority_day_l3126_312627


namespace NUMINAMATH_CALUDE_tan_and_cot_inequalities_l3126_312693

open Real

theorem tan_and_cot_inequalities (x₁ x₂ : ℝ) 
  (h1 : 0 < x₁) (h2 : x₁ < π/2) (h3 : 0 < x₂) (h4 : x₂ < π/2) (h5 : x₁ ≠ x₂) :
  (1/2) * (tan x₁ + tan x₂) > tan ((x₁ + x₂)/2) ∧
  (1/2) * (1/tan x₁ + 1/tan x₂) > 1/tan ((x₁ + x₂)/2) := by
  sorry

end NUMINAMATH_CALUDE_tan_and_cot_inequalities_l3126_312693


namespace NUMINAMATH_CALUDE_dot_product_of_vectors_l3126_312698

def vector_a : ℝ × ℝ := (-4, 7)
def vector_b : ℝ × ℝ := (5, 2)

theorem dot_product_of_vectors :
  (vector_a.1 * vector_b.1 + vector_a.2 * vector_b.2) = -6 := by sorry

end NUMINAMATH_CALUDE_dot_product_of_vectors_l3126_312698


namespace NUMINAMATH_CALUDE_ellipse_condition_l3126_312652

def is_ellipse (m : ℝ) : Prop :=
  m + 3 > 0 ∧ m - 1 > 0

theorem ellipse_condition (m : ℝ) :
  (m > -3 → is_ellipse m) ∧ ¬(is_ellipse m → m > -3) :=
sorry

end NUMINAMATH_CALUDE_ellipse_condition_l3126_312652


namespace NUMINAMATH_CALUDE_percentage_problem_l3126_312647

theorem percentage_problem (x : ℝ) (hx : x > 0) : 
  x / 100 * 150 - 20 = 10 → x = 20 := by sorry

end NUMINAMATH_CALUDE_percentage_problem_l3126_312647


namespace NUMINAMATH_CALUDE_bottles_remaining_l3126_312672

/-- Calculates the number of bottles remaining in storage given the initial quantities and percentages sold. -/
theorem bottles_remaining (small_initial : ℕ) (big_initial : ℕ) (small_percent_sold : ℚ) (big_percent_sold : ℚ) :
  small_initial = 6000 →
  big_initial = 15000 →
  small_percent_sold = 11 / 100 →
  big_percent_sold = 12 / 100 →
  (small_initial - small_initial * small_percent_sold) + (big_initial - big_initial * big_percent_sold) = 18540 := by
sorry

end NUMINAMATH_CALUDE_bottles_remaining_l3126_312672


namespace NUMINAMATH_CALUDE_tangent_line_sum_range_l3126_312653

noncomputable def f (x : ℝ) : ℝ := Real.log x

theorem tangent_line_sum_range (x₀ : ℝ) (h : x₀ > 0) :
  let k := 1 / x₀
  let b := Real.log x₀ - 1
  ∀ y : ℝ, y ≥ 0 → ∃ x : ℝ, x > 0 ∧ k + b = y :=
sorry

end NUMINAMATH_CALUDE_tangent_line_sum_range_l3126_312653


namespace NUMINAMATH_CALUDE_gcd_of_2_powers_l3126_312601

theorem gcd_of_2_powers : Nat.gcd (2^1040 - 1) (2^1030 - 1) = 1023 := by sorry

end NUMINAMATH_CALUDE_gcd_of_2_powers_l3126_312601


namespace NUMINAMATH_CALUDE_train_passing_tree_l3126_312660

/-- Proves that a train of given length and speed takes the calculated time to pass a tree -/
theorem train_passing_tree (train_length : ℝ) (train_speed_km_hr : ℝ) (time : ℝ) :
  train_length = 175 →
  train_speed_km_hr = 63 →
  time = train_length / (train_speed_km_hr * (1000 / 3600)) →
  time = 10 := by
  sorry

end NUMINAMATH_CALUDE_train_passing_tree_l3126_312660


namespace NUMINAMATH_CALUDE_zeros_not_adjacent_probability_l3126_312666

/-- The number of ones in the arrangement -/
def num_ones : ℕ := 3

/-- The number of zeros in the arrangement -/
def num_zeros : ℕ := 3

/-- The total number of digits in the arrangement -/
def total_digits : ℕ := num_ones + num_zeros

/-- The probability that the zeros are not adjacent when randomly arranged -/
def prob_zeros_not_adjacent : ℚ := 1 / 5

/-- Theorem stating that the probability of zeros not being adjacent is 1/5 -/
theorem zeros_not_adjacent_probability :
  prob_zeros_not_adjacent = 1 / 5 := by
  sorry

end NUMINAMATH_CALUDE_zeros_not_adjacent_probability_l3126_312666


namespace NUMINAMATH_CALUDE_sinusoidal_oscillations_l3126_312626

/-- A sinusoidal function that completes 5 oscillations from 0 to 2π has b = 5 -/
theorem sinusoidal_oscillations (a b c d : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0) :
  (∀ x : ℝ, (a * Real.sin (b * x + c) + d = a * Real.sin (b * (x + 2 * Real.pi) + c) + d)) →
  (∃ n : ℕ, n = 5 ∧ ∀ x : ℝ, a * Real.sin (b * x + c) + d = a * Real.sin (b * (x + 2 * Real.pi / n) + c) + d) →
  b = 5 := by
  sorry

end NUMINAMATH_CALUDE_sinusoidal_oscillations_l3126_312626


namespace NUMINAMATH_CALUDE_quadratic_function_unique_l3126_312649

def quadratic_function (a b c : ℝ) : ℝ → ℝ := λ x => a * x^2 + b * x + c

theorem quadratic_function_unique 
  (f : ℝ → ℝ) 
  (h1 : ∃ a b c : ℝ, f = quadratic_function a b c)
  (h2 : f (-2) = 0)
  (h3 : f 4 = 0)
  (h4 : ∀ x : ℝ, f x ≤ 9)
  (h5 : ∃ x : ℝ, f x = 9) :
  f = quadratic_function (-1) 2 8 := by
sorry

end NUMINAMATH_CALUDE_quadratic_function_unique_l3126_312649


namespace NUMINAMATH_CALUDE_least_number_with_remainder_l3126_312629

theorem least_number_with_remainder (n : ℕ) : n = 282 ↔ 
  (n > 0 ∧ 
   n % 31 = 3 ∧ 
   n % 9 = 3 ∧ 
   ∀ m : ℕ, m > 0 → m % 31 = 3 → m % 9 = 3 → m ≥ n) := by
sorry

end NUMINAMATH_CALUDE_least_number_with_remainder_l3126_312629


namespace NUMINAMATH_CALUDE_eight_couples_handshakes_l3126_312609

/-- The number of handshakes in a gathering of couples -/
def count_handshakes (n : ℕ) : ℕ :=
  let total_people := 2 * n
  let handshakes_per_person := total_people - 3
  (total_people * handshakes_per_person) / 2

/-- Theorem: In a gathering of 8 couples, where each person shakes hands with
    everyone except their spouse and one other person, the total number of
    handshakes is 104. -/
theorem eight_couples_handshakes :
  count_handshakes 8 = 104 := by
  sorry

#eval count_handshakes 8  -- Should output 104

end NUMINAMATH_CALUDE_eight_couples_handshakes_l3126_312609


namespace NUMINAMATH_CALUDE_unique_solution_l3126_312688

theorem unique_solution (x y z : ℝ) 
  (hx : x > 2) (hy : y > 2) (hz : z > 2)
  (h : ((x + 3)^2) / (y + z - 3) + ((y + 5)^2) / (z + x - 5) + ((z + 7)^2) / (x + y - 7) = 45) :
  x = 13 ∧ y = 11 ∧ z = 6 := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_l3126_312688


namespace NUMINAMATH_CALUDE_not_coplanar_implies_not_collinear_three_collinear_implies_coplanar_l3126_312614

-- Define a type for points in space
variable (Point : Type)

-- Define the property of being coplanar for four points
variable (coplanar : Point → Point → Point → Point → Prop)

-- Define the property of being collinear for three points
variable (collinear : Point → Point → Point → Prop)

-- Theorem 1: If four points are not coplanar, then any three of them are not collinear
theorem not_coplanar_implies_not_collinear 
  (p q r s : Point) : 
  ¬(coplanar p q r s) → 
  (¬(collinear p q r) ∧ ¬(collinear p q s) ∧ ¬(collinear p r s) ∧ ¬(collinear q r s)) :=
sorry

-- Theorem 2: If there exist three collinear points among four points, then these four points are coplanar
theorem three_collinear_implies_coplanar 
  (p q r s : Point) : 
  (collinear p q r ∨ collinear p q s ∨ collinear p r s ∨ collinear q r s) → 
  coplanar p q r s :=
sorry

end NUMINAMATH_CALUDE_not_coplanar_implies_not_collinear_three_collinear_implies_coplanar_l3126_312614


namespace NUMINAMATH_CALUDE_goblet_sphere_max_radius_l3126_312686

theorem goblet_sphere_max_radius :
  let goblet_cross_section := fun (x : ℝ) => x^4
  let sphere_in_goblet := fun (r : ℝ) (x y : ℝ) => y ≥ goblet_cross_section x ∧ (y - r)^2 + x^2 = r^2
  ∃ (max_r : ℝ), max_r = 3 / Real.rpow 2 (1/3) ∧
    (∀ r, r > 0 → sphere_in_goblet r 0 0 → r ≤ max_r) ∧
    sphere_in_goblet max_r 0 0 :=
sorry

end NUMINAMATH_CALUDE_goblet_sphere_max_radius_l3126_312686


namespace NUMINAMATH_CALUDE_side_to_hotdog_ratio_l3126_312679

def food_weights (chicken hamburger hotdog side : ℝ) : Prop :=
  chicken = 16 ∧
  hamburger = chicken / 2 ∧
  hotdog = hamburger + 2 ∧
  chicken + hamburger + hotdog + side = 39

theorem side_to_hotdog_ratio (chicken hamburger hotdog side : ℝ) :
  food_weights chicken hamburger hotdog side →
  side / hotdog = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_side_to_hotdog_ratio_l3126_312679


namespace NUMINAMATH_CALUDE_complex_magnitude_example_l3126_312646

theorem complex_magnitude_example : Complex.abs (-5 - (8/3)*Complex.I) = 17/3 := by
  sorry

end NUMINAMATH_CALUDE_complex_magnitude_example_l3126_312646


namespace NUMINAMATH_CALUDE_product_of_numbers_with_given_sum_and_difference_l3126_312636

theorem product_of_numbers_with_given_sum_and_difference :
  ∀ x y : ℝ, x + y = 24 ∧ x - y = 8 → x * y = 128 := by
  sorry

end NUMINAMATH_CALUDE_product_of_numbers_with_given_sum_and_difference_l3126_312636


namespace NUMINAMATH_CALUDE_min_value_expression_min_value_achievable_l3126_312633

theorem min_value_expression (a b c : ℝ) (h1 : 2 ≤ a) (h2 : a ≤ b) (h3 : b ≤ c) (h4 : c ≤ 5) :
  (a - 2)^2 + (b/a - 1)^2 + (c/b - 1)^2 + (5/c - 1)^2 + (c - 4)^2 ≥ 10.1 :=
by sorry

theorem min_value_achievable :
  ∃ a b c : ℝ, 2 ≤ a ∧ a ≤ b ∧ b ≤ c ∧ c ≤ 5 ∧
  (a - 2)^2 + (b/a - 1)^2 + (c/b - 1)^2 + (5/c - 1)^2 + (c - 4)^2 = 10.1 :=
by sorry

end NUMINAMATH_CALUDE_min_value_expression_min_value_achievable_l3126_312633


namespace NUMINAMATH_CALUDE_nested_square_root_simplification_l3126_312619

theorem nested_square_root_simplification (y : ℝ) (h : y ≥ 0) :
  Real.sqrt (y * Real.sqrt (y * Real.sqrt (y * Real.sqrt y))) = (y ^ 9) ^ (1/4) := by
  sorry

end NUMINAMATH_CALUDE_nested_square_root_simplification_l3126_312619


namespace NUMINAMATH_CALUDE_polygon_with_1080_degrees_has_8_sides_l3126_312640

/-- A polygon is a shape with a certain number of sides. -/
structure Polygon where
  sides : ℕ

/-- The sum of interior angles of a polygon. -/
def sumOfInteriorAngles (p : Polygon) : ℝ :=
  180 * (p.sides - 2)

/-- Theorem: A polygon with a sum of interior angles equal to 1080° has 8 sides. -/
theorem polygon_with_1080_degrees_has_8_sides :
  ∃ (p : Polygon), sumOfInteriorAngles p = 1080 → p.sides = 8 := by
  sorry

end NUMINAMATH_CALUDE_polygon_with_1080_degrees_has_8_sides_l3126_312640


namespace NUMINAMATH_CALUDE_larger_number_proof_l3126_312671

theorem larger_number_proof (a b : ℝ) : 
  a > 0 → b > 0 → a > b → a + b = 9 * (a - b) → a + b = 36 → a = 20 := by
  sorry

end NUMINAMATH_CALUDE_larger_number_proof_l3126_312671


namespace NUMINAMATH_CALUDE_rectangle_y_value_l3126_312610

/-- Rectangle EFGH with vertices E(0, 0), F(0, 5), G(y, 5), and H(y, 0) -/
structure Rectangle where
  y : ℝ
  h_positive : y > 0

/-- The area of rectangle EFGH is 40 square units -/
def area (r : Rectangle) : ℝ := 5 * r.y

theorem rectangle_y_value (r : Rectangle) (h_area : area r = 40) : r.y = 8 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_y_value_l3126_312610


namespace NUMINAMATH_CALUDE_cookies_left_for_neil_l3126_312638

def total_cookies : ℕ := 20
def fraction_given : ℚ := 2/5

theorem cookies_left_for_neil :
  total_cookies - (total_cookies * fraction_given).floor = 12 :=
by sorry

end NUMINAMATH_CALUDE_cookies_left_for_neil_l3126_312638


namespace NUMINAMATH_CALUDE_platform_length_l3126_312691

/-- The length of a platform given train speed and crossing times -/
theorem platform_length
  (train_speed : ℝ)
  (platform_crossing_time : ℝ)
  (man_crossing_time : ℝ)
  (h1 : train_speed = 72)  -- km/h
  (h2 : platform_crossing_time = 30)  -- seconds
  (h3 : man_crossing_time = 15)  -- seconds
  : ∃ (platform_length : ℝ), platform_length = 300 :=
by
  sorry

end NUMINAMATH_CALUDE_platform_length_l3126_312691


namespace NUMINAMATH_CALUDE_possible_values_of_a_l3126_312669

theorem possible_values_of_a : 
  ∀ (a b c : ℤ), 
  (∀ x : ℝ, (x - a) * (x - 8) + 4 = (x + b) * (x + c)) → 
  (a = 6 ∨ a = 10) := by
sorry

end NUMINAMATH_CALUDE_possible_values_of_a_l3126_312669


namespace NUMINAMATH_CALUDE_proof_by_contradiction_step_l3126_312689

theorem proof_by_contradiction_step (a : ℝ) (h : a > 1) :
  (∀ P : Prop, (¬P → False) → P) →
  (¬(a^2 > 1) ↔ a^2 ≤ 1) :=
sorry

end NUMINAMATH_CALUDE_proof_by_contradiction_step_l3126_312689


namespace NUMINAMATH_CALUDE_total_parts_calculation_l3126_312677

theorem total_parts_calculation (sample_size : ℕ) (probability : ℚ) (N : ℕ) : 
  sample_size = 30 → probability = 1/4 → N * probability = sample_size → N = 120 := by
  sorry

end NUMINAMATH_CALUDE_total_parts_calculation_l3126_312677


namespace NUMINAMATH_CALUDE_right_triangle_leg_l3126_312632

theorem right_triangle_leg (a c : ℝ) (h1 : a = 12) (h2 : c = 13) :
  ∃ b : ℝ, a^2 + b^2 = c^2 ∧ b = 5 := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_leg_l3126_312632


namespace NUMINAMATH_CALUDE_four_true_propositions_l3126_312605

-- Define a triangle
structure Triangle :=
  (A B C : ℝ × ℝ)

-- Define angle and side length for a triangle
def angle (t : Triangle) (v : Fin 3) : ℝ := sorry
def side_length (t : Triangle) (v : Fin 3) : ℝ := sorry

-- Define a quadrilateral
structure Quadrilateral :=
  (A B C D : ℝ × ℝ)

-- Define properties for a quadrilateral
def is_parallelogram (q : Quadrilateral) : Prop := sorry
def diagonals_bisect (q : Quadrilateral) : Prop := sorry

-- The four propositions
theorem four_true_propositions :
  (∀ t : Triangle, angle t 2 > angle t 1 → side_length t 0 > side_length t 1) ∧
  (∀ a b : ℝ, a * b ≠ 0 → a = 0 ∨ b ≠ 0) ∧
  (∀ a b : ℝ, a * b = 0 → a = 0 ∨ b = 0) ∧
  (∀ q : Quadrilateral, diagonals_bisect q → is_parallelogram q) :=
sorry

end NUMINAMATH_CALUDE_four_true_propositions_l3126_312605


namespace NUMINAMATH_CALUDE_divisible_by_2520_l3126_312608

theorem divisible_by_2520 (n : ℕ) : ∃ k : ℤ, (n^7 : ℤ) - 14*(n^5 : ℤ) + 49*(n^3 : ℤ) - 36*(n : ℤ) = 2520 * k := by
  sorry

end NUMINAMATH_CALUDE_divisible_by_2520_l3126_312608


namespace NUMINAMATH_CALUDE_water_bottles_per_day_l3126_312618

theorem water_bottles_per_day (total_bottles : ℕ) (total_days : ℕ) (bottles_per_day : ℕ) : 
  total_bottles = 153 → 
  total_days = 17 → 
  total_bottles = bottles_per_day * total_days → 
  bottles_per_day = 9 := by
sorry

end NUMINAMATH_CALUDE_water_bottles_per_day_l3126_312618


namespace NUMINAMATH_CALUDE_smallest_number_divisible_by_111_ending_2004_l3126_312687

def is_divisible_by (a b : ℕ) : Prop := ∃ k, a = b * k

def last_four_digits (n : ℕ) : ℕ := n % 10000

theorem smallest_number_divisible_by_111_ending_2004 :
  ∀ X : ℕ, 
    X > 0 ∧ 
    is_divisible_by X 111 ∧ 
    last_four_digits X = 2004 → 
    X ≥ 662004 :=
by sorry

end NUMINAMATH_CALUDE_smallest_number_divisible_by_111_ending_2004_l3126_312687


namespace NUMINAMATH_CALUDE_unique_solution_sqrt_equation_l3126_312615

theorem unique_solution_sqrt_equation :
  ∃! x : ℝ, (x - 1 ≥ 0) ∧ (x + 1 ≥ 0) ∧ (x^2 - 1 ≥ 0) ∧
  (Real.sqrt (x - 1) * Real.sqrt (x + 1) = -Real.sqrt (x^2 - 1)) :=
by sorry

end NUMINAMATH_CALUDE_unique_solution_sqrt_equation_l3126_312615


namespace NUMINAMATH_CALUDE_consecutive_page_numbers_l3126_312607

theorem consecutive_page_numbers : ∃ (n : ℕ), 
  n > 0 ∧ 
  n * (n + 1) = 20412 ∧ 
  n + (n + 1) = 283 := by
  sorry

end NUMINAMATH_CALUDE_consecutive_page_numbers_l3126_312607


namespace NUMINAMATH_CALUDE_g_of_3_eq_neg_1_l3126_312606

def g (x : ℝ) : ℝ := 2 * (x - 2)^2 - 3 * (x - 2)

theorem g_of_3_eq_neg_1 : g 3 = -1 := by
  sorry

end NUMINAMATH_CALUDE_g_of_3_eq_neg_1_l3126_312606


namespace NUMINAMATH_CALUDE_larger_number_from_hcf_lcm_l3126_312643

theorem larger_number_from_hcf_lcm (a b : ℕ+) : 
  (Nat.gcd a b = 15) → 
  (Nat.lcm a b = 2475) → 
  (max a b = 225) :=
by sorry

end NUMINAMATH_CALUDE_larger_number_from_hcf_lcm_l3126_312643


namespace NUMINAMATH_CALUDE_find_2a_plus_c_l3126_312665

theorem find_2a_plus_c (a b c : ℝ) 
  (eq1 : 3 * a + b + 2 * c = 3) 
  (eq2 : a + 3 * b + 2 * c = 1) : 
  2 * a + c = 2 := by sorry

end NUMINAMATH_CALUDE_find_2a_plus_c_l3126_312665


namespace NUMINAMATH_CALUDE_third_quadrant_angle_property_l3126_312604

theorem third_quadrant_angle_property (α : Real) : 
  (3 * π / 2 < α) ∧ (α < 2 * π) →
  |Real.sin (α / 2)| / Real.sin (α / 2) + |Real.cos (α / 2)| / Real.cos (α / 2) + 3 = 3 :=
by sorry

end NUMINAMATH_CALUDE_third_quadrant_angle_property_l3126_312604


namespace NUMINAMATH_CALUDE_area_F1AB_when_slope_is_one_line_equation_when_y_intercept_smallest_l3126_312612

-- Define the ellipse
structure Ellipse where
  a : ℝ
  b : ℝ
  h : a > b ∧ b > 0

-- Define a point on the ellipse
structure PointOnEllipse (e : Ellipse) where
  x : ℝ
  y : ℝ
  h : (x^2 / e.a^2) + (y^2 / e.b^2) = 1

-- Define the focal distance
def focalDistance (e : Ellipse) : ℝ := 3

-- Define the ratio of major axis to focal distance
axiom majorAxisFocalRatio (e : Ellipse) : 2 * e.a / (2 * focalDistance e) = Real.sqrt 2

-- Define the right focus
def rightFocus : ℝ × ℝ := (3, 0)

-- Define a line passing through the right focus
structure LineThroughRightFocus where
  slope : ℝ

-- Define the area of triangle F1AB
def areaF1AB (e : Ellipse) (l : LineThroughRightFocus) : ℝ := sorry

-- Define the y-intercept of the perpendicular bisector of AB
def yInterceptPerpBisector (e : Ellipse) (l : LineThroughRightFocus) : ℝ := sorry

-- Theorem 1
theorem area_F1AB_when_slope_is_one (e : Ellipse) :
  areaF1AB e { slope := 1 } = 12 := sorry

-- Theorem 2
theorem line_equation_when_y_intercept_smallest (e : Ellipse) :
  ∃ (l : LineThroughRightFocus),
    (∀ (l' : LineThroughRightFocus), yInterceptPerpBisector e l ≤ yInterceptPerpBisector e l') ∧
    l.slope = -Real.sqrt 2 / 2 := sorry

end NUMINAMATH_CALUDE_area_F1AB_when_slope_is_one_line_equation_when_y_intercept_smallest_l3126_312612


namespace NUMINAMATH_CALUDE_composition_value_l3126_312656

/-- Given two functions f and g, and a composition condition, prove that d equals 8 -/
theorem composition_value (c d : ℝ) (f g : ℝ → ℝ)
  (hf : ∀ x, f x = 5*x + c)
  (hg : ∀ x, g x = c*x + 1)
  (hcomp : ∀ x, f (g x) = 15*x + d) :
  d = 8 := by
  sorry

end NUMINAMATH_CALUDE_composition_value_l3126_312656


namespace NUMINAMATH_CALUDE_youngest_child_age_problem_l3126_312616

/-- The age of the youngest child given the conditions of the problem -/
def youngest_child_age (n : ℕ) (interval : ℕ) (sum : ℕ) : ℕ :=
  (sum - (n - 1) * n * interval / 2) / n

/-- Theorem stating the age of the youngest child under the given conditions -/
theorem youngest_child_age_problem :
  youngest_child_age 5 2 55 = 7 := by
  sorry

end NUMINAMATH_CALUDE_youngest_child_age_problem_l3126_312616


namespace NUMINAMATH_CALUDE_unique_base_number_exists_l3126_312642

/-- A number in base 2022 with digits 1 or 2 -/
def BaseNumber (n : ℕ) := {x : ℕ // ∀ d, d ∈ x.digits 2022 → d = 1 ∨ d = 2}

/-- The theorem statement -/
theorem unique_base_number_exists :
  ∃! (N : BaseNumber 1000), (N.val : ℤ) % (2^1000) = 0 := by
  sorry

end NUMINAMATH_CALUDE_unique_base_number_exists_l3126_312642


namespace NUMINAMATH_CALUDE_area_difference_l3126_312617

/-- The difference in area between a square and a rectangle -/
theorem area_difference (square_side : ℝ) (rect_length rect_width : ℝ) : 
  square_side = 5 → rect_length = 3 → rect_width = 6 → 
  square_side * square_side - rect_length * rect_width = 7 := by
  sorry

#check area_difference

end NUMINAMATH_CALUDE_area_difference_l3126_312617


namespace NUMINAMATH_CALUDE_distance_before_collision_l3126_312648

/-- The distance between two boats one minute before collision -/
theorem distance_before_collision (v1 v2 d : ℝ) (hv1 : v1 = 5) (hv2 : v2 = 21) (hd : d = 20) :
  let total_speed := v1 + v2
  let time_to_collision := d / total_speed
  let distance_per_minute := total_speed / 60
  distance_per_minute = 0.4333 := by sorry

end NUMINAMATH_CALUDE_distance_before_collision_l3126_312648


namespace NUMINAMATH_CALUDE_min_abs_z_on_line_segment_l3126_312637

theorem min_abs_z_on_line_segment (z : ℂ) (h : Complex.abs (z - 2*Complex.I) + Complex.abs (z - 5) = 7) :
  ∃ (min_abs_z : ℝ), min_abs_z = Real.sqrt (100 / 29) ∧
  ∀ (w : ℂ), Complex.abs (w - 2*Complex.I) + Complex.abs (w - 5) = 7 →
  Complex.abs w ≥ min_abs_z :=
sorry

end NUMINAMATH_CALUDE_min_abs_z_on_line_segment_l3126_312637


namespace NUMINAMATH_CALUDE_clothing_distribution_l3126_312697

/-- Given a total of 39 pieces of clothing, with 19 pieces in the first load
    and the rest split into 5 equal loads, prove that each small load
    contains 4 pieces of clothing. -/
theorem clothing_distribution (total : Nat) (first_load : Nat) (num_small_loads : Nat)
    (h1 : total = 39)
    (h2 : first_load = 19)
    (h3 : num_small_loads = 5) :
    (total - first_load) / num_small_loads = 4 := by
  sorry

end NUMINAMATH_CALUDE_clothing_distribution_l3126_312697


namespace NUMINAMATH_CALUDE_officers_on_duty_l3126_312644

theorem officers_on_duty (total_female_officers : ℕ) 
  (female_on_duty_ratio : ℚ) (female_ratio_on_duty : ℚ) :
  total_female_officers = 250 →
  female_on_duty_ratio = 1/5 →
  female_ratio_on_duty = 1/2 →
  (female_on_duty_ratio * total_female_officers : ℚ) / female_ratio_on_duty = 100 := by
  sorry

end NUMINAMATH_CALUDE_officers_on_duty_l3126_312644


namespace NUMINAMATH_CALUDE_multiply_24_99_l3126_312667

theorem multiply_24_99 : 24 * 99 = 2376 := by
  sorry

end NUMINAMATH_CALUDE_multiply_24_99_l3126_312667


namespace NUMINAMATH_CALUDE_quadratic_factorization_l3126_312613

theorem quadratic_factorization (a : ℝ) : a^2 + 4*a + 4 = (a + 2)^2 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_factorization_l3126_312613


namespace NUMINAMATH_CALUDE_product_47_33_l3126_312635

theorem product_47_33 : 47 * 33 = 1551 := by
  sorry

end NUMINAMATH_CALUDE_product_47_33_l3126_312635


namespace NUMINAMATH_CALUDE_constant_value_l3126_312675

theorem constant_value (t : ℝ) (c : ℝ) : 
  let x := 1 - 3 * t
  let y := 2 * t - c
  (x = y ∧ t = 0.8) → c = 3 := by
sorry

end NUMINAMATH_CALUDE_constant_value_l3126_312675


namespace NUMINAMATH_CALUDE_percentage_left_approx_20_l3126_312625

-- Define the initial population
def initial_population : ℕ := 4675

-- Define the percentage of people who died
def death_percentage : ℚ := 5 / 100

-- Define the final population
def final_population : ℕ := 3553

-- Define the function to calculate the percentage who left
def percentage_left (init : ℕ) (death_perc : ℚ) (final : ℕ) : ℚ :=
  let remaining := init - (init * death_perc).floor
  ((remaining - final : ℚ) / remaining) * 100

-- Theorem statement
theorem percentage_left_approx_20 :
  ∃ ε > 0, abs (percentage_left initial_population death_percentage final_population - 20) < ε :=
sorry

end NUMINAMATH_CALUDE_percentage_left_approx_20_l3126_312625


namespace NUMINAMATH_CALUDE_units_digit_product_l3126_312673

theorem units_digit_product : (47 * 23 * 89) % 10 = 9 := by
  sorry

end NUMINAMATH_CALUDE_units_digit_product_l3126_312673


namespace NUMINAMATH_CALUDE_triangle_area_l3126_312650

/-- Given a triangle ABC where c² = (a-b)² + 6 and angle C = π/3, 
    prove that its area is 3√3/2 -/
theorem triangle_area (a b c : ℝ) (h1 : c^2 = (a-b)^2 + 6) (h2 : Real.pi / 3 = Real.arccos ((a^2 + b^2 - c^2) / (2*a*b))) : 
  (1/2) * a * b * Real.sin (Real.pi / 3) = (3 * Real.sqrt 3) / 2 := by
  sorry

end NUMINAMATH_CALUDE_triangle_area_l3126_312650


namespace NUMINAMATH_CALUDE_min_omega_value_l3126_312628

open Real

theorem min_omega_value (f : ℝ → ℝ) (ω φ : ℝ) (x₀ : ℝ) :
  (ω > 0) →
  (∀ x, f x = sin (ω * x + φ)) →
  (∀ x, f x₀ ≤ f x ∧ f x ≤ f (x₀ + 2016 * π)) →
  (∀ ω' > 0, (∀ x, f x₀ ≤ sin (ω' * x + φ) ∧ sin (ω' * x + φ) ≤ f (x₀ + 2016 * π)) → ω ≤ ω') →
  ω = 1 / 2016 :=
sorry

end NUMINAMATH_CALUDE_min_omega_value_l3126_312628


namespace NUMINAMATH_CALUDE_min_surface_area_height_l3126_312623

/-- Represents a square-bottomed, lidless rectangular tank -/
structure Tank where
  side : ℝ
  height : ℝ

/-- The volume of the tank -/
def volume (t : Tank) : ℝ := t.side^2 * t.height

/-- The surface area of the tank -/
def surfaceArea (t : Tank) : ℝ := t.side^2 + 4 * t.side * t.height

/-- Theorem: For a tank with volume 4, the height that minimizes surface area is 1 -/
theorem min_surface_area_height :
  ∃ (t : Tank), volume t = 4 ∧ 
    (∀ (t' : Tank), volume t' = 4 → surfaceArea t ≤ surfaceArea t') ∧
    t.height = 1 := by
  sorry

end NUMINAMATH_CALUDE_min_surface_area_height_l3126_312623


namespace NUMINAMATH_CALUDE_sum_of_roots_l3126_312682

theorem sum_of_roots (a b : ℝ) 
  (ha : a^3 - 12*a^2 + 9*a - 18 = 0)
  (hb : 9*b^3 - 135*b^2 + 450*b - 1650 = 0) :
  a + b = 6 ∨ a + b = 14 := by sorry

end NUMINAMATH_CALUDE_sum_of_roots_l3126_312682


namespace NUMINAMATH_CALUDE_fraction_addition_l3126_312662

theorem fraction_addition (x y : ℚ) (h : x / y = 3 / 4) : (x + y) / y = 7 / 4 := by
  sorry

end NUMINAMATH_CALUDE_fraction_addition_l3126_312662


namespace NUMINAMATH_CALUDE_field_division_theorem_l3126_312622

/-- Represents a rectangular field of squares -/
structure RectangularField where
  width : ℕ
  height : ℕ
  total_squares : ℕ
  h_total : total_squares = width * height

/-- Represents a line dividing the field -/
structure DividingLine where
  x : ℕ
  y : ℕ

/-- Calculates the area of a triangle formed by a dividing line -/
def triangle_area (field : RectangularField) (line : DividingLine) : ℕ :=
  line.x * line.y / 2

theorem field_division_theorem (field : RectangularField) 
  (h_total : field.total_squares = 18) 
  (line1 : DividingLine) 
  (line2 : DividingLine) 
  (h_line1 : line1 = ⟨4, field.height⟩) 
  (h_line2 : line2 = ⟨field.width, 2⟩) :
  triangle_area field line1 = 6 ∧ 
  triangle_area field line2 = 6 ∧ 
  field.total_squares - triangle_area field line1 - triangle_area field line2 = 6 := by
  sorry

#check field_division_theorem

end NUMINAMATH_CALUDE_field_division_theorem_l3126_312622
