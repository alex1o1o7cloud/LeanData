import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_chord_length_sqrt_two_l655_65556

/-- The length of the chord intercepted by the line y=x on the circle x^2+y^2-2y=0 is √2 -/
theorem chord_length_sqrt_two : 
  ∃ (p q : ℝ × ℝ), 
    p.2 = p.1 ∧ 
    q.2 = q.1 ∧ 
    p.1^2 + p.2^2 - 2*p.2 = 0 ∧ 
    q.1^2 + q.2^2 - 2*q.2 = 0 ∧ 
    p ≠ q ∧ 
    Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2) = Real.sqrt 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_chord_length_sqrt_two_l655_65556


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_roof_dimension_difference_l655_65507

/-- Represents the dimensions of a rectangular roof -/
structure RoofDimensions where
  width : ℝ
  length : ℝ

/-- The roof dimensions satisfy the given conditions -/
def valid_roof (r : RoofDimensions) : Prop :=
  r.length = 4 * r.width ∧ r.width * r.length = 675

/-- The difference between length and width is approximately 38.97 -/
theorem roof_dimension_difference (r : RoofDimensions) 
  (h : valid_roof r) : 
  ∃ ε > 0, |r.length - r.width - 38.97| < ε := by
  sorry

#check roof_dimension_difference

end NUMINAMATH_CALUDE_ERRORFEEDBACK_roof_dimension_difference_l655_65507


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_coin_difference_l655_65533

/-- Proves that the difference between nickels and pennies combined and quarters and dimes is -7 --/
theorem coin_difference (total : ℤ) (quarters : ℤ) (dimes : ℤ) (nickel_ratio : ℤ) (penny_ratio : ℤ) :
  total = 127 →
  quarters = 39 →
  dimes = 28 →
  nickel_ratio = 3 →
  penny_ratio = 2 →
  (total - quarters - dimes) % (nickel_ratio + penny_ratio) = 0 →
  (total - quarters - dimes) / (nickel_ratio + penny_ratio) * nickel_ratio +
  (total - quarters - dimes) / (nickel_ratio + penny_ratio) * penny_ratio -
  (quarters + dimes) = -7 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_coin_difference_l655_65533


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_root_negative_eight_l655_65561

theorem cube_root_negative_eight : -Real.rpow (-8 : ℝ) (1/3 : ℝ) = 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_root_negative_eight_l655_65561


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_alpha_equals_four_fifths_l655_65511

/-- Given an angle α whose initial side is on the positive x-axis
    and whose terminal side passes through the point (3, 4),
    prove that sin α = 4/5 -/
theorem sin_alpha_equals_four_fifths (α : ℝ) :
  (∃ (x y : ℝ), x = 3 ∧ y = 4 ∧ 
   α = Real.arctan (y / x) ∧ 
   α ≥ 0 ∧ α < 2 * Real.pi) →
  Real.sin α = 4/5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_alpha_equals_four_fifths_l655_65511


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_range_l655_65587

-- Define the function
noncomputable def f (x : ℝ) : ℝ := Real.sin x - Real.sin (abs x)

-- State the theorem
theorem f_range : Set.range f = Set.Icc (-2) 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_range_l655_65587


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l655_65596

-- Define the condition
def condition (x : ℝ) : Prop := x * (Real.log 4 / Real.log 3) = 1

-- Theorem statement
theorem problem_solution (x : ℝ) (h : condition x) :
  x = Real.log 3 / Real.log 4 ∧ (4 : ℝ)^x + (4 : ℝ)^(-x) = 10/3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l655_65596


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_existence_of_Q_and_R_l655_65545

-- Define the circles and points
variable (C C' : Set (Fin 2 → ℝ))
variable (X Y P Q R : Fin 2 → ℝ)

-- Define the conditions
def circles_intersect (C C' : Set (Fin 2 → ℝ)) (X Y : Fin 2 → ℝ) : Prop :=
  X ∈ C ∧ X ∈ C' ∧ Y ∈ C ∧ Y ∈ C'

def XY_diameter (C : Set (Fin 2 → ℝ)) (X Y : Fin 2 → ℝ) : Prop :=
  ∀ Z : Fin 2 → ℝ, Z ∈ C → ‖X - Z‖ + ‖Z - Y‖ = ‖X - Y‖

def P_on_C'_inside_C (C C' : Set (Fin 2 → ℝ)) (P : Fin 2 → ℝ) : Prop :=
  P ∈ C' ∧ P ∈ interior C

-- Define perpendicularity
def perpendicular (A B C D : Fin 2 → ℝ) : Prop :=
  (B - A) • (D - C) = 0

-- Theorem statement
theorem existence_of_Q_and_R
    (C C' : Set (Fin 2 → ℝ))
    (X Y P : Fin 2 → ℝ) :
  circles_intersect C C' X Y →
  XY_diameter C X Y →
  P_on_C'_inside_C C C' P →
  ∃ Q R : Fin 2 → ℝ, Q ∈ C ∧ R ∈ C ∧ perpendicular Q R X Y ∧ perpendicular P Q P R :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_existence_of_Q_and_R_l655_65545


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_problem_l655_65543

theorem triangle_problem (A B C AB BC : ℝ) (h_AB : AB = Real.sqrt 3) (h_BC : BC = 2)
  (h_cosB : Real.cos B = -1/2) : 
  Real.sin C = Real.sqrt 3 / 2 ∧ 0 < C ∧ C ≤ Real.pi/3 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_problem_l655_65543


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_minus_sin_equals_one_l655_65508

theorem cos_minus_sin_equals_one (α : ℝ) : 
  (Real.cos (π + 2 * α)) / (Real.sin (α + π / 4)) = -Real.sqrt 2 → Real.cos α - Real.sin α = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_minus_sin_equals_one_l655_65508


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_ADE_area_l655_65558

noncomputable def hyperbola (x : ℝ) : ℝ := -6 / x

def original_line (k b x : ℝ) : ℝ := k * x + b

def new_line (k b x : ℝ) : ℝ := k * x + (b + 8)

def point_A : ℝ × ℝ := (-3, 2)
def point_B : ℝ × ℝ := (1, -6)
def point_D : ℝ × ℝ := (3, -2)
def point_E : ℝ × ℝ := (-1, 6)

theorem triangle_ADE_area : 
  ∀ (k b : ℝ),
  original_line k b (point_A.1) = point_A.2 ∧
  original_line k b (point_B.1) = point_B.2 ∧
  hyperbola (point_A.1) = point_A.2 ∧
  hyperbola (point_B.1) = point_B.2 →
  new_line k b (point_D.1) = point_D.2 ∧
  new_line k b (point_E.1) = point_E.2 ∧
  hyperbola (point_D.1) = point_D.2 ∧
  hyperbola (point_E.1) = point_E.2 →
  (abs ((point_A.1 * (point_D.2 - point_E.2) + 
         point_D.1 * (point_E.2 - point_A.2) + 
         point_E.1 * (point_A.2 - point_D.2)) / 2) : ℝ) = 16 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_ADE_area_l655_65558


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_twin_points_existence_l655_65541

-- Define the concept of twin points
noncomputable def are_twin_points (F : ℝ → ℝ → ℝ) (x₁ y₁ x₂ y₂ : ℝ) : Prop :=
  F x₁ y₁ = 0 ∧ F x₂ y₂ = 0 ∧ x₁ ≤ x₂ ∧ y₁ ≥ y₂

-- Define the four curves
noncomputable def curve1 (x y : ℝ) : ℝ := x^2/20 + y^2/16 - 1
noncomputable def curve2 (x y : ℝ) : ℝ := x^2/20 - y^2/16 - 1
noncomputable def curve3 (x y : ℝ) : ℝ := y^2 - 4*x
noncomputable def curve4 (x y : ℝ) : ℝ := abs x + abs y - 1

-- State the theorem
theorem twin_points_existence :
  (∃ x₁ y₁ x₂ y₂, are_twin_points curve1 x₁ y₁ x₂ y₂ ∧ x₁ * y₁ > 0 ∧ x₂ * y₂ > 0) ∧
  (∃ x₁ y₁ x₂ y₂, are_twin_points curve3 x₁ y₁ x₂ y₂) ∧
  (∃ x₁ y₁ x₂ y₂, are_twin_points curve4 x₁ y₁ x₂ y₂) ∧
  ¬(∃ x₁ y₁ x₂ y₂, are_twin_points curve2 x₁ y₁ x₂ y₂ ∧ x₁ * y₁ > 0 ∧ x₂ * y₂ > 0) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_twin_points_existence_l655_65541


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotone_decreasing_l655_65569

noncomputable def f (x : ℝ) := Real.sin x * Real.cos x + Real.sqrt 3 * (Real.cos x)^2 - Real.sqrt 3 / 2

theorem f_monotone_decreasing :
  ∀ x y : ℝ, x ∈ Set.Ioo (0 : ℝ) π → y ∈ Set.Ioo (0 : ℝ) π →
    (f x ≤ f y ↔ π/12 ≤ y ∧ y ≤ x ∧ x ≤ 7*π/12) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotone_decreasing_l655_65569


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_at_one_l655_65564

-- Define the function f(x) = x ln x
noncomputable def f (x : ℝ) : ℝ := x * Real.log x

-- State the theorem
theorem tangent_line_at_one :
  let f' : ℝ → ℝ := λ x ↦ Real.log x + 1  -- f'(x) = ln x + 1
  let tangent_line : ℝ → ℝ := λ x ↦ x - 1  -- y = x - 1
  (∀ x, x > 0 → HasDerivAt f (f' x) x) ∧  -- f'(x) is the derivative of f(x) for x > 0
  HasDerivAt f 1 1 ∧  -- The derivative of f at x=1 is 1
  f 1 = 0 ∧  -- f(1) = 0
  (∀ x, tangent_line x = f 1 + (x - 1) * (f' 1))  -- Equation of the tangent line
  := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_at_one_l655_65564


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_children_ages_solution_l655_65560

def is_arithmetic_sequence (s : List ℕ) : Prop :=
  ∃ d : ℕ, ∀ i : ℕ, i + 1 < s.length → s[i+1]! - s[i]! = d

def valid_children_ages (ages : List ℕ) : Prop :=
  ages.sum = 50 ∧
  ages.maximum? = some 13 ∧
  10 ∈ ages ∧
  is_arithmetic_sequence (ages.filter (λ x => x ≠ 10))

theorem children_ages_solution :
  ∃! ages : List ℕ, valid_children_ages ages ∧ ages = [13, 11, 10, 9, 7] :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_children_ages_solution_l655_65560


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_phi_for_even_function_l655_65595

-- Define the function f
noncomputable def f (φ : ℝ) (x : ℝ) : ℝ := Real.sin (2 * (x + Real.pi / 8) + φ)

-- State the theorem
theorem smallest_phi_for_even_function :
  ∃ (φ : ℝ), φ > 0 ∧ 
  (∀ (x : ℝ), f φ x = f φ (-x)) ∧
  (∀ (ψ : ℝ), ψ > 0 ∧ (∀ (x : ℝ), f ψ x = f ψ (-x)) → φ ≤ ψ) ∧
  φ = Real.pi / 4 := by
  sorry

-- Additional lemma to support the main theorem
lemma f_is_even (φ : ℝ) (h : φ = Real.pi / 4) :
  ∀ (x : ℝ), f φ x = f φ (-x) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_phi_for_even_function_l655_65595


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_water_height_after_valve_open_l655_65531

/-- The final height of water in a system of two connected vessels -/
noncomputable def final_water_height (initial_height : ℝ) (water_density oil_density : ℝ) : ℝ :=
  initial_height * water_density / (water_density + oil_density)

theorem water_height_after_valve_open
  (initial_height : ℝ)
  (water_density oil_density : ℝ)
  (h_initial : initial_height = 40)
  (h_water_density : water_density = 1000)
  (h_oil_density : oil_density = 700) :
  final_water_height initial_height water_density oil_density = 34 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_water_height_after_valve_open_l655_65531


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_blue_path_exists_l655_65588

/-- A city in the Rainbow Nation -/
structure City where
  name : String

/-- An airline route between two cities -/
inductive Route : Type
  | RedRocket : City → City → Route
  | BlueBoeing : City → City → Route

/-- A path connects two cities if it forms a valid sequence of routes -/
def connects_cities : List Route → City → City → Prop
  | [], start, finish => start = finish
  | (r::rs), start, finish => 
    match r with
    | Route.RedRocket c1 c2 => c1 = start ∧ connects_cities rs c2 finish
    | Route.BlueBoeing c1 c2 => c1 = start ∧ connects_cities rs c2 finish

/-- The graph representing the Rainbow Nation's air travel network -/
structure RainbowNation where
  cities : Set City
  routes : Set Route
  beanville : City
  mieliestad : City
  all_connected : ∀ (c1 c2 : City), c1 ∈ cities → c2 ∈ cities → 
    ∃ (path : List Route), connects_cities path c1 c2
  no_red_path : ¬∃ (path : List Route), 
    (∀ r ∈ path, ∃ c1 c2, r = Route.RedRocket c1 c2) ∧ 
    connects_cities path beanville mieliestad

/-- Main theorem: There exists a blue path with at most one stop between any two cities -/
theorem blue_path_exists (rn : RainbowNation) : 
  ∀ (c1 c2 : City), c1 ∈ rn.cities → c2 ∈ rn.cities → 
    ∃ (path : List Route), 
      (∀ r ∈ path, ∃ s t, r = Route.BlueBoeing s t) ∧ 
      connects_cities path c1 c2 ∧ 
      path.length ≤ 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_blue_path_exists_l655_65588


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_circle_on_chessboard_l655_65579

/-- The side length of a chessboard square -/
def chessboard_square_side : ℝ := 4

/-- The circumference of the largest circle that can be drawn entirely on the black squares of a chessboard -/
noncomputable def largest_circle_circumference : ℝ := 4 * Real.pi * Real.sqrt 10

/-- Theorem stating that the largest circle's circumference on black squares of a chessboard with square side length 4 is 4π√10 -/
theorem largest_circle_on_chessboard :
  largest_circle_circumference = 4 * Real.pi * Real.sqrt 10 :=
by
  -- Unfold the definition of largest_circle_circumference
  unfold largest_circle_circumference
  -- The equality is now trivial
  rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_circle_on_chessboard_l655_65579


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sine_translation_even_function_l655_65567

noncomputable def f (φ : Real) (x : Real) : Real := Real.sin (2 * (x + Real.pi / 8) + φ)

theorem sine_translation_even_function (φ : Real) :
  (∀ x, f φ x = f φ (-x)) → ∃ k : Int, φ = k * Real.pi + Real.pi / 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sine_translation_even_function_l655_65567


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_stewart_farm_horse_food_proof_l655_65506

/-- Calculates the total amount of horse food needed per day on Stewart farm -/
def stewart_farm_horse_food (sheep_count : ℕ) (sheep_horse_ratio : ℕ) (food_per_horse : ℕ) : ℕ :=
  (sheep_count * sheep_horse_ratio) * food_per_horse

/-- Proves the total amount of horse food needed per day on Stewart farm -/
theorem stewart_farm_horse_food_proof :
  stewart_farm_horse_food 8 7 230 = 12880 := by
  -- Unfold the definition of stewart_farm_horse_food
  unfold stewart_farm_horse_food
  -- Evaluate the arithmetic expression
  norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_stewart_farm_horse_food_proof_l655_65506


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_value_from_cos_supplement_l655_65549

theorem tan_value_from_cos_supplement (α : ℝ) 
  (h1 : α ∈ Set.Ioo (π / 2) π) 
  (h2 : Real.cos (π - α) = 3 / 5) : 
  Real.tan α = -4 / 3 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_value_from_cos_supplement_l655_65549


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetry_plane_theorem_l655_65517

/-- The plane of symmetry for two given points -/
def plane_of_symmetry (A A' : ℝ × ℝ × ℝ) : ℝ → ℝ → ℝ → Prop :=
  λ x y z ↦ 2/3 * x - 2/3 * y + 1/3 * z - 7/3 = 0

/-- Two points are symmetric with respect to a plane if their distances to any point on the plane are equal -/
def symmetric_points (A A' : ℝ × ℝ × ℝ) (P : ℝ → ℝ → ℝ → Prop) : Prop :=
  ∀ x y z, P x y z → 
    (x - A.1)^2 + (y - A.2.1)^2 + (z - A.2.2)^2 = 
    (x - A'.1)^2 + (y - A'.2.1)^2 + (z - A'.2.2)^2

theorem symmetry_plane_theorem (A A' : ℝ × ℝ × ℝ) 
  (h1 : A = (7, 1, 4)) (h2 : A' = (3, 5, 2)) : 
  symmetric_points A A' (plane_of_symmetry A A') := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetry_plane_theorem_l655_65517


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_composite_arrangement_theorem_l655_65542

/-- A function that checks if a number is composite -/
def IsComposite (k : ℕ) : Prop := k > 1 ∧ ∃ m, 1 < m ∧ m < k ∧ k % m = 0

/-- A function that checks if two natural numbers have a common factor greater than 1 -/
def HasCommonFactorGreaterThanOne (a b : ℕ) : Prop :=
  ∃ k : ℕ, k > 1 ∧ a % k = 0 ∧ b % k = 0

/-- The main theorem -/
theorem composite_arrangement_theorem (n : ℕ) (h : n ≥ 6) :
  ∃ (seq : List ℕ),
    (∀ k ∈ seq, k ≤ n ∧ IsComposite k) ∧
    (∀ k ≤ n, IsComposite k → k ∈ seq) ∧
    (∀ i j, i + 1 = j → j < seq.length → HasCommonFactorGreaterThanOne (seq.get ⟨i, by sorry⟩) (seq.get ⟨j, by sorry⟩)) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_composite_arrangement_theorem_l655_65542


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_semicircle_intersection_l655_65585

-- Define the parabola
def parabola (a : ℝ) (x y : ℝ) : Prop := y^2 = 4*a*x

-- Define the focus A
def focus (a : ℝ) : ℝ × ℝ := (a, 0)

-- Define point B
def point_B (a : ℝ) : ℝ × ℝ := (a + 4, 0)

-- Define the semicircle
def semicircle (a : ℝ) (x y : ℝ) : Prop :=
  (x - (a + 4))^2 + y^2 = (a + 4)^2

-- M and N are intersection points of parabola and semicircle
def intersection_points (a : ℝ) (M N : ℝ × ℝ) : Prop :=
  parabola a M.1 M.2 ∧ semicircle a M.1 M.2 ∧
  parabola a N.1 N.2 ∧ semicircle a N.1 N.2 ∧
  M ≠ N

-- P is midpoint of MN
def is_midpoint (P M N : ℝ × ℝ) : Prop :=
  P.1 = (M.1 + N.1) / 2 ∧ P.2 = (M.2 + N.2) / 2

-- Distance function
noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

-- Main theorem
theorem parabola_semicircle_intersection (a : ℝ) (M N P : ℝ × ℝ) 
  (h_a : a > 0)
  (h_intersection : intersection_points a M N)
  (h_midpoint : is_midpoint P M N) :
  (distance (focus a) M + distance (focus a) N = 8) ∧
  ¬∃ a' : ℝ, distance (focus a') M - distance (focus a') P = 
           distance (focus a') P - distance (focus a') N :=
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_semicircle_intersection_l655_65585


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_on_ray_sin_plus_cos_l655_65552

theorem angle_on_ray_sin_plus_cos (θ : Real) :
  (∃ (x y : Real), x ≤ 0 ∧ y = 2*x ∧ 
   Real.sin θ = y / Real.sqrt (x^2 + y^2) ∧
   Real.cos θ = x / Real.sqrt (x^2 + y^2)) →
  Real.sin θ + Real.cos θ = -3 * Real.sqrt 5 / 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_on_ray_sin_plus_cos_l655_65552


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_guard_demand_is_199_l655_65554

/-- Represents the scenario of a guard and an outsider --/
structure GuardOutsiderScenario where
  betAmount : ℕ  -- The amount of the bet
  guardDemand : ℕ  -- The amount the guard demands

/-- Calculates the outsider's loss if they pay the guard --/
def outsiderLossIfPay (scenario : GuardOutsiderScenario) : ℤ :=
  scenario.guardDemand - scenario.betAmount

/-- Calculates the outsider's loss if they don't pay the guard --/
def outsiderLossIfNotPay (scenario : GuardOutsiderScenario) : ℕ :=
  scenario.betAmount

/-- Determines if the outsider will pay based on their potential losses --/
def outsiderWillPay (scenario : GuardOutsiderScenario) : Prop :=
  (Int.toNat (outsiderLossIfPay scenario)) < outsiderLossIfNotPay scenario

/-- The maximum amount the guard can demand --/
def maxGuardDemand : ℕ := 199

/-- Theorem stating that 199 is the maximum amount the guard can demand --/
theorem max_guard_demand_is_199 :
  ∀ (demand : ℕ),
    let scenario := GuardOutsiderScenario.mk 100 demand
    demand ≤ maxGuardDemand ↔ outsiderWillPay scenario :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_guard_demand_is_199_l655_65554


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_no_special_numbers_exist_l655_65594

def digits (n : ℕ) : List ℕ :=
  if n < 10 then [n]
  else (n % 10) :: digits (n / 10)

def num_digits (n : ℕ) : ℕ :=
  (digits n).length

theorem no_special_numbers_exist (n : ℕ) : 
  ¬ ∃ (M N : ℕ), 
    (∀ d : ℕ, d ∈ digits M → d % 2 = 0) ∧ 
    (∀ d : ℕ, d ∈ digits N → d % 2 = 1) ∧
    (∀ d : ℕ, d < 10 → (d ∈ digits M ∨ d ∈ digits N)) ∧
    (num_digits M = n) ∧ 
    (num_digits N = n) ∧
    (M % N = 0) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_no_special_numbers_exist_l655_65594


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_f_period_l655_65534

noncomputable def f (x : ℝ) := Real.sin (Real.pi / 2 - x) * Real.sin x - Real.sqrt 3 * (Real.cos x) ^ 2

theorem f_properties :
  -- Smallest positive period is 2
  (∃ (p : ℝ), p > 0 ∧ (∀ (x : ℝ), f (x + p) = f x) ∧
    (∀ (q : ℝ), q > 0 ∧ (∀ (x : ℝ), f (x + q) = f x) → p ≤ q)) ∧
  -- Maximum value is 1 - √3/2
  (∀ (x : ℝ), f x ≤ 1 - Real.sqrt 3 / 2) ∧
  (∃ (x : ℝ), f x = 1 - Real.sqrt 3 / 2) ∧
  -- Monotonicity on [π/6, 2π/3]
  (∀ (x y : ℝ), Real.pi / 6 ≤ x ∧ x < y ∧ y ≤ 5 * Real.pi / 12 → f x < f y) ∧
  (∀ (x y : ℝ), 5 * Real.pi / 12 ≤ x ∧ x < y ∧ y ≤ 2 * Real.pi / 3 → f x > f y) :=
by sorry

-- Separate theorem for the period value
theorem f_period : ∃ (p : ℝ), p = 2 ∧ p > 0 ∧ (∀ (x : ℝ), f (x + p) = f x) ∧
    (∀ (q : ℝ), q > 0 ∧ (∀ (x : ℝ), f (x + q) = f x) → p ≤ q) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_f_period_l655_65534


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_equation_proof_l655_65521

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a line in 2D space -/
structure Line where
  slope : ℝ
  yIntercept : ℝ

/-- Check if a point lies on a line -/
def pointOnLine (p : Point) (l : Line) : Prop :=
  p.y = l.slope * p.x + l.yIntercept

/-- The slope of a line passing through two points -/
noncomputable def slopeBetweenPoints (p1 p2 : Point) : ℝ :=
  (p2.y - p1.y) / (p2.x - p1.x)

theorem line_equation_proof (p : Point) (m : ℝ) :
  p.x = 1 ∧ p.y = 2 ∧ m = 3 →
  ∃ (l : Line), l.slope = m ∧ l.yIntercept = -1 ∧ pointOnLine p l :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_equation_proof_l655_65521


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rawot_market_results_l655_65510

/-- Represents the demand function for a season --/
structure DemandFunction where
  intercept : ℚ
  slope : ℚ

/-- Represents the supply function for a season --/
structure SupplyFunction where
  intercept : ℚ
  slope : ℚ

/-- Represents the market conditions for a product in different seasons --/
structure Market where
  spring_demand : DemandFunction
  summer_demand : DemandFunction
  winter_demand : DemandFunction
  spring_supply : SupplyFunction
  winter_supply : SupplyFunction

/-- The market conditions for the product "rawot" in Konyr --/
def rawot_market : Market := {
  spring_demand := { intercept := 19, slope := 1 },
  summer_demand := { intercept := 38, slope := 2 },
  winter_demand := { intercept := 19/2, slope := 1/2 },
  spring_supply := { intercept := -8, slope := 2 },
  winter_supply := { intercept := -4, slope := 1 }
}

/-- Calculates the equilibrium price and quantity for a given demand and supply function --/
def equilibrium (demand : DemandFunction) (supply : SupplyFunction) : ℚ × ℚ :=
  let price := (demand.intercept - supply.intercept) / (supply.slope + demand.slope)
  let quantity := demand.intercept - demand.slope * price
  (price, quantity)

/-- Theorem stating the main results about the rawot market --/
theorem rawot_market_results (m : Market) (h : m = rawot_market) :
  let (summer_price, summer_quantity) := equilibrium m.summer_demand m.spring_supply
  let (winter_price, winter_quantity) := equilibrium m.winter_demand m.winter_supply
  let summer_deficit := m.summer_demand.intercept - 2 * winter_price - (m.spring_supply.intercept + m.spring_supply.slope * winter_price)
  let summer_expenditure := summer_price * summer_quantity
  let zero_demand_price := m.spring_demand.intercept
  (summer_deficit = 10) ∧
  (summer_expenditure = 345/2) ∧
  (zero_demand_price = 19) ∧
  (∀ season_demand : DemandFunction, season_demand.intercept = zero_demand_price → season_demand.intercept - season_demand.slope * zero_demand_price = 0) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_rawot_market_results_l655_65510


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_ratio_on_circle_l655_65583

/-- A point with integer coordinates on the circle x^2 + y^2 = 16 -/
structure CirclePoint where
  x : ℤ
  y : ℤ
  on_circle : x^2 + y^2 = 16

/-- Distance between two CirclePoints -/
noncomputable def distance (p q : CirclePoint) : ℝ :=
  Real.sqrt ((p.x - q.x)^2 + (p.y - q.y)^2)

/-- Statement of the problem -/
theorem max_ratio_on_circle :
  ∃ (A B C D : CirclePoint),
    ¬ ∃ (m n : ℤ), (distance A B)^2 = (m : ℝ)^2 / (n : ℝ)^2 ∧
    ¬ ∃ (k l : ℤ), (distance C D)^2 = (k : ℝ)^2 / (l : ℝ)^2 ∧
    ∀ (P Q R S : CirclePoint),
      (¬ ∃ (a b : ℤ), (distance P Q)^2 = (a : ℝ)^2 / (b : ℝ)^2) →
      (¬ ∃ (c d : ℤ), (distance R S)^2 = (c : ℝ)^2 / (d : ℝ)^2) →
      distance P Q / distance R S ≤ 2 * Real.sqrt 2 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_ratio_on_circle_l655_65583


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_parallel_to_plane_no_common_points_l655_65568

-- Define the basic structures
structure Point where

structure Line where

structure Plane where

-- Define the relationships
def parallel (l : Line) (p : Plane) : Prop := sorry

def inside (l : Line) (p : Plane) : Prop := sorry

def has_common_point (l1 l2 : Line) : Prop := sorry

-- State the theorem
theorem line_parallel_to_plane_no_common_points 
  (l : Line) (α : Plane) (m : Line) :
  parallel l α → inside m α → ¬ has_common_point l m := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_parallel_to_plane_no_common_points_l655_65568


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fraction_simplification_l655_65536

theorem fraction_simplification :
  (1 : ℝ) / ((1 : ℝ) + Real.sqrt 3) * (1 : ℝ) / ((1 : ℝ) - (3 : ℝ) ^ (1/3)) =
  (1 : ℝ) / ((1 : ℝ) - (3 : ℝ) ^ (1/3) + Real.sqrt 3 - (3 * Real.sqrt 3) ^ (1/3)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fraction_simplification_l655_65536


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_shortest_chord_specific_circle_l655_65504

/-- The length of the shortest chord passing through a point on a circle. -/
noncomputable def shortestChordLength (x₀ y₀ a b r : ℝ) : ℝ :=
  2 * Real.sqrt (r^2 - ((x₀ - a)^2 + (y₀ - b)^2))

/-- Theorem: The length of the shortest chord passing through (3,1) on the circle (x-2)^2 + (y-2)^2 = 4 is 2√2. -/
theorem shortest_chord_specific_circle :
  shortestChordLength 3 1 2 2 2 = 2 * Real.sqrt 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_shortest_chord_specific_circle_l655_65504


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_point_orthogonality_l655_65520

-- Define the ellipse C
noncomputable def ellipse_C (x y : ℝ) : Prop := x^2 / 4 + y^2 / 3 = 1

-- Define the line l
def line_l (k m x y : ℝ) : Prop := y = k * x + m

-- Define the tangent point P
noncomputable def point_P (k m : ℝ) : ℝ × ℝ := (-4 * k / m, 3 / m)

-- Define the intersection point Q
def point_Q (k m : ℝ) : ℝ × ℝ := (4, 4 * k + m)

-- Define the point M
def point_M : ℝ × ℝ := (1, 0)

-- Theorem statement
theorem tangent_point_orthogonality 
  (k m : ℝ) 
  (h_tangent : ∃! p : ℝ × ℝ, ellipse_C p.1 p.2 ∧ line_l k m p.1 p.2) :
  let P := point_P k m
  let Q := point_Q k m
  let M := point_M
  ((P.1 - M.1) * (Q.1 - M.1) + (P.2 - M.2) * (Q.2 - M.2) = 0) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_point_orthogonality_l655_65520


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangle_area_similarity_l655_65501

/-- Represents a rectangle with side lengths a and b -/
structure Rectangle where
  a : ℝ
  b : ℝ

/-- The area of a rectangle -/
def Rectangle.area (r : Rectangle) : ℝ := r.a * r.b

/-- The diagonal length of a rectangle -/
noncomputable def Rectangle.diagonal (r : Rectangle) : ℝ := Real.sqrt (r.a^2 + r.b^2)

/-- Similarity ratio between two rectangles -/
noncomputable def similarityRatio (r1 r2 : Rectangle) : ℝ := r2.a / r1.a

theorem rectangle_area_similarity (r1 r2 : Rectangle) 
  (h1 : r1.a = 3)
  (h2 : r1.area = 21)
  (h3 : r2.diagonal = 20)
  (h4 : similarityRatio r1 r2 = r1.b / r1.a) :
  abs (r2.area - 144.6) < 0.1 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangle_area_similarity_l655_65501


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_h_equality_solution_l655_65500

-- Define the function h as noncomputable
noncomputable def h (x : ℝ) : ℝ := ((x + 5) / 5) ^ (1/3)

-- State the theorem
theorem h_equality_solution :
  ∃ (x : ℝ), h (2 * x) = 4 * h x ∧ x = -315/62 := by
  -- Proof is skipped with 'sorry'
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_h_equality_solution_l655_65500


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_eccentricity_with_parabola_intersection_l655_65551

/-- The eccentricity of an ellipse intersecting a parabola under specific conditions -/
theorem ellipse_eccentricity_with_parabola_intersection :
  ∀ (a b : ℝ) (T : ℝ × ℝ),
    a > b → b > 0 →
    T.1 = 1 → T.2 = 2 →
    T.1^2 / a^2 + T.2^2 / b^2 = 1 →
    T.2^2 = 4 * T.1 →
    Real.sqrt (a^2 - b^2) = 1 →
    (Real.sqrt 2 - 1)^2 * a^2 = 1 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_eccentricity_with_parabola_intersection_l655_65551


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_election_winner_votes_l655_65573

theorem election_winner_votes (total_votes : ℕ) (winner_percentage : ℚ) (vote_difference : ℕ) : 
  winner_percentage = 7/10 →
  vote_difference = 280 →
  (winner_percentage * total_votes - (1 - winner_percentage) * total_votes).floor = vote_difference →
  (winner_percentage * total_votes).floor = 490 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_election_winner_votes_l655_65573


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_roots_l655_65539

/-- Given a quadratic trinomial ax^2 + bx + c with two roots,
    prove that 3ax^2 + 2(a+b)x + (b+c) also has two roots. -/
theorem quadratic_roots (a b c : ℝ) (h : b^2 - 4*a*c > 0) :
  ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧
    (3*a*x₁^2 + 2*(a+b)*x₁ + (b+c) = 0) ∧
    (3*a*x₂^2 + 2*(a+b)*x₂ + (b+c) = 0) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_roots_l655_65539


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_invariant_under_reversal_l655_65559

/-- Represents a three-digit number -/
structure ThreeDigitNumber where
  hundreds : Nat
  tens : Nat
  ones : Nat
  h_range : hundreds ∈ Finset.range 10
  t_range : tens ∈ Finset.range 10
  o_range : ones ∈ Finset.range 10

/-- Calculates the value of a three-digit number -/
def value (n : ThreeDigitNumber) : Nat :=
  100 * n.hundreds + 10 * n.tens + n.ones

/-- Reverses the digits of a three-digit number -/
def reverse (n : ThreeDigitNumber) : ThreeDigitNumber where
  hundreds := n.ones
  tens := n.tens
  ones := n.hundreds
  h_range := n.o_range
  t_range := n.t_range
  o_range := n.h_range

/-- The main theorem -/
theorem sum_invariant_under_reversal 
  (a b c : ThreeDigitNumber)
  (h_sum : value a + value b + value c = 1665)
  (h_distinct : {a.hundreds, a.tens, a.ones, b.hundreds, b.tens, b.ones, c.hundreds, c.tens, c.ones} = Finset.range 9) :
  value (reverse a) + value (reverse b) + value (reverse c) = 1665 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_invariant_under_reversal_l655_65559


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cheese_division_theorem_l655_65582

theorem cheese_division_theorem (initial_larger initial_smaller : ℕ) :
  initial_larger > initial_smaller →
  (let bite1 := initial_larger - initial_smaller
   let smaller1 := initial_smaller
   let bite2 := bite1 - smaller1
   let smaller2 := smaller1
   let bite3 := bite2 - smaller2
   let smaller3 := smaller2
   bite3 = 20 ∧ smaller3 = 20) →
  initial_larger + initial_smaller = 680 :=
by
  intros h1 h2
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cheese_division_theorem_l655_65582


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sphere_volume_constant_l655_65512

/-- The surface area of a cube with side length 3 -/
noncomputable def cube_surface_area : ℝ := 54

/-- The surface area of a sphere with the same surface area as the cube -/
noncomputable def sphere_surface_area : ℝ := cube_surface_area

/-- The volume of the sphere expressed in terms of K -/
noncomputable def sphere_volume (K : ℝ) : ℝ := (K * Real.sqrt 6) / Real.sqrt Real.pi

/-- Theorem stating that K = 36 for a sphere with the same surface area as a cube with side length 3 -/
theorem sphere_volume_constant : ∃ K : ℝ, 
  sphere_surface_area = 4 * Real.pi * (((3 * Real.sqrt 6) / (2 * Real.sqrt Real.pi)) ^ 2) ∧ 
  sphere_volume K = (4 / 3) * Real.pi * ((3 * Real.sqrt 6) / (2 * Real.sqrt Real.pi)) ^ 3 ∧ 
  K = 36 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sphere_volume_constant_l655_65512


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_valid_selection_l655_65530

/-- Represents a chessboard configuration -/
structure ChessBoard where
  pieces : Finset (Fin 8 × Fin 8)
  row_count : ∀ r, (pieces.filter (λ p ↦ p.1 = r)).card = 4
  col_count : ∀ c, (pieces.filter (λ p ↦ p.2 = c)).card = 4

/-- Represents a valid selection of pieces -/
def ValidSelection (board : ChessBoard) (selection : Finset (Fin 8 × Fin 8)) : Prop :=
  selection ⊆ board.pieces ∧
  selection.card = 8 ∧
  ∀ p q, p ∈ selection → q ∈ selection → p ≠ q → p.1 ≠ q.1 ∧ p.2 ≠ q.2

/-- Theorem: There exists a valid selection for any chessboard configuration -/
theorem exists_valid_selection (board : ChessBoard) : 
  ∃ selection, ValidSelection board selection := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_valid_selection_l655_65530


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_period_of_sin_minus_cos_l655_65562

theorem period_of_sin_minus_cos :
  ∃ (p : ℝ), p > 0 ∧ ∀ (x : ℝ), Real.sin x - Real.cos x = Real.sin (x + p) - Real.cos (x + p) ∧ p = 2 * Real.pi :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_period_of_sin_minus_cos_l655_65562


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fiftieth_term_of_sequence_l655_65586

noncomputable def my_sequence (n : ℕ) : ℝ := Real.sqrt (3 * (n - 1))

theorem fiftieth_term_of_sequence :
  my_sequence 50 = 7 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fiftieth_term_of_sequence_l655_65586


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_at_zero_l655_65570

-- Define the function f(x)
noncomputable def f (x : ℝ) : ℝ := x * Real.exp x + 2

-- Define the derivative of f(x)
noncomputable def f_derivative (x : ℝ) : ℝ := (x + 1) * Real.exp x

-- Theorem statement
theorem tangent_line_at_zero (x y : ℝ) :
  (y - f 0 = f_derivative 0 * (x - 0)) ↔ y = x + 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_at_zero_l655_65570


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_membership_change_l655_65524

-- Define the percentage changes
def fall_increase : ℝ := 0.08
def spring_decrease : ℝ := 0.19

-- Theorem statement
theorem membership_change :
  ∀ (initial : ℝ), initial > 0 →
    (initial * (1 + fall_increase) * (1 - spring_decrease) - initial) / initial * 100 = -12.52 :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_membership_change_l655_65524


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_fifth_term_l655_65571

/-- Represents an arithmetic sequence -/
structure ArithmeticSequence where
  a : ℕ → ℚ  -- The sequence
  d : ℚ      -- Common difference

/-- Sum of first n terms of an arithmetic sequence -/
def sum (seq : ArithmeticSequence) (n : ℕ) : ℚ :=
  n * (seq.a 1 + seq.a n) / 2

theorem arithmetic_sequence_fifth_term 
  (seq : ArithmeticSequence) 
  (h1 : sum seq 5 = 2 * sum seq 4)
  (h2 : seq.a 2 + seq.a 4 = 8) :
  seq.a 5 = 10 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_fifth_term_l655_65571


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_find_phi_l655_65577

noncomputable def f (x φ : ℝ) : ℝ := Real.sin (2 * x + φ)

noncomputable def g (x φ : ℝ) : ℝ := Real.sin (2 * x + Real.pi/4 + φ)

theorem find_phi :
  ∀ φ : ℝ, -Real.pi/2 < φ ∧ φ < Real.pi/2 →
  (∀ x : ℝ, g x φ = g (-x) φ) →
  φ = Real.pi/4 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_find_phi_l655_65577


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l655_65599

/-- Triangle ABC with specific properties -/
structure TriangleABC where
  -- Angles and sides
  A : ℝ
  B : ℝ
  C : ℝ
  a : ℝ
  b : ℝ
  c : ℝ
  -- Given conditions
  angle_order : A > B
  cos_C : Real.cos C = 5 / 13
  cos_A_minus_B : Real.cos (A - B) = 3 / 5
  side_c : c = 15

theorem triangle_properties (t : TriangleABC) : 
  Real.cos (2 * t.A) = -63 / 65 ∧ t.a = 2 * Real.sqrt 65 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l655_65599


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_omega_values_l655_65593

noncomputable def f (ω : ℝ) (x : ℝ) : ℝ := Real.sin (ω * x)

theorem omega_values (ω : ℝ) :
  (∀ x ∈ Set.Icc 0 π, Monotone (fun y ↦ f ω y)) →
  (∀ x : ℝ, f ω (4 * π - x) = f ω (4 * π + x)) →
  ω = 1/4 ∨ ω = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_omega_values_l655_65593


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_stock_price_average_exists_l655_65563

theorem stock_price_average_exists : ∃ (prices : Fin 14 → ℝ),
  prices 0 = 5 ∧ prices 6 = 5.14 ∧ prices 13 = 5 ∧
  5.09 < (Finset.sum Finset.univ prices) / 14 ∧ (Finset.sum Finset.univ prices) / 14 < 5.10 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_stock_price_average_exists_l655_65563


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_side_length_l655_65544

theorem triangle_side_length 
  (A B C : ℝ) 
  (a b c : ℝ) 
  (h1 : Real.cos C * Real.sin (π/6 - C) = -1/4)
  (h2 : a = 2)
  (h3 : 1/2 * a * b * Real.sin C = 2 * Real.sqrt 3) : 
  c = 2 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_side_length_l655_65544


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_centroid_vector_sum_l655_65514

/-- Given a triangle ABC with centroid M and an arbitrary point O, 
    prove that OM = (1/3) * (OA + OB + OC) -/
theorem centroid_vector_sum (A B C M O : EuclideanSpace ℝ (Fin 3)) 
  (h_centroid : M = (1/3 : ℝ) • (A + B + C)) : 
  (M - O) = (1/3 : ℝ) • ((A - O) + (B - O) + (C - O)) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_centroid_vector_sum_l655_65514


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_polygon_diagonal_inequality_l655_65576

-- Define the necessary structures and functions
structure Polygon where
  vertices : List (ℝ × ℝ)
  convex : Bool

def sum_of_diagonal_lengths (p : Polygon) : ℝ := sorry
def perimeter (p : Polygon) : ℝ := sorry

theorem polygon_diagonal_inequality (n : ℕ) (d p : ℝ) (polygon : Polygon)
  (h1 : n > 3) 
  (h2 : d > 0) 
  (h3 : p > 0) 
  (h4 : polygon.convex = true) 
  (h5 : d = sum_of_diagonal_lengths polygon) 
  (h6 : p = perimeter polygon) :
  (n : ℝ) - 3 < 2 * d / p ∧ 2 * d / p < ⌊(n : ℝ) / 2⌋ * ⌊((n : ℝ) + 1) / 2⌋ - 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_polygon_diagonal_inequality_l655_65576


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rams_location_on_map_l655_65553

/-- Represents the scale of a map --/
structure MapScale where
  inches_on_map : ℚ
  km_in_reality : ℚ

/-- Calculates the distance on a map given the actual distance and the map scale --/
def distance_on_map (actual_distance : ℚ) (scale : MapScale) : ℚ :=
  actual_distance * scale.inches_on_map / scale.km_in_reality

theorem rams_location_on_map 
  (map_distance : ℚ)
  (actual_distance : ℚ)
  (rams_actual_distance : ℚ)
  (h1 : map_distance = 312)
  (h2 : actual_distance = 136)
  (h3 : rams_actual_distance = 10897435897435898 / 1000000000000000) :
  ∃ ε > 0, |distance_on_map rams_actual_distance ⟨map_distance, actual_distance⟩ - 25| < ε := by
  sorry

#eval distance_on_map (10897435897435898 / 1000000000000000) ⟨312, 136⟩

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rams_location_on_map_l655_65553


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_focus_distance_sum_of_distances_to_foci_l655_65522

-- Define the ellipse
def is_on_ellipse (x y : ℝ) : Prop := x^2 / 25 + y^2 / 16 = 1

-- Define the distance from a point to a focus
noncomputable def distance_to_focus (x y fx fy : ℝ) : ℝ := Real.sqrt ((x - fx)^2 + (y - fy)^2)

-- Theorem statement
theorem ellipse_focus_distance 
  (x y fx1 fy1 fx2 fy2 : ℝ) 
  (h_on_ellipse : is_on_ellipse x y) 
  (h_focus1 : distance_to_focus x y fx1 fy1 = 3) :
  distance_to_focus x y fx2 fy2 = 7 :=
by
  sorry

-- Define the semi-major axis
def a : ℝ := 5

-- Theorem for the sum of distances to foci
theorem sum_of_distances_to_foci
  (x y fx1 fy1 fx2 fy2 : ℝ)
  (h_on_ellipse : is_on_ellipse x y) :
  distance_to_focus x y fx1 fy1 + distance_to_focus x y fx2 fy2 = 2 * a :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_focus_distance_sum_of_distances_to_foci_l655_65522


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_distances_between_20_and_21_l655_65540

noncomputable def point_A : ℝ × ℝ := (15, 3)
noncomputable def point_B : ℝ × ℝ := (0, 0)
noncomputable def point_D : ℝ × ℝ := (6, 8)

noncomputable def distance (p q : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

theorem sum_of_distances_between_20_and_21 :
  20 < distance point_A point_D + distance point_B point_D ∧
  distance point_A point_D + distance point_B point_D < 21 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_distances_between_20_and_21_l655_65540


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solve_equation_l655_65532

-- Define the complex number i
noncomputable def i : ℂ := Complex.I

-- Define the equation
noncomputable def equation (x : ℂ) (a : ℝ) : Prop := x^2 + (4 + i) * x + 4 + a * i = 0

-- Define z
noncomputable def z (a b : ℝ) : ℂ := a + b * i

-- Theorem statement
theorem solve_equation : 
  ∃ (a : ℝ) (b : ℝ), equation b a ∧ z a b = 2 - 2 * i := by
  -- Introduce the values of a and b
  let a := 2
  let b := -2
  
  -- Prove existence
  use a, b
  
  -- Split the conjunction
  apply And.intro
  
  -- Prove the equation holds
  · sorry
  
  -- Prove z equals 2 - 2i
  · sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_solve_equation_l655_65532


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_curve_line_intersection_distance_l655_65535

/-- The curve C in the rectangular coordinate system -/
def curve_C (a : ℝ) (x y : ℝ) : Prop :=
  (x - a)^2 + (y - a)^2 = 2 * a^2

/-- The line l in the rectangular coordinate system -/
def line_l (x y : ℝ) : Prop :=
  y = Real.sqrt 3 * (x + 1)

/-- Point P in the rectangular coordinate system -/
def point_P : ℝ × ℝ := (-1, 0)

/-- The sum of distances from P to the intersection points of C and l -/
def sum_distances : ℝ := 5

theorem curve_line_intersection_distance (a : ℝ) :
  (∃ (M N : ℝ × ℝ), 
    curve_C a M.1 M.2 ∧ 
    curve_C a N.1 N.2 ∧ 
    line_l M.1 M.2 ∧ 
    line_l N.1 N.2 ∧
    sum_distances = dist point_P M + dist point_P N) →
  a = 2 * Real.sqrt 3 - 2 :=
by
  sorry

#check curve_line_intersection_distance

end NUMINAMATH_CALUDE_ERRORFEEDBACK_curve_line_intersection_distance_l655_65535


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_general_term_formula_l655_65518

/-- The sequence (a_n) defined by the given recurrence relation -/
def a : ℕ → ℕ
  | 0 => 0  -- Add this case to cover Nat.zero
  | 1 => 0
  | n + 1 => a n + n

/-- Theorem stating that the general term of the sequence is n(n-1)/2 -/
theorem general_term_formula (n : ℕ) (h : n ≥ 1) : a n = n * (n - 1) / 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_general_term_formula_l655_65518


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_equation_and_chord_line_l655_65592

/-- Represents a parabola with parameter p > 0 -/
structure Parabola where
  p : ℝ
  h_p_pos : p > 0

/-- Represents a point on a parabola -/
structure ParabolaPoint (C : Parabola) where
  x : ℝ
  y : ℝ
  h_on_parabola : y^2 = 2 * C.p * x

/-- The directrix of a parabola -/
noncomputable def directrix (C : Parabola) : ℝ := -C.p / 2

theorem parabola_equation_and_chord_line 
  (C : Parabola) 
  (A : ParabolaPoint C)
  (h_A_x : A.x = 3)
  (h_A_dist : A.x - directrix C = 5) :
  (∃ (C' : Parabola), ∀ (x y : ℝ), y^2 = 2 * C'.p * x ↔ y^2 = 8 * x) ∧
  (∃ (m b : ℝ), m = 2 ∧ b = -4 ∧ ∀ (x y : ℝ), y = m * x + b ↔ y - 2*x + 4 = 0) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_equation_and_chord_line_l655_65592


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersecting_circles_m_value_l655_65547

/-- Two circles C₁ and C₂ with the following properties:
    1. They intersect at two points
    2. One intersection point has coordinates (9, 6)
    3. The product of their radii is 68
    4. The x-axis is tangent to both circles
    5. A line y = mx (m > 0) is tangent to both circles -/
structure IntersectingCircles where
  C₁ : Set (ℝ × ℝ)
  C₂ : Set (ℝ × ℝ)
  intersect_point : (9, 6) ∈ C₁ ∩ C₂
  radii_product : ∃ r₁ r₂ : ℝ, r₁ * r₂ = 68 ∧ 
    (∀ (x y : ℝ), (x, y) ∈ C₁ → (x - 9)^2 + (y - 6)^2 = r₁^2) ∧
    (∀ (x y : ℝ), (x, y) ∈ C₂ → (x - 9)^2 + (y - 6)^2 = r₂^2)
  x_axis_tangent : ∀ (x : ℝ), (x, 0) ∉ C₁ ∧ (x, 0) ∉ C₂
  m_line_tangent : ∃ (m : ℝ), m > 0 ∧ ∀ (x : ℝ), (x, m * x) ∉ C₁ ∧ (x, m * x) ∉ C₂

/-- The theorem stating that m = (12 * √221) / 49 for the given conditions -/
theorem intersecting_circles_m_value (ic : IntersectingCircles) : 
  ∃ (m : ℝ), m > 0 ∧ (∀ (x : ℝ), (x, m * x) ∉ ic.C₁ ∧ (x, m * x) ∉ ic.C₂) ∧ m = (12 * Real.sqrt 221) / 49 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersecting_circles_m_value_l655_65547


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_johns_computers_l655_65528

theorem johns_computers (unfixable_percent : ℝ) (wait_for_parts_percent : ℝ) (fixed_right_away : ℕ) : 
  unfixable_percent = 0.2 →
  wait_for_parts_percent = 0.4 →
  fixed_right_away = 8 →
  ∃ (total : ℕ), total = 20 ∧ (fixed_right_away : ℝ) / total = 1 - (unfixable_percent + wait_for_parts_percent) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_johns_computers_l655_65528


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_last_bead_color_l655_65546

/-- Represents the colors of beads -/
inductive Color
  | Red
  | Orange
  | Yellow
  | Green
  | Blue
  deriving Repr

/-- The pattern of beads -/
def pattern : List Color := [
  Color.Red, Color.Orange, Color.Yellow, Color.Yellow, Color.Green, Color.Blue
]

/-- The total number of beads in the necklace -/
def total_beads : Nat := 81

/-- Returns the color of the nth bead in the necklace -/
def necklace_color (n : Nat) : Color :=
  pattern[((n - 1) % pattern.length)]'(by
    have h : ∀ k, k % (pattern.length) < pattern.length := by
      intro k
      exact Nat.mod_lt k (Nat.zero_lt_succ _)
    exact h (n - 1)
  )

theorem last_bead_color :
  necklace_color total_beads = Color.Yellow := by
  -- Proof goes here
  sorry

#eval necklace_color total_beads

end NUMINAMATH_CALUDE_ERRORFEEDBACK_last_bead_color_l655_65546


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l655_65590

/-- The function f(x) as defined in the problem -/
noncomputable def f (ω : ℝ) (x : ℝ) : ℝ := Real.sqrt 3 * Real.sin (ω * x) * Real.cos (ω * x) - Real.sin (ω * x) ^ 2 + 1

/-- The theorem stating the main results of the problem -/
theorem problem_solution (ω : ℝ) (h_ω_pos : ω > 0) 
  (h_symmetric_axes : ∀ x : ℝ, f ω (x + π / (2 * ω)) = f ω x) :
  (ω = 1) ∧ 
  (∃ (A B C : ℝ) (hA : f ω A = 1) (ha : Real.sqrt 3 = 2 * (Real.sin C) * (Real.sin B)),
    3 * Real.sqrt 3 / 4 = Real.sqrt 3 * Real.sin B * Real.sin C / 2 ∧
    ∀ (A' B' C' : ℝ) (hA' : f ω A' = 1) (ha' : Real.sqrt 3 = 2 * (Real.sin C') * (Real.sin B')),
      3 * Real.sqrt 3 / 4 ≥ Real.sqrt 3 * Real.sin B' * Real.sin C' / 2) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l655_65590


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_ratio_is_three_fifths_l655_65580

noncomputable section

structure Triangle (X Y Z : ℝ × ℝ) where
  xy_length : Real.sqrt ((X.1 - Y.1)^2 + (X.2 - Y.2)^2) = 15
  xz_length : Real.sqrt ((X.1 - Z.1)^2 + (X.2 - Z.2)^2) = 25
  yz_length : Real.sqrt ((Y.1 - Z.1)^2 + (Y.2 - Z.2)^2) = 34

def is_angle_bisector (X Y Z T : ℝ × ℝ) : Prop :=
  ∃ (k : ℝ), k > 0 ∧ T.1 = X.1 + k * (Y.1 - X.1) ∧ T.2 = X.2 + k * (Y.2 - X.2)

noncomputable def area_ratio (X Y Z T : ℝ × ℝ) : ℝ :=
  (X.1 * Y.2 - X.2 * Y.1 + Y.1 * T.2 - Y.2 * T.1 + T.1 * X.2 - T.2 * X.1) /
  (X.1 * Z.2 - X.2 * Z.1 + Z.1 * T.2 - Z.2 * T.1 + T.1 * X.2 - T.2 * X.1)

theorem area_ratio_is_three_fifths (X Y Z T : ℝ × ℝ) 
  (tri : Triangle X Y Z) (bisector : is_angle_bisector X Y Z T) : 
  area_ratio X Y Z T = 3/5 := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_ratio_is_three_fifths_l655_65580


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_blackboard_operation_bound_max_operations_l655_65581

theorem blackboard_operation_bound (b : ℕ) (h1 : b < 2000) : 
  ∀ n : ℕ, (2000 - b : ℕ) / 2^n ≥ 1 → n ≤ 10 :=
by sorry

def operation_sequence (b : ℕ) (h1 : b < 2000) : ℕ → ℕ × ℕ
| 0 => (2000, b)
| n + 1 => 
    let (a, b) := operation_sequence b h1 n
    if a > b then (a, (a + b) / 2) else ((a + b) / 2, b)

theorem max_operations (b : ℕ) (h1 : b < 2000) :
  ∃ n : ℕ, n ≤ 10 ∧ 
    (let (a, b) := operation_sequence b h1 n
     (a + b) % 2 ≠ 0 ∨ a = b) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_blackboard_operation_bound_max_operations_l655_65581


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_angle_45_degrees_l655_65555

noncomputable def curve (x : ℝ) : ℝ := (1/2) * x^2 - 2

theorem tangent_angle_45_degrees :
  let point : ℝ × ℝ := (1, -3/2)
  let tangent_slope : ℝ := (deriv curve) point.1
  Real.arctan tangent_slope = Real.pi / 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_angle_45_degrees_l655_65555


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_etienne_money_greater_by_4_percent_l655_65550

/-- The exchange rate from euros to dollars -/
noncomputable def euro_to_dollar : ℚ := 13/10

/-- Diana's money in dollars -/
def diana_money : ℚ := 500

/-- Etienne's money in euros -/
def etienne_money : ℚ := 400

/-- Calculate the percentage difference between two values -/
noncomputable def percentage_difference (v1 v2 : ℚ) : ℚ :=
  (v1 - v2) / v2 * 100

/-- Theorem: The value of Etienne's money is 4% greater than Diana's -/
theorem etienne_money_greater_by_4_percent :
  percentage_difference (etienne_money * euro_to_dollar) diana_money = 4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_etienne_money_greater_by_4_percent_l655_65550


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_lower_bound_f_3_upper_bound_l655_65527

-- Define the function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := |x + 1/a| + |x - a|

-- Theorem 1: For all real x and positive real a, f(x) ≥ 2
theorem f_lower_bound (a : ℝ) (h : a > 0) : ∀ x : ℝ, f a x ≥ 2 := by
  sorry

-- Theorem 2: If f(3) ≤ 5, then (1 + √5)/2 ≤ a ≤ (5 + √21)/2
theorem f_3_upper_bound (a : ℝ) (h1 : a > 0) (h2 : f a 3 ≤ 5) :
  (1 + Real.sqrt 5) / 2 ≤ a ∧ a ≤ (5 + Real.sqrt 21) / 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_lower_bound_f_3_upper_bound_l655_65527


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_inequality_range_l655_65515

theorem quadratic_inequality_range :
  {k : ℝ | ∀ x : ℝ, k * x^2 - k * x + 1 > 0} = Set.Icc 0 4 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_inequality_range_l655_65515


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l655_65502

noncomputable def f (x : ℝ) : ℝ := 1 / (x^2 - 3*x + 2)

theorem domain_of_f :
  {x : ℝ | ∃ y, f x = y} = {x : ℝ | x < 1 ∨ (1 < x ∧ x < 2) ∨ 2 < x} :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l655_65502


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_A_work_days_l655_65565

/-- The number of days it takes for A to complete the work alone -/
noncomputable def days_for_A : ℝ := 5

/-- B's work rate per day -/
noncomputable def B_rate : ℝ := 1 / 10

/-- The number of days it takes for A, B, and C to complete the work together -/
noncomputable def days_together : ℝ := 2

/-- C's share of the total payment -/
noncomputable def C_share : ℝ := 200 / 500

theorem A_work_days :
  days_for_A = 5 ∧
  B_rate = 1 / 10 ∧
  days_together = 2 ∧
  C_share = 200 / 500 →
  1 / days_for_A + B_rate + C_share * (1 / days_together) = 1 / days_together :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_A_work_days_l655_65565


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_chord_length_chord_equation_midpoint_trajectory_l655_65529

-- Define the ellipse
def ellipse (x y : ℝ) : Prop := 5 * x^2 + 9 * y^2 = 45

-- Define the right focus point
def F : ℝ × ℝ := (2, 0)

-- Theorem 1: Length of chord
theorem chord_length : 
  ∃ A B : ℝ × ℝ, 
    ellipse A.1 A.2 ∧ 
    ellipse B.1 B.2 ∧ 
    A.2 = A.1 - 2 ∧ 
    B.2 = B.1 - 2 ∧ 
    ((A.1 - B.1)^2 + (A.2 - B.2)^2 : ℝ) = (30/7)^2 := by sorry

-- Theorem 2: Equation of line with midpoint (1,1)
theorem chord_equation :
  ∃ A B : ℝ × ℝ,
    ellipse A.1 A.2 ∧
    ellipse B.1 B.2 ∧
    (A.1 + B.1) / 2 = 1 ∧
    (A.2 + B.2) / 2 = 1 ∧
    5 * A.1 + 9 * A.2 = 14 ∧
    5 * B.1 + 9 * B.2 = 14 := by sorry

-- Theorem 3: Trajectory equation of midpoint
theorem midpoint_trajectory :
  ∀ A B : ℝ × ℝ,
    ellipse A.1 A.2 →
    ellipse B.1 B.2 →
    (B.2 - A.2) * F.1 = (B.1 - A.1) * F.2 →
    let P := ((A.1 + B.1) / 2, (A.2 + B.2) / 2)
    5 * P.1^2 + 9 * P.2^2 - 10 * P.1 = 0 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_chord_length_chord_equation_midpoint_trajectory_l655_65529


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_oliver_final_distance_l655_65523

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ
deriving Inhabited

/-- Calculates the distance between two points -/
noncomputable def distance (p1 p2 : Point) : ℝ :=
  Real.sqrt ((p1.x - p2.x)^2 + (p1.y - p2.y)^2)

/-- Oliver's cycling route -/
noncomputable def oliverRoute : List Point := [
  ⟨0, 0⟩,  -- Starting point
  ⟨0, 3⟩,  -- After cycling north for 3 miles
  ⟨1, 3 + Real.sqrt 3⟩,  -- After cycling northeast
  ⟨1 + Real.sqrt 3, 2 + Real.sqrt 3⟩  -- After cycling southeast
]

theorem oliver_final_distance :
  distance (oliverRoute.getLast!) (oliverRoute.head!) = Real.sqrt (11 + 6 * Real.sqrt 3) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_oliver_final_distance_l655_65523


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_monotonic_increase_intervals_l655_65519

noncomputable def f (x : ℝ) := Real.cos x ^ 2 + (Real.sqrt 3 / 2) * Real.sin (2 * x)

theorem monotonic_increase_intervals (k : ℤ) :
  StrictMonoOn f (Set.Icc (k * Real.pi - Real.pi / 3) (k * Real.pi + Real.pi / 6)) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_monotonic_increase_intervals_l655_65519


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_dave_wins_l655_65505

/-- Represents a player in the die-tossing game -/
inductive Player : Type
| Alice : Player
| Bob : Player
| Carol : Player
| Dave : Player

/-- The probability of tossing a six -/
def prob_six : ℚ := 1 / 6

/-- The probability of not tossing a six -/
def prob_not_six : ℚ := 1 - prob_six

/-- The probability of Dave winning in the first cycle -/
def prob_dave_first_cycle : ℚ := prob_not_six^3 * prob_six

/-- The probability of the game continuing after one full cycle -/
def prob_continue : ℚ := prob_not_six^4

/-- The theorem stating the probability of Dave being the first to toss a six -/
theorem prob_dave_wins : 
  prob_dave_first_cycle / (1 - prob_continue) = 125 / 671 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_dave_wins_l655_65505


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_complement_intersection_range_of_m_l655_65584

-- Define the sets and function
noncomputable def A : Set ℝ := {x | 1 < x ∧ x < 3}
noncomputable def f (x : ℝ) : ℝ := 1 / Real.sqrt (5 - x) + Real.log x
noncomputable def B : Set ℝ := {x | 0 < x ∧ x < 5}
noncomputable def C (m : ℝ) : Set ℝ := {x | 2*m - 1 < x ∧ x < m}

-- Theorem statements
theorem domain_of_f : Set.range f = B := by sorry

theorem complement_intersection :
  (Set.compl A ∩ B) = {x : ℝ | (0 < x ∧ x ≤ 1) ∨ (3 ≤ x ∧ x < 5)} := by sorry

theorem range_of_m (m : ℝ) :
  (A ∩ C m = C m) → m ≥ 1 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_complement_intersection_range_of_m_l655_65584


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_equation_l655_65574

-- Define a point in 2D space
def Point := ℝ × ℝ

-- Define a line in 2D space as a function ax + by + c = 0
def Line := ℝ → ℝ → ℝ → Prop

-- Define parallel lines
def parallel (l1 l2 : Line) : Prop :=
  ∃ (k : ℝ), k ≠ 0 ∧ ∀ x y c, l1 x y c ↔ l2 x y (k * c)

-- Define a point on a line
def point_on_line (p : Point) (l : Line) : Prop :=
  let (x, y) := p
  l x y 0

-- The given parallel line
def given_line : Line :=
  λ x y c ↦ 2 * x - y + c = 0

-- The point P
def P : Point := (2, 1)

-- Theorem statement
theorem line_equation : 
  ∃ (l : Line), 
    point_on_line P l ∧ 
    parallel l given_line ∧ 
    ∀ x y, l x y 0 ↔ 2 * x - y - 3 = 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_equation_l655_65574


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_n_equals_3_pow_n_minus_1_l655_65557

/-- The odd part of a positive integer -/
def oddPart : ℕ → ℕ
  | 0 => 1
  | n + 1 => if (n + 1) % 2 = 0 then oddPart ((n + 1) / 2) else n + 1

/-- The sequence a_k -/
def a (n : ℕ) : ℕ → ℕ
  | 0 => 2^n - 1
  | k + 1 => oddPart (3 * a n k + 1)

theorem a_n_equals_3_pow_n_minus_1 (n : ℕ) (h : Odd n) :
  a n n = 3^n - 1 := by
  sorry

#eval a 5 5  -- This will evaluate a_5 for testing purposes

end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_n_equals_3_pow_n_minus_1_l655_65557


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_alpha_plus_pi_half_l655_65589

/-- If the terminal side of angle α passes through point P(-1,2), 
    then sin(α + π/2) = -√5/5 -/
theorem sin_alpha_plus_pi_half (α : ℝ) (h : ∃ (t : ℝ), t > 0 ∧ t * (Real.cos α) = -1 ∧ t * (Real.sin α) = 2) : 
  Real.sin (α + π/2) = -Real.sqrt 5 / 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_alpha_plus_pi_half_l655_65589


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_eccentricity_range_for_specific_hyperbola_l655_65525

/-- Represents a hyperbola in the xy-plane -/
structure Hyperbola where
  a : ℝ
  b : ℝ
  h_pos_a : 0 < a
  h_pos_b : 0 < b

/-- The eccentricity of a hyperbola -/
noncomputable def eccentricity (h : Hyperbola) : ℝ := Real.sqrt (1 + (h.b / h.a)^2)

/-- Represents a line in the xy-plane -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Theorem stating the range of eccentricity for a hyperbola 
    given specific conditions -/
theorem eccentricity_range_for_specific_hyperbola (h : Hyperbola) (l : Line) :
  (l.a = 1 ∧ l.b = 1 ∧ l.c = -1) →  -- Line equation: x + y + 1 = 0
  (∃ x₁ y₁ x₂ y₂ : ℝ, 
    (x₁ + y₁ + 1 = 0 ∧ y₁ = (h.b / h.a) * x₁ ∧ x₁ < 0) ∧
    (x₂ + y₂ + 1 = 0 ∧ y₂ = -(h.b / h.a) * x₂ ∧ x₂ < 0)) →
  1 < eccentricity h ∧ eccentricity h < Real.sqrt 2 := by
  sorry

#check eccentricity_range_for_specific_hyperbola

end NUMINAMATH_CALUDE_ERRORFEEDBACK_eccentricity_range_for_specific_hyperbola_l655_65525


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pentagon_perimeter_l655_65578

-- Define the pentagon
structure Pentagon where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ
  D : ℝ × ℝ
  E : ℝ × ℝ

-- Define the distance function
noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

-- Define the perimeter function
noncomputable def perimeter (p : Pentagon) : ℝ :=
  distance p.A p.B + distance p.B p.C + distance p.C p.D + distance p.D p.E + distance p.E p.A

-- Theorem statement
theorem pentagon_perimeter (p : Pentagon) 
  (h1 : distance p.A p.B = 1)
  (h2 : distance p.B p.C = Real.sqrt 2)
  (h3 : distance p.C p.D = Real.sqrt 3)
  (h4 : distance p.D p.E = 2) :
  perimeter p = 3 + Real.sqrt 2 + Real.sqrt 3 + Real.sqrt 10 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_pentagon_perimeter_l655_65578


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_cos_equals_two_plus_cos_squared_l655_65597

-- Define the function f
def f : ℝ → ℝ := sorry

-- State the theorem
theorem f_cos_equals_two_plus_cos_squared (x : ℝ) :
  (∀ y, f (Real.sin y) = 2 - (Real.cos y)^2) →
  f (Real.cos x) = 2 + (Real.cos x)^2 :=
by
  intro h
  -- The proof would go here, but we'll use sorry for now
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_cos_equals_two_plus_cos_squared_l655_65597


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_derivative_l655_65513

noncomputable def f (x : ℝ) : ℝ := (1/2) * Real.sqrt (1 + 4 * Real.sin x)

-- State the theorem
theorem f_derivative (x : ℝ) : 
  deriv f x = Real.cos x / Real.sqrt (1 + 4 * Real.sin x) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_derivative_l655_65513


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_power_is_three_l655_65598

-- Define the rotation matrix
noncomputable def rotation_matrix : Matrix (Fin 2) (Fin 2) ℝ :=
  ![![1/2, -Real.sqrt 3/2],
    ![Real.sqrt 3/2, 1/2]]

-- Define the identity matrix
def identity_matrix : Matrix (Fin 2) (Fin 2) ℝ :=
  ![![1, 0],
    ![0, 1]]

-- Statement to prove
theorem smallest_power_is_three :
  (∀ k : ℕ, k > 0 → k < 3 → rotation_matrix ^ k ≠ identity_matrix) ∧
  rotation_matrix ^ 3 = identity_matrix := by
  sorry

#check smallest_power_is_three

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_power_is_three_l655_65598


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_omega_value_l655_65566

-- Define the function f
noncomputable def f (ω : ℝ) (x : ℝ) : ℝ := Real.sin (ω * x) + Real.sqrt 3 * Real.cos (ω * x)

-- Define what it means for a point to be an extreme point
def IsExtremePoint (f : ℝ → ℝ) (x : ℝ) : Prop :=
  ∃ ε > 0, ∀ y ∈ Set.Ioo (x - ε) (x + ε), f y ≤ f x ∨ f y ≥ f x

-- State the theorem
theorem omega_value (ω : ℝ) :
  ω > 0 ∧ 
  (∃ (x₁ x₂ : ℝ), π/6 < x₁ ∧ x₁ < x₂ ∧ x₂ < π/2 ∧ 
    (∀ x ∈ Set.Icc (π/6) (π/2), 
      IsExtremePoint (f ω) x ↔ (x = x₁ ∨ x = x₂))) ∧
  f ω (π/6) + f ω (π/2) = 0 →
  ω = 8 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_omega_value_l655_65566


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_k_value_l655_65537

-- Define the line and circle
def line_equation (m n x y : ℝ) : Prop := (m + 1) * x + (n + 1/2) * y = (6 + Real.sqrt 6) / 2

def circle_equation (x y : ℝ) : Prop := (x - 3)^2 + (y - Real.sqrt 6)^2 = 5

-- Define the tangency condition
def is_tangent (m n : ℝ) : Prop := ∃ x y, line_equation m n x y ∧ circle_equation x y

-- Define the inequality condition
def inequality_holds (k : ℕ) : Prop := ∀ m n : ℝ, m > 0 → n > 0 → 2 * m + n ≥ k

-- The main theorem
theorem max_k_value :
  ∀ m n : ℝ, m > 0 → n > 0 → is_tangent m n →
  ∃ k : ℕ, k > 0 ∧ inequality_holds k ∧ ∀ k' : ℕ, k' > k → ¬ inequality_holds k' :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_k_value_l655_65537


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_focal_property_l655_65591

-- Define the hyperbola E
def hyperbola (x y : ℝ) : Prop := x^2 / 9 - y^2 / 16 = 1

-- Define the foci F₁ and F₂
def F₁ : ℝ × ℝ := (-5, 0)
def F₂ : ℝ × ℝ := (5, 0)

-- Define the distance between two points
noncomputable def distance (p q : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

-- State the theorem
theorem hyperbola_focal_property (P : ℝ × ℝ) :
  hyperbola P.1 P.2 → distance P F₁ = 3 → distance P F₂ = 9 := by
  sorry

#check hyperbola_focal_property

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_focal_property_l655_65591


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_minimum_length_AB_l655_65503

-- Define the circle
def circleEquation (x y : ℝ) : Prop := x^2 + y^2 = 1

-- Define a point on the circle
structure PointOnCircle where
  x : ℝ
  y : ℝ
  on_circle : circleEquation x y

-- Define the tangent line at a point on the circle
def tangent_line (p : PointOnCircle) (x y : ℝ) : Prop :=
  y - p.y = -(p.x / p.y) * (x - p.x)

-- Define the intersection points A and B
def point_A (p : PointOnCircle) : ℝ × ℝ := (2 * p.x^2, 0)
def point_B (p : PointOnCircle) : ℝ × ℝ := (0, 2 * p.y^2)

-- Define the length of AB
noncomputable def length_AB (p : PointOnCircle) : ℝ :=
  Real.sqrt ((2 * p.x^2)^2 + (2 * p.y^2)^2)

-- The main theorem
theorem minimum_length_AB :
  (∀ p : PointOnCircle, length_AB p ≥ 2) ∧
  (∃ p : PointOnCircle, length_AB p = 2) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_minimum_length_AB_l655_65503


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_false_proposition_l655_65548

-- Define log base 10
noncomputable def log10 (x : ℝ) : ℝ := Real.log x / Real.log 10

theorem false_proposition : 
  (∃ x : ℝ, log10 x = 0) ∧ 
  (∃ x : ℝ, Real.tan x = 1) ∧ 
  ¬(∀ x : ℝ, x^3 > 0) ∧ 
  (∀ x : ℝ, (2 : ℝ)^x > 0) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_false_proposition_l655_65548


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l655_65575

noncomputable section

-- Define the functions f and g
def f (a : ℝ) (x : ℝ) : ℝ := (2 - a) * (x - 1) - 2 * Real.log x
def g (a : ℝ) (x : ℝ) : ℝ := f a x + x

-- State the theorem
theorem function_properties (a : ℝ) :
  -- Part 1
  (∃ (m : ℝ), m * (1 - 0) + 2 = g a 1 ∧ m = deriv (g a) 1) →
  (∀ x ∈ Set.Ioo 0 2, deriv (g a) x < 0) ∧
  -- Part 2
  (∀ x ∈ Set.Ioo 0 (1/2), f a x ≠ 0) →
  a ≥ 2 - 4 * Real.log 2 :=
by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l655_65575


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_orange_juice_percentage_is_five_percent_l655_65516

/-- Represents the composition of a liquid mixture --/
structure Mixture where
  water : ℝ
  orange_juice : ℝ

/-- Calculates the total volume of a mixture --/
noncomputable def total_volume (m : Mixture) : ℝ := m.water + m.orange_juice

/-- Calculates the percentage of orange juice in a mixture --/
noncomputable def orange_juice_percentage (m : Mixture) : ℝ :=
  (m.orange_juice / total_volume m) * 100

/-- The initial water volume --/
def initial_water : ℝ := 3

/-- The total volume of refreshment --/
def refreshment_volume : ℝ := 1

/-- The percentage of orange juice in the refreshment --/
def refreshment_orange_juice_percentage : ℝ := 20

/-- The final mixture after combining water and refreshment --/
noncomputable def final_mixture : Mixture :=
  { water := initial_water + refreshment_volume * (1 - refreshment_orange_juice_percentage / 100),
    orange_juice := refreshment_volume * (refreshment_orange_juice_percentage / 100) }

/-- Theorem stating that the percentage of orange juice in the final mixture is 5% --/
theorem orange_juice_percentage_is_five_percent :
  orange_juice_percentage final_mixture = 5 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_orange_juice_percentage_is_five_percent_l655_65516


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_rolling_square_area_l655_65509

/-- The area covered by a circle rolling around a square -/
theorem circle_rolling_square_area 
  (circle_diameter : ℝ) 
  (square_side : ℝ) 
  (h1 : circle_diameter = 1) 
  (h2 : square_side = 2) : 
  (4 * square_side * circle_diameter) + (π * (circle_diameter / 2)^2 * 4) = 11 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_rolling_square_area_l655_65509


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_mapping_theorem_l655_65572

/-- A mapping function from the upper half-plane with a cut to the upper half-plane without a cut -/
noncomputable def mapping_function (z : ℂ) (a : ℝ) : ℂ :=
  (z^2 + a^2) ^ (1/2 : ℂ)

/-- The theorem stating that the mapping function maps the upper half-plane with a cut to the upper half-plane without a cut -/
theorem mapping_theorem (a : ℝ) (h : a > 0) :
  ∀ z : ℂ, Complex.im z > 0 →
  (∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 → z ≠ t * Complex.I * a) →
  Complex.im (mapping_function z a) > 0 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_mapping_theorem_l655_65572


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_number_of_factors_N_l655_65538

def N : ℕ := 2^4 * 3^3 * 5^2 * 7^1

theorem number_of_factors_N : (Finset.filter (· ∣ N) (Finset.range (N + 1))).card = 120 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_number_of_factors_N_l655_65538


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trail_length_proof_l655_65526

theorem trail_length_proof (total_length : ℝ) (hiked_percentage : ℝ) (remaining_length : ℝ) : 
  hiked_percentage = 0.6 →
  remaining_length = 8 →
  (1 - hiked_percentage) * total_length = remaining_length →
  total_length = 20 := by
  intros h1 h2 h3
  -- Proof steps would go here
  sorry

#check trail_length_proof

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trail_length_proof_l655_65526
