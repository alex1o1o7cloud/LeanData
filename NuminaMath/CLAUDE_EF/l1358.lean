import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_journey_average_rate_l1358_135805

/-- Calculates the average rate of a boat journey with wind influence -/
noncomputable def average_rate_with_wind (downstream_speed : ℝ) (upstream_speed : ℝ) 
  (downstream_distance : ℝ) (upstream_distance : ℝ) (wind_speed : ℝ) : ℝ :=
  let downstream_speed_with_wind := downstream_speed - wind_speed
  let upstream_speed_with_wind := upstream_speed + wind_speed
  let downstream_time := downstream_distance / downstream_speed_with_wind
  let upstream_time := upstream_distance / upstream_speed_with_wind
  let total_distance := downstream_distance + upstream_distance
  let total_time := downstream_time + upstream_time
  total_distance / total_time

/-- The average rate of the entire journey is approximately 11.51 km/h -/
theorem journey_average_rate :
  let downstream_speed := (20 : ℝ)
  let upstream_speed := (4 : ℝ)
  let downstream_distance := (60 : ℝ)
  let upstream_distance := (30 : ℝ)
  let wind_speed := (3 : ℝ)
  abs ((average_rate_with_wind downstream_speed upstream_speed downstream_distance upstream_distance wind_speed) - 11.51) < 0.01 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_journey_average_rate_l1358_135805


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_highway_speed_limit_l1358_135863

noncomputable def old_speed_limit (distance : ℝ) (new_speed : ℝ) (time_difference : ℝ) : ℝ :=
  (distance * new_speed) / (distance - new_speed * time_difference)

theorem highway_speed_limit : 
  old_speed_limit 6 35 (1/15) = 57 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_highway_speed_limit_l1358_135863


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_none_is_correct_l1358_135851

/-- Represents the possible answers to the question --/
inductive Answer
  | No
  | Nothing
  | None
  | Neither

/-- The correct answer to the question --/
def correctAnswer : Answer := Answer.None

/-- Checks if a given answer is correct --/
def isCorrectAnswer (a : Answer) : Prop :=
  a = correctAnswer

/-- Theorem stating that Answer.None is the correct answer --/
theorem none_is_correct : isCorrectAnswer Answer.None := by
  rfl

#check none_is_correct

end NUMINAMATH_CALUDE_ERRORFEEDBACK_none_is_correct_l1358_135851


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_geq_x_squared_solution_set_l1358_135850

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := 
  if x ≤ 0 then x + 2 else -x + 2

-- Define the solution set
def solution_set : Set ℝ := {x : ℝ | -1 ≤ x ∧ x ≤ 1}

-- Theorem statement
theorem f_geq_x_squared_solution_set : 
  ∀ x : ℝ, f x ≥ x^2 ↔ x ∈ solution_set :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_geq_x_squared_solution_set_l1358_135850


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_at_2_l1358_135892

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (1/3) * x^2 + a * Real.log x

def tangent_line (x y : ℝ) (b : ℝ) : Prop := x - y + b = 0

theorem tangent_line_at_2 (a : ℝ) :
  (∃ b, tangent_line 2 (f a 2) b) → a = -1/3 :=
by
  intro h
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_at_2_l1358_135892


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_omega_value_l1358_135829

-- Define the function f
noncomputable def f (ω : ℝ) (x : ℝ) : ℝ := Real.sin (ω * x - Real.pi / 6) + 1 / 2

-- State the theorem
theorem omega_value (α β ω : ℝ) :
  f ω α = -1/2 →
  f ω β = 1/2 →
  |α - β| = 3*Real.pi/4 →
  ω = 2/3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_omega_value_l1358_135829


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_challenge_solvable_l1358_135885

/-- Represents a figure that can be placed on a captive's forehead -/
structure Figure where
  id : Nat

/-- Represents a captive with a figure on their forehead -/
structure Captive where
  id : Nat
  figure : Figure

/-- The challenge setup -/
structure Challenge where
  captives : List Captive
  figures : List Figure

def Challenge.valid (c : Challenge) : Prop :=
  c.captives.length ≥ 3 ∧
  ∀ f g : Figure, f ≠ g →
    (c.captives.filter (λ captive => captive.figure.id = f.id)).length ≠
    (c.captives.filter (λ captive => captive.figure.id = g.id)).length

/-- What a captive can see (all figures except their own) -/
def Captive.canSee (c : Captive) (challenge : Challenge) : List Figure :=
  (challenge.captives.filter (λ other => other.id ≠ c.id)).map (λ other => other.figure)

/-- The strategy for a captive to guess their figure -/
def Captive.strategy (c : Captive) (challenge : Challenge) : Figure :=
  let visibleFigures := c.canSee challenge
  match List.argmax (λ f => (visibleFigures.filter (λ g => g.id = f.id)).length) visibleFigures with
  | some f => f
  | none => Figure.mk 0  -- Default to Figure 0 if list is empty

theorem challenge_solvable (c : Challenge) (h : c.valid) :
  ∃ captive : Captive, captive ∈ c.captives ∧ captive.strategy c = captive.figure := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_challenge_solvable_l1358_135885


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_at_max_dot_product_l1358_135801

/-- Given a triangle ABC with the following properties:
  - Sides a, b, c are opposite to angles A, B, C respectively
  - 2a*cos(B) = c*cos(B) + b*cos(C)
  - Vector m = (cos(A), cos(2A))
  - Vector n = (12, -5)
  - Side length a = 4
Prove that when the dot product of vectors m and n is maximized, 
the area of the triangle is (4√3 + 9)/2 -/
theorem triangle_area_at_max_dot_product (a b c A B C : ℝ) 
  (h1 : 2 * a * Real.cos B = c * Real.cos B + b * Real.cos C)
  (h2 : a = 4) : 
  ∃ (max_dot : ℝ), 
    (∀ A', (Real.cos A' * 12 + Real.cos (2 * A') * (-5)) ≤ max_dot) ∧ 
    ((Real.cos A * 12 + Real.cos (2 * A) * (-5) = max_dot) → 
      (1/2 * a * b * Real.sin C = (4 * Real.sqrt 3 + 9) / 2)) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_at_max_dot_product_l1358_135801


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_average_sum_is_30_point_5_l1358_135830

/-- Represents a die with six faces -/
structure Die :=
  (faces : Fin 6 → ℤ)

/-- Calculate the sum of all face values of a die -/
def dieSum (d : Die) : ℤ :=
  (Finset.sum Finset.univ d.faces)

/-- The first die with faces 1, 2, 3, 4, 5, 6 -/
def firstDie : Die :=
  ⟨fun i => (i.val + 1)⟩

/-- The second die with faces t-10, t, t+10, t+20, t+30, t+40 where t = 12 -/
def secondDie : Die :=
  ⟨fun i => 12 + i.val * 10 - 10⟩

/-- The average of the 36 possible sums when rolling both dice -/
def averageSum : ℚ :=
  (dieSum firstDie + dieSum secondDie) / 6

theorem average_sum_is_30_point_5 : averageSum = 30.5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_average_sum_is_30_point_5_l1358_135830


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_function_properties_l1358_135802

noncomputable section

/-- A function f(x) = ax^2 - 2x + b where a ≠ 0 and f(x) attains a minimum value of 0 at x = 1 -/
def f (a b : ℝ) (x : ℝ) : ℝ := a * x^2 - 2 * x + b

/-- g(x) = f(x) / x -/
def g (a b : ℝ) (x : ℝ) : ℝ := f a b x / x

theorem quadratic_function_properties (a b : ℝ) (ha : a ≠ 0) :
  (∀ x, f a b x ≥ 0) ∧ f a b 1 = 0 →
  a = 1 ∧ b = 1 ∧
  (∀ x ∈ Set.Icc (1/2 : ℝ) 2,
    g a b (2^x - 1) ≥ 0 ∧
    g a b (2^x - 1) ≤ 4/3 ∧
    g a b (2^1 - 1) = 0 ∧
    g a b (2^2 - 1) = 4/3) := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_function_properties_l1358_135802


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_properties_l1358_135822

-- Define the circle equation
def circle_equation (x y : ℝ) : Prop :=
  x^2 - 3*y - 12 = -y^2 + 12*x + 72

-- Define the center and radius
noncomputable def circle_center : ℝ × ℝ := (6, 3/2)
noncomputable def circle_radius : ℝ := Real.sqrt 120.25

-- Theorem statement
theorem circle_properties :
  let (a, b) := circle_center
  let r := circle_radius
  (∀ x y, circle_equation x y ↔ (x - a)^2 + (y - b)^2 = r^2) ∧
  a + b + r = 18.45 := by
  sorry

#check circle_properties

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_properties_l1358_135822


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_pi_third_minus_alpha_l1358_135826

theorem sin_pi_third_minus_alpha (α : ℝ) 
  (h1 : Real.sin (π/6 + α) = Real.sqrt 3/3) 
  (h2 : α ∈ Set.Ioo (-π/4) (π/4)) : 
  Real.sin (π/3 - α) = Real.sqrt 6/3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_pi_third_minus_alpha_l1358_135826


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_difference_proof_l1358_135843

/-- The minimum difference between b and a, where f(a) = g(b) -/
noncomputable def min_difference : ℝ :=
  1 + (1/2) * Real.log 2

theorem min_difference_proof (f g : ℝ → ℝ) (hf : ∀ x, f x = Real.exp (2 * x)) (hg : ∀ x, g x = Real.log x + 1/2) :
  ∀ a : ℝ, ∃ b : ℝ, b > 0 ∧ f a = g b ∧
  ∀ a' b' : ℝ, b' > 0 → f a' = g b' → b' - a' ≥ min_difference :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_difference_proof_l1358_135843


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complement_intersection_equals_set_l1358_135839

/-- Given a universal set U, and two sets M and N, prove that the intersection of the complement of M with N equals {4,5} -/
theorem complement_intersection_equals_set (U M N : Set ℕ) : 
  U = {1,2,3,4,5,6} → 
  M = {1,2,3} → 
  N = {3,4,5} → 
  (Mᶜ ∩ N) = {4,5} := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_complement_intersection_equals_set_l1358_135839


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_g_is_even_l1358_135825

noncomputable def g (x : ℝ) : ℝ := 5 / (3 * x^8 - 7)

theorem g_is_even : ∀ x : ℝ, g (-x) = g x := by
  intro x
  simp [g]
  ring


end NUMINAMATH_CALUDE_ERRORFEEDBACK_g_is_even_l1358_135825


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_point_P_coordinates_l1358_135811

noncomputable def A : ℝ × ℝ := (6, 0)
noncomputable def B : ℝ × ℝ := (0, 4)

def on_coordinate_axis (P : ℝ × ℝ) : Prop :=
  P.1 = 0 ∨ P.2 = 0

noncomputable def triangle_area (A B P : ℝ × ℝ) : ℝ :=
  abs ((B.1 - A.1) * (P.2 - A.2) - (P.1 - A.1) * (B.2 - A.2)) / 2

theorem point_P_coordinates :
  ∀ P : ℝ × ℝ, 
    on_coordinate_axis P → 
    triangle_area A B P = 12 → 
    P = (12, 0) ∨ P = (0, 0) ∨ P = (0, 8) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_point_P_coordinates_l1358_135811


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_on_interval_l1358_135800

def f (x : ℝ) := x^2 + 12*x + 35

theorem min_value_on_interval :
  ∃ (x : ℝ), x ∈ Set.Icc (-10) 0 ∧
  ∀ y ∈ Set.Icc (-10) 0, f x ≤ f y ∧
  f x = -1 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_on_interval_l1358_135800


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_farmer_remaining_crops_l1358_135852

/-- Calculates the number of remaining crops after pest damage -/
def remaining_crops (corn_rows : ℕ) (potato_rows : ℕ) (corn_per_row : ℕ) (potatoes_per_row : ℕ) (damage_fraction : ℚ) : ℕ :=
  let total_crops := corn_rows * corn_per_row + potato_rows * potatoes_per_row
  (((1 - damage_fraction) * total_crops).num / (1 - damage_fraction).den).natAbs

/-- Theorem stating the number of remaining crops for the given scenario -/
theorem farmer_remaining_crops :
  remaining_crops 10 5 9 30 (1/2) = 120 := by
  sorry

#eval remaining_crops 10 5 9 30 (1/2)

end NUMINAMATH_CALUDE_ERRORFEEDBACK_farmer_remaining_crops_l1358_135852


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_barium_oxide_calculations_l1358_135853

/-- Calculates the weight and molar concentration of Barium oxide -/
theorem barium_oxide_calculations 
  (atomic_mass_Ba : ℝ) 
  (atomic_mass_O : ℝ) 
  (moles_BaO : ℝ) 
  (solution_volume : ℝ) :
  atomic_mass_Ba = 137.33 →
  atomic_mass_O = 16.00 →
  moles_BaO = 5 →
  solution_volume = 3 →
  (let molar_mass_BaO := atomic_mass_Ba + atomic_mass_O
   let weight_BaO := moles_BaO * molar_mass_BaO
   let molar_concentration := moles_BaO / solution_volume
   weight_BaO = 766.65 ∧ molar_concentration = 1.67) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_barium_oxide_calculations_l1358_135853


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_property_sequence_correct_l1358_135838

def sequenceA (n : ℕ) : ℚ :=
  if n = 1 then 1
  else (1/3) * ((4/3) ^ (n - 2))

def partialSum (n : ℕ) : ℚ :=
  Finset.sum (Finset.range n) sequenceA

theorem sequence_property (n : ℕ) :
  n ≥ 1 → sequenceA (n + 1) = (1/3) * partialSum n :=
by sorry

theorem sequence_correct (n : ℕ) :
  sequenceA n = if n = 1 then 1 else (1/3) * ((4/3) ^ (n - 2)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_property_sequence_correct_l1358_135838


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_perpendicular_condition_non_monotonic_condition_existence_of_a_l1358_135882

noncomputable def f (x : ℝ) : ℝ := Real.log x
noncomputable def g (m n x : ℝ) : ℝ := m * (x + n) / (x + 1)

-- Theorem 1
theorem tangent_perpendicular_condition (n : ℝ) : 
  (((deriv f 1) * (deriv (g 1 n) 1)) = -1) ↔ n = 5 := by sorry

-- Theorem 2
theorem non_monotonic_condition (m n : ℝ) :
  ¬ Monotone (fun x => f x - g m n x) ↔ m - n > 3 := by sorry

-- Theorem 3
theorem existence_of_a :
  ∃ a : ℝ, ∀ x : ℝ, x > 0 → f (2*a/x) * f (Real.exp (a*x)) + f (x/(2*a)) ≤ 0 ↔ a = Real.sqrt 2 / 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_perpendicular_condition_non_monotonic_condition_existence_of_a_l1358_135882


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_condition_l1358_135875

/-- Two lines are parallel if their slopes are equal -/
def parallel (m1 m2 : ℝ) : Prop := m1 = m2

/-- Slope of line l₁: ax + 2y - 1 = 0 -/
noncomputable def slope_l1 (a : ℝ) : ℝ := -a / 2

/-- Slope of line l₂: x + (a+1)y + 4 = 0 -/
noncomputable def slope_l2 (a : ℝ) : ℝ := -1 / (a + 1)

/-- The condition that a = 1 is sufficient but not necessary for the lines to be parallel -/
theorem parallel_condition (a : ℝ) :
  (a = 1 → parallel (slope_l1 a) (slope_l2 a)) ∧
  ¬(parallel (slope_l1 a) (slope_l2 a) → a = 1) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_condition_l1358_135875


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_jack_to_jill_paths_l1358_135898

/-- Represents a point on the grid --/
structure Point where
  x : ℕ
  y : ℕ

/-- Calculates the number of paths between two points on a grid --/
def countPaths (start : Point) (finish : Point) (avoid : Point) : ℕ :=
  sorry

/-- The starting point (Jack's house) --/
def start : Point := ⟨0, 0⟩

/-- The ending point (Jill's house) --/
def finish : Point := ⟨4, 3⟩

/-- The point to avoid (dangerous intersection) --/
def avoid : Point := ⟨2, 1⟩

/-- The total number of blocks Jack needs to bike --/
def totalBlocks : ℕ := 7

theorem jack_to_jill_paths :
  countPaths start finish avoid = 17 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_jack_to_jill_paths_l1358_135898


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_willam_land_percentage_is_12_5_l1358_135859

/-- Represents the farm tax scenario in Mr. Willam's village -/
structure FarmTaxScenario where
  totalTax : ℚ
  willamTax : ℚ

/-- Calculates the percentage of Mr. Willam's taxable land over the total taxable land -/
def willamLandPercentage (scenario : FarmTaxScenario) : ℚ :=
  (scenario.willamTax / scenario.totalTax) * 100

/-- Theorem stating that Mr. Willam's land percentage is 12.5% given the scenario -/
theorem willam_land_percentage_is_12_5 (scenario : FarmTaxScenario) 
  (h1 : scenario.totalTax = 4000)
  (h2 : scenario.willamTax = 500) : 
  willamLandPercentage scenario = 25/2 := by
  sorry

#check willam_land_percentage_is_12_5

end NUMINAMATH_CALUDE_ERRORFEEDBACK_willam_land_percentage_is_12_5_l1358_135859


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_properties_l1358_135818

noncomputable section

-- Define the ellipse E
def ellipse (a b : ℝ) (x y : ℝ) : Prop :=
  x^2 / a^2 + y^2 / b^2 = 1

-- Define the eccentricity
def eccentricity (a c : ℝ) : ℝ :=
  c / a

theorem ellipse_properties :
  ∀ (a b : ℝ), a > b ∧ b > 0 →
  ellipse a b 1 (2 * Real.sqrt 3 / 3) →
  eccentricity a (Real.sqrt (a^2 - b^2)) = Real.sqrt 3 / 3 →
  ∃ (x₀ y₀ : ℝ),
    -- 1. Standard equation of E
    (∀ (x y : ℝ), ellipse a b x y ↔ x^2/3 + y^2/2 = 1) ∧
    -- 2. Line PQ is tangent to E
    (x₀ ≠ 0 ∧ x₀ ≠ Real.sqrt 3 ∧ x₀ ≠ -Real.sqrt 3 →
     ellipse a b x₀ y₀ →
     ∃! (x y : ℝ), 
       ellipse a b x y ∧ 
       y - y₀ = -(2*x₀)/(3*y₀) * (x - x₀)) :=
by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_properties_l1358_135818


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_no_valid_partition_l1358_135827

/-- A binary string of length 10 -/
def BinaryString := Fin 10 → Bool

/-- The set of all binary strings of length 10 -/
def AllStrings : Set BinaryString := Set.univ

/-- The Hamming distance between two binary strings -/
def hammingDistance (s1 s2 : BinaryString) : Nat :=
  (Finset.sum (Finset.range 10) fun i => if s1 i = s2 i then 0 else 1)

/-- A valid partition of binary strings -/
def validPartition (p : Set BinaryString × Set BinaryString) : Prop :=
  let (g1, g2) := p
  g1 ∪ g2 = AllStrings ∧ g1 ∩ g2 = ∅ ∧
  (∀ s1 s2, s1 ∈ g1 → s2 ∈ g1 → s1 ≠ s2 → hammingDistance s1 s2 ≥ 3) ∧
  (∀ s1 s2, s1 ∈ g2 → s2 ∈ g2 → s1 ≠ s2 → hammingDistance s1 s2 ≥ 3)

theorem no_valid_partition : ¬∃ p : Set BinaryString × Set BinaryString, validPartition p := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_no_valid_partition_l1358_135827


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_reservoir_cost_l1358_135813

/-- Represents the cost function for constructing a rectangular prism-shaped open water reservoir -/
noncomputable def reservoir_cost (x : ℝ) : ℝ :=
  150 * (4800 / 3) + 120 * 6 * (x + 4800 / (3 * x))

/-- Theorem stating the minimum cost and optimal dimensions of the reservoir -/
theorem min_reservoir_cost :
  ∃ (x : ℝ), x > 0 ∧ reservoir_cost x = 297600 ∧ 
  (∀ (y : ℝ), y > 0 → reservoir_cost y ≥ 297600) ∧
  x = 40 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_reservoir_cost_l1358_135813


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_diamonds_in_G_10_l1358_135871

/-- Sequence of geometric figures -/
def G : ℕ → ℕ
  | 0 => 1  -- Adding case for 0
  | 1 => 1
  | 2 => 6
  | 3 => 22
  | n + 4 => G (n + 3) + 4 * (n + 3) * (n + 3)

theorem diamonds_in_G_10 : G 10 = 1142 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_diamonds_in_G_10_l1358_135871


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_sum_distances_l1358_135817

-- Define the points
def A : ℝ × ℝ := (1, 2)
def B : ℝ × ℝ := (1, 5)
def C : ℝ × ℝ := (3, 6)
def D : ℝ × ℝ := (7, -1)

-- Define the distance function
noncomputable def distance (p q : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

-- Define the sum of distances function
noncomputable def sum_distances (p : ℝ × ℝ) : ℝ :=
  distance p A + distance p B + distance p C + distance p D

-- Theorem statement
theorem min_sum_distances :
  ∀ p : ℝ × ℝ, sum_distances (2, 4) ≤ sum_distances p :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_sum_distances_l1358_135817


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l1358_135896

noncomputable section

variable (ω : ℝ)

def f (x : ℝ) : ℝ := Real.sin (ω * x)
def g (x : ℝ) : ℝ := Real.sqrt 3 * Real.cos (ω * x)
def h (x : ℝ) : ℝ := 3 * (Real.cos (ω * x))^2 + Real.sqrt 3 * Real.sin (ω * x) * Real.cos (ω * x)

theorem problem_solution (hω : ω > 0) 
  (h_intersection : ∃ (x₁ x₂ : ℝ), x₁ ≠ x₂ ∧ f ω x₁ = g ω x₁ ∧ f ω x₂ = g ω x₂ ∧ x₂ - x₁ = π) :
  (ω = 1) ∧ 
  (∀ x, h ω x = Real.sqrt 3 * Real.sin (2 * x + π / 3) + 3 / 2) ∧
  (∃ m : ℝ, m ∈ Set.Icc (π / 12) (7 * π / 12) ∧ 
    ∀ x, ∃! a, a ∈ Set.Icc 0 m ∧ h ω (a - x) = h ω (a + x)) :=
by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l1358_135896


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_distances_to_focus_l1358_135868

-- Define the parabola
def parabola (x y : ℝ) : Prop := y^2 = 4*x

-- Define the line
def line (x y : ℝ) : Prop := 2*x + y - 4 = 0

-- Define the focus of the parabola
def focus : ℝ × ℝ := (0, 1)

-- Define points A and B as the intersection of the parabola and the line
axiom A : ℝ × ℝ
axiom B : ℝ × ℝ
axiom A_on_parabola : parabola A.1 A.2
axiom A_on_line : line A.1 A.2
axiom B_on_parabola : parabola B.1 B.2
axiom B_on_line : line B.1 B.2

-- State the theorem
theorem sum_of_distances_to_focus : 
  Real.sqrt ((A.1 - focus.1)^2 + (A.2 - focus.2)^2) +
  Real.sqrt ((B.1 - focus.1)^2 + (B.2 - focus.2)^2) = 7 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_distances_to_focus_l1358_135868


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_dice_probabilities_l1358_135807

noncomputable def f (a b x : ℝ) : ℝ := (1/2) * a * x^2 + b * x + 1

def is_decreasing_in_interval (a b : ℤ) : Prop :=
  b ≤ a

def has_zero_points (a b : ℤ) : Prop :=
  b^2 - 2*a ≥ 0

def probability_decreasing : ℚ :=
  21 / 36

def probability_zero_points : ℚ :=
  24 / 36

theorem dice_probabilities (a b : ℤ) (h1 : a ≥ 1 ∧ a ≤ 6) (h2 : b ≥ 1 ∧ b ≤ 6) :
  (probability_decreasing = 7/12) ∧
  (probability_zero_points = 2/3) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_dice_probabilities_l1358_135807


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_cosines_is_zero_l1358_135887

theorem sum_of_cosines_is_zero (α : ℝ) : 
  Real.cos α + Real.cos (72 * π / 180 + α) + Real.cos (144 * π / 180 + α) + 
  Real.cos (216 * π / 180 + α) + Real.cos (288 * π / 180 + α) = 0 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_cosines_is_zero_l1358_135887


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_gaussian_is_probability_distribution_negative_binomial_is_probability_distribution_l1358_135841

noncomputable def gaussian_integral (x : ℝ) : ℝ := 
  (1 / Real.sqrt (2 * Real.pi)) * Real.exp (-(x^2) / 2)

def negative_binomial_sum (p : ℝ) (k : ℕ) (n : ℕ) : ℝ :=
  (Nat.choose (n - 1) (k - 1)) * p^k * (1 - p)^(n - k)

theorem gaussian_is_probability_distribution :
  (∫ (x : ℝ), gaussian_integral x) = 1 := by sorry

theorem negative_binomial_is_probability_distribution (p : ℝ) (k : ℕ) 
  (h1 : 0 < p ∧ p ≤ 1) (h2 : k > 0) :
  (∑' (n : ℕ), if n ≥ k then negative_binomial_sum p k n else 0) = 1 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_gaussian_is_probability_distribution_negative_binomial_is_probability_distribution_l1358_135841


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_correct_means_l1358_135883

-- Define the vehicle types
inductive VehicleType
  | Car
  | Motorcycle
  | SUV

-- Define the time of day
inductive TimeOfDay
  | Morning
  | Evening

-- Define the weather condition
inductive WeatherCondition
  | Sunny
  | Rainy

-- Define a function to represent the counts
def vehicleCounts (t : TimeOfDay) (w : WeatherCondition) (v : VehicleType) : Fin 5 → ℕ :=
  sorry -- Placeholder for the actual implementation

-- Define the mean calculation function
def calculateMean (counts : Fin 5 → ℕ) : ℚ :=
  (counts 0 + counts 1 + counts 2 + counts 3 + counts 4) / 5

-- Theorem statement
theorem correct_means :
  (calculateMean (vehicleCounts TimeOfDay.Morning WeatherCondition.Sunny VehicleType.Car) = 104/5) ∧
  (calculateMean (vehicleCounts TimeOfDay.Morning WeatherCondition.Sunny VehicleType.Motorcycle) = 3) ∧
  (calculateMean (vehicleCounts TimeOfDay.Morning WeatherCondition.Sunny VehicleType.SUV) = 51/5) ∧
  (calculateMean (vehicleCounts TimeOfDay.Morning WeatherCondition.Rainy VehicleType.Car) = 138/5) ∧
  (calculateMean (vehicleCounts TimeOfDay.Morning WeatherCondition.Rainy VehicleType.Motorcycle) = 6/5) ∧
  (calculateMean (vehicleCounts TimeOfDay.Morning WeatherCondition.Rainy VehicleType.SUV) = 61/5) ∧
  (calculateMean (vehicleCounts TimeOfDay.Evening WeatherCondition.Sunny VehicleType.Car) = 21) ∧
  (calculateMean (vehicleCounts TimeOfDay.Evening WeatherCondition.Sunny VehicleType.Motorcycle) = 13/5) ∧
  (calculateMean (vehicleCounts TimeOfDay.Evening WeatherCondition.Sunny VehicleType.SUV) = 9) ∧
  (calculateMean (vehicleCounts TimeOfDay.Evening WeatherCondition.Rainy VehicleType.Car) = 132/5) ∧
  (calculateMean (vehicleCounts TimeOfDay.Evening WeatherCondition.Rainy VehicleType.Motorcycle) = 4/5) ∧
  (calculateMean (vehicleCounts TimeOfDay.Evening WeatherCondition.Rainy VehicleType.SUV) = 58/5) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_correct_means_l1358_135883


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_irrational_among_options_l1358_135819

theorem irrational_among_options : 
  (¬ (∃ (a b : ℤ), b ≠ 0 ∧ -Real.sqrt 3 = (a : ℚ) / b)) ∧ 
  (∃ (a b : ℤ), b ≠ 0 ∧ Real.sqrt 16 = (a : ℚ) / b) ∧
  (∃ (a b : ℤ), b ≠ 0 ∧ (8 : ℚ) ^ (1/3) = (a : ℚ) / b) ∧
  (∃ (a b : ℤ), b ≠ 0 ∧ (7 : ℚ) / 3 = (a : ℚ) / b) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_irrational_among_options_l1358_135819


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_even_shifted_min_period_is_pi_f_shifted_is_even_l1358_135891

noncomputable def ω : ℝ := 2

noncomputable def f (x : ℝ) : ℝ := Real.sin (ω * x) + Real.sqrt 3 * Real.cos (ω * x)

noncomputable def φ : ℝ := Real.pi / 12

theorem f_even_shifted (x : ℝ) : f (x + φ) = f (-x - φ) := by
  sorry

theorem min_period_is_pi : ∀ T > 0, (∀ x, f (x + T) = f x) → T ≥ Real.pi := by
  sorry

theorem f_shifted_is_even : ∀ x, f (x + φ) = f (-x - φ) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_even_shifted_min_period_is_pi_f_shifted_is_even_l1358_135891


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_function_inequality_l1358_135842

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := if x ≥ 0 then x^2 else -x^2

-- State the theorem
theorem odd_function_inequality (a : ℝ) :
  (∀ x, f (-x) = -f x) →  -- f is odd
  (∀ x ≥ 0, f x = x^2) →  -- f(x) = x^2 for x ≥ 0
  (∀ x ∈ Set.Icc a (a + 2), f (x + a) ≥ 2 * f x) →  -- inequality condition
  a ≥ Real.sqrt 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_function_inequality_l1358_135842


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_total_vegetable_weight_l1358_135814

/-- Represents the weight of vegetables in kilograms -/
@[reducible] def Weight := ℚ

/-- Converts grams to kilograms -/
def gramsToKg (g : ℚ) : Weight := g / 1000

/-- Weight of cabbage in one trip -/
def cabbageWeight : Weight := 4 + gramsToKg 436

/-- Weight of radish in one trip -/
def radishWeight : Weight := gramsToKg 1999

/-- Number of round trips -/
def numTrips : ℕ := 2

theorem total_vegetable_weight : 
  (numTrips : ℚ) * (cabbageWeight + radishWeight) = 12.87 := by
  -- Proof steps would go here
  sorry

#eval (numTrips : ℚ) * (cabbageWeight + radishWeight)

end NUMINAMATH_CALUDE_ERRORFEEDBACK_total_vegetable_weight_l1358_135814


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_propositions_truth_l1358_135856

-- Define the propositions
def proposition1 : Prop := ∃ a b c : ℝ, a ≠ 0 ∧ b^2 - 4*a*c ≥ 0 ∧ ∃ x : ℝ, a*x^2 + b*x + c = 0

noncomputable def proposition2 : Prop := ∫ x in (0)..(Real.sqrt Real.pi), Real.sqrt (Real.pi - x^2) = Real.pi^2 / 4

def proposition3 : Prop := ∀ a b : ℝ, (0 < b ∧ b < a) → (0 < Real.rpow b (1/3) ∧ Real.rpow b (1/3) < Real.rpow a (1/3))

def proposition4 : Prop := ∀ m : ℝ, (∀ x : ℝ, m*x^2 - 2*(m+1)*x + (m+3) > 0) → m ≥ 1

-- State the theorem
theorem propositions_truth : proposition1 ∧ proposition2 ∧ proposition3 ∧ ¬proposition4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_propositions_truth_l1358_135856


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_reciprocal_distances_l1358_135835

-- Define the curves C1 and C2
noncomputable def C1 (t : ℝ) : ℝ × ℝ := (1 + (1/2) * t, (Real.sqrt 3 / 2) * t)

noncomputable def C2 (θ : ℝ) : ℝ :=
  Real.sqrt (12 / (3 + Real.sin θ ^ 2))

-- Define point F
def F : ℝ × ℝ := (1, 0)

-- Define the intersection points A and B as variables
variable (A B : ℝ × ℝ)

-- Define the conditions for A and B being on C1 and C2
def A_on_C1 := ∃ t₁ : ℝ, C1 t₁ = A
def B_on_C1 := ∃ t₂ : ℝ, C1 t₂ = B
def A_on_C2 := ∃ θ₁ : ℝ, (C2 θ₁ * Real.cos θ₁, C2 θ₁ * Real.sin θ₁) = A
def B_on_C2 := ∃ θ₂ : ℝ, (C2 θ₂ * Real.cos θ₂, C2 θ₂ * Real.sin θ₂) = B

-- State the theorem
theorem sum_of_reciprocal_distances :
  A_on_C1 A → B_on_C1 B → A_on_C2 A → B_on_C2 B →
  1 / ‖A - F‖ + 1 / ‖B - F‖ = 4/3 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_reciprocal_distances_l1358_135835


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_omega_range_for_two_max_points_l1358_135854

noncomputable section

/-- The function f(x) = 2sin(ωx + π/3) -/
def f (ω : ℝ) (x : ℝ) : ℝ := 2 * Real.sin (ω * x + Real.pi / 3)

/-- The theorem stating the range of ω for f(x) with exactly two maximum points in [0, 1] -/
theorem omega_range_for_two_max_points (ω : ℝ) :
  ω > 0 →
  (∀ x, x ∈ Set.Icc 0 1 → f ω x ∈ Set.Icc (-2) 2) →
  (∃ x₁ x₂, x₁ ∈ Set.Ioo 0 1 ∧ x₂ ∈ Set.Ioo 0 1 ∧ x₁ ≠ x₂ ∧
    (∀ x, x ∈ Set.Icc 0 1 → f ω x ≤ f ω x₁ ∧ f ω x ≤ f ω x₂) ∧
    (∀ x₃, x₃ ∈ Set.Ioo 0 1 → x₃ ≠ x₁ → x₃ ≠ x₂ →
      ∃ ε > 0, ∀ y, y ∈ Set.Ioo (x₃ - ε) (x₃ + ε) → f ω y < f ω x₃)) →
  13 * Real.pi / 6 ≤ ω ∧ ω < 25 * Real.pi / 6 :=
by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_omega_range_for_two_max_points_l1358_135854


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_center_proof_l1358_135881

-- Define the lines
noncomputable def line1 (x y : ℝ) : Prop := 3 * x - 5 * y = 15
noncomputable def line2 (x y : ℝ) : Prop := 3 * x - 5 * y = -45
noncomputable def centerLine (x y : ℝ) : Prop := x - 3 * y = 0

-- Define the center point
noncomputable def centerPoint : ℝ × ℝ := (-45/4, -15/4)

-- Theorem statement
theorem circle_center_proof :
  let (x, y) := centerPoint
  (centerLine x y) ∧
  (∃ r : ℝ, r > 0 ∧
    (∀ x' y' : ℝ, (x' - x)^2 + (y' - y)^2 = r^2 →
      (line1 x' y' ∨ line2 x' y'))) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_center_proof_l1358_135881


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_possible_b_values_l1358_135816

/-- The function f defined as b / (3x - 4) -/
noncomputable def f (b : ℝ) (x : ℝ) : ℝ := b / (3 * x - 4)

/-- The inverse function of f -/
noncomputable def f_inv (b : ℝ) (y : ℝ) : ℝ := 
  (b / y + 4) / 3

/-- Theorem stating that the sum of all possible values of b is 19/3 -/
theorem sum_of_possible_b_values :
  ∃ (b₁ b₂ : ℝ), 
    (∀ x, f b₁ (f_inv b₁ x) = x) ∧
    (∀ x, f b₂ (f_inv b₂ x) = x) ∧
    (f b₁ 3 = f_inv b₁ (b₁ + 2)) ∧
    (f b₂ 3 = f_inv b₂ (b₂ + 2)) ∧
    b₁ + b₂ = 19/3 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_possible_b_values_l1358_135816


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cricket_team_age_problem_l1358_135848

theorem cricket_team_age_problem (team_size : ℕ) (team_avg_age : ℝ) (wicket_keeper_age_diff : ℝ) (remaining_avg_age_diff : ℝ) :
  team_size = 11 →
  team_avg_age = 27 →
  wicket_keeper_age_diff = 3 →
  remaining_avg_age_diff = 1 →
  let total_age := team_size * team_avg_age
  let wicket_keeper_age := team_avg_age + wicket_keeper_age_diff
  let remaining_players := team_size - 2
  let remaining_avg_age := team_avg_age - remaining_avg_age_diff
  let other_player_age := total_age - wicket_keeper_age - (remaining_players * remaining_avg_age)
  (total_age - wicket_keeper_age - other_player_age) / remaining_players = 26 := by
  sorry

#check cricket_team_age_problem

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cricket_team_age_problem_l1358_135848


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_dart_in_square_probability_l1358_135897

/-- The side length of the regular octagon -/
noncomputable def octagon_side : ℝ := 2 + Real.sqrt 2

/-- The probability of a dart landing within the inscribed square of a regular octagon -/
noncomputable def dart_probability : ℝ := (4 + 3 * Real.sqrt 2) / (18 + 13 * Real.sqrt 2)

/-- The theorem stating the probability of a dart landing within the inscribed square -/
theorem dart_in_square_probability :
  let octagon_area := 2 * (1 + Real.sqrt 2) * octagon_side ^ 2
  let square_side := octagon_side * Real.sqrt 2
  let square_area := square_side ^ 2
  square_area / octagon_area = dart_probability := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_dart_in_square_probability_l1358_135897


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_properties_l1358_135810

-- Define the sequence a_n
noncomputable def a (n : ℤ) : ℝ := ((2 + Real.sqrt 3) ^ n - (2 - Real.sqrt 3) ^ n) / (2 * Real.sqrt 3)

-- Theorem statement
theorem a_properties :
  (∀ n : ℤ, ∃ m : ℤ, a n = m) ∧
  (∀ n : ℤ, (∃ k : ℤ, a n = k * 3) ↔ (∃ k : ℤ, n = 3 * k)) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_properties_l1358_135810


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_hyperbola_center_distance_l1358_135806

/-- The distance between the center of a circle and the center of a hyperbola,
    given specific conditions. -/
theorem circle_hyperbola_center_distance :
  ∀ (x y : ℝ),
  (x^2 / 9 - y^2 / 16 = 1) →  -- The circle's center lies on the hyperbola
  ((x = 3 ∧ y = 0) ∨ (x = -3 ∧ y = 0) ∨ (x = 4 ∧ y = 0) ∨ (x = -4 ∧ y = 0)) →  -- The circle passes through a vertex or focus
  Real.sqrt (x^2 + y^2) = 16/3 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_hyperbola_center_distance_l1358_135806


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_alpha_value_l1358_135833

noncomputable def f (x : ℝ) := Real.cos (2 * x - Real.pi / 3) + 2 * Real.sin x ^ 2

theorem sin_alpha_value (α : ℝ) (h_acute : 0 < α ∧ α < Real.pi / 2) 
  (h_f : f (α / 2) = 3 / 4) : Real.sin α = (Real.sqrt 15 - Real.sqrt 3) / 8 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_alpha_value_l1358_135833


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetry_plane_property_l1358_135860

/-- A convex polyhedron -/
class ConvexPolyhedron (α : Type*) [NormedAddCommGroup α] [NormedSpace ℝ α] where
  -- We don't need to define the full structure, just declare it exists
  dummy : Unit

/-- A plane in 3D space -/
class Plane (α : Type*) [NormedAddCommGroup α] [NormedSpace ℝ α] where
  -- We don't need to define the full structure, just declare it exists
  dummy : Unit

/-- A polyhedral angle at a vertex -/
class PolyhedralAngle (α : Type*) [NormedAddCommGroup α] [NormedSpace ℝ α] where
  -- We don't need to define the full structure, just declare it exists
  dummy : Unit

/-- An edge of a polyhedron -/
class PolyhedronEdge (α : Type*) [NormedAddCommGroup α] [NormedSpace ℝ α] where
  -- We don't need to define the full structure, just declare it exists
  dummy : Unit

/-- Symmetry of a polyhedron with respect to a plane -/
def is_symmetric {α : Type*} [NormedAddCommGroup α] [NormedSpace ℝ α] (P : ConvexPolyhedron α) (π : Plane α) : Prop :=
  sorry

/-- A plane passes through the midpoint of an edge -/
def passes_through_midpoint {α : Type*} [NormedAddCommGroup α] [NormedSpace ℝ α] (π : Plane α) (e : PolyhedronEdge α) : Prop :=
  sorry

/-- A plane is a symmetry plane of a polyhedral angle -/
def is_symmetry_plane_of_angle {α : Type*} [NormedAddCommGroup α] [NormedSpace ℝ α] (π : Plane α) (a : PolyhedralAngle α) : Prop :=
  sorry

theorem symmetry_plane_property 
  {α : Type*} [NormedAddCommGroup α] [NormedSpace ℝ α]
  (P : ConvexPolyhedron α) (π : Plane α) :
  is_symmetric P π →
  (∃ (e : PolyhedronEdge α), passes_through_midpoint π e) ∨
  (∃ (a : PolyhedralAngle α), is_symmetry_plane_of_angle π a) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetry_plane_property_l1358_135860


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cash_before_brokerage_proof_l1358_135837

/-- The cash realized on selling a stock before brokerage -/
noncomputable def cash_before_brokerage : ℝ := 108.27

/-- The brokerage rate as a decimal -/
noncomputable def brokerage_rate : ℝ := 1 / 400

/-- The net amount received after brokerage -/
noncomputable def net_amount : ℝ := 108

theorem cash_before_brokerage_proof :
  cash_before_brokerage * (1 - brokerage_rate) = net_amount := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cash_before_brokerage_proof_l1358_135837


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_properties_l1358_135845

noncomputable def sequence_a : ℕ → ℝ
  | 0 => 0
  | n + 1 => (3 * sequence_a n + Real.sqrt (5 * (sequence_a n)^2 + 4)) / 2

theorem sequence_properties :
  (∀ n : ℕ, ∃ k : ℤ, sequence_a n = k) ∧
  (∀ n : ℕ, ∃ k m l : ℕ,
    (sequence_a n * sequence_a (n + 1) + 1 = k^2) ∧
    (sequence_a (n + 1) * sequence_a (n + 2) + 1 = m^2) ∧
    (sequence_a n * sequence_a (n + 2) + 1 = l^2)) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_properties_l1358_135845


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rhombus_inscribed_circle_radius_l1358_135831

/-- The radius of a circle inscribed in a rhombus with given diagonal lengths -/
noncomputable def inscribed_circle_radius (d1 d2 : ℝ) : ℝ :=
  60 / Real.sqrt 241

theorem rhombus_inscribed_circle_radius :
  let d1 : ℝ := 8
  let d2 : ℝ := 30
  inscribed_circle_radius d1 d2 = 60 / Real.sqrt 241 := by
  -- Unfold the definition of inscribed_circle_radius
  unfold inscribed_circle_radius
  -- The equality holds by definition
  rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_rhombus_inscribed_circle_radius_l1358_135831


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_not_shaded_value_l1358_135803

/-- Represents a rectangle in the 2 by 2005 grid -/
structure Rectangle where
  left : Nat
  right : Nat
  top : Bool
  bottom : Bool

/-- The total number of rectangles in the grid -/
def total_rectangles : ℕ := 3 * (Nat.choose 2005 2)

/-- The number of rectangles that include a shaded square -/
def shaded_rectangles : ℕ := 3 * 1003^2

/-- The probability of choosing a rectangle that does not include a shaded square -/
def prob_not_shaded : ℚ :=
  1 - (shaded_rectangles : ℚ) / total_rectangles

theorem prob_not_shaded_value :
  prob_not_shaded = 1002 / 2005 := by
  sorry

#eval prob_not_shaded

end NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_not_shaded_value_l1358_135803


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_set_operations_and_subset_condition_l1358_135895

/-- The universal set U is the set of real numbers -/
def U : Set ℝ := Set.univ

/-- Set A is defined as {x | x < -4 or x > 1} -/
def A : Set ℝ := {x | x < -4 ∨ x > 1}

/-- Set B is defined as {x | -3 ≤ x - 1 ≤ 2} -/
def B : Set ℝ := {x | -3 ≤ x - 1 ∧ x - 1 ≤ 2}

/-- Set M is defined as {x | 2k - 1 ≤ x ≤ 2k + 1} for some real k -/
def M (k : ℝ) : Set ℝ := {x | 2*k - 1 ≤ x ∧ x ≤ 2*k + 1}

theorem set_operations_and_subset_condition :
  (A ∩ B = {x : ℝ | 1 < x ∧ x ≤ 3}) ∧
  ((Set.compl A) ∪ B = {x : ℝ | -4 ≤ x ∧ x ≤ 3}) ∧
  (∀ k, M k ⊆ B → -1/2 ≤ k ∧ k ≤ 1) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_set_operations_and_subset_condition_l1358_135895


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_focal_length_l1358_135894

/-- Given an ellipse with equation x^2/20 + y^2/36 = 1, its focal length is 8 -/
theorem ellipse_focal_length : 
  ∀ (x y : ℝ), (x^2 / 20 + y^2 / 36 = 1) → 
    ∃ (a b c : ℝ), 
      a^2 = 36 ∧ 
      b^2 = 20 ∧ 
      c^2 = a^2 - b^2 ∧ 
      2 * c = 8 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_focal_length_l1358_135894


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_a_bounds_and_decreasing_l1358_135886

def sequence_a : ℕ → ℚ
  | 0 => 1/2  -- Adding a case for 0
  | 1 => 1/2
  | n + 1 => 3 * sequence_a n - 3 * (sequence_a n)^2

theorem sequence_a_bounds_and_decreasing :
  (∀ n : ℕ, n ≥ 1 → 2/3 < sequence_a (2*n) ∧ sequence_a (2*n) ≤ 3/4) ∧
  (∀ n : ℕ, n ≥ 2 → sequence_a (2*n) < sequence_a (2*(n-1))) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_a_bounds_and_decreasing_l1358_135886


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_Q_fixed_l1358_135832

/-- A circle in which triangle ABC is inscribed -/
def Γ : EuclideanSpace ℝ (Fin 2) → Prop := sorry

/-- Triangle ABC inscribed in circle Γ -/
def Triangle (A B C : EuclideanSpace ℝ (Fin 2)) : Prop := sorry

/-- P is a point on arc AB not containing C -/
def OnArc (Γ : EuclideanSpace ℝ (Fin 2) → Prop) (A B P : EuclideanSpace ℝ (Fin 2)) : Prop := sorry

/-- I is the incenter of triangle ACP -/
def IsIncenter (I A C P : EuclideanSpace ℝ (Fin 2)) : Prop := sorry

/-- J is the incenter of triangle BCP -/
def IsIncenter' (J B C P : EuclideanSpace ℝ (Fin 2)) : Prop := sorry

/-- Q is the intersection of Γ and the circumcircle of triangle PIJ -/
def Intersection (Γ : EuclideanSpace ℝ (Fin 2) → Prop) (P I J Q : EuclideanSpace ℝ (Fin 2)) : Prop := sorry

/-- Theorem: Q is independent of P's position on arc AB -/
theorem Q_fixed (Γ : EuclideanSpace ℝ (Fin 2) → Prop) (A B C : EuclideanSpace ℝ (Fin 2)) 
  (h_inscribed : Triangle A B C) :
  ∀ P₁ P₂ I₁ I₂ J₁ J₂, 
    OnArc Γ A B P₁ → OnArc Γ A B P₂ → P₁ ≠ C → P₂ ≠ C → 
    IsIncenter I₁ A C P₁ → IsIncenter I₂ A C P₂ →
    IsIncenter' J₁ B C P₁ → IsIncenter' J₂ B C P₂ →
    ∃ Q, Intersection Γ P₁ I₁ J₁ Q ∧ Intersection Γ P₂ I₂ J₂ Q :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_Q_fixed_l1358_135832


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_valid_arrangement_l1358_135872

def is_valid_arrangement (arr : List Nat) : Prop :=
  arr.length = 100 ∧
  (∀ n, n ∈ arr → n ≥ 1 ∧ n ≤ 100) ∧
  (∀ n, n ∈ Finset.range 100 → n + 1 ∈ arr) ∧
  ∀ n, 2 ≤ n ∧ n ≤ 100 → 
    ∀ i, i + n ≤ 100 → 
      ¬ (n ∣ (List.sum (List.take n (List.drop i arr))))

theorem exists_valid_arrangement : 
  ∃ arr : List Nat, is_valid_arrangement arr := by
  sorry

#check exists_valid_arrangement

end NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_valid_arrangement_l1358_135872


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_OB_equals_1620_l1358_135864

/-- Represents a point on the road -/
inductive Point : Type
| A : Point
| O : Point
| B : Point

/-- Represents a person -/
inductive Person : Type
| X : Person
| Y : Person

/-- The distance between two points in meters -/
noncomputable def distance : Point → Point → ℝ := sorry

/-- The time in minutes since the start -/
noncomputable def time : ℝ := sorry

/-- The position of a person at a given time -/
noncomputable def position : Person → ℝ → Point := sorry

/-- The distance traveled by a person in a given time -/
noncomputable def distanceTraveled : Person → ℝ → ℝ := sorry

theorem distance_OB_equals_1620 
  (h1 : distance Point.A Point.O = 1620)
  (h2 : position Person.X 0 = Point.A)
  (h3 : position Person.Y 0 = Point.O)
  (h4 : position Person.X 36 = Point.B)
  (h5 : position Person.Y 36 = Point.B)
  (h6 : distanceTraveled Person.X 12 = distance Point.O (position Person.X 12))
  (h7 : distanceTraveled Person.Y 12 = distance Point.O (position Person.Y 12))
  : distance Point.O Point.B = 1620 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_OB_equals_1620_l1358_135864


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_sine_relation_l1358_135862

theorem triangle_sine_relation (A B C : Real) (α β y : ℕ) : 
  0 < A ∧ A < Real.pi/2 →
  0 < B ∧ B < Real.pi/2 →
  0 < C ∧ C < Real.pi →
  A + B + C = Real.pi →
  Real.sin A = 36 / α →
  Real.sin B = 12 / 13 →
  Real.sin C = β / y →
  Nat.gcd β y = 1 →
  β = 56 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_sine_relation_l1358_135862


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_65_60_25_l1358_135888

/-- The area of a triangle given its three side lengths using Heron's formula -/
noncomputable def triangle_area (a b c : ℝ) : ℝ :=
  let s := (a + b + c) / 2
  Real.sqrt (s * (s - a) * (s - b) * (s - c))

/-- Theorem: The area of a triangle with sides 65, 60, and 25 is 750 -/
theorem triangle_area_65_60_25 :
  triangle_area 65 60 25 = 750 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_65_60_25_l1358_135888


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_b_l1358_135824

-- Define the circle
def is_in_circle (x y : ℝ) : Prop := (x - 1)^2 + y^2 ≤ 1

-- Define the line
def on_line (x y b : ℝ) : Prop := x + Real.sqrt 3 * y + b = 0

-- Theorem statement
theorem range_of_b (b : ℝ) :
  (∃ x y : ℝ, is_in_circle x y ∧ on_line x y b) ↔ -3 ≤ b ∧ b ≤ 1 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_b_l1358_135824


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rohan_food_expense_l1358_135834

/-- Represents Rohan's monthly expenses and savings --/
structure RohanFinances where
  salary : ℚ
  house_rent_percent : ℚ
  entertainment_percent : ℚ
  conveyance_percent : ℚ
  savings : ℚ

/-- Calculates the percentage of salary spent on food --/
def food_expense_percent (r : RohanFinances) : ℚ :=
  100 - (r.house_rent_percent + r.entertainment_percent + r.conveyance_percent + (r.savings / r.salary * 100))

/-- Theorem stating that Rohan spends 40% of his salary on food --/
theorem rohan_food_expense (r : RohanFinances) 
  (h1 : r.salary = 5000)
  (h2 : r.house_rent_percent = 20)
  (h3 : r.entertainment_percent = 10)
  (h4 : r.conveyance_percent = 10)
  (h5 : r.savings = 1000) :
  food_expense_percent r = 40 := by
  sorry

#eval food_expense_percent { 
  salary := 5000, 
  house_rent_percent := 20, 
  entertainment_percent := 10, 
  conveyance_percent := 10, 
  savings := 1000 
}

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rohan_food_expense_l1358_135834


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l1358_135804

noncomputable def f (a b c : ℝ) (x : ℝ) : ℝ := (a * x^2 + 1) / (b * x + c)

theorem function_properties 
  (a b c : ℝ) 
  (h1 : f a b c 1 = 2) 
  (h2 : f a b c 2 = 3) :
  (∀ x, f a b c x = f a b c (-x) → f a b c x = (4 * x^2 + 5) / 3) ∧ 
  (∀ x, f a b c x = -f a b c (-x) → f a b c x = (4 * x^2 + 2) / (3 * x)) ∧
  (∀ x y, 0 < x ∧ x < y ∧ y < 1/2 → f 2 (3/2) 0 x > f 2 (3/2) 0 y) :=
by
  sorry

#check function_properties

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l1358_135804


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_line_slope_l1358_135855

-- Define the parabola
def parabola (x y : ℝ) : Prop := y^2 = 4*x

-- Define the focus of the parabola
def focus : ℝ × ℝ := (1, 0)

-- Define a line passing through the focus with slope k
def line (k : ℝ) (x y : ℝ) : Prop := y = k * (x - 1)

-- Define the point M
def point_M : ℝ × ℝ := (-1, 2)

-- Define the intersection points A and B
noncomputable def intersection_points (k : ℝ) : 
  ∃ (A B : ℝ × ℝ), parabola A.1 A.2 ∧ parabola B.1 B.2 ∧ 
                    line k A.1 A.2 ∧ line k B.1 B.2 := sorry

-- Define the perpendicularity condition
def perpendicular_condition (A B : ℝ × ℝ) : Prop :=
  let MA := (A.1 - point_M.1, A.2 - point_M.2)
  let MB := (B.1 - point_M.1, B.2 - point_M.2)
  MA.1 * MB.1 + MA.2 * MB.2 = 0

-- Theorem statement
theorem parabola_line_slope :
  ∃ (k : ℝ), 
    (∀ (x y : ℝ), line k x y → parabola x y) ∧
    (∃ (A B : ℝ × ℝ), 
      parabola A.1 A.2 ∧ parabola B.1 B.2 ∧ 
      line k A.1 A.2 ∧ line k B.1 B.2 ∧
      perpendicular_condition A B) →
    k = 1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_line_slope_l1358_135855


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_prime_divisibility_l1358_135809

theorem prime_divisibility (p q : ℕ) (n : ℤ) 
  (hp : Nat.Prime p) (hq : Nat.Prime q) (hpq : p ≠ q) 
  (hodd_p : Odd p) (hodd_q : Odd q)
  (hdiv_pq : (p * q : ℤ) ∣ (n^(p * q) + 1))
  (hdiv_p3q3 : ((p^3 * q^3 : ℕ) : ℤ) ∣ (n^(p * q) + 1)) :
  (p^2 : ℤ) ∣ (n + 1) ∨ (q^2 : ℤ) ∣ (n + 1) := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_prime_divisibility_l1358_135809


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complement_of_union_l1358_135866

theorem complement_of_union (U A B : Set ℕ) : 
  U = {1, 2, 3, 4, 5} →
  A = {1, 2} →
  B = {3, 4} →
  (U \ (A ∪ B) : Set ℕ) = {5} := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_complement_of_union_l1358_135866


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_asymptote_slope_l1358_135879

/-- Given a hyperbola with equation x²/a² - y²/b² = 1 where a > 0 and b > 0,
    if the line y = 2b intersects the hyperbola at a point A such that the slope of OA is -1,
    then the slope of the asymptotes of the hyperbola is ±√5/2 -/
theorem hyperbola_asymptote_slope (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  let f : ℝ → ℝ → Prop := λ x y => x^2 / a^2 - y^2 / b^2 = 1
  let A : ℝ × ℝ := (-Real.sqrt 5 * a, 2 * b)
  (f A.1 A.2 ∧ A.2 / A.1 = -1) →
  (b / a = Real.sqrt 5 / 2) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_asymptote_slope_l1358_135879


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_det_B_squared_minus_3B_l1358_135876

def B : Matrix (Fin 2) (Fin 2) ℝ := !![2, 4; 3, 2]

theorem det_B_squared_minus_3B : Matrix.det (B^2 - 3 • B) = 88 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_det_B_squared_minus_3B_l1358_135876


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_and_unique_integer_l1358_135858

-- Define the functions f and g
noncomputable def f (x : ℝ) : ℝ := Real.exp x * (2 * x - 1)
def g (a x : ℝ) : ℝ := a * x - a

-- State the theorem
theorem tangent_line_and_unique_integer (a : ℝ) :
  (∃ x₀ : ℝ, (∀ x : ℝ, g a x ≤ f x) ∧ (g a x₀ = f x₀) ∧ (∀ x : ℝ, x ≠ x₀ → g a x < f x)) →
  (a < 1) →
  (∃! x₀ : ℤ, f (x₀ : ℝ) < g a (x₀ : ℝ)) →
  (3 / (2 * Real.exp 1) ≤ a ∧ a < 1) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_and_unique_integer_l1358_135858


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_roots_sum_l1358_135893

theorem tan_roots_sum (α β : Real) : 
  (∃ (x : Real), x^2 + 6*x + 7 = 0 ∧ (x = Real.tan α ∨ x = Real.tan β)) →
  α ∈ Set.Ioo (-Real.pi/2) (Real.pi/2) →
  β ∈ Set.Ioo (-Real.pi/2) (Real.pi/2) →
  α + β = -3*Real.pi/4 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_roots_sum_l1358_135893


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_solution_is_negative_one_l1358_135878

-- Define the piecewise function f
noncomputable def f (x : ℝ) : ℝ :=
  if x < 2 then x^2 - 2*x else 2*x + 1

-- State the theorem
theorem unique_solution_is_negative_one :
  ∃! a : ℝ, f a = 3 ∧ a = -1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_solution_is_negative_one_l1358_135878


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_equation_proof_l1358_135823

-- Define the circle
def my_circle (x y : ℝ) : Prop := x^2 + (y - 3)^2 = 4

-- Define the given line
def given_line (x y : ℝ) : Prop := x + y + 1 = 0

-- Define the line we want to prove
def target_line (x y : ℝ) : Prop := x - y + 3 = 0

-- Theorem statement
theorem line_equation_proof :
  ∀ (x y : ℝ),
  (∃ (center_x center_y : ℝ), my_circle center_x center_y ∧ center_x = 0 ∧ center_y = 3) →
  (∀ (m : ℝ), (y - 3 = m * (x - 0)) → m * (-1) = -1) →
  target_line x y :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_equation_proof_l1358_135823


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l1358_135867

/-- An odd function f defined on ℝ -/
def f (a : ℝ) : ℝ → ℝ := sorry

/-- Property: f is an odd function -/
axiom f_odd (a : ℝ) : ∀ x, f a (-x) = -(f a x)

/-- Definition of f for positive x -/
axiom f_pos (a : ℝ) : ∀ x > 0, f a x = -x^2 + a*x + a + 1

/-- f is monotonically decreasing on ℝ -/
axiom f_decreasing (a : ℝ) : ∀ x y, x < y → f a x > f a y

theorem f_properties (a : ℝ) :
  (∀ x < 0, f a x = x^2 - a*x + a + 1) ∧ a ≤ 0 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l1358_135867


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_difference_divisibility_l1358_135808

theorem cube_difference_divisibility (p : ℕ) (a b : ℤ) 
  (h_prime : Nat.Prime p) 
  (h_three_divides : 3 ∣ p + 1) : 
  (p : ℤ) ∣ (a - b) ↔ (p : ℤ) ∣ (a^3 - b^3) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_difference_divisibility_l1358_135808


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_S_l1358_135865

noncomputable def floor (x : ℝ) : ℤ := Int.floor x

def S : Set (ℝ × ℝ) := {p : ℝ × ℝ | |floor (p.1 + p.2)| + |floor (p.1 - p.2)| ≤ 1}

theorem area_of_S : MeasureTheory.volume S = 5/2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_S_l1358_135865


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_C₂_equation_C₂_range_l1358_135847

-- Define the curve C₁
noncomputable def C₁ (α : ℝ) : ℝ × ℝ := (3/2 * Real.cos α, Real.sin α)

-- Define the relationship between points on C₁ and C₂
def C₂_point (M : ℝ × ℝ) : ℝ × ℝ := (2 * M.1, 2 * M.2)

-- Theorem for the general equation of C₂
theorem C₂_equation (x y : ℝ) :
  (∃ α : ℝ, C₂_point (C₁ α) = (x, y)) ↔ x^2/9 + y^2/4 = 1 := by
  sorry

-- Theorem for the range of x + 2y on C₂
theorem C₂_range (x y : ℝ) :
  (∃ α : ℝ, C₂_point (C₁ α) = (x, y)) → -5 ≤ x + 2*y ∧ x + 2*y ≤ 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_C₂_equation_C₂_range_l1358_135847


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ride_cost_is_four_l1358_135870

/-- The number of tickets required for a single ride on the rollercoaster or Catapult -/
def ride_cost : ℕ := 0

/-- The total number of tickets Turner needs -/
def total_tickets : ℕ := 21

/-- The number of times Turner rides the rollercoaster -/
def rollercoaster_rides : ℕ := 3

/-- The number of times Turner rides the Catapult -/
def catapult_rides : ℕ := 2

/-- The number of times Turner rides the Ferris wheel -/
def ferris_wheel_rides : ℕ := 1

/-- The cost of riding the Ferris wheel once -/
def ferris_wheel_cost : ℕ := 1

theorem ride_cost_is_four :
  ride_cost * (rollercoaster_rides + catapult_rides) + ferris_wheel_cost * ferris_wheel_rides = total_tickets →
  ride_cost = 4 := by
  intro h
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ride_cost_is_four_l1358_135870


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_great_white_shark_teeth_l1358_135869

/-- The number of teeth of different shark species -/
structure SharkTeeth where
  tiger : ℕ
  hammerhead : ℕ
  great_white : ℕ

/-- The conditions given in the problem -/
def shark_teeth_conditions (s : SharkTeeth) : Prop :=
  s.tiger = 180 ∧
  s.hammerhead = s.tiger / 6 ∧
  s.great_white = 2 * (s.tiger + s.hammerhead)

/-- The theorem stating that a great white shark has 420 teeth under the given conditions -/
theorem great_white_shark_teeth (s : SharkTeeth) 
  (h : shark_teeth_conditions s) : s.great_white = 420 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_great_white_shark_teeth_l1358_135869


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_integer_bound_l1358_135828

/-- The function f(x) = x(1 + ln x) / (x - 1) -/
noncomputable def f (x : ℝ) : ℝ := x * (1 + Real.log x) / (x - 1)

/-- The domain of f is x > 1 -/
def domain (x : ℝ) : Prop := x > 1

theorem max_integer_bound (k : ℤ) : 
  (∀ x, domain x → f x > (k : ℝ)) ↔ k ≤ 3 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_integer_bound_l1358_135828


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_square_exists_l1358_135849

/-- A regular pentagon -/
structure RegularPentagon where
  -- Define properties of a regular pentagon
  vertices : Fin 5 → ℝ × ℝ
  is_regular : True  -- Placeholder for regularity condition

/-- A square inscribed in a pentagon -/
structure InscribedSquare (p : RegularPentagon) where
  vertices : Fin 4 → ℝ × ℝ
  on_sides : ∀ i : Fin 4, ∃ j k : Fin 5, (vertices i).1 ∈ Set.Icc (p.vertices j).1 (p.vertices k).1 ∧
                                         (vertices i).2 ∈ Set.Icc (p.vertices j).2 (p.vertices k).2
  is_square : True  -- Placeholder for square properties

/-- Theorem: There exists a square inscribed in a regular pentagon with vertices on four sides -/
theorem inscribed_square_exists (p : RegularPentagon) : 
  ∃ s : InscribedSquare p, (∃ (i j k l : Fin 5), i ≠ j ∧ j ≠ k ∧ k ≠ l ∧ l ≠ i ∧
    (∀ v : Fin 4, ((s.vertices v).1 ∈ Set.Icc (p.vertices i).1 (p.vertices j).1 ∧
                   (s.vertices v).2 ∈ Set.Icc (p.vertices i).2 (p.vertices j).2) ∨
                  ((s.vertices v).1 ∈ Set.Icc (p.vertices j).1 (p.vertices k).1 ∧
                   (s.vertices v).2 ∈ Set.Icc (p.vertices j).2 (p.vertices k).2) ∨
                  ((s.vertices v).1 ∈ Set.Icc (p.vertices k).1 (p.vertices l).1 ∧
                   (s.vertices v).2 ∈ Set.Icc (p.vertices k).2 (p.vertices l).2) ∨
                  ((s.vertices v).1 ∈ Set.Icc (p.vertices l).1 (p.vertices i).1 ∧
                   (s.vertices v).2 ∈ Set.Icc (p.vertices l).2 (p.vertices i).2))) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_square_exists_l1358_135849


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_garden_division_count_l1358_135877

/-- Represents the number of ways to divide a garden of size (2n + 1) × (2n + 1) -/
def garden_divisions (n : ℕ) : ℕ :=
  2^n

/-- Represents the number of ways to divide a square garden -/
def number_of_ways_to_divide_garden 
  (side_length : ℕ) 
  (valid_rectangle_sizes : ℕ → Prop)
  (horizontal_count : ℕ) 
  (vertical_count : ℕ)
  (square_count : ℕ) : ℕ :=
  sorry

/-- Theorem stating the number of ways to divide the garden as specified -/
theorem garden_division_count (n : ℕ) :
  garden_divisions n = 
    number_of_ways_to_divide_garden 
      (2*n + 1) 
      (λ k ↦ k % 2 = 0 ∧ 1 ≤ k ∧ k ≤ 2*n + 1) 
      2 
      2 
      1 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_garden_division_count_l1358_135877


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_neither_odd_nor_even_l1358_135846

-- Define the function f(x) = x + cos x
noncomputable def f (x : ℝ) : ℝ := x + Real.cos x

-- Define what it means for a function to be odd
def is_odd (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

-- Define what it means for a function to be even
def is_even (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = f x

-- Theorem stating that f is neither odd nor even
theorem f_neither_odd_nor_even : ¬(is_odd f) ∧ ¬(is_even f) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_neither_odd_nor_even_l1358_135846


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_expression_m_upperbound_l1358_135873

-- Define the vectors a and b
noncomputable def a (x : ℝ) : Fin 2 → ℝ := ![Real.sqrt 3, Real.cos (2 * x)]
noncomputable def b (x : ℝ) : Fin 2 → ℝ := ![Real.sin (2 * x), 1]

-- Define the dot product function
def dot_product (v w : Fin 2 → ℝ) : ℝ := (v 0) * (w 0) + (v 1) * (w 1)

-- Define f(x) as the dot product of a and b
noncomputable def f (x : ℝ) : ℝ := dot_product (a x) (b x)

theorem f_expression (x : ℝ) : f x = 2 * Real.sin (2 * x + π / 6) := by sorry

theorem m_upperbound (m : ℝ) (h : ∀ x ∈ Set.Icc 0 (π / 2), f x + m ≤ 3) : m ≤ 1 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_expression_m_upperbound_l1358_135873


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_line_equation_l1358_135889

/-- Given line passing through (1, 0) and perpendicular to x - y + 2 = 0 -/
def given_line (x y : ℝ) : Prop := x + y - 1 = 0

/-- The line x - y + 2 = 0 -/
def original_line (x y : ℝ) : Prop := x - y + 2 = 0

/-- The slope of a line given by ax + by + c = 0 -/
noncomputable def line_slope (a b : ℝ) : ℝ := -a / b

theorem perpendicular_line_equation : 
  (given_line 1 0) ∧ 
  (line_slope 1 1 * line_slope 1 (-1) = -1) ∧
  (∀ x y : ℝ, given_line x y → (x, y) ≠ (1, 0) → 
    ((y - 0) / (x - 1) : ℝ) = line_slope 1 (-1)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_line_equation_l1358_135889


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_frustum_lateral_surface_area_l1358_135815

/-- The lateral surface area of a frustum of a right circular cone. -/
noncomputable def lateral_surface_area (r₁ r₂ h : ℝ) : ℝ :=
  let s := Real.sqrt (h^2 + (r₁ - r₂)^2)
  Real.pi * (r₁ + r₂) * s

/-- Theorem: The lateral surface area of a frustum with given dimensions is 84π√2 square inches. -/
theorem frustum_lateral_surface_area :
  lateral_surface_area 10 4 6 = 84 * Real.pi * Real.sqrt 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_frustum_lateral_surface_area_l1358_135815


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_integral_2x_plus_sqrt_1_minus_x_squared_l1358_135844

/-- The definite integral of (2x+√(1-x²)) from 0 to 1 is equal to 1 + π/4 -/
theorem integral_2x_plus_sqrt_1_minus_x_squared :
  ∫ x in Set.Icc 0 1, (2 * x + Real.sqrt (1 - x^2)) = 1 + π / 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_integral_2x_plus_sqrt_1_minus_x_squared_l1358_135844


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_four_travel_packs_needed_l1358_135840

/-- The minimum number of travel packs needed to completely fill a standard-size bottle -/
def min_travel_packs (travel_pack_capacity : ℚ) (bottle_capacity : ℚ) : ℕ :=
  (bottle_capacity / travel_pack_capacity).ceil.toNat

/-- Theorem stating that 4 travel packs are needed to fill the standard-size bottle -/
theorem four_travel_packs_needed :
  min_travel_packs 80 270 = 4 := by
  -- Unfold the definition of min_travel_packs
  unfold min_travel_packs
  -- Evaluate the expression
  norm_num
  -- QED
  rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_four_travel_packs_needed_l1358_135840


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_set_with_property_P_l1358_135812

def has_property_P (M : Finset ℕ) : Prop :=
  ∀ a b c, a ∈ M → b ∈ M → c ∈ M → a * b ≠ c

def is_valid_set (M : Finset ℕ) : Prop :=
  (∀ x ∈ M, x ≤ 2011) ∧ has_property_P M

def maximum_set_size : ℕ := 1968

theorem max_set_with_property_P :
  ∃ M : Finset ℕ, is_valid_set M ∧ M.card = maximum_set_size ∧
  ∀ N : Finset ℕ, is_valid_set N → N.card ≤ maximum_set_size :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_set_with_property_P_l1358_135812


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_anishaSequence_result_l1358_135890

def anishaSequence (n : ℕ) : ℚ → ℚ :=
  match n with
  | 0 => id
  | n+1 => if n % 2 = 0 then (· / 3) ∘ anishaSequence n else (· * 4) ∘ anishaSequence n

theorem anishaSequence_result :
  anishaSequence 10 2187000 = 9216000 := by
  sorry

#eval anishaSequence 10 2187000

end NUMINAMATH_CALUDE_ERRORFEEDBACK_anishaSequence_result_l1358_135890


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_existence_of_convex_pentagon_l1358_135836

/-- A point in a 2D plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- A set of 9 points in a 2D plane -/
def NinePoints : Type := Fin 9 → Point

/-- Predicate to check if three points are collinear -/
def collinear (p q r : Point) : Prop :=
  (q.y - p.y) * (r.x - q.x) = (r.y - q.y) * (q.x - p.x)

/-- Predicate to check if a set of points forms a convex pentagon -/
def is_convex_pentagon (s : Finset Point) : Prop :=
  s.card = 5 ∧ ∀ p q r, p ∈ s → q ∈ s → r ∈ s → ¬collinear p q r

theorem existence_of_convex_pentagon (points : NinePoints) 
  (h : ∀ i j k, i ≠ j → j ≠ k → i ≠ k → ¬collinear (points i) (points j) (points k)) :
  ∃ s : Finset Point, is_convex_pentagon s ∧ ∀ p, p ∈ s → ∃ i, p = points i := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_existence_of_convex_pentagon_l1358_135836


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_of_A_and_B_l1358_135821

-- Define the sets A and B
def A : Set ℝ := {x | x > 1/2}
def B : Set ℝ := {x | -1 < x ∧ x < 3}

-- State the theorem
theorem intersection_of_A_and_B : A ∩ B = Set.Ioo (1/2) 3 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_of_A_and_B_l1358_135821


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_count_pseudo_ultrafilters_l1358_135857

def N : ℕ := Nat.factorial 12

def X : Set ℕ := {d | d > 1 ∧ d ∣ N}

def isPseudoUltrafilter (U : Set ℕ) : Prop :=
  U ⊆ X ∧ U.Nonempty ∧
  (∀ a b, a ∈ X → b ∈ X → (a ∣ b ∧ a ∈ U) → b ∈ U) ∧
  (∀ a b, a ∈ U → b ∈ U → Nat.gcd a b ∈ U) ∧
  (∀ a b, a ∈ X → b ∈ X → a ∉ U → b ∉ U → Nat.lcm a b ∉ U)

theorem count_pseudo_ultrafilters :
  ∃ S : Finset (Set ℕ), 
    (∀ U ∈ S, isPseudoUltrafilter U) ∧
    (∀ U, isPseudoUltrafilter U → U ∈ S) ∧
    S.card = 19 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_count_pseudo_ultrafilters_l1358_135857


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_y_l1358_135874

noncomputable def y (x : ℝ) := Real.sqrt (Real.sin x) + Real.sqrt (Real.cos x - 1/2)

theorem domain_of_y :
  ∀ x : ℝ, (∃ k : ℤ, x ∈ Set.Icc (2 * k * Real.pi) ((1/3) * Real.pi + 2 * k * Real.pi)) ↔
    (Real.sin x ≥ 0 ∧ Real.cos x ≥ 1/2) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_y_l1358_135874


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_price_relation_buying_equation_book_pricing_and_cost_l1358_135861

-- Define the unit prices of the books
def price_sunzi : ℚ := 30
def price_zhoubi : ℚ := 40

-- Define the relationship between the prices
theorem price_relation : price_sunzi = (3/4) * price_zhoubi := by
  -- Proof skipped
  sorry

-- Define the equation for buying books
theorem buying_equation : (600 / price_sunzi) - (600 / price_zhoubi) = 5 := by
  -- Proof skipped
  sorry

-- Define the total number of books to be purchased
def total_books : ℕ := 80

-- Define the constraint on the quantities of books
def quantity_constraint (m : ℕ) : Prop := total_books - m ≥ m / 2

-- Define the discount rate
def discount_rate : ℚ := 4/5

-- Define the cost function
def cost (m : ℕ) : ℚ :=
  discount_rate * (m * price_sunzi + (total_books - m) * price_zhoubi)

-- State the theorem
theorem book_pricing_and_cost :
  ∃ m, quantity_constraint m ∧
       (∀ n, quantity_constraint n → cost m ≤ cost n) ∧
       cost m = 2136 := by
  -- Proof skipped
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_price_relation_buying_equation_book_pricing_and_cost_l1358_135861


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_g_at_1_critical_point_H_critical_point_location_min_a_value_l1358_135899

noncomputable section

-- Define the functions f and g
def f (x : ℝ) : ℝ := 1 / (x^2)
def g (x : ℝ) : ℝ := Real.log x

-- Define the combined function H
def H (m : ℝ) (x : ℝ) : ℝ := m * f x + 2 * g x

-- Theorem for the tangent line of g at x = 1
theorem tangent_line_g_at_1 :
  ∀ x, (g x - g 1) = (x - 1) * ((deriv g) 1) := by sorry

-- Theorem for the critical point of H
theorem critical_point_H (m : ℝ) (hm : m ≠ 0) :
  (∃ x > 0, (deriv (H m)) x = 0) ↔ m > 0 := by sorry

-- Theorem for the location of the critical point when it exists
theorem critical_point_location (m : ℝ) (hm : m > 0) :
  ∃ x > 0, (deriv (H m)) x = 0 ∧ x = Real.sqrt m := by sorry

-- Theorem for the minimum value of a
theorem min_a_value :
  (∀ a : ℝ, (∀ x : ℝ, x ∈ Set.Ioc 0 1 → a * f x + g x ≥ a) ↔ a ≥ 1/2) := by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_g_at_1_critical_point_H_critical_point_location_min_a_value_l1358_135899


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_least_n_with_all_ones_last_digits_l1358_135820

theorem least_n_with_all_ones_last_digits : 
  ∃ (k : ℕ), (71 : ℤ)^k ≡ -1 [ZMOD (10^2012)] ∧ 
  ∀ (n : ℕ), n < 71 → ¬∃ (k : ℕ), (n : ℤ)^k ≡ -1 [ZMOD (10^2012)] :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_least_n_with_all_ones_last_digits_l1358_135820


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rhombus_unique_property_l1358_135884

-- Define the properties of quadrilaterals
class Quadrilateral :=
  (diagonals_bisect : Bool)
  (opposite_angles_equal : Bool)

-- Define rectangle
class Rectangle extends Quadrilateral :=
  (diagonals_equal : Bool)

-- Define rhombus
class Rhombus extends Quadrilateral :=
  (diagonals_perpendicular : Bool)

-- Theorem statement
theorem rhombus_unique_property 
  (r : Rectangle) 
  (h : Rhombus) : 
  h.diagonals_perpendicular ∧ 
  ¬r.diagonals_equal ∧
  r.diagonals_bisect = h.diagonals_bisect ∧
  r.opposite_angles_equal = h.opposite_angles_equal :=
by
  sorry

#check rhombus_unique_property

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rhombus_unique_property_l1358_135884


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_remainder_theorem_l1358_135880

theorem remainder_theorem (x : ℕ) (h : (7 * x) % 26 = 1) :
  (13 + 3 * x) % 26 = 6 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_remainder_theorem_l1358_135880
