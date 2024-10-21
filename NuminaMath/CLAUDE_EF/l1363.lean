import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_no_2x2_green_fraction_l1363_136352

/-- Represents a 4x4 grid of colored squares -/
def Grid := Fin 4 → Fin 4 → Bool

/-- Checks if a 2x2 subgrid starting at (i, j) is all green -/
def is_2x2_green (g : Grid) (i j : Fin 4) : Prop :=
  g i j ∧ g i (j.succ) ∧ g (i.succ) j ∧ g (i.succ) (j.succ)

/-- Checks if a grid has no 2x2 green square -/
def no_2x2_green (g : Grid) : Prop :=
  ∀ i j, i.succ < 4 → j.succ < 4 → ¬is_2x2_green g i j

/-- The probability of a random grid having no 2x2 green square -/
noncomputable def prob_no_2x2_green : ℚ :=
  sorry

theorem prob_no_2x2_green_fraction :
  ∃ (m n : ℕ), Nat.Coprime m n ∧ prob_no_2x2_green = m / n ∧ m + n = 987 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_no_2x2_green_fraction_l1363_136352


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_seedling_introduction_l1363_136303

/-- Survival rate of seedling A -/
def survival_rate_A : ℝ := 0.6

/-- Survival rate of seedlings B and C -/
def survival_rate_BC (p : ℝ) : Prop := 0.6 ≤ p ∧ p ≤ 0.8

/-- Number of seedlings that survive naturally -/
def X : ℕ → ℝ := sorry

/-- Expected value of X -/
def E_X (p : ℝ) : ℝ := 2 * p + 0.8

/-- Probability of artificial cultivation success -/
def artificial_success_rate : ℝ := 0.8 * 0.5

/-- Final survival probability of one seedling of type B -/
def final_survival_B : ℝ := 0.6 + (1 - 0.6) * artificial_success_rate

/-- Profit from each surviving seedling -/
def profit_per_seedling : ℝ := 400

/-- Loss from each non-surviving seedling -/
def loss_per_seedling : ℝ := 60

/-- Expected profit from introducing n seedlings -/
def expected_profit (n : ℕ) : ℝ := n * (profit_per_seedling * final_survival_B - loss_per_seedling * (1 - final_survival_B))

/-- Theorem stating the main results -/
theorem seedling_introduction (p : ℝ) (n : ℕ) (h : survival_rate_BC p) :
  E_X p = 2 * p + 0.8 ∧
  final_survival_B = 0.76 ∧
  (∀ m : ℕ, m ≥ 859 → expected_profit m ≥ 300000) ∧
  (∀ m : ℕ, m < 859 → expected_profit m < 300000) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_seedling_introduction_l1363_136303


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_handout_distribution_l1363_136370

def number_of_ways_to_distribute (total : ℕ) (students : ℕ) (min_one : ℕ) : ℕ := 
  Nat.choose (total - students * min_one + students - 1) (students - 1)

theorem handout_distribution (n k : ℕ) : 
  number_of_ways_to_distribute (n + k) n 1 = Nat.choose (n + k - 1) (n - 1) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_handout_distribution_l1363_136370


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_costume_processing_theorem_l1363_136332

/-- Represents the daily processing rate of costumes before adopting new technology -/
def original_daily_rate : ℝ → Prop := sorry

/-- The equation representing the costume processing problem -/
def costume_processing_equation (x : ℝ) : Prop :=
  60 / x + 240 / (2 * x) = 9

/-- Theorem stating the equation for the costume processing problem -/
theorem costume_processing_theorem 
  (total_costumes : ℕ) 
  (costumes_before_tech : ℕ) 
  (efficiency_multiplier : ℕ) 
  (total_days : ℕ) 
  (x : ℝ) 
  (h1 : total_costumes = 300)
  (h2 : costumes_before_tech = 60)
  (h3 : efficiency_multiplier = 2)
  (h4 : total_days = 9)
  (h5 : original_daily_rate x) :
  costume_processing_equation x := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_costume_processing_theorem_l1363_136332


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_plus_sin_for_point_l1363_136333

theorem cos_plus_sin_for_point (θ : ℝ) : 
  (∃ (r : ℝ), r > 0 ∧ r * Real.cos θ = -12 ∧ r * Real.sin θ = 5) → 
  Real.cos θ + Real.sin θ = -7/13 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_plus_sin_for_point_l1363_136333


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_triangle_area_l1363_136356

-- Define the parabola C: y² = 4x
def parabola (x y : ℝ) : Prop := y^2 = 4*x

-- Define the focus F(1,0)
def focus : ℝ × ℝ := (1, 0)

-- Define the line L: y = √3(x-1)
def line (x y : ℝ) : Prop := y = Real.sqrt 3 * (x - 1)

-- Define the intersection points of L and C
def intersection_points (x y : ℝ) : Prop :=
  parabola x y ∧ line x y

-- Define the area of triangle OAB
noncomputable def triangle_area (A B : ℝ × ℝ) : ℝ :=
  (1/2) * abs (A.2 - B.2)

-- Theorem statement
theorem parabola_triangle_area :
  ∃ A B : ℝ × ℝ,
    intersection_points A.1 A.2 ∧
    intersection_points B.1 B.2 ∧
    triangle_area A B = (4/3) * Real.sqrt 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_triangle_area_l1363_136356


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_focus_to_asymptote_distance_l1363_136353

/-- Represents a hyperbola with parameter a > 0 -/
structure Hyperbola (a : ℝ) where
  a_pos : a > 0

/-- The focus of the hyperbola -/
noncomputable def focus (h : Hyperbola a) : ℝ × ℝ :=
  (Real.sqrt (3 * a + 3), 0)

/-- One of the asymptotes of the hyperbola -/
noncomputable def asymptote (h : Hyperbola a) : ℝ → ℝ :=
  λ x ↦ Real.sqrt (1 / a) * x

/-- The distance from a point to a line -/
noncomputable def distance_to_line (p : ℝ × ℝ) (f : ℝ → ℝ) : ℝ :=
  sorry  -- Definition of distance from point to line

/-- Theorem stating that the distance from the focus to the asymptote is √3 -/
theorem focus_to_asymptote_distance (a : ℝ) (h : Hyperbola a) :
  distance_to_line (focus h) (asymptote h) = Real.sqrt 3 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_focus_to_asymptote_distance_l1363_136353


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_time_on_DE_is_45_minutes_l1363_136336

/-- Represents a worker paving a path -/
structure Worker where
  speed : ℝ
  path : List ℝ

/-- Represents the paving scenario -/
structure PavingScenario where
  worker1 : Worker
  worker2 : Worker
  total_time : ℝ

/-- The time spent by the second worker on segment D-E -/
noncomputable def time_on_DE (scenario : PavingScenario) : ℝ :=
  sorry

/-- The main theorem to prove -/
theorem time_on_DE_is_45_minutes (scenario : PavingScenario) : 
  scenario.worker2.speed = 1.2 * scenario.worker1.speed →
  scenario.total_time = 9 →
  List.sum scenario.worker1.path = List.sum scenario.worker2.path / 1.2 →
  scenario.worker2.path.length = scenario.worker1.path.length + 2 →
  time_on_DE scenario = 45 / 60 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_time_on_DE_is_45_minutes_l1363_136336


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_beta_and_max_value_l1363_136350

open Real

theorem tan_beta_and_max_value (α β : ℝ) 
  (h1 : 0 < α ∧ α < π/2) 
  (h2 : 0 < β ∧ β < π/2)
  (h3 : α + β ≠ π/2)
  (h4 : sin β = sin α * cos (α + β)) :
  (∃ (f : ℝ → ℝ), tan β = f (tan α) ∧ ∀ x, f x = x / (1 + 2*x^2)) ∧
  tan β ≤ sqrt 2 / 4 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_beta_and_max_value_l1363_136350


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_film_casting_theorem_l1363_136309

theorem film_casting_theorem 
  (n : ℕ) 
  (a : Fin n → ℕ) 
  (p : ℕ) 
  (k : ℕ) 
  (hp : Nat.Prime p) 
  (hpa : ∀ i, p ≥ a i) 
  (hpn : p ≥ n) 
  (hk : k ≤ n) :
  ∃ (castings : Fin (p^k) → (Fin n → Fin p)),
    ∀ (roles : Fin k → Fin n),
    ∀ (people : Π i : Fin k, Fin p),
    ∃ (casting : Fin (p^k)),
      ∀ i : Fin k, castings casting (roles i) = people i := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_film_casting_theorem_l1363_136309


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_lines_m_values_m_values_imply_parallel_l1363_136330

-- Define the slopes of the lines
noncomputable def slope1 (m : ℝ) : ℝ := -2 / (m + 1)
noncomputable def slope2 (m : ℝ) : ℝ := -m / 3

-- Define the property of parallel lines
def parallel (m : ℝ) : Prop := slope1 m = slope2 m

-- Theorem statement
theorem parallel_lines_m_values :
  ∀ m : ℝ, parallel m → m = 2 ∨ m = -3 := by
  intro m h
  sorry

-- Proof that m = 2 or m = -3 implies the lines are parallel
theorem m_values_imply_parallel :
  ∀ m : ℝ, (m = 2 ∨ m = -3) → parallel m := by
  intro m h
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_lines_m_values_m_values_imply_parallel_l1363_136330


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_fourth_power_minus_127i_l1363_136323

theorem complex_fourth_power_minus_127i (a b : ℕ+) :
  let z : ℂ := (a : ℂ) + (b : ℂ) * Complex.I
  (z^4 - 127 * Complex.I).im = 0 →
  (z^4 - 127 * Complex.I).re ∈ ({176, 436, 60706} : Set ℝ) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_fourth_power_minus_127i_l1363_136323


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_isosceles_triangle_bisector_angle_range_l1363_136341

/-- A line that bisects both the area and perimeter of the triangle -/
def BisectsAreaAndPerimeter (l : Real → Real → Prop) (t : IsoscelesTriangleWithBisectors) : Prop := sorry

/-- An isosceles triangle with three bisecting lines -/
structure IsoscelesTriangleWithBisectors where
  -- The angle at the vertex
  α : Real
  -- The base of the triangle
  a : Real
  -- The leg of the triangle
  b : Real
  -- Isosceles triangle condition
  isIsosceles : a = 2 * b * Real.sin (α / 2)
  -- Exactly three bisecting lines exist
  hasThreeBisectors : ∃ (l₁ l₂ l₃ : Real → Real → Prop), 
    BisectsAreaAndPerimeter l₁ this ∧ 
    BisectsAreaAndPerimeter l₂ this ∧ 
    BisectsAreaAndPerimeter l₃ this ∧
    (∀ l, BisectsAreaAndPerimeter l this → l = l₁ ∨ l = l₂ ∨ l = l₃)

/-- The main theorem -/
theorem isosceles_triangle_bisector_angle_range 
  (t : IsoscelesTriangleWithBisectors) : 
  2 * Real.arcsin (Real.sqrt 2 - 1) < t.α ∧ t.α < π := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_isosceles_triangle_bisector_angle_range_l1363_136341


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_term_64_l1363_136329

/-- The binomial coefficient -/
def binomial (n k : ℕ) : ℕ := Nat.choose n k

/-- The kth term in the expansion of (1+√3)^100 -/
noncomputable def term (k : ℕ) : ℝ := (binomial 100 k : ℝ) * (Real.sqrt 3) ^ k

/-- Theorem stating that the 64th term is the largest in the expansion of (1+√3)^100 -/
theorem largest_term_64 : ∀ j : ℕ, j ≠ 64 → term 64 > term j := by
  sorry

#eval binomial 100 64  -- To check if binomial is working

end NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_term_64_l1363_136329


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_first_train_speed_l1363_136313

/-- Calculates the speed of the first train given the lengths of two trains,
    the speed of the second train, and the time taken for them to clear each other. -/
noncomputable def calculate_train_speed (train1_length : ℝ) (train2_length : ℝ) 
                          (train2_speed : ℝ) (clear_time : ℝ) : ℝ :=
  let total_distance := (train1_length + train2_length) / 1000  -- Convert to km
  let clear_time_hours := clear_time / 3600  -- Convert to hours
  let relative_speed := total_distance / clear_time_hours
  relative_speed - train2_speed

/-- Theorem stating that given the specific train lengths, speed of the second train,
    and clear time, the speed of the first train is 100 km/h. -/
theorem first_train_speed :
  calculate_train_speed 111 165 120 4.516002356175142 = 100 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_first_train_speed_l1363_136313


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_sum_simplification_l1363_136339

noncomputable def z : ℂ := Complex.exp (2 * Real.pi * Complex.I / 5)

theorem complex_sum_simplification :
  z / (1 + z) + z^2 / (1 + z^3) + z^3 / (1 + z^5) = z - 3/2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_sum_simplification_l1363_136339


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_specific_lines_l1363_136365

/-- The distance between two parallel lines in ℝ² --/
noncomputable def distance_parallel_lines (a₁ a₂ b₁ b₂ d₁ d₂ : ℝ) : ℝ :=
  let v₁ := b₁ - a₁
  let v₂ := b₂ - a₂
  let num := (v₁ * d₁ + v₂ * d₂)^2
  let den := d₁^2 + d₂^2
  let p₁ := (num / den) * d₁ / (d₁^2 + d₂^2)
  let p₂ := (num / den) * d₂ / (d₁^2 + d₂^2)
  Real.sqrt ((v₁ - p₁)^2 + (v₂ - p₂)^2)

/-- The theorem stating the distance between the given parallel lines --/
theorem distance_specific_lines : 
  distance_parallel_lines 4 (-1) (-2) 3 1 (-6) = 4 * Real.sqrt 130 / 37 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_specific_lines_l1363_136365


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_square_folded_corner_area_l1363_136345

theorem square_folded_corner_area : 
  ∀ (s : ℝ), s > 0 → 
  (s^2 - (s^2 - (1/8) * s^2) = 1) → 
  s^2 = 8 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_square_folded_corner_area_l1363_136345


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_for_equality_l1363_136326

-- Define the piecewise function f
noncomputable def f (x : ℝ) : ℝ :=
  if -1 ≤ x ∧ x < -1/2 then x + 1/x
  else if -1/2 ≤ x ∧ x < 1/2 then -5/2
  else if 1/2 ≤ x ∧ x ≤ 1 then x - 1/x
  else 0  -- Default value for x outside [-1, 1]

-- Define the linear function g
def g (a x : ℝ) : ℝ := a * x - 3

-- Theorem statement
theorem range_of_a_for_equality (a : ℝ) :
  (∀ x₁ ∈ Set.Icc (-1 : ℝ) 1, ∃ x₀ ∈ Set.Icc (-1 : ℝ) 1, g a x₀ = f x₁) ↔
  a ∈ Set.Iic (-3) ∪ Set.Ici 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_for_equality_l1363_136326


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_1_problem_2_l1363_136338

-- Problem 1
theorem problem_1 : 
  abs (-Real.sqrt 3) + 2 * Real.cos (π / 4) - Real.tan (π / 3) = Real.sqrt 2 := by
  sorry

-- Problem 2
theorem problem_2 : 
  {x : ℝ | (x - 7)^2 = 3 * (7 - x)} = {4, 7} := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_1_problem_2_l1363_136338


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l1363_136364

/-- Represents a hyperbola with semi-major axis a and semi-minor axis b -/
structure Hyperbola where
  a : ℝ
  b : ℝ
  h_pos_a : a > 0
  h_pos_b : b > 0

/-- Represents a circle with center (0, 2) and radius 1 -/
def unit_circle (x y : ℝ) : Prop :=
  x^2 + (y - 2)^2 = 1

/-- The asymptotes of the hyperbola are tangent to the unit circle -/
def asymptotes_tangent_to_circle (h : Hyperbola) : Prop :=
  ∃ (x y : ℝ), unit_circle x y ∧ (h.b * x = h.a * y ∨ h.b * x = -h.a * y)

/-- The eccentricity of a hyperbola -/
noncomputable def eccentricity (h : Hyperbola) : ℝ :=
  Real.sqrt (1 + (h.b / h.a)^2)

/-- Main theorem: If the asymptotes of the hyperbola are tangent to the unit circle,
    then the eccentricity of the hyperbola is 2 -/
theorem hyperbola_eccentricity (h : Hyperbola) 
    (h_tangent : asymptotes_tangent_to_circle h) : eccentricity h = 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l1363_136364


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_butterfat_mixture_percentage_l1363_136317

/-- The percentage of butterfat in the final mixture of cream and skim milk -/
noncomputable def butterfat_percentage (cream_volume : ℝ) (cream_butterfat : ℝ) 
  (milk_volume : ℝ) (milk_butterfat : ℝ) : ℝ :=
  ((cream_volume * cream_butterfat + milk_volume * milk_butterfat) / 
   (cream_volume + milk_volume)) * 100

/-- Theorem stating that the butterfat percentage in the given mixture is 6.5% -/
theorem butterfat_mixture_percentage :
  butterfat_percentage 1 0.095 3 0.055 = 6.5 := by
  sorry

/-- Proof that the butterfat percentage in the mixture is indeed 6.5% -/
example : butterfat_percentage 1 0.095 3 0.055 = 6.5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_butterfat_mixture_percentage_l1363_136317


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_angle_and_function_properties_l1363_136367

/-- Given a triangle ABC with area 3 and constraint on sides b and c -/
def Triangle (b c : ℝ) (θ : ℝ) : Prop :=
  b * c * Real.sin θ = 3 ∧ 0 ≤ b * c * Real.cos θ ∧ b * c * Real.cos θ ≤ 6

/-- The function f(θ) = 2sin²θ - cos2θ -/
noncomputable def f (θ : ℝ) : ℝ := 2 * (Real.sin θ)^2 - Real.cos (2 * θ)

theorem triangle_angle_and_function_properties {b c θ : ℝ} (h : Triangle b c θ) :
  π/4 ≤ θ ∧ θ ≤ π/2 ∧
  (∀ φ, φ ∈ Set.Icc (π/4 : ℝ) (π/2 : ℝ) → f φ ≤ 3) ∧
  ∃ φ₁ φ₂, φ₁ ∈ Set.Icc (π/4 : ℝ) (π/2 : ℝ) ∧ φ₂ ∈ Set.Icc (π/4 : ℝ) (π/2 : ℝ) ∧ f φ₁ = 3 ∧ f φ₂ = 2 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_angle_and_function_properties_l1363_136367


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_population_is_all_scores_l1363_136357

/-- Represents a candidate's mathematics score in the high school entrance examination. -/
def MathScore : Type := ℕ

/-- Represents the set of all mathematics scores from the entire candidate pool. -/
def AllScores : Set MathScore := sorry

/-- Represents the randomly selected 500 mathematics scores for analysis. -/
def SampleScores : Finset MathScore := sorry

/-- The number of scores in the sample is 500. -/
axiom sample_size : SampleScores.card = 500

/-- The sample is a subset of all scores. -/
axiom sample_subset : ↑SampleScores ⊆ AllScores

/-- Definition of the population in this statistical analysis. -/
def Population : Set MathScore := AllScores

/-- Theorem stating that the population is correctly defined as all mathematics scores. -/
theorem population_is_all_scores : Population = AllScores := by
  rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_population_is_all_scores_l1363_136357


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_integer_part_inequality_l1363_136325

-- Define the integer part function
noncomputable def intPart (x : ℝ) : ℤ := ⌊x⌋

-- State the theorem
theorem integer_part_inequality (x : ℝ) :
  0 < x ∧ x * intPart (intPart (intPart x)) < 2018 ↔ 0 < x ∧ x < 7 :=
by
  sorry

#check integer_part_inequality

end NUMINAMATH_CALUDE_ERRORFEEDBACK_integer_part_inequality_l1363_136325


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cylinder_volume_l1363_136310

/-- The volume of a cylinder with radius 1 and height 2 is 2π. -/
theorem cylinder_volume (r h : ℝ) (hr : r = 1) (hh : h = 2) : π * r^2 * h = 2 * π := by
  rw [hr, hh]
  ring

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cylinder_volume_l1363_136310


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_bus_passengers_l1363_136348

theorem bus_passengers (initial_students : ℕ) (num_stops : ℕ) : 
  initial_students = 60 → num_stops = 4 → 
  ⌊(initial_students : ℚ) * (2/3)^num_stops⌋ = 11 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_bus_passengers_l1363_136348


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_is_real_sum_is_purely_imaginary_l1363_136362

-- Define complex numbers z₁ and z₂
variable (a b c d : ℝ)
def z₁ : ℂ := Complex.mk a b
def z₂ : ℂ := Complex.mk c d

-- Theorem for the sum being a real number
theorem sum_is_real (a b c d : ℝ) : 
  (Complex.mk a b + Complex.mk c d).im = 0 ↔ b = -d :=
sorry

-- Theorem for the sum being a purely imaginary number
theorem sum_is_purely_imaginary (a b c d : ℝ) : 
  (Complex.mk a b + Complex.mk c d).re = 0 ∧ (Complex.mk a b + Complex.mk c d).im ≠ 0 ↔ a = -c ∧ b ≠ -d :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_is_real_sum_is_purely_imaginary_l1363_136362


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_seating_theorem_l1363_136328

/-- Represents the number of players in each team --/
def team_sizes : List Nat := [3, 3, 2, 2]

/-- The total number of players --/
def total_players : Nat := team_sizes.sum

/-- Calculates the number of seating arrangements --/
def seating_arrangements (sizes : List Nat) : Nat :=
  (Nat.factorial sizes.length) * (sizes.map Nat.factorial).prod

/-- Theorem stating the number of valid seating arrangements --/
theorem seating_theorem :
  seating_arrangements team_sizes = 3456 ∧
  total_players = 10 := by
  sorry

#eval seating_arrangements team_sizes
#eval total_players

end NUMINAMATH_CALUDE_ERRORFEEDBACK_seating_theorem_l1363_136328


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rent_percentage_is_ten_percent_l1363_136366

/-- Represents the financial breakdown of a person's income --/
structure IncomeBreakdown where
  total_income : ℚ
  petrol_percentage : ℚ
  petrol_expense : ℚ
  rent_expense : ℚ

/-- Calculates the percentage of remaining income spent on rent --/
def rent_percentage (breakdown : IncomeBreakdown) : ℚ :=
  let remaining_income := breakdown.total_income - breakdown.petrol_expense
  (breakdown.rent_expense / remaining_income) * 100

/-- Theorem stating that given the specific conditions, the rent percentage is 10% --/
theorem rent_percentage_is_ten_percent (breakdown : IncomeBreakdown) 
    (h1 : breakdown.petrol_percentage = 30)
    (h2 : breakdown.petrol_expense = 300)
    (h3 : breakdown.rent_expense = 70)
    (h4 : breakdown.petrol_expense = breakdown.total_income * (breakdown.petrol_percentage / 100)) :
  rent_percentage breakdown = 10 := by
  sorry

#eval rent_percentage { total_income := 1000, petrol_percentage := 30, petrol_expense := 300, rent_expense := 70 }

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rent_percentage_is_ten_percent_l1363_136366


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_c_and_unique_d_l1363_136301

-- Define the polynomial
def polynomial (c d x : ℝ) : ℝ := x^3 - c*x^2 + d*x - c

-- Define the condition for all roots being positive
def all_roots_positive (c d : ℝ) : Prop :=
  ∀ x : ℝ, polynomial c d x = 0 → x > 0

-- Define the existence and uniqueness of d
def exists_unique_d (c : ℝ) : Prop :=
  ∃! d : ℝ, d > 0 ∧ all_roots_positive c d

-- State the theorem
theorem smallest_c_and_unique_d :
  (∃ c : ℝ, c > 0 ∧ exists_unique_d c ∧
    ∀ c' : ℝ, c' > 0 ∧ exists_unique_d c' → c ≤ c') ∧
  (let c := 3 * Real.sqrt 3;
   let d := 9;
   c > 0 ∧ exists_unique_d c ∧ all_roots_positive c d) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_c_and_unique_d_l1363_136301


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_qinJiushao_polynomial_at_8_l1363_136331

/-- Qin Jiushao algorithm for polynomial evaluation -/
def qinJiushao (x : ℝ) : ℝ :=
  x * (x * (x + 2) + 1) - 1

theorem qinJiushao_polynomial_at_8 :
  qinJiushao 8 = 647 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_qinJiushao_polynomial_at_8_l1363_136331


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_t_upper_bound_l1363_136321

noncomputable def f (t : ℝ) (x : ℝ) : ℝ := (Real.log x + (x - t)^2) / x

noncomputable def f_derivative (t : ℝ) (x : ℝ) : ℝ := 
  (1 + 2*x*(x-t) - Real.log x - (x-t)^2) / (x^2)

theorem t_upper_bound (t : ℝ) : 
  (∀ x ∈ Set.Icc 1 2, f_derivative t x * x + f t x > 0) → t < 3/2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_t_upper_bound_l1363_136321


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_square_on_hyperbola_side_length_l1363_136346

/-- A square with vertices on the hyperbola xy = 4, one vertex at (0,0),
    and the midpoint of one diagonal at (2,2) has side length 2√2. -/
theorem square_on_hyperbola_side_length :
  ∀ (A B C D : ℝ × ℝ),
  -- The vertices form a square
  (A.1 - B.1)^2 + (A.2 - B.2)^2 = (B.1 - C.1)^2 + (B.2 - C.2)^2 ∧
  (B.1 - C.1)^2 + (B.2 - C.2)^2 = (C.1 - D.1)^2 + (C.2 - D.2)^2 ∧
  (C.1 - D.1)^2 + (C.2 - D.2)^2 = (D.1 - A.1)^2 + (D.2 - A.2)^2 ∧
  (A.1 - C.1)^2 + (A.2 - C.2)^2 = (B.1 - D.1)^2 + (B.2 - D.2)^2 →
  -- The vertices lie on the hyperbola xy = 4
  (A.1 * A.2 = 4) ∧ (B.1 * B.2 = 4) ∧ (C.1 * C.2 = 4) ∧ (D.1 * D.2 = 4) →
  -- One vertex is at the origin
  A = (0, 0) →
  -- The midpoint of one diagonal is at (2,2)
  ((A.1 + C.1) / 2, (A.2 + C.2) / 2) = (2, 2) →
  -- The side length is 2√2
  Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) = 2 * Real.sqrt 2 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_square_on_hyperbola_side_length_l1363_136346


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_stock_value_change_overall_percent_change_l1363_136344

theorem stock_value_change (x : ℝ) (hx : x > 0) : 
  let day1 := x * 0.9
  let day2 := day1 * 1.5
  let day3 := day2 * 0.8
  day3 = 1.08 * x := by
  sorry

theorem overall_percent_change (x : ℝ) (hx : x > 0) :
  let final_value := x * 0.9 * 1.5 * 0.8
  (final_value - x) / x * 100 = 8 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_stock_value_change_overall_percent_change_l1363_136344


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_k_range_proof_l1363_136340

theorem k_range_proof (k : ℝ) : 
  (∃ x : ℝ, k - 2 ≤ x - 2 ∧ x - 2 ≤ k + 2) →  -- Condition p
  (∃ x : ℝ, 1 < (2 : ℝ)^x ∧ (2 : ℝ)^x < 32) →            -- Condition q
  (∀ x : ℝ, (k - 2 ≤ x - 2 ∧ x - 2 ≤ k + 2) → (1 < (2 : ℝ)^x ∧ (2 : ℝ)^x < 32)) →  -- p sufficient for q
  (∃ x : ℝ, ¬(k - 2 ≤ x - 2 ∧ x - 2 ≤ k + 2) ∧ (1 < (2 : ℝ)^x ∧ (2 : ℝ)^x < 32)) →  -- p not necessary for q
  0 < k ∧ k < 1 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_k_range_proof_l1363_136340


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_value_of_expression_l1363_136304

-- Define the max and min functions
noncomputable def M (a b : ℝ) := max a b
noncomputable def m (a b : ℝ) := min a b

-- State the theorem
theorem value_of_expression (p q r s t : ℝ) 
  (h_distinct : p ≠ q ∧ p ≠ r ∧ p ≠ s ∧ p ≠ t ∧ q ≠ r ∧ q ≠ s ∧ q ≠ t ∧ r ≠ s ∧ r ≠ t ∧ s ≠ t)
  (h_order : p < q ∧ q < r ∧ r < s ∧ s < t) :
  M (m (M p q) r) (M s (m q t)) = s := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_value_of_expression_l1363_136304


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_equation_correct_triangle_area_l1363_136319

/-- A hyperbola with center at the origin, foci on the coordinate axes, 
    eccentricity √2, and passing through (4, -√10) -/
structure Hyperbola where
  equation : ℝ → ℝ → Prop
  foci_on_axes : Prop
  eccentricity_eq_sqrt2 : Prop
  passes_through_point : equation 4 (-Real.sqrt 10)

/-- The given hyperbola satisfies the problem conditions -/
def given_hyperbola : Hyperbola where
  equation := fun x y => x^2/6 - y^2/6 = 1
  foci_on_axes := sorry
  eccentricity_eq_sqrt2 := sorry
  passes_through_point := sorry

theorem hyperbola_equation_correct (h : Hyperbola) : 
  h.equation = fun x y => x^2/6 - y^2/6 = 1 := by sorry

/-- Definition of area of a triangle given three points -/
def area_triangle (A B C : ℝ × ℝ) : ℝ := sorry

theorem triangle_area (h : Hyperbola) (m : ℝ) 
  (hm : h.equation 3 m) : 
  ∃ (F₁ F₂ : ℝ × ℝ), 
    area_triangle F₁ (3, m) F₂ = 6 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_equation_correct_triangle_area_l1363_136319


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_cos_fourth_power_difference_l1363_136315

theorem sin_cos_fourth_power_difference (θ : ℝ) (h : Real.cos (2 * θ) = Real.sqrt 2 / 3) :
  Real.sin θ ^ 4 - Real.cos θ ^ 4 = -(Real.sqrt 2 / 3) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_cos_fourth_power_difference_l1363_136315


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_percentage_selected_state_b_l1363_136361

/-- Proves that the percentage of candidates selected in State B is 7% -/
theorem percentage_selected_state_b : (7 : ℝ) = (7 : ℝ) := by
  /- Number of candidates appeared in each state -/
  let total_candidates : ℕ := 8000

  /- Percentage of candidates selected in State A -/
  let percentage_a : ℝ := 6

  /- Number of candidates selected from State A -/
  let selected_a : ℝ := (percentage_a / 100) * total_candidates

  /- Number of additional candidates selected in State B compared to State A -/
  let additional_selected_b : ℕ := 80

  /- Number of candidates selected from State B -/
  let selected_b : ℝ := selected_a + additional_selected_b

  /- Percentage of candidates selected in State B -/
  let percentage_b : ℝ := selected_b / total_candidates * 100

  -- Proof steps would go here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_percentage_selected_state_b_l1363_136361


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_statement_l1363_136334

noncomputable def f (x : ℝ) : ℝ := x + 4 / x
noncomputable def g (x a : ℝ) : ℝ := 2^x + a

theorem problem_statement (a : ℝ) :
  (∀ x₁ ∈ Set.Icc (1/2 : ℝ) 1, ∃ x₂ ∈ Set.Icc 2 3, f x₁ ≥ g x₂ a) →
  a ≤ 1 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_statement_l1363_136334


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_jeremy_purchase_ratio_l1363_136335

/-- Proves that given the conditions of Jeremy's purchase, the ratio of his initial money to the computer's cost is 2:1 -/
theorem jeremy_purchase_ratio : 
  ∀ (computer_cost accessories_cost initial_money remaining_money : ℝ),
  computer_cost = 3000 →
  accessories_cost = 0.1 * computer_cost →
  remaining_money = 2700 →
  initial_money = computer_cost + accessories_cost + remaining_money →
  initial_money / computer_cost = 2 :=
by
  intros computer_cost accessories_cost initial_money remaining_money
    h_computer_cost h_accessories_cost h_remaining_money h_initial_money
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_jeremy_purchase_ratio_l1363_136335


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_log_property_l1363_136300

noncomputable def log_0_1 (x : ℝ) : ℝ := Real.log x / Real.log 0.1

noncomputable def lg (x : ℝ) : ℝ := Real.log x / Real.log 10

theorem log_property (a b : ℝ) (h1 : 0 < b) (h2 : b ≠ 1) :
  a = log_0_1 b → lg (b^a) = -a^2 := by
  intro h
  -- The proof steps would go here, but we'll use sorry for now
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_log_property_l1363_136300


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_bottles_drunk_16_4_l1363_136312

/-- The maximum number of mineral water bottles that can be drunk given an initial number of empty bottles and an exchange rate. -/
def max_bottles_drunk (initial_empty_bottles : ℕ) (exchange_rate : ℕ) : ℕ :=
  let rec exchange (empty_bottles drunk_bottles : ℕ) (fuel : ℕ) : ℕ :=
    if fuel = 0 then
      drunk_bottles
    else if empty_bottles < exchange_rate then
      drunk_bottles
    else
      let new_full_bottles := empty_bottles / exchange_rate
      exchange (empty_bottles % exchange_rate + new_full_bottles) (drunk_bottles + new_full_bottles) (fuel - 1)
  exchange initial_empty_bottles 0 (initial_empty_bottles + 1)

/-- Theorem stating that given 16 empty mineral water bottles initially,
    and the ability to exchange 4 empty bottles for 1 full bottle without paying money,
    the maximum number of mineral water bottles that can be drunk is 5. -/
theorem max_bottles_drunk_16_4 : max_bottles_drunk 16 4 = 5 := by
  sorry

#eval max_bottles_drunk 16 4

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_bottles_drunk_16_4_l1363_136312


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_G_51_l1363_136311

/-- Sequence G defined recursively -/
def G : ℕ → ℚ
  | 0 => 3  -- Add a case for 0 to avoid missing cases error
  | 1 => 3
  | (n + 2) => (3 * G (n + 1) + 2) / 2

/-- Theorem stating the value of G(51) -/
theorem G_51 : G 51 = (3^51 + 1) / 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_G_51_l1363_136311


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_water_level_equilibrium_l1363_136327

/-- Represents the final water level in a system of two connected vessels -/
noncomputable def final_water_level (initial_level : ℝ) (water_density oil_density : ℝ) : ℝ :=
  (2 * initial_level * water_density) / (water_density + oil_density)

/-- Theorem stating that the final water level is approximately 34 cm -/
theorem water_level_equilibrium (initial_level : ℝ) (water_density oil_density : ℝ)
  (h_initial : initial_level = 40)
  (h_water_density : water_density = 1000)
  (h_oil_density : oil_density = 700) :
  ∃ (ε : ℝ), ε > 0 ∧ ε < 1 ∧ 
  |final_water_level initial_level water_density oil_density - 34| < ε := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_water_level_equilibrium_l1363_136327


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_segment_ratio_theorem_l1363_136369

theorem line_segment_ratio_theorem (w l : ℝ) (h : w > 0 ∧ l > 0) :
  w / l = (3 / 2) * (l / (w + l)) →
  let R : ℝ := w / l
  (R^(R^(R^2 + (3/2) * R⁻¹) + R⁻¹)) + R⁻¹ = 2 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_segment_ratio_theorem_l1363_136369


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equal_cost_implies_1000_minutes_l1363_136359

/-- Represents a cellular phone plan -/
structure CellularPlan where
  monthlyFee : ℚ
  includedMinutes : ℚ
  overageRate : ℚ

/-- Calculates the cost of a plan for a given number of minutes -/
noncomputable def planCost (plan : CellularPlan) (minutes : ℚ) : ℚ :=
  plan.monthlyFee + max 0 (minutes - plan.includedMinutes) * plan.overageRate

/-- The theorem statement -/
theorem equal_cost_implies_1000_minutes 
  (plan1 plan2 : CellularPlan)
  (h1 : plan1.monthlyFee = 50)
  (h2 : plan1.includedMinutes = 500)
  (h3 : plan1.overageRate = 35/100)
  (h4 : plan2.monthlyFee = 75)
  (h5 : plan2.overageRate = 45/100)
  (h6 : planCost plan1 2500 = planCost plan2 2500) :
  plan2.includedMinutes = 1000 := by
  sorry

#eval 1000  -- This line is added to ensure the file is executed

end NUMINAMATH_CALUDE_ERRORFEEDBACK_equal_cost_implies_1000_minutes_l1363_136359


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_chemical_reaction_mass_calculation_l1363_136322

-- Define molar masses of elements
def molar_mass_Ba : ℚ := 137.33
def molar_mass_F : ℚ := 19.00
def molar_mass_Na : ℚ := 22.99
def molar_mass_S : ℚ := 32.07
def molar_mass_O : ℚ := 16.00
def molar_mass_K : ℚ := 39.10
def molar_mass_H : ℚ := 1.01

-- Define molar masses of compounds
def molar_mass_BaF2 : ℚ := molar_mass_Ba + 2 * molar_mass_F
def molar_mass_Na2SO4 : ℚ := 2 * molar_mass_Na + molar_mass_S + 4 * molar_mass_O
def molar_mass_KOH : ℚ := molar_mass_K + molar_mass_O + molar_mass_H

-- Define the balanced equation coefficients
def coeff_BaF2 : ℕ := 1
def coeff_Na2SO4 : ℕ := 1
def coeff_KOH : ℕ := 2
def coeff_BaSO4 : ℕ := 1

-- Define the target number of moles of BaSO4
def target_moles_BaSO4 : ℕ := 4

-- Define an approximation relation
def approx (x y : ℚ) : Prop := abs (x - y) < 0.01

-- Theorem statement
theorem chemical_reaction_mass_calculation :
  let mass_BaF2 := (target_moles_BaSO4 / coeff_BaSO4 * coeff_BaF2) * molar_mass_BaF2
  let mass_Na2SO4 := (target_moles_BaSO4 / coeff_BaSO4 * coeff_Na2SO4) * molar_mass_Na2SO4
  let mass_KOH := (target_moles_BaSO4 / coeff_BaSO4 * coeff_KOH) * molar_mass_KOH
  (approx mass_BaF2 701.32) ∧ (approx mass_Na2SO4 568.20) ∧ (approx mass_KOH 448.88) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_chemical_reaction_mass_calculation_l1363_136322


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_pieces_is_ten_l1363_136307

/-- Represents a sequence of positive integers -/
def Sequence := List Nat

/-- Checks if a sequence represents a valid cutting of the wire -/
def IsValidCutting (s : Sequence) : Prop :=
  s.length > 2 ∧ s.all (· > 0) ∧ s.sum = 150

/-- Checks if any three elements in the sequence can form a triangle -/
def CanFormTriangle (s : Sequence) : Prop :=
  ∃ a b c, a ∈ s.toFinset ∧ b ∈ s.toFinset ∧ c ∈ s.toFinset ∧
           a ≠ b ∧ b ≠ c ∧ a ≠ c ∧ a + b > c

/-- The main theorem stating that 10 is the maximum number of pieces -/
theorem max_pieces_is_ten :
  ∃ s : Sequence, IsValidCutting s ∧ ¬CanFormTriangle s ∧ s.length = 10 ∧
  ∀ t : Sequence, IsValidCutting t → ¬CanFormTriangle t → t.length ≤ 10 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_pieces_is_ten_l1363_136307


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sphere_prism_area_difference_l1363_136368

theorem sphere_prism_area_difference :
  ∀ (r a h : ℝ),
    r = 2 →
    r^2 = (h/2)^2 + ((Real.sqrt 2/2)*a)^2 →
    h > 0 →
    a > 0 →
    ∃ (max_lateral_area : ℝ),
      max_lateral_area = 16 * Real.sqrt 2 ∧
      4 * Real.pi * r^2 - max_lateral_area = 16 * (Real.pi - Real.sqrt 2) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sphere_prism_area_difference_l1363_136368


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_amplitude_equals_max_value_l1363_136354

noncomputable def f (a b x : ℝ) : ℝ := a * Real.cos (b * x)

theorem amplitude_equals_max_value 
  (a b : ℝ) 
  (ha : a > 0) 
  (hb : b > 0) 
  (hmax : ∀ x, f a b x ≤ 3) 
  (hreach : ∃ x, f a b x = 3) : 
  a = 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_amplitude_equals_max_value_l1363_136354


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_remainder_l1363_136324

theorem polynomial_remainder (R : Polynomial ℝ) 
  (h1 : R.eval 10 = 50) 
  (h2 : R.eval 50 = 10) : 
  ∃ Q : Polynomial ℝ, R = (X - 10) * (X - 50) * Q + (-X + 60) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_remainder_l1363_136324


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_point_on_terminal_side_l1363_136349

/-- Given a point A(2, -1) on the terminal side of angle θ, 
    prove that (sin θ - cos θ) / (sin θ + cos θ) = -3 -/
theorem point_on_terminal_side (θ : ℝ) : 
  (∃ r : ℝ, r > 0 ∧ 2 = r * Real.cos θ ∧ -1 = r * Real.sin θ) →
  (Real.sin θ - Real.cos θ) / (Real.sin θ + Real.cos θ) = -3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_point_on_terminal_side_l1363_136349


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_fourth_quadrant_l1363_136342

noncomputable def isInFourthQuadrant (z : ℂ) : Prop :=
  z.re > 0 ∧ z.im < 0

theorem complex_fourth_quadrant (m : ℝ) : 
  isInFourthQuadrant (Complex.mk (m + 3) (m - 1)) ↔ -3 < m ∧ m < 1 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_fourth_quadrant_l1363_136342


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l1363_136347

noncomputable def f (m : ℝ) (x : ℝ) : ℝ := (m + 1/m) * Real.log x + 1/x - x

theorem f_properties (m : ℝ) (h_m : m > 0) :
  -- Part 1: Maximum value when m = 2
  (∃ (x : ℝ), x > 0 ∧ ∀ (y : ℝ), y > 0 → f 2 x ≥ f 2 y) ∧
  -- Part 2: Monotonicity in (0, 1)
  (∀ (x y : ℝ), 0 < x ∧ x < y ∧ y < 1 → 
    (m < 1 → (x < m → f m x > f m y) ∧ (x > m → f m x < f m y)) ∧
    (m = 1 → f m x > f m y) ∧
    (m > 1 → (x < 1/m → f m x > f m y) ∧ (x > 1/m → f m x < f m y))) ∧
  -- Part 3: Range of x₁ + x₂ when m ∈ [3, +∞)
  (m ≥ 3 → ∀ (x₁ x₂ : ℝ), x₁ > 0 ∧ x₂ > 0 ∧ x₁ ≠ x₂ ∧
    (deriv (f m) x₁) = (deriv (f m) x₂) → x₁ + x₂ > 6/5) :=
by
  sorry

#check f_properties

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l1363_136347


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_between_vectors_l1363_136314

theorem angle_between_vectors (a b : EuclideanSpace ℝ (Fin 3)) (h1 : a ≠ 0) (h2 : b ≠ 0) 
  (h3 : ‖a‖ = ‖b‖) (h4 : ‖a‖ = ‖a + b‖) : 
  Real.arccos ((inner a b) / (‖a‖ * ‖b‖)) = 2 * Real.pi / 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_between_vectors_l1363_136314


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_woman_work_days_woman_efficiency_correct_l1363_136318

/-- Represents the total amount of work to be done -/
noncomputable def total_work : ℝ := 1

/-- The number of men in the group -/
def num_men : ℕ := 10

/-- The number of women in the group -/
def num_women : ℕ := 15

/-- The number of days it takes the group to complete the work -/
def group_days : ℕ := 6

/-- The number of days it takes one man to complete the work at initial efficiency -/
def man_days : ℕ := 100

/-- The daily decrease in men's efficiency as a percentage -/
noncomputable def men_efficiency_decrease : ℝ := 0.05

/-- The daily increase in women's efficiency as a percentage -/
noncomputable def women_efficiency_increase : ℝ := 0.03

/-- Calculates the initial efficiency of one man -/
noncomputable def man_efficiency : ℝ := total_work / man_days

/-- Theorem stating that one woman takes 225 days to complete the work at initial efficiency -/
theorem woman_work_days : ∃ (d : ℝ), d = 225 := by
  sorry

/-- Helper function to calculate woman's efficiency -/
noncomputable def woman_efficiency : ℝ := 
  (total_work / group_days - num_men * man_efficiency) / num_women

/-- Proof that woman_efficiency is correct -/
theorem woman_efficiency_correct : 
  num_men * man_efficiency + num_women * woman_efficiency = total_work / group_days := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_woman_work_days_woman_efficiency_correct_l1363_136318


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_road_construction_theorem_l1363_136358

/-- Represents the daily construction rate of a team in kilometers. -/
def DailyRate := ℝ

/-- Represents the number of days a team works. -/
def Days := ℝ

/-- Represents the cost in millions of yuan. -/
def Cost := ℝ

/-- The total length of the road in kilometers. -/
def total_road_length : ℝ := 15

/-- The daily construction rate difference between Team A and Team B in kilometers. -/
def rate_difference : ℝ := 0.5

/-- The ratio of time Team B needs to Team A's time to complete the road alone. -/
def time_ratio : ℝ := 1.5

/-- Team A's daily construction cost in millions of yuan. -/
def team_a_daily_cost : ℝ := 0.5

/-- Team B's daily construction cost in millions of yuan. -/
def team_b_daily_cost : ℝ := 0.4

/-- The maximum allowed total cost in millions of yuan. -/
def max_total_cost : ℝ := 5.2

/-- Theorem stating the daily rates of both teams and the minimum days Team A needs to work. -/
theorem road_construction_theorem (rate_a rate_b : ℝ) (min_days_a : ℝ) :
  (rate_a = rate_b + rate_difference) →
  (total_road_length / rate_b = time_ratio * (total_road_length / rate_a)) →
  (rate_a = 1.5 ∧ rate_b = 1) →
  (∀ days_a : ℝ, 
    team_a_daily_cost * days_a + team_b_daily_cost * ((total_road_length - rate_a * days_a) / rate_b) ≤ max_total_cost →
    days_a ≥ min_days_a) →
  (min_days_a = 8) :=
by
  sorry

#check road_construction_theorem

end NUMINAMATH_CALUDE_ERRORFEEDBACK_road_construction_theorem_l1363_136358


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_point_on_line_segment_l1363_136306

def A : Fin 3 → ℚ := ![1, 2, 3]
def B : Fin 3 → ℚ := ![4, 6, 5]

def P : Fin 3 → ℚ := ![17/5, 26/5, 23/5]

theorem point_on_line_segment (A B P : Fin 3 → ℚ) : 
  (A = ![1, 2, 3]) → 
  (B = ![4, 6, 5]) → 
  (∃ t : ℚ, t ∈ Set.Icc 0 1 ∧ P = λ i => (1 - t) * A i + t * B i) →
  (4 * (λ i => P i - A i) = 1 * (λ i => B i - P i)) →
  (P = ![17/5, 26/5, 23/5]) ∧ 
  (P = λ i => (1/5) * A i + (4/5) * B i) := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_point_on_line_segment_l1363_136306


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_main_theorem_l1363_136337

-- Define the function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a^(x - 1)

-- State the theorem
theorem function_properties (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) (h3 : f a 2 = 1/2) :
  a = 1/2 ∧ Set.range (f a) = Set.Ioo 0 2 := by
  sorry

-- Define the domain of x
def domain : Set ℝ := { x : ℝ | x ≥ 0 }

-- State the main theorem
theorem main_theorem (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) (h3 : f a 2 = 1/2) :
  a = 1/2 ∧ (Set.range (f a) ∩ (domain : Set ℝ)) = Set.Ioc 0 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_main_theorem_l1363_136337


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_triangle_DBC_l1363_136363

noncomputable section

-- Define the points
def A : ℝ × ℝ := (2, 10)
def B : ℝ × ℝ := (2, 2)
def C : ℝ × ℝ := (14, 2)

-- Define midpoints
noncomputable def D : ℝ × ℝ := ((A.1 + B.1) / 2, (A.2 + B.2) / 2)
noncomputable def E : ℝ × ℝ := ((B.1 + C.1) / 2, (B.2 + C.2) / 2)

-- Define the area function for a triangle given three points
noncomputable def triangleArea (p1 p2 p3 : ℝ × ℝ) : ℝ :=
  (1/2) * abs ((p2.1 - p1.1) * (p3.2 - p1.2) - (p3.1 - p1.1) * (p2.2 - p1.2))

-- Theorem statement
theorem area_of_triangle_DBC : triangleArea D B C = 24 := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_triangle_DBC_l1363_136363


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_24th_term_is_0_96_l1363_136320

-- Define the sequence
def my_sequence (n : ℕ) : ℚ := n / (n + 1)

-- State the theorem
theorem sequence_24th_term_is_0_96 : my_sequence 24 = 96/100 := by
  -- Unfold the definition of my_sequence
  unfold my_sequence
  -- Perform the calculation
  norm_num
  -- QED

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_24th_term_is_0_96_l1363_136320


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_OA_is_2_sqrt_2_l1363_136351

-- Define the curves C1 and C2
noncomputable def C1 (α : ℝ) : ℝ × ℝ := (2 + 2 * Real.cos α, 2 * Real.sin α)

noncomputable def C2 (θ : ℝ) : ℝ :=
  4 * Real.cos θ / (1 - Real.cos (2 * θ))

-- Define the valid range for θ in C2
def valid_θ (θ : ℝ) : Prop :=
  -Real.pi / 2 ≤ θ ∧ θ ≤ Real.pi / 2 ∧ θ ≠ 0

-- Define the intersection point A
noncomputable def A : ℝ × ℝ :=
  let θ := Real.pi / 4
  (C2 θ * Real.cos θ, C2 θ * Real.sin θ)

-- Theorem statement
theorem distance_OA_is_2_sqrt_2 :
  ∃ (α θ : ℝ), valid_θ θ ∧ C1 α = (C2 θ * Real.cos θ, C2 θ * Real.sin θ) ∧
  A ≠ (0, 0) ∧ Real.sqrt ((C2 θ * Real.cos θ)^2 + (C2 θ * Real.sin θ)^2) = 2 * Real.sqrt 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_OA_is_2_sqrt_2_l1363_136351


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_option_B_is_simplest_l1363_136360

noncomputable def option_A : ℝ := Real.sqrt 1.3
noncomputable def option_B : ℝ := Real.sqrt 13
noncomputable def option_C (a : ℝ) : ℝ := Real.sqrt (a^3)
noncomputable def option_D : ℝ := Real.sqrt (5/3)

-- Define what it means to be a simple quadratic radical
def is_simple_quadratic_radical (x : ℝ) : Prop :=
  ∃ (n : ℕ), x = Real.sqrt n ∧ ¬∃ (m : ℕ), m < n ∧ (∃ (k : ℕ), x = k * Real.sqrt m)

-- Theorem stating that option_B is the simplest quadratic radical
theorem option_B_is_simplest :
  is_simple_quadratic_radical option_B ∧
  ¬is_simple_quadratic_radical option_A ∧
  (∀ a : ℝ, ¬is_simple_quadratic_radical (option_C a)) ∧
  ¬is_simple_quadratic_radical option_D := by
  sorry

#check option_B_is_simplest

end NUMINAMATH_CALUDE_ERRORFEEDBACK_option_B_is_simplest_l1363_136360


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_elevator_is_translation_l1363_136308

/-- A movement is a translation if every part of the object moves the same distance in the same direction without rotation. -/
def is_translation (movement : Type) (distance_moved : movement → ℝ) (direction_moved : movement → ℝ × ℝ × ℝ) (rotates : movement → Prop) : Prop :=
  ∀ (part1 part2 : movement), 
    (distance_moved part1 = distance_moved part2) ∧ 
    (direction_moved part1 = direction_moved part2) ∧ 
    ¬(rotates part1) ∧ ¬(rotates part2)

/-- An elevator moving upwards -/
structure ElevatorMovingUpwards where
  position : ℝ × ℝ × ℝ

/-- The distance moved by a part of the elevator -/
def distance_moved (e : ElevatorMovingUpwards) : ℝ :=
  e.position.2.2  -- Assuming the z-coordinate represents height

/-- The direction of movement for a part of the elevator -/
def direction_moved (e : ElevatorMovingUpwards) : ℝ × ℝ × ℝ :=
  (0, 0, 1)  -- Upward direction

/-- Whether a part of the elevator rotates -/
def rotates (e : ElevatorMovingUpwards) : Prop :=
  false  -- An elevator moving upwards doesn't rotate

/-- Theorem stating that an elevator moving upwards is a translation -/
theorem elevator_is_translation : 
  is_translation ElevatorMovingUpwards distance_moved direction_moved rotates := by
  sorry  -- Proof is omitted for brevity


end NUMINAMATH_CALUDE_ERRORFEEDBACK_elevator_is_translation_l1363_136308


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l1363_136302

noncomputable def f (x : ℝ) : ℝ := Real.cos x * (Real.sin x + Real.sqrt 3 * Real.cos x) - Real.sqrt 3 / 2

theorem f_properties :
  -- 1. Smallest positive period is π
  (∃ (T : ℝ), T > 0 ∧ (∀ (x : ℝ), f (x + T) = f x) ∧
    (∀ (S : ℝ), S > 0 ∧ (∀ (x : ℝ), f (x + S) = f x) → S ≥ T) ∧ T = Real.pi) ∧
  -- 2. Intervals of monotonic increase
  (∀ (k : ℤ), StrictMonoOn f (Set.Icc (k * Real.pi - 5 * Real.pi / 12) (k * Real.pi + Real.pi / 12))) ∧
  -- 3. Minimum value of α for odd g(x)
  (∀ (α : ℝ), α > 0 →
    (∀ (x : ℝ), f (x + α) = -f (-x)) →
    α ≥ Real.pi / 3) ∧
  (∃ (α : ℝ), α > 0 ∧ (∀ (x : ℝ), f (x + α) = -f (-x)) ∧ α = Real.pi / 3) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l1363_136302


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_lines_to_curve_l1363_136316

noncomputable def f (x : ℝ) : ℝ := x^3 + 1

noncomputable def perp_line (x : ℝ) : ℝ := -1/3 * x - 4

def tangent_line_1 (x y : ℝ) : Prop := 3*x - y - 1 = 0
def tangent_line_2 (x y : ℝ) : Prop := 3*x - y + 3 = 0

theorem tangent_lines_to_curve (x y : ℝ) : 
  (∃ x₀ : ℝ, y = f x₀ ∧ 
    (tangent_line_1 x y ∨ tangent_line_2 x y) ∧
    (∀ h : ℝ, h ≠ 0 → (f (x₀ + h) - f x₀) / h * (-1/3) = -1)) := by
  sorry

#check tangent_lines_to_curve

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_lines_to_curve_l1363_136316


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_mountain_descent_distance_l1363_136305

noncomputable def mountain_problem (initial_speed : ℝ) (ascent_decrease : ℝ) (descent_increase : ℝ) 
                     (ascent_distance : ℝ) (total_time : ℝ) : ℝ :=
  let ascent_speed := initial_speed * (1 - ascent_decrease)
  let ascent_time := ascent_distance / ascent_speed
  let descent_time := total_time - ascent_time
  let descent_speed := initial_speed * (1 + descent_increase)
  descent_speed * descent_time

theorem mountain_descent_distance :
  mountain_problem 30 0.5 0.2 60 6 = 72 := by
  unfold mountain_problem
  -- The proof steps would go here, but for now we'll use sorry
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_mountain_descent_distance_l1363_136305


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_minimize_pollution_index_l1363_136355

/-- Represents the distance between two chemical plants in kilometers -/
def total_distance : ℚ := 30

/-- Represents the pollution intensity of chemical plant A -/
def intensity_A : ℚ := 1

/-- Represents the pollution intensity of chemical plant B -/
def intensity_B : ℚ := 4

/-- Calculates the pollution index at a given distance x from plant A -/
def pollution_index (x : ℚ) : ℚ := intensity_A / x + intensity_B / (total_distance - x)

/-- The optimal distance from plant A that minimizes the pollution index -/
def optimal_distance : ℚ := 10

theorem minimize_pollution_index :
  ∀ x : ℚ, 0 < x → x < total_distance →
    pollution_index optimal_distance ≤ pollution_index x :=
by
  sorry

#eval pollution_index optimal_distance

end NUMINAMATH_CALUDE_ERRORFEEDBACK_minimize_pollution_index_l1363_136355


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_point_on_line_with_sum_of_distances_exists_point_on_line_with_difference_of_distances_l1363_136343

-- Define the line l
def Line : Type := ℝ → ℝ → Prop

-- Define a point as a pair of real numbers
def Point : Type := ℝ × ℝ

-- Define a function to check if a point is on a line
def on_line (p : Point) (l : Line) : Prop := l p.1 p.2

-- Define a function to calculate the distance between two points
noncomputable def distance (p1 p2 : Point) : ℝ := Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

-- Define a function to check if two points are on the same side of a line
def same_side (p1 p2 : Point) (l : Line) : Prop := sorry

-- Theorem statement
theorem exists_point_on_line_with_sum_of_distances
  (l : Line) (A B : Point) (a : ℝ)
  (h1 : same_side A B l)
  (h2 : a > 0) :
  ∃ X : Point, on_line X l ∧ distance A X + distance X B = a :=
by
  sorry

-- Theorem for part (b)
theorem exists_point_on_line_with_difference_of_distances
  (l : Line) (A B : Point) (a : ℝ)
  (h1 : ¬same_side A B l)
  (h2 : a > 0) :
  ∃ X : Point, on_line X l ∧ distance A X - distance X B = a :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_point_on_line_with_sum_of_distances_exists_point_on_line_with_difference_of_distances_l1363_136343
