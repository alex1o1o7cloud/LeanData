import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_f_l1038_103889

-- Define the function f
noncomputable def f : ℝ → ℝ := fun x => 
  if x ≤ 0 then Real.exp x - 1 else -Real.exp (-x) + 1

-- State the theorem
theorem range_of_f :
  (∀ x, f (-x) = -f x) → -- f is odd
  Set.range f = Set.Ioo (-1) 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_f_l1038_103889


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_diamond_calculation_l1038_103872

/-- The diamond operation for real numbers -/
noncomputable def diamond (a b : ℝ) : ℝ := Real.sqrt (a^2 + b^2)

/-- Theorem stating that (3 ⋄ 4) ⋄ (7 ⋄ 24) = 5√26 -/
theorem diamond_calculation : diamond (diamond 3 4) (diamond 7 24) = 5 * Real.sqrt 26 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_diamond_calculation_l1038_103872


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_property_l1038_103895

/-- 
Given four sequences of natural numbers a, b, c, d, and their union sequence u,
this theorem states that if the first term a₁ is even, then a, b, c, d are 
arithmetic progressions and u satisfies the condition that each term after u₁
is the sum of the previous term and its last digit.
-/
theorem sequence_property (a b c d : ℕ → ℕ) (u : ℕ → ℕ) (h_even : Even (a 0)) :
  (∀ k, u (4*k) = a k) ∧ 
  (∀ k, u (4*k + 1) = b k) ∧ 
  (∀ k, u (4*k + 2) = c k) ∧ 
  (∀ k, u (4*k + 3) = d k) ∧ 
  (∀ n, n > 1 → u n = u (n-1) + (u (n-1) % 10)) →
  ∃ da db dc dd : ℕ, 
    (∀ k, a (k+1) = a k + da) ∧
    (∀ k, b (k+1) = b k + db) ∧
    (∀ k, c (k+1) = c k + dc) ∧
    (∀ k, d (k+1) = d k + dd) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_property_l1038_103895


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_jaylen_total_vegetables_l1038_103817

/-- The number of vegetables Jaylen has -/
def jaylen_vegetables (carrots cucumbers bell_peppers green_beans : ℕ) : ℕ :=
  carrots + cucumbers + bell_peppers + green_beans

/-- The number of bell peppers Jaylen has -/
def jaylen_bell_peppers (kristin_bell_peppers : ℕ) : ℕ :=
  2 * kristin_bell_peppers

/-- The number of green beans Jaylen has -/
def jaylen_green_beans (kristin_green_beans : ℕ) : ℕ :=
  kristin_green_beans / 2 - 3

theorem jaylen_total_vegetables :
  jaylen_vegetables 5 2 (jaylen_bell_peppers 2) (jaylen_green_beans 20) = 18 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_jaylen_total_vegetables_l1038_103817


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_set_operations_l1038_103839

def U : Set ℝ := Set.univ

def A : Set ℝ := {x | 1 ≤ x ∧ x ≤ 3 ∨ 4 < x ∧ x < 6}

def B : Set ℝ := {x | 2 ≤ x ∧ x < 5}

theorem set_operations :
  (Aᶜ = {x : ℝ | x < 1 ∨ 3 < x ∧ x ≤ 4 ∨ x ≥ 6}) ∧
  (Bᶜ = {x : ℝ | x < 2 ∨ x ≥ 5}) ∧
  (A ∩ Bᶜ = {x : ℝ | 1 ≤ x ∧ x < 2 ∨ 5 ≤ x ∧ x < 6}) ∧
  (Aᶜ ∪ B = {x : ℝ | x < 1 ∨ 2 ≤ x ∧ x < 5 ∨ x ≥ 6}) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_set_operations_l1038_103839


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_ratio_when_dot_product_maximized_l1038_103884

-- Define the triangle ABC
def triangle (A B C : Real) (a b c : Real) : Prop :=
  A + B + C = Real.pi ∧ a > 0 ∧ b > 0 ∧ c > 0

-- Define the conditions
def conditions (A B C : Real) (a b c : Real) : Prop :=
  triangle A B C a b c ∧ C = Real.pi/3 ∧ c = 2

-- Define the dot product of AC and AB
noncomputable def dot_product (A B C : Real) (a b c : Real) : Real :=
  b * c * Real.cos A

-- Theorem statement
theorem triangle_ratio_when_dot_product_maximized
  (A B C : Real) (a b c : Real)
  (h : conditions A B C a b c)
  (h_max : ∀ (A' B' : Real), conditions A' B' C a b c → 
           dot_product A' B' C a b c ≤ dot_product A B C a b c) :
  b / a = 2 + Real.sqrt 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_ratio_when_dot_product_maximized_l1038_103884


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_conic_intersection_bound_l1038_103811

-- Define the conic equation
noncomputable def ConicEquation (a b c d e f : ℝ) (x y : ℝ) : Prop :=
  a * x^2 + 2 * b * x * y + c * y^2 + 2 * d * x + 2 * e * y = f

-- Define the rational parametrization
noncomputable def RationalParametrization (P Q A : ℝ → ℝ) (t : ℝ) : ℝ × ℝ :=
  (P t / A t, Q t / A t)

-- Define the intersection polynomial
noncomputable def IntersectionPolynomial (a b c d e f : ℝ) (P Q A : ℝ → ℝ) (t : ℝ) : ℝ :=
  a * (P t)^2 + 2 * b * (P t) * (Q t) + c * (Q t)^2 + 
  2 * d * (P t) * (A t) + 2 * e * (Q t) * (A t) - f * (A t)^2

-- State the theorem
theorem conic_intersection_bound 
  (a b c d e f : ℝ) (P Q A : ℝ → ℝ) :
  ∃ (n : ℕ), n ≤ 4 ∧ 
  (∀ t : ℝ, IntersectionPolynomial a b c d e f P Q A t = 0) → 
  (∃ (S : Finset ℝ), Finset.card S = n ∧ 
    ∀ t ∈ S, ConicEquation a b c d e f (P t / A t) (Q t / A t)) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_conic_intersection_bound_l1038_103811


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_range_l1038_103882

-- Define the function
noncomputable def f (x : ℝ) : ℝ := x - Real.sqrt (x + 1)

-- State the theorem
theorem f_range : Set.range f = Set.Ici (-5/4) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_range_l1038_103882


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unknown_number_proof_l1038_103851

theorem unknown_number_proof (a b : ℕ) (h1 : a * 4 = b * 3) (h2 : Nat.lcm a b = 180) (h3 : b = 60) : a = 45 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unknown_number_proof_l1038_103851


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_A_l1038_103898

theorem min_value_A (x y z : ℝ) (hx : x ∈ Set.Ioo 0 1) (hy : y ∈ Set.Ioo 0 1) (hz : z ∈ Set.Ioo 0 1) :
  let A := ((x + 2*y) * Real.sqrt (x + y - x*y) + 
            (y + 2*z) * Real.sqrt (y + z - y*z) + 
            (z + 2*x) * Real.sqrt (z + x - z*x)) / (x*y + y*z + z*x)
  A ≥ 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_A_l1038_103898


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_mixed_pairs_count_l1038_103861

/-- The number of ways to form n mixed pairs from m men and w women -/
def number_of_ways_to_form_mixed_pairs (m w n : ℕ) : ℕ := sorry

theorem mixed_pairs_count (n : ℕ) : 
  (number_of_ways_to_form_mixed_pairs (2*n) (2*n) n) = 
  (Nat.factorial (2*n))^2 / (Nat.factorial n * 2^n) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_mixed_pairs_count_l1038_103861


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_president_properties_l1038_103894

-- Define the properties
inductive Property : Type where
  | highest_position : Property
  | follows_npc : Property
  | important_organ : Property
  | independent_power : Property

-- Define the President of China
structure President where
  properties : List Property

-- Define the correct properties
def correct_properties : List Property :=
  [Property.follows_npc, Property.important_organ]

-- Define a function to check if the properties are correct
def are_properties_correct (pres : President) : Prop :=
  pres.properties = correct_properties

-- Theorem: The correct properties for the President are
-- "follows NPC decisions" and "important state organ"
theorem president_properties :
  are_properties_correct (President.mk correct_properties) := by
  -- The proof is trivial as we defined correct_properties
  rfl

#check president_properties

end NUMINAMATH_CALUDE_ERRORFEEDBACK_president_properties_l1038_103894


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_focus_x_coordinate_l1038_103853

noncomputable def ellipse_focus (major_axis_start major_axis_end minor_axis_start minor_axis_end : ℝ × ℝ) : ℝ := 
  let center_x := (major_axis_start.fst + major_axis_end.fst) / 2
  let center_y := (major_axis_start.snd + major_axis_end.snd) / 2
  let a := (major_axis_end.fst - major_axis_start.fst) / 2
  let b := (minor_axis_start.snd - minor_axis_end.snd) / 2
  let c := Real.sqrt (a^2 - b^2)
  center_x + c

theorem ellipse_focus_x_coordinate :
  let major_axis_start : ℝ × ℝ := (0, -1)
  let major_axis_end : ℝ × ℝ := (8, -1)
  let minor_axis_start : ℝ × ℝ := (4, 0.5)
  let minor_axis_end : ℝ × ℝ := (4, -2.5)
  abs (ellipse_focus major_axis_start major_axis_end minor_axis_start minor_axis_end - 7.708) < 0.001 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_focus_x_coordinate_l1038_103853


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_shaded_area_two_circles_proof_l1038_103813

/-- The area of the shaded regions given two circles with radii 3 and 6 feet -/
def shaded_area_two_circles (π : ℝ) : ℝ :=
  let small_radius : ℝ := 3
  let large_radius : ℝ := 6
  let small_rectangle_area : ℝ := 2 * small_radius * small_radius
  let large_rectangle_area : ℝ := 2 * large_radius * large_radius
  let small_semicircle_area : ℝ := 0.5 * π * small_radius ^ 2
  let large_semicircle_area : ℝ := 0.5 * π * large_radius ^ 2
  small_rectangle_area + large_rectangle_area - small_semicircle_area - large_semicircle_area

/-- Proof of the shaded area theorem -/
theorem shaded_area_two_circles_proof : shaded_area_two_circles Real.pi = 90 - 22.5 * Real.pi := by
  -- Unfold the definition of shaded_area_two_circles
  unfold shaded_area_two_circles
  -- Simplify the expression
  simp [Real.pi]
  -- The proof steps would go here, but for now we use sorry
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_shaded_area_two_circles_proof_l1038_103813


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_city_graph_coloring_l1038_103870

/-- A graph representing cities and flight connections --/
structure CityGraph where
  V : Type*  -- Set of vertices (cities)
  E : Set (V × V)  -- Set of edges (flights)
  k : ℕ  -- Number of airlines
  airlines : Fin k → Set (V × V)  -- k sets of edges representing airlines
  symmetric : ∀ {u v : V}, (u, v) ∈ E ↔ (v, u) ∈ E
  airline_property : ∀ i : Fin k, ∀ e₁ e₂ : V × V, e₁ ∈ airlines i → e₂ ∈ airlines i → ∃ v, (v = e₁.1 ∨ v = e₁.2) ∧ (v = e₂.1 ∨ v = e₂.2)

/-- A k+2 coloring of the graph --/
def is_valid_coloring (G : CityGraph) (f : G.V → Fin (G.k + 2)) : Prop :=
  ∀ {u v : G.V}, (u, v) ∈ G.E → f u ≠ f v

/-- The main theorem --/
theorem city_graph_coloring (G : CityGraph) : 
  ∃ f : G.V → Fin (G.k + 2), is_valid_coloring G f := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_city_graph_coloring_l1038_103870


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_derivative_f₁_derivative_f₂_derivative_f₃_l1038_103852

-- Function 1
noncomputable def f₁ (x : ℝ) : ℝ := Real.exp x * Real.cos x

theorem derivative_f₁ : 
  deriv f₁ = fun x => Real.exp x * Real.cos x - Real.exp x * Real.sin x :=
by sorry

-- Function 2
noncomputable def f₂ (x : ℝ) : ℝ := x * (x^2 + 1/x + 1/x^3)

theorem derivative_f₂ : 
  deriv f₂ = fun x => 3*x^2 - 2/x^3 :=
by sorry

-- Function 3
noncomputable def f₃ (x : ℝ) : ℝ := Real.log (Real.sqrt (1 + x^2))

theorem derivative_f₃ : 
  deriv f₃ = fun x => x / (1 + x^2) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_derivative_f₁_derivative_f₂_derivative_f₃_l1038_103852


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_function_theorem_l1038_103896

/-- Given a function g(x) = 1 / (cx + d) where c and d are nonzero constants,
    prove that g⁻¹(-1) = (-1 - d) / c -/
theorem inverse_function_theorem (c d : ℝ) (hc : c ≠ 0) (hd : d ≠ 0) :
  let g : ℝ → ℝ := λ x => 1 / (c * x + d)
  (g⁻¹) (-1) = (-1 - d) / c :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_function_theorem_l1038_103896


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_l1038_103848

theorem triangle_area (A B C : ℝ) (a b c : ℝ) :
  Real.cos (2 * A) = Real.sin A →
  b * c = 2 →
  (1 / 2) * b * c * Real.sin A = 1 / 2 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_l1038_103848


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_total_amount_calculation_l1038_103812

/-- Represents the number of days it takes for a person to complete the work alone -/
structure WorkTime where
  days : ℚ
  days_positive : days > 0

/-- Calculates the total amount earned for a job given the work times and one person's share -/
def totalAmount (rahul_time : WorkTime) (rajesh_time : WorkTime) (rahul_share : ℚ) : ℚ :=
  let rahul_rate := 1 / rahul_time.days
  let rajesh_rate := 1 / rajesh_time.days
  let total_rate := rahul_rate + rajesh_rate
  let rahul_proportion := rahul_rate / total_rate
  rahul_share / rahul_proportion

theorem total_amount_calculation (rahul_time : WorkTime) (rajesh_time : WorkTime) (rahul_share : ℚ) 
  (h1 : rahul_time.days = 3)
  (h2 : rajesh_time.days = 2)
  (h3 : rahul_share = 68) :
  totalAmount rahul_time rajesh_time rahul_share = 170 := by
  sorry

#eval totalAmount ⟨3, by norm_num⟩ ⟨2, by norm_num⟩ 68

end NUMINAMATH_CALUDE_ERRORFEEDBACK_total_amount_calculation_l1038_103812


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_existence_l1038_103868

theorem sequence_existence (n : ℕ) :
  ∀ k : ℕ, k ≤ n → ∃ (x : ℕ → ℕ), StrictMono x ∧ (∀ i, i ≤ n → x i ≤ x (i + 1)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_existence_l1038_103868


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_projection_AC_on_AB_l1038_103887

noncomputable section

def A : Fin 3 → ℝ := ![2, 1, 3]
def B : Fin 3 → ℝ := ![2, -2, 6]
def C : Fin 3 → ℝ := ![3, 6, 6]

def vector_AC : Fin 3 → ℝ := ![C 0 - A 0, C 1 - A 1, C 2 - A 2]
def vector_AB : Fin 3 → ℝ := ![B 0 - A 0, B 1 - A 1, B 2 - A 2]

def dot_product (v w : Fin 3 → ℝ) : ℝ :=
  (v 0 * w 0) + (v 1 * w 1) + (v 2 * w 2)

def magnitude (v : Fin 3 → ℝ) : ℝ :=
  Real.sqrt ((v 0)^2 + (v 1)^2 + (v 2)^2)

def scalar_projection (v w : Fin 3 → ℝ) : ℝ :=
  dot_product v w / magnitude w

def projection_vector (v w : Fin 3 → ℝ) : Fin 3 → ℝ :=
  let scalar := scalar_projection v w
  let unit_w := ![w 0 / magnitude w, w 1 / magnitude w, w 2 / magnitude w]
  ![scalar * unit_w 0, scalar * unit_w 1, scalar * unit_w 2]

theorem projection_AC_on_AB :
  projection_vector vector_AC vector_AB = ![0, 1, -1] := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_projection_AC_on_AB_l1038_103887


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetry_center_phi_l1038_103881

noncomputable def f (x φ : ℝ) : ℝ := Real.sin (-2 * x + φ)

theorem symmetry_center_phi (φ : ℝ) : 
  (0 < φ ∧ φ < Real.pi) →
  (∀ x : ℝ, f x φ = f (2 * Real.pi / 3 - x) φ) →
  φ = 2 * Real.pi / 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetry_center_phi_l1038_103881


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_profit_maximizing_price_l1038_103860

/-- Represents the price reduction in dollars -/
def x : Type := ℝ

/-- Represents the daily sales volume in pieces -/
def y (x : ℝ) : ℝ := 4 * x + 100

/-- The initial selling price in dollars -/
def initial_price : ℝ := 200

/-- The cost price in dollars -/
def cost_price : ℝ := 100

/-- The minimum allowed selling price in dollars -/
def min_price : ℝ := 150

/-- The target daily profit in dollars -/
def target_profit : ℝ := 13600

/-- Theorem stating that the selling price resulting in the target profit is $185 -/
theorem profit_maximizing_price :
  ∃ (x : ℝ), 
    (initial_price - x - cost_price) * y x = target_profit ∧
    initial_price - x ≥ min_price ∧
    initial_price - x = 185 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_profit_maximizing_price_l1038_103860


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_lipschitz_constant_sqrt_l1038_103849

noncomputable def f (x : ℝ) : ℝ := Real.sqrt x

def is_lipschitz (f : ℝ → ℝ) (k : ℝ) : Prop :=
  ∀ x₁ x₂, x₁ ≥ 1 → x₂ ≥ 1 → |f x₁ - f x₂| ≤ k * |x₁ - x₂|

theorem min_lipschitz_constant_sqrt :
  ∀ k, (is_lipschitz f k) → k ≥ (1/2 : ℝ) ∧
  is_lipschitz f (1/2 : ℝ) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_lipschitz_constant_sqrt_l1038_103849


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cylinder_experiment_l1038_103890

/-- Represents a gas in a cylinder -/
structure Gas where
  name : String
  initial_pressure : ℝ
  initial_volume : ℝ
  final_volume : ℝ
  temperature : ℝ

/-- Represents the experimental setup -/
structure Experiment where
  nitrogen : Gas
  water_vapor : Gas
  initial_height : ℝ
  piston_speed : ℝ
  duration : ℝ

/-- Calculates the final pressure of a gas given isothermal conditions -/
noncomputable def final_pressure (g : Gas) : ℝ :=
  g.initial_pressure * g.initial_volume / g.final_volume

/-- Calculates the power ratio of two gases -/
noncomputable def power_ratio (g1 g2 : Gas) : ℝ :=
  final_pressure g1 / final_pressure g2

/-- Checks if there's a 30-second interval with work > 15 J -/
def exists_high_work_interval (e : Experiment) : Prop :=
  ∃ (t : ℝ), t ≥ 0 ∧ t + 30 ≤ e.duration * 60 ∧
    (final_pressure e.nitrogen * e.piston_speed * 30 > 15 ∨
     final_pressure e.water_vapor * e.piston_speed * 30 > 15)

theorem cylinder_experiment (e : Experiment) 
  (h1 : e.nitrogen.name = "nitrogen")
  (h2 : e.water_vapor.name = "water vapor")
  (h3 : e.nitrogen.initial_pressure = 0.5)
  (h4 : e.water_vapor.initial_pressure = 0.5)
  (h5 : e.nitrogen.initial_volume = 2)
  (h6 : e.water_vapor.initial_volume = 2)
  (h7 : e.nitrogen.temperature = 100)
  (h8 : e.water_vapor.temperature = 100)
  (h9 : e.initial_height = 1)
  (h10 : e.piston_speed = 10 / 60 / 100)
  (h11 : e.duration = 7.5)
  (h12 : e.nitrogen.final_volume = e.nitrogen.initial_volume * (1 - e.piston_speed * e.duration * 60 / e.initial_height))
  (h13 : e.water_vapor.final_volume = e.water_vapor.initial_volume * (1 - e.piston_speed * e.duration * 60 / e.initial_height))
  (h14 : final_pressure e.water_vapor = 1) :
  power_ratio e.nitrogen e.water_vapor = 2 ∧ exists_high_work_interval e := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cylinder_experiment_l1038_103890


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_complement_necessary_not_sufficient_l1038_103846

-- Define the sets A and B
def A (a : ℝ) : Set ℝ := {x | (x - a) * (x - 3 * a) < 0 ∧ a > 0}
def B : Set ℝ := {x | ∃ t, 2 < t ∧ t < 3 ∧ x = 2^(t - 2)}

-- Theorem 1: When a = 1, A ∩ (∁_ℝ B) = [2, 3)
theorem intersection_complement (a : ℝ) (ha : a = 1) :
  A a ∩ (Set.univ \ B) = Set.Icc 2 3 := by sorry

-- Theorem 2: When B ⊊ A, 2/3 ≤ a ≤ 1
theorem necessary_not_sufficient (a : ℝ) (h : B ⊂ A a) :
  2/3 ≤ a ∧ a ≤ 1 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_complement_necessary_not_sufficient_l1038_103846


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangle_area_perimeter_product_l1038_103828

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Calculates the distance between two points -/
noncomputable def distance (p1 p2 : Point) : ℝ :=
  Real.sqrt ((p1.x - p2.x)^2 + (p1.y - p2.y)^2)

/-- Calculates the area of a rectangle given two adjacent sides -/
def rectangleArea (side1 side2 : ℝ) : ℝ :=
  side1 * side2

/-- Calculates the perimeter of a rectangle given two adjacent sides -/
def rectanglePerimeter (side1 side2 : ℝ) : ℝ :=
  2 * (side1 + side2)

/-- The main theorem to be proved -/
theorem rectangle_area_perimeter_product :
  let e := Point.mk 2 3
  let f := Point.mk 3 1
  let g := Point.mk 1 0
  let h := Point.mk 0 2
  let ef := distance e f
  let fg := distance f g
  let area := rectangleArea ef fg
  let perimeter := rectanglePerimeter ef fg
  area * perimeter = 20 * Real.sqrt 5 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangle_area_perimeter_product_l1038_103828


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l1038_103826

-- Define the function f
noncomputable def f (x : ℝ) : ℝ :=
  if x ≥ 0 then x^2 + x else -3*x

-- State the theorem
theorem range_of_a (a : ℝ) :
  (a * (f a - f (-a)) > 0) ↔ (a < -2 ∨ a > 2) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l1038_103826


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadrilateral_inequality_l1038_103841

/-- Predicate to check if four points form a convex quadrilateral -/
def ConvexQuadrilateral (A B C D : ℝ × ℝ) : Prop :=
  sorry

/-- Function to calculate the area of a quadrilateral -/
def QuadrilateralArea (A B C D : ℝ × ℝ) : ℝ :=
  sorry

/-- Function to calculate the angle between two lines -/
def AngleBetweenLines (A B C D : ℝ × ℝ) : ℝ :=
  sorry

/-- Given a convex quadrilateral ABCD with area S, angle α between AB and CD,
    and angle β between AD and BC, prove the inequality. -/
theorem quadrilateral_inequality (A B C D : ℝ × ℝ) (S : ℝ) (α β : ℝ) :
  let AB := Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2)
  let BC := Real.sqrt ((B.1 - C.1)^2 + (B.2 - C.2)^2)
  let CD := Real.sqrt ((C.1 - D.1)^2 + (C.2 - D.2)^2)
  let AD := Real.sqrt ((A.1 - D.1)^2 + (A.2 - D.2)^2)
  ConvexQuadrilateral A B C D ∧ 
  QuadrilateralArea A B C D = S ∧
  AngleBetweenLines A B C D = α ∧
  AngleBetweenLines A D B C = β →
  AB * CD * Real.sin α + AD * BC * Real.sin β ≤ 2 * S ∧ 2 * S ≤ AB * CD + AD * BC :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadrilateral_inequality_l1038_103841


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l1038_103800

def f (x : ℝ) := x^2 - 2*x
def g (a x : ℝ) := a*x + 2

theorem range_of_a :
  ∀ (a : ℝ),
  (a > 0) →
  (∀ (x₁ : ℝ), x₁ ∈ Set.Icc (-1 : ℝ) 2 → 
    ∃ (x₀ : ℝ), x₀ ∈ Set.Icc (-1 : ℝ) 2 ∧ g a x₁ = f x₀) →
  a ∈ Set.Ioo (0 : ℝ) (1/2) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l1038_103800


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_land_development_profit_l1038_103827

/-- Calculates the profit from a land development project given the initial and final conditions. -/
theorem land_development_profit
  (initial_acres : ℕ)
  (initial_price_per_acre : ℕ)
  (final_price_per_acre : ℕ)
  (sold_fraction : ℚ)
  (h1 : initial_acres = 200)
  (h2 : initial_price_per_acre = 70)
  (h3 : final_price_per_acre = 200)
  (h4 : sold_fraction = 1/2) :
  (sold_fraction * initial_acres * final_price_per_acre : ℚ) -
  (initial_acres * initial_price_per_acre : ℚ) = 6000 := by
  sorry

#check land_development_profit

end NUMINAMATH_CALUDE_ERRORFEEDBACK_land_development_profit_l1038_103827


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_derivative_at_2_l1038_103801

-- Define the function f
def f (x : ℝ) (k : ℝ) : ℝ := x^2 + 3*x*k

-- State the theorem
theorem f_derivative_at_2 : 
  ∃ k, (∀ x, f x k = x^2 + 3*x*k) ∧ 
       ((deriv (fun x => f x k)) 2 = k) ∧ 
       k = -2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_derivative_at_2_l1038_103801


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pyarelal_loss_is_1800_l1038_103831

/-- Calculates Pyarelal's share of the loss given the ratio of capitals and total loss -/
def pyarelal_loss (ashok_capital_ratio : ℚ) (total_loss : ℕ) : ℚ :=
  let pyarelal_ratio : ℚ := 1
  let total_ratio : ℚ := pyarelal_ratio + ashok_capital_ratio
  let pyarelal_loss_ratio : ℚ := pyarelal_ratio / total_ratio
  pyarelal_loss_ratio * total_loss

/-- Theorem stating that Pyarelal's loss is 1800 given the conditions -/
theorem pyarelal_loss_is_1800 :
  pyarelal_loss (1/9 : ℚ) 2000 = 1800 := by
  sorry

#eval pyarelal_loss (1/9) 2000

end NUMINAMATH_CALUDE_ERRORFEEDBACK_pyarelal_loss_is_1800_l1038_103831


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_not_divisible_by_three_l1038_103822

theorem max_not_divisible_by_three (integers : Finset ℕ) 
  (h1 : integers.card = 6)
  (h2 : (integers.prod id) % 3 = 0) :
  ∃ (not_divisible : Finset ℕ), 
    not_divisible ⊆ integers ∧ 
    not_divisible.card ≤ 5 ∧
    ∀ n ∈ not_divisible, n % 3 ≠ 0 ∧
    ∀ (larger : Finset ℕ), 
      larger ⊆ integers → 
      (∀ n ∈ larger, n % 3 ≠ 0) → 
      larger.card ≤ not_divisible.card :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_not_divisible_by_three_l1038_103822


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_l1038_103837

theorem solution_set : 
  {(a, b, c, d) : ℤ × ℤ × ℤ × ℤ | 
    a * b - 2 * c * d = 3 ∧ 
    a * c + b * d = 1} = 
  {(1, 3, 1, 0), (-1, -3, -1, 0), (3, 1, 0, 1), (-3, -1, 0, -1)} := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_l1038_103837


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_square_parallelogram_is_minor_premise_l1038_103876

-- Define the structure of a syllogism
structure Syllogism where
  major_premise : String
  minor_premise : String
  conclusion : String

-- Define our specific syllogism
def square_parallelogram_syllogism : Syllogism :=
  { major_premise := "The diagonals of a parallelogram bisect each other",
    minor_premise := "A square is a parallelogram",
    conclusion := "The diagonals of a square bisect each other" }

-- Theorem statement
theorem square_parallelogram_is_minor_premise :
  square_parallelogram_syllogism.minor_premise = "A square is a parallelogram" :=
by
  -- Unfold the definition of square_parallelogram_syllogism
  rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_square_parallelogram_is_minor_premise_l1038_103876


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_expansion_coefficients_equal_l1038_103878

noncomputable def coeff_of_x_pow_n_minus_1 (n : ℕ) (x y : ℝ) : ℝ :=
  n * (2 * Real.sqrt y - 1)

def coeff_of_xy (n : ℕ) (x y : ℝ) : ℝ :=
  (-1)^(n-3) * 2 * n * (n-1) * (n-2)

theorem expansion_coefficients_equal (n : ℕ) (hn : n > 4) :
  (∃ (x y : ℝ), 
    (coeff_of_x_pow_n_minus_1 n x y = coeff_of_xy n x y) ∧ 
    (coeff_of_x_pow_n_minus_1 n x y ≠ 0) ∧
    (coeff_of_xy n x y ≠ 0)) →
  n = 51 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_expansion_coefficients_equal_l1038_103878


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_club_cost_effectiveness_l1038_103855

/-- Charge for Club A -/
def f (x : ℝ) : ℝ := 5 * x

/-- Charge for Club B -/
noncomputable def g (x : ℝ) : ℝ := 
  if x ≤ 30 then 90 else 2 * x + 30

theorem club_cost_effectiveness (x : ℝ) (h : 15 ≤ x ∧ x ≤ 40) :
  (15 ≤ x ∧ x < 18 → f x < g x) ∧
  (x = 18 → f x = g x) ∧
  (18 < x ∧ x ≤ 40 → f x > g x) := by
  sorry

#check club_cost_effectiveness

end NUMINAMATH_CALUDE_ERRORFEEDBACK_club_cost_effectiveness_l1038_103855


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_equation_l1038_103869

-- Define the function f as noncomputable due to the use of Real.sqrt
noncomputable def f (x : ℝ) : ℝ := Real.sqrt 3 * Real.cos x + Real.sin x

-- State the theorem
theorem tangent_line_equation :
  let x₀ : ℝ := π / 3
  let y₀ : ℝ := f x₀
  let m : ℝ := -Real.sqrt 3 * Real.sin x₀ + Real.cos x₀
  ∀ x y : ℝ, y - y₀ = m * (x - x₀) ↔ y = -x + π / 3 + Real.sqrt 3 :=
by
  -- Placeholder for the proof
  sorry

-- You can add more lemmas or theorems here if needed

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_equation_l1038_103869


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_new_flat_face_area_l1038_103891

/-- Represents a cylinder with given radius and height -/
structure Cylinder where
  radius : ℝ
  height : ℝ

/-- Represents the new flat face created by cutting the cylinder -/
noncomputable def NewFlatFace (c : Cylinder) : ℝ :=
  2 * c.radius * c.radius * Real.sqrt 2

theorem new_flat_face_area (c : Cylinder) 
  (h_radius : c.radius = 8)
  (h_height : c.height = 10)
  (h_angle : Real.pi / 2 = Real.pi / 2) -- 90° angle condition
  : NewFlatFace c = 32 * Real.sqrt 2 := by
  sorry

def d : ℤ := 0
def e : ℤ := 32
def f : ℤ := 2

#eval d + e + f  -- Evaluates to 34, which is d + e + f

end NUMINAMATH_CALUDE_ERRORFEEDBACK_new_flat_face_area_l1038_103891


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_same_song_is_1_54_l1038_103858

/-- Represents the three brothers -/
inductive Brother
| Abel
| Banach
| Gauss

/-- Represents the state of song collections -/
structure SongState where
  abel : ℕ
  banach : ℕ
  gauss : ℕ

/-- Initial state of song collections -/
def initial_state : SongState :=
  { abel := 9, banach := 6, gauss := 3 }

/-- Represents the action of copying songs -/
def copy_songs (from_brother to_brother : Brother) (state : SongState) : SongState :=
  match from_brother, to_brother with
  | Brother.Abel, Brother.Banach => { state with banach := state.abel }
  | Brother.Abel, Brother.Gauss => { state with gauss := state.abel }
  | Brother.Banach, Brother.Abel => { state with abel := state.banach }
  | Brother.Banach, Brother.Gauss => { state with gauss := state.banach }
  | Brother.Gauss, Brother.Abel => { state with abel := state.gauss }
  | Brother.Gauss, Brother.Banach => { state with banach := state.gauss }
  | _, _ => state  -- No change if copying to self

/-- Probability of all brothers playing the same song -/
def prob_same_song (state : SongState) : ℚ :=
  (1 / state.abel) * (1 / state.banach) * (1 / state.gauss)

/-- Theorem stating the probability of all brothers playing the same song -/
theorem prob_same_song_is_1_54 :
  ∃ (s1 s2 : SongState),
    (∃ (b1 b2 : Brother), s1 = copy_songs b1 Brother.Abel initial_state ∧
                           s2 = copy_songs b2 Brother.Banach s1) ∧
    prob_same_song s2 = 1/54 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_same_song_is_1_54_l1038_103858


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_mean_profit_first_half_l1038_103873

def total_days : ℕ := 30
def last_half_days : ℕ := 15
def mean_profit_month : ℚ := 350
def mean_profit_last_half : ℚ := 445

theorem mean_profit_first_half :
  (total_days * mean_profit_month - last_half_days * mean_profit_last_half) / last_half_days = 255 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_mean_profit_first_half_l1038_103873


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_point_opposite_sides_range_a_l1038_103838

def point_1 : ℝ × ℝ := (-1, -3)
def point_2 : ℝ × ℝ := (4, -6)

def line_equation (x y a : ℝ) : ℝ := 3 * x - 2 * y - a

def opposite_sides (p1 p2 : ℝ × ℝ) (f : ℝ → ℝ → ℝ → ℝ) (a : ℝ) : Prop :=
  (f p1.1 p1.2 a) * (f p2.1 p2.2 a) < 0

def range_of_a : Set ℝ := {a | a < -7 ∨ a > 24}

theorem point_opposite_sides_range_a (a : ℝ) :
  opposite_sides point_1 point_2 line_equation a → a ∈ range_of_a :=
by
  intro h
  -- The proof goes here
  sorry

#check point_opposite_sides_range_a

end NUMINAMATH_CALUDE_ERRORFEEDBACK_point_opposite_sides_range_a_l1038_103838


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_formula_l1038_103814

def a : ℕ → ℤ
  | 0 => 1  -- Added case for 0
  | 1 => 1
  | n + 1 => 2 * a n + n * (1 + 2^n)

theorem a_formula (n : ℕ) (h : n ≥ 1) : 
  a n = 2^(n-2) * (n^2 - n + 6) - n - 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_formula_l1038_103814


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_g_extrema_of_g_l1038_103820

-- Define the function g
noncomputable def g (x : ℝ) : ℝ := (Real.arccos x)^4 + (Real.arcsin x)^4

-- State the theorem about the range of g
theorem range_of_g :
  ∀ y, y ∈ Set.range g → -π^4/8 ≤ y ∧ y ≤ π^4/4 :=
by sorry

-- State the theorem about the extrema of g
theorem extrema_of_g :
  ∃ x₁ x₂, x₁ ∈ Set.Icc (-1) 1 ∧ x₂ ∈ Set.Icc (-1) 1 ∧ g x₁ = -π^4/8 ∧ g x₂ = π^4/4 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_g_extrema_of_g_l1038_103820


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_existence_of_product_term_l1038_103803

def sequence_a (b : ℕ) : ℕ → ℕ
  | 0 => 1
  | 1 => b
  | (n + 2) => 2 * sequence_a b (n + 1) - sequence_a b n + 2

theorem existence_of_product_term (b : ℕ) (hb : b > 0) :
  ∀ n : ℕ, ∃ m : ℕ, sequence_a b n * sequence_a b (n + 1) = sequence_a b m :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_existence_of_product_term_l1038_103803


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_mn_sum_is_seven_l1038_103871

def set1 (n : ℤ) : Finset ℤ := {n, n + 6, n + 8, n + 12, n + 18}
def set2 (m : ℤ) : Finset ℤ := {m, m + 2, m + 4, m + 6, m + 8}

-- Define custom median and mean functions
def median (s : Finset ℤ) : ℚ := sorry
def mean (s : Finset ℤ) : ℚ := sorry

theorem mn_sum_is_seven (n m : ℤ) 
  (h1 : median (set1 n) = 12)
  (h2 : mean (set2 m) = m + 5) : 
  m + n = 7 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_mn_sum_is_seven_l1038_103871


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trapezoid_area_l1038_103897

/-- The area of a trapezoid with bases 3y and 4y, and height y, is 7y^2/2 -/
theorem trapezoid_area (y : ℝ) : 
  (1/2 : ℝ) * ((3 * y) + (4 * y)) * y = (7/2 : ℝ) * y^2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trapezoid_area_l1038_103897


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_segment_length_l1038_103859

/-- The curve C in the xy-plane -/
def curve (x y : ℝ) : Prop := x^2/4 + y^2 = 1

/-- The line l in the xy-plane -/
def line (x y : ℝ) : Prop := x - Real.sqrt 3 * y + Real.sqrt 3 = 0

/-- The length of the line segment AB -/
noncomputable def segment_length : ℝ := 32/7

/-- Theorem stating that the length of the line segment formed by the intersection
    of the curve and the line is equal to segment_length -/
theorem intersection_segment_length :
  ∃ A B : ℝ × ℝ,
    curve A.1 A.2 ∧ curve B.1 B.2 ∧
    line A.1 A.2 ∧ line B.1 B.2 ∧
    A ≠ B ∧
    Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) = segment_length := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_segment_length_l1038_103859


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_isosceles_altitude_divides_apex_angle_equally_l1038_103879

/-- An isosceles triangle with an altitude from the apex to the base midpoint -/
structure IsoscelesTriangleWithAltitude where
  A : ℝ  -- Angle A
  B : ℝ  -- Angle B
  C : ℝ  -- Angle C
  C₁ : ℝ  -- Part of angle C divided by the altitude
  C₂ : ℝ  -- Part of angle C divided by the altitude
  isIsosceles : A = B
  altitudeDividesBase : True  -- Represents that the altitude divides AB into two equal segments
  altitudeDividesC : C = C₁ + C₂

/-- In an isosceles triangle with an altitude from the apex to the base midpoint, 
    the parts of the apex angle divided by the altitude are equal -/
theorem isosceles_altitude_divides_apex_angle_equally 
  (triangle : IsoscelesTriangleWithAltitude) : triangle.C₁ = triangle.C₂ := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_isosceles_altitude_divides_apex_angle_equally_l1038_103879


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_square_covers_circle_l1038_103818

/-- A square with side length 1 -/
def unit_square : Set (ℝ × ℝ) := {p | 0 ≤ p.1 ∧ p.1 ≤ 1 ∧ 0 ≤ p.2 ∧ p.2 ≤ 1}

/-- A circle with diameter greater than 1 -/
def large_circle : Set (ℝ × ℝ) := {p | ∃ (c : ℝ × ℝ) (r : ℝ), r > 1/2 ∧ (p.1 - c.1)^2 + (p.2 - c.2)^2 ≤ r^2}

/-- A partition of the unit square into two parts -/
def square_partition : Set (Set (ℝ × ℝ) × Set (ℝ × ℝ)) := 
  {p | p.1 ∪ p.2 = unit_square ∧ p.1 ∩ p.2 = ∅}

/-- The theorem stating that it's possible to cover the large circle with parts of the unit square -/
theorem square_covers_circle : ∃ (A B : Set (ℝ × ℝ)), (A, B) ∈ square_partition ∧ A ∪ B ⊇ large_circle := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_square_covers_circle_l1038_103818


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_perpendicular_range_l1038_103892

noncomputable def f (x : ℝ) := (1/2) * x^2 + 4 * Real.log x

theorem tangent_line_perpendicular_range (x₀ : ℝ) (h₁ : 1 ≤ x₀) (h₂ : x₀ ≤ 3) (m : ℝ) :
  (∃ (y₀ : ℝ), y₀ = f x₀ ∧ 
    (deriv f x₀ * (-1/m) = -1)) →
  4 ≤ m ∧ m ≤ 5 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_perpendicular_range_l1038_103892


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_circles_construction_l1038_103863

-- Define a point in 2D space
structure Point where
  x : ℝ
  y : ℝ

-- Define a triangle
structure Triangle where
  A : Point
  B : Point
  C : Point

-- Define a circle
structure Circle where
  center : Point
  radius : ℝ

-- Helper functions (not implemented, just signatures)
def CirclesIntersectAt (c1 c2 c3 : Circle) (p : Point) : Prop := sorry
def CircleTangentToSide (c1 c2 : Circle) (a b : Point) : Prop := sorry
def SimilarTriangles (t1 t2 : Triangle) : Prop := sorry

-- Theorem statement
theorem triangle_circles_construction (t : Triangle) :
  ∃ (c1 c2 c3 : Circle),
    -- Circles have equal radii
    c1.radius = c2.radius ∧ c2.radius = c3.radius ∧
    -- Circles intersect at a single point
    ∃ (I : Point), CirclesIntersectAt c1 c2 c3 I ∧
    -- Each side of the triangle is tangent to exactly two circles
    (CircleTangentToSide c1 c2 t.A t.B ∧
     CircleTangentToSide c2 c3 t.B t.C ∧
     CircleTangentToSide c3 c1 t.C t.A) ∧
    -- Centers of circles form a similar triangle
    SimilarTriangles
      (Triangle.mk c1.center c2.center c3.center)
      t :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_circles_construction_l1038_103863


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sphere_segment_surface_area_l1038_103821

/-- The surface area of a part of a sphere enclosed between two parallel planes -/
theorem sphere_segment_surface_area 
  (R h : ℝ) 
  (R_pos : R > 0) 
  (h_pos : h > 0) 
  (h_bound : h < 2*R) : 
  ∃ (S : ℝ), S = 2 * π * R * h := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sphere_segment_surface_area_l1038_103821


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_nacl_hcl_equality_l1038_103864

/-- Represents the number of moles of a substance -/
def Moles : Type := ℝ

/-- The reaction between NaCl and HNO3 has a 1:1 molar ratio -/
axiom reaction_ratio : Moles → Moles → Prop

/-- The number of moles of HCl formed -/
def hcl_formed : Moles := (3 : ℝ)

/-- The number of moles of HNO3 used -/
def hno3_used : Moles := (3 : ℝ)

/-- The number of moles of NaCl combined -/
def nacl_combined : Moles := (3 : ℝ)

theorem nacl_hcl_equality (h : reaction_ratio nacl_combined hcl_formed) :
  nacl_combined = hcl_formed := by
  sorry

#check nacl_hcl_equality

end NUMINAMATH_CALUDE_ERRORFEEDBACK_nacl_hcl_equality_l1038_103864


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_inequality_l1038_103834

-- Define a triangle as a triple of angles that sum to π
def Triangle := {angles : (Real × Real × Real) // angles.1 + angles.2.1 + angles.2.2 = Real.pi}

-- Define the function f
noncomputable def f (t : Triangle) : Real :=
  let (a, b, c) := t.val
  (Real.cos a)^2 / (1 + Real.cos a) + 
  (Real.cos b)^2 / (1 + Real.cos b) + 
  (Real.cos c)^2 / (1 + Real.cos c)

-- State the theorem
theorem triangle_inequality (t : Triangle) : f t ≥ 1/2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_inequality_l1038_103834


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_valid_sequence_l1038_103883

def is_valid_sequence (F : ℕ → ℕ) : Prop :=
  (∀ k : ℕ, ∃ n : ℕ, F n = k) ∧
  (∀ k : ℕ, k > 0 → Set.Infinite {n : ℕ | F n = k}) ∧
  (∀ n : ℕ, n ≥ 2 → F (F (n^163)) = F (F n) + F (F 361))

theorem exists_valid_sequence : ∃ F : ℕ → ℕ, is_valid_sequence F := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_valid_sequence_l1038_103883


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_decreasing_function_a_range_l1038_103816

-- Define the function f(x) as noncomputable
noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x ≤ 1 then (a - 2) * x + 4 else 3 * a / x

-- State the theorem
theorem decreasing_function_a_range (a : ℝ) :
  (∀ x y : ℝ, x < y → f a x > f a y) ↔ (0 < a ∧ a ≤ 1) :=
by
  sorry

-- You can add more lemmas or theorems here if needed

end NUMINAMATH_CALUDE_ERRORFEEDBACK_decreasing_function_a_range_l1038_103816


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_closed_form_l1038_103857

noncomputable def a : ℕ → ℝ
  | 0 => 1  -- Added case for 0 to cover all natural numbers
  | 1 => 1
  | (n + 2) => (1/16) * (1 + 4 * a (n + 1) + Real.sqrt (1 + 24 * a (n + 1)))

theorem a_closed_form (n : ℕ) (hn : n ≥ 1) :
  a n = (2^(2*n-1) + 3 * 2^(n-1) + 1) / (3 * 2^(2*n-1)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_closed_form_l1038_103857


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_product_and_closest_value_l1038_103819

theorem product_and_closest_value : 
  let product : ℝ := 0.0004 * 9000000
  let options : List ℝ := [320, 360, 3000, 3600, 4000]
  (product = 3600) ∧ 
  (∀ x ∈ options, |product - 3600| ≤ |product - x|) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_product_and_closest_value_l1038_103819


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_knights_wins_34_l1038_103830

structure Team where
  name : String
  wins : Nat
  deriving Repr

def teams : List Team := [
  ⟨"Knights", 34⟩,
  ⟨"Falcons", 15⟩,
  ⟨"Warriors", 18⟩,
  ⟨"Raiders", 30⟩,
  ⟨"Miners", 36⟩
]

theorem knights_wins_34 :
  ∃ (knights falcons warriors raiders miners : Team),
    knights ∈ teams ∧ falcons ∈ teams ∧ warriors ∈ teams ∧ raiders ∈ teams ∧ miners ∈ teams ∧
    knights.name = "Knights" ∧ falcons.name = "Falcons" ∧ warriors.name = "Warriors" ∧
    raiders.name = "Raiders" ∧ miners.name = "Miners" ∧
    warriors.wins > falcons.wins ∧
    knights.wins > raiders.wins ∧ knights.wins < miners.wins ∧
    raiders.wins > 25 ∧
    knights.wins = 34 := by
  sorry

#eval teams

end NUMINAMATH_CALUDE_ERRORFEEDBACK_knights_wins_34_l1038_103830


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l1038_103833

noncomputable def f (x : ℝ) : ℝ := x^3 + (1/2) * x^2 - 4*x

def interval : Set ℝ := {x | -2 ≤ x ∧ x ≤ 2}

theorem f_properties :
  -- 1. The derivative of f
  (∀ x, HasDerivAt f (3*x^2 + x - 4) x) ∧
  -- 2. The minimum value of f in the interval
  (∃ x ∈ interval, f x = -5/2 ∧ ∀ y ∈ interval, f y ≥ f x) ∧
  -- 3. The maximum value of f in the interval
  (∃ x ∈ interval, f x = 104/27 ∧ ∀ y ∈ interval, f y ≤ f x) := by
  sorry

#check f_properties

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l1038_103833


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_at_one_l1038_103823

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := x^2 + 1/x

-- Define the derivative of f
noncomputable def f_deriv (x : ℝ) : ℝ := 2*x - 1/x^2

-- State the theorem
theorem tangent_line_at_one (x y : ℝ) :
  (f_deriv 1) * (x - 1) = y - f 1 ↔ x - y + 1 = 0 := by
  sorry

#check tangent_line_at_one

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_at_one_l1038_103823


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_first_valve_fills_in_two_hours_l1038_103877

/-- The time in hours it takes for the first valve alone to fill the pool -/
noncomputable def first_valve_time (pool_capacity : ℝ) (both_valves_time : ℝ) (valve_difference : ℝ) : ℝ :=
  let combined_rate := pool_capacity / both_valves_time
  let first_valve_rate := (combined_rate - valve_difference) / 2
  pool_capacity / first_valve_rate / 60

/-- The theorem stating that under the given conditions, 
    the first valve alone takes 2 hours to fill the pool -/
theorem first_valve_fills_in_two_hours :
  first_valve_time 12000 48 50 = 2 := by
  sorry

-- Remove the #eval statement as it's not computable
-- #eval first_valve_time 12000 48 50

end NUMINAMATH_CALUDE_ERRORFEEDBACK_first_valve_fills_in_two_hours_l1038_103877


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_banana_cost_theorem_l1038_103880

/-- The cost of a given number of half-dozen fruits, given the cost of three half-dozen apples -/
noncomputable def fruit_cost (apple_cost : ℝ) (num_half_dozen : ℝ) : ℝ :=
  (apple_cost / 3) * num_half_dozen

/-- Theorem stating that if three half-dozen apples cost $9.36, then four half-dozen bananas
    (at the same price per fruit) will cost $12.48 -/
theorem banana_cost_theorem (apple_cost : ℝ) (h : apple_cost = 9.36) :
  fruit_cost apple_cost 4 = 12.48 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_banana_cost_theorem_l1038_103880


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_diophantine_equation_solutions_l1038_103893

theorem diophantine_equation_solutions :
  (∃ (S : Set ℕ+), Set.Infinite S ∧ ∀ (N : ℕ+), N ∈ S → ∃ (x y z t : ℕ+), x^2 + y^2 + z^2 + t^2 = N * x * y * z * t + N) ∧
  (∀ (k m : ℕ), ¬∃ (x y z t : ℕ+), x^2 + y^2 + z^2 + t^2 = (4 * k * (8 * m + 7)) * x * y * z * t + (4 * k * (8 * m + 7))) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_diophantine_equation_solutions_l1038_103893


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_thirteen_most_likely_l1038_103866

/-- A standard six-sided die -/
def Die : Finset ℕ := {1, 2, 3, 4, 5, 6}

/-- The set of possible sums before the final roll -/
def PossibleSums : Finset ℕ := {7, 8, 9, 10, 11, 12}

/-- The set of possible total sums after the final roll -/
def TotalSums : Finset ℕ := {8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18}

/-- Function to check if a number appears in all possible scenarios -/
def appearsInAllScenarios (n : ℕ) : Prop :=
  ∀ s, s ∈ PossibleSums → ∃ d, d ∈ Die ∧ s + d = n

/-- Theorem stating that 13 is the most likely total sum -/
theorem thirteen_most_likely :
  ∃! n, n ∈ TotalSums ∧ appearsInAllScenarios n ∧ 
  (∀ m, m ∈ TotalSums → appearsInAllScenarios m → m = n) :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_thirteen_most_likely_l1038_103866


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_of_f_positive_l1038_103844

-- Define the constants a and b
variable (a b : ℝ)

-- Define the conditions on a and b
variable (h1 : a > 1)
variable (h2 : 1 > b)
variable (h3 : b > 0)
variable (h4 : a - b = 1)

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := Real.log (a^x - b^x)

-- Statement to prove
theorem solution_set_of_f_positive (a b : ℝ) (h1 : a > 1) (h2 : 1 > b) (h3 : b > 0) (h4 : a - b = 1) :
  {x : ℝ | f a b x > 0} = Set.Ioi 1 :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_of_f_positive_l1038_103844


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_students_present_in_school_l1038_103804

theorem students_present_in_school (total_students : ℝ) 
  (home_learning_percent : ℝ) (h1 : home_learning_percent = 40) :
  (100 - home_learning_percent) / 2 = 30 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_students_present_in_school_l1038_103804


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_horizontal_asymptote_of_f_l1038_103805

-- Define the function f(x)
noncomputable def f (x : ℝ) : ℝ := (16*x^4 + 3*x^3 + 7*x^2 + 6*x + 2) / (4*x^4 + x^3 + 5*x^2 + 2*x + 1)

-- State the theorem about the horizontal asymptote of f(x)
theorem horizontal_asymptote_of_f : 
  ∀ ε > 0, ∃ M, ∀ x, |x| > M → |f x - 4| < ε := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_horizontal_asymptote_of_f_l1038_103805


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_equation_l1038_103825

/-- The function representing the curve -/
noncomputable def f (x : ℝ) : ℝ := (x^2 - 2*x - 3) / 4

/-- The x-coordinate of the point of tangency -/
def x₀ : ℝ := 4

/-- The slope of the tangent line at x₀ -/
noncomputable def m : ℝ := (x₀ - 1) / 2

/-- The y-coordinate of the point of tangency -/
noncomputable def y₀ : ℝ := f x₀

/-- The equation of the tangent line -/
noncomputable def tangent_line (x : ℝ) : ℝ := m * (x - x₀) + y₀

theorem tangent_line_equation :
  ∀ x, tangent_line x = (3/2) * x - 19/4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_equation_l1038_103825


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_variance_of_data_set_l1038_103842

noncomputable def data_set (x : ℝ) : List ℝ := [-1, x, 0, 1, -1]

noncomputable def mean (l : List ℝ) : ℝ := (l.sum) / l.length

noncomputable def variance (l : List ℝ) : ℝ :=
  let m := mean l
  (l.map (fun x => (x - m) ^ 2)).sum / l.length

theorem variance_of_data_set :
  ∃ x : ℝ, (mean (data_set x) = 0) ∧ (variance (data_set x) = 0.8) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_variance_of_data_set_l1038_103842


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_greatest_multiple_of_six_l1038_103845

def digits : List Nat := [1, 5, 7, 8, 6]

def is_valid_number (n : Nat) : Prop :=
  n ≥ 10000 ∧ n < 100000 ∧
  (∃ (d1 d2 d3 d4 d5 : Nat), 
    List.Perm [d1, d2, d3, d4, d5] digits ∧
    n = d1 * 10000 + d2 * 1000 + d3 * 100 + d4 * 10 + d5)

theorem greatest_multiple_of_six :
  ∀ n : Nat, is_valid_number n → n % 6 = 0 → n ≤ 76158 :=
by
  intro n h1 h2
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_greatest_multiple_of_six_l1038_103845


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_implies_m_range_l1038_103802

open Real

-- Define the function
noncomputable def f (x : ℝ) : ℝ := (3 * x + 2) / (x - 1)

-- State the theorem
theorem min_value_implies_m_range (m n : ℝ) :
  n = 2 →
  (∀ x ∈ Set.Ioo m n, f x ≥ 8) →
  (∃ x ∈ Set.Ioo m n, f x = 8) →
  m ∈ Set.Icc 1 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_implies_m_range_l1038_103802


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_of_vector_expression_l1038_103862

variable {V : Type*} [NormedAddCommGroup V] [InnerProductSpace ℝ V]

theorem max_value_of_vector_expression (p q r : V) 
  (hp : ‖p‖ = 2) (hq : ‖q‖ = 1) (hr : ‖r‖ = 3) :
  ‖p - 3 • q‖^2 + ‖q - 3 • r‖^2 + ‖r - 3 • p‖^2 ≤ 128 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_of_vector_expression_l1038_103862


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_extreme_values_and_g_zeros_l1038_103865

noncomputable def f (x : ℝ) : ℝ := 2 * x^3 - 3 * x^2 - 12 * x + 5

noncomputable def g (x m : ℝ) : ℝ := f x - m

theorem f_extreme_values_and_g_zeros :
  (∃ x₁ x₂ : ℝ, f x₁ = 12 ∧ f x₂ = -15 ∧ 
    ∀ x : ℝ, f x ≤ 12 ∧ f x ≥ -15) ∧
  (∀ m : ℝ, 
    ((m > 12 ∨ m < -15) → (∃! x : ℝ, g x m = 0)) ∧
    ((m = 12 ∨ m = -15) → (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ g x₁ m = 0 ∧ g x₂ m = 0 ∧ 
      ∀ x : ℝ, g x m = 0 → (x = x₁ ∨ x = x₂))) ∧
    ((-15 < m ∧ m < 12) → (∃ x₁ x₂ x₃ : ℝ, x₁ ≠ x₂ ∧ x₁ ≠ x₃ ∧ x₂ ≠ x₃ ∧ 
      g x₁ m = 0 ∧ g x₂ m = 0 ∧ g x₃ m = 0 ∧
      ∀ x : ℝ, g x m = 0 → (x = x₁ ∨ x = x₂ ∨ x = x₃)))) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_extreme_values_and_g_zeros_l1038_103865


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_slope_product_constant_l1038_103875

-- Define the hyperbola C
noncomputable def hyperbola (x y : ℝ) : Prop := x^2 / 4 - y^2 = 1

-- Define the right focus F and left vertex A
noncomputable def F : ℝ × ℝ := (Real.sqrt 5, 0)
noncomputable def A : ℝ × ℝ := (-2, 0)

-- Define point B
def B : ℝ × ℝ := (4, 0)

-- Define a line passing through B
def line_through_B (m : ℝ) (x y : ℝ) : Prop := y = m * (x - 4)

-- Define the intersection points P and Q
noncomputable def intersection_points (m : ℝ) : Set (ℝ × ℝ) :=
  {p | hyperbola p.1 p.2 ∧ line_through_B m p.1 p.2 ∧ p.1 > 0}

-- Define M and N as intersections of AP, AQ with y-axis
noncomputable def M (P : ℝ × ℝ) : ℝ × ℝ := (0, P.2 * 2 / (P.1 + 2))
noncomputable def N (Q : ℝ × ℝ) : ℝ × ℝ := (0, Q.2 * 2 / (Q.1 + 2))

-- Define slopes k₁ and k₂
noncomputable def k₁ (P : ℝ × ℝ) : ℝ := (B.2 - (M P).2) / (B.1 - (M P).1)
noncomputable def k₂ (Q : ℝ × ℝ) : ℝ := (B.2 - (N Q).2) / (B.1 - (N Q).1)

theorem hyperbola_slope_product_constant :
  ∀ m : ℝ, -2 < m → m < 2 →
  ∃ P Q : ℝ × ℝ, P ∈ intersection_points m ∧ Q ∈ intersection_points m ∧ P ≠ Q →
  (k₁ P) * (k₂ Q) = -1/48 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_slope_product_constant_l1038_103875


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_root_sum_product_l1038_103829

-- Define the quadratic equation
def quadratic_equation (b : ℝ) (x : ℝ) : Prop :=
  x^2 - b*x + 20 = 0

-- Define the roots of the equation
def roots (b : ℝ) : Set ℝ :=
  {x : ℝ | quadratic_equation b x}

-- Theorem statement
theorem quadratic_root_sum_product (b : ℝ) :
  (∀ x y, x ∈ roots b → y ∈ roots b → x * y = 20) →
  (∃ x y, x ∈ roots b ∧ y ∈ roots b ∧ x + y = 12) →
  b = -12 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_root_sum_product_l1038_103829


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_problem_l1038_103835

noncomputable section

/-- Ellipse E with semi-major axis a and semi-minor axis b -/
def Ellipse (a b : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | (p.1^2 / a^2) + (p.2^2 / b^2) = 1}

/-- Line l with slope k passing through (0,1) -/
def Line (k : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.2 = k * p.1 + 1}

/-- Distance between two points -/
noncomputable def distance (p q : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

theorem ellipse_problem (a b k : ℝ) (h_ab : a > b ∧ b > 0) 
    (h_ecc : (a^2 - b^2) / a^2 = 5 / 9)
    (h_intersect : ∃ A B : ℝ × ℝ, A ∈ Ellipse a b ∧ B ∈ Ellipse a b ∧ 
                   A ∈ Line k ∧ B ∈ Line k ∧ A ≠ B)
    (h_perp : ∃ A B : ℝ × ℝ, A ∈ Ellipse a b ∧ B ∈ Ellipse a b ∧ 
              A.2 = 1 ∧ B.2 = 1 ∧ distance A B = 3 * Real.sqrt 3)
    (h_isosceles : ∃ A B : ℝ × ℝ, A ∈ Ellipse a b ∧ B ∈ Ellipse a b ∧ 
                   A ∈ Line k ∧ B ∈ Line k ∧
                   distance A (5/12, 0) = distance B (5/12, 0)) :
  a = 3 ∧ b = 2 ∧ k = -2/3 := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_problem_l1038_103835


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_one_question_sufficient_l1038_103809

/-- Represents a wallet with its coin count -/
structure Wallet :=
  (coins : ℕ)

/-- Represents the state of all wallets -/
def WalletState := Fin 31 → Wallet

/-- Initial state where each wallet has 100 coins -/
def initialState : WalletState :=
  λ _ => ⟨100⟩

/-- State after one wallet transfers coins -/
def transferState (source : Fin 31) : WalletState :=
  λ i => if i ≤ source then ⟨100⟩ else ⟨101⟩

/-- Function to get the sum of coins in odd-indexed wallets -/
def sumOddWallets (state : WalletState) : ℕ :=
  (List.range 31).filter (λ i => i % 2 = 1)
    |>.map (λ i => (state ⟨i, sorry⟩).coins)
    |>.sum

/-- Theorem stating that one question is sufficient to identify the source wallet -/
theorem one_question_sufficient :
  ∀ source : Fin 31, ∃ (query : WalletState → ℕ) (identify : ℕ → Fin 31),
    identify (query (transferState source)) = source := by
  sorry

#eval sumOddWallets initialState

end NUMINAMATH_CALUDE_ERRORFEEDBACK_one_question_sufficient_l1038_103809


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_curve_intersections_l1038_103808

-- Define the curves
noncomputable def C₁ (t : ℝ) : ℝ × ℝ := ((2 + t) / 6, Real.sqrt t)
noncomputable def C₂ (s : ℝ) : ℝ × ℝ := (-(2 + s) / 6, -Real.sqrt s)
noncomputable def C₃ (θ : ℝ) : ℝ × ℝ := (2 * Real.cos θ, Real.sin θ)

-- State the theorem
theorem curve_intersections :
  -- Part 1: Cartesian equation of C₁
  (∀ x y, y ≥ 0 → (∃ t, C₁ t = (x, y)) ↔ y^2 = 6*x - 2) ∧
  -- Part 2: Intersection points of C₃ with C₁
  (∃ θ₁ θ₂ t₁ t₂, C₃ θ₁ = C₁ t₁ ∧ C₃ θ₁ = (1/2, 1) ∧
                   C₃ θ₂ = C₁ t₂ ∧ C₃ θ₂ = (1, 2)) ∧
  -- Part 3: Intersection points of C₃ with C₂
  (∃ θ₃ θ₄ s₁ s₂, C₃ θ₃ = C₂ s₁ ∧ C₃ θ₃ = (-1/2, -1) ∧
                   C₃ θ₄ = C₂ s₂ ∧ C₃ θ₄ = (-1, -2)) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_curve_intersections_l1038_103808


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_sqrt_sum_l1038_103850

theorem max_value_sqrt_sum (m n : ℝ) (hm : m > 0) (hn : n > 0) (h_sum : m + n = 5) :
  ∃ (max : ℝ), max = 4 ∧ 
  Real.sqrt (m + 1) + Real.sqrt (n + 2) ≤ max ∧ 
  ∃ (m' n' : ℝ), m' > 0 ∧ n' > 0 ∧ m' + n' = 5 ∧ 
  Real.sqrt (m' + 1) + Real.sqrt (n' + 2) = max :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_sqrt_sum_l1038_103850


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_simplification_and_evaluation_l1038_103899

noncomputable def f (α : Real) : Real :=
  (Real.sin (Real.pi + α) * Real.cos (2*Real.pi - α) * Real.sin ((3/2)*Real.pi - α)) /
  (Real.cos (-Real.pi - α) * Real.cos (Real.pi/2 + α))

theorem f_simplification_and_evaluation (α : Real) (a : Real) (h : a ≠ 0) :
  f α = Real.cos α ∧
  ((a > 0 → f α = 5/13) ∧ (a < 0 → f α = -5/13)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_simplification_and_evaluation_l1038_103899


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_children_return_to_start_l1038_103854

noncomputable def lake_perimeter : ℝ := 200

noncomputable def child_speed (k : ℕ+) : ℝ := lake_perimeter / k.val

def select_k (ks : Fin 10 → ℕ+) (i : Fin 10) : ℕ+ := ks i

theorem children_return_to_start (ks : Fin 10 → ℕ+) :
  ∃ M : ℕ+, ∀ i : Fin 10, (M.val : ℝ) * child_speed (select_k ks i) = lake_perimeter * (M.val / (select_k ks i).val : ℕ) :=
by sorry

#check children_return_to_start

end NUMINAMATH_CALUDE_ERRORFEEDBACK_children_return_to_start_l1038_103854


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_distance_circle_to_line_l1038_103847

/-- The circle equation x^2 + y^2 + 4x + 6y - 12 = 0 -/
def circleEq (x y : ℝ) : Prop := x^2 + y^2 + 4*x + 6*y - 12 = 0

/-- The line equation 3x + 4y - 12 = 0 -/
def lineEq (x y : ℝ) : Prop := 3*x + 4*y - 12 = 0

/-- The distance function from a point (x, y) to the line -/
noncomputable def distToLine (x y : ℝ) : ℝ := 
  |3*x + 4*y - 12| / Real.sqrt (3^2 + 4^2)

/-- Theorem stating that the maximum distance from any point on the circle to the line is 7 -/
theorem max_distance_circle_to_line : 
  ∃ (x y : ℝ), circleEq x y ∧ ∀ (x' y' : ℝ), circleEq x' y' → distToLine x' y' ≤ distToLine x y ∧ distToLine x y = 7 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_distance_circle_to_line_l1038_103847


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ammonium_bromide_weight_l1038_103806

-- Define atomic weights
def atomic_weight_N : ℝ := 14.01
def atomic_weight_H : ℝ := 1.01
def atomic_weight_Br : ℝ := 79.90

-- Define the number of moles
def moles : ℝ := 5

-- Define the molecular formula of Ammonium bromide
def NH4Br_formula : Fin 3 → ℕ := ![1, 4, 1]  -- [N, H, Br]

-- Theorem statement
theorem ammonium_bromide_weight :
  let molecular_weight := 
    atomic_weight_N * NH4Br_formula 0 +
    atomic_weight_H * NH4Br_formula 1 +
    atomic_weight_Br * NH4Br_formula 2
  moles * molecular_weight = 489.75 := by
  -- Unfold the definitions
  unfold moles atomic_weight_N atomic_weight_H atomic_weight_Br NH4Br_formula
  -- Simplify the expression
  simp
  -- Check the equality
  norm_num
  -- The proof is complete
  done

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ammonium_bromide_weight_l1038_103806


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_formula_area_maximized_equal_angles_max_area_equilateral_l1038_103856

-- Define a circle of radius 1
def Circle := {p : ℝ × ℝ | p.1^2 + p.2^2 = 1}

-- Define three points on the circle
variable (A B C : Circle)

-- Define angles of the triangle
noncomputable def angle_ABC (A B C : Circle) : ℝ := sorry
noncomputable def angle_BCA (A B C : Circle) : ℝ := sorry
noncomputable def angle_CAB (A B C : Circle) : ℝ := sorry

-- Area of the triangle
noncomputable def triangle_area (A B C : Circle) : ℝ := 
  (1/2) * (Real.sin (2 * angle_ABC A B C) + Real.sin (2 * angle_BCA A B C) + Real.sin (2 * angle_CAB A B C))

-- Theorem statements
theorem area_formula (A B C : Circle) : 
  triangle_area A B C = (1/2) * (Real.sin (2 * angle_ABC A B C) + Real.sin (2 * angle_BCA A B C) + Real.sin (2 * angle_CAB A B C)) := by sorry

theorem area_maximized_equal_angles (A B C C' : Circle) :
  triangle_area A B C ≤ triangle_area A B C' ↔ angle_BCA A B C = angle_CAB A B C := by sorry

theorem max_area_equilateral (A B C A' B' C' : Circle) :
  triangle_area A B C ≤ triangle_area A' B' C' ↔ 
  angle_ABC A B C = angle_BCA A B C ∧ angle_BCA A B C = angle_CAB A B C ∧ angle_ABC A B C = π/3 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_formula_area_maximized_equal_angles_max_area_equilateral_l1038_103856


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_third_place_prize_l1038_103888

theorem third_place_prize (num_people : ℕ) (contribution : ℝ) 
  (first_place_percentage : ℝ) (h1 : num_people = 8) (h2 : contribution = 5) 
  (h3 : first_place_percentage = 0.8) : ℝ := by
  
  let total_pot := num_people * contribution
  let first_place_prize := first_place_percentage * total_pot
  let remaining_prize := total_pot - first_place_prize
  let third_place_prize := remaining_prize / 2

  have : third_place_prize = 4 := by
    -- The proof steps would go here
    sorry

  exact third_place_prize


end NUMINAMATH_CALUDE_ERRORFEEDBACK_third_place_prize_l1038_103888


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_curve_line_intersection_l1038_103843

/-- Curve C in polar coordinates -/
noncomputable def curve_C (a : ℝ) (θ : ℝ) : ℝ := 
  2 * a * Real.cos θ / (Real.sin θ)^2

/-- Line l in parametric form -/
noncomputable def line_l (t : ℝ) : ℝ × ℝ := 
  (-2 + Real.sqrt 2 / 2 * t, -4 + Real.sqrt 2 / 2 * t)

/-- Distance between two points -/
noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

/-- Theorem stating that a = 1 satisfies all conditions -/
theorem curve_line_intersection (a : ℝ) : 
  (a > 0) →
  (∃ t1 t2 : ℝ, t1 ≠ t2 ∧ 
    curve_C a (Real.arctan ((line_l t1).2 / (line_l t1).1)) = Real.sqrt ((line_l t1).1^2 + (line_l t1).2^2) ∧
    curve_C a (Real.arctan ((line_l t2).2 / (line_l t2).1)) = Real.sqrt ((line_l t2).1^2 + (line_l t2).2^2) ∧
    distance (line_l t1) (line_l t2) = 2 * Real.sqrt 10) →
  a = 1 := by sorry

#check curve_line_intersection

end NUMINAMATH_CALUDE_ERRORFEEDBACK_curve_line_intersection_l1038_103843


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sphere_volume_variables_and_constants_l1038_103824

noncomputable def sphere_volume (R : ℝ) : ℝ := (4 / 3) * Real.pi * R^3

theorem sphere_volume_variables_and_constants :
  ∃ (V R : ℝ), V = sphere_volume R ∧
  (∀ c : ℝ, c ≠ 0 → sphere_volume (c * R) ≠ sphere_volume R) ∧
  ((4 : ℝ) / 3 = (4 : ℝ) / 3) ∧
  (Real.pi = Real.pi) := by
  sorry

#check sphere_volume_variables_and_constants

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sphere_volume_variables_and_constants_l1038_103824


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_x0_sequence_l1038_103807

theorem max_x0_sequence (n : ℕ) (h : n = 1995) :
  ∃ (x_max : ℝ), x_max > 0 ∧
  ∀ (x : ℕ → ℝ),
    (∀ i, x i > 0) →
    x 0 = x (n - 1) →
    (∀ i : Fin (n - 1), x i + 2 / x i = 2 * x (i + 1) + 1 / x (i + 1)) →
    x 0 ≤ x_max ∧
    x_max = 2^((n - 1) / 2) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_x0_sequence_l1038_103807


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_curve_length_is_correct_l1038_103810

/-- The length of the parametric curve described by (x,y) = (3 sin t, 3 cos t) from t = 0 to t = π/2 -/
noncomputable def curveLength : ℝ := 3 * Real.pi / 2

/-- The parametric equations of the curve -/
noncomputable def curve (t : ℝ) : ℝ × ℝ := (3 * Real.sin t, 3 * Real.cos t)

/-- The start point of the parameter range -/
def t₁ : ℝ := 0

/-- The end point of the parameter range -/
noncomputable def t₂ : ℝ := Real.pi / 2

theorem curve_length_is_correct : 
  curveLength = ∫ t in t₁..t₂, Real.sqrt ((3 * Real.cos t)^2 + (-3 * Real.sin t)^2) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_curve_length_is_correct_l1038_103810


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_points_k_value_l1038_103815

/-- A line contains the points (-2, 7), (7, k), and (25, 4). The value of k is 6. -/
theorem line_points_k_value :
  ∀ (k : ℝ), 
  (∀ (x y : ℝ × ℝ), 
    (x ∈ ({(-2, 7), (7, k), (25, 4)} : Set (ℝ × ℝ)) ∧ 
     y ∈ ({(-2, 7), (7, k), (25, 4)} : Set (ℝ × ℝ)) ∧ 
     x ≠ y) → 
    (y.2 - x.2) / (y.1 - x.1) = (7 - (-2)) / (7 - (-2))) → 
  k = 6 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_points_k_value_l1038_103815


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_A_is_pi_third_sum_b_c_range_l1038_103874

open Real

/-- Represents an acute triangle ABC with sides a, b, c opposite to angles A, B, C respectively. -/
structure AcuteTriangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ
  acute : 0 < A ∧ A < π/2 ∧ 0 < B ∧ B < π/2 ∧ 0 < C ∧ C < π/2

variable (t : AcuteTriangle)

/-- Theorem 1: If a = √3 and (sinB - sinA) / sinC = (b - c) / (a + b), then A = π/3 -/
theorem angle_A_is_pi_third (h1 : t.a = Real.sqrt 3) 
    (h2 : (Real.sin t.B - Real.sin t.A) / Real.sin t.C = (t.b - t.c) / (t.a + t.b)) : 
    t.A = π/3 := by
  sorry

/-- Theorem 2: If a = √3 and A = π/3, then b + c ∈ (3, 2√3] -/
theorem sum_b_c_range (h1 : t.a = Real.sqrt 3) (h2 : t.A = π/3) : 
    3 < t.b + t.c ∧ t.b + t.c ≤ 2 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_A_is_pi_third_sum_b_c_range_l1038_103874


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cubic_equation_solution_l1038_103885

theorem cubic_equation_solution (w : ℝ) (h : (w + 17)^2 = (4*w + 11)*(3*w + 5)) :
  ∃ ε > 0, |w^3 + 1 - 57.29| < ε := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cubic_equation_solution_l1038_103885


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_price_reduction_percentage_l1038_103832

theorem price_reduction_percentage : ∀ (original_price : ℝ),
  original_price > 0 →
  let first_reduction := 0.12
  let second_reduction := 0.10
  let price_after_first := original_price * (1 - first_reduction)
  let price_after_second := price_after_first * (1 - second_reduction)
  price_after_second / original_price = 0.792 := by
  intro original_price h_positive
  -- Proof steps would go here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_price_reduction_percentage_l1038_103832


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_kirsty_model_purchase_l1038_103867

/-- Calculates the maximum number of models that can be purchased given the original price, planned quantity, and new price. -/
def max_models_purchasable (original_price : ℚ) (planned_quantity : ℕ) (new_price : ℚ) : ℕ :=
  (original_price * planned_quantity / new_price).floor.toNat

/-- Proves that given a budget for 30 models at $0.45 each, and a new price of $0.50 per model,
    the maximum number of models that can be purchased is 27. -/
theorem kirsty_model_purchase : 
  max_models_purchasable (45/100) 30 (1/2) = 27 := by
  sorry

#eval max_models_purchasable (45/100) 30 (1/2)

end NUMINAMATH_CALUDE_ERRORFEEDBACK_kirsty_model_purchase_l1038_103867


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_plot_area_calculation_l1038_103886

/-- The area of a triangle given its base and height -/
noncomputable def triangle_area (base height : ℝ) : ℝ := (1 / 2) * base * height

/-- The problem statement -/
theorem plot_area_calculation (base_ABD height_ABD base_ABC height_ABC : ℝ) 
  (h1 : base_ABD = 6)
  (h2 : height_ABD = 3)
  (h3 : base_ABC = 3)
  (h4 : height_ABC = 3) :
  triangle_area base_ABD height_ABD - triangle_area base_ABC height_ABC = 4.5 := by
  -- Substitute the given values
  rw [h1, h2, h3, h4]
  -- Unfold the definition of triangle_area
  unfold triangle_area
  -- Simplify the expression
  simp [mul_comm, mul_assoc, sub_eq_add_neg, add_comm, add_assoc]
  -- The proof is complete
  norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_plot_area_calculation_l1038_103886


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_f_l1038_103840

-- Define the function f(x) = 1 - 2^x
noncomputable def f (x : ℝ) : ℝ := 1 - Real.rpow 2 x

-- State the theorem about the range of f
theorem range_of_f :
  Set.range f = Set.Iio 1 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_f_l1038_103840


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_four_solutions_range_of_t_l1038_103836

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := |x * Real.exp x|

-- Define the function g
noncomputable def g (t : ℝ) (x : ℝ) : ℝ := (f x)^2 - t * (f x)

-- State the theorem
theorem four_solutions_range_of_t :
  ∀ t : ℝ,
  (∃ x₁ x₂ x₃ x₄ : ℝ, 
    x₁ ≠ x₂ ∧ x₁ ≠ x₃ ∧ x₁ ≠ x₄ ∧ x₂ ≠ x₃ ∧ x₂ ≠ x₄ ∧ x₃ ≠ x₄ ∧
    g t x₁ = -1 ∧ g t x₂ = -1 ∧ g t x₃ = -1 ∧ g t x₄ = -1) →
  t > Real.exp 1 + Real.exp (-1) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_four_solutions_range_of_t_l1038_103836
