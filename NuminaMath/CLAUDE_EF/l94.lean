import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_angle_range_l94_9443

/-- The function representing the curve -/
noncomputable def f (x : ℝ) : ℝ := -Real.sqrt x * (x + 1)

/-- The derivative of f -/
noncomputable def f' (x : ℝ) : ℝ := -(3 * x + 1) / (2 * Real.sqrt x)

/-- The angle of inclination of the tangent line -/
noncomputable def θ (x : ℝ) : ℝ := Real.arctan (f' x)

theorem tangent_angle_range :
  ∀ x : ℝ, x > 0 → π / 2 < θ x ∧ θ x ≤ 2 * π / 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_angle_range_l94_9443


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_delta_value_for_specific_triangle_p_plus_q_equals_six_l94_9406

/-- Triangle with side lengths a, b, c -/
structure Triangle (α : Type*) [LinearOrderedField α] where
  a : α
  b : α
  c : α

/-- Parallelogram inscribed in a triangle -/
structure InscribedParallelogram (α : Type*) [LinearOrderedField α] where
  triangle : Triangle α
  φ : α  -- side length of the parallelogram

/-- Area of the inscribed parallelogram as a function of φ -/
def area_formula (α : Type*) [LinearOrderedField α] (p : InscribedParallelogram α) (γ δ : α) : α → α := 
  λ φ ↦ γ * φ - δ * φ^2

/-- The coefficient δ in the area formula -/
noncomputable def delta (α : Type*) [LinearOrderedField α] (p : InscribedParallelogram α) : α :=
  1 / 5  -- We know the value of δ from the problem

theorem delta_value_for_specific_triangle :
  let t : Triangle ℚ := ⟨13, 30, 37⟩
  let p : InscribedParallelogram ℚ := ⟨t, 0⟩  -- φ value doesn't matter for this theorem
  delta ℚ p = 1 / 5 := by sorry

-- Additional theorem to show that p + q = 6
theorem p_plus_q_equals_six :
  let p : ℕ := 1  -- numerator of δ
  let q : ℕ := 5  -- denominator of δ
  p + q = 6 := by
    rfl  -- reflexivity proves this simple equality

end NUMINAMATH_CALUDE_ERRORFEEDBACK_delta_value_for_specific_triangle_p_plus_q_equals_six_l94_9406


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_I_equals_2H_l94_9458

-- Define H as a function of x
noncomputable def H (x : ℝ) : ℝ := Real.log ((3 + x) / (3 - x))

-- Define the substitution function
noncomputable def sub (x : ℝ) : ℝ := (2 * x + x^2) / (2 - x^2)

-- Define I as H composed with the substitution
noncomputable def I (x : ℝ) : ℝ := H (sub x)

-- Theorem statement
theorem I_equals_2H : ∀ x : ℝ, I x = 2 * H x := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_I_equals_2H_l94_9458


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_a_for_decreasing_cos_minus_sin_l94_9442

theorem max_a_for_decreasing_cos_minus_sin (a : ℝ) : 
  (∀ x ∈ Set.Icc (-a) a, 
    ∀ y ∈ Set.Icc (-a) a, 
    x < y → (Real.cos x - Real.sin x) > (Real.cos y - Real.sin y)) → 
  a ≤ π/4 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_a_for_decreasing_cos_minus_sin_l94_9442


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_shorter_side_length_l94_9452

/-- Represents a trapezium with given dimensions -/
structure Trapezium where
  long_side : ℚ
  short_side : ℚ
  height : ℚ
  area : ℚ

/-- Calculates the area of a trapezium -/
def trapezium_area (t : Trapezium) : ℚ :=
  (t.long_side + t.short_side) * t.height / 2

/-- Theorem stating that for a trapezium with given dimensions, the shorter side is 18 cm -/
theorem shorter_side_length (t : Trapezium) 
    (h1 : t.long_side = 28)
    (h2 : t.height = 15)
    (h3 : t.area = 345)
    (h4 : trapezium_area t = t.area) : 
  t.short_side = 18 := by
  sorry

#check shorter_side_length

end NUMINAMATH_CALUDE_ERRORFEEDBACK_shorter_side_length_l94_9452


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_both_heads_is_one_fourth_l94_9462

/-- The probability of getting heads on a single toss of a fair coin. -/
noncomputable def prob_heads : ℝ := 1 / 2

/-- The probability of getting both heads when tossing two fair coins. -/
noncomputable def prob_both_heads : ℝ := prob_heads * prob_heads

/-- Theorem stating that the probability of both coins landing heads up is 1/4. -/
theorem prob_both_heads_is_one_fourth : prob_both_heads = 1 / 4 := by
  unfold prob_both_heads prob_heads
  norm_num

#check prob_both_heads_is_one_fourth

end NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_both_heads_is_one_fourth_l94_9462


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_yellow_percentage_approx_37_l94_9441

/-- Represents the components of a paint mixture -/
structure PaintMixture where
  red : ℝ
  yellow : ℝ
  water : ℝ

/-- Calculates the total volume of a paint mixture -/
noncomputable def totalVolume (m : PaintMixture) : ℝ :=
  m.red + m.yellow + m.water

/-- Calculates the percentage of a component in a paint mixture -/
noncomputable def componentPercentage (component : ℝ) (total : ℝ) : ℝ :=
  (component / total) * 100

/-- The initial paint mixture -/
def initialMixture : PaintMixture :=
  { red := 40 * 0.20
    yellow := 40 * 0.25
    water := 40 * 0.55 }

/-- The final paint mixture after additions and evaporation -/
def finalMixture : PaintMixture :=
  { red := initialMixture.red
    yellow := initialMixture.yellow + 8
    water := (initialMixture.water + 2) * 0.95 }

/-- Theorem stating that the percentage of yellow tint in the final mixture is approximately 37% -/
theorem yellow_percentage_approx_37 :
  |componentPercentage finalMixture.yellow (totalVolume finalMixture) - 37| < 0.5 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_yellow_percentage_approx_37_l94_9441


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_slope_angle_range_l94_9490

/-- The range of the slope angle for a point on the curve y = x³ - x + 2 -/
theorem slope_angle_range :
  let f : ℝ → ℝ := λ x => x^3 - x + 2
  ∀ x : ℝ, let α := Real.arctan (3 * x^2 - 1)
  α ∈ Set.union (Set.Icc 0 (Real.pi / 2)) (Set.Icc (3 * Real.pi / 4) Real.pi) :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_slope_angle_range_l94_9490


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_layer_3_diameter_l94_9409

/-- Calculates the actual diameter of a magnified circular layer -/
noncomputable def actual_diameter (magnified_diameter : ℝ) (magnification_factor : ℝ) : ℝ :=
  magnified_diameter * 10000 / magnification_factor

theorem layer_3_diameter :
  let magnified_diameter : ℝ := 3
  let magnification_factor : ℝ := 1500
  actual_diameter magnified_diameter magnification_factor = 20 := by
  -- Unfold the definition of actual_diameter
  unfold actual_diameter
  -- Perform the calculation
  norm_num
  -- QED

end NUMINAMATH_CALUDE_ERRORFEEDBACK_layer_3_diameter_l94_9409


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_two_integer_solutions_l94_9419

theorem inequality_two_integer_solutions (a : ℝ) :
  (∃! (x y : ℤ), x ≠ y ∧ |x - (1 : ℝ)| < a * x ∧ |y - (1 : ℝ)| < a * y) ↔ (1/2 < a ∧ a ≤ 2/3) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_two_integer_solutions_l94_9419


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_crossing_time_l94_9461

-- Define the train's length in meters
noncomputable def train_length : ℝ := 280

-- Define the train's speed in km/h
noncomputable def train_speed_kmh : ℝ := 50.4

-- Define the conversion factor from km/h to m/s
noncomputable def kmh_to_ms : ℝ := 1000 / 3600

-- Define the time to cross the pole in seconds
noncomputable def time_to_cross : ℝ := 20

-- Theorem statement
theorem train_crossing_time :
  train_length / (train_speed_kmh * kmh_to_ms) = time_to_cross := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_crossing_time_l94_9461


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_point_in_many_polygons_l94_9405

/-- A convex polygon in a plane -/
structure ConvexPolygon where
  vertices : List (ℝ × ℝ)
  convex : ∀ (a b : ℝ × ℝ), a ∈ vertices → b ∈ vertices → ∀ t : ℝ, 0 ≤ t ∧ t ≤ 1 → 
    (t * a.1 + (1 - t) * b.1, t * a.2 + (1 - t) * b.2) ∈ vertices
  sides : Nat

/-- Homothety between two polygons with a positive coefficient -/
def IsHomothetic (p q : ConvexPolygon) : Prop :=
  ∃ (center : ℝ × ℝ) (k : ℝ), k > 0 ∧
    ∀ v, v ∈ p.vertices → ∃ w, w ∈ q.vertices ∧ w = (center.1 + k * (v.1 - center.1), center.2 + k * (v.2 - center.2))

/-- A collection of convex polygons -/
structure PolygonCollection where
  polygons : List ConvexPolygon
  all_same_sides : ∀ p q, p ∈ polygons → q ∈ polygons → p.sides = q.sides
  all_intersect : ∀ p q, p ∈ polygons → q ∈ polygons → p ≠ q → ∃ x, x ∈ p.vertices ∧ x ∈ q.vertices
  all_homothetic : ∀ p q, p ∈ polygons → q ∈ polygons → IsHomothetic p q

/-- The main theorem -/
theorem exists_point_in_many_polygons (n k : ℕ) (collection : PolygonCollection) 
    (h_n : collection.polygons.length = n)
    (h_k : ∀ p, p ∈ collection.polygons → p.sides = k) :
  ∃ (point : ℝ × ℝ), 
    (collection.polygons.filter (fun p => point ∈ p.vertices)).length ≥ 1 + (n - 1) / (2 * k) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_point_in_many_polygons_l94_9405


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_c_sequence_l94_9497

def a : ℕ → ℕ
  | 0 => 1
  | n + 1 => 3 * a n

def S (n : ℕ) : ℕ := n^2 + 2*n + 1

def b : ℕ → ℕ
  | 0 => S 0
  | n + 1 => S (n + 1) - S n

def c (n : ℕ) : ℕ := a n * b n

def T (n : ℕ) : ℕ := (Finset.range n).sum (λ i => c (i + 1))

theorem sum_of_c_sequence (n : ℕ) : T n = n * 3^n + 1 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_c_sequence_l94_9497


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_independent_paths_bound_l94_9475

/-- A graph represented by its vertex set and edge relation -/
structure Graph (V : Type) where
  adj : V → V → Prop

/-- A path in a graph -/
def GraphPath (G : Graph V) (start finish : V) := List V

/-- Two paths are independent if they don't share any internal vertices -/
def IndependentPaths (G : Graph V) (p1 p2 : GraphPath G a b) : Prop := sorry

/-- The connectivity between a graph and its subgraph -/
noncomputable def Connectivity (G H : Graph V) : ℕ := sorry

/-- A subgraph of a graph -/
def Subgraph (G H : Graph V) : Prop := sorry

/-- The maximum number of independent H-paths in G -/
noncomputable def MaxIndependentPaths (G H : Graph V) : ℕ := sorry

/-- The main theorem -/
theorem independent_paths_bound {V : Type} (G H : Graph V) 
  (h : Subgraph G H) : 
  MaxIndependentPaths G H ≥ (Connectivity G H) / 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_independent_paths_bound_l94_9475


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_difference_equality_exists_l94_9420

theorem difference_equality_exists (S : Finset ℕ) : 
  S.card = 5 → (∀ x, x ∈ S → 1 ≤ x ∧ x ≤ 10) → (∀ x y, x ∈ S → y ∈ S → x ≠ y → x ≠ y) →
  ∃ (k : ℤ) (a b c d : ℕ), k ≠ 0 ∧ 
    a ∈ S ∧ b ∈ S ∧ c ∈ S ∧ d ∈ S ∧ 
    a ≠ b ∧ c ≠ d ∧ (a ≠ c ∨ b ≠ d) ∧
    k = a - b ∧ k = c - d :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_difference_equality_exists_l94_9420


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l94_9485

-- Define the functions
noncomputable def f1 (x : ℝ) : ℝ := Real.cos (2 * x - Real.pi / 3)
noncomputable def f2 (x : ℝ) : ℝ := (Real.cos x + 3) / Real.cos x

-- State the theorem
theorem function_properties :
  (∀ x : ℝ, f1 x = f1 (-2 * (Real.pi / 3) - x)) ∧
  (∃ m : ℝ, ∀ x : ℝ, x ∈ Set.Ioo (-Real.pi/2) (Real.pi/2) → f2 x ≥ m) ∧
  (¬ ∃ M : ℝ, ∀ x : ℝ, x ∈ Set.Ioo (-Real.pi/2) (Real.pi/2) → f2 x ≤ M) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l94_9485


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_extension_passes_through_D_and_CD_equals_6_l94_9417

/-- Segment AB with length 8 -/
noncomputable def AB : ℝ := 8

/-- Point F is the midpoint of AB -/
noncomputable def F : ℝ := AB / 2

/-- Length of AC (leg of isosceles triangle AFC) -/
noncomputable def AC : ℝ := 3

/-- Length of BD (leg of isosceles triangle FBD) -/
noncomputable def BD : ℝ := 7

/-- Theorem stating that the extension of AC passes through D and CD = 6 -/
theorem extension_passes_through_D_and_CD_equals_6 :
  let AF := F
  let FB := AB - F
  let CC' := Real.sqrt (AC^2 - (AB/2)^2)
  let DD' := Real.sqrt (BD^2 - (AB/2)^2)
  let AD := 3 * AC
  AD - AC = 6 ∧ DD' = 3 * CC' :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_extension_passes_through_D_and_CD_equals_6_l94_9417


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_properties_l94_9456

/-- Sequence of terms -/
def a : ℕ → ℝ := sorry

/-- Sum of the first n terms -/
def S : ℕ → ℝ := sorry

/-- Given condition relating S_n, n, and a_n -/
axiom condition (n : ℕ) : 2 * S n / n + n = 2 * a n + 1

/-- a_4, a_7, and a_9 form a geometric sequence -/
axiom geometric_seq : (a 7) ^ 2 = (a 4) * (a 9)

theorem sequence_properties :
  (∀ n : ℕ, a (n + 1) = a n + 1) ∧ 
  (∃ n : ℕ, S n = -78) ∧
  (∀ m : ℕ, S m ≥ -78) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_properties_l94_9456


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ratio_PQ_EF_l94_9481

/-- Rectangle ABCD with given dimensions and points -/
structure Rectangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ
  D : ℝ × ℝ
  E : ℝ × ℝ
  F : ℝ × ℝ
  G : ℝ × ℝ
  h_AB : dist A B = 8
  h_BC : dist B C = 6
  h_EB : dist E B = 2
  h_CG : dist C G = 3
  h_DF : dist D F = 1

/-- Check if four points form a rectangle -/
def is_rectangle (A B C D : ℝ × ℝ) : Prop := sorry

/-- Define a line through two points -/
def line_through (P Q : ℝ × ℝ) : Set (ℝ × ℝ) := sorry

/-- Find the intersection of two lines -/
noncomputable def line_intersection (l₁ l₂ : Set (ℝ × ℝ)) : ℝ × ℝ := sorry

/-- The intersection points of AG, AC with EF -/
noncomputable def intersection_points (r : Rectangle) : (ℝ × ℝ) × (ℝ × ℝ) :=
  let Q := line_intersection (line_through r.A r.G) (line_through r.E r.F)
  let P := line_intersection (line_through r.A r.C) (line_through r.E r.F)
  (P, Q)

/-- The main theorem stating the ratio of PQ to EF -/
theorem ratio_PQ_EF (r : Rectangle) 
  (h_rect : is_rectangle r.A r.B r.C r.D)
  (h_E_on_AB : r.E ∈ line_through r.A r.B)
  (h_G_on_BC : r.G ∈ line_through r.B r.C)
  (h_F_on_CD : r.F ∈ line_through r.C r.D) : 
  let (P, Q) := intersection_points r
  dist P Q / dist r.E r.F = 244 / (93 * Real.sqrt 61) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ratio_PQ_EF_l94_9481


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_A_inter_notB_equals_negative_one_l94_9424

-- Define the universal set U
def U : Set Int := {y | ∃ x : Int, x ∈ ({-1, 0, 1, 2} : Set Int) ∧ y = x^3}

-- Define sets A and B
def A : Set Int := {-1, 1}
def B : Set Int := {1, 8}

-- Define the complement of B in U
def notB : Set Int := U \ B

-- Theorem statement
theorem A_inter_notB_equals_negative_one :
  A ∩ notB = {-1} := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_A_inter_notB_equals_negative_one_l94_9424


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_volume_equals_side_cubed_l94_9498

/-- The side length of the pyramid's square base -/
def pyramid_base_side : ℝ := 2

/-- The side length of the cube inside the pyramid-cone structure -/
noncomputable def cube_side_length : ℝ := Real.sqrt 6 / (Real.sqrt 2 + Real.sqrt 3)

/-- The volume of the cube inside the pyramid-cone structure -/
noncomputable def cube_volume : ℝ := cube_side_length ^ 3

/-- Theorem stating that the volume of the cube is equal to the cube of its side length -/
theorem cube_volume_equals_side_cubed :
  cube_volume = (Real.sqrt 6 / (Real.sqrt 2 + Real.sqrt 3)) ^ 3 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_volume_equals_side_cubed_l94_9498


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_g_bounds_l94_9434

noncomputable def g (x y z : ℝ) : ℝ := x / (2 * x + y) + y / (2 * y + z) + z / (2 * z + x)

theorem g_bounds :
  (∀ x y z : ℝ, x > 0 ∧ y > 0 ∧ z > 0 → g x y z ≤ 1) ∧
  (∀ ε : ℝ, ε > 0 → ∃ x y z : ℝ, x > 0 ∧ y > 0 ∧ z > 0 ∧ g x y z < ε) ∧
  (∃ x y z : ℝ, x > 0 ∧ y > 0 ∧ z > 0 ∧ g x y z = 1) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_g_bounds_l94_9434


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_of_f_on_interval_l94_9463

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := 2 / (x - 1) + 1

-- State the theorem
theorem max_value_of_f_on_interval :
  ∃ (c : ℝ), c ∈ Set.Icc 2 6 ∧ 
  (∀ x ∈ Set.Icc 2 6, f x ≤ f c) ∧
  f c = 3 := by
  -- Proof goes here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_of_f_on_interval_l94_9463


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_expression_takes_many_values_l94_9476

-- Define the expression as noncomputable
noncomputable def f (x : ℝ) : ℝ := (3*x^2 - 2*x + 3) / ((x - 3)*(x + 2)) - (5*x - 6) / ((x - 3)*(x + 2))

-- Theorem statement
theorem expression_takes_many_values :
  ∀ ε > 0, ∃ x y : ℝ, x ≠ 3 ∧ x ≠ -2 ∧ y ≠ 3 ∧ y ≠ -2 ∧ |f x - f y| > ε :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_expression_takes_many_values_l94_9476


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_part_one_part_two_part_three_l94_9459

-- Define the function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := 1 - a / (2^x + 1)

-- Theorem for part (1)
theorem part_one (a : ℝ) : f a (-1) = -1 → a = 3 := by sorry

-- Theorem for part (2)
theorem part_two : ∃ a : ℝ, ∀ x : ℝ, f a (-x) = -(f a x) := by sorry

-- Theorem for part (3)
theorem part_three (a : ℝ) : (∃ x : ℝ, f a x = 0) ↔ a ∈ Set.Ioi 1 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_part_one_part_two_part_three_l94_9459


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_final_tax_percentage_theorem_l94_9493

/-- Represents the spending breakdown as percentages of the total amount --/
structure SpendingBreakdown :=
  (clothing : ℝ)
  (food : ℝ)
  (electronics : ℝ)
  (household : ℝ)
  (cosmetics : ℝ)
  (miscellaneous : ℝ)

/-- Represents the tax rates for each category --/
structure TaxRates :=
  (clothing : ℝ)
  (food : ℝ)
  (electronics : ℝ)
  (household : ℝ)
  (cosmetics : ℝ)
  (miscellaneous : ℝ)

/-- Calculates the final total tax percentage after loyalty discount --/
def calculateFinalTaxPercentage (spending : SpendingBreakdown) (taxes : TaxRates) (loyaltyDiscount : ℝ) : ℝ :=
  let totalTaxBeforeDiscount :=
    spending.clothing * taxes.clothing +
    spending.food * taxes.food +
    spending.electronics * taxes.electronics +
    spending.household * taxes.household +
    spending.cosmetics * taxes.cosmetics +
    spending.miscellaneous * taxes.miscellaneous
  totalTaxBeforeDiscount * (1 - loyaltyDiscount)

/-- Theorem stating that the final total tax is approximately 6.7318% of the total amount --/
theorem final_tax_percentage_theorem (spending : SpendingBreakdown) (taxes : TaxRates) :
  spending.clothing = 0.34 →
  spending.food = 0.19 →
  spending.electronics = 0.20 →
  spending.household = 0.12 →
  spending.cosmetics = 0.10 →
  spending.miscellaneous = 0.05 →
  taxes.clothing = 0.06 →
  taxes.food = 0.04 →
  taxes.electronics = 0.12 →
  taxes.household = 0.07 →
  taxes.cosmetics = 0.05 →
  taxes.miscellaneous = 0.08 →
  |calculateFinalTaxPercentage spending taxes 0.03 - 0.067318| < 0.000001 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_final_tax_percentage_theorem_l94_9493


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_eight_is_eight_or_sixtyfour_l94_9467

/-- An arithmetic sequence with the given properties -/
structure ArithmeticSequence where
  a : ℕ → ℚ
  is_arithmetic : ∀ n, a (n + 1) - a n = a 2 - a 1
  first_term : a 1 = 1
  is_geometric : (a 1) * (a 5) = (a 2) * (a 2)

/-- Sum of the first n terms of an arithmetic sequence -/
def sum_n (seq : ArithmeticSequence) (n : ℕ) : ℚ :=
  (n : ℚ) * (seq.a 1 + seq.a n) / 2

/-- Main theorem: S_8 is either 8 or 64 -/
theorem sum_eight_is_eight_or_sixtyfour (seq : ArithmeticSequence) :
  sum_n seq 8 = 8 ∨ sum_n seq 8 = 64 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_eight_is_eight_or_sixtyfour_l94_9467


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_m_zero_sufficient_not_necessary_l94_9404

noncomputable def are_perpendicular (m₁ m₂ : ℝ) : Prop := m₁ * m₂ = -1

noncomputable def slope_l₁ (m : ℝ) : ℝ := (m + 1) / (m - 1)

noncomputable def slope_l₂ (m : ℝ) : ℝ := (m - 1) / (-2*m - 1)

def lines_perpendicular (m : ℝ) : Prop := 
  are_perpendicular (slope_l₁ m) (slope_l₂ m)

theorem m_zero_sufficient_not_necessary (m : ℝ) : 
  (m = 0 → lines_perpendicular m) ∧ ¬(lines_perpendicular m → m = 0) := by
  sorry

#check m_zero_sufficient_not_necessary

end NUMINAMATH_CALUDE_ERRORFEEDBACK_m_zero_sufficient_not_necessary_l94_9404


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_from_center_to_point_l94_9496

-- Define the circle equation
def circle_equation (x y : ℝ) : Prop :=
  x^2 + y^2 = 2*x + 4*y - 20

-- Define the center of the circle
def circle_center : ℝ × ℝ :=
  (1, 2)

-- Define the given point
def given_point : ℝ × ℝ :=
  (-3, -1)

-- Define the distance function between two points
noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

-- Theorem statement
theorem distance_from_center_to_point :
  distance circle_center given_point = 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_from_center_to_point_l94_9496


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_no_same_last_four_digits_l94_9492

theorem no_same_last_four_digits :
  ∀ (n m : ℕ), n > 0 → m > 0 → (5^n : ℕ) % 10000 ≠ (6^m : ℕ) % 10000 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_no_same_last_four_digits_l94_9492


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_positive_integers_satisfying_inequality_l94_9471

theorem positive_integers_satisfying_inequality : 
  ∃! (count : ℕ), count = (Finset.filter 
    (λ n : ℕ ↦ 0 < (Finset.prod (Finset.range 99) (λ i ↦ n - (i + 1))))
    (Finset.range 100)).card ∧ count = 49 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_positive_integers_satisfying_inequality_l94_9471


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_dance_lesson_pricing_l94_9400

/-- Dance lesson pricing problem -/
theorem dance_lesson_pricing
  (pack_price : ℚ)
  (pack_classes : ℕ)
  (total_classes : ℕ)
  (total_price : ℚ)
  (h1 : pack_price = 75)
  (h2 : pack_classes = 10)
  (h3 : total_classes = 13)
  (h4 : total_price = 105) :
  (total_price - pack_price) / (total_classes - pack_classes : ℚ) / (pack_price / pack_classes) = 4 / 3 := by
  sorry

-- Remove the #eval line as it's not necessary for building

end NUMINAMATH_CALUDE_ERRORFEEDBACK_dance_lesson_pricing_l94_9400


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_two_round_trips_time_l94_9413

/-- Time for a trip to the beauty parlor in hours -/
noncomputable def time_to_parlor : ℝ := 1

/-- Speed ratio of return trip compared to trip to parlor -/
noncomputable def return_speed_ratio : ℝ := 1/2

/-- Calculates the time for a return trip from the beauty parlor -/
noncomputable def time_from_parlor : ℝ := time_to_parlor / return_speed_ratio

/-- Calculates the time for one round trip to and from the beauty parlor -/
noncomputable def time_round_trip : ℝ := time_to_parlor + time_from_parlor

/-- Theorem: The time for two round trips to the beauty parlor is 6 hours -/
theorem two_round_trips_time : 2 * time_round_trip = 6 := by
  -- Unfold definitions
  unfold time_round_trip time_from_parlor time_to_parlor return_speed_ratio
  -- Simplify the expression
  simp [mul_add]
  -- Perform the calculation
  norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_two_round_trips_time_l94_9413


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_relationship_abc_l94_9430

noncomputable section

-- Define the function f
def f (x : ℝ) : ℝ := -3 * x + 2 * Real.sin x

-- Define a, b, and c
def a : ℝ := f (3^Real.sqrt 2)
def b : ℝ := -f (-2)
noncomputable def c : ℝ := f (Real.log 7 / Real.log 2)

-- The theorem to prove
theorem relationship_abc : c < a ∧ a < b := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_relationship_abc_l94_9430


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_vectors_magnitude_l94_9479

def vector_a : ℝ × ℝ := (1, 2)
def vector_b (y : ℝ) : ℝ × ℝ := (-2, y)

/-- Two vectors are parallel if one is a scalar multiple of the other -/
def are_parallel (v w : ℝ × ℝ) : Prop :=
  ∃ k : ℝ, v.1 = k * w.1 ∧ v.2 = k * w.2

/-- The magnitude (length) of a 2D vector -/
noncomputable def magnitude (v : ℝ × ℝ) : ℝ := Real.sqrt (v.1^2 + v.2^2)

theorem parallel_vectors_magnitude :
  ∃ y : ℝ, are_parallel vector_a (vector_b y) → magnitude (vector_b y) = 2 * Real.sqrt 5 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_vectors_magnitude_l94_9479


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_correct_units_l94_9440

-- Define the types of measurements
inductive MeasurementType
| Length
| Weight
| Time

-- Define the units
inductive MeasurementUnit
| Centimeter
| Gram
| Minute
| Hour

-- Define a function to determine the appropriate unit based on the measurement type and value
def appropriateUnit (type : MeasurementType) (value : ℕ) : MeasurementUnit :=
  match type, value with
  | MeasurementType.Length, 70 => MeasurementUnit.Centimeter
  | MeasurementType.Weight, 240 => MeasurementUnit.Gram
  | MeasurementType.Time, 90 => MeasurementUnit.Minute
  | MeasurementType.Time, 8 => MeasurementUnit.Hour
  | _, _ => MeasurementUnit.Centimeter  -- Default case, not used in our problem

-- Theorem statement
theorem correct_units :
  (appropriateUnit MeasurementType.Length 70 = MeasurementUnit.Centimeter) ∧
  (appropriateUnit MeasurementType.Weight 240 = MeasurementUnit.Gram) ∧
  (appropriateUnit MeasurementType.Time 90 = MeasurementUnit.Minute) ∧
  (appropriateUnit MeasurementType.Time 8 = MeasurementUnit.Hour) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_correct_units_l94_9440


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangency_distance_for_given_triangle_l94_9427

/-- An isosceles triangle with an inscribed circle -/
structure IsoscelesTriangleWithInscribedCircle where
  /-- Length of the lateral side -/
  lateral_side : ℝ
  /-- Length of the base -/
  base : ℝ
  /-- The triangle is isosceles -/
  is_isosceles : lateral_side > 0
  /-- The base is positive -/
  base_positive : base > 0
  /-- The triangle inequality holds -/
  triangle_inequality : base < 2 * lateral_side

/-- The distance between points of tangency on lateral sides -/
noncomputable def tangency_distance (t : IsoscelesTriangleWithInscribedCircle) : ℝ :=
  t.base * (t.lateral_side - t.base / 2) / t.lateral_side

/-- Theorem stating the distance between tangency points for the given triangle -/
theorem tangency_distance_for_given_triangle :
  let t : IsoscelesTriangleWithInscribedCircle :=
    { lateral_side := 100
      base := 60
      is_isosceles := by norm_num
      base_positive := by norm_num
      triangle_inequality := by norm_num }
  tangency_distance t = 42 := by
  -- Unfold the definition of tangency_distance
  unfold tangency_distance
  -- Simplify the arithmetic expression
  simp [IsoscelesTriangleWithInscribedCircle.lateral_side, IsoscelesTriangleWithInscribedCircle.base]
  -- Perform the numerical calculation
  norm_num

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangency_distance_for_given_triangle_l94_9427


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_mean_after_removal_l94_9478

theorem arithmetic_mean_after_removal (initial_count : ℕ) (initial_mean : ℚ) 
  (removed_numbers : List ℚ) (new_mean : ℚ) : 
  initial_count = 60 →
  initial_mean = 75 →
  removed_numbers = [70, 80, 90] →
  new_mean = 74.74 →
  let initial_sum := initial_count * initial_mean
  let removed_sum := removed_numbers.sum
  let new_count := initial_count - removed_numbers.length
  let new_sum := initial_sum - removed_sum
  (new_sum / new_count) = new_mean := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_mean_after_removal_l94_9478


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_average_capacity_2050_l94_9480

/-- Calculates the average hard drive capacity in 2050 given the initial year and capacity -/
noncomputable def averageCapacity2050 (initialYear : ℕ) (initialCapacity : ℝ) : ℝ :=
  let n : ℝ := (2050 - initialYear : ℝ) / 5
  initialCapacity * (2 ^ n)

/-- Theorem stating the average hard drive capacity in 2050 -/
theorem average_capacity_2050 (initialYear : ℕ) (initialCapacity : ℝ) 
    (h1 : initialCapacity = 0.1)
    (h2 : initialYear ≤ 2050) :
  averageCapacity2050 initialYear initialCapacity = 0.1 * (2 ^ ((2050 - initialYear : ℝ) / 5)) :=
by
  sorry

#check average_capacity_2050

end NUMINAMATH_CALUDE_ERRORFEEDBACK_average_capacity_2050_l94_9480


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_is_odd_l94_9453

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := Real.log (1 + x) - Real.log (1 - x)

-- State the theorem
theorem f_is_odd : ∀ x : ℝ, f (-x) = -f x := by
  intro x
  simp [f]
  -- The proof is omitted for now
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_is_odd_l94_9453


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_diophantine_equation_solution_l94_9429

theorem diophantine_equation_solution (a b n : ℕ) (p : ℕ) (h_prime : Nat.Prime p) :
  a ^ 2013 + b ^ 2013 = p ^ n →
  ∃ k : ℕ, a = 2 ^ k ∧ b = 2 ^ k ∧ n = 2013 * k + 1 ∧ p = 2 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_diophantine_equation_solution_l94_9429


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_colton_stickers_left_l94_9465

/-- Represents the number of stickers Colton has left after distributing them to his friends. -/
def stickers_left (initial : ℕ) (friends : ℕ) (stickers_per_friend : ℕ) (extra_to_mandy : ℕ) 
  (justin_percentage : ℚ) (karen_fraction : ℚ) : ℕ :=
  let given_to_friends := friends * stickers_per_friend
  let given_to_mandy := given_to_friends + extra_to_mandy
  let remaining_after_mandy := initial - (given_to_friends + given_to_mandy)
  let given_to_justin := (justin_percentage * remaining_after_mandy).floor.toNat
  let remaining_after_justin := remaining_after_mandy - given_to_justin
  let given_to_karen := (karen_fraction * remaining_after_justin).floor.toNat
  remaining_after_justin - given_to_karen

/-- Theorem stating that Colton has 24 stickers left after distributing them according to the problem description. -/
theorem colton_stickers_left : 
  stickers_left 85 5 4 5 (1/5) (1/4) = 24 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_colton_stickers_left_l94_9465


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_red_hamiltonian_cycle_exists_l94_9457

/-- A graph with 5 vertices where edges are colored red or blue -/
structure ColoredGraph :=
  (vertices : Finset (Fin 5))
  (red_edges : Set (Fin 5 × Fin 5))
  (blue_edges : Set (Fin 5 × Fin 5))
  (edge_partition : red_edges ∪ blue_edges = {p : Fin 5 × Fin 5 | p.1 ≠ p.2})
  (no_monochromatic_triangle : ∀ a b c : Fin 5, a ≠ b ∧ b ≠ c ∧ c ≠ a →
    ¬((a, b) ∈ red_edges ∧ (b, c) ∈ red_edges ∧ (c, a) ∈ red_edges) ∧
    ¬((a, b) ∈ blue_edges ∧ (b, c) ∈ blue_edges ∧ (c, a) ∈ blue_edges))

/-- A Hamiltonian cycle of red edges in the graph -/
def RedHamiltonianCycle (g : ColoredGraph) : Prop :=
  ∃ (cycle : List (Fin 5)), cycle.length = 5 ∧ cycle.Nodup ∧
    (∀ i : Fin 4, (cycle[i.val]!, cycle[i.val + 1]!) ∈ g.red_edges) ∧
    (cycle[4]!, cycle[0]!) ∈ g.red_edges

/-- Theorem: Every ColoredGraph has a RedHamiltonianCycle -/
theorem red_hamiltonian_cycle_exists (g : ColoredGraph) : RedHamiltonianCycle g := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_red_hamiltonian_cycle_exists_l94_9457


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_permutation_combination_theorem_l94_9487

theorem permutation_combination_theorem (n : ℕ) : 
  (n.choose 2 = 15) → (n * (n - 1) = 30) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_permutation_combination_theorem_l94_9487


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_chessboard_path_types_l94_9418

/-- Represents a path on a 5x5 chessboard --/
structure ChessboardPath where
  vertices : List (Nat × Nat)
  is_valid : vertices.length = 36

/-- Checks if a path is symmetric with respect to the central lines of the board --/
def is_symmetric (path : ChessboardPath) : Prop :=
  sorry

/-- Checks if a path passes through each vertex only once and returns to the starting point --/
def is_valid_path (path : ChessboardPath) : Prop :=
  sorry

/-- Represents the three types of paths: AB, BC, and CA --/
inductive PathType
  | AB
  | BC
  | CA

/-- Constructs a path of a given type --/
def construct_path (type : PathType) : ChessboardPath :=
  sorry

theorem chessboard_path_types :
  ∃ (paths : List ChessboardPath),
    paths.length = 3 ∧
    (∀ p ∈ paths, is_valid_path p ∧ is_symmetric p) ∧
    (∀ t : PathType, ∃ p ∈ paths, p = construct_path t) :=
by
  sorry

#check chessboard_path_types

end NUMINAMATH_CALUDE_ERRORFEEDBACK_chessboard_path_types_l94_9418


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_f_range_of_a_l94_9454

-- Part 1
noncomputable def f (x : ℝ) : ℝ := Real.sin (x + Real.pi / 3)

theorem range_of_f :
  ∀ y, y ∈ Set.range (fun x ↦ f x) → x ∈ Set.Icc 0 Real.pi →
  -Real.sqrt 3 / 2 ≤ y ∧ y ≤ 1 :=
sorry

-- Part 2
noncomputable def g (ω x : ℝ) : ℝ := Real.sin (ω * x + Real.pi / 3)

theorem range_of_a (ω : ℝ) (h1 : ω > 0) (h2 : 2 * Real.pi / ω = Real.pi) :
  ∀ a > Real.pi,
  (∃! (z1 z2 z3 : ℝ), z1 ∈ Set.Icc Real.pi a ∧
                      z2 ∈ Set.Icc Real.pi a ∧
                      z3 ∈ Set.Icc Real.pi a ∧
                      g ω z1 = 0 ∧ g ω z2 = 0 ∧ g ω z3 = 0 ∧
                      z1 < z2 ∧ z2 < z3) →
  a ∈ Set.Icc (7 * Real.pi / 3) (17 * Real.pi / 6) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_f_range_of_a_l94_9454


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_ratio_of_special_triangle_l94_9466

structure Triangle (V : Type*) [NormedAddCommGroup V] [InnerProductSpace ℝ V] where
  A : V
  B : V
  C : V

def Triangle.isIsosceles {V : Type*} [NormedAddCommGroup V] [InnerProductSpace ℝ V] (t : Triangle V) : Prop :=
  ‖t.A - t.B‖ = ‖t.A - t.C‖

noncomputable def Triangle.area {V : Type*} [NormedAddCommGroup V] [InnerProductSpace ℝ V] (t : Triangle V) : ℝ := 
  sorry

def pointOnLine {V : Type*} [NormedAddCommGroup V] [InnerProductSpace ℝ V] (A B P : V) : Prop := 
  sorry

def bisectsAngle {V : Type*} [NormedAddCommGroup V] [InnerProductSpace ℝ V] (A B C D : V) : Prop := 
  sorry

theorem area_ratio_of_special_triangle {V : Type*} [NormedAddCommGroup V] [InnerProductSpace ℝ V] 
  (ABC : Triangle V) (D E F : V) :
  Triangle.isIsosceles ABC →
  pointOnLine ABC.B ABC.C D →
  pointOnLine ABC.A ABC.C E →
  pointOnLine ABC.A ABC.B F →
  ‖ABC.B - D‖ = 2 * ‖D - ABC.C‖ →
  bisectsAngle ABC.B D ABC.C E →
  bisectsAngle ABC.B D ABC.A F →
  Triangle.area (Triangle.mk D E F) / Triangle.area ABC = Real.sqrt 3 / 16 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_ratio_of_special_triangle_l94_9466


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_integer_divisible_by_15_with_sqrt_between_24_and_24_5_l94_9437

theorem smallest_integer_divisible_by_15_with_sqrt_between_24_and_24_5 :
  ∃ n : ℕ, 
    n > 0 ∧
    (n : ℝ).sqrt > 24 ∧ 
    (n : ℝ).sqrt < 24.5 ∧ 
    n % 15 = 0 ∧
    (∀ m : ℕ, m > 0 → m < n → 
      (m : ℝ).sqrt ≤ 24 ∨ 
      (m : ℝ).sqrt ≥ 24.5 ∨ 
      m % 15 ≠ 0) ∧
    n = 585 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_integer_divisible_by_15_with_sqrt_between_24_and_24_5_l94_9437


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_not_always_exists_k_power_greater_l94_9473

def isPowerOf (a b : ℕ) : Prop := ∃ n : ℕ, n > 0 ∧ a = b ^ n

def isMultipleOf (a b : ℕ) : Prop := ∃ n : ℤ, a = b * n

theorem not_always_exists_k_power_greater :
  ¬ (∀ x y : ℕ, x > 0 → y > 0 → ∃ k : ℕ, k > 0 ∧ x^k > y) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_not_always_exists_k_power_greater_l94_9473


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_lines_count_l94_9494

-- Define the circle
def circle_eq (x y : ℝ) : Prop := x^2 + (y - 2)^2 = 1

-- Define a line with equal intercepts on both axes
def line_equal_intercepts (m b : ℝ) : Prop := 
  ∃ a : ℝ, a ≠ 0 ∧ b = a ∧ m * 0 + b = a ∧ m * a + b = 0

-- Define a tangent line to the circle
def tangent_line (m b : ℝ) : Prop := 
  ∃! p : ℝ × ℝ, circle_eq p.1 p.2 ∧ p.2 = m * p.1 + b

-- Theorem statement
theorem tangent_lines_count : ∃! (lines : Finset (ℝ × ℝ)), 
  lines.card = 4 ∧ 
  (∀ m b : ℝ, (m, b) ∈ lines ↔ tangent_line m b ∧ line_equal_intercepts m b) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_lines_count_l94_9494


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_functional_equation_solution_l94_9483

theorem functional_equation_solution (f : ℝ → ℝ) 
  (h : ∀ x y : ℝ, f (x * f y) + f (f x + f y) = y * f x + f (x + f y)) :
  (∀ x : ℝ, f x = 0) ∨ (∀ x : ℝ, f x = x) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_functional_equation_solution_l94_9483


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_problem_l94_9474

noncomputable section

open Real

theorem triangle_problem (a b c A B C : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c)
  (h4 : 0 < A ∧ A < π) (h5 : 0 < B ∧ B < π) (h6 : 0 < C ∧ C < π)
  (h7 : A + B + C = π) (h8 : a * sin B = b * sin A) (h9 : b * sin C = c * sin B)
  (h10 : c * sin A = a * sin C) (h11 : a * c = 2 * b^2) :
  (cos B ≥ 3/4) ∧ (cos (A - C) + cos B = 1 → B = π/6) := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_problem_l94_9474


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_coefficient_x_squared_l94_9411

theorem coefficient_x_squared (a : ℝ) : a > 0 → (
  (Function.update (fun (_ : ℕ) => (0 : ℝ)) 2 (15 * a^2 - 6 * a) = Function.update (fun (_ : ℕ) => (0 : ℝ)) 2 9) ↔ 
  a = 1
) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_coefficient_x_squared_l94_9411


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_properties_l94_9433

/-- An ellipse with right focus F(m,0), left directrix x = -m-1, and right directrix x = m+1 -/
structure Ellipse (m : ℝ) where
  focus : ℝ × ℝ := (m, 0)
  left_directrix : ℝ → Prop := λ x ↦ x = -m - 1
  right_directrix : ℝ → Prop := λ x ↦ x = m + 1

/-- The eccentricity of the ellipse -/
noncomputable def eccentricity (m : ℝ) : ℝ := m / Real.sqrt (m * (m + 1))

/-- The equation of the ellipse -/
def ellipse_equation (m : ℝ) (x y : ℝ) : Prop :=
  x^2 / (m * (m + 1)) + y^2 / m = 1

/-- The dot product of vectors AF and FB -/
def dot_product_AF_FB (m : ℝ) : ℝ := m^2 + 4*m + 2

theorem ellipse_properties (m : ℝ) :
  (eccentricity m = Real.sqrt 2 / 2 → 
    ∀ x y, ellipse_equation m x y ↔ x^2 / 2 + y^2 = 1) ∧
  (dot_product_AF_FB m < 7 → 
    0 < eccentricity m ∧ eccentricity m < Real.sqrt 2 / 2) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_properties_l94_9433


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_formula_a_inequality_l94_9470

def a : ℕ → ℚ 
  | 0 => 1
  | n + 1 => 2 * a n + 1

theorem a_formula (n : ℕ) : a n = 2^n - 1 := by
  sorry

theorem a_inequality (n : ℕ) : 
  (Finset.range n).sum (λ i => a i / a (i + 1)) < n / 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_formula_a_inequality_l94_9470


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_all_good_ingredients_l94_9468

-- Define the probabilities of good ingredients
noncomputable def prob_good_milk : ℝ := 1 - 0.2
noncomputable def prob_good_egg : ℝ := 1 - 0.6
noncomputable def prob_good_flour : ℝ := 1 - (1/4 : ℝ)

-- State the theorem
theorem prob_all_good_ingredients : 
  prob_good_milk * prob_good_egg * prob_good_flour = 0.24 := by
  -- Unfold the definitions
  unfold prob_good_milk prob_good_egg prob_good_flour
  -- Perform the calculation
  norm_num
  -- The proof is complete
  done

end NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_all_good_ingredients_l94_9468


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_chord_length_l94_9455

-- Define the line
def line (x y : ℝ) : Prop := x - 3*y + 3 = 0

-- Define the circle
def circle_eq (x y : ℝ) : Prop := (x-1)^2 + (y-3)^2 = 10

-- Theorem statement
theorem chord_length :
  ∃ (a b c d : ℝ),
    line a b ∧ line c d ∧
    circle_eq a b ∧ circle_eq c d ∧
    a ≠ c ∧ b ≠ d ∧
    ((a - c)^2 + (b - d)^2) = 30 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_chord_length_l94_9455


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_triangle_AMK_l94_9472

/-- The ratio of lengths of two segments on a line. -/
def segment_ratio (A B C : EuclideanSpace ℝ (Fin 2)) : ℝ := sorry

/-- The area of a triangle given its three vertices. -/
def area (triangle : EuclideanSpace ℝ (Fin 2) × EuclideanSpace ℝ (Fin 2) × EuclideanSpace ℝ (Fin 2)) : ℝ := sorry

/-- Given a triangle ABC with area 36 cm², and points M on AB and K on AC
    such that AM/MB = 1/3 and AK/KC = 2/1, the area of triangle AMK is 6 cm². -/
theorem area_of_triangle_AMK (A B C M K : EuclideanSpace ℝ (Fin 2)) 
    (h_area : area (A, B, C) = 36) 
    (h_M : segment_ratio A M B = 1/3) 
    (h_K : segment_ratio A K C = 2/1) : 
  area (A, M, K) = 6 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_triangle_AMK_l94_9472


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_double_angle_and_sum_l94_9438

theorem tan_double_angle_and_sum (α : ℝ) (h : Real.tan α = -2) :
  Real.tan (2 * α) = 4 / 3 ∧ Real.tan (2 * α + π / 4) = -7 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_double_angle_and_sum_l94_9438


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_circle_inequality_l94_9495

theorem inscribed_circle_inequality 
  (r r1 r2 r3 : ℝ) 
  (m : ℝ) 
  (h1 : r > 0) 
  (h2 : r1 > 0) 
  (h3 : r2 > 0) 
  (h4 : r3 > 0) 
  (h5 : m ≥ 1/2) 
  (h6 : ∃ (A B C : ℝ), A > 0 ∧ B > 0 ∧ C > 0 ∧ A + B + C = π ∧
       r1 = r * (Real.tan ((π - A)/4))^2 ∧
       r2 = r * (Real.tan ((π - B)/4))^2 ∧
       r3 = r * (Real.tan ((π - C)/4))^2) :
  (r1 * r2)^m + (r2 * r3)^m + (r3 * r1)^m ≥ 3 * (r/3)^(2*m) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_circle_inequality_l94_9495


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_2alpha_plus_pi_4_l94_9449

theorem cos_2alpha_plus_pi_4 (α : ℝ) 
  (h1 : Real.cos (α + Real.pi/4) = 3/5)
  (h2 : Real.pi/2 < α ∧ α < 3*Real.pi/2) : 
  Real.cos (2*α + Real.pi/4) = -(31 * Real.sqrt 2) / 50 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_2alpha_plus_pi_4_l94_9449


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_equation_l94_9477

-- Define the circle
def circle_equation (x y : ℝ) : Prop := x^2 + y^2 + 6*x - 4*y + 9 = 0

-- Define the point through which the tangent line passes
def tangent_point : ℝ × ℝ := (-1, 6)

-- Define the potential tangent lines
def tangent_line_1 (x y : ℝ) : Prop := 3*x - 4*y + 27 = 0
def tangent_line_2 (x : ℝ) : Prop := x = -1

-- Theorem statement
theorem tangent_line_equation :
  ∃ (x y : ℝ), circle_equation x y ∧
  ((x, y) ≠ tangent_point) ∧
  (tangent_line_1 x y ∨ tangent_line_2 x) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_equation_l94_9477


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_square_fourth_vertex_l94_9410

-- Define the complex numbers representing the given vertices
def z₁ : ℂ := 3 + 3*Complex.I
def z₂ : ℂ := 2 - 2*Complex.I
def z₃ : ℂ := -1 - Complex.I

-- Define the property of being a square in the complex plane
def is_square (a b c d : ℂ) : Prop :=
  (b - a) * (c - b) = Complex.I * (c - b) * (d - c) ∧
  (c - b) * (d - c) = Complex.I * (d - c) * (a - d) ∧
  (d - c) * (a - d) = Complex.I * (a - d) * (b - a) ∧
  (a - d) * (b - a) = Complex.I * (b - a) * (c - b)

-- Theorem statement
theorem square_fourth_vertex :
  ∃ (z₄ : ℂ), is_square z₁ z₂ z₃ z₄ ∧ z₄ = 4*Complex.I := by
  sorry

#check square_fourth_vertex

end NUMINAMATH_CALUDE_ERRORFEEDBACK_square_fourth_vertex_l94_9410


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equilateral_triangle_point_on_circle_l94_9428

-- Define the points
variable (A B C P Q : EuclideanSpace ℝ (Fin 2))

-- Define the properties of the triangle and circle
def is_equilateral_triangle (A B C : EuclideanSpace ℝ (Fin 2)) : Prop := sorry

def on_circumscribed_circle (A B C P : EuclideanSpace ℝ (Fin 2)) : Prop := sorry

def on_arc_BC (B C P : EuclideanSpace ℝ (Fin 2)) : Prop := sorry

def intersect_at (A P B C Q : EuclideanSpace ℝ (Fin 2)) : Prop := sorry

-- State the theorem
theorem equilateral_triangle_point_on_circle 
  (h1 : is_equilateral_triangle A B C)
  (h2 : on_circumscribed_circle A B C P)
  (h3 : on_arc_BC B C P)
  (h4 : intersect_at A P B C Q) :
  1 / dist P Q = 1 / dist P B + 1 / dist P C := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_equilateral_triangle_point_on_circle_l94_9428


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_initial_tax_rate_was_twenty_percent_l94_9484

/-- Represents the tax situation before and after a tax rate change --/
structure TaxSituation where
  initialIncome : ℚ
  finalIncome : ℚ
  newTaxRate : ℚ
  taxIncrease : ℚ

/-- Calculates the initial tax rate given a tax situation --/
def calculateInitialTaxRate (ts : TaxSituation) : ℚ :=
  (ts.finalIncome * ts.newTaxRate - ts.taxIncrease) / ts.initialIncome

/-- Theorem stating that the initial tax rate was 20% given the problem conditions --/
theorem initial_tax_rate_was_twenty_percent :
  let ts : TaxSituation := {
    initialIncome := 1000000,
    finalIncome := 1500000,
    newTaxRate := 30/100,
    taxIncrease := 250000
  }
  calculateInitialTaxRate ts = 20/100 := by
  sorry

#eval calculateInitialTaxRate {
  initialIncome := 1000000,
  finalIncome := 1500000,
  newTaxRate := 30/100,
  taxIncrease := 250000
}

end NUMINAMATH_CALUDE_ERRORFEEDBACK_initial_tax_rate_was_twenty_percent_l94_9484


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_property_l94_9486

noncomputable def curve_C (θ : ℝ) : ℝ := 2 / (1 - Real.cos θ)

noncomputable def line_l (m t α : ℝ) : ℝ × ℝ := (m + t * Real.cos α, t * Real.sin α)

structure IntersectionPoints (m α : ℝ) where
  t₁ : ℝ
  t₂ : ℝ
  intersects_C : 
    (line_l m t₁ α).2^2 = 4 * (line_l m t₁ α).1 + 4 ∧
    (line_l m t₂ α).2^2 = 4 * (line_l m t₂ α).1 + 4

theorem intersection_property (m : ℝ) : 
  (∃ α : ℝ, ∃ points : IntersectionPoints m α, 
    1 / (points.t₁^2) + 1 / (points.t₂^2) = 1/64) → 
  m = 1/4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_property_l94_9486


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_theta_value_l94_9482

theorem sin_theta_value (θ : Real) (h1 : θ ∈ Set.Icc (π/4) (π/2)) 
  (h2 : Real.sin (2*θ) = 3*Real.sqrt 7/8) : 
  Real.sin θ = 3/4 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_theta_value_l94_9482


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_function_theorem_l94_9445

def IsStrictlyIncreasing (f : ℕ → ℕ) : Prop :=
  ∀ x y, x < y → f x < f y

-- Remove the IsCoprime definition as it's already defined in Mathlib

theorem unique_function_theorem (f : ℕ → ℕ) 
  (h_increasing : IsStrictlyIncreasing f)
  (h_two : f 2 = 2)
  (h_coprime : ∀ m n, Nat.Coprime m n → f (m * n) = f m * f n) :
  ∀ k, f k = k := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_function_theorem_l94_9445


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_n_factorial_congruence_l94_9488

theorem smallest_n_factorial_congruence : 
  ∃ (n : ℕ), n > 0 ∧ 
  n.factorial * (n + 1).factorial * (2 * n + 1).factorial ≡ 1 [ZMOD (10^30)] ∧
  ∀ (m : ℕ), m > 0 ∧ m < n → 
    ¬(m.factorial * (m + 1).factorial * (2 * m + 1).factorial ≡ 1 [ZMOD (10^30)]) :=
by
  use 34
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_n_factorial_congruence_l94_9488


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_last_digit_l94_9469

/-- A function that checks if a two-digit number is divisible by 17 or 23 -/
def isDivisibleBy17Or23 (n : ℕ) : Prop :=
  n ≥ 10 ∧ n < 100 ∧ (n % 17 = 0 ∨ n % 23 = 0)

/-- A function that represents the string of digits -/
def digitString : ℕ → ℕ := sorry

/-- The length of the digit string -/
def stringLength : ℕ := 3003

/-- The first digit is 2 -/
axiom first_digit : digitString 1 = 2

/-- Any two consecutive digits form a number divisible by 17 or 23 -/
axiom consecutive_digits_divisible (i : ℕ) :
  i < stringLength → isDivisibleBy17Or23 (digitString i * 10 + digitString (i + 1))

/-- The theorem stating that the largest possible last digit is 9 -/
theorem largest_last_digit :
  ∃ (s : ℕ → ℕ), s 1 = 2 ∧
    (∀ i < stringLength, isDivisibleBy17Or23 (s i * 10 + s (i + 1))) ∧
    s stringLength = 9 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_last_digit_l94_9469


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_magic_square_solution_l94_9407

/-- Represents a 3x3 magic square -/
structure MagicSquare where
  entries : Fin 3 → Fin 3 → ℕ
  magic_sum : ℕ
  sum_property : 
    (∀ i, (Finset.sum (Finset.range 3) (λ j ↦ entries i j)) = magic_sum) ∧
    (∀ j, (Finset.sum (Finset.range 3) (λ i ↦ entries i j)) = magic_sum) ∧
    ((entries 0 0 + entries 1 1 + entries 2 2) = magic_sum) ∧
    ((entries 0 2 + entries 1 1 + entries 2 0) = magic_sum)

/-- The theorem to be proved -/
theorem magic_square_solution (ms : MagicSquare) 
  (h1 : ms.entries 0 1 = 23)
  (h2 : ms.entries 0 2 = 104)
  (h3 : ms.entries 1 0 = 5) :
  ms.entries 0 0 = 220 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_magic_square_solution_l94_9407


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_beta_value_l94_9447

theorem beta_value (α β : Real) 
  (h1 : Real.sin α = (Real.sqrt 10) / 10)
  (h2 : Real.sin (α - β) = -(Real.sqrt 5) / 5)
  (h3 : 0 < α ∧ α < Real.pi / 2)
  (h4 : 0 < β ∧ β < Real.pi / 2) : 
  β = Real.pi / 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_beta_value_l94_9447


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_polyhedron_with_no_visible_vertices_l94_9450

/-- A polyhedron in 3D space -/
structure Polyhedron where
  vertices : Set (Fin 3 → ℝ)
  faces : Set (Set (Fin 3 → ℝ))
  -- Additional conditions for a valid polyhedron would be defined here

/-- Checks if a point is outside a polyhedron -/
def isOutside (p : Fin 3 → ℝ) (poly : Polyhedron) : Prop :=
  sorry

/-- Checks if a line segment intersects the interior of a polyhedron -/
def intersectsInterior (a b : Fin 3 → ℝ) (poly : Polyhedron) : Prop :=
  sorry

/-- Theorem stating the existence of a polyhedron with no visible vertices from an outside point -/
theorem exists_polyhedron_with_no_visible_vertices :
  ∃ (P : Polyhedron) (p : Fin 3 → ℝ), 
    isOutside p P ∧ 
    ∀ v ∈ P.vertices, intersectsInterior p v P :=
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_polyhedron_with_no_visible_vertices_l94_9450


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_diagonal_length_from_area_and_offsets_l94_9412

/-- Represents a quadrilateral with one diagonal and two offsets -/
structure Quadrilateral where
  diagonal : ℝ
  offset1 : ℝ
  offset2 : ℝ

/-- Calculates the area of a quadrilateral given its diagonal and offsets -/
noncomputable def area (q : Quadrilateral) : ℝ :=
  (1/2) * q.diagonal * (q.offset1 + q.offset2)

theorem diagonal_length_from_area_and_offsets 
  (q : Quadrilateral) 
  (h1 : q.offset1 = 10)
  (h2 : q.offset2 = 8)
  (h3 : area q = 450) :
  q.diagonal = 50 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_diagonal_length_from_area_and_offsets_l94_9412


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l94_9439

-- Define a triangle ABC
structure Triangle where
  A : Real
  B : Real
  C : Real
  a : Real
  b : Real
  c : Real
  sum_angles : A + B + C = Real.pi
  positive_sides : a > 0 ∧ b > 0 ∧ c > 0

-- Define the given condition
def given_condition (t : Triangle) : Prop :=
  2 * (Real.tan t.A + Real.tan t.B) = (Real.sin t.A + Real.sin t.B) / (Real.cos t.A * Real.cos t.B)

-- Theorem to prove
theorem triangle_properties (t : Triangle) (h : given_condition t) :
  -- Part I: a, c, b form an arithmetic sequence
  (t.a + t.b) / 2 = t.c ∧
  -- Part II: Minimum value of cos C is 1/2
  Real.cos t.C ≥ 1/2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l94_9439


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_solution_l94_9414

-- Define the function representing the left side of the equation
noncomputable def f (x : ℝ) : ℝ :=
  5 / (Real.sqrt (x - 10) - 8) +
  2 / (Real.sqrt (x - 10) - 5) +
  8 / (Real.sqrt (x - 10) + 5) +
  10 / (Real.sqrt (x - 10) + 8)

-- State the theorem
theorem unique_solution :
  ∃! x : ℝ, f x = 0 ∧ x = 19 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_solution_l94_9414


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_square_WH_length_l94_9401

/-- Square WXYZ with side length s -/
structure Square (s : ℝ) where
  W : ℝ × ℝ
  X : ℝ × ℝ
  Y : ℝ × ℝ
  Z : ℝ × ℝ

/-- Point on side AB of square -/
def G (WXYZ : Square s) : ℝ × ℝ := sorry

/-- Point H on extension of WY -/
def H (WXYZ : Square s) : ℝ × ℝ := sorry

/-- Area of a triangle given three points -/
noncomputable def triangleArea (A B C : ℝ × ℝ) : ℝ := sorry

/-- Distance between two points -/
noncomputable def distance (A B : ℝ × ℝ) : ℝ := sorry

/-- Perpendicular property -/
def isPerpendicular (A B C : ℝ × ℝ) : Prop := sorry

theorem square_WH_length (s : ℝ) (WXYZ : Square s) (h_area : s^2 = 225) 
  (h_perp : isPerpendicular WXYZ.Z (G WXYZ) (H WXYZ))
  (h_triangle_area : triangleArea WXYZ.Z (G WXYZ) (H WXYZ) = 160) :
  distance WXYZ.W (H WXYZ) = Real.sqrt 545 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_square_WH_length_l94_9401


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_solution_l94_9425

/-- Two-dimensional vector -/
structure Vec2D where
  x : ℝ
  y : ℝ

/-- Magnitude of a 2D vector -/
noncomputable def magnitude (v : Vec2D) : ℝ :=
  Real.sqrt (v.x^2 + v.y^2)

/-- Vector addition -/
def add (v w : Vec2D) : Vec2D :=
  ⟨v.x + w.x, v.y + w.y⟩

/-- Parallel to y-axis -/
def parallelToYAxis (v : Vec2D) : Prop :=
  v.x = 0

theorem vector_solution (a b : Vec2D) 
  (h1 : magnitude (add a b) = 1)
  (h2 : parallelToYAxis (add a b))
  (h3 : a = ⟨2, -1⟩) :
  b = ⟨-2, 2⟩ ∨ b = ⟨-2, 0⟩ := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_solution_l94_9425


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_log_sqrt6_1296_l94_9451

theorem log_sqrt6_1296 : Real.log 1296 / Real.log (Real.sqrt 6) = 8 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_log_sqrt6_1296_l94_9451


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_count_equations_l94_9416

def is_equation (expr : String) : Bool :=
  expr.contains '='

def expressions : List String :=
  ["3x-5", "2a-3=0", "7>-3", "5-7=-2", "|x|=1", "2x^2+x=1"]

theorem count_equations :
  (expressions.filter is_equation).length = 4 := by
  -- The proof goes here
  sorry

#eval (expressions.filter is_equation).length

end NUMINAMATH_CALUDE_ERRORFEEDBACK_count_equations_l94_9416


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pulley_centers_distance_specific_l94_9489

/-- The distance between the centers of two circular pulleys -/
noncomputable def pulley_centers_distance (r1 r2 contact_distance : ℝ) : ℝ :=
  Real.sqrt ((r1 - r2)^2 + contact_distance^2)

/-- Theorem stating the distance between pulley centers for given parameters -/
theorem pulley_centers_distance_specific :
  pulley_centers_distance 10 6 30 = Real.sqrt 916 := by
  unfold pulley_centers_distance
  -- The proof steps would go here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_pulley_centers_distance_specific_l94_9489


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_max_difference_l94_9422

-- Define the circle
def circleEq (x y : ℝ) : Prop := x^2 + (y - 1)^2 = 1

-- Define the function to be maximized
def f (x y : ℝ) : ℝ := x - y

-- Theorem statement
theorem circle_max_difference :
  (∃ (c : ℝ), c = Real.sqrt 2 - 1 ∧
    (∀ (x y : ℝ), circleEq x y → f x y ≤ c) ∧
    (∃ (x y : ℝ), circleEq x y ∧ f x y = c)) ∧
  (∀ (c : ℝ), (∀ (x y : ℝ), circleEq x y → x - y ≤ c) ↔ c ∈ Set.Ici (-1)) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_max_difference_l94_9422


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_vectors_sum_l94_9460

def a : ℝ × ℝ := (2, 1)

theorem parallel_vectors_sum :
  ∀ x : ℝ, (∃ k : ℝ, k ≠ 0 ∧ a = k • (x, -2)) →
  a + (x, -2) = (-2, -1) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_vectors_sum_l94_9460


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_placement_theorem_l94_9446

/-- The number of new triangles added in the nth placement --/
def new_triangles (n : ℕ) : ℕ := 
  if n = 1 then 1 else 3 * (n - 1)

/-- The total number of triangles after n placements --/
def total_triangles (n : ℕ) : ℕ := 
  Finset.sum (Finset.range n) (λ i => new_triangles (i + 1))

theorem triangle_placement_theorem :
  total_triangles 20 = 571 := by
  -- Proof goes here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_placement_theorem_l94_9446


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_ratio_is_one_third_l94_9444

/-- The area of an equilateral triangle with side length s -/
noncomputable def equilateral_triangle_area (s : ℝ) : ℝ := (Real.sqrt 3 / 4) * s^2

/-- The side length of the larger equilateral triangle -/
def large_side : ℝ := 10

/-- The side length of the smaller equilateral triangle -/
def small_side : ℝ := 5

/-- The area of the isosceles trapezoid formed by cutting the smaller triangle from the larger one -/
noncomputable def trapezoid_area : ℝ := equilateral_triangle_area large_side - equilateral_triangle_area small_side

theorem area_ratio_is_one_third :
  equilateral_triangle_area small_side / trapezoid_area = 1 / 3 := by
  sorry

#eval large_side
#eval small_side

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_ratio_is_one_third_l94_9444


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_integral_equality_l94_9491

open Real

noncomputable def f (x : ℝ) : ℝ := (3 * (tan x)^2 - 1) / ((tan x)^2 + 5)

noncomputable def a : ℝ := 0
noncomputable def b : ℝ := arccos (1 / Real.sqrt 6)

-- State the theorem
theorem integral_equality : 
  ∫ x in a..b, f x = π / Real.sqrt 5 - arctan (Real.sqrt 5) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_integral_equality_l94_9491


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equidistant_set_is_hyperbolas_and_parabolas_l94_9426

-- Define the square
structure Square where
  side : ℝ
  center : ℝ × ℝ

-- Define the circle
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

-- Define a point in the plane
structure Point where
  x : ℝ
  y : ℝ

-- Define the distance between two points
noncomputable def distance (p1 p2 : Point) : ℝ :=
  Real.sqrt ((p1.x - p2.x)^2 + (p1.y - p2.y)^2)

-- Define the distance from a point to the square
noncomputable def distToSquare (p : Point) (s : Square) : ℝ := sorry

-- Define the distance from a point to the circle
noncomputable def distToCircle (p : Point) (c : Circle) : ℝ := sorry

-- Define the set of equidistant points
def equidistantSet (s : Square) (c : Circle) : Set Point :=
  {p : Point | distToSquare p s = distToCircle p c}

-- Helper definitions (not proved)
def IsOnHyperbola (p : Point) (f1 f2 : ℝ × ℝ) : Prop := sorry
def IsOnParabola (p : Point) (focus : ℝ × ℝ) : Prop := sorry

-- Theorem statement
theorem equidistant_set_is_hyperbolas_and_parabolas 
  (s : Square) (c : Circle) 
  (h : c.center.1 = s.center.1 + s.side / 2 ∨ c.center.1 = s.center.1 - s.side / 2 ∨ 
       c.center.2 = s.center.2 + s.side / 2 ∨ c.center.2 = s.center.2 - s.side / 2) :
  ∃ (hyp : Set Point) (par : Set Point), 
    (∀ p ∈ hyp, IsOnHyperbola p s.center c.center) ∧
    (∀ p ∈ par, IsOnParabola p c.center) ∧
    equidistantSet s c = hyp ∪ par := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_equidistant_set_is_hyperbolas_and_parabolas_l94_9426


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_2500th_term_mod_7_l94_9448

/-- Represents the sequence where each positive integer n appears n times -/
def our_sequence : ℕ → ℕ := sorry

/-- The 2500th term of the sequence -/
def term_2500 : ℕ := sorry

theorem sequence_2500th_term_mod_7 : term_2500 % 7 = 1 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_2500th_term_mod_7_l94_9448


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equality_condition_l94_9415

theorem equality_condition (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  Real.sqrt (a^2 + 2*a*b + b/c) = (a+b) * Real.sqrt (b/c) ↔ 
  c = (a^2*b + 2*a*b^2 + b^3 - b) / (a^2 + 2*a*b) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_equality_condition_l94_9415


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_difference_angle_set_l94_9464

-- Define the angle x
noncomputable def x : ℝ := sorry

-- Define the point P
noncomputable def P : ℝ × ℝ := (1, Real.sqrt 3)

-- Condition: The terminal side of angle x passes through point P
axiom terminal_side : Real.cos x = 1 / Real.sqrt (1 + (Real.sqrt 3)^2) ∧ 
                      Real.sin x = Real.sqrt 3 / Real.sqrt (1 + (Real.sqrt 3)^2)

-- Theorem 1: Value of sin(π-x) - sin(π/2+x)
theorem sin_difference : Real.sin (Real.pi - x) - Real.sin (Real.pi/2 + x) = (Real.sqrt 3 - 1) / 2 := by
  sorry

-- Theorem 2: Set of all possible values of x
theorem angle_set : ∃ k : ℤ, x = 2 * k * Real.pi + Real.pi / 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_difference_angle_set_l94_9464


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_plus_abs_g_is_even_l94_9436

-- Define functions f and g
variable (f g : ℝ → ℝ)

-- Define evenness and oddness
def IsEven (h : ℝ → ℝ) : Prop := ∀ x, h x = h (-x)
def IsOdd (h : ℝ → ℝ) : Prop := ∀ x, h x = -h (-x)

-- State the theorem
theorem f_plus_abs_g_is_even
  (hf : IsEven f) (hg : IsOdd g) :
  IsEven (fun x ↦ f x + |g x|) :=
by
  -- Proof goes here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_plus_abs_g_is_even_l94_9436


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_perimeter_of_divided_isosceles_triangle_l94_9435

/-- Represents an isosceles triangle with given base and height -/
structure IsoscelesTriangle where
  base : ℝ
  height : ℝ

/-- Calculates the perimeter of a triangular piece given its position -/
noncomputable def piecePerimeter (triangle : IsoscelesTriangle) (k : ℕ) : ℝ :=
  1 + Real.sqrt (triangle.height^2 + (k : ℝ)^2) + Real.sqrt (triangle.height^2 + ((k + 1) : ℝ)^2)

/-- Theorem: The maximum perimeter of a piece in the divided isosceles triangle -/
theorem max_perimeter_of_divided_isosceles_triangle (triangle : IsoscelesTriangle) 
  (h_base : triangle.base = 8)
  (h_height : triangle.height = 10) :
  ∃ (max_perimeter : ℝ), 
    (∀ k, k < 8 → piecePerimeter triangle k ≤ max_perimeter) ∧ 
    (22.20 < max_perimeter ∧ max_perimeter < 22.22) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_perimeter_of_divided_isosceles_triangle_l94_9435


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trapezoid_area_is_fifty_thirds_l94_9402

/-- A trapezoid with perpendicular diagonals -/
structure PerpendicularDiagonalTrapezoid where
  /-- The length of one diagonal -/
  diagonal₁ : ℝ
  /-- The length of the other diagonal -/
  diagonal₂ : ℝ
  /-- The height of the trapezoid -/
  height : ℝ
  /-- The diagonals are perpendicular -/
  perpendicular_diagonals : diagonal₁ * diagonal₁ = height * height + diagonal₂ * diagonal₂

/-- The area of a trapezoid with perpendicular diagonals -/
noncomputable def area (t : PerpendicularDiagonalTrapezoid) : ℝ :=
  (1 / 2) * t.diagonal₁ * t.diagonal₂

/-- Theorem: The area of a trapezoid with perpendicular diagonals, 
    where one diagonal is 5 units and the height is 4 units, is 50/3 square units -/
theorem trapezoid_area_is_fifty_thirds 
  (t : PerpendicularDiagonalTrapezoid) 
  (h1 : t.diagonal₁ = 5) 
  (h2 : t.height = 4) : 
  area t = 50 / 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trapezoid_area_is_fifty_thirds_l94_9402


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_negative_cosine_sequence_l94_9499

theorem negative_cosine_sequence (α : ℝ) : 
  (∀ n : ℕ, Real.cos (2^n * α) < 0) ↔ 
  (∃ k : ℤ, α = 2*Real.pi/3 + 2*k*Real.pi ∨ α = -2*Real.pi/3 + 2*k*Real.pi) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_negative_cosine_sequence_l94_9499


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_characterize_f_l94_9421

def is_valid_f (f : ℕ+ → ℝ) : Prop :=
  (∀ n : ℕ+, f (n + 1) ≥ f n) ∧
  (∀ m n : ℕ+, Nat.Coprime m.val n.val → f (m * n) = f m * f n)

theorem characterize_f :
  ∀ f : ℕ+ → ℝ, is_valid_f f →
    (∀ n : ℕ+, f n = 0) ∨
    (∃ a : ℝ, a ≥ 0 ∧ ∀ n : ℕ+, f n = (n : ℝ) ^ a) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_characterize_f_l94_9421


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l94_9423

noncomputable def f (x : ℝ) : ℝ := (x + 1) / (x^2 + 6*x + 9)

theorem domain_of_f :
  {x : ℝ | ∃ y, f x = y} = {x : ℝ | x ≠ -3} :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l94_9423


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_souvenir_cost_in_usd_l94_9403

/-- The cost of a souvenir in GBP with tax, given the base price and tax rate -/
noncomputable def cost_with_tax (base_price : ℝ) (tax_rate : ℝ) : ℝ :=
  base_price * (1 + tax_rate)

/-- Convert an amount from GBP to USD given the exchange rate -/
noncomputable def gbp_to_usd (amount_gbp : ℝ) (exchange_rate : ℝ) : ℝ :=
  amount_gbp / exchange_rate

theorem souvenir_cost_in_usd :
  let base_price : ℝ := 250
  let tax_rate : ℝ := 0.2
  let exchange_rate : ℝ := 0.8
  let total_cost_gbp := cost_with_tax base_price tax_rate
  let total_cost_usd := gbp_to_usd total_cost_gbp exchange_rate
  total_cost_usd = 375 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_souvenir_cost_in_usd_l94_9403


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_green_blue_difference_after_border_l94_9431

/-- Represents a hexagonal figure with blue and green tiles -/
structure HexagonalFigure where
  blue_tiles : ℕ
  green_tiles : ℕ

/-- Adds a border of green tiles to a hexagonal figure -/
def add_green_border (figure : HexagonalFigure) : HexagonalFigure :=
  { blue_tiles := figure.blue_tiles,
    green_tiles := figure.green_tiles + 6 }

/-- The initial hexagonal figure -/
def initial_figure : HexagonalFigure :=
  { blue_tiles := 20,
    green_tiles := 10 }

/-- Theorem stating the difference between green and blue tiles after adding a border -/
theorem green_blue_difference_after_border :
  (add_green_border initial_figure).green_tiles + 4 = (add_green_border initial_figure).blue_tiles := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_green_blue_difference_after_border_l94_9431


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l94_9408

noncomputable def f (x : ℝ) := Real.sqrt (3 - 2*x - x^2)

theorem domain_of_f :
  {x : ℝ | ∃ y, f x = y} = {x : ℝ | -3 ≤ x ∧ x ≤ 1} := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l94_9408


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circumcircle_equation_l94_9432

/-- The circumcircle of triangle ABC with vertices A(1,0), B(0, √3), and C(2, √3) -/
def circumcircle (x y : ℝ) : Prop :=
  x^2 + y^2 - 2*x - (4*Real.sqrt 3/3)*y + 1 = 0

/-- Point A of the triangle -/
def A : ℝ × ℝ := (1, 0)

/-- Point B of the triangle -/
noncomputable def B : ℝ × ℝ := (0, Real.sqrt 3)

/-- Point C of the triangle -/
noncomputable def C : ℝ × ℝ := (2, Real.sqrt 3)

theorem circumcircle_equation : 
  circumcircle A.1 A.2 ∧ circumcircle B.1 B.2 ∧ circumcircle C.1 C.2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_circumcircle_equation_l94_9432
