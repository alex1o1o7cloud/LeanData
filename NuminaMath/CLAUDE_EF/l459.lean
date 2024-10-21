import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_box_midpoint_triangle_area_l459_45964

/-- Represents a rectangular box with given dimensions -/
structure RectangularBox where
  width : ℝ
  length : ℝ
  height : ℝ

/-- Calculates the area of the triangle formed by midpoints of three intersecting faces -/
noncomputable def midpointTriangleArea (box : RectangularBox) : ℝ :=
  let a := ((box.width / 2) ^ 2 + (box.length / 2) ^ 2).sqrt
  let b := ((box.width / 2) ^ 2 + (box.height / 2) ^ 2).sqrt
  let c := ((box.length / 2) ^ 2 + (box.height / 2) ^ 2).sqrt
  let s := (a + b + c) / 2
  (s * (s - a) * (s - b) * (s - c)).sqrt

/-- Main theorem -/
theorem box_midpoint_triangle_area (m n : ℕ) (h_coprime : Nat.Coprime m n) :
  let box := RectangularBox.mk 15 20 (m / n : ℝ)
  midpointTriangleArea box = 50 → m + n = 16 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_box_midpoint_triangle_area_l459_45964


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_median_squares_sum_l459_45988

/-- Given a triangle with side lengths 13, 13, and 10, the sum of the squares of the lengths of its medians is 363. -/
theorem median_squares_sum (a b c : ℝ) (h1 : a = 13) (h2 : b = 13) (h3 : c = 10) :
  let m1 := Real.sqrt ((2 * (a^2 + c^2) + b^2) / 4 - b^2 / 2)
  let m2 := Real.sqrt ((2 * (b^2 + c^2) + a^2) / 4 - a^2 / 2)
  let m3 := Real.sqrt ((2 * (a^2 + b^2) + c^2) / 4 - c^2 / 2)
  m1^2 + m2^2 + m3^2 = 363 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_median_squares_sum_l459_45988


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_impossibility_theorem_l459_45965

/-- Represents a pile of stones -/
structure Pile where
  stones : ℕ

/-- Represents the state of all piles -/
def PileState := List Pile

/-- The initial state of piles -/
def initial_state : PileState := [⟨51⟩, ⟨49⟩, ⟨5⟩]

/-- Combines two piles into one -/
def combine_piles (p1 p2 : Pile) : Pile :=
  ⟨p1.stones + p2.stones⟩

/-- Divides an even-numbered pile into two equal piles -/
def divide_pile (p : Pile) : Option (Pile × Pile) :=
  if p.stones % 2 = 0 then
    some (⟨p.stones / 2⟩, ⟨p.stones / 2⟩)
  else
    none

/-- Checks if a state consists of 105 piles with one stone each -/
def is_final_state (state : PileState) : Prop :=
  state.length = 105 ∧ state.all (λ p => p.stones = 1)

/-- Represents a single step transformation -/
inductive Step : PileState → PileState → Prop where
  | combine : (p1 p2 : Pile) → (rest : PileState) → 
      Step (p1 :: p2 :: rest) ((combine_piles p1 p2) :: rest)
  | divide : (p : Pile) → (p1 p2 : Pile) → (rest : PileState) → 
      divide_pile p = some (p1, p2) → Step (p :: rest) (p1 :: p2 :: rest)

/-- Represents multiple step transformations -/
def Reachable := Relation.ReflTransGen Step

/-- The main theorem stating the impossibility of reaching the desired state -/
theorem impossibility_theorem :
  ¬ ∃ (final_state : PileState),
    (Reachable initial_state final_state) ∧
    is_final_state final_state :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_impossibility_theorem_l459_45965


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_treehouse_materials_calculation_l459_45906

/-- Represents the materials needed for the treehouse --/
structure TreehouseMaterials where
  wooden_planks : ℕ
  iron_nails : ℕ
  fabric : ℝ
  metal_brackets : ℕ

/-- Represents the materials Colbert already has --/
structure ColbertMaterials where
  wooden_planks : ℕ
  iron_nails : ℕ
  fabric : ℝ
  metal_brackets : ℕ

/-- Represents the store's packaging and discount rules --/
structure StoreRules where
  planks_per_pack : ℕ
  nails_per_pack : ℕ
  bracket_discount_threshold : ℕ
  bracket_discount_rate : ℝ

/-- Calculates the materials Colbert needs to buy --/
def calculate_purchase (
  needed : TreehouseMaterials
) (owned : ColbertMaterials
) (rules : StoreRules
) : TreehouseMaterials :=
  sorry

theorem treehouse_materials_calculation (
  needed : TreehouseMaterials
) (owned : ColbertMaterials
) (rules : StoreRules
) : let purchase := calculate_purchase needed owned rules
    purchase.wooden_planks = 24 ∧
    purchase.iron_nails = 14 ∧
    purchase.fabric = 6.67 ∧
    purchase.metal_brackets = 30 :=
by
  sorry

#check treehouse_materials_calculation

end NUMINAMATH_CALUDE_ERRORFEEDBACK_treehouse_materials_calculation_l459_45906


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sixth_term_values_l459_45952

theorem sixth_term_values (a b : ℝ) : 
  (Real.sqrt (6 + a/b) = 6 * Real.sqrt (a/b)) → (a = 6 ∧ b = 35) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sixth_term_values_l459_45952


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_a_value_l459_45909

noncomputable def parabola_vertex (a b c : ℝ) : ℝ × ℝ :=
  (-b / (2 * a), c - b^2 / (4 * a))

noncomputable def parabola_equation (a b c : ℝ) (x : ℝ) : ℝ :=
  a * x^2 + b * x + c

theorem smallest_a_value (a b c : ℝ) : 
  parabola_vertex a b c = (-1/3, -1/9) →
  a > 0 →
  ∃ n : ℤ, 2 * a + b + 3 * c = n →
  (∀ a' : ℝ, a' > 0 → 
    (∃ b' c' : ℝ, parabola_vertex a' b' c' = (-1/3, -1/9) ∧ 
      ∃ n' : ℤ, 2 * a' + b' + 3 * c' = n') →
    a ≤ a') →
  a = 1/3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_a_value_l459_45909


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l459_45949

noncomputable def f (x : ℝ) := 2 * Real.sqrt 3 * Real.sin x ^ 2 + Real.sin (2 * x)

noncomputable def g (x : ℝ) := f ((x + Real.pi / 3) / 2)

theorem f_properties :
  (∃ T > 0, ∀ x, f (x + T) = f x ∧ ∀ S, 0 < S ∧ S < T → ∃ y, f (y + S) ≠ f y) ∧
  (∀ k : ℤ, StrictMonoOn f (Set.Icc (k * Real.pi - Real.pi / 12) (k * Real.pi + 5 * Real.pi / 12))) ∧
  g (Real.pi / 6) = Real.sqrt 3 + 1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l459_45949


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fruit_distribution_impossibility_l459_45982

/-- Represents the state of fruits in bowls on an infinite integer number line -/
def BowlState := ℤ → ℕ

/-- The initial state where each bowl contains exactly one fruit -/
def initialState : BowlState := fun _ => 1

/-- Represents a single move of transferring a fruit between adjacent bowls -/
def validMove (s1 s2 : BowlState) : Prop :=
  ∃ i : ℤ, (s2 i = s1 i - 1 ∧ (s2 (i + 1) = s1 (i + 1) + 1 ∨ s2 (i - 1) = s1 (i - 1) + 1)) ∧
    ∀ j : ℤ, j ≠ i ∧ j ≠ i + 1 ∧ j ≠ i - 1 → s2 j = s1 j

/-- Represents a sequence of n valid moves -/
def validMoveSequence (n : ℕ) (s1 s2 : BowlState) : Prop :=
  ∃ f : ℕ → BowlState, f 0 = s1 ∧ f n = s2 ∧
    ∀ i : ℕ, i < n → validMove (f i) (f (i + 1))

/-- The theorem stating that after 999 moves, it's impossible for every bowl to have exactly one fruit -/
theorem fruit_distribution_impossibility :
  ¬∃ (finalState : BowlState), validMoveSequence 999 initialState finalState ∧
    (∀ i : ℤ, finalState i = 1) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fruit_distribution_impossibility_l459_45982


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cattle_profit_calculation_l459_45910

/-- Calculates the profit from selling cattle given the specified conditions -/
theorem cattle_profit_calculation (num_cattle : ℕ) (purchase_cost feeding_cost_percentage weight selling_price : ℚ) : 
  num_cattle = 100 →
  purchase_cost = 40000 →
  feeding_cost_percentage = 20 / 100 →
  weight = 1000 →
  selling_price = 2 →
  (let feeding_cost := purchase_cost * feeding_cost_percentage
   let total_cost := purchase_cost + feeding_cost
   let revenue := num_cattle * weight * selling_price
   let profit := revenue - total_cost
   profit) = 112000 := by
  sorry

#check cattle_profit_calculation

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cattle_profit_calculation_l459_45910


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_congruent_triangle_with_same_color_points_l459_45914

/-- A color type with 1993 different colors -/
inductive Color
| c1 | c2 | c3 | c4 | c5 | c6 | c7 | c8 | c9 | c10
-- ... (add more colors as needed)
| c1993

/-- A point in the plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- A triangle in the plane -/
structure Triangle where
  a : Point
  b : Point
  c : Point

/-- The coloring of the plane -/
noncomputable def planeColoring : Point → Color := sorry

/-- Congruence relation between triangles -/
def congruent : Triangle → Triangle → Prop := sorry

/-- A point is on a side of a triangle (excluding vertices) -/
def onSide (p : Point) (t : Triangle) : Prop := sorry

/-- Main theorem -/
theorem exists_congruent_triangle_with_same_color_points
  (colorPresent : ∀ c : Color, ∃ p : Point, planeColoring p = c)
  (t : Triangle) :
  ∃ t' : Triangle, congruent t t' ∧
    ∃ c : Color, ∃ p1 p2 p3 : Point,
      onSide p1 t' ∧ onSide p2 t' ∧ onSide p3 t' ∧
      planeColoring p1 = c ∧ planeColoring p2 = c ∧ planeColoring p3 = c := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_congruent_triangle_with_same_color_points_l459_45914


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_walked_approx_6_6_l459_45935

/-- Represents the journey with three equal parts -/
structure Journey where
  total_time : ℚ
  flat_speed : ℚ
  uphill_speed : ℚ
  walking_speed : ℚ

/-- Calculates the distance walked given a journey -/
def distance_walked (j : Journey) : ℚ :=
  (j.total_time * j.flat_speed * j.uphill_speed * j.walking_speed) /
  (j.flat_speed * j.uphill_speed + j.flat_speed * j.walking_speed + j.uphill_speed * j.walking_speed)

/-- The main theorem stating that the distance walked is approximately 6.6 km -/
theorem distance_walked_approx_6_6 (j : Journey) 
  (h1 : j.total_time = 3)
  (h2 : j.flat_speed = 18)
  (h3 : j.uphill_speed = 6)
  (h4 : j.walking_speed = 4) :
  ∃ (ε : ℚ), ε > 0 ∧ ε < (1/10) ∧ |distance_walked j - (66/10)| < ε := by
  sorry

#eval distance_walked { total_time := 3, flat_speed := 18, uphill_speed := 6, walking_speed := 4 }

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_walked_approx_6_6_l459_45935


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_negative_integers_inequality_l459_45926

theorem max_negative_integers_inequality
  (a b c d e f g h : ℤ)
  (ha : a ≠ 0)
  (hc : c ≠ 0)
  (he : e ≠ 0)
  (h_ineq : (a * b^2 + c * d * e^3) * (f * g^2 * h + f^3 - g^2) < 0)
  (h_abs : abs d < abs f ∧ abs f < abs h) :
  ∃ (s : ℕ),
    s = 5 ∧
    s = Finset.card (Finset.filter (fun x => x < 0) {a, b, c, d, e, f, g, h}) ∧
    ∀ (neg_set : Finset ℤ),
      neg_set ⊆ {a, b, c, d, e, f, g, h} →
      (∀ x ∈ neg_set, x < 0) →
      (a * b^2 + c * d * e^3) * (f * g^2 * h + f^3 - g^2) < 0 →
      abs d < abs f ∧ abs f < abs h →
      Finset.card neg_set ≤ s :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_negative_integers_inequality_l459_45926


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pizza_slices_proof_l459_45943

def slices_per_pizza (kaplan_slices : ℕ) (bobby_pizzas : ℕ) (kaplan_fraction : ℚ) : ℚ :=
  (kaplan_slices : ℚ) / kaplan_fraction / (bobby_pizzas : ℚ)

theorem pizza_slices_proof (kaplan_slices : ℕ) (bobby_pizzas : ℕ) (kaplan_fraction : ℚ) 
  (h1 : kaplan_slices = 3)
  (h2 : bobby_pizzas = 2)
  (h3 : kaplan_fraction = 1/4) :
  slices_per_pizza kaplan_slices bobby_pizzas kaplan_fraction = 6 := by
  sorry

#eval slices_per_pizza 3 2 (1/4)

end NUMINAMATH_CALUDE_ERRORFEEDBACK_pizza_slices_proof_l459_45943


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_mySequence_limit_is_one_l459_45936

noncomputable def mySequence (n : ℕ) : ℚ :=
  (Nat.factorial n + Nat.factorial (n + 2)) / (Nat.factorial (n - 1) + Nat.factorial (n + 2))

theorem mySequence_limit_is_one :
  ∀ ε > 0, ∃ N, ∀ n ≥ N, |mySequence n - 1| < ε := by
  sorry

#check mySequence_limit_is_one

end NUMINAMATH_CALUDE_ERRORFEEDBACK_mySequence_limit_is_one_l459_45936


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cosine_inequality_l459_45918

theorem cosine_inequality (x : ℝ) (h : x ∈ Set.Ioo 0 (π / 2)) :
  Real.cos x < 1 - x^2 / 2 + x^4 / 16 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cosine_inequality_l459_45918


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_divisors_of_360_l459_45900

def num_divisors (n : ℕ) : ℕ :=
  (Finset.filter (· ∣ n) (Finset.range (n + 1))).card

theorem divisors_of_360 : num_divisors 360 = 24 := by
  -- The proof will be implemented here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_divisors_of_360_l459_45900


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_max_min_f_l459_45961

noncomputable def f (ω φ : ℝ) (x : ℝ) : ℝ := Real.sin (ω * x + φ)

theorem sum_max_min_f (ω φ : ℝ) :
  ω > 0 →
  |φ| < π / 2 →
  (∀ x, f ω φ (x + π / ω) = f ω φ x) →
  f ω φ (7 * π / 12) = 0 →
  (∃ x₁ x₂, x₁ ∈ Set.Icc 0 (π / 2) ∧ x₂ ∈ Set.Icc 0 (π / 2) ∧
    (∀ x ∈ Set.Icc 0 (π / 2), f ω φ x ≤ f ω φ x₁ ∧ f ω φ x₂ ≤ f ω φ x) ∧
    f ω φ x₁ + f ω φ x₂ = 1 / 2) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_max_min_f_l459_45961


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_greater_than_x_iff_in_open_interval_l459_45953

-- Define the function f
noncomputable def f (x : ℝ) : ℝ :=
  if x ≥ 0 then 1/2 * x - 1 else 1/x

-- State the theorem
theorem f_greater_than_x_iff_in_open_interval (a : ℝ) :
  f a > a ↔ a ∈ Set.Ioo (-1 : ℝ) 0 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_greater_than_x_iff_in_open_interval_l459_45953


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_initial_water_percentage_l459_45963

/-- 
Given a container with a capacity of 20 liters, prove that the initial percentage 
of water in the container is 30%, if adding 9 liters makes it 3/4 full.
-/
theorem initial_water_percentage 
  (container_capacity : ℝ) 
  (added_water : ℝ) 
  (final_fraction : ℝ) 
  (initial_percentage : ℝ)
  (h1 : container_capacity = 20)
  (h2 : added_water = 9)
  (h3 : final_fraction = 3/4)
  (h4 : (initial_percentage / 100) * container_capacity + added_water = 
        final_fraction * container_capacity)
  : initial_percentage = 30 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_initial_water_percentage_l459_45963


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_new_speed_ratio_l459_45987

-- Define the speed of the boat and the stream
variable (b s : ℝ)

-- Define the condition that rowing against the stream takes twice as long as with the stream
def rowing_time_condition : Prop := (1 / (b - s)) = 2 * (1 / (b + s))

-- Define the relationship between boat speed and stream speed
def speed_relationship : Prop := b = 3 * s

-- Define the new stream speed after 20% increase
def new_stream_speed (s : ℝ) : ℝ := 1.2 * s

-- Theorem stating the new ratio of boat speed to stream speed
theorem new_speed_ratio (h1 : rowing_time_condition b s) (h2 : speed_relationship b s) :
  b / new_stream_speed s = 5 / 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_new_speed_ratio_l459_45987


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l459_45951

noncomputable def f (x : ℝ) := (2 : ℝ)^(2*x) - (2 : ℝ)^(x+1) + 3

theorem f_properties :
  ∀ m : ℝ, m ≤ 0 →
  (∀ x ∈ Set.Icc (-1 : ℝ) 2, f x ≤ 11) ∧
  (∀ x ∈ Set.Icc m 0, f x ≥ 2) ∧
  (∀ x ∈ Set.Icc m 0, f x ≤ f m) ∧
  f m = (2 : ℝ)^(2*m) - (2 : ℝ)^(m+1) + 3 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l459_45951


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l459_45969

-- Define the function f(x)
noncomputable def f (x : ℝ) : ℝ := (2*x - 1) / (x + 1)

-- Define the domain
def D : Set ℝ := { x | 3 ≤ x ∧ x ≤ 5 }

-- Theorem statement
theorem f_properties :
  (∀ x₁ x₂, x₁ ∈ D → x₂ ∈ D → x₁ < x₂ → f x₁ < f x₂) ∧
  (∀ x ∈ D, f x ≥ 5/4) ∧
  (∀ x ∈ D, f x ≤ 3/2) ∧
  (∃ x ∈ D, f x = 5/4) ∧
  (∃ x ∈ D, f x = 3/2) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l459_45969


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solutions_count_l459_45976

theorem solutions_count :
  let solution_pairs := {(x, y) : Nat × Nat | 3 * x + 4 * y = 1000 ∧ x > 0 ∧ y > 0}
  Finset.card (Finset.filter (fun (x, y) => 3 * x + 4 * y = 1000 ∧ x > 0 ∧ y > 0) (Finset.range 1001 ×ˢ Finset.range 1001)) = 84 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_solutions_count_l459_45976


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_proof_l459_45931

/-- The function f(x) = x^2 - x - ln x -/
noncomputable def f (x : ℝ) : ℝ := x^2 - x - Real.log x

/-- Theorem statement -/
theorem inequality_proof (a x₁ x₂ : ℝ) (ha : a > 0) 
    (hx₁ : a * x₁ + f x₁ = x₁^2 - x₁) (hx₂ : a * x₂ + f x₂ = x₂^2 - x₂) 
    (hdistinct : x₁ ≠ x₂) : 
  Real.log x₁ + Real.log x₂ + 2 * Real.log a < 0 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_proof_l459_45931


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_intersection_theorem_l459_45904

noncomputable section

/-- The parabola y^2 = 4x -/
def parabola (x y : ℝ) : Prop := y^2 = 4*x

/-- The focus of the parabola y^2 = 4x -/
def focus : ℝ × ℝ := (1, 0)

/-- A line with slope k -/
def line (k m x y : ℝ) : Prop := y = k*x + m

/-- The condition that a point (x, y) is on both the line and the parabola -/
def intersection (k m x y : ℝ) : Prop := line k m x y ∧ parabola x y

/-- The distance between two points -/
noncomputable def distance (x1 y1 x2 y2 : ℝ) : ℝ := 
  Real.sqrt ((x2 - x1)^2 + (y2 - y1)^2)

theorem parabola_intersection_theorem (k : ℝ) :
  ∃ (m x1 y1 x2 y2 : ℝ),
    k ≠ 0 ∧
    intersection k m x1 y1 ∧
    intersection k m x2 y2 ∧
    (x1, y1) ≠ (x2, y2) ∧
    distance x1 y1 (focus.1) (focus.2) = 2 * distance x2 y2 (focus.1) (focus.2) →
    |k| = 2 * Real.sqrt 2 := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_intersection_theorem_l459_45904


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_garden_reconfiguration_l459_45967

/-- Represents a rectangular garden with given length and width -/
structure RectangularGarden where
  length : ℝ
  width : ℝ

/-- Calculates the area of a rectangular garden -/
noncomputable def RectangularGarden.area (g : RectangularGarden) : ℝ :=
  g.length * g.width

/-- Calculates the perimeter of a rectangular garden -/
noncomputable def RectangularGarden.perimeter (g : RectangularGarden) : ℝ :=
  2 * (g.length + g.width)

/-- Represents the original garden configuration -/
def original_garden : RectangularGarden :=
  { length := 80, width := 20 }

/-- Calculates the side length of a square garden with the same perimeter as the original -/
noncomputable def square_side (g : RectangularGarden) : ℝ :=
  g.perimeter / 4

/-- Calculates the width of a rectangular garden with length twice its width and same perimeter as the original -/
noncomputable def new_rect_width (g : RectangularGarden) : ℝ :=
  g.perimeter / 6

/-- Theorem stating the area increase and comparison for different configurations -/
theorem garden_reconfiguration (g : RectangularGarden) 
  (h : g = original_garden) : 
  let square_increase := (square_side g)^2 - g.area
  let rect_increase := 2 * (new_rect_width g)^2 - g.area
  square_increase = 900 ∧ 
  abs (rect_increase - 622.22) < 0.01 ∧ 
  square_increase > rect_increase := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_garden_reconfiguration_l459_45967


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_line_intersection_circle_l459_45959

/-- The ellipse equation -/
def ellipse (x y : ℝ) : Prop := x^2 / 3 + y^2 = 1

/-- The line equation -/
def line (k : ℝ) (x y : ℝ) : Prop := y = k * x + 2

/-- Check if a point is on the circle with diameter CD -/
def on_circle (x y xc yc xd yd : ℝ) : Prop :=
  (x - xc) * (xd - x) + (y - yc) * (yd - y) = 0

theorem ellipse_line_intersection_circle :
  ∃ (k : ℝ) (xc yc xd yd : ℝ),
    k = 7/6 ∧
    ellipse xc yc ∧
    ellipse xd yd ∧
    line k xc yc ∧
    line k xd yd ∧
    on_circle (-1) 0 xc yc xd yd :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_line_intersection_circle_l459_45959


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_inequality_l459_45999

-- Define the function f on the open interval (0, π/2)
def f : ℝ → ℝ := sorry

-- Define the derivative of f
def f' : ℝ → ℝ := sorry

-- Theorem statement
theorem function_inequality
  (h1 : ∀ x, 0 < x ∧ x < π/2 → Real.tan x * f x > f' x) :
  Real.sqrt 2 * f (π/4) < Real.sqrt 3 * f (π/6) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_inequality_l459_45999


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_increasing_on_interval_l459_45946

-- Define the function f(x) = 2^(x-2)
noncomputable def f (x : ℝ) : ℝ := 2^(x-2)

-- State the theorem
theorem f_increasing_on_interval : 
  ∀ x y : ℝ, 1 < x → x < y → f x < f y := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_increasing_on_interval_l459_45946


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_to_square_impossible_l459_45983

-- Define a type for shapes
inductive Shape
  | Circle : Shape
  | Square : Shape
  | Piece : Shape

-- Define a function to represent the angular measure of a shape's boundary
noncomputable def angularMeasure : Shape → ℝ
  | Shape.Circle => 2 * Real.pi
  | Shape.Square => 0
  | Shape.Piece => sorry

-- Define a function to represent cutting a shape into pieces
def cut : Shape → List Shape
  | Shape.Circle => sorry
  | Shape.Square => sorry
  | Shape.Piece => sorry

-- Define a function to represent reassembling pieces into a new shape
def reassemble : List Shape → Shape
  | _ => sorry

-- Theorem statement
theorem circle_to_square_impossible :
  ∀ (pieces : List Shape),
    (∃ (c : Shape), c = Shape.Circle ∧ cut c = pieces) →
    (reassemble pieces ≠ Shape.Square ∨ 
     angularMeasure (reassemble pieces) ≠ angularMeasure Shape.Circle) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_to_square_impossible_l459_45983


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_sum_is_1047_l459_45945

/-- A permutation of the digits 4, 5, 6, 7, 8, 9 -/
def ValidPermutation : Type := { p : Fin 6 → Fin 6 // Function.Injective p }

/-- Convert a permutation to two 3-digit numbers -/
def permToNumbers (p : ValidPermutation) : ℕ × ℕ :=
  (100 * (4 + p.val 0) + 10 * (5 + p.val 1) + (6 + p.val 2),
   100 * (7 + p.val 3) + 10 * (8 + p.val 4) + (9 + p.val 5))

/-- Sum of two numbers generated from a permutation -/
def sumFromPerm (p : ValidPermutation) : ℕ :=
  let (n1, n2) := permToNumbers p
  n1 + n2

/-- Theorem: The smallest sum of two 3-digit numbers formed by
    the digits 4, 5, 6, 7, 8, and 9, each used exactly once, is 1047 -/
theorem smallest_sum_is_1047 :
  (∀ p : ValidPermutation, 1047 ≤ sumFromPerm p) ∧
  (∃ p : ValidPermutation, sumFromPerm p = 1047) := by
  sorry

#check smallest_sum_is_1047

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_sum_is_1047_l459_45945


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_bridge_arch_ratio_change_l459_45954

/-- Represents a circular bridge arch --/
structure BridgeArch where
  radius : ℝ
  height : ℝ

/-- The original bridge design --/
noncomputable def original : BridgeArch :=
  { radius := 1, height := 3/4 }

/-- The built bridge --/
noncomputable def built : BridgeArch :=
  { radius := 2 * original.radius,
    height := (1/3) * original.height }

theorem bridge_arch_ratio_change :
  (original.height / original.radius = 3/4) ∧
  (built.height / built.radius = 1/8) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_bridge_arch_ratio_change_l459_45954


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetric_sine_value_l459_45915

noncomputable def f (ω φ : ℝ) (x : ℝ) : ℝ := Real.sin (ω * x + φ)

theorem symmetric_sine_value 
  (ω φ : ℝ) 
  (h1 : StrictMonoOn (f ω φ) (Set.Ioo (π/6) (2*π/3)))
  (h2 : ∀ (y : ℝ), f ω φ (π/6 - y) = f ω φ (π/6 + y))
  (h3 : ∀ (y : ℝ), f ω φ (2*π/3 - y) = f ω φ (2*π/3 + y)) :
  f ω φ (-5*π/12) = Real.sqrt 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetric_sine_value_l459_45915


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_solution_implies_unique_composition_solution_l459_45966

/-- A quadratic polynomial P(x) = x² + 1/4 -/
noncomputable def P (x : ℝ) : ℝ := x^2 + 1/4

theorem unique_solution_implies_unique_composition_solution :
  (∃! x, P x = x) → (∃! x, P (P x) = x) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_solution_implies_unique_composition_solution_l459_45966


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_period_and_range_l459_45941

noncomputable def a (x : ℝ) : ℝ × ℝ := (2 * Real.cos x, 1)
noncomputable def b (x : ℝ) : ℝ × ℝ := (Real.cos x, Real.sqrt 3 * Real.sin (2 * x))

noncomputable def f (x : ℝ) : ℝ := (a x).1 * (b x).1 + (a x).2 * (b x).2

def is_periodic (f : ℝ → ℝ) (T : ℝ) : Prop :=
  ∀ x, f (x + T) = f x

theorem f_period_and_range :
  (∃ T > 0, is_periodic f T ∧ ∀ T' > 0, is_periodic f T' → T ≤ T') ∧
  (∀ y ∈ Set.Icc 1 3, ∃ x ∈ Set.Icc (-π/6) (π/3), f x = y) ∧
  (∀ x ∈ Set.Icc (-π/6) (π/3), f x ∈ Set.Icc 1 3) := by
  sorry

#check f_period_and_range

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_period_and_range_l459_45941


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_decreasing_f_implies_a_range_l459_45948

-- Define the piecewise function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x < 1 then (3*a - 1)*x + 4*a else a^x

-- State the theorem
theorem decreasing_f_implies_a_range (a : ℝ) :
  (∀ x y : ℝ, x < y → f a x > f a y) →
  a ∈ Set.Icc (1/6 : ℝ) (1/3 : ℝ) ∧ a ≠ 1/3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_decreasing_f_implies_a_range_l459_45948


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_different_graphs_l459_45923

-- Define the three functions
noncomputable def f (x : ℝ) : ℝ := 2 * x - 3
noncomputable def g (x : ℝ) : ℝ := (4 * x^2 - 9) / (2 * x + 3)
noncomputable def h (x : ℝ) : ℝ := (4 * x^2 - 3 * Real.sin x) / (2 * x + 3)

-- Theorem statement
theorem different_graphs : 
  (∃ x : ℝ, f x ≠ g x) ∧ 
  (∃ x : ℝ, f x ≠ h x) ∧ 
  (∃ x : ℝ, g x ≠ h x) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_different_graphs_l459_45923


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_photo_ratio_theorem_l459_45981

/-- Represents the number of photos taken of different subjects -/
structure PhotoCounts where
  animals : ℕ
  flowers : ℕ
  scenery : ℕ

/-- Determines if the photo counts satisfy the given conditions -/
def satisfiesConditions (p : PhotoCounts) : Prop :=
  p.animals = 10 ∧
  p.scenery = p.flowers - 10 ∧
  p.animals + p.flowers + p.scenery = 45

/-- The ratio of flower photos to animal photos -/
def flowerToAnimalRatio (p : PhotoCounts) : ℚ :=
  (p.flowers : ℚ) / (p.animals : ℚ)

theorem photo_ratio_theorem (p : PhotoCounts) :
  satisfiesConditions p → flowerToAnimalRatio p = 11 / 5 := by
  sorry

#eval (11 : ℚ) / 5

end NUMINAMATH_CALUDE_ERRORFEEDBACK_photo_ratio_theorem_l459_45981


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_subset_count_of_P_l459_45980

def M : Finset ℕ := {0, 1, 2, 3, 4}
def N : Finset ℕ := {1, 3, 5}
def P : Finset ℕ := M ∩ N

theorem subset_count_of_P : Finset.card (Finset.powerset P) = 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_subset_count_of_P_l459_45980


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_prime_factor_of_expression_l459_45985

theorem largest_prime_factor_of_expression :
  (Nat.factors (15^2 + 10^3 + 5^6)).maximum? = some 139 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_prime_factor_of_expression_l459_45985


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sarah_trip_distance_l459_45937

def miles_per_gallon : ℚ := 28
def tank_capacity : ℚ := 16
def initial_distance : ℚ := 280
def gas_bought : ℚ := 6
def remaining_fraction : ℚ := 1/3

theorem sarah_trip_distance : 
  let initial_gas_used : ℚ := initial_distance / miles_per_gallon
  let remaining_gas_after_first_leg : ℚ := tank_capacity - initial_gas_used
  let gas_after_refuel : ℚ := remaining_gas_after_first_leg + gas_bought
  let gas_left_at_destination : ℚ := remaining_fraction * tank_capacity
  let gas_used_second_leg : ℚ := gas_after_refuel - gas_left_at_destination
  let second_leg_distance : ℚ := gas_used_second_leg * miles_per_gallon
  let total_distance : ℚ := initial_distance + second_leg_distance
  ⌊total_distance⌋ = 467 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sarah_trip_distance_l459_45937


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_at_means_l459_45955

noncomputable def f (x : ℝ) : ℝ := x^2 - 10*x + 16

-- Roots of f
def x₁ : ℝ := 8
def x₂ : ℝ := 2

-- Arithmetic mean of roots
noncomputable def arithmetic_mean : ℝ := (x₁ + x₂) / 2

-- Geometric mean of roots
noncomputable def geometric_mean : ℝ := Real.sqrt (x₁ * x₂)

theorem f_at_means :
  f arithmetic_mean = -9 ∧ f geometric_mean = -8 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_at_means_l459_45955


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_transportation_cost_minimization_l459_45927

/-- The transportation cost function -/
noncomputable def f (v : ℝ) : ℝ := 10000 / v + 4 * v

/-- The domain of the function -/
def D : Set ℝ := {v | 0 < v ∧ v ≤ 50}

theorem transportation_cost_minimization :
  (∀ v₁ v₂, v₁ ∈ D → v₂ ∈ D → v₁ < v₂ → f v₁ > f v₂) ∧
  (∀ v, v ∈ D → f v ≥ f 50) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_transportation_cost_minimization_l459_45927


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_canoe_kayak_rental_ratio_l459_45924

-- Define the daily rental costs
def canoe_cost : ℕ := 12
def kayak_cost : ℕ := 18

-- Define the total revenue
def total_revenue : ℕ := 504

-- Define the variables for number of canoes and kayaks
variable (c k : ℕ)

-- Define the conditions
def revenue_equation (c k : ℕ) : Prop := c * canoe_cost + k * kayak_cost = total_revenue
def canoe_kayak_difference (c k : ℕ) : Prop := c = k + 7

-- Define the ratio
def canoe_kayak_ratio (c k : ℕ) : Prop := c * 2 = k * 3

-- Theorem statement
theorem canoe_kayak_rental_ratio :
  ∀ c k : ℕ, revenue_equation c k → canoe_kayak_difference c k → canoe_kayak_ratio c k :=
by
  intros c k rev_eq diff_eq
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_canoe_kayak_rental_ratio_l459_45924


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_door_height_is_six_l459_45925

/-- The height of a door in a room with given dimensions and whitewashing costs -/
noncomputable def door_height (room_length room_width room_height : ℝ) 
                (cost_per_sqft : ℝ) 
                (door_width : ℝ) 
                (window_length window_width : ℝ) 
                (num_windows : ℕ) 
                (total_cost : ℝ) : ℝ :=
  let wall_area := 2 * (room_length + room_width) * room_height
  let window_area := (num_windows : ℝ) * window_length * window_width
  let whitewashed_area := (total_cost / cost_per_sqft) + window_area
  (wall_area - whitewashed_area) / door_width

theorem door_height_is_six :
  door_height 25 15 12 10 3 4 3 3 9060 = 6 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_door_height_is_six_l459_45925


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hexagon_ratio_theorem_l459_45920

-- Define the points
variable (A B C D E F U V W X Y Z : EuclideanSpace ℝ (Fin 2))

-- Define the hexagon
def is_convex_hexagon (A B C D E F : EuclideanSpace ℝ (Fin 2)) : Prop :=
  -- Add appropriate conditions for a convex hexagon
  sorry

-- Define the condition that points are intersections of triangle sides
def are_intersection_points (A B C D E F U V W X Y Z : EuclideanSpace ℝ (Fin 2)) : Prop :=
  -- Add appropriate conditions for intersection points
  sorry

-- Define the given ratio condition
def ratio_condition (A B C D E F U V W : EuclideanSpace ℝ (Fin 2)) : Prop :=
  ∃ (k : ℝ), norm (B - A) = k * norm (V - U) ∧
             norm (D - C) = k * norm (W - V) ∧
             norm (F - E) = k * norm (U - W)

-- State the theorem
theorem hexagon_ratio_theorem 
  (h1 : is_convex_hexagon A B C D E F)
  (h2 : are_intersection_points A B C D E F U V W X Y Z)
  (h3 : ratio_condition A B C D E F U V W) :
  ∃ (m : ℝ), norm (C - B) = m * norm (Y - X) ∧
             norm (E - D) = m * norm (Z - Y) ∧
             norm (A - F) = m * norm (X - Z) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hexagon_ratio_theorem_l459_45920


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_dot_product_in_triangle_l459_45950

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := 4 * Real.sqrt 3 * Real.sin x * Real.cos x - 4 * Real.sin x ^ 2 + 1

-- Define the triangle ABC
structure Triangle where
  A : ℝ  -- Angle A
  B : ℝ  -- Angle B
  C : ℝ  -- Angle C
  a : ℝ  -- Side opposite to A
  b : ℝ  -- Side opposite to B
  c : ℝ  -- Side opposite to C

-- State the theorem
theorem max_dot_product_in_triangle (t : Triangle) 
  (h1 : t.a = 2) 
  (h2 : ∀ x, f x ≤ f t.A) 
  (h3 : t.A ∈ Set.Ioo 0 Real.pi) :
  t.b * t.c * Real.cos t.A ≤ 6 + 4 * Real.sqrt 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_dot_product_in_triangle_l459_45950


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_100_value_l459_45995

def sequence_a : ℕ → ℕ
  | 0 => 2  -- Add this case to handle Nat.zero
  | 1 => 2
  | n + 1 => sequence_a n + 2 * n

theorem a_100_value : sequence_a 100 = 9902 := by
  -- Proof steps would go here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_100_value_l459_45995


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_geometrically_convex_l459_45940

noncomputable def f (x : ℝ) := (1/3:ℝ)^x - (1/4:ℝ)^x

theorem f_geometrically_convex :
  ∀ (x₁ x₂ : ℝ), x₁ ≥ 1 → x₂ ≥ 1 →
  f (Real.sqrt (x₁ * x₂)) ≥ Real.sqrt (f x₁ * f x₂) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_geometrically_convex_l459_45940


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_investment_time_calculation_l459_45974

/-- Given the investment ratio, profit ratio, and P's investment time, 
    calculate Q's investment time -/
theorem investment_time_calculation 
  (p_investment : ℝ) 
  (q_investment : ℝ) 
  (p_profit : ℝ) 
  (q_profit : ℝ) 
  (p_time : ℝ) 
  (h1 : p_investment / q_investment = 7 / 5.00001)
  (h2 : p_profit / q_profit = 7.00001 / 10)
  (h3 : p_time = 5) : 
  ∃ q_time : ℝ, 
    (p_investment * p_time) / (q_investment * q_time) = p_profit / q_profit ∧ 
    abs (q_time - 9.99857) < 0.00001 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_investment_time_calculation_l459_45974


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_arccot_three_fifths_l459_45944

theorem tan_arccot_three_fifths : Real.tan (Real.arctan (5 / 3)) = 5 / 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_arccot_three_fifths_l459_45944


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_between_given_lines_l459_45996

/-- The distance between two parallel lines -/
noncomputable def distance_parallel_lines (A B C D E F : ℝ) : ℝ :=
  |C - F| / Real.sqrt (A^2 + B^2)

theorem distance_between_given_lines :
  let l₁ : ℝ → ℝ → ℝ := λ x y ↦ x - 2*y - 2
  let l₂ : ℝ → ℝ → ℝ := λ x y ↦ 2*x - 4*y + 1
  distance_parallel_lines 1 (-2) (-2) 2 (-4) 1 = Real.sqrt 5 / 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_between_given_lines_l459_45996


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_impossible_to_equalize_l459_45932

/-- Represents a move on the chessboard -/
structure Move where
  row : Nat
  col : Nat

/-- Represents the state of the chessboard -/
def Board (n : Nat) := Fin n → Fin n → Int

/-- Applies a move to the board -/
def applyMove {n : Nat} (b : Board n) (m : Move) : Board n :=
  sorry

/-- Checks if all values on the board are equal -/
def allEqual {n : Nat} (b : Board n) : Prop :=
  sorry

/-- The main theorem stating the impossibility of equalizing the board -/
theorem impossible_to_equalize (n : Nat) (h : n > 100) :
  ∀ (b : Board n),
    (∃ (count : Nat), count = n * n - 1 ∧ 
      (∀ (i j : Fin n), b i j = 1 ∨ b i j = 0) ∧
      (count = (Finset.filter (λ (p : Fin n × Fin n) => b p.1 p.2 = 1) (Finset.univ.product Finset.univ)).card)) →
    ¬∃ (moves : List Move), allEqual (moves.foldl applyMove b) :=
  sorry

#check impossible_to_equalize

end NUMINAMATH_CALUDE_ERRORFEEDBACK_impossible_to_equalize_l459_45932


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_outcomes_sum_4_characterization_l459_45979

def Die : Type := Fin 6

def sum_of_dice (d1 d2 : Die) : Nat :=
  (d1.val + 1) + (d2.val + 1)

def outcomes_sum_4 : Set (Die × Die) :=
  {p | sum_of_dice p.1 p.2 = 4}

theorem outcomes_sum_4_characterization :
  outcomes_sum_4 = {(⟨2, by norm_num⟩, ⟨0, by norm_num⟩),
                    (⟨0, by norm_num⟩, ⟨2, by norm_num⟩),
                    (⟨1, by norm_num⟩, ⟨1, by norm_num⟩)} := by
  sorry

#check outcomes_sum_4_characterization

end NUMINAMATH_CALUDE_ERRORFEEDBACK_outcomes_sum_4_characterization_l459_45979


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_expression_l459_45960

theorem largest_expression (a b : ℝ) (ha : a > 1) (hb : b > 1) (hab : a ≠ b) :
  (a - b < a^2 + b^2) ∧ (a + b < a^2 + b^2) ∧ (2*a*b < a^2 + b^2) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_expression_l459_45960


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_infinite_series_sum_eq_five_fourths_l459_45968

/-- The sum of the infinite series Σ(2n + 3) / (n(n+1)(n+2)) for n from 1 to infinity -/
noncomputable def infinite_series_sum : ℝ :=
  ∑' n : ℕ, (2 * n + 3) / (n * (n + 1) * (n + 2))

/-- The sum of the infinite series is equal to 5/4 -/
theorem infinite_series_sum_eq_five_fourths : infinite_series_sum = 5/4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_infinite_series_sum_eq_five_fourths_l459_45968


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_equation_l459_45913

/-- A circle C with center (h, k) and radius r -/
structure Circle where
  h : ℝ
  k : ℝ
  r : ℝ

/-- The line on which the center of circle C lies -/
def centerLine (x y : ℝ) : Prop := x - 2*y - 3 = 0

/-- Point A through which circle C passes -/
def pointA : ℝ × ℝ := (2, -3)

/-- Point B through which circle C passes -/
def pointB : ℝ × ℝ := (-2, -5)

/-- Distance between two points -/
noncomputable def distance (x₁ y₁ x₂ y₂ : ℝ) : ℝ :=
  Real.sqrt ((x₂ - x₁)^2 + (y₂ - y₁)^2)

/-- Theorem stating the standard equation of circle C -/
theorem circle_equation (C : Circle) 
  (h_center : centerLine C.h C.k)
  (h_pointA : distance C.h C.k pointA.1 pointA.2 = C.r)
  (h_pointB : distance C.h C.k pointB.1 pointB.2 = C.r) :
  ∀ x y, (x - C.h)^2 + (y - C.k)^2 = C.r^2 ↔ (x + 1)^2 + (y + 2)^2 = 10 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_equation_l459_45913


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_five_odd_probability_l459_45938

-- Define a fair 6-sided die
def fair_die := Finset.range 6

-- Define the number of rolls
def num_rolls : ℕ := 6

-- Define the probability of rolling an odd number
def prob_odd : ℚ := 1/2

-- Define the probability of the desired outcome
def prob_five_odd : ℚ := (Nat.choose num_rolls 5 : ℚ) * prob_odd^5 * (1 - prob_odd)^(num_rolls - 5)

-- Theorem statement
theorem five_odd_probability : prob_five_odd = 3/32 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_five_odd_probability_l459_45938


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_absolute_roots_l459_45911

theorem sum_of_absolute_roots (x : ℂ) : 
  x^4 - 6*x^3 + 13*x^2 + 6*x - 40 = 0 → 
  ∃ r1 r2 r3 r4 : ℂ, 
    (x - r1) * (x - r2) * (x - r3) * (x - r4) = x^4 - 6*x^3 + 13*x^2 + 6*x - 40 ∧
    Complex.abs r1 + Complex.abs r2 + Complex.abs r3 + Complex.abs r4 = 5 + 2 * Real.sqrt 8.5 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_absolute_roots_l459_45911


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_two_point_three_repeating_equals_seven_thirds_l459_45957

/-- Represents a repeating decimal with a whole number part and a repeating digit -/
def repeating_decimal (whole : ℕ) (repeating : ℕ) : ℚ :=
  whole + (repeating : ℚ) / 9

/-- The repeating decimal 2.3̅ is equal to the fraction 7/3 -/
theorem two_point_three_repeating_equals_seven_thirds :
  repeating_decimal 2 3 = 7 / 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_two_point_three_repeating_equals_seven_thirds_l459_45957


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_carbon_dioxide_moles_l459_45975

/-- Represents the number of moles of a substance -/
def Moles : Type := ℕ

/-- Represents a chemical reaction between Sodium bicarbonate and Hydrochloric acid -/
structure Reaction where
  sodium_bicarbonate : Moles
  hydrochloric_acid : Moles
  water : Moles
  sodium_chloride : Moles
  carbon_dioxide : Moles

instance : OfNat Moles n where
  ofNat := n

/-- Theorem stating that 2 moles of Carbon dioxide are formed in the given reaction -/
theorem carbon_dioxide_moles (r : Reaction) :
  r.sodium_bicarbonate = 2 →
  r.hydrochloric_acid = 2 →
  r.water = 2 →
  r.sodium_chloride = 2 →
  r.carbon_dioxide = 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_carbon_dioxide_moles_l459_45975


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_tangent_length_l459_45989

/-- The minimum length of tangent lines from a line to a circle -/
theorem min_tangent_length 
  (l : Set (ℝ × ℝ)) 
  (c : Set (ℝ × ℝ)) 
  (hl : l = {p : ℝ × ℝ | p.1 - p.2 + 4 * Real.sqrt 2 = 0})
  (hc : c = {p : ℝ × ℝ | (p.1 - Real.sqrt 2 / 2)^2 + (p.2 + Real.sqrt 2 / 2)^2 = 1}) :
  (∃ (p q : ℝ × ℝ), p ∈ l ∧ q ∈ c ∧ 
    ∀ (p' q' : ℝ × ℝ), p' ∈ l → q' ∈ c → 
      Real.sqrt ((p'.1 - q'.1)^2 + (p'.2 - q'.2)^2) ≥ 2 * Real.sqrt 6) ∧
  (∃ (p q : ℝ × ℝ), p ∈ l ∧ q ∈ c ∧ 
    Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2) = 2 * Real.sqrt 6) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_tangent_length_l459_45989


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_s_value_l459_45929

/-- A polynomial of degree 4 with integer coefficients -/
structure Polynomial4 where
  p : ℤ
  q : ℤ
  r : ℤ
  s : ℤ

/-- The roots of the polynomial are all negative odd integers -/
def has_negative_odd_roots (poly : Polynomial4) : Prop :=
  ∃ (m₁ m₂ m₃ m₄ : ℤ), 
    (∀ i, i ∈ [m₁, m₂, m₃, m₄] → i > 0 ∧ Odd i) ∧
    (∀ x : ℤ, x^4 + poly.p * x^3 + poly.q * x^2 + poly.r * x + poly.s = 
          (x + m₁) * (x + m₂) * (x + m₃) * (x + m₄))

theorem polynomial_s_value (poly : Polynomial4) 
  (h1 : has_negative_odd_roots poly)
  (h2 : poly.p + poly.q + poly.r + poly.s = 2023) :
  poly.s = 624 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_s_value_l459_45929


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_multiples_between_1_and_300_l459_45984

theorem multiples_between_1_and_300 :
  ∃! n : ℕ, n > 0 ∧ (∃! s : Finset ℕ, 
    s.card = 33 ∧ 
    (∀ m ∈ s, m ≥ 1 ∧ m ≤ 300 ∧ m % n = 0) ∧
    (∀ m : ℕ, m ≥ 1 ∧ m ≤ 300 ∧ m % n = 0 → m ∈ s)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_multiples_between_1_and_300_l459_45984


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_domain_values_inequality_solution_set_l459_45973

-- Define the function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.sqrt (a * x^2 - 3 * x + 2)

-- Define the domain of f
def domain (b : ℝ) : Set ℝ := {x | x < 1 ∨ x > b}

-- Theorem for part (1)
theorem function_domain_values :
  ∃ (a b : ℝ), (∀ x, f a x ∈ Set.Iio 0 ↔ x ∈ domain b) ∧ a = 1 ∧ b = 2 := by
  sorry

-- Define the inequality
def inequality (a b c : ℝ) (x : ℝ) : Prop := (x - c) / (a * x - b) > 0

-- Theorem for part (2)
theorem inequality_solution_set (a b c : ℝ) :
  a = 1 ∧ b = 2 →
  (c > 2 → ∀ x, inequality a b c x ↔ (x > c ∨ x < 2)) ∧
  (c < 2 → ∀ x, inequality a b c x ↔ (x > 2 ∨ x < c)) ∧
  (c = 2 → ∀ x, inequality a b c x ↔ x ≠ 2) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_domain_values_inequality_solution_set_l459_45973


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_x_minus_y_l459_45994

theorem min_x_minus_y (x y : ℝ) (h1 : x ∈ Set.Icc 0 (2 * Real.pi))
  (h2 : y ∈ Set.Icc 0 (2 * Real.pi))
  (h3 : 2 * Real.sin x * Real.cos y - Real.sin x + Real.cos y = 1/2) :
  ∃ (min : ℝ), min = -Real.pi/2 ∧ ∀ (z : ℝ), z = x - y → min ≤ z :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_x_minus_y_l459_45994


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetry_axis_l459_45992

noncomputable def f (x : ℝ) := Real.cos (2 * x + Real.pi / 2)

theorem symmetry_axis (x : ℝ) :
  f ((-Real.pi/4) + x) = f ((-Real.pi/4) - x) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetry_axis_l459_45992


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f1_satisfies_negative_inversion_f2_does_not_satisfy_negative_inversion_f3_satisfies_negative_inversion_l459_45971

-- Define the "negative inversion" property
def negative_inversion (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, x ≠ 0 → f (1 / x) = -f x

-- Function 1
noncomputable def f1 (x : ℝ) : ℝ := x - 1 / x

-- Function 2
noncomputable def f2 (x : ℝ) : ℝ := x + 1 / x

-- Function 3 (piecewise)
noncomputable def f3 (x : ℝ) : ℝ :=
  if 0 < x ∧ x < 1 then x
  else if x = 1 then 0
  else if x > 1 then -1 / x
  else 0  -- This case is added to make the function total

-- Theorems to prove
theorem f1_satisfies_negative_inversion : negative_inversion f1 := by sorry

theorem f2_does_not_satisfy_negative_inversion : ¬negative_inversion f2 := by sorry

theorem f3_satisfies_negative_inversion : ∀ x : ℝ, x > 0 → f3 (1 / x) = -f3 x := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f1_satisfies_negative_inversion_f2_does_not_satisfy_negative_inversion_f3_satisfies_negative_inversion_l459_45971


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_currency_conversion_area_conversion_l459_45997

-- Define the conversion rate for hectares to square meters
def hectare_to_sqm : ℕ := 10000

-- Define a structure for Chinese currency
structure ChineseCurrency where
  yuan : ℕ
  jiao : ℕ
  fen : ℕ

-- Function to convert decimal yuan to ChineseCurrency
def decimal_to_chinese_currency (amount : ℚ) : ChineseCurrency :=
  { yuan := amount.floor.toNat,
    jiao := ((amount - amount.floor) * 10).floor.toNat,
    fen := ((amount * 100 - (amount * 100).floor) * 10).floor.toNat }

-- Theorem for the currency conversion
theorem currency_conversion :
  decimal_to_chinese_currency 7.81 = ChineseCurrency.mk 7 8 1 := by sorry

-- Theorem for the area conversion
theorem area_conversion :
  3 * hectare_to_sqm + 100 = 30100 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_currency_conversion_area_conversion_l459_45997


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_same_terminal_side_angle_l459_45930

/-- The angle that has the same terminal side as -π/3 in the range [0, 2π) is 5π/3 -/
theorem same_terminal_side_angle : ∃ (k : ℤ), 
  ((-π/3 + 2*π*k : ℝ) ≥ 0 ∧ (-π/3 + 2*π*k : ℝ) < 2*π ∧ (-π/3 + 2*π*k : ℝ) = 5*π/3) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_same_terminal_side_angle_l459_45930


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exactly_three_correct_probability_l459_45978

def num_packages : ℕ := 5
def num_houses : ℕ := 5

def correct_delivery (delivery : Fin num_packages → Fin num_houses) : ℕ :=
  (Finset.univ.filter (λ i => delivery i = i)).card

theorem exactly_three_correct_probability :
  (Fintype.card {delivery : Fin num_packages → Fin num_houses | correct_delivery delivery = 3} : ℚ) /
  (Fintype.card (Fin num_packages → Fin num_houses) : ℚ) = 1 / 6 := by
  sorry

#eval num_packages
#eval num_houses

end NUMINAMATH_CALUDE_ERRORFEEDBACK_exactly_three_correct_probability_l459_45978


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_propositions_correctness_l459_45919

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relations between lines and planes
variable (parallel : Line → Line → Prop)
variable (parallelToPlane : Line → Plane → Prop)
variable (perpendicular : Line → Line → Prop)
variable (perpendicularToPlane : Line → Plane → Prop)
variable (intersect : Line → Line → Prop)
variable (skew : Line → Line → Prop)
variable (contained : Line → Plane → Prop)

-- Define the lines and plane
variable (a b c : Line)
variable (M : Plane)

-- State that a, b, and c are distinct
variable (distinct_lines : a ≠ b ∧ b ≠ c ∧ a ≠ c)

-- Define the propositions
def proposition1 (Line Plane : Type)
  (parallel : Line → Line → Prop)
  (parallelToPlane : Line → Plane → Prop)
  (intersect : Line → Line → Prop)
  (skew : Line → Line → Prop)
  (a b : Line) (M : Plane) : Prop :=
  (parallelToPlane a M ∧ parallelToPlane b M) →
  (parallel a b ∨ intersect a b ∨ skew a b)

def proposition2 (Line Plane : Type)
  (parallel : Line → Line → Prop)
  (parallelToPlane : Line → Plane → Prop)
  (contained : Line → Plane → Prop)
  (a b : Line) (M : Plane) : Prop :=
  (contained b M ∧ parallel a b) → parallelToPlane a M

def proposition3 (Line : Type)
  (parallel : Line → Line → Prop)
  (perpendicular : Line → Line → Prop)
  (a b c : Line) : Prop :=
  (perpendicular a c ∧ perpendicular b c) → parallel a b

def proposition4 (Line Plane : Type)
  (parallel : Line → Line → Prop)
  (perpendicularToPlane : Line → Plane → Prop)
  (a b : Line) (M : Plane) : Prop :=
  (perpendicularToPlane a M ∧ perpendicularToPlane b M) → parallel a b

-- State the theorem
theorem propositions_correctness :
  proposition1 Line Plane parallel parallelToPlane intersect skew a b M ∧
  ¬proposition2 Line Plane parallel parallelToPlane contained a b M ∧
  ¬proposition3 Line parallel perpendicular a b c ∧
  proposition4 Line Plane parallel perpendicularToPlane a b M := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_propositions_correctness_l459_45919


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_rabbits_two_colors_l459_45912

/-- Represents the number of rabbits of a particular color -/
def RabbitCount := ℕ

/-- The total number of rabbits in the hat -/
def totalRabbits : ℕ := 100

/-- The number of colors of rabbits -/
def numColors : ℕ := 3

/-- The number of rabbits that guarantees all three colors -/
def allColorsGuarantee : ℕ := 81

/-- A function that checks if a given number of rabbits guarantees at least two colors -/
def guaranteesTwoColors (n : ℕ) : Prop :=
  ∀ (a b c : ℕ), a + b + c = totalRabbits →
  (∀ k : ℕ, k ≥ allColorsGuarantee → k ≤ a + b + c → a > 0 ∧ b > 0 ∧ c > 0) →
  n ≤ a + b + c → (a > 0 ∧ b > 0) ∨ (a > 0 ∧ c > 0) ∨ (b > 0 ∧ c > 0)

/-- The theorem stating that 61 is the minimum number of rabbits that guarantees two colors -/
theorem min_rabbits_two_colors :
  guaranteesTwoColors 61 ∧ ¬guaranteesTwoColors 60 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_rabbits_two_colors_l459_45912


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_x_value_f_range_l459_45970

noncomputable section

-- Define the vectors a and b
def a (x : ℝ) : ℝ × ℝ := (1, Real.cos (x - Real.pi/6))
def b (x : ℝ) : ℝ × ℝ := (Real.sqrt 3, Real.sqrt 3 * Real.sin (x - Real.pi/6))

-- Define the parallel condition
def parallel (v w : ℝ × ℝ) : Prop :=
  v.1 * w.2 = v.2 * w.1

-- Define the function f
def f (x : ℝ) : ℝ := 2 - 2 * Real.sin (2*x - Real.pi/6)

-- Theorem for part 1
theorem tan_x_value (x : ℝ) (h : parallel (a x) (b x)) :
  Real.tan x = 2 + Real.sqrt 3 := by sorry

-- Theorem for part 2
theorem f_range (x : ℝ) (h : x ∈ Set.Ioo 0 (Real.pi/2)) :
  f x ∈ Set.Icc 0 3 := by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_x_value_f_range_l459_45970


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_manger_approach_orders_l459_45903

-- Define the type for the three people
inductive Person : Type
  | M : Person  -- Melchior
  | C : Person  -- Caspar
  | B : Person  -- Balthazar
deriving BEq, DecidableEq

-- Define a type for the order of approaching the manger
def Order := List Person

-- Define a function to check if an order satisfies Melchior's conditions
def melchior_conditions (order : Order) : Prop :=
  (order.reverse.head? = some Person.M → order.head? ≠ some Person.C) ∧
  (order.head? = some Person.M → order.reverse.head? ≠ some Person.C)

-- Define a function to check if an order satisfies Balthazar's conditions
def balthazar_conditions (order : Order) : Prop :=
  (order.reverse.head? = some Person.B →
    ¬(order.indexOf Person.M > order.indexOf Person.C)) ∧
  (order.head? = some Person.B →
    ¬(order.indexOf Person.M < order.indexOf Person.C))

-- Define a function to check if an order satisfies Caspar's condition
def caspar_condition (order : Order) : Prop :=
  (order.head? ≠ some Person.C ∧ order.reverse.head? ≠ some Person.C) →
    ¬(order.indexOf Person.M < order.indexOf Person.B)

-- Define a function to check if an order is valid
def valid_order (order : Order) : Prop :=
  order.length = 3 ∧
  order.toFinset = {Person.M, Person.C, Person.B} ∧
  melchior_conditions order ∧
  balthazar_conditions order ∧
  caspar_condition order

-- The main theorem
theorem manger_approach_orders :
  ∃ (order1 order2 : Order),
    valid_order order1 ∧
    valid_order order2 ∧
    order1 ≠ order2 ∧
    (∀ (order : Order), valid_order order → (order = order1 ∨ order = order2)) :=
by
  -- We'll use sorry to skip the proof
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_manger_approach_orders_l459_45903


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_parameters_l459_45942

/-- The equation of an ellipse -/
def ellipse_equation (x y : ℝ) : Prop :=
  2 * x^2 + y^2 + 42 = 8 * x + 36 * y

/-- The center of the ellipse -/
def ellipse_center : ℝ × ℝ := (2, 18)

/-- The major axis length of the ellipse -/
noncomputable def major_axis_length : ℝ := Real.sqrt 290

/-- The minor axis length of the ellipse -/
noncomputable def minor_axis_length : ℝ := Real.sqrt 145

/-- Theorem stating that the given equation represents an ellipse with the specified center and axis lengths -/
theorem ellipse_parameters :
  ∀ x y : ℝ, ellipse_equation x y →
  ∃ h k a b : ℝ,
    h = ellipse_center.1 ∧
    k = ellipse_center.2 ∧
    a = major_axis_length ∧
    b = minor_axis_length ∧
    ((x - h) / b)^2 + ((y - k) / a)^2 = 1 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_parameters_l459_45942


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_water_tower_staircase_distance_l459_45908

/-- The total distance along a spiral staircase and vertical ladder on a cylindrical water tower -/
theorem water_tower_staircase_distance (r h l : ℝ) (hr : r = 10) (hh : h = 30) (hl : l = 5) :
  let circumference := 2 * Real.pi * r
  let staircase_height := h - l
  let staircase_length := Real.sqrt ((circumference^2) + (staircase_height^2))
  let total_distance := staircase_length + l
  abs (total_distance - 72.6) < 0.1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_water_tower_staircase_distance_l459_45908


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_g_one_value_l459_45917

-- Define the polynomial f(x) = x^2 + px + q
def f (p q : ℝ) : ℝ → ℝ := λ x ↦ x^2 + p*x + q

-- Define g(x) as a polynomial with leading coefficient 1
-- whose roots are the reciprocals of the roots of f(x)
def g (p q : ℝ) : ℝ → ℝ := sorry

-- State the theorem
theorem g_one_value (p q : ℝ) (h : p < q) :
  g p q 1 = (1 + p + q) / q := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_g_one_value_l459_45917


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_curler_ratio_l459_45928

theorem curler_ratio : 
  ∀ (pink blue green : ℕ),
  pink + blue + green = 16 →
  blue = 2 * pink →
  green = 4 →
  (pink : ℚ) / 16 = 1 / 4 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_curler_ratio_l459_45928


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l459_45993

noncomputable def f (x : ℝ) : ℝ := 2 * Real.sin x ^ 2 - Real.cos (2 * x + Real.pi / 3)

theorem f_properties :
  (∃ T > 0, ∀ x, f (x + T) = f x ∧ ∀ S, 0 < S → S < T → ∃ y, f (y + S) ≠ f y) ∧
  (∀ x ∈ Set.Icc 0 (Real.pi / 2), f x ≥ -1 / 2) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l459_45993


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tiling_count_tiling_mod_l459_45991

/-- The four possible colors for tiles -/
inductive Color
  | Red | Blue | Green | Yellow
deriving Repr, Inhabited

/-- A tiling of a 9x1 board using mx1 tiles -/
structure Tiling where
  tiles : List (Nat × Color)
  valid : (tiles.map Prod.fst).sum = 9
  all_colors : ∀ c : Color, c ∈ tiles.map Prod.snd

/-- The number of valid tilings -/
def M : Nat := sorry

theorem tiling_count : M = 2286720 := sorry

theorem tiling_mod : M % 1000 = 720 := sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tiling_count_tiling_mod_l459_45991


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_problem_l459_45958

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := Real.sin x ^ 2 + (Real.sqrt 2 / 2) * Real.cos (2 * x + Real.pi / 4)

-- Define the triangle ABC
structure Triangle :=
  (a b c : ℝ)
  (A B C : ℝ)

-- Theorem statement
theorem triangle_area_problem :
  -- The smallest positive period of f is π
  (∃ (T : ℝ), T > 0 ∧ (∀ (x : ℝ), f (x + T) = f x) ∧ 
    (∀ (T' : ℝ), T' > 0 → (∀ (x : ℝ), f (x + T') = f x) → T ≤ T') ∧ T = Real.pi) ∧
  -- In △ABC, given conditions
  (∀ (triangle : Triangle),
    triangle.b = Real.sqrt 3 →
    f (triangle.B / 2 + Real.pi / 4) = 1/8 →
    -- The maximum area of △ABC is (3√7)/4
    (1/2 * triangle.a * triangle.c * Real.sin triangle.B ≤ 3 * Real.sqrt 7 / 4)) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_problem_l459_45958


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parallelogram_diagonal_intersection_l459_45933

/-- If ABCD is a parallelogram with diagonals AC and BD intersecting at point O,
    and AB + AD = k * AO, then k = 2 -/
theorem parallelogram_diagonal_intersection (A B C D O : ℝ × ℝ) (k : ℝ) :
  (∀ X : ℝ × ℝ, (X - A) + (X - C) = (B - D)) →  -- ABCD is a parallelogram
  (∃ t : ℝ, O = A + t • (C - A) ∧ O = B + (1 - t) • (D - B)) →  -- AC and BD intersect at O
  (B - A) + (D - A) = k • (O - A) →  -- AB + AD = k * AO
  k = 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parallelogram_diagonal_intersection_l459_45933


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_special_sequence_exists_l459_45972

-- Define the type for our sequence
def SpecialSequence := ℕ+ → ℕ+

-- Define the tau function
def tau (n : ℕ+) : ℕ := (Nat.divisors n.val).card

-- Define the property that every positive integer occurs exactly once
def containsAllIntegers (a : SpecialSequence) : Prop :=
  ∀ n : ℕ+, ∃ i : ℕ+, a i = n ∧ ∀ j : ℕ+, j ≠ i → a j ≠ n

-- Define the divisibility property
def hasDivisibilityProperty (a : SpecialSequence) : Prop :=
  ∀ n : ℕ+, (n : ℕ) ∣ tau (n * (a (n + 1))^(n : ℕ) + (n + 1) * (a n)^((n + 1) : ℕ))

-- State the theorem
theorem special_sequence_exists : ∃ a : SpecialSequence, 
  containsAllIntegers a ∧ hasDivisibilityProperty a := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_special_sequence_exists_l459_45972


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_right_triangle_30_60_90_l459_45907

theorem right_triangle_30_60_90 (X Y Z : ℝ × ℝ) (hypotenuse : ℝ) :
  -- XYZ is a right triangle with XZ as hypotenuse
  (Z.1 - X.1)^2 + (Z.2 - X.2)^2 = hypotenuse^2 →
  (Y.1 - X.1)^2 + (Y.2 - X.2)^2 + (Z.1 - Y.1)^2 + (Z.2 - Y.2)^2 = hypotenuse^2 →
  -- Angle XYZ is 30°
  Real.cos (30 * π / 180) = (X.1 - Y.1) * (Z.1 - Y.1) + (X.2 - Y.2) * (Z.2 - Y.2) / 
    (((X.1 - Y.1)^2 + (X.2 - Y.2)^2) * ((Z.1 - Y.1)^2 + (Z.2 - Y.2)^2))^(1/2) →
  -- Length of YZ is 12
  (Z.1 - Y.1)^2 + (Z.2 - Y.2)^2 = 12^2 →
  -- Then the length of XY is 12√3
  (X.1 - Y.1)^2 + (X.2 - Y.2)^2 = (12 * Real.sqrt 3)^2 := by
    sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_right_triangle_30_60_90_l459_45907


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_of_inequality_l459_45956

-- Define the piecewise function f as noncomputable
noncomputable def f (x : ℝ) : ℝ :=
  if x ≥ 0 then Real.log (x + 1) else -2 * x^2

-- State the theorem
theorem solution_set_of_inequality :
  {x : ℝ | f (x + 2) < f (x^2 + 2*x)} = {x : ℝ | x < -2 ∨ x > 1} := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_of_inequality_l459_45956


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_moroccanRestaurantDishes_l459_45921

/-- A Moroccan restaurant received couscous shipments and makes dishes -/
structure CouscousRestaurant where
  shipment1 : ℕ
  shipment2 : ℕ
  shipment3 : ℕ
  poundsPerDish : ℕ

/-- The specific restaurant with given shipment amounts and pounds per dish -/
def moroccanRestaurant : CouscousRestaurant where
  shipment1 := 7
  shipment2 := 13
  shipment3 := 45
  poundsPerDish := 5

/-- Calculate the number of dishes that can be made -/
def calculateDishes (r : CouscousRestaurant) : ℕ :=
  (r.shipment1 + r.shipment2 + r.shipment3) / r.poundsPerDish

/-- Theorem stating that the restaurant can make 13 dishes -/
theorem moroccanRestaurantDishes :
  calculateDishes moroccanRestaurant = 13 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_moroccanRestaurantDishes_l459_45921


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_find_n_l459_45902

theorem find_n : ∃ n : ℚ, (7 : ℝ) ^ (5 * (n : ℝ)) = (1 / 7 : ℝ) ^ (2 * (n : ℝ) - 18) → n = 18 / 7 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_find_n_l459_45902


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_students_only_science_l459_45916

/-- The total number of students -/
def total : Nat := 120

/-- The number of students in Science -/
def science_count : Nat := 80

/-- The number of students in Math -/
def math_count : Nat := 75

/-- Define the universe of students -/
def U : Finset Nat := Finset.range total

variable (S M : Finset Nat)

/-- All students are in either Science, Math, or both -/
axiom all_students : S ∪ M = U

/-- The number of students in Science -/
axiom science_size : S.card = science_count

/-- The number of students in Math -/
axiom math_size : M.card = math_count

/-- The theorem to prove -/
theorem students_only_science : (S \ M).card = 45 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_students_only_science_l459_45916


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_branches_downwards_l459_45901

-- Define the parabola
noncomputable def parabola (a x : ℝ) : ℝ := 4 * a * x^2 + 4 * (a + 1) * x + a^2 + a + 3 + 1 / a

-- Theorem statement
theorem parabola_branches_downwards (a : ℝ) (h1 : a ≠ 0) 
  (h2 : ∃ x1 x2 : ℝ, x1 ≠ x2 ∧ parabola a x1 = 0 ∧ parabola a x2 = 0) :
  4 * a < 0 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_branches_downwards_l459_45901


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_union_A_B_complement_A_intersect_B_l459_45922

open Set

-- Define the universal set U
def U : Set ℝ := {x | -2 < x ∧ x < 12}

-- Define set A
def A : Set ℝ := {x | 3 ≤ x ∧ x < 7}

-- Define set B
def B : Set ℝ := {x | 2 ≤ x ∧ x < 5}

-- Theorem for A ∪ B
theorem union_A_B : A ∪ B = {x | 2 ≤ x ∧ x < 7} := by sorry

-- Theorem for (U \ A) ∩ B
theorem complement_A_intersect_B : (U \ A) ∩ B = {x | 2 ≤ x ∧ x < 3} := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_union_A_B_complement_A_intersect_B_l459_45922


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_number_opposite_parts_l459_45998

theorem complex_number_opposite_parts (b : ℝ) : 
  let z : ℂ := -b * Complex.I / (2 * Complex.I)
  (z.re = -z.im) → b = -2/3 := by
  intro h
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_number_opposite_parts_l459_45998


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_point_coordinates_l459_45962

-- Define the ellipse
def is_on_ellipse (x y : ℝ) : Prop := x^2/5 + y^2/4 = 1

-- Define the foci of the ellipse
def foci : ℝ × ℝ × ℝ × ℝ := (-1, 0, 1, 0)

-- Define the area of the triangle formed by P and the foci
def triangle_area (x y : ℝ) : ℝ := abs y

-- Theorem statement
theorem point_coordinates (x y : ℝ) :
  is_on_ellipse x y ∧ triangle_area x y = 1 →
  (x = Real.sqrt 15 / 2 ∨ x = -Real.sqrt 15 / 2) ∧ (y = 1 ∨ y = -1) := by
  sorry

#check point_coordinates

end NUMINAMATH_CALUDE_ERRORFEEDBACK_point_coordinates_l459_45962


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_partition_equal_sum_subsets_l459_45986

theorem partition_equal_sum_subsets (p : ℕ) (k : ℕ) :
  (p.Prime) →
  (∃ (partition : Finset (Finset ℕ)),
    (partition.card = p) ∧
    (∀ S ∈ partition, ∀ x ∈ S, x ≤ k) ∧
    (∀ S ∈ partition, ∀ T ∈ partition, S ≠ T → S ∩ T = ∅) ∧
    (∀ S ∈ partition, ∀ T ∈ partition, (S.sum id) = (T.sum id)) ∧
    (partition.biUnion id = Finset.range k)) ↔
  (∃ (n : ℕ), n % 2 = 0 ∧ (k = n * p ∨ k = n * p - 1)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_partition_equal_sum_subsets_l459_45986


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_bills_tv_width_l459_45905

/-- The width of Bill's TV in inches -/
def W : ℝ := sorry

/-- The height of Bill's TV in inches -/
def bill_height : ℝ := 100

/-- The width of Bob's TV in inches -/
def bob_width : ℝ := 70

/-- The height of Bob's TV in inches -/
def bob_height : ℝ := 60

/-- The weight of the TV per square inch in ounces -/
def weight_per_sq_inch : ℝ := 4

/-- The weight difference between the TVs in pounds -/
def weight_diff_pounds : ℝ := 150

/-- The number of ounces in a pound -/
def oz_per_pound : ℝ := 16

theorem bills_tv_width :
  W * bill_height * weight_per_sq_inch - bob_width * bob_height * weight_per_sq_inch = weight_diff_pounds * oz_per_pound →
  W = 48 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_bills_tv_width_l459_45905


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_geese_survival_theorem_l459_45939

noncomputable def geese_survival (number_of_eggs : ℕ) : ℕ :=
  (((2/5 : ℚ) * ((3/4 : ℚ) * ((1/2 : ℚ) * number_of_eggs))).floor).toNat

theorem geese_survival_theorem (number_of_eggs : ℕ) :
  geese_survival number_of_eggs = 
    (((2/5 : ℚ) * ((3/4 : ℚ) * ((1/2 : ℚ) * number_of_eggs))).floor).toNat :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_geese_survival_theorem_l459_45939


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_solution_range_l459_45977

theorem inequality_solution_range (a : ℝ) : 
  (∃ (s : Finset ℤ), (∀ x : ℤ, x ∈ s ↔ (2*x - 1)^2 < a*x^2) ∧ s.card = 3) →
  25/9 < a ∧ a ≤ 49/16 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_solution_range_l459_45977


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cinema_cup_holder_probability_l459_45990

/-- Probability that all n people place their drink in a cup holder -/
def p (n : ℕ) : ℝ :=
  sorry

/-- The sum of all p_n from n=1 to infinity -/
noncomputable def sum_p : ℝ :=
  sorry

theorem cinema_cup_holder_probability :
  sum_p = (2 * Real.sqrt (Real.exp 1) - 2) / (2 - Real.sqrt (Real.exp 1)) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cinema_cup_holder_probability_l459_45990


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_average_height_of_four_people_l459_45947

def height_difference (h₁ h₂ : ℕ) : ℕ := h₂ - h₁

theorem average_height_of_four_people
  (h₁ h₂ h₃ h₄ : ℕ)
  (order : h₁ < h₂ ∧ h₂ < h₃ ∧ h₃ < h₄)
  (diff_1_2 : height_difference h₁ h₂ = 2)
  (diff_2_3 : height_difference h₂ h₃ = 2)
  (diff_3_4 : height_difference h₃ h₄ = 6)
  (h₄_height : h₄ = 84) :
  (h₁ + h₂ + h₃ + h₄) / 4 = 78 := by
  sorry

#check average_height_of_four_people

end NUMINAMATH_CALUDE_ERRORFEEDBACK_average_height_of_four_people_l459_45947


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_right_triangle_points_l459_45934

/-- A point in a 2D plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- Check if three points form a right triangle -/
def isRightTriangle (p1 p2 p3 : Point) : Prop :=
  let v1 := (p2.x - p1.x, p2.y - p1.y)
  let v2 := (p3.x - p1.x, p3.y - p1.y)
  let v3 := (p3.x - p2.x, p3.y - p2.y)
  v1.1 * v2.1 + v1.2 * v2.2 = 0 ∨
  v1.1 * v3.1 + v1.2 * v3.2 = 0 ∨
  v2.1 * v3.1 + v2.2 * v3.2 = 0

/-- A set of points where any three form a right triangle -/
def RightTriangleSet (s : Set Point) : Prop :=
  ∀ p1 p2 p3, p1 ∈ s → p2 ∈ s → p3 ∈ s →
    p1 ≠ p2 ∧ p2 ≠ p3 ∧ p1 ≠ p3 → isRightTriangle p1 p2 p3

/-- The theorem stating that the maximum number of points is 4 -/
theorem max_right_triangle_points :
  ∃ (s : Finset Point), RightTriangleSet s ∧ s.card = 4 ∧
  ∀ (t : Finset Point), RightTriangleSet t → t.card ≤ 4 :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_right_triangle_points_l459_45934
