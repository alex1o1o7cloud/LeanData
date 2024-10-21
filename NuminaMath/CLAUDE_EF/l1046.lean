import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_distance_greater_than_three_l1046_104665

/-- A type representing a point in a plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- Calculate the distance between two points -/
noncomputable def distance (p q : Point) : ℝ :=
  Real.sqrt ((p.x - q.x)^2 + (p.y - q.y)^2)

/-- A configuration of 5 points in a plane -/
structure Configuration where
  points : Fin 5 → Point
  min_distance : ∀ (i j : Fin 5), i ≠ j → distance (points i) (points j) > 2

/-- Theorem: In any configuration of 5 points in a plane where the distance between
    any two points is greater than 2, there exists a pair of points with distance
    greater than 3 -/
theorem exists_distance_greater_than_three (config : Configuration) :
  ∃ (i j : Fin 5), i ≠ j ∧ distance (config.points i) (config.points j) > 3 := by
  sorry

#check exists_distance_greater_than_three

end NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_distance_greater_than_three_l1046_104665


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l1046_104637

-- Define the vector operation
def vector_op (a b : ℝ × ℝ) : ℝ × ℝ := (a.1 * b.1, a.2 * b.2)

-- Define the vectors
noncomputable def m : ℝ × ℝ := (2, 1/2)
noncomputable def n : ℝ × ℝ := (Real.pi/3, 0)

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := 
  let P := fun t : ℝ => (t, Real.sin t)
  let Q := fun t : ℝ => vector_op (P t) m + n
  (Q x).2

-- Theorem statement
theorem f_properties :
  (∃ (A : ℝ), ∀ (x : ℝ), f x ≤ A ∧ A = 1/2) ∧
  (∃ (T : ℝ), ∀ (x : ℝ), f (x + T) = f x ∧ T = 4*Real.pi ∧ 
    ∀ (T' : ℝ), (0 < T' ∧ T' < T) → ∃ (x : ℝ), f (x + T') ≠ f x) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l1046_104637


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_m_intersecting_line_m_l1046_104684

-- Define the line l: x - my + 3 = 0
def line (m : ℝ) (x y : ℝ) : Prop := x - m * y + 3 = 0

-- Define the circle C: x² + y² - 6x + 5 = 0
def circleC (x y : ℝ) : Prop := x^2 + y^2 - 6*x + 5 = 0

-- Theorem for the tangent case
theorem tangent_line_m (m : ℝ) : 
  (∃ x y : ℝ, line m x y ∧ circleC x y ∧ 
  (∀ x' y' : ℝ, line m x' y' → circleC x' y' → (x = x' ∧ y = y'))) → 
  (m = 2 * Real.sqrt 2 ∨ m = -2 * Real.sqrt 2) :=
sorry

-- Theorem for the intersecting case with specific chord length
theorem intersecting_line_m (m : ℝ) : 
  (∃ x₁ y₁ x₂ y₂ : ℝ, line m x₁ y₁ ∧ line m x₂ y₂ ∧ 
  circleC x₁ y₁ ∧ circleC x₂ y₂ ∧ 
  (x₁ - x₂)^2 + (y₁ - y₂)^2 = (2 * Real.sqrt 10 / 5)^2) → 
  (m = 3 ∨ m = -3) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_m_intersecting_line_m_l1046_104684


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_sum_is_46_l1046_104662

def Grid := Fin 2 → Fin 3 → Nat

def valid_arrangement (g : Grid) : Prop :=
  Finset.toSet {1, 5, 9, 12, 15, 18} = Finset.toSet {g 0 0, g 0 1, g 0 2, g 1 0, g 1 1, g 1 2}

def sum_equal (g : Grid) : Prop :=
  g 0 0 + g 0 1 + g 0 2 + g 1 2 = g 0 0 + g 1 0 + g 1 1 + g 1 2

def horizontal_sum (g : Grid) : Nat := g 0 0 + g 0 1 + g 0 2 + g 1 2

def vertical_sum (g : Grid) : Nat := g 0 0 + g 1 0 + g 1 1 + g 1 2

theorem max_sum_is_46 :
  ∀ g : Grid, valid_arrangement g → sum_equal g →
  (horizontal_sum g ≤ 46 ∧ vertical_sum g ≤ 46) ∧
  ∃ g' : Grid, valid_arrangement g' ∧ sum_equal g' ∧ 
  (horizontal_sum g' = 46 ∨ vertical_sum g' = 46) :=
by
  sorry

#check max_sum_is_46

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_sum_is_46_l1046_104662


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_combined_value_abc_l1046_104636

-- Define the conversion rate from paise to rupees
def paise_to_rupees (paise : ℚ) : ℚ := paise / 100

-- Define a as a natural number
def a : ℕ := 190

-- Define b as an integer
def b : ℤ := 3 * (a : ℤ) - 50

-- Define c as a natural number
def c : ℕ := ((a : ℤ) - b).natAbs ^ 2

-- Theorem statement
theorem combined_value_abc : 
  0.5 / 100 * (a : ℚ) = paise_to_rupees 95 ∧ 
  (a : ℕ) > 0 ∧
  (c : ℕ) > 0 ∧
  (a : ℚ) + (b : ℚ) + (c : ℚ) = 109610 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_combined_value_abc_l1046_104636


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_m_range_for_sufficient_condition_l1046_104623

theorem m_range_for_sufficient_condition (x m : ℝ) : 
  (∀ x, x^2 - 3*x - 4 ≤ 0 → x^2 - 6*x + 9 - m^2 ≤ 0) ∧ 
  (∃ x, x^2 - 6*x + 9 - m^2 ≤ 0 ∧ ¬(x^2 - 3*x - 4 ≤ 0)) →
  m ∈ Set.Iic (-4) ∪ Set.Ici 4 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_m_range_for_sufficient_condition_l1046_104623


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exactly_one_correct_proposition_l1046_104694

-- Define the basic geometric objects
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

structure Line3D where
  point : Point3D
  direction : Point3D

structure Plane3D where
  point : Point3D
  normal : Point3D

-- Define the geometric relations
def perpendicular (l : Line3D) (p : Plane3D) : Prop := sorry
def parallel_lines (l1 l2 : Line3D) : Prop := sorry
def parallel_planes (p1 p2 : Plane3D) : Prop := sorry
def perpendicular_planes (p1 p2 : Plane3D) : Prop := sorry
def line_in_plane (l : Line3D) (p : Plane3D) : Prop := sorry
def coplanar (l1 l2 : Line3D) : Prop := sorry

-- Define the propositions
def proposition1 : Prop :=
  ∀ (p1 p2 p3 : Plane3D), perpendicular_planes p1 p3 → perpendicular_planes p2 p3 → parallel_planes p1 p2

def proposition2 : Prop :=
  ∀ (l m : Line3D) (α : Plane3D), l ≠ m → perpendicular l α → parallel_lines l m → perpendicular m α

def proposition3 : Prop :=
  ∀ (α β : Plane3D) (m : Line3D), α ≠ β → line_in_plane m α →
    (perpendicular_planes α β ↔ perpendicular m β)

def proposition4 : Prop :=
  ∀ (a b : Line3D) (P : Point3D), ¬coplanar a b →
    ∃ (p : Plane3D), (perpendicular a p ∧ ∃ (l : Line3D), parallel_lines b l ∧ l.direction = p.normal) ∨
                     (perpendicular b p ∧ ∃ (l : Line3D), parallel_lines a l ∧ l.direction = p.normal)

-- The main theorem
theorem exactly_one_correct_proposition :
  (proposition1 = False) ∧
  (proposition2 = True) ∧
  (proposition3 = False) ∧
  (proposition4 = False) := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_exactly_one_correct_proposition_l1046_104694


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_l1046_104692

-- Define the piecewise function f
noncomputable def f (x : ℝ) : ℝ := 
  if x ≥ 0 then x^2 + x else -x^2 + x

-- State the theorem
theorem solution_set (x : ℝ) : 
  f (x^2 - x + 1) < 12 ↔ -1 < x ∧ x < 2 := by
  sorry

-- Note: The proof is omitted as per the instructions

end NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_l1046_104692


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sale_loss_percentage_l1046_104677

noncomputable def article_sale (initial_quantity : ℕ) (initial_price : ℚ) (initial_gain_percentage : ℚ)
                 (final_quantity : ℕ) (final_price : ℚ) : ℚ :=
  let initial_cost := initial_price / (1 + initial_gain_percentage)
  let cost_per_article := initial_cost / initial_quantity
  let final_cost := cost_per_article * final_quantity
  let loss := final_cost - final_price
  (loss / final_cost) * 100

theorem sale_loss_percentage :
  article_sale 20 60 (1/5) 35 70 = 20 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sale_loss_percentage_l1046_104677


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_soda_ratio_l1046_104683

theorem soda_ratio : 
  ∀ (total_cans cherry_cans : ℕ),
  total_cans = 24 →
  cherry_cans = 8 →
  let orange_cans := total_cans - cherry_cans
  orange_cans / cherry_cans = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_soda_ratio_l1046_104683


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_factorial_of_product_factorial_equality_l1046_104638

theorem factorial_of_product (a b c d e : ℕ) :
  Nat.factorial (a^2 * b * c^2 * d) = Nat.factorial ((a * a) * b * (c * c) * d) :=
by sorry

theorem factorial_equality : Nat.factorial (3^2 * 4 * 6^2 * 5) = Nat.factorial 6480 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_factorial_of_product_factorial_equality_l1046_104638


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_plan1_more_cost_effective_when_x_lt_100_l1046_104612

/-- Represents the cost of a purchase plan -/
structure PlanCost where
  suits : ℕ
  ties : ℕ
  cost : ℝ

/-- Calculates the cost of Plan 1 -/
def plan1Cost (suits ties : ℕ) : ℝ :=
  200 * (suits : ℝ) + 40 * ((ties : ℝ) - (suits : ℝ))

/-- Calculates the cost of Plan 2 -/
def plan2Cost (suits ties : ℕ) : ℝ :=
  0.9 * (200 * (suits : ℝ) + 40 * (ties : ℝ))

/-- Theorem stating that Plan 1 is more cost-effective than Plan 2 when x < 100 -/
theorem plan1_more_cost_effective_when_x_lt_100 (x : ℕ) (h1 : x > 20) (h2 : x < 100) :
  plan1Cost 20 x < plan2Cost 20 x := by
  sorry

#eval plan1Cost 20 30
#eval plan2Cost 20 30

end NUMINAMATH_CALUDE_ERRORFEEDBACK_plan1_more_cost_effective_when_x_lt_100_l1046_104612


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_different_color_probability_l1046_104628

def total_balls : ℕ := 7
def white_balls : ℕ := 4
def red_balls : ℕ := 2
def yellow_balls : ℕ := 1

/-- The probability of drawing two balls of different colors from a set of 7 balls 
    (4 white, 2 red, 1 yellow) is equal to 2/3. -/
theorem different_color_probability : 
  (Nat.choose white_balls 1 * Nat.choose red_balls 1 + 
   Nat.choose white_balls 1 * Nat.choose yellow_balls 1 + 
   Nat.choose red_balls 1 * Nat.choose yellow_balls 1 : ℚ) / Nat.choose total_balls 2 = 2 / 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_different_color_probability_l1046_104628


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_order_of_magnitude_l1046_104607

-- Define the constants
noncomputable def a : ℝ := 2^(11/10)
noncomputable def b : ℝ := 3^(6/10)
noncomputable def c : ℝ := Real.log 3 / Real.log 2

-- Theorem statement
theorem order_of_magnitude : a > b ∧ b > c := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_order_of_magnitude_l1046_104607


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_g_monotone_decreasing_l1046_104659

def f (x : ℝ) : ℝ := (x - 2)^2

def g (x : ℝ) : ℝ := f (x + 1)

theorem g_monotone_decreasing :
  ∀ x y, x ∈ Set.Icc (-2 : ℝ) (2 : ℝ) → y ∈ Set.Icc (-2 : ℝ) (2 : ℝ) → x < y → g y < g x :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_g_monotone_decreasing_l1046_104659


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_more_than_6_eq_793_1024_l1046_104634

/-- The number of grandchildren Mr. Lee has -/
def num_grandchildren : ℕ := 12

/-- The probability of a grandchild being a grandson -/
def prob_grandson : ℚ := 1/2

/-- The probability of having exactly k grandsons out of n grandchildren -/
def prob_exactly_k_grandsons (n k : ℕ) : ℚ :=
  (Nat.choose n k : ℚ) * prob_grandson^k * (1 - prob_grandson)^(n-k)

/-- The probability of having more than 6 grandsons or more than 6 granddaughters -/
def prob_more_than_6 : ℚ :=
  1 - prob_exactly_k_grandsons num_grandchildren 6

theorem prob_more_than_6_eq_793_1024 : 
  prob_more_than_6 = 793/1024 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_more_than_6_eq_793_1024_l1046_104634


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_problem_l1046_104697

/-- Triangle ABC with side lengths a, b, c opposite to angles A, B, C respectively -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- The given condition in the problem -/
def condition (t : Triangle) : Prop :=
  (t.c - t.b) * Real.sin t.C = (t.a - t.b) * (Real.sin t.A + Real.sin t.B)

/-- Triangle ABC is acute -/
def is_acute (t : Triangle) : Prop :=
  0 < t.A ∧ t.A < Real.pi/2 ∧
  0 < t.B ∧ t.B < Real.pi/2 ∧
  0 < t.C ∧ t.C < Real.pi/2

/-- The area of triangle ABC -/
noncomputable def area (t : Triangle) : ℝ :=
  1/2 * t.b * t.c * Real.sin t.A

theorem triangle_problem (t : Triangle) 
  (h1 : condition t) 
  (h2 : t.b = 2) 
  (h3 : is_acute t) : 
  t.A = Real.pi/3 ∧ Real.sqrt 3/2 < area t ∧ area t < 2*Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_problem_l1046_104697


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_existence_of_three_points_l1046_104640

/-- Given n ≥ 3 points on a plane, there exist three points A, B, and C
    such that 1 ≤ AB/AC < (n+1)/(n-1), where AB and AC are distances. -/
theorem existence_of_three_points (n : ℕ) (hn : n ≥ 3) 
  (points : Fin n → ℝ × ℝ) : 
  ∃ (i j k : Fin n), 
    let A := points i
    let B := points j
    let C := points k
    let AB := Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2)
    let AC := Real.sqrt ((A.1 - C.1)^2 + (A.2 - C.2)^2)
    1 ≤ AB / AC ∧ AB / AC < (n + 1 : ℝ) / (n - 1 : ℝ) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_existence_of_three_points_l1046_104640


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_dot_product_implies_sin_value_l1046_104618

noncomputable def a (x : ℝ) : ℝ × ℝ := (2 * Real.sin (x - Real.pi/4), Real.sqrt 3 * Real.sin x)

noncomputable def b (x : ℝ) : ℝ × ℝ := (Real.sin (x + Real.pi/4), 2 * Real.cos x)

noncomputable def f (x : ℝ) : ℝ := (a x).1 * (b x).1 + (a x).2 * (b x).2

theorem dot_product_implies_sin_value (α : ℝ) :
  f (α/2) = 2/5 → Real.sin (2*α + Real.pi/6) = 23/25 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_dot_product_implies_sin_value_l1046_104618


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_batting_average_problem_l1046_104615

theorem batting_average_problem (current_average : ℚ) (next_match_runs : ℕ) (new_average : ℚ) :
  current_average = 52 →
  next_match_runs = 78 →
  new_average = 54 →
  ∃ n : ℕ, n > 0 ∧ 
    (current_average * n + next_match_runs) / (n + 1 : ℚ) = new_average →
    n = 12 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_batting_average_problem_l1046_104615


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_more_red_polygons_l1046_104670

/-- Represents a point on the circle -/
inductive Point
| White
| Red

/-- The total number of points on the circle -/
def total_points : Nat := 1998

/-- The number of white points on the circle -/
def white_points : Nat := 1997

/-- The number of red points on the circle -/
def red_points : Nat := 1

/-- A polygon is represented as a set of points -/
def Polygon := Finset Point

/-- The set of all possible polygons -/
def all_polygons : Finset Polygon := sorry

/-- The set of polygons without the red point -/
def white_polygons : Finset Polygon := sorry

/-- The set of polygons with the red point -/
def red_polygons : Finset Polygon := sorry

theorem more_red_polygons :
  Finset.card red_polygons > Finset.card white_polygons := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_more_red_polygons_l1046_104670


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_valid_arrangements_count_l1046_104614

def digits : List ℕ := [1, 2, 2, 5]

def is_valid_arrangement (arr : List ℕ) : Bool :=
  arr.length = 4 && arr.getLast? = some 5 && arr.toFinset = digits.toFinset

def count_valid_arrangements : ℕ :=
  (List.permutations digits).filter is_valid_arrangement |>.length

theorem valid_arrangements_count : count_valid_arrangements = 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_valid_arrangements_count_l1046_104614


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_subset_partition_theorem_l1046_104698

theorem subset_partition_theorem (X : Type) [Finite X] (n : ℕ) (A : Fin n → Set X)
  (h_nonempty : ∀ i, Set.Nonempty (A i))
  (h_intersection : ∀ i j, i ≤ j → ¬ Set.Subsingleton (A i ∩ A j)) :
  ∃ (S T : Set X), S ∪ T = Set.univ ∧ S ∩ T = ∅ ∧ S ≠ ∅ ∧ T ≠ ∅ ∧
    ∀ i, (A i ∩ S).Nonempty ∧ (A i ∩ T).Nonempty :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_subset_partition_theorem_l1046_104698


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_product_remainder_l1046_104635

theorem product_remainder (a b c : ℕ) 
  (ha : a % 5 = 2) 
  (hb : b % 5 = 3) 
  (hc : c % 5 = 4) : 
  (a * b * c) % 5 = 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_product_remainder_l1046_104635


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_secret_code_sum_exists_unique_l1046_104602

def digit := Fin 10

structure CodeAssignment where
  F : digit
  L : digit
  Y : digit
  R : digit
  U : digit
  I : digit
  O : digit

def value (a : CodeAssignment) (word : List Char) : ℕ :=
  word.foldl (λ acc c => 10 * acc + 
    match c with
    | 'F' => a.F.val
    | 'L' => a.L.val
    | 'Y' => a.Y.val
    | 'R' => a.R.val
    | 'U' => a.U.val
    | 'I' => a.I.val
    | 'O' => a.O.val
    | _ => 0
  ) 0

theorem secret_code_sum_exists_unique :
  ∃! a : CodeAssignment, 
    a.I = ⟨1, by norm_num⟩ ∧ 
    a.O = ⟨0, by norm_num⟩ ∧
    value a ['F', 'L', 'Y'] + value a ['F', 'O', 'R'] + value a ['Y', 'O', 'U', 'R'] = 
    value a ['L', 'I', 'F', 'E'] :=
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_secret_code_sum_exists_unique_l1046_104602


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cubic_equation_solution_l1046_104610

theorem cubic_equation_solution 
  (a b c d : ℝ) 
  (h1 : a * d = b * c) 
  (h2 : a * d ≠ 0) 
  (h3 : b * d < 0) : 
  ∃ x1 x23 : ℝ, x1 = -b / a ∧ x23 = Real.sqrt (-d / b) ∧
  (∀ x : ℝ, a * x^3 + b * x^2 + c * x + d = 0 ↔ x = x1 ∨ x = x23 ∨ x = -x23) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cubic_equation_solution_l1046_104610


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_speed_conversion_theorem_l1046_104652

/-- Converts kilometers per hour to meters per second -/
noncomputable def km_per_hour_to_m_per_s (speed_km_h : ℝ) : ℝ :=
  speed_km_h * (1000 / 3600)

/-- The given speed in kilometers per hour -/
def given_speed : ℝ := 1.7

/-- The theorem stating that the given speed is approximately equal to 0.4722 m/s -/
theorem speed_conversion_theorem :
  ∃ ε > 0, |km_per_hour_to_m_per_s given_speed - 0.4722| < ε := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_speed_conversion_theorem_l1046_104652


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_surface_area_of_prism_with_hexagon_cross_section_l1046_104654

/-- Represents a rectangular prism with dimensions in the ratio 4:3:2 -/
structure RectangularPrism where
  k : ℝ
  length : ℝ := 4 * k
  width : ℝ := 3 * k
  height : ℝ := 2 * k

/-- Calculates the surface area of a rectangular prism -/
def surface_area (prism : RectangularPrism) : ℝ :=
  2 * (prism.length * prism.width + prism.length * prism.height + prism.width * prism.height)

/-- Calculates the minimum perimeter of a hexagonal cross-section -/
def min_hexagon_perimeter (prism : RectangularPrism) : ℝ :=
  2 * (prism.length + prism.width + prism.height)

theorem surface_area_of_prism_with_hexagon_cross_section :
  ∀ (prism : RectangularPrism), 
  prism.k > 0 →
  min_hexagon_perimeter prism = 36 →
  surface_area prism = 208 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_surface_area_of_prism_with_hexagon_cross_section_l1046_104654


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_sum_and_mean_l1046_104631

def arithmetic_sequence (a₁ : ℝ) (d : ℝ) (n : ℕ) : ℕ → ℝ :=
  λ i => a₁ + d * ((i : ℝ) - 1)

theorem arithmetic_sequence_sum_and_mean :
  let a₁ : ℝ := 1
  let d : ℝ := 2
  let n : ℕ := 11
  let seq := arithmetic_sequence a₁ d n
  let sum := (Finset.range n).sum seq
  let mean := sum / n
  sum = 121 ∧ mean = 11 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_sum_and_mean_l1046_104631


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sine_cosine_relation_l1046_104690

theorem sine_cosine_relation (α : ℝ) : 
  Real.cos (π / 4 - α) = 3 / 5 → Real.sin (3 * π / 4 - α) = 3 / 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sine_cosine_relation_l1046_104690


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_not_divides_wobbly_iff_multiple_of_10_or_25_l1046_104675

/-- A wobbly number is a positive integer whose digits in base 10 are alternately nonzero and zero, with the units digit being nonzero. -/
def IsWobbly (n : ℕ) : Prop :=
  n > 0 ∧ 
  ∃ (digits : List ℕ), 
    n.digits 10 = digits ∧ 
    digits.length > 1 ∧
    digits.head? ≠ some 0 ∧
    digits.getLast? ≠ some 0 ∧
    ∀ i, i + 1 < digits.length → 
      (digits[i]? = some 0 ↔ digits[i+1]? ≠ some 0) ∧
      (digits[i]? ≠ some 0 ↔ digits[i+1]? = some 0)

/-- A positive integer does not divide any wobbly number if and only if it is a multiple of 10 or 25. -/
theorem not_divides_wobbly_iff_multiple_of_10_or_25 (n : ℕ) (hn : n > 0) :
  (∀ m : ℕ, IsWobbly m → ¬(n ∣ m)) ↔ (∃ k : ℕ, n = 10 * k ∨ n = 25 * k) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_not_divides_wobbly_iff_multiple_of_10_or_25_l1046_104675


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_sine_ratio_inequality_l1046_104687

theorem triangle_sine_ratio_inequality (α β γ : ℝ) (h_triangle : α + β + γ = Real.pi) 
  (h_positive : 0 < α ∧ 0 < β ∧ 0 < γ) :
  |Real.sin α / Real.sin β + Real.sin β / Real.sin γ + Real.sin γ / Real.sin α - 
   (Real.sin β / Real.sin α + Real.sin γ / Real.sin β + Real.sin α / Real.sin γ)| < 1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_sine_ratio_inequality_l1046_104687


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_shift_symmetry_l1046_104648

open Real

-- Define the original function
noncomputable def f (x : ℝ) : ℝ := 3 * sin (2 * x + π / 4)

-- Define the shifted function
noncomputable def g (x φ : ℝ) : ℝ := f (x + φ)

-- Theorem statement
theorem shift_symmetry (φ : ℝ) 
  (h1 : 0 < φ) (h2 : φ < π / 2) 
  (h3 : ∀ x, g x φ = -g (-x) φ) : 
  φ = 3 * π / 8 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_shift_symmetry_l1046_104648


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_intersection_sequences_l1046_104656

noncomputable def f (x : ℝ) := Real.exp x

noncomputable def tangent_line (n : ℕ) (x : ℝ) : ℝ := f n + f n * (x - n)

noncomputable def A (n : ℕ) : ℝ := n + 1 / (Real.exp 1 - 1)

noncomputable def B (n : ℕ) : ℝ := (Real.exp (n + 1)) / (Real.exp 1 - 1)

theorem tangent_intersection_sequences :
  (∃ (a d : ℝ), ∀ n : ℕ, A n = a + n * d) ∧
  (∃ (b r : ℝ), ∀ n : ℕ, B n = b * r^n) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_intersection_sequences_l1046_104656


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_coefficient_x_squared_in_expansion_l1046_104686

/-- The coefficient of x^2 in the expansion of (x + 1/x + 2)^5 -/
def coefficient_x_squared : ℕ := 120

/-- The expansion of (x + 1/x + 2)^5 -/
noncomputable def expansion (x : ℝ) : ℝ := (x + 1/x + 2)^5

/-- Theorem stating the existence of the x^2 term in the expansion -/
theorem coefficient_x_squared_in_expansion :
  ∃ (f : ℝ → ℝ) (c : ℝ), ∀ x, x ≠ 0 →
    expansion x = f x + coefficient_x_squared * x^2 + c * x^2 * (x + 1/x) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_coefficient_x_squared_in_expansion_l1046_104686


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tetrahedron_projection_is_circumcenter_l1046_104619

/-- Represents a point in 3D space -/
structure Point where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Represents a plane in 3D space -/
structure Plane where
  a : ℝ
  b : ℝ
  c : ℝ
  d : ℝ

/-- Represents a tetrahedron with vertex P and base ABC -/
structure Tetrahedron where
  P : Point
  A : Point
  B : Point
  C : Point

/-- Represents a triangle with vertices A, B, and C -/
structure Triangle where
  A : Point
  B : Point
  C : Point

/-- The distance between two points -/
def Point.dist (p1 p2 : Point) : ℝ :=
  sorry

/-- The projection of a point onto a plane -/
def projection (P : Point) (plane : Plane) : Point :=
  sorry

/-- The circumcenter of a triangle -/
def circumcenter (t : Triangle) : Point :=
  sorry

/-- The plane defined by three points -/
def plane_of_points (A B C : Point) : Plane :=
  sorry

theorem tetrahedron_projection_is_circumcenter 
  (t : Tetrahedron) 
  (h : t.P.dist t.A = t.P.dist t.B ∧ t.P.dist t.B = t.P.dist t.C) :
  projection t.P (plane_of_points t.A t.B t.C) = 
    circumcenter {A := t.A, B := t.B, C := t.C} := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tetrahedron_projection_is_circumcenter_l1046_104619


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_polygon_coloring_l1046_104669

/-- Represents the number of valid colorings for an n-sided polygon --/
def T (n : ℕ) : ℤ :=
  2^n - 2 * ((-1 : ℤ)^n)

/-- The number of sides in our polygon --/
def sides : ℕ := 2003

/-- The number of colors available --/
def colors : ℕ := 3

theorem polygon_coloring :
  T sides = 2^sides - 2 :=
by
  -- Unfold the definition of T
  unfold T
  -- Simplify the expression
  simp
  -- The rest of the proof would go here
  sorry

#eval T sides

end NUMINAMATH_CALUDE_ERRORFEEDBACK_polygon_coloring_l1046_104669


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_squares_products_eq_factorial_minus_one_l1046_104672

/-- Define a function that represents the sum of squares of products of numbers
    in sets from {1, 2, ..., n} that do not contain consecutive numbers -/
def sum_squares_products (n : ℕ) : ℕ :=
  sorry

/-- The main theorem stating that the sum of squares of products is equal to (n+1)! - 1 -/
theorem sum_squares_products_eq_factorial_minus_one (n : ℕ) :
  sum_squares_products n = (Nat.factorial (n + 1)) - 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_squares_products_eq_factorial_minus_one_l1046_104672


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_2009_equals_2_l1046_104658

def sequence_a : ℕ → ℚ
  | 0 => 1/2  -- Add this case for 0
  | 1 => 1/2
  | n + 1 => 1 / (1 - sequence_a n)

theorem a_2009_equals_2 : sequence_a 2009 = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_2009_equals_2_l1046_104658


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_divisors_with_remainder_l1046_104679

theorem max_divisors_with_remainder (n : ℕ) (r : ℕ) : 
  (∃ (m : ℕ), m > 0 ∧ 
    (∀ (d : ℕ), d ∈ (Finset.filter (λ x : ℕ ↦ x > r ∧ (n - r) % x = 0) (Finset.range (n - r))) → 
      n % d = r) ∧
    (∀ (k : ℕ), k > m → 
      ¬(∀ (d : ℕ), d ∈ (Finset.filter (λ x : ℕ ↦ x > r ∧ (n - r) % x = 0) (Finset.range (n - r))) → 
        n % d = r))) →
  (∃ (max_m : ℕ), max_m = 11) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_divisors_with_remainder_l1046_104679


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_candy_flavors_l1046_104632

/-- Represents a ratio of orange to purple candies -/
structure CandyRatio where
  orange : Nat
  purple : Nat
deriving BEq, Repr

/-- Checks if a CandyRatio is valid (not 0:0) -/
def CandyRatio.isValid (r : CandyRatio) : Bool :=
  r.orange ≠ 0 || r.purple ≠ 0

/-- Simplifies a CandyRatio to its lowest terms -/
def CandyRatio.simplify (r : CandyRatio) : CandyRatio :=
  let gcd := Nat.gcd r.orange r.purple
  { orange := r.orange / gcd, purple := r.purple / gcd }

/-- Generates all possible CandyRatios given the maximum number of orange and purple candies -/
def generateAllRatios (maxOrange maxPurple : Nat) : List CandyRatio :=
  List.map (fun (p : Nat × Nat) => { orange := p.1, purple := p.2 })
    (List.product (List.range (maxOrange + 1)) (List.range (maxPurple + 1)))

/-- Counts the number of unique valid simplified ratios -/
def countUniqueRatios (ratios : List CandyRatio) : Nat :=
  (ratios.filter CandyRatio.isValid).map CandyRatio.simplify |>.eraseDups |>.length

theorem unique_candy_flavors :
  countUniqueRatios (generateAllRatios 6 4) = 14 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_candy_flavors_l1046_104632


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_log_equation_a_l1046_104633

theorem log_equation_a (x : ℝ) :
  x > 0 ∧ x ≠ 1 →
  (0.2 * (Real.log (1/32) / Real.log x) = -0.5) ↔ x = 4 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_log_equation_a_l1046_104633


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l1046_104604

noncomputable def f (x : ℝ) := Real.sin (2 * x) + Real.cos (2 * x)

theorem f_properties : 
  (∃ (p : ℝ), p > 0 ∧ (∀ (x : ℝ), f (x + p) = f x) ∧ 
    (∀ (q : ℝ), q > 0 ∧ (∀ (x : ℝ), f (x + q) = f x) → q ≥ p)) ∧
  (∀ (x : ℝ), f (x - π/4) = f (-x)) ∧
  (∀ (x : ℝ), x ∈ Set.Ioo (3*π/8) (π/2) → 
    ∃ (h : ℝ), h > 0 ∧ x + h ∈ Set.Ioo (3*π/8) (π/2) ∧ f (x + h) < f x) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l1046_104604


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_consecutive_even_numbers_sum_l1046_104644

theorem consecutive_even_numbers_sum (a b c d : ℕ) : 
  (a % 2 = 0 ∧ b = a + 2 ∧ c = b + 2 ∧ d = c + 2) →
  (a + b + c + d = 3924) →
  (a = 978 ∧ b = 980 ∧ c = 982 ∧ d = 984) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_consecutive_even_numbers_sum_l1046_104644


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hexagon_in_rectangle_sum_l1046_104603

/-- Regular hexagon inscribed in a rectangle -/
structure HexagonInRectangle where
  /-- Side length of the regular hexagon -/
  side_length : ℝ
  /-- Acute angle between hexagon side and rectangle diagonal -/
  θ : ℝ

/-- Properties of the HexagonInRectangle -/
axiom hex_rect_properties (h : HexagonInRectangle) :
  0 < h.side_length ∧ 0 < h.θ ∧ h.θ < Real.pi / 2

/-- The sine squared of θ is a rational number m/n -/
axiom sin_squared_rational (h : HexagonInRectangle) :
  ∃ (m n : ℕ), Nat.Coprime m n ∧ Real.sin h.θ ^ 2 = m / n

/-- Theorem: For a regular hexagon inscribed in a rectangle, m + n = 55 -/
theorem hexagon_in_rectangle_sum (h : HexagonInRectangle) :
  ∃ (m n : ℕ), Nat.Coprime m n ∧ Real.sin h.θ ^ 2 = m / n ∧ m + n = 55 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hexagon_in_rectangle_sum_l1046_104603


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_total_votes_in_election_l1046_104695

/-- Theorem: Total votes in an election with specific conditions -/
theorem total_votes_in_election 
  (x_percentage : Real) 
  (y_margin : ℕ) 
  (invalid_votes : ℕ) 
  (undecided_percentage : Real) : 
  x_percentage = 0.40 →
  y_margin = 3000 →
  invalid_votes = 1000 →
  undecided_percentage = 0.02 →
  (16320 : ℕ) = 
    let valid_votes := (y_margin : Real) / (1 - 2 * x_percentage)
    let total_with_invalid := valid_votes + invalid_votes
    ⌈(total_with_invalid : Real) / (1 - undecided_percentage)⌉ := by
  sorry

#check total_votes_in_election

end NUMINAMATH_CALUDE_ERRORFEEDBACK_total_votes_in_election_l1046_104695


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_interval_of_increase_minimum_value_l1046_104678

open Real

-- Part 1
noncomputable def f (x : ℝ) : ℝ := 2 * cos x * sin (x + π/6)

theorem interval_of_increase (k : ℤ) :
  StrictMonoOn f (Set.Icc (-π/3 + k*π) (π/6 + k*π)) := by sorry

-- Part 2
noncomputable def y (x : ℝ) : ℝ := 3 * (cos x)^2 - 4 * cos x + 1

theorem minimum_value :
  ∃ (x : ℝ), x ∈ Set.Icc 0 (π/2) ∧ 
  y x = -1/3 ∧ 
  ∀ (z : ℝ), z ∈ Set.Icc 0 (π/2) → y z ≥ -1/3 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_interval_of_increase_minimum_value_l1046_104678


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_line_equation_l1046_104674

/-- Given a line l: x + 2y + 1 = 0 and a point P(1, -2),
    prove that the line passing through P and perpendicular to l
    has the equation 2x - y = 4 -/
theorem perpendicular_line_equation (l : Set (ℝ × ℝ)) (P : ℝ × ℝ) :
  l = {(x, y) | x + 2*y + 1 = 0} →
  P = (1, -2) →
  ∃ (m : ℝ), (∀ (p q : ℝ × ℝ), p ∈ l ∧ q ∈ l → (p.1 - q.1) * (p.1 - q.1) + (p.2 - q.2) * (2*p.1 - p.2 - (2*q.1 - q.2)) = 0) ∧
              P ∈ {(x, y) | 2*x - y = 4} :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_line_equation_l1046_104674


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_students_taking_statistics_l1046_104699

theorem students_taking_statistics 
  (total_students : ℕ) 
  (history_students : ℕ) 
  (history_or_statistics : ℕ) 
  (history_not_statistics : ℕ) 
  (h1 : total_students = 90)
  (h2 : history_students = 36)
  (h3 : history_or_statistics = 57)
  (h4 : history_not_statistics = 25)
  : history_or_statistics - (history_students - history_not_statistics) = 32 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_students_taking_statistics_l1046_104699


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_chubby_checkerboard_black_squares_l1046_104666

/-- Represents a square on the checkerboard -/
inductive Square
| Black
| Red

/-- Represents a checkerboard -/
def Checkerboard (n : Nat) := Matrix (Fin n) (Fin n) Square

/-- Creates a checkerboard with the given dimensions and pattern -/
def create_checkerboard (n : Nat) : Checkerboard n :=
  sorry

/-- Counts the number of black squares on the checkerboard -/
def count_black_squares {n : Nat} (board : Checkerboard n) : Nat :=
  sorry

theorem chubby_checkerboard_black_squares :
  let board := create_checkerboard 35
  count_black_squares board = 613 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_chubby_checkerboard_black_squares_l1046_104666


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rationalizing_factor_and_simplification_l1046_104676

theorem rationalizing_factor_and_simplification :
  (∃ (x : ℝ), x * (Real.sqrt 2 - 1) = (Real.sqrt 2 + 1) * (Real.sqrt 2 - 1)) ∧
  (3 / (3 - Real.sqrt 6) = 3 + Real.sqrt 6) ∧
  (Real.sqrt 2019 - Real.sqrt 2018 < Real.sqrt 2018 - Real.sqrt 2017) ∧
  ((Finset.sum (Finset.range 2023) (fun i => 1 / (Real.sqrt (i + 2) + Real.sqrt (i + 1)))) * (Real.sqrt 2024 + 1) = 2023) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rationalizing_factor_and_simplification_l1046_104676


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_student_running_distance_l1046_104621

noncomputable def arithmetic_sequence_sum (a₁ : ℝ) (d : ℝ) (n : ℕ) : ℝ :=
  n * a₁ + (n * (n - 1) / 2) * d

theorem student_running_distance :
  let initial_distance : ℝ := 5000
  let daily_increase : ℝ := 200
  let days : ℕ := 7
  arithmetic_sequence_sum initial_distance daily_increase days = 39200 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_student_running_distance_l1046_104621


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_minor_premise_correct_syllogism_valid_l1046_104688

-- Define the exponential function
noncomputable def exponential (a : ℝ) (x : ℝ) : ℝ := a^x

-- Define the property of being an increasing function
def is_increasing (f : ℝ → ℝ) : Prop := ∀ x y, x < y → f x < f y

-- Define the syllogism structure
structure Syllogism :=
  (major_premise : Prop)
  (minor_premise : Prop)
  (conclusion : Prop)

-- Define our specific syllogism
def our_syllogism : Syllogism :=
  { major_premise := ∀ a, a > 1 → is_increasing (exponential a),
    minor_premise := ∃ a, a > 1 ∧ exponential a = exponential 2,
    conclusion := is_increasing (exponential 2) }

-- Theorem statement
theorem minor_premise_correct :
  our_syllogism.minor_premise = (∃ a, a > 1 ∧ exponential a = exponential 2) :=
by rfl

-- Additional theorem to demonstrate the syllogism
theorem syllogism_valid (h1 : our_syllogism.major_premise) 
                        (h2 : our_syllogism.minor_premise) : 
  our_syllogism.conclusion :=
by
  sorry  -- The actual proof would go here, but we're using sorry for now

end NUMINAMATH_CALUDE_ERRORFEEDBACK_minor_premise_correct_syllogism_valid_l1046_104688


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_not_good_curve_l1046_104642

-- Define a point in 2D space
structure Point2D where
  x : ℝ
  y : ℝ

-- Define a function to calculate the distance between two points
noncomputable def distance (p1 p2 : Point2D) : ℝ :=
  Real.sqrt ((p1.x - p2.x)^2 + (p1.y - p2.y)^2)

-- Define what it means for a curve to be a "good curve"
def isGoodCurve (C : Point2D → Prop) : Prop :=
  ∃ (M : Point2D), C M ∧ 
    |distance M ⟨-5, 0⟩ - distance M ⟨5, 0⟩| = 8

-- Define the circle x^2 + y^2 = 9
def circleEquation (p : Point2D) : Prop :=
  p.x^2 + p.y^2 = 9

-- Theorem stating that the circle is not a good curve
theorem circle_not_good_curve : ¬(isGoodCurve circleEquation) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_not_good_curve_l1046_104642


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_chord_length_is_sqrt_10_l1046_104643

-- Define the line
noncomputable def line_x (t : ℝ) : ℝ := 2 + t
noncomputable def line_y (t : ℝ) : ℝ := Real.sqrt 3 * t

-- Define the curve
def curve (x y : ℝ) : Prop := x^2 - y^2 = 1

-- Define the chord length
noncomputable def chord_length (t₁ t₂ : ℝ) : ℝ := Real.sqrt ((t₁ - t₂)^2)

theorem chord_length_is_sqrt_10 :
  ∃ t₁ t₂ : ℝ, 
    curve (line_x t₁) (line_y t₁) ∧ 
    curve (line_x t₂) (line_y t₂) ∧ 
    chord_length t₁ t₂ = Real.sqrt 10 := by
  sorry

#check chord_length_is_sqrt_10

end NUMINAMATH_CALUDE_ERRORFEEDBACK_chord_length_is_sqrt_10_l1046_104643


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_set_satisfies_conditions_l1046_104660

open Real

/-- The set of angles where the initial side coincides with the positive x-axis
    and the terminal side forms an angle of π/6 with the y-axis -/
def angle_set : Set ℝ :=
  {α | ∃ k : ℤ, α = k * Real.pi + Real.pi / 3 ∨ α = k * Real.pi - Real.pi / 3}

/-- The condition that the initial side coincides with the positive x-axis -/
def initial_side_condition (α : ℝ) : Prop :=
  ∃ k : ℤ, α = 2 * Real.pi * k

/-- The condition that the terminal side forms an angle of π/6 with the y-axis -/
def terminal_side_condition (α : ℝ) : Prop :=
  ∃ k : ℤ, α = k * Real.pi + Real.pi / 2 - Real.pi / 6 ∨ α = k * Real.pi + Real.pi / 2 + Real.pi / 6

/-- Theorem stating that the angle_set satisfies both conditions -/
theorem angle_set_satisfies_conditions :
  ∀ α : ℝ, α ∈ angle_set ↔ initial_side_condition α ∧ terminal_side_condition α :=
by
  sorry

#check angle_set_satisfies_conditions

end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_set_satisfies_conditions_l1046_104660


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_midpoint_of_intersection_l1046_104685

-- Define the line equations
noncomputable def line_x (t : ℝ) : ℝ := 1 + (1/2) * t
noncomputable def line_y (t : ℝ) : ℝ := -3 * Real.sqrt 3 + (Real.sqrt 3 / 2) * t

-- Define the circle equation
def circle_eq (x y : ℝ) : Prop := x^2 + y^2 = 16

-- Define the intersection points
def intersection_points : Set (ℝ × ℝ) :=
  {p | ∃ t, p.1 = line_x t ∧ p.2 = line_y t ∧ circle_eq p.1 p.2}

-- Theorem statement
theorem midpoint_of_intersection :
  ∃ A B : ℝ × ℝ, A ∈ intersection_points ∧ B ∈ intersection_points ∧
  A ≠ B ∧ ((A.1 + B.1) / 2 = 3 ∧ (A.2 + B.2) / 2 = -Real.sqrt 3) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_midpoint_of_intersection_l1046_104685


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_neither_sufficient_nor_necessary_l1046_104689

theorem neither_sufficient_nor_necessary : ∃ (x₁ x₂ : ℝ),
  (x₁ < π / 4 ∧ Real.tan x₁ ≥ 1) ∧ (x₂ ≥ π / 4 ∧ Real.tan x₂ < 1) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_neither_sufficient_nor_necessary_l1046_104689


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_data_set_range_l1046_104681

def data_set : List Int := [0, -1, 3, 2, 4]

def range (l : List Int) : Int :=
  match l.maximum?, l.minimum? with
  | some max, some min => max - min
  | _, _ => 0

theorem data_set_range : range data_set = 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_data_set_range_l1046_104681


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_bryden_payment_is_37_5_l1046_104671

/-- The amount a collector pays for state quarters, as a percentage of face value -/
def collector_rate : ℚ := 3000

/-- The number of state quarters Bryden has -/
def bryden_quarters : ℕ := 5

/-- The face value of a single state quarter in dollars -/
def quarter_value : ℚ := 1/4

/-- The amount Bryden receives for his state quarters -/
def bryden_payment : ℚ := (collector_rate / 100) * (bryden_quarters : ℚ) * quarter_value

theorem bryden_payment_is_37_5 : bryden_payment = 75/2 := by
  rw [bryden_payment, collector_rate, bryden_quarters, quarter_value]
  norm_num

#eval bryden_payment

end NUMINAMATH_CALUDE_ERRORFEEDBACK_bryden_payment_is_37_5_l1046_104671


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_midpoint_construction_l1046_104625

-- Define parallel lines
structure ParallelLines where
  line1 : Set (ℝ × ℝ)
  line2 : Set (ℝ × ℝ)
  parallel : ∀ (x y : ℝ × ℝ), x ∈ line1 → y ∈ line2 → (x.1 - y.1) * (x.2 - y.2) = 0

-- Define a segment on a line
structure Segment where
  line : Set (ℝ × ℝ)
  endpoint1 : ℝ × ℝ
  endpoint2 : ℝ × ℝ
  on_line : endpoint1 ∈ line ∧ endpoint2 ∈ line

-- Define a trapezoid
structure Trapezoid where
  base1 : Segment
  base2 : Segment
  leg1 : Segment
  leg2 : Segment
  is_trapezoid : ∀ (x y : ℝ × ℝ), x ∈ base1.line → y ∈ base2.line → (x.1 - y.1) * (x.2 - y.2) = 0

-- Define a midsegment of a trapezoid
noncomputable def midsegment (t : Trapezoid) : Segment :=
  { line := { p : ℝ × ℝ | ∃ (q r : ℝ × ℝ), q ∈ t.leg1.line ∧ r ∈ t.leg2.line ∧ p = ((q.1 + r.1) / 2, (q.2 + r.2) / 2) },
    endpoint1 := ((t.leg1.endpoint1.1 + t.leg1.endpoint2.1) / 2, (t.leg1.endpoint1.2 + t.leg1.endpoint2.2) / 2),
    endpoint2 := ((t.leg2.endpoint1.1 + t.leg2.endpoint2.1) / 2, (t.leg2.endpoint1.2 + t.leg2.endpoint2.2) / 2),
    on_line := by sorry }

-- Theorem statement
theorem midpoint_construction (pl : ParallelLines) (s : Segment) 
  (h : s.line = pl.line1 ∨ s.line = pl.line2) :
  ∃ (t : Trapezoid) (m : Segment), 
    (t.base1 = s ∨ t.base2 = s) ∧ 
    (m = midsegment t) ∧
    (∃ (midpoint : ℝ × ℝ), midpoint ∈ s.line ∧ midpoint ∈ m.line) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_midpoint_construction_l1046_104625


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_range_theorem_l1046_104650

-- Define the piecewise function f
noncomputable def f (x : ℝ) : ℝ :=
  if x ≤ 0 then 1/2 * x + 1 else -(x - 1)^2

-- Define the set of x values where f(x) ≥ -1
def S : Set ℝ := {x | f x ≥ -1}

-- Theorem statement
theorem f_range_theorem : S = Set.Icc (-4 : ℝ) 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_range_theorem_l1046_104650


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_strawberry_theorem_l1046_104601

/-- Represents the strawberry picking scenario -/
structure StrawberryPicking where
  entrance_fee : ℚ
  price_per_pound : ℚ
  num_people : ℕ
  total_paid : ℚ

/-- Calculates the pounds of strawberries picked given the scenario -/
def pounds_picked (scenario : StrawberryPicking) : ℚ :=
  (scenario.total_paid + scenario.entrance_fee * scenario.num_people) / scenario.price_per_pound

/-- Theorem stating that under the given conditions, 7 pounds of strawberries were picked -/
theorem strawberry_theorem (scenario : StrawberryPicking) 
  (h1 : scenario.entrance_fee = 4)
  (h2 : scenario.price_per_pound = 20)
  (h3 : scenario.num_people = 3)
  (h4 : scenario.total_paid = 128) :
  pounds_picked scenario = 7 := by
  sorry

#eval pounds_picked { entrance_fee := 4, price_per_pound := 20, num_people := 3, total_paid := 128 }

end NUMINAMATH_CALUDE_ERRORFEEDBACK_strawberry_theorem_l1046_104601


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_car_ownership_cost_theorem_l1046_104651

/-- Represents the total cost of owning a car for n years -/
noncomputable def total_cost (n : ℕ) : ℝ :=
  0.1 * (n ^ 2 : ℝ) + 1.0 * n + 10

/-- Represents the average annual cost of owning a car for n years -/
noncomputable def average_annual_cost (n : ℕ) : ℝ :=
  total_cost n / n

/-- The theorem states that the total cost function is correct and the most economical year is 10 -/
theorem car_ownership_cost_theorem :
  (∀ n : ℕ, n > 0 → total_cost n = 0.1 * (n ^ 2 : ℝ) + 1.0 * n + 10) ∧
  (∃! n : ℕ, n > 0 ∧ ∀ m : ℕ, m > 0 → average_annual_cost n ≤ average_annual_cost m) ∧
  (average_annual_cost 10 = 3) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_car_ownership_cost_theorem_l1046_104651


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_least_possible_lcm_l1046_104609

theorem least_possible_lcm (a b c : ℕ) 
  (h1 : Nat.lcm a b = 20) 
  (h2 : Nat.lcm b c = 21) : 
  ∃ (x : ℕ), (Nat.lcm a c = x) ∧ (∀ y, Nat.lcm a c ≤ y) ∧ (x = 420) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_least_possible_lcm_l1046_104609


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tetrachloromethane_formation_l1046_104664

/-- Represents the number of moles of a substance -/
def Moles : Type := ℝ

/-- Instance to allow using natural number literals for Moles -/
instance : OfNat Moles n where
  ofNat := (n : ℝ)

/-- Represents a chemical reaction with reactants and products -/
structure Reaction where
  methane : Moles
  chlorine : Moles
  tetrachloromethane : Moles
  hydrogenChloride : Moles

/-- The balanced chemical equation for the formation of Tetrachloromethane -/
def balancedEquation : Reaction → Prop := fun r =>
  r.methane = 1 ∧ r.chlorine = 4 ∧ r.tetrachloromethane = 1 ∧ r.hydrogenChloride = 4

/-- The available reactants -/
def availableReactants : Reaction → Prop := fun r =>
  r.methane = 1 ∧ r.chlorine = 4

/-- Theorem stating that given the available reactants, the amount of Tetrachloromethane formed is 1 mole -/
theorem tetrachloromethane_formation (r : Reaction) 
  (h1 : balancedEquation r) (h2 : availableReactants r) : r.tetrachloromethane = 1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tetrachloromethane_formation_l1046_104664


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_line_distance_l1046_104626

/-- The circle equation -/
def circle_eq (x y : ℝ) : Prop := x^2 + y^2 - 10*x - 6*y + 25 = 0

/-- The line equation -/
def line_eq (x y : ℝ) : Prop := 3*x + 4*y - 2 = 0

/-- The shortest distance from a point on the circle to the line -/
def shortest_distance : ℝ := 2

/-- Theorem: The shortest distance from any point on the given circle to the given line is 2 -/
theorem circle_line_distance :
  ∀ (x y : ℝ), circle_eq x y → 
  (∃ (d : ℝ), d = shortest_distance ∧ 
    ∀ (x' y' : ℝ), line_eq x' y' → 
      d ≤ Real.sqrt ((x - x')^2 + (y - y')^2)) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_line_distance_l1046_104626


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_equality_implies_a_value_l1046_104668

-- Define the function f as noncomputable due to its dependence on Real
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := 
  a * (Real.sin x * Real.cos x - Real.cos x ^ 2 + Real.cos (Real.pi / 2 + x) ^ 2)

-- State the theorem
theorem function_equality_implies_a_value :
  ∃ (a : ℝ), f a (-Real.pi / 3) = f a 0 ∧ a = 4 * Real.sqrt 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_equality_implies_a_value_l1046_104668


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_one_white_one_black_l1046_104622

theorem probability_one_white_one_black (total_balls : ℕ) (white_balls : ℕ) (black_balls : ℕ)
  (h_total : total_balls = white_balls + black_balls)
  (h_white : white_balls = 7)
  (h_black : black_balls = 3) :
  (white_balls : ℚ) / total_balls * (black_balls : ℚ) / total_balls +
  (black_balls : ℚ) / total_balls * (white_balls : ℚ) / total_balls = 42 / 100 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_one_white_one_black_l1046_104622


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_increasing_l1046_104617

def arithmetic_sequence (lambda : ℝ) (n : ℕ) : ℝ := 2 * n + lambda

def sum_of_terms (lambda : ℝ) (n : ℕ) : ℝ := n^2 + (1 + lambda) * n

theorem arithmetic_sequence_increasing (lambda : ℝ) :
  (∀ n ≥ 7, sum_of_terms lambda n < sum_of_terms lambda (n + 1)) ↔ lambda > -16 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_increasing_l1046_104617


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rent_comparison_l1046_104620

theorem rent_comparison (E : ℝ) (h1 : E > 0) : 
  (0.30 * (1.15 * E)) / (0.10 * E) = 3.45 := by
  -- Calculation steps would go here
  sorry

#check rent_comparison

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rent_comparison_l1046_104620


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_decreasing_log_half_power_implies_a_range_l1046_104680

-- Define the function f as noncomputable
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (Real.log a / Real.log (1/2)) ^ x

-- State the theorem
theorem decreasing_log_half_power_implies_a_range (a : ℝ) :
  (∀ x y : ℝ, x < y → f a y < f a x) → 1/2 < a ∧ a < 1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_decreasing_log_half_power_implies_a_range_l1046_104680


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_top_number_after_folds_l1046_104627

/-- Represents a 4x4 grid of integers -/
def Grid := Matrix (Fin 4) (Fin 4) ℕ

/-- Initializes the grid with numbers 1 to 16 -/
def init_grid : Grid := 
  Matrix.of (λ i j => i.val * 4 + j.val + 1)

/-- Represents a folding operation on the grid -/
inductive FoldOperation
| TopToBottom
| BottomToTop
| RightToLeft
| LeftToRight

/-- Applies a single fold operation to the grid -/
def apply_fold (g : Grid) (op : FoldOperation) : Grid :=
  sorry

/-- Applies all four fold operations in sequence -/
def apply_all_folds (g : Grid) : Grid :=
  apply_fold (apply_fold (apply_fold (apply_fold g FoldOperation.TopToBottom) 
    FoldOperation.BottomToTop) FoldOperation.RightToLeft) FoldOperation.LeftToRight

theorem top_number_after_folds :
  (apply_all_folds init_grid) 0 0 = 9 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_top_number_after_folds_l1046_104627


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_wff_classification_l1046_104641

-- Define a type for propositions
inductive Prop' where
  | var : String → Prop'

-- Define well-formed formulas
inductive WFF where
  | prop : Prop' → WFF
  | not : WFF → WFF
  | and : WFF → WFF → WFF
  | or : WFF → WFF → WFF
  | implies : WFF → WFF → WFF
  | iff : WFF → WFF → WFF

def is_wff : String → Bool
  | _ => sorry  -- Implementation of the is_wff function

-- Define the formulas from the problem
def formula1 : String := "((¬P→Q)→(Q→P))"
def formula2 : String := "(Q→R∧S)"
def formula3 : String := "(RS→T)"
def formula4 : String := "(P↔(R→S))"
def formula5 : String := "(((P→(Q→R))→((P→Q)→(P→R))))"

-- Theorem to prove
theorem wff_classification :
  (is_wff formula1 = true) ∧
  (is_wff formula2 = false) ∧
  (is_wff formula3 = false) ∧
  (is_wff formula4 = true) ∧
  (is_wff formula5 = true) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_wff_classification_l1046_104641


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_product_of_N_values_l1046_104657

-- Define the temperature difference at noon
variable (N : ℝ)

-- Define the temperatures at noon
variable (D : ℝ)  -- Denver's temperature at noon
variable (A : ℝ)  -- Austin's temperature at noon

-- Relationship between temperatures at noon
axiom noon_temp_diff : D = A + N

-- Temperatures at 6:00 PM
def D_6pm : ℝ := D - 7
def A_6pm : ℝ := A + 4

-- Temperature difference at 6:00 PM is 1 degree
axiom temp_diff_6pm : |D_6pm - A_6pm| = 1

-- Theorem: The product of all possible values of N is 120
theorem product_of_N_values : ∃ (N₁ N₂ : ℝ), N₁ ≠ N₂ ∧ (N = N₁ ∨ N = N₂) ∧ N₁ * N₂ = 120 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_product_of_N_values_l1046_104657


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cosine_sum_squared_l1046_104624

theorem cosine_sum_squared (α : ℝ) :
  (Real.cos α) ^ 2 + (Real.cos ((2/3) * Real.pi + α)) ^ 2 + (Real.cos ((2/3) * Real.pi - α)) ^ 2 = 3/2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cosine_sum_squared_l1046_104624


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_square_exterior_points_l1046_104661

/-- Given a square ABCD with side length 15 and exterior points E and F,
    prove that if BE = DF = 7 and AE = CF = 17, then EF² = 1216 -/
theorem square_exterior_points (A B C D E F : ℝ × ℝ) : 
  let side_length : ℝ := 15
  let square_ABCD := 
    (∀ (X Y : ℝ × ℝ), (X ∈ ({A, B, C, D} : Set (ℝ × ℝ)) ∧ Y ∈ ({A, B, C, D} : Set (ℝ × ℝ)) ∧ X ≠ Y) → 
      ‖X - Y‖ = side_length ∨ ‖X - Y‖ = side_length * Real.sqrt 2)
  let BE := ‖B - E‖
  let DF := ‖D - F‖
  let AE := ‖A - E‖
  let CF := ‖C - F‖
  let EF := ‖E - F‖
  square_ABCD ∧ BE = 7 ∧ DF = 7 ∧ AE = 17 ∧ CF = 17 →
  EF^2 = 1216 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_square_exterior_points_l1046_104661


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_alpha_half_plus_beta_eq_sqrt2_div_2_l1046_104630

theorem sin_alpha_half_plus_beta_eq_sqrt2_div_2 
  (α β : Real) 
  (h1 : α ∈ Set.Icc (π/2) (3*π/2))
  (h2 : β ∈ Set.Icc (-π/2) 0)
  (h3 : (α - π/2)^3 - Real.sin α - 2 = 0)
  (h4 : 8*β^3 + 2*(Real.cos β)^2 + 1 = 0) :
  Real.sin (α/2 + β) = Real.sqrt 2 / 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_alpha_half_plus_beta_eq_sqrt2_div_2_l1046_104630


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_direct_below_inverse_range_l1046_104682

noncomputable section

variable (b : ℝ)

/-- Direct proportion function passing through (2, b) -/
noncomputable def directProportion (x : ℝ) : ℝ := (b * x) / 2

/-- Inverse proportion function passing through (2, b) -/
noncomputable def inverseProportion (x : ℝ) : ℝ := (2 * b) / x

/-- The theorem stating the range where direct proportion is below inverse proportion -/
theorem direct_below_inverse_range (h : b > 0) :
  {x : ℝ | x > 0 ∧ directProportion b x < inverseProportion b x} = {x : ℝ | 0 < x ∧ x < 2} := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_direct_below_inverse_range_l1046_104682


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_motorcyclist_speed_increase_l1046_104693

/-- Represents the scenario of a motorcyclist and pedestrian meeting on a road --/
structure TravelScenario where
  v₁ : ℝ  -- Speed of the pedestrian
  v₂ : ℝ  -- Speed of the motorcyclist
  distance : ℝ  -- Distance between points A and B

/-- The time taken by the pedestrian with the ride is 4 times faster than walking the entire distance --/
def pedestrian_time_condition (s : TravelScenario) : Prop :=
  2 / (s.v₁ + s.v₂) = (1/4) * (s.distance / s.v₁)

/-- The ratio of time taken by the motorcyclist with and without returning --/
noncomputable def motorcyclist_time_ratio (s : TravelScenario) : ℝ :=
  let time_with_return := (2 * s.v₂ / (s.v₁ + s.v₂) + s.distance) / s.v₂
  let time_direct := s.distance / s.v₂
  time_with_return / time_direct

/-- Theorem stating the motorcyclist's time ratio --/
theorem motorcyclist_speed_increase 
  (s : TravelScenario) 
  (h₁ : s.v₁ > 0) 
  (h₂ : s.v₂ > 0) 
  (h₃ : s.distance > 0) 
  (h₄ : pedestrian_time_condition s) : 
  motorcyclist_time_ratio s = 2.75 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_motorcyclist_speed_increase_l1046_104693


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_operation_result_l1046_104606

theorem vector_operation_result : ∃ (a b result : Fin 2 → ℝ),
  a = ![2, 3] ∧ b = ![-1, 2] ∧ result = ![5, 4] ∧ (2 • a - b) = result := by
  -- Define the vectors
  let a : Fin 2 → ℝ := ![2, 3]
  let b : Fin 2 → ℝ := ![-1, 2]
  
  -- Define the operation result
  let result : Fin 2 → ℝ := ![5, 4]
  
  -- Provide the existence proof
  use a, b, result
  
  -- State and prove the equalities
  have h1 : a = ![2, 3] := rfl
  have h2 : b = ![-1, 2] := rfl
  have h3 : result = ![5, 4] := rfl
  have h4 : (2 • a - b) = result := by
    -- The actual proof would go here
    sorry
  
  -- Combine all parts of the proof
  exact ⟨h1, h2, h3, h4⟩


end NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_operation_result_l1046_104606


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equilateral_iff_interior_points_form_triangle_l1046_104691

/-- A triangle is defined by its three vertices -/
structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ

/-- A point is inside a triangle if it's a convex combination of the triangle's vertices -/
def IsInside (M : ℝ × ℝ) (T : Triangle) : Prop :=
  ∃ (α β γ : ℝ), α ≥ 0 ∧ β ≥ 0 ∧ γ ≥ 0 ∧ α + β + γ = 1 ∧
  M = (α * T.A.1 + β * T.B.1 + γ * T.C.1, α * T.A.2 + β * T.B.2 + γ * T.C.2)

/-- The distance between two points -/
noncomputable def Distance (P Q : ℝ × ℝ) : ℝ :=
  Real.sqrt ((P.1 - Q.1)^2 + (P.2 - Q.2)^2)

/-- Three segments can form a triangle if the sum of any two sides is greater than the third -/
def CanFormTriangle (a b c : ℝ) : Prop :=
  a + b > c ∧ b + c > a ∧ c + a > b

/-- A triangle is equilateral if all its sides have equal length -/
def IsEquilateral (T : Triangle) : Prop :=
  Distance T.A T.B = Distance T.B T.C ∧ Distance T.B T.C = Distance T.C T.A

/-- The main theorem -/
theorem equilateral_iff_interior_points_form_triangle (T : Triangle) :
  IsEquilateral T ↔
  ∀ M : ℝ × ℝ, IsInside M T →
    CanFormTriangle (Distance M T.A) (Distance M T.B) (Distance M T.C) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_equilateral_iff_interior_points_form_triangle_l1046_104691


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_adjacent_sum_to_7_l1046_104647

def divisors_147 : List ℕ := [3, 7, 21, 49, 147]

def valid_arrangement (arr : List ℕ) : Prop :=
  arr.Perm divisors_147 ∧
  ∀ i, (Nat.gcd (arr[i]!) (arr[(i + 1) % arr.length]!)) > 1

def adjacent_sum (arr : List ℕ) : ℕ :=
  let i := arr.indexOf 7
  arr[(i - 1 + arr.length) % arr.length]! + arr[(i + 1) % arr.length]!

theorem adjacent_sum_to_7 (arr : List ℕ) :
  valid_arrangement arr → adjacent_sum arr = 196 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_adjacent_sum_to_7_l1046_104647


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_last_number_is_thirty_l1046_104600

theorem last_number_is_thirty (numbers : Fin 8 → ℝ) 
  (avg_all : (Finset.sum Finset.univ (λ i => numbers i)) / 8 = 25)
  (avg_first_two : (numbers 0 + numbers 1) / 2 = 20)
  (avg_next_three : (numbers 2 + numbers 3 + numbers 4) / 3 = 26)
  (sixth_vs_seventh : numbers 5 + 4 = numbers 6)
  (sixth_vs_eighth : numbers 5 + 6 = numbers 7) :
  numbers 7 = 30 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_last_number_is_thirty_l1046_104600


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_square_distance_on_parabola_l1046_104663

/-- A point on a parabola -/
structure ParabolaPoint where
  x : ℝ
  y : ℝ
  on_parabola : y = x^2

/-- The distance from a point to a line -/
noncomputable def distance_to_line (p : ParabolaPoint) : ℝ :=
  |p.x + p.y + 4| / Real.sqrt 2

/-- The square of the distance between two points -/
def square_distance (p1 p2 : ParabolaPoint) : ℝ :=
  (p1.x - p2.x)^2 + (p1.y - p2.y)^2

/-- Theorem stating the square distance between two specific points on the parabola -/
theorem square_distance_on_parabola :
  ∃ (A B : ParabolaPoint),
    distance_to_line A = 8 * Real.sqrt 2 ∧
    distance_to_line B = 8 * Real.sqrt 2 ∧
    square_distance A B = 98 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_square_distance_on_parabola_l1046_104663


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_circle_problem_l1046_104608

/-- Parabola C: x^2 = 2py (p > 0) -/
def parabola (p : ℝ) (x y : ℝ) : Prop := x^2 = 2*p*y ∧ p > 0

/-- Circle O: x^2 + y^2 = 1 -/
def circleO (x y : ℝ) : Prop := x^2 + y^2 = 1

/-- Focus F of parabola C lies on circle O -/
def focus_on_circle (p : ℝ) : Prop := circleO 0 (p/2)

/-- A is an intersection point of C and O -/
def intersection_point (p : ℝ) (x y : ℝ) : Prop := parabola p x y ∧ circleO x y

/-- Line l is tangent to both C and O at points M and N respectively -/
def tangent_line (p : ℝ) (xm ym xn yn : ℝ) : Prop := 
  parabola p xm ym ∧ circleO xn yn ∧ 
  ∃ (a b c : ℝ), a*xm + b*ym + c = 0 ∧ a*xn + b*yn + c = 0

theorem parabola_circle_problem (p : ℝ) :
  (∀ x y, intersection_point p x y → ∃ xf yf, circleO xf yf ∧ Real.sqrt ((x - xf)^2 + (y - yf)^2) = Real.sqrt 5 - 1) ∧
  (∃ min_mn : ℝ, min_mn = 2 * Real.sqrt 2 ∧
    (∀ xm ym xn yn, tangent_line p xm ym xn yn → 
      Real.sqrt ((xm - xn)^2 + (ym - yn)^2) ≥ min_mn) ∧
    (∃ xm ym xn yn, tangent_line p xm ym xn yn ∧
      Real.sqrt ((xm - xn)^2 + (ym - yn)^2) = min_mn)) ∧
  (p = Real.sqrt 3) :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_circle_problem_l1046_104608


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_chemical_mixture_problem_l1046_104605

/-- The amount of chemical x added to a mixture -/
noncomputable def amount_added (initial_volume : ℝ) (initial_concentration : ℝ) (final_concentration : ℝ) : ℝ :=
  (final_concentration * initial_volume - initial_concentration * initial_volume) / (1 - final_concentration)

/-- Theorem stating the amount of chemical x added to the mixture -/
theorem chemical_mixture_problem :
  let initial_volume : ℝ := 80
  let initial_concentration : ℝ := 0.20
  let final_concentration : ℝ := 0.36
  amount_added initial_volume initial_concentration final_concentration = 20 := by
  -- Unfold the definition of amount_added
  unfold amount_added
  -- Simplify the expression
  simp
  -- The proof is completed with sorry for now
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_chemical_mixture_problem_l1046_104605


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_remainder_six_divisor_of_53_l1046_104639

theorem remainder_six_divisor_of_53 : 
  ∃! n : ℕ, n > 6 ∧ 53 % n = 6 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_remainder_six_divisor_of_53_l1046_104639


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_theta_is_one_third_l1046_104613

noncomputable section

/-- Represents a 2x2 matrix --/
def Matrix2x2 := Fin 2 → Fin 2 → ℝ

/-- Dilation matrix with scale factor k --/
def D (k : ℝ) : Matrix2x2 := λ i j => if i = j then k else 0

/-- Rotation matrix with angle θ --/
noncomputable def R (θ : ℝ) : Matrix2x2 := λ i j =>
  if i = 0 && j = 0 then Real.cos θ
  else if i = 0 && j = 1 then -Real.sin θ
  else if i = 1 && j = 0 then Real.sin θ
  else Real.cos θ

/-- Translation matrix with parameter t --/
def T (t : ℝ) : Matrix2x2 := λ i j =>
  if i = j then 1
  else if i = 0 && j = 1 then t
  else 0

/-- Matrix multiplication --/
def matMul (A B : Matrix2x2) : Matrix2x2 :=
  λ i j => (Finset.univ.sum λ k => A i k * B k j)

theorem tan_theta_is_one_third
  (k : ℝ)
  (θ : ℝ)
  (t : ℝ)
  (h1 : k > 0)
  (h2 : matMul (T t) (matMul (R θ) (D k)) 0 0 = 12)
  (h3 : matMul (T t) (matMul (R θ) (D k)) 0 1 = -3)
  (h4 : matMul (T t) (matMul (R θ) (D k)) 1 0 = 4)
  (h5 : matMul (T t) (matMul (R θ) (D k)) 1 1 = 12) :
  Real.tan θ = 1/3 := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_theta_is_one_third_l1046_104613


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_ratio_MNO_XYZ_l1046_104655

-- Define the triangle XYZ
variable (X Y Z : ℝ × ℝ)

-- Define points G, H, I on the sides of the triangle
noncomputable def G (X Y Z : ℝ × ℝ) : ℝ × ℝ := (2/5 * Y.1 + 3/5 * Z.1, 2/5 * Y.2 + 3/5 * Z.2)
noncomputable def H (X Y Z : ℝ × ℝ) : ℝ × ℝ := (2/5 * X.1 + 3/5 * Z.1, 2/5 * X.2 + 3/5 * Z.2)
noncomputable def I (X Y Z : ℝ × ℝ) : ℝ × ℝ := (2/5 * X.1 + 3/5 * Y.1, 2/5 * X.2 + 3/5 * Y.2)

-- Define the intersection points M, N, O
noncomputable def M (X Y Z : ℝ × ℝ) : ℝ × ℝ := sorry
noncomputable def N (X Y Z : ℝ × ℝ) : ℝ × ℝ := sorry
noncomputable def O (X Y Z : ℝ × ℝ) : ℝ × ℝ := sorry

-- Define the area function
noncomputable def area (A B C : ℝ × ℝ) : ℝ := sorry

-- State the theorem
theorem area_ratio_MNO_XYZ (X Y Z : ℝ × ℝ) :
  area (M X Y Z) (N X Y Z) (O X Y Z) / area X Y Z = 27 / 440 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_ratio_MNO_XYZ_l1046_104655


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tripleSum_eq_inverse_106505_l1046_104611

/-- The sum of 1 / (3^a * 4^b * 6^c) for all positive integers a, b, c where 1 ≤ a < b < c -/
noncomputable def tripleSum : ℚ :=
  ∑' (a : ℕ), ∑' (b : ℕ), ∑' (c : ℕ),
    if 1 ≤ a ∧ a < b ∧ b < c then
      (1 : ℚ) / (3^a * 4^b * 6^c)
    else
      0

/-- The sum of 1 / (3^a * 4^b * 6^c) for all positive integers a, b, c where 1 ≤ a < b < c
    equals 1/106505 -/
theorem tripleSum_eq_inverse_106505 : tripleSum = 1 / 106505 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tripleSum_eq_inverse_106505_l1046_104611


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_polynomial_satisfies_conditions_l1046_104653

def quadratic_polynomial (a b c : ℝ) : ℂ → ℂ := λ x ↦ a * x^2 + b * x + c

theorem quadratic_polynomial_satisfies_conditions :
  ∃ (a b c : ℝ),
    let p := quadratic_polynomial a b c
    (p (1 + 2*Complex.I) = 0) ∧  -- 1 + 2i is a root
    (b = 10) ∧           -- coefficient of x is 10
    (a = -5 ∧ b = 10 ∧ c = -25)  -- the polynomial is -5x^2 + 10x - 25
    := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_polynomial_satisfies_conditions_l1046_104653


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_obtuse_triangle_proportion_l1046_104629

theorem obtuse_triangle_proportion (n : ℕ) (h : n ≥ 5) :
  (7 * Nat.choose n 5 : ℚ) / (Nat.choose (n - 3) 2 * Nat.choose n 3) ≤ 7/10 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_obtuse_triangle_proportion_l1046_104629


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_COB_area_l1046_104667

/-- Point in 2D plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- Triangle defined by three points -/
structure Triangle where
  A : Point
  B : Point
  C : Point

/-- Calculate the area of a right triangle -/
noncomputable def areaRightTriangle (base height : ℝ) : ℝ :=
  (1 / 2) * base * height

theorem triangle_COB_area (k p : ℝ) (h1 : 0 < k) (h2 : k < 1) :
  let Q : Point := ⟨0, 20⟩
  let B : Point := ⟨10, 0⟩
  let O : Point := ⟨0, 0⟩
  let C : Point := ⟨0, k * p⟩
  let triangle : Triangle := ⟨O, B, C⟩
  areaRightTriangle 10 (k * p) = 5 * k * p := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_COB_area_l1046_104667


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_friends_walking_distance_l1046_104645

theorem friends_walking_distance 
  (lionel_miles : ℕ) 
  (esther_yards : ℕ) 
  (niklaus_feet : ℕ) : 
  lionel_miles = 4 ∧ 
  esther_yards = 975 ∧ 
  niklaus_feet = 1287 →
  (lionel_miles * 5280 + esther_yards * 3 + niklaus_feet) = 25332 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_friends_walking_distance_l1046_104645


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_appliance_cost_sum_l1046_104646

/-- The cost of a mixer in Rs. -/
def mixer_cost : ℝ := sorry

/-- The cost of a TV in Rs. -/
def tv_cost : ℝ := sorry

/-- The cost of a blender in Rs. -/
def blender_cost : ℝ := sorry

/-- Two mixers, one TV, and one blender cost Rs. 10,500 -/
axiom equation1 : 2 * mixer_cost + tv_cost + blender_cost = 10500

/-- Two TVs, one mixer, and two blenders cost Rs. 14,700 -/
axiom equation2 : 2 * tv_cost + mixer_cost + 2 * blender_cost = 14700

/-- The sum of the costs of one TV, one mixer, and one blender equals 18,900 -/
theorem appliance_cost_sum : tv_cost + mixer_cost + blender_cost = 18900 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_appliance_cost_sum_l1046_104646


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distributor_profit_is_twenty_percent_l1046_104673

/-- Calculates the distributor's profit percentage given the online store commission rate,
    producer's price, and buyer's observed price. -/
noncomputable def distributor_profit_percentage (commission_rate : ℝ) (producer_price : ℝ) (buyer_price : ℝ) : ℝ :=
  let distributor_revenue := buyer_price * (1 - commission_rate)
  let profit := distributor_revenue - producer_price
  (profit / producer_price) * 100

/-- Theorem stating that the distributor's profit percentage is 20% given the specified conditions. -/
theorem distributor_profit_is_twenty_percent :
  distributor_profit_percentage 0.2 17 25.5 = 20 := by
  sorry

-- Remove the #eval statement as it's not computable
-- #eval distributor_profit_percentage 0.2 17 25.5

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distributor_profit_is_twenty_percent_l1046_104673


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_lines_at_distance_one_given_lines_parallel_and_at_distance_one_l1046_104616

noncomputable def normalVector (a b : ℝ) : ℝ × ℝ := (a, b)

noncomputable def distanceBetweenParallelLines (a b c1 c2 : ℝ) : ℝ :=
  |c2 - c1| / Real.sqrt (a^2 + b^2)

def areParallel (a1 b1 a2 b2 : ℝ) : Prop :=
  a1 * b2 = a2 * b1

theorem parallel_lines_at_distance_one (a b c : ℝ) :
  let l1 := fun x y => a * x + b * y + c = 0
  let l2 := fun x y => a * x + b * y + (c + Real.sqrt (a^2 + b^2)) = 0
  let l3 := fun x y => a * x + b * y + (c - Real.sqrt (a^2 + b^2)) = 0
  (areParallel a b a b) ∧
  (distanceBetweenParallelLines a b c (c + Real.sqrt (a^2 + b^2)) = 1) ∧
  (distanceBetweenParallelLines a b c (c - Real.sqrt (a^2 + b^2)) = 1) :=
by sorry

theorem given_lines_parallel_and_at_distance_one :
  let l1 := fun x y => 3 * x + 4 * y - 1 = 0
  let l2 := fun x y => 3 * x + 4 * y + 4 = 0
  let l3 := fun x y => 3 * x + 4 * y - 6 = 0
  (areParallel 3 4 3 4) ∧
  (distanceBetweenParallelLines 3 4 (-1) 4 = 1) ∧
  (distanceBetweenParallelLines 3 4 (-1) (-6) = 1) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_lines_at_distance_one_given_lines_parallel_and_at_distance_one_l1046_104616


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_calculation_result_l1046_104649

theorem complex_calculation_result : 
  ∃ x : ℝ, abs (x - ((235.47 / 100 * 9876.34) / 16.37 + (4 / 7 * (2836.9 - 1355.8)))) < 0.01 := by
  use 2266.42
  norm_num
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_calculation_result_l1046_104649


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pure_imaginary_condition_l1046_104696

-- Define the complex number type
variable (z : ℂ)

-- Define a as a real number
variable (a : ℝ)

-- Define what it means for a complex number to be pure imaginary
def is_pure_imaginary (z : ℂ) : Prop := z.re = 0 ∧ z.im ≠ 0

-- State the theorem
theorem pure_imaginary_condition : is_pure_imaginary ((1 + a * Complex.I) * (2 + Complex.I)) → a = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_pure_imaginary_condition_l1046_104696
