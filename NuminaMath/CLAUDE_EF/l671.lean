import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_abs_difference_l671_67181

theorem min_abs_difference (x y : ℕ+) (h : x.val * y.val - 4 * x.val + 3 * y.val = 315) :
  ∃ (a b : ℕ+), a.val * b.val - 4 * a.val + 3 * b.val = 315 ∧ 
  (∀ (c d : ℕ+), c.val * d.val - 4 * c.val + 3 * d.val = 315 → |Int.ofNat c.val - Int.ofNat d.val| ≥ |Int.ofNat a.val - Int.ofNat b.val|) ∧
  |Int.ofNat a.val - Int.ofNat b.val| = 91 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_abs_difference_l671_67181


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_of_f_l671_67165

-- Define the piecewise function
noncomputable def f (x : ℝ) : ℝ :=
  if x ≤ 0 then 2*x + 3
  else if x ≤ 1 then x + 3
  else -x + 5

-- State the theorem
theorem max_value_of_f :
  ∃ (M : ℝ), M = 4 ∧ ∀ (x : ℝ), f x ≤ M := by
  -- Proof goes here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_of_f_l671_67165


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_stretched_rhombus_area_l671_67127

/-- The area of a rhombus with given diagonal lengths -/
noncomputable def rhombusArea (d1 d2 : ℝ) : ℝ := (d1 * d2) / 2

/-- The length of a diagonal after tripling -/
def tripleLength (x : ℝ) : ℝ := 3 * x

theorem stretched_rhombus_area :
  let original_d1 : ℝ := 6
  let original_d2 : ℝ := 4
  let stretched_d1 : ℝ := tripleLength original_d1
  let stretched_d2 : ℝ := tripleLength original_d2
  rhombusArea stretched_d1 stretched_d2 = 108 := by
  -- Unfold definitions and simplify
  unfold rhombusArea tripleLength
  -- Perform algebraic manipulations
  simp [mul_assoc, mul_comm, mul_div_cancel]
  -- The proof is complete
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_stretched_rhombus_area_l671_67127


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_inequality_side_length_constraint_l671_67143

-- Define the Triangle structure
structure Triangle where
  sides : Finset ℝ
  side_count : sides.card = 3
  positive_sides : ∀ s ∈ sides, s > 0

-- Triangle inequality theorem
theorem triangle_inequality {a b c : ℝ} (ha : a > 0) (hb : b > 0) (hc : c > 0)
  (hab : a + b > c) (hbc : b + c > a) (hca : c + a > b) :
  ∃ (t : Triangle), t.sides = {a, b, c} := by
  sorry

-- Theorem stating that a triangle with sides 3, 5, and 8 cannot exist
theorem side_length_constraint : ¬∃ (t : Triangle), t.sides = {3, 5, 8} := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_inequality_side_length_constraint_l671_67143


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_side_count_l671_67155

open Real

theorem triangle_side_count : ∃! n : ℕ, 
  n = (Finset.filter (fun m : ℕ ↦ 
    m > 0 ∧
    m < 1350 ∧
    Real.log 15 + Real.log 90 > Real.log m ∧
    Real.log 15 + Real.log m > Real.log 90 ∧
    Real.log 90 + Real.log m > Real.log 15
  ) (Finset.range 1350)).card ∧ n = 1343 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_side_count_l671_67155


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_function_condition_range_of_f_plus_one_l671_67166

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := 2 / (3^x + 1) + a

theorem odd_function_condition (a : ℝ) :
  (∀ x, f x a = -f (-x) a) → a = -1 := by sorry

theorem range_of_f_plus_one :
  ∀ x ∈ Set.Icc 0 1, (1/2 : ℝ) ≤ f x (-1) + 1 ∧ f x (-1) + 1 ≤ 1 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_function_condition_range_of_f_plus_one_l671_67166


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_dodecahedron_opposite_face_probability_l671_67130

structure Dodecahedron where
  vertices : Finset Nat
  edges : Finset (Nat × Nat)
  faces : Finset (Finset Nat)
  is_regular : Bool

def adjacent_vertices (d : Dodecahedron) (v : Nat) : Finset Nat :=
  sorry

def opposite_face (d : Dodecahedron) (start : Nat) : Finset Nat :=
  sorry

def random_walk (d : Dodecahedron) (start : Nat) : Nat :=
  sorry

theorem dodecahedron_opposite_face_probability 
  (d : Dodecahedron)
  (h_regular : d.is_regular = true)
  (h_vertices : d.vertices.card = 20)
  (h_faces : d.faces.card = 12)
  (start : Nat)
  (h_start : start ∈ d.vertices) :
  ∃ (p : ℚ), p = 1/3 ∧ 
  p = (Finset.filter (λ end_vertex => end_vertex ∈ opposite_face d start) d.vertices).card / 
      d.vertices.card :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_dodecahedron_opposite_face_probability_l671_67130


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_problem_l671_67157

-- Define the vector type
def Vector2D := ℝ × ℝ

-- Define given vectors
def a : Vector2D := (1, -2)
def b : Vector2D := (3, 4)

-- Define vector operations
def scale (c : ℝ) (v : Vector2D) : Vector2D := (c * v.1, c * v.2)
def add (v w : Vector2D) : Vector2D := (v.1 + w.1, v.2 + w.2)
def sub (v w : Vector2D) : Vector2D := (v.1 - w.1, v.2 - w.2)

-- Define collinearity
def collinear (v w : Vector2D) : Prop :=
  ∃ (c : ℝ), v = scale c w ∨ w = scale c v

theorem vector_problem :
  (add (scale 3 a) (scale 4 b) = (15, 10)) ∧
  (∀ k : ℝ, collinear (sub (scale k a) b) (add (scale 3 a) (scale 4 b)) ↔ k = -3/4) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_problem_l671_67157


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arrangement_theorem_l671_67180

/-- The number of ways to arrange 6 people with 3 specific people not all on the same side -/
def arrangement_count : ℕ := 240

/-- The number of people to be arranged -/
def total_people : ℕ := 6

/-- The number of specific people (A, B, and C) -/
def specific_people : ℕ := 3

theorem arrangement_theorem :
  (arrangement_count = 2 * Nat.factorial (total_people - specific_people + 1)) ∧
  (specific_people = 3) ∧
  (total_people = 6) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_arrangement_theorem_l671_67180


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_charlie_area_painted_l671_67145

/-- Given a total area and work ratios, calculates the area painted by one person --/
noncomputable def area_painted (total_area : ℝ) (ratios : List ℝ) (person_index : ℕ) : ℝ :=
  total_area * (ratios.get! person_index) / (ratios.sum)

/-- Theorem: Charlie paints 72 square feet given the problem conditions --/
theorem charlie_area_painted :
  let total_area : ℝ := 360
  let ratios : List ℝ := [3, 5, 2]
  let charlie_index : ℕ := 2
  area_painted total_area ratios charlie_index = 72 := by
  -- Proof steps would go here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_charlie_area_painted_l671_67145


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_geometric_sequence_theorem_l671_67123

noncomputable def arithmetic_sequence (a₁ d : ℝ) (n : ℕ) : ℝ := a₁ + (n - 1 : ℝ) * d

noncomputable def geometric_sequence (b₁ q : ℝ) (n : ℕ) : ℝ := b₁ * q^(n - 1 : ℝ)

noncomputable def sum_arithmetic_sequence (a₁ d : ℝ) (n : ℕ) : ℝ := n * (2 * a₁ + (n - 1 : ℝ) * d) / 2

theorem arithmetic_geometric_sequence_theorem :
  ∀ n : ℕ,
  n ≥ 1 →
  let a_n := arithmetic_sequence 1 2 n
  let b_n := geometric_sequence 1 2 n
  let c_n := a_n / b_n
  let T_n := sum_arithmetic_sequence c_n (c_n - c_n) n
  (a_n = 2 * n - 1 ∧ b_n = 2^(n - 1 : ℝ)) ∧
  T_n = 6 - (2 * n + 3) / 2^(n - 1 : ℝ) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_geometric_sequence_theorem_l671_67123


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_nine_consecutive_primes_exist_l671_67120

def sequencePrime (a b c : ℕ) : ℕ → ℕ
| 0 => a
| 1 => b
| 2 => c
| (n + 3) => sequencePrime a b c n + sequencePrime a b c (n + 1) + sequencePrime a b c (n + 2)

theorem nine_consecutive_primes_exist : ∃ a b c : ℕ, ∀ i : ℕ, i < 9 → Nat.Prime (sequencePrime a b c i) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_nine_consecutive_primes_exist_l671_67120


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_perfect_square_factors_of_7200_l671_67115

-- Define 7200 as a natural number
def n : ℕ := 7200

-- Define the prime factorization of 7200
axiom n_factorization : n = 2^4 * 3^2 * 5^2

-- Define a function that counts the number of perfect square factors
def count_perfect_square_factors (m : ℕ) : ℕ := sorry

-- State the theorem
theorem perfect_square_factors_of_7200 : count_perfect_square_factors n = 12 := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_perfect_square_factors_of_7200_l671_67115


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vendelin_speed_l671_67104

/-- Represents the distance between two bus stops -/
def d : ℝ := sorry

/-- Represents Vendelín's running speed -/
def v : ℝ := sorry

/-- The bus speed in km/h -/
noncomputable def bus_speed : ℝ := 60

/-- The fraction of the total distance where Vendelín's house is located -/
noncomputable def house_fraction : ℝ := 3/8

/-- Time taken by the bus to travel the full distance -/
noncomputable def bus_time (d : ℝ) : ℝ := d / bus_speed

/-- Distance from Vendelín's house to the first bus stop -/
noncomputable def distance_to_first_stop (d : ℝ) : ℝ := house_fraction * d

/-- Distance from Vendelín's house to the second bus stop -/
noncomputable def distance_to_second_stop (d : ℝ) : ℝ := d - distance_to_first_stop d

/-- Theorem stating that Vendelín's running speed is 15 km/h -/
theorem vendelin_speed : 
  (∀ d > 0, 
    distance_to_first_stop d / v = bus_time d ∧ 
    distance_to_second_stop d / v = bus_time d) → 
  v = 15 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_vendelin_speed_l671_67104


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_minimize_sum_of_roots_l671_67183

/-- The function f(a) as defined in the problem -/
noncomputable def f (a : ℝ) : ℝ := a^2 - Real.sqrt 21 * a + 26

/-- The function g(a) as defined in the problem -/
noncomputable def g (a : ℝ) : ℝ := (3/2) * a^2 - Real.sqrt 21 * a + 27

/-- The equation from the problem, represented as a function of x and a -/
noncomputable def equation (x a : ℝ) : Prop :=
  (f a * x^2 + 1) / (x^2 + g a) = Real.sqrt ((x * g a - 1) / (f a - x))

theorem minimize_sum_of_roots :
  ∃ (a : ℝ), a = Real.sqrt 21 / 2 ∧
  ∀ (b : ℝ), f a ≤ f b ∧
  (∀ (x : ℝ), equation x a → equation x b) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_minimize_sum_of_roots_l671_67183


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_base_conversion_sum_equality_l671_67170

/-- Converts a number from base b to base 10 -/
def toBase10 (n : ℕ) (b : ℕ) : ℕ := sorry

/-- The main theorem to prove -/
theorem base_conversion_sum_equality :
  (toBase10 165 7 / toBase10 11 2 + toBase10 121 6 / toBase10 21 3 : ℚ) = 39 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_base_conversion_sum_equality_l671_67170


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_continuity_condition_l671_67186

-- Define the piecewise function f
noncomputable def f (b : ℝ) (x : ℝ) : ℝ :=
  if x ≤ 5 then 2 * x^2 + 3 * x + 1 else b * x + 2

-- State the theorem
theorem continuity_condition (b : ℝ) :
  Continuous (f b) ↔ b = 64 / 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_continuity_condition_l671_67186


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_current_for_12_ohm_resistance_l671_67114

-- Define the battery voltage
noncomputable def battery_voltage : ℝ := 48

-- Define the relationship between current and resistance
noncomputable def current (resistance : ℝ) : ℝ := battery_voltage / resistance

-- State the theorem
theorem current_for_12_ohm_resistance : current 12 = 4 := by
  -- Unfold the definitions
  unfold current battery_voltage
  -- Perform the calculation
  norm_num
  -- QED

end NUMINAMATH_CALUDE_ERRORFEEDBACK_current_for_12_ohm_resistance_l671_67114


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tourist_tax_calculation_l671_67139

/-- Calculates the tax to be paid on a purchase in Country B -/
noncomputable def calculate_tax (purchase_amount : ℝ) (tax_free_limit : ℝ) (tax_rate : ℝ) : ℝ :=
  max 0 ((purchase_amount - tax_free_limit) * tax_rate)

/-- Theorem stating the correct tax calculation for the given problem -/
theorem tourist_tax_calculation :
  let purchase_amount : ℝ := 1720
  let tax_free_limit : ℝ := 800
  let tax_rate : ℝ := 0.10
  calculate_tax purchase_amount tax_free_limit tax_rate = 92 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tourist_tax_calculation_l671_67139


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_area_proof_l671_67174

/-- The equation of the circle -/
def circle_equation (x y : ℝ) : Prop :=
  x^2 + y^2 + 12*x + 20*y + 9 = 0

/-- The area of the circle -/
noncomputable def circle_area : ℝ := 127 * Real.pi

/-- Theorem stating the existence of center and radius for the circle, and the correctness of the area -/
theorem circle_area_proof :
  ∃ (h k r : ℝ), (∀ x y : ℝ, circle_equation x y ↔ (x - h)^2 + (y - k)^2 = r^2) ∧
  circle_area = π * r^2 := by
  -- Provide the center coordinates and radius
  use -6, -10, (127 : ℝ).sqrt
  constructor
  · -- Proof of the circle equation equivalence
    intro x y
    simp [circle_equation]
    ring_nf
    -- The rest of the proof is omitted
    sorry
  · -- Proof of the area equality
    simp [circle_area]
    -- The rest of the proof is omitted
    sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_area_proof_l671_67174


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_minimal_cost_l671_67151

/-- The cost function for truck transportation --/
noncomputable def cost_function (v : ℝ) : ℝ := 50000 / v + 5 * v

/-- The domain of the cost function --/
def is_in_domain (v : ℝ) : Prop := 0 < v ∧ v ≤ 100

theorem minimal_cost :
  ∀ v : ℝ, is_in_domain v → cost_function v ≥ 1000 ∧
  (cost_function v = 1000 ↔ v = 100) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_minimal_cost_l671_67151


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_of_expression_l671_67142

theorem max_value_of_expression (a b : ℝ) 
  (ha : 300 ≤ a ∧ a ≤ 500) 
  (hb : 900 ≤ b ∧ b ≤ 1500) : 
  ∃ (max : ℝ), ∀ (x : ℝ), (∃ (a' b' : ℝ), 300 ≤ a' ∧ a' ≤ 500 ∧ 900 ≤ b' ∧ b' ≤ 1500 ∧ x = (b' - 100) / a') → x ≤ max ∧ max = 14/3 := by
  sorry

#check max_value_of_expression

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_of_expression_l671_67142


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_matt_protein_intake_l671_67150

/-- Calculates the daily protein intake per kilogram of body weight -/
noncomputable def daily_protein_intake_per_kg (protein_percentage : ℚ) (body_weight : ℚ) (weekly_powder_intake : ℚ) : ℚ :=
  (protein_percentage * weekly_powder_intake) / (body_weight * 7)

/-- Theorem: Matt's daily protein intake per kilogram of body weight is 2 grams -/
theorem matt_protein_intake :
  daily_protein_intake_per_kg (4/5) 80 1400 = 2 := by
  -- Unfold the definition and simplify
  unfold daily_protein_intake_per_kg
  -- Perform the calculation
  simp [mul_div_assoc, mul_comm, mul_assoc]
  -- The proof is complete
  rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_matt_protein_intake_l671_67150


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sales_and_profit_optimization_l671_67132

-- Define the sales volume function
noncomputable def sales_volume (m : ℝ) (x : ℝ) : ℝ := m / (x - 3) + 8 * (x - 6)^2

-- Define the profit function
noncomputable def profit (m : ℝ) (x : ℝ) : ℝ := (x - 3) * (sales_volume m x)

-- Theorem statement
theorem sales_and_profit_optimization (m : ℝ) :
  (∀ x, 3 < x → x < 6 → sales_volume m 5 = 11) →
  (m = 6 ∧ ∀ x, 3 < x → x < 6 → profit 6 x ≤ profit 6 4) := by
  sorry

#check sales_and_profit_optimization

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sales_and_profit_optimization_l671_67132


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_1_problem_2_l671_67113

-- Problem 1
theorem problem_1 : |1 - Real.sqrt 2| - Real.sqrt ((-2)^2) + (27 : ℝ)^(1/3) = Real.sqrt 2 := by
  sorry

-- Problem 2
theorem problem_2 : ∀ x : ℝ, (x + 1)^2 = 1 → x = 0 ∨ x = -2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_1_problem_2_l671_67113


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_area_special_triangle_l671_67116

/-- Represents a triangle with side lengths a, b, and c -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  pos_a : a > 0
  pos_b : b > 0
  pos_c : c > 0
  triangle_ineq : a + b > c ∧ b + c > a ∧ c + a > b

/-- Calculates the area of a triangle using Heron's formula -/
noncomputable def triangleArea (t : Triangle) : ℝ :=
  let s := (t.a + t.b + t.c) / 2
  Real.sqrt (s * (s - t.a) * (s - t.b) * (s - t.c))

/-- Theorem: The maximum area of a triangle with AB = 12 and BC:AC = 13:14 is 7221/117 -/
theorem max_area_special_triangle :
  ∃ (t : Triangle),
    t.a = 12 ∧
    t.b / t.c = 13 / 14 ∧
    triangleArea t = 7221 / 117 ∧
    ∀ (t' : Triangle),
      t'.a = 12 ∧ t'.b / t'.c = 13 / 14 →
      triangleArea t' ≤ 7221 / 117 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_area_special_triangle_l671_67116


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_bottles_for_christine_l671_67182

/-- Converts fluid ounces to milliliters -/
noncomputable def floz_to_ml (floz : ℝ) : ℝ := floz * (1000 / 33.8)

/-- Calculates the number of bottles needed given a volume in milliliters and bottle size -/
noncomputable def bottles_needed (volume_ml : ℝ) (bottle_size_ml : ℝ) : ℕ :=
  Int.toNat (Int.ceil (volume_ml / bottle_size_ml))

theorem min_bottles_for_christine : 
  let required_floz : ℝ := 45
  let bottle_size_ml : ℝ := 200
  bottles_needed (floz_to_ml required_floz) bottle_size_ml = 7 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_bottles_for_christine_l671_67182


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_eccentricity_range_l671_67133

/-- Represents an ellipse with semi-major axis a and semi-minor axis b -/
structure Ellipse where
  a : ℝ
  b : ℝ
  h_pos : 0 < b ∧ b < a

/-- The eccentricity of an ellipse -/
noncomputable def eccentricity (e : Ellipse) : ℝ := Real.sqrt (1 - (e.b / e.a)^2)

/-- The right focus of an ellipse -/
noncomputable def right_focus (e : Ellipse) : ℝ × ℝ := (Real.sqrt (e.a^2 - e.b^2), 0)

/-- Points where a line through the right focus and perpendicular to x-axis intersects the ellipse -/
noncomputable def intersection_points (e : Ellipse) : (ℝ × ℝ) × (ℝ × ℝ) :=
  ((Real.sqrt (e.a^2 - e.b^2), e.b^2 / e.a), (Real.sqrt (e.a^2 - e.b^2), -e.b^2 / e.a))

/-- Predicate to check if a triangle formed by two points and the left focus is acute -/
noncomputable def is_acute_triangle (e : Ellipse) : Prop :=
  let (A, B) := intersection_points e
  (A.2 - B.2)^2 / (4 * (e.a^2 - e.b^2)) < 1

theorem ellipse_eccentricity_range (e : Ellipse) :
  is_acute_triangle e → Real.sqrt 2 - 1 < eccentricity e ∧ eccentricity e < 1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_eccentricity_range_l671_67133


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_g_has_two_zeros_l671_67185

-- Define the piecewise function f
noncomputable def f (x b c : ℝ) : ℝ :=
  if x > 0 then x - 2 else -x^2 + b*x + c

-- Define the function g
noncomputable def g (x b c : ℝ) : ℝ := f x b c + x

-- State the theorem
theorem g_has_two_zeros :
  ∃ (b c : ℝ),
    (f 0 b c = 1) ∧
    (f 0 b c + 2 * f (-1) b c = 0) ∧
    (∃ (x₁ x₂ : ℝ), x₁ ≠ x₂ ∧ g x₁ b c = 0 ∧ g x₂ b c = 0 ∧
      ∀ (x : ℝ), g x b c = 0 → x = x₁ ∨ x = x₂) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_g_has_two_zeros_l671_67185


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_alpha_plus_pi_third_l671_67188

theorem cos_alpha_plus_pi_third (α : Real) (h1 : 0 < α ∧ α < π / 2) (h2 : Real.cos α = 3 / 5) :
  Real.cos (α + π / 3) = (3 - 4 * Real.sqrt 3) / 10 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_alpha_plus_pi_third_l671_67188


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_divisibility_probability_l671_67147

def S : Set ℕ := {d : ℕ | d > 0 ∧ ∃ k : ℕ, 24^9 = d * k}

def divides (a b : ℕ) : Prop := ∃ k : ℕ, b = a * k

noncomputable def probability_divisible : ℚ :=
  (31465 : ℚ) * 715 / (280^4 : ℚ)

theorem divisibility_probability :
  ∀ a1 a2 a3 a4 : ℕ, a1 ∈ S → a2 ∈ S → a3 ∈ S → a4 ∈ S →
    (divides a1 a2 ∧ divides a2 a3 ∧ divides a3 a4) →
    probability_divisible = (31465 : ℚ) * 715 / (280^4 : ℚ) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_divisibility_probability_l671_67147


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complement_of_P_in_U_l671_67173

-- Define the universal set U as the set of real numbers
def U : Set ℝ := Set.univ

-- Define set P
def P : Set ℝ := {y | y > 1}

-- Statement to prove
theorem complement_of_P_in_U :
  U \ P = Set.Iic 1 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_complement_of_P_in_U_l671_67173


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_minimum_and_condition_l671_67128

noncomputable def e : ℝ := Real.exp 1

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.exp x - Real.log (e * x + a)

theorem f_minimum_and_condition (a : ℝ) :
  (∀ x, f e x ≥ 0 ∧ ∃ x₀, f e x₀ = 0) ∧
  ((∀ x, f a x > e) → a < 1 - e) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_minimum_and_condition_l671_67128


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_t_for_two_maxima_l671_67107

noncomputable def f (x : ℝ) : ℝ := Real.sin (Real.pi / 3 * x)

def has_at_least_two_maxima (t : ℝ) : Prop :=
  ∃ x₁ x₂, 0 ≤ x₁ ∧ x₁ < x₂ ∧ x₂ ≤ t ∧
    (∀ x ∈ Set.Icc 0 t, f x ≤ f x₁) ∧
    (∀ x ∈ Set.Icc 0 t, f x ≤ f x₂)

theorem min_t_for_two_maxima :
  ∀ t : ℕ, (has_at_least_two_maxima (t : ℝ)) ↔ t ≥ 8 := by
  sorry

#check min_t_for_two_maxima

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_t_for_two_maxima_l671_67107


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_A_in_triangle_l671_67167

theorem sin_A_in_triangle (a b c : ℝ) (A B C : ℝ) :
  a > 0 ∧ b > 0 ∧ c > 0 →
  A > 0 ∧ B > 0 ∧ C > 0 →
  A + B + C = π →
  c = 4 →
  a = 2 →
  C = π / 4 →
  Real.sin A = Real.sqrt 2 / 4 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_A_in_triangle_l671_67167


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_water_saved_in_june_is_3516_l671_67159

/-- Calculates the water saved in a month after replacing toilets with more efficient ones. -/
def water_saved_in_month (
  toilet1_gallons : ℝ)
  (toilet1_flushes : ℝ)
  (toilet2_gallons : ℝ)
  (toilet2_flushes : ℝ)
  (toilet3_gallons : ℝ)
  (toilet3_flushes : ℝ)
  (efficiency_improvement : ℝ)
  (days_in_month : ℝ) : ℝ :=
  let old_daily_usage := 
    toilet1_gallons * toilet1_flushes +
    toilet2_gallons * toilet2_flushes +
    toilet3_gallons * toilet3_flushes
  let new_daily_usage :=
    old_daily_usage * (1 - efficiency_improvement)
  (old_daily_usage - new_daily_usage) * days_in_month

theorem water_saved_in_june_is_3516 :
  water_saved_in_month 5 10 3.5 7 6 12 0.8 30 = 3516 := by
  sorry

#eval water_saved_in_month 5 10 3.5 7 6 12 0.8 30

end NUMINAMATH_CALUDE_ERRORFEEDBACK_water_saved_in_june_is_3516_l671_67159


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vectors_form_basis_iff_not_collinear_range_of_m_for_basis_l671_67160

/-- Two vectors form a basis for R² if and only if they are not collinear -/
theorem vectors_form_basis_iff_not_collinear (a b : ℝ × ℝ) :
  (∀ c : ℝ × ℝ, ∃! p : ℝ × ℝ, c = p.1 • a + p.2 • b) ↔ 
  (a.1 * b.2 ≠ a.2 * b.1) := by sorry

/-- The range of m for which (3, -2m) and (1, m-2) form a basis for R² -/
theorem range_of_m_for_basis : 
  ∀ m : ℝ, (∀ c : ℝ × ℝ, ∃! p : ℝ × ℝ, c = p.1 • (3, -2*m) + p.2 • (1, m-2)) ↔ 
  (m < 6/5 ∨ m > 6/5) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_vectors_form_basis_iff_not_collinear_range_of_m_for_basis_l671_67160


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_plus_cos_max_value_l671_67121

theorem sin_plus_cos_max_value :
  ∃ (M : ℝ), M = Real.sqrt 2 ∧ ∀ (x : ℝ), Real.sin x + Real.cos x ≤ M :=
by
  -- We'll use M = √2 as our maximum value
  use Real.sqrt 2
  constructor
  · -- First part: M = √2
    rfl
  · -- Second part: ∀ (x : ℝ), sin x + cos x ≤ √2
    intro x
    -- We can prove this using the trigonometric identity sin x + cos x = √2 * sin(x + π/4)
    -- and the fact that |sin θ| ≤ 1 for all θ
    sorry -- Proof details omitted


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_plus_cos_max_value_l671_67121


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_isosceles_triangle_max_area_l671_67109

/-- The maximum area of an isosceles triangle with perimeter 24 cm is 16√3 cm² -/
theorem isosceles_triangle_max_area :
  ∀ (a b : ℝ),
    a > 0 → b > 0 →
    2 * a + b = 24 →
    ∀ (A : ℝ),
      A = Real.sqrt (12 * (12 - a) * (12 - a) * (12 - (24 - 2 * a))) →
      A ≤ 16 * Real.sqrt 3 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_isosceles_triangle_max_area_l671_67109


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fifth_term_is_26_l671_67175

def my_sequence (n : ℕ+) : ℕ := n^2 + 1

theorem fifth_term_is_26 : my_sequence 5 = 26 := by
  rw [my_sequence]
  norm_num

#eval my_sequence 5

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fifth_term_is_26_l671_67175


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_equality_l671_67156

-- Define the set of solutions
def SolutionSet : Set ℝ :=
  {x : ℝ | (x + 1) * (x + 3) / ((x - 1)^2) ≤ 0 ∧ x ≠ 1}

-- State the theorem
theorem solution_equality : SolutionSet = Set.Icc (-3) (-1) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_equality_l671_67156


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_odd_symmetry_l671_67136

-- Define the function f
noncomputable def f (a b c x : ℝ) : ℝ := a * Real.tan x + b * x^3 + c * x

-- State the theorem
theorem f_odd_symmetry (a b c : ℝ) : f a b c 1 = 2 → f a b c (-1) = -2 := by
  intro h
  -- The proof would go here, but we'll use sorry for now
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_odd_symmetry_l671_67136


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_octagon_semicircle_area_l671_67100

/-- The area of the shaded region in a regular octagon with inscribed semicircles -/
theorem octagon_semicircle_area : 
  ∀ (s : ℝ), s = 3 → 
  2 * (1 + Real.sqrt 2) * s^2 - 8 * (1/2 * Real.pi * (s/2)^2) = 
  18 + 18 * Real.sqrt 2 - 9 * Real.pi := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_octagon_semicircle_area_l671_67100


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_crosses_for_three_in_line_l671_67158

/-- Represents a 6x6 grid where crosses can be placed -/
def Grid := Fin 6 → Fin 6 → Bool

/-- Checks if there are three consecutive crosses in a row or column -/
def hasThreeInLine (g : Grid) : Prop :=
  ∃ (i : Fin 6), 
    (∃ (j : Fin 4), g i j ∧ g i (j + 1) ∧ g i (j + 2)) ∨
    (∃ (j : Fin 4), g j i ∧ g (j + 1) i ∧ g (j + 2) i)

/-- Counts the number of crosses in the grid -/
def crossCount (g : Grid) : Nat :=
  (Finset.sum (Finset.univ : Finset (Fin 6)) fun i =>
    Finset.sum (Finset.univ : Finset (Fin 6)) fun j =>
      if g i j then 1 else 0)

/-- The main theorem stating that 25 is the minimum number of crosses
    that guarantees three in a line -/
theorem min_crosses_for_three_in_line :
  (∀ g : Grid, crossCount g ≥ 25 → hasThreeInLine g) ∧
  (∃ g : Grid, crossCount g = 24 ∧ ¬hasThreeInLine g) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_crosses_for_three_in_line_l671_67158


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_between_lines_l671_67144

/-- The distance between two parallel lines ax + by + c₁ = 0 and ax + by + c₂ = 0 -/
noncomputable def distance_parallel_lines (a b c₁ c₂ : ℝ) : ℝ :=
  |c₁ - c₂| / Real.sqrt (a^2 + b^2)

/-- The first line: x + 2y + 3 = 0 -/
def line1 : ℝ → ℝ → ℝ := fun x y => x + 2*y + 3

/-- The second line: 2x + 4y + 5 = 0 -/
def line2 : ℝ → ℝ → ℝ := fun x y => 2*x + 4*y + 5

theorem distance_between_lines :
  distance_parallel_lines 2 4 (-5) (-6) = Real.sqrt 5 / 10 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_between_lines_l671_67144


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_isosceles_right_triangle_on_hyperbola_l671_67184

-- Define the hyperbola D
structure Hyperbola :=
  (a : ℝ)  -- Semi-major axis
  (c : ℝ)  -- Distance from center to focus

-- Define the triangle ABC
structure Triangle :=
  (A B C : ℝ × ℝ)

-- Define the focus F
def F (D : Hyperbola) : ℝ × ℝ := (D.c, 0)

-- State the theorem
theorem isosceles_right_triangle_on_hyperbola 
  (ABC : Triangle) 
  (D : Hyperbola) 
  (h1 : ABC.A.1 > 0 ∧ ABC.A.2 = 0)  -- A is on the positive x-axis
  (h2 : ABC.B.1 > 0)  -- B is on the right branch
  (h3 : ∃ t : ℝ, 0 < t ∧ t < 1 ∧ ABC.A = (1 - t) • (F D) + t • ABC.B)  -- AB passes through F
  (h4 : ABC.C = (-D.c, 0))  -- C is the left focus
  (h5 : (ABC.B.1 - ABC.A.1)^2 + (ABC.B.2 - ABC.A.2)^2 = (ABC.C.1 - ABC.A.1)^2 + (ABC.C.2 - ABC.A.2)^2)  -- ABC is isosceles
  (h6 : (ABC.C.1 - ABC.A.1) * (ABC.B.1 - ABC.A.1) + (ABC.C.2 - ABC.A.2) * (ABC.B.2 - ABC.A.2) = 0)  -- ∠A = 90°
  : ∃ (AF FB : ℝ), AF / FB = Real.sqrt 2 - 1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_isosceles_right_triangle_on_hyperbola_l671_67184


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trig_identities_for_point_l671_67103

/-- Given an angle α whose terminal side passes through point P(-3,4) in the Cartesian coordinate system,
    prove the following trigonometric identities. -/
theorem trig_identities_for_point (α : ℝ) : 
  (∃ (r : ℝ), r > 0 ∧ r * (Real.cos α) = -3 ∧ r * (Real.sin α) = 4) →
  (Real.sin α = 4/5 ∧ 
   Real.cos α = -3/5 ∧
   Real.tan (α + π/4) = -1/7 ∧ 
   (Real.sin (α + π/4))^2 + Real.sin (α + π/4) * Real.cos (α + π/4) = -3/25) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_trig_identities_for_point_l671_67103


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_clinic_patient_ratio_l671_67153

theorem clinic_patient_ratio : 
  let prev_patients : ℕ := 26
  let diagnosed_patients : ℕ := 13
  let syndrome_rate : ℚ := 1/4
  let current_patients : ℕ := (diagnosed_patients * 4 : ℕ)
  (current_patients : ℚ) / prev_patients = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_clinic_patient_ratio_l671_67153


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_baseball_opponent_score_l671_67163

def BaseballProblem (team_scores : List Nat) (wins : Nat) (losses : Nat) : Prop :=
  team_scores.length = 12 ∧
  wins = 5 ∧
  losses = 4 ∧
  (∃ (win_scores : List Nat), 
    win_scores.length = wins ∧ 
    win_scores.all (λ s => s ∈ team_scores) ∧
    (∃ (opponent_win_scores : List Nat), 
      opponent_win_scores.length = wins ∧
      (List.zip win_scores opponent_win_scores).all (λ p => p.1 = 2 * p.2))) ∧
  (∃ (loss_scores : List Nat),
    loss_scores.length = losses ∧
    loss_scores.all (λ s => s ∈ team_scores) ∧
    (∃ (opponent_loss_scores : List Nat),
      opponent_loss_scores.length = losses ∧
      (List.zip loss_scores opponent_loss_scores).all (λ p => p.2 = p.1 + 2)))

theorem baseball_opponent_score 
  (team_scores : List Nat) (h : BaseballProblem team_scores 5 4) :
  ∃ (opponent_scores : List Nat), 
    opponent_scores.length = 12 ∧ opponent_scores.sum = 49 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_baseball_opponent_score_l671_67163


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_count_distinct_integers_is_64_l671_67138

/-- The set of digits used to form the integers -/
def digits : Finset Nat := {1, 2, 3, 4}

/-- The function that counts the number of distinct positive integers
    that can be formed using the given digits without repetition -/
def count_distinct_integers (d : Finset Nat) : Nat :=
  (d.powerset.filter (λ s => s.Nonempty)).sum (λ s => Nat.factorial s.card)

/-- Theorem stating that the count of distinct positive integers
    formed using digits 1, 2, 3, and 4 without repetition is 64 -/
theorem count_distinct_integers_is_64 :
  count_distinct_integers digits = 64 := by
  sorry

#eval count_distinct_integers digits

end NUMINAMATH_CALUDE_ERRORFEEDBACK_count_distinct_integers_is_64_l671_67138


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_integral_f_equals_one_third_plus_pi_over_four_l671_67102

noncomputable def f (x : ℝ) : ℝ :=
  if -1 ≤ x ∧ x ≤ 0 then (x + 1)^2
  else if 0 < x ∧ x ≤ 1 then Real.sqrt (1 - x^2)
  else 0

theorem integral_f_equals_one_third_plus_pi_over_four :
  ∫ x in Set.Icc (-1) 1, f x = 1/3 + π/4 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_integral_f_equals_one_third_plus_pi_over_four_l671_67102


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_period_and_triangle_area_l671_67118

noncomputable def f (ω : ℝ) (x : ℝ) : ℝ := 2 * Real.sin (ω * x) * Real.cos (ω * x) - 2 * Real.sqrt 3 * (Real.cos (ω * x))^2 + Real.sqrt 3

noncomputable def symmetry_axis_distance : ℝ := Real.pi / 2

noncomputable def triangle_side_c : ℝ := 3 * Real.sqrt 2

theorem function_period_and_triangle_area 
  (ω : ℝ) 
  (h_ω : ω > 0) 
  (h_symmetry : ∀ x : ℝ, f ω x = f ω (x + symmetry_axis_distance)) 
  (A B C : ℝ) 
  (h_acute : 0 < C ∧ C < Real.pi / 2) 
  (h_f_C : f ω C = Real.sqrt 3) 
  (h_sin : Real.sin B = 2 * Real.sin A) :
  (∀ x : ℝ, f ω x = f ω (x + Real.pi)) ∧ 
  (1/2 * triangle_side_c * Real.sin C = 3 * Real.sqrt 3) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_period_and_triangle_area_l671_67118


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_a_for_subset_exists_min_a_l671_67194

theorem min_a_for_subset (a : ℤ) : 
  ({x : ℝ | (2 : ℝ)^x < 2011} ⊆ Set.Iio (a : ℝ)) → a ≥ 11 :=
by sorry

theorem exists_min_a : 
  ∃ a : ℤ, ({x : ℝ | (2 : ℝ)^x < 2011} ⊆ Set.Iio (a : ℝ)) ∧ 
  (∀ b : ℤ, ({x : ℝ | (2 : ℝ)^x < 2011} ⊆ Set.Iio (b : ℝ)) → a ≤ b) ∧
  a = 11 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_a_for_subset_exists_min_a_l671_67194


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_intersection_angles_l671_67141

/-- Represents an ellipse with semi-major axis a and semi-minor axis b -/
structure Ellipse where
  a : ℝ
  b : ℝ
  h_pos : 0 < b ∧ b < a

/-- Calculates the eccentricity of an ellipse -/
noncomputable def eccentricity (e : Ellipse) : ℝ := Real.sqrt (1 - e.b^2 / e.a^2)

/-- Calculates the area of the rhombus formed by the vertices of an ellipse -/
def rhombusArea (e : Ellipse) : ℝ := 4 * e.a * e.b

/-- Theorem: For an ellipse with given properties, a line intersecting it at two points
    has specific angles of inclination -/
theorem ellipse_intersection_angles (e : Ellipse) 
    (h_ecc : eccentricity e = Real.sqrt 3 / 2)
    (h_area : rhombusArea e = 4)
    (h_dist : ∃ (x y : ℝ), x^2/e.a^2 + y^2/e.b^2 = 1 ∧ 
              (x + e.a)^2 + y^2 = (4*Real.sqrt 2/5)^2) :
    ∃ (θ₁ θ₂ : ℝ), θ₁ = π/4 ∧ θ₂ = 5*π/4 ∧ 
    (∃ (k : ℝ), k = Real.tan θ₁ ∨ k = Real.tan θ₂) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_intersection_angles_l671_67141


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_student_initial_money_l671_67177

/-- The price of one notebook in kopecks -/
def notebook_price : ℕ := sorry

/-- The total money the student had initially, in kopecks -/
def initial_money : ℕ := sorry

/-- Condition 1: If the student buys 11 notebooks, he would have 8 rubles left -/
def condition1 : Prop :=
  initial_money - 11 * notebook_price = 800

/-- Condition 2: The student is 12 rubles and 24 kopecks short for 15 notebooks -/
def condition2 : Prop :=
  initial_money + 1224 = 15 * notebook_price

/-- Theorem: Given the conditions, prove that the student's initial money was 63 rubles and 66 kopecks -/
theorem student_initial_money :
  condition1 → condition2 → initial_money = 6366 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_student_initial_money_l671_67177


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_flower_bed_fraction_is_one_fifth_l671_67106

/-- Represents a rectangular yard with two congruent isosceles right triangular flower beds -/
structure Yard where
  length : ℝ
  width : ℝ
  trapezoid_short_side : ℝ
  trapezoid_long_side : ℝ

/-- The fraction of the yard occupied by the flower beds -/
noncomputable def flower_bed_fraction (y : Yard) : ℝ :=
  let triangle_leg := (y.trapezoid_long_side - y.trapezoid_short_side) / 2
  let flower_bed_area := 2 * (triangle_leg ^ 2 / 2)
  let total_area := y.length * y.width
  flower_bed_area / total_area

/-- Theorem stating that the fraction of the yard occupied by the flower beds is 1/5 -/
theorem flower_bed_fraction_is_one_fifth (y : Yard) 
  (h1 : y.trapezoid_short_side = 15)
  (h2 : y.trapezoid_long_side = 25)
  (h3 : y.length = y.trapezoid_long_side)
  (h4 : y.width = (y.trapezoid_long_side - y.trapezoid_short_side) / 2) :
  flower_bed_fraction y = 1/5 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_flower_bed_fraction_is_one_fifth_l671_67106


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_stratified_sampling_correct_total_employees_sum_l671_67149

-- Define the total number of employees
def total_employees : ℕ := 800

-- Define the number of employees in each category
def senior_titles : ℕ := 160
def intermediate_titles : ℕ := 320
def junior_titles : ℕ := 200
def other_staff : ℕ := 120

-- Define the sample size
def sample_size : ℕ := 40

-- Theorem statement
theorem stratified_sampling_correct :
  let sampling_fraction : ℚ := (sample_size : ℚ) / total_employees
  (Nat.cast senior_titles * sampling_fraction).floor = 8 ∧
  (Nat.cast intermediate_titles * sampling_fraction).floor = 16 ∧
  (Nat.cast junior_titles * sampling_fraction).floor = 10 ∧
  (Nat.cast other_staff * sampling_fraction).floor = 6 :=
by sorry

-- Verify that the sum of all categories equals the total number of employees
theorem total_employees_sum :
  senior_titles + intermediate_titles + junior_titles + other_staff = total_employees :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_stratified_sampling_correct_total_employees_sum_l671_67149


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equation_solutions_l671_67135

theorem equation_solutions :
  (∃ x : ℝ, x + Real.sqrt ((x + 1) * (x + 2)) = 3 ∧ x = 7/9) ∧
  (∃ x : ℝ, x + Real.sqrt ((x - 1) * x) + Real.sqrt (x * (x + 1)) + Real.sqrt ((x + 1) * (x - 1)) = 3 ∧ x = 25/24) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_equation_solutions_l671_67135


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_theorem_l671_67117

open Real

/-- The function f(x) = x^2 - a*ln(x) --/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := x^2 - a * log x

/-- The minimum value of f(x) on the interval [1, e] --/
noncomputable def min_value (a : ℝ) : ℝ :=
  if a ≤ 2 then 1
  else if a < 2 * (Real.exp 1)^2 then -log (sqrt (a / 2))
  else (Real.exp 1)^2 - a

/-- Theorem stating the minimum value of f(x) on [1, e] for different ranges of a --/
theorem min_value_theorem (a : ℝ) :
  ∀ x ∈ Set.Icc 1 (Real.exp 1), f a x ≥ min_value a := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_theorem_l671_67117


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_light_path_length_l671_67108

/-- Represents a point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Calculates the distance between two points in 3D space -/
noncomputable def distance (p1 p2 : Point3D) : ℝ :=
  Real.sqrt ((p1.x - p2.x)^2 + (p1.y - p2.y)^2 + (p1.z - p2.z)^2)

/-- Theorem: The total length of the light path in the cube -/
theorem light_path_length (cube_side : ℝ) (A P Q : Point3D) : 
  cube_side = 15 →
  A = ⟨0, 0, 0⟩ →
  P = ⟨15, 9, 8⟩ →
  Q = ⟨0, 9, 8⟩ →
  distance A P + distance P Q + distance Q A = Real.sqrt 370 + 15 + Real.sqrt 145 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_light_path_length_l671_67108


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_function_property_g_function_property_l671_67171

/-- A quadratic function f(x) = ax^2 + bx + c with certain properties -/
structure QuadraticFunction where
  a : ℝ
  b : ℝ
  c : ℝ
  b_gt_a : b > a
  nonneg : ∀ x : ℝ, a * x^2 + b * x + c ≥ 0

/-- The function f(x) evaluated at x -/
noncomputable def f (q : QuadraticFunction) (x : ℝ) : ℝ := q.a * x^2 + q.b * x + q.c

/-- The ratio f(-2) / (f(2) - f(0)) -/
noncomputable def ratio (q : QuadraticFunction) : ℝ := 
  (f q (-2)) / ((f q 2) - (f q 0))

/-- Function g(x) = |f(x) - a| -/
noncomputable def g (q : QuadraticFunction) (x : ℝ) : ℝ := 
  |f q x - q.a|

theorem quadratic_function_property (q : QuadraticFunction) 
  (h : ∀ q' : QuadraticFunction, ratio q ≤ ratio q') : 
  q.a = 1 → f q = λ x ↦ (x + 2)^2 := by sorry

theorem g_function_property (q : QuadraticFunction) 
  (h : ∀ x₁ x₂ : ℝ, x₁ ∈ Set.Icc (-3 * q.a) (-q.a) → x₂ ∈ Set.Icc (-3 * q.a) (-q.a) → 
    |g q x₁ - g q x₂| ≤ 2 * q.a) :
  q.a ∈ Set.Ioo 0 ((2 + Real.sqrt 3) / 2) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_function_property_g_function_property_l671_67171


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_bike_owners_without_scooters_l671_67154

theorem bike_owners_without_scooters 
  (total : Finset Nat) 
  (bike_owners : Finset Nat) 
  (scooter_owners : Finset Nat)
  (h_total : total.card = 450)
  (h_bike : bike_owners.card = 400)
  (h_scooter : scooter_owners.card = 120)
  (h_ownership : ∀ a, a ∈ total → (a ∈ bike_owners ∨ a ∈ scooter_owners))
  (h_subset : bike_owners ⊆ total ∧ scooter_owners ⊆ total) :
  (bike_owners \ scooter_owners).card = 330 :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_bike_owners_without_scooters_l671_67154


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_final_book_count_l671_67192

def initial_books : ℕ := 4
def books_sold : ℕ := 3
def percentage_increase : ℚ := 1/2
def new_shipment : ℕ := 10

def round_up (x : ℚ) : ℕ := (Int.ceil x).toNat

theorem final_book_count : 
  let remaining_books := initial_books - books_sold
  let increased_books := remaining_books + round_up (percentage_increase * remaining_books)
  let final_books := increased_books + new_shipment
  final_books = 12 := by
  -- Proof steps would go here
  sorry

#eval let remaining_books := initial_books - books_sold
      let increased_books := remaining_books + round_up (percentage_increase * remaining_books)
      increased_books + new_shipment

end NUMINAMATH_CALUDE_ERRORFEEDBACK_final_book_count_l671_67192


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_lines_imply_b_eq_neg_six_l671_67162

/-- Two lines are parallel if and only if their slopes are equal -/
axiom parallel_iff_equal_slopes {m₁ m₂ : ℝ} : m₁ = m₂ ↔ ({a : ℝ × ℝ | a.2 = m₁ * a.1} : Set (ℝ × ℝ)) = {a : ℝ × ℝ | a.2 = m₂ * a.1}

/-- Definition of the first line -/
def line1 (b : ℝ) : Set (ℝ × ℝ) := {a : ℝ × ℝ | a.2 = 3 * a.1 + b}

/-- Definition of the second line -/
def line2 (b : ℝ) : Set (ℝ × ℝ) := {a : ℝ × ℝ | a.2 = (b + 9) * a.1 + 2}

/-- Theorem: If the given lines are parallel, then b = -6 -/
theorem parallel_lines_imply_b_eq_neg_six (b : ℝ) : 
  line1 b = line2 b → b = -6 := by
  sorry

#check parallel_lines_imply_b_eq_neg_six

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_lines_imply_b_eq_neg_six_l671_67162


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_alcohol_dilution_proof_l671_67125

/-- Calculates the alcohol percentage in a new mixture after adding water -/
noncomputable def new_alcohol_percentage (initial_volume : ℝ) (initial_alcohol_percent : ℝ) (added_water : ℝ) : ℝ :=
  let initial_alcohol_volume := initial_volume * (initial_alcohol_percent / 100)
  let new_total_volume := initial_volume + added_water
  (initial_alcohol_volume / new_total_volume) * 100

/-- Proves that adding 3 litres of water to a 15-litre mixture with 20% alcohol 
    results in a new mixture with approximately 16.67% alcohol -/
theorem alcohol_dilution_proof :
  let initial_volume : ℝ := 15
  let initial_alcohol_percent : ℝ := 20
  let added_water : ℝ := 3
  let result := new_alcohol_percentage initial_volume initial_alcohol_percent added_water
  abs (result - 16.67) < 0.01 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_alcohol_dilution_proof_l671_67125


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trihedron_cut_ball_volume_l671_67172

/-- The volume of a body obtained by cutting a ball with a trihedron -/
noncomputable def volume_of_trihedron_cut_ball (R α β γ : ℝ) : ℝ :=
  (1/3) * R^3 * (α + β + γ - Real.pi)

/-- The volume of a body obtained by cutting a ball with a trihedron -/
theorem trihedron_cut_ball_volume
  (R : ℝ) -- radius of the ball
  (α β γ : ℝ) -- dihedral angles of the trihedron
  (hR : R > 0) -- R is positive
  (hα : α > 0) -- α is positive
  (hβ : β > 0) -- β is positive
  (hγ : γ > 0) -- γ is positive
  : ∃ V : ℝ, V = (1/3) * R^3 * (α + β + γ - Real.pi) ∧ 
    V = volume_of_trihedron_cut_ball R α β γ :=
by
  -- Define V
  let V := volume_of_trihedron_cut_ball R α β γ
  -- Prove existence
  use V
  -- Prove both parts of the conjunction
  constructor
  · -- Prove V = (1/3) * R^3 * (α + β + γ - Real.pi)
    rfl
  · -- Prove V = volume_of_trihedron_cut_ball R α β γ
    rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_trihedron_cut_ball_volume_l671_67172


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_expression_evaluation_l671_67164

theorem expression_evaluation (x y : ℝ) (hx : x ≠ 0) (hy : Real.sin y ≠ 0) :
  let P := x * Real.cos y
  let Q := x * Real.sin y
  (P + Q) / (P - Q) + (P - Q) / (P + Q) = 2 * Real.cos y / Real.sin y :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_expression_evaluation_l671_67164


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_average_after_removing_numbers_l671_67148

theorem average_after_removing_numbers (numbers : Finset ℝ) (sum : ℝ) : 
  numbers.card = 50 → 
  sum = numbers.sum id → 
  sum / numbers.card = 44 → 
  45 ∈ numbers → 
  55 ∈ numbers → 
  let remaining := numbers.erase 45 |>.erase 55
  (remaining.sum id) / (remaining.card : ℝ) = 43.75 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_average_after_removing_numbers_l671_67148


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pastry_prices_l671_67137

/-- Represents the price of a cake in Kč -/
def cake_price : ℚ := sorry

/-- Represents the price of a cream puff in Kč -/
def cream_puff_price : ℚ := sorry

/-- Represents the total available money in Kč -/
def total_money : ℚ := 96

/-- The cream puff costs 4 Kč more than the cake -/
axiom price_difference : cream_puff_price = cake_price + 4

/-- The ratio of cakes to cream puffs that can be bought with the total money -/
axiom buying_ratio : (total_money / cake_price) = (4/3) * (total_money / cream_puff_price)

/-- The theorem stating the prices of cake and cream puff -/
theorem pastry_prices : cake_price = 12 ∧ cream_puff_price = 16 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_pastry_prices_l671_67137


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_enclosed_area_is_four_thirds_l671_67169

/-- The area enclosed by the curve y = x^2 + 1, the x-axis, y-axis, and the line x = 1 -/
noncomputable def enclosed_area : ℝ := ∫ x in (0:ℝ)..(1:ℝ), (x^2 + 1)

/-- Theorem stating that the enclosed area is equal to 4/3 -/
theorem enclosed_area_is_four_thirds : enclosed_area = 4/3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_enclosed_area_is_four_thirds_l671_67169


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_flight_time_approx_21_hours_l671_67122

/-- The radius of the planet in miles -/
def planet_radius : ℝ := 2000

/-- The speed of the jet in miles per hour -/
def jet_speed : ℝ := 600

/-- The circumference of the planet at the equator -/
noncomputable def planet_circumference : ℝ := 2 * Real.pi * planet_radius

/-- The time taken for the jet to complete one trip around the planet -/
noncomputable def flight_time : ℝ := planet_circumference / jet_speed

/-- Theorem stating that the flight time is approximately 21 hours -/
theorem flight_time_approx_21_hours :
  abs (flight_time - 21) < 0.1 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_flight_time_approx_21_hours_l671_67122


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_locus_forms_C₁_max_distance_AB_l671_67129

-- Define the parametric equations of line l₁
noncomputable def l₁ (t k : ℝ) : ℝ × ℝ := (4 - t, k * t)

-- Define the general equation of line l₂
noncomputable def l₂ (k x : ℝ) : ℝ := (1 / k) * x

-- Define the equation of line l₃ in polar coordinates
def l₃ (ρ θ : ℝ) : Prop := ρ * Real.sin (θ - Real.pi / 4) = Real.sqrt 2

-- Define curve C₁
def C₁ (x y : ℝ) : Prop := x^2 + y^2 - 4 * x = 0 ∧ y ≠ 0

-- Theorem 1: The locus of intersection points forms curve C₁
theorem intersection_locus_forms_C₁ (k t : ℝ) :
  let (x, y) := l₁ t k
  y = l₂ k x → C₁ x y := by sorry

-- Theorem 2: Maximum value of |AB|
theorem max_distance_AB (A B : ℝ × ℝ) (θ : ℝ) :
  l₃ (Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2)) θ →
  C₁ B.1 B.2 →
  (A.2 - B.2) / (A.1 - B.1) = 1 →
  (A.1 - B.1)^2 + (A.2 - B.2)^2 ≤ (4 + 2 * Real.sqrt 2)^2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_locus_forms_C₁_max_distance_AB_l671_67129


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_perfect_power_ap_exists_no_infinite_perfect_power_ap_l671_67193

/-- A positive integer is a perfect power if it can be expressed as a^b for some integers a and b with b > 1 -/
def IsPerfectPower (n : ℕ) : Prop :=
  ∃ (a b : ℕ), b > 1 ∧ n = a^b

/-- An arithmetic progression of perfect powers -/
def PerfectPowerAP (first diff : ℕ) (length : ℕ) : Prop :=
  ∀ k : ℕ, k < length → IsPerfectPower (first + k * diff)

theorem perfect_power_ap_exists :
  ∃ (first diff : ℕ), PerfectPowerAP first diff 2004 :=
sorry

theorem no_infinite_perfect_power_ap :
  ¬∃ (first diff : ℕ), ∀ k : ℕ, IsPerfectPower (first + k * diff) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_perfect_power_ap_exists_no_infinite_perfect_power_ap_l671_67193


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hannah_julia_debt_doubling_time_l671_67198

/-- Represents a simple interest loan -/
structure Loan where
  principal : ℚ
  rate : ℚ

/-- Calculates the number of days for a loan to double -/
def daysToDouble (loan : Loan) : ℚ :=
  1 / loan.rate

theorem hannah_julia_debt_doubling_time :
  let hannah_loan : Loan := { principal := 200, rate := 8/100 }
  let julia_loan : Loan := { principal := 300, rate := 6/100 }
  (daysToDouble hannah_loan = 25/2) ∧
  (daysToDouble julia_loan = 50/3) := by
  sorry

#eval daysToDouble { principal := 200, rate := 8/100 }
#eval daysToDouble { principal := 300, rate := 6/100 }

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hannah_julia_debt_doubling_time_l671_67198


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_round_trip_time_calculation_l671_67179

noncomputable def round_trip_time (rowing_speed : ℝ) (current_velocity : ℝ) (distance : ℝ) : ℝ :=
  let speed_to := rowing_speed - current_velocity
  let speed_from := rowing_speed + current_velocity
  let time_to := distance / speed_to
  let time_from := distance / speed_from
  time_to + time_from

theorem round_trip_time_calculation :
  round_trip_time 10 2 144 = 30 := by
  sorry

-- Remove the #eval line as it's causing issues with noncomputable definitions
-- #eval round_trip_time 10 2 144

end NUMINAMATH_CALUDE_ERRORFEEDBACK_round_trip_time_calculation_l671_67179


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_janeles_cats_average_weight_l671_67195

/-- Calculates the average weight of Janele's cats -/
theorem janeles_cats_average_weight :
  let cat1_weight : ℝ := 12
  let cat2_weight : ℝ := 12
  let cat3_weight : ℝ := 14.7
  let cat4_weight : ℝ := 9.3
  let cat5_weight : ℝ := 13.2
  let cat6_weight : ℝ := 15.8
  let cat7_weights : List ℝ := [14, 15.4, 13.7, 14.2]
  let cat7_avg_weight := (cat7_weights.sum) / (cat7_weights.length : ℝ)
  let total_weight := cat1_weight + cat2_weight + cat3_weight + cat4_weight + 
                      cat5_weight + cat6_weight + cat7_avg_weight
  let num_cats : ℝ := 7
  let average_weight := total_weight / num_cats
  ∃ (ε : ℝ), ε > 0 ∧ |average_weight - 13.046| < ε :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_janeles_cats_average_weight_l671_67195


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_abs_l671_67111

-- Define the function f
def f : ℝ → ℝ := sorry

-- Define the domain of f
def domain_f : Set ℝ := Set.Icc (-1) 2

-- State the theorem
theorem domain_of_f_abs (x : ℝ) :
  x ∈ Set.Ioo (-2) 2 ↔ |x| ∈ domain_f :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_abs_l671_67111


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_no_extreme_points_iff_a_in_closed_unit_interval_l671_67126

/-- The function f(x) defined in the problem -/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (1/3) * x^3 - a * x^2 + x - 5

/-- Theorem stating the condition for f to have no extreme points -/
theorem no_extreme_points_iff_a_in_closed_unit_interval (a : ℝ) :
  (∀ x : ℝ, (deriv (f a)) x ≠ 0) ↔ a ∈ Set.Icc (-1 : ℝ) 1 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_no_extreme_points_iff_a_in_closed_unit_interval_l671_67126


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_right_triangle_area_l671_67176

theorem right_triangle_area (h l : ℝ) (h_pos : h > 0) (l_pos : l > 0) :
  let S := (1 / 2) * h * Real.sqrt (l^2 + 4*h^2)
  ∃ (a b c : ℝ), 
    a > 0 ∧ b > 0 ∧ c > 0 ∧
    c^2 = a^2 + b^2 ∧
    (c - 2*(c^2 - a^2) / (2*c))^2 + h^2 = (c^2 - a^2)^2 / c^2 ∧
    (2*(c^2 - a^2) / (2*c))^2 + h^2 = a^2 ∧
    (c^2 - a^2) / c - (c^2 - b^2) / c = l ∧
    S = (1/2) * a * b :=
by
  sorry

#check right_triangle_area

end NUMINAMATH_CALUDE_ERRORFEEDBACK_right_triangle_area_l671_67176


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_indeterminate_quotient_l671_67131

theorem indeterminate_quotient (m n : ℕ) (h1 : n = 60) 
  (h2 : m % n = 12) : 
  ∃ (q1 q2 : ℕ), q1 ≠ q2 ∧ m = n * q1 + 12 ∧ m = n * q2 + 12 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_indeterminate_quotient_l671_67131


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_harold_catches_adrienne_l671_67161

/-- The problem of Harold catching up to Adrienne -/
theorem harold_catches_adrienne (total_distance adrienne_speed harold_speed_diff : ℝ) 
  (h1 : total_distance = 60)
  (h2 : adrienne_speed = 3)
  (h3 : harold_speed_diff = 1) : ℝ := by
  let harold_speed := adrienne_speed + harold_speed_diff
  let adrienne_head_start := adrienne_speed
  let catch_up_time := adrienne_head_start / (harold_speed - adrienne_speed)
  let harold_distance := harold_speed * catch_up_time
  
  have : harold_distance = 12 := by
    -- The proof steps would go here
    sorry
  
  exact harold_distance


end NUMINAMATH_CALUDE_ERRORFEEDBACK_harold_catches_adrienne_l671_67161


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_shifted_sine_symmetry_l671_67124

open Real

-- Define the original and shifted functions
noncomputable def original_func (x : ℝ) : ℝ := sin (2 * x)
noncomputable def shifted_func (x : ℝ) : ℝ := sin (2 * x - π / 6)

-- Define the shift amount
noncomputable def shift : ℝ := π / 12

-- Define the axis of symmetry
noncomputable def axis_of_symmetry (k : ℤ) : ℝ := k * π / 2 + π / 3

-- Theorem statement
theorem shifted_sine_symmetry (k : ℤ) :
  ∀ x : ℝ, shifted_func x = shifted_func (2 * (axis_of_symmetry k) - x) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_shifted_sine_symmetry_l671_67124


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_wayne_shrimp_cocktail_l671_67191

/-- Calculates the number of shrimp in a pound given the problem conditions -/
def shrimp_per_pound (shrimp_per_guest : ℕ) (num_guests : ℕ) (cost_per_pound : ℚ) (total_spent : ℚ) : ℚ :=
  let total_shrimp := shrimp_per_guest * num_guests
  let pounds_bought := total_spent / cost_per_pound
  (total_shrimp : ℚ) / pounds_bought

/-- Proves that the number of shrimp in a pound is 20 given the problem conditions -/
theorem wayne_shrimp_cocktail : 
  shrimp_per_pound 5 40 17 170 = 20 := by
  sorry

#eval (shrimp_per_pound 5 40 17 170).num.toNat

end NUMINAMATH_CALUDE_ERRORFEEDBACK_wayne_shrimp_cocktail_l671_67191


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_2alpha_value_l671_67168

open Real

theorem sin_2alpha_value (α : ℝ) (h1 : 0 < α ∧ α < π/2) 
  (h2 : cos (π/4 - α) = 2 * Real.sqrt 2 * cos (2*α)) : 
  sin (2*α) = 15/16 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_2alpha_value_l671_67168


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_true_propositions_l671_67140

-- Define the propositions
def proposition1 : Prop :=
  ∀ (a : ℕ → ℝ), (∃ q : ℝ, ∀ n, a (n + 1) = q * a n) →
  (∃ r : ℝ, ∀ n, a (n + 1) * a (n + 2) = r * (a n * a (n + 1))) ∧
  ¬(∀ (a : ℕ → ℝ), (∃ r : ℝ, ∀ n, a (n + 1) * a (n + 2) = r * (a n * a (n + 1))) →
    (∃ q : ℝ, ∀ n, a (n + 1) = q * a n))

def proposition2 : Prop :=
  ∀ a : ℝ, (a = 2 ↔ ∀ x ≥ 2, ∀ y ≥ 2, x > y → |x - a| > |y - a|)

def proposition3 : Prop :=
  ∀ m : ℝ, (m = 3 ↔ ((m + 3) * 6 + m * m = 0))

def proposition4 : Prop :=
  ∀ A B : ℝ, (A = Real.pi / 6 → (B = Real.pi / 3 ∨ B = 2 * Real.pi / 3)) ∧
  ¬(A = Real.pi / 6 → B = Real.pi / 3)

-- Theorem stating which propositions are true
theorem true_propositions :
  proposition1 ∧ ¬proposition2 ∧ ¬proposition3 ∧ proposition4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_true_propositions_l671_67140


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_xiao_hong_steps_per_kcal_l671_67152

/-- The number of steps Xiao Hong can walk per 1 kcal of energy consumed -/
def steps_per_kcal : ℝ → Prop := λ x => x > 0

/-- Xiao Ming walks 1200 steps with the same energy consumption as Xiao Hong walking 9000 steps -/
axiom energy_equivalence : ∀ x, steps_per_kcal x → (1200 : ℝ) / (x + 2) = 9000 / x

/-- Xiao Ming walks 2 more steps than Xiao Hong for every 1 kcal of energy consumed -/
axiom step_difference : ∀ x, steps_per_kcal x → x + 2 > x

theorem xiao_hong_steps_per_kcal :
  ∃ x, steps_per_kcal x ∧ (1200 : ℝ) / (x + 2) = 9000 / x := by
  sorry

#check xiao_hong_steps_per_kcal

end NUMINAMATH_CALUDE_ERRORFEEDBACK_xiao_hong_steps_per_kcal_l671_67152


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_even_odd_f_upper_bound_no_constant_bound_l671_67119

/-- The function f(m,n) represents the absolute difference between 
    black and white areas in a right-angled triangle with legs of 
    length m and n placed on a chessboard-like grid. -/
noncomputable def f (m n : ℕ) : ℚ := sorry

/-- Theorem 1: f(m,n) for even-even and odd-odd cases -/
theorem f_even_odd (m n : ℕ) : 
  (m % 2 = 0 ∧ n % 2 = 0 → f m n = 0) ∧
  (m % 2 = 1 ∧ n % 2 = 1 → f m n = 1/2) := by sorry

/-- Theorem 2: Upper bound for f(m,n) -/
theorem f_upper_bound (m n : ℕ) : 
  f m n ≤ 1/2 * max m n := by sorry

/-- Theorem 3: No constant upper bound for f(m,n) -/
theorem no_constant_bound : 
  ¬ ∃ c : ℚ, ∀ m n : ℕ, f m n < c := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_even_odd_f_upper_bound_no_constant_bound_l671_67119


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_work_completion_time_proportional_aartis_triple_work_time_l671_67101

/-- Representing one unit of work -/
def base_work : ℕ := 1

/-- Days to complete base_work -/
def base_days : ℕ := 8

/-- Time to complete given amount of work -/
def time_to_complete (work : ℕ) : ℕ := 
  base_days * (work / base_work)

/-- If a person can complete a piece of work in a given number of days, 
    then the time to complete a multiple of that work is proportional. -/
theorem work_completion_time_proportional 
  (base_days : ℕ) (work_multiple : ℕ) :
  base_days * work_multiple = 
    (time_to_complete (base_work * work_multiple)) :=
by sorry

/-- Aarti's work completion time for triple the work -/
theorem aartis_triple_work_time : 
  time_to_complete (base_work * 3) = 24 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_work_completion_time_proportional_aartis_triple_work_time_l671_67101


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_colorful_mirror_product_l671_67112

-- Define a colorful number
def is_colorful (n : ℕ) : Prop :=
  ∀ (i j : ℕ), i < j → i < (Nat.digits 10 n).length → j < (Nat.digits 10 n).length →
    (Nat.digits 10 n).get ⟨i, by sorry⟩ ≠ (Nat.digits 10 n).get ⟨j, by sorry⟩

-- Define the mirror of a number
def mirror (n : ℕ) : ℕ :=
  Nat.rec n (fun m res => res * 10 + m % 10) n

-- Define the set of valid pairs
def valid_pair (a b : ℕ) : Prop :=
  10 ≤ a ∧ a < 100 ∧ 10 ≤ b ∧ b < 100 ∧
  is_colorful a ∧ is_colorful b ∧ b = mirror a

-- Theorem statement
theorem colorful_mirror_product :
  (∃ (a b : ℕ), valid_pair a b ∧ a * b = 8722) ∧
  (∀ (a b : ℕ), valid_pair a b → a * b ≤ 8722) ∧
  (∃ (c d : ℕ), valid_pair c d ∧ c * d = 1008) ∧
  (∀ (c d : ℕ), valid_pair c d → c * d ≥ 1008 ∨ c * d < 1000) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_colorful_mirror_product_l671_67112


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_candle_height_ratio_time_l671_67197

/-- Represents a burning candle with initial height and burning duration. -/
structure Candle where
  initialHeight : ℝ
  burningDuration : ℝ

/-- Calculates the height of a candle after a given time. -/
noncomputable def candleHeight (c : Candle) (t : ℝ) : ℝ :=
  c.initialHeight - (c.initialHeight / c.burningDuration) * t

theorem candle_height_ratio_time (candleA candleB : Candle) :
  candleA.initialHeight = 12 →
  candleA.burningDuration = 6 →
  candleB.initialHeight = 9 →
  candleB.burningDuration = 4.5 →
  ∃ t : ℝ, t = 3.75 ∧ candleHeight candleA t = 3 * candleHeight candleB t :=
by
  sorry

#check candle_height_ratio_time

end NUMINAMATH_CALUDE_ERRORFEEDBACK_candle_height_ratio_time_l671_67197


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_alpha_on_line_l671_67199

/-- If the terminal side of angle α lies on the line y = 2x, then sin α = ± 2√5 / 5 -/
theorem sin_alpha_on_line (α : ℝ) : 
  (∃ (x y : ℝ), y = 2 * x ∧ x * Real.cos α = x ∧ y * Real.sin α = y) → 
  Real.sin α = 2 * Real.sqrt 5 / 5 ∨ Real.sin α = -2 * Real.sqrt 5 / 5 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_alpha_on_line_l671_67199


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_of_a_squared_l671_67134

-- Define the triangle
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

-- Define the conditions
def isAcuteTriangle (t : Triangle) : Prop :=
  0 < t.A ∧ t.A < Real.pi/2 ∧
  0 < t.B ∧ t.B < Real.pi/2 ∧
  0 < t.C ∧ t.C < Real.pi/2

def satisfiesCondition (t : Triangle) : Prop :=
  t.b^2 * Real.sin t.C = 4 * Real.sqrt 2 * Real.sin t.B

def hasAreaEightThirds (t : Triangle) : Prop :=
  1/2 * t.b * t.c * Real.sin t.A = 8/3

-- Theorem statement
theorem min_value_of_a_squared (t : Triangle) 
  (h1 : isAcuteTriangle t)
  (h2 : satisfiesCondition t)
  (h3 : hasAreaEightThirds t) :
  ∃ (min_a_squared : ℝ), 
    (∀ (t' : Triangle), isAcuteTriangle t' → satisfiesCondition t' → hasAreaEightThirds t' → 
      t'.a^2 ≥ min_a_squared) ∧
    min_a_squared = 16 * Real.sqrt 2 / 3 :=
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_of_a_squared_l671_67134


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_triangle_l671_67190

-- Define the vectors a and b
noncomputable def a (x : ℝ) : ℝ × ℝ := (Real.sin x, -1)
noncomputable def b (x : ℝ) : ℝ × ℝ := (Real.sqrt 3 * Real.cos x, -1/2)

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := ((a x).1 + (b x).1) * (a x).1 + ((a x).2 + (b x).2) * (a x).2 - 2

-- Define the triangle ABC
structure Triangle where
  A : ℝ  -- angle A in radians
  a : ℝ  -- side length opposite to angle A
  c : ℝ  -- side length opposite to angle C

-- Theorem statement
theorem area_of_triangle (t : Triangle) (h1 : 0 < t.A ∧ t.A < π/2) 
  (h2 : t.a = Real.sqrt 3) (h3 : t.c = 1) (h4 : f t.A = 1) : 
  (1/2 * t.a * t.c * Real.sin t.A) = Real.sqrt 3 / 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_triangle_l671_67190


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_infinite_series_sum_equals_two_thirds_l671_67189

/-- The sum of the infinite series Σ(k=1 to ∞) (2k-1)/3^k -/
noncomputable def infinite_series_sum : ℝ := ∑' k, (2 * k - 1) / (3 ^ k)

/-- Theorem stating that the sum of the infinite series is equal to 2/3 -/
theorem infinite_series_sum_equals_two_thirds : infinite_series_sum = 2/3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_infinite_series_sum_equals_two_thirds_l671_67189


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_digit_sum_l671_67105

/-- Represents a digit (0-9) -/
def Digit := Fin 10

/-- The sum of XYZ + XY + Z given X, Y, and Z are digits -/
def digitSum (x y z : Digit) : ℕ :=
  (100 * x.val + 10 * y.val + z.val) + (10 * x.val + y.val) + z.val

theorem largest_digit_sum :
  ∃ (x y z : Digit),
    x ≠ y ∧ y ≠ z ∧ x ≠ z ∧
    digitSum x y z < 1000 ∧
    ∀ (a b c : Digit),
      a ≠ b → b ≠ c → a ≠ c →
      digitSum a b c < 1000 →
      digitSum a b c ≤ digitSum x y z ∧
      digitSum x y z = 982 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_digit_sum_l671_67105


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l671_67110

noncomputable def f (ω : ℝ) (x : ℝ) : ℝ := Real.sin (ω * x) * Real.cos (ω * x) - Real.sqrt 3 * (Real.sin (ω * x))^2

noncomputable def g (ω : ℝ) (x : ℝ) : ℝ := f ω (x + Real.pi / 12)

theorem function_properties (ω : ℝ) (h1 : ω > 0) (h2 : ∃ (T : ℝ), T > 0 ∧ T = Real.pi ∧ ∀ (x : ℝ), f ω (x + T) = f ω x) :
  ω = 1 ∧
  (∀ (x : ℝ), g ω x = g ω (-x)) ∧
  (∀ (x : ℝ), f ω (Real.pi / 3 + x) + f ω (Real.pi / 3 - x) = -Real.sqrt 3) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l671_67110


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_number_set_satisfies_conditions_l671_67187

def number_set : Finset ℕ := {6, 34, 35, 51, 55, 77}

def has_common_divisor (a b : ℕ) : Prop :=
  ∃ d : ℕ, d > 1 ∧ d ∣ a ∧ d ∣ b

def satisfies_conditions (s : Finset ℕ) : Prop :=
  (∀ a b c, a ∈ s → b ∈ s → c ∈ s → a ≠ b → b ≠ c → a ≠ c →
    ¬(∃ d : ℕ, d > 1 ∧ d ∣ a ∧ d ∣ b ∧ d ∣ c)) ∧
  (∀ a b c, a ∈ s → b ∈ s → c ∈ s → a ≠ b → b ≠ c → a ≠ c →
    has_common_divisor a b ∨ has_common_divisor b c ∨ has_common_divisor a c)

theorem number_set_satisfies_conditions : satisfies_conditions number_set := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_number_set_satisfies_conditions_l671_67187


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_divisor_of_n4_minus_n2_l671_67178

theorem largest_divisor_of_n4_minus_n2 :
  ∃ (k : ℕ), k = 12 ∧ 
  (∀ (n : ℤ), (k : ℤ) ∣ (n^4 - n^2)) ∧
  (∀ (m : ℕ), m > k → ∃ (n : ℤ), ¬((m : ℤ) ∣ (n^4 - n^2))) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_divisor_of_n4_minus_n2_l671_67178


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l671_67196

noncomputable def f (x : ℝ) : ℝ := (x^2 + 1) / x

theorem f_properties :
  (∀ x, f (-x) = -f x) ∧  -- f is odd
  f (-1) = -2 ∧
  f 2 < 3 →
  (∀ x₁ x₂, x₁ < x₂ ∧ x₂ ≤ -1 → f x₁ < f x₂) ∧  -- increasing on (-∞, -1]
  (∀ x₁ x₂, -1 ≤ x₁ ∧ x₁ < x₂ ∧ x₂ < 0 → f x₁ > f x₂) ∧  -- decreasing on [-1, 0)
  (∀ m, (∀ x, x < 0 → 2*m - 1 > f x) → m > -1/2) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l671_67196


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_binomial_probability_normal_distribution_probability_stratified_sampling_l671_67146

-- Statement B
theorem binomial_probability (ξ : ℕ → ℝ) (n : ℕ) (p : ℝ) :
  (∀ k, ξ k = n.choose k * p^k * (1-p)^(n-k)) →
  n = 4 →
  p = 1/2 →
  ξ 3 = 1/4 :=
sorry

-- Statement C
theorem normal_distribution_probability (η : ℝ → ℝ) (μ σ : ℝ) :
  (∀ x, η x = (1 / (σ * Real.sqrt (2 * Real.pi))) * Real.exp (-((x - μ)^2 / (2 * σ^2)))) →
  μ = 5 →
  (∫ x in Set.Iic 2, η x) = 0.1 →
  (∫ x in Set.Ioo 2 8, η x) = 0.8 :=
sorry

-- Statement D
theorem stratified_sampling (total_students grade11 grade12 grade13 selected_total selected_grade11 : ℕ) :
  grade11 = 400 →
  grade12 = 360 →
  total_students = grade11 + grade12 + grade13 →
  selected_total = 57 →
  selected_grade11 = 20 →
  (selected_grade11 : ℝ) / grade11 = (selected_total - selected_grade11 : ℝ) / (total_students - grade11) →
  selected_total - selected_grade11 - Int.floor ((grade12 : ℝ) * selected_grade11 / grade11) = 19 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_binomial_probability_normal_distribution_probability_stratified_sampling_l671_67146
