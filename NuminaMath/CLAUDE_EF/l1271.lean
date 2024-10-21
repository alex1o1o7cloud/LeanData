import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_distance_l1271_127123

/-- The distance between two points (t, sin t) and (t-5, cos t) for any real t -/
noncomputable def distance (t : ℝ) : ℝ := Real.sqrt (25 + (Real.sin t - Real.cos t)^2)

/-- The maximum distance between the two points for any real t -/
theorem max_distance : ∃ (t : ℝ), ∀ (s : ℝ), distance s ≤ Real.sqrt 29 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_distance_l1271_127123


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_johns_horizontal_distance_l1271_127164

/-- Calculates the horizontal distance traveled given a vertical distance and a ratio of vertical to horizontal movement. -/
noncomputable def horizontalDistance (verticalDist : ℝ) (verticalRatio : ℝ) (horizontalRatio : ℝ) : ℝ :=
  (verticalDist * horizontalRatio) / verticalRatio

theorem johns_horizontal_distance :
  let verticalDist : ℝ := 1450 - 100
  let verticalRatio : ℝ := 1
  let horizontalRatio : ℝ := 2
  horizontalDistance verticalDist verticalRatio horizontalRatio = 2700 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_johns_horizontal_distance_l1271_127164


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_special_number_50th_digit_l1271_127114

/-- Represents a 150-digit positive integer where all digits except the 75th are 5 -/
def SpecialNumber := { n : ℕ // n ≥ 10^149 ∧ n < 10^150 }

/-- Returns the nth digit of a natural number (1-indexed from left) -/
def nthDigit (n : ℕ) (num : ℕ) : ℕ := sorry

/-- Returns the sum of digits of a natural number -/
def digitSum (n : ℕ) : ℕ := sorry

/-- Checks if a number is a SpecialNumber -/
def isSpecialNumber (n : SpecialNumber) : Prop :=
  ∀ i : ℕ, i ≥ 1 ∧ i ≤ 150 ∧ i ≠ 75 → nthDigit i n.val = 5

theorem special_number_50th_digit 
  (p : SpecialNumber) 
  (h1 : isSpecialNumber p) 
  (h2 : digitSum p.val % 23 = 0) : 
  nthDigit 50 p.val = 5 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_special_number_50th_digit_l1271_127114


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_harlys_dogs_taken_back_l1271_127125

/-- Calculates the number of dogs Harly had to take back -/
def dogs_taken_back (initial : ℕ) (adopted_percentage : ℚ) (final : ℕ) : ℕ :=
  (final : ℤ) - ((initial : ℤ) - Int.floor ((initial : ℚ) * adopted_percentage)) |>.toNat

theorem harlys_dogs_taken_back :
  dogs_taken_back 80 (40 / 100) 53 = 5 := by
  rfl

#eval dogs_taken_back 80 (40 / 100) 53

end NUMINAMATH_CALUDE_ERRORFEEDBACK_harlys_dogs_taken_back_l1271_127125


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_count_valid_numbers_l1271_127189

/-- A function that checks if a number contains any of the digits 6, 7, 8, 9, or 0 -/
def containsDisallowedDigit (n : Nat) : Bool :=
  let digits := Nat.digits 10 n
  digits.any (fun d => d = 6 ∨ d = 7 ∨ d = 8 ∨ d = 9 ∨ d = 0)

/-- The set of numbers less than 1000, divisible by 4, and not containing disallowed digits -/
def validNumbers : Finset Nat :=
  Finset.filter (fun n => n < 1000 ∧ n % 4 = 0 ∧ ¬containsDisallowedDigit n) (Finset.range 1000)

/-- Theorem stating that the count of valid numbers is 31 -/
theorem count_valid_numbers : validNumbers.card = 31 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_count_valid_numbers_l1271_127189


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_water_bottles_duration_l1271_127106

/-- Represents the number of days the water bottles last --/
def bottlesDuration (initialBottles : ℕ) (dailyConsumption : ℚ) : ℕ :=
  (initialBottles : ℚ) / dailyConsumption |>.floor.toNat

/-- Calculates the daily net change in bottles --/
def dailyNetChange (dailyDrink : ℕ) (biDailyShare : ℕ) (triDailyBuy : ℕ) : ℚ :=
  -(dailyDrink : ℚ) - (biDailyShare : ℚ) / 2 + (triDailyBuy : ℚ) / 3

theorem water_bottles_duration :
  let initialBottles : ℕ := 28
  let dailyDrink : ℕ := 7
  let biDailyShare : ℕ := 3
  let triDailyBuy : ℕ := 5
  let netChange := dailyNetChange dailyDrink biDailyShare triDailyBuy
  bottlesDuration initialBottles netChange = 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_water_bottles_duration_l1271_127106


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_high_jump_sneakers_cost_l1271_127143

/-- The cost of High Jump sneakers given Joey's earnings from various activities -/
theorem high_jump_sneakers_cost 
  (lawns : ℕ) (lawn_price : ℕ) (figures : ℕ) (figure_price : ℕ) 
  (hours : ℕ) (hourly_rate : ℕ) (sneaker_cost : ℕ) :
  lawns = 3 →
  lawn_price = 8 →
  figures = 2 →
  figure_price = 9 →
  hours = 10 →
  hourly_rate = 5 →
  sneaker_cost = lawns * lawn_price + figures * figure_price + hours * hourly_rate →
  sneaker_cost = 92 := by
  sorry

#check high_jump_sneakers_cost

end NUMINAMATH_CALUDE_ERRORFEEDBACK_high_jump_sneakers_cost_l1271_127143


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_curvature_points_l1271_127194

/-- The radius of curvature for a function y = f(x) --/
noncomputable def radiusOfCurvature (f : ℝ → ℝ) (x : ℝ) : ℝ :=
  (1 + (deriv f x)^2)^(3/2) / |deriv^[2] f x|

/-- The parabola function --/
noncomputable def parabola (x : ℝ) : ℝ := Real.sqrt 2 * x^2

theorem parabola_curvature_points :
  ∀ x : ℝ, radiusOfCurvature parabola x = 1 ↔ x = Real.sqrt 2 / 4 ∨ x = -Real.sqrt 2 / 4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_curvature_points_l1271_127194


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_minimum_value_l1271_127181

noncomputable def f (x : ℝ) : ℝ := x + (3*x)/(x^2 + 3) + (x*(x + 3))/(x^2 + 1) + (3*(x + 1))/(x*(x^2 + 1))

theorem f_minimum_value (x : ℝ) (h : x > 0) : f x ≥ 7 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_minimum_value_l1271_127181


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_first_minimum_phase_shift_l1271_127156

/-- The cosine function f(x) = 3 cos(2x - π) -/
noncomputable def f (x : ℝ) : ℝ := 3 * Real.cos (2 * x - Real.pi)

/-- The phase shift that results in the first minimum of f(x) -/
noncomputable def phase_shift : ℝ := Real.pi

/-- Theorem stating that the phase_shift is the smallest positive value that results in the first minimum of f(x) -/
theorem first_minimum_phase_shift :
  ∀ φ : ℝ, φ > 0 → (∀ x : ℝ, f (x + φ) ≥ f (x + phase_shift)) → φ ≥ phase_shift := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_first_minimum_phase_shift_l1271_127156


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_negative_integers_l1271_127135

theorem max_negative_integers (a b c d e f : ℤ) (h : a * b + c * d * e * f < 0) :
  ∃ (w : ℕ), w ≤ 4 ∧ 
  ∀ (x : ℕ), (∃ (s : Fin 6 → ℤ), (s 0 = a ∧ s 1 = b ∧ s 2 = c ∧ s 3 = d ∧ s 4 = e ∧ s 5 = f) ∧ 
              x = (Finset.filter (λ i => s i < 0) (Finset.univ : Finset (Fin 6))).card) →
  x ≤ w :=
by
  -- We claim that w = 4 satisfies the conditions
  use 4
  constructor
  · -- Trivially, 4 ≤ 4
    simp
  · -- For any x satisfying the given condition, we need to show x ≤ 4
    intro x ⟨s, hs, hx⟩
    -- We don't need to prove the full theorem here, just that x ≤ 4
    sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_negative_integers_l1271_127135


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_theta_value_l1271_127141

theorem tan_theta_value (θ m : ℝ) 
  (h1 : Real.sin θ = (m - 3) / (m + 5))
  (h2 : Real.cos θ = (4 - 2*m) / (m + 5))
  (h3 : π / 2 < θ)
  (h4 : θ < π) :
  Real.tan θ = -5 / 12 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_theta_value_l1271_127141


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_equation_proof_l1271_127178

noncomputable def ellipse_equation (a b : ℝ) (x y : ℝ) : Prop :=
  x^2 / a^2 + y^2 / b^2 = 1

noncomputable def eccentricity (a b : ℝ) : ℝ :=
  Real.sqrt (1 - b^2 / a^2)

noncomputable def distance_point_to_line (x y : ℝ) (A B C : ℝ) : ℝ :=
  |A * x + B * y + C| / Real.sqrt (A^2 + B^2)

theorem ellipse_equation_proof (a b : ℝ) :
  a > b ∧ b > 0 ∧
  eccentricity a b = Real.sqrt 3 / 2 ∧
  distance_point_to_line 0 0 1 (-1) (Real.sqrt 2) = b →
  ellipse_equation 2 1 = ellipse_equation a b :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_equation_proof_l1271_127178


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_leftmost_vertex_coordinate_l1271_127193

theorem leftmost_vertex_coordinate (n : ℕ) : 
  n > 0 →
  (Real.log ((n + 1)^2 / (n * (n + 2))) = Real.log (24 / 23)) →
  n = 12 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_leftmost_vertex_coordinate_l1271_127193


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_magnitude_c_l1271_127136

theorem max_magnitude_c (a b c : ℝ × ℝ) : 
  ‖a‖ = 1 → 
  ‖b‖ = 1 → 
  a • b = 0 → 
  (a - c) • (a - c) = 0 → 
  ∃ (k : ℝ), ‖c‖ ≤ k ∧ k = Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_magnitude_c_l1271_127136


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_at_pi_sixth_l1271_127151

noncomputable section

/-- The period of a cosine function with coefficient 2 in the argument -/
def period_cos (a : ℝ) : ℝ := 2 * Real.pi / 2

/-- The period of a sine function with coefficient ω in the argument -/
def period_sin (ω : ℝ) : ℝ := 2 * Real.pi / ω

/-- Function f(x) -/
def f (ω : ℝ) (x : ℝ) : ℝ := 3 * Real.sin (ω * x - Real.pi / 6)

/-- Function g(x) -/
def g (φ : ℝ) (x : ℝ) : ℝ := 2 * Real.cos (2 * x + φ) + 1

/-- Theorem stating that if the axes of symmetry of f and g are identical,
    then f(π/6) = 3/2 -/
theorem f_at_pi_sixth (ω : ℝ) (φ : ℝ) (h1 : ω > 0) 
    (h2 : period_sin ω = period_cos 2) : f ω (Real.pi / 6) = 3 / 2 := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_at_pi_sixth_l1271_127151


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_even_function_condition_l1271_127153

noncomputable def f (x α β : Real) : Real :=
  if x ≤ 0 then Real.sin (x + α) else Real.cos (x - β)

theorem even_function_condition (α β : Real) : 
  (∀ x, f x α β = f (-x) α β) → α = β + π/2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_even_function_condition_l1271_127153


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_product_equality_l1271_127158

open Real

theorem trigonometric_product_equality :
  let c := π / 7
  (sin (2 * c) * sin (4 * c) * sin (6 * c) * sin (8 * c) * sin (10 * c)) /
  (sin c * sin (3 * c) * sin (5 * c) * sin (7 * c) * sin (9 * c)) =
  sin (π / 7) / sin (2 * π / 7) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_product_equality_l1271_127158


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_expenses_opposite_of_income_l1271_127134

/-- Represents a monetary value in yuan -/
structure Yuan where
  value : ℤ

/-- Represents income in yuan -/
def income (x : ℕ) : Yuan where
  value := x

/-- Represents expenses in yuan -/
def expenses (x : ℕ) : Yuan where
  value := -x

instance : Neg Yuan where
  neg y := ⟨-y.value⟩

instance : OfNat Yuan n where
  ofNat := ⟨n⟩

/-- Theorem stating that if income of 5 yuan is denoted as +5, then expenses of 5 yuan should be denoted as -5 -/
theorem expenses_opposite_of_income :
  income 5 = (5 : Yuan) → expenses 5 = (-5 : Yuan) := by
  intro h
  rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_expenses_opposite_of_income_l1271_127134


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_face_sum_18_l1271_127147

-- Define the set of numbers to be used
def cube_numbers : Finset ℕ := {0, 1, 2, 3, 4, 5, 6, 7}

-- Define a function to check if a number is prime
def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m > 1 → m < n → ¬(n % m = 0)

-- Define a type for cube vertices
inductive Vertex
| v1 | v2 | v3 | v4 | v5 | v6 | v7 | v8

-- Define a type for cube edges
inductive Edge
| e1 | e2 | e3 | e4 | e5 | e6 | e7 | e8 | e9 | e10 | e11 | e12

-- Define a type for cube faces
inductive Face
| f1 | f2 | f3 | f4 | f5 | f6

-- Define functions to get the start and end vertices of an edge
def edge_start : Edge → Vertex
| Edge.e1 => Vertex.v1
| Edge.e2 => Vertex.v1
| Edge.e3 => Vertex.v1
| _ => Vertex.v1  -- Placeholder, extend for all edges

def edge_end : Edge → Vertex
| Edge.e1 => Vertex.v2
| Edge.e2 => Vertex.v3
| Edge.e3 => Vertex.v4
| _ => Vertex.v2  -- Placeholder, extend for all edges

-- Define a function to get the vertices of a face
def face_vertices : Face → List Vertex
| Face.f1 => [Vertex.v1, Vertex.v2, Vertex.v3, Vertex.v4]
| _ => [Vertex.v1, Vertex.v2, Vertex.v3, Vertex.v4]  -- Placeholder, extend for all faces

-- Define a valid cube configuration
def valid_cube_config (config : Vertex → ℕ) : Prop :=
  (∀ v, config v ∈ cube_numbers) ∧
  (∀ v1 v2, v1 ≠ v2 → config v1 ≠ config v2) ∧
  (∀ e : Edge, is_prime (config (edge_start e) + config (edge_end e)))

-- Define a function to get the sum of a face
def face_sum (config : Vertex → ℕ) (face : Face) : ℕ :=
  (face_vertices face).map config |>.sum

-- Theorem statement
theorem max_face_sum_18 :
  ∀ (config : Vertex → ℕ),
    valid_cube_config config →
    ∀ (face : Face), face_sum config face ≤ 18 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_face_sum_18_l1271_127147


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circuit_probabilities_l1271_127168

-- Define the probabilities
variable (p q : ℝ)

-- Define the conditions
variable (hp : 0 ≤ p ∧ p ≤ 1)
variable (hq : 0 ≤ q ∧ q ≤ 1)
variable (hpq : p + q = 1)

-- Define the probability of receiving a signal at the output
noncomputable def P_S (p q : ℝ) : ℝ := q * (1 - p^2)^2 + p * q^2 * (2 - q^2)

-- Define the conditional probability that relay E was closed given a signal was received
noncomputable def P_E_given_S (p q : ℝ) : ℝ := ((2 * q^2 - q^4) * p) / (q * (1 - p^2)^2 + p * q^2 * (2 - q^2))

-- State the theorem
theorem circuit_probabilities (p q : ℝ) (hp : 0 ≤ p ∧ p ≤ 1) (hq : 0 ≤ q ∧ q ≤ 1) (hpq : p + q = 1) :
  P_S p q = q * (1 - p^2)^2 + p * q^2 * (2 - q^2) ∧
  P_E_given_S p q = ((2 * q^2 - q^4) * p) / (q * (1 - p^2)^2 + p * q^2 * (2 - q^2)) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_circuit_probabilities_l1271_127168


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_number_of_factors_of_M_l1271_127195

def M : ℕ := 47^4 + 4*47^3 + 6*47^2 + 4*47 + 1

theorem number_of_factors_of_M : 
  (Finset.filter (λ x : ℕ ↦ x > 0 ∧ M % x = 0) (Finset.range (M + 1))).card = 85 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_number_of_factors_of_M_l1271_127195


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_port_perry_lazy_harbor_difference_l1271_127104

/-- The population difference between Port Perry and Lazy Harbor --/
def population_difference (p l : ℕ) : ℕ := p - l

theorem port_perry_lazy_harbor_difference 
  (wellington port_perry lazy_harbor : ℕ) 
  (h1 : wellington = 900)
  (h2 : port_perry = 7 * wellington)
  (h3 : port_perry + lazy_harbor = 11800) :
  population_difference port_perry lazy_harbor = 800 := by
  sorry

#check port_perry_lazy_harbor_difference

end NUMINAMATH_CALUDE_ERRORFEEDBACK_port_perry_lazy_harbor_difference_l1271_127104


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_beta_values_l1271_127105

-- Define the angle β
noncomputable def β : ℝ := sorry

-- Define the condition that the terminal side of β lies on y = -√3x
def terminal_side_condition : Prop :=
  ∃ (x y : ℝ), y = -Real.sqrt 3 * x ∧ 
  (Real.cos β = x / Real.sqrt (x^2 + y^2)) ∧ 
  (Real.sin β = y / Real.sqrt (x^2 + y^2))

-- Define the range condition for β
def range_condition : Prop :=
  -Real.pi ≤ β ∧ β ≤ Real.pi

-- State the theorem
theorem angle_beta_values : 
  terminal_side_condition → range_condition → (β = -Real.pi/3 ∨ β = 2*Real.pi/3) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_beta_values_l1271_127105


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_number_of_subsets_A_intersect_B_l1271_127142

-- Define the sets A and B
def A : Set (ℝ × ℝ) := {p | p.1^2 + p.2^2 = 1}
def B : Set (ℝ × ℝ) := {p | p.2 = |p.1|}

-- Define the intersection of A and B
def A_intersect_B : Set (ℝ × ℝ) := A ∩ B

-- Define the specific points in the intersection
def intersection_points : Set (ℝ × ℝ) := {(-Real.sqrt 2 / 2, -Real.sqrt 2 / 2), (Real.sqrt 2 / 2, Real.sqrt 2 / 2)}

-- State the theorem
theorem number_of_subsets_A_intersect_B :
  Finset.card (Finset.powerset {(-Real.sqrt 2 / 2, -Real.sqrt 2 / 2), (Real.sqrt 2 / 2, Real.sqrt 2 / 2)}) = 4 := by
  -- The proof goes here
  sorry

-- Lemma to show that A_intersect_B is equal to the specific points
lemma A_intersect_B_eq_intersection_points : A_intersect_B = intersection_points := by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_number_of_subsets_A_intersect_B_l1271_127142


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_angle_implies_x_coordinate_l1271_127127

/-- Given vectors a and b in R^2, if the angle between them is 2π/3, then the x-coordinate of b is -2 -/
theorem vector_angle_implies_x_coordinate (a b : ℝ × ℝ) :
  a = (2, 0) →
  ∃ x : ℝ, b = (x, 2 * Real.sqrt 3) →
  Real.arccos ((a.1 * b.1 + a.2 * b.2) / (Real.sqrt (a.1^2 + a.2^2) * Real.sqrt (b.1^2 + b.2^2))) = 2 * π / 3 →
  x = -2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_angle_implies_x_coordinate_l1271_127127


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exist_n_plus_one_faces_same_edges_l1271_127174

/-- A polyhedron is a structure with faces, edges, and vertices. -/
structure Polyhedron where
  faces : ℕ
  edges : ℕ
  vertices : ℕ

/-- A function that returns the number of edges for each face of a polyhedron. -/
def face_edges (p : Polyhedron) : Fin p.faces → ℕ := sorry

/-- Theorem: For any polyhedron with 7n faces, there exist n+1 faces with the same number of edges. -/
theorem exist_n_plus_one_faces_same_edges (n : ℕ) (p : Polyhedron) 
  (h_faces : p.faces = 7 * n) :
  ∃ (e : ℕ), ∃ (S : Finset (Fin p.faces)), S.card = n + 1 ∧ 
    ∀ (i : Fin p.faces), i ∈ S → face_edges p i = e := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_exist_n_plus_one_faces_same_edges_l1271_127174


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_remainder_modulo_thirteen_l1271_127126

theorem sum_remainder_modulo_thirteen (a b c d e : ℕ) 
  (ha : a % 13 = 3)
  (hb : b % 13 = 5)
  (hc : c % 13 = 7)
  (hd : d % 13 = 9)
  (he : e % 13 = 11) :
  (a + b + c + d + e) % 13 = 9 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_remainder_modulo_thirteen_l1271_127126


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_employees_within_height_range_l1271_127162

/-- Represents the normal distribution parameters -/
structure NormalDistParams where
  μ : ℝ
  σ : ℝ

/-- Represents the probability of a value falling within a given range of a normal distribution -/
noncomputable def probability_within_range (params : NormalDistParams) (lower : ℝ) (upper : ℝ) : ℝ :=
  sorry

/-- The total number of employees -/
def total_employees : ℕ := 10000

/-- The parameters of the height distribution -/
def height_distribution : NormalDistParams :=
  { μ := 173, σ := 5 }

/-- The lower bound of the height range -/
def lower_height : ℝ := 163

/-- The upper bound of the height range -/
def upper_height : ℝ := 183

/-- Theorem stating that the number of employees within the given height range is approximately 9540 -/
theorem employees_within_height_range :
  ∃ (ε : ℝ), ε > 0 ∧ ε < 1 ∧ 
  (⌊total_employees * probability_within_range height_distribution lower_height upper_height⌋ : ℝ) - 9540 < ε :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_employees_within_height_range_l1271_127162


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_break_even_theorem_l1271_127148

/-- The cost of producing N items -/
noncomputable def cost (N : ℝ) : ℝ := 2100 * Real.sqrt 10 / Real.sqrt N

/-- The revenue from selling N items -/
def revenue (N : ℝ) : ℝ := 30 * N

/-- The break-even point -/
noncomputable def break_even_point : ℝ := 10 * (49 : ℝ) ^ (1/3)

theorem break_even_theorem :
  ∃ N : ℝ, N > 0 ∧ cost N = revenue N ∧ N = break_even_point := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_break_even_theorem_l1271_127148


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_volume_regular_truncated_pyramid_l1271_127159

/-- The volume of a regular truncated pyramid with square bases -/
noncomputable def truncated_pyramid_volume (a b : ℝ) : ℝ :=
  (Real.sqrt 2 / 6) * (a^3 - b^3)

/-- Function to calculate the volume of a regular truncated pyramid -/
noncomputable def volume_of_regular_truncated_pyramid (a b angle : ℝ) : ℝ :=
  sorry

/-- Theorem: Volume of a regular truncated pyramid with square bases -/
theorem volume_regular_truncated_pyramid
  (a b : ℝ)
  (h1 : a > b)
  (h2 : a > 0)
  (h3 : b > 0) :
  ∃ (V : ℝ), V = truncated_pyramid_volume a b ∧
  V = volume_of_regular_truncated_pyramid a b 45 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_volume_regular_truncated_pyramid_l1271_127159


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_expansion_coefficient_sum_l1271_127175

theorem expansion_coefficient_sum (n : ℕ) : 
  (∀ x : ℝ, x ≠ 0 → (x^2 + x^(-3 : ℤ))^n = 32) → n = 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_expansion_coefficient_sum_l1271_127175


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_positive_solution_approx_l1271_127183

noncomputable def f (x : ℝ) := (x^2 - 40*x - 8) * (x^2 + 20*x + 4) - 25*(x^2 - 1/2)

theorem positive_solution_approx :
  ∃ (x : ℝ), x > 0 ∧ f x = 0 ∧ abs (x - (25 + 2 * Real.sqrt 159)) < 1e-6 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_positive_solution_approx_l1271_127183


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rationalize_denominator_cube_root_l1271_127109

theorem rationalize_denominator_cube_root :
  (1 : ℝ) / (Real.rpow 3 (1/3) + Real.rpow 27 (1/3)) = (3 - Real.rpow 3 (1/3)) / (9 - 3 * Real.rpow 3 (1/3)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rationalize_denominator_cube_root_l1271_127109


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_circle_intersection_l1271_127184

-- Define the line and circle
def line_equation (a b x y : ℝ) : ℝ := (a + 2*b)*x + (b - a)*y + a - b
def circle_equation (x y m : ℝ) : ℝ := x^2 + y^2 - m

-- State the theorem
theorem line_circle_intersection (a b : ℝ) :
  (∀ m : ℝ, ∃ x y : ℝ, line_equation a b x y = 0 ∧ circle_equation x y m = 0) →
  ∀ m : ℝ, m ≥ (1/2) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_circle_intersection_l1271_127184


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_equation_from_intercepts_l1271_127172

/-- A line in a 2D plane -/
structure Line where
  /-- The equation of the line in the form ax + by + c = 0 -/
  a : ℝ
  b : ℝ
  c : ℝ

/-- The x-intercept of a line -/
noncomputable def xIntercept (l : Line) : ℝ := -l.c / l.a

/-- The y-intercept of a line -/
noncomputable def yIntercept (l : Line) : ℝ := -l.c / l.b

theorem line_equation_from_intercepts (l : Line) :
  xIntercept l = 3 ∧ yIntercept l = -2 →
  l.a = 2 ∧ l.b = -3 ∧ l.c = -6 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_equation_from_intercepts_l1271_127172


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_third_median_length_l1271_127185

/-- Triangle with perpendicular medians -/
structure TriangleWithPerpendicularMedians where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ
  D : ℝ × ℝ
  E : ℝ × ℝ
  F : ℝ × ℝ
  h_midpoint_D : D = ((B.1 + C.1) / 2, (B.2 + C.2) / 2)
  h_midpoint_E : E = ((A.1 + C.1) / 2, (A.2 + C.2) / 2)
  h_midpoint_F : F = ((A.1 + B.1) / 2, (A.2 + B.2) / 2)
  h_perpendicular : (A.1 - D.1) * (B.1 - E.1) + (A.2 - D.2) * (B.2 - E.2) = 0
  h_length_AD : (A.1 - D.1)^2 + (A.2 - D.2)^2 = 18^2
  h_length_BE : (B.1 - E.1)^2 + (B.2 - E.2)^2 = 13.5^2

/-- The length of the third median CF is 22.5 -/
theorem third_median_length (t : TriangleWithPerpendicularMedians) :
  (t.C.1 - t.F.1)^2 + (t.C.2 - t.F.2)^2 = 22.5^2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_third_median_length_l1271_127185


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_t_of_f_8_l1271_127144

-- Define the functions t and f
noncomputable def t (x : ℝ) : ℝ := Real.sqrt (5 * x + 1)
noncomputable def f (x : ℝ) : ℝ := 8 - t x

-- State the theorem
theorem t_of_f_8 : t (f 8) = Real.sqrt (41 - 5 * Real.sqrt 41) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_t_of_f_8_l1271_127144


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_solution_l1271_127101

/-- The function representing the left side of the equation -/
noncomputable def f (x : ℝ) : ℝ := Real.sqrt (2 + Real.sqrt (3 + Real.sqrt x))

/-- The function representing the right side of the equation -/
noncomputable def g (x : ℝ) : ℝ := (2 + Real.sqrt x) ^ (1/3)

/-- The statement of the theorem -/
theorem unique_solution :
  ∃! x : ℝ, x ≥ 0 ∧ f x = g x ∧ x = 36 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_solution_l1271_127101


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_angle_measure_l1271_127199

/-- Two triangles are congruent -/
def congruent_triangles (ABC ACD : Set Point) : Prop := sorry

/-- The measure of an angle in degrees -/
def angle_measure (p q r : Point) : ℝ := sorry

/-- The distance between two points -/
def dist (p q : Point) : ℝ := sorry

theorem triangle_angle_measure 
  (A B C D : Point) 
  (h1 : congruent_triangles {A, B, C} {A, C, D})
  (h2 : dist A B = dist A C)
  (h3 : dist A C = dist A D)
  (h4 : angle_measure B A C = 40) :
  angle_measure B D C = 20 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_angle_measure_l1271_127199


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_median_divides_area_equally_l1271_127176

/-- A triangle -/
structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ

/-- The median of a triangle -/
def median (t : Triangle) (vertex : ℝ × ℝ) (midpoint : ℝ × ℝ) : Prop :=
  (vertex = t.A ∨ vertex = t.B ∨ vertex = t.C) ∧
  ((midpoint.1 = (t.A.1 + t.B.1) / 2 ∧ midpoint.2 = (t.A.2 + t.B.2) / 2) ∨
   (midpoint.1 = (t.B.1 + t.C.1) / 2 ∧ midpoint.2 = (t.B.2 + t.C.2) / 2) ∨
   (midpoint.1 = (t.C.1 + t.A.1) / 2 ∧ midpoint.2 = (t.C.2 + t.A.2) / 2))

/-- The area of a triangle -/
noncomputable def area (t : Triangle) : ℝ :=
  sorry

/-- Theorem: The median of a triangle divides the triangle's area into two equal parts -/
theorem median_divides_area_equally (t : Triangle) (v m : ℝ × ℝ) 
  (h : median t v m) : 
  ∃ (t1 t2 : Triangle), area t1 = area t2 ∧ area t = area t1 + area t2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_median_divides_area_equally_l1271_127176


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_when_a_neg_two_range_of_a_for_solution_l1271_127117

-- Define the functions f and g
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := |x - 1| + |x + a|
noncomputable def g (x : ℝ) : ℝ := (1/2) * x + 3

-- Part 1
theorem solution_set_when_a_neg_two :
  {x : ℝ | f (-2) x < g x} = {x : ℝ | 0 < x ∧ x < 4} := by sorry

-- Part 2
theorem range_of_a_for_solution :
  ∀ a : ℝ, a > -1 →
  (∃ x : ℝ, x ∈ Set.Icc (-a) 1 ∧ f a x ≤ g x) →
  a ∈ Set.Ioo (-1) (5/2) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_when_a_neg_two_range_of_a_for_solution_l1271_127117


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hexadecagon_area_160_perimeter_l1271_127122

/-- A hexadecagon inscribed in a square -/
structure InscribedHexadecagon :=
  (side_length : ℝ)
  (perimeter : ℝ)
  (h_perimeter : perimeter = 4 * side_length)
  (h_vertices : side_length = 4 * (side_length / 4))

/-- The area of the inscribed hexadecagon -/
noncomputable def area_hexadecagon (h : InscribedHexadecagon) : ℝ :=
  h.side_length^2 - 8 * ((h.side_length / 4)^2 / 2)

theorem hexadecagon_area_160_perimeter :
  ∀ h : InscribedHexadecagon, h.perimeter = 160 → area_hexadecagon h = 1200 := by
  sorry

#check hexadecagon_area_160_perimeter

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hexadecagon_area_160_perimeter_l1271_127122


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unfair_coin_probability_l1271_127139

/-- An unfair coin with probability of tails -/
noncomputable def unfair_coin_prob_tails : ℝ := 3/4

/-- Total number of coin flips -/
def total_flips : ℕ := 5

/-- Minimum number of tails required -/
def min_tails : ℕ := 3

/-- Number of guaranteed tail flips at the end -/
def guaranteed_tails : ℕ := 2

/-- Probability of the desired outcome -/
noncomputable def desired_probability : ℝ := 567/1024

/-- Theorem stating the probability of the desired outcome for the unfair coin flips -/
theorem unfair_coin_probability :
  let p_tail := unfair_coin_prob_tails
  let n := total_flips
  let k := min_tails
  let m := guaranteed_tails
  (1 - (1 - p_tail)^(n - m)) * p_tail^m = desired_probability := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_unfair_coin_probability_l1271_127139


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_m_l1271_127190

-- Define the functions f and g
noncomputable def f (x : ℝ) : ℝ := 1 / Real.sqrt (4 + 3*x - x^2)
def g (x : ℝ) : ℝ := -x^2 - 2*x + 2

-- Define the sets A, B, and C
def A : Set ℝ := {x | -1 < x ∧ x < 4}
def B : Set ℝ := {x | -1 ≤ x ∧ x ≤ 3}
def C (m : ℝ) : Set ℝ := {x | m ≤ x ∧ x ≤ m + 2}

-- State the theorem
theorem range_of_m :
  ∀ m : ℝ, (C m ∩ (A ∪ B) = C m) ↔ (-1 ≤ m ∧ m < 2) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_m_l1271_127190


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_gray_area_between_circles_l1271_127115

/-- The area of the gray region between two circles, where the radius of the larger circle
    is four times the radius of the smaller circle, and the diameter of the smaller circle is 2 units. -/
theorem gray_area_between_circles (π : ℝ) : 
  let smaller_diameter : ℝ := 2
  let smaller_radius : ℝ := smaller_diameter / 2
  let larger_radius : ℝ := 4 * smaller_radius
  let larger_area : ℝ := π * larger_radius ^ 2
  let smaller_area : ℝ := π * smaller_radius ^ 2
  let gray_area : ℝ := larger_area - smaller_area
  gray_area = 15 * π := by
    -- Proof steps would go here
    sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_gray_area_between_circles_l1271_127115


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_points_at_distance_sqrt_two_solution_points_l1271_127152

/-- A point on the given line parametrized by t -/
def line_point (t : ℝ) : ℝ × ℝ := (3 - t, 4 + t)

/-- The fixed point P -/
def point_p : ℝ × ℝ := (3, 4)

/-- Distance between two points in ℝ² -/
noncomputable def distance (p q : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

/-- The theorem stating that (4, 3) and (2, 5) are the only points on the line
    at distance √2 from P(3, 4) -/
theorem line_points_at_distance_sqrt_two :
  {t : ℝ | distance (line_point t) point_p = Real.sqrt 2} =
  {t : ℝ | t = 1 ∨ t = -1} :=
by sorry

/-- The main theorem proving the solution -/
theorem solution_points :
  {p : ℝ × ℝ | ∃ t, p = line_point t ∧ distance p point_p = Real.sqrt 2} =
  {(4, 3), (2, 5)} :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_points_at_distance_sqrt_two_solution_points_l1271_127152


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_relation_l1271_127102

theorem angle_relation (α β : ℝ) : 
  α ∈ Set.Ioo 0 (π/2) →
  β ∈ Set.Ioo 0 (π/4) →
  Real.tan α = (Real.cos β + Real.sin β) / (Real.cos β - Real.sin β) →
  α - β = π/4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_relation_l1271_127102


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l1271_127118

/-- The function f(x) defined in the problem -/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * Real.log x + (1 - a) / 2 * x^2 - x

/-- The theorem statement -/
theorem range_of_a (a : ℝ) (h_a : a ≠ 1) :
  (∃ x₀ : ℝ, x₀ ≥ 1 ∧ f a x₀ < a / (a - 1)) →
  a ∈ Set.Ioo (-Real.sqrt 2 - 1) (Real.sqrt 2 - 1) ∪ Set.Ioi 1 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l1271_127118


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_correct_answer_is_it_l1271_127155

def correct_answer : String := "it"

theorem correct_answer_is_it : correct_answer = "it" := by
  rfl

#eval correct_answer

end NUMINAMATH_CALUDE_ERRORFEEDBACK_correct_answer_is_it_l1271_127155


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_basketball_league_female_ratio_l1271_127110

theorem basketball_league_female_ratio :
  ∃ (total_increase_rate male_increase_rate female_increase_rate : ℚ)
    (last_year_males : ℕ)
    (this_year_males last_year_females this_year_females total_participants : ℚ),
  total_increase_rate = 115 / 100 ∧
  male_increase_rate = 110 / 100 ∧
  female_increase_rate = 125 / 100 ∧
  last_year_males = 30 ∧
  this_year_males = (last_year_males : ℚ) * male_increase_rate ∧
  last_year_females = (total_increase_rate * (last_year_males : ℚ) - this_year_males) / (female_increase_rate - total_increase_rate) ∧
  this_year_females = last_year_females * female_increase_rate ∧
  total_participants = this_year_males + this_year_females ∧
  this_year_females / total_participants = 25 / 69 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_basketball_league_female_ratio_l1271_127110


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_three_digit_number_l1271_127163

theorem unique_three_digit_number : ∃! (n : ℕ), 
  (100 ≤ n ∧ n < 1000) ∧ 
  (∃ (P B G : ℕ), 
    P ≠ B ∧ P ≠ G ∧ B ≠ G ∧
    0 ≤ P ∧ P ≤ 9 ∧ 0 ≤ B ∧ B ≤ 9 ∧ 0 ≤ G ∧ G ≤ 9 ∧
    n = 100 * P + 10 * B + G ∧
    n = (P + B + G) * (P + B + G + 1)) ∧
  n = 156 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_three_digit_number_l1271_127163


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_polar_equation_rho_eq_one_is_circle_l1271_127130

/-- The polar equation ρ = 1 represents a circle in the Cartesian coordinate system. -/
theorem polar_equation_rho_eq_one_is_circle :
  ∀ (x y : ℝ), (∃ θ : ℝ, x = Real.cos θ ∧ y = Real.sin θ) ↔ x^2 + y^2 = 1 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_polar_equation_rho_eq_one_is_circle_l1271_127130


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_2theta_special_angle_l1271_127132

theorem tan_2theta_special_angle (θ : ℝ) 
  (h1 : Real.sin (π - θ) = 1/2) 
  (h2 : π/2 < θ ∧ θ < π) : 
  Real.tan (2*θ) = -Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_2theta_special_angle_l1271_127132


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_incircle_tangent_points_equidistant_l1271_127157

open EuclideanSpace

-- Define the triangle ABC
variable {E : Type*} [NormedAddCommGroup E] [InnerProductSpace ℝ E] [FiniteDimensional ℝ E]
variable (A B C : E)

-- Define the right angle at B
variable (right_angle_B : inner (B - A) (C - B) = 0)

-- Define the altitude BH
variable (H : E)
variable (altitude_BH : inner (H - B) (C - A) = 0)

-- Define the incircle points
variable (H₁ B₁ H₂ B₂ : E)

-- Define the properties of incircle points
variable (H₁_on_AB : ∃ t : ℝ, 0 < t ∧ t < 1 ∧ H₁ = (1 - t) • A + t • B)
variable (B₁_on_AH : ∃ t : ℝ, 0 < t ∧ t < 1 ∧ B₁ = (1 - t) • A + t • H)
variable (H₂_on_CB : ∃ t : ℝ, 0 < t ∧ t < 1 ∧ H₂ = (1 - t) • C + t • B)
variable (B₂_on_CH : ∃ t : ℝ, 0 < t ∧ t < 1 ∧ B₂ = (1 - t) • C + t • H)

-- Define O as the circumcenter of triangle H₁BH₂
variable (O : E)
variable (O_circumcenter : dist O H₁ = dist O B ∧ dist O B = dist O H₂)

-- State the theorem
theorem incircle_tangent_points_equidistant :
  dist O B₁ = dist O B₂ := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_incircle_tangent_points_equidistant_l1271_127157


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exponential_inequality_l1271_127116

theorem exponential_inequality (a b : ℝ) (h : a > b) : (2 : ℝ)^a > (2 : ℝ)^b := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_exponential_inequality_l1271_127116


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_integer_list_length_l1271_127140

/-- A list of integers containing 2018 -/
def IntegerList := List Int

/-- Predicate to check if a list contains 2018 -/
def contains_2018 (l : IntegerList) : Prop :=
  l.contains 2018

/-- Sum of all integers in the list -/
def list_sum (l : IntegerList) : Int :=
  l.sum

/-- Product of all integers in the list -/
def list_product (l : IntegerList) : Int :=
  l.prod

/-- Number of integers in the list -/
def list_length (l : IntegerList) : Nat :=
  l.length

theorem integer_list_length :
  ∀ l : IntegerList,
    contains_2018 l →
    list_sum l = 2018 →
    list_product l = 2018 →
    list_length l = 2017 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_integer_list_length_l1271_127140


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_two_x_value_l1271_127191

/-- A function f such that f(x) = sin x + cos x -/
noncomputable def f (x : ℝ) : ℝ := Real.sin x + Real.cos x

/-- The derivative of f -/
noncomputable def f' (x : ℝ) : ℝ := Real.cos x - Real.sin x

theorem tan_two_x_value (x : ℝ) : 
  (∀ x, f' x = (3 : ℝ) * f x) → Real.tan (2 * x) = -4/3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_two_x_value_l1271_127191


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_P_equals_one_value_l1271_127165

open Complex

def V : Set ℂ := {2*I, -2*I, 1+I, -1+I, 1-I, -1-I, Real.sqrt 2 + Real.sqrt 2*I, -(Real.sqrt 2 + Real.sqrt 2*I)}

noncomputable def random_selection : Fin 16 → V := sorry

noncomputable def P : ℂ := (Finset.range 16).prod (λ i => random_selection i)

noncomputable def prob_P_equals_one : ℝ := sorry

theorem prob_P_equals_one_value : prob_P_equals_one = 1 / 2^14 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_P_equals_one_value_l1271_127165


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_larger_ball_radius_l1271_127186

/-- The volume of a sphere given its radius -/
noncomputable def sphereVolume (r : ℝ) : ℝ := (4 / 3) * Real.pi * r ^ 3

/-- The radius of a sphere given its volume -/
noncomputable def sphereRadius (v : ℝ) : ℝ := (3 * v / (4 * Real.pi)) ^ (1 / 3 : ℝ)

theorem larger_ball_radius (r : ℝ) (n : ℕ) (h : r = 2) (h' : n = 12) :
  sphereRadius (n * sphereVolume r) = (96 : ℝ) ^ (1 / 3 : ℝ) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_larger_ball_radius_l1271_127186


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_regression_line_properties_l1271_127149

/-- Data points representing price and demand -/
noncomputable def data_points : List (ℝ × ℝ) := [(10, 11), (15, 10), (20, 8), (25, 6), (30, 5)]

/-- Regression line equation: y = b * x + 14.4 -/
def regression_line (b : ℝ) (x : ℝ) : ℝ := b * x + 14.4

/-- Mean of x values -/
noncomputable def mean_x : ℝ := (data_points.map (λ p => p.1)).sum / data_points.length

/-- Mean of y values -/
noncomputable def mean_y : ℝ := (data_points.map (λ p => p.2)).sum / data_points.length

theorem regression_line_properties :
  ∃ b : ℝ, 
    b < 0 ∧ 
    regression_line b mean_x = mean_y ∧ 
    b = -0.32 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_regression_line_properties_l1271_127149


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_count_whole_numbers_between_sqrt18_and_sqrt98_l1271_127167

theorem count_whole_numbers_between_sqrt18_and_sqrt98 :
  (Finset.filter (fun n : ℕ => Real.sqrt 18 < n ∧ n < Real.sqrt 98) (Finset.range 100)).card = 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_count_whole_numbers_between_sqrt18_and_sqrt98_l1271_127167


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_lakeside_sports_club_member_ratio_l1271_127121

theorem lakeside_sports_club_member_ratio :
  ∀ (f m : ℕ),
  f > 0 → m > 0 →
  let c := m
  let total := f + m + c
  (35 * f + 30 * m + 10 * c : ℝ) / total = 25 →
  f = m := by
  intro f m hf hm
  let c := m
  let total := f + m + c
  intro h
  sorry

#check lakeside_sports_club_member_ratio

end NUMINAMATH_CALUDE_ERRORFEEDBACK_lakeside_sports_club_member_ratio_l1271_127121


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_favorite_number_is_thirteen_l1271_127161

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m > 1 → m < n → ¬(n % m = 0)

def reverse_digits (n : ℕ) : ℕ :=
  let tens := n / 10
  let ones := n % 10
  ones * 10 + tens

def first_digit (n : ℕ) : ℕ := n / 10

def second_digit (n : ℕ) : ℕ := n % 10

def is_two_digit (n : ℕ) : Prop := n ≥ 10 ∧ n ≤ 99

def satisfies_conditions (n : ℕ) : Prop :=
  is_two_digit n ∧
  is_prime n ∧
  is_prime (reverse_digits n) ∧
  first_digit n < second_digit n ∧
  (∃ m : ℕ, m ≠ n ∧ 
    (is_two_digit m ∧ 
     is_prime m ∧ 
     is_prime (reverse_digits m) ∧ 
     first_digit m < second_digit m) ∧
    (first_digit m = first_digit n ∨ second_digit m = second_digit n)) ∧
  (∀ m : ℕ, 
    (is_two_digit m ∧ 
     is_prime m ∧ 
     is_prime (reverse_digits m) ∧ 
     first_digit m < second_digit m) →
    second_digit m = second_digit n → m = n)

theorem favorite_number_is_thirteen :
  ∃! n : ℕ, satisfies_conditions n ∧ n = 13 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_favorite_number_is_thirteen_l1271_127161


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_cosine_sum_l1271_127146

theorem triangle_cosine_sum (A B C : ℝ) : 
  A + B + C = π → A = 3 * B → A = 9 * C → 
  Real.cos A * Real.cos B + Real.cos B * Real.cos C + Real.cos C * Real.cos A = -1/4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_cosine_sum_l1271_127146


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_uncertain_relationship_l1271_127179

-- Define a type for lines in space
structure Line3D where
  -- You might want to add more properties here, but for this problem, 
  -- we only need to distinguish between different lines
  id : ℕ

-- Define a perpendicular relation between lines
def perpendicular (l1 l2 : Line3D) : Prop :=
  sorry -- The actual definition is not important for this statement

-- Define a type to represent the positional relationship between two lines
inductive PositionalRelationship
  | perpendicular
  | parallel
  | intersecting
  | skew

-- The main theorem
theorem uncertain_relationship 
  (l1 l2 l3 l4 : Line3D) 
  (distinct : l1 ≠ l2 ∧ l1 ≠ l3 ∧ l1 ≠ l4 ∧ l2 ≠ l3 ∧ l2 ≠ l4 ∧ l3 ≠ l4)
  (perp12 : perpendicular l1 l2)
  (perp23 : perpendicular l2 l3)
  (perp34 : perpendicular l3 l4) :
  ∀ (r : PositionalRelationship), 
    ∃ (l1' l2' l3' l4' : Line3D),
      l1' ≠ l2' ∧ l1' ≠ l3' ∧ l1' ≠ l4' ∧ l2' ≠ l3' ∧ l2' ≠ l4' ∧ l3' ≠ l4' ∧
      perpendicular l1' l2' ∧ perpendicular l2' l3' ∧ perpendicular l3' l4' ∧
      (∃ (r' : PositionalRelationship), r' ≠ r) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_uncertain_relationship_l1271_127179


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complement_I_in_nat_is_empty_l1271_127197

def I : Set ℤ := {x : ℤ | x ≥ -1}

theorem complement_I_in_nat_is_empty :
  (Set.univ : Set ℕ) \ (I.image Int.natAbs) = ∅ :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_complement_I_in_nat_is_empty_l1271_127197


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ordering_abc_l1271_127108

-- Define constants for the angles in radians
noncomputable def angle17 : ℝ := 17 * Real.pi / 180
noncomputable def angle45 : ℝ := 45 * Real.pi / 180
noncomputable def angle13 : ℝ := 13 * Real.pi / 180

-- Define a, b, and c as in the problem
noncomputable def a : ℝ := Real.sin angle17 * Real.cos angle45 + Real.cos angle17 * Real.sin angle45
noncomputable def b : ℝ := 2 * (Real.cos angle13)^2 - 1
noncomputable def c : ℝ := Real.sqrt 3 / 2

-- State the theorem
theorem ordering_abc : c < a ∧ a < b := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ordering_abc_l1271_127108


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_car_journey_l1271_127166

/-- Proves that a car driving x miles on local roads at 20 mph and 180 miles on the highway at 60 mph,
    with an average speed of 44 mph for the entire trip, implies x equals 40 miles. -/
theorem car_journey (x : ℝ) : 
  (x / 20 + 180 / 60 : ℝ) > 0 →
  (x + 180) / (x / 20 + 180 / 60) = 44 →
  x = 40 := by
  intro h1 h2
  -- The proof steps would go here
  sorry

#check car_journey

end NUMINAMATH_CALUDE_ERRORFEEDBACK_car_journey_l1271_127166


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_lines_coefficient_l1271_127187

/-- Two lines are parallel if their slopes are equal -/
def parallel (a b d e : ℝ) : Prop :=
  a / b = d / e ∧ b ≠ 0 ∧ e ≠ 0

/-- The coefficient of x in the expansion of (x + 1/x - a)^5 -/
noncomputable def coefficientOfX (a : ℝ) : ℝ :=
  let x : ℝ := Real.pi  -- Using pi as a placeholder for x
  5 * (1/x - a)^4 + 10 * x * (1/x - a)^3 * (-1/x^2) +
  10 * x^2 * (1/x - a)^2 * (1/x^3) + 5 * x^3 * (1/x - a) * (-1/x^4) +
  x^4 * (1/x^5)

theorem parallel_lines_coefficient (a : ℝ) :
  parallel 1 a 2 4 →
  coefficientOfX a = 210 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_lines_coefficient_l1271_127187


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_jonathan_oil_purchase_l1271_127170

/-- Calculates the minimum number of bottles needed to contain a given volume of oil -/
def min_bottles_needed (required_fl_oz : ℚ) (bottle_size_ml : ℚ) (fl_oz_per_liter : ℚ) : ℕ :=
  let required_ml := required_fl_oz * 1000 / fl_oz_per_liter
  (required_ml / bottle_size_ml).ceil.toNat

/-- Proves that 12 bottles are needed for Jonathan's oil purchase -/
theorem jonathan_oil_purchase :
  min_bottles_needed 60 150 33.8 = 12 := by
  sorry

#eval min_bottles_needed 60 150 33.8

end NUMINAMATH_CALUDE_ERRORFEEDBACK_jonathan_oil_purchase_l1271_127170


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_first_ten_nicely_odd_l1271_127128

/-- A number is "nicely odd" if it's an integer greater than 1 that equals the product of its
    distinct proper divisors, where all involved primes are odd. -/
def NicelyOdd (n : ℕ) : Prop :=
  n > 1 ∧
  (∃ p q : ℕ, Nat.Prime p ∧ Nat.Prime q ∧ Odd p ∧ Odd q ∧ p ≠ q ∧ n = p * q) ∨
  (∃ p : ℕ, Nat.Prime p ∧ Odd p ∧ n = p^3)

/-- The first ten "nicely odd" numbers -/
def FirstTenNicelyOdd : Finset ℕ :=
  sorry

theorem sum_first_ten_nicely_odd :
  (FirstTenNicelyOdd.sum id) = 775 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_first_ten_nicely_odd_l1271_127128


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_constant_term_is_15_l1271_127177

/-- Given a real number a, this function represents the expansion of (1+x+ax³)(x+1/x)^5 --/
noncomputable def expansion (a : ℝ) (x : ℝ) : ℝ := (1 + x + a*x^3) * (x + 1/x)^5

/-- This function represents the sum of coefficients in the expansion --/
def sum_of_coefficients (a : ℝ) : ℝ := 
  (a + 2) * 2^5

/-- This theorem states that if the sum of coefficients is 96, then the constant term is 15 --/
theorem constant_term_is_15 (a : ℝ) (h : sum_of_coefficients a = 96) :
  ∃ (f : ℝ → ℝ), (∀ x ≠ 0, f x = expansion a x) ∧ (f 0 = 15) := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_constant_term_is_15_l1271_127177


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_kyler_wins_seven_l1271_127100

/-- Represents a player in the chess tournament -/
inductive Player
| Peter
| Emma
| Kyler

/-- Represents the number of games won and lost by a player -/
structure GameRecord where
  wins : ℕ
  losses : ℕ

/-- The tournament results -/
def tournamentResults : Player → GameRecord
  | Player.Peter => ⟨5, 2⟩
  | Player.Emma => ⟨6, 1⟩
  | Player.Kyler => ⟨0, 5⟩  -- Initialize Kyler's wins to 0, we'll prove it's 7 later

/-- The wins form an arithmetic progression -/
axiom wins_in_arithmetic_progression :
  ∃ (a d : ℤ), (tournamentResults Player.Peter).wins = (a - d).toNat ∧
               (tournamentResults Player.Emma).wins = a.toNat ∧
               (tournamentResults Player.Kyler).wins = (a + d).toNat

theorem kyler_wins_seven :
  (tournamentResults Player.Kyler).wins = 7 := by
  sorry

#check kyler_wins_seven

end NUMINAMATH_CALUDE_ERRORFEEDBACK_kyler_wins_seven_l1271_127100


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_alex_cakes_l1271_127180

theorem alex_cakes (slices_per_cake : ℕ) (slices_left : ℕ) (slices_eaten : ℕ) :
  slices_per_cake = 8 →
  slices_left = 5 →
  slices_eaten = 3 →
  ∃ (initial_cakes : ℕ),
    let initial_slices := initial_cakes * slices_per_cake
    let remaining_after_friends := (3 / 4 : ℚ) * initial_slices
    let remaining_after_family := (1 / 2 : ℚ) * initial_slices
    (remaining_after_family : ℚ) - slices_eaten = slices_left ∧
    initial_cakes = 2 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_alex_cakes_l1271_127180


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_goat_grazing_area_theorem_l1271_127169

/-- The area a goat can graze when tethered to the corner of a rectangular barn -/
noncomputable def goatGrazingArea (barnWidth : ℝ) (barnLength : ℝ) (leashLength : ℝ) : ℝ :=
  (3 / 4) * Real.pi * leashLength^2

/-- Theorem: The area a goat can graze when tethered to the corner of a 4m by 6m rectangular barn
    with a 5m leash, only able to go around the outside of the barn, is equal to (75/4)π square meters -/
theorem goat_grazing_area_theorem (barnWidth barnLength leashLength : ℝ)
    (h1 : barnWidth = 4)
    (h2 : barnLength = 6)
    (h3 : leashLength = 5) :
    goatGrazingArea barnWidth barnLength leashLength = (75 / 4) * Real.pi := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_goat_grazing_area_theorem_l1271_127169


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_statement_two_correct_statement_one_incorrect_statement_three_incorrect_statement_four_incorrect_l1271_127111

theorem statement_two_correct :
  ∃ (x : ℝ), (x^2 - 2*x - 3 = 0 ∧ x ≠ 3) ∧
  (∀ (y : ℝ), y = 3 → y^2 - 2*y - 3 = 0) := by
  sorry

theorem statement_one_incorrect :
  ∃ (a b m : ℝ), a < b ∧ ¬(a*m^2 < b*m^2) := by
  sorry

theorem statement_three_incorrect :
  ∃ (p q : Prop), p ∨ q ∧ ¬(p ∧ q) := by
  sorry

theorem statement_four_incorrect :
  ∃ (x : ℝ), x > 1 ∧ ¬(x > 2) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_statement_two_correct_statement_one_incorrect_statement_three_incorrect_statement_four_incorrect_l1271_127111


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_set_equality_l1271_127182

theorem set_equality : {x : ℕ | x < 5} = {0, 1, 2, 3, 4} := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_set_equality_l1271_127182


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_problem_l1271_127137

theorem triangle_problem (A B C : ℝ) (h_triangle : A + B + C = Real.pi) 
  (h_cos_A : Real.cos A = -5/13) (h_cos_B : Real.cos B = 3/5) (h_BC : BC = 5) :
  Real.sin C = 16/65 ∧ (1/2 * 5 * AC * Real.sin C = 8/3) := by
  sorry
where
  BC := Real.sqrt ((B - C)^2)
  AC := Real.sqrt ((A - C)^2)

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_problem_l1271_127137


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_circle_intersections_line_line_intersections_circle_line_intersections_max_intersections_theorem_l1271_127198

/-- The maximum number of intersection points between two circles and three lines -/
def max_intersections : ℕ := 17

/-- Two circles can intersect at most in 2 points -/
theorem circle_circle_intersections : ℕ := 2

/-- Three lines can intersect in at most 3 points -/
theorem line_line_intersections : ℕ := 3

/-- Each line can intersect each circle in at most 2 points -/
theorem circle_line_intersections (num_circles num_lines : ℕ) : ℕ := 2 * num_circles * num_lines

/-- Theorem: The maximum number of intersection points between 2 circles and 3 straight lines is 17 -/
theorem max_intersections_theorem : 
  circle_circle_intersections + line_line_intersections + circle_line_intersections 2 3 = max_intersections := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_circle_intersections_line_line_intersections_circle_line_intersections_max_intersections_theorem_l1271_127198


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_class_composition_theorem_l1271_127103

/-- Represents the number of students in a class -/
structure ClassComposition where
  female : ℕ
  male : ℕ

/-- Calculates the simplest integer ratio of male students to total students -/
def maleToTotalRatio (c : ClassComposition) : ℚ :=
  c.male / (c.female + c.male)

/-- Calculates the percentage by which female students are less than male students -/
def femaleLessThanMalePercentage (c : ClassComposition) : ℚ :=
  (c.male - c.female : ℚ) / c.male * 100

theorem class_composition_theorem (c : ClassComposition) 
    (h1 : c.female = 15) (h2 : c.male = 25) : 
    maleToTotalRatio c = 5 / 8 ∧ 
    femaleLessThanMalePercentage c = 40 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_class_composition_theorem_l1271_127103


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_right_triangle_hypotenuse_l1271_127192

theorem right_triangle_hypotenuse (P Q R : EuclideanSpace ℝ (Fin 2)) (tanQ : ℝ) (QP : ℝ) :
  (‖R - Q‖ = ‖P - Q‖) →  -- Right angle at Q
  (tanQ = 0.5) →
  (‖P - Q‖ = 16) →
  ‖R - Q‖ = 8 * Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_right_triangle_hypotenuse_l1271_127192


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_m_no_intersection_chord_length_when_m_zero_l1271_127107

-- Define the line l
noncomputable def line_l (m t : ℝ) : ℝ × ℝ := (t / 2, m + (Real.sqrt 3 / 2) * t)

-- Define the curve C in Cartesian coordinates
def curve_C (x y : ℝ) : Prop := x^2 + y^2 - 2*x - 4 = 0

-- Theorem for the range of m
theorem range_of_m_no_intersection (m : ℝ) : 
  (∀ t : ℝ, ¬ curve_C (line_l m t).1 (line_l m t).2) ↔ 
  (m < -Real.sqrt 3 - 2 * Real.sqrt 5 ∨ m > -Real.sqrt 3 + 2 * Real.sqrt 5) :=
by sorry

-- Theorem for the length of the chord when m = 0
theorem chord_length_when_m_zero : 
  ∃ p q : ℝ × ℝ, 
    curve_C p.1 p.2 ∧ 
    curve_C q.1 q.2 ∧ 
    (∃ t₁ t₂ : ℝ, p = line_l 0 t₁ ∧ q = line_l 0 t₂) ∧
    Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2) = Real.sqrt 17 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_m_no_intersection_chord_length_when_m_zero_l1271_127107


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_curves_intersection_l1271_127171

-- Define the curves C₁, C₂, and C₃
noncomputable def C₁ (a : ℝ) (t : ℝ) : ℝ × ℝ := (a * Real.cos t, 1 + a * Real.sin t)

noncomputable def C₂ (θ : ℝ) : ℝ := 4 * Real.cos θ

noncomputable def C₃ (α₀ : ℝ) (x : ℝ) : ℝ := Real.tan α₀ * x

-- Define the condition that tan α₀ = 2
def tan_α₀_eq_2 (α₀ : ℝ) : Prop := Real.tan α₀ = 2

-- Define the condition that all common points of C₁ and C₂ lie on C₃
def common_points_on_C₃ (a : ℝ) (α₀ : ℝ) : Prop :=
  ∀ t θ, C₁ a t = (C₂ θ * Real.cos θ, C₂ θ * Real.sin θ) →
    (C₁ a t).2 = C₃ α₀ (C₁ a t).1

-- Theorem statement
theorem curves_intersection (a : ℝ) (α₀ : ℝ) 
  (h₁ : a > 0) 
  (h₂ : tan_α₀_eq_2 α₀) 
  (h₃ : common_points_on_C₃ a α₀) : 
  a = 1 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_curves_intersection_l1271_127171


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_distance_difference_l1271_127138

/-- Given points A and B, and a line l, find the point P on l that maximizes |PA| - |PB| --/
theorem max_distance_difference (A B P : ℝ × ℝ) (l : Set (ℝ × ℝ)) : 
  A = (4, 1) →
  B = (0, 4) →
  l = {(x, y) | 3 * x - y - 1 = 0} →
  P ∈ l →
  P = (2, 5) →
  ∀ Q ∈ l, dist A P - dist B P ≥ dist A Q - dist B Q :=
by sorry

/-- Helper function to calculate Euclidean distance between two points --/
noncomputable def dist (p q : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_distance_difference_l1271_127138


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pie_chart_most_suitable_example_class_pie_chart_suitable_l1271_127188

/-- Represents different types of statistical graphs -/
inductive GraphType
| BarGraph
| PieChart
| LineGraph

/-- Represents a class with different categories of students -/
structure ClassComposition where
  total_students : ℕ
  excellent_students : ℕ
  outstanding_leaders : ℕ

/-- Determines if a graph type is suitable for showing percentage relationships -/
def is_suitable_for_percentages (graph : GraphType) : Prop :=
  match graph with
  | GraphType.PieChart => True
  | _ => False

/-- Theorem stating that a pie chart is the most suitable graph type for the given class composition -/
theorem pie_chart_most_suitable (c : ClassComposition) :
  is_suitable_for_percentages GraphType.PieChart ∧
  (∀ g : GraphType, is_suitable_for_percentages g → g = GraphType.PieChart) := by
  sorry

/-- Example class composition -/
def example_class : ClassComposition :=
  { total_students := 50
  , excellent_students := 10
  , outstanding_leaders := 5 }

/-- Applying the theorem to the example class -/
theorem example_class_pie_chart_suitable :
  is_suitable_for_percentages GraphType.PieChart ∧
  (∀ g : GraphType, is_suitable_for_percentages g → g = GraphType.PieChart) := by
  exact pie_chart_most_suitable example_class


end NUMINAMATH_CALUDE_ERRORFEEDBACK_pie_chart_most_suitable_example_class_pie_chart_suitable_l1271_127188


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_product_greater_sum_l1271_127113

def S : Finset ℕ := Finset.filter (λ n => 1 ≤ n ∧ n ≤ 6) (Finset.range 7)

def favorable_pairs : Finset (ℕ × ℕ) :=
  Finset.filter (λ (a, b) => a * b > a + b) (S.product S)

theorem probability_product_greater_sum :
  (favorable_pairs.card : ℚ) / (S.card ^ 2 : ℚ) = 1 / 2 := by
  sorry

#eval favorable_pairs.card
#eval S.card ^ 2

end NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_product_greater_sum_l1271_127113


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_property_l1271_127120

def sequenceList : List ℕ := [11, 12, 27, 11, 12, 27, 11, 12]

def sum_of_three (l : List ℕ) (i : ℕ) : ℕ :=
  (l.get? i).getD 0 + (l.get? (i+1)).getD 0 + (l.get? (i+2)).getD 0

theorem sequence_property :
  sequenceList.length = 8 ∧
  (∀ i, i + 2 < sequenceList.length → sum_of_three sequenceList i = 50) ∧
  sequenceList.head? = some 11 ∧
  sequenceList.getLast? = some 12 ∧
  (∀ other : List ℕ,
    other.length = 8 ∧
    (∀ i, i + 2 < other.length → sum_of_three other i = 50) ∧
    other.head? = some 11 ∧
    other.getLast? = some 12 →
    other = sequenceList) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_property_l1271_127120


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_seventh_circle_coincides_l1271_127131

/-- Represents a circle inscribed in a triangle --/
structure InscribedCircle where
  radius : ℝ
  angle : ℝ

/-- Represents a triangle with a sequence of inscribed circles --/
structure TriangleWithCircles where
  A : ℝ  -- Angle A
  B : ℝ  -- Angle B
  C : ℝ  -- Angle C
  S : Fin 7 → InscribedCircle

/-- The condition that circles touch each other in the specified pattern --/
noncomputable def circles_touch (t : TriangleWithCircles) : Prop :=
  ∀ i : Fin 6, 
    (t.S i).radius + (t.S (i + 1)).radius = 
      Real.sqrt ((t.S i).radius * (t.S (i + 1)).radius * 
        (Real.tan ((t.S i).angle / 2)⁻¹ + Real.tan ((t.S (i + 1)).angle / 2)⁻¹))

/-- The theorem stating that S₇ coincides with S₁ --/
theorem seventh_circle_coincides (t : TriangleWithCircles) 
  (h1 : t.A + t.B + t.C = π)
  (h2 : (t.S 0).angle = t.A)
  (h3 : (t.S 1).angle = t.B)
  (h4 : (t.S 2).angle = t.C)
  (h5 : circles_touch t) :
  (t.S 0).radius = (t.S 6).radius := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_seventh_circle_coincides_l1271_127131


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_increasing_intervals_g_value_at_pi_sixth_l1271_127173

noncomputable section

open Real

def f (x : ℝ) : ℝ := 2 * sqrt 3 * sin (π - x) * sin x - (sin x - cos x)^2

def g (x : ℝ) : ℝ := f (x/2 + π/3)

theorem f_increasing_intervals (x : ℝ) :
  (∃ k : ℤ, x ∈ Set.Icc ((k : ℝ) * π - π/12) ((k : ℝ) * π + 5*π/12)) ↔ deriv f x > 0 := by sorry

theorem g_value_at_pi_sixth : g (π/6) = sqrt 3 := by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_increasing_intervals_g_value_at_pi_sixth_l1271_127173


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_of_prime_ball_l1271_127112

def balls : Finset ℕ := {5, 6, 7, 8, 9, 10, 11, 12}

def is_prime (n : ℕ) : Prop := Nat.Prime n

def prime_balls : Finset ℕ := balls.filter (fun n => Nat.Prime n)

theorem probability_of_prime_ball : 
  (prime_balls.card : ℚ) / (balls.card : ℚ) = 3 / 8 := by
  -- Evaluate prime_balls
  have h1 : prime_balls = {5, 7, 11} := by
    simp [prime_balls, balls, Nat.Prime]
    -- You might need to prove primality for each number here
    sorry
  
  -- Calculate cardinalities
  have h2 : prime_balls.card = 3 := by simp [h1]
  have h3 : balls.card = 8 := by simp [balls]
  
  -- Perform the division
  calc
    (prime_balls.card : ℚ) / (balls.card : ℚ) 
    = 3 / 8 := by
      rw [h2, h3]
      -- You might need to prove this equality
      sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_of_prime_ball_l1271_127112


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_chord_length_l1271_127150

/-- The length of the chord formed by the intersection of a circle and a line --/
theorem chord_length :
  ∃ (x y : ℝ → ℝ) (t : ℝ),
  (∀ θ : ℝ, (x (Real.cos θ) - 2)^2 + (y (Real.sin θ))^2 = 4) →  -- Circle equation
  (∀ t : ℝ, x t = 4 * t^2 ∧ y t = 4 * t) →             -- Parametric equation of C₂
  (∀ x : ℝ, y x = 2 * x - 2) →                         -- Line equation
  ∃ chord : ℝ, chord = 8 * Real.sqrt 5 / 5 :=           -- Chord length
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_chord_length_l1271_127150


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fake_coin_strategy_exists_l1271_127129

/-- Represents a strategy to find a fake coin -/
structure FakeCoinStrategy (k : ℕ) where
  /-- The number of tests required by the strategy -/
  num_tests : ℕ
  /-- Proof that the strategy finds the fake coin -/
  finds_fake_coin : True
  /-- Proof that the number of tests is at most 2^k + k + 2 -/
  test_bound : num_tests ≤ 2^k + k + 2

/-- Represents the action of a dog barking -/
def barks (dog : ℕ) : Prop := sorry

/-- Theorem stating the existence of a strategy to find the fake coin -/
theorem fake_coin_strategy_exists (k : ℕ) :
  ∃ (strategy : FakeCoinStrategy k),
    /- The strategy works for 2^(2^k) coins -/
    (∃ (n : ℕ), n = 2^(2^k)) ∧
    /- There is exactly one fake coin -/
    (∃ (fake : ℕ), fake < 2^(2^k)) ∧
    /- There are unlimited service dogs -/
    (∀ (d : ℕ), ∃ (dog : ℕ), dog = d) ∧
    /- One dog is sick (but we don't know which one) -/
    (∃ (sick : ℕ), sick ≥ 0) ∧
    /- A healthy dog barks iff the fake coin is among the tested coins -/
    (∀ (test : Finset ℕ), (fake ∈ test) ↔ (∃ (healthy_dog : ℕ), healthy_dog ≠ sick → barks healthy_dog)) ∧
    /- The behavior of the sick dog is random -/
    (∀ (test : Finset ℕ), ∃ (bark_prob : ℚ), 0 ≤ bark_prob ∧ bark_prob ≤ 1) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_fake_coin_strategy_exists_l1271_127129


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_four_solutions_in_interval_l1271_127196

noncomputable def g (n : ℕ) (x : ℝ) : ℝ := Real.sin x ^ n + Real.cos x ^ n + Real.tan x ^ n

theorem four_solutions_in_interval :
  ∃ (S : Finset ℝ), S.card = 4 ∧
  (∀ x ∈ S, 0 ≤ x ∧ x ≤ Real.pi / 2) ∧
  (∀ x ∈ S, 4 * g 4 x - 3 * g 6 x = g 2 x + 1) ∧
  (∀ x, 0 ≤ x ∧ x ≤ Real.pi / 2 → 4 * g 4 x - 3 * g 6 x = g 2 x + 1 → x ∈ S) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_four_solutions_in_interval_l1271_127196


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_place_mat_length_theorem_l1271_127154

/-- The length of a place mat on a circular table --/
noncomputable def place_mat_length (r : ℝ) (n : ℕ) (w : ℝ) : ℝ :=
  (3 * Real.sqrt 35) / 10 + 1 / 2

theorem place_mat_length_theorem (r w x : ℝ) (n : ℕ) :
  r = 3 ∧ n = 8 ∧ w = 1 →
  x = place_mat_length r n w :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_place_mat_length_theorem_l1271_127154


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l1271_127119

noncomputable def f (x : ℝ) : ℝ := x^3 - 1/x^3

theorem f_properties :
  (∀ x : ℝ, x ≠ 0 → f (-x) = -f x) ∧
  (∀ x y : ℝ, 0 < x → 0 < y → x < y → f x < f y) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l1271_127119


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_givers_29_students_l1271_127160

/-- Represents a student in the ball exchange scenario -/
structure Student where
  given : ℕ
  received : ℕ

/-- The maximum number of students who can truthfully claim they gave more than they received -/
def max_givers (students : List Student) : ℕ :=
  students.filter (fun s => s.given > s.received) |>.length

/-- The theorem stating the maximum number of givers in a group of 29 students -/
theorem max_givers_29_students :
  ∀ (students : List Student),
    students.length = 29 →
    (students.map Student.given).sum = (students.map Student.received).sum →
    max_givers students ≤ 14 := by
  sorry

#check max_givers_29_students

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_givers_29_students_l1271_127160


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_area_circle_l1271_127133

/-- The curve on which the center of the circle lies -/
noncomputable def center_curve (x : ℝ) : ℝ := 2 / x

/-- The line to which the circle is tangent -/
def tangent_line (x y : ℝ) : ℝ := 2*x + y + 1

/-- The distance from a point (a, 2/a) to the tangent line -/
noncomputable def distance_to_line (a : ℝ) : ℝ := |2*a + 2/a + 1| / Real.sqrt 5

/-- The equation of the circle with center (1, 2) and radius √5 -/
def circle_equation (x y : ℝ) : Prop := (x - 1)^2 + (y - 2)^2 = 5

theorem smallest_area_circle :
  ∀ a > 0,
  ∀ x y : ℝ,
  y = center_curve x →
  (x - a)^2 + (y - center_curve a)^2 = (distance_to_line a)^2 →
  tangent_line x y = 0 →
  (x - a)^2 + (y - center_curve a)^2 ≥ 5 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_area_circle_l1271_127133


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_correct_l1271_127145

/-- The probability of a circular token covering part of the black region on a specially marked rectangle --/
noncomputable def token_black_region_probability : ℝ → ℝ → ℝ → ℝ → ℝ :=
  fun rectangle_length rectangle_width triangle_leg_length token_diameter =>
    let token_radius := token_diameter / 2
    let valid_center_area := (rectangle_length - 2 * token_radius) * (rectangle_width - 2 * token_radius)
    let triangle_area := triangle_leg_length * triangle_leg_length / 2
    let additional_area := Real.pi * token_radius * token_radius / 4 + triangle_leg_length * token_radius / Real.sqrt 2
    let total_black_area := 2 * (triangle_area + additional_area)
    total_black_area / valid_center_area

/-- The probability is correct for the given dimensions --/
theorem probability_correct :
  token_black_region_probability 10 6 3 2 = (9 + Real.pi/2 + 3 * Real.sqrt 2) / 32 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_correct_l1271_127145


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_octagon_in_square_l1271_127124

/-- The area of a regular octagon inscribed in a square --/
noncomputable def octagon_area (square_perimeter : ℝ) : ℝ :=
  2 * (square_perimeter / 4)^2

/-- The radius of a circle inscribed in a regular octagon --/
noncomputable def inscribed_circle_radius (octagon_area : ℝ) : ℝ :=
  Real.sqrt (octagon_area / (2 * (1 + Real.sqrt 2)))

theorem octagon_in_square (square_perimeter : ℝ) 
  (h : square_perimeter = 160) :
  let area := octagon_area square_perimeter
  let radius := inscribed_circle_radius area
  area = 1600 ∧ 21.64 < radius ∧ radius < 21.66 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_octagon_in_square_l1271_127124
