import Mathlib

namespace NUMINAMATH_CALUDE_max_value_sine_sum_l761_76189

theorem max_value_sine_sum : 
  ∀ x : ℝ, 3 * Real.sin (x + π/9) + 5 * Real.sin (x + 4*π/9) ≤ 7 ∧ 
  ∃ x : ℝ, 3 * Real.sin (x + π/9) + 5 * Real.sin (x + 4*π/9) = 7 :=
sorry

end NUMINAMATH_CALUDE_max_value_sine_sum_l761_76189


namespace NUMINAMATH_CALUDE_largest_n_satisfying_inequality_l761_76181

theorem largest_n_satisfying_inequality :
  ∃ n : ℤ, (1/3 : ℚ) + (n : ℚ)/7 < 1 ∧
  n = 4 ∧
  ∀ m : ℤ, (1/3 : ℚ) + (m : ℚ)/7 < 1 → m ≤ n :=
by sorry

end NUMINAMATH_CALUDE_largest_n_satisfying_inequality_l761_76181


namespace NUMINAMATH_CALUDE_triangle_side_length_l761_76134

-- Define the triangle
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ

-- State the theorem
theorem triangle_side_length (t : Triangle) :
  (t.b - t.c = 2) →
  (1/2 * t.b * t.c * Real.sqrt (1 - (-1/4)^2) = 3 * Real.sqrt 15) →
  (Real.cos t.A = -1/4) →
  t.a = 8 := by
  sorry

end NUMINAMATH_CALUDE_triangle_side_length_l761_76134


namespace NUMINAMATH_CALUDE_power_fraction_simplification_l761_76190

theorem power_fraction_simplification :
  (3^2015 + 3^2013) / (3^2015 - 3^2013) = 5/4 := by
  sorry

end NUMINAMATH_CALUDE_power_fraction_simplification_l761_76190


namespace NUMINAMATH_CALUDE_license_plate_count_l761_76136

/-- The number of letters in the alphabet -/
def num_letters : ℕ := 26

/-- The number of odd digits -/
def num_odd_digits : ℕ := 5

/-- The number of even digits -/
def num_even_digits : ℕ := 5

/-- The number of license plates with the given conditions -/
def num_license_plates : ℕ := num_letters^3 * num_odd_digits^2 * num_even_digits

theorem license_plate_count : num_license_plates = 2197000 := by
  sorry

end NUMINAMATH_CALUDE_license_plate_count_l761_76136


namespace NUMINAMATH_CALUDE_total_students_at_concert_l761_76109

/-- The number of buses going to the concert -/
def num_buses : ℕ := 18

/-- The number of students each bus took -/
def students_per_bus : ℕ := 65

/-- Theorem stating the total number of students who went to the concert -/
theorem total_students_at_concert : num_buses * students_per_bus = 1170 := by
  sorry

end NUMINAMATH_CALUDE_total_students_at_concert_l761_76109


namespace NUMINAMATH_CALUDE_binomial_expectation_and_variance_l761_76167

/-- A random variable following a binomial distribution with n trials and probability p -/
structure BinomialDistribution (n : ℕ) (p : ℝ) where
  ξ : ℝ → ℝ  -- The random variable

/-- The expected value of a random variable -/
def expectation (X : ℝ → ℝ) : ℝ := sorry

/-- The variance of a random variable -/
def variance (X : ℝ → ℝ) : ℝ := sorry

theorem binomial_expectation_and_variance 
  (ξ : BinomialDistribution 5 (1/2)) 
  (η : ℝ → ℝ) 
  (h : η = λ x => 5 * ξ.ξ x) : 
  expectation η = 25/2 ∧ variance η = 125/4 := by sorry

end NUMINAMATH_CALUDE_binomial_expectation_and_variance_l761_76167


namespace NUMINAMATH_CALUDE_polynomial_divisibility_l761_76180

-- Define the polynomial ring over the complex numbers
variable (x : ℂ)

-- Define the polynomials
variable (f g h k : ℂ → ℂ)

-- Define the conditions
axiom condition1 : ∀ x, (x^2 + 1) * (h x) + (x - 1) * (f x) + (x - 2) * (g x) = 0
axiom condition2 : ∀ x, (x^2 + 1) * (k x) + (x + 1) * (f x) + (x + 2) * (g x) = 0

-- State the theorem
theorem polynomial_divisibility :
  (∃ q : ℂ → ℂ, ∀ x, f x * g x = (x^2 + 1) * q x) :=
sorry

end NUMINAMATH_CALUDE_polynomial_divisibility_l761_76180


namespace NUMINAMATH_CALUDE_packing_problem_l761_76173

theorem packing_problem :
  ∃! n : ℕ, 500 ≤ n ∧ n ≤ 600 ∧ n % 20 = 13 ∧ n % 27 = 20 ∧ n = 533 := by
  sorry

end NUMINAMATH_CALUDE_packing_problem_l761_76173


namespace NUMINAMATH_CALUDE_infinitely_many_cube_sums_l761_76110

theorem infinitely_many_cube_sums (n : ℕ) : 
  ∃ (f : ℕ → ℕ), Function.Injective f ∧ 
  ∀ (k : ℕ), ∃ (m : ℕ+), (n^6 + 3 * (f k)) = m^3 := by
  sorry

end NUMINAMATH_CALUDE_infinitely_many_cube_sums_l761_76110


namespace NUMINAMATH_CALUDE_complex_solutions_count_l761_76152

open Complex

theorem complex_solutions_count : 
  ∃ (S : Finset ℂ), (∀ z ∈ S, (z^4 - 1) / (z^2 + z + 1) = 0) ∧ 
                    (∀ z : ℂ, (z^4 - 1) / (z^2 + z + 1) = 0 → z ∈ S) ∧
                    Finset.card S = 4 := by
  sorry

end NUMINAMATH_CALUDE_complex_solutions_count_l761_76152


namespace NUMINAMATH_CALUDE_min_rooms_for_departments_l761_76183

/-- Given two departments with student counts and room constraints, 
    calculate the minimum number of rooms required. -/
theorem min_rooms_for_departments (dept1_count dept2_count : ℕ) : 
  dept1_count = 72 →
  dept2_count = 5824 →
  ∃ (room_size : ℕ), 
    room_size > 0 ∧
    dept1_count % room_size = 0 ∧
    dept2_count % room_size = 0 ∧
    (dept1_count / room_size + dept2_count / room_size) = 737 := by
  sorry

end NUMINAMATH_CALUDE_min_rooms_for_departments_l761_76183


namespace NUMINAMATH_CALUDE_tank_full_time_l761_76117

/-- Represents a water tank with pipes for filling and draining. -/
structure WaterTank where
  capacity : ℕ
  pipeA_rate : ℕ
  pipeB_rate : ℕ
  pipeC_rate : ℕ

/-- Calculates the time required to fill the tank given the pipe rates and capacity. -/
def time_to_fill (tank : WaterTank) : ℕ :=
  let net_fill_per_cycle := tank.pipeA_rate + tank.pipeB_rate - tank.pipeC_rate
  let cycles := tank.capacity / net_fill_per_cycle
  cycles * 3

/-- Theorem stating that the given tank will be full after 48 minutes. -/
theorem tank_full_time (tank : WaterTank) 
  (h1 : tank.capacity = 800)
  (h2 : tank.pipeA_rate = 40)
  (h3 : tank.pipeB_rate = 30)
  (h4 : tank.pipeC_rate = 20) :
  time_to_fill tank = 48 := by
  sorry

#eval time_to_fill { capacity := 800, pipeA_rate := 40, pipeB_rate := 30, pipeC_rate := 20 }

end NUMINAMATH_CALUDE_tank_full_time_l761_76117


namespace NUMINAMATH_CALUDE_tom_search_days_l761_76162

/-- Calculates the number of days Tom searched for an item given the daily rates and total cost -/
def search_days (initial_rate : ℕ) (initial_days : ℕ) (subsequent_rate : ℕ) (total_cost : ℕ) : ℕ :=
  let initial_cost := initial_rate * initial_days
  let remaining_cost := total_cost - initial_cost
  let additional_days := remaining_cost / subsequent_rate
  initial_days + additional_days

/-- Proves that Tom searched for 10 days given the specified rates and total cost -/
theorem tom_search_days :
  search_days 100 5 60 800 = 10 := by
  sorry

end NUMINAMATH_CALUDE_tom_search_days_l761_76162


namespace NUMINAMATH_CALUDE_min_value_problem_l761_76118

-- Define the function f
def f (x a b c : ℝ) : ℝ := |x + a| + |x - b| + c

-- State the theorem
theorem min_value_problem (a b c : ℝ) 
  (ha : a > 0) (hb : b > 0) (hc : c > 0)
  (hmin : ∀ x, f x a b c ≥ 4) 
  (hex : ∃ x, f x a b c = 4) : 
  (a + b + c = 4) ∧ 
  (∀ a' b' c' : ℝ, a' > 0 → b' > 0 → c' > 0 → a' + b' + c' = 4 → 
    (1/4) * a'^2 + (1/9) * b'^2 + c'^2 ≥ 8/7) ∧
  (∃ a' b' c' : ℝ, a' > 0 ∧ b' > 0 ∧ c' > 0 ∧ a' + b' + c' = 4 ∧
    (1/4) * a'^2 + (1/9) * b'^2 + c'^2 = 8/7) :=
by sorry

end NUMINAMATH_CALUDE_min_value_problem_l761_76118


namespace NUMINAMATH_CALUDE_tangent_line_y_intercept_l761_76150

-- Define the circles
def circle1_center : ℝ × ℝ := (3, 0)
def circle1_radius : ℝ := 3
def circle2_center : ℝ × ℝ := (8, 0)
def circle2_radius : ℝ := 2

-- Define the tangent line
def tangent_line (y_intercept : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | ∃ (m : ℝ), p.2 = m * p.1 + y_intercept}

-- Define the condition for the line to be tangent to a circle
def is_tangent_to_circle (line : Set (ℝ × ℝ)) (center : ℝ × ℝ) (radius : ℝ) : Prop :=
  ∃ (p : ℝ × ℝ), p ∈ line ∧ 
    (p.1 - center.1) ^ 2 + (p.2 - center.2) ^ 2 = radius ^ 2 ∧
    ∀ (q : ℝ × ℝ), q ∈ line → q ≠ p → 
      (q.1 - center.1) ^ 2 + (q.2 - center.2) ^ 2 > radius ^ 2

-- Theorem statement
theorem tangent_line_y_intercept : 
  ∃ (y_intercept : ℝ), 
    y_intercept = 2 * Real.sqrt 104 ∧
    let line := tangent_line y_intercept
    is_tangent_to_circle line circle1_center circle1_radius ∧
    is_tangent_to_circle line circle2_center circle2_radius ∧
    ∀ (p : ℝ × ℝ), p ∈ line → p.1 ≥ 0 ∧ p.2 ≥ 0 := by
  sorry

end NUMINAMATH_CALUDE_tangent_line_y_intercept_l761_76150


namespace NUMINAMATH_CALUDE_trigonometric_identities_l761_76178

theorem trigonometric_identities (α : Real) (h : Real.tan α = 2) :
  (Real.cos (π/2 + α) * Real.sin (3*π/2 - α)) / Real.tan (-π + α) = 1/5 ∧
  (1 + 3*Real.sin α*Real.cos α) / (Real.sin α^2 - 2*Real.cos α^2) = 11/2 := by
  sorry

end NUMINAMATH_CALUDE_trigonometric_identities_l761_76178


namespace NUMINAMATH_CALUDE_quadratic_function_range_l761_76196

/-- A quadratic function -/
def QuadraticFunction (f : ℝ → ℝ) : Prop :=
  ∃ a b c : ℝ, a ≠ 0 ∧ ∀ x, f x = a * x^2 + b * x + c

/-- The range of a function -/
def Range (f : ℝ → ℝ) (S : Set ℝ) : Prop :=
  ∀ y, y ∈ S ↔ ∃ x, f x = y

/-- The theorem statement -/
theorem quadratic_function_range
  (f g : ℝ → ℝ)
  (h1 : f = g)
  (h2 : QuadraticFunction f)
  (h3 : Range (f ∘ g) (Set.Ici 0)) :
  Range (fun x ↦ g x) (Set.Ici 0) := by
sorry

end NUMINAMATH_CALUDE_quadratic_function_range_l761_76196


namespace NUMINAMATH_CALUDE_simplify_radical_product_l761_76146

theorem simplify_radical_product (y : ℝ) (h : y > 0) :
  Real.sqrt (50 * y) * Real.sqrt (18 * y) * Real.sqrt (32 * y) = 30 * y * Real.sqrt (2 * y) := by
  sorry

end NUMINAMATH_CALUDE_simplify_radical_product_l761_76146


namespace NUMINAMATH_CALUDE_completing_square_l761_76151

theorem completing_square (x : ℝ) : x^2 + 2*x - 3 = 0 ↔ (x + 1)^2 = 4 := by sorry

end NUMINAMATH_CALUDE_completing_square_l761_76151


namespace NUMINAMATH_CALUDE_triangulation_theorem_l761_76184

/-- A triangulation of a convex polygon with interior points. -/
structure Triangulation where
  /-- The number of vertices in the original polygon. -/
  polygon_vertices : ℕ
  /-- The number of additional interior points. -/
  interior_points : ℕ
  /-- The property that no three interior points are collinear. -/
  no_collinear_interior : Prop

/-- The number of triangles in a triangulation. -/
def num_triangles (t : Triangulation) : ℕ :=
  2 * (t.polygon_vertices + t.interior_points) - 2

/-- The main theorem about the number of triangles in the specific triangulation. -/
theorem triangulation_theorem (t : Triangulation) 
  (h1 : t.polygon_vertices = 1000)
  (h2 : t.interior_points = 500)
  (h3 : t.no_collinear_interior) :
  num_triangles t = 2998 := by
  sorry

end NUMINAMATH_CALUDE_triangulation_theorem_l761_76184


namespace NUMINAMATH_CALUDE_triangle_construction_exists_l761_76131

-- Define the necessary types and structures
structure Point where
  x : ℝ
  y : ℝ

structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

-- Define the given points and line
variable (A B : Point)
variable (bisector : Line)

-- Define the reflection of a point over a line
def reflect (p : Point) (l : Line) : Point :=
  sorry

-- Define a function to check if three points are collinear
def collinear (p q r : Point) : Prop :=
  sorry

-- Define a function to check if a point lies on a line
def point_on_line (p : Point) (l : Line) : Prop :=
  sorry

-- Define a function to calculate the distance between two points
def distance (p q : Point) : ℝ :=
  sorry

-- Theorem statement
theorem triangle_construction_exists :
  ∃ C : Point,
    point_on_line C bisector ∧
    distance A C = distance (reflect A bisector) C ∧
    ¬ collinear C A B :=
  sorry

end NUMINAMATH_CALUDE_triangle_construction_exists_l761_76131


namespace NUMINAMATH_CALUDE_fib_divisibility_spacing_l761_76157

/-- Fibonacci sequence -/
def fib : ℕ → ℕ
  | 0 => 0
  | 1 => 1
  | n + 2 => fib (n + 1) + fib n

/-- Theorem: Numbers in Fibonacci sequence divisible by m are equally spaced -/
theorem fib_divisibility_spacing (m : ℕ) (h : m > 0) :
  ∃ d : ℕ, d > 0 ∧ ∀ n : ℕ, m ∣ fib n → m ∣ fib (n + d) :=
sorry

end NUMINAMATH_CALUDE_fib_divisibility_spacing_l761_76157


namespace NUMINAMATH_CALUDE_proposition_l761_76108

theorem proposition (a b c : ℝ) (h1 : a > b) (h2 : b > c) (h3 : a + b + c = 0) :
  (Real.sqrt (b^2 - a*c)) / a < Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_proposition_l761_76108


namespace NUMINAMATH_CALUDE_unique_solution_exponential_equation_l761_76170

theorem unique_solution_exponential_equation :
  ∃! x : ℝ, (2 : ℝ)^(4*x+2) * (4 : ℝ)^(2*x+8) = (8 : ℝ)^(3*x+7) :=
by sorry

end NUMINAMATH_CALUDE_unique_solution_exponential_equation_l761_76170


namespace NUMINAMATH_CALUDE_cubic_polynomial_property_l761_76128

theorem cubic_polynomial_property (n : ℕ+) : 
  ∃ k : ℤ, (n^3 : ℚ) + (3/2) * n^2 + (1/2) * n - 1 = k ∧ k % 3 = 2 := by
  sorry

end NUMINAMATH_CALUDE_cubic_polynomial_property_l761_76128


namespace NUMINAMATH_CALUDE_system_of_equations_solution_l761_76144

theorem system_of_equations_solution :
  ∃! (x y : ℝ), 3 * x + 4 * y = 16 ∧ 5 * x - 6 * y = 33 ∧ x = 6 ∧ y = -1/2 := by
  sorry

end NUMINAMATH_CALUDE_system_of_equations_solution_l761_76144


namespace NUMINAMATH_CALUDE_domain_intersection_and_union_range_of_p_l761_76132

def A : Set ℝ := {x | x^2 - x - 2 > 0}
def B : Set ℝ := {x | 3 - |x| ≥ 0}
def C (p : ℝ) : Set ℝ := {x | 4*x + p < 0}

theorem domain_intersection_and_union :
  (A ∩ B = {x | -3 ≤ x ∧ x < -1 ∨ 2 < x ∧ x ≤ 3}) ∧
  (A ∪ B = Set.univ) :=
sorry

theorem range_of_p (p : ℝ) :
  (C p ⊆ A) → p ≥ 4 :=
sorry

end NUMINAMATH_CALUDE_domain_intersection_and_union_range_of_p_l761_76132


namespace NUMINAMATH_CALUDE_supplement_of_angle_with_30_degree_complement_l761_76175

theorem supplement_of_angle_with_30_degree_complement :
  ∀ (angle : ℝ), 
  (90 - angle = 30) →
  (180 - angle = 120) :=
by
  sorry

end NUMINAMATH_CALUDE_supplement_of_angle_with_30_degree_complement_l761_76175


namespace NUMINAMATH_CALUDE_clock_in_probability_l761_76193

/-- The probability of an employee clocking in on time given a total time window and valid clock-in time -/
theorem clock_in_probability (total_window : ℕ) (valid_time : ℕ) 
  (h1 : total_window = 40) 
  (h2 : valid_time = 15) : 
  (valid_time : ℚ) / total_window = 3/8 := by
  sorry

end NUMINAMATH_CALUDE_clock_in_probability_l761_76193


namespace NUMINAMATH_CALUDE_dormitory_to_city_distance_prove_dormitory_to_city_distance_l761_76171

theorem dormitory_to_city_distance : ℝ → Prop :=
  fun D : ℝ =>
    (1/4 : ℝ) * D + (1/2 : ℝ) * D + 10 = D → D = 40

-- The proof is omitted
theorem prove_dormitory_to_city_distance :
  ∃ D : ℝ, dormitory_to_city_distance D :=
by
  sorry

end NUMINAMATH_CALUDE_dormitory_to_city_distance_prove_dormitory_to_city_distance_l761_76171


namespace NUMINAMATH_CALUDE_asteroid_fragments_proof_l761_76114

theorem asteroid_fragments_proof :
  ∃ (X n : ℕ), 
    X > 0 ∧ 
    n > 0 ∧ 
    X / 5 + 26 + n * (X / 7) = X ∧ 
    X = 70 := by
  sorry

end NUMINAMATH_CALUDE_asteroid_fragments_proof_l761_76114


namespace NUMINAMATH_CALUDE_disease_probabilities_l761_76104

/-- Represents a disease with its incidence rate and probability of showing symptom S -/
structure Disease where
  incidenceRate : ℝ
  probSymptomS : ℝ

/-- Given three diseases and their properties, proves statements about probabilities -/
theorem disease_probabilities (d₁ d₂ d₃ : Disease)
  (h₁ : d₁.incidenceRate = 0.02 ∧ d₁.probSymptomS = 0.4)
  (h₂ : d₂.incidenceRate = 0.05 ∧ d₂.probSymptomS = 0.18)
  (h₃ : d₃.incidenceRate = 0.005 ∧ d₃.probSymptomS = 0.6)
  (h_no_other : ∀ d, d ≠ d₁ ∧ d ≠ d₂ ∧ d ≠ d₃ → d.probSymptomS = 0) :
  let p_s := d₁.incidenceRate * d₁.probSymptomS +
             d₂.incidenceRate * d₂.probSymptomS +
             d₃.incidenceRate * d₃.probSymptomS
  p_s = 0.02 ∧
  (d₁.incidenceRate * d₁.probSymptomS) / p_s = 0.4 ∧
  (d₂.incidenceRate * d₂.probSymptomS) / p_s = 0.45 :=
by sorry


end NUMINAMATH_CALUDE_disease_probabilities_l761_76104


namespace NUMINAMATH_CALUDE_age_problem_l761_76174

theorem age_problem (a b c d : ℕ) : 
  a = b + 2 →
  b = 2 * c →
  d = b / 2 →
  a + b + c + d = 44 →
  b = 14 := by
sorry

end NUMINAMATH_CALUDE_age_problem_l761_76174


namespace NUMINAMATH_CALUDE_latin_square_symmetric_diagonal_l761_76154

/-- A Latin square of order 7 -/
def LatinSquare7 (A : Fin 7 → Fin 7 → Fin 7) : Prop :=
  ∀ i j : Fin 7, ∀ k : Fin 7, (∃! x : Fin 7, A i x = k) ∧ (∃! y : Fin 7, A y j = k)

/-- Symmetry with respect to the main diagonal -/
def SymmetricMatrix (A : Fin 7 → Fin 7 → Fin 7) : Prop :=
  ∀ i j : Fin 7, A i j = A j i

/-- All numbers from 1 to 7 appear on the main diagonal -/
def AllNumbersOnDiagonal (A : Fin 7 → Fin 7 → Fin 7) : Prop :=
  ∀ k : Fin 7, ∃ i : Fin 7, A i i = k

theorem latin_square_symmetric_diagonal 
  (A : Fin 7 → Fin 7 → Fin 7) 
  (h1 : LatinSquare7 A) 
  (h2 : SymmetricMatrix A) : 
  AllNumbersOnDiagonal A :=
sorry

end NUMINAMATH_CALUDE_latin_square_symmetric_diagonal_l761_76154


namespace NUMINAMATH_CALUDE_distance_after_time_l761_76107

theorem distance_after_time (adam_speed simon_speed : ℝ) (time : ℝ) (distance : ℝ) : 
  adam_speed = 5 →
  simon_speed = 12 →
  time = 5 →
  distance = 65 →
  (adam_speed * time) ^ 2 + (simon_speed * time) ^ 2 = distance ^ 2 := by
sorry

end NUMINAMATH_CALUDE_distance_after_time_l761_76107


namespace NUMINAMATH_CALUDE_circle_problem_l761_76124

noncomputable section

-- Define the line l: y = kx
def line_l (k : ℝ) (x y : ℝ) : Prop := y = k * x

-- Define circle C₁: (x-1)² + y² = 1
def circle_C₁ (x y : ℝ) : Prop := (x - 1)^2 + y^2 = 1

-- Define point M
def point_M : ℝ × ℝ := (3, Real.sqrt 3)

-- Define the tangency condition for C₂ and l at M
def tangent_C₂_l (k : ℝ) : Prop := line_l k 3 (Real.sqrt 3)

-- Define the external tangency condition for C₁ and C₂
def external_tangent_C₁_C₂ (m n R : ℝ) : Prop :=
  (m - 1)^2 + n^2 = (1 + R)^2

-- Main theorem
theorem circle_problem (k : ℝ) :
  (∃ m n R, external_tangent_C₁_C₂ m n R ∧ tangent_C₂_l k) →
  (k = Real.sqrt 3 / 3) ∧
  (∃ A B : ℝ × ℝ, circle_C₁ A.1 A.2 ∧ circle_C₁ B.1 B.2 ∧ line_l k A.1 A.2 ∧ line_l k B.1 B.2 ∧
    ((A.1 - B.1)^2 + (A.2 - B.2)^2 = 3)) ∧
  (∃ m n : ℝ, ((m = 4 ∧ n = 0) ∨ (m = 0 ∧ n = 4 * Real.sqrt 3)) ∧
    (∀ x y : ℝ, (x - m)^2 + (y - n)^2 = (if m = 4 then 4 else 36))) :=
sorry

end

end NUMINAMATH_CALUDE_circle_problem_l761_76124


namespace NUMINAMATH_CALUDE_arithmetic_sequence_problem_l761_76187

/-- An arithmetic sequence -/
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

/-- The main theorem -/
theorem arithmetic_sequence_problem (a : ℕ → ℝ) 
  (h_arith : arithmetic_sequence a) 
  (h_sum : a 1 + 3 * a 8 + a 15 = 60) : 
  2 * a 9 - a 10 = 12 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_problem_l761_76187


namespace NUMINAMATH_CALUDE_cube_volume_l761_76130

/-- The volume of a cube with total edge length 48 cm is 64 cm³. -/
theorem cube_volume (edge_sum : ℝ) (h : edge_sum = 48) : 
  (edge_sum / 12)^3 = 64 := by
  sorry

end NUMINAMATH_CALUDE_cube_volume_l761_76130


namespace NUMINAMATH_CALUDE_max_value_fourth_root_sum_l761_76153

theorem max_value_fourth_root_sum (a b c d : ℝ) 
  (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hd : 0 < d)
  (hsum : a + b + c + d ≤ 4) :
  (a^2 + 3*a*b)^(1/4) + (b^2 + 3*b*c)^(1/4) + (c^2 + 3*c*d)^(1/4) + (d^2 + 3*d*a)^(1/4) ≤ 4 * Real.sqrt 2 :=
by sorry

end NUMINAMATH_CALUDE_max_value_fourth_root_sum_l761_76153


namespace NUMINAMATH_CALUDE_teacups_left_result_l761_76191

/-- Calculates the number of teacups left after arranging --/
def teacups_left (total_boxes : ℕ) (pan_boxes : ℕ) (rows_per_box : ℕ) (cups_per_row : ℕ) (broken_per_box : ℕ) : ℕ :=
  let remaining_boxes := total_boxes - pan_boxes
  let decoration_boxes := remaining_boxes / 2
  let teacup_boxes := remaining_boxes - decoration_boxes
  let cups_per_box := rows_per_box * cups_per_row
  let total_cups := teacup_boxes * cups_per_box
  let broken_cups := teacup_boxes * broken_per_box
  total_cups - broken_cups

/-- Theorem stating the number of teacups left after arranging --/
theorem teacups_left_result : teacups_left 26 6 5 4 2 = 180 := by
  sorry

end NUMINAMATH_CALUDE_teacups_left_result_l761_76191


namespace NUMINAMATH_CALUDE_range_of_a_l761_76168

def point_P (a : ℝ) : ℝ × ℝ := (3*a - 9, a + 2)

def on_terminal_side (p : ℝ × ℝ) (α : ℝ) : Prop :=
  (p.1 ≥ 0 ∧ p.2 ≥ 0) ∨ (p.1 ≤ 0 ∧ p.2 ≥ 0) ∨ (p.1 ≤ 0 ∧ p.2 ≤ 0) ∨ (p.1 ≥ 0 ∧ p.2 ≤ 0)

theorem range_of_a (α : ℝ) :
  (∀ a : ℝ, on_terminal_side (point_P a) α ∧ Real.cos α ≤ 0 ∧ Real.sin α > 0) →
  (∀ a : ℝ, a ∈ Set.Ioc (-2) 3) :=
by sorry

end NUMINAMATH_CALUDE_range_of_a_l761_76168


namespace NUMINAMATH_CALUDE_no_constant_absolute_value_inequality_l761_76158

theorem no_constant_absolute_value_inequality :
  ¬ ∃ (a b c : ℝ), ∀ (x y : ℝ), 
    |x + a| + |x + y + b| + |y + c| > |x| + |x + y| + |y| := by
  sorry

end NUMINAMATH_CALUDE_no_constant_absolute_value_inequality_l761_76158


namespace NUMINAMATH_CALUDE_remainder_of_2_pow_1999_plus_1_mod_17_l761_76143

theorem remainder_of_2_pow_1999_plus_1_mod_17 :
  (2^1999 + 1) % 17 = 10 := by sorry

end NUMINAMATH_CALUDE_remainder_of_2_pow_1999_plus_1_mod_17_l761_76143


namespace NUMINAMATH_CALUDE_quadratic_sequence_bound_l761_76142

/-- Represents a quadratic equation ax² + bx + c = 0 -/
structure QuadraticEquation where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Represents the solutions of a quadratic equation -/
structure QuadraticSolution where
  x₁ : ℝ
  x₂ : ℝ

/-- Function to get the next quadratic equation in the sequence -/
def nextEquation (eq : QuadraticEquation) (sol : QuadraticSolution) : QuadraticEquation :=
  { a := 1, b := -sol.x₁, c := -sol.x₂ }

/-- Theorem stating that the sequence of quadratic equations has at most 5 elements -/
theorem quadratic_sequence_bound
  (a₁ b₁ : ℝ)
  (h₁ : a₁ ≠ 0)
  (h₂ : b₁ ≠ 0)
  (initial : QuadraticEquation)
  (h₃ : initial = { a := 1, b := a₁, c := b₁ })
  (next : QuadraticEquation → QuadraticSolution → QuadraticEquation)
  (h₄ : ∀ eq sol, next eq sol = nextEquation eq sol) :
  ∃ n : ℕ, n ≤ 5 ∧ ∀ m : ℕ, m > n →
    ¬∃ (seq : ℕ → QuadraticEquation) (sols : ℕ → QuadraticSolution),
      (seq 0 = initial) ∧
      (∀ k < m, seq (k + 1) = next (seq k) (sols k)) ∧
      (∀ k < m, (sols k).x₁ ≤ (sols k).x₂) :=
sorry

end NUMINAMATH_CALUDE_quadratic_sequence_bound_l761_76142


namespace NUMINAMATH_CALUDE_mans_rate_l761_76161

/-- The man's rate in still water given his speeds with and against the stream -/
theorem mans_rate (speed_with_stream speed_against_stream : ℝ) 
  (h1 : speed_with_stream = 24)
  (h2 : speed_against_stream = 10) : 
  (speed_with_stream + speed_against_stream) / 2 = 17 := by
  sorry

end NUMINAMATH_CALUDE_mans_rate_l761_76161


namespace NUMINAMATH_CALUDE_rectangle_area_l761_76127

/-- 
A rectangle with diagonal length x and length three times its width 
has an area of (3/10)x^2
-/
theorem rectangle_area (x : ℝ) (h : x > 0) : 
  ∃ w : ℝ, w > 0 ∧ 
    w^2 + (3*w)^2 = x^2 ∧ 
    3 * w^2 = (3/10) * x^2 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_area_l761_76127


namespace NUMINAMATH_CALUDE_inequality_proofs_l761_76103

theorem inequality_proofs :
  (∀ a b : ℝ, a^2 + b^2 + 3 ≥ a*b + Real.sqrt 3 * (a + b)) ∧
  (Real.sqrt 6 + Real.sqrt 7 > 2 * Real.sqrt 2 + Real.sqrt 5) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proofs_l761_76103


namespace NUMINAMATH_CALUDE_infinite_geometric_series_ratio_l761_76159

/-- For an infinite geometric series with first term a and sum S,
    the common ratio r can be calculated. -/
theorem infinite_geometric_series_ratio 
  (a : ℝ) (S : ℝ) (h1 : a = 512) (h2 : S = 3072) :
  ∃ r : ℝ, r = 5 / 6 ∧ S = a / (1 - r) := by
sorry

end NUMINAMATH_CALUDE_infinite_geometric_series_ratio_l761_76159


namespace NUMINAMATH_CALUDE_sum_base_6_100_equals_666_l761_76198

def base_6_to_10 (n : ℕ) : ℕ := sorry

def sum_base_6 (n : ℕ) : ℕ := sorry

theorem sum_base_6_100_equals_666 :
  sum_base_6 (base_6_to_10 100) = 666 := by sorry

end NUMINAMATH_CALUDE_sum_base_6_100_equals_666_l761_76198


namespace NUMINAMATH_CALUDE_A_intersect_C_R_B_eq_interval_l761_76101

-- Define set A
def A : Set ℝ := {x | x^2 + x - 6 ≤ 0}

-- Define set B
def B : Set ℝ := {y | ∃ x, y = Real.sqrt x ∧ 0 ≤ x ∧ x ≤ 4}

-- Define the complement of B relative to ℝ
def C_R_B : Set ℝ := (Set.univ : Set ℝ) \ B

-- Theorem statement
theorem A_intersect_C_R_B_eq_interval :
  A ∩ C_R_B = Set.Icc (-3 : ℝ) 0 := by sorry

end NUMINAMATH_CALUDE_A_intersect_C_R_B_eq_interval_l761_76101


namespace NUMINAMATH_CALUDE_stratified_sampling_athletes_l761_76155

theorem stratified_sampling_athletes (total_male : ℕ) (total_female : ℕ) 
  (selected_male : ℕ) (selected_female : ℕ) 
  (h1 : total_male = 56) (h2 : total_female = 42) (h3 : selected_male = 8) :
  (selected_male : ℚ) / total_male = selected_female / total_female → selected_female = 6 := by
  sorry

end NUMINAMATH_CALUDE_stratified_sampling_athletes_l761_76155


namespace NUMINAMATH_CALUDE_birds_on_fence_l761_76133

theorem birds_on_fence (initial_birds : ℕ) (additional_birds : ℕ) : 
  initial_birds = 2 → additional_birds = 4 → initial_birds + additional_birds = 6 := by
sorry

end NUMINAMATH_CALUDE_birds_on_fence_l761_76133


namespace NUMINAMATH_CALUDE_sum_of_reciprocals_of_roots_l761_76122

/-- Given a quadratic polynomial 7x^2 + 2x + 6, if α and β are the reciprocals of its roots,
    then their sum is equal to -1/3. -/
theorem sum_of_reciprocals_of_roots (α β : ℝ) : 
  (∃ a b : ℝ, (7 * a^2 + 2 * a + 6 = 0) ∧ 
              (7 * b^2 + 2 * b + 6 = 0) ∧ 
              (α = 1 / a) ∧ 
              (β = 1 / b)) → 
  α + β = -1/3 := by
sorry

end NUMINAMATH_CALUDE_sum_of_reciprocals_of_roots_l761_76122


namespace NUMINAMATH_CALUDE_roberto_cost_per_dozen_approx_l761_76172

/-- Represents the chicken and egg scenario for Roberto --/
structure ChickenScenario where
  num_chickens : ℕ
  chicken_cost : ℚ
  weekly_feed_cost : ℚ
  eggs_per_chicken_per_week : ℕ
  break_even_weeks : ℕ

/-- Calculates the cost per dozen eggs given a ChickenScenario --/
def cost_per_dozen (scenario : ChickenScenario) : ℚ :=
  let total_cost := scenario.num_chickens * scenario.chicken_cost + 
                    scenario.weekly_feed_cost * scenario.break_even_weeks
  let total_eggs := scenario.num_chickens * scenario.eggs_per_chicken_per_week * 
                    scenario.break_even_weeks
  let total_dozens := total_eggs / 12
  total_cost / total_dozens

/-- Roberto's specific scenario --/
def roberto_scenario : ChickenScenario :=
  { num_chickens := 4
  , chicken_cost := 20
  , weekly_feed_cost := 1
  , eggs_per_chicken_per_week := 3
  , break_even_weeks := 81 }

/-- Theorem stating that Roberto's cost per dozen eggs is approximately $1.99 --/
theorem roberto_cost_per_dozen_approx :
  abs (cost_per_dozen roberto_scenario - 1.99) < 0.01 := by
  sorry


end NUMINAMATH_CALUDE_roberto_cost_per_dozen_approx_l761_76172


namespace NUMINAMATH_CALUDE_binomial_product_l761_76112

theorem binomial_product : (Nat.choose 10 3) * (Nat.choose 8 3) = 6720 := by
  sorry

end NUMINAMATH_CALUDE_binomial_product_l761_76112


namespace NUMINAMATH_CALUDE_max_value_of_f_l761_76126

theorem max_value_of_f (x : ℝ) (h : 0 < x ∧ x < 2) : 
  ∃ (max_val : ℝ), max_val = 16/3 ∧ ∀ y ∈ Set.Ioo 0 2, x * (8 - 3 * x) ≤ max_val :=
sorry

end NUMINAMATH_CALUDE_max_value_of_f_l761_76126


namespace NUMINAMATH_CALUDE_expression_evaluation_l761_76139

theorem expression_evaluation : 
  (4 * 6) / (12 * 14) * (8 * 12 * 14) / (4 * 6 * 8) - 1 = 0 := by
sorry

end NUMINAMATH_CALUDE_expression_evaluation_l761_76139


namespace NUMINAMATH_CALUDE_simplify_expression_l761_76105

theorem simplify_expression (x : ℝ) : (3*x - 4)*(x + 6) - (x + 4)*(3*x - 2) = 4*x - 16 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l761_76105


namespace NUMINAMATH_CALUDE_sqrt_sum_equality_l761_76111

theorem sqrt_sum_equality : 
  Real.sqrt (20 - 8 * Real.sqrt 5) + Real.sqrt (20 + 8 * Real.sqrt 5) = 2 * Real.sqrt 14 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_sum_equality_l761_76111


namespace NUMINAMATH_CALUDE_square_perimeter_l761_76141

theorem square_perimeter (area : ℝ) (side : ℝ) (perimeter : ℝ) : 
  area = 324 → 
  area = side * side →
  perimeter = 4 * side →
  perimeter = 72 := by
  sorry

end NUMINAMATH_CALUDE_square_perimeter_l761_76141


namespace NUMINAMATH_CALUDE_intersection_of_M_and_N_l761_76135

def M : Set (ℝ × ℝ) := {a | ∃ x : ℝ, a = (-1, 1) + x • (1, 2)}
def N : Set (ℝ × ℝ) := {a | ∃ x : ℝ, a = (1, -2) + x • (2, 3)}

theorem intersection_of_M_and_N :
  M ∩ N = {(-13, -23)} := by
  sorry

end NUMINAMATH_CALUDE_intersection_of_M_and_N_l761_76135


namespace NUMINAMATH_CALUDE_min_convex_division_rotated_ngon_l761_76169

/-- A regular n-gon. -/
structure RegularNGon (n : ℕ) where
  -- Add necessary fields here

/-- Rotate a regular n-gon by an angle around its center. -/
def rotate (M : RegularNGon n) (angle : ℝ) : RegularNGon n :=
  sorry

/-- The union of two regular n-gons. -/
def union (M M' : RegularNGon n) : Set (ℝ × ℝ) :=
  sorry

/-- A convex polygon. -/
structure ConvexPolygon where
  -- Add necessary fields here

/-- The minimum number of convex polygons needed to divide a set. -/
def minConvexDivision (S : Set (ℝ × ℝ)) : ℕ :=
  sorry

theorem min_convex_division_rotated_ngon (n : ℕ) (M : RegularNGon n) :
  minConvexDivision (union M (rotate M (π / n))) = n + 1 :=
sorry

end NUMINAMATH_CALUDE_min_convex_division_rotated_ngon_l761_76169


namespace NUMINAMATH_CALUDE_second_square_width_l761_76123

/-- Represents the dimensions of a rectangular piece of fabric -/
structure Fabric where
  length : ℝ
  width : ℝ

/-- Calculates the area of a rectangular piece of fabric -/
def area (f : Fabric) : ℝ := f.length * f.width

/-- Represents the three pieces of fabric and the desired flag dimensions -/
structure FlagProblem where
  first : Fabric
  second : Fabric
  third : Fabric
  flagLength : ℝ
  flagHeight : ℝ

theorem second_square_width (p : FlagProblem)
  (h1 : p.first = { length := 8, width := 5 })
  (h2 : p.second.length = 10)
  (h3 : p.third = { length := 5, width := 5 })
  (h4 : p.flagLength = 15)
  (h5 : p.flagHeight = 9) :
  p.second.width = 7 := by
  sorry

end NUMINAMATH_CALUDE_second_square_width_l761_76123


namespace NUMINAMATH_CALUDE_books_read_is_seven_l761_76102

-- Define the number of movies watched
def movies_watched : ℕ := 21

-- Define the relationship between movies watched and books read
def books_read : ℕ := movies_watched - 14

-- Theorem to prove
theorem books_read_is_seven : books_read = 7 := by
  sorry

end NUMINAMATH_CALUDE_books_read_is_seven_l761_76102


namespace NUMINAMATH_CALUDE_triangle_sides_max_sum_squares_l761_76160

theorem triangle_sides_max_sum_squares (a b c : ℝ) (A B C : ℝ) :
  a > 0 → b > 0 → c > 0 →
  A > 0 → B > 0 → C > 0 →
  A + B + C = π →
  (1/2) * c^2 = (1/2) * a * b * Real.sin C →
  a * b = Real.sqrt 2 →
  ∃ (max : ℝ), max = 4 ∧ ∀ (a' b' c' : ℝ),
    a' > 0 → b' > 0 → c' > 0 →
    (1/2) * c'^2 = (1/2) * a' * b' * Real.sin C →
    a' * b' = Real.sqrt 2 →
    a'^2 + b'^2 + c'^2 ≤ max :=
by sorry

end NUMINAMATH_CALUDE_triangle_sides_max_sum_squares_l761_76160


namespace NUMINAMATH_CALUDE_banana_cost_is_three_l761_76195

/-- The cost of a single fruit item -/
structure FruitCost where
  apple : ℕ
  orange : ℕ
  banana : ℕ

/-- The quantity of fruits bought -/
structure FruitQuantity where
  apple : ℕ
  orange : ℕ
  banana : ℕ

/-- Calculate the discount based on the total number of fruits -/
def calculateDiscount (totalFruits : ℕ) : ℕ :=
  totalFruits / 5

/-- Calculate the total cost of fruits before discount -/
def calculateTotalCost (cost : FruitCost) (quantity : FruitQuantity) : ℕ :=
  cost.apple * quantity.apple + cost.orange * quantity.orange + cost.banana * quantity.banana

/-- The main theorem to prove -/
theorem banana_cost_is_three
  (cost : FruitCost)
  (quantity : FruitQuantity)
  (h1 : cost.apple = 1)
  (h2 : cost.orange = 2)
  (h3 : quantity.apple = 5)
  (h4 : quantity.orange = 3)
  (h5 : quantity.banana = 2)
  (h6 : calculateTotalCost cost quantity - calculateDiscount (quantity.apple + quantity.orange + quantity.banana) = 15) :
  cost.banana = 3 := by
  sorry

#check banana_cost_is_three

end NUMINAMATH_CALUDE_banana_cost_is_three_l761_76195


namespace NUMINAMATH_CALUDE_exists_unrepresentable_group_l761_76129

/-- Represents a person in the group -/
structure Person :=
  (id : ℕ)

/-- Represents the acquaintance relationship between two people -/
def Acquainted (p1 p2 : Person) : Prop := sorry

/-- Represents a chord in a circle -/
structure Chord :=
  (person : Person)

/-- Represents the intersection of two chords -/
def Intersects (c1 c2 : Chord) : Prop := sorry

/-- The main theorem stating that there exists a group of people whose acquaintance relationships
    cannot be represented by intersecting chords in a circle -/
theorem exists_unrepresentable_group :
  ∃ (group : Set Person) (acquaintance : Person → Person → Prop),
    ¬∃ (chord_assignment : Person → Chord),
      ∀ (p1 p2 : Person),
        p1 ∈ group → p2 ∈ group → p1 ≠ p2 →
          (acquaintance p1 p2 ↔ Intersects (chord_assignment p1) (chord_assignment p2)) :=
sorry

end NUMINAMATH_CALUDE_exists_unrepresentable_group_l761_76129


namespace NUMINAMATH_CALUDE_shaded_area_between_squares_l761_76186

/-- Given a larger square with area 10 cm² and a smaller square with area 4 cm²,
    where the diagonals of the larger square contain the diagonals of the smaller square,
    prove that the area of one of the four identical regions formed between the squares is 1.5 cm². -/
theorem shaded_area_between_squares (larger_square_area smaller_square_area : ℝ)
  (h1 : larger_square_area = 10)
  (h2 : smaller_square_area = 4)
  (h3 : larger_square_area > smaller_square_area)
  (h4 : ∃ (n : ℕ), n = 4 ∧ n * (larger_square_area - smaller_square_area) / n = 1.5) :
  ∃ (shaded_area : ℝ), shaded_area = 1.5 := by
  sorry

end NUMINAMATH_CALUDE_shaded_area_between_squares_l761_76186


namespace NUMINAMATH_CALUDE_proportion_equality_l761_76147

theorem proportion_equality (x : ℝ) : (0.25 / x = 2 / 6) → x = 0.75 := by
  sorry

end NUMINAMATH_CALUDE_proportion_equality_l761_76147


namespace NUMINAMATH_CALUDE_logarithm_square_sum_l761_76185

theorem logarithm_square_sum (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) 
  (h1 : a * b * c = 10^11) 
  (h2 : Real.log a * Real.log (b * c) + Real.log b * Real.log (c * a) + Real.log c * Real.log (a * b) = 40 * Real.log 10) : 
  Real.sqrt ((Real.log a)^2 + (Real.log b)^2 + (Real.log c)^2) = 9 * Real.log 10 := by
sorry

end NUMINAMATH_CALUDE_logarithm_square_sum_l761_76185


namespace NUMINAMATH_CALUDE_inequality_properties_l761_76194

theorem inequality_properties (a b c : ℝ) :
  (∀ (a b c : ℝ), a * c^2 > b * c^2 → a > b) ∧
  (∀ (a b c : ℝ), a > b → a * (2^c) > b * (2^c)) :=
by sorry

end NUMINAMATH_CALUDE_inequality_properties_l761_76194


namespace NUMINAMATH_CALUDE_f_properties_l761_76140

def f (x : ℝ) := x^2

theorem f_properties : 
  (∀ x, f (-x) = f x) ∧ 
  (∀ x y, 0 < x ∧ x < y → f x < f y) :=
sorry

end NUMINAMATH_CALUDE_f_properties_l761_76140


namespace NUMINAMATH_CALUDE_equilateral_triangle_area_l761_76164

theorem equilateral_triangle_area (perimeter : ℝ) (area : ℝ) : 
  perimeter = 30 → area = 25 * Real.sqrt 3 → 
  ∃ (side : ℝ), side > 0 ∧ 3 * side = perimeter ∧ area = (Real.sqrt 3 / 4) * side^2 :=
by sorry

end NUMINAMATH_CALUDE_equilateral_triangle_area_l761_76164


namespace NUMINAMATH_CALUDE_previous_painting_price_l761_76177

/-- 
Given a painter whose most recent painting sold for $44,000, and this price is $1000 less than 
five times more than his previous painting, prove that the price of the previous painting was $9,000.
-/
theorem previous_painting_price (recent_price previous_price : ℕ) : 
  recent_price = 44000 ∧ 
  recent_price = 5 * previous_price - 1000 →
  previous_price = 9000 := by
sorry

end NUMINAMATH_CALUDE_previous_painting_price_l761_76177


namespace NUMINAMATH_CALUDE_rational_cosine_values_l761_76148

theorem rational_cosine_values : 
  {k : ℚ | 0 ≤ k ∧ k ≤ 1/2 ∧ ∃ (q : ℚ), Real.cos (k * Real.pi) = q} = {0, 1/2, 1/3} := by sorry

end NUMINAMATH_CALUDE_rational_cosine_values_l761_76148


namespace NUMINAMATH_CALUDE_smoking_and_sickness_are_distinct_categorical_variables_l761_76100

-- Define a structure for categorical variables
structure CategoricalVariable where
  name : String
  values : List String

-- Define the "Whether smoking" categorical variable
def whetherSmoking : CategoricalVariable := {
  name := "Whether smoking"
  values := ["smoking", "not smoking"]
}

-- Define the "Whether sick" categorical variable
def whetherSick : CategoricalVariable := {
  name := "Whether sick"
  values := ["sick", "not sick"]
}

-- Theorem to prove that "Whether smoking" and "Whether sick" are two distinct categorical variables
theorem smoking_and_sickness_are_distinct_categorical_variables :
  whetherSmoking ≠ whetherSick :=
sorry

end NUMINAMATH_CALUDE_smoking_and_sickness_are_distinct_categorical_variables_l761_76100


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l761_76188

/-- Arithmetic sequence sum -/
def S (n : ℕ) : ℕ := n^2

/-- Theorem: For an arithmetic sequence with a_1 = 1 and d = 2,
    if S_{k+2} - S_k = 24, then k = 5 -/
theorem arithmetic_sequence_sum (k : ℕ) :
  S (k + 2) - S k = 24 → k = 5 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l761_76188


namespace NUMINAMATH_CALUDE_total_profit_is_23200_l761_76192

/-- Represents the business investment scenario -/
structure BusinessInvestment where
  b_investment : ℝ
  b_period : ℝ
  a_investment : ℝ := 3 * b_investment
  a_period : ℝ := 2 * b_period
  c_investment : ℝ := 2 * b_investment
  c_period : ℝ := 0.5 * b_period
  a_rate : ℝ := 0.10
  b_rate : ℝ := 0.15
  c_rate : ℝ := 0.12
  b_profit : ℝ := 4000

/-- Calculates the total profit for the business investment -/
def total_profit (bi : BusinessInvestment) : ℝ :=
  bi.a_investment * bi.a_period * bi.a_rate +
  bi.b_investment * bi.b_period * bi.b_rate +
  bi.c_investment * bi.c_period * bi.c_rate

/-- Theorem stating that the total profit is 23200 -/
theorem total_profit_is_23200 (bi : BusinessInvestment) :
  total_profit bi = 23200 := by
  sorry

end NUMINAMATH_CALUDE_total_profit_is_23200_l761_76192


namespace NUMINAMATH_CALUDE_imaginary_part_of_z_l761_76120

theorem imaginary_part_of_z (z : ℂ) : z = (Complex.I ^ 2017) / (1 - 2 * Complex.I) → z.im = 1/5 := by
  sorry

end NUMINAMATH_CALUDE_imaginary_part_of_z_l761_76120


namespace NUMINAMATH_CALUDE_fraction_multiplication_and_subtraction_l761_76113

theorem fraction_multiplication_and_subtraction :
  (5 : ℚ) / 6 * ((2 : ℚ) / 3 - (1 : ℚ) / 9) = (25 : ℚ) / 54 := by
  sorry

end NUMINAMATH_CALUDE_fraction_multiplication_and_subtraction_l761_76113


namespace NUMINAMATH_CALUDE_divisibility_proof_l761_76179

theorem divisibility_proof (a b c : ℤ) (n : ℕ) 
  (sum_condition : a + b + c = 1)
  (square_sum_condition : a^2 + b^2 + c^2 = 2*n + 1) :
  ∃ k : ℤ, a^3 + b^2 - a^2 - b^3 = k * n := by
  sorry

end NUMINAMATH_CALUDE_divisibility_proof_l761_76179


namespace NUMINAMATH_CALUDE_inequality_proof_l761_76106

theorem inequality_proof (k n : ℕ) (hk : k > 0) (hn : n > 0) (hkn : k ≤ n) :
  1 + (k : ℝ) / n ≤ (1 + 1 / n) ^ k ∧ (1 + 1 / n) ^ k < 1 + (k : ℝ) / n + (k : ℝ)^2 / n^2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l761_76106


namespace NUMINAMATH_CALUDE_employee_salaries_l761_76119

/-- Proves that the salaries of employees m, n, p, and q sum up to $3000 given the stated conditions --/
theorem employee_salaries (n m p q : ℝ) : 
  (m = 1.4 * n) →
  (p = 0.85 * (m - n)) →
  (q = 1.1 * p) →
  (n + m + p + q = 3000) :=
by
  sorry

end NUMINAMATH_CALUDE_employee_salaries_l761_76119


namespace NUMINAMATH_CALUDE_grandmas_apples_l761_76199

/-- The problem of Grandma's apple purchase --/
theorem grandmas_apples :
  ∀ (tuesday_price : ℝ) (tuesday_kg : ℝ) (saturday_kg : ℝ),
    tuesday_kg > 0 →
    tuesday_price > 0 →
    tuesday_price * tuesday_kg = 20 →
    saturday_kg = 1.5 * tuesday_kg →
    (tuesday_price - 1) * saturday_kg = 24 →
    saturday_kg = 6 := by
  sorry


end NUMINAMATH_CALUDE_grandmas_apples_l761_76199


namespace NUMINAMATH_CALUDE_point_not_on_line_l761_76116

/-- Point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Check if three points are collinear -/
def are_collinear (A B C : Point3D) : Prop :=
  ∃ k : ℝ, (C.x - A.x, C.y - A.y, C.z - A.z) = k • (B.x - A.x, B.y - A.y, B.z - A.z)

theorem point_not_on_line : 
  let A : Point3D := ⟨-1, 1, 2⟩
  let B : Point3D := ⟨3, 6, -1⟩
  let C : Point3D := ⟨7, 9, 0⟩
  ¬(are_collinear A B C) := by
  sorry

end NUMINAMATH_CALUDE_point_not_on_line_l761_76116


namespace NUMINAMATH_CALUDE_eight_digit_number_theorem_l761_76121

def is_coprime (a b : ℕ) : Prop := Nat.gcd a b = 1

def move_last_to_first (n : ℕ) : ℕ :=
  let last_digit := n % 10
  let rest := n / 10
  last_digit * 10^7 + rest

theorem eight_digit_number_theorem (B : ℕ) (hB1 : is_coprime B 36) (hB2 : B > 7777777) :
  let A := move_last_to_first B
  (∃ A_min A_max : ℕ, 
    (∀ A' : ℕ, (∃ B' : ℕ, A' = move_last_to_first B' ∧ is_coprime B' 36 ∧ B' > 7777777) → 
      A_min ≤ A' ∧ A' ≤ A_max) ∧
    A_min = 17777779 ∧ 
    A_max = 99999998) :=
sorry

end NUMINAMATH_CALUDE_eight_digit_number_theorem_l761_76121


namespace NUMINAMATH_CALUDE_only_integer_solution_is_two_l761_76156

theorem only_integer_solution_is_two :
  ∀ x : ℤ, (0 < (x - 1)^2 / (x + 1) ∧ (x - 1)^2 / (x + 1) < 1) ↔ x = 2 :=
by sorry

end NUMINAMATH_CALUDE_only_integer_solution_is_two_l761_76156


namespace NUMINAMATH_CALUDE_minimum_shots_for_high_probability_l761_76182

theorem minimum_shots_for_high_probability (p : ℝ) (n : ℕ) : 
  p = 1/2 → 
  (1 - (1 - p)^n > 0.9 ↔ n ≥ 4) :=
by sorry

end NUMINAMATH_CALUDE_minimum_shots_for_high_probability_l761_76182


namespace NUMINAMATH_CALUDE_absolute_value_equation_solution_l761_76125

theorem absolute_value_equation_solution :
  ∀ x : ℝ, (|2 * x + 6| = 3 * x + 9) ↔ (x = -3) :=
by sorry

end NUMINAMATH_CALUDE_absolute_value_equation_solution_l761_76125


namespace NUMINAMATH_CALUDE_undergrad_play_count_l761_76176

-- Define the total number of students
def total_students : ℕ := 800

-- Define the percentage of undergraduates who play a musical instrument
def undergrad_play_percent : ℚ := 25 / 100

-- Define the percentage of postgraduates who do not play a musical instrument
def postgrad_not_play_percent : ℚ := 20 / 100

-- Define the percentage of all students who do not play a musical instrument
def total_not_play_percent : ℚ := 355 / 1000

-- Theorem stating that the number of undergraduates who play a musical instrument is 57
theorem undergrad_play_count : ℕ := by
  sorry

end NUMINAMATH_CALUDE_undergrad_play_count_l761_76176


namespace NUMINAMATH_CALUDE_problem_solution_l761_76166

theorem problem_solution (x y z : ℝ) 
  (h1 : 5 = 0.25 * x)
  (h2 : 5 = 0.10 * y)
  (h3 : z = 2 * y) :
  x - z = -80 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l761_76166


namespace NUMINAMATH_CALUDE_combined_average_marks_l761_76115

theorem combined_average_marks (class1_students class2_students class3_students : ℕ)
  (class1_avg class2_avg class3_avg : ℚ)
  (h1 : class1_students = 35)
  (h2 : class2_students = 45)
  (h3 : class3_students = 25)
  (h4 : class1_avg = 40)
  (h5 : class2_avg = 60)
  (h6 : class3_avg = 75) :
  (class1_students * class1_avg + class2_students * class2_avg + class3_students * class3_avg) /
  (class1_students + class2_students + class3_students) = 5975 / 105 :=
by sorry

end NUMINAMATH_CALUDE_combined_average_marks_l761_76115


namespace NUMINAMATH_CALUDE_max_length_complex_l761_76197

theorem max_length_complex (ω : ℂ) (h : Complex.abs ω = 1) :
  ∃ (max : ℝ), max = 108 ∧ ∀ (z : ℂ), Complex.abs ((ω + 2)^3 * (ω - 3)^2) ≤ max :=
sorry

end NUMINAMATH_CALUDE_max_length_complex_l761_76197


namespace NUMINAMATH_CALUDE_simplify_sum_of_radicals_l761_76163

theorem simplify_sum_of_radicals : 
  Real.sqrt (12 + 8 * Real.sqrt 3) + Real.sqrt (12 - 8 * Real.sqrt 3) = 4 * Real.sqrt 2 := by
sorry

end NUMINAMATH_CALUDE_simplify_sum_of_radicals_l761_76163


namespace NUMINAMATH_CALUDE_proposition_q_must_be_true_l761_76149

theorem proposition_q_must_be_true (p q : Prop) 
  (h1 : ¬p) (h2 : p ∨ q) : q := by
  sorry

end NUMINAMATH_CALUDE_proposition_q_must_be_true_l761_76149


namespace NUMINAMATH_CALUDE_distance_minus_one_to_2023_l761_76137

/-- The distance between two points on a number line -/
def distance (a b : ℝ) : ℝ := |b - a|

/-- Theorem: The distance between points representing -1 and 2023 on a number line is 2024 -/
theorem distance_minus_one_to_2023 : distance (-1) 2023 = 2024 := by
  sorry

end NUMINAMATH_CALUDE_distance_minus_one_to_2023_l761_76137


namespace NUMINAMATH_CALUDE_dot_product_theorem_l761_76145

def a : ℝ × ℝ := (1, -1)
def b : ℝ × ℝ := (-1, 2)

theorem dot_product_theorem : (2 • a + b) • a = 1 := by sorry

end NUMINAMATH_CALUDE_dot_product_theorem_l761_76145


namespace NUMINAMATH_CALUDE_camerons_list_count_camerons_list_count_is_871_l761_76165

theorem camerons_list_count : ℕ → Prop :=
  fun count =>
    let smallest_square := 900
    let smallest_cube := 27000
    (∀ k : ℕ, k < smallest_square → ¬∃ m : ℕ, k = 30 * m * m) ∧
    (∀ k : ℕ, k < smallest_cube → ¬∃ m : ℕ, k = 30 * m * m * m) ∧
    count = (smallest_cube / 30 - smallest_square / 30 + 1)

theorem camerons_list_count_is_871 : camerons_list_count 871 := by
  sorry

end NUMINAMATH_CALUDE_camerons_list_count_camerons_list_count_is_871_l761_76165


namespace NUMINAMATH_CALUDE_negation_of_proposition_l761_76138

theorem negation_of_proposition :
  (¬ ∀ x : ℝ, |x - 2| + |x - 4| > 3) ↔ (∃ x : ℝ, |x - 2| + |x - 4| ≤ 3) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_proposition_l761_76138
