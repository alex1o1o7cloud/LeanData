import Mathlib

namespace NUMINAMATH_CALUDE_necklace_diamonds_l2800_280099

theorem necklace_diamonds (total_necklaces : ℕ) (diamonds_type1 diamonds_type2 : ℕ) (total_diamonds : ℕ) :
  total_necklaces = 20 →
  diamonds_type1 = 2 →
  diamonds_type2 = 5 →
  total_diamonds = 79 →
  ∃ (x y : ℕ), x + y = total_necklaces ∧ 
                diamonds_type1 * x + diamonds_type2 * y = total_diamonds ∧
                y = 13 :=
by sorry

end NUMINAMATH_CALUDE_necklace_diamonds_l2800_280099


namespace NUMINAMATH_CALUDE_unique_divisible_by_792_l2800_280038

/-- Represents a 7-digit number in the form 13xy45z -/
def number (x y z : Nat) : Nat :=
  1300000 + x * 10000 + y * 1000 + 450 + z

/-- Checks if a number is of the form 13xy45z where x, y, z are single digits -/
def isValidForm (n : Nat) : Prop :=
  ∃ x y z, x < 10 ∧ y < 10 ∧ z < 10 ∧ n = number x y z

theorem unique_divisible_by_792 :
  ∃! n, isValidForm n ∧ n % 792 = 0 ∧ n = 1380456 :=
sorry

end NUMINAMATH_CALUDE_unique_divisible_by_792_l2800_280038


namespace NUMINAMATH_CALUDE_waiter_theorem_l2800_280091

def waiter_problem (total_customers : ℕ) (left_customers : ℕ) (people_per_table : ℕ) : Prop :=
  let remaining_customers := total_customers - left_customers
  remaining_customers / people_per_table = 3

theorem waiter_theorem : waiter_problem 21 12 3 := by
  sorry

end NUMINAMATH_CALUDE_waiter_theorem_l2800_280091


namespace NUMINAMATH_CALUDE_sum_of_odd_integers_11_to_51_l2800_280045

theorem sum_of_odd_integers_11_to_51 (a₁ : ℕ) (aₙ : ℕ) (d : ℕ) (n : ℕ) :
  a₁ = 11 →
  aₙ = 51 →
  d = 2 →
  aₙ = a₁ + (n - 1) * d →
  (n : ℚ) / 2 * (a₁ + aₙ) = 651 :=
by sorry

end NUMINAMATH_CALUDE_sum_of_odd_integers_11_to_51_l2800_280045


namespace NUMINAMATH_CALUDE_complex_equation_solution_l2800_280020

theorem complex_equation_solution (a b : ℝ) :
  (Complex.I : ℂ) * 2 = (1 : ℂ) + (Complex.I : ℂ) * 2 * a + b → a = 1 ∧ b = -1 := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l2800_280020


namespace NUMINAMATH_CALUDE_problem_statement_l2800_280078

theorem problem_statement (a b x y : ℝ) 
  (h1 : a*x + b*y = 2)
  (h2 : a*x^2 + b*y^2 = 5)
  (h3 : a*x^3 + b*y^3 = 10)
  (h4 : a*x^4 + b*y^4 = 30) :
  a*x^5 + b*y^5 = 40 := by sorry

end NUMINAMATH_CALUDE_problem_statement_l2800_280078


namespace NUMINAMATH_CALUDE_class_size_l2800_280049

theorem class_size (total : ℝ) 
  (h1 : 0.25 * total = total - (0.75 * total))
  (h2 : 0.1875 * total = 0.25 * (0.75 * total))
  (h3 : 18 = 0.75 * total - 0.1875 * total) : 
  total = 32 := by
  sorry

end NUMINAMATH_CALUDE_class_size_l2800_280049


namespace NUMINAMATH_CALUDE_min_value_expression_min_value_attainable_l2800_280048

theorem min_value_expression (a b c : ℝ) (h1 : b > c) (h2 : c > a) (h3 : b ≠ 0) :
  (5 * (a + b)^2 + 4 * (b - c)^2 + 3 * (c - a)^2) / (2 * b^2) ≥ 24 :=
by sorry

theorem min_value_attainable (ε : ℝ) (hε : ε > 0) :
  ∃ (a b c : ℝ), b > c ∧ c > a ∧ b ≠ 0 ∧
  (5 * (a + b)^2 + 4 * (b - c)^2 + 3 * (c - a)^2) / (2 * b^2) < 24 + ε :=
by sorry

end NUMINAMATH_CALUDE_min_value_expression_min_value_attainable_l2800_280048


namespace NUMINAMATH_CALUDE_magazine_circulation_ratio_l2800_280077

/-- The circulation ratio problem for magazine P -/
theorem magazine_circulation_ratio 
  (avg_circulation : ℝ) -- Average yearly circulation for 1962-1970
  (h : avg_circulation > 0) -- Assumption that circulation is positive
  : (4 * avg_circulation) / (4 * avg_circulation + 9 * avg_circulation) = 4 / 13 := by
  sorry

end NUMINAMATH_CALUDE_magazine_circulation_ratio_l2800_280077


namespace NUMINAMATH_CALUDE_march_greatest_percent_difference_l2800_280043

/-- Represents the sales data for a group in a given month -/
structure SalesData where
  drummers : ℕ
  buglePlayers : ℕ

/-- Represents the fixed costs for each group -/
structure FixedCosts where
  drummers : ℕ
  buglePlayers : ℕ

/-- Calculates the net earnings for a group given sales and fixed cost -/
def netEarnings (sales : ℕ) (cost : ℕ) : ℤ :=
  (sales : ℤ) - (sales * cost : ℤ)

/-- Calculates the percent difference between two integer values -/
def percentDifference (a b : ℤ) : ℚ :=
  if b ≠ 0 then (a - b : ℚ) / (b.natAbs : ℚ) * 100
  else if a > 0 then 100
  else if a < 0 then -100
  else 0

/-- Theorem stating that March has the greatest percent difference in net earnings -/
theorem march_greatest_percent_difference 
  (sales : Fin 5 → SalesData) 
  (costs : FixedCosts) 
  (h_jan : sales 0 = ⟨150, 100⟩)
  (h_feb : sales 1 = ⟨200, 150⟩)
  (h_mar : sales 2 = ⟨180, 180⟩)
  (h_apr : sales 3 = ⟨120, 160⟩)
  (h_may : sales 4 = ⟨80, 120⟩)
  (h_costs : costs = ⟨1, 2⟩) :
  ∀ (i : Fin 5), i ≠ 2 → 
    (abs (percentDifference 
      (netEarnings (sales 2).drummers costs.drummers)
      (netEarnings (sales 2).buglePlayers costs.buglePlayers)) ≥
     abs (percentDifference
      (netEarnings (sales i).drummers costs.drummers)
      (netEarnings (sales i).buglePlayers costs.buglePlayers))) :=
by sorry

end NUMINAMATH_CALUDE_march_greatest_percent_difference_l2800_280043


namespace NUMINAMATH_CALUDE_least_m_is_207_l2800_280093

def x : ℕ → ℚ
  | 0 => 7
  | n + 1 => (x n ^ 2 + 6 * x n + 8) / (x n + 7)

def is_least_m (m : ℕ) : Prop :=
  x m ≤ 5 + 1 / 2^15 ∧ ∀ k < m, x k > 5 + 1 / 2^15

theorem least_m_is_207 : is_least_m 207 := by
  sorry

end NUMINAMATH_CALUDE_least_m_is_207_l2800_280093


namespace NUMINAMATH_CALUDE_function_identity_l2800_280058

-- Define the property that the function f must satisfy
def SatisfiesProperty (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, f (x^2 + f y) = y + (f x)^2

-- State the theorem
theorem function_identity {f : ℝ → ℝ} (h : SatisfiesProperty f) : 
  ∀ x : ℝ, f x = x := by sorry

end NUMINAMATH_CALUDE_function_identity_l2800_280058


namespace NUMINAMATH_CALUDE_pharmacy_tubs_l2800_280053

theorem pharmacy_tubs (total_needed : ℕ) (in_storage : ℕ) : 
  total_needed = 100 →
  in_storage = 20 →
  let to_buy := total_needed - in_storage
  let from_new_vendor := to_buy / 4
  let from_usual_vendor := to_buy - from_new_vendor
  from_usual_vendor = 60 := by
  sorry

end NUMINAMATH_CALUDE_pharmacy_tubs_l2800_280053


namespace NUMINAMATH_CALUDE_complex_magnitude_squared_l2800_280008

theorem complex_magnitude_squared (z : ℂ) (h : z + Complex.abs z = -3 + 4*I) : Complex.abs z ^ 2 = 625 / 36 := by
  sorry

end NUMINAMATH_CALUDE_complex_magnitude_squared_l2800_280008


namespace NUMINAMATH_CALUDE_abs_h_value_l2800_280029

theorem abs_h_value (h : ℝ) : 
  (∃ x₁ x₂ x₃ x₄ : ℝ, 
    (x₁^4 + 4*h*x₁^2 = 2) ∧ 
    (x₂^4 + 4*h*x₂^2 = 2) ∧ 
    (x₃^4 + 4*h*x₃^2 = 2) ∧ 
    (x₄^4 + 4*h*x₄^2 = 2) ∧ 
    (x₁^2 + x₂^2 + x₃^2 + x₄^2 = 34)) → 
  |h| = 17/4 := by
sorry

end NUMINAMATH_CALUDE_abs_h_value_l2800_280029


namespace NUMINAMATH_CALUDE_fraction_always_defined_l2800_280004

theorem fraction_always_defined (x : ℝ) : (x^2 + 1) ≠ 0 := by
  sorry

#check fraction_always_defined

end NUMINAMATH_CALUDE_fraction_always_defined_l2800_280004


namespace NUMINAMATH_CALUDE_perimeter_C_is_24_l2800_280039

/-- Represents a polygon in the triangular grid -/
structure Polygon where
  perimeter : ℝ

/-- Represents the triangular grid with four polygons -/
structure TriangularGrid where
  A : Polygon
  B : Polygon
  C : Polygon
  D : Polygon

/-- The perimeter of triangle C in the given triangular grid -/
def perimeter_C (grid : TriangularGrid) : ℝ :=
  -- Definition to be proved
  24

/-- Theorem stating that the perimeter of triangle C is 24 cm -/
theorem perimeter_C_is_24 (grid : TriangularGrid) 
    (h1 : grid.A.perimeter = 56)
    (h2 : grid.B.perimeter = 34)
    (h3 : grid.D.perimeter = 42) :
  perimeter_C grid = 24 := by
  sorry

end NUMINAMATH_CALUDE_perimeter_C_is_24_l2800_280039


namespace NUMINAMATH_CALUDE_negation_of_universal_proposition_l2800_280063

theorem negation_of_universal_proposition :
  (¬ ∀ x : ℝ, |x - 1| - |x + 1| ≤ 3) ↔ (∃ x₀ : ℝ, |x₀ - 1| - |x₀ + 1| > 3) := by
  sorry

end NUMINAMATH_CALUDE_negation_of_universal_proposition_l2800_280063


namespace NUMINAMATH_CALUDE_vector_calculation_l2800_280007

/-- Given vectors a, b, and c in ℝ³, prove that a + 2b - 3c equals (-7, -1, -1) -/
theorem vector_calculation (a b c : ℝ × ℝ × ℝ) 
  (ha : a = (2, 0, 1)) 
  (hb : b = (-3, 1, -1)) 
  (hc : c = (1, 1, 0)) : 
  a + 2 • b - 3 • c = (-7, -1, -1) := by
  sorry

end NUMINAMATH_CALUDE_vector_calculation_l2800_280007


namespace NUMINAMATH_CALUDE_lindas_coins_l2800_280054

/-- Represents the number of coins Linda has initially and receives from her mother --/
structure CoinCounts where
  initial_dimes : Nat
  initial_quarters : Nat
  initial_nickels : Nat
  mother_dimes : Nat
  mother_quarters : Nat
  mother_nickels : Nat

/-- The theorem statement --/
theorem lindas_coins (c : CoinCounts) 
  (h1 : c.initial_dimes = 2)
  (h2 : c.initial_quarters = 6)
  (h3 : c.initial_nickels = 5)
  (h4 : c.mother_dimes = 2)
  (h5 : c.mother_nickels = 2 * c.initial_nickels)
  (h6 : c.initial_dimes + c.initial_quarters + c.initial_nickels + 
        c.mother_dimes + c.mother_quarters + c.mother_nickels = 35) :
  c.mother_quarters = 10 := by
  sorry

end NUMINAMATH_CALUDE_lindas_coins_l2800_280054


namespace NUMINAMATH_CALUDE_green_ball_probability_l2800_280076

/-- Represents a container with red and green balls -/
structure Container where
  red : ℕ
  green : ℕ

/-- Calculates the probability of selecting a green ball from a container -/
def greenProbability (c : Container) : ℚ :=
  c.green / (c.red + c.green)

/-- Calculates the total probability of selecting a green ball from three containers -/
def totalGreenProbability (c1 c2 c3 : Container) : ℚ :=
  (1 / 3) * (greenProbability c1 + greenProbability c2 + greenProbability c3)

/-- Theorem stating the probability of selecting a green ball -/
theorem green_ball_probability :
  let c1 : Container := ⟨8, 4⟩
  let c2 : Container := ⟨2, 4⟩
  let c3 : Container := ⟨2, 6⟩
  totalGreenProbability c1 c2 c3 = 7 / 12 := by
  sorry


end NUMINAMATH_CALUDE_green_ball_probability_l2800_280076


namespace NUMINAMATH_CALUDE_polygon_sides_count_l2800_280006

/-- Given a polygon where the sum of its interior angles is 180° less than three times
    the sum of its exterior angles, prove that it has 5 sides. -/
theorem polygon_sides_count (n : ℕ) : n > 2 →
  (n - 2) * 180 = 3 * 360 - 180 →
  n = 5 := by
  sorry

end NUMINAMATH_CALUDE_polygon_sides_count_l2800_280006


namespace NUMINAMATH_CALUDE_ratio_of_percentages_l2800_280031

theorem ratio_of_percentages (P Q M N : ℝ) 
  (hM : M = 0.4 * Q) 
  (hQ : Q = 0.25 * P) 
  (hN : N = 0.6 * P) 
  (hP : P ≠ 0) : 
  M / N = 1 / 6 := by
  sorry

end NUMINAMATH_CALUDE_ratio_of_percentages_l2800_280031


namespace NUMINAMATH_CALUDE_best_marksman_score_prove_best_marksman_score_l2800_280046

/-- Calculates the best marksman's score in a shooting competition. -/
theorem best_marksman_score (team_size : ℕ) (hypothetical_best_score : ℕ) (hypothetical_average : ℕ) (actual_total_score : ℕ) : ℕ :=
  let hypothetical_total := (team_size - 1) * hypothetical_average + hypothetical_best_score
  hypothetical_best_score - (hypothetical_total - actual_total_score)

/-- Proves that the best marksman's score is 77 given the problem conditions. -/
theorem prove_best_marksman_score :
  best_marksman_score 8 92 84 665 = 77 := by
  sorry

end NUMINAMATH_CALUDE_best_marksman_score_prove_best_marksman_score_l2800_280046


namespace NUMINAMATH_CALUDE_unique_a_value_l2800_280089

theorem unique_a_value (a : ℝ) : 3 ∈ ({1, a, a - 2} : Set ℝ) → a = 5 := by
  sorry

end NUMINAMATH_CALUDE_unique_a_value_l2800_280089


namespace NUMINAMATH_CALUDE_smallest_base_is_five_l2800_280028

/-- Representation of a number in base b -/
def BaseRepresentation (digits : List Nat) (b : Nat) : Nat :=
  digits.foldl (fun acc d => acc * b + d) 0

/-- Condition: In base b, 12_b squared equals 144_b -/
def SquareCondition (b : Nat) : Prop :=
  (BaseRepresentation [1, 2] b) ^ 2 = BaseRepresentation [1, 4, 4] b

/-- The smallest base b greater than 4 for which 12_b squared equals 144_b is 5 -/
theorem smallest_base_is_five :
  ∃ (b : Nat), b > 4 ∧ SquareCondition b ∧ ∀ (k : Nat), k > 4 ∧ k < b → ¬SquareCondition k :=
by sorry

end NUMINAMATH_CALUDE_smallest_base_is_five_l2800_280028


namespace NUMINAMATH_CALUDE_journey_proof_l2800_280081

/-- Represents the distance-time relationship for a journey -/
def distance_from_destination (total_distance : ℝ) (speed : ℝ) (time : ℝ) : ℝ :=
  total_distance - speed * time

theorem journey_proof (total_distance : ℝ) (speed : ℝ) (time : ℝ) 
  (h1 : total_distance = 174)
  (h2 : speed = 60)
  (h3 : time = 1.5) :
  distance_from_destination total_distance speed time = 84 := by
  sorry

#check journey_proof

end NUMINAMATH_CALUDE_journey_proof_l2800_280081


namespace NUMINAMATH_CALUDE_triangle_side_length_validity_l2800_280080

theorem triangle_side_length_validity 
  (a b c : ℝ) 
  (ha : a = 5) 
  (hb : b = 8) 
  (hc : c = 6) : 
  a + b > c ∧ a + c > b ∧ b + c > a :=
sorry

end NUMINAMATH_CALUDE_triangle_side_length_validity_l2800_280080


namespace NUMINAMATH_CALUDE_divide_number_80_l2800_280090

theorem divide_number_80 (smaller larger : ℝ) : 
  smaller + larger = 80 ∧ 
  larger / 2 = smaller + 10 → 
  smaller = 20 ∧ larger = 60 := by
sorry

end NUMINAMATH_CALUDE_divide_number_80_l2800_280090


namespace NUMINAMATH_CALUDE_nonreal_roots_product_l2800_280003

theorem nonreal_roots_product (x : ℂ) : 
  (x^4 - 6*x^3 + 15*x^2 - 20*x = 984) →
  (∃ a b : ℂ, (a ≠ b) ∧ (a.im ≠ 0) ∧ (b.im ≠ 0) ∧
   (x^4 - 6*x^3 + 15*x^2 - 20*x = 984 → (x = a ∨ x = b ∨ x.im = 0)) ∧
   (a * b = 4 - Real.sqrt 1000)) :=
sorry

end NUMINAMATH_CALUDE_nonreal_roots_product_l2800_280003


namespace NUMINAMATH_CALUDE_inscribed_triangle_with_parallel_sides_l2800_280032

/-- A line in the plane -/
structure Line where
  -- Add necessary fields for a line

/-- A circle in the plane -/
structure Circle where
  -- Add necessary fields for a circle

/-- A point in the plane -/
structure Point where
  -- Add necessary fields for a point

/-- A triangle in the plane -/
structure Triangle where
  A : Point
  B : Point
  C : Point

/-- Check if two lines are parallel -/
def are_parallel (l1 l2 : Line) : Prop :=
  sorry

/-- Check if a point lies on a circle -/
def point_on_circle (p : Point) (c : Circle) : Prop :=
  sorry

/-- Check if a line is parallel to a side of a triangle -/
def line_parallel_to_side (l : Line) (t : Triangle) : Prop :=
  sorry

/-- Main theorem: Given three pairwise non-parallel lines and a circle,
    there exists a triangle inscribed in the circle with sides parallel to the given lines -/
theorem inscribed_triangle_with_parallel_sides
  (l1 l2 l3 : Line) (c : Circle)
  (h1 : ¬ are_parallel l1 l2)
  (h2 : ¬ are_parallel l2 l3)
  (h3 : ¬ are_parallel l3 l1) :
  ∃ (t : Triangle),
    point_on_circle t.A c ∧
    point_on_circle t.B c ∧
    point_on_circle t.C c ∧
    line_parallel_to_side l1 t ∧
    line_parallel_to_side l2 t ∧
    line_parallel_to_side l3 t :=
  sorry

end NUMINAMATH_CALUDE_inscribed_triangle_with_parallel_sides_l2800_280032


namespace NUMINAMATH_CALUDE_complex_equation_system_l2800_280019

theorem complex_equation_system (p q r s t u : ℂ) 
  (hp : p ≠ 0) (hq : q ≠ 0) (hr : r ≠ 0) (hs : s ≠ 0) (ht : t ≠ 0) (hu : u ≠ 0)
  (eq1 : p = (q + r) / (s - 3))
  (eq2 : q = (p + r) / (t - 3))
  (eq3 : r = (p + q) / (u - 3))
  (eq4 : s * t + s * u + t * u = 10)
  (eq5 : s + t + u = 6) :
  s * t * u = 11 := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_system_l2800_280019


namespace NUMINAMATH_CALUDE_polygon_with_1800_degree_sum_is_dodecagon_l2800_280027

theorem polygon_with_1800_degree_sum_is_dodecagon :
  ∀ n : ℕ, 
  n ≥ 3 →
  (n - 2) * 180 = 1800 →
  n = 12 :=
by
  sorry

end NUMINAMATH_CALUDE_polygon_with_1800_degree_sum_is_dodecagon_l2800_280027


namespace NUMINAMATH_CALUDE_xyz_mod_8_l2800_280074

theorem xyz_mod_8 (x y z : ℕ) : 
  x < 8 → y < 8 → z < 8 → x > 0 → y > 0 → z > 0 →
  (x * y * z) % 8 = 1 → 
  (3 * z) % 8 = 5 → 
  (7 * y) % 8 = (4 + y) % 8 → 
  (x + y + z) % 8 = 1 := by
  sorry

end NUMINAMATH_CALUDE_xyz_mod_8_l2800_280074


namespace NUMINAMATH_CALUDE_triangle_angle_values_l2800_280022

theorem triangle_angle_values (a b c A B C : ℝ) :
  -- Triangle ABC exists
  0 < A ∧ 0 < B ∧ 0 < C ∧ A + B + C = π →
  -- Sides a, b, c are opposite to angles A, B, C respectively
  a > 0 ∧ b > 0 ∧ c > 0 →
  -- Given conditions
  b * Real.sin A = Real.sqrt 3 * a * Real.cos B →
  Real.cos A * Real.sin C = (Real.sqrt 3 - 1) / 4 →
  -- Conclusions
  B = π / 3 ∧ A = 5 * π / 12 := by
  sorry

end NUMINAMATH_CALUDE_triangle_angle_values_l2800_280022


namespace NUMINAMATH_CALUDE_tangent_line_equation_l2800_280017

-- Define the function f(x)
def f (x : ℝ) : ℝ := x^3 + x - 8

-- Define the point of tangency
def point : ℝ × ℝ := (1, -6)

-- State the theorem
theorem tangent_line_equation :
  ∃ (m b : ℝ), 
    (∀ x y : ℝ, y = m * x + b ↔ m * x - y + b = 0) ∧
    (HasDerivAt f (m * x - y + b) point.1) ∧
    (f point.1 = point.2) ∧
    (m = 4 ∧ b = -10) := by
  sorry

end NUMINAMATH_CALUDE_tangent_line_equation_l2800_280017


namespace NUMINAMATH_CALUDE_probability_x2_y2_leq_1_probability_equals_pi_over_16_l2800_280015

/-- The probability that x^2 + y^2 ≤ 1 when x and y are randomly chosen from [0,2] -/
theorem probability_x2_y2_leq_1 : ℝ :=
  let total_area : ℝ := 4 -- Area of the square [0,2] × [0,2]
  let circle_area : ℝ := Real.pi / 4 -- Area of the quarter circle x^2 + y^2 ≤ 1 in the first quadrant
  circle_area / total_area

/-- The main theorem stating that the probability is equal to π/16 -/
theorem probability_equals_pi_over_16 : probability_x2_y2_leq_1 = Real.pi / 16 := by
  sorry


end NUMINAMATH_CALUDE_probability_x2_y2_leq_1_probability_equals_pi_over_16_l2800_280015


namespace NUMINAMATH_CALUDE_max_diff_even_digit_numbers_l2800_280057

/-- A function that checks if a natural number has all even digits -/
def all_even_digits (n : ℕ) : Prop :=
  ∀ d, d ∈ n.digits 10 → d % 2 = 0

/-- A function that checks if a natural number has at least one odd digit -/
def has_odd_digit (n : ℕ) : Prop :=
  ∃ d, d ∈ n.digits 10 ∧ d % 2 = 1

/-- The theorem stating the maximum difference between two 6-digit numbers with all even digits -/
theorem max_diff_even_digit_numbers :
  ∃ (a b : ℕ),
    100000 ≤ a ∧ a < b ∧ b < 1000000 ∧
    all_even_digits a ∧
    all_even_digits b ∧
    (∀ k, a < k ∧ k < b → has_odd_digit k) ∧
    b - a = 111112 ∧
    (∀ a' b', 100000 ≤ a' ∧ a' < b' ∧ b' < 1000000 ∧
              all_even_digits a' ∧
              all_even_digits b' ∧
              (∀ k, a' < k ∧ k < b' → has_odd_digit k) →
              b' - a' ≤ 111112) :=
by sorry

end NUMINAMATH_CALUDE_max_diff_even_digit_numbers_l2800_280057


namespace NUMINAMATH_CALUDE_sum_of_zeros_is_eight_l2800_280012

/-- A function that is symmetric about x = 2 and has exactly four distinct zeros -/
def SymmetricFunction (f : ℝ → ℝ) : Prop :=
  (∀ x : ℝ, f (2 + x) = f (2 - x)) ∧
  (∃! (z₁ z₂ z₃ z₄ : ℝ), z₁ ≠ z₂ ∧ z₁ ≠ z₃ ∧ z₁ ≠ z₄ ∧ z₂ ≠ z₃ ∧ z₂ ≠ z₄ ∧ z₃ ≠ z₄ ∧
    f z₁ = 0 ∧ f z₂ = 0 ∧ f z₃ = 0 ∧ f z₄ = 0)

/-- The theorem stating that the sum of zeros for a symmetric function with four distinct zeros is 8 -/
theorem sum_of_zeros_is_eight (f : ℝ → ℝ) (h : SymmetricFunction f) :
  ∃ z₁ z₂ z₃ z₄ : ℝ, z₁ ≠ z₂ ∧ z₁ ≠ z₃ ∧ z₁ ≠ z₄ ∧ z₂ ≠ z₃ ∧ z₂ ≠ z₄ ∧ z₃ ≠ z₄ ∧
    f z₁ = 0 ∧ f z₂ = 0 ∧ f z₃ = 0 ∧ f z₄ = 0 ∧
    z₁ + z₂ + z₃ + z₄ = 8 :=
by sorry

end NUMINAMATH_CALUDE_sum_of_zeros_is_eight_l2800_280012


namespace NUMINAMATH_CALUDE_exists_acute_triangle_l2800_280018

/-- A set of 5 positive real numbers representing segment lengths -/
def SegmentSet : Type := Fin 5 → ℝ

/-- Predicate to check if three numbers can form a triangle -/
def can_form_triangle (a b c : ℝ) : Prop :=
  a > 0 ∧ b > 0 ∧ c > 0 ∧ a + b > c ∧ b + c > a ∧ c + a > b

/-- Predicate to check if a triangle is acute-angled -/
def is_acute_triangle (a b c : ℝ) : Prop :=
  a^2 + b^2 > c^2 ∧ b^2 + c^2 > a^2 ∧ c^2 + a^2 > b^2

/-- Main theorem: Given 5 segments where any three can form a triangle,
    there exists at least one acute-angled triangle -/
theorem exists_acute_triangle (s : SegmentSet) 
  (h_positive : ∀ i, s i > 0)
  (h_triangle : ∀ i j k, i ≠ j → j ≠ k → k ≠ i → can_form_triangle (s i) (s j) (s k)) :
  ∃ i j k, i ≠ j ∧ j ≠ k ∧ k ≠ i ∧ is_acute_triangle (s i) (s j) (s k) := by
  sorry


end NUMINAMATH_CALUDE_exists_acute_triangle_l2800_280018


namespace NUMINAMATH_CALUDE_quadratic_function_max_value_l2800_280026

-- Define the quadratic function f
def f (a b c : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x + c

-- Define the conditions
def in_band (y k l : ℝ) : Prop := k ≤ y ∧ y ≤ l

theorem quadratic_function_max_value
  (a b c : ℝ)  -- Coefficients of the quadratic function
  (h1 : in_band (f a b c (-2) + 2) 0 4)
  (h2 : in_band (f a b c 0 + 2) 0 4)
  (h3 : in_band (f a b c 2 + 2) 0 4)
  (h4 : ∀ t : ℝ, in_band (t + 1) (-1) 3 → |f a b c t| ≤ 5/2)
  : (∃ t : ℝ, in_band (t + 1) (-1) 3 ∧ |f a b c t| = 5/2) :=
sorry

end NUMINAMATH_CALUDE_quadratic_function_max_value_l2800_280026


namespace NUMINAMATH_CALUDE_binomial_expansion_coefficient_l2800_280011

theorem binomial_expansion_coefficient (x : ℝ) : 
  ∃ (c : ℕ), c = 45 ∧ 
  (∃ (terms : ℕ → ℝ), 
    (∀ r, terms r = (Nat.choose 10 r) * (-1)^r * x^(5 - 3*r/2)) ∧
    (∃ r, 5 - 3*r/2 = 2 ∧ terms r = c * x^2)) := by
  sorry

end NUMINAMATH_CALUDE_binomial_expansion_coefficient_l2800_280011


namespace NUMINAMATH_CALUDE_pizza_sharing_l2800_280092

theorem pizza_sharing (total_slices : ℕ) (buzz_ratio waiter_ratio : ℕ) : 
  total_slices = 78 → 
  buzz_ratio = 5 → 
  waiter_ratio = 8 → 
  (waiter_ratio * total_slices) / (buzz_ratio + waiter_ratio) - 20 = 28 := by
sorry

end NUMINAMATH_CALUDE_pizza_sharing_l2800_280092


namespace NUMINAMATH_CALUDE_python_to_boa_ratio_l2800_280070

/-- The ratio of pythons to boa constrictors in a park -/
theorem python_to_boa_ratio :
  let total_snakes : ℕ := 200
  let boa_constrictors : ℕ := 40
  let rattlesnakes : ℕ := 40
  let pythons : ℕ := total_snakes - (boa_constrictors + rattlesnakes)
  (pythons : ℚ) / boa_constrictors = 3 := by
  sorry

end NUMINAMATH_CALUDE_python_to_boa_ratio_l2800_280070


namespace NUMINAMATH_CALUDE_absolute_value_equality_l2800_280052

theorem absolute_value_equality (a b : ℝ) : |a| = |b| → a = b ∨ a = -b := by
  sorry

end NUMINAMATH_CALUDE_absolute_value_equality_l2800_280052


namespace NUMINAMATH_CALUDE_division_problem_l2800_280068

theorem division_problem (n : ℕ) : n / 4 = 5 ∧ n % 4 = 3 → n = 23 := by
  sorry

end NUMINAMATH_CALUDE_division_problem_l2800_280068


namespace NUMINAMATH_CALUDE_complex_multiplication_l2800_280009

theorem complex_multiplication (z : ℂ) : 
  (z.re = -1 ∧ z.im = 1) → z * (1 + Complex.I) = -2 := by
  sorry

end NUMINAMATH_CALUDE_complex_multiplication_l2800_280009


namespace NUMINAMATH_CALUDE_min_correct_problems_is_16_l2800_280094

/-- AMC 10 scoring system and John's strategy -/
structure AMC10 where
  total_problems : Nat
  attempted_problems : Nat
  correct_points : Nat
  unanswered_points : Nat
  min_total_score : Nat

/-- Calculate the minimum number of correctly solved problems -/
def min_correct_problems (test : AMC10) : Nat :=
  let unanswered := test.total_problems - test.attempted_problems
  let unanswered_score := unanswered * test.unanswered_points
  let required_score := test.min_total_score - unanswered_score
  (required_score + test.correct_points - 1) / test.correct_points

/-- Theorem: The minimum number of correctly solved problems is 16 -/
theorem min_correct_problems_is_16 (test : AMC10) 
  (h1 : test.total_problems = 25)
  (h2 : test.attempted_problems = 20)
  (h3 : test.correct_points = 7)
  (h4 : test.unanswered_points = 2)
  (h5 : test.min_total_score = 120) :
  min_correct_problems test = 16 := by
  sorry

end NUMINAMATH_CALUDE_min_correct_problems_is_16_l2800_280094


namespace NUMINAMATH_CALUDE_simplify_expression_l2800_280037

theorem simplify_expression (x : ℝ) : (3*x)^4 + (4*x)*(x^5) = 81*x^4 + 4*x^6 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l2800_280037


namespace NUMINAMATH_CALUDE_furniture_factory_solution_valid_furniture_factory_solution_optimal_l2800_280033

/-- Represents the solution to the furniture factory worker allocation problem -/
def furniture_factory_solution (total_workers : ℕ) 
  (tabletops_per_worker : ℕ) (legs_per_worker : ℕ) 
  (legs_per_table : ℕ) : ℕ × ℕ :=
  (20, 40)

/-- Theorem stating that the solution satisfies the problem conditions -/
theorem furniture_factory_solution_valid 
  (total_workers : ℕ) (tabletops_per_worker : ℕ) 
  (legs_per_worker : ℕ) (legs_per_table : ℕ) :
  let (tabletop_workers, leg_workers) := 
    furniture_factory_solution total_workers tabletops_per_worker legs_per_worker legs_per_table
  (total_workers = tabletop_workers + leg_workers) ∧ 
  (tabletops_per_worker * tabletop_workers * legs_per_table = legs_per_worker * leg_workers) ∧
  (total_workers = 60) ∧ 
  (tabletops_per_worker = 3) ∧ 
  (legs_per_worker = 6) ∧ 
  (legs_per_table = 4) :=
by
  sorry

/-- Theorem stating that the solution maximizes production -/
theorem furniture_factory_solution_optimal 
  (total_workers : ℕ) (tabletops_per_worker : ℕ) 
  (legs_per_worker : ℕ) (legs_per_table : ℕ) :
  let (tabletop_workers, leg_workers) := 
    furniture_factory_solution total_workers tabletops_per_worker legs_per_worker legs_per_table
  ∀ (x y : ℕ), 
    (x + y = total_workers) → 
    (tabletops_per_worker * x * legs_per_table = legs_per_worker * y) →
    (tabletops_per_worker * x ≤ tabletops_per_worker * tabletop_workers) :=
by
  sorry

end NUMINAMATH_CALUDE_furniture_factory_solution_valid_furniture_factory_solution_optimal_l2800_280033


namespace NUMINAMATH_CALUDE_consistency_comparison_l2800_280005

/-- Represents a student's performance in a series of games -/
structure StudentPerformance where
  numGames : ℕ
  avgScore : ℝ
  stdDev : ℝ

/-- Defines what it means for a student to perform more consistently -/
def MoreConsistent (a b : StudentPerformance) : Prop :=
  a.numGames = b.numGames ∧ a.avgScore = b.avgScore ∧ a.stdDev < b.stdDev

theorem consistency_comparison (a b : StudentPerformance) 
  (h1 : a.numGames = b.numGames) 
  (h2 : a.avgScore = b.avgScore) 
  (h3 : a.stdDev < b.stdDev) : 
  MoreConsistent a b :=
sorry

end NUMINAMATH_CALUDE_consistency_comparison_l2800_280005


namespace NUMINAMATH_CALUDE_set_operations_and_subset_l2800_280095

-- Define the sets A, B, and C
def A : Set ℝ := {x | -3 ≤ x ∧ x < 6}
def B : Set ℝ := {x | 2 < x ∧ x < 9}
def C (a : ℝ) : Set ℝ := {x | a < x ∧ x < 2*a + 1}

-- State the theorem
theorem set_operations_and_subset :
  (A ∩ B = {x | 2 < x ∧ x < 6}) ∧
  (A ∪ (Set.univ \ B) = {x | x < 6 ∨ 9 ≤ x}) ∧
  (∀ a : ℝ, C a ⊆ A → a ≤ 5/2) :=
sorry

end NUMINAMATH_CALUDE_set_operations_and_subset_l2800_280095


namespace NUMINAMATH_CALUDE_cyclist_round_trip_time_l2800_280071

/-- Calculates the total time for a cyclist's round trip given specific conditions. -/
theorem cyclist_round_trip_time 
  (total_distance : ℝ)
  (first_segment_distance : ℝ)
  (second_segment_distance : ℝ)
  (first_segment_speed : ℝ)
  (second_segment_speed : ℝ)
  (return_speed : ℝ)
  (h1 : total_distance = first_segment_distance + second_segment_distance)
  (h2 : first_segment_distance = 12)
  (h3 : second_segment_distance = 24)
  (h4 : first_segment_speed = 8)
  (h5 : second_segment_speed = 12)
  (h6 : return_speed = 9)
  : (first_segment_distance / first_segment_speed + 
     second_segment_distance / second_segment_speed + 
     total_distance / return_speed) = 7.5 := by
  sorry

end NUMINAMATH_CALUDE_cyclist_round_trip_time_l2800_280071


namespace NUMINAMATH_CALUDE_ajax_exercise_hours_per_day_l2800_280067

/-- Calculates the number of hours Ajax needs to exercise per day to reach his weight loss goal. -/
theorem ajax_exercise_hours_per_day 
  (initial_weight_kg : ℝ)
  (weight_loss_per_hour_lbs : ℝ)
  (kg_to_lbs_conversion : ℝ)
  (final_weight_lbs : ℝ)
  (days_of_exercise : ℕ)
  (h1 : initial_weight_kg = 80)
  (h2 : weight_loss_per_hour_lbs = 1.5)
  (h3 : kg_to_lbs_conversion = 2.2)
  (h4 : final_weight_lbs = 134)
  (h5 : days_of_exercise = 14) :
  (initial_weight_kg * kg_to_lbs_conversion - final_weight_lbs) / (weight_loss_per_hour_lbs * days_of_exercise) = 2 := by
  sorry

#check ajax_exercise_hours_per_day

end NUMINAMATH_CALUDE_ajax_exercise_hours_per_day_l2800_280067


namespace NUMINAMATH_CALUDE_percentage_of_a_l2800_280072

theorem percentage_of_a (a b c : ℝ) (P : ℝ) : 
  (P / 100) * a = 8 →
  0.08 * b = 2 →
  c = b / a →
  P = 100 := by
sorry

end NUMINAMATH_CALUDE_percentage_of_a_l2800_280072


namespace NUMINAMATH_CALUDE_chairs_per_row_l2800_280062

theorem chairs_per_row (total_rows : ℕ) (occupied_seats : ℕ) (unoccupied_seats : ℕ) :
  total_rows = 40 →
  occupied_seats = 790 →
  unoccupied_seats = 10 →
  (occupied_seats + unoccupied_seats) / total_rows = 20 :=
by
  sorry

end NUMINAMATH_CALUDE_chairs_per_row_l2800_280062


namespace NUMINAMATH_CALUDE_price_reduction_percentage_l2800_280055

/-- Proves that given a price reduction x%, if the sale increases by 80% and the net effect on the sale is 53%, then x = 15. -/
theorem price_reduction_percentage (x : ℝ) : 
  (1 - x / 100) * 1.80 = 1.53 → x = 15 := by
  sorry

end NUMINAMATH_CALUDE_price_reduction_percentage_l2800_280055


namespace NUMINAMATH_CALUDE_p_necessary_not_sufficient_for_q_l2800_280010

theorem p_necessary_not_sufficient_for_q :
  (∃ x, x < 2 ∧ ¬(-2 < x ∧ x < 2)) ∧
  (∀ x, -2 < x ∧ x < 2 → x < 2) :=
by sorry

end NUMINAMATH_CALUDE_p_necessary_not_sufficient_for_q_l2800_280010


namespace NUMINAMATH_CALUDE_exponent_division_l2800_280079

theorem exponent_division (a : ℝ) : a^3 / a^2 = a := by
  sorry

end NUMINAMATH_CALUDE_exponent_division_l2800_280079


namespace NUMINAMATH_CALUDE_smallest_n_for_trig_inequality_l2800_280042

theorem smallest_n_for_trig_inequality : 
  (∃ (n : ℕ), n > 0 ∧ ∀ (x : ℝ), Real.sin x ^ n + Real.cos x ^ n ≤ 2 / n) ∧ 
  (∀ (n : ℕ), n > 0 ∧ (∀ (x : ℝ), Real.sin x ^ n + Real.cos x ^ n ≤ 2 / n) → n ≥ 4) :=
by sorry

end NUMINAMATH_CALUDE_smallest_n_for_trig_inequality_l2800_280042


namespace NUMINAMATH_CALUDE_integral_reciprocal_one_plus_x_squared_l2800_280023

theorem integral_reciprocal_one_plus_x_squared : 
  ∫ (x : ℝ) in (0)..(Real.sqrt 3), 1 / (1 + x^2) = π / 3 := by
  sorry

end NUMINAMATH_CALUDE_integral_reciprocal_one_plus_x_squared_l2800_280023


namespace NUMINAMATH_CALUDE_bug_probability_l2800_280001

/-- Probability of the bug being at vertex A after n meters -/
def P : ℕ → ℚ
  | 0 => 1
  | n + 1 => 1/3 * (1 - P n)

/-- The probability of the bug being at vertex A after 8 meters is 1823/6561 -/
theorem bug_probability : P 8 = 1823 / 6561 := by
  sorry

end NUMINAMATH_CALUDE_bug_probability_l2800_280001


namespace NUMINAMATH_CALUDE_cube_split_l2800_280066

theorem cube_split (m : ℕ) (h1 : m > 1) : 
  (∃ k : ℕ, k ≥ 0 ∧ k < m ∧ m^2 - m + 1 + 2*k = 73) → m = 9 := by
  sorry

end NUMINAMATH_CALUDE_cube_split_l2800_280066


namespace NUMINAMATH_CALUDE_complex_equality_l2800_280047

theorem complex_equality (a : ℝ) : 
  (Complex.re ((a - Complex.I) * (1 - Complex.I) * Complex.I) = 
   Complex.im ((a - Complex.I) * (1 - Complex.I) * Complex.I)) → 
  a = 0 := by
  sorry

end NUMINAMATH_CALUDE_complex_equality_l2800_280047


namespace NUMINAMATH_CALUDE_prob_three_non_defective_l2800_280056

/-- The probability of selecting 3 non-defective pencils from a box of 7 pencils, where 2 are defective. -/
theorem prob_three_non_defective (total : Nat) (defective : Nat) (selected : Nat) :
  total = 7 →
  defective = 2 →
  selected = 3 →
  (Nat.choose (total - defective) selected : ℚ) / (Nat.choose total selected : ℚ) = 2 / 7 := by
  sorry

end NUMINAMATH_CALUDE_prob_three_non_defective_l2800_280056


namespace NUMINAMATH_CALUDE_intersecting_circles_angle_equality_l2800_280024

-- Define the types for points and circles
variable (Point Circle : Type)

-- Define the center of a circle
variable (center : Circle → Point)

-- Define the property of a point being on a circle
variable (on_circle : Point → Circle → Prop)

-- Define the property of two circles intersecting
variable (intersect : Circle → Circle → Prop)

-- Define the property of points being collinear
variable (collinear : Point → Point → Point → Prop)

-- Define the angle between three points
variable (angle : Point → Point → Point → ℝ)

-- State the theorem
theorem intersecting_circles_angle_equality
  (C1 C2 : Circle) (O1 O2 P Q U V : Point) :
  center C1 = O1 →
  center C2 = O2 →
  intersect C1 C2 →
  on_circle P C1 →
  on_circle P C2 →
  on_circle Q C1 →
  on_circle Q C2 →
  on_circle U C1 →
  on_circle V C2 →
  collinear U P V →
  angle U Q V = angle O1 Q O2 := by
  sorry

end NUMINAMATH_CALUDE_intersecting_circles_angle_equality_l2800_280024


namespace NUMINAMATH_CALUDE_spongebob_earnings_l2800_280050

/-- Represents the earnings from selling burgers -/
def burger_earnings (num_burgers : ℕ) (price_per_burger : ℚ) : ℚ :=
  num_burgers * price_per_burger

/-- Represents the earnings from selling large fries -/
def fries_earnings (num_fries : ℕ) (price_per_fries : ℚ) : ℚ :=
  num_fries * price_per_fries

/-- Represents the total earnings for the day -/
def total_earnings (burger_earn : ℚ) (fries_earn : ℚ) : ℚ :=
  burger_earn + fries_earn

theorem spongebob_earnings :
  let num_burgers : ℕ := 30
  let price_per_burger : ℚ := 2
  let num_fries : ℕ := 12
  let price_per_fries : ℚ := 3/2
  let burger_earn := burger_earnings num_burgers price_per_burger
  let fries_earn := fries_earnings num_fries price_per_fries
  total_earnings burger_earn fries_earn = 78 := by
sorry

end NUMINAMATH_CALUDE_spongebob_earnings_l2800_280050


namespace NUMINAMATH_CALUDE_workers_wage_problem_l2800_280082

/-- Worker's wage problem -/
theorem workers_wage_problem (total_days : ℕ) (overall_avg : ℝ) 
  (first_5_avg : ℝ) (second_5_avg : ℝ) (third_5_increase : ℝ) (last_5_decrease : ℝ) :
  total_days = 20 →
  overall_avg = 100 →
  first_5_avg = 90 →
  second_5_avg = 110 →
  third_5_increase = 0.05 →
  last_5_decrease = 0.10 →
  ∃ (eleventh_day_wage : ℝ),
    eleventh_day_wage = second_5_avg * (1 + third_5_increase) ∧
    eleventh_day_wage = 115.50 :=
by sorry

end NUMINAMATH_CALUDE_workers_wage_problem_l2800_280082


namespace NUMINAMATH_CALUDE_max_xy_min_reciprocal_sum_min_squared_sum_max_sqrt_sum_l2800_280083

variable (x y : ℝ)

-- Define the condition
def condition (x y : ℝ) : Prop := x > 0 ∧ y > 0 ∧ 2 * x + y = 1

-- Theorems to prove
theorem max_xy (h : condition x y) : 
  ∃ (m : ℝ), m = 1/8 ∧ ∀ (a b : ℝ), condition a b → a * b ≤ m :=
sorry

theorem min_reciprocal_sum (h : condition x y) :
  ∃ (m : ℝ), m = 9 ∧ ∀ (a b : ℝ), condition a b → m ≤ 2/a + 1/b :=
sorry

theorem min_squared_sum (h : condition x y) :
  ∃ (m : ℝ), m = 1/2 ∧ ∀ (a b : ℝ), condition a b → m ≤ 4*a^2 + b^2 :=
sorry

theorem max_sqrt_sum (h : condition x y) :
  ∃ (m : ℝ), m < 2 ∧ ∀ (a b : ℝ), condition a b → Real.sqrt (2*a) + Real.sqrt b ≤ m :=
sorry

end NUMINAMATH_CALUDE_max_xy_min_reciprocal_sum_min_squared_sum_max_sqrt_sum_l2800_280083


namespace NUMINAMATH_CALUDE_mean_practice_hours_l2800_280064

def practice_hours : List ℕ := [1, 2, 3, 4, 5, 8, 10]
def student_counts : List ℕ := [4, 5, 3, 7, 2, 3, 1]

def total_hours : ℕ := (List.zip practice_hours student_counts).map (fun (h, c) => h * c) |>.sum
def total_students : ℕ := student_counts.sum

theorem mean_practice_hours :
  (total_hours : ℚ) / (total_students : ℚ) = 95 / 25 := by sorry

#eval (95 : ℚ) / 25  -- This should evaluate to 3.8

end NUMINAMATH_CALUDE_mean_practice_hours_l2800_280064


namespace NUMINAMATH_CALUDE_triangle_increase_l2800_280002

theorem triangle_increase (AB BC : ℝ) (h1 : AB = 24) (h2 : BC = 10) :
  let AC := Real.sqrt (AB^2 + BC^2)
  let AB' := AB + 6
  let BC' := BC + 6
  let AC' := Real.sqrt (AB'^2 + BC'^2)
  AC' - AC = 8 := by sorry

end NUMINAMATH_CALUDE_triangle_increase_l2800_280002


namespace NUMINAMATH_CALUDE_intersection_points_sum_greater_than_two_l2800_280000

open Real

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := x * log x - a * x^2 + (2 * a - 1) * x

theorem intersection_points_sum_greater_than_two (a t : ℝ) (x₁ x₂ : ℝ) :
  a ≤ 0 →
  -1 < t →
  t < 0 →
  x₁ < x₂ →
  f a x₁ = t →
  f a x₂ = t →
  x₁ + x₂ > 2 := by
  sorry

end NUMINAMATH_CALUDE_intersection_points_sum_greater_than_two_l2800_280000


namespace NUMINAMATH_CALUDE_monotonic_f_implies_a_range_l2800_280051

-- Define the function f(x)
def f (a : ℝ) (x : ℝ) : ℝ := x^2 + 2*(a-1)*x + 2

-- Define the property of being monotonic in an interval
def isMonotonicIn (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x y, a < x ∧ x < y ∧ y < b → (f x < f y ∨ f y < f x)

-- Theorem statement
theorem monotonic_f_implies_a_range (a : ℝ) :
  isMonotonicIn (f a) 1 2 → a ≤ -1 ∨ a ≥ 0 := by
  sorry

end NUMINAMATH_CALUDE_monotonic_f_implies_a_range_l2800_280051


namespace NUMINAMATH_CALUDE_distance_school_to_david_value_total_distance_sum_l2800_280059

/-- The distance Craig walked from school to David's house -/
def distance_school_to_david : ℝ := sorry

/-- The distance Craig walked from David's house to his own house -/
def distance_david_to_craig : ℝ := 0.7

/-- The total distance Craig walked -/
def total_distance : ℝ := 0.9

/-- Theorem stating that the distance from school to David's house is 0.2 miles -/
theorem distance_school_to_david_value :
  distance_school_to_david = 0.2 :=
by
  sorry

/-- Theorem stating that the total distance is the sum of the two parts -/
theorem total_distance_sum :
  total_distance = distance_school_to_david + distance_david_to_craig :=
by
  sorry

end NUMINAMATH_CALUDE_distance_school_to_david_value_total_distance_sum_l2800_280059


namespace NUMINAMATH_CALUDE_bus_seat_capacity_l2800_280075

/-- Represents the seating arrangement and capacity of a bus -/
structure BusSeating where
  left_seats : ℕ
  right_seats : ℕ
  back_seat_capacity : ℕ
  total_capacity : ℕ

/-- Calculates the number of people each regular seat can hold -/
def seats_capacity (bus : BusSeating) : ℚ :=
  (bus.total_capacity - bus.back_seat_capacity) / (bus.left_seats + bus.right_seats)

/-- Theorem stating that for the given bus configuration, each seat can hold 3 people -/
theorem bus_seat_capacity :
  let bus : BusSeating := {
    left_seats := 15,
    right_seats := 12,
    back_seat_capacity := 7,
    total_capacity := 88
  }
  seats_capacity bus = 3 := by sorry

end NUMINAMATH_CALUDE_bus_seat_capacity_l2800_280075


namespace NUMINAMATH_CALUDE_ben_bought_three_cards_l2800_280013

/-- The number of cards Ben bought -/
def cards_bought : ℕ := 3

/-- The number of cards Tim had -/
def tim_cards : ℕ := 20

/-- The number of cards Ben initially had -/
def ben_initial_cards : ℕ := 37

theorem ben_bought_three_cards :
  (ben_initial_cards + cards_bought = 2 * tim_cards) ∧
  (cards_bought = 3) := by
  sorry

end NUMINAMATH_CALUDE_ben_bought_three_cards_l2800_280013


namespace NUMINAMATH_CALUDE_iphone_savings_l2800_280085

def iphone_cost : ℝ := 600
def discount_rate : ℝ := 0.05
def num_phones : ℕ := 3

def individual_cost : ℝ := iphone_cost * num_phones
def discounted_cost : ℝ := individual_cost * (1 - discount_rate)
def savings : ℝ := individual_cost - discounted_cost

theorem iphone_savings : savings = 90 := by
  sorry

end NUMINAMATH_CALUDE_iphone_savings_l2800_280085


namespace NUMINAMATH_CALUDE_lending_interest_rate_l2800_280087

/-- The interest rate at which a person lends money, given specific borrowing and lending conditions -/
theorem lending_interest_rate (borrowed_amount : ℝ) (borrowing_rate : ℝ) (lending_years : ℝ) (yearly_gain : ℝ) : 
  borrowed_amount = 5000 →
  borrowing_rate = 4 →
  lending_years = 2 →
  yearly_gain = 200 →
  (borrowed_amount * borrowing_rate * lending_years / 100 + 2 * yearly_gain) / (borrowed_amount * lending_years / 100) = 8 := by
  sorry

end NUMINAMATH_CALUDE_lending_interest_rate_l2800_280087


namespace NUMINAMATH_CALUDE_wilson_children_ages_l2800_280035

theorem wilson_children_ages (a b c : ℕ) (h1 : a + b + c = 21) (h2 : a = 4) (h3 : b = 7) : c = 10 := by
  sorry

end NUMINAMATH_CALUDE_wilson_children_ages_l2800_280035


namespace NUMINAMATH_CALUDE_domain_of_g_l2800_280065

-- Define the function f with domain (-1, 0)
def f : Set ℝ := {x : ℝ | -1 < x ∧ x < 0}

-- Define the function g(x) = f(2x+1)
def g (x : ℝ) : Prop := (2 * x + 1) ∈ f

-- Theorem statement
theorem domain_of_g :
  {x : ℝ | g x} = {x : ℝ | -1 < x ∧ x < -1/2} :=
by sorry

end NUMINAMATH_CALUDE_domain_of_g_l2800_280065


namespace NUMINAMATH_CALUDE_total_pencils_l2800_280097

/-- Given that each child has 2 pencils and there are 9 children, 
    prove that the total number of pencils is 18. -/
theorem total_pencils (pencils_per_child : ℕ) (num_children : ℕ) 
  (h1 : pencils_per_child = 2) (h2 : num_children = 9) : 
  pencils_per_child * num_children = 18 := by
sorry

end NUMINAMATH_CALUDE_total_pencils_l2800_280097


namespace NUMINAMATH_CALUDE_trigonometric_simplification_l2800_280086

theorem trigonometric_simplification (α : ℝ) :
  (-Real.sin (π + α) + Real.sin (-α) - Real.tan (2*π + α)) /
  (Real.tan (α + π) + Real.cos (-α) + Real.cos (π - α)) = -1 := by
  sorry

end NUMINAMATH_CALUDE_trigonometric_simplification_l2800_280086


namespace NUMINAMATH_CALUDE_absolute_value_inequality_l2800_280084

theorem absolute_value_inequality (x : ℝ) (h : x ≠ 1) :
  |((2 * x - 1) / (x - 1))| > 3 ↔ (4/5 < x ∧ x < 1) ∨ (1 < x ∧ x < 2) := by
  sorry

end NUMINAMATH_CALUDE_absolute_value_inequality_l2800_280084


namespace NUMINAMATH_CALUDE_function_symmetry_l2800_280073

/-- The function f(x) = (1-x)/(1+x) is symmetric about the line y = x -/
theorem function_symmetry (x : ℝ) : 
  let f : ℝ → ℝ := λ x ↦ (1 - x) / (1 + x)
  f (f x) = x :=
sorry

end NUMINAMATH_CALUDE_function_symmetry_l2800_280073


namespace NUMINAMATH_CALUDE_arithmetic_mean_of_numbers_l2800_280016

def numbers : List ℕ := [3, 11, 7, 9, 15, 13, 8, 19, 17, 21, 14, 7]

theorem arithmetic_mean_of_numbers :
  (numbers.sum : ℚ) / numbers.length = 12 := by sorry

end NUMINAMATH_CALUDE_arithmetic_mean_of_numbers_l2800_280016


namespace NUMINAMATH_CALUDE_age_difference_proof_l2800_280069

/-- Proves the number of years ago when the elder person was twice as old as the younger person -/
theorem age_difference_proof (younger_age elder_age years_ago : ℕ) : 
  younger_age = 35 →
  elder_age - younger_age = 20 →
  elder_age - years_ago = 2 * (younger_age - years_ago) →
  years_ago = 15 := by
sorry

end NUMINAMATH_CALUDE_age_difference_proof_l2800_280069


namespace NUMINAMATH_CALUDE_max_red_tiles_l2800_280025

/-- Represents a square grid of tiles -/
structure TileGrid where
  size : Nat
  colors : Nat

/-- Represents the maximum number of tiles of a single color in a grid -/
def maxSingleColorTiles (grid : TileGrid) : Nat :=
  (grid.size / 2) ^ 2

/-- The problem statement -/
theorem max_red_tiles (grid : TileGrid) :
  grid.size = 100 →
  grid.colors = 4 →
  maxSingleColorTiles grid = 2500 :=
by sorry

end NUMINAMATH_CALUDE_max_red_tiles_l2800_280025


namespace NUMINAMATH_CALUDE_same_last_four_digits_theorem_l2800_280041

theorem same_last_four_digits_theorem (N : ℕ) (a b c d : Fin 10) :
  (a ≠ 0) →
  (N % 10000 = a * 1000 + b * 100 + c * 10 + d) →
  ((N + 2) % 10000 = a * 1000 + b * 100 + c * 10 + d) →
  (a * 100 + b * 10 + c = 999) :=
by sorry

end NUMINAMATH_CALUDE_same_last_four_digits_theorem_l2800_280041


namespace NUMINAMATH_CALUDE_cost_difference_analysis_l2800_280040

/-- Represents the cost difference between option 2 and option 1 -/
def cost_difference (x : ℝ) : ℝ := 54 * x + 9000 - (60 * x + 8800)

/-- Proves that the cost difference is 6x - 200 for x > 20, and positive when x = 30 -/
theorem cost_difference_analysis :
  (∀ x > 20, cost_difference x = 6 * x - 200) ∧
  (cost_difference 30 > 0) := by
  sorry

end NUMINAMATH_CALUDE_cost_difference_analysis_l2800_280040


namespace NUMINAMATH_CALUDE_jimin_candies_count_l2800_280030

/-- The number of candies Jimin gave to Yuna -/
def candies_to_yuna : ℕ := 25

/-- The number of candies Jimin gave to her sister -/
def candies_to_sister : ℕ := 13

/-- The total number of candies Jimin had at first -/
def total_candies : ℕ := candies_to_yuna + candies_to_sister

theorem jimin_candies_count : total_candies = 38 := by
  sorry

end NUMINAMATH_CALUDE_jimin_candies_count_l2800_280030


namespace NUMINAMATH_CALUDE_day_relationship_l2800_280061

/-- Represents days of the week -/
inductive DayOfWeek
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday
  | Saturday
  | Sunday

/-- Represents a specific day in a year -/
structure DayInYear where
  year : ℤ
  dayNumber : ℕ

/-- Function to determine the day of the week for a given day in a year -/
def dayOfWeek (d : DayInYear) : DayOfWeek := sorry

/-- Theorem stating the relationship between specific days and their days of the week -/
theorem day_relationship (M : ℤ) :
  (dayOfWeek ⟨M, 250⟩ = DayOfWeek.Friday) →
  (dayOfWeek ⟨M + 1, 150⟩ = DayOfWeek.Friday) →
  (dayOfWeek ⟨M - 1, 50⟩ = DayOfWeek.Wednesday) := by
  sorry

end NUMINAMATH_CALUDE_day_relationship_l2800_280061


namespace NUMINAMATH_CALUDE_remaining_walk_time_l2800_280036

theorem remaining_walk_time (total_distance : ℝ) (speed : ℝ) (walked_distance : ℝ) : 
  total_distance = 2.5 → 
  speed = 1 / 20 → 
  walked_distance = 1 → 
  (total_distance - walked_distance) / speed = 30 := by
sorry

end NUMINAMATH_CALUDE_remaining_walk_time_l2800_280036


namespace NUMINAMATH_CALUDE_school_poll_intersection_l2800_280088

theorem school_poll_intersection (T C D : Finset ℕ) (h1 : T.card = 230) 
  (h2 : C.card = 171) (h3 : D.card = 137) 
  (h4 : (T \ C).card + (T \ D).card - T.card = 37) : 
  (C ∩ D).card = 115 := by
  sorry

end NUMINAMATH_CALUDE_school_poll_intersection_l2800_280088


namespace NUMINAMATH_CALUDE_hoseok_wire_length_l2800_280014

/-- The length of wire Hoseok bought, given the conditions of the problem -/
def wire_length (triangle_side_length : ℝ) (remaining_wire : ℝ) : ℝ :=
  3 * triangle_side_length + remaining_wire

/-- Theorem stating that the length of wire Hoseok bought is 72 cm -/
theorem hoseok_wire_length :
  wire_length 19 15 = 72 := by
  sorry

end NUMINAMATH_CALUDE_hoseok_wire_length_l2800_280014


namespace NUMINAMATH_CALUDE_normal_peak_at_median_l2800_280034

/-- Represents a normal distribution with mean μ and standard deviation σ -/
structure NormalDistribution (μ σ : ℝ) where
  μ_pos : 0 < σ

/-- The probability density function of a normal distribution -/
noncomputable def pdf (nd : NormalDistribution μ σ) (x : ℝ) : ℝ := sorry

/-- The cumulative distribution function of a normal distribution -/
noncomputable def cdf (nd : NormalDistribution μ σ) (x : ℝ) : ℝ := sorry

/-- The theorem stating that if P(X < 0.3) = 0.5 for a normal distribution,
    then the peak of its PDF occurs at x = 0.3 -/
theorem normal_peak_at_median (μ σ : ℝ) (nd : NormalDistribution μ σ) 
    (h : cdf nd 0.3 = 0.5) : 
    ∀ x : ℝ, pdf nd x ≤ pdf nd 0.3 := by sorry

end NUMINAMATH_CALUDE_normal_peak_at_median_l2800_280034


namespace NUMINAMATH_CALUDE_cos_neg_three_pi_half_l2800_280096

theorem cos_neg_three_pi_half : Real.cos (-3 * π / 2) = 0 := by
  sorry

end NUMINAMATH_CALUDE_cos_neg_three_pi_half_l2800_280096


namespace NUMINAMATH_CALUDE_smallest_six_digit_negative_congruent_to_5_mod_17_l2800_280098

theorem smallest_six_digit_negative_congruent_to_5_mod_17 :
  ∀ n : ℤ, -999999 ≤ n ∧ n < -99999 ∧ n ≡ 5 [ZMOD 17] → n ≥ -100011 :=
by sorry

end NUMINAMATH_CALUDE_smallest_six_digit_negative_congruent_to_5_mod_17_l2800_280098


namespace NUMINAMATH_CALUDE_special_sequence_lower_bound_l2800_280060

/-- A sequence of n consecutive natural numbers with special divisor properties -/
structure SpecialSequence (n : ℕ) :=
  (original : Fin n → ℕ)
  (divisors : Fin n → ℕ)
  (original_ascending : ∀ i j, i < j → original i < original j)
  (divisors_ascending : ∀ i j, i < j → divisors i < divisors j)
  (divisor_property : ∀ i, 1 < divisors i ∧ divisors i < original i ∧ divisors i ∣ original i)

/-- All prime numbers smaller than n -/
def primes_less_than (n : ℕ) : Finset ℕ :=
  (Finset.range n).filter Nat.Prime

/-- The main theorem -/
theorem special_sequence_lower_bound (n : ℕ) (seq : SpecialSequence n) :
  ∀ i, seq.original i > (n ^ (primes_less_than n).card) / (primes_less_than n).prod id :=
sorry

end NUMINAMATH_CALUDE_special_sequence_lower_bound_l2800_280060


namespace NUMINAMATH_CALUDE_shortest_altitude_of_triangle_l2800_280021

/-- Given a triangle with sides 9, 12, and 15, its shortest altitude has length 7.2 -/
theorem shortest_altitude_of_triangle (a b c h : ℝ) : 
  a = 9 ∧ b = 12 ∧ c = 15 ∧ 
  a^2 + b^2 = c^2 ∧
  h * c = 2 * (1/2 * a * b) →
  h = 7.2 := by
  sorry

end NUMINAMATH_CALUDE_shortest_altitude_of_triangle_l2800_280021


namespace NUMINAMATH_CALUDE_sum_of_squares_l2800_280044

open BigOperators

/-- Given a sequence {aₙ} where the sum of the first n terms is 3ⁿ - 1,
    prove that the sum of squares of the first n terms is (1/2)(9ⁿ - 1) -/
theorem sum_of_squares (a : ℕ → ℝ) (n : ℕ) :
  (∀ k : ℕ, k ≤ n → ∑ i in Finset.range k, a i = 3^k - 1) →
  ∑ i in Finset.range n, (a i)^2 = (1/2) * (9^n - 1) :=
by sorry

end NUMINAMATH_CALUDE_sum_of_squares_l2800_280044
