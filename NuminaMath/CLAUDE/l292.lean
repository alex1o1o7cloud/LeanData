import Mathlib

namespace NUMINAMATH_CALUDE_no_hamiltonian_cycle_in_circ_2016_2_3_l292_29235

/-- A circulant digraph with n vertices and jump sizes a and b -/
structure CirculantDigraph (n : ℕ) (a b : ℕ) where
  vertices : Fin n

/-- Condition for the existence of a Hamiltonian cycle in a circulant digraph -/
def has_hamiltonian_cycle (G : CirculantDigraph n a b) : Prop :=
  ∃ (s t : ℕ), s + t = Nat.gcd n (a - b) ∧ Nat.gcd n (s * a + t * b) = 1

/-- The main theorem about the non-existence of a Hamiltonian cycle in Circ(2016; 2, 3) -/
theorem no_hamiltonian_cycle_in_circ_2016_2_3 :
  ¬ ∃ (G : CirculantDigraph 2016 2 3), has_hamiltonian_cycle G :=
by sorry

end NUMINAMATH_CALUDE_no_hamiltonian_cycle_in_circ_2016_2_3_l292_29235


namespace NUMINAMATH_CALUDE_intersection_distance_l292_29240

/-- Represents a point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- The start point of the line -/
def startPoint : Point3D := ⟨1, 2, 3⟩

/-- The end point of the line -/
def endPoint : Point3D := ⟨3, 6, 7⟩

/-- The center of the unit sphere -/
def sphereCenter : Point3D := ⟨0, 0, 0⟩

/-- The radius of the unit sphere -/
def sphereRadius : ℝ := 1

/-- Theorem stating that the distance between the two intersection points of the line and the unit sphere is 12√145/33 -/
theorem intersection_distance : 
  ∃ (p1 p2 : Point3D), 
    (∃ (t1 t2 : ℝ), 
      p1 = ⟨startPoint.x + t1 * (endPoint.x - startPoint.x), 
            startPoint.y + t1 * (endPoint.y - startPoint.y), 
            startPoint.z + t1 * (endPoint.z - startPoint.z)⟩ ∧
      p2 = ⟨startPoint.x + t2 * (endPoint.x - startPoint.x), 
            startPoint.y + t2 * (endPoint.y - startPoint.y), 
            startPoint.z + t2 * (endPoint.z - startPoint.z)⟩ ∧
      (p1.x - sphereCenter.x)^2 + (p1.y - sphereCenter.y)^2 + (p1.z - sphereCenter.z)^2 = sphereRadius^2 ∧
      (p2.x - sphereCenter.x)^2 + (p2.y - sphereCenter.y)^2 + (p2.z - sphereCenter.z)^2 = sphereRadius^2) →
    ((p1.x - p2.x)^2 + (p1.y - p2.y)^2 + (p1.z - p2.z)^2) = (12 * Real.sqrt 145 / 33)^2 := by
  sorry

end NUMINAMATH_CALUDE_intersection_distance_l292_29240


namespace NUMINAMATH_CALUDE_smallest_four_digit_divisible_by_35_second_smallest_four_digit_divisible_by_35_l292_29255

def is_four_digit (n : ℕ) : Prop := 1000 ≤ n ∧ n ≤ 9999

def divisible_by_35 (n : ℕ) : Prop := n % 35 = 0

theorem smallest_four_digit_divisible_by_35 :
  (∀ n : ℕ, is_four_digit n ∧ divisible_by_35 n → 1005 ≤ n) ∧
  is_four_digit 1005 ∧ divisible_by_35 1005 :=
sorry

theorem second_smallest_four_digit_divisible_by_35 :
  (∀ n : ℕ, is_four_digit n ∧ divisible_by_35 n ∧ n ≠ 1005 → 1045 ≤ n) ∧
  is_four_digit 1045 ∧ divisible_by_35 1045 :=
sorry

end NUMINAMATH_CALUDE_smallest_four_digit_divisible_by_35_second_smallest_four_digit_divisible_by_35_l292_29255


namespace NUMINAMATH_CALUDE_graph_chromatic_number_l292_29213

/-- Represents a vertex in the graph -/
inductive Vertex : Type
| x : Vertex
| y : Vertex
| z : Vertex
| w : Vertex
| v : Vertex
| u : Vertex

/-- The graph structure -/
def Graph : Type := Vertex → Vertex → Prop

/-- The degree of a vertex in the graph -/
def degree (G : Graph) (v : Vertex) : ℕ := sorry

/-- The chromatic number of a graph -/
def chromaticNumber (G : Graph) : ℕ := sorry

/-- Our specific graph -/
def G : Graph := sorry

/-- The theorem stating that the chromatic number of our graph is 3 -/
theorem graph_chromatic_number :
  (degree G Vertex.x = 5) →
  (degree G Vertex.z = 4) →
  (degree G Vertex.y = 3) →
  (¬ G Vertex.x Vertex.u) →
  (¬ G Vertex.z Vertex.v) →
  (G Vertex.x Vertex.y ∧ G Vertex.y Vertex.z ∧ G Vertex.z Vertex.x) →
  chromaticNumber G = 3 := by
  sorry

end NUMINAMATH_CALUDE_graph_chromatic_number_l292_29213


namespace NUMINAMATH_CALUDE_probability_theorem_l292_29275

def num_questions : ℕ := 5

def valid_sum (a b : ℕ) : Prop :=
  4 ≤ a + b ∧ a + b < 8

def num_valid_combinations : ℕ := 7

def total_combinations : ℕ := num_questions * (num_questions - 1) / 2

theorem probability_theorem :
  (num_valid_combinations : ℚ) / (total_combinations : ℚ) = 7 / 10 := by sorry

end NUMINAMATH_CALUDE_probability_theorem_l292_29275


namespace NUMINAMATH_CALUDE_angle_A_measure_l292_29206

noncomputable def triangle_ABC (A B C : ℝ) (a b c : ℝ) : Prop :=
  -- Define a triangle ABC with sides a, b, c opposite to angles A, B, C
  true  -- Placeholder, as we don't need to specify all triangle properties

theorem angle_A_measure (A B C : ℝ) (a b c : ℝ) :
  triangle_ABC A B C a b c →
  a = Real.sqrt 2 →
  b = 2 →
  Real.sin B + Real.cos B = Real.sqrt 2 →
  A = π / 6 := by
  sorry


end NUMINAMATH_CALUDE_angle_A_measure_l292_29206


namespace NUMINAMATH_CALUDE_aloks_order_l292_29231

/-- Given Alok's order and payment information, prove the number of mixed vegetable plates ordered -/
theorem aloks_order (chapati_count : ℕ) (rice_count : ℕ) (icecream_count : ℕ) 
  (chapati_cost : ℕ) (rice_cost : ℕ) (vegetable_cost : ℕ) (total_paid : ℕ) :
  chapati_count = 16 →
  rice_count = 5 →
  icecream_count = 6 →
  chapati_cost = 6 →
  rice_cost = 45 →
  vegetable_cost = 70 →
  total_paid = 1015 →
  ∃ (vegetable_count : ℕ), 
    total_paid = chapati_count * chapati_cost + rice_count * rice_cost + vegetable_count * vegetable_cost + 
      (total_paid - (chapati_count * chapati_cost + rice_count * rice_cost + vegetable_count * vegetable_cost)) ∧
    vegetable_count = 9 :=
by sorry

end NUMINAMATH_CALUDE_aloks_order_l292_29231


namespace NUMINAMATH_CALUDE_sin_double_angle_given_sin_pi_fourth_minus_x_l292_29282

theorem sin_double_angle_given_sin_pi_fourth_minus_x
  (x : ℝ) (h : Real.sin (π/4 - x) = 3/5) :
  Real.sin (2*x) = 7/25 := by
  sorry

end NUMINAMATH_CALUDE_sin_double_angle_given_sin_pi_fourth_minus_x_l292_29282


namespace NUMINAMATH_CALUDE_cos_pi_minus_alpha_l292_29271

theorem cos_pi_minus_alpha (α : Real) 
  (h1 : 0 < α) 
  (h2 : α < π) 
  (h3 : 3 * Real.sin (2 * α) = Real.sin α) : 
  Real.cos (π - α) = -1/6 := by
sorry

end NUMINAMATH_CALUDE_cos_pi_minus_alpha_l292_29271


namespace NUMINAMATH_CALUDE_prism_intersection_area_l292_29286

/-- Represents a point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Represents a plane in 3D space -/
structure Plane where
  a : ℝ
  b : ℝ
  c : ℝ
  d : ℝ

/-- Represents a rectangular prism -/
structure RectangularPrism where
  width : ℝ
  length : ℝ
  height : ℝ

/-- Calculate the area of intersection between a rectangular prism and a plane -/
def intersectionArea (prism : RectangularPrism) (plane : Plane) : ℝ :=
  sorry

theorem prism_intersection_area :
  let prism : RectangularPrism := ⟨8, 12, 0⟩  -- height is arbitrary, set to 0
  let plane : Plane := ⟨3, -5, 6, 30⟩
  intersectionArea prism plane = 64.92 := by sorry

end NUMINAMATH_CALUDE_prism_intersection_area_l292_29286


namespace NUMINAMATH_CALUDE_condition_relationship_l292_29297

theorem condition_relationship :
  (∀ x : ℝ, (x - 1) / (x + 2) ≥ 0 → (x - 1) * (x + 2) ≥ 0) ∧
  (∃ x : ℝ, (x - 1) * (x + 2) ≥ 0 ∧ ¬((x - 1) / (x + 2) ≥ 0)) :=
by sorry

end NUMINAMATH_CALUDE_condition_relationship_l292_29297


namespace NUMINAMATH_CALUDE_minimum_cars_with_all_characteristics_l292_29269

theorem minimum_cars_with_all_characteristics 
  (total : ℕ) 
  (zhiguli dark_colored male_drivers with_passengers : ℕ) 
  (h_total : total = 20)
  (h_zhiguli : zhiguli = 14)
  (h_dark : dark_colored = 15)
  (h_male : male_drivers = 17)
  (h_passengers : with_passengers = 18) :
  total - ((total - zhiguli) + (total - dark_colored) + (total - male_drivers) + (total - with_passengers)) = 4 := by
sorry

end NUMINAMATH_CALUDE_minimum_cars_with_all_characteristics_l292_29269


namespace NUMINAMATH_CALUDE_largest_even_digit_multiple_of_5_proof_l292_29274

def has_only_even_digits (n : ℕ) : Prop :=
  ∀ d, d ∈ n.digits 10 → d % 2 = 0

def largest_even_digit_multiple_of_5 : ℕ := 86880

theorem largest_even_digit_multiple_of_5_proof :
  (has_only_even_digits largest_even_digit_multiple_of_5) ∧
  (largest_even_digit_multiple_of_5 < 100000) ∧
  (largest_even_digit_multiple_of_5 % 5 = 0) ∧
  (∀ m : ℕ, m > largest_even_digit_multiple_of_5 →
    ¬(has_only_even_digits m ∧ m < 100000 ∧ m % 5 = 0)) :=
by sorry

end NUMINAMATH_CALUDE_largest_even_digit_multiple_of_5_proof_l292_29274


namespace NUMINAMATH_CALUDE_divisibility_problem_specific_divisibility_problem_l292_29211

theorem divisibility_problem (n : ℕ) (d : ℕ) (h : d > 0) :
  ∃ (x y : ℕ),
    (x = (d - n % d) % d) ∧
    (y = n % d) ∧
    ((n + x) % d = 0) ∧
    ((n - y) % d = 0) ∧
    (∀ x' : ℕ, x' < x → (n + x') % d ≠ 0) ∧
    (∀ y' : ℕ, y' < y → (n - y') % d ≠ 0) :=
by sorry

-- Specific instance for the given problem
theorem specific_divisibility_problem :
  ∃ (x y : ℕ),
    (x = 10) ∧
    (y = 27) ∧
    ((1100 + x) % 37 = 0) ∧
    ((1100 - y) % 37 = 0) ∧
    (∀ x' : ℕ, x' < x → (1100 + x') % 37 ≠ 0) ∧
    (∀ y' : ℕ, y' < y → (1100 - y') % 37 ≠ 0) :=
by sorry

end NUMINAMATH_CALUDE_divisibility_problem_specific_divisibility_problem_l292_29211


namespace NUMINAMATH_CALUDE_complement_of_B_in_A_l292_29215

def A : Set ℕ := {2, 3, 4}
def B (a : ℕ) : Set ℕ := {a + 2, a}

theorem complement_of_B_in_A (a : ℕ) (h : A ∩ B a = B a) : 
  (A \ B a) = {3} := by
  sorry

end NUMINAMATH_CALUDE_complement_of_B_in_A_l292_29215


namespace NUMINAMATH_CALUDE_wheat_field_and_fertilizer_l292_29242

theorem wheat_field_and_fertilizer 
  (field_size : ℕ) 
  (fertilizer_amount : ℕ) 
  (h1 : 6 * field_size = fertilizer_amount + 300)
  (h2 : 5 * field_size + 200 = fertilizer_amount) :
  field_size = 500 ∧ fertilizer_amount = 2700 := by
sorry

end NUMINAMATH_CALUDE_wheat_field_and_fertilizer_l292_29242


namespace NUMINAMATH_CALUDE_prize_selection_theorem_l292_29236

/-- Represents the systematic sampling of prizes -/
def systematicSampling (totalPrizes : ℕ) (sampleSize : ℕ) (firstPrize : ℕ) : List ℕ :=
  let interval := totalPrizes / sampleSize
  List.range sampleSize |>.map (fun i => firstPrize + i * interval)

/-- Theorem: Given the conditions of the prize selection, the other four prizes are 46, 86, 126, and 166 -/
theorem prize_selection_theorem (totalPrizes : ℕ) (sampleSize : ℕ) (firstPrize : ℕ) 
    (h1 : totalPrizes = 200)
    (h2 : sampleSize = 5)
    (h3 : firstPrize = 6) :
  systematicSampling totalPrizes sampleSize firstPrize = [6, 46, 86, 126, 166] := by
  sorry

#eval systematicSampling 200 5 6

end NUMINAMATH_CALUDE_prize_selection_theorem_l292_29236


namespace NUMINAMATH_CALUDE_sunday_to_saturday_ratio_l292_29291

/-- Tameka's cracker box sales over three days -/
structure CrackerSales where
  friday : ℕ
  saturday : ℕ
  sunday : ℕ
  total : ℕ
  h1 : friday = 40
  h2 : saturday = 2 * friday - 10
  h3 : total = friday + saturday + sunday
  h4 : total = 145

/-- The ratio of boxes sold on Sunday to boxes sold on Saturday is 1/2 -/
theorem sunday_to_saturday_ratio (sales : CrackerSales) :
  sales.sunday / sales.saturday = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_sunday_to_saturday_ratio_l292_29291


namespace NUMINAMATH_CALUDE_minimum_packs_for_90_cans_l292_29245

/-- Represents the available pack sizes for soda cans -/
def PackSizes : List Nat := [6, 12, 24]

/-- The total number of cans we need to buy -/
def TotalCans : Nat := 90

/-- A function that calculates the minimum number of packs needed -/
def MinimumPacks (packSizes : List Nat) (totalCans : Nat) : Nat :=
  sorry -- Proof implementation goes here

/-- Theorem stating that the minimum number of packs needed is 5 -/
theorem minimum_packs_for_90_cans : 
  MinimumPacks PackSizes TotalCans = 5 := by
  sorry -- Proof goes here

end NUMINAMATH_CALUDE_minimum_packs_for_90_cans_l292_29245


namespace NUMINAMATH_CALUDE_vector_at_negative_one_l292_29238

/-- A line parameterized by t in 3D space -/
structure ParametricLine where
  -- The vector on the line at t = 0
  origin : Fin 3 → ℝ
  -- The vector on the line at t = 1
  point_at_one : Fin 3 → ℝ

/-- The vector on the line at a given t -/
def vector_at_t (line : ParametricLine) (t : ℝ) : Fin 3 → ℝ :=
  λ i => line.origin i + t * (line.point_at_one i - line.origin i)

/-- The theorem stating the vector at t = -1 for the given line -/
theorem vector_at_negative_one (line : ParametricLine) 
  (h0 : line.origin = λ i => [2, 4, 9].get i)
  (h1 : line.point_at_one = λ i => [3, 1, 5].get i) :
  vector_at_t line (-1) = λ i => [1, 7, 13].get i := by
  sorry

end NUMINAMATH_CALUDE_vector_at_negative_one_l292_29238


namespace NUMINAMATH_CALUDE_simple_interest_problem_l292_29276

/-- Given a principal sum and an interest rate, if increasing the rate by 5% over 10 years
    results in Rs. 600 more interest, then the principal sum must be Rs. 1200. -/
theorem simple_interest_problem (P R : ℝ) (h : P > 0) (r : R > 0) :
  (P * (R + 5) * 10) / 100 - (P * R * 10) / 100 = 600 →
  P = 1200 := by
sorry

end NUMINAMATH_CALUDE_simple_interest_problem_l292_29276


namespace NUMINAMATH_CALUDE_coefficient_of_x_in_expansion_l292_29262

theorem coefficient_of_x_in_expansion : ∃ (c : ℤ), c = -16 ∧ 
  (∀ (x : ℝ), (1 + x) * (2 - x)^4 = c * x + (2^4 + 4 * 2^3 * x^0 + 6 * 2^2 * x^2 + 4 * 2 * x^3 + x^4)) := by
  sorry

end NUMINAMATH_CALUDE_coefficient_of_x_in_expansion_l292_29262


namespace NUMINAMATH_CALUDE_diophantine_equation_solution_l292_29268

theorem diophantine_equation_solution (x y : ℕ) (h : 65 * x - 43 * y = 2) :
  ∃ t : ℤ, t ≤ 0 ∧ x = 4 - 43 * t ∧ y = 6 - 65 * t := by
  sorry

end NUMINAMATH_CALUDE_diophantine_equation_solution_l292_29268


namespace NUMINAMATH_CALUDE_greatest_n_product_consecutive_odds_l292_29208

theorem greatest_n_product_consecutive_odds : 
  ∃ (n : ℕ), n = 899 ∧ 
  n < 1000 ∧ 
  (∃ (m : ℤ), 4 * n^3 - 3 * n = (2 * m - 1) * (2 * m + 1)) ∧
  (∀ (k : ℕ), k < 1000 → k > n → 
    ¬∃ (m : ℤ), 4 * k^3 - 3 * k = (2 * m - 1) * (2 * m + 1)) :=
by sorry

end NUMINAMATH_CALUDE_greatest_n_product_consecutive_odds_l292_29208


namespace NUMINAMATH_CALUDE_probability_of_two_mismatches_l292_29246

/-- Represents a set of pens and caps -/
structure PenSet :=
  (pens : Finset (Fin 3))
  (caps : Finset (Fin 3))

/-- Represents a pairing of pens and caps -/
def Pairing := Fin 3 → Fin 3

/-- The set of all possible pairings -/
def allPairings : Finset Pairing := sorry

/-- Predicate for a pairing that mismatches two pairs -/
def mismatchesTwoPairs (p : Pairing) : Prop := sorry

/-- The number of pairings that mismatch two pairs -/
def numMismatchedPairings : Nat := sorry

theorem probability_of_two_mismatches (ps : PenSet) :
  (numMismatchedPairings : ℚ) / (Finset.card allPairings : ℚ) = 1 / 2 := by sorry

end NUMINAMATH_CALUDE_probability_of_two_mismatches_l292_29246


namespace NUMINAMATH_CALUDE_max_n_value_l292_29247

theorem max_n_value (a b c : ℝ) (n : ℕ) 
  (h1 : a > b) (h2 : b > c)
  (h3 : ∀ a b c, a > b → b > c → 1 / (a - b) + 1 / (b - c) ≥ n / (a - c)) :
  n ≤ 4 ∧ ∃ a b c, a > b ∧ b > c ∧ 1 / (a - b) + 1 / (b - c) = 4 / (a - c) :=
sorry

end NUMINAMATH_CALUDE_max_n_value_l292_29247


namespace NUMINAMATH_CALUDE_geometric_sequence_product_roots_product_geometric_sequence_problem_l292_29233

def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

theorem geometric_sequence_product (a : ℕ → ℝ) (h : geometric_sequence a) :
  ∀ i j k l : ℕ, i + j = k + l → a i * a j = a k * a l :=
sorry

theorem roots_product (p q r : ℝ) (x y : ℝ) (hx : p * x^2 + q * x + r = 0) (hy : p * y^2 + q * y + r = 0) :
  x * y = r / p :=
sorry

theorem geometric_sequence_problem (a : ℕ → ℝ) 
  (h_geo : geometric_sequence a)
  (h_roots : 3 * a 1^2 - 2 * a 1 - 6 = 0 ∧ 3 * a 10^2 - 2 * a 10 - 6 = 0) :
  a 4 * a 7 = -2 :=
sorry

end NUMINAMATH_CALUDE_geometric_sequence_product_roots_product_geometric_sequence_problem_l292_29233


namespace NUMINAMATH_CALUDE_inequality_proof_l292_29253

theorem inequality_proof (a : ℝ) (h1 : 0 < a) (h2 : a < 1/2) :
  a^(Real.sqrt a) > a^(a^a) ∧ a^(a^a) > a := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l292_29253


namespace NUMINAMATH_CALUDE_sqrt_sum_implies_product_l292_29281

theorem sqrt_sum_implies_product (x : ℝ) :
  Real.sqrt (10 + x) + Real.sqrt (30 - x) = 8 →
  (10 + x) * (30 - x) = 144 := by
sorry

end NUMINAMATH_CALUDE_sqrt_sum_implies_product_l292_29281


namespace NUMINAMATH_CALUDE_students_liking_both_desserts_l292_29266

/-- Given a class of students, calculate the number who like both apple pie and chocolate cake. -/
theorem students_liking_both_desserts 
  (total_students : ℕ) 
  (like_apple_pie : ℕ) 
  (like_chocolate_cake : ℕ) 
  (like_neither : ℕ) 
  (h1 : total_students = 50)
  (h2 : like_apple_pie = 22)
  (h3 : like_chocolate_cake = 20)
  (h4 : like_neither = 15) :
  like_apple_pie + like_chocolate_cake - (total_students - like_neither) = 7 := by
  sorry

#check students_liking_both_desserts

end NUMINAMATH_CALUDE_students_liking_both_desserts_l292_29266


namespace NUMINAMATH_CALUDE_expression_simplification_l292_29257

theorem expression_simplification (x : ℝ) : 
  ((3 * x - 1) - 5 * x) / 3 = -2/3 * x - 1/3 := by sorry

end NUMINAMATH_CALUDE_expression_simplification_l292_29257


namespace NUMINAMATH_CALUDE_unique_cube_root_property_l292_29234

theorem unique_cube_root_property : ∃! (n : ℕ), 
  n > 0 ∧ 
  (∃ (m : ℕ), n = m^3 ∧ m = n / 1000) ∧
  n = 32768 := by
  sorry

end NUMINAMATH_CALUDE_unique_cube_root_property_l292_29234


namespace NUMINAMATH_CALUDE_tire_purchase_cost_total_cost_proof_l292_29252

/-- Calculates the total cost of purchasing tires with given prices and tax rate -/
theorem tire_purchase_cost (num_tires : ℕ) (price1 : ℚ) (price2 : ℚ) (tax_rate : ℚ) : ℚ :=
  let first_group_cost := min num_tires 4 * price1
  let second_group_cost := max (num_tires - 4) 0 * price2
  let subtotal := first_group_cost + second_group_cost
  let tax := subtotal * tax_rate
  subtotal + tax

/-- Proves that the total cost of purchasing 8 tires with given prices and tax rate is 3.78 -/
theorem total_cost_proof :
  tire_purchase_cost 8 (1/2) (2/5) (1/20) = 189/50 :=
by sorry

end NUMINAMATH_CALUDE_tire_purchase_cost_total_cost_proof_l292_29252


namespace NUMINAMATH_CALUDE_randy_cheese_purchase_l292_29285

/-- The number of slices in a package of cheddar cheese -/
def cheddar_slices : ℕ := 12

/-- The number of slices in a package of Swiss cheese -/
def swiss_slices : ℕ := 28

/-- The smallest number of slices of each type that Randy could have bought -/
def smallest_equal_slices : ℕ := 84

theorem randy_cheese_purchase :
  smallest_equal_slices = Nat.lcm cheddar_slices swiss_slices ∧
  smallest_equal_slices % cheddar_slices = 0 ∧
  smallest_equal_slices % swiss_slices = 0 ∧
  ∀ n : ℕ, (n % cheddar_slices = 0 ∧ n % swiss_slices = 0) → n ≥ smallest_equal_slices := by
  sorry

end NUMINAMATH_CALUDE_randy_cheese_purchase_l292_29285


namespace NUMINAMATH_CALUDE_valid_draws_eq_189_l292_29272

def total_cards : ℕ := 12
def cards_per_color : ℕ := 3
def num_colors : ℕ := 4
def cards_to_draw : ℕ := 3

def valid_draws : ℕ := Nat.choose total_cards cards_to_draw - 
                        (num_colors * Nat.choose cards_per_color cards_to_draw) - 
                        (Nat.choose cards_per_color 2 * Nat.choose (total_cards - cards_per_color) 1)

theorem valid_draws_eq_189 : valid_draws = 189 := by sorry

end NUMINAMATH_CALUDE_valid_draws_eq_189_l292_29272


namespace NUMINAMATH_CALUDE_marta_tips_l292_29226

/-- Calculates the amount of tips Marta received given her total earnings, hourly rate, and hours worked -/
def tips_received (total_earnings hourly_rate hours_worked : ℕ) : ℕ :=
  total_earnings - hourly_rate * hours_worked

/-- Proves that Marta received $50 in tips -/
theorem marta_tips : tips_received 240 10 19 = 50 := by
  sorry

end NUMINAMATH_CALUDE_marta_tips_l292_29226


namespace NUMINAMATH_CALUDE_completing_square_sum_l292_29254

theorem completing_square_sum (a b c : ℤ) : 
  (∀ x : ℝ, 49 * x^2 + 70 * x - 121 = 0 ↔ (a * x + b)^2 = c) →
  a > 0 →
  a + b + c = 158 := by
  sorry

end NUMINAMATH_CALUDE_completing_square_sum_l292_29254


namespace NUMINAMATH_CALUDE_log_equality_implies_golden_ratio_l292_29280

theorem log_equality_implies_golden_ratio (p q : ℝ) (hp : p > 0) (hq : q > 0) :
  (Real.log p / Real.log 9 = Real.log q / Real.log 12) ∧
  (Real.log p / Real.log 9 = Real.log (p + q) / Real.log 16) →
  q / p = (1 + Real.sqrt 5) / 2 := by
  sorry

end NUMINAMATH_CALUDE_log_equality_implies_golden_ratio_l292_29280


namespace NUMINAMATH_CALUDE_halfway_between_fractions_average_of_fractions_l292_29284

theorem halfway_between_fractions :
  (1/8 : ℚ) + (1/10 : ℚ) = (9/40 : ℚ) :=
by sorry

theorem average_of_fractions :
  ((1/8 : ℚ) + (1/10 : ℚ)) / 2 = (9/80 : ℚ) :=
by sorry

end NUMINAMATH_CALUDE_halfway_between_fractions_average_of_fractions_l292_29284


namespace NUMINAMATH_CALUDE_steps_down_empire_state_proof_l292_29216

/-- The number of steps taken to get down the Empire State Building -/
def steps_down_empire_state : ℕ := sorry

/-- The number of steps taken from the Empire State Building to Madison Square Garden -/
def steps_to_madison_square : ℕ := 315

/-- The total number of steps taken to get to Madison Square Garden -/
def total_steps : ℕ := 991

/-- Theorem stating that the number of steps taken to get down the Empire State Building is 676 -/
theorem steps_down_empire_state_proof : 
  steps_down_empire_state = total_steps - steps_to_madison_square := by sorry

end NUMINAMATH_CALUDE_steps_down_empire_state_proof_l292_29216


namespace NUMINAMATH_CALUDE_max_leftover_pencils_l292_29241

theorem max_leftover_pencils :
  ∀ (n : ℕ), 
  ∃ (q : ℕ), 
  n = 7 * q + (n % 7) ∧ 
  n % 7 ≤ 6 ∧
  ∀ (r : ℕ), r > n % 7 → r > 6 := by
  sorry

end NUMINAMATH_CALUDE_max_leftover_pencils_l292_29241


namespace NUMINAMATH_CALUDE_rectangular_prism_sum_l292_29295

/-- A rectangular prism is a three-dimensional geometric shape. -/
structure RectangularPrism where
  edges : ℕ
  corners : ℕ
  faces : ℕ

/-- The properties of a rectangular prism -/
def is_valid_rectangular_prism (rp : RectangularPrism) : Prop :=
  rp.edges = 12 ∧ rp.corners = 8 ∧ rp.faces = 6

/-- The theorem stating that the sum of edges, corners, and faces of a rectangular prism is 26 -/
theorem rectangular_prism_sum (rp : RectangularPrism) 
  (h : is_valid_rectangular_prism rp) : 
  rp.edges + rp.corners + rp.faces = 26 := by
  sorry

end NUMINAMATH_CALUDE_rectangular_prism_sum_l292_29295


namespace NUMINAMATH_CALUDE_highest_power_of_six_in_twelve_factorial_l292_29277

/-- The highest power of 6 that divides 12! is 6^5 -/
theorem highest_power_of_six_in_twelve_factorial :
  ∃ k : ℕ, (12 : ℕ).factorial = 6^5 * k ∧ ¬(∃ m : ℕ, (12 : ℕ).factorial = 6^6 * m) := by
  sorry

end NUMINAMATH_CALUDE_highest_power_of_six_in_twelve_factorial_l292_29277


namespace NUMINAMATH_CALUDE_linear_function_point_relation_l292_29261

/-- Given that (-1, y₁) and (3, y₂) lie on the graph of y = 2x + 1, prove that y₁ < y₂ -/
theorem linear_function_point_relation (y₁ y₂ : ℝ) :
  (y₁ = 2 * (-1) + 1) → (y₂ = 2 * 3 + 1) → y₁ < y₂ := by
  sorry

end NUMINAMATH_CALUDE_linear_function_point_relation_l292_29261


namespace NUMINAMATH_CALUDE_fourth_number_in_sequence_l292_29270

theorem fourth_number_in_sequence (s : Fin 7 → ℝ) 
  (h1 : (s 0 + s 1 + s 2 + s 3) / 4 = 4)
  (h2 : (s 3 + s 4 + s 5 + s 6) / 4 = 4)
  (h3 : (s 0 + s 1 + s 2 + s 3 + s 4 + s 5 + s 6) / 7 = 3) :
  s 3 = 5.5 := by
  sorry

end NUMINAMATH_CALUDE_fourth_number_in_sequence_l292_29270


namespace NUMINAMATH_CALUDE_integer_root_of_cubic_l292_29218

theorem integer_root_of_cubic (b c : ℚ) :
  (∃ x : ℤ, x^3 + b*x + c = 0) →
  (Complex.exp (3 - Real.sqrt 3))^3 + b*(Complex.exp (3 - Real.sqrt 3)) + c = 0 →
  (∃ x : ℤ, x^3 + b*x + c = 0 ∧ x = -6) :=
by sorry

end NUMINAMATH_CALUDE_integer_root_of_cubic_l292_29218


namespace NUMINAMATH_CALUDE_percent_relation_l292_29258

theorem percent_relation (x y z : ℝ) 
  (h1 : 0.45 * z = 0.39 * y) 
  (h2 : y = 0.75 * x) : 
  z = 0.65 * x := by
sorry

end NUMINAMATH_CALUDE_percent_relation_l292_29258


namespace NUMINAMATH_CALUDE_barbaras_total_cost_l292_29228

/-- The cost of Barbara's purchase at the butcher's --/
def barbaras_purchase_cost (steak_weight : Real) (steak_price : Real) 
  (chicken_weight : Real) (chicken_price : Real) : Real :=
  steak_weight * steak_price + chicken_weight * chicken_price

/-- Theorem stating the total cost of Barbara's purchase --/
theorem barbaras_total_cost : 
  barbaras_purchase_cost 2 15 1.5 8 = 42 := by
  sorry

end NUMINAMATH_CALUDE_barbaras_total_cost_l292_29228


namespace NUMINAMATH_CALUDE_trapezoid_shorter_diagonal_l292_29264

structure Trapezoid where
  EF : ℝ
  GH : ℝ
  side1 : ℝ
  side2 : ℝ
  acute_E : Bool
  acute_F : Bool

def shorter_diagonal (t : Trapezoid) : ℝ := sorry

theorem trapezoid_shorter_diagonal 
  (t : Trapezoid) 
  (h1 : t.EF = 20) 
  (h2 : t.GH = 26) 
  (h3 : t.side1 = 13) 
  (h4 : t.side2 = 15) 
  (h5 : t.acute_E = true) 
  (h6 : t.acute_F = true) : 
  shorter_diagonal t = Real.sqrt 1496 / 3 := by sorry

end NUMINAMATH_CALUDE_trapezoid_shorter_diagonal_l292_29264


namespace NUMINAMATH_CALUDE_base4_addition_subtraction_l292_29223

/-- Converts a base 4 number represented as a list of digits to a natural number. -/
def base4ToNat (digits : List Nat) : Nat :=
  digits.foldl (fun acc d => 4 * acc + d) 0

/-- Converts a natural number to its base 4 representation as a list of digits. -/
def natToBase4 (n : Nat) : List Nat :=
  if n = 0 then [0] else
  let rec aux (m : Nat) (acc : List Nat) : List Nat :=
    if m = 0 then acc else aux (m / 4) ((m % 4) :: acc)
  aux n []

theorem base4_addition_subtraction :
  let a := base4ToNat [3, 2, 1]
  let b := base4ToNat [2, 0, 3]
  let c := base4ToNat [1, 1, 2]
  let result := base4ToNat [1, 0, 2, 1]
  (a + b) - c = result := by sorry

end NUMINAMATH_CALUDE_base4_addition_subtraction_l292_29223


namespace NUMINAMATH_CALUDE_cos_150_degrees_l292_29237

theorem cos_150_degrees : Real.cos (150 * π / 180) = -Real.sqrt 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_cos_150_degrees_l292_29237


namespace NUMINAMATH_CALUDE_car_speed_problem_l292_29265

theorem car_speed_problem (S : ℝ) : 
  (S * 1.3 + 10 = 205) → S = 150 := by
  sorry

end NUMINAMATH_CALUDE_car_speed_problem_l292_29265


namespace NUMINAMATH_CALUDE_exactly_three_primes_probability_l292_29222

-- Define a die as a type with 6 possible outcomes
def Die := Fin 6

-- Define a function to check if a number is prime (for a 6-sided die)
def isPrime (n : Die) : Bool :=
  n.val + 1 = 2 || n.val + 1 = 3 || n.val + 1 = 5

-- Define the probability of rolling a prime number on a single die
def probPrime : ℚ := 1/2

-- Define the number of dice
def numDice : ℕ := 6

-- Define the number of dice we want to show prime numbers
def targetPrimes : ℕ := 3

-- State the theorem
theorem exactly_three_primes_probability :
  (numDice.choose targetPrimes : ℚ) * probPrime^targetPrimes * (1 - probPrime)^(numDice - targetPrimes) = 5/16 := by
  sorry

end NUMINAMATH_CALUDE_exactly_three_primes_probability_l292_29222


namespace NUMINAMATH_CALUDE_dress_price_difference_l292_29229

/-- Given a dress with an original price that was discounted by 15% to $85, 
    and then increased by 25%, prove that the difference between the original 
    price and the final price is $6.25. -/
theorem dress_price_difference (original_price : ℝ) : 
  original_price * (1 - 0.15) = 85 →
  original_price - (85 * (1 + 0.25)) = -6.25 := by
sorry

end NUMINAMATH_CALUDE_dress_price_difference_l292_29229


namespace NUMINAMATH_CALUDE_power_three_mod_five_l292_29293

theorem power_three_mod_five : 3^244 % 5 = 1 := by
  sorry

end NUMINAMATH_CALUDE_power_three_mod_five_l292_29293


namespace NUMINAMATH_CALUDE_odd_m_triple_g_36_l292_29201

def g (n : ℤ) : ℤ := 
  if n % 2 = 1 then 2 * n + 3
  else if n % 3 = 0 then n / 3
  else n - 1

theorem odd_m_triple_g_36 (m : ℤ) (h_odd : m % 2 = 1) :
  g (g (g m)) = 36 → m = 54 :=
by
  sorry

end NUMINAMATH_CALUDE_odd_m_triple_g_36_l292_29201


namespace NUMINAMATH_CALUDE_unique_solution_l292_29259

def cross_product (v w : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  (v.2.1 * w.2.2 - v.2.2 * w.2.1, v.2.2 * w.1 - v.1 * w.2.2, v.1 * w.2.1 - v.2.1 * w.1)

theorem unique_solution (a b c d e f : ℝ) :
  cross_product (3, a, c) (6, b, d) = (0, 0, 0) ∧
  cross_product (4, b, f) (8, e, d) = (0, 0, 0) →
  (a, b, c, d, e, f) = (1, 2, 1, 2, 4, 1) :=
by sorry

end NUMINAMATH_CALUDE_unique_solution_l292_29259


namespace NUMINAMATH_CALUDE_negation_of_forall_geq_two_l292_29263

theorem negation_of_forall_geq_two :
  ¬(∀ x : ℝ, x > 0 → x + 1/x ≥ 2) ↔ ∃ x : ℝ, x > 0 ∧ x + 1/x < 2 := by sorry

end NUMINAMATH_CALUDE_negation_of_forall_geq_two_l292_29263


namespace NUMINAMATH_CALUDE_problem_solution_l292_29279

def f (x a : ℝ) : ℝ := |2*x - 1| + |x + a|

theorem problem_solution :
  (∀ x : ℝ, f x 1 ≥ 3 ↔ x ≥ 1 ∨ x ≤ -1) ∧
  ((∃ x : ℝ, f x a ≤ |a - 1|) → a ≤ 1/4) :=
by sorry

end NUMINAMATH_CALUDE_problem_solution_l292_29279


namespace NUMINAMATH_CALUDE_condition1_implies_bijective_condition2_implies_bijective_condition3_implies_not_injective_not_surjective_condition4_not_necessarily_injective_or_surjective_l292_29287

-- Define the function type
def RealFunction := ℝ → ℝ

-- Define the properties
def Injective (f : RealFunction) : Prop :=
  ∀ x y, f x = f y → x = y

def Surjective (f : RealFunction) : Prop :=
  ∀ y, ∃ x, f x = y

def Bijective (f : RealFunction) : Prop :=
  Injective f ∧ Surjective f

-- Theorem statements
theorem condition1_implies_bijective (f : RealFunction) 
  (h : ∀ x, f (f x - 1) = x + 1) : Bijective f := by sorry

theorem condition2_implies_bijective (f : RealFunction) 
  (h : ∀ x y, f (x + f y) = f x + y^5) : Bijective f := by sorry

theorem condition3_implies_not_injective_not_surjective (f : RealFunction) 
  (h : ∀ x, f (f x) = Real.sin x) : ¬(Injective f) ∧ ¬(Surjective f) := by sorry

theorem condition4_not_necessarily_injective_or_surjective : 
  ∃ f : RealFunction, (∀ x y, f (x + y^2) = f x * f y + x * f y - y^3 * f x) ∧ 
  ¬(Injective f) ∧ ¬(Surjective f) := by sorry

end NUMINAMATH_CALUDE_condition1_implies_bijective_condition2_implies_bijective_condition3_implies_not_injective_not_surjective_condition4_not_necessarily_injective_or_surjective_l292_29287


namespace NUMINAMATH_CALUDE_stone150_is_8_l292_29299

/-- Represents the circular arrangement of stones with the given counting pattern. -/
def StoneCircle := Fin 15

/-- The number of counts before the pattern repeats. -/
def patternLength : ℕ := 28

/-- Maps a count to its corresponding stone in the circle. -/
def countToStone (count : ℕ) : StoneCircle :=
  sorry

/-- The stone that is counted as 150. -/
def stone150 : StoneCircle :=
  countToStone 150

/-- The original stone number that corresponds to the 150th count. -/
theorem stone150_is_8 : stone150 = ⟨8, sorry⟩ :=
  sorry

end NUMINAMATH_CALUDE_stone150_is_8_l292_29299


namespace NUMINAMATH_CALUDE_parallel_vectors_sum_l292_29243

/-- Two vectors are parallel if one is a scalar multiple of the other -/
def parallel (a b : ℝ × ℝ) : Prop :=
  ∃ k : ℝ, a.1 * b.2 = k * a.2 * b.1

/-- Vector addition -/
def vec_add (a b : ℝ × ℝ) : ℝ × ℝ :=
  (a.1 + b.1, a.2 + b.2)

/-- Scalar multiplication of a vector -/
def vec_scalar_mul (k : ℝ) (v : ℝ × ℝ) : ℝ × ℝ :=
  (k * v.1, k * v.2)

theorem parallel_vectors_sum (y : ℝ) :
  let a : ℝ × ℝ := (1, 2)
  let b : ℝ × ℝ := (2, y)
  parallel a b → vec_add a (vec_scalar_mul 2 b) = (5, 10) := by
  sorry

end NUMINAMATH_CALUDE_parallel_vectors_sum_l292_29243


namespace NUMINAMATH_CALUDE_shortest_distance_on_cube_face_l292_29290

/-- The shortest distance on the surface of a cube between midpoints of opposite edges on the same face -/
theorem shortest_distance_on_cube_face (edge_length : ℝ) (h : edge_length = 2) :
  let midpoint_distance := Real.sqrt 2
  ∃ (path : ℝ), path ≥ midpoint_distance ∧
    (∀ (other_path : ℝ), other_path ≥ midpoint_distance → path ≤ other_path) :=
by sorry

end NUMINAMATH_CALUDE_shortest_distance_on_cube_face_l292_29290


namespace NUMINAMATH_CALUDE_triangle_equilateral_l292_29205

theorem triangle_equilateral (a b c : ℝ) (A B C : ℝ) :
  (a + b + c) * (b + c - a) = 3 * a * b * c →
  Real.sin A = 2 * Real.sin B * Real.cos C →
  a > 0 ∧ b > 0 ∧ c > 0 →
  A > 0 ∧ B > 0 ∧ C > 0 →
  A + B + C = Real.pi →
  a = b ∧ b = c ∧ A = B ∧ B = C ∧ A = Real.pi / 3 :=
sorry

end NUMINAMATH_CALUDE_triangle_equilateral_l292_29205


namespace NUMINAMATH_CALUDE_intersection_equals_result_l292_29227

-- Define the sets M and N
def M : Set ℝ := {x | (x - 3) * Real.sqrt (x - 1) ≥ 0}
def N : Set ℝ := {x | (x - 3) * (x - 1) ≥ 0}

-- Define the intersection of M and N
def M_intersect_N : Set ℝ := M ∩ N

-- Theorem statement
theorem intersection_equals_result : M_intersect_N = {x | x ≥ 3 ∨ x = 1} := by
  sorry

end NUMINAMATH_CALUDE_intersection_equals_result_l292_29227


namespace NUMINAMATH_CALUDE_triangle_side_length_l292_29273

theorem triangle_side_length (A B C : ℝ) (a b c : ℝ) :
  A = π / 3 →
  a = Real.sqrt 3 →
  b = 1 →
  (0 < A ∧ A < π) →
  (0 < B ∧ B < π) →
  (0 < C ∧ C < π) →
  A + B + C = π →
  a / Real.sin A = b / Real.sin B →
  a / Real.sin A = c / Real.sin C →
  c = 2 := by
sorry

end NUMINAMATH_CALUDE_triangle_side_length_l292_29273


namespace NUMINAMATH_CALUDE_polynomial_remainder_l292_29244

theorem polynomial_remainder (x : ℝ) : 
  (8 * x^3 - 20 * x^2 + 28 * x - 26) % (4 * x - 8) = 14 := by
sorry

end NUMINAMATH_CALUDE_polynomial_remainder_l292_29244


namespace NUMINAMATH_CALUDE_at_least_one_le_quarter_l292_29256

theorem at_least_one_le_quarter (a b c : Real) 
  (ha : 0 < a ∧ a < 1) 
  (hb : 0 < b ∧ b < 1) 
  (hc : 0 < c ∧ c < 1) : 
  (a * (1 - b) ≤ 1/4) ∨ (b * (1 - c) ≤ 1/4) ∨ (c * (1 - a) ≤ 1/4) := by
  sorry

end NUMINAMATH_CALUDE_at_least_one_le_quarter_l292_29256


namespace NUMINAMATH_CALUDE_min_cubes_in_prism_l292_29220

/-- Given a rectangular prism built with N identical 1-cm cubes,
    where 420 cubes are hidden from a viewpoint showing three faces,
    the minimum possible value of N is 630. -/
theorem min_cubes_in_prism (N : ℕ) (l m n : ℕ) : 
  (l - 1) * (m - 1) * (n - 1) = 420 →
  N = l * m * n →
  (∀ l' m' n' : ℕ, (l' - 1) * (m' - 1) * (n' - 1) = 420 → l' * m' * n' ≥ N) →
  N = 630 := by
  sorry

end NUMINAMATH_CALUDE_min_cubes_in_prism_l292_29220


namespace NUMINAMATH_CALUDE_value_of_x_minus_y_l292_29251

theorem value_of_x_minus_y (x y z : ℝ) 
  (eq1 : 3 * x - 5 * y = 5)
  (eq2 : x / (x + y) = 5 / 7)
  (eq3 : x + z * y = 10) :
  x - y = 3 := by
sorry

end NUMINAMATH_CALUDE_value_of_x_minus_y_l292_29251


namespace NUMINAMATH_CALUDE_max_distance_to_line_l292_29283

/-- The intersection point of two lines -/
def intersection_point (l1 l2 : ℝ × ℝ → Prop) : ℝ × ℝ :=
  sorry

/-- The distance between two points in ℝ² -/
def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  sorry

/-- A line in ℝ² represented by its equation -/
def line (a b c : ℝ) : ℝ × ℝ → Prop :=
  fun p => a * p.1 + b * p.2 + c = 0

theorem max_distance_to_line :
  let l1 := line 1 1 (-1)
  let l2 := line 1 (-2) (-4)
  let p := intersection_point l1 l2
  ∀ k : ℝ,
    let l3 := line k (-1) (1 + 2*k)
    ∀ q : ℝ × ℝ,
      l3 q →
      distance p q ≤ 2 * Real.sqrt 5 :=
by sorry

end NUMINAMATH_CALUDE_max_distance_to_line_l292_29283


namespace NUMINAMATH_CALUDE_seven_power_minus_two_power_l292_29296

theorem seven_power_minus_two_power : 
  ∀ x y : ℕ+, 7^(x.val) - 3 * 2^(y.val) = 1 ↔ (x = 1 ∧ y = 1) ∨ (x = 2 ∧ y = 4) :=
by sorry

end NUMINAMATH_CALUDE_seven_power_minus_two_power_l292_29296


namespace NUMINAMATH_CALUDE_x_squared_coefficient_in_expansion_l292_29207

/-- The coefficient of x² in the expansion of (2+x)(1-2x)^5 is 70 -/
theorem x_squared_coefficient_in_expansion : Int := by
  sorry

end NUMINAMATH_CALUDE_x_squared_coefficient_in_expansion_l292_29207


namespace NUMINAMATH_CALUDE_multiplication_addition_equality_l292_29250

theorem multiplication_addition_equality : 26 * 33 + 67 * 26 = 2600 := by sorry

end NUMINAMATH_CALUDE_multiplication_addition_equality_l292_29250


namespace NUMINAMATH_CALUDE_sum_six_odd_squares_not_2020_l292_29210

theorem sum_six_odd_squares_not_2020 : ¬ ∃ (a b c d e f : ℤ),
  (2 * a + 1)^2 + (2 * b + 1)^2 + (2 * c + 1)^2 + 
  (2 * d + 1)^2 + (2 * e + 1)^2 + (2 * f + 1)^2 = 2020 :=
by sorry

end NUMINAMATH_CALUDE_sum_six_odd_squares_not_2020_l292_29210


namespace NUMINAMATH_CALUDE_building_cost_l292_29200

/-- Calculates the total cost of all units in a building -/
def total_cost (total_units : ℕ) (cost_1bed : ℕ) (cost_2bed : ℕ) (num_2bed : ℕ) : ℕ :=
  let num_1bed := total_units - num_2bed
  num_1bed * cost_1bed + num_2bed * cost_2bed

/-- Proves that the total cost of all units in the given building is 4950 dollars -/
theorem building_cost : total_cost 12 360 450 7 = 4950 := by
  sorry

end NUMINAMATH_CALUDE_building_cost_l292_29200


namespace NUMINAMATH_CALUDE_folded_square_perimeter_ratio_l292_29230

theorem folded_square_perimeter_ratio :
  let square_side : ℝ := 10
  let folded_width : ℝ := square_side / 2
  let folded_height : ℝ := square_side
  let triangle_perimeter : ℝ := folded_width + folded_height + Real.sqrt (folded_width ^ 2 + folded_height ^ 2)
  let pentagon_perimeter : ℝ := 2 * folded_height + folded_width + Real.sqrt (folded_width ^ 2 + folded_height ^ 2) + folded_width
  triangle_perimeter / pentagon_perimeter = (3 + Real.sqrt 5) / (6 + Real.sqrt 5) := by
  sorry

end NUMINAMATH_CALUDE_folded_square_perimeter_ratio_l292_29230


namespace NUMINAMATH_CALUDE_last_colored_cell_position_l292_29267

/-- Represents a cell position in the grid -/
structure CellPosition where
  row : Nat
  col : Nat

/-- Represents the dimensions of the rectangle -/
structure RectangleDimensions where
  width : Nat
  height : Nat

/-- Represents the coloring process in a spiral pattern -/
def spiralColor (dim : RectangleDimensions) : CellPosition :=
  sorry

/-- Theorem: The last cell colored in a 200x100 rectangle with spiral coloring is at (51, 50) -/
theorem last_colored_cell_position :
  let dim : RectangleDimensions := ⟨200, 100⟩
  spiralColor dim = ⟨51, 50⟩ := by
  sorry

end NUMINAMATH_CALUDE_last_colored_cell_position_l292_29267


namespace NUMINAMATH_CALUDE_find_divisor_l292_29221

theorem find_divisor : 
  ∃ d : ℕ, d > 0 ∧ 136 = 9 * d + 1 :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_find_divisor_l292_29221


namespace NUMINAMATH_CALUDE_afternoon_rowers_l292_29214

theorem afternoon_rowers (morning evening total : ℕ) 
  (h1 : morning = 36)
  (h2 : evening = 49)
  (h3 : total = 98)
  : total - morning - evening = 13 := by
  sorry

end NUMINAMATH_CALUDE_afternoon_rowers_l292_29214


namespace NUMINAMATH_CALUDE_product_lcm_hcf_relation_l292_29278

theorem product_lcm_hcf_relation (a b : ℕ+) 
  (h_product : a * b = 571536)
  (h_lcm : Nat.lcm a b = 31096) :
  Nat.gcd a b = 18 := by
  sorry

end NUMINAMATH_CALUDE_product_lcm_hcf_relation_l292_29278


namespace NUMINAMATH_CALUDE_circle_area_from_circumference_l292_29202

theorem circle_area_from_circumference (C : ℝ) (r : ℝ) (h : C = 36 * Real.pi) :
  r * r * Real.pi = 324 * Real.pi :=
by
  sorry

end NUMINAMATH_CALUDE_circle_area_from_circumference_l292_29202


namespace NUMINAMATH_CALUDE_jansen_family_has_three_children_l292_29203

/-- Represents the Jansen family structure -/
structure JansenFamily where
  mother_age : ℝ
  father_age : ℝ
  grandfather_age : ℝ
  num_children : ℕ
  children_total_age : ℝ

/-- The Jansen family satisfies the given conditions -/
def is_valid_jansen_family (family : JansenFamily) : Prop :=
  family.father_age = 50 ∧
  family.grandfather_age = 70 ∧
  (family.mother_age + family.father_age + family.grandfather_age + family.children_total_age) / 
    (3 + family.num_children : ℝ) = 25 ∧
  (family.mother_age + family.grandfather_age + family.children_total_age) / 
    (2 + family.num_children : ℝ) = 20

/-- The number of children in a valid Jansen family is 3 -/
theorem jansen_family_has_three_children (family : JansenFamily) 
    (h : is_valid_jansen_family family) : family.num_children = 3 := by
  sorry

#check jansen_family_has_three_children

end NUMINAMATH_CALUDE_jansen_family_has_three_children_l292_29203


namespace NUMINAMATH_CALUDE_unique_valid_sequence_l292_29209

/-- Represents a sequence of positive integers satisfying the given conditions -/
def ValidSequence (a : Fin 5 → ℕ+) : Prop :=
  a 0 = 1 ∧
  (99 : ℚ) / 100 = (a 0 : ℚ) / a 1 + (a 1 : ℚ) / a 2 + (a 2 : ℚ) / a 3 + (a 3 : ℚ) / a 4 ∧
  ∀ k : Fin 3, ((a (k + 1) : ℕ) - 1) * (a (k - 1) : ℕ) ≥ (a k : ℕ)^2 * ((a k : ℕ) - 1)

/-- The theorem stating that there is only one valid sequence -/
theorem unique_valid_sequence :
  ∃! a : Fin 5 → ℕ+, ValidSequence a ∧
    a 0 = 1 ∧ a 1 = 2 ∧ a 2 = 5 ∧ a 3 = 56 ∧ a 4 = 25^2 * 56 := by
  sorry

end NUMINAMATH_CALUDE_unique_valid_sequence_l292_29209


namespace NUMINAMATH_CALUDE_bathroom_width_l292_29289

/-- A rectangular bathroom with given length and area has a specific width -/
theorem bathroom_width (length area : ℝ) (h1 : length = 4) (h2 : area = 8) :
  area / length = 2 := by
  sorry

end NUMINAMATH_CALUDE_bathroom_width_l292_29289


namespace NUMINAMATH_CALUDE_necessary_condition_for_A_l292_29212

-- Define the set A
def A : Set ℝ := {x | (x - 2) / (x + 1) ≤ 0}

-- State the theorem
theorem necessary_condition_for_A (a : ℝ) :
  (∀ x ∈ A, x ≥ a) → a ≤ -1 := by
  sorry

end NUMINAMATH_CALUDE_necessary_condition_for_A_l292_29212


namespace NUMINAMATH_CALUDE_book_reading_time_l292_29217

theorem book_reading_time (total_pages : ℕ) (rate1 rate2 : ℕ) (days1 days2 : ℕ) : 
  total_pages = 525 →
  rate1 = 25 →
  rate2 = 21 →
  days1 * rate1 = total_pages →
  days2 * rate2 = total_pages →
  (days1 = 21 ∧ days2 = 25) := by
  sorry

end NUMINAMATH_CALUDE_book_reading_time_l292_29217


namespace NUMINAMATH_CALUDE_cost_doubling_cost_percentage_increase_l292_29292

theorem cost_doubling (t b : ℝ) (t_pos : t > 0) (b_pos : b > 0) : 
  t * (2 * b)^4 = 16 * (t * b^4) := by
  sorry

theorem cost_percentage_increase (t b : ℝ) (t_pos : t > 0) (b_pos : b > 0) :
  (t * (2 * b)^4) / (t * b^4) * 100 = 1600 := by
  sorry

end NUMINAMATH_CALUDE_cost_doubling_cost_percentage_increase_l292_29292


namespace NUMINAMATH_CALUDE_city_mpg_is_32_l292_29219

/-- Represents the fuel efficiency of a car in different driving conditions -/
structure CarFuelEfficiency where
  highway_miles_per_tank : ℝ
  city_miles_per_tank : ℝ
  highway_city_mpg_difference : ℝ

/-- Calculates the city miles per gallon given the car's fuel efficiency data -/
def city_mpg (car : CarFuelEfficiency) : ℝ :=
  sorry

/-- Theorem stating that for the given car data, the city MPG is 32 -/
theorem city_mpg_is_32 (car : CarFuelEfficiency)
  (h1 : car.highway_miles_per_tank = 462)
  (h2 : car.city_miles_per_tank = 336)
  (h3 : car.highway_city_mpg_difference = 12) :
  city_mpg car = 32 := by
  sorry

end NUMINAMATH_CALUDE_city_mpg_is_32_l292_29219


namespace NUMINAMATH_CALUDE_min_value_of_expression_lower_bound_achievable_l292_29248

theorem min_value_of_expression (x : ℝ) (h : x > 1) :
  2*x + 7/(x-1) ≥ 2*Real.sqrt 14 + 2 :=
sorry

theorem lower_bound_achievable :
  ∃ x > 1, 2*x + 7/(x-1) = 2*Real.sqrt 14 + 2 :=
sorry

end NUMINAMATH_CALUDE_min_value_of_expression_lower_bound_achievable_l292_29248


namespace NUMINAMATH_CALUDE_equation_solution_l292_29298

theorem equation_solution (a b : ℝ) :
  (a + b - 1)^2 = a^2 + b^2 - 1 ↔ a = 1 ∨ b = 1 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l292_29298


namespace NUMINAMATH_CALUDE_total_tickets_sold_l292_29294

theorem total_tickets_sold
  (total_receipts : ℕ)
  (advance_ticket_cost : ℕ)
  (same_day_ticket_cost : ℕ)
  (advance_tickets_sold : ℕ)
  (h1 : total_receipts = 1600)
  (h2 : advance_ticket_cost = 20)
  (h3 : same_day_ticket_cost = 30)
  (h4 : advance_tickets_sold = 20)
  : ∃ (same_day_tickets_sold : ℕ),
    advance_ticket_cost * advance_tickets_sold +
    same_day_ticket_cost * same_day_tickets_sold = total_receipts ∧
    advance_tickets_sold + same_day_tickets_sold = 60 :=
by
  sorry

end NUMINAMATH_CALUDE_total_tickets_sold_l292_29294


namespace NUMINAMATH_CALUDE_sum_squared_equals_four_l292_29249

theorem sum_squared_equals_four (a b c : ℝ) 
  (h : a^2 + b^2 + c^2 - 2*a + 4*b - 6*c + 14 = 0) : 
  (a + b + c)^2 = 4 := by
  sorry

end NUMINAMATH_CALUDE_sum_squared_equals_four_l292_29249


namespace NUMINAMATH_CALUDE_midpoint_coordinate_sum_l292_29204

/-- The sum of the coordinates of the midpoint of a segment with endpoints (10, 7) and (4, -3) is 9 -/
theorem midpoint_coordinate_sum : 
  let p1 : ℝ × ℝ := (10, 7)
  let p2 : ℝ × ℝ := (4, -3)
  let midpoint : ℝ × ℝ := ((p1.1 + p2.1) / 2, (p1.2 + p2.2) / 2)
  midpoint.1 + midpoint.2 = 9 := by sorry

end NUMINAMATH_CALUDE_midpoint_coordinate_sum_l292_29204


namespace NUMINAMATH_CALUDE_square_of_binomial_constant_l292_29224

theorem square_of_binomial_constant (p : ℝ) : 
  (∃ a b : ℝ, ∀ x : ℝ, 16 * x^2 + 40 * x + p = (a * x + b)^2) → p = 25 := by
  sorry

end NUMINAMATH_CALUDE_square_of_binomial_constant_l292_29224


namespace NUMINAMATH_CALUDE_min_reciprocal_sum_l292_29232

theorem min_reciprocal_sum (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : 2*x + y = 1) :
  (1/x + 1/y) ≥ 3 + 2*Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_min_reciprocal_sum_l292_29232


namespace NUMINAMATH_CALUDE_same_solution_implies_c_equals_four_l292_29239

theorem same_solution_implies_c_equals_four :
  ∀ x c : ℝ,
  (3 * x + 9 = 0) →
  (c * x + 15 = 3) →
  c = 4 :=
by
  sorry

end NUMINAMATH_CALUDE_same_solution_implies_c_equals_four_l292_29239


namespace NUMINAMATH_CALUDE_expense_difference_zero_l292_29225

def vacation_expenses (anne_paid beth_paid carlos_paid : ℕ) (a b : ℕ) : Prop :=
  let total := anne_paid + beth_paid + carlos_paid
  let share := total / 3
  (anne_paid + b = share + a) ∧
  (beth_paid = share + b) ∧
  (carlos_paid + a = share)

theorem expense_difference_zero 
  (anne_paid beth_paid carlos_paid : ℕ) 
  (a b : ℕ) 
  (h : vacation_expenses anne_paid beth_paid carlos_paid a b) :
  a - b = 0 :=
sorry

end NUMINAMATH_CALUDE_expense_difference_zero_l292_29225


namespace NUMINAMATH_CALUDE_P_necessary_not_sufficient_for_Q_l292_29260

theorem P_necessary_not_sufficient_for_Q :
  (∀ x : ℝ, (x + 2) * (x - 1) < 0 → x < 1) ∧
  (∃ x : ℝ, x < 1 ∧ (x + 2) * (x - 1) ≥ 0) := by
  sorry

end NUMINAMATH_CALUDE_P_necessary_not_sufficient_for_Q_l292_29260


namespace NUMINAMATH_CALUDE_first_term_is_34_l292_29288

/-- Represents an arithmetic sequence with given properties -/
structure ArithmeticSequence where
  second_term : ℕ
  common_difference : ℕ
  last_term : ℕ

/-- Theorem: The first term of the specific arithmetic sequence is 34 -/
theorem first_term_is_34 (seq : ArithmeticSequence) 
  (h1 : seq.second_term = 45)
  (h2 : seq.common_difference = 11)
  (h3 : seq.last_term = 89) :
  seq.second_term - seq.common_difference = 34 := by
  sorry

end NUMINAMATH_CALUDE_first_term_is_34_l292_29288
