import Mathlib

namespace NUMINAMATH_CALUDE_sin_product_theorem_l2962_296233

theorem sin_product_theorem :
  Real.sin (10 * Real.pi / 180) *
  Real.sin (30 * Real.pi / 180) *
  Real.sin (50 * Real.pi / 180) *
  Real.sin (70 * Real.pi / 180) = 1 / 16 := by
  sorry

end NUMINAMATH_CALUDE_sin_product_theorem_l2962_296233


namespace NUMINAMATH_CALUDE_reciprocal_inequality_l2962_296266

theorem reciprocal_inequality (a b : ℝ) (h1 : a < b) (h2 : b < 0) : 1 / a > 1 / b := by
  sorry

end NUMINAMATH_CALUDE_reciprocal_inequality_l2962_296266


namespace NUMINAMATH_CALUDE_function_and_inequality_properties_l2962_296241

-- Define the function f
def f (x : ℝ) : ℝ := |x + 2| - |2*x - 1|

-- Define the theorem
theorem function_and_inequality_properties :
  ∀ (a b : ℝ), a ≠ 0 →
  (∀ (x m : ℝ), |b + 2*a| - |2*b - a| ≥ |a| * (|x + 1| + |x - m|)) →
  (∀ (x : ℝ), f x > -5 ↔ x ∈ Set.Ioo (-2) 8) ∧
  (∀ (m : ℝ), m ∈ Set.Icc (-7/2) (3/2)) :=
by sorry

end NUMINAMATH_CALUDE_function_and_inequality_properties_l2962_296241


namespace NUMINAMATH_CALUDE_probability_prime_sum_digits_l2962_296206

def ball_numbers : List Nat := [10, 11, 13, 14, 17, 19, 21, 23]

def sum_of_digits (n : Nat) : Nat :=
  n.repr.foldl (fun acc d => acc + d.toNat - 48) 0

def is_prime (n : Nat) : Bool :=
  n > 1 && (List.range (n - 1)).all (fun d => d <= 1 || n % (d + 2) ≠ 0)

theorem probability_prime_sum_digits :
  let favorable_outcomes := (ball_numbers.map sum_of_digits).filter is_prime |>.length
  let total_outcomes := ball_numbers.length
  (favorable_outcomes : ℚ) / total_outcomes = 1 / 2 := by sorry

end NUMINAMATH_CALUDE_probability_prime_sum_digits_l2962_296206


namespace NUMINAMATH_CALUDE_principal_booking_l2962_296282

/-- The number of rooms needed to accommodate a class on a field trip -/
def rooms_needed (total_students : ℕ) (students_per_room : ℕ) : ℕ :=
  (total_students + students_per_room - 1) / students_per_room

/-- Theorem: The principal needs to book 6 rooms for 30 students -/
theorem principal_booking : 
  let total_students : ℕ := 30
  let queen_bed_capacity : ℕ := 2
  let pullout_couch_capacity : ℕ := 1
  let room_capacity : ℕ := 2 * queen_bed_capacity + pullout_couch_capacity
  rooms_needed total_students room_capacity = 6 := by
sorry

end NUMINAMATH_CALUDE_principal_booking_l2962_296282


namespace NUMINAMATH_CALUDE_system_of_equations_solution_system_of_inequalities_solution_l2962_296212

-- System of equations
theorem system_of_equations_solution (x y : ℝ) :
  (2 * x + y = 32 ∧ 2 * x - y = 0) → (x = 8 ∧ y = 16) := by sorry

-- System of inequalities
theorem system_of_inequalities_solution (x : ℝ) :
  (3 * x - 1 < 5 - 2 * x ∧ 5 * x + 1 ≥ 2 * x + 3) →
  (2 / 3 ≤ x ∧ x < 6 / 5) := by sorry

end NUMINAMATH_CALUDE_system_of_equations_solution_system_of_inequalities_solution_l2962_296212


namespace NUMINAMATH_CALUDE_solve_for_h_l2962_296230

/-- The y-intercept of the first equation -/
def y_intercept1 : ℝ := 2025

/-- The y-intercept of the second equation -/
def y_intercept2 : ℝ := 2026

/-- The first equation -/
def equation1 (h j x y : ℝ) : Prop := y = 4 * (x - h)^2 + j

/-- The second equation -/
def equation2 (h k x y : ℝ) : Prop := y = x^3 - 3 * (x - h)^2 + k

/-- Positive integer x-intercepts for the first equation -/
def positive_integer_roots1 (h j : ℝ) : Prop :=
  ∃ (x1 x2 : ℕ), x1 ≠ 0 ∧ x2 ≠ 0 ∧ equation1 h j x1 0 ∧ equation1 h j x2 0

/-- Positive integer x-intercepts for the second equation -/
def positive_integer_roots2 (h k : ℝ) : Prop :=
  ∃ (x1 x2 : ℕ), x1 ≠ 0 ∧ x2 ≠ 0 ∧ equation2 h k x1 0 ∧ equation2 h k x2 0

/-- The main theorem -/
theorem solve_for_h :
  ∃ (h j k : ℝ),
    equation1 h j 0 y_intercept1 ∧
    equation2 h k 0 y_intercept2 ∧
    positive_integer_roots1 h j ∧
    positive_integer_roots2 h k ∧
    h = 45 := by sorry

end NUMINAMATH_CALUDE_solve_for_h_l2962_296230


namespace NUMINAMATH_CALUDE_button_to_magnet_ratio_l2962_296270

/-- Represents the number of earrings in a set -/
def earrings_per_set : ℕ := 2

/-- Represents the number of sets Rebecca wants to make -/
def sets : ℕ := 4

/-- Represents the total number of gemstones needed -/
def total_gemstones : ℕ := 24

/-- Represents the number of magnets used in each earring -/
def magnets_per_earring : ℕ := 2

/-- Represents the ratio of gemstones to buttons -/
def gemstone_to_button_ratio : ℕ := 3

/-- Theorem stating the ratio of buttons to magnets for each earring -/
theorem button_to_magnet_ratio :
  let total_earrings := sets * earrings_per_set
  let total_buttons := total_gemstones / gemstone_to_button_ratio
  let buttons_per_earring := total_buttons / total_earrings
  (buttons_per_earring : ℚ) / magnets_per_earring = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_button_to_magnet_ratio_l2962_296270


namespace NUMINAMATH_CALUDE_square_root_of_neg_five_squared_l2962_296289

theorem square_root_of_neg_five_squared : Real.sqrt ((-5)^2) = 5 ∨ Real.sqrt ((-5)^2) = -5 := by
  sorry

end NUMINAMATH_CALUDE_square_root_of_neg_five_squared_l2962_296289


namespace NUMINAMATH_CALUDE_total_books_l2962_296208

/-- The total number of books Sandy, Benny, and Tim have together is 67. -/
theorem total_books (sandy_books benny_books tim_books : ℕ) 
  (h1 : sandy_books = 10)
  (h2 : benny_books = 24)
  (h3 : tim_books = 33) :
  sandy_books + benny_books + tim_books = 67 := by
  sorry

end NUMINAMATH_CALUDE_total_books_l2962_296208


namespace NUMINAMATH_CALUDE_age_difference_l2962_296260

/-- Represents the ages of four people: Patrick, Michael, Monica, and Nathan. -/
structure Ages where
  patrick : ℝ
  michael : ℝ
  monica : ℝ
  nathan : ℝ

/-- The conditions of the problem -/
def satisfies_conditions (ages : Ages) : Prop :=
  ages.patrick / ages.michael = 3 / 5 ∧
  ages.michael / ages.monica = 3 / 5 ∧
  ages.monica / ages.nathan = 4 / 7 ∧
  ages.patrick + ages.michael + ages.monica + ages.nathan = 142

/-- The theorem stating the difference between Patrick's and Nathan's ages -/
theorem age_difference (ages : Ages) (h : satisfies_conditions ages) :
  ∃ ε > 0, |ages.patrick - ages.nathan - 1.46| < ε :=
sorry

end NUMINAMATH_CALUDE_age_difference_l2962_296260


namespace NUMINAMATH_CALUDE_complex_addition_simplification_l2962_296258

theorem complex_addition_simplification :
  (4 : ℂ) + 3*I + (-7 : ℂ) + 5*I = -3 + 8*I :=
by sorry

end NUMINAMATH_CALUDE_complex_addition_simplification_l2962_296258


namespace NUMINAMATH_CALUDE_smallest_n_congruence_l2962_296226

theorem smallest_n_congruence (n : ℕ) : 
  (∀ m : ℕ, 0 < m → m < 7 → (7^m : ℤ) % 5 ≠ m^7 % 5) ∧ 
  (7^7 : ℤ) % 5 = 7^7 % 5 := by
sorry

end NUMINAMATH_CALUDE_smallest_n_congruence_l2962_296226


namespace NUMINAMATH_CALUDE_geometric_sequence_properties_l2962_296248

-- Define a geometric sequence
def is_geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, q ≠ 0 ∧ ∀ n : ℕ, a (n + 1) = q * a n

-- Main theorem
theorem geometric_sequence_properties (a : ℕ → ℝ) (h : is_geometric_sequence a) :
  (is_geometric_sequence (fun n => |a n|)) ∧
  (is_geometric_sequence (fun n => a n * a (n + 1))) ∧
  (is_geometric_sequence (fun n => 1 / a n)) ∧
  ¬(∀ (a : ℕ → ℝ), is_geometric_sequence a → is_geometric_sequence (fun n => Real.log (a n ^ 2))) :=
by sorry

end NUMINAMATH_CALUDE_geometric_sequence_properties_l2962_296248


namespace NUMINAMATH_CALUDE_collinear_probability_5x4_l2962_296236

/-- Represents a rectangular array of dots. -/
structure DotArray :=
  (rows : ℕ)
  (cols : ℕ)

/-- The number of ways to choose k items from n items. -/
def choose (n k : ℕ) : ℕ := Nat.choose n k

/-- The number of collinear sets of 4 dots in a 5x4 array. -/
def collinearSets (arr : DotArray) : ℕ := arr.cols * choose arr.rows 4

/-- The total number of ways to choose 4 dots from the array. -/
def totalChoices (arr : DotArray) : ℕ := choose (arr.rows * arr.cols) 4

/-- The probability of choosing 4 collinear dots. -/
def collinearProbability (arr : DotArray) : ℚ :=
  collinearSets arr / totalChoices arr

/-- Theorem: The probability of choosing 4 collinear dots in a 5x4 array is 4/969. -/
theorem collinear_probability_5x4 :
  collinearProbability ⟨5, 4⟩ = 4 / 969 := by
  sorry

end NUMINAMATH_CALUDE_collinear_probability_5x4_l2962_296236


namespace NUMINAMATH_CALUDE_final_result_proof_l2962_296268

theorem final_result_proof (chosen_number : ℕ) (h : chosen_number = 740) : 
  (chosen_number / 4 : ℚ) - 175 = 10 := by
  sorry

end NUMINAMATH_CALUDE_final_result_proof_l2962_296268


namespace NUMINAMATH_CALUDE_equation_graph_is_two_lines_l2962_296255

-- Define the set of points satisfying the original equation
def S : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | (p.1 - p.2)^2 = 3 * p.1^2 + p.2^2}

-- Define the two lines
def L1 : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.1 = 0}
def L2 : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.2 = -p.1}

-- State the theorem
theorem equation_graph_is_two_lines : S = L1 ∪ L2 := by
  sorry

end NUMINAMATH_CALUDE_equation_graph_is_two_lines_l2962_296255


namespace NUMINAMATH_CALUDE_chromatic_number_of_our_graph_l2962_296253

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

/-- Predicate to check if three vertices form a triangle in the graph -/
def isTriangle (G : Graph) (a b c : Vertex) : Prop := sorry

/-- The chromatic number of a graph -/
def chromaticNumber (G : Graph) : ℕ := sorry

/-- Our specific graph -/
def ourGraph : Graph := sorry

theorem chromatic_number_of_our_graph :
  let G := ourGraph
  (degree G Vertex.x = 5) →
  (degree G Vertex.z = 4) →
  (degree G Vertex.y = 3) →
  isTriangle G Vertex.x Vertex.y Vertex.z →
  chromaticNumber G = 3 := by sorry

end NUMINAMATH_CALUDE_chromatic_number_of_our_graph_l2962_296253


namespace NUMINAMATH_CALUDE_friends_assignment_l2962_296286

/-- The number of ways to assign friends to rooms -/
def assignFriends (n : ℕ) (m : ℕ) (maxPerRoom : ℕ) : ℕ :=
  sorry

/-- Theorem stating the correct number of assignments for the given problem -/
theorem friends_assignment :
  assignFriends 7 7 3 = 15120 := by
  sorry

end NUMINAMATH_CALUDE_friends_assignment_l2962_296286


namespace NUMINAMATH_CALUDE_julia_pet_food_cost_l2962_296295

/-- The total amount Julia spent on food for her animals -/
def total_spent (weekly_total : ℕ) (rabbit_weeks : ℕ) (parrot_weeks : ℕ) (rabbit_cost : ℕ) : ℕ :=
  let parrot_cost := weekly_total - rabbit_cost
  rabbit_weeks * rabbit_cost + parrot_weeks * parrot_cost

/-- Theorem stating the total amount Julia spent on food for her animals -/
theorem julia_pet_food_cost :
  total_spent 30 5 3 12 = 114 := by
  sorry

end NUMINAMATH_CALUDE_julia_pet_food_cost_l2962_296295


namespace NUMINAMATH_CALUDE_family_weight_problem_l2962_296229

theorem family_weight_problem (total_weight daughter_weight : ℝ) 
  (h1 : total_weight = 150)
  (h2 : daughter_weight = 42) :
  ∃ (grandmother_weight child_weight : ℝ),
    grandmother_weight + daughter_weight + child_weight = total_weight ∧
    child_weight = (1 / 5) * grandmother_weight ∧
    daughter_weight + child_weight = 60 := by
  sorry

end NUMINAMATH_CALUDE_family_weight_problem_l2962_296229


namespace NUMINAMATH_CALUDE_gym_cost_is_650_l2962_296225

/-- Calculates the total cost of two gym memberships for a year -/
def total_gym_cost (cheap_monthly_fee : ℕ) (cheap_signup_fee : ℕ) : ℕ :=
  let expensive_monthly_fee := 3 * cheap_monthly_fee
  let expensive_signup_fee := 4 * expensive_monthly_fee
  let cheap_yearly_cost := 12 * cheap_monthly_fee + cheap_signup_fee
  let expensive_yearly_cost := 12 * expensive_monthly_fee + expensive_signup_fee
  cheap_yearly_cost + expensive_yearly_cost

/-- Proves that the total cost of two gym memberships for a year is $650 -/
theorem gym_cost_is_650 : total_gym_cost 10 50 = 650 := by
  sorry

end NUMINAMATH_CALUDE_gym_cost_is_650_l2962_296225


namespace NUMINAMATH_CALUDE_fraction_sum_product_equality_l2962_296252

theorem fraction_sum_product_equality (x y : ℤ) :
  (19 : ℚ) / x + (96 : ℚ) / y = ((19 : ℚ) / x) * ((96 : ℚ) / y) →
  ∃ m : ℤ, x = 19 * m ∧ y = 96 - 96 * m :=
by sorry

end NUMINAMATH_CALUDE_fraction_sum_product_equality_l2962_296252


namespace NUMINAMATH_CALUDE_last_three_digits_of_8_to_108_l2962_296280

theorem last_three_digits_of_8_to_108 : 8^108 ≡ 38 [ZMOD 1000] := by
  sorry

end NUMINAMATH_CALUDE_last_three_digits_of_8_to_108_l2962_296280


namespace NUMINAMATH_CALUDE_inequality_relationship_l2962_296254

theorem inequality_relationship (x : ℝ) :
  ¬(((x - 1) * (x + 3) < 0 → (x + 1) * (x - 3) < 0) ∧
    ((x + 1) * (x - 3) < 0 → (x - 1) * (x + 3) < 0)) :=
by sorry

end NUMINAMATH_CALUDE_inequality_relationship_l2962_296254


namespace NUMINAMATH_CALUDE_larger_number_is_448_l2962_296223

/-- Given two positive integers with specific HCF and LCM properties, prove the larger number is 448 -/
theorem larger_number_is_448 (a b : ℕ+) : 
  (Nat.gcd a b = 32) →
  (∃ (x y : ℕ+), x = 13 ∧ y = 14 ∧ Nat.lcm a b = 32 * x * y) →
  max a b = 448 := by
sorry

end NUMINAMATH_CALUDE_larger_number_is_448_l2962_296223


namespace NUMINAMATH_CALUDE_exactly_one_true_proposition_l2962_296203

-- Define the basic geometric concepts
def Line : Type := sorry
def Plane : Type := sorry

-- Define the geometric relationships
def skew (l1 l2 : Line) : Prop := sorry
def perpendicular (p1 p2 : Plane) : Prop := sorry
def parallel (p1 p2 : Plane) : Prop := sorry
def in_plane (l : Line) (p : Plane) : Prop := sorry
def oblique_to_plane (l : Line) (p : Plane) : Prop := sorry

-- Define the propositions
def prop1 : Prop := ∀ (p1 p2 : Plane) (l1 l2 : Line), 
  p1 ≠ p2 → in_plane l1 p1 → in_plane l2 p2 → skew l1 l2

def prop2 : Prop := ∀ (p1 p2 : Plane) (l : Line), 
  oblique_to_plane l p1 → 
  (perpendicular p1 p2 ∧ in_plane l p2) → 
  ∀ (p3 : Plane), perpendicular p1 p3 ∧ in_plane l p3 → p2 = p3

def prop3 : Prop := ∀ (p1 p2 p3 : Plane), 
  perpendicular p1 p2 → perpendicular p1 p3 → parallel p2 p3

-- Theorem statement
theorem exactly_one_true_proposition : 
  (prop1 = False ∧ prop2 = True ∧ prop3 = False) := by sorry

end NUMINAMATH_CALUDE_exactly_one_true_proposition_l2962_296203


namespace NUMINAMATH_CALUDE_line_parameterization_l2962_296240

/-- Given a line y = 2x - 30 parameterized by (x,y) = (g(t), 12t - 10), 
    prove that g(t) = 6t + 10 -/
theorem line_parameterization (g : ℝ → ℝ) : 
  (∀ t : ℝ, 12 * t - 10 = 2 * g t - 30) → 
  (∀ t : ℝ, g t = 6 * t + 10) := by
sorry

end NUMINAMATH_CALUDE_line_parameterization_l2962_296240


namespace NUMINAMATH_CALUDE_sqrt_equation_solution_l2962_296243

theorem sqrt_equation_solution (y : ℝ) : 
  Real.sqrt (2 + Real.sqrt (3 * y - 4)) = Real.sqrt 9 → y = 53 / 3 := by
sorry

end NUMINAMATH_CALUDE_sqrt_equation_solution_l2962_296243


namespace NUMINAMATH_CALUDE_quadratic_inequality_range_l2962_296204

theorem quadratic_inequality_range (k : ℝ) : 
  (∀ x : ℝ, k * x^2 + k * x - 3/4 < 0) ↔ k ∈ Set.Ioc (-3) 0 := by
sorry

end NUMINAMATH_CALUDE_quadratic_inequality_range_l2962_296204


namespace NUMINAMATH_CALUDE_double_side_halves_energy_l2962_296211

/-- Represents the energy stored between two point charges -/
structure EnergyBetweenCharges where
  distance : ℝ
  charge1 : ℝ
  charge2 : ℝ
  energy : ℝ

/-- Represents a configuration of three point charges in an equilateral triangle -/
structure TriangleConfiguration where
  sideLength : ℝ
  charge : ℝ
  totalEnergy : ℝ

/-- The relation between energy, distance, and charges -/
axiom energy_proportionality 
  (e1 e2 : EnergyBetweenCharges) : 
  e1.charge1 = e2.charge1 → e1.charge2 = e2.charge2 → 
  e1.energy * e1.distance = e2.energy * e2.distance

/-- The total energy in a triangle configuration is the sum of energies between pairs -/
axiom triangle_energy 
  (tc : TriangleConfiguration) (e : EnergyBetweenCharges) :
  e.distance = tc.sideLength → e.charge1 = tc.charge → e.charge2 = tc.charge →
  tc.totalEnergy = 3 * e.energy

/-- Theorem: Doubling the side length of the triangle halves the total energy -/
theorem double_side_halves_energy 
  (tc1 tc2 : TriangleConfiguration) :
  tc1.charge = tc2.charge →
  tc2.sideLength = 2 * tc1.sideLength →
  tc2.totalEnergy = tc1.totalEnergy / 2 := by
  sorry

end NUMINAMATH_CALUDE_double_side_halves_energy_l2962_296211


namespace NUMINAMATH_CALUDE_provisions_problem_l2962_296215

/-- The number of days the provisions last for the initial group -/
def initial_days : ℝ := 12

/-- The number of additional men joining the group -/
def additional_men : ℕ := 300

/-- The number of days the provisions last after the additional men join -/
def new_days : ℝ := 9.662337662337663

/-- The initial number of men in the group -/
def initial_men : ℕ := 1240

theorem provisions_problem :
  ∃ (M : ℝ), 
    (M ≥ 0) ∧ 
    (abs (M - initial_men) < 1) ∧
    (M * initial_days = (M + additional_men) * new_days) :=
by sorry

end NUMINAMATH_CALUDE_provisions_problem_l2962_296215


namespace NUMINAMATH_CALUDE_inverse_variation_problem_l2962_296201

-- Define the relationship between x and y
def inverse_variation (x y : ℝ) (k : ℝ) : Prop := x * y^3 = k

-- Theorem statement
theorem inverse_variation_problem (x₁ x₂ y₁ y₂ k : ℝ) 
  (h1 : inverse_variation x₁ y₁ k)
  (h2 : x₁ = 8)
  (h3 : y₁ = 1)
  (h4 : y₂ = 2)
  (h5 : inverse_variation x₂ y₂ k) :
  x₂ = 1 := by
sorry

end NUMINAMATH_CALUDE_inverse_variation_problem_l2962_296201


namespace NUMINAMATH_CALUDE_divisibility_of_fraction_l2962_296259

theorem divisibility_of_fraction (a b n : ℕ) (h1 : a ≠ b) (h2 : n ∣ (a^n - b^n)) :
  n ∣ ((a^n - b^n) / (a - b)) :=
sorry

end NUMINAMATH_CALUDE_divisibility_of_fraction_l2962_296259


namespace NUMINAMATH_CALUDE_marcus_has_more_cards_l2962_296224

/-- The number of baseball cards Marcus has -/
def marcus_cards : ℕ := 210

/-- The number of baseball cards Carter has -/
def carter_cards : ℕ := 152

/-- The difference in baseball cards between Marcus and Carter -/
def card_difference : ℕ := marcus_cards - carter_cards

theorem marcus_has_more_cards : card_difference = 58 := by
  sorry

end NUMINAMATH_CALUDE_marcus_has_more_cards_l2962_296224


namespace NUMINAMATH_CALUDE_negation_of_squared_plus_one_geq_one_l2962_296217

theorem negation_of_squared_plus_one_geq_one :
  (¬ ∀ x : ℝ, x^2 + 1 ≥ 1) ↔ (∃ x : ℝ, x^2 + 1 < 1) := by sorry

end NUMINAMATH_CALUDE_negation_of_squared_plus_one_geq_one_l2962_296217


namespace NUMINAMATH_CALUDE_chord_equation_l2962_296222

/-- The equation of the line on which the chord common to two circles lies -/
theorem chord_equation (r : ℝ) (ρ θ : ℝ) (h : r > 0) :
  (ρ = r ∨ ρ = -2 * r * Real.sin (θ + π/4)) →
  Real.sqrt 2 * ρ * (Real.sin θ + Real.cos θ) = -r :=
sorry

end NUMINAMATH_CALUDE_chord_equation_l2962_296222


namespace NUMINAMATH_CALUDE_remainder_2519_div_3_l2962_296249

theorem remainder_2519_div_3 : 2519 % 3 = 2 := by
  sorry

end NUMINAMATH_CALUDE_remainder_2519_div_3_l2962_296249


namespace NUMINAMATH_CALUDE_new_person_weight_l2962_296231

theorem new_person_weight (initial_count : ℕ) (weight_increase : ℝ) (replaced_weight : ℝ) :
  initial_count = 8 →
  weight_increase = 3.5 →
  replaced_weight = 62 →
  initial_count * weight_increase + replaced_weight = 90 :=
by
  sorry

end NUMINAMATH_CALUDE_new_person_weight_l2962_296231


namespace NUMINAMATH_CALUDE_min_value_of_f_l2962_296292

-- Define the function f(x)
def f (x : ℝ) : ℝ := x^2 - 4*x + 4

-- Theorem stating that the minimum value of f(x) is 0
theorem min_value_of_f :
  ∃ (x₀ : ℝ), ∀ (x : ℝ), f x ≥ f x₀ ∧ f x₀ = 0 :=
by sorry

end NUMINAMATH_CALUDE_min_value_of_f_l2962_296292


namespace NUMINAMATH_CALUDE_sum_of_first_12_terms_of_arithmetic_sequence_l2962_296234

/-- Given the sum of the first 4 terms and the sum of the first 8 terms of an arithmetic sequence,
    this theorem proves that the sum of the first 12 terms is 210. -/
theorem sum_of_first_12_terms_of_arithmetic_sequence 
  (S₄ S₈ : ℕ) (h₁ : S₄ = 30) (h₂ : S₈ = 100) : ∃ S₁₂ : ℕ, S₁₂ = 210 :=
by
  sorry


end NUMINAMATH_CALUDE_sum_of_first_12_terms_of_arithmetic_sequence_l2962_296234


namespace NUMINAMATH_CALUDE_parker_dumbbell_weight_l2962_296275

/-- Given an initial setup of dumbbells and additional dumbbells added, 
    calculate the total weight Parker is using for his exercises. -/
theorem parker_dumbbell_weight 
  (initial_count : ℕ) 
  (additional_count : ℕ) 
  (weight_per_dumbbell : ℕ) : 
  initial_count = 4 → 
  additional_count = 2 → 
  weight_per_dumbbell = 20 → 
  (initial_count + additional_count) * weight_per_dumbbell = 120 := by
  sorry

end NUMINAMATH_CALUDE_parker_dumbbell_weight_l2962_296275


namespace NUMINAMATH_CALUDE_mary_max_earnings_l2962_296242

/-- Calculates the maximum weekly earnings for a worker under specific pay conditions -/
def maxWeeklyEarnings (maxHours : ℕ) (regularRate : ℚ) (overtime1Multiplier : ℚ) (overtime2Multiplier : ℚ) : ℚ :=
  let regularHours := min maxHours 40
  let overtime1Hours := min (maxHours - regularHours) 20
  let overtime2Hours := maxHours - regularHours - overtime1Hours
  let regularPay := regularRate * regularHours
  let overtime1Pay := regularRate * overtime1Multiplier * overtime1Hours
  let overtime2Pay := regularRate * overtime2Multiplier * overtime2Hours
  regularPay + overtime1Pay + overtime2Pay

theorem mary_max_earnings :
  maxWeeklyEarnings 80 15 1.6 2 = 1680 := by
  sorry

#eval maxWeeklyEarnings 80 15 1.6 2

end NUMINAMATH_CALUDE_mary_max_earnings_l2962_296242


namespace NUMINAMATH_CALUDE_pop_albums_count_l2962_296227

def country_albums : ℕ := 2
def songs_per_album : ℕ := 6
def total_songs : ℕ := 30

theorem pop_albums_count : 
  ∃ (pop_albums : ℕ), 
    country_albums * songs_per_album + pop_albums * songs_per_album = total_songs ∧ 
    pop_albums = 3 := by
  sorry

end NUMINAMATH_CALUDE_pop_albums_count_l2962_296227


namespace NUMINAMATH_CALUDE_distance_between_circumcenters_l2962_296277

-- Define a triangle ABC
structure Triangle :=
  (A B C : ℝ × ℝ)

-- Define the side lengths of the triangle
def side_lengths (t : Triangle) : ℝ × ℝ × ℝ :=
  (dist t.A t.B, dist t.B t.C, dist t.C t.A)

-- Define the orthocenter of a triangle
def orthocenter (t : Triangle) : ℝ × ℝ :=
  sorry

-- Define the circumcenter of a triangle
def circumcenter (t : Triangle) : ℝ × ℝ :=
  sorry

-- Theorem statement
theorem distance_between_circumcenters (t : Triangle) :
  let H := orthocenter t
  let side_len := side_lengths t
  side_len.1 = 13 ∧ side_len.2.1 = 14 ∧ side_len.2.2 = 15 →
  dist (circumcenter ⟨t.A, H, t.B⟩) (circumcenter ⟨t.A, H, t.C⟩) = 14 :=
sorry

end NUMINAMATH_CALUDE_distance_between_circumcenters_l2962_296277


namespace NUMINAMATH_CALUDE_probability_a_squared_geq_4b_l2962_296261

-- Define the set of numbers
def S : Set Nat := {1, 2, 3, 4}

-- Define the condition
def condition (a b : Nat) : Prop := a^2 ≥ 4*b

-- Define the total number of ways to select two numbers
def total_selections : Nat := 12

-- Define the number of favorable selections
def favorable_selections : Nat := 6

-- State the theorem
theorem probability_a_squared_geq_4b :
  (favorable_selections : ℚ) / total_selections = 1 / 2 := by sorry

end NUMINAMATH_CALUDE_probability_a_squared_geq_4b_l2962_296261


namespace NUMINAMATH_CALUDE_detergent_quarts_in_altered_solution_l2962_296274

/-- Represents the ratio of bleach : detergent : water in a cleaning solution -/
structure CleaningSolution :=
  (bleach : ℚ)
  (detergent : ℚ)
  (water : ℚ)

/-- Calculates the amount of detergent in quarts given the conditions of the problem -/
def calculate_detergent_quarts (original : CleaningSolution) (water_gallons : ℚ) : ℚ :=
  let new_ratio := CleaningSolution.mk 
    (original.bleach * 3) 
    original.detergent
    (original.water / 2)
  let total_parts := new_ratio.bleach + new_ratio.detergent + new_ratio.water
  let detergent_gallons := (new_ratio.detergent / new_ratio.water) * water_gallons
  detergent_gallons * 4

/-- Theorem stating that the altered solution will contain 160 quarts of detergent -/
theorem detergent_quarts_in_altered_solution :
  let original := CleaningSolution.mk 2 25 100
  calculate_detergent_quarts original 80 = 160 := by
  sorry


end NUMINAMATH_CALUDE_detergent_quarts_in_altered_solution_l2962_296274


namespace NUMINAMATH_CALUDE_work_done_by_resistive_force_l2962_296288

def mass : Real := 0.01
def initial_velocity : Real := 400
def final_velocity : Real := 100

def kinetic_energy (m : Real) (v : Real) : Real :=
  0.5 * m * v^2

def work_done (m : Real) (v1 : Real) (v2 : Real) : Real :=
  kinetic_energy m v1 - kinetic_energy m v2

theorem work_done_by_resistive_force :
  work_done mass initial_velocity final_velocity = 750 := by
  sorry

end NUMINAMATH_CALUDE_work_done_by_resistive_force_l2962_296288


namespace NUMINAMATH_CALUDE_coeff_x_cube_p_l2962_296218

-- Define the polynomial
def p (x : ℝ) := x^2 - 3*x + 3

-- Define the coefficient of x in the expansion of p(x)^3
def coeff_x (p : ℝ → ℝ) : ℝ :=
  (p 1 - p 0) - (p (-1) - p 0)

-- Theorem statement
theorem coeff_x_cube_p : coeff_x (fun x ↦ (p x)^3) = -81 := by
  sorry

end NUMINAMATH_CALUDE_coeff_x_cube_p_l2962_296218


namespace NUMINAMATH_CALUDE_count_multiples_of_seven_between_squares_l2962_296246

theorem count_multiples_of_seven_between_squares : 
  ∃! (s : Finset ℕ), 
    (∀ n ∈ s, n % 7 = 0 ∧ (17 : ℝ) < Real.sqrt n ∧ Real.sqrt n < 18) ∧
    (∀ n : ℕ, n % 7 = 0 ∧ (17 : ℝ) < Real.sqrt n ∧ Real.sqrt n < 18 → n ∈ s) ∧
    Finset.card s = 5 := by
  sorry

end NUMINAMATH_CALUDE_count_multiples_of_seven_between_squares_l2962_296246


namespace NUMINAMATH_CALUDE_binomial_sum_problem_l2962_296210

theorem binomial_sum_problem (a : ℚ) (a₀ a₁ a₂ a₃ a₄ a₅ a₆ a₇ a₈ a₉ : ℚ) :
  (∀ x, (a*x + 1)^5 * (x + 2)^4 = a₀*(x + 2)^9 + a₁*(x + 2)^8 + a₂*(x + 2)^7 + 
                                   a₃*(x + 2)^6 + a₄*(x + 2)^5 + a₅*(x + 2)^4 + 
                                   a₆*(x + 2)^3 + a₇*(x + 2)^2 + a₈*(x + 2) + a₉) →
  a₀ + a₁ + a₂ + a₃ + a₄ + a₅ + a₆ + a₇ + a₈ + a₉ = 1024 →
  a₀ + a₂ + a₄ + a₆ + a₈ = (2^10 - 14^5) / 2 := by
sorry

end NUMINAMATH_CALUDE_binomial_sum_problem_l2962_296210


namespace NUMINAMATH_CALUDE_no_real_solutions_for_abs_equation_l2962_296216

theorem no_real_solutions_for_abs_equation : 
  ¬ ∃ x : ℝ, |x^2 - 3| = 2*x + 6 := by
sorry

end NUMINAMATH_CALUDE_no_real_solutions_for_abs_equation_l2962_296216


namespace NUMINAMATH_CALUDE_friends_money_distribution_l2962_296267

theorem friends_money_distribution (x : ℚ) :
  x > 0 →
  let total := 6*x + 5*x + 4*x + 7*x + 0
  let pete_received := x + x + x + x
  pete_received / total = 2 / 11 := by
sorry

end NUMINAMATH_CALUDE_friends_money_distribution_l2962_296267


namespace NUMINAMATH_CALUDE_midpoint_sum_l2962_296264

/-- The sum of the coordinates of the midpoint of a line segment with endpoints (3, 4) and (10, 20) is 18.5 -/
theorem midpoint_sum : 
  let x₁ : ℝ := 3
  let y₁ : ℝ := 4
  let x₂ : ℝ := 10
  let y₂ : ℝ := 20
  let midpoint_x := (x₁ + x₂) / 2
  let midpoint_y := (y₁ + y₂) / 2
  midpoint_x + midpoint_y = 18.5 := by
sorry

end NUMINAMATH_CALUDE_midpoint_sum_l2962_296264


namespace NUMINAMATH_CALUDE_statement3_is_analogous_reasoning_l2962_296256

-- Define the concept of a geometric figure
structure GeometricFigure where
  name : String

-- Define the concept of a property for geometric figures
structure Property where
  description : String

-- Define the concept of reasoning
inductive Reasoning
| Analogous
| Inductive
| Deductive

-- Define the statement about equilateral triangles
def equilateralTriangleProperty : Property :=
  { description := "The sum of distances from a point inside to its sides is constant" }

-- Define the statement about regular tetrahedrons
def regularTetrahedronProperty : Property :=
  { description := "The sum of distances from a point inside to its faces is constant" }

-- Define the reasoning process in statement ③
def statement3 (equilateralTriangle regularTetrahedron : GeometricFigure)
               (equilateralProp tetrahedronProp : Property) : Prop :=
  (equilateralProp = equilateralTriangleProperty) →
  (tetrahedronProp = regularTetrahedronProperty) →
  ∃ (r : Reasoning), r = Reasoning.Analogous

-- Theorem statement
theorem statement3_is_analogous_reasoning 
  (equilateralTriangle regularTetrahedron : GeometricFigure)
  (equilateralProp tetrahedronProp : Property) :
  statement3 equilateralTriangle regularTetrahedron equilateralProp tetrahedronProp :=
by
  sorry

end NUMINAMATH_CALUDE_statement3_is_analogous_reasoning_l2962_296256


namespace NUMINAMATH_CALUDE_bike_ride_problem_l2962_296207

theorem bike_ride_problem (total_distance : ℝ) (total_time : ℝ) (speed_good : ℝ) (speed_tired : ℝ) :
  total_distance = 122 →
  total_time = 8 →
  speed_good = 20 →
  speed_tired = 12 →
  ∃ time_feeling_good : ℝ,
    time_feeling_good * speed_good + (total_time - time_feeling_good) * speed_tired = total_distance ∧
    time_feeling_good = 13 / 4 :=
by sorry

end NUMINAMATH_CALUDE_bike_ride_problem_l2962_296207


namespace NUMINAMATH_CALUDE_sector_area_l2962_296247

/-- Given a circular sector where the arc length is 4 cm and the central angle is 2 radians,
    prove that the area of the sector is 4 cm². -/
theorem sector_area (s : ℝ) (θ : ℝ) (A : ℝ) : 
  s = 4 → θ = 2 → s = 2 * θ → A = (1/2) * (s/θ)^2 * θ → A = 4 := by
  sorry

end NUMINAMATH_CALUDE_sector_area_l2962_296247


namespace NUMINAMATH_CALUDE_exponent_division_l2962_296269

theorem exponent_division (a : ℝ) : a^8 / a^2 = a^6 := by
  sorry

end NUMINAMATH_CALUDE_exponent_division_l2962_296269


namespace NUMINAMATH_CALUDE_task_completion_proof_l2962_296257

def task_completion (x : ℝ) : Prop :=
  let a := x
  let b := x + 6
  let c := x + 9
  (3 / a + 4 / b = 9 / c) ∧ (a = 18) ∧ (b = 24) ∧ (c = 27)

theorem task_completion_proof : ∃ x : ℝ, task_completion x := by
  sorry

end NUMINAMATH_CALUDE_task_completion_proof_l2962_296257


namespace NUMINAMATH_CALUDE_intersection_M_N_l2962_296285

-- Define the sets M and N
def M : Set ℝ := {x | x < 3}
def N : Set ℝ := {x | x^2 - 6*x + 8 < 0}

-- State the theorem
theorem intersection_M_N : M ∩ N = {x | 2 < x ∧ x < 3} := by sorry

end NUMINAMATH_CALUDE_intersection_M_N_l2962_296285


namespace NUMINAMATH_CALUDE_fraction_simplification_l2962_296221

theorem fraction_simplification :
  (1722^2 - 1715^2) / (1731^2 - 1708^2) = (7 * 3437) / (23 * 3439) := by
  sorry

end NUMINAMATH_CALUDE_fraction_simplification_l2962_296221


namespace NUMINAMATH_CALUDE_sum_of_satisfying_numbers_is_34_l2962_296238

def satisfies_condition (n : ℕ) : Prop :=
  1.5 * (n : ℝ) - 5.5 > 4.5

def sum_of_satisfying_numbers : ℕ :=
  (Finset.range 4).sum (fun i => i + 7)

theorem sum_of_satisfying_numbers_is_34 :
  sum_of_satisfying_numbers = 34 ∧
  ∀ n, 7 ≤ n → n ≤ 10 → satisfies_condition n :=
sorry

end NUMINAMATH_CALUDE_sum_of_satisfying_numbers_is_34_l2962_296238


namespace NUMINAMATH_CALUDE_cars_per_row_in_section_G_l2962_296271

/-- The number of rows in Section G -/
def section_G_rows : ℕ := 15

/-- The number of rows in Section H -/
def section_H_rows : ℕ := 20

/-- The number of cars per row in Section H -/
def section_H_cars_per_row : ℕ := 9

/-- The number of cars Nate can walk past per minute -/
def cars_per_minute : ℕ := 11

/-- The number of minutes Nate spent searching -/
def search_time : ℕ := 30

/-- The number of cars per row in Section G -/
def section_G_cars_per_row : ℕ := 10

theorem cars_per_row_in_section_G :
  section_G_cars_per_row = 
    (cars_per_minute * search_time - section_H_rows * section_H_cars_per_row) / section_G_rows :=
by sorry

end NUMINAMATH_CALUDE_cars_per_row_in_section_G_l2962_296271


namespace NUMINAMATH_CALUDE_inequality_proof_l2962_296213

theorem inequality_proof (m n : ℝ) (hm : m > 0) (hn : n > 0) (hne : m ≠ n) :
  (m - n) / (Real.log m - Real.log n) < (m + n) / 2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l2962_296213


namespace NUMINAMATH_CALUDE_living_room_walls_count_l2962_296263

/-- The number of walls in Eric's living room -/
def living_room_walls : ℕ := 7

/-- The time Eric spent removing wallpaper from one wall in the dining room (in hours) -/
def time_per_wall : ℕ := 2

/-- The total time it will take Eric to remove wallpaper from the living room (in hours) -/
def total_time : ℕ := 14

/-- Theorem stating that the number of walls in Eric's living room is 7 -/
theorem living_room_walls_count :
  living_room_walls = total_time / time_per_wall :=
by sorry

end NUMINAMATH_CALUDE_living_room_walls_count_l2962_296263


namespace NUMINAMATH_CALUDE_negation_of_universal_proposition_l2962_296283

theorem negation_of_universal_proposition :
  (¬ ∀ x : ℝ, x^2 - x + 3 > 0) ↔ (∃ x : ℝ, x^2 - x + 3 ≤ 0) := by
  sorry

end NUMINAMATH_CALUDE_negation_of_universal_proposition_l2962_296283


namespace NUMINAMATH_CALUDE_correct_participants_cars_needed_rental_plans_and_min_cost_l2962_296262

/-- Represents the number of teachers -/
def teachers : ℕ := 6

/-- Represents the number of students -/
def students : ℕ := 234

/-- Represents the total number of participants -/
def total_participants : ℕ := teachers + students

/-- Represents the capacity of bus A -/
def bus_A_capacity : ℕ := 45

/-- Represents the capacity of bus B -/
def bus_B_capacity : ℕ := 30

/-- Represents the rental cost of bus A -/
def bus_A_cost : ℕ := 400

/-- Represents the rental cost of bus B -/
def bus_B_cost : ℕ := 280

/-- Represents the total rental cost limit -/
def total_cost_limit : ℕ := 2300

/-- Theorem stating the correctness of the number of teachers and students -/
theorem correct_participants : teachers = 6 ∧ students = 234 ∧
  38 * teachers + 6 = students ∧ 40 * teachers - 6 = students := by sorry

/-- Theorem stating the number of cars needed -/
theorem cars_needed : ∃ (n : ℕ), n = 6 ∧ 
  n * bus_A_capacity ≥ total_participants ∧
  n ≥ teachers := by sorry

/-- Theorem stating the number of rental car plans and minimum cost -/
theorem rental_plans_and_min_cost : 
  ∃ (plans : ℕ) (min_cost : ℕ), plans = 2 ∧ min_cost = 2160 ∧
  ∀ (x : ℕ), 4 ≤ x ∧ x ≤ 5 →
    x * bus_A_capacity + (6 - x) * bus_B_capacity ≥ total_participants ∧
    x * bus_A_cost + (6 - x) * bus_B_cost ≤ total_cost_limit ∧
    (x = 4 → x * bus_A_cost + (6 - x) * bus_B_cost = min_cost) := by sorry

end NUMINAMATH_CALUDE_correct_participants_cars_needed_rental_plans_and_min_cost_l2962_296262


namespace NUMINAMATH_CALUDE_arctan_equation_solution_l2962_296273

theorem arctan_equation_solution :
  ∃ x : ℚ, 2 * Real.arctan (1/3) + 4 * Real.arctan (1/5) + Real.arctan (1/x) = π/4 ∧ x = -978/2029 := by
  sorry

end NUMINAMATH_CALUDE_arctan_equation_solution_l2962_296273


namespace NUMINAMATH_CALUDE_perfect_square_trinomial_m_value_l2962_296245

/-- A trinomial ax^2 + bx + c is a perfect square if there exists a real number k such that
    ax^2 + bx + c = (x + k)^2 for all x. -/
def IsPerfectSquareTrinomial (a b c : ℝ) : Prop :=
  ∃ k : ℝ, ∀ x : ℝ, a * x^2 + b * x + c = (x + k)^2

theorem perfect_square_trinomial_m_value :
  ∀ m : ℝ, IsPerfectSquareTrinomial 1 m 9 → m = 6 ∨ m = -6 :=
by sorry

end NUMINAMATH_CALUDE_perfect_square_trinomial_m_value_l2962_296245


namespace NUMINAMATH_CALUDE_complex_exponential_product_l2962_296232

theorem complex_exponential_product (α β : ℝ) :
  Complex.exp (Complex.I * α) + Complex.exp (Complex.I * β) = -1/3 + 4/5 * Complex.I →
  (Complex.exp (-Complex.I * α) + Complex.exp (-Complex.I * β)) *
  (Complex.exp (Complex.I * α) + Complex.exp (Complex.I * β)) = 169/225 := by
  sorry

end NUMINAMATH_CALUDE_complex_exponential_product_l2962_296232


namespace NUMINAMATH_CALUDE_trapezoid_area_l2962_296281

theorem trapezoid_area (large_square_side : ℝ) (small_square_side : ℝ) :
  large_square_side = 4 →
  small_square_side = 1 →
  let large_square_area := large_square_side ^ 2
  let small_square_area := small_square_side ^ 2
  let total_trapezoid_area := large_square_area - small_square_area
  let num_trapezoids := 4
  (total_trapezoid_area / num_trapezoids : ℝ) = 15 / 4 := by
  sorry

end NUMINAMATH_CALUDE_trapezoid_area_l2962_296281


namespace NUMINAMATH_CALUDE_unique_solution_exponential_system_l2962_296250

theorem unique_solution_exponential_system :
  ∀ x y z : ℝ, x > 0 ∧ y > 0 ∧ z > 0 →
  (x^y = z ∧ y^z = x ∧ z^x = y) →
  x = 1 ∧ y = 1 ∧ z = 1 := by
sorry

end NUMINAMATH_CALUDE_unique_solution_exponential_system_l2962_296250


namespace NUMINAMATH_CALUDE_kaleb_books_l2962_296200

theorem kaleb_books (initial_books sold_books new_books : ℕ) :
  initial_books = 34 →
  sold_books = 17 →
  new_books = 7 →
  initial_books - sold_books + new_books = 24 :=
by sorry

end NUMINAMATH_CALUDE_kaleb_books_l2962_296200


namespace NUMINAMATH_CALUDE_cyclic_sum_divisibility_l2962_296298

theorem cyclic_sum_divisibility (x y z : ℤ) (hxy : x ≠ y) (hyz : y ≠ z) (hzx : z ≠ x) :
  ∃ k : ℤ, (x - y)^5 + (y - z)^5 + (z - x)^5 = k * (5 * (x - y) * (y - z) * (z - x)) := by
  sorry

end NUMINAMATH_CALUDE_cyclic_sum_divisibility_l2962_296298


namespace NUMINAMATH_CALUDE_mikes_muffins_l2962_296272

/-- The number of muffins in a dozen -/
def dozen : ℕ := 12

/-- The number of boxes Mike needs to pack all his muffins -/
def boxes : ℕ := 8

/-- Mike's muffins theorem -/
theorem mikes_muffins : dozen * boxes = 96 := by
  sorry

end NUMINAMATH_CALUDE_mikes_muffins_l2962_296272


namespace NUMINAMATH_CALUDE_parallel_vectors_magnitude_l2962_296251

/-- Given two parallel vectors a and b, prove that the magnitude of b is √13 -/
theorem parallel_vectors_magnitude (x : ℝ) :
  let a : Fin 2 → ℝ := ![(-4), 6]
  let b : Fin 2 → ℝ := ![2, x]
  (∃ (k : ℝ), ∀ i, b i = k * a i) →  -- Parallel vectors condition
  Real.sqrt ((b 0) ^ 2 + (b 1) ^ 2) = Real.sqrt 13 := by
  sorry

end NUMINAMATH_CALUDE_parallel_vectors_magnitude_l2962_296251


namespace NUMINAMATH_CALUDE_fraction_of_half_is_one_seventh_l2962_296244

theorem fraction_of_half_is_one_seventh : (1 : ℚ) / 7 / ((1 : ℚ) / 2) = 2 / 7 := by
  sorry

end NUMINAMATH_CALUDE_fraction_of_half_is_one_seventh_l2962_296244


namespace NUMINAMATH_CALUDE_cone_volume_l2962_296296

/-- Given a cone with slant height 13 cm and height 12 cm, its volume is 100π cubic centimeters -/
theorem cone_volume (s h r : ℝ) (hs : s = 13) (hh : h = 12) 
  (hpythag : s^2 = h^2 + r^2) : (1/3 : ℝ) * π * r^2 * h = 100 * π := by
  sorry

end NUMINAMATH_CALUDE_cone_volume_l2962_296296


namespace NUMINAMATH_CALUDE_compound_hydrogen_count_l2962_296287

/-- Represents the number of atoms of each element in the compound -/
structure Compound where
  carbon : ℕ
  hydrogen : ℕ
  oxygen : ℕ

/-- Calculates the molecular weight of a compound given atomic weights -/
def molecularWeight (c : Compound) (carbonWeight oxygenWeight hydrogenWeight : ℕ) : ℕ :=
  c.carbon * carbonWeight + c.hydrogen * hydrogenWeight + c.oxygen * oxygenWeight

theorem compound_hydrogen_count :
  ∀ (c : Compound),
    c.carbon = 3 →
    c.oxygen = 1 →
    molecularWeight c 12 16 1 = 58 →
    c.hydrogen = 6 :=
by sorry

end NUMINAMATH_CALUDE_compound_hydrogen_count_l2962_296287


namespace NUMINAMATH_CALUDE_jimmy_sandwiches_l2962_296235

/-- The number of sandwiches Jimmy can make given the number of bread packs,
    slices per pack, and slices needed per sandwich. -/
def sandwiches_made (bread_packs : ℕ) (slices_per_pack : ℕ) (slices_per_sandwich : ℕ) : ℕ :=
  (bread_packs * slices_per_pack) / slices_per_sandwich

/-- Theorem stating that Jimmy made 8 sandwiches under the given conditions. -/
theorem jimmy_sandwiches :
  sandwiches_made 4 4 2 = 8 := by
  sorry

end NUMINAMATH_CALUDE_jimmy_sandwiches_l2962_296235


namespace NUMINAMATH_CALUDE_largest_three_digit_multiple_of_8_with_digit_sum_24_l2962_296237

def is_three_digit (n : ℕ) : Prop := 100 ≤ n ∧ n ≤ 999

def digit_sum (n : ℕ) : ℕ :=
  (n / 100) + ((n / 10) % 10) + (n % 10)

theorem largest_three_digit_multiple_of_8_with_digit_sum_24 :
  ∀ n : ℕ, is_three_digit n → n % 8 = 0 → digit_sum n = 24 → n ≤ 888 :=
by sorry

end NUMINAMATH_CALUDE_largest_three_digit_multiple_of_8_with_digit_sum_24_l2962_296237


namespace NUMINAMATH_CALUDE_system_solution_l2962_296297

-- Define the system of equations
def system (x y : ℝ) : Prop :=
  x + y = 4 ∧ 2 * x - y = 2

-- State the theorem
theorem system_solution :
  ∃! (x y : ℝ), system x y ∧ x = 2 ∧ y = 2 := by
  sorry

end NUMINAMATH_CALUDE_system_solution_l2962_296297


namespace NUMINAMATH_CALUDE_total_toys_is_160_l2962_296205

/-- The number of toys Kamari has -/
def kamari_toys : ℕ := 65

/-- The number of additional toys Anais has compared to Kamari -/
def anais_extra_toys : ℕ := 30

/-- The total number of toys Anais and Kamari have together -/
def total_toys : ℕ := kamari_toys + (kamari_toys + anais_extra_toys)

/-- Theorem stating that the total number of toys is 160 -/
theorem total_toys_is_160 : total_toys = 160 := by
  sorry

end NUMINAMATH_CALUDE_total_toys_is_160_l2962_296205


namespace NUMINAMATH_CALUDE_q_must_be_false_l2962_296291

theorem q_must_be_false (h1 : ¬(p ∧ q)) (h2 : ¬¬p) : ¬q := by
  sorry

end NUMINAMATH_CALUDE_q_must_be_false_l2962_296291


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_set_l2962_296265

theorem quadratic_inequality_solution_set 
  (a b c : ℝ) 
  (h : Set.Ioo (-1/3 : ℝ) 2 = {x | a * x^2 + b * x + c > 0}) :
  Set.Ioo (-3 : ℝ) (1/2) = {x | c * x^2 + b * x + a < 0} := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_set_l2962_296265


namespace NUMINAMATH_CALUDE_rhombus_perimeter_l2962_296214

/-- The perimeter of a rhombus with given diagonals -/
theorem rhombus_perimeter (d1 d2 : ℝ) (h1 : d1 = 10) (h2 : d2 = 24) : 
  4 * Real.sqrt ((d1/2)^2 + (d2/2)^2) = 52 := by
  sorry

#check rhombus_perimeter

end NUMINAMATH_CALUDE_rhombus_perimeter_l2962_296214


namespace NUMINAMATH_CALUDE_probability_of_three_ones_l2962_296284

def probability_of_sum_three (n : ℕ) (sides : ℕ) (target_sum : ℕ) : ℚ :=
  if n = 3 ∧ sides = 6 ∧ target_sum = 3 then 1 / 216 else 0

theorem probability_of_three_ones :
  probability_of_sum_three 3 6 3 = 1 / 216 := by
  sorry

end NUMINAMATH_CALUDE_probability_of_three_ones_l2962_296284


namespace NUMINAMATH_CALUDE_max_distance_from_origin_l2962_296202

def center : ℝ × ℝ := (5, -2)
def rope_length : ℝ := 12

theorem max_distance_from_origin :
  let max_dist := rope_length + Real.sqrt ((center.1 ^ 2) + (center.2 ^ 2))
  ∀ p : ℝ × ℝ, Real.sqrt ((p.1 - center.1) ^ 2 + (p.2 - center.2) ^ 2) ≤ rope_length →
    Real.sqrt (p.1 ^ 2 + p.2 ^ 2) ≤ max_dist :=
by sorry

end NUMINAMATH_CALUDE_max_distance_from_origin_l2962_296202


namespace NUMINAMATH_CALUDE_cryptarithmetic_puzzle_l2962_296293

theorem cryptarithmetic_puzzle (D E F G : ℕ) : 
  (∀ (X Y : ℕ), (X = D ∨ X = E ∨ X = F ∨ X = G) ∧ (Y = D ∨ Y = E ∨ Y = F ∨ Y = G) ∧ X ≠ Y → X ≠ Y) →
  F - E = D - 1 →
  D + E + F = 16 →
  F - E = D →
  G = F - E →
  G = 5 := by
sorry

end NUMINAMATH_CALUDE_cryptarithmetic_puzzle_l2962_296293


namespace NUMINAMATH_CALUDE_remainder_of_Q_mod_1000_l2962_296278

theorem remainder_of_Q_mod_1000 :
  (202^1 + 20^21 + 2^21) % 1000 = 354 := by
  sorry

end NUMINAMATH_CALUDE_remainder_of_Q_mod_1000_l2962_296278


namespace NUMINAMATH_CALUDE_reflect_x_of_P_l2962_296294

/-- Represents a point in the Cartesian coordinate system -/
structure Point where
  x : ℝ
  y : ℝ

/-- Reflects a point across the x-axis -/
def reflect_x (p : Point) : Point :=
  { x := p.x, y := -p.y }

/-- The theorem stating that reflecting the point P(-2, √5) across the x-axis 
    results in the point (-2, -√5) -/
theorem reflect_x_of_P : 
  let P : Point := { x := -2, y := Real.sqrt 5 }
  reflect_x P = { x := -2, y := -Real.sqrt 5 } := by
  sorry

end NUMINAMATH_CALUDE_reflect_x_of_P_l2962_296294


namespace NUMINAMATH_CALUDE_even_odd_sum_difference_l2962_296220

/-- Sum of first n positive even integers -/
def sum_even (n : ℕ) : ℕ := 2 * n * (n + 1)

/-- Sum of first n positive odd integers -/
def sum_odd (n : ℕ) : ℕ := n * n

/-- The positive difference between the sum of the first 30 positive even integers
    and the sum of the first 30 positive odd integers is 30 -/
theorem even_odd_sum_difference : sum_even 30 - sum_odd 30 = 30 := by
  sorry

end NUMINAMATH_CALUDE_even_odd_sum_difference_l2962_296220


namespace NUMINAMATH_CALUDE_smallest_number_of_sweets_l2962_296279

theorem smallest_number_of_sweets (x : ℕ) : 
  x > 0 ∧ 
  x % 6 = 5 ∧ 
  x % 8 = 7 ∧ 
  x % 9 = 8 ∧ 
  (∀ y : ℕ, y > 0 → y % 6 = 5 → y % 8 = 7 → y % 9 = 8 → x ≤ y) → 
  x = 71 := by
sorry

end NUMINAMATH_CALUDE_smallest_number_of_sweets_l2962_296279


namespace NUMINAMATH_CALUDE_decimal_difference_proof_l2962_296219

/-- The repeating decimal 0.727272... --/
def repeating_decimal : ℚ := 8 / 11

/-- The terminating decimal 0.72 --/
def terminating_decimal : ℚ := 72 / 100

/-- The difference between the repeating decimal and the terminating decimal --/
def decimal_difference : ℚ := repeating_decimal - terminating_decimal

theorem decimal_difference_proof : decimal_difference = 2 / 275 := by
  sorry

end NUMINAMATH_CALUDE_decimal_difference_proof_l2962_296219


namespace NUMINAMATH_CALUDE_intersection_of_M_and_N_l2962_296209

def M : Set ℕ := {0, 1, 2}
def N : Set ℕ := {0, 3, 4}

theorem intersection_of_M_and_N :
  M ∩ N = {0} := by sorry

end NUMINAMATH_CALUDE_intersection_of_M_and_N_l2962_296209


namespace NUMINAMATH_CALUDE_inequality_proof_l2962_296299

theorem inequality_proof (a b : ℝ) (ha : a ≥ 0) (hb : b ≥ 0) :
  (1/2) * (a + b)^2 + (1/4) * (a + b) ≥ a * Real.sqrt b + b * Real.sqrt a :=
by sorry

end NUMINAMATH_CALUDE_inequality_proof_l2962_296299


namespace NUMINAMATH_CALUDE_rationalized_factor_simplify_fraction_special_sqrt_l2962_296276

-- Part 1
theorem rationalized_factor (x : ℝ) : 
  (3 + Real.sqrt 11) * (3 - Real.sqrt 11) = -2 :=
sorry

-- Part 2
theorem simplify_fraction (b : ℝ) (h1 : b ≥ 0) (h2 : b ≠ 1) : 
  (1 - b) / (1 - Real.sqrt b) = 1 + Real.sqrt b :=
sorry

-- Part 3
theorem special_sqrt (a b : ℝ) 
  (ha : a = 1 / (Real.sqrt 3 - 2)) 
  (hb : b = 1 / (Real.sqrt 3 + 2)) : 
  Real.sqrt (a^2 + b^2 + 2) = 4 :=
sorry

end NUMINAMATH_CALUDE_rationalized_factor_simplify_fraction_special_sqrt_l2962_296276


namespace NUMINAMATH_CALUDE_square_root_of_four_l2962_296239

theorem square_root_of_four : 
  {x : ℝ | x ^ 2 = 4} = {2, -2} := by sorry

end NUMINAMATH_CALUDE_square_root_of_four_l2962_296239


namespace NUMINAMATH_CALUDE_cow_count_is_seven_l2962_296290

/-- Represents the number of animals in the group -/
structure AnimalCount where
  cows : ℕ
  chickens : ℕ

/-- The total number of legs in the group -/
def totalLegs (ac : AnimalCount) : ℕ := 4 * ac.cows + 2 * ac.chickens

/-- The total number of heads in the group -/
def totalHeads (ac : AnimalCount) : ℕ := ac.cows + ac.chickens

/-- The main theorem stating that if the number of legs is 14 more than twice the number of heads,
    then the number of cows is 7 -/
theorem cow_count_is_seven (ac : AnimalCount) :
  totalLegs ac = 2 * totalHeads ac + 14 → ac.cows = 7 := by
  sorry


end NUMINAMATH_CALUDE_cow_count_is_seven_l2962_296290


namespace NUMINAMATH_CALUDE_ab_value_l2962_296228

-- Define the base 10 logarithm
noncomputable def log10 (x : ℝ) := Real.log x / Real.log 10

-- Define the main theorem
theorem ab_value (a b : ℝ) (ha : 0 < a) (hb : 0 < b) :
  (∃ (w x y z : ℕ), w > 0 ∧ x > 0 ∧ y > 0 ∧ z > 0 ∧
    (w : ℝ) = (log10 a) ^ (1/3 : ℝ) ∧
    (x : ℝ) = (log10 b) ^ (1/3 : ℝ) ∧
    (y : ℝ) = log10 (a ^ (1/3 : ℝ)) ∧
    (z : ℝ) = log10 (b ^ (1/3 : ℝ)) ∧
    w + x + y + z = 12) →
  a * b = 10^9 :=
by sorry

end NUMINAMATH_CALUDE_ab_value_l2962_296228
