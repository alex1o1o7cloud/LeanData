import Mathlib

namespace NUMINAMATH_CALUDE_magnitude_z_equals_magnitude_iz_l2290_229003

theorem magnitude_z_equals_magnitude_iz (z : ℂ) : Complex.abs z = Complex.abs (Complex.I * z) := by
  sorry

end NUMINAMATH_CALUDE_magnitude_z_equals_magnitude_iz_l2290_229003


namespace NUMINAMATH_CALUDE_gecko_cost_is_fifteen_l2290_229028

/-- Represents the cost of feeding Harry's pets -/
structure PetFeedingCost where
  geckos : ℕ
  iguanas : ℕ
  snakes : ℕ
  snake_cost : ℕ
  iguana_cost : ℕ
  total_annual_cost : ℕ

/-- Calculates the monthly cost per gecko -/
def gecko_monthly_cost (p : PetFeedingCost) : ℚ :=
  (p.total_annual_cost / 12 - (p.snakes * p.snake_cost + p.iguanas * p.iguana_cost)) / p.geckos

/-- Theorem stating that the monthly cost per gecko is $15 -/
theorem gecko_cost_is_fifteen (p : PetFeedingCost) 
    (h1 : p.geckos = 3)
    (h2 : p.iguanas = 2)
    (h3 : p.snakes = 4)
    (h4 : p.snake_cost = 10)
    (h5 : p.iguana_cost = 5)
    (h6 : p.total_annual_cost = 1140) :
    gecko_monthly_cost p = 15 := by
  sorry

end NUMINAMATH_CALUDE_gecko_cost_is_fifteen_l2290_229028


namespace NUMINAMATH_CALUDE_det_A_eq_cube_l2290_229094

/-- The matrix A as defined in the problem -/
def A (x y z : ℝ) : Matrix (Fin 3) (Fin 3) ℝ :=
  !![1 + x^2 - y^2 - z^2, 2*(x*y + z), 2*(z*x - y);
    2*(x*y - z), 1 + y^2 - z^2 - x^2, 2*(y*z + x);
    2*(z*x + y), 2*(y*z - x), 1 + z^2 - x^2 - y^2]

/-- The theorem stating that the determinant of A is equal to (1 + x^2 + y^2 + z^2)^3 -/
theorem det_A_eq_cube (x y z : ℝ) : 
  Matrix.det (A x y z) = (1 + x^2 + y^2 + z^2)^3 := by
  sorry

end NUMINAMATH_CALUDE_det_A_eq_cube_l2290_229094


namespace NUMINAMATH_CALUDE_arithmetic_geometric_sequence_ratio_l2290_229075

theorem arithmetic_geometric_sequence_ratio 
  (a : ℕ → ℝ) 
  (d : ℝ) 
  (h1 : d ≠ 0)
  (h2 : ∀ n, a (n + 1) = a n + d)  -- arithmetic sequence definition
  (h3 : (a 5)^2 = a 1 * a 17)  -- geometric sequence property
  : (a 5) / (a 1) = 3 :=
by sorry

end NUMINAMATH_CALUDE_arithmetic_geometric_sequence_ratio_l2290_229075


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_l2290_229088

theorem sufficient_not_necessary (x : ℝ) :
  ((x + 1) * (x - 2) > 0 → abs x ≥ 1) ∧
  ¬(abs x ≥ 1 → (x + 1) * (x - 2) > 0) :=
by sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_l2290_229088


namespace NUMINAMATH_CALUDE_parabola_line_intersections_l2290_229025

theorem parabola_line_intersections (a b c : ℝ) (ha : a ≠ 0) :
  (∀ (x y : ℝ), (y = a * x^2 + b * x + c ∧ y = a * x + b) → 
    (∃! p : ℝ × ℝ, p.1 = x ∧ p.2 = y)) ∧
  (∀ (x y : ℝ), (y = a * x^2 + b * x + c ∧ y = b * x + c) → 
    (∃! p : ℝ × ℝ, p.1 = x ∧ p.2 = y)) ∧
  (∀ (x y : ℝ), (y = a * x^2 + b * x + c ∧ y = c * x + a) → 
    (∃! p : ℝ × ℝ, p.1 = x ∧ p.2 = y)) ∧
  (∀ (x y : ℝ), (y = a * x^2 + b * x + c ∧ y = b * x + a) → 
    (∃! p : ℝ × ℝ, p.1 = x ∧ p.2 = y)) ∧
  (∀ (x y : ℝ), (y = a * x^2 + b * x + c ∧ y = c * x + b) → 
    (∃! p : ℝ × ℝ, p.1 = x ∧ p.2 = y)) ∧
  (∀ (x y : ℝ), (y = a * x^2 + b * x + c ∧ y = a * x + c) → 
    (∃! p : ℝ × ℝ, p.1 = x ∧ p.2 = y)) →
  1 ≤ c / a ∧ c / a ≤ 5 := by
sorry

end NUMINAMATH_CALUDE_parabola_line_intersections_l2290_229025


namespace NUMINAMATH_CALUDE_line_intercepts_sum_l2290_229039

theorem line_intercepts_sum (x y : ℝ) : 
  x / 3 - y / 4 = 1 → x + y = -1 := by
  sorry

end NUMINAMATH_CALUDE_line_intercepts_sum_l2290_229039


namespace NUMINAMATH_CALUDE_factorization_2m_squared_minus_18_l2290_229000

theorem factorization_2m_squared_minus_18 (m : ℝ) : 2 * m^2 - 18 = 2 * (m + 3) * (m - 3) := by
  sorry

end NUMINAMATH_CALUDE_factorization_2m_squared_minus_18_l2290_229000


namespace NUMINAMATH_CALUDE_number_of_c_animals_l2290_229030

/-- Given the number of (A) and (B) animals, and the relationship between (A), (B), and (C) animals,
    prove that the number of (C) animals is 5. -/
theorem number_of_c_animals (a b : ℕ) (h1 : a = 45) (h2 : b = 32) 
    (h3 : b + c = a - 8) : c = 5 :=
by sorry

end NUMINAMATH_CALUDE_number_of_c_animals_l2290_229030


namespace NUMINAMATH_CALUDE_infinitely_many_primes_divide_fib_l2290_229010

-- Define the Fibonacci sequence
def fib : ℕ → ℕ
  | 0 => 0
  | 1 => 1
  | (n + 2) => fib (n + 1) + fib n

-- State the theorem
theorem infinitely_many_primes_divide_fib : 
  ∃ (S : Set ℕ), (Set.Infinite S) ∧ (∀ p ∈ S, Prime p ∧ (fib (p - 1) % p = 0)) :=
sorry

end NUMINAMATH_CALUDE_infinitely_many_primes_divide_fib_l2290_229010


namespace NUMINAMATH_CALUDE_isosceles_trapezoid_rotation_l2290_229015

/-- Represents an isosceles trapezoid -/
structure IsoscelesTrapezoid where
  -- Add necessary fields for an isosceles trapezoid
  longerBase : ℝ
  shorterBase : ℝ
  height : ℝ
  -- Add necessary conditions for an isosceles trapezoid
  longerBase_gt_shorterBase : longerBase > shorterBase
  bases_positive : longerBase > 0 ∧ shorterBase > 0
  height_positive : height > 0

/-- Represents a solid of revolution -/
inductive SolidOfRevolution
  | Cylinder
  | Cone
  | FrustumOfCone

/-- The result of rotating an isosceles trapezoid around its longer base -/
def rotateIsoscelesTrapezoid (t : IsoscelesTrapezoid) : List SolidOfRevolution :=
  [SolidOfRevolution.Cylinder, SolidOfRevolution.Cone, SolidOfRevolution.Cone]

theorem isosceles_trapezoid_rotation (t : IsoscelesTrapezoid) :
  rotateIsoscelesTrapezoid t = [SolidOfRevolution.Cylinder, SolidOfRevolution.Cone, SolidOfRevolution.Cone] := by
  sorry

end NUMINAMATH_CALUDE_isosceles_trapezoid_rotation_l2290_229015


namespace NUMINAMATH_CALUDE_average_age_combined_l2290_229093

/-- The average age of a group of fifth-graders, parents, and teachers -/
theorem average_age_combined (num_fifth_graders : ℕ) (num_parents : ℕ) (num_teachers : ℕ)
  (avg_age_fifth_graders : ℚ) (avg_age_parents : ℚ) (avg_age_teachers : ℚ)
  (h1 : num_fifth_graders = 40)
  (h2 : num_parents = 60)
  (h3 : num_teachers = 10)
  (h4 : avg_age_fifth_graders = 10)
  (h5 : avg_age_parents = 35)
  (h6 : avg_age_teachers = 45) :
  (num_fifth_graders * avg_age_fifth_graders +
   num_parents * avg_age_parents +
   num_teachers * avg_age_teachers) /
  (num_fifth_graders + num_parents + num_teachers : ℚ) = 295 / 11 := by
  sorry

end NUMINAMATH_CALUDE_average_age_combined_l2290_229093


namespace NUMINAMATH_CALUDE_smallest_weight_set_has_11_weights_l2290_229064

/-- A set of weights that can be divided into equal piles -/
structure WeightSet where
  weights : List ℕ
  divisible_by_4 : ∃ (n : ℕ), 4 * n = weights.sum
  divisible_by_5 : ∃ (n : ℕ), 5 * n = weights.sum
  divisible_by_6 : ∃ (n : ℕ), 6 * n = weights.sum

/-- The property of being the smallest set of weights divisible by 4, 5, and 6 -/
def is_smallest_weight_set (ws : WeightSet) : Prop :=
  ∀ (other : WeightSet), other.weights.length ≥ ws.weights.length

/-- The theorem stating that 11 is the smallest number of weights divisible by 4, 5, and 6 -/
theorem smallest_weight_set_has_11_weights :
  ∃ (ws : WeightSet), ws.weights.length = 11 ∧ is_smallest_weight_set ws :=
sorry

end NUMINAMATH_CALUDE_smallest_weight_set_has_11_weights_l2290_229064


namespace NUMINAMATH_CALUDE_odd_integer_pairs_theorem_l2290_229069

def phi : ℕ → ℕ := sorry  -- Euler's totient function

theorem odd_integer_pairs_theorem (a b : ℕ) (ha : Odd a) (hb : Odd b) (ha_gt_1 : a > 1) (hb_gt_1 : b > 1) :
  7 * (phi a)^2 - phi (a * b) + 11 * (phi b)^2 = 2 * (a^2 + b^2) →
  ∃ x : ℕ, a = 15 * 3^x ∧ b = 3 * 3^x :=
sorry

end NUMINAMATH_CALUDE_odd_integer_pairs_theorem_l2290_229069


namespace NUMINAMATH_CALUDE_product_of_polynomials_l2290_229050

theorem product_of_polynomials (p q : ℝ) : 
  (∀ k : ℝ, (5 * k^2 - 2 * k + p) * (4 * k^2 + q * k - 6) = 20 * k^4 - 18 * k^3 - 31 * k^2 + 12 * k + 18) →
  p + q = -3 := by
sorry

end NUMINAMATH_CALUDE_product_of_polynomials_l2290_229050


namespace NUMINAMATH_CALUDE_water_transfer_problem_l2290_229076

theorem water_transfer_problem (left_initial right_initial : ℕ) 
  (difference_after_transfer : ℕ) (h1 : left_initial = 2800) 
  (h2 : right_initial = 1500) (h3 : difference_after_transfer = 360) :
  ∃ (x : ℕ), x = 470 ∧ 
  left_initial - x = right_initial + x + difference_after_transfer :=
sorry

end NUMINAMATH_CALUDE_water_transfer_problem_l2290_229076


namespace NUMINAMATH_CALUDE_number_of_proper_subsets_of_A_l2290_229086

-- Define the universal set U
def U : Finset Nat := {0, 1, 2, 3}

-- Define set A based on its complement in U
def A : Finset Nat := U \ {2}

-- Theorem statement
theorem number_of_proper_subsets_of_A :
  (Finset.powerset A).card - 1 = 7 := by
  sorry

end NUMINAMATH_CALUDE_number_of_proper_subsets_of_A_l2290_229086


namespace NUMINAMATH_CALUDE_common_roots_product_l2290_229041

-- Define the cubic equations
def cubic1 (x C : ℝ) : ℝ := x^3 + 3*x^2 + C*x + 15
def cubic2 (x D : ℝ) : ℝ := x^3 + D*x^2 + 70

-- Define the condition of having two common roots
def has_two_common_roots (C D : ℝ) : Prop :=
  ∃ (p q : ℝ), p ≠ q ∧ cubic1 p C = 0 ∧ cubic1 q C = 0 ∧ cubic2 p D = 0 ∧ cubic2 q D = 0

-- The main theorem
theorem common_roots_product (C D : ℝ) : 
  has_two_common_roots C D → 
  ∃ (p q : ℝ), p * q = 10 * (7/2)^(1/3) :=
sorry

end NUMINAMATH_CALUDE_common_roots_product_l2290_229041


namespace NUMINAMATH_CALUDE_greatest_power_of_seven_l2290_229099

def r : ℕ := (List.range 50).foldl (· * ·) 1

theorem greatest_power_of_seven (k : ℕ) : k ≤ 8 ↔ (7^k : ℕ) ∣ r :=
sorry

end NUMINAMATH_CALUDE_greatest_power_of_seven_l2290_229099


namespace NUMINAMATH_CALUDE_total_shaded_area_l2290_229060

/-- Given a square carpet with the following properties:
  * Total side length of 16 feet
  * Contains one large shaded square and twelve smaller congruent shaded squares
  * Ratio of carpet side to large shaded square side (S) is 4:1
  * Ratio of large shaded square side (S) to smaller shaded square side (T) is 2:1
  The total shaded area is 64 square feet. -/
theorem total_shaded_area (carpet_side : ℝ) (S T : ℝ) : 
  carpet_side = 16 ∧ 
  carpet_side / S = 4 ∧ 
  S / T = 2 → 
  S^2 + 12 * T^2 = 64 := by
  sorry

end NUMINAMATH_CALUDE_total_shaded_area_l2290_229060


namespace NUMINAMATH_CALUDE_factor_expression_l2290_229071

theorem factor_expression (x : ℝ) : 6 * x^3 - 54 * x = 6 * x * (x + 3) * (x - 3) := by
  sorry

end NUMINAMATH_CALUDE_factor_expression_l2290_229071


namespace NUMINAMATH_CALUDE_west_60_meters_representation_l2290_229026

/-- Represents the direction of movement --/
inductive Direction
  | East
  | West

/-- Represents a movement with direction and distance --/
structure Movement where
  direction : Direction
  distance : ℝ

/-- Converts a movement to its numerical representation --/
def Movement.toNumber (m : Movement) : ℝ :=
  match m.direction with
  | Direction.East => -m.distance
  | Direction.West => m.distance

theorem west_60_meters_representation :
  ∀ (m : Movement),
    m.direction = Direction.West ∧
    m.distance = 60 →
    m.toNumber = 60 :=
by sorry

end NUMINAMATH_CALUDE_west_60_meters_representation_l2290_229026


namespace NUMINAMATH_CALUDE_sum_product_solution_l2290_229035

theorem sum_product_solution (S P : ℝ) (x y : ℝ) (h1 : x + y = S) (h2 : x * y = P) :
  ((x = (S + Real.sqrt (S^2 - 4*P)) / 2 ∧ y = (S - Real.sqrt (S^2 - 4*P)) / 2) ∨
   (x = (S - Real.sqrt (S^2 - 4*P)) / 2 ∧ y = (S + Real.sqrt (S^2 - 4*P)) / 2)) :=
by sorry

end NUMINAMATH_CALUDE_sum_product_solution_l2290_229035


namespace NUMINAMATH_CALUDE_carol_savings_per_week_l2290_229087

/-- Proves that Carol saves $9 per week given the initial conditions and final equality of savings --/
theorem carol_savings_per_week (carol_initial : ℕ) (mike_initial : ℕ) (mike_savings : ℕ) (weeks : ℕ)
  (h1 : carol_initial = 60)
  (h2 : mike_initial = 90)
  (h3 : mike_savings = 3)
  (h4 : weeks = 5)
  (h5 : ∃ (carol_savings : ℕ), carol_initial + weeks * carol_savings = mike_initial + weeks * mike_savings) :
  ∃ (carol_savings : ℕ), carol_savings = 9 := by
  sorry

end NUMINAMATH_CALUDE_carol_savings_per_week_l2290_229087


namespace NUMINAMATH_CALUDE_ice_cream_flavors_l2290_229045

theorem ice_cream_flavors (total_flavors : ℕ) (tried_two_years_ago : ℕ) (tried_last_year : ℕ) : 
  total_flavors = 100 →
  tried_two_years_ago = total_flavors / 4 →
  tried_last_year = 2 * tried_two_years_ago →
  total_flavors - (tried_two_years_ago + tried_last_year) = 25 :=
by
  sorry

end NUMINAMATH_CALUDE_ice_cream_flavors_l2290_229045


namespace NUMINAMATH_CALUDE_prob_odd_total_is_221_441_l2290_229072

/-- Represents a standard die with one dot removed randomly -/
structure ModifiedDie :=
  (remaining_dots : Fin 21)

/-- The probability of a modified die showing an odd number of dots on top -/
def prob_odd_top (d : ModifiedDie) : ℚ := 11 / 21

/-- The probability of a modified die showing an even number of dots on top -/
def prob_even_top (d : ModifiedDie) : ℚ := 10 / 21

/-- The probability of two modified dice showing an odd total number of dots on top when rolled simultaneously -/
def prob_odd_total (d1 d2 : ModifiedDie) : ℚ :=
  (prob_odd_top d1 * prob_odd_top d2) + (prob_even_top d1 * prob_even_top d2)

theorem prob_odd_total_is_221_441 (d1 d2 : ModifiedDie) :
  prob_odd_total d1 d2 = 221 / 441 := by
  sorry

end NUMINAMATH_CALUDE_prob_odd_total_is_221_441_l2290_229072


namespace NUMINAMATH_CALUDE_melanie_book_count_l2290_229092

/-- The total number of books Melanie has after buying more books is equal to the sum of her initial book count and the number of books she bought. -/
theorem melanie_book_count (initial_books new_books : ℝ) :
  let total_books := initial_books + new_books
  total_books = initial_books + new_books :=
by sorry

end NUMINAMATH_CALUDE_melanie_book_count_l2290_229092


namespace NUMINAMATH_CALUDE_parabola_equation_l2290_229048

/-- A parabola in the Cartesian coordinate system with focus at (-2, 0) has the standard equation y^2 = -8x -/
theorem parabola_equation (x y : ℝ) : 
  (∃ (p : ℝ), p > 0 ∧ y^2 = -2*p*x ∧ p/2 = 2) → 
  y^2 = -8*x := by
sorry

end NUMINAMATH_CALUDE_parabola_equation_l2290_229048


namespace NUMINAMATH_CALUDE_ellipse_m_range_l2290_229042

/-- An ellipse equation with parameter m -/
def ellipse_equation (x y m : ℝ) : Prop :=
  x^2 / (m + 2) - y^2 / (m + 1) = 1

/-- Condition for foci on y-axis -/
def foci_on_y_axis (m : ℝ) : Prop :=
  -(m + 1) > m + 2 ∧ m + 2 > 0

/-- Theorem stating the range of m for the given ellipse -/
theorem ellipse_m_range :
  ∀ m : ℝ, (∃ x y : ℝ, ellipse_equation x y m) ∧ foci_on_y_axis m ↔ -2 < m ∧ m < -3/2 :=
sorry

end NUMINAMATH_CALUDE_ellipse_m_range_l2290_229042


namespace NUMINAMATH_CALUDE_tangent_lines_condition_l2290_229049

/-- The function f(x) = 4x + ax² has two tangent lines passing through (1,1) iff a ∈ (-∞, -3) ∪ (0, +∞) -/
theorem tangent_lines_condition (a : ℝ) :
  (∃ (x₁ x₂ : ℝ), x₁ ≠ x₂ ∧
    (4 * x₁ + a * x₁^2 - (4 + 2*a*x₁) * x₁ + (4 + 2*a*x₁) = 1) ∧
    (4 * x₂ + a * x₂^2 - (4 + 2*a*x₂) * x₂ + (4 + 2*a*x₂) = 1)) ↔
  (a < -3 ∨ a > 0) :=
by sorry


end NUMINAMATH_CALUDE_tangent_lines_condition_l2290_229049


namespace NUMINAMATH_CALUDE_candy_bag_problem_l2290_229095

theorem candy_bag_problem (n : ℕ) (r : ℕ) : 
  n > 0 →  -- Ensure the bag is not empty
  r > 0 →  -- Ensure there are red candies
  r ≤ n →  -- Ensure the number of red candies doesn't exceed the total
  (r : ℚ) / n = 5 / 6 →  -- Probability of choosing a red candy
  n = 6 :=
by sorry

end NUMINAMATH_CALUDE_candy_bag_problem_l2290_229095


namespace NUMINAMATH_CALUDE_smaller_bill_denomination_l2290_229020

def total_amount : ℕ := 1000
def fraction_smaller : ℚ := 3 / 10
def larger_denomination : ℕ := 100
def total_bills : ℕ := 13

theorem smaller_bill_denomination :
  ∃ (smaller_denomination : ℕ),
    (fraction_smaller * total_amount) / smaller_denomination +
    ((1 - fraction_smaller) * total_amount) / larger_denomination = total_bills ∧
    smaller_denomination = 50 := by
  sorry

end NUMINAMATH_CALUDE_smaller_bill_denomination_l2290_229020


namespace NUMINAMATH_CALUDE_circle_properties_l2290_229063

/-- 
Given an equation x^2 + y^2 - 2x + 4y + m = 0 representing a circle,
prove that the center coordinates are (1, -2) and the range of m is (-∞, 5)
-/
theorem circle_properties (x y m : ℝ) :
  (x^2 + y^2 - 2*x + 4*y + m = 0) →
  (∃ r : ℝ, r > 0 ∧ (x - 1)^2 + (y + 2)^2 = r^2) →
  ((1, -2) = (1, -2) ∧ m < 5) := by
sorry

end NUMINAMATH_CALUDE_circle_properties_l2290_229063


namespace NUMINAMATH_CALUDE_teacher_worksheets_proof_l2290_229052

def calculate_total_worksheets (initial : ℕ) (graded : ℕ) (additional : ℕ) : ℕ :=
  let remaining := initial - graded
  let after_additional := remaining + additional
  after_additional + 2 * after_additional

theorem teacher_worksheets_proof : 
  let initial := 6
  let graded := 4
  let additional := 18
  calculate_total_worksheets initial graded additional = 60 := by
  sorry

end NUMINAMATH_CALUDE_teacher_worksheets_proof_l2290_229052


namespace NUMINAMATH_CALUDE_group_commutativity_l2290_229008

theorem group_commutativity (G : Type*) [Group G] (m n : ℕ) 
  (coprime_mn : Nat.Coprime m n)
  (surj_m : Function.Surjective (fun x : G => x^(m+1)))
  (surj_n : Function.Surjective (fun x : G => x^(n+1)))
  (endo_m : ∀ (x y : G), (x*y)^(m+1) = x^(m+1) * y^(m+1))
  (endo_n : ∀ (x y : G), (x*y)^(n+1) = x^(n+1) * y^(n+1)) :
  ∀ (a b : G), a * b = b * a := by
  sorry

end NUMINAMATH_CALUDE_group_commutativity_l2290_229008


namespace NUMINAMATH_CALUDE_dog_reachable_area_is_8pi_l2290_229098

/-- The area a dog can reach when tethered to a vertex of a regular hexagonal doghouse -/
def dogReachableArea (side_length : ℝ) (rope_length : ℝ) : ℝ :=
  -- Define the area calculation here
  sorry

/-- Theorem stating the area a dog can reach for the given conditions -/
theorem dog_reachable_area_is_8pi :
  dogReachableArea 2 3 = 8 * Real.pi := by
  sorry

end NUMINAMATH_CALUDE_dog_reachable_area_is_8pi_l2290_229098


namespace NUMINAMATH_CALUDE_largest_four_digit_divisible_by_33_l2290_229055

/-- A function that checks if a number is a four-digit number -/
def is_four_digit (n : ℕ) : Prop :=
  1000 ≤ n ∧ n ≤ 9999

/-- A function that checks if a number is divisible by 33 -/
def divisible_by_33 (n : ℕ) : Prop :=
  n % 33 = 0

/-- Theorem stating that 9999 is the largest four-digit number divisible by 33 -/
theorem largest_four_digit_divisible_by_33 :
  is_four_digit 9999 ∧ 
  divisible_by_33 9999 ∧ 
  ∀ n : ℕ, is_four_digit n → divisible_by_33 n → n ≤ 9999 :=
by
  sorry

end NUMINAMATH_CALUDE_largest_four_digit_divisible_by_33_l2290_229055


namespace NUMINAMATH_CALUDE_digital_earth_data_source_is_high_speed_network_databases_l2290_229082

/-- Represents the possible sources of spatial data for the digital Earth -/
inductive SpatialDataSource
  | SatelliteRemoteSensing
  | HighSpeedNetworkDatabases
  | InformationHighway
  | GISExchangeData

/-- Represents the digital Earth -/
structure DigitalEarth where
  mainDataSource : SpatialDataSource

/-- Axiom: The main source of basic spatial data for the digital Earth is from high-speed network databases -/
axiom digital_earth_main_data_source :
  ∀ (de : DigitalEarth), de.mainDataSource = SpatialDataSource.HighSpeedNetworkDatabases

/-- Theorem: The main source of basic spatial data for the digital Earth is from high-speed network databases -/
theorem digital_earth_data_source_is_high_speed_network_databases (de : DigitalEarth) :
  de.mainDataSource = SpatialDataSource.HighSpeedNetworkDatabases :=
by sorry

end NUMINAMATH_CALUDE_digital_earth_data_source_is_high_speed_network_databases_l2290_229082


namespace NUMINAMATH_CALUDE_one_ball_selection_l2290_229070

/-- The number of red balls in the bag -/
def num_red_balls : ℕ := 2

/-- The number of blue balls in the bag -/
def num_blue_balls : ℕ := 4

/-- Each ball has a different number -/
axiom balls_are_distinct : True

/-- The number of ways to select one ball from the bag -/
def ways_to_select_one_ball : ℕ := num_red_balls + num_blue_balls

theorem one_ball_selection :
  ways_to_select_one_ball = 6 :=
by sorry

end NUMINAMATH_CALUDE_one_ball_selection_l2290_229070


namespace NUMINAMATH_CALUDE_light_bulb_ratio_l2290_229097

theorem light_bulb_ratio (initial : ℕ) (used : ℕ) (left : ℕ) : 
  initial = 40 → used = 16 → left = 12 → 
  (initial - used - left) = left := by
  sorry

end NUMINAMATH_CALUDE_light_bulb_ratio_l2290_229097


namespace NUMINAMATH_CALUDE_max_area_quadrilateral_l2290_229021

/-- Given a point P in the first quadrant and points A on the x-axis and B on the y-axis 
    such that PA = PB = 2, the maximum area of quadrilateral PAOB is 2 + 2√2. -/
theorem max_area_quadrilateral (P A B : ℝ × ℝ) : 
  (0 < P.1 ∧ 0 < P.2) →  -- P is in the first quadrant
  A.2 = 0 →  -- A is on the x-axis
  B.1 = 0 →  -- B is on the y-axis
  Real.sqrt ((P.1 - A.1)^2 + P.2^2) = 2 →  -- PA = 2
  Real.sqrt (P.1^2 + (P.2 - B.2)^2) = 2 →  -- PB = 2
  (∃ (area : ℝ), ∀ (Q : ℝ × ℝ), 
    (0 < Q.1 ∧ 0 < Q.2) →
    Real.sqrt ((Q.1 - A.1)^2 + Q.2^2) = 2 →
    Real.sqrt (Q.1^2 + (Q.2 - B.2)^2) = 2 →
    (1/2 * |A.1 * Q.1 + B.2 * Q.2| ≤ area)) ∧
  (1/2 * |A.1 * P.1 + B.2 * P.2| = 2 + 2 * Real.sqrt 2) :=
by sorry

end NUMINAMATH_CALUDE_max_area_quadrilateral_l2290_229021


namespace NUMINAMATH_CALUDE_x_squared_less_than_one_iff_l2290_229006

theorem x_squared_less_than_one_iff (x : ℝ) : -1 < x ∧ x < 1 ↔ x^2 < 1 := by sorry

end NUMINAMATH_CALUDE_x_squared_less_than_one_iff_l2290_229006


namespace NUMINAMATH_CALUDE_percentage_of_quarters_l2290_229005

theorem percentage_of_quarters (dimes quarters half_dollars : ℕ) : 
  dimes = 75 → quarters = 35 → half_dollars = 15 →
  (quarters * 25 : ℚ) / (dimes * 10 + quarters * 25 + half_dollars * 50) = 368 / 1000 := by
  sorry

end NUMINAMATH_CALUDE_percentage_of_quarters_l2290_229005


namespace NUMINAMATH_CALUDE_fourth_power_nested_square_roots_l2290_229057

theorem fourth_power_nested_square_roots :
  (Real.sqrt (1 + Real.sqrt (2 + Real.sqrt (3 + Real.sqrt 4))))^4 = 3 + Real.sqrt 5 + 2 * Real.sqrt (2 + Real.sqrt 5) :=
by sorry

end NUMINAMATH_CALUDE_fourth_power_nested_square_roots_l2290_229057


namespace NUMINAMATH_CALUDE_inequality_solution_l2290_229033

theorem inequality_solution (x : ℝ) : 
  1 / (x - 2) + 4 / (x + 5) ≥ 1 ↔ x ∈ Set.Icc (-1) (7/2) :=
by sorry

end NUMINAMATH_CALUDE_inequality_solution_l2290_229033


namespace NUMINAMATH_CALUDE_triangle_properties_l2290_229080

-- Define the triangle ABC
structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ

-- Define a line equation
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

def Triangle.hasAltitude (t : Triangle) (l : Line) : Prop :=
  l.a * t.A.1 + l.b * t.A.2 + l.c = 0

def Triangle.hasAngleBisector (t : Triangle) (l : Line) : Prop :=
  l.a * t.B.1 + l.b * t.B.2 + l.c = 0

theorem triangle_properties (t : Triangle) (altitude : Line) (bisector : Line) :
  t.A = (1, 1) →
  altitude = { a := 3, b := 1, c := -12 } →
  bisector = { a := 1, b := -2, c := 4 } →
  t.hasAltitude altitude →
  t.hasAngleBisector bisector →
  t.B = (-8, -2) ∧
  (∃ (l : Line), l = { a := 9, b := -13, c := 46 } ∧ 
    l.a * t.B.1 + l.b * t.B.2 + l.c = 0 ∧
    l.a * t.C.1 + l.b * t.C.2 + l.c = 0) :=
by sorry

end NUMINAMATH_CALUDE_triangle_properties_l2290_229080


namespace NUMINAMATH_CALUDE_equation_solution_l2290_229022

theorem equation_solution :
  ∃! y : ℝ, 7 * (4 * y + 3) - 5 = -3 * (2 - 8 * y) + 5 * y ∧ y = 22 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l2290_229022


namespace NUMINAMATH_CALUDE_units_digit_of_7_pow_3_pow_5_l2290_229090

def units_digit_pattern : ℕ → ℕ
| 0 => 7
| 1 => 9
| 2 => 3
| 3 => 1
| n + 4 => units_digit_pattern n

def power_mod (base exponent modulus : ℕ) : ℕ :=
  (base ^ exponent) % modulus

theorem units_digit_of_7_pow_3_pow_5 :
  units_digit_pattern (power_mod 3 5 4) = 3 := by sorry

end NUMINAMATH_CALUDE_units_digit_of_7_pow_3_pow_5_l2290_229090


namespace NUMINAMATH_CALUDE_inequality_proof_l2290_229013

theorem inequality_proof (a b c : ℝ) (h : Real.sqrt a ≥ Real.sqrt (b * c) ∧ Real.sqrt (b * c) ≥ Real.sqrt a - c) : b * c ≥ b + c := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l2290_229013


namespace NUMINAMATH_CALUDE_music_movements_duration_l2290_229018

theorem music_movements_duration 
  (a b c : ℝ) 
  (h_order : a ≤ b ∧ b ≤ c) 
  (h_total : a + b + c = 60) 
  (h_max : c ≤ a + b) 
  (h_diff : b - a ≥ 3 ∧ c - b ≥ 3) 
  (h_positive : a > 0 ∧ b > 0 ∧ c > 0) : 
  3 ≤ a ∧ a ≤ 17 := by
sorry

end NUMINAMATH_CALUDE_music_movements_duration_l2290_229018


namespace NUMINAMATH_CALUDE_happy_number_transformation_l2290_229085

def is_happy_number (n : ℕ) : Prop :=
  n ≥ 100 ∧ n < 1000 ∧
  (n / 100 + (n / 10 % 10) - n % 10 = 6)

def transform (m : ℕ) : ℕ :=
  let c := m % 10
  let a := m / 100
  let b := (m / 10) % 10
  2 * c * 100 + a * 10 + b

theorem happy_number_transformation :
  {m : ℕ | is_happy_number m ∧ is_happy_number (transform m)} = {532, 464} := by
  sorry

end NUMINAMATH_CALUDE_happy_number_transformation_l2290_229085


namespace NUMINAMATH_CALUDE_polynomial_divisibility_l2290_229047

theorem polynomial_divisibility : ∃ q : Polynomial ℝ, 
  (X^3 * 6 + X^2 * 1 + -1) = (X * 2 + -1) * q :=
by
  sorry

end NUMINAMATH_CALUDE_polynomial_divisibility_l2290_229047


namespace NUMINAMATH_CALUDE_constant_product_l2290_229053

-- Define the parabolas and points
def parabola1 (p : ℝ) (x y : ℝ) : Prop := y^2 = 2*p*x ∧ p > 0
def parabola2 (q : ℝ) (x y : ℝ) : Prop := x^2 = 2*q*y ∧ q > 0

-- Define the property of being distinct points on parabola1
def distinct_points_on_parabola1 (p : ℝ) (x₁ y₁ x₂ y₂ x₃ y₃ : ℝ) : Prop :=
  parabola1 p x₁ y₁ ∧ parabola1 p x₂ y₂ ∧ parabola1 p x₃ y₃ ∧
  (x₁ ≠ x₂ ∨ y₁ ≠ y₂) ∧ (x₁ ≠ x₃ ∨ y₁ ≠ y₃) ∧ (x₂ ≠ x₃ ∨ y₂ ≠ y₃)

-- Define the property of two sides being tangent to parabola2
def two_sides_tangent (q : ℝ) (x₁ y₁ x₂ y₂ x₃ y₃ : ℝ) : Prop :=
  ∃ (xt₁ yt₁ xt₂ yt₂ : ℝ),
    parabola2 q xt₁ yt₁ ∧ parabola2 q xt₂ yt₂ ∧
    (xt₁ - x₁) * (y₂ - y₁) = (yt₁ - y₁) * (x₂ - x₁) ∧
    (xt₁ - x₂) * (y₁ - y₂) = (yt₁ - y₂) * (x₁ - x₂) ∧
    (xt₂ - x₁) * (y₃ - y₁) = (yt₂ - y₁) * (x₃ - x₁) ∧
    (xt₂ - x₃) * (y₁ - y₃) = (yt₂ - y₃) * (x₁ - x₃)

-- Theorem statement
theorem constant_product (p q : ℝ) (x₁ y₁ x₂ y₂ x₃ y₃ : ℝ) :
  distinct_points_on_parabola1 p x₁ y₁ x₂ y₂ x₃ y₃ →
  two_sides_tangent q x₁ y₁ x₂ y₂ x₃ y₃ →
  ∃ (c : ℝ), ∀ (i j : Fin 3), i ≠ j →
    let y := [y₁, y₂, y₃]
    y[i] * y[j] * (y[i] + y[j]) = c :=
by sorry

end NUMINAMATH_CALUDE_constant_product_l2290_229053


namespace NUMINAMATH_CALUDE_square_root_sum_equality_l2290_229058

theorem square_root_sum_equality : 
  Real.sqrt (16 - 8 * Real.sqrt 3) + 2 * Real.sqrt (16 + 8 * Real.sqrt 3) = 10 + 6 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_square_root_sum_equality_l2290_229058


namespace NUMINAMATH_CALUDE_sport_formulation_water_amount_l2290_229065

/-- Represents the ratio of ingredients in a drink formulation -/
structure DrinkRatio :=
  (flavoring : ℚ)
  (corn_syrup : ℚ)
  (water : ℚ)

/-- The standard formulation ratio -/
def standard_ratio : DrinkRatio :=
  ⟨1, 12, 30⟩

/-- The sport formulation ratio -/
def sport_ratio : DrinkRatio :=
  ⟨1, 4, 60⟩

/-- Theorem stating the relationship between corn syrup and water in the sport formulation -/
theorem sport_formulation_water_amount
  (corn_syrup_amount : ℚ)
  (h1 : corn_syrup_amount = 5)
  (h2 : sport_ratio.flavoring * sport_ratio.water = 
        2 * (standard_ratio.flavoring * standard_ratio.water))
  (h3 : sport_ratio.flavoring * sport_ratio.corn_syrup = 
        3 * (standard_ratio.flavoring * standard_ratio.corn_syrup)) :
  corn_syrup_amount * (sport_ratio.water / sport_ratio.corn_syrup) = 75 :=
sorry

end NUMINAMATH_CALUDE_sport_formulation_water_amount_l2290_229065


namespace NUMINAMATH_CALUDE_right_triangle_hypotenuse_l2290_229068

theorem right_triangle_hypotenuse (a b c : ℝ) : 
  a = 60 → b = 80 → c^2 = a^2 + b^2 → c = 100 := by sorry

end NUMINAMATH_CALUDE_right_triangle_hypotenuse_l2290_229068


namespace NUMINAMATH_CALUDE_permutations_of_six_distinct_objects_l2290_229079

theorem permutations_of_six_distinct_objects : Nat.factorial 6 = 720 := by
  sorry

end NUMINAMATH_CALUDE_permutations_of_six_distinct_objects_l2290_229079


namespace NUMINAMATH_CALUDE_sector_area_l2290_229004

/-- Given a circular sector with an arc length of 2 cm and a central angle of 2 radians,
    prove that the area of the sector is 1 cm². -/
theorem sector_area (arc_length : ℝ) (central_angle : ℝ) (h1 : arc_length = 2) (h2 : central_angle = 2) :
  (1 / 2) * (arc_length / central_angle)^2 * central_angle = 1 := by
  sorry

end NUMINAMATH_CALUDE_sector_area_l2290_229004


namespace NUMINAMATH_CALUDE_product_sum_geq_geometric_mean_sum_l2290_229019

theorem product_sum_geq_geometric_mean_sum {a b c : ℝ} (ha : a ≥ 0) (hb : b ≥ 0) (hc : c ≥ 0) :
  a * b + b * c + c * a ≥ a * Real.sqrt (b * c) + b * Real.sqrt (a * c) + c * Real.sqrt (a * b) := by
  sorry

end NUMINAMATH_CALUDE_product_sum_geq_geometric_mean_sum_l2290_229019


namespace NUMINAMATH_CALUDE_sheep_herds_l2290_229014

theorem sheep_herds (total_sheep : ℕ) (sheep_per_herd : ℕ) (h1 : total_sheep = 60) (h2 : sheep_per_herd = 20) :
  total_sheep / sheep_per_herd = 3 := by
sorry

end NUMINAMATH_CALUDE_sheep_herds_l2290_229014


namespace NUMINAMATH_CALUDE_pentagonal_prism_vertices_l2290_229038

/-- Definition of a pentagonal prism -/
structure PentagonalPrism :=
  (bases : ℕ)
  (rectangular_faces : ℕ)
  (h_bases : bases = 2)
  (h_faces : rectangular_faces = 5)

/-- The number of vertices in a pentagonal prism -/
def num_vertices (p : PentagonalPrism) : ℕ := 10

/-- Theorem stating that a pentagonal prism has 10 vertices -/
theorem pentagonal_prism_vertices (p : PentagonalPrism) : num_vertices p = 10 := by
  sorry

end NUMINAMATH_CALUDE_pentagonal_prism_vertices_l2290_229038


namespace NUMINAMATH_CALUDE_park_circle_diameter_l2290_229056

/-- Represents the circular arrangement in the park -/
structure ParkCircle where
  fountain_diameter : ℝ
  garden_width : ℝ
  inner_path_width : ℝ
  outer_path_width : ℝ

/-- Calculates the diameter of the outermost boundary of the park circle -/
def outer_diameter (park : ParkCircle) : ℝ :=
  park.fountain_diameter + 2 * (park.garden_width + park.inner_path_width + park.outer_path_width)

/-- Theorem stating that for the given dimensions, the outer diameter is 50 feet -/
theorem park_circle_diameter :
  let park : ParkCircle := {
    fountain_diameter := 12,
    garden_width := 9,
    inner_path_width := 3,
    outer_path_width := 7
  }
  outer_diameter park = 50 := by
  sorry

end NUMINAMATH_CALUDE_park_circle_diameter_l2290_229056


namespace NUMINAMATH_CALUDE_product_of_powers_plus_one_l2290_229051

theorem product_of_powers_plus_one : (3 + 1) * (3^2 + 1^2) * (3^4 + 1^4) * (3^8 + 1^8) = 21527360 := by
  sorry

end NUMINAMATH_CALUDE_product_of_powers_plus_one_l2290_229051


namespace NUMINAMATH_CALUDE_max_abs_z_on_line_segment_l2290_229067

theorem max_abs_z_on_line_segment (z : ℂ) :
  Complex.abs (z - 6*I) + Complex.abs (z - 5) = Real.sqrt 61 →
  Complex.abs z ≤ 6 ∧ ∃ w : ℂ, Complex.abs (w - 6*I) + Complex.abs (w - 5) = Real.sqrt 61 ∧ Complex.abs w = 6 :=
by sorry

end NUMINAMATH_CALUDE_max_abs_z_on_line_segment_l2290_229067


namespace NUMINAMATH_CALUDE_jerry_aunt_money_l2290_229061

/-- The amount of money Jerry received from his aunt -/
def aunt_money : ℝ := 9.05

/-- The amount of money Jerry received from his uncle -/
def uncle_money : ℝ := aunt_money

/-- The amount of money Jerry received from his friends -/
def friends_money : ℝ := 22 + 23 + 22 + 22

/-- The amount of money Jerry received from his sister -/
def sister_money : ℝ := 7

/-- The mean of all the money Jerry received -/
def mean_money : ℝ := 16.3

/-- The number of sources Jerry received money from -/
def num_sources : ℕ := 7

theorem jerry_aunt_money :
  (friends_money + sister_money + aunt_money + uncle_money) / num_sources = mean_money :=
sorry

end NUMINAMATH_CALUDE_jerry_aunt_money_l2290_229061


namespace NUMINAMATH_CALUDE_favorite_song_not_heard_probability_l2290_229066

-- Define the number of songs
def num_songs : ℕ := 10

-- Define the duration of the shortest song (in seconds)
def shortest_song : ℕ := 40

-- Define the increment in duration for each subsequent song (in seconds)
def duration_increment : ℕ := 40

-- Define the duration of the favorite song (in seconds)
def favorite_song_duration : ℕ := 240

-- Define the total playtime we're considering (in seconds)
def total_playtime : ℕ := 300

-- Function to calculate the duration of the nth song
def song_duration (n : ℕ) : ℕ := shortest_song + (n - 1) * duration_increment

-- Theorem stating the probability of not hearing the favorite song in its entirety
theorem favorite_song_not_heard_probability :
  let favorite_song_index : ℕ := (favorite_song_duration - shortest_song) / duration_increment + 1
  (favorite_song_index ≤ num_songs) →
  (∀ n : ℕ, n < favorite_song_index → song_duration n + favorite_song_duration > total_playtime) →
  (num_songs - 1 : ℚ) / num_songs = 9 / 10 :=
by sorry

end NUMINAMATH_CALUDE_favorite_song_not_heard_probability_l2290_229066


namespace NUMINAMATH_CALUDE_kaleb_allowance_l2290_229023

theorem kaleb_allowance (savings : ℕ) (toy_cost : ℕ) (num_toys : ℕ) (allowance : ℕ) : 
  savings = 21 → 
  toy_cost = 6 → 
  num_toys = 6 → 
  allowance = num_toys * toy_cost - savings → 
  allowance = 15 := by
sorry

end NUMINAMATH_CALUDE_kaleb_allowance_l2290_229023


namespace NUMINAMATH_CALUDE_space_station_cost_share_l2290_229037

/-- Proves that if a total cost of 50 billion dollars is shared equally among 500 million people,
    then each person's share is 100 dollars. -/
theorem space_station_cost_share :
  let total_cost : ℝ := 50 * 10^9  -- 50 billion dollars
  let num_people : ℝ := 500 * 10^6  -- 500 million people
  let share_per_person : ℝ := total_cost / num_people
  share_per_person = 100 := by sorry

end NUMINAMATH_CALUDE_space_station_cost_share_l2290_229037


namespace NUMINAMATH_CALUDE_planes_parallel_if_perp_to_parallel_lines_l2290_229078

-- Define the types for planes and lines
variable (α β : Plane) (l m : Line)

-- Define the relationships between planes and lines
def perpendicular (l : Line) (α : Plane) : Prop := sorry
def parallel_lines (l m : Line) : Prop := sorry
def parallel_planes (α β : Plane) : Prop := sorry

-- State the theorem
theorem planes_parallel_if_perp_to_parallel_lines 
  (h1 : perpendicular l α) 
  (h2 : perpendicular m β) 
  (h3 : parallel_lines l m) : 
  parallel_planes α β := by sorry

end NUMINAMATH_CALUDE_planes_parallel_if_perp_to_parallel_lines_l2290_229078


namespace NUMINAMATH_CALUDE_seagulls_remaining_l2290_229043

theorem seagulls_remaining (initial : ℕ) (scared_fraction : ℚ) (flew_fraction : ℚ) : 
  initial = 36 → scared_fraction = 1/4 → flew_fraction = 1/3 → 
  (initial - initial * scared_fraction - (initial - initial * scared_fraction) * flew_fraction : ℚ) = 18 := by
  sorry

end NUMINAMATH_CALUDE_seagulls_remaining_l2290_229043


namespace NUMINAMATH_CALUDE_add_6666_seconds_to_3pm_l2290_229012

/-- Represents a time of day in hours, minutes, and seconds -/
structure TimeOfDay where
  hours : Nat
  minutes : Nat
  seconds : Nat
  deriving Repr

/-- Converts seconds to a TimeOfDay structure -/
def secondsToTime (totalSeconds : Nat) : TimeOfDay :=
  let hours := totalSeconds / 3600
  let remainingSeconds := totalSeconds % 3600
  let minutes := remainingSeconds / 60
  let seconds := remainingSeconds % 60
  { hours := hours, minutes := minutes, seconds := seconds }

/-- Adds a TimeOfDay to another TimeOfDay, handling overflow -/
def addTime (t1 t2 : TimeOfDay) : TimeOfDay :=
  let totalSeconds := (t1.hours * 3600 + t1.minutes * 60 + t1.seconds) +
                      (t2.hours * 3600 + t2.minutes * 60 + t2.seconds)
  secondsToTime totalSeconds

theorem add_6666_seconds_to_3pm (startTime : TimeOfDay) (elapsedSeconds : Nat) :
  startTime.hours = 15 ∧ startTime.minutes = 0 ∧ startTime.seconds = 0 ∧ 
  elapsedSeconds = 6666 →
  let endTime := addTime startTime (secondsToTime elapsedSeconds)
  endTime.hours = 16 ∧ endTime.minutes = 51 ∧ endTime.seconds = 6 := by
  sorry

end NUMINAMATH_CALUDE_add_6666_seconds_to_3pm_l2290_229012


namespace NUMINAMATH_CALUDE_hexagon_area_is_32_l2290_229059

/-- A hexagon surrounded by triangles forming a rectangle -/
structure HexagonWithTriangles where
  num_triangles : ℕ
  triangle_area : ℝ
  rectangle_area : ℝ

/-- The area of the hexagon -/
def hexagon_area (h : HexagonWithTriangles) : ℝ :=
  h.rectangle_area - h.num_triangles * h.triangle_area

/-- Theorem: The area of the hexagon is 32 square units -/
theorem hexagon_area_is_32 (h : HexagonWithTriangles) 
    (h_num_triangles : h.num_triangles = 4)
    (h_triangle_area : h.triangle_area = 2)
    (h_rectangle_area : h.rectangle_area = 40) : 
  hexagon_area h = 32 := by
  sorry

end NUMINAMATH_CALUDE_hexagon_area_is_32_l2290_229059


namespace NUMINAMATH_CALUDE_sum_of_squares_equals_two_l2290_229001

theorem sum_of_squares_equals_two (x y z : ℤ) 
  (h1 : |x + y| + |y + z| + |z + x| = 4)
  (h2 : |x - y| + |y - z| + |z - x| = 2) : 
  x^2 + y^2 + z^2 = 2 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_squares_equals_two_l2290_229001


namespace NUMINAMATH_CALUDE_abes_budget_l2290_229046

/-- Abe's restaurant budget problem -/
theorem abes_budget (B : ℚ) 
  (food_expense : B / 3 = B - (B / 4 + 1250))
  (supplies_expense : B / 4 = B - (B / 3 + 1250))
  (wages_expense : 1250 = B - (B / 3 + B / 4))
  (total_expense : B = B / 3 + B / 4 + 1250) :
  B = 3000 := by
  sorry

end NUMINAMATH_CALUDE_abes_budget_l2290_229046


namespace NUMINAMATH_CALUDE_function_and_inequality_solution_l2290_229027

noncomputable section

variables (f : ℝ → ℝ) (f' : ℝ → ℝ)

theorem function_and_inequality_solution 
  (h1 : ∀ x, HasDerivAt f (f' x) x)
  (h2 : f 0 = 2020)
  (h3 : ∀ x, f' x = f x - 2) :
  (∀ x, f x = 2 + 2018 * Real.exp x) ∧ 
  {x : ℝ | f x + 4034 > 2 * f' x} = {x : ℝ | x < Real.log 2} := by
  sorry

end

end NUMINAMATH_CALUDE_function_and_inequality_solution_l2290_229027


namespace NUMINAMATH_CALUDE_min_sum_of_bases_l2290_229007

theorem min_sum_of_bases (a b : ℕ+) : 
  (3 * a + 5 = 5 * b + 3) → 
  (∀ (x y : ℕ+), (3 * x + 5 = 5 * y + 3) → (x + y ≥ a + b)) →
  a + b = 10 := by
sorry

end NUMINAMATH_CALUDE_min_sum_of_bases_l2290_229007


namespace NUMINAMATH_CALUDE_terrell_weight_lifting_l2290_229044

/-- The number of times Terrell lifts the weights in his initial routine -/
def initial_lifts : ℕ := 10

/-- The weight of each dumbbell in Terrell's initial routine (in pounds) -/
def initial_weight : ℕ := 25

/-- The number of dumbbells Terrell uses -/
def num_dumbbells : ℕ := 3

/-- The weight of each dumbbell in Terrell's new routine (in pounds) -/
def new_weight : ℕ := 20

/-- The total weight Terrell lifts in his initial routine -/
def total_initial_weight : ℕ := num_dumbbells * initial_weight * initial_lifts

/-- The minimum number of times Terrell needs to lift the new weights to match or exceed the initial total weight -/
def min_new_lifts : ℕ := 13

theorem terrell_weight_lifting :
  num_dumbbells * new_weight * min_new_lifts ≥ total_initial_weight :=
sorry

end NUMINAMATH_CALUDE_terrell_weight_lifting_l2290_229044


namespace NUMINAMATH_CALUDE_cheryl_material_used_l2290_229073

-- Define the amounts of materials
def material1 : ℚ := 2 / 9
def material2 : ℚ := 1 / 8
def leftover : ℚ := 4 / 18

-- Define the total amount bought
def total_bought : ℚ := material1 + material2

-- Define the theorem
theorem cheryl_material_used :
  total_bought - leftover = 1 / 8 := by
  sorry

end NUMINAMATH_CALUDE_cheryl_material_used_l2290_229073


namespace NUMINAMATH_CALUDE_interest_rate_calculation_l2290_229040

/-- Calculates the interest rate for a purchase with a payment plan. -/
theorem interest_rate_calculation (purchase_price down_payment monthly_payment num_months : ℚ)
  (h_purchase : purchase_price = 112)
  (h_down : down_payment = 12)
  (h_monthly : monthly_payment = 10)
  (h_months : num_months = 12) :
  ∃ (interest_rate : ℚ), 
    (abs (interest_rate - 17.9) < 0.05) ∧ 
    (interest_rate = (((down_payment + monthly_payment * num_months) - purchase_price) / purchase_price) * 100) :=
by sorry

end NUMINAMATH_CALUDE_interest_rate_calculation_l2290_229040


namespace NUMINAMATH_CALUDE_triangle_inequality_expression_l2290_229034

theorem triangle_inequality_expression (a b c : ℝ) : 
  (a > 0) → (b > 0) → (c > 0) → 
  (a + b > c) → (b + c > a) → (c + a > b) →
  (a^2 + b^2 - c^2 - 2*a*b < 0) :=
by
  sorry

end NUMINAMATH_CALUDE_triangle_inequality_expression_l2290_229034


namespace NUMINAMATH_CALUDE_sum_of_roots_l2290_229002

theorem sum_of_roots (M : ℝ) : (∃ M₁ M₂ : ℝ, M₁ * (M₁ - 8) = 7 ∧ M₂ * (M₂ - 8) = 7 ∧ M₁ + M₂ = 8) := by
  sorry

end NUMINAMATH_CALUDE_sum_of_roots_l2290_229002


namespace NUMINAMATH_CALUDE_pizza_order_total_l2290_229083

theorem pizza_order_total (m : ℕ) (total_pizzas : ℚ) : 
  m > 17 →
  (10 : ℚ) / m + 17 * ((10 : ℚ) / m) / 2 = total_pizzas →
  total_pizzas = 11 := by
  sorry

end NUMINAMATH_CALUDE_pizza_order_total_l2290_229083


namespace NUMINAMATH_CALUDE_min_xy_value_l2290_229096

theorem min_xy_value (x y : ℕ+) (h : (1 : ℚ) / x + (1 : ℚ) / (3 * y) = (1 : ℚ) / 8) : 
  (x * y : ℕ) ≥ 96 := by
sorry

end NUMINAMATH_CALUDE_min_xy_value_l2290_229096


namespace NUMINAMATH_CALUDE_temperature_conversion_l2290_229054

theorem temperature_conversion (k : ℝ) (t : ℝ) : 
  (t = 5 / 9 * (k - 32)) → (k = 167) → (t = 75) := by
  sorry

end NUMINAMATH_CALUDE_temperature_conversion_l2290_229054


namespace NUMINAMATH_CALUDE_fraction_simplification_l2290_229077

theorem fraction_simplification (x : ℝ) : (x - 1) / 3 + (-2 - 3 * x) / 2 = (-7 * x - 8) / 6 := by
  sorry

end NUMINAMATH_CALUDE_fraction_simplification_l2290_229077


namespace NUMINAMATH_CALUDE_smallest_prime_twelve_less_than_square_l2290_229074

theorem smallest_prime_twelve_less_than_square : ∃ n : ℕ, 
  (n > 0) ∧ 
  (Nat.Prime n) ∧ 
  (∃ m : ℕ, n = m^2 - 12) ∧
  (∀ k : ℕ, k > 0 → Nat.Prime k → (∃ l : ℕ, k = l^2 - 12) → k ≥ n) ∧
  n = 13 := by
sorry

end NUMINAMATH_CALUDE_smallest_prime_twelve_less_than_square_l2290_229074


namespace NUMINAMATH_CALUDE_floor_tiling_floor_covered_l2290_229089

/-- A square floor of size n × n can be completely covered by an equal number of 2 × 2 and 3 × 1 tiles if and only if n is a multiple of 7. -/
theorem floor_tiling (n : ℕ) : 
  (∃ (a : ℕ), n^2 = 7 * a) ↔ ∃ (k : ℕ), n = 7 * k :=
by sorry

/-- The number of tiles of each type needed to cover a square floor of size n × n, where n is a multiple of 7. -/
def num_tiles (n : ℕ) (h : ∃ (k : ℕ), n = 7 * k) : ℕ :=
  n^2 / 7

/-- Verification that the floor is completely covered using an equal number of 2 × 2 and 3 × 1 tiles. -/
theorem floor_covered (n : ℕ) (h : ∃ (k : ℕ), n = 7 * k) :
  let a := num_tiles n h
  4 * a + 3 * a = n^2 :=
by sorry

end NUMINAMATH_CALUDE_floor_tiling_floor_covered_l2290_229089


namespace NUMINAMATH_CALUDE_last_colored_number_l2290_229036

/-- The number of columns in the table -/
def num_columns : ℕ := 8

/-- The triangular number sequence -/
def triangular_number (n : ℕ) : ℕ := n * (n + 1) / 2

/-- The position of a number in the table -/
def position (n : ℕ) : ℕ := n % num_columns

/-- Predicate to check if a number is colored -/
def is_colored (n : ℕ) : Prop := ∃ k : ℕ, n = triangular_number k

/-- Predicate to check if all columns are colored up to a certain number -/
def all_columns_colored (n : ℕ) : Prop :=
  ∀ col : ℕ, col < num_columns → ∃ m : ℕ, m ≤ n ∧ is_colored m ∧ position m = col

/-- The main theorem -/
theorem last_colored_number :
  ∃ n : ℕ, n = 120 ∧ is_colored n ∧ all_columns_colored n ∧
  ∀ m : ℕ, m < n → ¬(all_columns_colored m) :=
sorry

end NUMINAMATH_CALUDE_last_colored_number_l2290_229036


namespace NUMINAMATH_CALUDE_minimum_rental_fee_is_3520_l2290_229009

/-- Represents a bus type with its seat capacity and rental fee. -/
structure BusType where
  seats : ℕ
  fee : ℕ

/-- Calculates the minimum rental fee for transporting a given number of people
    using two types of buses. -/
def minimumRentalFee (people : ℕ) (busA : BusType) (busB : BusType) : ℕ :=
  let totalBuses := 8
  let x := 4  -- number of Type A buses
  x * busA.fee + (totalBuses - x) * busB.fee

/-- Theorem stating that the minimum rental fee for 360 people using the given bus types is 3520 yuan. -/
theorem minimum_rental_fee_is_3520 :
  let people := 360
  let busA := BusType.mk 40 400
  let busB := BusType.mk 50 480
  minimumRentalFee people busA busB = 3520 := by
  sorry

#eval minimumRentalFee 360 (BusType.mk 40 400) (BusType.mk 50 480)

end NUMINAMATH_CALUDE_minimum_rental_fee_is_3520_l2290_229009


namespace NUMINAMATH_CALUDE_decimal_to_fraction_l2290_229016

theorem decimal_to_fraction : 
  (2.75 : ℚ) = 11 / 4 := by sorry

end NUMINAMATH_CALUDE_decimal_to_fraction_l2290_229016


namespace NUMINAMATH_CALUDE_min_value_of_expression_l2290_229024

theorem min_value_of_expression (m n : ℝ) (hm : m > 0) (hn : n > 0) : 
  let a : ℝ × ℝ := (m, 1)
  let b : ℝ × ℝ := (4 - n, 2)
  (∃ (k : ℝ), a = k • b) → 
  (∀ (x y : ℝ), x > 0 → y > 0 → (1 / x + 8 / y ≥ 9 / 2) ∧ 
  (∃ (x₀ y₀ : ℝ), x₀ > 0 ∧ y₀ > 0 ∧ 1 / x₀ + 8 / y₀ = 9 / 2)) :=
by sorry

end NUMINAMATH_CALUDE_min_value_of_expression_l2290_229024


namespace NUMINAMATH_CALUDE_sum_of_coefficients_l2290_229081

theorem sum_of_coefficients (a₀ a₁ a₂ a₃ a₄ a₅ a₆ a₇ a₈ : ℝ) :
  (∀ x : ℝ, (1 + 2*x) * (1 - 2*x)^7 = a₀ + a₁*x + a₂*x^2 + a₃*x^3 + a₄*x^4 + a₅*x^5 + a₆*x^6 + a₇*x^7 + a₈*x^8) →
  a₀ + a₁ + a₂ + a₃ + a₄ + a₅ + a₆ + a₇ = 253 :=
by sorry

end NUMINAMATH_CALUDE_sum_of_coefficients_l2290_229081


namespace NUMINAMATH_CALUDE_belmont_basketball_winning_percentage_l2290_229031

theorem belmont_basketball_winning_percentage 
  (X : ℕ) (Y Z : ℝ) (h1 : 0 < Y) (h2 : Y < 100) (h3 : 0 < Z) (h4 : Z < 100) :
  let G := X * ((Y / 100) - (Z / 100)) / (Z / 100 - 1)
  ∃ (G : ℝ), (Z / 100) * (X + G) = (Y / 100) * X + G :=
by sorry

end NUMINAMATH_CALUDE_belmont_basketball_winning_percentage_l2290_229031


namespace NUMINAMATH_CALUDE_weight_difference_l2290_229084

/-- The weights of individuals A, B, C, D, and E --/
structure Weights where
  A : ℝ
  B : ℝ
  C : ℝ
  D : ℝ
  E : ℝ

/-- The conditions of the problem --/
def WeightConditions (w : Weights) : Prop :=
  (w.A + w.B + w.C) / 3 = 84 ∧
  (w.A + w.B + w.C + w.D) / 4 = 80 ∧
  (w.B + w.C + w.D + w.E) / 4 = 79 ∧
  w.A = 77 ∧
  w.E > w.D

/-- The theorem to prove --/
theorem weight_difference (w : Weights) (h : WeightConditions w) : w.E - w.D = 5 := by
  sorry

end NUMINAMATH_CALUDE_weight_difference_l2290_229084


namespace NUMINAMATH_CALUDE_initial_snowflakes_count_l2290_229017

/-- Calculates the initial number of snowflakes given the rate of snowfall and total snowflakes after one hour -/
def initial_snowflakes (rate : ℕ) (interval : ℕ) (total_after_hour : ℕ) : ℕ :=
  total_after_hour - (60 / interval) * rate

/-- Theorem: The initial number of snowflakes is 10 -/
theorem initial_snowflakes_count : initial_snowflakes 4 5 58 = 10 := by
  sorry

end NUMINAMATH_CALUDE_initial_snowflakes_count_l2290_229017


namespace NUMINAMATH_CALUDE_prove_earnings_l2290_229011

/-- Gondor's earnings from repairing devices -/
def earnings_problem : Prop :=
  let phone_repair_fee : ℕ := 10
  let laptop_repair_fee : ℕ := 20
  let monday_phones : ℕ := 3
  let tuesday_phones : ℕ := 5
  let wednesday_laptops : ℕ := 2
  let thursday_laptops : ℕ := 4
  let total_earnings : ℕ := 
    phone_repair_fee * (monday_phones + tuesday_phones) +
    laptop_repair_fee * (wednesday_laptops + thursday_laptops)
  total_earnings = 200

theorem prove_earnings : earnings_problem := by
  sorry

end NUMINAMATH_CALUDE_prove_earnings_l2290_229011


namespace NUMINAMATH_CALUDE_finite_solutions_of_exponential_equation_l2290_229029

theorem finite_solutions_of_exponential_equation :
  ∃ (S : Finset (ℕ × ℕ × ℕ × ℕ)),
    ∀ (x y z n : ℕ), (2^x : ℕ) + 5^y - 31^z = n.factorial → (x, y, z, n) ∈ S := by
  sorry

end NUMINAMATH_CALUDE_finite_solutions_of_exponential_equation_l2290_229029


namespace NUMINAMATH_CALUDE_triangle_median_sum_bounds_l2290_229091

/-- For any triangle, the sum of its medians is greater than 3/4 of its perimeter
    but less than its perimeter. -/
theorem triangle_median_sum_bounds (a b c m_a m_b m_c : ℝ) 
  (h_positive : a > 0 ∧ b > 0 ∧ c > 0)
  (h_triangle : a + b > c ∧ b + c > a ∧ c + a > b)
  (h_median_a : m_a^2 = (2*b^2 + 2*c^2 - a^2) / 4)
  (h_median_b : m_b^2 = (2*c^2 + 2*a^2 - b^2) / 4)
  (h_median_c : m_c^2 = (2*a^2 + 2*b^2 - c^2) / 4) :
  3/4 * (a + b + c) < m_a + m_b + m_c ∧ m_a + m_b + m_c < a + b + c := by
sorry

end NUMINAMATH_CALUDE_triangle_median_sum_bounds_l2290_229091


namespace NUMINAMATH_CALUDE_secret_code_is_819_l2290_229032

/-- Represents a three-digit code -/
structure Code :=
  (d1 d2 d3 : Nat)
  (h1 : d1 < 10)
  (h2 : d2 < 10)
  (h3 : d3 < 10)

/-- Checks if a given digit is in the correct position in the code -/
def correctPosition (c : Code) (pos : Nat) (digit : Nat) : Prop :=
  match pos with
  | 1 => c.d1 = digit
  | 2 => c.d2 = digit
  | 3 => c.d3 = digit
  | _ => False

/-- Checks if a given digit is in the code but in the wrong position -/
def correctButWrongPosition (c : Code) (pos : Nat) (digit : Nat) : Prop :=
  (c.d1 = digit ∨ c.d2 = digit ∨ c.d3 = digit) ∧ ¬correctPosition c pos digit

/-- Represents the clues given in the problem -/
def clues (c : Code) : Prop :=
  (∃ d, (d = 0 ∨ d = 7 ∨ d = 9) ∧ (correctPosition c 1 d ∨ correctPosition c 2 d ∨ correctPosition c 3 d)) ∧
  (c.d1 ≠ 0 ∧ c.d2 ≠ 3 ∧ c.d3 ≠ 2) ∧
  (∃ d1 d2, (d1 = 1 ∨ d1 = 0 ∨ d1 = 8) ∧ (d2 = 1 ∨ d2 = 0 ∨ d2 = 8) ∧ d1 ≠ d2 ∧
    correctButWrongPosition c 1 d1 ∧ correctButWrongPosition c 2 d2) ∧
  (∃ d, (d = 9 ∨ d = 2 ∨ d = 6) ∧ correctButWrongPosition c 1 d) ∧
  (∃ d, (d = 6 ∨ d = 7 ∨ d = 8) ∧ correctButWrongPosition c 2 d)

theorem secret_code_is_819 : ∀ c : Code, clues c → c.d1 = 8 ∧ c.d2 = 1 ∧ c.d3 = 9 := by
  sorry

end NUMINAMATH_CALUDE_secret_code_is_819_l2290_229032


namespace NUMINAMATH_CALUDE_no_infinite_sequence_with_greater_than_neighbors_average_l2290_229062

theorem no_infinite_sequence_with_greater_than_neighbors_average :
  ¬ ∃ (a : ℕ → ℕ+), ∀ n : ℕ, n ≥ 1 → (a n : ℚ) > ((a (n - 1) : ℚ) + (a (n + 1) : ℚ)) / 2 :=
sorry

end NUMINAMATH_CALUDE_no_infinite_sequence_with_greater_than_neighbors_average_l2290_229062
