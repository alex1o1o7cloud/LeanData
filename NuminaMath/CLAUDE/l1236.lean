import Mathlib

namespace NUMINAMATH_CALUDE_situps_problem_l1236_123620

def total_situps (diana_rate : ℕ) (diana_total : ℕ) (hani_extra : ℕ) : ℕ :=
  let diana_time := diana_total / diana_rate
  let hani_rate := diana_rate + hani_extra
  let hani_total := hani_rate * diana_time
  diana_total + hani_total

theorem situps_problem :
  total_situps 4 40 3 = 110 :=
by sorry

end NUMINAMATH_CALUDE_situps_problem_l1236_123620


namespace NUMINAMATH_CALUDE_unique_number_with_specific_divisors_l1236_123618

theorem unique_number_with_specific_divisors : ∃! (N : ℕ),
  (5 ∣ N) ∧ (49 ∣ N) ∧ (Finset.card (Nat.divisors N) = 10) :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_unique_number_with_specific_divisors_l1236_123618


namespace NUMINAMATH_CALUDE_pants_cost_l1236_123631

theorem pants_cost (total_cost : ℕ) (tshirt_cost : ℕ) (num_tshirts : ℕ) (num_pants : ℕ) :
  total_cost = 1500 →
  tshirt_cost = 100 →
  num_tshirts = 5 →
  num_pants = 4 →
  (total_cost - num_tshirts * tshirt_cost) / num_pants = 250 := by
  sorry

end NUMINAMATH_CALUDE_pants_cost_l1236_123631


namespace NUMINAMATH_CALUDE_stating_regular_duck_price_is_correct_l1236_123689

/-- The price of a regular size rubber duck in the city's charity race. -/
def regular_duck_price : ℚ :=
  3

/-- The price of a large size rubber duck in the city's charity race. -/
def large_duck_price : ℚ :=
  5

/-- The number of regular size ducks sold in the charity race. -/
def regular_ducks_sold : ℕ :=
  221

/-- The number of large size ducks sold in the charity race. -/
def large_ducks_sold : ℕ :=
  185

/-- The total amount raised in the charity race. -/
def total_raised : ℚ :=
  1588

/-- 
Theorem stating that the regular duck price is correct given the conditions of the charity race.
-/
theorem regular_duck_price_is_correct :
  regular_duck_price * regular_ducks_sold + large_duck_price * large_ducks_sold = total_raised :=
by sorry

end NUMINAMATH_CALUDE_stating_regular_duck_price_is_correct_l1236_123689


namespace NUMINAMATH_CALUDE_simplify_expression_l1236_123619

theorem simplify_expression :
  ∃ (d e f : ℕ+), 
    (∀ (p : ℕ), Prime p → ¬(p^2 ∣ f.val)) ∧
    (((Real.sqrt 3 - 1) ^ (2 - Real.sqrt 2)) / ((Real.sqrt 3 + 1) ^ (2 + Real.sqrt 2)) = d.val - e.val * Real.sqrt f.val) ∧
    (d.val = 14 ∧ e.val = 8 ∧ f.val = 3) :=
by sorry

end NUMINAMATH_CALUDE_simplify_expression_l1236_123619


namespace NUMINAMATH_CALUDE_paint_house_time_l1236_123608

/-- Represents the time taken to paint a house given the number of workers and their efficiency -/
def paintTime (workers : ℕ) (efficiency : ℚ) (time : ℚ) : Prop :=
  (workers : ℚ) * efficiency * time = 40

theorem paint_house_time :
  paintTime 5 (4/5) 8 → paintTime 4 (4/5) 10 := by sorry

end NUMINAMATH_CALUDE_paint_house_time_l1236_123608


namespace NUMINAMATH_CALUDE_prisoner_release_time_l1236_123666

def prisoner_age : ℕ := 25
def warden_age : ℕ := 54

theorem prisoner_release_time : 
  ∃ (years : ℕ), warden_age + years = 2 * (prisoner_age + years) ∧ years = 4 :=
by sorry

end NUMINAMATH_CALUDE_prisoner_release_time_l1236_123666


namespace NUMINAMATH_CALUDE_work_completion_time_l1236_123685

-- Define the work capacity ratio of A to B
def ratio_A_to_B : ℚ := 3 / 2

-- Define the time A takes to complete the work alone
def time_A_alone : ℕ := 45

-- Define the function to calculate the time taken when A and B work together
def time_together (ratio : ℚ) (time_alone : ℕ) : ℚ :=
  (ratio * time_alone) / (ratio + 1)

-- Theorem statement
theorem work_completion_time :
  time_together ratio_A_to_B time_A_alone = 27 := by
  sorry

end NUMINAMATH_CALUDE_work_completion_time_l1236_123685


namespace NUMINAMATH_CALUDE_line_equation_proof_l1236_123650

/-- Represents a line in 2D space -/
structure Line where
  slope : ℝ
  yIntercept : ℝ

/-- Checks if a point (x, y) lies on the line -/
def Line.containsPoint (l : Line) (x y : ℝ) : Prop :=
  y = l.slope * x + l.yIntercept

/-- The line we're interested in -/
def ourLine : Line :=
  { slope := 2, yIntercept := 5 }

theorem line_equation_proof :
  (∀ x y : ℝ, ourLine.containsPoint x y ↔ -2 * x + y = 1) ∧
  ourLine.containsPoint (-2) 3 :=
sorry

end NUMINAMATH_CALUDE_line_equation_proof_l1236_123650


namespace NUMINAMATH_CALUDE_ufo_convention_attendees_l1236_123677

theorem ufo_convention_attendees :
  let total_attendees : ℕ := 450
  let male_female_difference : ℕ := 26
  let male_attendees : ℕ := (total_attendees + male_female_difference) / 2
  male_attendees = 238 :=
by sorry

end NUMINAMATH_CALUDE_ufo_convention_attendees_l1236_123677


namespace NUMINAMATH_CALUDE_geometric_sequence_common_ratio_l1236_123686

def is_geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, ∀ n : ℕ, a (n + 1) = a n * q

theorem geometric_sequence_common_ratio 
  (a : ℕ → ℝ) 
  (h_positive : ∀ n, a n > 0)
  (h_a2 : a 2 = 1)
  (h_a4a6 : a 4 * a 6 = 64) :
  ∃ q : ℝ, is_geometric_sequence a ∧ q = 2 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_common_ratio_l1236_123686


namespace NUMINAMATH_CALUDE_smallest_class_size_l1236_123699

/-- Represents the number of students in a physical education class. -/
def class_size (x : ℕ) : ℕ := 5 * x + 3

/-- Theorem stating the smallest possible class size satisfying the given conditions. -/
theorem smallest_class_size :
  ∀ n : ℕ, class_size n > 50 → class_size 10 ≤ class_size n :=
by
  sorry

#eval class_size 10  -- Should output 53

end NUMINAMATH_CALUDE_smallest_class_size_l1236_123699


namespace NUMINAMATH_CALUDE_at_least_two_boundary_triangles_l1236_123679

/-- A polygon divided into triangles by non-intersecting diagonals -/
structure TriangulatedPolygon where
  /-- The number of sides of the polygon -/
  n : ℕ
  /-- The number of triangles with exactly i sides as sides of the polygon -/
  k : Fin 3 → ℕ
  /-- The total number of triangles is n - 2 -/
  total_triangles : k 0 + k 1 + k 2 = n - 2
  /-- The total number of polygon sides used in forming triangles is n -/
  total_sides : 2 * k 2 + k 1 = n

/-- 
In a polygon divided into triangles by non-intersecting diagonals, 
there are at least two triangles that have at least two sides 
coinciding with the sides of the original polygon.
-/
theorem at_least_two_boundary_triangles (p : TriangulatedPolygon) : 
  p.k 2 ≥ 2 := by
  sorry

end NUMINAMATH_CALUDE_at_least_two_boundary_triangles_l1236_123679


namespace NUMINAMATH_CALUDE_partnership_profit_l1236_123603

/-- Calculates the total profit of a partnership given the investments and one partner's profit share -/
theorem partnership_profit
  (a_investment b_investment c_investment : ℕ)
  (c_profit_share : ℕ)
  (h1 : a_investment = 30000)
  (h2 : b_investment = 45000)
  (h3 : c_investment = 50000)
  (h4 : c_profit_share = 36000) :
  ∃ (total_profit : ℕ), total_profit = 90000 ∧
    total_profit * c_investment = (a_investment + b_investment + c_investment) * c_profit_share :=
by sorry

end NUMINAMATH_CALUDE_partnership_profit_l1236_123603


namespace NUMINAMATH_CALUDE_unique_nonzero_elements_in_rows_and_columns_l1236_123665

open Matrix

theorem unique_nonzero_elements_in_rows_and_columns
  (n : ℕ)
  (A : Matrix (Fin n) (Fin n) ℝ)
  (h_nonneg : ∀ i j, 0 ≤ A i j)
  (h_nonsingular : IsUnit (det A))
  (h_inv_nonneg : ∀ i j, 0 ≤ A⁻¹ i j) :
  (∀ i, ∃! j, A i j ≠ 0) ∧ (∀ j, ∃! i, A i j ≠ 0) := by
  sorry

end NUMINAMATH_CALUDE_unique_nonzero_elements_in_rows_and_columns_l1236_123665


namespace NUMINAMATH_CALUDE_sqrt_simplification_l1236_123651

theorem sqrt_simplification :
  (Real.sqrt 18 - Real.sqrt 2) / Real.sqrt 2 = 2 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_simplification_l1236_123651


namespace NUMINAMATH_CALUDE_shortest_halving_segment_345_triangle_l1236_123621

/-- Triangle with sides a, b, c -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ

/-- The shortest segment that halves the area of a triangle -/
def shortestHalvingSegment (t : Triangle) : ℝ :=
  sorry

/-- Theorem: The shortest segment that halves the area of a 3-4-5 triangle has length 2 -/
theorem shortest_halving_segment_345_triangle :
  let t : Triangle := { a := 3, b := 4, c := 5 }
  shortestHalvingSegment t = 2 := by
  sorry

end NUMINAMATH_CALUDE_shortest_halving_segment_345_triangle_l1236_123621


namespace NUMINAMATH_CALUDE_range_of_S_l1236_123690

theorem range_of_S (a b : ℝ) 
  (h : ∀ x ∈ Set.Icc 0 1, |a * x + b| ≤ 1) : 
  ∃ S : ℝ, S = (a + 1) * (b + 1) ∧ -2 ≤ S ∧ S ≤ 9/4 :=
sorry

end NUMINAMATH_CALUDE_range_of_S_l1236_123690


namespace NUMINAMATH_CALUDE_equation_equivalence_l1236_123656

theorem equation_equivalence (x : ℚ) : (x - 1) / 2 - x / 5 = 1 ↔ 5 * (x - 1) - 2 * x = 10 := by
  sorry

end NUMINAMATH_CALUDE_equation_equivalence_l1236_123656


namespace NUMINAMATH_CALUDE_distribute_6_4_l1236_123637

/-- The number of ways to distribute n distinguishable balls into k indistinguishable boxes -/
def distribute (n k : ℕ) : ℕ :=
  sorry

/-- The main theorem stating that there are 182 ways to distribute 6 distinguishable balls into 4 indistinguishable boxes -/
theorem distribute_6_4 : distribute 6 4 = 182 := by
  sorry

end NUMINAMATH_CALUDE_distribute_6_4_l1236_123637


namespace NUMINAMATH_CALUDE_min_perimeter_triangle_l1236_123698

theorem min_perimeter_triangle (a b : ℝ) (h1 : 0 < b) (h2 : b < a) :
  let min_perimeter := Real.sqrt (2 * (a^2 + b^2))
  ∀ c d : ℝ, (c ≥ 0) → (d ≥ 0) → 
    Real.sqrt ((a - c)^2 + b^2) + Real.sqrt ((d - c)^2 + d^2) + Real.sqrt ((a - d)^2 + (b - d)^2) ≥ min_perimeter :=
by sorry


end NUMINAMATH_CALUDE_min_perimeter_triangle_l1236_123698


namespace NUMINAMATH_CALUDE_concert_probability_at_least_seven_concert_probability_is_one_ninth_l1236_123625

/-- The probability that at least 7 out of 8 people stay for an entire concert,
    given that 4 are certain to stay and 4 have a 1/3 probability of staying. -/
theorem concert_probability_at_least_seven (total_people : Nat) (certain_people : Nat)
    (uncertain_people : Nat) (stay_prob : ℚ) : ℚ :=
  let total_people := 8
  let certain_people := 4
  let uncertain_people := 4
  let stay_prob := 1 / 3
  1 / 9

theorem concert_probability_is_one_ninth :
    concert_probability_at_least_seven 8 4 4 (1 / 3) = 1 / 9 := by
  sorry

end NUMINAMATH_CALUDE_concert_probability_at_least_seven_concert_probability_is_one_ninth_l1236_123625


namespace NUMINAMATH_CALUDE_min_value_expression_l1236_123674

theorem min_value_expression (x y : ℝ) (hx : x > 0) (hy : y > 0) :
  (x / (x + 2 * y)) + (y / x) ≥ Real.sqrt 2 - 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_min_value_expression_l1236_123674


namespace NUMINAMATH_CALUDE_division_problem_l1236_123661

theorem division_problem (divisor : ℕ) : 
  22 = divisor * 7 + 1 → divisor = 3 := by
  sorry

end NUMINAMATH_CALUDE_division_problem_l1236_123661


namespace NUMINAMATH_CALUDE_sum_of_y_values_l1236_123612

/-- Given 5 sets of data points, prove the sum of y values -/
theorem sum_of_y_values 
  (x₁ x₂ x₃ x₄ x₅ y₁ y₂ y₃ y₄ y₅ : ℝ) 
  (h_sum_x : x₁ + x₂ + x₃ + x₄ + x₅ = 150) 
  (h_regression : ∀ x, (x₁ + x₂ + x₃ + x₄ + x₅) / 5 = x → 
    (y₁ + y₂ + y₃ + y₄ + y₅) / 5 = 0.67 * x + 24.9) : 
  y₁ + y₂ + y₃ + y₄ + y₅ = 225 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_y_values_l1236_123612


namespace NUMINAMATH_CALUDE_sum_of_complex_numbers_l1236_123638

theorem sum_of_complex_numbers :
  let z₁ : ℂ := Complex.mk 2 5
  let z₂ : ℂ := Complex.mk 3 (-7)
  z₁ + z₂ = Complex.mk 5 (-2) :=
by sorry

end NUMINAMATH_CALUDE_sum_of_complex_numbers_l1236_123638


namespace NUMINAMATH_CALUDE_exam_score_proof_l1236_123602

theorem exam_score_proof (score1 score2 score3 score4 : ℕ) : 
  score1 = 70 → score2 = 80 → score3 = 90 → 
  (score1 + score2 + score3 + score4) / 4 = 70 → 
  score4 = 40 := by
sorry

end NUMINAMATH_CALUDE_exam_score_proof_l1236_123602


namespace NUMINAMATH_CALUDE_koschei_coins_theorem_l1236_123648

theorem koschei_coins_theorem :
  ∃! n : ℕ, 300 ≤ n ∧ n ≤ 400 ∧ n % 10 = 7 ∧ n % 12 = 9 :=
by sorry

end NUMINAMATH_CALUDE_koschei_coins_theorem_l1236_123648


namespace NUMINAMATH_CALUDE_alices_favorite_number_l1236_123697

def is_favorite_number (n : ℕ) : Prop :=
  50 ≤ n ∧ n ≤ 100 ∧ 
  n % 11 = 0 ∧
  n % 2 ≠ 0 ∧
  (n / 10 + n % 10) % 5 = 0

theorem alices_favorite_number :
  ∃! n : ℕ, is_favorite_number n ∧ n = 55 := by
sorry

end NUMINAMATH_CALUDE_alices_favorite_number_l1236_123697


namespace NUMINAMATH_CALUDE_molecular_weight_not_affects_l1236_123680

-- Define plasma osmotic pressure
def plasma_osmotic_pressure : ℝ → ℝ := sorry

-- Define factors that affect plasma osmotic pressure
def protein_content : ℝ := sorry
def cl_content : ℝ := sorry
def na_content : ℝ := sorry
def molecular_weight_protein : ℝ := sorry

-- State that protein content affects plasma osmotic pressure
axiom protein_content_affects : ∃ (ε : ℝ), ε ≠ 0 ∧ 
  plasma_osmotic_pressure (protein_content + ε) ≠ plasma_osmotic_pressure protein_content

-- State that Cl- content affects plasma osmotic pressure
axiom cl_content_affects : ∃ (ε : ℝ), ε ≠ 0 ∧ 
  plasma_osmotic_pressure (cl_content + ε) ≠ plasma_osmotic_pressure cl_content

-- State that Na+ content affects plasma osmotic pressure
axiom na_content_affects : ∃ (ε : ℝ), ε ≠ 0 ∧ 
  plasma_osmotic_pressure (na_content + ε) ≠ plasma_osmotic_pressure na_content

-- Theorem: Molecular weight of plasma protein does not affect plasma osmotic pressure
theorem molecular_weight_not_affects : ∀ (ε : ℝ), ε ≠ 0 → 
  plasma_osmotic_pressure (molecular_weight_protein + ε) = plasma_osmotic_pressure molecular_weight_protein :=
sorry

end NUMINAMATH_CALUDE_molecular_weight_not_affects_l1236_123680


namespace NUMINAMATH_CALUDE_difference_exists_l1236_123604

def is_valid_sequence (x : ℕ → ℕ) : Prop :=
  x 1 = 1 ∧ ∀ n : ℕ, n ≥ 1 → x n < x (n + 1) ∧ x (n + 1) ≤ 2 * n

theorem difference_exists (x : ℕ → ℕ) (h : is_valid_sequence x) :
  ∀ k : ℕ, ∃ r s : ℕ, x r - x s = k :=
sorry

end NUMINAMATH_CALUDE_difference_exists_l1236_123604


namespace NUMINAMATH_CALUDE_algebra_class_size_l1236_123639

/-- Given an Algebra 1 class where there are 11 girls and 5 fewer girls than boys,
    prove that the total number of students in the class is 27. -/
theorem algebra_class_size :
  ∀ (num_girls num_boys : ℕ),
    num_girls = 11 →
    num_girls + 5 = num_boys →
    num_girls + num_boys = 27 :=
by
  sorry

end NUMINAMATH_CALUDE_algebra_class_size_l1236_123639


namespace NUMINAMATH_CALUDE_savings_difference_l1236_123660

/-- The original price of the office supplies -/
def original_price : ℝ := 15000

/-- The first discount rate in the successive discounts option -/
def discount1 : ℝ := 0.30

/-- The second discount rate in the successive discounts option -/
def discount2 : ℝ := 0.15

/-- The single discount rate in the alternative option -/
def single_discount : ℝ := 0.40

/-- The price after applying two successive discounts -/
def price_after_successive_discounts : ℝ :=
  original_price * (1 - discount1) * (1 - discount2)

/-- The price after applying a single discount -/
def price_after_single_discount : ℝ :=
  original_price * (1 - single_discount)

/-- Theorem stating the difference in savings between the two discount options -/
theorem savings_difference :
  price_after_single_discount - price_after_successive_discounts = 75 := by
  sorry

end NUMINAMATH_CALUDE_savings_difference_l1236_123660


namespace NUMINAMATH_CALUDE_equation_roots_imply_m_range_l1236_123652

theorem equation_roots_imply_m_range (m : ℝ) : 
  (∃ x : ℝ, x^2 + 4*m*x + 4*m^2 + 2*m + 3 = 0 ∨ x^2 + (2*m + 1)*x + m^2 = 0) → 
  (m ≤ -3/2 ∨ m ≥ -1/4) := by
sorry

end NUMINAMATH_CALUDE_equation_roots_imply_m_range_l1236_123652


namespace NUMINAMATH_CALUDE_circle_line_symmetry_l1236_123668

/-- A circle in the 2D plane -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- A line in the 2D plane of the form ax + by + c = 0 -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Two points are symmetric with respect to a line -/
def symmetric_points (p q : ℝ × ℝ) (l : Line) : Prop :=
  sorry

theorem circle_line_symmetry (c : Circle) (l : Line) :
  c.center.1 = -1 ∧ c.center.2 = 3 ∧ c.radius = 3 ∧
  l.a = 1 ∧ l.c = 4 ∧
  ∃ (p q : ℝ × ℝ), (p.1 + 1)^2 + (p.2 - 3)^2 = 9 ∧
                   (q.1 + 1)^2 + (q.2 - 3)^2 = 9 ∧
                   symmetric_points p q l →
  l.b = -1 :=
sorry

end NUMINAMATH_CALUDE_circle_line_symmetry_l1236_123668


namespace NUMINAMATH_CALUDE_copper_in_mixture_l1236_123613

theorem copper_in_mixture (lead_percentage : Real) (copper_percentage : Real) (lead_mass : Real) (copper_mass : Real) : 
  lead_percentage = 0.25 →
  copper_percentage = 0.60 →
  lead_mass = 5 →
  copper_mass = 12 →
  copper_mass = (copper_percentage / lead_percentage) * lead_mass :=
by
  sorry

#check copper_in_mixture

end NUMINAMATH_CALUDE_copper_in_mixture_l1236_123613


namespace NUMINAMATH_CALUDE_min_distance_line_circle_min_distance_specific_line_circle_l1236_123655

/-- Given a line and a circle in a 2D plane, this theorem states that 
    the minimum distance between any point on the line and any point on the circle 
    is equal to the difference between the distance from the circle's center 
    to the line and the radius of the circle. -/
theorem min_distance_line_circle (a b c d e f : ℝ) :
  let line := {p : ℝ × ℝ | a * p.1 + b * p.2 + c = 0}
  let circle := {p : ℝ × ℝ | (p.1 - d)^2 + (p.2 - e)^2 = f^2}
  let center := (d, e)
  let radius := f
  let dist_center_to_line := |a * d + b * e + c| / Real.sqrt (a^2 + b^2)
  ∃ (p : ℝ × ℝ) (q : ℝ × ℝ), p ∈ line ∧ q ∈ circle ∧
    ∀ (p' : ℝ × ℝ) (q' : ℝ × ℝ), p' ∈ line → q' ∈ circle →
      dist_center_to_line - radius ≤ Real.sqrt ((p'.1 - q'.1)^2 + (p'.2 - q'.2)^2) :=
by sorry

/-- The minimum distance between a point on the line 2x + y - 6 = 0
    and a point on the circle (x-1)² + (y+2)² = 5 is √5/5. -/
theorem min_distance_specific_line_circle :
  let line := {p : ℝ × ℝ | 2 * p.1 + p.2 - 6 = 0}
  let circle := {p : ℝ × ℝ | (p.1 - 1)^2 + (p.2 + 2)^2 = 5}
  ∃ (p : ℝ × ℝ) (q : ℝ × ℝ), p ∈ line ∧ q ∈ circle ∧
    ∀ (p' : ℝ × ℝ) (q' : ℝ × ℝ), p' ∈ line → q' ∈ circle →
      Real.sqrt 5 / 5 ≤ Real.sqrt ((p'.1 - q'.1)^2 + (p'.2 - q'.2)^2) :=
by sorry

end NUMINAMATH_CALUDE_min_distance_line_circle_min_distance_specific_line_circle_l1236_123655


namespace NUMINAMATH_CALUDE_function_periodicity_l1236_123632

/-- A function satisfying the given functional equation is periodic with period 4k -/
theorem function_periodicity (f : ℝ → ℝ) (k : ℝ) (hk : k ≠ 0) 
  (h : ∀ x, f (x + k) * (1 - f x) = 1 + f x) : 
  ∀ x, f (x + 4 * k) = f x := by
  sorry

end NUMINAMATH_CALUDE_function_periodicity_l1236_123632


namespace NUMINAMATH_CALUDE_exists_bijection_sum_inverse_neg_l1236_123615

theorem exists_bijection_sum_inverse_neg : 
  ∃ (f : ℝ → ℝ), Function.Bijective f ∧ ∀ x : ℝ, f x + (Function.invFun f) x = -x := by
  sorry

end NUMINAMATH_CALUDE_exists_bijection_sum_inverse_neg_l1236_123615


namespace NUMINAMATH_CALUDE_matrix_identities_l1236_123669

variable {n : ℕ} (hn : n ≥ 2)
variable (k : ℝ)
variable (A B C D : Matrix (Fin n) (Fin n) ℂ)

theorem matrix_identities 
  (h1 : A * C + k • (B * D) = 1)
  (h2 : A * D = B * C) :
  C * A + k • (D * B) = 1 ∧ D * A = C * B := by
sorry

end NUMINAMATH_CALUDE_matrix_identities_l1236_123669


namespace NUMINAMATH_CALUDE_cubic_inequality_l1236_123688

theorem cubic_inequality (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  2 * (a^3 + b^3 + c^3) ≥ a*b^2 + a^2*b + b*c^2 + b^2*c + a*c^2 + a^2*c := by
  sorry

end NUMINAMATH_CALUDE_cubic_inequality_l1236_123688


namespace NUMINAMATH_CALUDE_tournament_sequences_l1236_123636

theorem tournament_sequences (n : ℕ) : (2 * n).choose n = 3432 :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_CALUDE_tournament_sequences_l1236_123636


namespace NUMINAMATH_CALUDE_fraction_subtraction_l1236_123682

theorem fraction_subtraction : 
  (2 + 4 + 6) / (1 + 3 + 5) - (1 + 3 + 5) / (2 + 4 + 6) = 7 / 12 := by
  sorry

end NUMINAMATH_CALUDE_fraction_subtraction_l1236_123682


namespace NUMINAMATH_CALUDE_probability_intersection_independent_events_l1236_123675

theorem probability_intersection_independent_events 
  (p : Set α → ℝ) (a b : Set α) 
  (ha : p a = 4/5) 
  (hb : p b = 2/5) 
  (hab_indep : p (a ∩ b) = p a * p b) : 
  p (a ∩ b) = 8/25 := by
sorry

end NUMINAMATH_CALUDE_probability_intersection_independent_events_l1236_123675


namespace NUMINAMATH_CALUDE_polynomial_coefficient_sum_l1236_123624

theorem polynomial_coefficient_sum (a a₁ a₂ a₃ a₄ : ℝ) :
  (∀ x, (1 - 2*x)^4 = a + a₁*x + a₂*x^2 + a₃*x^3 + a₄*x^4) →
  a₁ - 2*a₂ + 3*a₃ - 4*a₄ = -216 := by
sorry

end NUMINAMATH_CALUDE_polynomial_coefficient_sum_l1236_123624


namespace NUMINAMATH_CALUDE_triangle_theorem_l1236_123607

-- Define the triangle ABC
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

-- Define the conditions
def condition1 (t : Triangle) : Prop :=
  t.c * Real.sin t.B + 2 * Real.cos t.A = t.b * Real.sin t.C + 1

def condition2 (t : Triangle) : Prop :=
  Real.cos (2 * t.A) - 3 * Real.cos (t.B + t.C) - 1 = 0

def condition3 (t : Triangle) : Prop :=
  ∃ k : ℝ, k * Real.sqrt 3 * t.b = t.a * Real.sin t.B ∧ k * t.a = Real.cos t.A

-- Define the theorem
theorem triangle_theorem (t : Triangle) 
  (h : condition1 t ∨ condition2 t ∨ condition3 t) : 
  t.A = Real.pi / 3 ∧ 
  (t.a * t.b * Real.sin t.C / 2 = Real.sqrt 3 / 2 → t.a ≥ Real.sqrt 2) :=
sorry

end NUMINAMATH_CALUDE_triangle_theorem_l1236_123607


namespace NUMINAMATH_CALUDE_geometric_sequence_from_arithmetic_l1236_123643

/-- Given a geometric sequence {b_n} where b_1 = 3, and whose 7th, 10th, and 15th terms
    form consecutive terms of an arithmetic sequence with non-zero common difference,
    prove that the general form of b_n is 3 * (5/3)^(n-1). -/
theorem geometric_sequence_from_arithmetic (b : ℕ → ℚ) (d : ℚ) :
  b 1 = 3 →
  d ≠ 0 →
  (∃ a : ℚ, b 7 = a + 6 * d ∧ b 10 = a + 9 * d ∧ b 15 = a + 14 * d) →
  (∀ n : ℕ, b n = 3 * (5/3)^(n-1)) :=
by sorry

end NUMINAMATH_CALUDE_geometric_sequence_from_arithmetic_l1236_123643


namespace NUMINAMATH_CALUDE_max_value_of_squares_l1236_123640

theorem max_value_of_squares (x y z : ℤ) 
  (eq1 : x * y + x + y = 20)
  (eq2 : y * z + y + z = 6)
  (eq3 : x * z + x + z = 2) :
  ∃ (a b c : ℤ), a * b + a + b = 20 ∧ b * c + b + c = 6 ∧ a * c + a + c = 2 ∧
    ∀ (x y z : ℤ), x * y + x + y = 20 → y * z + y + z = 6 → x * z + x + z = 2 →
      x^2 + y^2 + z^2 ≤ a^2 + b^2 + c^2 ∧ a^2 + b^2 + c^2 = 84 :=
sorry

end NUMINAMATH_CALUDE_max_value_of_squares_l1236_123640


namespace NUMINAMATH_CALUDE_equation_transformation_l1236_123673

theorem equation_transformation (x y : ℝ) : x - 2 = y - 2 → x = y := by
  sorry

end NUMINAMATH_CALUDE_equation_transformation_l1236_123673


namespace NUMINAMATH_CALUDE_sum_of_exponents_outside_radical_l1236_123630

-- Define the expression
def original_expression (a b c : ℝ) : ℝ := (24 * a^4 * b^6 * c^11) ^ (1/3)

-- Define the simplified expression
def simplified_expression (a b c : ℝ) : ℝ := 2 * a * b^2 * c^3 * ((3 * a * c^2) ^ (1/3))

-- State the theorem
theorem sum_of_exponents_outside_radical :
  ∀ a b c : ℝ, a ≠ 0 → b ≠ 0 → c ≠ 0 →
  original_expression a b c = simplified_expression a b c ∧
  (1 + 2 + 3 = 6) := by sorry

end NUMINAMATH_CALUDE_sum_of_exponents_outside_radical_l1236_123630


namespace NUMINAMATH_CALUDE_cheryls_mms_l1236_123614

/-- Cheryl's M&M's Problem -/
theorem cheryls_mms (initial : ℕ) (after_lunch : ℕ) (after_dinner : ℕ) (given_to_sister : ℕ) :
  initial = 25 →
  after_lunch = 7 →
  after_dinner = 5 →
  given_to_sister = initial - (after_lunch + after_dinner) →
  given_to_sister = 13 := by
sorry

end NUMINAMATH_CALUDE_cheryls_mms_l1236_123614


namespace NUMINAMATH_CALUDE_apple_count_theorem_l1236_123672

/-- The number of apples originally on the tree -/
def original_apples : ℕ := 9

/-- The number of apples picked from the tree -/
def picked_apples : ℕ := 2

/-- The number of apples remaining on the tree -/
def remaining_apples : ℕ := 7

/-- Theorem stating that the original number of apples is equal to
    the sum of remaining and picked apples -/
theorem apple_count_theorem :
  original_apples = remaining_apples + picked_apples :=
by sorry

end NUMINAMATH_CALUDE_apple_count_theorem_l1236_123672


namespace NUMINAMATH_CALUDE_line_l_equation_l1236_123696

-- Define the points and lines
def P : ℝ × ℝ := (-1, 1)
def l₁ : Set (ℝ × ℝ) := {(x, y) | x + 2*y - 5 = 0}
def l₂ : Set (ℝ × ℝ) := {(x, y) | x + 2*y - 3 = 0}
def l₃ : Set (ℝ × ℝ) := {(x, y) | x - y - 1 = 0}

-- Define the line l (we'll prove this is correct)
def l : Set (ℝ × ℝ) := {(x, y) | y = 1}

-- Define the properties of the problem
theorem line_l_equation (M : ℝ × ℝ) :
  P ∈ l ∧  -- l passes through P
  (∃ M₁ M₂, M₁ ∈ l ∩ l₁ ∧ M₂ ∈ l ∩ l₂ ∧  -- l intersects l₁ and l₂
    M = ((M₁.1 + M₂.1) / 2, (M₁.2 + M₂.2) / 2)) ∧  -- M is the midpoint of M₁M₂
  M ∈ l₃  -- M lies on l₃
  → l = {(x, y) | y = 1} :=
by sorry

end NUMINAMATH_CALUDE_line_l_equation_l1236_123696


namespace NUMINAMATH_CALUDE_carwash_problem_l1236_123633

/-- Represents the number of vehicles of each type washed --/
structure VehicleCounts where
  cars : ℕ
  trucks : ℕ
  suvs : ℕ

/-- Represents the prices for washing each type of vehicle --/
structure WashPrices where
  car : ℕ
  truck : ℕ
  suv : ℕ

/-- Calculates the total amount raised from a car wash --/
def totalRaised (counts : VehicleCounts) (prices : WashPrices) : ℕ :=
  counts.cars * prices.car + counts.trucks * prices.truck + counts.suvs * prices.suv

/-- The main theorem to prove --/
theorem carwash_problem (prices : WashPrices) 
    (h_car_price : prices.car = 5)
    (h_truck_price : prices.truck = 6)
    (h_suv_price : prices.suv = 7)
    (h_total : totalRaised { cars := 7, trucks := 5, suvs := 5 } prices = 100) :
  ∃ (n : ℕ), n = 7 ∧ 
    totalRaised { cars := n, trucks := 5, suvs := 5 } prices = 100 :=
by
  sorry


end NUMINAMATH_CALUDE_carwash_problem_l1236_123633


namespace NUMINAMATH_CALUDE_park_length_l1236_123627

/-- Given a rectangular park with width 9 km and perimeter 46 km, its length is 14 km. -/
theorem park_length (width : ℝ) (perimeter : ℝ) (length : ℝ) : 
  width = 9 → perimeter = 46 → perimeter = 2 * (length + width) → length = 14 := by
  sorry

end NUMINAMATH_CALUDE_park_length_l1236_123627


namespace NUMINAMATH_CALUDE_bread_waste_savings_l1236_123695

/-- Represents the daily bread waste and cost parameters -/
structure BreadWaste where
  pieces_per_day : ℕ
  pieces_per_loaf : ℕ
  cost_per_loaf : ℕ

/-- Calculates the money saved over a given number of days -/
def money_saved (waste : BreadWaste) (days : ℕ) : ℕ :=
  (days * waste.pieces_per_day * waste.cost_per_loaf) / (2 * waste.pieces_per_loaf)

/-- Theorem stating the money saved in 20 and 60 days -/
theorem bread_waste_savings (waste : BreadWaste) 
  (h1 : waste.pieces_per_day = 7)
  (h2 : waste.pieces_per_loaf = 14)
  (h3 : waste.cost_per_loaf = 35) :
  money_saved waste 20 = 350 ∧ money_saved waste 60 = 1050 := by
  sorry

#eval money_saved ⟨7, 14, 35⟩ 20
#eval money_saved ⟨7, 14, 35⟩ 60

end NUMINAMATH_CALUDE_bread_waste_savings_l1236_123695


namespace NUMINAMATH_CALUDE_stratified_sampling_male_athletes_l1236_123647

theorem stratified_sampling_male_athletes :
  let total_athletes : ℕ := 28 + 21
  let male_athletes : ℕ := 28
  let sample_size : ℕ := 14
  let selected_male_athletes : ℕ := (male_athletes * sample_size) / total_athletes
  selected_male_athletes = 8 := by
  sorry

end NUMINAMATH_CALUDE_stratified_sampling_male_athletes_l1236_123647


namespace NUMINAMATH_CALUDE_product_equality_l1236_123611

theorem product_equality (x y : ℝ) :
  (3 * x^2 - 4 * y^3) * (9 * x^4 + 12 * x^2 * y^3 + 16 * y^6) = 27 * x^6 - 64 * y^9 := by
  sorry

end NUMINAMATH_CALUDE_product_equality_l1236_123611


namespace NUMINAMATH_CALUDE_perfect_squares_from_equation_l1236_123678

theorem perfect_squares_from_equation (x y : ℕ) (h : 2 * x^2 + x = 3 * y^2 + y) :
  ∃ (a b c : ℕ), (x - y = a^2) ∧ (2 * x + 2 * y + 1 = b^2) ∧ (3 * x + 3 * y + 1 = c^2) := by
  sorry

end NUMINAMATH_CALUDE_perfect_squares_from_equation_l1236_123678


namespace NUMINAMATH_CALUDE_barons_claim_l1236_123671

/-- Define the type of weight sets -/
def WeightSet := Fin 1000 → ℕ

/-- The condition that all weights are different -/
def all_different (w : WeightSet) : Prop :=
  ∀ i j, i ≠ j → w i ≠ w j

/-- The sum of one of each weight -/
def sum_of_weights (w : WeightSet) : ℕ :=
  Finset.sum Finset.univ (λ i => w i)

/-- The uniqueness of the sum -/
def unique_sum (w : WeightSet) : Prop :=
  ∀ s : Finset (Fin 1000), s.card < 1000 → Finset.sum s (λ i => w i) ≠ sum_of_weights w

/-- The main theorem -/
theorem barons_claim :
  ∃ w : WeightSet,
    all_different w ∧
    sum_of_weights w < 2^1010 ∧
    unique_sum w :=
  sorry

end NUMINAMATH_CALUDE_barons_claim_l1236_123671


namespace NUMINAMATH_CALUDE_water_remaining_l1236_123600

theorem water_remaining (initial_water : ℚ) (used_water : ℚ) : 
  initial_water = 7/2 ∧ used_water = 7/3 → initial_water - used_water = 7/6 := by
  sorry

#check water_remaining

end NUMINAMATH_CALUDE_water_remaining_l1236_123600


namespace NUMINAMATH_CALUDE_benny_pumpkin_pies_l1236_123683

/-- Represents the number of pumpkin pies Benny plans to make -/
def num_pumpkin_pies : ℕ := sorry

/-- The cost to make one pumpkin pie -/
def pumpkin_pie_cost : ℕ := 3

/-- The cost to make one cherry pie -/
def cherry_pie_cost : ℕ := 5

/-- The number of cherry pies Benny plans to make -/
def num_cherry_pies : ℕ := 12

/-- The profit Benny wants to make -/
def desired_profit : ℕ := 20

/-- The price Benny charges for each pie -/
def pie_price : ℕ := 5

/-- Theorem stating that the number of pumpkin pies Benny plans to make is 10 -/
theorem benny_pumpkin_pies : 
  num_pumpkin_pies = 10 := by
  sorry

end NUMINAMATH_CALUDE_benny_pumpkin_pies_l1236_123683


namespace NUMINAMATH_CALUDE_quadratic_expression_value_l1236_123622

theorem quadratic_expression_value : (3^2 : ℝ) - 3*3 + 2 = 2 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_expression_value_l1236_123622


namespace NUMINAMATH_CALUDE_new_ratio_after_adding_water_l1236_123693

/-- Given a mixture of alcohol and water with an initial ratio and known quantities,
    this theorem proves the new ratio after adding water. -/
theorem new_ratio_after_adding_water
  (initial_alcohol : ℝ)
  (initial_water : ℝ)
  (added_water : ℝ)
  (h1 : initial_alcohol / initial_water = 4 / 3)
  (h2 : initial_alcohol = 20)
  (h3 : added_water = 4) :
  initial_alcohol / (initial_water + added_water) = 20 / 19 := by
  sorry

end NUMINAMATH_CALUDE_new_ratio_after_adding_water_l1236_123693


namespace NUMINAMATH_CALUDE_quadratic_function_properties_l1236_123634

/- Define the quadratic function f(x) -/
def f (x : ℝ) := 2 * x^2 - 10 * x

/- Theorem stating the properties of f(x) and the solution sets -/
theorem quadratic_function_properties :
  (∀ x, f x < 0 ↔ 0 < x ∧ x < 5) ∧
  (∀ x ∈ Set.Icc (-1) 4, f x ≤ 12) ∧
  (∃ x ∈ Set.Icc (-1) 4, f x = 12) ∧
  (∀ a < 0,
    (∀ x, (2 * x^2 + (a - 10) * x + 5) / f x > 1 ↔
      ((-1 < a ∧ a < 0 ∧ (x < 0 ∨ (5 < x ∧ x < -5/a))) ∨
       (a = -1 ∧ x < 0) ∨
       (a < -1 ∧ (x < 0 ∨ (-5/a < x ∧ x < 5))))))
  := by sorry

end NUMINAMATH_CALUDE_quadratic_function_properties_l1236_123634


namespace NUMINAMATH_CALUDE_functional_equation_solution_l1236_123658

/-- A function satisfying g(xy) = xg(y) for all real numbers x and y -/
def FunctionalEquation (g : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, g (x * y) = x * g y

theorem functional_equation_solution (g : ℝ → ℝ) 
  (h1 : FunctionalEquation g) (h2 : g 1 = 30) : 
  g 50 = 1500 ∧ g 0.5 = 15 := by
  sorry

end NUMINAMATH_CALUDE_functional_equation_solution_l1236_123658


namespace NUMINAMATH_CALUDE_seven_by_six_grid_paths_l1236_123623

/-- The number of paths on a grid from bottom-left to top-right -/
def grid_paths (width height : ℕ) : ℕ :=
  Nat.choose (width + height) height

theorem seven_by_six_grid_paths :
  grid_paths 7 6 = 1716 := by
  sorry

end NUMINAMATH_CALUDE_seven_by_six_grid_paths_l1236_123623


namespace NUMINAMATH_CALUDE_solve_dining_problem_l1236_123684

def dining_problem (total_bill : ℚ) (tip_percentage : ℚ) (individual_share : ℚ) : Prop :=
  let tip := total_bill * tip_percentage
  let total_with_tip := total_bill + tip
  let num_people := total_with_tip / individual_share
  num_people = 5

theorem solve_dining_problem :
  dining_problem 139 (1/10) (3058/100) := by
  sorry

end NUMINAMATH_CALUDE_solve_dining_problem_l1236_123684


namespace NUMINAMATH_CALUDE_artist_profit_calculation_l1236_123605

/-- Calculates the total profit for an artist given contest winnings, painting sales, and expenses. -/
theorem artist_profit_calculation 
  (contest_prize : ℕ) 
  (num_paintings_sold : ℕ) 
  (price_per_painting : ℕ) 
  (art_supplies_cost : ℕ) 
  (exhibition_fee : ℕ) 
  (h1 : contest_prize = 150)
  (h2 : num_paintings_sold = 3)
  (h3 : price_per_painting = 50)
  (h4 : art_supplies_cost = 40)
  (h5 : exhibition_fee = 20) :
  contest_prize + num_paintings_sold * price_per_painting - (art_supplies_cost + exhibition_fee) = 240 :=
by sorry

end NUMINAMATH_CALUDE_artist_profit_calculation_l1236_123605


namespace NUMINAMATH_CALUDE_circle_equation_correct_l1236_123601

/-- The standard equation of a circle with center (a, b) and radius r -/
def CircleEquation (x y a b r : ℝ) : Prop :=
  (x - a)^2 + (y - b)^2 = r^2

/-- Theorem: The equation (x - 2)^2 + (y + 1)^2 = 4 represents a circle with center (2, -1) and radius 2 -/
theorem circle_equation_correct :
  ∀ x y : ℝ, CircleEquation x y 2 (-1) 2 ↔ (x - 2)^2 + (y + 1)^2 = 4 := by
  sorry

end NUMINAMATH_CALUDE_circle_equation_correct_l1236_123601


namespace NUMINAMATH_CALUDE_reciprocal_of_negative_sin_60_l1236_123644

theorem reciprocal_of_negative_sin_60 :
  ((-Real.sin (π / 3))⁻¹) = -(2 * Real.sqrt 3) / 3 := by sorry

end NUMINAMATH_CALUDE_reciprocal_of_negative_sin_60_l1236_123644


namespace NUMINAMATH_CALUDE_mango_price_reduction_l1236_123610

/-- Represents the price reduction problem for mangoes --/
theorem mango_price_reduction 
  (original_price : ℝ) 
  (original_quantity : ℕ) 
  (total_spent : ℝ) 
  (additional_mangoes : ℕ) :
  original_price = 416.67 →
  original_quantity = 125 →
  total_spent = 360 →
  additional_mangoes = 12 →
  let original_price_per_mango := original_price / original_quantity
  let original_bought_quantity := total_spent / original_price_per_mango
  let new_quantity := original_bought_quantity + additional_mangoes
  let new_price_per_mango := total_spent / new_quantity
  let price_reduction_percentage := (original_price_per_mango - new_price_per_mango) / original_price_per_mango * 100
  price_reduction_percentage = 10 := by
sorry


end NUMINAMATH_CALUDE_mango_price_reduction_l1236_123610


namespace NUMINAMATH_CALUDE_expression_equality_l1236_123641

theorem expression_equality : 
  Real.sqrt 25 - Real.sqrt 3 + |Real.sqrt 3 - 2| + ((-8 : ℝ) ^ (1/3)) = 5 - 2 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_expression_equality_l1236_123641


namespace NUMINAMATH_CALUDE_octagonal_pyramid_sum_l1236_123649

-- Define the structure of an octagonal pyramid
structure OctagonalPyramid where
  base_vertices : Nat
  base_edges : Nat
  triangular_faces : Nat
  apex_vertex : Nat
  apex_edges : Nat

-- Define the properties of an octagonal pyramid
def is_octagonal_pyramid (p : OctagonalPyramid) : Prop :=
  p.base_vertices = 8 ∧
  p.base_edges = 8 ∧
  p.triangular_faces = 8 ∧
  p.apex_vertex = 1 ∧
  p.apex_edges = 8

-- Calculate the total number of faces
def total_faces (p : OctagonalPyramid) : Nat :=
  1 + p.triangular_faces

-- Calculate the total number of edges
def total_edges (p : OctagonalPyramid) : Nat :=
  p.base_edges + p.apex_edges

-- Calculate the total number of vertices
def total_vertices (p : OctagonalPyramid) : Nat :=
  p.base_vertices + p.apex_vertex

-- Theorem: The sum of faces, edges, and vertices of an octagonal pyramid is 34
theorem octagonal_pyramid_sum (p : OctagonalPyramid) 
  (h : is_octagonal_pyramid p) : 
  total_faces p + total_edges p + total_vertices p = 34 := by
  sorry

end NUMINAMATH_CALUDE_octagonal_pyramid_sum_l1236_123649


namespace NUMINAMATH_CALUDE_prove_c_minus_d_equals_negative_three_l1236_123662

-- Define the function g
noncomputable def g : ℝ → ℝ := sorry

-- Define c and d
noncomputable def c : ℝ := sorry
noncomputable def d : ℝ := sorry

-- State the theorem
theorem prove_c_minus_d_equals_negative_three :
  Function.Injective g ∧ g c = d ∧ g d = 5 → c - d = -3 := by sorry

end NUMINAMATH_CALUDE_prove_c_minus_d_equals_negative_three_l1236_123662


namespace NUMINAMATH_CALUDE_sqrt_27_div_3_eq_sqrt_3_l1236_123694

theorem sqrt_27_div_3_eq_sqrt_3 : Real.sqrt 27 / 3 = Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_27_div_3_eq_sqrt_3_l1236_123694


namespace NUMINAMATH_CALUDE_trigonometric_product_upper_bound_l1236_123654

theorem trigonometric_product_upper_bound :
  ∀ x y z : ℝ,
  (Real.sin (2 * x) + Real.sin (3 * y) + Real.sin (4 * z)) *
  (Real.cos (2 * x) + Real.cos (3 * y) + Real.cos (4 * z)) ≤ 4.5 ∧
  ∃ x y z : ℝ,
  (Real.sin (2 * x) + Real.sin (3 * y) + Real.sin (4 * z)) *
  (Real.cos (2 * x) + Real.cos (3 * y) + Real.cos (4 * z)) = 4.5 :=
by sorry

end NUMINAMATH_CALUDE_trigonometric_product_upper_bound_l1236_123654


namespace NUMINAMATH_CALUDE_product_of_numbers_l1236_123617

theorem product_of_numbers (x y : ℝ) (h1 : x - y = 11) (h2 : x^2 + y^2 = 205) : x * y = 42 := by
  sorry

end NUMINAMATH_CALUDE_product_of_numbers_l1236_123617


namespace NUMINAMATH_CALUDE_parabola_equation_l1236_123691

/-- A parabola with focus F and point A on the curve, where |FA| is the radius of a circle
    intersecting the parabola's axis at B and C, forming an equilateral triangle FBC. -/
structure ParabolaWithTriangle where
  -- The parameter of the parabola
  p : ℝ
  -- The coordinates of points A, B, C, and F
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ
  F : ℝ × ℝ

/-- Properties of the parabola and associated triangle -/
def ParabolaProperties (P : ParabolaWithTriangle) : Prop :=
  -- A lies on the parabola y^2 = 2px
  P.A.2^2 = 2 * P.p * P.A.1 ∧
  -- F is the focus (p/2, 0)
  P.F = (P.p/2, 0) ∧
  -- B and C are on the x-axis
  P.B.2 = 0 ∧ P.C.2 = 0 ∧
  -- |FA| = |FB| = |FC|
  (P.A.1 - P.F.1)^2 + (P.A.2 - P.F.2)^2 = 
  (P.B.1 - P.F.1)^2 + (P.B.2 - P.F.2)^2 ∧
  (P.A.1 - P.F.1)^2 + (P.A.2 - P.F.2)^2 = 
  (P.C.1 - P.F.1)^2 + (P.C.2 - P.F.2)^2 ∧
  -- Area of triangle ABC is 128/3
  abs ((P.A.1 - P.C.1) * (P.B.2 - P.C.2) - (P.B.1 - P.C.1) * (P.A.2 - P.C.2)) / 2 = 128/3

theorem parabola_equation (P : ParabolaWithTriangle) 
  (h : ParabolaProperties P) : P.p = 8 ∧ ∀ (x y : ℝ), y^2 = 16*x ↔ y^2 = 2*P.p*x := by
  sorry

end NUMINAMATH_CALUDE_parabola_equation_l1236_123691


namespace NUMINAMATH_CALUDE_infinite_squares_in_progression_l1236_123692

/-- An arithmetic progression with positive integer members -/
structure ArithmeticProgression where
  a : ℕ+  -- First term
  d : ℕ+  -- Common difference

/-- Predicate to check if a number is in the arithmetic progression -/
def inProgression (ap : ArithmeticProgression) (n : ℕ) : Prop :=
  ∃ k : ℕ, n = ap.a + k * ap.d

/-- Predicate to check if a number is a perfect square -/
def isPerfectSquare (n : ℕ) : Prop :=
  ∃ m : ℕ, n = m * m

theorem infinite_squares_in_progression (ap : ArithmeticProgression) :
  (∃ n : ℕ, inProgression ap n ∧ isPerfectSquare n) →
  (∀ N : ℕ, ∃ n : ℕ, n > N ∧ inProgression ap n ∧ isPerfectSquare n) :=
sorry

end NUMINAMATH_CALUDE_infinite_squares_in_progression_l1236_123692


namespace NUMINAMATH_CALUDE_cos_shift_equals_sin_shift_l1236_123635

theorem cos_shift_equals_sin_shift (x : ℝ) : 
  Real.cos (x + π/3) = Real.sin (x + 5*π/6) := by
  sorry

end NUMINAMATH_CALUDE_cos_shift_equals_sin_shift_l1236_123635


namespace NUMINAMATH_CALUDE_f_2015_equals_2_l1236_123626

/-- A function satisfying the given conditions -/
def f_conditions (f : ℝ → ℝ) : Prop :=
  (∀ x : ℝ, f (x + 6) + f x = 0) ∧
  (∀ x : ℝ, f (x - 1) = f (3 - x)) ∧
  (f 1 = -2)

/-- Theorem stating that any function satisfying the conditions has f(2015) = 2 -/
theorem f_2015_equals_2 (f : ℝ → ℝ) (hf : f_conditions f) : f 2015 = 2 := by
  sorry

end NUMINAMATH_CALUDE_f_2015_equals_2_l1236_123626


namespace NUMINAMATH_CALUDE_regular_polygon_with_120_degree_interior_angle_l1236_123657

theorem regular_polygon_with_120_degree_interior_angle :
  ∀ n : ℕ, n ≥ 3 →
  (180 * (n - 2) / n : ℚ) = 120 →
  n = 6 :=
by
  sorry

end NUMINAMATH_CALUDE_regular_polygon_with_120_degree_interior_angle_l1236_123657


namespace NUMINAMATH_CALUDE_floor_equation_solution_l1236_123629

theorem floor_equation_solution (a : ℝ) : 
  (∀ n : ℕ, 4 * ⌊a * n⌋ = n + ⌊a * ⌊a * n⌋⌋) ↔ a = 2 + Real.sqrt 3 :=
sorry

end NUMINAMATH_CALUDE_floor_equation_solution_l1236_123629


namespace NUMINAMATH_CALUDE_boys_in_school_l1236_123642

theorem boys_in_school (total_students : ℕ) (boy_ratio girl_ratio : ℕ) : 
  total_students = 48 → 
  boy_ratio = 7 →
  girl_ratio = 1 →
  (boy_ratio : ℚ) / girl_ratio = (number_of_boys : ℚ) / (total_students - number_of_boys) →
  number_of_boys = 42 :=
by
  sorry

end NUMINAMATH_CALUDE_boys_in_school_l1236_123642


namespace NUMINAMATH_CALUDE_A_infinite_l1236_123667

/-- A function that represents z = n^4 + a -/
def z (n a : ℕ) : ℕ := n^4 + a

/-- The set of natural numbers a such that z(n, a) is composite for all n -/
def A : Set ℕ := {a : ℕ | ∀ n : ℕ, ¬ Nat.Prime (z n a)}

/-- Theorem stating that A is infinite -/
theorem A_infinite : Set.Infinite A := by sorry

end NUMINAMATH_CALUDE_A_infinite_l1236_123667


namespace NUMINAMATH_CALUDE_characterization_of_S_l1236_123659

open Set

-- Define the propositions p and q
def p (a : ℝ) : Prop := ∃ x : ℝ, x^2 + 2*a*x + 4 = 0
def q (a : ℝ) : Prop := ∀ x y : ℝ, x < y → (3 - 2*a)^x < (3 - 2*a)^y

-- Define the set of a values that satisfy both p and q
def S : Set ℝ := {a : ℝ | p a ∧ q a}

-- State the theorem
theorem characterization_of_S : S = Iic (-2) := by sorry

end NUMINAMATH_CALUDE_characterization_of_S_l1236_123659


namespace NUMINAMATH_CALUDE_jerrys_action_figures_l1236_123628

theorem jerrys_action_figures (initial_figures : ℕ) : 
  initial_figures + 4 - 1 = 6 → initial_figures = 3 :=
by
  sorry

end NUMINAMATH_CALUDE_jerrys_action_figures_l1236_123628


namespace NUMINAMATH_CALUDE_unique_number_l1236_123663

def is_valid_number (n : ℕ) : Prop :=
  100 ≤ n ∧ n < 1000 ∧ n % 10 = 2 ∧
  200 + (n / 10) = n + 18

theorem unique_number : ∃! n : ℕ, is_valid_number n ∧ n = 202 :=
sorry

end NUMINAMATH_CALUDE_unique_number_l1236_123663


namespace NUMINAMATH_CALUDE_derivative_f_at_2_l1236_123609

def f (x : ℝ) : ℝ := (x + 1)^2 * (x - 1)

theorem derivative_f_at_2 : 
  deriv f 2 = 15 := by sorry

end NUMINAMATH_CALUDE_derivative_f_at_2_l1236_123609


namespace NUMINAMATH_CALUDE_division_remainder_problem_l1236_123664

theorem division_remainder_problem (L S R : ℕ) : 
  L - S = 1365 → 
  L = 1631 → 
  L = 6 * S + R → 
  R = 35 := by
sorry

end NUMINAMATH_CALUDE_division_remainder_problem_l1236_123664


namespace NUMINAMATH_CALUDE_polly_cooking_time_l1236_123681

/-- Represents the cooking time for a week -/
structure CookingTime where
  breakfast_daily : ℕ
  lunch_daily : ℕ
  dinner_four_days : ℕ
  total_week : ℕ

/-- Calculates the time spent cooking dinner on the remaining days -/
def remaining_dinner_time (c : CookingTime) : ℕ :=
  c.total_week - (7 * (c.breakfast_daily + c.lunch_daily) + 4 * c.dinner_four_days)

/-- Theorem stating that given the conditions, Polly spends 90 minutes cooking dinner on the remaining days -/
theorem polly_cooking_time :
  let c : CookingTime := {
    breakfast_daily := 20,
    lunch_daily := 5,
    dinner_four_days := 10,
    total_week := 305
  }
  remaining_dinner_time c = 90 := by sorry

end NUMINAMATH_CALUDE_polly_cooking_time_l1236_123681


namespace NUMINAMATH_CALUDE_billion_to_scientific_notation_l1236_123616

theorem billion_to_scientific_notation :
  let billion : ℝ := 10^9
  8.26 * billion = 8.26 * 10^9 := by
  sorry

end NUMINAMATH_CALUDE_billion_to_scientific_notation_l1236_123616


namespace NUMINAMATH_CALUDE_circle_equation_tangent_to_x_axis_l1236_123676

/-- A circle in the 2D plane --/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- Check if a point is on a circle --/
def Circle.isOn (c : Circle) (p : ℝ × ℝ) : Prop :=
  (p.1 - c.center.1)^2 + (p.2 - c.center.2)^2 = c.radius^2

/-- Check if a circle is tangent to the x-axis --/
def Circle.tangentToXAxis (c : Circle) : Prop :=
  c.center.2 = c.radius

theorem circle_equation_tangent_to_x_axis (x y : ℝ) :
  (x - 5)^2 + (y - 4)^2 = 16 ↔
  ∃ (c : Circle), c.center = (5, 4) ∧ c.radius = 4 ∧
  c.isOn (x, y) ∧ c.tangentToXAxis :=
sorry

end NUMINAMATH_CALUDE_circle_equation_tangent_to_x_axis_l1236_123676


namespace NUMINAMATH_CALUDE_tank_dimension_l1236_123670

theorem tank_dimension (cost_per_sqft : ℝ) (total_cost : ℝ) (length : ℝ) (width : ℝ) :
  cost_per_sqft = 20 →
  total_cost = 1440 →
  length = 3 →
  width = 6 →
  ∃ height : ℝ, 
    height = 2 ∧ 
    total_cost = cost_per_sqft * (2 * (length * width + length * height + width * height)) :=
by sorry

end NUMINAMATH_CALUDE_tank_dimension_l1236_123670


namespace NUMINAMATH_CALUDE_solution_comparison_l1236_123653

theorem solution_comparison (a a' b b' : ℝ) 
  (ha : a > 0) (ha' : a' > 0) 
  (heq1 : ∃ x, 2 * a * x + b = 0) 
  (heq2 : ∃ x', 2 * a' * x' + b' = 0) 
  (hineq : (- b / (2 * a))^2 > (- b' / (2 * a'))^2) : 
  b^2 / a^2 > b'^2 / a'^2 := by
  sorry

end NUMINAMATH_CALUDE_solution_comparison_l1236_123653


namespace NUMINAMATH_CALUDE_plywood_cut_theorem_l1236_123646

theorem plywood_cut_theorem :
  ∃ (a b c d : Set (ℝ × ℝ)),
    -- The original square has area 625 cm²
    (∀ (x y : ℝ), (x, y) ∈ a ∪ b ∪ c ∪ d → 0 ≤ x ∧ x ≤ 25 ∧ 0 ≤ y ∧ y ≤ 25) ∧
    -- The four parts are disjoint
    (a ∩ b = ∅ ∧ a ∩ c = ∅ ∧ a ∩ d = ∅ ∧ b ∩ c = ∅ ∧ b ∩ d = ∅ ∧ c ∩ d = ∅) ∧
    -- The four parts cover the entire original square
    (∀ (x y : ℝ), 0 ≤ x ∧ x ≤ 25 ∧ 0 ≤ y ∧ y ≤ 25 → (x, y) ∈ a ∪ b ∪ c ∪ d) ∧
    -- The parts can be rearranged into two squares
    (∃ (s₁ s₂ : Set (ℝ × ℝ)),
      -- First square has side length 24 cm
      (∀ (x y : ℝ), (x, y) ∈ s₁ → 0 ≤ x ∧ x ≤ 24 ∧ 0 ≤ y ∧ y ≤ 24) ∧
      -- Second square has side length 7 cm
      (∀ (x y : ℝ), (x, y) ∈ s₂ → 0 ≤ x ∧ x ≤ 7 ∧ 0 ≤ y ∧ y ≤ 7) ∧
      -- The rearranged squares cover the same area as the original parts
      (∀ (x y : ℝ), (x, y) ∈ s₁ ∪ s₂ ↔ (x, y) ∈ a ∪ b ∪ c ∪ d)) :=
by
  sorry


end NUMINAMATH_CALUDE_plywood_cut_theorem_l1236_123646


namespace NUMINAMATH_CALUDE_correct_initial_amounts_l1236_123645

/-- Represents the initial amounts of money John and Richard had --/
structure InitialMoney where
  john : ℚ
  richard : ℚ

/-- Represents the final amounts of money John and Richard had after transactions --/
structure FinalMoney where
  john : ℚ
  richard : ℚ

/-- Calculates the final money based on initial money and described transactions --/
def calculateFinalMoney (initial : InitialMoney) : FinalMoney :=
  { john := initial.john - (initial.richard + initial.john),
    richard := 2 * initial.richard + 2 * initial.john }

/-- Theorem stating the correct initial amounts given the final amounts --/
theorem correct_initial_amounts :
  ∃ (initial : InitialMoney),
    let final := calculateFinalMoney initial
    final.john = 7/2 ∧ final.richard = 3 ∧
    initial.john = 5/2 ∧ initial.richard = 3/2 := by
  sorry

end NUMINAMATH_CALUDE_correct_initial_amounts_l1236_123645


namespace NUMINAMATH_CALUDE_infinite_points_in_region_l1236_123687

/-- The set of points with positive rational coordinates satisfying x + 2y ≤ 6 is infinite -/
theorem infinite_points_in_region : 
  Set.Infinite {p : ℚ × ℚ | 0 < p.1 ∧ 0 < p.2 ∧ p.1 + 2 * p.2 ≤ 6} := by sorry

end NUMINAMATH_CALUDE_infinite_points_in_region_l1236_123687


namespace NUMINAMATH_CALUDE_canoe_trip_average_distance_l1236_123606

/-- Proves that given a 6-day canoe trip with a total distance of 168 km, 
    where 3/7 of the distance is completed in 3 days, 
    the average distance per day for the remaining days is 32 km. -/
theorem canoe_trip_average_distance 
  (total_distance : ℝ) 
  (total_days : ℕ) 
  (completed_fraction : ℚ) 
  (completed_days : ℕ) 
  (h1 : total_distance = 168)
  (h2 : total_days = 6)
  (h3 : completed_fraction = 3/7)
  (h4 : completed_days = 3) : 
  (total_distance * (1 - completed_fraction)) / (total_days - completed_days) = 32 := by
  sorry

end NUMINAMATH_CALUDE_canoe_trip_average_distance_l1236_123606
