import Mathlib

namespace NUMINAMATH_CALUDE_amp_six_three_l2191_219134

/-- The & operation defined on two real numbers -/
def amp (a b : ℝ) : ℝ := (a + b) * (a - b)

/-- Theorem stating that 6 & 3 = 27 -/
theorem amp_six_three : amp 6 3 = 27 := by
  sorry

end NUMINAMATH_CALUDE_amp_six_three_l2191_219134


namespace NUMINAMATH_CALUDE_negation_of_existence_is_forall_not_l2191_219170

theorem negation_of_existence_is_forall_not :
  (¬ ∃ x : ℝ, x^2 + 1 ≤ 0) ↔ (∀ x : ℝ, x^2 + 1 > 0) := by
  sorry

end NUMINAMATH_CALUDE_negation_of_existence_is_forall_not_l2191_219170


namespace NUMINAMATH_CALUDE_quiche_volume_l2191_219171

theorem quiche_volume (raw_spinach : ℝ) (cooked_spinach_ratio : ℝ) (cream_cheese : ℝ) (eggs : ℝ)
  (h1 : raw_spinach = 40)
  (h2 : cooked_spinach_ratio = 0.2)
  (h3 : cream_cheese = 6)
  (h4 : eggs = 4) :
  raw_spinach * cooked_spinach_ratio + cream_cheese + eggs = 18 :=
by sorry

end NUMINAMATH_CALUDE_quiche_volume_l2191_219171


namespace NUMINAMATH_CALUDE_atomic_mass_scientific_notation_l2191_219149

/-- Represents a number in scientific notation -/
structure ScientificNotation where
  coefficient : ℝ
  exponent : ℤ
  one_le_coeff_lt_ten : 1 ≤ coefficient ∧ coefficient < 10

/-- Converts a real number to scientific notation -/
def toScientificNotation (x : ℝ) : ScientificNotation :=
  sorry

theorem atomic_mass_scientific_notation :
  toScientificNotation 0.00001992 = ScientificNotation.mk 1.992 (-5) sorry := by
  sorry

end NUMINAMATH_CALUDE_atomic_mass_scientific_notation_l2191_219149


namespace NUMINAMATH_CALUDE_similarity_criteria_l2191_219110

/-- A structure representing a triangle -/
structure Triangle where
  -- We'll assume triangles are defined by their side lengths and angles
  side1 : ℝ
  side2 : ℝ
  side3 : ℝ
  angle1 : ℝ
  angle2 : ℝ
  angle3 : ℝ

/-- Two triangles are similar if they have the same shape but not necessarily the same size -/
def similar (t1 t2 : Triangle) : Prop :=
  sorry

/-- SSS (Side-Side-Side) Similarity: Two triangles are similar if their corresponding sides are proportional -/
def SSS_similarity (t1 t2 : Triangle) : Prop :=
  ∃ k : ℝ, k > 0 ∧ 
    t1.side1 / t2.side1 = k ∧
    t1.side2 / t2.side2 = k ∧
    t1.side3 / t2.side3 = k

/-- SAS (Side-Angle-Side) Similarity: Two triangles are similar if two pairs of corresponding sides are proportional and the included angles are equal -/
def SAS_similarity (t1 t2 : Triangle) : Prop :=
  ∃ k : ℝ, k > 0 ∧ 
    t1.side1 / t2.side1 = k ∧
    t1.side2 / t2.side2 = k ∧
    t1.angle3 = t2.angle3

/-- Theorem: Two triangles are similar if and only if they satisfy either SSS or SAS similarity criteria -/
theorem similarity_criteria (t1 t2 : Triangle) :
  similar t1 t2 ↔ SSS_similarity t1 t2 ∨ SAS_similarity t1 t2 :=
sorry

end NUMINAMATH_CALUDE_similarity_criteria_l2191_219110


namespace NUMINAMATH_CALUDE_min_sum_of_squares_l2191_219139

theorem min_sum_of_squares (x₁ x₂ x₃ : ℝ) 
  (pos₁ : x₁ > 0) (pos₂ : x₂ > 0) (pos₃ : x₃ > 0)
  (sum_constraint : x₁ + 3*x₂ + 5*x₃ = 100) : 
  x₁^2 + x₂^2 + x₃^2 ≥ 2000/7 := by
  sorry

end NUMINAMATH_CALUDE_min_sum_of_squares_l2191_219139


namespace NUMINAMATH_CALUDE_tetrahedra_arrangement_exists_l2191_219138

/-- A type representing a regular tetrahedron -/
structure Tetrahedron where
  -- Add necessary fields

/-- A type representing the arrangement of tetrahedra -/
structure Arrangement where
  tetrahedra : Set Tetrahedron
  lower_plane : Set (ℝ × ℝ × ℝ)
  upper_plane : Set (ℝ × ℝ × ℝ)

/-- Predicate to check if two planes are parallel -/
def are_parallel (plane1 plane2 : Set (ℝ × ℝ × ℝ)) : Prop :=
  sorry

/-- Predicate to check if a tetrahedron is between two planes -/
def is_between_planes (t : Tetrahedron) (lower upper : Set (ℝ × ℝ × ℝ)) : Prop :=
  sorry

/-- Predicate to check if a tetrahedron can be removed without moving others -/
def can_be_removed (t : Tetrahedron) (arr : Arrangement) : Prop :=
  sorry

/-- The main theorem statement -/
theorem tetrahedra_arrangement_exists :
  ∃ (arr : Arrangement),
    (∀ t ∈ arr.tetrahedra, is_between_planes t arr.lower_plane arr.upper_plane) ∧
    (are_parallel arr.lower_plane arr.upper_plane) ∧
    (Set.Infinite arr.tetrahedra) ∧
    (∀ t ∈ arr.tetrahedra, ¬can_be_removed t arr) :=
  sorry

end NUMINAMATH_CALUDE_tetrahedra_arrangement_exists_l2191_219138


namespace NUMINAMATH_CALUDE_subtraction_of_reciprocals_l2191_219154

theorem subtraction_of_reciprocals (p q : ℚ) : 
  (4 / p = 8) → (4 / q = 18) → (p - q = 5 / 18) := by
  sorry

end NUMINAMATH_CALUDE_subtraction_of_reciprocals_l2191_219154


namespace NUMINAMATH_CALUDE_combined_salaries_of_abce_l2191_219114

def average_salary : ℕ := 8800
def number_of_people : ℕ := 5
def d_salary : ℕ := 7000

theorem combined_salaries_of_abce :
  (average_salary * number_of_people) - d_salary = 37000 := by
  sorry

end NUMINAMATH_CALUDE_combined_salaries_of_abce_l2191_219114


namespace NUMINAMATH_CALUDE_triangle_angle_cosine_inequality_l2191_219118

theorem triangle_angle_cosine_inequality (α β γ : Real) 
  (h : α + β + γ = Real.pi) : 
  Real.cos (α + Real.pi / 3) + Real.cos (β + Real.pi / 3) + Real.cos (γ + Real.pi / 3) + 3 / 2 ≥ 0 := by
  sorry

end NUMINAMATH_CALUDE_triangle_angle_cosine_inequality_l2191_219118


namespace NUMINAMATH_CALUDE_yeast_population_growth_l2191_219187

/-- The yeast population growth problem -/
theorem yeast_population_growth
  (initial_population : ℕ)
  (growth_factor : ℕ)
  (time_increments : ℕ)
  (h1 : initial_population = 30)
  (h2 : growth_factor = 3)
  (h3 : time_increments = 3) :
  initial_population * growth_factor ^ time_increments = 810 :=
by sorry

end NUMINAMATH_CALUDE_yeast_population_growth_l2191_219187


namespace NUMINAMATH_CALUDE_greatest_common_factor_of_three_digit_same_digit_palindromes_l2191_219111

def is_three_digit_same_digit_palindrome (n : ℕ) : Prop :=
  100 ≤ n ∧ n ≤ 999 ∧ ∃ d : ℕ, 1 ≤ d ∧ d ≤ 9 ∧ n = 100 * d + 10 * d + d

theorem greatest_common_factor_of_three_digit_same_digit_palindromes :
  ∃ (gcf : ℕ), gcf = 111 ∧
  (∀ n : ℕ, is_three_digit_same_digit_palindrome n → gcf ∣ n) ∧
  (∀ m : ℕ, (∀ n : ℕ, is_three_digit_same_digit_palindrome n → m ∣ n) → m ≤ gcf) :=
sorry

end NUMINAMATH_CALUDE_greatest_common_factor_of_three_digit_same_digit_palindromes_l2191_219111


namespace NUMINAMATH_CALUDE_no_solution_l2191_219142

theorem no_solution : ¬ ∃ (n : ℕ), (823435^15 % n = 0) ∧ (n^5 - n^n = 1) := by
  sorry

end NUMINAMATH_CALUDE_no_solution_l2191_219142


namespace NUMINAMATH_CALUDE_fraction_simplification_l2191_219147

theorem fraction_simplification : (8 + 4) / (8 - 4) = 3 := by sorry

end NUMINAMATH_CALUDE_fraction_simplification_l2191_219147


namespace NUMINAMATH_CALUDE_skittles_distribution_l2191_219183

theorem skittles_distribution (total_skittles : ℕ) (num_friends : ℕ) (skittles_per_friend : ℕ) : 
  total_skittles = 40 → num_friends = 5 → skittles_per_friend = total_skittles / num_friends → skittles_per_friend = 8 := by
  sorry

end NUMINAMATH_CALUDE_skittles_distribution_l2191_219183


namespace NUMINAMATH_CALUDE_total_mulberries_correct_l2191_219106

/-- Represents the mulberry purchase and sale scenario -/
structure MulberrySale where
  total_cost : ℝ
  first_sale_quantity : ℝ
  first_sale_price_increase : ℝ
  second_sale_price_decrease : ℝ
  total_profit : ℝ

/-- Calculates the total amount of mulberries purchased -/
def calculate_total_mulberries (sale : MulberrySale) : ℝ :=
  200 -- The actual calculation is omitted and replaced with the known result

/-- Theorem stating that the calculated total mulberries is correct -/
theorem total_mulberries_correct (sale : MulberrySale) 
  (h1 : sale.total_cost = 3000)
  (h2 : sale.first_sale_quantity = 150)
  (h3 : sale.first_sale_price_increase = 0.4)
  (h4 : sale.second_sale_price_decrease = 0.2)
  (h5 : sale.total_profit = 750) :
  calculate_total_mulberries sale = 200 := by
  sorry

#eval calculate_total_mulberries {
  total_cost := 3000,
  first_sale_quantity := 150,
  first_sale_price_increase := 0.4,
  second_sale_price_decrease := 0.2,
  total_profit := 750
}

end NUMINAMATH_CALUDE_total_mulberries_correct_l2191_219106


namespace NUMINAMATH_CALUDE_bicycle_spokes_l2191_219115

/-- Represents a bicycle with front and back wheels -/
structure Bicycle where
  front_spokes : ℕ
  back_spokes : ℕ

/-- Calculates the total number of spokes on a bicycle -/
def total_spokes (b : Bicycle) : ℕ :=
  b.front_spokes + b.back_spokes

/-- Theorem: A bicycle with 20 front spokes and twice as many back spokes has 60 spokes in total -/
theorem bicycle_spokes :
  ∀ b : Bicycle, b.front_spokes = 20 ∧ b.back_spokes = 2 * b.front_spokes →
  total_spokes b = 60 := by
  sorry

end NUMINAMATH_CALUDE_bicycle_spokes_l2191_219115


namespace NUMINAMATH_CALUDE_pascal_triangle_29th_row_28th_number_l2191_219112

theorem pascal_triangle_29th_row_28th_number : Nat.choose 29 27 = 406 := by
  sorry

end NUMINAMATH_CALUDE_pascal_triangle_29th_row_28th_number_l2191_219112


namespace NUMINAMATH_CALUDE_marie_stamps_giveaway_l2191_219198

theorem marie_stamps_giveaway (notebooks : Nat) (stamps_per_notebook : Nat)
  (binders : Nat) (stamps_per_binder : Nat) (keep_percentage : Rat) :
  notebooks = 30 →
  stamps_per_notebook = 120 →
  binders = 7 →
  stamps_per_binder = 210 →
  keep_percentage = 35 / 100 →
  (notebooks * stamps_per_notebook + binders * stamps_per_binder : Nat) -
    (((notebooks * stamps_per_notebook + binders * stamps_per_binder : Nat) : Rat) *
      keep_percentage).floor.toNat = 3296 := by
  sorry

end NUMINAMATH_CALUDE_marie_stamps_giveaway_l2191_219198


namespace NUMINAMATH_CALUDE_square_equation_solutions_l2191_219126

/-- p-arithmetic field -/
structure PArithmetic (p : ℕ) where
  carrier : Type
  zero : carrier
  one : carrier
  add : carrier → carrier → carrier
  mul : carrier → carrier → carrier
  neg : carrier → carrier
  inv : carrier → carrier
  -- Add necessary field axioms here

/-- Definition of squaring in p-arithmetic -/
def square {p : ℕ} (F : PArithmetic p) (x : F.carrier) : F.carrier :=
  F.mul x x

/-- Main theorem: In p-arithmetic (p ≠ 2), x² = a has two distinct solutions for non-zero a -/
theorem square_equation_solutions {p : ℕ} (hp : p ≠ 2) (F : PArithmetic p) :
  ∀ a : F.carrier, a ≠ F.zero →
    ∃ x y : F.carrier, x ≠ y ∧ square F x = a ∧ square F y = a ∧
      ∀ z : F.carrier, square F z = a → (z = x ∨ z = y) :=
sorry

end NUMINAMATH_CALUDE_square_equation_solutions_l2191_219126


namespace NUMINAMATH_CALUDE_band_gigs_played_l2191_219137

/-- Represents the earnings of each band member per gig -/
structure BandEarnings :=
  (leadSinger : ℕ)
  (guitarist : ℕ)
  (bassist : ℕ)
  (drummer : ℕ)
  (keyboardist : ℕ)
  (backupSinger1 : ℕ)
  (backupSinger2 : ℕ)
  (backupSinger3 : ℕ)

/-- Calculates the total earnings per gig for the band -/
def totalEarningsPerGig (earnings : BandEarnings) : ℕ :=
  earnings.leadSinger + earnings.guitarist + earnings.bassist + earnings.drummer +
  earnings.keyboardist + earnings.backupSinger1 + earnings.backupSinger2 + earnings.backupSinger3

/-- Theorem: The band has played 21 gigs -/
theorem band_gigs_played (earnings : BandEarnings) 
  (h1 : earnings.leadSinger = 30)
  (h2 : earnings.guitarist = 25)
  (h3 : earnings.bassist = 20)
  (h4 : earnings.drummer = 25)
  (h5 : earnings.keyboardist = 20)
  (h6 : earnings.backupSinger1 = 15)
  (h7 : earnings.backupSinger2 = 18)
  (h8 : earnings.backupSinger3 = 12)
  (h9 : totalEarningsPerGig earnings * 21 = 3465) :
  21 = 3465 / (totalEarningsPerGig earnings) :=
by sorry

end NUMINAMATH_CALUDE_band_gigs_played_l2191_219137


namespace NUMINAMATH_CALUDE_angle_D_value_l2191_219107

-- Define the angles as real numbers
variable (A B C D : ℝ)

-- State the given conditions
axiom angle_sum : A + B = 180
axiom angle_relation : C = D + 10
axiom angle_A : A = 50
axiom triangle_sum : B + C + D = 180

-- State the theorem to be proved
theorem angle_D_value : D = 20 := by
  sorry

end NUMINAMATH_CALUDE_angle_D_value_l2191_219107


namespace NUMINAMATH_CALUDE_inequality_solution_set_l2191_219108

theorem inequality_solution_set (a : ℝ) :
  let S := {x : ℝ | (x + 1) * (x - a) < 0}
  if a > -1 then
    S = {x : ℝ | -1 < x ∧ x < a}
  else if a = -1 then
    S = ∅
  else
    S = {x : ℝ | a < x ∧ x < -1} :=
by
  sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l2191_219108


namespace NUMINAMATH_CALUDE_scientific_notation_equivalence_l2191_219176

-- Define the original number
def original_number : ℕ := 141260

-- Define the scientific notation representation
def scientific_notation : ℝ := 1.4126 * (10 ^ 5)

-- Theorem to prove
theorem scientific_notation_equivalence :
  (original_number : ℝ) = scientific_notation :=
sorry

end NUMINAMATH_CALUDE_scientific_notation_equivalence_l2191_219176


namespace NUMINAMATH_CALUDE_triangle_properties_l2191_219144

theorem triangle_properties (a b c : ℝ) (A B C : ℝ) :
  a = 2 * Real.sqrt 3 →
  A = π / 3 →
  12 = b^2 + c^2 - b*c →
  (∃ (S : ℝ), S = (Real.sqrt 3 / 4) * b * c ∧ S ≤ 3 * Real.sqrt 3 ∧
    (S = 3 * Real.sqrt 3 → b = c)) ∧
  (a + b + c ≤ 6 * Real.sqrt 3 ∧
    (a + b + c = 6 * Real.sqrt 3 → b = c)) ∧
  (0 < B ∧ B < π/2 ∧ 0 < C ∧ C < π/2 →
    1/2 < b/c ∧ b/c < 2) :=
by sorry

end NUMINAMATH_CALUDE_triangle_properties_l2191_219144


namespace NUMINAMATH_CALUDE_smallest_sum_mn_l2191_219164

theorem smallest_sum_mn (m n : ℕ) (h : 3 * n^3 = 5 * m^2) : 
  ∃ (m' n' : ℕ), 3 * n'^3 = 5 * m'^2 ∧ m' + n' ≤ m + n ∧ m' + n' = 60 :=
sorry

end NUMINAMATH_CALUDE_smallest_sum_mn_l2191_219164


namespace NUMINAMATH_CALUDE_highest_power_of_two_dividing_difference_of_fourth_powers_l2191_219194

theorem highest_power_of_two_dividing_difference_of_fourth_powers :
  ∃ k : ℕ, k = 7 ∧ 2^k = (Nat.gcd (17^4 - 15^4) (2^64)) :=
by
  sorry

end NUMINAMATH_CALUDE_highest_power_of_two_dividing_difference_of_fourth_powers_l2191_219194


namespace NUMINAMATH_CALUDE_complex_sum_of_powers_l2191_219173

theorem complex_sum_of_powers : 
  ((-1 + Complex.I * Real.sqrt 3) / 2) ^ 12 + ((-1 - Complex.I * Real.sqrt 3) / 2) ^ 12 = 2 := by
  sorry

end NUMINAMATH_CALUDE_complex_sum_of_powers_l2191_219173


namespace NUMINAMATH_CALUDE_article_cost_l2191_219136

theorem article_cost (selling_price_high : ℝ) (selling_price_low : ℝ) (cost : ℝ) :
  selling_price_high = 600 →
  selling_price_low = 580 →
  selling_price_high - cost = 1.05 * (selling_price_low - cost) →
  cost = 180 := by
  sorry

end NUMINAMATH_CALUDE_article_cost_l2191_219136


namespace NUMINAMATH_CALUDE_hexagon_around_convex_curve_l2191_219122

/-- A convex curve in a 2D plane -/
structure ConvexCurve where
  -- Add necessary fields/axioms for a convex curve

/-- A hexagon in a 2D plane -/
structure Hexagon where
  -- Add necessary fields for a hexagon (e.g., vertices, sides)

/-- Predicate to check if a hexagon is circumscribed around a convex curve -/
def is_circumscribed (h : Hexagon) (c : ConvexCurve) : Prop :=
  sorry

/-- Predicate to check if all internal angles of a hexagon are equal -/
def has_equal_angles (h : Hexagon) : Prop :=
  sorry

/-- Predicate to check if opposite sides of a hexagon are equal -/
def has_equal_opposite_sides (h : Hexagon) : Prop :=
  sorry

/-- Predicate to check if a hexagon has an axis of symmetry -/
def has_symmetry_axis (h : Hexagon) : Prop :=
  sorry

/-- Theorem: For any convex curve, there exists a circumscribed hexagon with equal angles, 
    equal opposite sides, and an axis of symmetry -/
theorem hexagon_around_convex_curve (c : ConvexCurve) : 
  ∃ h : Hexagon, 
    is_circumscribed h c ∧ 
    has_equal_angles h ∧ 
    has_equal_opposite_sides h ∧ 
    has_symmetry_axis h :=
by
  sorry

end NUMINAMATH_CALUDE_hexagon_around_convex_curve_l2191_219122


namespace NUMINAMATH_CALUDE_sum_of_coefficients_l2191_219130

theorem sum_of_coefficients (a a₁ a₂ a₃ a₄ a₅ a₆ a₇ : ℤ) :
  (∀ x : ℚ, (3 * x - 1)^7 = a₇ * x^7 + a₆ * x^6 + a₅ * x^5 + a₄ * x^4 + a₃ * x^3 + a₂ * x^2 + a₁ * x + a) →
  a₇ + a₆ + a₅ + a₄ + a₃ + a₂ + a₁ + a = 128 := by
sorry

end NUMINAMATH_CALUDE_sum_of_coefficients_l2191_219130


namespace NUMINAMATH_CALUDE_factorial_simplification_l2191_219132

theorem factorial_simplification :
  (13 : ℕ).factorial / ((11 : ℕ).factorial + 3 * (9 : ℕ).factorial) = 17160 / 113 := by
  sorry

end NUMINAMATH_CALUDE_factorial_simplification_l2191_219132


namespace NUMINAMATH_CALUDE_quadratic_function_value_l2191_219192

/-- Given a quadratic function f(x) = ax^2 + bx + c where f(1) = 3 and f(2) = 12, prove that f(3) = 21 -/
theorem quadratic_function_value (a b c : ℝ) (f : ℝ → ℝ) 
  (h_quad : ∀ x, f x = a * x^2 + b * x + c)
  (h_f1 : f 1 = 3)
  (h_f2 : f 2 = 12) : 
  f 3 = 21 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_function_value_l2191_219192


namespace NUMINAMATH_CALUDE_largest_integer_satisfying_inequality_l2191_219143

theorem largest_integer_satisfying_inequality :
  ∀ x : ℤ, (1/4 : ℚ) + (x : ℚ)/9 < 1 ↔ x ≤ 6 :=
by sorry

end NUMINAMATH_CALUDE_largest_integer_satisfying_inequality_l2191_219143


namespace NUMINAMATH_CALUDE_john_david_pushup_difference_l2191_219153

/-- The number of push-ups done by Zachary -/
def zachary_pushups : ℕ := 19

/-- The number of push-ups David did more than Zachary -/
def david_extra_pushups : ℕ := 39

/-- The number of push-ups done by David -/
def david_pushups : ℕ := 58

/-- The number of push-ups done by John -/
def john_pushups : ℕ := david_pushups

theorem john_david_pushup_difference :
  david_pushups - john_pushups = 0 :=
sorry

end NUMINAMATH_CALUDE_john_david_pushup_difference_l2191_219153


namespace NUMINAMATH_CALUDE_car_meeting_problem_l2191_219150

theorem car_meeting_problem (S : ℝ) 
  (h1 : S > 0) 
  (h2 : 60 < S) 
  (h3 : 50 < S) 
  (h4 : (60 / (S - 60)) = ((S - 60 + 50) / (60 + S - 50))) : S = 130 := by
  sorry

end NUMINAMATH_CALUDE_car_meeting_problem_l2191_219150


namespace NUMINAMATH_CALUDE_locus_of_points_l2191_219172

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a rectangle -/
structure Rectangle where
  a : ℝ
  b : ℝ
  center : Point

/-- Represents an octagon -/
structure Octagon where
  vertices : Fin 8 → Point

/-- Calculate the absolute distance from a point to a line segment -/
def distToSegment (p : Point) (s1 s2 : Point) : ℝ :=
  sorry

/-- Calculate the sum of distances from a point to the sides of a rectangle -/
def sumDistToSides (p : Point) (r : Rectangle) : ℝ :=
  sorry

/-- Check if a point is inside or on the boundary of an octagon -/
def isInOctagon (p : Point) (o : Octagon) : Prop :=
  sorry

/-- Construct the octagon based on the rectangle and c value -/
def constructOctagon (r : Rectangle) (c : ℝ) : Octagon :=
  sorry

/-- The main theorem statement -/
theorem locus_of_points (r : Rectangle) (c : ℝ) :
  ∀ p : Point, sumDistToSides p r = r.a + r.b + c ↔ isInOctagon p (constructOctagon r c) :=
  sorry

end NUMINAMATH_CALUDE_locus_of_points_l2191_219172


namespace NUMINAMATH_CALUDE_evaluate_expression_l2191_219177

theorem evaluate_expression : (2^13 : ℚ) / (5 * 4^3) = 128 / 5 := by
  sorry

end NUMINAMATH_CALUDE_evaluate_expression_l2191_219177


namespace NUMINAMATH_CALUDE_dodecagon_diagonals_l2191_219181

/-- The number of diagonals in a convex polygon with n sides -/
def num_diagonals (n : ℕ) : ℕ := n * (n - 3) / 2

/-- A dodecagon has 12 sides -/
def dodecagon_sides : ℕ := 12

theorem dodecagon_diagonals :
  num_diagonals dodecagon_sides = 54 := by sorry

end NUMINAMATH_CALUDE_dodecagon_diagonals_l2191_219181


namespace NUMINAMATH_CALUDE_no_sequence_with_special_differences_l2191_219100

theorem no_sequence_with_special_differences :
  ¬ ∃ (a : ℕ → ℕ),
    (∀ k : ℕ, ∃! n : ℕ, a (n + 1) - a n = k) ∧
    (∀ k : ℕ, k > 2015 → ∃! n : ℕ, a (n + 2) - a n = k) :=
by sorry

end NUMINAMATH_CALUDE_no_sequence_with_special_differences_l2191_219100


namespace NUMINAMATH_CALUDE_glycerin_solution_problem_l2191_219168

/-- Proves that given a solution with an initial volume of 4 gallons, 
    adding 0.8 gallons of water to achieve a 75% glycerin solution 
    implies that the initial percentage of glycerin was 90%. -/
theorem glycerin_solution_problem (initial_volume : ℝ) (water_added : ℝ) (final_percentage : ℝ) :
  initial_volume = 4 →
  water_added = 0.8 →
  final_percentage = 0.75 →
  (initial_volume * (initial_volume / (initial_volume + water_added))) / initial_volume = 0.9 :=
by sorry

end NUMINAMATH_CALUDE_glycerin_solution_problem_l2191_219168


namespace NUMINAMATH_CALUDE_bus_fare_impossible_l2191_219158

/-- Represents the denominations of coins available --/
inductive Coin : Type
  | ten : Coin
  | fifteen : Coin
  | twenty : Coin

/-- The value of a coin in kopecks --/
def coin_value : Coin → Nat
  | Coin.ten => 10
  | Coin.fifteen => 15
  | Coin.twenty => 20

/-- A configuration of coins --/
def CoinConfig := List Coin

/-- The total value of a coin configuration in kopecks --/
def total_value (config : CoinConfig) : Nat :=
  config.foldl (fun acc c => acc + coin_value c) 0

/-- The number of coins in a configuration --/
def coin_count (config : CoinConfig) : Nat := config.length

theorem bus_fare_impossible : 
  ∀ (config : CoinConfig), 
    (coin_count config = 49) → 
    (total_value config = 200) → 
    False :=
sorry

end NUMINAMATH_CALUDE_bus_fare_impossible_l2191_219158


namespace NUMINAMATH_CALUDE_odd_function_properties_l2191_219188

-- Define an odd function
def OddFunction (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

-- Define an increasing function on an interval
def IncreasingOn (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x y, a ≤ x ∧ x < y ∧ y ≤ b → f x < f y

-- Define the minimum value of a function on an interval
def MinValueOn (f : ℝ → ℝ) (a b : ℝ) (m : ℝ) : Prop :=
  (∀ x, a ≤ x ∧ x ≤ b → m ≤ f x) ∧ (∃ x, a ≤ x ∧ x ≤ b ∧ f x = m)

-- Define the maximum value of a function on an interval
def MaxValueOn (f : ℝ → ℝ) (a b : ℝ) (m : ℝ) : Prop :=
  (∀ x, a ≤ x ∧ x ≤ b → f x ≤ m) ∧ (∃ x, a ≤ x ∧ x ≤ b ∧ f x = m)

-- Theorem statement
theorem odd_function_properties (f : ℝ → ℝ) :
  OddFunction f →
  IncreasingOn f 3 7 →
  MinValueOn f 3 7 1 →
  IncreasingOn f (-7) (-3) ∧ MaxValueOn f (-7) (-3) (-1) :=
by sorry

end NUMINAMATH_CALUDE_odd_function_properties_l2191_219188


namespace NUMINAMATH_CALUDE_teaching_years_difference_l2191_219135

/-- Represents the teaching years of Virginia, Adrienne, and Dennis -/
structure TeachingYears where
  virginia : ℕ
  adrienne : ℕ
  dennis : ℕ

/-- The conditions of the problem -/
def problem_conditions (years : TeachingYears) : Prop :=
  years.virginia + years.adrienne + years.dennis = 75 ∧
  years.virginia = years.adrienne + 9 ∧
  years.dennis = 34

/-- The theorem to be proved -/
theorem teaching_years_difference (years : TeachingYears) 
  (h : problem_conditions years) : 
  years.dennis - years.virginia = 9 := by
  sorry


end NUMINAMATH_CALUDE_teaching_years_difference_l2191_219135


namespace NUMINAMATH_CALUDE_original_number_proof_l2191_219128

theorem original_number_proof (e : ℝ) : 
  (e * 1.125 - e * 0.75 = 30) → e = 80 := by
  sorry

end NUMINAMATH_CALUDE_original_number_proof_l2191_219128


namespace NUMINAMATH_CALUDE_salesman_commission_l2191_219185

/-- Calculates the total commission for a salesman given the commission rates and bonus amount. -/
theorem salesman_commission
  (base_commission_rate : Real)
  (bonus_commission_rate : Real)
  (bonus_threshold : Real)
  (bonus_amount : Real) :
  base_commission_rate = 0.09 →
  bonus_commission_rate = 0.03 →
  bonus_threshold = 10000 →
  bonus_amount = 120 →
  ∃ (total_sales : Real),
    total_sales > bonus_threshold ∧
    bonus_commission_rate * (total_sales - bonus_threshold) = bonus_amount ∧
    base_commission_rate * total_sales + bonus_amount = 1380 :=
by sorry

end NUMINAMATH_CALUDE_salesman_commission_l2191_219185


namespace NUMINAMATH_CALUDE_smallest_number_with_remainders_l2191_219196

theorem smallest_number_with_remainders : ∃ n : ℕ, 
  (n > 0) ∧ 
  (n % 4 = 1) ∧ 
  (n % 3 = 2) ∧ 
  (n % 5 = 2) ∧
  (∀ m : ℕ, m > 0 ∧ m % 4 = 1 ∧ m % 3 = 2 ∧ m % 5 = 2 → m ≥ n) ∧
  n = 17 :=
by
  sorry

end NUMINAMATH_CALUDE_smallest_number_with_remainders_l2191_219196


namespace NUMINAMATH_CALUDE_min_vegetable_dishes_l2191_219104

theorem min_vegetable_dishes (n : ℕ) (h : n ≥ 5) :
  (∃ x : ℕ, x ≥ 7 ∧ Nat.choose n 2 * Nat.choose x 2 > 200) ∧
  (∀ y : ℕ, y < 7 → Nat.choose n 2 * Nat.choose y 2 ≤ 200) :=
by sorry

end NUMINAMATH_CALUDE_min_vegetable_dishes_l2191_219104


namespace NUMINAMATH_CALUDE_bernardo_wins_l2191_219157

theorem bernardo_wins (N : ℕ) : N = 63 ↔ 
  N ≤ 1999 ∧ 
  (∀ m : ℕ, m < N → 
    (3*m < 3000 ∧
     3*m + 100 < 3000 ∧
     9*m + 300 < 3000 ∧
     9*m + 400 < 3000 ∧
     27*m + 1200 < 3000 ∧
     27*m + 1300 < 3000)) ∧
  (3*N < 3000 ∧
   3*N + 100 < 3000 ∧
   9*N + 300 < 3000 ∧
   9*N + 400 < 3000 ∧
   27*N + 1200 < 3000 ∧
   27*N + 1300 ≥ 3000) :=
by sorry

end NUMINAMATH_CALUDE_bernardo_wins_l2191_219157


namespace NUMINAMATH_CALUDE_initial_orchids_l2191_219131

theorem initial_orchids (initial_roses : ℕ) (final_orchids : ℕ) (final_roses : ℕ) :
  initial_roses = 9 →
  final_orchids = 13 →
  final_roses = 3 →
  final_orchids - final_roses = 10 →
  ∃ initial_orchids : ℕ, initial_orchids = 3 :=
by sorry

end NUMINAMATH_CALUDE_initial_orchids_l2191_219131


namespace NUMINAMATH_CALUDE_function_equality_l2191_219197

theorem function_equality (f g h : ℕ → ℕ) 
  (h_injective : Function.Injective h)
  (g_surjective : Function.Surjective g)
  (f_def : ∀ n, f n = g n - h n + 1) :
  ∀ n, f n = 1 :=
sorry

end NUMINAMATH_CALUDE_function_equality_l2191_219197


namespace NUMINAMATH_CALUDE_additional_correct_answers_needed_l2191_219125

def total_problems : ℕ := 80
def arithmetic_problems : ℕ := 15
def algebra_problems : ℕ := 25
def geometry_problems : ℕ := 40
def arithmetic_correct_ratio : ℚ := 4/5
def algebra_correct_ratio : ℚ := 1/2
def geometry_correct_ratio : ℚ := 11/20
def passing_grade_ratio : ℚ := 13/20

def correct_answers : ℕ := 
  (arithmetic_problems * arithmetic_correct_ratio).ceil.toNat +
  (algebra_problems * algebra_correct_ratio).ceil.toNat +
  (geometry_problems * geometry_correct_ratio).ceil.toNat

def passing_threshold : ℕ := (total_problems * passing_grade_ratio).ceil.toNat

theorem additional_correct_answers_needed : 
  passing_threshold - correct_answers = 5 := by sorry

end NUMINAMATH_CALUDE_additional_correct_answers_needed_l2191_219125


namespace NUMINAMATH_CALUDE_work_completion_time_l2191_219160

/-- 
Given:
- A can do a work in 14 days
- A and B together can do the same work in 10 days

Prove that B can do the work alone in 35 days
-/
theorem work_completion_time (work : ℝ) (a_rate b_rate : ℝ) 
  (h1 : a_rate = work / 14)
  (h2 : a_rate + b_rate = work / 10) :
  b_rate = work / 35 :=
sorry

end NUMINAMATH_CALUDE_work_completion_time_l2191_219160


namespace NUMINAMATH_CALUDE_yeast_growth_proof_l2191_219121

def yeast_population (initial_population : ℕ) (growth_factor : ℕ) (interval : ℕ) (time : ℕ) : ℕ :=
  initial_population * growth_factor ^ (time / interval)

theorem yeast_growth_proof (initial_population : ℕ) (growth_factor : ℕ) (interval : ℕ) (time : ℕ) :
  initial_population = 50 →
  growth_factor = 3 →
  interval = 5 →
  time = 20 →
  yeast_population initial_population growth_factor interval time = 4050 :=
by
  sorry

end NUMINAMATH_CALUDE_yeast_growth_proof_l2191_219121


namespace NUMINAMATH_CALUDE_smallest_n_same_factors_l2191_219175

/-- Count the number of factors of a natural number -/
def countFactors (n : ℕ) : ℕ := sorry

/-- Check if three consecutive numbers have the same number of factors -/
def sameFactorCount (n : ℕ) : Prop :=
  countFactors n = countFactors (n + 1) ∧ countFactors n = countFactors (n + 2)

/-- 33 is the smallest natural number n such that n, n+1, and n+2 have the same number of factors -/
theorem smallest_n_same_factors : 
  (∀ m : ℕ, m < 33 → ¬(sameFactorCount m)) ∧ sameFactorCount 33 := by sorry

end NUMINAMATH_CALUDE_smallest_n_same_factors_l2191_219175


namespace NUMINAMATH_CALUDE_distance_p_to_y_axis_l2191_219145

/-- The distance from a point to the y-axis is the absolute value of its x-coordinate. -/
def distance_to_y_axis (x y : ℝ) : ℝ := |x|

/-- Given point P(-3, 5), prove that its distance to the y-axis is 3. -/
theorem distance_p_to_y_axis :
  let P : ℝ × ℝ := (-3, 5)
  distance_to_y_axis P.1 P.2 = 3 := by
  sorry

end NUMINAMATH_CALUDE_distance_p_to_y_axis_l2191_219145


namespace NUMINAMATH_CALUDE_add_preserves_inequality_l2191_219163

theorem add_preserves_inequality (a b : ℝ) (h : a > b) : a + 2 > b + 2 := by
  sorry

end NUMINAMATH_CALUDE_add_preserves_inequality_l2191_219163


namespace NUMINAMATH_CALUDE_factor_t_squared_minus_144_l2191_219119

theorem factor_t_squared_minus_144 (t : ℝ) : t^2 - 144 = (t - 12) * (t + 12) := by
  sorry

end NUMINAMATH_CALUDE_factor_t_squared_minus_144_l2191_219119


namespace NUMINAMATH_CALUDE_newspaper_conference_max_overlap_l2191_219116

theorem newspaper_conference_max_overlap (total : ℕ) (writers : ℕ) (editors : ℕ) (x : ℕ) :
  total = 100 →
  writers = 40 →
  editors > 38 →
  (total = writers + editors - x + 2 * x) →
  (∀ y : ℕ, y > x → ¬(total = writers + editors - y + 2 * y)) →
  x = 21 :=
by sorry

end NUMINAMATH_CALUDE_newspaper_conference_max_overlap_l2191_219116


namespace NUMINAMATH_CALUDE_grinder_purchase_price_l2191_219199

theorem grinder_purchase_price 
  (x : ℝ) -- purchase price of grinder
  (h1 : 0.96 * x + 9200 = x + 8600) -- equation representing the overall transaction
  : x = 15000 := by
  sorry

end NUMINAMATH_CALUDE_grinder_purchase_price_l2191_219199


namespace NUMINAMATH_CALUDE_rachel_homework_l2191_219191

/-- Rachel's homework problem -/
theorem rachel_homework (math_pages reading_pages total_pages : ℕ) : 
  math_pages = 8 ∧ 
  math_pages = reading_pages + 3 →
  total_pages = math_pages + reading_pages ∧
  total_pages = 13 := by
  sorry

end NUMINAMATH_CALUDE_rachel_homework_l2191_219191


namespace NUMINAMATH_CALUDE_expected_value_is_negative_half_l2191_219161

/-- A three-sided coin with probabilities and payoffs -/
structure ThreeSidedCoin where
  prob_heads : ℚ
  prob_tails : ℚ
  prob_edge : ℚ
  payoff_heads : ℚ
  payoff_tails : ℚ
  payoff_edge : ℚ

/-- Expected value of winnings for a three-sided coin -/
def expected_value (coin : ThreeSidedCoin) : ℚ :=
  coin.prob_heads * coin.payoff_heads +
  coin.prob_tails * coin.payoff_tails +
  coin.prob_edge * coin.payoff_edge

/-- Theorem: Expected value of winnings for the given coin is -1/2 -/
theorem expected_value_is_negative_half :
  let coin : ThreeSidedCoin := {
    prob_heads := 1/4,
    prob_tails := 2/4,
    prob_edge := 1/4,
    payoff_heads := 4,
    payoff_tails := -3,
    payoff_edge := 0
  }
  expected_value coin = -1/2 := by
  sorry

end NUMINAMATH_CALUDE_expected_value_is_negative_half_l2191_219161


namespace NUMINAMATH_CALUDE_task_completion_correct_l2191_219151

/-- Represents the number of days it takes for a person to complete the task alone -/
structure PersonWorkRate where
  days : ℝ
  days_positive : days > 0

/-- Represents the scenario of two people working on a task -/
structure WorkScenario where
  person_a : PersonWorkRate
  person_b : PersonWorkRate
  days_a_alone : ℝ
  days_together : ℝ
  days_a_alone_nonnegative : days_a_alone ≥ 0
  days_together_nonnegative : days_together ≥ 0

/-- The equation representing the completion of the task -/
def task_completion_equation (scenario : WorkScenario) : Prop :=
  (scenario.days_together + scenario.days_a_alone) / scenario.person_a.days +
  scenario.days_together / scenario.person_b.days = 1

/-- The theorem stating that the given equation correctly represents the completion of the task -/
theorem task_completion_correct (scenario : WorkScenario)
  (h1 : scenario.person_a.days = 3)
  (h2 : scenario.person_b.days = 5)
  (h3 : scenario.days_a_alone = 1) :
  task_completion_equation scenario :=
sorry

end NUMINAMATH_CALUDE_task_completion_correct_l2191_219151


namespace NUMINAMATH_CALUDE_expression_simplification_l2191_219105

theorem expression_simplification (x y : ℝ) (hx : x = 2) (hy : y = 2016) :
  (3 * x + 2 * y) * (3 * x - 2 * y) - (x + 2 * y) * (5 * x - 2 * y) / (8 * x) = -2015 :=
by sorry

end NUMINAMATH_CALUDE_expression_simplification_l2191_219105


namespace NUMINAMATH_CALUDE_inequality_proof_l2191_219140

/-- For all real x greater than -1, 1 - e^(-x) is greater than or equal to x/(x+1) -/
theorem inequality_proof (x : ℝ) (h : x > -1) : 1 - Real.exp (-x) ≥ x / (x + 1) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l2191_219140


namespace NUMINAMATH_CALUDE_min_force_to_prevent_slipping_l2191_219148

/-- The minimum force needed to keep a book from slipping -/
theorem min_force_to_prevent_slipping 
  (M : ℝ) -- Mass of the book
  (g : ℝ) -- Acceleration due to gravity
  (μs : ℝ) -- Coefficient of static friction
  (h1 : M > 0) -- Mass is positive
  (h2 : g > 0) -- Gravity is positive
  (h3 : μs > 0) -- Coefficient of static friction is positive
  : 
  ∃ (F : ℝ), F = M * g / μs ∧ F ≥ M * g ∧ ∀ (F' : ℝ), F' < F → F' * μs < M * g :=
sorry

end NUMINAMATH_CALUDE_min_force_to_prevent_slipping_l2191_219148


namespace NUMINAMATH_CALUDE_pauls_weekly_spending_l2191_219102

/-- Given Paul's earnings and the duration the money lasted, calculate his weekly spending. -/
theorem pauls_weekly_spending (lawn_mowing : ℕ) (weed_eating : ℕ) (weeks : ℕ) 
  (h1 : lawn_mowing = 44)
  (h2 : weed_eating = 28)
  (h3 : weeks = 8)
  (h4 : weeks > 0) :
  (lawn_mowing + weed_eating) / weeks = 9 := by
  sorry

#check pauls_weekly_spending

end NUMINAMATH_CALUDE_pauls_weekly_spending_l2191_219102


namespace NUMINAMATH_CALUDE_sons_age_l2191_219162

theorem sons_age (father_age son_age : ℕ) : 
  father_age = son_age + 26 →
  father_age + 2 = 2 * (son_age + 2) →
  son_age = 24 := by
sorry

end NUMINAMATH_CALUDE_sons_age_l2191_219162


namespace NUMINAMATH_CALUDE_similar_triangles_solution_l2191_219193

/-- Two similar right triangles with legs 15 and 12 in the first triangle, 
    and y and 9 in the second triangle. -/
def similar_triangles (y : ℝ) : Prop :=
  15 / y = 12 / 9

theorem similar_triangles_solution :
  ∃ y : ℝ, similar_triangles y ∧ y = 11.25 := by
  sorry

end NUMINAMATH_CALUDE_similar_triangles_solution_l2191_219193


namespace NUMINAMATH_CALUDE_factorization_equality_minimum_value_minimum_achieved_l2191_219129

-- Problem 1
theorem factorization_equality (m n : ℝ) : 
  m^2 - 4*m*n + 3*n^2 = (m - 3*n) * (m - n) := by sorry

-- Problem 2
theorem minimum_value (m : ℝ) : 
  m^2 - 3*m + 2015 ≥ 2012 + 3/4 := by sorry

-- The minimum is achievable
theorem minimum_achieved (ε : ℝ) (hε : ε > 0) : 
  ∃ m : ℝ, m^2 - 3*m + 2015 < 2012 + 3/4 + ε := by sorry

end NUMINAMATH_CALUDE_factorization_equality_minimum_value_minimum_achieved_l2191_219129


namespace NUMINAMATH_CALUDE_count_valid_labelings_l2191_219189

/-- A labeling of the edges of a rectangular prism with 0s and 1s. -/
def Labeling := Fin 12 → Fin 2

/-- The set of faces of a rectangular prism. -/
def Face := Fin 6

/-- The edges that make up each face of the rectangular prism. -/
def face_edges : Face → Finset (Fin 12) :=
  sorry

/-- The sum of labels on a given face for a given labeling. -/
def face_sum (l : Labeling) (f : Face) : Nat :=
  (face_edges f).sum (fun e => l e)

/-- A labeling is valid if the sum of labels on each face is exactly 2. -/
def is_valid_labeling (l : Labeling) : Prop :=
  ∀ f : Face, face_sum l f = 2

/-- The set of all valid labelings. -/
def valid_labelings : Finset Labeling :=
  sorry

theorem count_valid_labelings :
  valid_labelings.card = 16 :=
sorry

end NUMINAMATH_CALUDE_count_valid_labelings_l2191_219189


namespace NUMINAMATH_CALUDE_perfect_squares_among_options_l2191_219109

def is_perfect_square (n : ℕ) : Prop :=
  ∃ m : ℕ, n = m * m

theorem perfect_squares_among_options :
  (is_perfect_square (3^3 * 4^5 * 7^7) = false) ∧
  (is_perfect_square (3^4 * 4^4 * 7^6) = true) ∧
  (is_perfect_square (3^6 * 4^3 * 7^8) = true) ∧
  (is_perfect_square (3^5 * 4^6 * 7^5) = false) ∧
  (is_perfect_square (3^4 * 4^6 * 7^7) = false) :=
by sorry

end NUMINAMATH_CALUDE_perfect_squares_among_options_l2191_219109


namespace NUMINAMATH_CALUDE_inequality_solution_l2191_219127

-- Define the inequality function
def f (a : ℝ) (x : ℝ) : Prop := a * x^2 + (1 - a) * x - 1 > 0

-- Define the solution set
def solution_set (a : ℝ) : Set ℝ :=
  if -1 < a ∧ a < 0 then {x | 1 < x ∧ x < -1/a}
  else if a = -1 then ∅
  else if a < -1 then {x | -1/a < x ∧ x < 1}
  else ∅

-- Theorem statement
theorem inequality_solution (a : ℝ) (h : a < 0) :
  {x : ℝ | f a x} = solution_set a :=
sorry

end NUMINAMATH_CALUDE_inequality_solution_l2191_219127


namespace NUMINAMATH_CALUDE_negative_a_squared_times_a_fourth_l2191_219103

theorem negative_a_squared_times_a_fourth (a : ℝ) : (-a)^2 * a^4 = a^6 := by
  sorry

end NUMINAMATH_CALUDE_negative_a_squared_times_a_fourth_l2191_219103


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_and_negation_l2191_219156

theorem sufficient_not_necessary_and_negation :
  (∀ a : ℝ, a > 1 → (1 / a < 1)) ∧
  (∃ a : ℝ, 1 / a < 1 ∧ a ≤ 1) ∧
  (¬(∀ x : ℝ, x^2 + x + 1 < 0) ↔ (∃ x : ℝ, x^2 + x + 1 ≥ 0)) :=
by sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_and_negation_l2191_219156


namespace NUMINAMATH_CALUDE_ratio_of_divisor_sums_l2191_219180

def M : ℕ := 36 * 36 * 75 * 224

def sum_odd_divisors (n : ℕ) : ℕ := sorry
def sum_even_divisors (n : ℕ) : ℕ := sorry

theorem ratio_of_divisor_sums :
  (sum_odd_divisors M : ℚ) / (sum_even_divisors M : ℚ) = 1 / 510 := by sorry

end NUMINAMATH_CALUDE_ratio_of_divisor_sums_l2191_219180


namespace NUMINAMATH_CALUDE_new_person_weight_l2191_219190

theorem new_person_weight (initial_count : ℕ) (weight_increase : ℝ) (replaced_weight : ℝ) :
  initial_count = 8 →
  weight_increase = 2.5 →
  replaced_weight = 85 →
  ∃ (new_weight : ℝ), new_weight = 105 ∧
    (initial_count : ℝ) * weight_increase = new_weight - replaced_weight :=
by sorry

end NUMINAMATH_CALUDE_new_person_weight_l2191_219190


namespace NUMINAMATH_CALUDE_area_18_rectangles_l2191_219184

def rectangle_pairs : Set (ℕ × ℕ) :=
  {(1, 18), (2, 9), (3, 6), (6, 3), (9, 2), (18, 1)}

theorem area_18_rectangles :
  ∀ (w l : ℕ), w > 0 → l > 0 → w * l = 18 ↔ (w, l) ∈ rectangle_pairs := by
  sorry

end NUMINAMATH_CALUDE_area_18_rectangles_l2191_219184


namespace NUMINAMATH_CALUDE_impossibleTo2012_l2191_219195

/-- Represents a 5x5 board with integer values -/
def Board := Fin 5 → Fin 5 → ℤ

/-- Checks if two cells are adjacent (share a common side) -/
def adjacent (i j i' j' : Fin 5) : Bool :=
  (i = i' ∧ (j.val + 1 = j'.val ∨ j.val = j'.val + 1)) ∨
  (j = j' ∧ (i.val + 1 = i'.val ∨ i.val = i'.val + 1))

/-- Represents a single move on the board -/
def move (b : Board) (i j : Fin 5) : Board :=
  fun i' j' => if i' = i ∧ j' = j ∨ adjacent i j i' j' then b i' j' + 1 else b i' j'

/-- Checks if all cells in the board have the value 2012 -/
def allCells2012 (b : Board) : Prop :=
  ∀ i j, b i j = 2012

/-- The initial board with all cells set to zero -/
def initialBoard : Board :=
  fun _ _ => 0

theorem impossibleTo2012 : ¬ ∃ (moves : List (Fin 5 × Fin 5)), 
  allCells2012 (moves.foldl (fun b (i, j) => move b i j) initialBoard) :=
sorry

end NUMINAMATH_CALUDE_impossibleTo2012_l2191_219195


namespace NUMINAMATH_CALUDE_quadratic_roots_relation_l2191_219113

theorem quadratic_roots_relation (x₁ x₂ : ℝ) : 
  (3 * x₁^2 - 5 * x₁ - 7 = 0) → 
  (3 * x₂^2 - 5 * x₂ - 7 = 0) → 
  (x₁ + x₂ = 5/3) ∧ (x₁ * x₂ = -7/3) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_roots_relation_l2191_219113


namespace NUMINAMATH_CALUDE_sum_of_digits_l2191_219155

theorem sum_of_digits (a₁ a₂ b c : ℕ) :
  a₁ < 10 → a₂ < 10 → b < 10 → c < 10 →
  100 * (10 * a₁ + a₂) + 10 * b + 7 * c = 2024 →
  a₁ + a₂ + b + c = 5 :=
by sorry

end NUMINAMATH_CALUDE_sum_of_digits_l2191_219155


namespace NUMINAMATH_CALUDE_journey_times_equal_l2191_219169

-- Define the variables
def distance1 : ℝ := 120
def distance2 : ℝ := 240

-- Define the theorem
theorem journey_times_equal (speed1 : ℝ) (h1 : speed1 > 0) :
  distance1 / speed1 = distance2 / (2 * speed1) :=
by sorry

end NUMINAMATH_CALUDE_journey_times_equal_l2191_219169


namespace NUMINAMATH_CALUDE_smaller_rectangle_area_percentage_l2191_219141

/-- A circle with a rectangle inscribed in it -/
structure InscribedRectangle where
  center : ℝ × ℝ
  radius : ℝ
  rect_length : ℝ
  rect_width : ℝ

/-- A smaller rectangle with one side coinciding with the larger rectangle and two vertices on the circle -/
structure SmallerRectangle where
  length : ℝ
  width : ℝ

/-- The configuration of the inscribed rectangle and the smaller rectangle -/
structure Configuration where
  inscribed : InscribedRectangle
  smaller : SmallerRectangle

/-- The theorem stating that the area of the smaller rectangle is 0% of the area of the larger rectangle -/
theorem smaller_rectangle_area_percentage (config : Configuration) : 
  (config.smaller.length * config.smaller.width) / (config.inscribed.rect_length * config.inscribed.rect_width) = 0 := by
  sorry

end NUMINAMATH_CALUDE_smaller_rectangle_area_percentage_l2191_219141


namespace NUMINAMATH_CALUDE_other_endpoint_of_diameter_l2191_219117

/-- A circle in a 2D coordinate plane --/
structure Circle where
  center : ℝ × ℝ

/-- A diameter of a circle --/
structure Diameter where
  circle : Circle
  endpoint1 : ℝ × ℝ
  endpoint2 : ℝ × ℝ

/-- The given circle P --/
def circleP : Circle :=
  { center := (3, 4) }

/-- The diameter of circle P --/
def diameterP : Diameter :=
  { circle := circleP
    endpoint1 := (0, 0)
    endpoint2 := (-3, -4) }

/-- Theorem: The other endpoint of the diameter is at (-3, -4) --/
theorem other_endpoint_of_diameter :
  diameterP.endpoint2 = (-3, -4) := by
  sorry

#check other_endpoint_of_diameter

end NUMINAMATH_CALUDE_other_endpoint_of_diameter_l2191_219117


namespace NUMINAMATH_CALUDE_least_subtraction_for_divisibility_solution_9876543210_and_29_l2191_219133

theorem least_subtraction_for_divisibility (n : ℕ) (d : ℕ) (h : d > 0) :
  let r := n % d
  (∃ (k : ℕ), (n - r) = d * k) ∧ (∀ (m : ℕ), m < r → ¬(∃ (k : ℕ), (n - m) = d * k)) :=
by
  sorry

theorem solution_9876543210_and_29 :
  let n : ℕ := 9876543210
  let d : ℕ := 29
  let r : ℕ := n % d
  r = 6 ∧
  (∃ (k : ℕ), (n - r) = d * k) ∧
  (∀ (m : ℕ), m < r → ¬(∃ (k : ℕ), (n - m) = d * k)) :=
by
  sorry

end NUMINAMATH_CALUDE_least_subtraction_for_divisibility_solution_9876543210_and_29_l2191_219133


namespace NUMINAMATH_CALUDE_nigel_money_problem_l2191_219178

theorem nigel_money_problem (initial_amount : ℕ) (mother_gift : ℕ) (final_amount : ℕ) : 
  initial_amount = 45 →
  mother_gift = 80 →
  final_amount = 2 * initial_amount + 10 →
  initial_amount - (final_amount - mother_gift) = 25 :=
by
  sorry

end NUMINAMATH_CALUDE_nigel_money_problem_l2191_219178


namespace NUMINAMATH_CALUDE_vector_equation_solution_l2191_219174

/-- Given real numbers a and b satisfying a vector equation, prove they equal specific values. -/
theorem vector_equation_solution :
  ∀ (a b : ℝ),
  (![3, 2] : Fin 2 → ℝ) + a • ![6, -4] = ![(-1), 1] + b • ![(-3), 5] →
  a = -23/18 ∧ b = 5/9 := by
  sorry

end NUMINAMATH_CALUDE_vector_equation_solution_l2191_219174


namespace NUMINAMATH_CALUDE_min_value_theorem_l2191_219120

theorem min_value_theorem (a b : ℝ) (ha : a > 0) (hb : b > 0) (h_line : 6 * a + 3 * b = 1) :
  1 / (5 * a + 2 * b) + 2 / (a + b) ≥ 3 + 2 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_min_value_theorem_l2191_219120


namespace NUMINAMATH_CALUDE_potato_shipment_l2191_219124

/-- The initial amount of potatoes shipped in kg -/
def initial_potatoes : ℕ := 6500

/-- The amount of damaged potatoes in kg -/
def damaged_potatoes : ℕ := 150

/-- The weight of each bag of potatoes in kg -/
def bag_weight : ℕ := 50

/-- The price of each bag of potatoes in dollars -/
def bag_price : ℕ := 72

/-- The total revenue from selling the potatoes in dollars -/
def total_revenue : ℕ := 9144

theorem potato_shipment :
  initial_potatoes = 
    (total_revenue / bag_price) * bag_weight + damaged_potatoes :=
by sorry

end NUMINAMATH_CALUDE_potato_shipment_l2191_219124


namespace NUMINAMATH_CALUDE_system_equations_properties_l2191_219166

theorem system_equations_properties (x y a : ℝ) 
  (eq1 : 3 * x + 2 * y = 8 + a) 
  (eq2 : 2 * x + 3 * y = 3 * a) : 
  (x = -y → a = -2) ∧ 
  (x - y = 8 - 2 * a) ∧ 
  (7 * x + 3 * y = 24) ∧ 
  (x = -3/7 * y + 24/7) := by
sorry

end NUMINAMATH_CALUDE_system_equations_properties_l2191_219166


namespace NUMINAMATH_CALUDE_exists_circumscribing_square_l2191_219179

/-- A type representing a bounded convex shape in a plane -/
structure BoundedConvexShape where
  -- Add necessary fields/axioms to define a bounded convex shape
  is_bounded : Bool
  is_convex : Bool

/-- A type representing a square in a plane -/
structure Square where
  -- Add necessary fields to define a square

/-- Predicate to check if a square circumscribes a bounded convex shape -/
def circumscribes (s : Square) (shape : BoundedConvexShape) : Prop :=
  sorry -- Define the circumscription condition

/-- Theorem stating that every bounded convex shape can be circumscribed by a square -/
theorem exists_circumscribing_square (shape : BoundedConvexShape) :
  shape.is_bounded ∧ shape.is_convex → ∃ s : Square, circumscribes s shape := by
  sorry


end NUMINAMATH_CALUDE_exists_circumscribing_square_l2191_219179


namespace NUMINAMATH_CALUDE_intersection_M_N_l2191_219146

-- Define set M
def M : Set ℝ := {x | ∃ y, y = Real.sqrt (x - x^2)}

-- Define set N
def N : Set ℝ := {y | ∃ x, y = Real.sin x}

-- Theorem statement
theorem intersection_M_N : M ∩ N = Set.Icc 0 1 := by
  sorry

end NUMINAMATH_CALUDE_intersection_M_N_l2191_219146


namespace NUMINAMATH_CALUDE_fractional_inequality_solution_set_l2191_219159

theorem fractional_inequality_solution_set (x : ℝ) : 
  (x - 1) / (2 * x + 3) > 1 ↔ -4 < x ∧ x < -3/2 :=
by sorry

end NUMINAMATH_CALUDE_fractional_inequality_solution_set_l2191_219159


namespace NUMINAMATH_CALUDE_tree_distance_l2191_219123

theorem tree_distance (n : ℕ) (d : ℝ) (h1 : n = 8) (h2 : d = 80) :
  let distance_between (i j : ℕ) := d * (j - i) / 4
  distance_between 1 n = 140 := by
  sorry

end NUMINAMATH_CALUDE_tree_distance_l2191_219123


namespace NUMINAMATH_CALUDE_complex_magnitude_power_four_l2191_219167

theorem complex_magnitude_power_four : 
  Complex.abs ((1 - Complex.I * Real.sqrt 3) ^ 4) = 16 := by sorry

end NUMINAMATH_CALUDE_complex_magnitude_power_four_l2191_219167


namespace NUMINAMATH_CALUDE_complex_equation_solution_l2191_219186

theorem complex_equation_solution (a b : ℝ) 
  (h : (a - 1 : ℂ) + 2*a*I = -4 + b*I) : b = -6 := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l2191_219186


namespace NUMINAMATH_CALUDE_smallest_fraction_between_l2191_219152

theorem smallest_fraction_between (p q : ℕ+) : 
  (3 : ℚ) / 5 < (p : ℚ) / q ∧ 
  (p : ℚ) / q < 5 / 8 ∧ 
  (∀ (r s : ℕ+), (3 : ℚ) / 5 < (r : ℚ) / s ∧ (r : ℚ) / s < 5 / 8 → s ≥ q) →
  q - p = 5 := by
  sorry

end NUMINAMATH_CALUDE_smallest_fraction_between_l2191_219152


namespace NUMINAMATH_CALUDE_circle_through_point_on_x_axis_l2191_219101

def circle_equation (center : ℝ × ℝ) (radius : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | (p.1 - center.1)^2 + (p.2 - center.2)^2 = radius^2}

theorem circle_through_point_on_x_axis 
  (center : ℝ × ℝ) 
  (h_center_on_x_axis : center.2 = 0) 
  (h_radius : radius = 1) 
  (h_through_point : (2, 1) ∈ circle_equation center radius) :
  circle_equation center radius = circle_equation (2, 0) 1 := by
sorry

end NUMINAMATH_CALUDE_circle_through_point_on_x_axis_l2191_219101


namespace NUMINAMATH_CALUDE_average_minutes_theorem_l2191_219165

/-- Represents the distribution of attendees and their listening durations --/
structure LectureAttendance where
  total_attendees : ℕ
  full_listeners : ℕ
  sleepers : ℕ
  half_listeners : ℕ
  quarter_listeners : ℕ
  lecture_duration : ℕ

/-- Calculates the average minutes heard by attendees --/
def average_minutes_heard (attendance : LectureAttendance) : ℚ :=
  let full_minutes := attendance.full_listeners * attendance.lecture_duration
  let half_minutes := attendance.half_listeners * (attendance.lecture_duration / 2)
  let quarter_minutes := attendance.quarter_listeners * (attendance.lecture_duration / 4)
  let total_minutes := full_minutes + half_minutes + quarter_minutes
  (total_minutes : ℚ) / attendance.total_attendees

/-- The theorem stating the average minutes heard is 59.1 --/
theorem average_minutes_theorem (attendance : LectureAttendance) 
  (h1 : attendance.lecture_duration = 120)
  (h2 : attendance.full_listeners = attendance.total_attendees * 30 / 100)
  (h3 : attendance.sleepers = attendance.total_attendees * 15 / 100)
  (h4 : attendance.half_listeners = (attendance.total_attendees - attendance.full_listeners - attendance.sleepers) * 40 / 100)
  (h5 : attendance.quarter_listeners = attendance.total_attendees - attendance.full_listeners - attendance.sleepers - attendance.half_listeners) :
  average_minutes_heard attendance = 591/10 := by
  sorry

end NUMINAMATH_CALUDE_average_minutes_theorem_l2191_219165


namespace NUMINAMATH_CALUDE_exists_special_quadrilateral_l2191_219182

/-- Represents a quadrilateral with its properties -/
structure Quadrilateral where
  sides : Fin 4 → ℕ
  diagonals : Fin 2 → ℕ
  area : ℕ
  radius : ℕ

/-- Predicate to check if the quadrilateral is cyclic -/
def isCyclic (q : Quadrilateral) : Prop := sorry

/-- Predicate to check if the side lengths are pairwise distinct -/
def hasPairwiseDistinctSides (q : Quadrilateral) : Prop :=
  ∀ i j, i ≠ j → q.sides i ≠ q.sides j

/-- Theorem stating the existence of a quadrilateral with the required properties -/
theorem exists_special_quadrilateral :
  ∃ q : Quadrilateral,
    isCyclic q ∧
    hasPairwiseDistinctSides q :=
  sorry

end NUMINAMATH_CALUDE_exists_special_quadrilateral_l2191_219182
