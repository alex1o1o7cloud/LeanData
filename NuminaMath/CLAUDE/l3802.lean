import Mathlib

namespace NUMINAMATH_CALUDE_multiple_y_solutions_l3802_380249

theorem multiple_y_solutions : ∃ (y₁ y₂ : ℝ), y₁ ≠ y₂ ∧
  (∃ (x₁ : ℝ), x₁^2 + y₁^2 - 10 = 0 ∧ x₁^2 - x₁*y₁ - 3*y₁ + 12 = 0) ∧
  (∃ (x₂ : ℝ), x₂^2 + y₂^2 - 10 = 0 ∧ x₂^2 - x₂*y₂ - 3*y₂ + 12 = 0) :=
by sorry

end NUMINAMATH_CALUDE_multiple_y_solutions_l3802_380249


namespace NUMINAMATH_CALUDE_average_and_difference_l3802_380292

theorem average_and_difference (x : ℝ) : 
  (23 + x) / 2 = 27 → |x - 23| = 8 := by
sorry

end NUMINAMATH_CALUDE_average_and_difference_l3802_380292


namespace NUMINAMATH_CALUDE_consecutive_integers_sum_l3802_380275

theorem consecutive_integers_sum (x : ℤ) : 
  x * (x + 1) * (x + 2) = 336 → x + (x + 1) + (x + 2) = 21 := by
sorry

end NUMINAMATH_CALUDE_consecutive_integers_sum_l3802_380275


namespace NUMINAMATH_CALUDE_square_difference_133_l3802_380286

theorem square_difference_133 : 
  ∃ (a b c d : ℕ), 
    a * a - b * b = 133 ∧ 
    c * c - d * d = 133 ∧ 
    a > b ∧ c > d ∧ 
    (a ≠ c ∨ b ≠ d) := by
  sorry

end NUMINAMATH_CALUDE_square_difference_133_l3802_380286


namespace NUMINAMATH_CALUDE_dessert_coffee_probability_l3802_380208

theorem dessert_coffee_probability
  (p_dessert_and_coffee : ℝ)
  (p_no_dessert : ℝ)
  (h1 : p_dessert_and_coffee = 0.6)
  (h2 : p_no_dessert = 0.2500000000000001) :
  p_dessert_and_coffee + (1 - p_no_dessert - p_dessert_and_coffee) = 0.75 :=
by sorry

end NUMINAMATH_CALUDE_dessert_coffee_probability_l3802_380208


namespace NUMINAMATH_CALUDE_division_problem_l3802_380279

theorem division_problem (x y z total : ℚ) : 
  x / y = 5 / 7 →
  x / z = 5 / 11 →
  y = 150 →
  total = x + y + z →
  total = 493 := by
sorry

end NUMINAMATH_CALUDE_division_problem_l3802_380279


namespace NUMINAMATH_CALUDE_ratio_of_a_to_c_l3802_380297

theorem ratio_of_a_to_c (a b c : ℚ) 
  (h1 : a / b = 5 / 3) 
  (h2 : b / c = 1 / 5) : 
  a / c = 1 / 3 := by
  sorry

end NUMINAMATH_CALUDE_ratio_of_a_to_c_l3802_380297


namespace NUMINAMATH_CALUDE_polynomial_simplification_l3802_380265

theorem polynomial_simplification (x : ℝ) :
  (2*x^6 + 3*x^5 + x^4 + 3*x^3 + 2*x + 15) - (x^6 + 4*x^5 + 2*x^3 - x^2 + 5) =
  x^6 - x^5 + x^4 + x^3 + x^2 + 2*x + 10 := by
sorry

end NUMINAMATH_CALUDE_polynomial_simplification_l3802_380265


namespace NUMINAMATH_CALUDE_least_five_digit_divisible_by_18_12_15_l3802_380210

theorem least_five_digit_divisible_by_18_12_15 :
  ∀ n : ℕ, 10000 ≤ n ∧ n < 100000 ∧ 18 ∣ n ∧ 12 ∣ n ∧ 15 ∣ n → n ≥ 10080 :=
by sorry

end NUMINAMATH_CALUDE_least_five_digit_divisible_by_18_12_15_l3802_380210


namespace NUMINAMATH_CALUDE_symmetric_axis_of_translated_sine_l3802_380244

theorem symmetric_axis_of_translated_sine (f g : ℝ → ℝ) :
  (∀ x, f x = Real.sin (2 * x - π / 6)) →
  (∀ x, g x = f (x - π / 4)) →
  (∀ x, g x = Real.sin (2 * x - 2 * π / 3)) →
  (π / 12 : ℝ) ∈ {x | ∀ y, g (x + y) = g (x - y)} :=
by sorry

end NUMINAMATH_CALUDE_symmetric_axis_of_translated_sine_l3802_380244


namespace NUMINAMATH_CALUDE_gcd_2210_145_l3802_380245

theorem gcd_2210_145 : Int.gcd 2210 145 = 5 := by
  sorry

end NUMINAMATH_CALUDE_gcd_2210_145_l3802_380245


namespace NUMINAMATH_CALUDE_inequality_proof_l3802_380238

theorem inequality_proof (a b c : ℝ) 
  (h1 : c < b) (h2 : b < a) (h3 : a + b + c = 0) : 
  c * b^2 ≤ a * b^2 ∧ a * b > a * c := by sorry

end NUMINAMATH_CALUDE_inequality_proof_l3802_380238


namespace NUMINAMATH_CALUDE_distribute_five_to_three_l3802_380276

def distribute_objects (n : ℕ) (k : ℕ) : ℕ :=
  let c311 := (n.choose 3) * ((n - 3).choose 1) * ((n - 4).choose 1) / 2
  let c221 := (n.choose 2) * ((n - 2).choose 2) * ((n - 4).choose 1) / 2
  (c311 + c221) * 6

theorem distribute_five_to_three :
  distribute_objects 5 3 = 150 := by sorry

end NUMINAMATH_CALUDE_distribute_five_to_three_l3802_380276


namespace NUMINAMATH_CALUDE_sequence_properties_l3802_380234

def sequence_a (n : ℕ+) : ℚ := (1/2) ^ (n.val - 2)

def sum_S (n : ℕ+) : ℚ := 4 * (1 - (1/2) ^ n.val)

theorem sequence_properties :
  ∀ (n : ℕ+),
  (∀ (m : ℕ+), sum_S (m + 1) = (1/2) * sum_S m + 2) →
  sequence_a 1 = 2 →
  sequence_a 2 = 1 →
  (∀ (k : ℕ+), sequence_a k = (1/2) ^ (k.val - 2)) ∧
  (∀ (t : ℕ+), (∀ (n : ℕ+), (sequence_a t * sum_S (n + 1) - 1) / (sequence_a t * sequence_a (n + 1) - 1) < 1/2) ↔ (t = 3 ∨ t = 4)) ∧
  (∀ (m n k : ℕ+), m ≠ n → n ≠ k → m ≠ k → sequence_a m + sequence_a n ≠ sequence_a k) :=
by sorry

end NUMINAMATH_CALUDE_sequence_properties_l3802_380234


namespace NUMINAMATH_CALUDE_train_length_l3802_380251

/-- The length of a train given its speed and time to pass a fixed point -/
theorem train_length (speed : ℝ) (time : ℝ) : 
  speed = 63 → time = 32 → (speed * (5/18) * time) = 373.33 := by
  sorry

end NUMINAMATH_CALUDE_train_length_l3802_380251


namespace NUMINAMATH_CALUDE_arith_progression_poly_j_value_l3802_380230

/-- A polynomial of degree 4 with four distinct real zeros in arithmetic progression -/
structure ArithProgressionPoly where
  j : ℝ
  k : ℝ
  zeros : Fin 4 → ℝ
  distinct : ∀ (i j : Fin 4), i ≠ j → zeros i ≠ zeros j
  arith_prog : ∃ (a d : ℝ), ∀ (i : Fin 4), zeros i = a + i.val * d
  is_zero : ∀ (i : Fin 4), zeros i ^ 4 + j * (zeros i ^ 2) + k * zeros i + 225 = 0

theorem arith_progression_poly_j_value (p : ArithProgressionPoly) : p.j = -50 := by
  sorry

end NUMINAMATH_CALUDE_arith_progression_poly_j_value_l3802_380230


namespace NUMINAMATH_CALUDE_coefficient_x3y4_in_binomial_expansion_l3802_380241

theorem coefficient_x3y4_in_binomial_expansion :
  (Finset.range 8).sum (fun k => (Nat.choose 7 k) * (X : ℕ → ℕ) k * (Y : ℕ → ℕ) (7 - k)) =
  35 * (X : ℕ → ℕ) 3 * (Y : ℕ → ℕ) 4 + 
  (Finset.range 8).sum (fun k => if k ≠ 3 then (Nat.choose 7 k) * (X : ℕ → ℕ) k * (Y : ℕ → ℕ) (7 - k) else 0) :=
by sorry

end NUMINAMATH_CALUDE_coefficient_x3y4_in_binomial_expansion_l3802_380241


namespace NUMINAMATH_CALUDE_quadratic_roots_condition_necessary_not_sufficient_condition_l3802_380212

-- Define the quadratic equation
def quadratic (a x : ℝ) : ℝ := x^2 - 2*a*x + 2*a^2 - a - 6

-- Define the proposition p
def has_real_roots (a : ℝ) : Prop := ∃ x : ℝ, quadratic a x = 0

-- Define the proposition q
def q (m a : ℝ) : Prop := m - 1 ≤ a ∧ a ≤ m + 3

theorem quadratic_roots_condition (a : ℝ) :
  ¬(has_real_roots a) ↔ (a < -2 ∨ a > 3) := by sorry

theorem necessary_not_sufficient_condition (m : ℝ) :
  (∀ a : ℝ, q m a → has_real_roots a) ∧
  (∃ a : ℝ, has_real_roots a ∧ ¬(q m a)) →
  -1 ≤ m ∧ m < 0 := by sorry

end NUMINAMATH_CALUDE_quadratic_roots_condition_necessary_not_sufficient_condition_l3802_380212


namespace NUMINAMATH_CALUDE_product_of_square_roots_of_nine_l3802_380295

theorem product_of_square_roots_of_nine (a b : ℝ) : 
  a ^ 2 = 9 ∧ b ^ 2 = 9 ∧ a ≠ b → a * b = -9 := by
  sorry

end NUMINAMATH_CALUDE_product_of_square_roots_of_nine_l3802_380295


namespace NUMINAMATH_CALUDE_new_circle_externally_tangent_l3802_380217

/-- Given circle equation -/
def given_circle (x y : ℝ) : Prop :=
  x^2 + y^2 + 2*x - 4*y + 1 = 0

/-- Center of the new circle -/
def new_center : ℝ × ℝ := (2, -2)

/-- Equation of the new circle -/
def new_circle (x y : ℝ) : Prop :=
  (x - 2)^2 + (y + 2)^2 = 9

/-- Theorem stating that the new circle is externally tangent to the given circle -/
theorem new_circle_externally_tangent :
  ∃ (x y : ℝ), given_circle x y ∧ new_circle x y ∧
  (∀ (x' y' : ℝ), given_circle x' y' ∧ new_circle x' y' → (x, y) = (x', y')) :=
sorry

end NUMINAMATH_CALUDE_new_circle_externally_tangent_l3802_380217


namespace NUMINAMATH_CALUDE_impossibility_of_sequence_conditions_l3802_380263

def is_valid_sequence (a : ℕ → ℝ) (c : ℝ) : Prop :=
  (∀ n : ℕ, a (n + 3) = a n) ∧
  (∀ n : ℕ, a n * a (n + 3) - a (n + 1) * a (n + 2) = c)

theorem impossibility_of_sequence_conditions : 
  ¬∃ (a : ℕ → ℝ) (c : ℝ), is_valid_sequence a c ∧ a 1 = 2 ∧ c = 2 :=
sorry

end NUMINAMATH_CALUDE_impossibility_of_sequence_conditions_l3802_380263


namespace NUMINAMATH_CALUDE_partial_fraction_decomposition_l3802_380261

theorem partial_fraction_decomposition :
  ∀ x : ℝ, x ≠ 7 → x ≠ -2 →
  (5 * x - 4) / (x^2 - 5*x - 14) = (31/9) / (x - 7) + (14/9) / (x + 2) := by
sorry

end NUMINAMATH_CALUDE_partial_fraction_decomposition_l3802_380261


namespace NUMINAMATH_CALUDE_vector_a_start_point_l3802_380269

/-- The endpoint of vector a -/
def B : ℝ × ℝ := (1, 0)

/-- Vector b -/
def b : ℝ × ℝ := (-3, -4)

/-- Vector c -/
def c : ℝ × ℝ := (1, 1)

/-- Vector a in terms of b and c -/
def a : ℝ × ℝ := (3 * b.1 - 2 * c.1, 3 * b.2 - 2 * c.2)

/-- The starting point of vector a -/
def start_point : ℝ × ℝ := (B.1 - a.1, B.2 - a.2)

theorem vector_a_start_point : start_point = (12, 14) := by
  sorry

end NUMINAMATH_CALUDE_vector_a_start_point_l3802_380269


namespace NUMINAMATH_CALUDE_multiply_by_seven_equals_98_l3802_380259

theorem multiply_by_seven_equals_98 (x : ℝ) : x * 7 = 98 ↔ x = 14 := by
  sorry

end NUMINAMATH_CALUDE_multiply_by_seven_equals_98_l3802_380259


namespace NUMINAMATH_CALUDE_polynomial_value_l3802_380227

theorem polynomial_value (x y : ℝ) (h : x - y = 1) :
  x^4 - x*y^3 - x^3*y - 3*x^2*y + 3*x*y^2 + y^4 = 1 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_value_l3802_380227


namespace NUMINAMATH_CALUDE_omitted_angle_measure_l3802_380289

/-- The sum of interior angles of a polygon with n sides is (n-2) * 180° --/
def sum_interior_angles (n : ℕ) : ℕ := (n - 2) * 180

/-- The property that the sum of interior angles is divisible by 180° --/
def is_valid_sum (s : ℕ) : Prop := ∃ k : ℕ, s = k * 180

/-- The sum calculated by Angela --/
def angela_sum : ℕ := 2583

/-- The theorem to prove --/
theorem omitted_angle_measure :
  ∃ (n : ℕ), 
    n > 2 ∧ 
    is_valid_sum (sum_interior_angles n) ∧ 
    sum_interior_angles n = angela_sum + 117 := by
  sorry

end NUMINAMATH_CALUDE_omitted_angle_measure_l3802_380289


namespace NUMINAMATH_CALUDE_sum_of_squares_l3802_380267

theorem sum_of_squares (x y z p q r : ℝ) 
  (h1 : x + y = p) 
  (h2 : y + z = q) 
  (h3 : z + x = r) 
  (hp : p ≠ 0) 
  (hq : q ≠ 0) 
  (hr : r ≠ 0) : 
  x^2 + y^2 + z^2 = (p^2 + q^2 + r^2 - p*q - q*r - r*p) / 2 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_squares_l3802_380267


namespace NUMINAMATH_CALUDE_inequality_proof_l3802_380252

theorem inequality_proof (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) (hsum : x + y + z = 1) :
  (1 / x^2 - x) * (1 / y^2 - y) * (1 / z^2 - z) ≥ (26 / 3)^3 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l3802_380252


namespace NUMINAMATH_CALUDE_remainder_16_pow_2048_mod_11_l3802_380220

theorem remainder_16_pow_2048_mod_11 : 16^2048 % 11 = 4 := by
  sorry

end NUMINAMATH_CALUDE_remainder_16_pow_2048_mod_11_l3802_380220


namespace NUMINAMATH_CALUDE_quadratic_inequality_implies_range_l3802_380235

theorem quadratic_inequality_implies_range (a : ℝ) :
  (∀ x : ℝ, a * x^2 + 2 * a * x - (a + 2) < 0) →
  -1 < a ∧ a ≤ 0 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_implies_range_l3802_380235


namespace NUMINAMATH_CALUDE_sqrt_inequality_l3802_380232

theorem sqrt_inequality (x : ℝ) :
  x > 0 → (Real.sqrt x > 3 * x - 2 ↔ 4/9 < x ∧ x < 1) := by sorry

end NUMINAMATH_CALUDE_sqrt_inequality_l3802_380232


namespace NUMINAMATH_CALUDE_certain_number_proof_l3802_380271

theorem certain_number_proof : ∃ x : ℕ, (2994 : ℚ) / x = 177 ∧ x = 17 := by
  sorry

end NUMINAMATH_CALUDE_certain_number_proof_l3802_380271


namespace NUMINAMATH_CALUDE_mike_owes_laura_l3802_380272

theorem mike_owes_laura (rate : ℚ) (rooms : ℚ) (total : ℚ) : 
  rate = 13 / 3 → rooms = 8 / 5 → total = rate * rooms → total = 104 / 15 := by
  sorry

end NUMINAMATH_CALUDE_mike_owes_laura_l3802_380272


namespace NUMINAMATH_CALUDE_triangle_base_and_area_l3802_380254

theorem triangle_base_and_area (height : ℝ) (base : ℝ) (h_height : height = 12) 
  (h_ratio : height = 2 / 3 * base) : base = 18 ∧ height * base / 2 = 108 := by
  sorry

end NUMINAMATH_CALUDE_triangle_base_and_area_l3802_380254


namespace NUMINAMATH_CALUDE_isosceles_triangle_theorem_congruent_triangles_theorem_supplementary_angles_not_always_equal_supplements_of_equal_angles_are_equal_proposition_c_is_false_l3802_380218

-- Define the basic geometric concepts
def Triangle : Type := sorry
def Angle : Type := sorry
def Line : Type := sorry

-- Define the properties and relations
def equal_sides (t : Triangle) (s1 s2 : Nat) : Prop := sorry
def equal_angles (t : Triangle) (a1 a2 : Nat) : Prop := sorry
def congruent (t1 t2 : Triangle) : Prop := sorry
def corresponding_sides_equal (t1 t2 : Triangle) : Prop := sorry
def supplementary (a1 a2 : Angle) : Prop := sorry
def parallel (l1 l2 : Line) : Prop := sorry
def supplement_of (a1 a2 : Angle) : Prop := sorry

-- Theorem statements
theorem isosceles_triangle_theorem (t : Triangle) (s1 s2 a1 a2 : Nat) :
  equal_sides t s1 s2 → equal_angles t a1 a2 := sorry

theorem congruent_triangles_theorem (t1 t2 : Triangle) :
  congruent t1 t2 → corresponding_sides_equal t1 t2 := sorry

theorem supplementary_angles_not_always_equal :
  ∃ (a1 a2 : Angle), supplementary a1 a2 ∧ a1 ≠ a2 := sorry

theorem supplements_of_equal_angles_are_equal (a1 a2 a3 a4 : Angle) :
  a1 = a2 → supplement_of a1 a3 → supplement_of a2 a4 → a3 = a4 := sorry

-- The main theorem proving that proposition C is false while others are true
theorem proposition_c_is_false :
  (∀ (t : Triangle) (s1 s2 a1 a2 : Nat), equal_sides t s1 s2 → equal_angles t a1 a2) ∧
  (∀ (t1 t2 : Triangle), congruent t1 t2 → corresponding_sides_equal t1 t2) ∧
  (∃ (a1 a2 : Angle) (l1 l2 : Line), supplementary a1 a2 ∧ a1 ≠ a2 ∧ parallel l1 l2) ∧
  (∀ (a1 a2 a3 a4 : Angle), a1 = a2 → supplement_of a1 a3 → supplement_of a2 a4 → a3 = a4) :=
sorry

end NUMINAMATH_CALUDE_isosceles_triangle_theorem_congruent_triangles_theorem_supplementary_angles_not_always_equal_supplements_of_equal_angles_are_equal_proposition_c_is_false_l3802_380218


namespace NUMINAMATH_CALUDE_oil_price_reduction_l3802_380260

/-- Represents the price reduction percentage of oil -/
def price_reduction : ℚ := 30 / 100

/-- Represents the additional amount of oil that can be bought after the price reduction -/
def additional_oil : ℚ := 9

/-- Represents the fixed amount spent on oil -/
def fixed_amount : ℚ := 900

/-- Represents the price increase percentage of oil compared to rice -/
def price_increase : ℚ := 50 / 100

/-- Represents the prime number that divides the reduced oil price -/
def prime_divisor : ℕ := 5

theorem oil_price_reduction (original_price reduced_price rice_price : ℚ) : 
  reduced_price = original_price * (1 - price_reduction) →
  fixed_amount / original_price - fixed_amount / reduced_price = additional_oil →
  ∃ (n : ℕ), reduced_price = n * prime_divisor →
  original_price = rice_price * (1 + price_increase) →
  original_price = 857142 / 20000 ∧ 
  reduced_price = 30 ∧
  rice_price = 571428 / 20000 := by
  sorry

#eval 857142 / 20000  -- Outputs 42.8571
#eval 571428 / 20000  -- Outputs 28.5714

end NUMINAMATH_CALUDE_oil_price_reduction_l3802_380260


namespace NUMINAMATH_CALUDE_exists_valid_31_min_students_smallest_total_l3802_380268

/-- Represents the number of students in each grade --/
structure GradeCount where
  ninth : ℕ
  tenth : ℕ
  eleventh : ℕ

/-- The ratios between grades are correct --/
def valid_ratios (gc : GradeCount) : Prop :=
  4 * gc.ninth = 3 * gc.eleventh ∧ 6 * gc.tenth = 5 * gc.eleventh

/-- The total number of students --/
def total_students (gc : GradeCount) : ℕ :=
  gc.ninth + gc.tenth + gc.eleventh

/-- There exists a valid configuration with 31 students --/
theorem exists_valid_31 : ∃ gc : GradeCount, valid_ratios gc ∧ total_students gc = 31 := by
  sorry

/-- Any valid configuration has at least 31 students --/
theorem min_students (gc : GradeCount) (h : valid_ratios gc) : total_students gc ≥ 31 := by
  sorry

/-- The smallest possible number of students is 31 --/
theorem smallest_total : (∃ gc : GradeCount, valid_ratios gc ∧ total_students gc = 31) ∧
  (∀ gc : GradeCount, valid_ratios gc → total_students gc ≥ 31) := by
  sorry

end NUMINAMATH_CALUDE_exists_valid_31_min_students_smallest_total_l3802_380268


namespace NUMINAMATH_CALUDE_officer_selection_count_l3802_380281

def total_members : ℕ := 25
def num_officers : ℕ := 3

-- Define a structure to represent a pair of members
structure MemberPair :=
  (member1 : ℕ)
  (member2 : ℕ)

-- Define the two special pairs
def pair1 : MemberPair := ⟨1, 2⟩  -- Rachel and Simon
def pair2 : MemberPair := ⟨3, 4⟩  -- Penelope and Quentin

-- Function to calculate the number of ways to choose officers
def count_officer_choices (total : ℕ) (officers : ℕ) (pair1 pair2 : MemberPair) : ℕ := 
  sorry

-- Theorem statement
theorem officer_selection_count :
  count_officer_choices total_members num_officers pair1 pair2 = 8072 :=
sorry

end NUMINAMATH_CALUDE_officer_selection_count_l3802_380281


namespace NUMINAMATH_CALUDE_age_difference_l3802_380294

theorem age_difference (A B : ℕ) : B = 39 → A + 10 = 2 * (B - 10) → A - B = 9 := by
  sorry

end NUMINAMATH_CALUDE_age_difference_l3802_380294


namespace NUMINAMATH_CALUDE_cube_volume_l3802_380231

theorem cube_volume (edge_sum : ℝ) (h : edge_sum = 96) : 
  let edge_length := edge_sum / 12
  let volume := edge_length ^ 3
  volume = 512 := by sorry

end NUMINAMATH_CALUDE_cube_volume_l3802_380231


namespace NUMINAMATH_CALUDE_function_inequality_l3802_380284

-- Define the function f
variable (f : ℝ → ℝ)

-- Define the conditions
variable (h1 : ∀ x, (x - 1) * (deriv f x) ≤ 0)
variable (h2 : ∀ x, f (x + 1) = f (-x + 1))

-- Define the theorem
theorem function_inequality (x₁ x₂ : ℝ) (h3 : |x₁ - 1| < |x₂ - 1|) : 
  f (2 - x₁) ≥ f (2 - x₂) := by sorry

end NUMINAMATH_CALUDE_function_inequality_l3802_380284


namespace NUMINAMATH_CALUDE_express_train_meetings_l3802_380214

/-- Represents time in hours and minutes -/
structure Time where
  hours : Nat
  minutes : Nat

/-- Calculates the difference between two times in minutes -/
def timeDifference (t1 t2 : Time) : Nat :=
  (t2.hours - t1.hours) * 60 + (t2.minutes - t1.minutes)

/-- The problem statement -/
theorem express_train_meetings :
  let travelTime : Nat := 210 -- 3 hours and 30 minutes in minutes
  let departureInterval : Nat := 60 -- 1 hour in minutes
  let firstDeparture : Time := ⟨6, 0⟩ -- 6:00 AM
  let expressDeparture : Time := ⟨9, 0⟩ -- 9:00 AM
  let expressArrival : Time := ⟨12, 30⟩ -- 12:30 PM (9:00 AM + 3h30m)
  
  (timeDifference firstDeparture expressDeparture / departureInterval + 1) -
  (timeDifference firstDeparture expressArrival / departureInterval + 1) = 6 := by
  sorry

end NUMINAMATH_CALUDE_express_train_meetings_l3802_380214


namespace NUMINAMATH_CALUDE_mildred_initial_oranges_l3802_380256

/-- The number of oranges Mildred's father gave her -/
def oranges_from_father : ℕ := 2

/-- The total number of oranges Mildred has after receiving oranges from her father -/
def total_oranges : ℕ := 79

/-- The number of oranges Mildred initially collected -/
def initial_oranges : ℕ := total_oranges - oranges_from_father

theorem mildred_initial_oranges :
  initial_oranges = 77 :=
by sorry

end NUMINAMATH_CALUDE_mildred_initial_oranges_l3802_380256


namespace NUMINAMATH_CALUDE_least_subtraction_for_divisibility_l3802_380240

theorem least_subtraction_for_divisibility : 
  ∃ (x : ℕ), x = 6 ∧ 
  (∀ (y : ℕ), y < x → ¬(29 ∣ (87654321 - y))) ∧ 
  (29 ∣ (87654321 - x)) :=
sorry

end NUMINAMATH_CALUDE_least_subtraction_for_divisibility_l3802_380240


namespace NUMINAMATH_CALUDE_primes_up_to_100_l3802_380258

theorem primes_up_to_100 : 
  {p : ℕ | Nat.Prime p ∧ 2 ≤ p ∧ p ≤ 100} = 
  {2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97} := by
  sorry

end NUMINAMATH_CALUDE_primes_up_to_100_l3802_380258


namespace NUMINAMATH_CALUDE_joe_money_left_l3802_380290

/-- The amount of money Joe has left after shopping and donating to charity -/
def money_left (initial_amount notebooks books pens stickers notebook_price book_price pen_price sticker_price charity : ℕ) : ℕ :=
  initial_amount - (notebooks * notebook_price + books * book_price + pens * pen_price + stickers * sticker_price + charity)

/-- Theorem stating that Joe has $60 left after his shopping trip and charity donation -/
theorem joe_money_left :
  money_left 150 7 2 5 3 4 12 2 6 10 = 60 := by
  sorry

#eval money_left 150 7 2 5 3 4 12 2 6 10

end NUMINAMATH_CALUDE_joe_money_left_l3802_380290


namespace NUMINAMATH_CALUDE_tan_double_angle_l3802_380228

/-- Given an angle θ with vertex at the origin, initial side on the positive x-axis,
    and terminal side passing through (-1, 2), prove that tan 2θ = 4/3 -/
theorem tan_double_angle (θ : ℝ) : 
  (∃ (x y : ℝ), x = -1 ∧ y = 2 ∧ Real.tan θ = y / x) → 
  Real.tan (2 * θ) = 4 / 3 := by
  sorry

end NUMINAMATH_CALUDE_tan_double_angle_l3802_380228


namespace NUMINAMATH_CALUDE_peter_marbles_l3802_380246

theorem peter_marbles (initial_marbles lost_marbles : ℕ) 
  (h1 : initial_marbles = 33)
  (h2 : lost_marbles = 15) :
  initial_marbles - lost_marbles = 18 := by
  sorry

end NUMINAMATH_CALUDE_peter_marbles_l3802_380246


namespace NUMINAMATH_CALUDE_complex_sum_magnitude_l3802_380264

theorem complex_sum_magnitude (a b c : ℂ) :
  Complex.abs a = 2 →
  Complex.abs b = 2 →
  Complex.abs c = 2 →
  a^3 / (b * c) + b^3 / (a * c) + c^3 / (a * b) = 0 →
  Complex.abs (a + b + c) = 6 + 2 * Real.sqrt 6 ∨
  Complex.abs (a + b + c) = 6 - 2 * Real.sqrt 6 := by
  sorry

end NUMINAMATH_CALUDE_complex_sum_magnitude_l3802_380264


namespace NUMINAMATH_CALUDE_optimal_reading_distribution_l3802_380257

theorem optimal_reading_distribution 
  (total_time : ℕ) 
  (disc_capacity : ℕ) 
  (max_unused_space : ℕ) 
  (h1 : total_time = 630) 
  (h2 : disc_capacity = 80) 
  (h3 : max_unused_space = 4) :
  ∃ (num_discs : ℕ), 
    num_discs > 0 ∧ 
    num_discs * (disc_capacity - max_unused_space) ≥ total_time ∧
    (num_discs - 1) * disc_capacity < total_time ∧
    total_time / num_discs = 70 :=
sorry

end NUMINAMATH_CALUDE_optimal_reading_distribution_l3802_380257


namespace NUMINAMATH_CALUDE_line_segment_proportion_l3802_380203

theorem line_segment_proportion (a b c d : ℝ) :
  a = 1 →
  b = 2 →
  c = 3 →
  (a / b = c / d) →
  d = 6 := by
sorry

end NUMINAMATH_CALUDE_line_segment_proportion_l3802_380203


namespace NUMINAMATH_CALUDE_sum_row_10_pascal_l3802_380207

/-- Sum of numbers in a row of Pascal's Triangle -/
def pascal_row_sum (n : ℕ) : ℕ := 2^n

/-- Row 10 of Pascal's Triangle -/
def row_10 : ℕ := 10

theorem sum_row_10_pascal : pascal_row_sum row_10 = 1024 := by
  sorry

end NUMINAMATH_CALUDE_sum_row_10_pascal_l3802_380207


namespace NUMINAMATH_CALUDE_simplify_and_rationalize_l3802_380287

theorem simplify_and_rationalize :
  (Real.sqrt 5 / Real.sqrt 2) * (Real.sqrt 9 / Real.sqrt 13) * (Real.sqrt 22 / Real.sqrt 7) = 
  (3 * Real.sqrt 20020) / 182 := by sorry

end NUMINAMATH_CALUDE_simplify_and_rationalize_l3802_380287


namespace NUMINAMATH_CALUDE_black_stones_count_l3802_380293

theorem black_stones_count (total : ℕ) (difference : ℕ) (black : ℕ) : 
  total = 950 → 
  difference = 150 → 
  total = black + (black + difference) → 
  black = 400 := by
sorry

end NUMINAMATH_CALUDE_black_stones_count_l3802_380293


namespace NUMINAMATH_CALUDE_triangle_angle_A_l3802_380229

theorem triangle_angle_A (a b : ℝ) (B : Real) (A : Real) : 
  a = 4 → 
  b = 4 * Real.sqrt 3 → 
  B = 60 * π / 180 →
  (a / Real.sin A = b / Real.sin B) →
  A = 30 * π / 180 := by
sorry

end NUMINAMATH_CALUDE_triangle_angle_A_l3802_380229


namespace NUMINAMATH_CALUDE_linear_system_solution_existence_l3802_380211

theorem linear_system_solution_existence
  (a b c d : ℤ)
  (h_nonzero : a * d - b * c ≠ 0)
  (b₁ b₂ : ℤ)
  (h_b₁ : ∃ k : ℤ, b₁ = (a * d - b * c) * k)
  (h_b₂ : ∃ q : ℤ, b₂ = (a * d - b * c) * q) :
  ∃ x y : ℤ, a * x + b * y = b₁ ∧ c * x + d * y = b₂ :=
sorry

end NUMINAMATH_CALUDE_linear_system_solution_existence_l3802_380211


namespace NUMINAMATH_CALUDE_nine_multiple_plus_k_equals_ones_l3802_380270

/-- Given a natural number N and a positive integer k, there exists a number M
    consisting of k ones such that N · 9 + k = M. -/
theorem nine_multiple_plus_k_equals_ones (N : ℕ) (k : ℕ+) :
  ∃ M : ℕ, (∀ d : ℕ, d < k → (M / 10^d) % 10 = 1) ∧ N * 9 + k = M :=
sorry

end NUMINAMATH_CALUDE_nine_multiple_plus_k_equals_ones_l3802_380270


namespace NUMINAMATH_CALUDE_no_integer_solutions_l3802_380285

theorem no_integer_solutions :
  ¬ ∃ (x y z : ℤ), x^4 + y^4 + z^4 = 2*x^2*y^2 + 2*y^2*z^2 + 2*z^2*x^2 + 24 := by
  sorry

end NUMINAMATH_CALUDE_no_integer_solutions_l3802_380285


namespace NUMINAMATH_CALUDE_f_odd_and_periodic_l3802_380247

def is_odd (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

def is_periodic (f : ℝ → ℝ) (p : ℝ) : Prop := ∀ x, f (x + p) = f x

theorem f_odd_and_periodic (f : ℝ → ℝ) 
  (h1 : ∀ x, f (10 + x) = f (10 - x))
  (h2 : ∀ x, f (20 - x) = -f (20 + x)) :
  is_odd f ∧ is_periodic f 40 := by
  sorry

end NUMINAMATH_CALUDE_f_odd_and_periodic_l3802_380247


namespace NUMINAMATH_CALUDE_complement_A_intersect_B_a_greater_than_one_l3802_380283

-- Define the sets A, B, and C
def A : Set ℝ := {x | 1 ≤ x ∧ x ≤ 7}
def B : Set ℝ := {x | 2 < x ∧ x < 10}
def C (a : ℝ) : Set ℝ := {x | x < a}

-- Theorem for the first part of the problem
theorem complement_A_intersect_B :
  (Set.univ \ A) ∩ B = {x | 7 < x ∧ x < 10} := by sorry

-- Theorem for the second part of the problem
theorem a_greater_than_one (h : A ∩ C a ≠ ∅) : a > 1 := by sorry

end NUMINAMATH_CALUDE_complement_A_intersect_B_a_greater_than_one_l3802_380283


namespace NUMINAMATH_CALUDE_max_value_of_f_l3802_380248

/-- Given that f(x) = 2sin(x) - cos(x) reaches its maximum value when x = θ, prove that sin(θ) = 2√5/5 -/
theorem max_value_of_f (θ : ℝ) : 
  (∀ x, 2 * Real.sin x - Real.cos x ≤ 2 * Real.sin θ - Real.cos θ) →
  Real.sin θ = 2 * Real.sqrt 5 / 5 :=
by sorry

end NUMINAMATH_CALUDE_max_value_of_f_l3802_380248


namespace NUMINAMATH_CALUDE_final_price_calculation_final_price_is_841_32_l3802_380278

/-- Calculates the final price of a TV and sound system after discounts and tax --/
theorem final_price_calculation (tv_price sound_price : ℝ) 
  (tv_discount1 tv_discount2 sound_discount tax_rate : ℝ) : ℝ :=
  let tv_after_discounts := tv_price * (1 - tv_discount1) * (1 - tv_discount2)
  let sound_after_discount := sound_price * (1 - sound_discount)
  let total_before_tax := tv_after_discounts + sound_after_discount
  let tax_amount := total_before_tax * tax_rate
  let final_price := total_before_tax + tax_amount
  final_price

/-- Theorem stating that the final price is $841.32 given the specific conditions --/
theorem final_price_is_841_32 : 
  final_price_calculation 600 400 0.1 0.15 0.2 0.08 = 841.32 := by
  sorry

end NUMINAMATH_CALUDE_final_price_calculation_final_price_is_841_32_l3802_380278


namespace NUMINAMATH_CALUDE_train_vs_airplane_capacity_difference_l3802_380201

/-- The passenger capacity of a single train car -/
def train_car_capacity : ℕ := 60

/-- The passenger capacity of a 747 airplane -/
def airplane_capacity : ℕ := 366

/-- The number of cars in the train -/
def train_cars : ℕ := 16

/-- The number of airplanes -/
def num_airplanes : ℕ := 2

/-- The theorem stating the difference in passenger capacity -/
theorem train_vs_airplane_capacity_difference :
  train_cars * train_car_capacity - num_airplanes * airplane_capacity = 228 := by
  sorry

end NUMINAMATH_CALUDE_train_vs_airplane_capacity_difference_l3802_380201


namespace NUMINAMATH_CALUDE_carrots_rows_planted_l3802_380243

/-- Calculates the number of rows of carrots planted given the planting conditions -/
theorem carrots_rows_planted (plants_per_row : ℕ) (planting_time : ℕ) (planting_rate : ℕ) : 
  plants_per_row > 0 →
  planting_time * planting_rate / plants_per_row = 400 :=
by
  intro h
  sorry

#check carrots_rows_planted 300 20 6000

end NUMINAMATH_CALUDE_carrots_rows_planted_l3802_380243


namespace NUMINAMATH_CALUDE_equation_solution_l3802_380299

theorem equation_solution :
  ∃ x : ℚ, (5 + 3.5 * x = 2.1 * x - 25) ∧ (x = -150 / 7) := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l3802_380299


namespace NUMINAMATH_CALUDE_point_on_line_l3802_380224

/-- Given a line with slope 2 and y-intercept 2, the y-coordinate of a point on this line
    with x-coordinate 498 is 998. -/
theorem point_on_line (line : ℝ → ℝ) (x y : ℝ) : 
  (∀ t, line t = 2 * t + 2) →  -- Condition 1 and 2: slope is 2, y-intercept is 2
  x = 498 →                    -- Condition 4: x-coordinate is 498
  y = line x →                 -- Condition 3: the point (x, y) is on the line
  y = 998 := by                -- Question: prove y = 998
sorry


end NUMINAMATH_CALUDE_point_on_line_l3802_380224


namespace NUMINAMATH_CALUDE_positive_root_implies_m_value_l3802_380233

theorem positive_root_implies_m_value 
  (h : ∃ (x : ℝ), x > 0 ∧ (6 - x) / (x - 3) - (2 * m) / (x - 3) = 0) : 
  m = 3/2 := by
  sorry

end NUMINAMATH_CALUDE_positive_root_implies_m_value_l3802_380233


namespace NUMINAMATH_CALUDE_profit_calculation_l3802_380282

-- Define the buying and selling rates
def buy_rate : ℚ := 5 / 6
def sell_rate : ℚ := 4 / 8

-- Define the target profit
def target_profit : ℚ := 120

-- Define the number of disks to be sold
def disks_to_sell : ℕ := 150

-- Theorem statement
theorem profit_calculation :
  (disks_to_sell : ℚ) * (1 / sell_rate - 1 / buy_rate) = target_profit := by
  sorry

end NUMINAMATH_CALUDE_profit_calculation_l3802_380282


namespace NUMINAMATH_CALUDE_max_rectangles_6x6_grid_l3802_380202

/-- Counts the number of rectangles in a right triangle grid of size n x n -/
def count_rectangles (n : ℕ) : ℕ :=
  (n * (n + 1) * (2 * n + 1)) / 6

/-- The maximum number of rectangles in a 6x6 right triangle grid is 126 -/
theorem max_rectangles_6x6_grid :
  count_rectangles 6 = 126 := by sorry

end NUMINAMATH_CALUDE_max_rectangles_6x6_grid_l3802_380202


namespace NUMINAMATH_CALUDE_john_taller_than_lena_l3802_380291

/-- Proves that John is 15 cm taller than Lena given the problem conditions -/
theorem john_taller_than_lena (john_height rebeca_height lena_height : ℕ) :
  john_height = 152 →
  john_height = rebeca_height - 6 →
  lena_height + rebeca_height = 295 →
  john_height - lena_height = 15 :=
by
  sorry

end NUMINAMATH_CALUDE_john_taller_than_lena_l3802_380291


namespace NUMINAMATH_CALUDE_art_gallery_theorem_l3802_380226

theorem art_gallery_theorem (total_pieces : ℕ) : 
  (total_pieces : ℚ) * (1 / 3) * (1 / 6) + 
  (total_pieces : ℚ) * (2 / 3) * (2 / 3) = 800 →
  total_pieces = 1800 := by
  sorry

end NUMINAMATH_CALUDE_art_gallery_theorem_l3802_380226


namespace NUMINAMATH_CALUDE_parabola_and_line_properties_l3802_380288

/-- Parabola with vertex at origin and directrix x = -1 -/
structure Parabola where
  equation : ℝ → ℝ → Prop
  vertex_at_origin : equation 0 0
  directrix : ℝ → Prop
  directrix_eq : ∀ x, directrix x ↔ x = -1

/-- Line passing through two points on the parabola -/
structure IntersectingLine (p : Parabola) where
  equation : ℝ → ℝ → Prop
  passes_through_focus : equation 1 0
  intersects_parabola : ∃ (x₁ y₁ x₂ y₂ : ℝ), 
    equation x₁ y₁ ∧ p.equation x₁ y₁ ∧
    equation x₂ y₂ ∧ p.equation x₂ y₂ ∧
    x₁ ≠ x₂
  midpoint_x_coord : ℝ
  midpoint_condition : ∀ (x₁ y₁ x₂ y₂ : ℝ),
    equation x₁ y₁ ∧ p.equation x₁ y₁ ∧
    equation x₂ y₂ ∧ p.equation x₂ y₂ ∧
    x₁ ≠ x₂ →
    (x₁ + x₂) / 2 = midpoint_x_coord

/-- Main theorem about the parabola and intersecting line -/
theorem parabola_and_line_properties (p : Parabola) (l : IntersectingLine p) 
    (h_midpoint : l.midpoint_x_coord = 2) :
  (∀ x y, p.equation x y ↔ y^2 = 4*x) ∧
  (∀ x y, l.equation x y ↔ (y = Real.sqrt 2 * x - Real.sqrt 2 ∨ 
                            y = -Real.sqrt 2 * x + Real.sqrt 2)) := by
  sorry

end NUMINAMATH_CALUDE_parabola_and_line_properties_l3802_380288


namespace NUMINAMATH_CALUDE_clara_has_68_stickers_l3802_380215

/-- Calculates the number of stickers Clara has left after a series of transactions -/
def claras_stickers : ℕ :=
  let initial := 100
  let after_boy := initial - 10
  let after_teacher := after_boy + 50
  let after_classmates := after_teacher - 20
  let after_exchange := after_classmates - 15 + 30
  let to_friends := after_exchange / 2
  after_exchange - to_friends

/-- Proves that Clara ends up with 68 stickers -/
theorem clara_has_68_stickers : claras_stickers = 68 := by
  sorry

#eval claras_stickers

end NUMINAMATH_CALUDE_clara_has_68_stickers_l3802_380215


namespace NUMINAMATH_CALUDE_triangle_side_sum_l3802_380216

/-- Given real numbers x, y, and z, if 1/|x^2+2yz|, 1/|y^2+2zx|, and 1/|z^2+2xy| 
    form the sides of a non-degenerate triangle, then xy + yz + zx = 0 -/
theorem triangle_side_sum (x y z : ℝ) 
  (h1 : 1 / |x^2 + 2*y*z| + 1 / |y^2 + 2*z*x| > 1 / |z^2 + 2*x*y|)
  (h2 : 1 / |y^2 + 2*z*x| + 1 / |z^2 + 2*x*y| > 1 / |x^2 + 2*y*z|)
  (h3 : 1 / |z^2 + 2*x*y| + 1 / |x^2 + 2*y*z| > 1 / |y^2 + 2*z*x|)
  (h4 : |x^2 + 2*y*z| ≠ 0)
  (h5 : |y^2 + 2*z*x| ≠ 0)
  (h6 : |z^2 + 2*x*y| ≠ 0) :
  x*y + y*z + z*x = 0 := by
sorry

end NUMINAMATH_CALUDE_triangle_side_sum_l3802_380216


namespace NUMINAMATH_CALUDE_derivative_of_f_l3802_380223

noncomputable def f (x : ℝ) : ℝ := Real.cos (2 * x) * Real.log x

theorem derivative_of_f (x : ℝ) (h : x > 0) :
  deriv f x = -2 * Real.sin (2 * x) * Real.log x + Real.cos (2 * x) / x :=
by sorry

end NUMINAMATH_CALUDE_derivative_of_f_l3802_380223


namespace NUMINAMATH_CALUDE_problem_solution_l3802_380298

theorem problem_solution : 
  (1/2 - 1/4 + 1/12) * (-12) = -4 ∧ 
  -(3^2) + (-5)^2 * (4/5) - |(-6)| = 5 := by
sorry

end NUMINAMATH_CALUDE_problem_solution_l3802_380298


namespace NUMINAMATH_CALUDE_trigonometric_identity_l3802_380250

theorem trigonometric_identity : 
  let cos_45 : ℝ := Real.sqrt 2 / 2
  let tan_30 : ℝ := Real.sqrt 3 / 3
  let sin_60 : ℝ := Real.sqrt 3 / 2
  cos_45^2 + tan_30 * sin_60 = 1 := by
sorry

end NUMINAMATH_CALUDE_trigonometric_identity_l3802_380250


namespace NUMINAMATH_CALUDE_fraction_invariance_l3802_380280

theorem fraction_invariance (x y : ℝ) : 
  (2 * x) / (3 * x - y) = (2 * (3 * x)) / (3 * (3 * x) - (3 * y)) :=
by sorry

end NUMINAMATH_CALUDE_fraction_invariance_l3802_380280


namespace NUMINAMATH_CALUDE_stone_game_loser_l3802_380255

/-- Represents a pile of stones -/
structure Pile :=
  (count : Nat)

/-- Represents the game state -/
structure GameState :=
  (piles : List Pile)
  (currentPlayer : Nat)

/-- Defines a valid move in the game -/
def validMove (state : GameState) : Prop :=
  ∃ (p : Pile) (n : Nat), p ∈ state.piles ∧ 1 ≤ n ∧ n < p.count

/-- The initial game state -/
def initialState : GameState :=
  { piles := [⟨6⟩, ⟨8⟩, ⟨8⟩, ⟨9⟩], currentPlayer := 1 }

/-- The number of players -/
def numPlayers : Nat := 5

/-- The losing player -/
def losingPlayer : Nat := 3

theorem stone_game_loser :
  ¬∃ (moves : Nat), 
    (moves + initialState.piles.length = (initialState.piles.map Pile.count).sum) ∧
    (moves % numPlayers + 1 = losingPlayer) ∧
    (∀ (state : GameState), state.piles.length ≤ moves + initialState.piles.length → validMove state) :=
sorry

end NUMINAMATH_CALUDE_stone_game_loser_l3802_380255


namespace NUMINAMATH_CALUDE_original_class_strength_l3802_380219

/-- Proves that the original strength of an adult class is 17 students given the conditions. -/
theorem original_class_strength (original_average : ℝ) (new_students : ℕ) (new_average : ℝ) (average_decrease : ℝ) :
  original_average = 40 →
  new_students = 17 →
  new_average = 32 →
  average_decrease = 4 →
  ∃ (x : ℕ), x = 17 ∧ 
    (x : ℝ) * original_average + (new_students : ℝ) * new_average = 
      ((x : ℝ) + (new_students : ℝ)) * (original_average - average_decrease) := by
  sorry

#check original_class_strength

end NUMINAMATH_CALUDE_original_class_strength_l3802_380219


namespace NUMINAMATH_CALUDE_inequality_not_always_true_l3802_380236

theorem inequality_not_always_true (a b : ℝ) (h : a > b) :
  ∃ c : ℝ, ¬(a * c^2 > b * c^2) :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_CALUDE_inequality_not_always_true_l3802_380236


namespace NUMINAMATH_CALUDE_range_of_a_l3802_380213

theorem range_of_a (a : ℝ) : 
  (∀ x : ℝ, x^2 - 8*x - 20 > 0 → x^2 - 2*x + 1 - a^2 > 0) ∧ 
  (∃ x : ℝ, x^2 - 2*x + 1 - a^2 > 0 ∧ x^2 - 8*x - 20 ≤ 0) ∧ 
  a > 0 
  ↔ 0 < a ∧ a ≤ 3 := by sorry

end NUMINAMATH_CALUDE_range_of_a_l3802_380213


namespace NUMINAMATH_CALUDE_unique_solution_for_exponential_equation_l3802_380262

theorem unique_solution_for_exponential_equation :
  ∀ n p : ℕ+,
    Nat.Prime p →
    3^(p : ℕ) - n * p = n + p →
    n = 6 ∧ p = 3 := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_for_exponential_equation_l3802_380262


namespace NUMINAMATH_CALUDE_remainder_3_pow_2023_mod_17_l3802_380222

theorem remainder_3_pow_2023_mod_17 : 3^2023 % 17 = 7 := by
  sorry

end NUMINAMATH_CALUDE_remainder_3_pow_2023_mod_17_l3802_380222


namespace NUMINAMATH_CALUDE_min_difference_of_composite_functions_l3802_380253

open Real

theorem min_difference_of_composite_functions :
  let f : ℝ → ℝ := λ x ↦ Real.exp (3 * x - 1)
  let g : ℝ → ℝ := λ x ↦ 1 / 3 + Real.log x
  ∃ (min_diff : ℝ), min_diff = (2 + Real.log 3) / 3 ∧
    ∀ m n : ℝ, f m = g n → n - m ≥ min_diff :=
by sorry

end NUMINAMATH_CALUDE_min_difference_of_composite_functions_l3802_380253


namespace NUMINAMATH_CALUDE_simplify_polynomial_l3802_380239

theorem simplify_polynomial (w : ℝ) : 
  3 * w + 5 - 6 * w^2 + 4 * w - 7 + 9 * w^2 = 3 * w^2 + 7 * w - 2 := by
  sorry

end NUMINAMATH_CALUDE_simplify_polynomial_l3802_380239


namespace NUMINAMATH_CALUDE_circle_properties_l3802_380237

-- Define a circle
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

-- Define a diameter
def diameter (c : Circle) (p q : ℝ × ℝ) : Prop :=
  (p.1 - c.center.1)^2 + (p.2 - c.center.2)^2 = c.radius^2 ∧
  (q.1 - c.center.1)^2 + (q.2 - c.center.2)^2 = c.radius^2 ∧
  (p.1 - q.1)^2 + (p.2 - q.2)^2 = (2 * c.radius)^2

-- Define a point on the circle
def onCircle (c : Circle) (p : ℝ × ℝ) : Prop :=
  (p.1 - c.center.1)^2 + (p.2 - c.center.2)^2 = c.radius^2

-- Theorem statement
theorem circle_properties (c : Circle) :
  (∀ p q : ℝ × ℝ, diameter c p q → ∀ r s : ℝ × ℝ, diameter c r s → 
    (p.1 - q.1)^2 + (p.2 - q.2)^2 = (r.1 - s.1)^2 + (r.2 - s.2)^2) ∧
  (∀ p q : ℝ × ℝ, onCircle c p → onCircle c q → 
    (p.1 - c.center.1)^2 + (p.2 - c.center.2)^2 = (q.1 - c.center.1)^2 + (q.2 - c.center.2)^2) :=
sorry

end NUMINAMATH_CALUDE_circle_properties_l3802_380237


namespace NUMINAMATH_CALUDE_log_sum_adjacent_terms_l3802_380204

-- Define a geometric sequence
def is_geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, r ≠ 0 ∧ ∀ n : ℕ, a (n + 1) = a n * r

-- State the theorem
theorem log_sum_adjacent_terms 
  (a : ℕ → ℝ) 
  (h_geo : is_geometric_sequence a) 
  (h_pos : ∀ n, a n > 0) 
  (h_a5 : a 5 = 10) : 
  Real.log (a 4) + Real.log (a 6) = 2 := by
sorry

end NUMINAMATH_CALUDE_log_sum_adjacent_terms_l3802_380204


namespace NUMINAMATH_CALUDE_exists_multiple_of_three_l3802_380273

def CircleNumbers (n : ℕ) := Fin n → ℕ

def ValidCircle (nums : CircleNumbers 99) : Prop :=
  ∀ i : Fin 99, 
    (nums i - nums (i + 1) = 1) ∨ 
    (nums i - nums (i + 1) = 2) ∨ 
    (nums i / nums (i + 1) = 2)

theorem exists_multiple_of_three (nums : CircleNumbers 99) 
  (h : ValidCircle nums) : 
  ∃ i : Fin 99, 3 ∣ nums i :=
sorry

end NUMINAMATH_CALUDE_exists_multiple_of_three_l3802_380273


namespace NUMINAMATH_CALUDE_age_equality_l3802_380277

/-- Proves that the number of years after which grandfather's age equals the sum of Xiaoming and father's ages is 14, given their current ages. -/
theorem age_equality (grandfather_age father_age xiaoming_age : ℕ) 
  (h1 : grandfather_age = 60)
  (h2 : father_age = 35)
  (h3 : xiaoming_age = 11) : 
  ∃ (years : ℕ), grandfather_age + years = (father_age + years) + (xiaoming_age + years) ∧ years = 14 := by
  sorry

end NUMINAMATH_CALUDE_age_equality_l3802_380277


namespace NUMINAMATH_CALUDE_fifty_cows_fifty_bags_l3802_380206

/-- The number of bags of husk eaten by a group of cows over a fixed period -/
def bagsEaten (numCows : ℕ) (daysPerBag : ℕ) (totalDays : ℕ) : ℕ :=
  numCows * (totalDays / daysPerBag)

/-- Theorem: 50 cows eat 50 bags of husk in 50 days -/
theorem fifty_cows_fifty_bags :
  bagsEaten 50 50 50 = 50 := by
  sorry

end NUMINAMATH_CALUDE_fifty_cows_fifty_bags_l3802_380206


namespace NUMINAMATH_CALUDE_quadratic_rational_solutions_l3802_380266

/-- A function that checks if a quadratic equation with given coefficients has rational solutions -/
def has_rational_solutions (a b c : ℤ) : Prop :=
  ∃ x : ℚ, a * x^2 + b * x + c = 0

/-- The set of positive integers k for which 3x^2 + 17x + k = 0 has rational solutions -/
def valid_k_set : Set ℤ :=
  {k : ℤ | k > 0 ∧ has_rational_solutions 3 17 k}

theorem quadratic_rational_solutions :
  ∃ k₁ k₂ : ℕ,
    k₁ ≠ k₂ ∧
    (↑k₁ : ℤ) ∈ valid_k_set ∧
    (↑k₂ : ℤ) ∈ valid_k_set ∧
    valid_k_set = {↑k₁, ↑k₂} ∧
    k₁ * k₂ = 240 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_rational_solutions_l3802_380266


namespace NUMINAMATH_CALUDE_smallest_lcm_with_gcd_five_l3802_380296

theorem smallest_lcm_with_gcd_five (k l : ℕ) : 
  1000 ≤ k ∧ k < 10000 →
  1000 ≤ l ∧ l < 10000 →
  Nat.gcd k l = 5 →
  ∀ m n : ℕ, 1000 ≤ m ∧ m < 10000 → 1000 ≤ n ∧ n < 10000 → Nat.gcd m n = 5 →
  Nat.lcm k l ≤ Nat.lcm m n →
  Nat.lcm k l = 203010 :=
by sorry

end NUMINAMATH_CALUDE_smallest_lcm_with_gcd_five_l3802_380296


namespace NUMINAMATH_CALUDE_cube_root_x_plus_3y_equals_3_l3802_380274

theorem cube_root_x_plus_3y_equals_3 (x y : ℝ) 
  (h : y = Real.sqrt (3 - x) + Real.sqrt (x - 3) + 8) :
  (x + 3 * y) ^ (1/3 : ℝ) = 3 := by
  sorry

end NUMINAMATH_CALUDE_cube_root_x_plus_3y_equals_3_l3802_380274


namespace NUMINAMATH_CALUDE_events_related_95_percent_confidence_l3802_380200

-- Define the confidence level
def confidence_level : ℝ := 0.95

-- Define the critical value for 95% confidence
def critical_value : ℝ := 3.841

-- Define the relation between events A and B
def events_related (K : ℝ) : Prop := K^2 > critical_value

-- Theorem statement
theorem events_related_95_percent_confidence (K : ℝ) :
  events_related K ↔ K^2 > critical_value :=
sorry

end NUMINAMATH_CALUDE_events_related_95_percent_confidence_l3802_380200


namespace NUMINAMATH_CALUDE_movie_theater_adult_price_l3802_380242

/-- Proves that the adult ticket price is $6.75 given the conditions of the movie theater problem -/
theorem movie_theater_adult_price :
  let children_price : ℚ := 9/2
  let num_children : ℕ := 48
  let child_adult_diff : ℕ := 20
  let total_receipts : ℚ := 405
  let num_adults : ℕ := num_children - child_adult_diff
  let adult_price : ℚ := (total_receipts - children_price * num_children) / num_adults
  adult_price = 27/4 := by
sorry

end NUMINAMATH_CALUDE_movie_theater_adult_price_l3802_380242


namespace NUMINAMATH_CALUDE_max_area_rectangular_enclosure_l3802_380205

/-- Given a rectangular area with perimeter P (excluding one side) and length L twice the width W,
    the maximum area A is (P/4)^2 square units. -/
theorem max_area_rectangular_enclosure (P : ℝ) (h : P > 0) :
  let W := P / 4
  let L := 2 * W
  let A := L * W
  A = (P / 4) ^ 2 := by
  sorry

#check max_area_rectangular_enclosure

end NUMINAMATH_CALUDE_max_area_rectangular_enclosure_l3802_380205


namespace NUMINAMATH_CALUDE_simplify_trigonometric_expression_l3802_380209

theorem simplify_trigonometric_expression (α : Real) 
  (h1 : π < α ∧ α < 3*π/2) : 
  Real.sqrt ((1 + Real.sin α) / (1 - Real.sin α)) - 
  Real.sqrt ((1 - Real.sin α) / (1 + Real.sin α)) = 
  -2 * Real.tan α := by
  sorry

end NUMINAMATH_CALUDE_simplify_trigonometric_expression_l3802_380209


namespace NUMINAMATH_CALUDE_basketball_playoff_condition_l3802_380221

/-- A basketball team's playoff qualification condition -/
theorem basketball_playoff_condition (x : ℕ) : 
  (∀ (game : ℕ), game ≤ 32 → (game = 32 - x ∨ game = x)) →  -- Each game is either won or lost
  (2 * x + (32 - x) ≥ 48) →                                  -- Points condition
  (x ≤ 32) →                                                 -- Cannot win more games than played
  (2 * x + (32 - x) ≥ 48) :=                                 -- Conclusion: same as second hypothesis
by sorry

end NUMINAMATH_CALUDE_basketball_playoff_condition_l3802_380221


namespace NUMINAMATH_CALUDE_arithmetic_sequence_properties_l3802_380225

def arithmetic_sequence (a : ℕ → ℝ) (d : ℝ) :=
  ∀ n, a (n + 1) = a n + d

theorem arithmetic_sequence_properties
  (a : ℕ → ℝ) (d : ℝ)
  (h_arithmetic : arithmetic_sequence a d)
  (h_a2 : a 2 = 5)
  (h_sum : a 6 + a 8 = 30) :
  d = 2 ∧
  (∀ n, a n = 2 * n + 1) ∧
  (∀ n, 1 / ((a n)^2 - 1) = (1/4) * (1/n - 1/(n+1))) :=
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_properties_l3802_380225
