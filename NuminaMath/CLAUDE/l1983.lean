import Mathlib

namespace NUMINAMATH_CALUDE_total_seashells_l1983_198378

theorem total_seashells (joan_shells jessica_shells : ℕ) : 
  joan_shells = 6 → jessica_shells = 8 → joan_shells + jessica_shells = 14 := by
  sorry

end NUMINAMATH_CALUDE_total_seashells_l1983_198378


namespace NUMINAMATH_CALUDE_work_increase_percentage_l1983_198315

theorem work_increase_percentage (p : ℕ) (W : ℝ) (h : p > 0) : 
  let absent_ratio : ℝ := 1 / 6
  let present_ratio : ℝ := 1 - absent_ratio
  let original_work_per_person : ℝ := W / p
  let new_work_per_person : ℝ := W / (p * present_ratio)
  let work_increase : ℝ := new_work_per_person - original_work_per_person
  let percentage_increase : ℝ := (work_increase / original_work_per_person) * 100
  percentage_increase = 20 := by sorry

end NUMINAMATH_CALUDE_work_increase_percentage_l1983_198315


namespace NUMINAMATH_CALUDE_inequality_relation_l1983_198390

theorem inequality_relation (a : ℝ) (h : a^2 + a < 0) : -a > a^2 ∧ a^2 > -a^3 := by
  sorry

end NUMINAMATH_CALUDE_inequality_relation_l1983_198390


namespace NUMINAMATH_CALUDE_greatest_rational_root_quadratic_l1983_198323

theorem greatest_rational_root_quadratic (a b c : ℕ) (ha : a ≤ 100) (hb : b ≤ 100) (hc : c ≤ 100) (ha_pos : a > 0) :
  ∀ (p q : ℤ), q ≠ 0 → a * (p / q)^2 + b * (p / q) + c = 0 →
  (p : ℚ) / q ≤ (-1 : ℚ) / 99 :=
sorry

end NUMINAMATH_CALUDE_greatest_rational_root_quadratic_l1983_198323


namespace NUMINAMATH_CALUDE_quadratic_solution_l1983_198391

theorem quadratic_solution (a b : ℝ) : 
  (∀ t : ℝ, t^2 - 12*t + 20 = 0 ↔ t = a ∨ t = b) →
  a > b →
  a - b = 8 →
  a = 10 := by
sorry

end NUMINAMATH_CALUDE_quadratic_solution_l1983_198391


namespace NUMINAMATH_CALUDE_modulus_of_complex_difference_l1983_198324

theorem modulus_of_complex_difference (z : ℂ) : z = -1 - Complex.I → Complex.abs (2 - z) = Real.sqrt 10 := by
  sorry

end NUMINAMATH_CALUDE_modulus_of_complex_difference_l1983_198324


namespace NUMINAMATH_CALUDE_incenter_distance_l1983_198363

/-- Represents a triangle with vertices P, Q, and R -/
structure Triangle (P Q R : ℝ × ℝ) : Prop where
  isosceles : dist P Q = dist P R
  pq_length : dist P Q = 17
  qr_length : dist Q R = 16

/-- Represents the incenter of a triangle -/
def incenter (t : Triangle P Q R) : ℝ × ℝ := sorry

/-- Represents the incircle of a triangle -/
def incircle (t : Triangle P Q R) : Set (ℝ × ℝ) := sorry

/-- Represents a point where the incircle touches a side of the triangle -/
def touchPoint (t : Triangle P Q R) (side : (ℝ × ℝ) × (ℝ × ℝ)) : ℝ × ℝ := sorry

theorem incenter_distance (P Q R : ℝ × ℝ) (t : Triangle P Q R) :
  let J := incenter t
  let C := touchPoint t (Q, R)
  dist C J = Real.sqrt 87.04 := by sorry

end NUMINAMATH_CALUDE_incenter_distance_l1983_198363


namespace NUMINAMATH_CALUDE_line_l1_equation_line_l2_intersection_range_l1983_198301

-- Define the circle C
def circle_C (x y : ℝ) : Prop := x^2 + y^2 - 6*x - 4*y + 4 = 0

-- Define the midpoint P of the chord intercepted by line l1
def midpoint_P : ℝ × ℝ := (5, 3)

-- Define line l1
def line_l1 (x y : ℝ) : Prop := 2*x + y - 13 = 0

-- Define line l2
def line_l2 (x y b : ℝ) : Prop := x + y + b = 0

-- Theorem for the equation of line l1
theorem line_l1_equation :
  ∀ x y : ℝ, line_l1 x y ↔ (∃ t : ℝ, x = midpoint_P.1 + t ∧ y = midpoint_P.2 - 2*t) :=
sorry

-- Theorem for the range of b
theorem line_l2_intersection_range :
  ∀ b : ℝ, (∃ x y : ℝ, circle_C x y ∧ line_l2 x y b) ↔ -3 * Real.sqrt 2 - 5 < b ∧ b < 3 * Real.sqrt 2 - 5 :=
sorry

end NUMINAMATH_CALUDE_line_l1_equation_line_l2_intersection_range_l1983_198301


namespace NUMINAMATH_CALUDE_masons_father_age_l1983_198342

/-- Given the ages of Mason and Sydney, and their relationship to Mason's father's age,
    prove that Mason's father is 66 years old. -/
theorem masons_father_age (mason_age sydney_age father_age : ℕ) : 
  mason_age = 20 →
  sydney_age = 3 * mason_age →
  father_age = sydney_age + 6 →
  father_age = 66 := by
  sorry

end NUMINAMATH_CALUDE_masons_father_age_l1983_198342


namespace NUMINAMATH_CALUDE_reciprocal_sum_fractions_l1983_198303

theorem reciprocal_sum_fractions : 
  (1 / (1/4 + 1/6 + 1/9) : ℚ) = 36/19 := by sorry

end NUMINAMATH_CALUDE_reciprocal_sum_fractions_l1983_198303


namespace NUMINAMATH_CALUDE_student_distribution_l1983_198393

/-- The number of ways to choose k items from n items -/
def choose (n k : ℕ) : ℕ := sorry

/-- The number of permutations of n items -/
def factorial (n : ℕ) : ℕ := sorry

theorem student_distribution (total : ℕ) (male : ℕ) (schemes : ℕ) :
  total = 8 →
  choose male 2 * choose (total - male) 1 * factorial 3 = schemes →
  schemes = 90 →
  male = 3 ∧ total - male = 5 := by sorry

end NUMINAMATH_CALUDE_student_distribution_l1983_198393


namespace NUMINAMATH_CALUDE_total_nylon_needed_l1983_198328

/-- The amount of nylon needed for a dog collar in inches -/
def dog_collar_nylon : ℕ := 18

/-- The amount of nylon needed for a cat collar in inches -/
def cat_collar_nylon : ℕ := 10

/-- The number of dog collars to be made -/
def num_dog_collars : ℕ := 9

/-- The number of cat collars to be made -/
def num_cat_collars : ℕ := 3

/-- Theorem stating the total amount of nylon needed -/
theorem total_nylon_needed : 
  dog_collar_nylon * num_dog_collars + cat_collar_nylon * num_cat_collars = 192 := by
  sorry

end NUMINAMATH_CALUDE_total_nylon_needed_l1983_198328


namespace NUMINAMATH_CALUDE_largest_square_divisor_l1983_198343

theorem largest_square_divisor : 
  ∃ (x : ℕ), x = 12 ∧ 
  x^2 ∣ (24 * 35 * 46 * 57) ∧ 
  ∀ (y : ℕ), y > x → ¬(y^2 ∣ (24 * 35 * 46 * 57)) := by
  sorry

end NUMINAMATH_CALUDE_largest_square_divisor_l1983_198343


namespace NUMINAMATH_CALUDE_journey_speed_calculation_l1983_198366

/-- 
Given a journey with total distance D and total time T, 
where a person travels 2/3 of D in 1/3 of T at 40 km/h,
prove that they must travel at 10 km/h for the remaining distance
to reach the destination on time.
-/
theorem journey_speed_calculation 
  (D T : ℝ) 
  (h1 : D > 0) 
  (h2 : T > 0) 
  (h3 : (2/3 * D) / (1/3 * T) = 40) : 
  (1/3 * D) / (2/3 * T) = 10 := by
sorry

end NUMINAMATH_CALUDE_journey_speed_calculation_l1983_198366


namespace NUMINAMATH_CALUDE_min_sum_squares_l1983_198314

theorem min_sum_squares (x y z : ℝ) : 
  x^3 + y^3 + z^3 - 3*x*y*z = 8 → 
  ∀ a b c : ℝ, a^3 + b^3 + c^3 - 3*a*b*c = 8 → 
  x^2 + y^2 + z^2 ≤ a^2 + b^2 + c^2 ∧ 
  ∃ p q r : ℝ, p^3 + q^3 + r^3 - 3*p*q*r = 8 ∧ p^2 + q^2 + r^2 = 4 :=
by sorry

end NUMINAMATH_CALUDE_min_sum_squares_l1983_198314


namespace NUMINAMATH_CALUDE_vector_sum_proof_l1983_198392

/-- Given vectors a and b in ℝ², prove that a + 2b = (-3, 4) -/
theorem vector_sum_proof (a b : ℝ × ℝ) (h1 : a = (1, 2)) (h2 : b = (-2, 1)) :
  a + 2 • b = (-3, 4) := by
  sorry

end NUMINAMATH_CALUDE_vector_sum_proof_l1983_198392


namespace NUMINAMATH_CALUDE_encoded_bec_value_l1983_198321

/-- Represents the encoding of a base 7 digit --/
inductive Encoding
  | A | B | C | D | E | F | G

/-- Represents a number in the encoded form --/
def EncodedNumber := List Encoding

/-- Converts an EncodedNumber to its base 10 representation --/
def to_base_10 (n : EncodedNumber) : ℕ := sorry

/-- Checks if three EncodedNumbers are consecutive integers --/
def are_consecutive (a b c : EncodedNumber) : Prop := sorry

theorem encoded_bec_value :
  ∃ (encode : Fin 7 → Encoding),
    Function.Injective encode ∧
    (∃ (x : ℕ), 
      are_consecutive 
        [encode (x % 7), encode ((x + 1) % 7), encode ((x + 2) % 7)]
        [encode ((x + 1) % 7), encode ((x + 2) % 7), encode ((x + 3) % 7)]
        [encode ((x + 2) % 7), encode ((x + 3) % 7), encode ((x + 4) % 7)]) →
    to_base_10 [Encoding.B, Encoding.E, Encoding.C] = 336 :=
sorry

end NUMINAMATH_CALUDE_encoded_bec_value_l1983_198321


namespace NUMINAMATH_CALUDE_city_population_ratio_l1983_198341

/-- Given the population relationships between cities X, Y, and Z, 
    prove that the ratio of City X's population to City Z's population is 6:1 -/
theorem city_population_ratio 
  (Z : ℕ) -- Population of City Z
  (Y : ℕ) -- Population of City Y
  (X : ℕ) -- Population of City X
  (h1 : Y = 2 * Z) -- City Y's population is twice City Z's
  (h2 : ∃ k : ℕ, X = k * Y) -- City X's population is some multiple of City Y's
  (h3 : X = 6 * Z) -- The ratio of City X's to City Z's population is 6
  : X / Z = 6 := by
  sorry

end NUMINAMATH_CALUDE_city_population_ratio_l1983_198341


namespace NUMINAMATH_CALUDE_prime_cube_plus_two_l1983_198345

theorem prime_cube_plus_two (m : ℕ) : 
  Prime m → Prime (m^2 + 2) → m = 3 ∧ Prime (m^3 + 2) :=
by sorry

end NUMINAMATH_CALUDE_prime_cube_plus_two_l1983_198345


namespace NUMINAMATH_CALUDE_problem_solution_l1983_198351

theorem problem_solution (x y : ℝ) (hx : x > 1) (hy : y > 1)
  (h : (Real.log x / Real.log 4)^5 + (Real.log y / Real.log 5)^5 + 10 = 
       10 * (Real.log x / Real.log 4) * (Real.log y / Real.log 5)) :
  x^2 + y^2 = 4^(2*5^(1/5)) + 5^(2*5^(1/5)) := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l1983_198351


namespace NUMINAMATH_CALUDE_stability_comparison_l1983_198354

/-- Represents a student's performance in a series of matches -/
structure StudentPerformance where
  average_score : ℝ
  variance : ℝ

/-- Defines the stability of scores based on variance -/
def more_stable (a b : StudentPerformance) : Prop :=
  a.variance < b.variance

theorem stability_comparison 
  (student_A student_B : StudentPerformance)
  (h_avg : student_A.average_score = student_B.average_score)
  (h_var_A : student_A.variance = 0.2)
  (h_var_B : student_B.variance = 0.8) :
  more_stable student_A student_B :=
sorry

end NUMINAMATH_CALUDE_stability_comparison_l1983_198354


namespace NUMINAMATH_CALUDE_circle_radius_is_four_l1983_198388

theorem circle_radius_is_four (r : ℝ) (h : 2 * (2 * Real.pi * r) = Real.pi * r^2) : r = 4 := by
  sorry

end NUMINAMATH_CALUDE_circle_radius_is_four_l1983_198388


namespace NUMINAMATH_CALUDE_sum_of_last_two_digits_of_8_pow_351_l1983_198398

theorem sum_of_last_two_digits_of_8_pow_351 :
  ∃ (a b : ℕ), a < 10 ∧ b < 10 ∧ (8^351) % 100 = 10 * a + b ∧ a + b = 11 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_last_two_digits_of_8_pow_351_l1983_198398


namespace NUMINAMATH_CALUDE_part_one_part_two_part_three_l1983_198356

-- Define the quadratic equation
def quadratic_equation (m : ℝ) (x : ℝ) : Prop :=
  3 * x^2 + m * x + 2 = 0

-- Part I
theorem part_one (m : ℝ) : quadratic_equation m 2 → m = -7 := by
  sorry

-- Part II
theorem part_two :
  (∃ x₁ x₂ : ℝ, x₁ = 1 ∧ x₂ = 2/3 ∧ quadratic_equation (-5) x₁ ∧ quadratic_equation (-5) x₂) := by
  sorry

-- Part III
theorem part_three (m : ℝ) :
  m ≥ 5 →
  ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ quadratic_equation m x₁ ∧ quadratic_equation m x₂ := by
  sorry

end NUMINAMATH_CALUDE_part_one_part_two_part_three_l1983_198356


namespace NUMINAMATH_CALUDE_pencil_distribution_l1983_198349

/-- The number of ways to distribute n identical objects among k people,
    where each person must receive at least one object. -/
def distribute (n k : ℕ) : ℕ := sorry

theorem pencil_distribution :
  distribute 7 4 = 52 := by sorry

end NUMINAMATH_CALUDE_pencil_distribution_l1983_198349


namespace NUMINAMATH_CALUDE_pascal_triangle_elements_l1983_198333

/-- The number of elements in the nth row of Pascal's Triangle -/
def elementsInRow (n : ℕ) : ℕ := n + 1

/-- The sum of elements from row 0 to row n of Pascal's Triangle -/
def sumElements (n : ℕ) : ℕ := (n + 1) * (n + 2) / 2

theorem pascal_triangle_elements : sumElements 29 = 465 := by
  sorry

end NUMINAMATH_CALUDE_pascal_triangle_elements_l1983_198333


namespace NUMINAMATH_CALUDE_right_triangle_power_equality_l1983_198372

theorem right_triangle_power_equality (a b c : ℝ) (n : ℕ) 
  (h_right_triangle : a^2 + b^2 = c^2)
  (h_n_gt_2 : n > 2)
  (h_equality : (a^n + b^n + c^n)^2 = 2*(a^(2*n) + b^(2*n) + c^(2*n))) :
  n = 4 := by sorry

end NUMINAMATH_CALUDE_right_triangle_power_equality_l1983_198372


namespace NUMINAMATH_CALUDE_binomial_18_9_l1983_198396

theorem binomial_18_9 (h1 : Nat.choose 16 7 = 11440) 
                      (h2 : Nat.choose 16 8 = 12870) 
                      (h3 : Nat.choose 16 9 = 11440) : 
  Nat.choose 18 9 = 48620 := by
  sorry

end NUMINAMATH_CALUDE_binomial_18_9_l1983_198396


namespace NUMINAMATH_CALUDE_greatest_fraction_l1983_198302

def is_valid_fraction (a b : ℕ) : Prop :=
  a + b = 101 ∧ a / b ≤ 1 / 3

theorem greatest_fraction :
  ∀ a b : ℕ, is_valid_fraction a b → a / b ≤ 25 / 76 :=
sorry

end NUMINAMATH_CALUDE_greatest_fraction_l1983_198302


namespace NUMINAMATH_CALUDE_no_real_solution_l1983_198364

theorem no_real_solution : ¬∃ (x : ℝ), |3*x + 1| + 6 = 0 := by
  sorry

end NUMINAMATH_CALUDE_no_real_solution_l1983_198364


namespace NUMINAMATH_CALUDE_combined_population_theorem_l1983_198355

/-- The combined population of New York and New England -/
def combined_population (new_england_population : ℕ) : ℕ :=
  new_england_population + (2 * new_england_population) / 3

/-- Theorem stating the combined population of New York and New England -/
theorem combined_population_theorem :
  combined_population 2100000 = 3500000 := by
  sorry

#eval combined_population 2100000

end NUMINAMATH_CALUDE_combined_population_theorem_l1983_198355


namespace NUMINAMATH_CALUDE_geometric_sequence_sum_l1983_198358

/-- A positive geometric sequence -/
def GeometricSequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, r > 0 ∧ ∀ n : ℕ, a (n + 1) = a n * r

theorem geometric_sequence_sum (a : ℕ → ℝ) :
  GeometricSequence a →
  (a 1 * a 5 + 2 * a 3 * a 6 + a 1 * a 11 = 16) →
  a 3 + a 6 = 4 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_sum_l1983_198358


namespace NUMINAMATH_CALUDE_reflection_segment_length_d_to_d_prime_length_l1983_198386

/-- The length of the segment from a point to its reflection over the x-axis --/
theorem reflection_segment_length (x y : ℝ) : 
  let d : ℝ × ℝ := (x, y)
  let d_reflected : ℝ × ℝ := (x, -y)
  Real.sqrt ((d.1 - d_reflected.1)^2 + (d.2 - d_reflected.2)^2) = 2 * |y| :=
by sorry

/-- The length of the segment from D(-5, 3) to its reflection D' over the x-axis is 6 --/
theorem d_to_d_prime_length : 
  let d : ℝ × ℝ := (-5, 3)
  let d_reflected : ℝ × ℝ := (-5, -3)
  Real.sqrt ((d.1 - d_reflected.1)^2 + (d.2 - d_reflected.2)^2) = 6 :=
by sorry

end NUMINAMATH_CALUDE_reflection_segment_length_d_to_d_prime_length_l1983_198386


namespace NUMINAMATH_CALUDE_sarah_speed_calculation_l1983_198304

def eugene_speed : ℚ := 5

def carlos_speed_ratio : ℚ := 4/5

def sarah_speed_ratio : ℚ := 6/7

def carlos_speed : ℚ := eugene_speed * carlos_speed_ratio

def sarah_speed : ℚ := carlos_speed * sarah_speed_ratio

theorem sarah_speed_calculation : sarah_speed = 24/7 := by
  sorry

end NUMINAMATH_CALUDE_sarah_speed_calculation_l1983_198304


namespace NUMINAMATH_CALUDE_extended_quadrilateral_area_l1983_198306

-- Define the quadrilateral ABCD
structure Quadrilateral (V : Type*) [AddCommGroup V] [Module ℝ V] :=
  (A B C D : V)

-- Define the extended quadrilateral A₁B₁C₁D₁
structure ExtendedQuadrilateral (V : Type*) [AddCommGroup V] [Module ℝ V] extends Quadrilateral V :=
  (A₁ B₁ C₁ D₁ : V)
  (hDA₁ : A₁ - D = 2 • (A - D))
  (hAB₁ : B₁ - A = 2 • (B - A))
  (hBC₁ : C₁ - B = 2 • (C - B))
  (hCD₁ : D₁ - C = 2 • (D - C))

-- Define the area function
noncomputable def area {V : Type*} [AddCommGroup V] [Module ℝ V] (q : Quadrilateral V) : ℝ := sorry

-- State the theorem
theorem extended_quadrilateral_area {V : Type*} [AddCommGroup V] [Module ℝ V] 
  (q : ExtendedQuadrilateral V) :
  area {A := q.A₁, B := q.B₁, C := q.C₁, D := q.D₁} = 5 * area {A := q.A, B := q.B, C := q.C, D := q.D} :=
sorry

end NUMINAMATH_CALUDE_extended_quadrilateral_area_l1983_198306


namespace NUMINAMATH_CALUDE_tom_apple_purchase_l1983_198369

-- Define the given constants
def apple_price : ℝ := 70
def mango_amount : ℝ := 9
def mango_price : ℝ := 65
def total_paid : ℝ := 1145

-- Define the theorem
theorem tom_apple_purchase :
  ∃ (apple_amount : ℝ),
    apple_amount * apple_price + mango_amount * mango_price = total_paid ∧
    apple_amount = 8 := by
  sorry

end NUMINAMATH_CALUDE_tom_apple_purchase_l1983_198369


namespace NUMINAMATH_CALUDE_stratified_sampling_l1983_198317

/-- Represents the total number of teachers -/
def total_teachers : ℕ := 300

/-- Represents the number of senior teachers -/
def senior_teachers : ℕ := 90

/-- Represents the number of intermediate teachers -/
def intermediate_teachers : ℕ := 150

/-- Represents the number of junior teachers -/
def junior_teachers : ℕ := 60

/-- Represents the sample size -/
def sample_size : ℕ := 60

/-- Theorem stating the correct stratified sampling for each teacher category -/
theorem stratified_sampling :
  (senior_teachers * sample_size) / total_teachers = 18 ∧
  (intermediate_teachers * sample_size) / total_teachers = 30 ∧
  (junior_teachers * sample_size) / total_teachers = 12 := by
  sorry

end NUMINAMATH_CALUDE_stratified_sampling_l1983_198317


namespace NUMINAMATH_CALUDE_n_equals_two_l1983_198310

theorem n_equals_two (x y n : ℕ) : 
  x = 3 → y = 1 → n = x - y^(x-(y+1)) → n = 2 := by
sorry

end NUMINAMATH_CALUDE_n_equals_two_l1983_198310


namespace NUMINAMATH_CALUDE_factorial_ratio_equals_24_l1983_198370

theorem factorial_ratio_equals_24 :
  ∃! (n : ℕ), n > 3 ∧ n.factorial / (n - 3).factorial = 24 := by
  sorry

end NUMINAMATH_CALUDE_factorial_ratio_equals_24_l1983_198370


namespace NUMINAMATH_CALUDE_geometric_sum_half_five_l1983_198384

def geometric_sum (a : ℚ) (r : ℚ) (n : ℕ) : ℚ :=
  a * (1 - r^n) / (1 - r)

theorem geometric_sum_half_five :
  geometric_sum (1/2) (1/2) 5 = 31/32 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sum_half_five_l1983_198384


namespace NUMINAMATH_CALUDE_special_function_value_at_neg_two_l1983_198331

/-- A function satisfying the given property -/
def special_function (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, f (x + y) = f x + f y + 4 * x * y

theorem special_function_value_at_neg_two
  (f : ℝ → ℝ) (h : special_function f) (h1 : f 1 = 2) :
  f (-2) = 8 := by
  sorry

end NUMINAMATH_CALUDE_special_function_value_at_neg_two_l1983_198331


namespace NUMINAMATH_CALUDE_tangent_line_at_2_a_range_l1983_198368

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := x^3 - a*x^2 + 10

-- Part 1: Tangent line when a = 1
theorem tangent_line_at_2 :
  ∃ (m b : ℝ), ∀ (x y : ℝ),
    y = m*x + b ↔ y = 8*x - 2 ∧ 
    (∃ (h : ℝ), h ≠ 0 ∧ (f 1 (2 + h) - f 1 2) / h = m) :=
sorry

-- Part 2: Range of a
theorem a_range :
  ∀ (a : ℝ), (∃ (x : ℝ), x ∈ Set.Icc 1 2 ∧ f a x < 0) ↔ a > 9/2 :=
sorry

end NUMINAMATH_CALUDE_tangent_line_at_2_a_range_l1983_198368


namespace NUMINAMATH_CALUDE_larger_solid_volume_is_4_point_5_l1983_198380

-- Define the rectangular prism
def rectangular_prism (length width height : ℝ) := length * width * height

-- Define a plane that cuts the prism
structure cutting_plane (length width height : ℝ) :=
  (passes_through_vertex : Bool)
  (passes_through_midpoint_edge1 : Bool)
  (passes_through_midpoint_edge2 : Bool)

-- Define the volume of the larger solid resulting from the cut
def larger_solid_volume (length width height : ℝ) (plane : cutting_plane length width height) : ℝ :=
  sorry

-- Theorem statement
theorem larger_solid_volume_is_4_point_5 :
  ∀ (plane : cutting_plane 2 1 3),
    plane.passes_through_vertex = true ∧
    plane.passes_through_midpoint_edge1 = true ∧
    plane.passes_through_midpoint_edge2 = true →
    larger_solid_volume 2 1 3 plane = 4.5 :=
by sorry

end NUMINAMATH_CALUDE_larger_solid_volume_is_4_point_5_l1983_198380


namespace NUMINAMATH_CALUDE_unique_base_solution_l1983_198320

/-- Convert a number from base b to decimal --/
def to_decimal (digits : List Nat) (b : Nat) : Nat :=
  digits.foldl (fun acc d => acc * b + d) 0

/-- Check if the equation holds for a given base --/
def equation_holds (b : Nat) : Prop :=
  to_decimal [2, 5, 1] b + to_decimal [1, 7, 4] b = to_decimal [4, 3, 5] b

theorem unique_base_solution :
  ∃! b : Nat, b > 1 ∧ equation_holds b :=
sorry

end NUMINAMATH_CALUDE_unique_base_solution_l1983_198320


namespace NUMINAMATH_CALUDE_condo_penthouse_floors_l1983_198382

/-- Represents a condo building with regular and penthouse floors -/
structure Condo where
  total_floors : ℕ
  regular_units_per_floor : ℕ
  penthouse_units_per_floor : ℕ
  total_units : ℕ

/-- Calculates the number of penthouse floors in a condo -/
def penthouse_floors (c : Condo) : ℕ :=
  c.total_floors - (c.total_units - 2 * c.total_floors) / (c.regular_units_per_floor - c.penthouse_units_per_floor)

/-- Theorem stating that the condo with given specifications has 2 penthouse floors -/
theorem condo_penthouse_floors :
  let c : Condo := {
    total_floors := 23,
    regular_units_per_floor := 12,
    penthouse_units_per_floor := 2,
    total_units := 256
  }
  penthouse_floors c = 2 := by
  sorry

end NUMINAMATH_CALUDE_condo_penthouse_floors_l1983_198382


namespace NUMINAMATH_CALUDE_absolute_value_probability_l1983_198338

theorem absolute_value_probability (x : ℝ) : ℝ := by
  have h : ∀ x : ℝ, |x| ≥ 0 := by sorry
  have event : Set ℝ := {x | |x| < 0}
  have prob_event : ℝ := 0
  sorry

end NUMINAMATH_CALUDE_absolute_value_probability_l1983_198338


namespace NUMINAMATH_CALUDE_center_value_is_35_l1983_198371

/-- Represents a 4x4 array where each row and column forms an arithmetic sequence -/
def ArithmeticArray := Matrix (Fin 4) (Fin 4) ℝ

/-- Checks if a row or column is an arithmetic sequence -/
def is_arithmetic_sequence (seq : Fin 4 → ℝ) : Prop :=
  ∃ d : ℝ, ∀ i j : Fin 4, i.val < j.val → seq j - seq i = d * (j.val - i.val)

/-- Definition of our specific arithmetic array -/
def special_array (A : ArithmeticArray) : Prop :=
  (∀ i : Fin 4, is_arithmetic_sequence (λ j => A i j)) ∧ 
  (∀ j : Fin 4, is_arithmetic_sequence (λ i => A i j)) ∧
  A 0 0 = 3 ∧ A 0 3 = 27 ∧ A 3 0 = 6 ∧ A 3 3 = 66

/-- The center value of the array -/
def center_value (A : ArithmeticArray) : ℝ := A 1 1

theorem center_value_is_35 (A : ArithmeticArray) (h : special_array A) : 
  center_value A = 35 := by
  sorry

end NUMINAMATH_CALUDE_center_value_is_35_l1983_198371


namespace NUMINAMATH_CALUDE_store_goods_values_l1983_198347

/-- Given a store with two grades of goods, prove the initial values of the goods. -/
theorem store_goods_values (x y : ℝ) (a b : ℝ) (h1 : x + y = 450)
  (h2 : y / b * (a + b) = 400) (h3 : x / a * (a + b) = 480) :
  x = 300 ∧ y = 150 := by
  sorry


end NUMINAMATH_CALUDE_store_goods_values_l1983_198347


namespace NUMINAMATH_CALUDE_quadratic_equation_solutions_l1983_198344

theorem quadratic_equation_solutions :
  (∀ x : ℝ, 3 * x^2 - 6 * x - 2 = 0 ↔ x = (3 + Real.sqrt 15) / 3 ∨ x = (3 - Real.sqrt 15) / 3) ∧
  (∀ x : ℝ, x^2 + 6 * x + 8 = 0 ↔ x = -2 ∨ x = -4) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_solutions_l1983_198344


namespace NUMINAMATH_CALUDE_brothers_money_distribution_l1983_198397

/-- Represents the money distribution among four brothers -/
structure MoneyDistribution where
  john : ℕ
  william : ℕ
  charles : ℕ
  thomas : ℕ

/-- Checks if the given money distribution satisfies all conditions -/
def satisfies_conditions (d : MoneyDistribution) : Prop :=
  d.john + 2 = d.william - 2 ∧
  d.john + 2 = 2 * d.charles ∧
  d.john + 2 = d.thomas / 2 ∧
  d.john + d.william + d.charles + d.thomas = 45

/-- Checks if the given money distribution can be represented with 6 coins -/
def can_be_represented_with_six_coins (d : MoneyDistribution) : Prop :=
  ∃ (j1 j2 w1 w2 c t : ℕ),
    j1 + j2 = d.john ∧
    w1 + w2 = d.william ∧
    c = d.charles ∧
    t = d.thomas

/-- The main theorem stating the unique solution for the brothers' money distribution -/
theorem brothers_money_distribution :
  ∃! (d : MoneyDistribution),
    satisfies_conditions d ∧
    can_be_represented_with_six_coins d ∧
    d.john = 8 ∧ d.william = 12 ∧ d.charles = 5 ∧ d.thomas = 20 :=
by
  sorry

end NUMINAMATH_CALUDE_brothers_money_distribution_l1983_198397


namespace NUMINAMATH_CALUDE_log_expression_equals_negative_four_l1983_198352

-- Define the logarithm base 10 function
noncomputable def lg (x : ℝ) := Real.log x / Real.log 10

-- State the theorem
theorem log_expression_equals_negative_four :
  (lg 8 + lg 125 - lg 2 - lg 5) / (lg (Real.sqrt 10) * lg 0.1) = -4 := by
  sorry

end NUMINAMATH_CALUDE_log_expression_equals_negative_four_l1983_198352


namespace NUMINAMATH_CALUDE_square_plus_abs_zero_implies_both_zero_l1983_198319

theorem square_plus_abs_zero_implies_both_zero (a b : ℝ) : 
  a^2 + |b| = 0 → a = 0 ∧ b = 0 := by
  sorry

end NUMINAMATH_CALUDE_square_plus_abs_zero_implies_both_zero_l1983_198319


namespace NUMINAMATH_CALUDE_max_value_of_y_l1983_198361

def y (x : ℝ) : ℝ := |x + 1| - 2 * |x| + |x - 2|

theorem max_value_of_y :
  ∃ (α : ℝ), α = 3 ∧ ∀ x, -1 ≤ x → x ≤ 2 → y x ≤ α :=
by sorry

end NUMINAMATH_CALUDE_max_value_of_y_l1983_198361


namespace NUMINAMATH_CALUDE_rectangle_area_l1983_198367

/-- Given a rectangle with width w and length L, where L = w^2 and L + w = 25,
    prove that the area of the rectangle is (√101 - 1)^3 / 8 square inches. -/
theorem rectangle_area (w L : ℝ) (h1 : L = w^2) (h2 : L + w = 25) :
  w * L = ((Real.sqrt 101 - 1)^3) / 8 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_area_l1983_198367


namespace NUMINAMATH_CALUDE_equation_solution_l1983_198309

theorem equation_solution (A : ℕ+) : 
  (∃! (x₁ y₁ x₂ y₂ : ℕ+), x₁ ≠ x₂ ∧ y₁ ≠ y₂ ∧ 
    A * x₁ + 10 * y₁ = 100 ∧ A * x₂ + 10 * y₂ = 100) ↔ A = 10 :=
by sorry

end NUMINAMATH_CALUDE_equation_solution_l1983_198309


namespace NUMINAMATH_CALUDE_correct_dispatch_plans_l1983_198322

/-- The number of teachers available for selection -/
def total_teachers : ℕ := 8

/-- The number of teachers to be selected -/
def selected_teachers : ℕ := 4

/-- The number of remote areas -/
def remote_areas : ℕ := 4

/-- Function to calculate the number of ways to select teachers -/
def select_teachers : ℕ :=
  let with_a_c := Nat.choose (total_teachers - 3) (selected_teachers - 2)
  let without_a_c := Nat.choose (total_teachers - 2) selected_teachers
  with_a_c + without_a_c

/-- Function to calculate the number of ways to arrange teachers in areas -/
def arrange_teachers : ℕ := Nat.factorial selected_teachers

/-- The total number of different dispatch plans -/
def total_dispatch_plans : ℕ := select_teachers * arrange_teachers

theorem correct_dispatch_plans : total_dispatch_plans = 600 := by
  sorry

end NUMINAMATH_CALUDE_correct_dispatch_plans_l1983_198322


namespace NUMINAMATH_CALUDE_first_team_pies_l1983_198373

/-- Given a catering problem with three teams making pies, prove the number of pies made by the first team. -/
theorem first_team_pies (total_pies : ℕ) (team2_pies : ℕ) (team3_pies : ℕ)
  (h_total : total_pies = 750)
  (h_team2 : team2_pies = 275)
  (h_team3 : team3_pies = 240) :
  total_pies - team2_pies - team3_pies = 235 := by
  sorry

#check first_team_pies

end NUMINAMATH_CALUDE_first_team_pies_l1983_198373


namespace NUMINAMATH_CALUDE_equation_is_linear_l1983_198335

/-- A linear equation in two variables has the form ax + by = c, where a, b, and c are constants, and x and y are variables. -/
def IsLinearEquationInTwoVariables (f : ℝ → ℝ → ℝ) : Prop :=
  ∃ (a b c : ℝ), ∀ x y, f x y = a * x + b * y - c

/-- The equation 4x - y = 3 -/
def equation (x y : ℝ) : ℝ := 4 * x - y - 3

theorem equation_is_linear : IsLinearEquationInTwoVariables equation := by
  sorry


end NUMINAMATH_CALUDE_equation_is_linear_l1983_198335


namespace NUMINAMATH_CALUDE_combined_return_is_ten_percent_l1983_198375

/-- The combined yearly return percentage of two investments -/
def combined_return_percentage (investment1 investment2 return1 return2 : ℚ) : ℚ :=
  ((investment1 * return1 + investment2 * return2) / (investment1 + investment2)) * 100

/-- Theorem: The combined yearly return percentage of a $500 investment with 7% return
    and a $1500 investment with 11% return is 10% -/
theorem combined_return_is_ten_percent :
  combined_return_percentage 500 1500 (7/100) (11/100) = 10 := by
  sorry

end NUMINAMATH_CALUDE_combined_return_is_ten_percent_l1983_198375


namespace NUMINAMATH_CALUDE_best_estimate_on_number_line_l1983_198332

theorem best_estimate_on_number_line (x : ℝ) (h1 : x < 0) (h2 : -2 < x) (h3 : x < -1) :
  let options := [1.3, -1.3, -2.7, 0.7, -0.7]
  (-1.3 : ℝ) = options.argmin (fun y => |x - y|) := by
  sorry

end NUMINAMATH_CALUDE_best_estimate_on_number_line_l1983_198332


namespace NUMINAMATH_CALUDE_eraser_price_l1983_198337

/-- Proves that the price of an eraser is $1 given the problem conditions --/
theorem eraser_price (pencils_sold : ℕ) (total_earnings : ℝ) 
  (h1 : pencils_sold = 20)
  (h2 : total_earnings = 80)
  (h3 : ∀ p : ℝ, p > 0 → 
    pencils_sold * p + 2 * pencils_sold * (p / 2) = total_earnings) :
  ∃ (pencil_price : ℝ), 
    pencil_price > 0 ∧ 
    pencil_price / 2 = 1 := by
  sorry

end NUMINAMATH_CALUDE_eraser_price_l1983_198337


namespace NUMINAMATH_CALUDE_reflection_line_sum_l1983_198325

/-- Given a line y = mx + b, if the reflection of point (2,3) across this line is (10,7), then m + b = 15 -/
theorem reflection_line_sum (m b : ℝ) : 
  (∃ (x y : ℝ), 
    -- The midpoint of the original and reflected points lies on the line
    y = m * x + b ∧ 
    x = (2 + 10) / 2 ∧ 
    y = (3 + 7) / 2 ∧
    -- The line is perpendicular to the line segment between the original and reflected points
    m * ((10 - 2) / (7 - 3)) = -1) → 
  m + b = 15 := by
sorry

end NUMINAMATH_CALUDE_reflection_line_sum_l1983_198325


namespace NUMINAMATH_CALUDE_complex_square_plus_one_zero_l1983_198383

theorem complex_square_plus_one_zero (x : ℂ) : x^2 + 1 = 0 → x = Complex.I ∨ x = -Complex.I := by
  sorry

end NUMINAMATH_CALUDE_complex_square_plus_one_zero_l1983_198383


namespace NUMINAMATH_CALUDE_spaceship_journey_l1983_198340

def total_distance : Real := 0.7
def earth_to_x : Real := 0.5
def y_to_earth : Real := 0.1

theorem spaceship_journey : 
  total_distance - earth_to_x - y_to_earth = 0.1 := by sorry

end NUMINAMATH_CALUDE_spaceship_journey_l1983_198340


namespace NUMINAMATH_CALUDE_quadratic_roots_transformation_l1983_198318

theorem quadratic_roots_transformation (a b : ℝ) (r₁ r₂ : ℝ) : 
  (r₁^2 - a*r₁ + b = 0) → 
  (r₂^2 - a*r₂ + b = 0) → 
  ∃ (x : ℝ), x^2 - (a^2 + a - 2*b)*x + (a^3 - a*b) = 0 ∧ 
  (x = r₁^2 + r₂ ∨ x = r₁ + r₂^2) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_roots_transformation_l1983_198318


namespace NUMINAMATH_CALUDE_richmond_tigers_ticket_sales_l1983_198395

theorem richmond_tigers_ticket_sales (total_tickets second_half_tickets : ℕ) 
    (h1 : total_tickets = 9570)
    (h2 : second_half_tickets = 5703) :
  total_tickets - second_half_tickets = 3867 := by
  sorry

end NUMINAMATH_CALUDE_richmond_tigers_ticket_sales_l1983_198395


namespace NUMINAMATH_CALUDE_linear_function_max_value_l1983_198376

theorem linear_function_max_value (m : ℝ) :
  (∀ x : ℝ, -1 ≤ x ∧ x ≤ 3 → m * x - 2 * m ≤ 6) ∧
  (∃ x : ℝ, -1 ≤ x ∧ x ≤ 3 ∧ m * x - 2 * m = 6) →
  m = -2 ∨ m = 6 := by
sorry

end NUMINAMATH_CALUDE_linear_function_max_value_l1983_198376


namespace NUMINAMATH_CALUDE_prime_implication_l1983_198350

theorem prime_implication (p : ℕ) (hp : Prime p) (h8p2_1 : Prime (8 * p^2 + 1)) :
  Prime (8 * p^2 - p + 2) := by
  sorry

end NUMINAMATH_CALUDE_prime_implication_l1983_198350


namespace NUMINAMATH_CALUDE_zachary_pushups_l1983_198326

theorem zachary_pushups (david_pushups : ℕ) (difference : ℕ) (zachary_pushups : ℕ) 
  (h1 : david_pushups = 37)
  (h2 : david_pushups = zachary_pushups + difference)
  (h3 : difference = 30) : 
  zachary_pushups = 7 := by
  sorry

end NUMINAMATH_CALUDE_zachary_pushups_l1983_198326


namespace NUMINAMATH_CALUDE_set_relationship_l1983_198336

/-- Definition of set E -/
def E : Set ℚ := { e | ∃ m : ℤ, e = m + 1/6 }

/-- Definition of set F -/
def F : Set ℚ := { f | ∃ n : ℤ, f = n/2 - 1/3 }

/-- Definition of set G -/
def G : Set ℚ := { g | ∃ p : ℤ, g = p/2 + 1/6 }

/-- Theorem stating the relationship among sets E, F, and G -/
theorem set_relationship : E ⊆ F ∧ F = G := by
  sorry

end NUMINAMATH_CALUDE_set_relationship_l1983_198336


namespace NUMINAMATH_CALUDE_sequence_sum_7_l1983_198311

def sequence_sum (a : ℕ → ℚ) (n : ℕ) : ℚ :=
  Finset.sum (Finset.range n) (λ i => a (i + 1))

theorem sequence_sum_7 (a : ℕ → ℚ) (S : ℕ → ℚ) :
  (∀ n, S n = sequence_sum a n) →
  a 1 = 1/2 →
  (∀ n, a (n + 1) = 2 * S n + 1) →
  S 7 = 1457/2 := by
sorry

end NUMINAMATH_CALUDE_sequence_sum_7_l1983_198311


namespace NUMINAMATH_CALUDE_ratio_s4_s5_l1983_198316

/-- An arithmetic sequence with a given ratio of second to third term -/
structure ArithmeticSequence where
  a : ℕ → ℚ
  is_arithmetic : ∀ n, a (n + 2) - a (n + 1) = a (n + 1) - a n
  ratio_condition : a 2 / a 3 = 1 / 3

/-- Sum of the first n terms of an arithmetic sequence -/
def sum_n (seq : ArithmeticSequence) (n : ℕ) : ℚ :=
  (seq.a 1 + seq.a n) * n / 2

/-- The main theorem: ratio of S_4 to S_5 is 8/15 -/
theorem ratio_s4_s5 (seq : ArithmeticSequence) :
  sum_n seq 4 / sum_n seq 5 = 8 / 15 := by
  sorry

end NUMINAMATH_CALUDE_ratio_s4_s5_l1983_198316


namespace NUMINAMATH_CALUDE_p_satisfies_conditions_l1983_198389

/-- A quadratic polynomial that satisfies specific conditions -/
def p (x : ℝ) : ℝ := -3 * x^2 - 9 * x + 84

/-- Theorem stating that p satisfies the required conditions -/
theorem p_satisfies_conditions :
  p (-7) = 0 ∧ p 4 = 0 ∧ p 5 = -36 := by
  sorry

end NUMINAMATH_CALUDE_p_satisfies_conditions_l1983_198389


namespace NUMINAMATH_CALUDE_friend_ratio_proof_l1983_198307

theorem friend_ratio_proof (julian_total : ℕ) (julian_boy_percent : ℚ) (julian_girl_percent : ℚ)
  (boyd_total : ℕ) (boyd_boy_percent : ℚ) :
  julian_total = 80 →
  julian_boy_percent = 60 / 100 →
  julian_girl_percent = 40 / 100 →
  boyd_total = 100 →
  boyd_boy_percent = 36 / 100 →
  (boyd_total - boyd_total * boyd_boy_percent) / (julian_total * julian_girl_percent) = 2 / 1 :=
by sorry

end NUMINAMATH_CALUDE_friend_ratio_proof_l1983_198307


namespace NUMINAMATH_CALUDE_solve_for_y_l1983_198353

theorem solve_for_y (x y : ℝ) (h1 : x^2 - 4*x = y + 5) (h2 : x = 7) : y = 16 := by
  sorry

end NUMINAMATH_CALUDE_solve_for_y_l1983_198353


namespace NUMINAMATH_CALUDE_ternary_10201_equals_100_l1983_198313

/-- Converts a list of digits in base 3 to a natural number in base 10 -/
def ternary_to_decimal (digits : List Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * (3 ^ i)) 0

theorem ternary_10201_equals_100 :
  ternary_to_decimal [1, 0, 2, 0, 1] = 100 := by
  sorry

end NUMINAMATH_CALUDE_ternary_10201_equals_100_l1983_198313


namespace NUMINAMATH_CALUDE_semicircle_circumference_from_rectangle_square_l1983_198357

/-- Given a rectangle and a square with equal perimeters, prove the circumference of a semicircle
    whose diameter is equal to the side of the square. -/
theorem semicircle_circumference_from_rectangle_square 
  (rect_length : ℝ) (rect_breadth : ℝ) (square_side : ℝ) :
  rect_length = 8 →
  rect_breadth = 6 →
  2 * (rect_length + rect_breadth) = 4 * square_side →
  ∃ (semicircle_circumference : ℝ), 
    semicircle_circumference = Real.pi * square_side / 2 + square_side :=
by sorry

end NUMINAMATH_CALUDE_semicircle_circumference_from_rectangle_square_l1983_198357


namespace NUMINAMATH_CALUDE_max_A_value_l1983_198339

/-- Represents the board configuration after chip removal operations -/
structure BoardConfig where
  white_columns : Nat
  white_rows : Nat
  black_columns : Nat
  black_rows : Nat

/-- Calculates the number of remaining chips for a given color -/
def remaining_chips (config : BoardConfig) (color : Bool) : Nat :=
  if color then config.white_columns * config.white_rows
  else config.black_columns * config.black_rows

/-- The size of the board -/
def board_size : Nat := 2018

/-- Theorem stating the maximum value of A -/
theorem max_A_value :
  ∃ (config : BoardConfig),
    config.white_columns + config.black_columns = board_size ∧
    config.white_rows + config.black_rows = board_size ∧
    ∀ (other_config : BoardConfig),
      other_config.white_columns + other_config.black_columns = board_size →
      other_config.white_rows + other_config.black_rows = board_size →
      min (remaining_chips config true) (remaining_chips config false) ≥
      min (remaining_chips other_config true) (remaining_chips other_config false) ∧
    min (remaining_chips config true) (remaining_chips config false) = 1018081 :=
sorry

end NUMINAMATH_CALUDE_max_A_value_l1983_198339


namespace NUMINAMATH_CALUDE_triangle_perimeter_l1983_198305

theorem triangle_perimeter (a b c : ℝ) : 
  (a ≥ 0) → (b ≥ 0) → (c ≥ 0) →
  (a^2 + 5*b^2 + c^2 - 4*a*b - 6*b - 10*c + 34 = 0) →
  (a + b + c = 14) := by
sorry

end NUMINAMATH_CALUDE_triangle_perimeter_l1983_198305


namespace NUMINAMATH_CALUDE_complex_fraction_real_l1983_198399

/-- Given that i is the imaginary unit and (a+2i)/(1+i) is a real number, prove that a = 2 -/
theorem complex_fraction_real (a : ℝ) : 
  (Complex.I : ℂ)^2 = -1 →
  (↑a + 2 * Complex.I) / (1 + Complex.I) ∈ Set.range (Complex.ofReal : ℝ → ℂ) → 
  a = 2 := by
  sorry

end NUMINAMATH_CALUDE_complex_fraction_real_l1983_198399


namespace NUMINAMATH_CALUDE_eight_holes_when_unfolded_l1983_198348

/-- Represents a fold on the triangular paper -/
structure Fold where
  vertex : ℕ
  midpoint : ℕ

/-- Represents the triangular paper with its folds and holes -/
structure TriangularPaper where
  folds : List Fold
  holes : ℕ

/-- Performs a fold on the triangular paper -/
def applyFold (paper : TriangularPaper) (fold : Fold) : TriangularPaper :=
  { folds := fold :: paper.folds, holes := paper.holes }

/-- Punches holes in the folded paper -/
def punchHoles (paper : TriangularPaper) (n : ℕ) : TriangularPaper :=
  { folds := paper.folds, holes := paper.holes + n }

/-- Unfolds the paper and calculates the total number of holes -/
def unfold (paper : TriangularPaper) : ℕ :=
  match paper.folds.length with
  | 0 => paper.holes
  | 1 => 2 * paper.holes
  | _ => 4 * paper.holes

/-- Theorem stating that folding an equilateral triangle twice and punching two holes results in eight holes when unfolded -/
theorem eight_holes_when_unfolded (initialPaper : TriangularPaper) : 
  let firstFold := Fold.mk 1 2
  let secondFold := Fold.mk 3 4
  let foldedPaper := applyFold (applyFold initialPaper firstFold) secondFold
  let punchedPaper := punchHoles foldedPaper 2
  unfold punchedPaper = 8 := by
  sorry


end NUMINAMATH_CALUDE_eight_holes_when_unfolded_l1983_198348


namespace NUMINAMATH_CALUDE_pencil_count_l1983_198377

theorem pencil_count (group_size : ℕ) (num_groups : ℕ) (h1 : group_size = 11) (h2 : num_groups = 14) :
  group_size * num_groups = 154 := by
  sorry

end NUMINAMATH_CALUDE_pencil_count_l1983_198377


namespace NUMINAMATH_CALUDE_simplify_expression_l1983_198346

theorem simplify_expression (x : ℝ) : 
  2*x - 3*(2-x) + (1/2)*(3-2*x) - 5*(2+3*x) = -11*x - 15.5 := by
sorry

end NUMINAMATH_CALUDE_simplify_expression_l1983_198346


namespace NUMINAMATH_CALUDE_sum_f_mod_1000_l1983_198334

-- Define the function f
def f (n : ℕ) : ℕ := 
  (Finset.filter (fun d => d < n ∨ Nat.gcd d n ≠ 1) (Nat.divisors (2024^2024))).card

-- State the theorem
theorem sum_f_mod_1000 : 
  (Finset.sum (Finset.range (2024^2024 + 1)) f) % 1000 = 224 := by sorry

end NUMINAMATH_CALUDE_sum_f_mod_1000_l1983_198334


namespace NUMINAMATH_CALUDE_interest_rate_calculation_l1983_198362

/-- Simple interest calculation function -/
def simple_interest (principal : ℝ) (rate : ℝ) (time : ℝ) : ℝ :=
  principal * rate * time

theorem interest_rate_calculation (principal1 principal2 time1 time2 rate1 : ℝ) 
  (h1 : principal1 = 100)
  (h2 : principal2 = 600)
  (h3 : time1 = 48)
  (h4 : time2 = 4)
  (h5 : rate1 = 0.05)
  (h6 : simple_interest principal1 rate1 time1 = simple_interest principal2 ((10 : ℝ) / 100) time2) :
  ∃ (rate2 : ℝ), rate2 = (10 : ℝ) / 100 ∧ 
    simple_interest principal1 rate1 time1 = simple_interest principal2 rate2 time2 :=
by sorry

end NUMINAMATH_CALUDE_interest_rate_calculation_l1983_198362


namespace NUMINAMATH_CALUDE_initial_average_production_l1983_198385

theorem initial_average_production 
  (n : ℕ) 
  (today_production : ℕ) 
  (new_average : ℚ) 
  (h1 : n = 8) 
  (h2 : today_production = 95) 
  (h3 : new_average = 55) : 
  (n : ℚ) * (n * new_average - today_production) / (n * (n + 1)) = 50 := by
  sorry

end NUMINAMATH_CALUDE_initial_average_production_l1983_198385


namespace NUMINAMATH_CALUDE_tree_boy_growth_rate_ratio_l1983_198379

/-- Given the initial and final heights of a tree and a boy, calculate the ratio of their growth rates. -/
theorem tree_boy_growth_rate_ratio
  (tree_initial : ℝ) (tree_final : ℝ)
  (boy_initial : ℝ) (boy_final : ℝ)
  (h_tree_initial : tree_initial = 16)
  (h_tree_final : tree_final = 40)
  (h_boy_initial : boy_initial = 24)
  (h_boy_final : boy_final = 36) :
  (tree_final - tree_initial) / (boy_final - boy_initial) = 2 := by
sorry

end NUMINAMATH_CALUDE_tree_boy_growth_rate_ratio_l1983_198379


namespace NUMINAMATH_CALUDE_min_value_of_f_l1983_198360

/-- The base-10 logarithm function -/
noncomputable def lg (x : ℝ) : ℝ := Real.log x / Real.log 10

/-- The function to be minimized -/
noncomputable def f (x : ℝ) : ℝ := lg x + (Real.log 10) / (Real.log x)

theorem min_value_of_f :
  ∀ x > 1, f x ≥ 2 ∧ f 10 = 2 :=
by sorry

end NUMINAMATH_CALUDE_min_value_of_f_l1983_198360


namespace NUMINAMATH_CALUDE_male_salmon_count_l1983_198394

def total_salmon : ℕ := 971639
def female_salmon : ℕ := 259378

theorem male_salmon_count : total_salmon - female_salmon = 712261 := by
  sorry

end NUMINAMATH_CALUDE_male_salmon_count_l1983_198394


namespace NUMINAMATH_CALUDE_strawberry_cakes_ordered_l1983_198308

/-- The number of strawberry cakes Leila ordered -/
def strawberry_cakes : ℕ := 
  let chocolate_cake_price : ℕ := 12
  let strawberry_cake_price : ℕ := 22
  let chocolate_cakes_ordered : ℕ := 3
  let total_payment : ℕ := 168
  (total_payment - chocolate_cake_price * chocolate_cakes_ordered) / strawberry_cake_price

theorem strawberry_cakes_ordered : strawberry_cakes = 6 := by
  sorry

end NUMINAMATH_CALUDE_strawberry_cakes_ordered_l1983_198308


namespace NUMINAMATH_CALUDE_subtracted_value_l1983_198387

theorem subtracted_value (N : ℝ) (x : ℝ) : 
  ((N - x) / 7 = 7) ∧ ((N - 2) / 13 = 4) → x = 5 := by
  sorry

end NUMINAMATH_CALUDE_subtracted_value_l1983_198387


namespace NUMINAMATH_CALUDE_quadratic_unique_solution_l1983_198312

theorem quadratic_unique_solution (a c : ℝ) : 
  (∃! x, a * x^2 - 6 * x + c = 0) →  -- exactly one solution
  (a + c = 14) →                     -- sum condition
  (a > c) →                          -- inequality condition
  (a = 7 + 2 * Real.sqrt 10 ∧ c = 7 - 2 * Real.sqrt 10) := by
sorry

end NUMINAMATH_CALUDE_quadratic_unique_solution_l1983_198312


namespace NUMINAMATH_CALUDE_polynomial_value_l1983_198374

theorem polynomial_value (x y : ℝ) (h : x - y = 5) :
  (x - y)^2 + 2*(x - y) - 10 = 25 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_value_l1983_198374


namespace NUMINAMATH_CALUDE_devin_teaching_difference_l1983_198300

def total_teaching_years : ℕ := 70
def tom_teaching_years : ℕ := 50

theorem devin_teaching_difference : 
  ∃ (devin_years : ℕ), 
    devin_years + tom_teaching_years = total_teaching_years ∧ 
    devin_years < tom_teaching_years / 2 ∧
    tom_teaching_years / 2 - devin_years = 5 := by
  sorry

end NUMINAMATH_CALUDE_devin_teaching_difference_l1983_198300


namespace NUMINAMATH_CALUDE_sarah_christmas_shopping_l1983_198330

/-- The amount of money Sarah started with for Christmas shopping. -/
def initial_amount : ℕ := 100

/-- The cost of each toy car. -/
def toy_car_cost : ℕ := 11

/-- The number of toy cars Sarah bought. -/
def num_toy_cars : ℕ := 2

/-- The cost of the scarf. -/
def scarf_cost : ℕ := 10

/-- The cost of the beanie. -/
def beanie_cost : ℕ := 14

/-- The cost of the necklace. -/
def necklace_cost : ℕ := 20

/-- The cost of the gloves. -/
def gloves_cost : ℕ := 12

/-- The cost of the book. -/
def book_cost : ℕ := 15

/-- The amount of money Sarah has remaining after purchasing all gifts. -/
def remaining_amount : ℕ := 7

/-- Theorem stating that the initial amount is equal to the sum of all gift costs plus the remaining amount. -/
theorem sarah_christmas_shopping :
  initial_amount = 
    num_toy_cars * toy_car_cost + 
    scarf_cost + 
    beanie_cost + 
    necklace_cost + 
    gloves_cost + 
    book_cost + 
    remaining_amount :=
by sorry

end NUMINAMATH_CALUDE_sarah_christmas_shopping_l1983_198330


namespace NUMINAMATH_CALUDE_father_son_age_problem_l1983_198329

theorem father_son_age_problem (x : ℕ) : x = 4 :=
by
  -- Son's current age
  let son_age : ℕ := 8
  -- Father's current age
  let father_age : ℕ := 4 * son_age
  -- In x years, father's age will be 3 times son's age
  have h : father_age + x = 3 * (son_age + x) := by sorry
  sorry

end NUMINAMATH_CALUDE_father_son_age_problem_l1983_198329


namespace NUMINAMATH_CALUDE_quadratic_equation_completion_square_l1983_198381

theorem quadratic_equation_completion_square (x m n : ℝ) : 
  (9 * x^2 - 36 * x - 81 = 0) → 
  ((x + m)^2 = n) →
  (m + n = 11) := by
sorry

end NUMINAMATH_CALUDE_quadratic_equation_completion_square_l1983_198381


namespace NUMINAMATH_CALUDE_two_ladies_walk_l1983_198365

/-- The combined distance walked by two ladies in Central Park -/
def combined_distance (lady1_distance lady2_distance : ℝ) : ℝ :=
  lady1_distance + lady2_distance

/-- Theorem: The combined distance of two ladies is 12 miles when one walks twice as far as the other, and the second lady walks 4 miles -/
theorem two_ladies_walk :
  ∀ (lady1_distance lady2_distance : ℝ),
  lady2_distance = 4 →
  lady1_distance = 2 * lady2_distance →
  combined_distance lady1_distance lady2_distance = 12 :=
by
  sorry

end NUMINAMATH_CALUDE_two_ladies_walk_l1983_198365


namespace NUMINAMATH_CALUDE_coefficient_x_cubed_in_binomial_expansion_l1983_198359

theorem coefficient_x_cubed_in_binomial_expansion : 
  let n : ℕ := 5
  let a : ℝ := 1
  let b : ℝ := 2
  let r : ℕ := 3
  let coeff : ℝ := (n.choose r) * a^(n-r) * b^r
  coeff = 80 := by sorry

end NUMINAMATH_CALUDE_coefficient_x_cubed_in_binomial_expansion_l1983_198359


namespace NUMINAMATH_CALUDE_tangent_point_x_coordinate_l1983_198327

theorem tangent_point_x_coordinate 
  (f : ℝ → ℝ) 
  (h₁ : ∀ x, f x = x^2 + 1) 
  (h₂ : ∃ x, deriv f x = 4) : 
  ∃ x, deriv f x = 4 ∧ x = 2 := by
  sorry

end NUMINAMATH_CALUDE_tangent_point_x_coordinate_l1983_198327
