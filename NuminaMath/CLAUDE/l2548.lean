import Mathlib

namespace NUMINAMATH_CALUDE_bret_reading_time_l2548_254807

/-- The time Bret spends reading a book during a train ride -/
def time_reading_book (total_time dinner_time movie_time nap_time : ℕ) : ℕ :=
  total_time - (dinner_time + movie_time + nap_time)

/-- Theorem: Bret spends 2 hours reading a book during his train ride -/
theorem bret_reading_time :
  time_reading_book 9 1 3 3 = 2 := by
  sorry

end NUMINAMATH_CALUDE_bret_reading_time_l2548_254807


namespace NUMINAMATH_CALUDE_number_of_employees_l2548_254898

def gift_cost : ℕ := 100
def boss_contribution : ℕ := 15
def employee_contribution : ℕ := 11

theorem number_of_employees : 
  ∃ (n : ℕ), 
    gift_cost = boss_contribution + 2 * boss_contribution + n * employee_contribution ∧ 
    n = 5 := by
  sorry

end NUMINAMATH_CALUDE_number_of_employees_l2548_254898


namespace NUMINAMATH_CALUDE_students_with_dogs_and_amphibians_but_not_cats_l2548_254896

theorem students_with_dogs_and_amphibians_but_not_cats (total_students : ℕ) 
  (students_with_dogs : ℕ) (students_with_cats : ℕ) (students_with_amphibians : ℕ) 
  (students_without_pets : ℕ) :
  total_students = 40 →
  students_with_dogs = 24 →
  students_with_cats = 10 →
  students_with_amphibians = 8 →
  students_without_pets = 6 →
  (∃ (x y z : ℕ),
    x + y = students_with_dogs ∧
    y + z = students_with_amphibians ∧
    x + y + z = total_students - students_without_pets ∧
    y = 0) :=
by sorry

end NUMINAMATH_CALUDE_students_with_dogs_and_amphibians_but_not_cats_l2548_254896


namespace NUMINAMATH_CALUDE_pet_shop_dogs_count_l2548_254887

/-- Given a pet shop with dogs, cats, and bunnies in stock, this theorem proves
    the number of dogs based on the given ratio and total count of dogs and bunnies. -/
theorem pet_shop_dogs_count
  (ratio_dogs : ℕ)
  (ratio_cats : ℕ)
  (ratio_bunnies : ℕ)
  (total_dogs_and_bunnies : ℕ)
  (h_ratio : ratio_dogs = 3 ∧ ratio_cats = 5 ∧ ratio_bunnies = 9)
  (h_total : total_dogs_and_bunnies = 204) :
  (ratio_dogs * total_dogs_and_bunnies) / (ratio_dogs + ratio_bunnies) = 51 :=
by
  sorry


end NUMINAMATH_CALUDE_pet_shop_dogs_count_l2548_254887


namespace NUMINAMATH_CALUDE_optimal_usage_time_l2548_254856

/-- Profit function for the yacht (in ten thousand yuan) -/
def profit (x : ℕ+) : ℚ := -x^2 + 22*x - 49

/-- Average annual profit function -/
def avgProfit (x : ℕ+) : ℚ := profit x / x

/-- Theorem stating that 7 years maximizes the average annual profit -/
theorem optimal_usage_time :
  ∀ x : ℕ+, avgProfit 7 ≥ avgProfit x :=
sorry

end NUMINAMATH_CALUDE_optimal_usage_time_l2548_254856


namespace NUMINAMATH_CALUDE_sandy_shopping_money_l2548_254804

theorem sandy_shopping_money (remaining_amount : ℝ) (spent_percentage : ℝ) (initial_amount : ℝ) :
  remaining_amount = 224 ∧
  spent_percentage = 0.3 ∧
  remaining_amount = initial_amount * (1 - spent_percentage) →
  initial_amount = 320 :=
by sorry

end NUMINAMATH_CALUDE_sandy_shopping_money_l2548_254804


namespace NUMINAMATH_CALUDE_probability_of_condition_l2548_254873

def is_valid_pair (a b : ℕ) : Prop :=
  1 ≤ a ∧ a ≤ 60 ∧ 1 ≤ b ∧ b ≤ 60 ∧ a ≠ b

def satisfies_condition (a b : ℕ) : Prop :=
  ∃ (m : ℕ), 4 ∣ m ∧ a * b + a + b = m - 1

def total_valid_pairs : ℕ := Nat.choose 60 2

def satisfying_pairs : ℕ := 1350

theorem probability_of_condition :
  (satisfying_pairs : ℚ) / total_valid_pairs = 45 / 59 :=
sorry

end NUMINAMATH_CALUDE_probability_of_condition_l2548_254873


namespace NUMINAMATH_CALUDE_sin_negative_45_degrees_l2548_254808

theorem sin_negative_45_degrees : Real.sin (-(π / 4)) = -(Real.sqrt 2 / 2) := by
  sorry

end NUMINAMATH_CALUDE_sin_negative_45_degrees_l2548_254808


namespace NUMINAMATH_CALUDE_octagon_perimeter_96cm_l2548_254885

/-- A regular octagon is a polygon with 8 equal sides -/
structure RegularOctagon where
  side_length : ℝ
  
/-- The perimeter of a regular octagon -/
def perimeter (octagon : RegularOctagon) : ℝ :=
  8 * octagon.side_length

theorem octagon_perimeter_96cm :
  ∀ (octagon : RegularOctagon),
    octagon.side_length = 12 →
    perimeter octagon = 96 := by
  sorry

end NUMINAMATH_CALUDE_octagon_perimeter_96cm_l2548_254885


namespace NUMINAMATH_CALUDE_decimal_expansion_prime_modulo_l2548_254823

theorem decimal_expansion_prime_modulo
  (p : ℕ) (r : ℕ) (hp : p.Prime) (hp_gt_5 : p > 5)
  (hr : ∃ (a : ℕ → ℕ), (∀ i, a i < 10) ∧
    (1 : ℚ) / p = ∑' i, (a i : ℚ) / (10 ^ (i + 1)) - ∑' i, (a (i % r) : ℚ) / (10 ^ (i + r + 1)))
  : 10 ^ r ≡ 1 [MOD p] :=
sorry

end NUMINAMATH_CALUDE_decimal_expansion_prime_modulo_l2548_254823


namespace NUMINAMATH_CALUDE_division_multiplication_equality_l2548_254886

theorem division_multiplication_equality (a b : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) :
  1 / (b / a) * (a / b) = a^2 / b^2 := by
  sorry

end NUMINAMATH_CALUDE_division_multiplication_equality_l2548_254886


namespace NUMINAMATH_CALUDE_second_chapter_length_l2548_254899

theorem second_chapter_length (total_pages first_chapter_pages : ℕ) 
  (h1 : total_pages = 94)
  (h2 : first_chapter_pages = 48) :
  total_pages - first_chapter_pages = 46 := by
  sorry

end NUMINAMATH_CALUDE_second_chapter_length_l2548_254899


namespace NUMINAMATH_CALUDE_x_plus_y_value_l2548_254812

theorem x_plus_y_value (x y : ℝ) 
  (h1 : x + Real.sin y = 2010)
  (h2 : x + 2010 * Real.cos y = 2009)
  (h3 : 0 ≤ y ∧ y ≤ Real.pi / 2) :
  x + y = 2009 + Real.pi / 2 := by
sorry

end NUMINAMATH_CALUDE_x_plus_y_value_l2548_254812


namespace NUMINAMATH_CALUDE_fraction_equality_l2548_254814

theorem fraction_equality (a b : ℝ) (h : a ≠ 0) : b / a = (a * b) / (a * a) := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_l2548_254814


namespace NUMINAMATH_CALUDE_f_properties_l2548_254860

def f (x : ℝ) : ℝ := x^3 - 6*x + 5

theorem f_properties :
  let f := f
  ∀ x y a : ℝ,
  (x < -Real.sqrt 2 ∧ y > -Real.sqrt 2 ∧ y < Real.sqrt 2 → f x < f y) ∧
  (x > Real.sqrt 2 ∧ y > -Real.sqrt 2 ∧ y < Real.sqrt 2 → f x < f y) ∧
  (f (-Real.sqrt 2) = 5 + 4 * Real.sqrt 2) ∧
  (f (Real.sqrt 2) = 5 - 4 * Real.sqrt 2) ∧
  (∃ x₁ x₂ x₃ : ℝ, x₁ ≠ x₂ ∧ x₁ ≠ x₃ ∧ x₂ ≠ x₃ ∧ f x₁ = a ∧ f x₂ = a ∧ f x₃ = a
    ↔ 5 - 4 * Real.sqrt 2 < a ∧ a < 5 + 4 * Real.sqrt 2) :=
by
  sorry

end NUMINAMATH_CALUDE_f_properties_l2548_254860


namespace NUMINAMATH_CALUDE_increase_by_percentage_l2548_254802

/-- Prove that increasing 500 by 30% results in 650. -/
theorem increase_by_percentage (initial : ℝ) (percentage : ℝ) (result : ℝ) : 
  initial = 500 → percentage = 30 → result = initial * (1 + percentage / 100) → result = 650 := by
  sorry

end NUMINAMATH_CALUDE_increase_by_percentage_l2548_254802


namespace NUMINAMATH_CALUDE_isosceles_triangle_perimeter_l2548_254859

/-- A triangle with two equal sides and side lengths of 3 and 5 has a perimeter of either 11 or 13 -/
theorem isosceles_triangle_perimeter : ∀ (a b : ℝ), 
  a = 3 ∧ b = 5 →
  (∃ (p : ℝ), (p = 11 ∨ p = 13) ∧ 
   ((2 * a + b = p ∧ a + a > b) ∨ (2 * b + a = p ∧ b + b > a))) :=
by sorry


end NUMINAMATH_CALUDE_isosceles_triangle_perimeter_l2548_254859


namespace NUMINAMATH_CALUDE_impossible_triangle_angles_l2548_254866

-- Define a triangle
structure Triangle where
  -- We don't need to specify the actual properties of a triangle here

-- Define the sum of interior angles of a triangle
def sum_of_interior_angles (t : Triangle) : ℝ := 180

-- Theorem: It is impossible for the sum of the interior angles of a triangle to be 360°
theorem impossible_triangle_angles (t : Triangle) : sum_of_interior_angles t ≠ 360 := by
  sorry

end NUMINAMATH_CALUDE_impossible_triangle_angles_l2548_254866


namespace NUMINAMATH_CALUDE_arithmetic_geometric_ratio_l2548_254813

-- Define an arithmetic sequence
def arithmetic_sequence (a₁ d : ℝ) (n : ℕ) : ℝ := a₁ + (n - 1) * d

-- Define the problem
theorem arithmetic_geometric_ratio
  (d : ℝ) (h_d : d ≠ 0)
  (h_geom : ∃ a₁ : ℝ, (arithmetic_sequence a₁ d 3)^2 = 
    (arithmetic_sequence a₁ d 2) * (arithmetic_sequence a₁ d 9)) :
  ∃ a₁ : ℝ, (arithmetic_sequence a₁ d 2 + arithmetic_sequence a₁ d 3 + arithmetic_sequence a₁ d 4) /
            (arithmetic_sequence a₁ d 4 + arithmetic_sequence a₁ d 5 + arithmetic_sequence a₁ d 6) = 3/8 :=
by sorry

end NUMINAMATH_CALUDE_arithmetic_geometric_ratio_l2548_254813


namespace NUMINAMATH_CALUDE_elevation_equals_depression_l2548_254806

/-- The elevation angle from point a to point b -/
def elevation_angle (a b : Point) : ℝ := sorry

/-- The depression angle from point b to point a -/
def depression_angle (b a : Point) : ℝ := sorry

/-- Theorem stating that the elevation angle from a to b equals the depression angle from b to a -/
theorem elevation_equals_depression (a b : Point) :
  elevation_angle a b = depression_angle b a := by sorry

end NUMINAMATH_CALUDE_elevation_equals_depression_l2548_254806


namespace NUMINAMATH_CALUDE_triangle_is_obtuse_l2548_254865

theorem triangle_is_obtuse (A : Real) (h1 : 0 < A ∧ A < π) 
  (h2 : Real.sin A + Real.cos A = 7 / 12) : π / 2 < A ∧ A < π := by
  sorry

end NUMINAMATH_CALUDE_triangle_is_obtuse_l2548_254865


namespace NUMINAMATH_CALUDE_triangle_ratio_theorem_l2548_254805

/-- Given a triangle ABC with side lengths a, b, c, altitudes ha, hb, hc, and circumradius R,
    the ratio of the sum of pairwise products of side lengths to the sum of altitudes
    is equal to the diameter of the circumscribed circle. -/
theorem triangle_ratio_theorem (a b c ha hb hc R : ℝ) :
  a > 0 → b > 0 → c > 0 → ha > 0 → hb > 0 → hc > 0 → R > 0 →
  (a * b + b * c + a * c) / (ha + hb + hc) = 2 * R := by
  sorry

end NUMINAMATH_CALUDE_triangle_ratio_theorem_l2548_254805


namespace NUMINAMATH_CALUDE_min_value_expression_l2548_254888

theorem min_value_expression (c d : ℝ) (hc : c ≠ 0) (hd : d ≠ 0) :
  ∃ (m : ℝ), (∀ c d : ℝ, c ≠ 0 → d ≠ 0 → c^2 + d^2 + 4/c^2 + 2*d/c ≥ m) ∧
  (∃ c d : ℝ, c ≠ 0 ∧ d ≠ 0 ∧ c^2 + d^2 + 4/c^2 + 2*d/c = m) ∧
  m = 2 * Real.sqrt 3 :=
sorry

end NUMINAMATH_CALUDE_min_value_expression_l2548_254888


namespace NUMINAMATH_CALUDE_incircle_tangent_concurrency_l2548_254809

-- Define the types for points and circles
variable (Point Circle : Type)

-- Define the necessary geometric concepts
variable (is_convex_quadrilateral : Point → Point → Point → Point → Prop)
variable (is_incircle : Circle → Point → Point → Point → Prop)
variable (center_of : Circle → Point)
variable (second_common_external_tangent_touches : Circle → Circle → Point → Point → Prop)
variable (line_through : Point → Point → Set Point)
variable (concurrent : Set Point → Set Point → Set Point → Prop)

-- State the theorem
theorem incircle_tangent_concurrency 
  (A B C D : Point) 
  (ωA ωB : Circle) 
  (I J K L : Point) :
  is_convex_quadrilateral A B C D →
  is_incircle ωA A C D →
  is_incircle ωB B C D →
  I = center_of ωA →
  J = center_of ωB →
  second_common_external_tangent_touches ωA ωB K L →
  concurrent (line_through A K) (line_through B L) (line_through I J) :=
by sorry

end NUMINAMATH_CALUDE_incircle_tangent_concurrency_l2548_254809


namespace NUMINAMATH_CALUDE_max_subjects_per_teacher_l2548_254829

/-- Proves that the maximum number of subjects a teacher can teach is 4 -/
theorem max_subjects_per_teacher (total_subjects : ℕ) (min_teachers : ℕ) : 
  total_subjects = 16 → min_teachers = 4 → (total_subjects / min_teachers : ℕ) = 4 := by
  sorry

end NUMINAMATH_CALUDE_max_subjects_per_teacher_l2548_254829


namespace NUMINAMATH_CALUDE_mikes_books_l2548_254842

/-- Calculates the final number of books Mike has after selling, receiving gifts, and buying books. -/
def final_book_count (initial : ℝ) (sold : ℝ) (gifts : ℝ) (bought : ℝ) : ℝ :=
  initial - sold + gifts + bought

/-- Theorem stating that Mike's final book count is 21.5 given the problem conditions. -/
theorem mikes_books :
  final_book_count 51.5 45.75 12.25 3.5 = 21.5 := by
  sorry

end NUMINAMATH_CALUDE_mikes_books_l2548_254842


namespace NUMINAMATH_CALUDE_rational_solutions_quadratic_l2548_254828

theorem rational_solutions_quadratic (k : ℕ+) : 
  (∃ x : ℚ, k * x^2 + 24 * x + k = 0) ↔ (k = 8 ∨ k = 12) := by
sorry

end NUMINAMATH_CALUDE_rational_solutions_quadratic_l2548_254828


namespace NUMINAMATH_CALUDE_fraction_addition_l2548_254869

theorem fraction_addition : (3 : ℚ) / 5 + (2 : ℚ) / 5 = 1 := by sorry

end NUMINAMATH_CALUDE_fraction_addition_l2548_254869


namespace NUMINAMATH_CALUDE_tissue_length_l2548_254862

/-- The total length of overlapped tissue pieces. -/
def totalLength (n : ℕ) (pieceLength : ℝ) (overlap : ℝ) : ℝ :=
  pieceLength + (n - 1 : ℝ) * (pieceLength - overlap)

/-- Theorem stating the total length of 30 pieces of tissue, each 25 cm long,
    overlapped by 6 cm, is 576 cm. -/
theorem tissue_length :
  totalLength 30 25 6 = 576 := by
  sorry

end NUMINAMATH_CALUDE_tissue_length_l2548_254862


namespace NUMINAMATH_CALUDE_a_equals_one_sufficient_not_necessary_l2548_254848

-- Define the equation
def circle_equation (x y a : ℝ) : Prop :=
  x^2 + y^2 - 2*x + 2*y + a = 0

-- Define what it means for the equation to represent a circle
def represents_circle (a : ℝ) : Prop :=
  ∃ (h k r : ℝ), r > 0 ∧ ∀ (x y : ℝ), circle_equation x y a ↔ (x - h)^2 + (y - k)^2 = r^2

-- Theorem statement
theorem a_equals_one_sufficient_not_necessary :
  (represents_circle 1) ∧
  (∃ (a : ℝ), a ≠ 1 ∧ represents_circle a) :=
sorry

end NUMINAMATH_CALUDE_a_equals_one_sufficient_not_necessary_l2548_254848


namespace NUMINAMATH_CALUDE_fifth_power_sum_l2548_254803

theorem fifth_power_sum (a b x y : ℝ) 
  (h1 : a * x + b * y = 4)
  (h2 : a * x^2 + b * y^2 = 9)
  (h3 : a * x^3 + b * y^3 = 23)
  (h4 : a * x^4 + b * y^4 = 54) :
  a * x^5 + b * y^5 = 470 := by
  sorry


end NUMINAMATH_CALUDE_fifth_power_sum_l2548_254803


namespace NUMINAMATH_CALUDE_complex_sum_representation_l2548_254800

theorem complex_sum_representation : ∃ (r θ : ℝ), 
  15 * Complex.exp (Complex.I * (π / 7)) + 15 * Complex.exp (Complex.I * (9 * π / 14)) = r * Complex.exp (Complex.I * θ) ∧ 
  r = 15 * Real.sqrt 2 ∧ 
  θ = 11 * π / 28 := by
  sorry

end NUMINAMATH_CALUDE_complex_sum_representation_l2548_254800


namespace NUMINAMATH_CALUDE_probability_at_least_one_suit_each_l2548_254854

-- Define a standard deck of cards
def StandardDeck : Type := Unit

-- Define the number of suits in a standard deck
def NumberOfSuits : ℕ := 4

-- Define the number of cards drawn
def NumberOfDraws : ℕ := 5

-- Define the probability of drawing a card from a specific suit
def ProbabilityOfSuit : ℚ := 1 / 4

-- Define the probability of drawing at least one card from each suit
def ProbabilityAtLeastOneSuitEach : ℚ := 3 / 32

-- State the theorem
theorem probability_at_least_one_suit_each (deck : StandardDeck) :
  ProbabilityAtLeastOneSuitEach = 3 / 32 := by
  sorry

end NUMINAMATH_CALUDE_probability_at_least_one_suit_each_l2548_254854


namespace NUMINAMATH_CALUDE_cube_frame_problem_solution_l2548_254825

/-- Represents the cube frame construction problem. -/
structure CubeFrameProblem where
  bonnie_wire_length : ℕ        -- Length of each wire piece Bonnie uses
  bonnie_wire_count : ℕ         -- Number of wire pieces Bonnie uses
  roark_wire_length : ℕ         -- Length of each wire piece Roark uses
  roark_cube_edge_length : ℕ    -- Edge length of Roark's unit cubes

/-- The solution to the cube frame problem. -/
def cubeProblemSolution (p : CubeFrameProblem) : ℚ :=
  let bonnie_total_length := p.bonnie_wire_length * p.bonnie_wire_count
  let bonnie_cube_volume := p.bonnie_wire_length ^ 3
  let roark_cube_count := bonnie_cube_volume
  let roark_wire_per_cube := 12 * p.roark_wire_length
  let roark_total_length := roark_cube_count * roark_wire_per_cube
  bonnie_total_length / roark_total_length

/-- Theorem stating the solution to the cube frame problem. -/
theorem cube_frame_problem_solution (p : CubeFrameProblem) 
  (h1 : p.bonnie_wire_length = 8)
  (h2 : p.bonnie_wire_count = 12)
  (h3 : p.roark_wire_length = 2)
  (h4 : p.roark_cube_edge_length = 1) :
  cubeProblemSolution p = 1 / 128 := by
  sorry

end NUMINAMATH_CALUDE_cube_frame_problem_solution_l2548_254825


namespace NUMINAMATH_CALUDE_roundness_of_1764000_l2548_254864

/-- The roundness of a positive integer is the sum of the exponents in its prime factorization. -/
def roundness (n : ℕ+) : ℕ := sorry

/-- Theorem: The roundness of 1,764,000 is 11. -/
theorem roundness_of_1764000 : roundness 1764000 = 11 := by sorry

end NUMINAMATH_CALUDE_roundness_of_1764000_l2548_254864


namespace NUMINAMATH_CALUDE_min_triangle_area_l2548_254895

/-- An acute-angled triangle with an inscribed square --/
structure TriangleWithSquare where
  /-- The base length of the triangle --/
  b : ℝ
  /-- The height of the triangle --/
  h : ℝ
  /-- The side length of the inscribed square --/
  s : ℝ
  /-- The triangle is acute-angled --/
  acute : 0 < b ∧ 0 < h
  /-- The square is inscribed as described --/
  square_inscribed : s = (b * h) / (b + h)

/-- The theorem stating the minimum area of the triangle --/
theorem min_triangle_area (t : TriangleWithSquare) (h_area : t.s^2 = 2017) :
  2 * t.s^2 ≤ (t.b * t.h) / 2 ∧
  ∃ (t' : TriangleWithSquare), t'.s^2 = 2017 ∧ (t'.b * t'.h) / 2 = 2 * t'.s^2 := by
  sorry

#check min_triangle_area

end NUMINAMATH_CALUDE_min_triangle_area_l2548_254895


namespace NUMINAMATH_CALUDE_fourth_number_proof_l2548_254889

theorem fourth_number_proof (x : ℝ) : 
  (3.6 * 0.48 * 2.50) / (x * 0.09 * 0.5) = 800.0000000000001 → x = 0.3 := by
  sorry

end NUMINAMATH_CALUDE_fourth_number_proof_l2548_254889


namespace NUMINAMATH_CALUDE_sin_symmetry_condition_l2548_254844

def is_symmetric_about_y_axis (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f x = f (-x)

theorem sin_symmetry_condition (φ : ℝ) :
  (φ = π / 2 → is_symmetric_about_y_axis (fun x ↦ Real.sin (x + φ))) ∧
  ¬(is_symmetric_about_y_axis (fun x ↦ Real.sin (x + φ)) → φ = π / 2) :=
by sorry

end NUMINAMATH_CALUDE_sin_symmetry_condition_l2548_254844


namespace NUMINAMATH_CALUDE_existence_of_abc_l2548_254852

theorem existence_of_abc (n : ℕ) : ∃ (a b c : ℤ),
  n = Int.gcd a b * (c^2 - a*b) + Int.gcd b c * (a^2 - b*c) + Int.gcd c a * (b^2 - c*a) := by
  sorry

end NUMINAMATH_CALUDE_existence_of_abc_l2548_254852


namespace NUMINAMATH_CALUDE_jelly_ratio_l2548_254883

def jelly_problem (grape strawberry raspberry plum : ℕ) : Prop :=
  grape = 2 * strawberry ∧
  raspberry = 2 * plum ∧
  plum = 6 ∧
  strawberry = 18 ∧
  raspberry * 3 = grape

theorem jelly_ratio :
  ∀ grape strawberry raspberry plum : ℕ,
  jelly_problem grape strawberry raspberry plum →
  raspberry * 3 = grape :=
by
  sorry

end NUMINAMATH_CALUDE_jelly_ratio_l2548_254883


namespace NUMINAMATH_CALUDE_minimum_bottles_needed_l2548_254818

/-- The capacity of a medium-sized bottle in milliliters -/
def medium_bottle_capacity : ℕ := 80

/-- The capacity of a very large bottle in milliliters -/
def large_bottle_capacity : ℕ := 1200

/-- The maximum number of additional bottles allowed -/
def max_additional_bottles : ℕ := 5

/-- The minimum number of medium-sized bottles needed -/
def min_bottles_needed : ℕ := 15

theorem minimum_bottles_needed :
  (large_bottle_capacity / medium_bottle_capacity = min_bottles_needed) ∧
  (min_bottles_needed + max_additional_bottles ≥ 
   (large_bottle_capacity + medium_bottle_capacity - 1) / medium_bottle_capacity) :=
sorry

end NUMINAMATH_CALUDE_minimum_bottles_needed_l2548_254818


namespace NUMINAMATH_CALUDE_questionnaire_C_count_l2548_254833

/-- Represents the total population size -/
def population_size : ℕ := 1000

/-- Represents the sample size -/
def sample_size : ℕ := 50

/-- Represents the first number drawn in the systematic sample -/
def first_number : ℕ := 8

/-- Represents the lower bound of the interval for questionnaire C -/
def lower_bound : ℕ := 751

/-- Represents the upper bound of the interval for questionnaire C -/
def upper_bound : ℕ := 1000

/-- Theorem stating that the number of people taking questionnaire C is 12 -/
theorem questionnaire_C_count :
  (Finset.filter (fun n => lower_bound ≤ (first_number + (n - 1) * (population_size / sample_size)) ∧
                           (first_number + (n - 1) * (population_size / sample_size)) ≤ upper_bound)
                 (Finset.range sample_size)).card = 12 :=
by sorry

end NUMINAMATH_CALUDE_questionnaire_C_count_l2548_254833


namespace NUMINAMATH_CALUDE_time_to_find_artifacts_is_120_months_l2548_254863

/-- The time taken to find two artifacts given research and expedition times for the first artifact,
    and a multiplier for the second artifact's time. -/
def time_to_find_artifacts (research_time_1 : ℕ) (expedition_time_1 : ℕ) (multiplier : ℕ) : ℕ :=
  let first_artifact_time := research_time_1 + expedition_time_1
  let second_artifact_time := multiplier * first_artifact_time
  first_artifact_time + second_artifact_time

/-- Theorem stating that the time to find both artifacts is 120 months. -/
theorem time_to_find_artifacts_is_120_months :
  time_to_find_artifacts 6 24 3 = 120 := by
  sorry

end NUMINAMATH_CALUDE_time_to_find_artifacts_is_120_months_l2548_254863


namespace NUMINAMATH_CALUDE_tomato_plants_problem_l2548_254892

theorem tomato_plants_problem (plant1 plant2 plant3 plant4 : ℕ) : 
  plant1 = 8 →
  plant2 = plant1 + 4 →
  plant3 = 3 * (plant1 + plant2) →
  plant4 = 3 * (plant1 + plant2) →
  plant1 + plant2 + plant3 + plant4 = 140 →
  plant2 - plant1 = 4 := by
sorry

end NUMINAMATH_CALUDE_tomato_plants_problem_l2548_254892


namespace NUMINAMATH_CALUDE_water_and_milk_amounts_l2548_254876

/-- Sarah's special bread recipe -/
def special_bread_recipe (flour water milk : ℚ) : Prop :=
  water / flour = 75 / 300 ∧ milk / flour = 60 / 300

/-- The amount of flour Sarah uses -/
def flour_amount : ℚ := 900

/-- The theorem stating the required amounts of water and milk -/
theorem water_and_milk_amounts :
  ∀ water milk : ℚ,
  special_bread_recipe flour_amount water milk →
  water = 225 ∧ milk = 180 := by sorry

end NUMINAMATH_CALUDE_water_and_milk_amounts_l2548_254876


namespace NUMINAMATH_CALUDE_madeline_money_problem_l2548_254853

/-- Madeline's money problem -/
theorem madeline_money_problem (madeline_money : ℝ) (brother_money : ℝ) : 
  brother_money = (1/2) * madeline_money →
  madeline_money + brother_money = 72 →
  madeline_money = 48 := by
sorry

end NUMINAMATH_CALUDE_madeline_money_problem_l2548_254853


namespace NUMINAMATH_CALUDE_parallelogram_df_and_area_l2548_254810

/-- Represents a parallelogram ABCD with altitudes DE and DF -/
structure Parallelogram where
  -- Length of side DC
  dc : ℝ
  -- Length of EB (part of base AB)
  eb : ℝ
  -- Length of altitude DE
  de : ℝ
  -- Assumption that ABCD is a parallelogram
  is_parallelogram : True

/-- Properties of the parallelogram -/
def parallelogram_properties (p : Parallelogram) : Prop :=
  p.dc = 15 ∧ p.eb = 3 ∧ p.de = 5

/-- Theorem about the length of DF and the area of the parallelogram -/
theorem parallelogram_df_and_area (p : Parallelogram) 
  (h : parallelogram_properties p) :
  ∃ (df area : ℝ), df = 5 ∧ area = 75 := by
  sorry


end NUMINAMATH_CALUDE_parallelogram_df_and_area_l2548_254810


namespace NUMINAMATH_CALUDE_y_decreases_as_x_increases_l2548_254870

/-- A linear function y = -2x - 3 -/
def f (x : ℝ) : ℝ := -2 * x - 3

/-- Theorem: For any two points on the graph of f, if the x-coordinate of the first point
    is less than the x-coordinate of the second point, then the y-coordinate of the first point
    is greater than the y-coordinate of the second point. -/
theorem y_decreases_as_x_increases (x₁ x₂ y₁ y₂ : ℝ) 
  (h1 : f x₁ = y₁) 
  (h2 : f x₂ = y₂) 
  (h3 : x₁ < x₂) : 
  y₁ > y₂ := by
  sorry

end NUMINAMATH_CALUDE_y_decreases_as_x_increases_l2548_254870


namespace NUMINAMATH_CALUDE_trigonometric_identities_l2548_254861

theorem trigonometric_identities (α : ℝ) (h : 3 * Real.sin α - 2 * Real.cos α = 0) :
  (((Real.cos α - Real.sin α) / (Real.cos α + Real.sin α)) + 
   ((Real.cos α + Real.sin α) / (Real.cos α - Real.sin α)) = 5 / 1) ∧
  (Real.sin α ^ 2 - 2 * Real.sin α * Real.cos α + 4 * Real.cos α ^ 2 = 28 / 13) := by
sorry

end NUMINAMATH_CALUDE_trigonometric_identities_l2548_254861


namespace NUMINAMATH_CALUDE_existence_of_n_consecutive_with_one_prime_l2548_254821

theorem existence_of_n_consecutive_with_one_prime (n : ℕ) : 
  ∃ k : ℕ, ∃! i : Fin n, Nat.Prime ((k : ℕ) + i) := by
  sorry

end NUMINAMATH_CALUDE_existence_of_n_consecutive_with_one_prime_l2548_254821


namespace NUMINAMATH_CALUDE_jessica_rearrangements_time_l2548_254894

def name_length : ℕ := 7
def repeated_s : ℕ := 2
def repeated_a : ℕ := 2
def rearrangements_per_minute : ℕ := 15

def total_rearrangements : ℕ := name_length.factorial / (repeated_s.factorial * repeated_a.factorial)

theorem jessica_rearrangements_time :
  (total_rearrangements : ℚ) / rearrangements_per_minute / 60 = 1.4 := by
  sorry

end NUMINAMATH_CALUDE_jessica_rearrangements_time_l2548_254894


namespace NUMINAMATH_CALUDE_coffee_purchase_proof_l2548_254838

/-- Given a gift card balance, coffee price per pound, and remaining balance,
    calculate the number of pounds of coffee purchased. -/
def coffee_purchased (gift_card_balance : ℚ) (coffee_price : ℚ) (remaining_balance : ℚ) : ℚ :=
  (gift_card_balance - remaining_balance) / coffee_price

/-- Prove that given the specified conditions, the amount of coffee purchased is 4 pounds. -/
theorem coffee_purchase_proof (gift_card_balance : ℚ) (coffee_price : ℚ) (remaining_balance : ℚ)
  (h1 : gift_card_balance = 70)
  (h2 : coffee_price = 8.58)
  (h3 : remaining_balance = 35.68) :
  coffee_purchased gift_card_balance coffee_price remaining_balance = 4 := by
  sorry

end NUMINAMATH_CALUDE_coffee_purchase_proof_l2548_254838


namespace NUMINAMATH_CALUDE_money_division_l2548_254846

/-- Proves that given a sum of money divided between two people x and y in the ratio 2:8,
    where x receives $1000, the total amount of money is $5000. -/
theorem money_division (x y total : ℕ) : 
  x + y = total → 
  x = 1000 → 
  2 * total = 10 * x → 
  total = 5000 := by
sorry

end NUMINAMATH_CALUDE_money_division_l2548_254846


namespace NUMINAMATH_CALUDE_llama_breeding_problem_llama_breeding_solution_l2548_254858

theorem llama_breeding_problem (pregnant_llamas : ℕ) (twin_pregnancies : ℕ) 
  (traded_calves : ℕ) (new_adults : ℕ) (final_herd : ℕ) : ℕ :=
  let single_pregnancies := pregnant_llamas - twin_pregnancies
  let total_calves := twin_pregnancies * 2 + single_pregnancies * 1
  let remaining_calves := total_calves - traded_calves
  let pre_sale_herd := final_herd / (2/3)
  let pre_sale_adults := pre_sale_herd - remaining_calves - new_adults
  let original_adults := pre_sale_adults - new_adults
  total_calves

theorem llama_breeding_solution : 
  llama_breeding_problem 9 5 8 2 18 = 14 := by sorry

end NUMINAMATH_CALUDE_llama_breeding_problem_llama_breeding_solution_l2548_254858


namespace NUMINAMATH_CALUDE_candy_distribution_l2548_254884

theorem candy_distribution (total_candy : ℕ) (total_bags : ℕ) (chocolate_hearts_bags : ℕ) (chocolate_kisses_bags : ℕ) 
  (h1 : total_candy = 63)
  (h2 : total_bags = 9)
  (h3 : chocolate_hearts_bags = 2)
  (h4 : chocolate_kisses_bags = 3)
  (h5 : total_candy % total_bags = 0) :
  let candy_per_bag := total_candy / total_bags
  let chocolate_bags := chocolate_hearts_bags + chocolate_kisses_bags
  let non_chocolate_bags := total_bags - chocolate_bags
  non_chocolate_bags * candy_per_bag = 28 := by
sorry


end NUMINAMATH_CALUDE_candy_distribution_l2548_254884


namespace NUMINAMATH_CALUDE_complex_equation_solution_l2548_254882

theorem complex_equation_solution :
  ∀ a : ℂ, (1 - I)^3 / (1 + I) = a + 3*I → a = -2 := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l2548_254882


namespace NUMINAMATH_CALUDE_circle_intersection_symmetry_l2548_254881

-- Define a circle type
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

-- Define the problem setup
def problem_setup (c1 c2 : Circle) (A B : ℝ × ℝ) : Prop :=
  -- The circles intersect at points A and B
  A ∈ {p : ℝ × ℝ | (p.1 - c1.center.1)^2 + (p.2 - c1.center.2)^2 = c1.radius^2} ∧
  A ∈ {p : ℝ × ℝ | (p.1 - c2.center.1)^2 + (p.2 - c2.center.2)^2 = c2.radius^2} ∧
  B ∈ {p : ℝ × ℝ | (p.1 - c1.center.1)^2 + (p.2 - c1.center.2)^2 = c1.radius^2} ∧
  B ∈ {p : ℝ × ℝ | (p.1 - c2.center.1)^2 + (p.2 - c2.center.2)^2 = c2.radius^2} ∧
  -- Centers of both circles are on the x-axis
  c1.center.2 = 0 ∧
  c2.center.2 = 0 ∧
  -- Coordinates of point A are (-3, 2)
  A = (-3, 2)

-- Theorem statement
theorem circle_intersection_symmetry (c1 c2 : Circle) (A B : ℝ × ℝ) :
  problem_setup c1 c2 A B → B = (-3, -2) :=
by
  sorry

end NUMINAMATH_CALUDE_circle_intersection_symmetry_l2548_254881


namespace NUMINAMATH_CALUDE_m_equality_l2548_254891

theorem m_equality (M : ℕ) (h : M^2 = 16^81 * 81^16) : M = 6^64 * 2^260 := by
  sorry

end NUMINAMATH_CALUDE_m_equality_l2548_254891


namespace NUMINAMATH_CALUDE_quadratic_two_distinct_zeros_l2548_254855

/-- A quadratic function of the form y = ax^2 + 4x - 2 has two distinct zeros if and only if a > -2 and a ≠ 0 -/
theorem quadratic_two_distinct_zeros (a : ℝ) :
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ a * x₁^2 + 4 * x₁ - 2 = 0 ∧ a * x₂^2 + 4 * x₂ - 2 = 0) ↔
  (a > -2 ∧ a ≠ 0) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_two_distinct_zeros_l2548_254855


namespace NUMINAMATH_CALUDE_triangle_cosine_l2548_254851

theorem triangle_cosine (A B C : Real) :
  -- Triangle conditions
  A + B + C = Real.pi →
  -- Given conditions
  Real.sin A = 3 / 5 →
  Real.cos B = 5 / 13 →
  -- Conclusion
  Real.cos C = 16 / 65 := by
  sorry

end NUMINAMATH_CALUDE_triangle_cosine_l2548_254851


namespace NUMINAMATH_CALUDE_final_diaries_count_l2548_254897

def calculate_final_diaries (initial : ℕ) : ℕ :=
  let after_buying := initial + 3 * initial
  let lost := (3 * after_buying) / 5
  after_buying - lost

theorem final_diaries_count : calculate_final_diaries 15 = 24 := by
  sorry

end NUMINAMATH_CALUDE_final_diaries_count_l2548_254897


namespace NUMINAMATH_CALUDE_tan_alpha_value_l2548_254875

theorem tan_alpha_value (α : Real) (h1 : α ∈ Set.Ioo 0 (π / 2)) 
  (h2 : Real.tan (2 * α) = Real.cos α / (2 - Real.sin α)) : 
  Real.tan α = Real.sqrt 15 / 15 := by
  sorry

end NUMINAMATH_CALUDE_tan_alpha_value_l2548_254875


namespace NUMINAMATH_CALUDE_age_ratio_future_l2548_254826

/-- Given Alan's current age a and Bella's current age b, prove that the number of years
    until their age ratio is 3:2 is 7, given the conditions on their past ages. -/
theorem age_ratio_future (a b : ℕ) (h1 : a - 3 = 2 * (b - 3)) (h2 : a - 8 = 3 * (b - 8)) :
  ∃ x : ℕ, x = 7 ∧ (a + x) * 2 = (b + x) * 3 :=
sorry

end NUMINAMATH_CALUDE_age_ratio_future_l2548_254826


namespace NUMINAMATH_CALUDE_blue_stamp_price_l2548_254857

/-- Given a collection of stamps and their prices, prove the price of blue stamps --/
theorem blue_stamp_price
  (red_count : ℕ)
  (blue_count : ℕ)
  (yellow_count : ℕ)
  (red_price : ℚ)
  (yellow_price : ℚ)
  (total_earnings : ℚ)
  (h1 : red_count = 20)
  (h2 : blue_count = 80)
  (h3 : yellow_count = 7)
  (h4 : red_price = 11/10)
  (h5 : yellow_price = 2)
  (h6 : total_earnings = 100) :
  (total_earnings - red_count * red_price - yellow_count * yellow_price) / blue_count = 4/5 :=
by sorry

end NUMINAMATH_CALUDE_blue_stamp_price_l2548_254857


namespace NUMINAMATH_CALUDE_quadrilateral_diagonal_lengths_l2548_254847

/-- A quadrilateral with side lengths 7, 9, 15, and 10 has 10 possible whole number lengths for its diagonal. -/
theorem quadrilateral_diagonal_lengths :
  ∃ (lengths : Finset ℕ),
    (Finset.card lengths = 10) ∧
    (∀ x ∈ lengths,
      (x + 7 > 9) ∧ (x + 9 > 7) ∧ (7 + 9 > x) ∧
      (x + 10 > 15) ∧ (x + 15 > 10) ∧ (10 + 15 > x) ∧
      (x ≥ 6) ∧ (x ≤ 15)) :=
by sorry

end NUMINAMATH_CALUDE_quadrilateral_diagonal_lengths_l2548_254847


namespace NUMINAMATH_CALUDE_figure_to_square_possible_l2548_254879

/-- Represents a point on a grid --/
structure GridPoint where
  x : ℤ
  y : ℤ

/-- Represents a triangle on a grid --/
structure GridTriangle where
  p1 : GridPoint
  p2 : GridPoint
  p3 : GridPoint

/-- Represents a figure on a grid --/
structure GridFigure where
  points : List GridPoint

/-- Function to calculate the area of a grid figure --/
def calculateArea (figure : GridFigure) : ℕ :=
  sorry

/-- Function to check if a list of triangles forms a square --/
def formsSquare (triangles : List GridTriangle) : Prop :=
  sorry

/-- The main theorem --/
theorem figure_to_square_possible (figure : GridFigure) : 
  ∃ (triangles : List GridTriangle), 
    triangles.length = 5 ∧ 
    (calculateArea figure = calculateArea (GridFigure.mk (triangles.bind (λ t => [t.p1, t.p2, t.p3]))) ∧
    formsSquare triangles) :=
  sorry

end NUMINAMATH_CALUDE_figure_to_square_possible_l2548_254879


namespace NUMINAMATH_CALUDE_sum_of_common_ratios_is_three_l2548_254827

/-- Given two geometric sequences with different common ratios s and t, 
    both starting with term m, if m s^2 - m t^2 = 3(m s - m t), then s + t = 3 -/
theorem sum_of_common_ratios_is_three 
  (m : ℝ) (s t : ℝ) (h_diff : s ≠ t) (h_m_nonzero : m ≠ 0) 
  (h_eq : m * s^2 - m * t^2 = 3 * (m * s - m * t)) : 
  s + t = 3 := by
sorry

end NUMINAMATH_CALUDE_sum_of_common_ratios_is_three_l2548_254827


namespace NUMINAMATH_CALUDE_max_min_values_of_f_l2548_254836

def f (x : ℝ) : ℝ := -x^2 + 4*x + 5

theorem max_min_values_of_f :
  let a : ℝ := 1
  let b : ℝ := 4
  (∀ x ∈ Set.Icc a b, f x ≤ 9) ∧
  (∃ x ∈ Set.Icc a b, f x = 9) ∧
  (∀ x ∈ Set.Icc a b, f x ≥ 5) ∧
  (∃ x ∈ Set.Icc a b, f x = 5) :=
by sorry

end NUMINAMATH_CALUDE_max_min_values_of_f_l2548_254836


namespace NUMINAMATH_CALUDE_min_value_trig_expression_l2548_254871

theorem min_value_trig_expression (α β : ℝ) : 
  (2 * Real.cos α + 5 * Real.sin β - 8)^2 + (2 * Real.sin α + 5 * Real.cos β - 15)^2 ≥ 100 := by
  sorry

end NUMINAMATH_CALUDE_min_value_trig_expression_l2548_254871


namespace NUMINAMATH_CALUDE_total_spent_pinball_l2548_254890

/-- The amount of money in dollars represented by a half-dollar coin -/
def half_dollar_value : ℚ := 0.5

/-- The number of half-dollars Joan spent on Wednesday -/
def wednesday_spent : ℕ := 4

/-- The number of half-dollars Joan spent on Thursday -/
def thursday_spent : ℕ := 14

/-- The number of half-dollars Joan spent on Friday -/
def friday_spent : ℕ := 8

/-- Theorem: The total amount Joan spent playing pinball over three days is $13.00 -/
theorem total_spent_pinball :
  (wednesday_spent + thursday_spent + friday_spent : ℚ) * half_dollar_value = 13 :=
by sorry

end NUMINAMATH_CALUDE_total_spent_pinball_l2548_254890


namespace NUMINAMATH_CALUDE_roots_and_d_values_l2548_254839

-- Define the polynomial p(x)
def p (c d x : ℝ) : ℝ := x^3 + c*x + d

-- Define the polynomial q(x)
def q (c d x : ℝ) : ℝ := x^3 + c*x + d + 144

-- State the theorem
theorem roots_and_d_values (u v c d : ℝ) :
  (p c d u = 0) ∧ (p c d v = 0) ∧ 
  (q c d (u + 3) = 0) ∧ (q c d (v - 2) = 0) →
  (d = 84 ∨ d = -15) := by
  sorry


end NUMINAMATH_CALUDE_roots_and_d_values_l2548_254839


namespace NUMINAMATH_CALUDE_chi_square_relationship_confidence_l2548_254845

/-- The critical value for 99% confidence level in this χ² test -/
def critical_value : ℝ := 6.635

/-- The observed χ² value -/
def observed_chi_square : ℝ := 8.654

/-- The confidence level as a percentage -/
def confidence_level : ℝ := 99

theorem chi_square_relationship_confidence :
  observed_chi_square > critical_value →
  confidence_level = 99 := by
sorry

end NUMINAMATH_CALUDE_chi_square_relationship_confidence_l2548_254845


namespace NUMINAMATH_CALUDE_three_cones_theorem_l2548_254840

/-- Represents a cone with apex A -/
structure Cone where
  apexAngle : ℝ

/-- Represents a plane -/
structure Plane where

/-- Checks if a cone touches a plane -/
def touchesPlane (c : Cone) (p : Plane) : Prop :=
  sorry

/-- Checks if two cones are identical -/
def areIdentical (c1 c2 : Cone) : Prop :=
  c1.apexAngle = c2.apexAngle

/-- Checks if cones lie on the same side of a plane -/
def onSameSide (c1 c2 c3 : Cone) (p : Plane) : Prop :=
  sorry

theorem three_cones_theorem (c1 c2 c3 : Cone) (p : Plane) :
  areIdentical c1 c2 →
  c3.apexAngle = π / 2 →
  touchesPlane c1 p →
  touchesPlane c2 p →
  touchesPlane c3 p →
  onSameSide c1 c2 c3 p →
  c1.apexAngle = 2 * Real.arctan (4 / 5) :=
sorry

end NUMINAMATH_CALUDE_three_cones_theorem_l2548_254840


namespace NUMINAMATH_CALUDE_square_plot_area_l2548_254832

/-- Given a square plot with a perimeter that costs a certain amount to fence at a given price per foot, 
    this theorem proves that the area of the plot is as calculated. -/
theorem square_plot_area 
  (perimeter_cost : ℝ) 
  (price_per_foot : ℝ) 
  (perimeter_cost_positive : perimeter_cost > 0)
  (price_per_foot_positive : price_per_foot > 0)
  (h_cost : perimeter_cost = 3944)
  (h_price : price_per_foot = 58) : 
  (perimeter_cost / (4 * price_per_foot))^2 = 289 := by
  sorry

#eval (3944 / (4 * 58))^2  -- Should evaluate to 289.0

end NUMINAMATH_CALUDE_square_plot_area_l2548_254832


namespace NUMINAMATH_CALUDE_smallest_sum_B_plus_c_l2548_254815

-- Define a digit in base 5
def is_base5_digit (B : ℕ) : Prop := 0 ≤ B ∧ B < 5

-- Define a base greater than or equal to 6
def is_valid_base (c : ℕ) : Prop := c ≥ 6

-- Define the equality BBB_5 = 44_c
def number_equality (B c : ℕ) : Prop := 31 * B = 4 * (c + 1)

-- Theorem statement
theorem smallest_sum_B_plus_c :
  ∀ B c : ℕ,
  is_base5_digit B →
  is_valid_base c →
  number_equality B c →
  (∀ B' c' : ℕ, is_base5_digit B' → is_valid_base c' → number_equality B' c' → B + c ≤ B' + c') →
  B + c = 8 :=
sorry

end NUMINAMATH_CALUDE_smallest_sum_B_plus_c_l2548_254815


namespace NUMINAMATH_CALUDE_arithmetic_mean_18_27_45_l2548_254837

def arithmetic_mean (a b c : ℕ) : ℚ :=
  (a + b + c : ℚ) / 3

theorem arithmetic_mean_18_27_45 :
  arithmetic_mean 18 27 45 = 30 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_mean_18_27_45_l2548_254837


namespace NUMINAMATH_CALUDE_profit_function_satisfies_conditions_max_profit_at_45_profit_function_is_quadratic_l2548_254834

/-- The profit function for a toy store -/
def profit_function (x : ℝ) : ℝ := -2 * (x - 30) * (x - 60)

/-- The theorem stating that the profit function satisfies all required conditions -/
theorem profit_function_satisfies_conditions :
  (profit_function 30 = 0) ∧ 
  (∃ (max_profit : ℝ), max_profit = profit_function 45 ∧ 
    ∀ (x : ℝ), profit_function x ≤ max_profit) ∧
  (profit_function 45 = 450) ∧
  (profit_function 60 = 0) := by
  sorry

/-- The maximum profit occurs at x = 45 -/
theorem max_profit_at_45 :
  ∀ (x : ℝ), profit_function x ≤ profit_function 45 := by
  sorry

/-- The profit function is a quadratic function -/
theorem profit_function_is_quadratic :
  ∃ (a b c : ℝ), ∀ (x : ℝ), profit_function x = a * x^2 + b * x + c := by
  sorry

end NUMINAMATH_CALUDE_profit_function_satisfies_conditions_max_profit_at_45_profit_function_is_quadratic_l2548_254834


namespace NUMINAMATH_CALUDE_no_integer_root_seven_l2548_254824

theorem no_integer_root_seven
  (P : Int → Int)  -- P is a polynomial with integer coefficients
  (h_int_coeff : ∀ x, ∃ y, P x = y)  -- P has integer coefficients
  (a b c d : Int)  -- a, b, c, d are integers
  (h_distinct : a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d)  -- a, b, c, d are distinct
  (h_equal_four : P a = 4 ∧ P b = 4 ∧ P c = 4 ∧ P d = 4)  -- P(a) = P(b) = P(c) = P(d) = 4
  : ¬ ∃ e : Int, P e = 7 := by  -- There does not exist an integer e such that P(e) = 7
  sorry

end NUMINAMATH_CALUDE_no_integer_root_seven_l2548_254824


namespace NUMINAMATH_CALUDE_complex_fraction_simplification_l2548_254880

-- Define the complex number i
def i : ℂ := Complex.I

-- State the theorem
theorem complex_fraction_simplification :
  (5 + 7 * i) / (3 - 4 * i + 2 * i^2) = -23/17 + (27/17) * i :=
by
  sorry

end NUMINAMATH_CALUDE_complex_fraction_simplification_l2548_254880


namespace NUMINAMATH_CALUDE_cuboid_surface_area_l2548_254893

theorem cuboid_surface_area 
  (x y z : ℝ) 
  (edge_sum : 4*x + 4*y + 4*z = 160) 
  (diagonal : x^2 + y^2 + z^2 = 25^2) : 
  2*(x*y + y*z + z*x) = 975 := by
sorry

end NUMINAMATH_CALUDE_cuboid_surface_area_l2548_254893


namespace NUMINAMATH_CALUDE_remaining_bulbs_correct_l2548_254811

def calculate_remaining_bulbs (initial_led : ℕ) (initial_incandescent : ℕ)
  (used_led : ℕ) (used_incandescent : ℕ)
  (alex_percent : ℚ) (bob_percent : ℚ) (charlie_led_percent : ℚ) (charlie_incandescent_percent : ℚ)
  : (ℕ × ℕ) :=
  sorry

theorem remaining_bulbs_correct :
  let initial_led := 24
  let initial_incandescent := 16
  let used_led := 10
  let used_incandescent := 6
  let alex_percent := 1/2
  let bob_percent := 1/4
  let charlie_led_percent := 1/5
  let charlie_incandescent_percent := 3/10
  calculate_remaining_bulbs initial_led initial_incandescent used_led used_incandescent
    alex_percent bob_percent charlie_led_percent charlie_incandescent_percent = (6, 6) :=
by
  sorry

end NUMINAMATH_CALUDE_remaining_bulbs_correct_l2548_254811


namespace NUMINAMATH_CALUDE_simplify_expression_l2548_254822

theorem simplify_expression (x : ℝ) : 3 * x - 5 * x^2 + 7 + (2 - 3 * x + 5 * x^2) = 9 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l2548_254822


namespace NUMINAMATH_CALUDE_curve_equation_and_m_range_l2548_254874

-- Define the curve C
def C : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.1 > 0 ∧ Real.sqrt ((p.1 - 1)^2 + p.2^2) - p.1 = 1}

-- Define the function for the dot product of vectors FA and FB
def dotProductFAFB (m : ℝ) (A B : ℝ × ℝ) : ℝ :=
  (A.1 - 1) * (B.1 - 1) + A.2 * B.2

theorem curve_equation_and_m_range :
  -- Part 1: The equation of curve C
  (∀ p : ℝ × ℝ, p ∈ C ↔ p.1 > 0 ∧ p.2^2 = 4 * p.1) ∧
  -- Part 2: Existence of m
  (∃ m : ℝ, m > 0 ∧
    ∀ A B : ℝ × ℝ, A ∈ C → B ∈ C →
      (∃ t : ℝ, A.1 = t * A.2 + m ∧ B.1 = t * B.2 + m) →
        dotProductFAFB m A B < 0) ∧
  -- Part 3: Range of m
  (∀ m : ℝ, (∀ A B : ℝ × ℝ, A ∈ C → B ∈ C →
    (∃ t : ℝ, A.1 = t * A.2 + m ∧ B.1 = t * B.2 + m) →
      dotProductFAFB m A B < 0) ↔
        m > 3 - 2 * Real.sqrt 2 ∧ m < 3 + 2 * Real.sqrt 2) :=
by sorry

end NUMINAMATH_CALUDE_curve_equation_and_m_range_l2548_254874


namespace NUMINAMATH_CALUDE_distance_from_origin_l2548_254830

theorem distance_from_origin (x y : ℝ) (h1 : y = 15) (h2 : x > 3)
  (h3 : Real.sqrt ((x - 3)^2 + (y - 8)^2) = 11) :
  Real.sqrt (x^2 + y^2) = Real.sqrt 306 := by sorry

end NUMINAMATH_CALUDE_distance_from_origin_l2548_254830


namespace NUMINAMATH_CALUDE_royal_family_children_count_l2548_254816

/-- Represents the royal family -/
structure RoyalFamily where
  king_age : ℕ
  queen_age : ℕ
  num_sons : ℕ
  num_daughters : ℕ
  children_total_age : ℕ

/-- The possible numbers of children for the royal family -/
def possible_children_numbers : Set ℕ := {7, 9}

theorem royal_family_children_count (family : RoyalFamily) 
  (h1 : family.king_age = 35)
  (h2 : family.queen_age = 35)
  (h3 : family.num_sons = 3)
  (h4 : family.num_daughters ≥ 1)
  (h5 : family.children_total_age = 35)
  (h6 : family.num_sons + family.num_daughters ≤ 20)
  (h7 : ∃ (n : ℕ), n > 0 ∧ family.king_age + n + family.queen_age + n = family.children_total_age + n * (family.num_sons + family.num_daughters)) :
  (family.num_sons + family.num_daughters) ∈ possible_children_numbers :=
sorry

end NUMINAMATH_CALUDE_royal_family_children_count_l2548_254816


namespace NUMINAMATH_CALUDE_unique_pairs_satisfying_equation_l2548_254843

theorem unique_pairs_satisfying_equation :
  ∀ n m : ℕ, m^2 + 2 * 3^n = m * (2^(n + 1) - 1) ↔ (n = 3 ∧ m = 6) ∨ (n = 3 ∧ m = 9) :=
by sorry

end NUMINAMATH_CALUDE_unique_pairs_satisfying_equation_l2548_254843


namespace NUMINAMATH_CALUDE_sum_not_prime_l2548_254878

theorem sum_not_prime (a b c d : ℕ+) (h : a * b = c * d) : 
  ¬ Nat.Prime (a.val + b.val + c.val + d.val) := by
  sorry

end NUMINAMATH_CALUDE_sum_not_prime_l2548_254878


namespace NUMINAMATH_CALUDE_line_intersection_x_axis_l2548_254819

/-- Given a line passing through (2, 6) and (5, c) that intersects the x-axis at (d, 0), prove that d = -16 -/
theorem line_intersection_x_axis (c : ℝ) (d : ℝ) : 
  (∃ (m : ℝ), (6 - 0) = m * (2 - d) ∧ (c - 6) = m * (5 - 2)) → d = -16 := by
  sorry

end NUMINAMATH_CALUDE_line_intersection_x_axis_l2548_254819


namespace NUMINAMATH_CALUDE_largest_n_binomial_equality_l2548_254867

theorem largest_n_binomial_equality : ∃ (n : ℕ), (
  (Nat.choose 10 4 + Nat.choose 10 5 = Nat.choose 11 n) ∧
  (∀ m : ℕ, Nat.choose 10 4 + Nat.choose 10 5 = Nat.choose 11 m → m ≤ n)
) ∧ n = 6 := by
  sorry

end NUMINAMATH_CALUDE_largest_n_binomial_equality_l2548_254867


namespace NUMINAMATH_CALUDE_complex_modulus_problem_l2548_254835

theorem complex_modulus_problem (z : ℂ) (h : (1 - Complex.I) * z = 1) : 
  Complex.abs (2 * z - 3) = Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_complex_modulus_problem_l2548_254835


namespace NUMINAMATH_CALUDE_obesity_probability_l2548_254872

theorem obesity_probability (P_obese_male P_obese_female : ℝ) 
  (ratio_male_female : ℚ) :
  P_obese_male = 1/5 →
  P_obese_female = 1/10 →
  ratio_male_female = 3/2 →
  let P_male := ratio_male_female / (1 + ratio_male_female)
  let P_female := 1 - P_male
  let P_obese := P_male * P_obese_male + P_female * P_obese_female
  (P_male * P_obese_male) / P_obese = 3/4 := by
sorry

end NUMINAMATH_CALUDE_obesity_probability_l2548_254872


namespace NUMINAMATH_CALUDE_areas_theorem_l2548_254850

-- Define the areas A, B, and C
def A : ℝ := sorry
def B : ℝ := sorry
def C : ℝ := sorry

-- State the theorem
theorem areas_theorem :
  -- Condition for A: square with diagonal 2√2
  (∃ (s : ℝ), s * s = A ∧ s * Real.sqrt 2 = 2 * Real.sqrt 2) →
  -- Condition for B: rectangle with given vertices
  (∃ (w h : ℝ), w * h = B ∧ w = 4 ∧ h = 2) →
  -- Condition for C: triangle formed by axes and line y = -x/2 + 2
  (∃ (base height : ℝ), (1/2) * base * height = C ∧ base = 4 ∧ height = 2) →
  -- Conclusion
  A = 4 ∧ B = 8 ∧ C = 4 := by
sorry


end NUMINAMATH_CALUDE_areas_theorem_l2548_254850


namespace NUMINAMATH_CALUDE_michaels_brothers_ages_l2548_254877

theorem michaels_brothers_ages (michael_age : ℕ) (older_brother_age : ℕ) (younger_brother_age : ℕ) :
  older_brother_age = 2 * (michael_age - 1) + 1 →
  younger_brother_age = older_brother_age / 3 →
  michael_age + older_brother_age + younger_brother_age = 28 →
  younger_brother_age = 5 := by
  sorry

end NUMINAMATH_CALUDE_michaels_brothers_ages_l2548_254877


namespace NUMINAMATH_CALUDE_min_distinct_lines_for_31_links_l2548_254868

/-- A polygonal chain in a plane -/
structure PolygonalChain where
  links : ℕ
  non_self_intersecting : Bool
  adjacent_links_not_collinear : Bool

/-- The minimum number of distinct lines needed to contain all links of a polygonal chain -/
def min_distinct_lines (chain : PolygonalChain) : ℕ := sorry

/-- Theorem: For a non-self-intersecting polygonal chain with 31 links where adjacent links are not collinear, 
    the minimum number of distinct lines that can contain all links is 9 -/
theorem min_distinct_lines_for_31_links : 
  ∀ (chain : PolygonalChain), 
    chain.links = 31 ∧ 
    chain.non_self_intersecting = true ∧ 
    chain.adjacent_links_not_collinear = true → 
    min_distinct_lines chain = 9 := by sorry

end NUMINAMATH_CALUDE_min_distinct_lines_for_31_links_l2548_254868


namespace NUMINAMATH_CALUDE_quadratic_solution_difference_squared_l2548_254817

theorem quadratic_solution_difference_squared :
  ∀ a b : ℝ,
  (2 * a^2 - 7 * a + 6 = 0) →
  (2 * b^2 - 7 * b + 6 = 0) →
  (a ≠ b) →
  (a - b)^2 = (1 : ℝ) / 4 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_solution_difference_squared_l2548_254817


namespace NUMINAMATH_CALUDE_part_I_part_II_l2548_254841

-- Define propositions p and q
def p (a : ℝ) : Prop := a > 0
def q (a : ℝ) : Prop := ∀ x : ℝ, x ∈ Set.Icc (-2) (-1) → a - x ≥ 0

-- Part I
theorem part_I (a : ℝ) (hq : q a) : a ∈ {a : ℝ | a ≥ -1} := by sorry

-- Part II
theorem part_II (a : ℝ) (h_or : p a ∨ q a) (h_not_and : ¬(p a ∧ q a)) : 
  a ∈ Set.Icc (-1) 0 := by sorry

end NUMINAMATH_CALUDE_part_I_part_II_l2548_254841


namespace NUMINAMATH_CALUDE_cosine_amplitude_and_shift_l2548_254801

/-- Given a cosine function that oscillates between 5 and 1, prove its amplitude and vertical shift. -/
theorem cosine_amplitude_and_shift (a b c d : ℝ) : 
  (∀ x, 1 ≤ a * Real.cos (b * x + c) + d ∧ a * Real.cos (b * x + c) + d ≤ 5) →
  a = 2 ∧ d = 3 := by
  sorry

end NUMINAMATH_CALUDE_cosine_amplitude_and_shift_l2548_254801


namespace NUMINAMATH_CALUDE_necessary_not_sufficient_l2548_254849

/-- Represents an ellipse equation with parameter m -/
def is_ellipse (m : ℝ) : Prop :=
  (m > 1) ∧ (m < 3) ∧ (m ≠ 2)

/-- The condition given in the problem -/
def given_condition (m : ℝ) : Prop :=
  (m > 1) ∧ (m < 3)

/-- Theorem stating that the given condition is necessary but not sufficient -/
theorem necessary_not_sufficient :
  (∀ m : ℝ, is_ellipse m → given_condition m) ∧
  ¬(∀ m : ℝ, given_condition m → is_ellipse m) :=
sorry

end NUMINAMATH_CALUDE_necessary_not_sufficient_l2548_254849


namespace NUMINAMATH_CALUDE_officer_selection_count_l2548_254820

def club_members : ℕ := 12
def officer_positions : ℕ := 5

theorem officer_selection_count :
  (club_members.factorial) / ((club_members - officer_positions).factorial) = 95040 := by
  sorry

end NUMINAMATH_CALUDE_officer_selection_count_l2548_254820


namespace NUMINAMATH_CALUDE_colored_paper_count_l2548_254831

theorem colored_paper_count (people : ℕ) (pieces_per_person : ℕ) (leftover : ℕ) : 
  people = 6 → pieces_per_person = 7 → leftover = 3 → 
  people * pieces_per_person + leftover = 45 := by
  sorry

end NUMINAMATH_CALUDE_colored_paper_count_l2548_254831
