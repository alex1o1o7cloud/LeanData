import Mathlib

namespace NUMINAMATH_CALUDE_inverse_sum_zero_l1969_196972

def A : Matrix (Fin 2) (Fin 2) ℝ := !![5, 1; 7, 3]

theorem inverse_sum_zero :
  ∃ (B : Matrix (Fin 2) (Fin 2) ℝ), 
    A * B = 1 ∧ B * A = 1 →
    B 0 0 + B 0 1 + B 1 0 + B 1 1 = 0 := by
  sorry

end NUMINAMATH_CALUDE_inverse_sum_zero_l1969_196972


namespace NUMINAMATH_CALUDE_joan_picked_37_oranges_l1969_196906

/-- The number of oranges picked by Sara -/
def sara_oranges : ℕ := 10

/-- The total number of oranges picked -/
def total_oranges : ℕ := 47

/-- The number of oranges picked by Joan -/
def joan_oranges : ℕ := total_oranges - sara_oranges

theorem joan_picked_37_oranges : joan_oranges = 37 := by
  sorry

end NUMINAMATH_CALUDE_joan_picked_37_oranges_l1969_196906


namespace NUMINAMATH_CALUDE_triple_hash_40_l1969_196912

-- Define the # operation
def hash (N : ℝ) : ℝ := 0.3 * N + 2

-- State the theorem
theorem triple_hash_40 : hash (hash (hash 40)) = 3.86 := by
  sorry

end NUMINAMATH_CALUDE_triple_hash_40_l1969_196912


namespace NUMINAMATH_CALUDE_calculation_proof_l1969_196919

theorem calculation_proof : 2.5 * 8 * (5.2 + 4.8)^2 = 2000 := by
  sorry

end NUMINAMATH_CALUDE_calculation_proof_l1969_196919


namespace NUMINAMATH_CALUDE_stacy_heather_walk_stacy_heather_initial_distance_l1969_196921

/-- The problem of Stacy and Heather walking towards each other -/
theorem stacy_heather_walk (stacy_speed heather_speed : ℝ) 
  (heather_start_delay : ℝ) (heather_distance : ℝ) : ℝ :=
  let initial_distance : ℝ := 
    by {
      -- Define the conditions
      have h1 : stacy_speed = heather_speed + 1 := by sorry
      have h2 : heather_speed = 5 := by sorry
      have h3 : heather_start_delay = 24 / 60 := by sorry
      have h4 : heather_distance = 5.7272727272727275 := by sorry

      -- Calculate the initial distance
      sorry
    }
  initial_distance

/-- The theorem stating that Stacy and Heather were initially 15 miles apart -/
theorem stacy_heather_initial_distance : 
  stacy_heather_walk 6 5 (24/60) 5.7272727272727275 = 15 := by sorry

end NUMINAMATH_CALUDE_stacy_heather_walk_stacy_heather_initial_distance_l1969_196921


namespace NUMINAMATH_CALUDE_ellipse_standard_equation_ellipse_trajectory_equation_l1969_196956

/-- An ellipse with center at origin, left focus at (-√3, 0), right vertex at (2, 0), and point A at (1, 1/2) -/
structure Ellipse where
  center : ℝ × ℝ
  left_focus : ℝ × ℝ
  right_vertex : ℝ × ℝ
  point_A : ℝ × ℝ
  h_center : center = (0, 0)
  h_left_focus : left_focus = (-Real.sqrt 3, 0)
  h_right_vertex : right_vertex = (2, 0)
  h_point_A : point_A = (1, 1/2)

/-- The standard equation of the ellipse -/
def standard_equation (e : Ellipse) (x y : ℝ) : Prop :=
  x^2 / 4 + y^2 = 1

/-- The trajectory equation of the midpoint M of line segment PA -/
def trajectory_equation (e : Ellipse) (x y : ℝ) : Prop :=
  (2*x - 1)^2 / 4 + (2*y - 1/2)^2 = 1

/-- Theorem stating the standard equation of the ellipse -/
theorem ellipse_standard_equation (e : Ellipse) :
  ∀ x y : ℝ, (x, y) ∈ {p : ℝ × ℝ | standard_equation e p.1 p.2} ↔ 
    (x, y) ∈ {p : ℝ × ℝ | x^2 / 4 + y^2 = 1} :=
sorry

/-- Theorem stating the trajectory equation of the midpoint M -/
theorem ellipse_trajectory_equation (e : Ellipse) :
  ∀ x y : ℝ, (x, y) ∈ {p : ℝ × ℝ | trajectory_equation e p.1 p.2} ↔ 
    (x, y) ∈ {p : ℝ × ℝ | (2*x - 1)^2 / 4 + (2*y - 1/2)^2 = 1} :=
sorry

end NUMINAMATH_CALUDE_ellipse_standard_equation_ellipse_trajectory_equation_l1969_196956


namespace NUMINAMATH_CALUDE_probability_at_least_one_type_b_l1969_196926

def total_questions : ℕ := 5
def type_a_questions : ℕ := 2
def type_b_questions : ℕ := 3
def selected_questions : ℕ := 2

theorem probability_at_least_one_type_b :
  let total_combinations := Nat.choose total_questions selected_questions
  let all_type_a_combinations := Nat.choose type_a_questions selected_questions
  (total_combinations - all_type_a_combinations) / total_combinations = 9 / 10 := by
  sorry

end NUMINAMATH_CALUDE_probability_at_least_one_type_b_l1969_196926


namespace NUMINAMATH_CALUDE_total_food_eaten_l1969_196961

/-- The amount of food Ella eats in one day, in pounds. -/
def ellaFoodPerDay : ℝ := 20

/-- The ratio of food Ella's dog eats compared to Ella. -/
def dogFoodRatio : ℝ := 4

/-- The number of days considered. -/
def numDays : ℝ := 10

/-- Theorem stating the total amount of food eaten by Ella and her dog in the given period. -/
theorem total_food_eaten : 
  (ellaFoodPerDay * numDays) + (ellaFoodPerDay * dogFoodRatio * numDays) = 1000 := by
  sorry

end NUMINAMATH_CALUDE_total_food_eaten_l1969_196961


namespace NUMINAMATH_CALUDE_sum_of_fractions_geq_three_l1969_196989

theorem sum_of_fractions_geq_three (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  (1 + a^2) / (1 + a*b) + (1 + b^2) / (1 + b*c) + (1 + c^2) / (1 + c*a) ≥ 3 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_fractions_geq_three_l1969_196989


namespace NUMINAMATH_CALUDE_reflection_sum_coordinates_l1969_196969

/-- Given a point C with coordinates (x, 8), when reflected over the y-axis to point D,
    the sum of all coordinate values of C and D is 16. -/
theorem reflection_sum_coordinates (x : ℝ) : 
  let C : ℝ × ℝ := (x, 8)
  let D : ℝ × ℝ := (-x, 8)
  x + 8 + (-x) + 8 = 16 := by sorry

end NUMINAMATH_CALUDE_reflection_sum_coordinates_l1969_196969


namespace NUMINAMATH_CALUDE_solve_exponential_equation_l1969_196984

theorem solve_exponential_equation :
  ∃ w : ℝ, (2 : ℝ)^(2*w) = 8^(w-4) → w = 12 := by
  sorry

end NUMINAMATH_CALUDE_solve_exponential_equation_l1969_196984


namespace NUMINAMATH_CALUDE_max_n_for_factorization_l1969_196992

theorem max_n_for_factorization : 
  (∃ (n : ℤ), ∀ (x : ℝ), ∃ (A B : ℤ), 
    6 * x^2 + n * x + 144 = (6 * x + A) * (x + B)) ∧
  (∀ (m : ℤ), m > 865 → 
    ¬∃ (A B : ℤ), ∀ (x : ℝ), 6 * x^2 + m * x + 144 = (6 * x + A) * (x + B)) :=
by sorry

end NUMINAMATH_CALUDE_max_n_for_factorization_l1969_196992


namespace NUMINAMATH_CALUDE_license_plate_palindrome_probability_l1969_196995

def letter_count : ℕ := 26
def digit_count : ℕ := 10
def plate_length : ℕ := 4

def prob_digit_palindrome : ℚ := (digit_count ^ 2) / (digit_count ^ plate_length)
def prob_letter_palindrome : ℚ := (letter_count ^ 2) / (letter_count ^ plate_length)

theorem license_plate_palindrome_probability :
  let prob_at_least_one_palindrome := prob_digit_palindrome + prob_letter_palindrome - 
    (prob_digit_palindrome * prob_letter_palindrome)
  prob_at_least_one_palindrome = 97 / 8450 := by
  sorry

end NUMINAMATH_CALUDE_license_plate_palindrome_probability_l1969_196995


namespace NUMINAMATH_CALUDE_triangle_inequality_l1969_196910

theorem triangle_inequality (a b c : ℝ) 
  (ha : 0 < a) (hb : 0 < b) (hc : 0 < c)
  (hab : a + b > c) (hbc : b + c > a) (hca : c + a > b) :
  (a * b + 1) / (a^2 + c * a + 1) + 
  (b * c + 1) / (b^2 + a * b + 1) + 
  (c * a + 1) / (c^2 + b * c + 1) > 3/2 := by
sorry

end NUMINAMATH_CALUDE_triangle_inequality_l1969_196910


namespace NUMINAMATH_CALUDE_fraction_equality_implies_numerator_equality_l1969_196940

theorem fraction_equality_implies_numerator_equality
  (a b c : ℝ) (h1 : c ≠ 0) (h2 : a / c = b / c) : a = b :=
by sorry

end NUMINAMATH_CALUDE_fraction_equality_implies_numerator_equality_l1969_196940


namespace NUMINAMATH_CALUDE_sum_of_powers_l1969_196954

theorem sum_of_powers (w : ℂ) (hw : w^3 + w^2 + 1 = 0) :
  w^100 + w^101 + w^102 + w^103 + w^104 = 5 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_powers_l1969_196954


namespace NUMINAMATH_CALUDE_inscribed_cube_volume_l1969_196948

theorem inscribed_cube_volume (outer_cube_edge : ℝ) (sphere_diameter : ℝ) 
  (inner_cube_edge : ℝ) (inner_cube_volume : ℝ) :
  outer_cube_edge = 12 →
  sphere_diameter = outer_cube_edge →
  sphere_diameter = inner_cube_edge * Real.sqrt 3 →
  inner_cube_volume = inner_cube_edge ^ 3 →
  inner_cube_volume = 192 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_inscribed_cube_volume_l1969_196948


namespace NUMINAMATH_CALUDE_union_of_A_and_B_l1969_196923

-- Define the sets A and B
def A : Set ℝ := {x | x^2 ≤ 4}
def B : Set ℝ := {x | x > 0}

-- State the theorem
theorem union_of_A_and_B : A ∪ B = {x : ℝ | x ≥ -2} := by sorry

end NUMINAMATH_CALUDE_union_of_A_and_B_l1969_196923


namespace NUMINAMATH_CALUDE_frequency_of_boys_born_l1969_196957

theorem frequency_of_boys_born (total : ℕ) (boys : ℕ) (h1 : total = 1000) (h2 : boys = 515) :
  (boys : ℚ) / total = 0.515 := by
sorry

end NUMINAMATH_CALUDE_frequency_of_boys_born_l1969_196957


namespace NUMINAMATH_CALUDE_ellipse_condition_l1969_196933

def ellipse_equation (x y k : ℝ) : Prop :=
  3 * x^2 + 6 * y^2 - 12 * x + 18 * y = k

def is_non_degenerate_ellipse (k : ℝ) : Prop :=
  ∃ (a b h c d : ℝ), a > 0 ∧ b > 0 ∧
    ∀ (x y : ℝ), ellipse_equation x y k ↔ (x - h)^2 / a^2 + (y - d)^2 / b^2 = 1

theorem ellipse_condition (k : ℝ) :
  is_non_degenerate_ellipse k ↔ k > -51/2 :=
sorry

end NUMINAMATH_CALUDE_ellipse_condition_l1969_196933


namespace NUMINAMATH_CALUDE_truncated_cone_complete_height_l1969_196947

/-- Given a truncated cone with height h, upper radius r, and lower radius R,
    the height H of the corresponding complete cone is hR / (R - r) -/
theorem truncated_cone_complete_height
  (h r R : ℝ) (h_pos : h > 0) (r_pos : r > 0) (R_pos : R > 0) (r_lt_R : r < R) :
  ∃ H : ℝ, H = h * R / (R - r) ∧ H > h := by
  sorry

#check truncated_cone_complete_height

end NUMINAMATH_CALUDE_truncated_cone_complete_height_l1969_196947


namespace NUMINAMATH_CALUDE_union_of_A_and_B_l1969_196922

-- Define set A
def A : Set ℝ := {x | x^2 < 4}

-- Define set B
def B : Set ℝ := {y | ∃ x ∈ A, y = x^2 - 2*x - 1}

-- Define the union of A and B
def AUnionB : Set ℝ := {x | -2 ≤ x ∧ x < 7}

-- Theorem statement
theorem union_of_A_and_B : A ∪ B = AUnionB := by sorry

end NUMINAMATH_CALUDE_union_of_A_and_B_l1969_196922


namespace NUMINAMATH_CALUDE_square_sum_xy_l1969_196978

theorem square_sum_xy (x y : ℕ+) 
  (h1 : x * y + x + y = 119)
  (h2 : x^2 * y + x * y^2 = 1680) :
  x^2 + y^2 = 1057 := by
  sorry

end NUMINAMATH_CALUDE_square_sum_xy_l1969_196978


namespace NUMINAMATH_CALUDE_min_value_fraction_l1969_196939

theorem min_value_fraction (x : ℝ) (h : x > 9) : 
  x^2 / (x - 9) ≥ 36 ∧ ∃ y > 9, y^2 / (y - 9) = 36 := by
  sorry

end NUMINAMATH_CALUDE_min_value_fraction_l1969_196939


namespace NUMINAMATH_CALUDE_total_diagonals_specific_prism_l1969_196952

/-- A rectangular prism with edge lengths a, b, and c. -/
structure RectangularPrism where
  a : ℝ
  b : ℝ
  c : ℝ

/-- The number of face diagonals in a rectangular prism. -/
def face_diagonals (p : RectangularPrism) : ℕ := 12

/-- The number of space diagonals in a rectangular prism. -/
def space_diagonals (p : RectangularPrism) : ℕ := 4

/-- The total number of diagonals in a rectangular prism. -/
def total_diagonals (p : RectangularPrism) : ℕ :=
  face_diagonals p + space_diagonals p

/-- Theorem: The total number of diagonals in a rectangular prism
    with edge lengths 4, 6, and 8 is 16. -/
theorem total_diagonals_specific_prism :
  ∃ p : RectangularPrism, p.a = 4 ∧ p.b = 6 ∧ p.c = 8 ∧ total_diagonals p = 16 := by
  sorry

end NUMINAMATH_CALUDE_total_diagonals_specific_prism_l1969_196952


namespace NUMINAMATH_CALUDE_certain_and_uncertain_digits_l1969_196934

def value : ℝ := 945.673
def absolute_error : ℝ := 0.03

def is_certain (digit : ℕ) (place_value : ℝ) : Prop :=
  place_value > absolute_error

def is_uncertain (digit : ℕ) (place_value : ℝ) : Prop :=
  place_value < absolute_error

theorem certain_and_uncertain_digits :
  (is_certain 9 100) ∧
  (is_certain 4 10) ∧
  (is_certain 5 1) ∧
  (is_certain 6 0.1) ∧
  (is_uncertain 7 0.01) ∧
  (is_uncertain 3 0.001) :=
by sorry

end NUMINAMATH_CALUDE_certain_and_uncertain_digits_l1969_196934


namespace NUMINAMATH_CALUDE_sum_of_square_roots_equals_one_l1969_196979

theorem sum_of_square_roots_equals_one (a b c : ℝ) 
  (h1 : a * c / (a + b) + b * a / (b + c) + c * b / (c + a) = -12)
  (h2 : b * c / (a + b) + c * a / (b + c) + a * b / (c + a) = 15) :
  Real.sqrt b / (a + b) + Real.sqrt c / (b + c) + Real.sqrt a / (c + a) = 1 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_square_roots_equals_one_l1969_196979


namespace NUMINAMATH_CALUDE_arithmetic_sequence_difference_l1969_196927

def is_arithmetic_sequence (a b c d e : ℝ) : Prop :=
  ∃ r : ℝ, b - a = r ∧ c - b = r ∧ d - c = r ∧ e - d = r

theorem arithmetic_sequence_difference 
  (a b c : ℝ) (h : is_arithmetic_sequence 2 a b c 9) : 
  c - a = (7 : ℝ) / 2 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_difference_l1969_196927


namespace NUMINAMATH_CALUDE_negation_equivalence_l1969_196983

theorem negation_equivalence (m : ℤ) : 
  (¬ ∃ x : ℤ, x^2 + x + m < 0) ↔ (∀ x : ℤ, x^2 + x + m ≥ 0) := by
  sorry

end NUMINAMATH_CALUDE_negation_equivalence_l1969_196983


namespace NUMINAMATH_CALUDE_initial_mixture_volume_l1969_196967

/-- Given a mixture where water is 10% of the total volume, if adding 14 liters of water
    results in a new mixture with 25% water, then the initial volume of the mixture was 70 liters. -/
theorem initial_mixture_volume (V : ℝ) : 
  (0.1 * V + 14) / (V + 14) = 0.25 → V = 70 := by
  sorry

end NUMINAMATH_CALUDE_initial_mixture_volume_l1969_196967


namespace NUMINAMATH_CALUDE_deleted_pictures_count_l1969_196976

def zoo_pictures : ℕ := 15
def museum_pictures : ℕ := 18
def remaining_pictures : ℕ := 2

theorem deleted_pictures_count :
  zoo_pictures + museum_pictures - remaining_pictures = 31 := by
  sorry

end NUMINAMATH_CALUDE_deleted_pictures_count_l1969_196976


namespace NUMINAMATH_CALUDE_musicians_count_l1969_196997

theorem musicians_count : ∃! n : ℕ, 
  80 < n ∧ n < 130 ∧ 
  n % 4 = 3 ∧ 
  n % 5 = 3 ∧ 
  n % 6 = 3 ∧ 
  n = 123 := by
sorry

end NUMINAMATH_CALUDE_musicians_count_l1969_196997


namespace NUMINAMATH_CALUDE_new_surface_area_after_increase_l1969_196916

-- Define the original edge length
def original_edge : ℝ := 7

-- Define the increase percentage
def increase_percentage : ℝ := 0.15

-- Define the function to calculate the new edge length
def new_edge (e : ℝ) (p : ℝ) : ℝ := e * (1 + p)

-- Define the function to calculate the surface area of a cube
def surface_area (e : ℝ) : ℝ := 6 * e^2

-- State the theorem
theorem new_surface_area_after_increase :
  surface_area (new_edge original_edge increase_percentage) = 388.815 := by
  sorry

end NUMINAMATH_CALUDE_new_surface_area_after_increase_l1969_196916


namespace NUMINAMATH_CALUDE_arithmetic_sequence_3_2_3_3_l1969_196951

/-- An arithmetic sequence with three terms -/
structure ArithmeticSequence3 where
  a : ℝ  -- first term
  b : ℝ  -- second term
  c : ℝ  -- third term
  is_arithmetic : b - a = c - b

/-- The second term of an arithmetic sequence with 3^2 as first term and 3^3 as third term is 18 -/
theorem arithmetic_sequence_3_2_3_3 :
  ∃ (seq : ArithmeticSequence3), seq.a = 3^2 ∧ seq.c = 3^3 ∧ seq.b = 18 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_3_2_3_3_l1969_196951


namespace NUMINAMATH_CALUDE_infinite_product_equals_nine_l1969_196907

/-- The series S is defined as the sum 1/2 + 2/4 + 3/8 + 4/16 + ... -/
def S : ℝ := 2

/-- The infinite product P is defined as 3^(1/2) * 9^(1/4) * 27^(1/8) * 81^(1/16) * ... -/
noncomputable def P : ℝ := Real.rpow 3 S

theorem infinite_product_equals_nine : P = 9 := by sorry

end NUMINAMATH_CALUDE_infinite_product_equals_nine_l1969_196907


namespace NUMINAMATH_CALUDE_chord_length_line_circle_intersection_l1969_196949

/-- The length of the chord formed by the intersection of a line and a circle -/
theorem chord_length_line_circle_intersection (x y : ℝ) :
  (3 * x + 4 * y - 5 = 0) →  -- Line equation
  (x^2 + y^2 = 4) →          -- Circle equation
  ∃ (A B : ℝ × ℝ),           -- Intersection points A and B
    (3 * A.1 + 4 * A.2 - 5 = 0) ∧
    (A.1^2 + A.2^2 = 4) ∧
    (3 * B.1 + 4 * B.2 - 5 = 0) ∧
    (B.1^2 + B.2^2 = 4) ∧
    Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) = 2 * Real.sqrt 3 :=
by
  sorry

end NUMINAMATH_CALUDE_chord_length_line_circle_intersection_l1969_196949


namespace NUMINAMATH_CALUDE_imaginary_cube_l1969_196917

/-- Given that i is the imaginary unit, prove that 1 + i^3 = 1 - i -/
theorem imaginary_cube (i : ℂ) (h : i^2 = -1) : 1 + i^3 = 1 - i := by
  sorry

end NUMINAMATH_CALUDE_imaginary_cube_l1969_196917


namespace NUMINAMATH_CALUDE_functional_equation_solution_l1969_196959

/-- A function satisfying the given functional equation -/
def FunctionalEquation (f : ℝ → ℝ) : Prop :=
  ∀ x y, x ∈ (Set.Ioo (-1) 1) → y ∈ (Set.Ioo (-1) 1) → (x + y) ∈ (Set.Ioo (-1) 1) →
    f (x + y) = (f x + f y) / (1 - f x * f y)

theorem functional_equation_solution :
  ∀ f : ℝ → ℝ, ContinuousOn f (Set.Ioo (-1) 1) →
  FunctionalEquation f →
  ∃ a : ℝ, |a| ≤ π/2 ∧ ∀ x ∈ (Set.Ioo (-1) 1), f x = Real.tan (a * x) := by
  sorry

end NUMINAMATH_CALUDE_functional_equation_solution_l1969_196959


namespace NUMINAMATH_CALUDE_range_of_x_less_than_6_range_of_a_for_f_less_than_a_l1969_196953

-- Define the function f(x)
def f (x : ℝ) : ℝ := |x - 3| + |x + 1|

-- Theorem for part I
theorem range_of_x_less_than_6 :
  ∀ x : ℝ, f x < 6 ↔ x ∈ Set.Ioo (-2 : ℝ) 4 :=
sorry

-- Theorem for part II
theorem range_of_a_for_f_less_than_a :
  ∀ a : ℝ, (∃ x : ℝ, f x < a) ↔ a ∈ Set.Ioi 4 :=
sorry

end NUMINAMATH_CALUDE_range_of_x_less_than_6_range_of_a_for_f_less_than_a_l1969_196953


namespace NUMINAMATH_CALUDE_coprime_and_indivisible_l1969_196964

theorem coprime_and_indivisible (n : ℕ) (h1 : n > 3) (h2 : Odd n) :
  ∃ (a b : ℕ), a > 0 ∧ b > 0 ∧ Nat.gcd (a * b * (a + b)) n = 1 ∧ ¬(n ∣ (a - b)) := by
  sorry

end NUMINAMATH_CALUDE_coprime_and_indivisible_l1969_196964


namespace NUMINAMATH_CALUDE_pie_slices_today_l1969_196966

/-- The number of slices of pie served during lunch today -/
def lunch_slices : ℕ := 7

/-- The number of slices of pie served during dinner today -/
def dinner_slices : ℕ := 5

/-- The total number of slices of pie served today -/
def total_slices : ℕ := lunch_slices + dinner_slices

theorem pie_slices_today : total_slices = 12 := by
  sorry

end NUMINAMATH_CALUDE_pie_slices_today_l1969_196966


namespace NUMINAMATH_CALUDE_air_conditioner_problem_l1969_196960

/-- Represents the selling prices and quantities of air conditioners --/
structure AirConditioner where
  price_A : ℝ
  price_B : ℝ
  quantity_A : ℕ
  quantity_B : ℕ

/-- Represents the cost prices of air conditioners --/
structure CostPrices where
  cost_A : ℝ
  cost_B : ℝ

/-- The theorem statement for the air conditioner problem --/
theorem air_conditioner_problem 
  (sale1 : AirConditioner)
  (sale2 : AirConditioner)
  (costs : CostPrices)
  (h1 : sale1.quantity_A = 3 ∧ sale1.quantity_B = 5)
  (h2 : sale2.quantity_A = 4 ∧ sale2.quantity_B = 10)
  (h3 : sale1.price_A * sale1.quantity_A + sale1.price_B * sale1.quantity_B = 23500)
  (h4 : sale2.price_A * sale2.quantity_A + sale2.price_B * sale2.quantity_B = 42000)
  (h5 : costs.cost_A = 1800 ∧ costs.cost_B = 2400)
  (h6 : sale1.price_A = sale2.price_A ∧ sale1.price_B = sale2.price_B) :
  sale1.price_A = 2500 ∧ 
  sale1.price_B = 3200 ∧ 
  (∃ m : ℕ, 
    m ≥ 30 ∧ 
    (sale1.price_A - costs.cost_A) * (50 - m) + (sale1.price_B - costs.cost_B) * m ≥ 38000 ∧
    ∀ n : ℕ, n < 30 → (sale1.price_A - costs.cost_A) * (50 - n) + (sale1.price_B - costs.cost_B) * n < 38000) := by
  sorry

end NUMINAMATH_CALUDE_air_conditioner_problem_l1969_196960


namespace NUMINAMATH_CALUDE_f_is_quadratic_l1969_196909

/-- Definition of a quadratic equation in one variable -/
def is_quadratic_equation (f : ℝ → ℝ) : Prop :=
  ∃ (a b c : ℝ), a ≠ 0 ∧ ∀ x, f x = a * x^2 + b * x + c

/-- The function representing the equation 5y = 5y² -/
def f (y : ℝ) : ℝ := 5 * y^2 - 5 * y

/-- Theorem stating that f is a quadratic equation -/
theorem f_is_quadratic : is_quadratic_equation f := by
  sorry


end NUMINAMATH_CALUDE_f_is_quadratic_l1969_196909


namespace NUMINAMATH_CALUDE_squares_after_six_operations_l1969_196973

/-- Calculates the number of squares after n operations -/
def num_squares (n : ℕ) : ℕ := 5 + 3 * n

/-- The number of squares after 6 operations is 29 -/
theorem squares_after_six_operations :
  num_squares 6 = 29 := by
  sorry

end NUMINAMATH_CALUDE_squares_after_six_operations_l1969_196973


namespace NUMINAMATH_CALUDE_index_card_area_l1969_196943

theorem index_card_area (width height : ℝ) (h1 : width = 5) (h2 : height = 8) : 
  ((width - 2) * height = 24 → width * (height - 2) = 30) ∧ 
  ((width * (height - 2) = 24 → (width - 2) * height = 30)) :=
by sorry

end NUMINAMATH_CALUDE_index_card_area_l1969_196943


namespace NUMINAMATH_CALUDE_joan_football_games_l1969_196986

theorem joan_football_games (games_this_year games_total : ℕ) 
  (h1 : games_this_year = 4)
  (h2 : games_total = 13) :
  games_total - games_this_year = 9 := by
  sorry

end NUMINAMATH_CALUDE_joan_football_games_l1969_196986


namespace NUMINAMATH_CALUDE_total_like_count_l1969_196944

/-- Represents the number of employees with a "dislike" attitude -/
def dislike_count : ℕ := sorry

/-- Represents the number of employees with a "neutral" attitude -/
def neutral_count : ℕ := dislike_count + 12

/-- Represents the number of employees with a "like" attitude -/
def like_count : ℕ := 6 * dislike_count

/-- Represents the ratio of employees with each attitude in the stratified sample -/
def sample_ratio : ℕ × ℕ × ℕ := (6, 1, 3)

theorem total_like_count : like_count = 36 := by sorry

end NUMINAMATH_CALUDE_total_like_count_l1969_196944


namespace NUMINAMATH_CALUDE_fixed_point_parabola_l1969_196963

theorem fixed_point_parabola (s : ℝ) : 
  let f : ℝ → ℝ := λ x ↦ 4 * x^2 + s * x - 3 * s
  f 3 = 36 := by
  sorry

end NUMINAMATH_CALUDE_fixed_point_parabola_l1969_196963


namespace NUMINAMATH_CALUDE_fuel_station_theorem_l1969_196987

/-- Represents the fuel station problem --/
def fuel_station_problem (service_cost : ℚ) (fuel_cost_per_liter : ℚ) 
  (num_minivans : ℕ) (num_trucks : ℕ) (total_cost : ℚ) (minivan_tank : ℚ) : Prop :=
  let total_service_cost := (num_minivans + num_trucks : ℚ) * service_cost
  let total_fuel_cost := total_cost - total_service_cost
  let minivan_fuel_cost := (num_minivans : ℚ) * minivan_tank * fuel_cost_per_liter
  let truck_fuel_cost := total_fuel_cost - minivan_fuel_cost
  let truck_fuel_liters := truck_fuel_cost / fuel_cost_per_liter
  let truck_tank := truck_fuel_liters / (num_trucks : ℚ)
  let percentage_increase := (truck_tank - minivan_tank) / minivan_tank * 100
  percentage_increase = 120

/-- The main theorem to be proved --/
theorem fuel_station_theorem : 
  fuel_station_problem 2.2 0.7 4 2 395.4 65 := by
  sorry

end NUMINAMATH_CALUDE_fuel_station_theorem_l1969_196987


namespace NUMINAMATH_CALUDE_remainder_of_binary_number_div_four_l1969_196904

def binary_number : ℕ := 110110111101

theorem remainder_of_binary_number_div_four :
  binary_number % 4 = 1 := by
  sorry

end NUMINAMATH_CALUDE_remainder_of_binary_number_div_four_l1969_196904


namespace NUMINAMATH_CALUDE_runners_meet_again_l1969_196902

-- Define the track circumference
def track_circumference : ℝ := 400

-- Define the runners' speeds
def runner1_speed : ℝ := 5.0
def runner2_speed : ℝ := 5.5
def runner3_speed : ℝ := 6.0

-- Define the time when runners meet again
def meeting_time : ℝ := 800

-- Theorem statement
theorem runners_meet_again :
  ∀ (t : ℝ), t = meeting_time →
  (runner1_speed * t) % track_circumference = 
  (runner2_speed * t) % track_circumference ∧
  (runner2_speed * t) % track_circumference = 
  (runner3_speed * t) % track_circumference :=
by
  sorry

#check runners_meet_again

end NUMINAMATH_CALUDE_runners_meet_again_l1969_196902


namespace NUMINAMATH_CALUDE_haleys_concert_tickets_l1969_196900

theorem haleys_concert_tickets (ticket_price : ℕ) (extra_tickets : ℕ) (total_spent : ℕ) : 
  ticket_price = 4 → extra_tickets = 5 → total_spent = 32 → 
  ∃ (tickets_for_friends : ℕ), 
    ticket_price * (tickets_for_friends + extra_tickets) = total_spent ∧ 
    tickets_for_friends = 3 :=
by sorry

end NUMINAMATH_CALUDE_haleys_concert_tickets_l1969_196900


namespace NUMINAMATH_CALUDE_pens_probability_l1969_196968

theorem pens_probability (total_pens : Nat) (defective_pens : Nat) (bought_pens : Nat) :
  total_pens = 12 →
  defective_pens = 4 →
  bought_pens = 2 →
  (total_pens - defective_pens : ℚ) / total_pens * 
  ((total_pens - defective_pens - 1 : ℚ) / (total_pens - 1)) = 14 / 33 := by
  sorry

#eval (14 : ℚ) / 33 -- To verify the approximate decimal value

end NUMINAMATH_CALUDE_pens_probability_l1969_196968


namespace NUMINAMATH_CALUDE_maya_lifting_improvement_l1969_196993

theorem maya_lifting_improvement (america_initial : ℕ) (america_peak : ℕ) : 
  america_initial = 240 →
  america_peak = 300 →
  (america_peak / 2 : ℕ) - (america_initial / 4 : ℕ) = 90 := by
  sorry

end NUMINAMATH_CALUDE_maya_lifting_improvement_l1969_196993


namespace NUMINAMATH_CALUDE_praveen_age_multiplier_l1969_196985

def present_age : ℕ := 20

def age_3_years_back : ℕ := present_age - 3

def age_after_10_years : ℕ := present_age + 10

theorem praveen_age_multiplier :
  (age_after_10_years : ℚ) / age_3_years_back = 30 / 17 := by sorry

end NUMINAMATH_CALUDE_praveen_age_multiplier_l1969_196985


namespace NUMINAMATH_CALUDE_quadratic_range_l1969_196988

def f (x : ℝ) := x^2 - 4*x + 1

theorem quadratic_range : 
  ∀ y ∈ Set.Icc (-2 : ℝ) 6, ∃ x ∈ Set.Icc 3 5, f x = y ∧
  ∀ x ∈ Set.Icc 3 5, f x ∈ Set.Icc (-2 : ℝ) 6 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_range_l1969_196988


namespace NUMINAMATH_CALUDE_mark_tree_count_l1969_196938

/-- Calculates the final number of trees after planting and removing sessions -/
def final_tree_count (x y : ℕ) (plant_rate remove_rate : ℕ) : ℤ :=
  let days : ℕ := y / plant_rate
  let removed : ℕ := days * remove_rate
  (x : ℤ) + (y : ℤ) - (removed : ℤ)

/-- Theorem stating the final number of trees after Mark's planting session -/
theorem mark_tree_count (x : ℕ) : final_tree_count x 12 2 3 = (x : ℤ) - 6 := by
  sorry

end NUMINAMATH_CALUDE_mark_tree_count_l1969_196938


namespace NUMINAMATH_CALUDE_tour_group_dish_choices_l1969_196962

/-- Represents the number of people in the tour group -/
def total_people : ℕ := 92

/-- Represents the number of different dish combinations -/
def dish_combinations : ℕ := 9

/-- Represents the minimum number of people who must choose the same combination -/
def min_same_choice : ℕ := total_people / dish_combinations + 1

theorem tour_group_dish_choices :
  ∃ (combination : Fin dish_combinations),
    (Finset.filter (λ person : Fin total_people =>
      person.val % dish_combinations = combination.val) (Finset.univ : Finset (Fin total_people))).card
    ≥ min_same_choice :=
sorry

end NUMINAMATH_CALUDE_tour_group_dish_choices_l1969_196962


namespace NUMINAMATH_CALUDE_tennis_tournament_result_l1969_196915

/-- Represents the number of participants with k points after m rounds in a tournament with 2^n participants -/
def f (n m k : ℕ) : ℕ := 2^(n - m) * (m.choose k)

/-- The number of participants in the tournament -/
def num_participants : ℕ := 254

/-- The number of rounds in the tournament -/
def num_rounds : ℕ := 8

/-- The number of points we're interested in -/
def target_points : ℕ := 5

theorem tennis_tournament_result :
  f 8 num_rounds target_points = 56 :=
sorry

#eval f 8 num_rounds target_points

end NUMINAMATH_CALUDE_tennis_tournament_result_l1969_196915


namespace NUMINAMATH_CALUDE_tea_store_theorem_l1969_196930

/-- Represents the number of ways to buy items from a tea store. -/
def teaStoreCombinations (cups saucers spoons : ℕ) : ℕ :=
  let cupSaucer := cups * saucers
  let cupSpoon := cups * spoons
  let saucerSpoon := saucers * spoons
  let all := cups * saucers * spoons
  cups + saucers + spoons + cupSaucer + cupSpoon + saucerSpoon + all

/-- Theorem stating the total number of combinations for a specific tea store inventory. -/
theorem tea_store_theorem :
  teaStoreCombinations 5 3 4 = 119 := by
  sorry

end NUMINAMATH_CALUDE_tea_store_theorem_l1969_196930


namespace NUMINAMATH_CALUDE_cubic_equation_root_l1969_196945

theorem cubic_equation_root (a b : ℚ) : 
  (3 - 5 * Real.sqrt 2)^3 + a * (3 - 5 * Real.sqrt 2)^2 + b * (3 - 5 * Real.sqrt 2) - 47 = 0 → 
  a = -199/41 := by
sorry

end NUMINAMATH_CALUDE_cubic_equation_root_l1969_196945


namespace NUMINAMATH_CALUDE_gecko_eggs_hatched_l1969_196928

/-- The number of eggs that actually hatch from a gecko's yearly egg-laying, given the total number of eggs, infertility rate, and calcification issue rate. -/
theorem gecko_eggs_hatched (total_eggs : ℕ) (infertility_rate : ℚ) (calcification_rate : ℚ) : 
  total_eggs = 30 →
  infertility_rate = 1/5 →
  calcification_rate = 1/3 →
  (total_eggs : ℚ) * (1 - infertility_rate) * (1 - calcification_rate) = 16 := by
  sorry

end NUMINAMATH_CALUDE_gecko_eggs_hatched_l1969_196928


namespace NUMINAMATH_CALUDE_rabbit_count_l1969_196905

/-- Calculates the number of rabbits given land dimensions and clearing rates -/
theorem rabbit_count (land_width : ℝ) (land_length : ℝ) (rabbit_clear_rate : ℝ) (days_to_clear : ℝ) : 
  land_width = 200 ∧ 
  land_length = 900 ∧ 
  rabbit_clear_rate = 10 ∧ 
  days_to_clear = 20 → 
  (land_width * land_length) / 9 / (rabbit_clear_rate * days_to_clear) = 100 := by
  sorry

#check rabbit_count

end NUMINAMATH_CALUDE_rabbit_count_l1969_196905


namespace NUMINAMATH_CALUDE_olivia_wednesday_hours_l1969_196996

/-- Calculates the number of hours Olivia worked on Wednesday -/
def wednesday_hours (hourly_rate : ℕ) (monday_hours : ℕ) (friday_hours : ℕ) (total_earnings : ℕ) : ℕ :=
  (total_earnings - hourly_rate * (monday_hours + friday_hours)) / hourly_rate

/-- Proves that Olivia worked 3 hours on Wednesday given the conditions -/
theorem olivia_wednesday_hours :
  wednesday_hours 9 4 6 117 = 3 := by
sorry

end NUMINAMATH_CALUDE_olivia_wednesday_hours_l1969_196996


namespace NUMINAMATH_CALUDE_grover_profit_l1969_196936

def number_of_boxes : ℕ := 3
def masks_per_box : ℕ := 20
def total_cost : ℚ := 15
def selling_price_per_mask : ℚ := 1/2

def total_masks : ℕ := number_of_boxes * masks_per_box
def total_revenue : ℚ := (total_masks : ℚ) * selling_price_per_mask
def profit : ℚ := total_revenue - total_cost

theorem grover_profit : profit = 15 := by
  sorry

end NUMINAMATH_CALUDE_grover_profit_l1969_196936


namespace NUMINAMATH_CALUDE_red_light_is_random_event_l1969_196925

/-- Definition of a random event -/
def is_random_event (event : Type) : Prop :=
  ∃ (outcome : event → Prop) (probability : event → ℝ),
    (∀ e : event, 0 ≤ probability e ∧ probability e ≤ 1) ∧
    (∀ e : event, outcome e ↔ probability e > 0)

/-- Representation of passing through an intersection with a traffic signal -/
inductive TrafficSignalEvent
| RedLight
| GreenLight
| YellowLight

/-- Theorem stating that encountering a red light at an intersection is a random event -/
theorem red_light_is_random_event :
  is_random_event TrafficSignalEvent :=
sorry

end NUMINAMATH_CALUDE_red_light_is_random_event_l1969_196925


namespace NUMINAMATH_CALUDE_sum_in_second_quadrant_l1969_196901

/-- Given two complex numbers z₁ and z₂, prove that their sum is in the second quadrant -/
theorem sum_in_second_quadrant (z₁ z₂ : ℂ) 
  (h₁ : z₁ = -3 + 4*I) (h₂ : z₂ = 2 - 3*I) : 
  let z := z₁ + z₂
  z.re < 0 ∧ z.im > 0 := by
  sorry

#check sum_in_second_quadrant

end NUMINAMATH_CALUDE_sum_in_second_quadrant_l1969_196901


namespace NUMINAMATH_CALUDE_greater_than_reciprocal_reciprocal_comparison_l1969_196942

theorem greater_than_reciprocal (x : ℝ) : Prop :=
  x ≠ 0 ∧ x > 1 / x

theorem reciprocal_comparison : 
  ¬ greater_than_reciprocal (-3/2) ∧
  ¬ greater_than_reciprocal (-1) ∧
  ¬ greater_than_reciprocal (1/3) ∧
  greater_than_reciprocal 2 ∧
  greater_than_reciprocal 3 := by
sorry

end NUMINAMATH_CALUDE_greater_than_reciprocal_reciprocal_comparison_l1969_196942


namespace NUMINAMATH_CALUDE_quotient_rational_l1969_196924

-- Define the set A as a subset of positive reals
def A : Set ℝ := {x : ℝ | x > 0}

-- Define the property that A is non-empty
axiom A_nonempty : Set.Nonempty A

-- Define the condition that for all a, b, c in A, ab + bc + ca is rational
axiom sum_rational (a b c : ℝ) (ha : a ∈ A) (hb : b ∈ A) (hc : c ∈ A) :
  ∃ (q : ℚ), (a * b + b * c + c * a : ℝ) = q

-- State the theorem to be proved
theorem quotient_rational (a b : ℝ) (ha : a ∈ A) (hb : b ∈ A) :
  ∃ (q : ℚ), (a / b : ℝ) = q :=
sorry

end NUMINAMATH_CALUDE_quotient_rational_l1969_196924


namespace NUMINAMATH_CALUDE_pepperoni_to_crust_ratio_is_one_to_three_l1969_196946

/-- Represents the calorie content of various food items and portions consumed --/
structure FoodCalories where
  lettuce : ℕ
  carrot : ℕ
  dressing : ℕ
  crust : ℕ
  cheese : ℕ
  saladPortion : ℚ
  pizzaPortion : ℚ
  totalConsumed : ℕ

/-- Calculates the ratio of pepperoni calories to crust calories --/
def pepperoniToCrustRatio (food : FoodCalories) : ℚ × ℚ :=
  sorry

/-- Theorem stating that given the conditions, the ratio of pepperoni to crust calories is 1:3 --/
theorem pepperoni_to_crust_ratio_is_one_to_three 
  (food : FoodCalories)
  (h1 : food.lettuce = 50)
  (h2 : food.carrot = 2 * food.lettuce)
  (h3 : food.dressing = 210)
  (h4 : food.crust = 600)
  (h5 : food.cheese = 400)
  (h6 : food.saladPortion = 1/4)
  (h7 : food.pizzaPortion = 1/5)
  (h8 : food.totalConsumed = 330) :
  pepperoniToCrustRatio food = (1, 3) :=
sorry

end NUMINAMATH_CALUDE_pepperoni_to_crust_ratio_is_one_to_three_l1969_196946


namespace NUMINAMATH_CALUDE_calculator_profit_l1969_196998

theorem calculator_profit : 
  let selling_price : ℝ := 64
  let profit_percentage : ℝ := 0.6
  let loss_percentage : ℝ := 0.2
  let cost_price1 : ℝ := selling_price / (1 + profit_percentage)
  let cost_price2 : ℝ := selling_price / (1 - loss_percentage)
  let total_cost : ℝ := cost_price1 + cost_price2
  let total_revenue : ℝ := 2 * selling_price
  total_revenue - total_cost = 8 := by
  sorry

end NUMINAMATH_CALUDE_calculator_profit_l1969_196998


namespace NUMINAMATH_CALUDE_optimal_square_perimeter_l1969_196958

/-- Given a wire of length 1 cut into two pieces to form a square and a circle,
    the perimeter of the square that minimizes the sum of their areas is π / (π + 4) -/
theorem optimal_square_perimeter :
  ∃ (x : ℝ), x > 0 ∧ x < 1 ∧
  (∀ (y : ℝ), y > 0 → y < 1 →
    x^2 / 16 + (1 - x)^2 / (4 * π) ≤ y^2 / 16 + (1 - y)^2 / (4 * π)) ∧
  x = π / (π + 4) := by
  sorry

end NUMINAMATH_CALUDE_optimal_square_perimeter_l1969_196958


namespace NUMINAMATH_CALUDE_simplify_and_evaluate_l1969_196955

theorem simplify_and_evaluate (x : ℝ) (h : x^2 + 4*x - 4 = 0) :
  3*(x-2)^2 - 6*(x+1)*(x-1) = 6 := by sorry

end NUMINAMATH_CALUDE_simplify_and_evaluate_l1969_196955


namespace NUMINAMATH_CALUDE_zinc_in_mixture_l1969_196913

/-- Given a mixture of zinc and copper with a ratio of 9:11 and a total weight of 74 kg,
    prove that the amount of zinc in the mixture is 33.3 kg. -/
theorem zinc_in_mixture (ratio_zinc : ℚ) (ratio_copper : ℚ) (total_weight : ℚ) :
  ratio_zinc = 9 →
  ratio_copper = 11 →
  total_weight = 74 →
  (ratio_zinc / (ratio_zinc + ratio_copper)) * total_weight = 33.3 := by
  sorry

end NUMINAMATH_CALUDE_zinc_in_mixture_l1969_196913


namespace NUMINAMATH_CALUDE_f_composition_half_equals_one_l1969_196980

-- Define the function f
def f (x : ℝ) : ℝ := |x - 1| - |x|

-- State the theorem
theorem f_composition_half_equals_one : f (f (1/2)) = 1 := by
  sorry

end NUMINAMATH_CALUDE_f_composition_half_equals_one_l1969_196980


namespace NUMINAMATH_CALUDE_isbn_problem_l1969_196937

/-- ISBN check digit calculation -/
def isbn_check_digit (A B C D E F G H I : ℕ) : ℕ :=
  let S := 10*A + 9*B + 8*C + 7*D + 6*E + 5*F + 4*G + 3*H + 2*I
  let r := S % 11
  if r = 0 then 0
  else if r = 1 then 10  -- Represented by 'x' in the problem
  else 11 - r

/-- The problem statement -/
theorem isbn_problem (y : ℕ) : 
  isbn_check_digit 9 6 2 y 7 0 7 0 1 = 5 → y = 7 := by
sorry

end NUMINAMATH_CALUDE_isbn_problem_l1969_196937


namespace NUMINAMATH_CALUDE_paint_usage_l1969_196929

theorem paint_usage (total_paint : ℚ) (first_week_fraction : ℚ) (second_week_fraction : ℚ) :
  total_paint = 360 →
  first_week_fraction = 1 / 9 →
  second_week_fraction = 1 / 5 →
  let first_week_usage := first_week_fraction * total_paint
  let remaining_paint := total_paint - first_week_usage
  let second_week_usage := second_week_fraction * remaining_paint
  first_week_usage + second_week_usage = 104 := by
  sorry

end NUMINAMATH_CALUDE_paint_usage_l1969_196929


namespace NUMINAMATH_CALUDE_sqrt_eight_and_one_ninth_l1969_196990

theorem sqrt_eight_and_one_ninth (x : ℝ) : 
  x = Real.sqrt (8 + 1 / 9) → x = Real.sqrt 73 / 3 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_eight_and_one_ninth_l1969_196990


namespace NUMINAMATH_CALUDE_art_club_students_l1969_196991

/-- The number of students in the art club -/
def num_students : ℕ := 15

/-- The number of artworks each student makes per quarter -/
def artworks_per_quarter : ℕ := 2

/-- The number of quarters in a school year -/
def quarters_per_year : ℕ := 4

/-- The total number of artworks collected in two school years -/
def total_artworks : ℕ := 240

/-- Theorem stating that the number of students in the art club is 15 -/
theorem art_club_students :
  num_students * artworks_per_quarter * quarters_per_year * 2 = total_artworks :=
by sorry

end NUMINAMATH_CALUDE_art_club_students_l1969_196991


namespace NUMINAMATH_CALUDE_valid_parameterization_l1969_196935

/-- A structure representing a vector parameterization of a line -/
structure VectorParam where
  x₀ : ℝ
  y₀ : ℝ
  dx : ℝ
  dy : ℝ

/-- Predicate to check if a vector parameterization represents the line y = 2x - 7 -/
def IsValidParam (p : VectorParam) : Prop :=
  p.y₀ = 2 * p.x₀ - 7 ∧ ∃ (k : ℝ), p.dx = k * 1 ∧ p.dy = k * 2

/-- Theorem stating the conditions for a valid parameterization of y = 2x - 7 -/
theorem valid_parameterization (p : VectorParam) :
  IsValidParam p ↔ 
  ∀ (t : ℝ), (p.y₀ + t * p.dy) = 2 * (p.x₀ + t * p.dx) - 7 :=
sorry

end NUMINAMATH_CALUDE_valid_parameterization_l1969_196935


namespace NUMINAMATH_CALUDE_min_value_reciprocal_product_l1969_196931

theorem min_value_reciprocal_product (a b : ℝ) 
  (ha : a > 0) (hb : b > 0) (h_sum : 2*a + b = 4) : 
  (∀ x y : ℝ, x > 0 → y > 0 → 2*x + y = 4 → 1/(x*y) ≥ 1/2) ∧ 
  (∃ x y : ℝ, x > 0 ∧ y > 0 ∧ 2*x + y = 4 ∧ 1/(x*y) = 1/2) :=
sorry

end NUMINAMATH_CALUDE_min_value_reciprocal_product_l1969_196931


namespace NUMINAMATH_CALUDE_magnitude_of_Z_l1969_196999

def Z : ℂ := Complex.mk 3 (-4)

theorem magnitude_of_Z : Complex.abs Z = 5 := by
  sorry

end NUMINAMATH_CALUDE_magnitude_of_Z_l1969_196999


namespace NUMINAMATH_CALUDE_division_remainder_l1969_196932

theorem division_remainder (dividend : ℕ) (divisor : ℕ) (quotient : ℕ) (remainder : ℕ) 
  (h1 : dividend = 22)
  (h2 : divisor = 3)
  (h3 : quotient = 7)
  (h4 : dividend = divisor * quotient + remainder) :
  remainder = 1 := by
  sorry

end NUMINAMATH_CALUDE_division_remainder_l1969_196932


namespace NUMINAMATH_CALUDE_simplify_sqrt_difference_l1969_196918

theorem simplify_sqrt_difference : 
  (Real.sqrt 704 / Real.sqrt 64) - (Real.sqrt 300 / Real.sqrt 75) = Real.sqrt 11 - 2 := by
  sorry

end NUMINAMATH_CALUDE_simplify_sqrt_difference_l1969_196918


namespace NUMINAMATH_CALUDE_average_of_quadratic_roots_l1969_196965

theorem average_of_quadratic_roots (c : ℝ) 
  (h : ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ 3 * x₁^2 - 6 * x₁ + c = 0 ∧ 3 * x₂^2 - 6 * x₂ + c = 0) :
  ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ 
    3 * x₁^2 - 6 * x₁ + c = 0 ∧ 
    3 * x₂^2 - 6 * x₂ + c = 0 ∧
    (x₁ + x₂) / 2 = 1 :=
by sorry

end NUMINAMATH_CALUDE_average_of_quadratic_roots_l1969_196965


namespace NUMINAMATH_CALUDE_tomatoes_left_after_yesterday_l1969_196941

theorem tomatoes_left_after_yesterday (initial_tomatoes picked_yesterday : ℕ) : 
  initial_tomatoes = 160 → picked_yesterday = 56 → initial_tomatoes - picked_yesterday = 104 := by
  sorry

end NUMINAMATH_CALUDE_tomatoes_left_after_yesterday_l1969_196941


namespace NUMINAMATH_CALUDE_line_equation_proof_l1969_196950

-- Define the two given lines
def line1 (x y : ℝ) : Prop := 3 * x + 2 * y - 5 = 0
def line2 (x y : ℝ) : Prop := 2 * x + 3 * y - 5 = 0

-- Define the intersection point
def intersection_point : ℝ × ℝ := (1, 1)

-- Define the y-intercept
def y_intercept : ℝ := -5

-- Define the equation of the line we want to prove
def target_line (x y : ℝ) : Prop := 6 * x - y - 5 = 0

-- Theorem statement
theorem line_equation_proof :
  ∀ (x y : ℝ),
    (x, y) = intersection_point →
    line1 x y ∧ line2 x y →
    target_line 0 y_intercept →
    target_line x y :=
by
  sorry

end NUMINAMATH_CALUDE_line_equation_proof_l1969_196950


namespace NUMINAMATH_CALUDE_x_range_l1969_196975

theorem x_range (x : ℝ) : 
  let p := x^2 - 2*x - 3 ≥ 0
  let q := 0 < x ∧ x < 4
  (¬q) → (p ∨ q) → (x ≤ -1 ∨ x ≥ 4) :=
by
  sorry

end NUMINAMATH_CALUDE_x_range_l1969_196975


namespace NUMINAMATH_CALUDE_problem_solution_l1969_196981

/-- Proposition p -/
def p (m x : ℝ) : Prop := m * x + 1 > 0

/-- Proposition q -/
def q (x : ℝ) : Prop := (3 * x - 1) * (x + 2) < 0

theorem problem_solution (m : ℝ) (hm : m > 0) :
  (∃ a b : ℝ, a < b ∧ 
    (m = 1 → 
      (∀ x : ℝ, p m x ∧ q x ↔ a < x ∧ x < b) ∧
      a = -1 ∧ b = 1/3)) ∧
  (∀ x : ℝ, q x → p m x) ∧ 
  (∃ x : ℝ, p m x ∧ ¬q x) →
  0 < m ∧ m ≤ 1/2 :=
sorry

end NUMINAMATH_CALUDE_problem_solution_l1969_196981


namespace NUMINAMATH_CALUDE_two_different_buttons_l1969_196908

/-- Represents the size of a button -/
inductive Size
| Big
| Small

/-- Represents the color of a button -/
inductive Color
| White
| Black

/-- Represents a button with a size and color -/
structure Button :=
  (size : Size)
  (color : Color)

/-- A set of buttons satisfying the given conditions -/
structure ButtonSet :=
  (buttons : Set Button)
  (has_big : ∃ b ∈ buttons, b.size = Size.Big)
  (has_small : ∃ b ∈ buttons, b.size = Size.Small)
  (has_white : ∃ b ∈ buttons, b.color = Color.White)
  (has_black : ∃ b ∈ buttons, b.color = Color.Black)

/-- Theorem stating that there exist two buttons with different size and color -/
theorem two_different_buttons (bs : ButtonSet) :
  ∃ (b1 b2 : Button), b1 ∈ bs.buttons ∧ b2 ∈ bs.buttons ∧
  b1.size ≠ b2.size ∧ b1.color ≠ b2.color :=
sorry

end NUMINAMATH_CALUDE_two_different_buttons_l1969_196908


namespace NUMINAMATH_CALUDE_ball_probability_6_l1969_196971

def ball_probability (n : ℕ) : ℚ :=
  match n with
  | 0 => 1
  | 1 => 0
  | k + 2 => (1 / 3) * (1 - ball_probability (k + 1))

theorem ball_probability_6 :
  ball_probability 6 = 61 / 243 := by
  sorry

end NUMINAMATH_CALUDE_ball_probability_6_l1969_196971


namespace NUMINAMATH_CALUDE_system_solution_ratio_l1969_196994

theorem system_solution_ratio (x y a b : ℝ) 
  (h1 : 6 * x - 4 * y = a)
  (h2 : 6 * y - 9 * x = b)
  (hx : x ≠ 0)
  (hy : y ≠ 0)
  (hb : b ≠ 0) :
  a / b = -2 / 3 := by
sorry

end NUMINAMATH_CALUDE_system_solution_ratio_l1969_196994


namespace NUMINAMATH_CALUDE_article_sale_loss_l1969_196920

theorem article_sale_loss (cost : ℝ) (profit_rate : ℝ) (discount_rate : ℝ) : 
  profit_rate = 0.425 → 
  discount_rate = 2/3 →
  let original_price := cost * (1 + profit_rate)
  let discounted_price := original_price * discount_rate
  let loss := cost - discounted_price
  let loss_rate := loss / cost
  loss_rate = 0.05 := by
  sorry

end NUMINAMATH_CALUDE_article_sale_loss_l1969_196920


namespace NUMINAMATH_CALUDE_police_emergency_number_prime_divisor_l1969_196914

theorem police_emergency_number_prime_divisor (k : ℕ) :
  ∃ (p : ℕ), Prime p ∧ p > 7 ∧ p ∣ (1000 * k + 133) := by
  sorry

end NUMINAMATH_CALUDE_police_emergency_number_prime_divisor_l1969_196914


namespace NUMINAMATH_CALUDE_correct_operation_result_l1969_196977

theorem correct_operation_result (x : ℝ) : 
  (x / 8 - 12 = 32) → (x * 8 + 12 = 2828) := by sorry

end NUMINAMATH_CALUDE_correct_operation_result_l1969_196977


namespace NUMINAMATH_CALUDE_polynomial_division_remainder_l1969_196974

theorem polynomial_division_remainder : ∃ q : Polynomial ℝ, 
  2 * X^4 + 10 * X^3 - 35 * X^2 - 40 * X + 12 = 
  (X^2 + 7 * X - 8) * q + (-135 * X + 84) := by
  sorry

end NUMINAMATH_CALUDE_polynomial_division_remainder_l1969_196974


namespace NUMINAMATH_CALUDE_marathon_theorem_l1969_196970

def marathon_problem (total_distance : ℝ) (day1_percent : ℝ) (day2_percent : ℝ) : ℝ :=
  let day1_distance := total_distance * day1_percent
  let remaining_after_day1 := total_distance - day1_distance
  let day2_distance := remaining_after_day1 * day2_percent
  let day3_distance := total_distance - day1_distance - day2_distance
  day3_distance

theorem marathon_theorem :
  marathon_problem 70 0.2 0.5 = 28 := by
  sorry

end NUMINAMATH_CALUDE_marathon_theorem_l1969_196970


namespace NUMINAMATH_CALUDE_expansion_has_four_terms_l1969_196911

/-- The expression after substituting 2x for the asterisk and expanding -/
def expanded_expression (x : ℝ) : ℝ := x^6 + x^4 + 4*x^2 + 4

/-- The original expression with the asterisk replaced by 2x -/
def original_expression (x : ℝ) : ℝ := (x^3 - 2)^2 + (x^2 + 2*x)^2

theorem expansion_has_four_terms :
  ∀ x : ℝ, original_expression x = expanded_expression x ∧ 
  (∃ a b c d : ℝ, expanded_expression x = a*x^6 + b*x^4 + c*x^2 + d ∧ 
   a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0 ∧ d ≠ 0) :=
by sorry

end NUMINAMATH_CALUDE_expansion_has_four_terms_l1969_196911


namespace NUMINAMATH_CALUDE_games_after_increase_l1969_196903

def initial_games : ℕ := 7
def increase_percentage : ℚ := 30 / 100

theorem games_after_increase :
  let new_games := Int.floor (increase_percentage * initial_games)
  initial_games + new_games = 9 := by
  sorry

end NUMINAMATH_CALUDE_games_after_increase_l1969_196903


namespace NUMINAMATH_CALUDE_f_shifted_f_identity_l1969_196982

-- Define the function f
def f : ℝ → ℝ := fun x => x^2 + 1

-- State the theorem
theorem f_shifted (x : ℝ) : f (x - 1) = x^2 - 2*x + 2 := by
  sorry

-- Prove that f(x) = x^2 + 1
theorem f_identity (x : ℝ) : f x = x^2 + 1 := by
  sorry

end NUMINAMATH_CALUDE_f_shifted_f_identity_l1969_196982
