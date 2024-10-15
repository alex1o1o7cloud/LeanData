import Mathlib

namespace NUMINAMATH_CALUDE_system_solution_l3974_397452

theorem system_solution : 
  ∀ x y : ℝ, 
  (x + y + Real.sqrt (x * y) = 28 ∧ x^2 + y^2 + x * y = 336) ↔ 
  ((x = 4 ∧ y = 16) ∨ (x = 16 ∧ y = 4)) := by
sorry

end NUMINAMATH_CALUDE_system_solution_l3974_397452


namespace NUMINAMATH_CALUDE_complex_modulus_l3974_397412

theorem complex_modulus (z : ℂ) : (1 + I) * z = 2 * I → Complex.abs z = Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_complex_modulus_l3974_397412


namespace NUMINAMATH_CALUDE_max_intersections_for_given_points_l3974_397451

/-- The maximum number of intersection points in the first quadrant -/
def max_intersection_points (x_points y_points : ℕ) : ℕ :=
  (x_points * y_points * (x_points - 1) * (y_points - 1)) / 4

/-- Theorem stating the maximum number of intersection points for the given conditions -/
theorem max_intersections_for_given_points :
  max_intersection_points 5 3 = 30 := by sorry

end NUMINAMATH_CALUDE_max_intersections_for_given_points_l3974_397451


namespace NUMINAMATH_CALUDE_factorial_difference_l3974_397435

theorem factorial_difference : Nat.factorial 10 - Nat.factorial 9 = 3265920 := by
  sorry

end NUMINAMATH_CALUDE_factorial_difference_l3974_397435


namespace NUMINAMATH_CALUDE_other_root_of_quadratic_l3974_397402

theorem other_root_of_quadratic (k : ℝ) : 
  (1 : ℝ) ^ 2 + k * 1 - 2 = 0 → 
  ∃ (x : ℝ), x ≠ 1 ∧ x ^ 2 + k * x - 2 = 0 ∧ x = -2 :=
by sorry

end NUMINAMATH_CALUDE_other_root_of_quadratic_l3974_397402


namespace NUMINAMATH_CALUDE_ab_bc_ratio_is_two_plus_sqrt_three_l3974_397415

-- Define the quadrilateral ABCD
structure Quadrilateral (A B C D : ℝ × ℝ) : Prop where
  right_angle_B : (B.1 - A.1) * (C.1 - B.1) + (B.2 - A.2) * (C.2 - B.2) = 0
  right_angle_C : (C.1 - B.1) * (D.1 - C.1) + (C.2 - B.2) * (D.2 - C.2) = 0

-- Define similarity of triangles
def similar_triangles (A B C D E F : ℝ × ℝ) : Prop :=
  ∃ k > 0, (B.1 - A.1)^2 + (B.2 - A.2)^2 = k * ((E.1 - D.1)^2 + (E.2 - D.2)^2) ∧
            (C.1 - B.1)^2 + (C.2 - B.2)^2 = k * ((F.1 - E.1)^2 + (F.2 - E.2)^2) ∧
            (A.1 - C.1)^2 + (A.2 - C.2)^2 = k * ((D.1 - F.1)^2 + (D.2 - F.2)^2)

-- Define the area of a triangle
def triangle_area (A B C : ℝ × ℝ) : ℝ :=
  0.5 * abs ((B.1 - A.1) * (C.2 - A.2) - (C.1 - A.1) * (B.2 - A.2))

-- Main theorem
theorem ab_bc_ratio_is_two_plus_sqrt_three
  (A B C D E : ℝ × ℝ)
  (h_quad : Quadrilateral A B C D)
  (h_sim_ABC_BCD : similar_triangles A B C B C D)
  (h_AB_gt_BC : (A.1 - B.1)^2 + (A.2 - B.2)^2 > (B.1 - C.1)^2 + (B.2 - C.2)^2)
  (h_E_interior : ∃ t u : ℝ, 0 < t ∧ t < 1 ∧ 0 < u ∧ u < 1 ∧
    E = (t * A.1 + (1 - t) * C.1, u * B.2 + (1 - u) * D.2))
  (h_sim_ABC_CEB : similar_triangles A B C C E B)
  (h_area_ratio : triangle_area A E D = 25 * triangle_area C E B) :
  Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) / Real.sqrt ((B.1 - C.1)^2 + (B.2 - C.2)^2) = 2 + Real.sqrt 3 :=
sorry

end NUMINAMATH_CALUDE_ab_bc_ratio_is_two_plus_sqrt_three_l3974_397415


namespace NUMINAMATH_CALUDE_special_school_student_count_l3974_397404

/-- Represents a school for deaf and blind students -/
structure School where
  deaf_students : ℕ
  blind_students : ℕ

/-- The total number of students in the school -/
def total_students (s : School) : ℕ := s.deaf_students + s.blind_students

/-- Theorem: Given a school where the deaf student population is three times 
    the size of the blind student population, and the number of deaf students 
    is 180, the total number of students is 240. -/
theorem special_school_student_count :
  ∀ (s : School),
  s.deaf_students = 180 →
  s.deaf_students = 3 * s.blind_students →
  total_students s = 240 := by
  sorry

end NUMINAMATH_CALUDE_special_school_student_count_l3974_397404


namespace NUMINAMATH_CALUDE_triangle_with_arithmetic_progression_sides_and_perimeter_15_l3974_397439

def is_arithmetic_progression (a b c : ℕ) : Prop :=
  b - a = c - b

def is_valid_triangle (a b c : ℕ) : Prop :=
  a + b > c ∧ a + c > b ∧ b + c > a

theorem triangle_with_arithmetic_progression_sides_and_perimeter_15 :
  ∀ a b c : ℕ,
    a + b + c = 15 →
    is_arithmetic_progression a b c →
    is_valid_triangle a b c →
    ((a = 5 ∧ b = 5 ∧ c = 5) ∨
     (a = 4 ∧ b = 5 ∧ c = 6) ∨
     (a = 3 ∧ b = 5 ∧ c = 7)) :=
by sorry

end NUMINAMATH_CALUDE_triangle_with_arithmetic_progression_sides_and_perimeter_15_l3974_397439


namespace NUMINAMATH_CALUDE_fibonacci_sum_convergence_l3974_397460

def fibonacci : ℕ → ℕ
  | 0 => 0
  | 1 => 1
  | (n + 2) => fibonacci (n + 1) + fibonacci n

theorem fibonacci_sum_convergence :
  let S : ℝ := ∑' n, (fibonacci n : ℝ) / 5^n
  S = 5/19 := by sorry

end NUMINAMATH_CALUDE_fibonacci_sum_convergence_l3974_397460


namespace NUMINAMATH_CALUDE_remainder_problem_l3974_397453

theorem remainder_problem : (98 * 103 + 7) % 12 = 1 := by sorry

end NUMINAMATH_CALUDE_remainder_problem_l3974_397453


namespace NUMINAMATH_CALUDE_greatest_divisor_with_remainders_l3974_397449

theorem greatest_divisor_with_remainders (a b r1 r2 : ℕ) (ha : a = 6215) (hb : b = 7373) (hr1 : r1 = 23) (hr2 : r2 = 29) :
  Nat.gcd (a - r1) (b - r2) = 96 := by
  sorry

end NUMINAMATH_CALUDE_greatest_divisor_with_remainders_l3974_397449


namespace NUMINAMATH_CALUDE_min_both_composers_l3974_397488

theorem min_both_composers (total : ℕ) (beethoven : ℕ) (chopin : ℕ) 
  (h1 : total = 130) 
  (h2 : beethoven = 110) 
  (h3 : chopin = 90) 
  (h4 : beethoven ≤ total) 
  (h5 : chopin ≤ total) : 
  (beethoven + chopin - total : ℤ) ≥ 70 := by
  sorry

end NUMINAMATH_CALUDE_min_both_composers_l3974_397488


namespace NUMINAMATH_CALUDE_denominator_problem_l3974_397475

theorem denominator_problem (numerator denominator : ℤ) : 
  denominator = numerator - 4 →
  numerator + 6 = 3 * denominator →
  denominator = 5 := by
sorry

end NUMINAMATH_CALUDE_denominator_problem_l3974_397475


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l3974_397405

/-- An arithmetic sequence is a sequence where the difference between each consecutive term is constant. -/
def is_arithmetic_sequence (a : ℕ → ℚ) : Prop :=
  ∃ d : ℚ, ∀ n : ℕ, a (n + 1) = a n + d

/-- Given an arithmetic sequence a with a₂ = 3 and a₅ + a₇ = 10, prove that a₁ + a₁₀ = 9.5 -/
theorem arithmetic_sequence_sum (a : ℕ → ℚ) 
  (h_arith : is_arithmetic_sequence a) 
  (h_a2 : a 2 = 3) 
  (h_sum : a 5 + a 7 = 10) : 
  a 1 + a 10 = 9.5 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l3974_397405


namespace NUMINAMATH_CALUDE_grey_area_ratio_l3974_397463

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a square piece of paper -/
structure Square where
  sideLength : ℝ
  a : Point
  b : Point
  c : Point
  d : Point

/-- Represents a kite shape -/
structure Kite where
  a : Point
  e : Point
  c : Point
  f : Point

/-- Function to fold the paper along a line -/
def foldPaper (s : Square) (p : Point) : Kite :=
  sorry

/-- Theorem stating the ratio of grey area to total area of the kite -/
theorem grey_area_ratio (s : Square) (e f : Point) :
  let k := foldPaper s e
  let k' := foldPaper s f
  let greyArea := sorry
  let totalArea := sorry
  greyArea / totalArea = 1 / Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_grey_area_ratio_l3974_397463


namespace NUMINAMATH_CALUDE_original_average_age_proof_l3974_397491

theorem original_average_age_proof (N : ℕ) (A : ℝ) : 
  A = 50 →
  (N * A + 12 * 32) / (N + 12) = 46 →
  A = 50 := by
sorry

end NUMINAMATH_CALUDE_original_average_age_proof_l3974_397491


namespace NUMINAMATH_CALUDE_boys_on_playground_l3974_397459

/-- The number of boys on a playground, given the total number of children and the number of girls. -/
def number_of_boys (total_children : ℕ) (number_of_girls : ℕ) : ℕ :=
  total_children - number_of_girls

/-- Theorem stating that the number of boys on the playground is 40. -/
theorem boys_on_playground : number_of_boys 117 77 = 40 := by
  sorry

end NUMINAMATH_CALUDE_boys_on_playground_l3974_397459


namespace NUMINAMATH_CALUDE_greatest_n_no_substring_divisible_by_9_l3974_397420

-- Define a function to check if a number is divisible by 9
def divisible_by_9 (n : ℕ) : Prop := n % 9 = 0

-- Define a function to get all integer substrings of a number
def integer_substrings (n : ℕ) : List ℕ := sorry

-- Define the property that no integer substring is divisible by 9
def no_substring_divisible_by_9 (n : ℕ) : Prop :=
  ∀ m ∈ integer_substrings n, ¬(divisible_by_9 m)

-- State the theorem
theorem greatest_n_no_substring_divisible_by_9 :
  (∀ k > 88888888, ¬(no_substring_divisible_by_9 k)) ∧
  (no_substring_divisible_by_9 88888888) :=
sorry

end NUMINAMATH_CALUDE_greatest_n_no_substring_divisible_by_9_l3974_397420


namespace NUMINAMATH_CALUDE_fixed_point_and_parabola_l3974_397426

/-- The fixed point P that the line passes through for all values of a -/
def P : ℝ × ℝ := (2, -8)

/-- The line equation for any real number a -/
def line_equation (a x y : ℝ) : Prop :=
  (2*a + 3)*x + y - 4*a + 2 = 0

/-- The parabola equation with y-axis as the axis of symmetry -/
def parabola_equation_y (x y : ℝ) : Prop :=
  y^2 = 32*x

/-- The parabola equation with x-axis as the axis of symmetry -/
def parabola_equation_x (x y : ℝ) : Prop :=
  x^2 = -1/2*y

theorem fixed_point_and_parabola :
  (∀ a : ℝ, line_equation a P.1 P.2) ∧
  (parabola_equation_y P.1 P.2 ∨ parabola_equation_x P.1 P.2) :=
sorry

end NUMINAMATH_CALUDE_fixed_point_and_parabola_l3974_397426


namespace NUMINAMATH_CALUDE_equation_solution_l3974_397496

theorem equation_solution :
  let y : ℚ := 20 / 7
  2 / y + (3 / y) / (6 / y) = 1.2 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l3974_397496


namespace NUMINAMATH_CALUDE_sum_of_100th_terms_l3974_397444

/-- Given two arithmetic sequences {a_n} and {b_n} satisfying certain conditions,
    prove that the sum of their 100th terms is 383. -/
theorem sum_of_100th_terms (a b : ℕ → ℝ) : 
  (∀ n m : ℕ, a (n + 1) - a n = a (m + 1) - a m) →  -- a_n is arithmetic
  (∀ n m : ℕ, b (n + 1) - b n = b (m + 1) - b m) →  -- b_n is arithmetic
  a 5 + b 5 = 3 →
  a 9 + b 9 = 19 →
  a 100 + b 100 = 383 := by
sorry

end NUMINAMATH_CALUDE_sum_of_100th_terms_l3974_397444


namespace NUMINAMATH_CALUDE_max_product_representation_l3974_397462

def representation_sum (n : ℕ) : List ℕ → Prop :=
  λ l => l.sum = n ∧ l.all (· > 0)

theorem max_product_representation (n : ℕ) :
  ∃ (l : List ℕ), representation_sum 2015 l ∧
    ∀ (m : List ℕ), representation_sum 2015 m →
      l.prod ≥ m.prod :=
by
  sorry

#check max_product_representation 2015

end NUMINAMATH_CALUDE_max_product_representation_l3974_397462


namespace NUMINAMATH_CALUDE_mark_parking_tickets_l3974_397428

theorem mark_parking_tickets (total_tickets : ℕ) (sarah_speeding : ℕ)
  (h1 : total_tickets = 24)
  (h2 : sarah_speeding = 6) :
  ∃ (mark_parking sarah_parking : ℕ),
    mark_parking = 2 * sarah_parking ∧
    total_tickets = sarah_parking + mark_parking + 2 * sarah_speeding ∧
    mark_parking = 8 := by
  sorry

end NUMINAMATH_CALUDE_mark_parking_tickets_l3974_397428


namespace NUMINAMATH_CALUDE_sum_of_odd_coefficients_l3974_397441

theorem sum_of_odd_coefficients (a a₁ a₂ a₃ a₄ a₅ : ℤ) :
  (∀ x, (2*x + 1)^5 = a + a₁*x + a₂*x^2 + a₃*x^3 + a₄*x^4 + a₅*x^5) →
  a₁ + a₃ + a₅ = 122 := by
sorry

end NUMINAMATH_CALUDE_sum_of_odd_coefficients_l3974_397441


namespace NUMINAMATH_CALUDE_f_neg_one_eq_neg_two_l3974_397407

/- Define an odd function f -/
def f (x : ℝ) : ℝ := sorry

/- State the properties of f -/
axiom f_odd : ∀ x, f (-x) = -f x
axiom f_positive : ∀ x > 0, f x = x^2 + 1/x

/- Theorem to prove -/
theorem f_neg_one_eq_neg_two : f (-1) = -2 := by sorry

end NUMINAMATH_CALUDE_f_neg_one_eq_neg_two_l3974_397407


namespace NUMINAMATH_CALUDE_benton_school_earnings_l3974_397440

/-- Represents the total earnings of students from a school -/
def school_earnings (students : ℕ) (days : ℕ) (daily_wage : ℚ) : ℚ :=
  students * days * daily_wage

/-- Calculates the daily wage per student given the total amount and total student-days -/
def calculate_daily_wage (total_amount : ℚ) (total_student_days : ℕ) : ℚ :=
  total_amount / total_student_days

theorem benton_school_earnings :
  let adams_students : ℕ := 4
  let adams_days : ℕ := 4
  let benton_students : ℕ := 5
  let benton_days : ℕ := 6
  let camden_students : ℕ := 6
  let camden_days : ℕ := 7
  let total_amount : ℚ := 780

  let total_student_days : ℕ := 
    adams_students * adams_days + 
    benton_students * benton_days + 
    camden_students * camden_days

  let daily_wage : ℚ := calculate_daily_wage total_amount total_student_days

  let benton_earnings : ℚ := school_earnings benton_students benton_days daily_wage

  ⌊benton_earnings⌋ = 266 :=
by sorry

end NUMINAMATH_CALUDE_benton_school_earnings_l3974_397440


namespace NUMINAMATH_CALUDE_mary_bought_48_cards_l3974_397411

/-- Calculates the number of baseball cards Mary bought -/
def cards_mary_bought (initial_cards : ℕ) (torn_cards : ℕ) (cards_from_fred : ℕ) (final_total : ℕ) : ℕ :=
  final_total - (initial_cards - torn_cards + cards_from_fred)

/-- Proves that Mary bought 48 baseball cards -/
theorem mary_bought_48_cards : cards_mary_bought 18 8 26 84 = 48 := by
  sorry

#eval cards_mary_bought 18 8 26 84

end NUMINAMATH_CALUDE_mary_bought_48_cards_l3974_397411


namespace NUMINAMATH_CALUDE_range_of_half_difference_l3974_397433

theorem range_of_half_difference (α β : Real) 
  (h1 : -π/2 ≤ α) (h2 : α < β) (h3 : β ≤ π/2) :
  ∀ x, x ∈ Set.Icc (-π/2) 0 ↔ ∃ α' β', -π/2 ≤ α' ∧ α' < β' ∧ β' ≤ π/2 ∧ x = (α' - β') / 2 :=
by sorry

end NUMINAMATH_CALUDE_range_of_half_difference_l3974_397433


namespace NUMINAMATH_CALUDE_g_properties_imply_g_50_l3974_397410

noncomputable def g (p q r s x : ℝ) : ℝ := (p * x + q) / (r * x + s)

theorem g_properties_imply_g_50 (p q r s : ℝ) 
  (hp : p ≠ 0) (hq : q ≠ 0) (hr : r ≠ 0) (hs : s ≠ 0) :
  g p q r s 23 = 23 →
  g p q r s 101 = 101 →
  (∀ x : ℝ, x ≠ -s/r → g p q r s (g p q r s x) = x) →
  g p q r s 50 = -61 := by sorry

end NUMINAMATH_CALUDE_g_properties_imply_g_50_l3974_397410


namespace NUMINAMATH_CALUDE_solution_is_correct_l3974_397429

/-- The imaginary unit i such that i^2 = -1 -/
noncomputable def i : ℂ := Complex.I

/-- The equation to be solved -/
def equation (z : ℂ) : Prop := 2 * z + (5 - 3 * i) = 6 + 11 * i

/-- The theorem stating that 1/2 + 7i is the solution to the equation -/
theorem solution_is_correct : equation (1/2 + 7 * i) := by
  sorry

end NUMINAMATH_CALUDE_solution_is_correct_l3974_397429


namespace NUMINAMATH_CALUDE_hyperbola_asymptote_l3974_397473

/-- The asymptote of a hyperbola with specific properties -/
theorem hyperbola_asymptote (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  let C : ℝ → ℝ → Prop := λ x y => x^2 / a^2 - y^2 / b^2 = 1
  let F₁ : ℝ × ℝ := (-c, 0)
  let F₂ : ℝ × ℝ := (c, 0)
  let asymptote_parallel : ℝ → ℝ → Prop := λ x y => y = (b / a) * (x - c)
  ∃ (P : ℝ × ℝ), 
    C P.1 P.2 ∧ 
    asymptote_parallel P.1 P.2 ∧
    ((P.1 + c) * (P.1 - c) + P.2^2 = 0) →
    (∀ (x y : ℝ), y = 2 * x ∨ y = -2 * x ↔ x^2 / a^2 - y^2 / b^2 = 0) :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_asymptote_l3974_397473


namespace NUMINAMATH_CALUDE_total_fruits_l3974_397493

theorem total_fruits (total_baskets : ℕ) 
                     (apple_baskets orange_baskets : ℕ) 
                     (apples_per_basket oranges_per_basket pears_per_basket : ℕ) : 
  total_baskets = 127 →
  apple_baskets = 79 →
  orange_baskets = 30 →
  apples_per_basket = 75 →
  oranges_per_basket = 143 →
  pears_per_basket = 56 →
  (apple_baskets * apples_per_basket + 
   orange_baskets * oranges_per_basket + 
   (total_baskets - apple_baskets - orange_baskets) * pears_per_basket) = 11223 :=
by
  sorry

#check total_fruits

end NUMINAMATH_CALUDE_total_fruits_l3974_397493


namespace NUMINAMATH_CALUDE_sum_of_squares_l3974_397431

theorem sum_of_squares (a b : ℝ) (h1 : a - b = 3) (h2 : a * b = 13) : a^2 + b^2 = 35 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_squares_l3974_397431


namespace NUMINAMATH_CALUDE_min_value_theorem_min_value_is_four_l3974_397469

theorem min_value_theorem (x : ℝ) (h : x ≥ 2) :
  (∀ y : ℝ, y ≥ 2 → x + 4/x ≤ y + 4/y) ↔ x = 2 :=
by sorry

theorem min_value_is_four (x : ℝ) (h : x ≥ 2) :
  x + 4/x ≥ 4 :=
by sorry

end NUMINAMATH_CALUDE_min_value_theorem_min_value_is_four_l3974_397469


namespace NUMINAMATH_CALUDE_craftsman_earnings_solution_l3974_397422

def craftsman_earnings (hours_worked : ℕ) (wage_A wage_B : ℚ) : Prop :=
  let earnings_A := hours_worked * wage_A
  let earnings_B := hours_worked * wage_B
  wage_A ≠ wage_B ∧
  (hours_worked - 1) * wage_A = 720 ∧
  (hours_worked - 5) * wage_B = 800 ∧
  (hours_worked - 1) * wage_B - (hours_worked - 5) * wage_A = 360 ∧
  earnings_A = 750 ∧
  earnings_B = 1000

theorem craftsman_earnings_solution :
  ∃ (hours_worked : ℕ) (wage_A wage_B : ℚ),
    craftsman_earnings hours_worked wage_A wage_B :=
by
  sorry

end NUMINAMATH_CALUDE_craftsman_earnings_solution_l3974_397422


namespace NUMINAMATH_CALUDE_total_pictures_correct_l3974_397455

/-- The number of pictures Bianca uploaded to Facebook -/
def total_pictures : ℕ := 33

/-- The number of pictures in the first album -/
def first_album_pictures : ℕ := 27

/-- The number of additional albums -/
def additional_albums : ℕ := 3

/-- The number of pictures in each additional album -/
def pictures_per_additional_album : ℕ := 2

/-- Theorem stating that the total number of pictures is correct -/
theorem total_pictures_correct : 
  total_pictures = first_album_pictures + additional_albums * pictures_per_additional_album := by
  sorry

end NUMINAMATH_CALUDE_total_pictures_correct_l3974_397455


namespace NUMINAMATH_CALUDE_unique_age_sum_of_digits_l3974_397474

theorem unique_age_sum_of_digits : ∃! y : ℕ,
  1900 ≤ y ∧ y < 2000 ∧
  1988 - y = 22 ∧
  1988 - y = (y / 1000) + ((y / 100) % 10) + ((y / 10) % 10) + (y % 10) :=
by sorry

end NUMINAMATH_CALUDE_unique_age_sum_of_digits_l3974_397474


namespace NUMINAMATH_CALUDE_sixth_term_is_geometric_mean_l3974_397438

/-- An arithmetic sequence -/
def ArithmeticSequence (a : ℕ → ℝ) : Prop :=
  ∃ (d : ℝ), ∀ n : ℕ, a (n + 1) = a n + d

/-- The second term is the geometric mean of the first and fourth terms -/
def SecondTermIsGeometricMean (a : ℕ → ℝ) : Prop :=
  a 2 = Real.sqrt (a 1 * a 4)

theorem sixth_term_is_geometric_mean
  (a : ℕ → ℝ)
  (h_arith : ArithmeticSequence a)
  (h_second : SecondTermIsGeometricMean a) :
  a 6 = Real.sqrt (a 4 * a 9) :=
by sorry

end NUMINAMATH_CALUDE_sixth_term_is_geometric_mean_l3974_397438


namespace NUMINAMATH_CALUDE_correct_paint_time_equation_l3974_397477

/-- Represents the time needed for three people to paint a room together, given their individual rates and a break time. -/
def paint_time (rate1 rate2 rate3 break_time : ℝ) (t : ℝ) : Prop :=
  (1 / rate1 + 1 / rate2 + 1 / rate3) * (t - break_time) = 1

/-- Theorem stating that the equation correctly represents the painting time for Doug, Dave, and Ralph. -/
theorem correct_paint_time_equation :
  ∀ t : ℝ, paint_time 6 8 12 1.5 t ↔ (1/6 + 1/8 + 1/12) * (t - 1.5) = 1 :=
by sorry

end NUMINAMATH_CALUDE_correct_paint_time_equation_l3974_397477


namespace NUMINAMATH_CALUDE_quadratic_equation_solution_l3974_397481

theorem quadratic_equation_solution : ∃ x₁ x₂ : ℝ, 
  (x₁ = 1 + Real.sqrt 6 ∧ x₁^2 - 2*x₁ - 5 = 0) ∧
  (x₂ = 1 - Real.sqrt 6 ∧ x₂^2 - 2*x₂ - 5 = 0) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_solution_l3974_397481


namespace NUMINAMATH_CALUDE_min_transportation_cost_l3974_397413

/-- Transportation problem between two cities and two towns -/
structure TransportationProblem where
  cityA_goods : ℕ
  cityB_goods : ℕ
  townA_needs : ℕ
  townB_needs : ℕ
  costA_to_A : ℕ
  costA_to_B : ℕ
  costB_to_A : ℕ
  costB_to_B : ℕ

/-- Define the specific problem instance -/
def problem : TransportationProblem := {
  cityA_goods := 120
  cityB_goods := 130
  townA_needs := 140
  townB_needs := 110
  costA_to_A := 300
  costA_to_B := 150
  costB_to_A := 200
  costB_to_B := 100
}

/-- Total transportation cost function -/
def total_cost (p : TransportationProblem) (x : ℕ) : ℕ :=
  p.costA_to_A * x + p.costA_to_B * (p.cityA_goods - x) +
  p.costB_to_A * (p.townA_needs - x) + p.costB_to_B * (p.townB_needs - p.cityA_goods + x)

/-- Theorem: The minimum total transportation cost is 45500 yuan -/
theorem min_transportation_cost :
  ∃ x, x ≥ 10 ∧ x ≤ 120 ∧
  (∀ y, y ≥ 10 → y ≤ 120 → total_cost problem x ≤ total_cost problem y) ∧
  total_cost problem x = 45500 :=
sorry

end NUMINAMATH_CALUDE_min_transportation_cost_l3974_397413


namespace NUMINAMATH_CALUDE_origin_outside_circle_l3974_397495

theorem origin_outside_circle (a : ℝ) (h1 : 0 < a) (h2 : a < 1) :
  let circle := {(x, y) : ℝ × ℝ | x^2 + y^2 + 2*a*x + 2*y + (a-1)^2 = 0}
  (0, 0) ∉ circle ∧ ∃ (p : ℝ × ℝ), p ∈ circle ∧ dist p (0, 0) < dist (0, 0) p := by
  sorry

end NUMINAMATH_CALUDE_origin_outside_circle_l3974_397495


namespace NUMINAMATH_CALUDE_max_m_value_l3974_397494

theorem max_m_value (m : ℝ) : 
  let A := {x : ℝ | m + 1 ≤ x ∧ x ≤ 3 * m - 1}
  let B := {x : ℝ | 1 ≤ x ∧ x ≤ 10}
  A ⊆ B → m ≤ 11 / 3 :=
by sorry

end NUMINAMATH_CALUDE_max_m_value_l3974_397494


namespace NUMINAMATH_CALUDE_monotonicity_and_tangent_line_and_max_k_l3974_397424

-- Define the function f(x)
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.exp x - a * x - 2

-- Define the derivative of f(x)
noncomputable def f_deriv (a : ℝ) (x : ℝ) : ℝ := Real.exp x - a

theorem monotonicity_and_tangent_line_and_max_k :
  -- Part 1: Monotonicity of f(x)
  (∀ a : ℝ, a ≤ 0 → StrictMono (f a)) ∧
  (∀ a : ℝ, a > 0 → 
    (∀ x y : ℝ, x < y → y < Real.log a → f a y < f a x) ∧
    (∀ x y : ℝ, Real.log a < x → x < y → f a x < f a y)) ∧
  
  -- Part 2: Tangent line condition
  (∀ a : ℝ, (∃ x₀ : ℝ, f_deriv a x₀ = Real.exp 1 ∧ 
    f a x₀ = Real.exp x₀ - 2) → a = 0) ∧
  
  -- Part 3: Maximum value of k
  (∀ k : ℤ, (∀ x : ℝ, x > 0 → (x - ↑k) * (f_deriv 1 x) + x + 1 > 0) → 
    k ≤ 2) ∧
  (∃ x : ℝ, x > 0 ∧ (x - 2) * (f_deriv 1 x) + x + 1 > 0)
  := by sorry

end NUMINAMATH_CALUDE_monotonicity_and_tangent_line_and_max_k_l3974_397424


namespace NUMINAMATH_CALUDE_min_value_expression_l3974_397401

theorem min_value_expression (x y z : ℝ) (h1 : 2 ≤ x) (h2 : x ≤ y) (h3 : y ≤ z) (h4 : z ≤ 5) :
  (x - 2)^2 + (y/x - 1)^2 + (z/y - 1)^2 + (5/z - 1)^2 ≥ 4 * (5^(1/4) - 1)^2 := by
  sorry

end NUMINAMATH_CALUDE_min_value_expression_l3974_397401


namespace NUMINAMATH_CALUDE_tanner_money_left_l3974_397456

def savings : List ℝ := [17, 48, 25, 55]
def video_game_price : ℝ := 49
def shoes_price : ℝ := 65
def discount_rate : ℝ := 0.1
def tax_rate : ℝ := 0.05

def total_savings : ℝ := savings.sum

def discounted_video_game_price : ℝ := video_game_price * (1 - discount_rate)

def total_cost_before_tax : ℝ := discounted_video_game_price + shoes_price

def sales_tax : ℝ := total_cost_before_tax * tax_rate

def total_cost_with_tax : ℝ := total_cost_before_tax + sales_tax

def money_left : ℝ := total_savings - total_cost_with_tax

theorem tanner_money_left :
  ∃ (ε : ℝ), money_left = 30.44 + ε ∧ abs ε < 0.005 := by
  sorry

end NUMINAMATH_CALUDE_tanner_money_left_l3974_397456


namespace NUMINAMATH_CALUDE_divisor_condition_l3974_397489

def satisfies_condition (n : ℕ) : Prop :=
  n > 1 ∧ ∀ k l : ℕ, k ∣ n → l ∣ n → k < n → l < n →
    (2*k - l) ∣ n ∨ (2*l - k) ∣ n

theorem divisor_condition (n : ℕ) :
  satisfies_condition n ↔ Nat.Prime n ∨ n ∈ ({6, 9, 15} : Set ℕ) :=
sorry

end NUMINAMATH_CALUDE_divisor_condition_l3974_397489


namespace NUMINAMATH_CALUDE_king_middle_school_teachers_l3974_397490

theorem king_middle_school_teachers :
  let total_students : ℕ := 1500
  let classes_per_student : ℕ := 5
  let regular_class_size : ℕ := 30
  let specialized_classes : ℕ := 10
  let specialized_class_size : ℕ := 15
  let classes_per_teacher : ℕ := 3

  let total_class_instances : ℕ := total_students * classes_per_student
  let specialized_class_instances : ℕ := specialized_classes * specialized_class_size
  let regular_class_instances : ℕ := total_class_instances - specialized_class_instances
  let regular_classes : ℕ := regular_class_instances / regular_class_size
  let total_classes : ℕ := regular_classes + specialized_classes
  let number_of_teachers : ℕ := total_classes / classes_per_teacher

  number_of_teachers = 85 := by sorry

end NUMINAMATH_CALUDE_king_middle_school_teachers_l3974_397490


namespace NUMINAMATH_CALUDE_right_triangle_area_l3974_397466

theorem right_triangle_area (a b c : ℝ) (h1 : a = 48) (h2 : c = 50) (h3 : a^2 + b^2 = c^2) : 
  (1/2) * a * b = 336 := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_area_l3974_397466


namespace NUMINAMATH_CALUDE_sam_seashells_l3974_397485

/-- The number of seashells Sam has after giving some to Joan -/
def remaining_seashells (initial : ℕ) (given : ℕ) : ℕ :=
  initial - given

theorem sam_seashells : remaining_seashells 35 18 = 17 := by
  sorry

end NUMINAMATH_CALUDE_sam_seashells_l3974_397485


namespace NUMINAMATH_CALUDE_octal_minus_base9_equals_152294_l3974_397450

def base_to_decimal (digits : List Nat) (base : Nat) : Nat :=
  digits.foldr (fun d acc => d + base * acc) 0

theorem octal_minus_base9_equals_152294 :
  let octal_num := [5, 4, 3, 2, 1, 0]
  let base9_num := [4, 3, 2, 1, 0]
  base_to_decimal octal_num 8 - base_to_decimal base9_num 9 = 152294 := by
  sorry

end NUMINAMATH_CALUDE_octal_minus_base9_equals_152294_l3974_397450


namespace NUMINAMATH_CALUDE_tan_315_and_radian_conversion_l3974_397499

theorem tan_315_and_radian_conversion :
  Real.tan (315 * π / 180) = -1 ∧ 315 * π / 180 = 7 * π / 4 := by
  sorry

end NUMINAMATH_CALUDE_tan_315_and_radian_conversion_l3974_397499


namespace NUMINAMATH_CALUDE_probability_of_valid_sequence_l3974_397480

/-- Fibonacci sequence -/
def fib : ℕ → ℕ
  | 0 => 0
  | 1 => 1
  | n + 2 => fib (n + 1) + fib n

/-- Number of valid sequences of length n -/
def validSequences (n : ℕ) : ℕ := fib (n + 2)

/-- Total number of possible sequences of length n -/
def totalSequences (n : ℕ) : ℕ := 2^n

/-- The probability of a valid sequence of length 8 -/
def probability : ℚ := validSequences 8 / totalSequences 8

theorem probability_of_valid_sequence :
  probability = 55 / 256 := by sorry

end NUMINAMATH_CALUDE_probability_of_valid_sequence_l3974_397480


namespace NUMINAMATH_CALUDE_carnival_game_ratio_l3974_397484

/-- The ratio of winners to losers in a carnival game -/
def carnival_ratio (winners losers : ℕ) : ℚ :=
  winners / losers

/-- Simplify a ratio by dividing both numerator and denominator by their GCD -/
def simplify_ratio (n d : ℕ) : ℚ :=
  (n / Nat.gcd n d) / (d / Nat.gcd n d)

theorem carnival_game_ratio :
  simplify_ratio 28 7 = 4 / 1 := by
  sorry

end NUMINAMATH_CALUDE_carnival_game_ratio_l3974_397484


namespace NUMINAMATH_CALUDE_other_number_proof_l3974_397492

theorem other_number_proof (A B : ℕ) : 
  A > 0 → B > 0 →
  Nat.lcm A B = 9699690 →
  Nat.gcd A B = 385 →
  A = 44530 →
  B = 83891 := by
sorry

end NUMINAMATH_CALUDE_other_number_proof_l3974_397492


namespace NUMINAMATH_CALUDE_min_n_for_constant_term_l3974_397472

theorem min_n_for_constant_term (x : ℝ) : 
  (∃ (n : ℕ), n > 0 ∧ (∃ (r : ℕ), r ≤ n ∧ 3 * n = 7 * r)) ∧
  (∀ (m : ℕ), m > 0 → (∃ (r : ℕ), r ≤ m ∧ 3 * m = 7 * r) → m ≥ 7) :=
by sorry

end NUMINAMATH_CALUDE_min_n_for_constant_term_l3974_397472


namespace NUMINAMATH_CALUDE_total_amount_formula_total_amount_after_five_months_l3974_397423

/-- Savings account with monthly interest -/
structure SavingsAccount where
  initialDeposit : ℝ
  monthlyInterestRate : ℝ

/-- Calculate total amount after x months -/
def totalAmount (account : SavingsAccount) (months : ℝ) : ℝ :=
  account.initialDeposit + account.initialDeposit * account.monthlyInterestRate * months

/-- Theorem: Total amount after x months is 100 + 0.36x -/
theorem total_amount_formula (account : SavingsAccount) (months : ℝ) 
    (h1 : account.initialDeposit = 100)
    (h2 : account.monthlyInterestRate = 0.0036) : 
    totalAmount account months = 100 + 0.36 * months := by
  sorry

/-- Theorem: Total amount after 5 months is 101.8 -/
theorem total_amount_after_five_months (account : SavingsAccount) 
    (h1 : account.initialDeposit = 100)
    (h2 : account.monthlyInterestRate = 0.0036) : 
    totalAmount account 5 = 101.8 := by
  sorry

end NUMINAMATH_CALUDE_total_amount_formula_total_amount_after_five_months_l3974_397423


namespace NUMINAMATH_CALUDE_kirsty_model_purchase_l3974_397448

/-- The number of models Kirsty can buy at the new price -/
def new_quantity : ℕ := 27

/-- The initial price of each model in dollars -/
def initial_price : ℚ := 45/100

/-- The new price of each model in dollars -/
def new_price : ℚ := 1/2

/-- The initial number of models Kirsty planned to buy -/
def initial_quantity : ℕ := 30

theorem kirsty_model_purchase :
  initial_quantity * initial_price = new_quantity * new_price :=
sorry


end NUMINAMATH_CALUDE_kirsty_model_purchase_l3974_397448


namespace NUMINAMATH_CALUDE_intersecting_chords_theorem_l3974_397461

/-- Given two intersecting chords in a circle, where one chord is divided into segments
    of 12 cm and 18 cm, and the other chord is divided in the ratio 3:8,
    prove that the length of the second chord is 33 cm. -/
theorem intersecting_chords_theorem (chord1_seg1 chord1_seg2 : ℝ)
  (chord2_ratio1 chord2_ratio2 : ℕ) :
  chord1_seg1 = 12 →
  chord1_seg2 = 18 →
  chord2_ratio1 = 3 →
  chord2_ratio2 = 8 →
  chord1_seg1 * chord1_seg2 = (chord2_ratio1 : ℝ) * (chord2_ratio2 : ℝ) * ((33 : ℝ) / (chord2_ratio1 + chord2_ratio2))^2 :=
by sorry

end NUMINAMATH_CALUDE_intersecting_chords_theorem_l3974_397461


namespace NUMINAMATH_CALUDE_Q_not_subset_P_l3974_397458

-- Define set P
def P : Set ℝ := {y | y ≥ 0}

-- Define set Q
def Q : Set ℝ := {y | ∃ x, y = Real.log x}

-- Theorem statement
theorem Q_not_subset_P : ¬(Q ⊆ P ∧ P ∩ Q = Q) := by
  sorry

end NUMINAMATH_CALUDE_Q_not_subset_P_l3974_397458


namespace NUMINAMATH_CALUDE_dorothy_found_57_pieces_l3974_397478

/-- The number of sea glass pieces found by Dorothy -/
def dorothy_total (blanche_green blanche_red rose_red rose_blue : ℕ) : ℕ :=
  let dorothy_red := 2 * (blanche_red + rose_red)
  let dorothy_blue := 3 * rose_blue
  dorothy_red + dorothy_blue

/-- Theorem stating that Dorothy found 57 pieces of sea glass -/
theorem dorothy_found_57_pieces :
  dorothy_total 12 3 9 11 = 57 := by
  sorry

#eval dorothy_total 12 3 9 11

end NUMINAMATH_CALUDE_dorothy_found_57_pieces_l3974_397478


namespace NUMINAMATH_CALUDE_factorial_ratio_equals_seven_and_half_l3974_397417

theorem factorial_ratio_equals_seven_and_half :
  (Nat.factorial 10 * Nat.factorial 7 * Nat.factorial 3) / (Nat.factorial 9 * Nat.factorial 8) = 15 / 2 := by
  sorry

end NUMINAMATH_CALUDE_factorial_ratio_equals_seven_and_half_l3974_397417


namespace NUMINAMATH_CALUDE_grunters_win_probability_l3974_397414

theorem grunters_win_probability (n : ℕ) (p : ℚ) (h1 : n = 6) (h2 : p = 3/5) :
  p ^ n = 729 / 15625 := by
  sorry

end NUMINAMATH_CALUDE_grunters_win_probability_l3974_397414


namespace NUMINAMATH_CALUDE_rhombus_perimeter_l3974_397457

theorem rhombus_perimeter (d1 d2 : ℝ) (h1 : d1 = 12) (h2 : d2 = 16) : 
  let side := Real.sqrt ((d1/2)^2 + (d2/2)^2)
  4 * side = 40 := by
  sorry

end NUMINAMATH_CALUDE_rhombus_perimeter_l3974_397457


namespace NUMINAMATH_CALUDE_pascals_triangle_sum_l3974_397419

theorem pascals_triangle_sum (n : ℕ) : 
  n = 51 → Nat.choose n 4 + Nat.choose n 6 = 18249360 := by
  sorry

end NUMINAMATH_CALUDE_pascals_triangle_sum_l3974_397419


namespace NUMINAMATH_CALUDE_complex_number_problem_l3974_397465

/-- Given complex numbers z₁ and z₂ satisfying certain conditions, prove that z₂ = 6 + 2i -/
theorem complex_number_problem (z₁ z₂ : ℂ) : 
  ((z₁ - 2) * Complex.I = 1 + Complex.I) →
  (z₂.im = 2) →
  ((z₁ * z₂).im = 0) →
  z₂ = 6 + 2 * Complex.I :=
by
  sorry


end NUMINAMATH_CALUDE_complex_number_problem_l3974_397465


namespace NUMINAMATH_CALUDE_inequality_proof_l3974_397409

theorem inequality_proof (w x y z : ℝ) 
  (h_non_neg : w ≥ 0 ∧ x ≥ 0 ∧ y ≥ 0 ∧ z ≥ 0) 
  (h_sum : w * x + x * y + y * z + z * w = 1) : 
  w^3 / (x + y + z) + x^3 / (w + y + z) + y^3 / (w + x + z) + z^3 / (w + x + y) ≥ 1/3 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l3974_397409


namespace NUMINAMATH_CALUDE_quadratic_inequality_properties_l3974_397406

-- Define the quadratic function
def f (a b c x : ℝ) := a * x^2 + b * x + c

-- Define the solution set
def solution_set (a b c : ℝ) := {x : ℝ | f a b c x < 0}

-- State the theorem
theorem quadratic_inequality_properties
  (a b c : ℝ)
  (h : solution_set a b c = {x : ℝ | x < -1 ∨ x > 3}) :
  a < 0 ∧ 
  a + b + c > 0 ∧ 
  solution_set c (-b) a = {x : ℝ | -1/3 < x ∧ x < 1} :=
by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_properties_l3974_397406


namespace NUMINAMATH_CALUDE_max_sum_of_two_max_sum_is_zero_l3974_397468

def number_set : Finset Int := {1, -1, -2}

theorem max_sum_of_two (a b : Int) (ha : a ∈ number_set) (hb : b ∈ number_set) (hab : a ≠ b) :
  ∃ (x y : Int), x ∈ number_set ∧ y ∈ number_set ∧ x ≠ y ∧ x + y ≥ a + b :=
sorry

theorem max_sum_is_zero :
  ∃ (a b : Int), a ∈ number_set ∧ b ∈ number_set ∧ a ≠ b ∧
  (∀ (x y : Int), x ∈ number_set → y ∈ number_set → x ≠ y → a + b ≥ x + y) ∧
  a + b = 0 :=
sorry

end NUMINAMATH_CALUDE_max_sum_of_two_max_sum_is_zero_l3974_397468


namespace NUMINAMATH_CALUDE_sqrt_inequality_l3974_397446

theorem sqrt_inequality (a : ℝ) (h : a ≥ 3) :
  Real.sqrt a - Real.sqrt (a - 2) < Real.sqrt (a - 1) - Real.sqrt (a - 3) := by
  sorry

end NUMINAMATH_CALUDE_sqrt_inequality_l3974_397446


namespace NUMINAMATH_CALUDE_average_difference_l3974_397432

theorem average_difference (a b c : ℝ) 
  (h1 : (a + b) / 2 = 110) 
  (h2 : (b + c) / 2 = 170) : 
  a - c = -120 := by
sorry

end NUMINAMATH_CALUDE_average_difference_l3974_397432


namespace NUMINAMATH_CALUDE_quadratic_has_two_distinct_roots_l3974_397443

theorem quadratic_has_two_distinct_roots 
  (a b c : ℝ) 
  (h1 : 5*a + 3*b + 2*c = 0) 
  (h2 : a ≠ 0) : 
  ∃ x y : ℝ, x ≠ y ∧ a*x^2 + b*x + c = 0 ∧ a*y^2 + b*y + c = 0 :=
sorry

end NUMINAMATH_CALUDE_quadratic_has_two_distinct_roots_l3974_397443


namespace NUMINAMATH_CALUDE_inequality_proofs_l3974_397434

theorem inequality_proofs 
  (h : ∀ x > 0, 1 / (1 + x) < Real.log (1 + 1 / x) ∧ Real.log (1 + 1 / x) < 1 / x) :
  (1 + 1/2 + 1/3 + 1/4 + 1/5 + 1/6 + 1/7 > Real.log 8) ∧
  (1/2 + 1/3 + 1/4 + 1/5 + 1/6 + 1/7 + 1/8 < Real.log 8) ∧
  ((1 : ℝ) / 1 + 8 / 8 + 28 / 64 + 56 / 512 + 70 / 4096 + 56 / 32768 + 28 / 262144 + 8 / 2097152 + 1 / 16777216 < Real.exp 1) :=
by sorry

end NUMINAMATH_CALUDE_inequality_proofs_l3974_397434


namespace NUMINAMATH_CALUDE_megans_vacation_pictures_l3974_397454

theorem megans_vacation_pictures (zoo_pics museum_pics deleted_pics : ℕ) 
  (h1 : zoo_pics = 15)
  (h2 : museum_pics = 18)
  (h3 : deleted_pics = 31) :
  zoo_pics + museum_pics - deleted_pics = 2 := by
  sorry

end NUMINAMATH_CALUDE_megans_vacation_pictures_l3974_397454


namespace NUMINAMATH_CALUDE_at_least_two_inequalities_hold_l3974_397476

theorem at_least_two_inequalities_hold (a b c : ℝ) 
  (ha : a > 0) (hb : b > 0) (hc : c > 0) 
  (h_sum : a + b + c ≥ a * b * c) : 
  (2 / a + 3 / b + 6 / c ≥ 6 ∧ 2 / b + 3 / c + 6 / a ≥ 6) ∨
  (2 / b + 3 / c + 6 / a ≥ 6 ∧ 2 / c + 3 / a + 6 / b ≥ 6) ∨
  (2 / c + 3 / a + 6 / b ≥ 6 ∧ 2 / a + 3 / b + 6 / c ≥ 6) :=
by sorry

end NUMINAMATH_CALUDE_at_least_two_inequalities_hold_l3974_397476


namespace NUMINAMATH_CALUDE_tree_height_when_boy_grows_l3974_397418

-- Define the problem parameters
def initial_tree_height : ℝ := 16
def initial_boy_height : ℝ := 24
def final_boy_height : ℝ := 36

-- Define the growth rate relationship
def tree_growth_rate (boy_growth : ℝ) : ℝ := 2 * boy_growth

-- Theorem statement
theorem tree_height_when_boy_grows (boy_growth : ℝ) 
  (h : final_boy_height = initial_boy_height + boy_growth) :
  initial_tree_height + tree_growth_rate boy_growth = 40 :=
by
  sorry


end NUMINAMATH_CALUDE_tree_height_when_boy_grows_l3974_397418


namespace NUMINAMATH_CALUDE_sharmila_hourly_wage_l3974_397430

/-- Represents Sharmila's work schedule and earnings -/
structure WorkSchedule where
  monday_hours : ℕ
  wednesday_hours : ℕ
  friday_hours : ℕ
  tuesday_hours : ℕ
  thursday_hours : ℕ
  weekly_earnings : ℕ

/-- Calculates the total hours worked in a week -/
def total_hours (schedule : WorkSchedule) : ℕ :=
  3 * schedule.monday_hours + 2 * schedule.tuesday_hours

/-- Calculates the hourly wage given a work schedule -/
def hourly_wage (schedule : WorkSchedule) : ℚ :=
  schedule.weekly_earnings / (total_hours schedule)

/-- Sharmila's work schedule -/
def sharmila_schedule : WorkSchedule := {
  monday_hours := 10,
  wednesday_hours := 10,
  friday_hours := 10,
  tuesday_hours := 8,
  thursday_hours := 8,
  weekly_earnings := 460
}

/-- Theorem stating that Sharmila's hourly wage is $10 -/
theorem sharmila_hourly_wage :
  hourly_wage sharmila_schedule = 10 := by sorry

end NUMINAMATH_CALUDE_sharmila_hourly_wage_l3974_397430


namespace NUMINAMATH_CALUDE_chris_bowling_score_l3974_397442

/-- Proves Chris's bowling score given Sarah and Greg's score conditions -/
theorem chris_bowling_score (sarah_score greg_score : ℕ) : 
  sarah_score = greg_score + 60 →
  (sarah_score + greg_score) / 2 = 110 →
  let avg := (sarah_score + greg_score) / 2
  let chris_score := (avg * 120) / 100
  chris_score = 132 := by
sorry

end NUMINAMATH_CALUDE_chris_bowling_score_l3974_397442


namespace NUMINAMATH_CALUDE_B_cannot_be_possible_l3974_397497

-- Define the set A
def A : Set ℝ := {x : ℝ | ∃ y : ℝ, y = Real.sqrt (x - 1)}

-- Define the set B (the one we want to prove cannot be possible)
def B : Set ℝ := {x : ℝ | x ≥ -1}

-- Theorem statement
theorem B_cannot_be_possible : A ∩ B = ∅ → False := by
  sorry

end NUMINAMATH_CALUDE_B_cannot_be_possible_l3974_397497


namespace NUMINAMATH_CALUDE_parallelogram_area_theorem_l3974_397486

/-- A point with integer coordinates -/
structure LatticePoint where
  x : ℤ
  y : ℤ

/-- A parallelogram defined by four lattice points -/
structure Parallelogram where
  v1 : LatticePoint
  v2 : LatticePoint
  v3 : LatticePoint
  v4 : LatticePoint

/-- Checks if a point is inside or on the edges of a parallelogram (excluding vertices) -/
def isInsideOrOnEdge (p : LatticePoint) (para : Parallelogram) : Prop :=
  sorry

/-- Calculates the area of a parallelogram -/
def area (para : Parallelogram) : ℚ :=
  sorry

theorem parallelogram_area_theorem (para : Parallelogram) :
  (∃ p : LatticePoint, isInsideOrOnEdge p para) → area para > 1 :=
by sorry

end NUMINAMATH_CALUDE_parallelogram_area_theorem_l3974_397486


namespace NUMINAMATH_CALUDE_largest_divisor_of_consecutive_sum_l3974_397471

theorem largest_divisor_of_consecutive_sum (a : ℤ) : 
  ∃ (d : ℤ), d > 0 ∧ d ∣ (a - 1 + a + a + 1) ∧ 
  ∀ (k : ℤ), k > d → ∃ (n : ℤ), ¬(k ∣ (n - 1 + n + n + 1)) :=
by sorry

end NUMINAMATH_CALUDE_largest_divisor_of_consecutive_sum_l3974_397471


namespace NUMINAMATH_CALUDE_ninth_row_fourth_number_l3974_397467

/-- The start of the i-th row in the sequence -/
def row_start (i : ℕ) : ℕ := 2 + 4 * 6 * (i - 1)

/-- The n-th number in the i-th row of the sequence -/
def seq_number (i n : ℕ) : ℕ := row_start i + 4 * (n - 1)

theorem ninth_row_fourth_number : seq_number 9 4 = 206 := by
  sorry

end NUMINAMATH_CALUDE_ninth_row_fourth_number_l3974_397467


namespace NUMINAMATH_CALUDE_rectangle_area_perimeter_relation_l3974_397464

theorem rectangle_area_perimeter_relation (x : ℝ) :
  let length : ℝ := 4 * x
  let width : ℝ := x + 10
  let area : ℝ := length * width
  let perimeter : ℝ := 2 * (length + width)
  (area = 2 * perimeter) → x = (Real.sqrt 41 - 1) / 2 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_area_perimeter_relation_l3974_397464


namespace NUMINAMATH_CALUDE_volume_ratio_of_cubes_l3974_397400

-- Define the edge lengths in inches
def small_cube_edge : ℚ := 4
def large_cube_edge : ℚ := 24  -- 2 feet = 24 inches

-- Define the volumes of the cubes
def small_cube_volume : ℚ := small_cube_edge ^ 3
def large_cube_volume : ℚ := large_cube_edge ^ 3

-- Theorem statement
theorem volume_ratio_of_cubes : 
  small_cube_volume / large_cube_volume = 1 / 216 := by
  sorry

end NUMINAMATH_CALUDE_volume_ratio_of_cubes_l3974_397400


namespace NUMINAMATH_CALUDE_valid_sequence_count_l3974_397470

def word : String := "EQUALS"

def valid_sequence (s : String) : Prop :=
  s.length = 4 ∧
  s.toList.toFinset ⊆ word.toList.toFinset ∧
  s.front = 'L' ∧
  s.back = 'Q' ∧
  s.toList.toFinset.card = 4

def count_valid_sequences : ℕ :=
  (word.toList.toFinset.filter (λ c => c ≠ 'L' ∧ c ≠ 'Q')).card *
  ((word.toList.toFinset.filter (λ c => c ≠ 'L' ∧ c ≠ 'Q')).card - 1)

theorem valid_sequence_count :
  count_valid_sequences = 12 :=
sorry

end NUMINAMATH_CALUDE_valid_sequence_count_l3974_397470


namespace NUMINAMATH_CALUDE_inequality_proofs_l3974_397403

theorem inequality_proofs :
  (∀ x : ℝ, |x - 1| < 1 - 2*x ↔ x ∈ Set.Ioo 0 1) ∧
  (∀ x : ℝ, |x - 1| - |x + 1| > x ↔ x ∈ Set.Ioi (-1) ∪ Set.Ico (-1) 0) :=
by sorry

end NUMINAMATH_CALUDE_inequality_proofs_l3974_397403


namespace NUMINAMATH_CALUDE_diagonal_to_larger_base_ratio_l3974_397447

/-- An isosceles trapezoid with specific properties -/
structure IsoscelesTrapezoid where
  /-- The length of the smaller base -/
  smaller_base : ℝ
  /-- The length of the larger base -/
  larger_base : ℝ
  /-- The length of the diagonal -/
  diagonal : ℝ
  /-- The height of the trapezoid -/
  altitude : ℝ
  /-- The smaller base is positive -/
  smaller_base_pos : 0 < smaller_base
  /-- The larger base is greater than the smaller base -/
  base_order : smaller_base < larger_base
  /-- The smaller base equals half the diagonal -/
  smaller_base_eq_half_diagonal : smaller_base = diagonal / 2
  /-- The altitude equals two-thirds of the smaller base -/
  altitude_eq_two_thirds_smaller_base : altitude = 2 / 3 * smaller_base

/-- The ratio of the diagonal to the larger base in the specific isosceles trapezoid -/
theorem diagonal_to_larger_base_ratio (t : IsoscelesTrapezoid) : 
  t.diagonal / t.larger_base = 4 / 9 := by
  sorry

end NUMINAMATH_CALUDE_diagonal_to_larger_base_ratio_l3974_397447


namespace NUMINAMATH_CALUDE_croissant_mix_time_l3974_397425

/-- The time it takes to make croissants -/
def croissant_making : Prop :=
  let fold_count : ℕ := 4
  let fold_time : ℕ := 5
  let rest_count : ℕ := 4
  let rest_time : ℕ := 75
  let bake_time : ℕ := 30
  let total_time : ℕ := 6 * 60

  let fold_total : ℕ := fold_count * fold_time
  let rest_total : ℕ := rest_count * rest_time
  let known_time : ℕ := fold_total + rest_total + bake_time

  let mix_time : ℕ := total_time - known_time

  mix_time = 10

theorem croissant_mix_time : croissant_making := by
  sorry

end NUMINAMATH_CALUDE_croissant_mix_time_l3974_397425


namespace NUMINAMATH_CALUDE_sine_function_inequality_l3974_397416

noncomputable def f (x : ℝ) (φ : ℝ) : ℝ := Real.sin (2 * x) * Real.cos φ + Real.cos (2 * x) * Real.sin φ

theorem sine_function_inequality 
  (φ : ℝ) 
  (h : ∀ x : ℝ, f x φ ≤ f (2 * Real.pi / 9) φ) : 
  f (2 * Real.pi / 3) φ < f (5 * Real.pi / 6) φ ∧ 
  f (5 * Real.pi / 6) φ < f (7 * Real.pi / 6) φ :=
sorry

end NUMINAMATH_CALUDE_sine_function_inequality_l3974_397416


namespace NUMINAMATH_CALUDE_triangle_inequality_with_median_l3974_397498

/-- 
For any triangle with side lengths a, b, and c, and median length m_a 
from vertex A to the midpoint of side BC, the inequality a^2 + 4m_a^2 ≤ (b+c)^2 holds.
-/
theorem triangle_inequality_with_median 
  (a b c m_a : ℝ) 
  (h_positive : 0 < a ∧ 0 < b ∧ 0 < c ∧ 0 < m_a) 
  (h_triangle : a < b + c ∧ b < a + c ∧ c < a + b) 
  (h_median : m_a > 0) : 
  a^2 + 4 * m_a^2 ≤ (b + c)^2 := by
  sorry

end NUMINAMATH_CALUDE_triangle_inequality_with_median_l3974_397498


namespace NUMINAMATH_CALUDE_lines_parallel_iff_a_eq_zero_l3974_397483

/-- Two lines in the form of x-2ay=1 and 2x-2ay=1 are parallel if and only if a=0 -/
theorem lines_parallel_iff_a_eq_zero (a : ℝ) :
  (∀ x y : ℝ, x - 2*a*y = 1 ↔ 2*x - 2*a*y = 1) ↔ a = 0 := by
  sorry

end NUMINAMATH_CALUDE_lines_parallel_iff_a_eq_zero_l3974_397483


namespace NUMINAMATH_CALUDE_largest_divisor_of_n_l3974_397437

theorem largest_divisor_of_n (n : ℕ+) (h : 650 ∣ n^3) : 130 ∣ n := by
  sorry

end NUMINAMATH_CALUDE_largest_divisor_of_n_l3974_397437


namespace NUMINAMATH_CALUDE_max_value_at_two_l3974_397487

-- Define the function f
def f (c : ℝ) (x : ℝ) : ℝ := x * (x - c)^2

-- State the theorem
theorem max_value_at_two (c : ℝ) :
  (∀ x : ℝ, f c x ≤ f c 2) → c = 6 := by
  sorry

end NUMINAMATH_CALUDE_max_value_at_two_l3974_397487


namespace NUMINAMATH_CALUDE_bank_teller_problem_l3974_397445

theorem bank_teller_problem (total_bills : ℕ) (total_value : ℕ) 
  (h1 : total_bills = 54)
  (h2 : total_value = 780) :
  ∃ (five_dollar_bills twenty_dollar_bills : ℕ),
    five_dollar_bills + twenty_dollar_bills = total_bills ∧
    5 * five_dollar_bills + 20 * twenty_dollar_bills = total_value ∧
    five_dollar_bills = 20 := by
  sorry

end NUMINAMATH_CALUDE_bank_teller_problem_l3974_397445


namespace NUMINAMATH_CALUDE_newberg_airport_passengers_l3974_397482

theorem newberg_airport_passengers : 
  let on_time_passengers : ℕ := 14507
  let late_passengers : ℕ := 213
  on_time_passengers + late_passengers = 14720 := by sorry

end NUMINAMATH_CALUDE_newberg_airport_passengers_l3974_397482


namespace NUMINAMATH_CALUDE_exists_subsequences_forming_2520_l3974_397427

def infinite_sequence : ℕ → ℕ
  | n => match n % 6 with
         | 0 => 2
         | 1 => 0
         | 2 => 1
         | 3 => 5
         | 4 => 2
         | 5 => 0
         | _ => 0  -- This case should never occur

def is_subsequence (s : List ℕ) : Prop :=
  ∃ start : ℕ, ∀ i : ℕ, i < s.length → s.get ⟨i, by sorry⟩ = infinite_sequence (start + i)

def concatenate_to_number (s1 s2 : List ℕ) : ℕ :=
  (s1 ++ s2).foldl (λ acc d => acc * 10 + d) 0

theorem exists_subsequences_forming_2520 :
  ∃ (s1 s2 : List ℕ),
    s1 ≠ [] ∧ s2 ≠ [] ∧
    is_subsequence s1 ∧
    is_subsequence s2 ∧
    concatenate_to_number s1 s2 = 2520 ∧
    2520 % 45 = 0 := by
  sorry

end NUMINAMATH_CALUDE_exists_subsequences_forming_2520_l3974_397427


namespace NUMINAMATH_CALUDE_paint_calculation_l3974_397436

/-- Represents the number of rooms that can be painted with a given amount of paint -/
structure PaintCoverage where
  totalRooms : ℕ
  cansUsed : ℕ

/-- Calculates the number of cans used for a given number of rooms -/
def cansForRooms (initialCoverage finalCoverage : PaintCoverage) (roomsToPaint : ℕ) : ℕ :=
  let roomsPerCan := (initialCoverage.totalRooms - finalCoverage.totalRooms) / 
                     (initialCoverage.cansUsed - finalCoverage.cansUsed)
  roomsToPaint / roomsPerCan

theorem paint_calculation (initialCoverage finalCoverage : PaintCoverage) 
  (h1 : initialCoverage.totalRooms = 45)
  (h2 : finalCoverage.totalRooms = 36)
  (h3 : initialCoverage.cansUsed - finalCoverage.cansUsed = 4) :
  cansForRooms initialCoverage finalCoverage 36 = 16 := by
  sorry

end NUMINAMATH_CALUDE_paint_calculation_l3974_397436


namespace NUMINAMATH_CALUDE_binomial_expansion_problem_l3974_397479

theorem binomial_expansion_problem (a b : ℝ) : 
  (∃ c d e : ℝ, (1 + a * x)^5 = 1 + 10*x + b*x^2 + c*x^3 + d*x^4 + a^5*x^5) → 
  a - b = -38 := by
sorry

end NUMINAMATH_CALUDE_binomial_expansion_problem_l3974_397479


namespace NUMINAMATH_CALUDE_union_of_A_and_B_l3974_397421

-- Define set A
def A : Set ℝ := {x | 0 < 3 - x ∧ 3 - x ≤ 2}

-- Define set B
def B : Set ℝ := {x | 0 ≤ x ∧ x ≤ 2}

-- Theorem statement
theorem union_of_A_and_B : A ∪ B = {x : ℝ | 0 ≤ x ∧ x < 3} := by sorry

end NUMINAMATH_CALUDE_union_of_A_and_B_l3974_397421


namespace NUMINAMATH_CALUDE_smallest_positive_solution_congruence_l3974_397408

theorem smallest_positive_solution_congruence :
  ∃ (x : ℕ), x > 0 ∧ (4 * x) % 37 = 17 % 37 ∧ 
  ∀ (y : ℕ), y > 0 → (4 * y) % 37 = 17 % 37 → x ≤ y :=
by sorry

end NUMINAMATH_CALUDE_smallest_positive_solution_congruence_l3974_397408
