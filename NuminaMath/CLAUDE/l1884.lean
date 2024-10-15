import Mathlib

namespace NUMINAMATH_CALUDE_division_problem_l1884_188456

theorem division_problem (x y : ℕ+) : 
  (x : ℝ) / y = 96.15 → 
  ∃ q : ℕ, x = q * y + 9 →
  y = 60 := by
sorry

end NUMINAMATH_CALUDE_division_problem_l1884_188456


namespace NUMINAMATH_CALUDE_twentieth_term_of_specific_sequence_l1884_188428

/-- The nth term of an arithmetic sequence -/
def arithmetic_sequence (a₁ : ℝ) (d : ℝ) (n : ℕ) : ℝ :=
  a₁ + (n - 1 : ℝ) * d

theorem twentieth_term_of_specific_sequence :
  arithmetic_sequence 2 4 20 = 78 := by
  sorry

end NUMINAMATH_CALUDE_twentieth_term_of_specific_sequence_l1884_188428


namespace NUMINAMATH_CALUDE_jack_bought_36_books_l1884_188430

/-- The number of books Jack bought each month -/
def books_per_month : ℕ := sorry

/-- The price of each book in dollars -/
def price_per_book : ℕ := 20

/-- The total sale price at the end of the year in dollars -/
def total_sale_price : ℕ := 500

/-- The total loss in dollars -/
def total_loss : ℕ := 220

/-- Theorem stating that Jack bought 36 books each month -/
theorem jack_bought_36_books : books_per_month = 36 := by
  sorry

end NUMINAMATH_CALUDE_jack_bought_36_books_l1884_188430


namespace NUMINAMATH_CALUDE_problem_solution_l1884_188438

def f (a x : ℝ) : ℝ := |2*x - a| + |x - 1|

theorem problem_solution :
  (∀ a : ℝ, (∀ x : ℝ, f a x + |x - 1| ≥ 2) → a ≤ 0 ∨ a ≥ 4) ∧
  (∀ a : ℝ, a < 2 → (∃ x : ℝ, ∀ y : ℝ, f a x ≤ f a y) → f a (a/2) = a - 1 → a = 4/3) :=
by sorry

end NUMINAMATH_CALUDE_problem_solution_l1884_188438


namespace NUMINAMATH_CALUDE_valid_money_distribution_exists_l1884_188422

/-- Represents the distribution of money among 5 people --/
structure MoneyDistribution where
  total : ℝ
  a : ℝ
  b : ℝ
  c : ℝ
  d : ℝ
  e : ℝ

/-- Checks if a MoneyDistribution satisfies the given conditions --/
def satisfiesConditions (dist : MoneyDistribution) : Prop :=
  dist.total = 5000 ∧
  dist.a / dist.b = 3 / 2 ∧
  dist.b / dist.c = 4 / 5 ∧
  dist.d = 0.6 * dist.c ∧
  dist.e = 0.6 * dist.c ∧
  dist.a + dist.b + dist.c + dist.d + dist.e = dist.total

/-- Theorem stating the existence of a valid money distribution --/
theorem valid_money_distribution_exists : ∃ (dist : MoneyDistribution), satisfiesConditions dist := by
  sorry

#check valid_money_distribution_exists

end NUMINAMATH_CALUDE_valid_money_distribution_exists_l1884_188422


namespace NUMINAMATH_CALUDE_reflection_across_x_axis_l1884_188472

-- Define the original function g(x)
def g (x : ℝ) : ℝ := x^2 - 4

-- Define the reflected function h(x)
def h (x : ℝ) : ℝ := -x^2 + 4

-- Theorem stating that h(x) is the reflection of g(x) across the x-axis
theorem reflection_across_x_axis :
  ∀ x : ℝ, h x = -(g x) :=
by
  sorry

end NUMINAMATH_CALUDE_reflection_across_x_axis_l1884_188472


namespace NUMINAMATH_CALUDE_quadratic_function_properties_l1884_188414

def f (x : ℝ) : ℝ := x^2 + 1

theorem quadratic_function_properties :
  (f 0 = 1) ∧ (∀ x : ℝ, deriv f x > 0) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_function_properties_l1884_188414


namespace NUMINAMATH_CALUDE_identical_rows_from_increasing_sums_l1884_188434

theorem identical_rows_from_increasing_sums 
  (n : ℕ) 
  (row1 row2 : Fin n → ℝ) 
  (distinct : ∀ i j, i ≠ j → row1 i ≠ row1 j) 
  (increasing_row1 : ∀ i j, i < j → row1 i < row1 j) 
  (same_elements : ∀ x, ∃ i, row1 i = x ↔ ∃ j, row2 j = x) 
  (increasing_sums : ∀ i j, i < j → row1 i + row2 i < row1 j + row2 j) : 
  ∀ i, row1 i = row2 i :=
sorry

end NUMINAMATH_CALUDE_identical_rows_from_increasing_sums_l1884_188434


namespace NUMINAMATH_CALUDE_greatest_four_digit_number_with_remainders_l1884_188474

theorem greatest_four_digit_number_with_remainders :
  ∃ n : ℕ,
    n ≤ 9999 ∧
    n > 999 ∧
    n % 15 = 2 ∧
    n % 24 = 8 ∧
    (∀ m : ℕ, m ≤ 9999 ∧ m > 999 ∧ m % 15 = 2 → m ≤ n) ∧
    n = 9992 :=
by sorry

end NUMINAMATH_CALUDE_greatest_four_digit_number_with_remainders_l1884_188474


namespace NUMINAMATH_CALUDE_polynomial_evaluation_l1884_188488

theorem polynomial_evaluation :
  ∀ x : ℝ, x > 0 → x^2 - 3*x - 9 = 0 →
  x^4 - 3*x^3 - 9*x^2 + 27*x - 8 = 8 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_evaluation_l1884_188488


namespace NUMINAMATH_CALUDE_scientific_notation_180_million_l1884_188446

/-- Proves that 180 million in scientific notation is equal to 1.8 × 10^8 -/
theorem scientific_notation_180_million :
  (180000000 : ℝ) = 1.8 * (10 ^ 8) := by
  sorry

end NUMINAMATH_CALUDE_scientific_notation_180_million_l1884_188446


namespace NUMINAMATH_CALUDE_consecutive_non_primes_l1884_188497

theorem consecutive_non_primes (n : ℕ) (h : n ≥ 1) :
  ∃ (k : ℕ), ∀ (i : ℕ), i ∈ Finset.range n → 
    ¬ Nat.Prime (k + i) ∧ 
    (∀ (j : ℕ), j ∈ Finset.range n → k + i = k + j → i = j) :=
by sorry

end NUMINAMATH_CALUDE_consecutive_non_primes_l1884_188497


namespace NUMINAMATH_CALUDE_N_satisfies_equation_l1884_188479

def N : Matrix (Fin 2) (Fin 2) ℝ := !![2, 2; 1, 2]

theorem N_satisfies_equation : 
  N^3 - 3 • N^2 + 4 • N = !![6, 12; 3, 6] := by sorry

end NUMINAMATH_CALUDE_N_satisfies_equation_l1884_188479


namespace NUMINAMATH_CALUDE_perpendicular_distance_to_plane_l1884_188490

/-- The perpendicular distance from a point to a plane --/
def perpendicularDistance (p : ℝ × ℝ × ℝ) (plane : Set (ℝ × ℝ × ℝ)) : ℝ :=
  sorry

/-- The plane containing three points --/
def planeThroughPoints (a b c : ℝ × ℝ × ℝ) : Set (ℝ × ℝ × ℝ) :=
  sorry

theorem perpendicular_distance_to_plane :
  let a : ℝ × ℝ × ℝ := (0, 0, 0)
  let b : ℝ × ℝ × ℝ := (5, 0, 0)
  let c : ℝ × ℝ × ℝ := (0, 3, 0)
  let d : ℝ × ℝ × ℝ := (0, 0, 6)
  let plane := planeThroughPoints a b c
  perpendicularDistance d plane = 6 := by
  sorry

end NUMINAMATH_CALUDE_perpendicular_distance_to_plane_l1884_188490


namespace NUMINAMATH_CALUDE_percentage_passed_all_subjects_l1884_188468

/-- Percentage of students who failed in Hindi -/
def A : ℝ := 30

/-- Percentage of students who failed in English -/
def B : ℝ := 45

/-- Percentage of students who failed in Math -/
def C : ℝ := 25

/-- Percentage of students who failed in Science -/
def D : ℝ := 40

/-- Percentage of students who failed in both Hindi and English -/
def AB : ℝ := 12

/-- Percentage of students who failed in both Hindi and Math -/
def AC : ℝ := 15

/-- Percentage of students who failed in both Hindi and Science -/
def AD : ℝ := 18

/-- Percentage of students who failed in both English and Math -/
def BC : ℝ := 20

/-- Percentage of students who failed in both English and Science -/
def BD : ℝ := 22

/-- Percentage of students who failed in both Math and Science -/
def CD : ℝ := 24

/-- Percentage of students who failed in all four subjects -/
def ABCD : ℝ := 10

/-- The total percentage -/
def total : ℝ := 100

theorem percentage_passed_all_subjects :
  total - (A + B + C + D - (AB + AC + AD + BC + BD + CD) + ABCD) = 61 := by
  sorry

end NUMINAMATH_CALUDE_percentage_passed_all_subjects_l1884_188468


namespace NUMINAMATH_CALUDE_multiples_of_6_not_18_under_350_l1884_188480

def count_multiples (n : ℕ) (m : ℕ) : ℕ :=
  (n - 1) / m

theorem multiples_of_6_not_18_under_350 : 
  (count_multiples 350 6) - (count_multiples 350 18) = 39 := by
  sorry

end NUMINAMATH_CALUDE_multiples_of_6_not_18_under_350_l1884_188480


namespace NUMINAMATH_CALUDE_triangle_ratio_l1884_188489

theorem triangle_ratio (a b c : ℝ) (A B C : ℝ) : 
  0 < a ∧ 0 < b ∧ 0 < c ∧ 
  0 < A ∧ A < π ∧ 0 < B ∧ B < π ∧ 0 < C ∧ C < π ∧
  A + B + C = π ∧
  a / (Real.sin A) = b / (Real.sin B) ∧
  b / (Real.sin B) = c / (Real.sin C) ∧
  b * Real.sin A * Real.sin B + a * (Real.cos B)^2 = 2 * c →
  a / c = 2 := by sorry

end NUMINAMATH_CALUDE_triangle_ratio_l1884_188489


namespace NUMINAMATH_CALUDE_range_of_3a_plus_4b_l1884_188459

theorem range_of_3a_plus_4b (a b : ℝ) (ha : a > 0) (hb : b > 0)
  (h1 : 2 * a + 2 * b ≤ 15) (h2 : 4 / a + 3 / b ≤ 2) :
  ∃ (min max : ℝ), min = 24 ∧ max = 27 ∧
  (∀ x, (∃ a' b' : ℝ, a' > 0 ∧ b' > 0 ∧
    2 * a' + 2 * b' ≤ 15 ∧ 4 / a' + 3 / b' ≤ 2 ∧
    x = 3 * a' + 4 * b') → min ≤ x ∧ x ≤ max) :=
sorry

end NUMINAMATH_CALUDE_range_of_3a_plus_4b_l1884_188459


namespace NUMINAMATH_CALUDE_value_of_z_l1884_188495

theorem value_of_z (x y z : ℝ) (hx : x = 3) (hy : y = 2 * x) (hz : z = 3 * y) : z = 18 := by
  sorry

end NUMINAMATH_CALUDE_value_of_z_l1884_188495


namespace NUMINAMATH_CALUDE_min_a_is_minimum_l1884_188455

/-- The inequality that holds for all x ≥ 0 -/
def inequality (a : ℝ) : Prop :=
  ∀ x : ℝ, x ≥ 0 → x * Real.exp x + a * Real.exp x * Real.log (x + 1) + 1 ≥ Real.exp x * (x + 1) ^ a

/-- The minimum value of a that satisfies the inequality -/
def min_a : ℝ := -1

/-- Theorem stating that min_a is the minimum value satisfying the inequality -/
theorem min_a_is_minimum :
  (∀ a : ℝ, inequality a → a ≥ min_a) ∧ inequality min_a := by sorry

end NUMINAMATH_CALUDE_min_a_is_minimum_l1884_188455


namespace NUMINAMATH_CALUDE_perpendicular_planes_l1884_188463

-- Define the types for line and plane
variable (L : Type) [LinearOrder L]
variable (P : Type)

-- Define the relations
variable (perpendicular : L → P → Prop)
variable (contains : P → L → Prop)
variable (perp_planes : P → P → Prop)

-- State the theorem
theorem perpendicular_planes 
  (l : L) (α β : P) 
  (h1 : perpendicular l α) 
  (h2 : contains β l) : 
  perp_planes α β :=
sorry

end NUMINAMATH_CALUDE_perpendicular_planes_l1884_188463


namespace NUMINAMATH_CALUDE_function_range_theorem_l1884_188423

open Real

theorem function_range_theorem (a : ℝ) (h₁ : a > 0) :
  (∀ x₁ ∈ Set.Icc (-1 : ℝ) 2, ∃ x₀ ∈ Set.Icc (-1 : ℝ) 2, 
    a * x₁ + 2 = x₀^2 - 2*x₀) → 0 < a ∧ a ≤ 1/2 := by
  sorry

end NUMINAMATH_CALUDE_function_range_theorem_l1884_188423


namespace NUMINAMATH_CALUDE_rectangle_dimensions_l1884_188426

/-- Represents a rectangle with length and width -/
structure Rectangle where
  length : ℝ
  width : ℝ

/-- Condition that length is greater than width -/
def Rectangle.lengthGreaterThanWidth (r : Rectangle) : Prop :=
  r.length > r.width

/-- Perimeter of a rectangle -/
def Rectangle.perimeter (r : Rectangle) : ℝ :=
  2 * (r.length + r.width)

/-- Area of a rectangle -/
def Rectangle.area (r : Rectangle) : ℝ :=
  r.length * r.width

/-- Theorem stating the dimensions of the rectangle -/
theorem rectangle_dimensions (r : Rectangle) 
  (h1 : r.lengthGreaterThanWidth)
  (h2 : r.perimeter = 18)
  (h3 : r.area = 18) :
  r.length = 6 ∧ r.width = 3 := by
  sorry


end NUMINAMATH_CALUDE_rectangle_dimensions_l1884_188426


namespace NUMINAMATH_CALUDE_new_ratio_first_term_l1884_188471

/-- Given an original ratio of 7:11, when 5 is added to both terms, 
    the first term of the new ratio is 12. -/
theorem new_ratio_first_term : 
  let original_first : ℕ := 7
  let original_second : ℕ := 11
  let added_number : ℕ := 5
  let new_first : ℕ := original_first + added_number
  new_first = 12 := by sorry

end NUMINAMATH_CALUDE_new_ratio_first_term_l1884_188471


namespace NUMINAMATH_CALUDE_isosceles_triangle_base_length_l1884_188491

/-- An isosceles triangle with perimeter 10 and one side length 2 -/
structure IsoscelesTriangle where
  perimeter : ℝ
  side_length : ℝ
  perimeter_eq : perimeter = 10
  side_length_eq : side_length = 2

/-- The base length of the isosceles triangle -/
def base_length (t : IsoscelesTriangle) : ℝ := 4

theorem isosceles_triangle_base_length (t : IsoscelesTriangle) :
  base_length t = 4 :=
by sorry

end NUMINAMATH_CALUDE_isosceles_triangle_base_length_l1884_188491


namespace NUMINAMATH_CALUDE_arccos_one_half_l1884_188496

theorem arccos_one_half : Real.arccos (1/2) = π/3 := by sorry

end NUMINAMATH_CALUDE_arccos_one_half_l1884_188496


namespace NUMINAMATH_CALUDE_volumes_not_equal_implies_cross_sections_not_equal_cross_sections_equal_not_implies_volumes_not_equal_l1884_188440

/-- Represents a geometric shape with height and volume -/
structure GeometricShape where
  height : ℝ
  volume : ℝ

/-- Represents the cross-sectional area of a shape at a given height -/
def crossSectionalArea (shape : GeometricShape) (h : ℝ) : ℝ :=
  sorry

/-- Cavalieri's Principle -/
axiom cavalieri_principle (A B : GeometricShape) :
  A.height = B.height →
  (∀ h, 0 ≤ h ∧ h ≤ A.height → crossSectionalArea A h = crossSectionalArea B h) →
  A.volume = B.volume

theorem volumes_not_equal_implies_cross_sections_not_equal
  (A B : GeometricShape) (h_height : A.height = B.height) :
  A.volume ≠ B.volume →
  ∃ h, 0 ≤ h ∧ h ≤ A.height ∧ crossSectionalArea A h ≠ crossSectionalArea B h :=
sorry

theorem cross_sections_equal_not_implies_volumes_not_equal
  (A B : GeometricShape) (h_height : A.height = B.height) :
  ¬(∀ h, 0 ≤ h ∧ h ≤ A.height → crossSectionalArea A h = crossSectionalArea B h →
    A.volume ≠ B.volume) :=
sorry

end NUMINAMATH_CALUDE_volumes_not_equal_implies_cross_sections_not_equal_cross_sections_equal_not_implies_volumes_not_equal_l1884_188440


namespace NUMINAMATH_CALUDE_quadratic_roots_l1884_188457

def f (m : ℝ) (x : ℝ) : ℝ := x^2 - 5*x + m

theorem quadratic_roots (m : ℝ) (h : f m 1 = 0) : 
  ∃ (r₁ r₂ : ℝ), r₁ = 1 ∧ r₂ = 4 ∧ ∀ x, f m x = 0 ↔ x = r₁ ∨ x = r₂ := by
  sorry

end NUMINAMATH_CALUDE_quadratic_roots_l1884_188457


namespace NUMINAMATH_CALUDE_cos_120_degrees_l1884_188447

theorem cos_120_degrees : Real.cos (2 * Real.pi / 3) = -1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_cos_120_degrees_l1884_188447


namespace NUMINAMATH_CALUDE_degree_of_minus_five_x_squared_y_l1884_188465

def monomial_degree (m : ℤ → ℤ → ℤ) : ℕ :=
  sorry

theorem degree_of_minus_five_x_squared_y :
  monomial_degree (fun x y ↦ -5 * x^2 * y) = 3 :=
sorry

end NUMINAMATH_CALUDE_degree_of_minus_five_x_squared_y_l1884_188465


namespace NUMINAMATH_CALUDE_solve_star_equation_l1884_188454

-- Define the star operation
def star (a b : ℝ) : ℝ := a * b + 3 * b - a

-- Theorem statement
theorem solve_star_equation :
  ∀ y : ℝ, star 7 y = 47 → y = 5.4 := by
  sorry

end NUMINAMATH_CALUDE_solve_star_equation_l1884_188454


namespace NUMINAMATH_CALUDE_division_multiplication_result_l1884_188444

theorem division_multiplication_result : (9 / 6) * 12 = 18 := by
  sorry

end NUMINAMATH_CALUDE_division_multiplication_result_l1884_188444


namespace NUMINAMATH_CALUDE_least_xy_value_l1884_188448

theorem least_xy_value (x y : ℕ+) (h : (1 : ℚ) / x + (1 : ℚ) / (3 * y) = (1 : ℚ) / 8) :
  (x * y : ℕ) ≥ 96 ∧ ∃ (a b : ℕ+), (a : ℚ) / b + (1 : ℚ) / (3 * b) = (1 : ℚ) / 8 ∧ (a * b : ℕ) = 96 :=
sorry

end NUMINAMATH_CALUDE_least_xy_value_l1884_188448


namespace NUMINAMATH_CALUDE_general_term_formula_l1884_188481

def S (n : ℕ) : ℤ := 3 * n^2 - 2 * n + 1

def a (n : ℕ) : ℤ :=
  if n = 1 then 2 else 6 * n - 5

theorem general_term_formula (n : ℕ) :
  (n = 1 ∧ a n = S n) ∨
  (n ≥ 2 ∧ a n = S n - S (n-1)) :=
sorry

end NUMINAMATH_CALUDE_general_term_formula_l1884_188481


namespace NUMINAMATH_CALUDE_one_negative_root_condition_l1884_188450

/-- A polynomial of the form x^4 + 3px^3 + 6x^2 + 3px + 1 -/
def polynomial (p : ℝ) (x : ℝ) : ℝ := x^4 + 3*p*x^3 + 6*x^2 + 3*p*x + 1

/-- The condition that the polynomial has exactly one negative real root -/
def has_one_negative_root (p : ℝ) : Prop :=
  ∃! x : ℝ, x < 0 ∧ polynomial p x = 0

/-- Theorem stating the condition on p for the polynomial to have exactly one negative real root -/
theorem one_negative_root_condition (p : ℝ) :
  has_one_negative_root p ↔ p ≥ 4/3 := by sorry

end NUMINAMATH_CALUDE_one_negative_root_condition_l1884_188450


namespace NUMINAMATH_CALUDE_average_of_xyz_l1884_188451

theorem average_of_xyz (x y z : ℝ) (h : (5 / 4) * (x + y + z) = 15) : 
  (x + y + z) / 3 = 4 := by
sorry

end NUMINAMATH_CALUDE_average_of_xyz_l1884_188451


namespace NUMINAMATH_CALUDE_triangle_angle_A_l1884_188484

theorem triangle_angle_A (A : Real) : 
  4 * Real.pi * Real.sin A - 3 * Real.arccos (-1/2) = 0 →
  (A = Real.pi / 6 ∨ A = 5 * Real.pi / 6) :=
by sorry

end NUMINAMATH_CALUDE_triangle_angle_A_l1884_188484


namespace NUMINAMATH_CALUDE_inequality_solution_set_a_range_for_inequality_l1884_188445

-- Define the functions f and g
def f (x : ℝ) : ℝ := |x + 3|
def g (x : ℝ) : ℝ := |2*x - 1|

-- Statement for the first part of the problem
theorem inequality_solution_set :
  {x : ℝ | f x < g x} = {x : ℝ | x < -2/3 ∨ x > 4} :=
sorry

-- Statement for the second part of the problem
theorem a_range_for_inequality (a : ℝ) :
  (∀ x : ℝ, 2 * f x + g x > a * x + 4) ↔ -1 < a ∧ a ≤ 4 :=
sorry

end NUMINAMATH_CALUDE_inequality_solution_set_a_range_for_inequality_l1884_188445


namespace NUMINAMATH_CALUDE_intersection_M_N_l1884_188409

-- Define the sets M and N
def M : Set ℝ := {y | ∃ x, y = x^2}
def N : Set ℝ := {x | ∃ y, (x^2/2) + y^2 = 1}

-- State the theorem
theorem intersection_M_N :
  M ∩ N = Set.Icc 0 (Real.sqrt 2) := by sorry

end NUMINAMATH_CALUDE_intersection_M_N_l1884_188409


namespace NUMINAMATH_CALUDE_number_of_digits_l1884_188432

theorem number_of_digits (N : ℕ) : N = 2^12 * 5^8 → (Nat.digits 10 N).length = 10 := by sorry

end NUMINAMATH_CALUDE_number_of_digits_l1884_188432


namespace NUMINAMATH_CALUDE_theater_camp_talents_l1884_188424

theorem theater_camp_talents (total_students : ℕ) 
  (cannot_sing cannot_dance both_talents : ℕ) : 
  total_students = 120 →
  cannot_sing = 30 →
  cannot_dance = 50 →
  both_talents = 10 →
  (total_students - cannot_sing) + (total_students - cannot_dance) - both_talents = 130 :=
by sorry

end NUMINAMATH_CALUDE_theater_camp_talents_l1884_188424


namespace NUMINAMATH_CALUDE_problem_solution_l1884_188461

/-- The probability that student A solves the problem -/
def prob_A : ℚ := 1/5

/-- The probability that student B solves the problem -/
def prob_B : ℚ := 1/4

/-- The probability that student C solves the problem -/
def prob_C : ℚ := 1/3

/-- The probability that exactly two students solve the problem -/
def prob_two_solve : ℚ := 
  prob_A * prob_B * (1 - prob_C) + 
  prob_A * prob_C * (1 - prob_B) + 
  (1 - prob_A) * prob_B * prob_C

/-- The probability that the problem is not solved -/
def prob_not_solved : ℚ := (1 - prob_A) * (1 - prob_B) * (1 - prob_C)

/-- The probability that the problem is solved -/
def prob_solved : ℚ := 1 - prob_not_solved

theorem problem_solution : 
  prob_two_solve = 3/20 ∧ prob_solved = 3/5 := by sorry

end NUMINAMATH_CALUDE_problem_solution_l1884_188461


namespace NUMINAMATH_CALUDE_quadratic_inequality_l1884_188419

-- Define the function f
def f (b c : ℝ) (x : ℝ) : ℝ := x^2 - b*x + c

-- State the theorem
theorem quadratic_inequality (b c : ℝ) :
  (∀ x, f b c (1 + x) = f b c (1 - x)) →
  f b c 0 = 3 →
  ∀ x, f b c (b^x) ≤ f b c (c^x) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_l1884_188419


namespace NUMINAMATH_CALUDE_arithmetic_sequence_ratio_l1884_188467

/-- Given arithmetic sequences a and b with sums S and T respectively, 
    if S_n / T_n = (2n-1) / (3n+2) for all n, then a_7 / b_7 = 25 / 41 -/
theorem arithmetic_sequence_ratio 
  (a b : ℕ → ℚ) 
  (S T : ℕ → ℚ) 
  (h1 : ∀ n, S n = (n / 2) * (a 1 + a n)) 
  (h2 : ∀ n, T n = (n / 2) * (b 1 + b n)) 
  (h3 : ∀ n, S n / T n = (2 * n - 1) / (3 * n + 2)) : 
  a 7 / b 7 = 25 / 41 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_ratio_l1884_188467


namespace NUMINAMATH_CALUDE_complex_modulus_l1884_188493

theorem complex_modulus (z : ℂ) (h : (2 - Complex.I) * z = Complex.I) : Complex.abs z = Real.sqrt 5 / 5 := by
  sorry

end NUMINAMATH_CALUDE_complex_modulus_l1884_188493


namespace NUMINAMATH_CALUDE_sugar_trader_profit_l1884_188420

/-- Represents the profit calculation for a sugar trader. -/
theorem sugar_trader_profit (Q : ℝ) (C : ℝ) : Q > 0 → C > 0 → 
  (Q - 1200) * (1.08 * C) + 1200 * (1.12 * C) = Q * C * 1.11 → Q = 1600 := by
  sorry

#check sugar_trader_profit

end NUMINAMATH_CALUDE_sugar_trader_profit_l1884_188420


namespace NUMINAMATH_CALUDE_largest_expression_l1884_188477

theorem largest_expression : 
  let expr1 := 2 + (-2)
  let expr2 := 2 - (-2)
  let expr3 := 2 * (-2)
  let expr4 := 2 / (-2)
  expr2 = max expr1 (max expr2 (max expr3 expr4)) := by sorry

end NUMINAMATH_CALUDE_largest_expression_l1884_188477


namespace NUMINAMATH_CALUDE_inverse_functions_theorem_l1884_188435

-- Define the set of graph labels
inductive GraphLabel
| A | B | C | D | E

-- Define a property for a function to have an inverse based on the Horizontal Line Test
def has_inverse (g : GraphLabel) : Prop :=
  match g with
  | GraphLabel.B => True
  | GraphLabel.C => True
  | _ => False

-- Define a function that checks if a graph passes the Horizontal Line Test
def passes_horizontal_line_test (g : GraphLabel) : Prop :=
  has_inverse g

-- Theorem statement
theorem inverse_functions_theorem :
  (∀ g : GraphLabel, has_inverse g ↔ passes_horizontal_line_test g) ∧
  (has_inverse GraphLabel.B ∧ has_inverse GraphLabel.C) ∧
  (¬ has_inverse GraphLabel.A ∧ ¬ has_inverse GraphLabel.D ∧ ¬ has_inverse GraphLabel.E) :=
sorry

end NUMINAMATH_CALUDE_inverse_functions_theorem_l1884_188435


namespace NUMINAMATH_CALUDE_bailey_dog_treats_l1884_188466

theorem bailey_dog_treats :
  let total_items : ℕ := 4 * 5
  let chew_toys : ℕ := 2
  let rawhide_bones : ℕ := 10
  let dog_treats : ℕ := total_items - (chew_toys + rawhide_bones)
  dog_treats = 8 := by
sorry

end NUMINAMATH_CALUDE_bailey_dog_treats_l1884_188466


namespace NUMINAMATH_CALUDE_binary_110011_equals_51_l1884_188439

def binary_to_decimal (b : List Bool) : ℕ :=
  b.enum.foldl (fun acc (i, bit) => acc + if bit then 2^i else 0) 0

def binary_110011 : List Bool := [true, true, false, false, true, true]

theorem binary_110011_equals_51 : binary_to_decimal binary_110011 = 51 := by
  sorry

end NUMINAMATH_CALUDE_binary_110011_equals_51_l1884_188439


namespace NUMINAMATH_CALUDE_middleAgedInPerformance_l1884_188433

/-- Represents the number of employees in each age group -/
structure EmployeeGroups where
  elderly : ℕ
  middleAged : ℕ
  young : ℕ

/-- Calculates the number of middle-aged employees selected in a stratified sample -/
def middleAgedSelected (total : ℕ) (groups : EmployeeGroups) (sampleSize : ℕ) : ℕ :=
  (sampleSize * groups.middleAged) / (groups.elderly + groups.middleAged + groups.young)

/-- Theorem: The number of middle-aged employees selected in the performance is 15 -/
theorem middleAgedInPerformance (total : ℕ) (groups : EmployeeGroups) (sampleSize : ℕ) 
    (h1 : total = 1200)
    (h2 : groups.elderly = 100)
    (h3 : groups.middleAged = 500)
    (h4 : groups.young = 600)
    (h5 : sampleSize = 36) :
  middleAgedSelected total groups sampleSize = 15 := by
  sorry

#eval middleAgedSelected 1200 ⟨100, 500, 600⟩ 36

end NUMINAMATH_CALUDE_middleAgedInPerformance_l1884_188433


namespace NUMINAMATH_CALUDE_distance_to_town_l1884_188476

theorem distance_to_town (d : ℝ) : 
  (∀ x, x ≥ 6 → d < x) →  -- A's statement is false
  (∀ y, y ≤ 5 → d > y) →  -- B's statement is false
  (∀ z, z ≤ 4 → d > z) →  -- C's statement is false
  d ∈ Set.Ioo 5 6 := by
sorry

end NUMINAMATH_CALUDE_distance_to_town_l1884_188476


namespace NUMINAMATH_CALUDE_f_properties_l1884_188431

noncomputable def f (x : ℝ) : ℝ := Real.cos x ^ 4 + 2 * Real.sin x * Real.cos x - Real.sin x ^ 4

theorem f_properties :
  (∀ x : ℝ, f (-x) ≠ f x ∧ f (-x) ≠ -f x) ∧
  (∀ ε > 0, ∃ x : ℝ, x > 0 ∧ x < π + ε ∧ f (x + π) = f x) ∧
  (∀ k : ℤ, StrictMonoOn f (Set.Icc (- 3 * Real.pi / 8 + k * Real.pi) (k * Real.pi + Real.pi / 8))) ∧
  (∃ x : ℝ, x ∈ Set.Icc 0 (Real.pi / 2) ∧ f x = Real.sqrt 2 ∧ ∀ y ∈ Set.Icc 0 (Real.pi / 2), f y ≤ f x) ∧
  (∃ x : ℝ, x ∈ Set.Icc 0 (Real.pi / 2) ∧ f x = -1 ∧ ∀ y ∈ Set.Icc 0 (Real.pi / 2), f y ≥ f x) :=
by sorry

end NUMINAMATH_CALUDE_f_properties_l1884_188431


namespace NUMINAMATH_CALUDE_max_daily_sales_l1884_188429

def salesVolume (t : ℕ) : ℝ := -2 * t + 200

def price (t : ℕ) : ℝ :=
  if t ≤ 30 then 12 * t + 30 else 45

def dailySales (t : ℕ) : ℝ :=
  salesVolume t * price t

theorem max_daily_sales :
  ∃ (t : ℕ), 1 ≤ t ∧ t ≤ 50 ∧ ∀ (s : ℕ), 1 ≤ s ∧ s ≤ 50 → dailySales s ≤ dailySales t ∧ dailySales t = 54600 := by
  sorry

end NUMINAMATH_CALUDE_max_daily_sales_l1884_188429


namespace NUMINAMATH_CALUDE_sad_children_count_l1884_188405

theorem sad_children_count (total : ℕ) (happy : ℕ) (neither : ℕ) 
  (boys : ℕ) (girls : ℕ) (happy_boys : ℕ) (sad_girls : ℕ) (neither_boys : ℕ)
  (h1 : total = 60)
  (h2 : happy = 30)
  (h3 : neither = 20)
  (h4 : boys = 19)
  (h5 : girls = 41)
  (h6 : happy_boys = 6)
  (h7 : sad_girls = 4)
  (h8 : neither_boys = 7)
  (h9 : total = happy + neither + (total - happy - neither)) :
  total - happy - neither = 10 := by
sorry

end NUMINAMATH_CALUDE_sad_children_count_l1884_188405


namespace NUMINAMATH_CALUDE_point_on_circle_l1884_188416

/-- The coordinates of a point on the unit circle after moving counterclockwise from (1, 0) by an arc length of 2π/3 -/
theorem point_on_circle (P : ℝ × ℝ) (Q : ℝ × ℝ) : 
  P = (1, 0) → 
  (Q.1 - P.1)^2 + (Q.2 - P.2)^2 = (2 * Real.pi / 3)^2 →
  Q.1^2 + Q.2^2 = 1 →
  Q = (-1/2, Real.sqrt 3 / 2) :=
by sorry

end NUMINAMATH_CALUDE_point_on_circle_l1884_188416


namespace NUMINAMATH_CALUDE_intersection_of_M_and_N_l1884_188401

def M : Set ℝ := {-1, 0, 1}
def N : Set ℝ := {y | ∃ x, y = Real.cos x}

theorem intersection_of_M_and_N : M ∩ N = M := by sorry

end NUMINAMATH_CALUDE_intersection_of_M_and_N_l1884_188401


namespace NUMINAMATH_CALUDE_scientific_notation_79000_l1884_188492

theorem scientific_notation_79000 : 79000 = 7.9 * (10 ^ 4) := by
  sorry

end NUMINAMATH_CALUDE_scientific_notation_79000_l1884_188492


namespace NUMINAMATH_CALUDE_opposite_of_negative_two_l1884_188499

theorem opposite_of_negative_two : 
  ∃ x : ℤ, x + (-2) = 0 ∧ x = 2 := by
  sorry

end NUMINAMATH_CALUDE_opposite_of_negative_two_l1884_188499


namespace NUMINAMATH_CALUDE_quadrilateral_diagonal_count_l1884_188498

/-- A quadrilateral with side lengths 9, 11, 15, and 14 has exactly 17 possible whole number lengths for a diagonal. -/
theorem quadrilateral_diagonal_count : ∃ (possible_lengths : Finset ℕ),
  (∀ d ∈ possible_lengths, 
    -- Triangle inequality for both triangles formed by the diagonal
    9 + d > 11 ∧ d + 11 > 9 ∧ 9 + 11 > d ∧
    14 + d > 15 ∧ d + 15 > 14 ∧ 14 + 15 > d) ∧
  (∀ d : ℕ, 
    (9 + d > 11 ∧ d + 11 > 9 ∧ 9 + 11 > d ∧
     14 + d > 15 ∧ d + 15 > 14 ∧ 14 + 15 > d) → d ∈ possible_lengths) ∧
  Finset.card possible_lengths = 17 := by
sorry

end NUMINAMATH_CALUDE_quadrilateral_diagonal_count_l1884_188498


namespace NUMINAMATH_CALUDE_square_equation_solution_l1884_188437

theorem square_equation_solution : 
  ∃ x : ℚ, ((3 * x + 15)^2 = 3 * (4 * x + 40)) ∧ (x = -5/3 ∨ x = -7) :=
by sorry

end NUMINAMATH_CALUDE_square_equation_solution_l1884_188437


namespace NUMINAMATH_CALUDE_cs_candidates_count_l1884_188408

theorem cs_candidates_count (m : ℕ) (n : ℕ) : 
  m = 4 → 
  m * (n.choose 2) = 84 → 
  n = 7 :=
by sorry

end NUMINAMATH_CALUDE_cs_candidates_count_l1884_188408


namespace NUMINAMATH_CALUDE_function_inequality_l1884_188403

open Real

theorem function_inequality (f : ℝ → ℝ) (h : ∀ x > 0, 2 * f x < x * (deriv f x) ∧ x * (deriv f x) < 3 * f x) :
  4 < f 2 / f 1 ∧ f 2 / f 1 < 8 := by
  sorry

end NUMINAMATH_CALUDE_function_inequality_l1884_188403


namespace NUMINAMATH_CALUDE_line_properties_l1884_188486

-- Define the line equation
def line_equation (k x y : ℝ) : Prop := y + 1 = k * (x - 2)

-- Theorem statement
theorem line_properties :
  -- 1. Countless lines through (2, -1)
  (∃ (S : Set ℝ), Infinite S ∧ ∀ k ∈ S, line_equation k 2 (-1)) ∧
  -- 2. Always passes through a fixed point
  (∃ (x₀ y₀ : ℝ), ∀ k, line_equation k x₀ y₀) ∧
  -- 3. Cannot be perpendicular to x-axis
  (∀ k, line_equation k 0 0 → k ≠ 0) :=
sorry

end NUMINAMATH_CALUDE_line_properties_l1884_188486


namespace NUMINAMATH_CALUDE_polynomial_remainder_l1884_188485

theorem polynomial_remainder (x : ℝ) : 
  (x^3 - 2*x^2 + 4*x - 1) % (x - 2) = 7 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_remainder_l1884_188485


namespace NUMINAMATH_CALUDE_max_k_value_l1884_188400

-- Define the circle C
def circle_C (x y : ℝ) : Prop := x^2 + y^2 - 8*x + 15 = 0

-- Define the line
def line (k x : ℝ) (y : ℝ) : Prop := y = k*x - 2

-- Define the condition for a point on the line to be the center of a circle with radius 1 that intersects C
def intersects_C (k x y : ℝ) : Prop :=
  line k x y ∧ ∃ (x' y' : ℝ), circle_C x' y' ∧ (x - x')^2 + (y - y')^2 = 1

-- Theorem statement
theorem max_k_value :
  ∃ (k_max : ℝ), k_max = 4/3 ∧
  (∀ k : ℝ, (∃ x y : ℝ, intersects_C k x y) → k ≤ k_max) ∧
  (∃ x y : ℝ, intersects_C k_max x y) :=
sorry

end NUMINAMATH_CALUDE_max_k_value_l1884_188400


namespace NUMINAMATH_CALUDE_two_week_egg_consumption_l1884_188487

/-- Calculates the total number of eggs consumed over a given number of days,
    given a daily egg consumption rate. -/
def totalEggsConsumed (dailyConsumption : ℕ) (days : ℕ) : ℕ :=
  dailyConsumption * days

/-- Theorem stating that consuming 3 eggs daily for 14 days results in 42 eggs consumed. -/
theorem two_week_egg_consumption :
  totalEggsConsumed 3 14 = 42 := by
  sorry

end NUMINAMATH_CALUDE_two_week_egg_consumption_l1884_188487


namespace NUMINAMATH_CALUDE_range_of_t_l1884_188407

-- Define the quadratic function
def f (x t : ℝ) : ℝ := x^2 - 3*x + t

-- Define the solution set A
def A (t : ℝ) : Set ℝ := {x | f x t ≤ 0}

-- Define the condition for the intersection
def intersection_nonempty (t : ℝ) : Prop := 
  ∃ x, x ∈ A t ∧ x ≤ t

-- State the theorem
theorem range_of_t : 
  ∀ t : ℝ, intersection_nonempty t ↔ t ∈ Set.Icc 0 (9/4) :=
sorry

end NUMINAMATH_CALUDE_range_of_t_l1884_188407


namespace NUMINAMATH_CALUDE_parallelogram_side_comparison_l1884_188475

structure Parallelogram where
  sides : Fin 4 → ℝ
  parallel : sides 0 = sides 2 ∧ sides 1 = sides 3

def inscribed (P Q : Parallelogram) : Prop :=
  ∀ i : Fin 4, P.sides i ≤ Q.sides i

theorem parallelogram_side_comparison 
  (P₁ P₂ P₃ : Parallelogram)
  (h₁ : inscribed P₂ P₁)
  (h₂ : inscribed P₃ P₂)
  (h₃ : ∀ i : Fin 4, P₃.sides i ≤ P₁.sides i) :
  ∃ i : Fin 4, P₁.sides i ≤ 2 * P₃.sides i :=
sorry

end NUMINAMATH_CALUDE_parallelogram_side_comparison_l1884_188475


namespace NUMINAMATH_CALUDE_triangle_existence_l1884_188411

theorem triangle_existence (x : ℤ) : 
  (5 + x > 0) ∧ (2*x + 1 > 0) ∧ (3*x > 0) ∧
  (5 + x + 2*x + 1 > 3*x) ∧ (5 + x + 3*x > 2*x + 1) ∧ (2*x + 1 + 3*x > 5 + x) ↔ 
  x ≥ 2 := by
sorry

end NUMINAMATH_CALUDE_triangle_existence_l1884_188411


namespace NUMINAMATH_CALUDE_binary_remainder_by_8_l1884_188442

/-- The remainder when 101110100101₂ is divided by 8 is 5. -/
theorem binary_remainder_by_8 : (101110100101 : Nat) % 8 = 5 := by
  sorry

end NUMINAMATH_CALUDE_binary_remainder_by_8_l1884_188442


namespace NUMINAMATH_CALUDE_triangle_circumcircle_diameter_l1884_188421

theorem triangle_circumcircle_diameter 
  (a : Real) 
  (B : Real) 
  (S : Real) : 
  a = 1 → 
  B = π / 4 → 
  S = 2 → 
  ∃ (b c d : Real), 
    c = 4 * Real.sqrt 2 ∧ 
    b^2 = a^2 + c^2 - 2*a*c*(Real.cos B) ∧ 
    b = 5 ∧ 
    d = b / (Real.sin B) ∧ 
    d = 5 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_triangle_circumcircle_diameter_l1884_188421


namespace NUMINAMATH_CALUDE_q_definition_l1884_188460

/-- Given p: x ≤ 1, and ¬p is a sufficient but not necessary condition for q,
    prove that q can be defined as x > 0 -/
theorem q_definition (x : ℝ) :
  (∃ p : Prop, (p ↔ x ≤ 1) ∧ 
   (∃ q : Prop, (¬p → q) ∧ ¬(q → ¬p))) →
  ∃ q : Prop, q ↔ x > 0 :=
by sorry

end NUMINAMATH_CALUDE_q_definition_l1884_188460


namespace NUMINAMATH_CALUDE_three_digit_number_theorem_l1884_188473

def is_three_digit (n : ℕ) : Prop := 100 ≤ n ∧ n ≤ 999

def satisfies_condition (n : ℕ) : Prop :=
  is_three_digit n ∧
  2 * n = (n % 100) * 10 + n / 100 + (n / 10 % 10) * 100 + n % 10

def solution_set : Finset ℕ :=
  {111, 222, 333, 370, 407, 444, 481, 518, 555, 592, 629, 666, 777, 888, 999}

theorem three_digit_number_theorem :
  ∀ n : ℕ, satisfies_condition n ↔ n ∈ solution_set := by sorry

end NUMINAMATH_CALUDE_three_digit_number_theorem_l1884_188473


namespace NUMINAMATH_CALUDE_truck_problem_l1884_188443

theorem truck_problem (T b c : ℝ) (hT : T > 0) (hb : b > 0) (hc : c > 0) :
  let x := (b * c + Real.sqrt (b^2 * c^2 + 4 * b * c * T)) / (2 * c)
  x * (x - b) * c = T * x ∧ (x - b) * (T / x + c) = T :=
by sorry

end NUMINAMATH_CALUDE_truck_problem_l1884_188443


namespace NUMINAMATH_CALUDE_unclaimed_fraction_is_correct_l1884_188412

/-- Represents a participant in the chocolate distribution --/
inductive Participant
  | Dave
  | Emma
  | Frank
  | George

/-- The ratio of chocolate distribution for each participant --/
def distribution_ratio (p : Participant) : Rat :=
  match p with
  | Participant.Dave => 4/10
  | Participant.Emma => 3/10
  | Participant.Frank => 2/10
  | Participant.George => 1/10

/-- The order in which participants claim their share --/
def claim_order : List Participant :=
  [Participant.Dave, Participant.Emma, Participant.Frank, Participant.George]

/-- Calculate the fraction of chocolates claimed by a participant --/
def claimed_fraction (p : Participant) (remaining : Rat) : Rat :=
  (distribution_ratio p) * remaining

/-- Calculate the fraction of chocolates that remains unclaimed --/
def unclaimed_fraction : Rat :=
  let initial_remaining : Rat := 1
  let final_remaining := claim_order.foldl
    (fun remaining p => remaining - claimed_fraction p remaining)
    initial_remaining
  final_remaining

/-- Theorem: The fraction of chocolates that remains unclaimed is 37.8/125 --/
theorem unclaimed_fraction_is_correct :
  unclaimed_fraction = 378/1250 := by
  sorry


end NUMINAMATH_CALUDE_unclaimed_fraction_is_correct_l1884_188412


namespace NUMINAMATH_CALUDE_parabola_uniqueness_l1884_188458

/-- A tangent line to a parabola -/
structure Tangent where
  line : Line2D

/-- A parabola in 2D space -/
structure Parabola where
  focus : Point2D
  directrix : Line2D

/-- The vertex tangent of a parabola -/
def vertexTangent (p : Parabola) : Tangent :=
  sorry

/-- Determines if a given tangent is valid for a parabola -/
def isValidTangent (p : Parabola) (t : Tangent) : Prop :=
  sorry

theorem parabola_uniqueness 
  (t : Tangent) (t₁ : Tangent) (t₂ : Tangent) : 
  ∃! p : Parabola, 
    (vertexTangent p = t) ∧ 
    (isValidTangent p t₁) ∧ 
    (isValidTangent p t₂) :=
sorry

end NUMINAMATH_CALUDE_parabola_uniqueness_l1884_188458


namespace NUMINAMATH_CALUDE_range_of_c_l1884_188425

def p (c : ℝ) : Prop := ∀ x y : ℝ, x < y → c^x > c^y

def q (c : ℝ) : Prop := ∀ x : ℝ, 2*c*x^2 + 2*x + 1 > 0

theorem range_of_c (c : ℝ) (h1 : c > 0) (h2 : (p c ∨ q c) ∧ ¬(p c ∧ q c)) :
  c ∈ Set.Ioo 0 (1/2) ∪ Set.Ici 1 :=
sorry

end NUMINAMATH_CALUDE_range_of_c_l1884_188425


namespace NUMINAMATH_CALUDE_quadratic_root_value_l1884_188415

theorem quadratic_root_value (b : ℝ) : 
  (∀ x : ℝ, x^2 + Real.sqrt (b - 1) * x + b^2 - 4 = 0 → x = 0) →
  (b - 1 ≥ 0) →
  b = 2 := by
sorry

end NUMINAMATH_CALUDE_quadratic_root_value_l1884_188415


namespace NUMINAMATH_CALUDE_area_bounded_by_function_and_double_tangent_l1884_188462

-- Define the function
def f (x : ℝ) : ℝ := -x^4 + 16*x^3 - 78*x^2 + 50*x - 2

-- Define the derivative of the function
def f' (x : ℝ) : ℝ := -4*x^3 + 48*x^2 - 156*x + 50

-- Theorem statement
theorem area_bounded_by_function_and_double_tangent :
  ∃ (a b : ℝ),
    a < b ∧
    f' a = f' b ∧
    (f b - f a) / (b - a) = f' a ∧
    (∫ (x : ℝ) in a..b, (((f b - f a) / (b - a)) * (x - a) + f a) - f x) = 1296 / 5 :=
sorry

end NUMINAMATH_CALUDE_area_bounded_by_function_and_double_tangent_l1884_188462


namespace NUMINAMATH_CALUDE_quadratic_properties_l1884_188464

-- Define the quadratic function
def f (a b c : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x + c

-- State the theorem
theorem quadratic_properties
  (a b c : ℝ)
  (ha : a < 0)
  (h_root : f a b c (-1) = 0)
  (h_sym : -b / (2 * a) = 1) :
  (a - b + c = 0) ∧
  (∀ m : ℝ, f a b c m ≤ -4 * a) ∧
  (∀ x₁ x₂ : ℝ, x₁ < x₂ → f a b c x₁ = -1 → f a b c x₂ = -1 → x₁ < -1 ∧ x₂ > 3) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_properties_l1884_188464


namespace NUMINAMATH_CALUDE_ratio_problem_l1884_188418

theorem ratio_problem (second_part : ℝ) (percent : ℝ) (first_part : ℝ) : 
  second_part = 5 →
  percent = 120 →
  first_part / second_part = percent / 100 →
  first_part = 6 :=
by
  sorry

end NUMINAMATH_CALUDE_ratio_problem_l1884_188418


namespace NUMINAMATH_CALUDE_selection_with_girl_count_l1884_188469

def num_boys : Nat := 4
def num_girls : Nat := 3
def num_selected : Nat := 3
def num_tasks : Nat := 3

theorem selection_with_girl_count :
  (Nat.choose (num_boys + num_girls) num_selected * Nat.factorial num_tasks) -
  (Nat.choose num_boys num_selected * Nat.factorial num_tasks) = 186 := by
  sorry

end NUMINAMATH_CALUDE_selection_with_girl_count_l1884_188469


namespace NUMINAMATH_CALUDE_apple_bags_theorem_l1884_188482

def is_valid_total (n : ℕ) : Prop :=
  70 ≤ n ∧ n ≤ 80 ∧ (∃ k : ℕ, n = 6 * k)

theorem apple_bags_theorem :
  ∀ n : ℕ, is_valid_total n ↔ (n = 72 ∨ n = 78) :=
by sorry

end NUMINAMATH_CALUDE_apple_bags_theorem_l1884_188482


namespace NUMINAMATH_CALUDE_three_oclock_angle_l1884_188413

/-- The angle between the hour hand and minute hand at a given time -/
def clock_angle (hour : ℕ) (minute : ℕ) : ℝ :=
  sorry

theorem three_oclock_angle :
  clock_angle 3 0 = π / 2 := by
  sorry

end NUMINAMATH_CALUDE_three_oclock_angle_l1884_188413


namespace NUMINAMATH_CALUDE_parabola_point_order_l1884_188404

/-- Parabola equation -/
def parabola (x y : ℝ) : Prop := y = (x - 1)^2 - 2

theorem parabola_point_order (a b c d : ℝ) :
  parabola a 2 →
  parabola b 6 →
  parabola c d →
  d < 1 →
  a < 0 →
  b > 0 →
  a < c ∧ c < b := by
  sorry

end NUMINAMATH_CALUDE_parabola_point_order_l1884_188404


namespace NUMINAMATH_CALUDE_f_composition_value_l1884_188449

noncomputable section

def f (x : ℝ) : ℝ :=
  if x ≤ 0 then Real.exp x else Real.log x

theorem f_composition_value : f (f (1 / Real.exp 1)) = 1 / Real.exp 1 := by
  sorry

end NUMINAMATH_CALUDE_f_composition_value_l1884_188449


namespace NUMINAMATH_CALUDE_undeveloped_sections_l1884_188494

/-- Proves that the number of undeveloped sections is 3 given the specified conditions -/
theorem undeveloped_sections
  (section_area : ℝ)
  (total_undeveloped_area : ℝ)
  (h1 : section_area = 2435)
  (h2 : total_undeveloped_area = 7305) :
  total_undeveloped_area / section_area = 3 := by
  sorry

end NUMINAMATH_CALUDE_undeveloped_sections_l1884_188494


namespace NUMINAMATH_CALUDE_palmer_photo_ratio_l1884_188470

/-- Given the information about Palmer's photo collection before and after her trip to Bali,
    prove that the ratio of new pictures taken in the second week to the first week is 3:1. -/
theorem palmer_photo_ratio (initial_photos : ℕ) (final_photos : ℕ) (first_week : ℕ) (third_fourth_weeks : ℕ)
    (h1 : initial_photos = 100)
    (h2 : final_photos = 380)
    (h3 : first_week = 50)
    (h4 : third_fourth_weeks = 80) :
    (final_photos - initial_photos - first_week - third_fourth_weeks) / first_week = 3 := by
  sorry

#check palmer_photo_ratio

end NUMINAMATH_CALUDE_palmer_photo_ratio_l1884_188470


namespace NUMINAMATH_CALUDE_distinct_paths_theorem_l1884_188483

/-- The number of distinct paths in a rectangular grid from point C to point D -/
def distinct_paths (right_steps : ℕ) (up_steps : ℕ) : ℕ :=
  Nat.choose (right_steps + up_steps) up_steps

/-- Theorem: The number of distinct paths from C to D is equal to (10 choose 3) -/
theorem distinct_paths_theorem :
  distinct_paths 7 3 = 120 := by
  sorry

end NUMINAMATH_CALUDE_distinct_paths_theorem_l1884_188483


namespace NUMINAMATH_CALUDE_ceiling_plus_one_l1884_188410

-- Define the ceiling function
noncomputable def ceiling (x : ℝ) : ℤ :=
  Int.ceil x

-- State the theorem
theorem ceiling_plus_one (x : ℝ) : ceiling (x + 1) = ceiling x + 1 := by
  sorry

end NUMINAMATH_CALUDE_ceiling_plus_one_l1884_188410


namespace NUMINAMATH_CALUDE_reflection_theorem_l1884_188427

/-- Reflects a point across the x-axis -/
def reflect_x (p : ℝ × ℝ) : ℝ × ℝ := (p.1, -p.2)

/-- Reflects a point across the line y = x - 2 -/
def reflect_line (p : ℝ × ℝ) : ℝ × ℝ :=
  let p_translated := (p.1, p.2 - 2)
  let p_reflected := (p_translated.2, p_translated.1)
  (p_reflected.1, p_reflected.2 + 2)

/-- The triangle ABC -/
def triangle_ABC : Set (ℝ × ℝ) :=
  {(3, 4), (6, 8), (5, 1)}

theorem reflection_theorem :
  let A : ℝ × ℝ := (3, 4)
  let A' := reflect_x A
  let A'' := reflect_line A'
  A'' = (-6, 5) :=
by
  sorry

end NUMINAMATH_CALUDE_reflection_theorem_l1884_188427


namespace NUMINAMATH_CALUDE_max_value_sum_sqrt_l1884_188436

theorem max_value_sum_sqrt (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (sum_eq_one : a + b + c = 1) :
  ∃ (max : ℝ), max = 3 * Real.sqrt 2 ∧ 
  (∀ a' b' c' : ℝ, 0 < a' → 0 < b' → 0 < c' → a' + b' + c' = 1 →
    Real.sqrt (3 * a' + 1) + Real.sqrt (3 * b' + 1) + Real.sqrt (3 * c' + 1) ≤ max) ∧
  Real.sqrt (3 * a + 1) + Real.sqrt (3 * b + 1) + Real.sqrt (3 * c + 1) = max :=
sorry

end NUMINAMATH_CALUDE_max_value_sum_sqrt_l1884_188436


namespace NUMINAMATH_CALUDE_sin_function_parameters_l1884_188406

def period : ℝ := 8
def max_x : ℝ := 1

theorem sin_function_parameters (ω φ : ℝ) : 
  (2 * π / period = ω) → 
  (ω * max_x + φ = π / 2) → 
  (ω = π / 4 ∧ φ = π / 4) := by sorry

end NUMINAMATH_CALUDE_sin_function_parameters_l1884_188406


namespace NUMINAMATH_CALUDE_twelfth_day_is_monday_l1884_188452

/-- Represents days of the week -/
inductive DayOfWeek
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday
  | Saturday
  | Sunday

/-- Represents a month with specific properties -/
structure Month where
  firstDay : DayOfWeek
  lastDay : DayOfWeek
  numDays : Nat
  numFridays : Nat

/-- Given a starting day and a number of days, calculates the resulting day of the week -/
def advanceDays (start : DayOfWeek) (days : Nat) : DayOfWeek :=
  sorry

/-- Theorem stating that under given conditions, the 12th day of the month is a Monday -/
theorem twelfth_day_is_monday (m : Month) 
  (h1 : m.numFridays = 5)
  (h2 : m.firstDay ≠ DayOfWeek.Friday)
  (h3 : m.lastDay ≠ DayOfWeek.Friday)
  (h4 : m.numDays ≥ 12) :
  advanceDays m.firstDay 11 = DayOfWeek.Monday :=
  sorry

end NUMINAMATH_CALUDE_twelfth_day_is_monday_l1884_188452


namespace NUMINAMATH_CALUDE_sqrt_65_bounds_l1884_188478

theorem sqrt_65_bounds (n : ℕ+) : n < Real.sqrt 65 ∧ Real.sqrt 65 < n + 1 → n = 8 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_65_bounds_l1884_188478


namespace NUMINAMATH_CALUDE_pencil_boxes_count_l1884_188453

theorem pencil_boxes_count (book_boxes : ℕ) (books_per_box : ℕ) (pencils_per_box : ℕ) (total_items : ℕ) :
  book_boxes = 19 →
  books_per_box = 46 →
  pencils_per_box = 170 →
  total_items = 1894 →
  (total_items - book_boxes * books_per_box) / pencils_per_box = 6 :=
by
  sorry

#check pencil_boxes_count

end NUMINAMATH_CALUDE_pencil_boxes_count_l1884_188453


namespace NUMINAMATH_CALUDE_binary_101101110_equals_octal_556_l1884_188402

/-- Converts a binary number (represented as a list of bits) to a decimal number -/
def binary_to_decimal (bits : List Nat) : Nat :=
  bits.enum.foldl (fun acc (i, b) => acc + b * 2^i) 0

/-- Converts a decimal number to an octal number (represented as a list of digits) -/
def decimal_to_octal (n : Nat) : List Nat :=
  if n = 0 then [0] else
    let rec aux (m : Nat) (acc : List Nat) : List Nat :=
      if m = 0 then acc else aux (m / 8) ((m % 8) :: acc)
    aux n []

theorem binary_101101110_equals_octal_556 :
  decimal_to_octal (binary_to_decimal [0, 1, 1, 1, 0, 1, 1, 0, 1]) = [5, 5, 6] := by
  sorry

end NUMINAMATH_CALUDE_binary_101101110_equals_octal_556_l1884_188402


namespace NUMINAMATH_CALUDE_jesse_stamp_collection_l1884_188417

theorem jesse_stamp_collection (total : ℕ) (european : ℕ) (asian : ℕ) 
  (h1 : total = 444)
  (h2 : european = 3 * asian)
  (h3 : total = european + asian) :
  european = 333 := by
sorry

end NUMINAMATH_CALUDE_jesse_stamp_collection_l1884_188417


namespace NUMINAMATH_CALUDE_total_birds_l1884_188441

theorem total_birds (cardinals : ℕ) (robins : ℕ) (blue_jays : ℕ) (sparrows : ℕ) (pigeons : ℕ) (finches : ℕ) : 
  cardinals = 3 →
  robins = 4 * cardinals →
  blue_jays = 2 * cardinals →
  sparrows = 3 * cardinals + 1 →
  pigeons = 3 * blue_jays →
  finches = robins / 2 →
  cardinals + robins + blue_jays + sparrows + pigeons + finches = 55 := by
sorry

end NUMINAMATH_CALUDE_total_birds_l1884_188441
