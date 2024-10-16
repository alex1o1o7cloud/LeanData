import Mathlib

namespace NUMINAMATH_CALUDE_max_product_sum_2000_l4025_402522

theorem max_product_sum_2000 :
  ∃ (a b : ℤ), a + b = 2000 ∧
  ∀ (x y : ℤ), x + y = 2000 → x * y ≤ a * b ∧
  a * b = 1000000 := by
  sorry

end NUMINAMATH_CALUDE_max_product_sum_2000_l4025_402522


namespace NUMINAMATH_CALUDE_prob_at_least_one_correct_l4025_402540

/-- The probability of subscribing to at least one of two newspapers -/
def prob_at_least_one (p1 p2 : ℝ) : ℝ :=
  1 - (1 - p1) * (1 - p2)

theorem prob_at_least_one_correct (p1 p2 : ℝ) 
  (h1 : 0 ≤ p1 ∧ p1 ≤ 1) (h2 : 0 ≤ p2 ∧ p2 ≤ 1) : 
  prob_at_least_one p1 p2 = 1 - (1 - p1) * (1 - p2) := by
  sorry

end NUMINAMATH_CALUDE_prob_at_least_one_correct_l4025_402540


namespace NUMINAMATH_CALUDE_bug_flower_consumption_l4025_402529

theorem bug_flower_consumption (num_bugs : ℝ) (flowers_per_bug : ℝ) : 
  num_bugs = 2.0 → flowers_per_bug = 1.5 → num_bugs * flowers_per_bug = 3.0 := by
  sorry

end NUMINAMATH_CALUDE_bug_flower_consumption_l4025_402529


namespace NUMINAMATH_CALUDE_johns_weight_change_l4025_402519

theorem johns_weight_change (initial_weight : ℝ) (loss_percentage : ℝ) (weight_gain : ℝ) : 
  initial_weight = 220 →
  loss_percentage = 10 →
  weight_gain = 2 →
  initial_weight * (1 - loss_percentage / 100) + weight_gain = 200 := by
  sorry

end NUMINAMATH_CALUDE_johns_weight_change_l4025_402519


namespace NUMINAMATH_CALUDE_quadratic_properties_l4025_402590

-- Define the quadratic function
def f (x : ℝ) : ℝ := -3 * x^2 + 9 * x + 4

-- Theorem stating the properties of the function
theorem quadratic_properties :
  (∃ (max_value : ℝ), ∀ (x : ℝ), f x ≤ max_value ∧ max_value = 10.75) ∧
  (∃ (max_point : ℝ), max_point > 0 ∧ max_point = 1.5 ∧ ∀ (x : ℝ), f x ≤ f max_point) ∧
  (∀ (x y : ℝ), x > 1.5 → y > x → f y < f x) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_properties_l4025_402590


namespace NUMINAMATH_CALUDE_six_digit_divisibility_l4025_402580

theorem six_digit_divisibility (A B : ℕ) 
  (hA : A ≥ 100 ∧ A < 1000) 
  (hB : B ≥ 100 ∧ B < 1000) 
  (hAnotDiv : ¬ (37 ∣ A)) 
  (hBnotDiv : ¬ (37 ∣ B)) 
  (hSum : 37 ∣ (A + B)) : 
  37 ∣ (1000 * A + B) := by
  sorry

end NUMINAMATH_CALUDE_six_digit_divisibility_l4025_402580


namespace NUMINAMATH_CALUDE_gcd_problem_l4025_402599

theorem gcd_problem (b : ℤ) (h : ∃ k : ℤ, b = 2700 * k) :
  Int.gcd (b^2 + 27*b + 75) (b + 25) = 25 := by
  sorry

end NUMINAMATH_CALUDE_gcd_problem_l4025_402599


namespace NUMINAMATH_CALUDE_comparison_of_products_l4025_402515

theorem comparison_of_products (a₁ a₂ b₁ b₂ : ℝ) 
  (h1 : a₁ < a₂) (h2 : b₁ < b₂) : a₁ * b₁ + a₂ * b₂ > a₁ * b₂ + a₂ * b₁ := by
  sorry

end NUMINAMATH_CALUDE_comparison_of_products_l4025_402515


namespace NUMINAMATH_CALUDE_survey_total_is_260_l4025_402501

/-- Represents the survey results of households using different brands of soap -/
structure SoapSurvey where
  neither : Nat
  onlyA : Nat
  onlyB : Nat
  both : Nat

/-- Calculates the total number of households surveyed -/
def totalHouseholds (survey : SoapSurvey) : Nat :=
  survey.neither + survey.onlyA + survey.onlyB + survey.both

/-- Theorem stating the total number of households surveyed is 260 -/
theorem survey_total_is_260 : ∃ (survey : SoapSurvey),
  survey.neither = 80 ∧
  survey.onlyA = 60 ∧
  survey.onlyB = 3 * survey.both ∧
  survey.both = 30 ∧
  totalHouseholds survey = 260 := by
  sorry

end NUMINAMATH_CALUDE_survey_total_is_260_l4025_402501


namespace NUMINAMATH_CALUDE_intersection_line_and_chord_length_l4025_402561

-- Define the circles
def C₁ (x y : ℝ) : Prop := x^2 + y^2 + 2*x - 6*y + 1 = 0
def C₂ (x y : ℝ) : Prop := x^2 + y^2 - 4*x + 2*y - 11 = 0

-- Define the line equation
def lineAB (x y : ℝ) : Prop := 3*x - 4*y + 6 = 0

-- Theorem statement
theorem intersection_line_and_chord_length :
  ∃ (A B : ℝ × ℝ),
    C₁ A.1 A.2 ∧ C₁ B.1 B.2 ∧
    C₂ A.1 A.2 ∧ C₂ B.1 B.2 ∧
    (∀ (x y : ℝ), C₁ x y ∧ C₂ x y → lineAB x y) ∧
    Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) = 24/5 :=
by sorry

end NUMINAMATH_CALUDE_intersection_line_and_chord_length_l4025_402561


namespace NUMINAMATH_CALUDE_smallest_number_l4025_402526

def numbers : List ℤ := [-2, -1, 1, 2]

theorem smallest_number : ∀ x ∈ numbers, -2 ≤ x := by
  sorry

end NUMINAMATH_CALUDE_smallest_number_l4025_402526


namespace NUMINAMATH_CALUDE_collinear_points_k_value_l4025_402553

variable (V : Type*) [AddCommGroup V] [Module ℝ V]
variable (a b : V)

-- Define vectors A, B, and D
def A (k : ℝ) := 2 • a + k • b
def B := a + b
def D := a - 2 • b

-- Define collinearity
def collinear (x y z : V) : Prop := ∃ (t : ℝ), y - x = t • (z - x)

-- Theorem statement
theorem collinear_points_k_value
  (ha : a ≠ 0)
  (hb : b ≠ 0)
  (hnc : ¬ ∃ (r : ℝ), a = r • b)
  (hcol : collinear V (A V a b k) (B V a b) (D V a b)) :
  k = -1 :=
sorry

end NUMINAMATH_CALUDE_collinear_points_k_value_l4025_402553


namespace NUMINAMATH_CALUDE_boys_neither_happy_nor_sad_l4025_402597

theorem boys_neither_happy_nor_sad 
  (total_children : Nat)
  (happy_children : Nat)
  (sad_children : Nat)
  (neither_children : Nat)
  (total_boys : Nat)
  (total_girls : Nat)
  (happy_boys : Nat)
  (sad_girls : Nat)
  (h1 : total_children = 60)
  (h2 : happy_children = 30)
  (h3 : sad_children = 10)
  (h4 : neither_children = 20)
  (h5 : total_boys = 19)
  (h6 : total_girls = 41)
  (h7 : happy_boys = 6)
  (h8 : sad_girls = 4)
  (h9 : total_children = happy_children + sad_children + neither_children)
  (h10 : total_children = total_boys + total_girls) :
  total_boys - happy_boys - (sad_children - sad_girls) = 7 := by
  sorry

end NUMINAMATH_CALUDE_boys_neither_happy_nor_sad_l4025_402597


namespace NUMINAMATH_CALUDE_geometric_sequence_value_l4025_402574

theorem geometric_sequence_value (b : ℝ) (h1 : b > 0) : 
  (∃ r : ℝ, r > 0 ∧ b = 30 * r ∧ 15/4 = b * r) → b = 15 * Real.sqrt 2 / 2 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_value_l4025_402574


namespace NUMINAMATH_CALUDE_mark_cookies_sold_l4025_402534

theorem mark_cookies_sold (n : ℕ) (mark_sold ann_sold : ℕ) : 
  n = 12 →
  mark_sold < n →
  ann_sold = n - 2 →
  mark_sold ≥ 1 →
  ann_sold ≥ 1 →
  mark_sold + ann_sold < n →
  mark_sold = n - 11 :=
by sorry

end NUMINAMATH_CALUDE_mark_cookies_sold_l4025_402534


namespace NUMINAMATH_CALUDE_quadratic_range_l4025_402538

theorem quadratic_range (a : ℝ) :
  (∀ x : ℝ, x^2 + (a - 1)*x + 1 ≥ 0) → -1 ≤ a ∧ a ≤ 3 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_range_l4025_402538


namespace NUMINAMATH_CALUDE_balloon_difference_is_one_l4025_402551

/-- The number of balloons Jake has more than Allan -/
def balloon_difference (allan_balloons jake_initial_balloons jake_bought_balloons : ℕ) : ℕ :=
  (jake_initial_balloons + jake_bought_balloons) - allan_balloons

/-- Theorem stating the difference in balloons between Jake and Allan -/
theorem balloon_difference_is_one :
  balloon_difference 6 3 4 = 1 := by
  sorry

end NUMINAMATH_CALUDE_balloon_difference_is_one_l4025_402551


namespace NUMINAMATH_CALUDE_paul_candy_count_l4025_402530

theorem paul_candy_count :
  ∀ (chocolate_boxes caramel_boxes pieces_per_box : ℕ),
    chocolate_boxes = 6 →
    caramel_boxes = 4 →
    pieces_per_box = 9 →
    chocolate_boxes * pieces_per_box + caramel_boxes * pieces_per_box = 90 :=
by
  sorry

end NUMINAMATH_CALUDE_paul_candy_count_l4025_402530


namespace NUMINAMATH_CALUDE_max_value_of_roots_sum_l4025_402571

/-- Given a quadratic polynomial x^2 - sx + q with roots r₁ and r₂ satisfying
    certain conditions, the maximum value of 1/r₁¹¹ + 1/r₂¹¹ is 2. -/
theorem max_value_of_roots_sum (s q r₁ r₂ : ℝ) : 
  r₁ + r₂ = s ∧ r₁ * r₂ = q ∧ 
  r₁ + r₂ = r₁^2 + r₂^2 ∧ 
  r₁ + r₂ = r₁^10 + r₂^10 →
  ∃ (M : ℝ), M = 2 ∧ ∀ (s' q' r₁' r₂' : ℝ), 
    (r₁' + r₂' = s' ∧ r₁' * r₂' = q' ∧ 
     r₁' + r₂' = r₁'^2 + r₂'^2 ∧ 
     r₁' + r₂' = r₁'^10 + r₂'^10) →
    1 / r₁'^11 + 1 / r₂'^11 ≤ M :=
by sorry

end NUMINAMATH_CALUDE_max_value_of_roots_sum_l4025_402571


namespace NUMINAMATH_CALUDE_ratio_of_Δy_to_Δx_l4025_402558

-- Define the curve
def f (x : ℝ) : ℝ := x^2 + 1

-- Define the two points on the curve
def point1 : ℝ × ℝ := (1, 2)
def point2 (Δx : ℝ) : ℝ × ℝ := (1 + Δx, f (1 + Δx))

-- Define Δy
def Δy (Δx : ℝ) : ℝ := (point2 Δx).2 - point1.2

-- Theorem statement
theorem ratio_of_Δy_to_Δx (Δx : ℝ) (h : Δx ≠ 0) :
  Δy Δx / Δx = Δx + 2 :=
by sorry

end NUMINAMATH_CALUDE_ratio_of_Δy_to_Δx_l4025_402558


namespace NUMINAMATH_CALUDE_complex_multiplication_division_l4025_402523

theorem complex_multiplication_division (z₁ z₂ : ℂ) :
  z₁ = 1 + Complex.I →
  z₂ = 2 - Complex.I →
  (z₁ * z₂) / Complex.I = 1 - 3 * Complex.I :=
by sorry

end NUMINAMATH_CALUDE_complex_multiplication_division_l4025_402523


namespace NUMINAMATH_CALUDE_product_equality_implies_n_equals_six_l4025_402546

theorem product_equality_implies_n_equals_six (n : ℕ) : 
  2 * 2 * 3 * 3 * 5 * 6 = 5 * 6 * n * n → n = 6 := by
  sorry

end NUMINAMATH_CALUDE_product_equality_implies_n_equals_six_l4025_402546


namespace NUMINAMATH_CALUDE_positive_solution_equation_l4025_402536

theorem positive_solution_equation (x : ℝ) :
  x = 20 + Real.sqrt 409 →
  x > 0 ∧
  (1 / 3) * (2 * x^2 + 3) = (x^2 - 40 * x - 8) * (x^2 + 20 * x + 4) :=
by sorry

end NUMINAMATH_CALUDE_positive_solution_equation_l4025_402536


namespace NUMINAMATH_CALUDE_pqr_value_l4025_402544

theorem pqr_value (p q r : ℂ) 
  (eq1 : p * q + 5 * q = -20)
  (eq2 : q * r + 5 * r = -20)
  (eq3 : r * p + 5 * p = -20) : 
  p * q * r = 80 := by
sorry

end NUMINAMATH_CALUDE_pqr_value_l4025_402544


namespace NUMINAMATH_CALUDE_mod_congruence_l4025_402511

theorem mod_congruence (m : ℕ) : 
  198 * 963 ≡ m [ZMOD 50] → 0 ≤ m → m < 50 → m = 24 := by
  sorry

end NUMINAMATH_CALUDE_mod_congruence_l4025_402511


namespace NUMINAMATH_CALUDE_matrix_product_is_zero_l4025_402584

def matrix_product_zero (d e f : ℝ) : Prop :=
  let A : Matrix (Fin 3) (Fin 3) ℝ := !![0, d, -e; -d, 0, f; e, -f, 0]
  let B : Matrix (Fin 3) (Fin 3) ℝ := !![d^2, d*e, d*f; d*e, e^2, e*f; d*f, e*f, f^2]
  A * B = 0

theorem matrix_product_is_zero (d e f : ℝ) : matrix_product_zero d e f := by
  sorry

end NUMINAMATH_CALUDE_matrix_product_is_zero_l4025_402584


namespace NUMINAMATH_CALUDE_negative_integer_equation_solution_l4025_402513

theorem negative_integer_equation_solution :
  ∀ N : ℤ, (N < 0) → (2 * N^2 + N = 15) → (N = -3) := by
  sorry

end NUMINAMATH_CALUDE_negative_integer_equation_solution_l4025_402513


namespace NUMINAMATH_CALUDE_dereks_car_dog_ratio_l4025_402563

/-- Represents Derek's possessions at different ages --/
structure DereksPossessions where
  dogs_at_6 : ℕ
  cars_at_6 : ℕ
  dogs_at_16 : ℕ
  cars_at_16 : ℕ

/-- Theorem stating the ratio of cars to dogs when Derek is 16 --/
theorem dereks_car_dog_ratio (d : DereksPossessions) 
  (h1 : d.dogs_at_6 = 90)
  (h2 : d.dogs_at_6 = 3 * d.cars_at_6)
  (h3 : d.dogs_at_16 = 120)
  (h4 : d.cars_at_16 = d.cars_at_6 + 210)
  : d.cars_at_16 / d.dogs_at_16 = 2 := by
  sorry

#check dereks_car_dog_ratio

end NUMINAMATH_CALUDE_dereks_car_dog_ratio_l4025_402563


namespace NUMINAMATH_CALUDE_ten_row_triangle_pieces_l4025_402516

/-- Calculates the sum of the first n natural numbers -/
def sum_of_naturals (n : ℕ) : ℕ := n * (n + 1) / 2

/-- Calculates the number of rods in an n-row triangle -/
def num_rods (n : ℕ) : ℕ := 3 * sum_of_naturals n

/-- Calculates the number of connectors in an n-row triangle -/
def num_connectors (n : ℕ) : ℕ := sum_of_naturals (n + 1)

/-- Calculates the total number of pieces in an n-row triangle -/
def total_pieces (n : ℕ) : ℕ := num_rods n + num_connectors n

theorem ten_row_triangle_pieces :
  total_pieces 10 = 231 := by
  sorry

end NUMINAMATH_CALUDE_ten_row_triangle_pieces_l4025_402516


namespace NUMINAMATH_CALUDE_a_fourth_minus_b_fourth_l4025_402572

theorem a_fourth_minus_b_fourth (a b : ℝ) 
  (h1 : a - b = 1) 
  (h2 : a^2 - b^2 = -1) : 
  a^4 - b^4 = -1 := by
sorry

end NUMINAMATH_CALUDE_a_fourth_minus_b_fourth_l4025_402572


namespace NUMINAMATH_CALUDE_no_five_digit_flippy_divisible_by_11_l4025_402573

/-- A flippy number is a number whose digits alternate between two distinct digits. -/
def is_flippy (n : ℕ) : Prop :=
  ∃ (a b : ℕ), a ≠ b ∧ 
  (∃ (d1 d2 d3 d4 d5 : ℕ), 
    n = d1 * 10000 + d2 * 1000 + d3 * 100 + d4 * 10 + d5 ∧
    ((d1 = a ∧ d2 = b ∧ d3 = a ∧ d4 = b ∧ d5 = a) ∨
     (d1 = b ∧ d2 = a ∧ d3 = b ∧ d4 = a ∧ d5 = b)))

/-- A number is five digits long if it's between 10000 and 99999, inclusive. -/
def is_five_digit (n : ℕ) : Prop := 10000 ≤ n ∧ n ≤ 99999

theorem no_five_digit_flippy_divisible_by_11 : 
  ¬∃ (n : ℕ), is_flippy n ∧ is_five_digit n ∧ n % 11 = 0 :=
sorry

end NUMINAMATH_CALUDE_no_five_digit_flippy_divisible_by_11_l4025_402573


namespace NUMINAMATH_CALUDE_sqrt_108_simplification_l4025_402507

theorem sqrt_108_simplification : Real.sqrt 108 = 6 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_108_simplification_l4025_402507


namespace NUMINAMATH_CALUDE_hack_represents_8634_l4025_402535

-- Define the mapping of letters to digits
def letter_to_digit : Char → Nat
| 'Q' => 0
| 'U' => 1
| 'I' => 2
| 'C' => 3
| 'K' => 4
| 'M' => 5
| 'A' => 6
| 'T' => 7
| 'H' => 8
| 'S' => 9
| _ => 0  -- Default case for other characters

-- Define the code word
def code_word : List Char := ['H', 'A', 'C', 'K']

-- Theorem to prove
theorem hack_represents_8634 :
  (code_word.map letter_to_digit).foldl (fun acc d => acc * 10 + d) 0 = 8634 := by
  sorry

end NUMINAMATH_CALUDE_hack_represents_8634_l4025_402535


namespace NUMINAMATH_CALUDE_asymptote_sum_l4025_402552

/-- 
Given a rational function y = x / (x³ + Ax² + Bx + C) where A, B, C are integers,
if the graph has vertical asymptotes at x = -3, 0, and 2,
then A + B + C = -5
-/
theorem asymptote_sum (A B C : ℤ) : 
  (∀ x : ℝ, x ≠ -3 ∧ x ≠ 0 ∧ x ≠ 2 → 
    ∃ y : ℝ, y = x / (x^3 + A*x^2 + B*x + C)) →
  A + B + C = -5 := by
  sorry

end NUMINAMATH_CALUDE_asymptote_sum_l4025_402552


namespace NUMINAMATH_CALUDE_function_range_l4025_402547

theorem function_range (m : ℝ) : 
  (∃ x y : ℝ, x ≠ y ∧ x^2 - 2*m*x + m + 2 = 0 ∧ y^2 - 2*m*y + m + 2 = 0) ∧ 
  (∀ x ≥ 1, ∀ y ≥ x, (y^2 - 2*m*y + m + 2) ≥ (x^2 - 2*m*x + m + 2)) →
  m < -1 :=
by sorry

end NUMINAMATH_CALUDE_function_range_l4025_402547


namespace NUMINAMATH_CALUDE_union_of_A_and_B_l4025_402517

def A : Set ℝ := {x | x^2 - 3*x + 2 = 0}
def B : Set ℝ := {x | x^3 = x}

theorem union_of_A_and_B : A ∪ B = {-1, 0, 1, 2} := by sorry

end NUMINAMATH_CALUDE_union_of_A_and_B_l4025_402517


namespace NUMINAMATH_CALUDE_saroj_current_age_l4025_402505

/-- Represents the age of a person at different points in time -/
structure PersonAge where
  sixYearsAgo : ℕ
  current : ℕ
  fourYearsHence : ℕ

/-- The problem statement -/
theorem saroj_current_age 
  (vimal saroj : PersonAge)
  (h1 : vimal.sixYearsAgo * 5 = saroj.sixYearsAgo * 6)
  (h2 : vimal.fourYearsHence * 10 = saroj.fourYearsHence * 11)
  (h3 : vimal.current = vimal.sixYearsAgo + 6)
  (h4 : saroj.current = saroj.sixYearsAgo + 6)
  (h5 : vimal.fourYearsHence = vimal.current + 4)
  (h6 : saroj.fourYearsHence = saroj.current + 4)
  : saroj.current = 16 := by
  sorry

end NUMINAMATH_CALUDE_saroj_current_age_l4025_402505


namespace NUMINAMATH_CALUDE_kitchen_planks_l4025_402504

/-- Represents the number of wooden planks used in Andrew's house flooring project. -/
structure FlooringProject where
  bedroom : ℕ
  livingRoom : ℕ
  guestBedroom : ℕ
  hallway : ℕ
  kitchen : ℕ
  leftover : ℕ
  replacedBedroom : ℕ
  replacedGuestBedroom : ℕ

/-- Theorem stating the number of planks used for the kitchen in Andrew's flooring project. -/
theorem kitchen_planks (project : FlooringProject) 
    (h1 : project.bedroom = 8)
    (h2 : project.livingRoom = 20)
    (h3 : project.guestBedroom = project.bedroom - 2)
    (h4 : project.hallway = 4 * 2)
    (h5 : project.leftover = 6)
    (h6 : project.replacedBedroom = 3)
    (h7 : project.replacedGuestBedroom = 3)
    : project.kitchen = 6 := by
  sorry


end NUMINAMATH_CALUDE_kitchen_planks_l4025_402504


namespace NUMINAMATH_CALUDE_quartic_equation_real_roots_l4025_402502

theorem quartic_equation_real_roots :
  ∃ (a b c d : ℝ), a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d ∧
  (∀ x : ℝ, 3 * x^4 + x^3 - 6 * x^2 + x + 3 = 0 ↔ (x = a ∨ x = b ∨ x = c ∨ x = d)) :=
by sorry

end NUMINAMATH_CALUDE_quartic_equation_real_roots_l4025_402502


namespace NUMINAMATH_CALUDE_mixed_nuts_cost_per_serving_l4025_402569

/-- Calculates the cost per serving of mixed nuts in cents -/
def cost_per_serving (bag_cost : ℚ) (bag_content : ℚ) (coupon_value : ℚ) (serving_size : ℚ) : ℚ :=
  ((bag_cost - coupon_value) / bag_content) * serving_size * 100

/-- Theorem: The cost per serving of mixed nuts is 50 cents -/
theorem mixed_nuts_cost_per_serving :
  cost_per_serving 25 40 5 1 = 50 := by
  sorry

#eval cost_per_serving 25 40 5 1

end NUMINAMATH_CALUDE_mixed_nuts_cost_per_serving_l4025_402569


namespace NUMINAMATH_CALUDE_simplify_fraction_l4025_402591

theorem simplify_fraction (x : ℝ) (h : x ≠ 2) :
  (1 + 1 / (x - 2)) / ((x - x^2) / (x - 2)) = -(x - 1) / x := by
  sorry

end NUMINAMATH_CALUDE_simplify_fraction_l4025_402591


namespace NUMINAMATH_CALUDE_sphere_volume_circumscribing_cube_l4025_402575

/-- The volume of a sphere circumscribing a cube with edge length 2 cm -/
theorem sphere_volume_circumscribing_cube : 
  let cube_edge : ℝ := 2
  let sphere_radius : ℝ := cube_edge * Real.sqrt 3 / 2
  let sphere_volume : ℝ := (4 / 3) * Real.pi * sphere_radius ^ 3
  sphere_volume = 4 * Real.sqrt 3 * Real.pi := by
  sorry

end NUMINAMATH_CALUDE_sphere_volume_circumscribing_cube_l4025_402575


namespace NUMINAMATH_CALUDE_students_taking_neither_l4025_402541

theorem students_taking_neither (total : ℕ) (chem : ℕ) (bio : ℕ) (both : ℕ) 
  (h1 : total = 75) 
  (h2 : chem = 40) 
  (h3 : bio = 35) 
  (h4 : both = 25) : 
  total - (chem + bio - both) = 25 :=
by sorry

end NUMINAMATH_CALUDE_students_taking_neither_l4025_402541


namespace NUMINAMATH_CALUDE_triangle_perimeter_l4025_402500

theorem triangle_perimeter (a b c : ℝ) (A B C : ℝ) :
  b = 9 →
  a = 2 * c →
  B = π / 3 →
  a + b + c = 9 + 9 * Real.sqrt 3 :=
by sorry

end NUMINAMATH_CALUDE_triangle_perimeter_l4025_402500


namespace NUMINAMATH_CALUDE_age_sum_proof_l4025_402528

-- Define the son's current age
def son_age : ℕ := 36

-- Define the father's current age
def father_age : ℕ := 72

-- Theorem stating the conditions and the result to prove
theorem age_sum_proof :
  -- 18 years ago, father was 3 times as old as son
  (father_age - 18 = 3 * (son_age - 18)) ∧
  -- Now, father is twice as old as son
  (father_age = 2 * son_age) →
  -- The sum of their present ages is 108
  son_age + father_age = 108 := by
  sorry

end NUMINAMATH_CALUDE_age_sum_proof_l4025_402528


namespace NUMINAMATH_CALUDE_ordered_pairs_satisfying_inequalities_l4025_402512

theorem ordered_pairs_satisfying_inequalities :
  ∃! (s : Finset (ℤ × ℤ)), 
    (∀ (a b : ℤ), (a, b) ∈ s ↔ 
      (a^2 + b^2 < 16 ∧ 
       a^2 + b^2 < 8*a ∧ 
       a^2 + b^2 < 8*b)) ∧
    s.card = 6 := by
  sorry

end NUMINAMATH_CALUDE_ordered_pairs_satisfying_inequalities_l4025_402512


namespace NUMINAMATH_CALUDE_factorization_of_a_squared_minus_2a_l4025_402562

theorem factorization_of_a_squared_minus_2a (a : ℝ) : a^2 - 2*a = a*(a - 2) := by
  sorry

end NUMINAMATH_CALUDE_factorization_of_a_squared_minus_2a_l4025_402562


namespace NUMINAMATH_CALUDE_additional_amount_for_free_shipping_l4025_402560

def free_shipping_threshold : ℝ := 50.00
def discount_rate : ℝ := 0.25
def book1_price : ℝ := 13.00
def book2_price : ℝ := 15.00
def book3_price : ℝ := 10.00
def book4_price : ℝ := 10.00

def discounted_price (price : ℝ) : ℝ := price * (1 - discount_rate)

def total_cost : ℝ := 
  discounted_price book1_price + discounted_price book2_price + book3_price + book4_price

theorem additional_amount_for_free_shipping : 
  free_shipping_threshold - total_cost = 9.00 := by sorry

end NUMINAMATH_CALUDE_additional_amount_for_free_shipping_l4025_402560


namespace NUMINAMATH_CALUDE_assignment_n_plus_one_increases_by_one_l4025_402576

/-- Represents a variable in a programming language -/
structure Variable where
  name : String
  value : Int

/-- Represents an expression in a programming language -/
inductive Expression where
  | Const : Int → Expression
  | Var : Variable → Expression
  | Add : Expression → Expression → Expression

/-- Represents an assignment statement in a programming language -/
structure AssignmentStatement where
  lhs : Variable
  rhs : Expression

/-- Evaluates an expression given the current state of variables -/
def evalExpression (expr : Expression) (state : List Variable) : Int :=
  match expr with
  | Expression.Const n => n
  | Expression.Var v => v.value
  | Expression.Add e1 e2 => evalExpression e1 state + evalExpression e2 state

/-- Executes an assignment statement and returns the updated state -/
def executeAssignment (stmt : AssignmentStatement) (state : List Variable) : List Variable :=
  let newValue := evalExpression stmt.rhs state
  state.map fun v => if v.name = stmt.lhs.name then { v with value := newValue } else v

/-- Theorem: N=N+1 increases the value of N by 1 -/
theorem assignment_n_plus_one_increases_by_one (n : Variable) (state : List Variable) :
  let stmt : AssignmentStatement := { lhs := n, rhs := Expression.Add (Expression.Var n) (Expression.Const 1) }
  let newState := executeAssignment stmt state
  let oldValue := (state.find? fun v => v.name = n.name).map (fun v => v.value)
  let newValue := (newState.find? fun v => v.name = n.name).map (fun v => v.value)
  (oldValue.isSome ∧ newValue.isSome) →
  newValue = oldValue.map (fun v => v + 1) :=
by
  sorry

end NUMINAMATH_CALUDE_assignment_n_plus_one_increases_by_one_l4025_402576


namespace NUMINAMATH_CALUDE_min_sum_of_product_36_l4025_402596

theorem min_sum_of_product_36 (c d : ℤ) (h : c * d = 36) :
  ∃ (m : ℤ), m = -37 ∧ c + d ≥ m ∧ ∃ (c' d' : ℤ), c' * d' = 36 ∧ c' + d' = m := by
  sorry

end NUMINAMATH_CALUDE_min_sum_of_product_36_l4025_402596


namespace NUMINAMATH_CALUDE_median_on_hypotenuse_of_right_triangle_l4025_402592

theorem median_on_hypotenuse_of_right_triangle 
  (a b : ℝ) (ha : a = 6) (hb : b = 8) : 
  let c := Real.sqrt (a^2 + b^2)
  let m := c / 2
  m = 5 := by sorry

end NUMINAMATH_CALUDE_median_on_hypotenuse_of_right_triangle_l4025_402592


namespace NUMINAMATH_CALUDE_coastline_scientific_notation_l4025_402570

theorem coastline_scientific_notation : 
  37515000 = 3.7515 * (10 : ℝ)^7 := by
  sorry

end NUMINAMATH_CALUDE_coastline_scientific_notation_l4025_402570


namespace NUMINAMATH_CALUDE_cans_per_bag_l4025_402510

theorem cans_per_bag (total_bags : ℕ) (total_cans : ℕ) (h1 : total_bags = 8) (h2 : total_cans = 40) :
  total_cans / total_bags = 5 := by
  sorry

end NUMINAMATH_CALUDE_cans_per_bag_l4025_402510


namespace NUMINAMATH_CALUDE_smallest_visible_sum_l4025_402589

/-- Represents a die with 6 faces -/
structure Die :=
  (faces : Fin 6 → ℕ)
  (opposite_sum : ∀ i : Fin 3, faces i + faces (i + 3) = 7)

/-- Represents a 4x4x4 cube made of dice -/
def GiantCube := Fin 4 → Fin 4 → Fin 4 → Die

/-- The sum of visible values on the 6 faces of the giant cube -/
def visible_sum (cube : GiantCube) : ℕ :=
  sorry

/-- The theorem stating the smallest possible sum of visible values -/
theorem smallest_visible_sum (cube : GiantCube) :
  visible_sum cube ≥ 144 :=
sorry

end NUMINAMATH_CALUDE_smallest_visible_sum_l4025_402589


namespace NUMINAMATH_CALUDE_regular_star_n_value_l4025_402568

/-- Represents an n-pointed regular star diagram -/
structure RegularStar where
  n : ℕ
  edge_length : ℝ
  angle_A : ℝ
  angle_B : ℝ

/-- The properties of the regular star diagram -/
def is_valid_regular_star (star : RegularStar) : Prop :=
  star.n > 0 ∧
  star.edge_length > 0 ∧
  star.angle_A > 0 ∧
  star.angle_B > 0 ∧
  star.angle_A = (5 / 14) * star.angle_B ∧
  star.n * (star.angle_A + star.angle_B) = 360

theorem regular_star_n_value (star : RegularStar) 
  (h : is_valid_regular_star star) : star.n = 133 := by
  sorry

#check regular_star_n_value

end NUMINAMATH_CALUDE_regular_star_n_value_l4025_402568


namespace NUMINAMATH_CALUDE_product_trailing_zeros_l4025_402598

def trailing_zeros (n : ℕ) : ℕ := sorry

theorem product_trailing_zeros : 
  trailing_zeros (45 * 320 * 125) = 5 := by sorry

end NUMINAMATH_CALUDE_product_trailing_zeros_l4025_402598


namespace NUMINAMATH_CALUDE_arrangement_counts_l4025_402539

/-- The number of singing programs -/
def num_singing : ℕ := 5

/-- The number of dance programs -/
def num_dance : ℕ := 4

/-- The total number of programs -/
def total_programs : ℕ := num_singing + num_dance

/-- Calculates the number of permutations of n items taken r at a time -/
def permutations (n : ℕ) (r : ℕ) : ℕ :=
  Nat.factorial n / Nat.factorial (n - r)

/-- The number of arrangements where no two dance programs are adjacent -/
def non_adjacent_arrangements : ℕ :=
  permutations num_singing num_singing * permutations (num_singing + 1) num_dance

/-- The number of arrangements with alternating singing and dance programs -/
def alternating_arrangements : ℕ :=
  permutations num_singing num_singing * permutations num_dance num_dance

theorem arrangement_counts :
  non_adjacent_arrangements = 43200 ∧ alternating_arrangements = 2880 := by
  sorry

end NUMINAMATH_CALUDE_arrangement_counts_l4025_402539


namespace NUMINAMATH_CALUDE_unique_prime_cube_l4025_402559

theorem unique_prime_cube (p : ℕ) : 
  Prime p ∧ ∃ (a : ℕ), a > 0 ∧ (16 * p + 1 = a^3) ↔ p = 307 := by
  sorry

end NUMINAMATH_CALUDE_unique_prime_cube_l4025_402559


namespace NUMINAMATH_CALUDE_tangent_line_triangle_area_l4025_402564

noncomputable def f (x : ℝ) : ℝ := Real.log x + x

theorem tangent_line_triangle_area :
  let P : ℝ × ℝ := (1, f 1)
  let f' : ℝ → ℝ := fun x ↦ 1 / x + 1
  let tangent_slope : ℝ := f' 1
  let tangent_line (x : ℝ) : ℝ := tangent_slope * (x - P.1) + P.2
  let x_intercept : ℝ := P.1 - P.2 / tangent_slope
  let y_intercept : ℝ := tangent_line 0
  (1/2 : ℝ) * abs (x_intercept * y_intercept) = 1/4 := by
  sorry

end NUMINAMATH_CALUDE_tangent_line_triangle_area_l4025_402564


namespace NUMINAMATH_CALUDE_max_value_of_quadratic_l4025_402521

open Real

theorem max_value_of_quadratic (x : ℝ) (h : 0 < x ∧ x < 1) : 
  ∃ (max_val : ℝ), max_val = 1/4 ∧ ∀ y, 0 < y ∧ y < 1 → y * (1 - y) ≤ max_val :=
by sorry

end NUMINAMATH_CALUDE_max_value_of_quadratic_l4025_402521


namespace NUMINAMATH_CALUDE_inequality_proof_l4025_402514

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  a^a * b^b * c^c ≥ 1 / (a * b * c) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l4025_402514


namespace NUMINAMATH_CALUDE_abs_sum_inequality_solution_existence_l4025_402555

theorem abs_sum_inequality_solution_existence (a : ℝ) :
  (∃ x : ℝ, |x - 4| + |x - 3| < a) ↔ a > 1 := by
  sorry

end NUMINAMATH_CALUDE_abs_sum_inequality_solution_existence_l4025_402555


namespace NUMINAMATH_CALUDE_steak_weight_for_tommy_family_l4025_402588

/-- Given a family where each member wants one pound of steak, 
    this function calculates the weight of each steak needed to be purchased. -/
def steak_weight (family_size : ℕ) (num_steaks : ℕ) : ℚ :=
  (family_size : ℚ) / (num_steaks : ℚ)

/-- Proves that for a family of 5 members, each wanting one pound of steak,
    and needing to buy 4 steaks, the weight of each steak is 1.25 pounds. -/
theorem steak_weight_for_tommy_family : 
  steak_weight 5 4 = 5/4 := by sorry

end NUMINAMATH_CALUDE_steak_weight_for_tommy_family_l4025_402588


namespace NUMINAMATH_CALUDE_polynomial_roots_sum_absolute_l4025_402525

theorem polynomial_roots_sum_absolute (m : ℤ) (p q r : ℤ) : 
  (∃ (m : ℤ), ∀ (x : ℤ), x^3 - 707*x + m = 0 ↔ x = p ∨ x = q ∨ x = r) →
  |p| + |q| + |r| = 122 := by
sorry

end NUMINAMATH_CALUDE_polynomial_roots_sum_absolute_l4025_402525


namespace NUMINAMATH_CALUDE_equal_area_rectangles_l4025_402527

/-- Represents the dimensions of a rectangle -/
structure Rectangle where
  length : ℝ
  width : ℝ

/-- Calculates the area of a rectangle -/
def area (r : Rectangle) : ℝ := r.length * r.width

theorem equal_area_rectangles (carol jordan : Rectangle) 
  (h1 : carol.length = 15)
  (h2 : carol.width = 20)
  (h3 : jordan.length = 6)
  (h4 : area carol = area jordan) :
  jordan.width = 50 := by
  sorry

end NUMINAMATH_CALUDE_equal_area_rectangles_l4025_402527


namespace NUMINAMATH_CALUDE_unique_triplet_satisfying_conditions_l4025_402583

theorem unique_triplet_satisfying_conditions :
  ∃! (a b c : ℝ),
    ({a^2 - 4*c, b^2 - 2*a, c^2 - 2*b} : Set ℝ) = {a - c, b - 4*c, a + b} ∧
    2*a + 2*b + 6 = 5*c ∧
    (a^2 - 4*c ≠ b^2 - 2*a ∧ a^2 - 4*c ≠ c^2 - 2*b ∧ b^2 - 2*a ≠ c^2 - 2*b) ∧
    (a - c ≠ b - 4*c ∧ a - c ≠ a + b ∧ b - 4*c ≠ a + b) ∧
    a = 1 ∧ b = 1 ∧ c = 2 := by
  sorry

end NUMINAMATH_CALUDE_unique_triplet_satisfying_conditions_l4025_402583


namespace NUMINAMATH_CALUDE_angle_range_in_triangle_l4025_402531

open Real

theorem angle_range_in_triangle (A : ℝ) (h1 : sin A + cos A > 0) (h2 : tan A < sin A) :
  π / 2 < A ∧ A < 3 * π / 4 := by
  sorry

end NUMINAMATH_CALUDE_angle_range_in_triangle_l4025_402531


namespace NUMINAMATH_CALUDE_simplify_expression_l4025_402593

theorem simplify_expression (m : ℝ) (hm : m > 0) :
  (m^(1/2) * 3*m * 4*m) / ((6*m)^5 * m^(1/4)) = 1 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l4025_402593


namespace NUMINAMATH_CALUDE_probability_theorem_l4025_402586

/-- The set of ball numbers in the bag -/
def BallNumbers : Finset ℕ := {1, 2, 3, 4}

/-- The probability of drawing two balls with sum not exceeding 4 -/
def prob_sum_not_exceeding_4 : ℚ :=
  (Finset.filter (fun pair => pair.1 + pair.2 ≤ 4) (BallNumbers.product BallNumbers)).card /
  (BallNumbers.product BallNumbers).card

/-- The probability of drawing two balls with replacement where n < m + 2 -/
def prob_n_less_than_m_plus_2 : ℚ :=
  (Finset.filter (fun pair => pair.2 < pair.1 + 2) (BallNumbers.product BallNumbers)).card /
  (BallNumbers.product BallNumbers).card

theorem probability_theorem :
  prob_sum_not_exceeding_4 = 1/3 ∧ prob_n_less_than_m_plus_2 = 13/16 := by
  sorry

end NUMINAMATH_CALUDE_probability_theorem_l4025_402586


namespace NUMINAMATH_CALUDE_equilateral_triangle_perimeter_l4025_402543

theorem equilateral_triangle_perimeter (s : ℝ) (h : s > 0) : 
  (s^2 * Real.sqrt 3) / 4 = s → 3 * s = 4 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_equilateral_triangle_perimeter_l4025_402543


namespace NUMINAMATH_CALUDE_repeating_decimal_eq_fraction_l4025_402565

/-- The repeating decimal 0.6̄3 as a real number -/
def repeating_decimal : ℚ := 19/30

/-- Theorem stating that the repeating decimal 0.6̄3 is equal to 19/30 -/
theorem repeating_decimal_eq_fraction : repeating_decimal = 19/30 := by sorry

end NUMINAMATH_CALUDE_repeating_decimal_eq_fraction_l4025_402565


namespace NUMINAMATH_CALUDE_small_paintings_sold_l4025_402537

/-- Given the prices of paintings and sales information, prove the number of small paintings sold. -/
theorem small_paintings_sold
  (large_price : ℕ)
  (small_price : ℕ)
  (large_sold : ℕ)
  (total_earnings : ℕ)
  (h1 : large_price = 100)
  (h2 : small_price = 80)
  (h3 : large_sold = 5)
  (h4 : total_earnings = 1140) :
  (total_earnings - large_price * large_sold) / small_price = 8 := by
  sorry

end NUMINAMATH_CALUDE_small_paintings_sold_l4025_402537


namespace NUMINAMATH_CALUDE_power_of_four_three_halves_l4025_402579

theorem power_of_four_three_halves : (4 : ℝ) ^ (3/2) = 8 := by
  sorry

end NUMINAMATH_CALUDE_power_of_four_three_halves_l4025_402579


namespace NUMINAMATH_CALUDE_max_median_is_four_point_five_l4025_402518

/-- Represents the soda shop scenario -/
structure SodaShop where
  total_cans : ℕ
  total_customers : ℕ
  min_cans_per_customer : ℕ
  h_total_cans : total_cans = 310
  h_total_customers : total_customers = 120
  h_min_cans : min_cans_per_customer = 1

/-- Calculates the maximum possible median number of cans bought per customer -/
def max_median_cans (shop : SodaShop) : ℚ :=
  sorry

/-- Theorem stating that the maximum possible median is 4.5 -/
theorem max_median_is_four_point_five (shop : SodaShop) :
  max_median_cans shop = 4.5 := by
  sorry

end NUMINAMATH_CALUDE_max_median_is_four_point_five_l4025_402518


namespace NUMINAMATH_CALUDE_exists_tricolor_right_triangle_l4025_402545

/-- A color type with three possible values -/
inductive Color
  | One
  | Two
  | Three

/-- A point on the integer plane -/
structure Point where
  x : Int
  y : Int

/-- A coloring of the integer plane -/
def Coloring := Point → Color

/-- Predicate for a right triangle -/
def is_right_triangle (p1 p2 p3 : Point) : Prop :=
  (p2.x - p1.x) * (p3.x - p1.x) + (p2.y - p1.y) * (p3.y - p1.y) = 0

/-- Main theorem -/
theorem exists_tricolor_right_triangle (c : Coloring) 
  (h1 : ∃ p : Point, c p = Color.One)
  (h2 : ∃ p : Point, c p = Color.Two)
  (h3 : ∃ p : Point, c p = Color.Three) :
  ∃ p1 p2 p3 : Point, 
    is_right_triangle p1 p2 p3 ∧ 
    c p1 ≠ c p2 ∧ c p2 ≠ c p3 ∧ c p3 ≠ c p1 :=
sorry

end NUMINAMATH_CALUDE_exists_tricolor_right_triangle_l4025_402545


namespace NUMINAMATH_CALUDE_hyperbola_foci_coordinates_l4025_402524

/-- Given a hyperbola passing through a specific point, prove the coordinates of its foci -/
theorem hyperbola_foci_coordinates :
  ∀ (a : ℝ),
  (((2 * Real.sqrt 2) ^ 2) / a ^ 2) - 1 ^ 2 = 1 →
  ∃ (c : ℝ),
  c ^ 2 = 5 ∧
  (∀ (x y : ℝ), x ^ 2 / a ^ 2 - y ^ 2 = 1 → 
    ((x = c ∧ y = 0) ∨ (x = -c ∧ y = 0))) :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_foci_coordinates_l4025_402524


namespace NUMINAMATH_CALUDE_painted_cubes_l4025_402533

theorem painted_cubes (total_cubes : ℕ) (unpainted_cubes : ℕ) (side_length : ℕ) : 
  unpainted_cubes = 24 →
  side_length = 5 →
  total_cubes = side_length^3 →
  total_cubes - unpainted_cubes = 101 := by
  sorry

end NUMINAMATH_CALUDE_painted_cubes_l4025_402533


namespace NUMINAMATH_CALUDE_vector_from_origin_to_line_l4025_402556

/-- A line parameterized by t -/
structure ParametricLine where
  x : ℝ → ℝ
  y : ℝ → ℝ

/-- The given line -/
def givenLine : ParametricLine where
  x := λ t => 3 * t + 1
  y := λ t => 2 * t + 3

/-- Check if a vector is parallel to another vector -/
def isParallel (v w : ℝ × ℝ) : Prop :=
  ∃ k : ℝ, v.1 = k * w.1 ∧ v.2 = k * w.2

/-- Check if a point lies on the given line -/
def liesOnLine (p : ℝ × ℝ) : Prop :=
  ∃ t : ℝ, p.1 = givenLine.x t ∧ p.2 = givenLine.y t

theorem vector_from_origin_to_line : 
  liesOnLine (-3, -2) ∧ 
  isParallel (-3, -2) (3, 2) := by
  sorry

#check vector_from_origin_to_line

end NUMINAMATH_CALUDE_vector_from_origin_to_line_l4025_402556


namespace NUMINAMATH_CALUDE_coffee_stock_problem_l4025_402503

/-- Represents the coffee stock problem --/
theorem coffee_stock_problem 
  (initial_stock : ℝ) 
  (initial_decaf_percent : ℝ) 
  (second_batch_decaf_percent : ℝ) 
  (final_decaf_percent : ℝ) 
  (second_batch : ℝ)
  (h1 : initial_stock = 400)
  (h2 : initial_decaf_percent = 0.20)
  (h3 : second_batch_decaf_percent = 0.60)
  (h4 : final_decaf_percent = 0.28000000000000004)
  (h5 : (initial_stock * initial_decaf_percent + second_batch * second_batch_decaf_percent) / 
        (initial_stock + second_batch) = final_decaf_percent) : 
  second_batch = 100 := by
  sorry

end NUMINAMATH_CALUDE_coffee_stock_problem_l4025_402503


namespace NUMINAMATH_CALUDE_octal_367_equals_decimal_247_l4025_402554

-- Define the octal number as a list of digits
def octal_number : List Nat := [3, 6, 7]

-- Define the conversion function from octal to decimal
def octal_to_decimal (digits : List Nat) : Nat :=
  digits.enum.foldr (fun (i, d) acc => acc + d * (8 ^ i)) 0

-- Theorem statement
theorem octal_367_equals_decimal_247 :
  octal_to_decimal octal_number = 247 := by
  sorry

end NUMINAMATH_CALUDE_octal_367_equals_decimal_247_l4025_402554


namespace NUMINAMATH_CALUDE_fixed_point_of_line_l4025_402577

/-- The fixed point of the line (2k-1)x-(k+3)y-(k-11)=0 for all real k -/
theorem fixed_point_of_line (k : ℝ) : (2*k - 1) * 2 - (k + 3) * 3 - (k - 11) = 0 := by
  sorry

end NUMINAMATH_CALUDE_fixed_point_of_line_l4025_402577


namespace NUMINAMATH_CALUDE_triangle_cosine_inequality_l4025_402595

theorem triangle_cosine_inequality (A B C : ℝ) (h_triangle : A + B + C = π) :
  1/3 * (Real.cos A + Real.cos B + Real.cos C) ≤ 1/2 ∧
  1/2 ≤ Real.sqrt (1/3 * (Real.cos A^2 + Real.cos B^2 + Real.cos C^2)) := by
  sorry

end NUMINAMATH_CALUDE_triangle_cosine_inequality_l4025_402595


namespace NUMINAMATH_CALUDE_line_intersection_range_l4025_402578

theorem line_intersection_range (m : ℝ) : 
  (∀ x y : ℝ, y = (m + 1) * x + m - 1 → (x = 0 → y ≤ 0)) → m ≤ 1 := by
  sorry

end NUMINAMATH_CALUDE_line_intersection_range_l4025_402578


namespace NUMINAMATH_CALUDE_min_value_reciprocal_l4025_402557

theorem min_value_reciprocal (a b : ℝ) (h1 : a + a * b + 2 * b = 30) (h2 : a > 0) (h3 : b > 0) :
  (∀ x y : ℝ, x > 0 → y > 0 → x + x * y + 2 * y = 30 → 1 / (a * b) ≤ 1 / (x * y)) ∧ 
  (∃ x y : ℝ, x > 0 ∧ y > 0 ∧ x + x * y + 2 * y = 30 ∧ 1 / (x * y) = 1 / 18) :=
by sorry

end NUMINAMATH_CALUDE_min_value_reciprocal_l4025_402557


namespace NUMINAMATH_CALUDE_geometric_arithmetic_ratio_l4025_402567

/-- Given a geometric sequence {a_n} with common ratio q (q ≠ 1),
    if a_1, a_3, a_2 form an arithmetic sequence, then q = -1/2 -/
theorem geometric_arithmetic_ratio (a : ℕ → ℝ) (q : ℝ) (h1 : q ≠ 1)
    (h2 : ∀ n, a (n + 1) = a n * q)  -- geometric sequence condition
    (h3 : 2 * a 3 = a 1 + a 2)       -- arithmetic sequence condition
    : q = -1/2 := by
  sorry

end NUMINAMATH_CALUDE_geometric_arithmetic_ratio_l4025_402567


namespace NUMINAMATH_CALUDE_wooden_block_surface_area_l4025_402532

theorem wooden_block_surface_area (A₁ A₂ A₃ A₄ A₅ A₆ A₇ : ℕ) 
  (h₁ : A₁ = 148)
  (h₂ : A₂ = 46)
  (h₃ : A₃ = 72)
  (h₄ : A₄ = 28)
  (h₅ : A₅ = 88)
  (h₆ : A₆ = 126)
  (h₇ : A₇ = 58) :
  ∃ A₈ : ℕ, A₈ = 22 ∧ A₁ + A₂ + A₃ + A₄ - (A₅ + A₆ + A₇) = A₈ :=
by sorry

end NUMINAMATH_CALUDE_wooden_block_surface_area_l4025_402532


namespace NUMINAMATH_CALUDE_first_year_after_2020_with_digit_sum_15_l4025_402550

def sumOfDigits (n : ℕ) : ℕ :=
  if n < 10 then n else n % 10 + sumOfDigits (n / 10)

def isFirstYearAfter2020WithDigitSum15 (year : ℕ) : Prop :=
  year > 2020 ∧ 
  sumOfDigits year = 15 ∧ 
  ∀ y, 2020 < y ∧ y < year → sumOfDigits y ≠ 15

theorem first_year_after_2020_with_digit_sum_15 :
  isFirstYearAfter2020WithDigitSum15 2058 := by
  sorry

end NUMINAMATH_CALUDE_first_year_after_2020_with_digit_sum_15_l4025_402550


namespace NUMINAMATH_CALUDE_joyce_apples_l4025_402549

/-- Proves that if Joyce starts with 75 apples and gives 52 to Larry, she ends up with 23 apples -/
theorem joyce_apples : ∀ (initial_apples given_apples remaining_apples : ℕ),
  initial_apples = 75 →
  given_apples = 52 →
  remaining_apples = initial_apples - given_apples →
  remaining_apples = 23 := by
  sorry


end NUMINAMATH_CALUDE_joyce_apples_l4025_402549


namespace NUMINAMATH_CALUDE_tan_70_cos_10_expression_l4025_402582

theorem tan_70_cos_10_expression : 
  Real.tan (70 * π / 180) * Real.cos (10 * π / 180) * (Real.sqrt 3 * Real.tan (20 * π / 180) - 1) = -1 := by
  sorry

end NUMINAMATH_CALUDE_tan_70_cos_10_expression_l4025_402582


namespace NUMINAMATH_CALUDE_officer_selection_theorem_l4025_402566

/-- Represents the Chemistry Club and its officer selection process -/
structure ChemistryClub where
  totalMembers : Nat
  aliceAndBobCondition : Bool
  ronaldCondition : Bool

/-- Calculates the number of ways to select officers -/
def selectOfficers (club : ChemistryClub) : Nat :=
  let withoutAliceBob := (club.totalMembers - 3) * (club.totalMembers - 4) * (club.totalMembers - 5)
  let withAliceBob := 6
  withoutAliceBob + withAliceBob

/-- The main theorem stating the number of ways to select officers -/
theorem officer_selection_theorem (club : ChemistryClub) 
  (h1 : club.totalMembers = 25)
  (h2 : club.aliceAndBobCondition = true)
  (h3 : club.ronaldCondition = true) :
  selectOfficers club = 9246 := by
  sorry

#eval selectOfficers { totalMembers := 25, aliceAndBobCondition := true, ronaldCondition := true }

end NUMINAMATH_CALUDE_officer_selection_theorem_l4025_402566


namespace NUMINAMATH_CALUDE_fraction_inequality_l4025_402581

theorem fraction_inequality (x : ℝ) :
  -3 ≤ x ∧ x ≤ 3 →
  (8 * x - 3 < 9 + 5 * x ↔ -3 ≤ x ∧ x < 3) :=
by sorry

end NUMINAMATH_CALUDE_fraction_inequality_l4025_402581


namespace NUMINAMATH_CALUDE_representation_of_2008_l4025_402520

theorem representation_of_2008 : ∃ (a b c : ℕ), 
  2008 = a + 40 * b + 40 * c ∧ 
  (1 : ℚ) / a + (b : ℚ) / 40 + (c : ℚ) / 40 = 1 := by
  sorry

end NUMINAMATH_CALUDE_representation_of_2008_l4025_402520


namespace NUMINAMATH_CALUDE_completing_square_transformation_l4025_402585

theorem completing_square_transformation (x : ℝ) :
  (x^2 - 8*x - 1 = 0) ↔ ((x - 4)^2 = 17) :=
by sorry

end NUMINAMATH_CALUDE_completing_square_transformation_l4025_402585


namespace NUMINAMATH_CALUDE_sum_less_than_addends_implies_negative_l4025_402594

theorem sum_less_than_addends_implies_negative (a b : ℝ) :
  (a + b < a ∧ a + b < b) → (a < 0 ∧ b < 0) := by
  sorry

end NUMINAMATH_CALUDE_sum_less_than_addends_implies_negative_l4025_402594


namespace NUMINAMATH_CALUDE_rent_increase_percentage_l4025_402587

/-- Proves that the rent increase percentage is 25% given the specified conditions -/
theorem rent_increase_percentage 
  (num_friends : ℕ) 
  (initial_avg_rent : ℝ) 
  (new_avg_rent : ℝ) 
  (original_rent : ℝ) : 
  num_friends = 4 → 
  initial_avg_rent = 800 → 
  new_avg_rent = 850 → 
  original_rent = 800 → 
  (new_avg_rent * num_friends - initial_avg_rent * num_friends) / original_rent * 100 = 25 := by
  sorry

end NUMINAMATH_CALUDE_rent_increase_percentage_l4025_402587


namespace NUMINAMATH_CALUDE_line_passes_through_points_l4025_402548

/-- Given a line y = (1/2)x + c passing through points (b+4, 5) and (-2, 2),
    prove that c = 3 -/
theorem line_passes_through_points (b : ℝ) :
  ∃ c : ℝ, (5 : ℝ) = (1/2 : ℝ) * (b + 4) + c ∧ (2 : ℝ) = (1/2 : ℝ) * (-2) + c ∧ c = 3 := by
  sorry

end NUMINAMATH_CALUDE_line_passes_through_points_l4025_402548


namespace NUMINAMATH_CALUDE_sum_of_roots_l4025_402506

theorem sum_of_roots (k p : ℝ) (x₁ x₂ : ℝ) :
  (4 * x₁^2 - k * x₁ - p = 0) →
  (4 * x₂^2 - k * x₂ - p = 0) →
  (x₁ ≠ x₂) →
  (x₁ + x₂ = k / 4) := by
sorry

end NUMINAMATH_CALUDE_sum_of_roots_l4025_402506


namespace NUMINAMATH_CALUDE_saras_quarters_l4025_402509

/-- Sara's quarters problem -/
theorem saras_quarters (initial_quarters final_quarters dad_quarters : ℕ) 
  (h1 : initial_quarters = 21)
  (h2 : dad_quarters = 49)
  (h3 : final_quarters = initial_quarters + dad_quarters) :
  final_quarters = 70 := by
  sorry

end NUMINAMATH_CALUDE_saras_quarters_l4025_402509


namespace NUMINAMATH_CALUDE_customer_income_proof_l4025_402542

/-- Proves that given a group of 50 customers with an average income of $45,000, 
    where 10 of these customers have an average income of $55,000, 
    the average income of the remaining 40 customers is $42,500. -/
theorem customer_income_proof (total_customers : Nat) (wealthy_customers : Nat)
  (remaining_customers : Nat) (total_avg_income : ℝ) (wealthy_avg_income : ℝ) :
  total_customers = 50 →
  wealthy_customers = 10 →
  remaining_customers = total_customers - wealthy_customers →
  total_avg_income = 45000 →
  wealthy_avg_income = 55000 →
  (total_customers * total_avg_income - wealthy_customers * wealthy_avg_income) / remaining_customers = 42500 :=
by sorry

end NUMINAMATH_CALUDE_customer_income_proof_l4025_402542


namespace NUMINAMATH_CALUDE_mohamed_age_ratio_l4025_402508

/-- Represents a person's age -/
structure Age :=
  (value : ℕ)

/-- Represents the current year -/
def currentYear : ℕ := 2023

theorem mohamed_age_ratio (kody : Age) (mohamed : Age) :
  kody.value = 32 →
  (currentYear - 4 : ℕ) - kody.value + 4 = 2 * ((currentYear - 4 : ℕ) - mohamed.value + 4) →
  ∃ k : ℕ, mohamed.value = 30 * k →
  mohamed.value / 30 = 2 := by
  sorry

end NUMINAMATH_CALUDE_mohamed_age_ratio_l4025_402508
