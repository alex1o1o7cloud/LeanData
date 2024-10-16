import Mathlib

namespace NUMINAMATH_CALUDE_range_of_expression_l958_95868

theorem range_of_expression (x y : ℝ) (hx : x ≥ 0) (hy : y ≥ 0) (hxy : x + y ≤ 1) :
  2/3 ≤ 4*x^2 + 4*y^2 + (1 - x - y)^2 ∧ 4*x^2 + 4*y^2 + (1 - x - y)^2 ≤ 4 := by
  sorry

end NUMINAMATH_CALUDE_range_of_expression_l958_95868


namespace NUMINAMATH_CALUDE_triangle_side_and_area_l958_95860

theorem triangle_side_and_area 
  (A B C : ℝ) -- Angles of the triangle
  (a b c : ℝ) -- Sides of the triangle
  (h1 : c = 2)
  (h2 : Real.sin A = 2 * Real.sin C)
  (h3 : Real.cos B = 1/4)
  (h4 : a = c * Real.sin A / Real.sin C) -- Sine law
  (h5 : 0 < Real.sin C) -- Assumption to avoid division by zero
  (h6 : 0 ≤ A ∧ A < π) -- Assumption for valid angle A
  (h7 : 0 ≤ B ∧ B < π) -- Assumption for valid angle B
  (h8 : 0 ≤ C ∧ C < π) -- Assumption for valid angle C
  : a = 4 ∧ (1/2 * a * c * Real.sin B = Real.sqrt 15) := by
  sorry

#check triangle_side_and_area

end NUMINAMATH_CALUDE_triangle_side_and_area_l958_95860


namespace NUMINAMATH_CALUDE_sevenPointOneTwoThreeBar_eq_fraction_l958_95859

/-- Represents a repeating decimal with an integer part and a repeating fractional part -/
structure RepeatingDecimal where
  integerPart : ℤ
  repeatingPart : ℕ

/-- Converts a RepeatingDecimal to a rational number -/
def RepeatingDecimal.toRational (x : RepeatingDecimal) : ℚ :=
  sorry

/-- The repeating decimal 7.123̄ -/
def sevenPointOneTwoThreeBar : RepeatingDecimal :=
  { integerPart := 7, repeatingPart := 123 }

theorem sevenPointOneTwoThreeBar_eq_fraction :
  RepeatingDecimal.toRational sevenPointOneTwoThreeBar = 2372 / 333 := by
  sorry

end NUMINAMATH_CALUDE_sevenPointOneTwoThreeBar_eq_fraction_l958_95859


namespace NUMINAMATH_CALUDE_rectangle_area_l958_95818

theorem rectangle_area (perimeter : ℝ) (length_ratio width_ratio : ℕ) : 
  perimeter = 280 →
  length_ratio = 5 →
  width_ratio = 2 →
  ∃ (length width : ℝ),
    length / width = length_ratio / width_ratio ∧
    2 * (length + width) = perimeter ∧
    length * width = 4000 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_area_l958_95818


namespace NUMINAMATH_CALUDE_power_difference_equality_l958_95848

theorem power_difference_equality (a b : ℕ) (h1 : a = 3) (h2 : b = 2) :
  (a^b)^a - (b^a)^b = 665 := by
  sorry

end NUMINAMATH_CALUDE_power_difference_equality_l958_95848


namespace NUMINAMATH_CALUDE_banquet_plates_l958_95872

/-- The total number of plates served at a banquet -/
theorem banquet_plates (lobster_rolls : ℕ) (spicy_hot_noodles : ℕ) (seafood_noodles : ℕ)
  (h1 : lobster_rolls = 25)
  (h2 : spicy_hot_noodles = 14)
  (h3 : seafood_noodles = 16) :
  lobster_rolls + spicy_hot_noodles + seafood_noodles = 55 := by
  sorry

end NUMINAMATH_CALUDE_banquet_plates_l958_95872


namespace NUMINAMATH_CALUDE_right_angle_vector_condition_l958_95844

/-- Given two vectors OA and OB in a Cartesian coordinate plane, 
    if the angle ABO is 90 degrees, then the t-coordinate of OA is 5. -/
theorem right_angle_vector_condition (t : ℝ) : 
  let OA : ℝ × ℝ := (-1, t)
  let OB : ℝ × ℝ := (2, 2)
  (OB.1 * (OB.1 - OA.1) + OB.2 * (OB.2 - OA.2) = 0) →
  t = 5 := by
sorry

end NUMINAMATH_CALUDE_right_angle_vector_condition_l958_95844


namespace NUMINAMATH_CALUDE_amusement_park_line_count_l958_95837

theorem amusement_park_line_count : 
  ∀ (eunji_position : ℕ) (people_behind : ℕ),
    eunji_position = 6 →
    people_behind = 7 →
    eunji_position + people_behind = 13 := by
  sorry

end NUMINAMATH_CALUDE_amusement_park_line_count_l958_95837


namespace NUMINAMATH_CALUDE_tan_theta_in_terms_of_x_l958_95895

theorem tan_theta_in_terms_of_x (θ : Real) (x : Real) 
  (h_acute : 0 < θ ∧ θ < π / 2) 
  (h_x : x > 1) 
  (h_cos : Real.cos (θ / 2) = Real.sqrt ((x - 1) / (2 * x))) : 
  Real.tan θ = -x * Real.sqrt (1 - 1 / x^2) := by
  sorry

end NUMINAMATH_CALUDE_tan_theta_in_terms_of_x_l958_95895


namespace NUMINAMATH_CALUDE_tangent_line_circle_min_sum_l958_95864

theorem tangent_line_circle_min_sum (m n : ℝ) : 
  m > 0 → n > 0 → 
  (∃ x y : ℝ, (m + 1) * x + (n + 1) * y - 2 = 0 ∧ 
               (x - 1)^2 + (y - 1)^2 = 1 ∧
               ∀ a b : ℝ, (x - a)^2 + (y - b)^2 ≤ 1 → 
                          (m + 1) * a + (n + 1) * b - 2 ≠ 0) →
  (∀ p q : ℝ, p > 0 → q > 0 → 
    (∃ x y : ℝ, (p + 1) * x + (q + 1) * y - 2 = 0 ∧ 
                (x - 1)^2 + (y - 1)^2 = 1 ∧
                ∀ a b : ℝ, (x - a)^2 + (y - b)^2 ≤ 1 → 
                           (p + 1) * a + (q + 1) * b - 2 ≠ 0) →
    m + n ≤ p + q) →
  m + n = 2 + 2 * Real.sqrt 2 :=
by sorry

end NUMINAMATH_CALUDE_tangent_line_circle_min_sum_l958_95864


namespace NUMINAMATH_CALUDE_angle_B_is_pi_over_four_max_area_when_b_is_two_max_area_equality_condition_l958_95840

-- Define a triangle
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

-- Define the conditions
def triangleCondition (t : Triangle) : Prop :=
  t.a = t.b * Real.cos t.C + t.c * Real.sin t.B

-- Theorem for part 1
theorem angle_B_is_pi_over_four (t : Triangle) (h : triangleCondition t) :
  t.B = π / 4 := by sorry

-- Theorem for part 2
theorem max_area_when_b_is_two (t : Triangle) (h1 : triangleCondition t) (h2 : t.b = 2) :
  (1 / 2) * t.a * t.c * Real.sin t.B ≤ Real.sqrt 2 + 1 := by sorry

-- Theorem for equality condition in part 2
theorem max_area_equality_condition (t : Triangle) (h1 : triangleCondition t) (h2 : t.b = 2) :
  (1 / 2) * t.a * t.c * Real.sin t.B = Real.sqrt 2 + 1 ↔ t.a = t.c := by sorry

end NUMINAMATH_CALUDE_angle_B_is_pi_over_four_max_area_when_b_is_two_max_area_equality_condition_l958_95840


namespace NUMINAMATH_CALUDE_range_of_x_when_a_is_one_range_of_a_when_q_sufficient_not_necessary_l958_95809

-- Define the propositions p and q
def p (a x : ℝ) : Prop := x^2 - 4*a*x + 3*a^2 < 0

def q (x : ℝ) : Prop := x^2 - x - 6 ≤ 0 ∧ x^2 + 2*x - 8 > 0

-- Theorem for part (1)
theorem range_of_x_when_a_is_one (x : ℝ) (h : p 1 x ∧ q x) : 2 < x ∧ x < 3 := by
  sorry

-- Theorem for part (2)
theorem range_of_a_when_q_sufficient_not_necessary (a : ℝ) 
  (h1 : a > 0)
  (h2 : ∀ x, q x → p a x)
  (h3 : ∃ x, p a x ∧ ¬q x) : 
  1 < a ∧ a ≤ 2 := by
  sorry

end NUMINAMATH_CALUDE_range_of_x_when_a_is_one_range_of_a_when_q_sufficient_not_necessary_l958_95809


namespace NUMINAMATH_CALUDE_area_swept_specific_triangle_l958_95891

/-- Represents a triangle with sides and height -/
structure Triangle where
  bc : ℝ
  ab : ℝ
  ad : ℝ

/-- Calculates the area swept by a triangle moving upward -/
def area_swept (t : Triangle) (speed : ℝ) (time : ℝ) : ℝ :=
  sorry

/-- Theorem stating the area swept by the specific triangle -/
theorem area_swept_specific_triangle :
  let t : Triangle := { bc := 6, ab := 5, ad := 4 }
  area_swept t 3 2 = 66 := by sorry

end NUMINAMATH_CALUDE_area_swept_specific_triangle_l958_95891


namespace NUMINAMATH_CALUDE_quadratic_roots_problem_l958_95846

theorem quadratic_roots_problem (a b c : ℤ) (h_prime : Prime (a + b + c)) :
  let f : ℤ → ℤ := λ x => a * x * x + b * x + c
  (∃ x y : ℕ, x ≠ y ∧ f x = 0 ∧ f y = 0) →  -- roots are distinct positive integers
  (∃ r : ℕ, f r = -55) →                    -- substituting one root gives -55
  (∃ x y : ℕ, x = 2 ∧ y = 7 ∧ f x = 0 ∧ f y = 0) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_roots_problem_l958_95846


namespace NUMINAMATH_CALUDE_sales_tax_rate_is_twenty_percent_l958_95850

/-- Calculates the sales tax rate given the cost of items and total amount spent --/
def calculate_sales_tax_rate (milk_cost banana_cost total_spent : ℚ) : ℚ :=
  let items_cost := milk_cost + banana_cost
  let tax_amount := total_spent - items_cost
  (tax_amount / items_cost) * 100

theorem sales_tax_rate_is_twenty_percent : 
  calculate_sales_tax_rate 3 2 6 = 20 := by sorry

end NUMINAMATH_CALUDE_sales_tax_rate_is_twenty_percent_l958_95850


namespace NUMINAMATH_CALUDE_minimum_buses_needed_l958_95894

def students : ℕ := 535
def bus_capacity : ℕ := 45

theorem minimum_buses_needed : 
  ∃ (n : ℕ), n * bus_capacity ≥ students ∧ 
  ∀ (m : ℕ), m * bus_capacity ≥ students → n ≤ m :=
by sorry

end NUMINAMATH_CALUDE_minimum_buses_needed_l958_95894


namespace NUMINAMATH_CALUDE_total_notes_count_l958_95871

/-- Given a total amount of 192 rupees in equal numbers of 1-rupee, 5-rupee, and 10-rupee notes,
    prove that the total number of notes is 36. -/
theorem total_notes_count (total_amount : ℕ) (note_count : ℕ) : 
  total_amount = 192 →
  note_count * 1 + note_count * 5 + note_count * 10 = total_amount →
  3 * note_count = 36 :=
by
  sorry

end NUMINAMATH_CALUDE_total_notes_count_l958_95871


namespace NUMINAMATH_CALUDE_frank_final_balance_l958_95802

def frank_money_problem (initial_amount : ℤ) 
                        (game_cost : ℤ) 
                        (keychain_cost : ℤ) 
                        (friend_gift : ℤ) 
                        (allowance : ℤ) 
                        (bus_ticket_cost : ℤ) : Prop :=
  initial_amount = 11 ∧
  game_cost = 3 ∧
  keychain_cost = 2 ∧
  friend_gift = 4 ∧
  allowance = 14 ∧
  bus_ticket_cost = 5 ∧
  initial_amount - game_cost - keychain_cost + friend_gift + allowance - bus_ticket_cost = 19

theorem frank_final_balance :
  ∀ (initial_amount game_cost keychain_cost friend_gift allowance bus_ticket_cost : ℤ),
  frank_money_problem initial_amount game_cost keychain_cost friend_gift allowance bus_ticket_cost :=
by
  sorry

end NUMINAMATH_CALUDE_frank_final_balance_l958_95802


namespace NUMINAMATH_CALUDE_largest_number_with_digit_sum_20_l958_95843

def is_valid_number (n : ℕ) : Prop :=
  ∀ d, d ∈ n.digits 10 → d = 3 ∨ d = 4

def digit_sum (n : ℕ) : ℕ :=
  (n.digits 10).sum

theorem largest_number_with_digit_sum_20 :
  ∀ n : ℕ, is_valid_number n → digit_sum n = 20 → n ≤ 443333 :=
sorry

end NUMINAMATH_CALUDE_largest_number_with_digit_sum_20_l958_95843


namespace NUMINAMATH_CALUDE_annes_cats_weight_l958_95890

/-- The total weight of Anne's four cats -/
def total_weight (first_female_weight : ℝ) : ℝ :=
  let second_female_weight := 1.5 * first_female_weight
  let first_male_weight := 2 * first_female_weight
  let second_male_weight := first_female_weight + second_female_weight
  first_female_weight + second_female_weight + first_male_weight + second_male_weight

/-- Theorem stating that the total weight of Anne's four cats is 14 kilograms -/
theorem annes_cats_weight : total_weight 2 = 14 := by
  sorry

end NUMINAMATH_CALUDE_annes_cats_weight_l958_95890


namespace NUMINAMATH_CALUDE_pauls_books_l958_95816

theorem pauls_books (books_sold : ℕ) (books_left : ℕ) : 
  books_sold = 137 → books_left = 105 → books_sold + books_left = 242 :=
by sorry

end NUMINAMATH_CALUDE_pauls_books_l958_95816


namespace NUMINAMATH_CALUDE_crayon_cost_proof_l958_95822

/-- The cost of one pack of crayons -/
def pack_cost : ℚ := 5/2

/-- The number of packs Michael initially has -/
def initial_packs : ℕ := 4

/-- The number of packs Michael buys -/
def bought_packs : ℕ := 2

/-- The total value of all packs after purchase -/
def total_value : ℚ := 15

theorem crayon_cost_proof :
  (initial_packs + bought_packs : ℚ) * pack_cost = total_value := by
  sorry

end NUMINAMATH_CALUDE_crayon_cost_proof_l958_95822


namespace NUMINAMATH_CALUDE_oldest_bride_age_l958_95887

theorem oldest_bride_age (bride_age groom_age : ℕ) : 
  bride_age = groom_age + 19 →
  bride_age + groom_age = 185 →
  bride_age = 102 := by
sorry

end NUMINAMATH_CALUDE_oldest_bride_age_l958_95887


namespace NUMINAMATH_CALUDE_euler_family_mean_age_l958_95855

def euler_family_ages : List ℕ := [5, 8, 8, 8, 12, 12]

theorem euler_family_mean_age :
  (euler_family_ages.sum : ℚ) / euler_family_ages.length = 53 / 6 := by
  sorry

end NUMINAMATH_CALUDE_euler_family_mean_age_l958_95855


namespace NUMINAMATH_CALUDE_equation_solution_l958_95824

theorem equation_solution : 
  let eq := fun x : ℝ => 81 * (1 - x)^2 - 64
  ∃ (x1 x2 : ℝ), x1 = 1/9 ∧ x2 = 17/9 ∧ eq x1 = 0 ∧ eq x2 = 0 ∧
  ∀ (x : ℝ), eq x = 0 → x = x1 ∨ x = x2 :=
by
  sorry

#check equation_solution

end NUMINAMATH_CALUDE_equation_solution_l958_95824


namespace NUMINAMATH_CALUDE_a1_plus_a3_equals_24_l958_95874

theorem a1_plus_a3_equals_24 (x : ℝ) (a₀ a₁ a₂ a₃ a₄ : ℝ) 
  (h : (1 - 2/x)^4 = a₀ + a₁*(1/x) + a₂*(1/x)^2 + a₃*(1/x)^3 + a₄*(1/x)^4) :
  a₁ + a₃ = 24 := by
sorry

end NUMINAMATH_CALUDE_a1_plus_a3_equals_24_l958_95874


namespace NUMINAMATH_CALUDE_solve_for_A_l958_95828

theorem solve_for_A : ∃ A : ℤ, (2 * A - 6 + 4 = 26) ∧ A = 14 := by
  sorry

end NUMINAMATH_CALUDE_solve_for_A_l958_95828


namespace NUMINAMATH_CALUDE_quadratic_roots_pure_imaginary_l958_95836

theorem quadratic_roots_pure_imaginary (m : ℂ) (h : m.re = 0 ∧ m.im ≠ 0) :
  ∃ (z₁ z₂ : ℂ), z₁.re = 0 ∧ z₂.re = 0 ∧ 
  8 * z₁^2 + 4 * Complex.I * z₁ - m = 0 ∧
  8 * z₂^2 + 4 * Complex.I * z₂ - m = 0 :=
sorry

end NUMINAMATH_CALUDE_quadratic_roots_pure_imaginary_l958_95836


namespace NUMINAMATH_CALUDE_tesseract_parallel_edge_pairs_l958_95861

/-- A tesseract is a 4-dimensional hypercube -/
structure Tesseract where
  dim : Nat
  dim_eq : dim = 4

/-- The number of pairs of parallel edges in a tesseract -/
def parallel_edge_pairs (t : Tesseract) : Nat := 36

/-- Theorem: A tesseract has 36 pairs of parallel edges -/
theorem tesseract_parallel_edge_pairs (t : Tesseract) : 
  parallel_edge_pairs t = 36 := by sorry

end NUMINAMATH_CALUDE_tesseract_parallel_edge_pairs_l958_95861


namespace NUMINAMATH_CALUDE_no_integer_both_roots_finite_decimal_l958_95803

theorem no_integer_both_roots_finite_decimal (n : ℤ) (hn : n ≠ 0) :
  ¬(∃ (x₁ x₂ : ℚ), 
    (x₁ ≠ x₂) ∧
    ((4 * n^2 - 1) * x₁^2 - 4 * n^2 * x₁ + n^2 = 0) ∧
    ((4 * n^2 - 1) * x₂^2 - 4 * n^2 * x₂ + n^2 = 0) ∧
    (∃ (a b c d : ℤ), x₁ = (a : ℚ) / (2^b * 5^c) ∧ x₂ = (d : ℚ) / (2^b * 5^c))) :=
sorry

end NUMINAMATH_CALUDE_no_integer_both_roots_finite_decimal_l958_95803


namespace NUMINAMATH_CALUDE_odd_even_array_parity_l958_95857

/-- Represents an n × n array where each entry is either 1 or -1 -/
def OddEvenArray (n : ℕ) := Fin n → Fin n → Int

/-- Counts the number of rows with an odd number of -1s -/
def oddRowCount (A : OddEvenArray n) : ℕ := sorry

/-- Counts the number of columns with an odd number of -1s -/
def oddColumnCount (A : OddEvenArray n) : ℕ := sorry

/-- The main theorem -/
theorem odd_even_array_parity (n : ℕ) (hn : Odd n) (A : OddEvenArray n) :
  Even (oddRowCount A + oddColumnCount A) := by sorry

end NUMINAMATH_CALUDE_odd_even_array_parity_l958_95857


namespace NUMINAMATH_CALUDE_binary_1100_equals_12_l958_95812

def binary_to_decimal (b : List Bool) : Nat :=
  b.enum.foldl (fun acc (i, bit) => acc + if bit then 2^i else 0) 0

theorem binary_1100_equals_12 :
  binary_to_decimal [false, false, true, true] = 12 := by
  sorry

end NUMINAMATH_CALUDE_binary_1100_equals_12_l958_95812


namespace NUMINAMATH_CALUDE_square_sum_difference_l958_95807

theorem square_sum_difference : 102 * 102 + 98 * 98 = 800 := by
  sorry

end NUMINAMATH_CALUDE_square_sum_difference_l958_95807


namespace NUMINAMATH_CALUDE_min_abs_z_on_line_segment_l958_95865

theorem min_abs_z_on_line_segment (z : ℂ) (h : Complex.abs (z - 6 * Complex.I) + Complex.abs (z - 5) = 7) :
  ∃ (w : ℂ), Complex.abs w ≤ Complex.abs z ∧ Complex.abs w = 30 / Real.sqrt 61 := by
  sorry

end NUMINAMATH_CALUDE_min_abs_z_on_line_segment_l958_95865


namespace NUMINAMATH_CALUDE_cube_minus_cylinder_volume_l958_95879

/-- The remaining volume of a cube after removing a cylindrical section -/
theorem cube_minus_cylinder_volume (cube_side : ℝ) (cylinder_radius : ℝ) :
  cube_side = 6 →
  cylinder_radius = 3 →
  cube_side ^ 3 - π * cylinder_radius ^ 2 * cube_side = 216 - 54 * π := by
  sorry

end NUMINAMATH_CALUDE_cube_minus_cylinder_volume_l958_95879


namespace NUMINAMATH_CALUDE_group_size_proof_l958_95826

theorem group_size_proof (total_rupees : ℚ) (h1 : total_rupees = 72.25) : ∃ n : ℕ, 
  (n : ℚ) * (n : ℚ) = total_rupees * 100 ∧ n = 85 := by
  sorry

end NUMINAMATH_CALUDE_group_size_proof_l958_95826


namespace NUMINAMATH_CALUDE_pencil_arrangement_theorem_l958_95834

def total_pencils (total_rows : ℕ) (pattern_length : ℕ) (pencils_second_row : ℕ) : ℕ :=
  let pattern_repeats := total_rows / pattern_length
  let pencils_fifth_row := pencils_second_row + pencils_second_row / 2
  let pencil_rows_per_pattern := 2
  pattern_repeats * pencil_rows_per_pattern * (pencils_second_row + pencils_fifth_row)

theorem pencil_arrangement_theorem :
  total_pencils 30 6 76 = 950 := by
  sorry

end NUMINAMATH_CALUDE_pencil_arrangement_theorem_l958_95834


namespace NUMINAMATH_CALUDE_rectangle_ratio_l958_95882

theorem rectangle_ratio (s : ℝ) (x y : ℝ) (h1 : x > 0) (h2 : y > 0) (h3 : s > 0)
  (h4 : s + 2*x = 3*s) -- The outer boundary is 3 times the inner square side
  (h5 : 2*y = s) -- The shorter side spans half the inner square side
  : x / y = 2 := by
sorry

end NUMINAMATH_CALUDE_rectangle_ratio_l958_95882


namespace NUMINAMATH_CALUDE_photo_arrangements_l958_95845

/-- Represents the number of students in the photo --/
def num_students : ℕ := 5

/-- Represents the constraint that B and C must stand together --/
def bc_together : Prop := True

/-- Represents the constraint that A cannot stand next to B --/
def a_not_next_to_b : Prop := True

/-- The number of different arrangements --/
def num_arrangements : ℕ := 36

/-- Theorem stating that the number of arrangements is 36 --/
theorem photo_arrangements :
  (num_students = 5) →
  bc_together →
  a_not_next_to_b →
  num_arrangements = 36 := by
  sorry

end NUMINAMATH_CALUDE_photo_arrangements_l958_95845


namespace NUMINAMATH_CALUDE_product_from_lcm_and_gcd_l958_95862

theorem product_from_lcm_and_gcd (a b : ℕ+) 
  (h1 : Nat.lcm a b = 48) 
  (h2 : Nat.gcd a b = 8) : 
  a * b = 384 := by
sorry

end NUMINAMATH_CALUDE_product_from_lcm_and_gcd_l958_95862


namespace NUMINAMATH_CALUDE_inverse_of_proposition_l958_95820

-- Define the original proposition
def original_proposition (x : ℝ) : Prop :=
  x < 0 → x^2 > 0

-- Define the inverse proposition
def inverse_proposition (x : ℝ) : Prop :=
  x^2 > 0 → x < 0

-- Theorem stating that the inverse_proposition is indeed the inverse of the original_proposition
theorem inverse_of_proposition :
  (∀ x : ℝ, original_proposition x) ↔ (∀ x : ℝ, inverse_proposition x) :=
sorry

end NUMINAMATH_CALUDE_inverse_of_proposition_l958_95820


namespace NUMINAMATH_CALUDE_magnitude_of_sum_l958_95806

def a (m : ℝ) : ℝ × ℝ := (4, m)
def b : ℝ × ℝ := (1, -2)

theorem magnitude_of_sum (m : ℝ) 
  (h : (a m).1 * b.1 + (a m).2 * b.2 = 0) : 
  Real.sqrt (((a m).1 + 2 * b.1)^2 + ((a m).2 + 2 * b.2)^2) = 2 * Real.sqrt 10 := by
  sorry

end NUMINAMATH_CALUDE_magnitude_of_sum_l958_95806


namespace NUMINAMATH_CALUDE_factorial_ratio_equals_sixty_sevenths_l958_95873

theorem factorial_ratio_equals_sixty_sevenths : (Nat.factorial 10 * Nat.factorial 6 * Nat.factorial 3) / (Nat.factorial 9 * Nat.factorial 7) = 60 / 7 := by
  sorry

end NUMINAMATH_CALUDE_factorial_ratio_equals_sixty_sevenths_l958_95873


namespace NUMINAMATH_CALUDE_shaded_area_calculation_l958_95885

/-- Calculates the area of the shaded region in a grid with two unshaded triangles. -/
theorem shaded_area_calculation (grid_width grid_height : ℝ)
  (large_triangle_base large_triangle_height : ℝ)
  (small_triangle_base small_triangle_height : ℝ)
  (h1 : grid_width = 15)
  (h2 : grid_height = 5)
  (h3 : large_triangle_base = grid_width)
  (h4 : large_triangle_height = grid_height)
  (h5 : small_triangle_base = 3)
  (h6 : small_triangle_height = 2) :
  grid_width * grid_height - (1/2 * large_triangle_base * large_triangle_height) -
  (1/2 * small_triangle_base * small_triangle_height) = 34.5 := by
  sorry

end NUMINAMATH_CALUDE_shaded_area_calculation_l958_95885


namespace NUMINAMATH_CALUDE_ab_equation_sum_l958_95823

theorem ab_equation_sum (A B : ℕ) : 
  A ≠ B → 
  A < 10 → 
  B < 10 → 
  (10 * A + B) * 6 = 100 * B + 10 * B + B → 
  A + B = 11 :=
by sorry

end NUMINAMATH_CALUDE_ab_equation_sum_l958_95823


namespace NUMINAMATH_CALUDE_dividend_calculation_l958_95898

theorem dividend_calculation (divisor quotient remainder : ℕ) 
  (h1 : divisor = 17) 
  (h2 : quotient = 9) 
  (h3 : remainder = 9) : 
  divisor * quotient + remainder = 162 := by
  sorry

end NUMINAMATH_CALUDE_dividend_calculation_l958_95898


namespace NUMINAMATH_CALUDE_squad_size_problem_l958_95899

theorem squad_size_problem (total : ℕ) (transfer : ℕ) 
  (h1 : total = 146) (h2 : transfer = 11) : 
  (∃ (first second : ℕ), 
    first + second = total ∧ 
    first - transfer = second + transfer ∧
    first = 84 ∧ 
    second = 62) := by
  sorry

end NUMINAMATH_CALUDE_squad_size_problem_l958_95899


namespace NUMINAMATH_CALUDE_triangle_construction_from_polygon_centers_l958_95854

/-- Centers of regular n-sided polygons externally inscribed on triangle sides -/
structure PolygonCenters (n : ℕ) where
  X : ℝ × ℝ
  Y : ℝ × ℝ
  Z : ℝ × ℝ

/-- Triangle vertices -/
structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ

/-- Rotation by angle α around point P -/
def rotate (P : ℝ × ℝ) (α : ℝ) (Q : ℝ × ℝ) : ℝ × ℝ := sorry

/-- Check if three points form a regular triangle -/
def isRegularTriangle (P Q R : ℝ × ℝ) : Prop := sorry

/-- Theorem about triangle construction from polygon centers -/
theorem triangle_construction_from_polygon_centers (n : ℕ) (centers : PolygonCenters n) :
  (n ≥ 4 → ∃! t : Triangle, 
    rotate centers.Y (2 * π / n) t.A = t.C ∧
    rotate centers.X (2 * π / n) t.C = t.B ∧
    rotate centers.Z (2 * π / n) t.B = t.A) ∧
  (n = 3 → isRegularTriangle centers.X centers.Y centers.Z → 
    ∃ t : Set Triangle, Infinite t ∧ 
    ∀ tri ∈ t, rotate centers.Y (2 * π / 3) tri.A = tri.C ∧
               rotate centers.X (2 * π / 3) tri.C = tri.B ∧
               rotate centers.Z (2 * π / 3) tri.B = tri.A) :=
by sorry

end NUMINAMATH_CALUDE_triangle_construction_from_polygon_centers_l958_95854


namespace NUMINAMATH_CALUDE_marias_coffee_order_l958_95889

/-- Maria's daily coffee order calculation -/
theorem marias_coffee_order (visits_per_day : ℕ) (cups_per_visit : ℕ)
  (h1 : visits_per_day = 2)
  (h2 : cups_per_visit = 3) :
  visits_per_day * cups_per_visit = 6 := by
  sorry

end NUMINAMATH_CALUDE_marias_coffee_order_l958_95889


namespace NUMINAMATH_CALUDE_pencil_cost_l958_95815

theorem pencil_cost (total_students : Nat) (total_cost : Nat) 
  (h1 : total_students = 36)
  (h2 : total_cost = 1881)
  (s : Nat) (c : Nat) (n : Nat)
  (h3 : s > total_students / 2)
  (h4 : c > n)
  (h5 : n > 1)
  (h6 : s * c * n = total_cost) :
  c = 17 := by
  sorry

end NUMINAMATH_CALUDE_pencil_cost_l958_95815


namespace NUMINAMATH_CALUDE_sin_alpha_minus_pi_fourth_l958_95833

theorem sin_alpha_minus_pi_fourth (α : Real) : 
  α ∈ Set.Icc (π) (3*π/2) →   -- α is in the third quadrant
  Real.tan (α + π/4) = -2 →   -- tan(α + π/4) = -2
  Real.sin (α - π/4) = -Real.sqrt 5 / 5 := by
sorry

end NUMINAMATH_CALUDE_sin_alpha_minus_pi_fourth_l958_95833


namespace NUMINAMATH_CALUDE_video_votes_total_l958_95813

/-- Represents the voting system for a video -/
structure VideoVotes where
  totalVotes : ℕ
  likePercentage : ℚ
  finalScore : ℤ

/-- Theorem: Given the conditions, the total number of votes is 240 -/
theorem video_votes_total (v : VideoVotes) 
  (h1 : v.likePercentage = 3/4)
  (h2 : v.finalScore = 120) :
  v.totalVotes = 240 := by
  sorry


end NUMINAMATH_CALUDE_video_votes_total_l958_95813


namespace NUMINAMATH_CALUDE_janinas_pancakes_l958_95841

-- Define the variables
def daily_rent : ℕ := 30
def daily_supplies : ℕ := 12
def price_per_pancake : ℕ := 2

-- Define the function to calculate the number of pancakes needed
def pancakes_needed (rent : ℕ) (supplies : ℕ) (price : ℕ) : ℕ :=
  (rent + supplies) / price

-- Theorem statement
theorem janinas_pancakes :
  pancakes_needed daily_rent daily_supplies price_per_pancake = 21 := by
sorry

end NUMINAMATH_CALUDE_janinas_pancakes_l958_95841


namespace NUMINAMATH_CALUDE_circle_radius_is_one_l958_95847

/-- The equation of a circle is x^2 + y^2 + 2x + 2y + 1 = 0. This theorem proves that the radius of this circle is 1. -/
theorem circle_radius_is_one :
  ∃ (h : ℝ → ℝ → Prop),
    (∀ x y : ℝ, h x y ↔ x^2 + y^2 + 2*x + 2*y + 1 = 0) →
    (∃ c : ℝ × ℝ, ∃ r : ℝ, r = 1 ∧ ∀ x y : ℝ, h x y ↔ (x - c.1)^2 + (y - c.2)^2 = r^2) :=
by sorry

end NUMINAMATH_CALUDE_circle_radius_is_one_l958_95847


namespace NUMINAMATH_CALUDE_unique_number_l958_95827

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d > 1 → d < n → ¬(n % d = 0)

def satisfies_conditions (n : ℕ) : Prop :=
  let h := n / 100
  let t := (n / 10) % 10
  let u := n % 10
  100 ≤ n ∧ n < 1000 ∧  -- three-digit number
  h = 2 * t ∧           -- hundreds digit is twice the tens digit
  u = 2 * t^3 ∧         -- units digit is double the cube of tens digit
  is_prime (h + t + u)  -- sum of digits is prime

theorem unique_number : ∀ n : ℕ, satisfies_conditions n ↔ n = 212 :=
sorry

end NUMINAMATH_CALUDE_unique_number_l958_95827


namespace NUMINAMATH_CALUDE_max_value_of_2x_plus_y_l958_95886

theorem max_value_of_2x_plus_y (x y : ℝ) (h : 3 * x^2 + 2 * y^2 ≤ 6) :
  ∃ (M : ℝ), M = Real.sqrt 11 ∧ 2 * x + y ≤ M ∧ ∃ (x₀ y₀ : ℝ), 3 * x₀^2 + 2 * y₀^2 ≤ 6 ∧ 2 * x₀ + y₀ = M :=
by sorry

end NUMINAMATH_CALUDE_max_value_of_2x_plus_y_l958_95886


namespace NUMINAMATH_CALUDE_acute_inclination_implies_ab_negative_l958_95893

-- Define a line with coefficients a, b, and c
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

-- Define the property of having an acute angle of inclination
def hasAcuteInclination (l : Line) : Prop :=
  0 < -l.a / l.b ∧ -l.a / l.b < 1

-- Theorem statement
theorem acute_inclination_implies_ab_negative (l : Line) :
  hasAcuteInclination l → l.a * l.b < 0 := by
  sorry

end NUMINAMATH_CALUDE_acute_inclination_implies_ab_negative_l958_95893


namespace NUMINAMATH_CALUDE_quadratic_transformation_l958_95858

theorem quadratic_transformation (p q r : ℤ) : 
  (∀ x, (p * x + q)^2 + r = 4 * x^2 - 16 * x + 15) → 
  p * q = -8 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_transformation_l958_95858


namespace NUMINAMATH_CALUDE_marias_candy_l958_95811

/-- The number of candy pieces Maria ate -/
def pieces_eaten : ℕ := 64

/-- The number of candy pieces Maria has left -/
def pieces_left : ℕ := 3

/-- The initial number of candy pieces Maria had -/
def initial_pieces : ℕ := pieces_eaten + pieces_left

theorem marias_candy : initial_pieces = 67 := by
  sorry

end NUMINAMATH_CALUDE_marias_candy_l958_95811


namespace NUMINAMATH_CALUDE_factor_calculation_l958_95853

theorem factor_calculation : ∃ f : ℚ, (2 * 7 + 9) * f = 69 ∧ f = 3 := by
  sorry

end NUMINAMATH_CALUDE_factor_calculation_l958_95853


namespace NUMINAMATH_CALUDE_parabola_vertex_x_coordinate_l958_95842

/-- The x-coordinate of the vertex of a parabola given three points it passes through -/
theorem parabola_vertex_x_coordinate 
  (a b c : ℝ) 
  (h1 : a * (-2)^2 + b * (-2) + c = 8)
  (h2 : a * 4^2 + b * 4 + c = 8)
  (h3 : a * 7^2 + b * 7 + c = 15) :
  let f := fun x => a * x^2 + b * x + c
  ∃ x₀, ∀ x, f x ≥ f x₀ ∧ x₀ = 1 :=
by sorry

end NUMINAMATH_CALUDE_parabola_vertex_x_coordinate_l958_95842


namespace NUMINAMATH_CALUDE_complex_equation_implies_sum_l958_95830

theorem complex_equation_implies_sum (x y : ℝ) :
  (x + y : ℂ) + (y - 1) * I = (2 * x + 3 * y : ℂ) + (2 * y + 1) * I →
  x + y = 2 := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_implies_sum_l958_95830


namespace NUMINAMATH_CALUDE_recycling_points_l958_95821

/-- Calculates the points earned from recycling paper --/
def points_earned (pounds_per_point : ℕ) (chloe_pounds : ℕ) (friends_pounds : ℕ) : ℕ :=
  (chloe_pounds + friends_pounds) / pounds_per_point

/-- Theorem: Given the recycling conditions, the total points earned is 5 --/
theorem recycling_points : points_earned 6 28 2 = 5 := by
  sorry

end NUMINAMATH_CALUDE_recycling_points_l958_95821


namespace NUMINAMATH_CALUDE_johns_allowance_l958_95825

/-- John's weekly allowance problem -/
theorem johns_allowance (A : ℝ) : A = 2.40 :=
  let arcade_spent := (3 : ℝ) / 5 * A
  let remaining_after_arcade := A - arcade_spent
  let toy_store_spent := (1 : ℝ) / 3 * remaining_after_arcade
  let remaining_after_toy_store := remaining_after_arcade - toy_store_spent
  by
    have h1 : remaining_after_toy_store = 0.64
    sorry
    -- Proof goes here
    sorry

#check johns_allowance

end NUMINAMATH_CALUDE_johns_allowance_l958_95825


namespace NUMINAMATH_CALUDE_bus_assignment_count_l958_95875

def num_boys : ℕ := 6
def num_girls : ℕ := 4
def num_buses : ℕ := 5
def attendants_per_bus : ℕ := 2

theorem bus_assignment_count : 
  (Nat.choose num_buses 3) * 
  (Nat.factorial num_boys / (Nat.factorial attendants_per_bus ^ 3)) * 
  (Nat.factorial num_girls / (Nat.factorial attendants_per_bus ^ 2)) * 
  (1 / Nat.factorial 3) * 
  (1 / Nat.factorial 2) * 
  Nat.factorial num_buses = 54000 := by
sorry

end NUMINAMATH_CALUDE_bus_assignment_count_l958_95875


namespace NUMINAMATH_CALUDE_edric_hourly_rate_l958_95817

/-- Edric's salary calculation --/
def salary_calculation (B C S P D : ℚ) (H W : ℕ) : ℚ :=
  let E := B + (C * S) + P - D
  let T := (H * W * 4 : ℚ)
  E / T

/-- Edric's hourly rate is approximately $3.86 --/
theorem edric_hourly_rate :
  let B := 576
  let C := 3 / 100
  let S := 4000
  let P := 75
  let D := 30
  let H := 8
  let W := 6
  abs (salary_calculation B C S P D H W - 386 / 100) < 1 / 100 := by
  sorry

end NUMINAMATH_CALUDE_edric_hourly_rate_l958_95817


namespace NUMINAMATH_CALUDE_greatest_sum_consecutive_integers_product_less_500_l958_95849

/-- The greatest possible sum of two consecutive integers whose product is less than 500 is 43 -/
theorem greatest_sum_consecutive_integers_product_less_500 : 
  (∃ n : ℤ, n * (n + 1) < 500 ∧ 
    ∀ m : ℤ, m * (m + 1) < 500 → m + (m + 1) ≤ n + (n + 1)) ∧
  (∀ n : ℤ, n * (n + 1) < 500 → n + (n + 1) ≤ 43) :=
by sorry

end NUMINAMATH_CALUDE_greatest_sum_consecutive_integers_product_less_500_l958_95849


namespace NUMINAMATH_CALUDE_greatest_distance_between_circle_centers_l958_95877

/-- The greatest possible distance between the centers of two circles in a rectangle -/
theorem greatest_distance_between_circle_centers 
  (rectangle_width : ℝ) 
  (rectangle_height : ℝ) 
  (circle_diameter : ℝ) 
  (h1 : rectangle_width = 20) 
  (h2 : rectangle_height = 15) 
  (h3 : circle_diameter = 10) :
  ∃ (d : ℝ), d = 5 * Real.sqrt 5 ∧ 
  ∀ (d' : ℝ), d' ≤ d ∧ 
  ∃ (x1 y1 x2 y2 : ℝ), 
    0 ≤ x1 ∧ x1 ≤ rectangle_width ∧
    0 ≤ y1 ∧ y1 ≤ rectangle_height ∧
    0 ≤ x2 ∧ x2 ≤ rectangle_width ∧
    0 ≤ y2 ∧ y2 ≤ rectangle_height ∧
    circle_diameter / 2 ≤ x1 ∧ x1 ≤ rectangle_width - circle_diameter / 2 ∧
    circle_diameter / 2 ≤ y1 ∧ y1 ≤ rectangle_height - circle_diameter / 2 ∧
    circle_diameter / 2 ≤ x2 ∧ x2 ≤ rectangle_width - circle_diameter / 2 ∧
    circle_diameter / 2 ≤ y2 ∧ y2 ≤ rectangle_height - circle_diameter / 2 ∧
    d' = Real.sqrt ((x2 - x1)^2 + (y2 - y1)^2) :=
by sorry

end NUMINAMATH_CALUDE_greatest_distance_between_circle_centers_l958_95877


namespace NUMINAMATH_CALUDE_toys_sold_is_eighteen_l958_95884

/-- Given a selling price, gain equal to the cost of 3 toys, and the cost of one toy,
    calculate the number of toys sold. -/
def number_of_toys_sold (selling_price gain cost_per_toy : ℕ) : ℕ :=
  (selling_price - gain) / cost_per_toy

/-- Theorem stating that given the conditions in the problem, 
    the number of toys sold is 18. -/
theorem toys_sold_is_eighteen :
  let selling_price := 21000
  let gain := 3 * 1000
  let cost_per_toy := 1000
  number_of_toys_sold selling_price gain cost_per_toy = 18 := by
  sorry

#eval number_of_toys_sold 21000 (3 * 1000) 1000

end NUMINAMATH_CALUDE_toys_sold_is_eighteen_l958_95884


namespace NUMINAMATH_CALUDE_simplify_fraction_l958_95832

theorem simplify_fraction : 18 * (8 / 12) * (1 / 6) = 2 := by
  sorry

end NUMINAMATH_CALUDE_simplify_fraction_l958_95832


namespace NUMINAMATH_CALUDE_fraction_meaningful_l958_95883

theorem fraction_meaningful (x : ℝ) : 
  (∃ y : ℝ, y = 1 / (x + 1)) ↔ x ≠ -1 :=
sorry

end NUMINAMATH_CALUDE_fraction_meaningful_l958_95883


namespace NUMINAMATH_CALUDE_right_triangle_hypotenuse_l958_95876

theorem right_triangle_hypotenuse (a b c : ℝ) : 
  a > 0 → b > 0 → c > 0 →  -- positive sides
  a + b + c = 40 →  -- perimeter condition
  (1/2) * a * b = 30 →  -- area condition
  a^2 + b^2 = c^2 →  -- right triangle (Pythagorean theorem)
  c = 18.5 := by
sorry

end NUMINAMATH_CALUDE_right_triangle_hypotenuse_l958_95876


namespace NUMINAMATH_CALUDE_polynomial_division_theorem_l958_95829

/-- The polynomial P(x) -/
def P (x : ℝ) : ℝ := x^6 - 6*x^4 - 4*x^3 + 9*x^2 + 12*x + 4

/-- The derivative of P(x) -/
def P' (x : ℝ) : ℝ := 6*x^5 - 24*x^3 - 12*x^2 + 18*x + 12

/-- The greatest common divisor of P(x) and P'(x) -/
noncomputable def Q (x : ℝ) : ℝ := x^4 + x^3 - 3*x^2 - 5*x - 2

/-- The resulting polynomial R(x) -/
def R (x : ℝ) : ℝ := x^2 - x - 2

theorem polynomial_division_theorem :
  ∀ x : ℝ, P x = Q x * R x :=
by sorry

end NUMINAMATH_CALUDE_polynomial_division_theorem_l958_95829


namespace NUMINAMATH_CALUDE_max_value_expression_l958_95851

theorem max_value_expression (a b : ℝ) (ha : 0 ≤ a ∧ a ≤ 1) (hb : 0 ≤ b ∧ b ≤ 1) :
  (a - b^2) * (b - a^2) ≤ 1/16 := by
  sorry

end NUMINAMATH_CALUDE_max_value_expression_l958_95851


namespace NUMINAMATH_CALUDE_expansion_term_count_l958_95897

/-- The number of terms in a polynomial -/
def num_terms (p : Polynomial ℚ) : ℕ := sorry

/-- The expansion of the product of two polynomials -/
def expand_product (p q : Polynomial ℚ) : Polynomial ℚ := sorry

theorem expansion_term_count :
  let p := X + Y + Z
  let q := U + V + W + X
  num_terms (expand_product p q) = 12 := by sorry

end NUMINAMATH_CALUDE_expansion_term_count_l958_95897


namespace NUMINAMATH_CALUDE_freshmen_in_liberal_arts_l958_95866

theorem freshmen_in_liberal_arts 
  (total_students : ℝ) 
  (freshmen_ratio : ℝ) 
  (psych_majors_ratio : ℝ) 
  (freshmen_psych_lib_arts_ratio : ℝ) 
  (h1 : freshmen_ratio = 0.4)
  (h2 : psych_majors_ratio = 0.5)
  (h3 : freshmen_psych_lib_arts_ratio = 0.1) :
  (freshmen_psych_lib_arts_ratio * total_students) / (psych_majors_ratio * (freshmen_ratio * total_students)) = 0.5 := by
  sorry

end NUMINAMATH_CALUDE_freshmen_in_liberal_arts_l958_95866


namespace NUMINAMATH_CALUDE_ordering_abc_l958_95801

theorem ordering_abc : 
  let a : ℝ := Real.exp (Real.sqrt 2)
  let b : ℝ := 2 + Real.sqrt 2
  let c : ℝ := Real.log (12 + 6 * Real.sqrt 2)
  a > b ∧ b > c := by
  sorry

end NUMINAMATH_CALUDE_ordering_abc_l958_95801


namespace NUMINAMATH_CALUDE_average_words_per_puzzle_l958_95856

/-- Represents the number of days in a week -/
def days_per_week : ℕ := 7

/-- Represents the number of weeks a pencil lasts -/
def weeks_per_pencil : ℕ := 2

/-- Represents the total number of words to use up a pencil -/
def words_per_pencil : ℕ := 1050

/-- Represents Bert's daily crossword puzzle habit -/
def puzzles_per_day : ℕ := 1

/-- Theorem stating the average number of words in each crossword puzzle -/
theorem average_words_per_puzzle :
  (words_per_pencil / (weeks_per_pencil * days_per_week)) = 75 := by
  sorry

end NUMINAMATH_CALUDE_average_words_per_puzzle_l958_95856


namespace NUMINAMATH_CALUDE_no_x_term_iff_k_eq_two_l958_95831

/-- The polynomial x^2 + (k-2)x - 3 does not contain the term with x if and only if k = 2 -/
theorem no_x_term_iff_k_eq_two (k : ℝ) : 
  (∀ x : ℝ, x^2 + (k-2)*x - 3 = x^2 - 3) ↔ k = 2 := by
sorry

end NUMINAMATH_CALUDE_no_x_term_iff_k_eq_two_l958_95831


namespace NUMINAMATH_CALUDE_factors_of_M_l958_95804

/-- The number of natural-number factors of M, where M = 2^3 · 3^5 · 5^3 · 7^1 · 11^2 -/
def number_of_factors (M : ℕ) : ℕ :=
  if M = 2^3 * 3^5 * 5^3 * 7^1 * 11^2 then 576 else 0

/-- Theorem stating that the number of natural-number factors of M is 576 -/
theorem factors_of_M :
  number_of_factors (2^3 * 3^5 * 5^3 * 7^1 * 11^2) = 576 :=
by sorry

end NUMINAMATH_CALUDE_factors_of_M_l958_95804


namespace NUMINAMATH_CALUDE_function_symmetric_about_origin_l958_95852

/-- The function f(x) = x^3 - x is symmetric about the origin. -/
theorem function_symmetric_about_origin (x : ℝ) : let f := λ x : ℝ => x^3 - x
  f (-x) = -f x := by
  sorry

end NUMINAMATH_CALUDE_function_symmetric_about_origin_l958_95852


namespace NUMINAMATH_CALUDE_shaded_cubes_count_shaded_cubes_count_proof_l958_95838

/-- Represents a 3x3x3 cube with a specific shading pattern -/
structure ShadedCube where
  /-- The number of smaller cubes in each dimension of the large cube -/
  size : Nat
  /-- The number of shaded squares on each face -/
  shaded_per_face : Nat
  /-- The total number of smaller cubes in the large cube -/
  total_cubes : Nat
  /-- Assertion that the cube is 3x3x3 -/
  size_is_three : size = 3
  /-- Assertion that the total number of cubes is correct -/
  total_is_correct : total_cubes = size ^ 3
  /-- Assertion that each face has 5 shaded squares -/
  five_shaded_per_face : shaded_per_face = 5

/-- Theorem stating that the number of shaded cubes is 20 -/
theorem shaded_cubes_count (c : ShadedCube) : Nat :=
  20

#check shaded_cubes_count

/-- Proof of the theorem -/
theorem shaded_cubes_count_proof (c : ShadedCube) : shaded_cubes_count c = 20 := by
  sorry

end NUMINAMATH_CALUDE_shaded_cubes_count_shaded_cubes_count_proof_l958_95838


namespace NUMINAMATH_CALUDE_perpendicular_line_proof_l958_95880

-- Define the given line
def given_line (x y : ℝ) : Prop := x - 2 * y + 3 = 0

-- Define the point P
def point_P : ℝ × ℝ := (-1, 3)

-- Define the perpendicular line
def perpendicular_line (x y : ℝ) : Prop := 2 * x + y - 1 = 0

-- Theorem statement
theorem perpendicular_line_proof :
  -- The perpendicular line passes through point P
  perpendicular_line point_P.1 point_P.2 ∧
  -- The perpendicular line is indeed perpendicular to the given line
  ∀ (x₁ y₁ x₂ y₂ : ℝ),
    given_line x₁ y₁ → given_line x₂ y₂ →
    perpendicular_line x₁ y₁ → perpendicular_line x₂ y₂ →
    (x₂ - x₁) * (x₂ - x₁) + (y₂ - y₁) * (y₂ - y₁) ≠ 0 →
    ((x₂ - x₁) * (1) + (y₂ - y₁) * (-2)) * ((x₂ - x₁) * (2) + (y₂ - y₁) * (1)) = 0 :=
by sorry

end NUMINAMATH_CALUDE_perpendicular_line_proof_l958_95880


namespace NUMINAMATH_CALUDE_problem_statement_l958_95819

def x : ℕ := 18
def y : ℕ := 8
def z : ℕ := 2

theorem problem_statement :
  -- (A) The arithmetic mean of x and y is greater than their geometric mean
  (x + y) / 2 > Real.sqrt (x * y) ∧
  -- (B) The sum of x and z is greater than their product divided by the sum of x and y
  (x + z : ℝ) > (x * z : ℝ) / (x + y) ∧
  -- (C) If the product of x and z is fixed, their sum can be made arbitrarily large
  (∀ ε > 0, ∃ k > 0, k + (x * z : ℝ) / k > 1 / ε) ∧
  -- (D) The arithmetic mean of x, y, and z is NOT greater than the sum of their squares divided by their sum
  ¬((x + y + z : ℝ) / 3 > (x^2 + y^2 + z^2 : ℝ) / (x + y + z)) :=
by sorry

end NUMINAMATH_CALUDE_problem_statement_l958_95819


namespace NUMINAMATH_CALUDE_waiter_tip_calculation_l958_95892

theorem waiter_tip_calculation (total_customers : ℕ) (non_tipping_customers : ℕ) (total_tips : ℕ) :
  total_customers = 7 →
  non_tipping_customers = 4 →
  total_tips = 27 →
  (total_tips / (total_customers - non_tipping_customers) : ℚ) = 9 := by
  sorry

end NUMINAMATH_CALUDE_waiter_tip_calculation_l958_95892


namespace NUMINAMATH_CALUDE_john_driving_time_l958_95808

theorem john_driving_time (speed : ℝ) (time_before_lunch : ℝ) (total_distance : ℝ) :
  speed = 55 →
  time_before_lunch = 2 →
  total_distance = 275 →
  (total_distance - speed * time_before_lunch) / speed = 3 := by
  sorry

end NUMINAMATH_CALUDE_john_driving_time_l958_95808


namespace NUMINAMATH_CALUDE_cubic_equation_solutions_l958_95888

theorem cubic_equation_solutions : 
  ∀ m n : ℤ, m^3 - n^3 = 2*m*n + 8 ↔ (m = 2 ∧ n = 0) ∨ (m = 0 ∧ n = -2) := by
  sorry

end NUMINAMATH_CALUDE_cubic_equation_solutions_l958_95888


namespace NUMINAMATH_CALUDE_least_value_with_specific_remainders_l958_95863

theorem least_value_with_specific_remainders :
  ∃ (N : ℕ), 
    N > 0 ∧
    N % 6 = 5 ∧
    N % 5 = 4 ∧
    N % 4 = 3 ∧
    N % 3 = 2 ∧
    N % 2 = 1 ∧
    (∀ (M : ℕ), M > 0 ∧ 
      M % 6 = 5 ∧
      M % 5 = 4 ∧
      M % 4 = 3 ∧
      M % 3 = 2 ∧
      M % 2 = 1 → M ≥ N) ∧
    N = 59 :=
by sorry

end NUMINAMATH_CALUDE_least_value_with_specific_remainders_l958_95863


namespace NUMINAMATH_CALUDE_bonus_is_ten_dollars_l958_95814

/-- Represents the payment structure for Brady's transcription job -/
structure TranscriptionJob where
  base_pay : ℚ  -- Base pay per card in dollars
  cards_for_bonus : ℕ  -- Number of cards needed for a bonus
  total_cards : ℕ  -- Total number of cards transcribed
  total_pay : ℚ  -- Total pay including bonuses in dollars

/-- Calculates the bonus amount per bonus interval -/
def bonus_amount (job : TranscriptionJob) : ℚ :=
  let base_total := job.base_pay * job.total_cards
  let bonus_count := job.total_cards / job.cards_for_bonus
  (job.total_pay - base_total) / bonus_count

/-- Theorem stating that the bonus amount is $10 for every 100 cards -/
theorem bonus_is_ten_dollars (job : TranscriptionJob) 
  (h1 : job.base_pay = 70 / 100)
  (h2 : job.cards_for_bonus = 100)
  (h3 : job.total_cards = 200)
  (h4 : job.total_pay = 160) :
  bonus_amount job = 10 := by
  sorry

end NUMINAMATH_CALUDE_bonus_is_ten_dollars_l958_95814


namespace NUMINAMATH_CALUDE_gravel_path_cost_l958_95810

def plot_length : ℝ := 110
def plot_width : ℝ := 65
def path_width : ℝ := 2.5
def cost_per_sq_meter_paise : ℝ := 80

theorem gravel_path_cost :
  let larger_length := plot_length + 2 * path_width
  let larger_width := plot_width + 2 * path_width
  let larger_area := larger_length * larger_width
  let plot_area := plot_length * plot_width
  let path_area := larger_area - plot_area
  let cost_per_sq_meter_rupees := cost_per_sq_meter_paise / 100
  path_area * cost_per_sq_meter_rupees = 720 :=
by sorry

end NUMINAMATH_CALUDE_gravel_path_cost_l958_95810


namespace NUMINAMATH_CALUDE_barefoot_kids_count_l958_95869

theorem barefoot_kids_count (total kids_with_socks kids_with_shoes kids_with_both : ℕ) :
  total = 22 ∧ kids_with_socks = 12 ∧ kids_with_shoes = 8 ∧ kids_with_both = 6 →
  total - ((kids_with_socks - kids_with_both) + (kids_with_shoes - kids_with_both) + kids_with_both) = 8 := by
sorry

end NUMINAMATH_CALUDE_barefoot_kids_count_l958_95869


namespace NUMINAMATH_CALUDE_order_of_3_is_2_l958_95805

def f (x : ℕ) : ℕ := x^2 % 13

def iterate_f (n : ℕ) (x : ℕ) : ℕ :=
  match n with
  | 0 => x
  | n+1 => f (iterate_f n x)

theorem order_of_3_is_2 : 
  (∃ m : ℕ, m > 0 ∧ iterate_f m 3 = 3) ∧ 
  (∀ k : ℕ, k > 0 ∧ k < 2 → iterate_f k 3 ≠ 3) :=
sorry

end NUMINAMATH_CALUDE_order_of_3_is_2_l958_95805


namespace NUMINAMATH_CALUDE_betty_order_cost_l958_95878

/-- The total cost of Betty's order -/
def total_cost (slipper_price lipstick_price hair_color_price sunglasses_price tshirt_price : ℚ) 
  (slipper_qty lipstick_qty hair_color_qty sunglasses_qty tshirt_qty : ℕ) : ℚ :=
  slipper_price * slipper_qty + 
  lipstick_price * lipstick_qty + 
  hair_color_price * hair_color_qty + 
  sunglasses_price * sunglasses_qty + 
  tshirt_price * tshirt_qty

/-- The theorem stating that Betty's total order cost is $110.25 -/
theorem betty_order_cost : 
  total_cost 2.5 1.25 3 5.75 12.25 6 4 8 3 4 = 110.25 := by
  sorry

end NUMINAMATH_CALUDE_betty_order_cost_l958_95878


namespace NUMINAMATH_CALUDE_matrix_product_sum_l958_95881

def A (y : ℝ) : Matrix (Fin 2) (Fin 2) ℝ := !![1, 2; y, 4]
def B (x : ℝ) : Matrix (Fin 2) (Fin 2) ℝ := !![x, 6; 7, 8]

theorem matrix_product_sum (x y : ℝ) :
  A y * B x = !![19, 22; 43, 50] →
  x + y = 8 := by sorry

end NUMINAMATH_CALUDE_matrix_product_sum_l958_95881


namespace NUMINAMATH_CALUDE_hexagon_area_equals_six_l958_95870

/-- Given an equilateral triangle with area 4 and a regular hexagon with the same perimeter,
    prove that the area of the hexagon is 6. -/
theorem hexagon_area_equals_six (s t : ℝ) : 
  s > 0 → t > 0 → -- Positive side lengths
  3 * s = 6 * t → -- Equal perimeters
  s^2 * Real.sqrt 3 / 4 = 4 → -- Triangle area
  6 * (t^2 * Real.sqrt 3 / 4) = 6 := by
sorry


end NUMINAMATH_CALUDE_hexagon_area_equals_six_l958_95870


namespace NUMINAMATH_CALUDE_factor_sum_l958_95839

theorem factor_sum (P Q : ℝ) : 
  (∃ b c : ℝ, (X^2 + 4*X + 3) * (X^2 + b*X + c) = X^4 + P*X^2 + Q) →
  P + Q = -1 :=
by sorry

end NUMINAMATH_CALUDE_factor_sum_l958_95839


namespace NUMINAMATH_CALUDE_locus_is_apollonian_circle_l958_95896

/-- An Apollonian circle is the locus of points with a constant ratio of distances to two fixed points. -/
def ApollonianCircle (A B : ℝ × ℝ) (k : ℝ) : Set (ℝ × ℝ) :=
  {M | dist A M / dist M B = k}

/-- The locus of points M satisfying |AM| : |MB| = k ≠ 1, where A and B are fixed points, is an Apollonian circle. -/
theorem locus_is_apollonian_circle (A B : ℝ × ℝ) (k : ℝ) (h : k ≠ 1) :
  {M : ℝ × ℝ | dist A M / dist M B = k} = ApollonianCircle A B k := by
  sorry

end NUMINAMATH_CALUDE_locus_is_apollonian_circle_l958_95896


namespace NUMINAMATH_CALUDE_annie_laps_bonnie_l958_95800

/-- The length of the circular track in meters -/
def track_length : ℝ := 500

/-- Annie's speed relative to Bonnie's -/
def annie_speed_ratio : ℝ := 1.5

/-- The number of laps Annie has run when she first laps Bonnie -/
def annie_laps : ℝ := 3

theorem annie_laps_bonnie :
  track_length > 0 →
  annie_speed_ratio = 1.5 →
  (annie_laps * track_length) / annie_speed_ratio = (annie_laps - 1) * track_length :=
by sorry

end NUMINAMATH_CALUDE_annie_laps_bonnie_l958_95800


namespace NUMINAMATH_CALUDE_expression_simplification_and_evaluation_l958_95867

theorem expression_simplification_and_evaluation (x : ℝ) (h : x = Real.sqrt 2 - 1) :
  (1 - 1 / (x + 1)) / (x / (x^2 + 2*x + 1)) = Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_and_evaluation_l958_95867


namespace NUMINAMATH_CALUDE_lego_count_l958_95835

theorem lego_count (initial : Nat) (lost : Nat) (remaining : Nat) : 
  initial = 380 → lost = 57 → remaining = initial - lost → remaining = 323 := by
  sorry

end NUMINAMATH_CALUDE_lego_count_l958_95835
