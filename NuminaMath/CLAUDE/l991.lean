import Mathlib

namespace NUMINAMATH_CALUDE_range_of_a_l991_99177

-- Define the function f
def f (x : ℝ) : ℝ := x * (abs x + 4)

-- State the theorem
theorem range_of_a (a : ℝ) :
  (f (a^2) + f a < 0) → (-1 < a ∧ a < 0) :=
by sorry

end NUMINAMATH_CALUDE_range_of_a_l991_99177


namespace NUMINAMATH_CALUDE_max_prob_at_one_l991_99195

def binomial_prob (n k : ℕ) (p : ℝ) : ℝ :=
  (n.choose k) * p^k * (1 - p)^(n - k)

theorem max_prob_at_one :
  let n : ℕ := 5
  let p : ℝ := 1/4
  ∀ k : ℕ, k ≠ 1 → k ≤ n → binomial_prob n 1 p > binomial_prob n k p :=
by sorry

end NUMINAMATH_CALUDE_max_prob_at_one_l991_99195


namespace NUMINAMATH_CALUDE_book_arrangement_theorem_l991_99172

theorem book_arrangement_theorem :
  let math_books : ℕ := 4
  let english_books : ℕ := 4
  let group_arrangements : ℕ := 2  -- math books and English books as two groups
  let total_arrangements : ℕ := group_arrangements.factorial * math_books.factorial * english_books.factorial
  total_arrangements = 1152 := by
  sorry

end NUMINAMATH_CALUDE_book_arrangement_theorem_l991_99172


namespace NUMINAMATH_CALUDE_apple_distribution_l991_99135

theorem apple_distribution (students : ℕ) (apples : ℕ) : 
  (apples = 4 * students + 3) ∧ 
  (6 * (students - 1) ≤ apples) ∧ 
  (apples ≤ 6 * (students - 1) + 2) →
  (students = 4 ∧ apples = 19) :=
sorry

end NUMINAMATH_CALUDE_apple_distribution_l991_99135


namespace NUMINAMATH_CALUDE_subset_probability_l991_99109

def S : Finset Char := {'a', 'b', 'c', 'd', 'e'}
def T : Finset Char := {'a', 'b', 'c'}

theorem subset_probability : 
  (Finset.filter (fun X => X ⊆ T) (Finset.powerset S)).card / (Finset.powerset S).card = 1 / 4 := by
  sorry

end NUMINAMATH_CALUDE_subset_probability_l991_99109


namespace NUMINAMATH_CALUDE_pencils_bought_on_monday_l991_99155

theorem pencils_bought_on_monday (P : ℕ) : P = 20 :=
  by
  -- Define the number of pencils bought on Tuesday
  let tuesday_pencils := 18

  -- Define the number of pencils bought on Wednesday
  let wednesday_pencils := 3 * tuesday_pencils

  -- Define the total number of pencils
  let total_pencils := 92

  -- Assert that the sum of pencils from all days equals the total
  have h : P + tuesday_pencils + wednesday_pencils = total_pencils := by sorry

  -- Prove that P equals 20
  sorry

end NUMINAMATH_CALUDE_pencils_bought_on_monday_l991_99155


namespace NUMINAMATH_CALUDE_parallel_lines_sum_l991_99162

/-- Two parallel lines with a specific distance between them -/
structure ParallelLines where
  m : ℝ
  n : ℝ
  m_pos : m > 0
  parallel : 1 / (-2) = 2 / n
  distance : 2 * Real.sqrt 5 = |m + 3| / Real.sqrt 5

/-- The sum of coefficients m and n for parallel lines with given properties -/
theorem parallel_lines_sum (l : ParallelLines) : l.m + l.n = 3 := by
  sorry

end NUMINAMATH_CALUDE_parallel_lines_sum_l991_99162


namespace NUMINAMATH_CALUDE_no_nontrivial_solutions_l991_99180

theorem no_nontrivial_solutions (p : ℕ) (hp : Nat.Prime p) (hp_odd : p % 2 = 1)
  (h2p1 : Nat.Prime (2 * p + 1)) :
  ∀ x y z : ℤ, x^p + 2*y^p + 5*z^p = 0 → x = 0 ∧ y = 0 ∧ z = 0 := by
  sorry

end NUMINAMATH_CALUDE_no_nontrivial_solutions_l991_99180


namespace NUMINAMATH_CALUDE_carrots_taken_l991_99104

theorem carrots_taken (initial_carrots remaining_carrots : ℕ) :
  initial_carrots = 6 →
  remaining_carrots = 3 →
  initial_carrots - remaining_carrots = 3 :=
by sorry

end NUMINAMATH_CALUDE_carrots_taken_l991_99104


namespace NUMINAMATH_CALUDE_star_calculation_l991_99137

def star (a b : ℚ) : ℚ := (a + b) / 4

theorem star_calculation : star (star 3 8) 6 = 35 / 16 := by
  sorry

end NUMINAMATH_CALUDE_star_calculation_l991_99137


namespace NUMINAMATH_CALUDE_brothers_savings_l991_99110

def isabelle_ticket_cost : ℕ := 20
def brother_ticket_cost : ℕ := 10
def number_of_brothers : ℕ := 2
def isabelle_savings : ℕ := 5
def work_weeks : ℕ := 10
def weekly_earnings : ℕ := 3

def total_ticket_cost : ℕ := isabelle_ticket_cost + number_of_brothers * brother_ticket_cost

def isabelle_total_earnings : ℕ := isabelle_savings + work_weeks * weekly_earnings

theorem brothers_savings : 
  total_ticket_cost - isabelle_total_earnings = 5 := by sorry

end NUMINAMATH_CALUDE_brothers_savings_l991_99110


namespace NUMINAMATH_CALUDE_complex_determinant_equation_l991_99169

def determinant (a b c d : ℂ) : ℂ := a * d - b * c

theorem complex_determinant_equation :
  ∀ z : ℂ, determinant z i 1 i = 1 + i → z = 2 - i := by sorry

end NUMINAMATH_CALUDE_complex_determinant_equation_l991_99169


namespace NUMINAMATH_CALUDE_max_product_of_digits_l991_99185

def is_digit (n : ℕ) : Prop := n ≤ 9

theorem max_product_of_digits (E F G H : ℕ) 
  (hE : is_digit E) (hF : is_digit F) (hG : is_digit G) (hH : is_digit H)
  (distinct : E ≠ F ∧ E ≠ G ∧ E ≠ H ∧ F ≠ G ∧ F ≠ H ∧ G ≠ H)
  (h_int : ∃ (k : ℕ), E * F = k * (G - H))
  (h_max : ∀ (E' F' G' H' : ℕ), 
    is_digit E' → is_digit F' → is_digit G' → is_digit H' →
    E' ≠ F' ∧ E' ≠ G' ∧ E' ≠ H' ∧ F' ≠ G' ∧ F' ≠ H' ∧ G' ≠ H' →
    (∃ (k' : ℕ), E' * F' = k' * (G' - H')) →
    E * F ≥ E' * F') :
  E * F = 72 :=
sorry

end NUMINAMATH_CALUDE_max_product_of_digits_l991_99185


namespace NUMINAMATH_CALUDE_star_one_one_eq_neg_eleven_l991_99146

/-- A custom binary operation on real numbers -/
noncomputable def star (a b : ℝ) (x y : ℝ) : ℝ := a * x + b * y

/-- Theorem stating that given the conditions, 1 * 1 = -11 -/
theorem star_one_one_eq_neg_eleven 
  (a b : ℝ) 
  (h1 : star a b 3 5 = 15) 
  (h2 : star a b 4 7 = 28) : 
  star a b 1 1 = -11 := by
  sorry

#check star_one_one_eq_neg_eleven

end NUMINAMATH_CALUDE_star_one_one_eq_neg_eleven_l991_99146


namespace NUMINAMATH_CALUDE_max_product_sum_180_l991_99132

theorem max_product_sum_180 : 
  ∀ a b : ℤ, a + b = 180 → a * b ≤ 8100 := by
  sorry

end NUMINAMATH_CALUDE_max_product_sum_180_l991_99132


namespace NUMINAMATH_CALUDE_quadratic_function_property_l991_99123

theorem quadratic_function_property (a b c : ℝ) :
  (∀ x, (1 < x ∧ x < c) → (a * x^2 + b * x + c < 0)) →
  a = 1 := by
sorry

end NUMINAMATH_CALUDE_quadratic_function_property_l991_99123


namespace NUMINAMATH_CALUDE_sailboat_speed_proof_l991_99189

/-- The speed of a sailboat with two sails in knots -/
def speed_two_sails : ℝ := 50

/-- The conversion factor from nautical miles to land miles -/
def nautical_to_land : ℝ := 1.15

/-- The time spent sailing with one sail in hours -/
def time_one_sail : ℝ := 4

/-- The time spent sailing with two sails in hours -/
def time_two_sails : ℝ := 4

/-- The total distance traveled in land miles -/
def total_distance_land : ℝ := 345

/-- The speed of the sailboat with one sail in knots -/
def speed_one_sail : ℝ := 25

theorem sailboat_speed_proof :
  speed_one_sail * time_one_sail + speed_two_sails * time_two_sails =
  total_distance_land / nautical_to_land := by
  sorry

end NUMINAMATH_CALUDE_sailboat_speed_proof_l991_99189


namespace NUMINAMATH_CALUDE_square_with_semicircles_area_ratio_l991_99144

/-- The ratio of areas for a square with semicircular arcs -/
theorem square_with_semicircles_area_ratio :
  let square_side : ℝ := 6
  let square_area : ℝ := square_side ^ 2
  let semicircle_radius : ℝ := square_side / 2
  let semicircle_area : ℝ := π * semicircle_radius ^ 2 / 2
  let new_figure_area : ℝ := square_area + 4 * semicircle_area
  new_figure_area / square_area = 1 + π / 2 :=
by sorry

end NUMINAMATH_CALUDE_square_with_semicircles_area_ratio_l991_99144


namespace NUMINAMATH_CALUDE_drill_bits_purchase_cost_l991_99116

/-- The total cost of a purchase with tax -/
def total_cost (num_sets : ℕ) (price_per_set : ℚ) (tax_rate : ℚ) : ℚ :=
  let pre_tax_cost := num_sets * price_per_set
  let tax := pre_tax_cost * tax_rate
  pre_tax_cost + tax

/-- Theorem: The total cost for 5 sets of drill bits at $6 each with 10% tax is $33 -/
theorem drill_bits_purchase_cost :
  total_cost 5 6 (1/10) = 33 := by
  sorry

end NUMINAMATH_CALUDE_drill_bits_purchase_cost_l991_99116


namespace NUMINAMATH_CALUDE_philip_intersections_l991_99163

theorem philip_intersections (crosswalks_per_intersection : ℕ) 
                              (lines_per_crosswalk : ℕ) 
                              (total_lines : ℕ) :
  crosswalks_per_intersection = 4 →
  lines_per_crosswalk = 20 →
  total_lines = 400 →
  total_lines / (crosswalks_per_intersection * lines_per_crosswalk) = 5 :=
by
  sorry

end NUMINAMATH_CALUDE_philip_intersections_l991_99163


namespace NUMINAMATH_CALUDE_episode_length_l991_99193

theorem episode_length
  (num_episodes : ℕ)
  (watching_hours_per_day : ℕ)
  (total_days : ℕ)
  (h1 : num_episodes = 90)
  (h2 : watching_hours_per_day = 2)
  (h3 : total_days = 15) :
  (total_days * watching_hours_per_day * 60) / num_episodes = 20 :=
by sorry

end NUMINAMATH_CALUDE_episode_length_l991_99193


namespace NUMINAMATH_CALUDE_plate_on_square_table_l991_99115

/-- Given a square table with a round plate, if the distances from the plate to the table edges
    on one side are 10 cm and 63 cm, and on the opposite side are 20 cm and x cm,
    then x = 53 cm. -/
theorem plate_on_square_table (x : ℝ) : x = 53 := by
  sorry

end NUMINAMATH_CALUDE_plate_on_square_table_l991_99115


namespace NUMINAMATH_CALUDE_positive_integer_solutions_m_value_when_x_equals_y_fixed_solution_l991_99134

-- Define the system of equations
def equation1 (x y : ℤ) : Prop := 2*x + y - 6 = 0
def equation2 (x y m : ℤ) : Prop := 2*x - 2*y + m*y + 8 = 0

-- Theorem for part 1
theorem positive_integer_solutions :
  ∀ x y : ℤ, x > 0 ∧ y > 0 ∧ equation1 x y ↔ (x = 2 ∧ y = 2) ∨ (x = 1 ∧ y = 4) :=
sorry

-- Theorem for part 2
theorem m_value_when_x_equals_y :
  ∃ m : ℤ, ∀ x y : ℤ, x = y ∧ equation1 x y ∧ equation2 x y m → m = -4 :=
sorry

-- Theorem for part 3
theorem fixed_solution :
  ∀ m : ℤ, equation2 (-4) 0 m :=
sorry

end NUMINAMATH_CALUDE_positive_integer_solutions_m_value_when_x_equals_y_fixed_solution_l991_99134


namespace NUMINAMATH_CALUDE_square_sum_of_product_and_sum_l991_99191

theorem square_sum_of_product_and_sum (p q : ℝ) 
  (h1 : p * q = 12) 
  (h2 : p + q = 8) : 
  p^2 + q^2 = 40 := by
  sorry

end NUMINAMATH_CALUDE_square_sum_of_product_and_sum_l991_99191


namespace NUMINAMATH_CALUDE_sqrt_54_minus_4_bounds_l991_99100

theorem sqrt_54_minus_4_bounds : 3 < Real.sqrt 54 - 4 ∧ Real.sqrt 54 - 4 < 4 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_54_minus_4_bounds_l991_99100


namespace NUMINAMATH_CALUDE_negation_of_tan_gt_sin_l991_99158

open Real

theorem negation_of_tan_gt_sin :
  (¬ (∀ x, -π/2 < x ∧ x < π/2 → tan x > sin x)) ↔
  (∃ x, -π/2 < x ∧ x < π/2 ∧ tan x ≤ sin x) := by sorry

end NUMINAMATH_CALUDE_negation_of_tan_gt_sin_l991_99158


namespace NUMINAMATH_CALUDE_sector_central_angle_l991_99150

/-- The central angle of a sector with radius R and circumference 3R is 1 radian. -/
theorem sector_central_angle (R : ℝ) (R_pos : R > 0) : 
  let circumference := 3 * R
  let central_angle := circumference / R - 2
  central_angle = 1 := by sorry

end NUMINAMATH_CALUDE_sector_central_angle_l991_99150


namespace NUMINAMATH_CALUDE_max_omega_value_l991_99118

/-- The function f(x) defined in the problem -/
noncomputable def f (ω φ : ℝ) (x : ℝ) : ℝ := 2 * Real.sin (ω * x + φ)

/-- The theorem stating the maximum value of ω -/
theorem max_omega_value (ω φ : ℝ) :
  ω > 0 →
  0 < φ ∧ φ < Real.pi / 2 →
  f ω φ (-Real.pi / 4) = 0 →
  (∀ x, f ω φ (Real.pi / 4 - x) = f ω φ (Real.pi / 4 + x)) →
  (∀ x y, Real.pi / 18 < x ∧ x < y ∧ y < 2 * Real.pi / 9 → 
    (f ω φ x < f ω φ y ∨ f ω φ x > f ω φ y)) →
  ω ≤ 5 :=
by sorry

end NUMINAMATH_CALUDE_max_omega_value_l991_99118


namespace NUMINAMATH_CALUDE_competitive_exam_selection_difference_l991_99133

theorem competitive_exam_selection_difference (total_candidates : ℕ) 
  (selection_rate_A : ℚ) (selection_rate_B : ℚ) : 
  total_candidates = 7900 → 
  selection_rate_A = 6 / 100 →
  selection_rate_B = 7 / 100 →
  (selection_rate_B - selection_rate_A) * total_candidates = 79 := by
sorry

end NUMINAMATH_CALUDE_competitive_exam_selection_difference_l991_99133


namespace NUMINAMATH_CALUDE_overlapping_triangle_area_l991_99159

/-- Given a rectangle with length 8 and width 4, when folded along its diagonal,
    the area of the overlapping triangle is 10. -/
theorem overlapping_triangle_area (length width : ℝ) (h1 : length = 8) (h2 : width = 4) :
  let diagonal := Real.sqrt (length ^ 2 + width ^ 2)
  let overlap_base := (length ^ 2 + width ^ 2) / (2 * length)
  let overlap_height := width
  let overlap_area := (1 / 2) * overlap_base * overlap_height
  overlap_area = 10 := by
sorry

end NUMINAMATH_CALUDE_overlapping_triangle_area_l991_99159


namespace NUMINAMATH_CALUDE_solve_equation_and_evaluate_l991_99149

theorem solve_equation_and_evaluate (x : ℚ) : 
  (4 * x - 3 = 13 * x + 12) → (5 * (x + 4) = 35 / 3) := by
  sorry

end NUMINAMATH_CALUDE_solve_equation_and_evaluate_l991_99149


namespace NUMINAMATH_CALUDE_minimum_bailing_rate_l991_99147

/-- Represents the minimum bailing rate problem --/
theorem minimum_bailing_rate
  (distance_to_shore : ℝ)
  (rowing_speed : ℝ)
  (water_intake_rate : ℝ)
  (max_water_capacity : ℝ)
  (h1 : distance_to_shore = 2)
  (h2 : rowing_speed = 3)
  (h3 : water_intake_rate = 8)
  (h4 : max_water_capacity = 50)
  : ∃ (min_bailing_rate : ℝ),
    min_bailing_rate ≥ 7 ∧
    (distance_to_shore / rowing_speed) * (water_intake_rate - min_bailing_rate) ≤ max_water_capacity ∧
    ∀ (r : ℝ), r < min_bailing_rate →
      (distance_to_shore / rowing_speed) * (water_intake_rate - r) > max_water_capacity :=
by sorry

end NUMINAMATH_CALUDE_minimum_bailing_rate_l991_99147


namespace NUMINAMATH_CALUDE_class_ratio_and_total_l991_99194

theorem class_ratio_and_total (num_girls : ℕ) (num_boys : ℕ) : 
  (3 : ℚ) / 7 * num_boys = (6 : ℚ) / 11 * num_girls → 
  num_girls = 22 →
  (num_boys : ℚ) / num_girls = 14 / 11 ∧ num_boys + num_girls = 50 := by
  sorry

end NUMINAMATH_CALUDE_class_ratio_and_total_l991_99194


namespace NUMINAMATH_CALUDE_student_selection_l991_99173

/-- The number of ways to select 3 students from a group of 4 boys and 3 girls, 
    including both boys and girls. -/
theorem student_selection (boys : Nat) (girls : Nat) : 
  boys = 4 → girls = 3 → Nat.choose boys 2 * Nat.choose girls 1 + 
                         Nat.choose boys 1 * Nat.choose girls 2 = 30 := by
  sorry

#eval Nat.choose 4 2 * Nat.choose 3 1 + Nat.choose 4 1 * Nat.choose 3 2

end NUMINAMATH_CALUDE_student_selection_l991_99173


namespace NUMINAMATH_CALUDE_grid_block_selection_l991_99156

theorem grid_block_selection (n : ℕ) (k : ℕ) : 
  n = 7 → k = 4 → (n.choose k) * (n.choose k) * k.factorial = 29400 := by
  sorry

end NUMINAMATH_CALUDE_grid_block_selection_l991_99156


namespace NUMINAMATH_CALUDE_range_of_a_range_of_b_l991_99131

-- Define the functions f and g
def f (a : ℝ) (x : ℝ) : ℝ := x^2 - 4*x + a + 3
def g (b : ℝ) (x : ℝ) : ℝ := b*x + 5 - 2*b

-- Theorem for part 1
theorem range_of_a (a : ℝ) :
  (∃ x ∈ Set.Icc (-1 : ℝ) 1, f a x = 0) →
  -8 ≤ a ∧ a ≤ 0 :=
sorry

-- Theorem for part 2
theorem range_of_b (b : ℝ) :
  (∀ x₁ ∈ Set.Icc (1 : ℝ) 4, ∃ x₂ ∈ Set.Icc (1 : ℝ) 4, g b x₁ = f 3 x₂) →
  -1 ≤ b ∧ b ≤ 1/2 :=
sorry

end NUMINAMATH_CALUDE_range_of_a_range_of_b_l991_99131


namespace NUMINAMATH_CALUDE_arithmetic_number_difference_l991_99197

-- Define a function to check if a number is a valid 3-digit arithmetic number
def isArithmeticNumber (n : ℕ) : Prop :=
  100 ≤ n ∧ n ≤ 999 ∧
  ∃ (a b c : ℕ), n = 100 * a + 10 * b + c ∧
                  a ≠ b ∧ b ≠ c ∧ a ≠ c ∧
                  b - a = c - b

-- Define the largest and smallest arithmetic numbers
def largestArithmeticNumber : ℕ := 759
def smallestArithmeticNumber : ℕ := 123

-- State the theorem
theorem arithmetic_number_difference :
  isArithmeticNumber largestArithmeticNumber ∧
  isArithmeticNumber smallestArithmeticNumber ∧
  (∀ n : ℕ, isArithmeticNumber n → smallestArithmeticNumber ≤ n ∧ n ≤ largestArithmeticNumber) ∧
  largestArithmeticNumber - smallestArithmeticNumber = 636 := by
  sorry


end NUMINAMATH_CALUDE_arithmetic_number_difference_l991_99197


namespace NUMINAMATH_CALUDE_greatest_abcba_divisible_by_13_l991_99176

def is_valid_abcba (n : ℕ) : Prop :=
  10000 ≤ n ∧ n < 100000 ∧
  ∃ (a b c : ℕ),
    a < 10 ∧ b < 10 ∧ c < 10 ∧
    a ≠ b ∧ b ≠ c ∧ a ≠ c ∧
    n = 10000 * a + 1000 * b + 100 * c + 10 * b + a

theorem greatest_abcba_divisible_by_13 :
  ∀ n : ℕ, is_valid_abcba n → n % 13 = 0 → n ≤ 83638 :=
by sorry

end NUMINAMATH_CALUDE_greatest_abcba_divisible_by_13_l991_99176


namespace NUMINAMATH_CALUDE_partial_fraction_sum_l991_99105

theorem partial_fraction_sum (x : ℝ) (A B C D E : ℝ) : 
  (1 : ℝ) / (x * (x + 1) * (x + 2) * (x + 3) * (x + 4)) = 
    A / x + B / (x + 1) + C / (x + 2) + D / (x + 3) + E / (x + 4) →
  A + B + C + D + E = 0 := by
sorry

end NUMINAMATH_CALUDE_partial_fraction_sum_l991_99105


namespace NUMINAMATH_CALUDE_sum_diff_difference_is_six_l991_99106

/-- A two-digit number with specific properties -/
structure TwoDigitNumber where
  tens : Nat
  ones : Nat
  is_two_digit : 10 ≤ 10 * tens + ones ∧ 10 * tens + ones < 100
  digit_ratio : ones = 2 * tens
  interchange_diff : 10 * ones + tens - (10 * tens + ones) = 36

/-- The difference between the sum and difference of digits for a TwoDigitNumber -/
def sum_diff_difference (n : TwoDigitNumber) : Nat :=
  (n.tens + n.ones) - (n.ones - n.tens)

theorem sum_diff_difference_is_six (n : TwoDigitNumber) :
  sum_diff_difference n = 6 := by
  sorry

end NUMINAMATH_CALUDE_sum_diff_difference_is_six_l991_99106


namespace NUMINAMATH_CALUDE_exists_binary_sequence_with_geometric_partial_sums_l991_99183

/-- A sequence where each term is either 0 or 1 -/
def BinarySequence := ℕ → Fin 2

/-- The partial sum of the first n terms of a BinarySequence -/
def PartialSum (a : BinarySequence) (n : ℕ) : ℕ :=
  (Finset.range n).sum (fun i => a i)

/-- A sequence of partial sums forms a geometric sequence -/
def IsGeometricSequence (S : ℕ → ℕ) : Prop :=
  ∃ (r : ℚ), ∀ n : ℕ, S (n + 1) = (r : ℚ) * S n

/-- There exists a BinarySequence whose partial sums form a geometric sequence -/
theorem exists_binary_sequence_with_geometric_partial_sums :
  ∃ (a : BinarySequence), IsGeometricSequence (PartialSum a) := by
  sorry

end NUMINAMATH_CALUDE_exists_binary_sequence_with_geometric_partial_sums_l991_99183


namespace NUMINAMATH_CALUDE_disjunction_true_l991_99196

open Real

theorem disjunction_true : 
  (¬(∀ α : ℝ, sin (π - α) ≠ -sin α)) ∨ (∃ x : ℝ, x ≥ 0 ∧ sin x > x) := by
  sorry

end NUMINAMATH_CALUDE_disjunction_true_l991_99196


namespace NUMINAMATH_CALUDE_restaurant_bill_change_l991_99190

theorem restaurant_bill_change (meal_cost drink_cost tip_percentage bill_amount : ℚ) : 
  meal_cost = 10 ∧ 
  drink_cost = 2.5 ∧ 
  tip_percentage = 0.2 ∧ 
  bill_amount = 20 → 
  bill_amount - (meal_cost + drink_cost + (meal_cost + drink_cost) * tip_percentage) = 5 := by
  sorry

end NUMINAMATH_CALUDE_restaurant_bill_change_l991_99190


namespace NUMINAMATH_CALUDE_circles_intersect_l991_99179

-- Define the circles
def circle_O1 (x y : ℝ) : Prop := (x - 1)^2 + y^2 = 1
def circle_O2 (x y : ℝ) : Prop := x^2 + (y - 3)^2 = 9

-- Define the centers and radii
def center_O1 : ℝ × ℝ := (1, 0)
def center_O2 : ℝ × ℝ := (0, 3)
def radius_O1 : ℝ := 1
def radius_O2 : ℝ := 3

-- Theorem stating that the circles are intersecting
theorem circles_intersect : 
  let d := Real.sqrt ((center_O1.1 - center_O2.1)^2 + (center_O1.2 - center_O2.2)^2)
  (radius_O2 - radius_O1 < d) ∧ (d < radius_O1 + radius_O2) := by
  sorry


end NUMINAMATH_CALUDE_circles_intersect_l991_99179


namespace NUMINAMATH_CALUDE_knights_and_knaves_l991_99114

-- Define the type for individuals
inductive Person : Type
| A
| B
| C

-- Define the type for knight/knave status
inductive Status : Type
| Knight
| Knave

-- Function to determine if a person is a knight
def isKnight (p : Person) (s : Person → Status) : Prop :=
  s p = Status.Knight

-- Function to determine if a person is a knave
def isKnave (p : Person) (s : Person → Status) : Prop :=
  s p = Status.Knave

-- A's statement
def A_statement (s : Person → Status) : Prop :=
  isKnight Person.C s → isKnave Person.B s

-- C's statement
def C_statement (s : Person → Status) : Prop :=
  (isKnight Person.A s ∧ isKnave Person.C s) ∨ (isKnave Person.A s ∧ isKnight Person.C s)

-- Main theorem
theorem knights_and_knaves :
  ∃ (s : Person → Status),
    (∀ p, (isKnight p s → A_statement s = true) ∧ (isKnave p s → A_statement s = false)) ∧
    (∀ p, (isKnight p s → C_statement s = true) ∧ (isKnave p s → C_statement s = false)) ∧
    isKnave Person.A s ∧ isKnight Person.B s ∧ isKnight Person.C s :=
sorry

end NUMINAMATH_CALUDE_knights_and_knaves_l991_99114


namespace NUMINAMATH_CALUDE_metal_sheet_dimensions_l991_99124

theorem metal_sheet_dimensions (a : ℝ) :
  (a > 0) →
  (2*a > 6) →
  (a > 6) →
  (3 * (2*a - 6) * (a - 6) = 168) →
  (a = 10) := by
sorry

end NUMINAMATH_CALUDE_metal_sheet_dimensions_l991_99124


namespace NUMINAMATH_CALUDE_f_negative_iff_x_in_unit_interval_l991_99103

/-- The function f(x) = x^2 - x^(1/2) is negative if and only if x is in the open interval (0, 1) -/
theorem f_negative_iff_x_in_unit_interval (x : ℝ) :
  x^2 - x^(1/2) < 0 ↔ 0 < x ∧ x < 1 := by
  sorry

end NUMINAMATH_CALUDE_f_negative_iff_x_in_unit_interval_l991_99103


namespace NUMINAMATH_CALUDE_system_solution_l991_99143

theorem system_solution (x : Fin 1995 → ℤ) 
  (h : ∀ i : Fin 1995, x i ^ 2 = 1 + x ((i + 1993) % 1995) * x ((i + 1994) % 1995)) :
  (∀ i : Fin 1995, i % 3 = 1 → x i = 0) ∧
  (∀ i : Fin 1995, i % 3 ≠ 1 → x i = 1 ∨ x i = -1) ∧
  (∀ i : Fin 1995, i % 3 ≠ 1 → x i = -x ((i + 1) % 1995)) :=
by sorry

end NUMINAMATH_CALUDE_system_solution_l991_99143


namespace NUMINAMATH_CALUDE_class_test_probability_l991_99127

theorem class_test_probability (p_first p_second p_both : ℝ) 
  (h1 : p_first = 0.75)
  (h2 : p_second = 0.25)
  (h3 : p_both = 0.20) :
  1 - (p_first + p_second - p_both) = 0.20 := by
  sorry

end NUMINAMATH_CALUDE_class_test_probability_l991_99127


namespace NUMINAMATH_CALUDE_sqrt_expression_simplification_l991_99157

theorem sqrt_expression_simplification :
  2 * Real.sqrt 3 - (3 * Real.sqrt 2 + Real.sqrt 3) = Real.sqrt 3 - 3 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_expression_simplification_l991_99157


namespace NUMINAMATH_CALUDE_reflect_P_across_x_axis_l991_99165

/-- Reflects a point across the x-axis in a 2D Cartesian coordinate system. -/
def reflect_x (p : ℝ × ℝ) : ℝ × ℝ := (p.1, -p.2)

/-- The original point P in the Cartesian coordinate system. -/
def P : ℝ × ℝ := (-2, -3)

/-- Theorem: Reflecting the point P(-2,-3) across the x-axis results in the coordinates (-2, 3). -/
theorem reflect_P_across_x_axis :
  reflect_x P = (-2, 3) := by sorry

end NUMINAMATH_CALUDE_reflect_P_across_x_axis_l991_99165


namespace NUMINAMATH_CALUDE_log_sum_problem_l991_99160

-- Define the logarithm function
noncomputable def log (base : ℝ) (x : ℝ) : ℝ := Real.log x / Real.log base

-- State the theorem
theorem log_sum_problem (x y z : ℝ) 
  (hx : log 3 (log 4 (log 5 x)) = 0)
  (hy : log 4 (log 5 (log 3 y)) = 0)
  (hz : log 5 (log 3 (log 4 z)) = 0) :
  x + y + z = 932 := by
  sorry

end NUMINAMATH_CALUDE_log_sum_problem_l991_99160


namespace NUMINAMATH_CALUDE_equal_positive_integers_l991_99102

theorem equal_positive_integers (a b : ℕ) (h : ∀ n : ℕ, n > 0 → ∃ k : ℕ, b^n + n = k * (a^n + n)) : a = b := by
  sorry

end NUMINAMATH_CALUDE_equal_positive_integers_l991_99102


namespace NUMINAMATH_CALUDE_inequality_problems_l991_99136

theorem inequality_problems (x : ℝ) :
  ((-x^2 + 4*x - 4 < 0) ↔ (x ≠ 2)) ∧
  ((((1 - x) / (x - 5)) > 0) ↔ (1 < x ∧ x < 5)) := by
  sorry

end NUMINAMATH_CALUDE_inequality_problems_l991_99136


namespace NUMINAMATH_CALUDE_cubic_function_property_l991_99171

/-- A cubic function with integer coefficients -/
def f (a b c : ℤ) (x : ℝ) : ℝ := x^3 + a*x^2 + b*x + c

/-- Theorem: If f(a) = a^3 and f(b) = b^3, then c = 16 -/
theorem cubic_function_property (a b c : ℤ) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) 
  (h1 : f a b c a = a^3) (h2 : f a b c b = b^3) : c = 16 := by
  sorry

end NUMINAMATH_CALUDE_cubic_function_property_l991_99171


namespace NUMINAMATH_CALUDE_quadratic_equation_transformation_l991_99122

theorem quadratic_equation_transformation (x : ℝ) : 
  ((x - 1) * (x + 1) = 1) ↔ (x^2 - 2 = 0) := by
sorry

end NUMINAMATH_CALUDE_quadratic_equation_transformation_l991_99122


namespace NUMINAMATH_CALUDE_sum_of_numbers_l991_99151

theorem sum_of_numbers : let numbers := [0.8, 1/2, 0.5]
  (∀ x ∈ numbers, x ≤ 2) →
  numbers.sum = 1.8 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_numbers_l991_99151


namespace NUMINAMATH_CALUDE_complex_number_and_imaginary_root_l991_99181

theorem complex_number_and_imaginary_root (z : ℂ) (m : ℂ) : 
  (∃ (r : ℝ), z + Complex.I = r) →
  (∃ (s : ℝ), z / (1 - Complex.I) = s) →
  (∃ (t : ℝ), m = Complex.I * t) →
  (∃ (x : ℝ), (x^2 : ℂ) + x * (1 + z) - (3 * m - 1) * Complex.I = 0) →
  z = 1 - Complex.I ∧ m = -Complex.I := by
sorry

end NUMINAMATH_CALUDE_complex_number_and_imaginary_root_l991_99181


namespace NUMINAMATH_CALUDE_perpendicular_to_plane_iff_perpendicular_to_all_lines_perpendicular_parallel_transitive_l991_99111

-- Define the basic types
variable (Point Line Plane : Type)

-- Define the relations
variable (perpendicular : Line → Line → Prop)
variable (perpendicular_to_plane : Line → Plane → Prop)
variable (in_plane : Line → Plane → Prop)
variable (parallel : Line → Line → Prop)

-- Theorem 1: A line is perpendicular to a plane iff it's perpendicular to every line in the plane
theorem perpendicular_to_plane_iff_perpendicular_to_all_lines 
  (l : Line) (p : Plane) :
  perpendicular_to_plane l p ↔ 
  ∀ (m : Line), in_plane m p → perpendicular l m :=
sorry

-- Theorem 2: If a is parallel to b, and l is perpendicular to a, then l is perpendicular to b
theorem perpendicular_parallel_transitive 
  (a b l : Line) :
  parallel a b → perpendicular l a → perpendicular l b :=
sorry

end NUMINAMATH_CALUDE_perpendicular_to_plane_iff_perpendicular_to_all_lines_perpendicular_parallel_transitive_l991_99111


namespace NUMINAMATH_CALUDE_manuscript_cost_calculation_l991_99130

/-- Calculates the total cost of typing a manuscript with given conditions -/
def manuscript_typing_cost (total_pages : ℕ) (first_typing_rate : ℕ) (revision_rate : ℕ) 
  (pages_revised_once : ℕ) (pages_revised_twice : ℕ) : ℕ :=
  let pages_not_revised := total_pages - pages_revised_once - pages_revised_twice
  let initial_typing_cost := total_pages * first_typing_rate
  let first_revision_cost := pages_revised_once * revision_rate
  let second_revision_cost := pages_revised_twice * revision_rate * 2
  initial_typing_cost + first_revision_cost + second_revision_cost

theorem manuscript_cost_calculation :
  manuscript_typing_cost 100 10 5 30 20 = 1350 := by
  sorry

end NUMINAMATH_CALUDE_manuscript_cost_calculation_l991_99130


namespace NUMINAMATH_CALUDE_interest_rate_first_part_l991_99187

/-- Given a total sum and a second part, calculates the first part. -/
def firstPart (total second : ℝ) : ℝ := total - second

/-- Calculates simple interest. -/
def simpleInterest (principal rate time : ℝ) : ℝ := principal * rate * time

/-- Theorem stating the interest rate for the first part is 3% per annum. -/
theorem interest_rate_first_part (total second : ℝ) (h1 : total = 2704) (h2 : second = 1664) :
  let first := firstPart total second
  let rate2 := 0.05
  let time1 := 8
  let time2 := 3
  simpleInterest first ((3 : ℝ) / 100) time1 = simpleInterest second rate2 time2 := by
  sorry

#check interest_rate_first_part

end NUMINAMATH_CALUDE_interest_rate_first_part_l991_99187


namespace NUMINAMATH_CALUDE_brandy_energy_drinks_l991_99148

/-- The number of energy drinks Brandy drank -/
def num_drinks : ℕ := 4

/-- The maximum safe amount of caffeine per day in mg -/
def max_caffeine : ℕ := 500

/-- The amount of caffeine in each energy drink in mg -/
def caffeine_per_drink : ℕ := 120

/-- The amount of additional caffeine Brandy can safely consume after drinking the energy drinks in mg -/
def remaining_caffeine : ℕ := 20

theorem brandy_energy_drinks :
  num_drinks * caffeine_per_drink + remaining_caffeine = max_caffeine :=
sorry

end NUMINAMATH_CALUDE_brandy_energy_drinks_l991_99148


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l991_99119

/-- An arithmetic sequence. -/
def arithmetic_sequence (a : ℕ → ℚ) : Prop :=
  ∃ d : ℚ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_sum (a : ℕ → ℚ) :
  arithmetic_sequence a →
  a 2 + a 4 + a 5 + a 6 + a 8 = 25 →
  a 2 + a 8 = 10 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l991_99119


namespace NUMINAMATH_CALUDE_passing_percentage_l991_99168

def total_marks : ℕ := 400
def student_marks : ℕ := 150
def failed_by : ℕ := 30

theorem passing_percentage : 
  (((student_marks + failed_by : ℚ) / total_marks) * 100 = 45) := by sorry

end NUMINAMATH_CALUDE_passing_percentage_l991_99168


namespace NUMINAMATH_CALUDE_jerry_firecrackers_l991_99170

theorem jerry_firecrackers (F : ℕ) : 
  (F ≥ 12) →
  (5 * (F - 12) / 6 = 30) →
  F = 48 := by sorry

end NUMINAMATH_CALUDE_jerry_firecrackers_l991_99170


namespace NUMINAMATH_CALUDE_surrounding_circles_radius_l991_99113

theorem surrounding_circles_radius (r : ℝ) : 
  (∃ (A B C D : ℝ × ℝ),
    -- Define the centers of the four surrounding circles
    (A.1 - B.1)^2 + (A.2 - B.2)^2 = (2*r)^2 ∧
    (B.1 - C.1)^2 + (B.2 - C.2)^2 = (2*r)^2 ∧
    (C.1 - D.1)^2 + (C.2 - D.2)^2 = (2*r)^2 ∧
    (D.1 - A.1)^2 + (D.2 - A.2)^2 = (2*r)^2 ∧
    -- Ensure the surrounding circles touch the central circle
    (A.1^2 + A.2^2 = (r + 2)^2) ∧
    (B.1^2 + B.2^2 = (r + 2)^2) ∧
    (C.1^2 + C.2^2 = (r + 2)^2) ∧
    (D.1^2 + D.2^2 = (r + 2)^2) ∧
    -- Ensure the surrounding circles touch each other
    (A.1 - B.1)^2 + (A.2 - B.2)^2 = (2*r)^2 ∧
    (B.1 - C.1)^2 + (B.2 - C.2)^2 = (2*r)^2 ∧
    (C.1 - D.1)^2 + (C.2 - D.2)^2 = (2*r)^2 ∧
    (D.1 - A.1)^2 + (D.2 - A.2)^2 = (2*r)^2) →
  r = 2 + Real.sqrt 2 := by
sorry

end NUMINAMATH_CALUDE_surrounding_circles_radius_l991_99113


namespace NUMINAMATH_CALUDE_rope_jumps_percentage_l991_99166

def rope_jumps : List ℕ := [50, 77, 83, 91, 93, 101, 87, 102, 111, 63, 117, 89, 121, 130, 133, 146, 88, 158, 177, 188]

def total_students : ℕ := 20

def in_range (x : ℕ) : Bool := 80 ≤ x ∧ x ≤ 100

def count_in_range (l : List ℕ) : ℕ := (l.filter in_range).length

theorem rope_jumps_percentage :
  (count_in_range rope_jumps : ℚ) / total_students * 100 = 30 := by
  sorry

end NUMINAMATH_CALUDE_rope_jumps_percentage_l991_99166


namespace NUMINAMATH_CALUDE_kenneth_theorem_l991_99125

def kenneth_problem (earnings : ℝ) (joystick_percentage : ℝ) : Prop :=
  let joystick_cost := earnings * (joystick_percentage / 100)
  let remaining := earnings - joystick_cost
  earnings = 450 ∧ joystick_percentage = 10 → remaining = 405

theorem kenneth_theorem : kenneth_problem 450 10 := by
  sorry

end NUMINAMATH_CALUDE_kenneth_theorem_l991_99125


namespace NUMINAMATH_CALUDE_complement_of_A_l991_99167

-- Define the universal set U
def U : Finset ℕ := {1,2,3,4,5,6,7}

-- Define set A
def A : Finset ℕ := Finset.filter (fun x => 1 ≤ x ∧ x ≤ 6) U

-- Theorem statement
theorem complement_of_A : (U \ A) = {7} := by sorry

end NUMINAMATH_CALUDE_complement_of_A_l991_99167


namespace NUMINAMATH_CALUDE_zoo_count_l991_99108

theorem zoo_count (total_heads : ℕ) (total_legs : ℕ) : 
  total_heads = 300 → 
  total_legs = 710 → 
  ∃ (birds mammals unique : ℕ), 
    birds + mammals + unique = total_heads ∧
    2 * birds + 4 * mammals + 3 * unique = total_legs ∧
    birds = 230 := by
  sorry

end NUMINAMATH_CALUDE_zoo_count_l991_99108


namespace NUMINAMATH_CALUDE_right_triangle_perimeter_l991_99121

theorem right_triangle_perimeter : ∀ (a b c : ℕ),
  a > 0 → b > 0 → c > 0 →
  a = 11 →
  a * a + b * b = c * c →
  a + b + c = 132 :=
by
  sorry

end NUMINAMATH_CALUDE_right_triangle_perimeter_l991_99121


namespace NUMINAMATH_CALUDE_election_percentage_l991_99145

theorem election_percentage (total_votes : ℕ) (winning_margin : ℕ) (winning_percentage : ℚ) : 
  total_votes = 7520 →
  winning_margin = 1504 →
  winning_percentage = 60 →
  (winning_percentage / 100) * total_votes - (total_votes - (winning_percentage / 100) * total_votes) = winning_margin :=
by sorry

end NUMINAMATH_CALUDE_election_percentage_l991_99145


namespace NUMINAMATH_CALUDE_correct_ages_l991_99175

/-- Represents the ages of Albert, Mary, Betty, and Carol -/
structure Ages where
  albert : ℕ
  mary : ℕ
  betty : ℕ
  carol : ℕ

/-- The conditions given in the problem -/
def satisfiesConditions (ages : Ages) : Prop :=
  ages.albert = 2 * ages.mary ∧
  ages.albert = 4 * ages.betty ∧
  ages.mary = ages.albert - 10 ∧
  ages.carol = ages.betty + 3 ∧
  ages.carol = ages.mary / 2

/-- The theorem to prove -/
theorem correct_ages :
  ∃ (ages : Ages), satisfiesConditions ages ∧
    ages.albert = 20 ∧
    ages.mary = 10 ∧
    ages.betty = 2 ∧
    ages.carol = 5 := by
  sorry

end NUMINAMATH_CALUDE_correct_ages_l991_99175


namespace NUMINAMATH_CALUDE_percent_less_problem_l991_99140

theorem percent_less_problem (w x y z : ℝ) : 
  x = y * (1 - z / 100) →
  y = 1.4 * w →
  x = 5 * w / 4 →
  z = 10.71 := by
sorry

end NUMINAMATH_CALUDE_percent_less_problem_l991_99140


namespace NUMINAMATH_CALUDE_incorrect_calculation_l991_99107

theorem incorrect_calculation (a : ℝ) (n : ℕ) : 
  a^(2*n) * (a^(2*n))^3 / a^(4*n) ≠ a^2 :=
sorry

end NUMINAMATH_CALUDE_incorrect_calculation_l991_99107


namespace NUMINAMATH_CALUDE_problem_solution_l991_99174

theorem problem_solution : 
  let A : ℤ := -5 * -3
  let B : ℤ := 2 - 2
  A + B = 15 := by sorry

end NUMINAMATH_CALUDE_problem_solution_l991_99174


namespace NUMINAMATH_CALUDE_arithmetic_progression_relationship_l991_99154

theorem arithmetic_progression_relationship (x y z d : ℝ) : 
  (x + (y - z) ≠ y + (z - x) ∧ 
   y + (z - x) ≠ z + (x - y) ∧ 
   x + (y - z) ≠ z + (x - y)) →
  (x + (y - z) ≠ 0 ∧ y + (z - x) ≠ 0 ∧ z + (x - y) ≠ 0) →
  (y + (z - x)) - (x + (y - z)) = d →
  (z + (x - y)) - (y + (z - x)) = d →
  (x = y + d / 2 ∧ z = y + d) :=
by sorry

end NUMINAMATH_CALUDE_arithmetic_progression_relationship_l991_99154


namespace NUMINAMATH_CALUDE_perpendicular_line_to_plane_perpendicular_planes_l991_99192

-- Define the types for our geometric objects
variable (Point Line Plane : Type)

-- Define the relationships between geometric objects
variable (contains : Plane → Line → Prop)
variable (perpendicular : Line → Line → Prop)
variable (perpendicularLP : Line → Plane → Prop)
variable (perpendicularPP : Plane → Plane → Prop)
variable (intersect : Line → Line → Prop)

-- Statement 1
theorem perpendicular_line_to_plane 
  (α : Plane) (a l₁ l₂ : Line) :
  contains α l₁ → 
  contains α l₂ → 
  intersect l₁ l₂ → 
  perpendicular a l₁ → 
  perpendicular a l₂ → 
  perpendicularLP a α :=
sorry

-- Statement 6
theorem perpendicular_planes 
  (α β : Plane) (b : Line) :
  contains β b → 
  perpendicularLP b α → 
  perpendicularPP β α :=
sorry

end NUMINAMATH_CALUDE_perpendicular_line_to_plane_perpendicular_planes_l991_99192


namespace NUMINAMATH_CALUDE_common_factor_proof_l991_99182

variables (a b c : ℕ+)

theorem common_factor_proof : Nat.gcd (4 * a^2 * b^2 * c) (6 * a * b^3) = 2 * a * b^2 := by
  sorry

end NUMINAMATH_CALUDE_common_factor_proof_l991_99182


namespace NUMINAMATH_CALUDE_windshield_wiper_area_l991_99152

/-- The area swept by two semicircular windshield wipers -/
theorem windshield_wiper_area (L : ℝ) (h : L > 0) :
  let area := (2 / 3 * π + Real.sqrt 3 / 4) * L^2
  area = (π * L^2) - ((1 / 3 * π - Real.sqrt 3 / 4) * L^2) :=
by sorry

end NUMINAMATH_CALUDE_windshield_wiper_area_l991_99152


namespace NUMINAMATH_CALUDE_movie_theater_deal_l991_99128

/-- Movie theater deal problem -/
theorem movie_theater_deal (deal_price : ℝ) (ticket_price : ℝ) (savings : ℝ)
  (h1 : deal_price = 20)
  (h2 : ticket_price = 8)
  (h3 : savings = 2) :
  let popcorn_price := ticket_price - 3
  let total_normal_price := deal_price + savings
  let drink_price := (total_normal_price - ticket_price - popcorn_price) * (2/3)
  drink_price - popcorn_price = 1 := by sorry

end NUMINAMATH_CALUDE_movie_theater_deal_l991_99128


namespace NUMINAMATH_CALUDE_last_two_digits_sum_l991_99186

theorem last_two_digits_sum (n : ℕ) : (7^30 + 13^30) % 100 = 0 := by
  sorry

end NUMINAMATH_CALUDE_last_two_digits_sum_l991_99186


namespace NUMINAMATH_CALUDE_stating_children_count_l991_99198

/-- Represents the number of children in the problem -/
def num_children : ℕ := 6

/-- Represents the age of the youngest child -/
def youngest_age : ℕ := 7

/-- Represents the interval between children's ages -/
def age_interval : ℕ := 3

/-- Represents the sum of all children's ages -/
def total_age : ℕ := 65

/-- 
  Theorem stating that given the conditions of the problem,
  the number of children is 6
-/
theorem children_count : 
  (∃ (n : ℕ), 
    n * (2 * youngest_age + (n - 1) * age_interval) = 2 * total_age ∧
    n = num_children) :=
by sorry

end NUMINAMATH_CALUDE_stating_children_count_l991_99198


namespace NUMINAMATH_CALUDE_range_of_m_l991_99161

/-- Proposition p: x < -2 or x > 10 -/
def p (x : ℝ) : Prop := x < -2 ∨ x > 10

/-- Proposition q: 1-m ≤ x ≤ 1+m^2 -/
def q (x m : ℝ) : Prop := 1 - m ≤ x ∧ x ≤ 1 + m^2

/-- ¬p is a sufficient but not necessary condition for q -/
def suff_not_nec (m : ℝ) : Prop :=
  (∀ x, ¬(p x) → q x m) ∧ ∃ x, q x m ∧ p x

theorem range_of_m :
  {m : ℝ | suff_not_nec m} = {m : ℝ | m ≥ 3} :=
sorry

end NUMINAMATH_CALUDE_range_of_m_l991_99161


namespace NUMINAMATH_CALUDE_largest_product_of_three_l991_99188

def S : Finset Int := {-4, -3, -1, 3, 5}

theorem largest_product_of_three (a b c : Int) 
  (ha : a ∈ S) (hb : b ∈ S) (hc : c ∈ S) 
  (hdistinct : a ≠ b ∧ b ≠ c ∧ a ≠ c) : 
  (∀ x y z : Int, x ∈ S → y ∈ S → z ∈ S → 
    x ≠ y ∧ y ≠ z ∧ x ≠ z → 
    x * y * z ≤ 60) ∧ 
  (∃ x y z : Int, x ∈ S ∧ y ∈ S ∧ z ∈ S ∧ 
    x ≠ y ∧ y ≠ z ∧ x ≠ z ∧ 
    x * y * z = 60) :=
by
  sorry

end NUMINAMATH_CALUDE_largest_product_of_three_l991_99188


namespace NUMINAMATH_CALUDE_count_parallelograms_l991_99178

/-- Represents a point in 2D space -/
structure Point where
  x : ℤ
  y : ℤ

/-- Represents a parallelogram with vertices P, Q, R, S -/
structure Parallelogram where
  P : Point
  Q : Point
  R : Point
  S : Point

/-- Calculates the area of a parallelogram using the shoelace formula -/
def area (p : Parallelogram) : ℚ :=
  (1 / 2 : ℚ) * |p.Q.x * p.S.y - p.S.x * p.Q.y|

/-- Checks if a point is in the first quadrant -/
def isFirstQuadrant (p : Point) : Prop :=
  p.x > 0 ∧ p.y > 0

/-- Checks if a point is on the line y = mx -/
def isOnLine (p : Point) (m : ℤ) : Prop :=
  p.y = m * p.x

/-- The main theorem to be proved -/
theorem count_parallelograms :
  let validParallelogram (p : Parallelogram) : Prop :=
    p.P = ⟨0, 0⟩ ∧
    isFirstQuadrant p.Q ∧
    isFirstQuadrant p.R ∧
    isFirstQuadrant p.S ∧
    isOnLine p.Q 2 ∧
    isOnLine p.S 3 ∧
    area p = 2000000
  (parallelograms : Finset Parallelogram) →
  (∀ p ∈ parallelograms, validParallelogram p) →
  parallelograms.card = 196 :=
sorry

end NUMINAMATH_CALUDE_count_parallelograms_l991_99178


namespace NUMINAMATH_CALUDE_okeydokey_receives_25_earthworms_l991_99184

/-- The number of apples Okeydokey paid -/
def okeydokey_apples : ℕ := 5

/-- The number of apples Artichokey paid -/
def artichokey_apples : ℕ := 7

/-- The total number of earthworms in the box -/
def total_earthworms : ℕ := 60

/-- Calculate the number of earthworms Okeydokey should receive -/
def okeydokey_earthworms : ℕ :=
  (okeydokey_apples * total_earthworms) / (okeydokey_apples + artichokey_apples)

/-- Theorem stating that Okeydokey should receive 25 earthworms -/
theorem okeydokey_receives_25_earthworms :
  okeydokey_earthworms = 25 := by
  sorry

end NUMINAMATH_CALUDE_okeydokey_receives_25_earthworms_l991_99184


namespace NUMINAMATH_CALUDE_base_b_is_eight_l991_99141

/-- Given that in base b, the square of 13_b is 211_b, prove that b = 8 -/
theorem base_b_is_eight (b : ℕ) (h : b > 1) :
  (1 * b + 3)^2 = 2 * b^2 + 1 * b + 1 → b = 8 := by
  sorry

end NUMINAMATH_CALUDE_base_b_is_eight_l991_99141


namespace NUMINAMATH_CALUDE_polygon_problem_l991_99129

/-- Represents a polygon with a given number of sides -/
structure Polygon where
  sides : ℕ

/-- The sum of interior angles of a polygon -/
def interiorAngleSum (p : Polygon) : ℕ := 180 * (p.sides - 2)

/-- The number of diagonals in a polygon -/
def diagonalCount (p : Polygon) : ℕ := p.sides * (p.sides - 3) / 2

theorem polygon_problem (x y : Polygon) 
  (h1 : interiorAngleSum x + interiorAngleSum y = 1440)
  (h2 : x.sides * 3 = y.sides) :
  720 = 360 + 360 ∧ 
  x.sides = 3 ∧ 
  y.sides = 9 ∧ 
  diagonalCount y = 27 := by
  sorry

end NUMINAMATH_CALUDE_polygon_problem_l991_99129


namespace NUMINAMATH_CALUDE_acute_angle_theorem_l991_99126

def is_acute_angle (θ : ℝ) : Prop := 0 < θ ∧ θ < 90

theorem acute_angle_theorem (θ : ℝ) (h1 : is_acute_angle θ) 
  (h2 : 4 * (90 - θ) = (180 - θ) + 60) : θ = 40 := by
  sorry

end NUMINAMATH_CALUDE_acute_angle_theorem_l991_99126


namespace NUMINAMATH_CALUDE_sqrt_sum_comparison_l991_99153

theorem sqrt_sum_comparison : Real.sqrt 3 + Real.sqrt 5 > Real.sqrt 2 + Real.sqrt 6 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_sum_comparison_l991_99153


namespace NUMINAMATH_CALUDE_distance_knoxville_to_los_angeles_l991_99117

/-- A point on the complex plane representing a city --/
structure City where
  coord : ℂ

/-- A map of cities on the complex plane that preserves distances --/
structure CityMap where
  los_angeles : City
  boston : City
  knoxville : City
  preserves_distances : True

theorem distance_knoxville_to_los_angeles (map : CityMap)
  (h1 : map.los_angeles.coord = 0)
  (h2 : map.boston.coord = 2600 * I)
  (h3 : map.knoxville.coord = 780 + 1040 * I) :
  Complex.abs (map.knoxville.coord - map.los_angeles.coord) = 1300 := by
  sorry

end NUMINAMATH_CALUDE_distance_knoxville_to_los_angeles_l991_99117


namespace NUMINAMATH_CALUDE_quadratic_root_problem_l991_99112

theorem quadratic_root_problem (k : ℝ) :
  (∃ x : ℝ, x^2 + k*x - 2 = 0 ∧ x = -2) →
  (∃ y : ℝ, y^2 + k*y - 2 = 0 ∧ y = 1) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_root_problem_l991_99112


namespace NUMINAMATH_CALUDE_mint_code_is_6785_l991_99138

-- Define a function that maps characters to digits based on their position in GREAT MIND
def code_to_digit (c : Char) : Nat :=
  match c with
  | 'G' => 1
  | 'R' => 2
  | 'E' => 3
  | 'A' => 4
  | 'T' => 5
  | 'M' => 6
  | 'I' => 7
  | 'N' => 8
  | 'D' => 9
  | _ => 0

-- Define a function that converts a string to a number using the code
def code_to_number (s : String) : Nat :=
  s.foldl (fun acc c => acc * 10 + code_to_digit c) 0

-- Theorem stating that MINT represents 6785
theorem mint_code_is_6785 : code_to_number "MINT" = 6785 := by
  sorry

end NUMINAMATH_CALUDE_mint_code_is_6785_l991_99138


namespace NUMINAMATH_CALUDE_circle_chords_and_triangles_l991_99101

/-- Given 10 points on the circumference of a circle, prove the number of chords and triangles -/
theorem circle_chords_and_triangles (n : ℕ) (hn : n = 10) :
  (Nat.choose n 2 = 45) ∧ (Nat.choose n 3 = 120) := by
  sorry

#check circle_chords_and_triangles

end NUMINAMATH_CALUDE_circle_chords_and_triangles_l991_99101


namespace NUMINAMATH_CALUDE_students_on_korabelnaya_street_l991_99120

theorem students_on_korabelnaya_street (n : ℕ) : 
  n < 50 → 
  n % 7 = 0 → 
  n % 3 = 0 → 
  n % 2 = 0 → 
  n / 7 + n / 3 + n / 2 < n → 
  n - (n / 7 + n / 3 + n / 2) = 1 := by
sorry

end NUMINAMATH_CALUDE_students_on_korabelnaya_street_l991_99120


namespace NUMINAMATH_CALUDE_coefficient_x4_in_expansion_l991_99199

theorem coefficient_x4_in_expansion (x : ℝ) : 
  (Finset.range 9).sum (λ k => (Nat.choose 8 k : ℝ) * x^k * (2 * Real.sqrt 3)^(8 - k)) = 
  10080 * x^4 + (Finset.range 9).sum (λ k => if k ≠ 4 then (Nat.choose 8 k : ℝ) * x^k * (2 * Real.sqrt 3)^(8 - k) else 0) := by
sorry

end NUMINAMATH_CALUDE_coefficient_x4_in_expansion_l991_99199


namespace NUMINAMATH_CALUDE_divisibility_of_squares_sum_l991_99164

theorem divisibility_of_squares_sum (p x y z : ℕ) : 
  Prime p → 
  0 < x → x < y → y < z → z < p → 
  x^3 % p = y^3 % p → y^3 % p = z^3 % p →
  (x^2 + y^2 + z^2) % (x + y + z) = 0 :=
by sorry

end NUMINAMATH_CALUDE_divisibility_of_squares_sum_l991_99164


namespace NUMINAMATH_CALUDE_downstream_distance_is_24km_l991_99139

/-- Represents the swimming scenario with given conditions -/
structure SwimmingScenario where
  upstream_distance : ℝ
  upstream_time : ℝ
  downstream_time : ℝ
  still_water_speed : ℝ

/-- Calculates the downstream distance given a swimming scenario -/
def downstream_distance (scenario : SwimmingScenario) : ℝ :=
  sorry

/-- Theorem stating that under the given conditions, the downstream distance is 24 km -/
theorem downstream_distance_is_24km 
  (scenario : SwimmingScenario)
  (h1 : scenario.upstream_distance = 12)
  (h2 : scenario.upstream_time = 6)
  (h3 : scenario.downstream_time = 6)
  (h4 : scenario.still_water_speed = 3) :
  downstream_distance scenario = 24 :=
sorry

end NUMINAMATH_CALUDE_downstream_distance_is_24km_l991_99139


namespace NUMINAMATH_CALUDE_parabola_tangent_hyperbola_l991_99142

/-- A parabola is tangent to a hyperbola if and only if m is 4 or 8 -/
theorem parabola_tangent_hyperbola :
  ∀ m : ℝ,
  (∀ x y : ℝ, y = x^2 + 3 ∧ y^2 - m*x^2 = 4 →
    ∃! p : ℝ × ℝ, p.1 = x ∧ p.2 = y) ↔ (m = 4 ∨ m = 8) := by
  sorry

end NUMINAMATH_CALUDE_parabola_tangent_hyperbola_l991_99142
