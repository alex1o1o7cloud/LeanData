import Mathlib

namespace degree_of_specific_monomial_l4016_401682

/-- The degree of a monomial is the sum of the exponents of its variables -/
def degree_of_monomial (m : ℕ) (n : ℕ) : ℕ := m + n

/-- The degree of the monomial (1/7)mn^2 is 3 -/
theorem degree_of_specific_monomial : degree_of_monomial 1 2 = 3 := by
  sorry

end degree_of_specific_monomial_l4016_401682


namespace paper_pieces_sum_l4016_401677

/-- The number of pieces of paper picked up by Olivia and Edward -/
theorem paper_pieces_sum (olivia_pieces edward_pieces : ℕ) 
  (h_olivia : olivia_pieces = 16) 
  (h_edward : edward_pieces = 3) : 
  olivia_pieces + edward_pieces = 19 := by
  sorry

end paper_pieces_sum_l4016_401677


namespace solution_theorem_l4016_401621

-- Define the function f(x)
def f (a : ℝ) (x : ℝ) : ℝ := a * x^2 - 3 * a * x + a^2 - 3

theorem solution_theorem :
  -- Part I
  (∃ (l b : ℝ), ∀ x, f 3 x < 0 ↔ l < x ∧ x < b) ∧
  (∃ (l : ℝ), ∀ x, f 3 x < 0 ↔ l < x ∧ x < 2) ∧
  -- Part II
  (∀ a : ℝ, a < 0 → (∀ x, -3 ≤ x ∧ x ≤ 3 → f a x < 4) → -7/4 < a) ∧
  (∀ a : ℝ, a < 0 → (∀ x, -3 ≤ x ∧ x ≤ 3 → f a x < 4) → a < 0) :=
by sorry

end solution_theorem_l4016_401621


namespace coin_counting_machine_result_l4016_401686

def coin_value (coin_type : String) : ℚ :=
  match coin_type with
  | "quarter" => 25 / 100
  | "dime" => 10 / 100
  | "nickel" => 5 / 100
  | "penny" => 1 / 100
  | _ => 0

def total_value (quarters dimes nickels pennies : ℕ) : ℚ :=
  quarters * coin_value "quarter" +
  dimes * coin_value "dime" +
  nickels * coin_value "nickel" +
  pennies * coin_value "penny"

def fee_percentage : ℚ := 10 / 100

theorem coin_counting_machine_result 
  (quarters dimes nickels pennies : ℕ) : 
  quarters = 76 → dimes = 85 → nickels = 20 → pennies = 150 →
  (total_value quarters dimes nickels pennies) * (1 - fee_percentage) = 27 :=
by
  sorry

end coin_counting_machine_result_l4016_401686


namespace water_mixture_percentage_l4016_401601

theorem water_mixture_percentage (initial_volume : ℝ) (added_water : ℝ) (final_water_percentage : ℝ) :
  initial_volume = 150 ∧
  added_water = 10 ∧
  final_water_percentage = 25 →
  (initial_volume * (20 / 100) + added_water) / (initial_volume + added_water) = final_water_percentage / 100 :=
by sorry

end water_mixture_percentage_l4016_401601


namespace half_sum_of_odd_squares_is_sum_of_squares_l4016_401681

theorem half_sum_of_odd_squares_is_sum_of_squares (a b : ℕ) (ha : Odd a) (hb : Odd b) (hab : a ≠ b) :
  ∃ x y : ℕ, (a^2 + b^2) / 2 = x^2 + y^2 := by
  sorry

end half_sum_of_odd_squares_is_sum_of_squares_l4016_401681


namespace book_pages_from_digits_l4016_401619

/-- Given a book with pages numbered consecutively starting from 1,
    this function calculates the total number of digits used to number all pages. -/
def totalDigits (n : ℕ) : ℕ :=
  let oneDigit := min n 9
  let twoDigit := max 0 (min n 99 - 9)
  let threeDigit := max 0 (n - 99)
  oneDigit + 2 * twoDigit + 3 * threeDigit

/-- Theorem stating that a book with 672 digits used for page numbering has 260 pages. -/
theorem book_pages_from_digits :
  ∃ (n : ℕ), totalDigits n = 672 ∧ n = 260 := by
  sorry

end book_pages_from_digits_l4016_401619


namespace union_complement_equal_to_set_l4016_401659

def U : Finset Nat := {0, 1, 2, 4, 6, 8}
def M : Finset Nat := {0, 4, 6}
def N : Finset Nat := {0, 1, 6}

theorem union_complement_equal_to_set :
  M ∪ (U \ N) = {0, 2, 4, 6, 8} := by sorry

end union_complement_equal_to_set_l4016_401659


namespace chess_team_selection_ways_l4016_401626

def total_members : ℕ := 18
def num_siblings : ℕ := 4
def team_size : ℕ := 8
def max_siblings_in_team : ℕ := 2

theorem chess_team_selection_ways :
  (Nat.choose total_members team_size) -
  (Nat.choose num_siblings (num_siblings) * Nat.choose (total_members - num_siblings) (team_size - num_siblings)) = 42757 := by
  sorry

end chess_team_selection_ways_l4016_401626


namespace certain_number_proof_l4016_401695

theorem certain_number_proof (p q : ℚ) 
  (h1 : 3 / p = 8)
  (h2 : 3 / q = 18)
  (h3 : p - q = 0.20833333333333334) : 
  q = 1 / 6 := by
  sorry

end certain_number_proof_l4016_401695


namespace right_triangle_arctan_sum_l4016_401606

theorem right_triangle_arctan_sum (a b c : ℝ) (h : a^2 + b^2 = c^2) :
  Real.arctan (a / (b + c)) + Real.arctan (b / (a + c)) + Real.arctan (c / (a + b)) = 0 := by
  sorry

end right_triangle_arctan_sum_l4016_401606


namespace percentage_problem_l4016_401693

/-- Prove that the percentage is 50% given the conditions -/
theorem percentage_problem (x : ℝ) (a : ℝ) : 
  (x / 100) * a = 95 → a = 190 → x = 50 := by
  sorry

end percentage_problem_l4016_401693


namespace complex_real_condition_l4016_401671

theorem complex_real_condition (a : ℝ) : 
  (Complex.I : ℂ)^2 = -1 →
  (↑(1 : ℝ) - Complex.I) * (↑a + Complex.I) ∈ Set.range (Complex.ofReal) →
  a = 1 := by
sorry

end complex_real_condition_l4016_401671


namespace sum_of_reciprocal_relations_l4016_401627

theorem sum_of_reciprocal_relations (x y : ℝ) 
  (h1 : 1/x + 1/y = 4) 
  (h2 : 1/x - 1/y = -6) : 
  x + y = -4/5 := by
sorry

end sum_of_reciprocal_relations_l4016_401627


namespace A_subset_of_neg_one_one_l4016_401657

def A : Set ℝ := {x | x^2 - 1 = 0}

theorem A_subset_of_neg_one_one : A ⊆ {-1, 1} := by
  sorry

end A_subset_of_neg_one_one_l4016_401657


namespace tangent_line_condition_f_positive_condition_three_zeros_condition_l4016_401631

noncomputable section

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := Real.log x - a * x + a / x

-- Define the derivative of f
def f_deriv (a : ℝ) (x : ℝ) : ℝ := 1 / x - a * (1 + 1 / x^2)

theorem tangent_line_condition (a : ℝ) :
  (f_deriv a 1 = (4 - f a 1) / 2) → a = -1/2 := by sorry

theorem f_positive_condition (a : ℝ) :
  0 < a → a < 1 → f a (a^2 / 2) > 0 := by sorry

theorem three_zeros_condition (a : ℝ) :
  (∃ x y z : ℝ, x ≠ y ∧ y ≠ z ∧ x ≠ z ∧ f a x = 0 ∧ f a y = 0 ∧ f a z = 0) ↔
  (0 < a ∧ a < 1/2) := by sorry

end

end tangent_line_condition_f_positive_condition_three_zeros_condition_l4016_401631


namespace expression_simplification_l4016_401674

theorem expression_simplification (x : ℤ) 
  (h1 : 2 * (x - 1) < x + 1) 
  (h2 : 5 * x + 3 ≥ 2 * x) : 
  (2 : ℚ) / (x^2 + x) / (1 - (x - 1) / (x^2 - 1)) = 1/2 :=
by sorry

end expression_simplification_l4016_401674


namespace sqrt_3_irrational_l4016_401644

theorem sqrt_3_irrational :
  ∀ (a b c : ℚ), (a = 1/2 ∧ b = 1/5 ∧ c = -5) →
  ¬ ∃ (p q : ℤ), q ≠ 0 ∧ Real.sqrt 3 = p / q := by
  sorry

end sqrt_3_irrational_l4016_401644


namespace bench_and_student_count_l4016_401694

theorem bench_and_student_count :
  ∃ (a b s : ℕ), 
    (s = a * b + 5 ∧ s = 8 * b - 4) →
    ((b = 9 ∧ s = 68) ∨ (b = 3 ∧ s = 20)) := by
  sorry

end bench_and_student_count_l4016_401694


namespace m_value_proof_l4016_401641

theorem m_value_proof (a b c d e f : ℝ) (m n : ℝ) 
  (h1 : a > 0) (h2 : b > 0) (h3 : c > 0) (h4 : d > 0) (h5 : e > 0) (h6 : f > 0)
  (h7 : a^m = b^3)
  (h8 : c^n = d^2)
  (h9 : ((a^(m+1))/(b^(m+1))) * ((c^n)/(d^n)) = 1/(2*(e^(35*f)))) :
  m = 3 := by sorry

end m_value_proof_l4016_401641


namespace min_value_theorem_l4016_401655

theorem min_value_theorem (a b : ℝ) (h : a * b > 0) :
  a^2 + 4*b^2 + 1/(a*b) ≥ 4 ∧
  (a^2 + 4*b^2 + 1/(a*b) = 4 ↔ a = 1/Real.rpow 2 (1/4) ∧ b = 1/Real.rpow 2 (1/4)) :=
sorry

end min_value_theorem_l4016_401655


namespace platform_length_l4016_401634

/-- Given a train of length 600 m that crosses a platform in 39 seconds
    and a signal pole in 18 seconds, the length of the platform is 700 m. -/
theorem platform_length
  (train_length : ℝ)
  (time_platform : ℝ)
  (time_pole : ℝ)
  (h1 : train_length = 600)
  (h2 : time_platform = 39)
  (h3 : time_pole = 18) :
  (train_length * time_platform / time_pole) - train_length = 700 :=
by sorry

#check platform_length

end platform_length_l4016_401634


namespace quadratic_max_value_l4016_401617

/-- Given a quadratic function y = x² + 2x - 2, prove that if the maximum value of y is 1
    when a ≤ x ≤ 1/2, then a = -3. -/
theorem quadratic_max_value (y : ℝ → ℝ) (a : ℝ) :
  (∀ x, y x = x^2 + 2*x - 2) →
  (∀ x, a ≤ x → x ≤ 1/2 → y x ≤ 1) →
  (∃ x, a ≤ x ∧ x ≤ 1/2 ∧ y x = 1) →
  a = -3 :=
sorry

end quadratic_max_value_l4016_401617


namespace max_sum_of_factors_l4016_401653

theorem max_sum_of_factors (a b c d : ℕ+) : 
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d →
  a * b * c * d = 1995 →
  ∀ w x y z : ℕ+, w ≠ x ∧ w ≠ y ∧ w ≠ z ∧ x ≠ y ∧ x ≠ z ∧ y ≠ z →
    w * x * y * z = 1995 →
    a + b + c + d ≥ w + x + y + z :=
by
  sorry

end max_sum_of_factors_l4016_401653


namespace power_of_power_l4016_401636

theorem power_of_power : (3^3)^4 = 531441 := by
  sorry

end power_of_power_l4016_401636


namespace lisa_window_width_l4016_401608

/-- Represents the dimensions of a window pane -/
structure Pane where
  width : ℝ
  height : ℝ

/-- Represents the dimensions and layout of a window -/
structure Window where
  pane : Pane
  rows : ℕ
  columns : ℕ
  border_width : ℝ

/-- Calculates the total width of the window -/
def window_width (w : Window) : ℝ :=
  w.columns * w.pane.width + (w.columns + 1) * w.border_width

/-- The main theorem stating the width of Lisa's window -/
theorem lisa_window_width :
  ∃ (w : Window),
    w.rows = 3 ∧
    w.columns = 4 ∧
    w.border_width = 3 ∧
    w.pane.width / w.pane.height = 3 / 4 ∧
    window_width w = 51 :=
sorry

end lisa_window_width_l4016_401608


namespace intersecting_lines_sum_l4016_401618

/-- Two lines intersecting at a point implies a specific sum of their slopes and y-intercepts -/
theorem intersecting_lines_sum (m b : ℚ) : 
  (∀ x y : ℚ, y = m * x + 2 → y = 4 * x + b → x = 4 ∧ y = 8) → 
  b + m = -13/2 := by
  sorry

end intersecting_lines_sum_l4016_401618


namespace binomial_600_600_eq_1_l4016_401649

theorem binomial_600_600_eq_1 : Nat.choose 600 600 = 1 := by
  sorry

end binomial_600_600_eq_1_l4016_401649


namespace sum_of_numbers_greater_than_04_l4016_401612

def numbers : List ℚ := [0.8, 1/2, 0.3, 1/3]

theorem sum_of_numbers_greater_than_04 : 
  (numbers.filter (λ x => x > 0.4)).sum = 1.3 := by
  sorry

end sum_of_numbers_greater_than_04_l4016_401612


namespace orange_tree_problem_l4016_401690

theorem orange_tree_problem (total_trees : ℕ) (tree_a_percent : ℚ) (tree_b_percent : ℚ)
  (tree_b_oranges : ℕ) (tree_b_good_ratio : ℚ) (tree_a_good_ratio : ℚ) (total_good_oranges : ℕ) :
  tree_a_percent = 1/2 →
  tree_b_percent = 1/2 →
  tree_b_oranges = 15 →
  tree_b_good_ratio = 1/3 →
  tree_a_good_ratio = 3/5 →
  total_trees = 10 →
  total_good_oranges = 55 →
  ∃ (tree_a_oranges : ℕ), 
    (tree_a_percent * total_trees : ℚ) * (tree_a_oranges : ℚ) * tree_a_good_ratio +
    (tree_b_percent * total_trees : ℚ) * (tree_b_oranges : ℚ) * tree_b_good_ratio =
    total_good_oranges ∧
    tree_a_oranges = 10 :=
by sorry

end orange_tree_problem_l4016_401690


namespace monthly_salary_is_1000_l4016_401616

/-- Calculates the monthly salary given savings rate, expense increase, and new savings amount -/
def calculate_salary (savings_rate : ℚ) (expense_increase : ℚ) (new_savings : ℚ) : ℚ :=
  new_savings / (savings_rate - (1 - savings_rate) * expense_increase)

/-- Theorem stating that under the given conditions, the monthly salary is 1000 -/
theorem monthly_salary_is_1000 : 
  let savings_rate : ℚ := 25 / 100
  let expense_increase : ℚ := 10 / 100
  let new_savings : ℚ := 175
  calculate_salary savings_rate expense_increase new_savings = 1000 := by
  sorry

#eval calculate_salary (25/100) (10/100) 175

end monthly_salary_is_1000_l4016_401616


namespace region_perimeter_l4016_401669

theorem region_perimeter (total_area : ℝ) (num_squares : ℕ) (row1_squares row2_squares : ℕ) :
  total_area = 400 →
  num_squares = 8 →
  row1_squares = 3 →
  row2_squares = 5 →
  row1_squares + row2_squares = num_squares →
  let square_area := total_area / num_squares
  let side_length := Real.sqrt square_area
  let perimeter := (2 * (row1_squares + row2_squares) + 2) * side_length
  perimeter = 90 * Real.sqrt 2 := by
sorry

end region_perimeter_l4016_401669


namespace salt_solution_mixture_l4016_401696

/-- The volume of a 60% salt solution needed to mix with 1 liter of pure water to create a 20% salt solution -/
def salt_solution_volume : ℝ := 0.5

/-- The concentration of salt in the original solution -/
def original_concentration : ℝ := 0.6

/-- The concentration of salt in the final mixture -/
def final_concentration : ℝ := 0.2

/-- The volume of pure water added -/
def pure_water_volume : ℝ := 1

theorem salt_solution_mixture :
  salt_solution_volume * original_concentration = 
  (pure_water_volume + salt_solution_volume) * final_concentration :=
sorry

end salt_solution_mixture_l4016_401696


namespace large_tent_fabric_is_8_l4016_401622

/-- The amount of fabric needed for a small tent -/
def small_tent_fabric : ℝ := 4

/-- The amount of fabric needed for a large tent -/
def large_tent_fabric : ℝ := 2 * small_tent_fabric

/-- Theorem: The fabric needed for a large tent is 8 square meters -/
theorem large_tent_fabric_is_8 : large_tent_fabric = 8 := by
  sorry

end large_tent_fabric_is_8_l4016_401622


namespace ms_jones_class_size_l4016_401673

theorem ms_jones_class_size :
  ∀ (total_students : ℕ),
    (total_students : ℝ) * 0.3 * (1/3) * 10 = 50 →
    total_students = 50 := by
  sorry

end ms_jones_class_size_l4016_401673


namespace sin_q_in_special_right_triangle_l4016_401623

/-- Given a right triangle PQR with ∠P as the right angle, PR = 40, and QR = 41, prove that sin Q = 9/41 -/
theorem sin_q_in_special_right_triangle (P Q R : ℝ × ℝ) : 
  let pq := Real.sqrt ((Q.1 - P.1)^2 + (Q.2 - P.2)^2)
  let pr := Real.sqrt ((R.1 - P.1)^2 + (R.2 - P.2)^2)
  let qr := Real.sqrt ((R.1 - Q.1)^2 + (R.2 - Q.2)^2)
  (pq^2 + pr^2 = qr^2) →  -- Right angle at P
  pr = 40 →
  qr = 41 →
  (Q.2 - P.2) / qr = 9 / 41 :=
by sorry

end sin_q_in_special_right_triangle_l4016_401623


namespace number_operation_result_l4016_401675

theorem number_operation_result : 
  let x : ℕ := 265
  (x / 5 + 8 : ℚ) = 61 := by sorry

end number_operation_result_l4016_401675


namespace triple_solution_l4016_401678

theorem triple_solution (a b c : ℝ) : 
  a * b * c = 8 ∧ 
  a^2 * b + b^2 * c + c^2 * a = 73 ∧ 
  a * (b - c)^2 + b * (c - a)^2 + c * (a - b)^2 = 98 →
  ((a = 4 ∧ b = 4 ∧ c = 1/2) ∨
   (a = 4 ∧ b = 1/2 ∧ c = 4) ∨
   (a = 1/2 ∧ b = 4 ∧ c = 4) ∨
   (a = 1 ∧ b = 1 ∧ c = 8) ∨
   (a = 1 ∧ b = 8 ∧ c = 1) ∨
   (a = 8 ∧ b = 1 ∧ c = 1)) := by
  sorry

end triple_solution_l4016_401678


namespace log_difference_decreases_l4016_401625

theorem log_difference_decreases (m n : ℕ) (h : m > n) :
  Real.log (1 + 1 / (m : ℝ)) < Real.log (1 + 1 / (n : ℝ)) := by
  sorry

end log_difference_decreases_l4016_401625


namespace water_tank_solution_l4016_401664

/-- Represents the water tank problem --/
def WaterTankProblem (tankCapacity : ℝ) (initialFill : ℝ) (firstDayCollection : ℝ) (thirdDayOverflow : ℝ) : Prop :=
  let initialWater := tankCapacity * initialFill
  let afterFirstDay := initialWater + firstDayCollection
  let secondDayCollection := tankCapacity - afterFirstDay
  secondDayCollection - firstDayCollection = 30

/-- Theorem statement for the water tank problem --/
theorem water_tank_solution :
  WaterTankProblem 100 (2/5) 15 25 := by
  sorry


end water_tank_solution_l4016_401664


namespace factorization_theorem_l4016_401632

theorem factorization_theorem (a b : ℝ) : 6 * a * b - a^2 - 9 * b^2 = -(a - 3 * b)^2 := by
  sorry

end factorization_theorem_l4016_401632


namespace residue_of_negative_998_mod_28_l4016_401648

theorem residue_of_negative_998_mod_28 :
  ∃ (q : ℤ), -998 = 28 * q + 10 ∧ (0 ≤ 10) ∧ (10 < 28) := by sorry

end residue_of_negative_998_mod_28_l4016_401648


namespace new_partner_associate_ratio_l4016_401650

/-- Given a firm with partners and associates, this theorem proves the new ratio
    after hiring additional associates. -/
theorem new_partner_associate_ratio
  (initial_partner_count : ℕ)
  (initial_associate_count : ℕ)
  (additional_associates : ℕ)
  (h1 : initial_partner_count = 18)
  (h2 : initial_associate_count = 567)
  (h3 : additional_associates = 45) :
  (initial_partner_count : ℚ) / (initial_associate_count + additional_associates : ℚ) = 1 / 34 := by
  sorry

#check new_partner_associate_ratio

end new_partner_associate_ratio_l4016_401650


namespace smallest_ellipse_area_l4016_401620

/-- The smallest area of an ellipse containing two specific circles -/
theorem smallest_ellipse_area (p q : ℝ) (h_ellipse : ∀ (x y : ℝ), x^2 / p^2 + y^2 / q^2 = 1 → 
  ((x - 2)^2 + y^2 = 4 ∨ (x + 2)^2 + y^2 = 4)) :
  ∃ (m : ℝ), m = 3 * Real.sqrt 3 / 2 ∧ 
    ∀ (p' q' : ℝ), (∀ (x y : ℝ), x^2 / p'^2 + y^2 / q'^2 = 1 → 
      ((x - 2)^2 + y^2 = 4 ∨ (x + 2)^2 + y^2 = 4)) → 
    p' * q' * Real.pi ≥ m * Real.pi :=
by
  sorry

end smallest_ellipse_area_l4016_401620


namespace circle_diameter_from_area_l4016_401692

theorem circle_diameter_from_area (A : ℝ) (r : ℝ) (d : ℝ) : A = 4 * Real.pi → A = Real.pi * r^2 → d = 2 * r → d = 4 := by
  sorry

end circle_diameter_from_area_l4016_401692


namespace chinese_remainder_theorem_example_l4016_401665

theorem chinese_remainder_theorem_example (x : ℤ) : 
  x ≡ 2 [ZMOD 7] → x ≡ 3 [ZMOD 6] → x ≡ 9 [ZMOD 42] := by
  sorry

end chinese_remainder_theorem_example_l4016_401665


namespace parabola_standard_equation_l4016_401635

/-- A parabola with vertex at the origin and axis of symmetry along coordinate axes -/
structure Parabola where
  focus_on_line : ∃ (x y : ℝ), 2*x - y - 4 = 0

/-- The standard equation of a parabola -/
inductive StandardEquation where
  | vert : StandardEquation  -- y² = 8x
  | horz : StandardEquation  -- x² = -16y

/-- Theorem: Given a parabola with vertex at origin, axis of symmetry along coordinate axes,
    and focus on the line 2x - y - 4 = 0, its standard equation is either y² = 8x or x² = -16y -/
theorem parabola_standard_equation (p : Parabola) : 
  ∃ (eq : StandardEquation), 
    (eq = StandardEquation.vert ∨ eq = StandardEquation.horz) := by
  sorry

end parabola_standard_equation_l4016_401635


namespace set_expressions_correct_l4016_401661

def solution_set : Set ℝ := {x | x^2 - 4 = 0}
def prime_set : Set ℕ := {p | Nat.Prime p ∧ 0 < 2 * p ∧ 2 * p < 18}
def even_set : Set ℤ := {x | ∃ n : ℤ, x = 2 * n}
def fourth_quadrant : Set (ℝ × ℝ) := {p | p.1 > 0 ∧ p.2 < 0}

theorem set_expressions_correct :
  solution_set = {-2, 2} ∧
  prime_set = {2, 3, 5, 7} ∧
  even_set = {x | ∃ n : ℤ, x = 2 * n} ∧
  fourth_quadrant = {p : ℝ × ℝ | p.1 > 0 ∧ p.2 < 0} :=
by sorry

end set_expressions_correct_l4016_401661


namespace symmetry_about_x_equals_one_l4016_401679

theorem symmetry_about_x_equals_one (f : ℝ → ℝ) (x : ℝ) : f (x - 1) = f (-(x - 1) + 1) := by
  sorry

end symmetry_about_x_equals_one_l4016_401679


namespace points_ABD_collinear_l4016_401604

-- Define the vector space
variable {V : Type*} [AddCommGroup V] [Module ℝ V]

-- Define vectors a and b
variable (a b : V)

-- Define points A, B, C, D
variable (A B C D : V)

-- State the theorem
theorem points_ABD_collinear
  (h_not_collinear : ¬ ∃ (k : ℝ), a = k • b)
  (h_AB : B - A = a + 2 • b)
  (h_BC : C - B = -3 • a + 7 • b)
  (h_CD : D - C = 4 • a - 5 • b) :
  ∃ (k : ℝ), D - A = k • (B - A) :=
sorry

end points_ABD_collinear_l4016_401604


namespace xy_length_l4016_401670

/-- Triangle similarity and side length properties -/
structure TriangleSimilarity where
  PQ : ℝ
  QR : ℝ
  YZ : ℝ
  perimeter_XYZ : ℝ
  similar : Bool  -- Represents that PQR is similar to XYZ

/-- Main theorem: XY length in similar triangles -/
theorem xy_length (t : TriangleSimilarity) 
  (h1 : t.PQ = 8)
  (h2 : t.QR = 16)
  (h3 : t.YZ = 24)
  (h4 : t.perimeter_XYZ = 60)
  (h5 : t.similar = true) : 
  ∃ XY : ℝ, XY = 12 ∧ XY + t.YZ + (t.perimeter_XYZ - XY - t.YZ) = t.perimeter_XYZ :=
sorry

end xy_length_l4016_401670


namespace farm_ratio_after_transaction_l4016_401687

/-- Represents the number of animals on the farm -/
structure FarmAnimals where
  horses : ℕ
  cows : ℕ

/-- Represents the ratio of horses to cows -/
structure Ratio where
  horses : ℕ
  cows : ℕ

def initial_ratio : Ratio := { horses := 5, cows := 1 }

def transaction (farm : FarmAnimals) : FarmAnimals :=
  { horses := farm.horses - 15, cows := farm.cows + 15 }

theorem farm_ratio_after_transaction (farm : FarmAnimals) :
  farm.horses = 5 * farm.cows →
  (transaction farm).horses = (transaction farm).cows + 50 →
  ∃ (k : ℕ), k > 0 ∧ (transaction farm).horses = 17 * k ∧ (transaction farm).cows = 7 * k :=
sorry

end farm_ratio_after_transaction_l4016_401687


namespace percent_relation_l4016_401609

theorem percent_relation (a b : ℝ) (h : a = 1.25 * b) : 4 * b = 3.2 * a := by
  sorry

end percent_relation_l4016_401609


namespace pond_volume_1400_l4016_401610

/-- The volume of a rectangular prism-shaped pond -/
def pond_volume (length width depth : ℝ) : ℝ := length * width * depth

/-- Theorem: The volume of a pond with dimensions 28 m x 10 m x 5 m is 1400 cubic meters -/
theorem pond_volume_1400 :
  pond_volume 28 10 5 = 1400 := by
  sorry

end pond_volume_1400_l4016_401610


namespace range_of_expression_l4016_401607

theorem range_of_expression (α β : ℝ) 
  (h1 : 0 < α) (h2 : α < π / 2) 
  (h3 : 0 < β) (h4 : β < π / 2) : 
  -π / 6 < 2 * α - β / 3 ∧ 2 * α - β / 3 < π := by
  sorry

end range_of_expression_l4016_401607


namespace corn_selling_price_l4016_401652

/-- Calculates the selling price per bag of corn to achieve a desired profit percentage --/
theorem corn_selling_price 
  (seed_cost fertilizer_cost labor_cost : ℕ) 
  (num_bags : ℕ) 
  (profit_percentage : ℚ) 
  (h1 : seed_cost = 50)
  (h2 : fertilizer_cost = 35)
  (h3 : labor_cost = 15)
  (h4 : num_bags = 10)
  (h5 : profit_percentage = 10 / 100) :
  (seed_cost + fertilizer_cost + labor_cost : ℚ) * (1 + profit_percentage) / num_bags = 11 := by
sorry

end corn_selling_price_l4016_401652


namespace classroom_students_l4016_401629

theorem classroom_students (total_notebooks : ℕ) (notebooks_per_half1 : ℕ) (notebooks_per_half2 : ℕ) :
  total_notebooks = 112 →
  notebooks_per_half1 = 5 →
  notebooks_per_half2 = 3 →
  ∃ (num_students : ℕ),
    num_students % 2 = 0 ∧
    (num_students / 2) * notebooks_per_half1 + (num_students / 2) * notebooks_per_half2 = total_notebooks ∧
    num_students = 28 := by
  sorry

end classroom_students_l4016_401629


namespace unique_divisible_by_18_l4016_401688

/-- The function that constructs the four-digit number x47x from a single digit x -/
def construct_number (x : ℕ) : ℕ := 1000 * x + 470 + x

/-- Predicate that checks if a number is a single digit -/
def is_single_digit (x : ℕ) : Prop := x ≥ 0 ∧ x ≤ 9

theorem unique_divisible_by_18 :
  ∃! x : ℕ, is_single_digit x ∧ (construct_number x) % 18 = 0 ∧ x = 8 := by
  sorry

end unique_divisible_by_18_l4016_401688


namespace power_equality_l4016_401637

theorem power_equality (m n : ℕ) (h1 : 2^m = 5) (h2 : 4^n = 3) : 4^(3*n - m) = 27/25 := by
  sorry

end power_equality_l4016_401637


namespace max_value_x_minus_2y_l4016_401684

theorem max_value_x_minus_2y (x y : ℝ) (h : x^2 - 4*x + y^2 = 0) :
  ∃ (max : ℝ), (∀ (x' y' : ℝ), x'^2 - 4*x' + y'^2 = 0 → x' - 2*y' ≤ max) ∧ 
  (∃ (x₀ y₀ : ℝ), x₀^2 - 4*x₀ + y₀^2 = 0 ∧ x₀ - 2*y₀ = max) ∧ 
  max = 2*Real.sqrt 5 + 2 :=
sorry

end max_value_x_minus_2y_l4016_401684


namespace fib_inequality_fib_upper_bound_l4016_401672

-- Define Fibonacci sequence
def fib : ℕ → ℕ
  | 0 => 0
  | 1 => 1
  | (n + 2) => fib n + fib (n + 1)

-- Statement 1
theorem fib_inequality (n : ℕ) (h : n ≥ 2) : fib (n + 5) > 10 * fib n := by
  sorry

-- Statement 2
theorem fib_upper_bound (n k : ℕ) (h : fib (n + 1) < 10^k) : n ≤ 5 * k := by
  sorry

end fib_inequality_fib_upper_bound_l4016_401672


namespace circle_passes_through_points_and_center_on_line_l4016_401656

-- Define the points M and N
def M : ℝ × ℝ := (5, 2)
def N : ℝ × ℝ := (3, 2)

-- Define the line equation y = 2x - 3
def line_equation (x y : ℝ) : Prop := y = 2 * x - 3

-- Define the circle equation
def circle_equation (x y : ℝ) : Prop := (x - 4)^2 + (y - 5)^2 = 10

-- Theorem statement
theorem circle_passes_through_points_and_center_on_line :
  ∃ (center : ℝ × ℝ),
    line_equation center.1 center.2 ∧
    circle_equation M.1 M.2 ∧
    circle_equation N.1 N.2 :=
sorry

end circle_passes_through_points_and_center_on_line_l4016_401656


namespace tangent_line_and_extrema_l4016_401667

noncomputable def f (x : ℝ) : ℝ := Real.exp x * Real.cos x - x

theorem tangent_line_and_extrema :
  let a : ℝ := 0
  let b : ℝ := Real.pi / 2
  -- Tangent line at (0, f(0)) is y = 1
  (∀ y, HasDerivAt f 0 y → y = 0) ∧
  f 0 = 1 ∧
  -- Maximum value is 1 at x = 0
  (∀ x ∈ Set.Icc a b, f x ≤ f 0) ∧
  -- Minimum value is -π/2 at x = π/2
  (∀ x ∈ Set.Icc a b, f b ≤ f x) ∧
  f b = -Real.pi / 2 := by
  sorry

end tangent_line_and_extrema_l4016_401667


namespace base_conversion_and_arithmetic_l4016_401660

def base_to_decimal (digits : List Nat) (base : Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * base ^ i) 0

def decimal_2468_7 : Nat := base_to_decimal [8, 6, 4, 2] 7
def decimal_121_5 : Nat := base_to_decimal [1, 2, 1] 5
def decimal_3451_6 : Nat := base_to_decimal [1, 5, 4, 3] 6
def decimal_7891_7 : Nat := base_to_decimal [1, 9, 8, 7] 7

theorem base_conversion_and_arithmetic :
  (decimal_2468_7 / decimal_121_5 : Nat) - decimal_3451_6 + decimal_7891_7 = 2059 := by
  sorry

end base_conversion_and_arithmetic_l4016_401660


namespace aaron_brothers_count_l4016_401668

/-- 
Given that Bennett has 6 brothers and the number of Bennett's brothers is two less than twice 
the number of Aaron's brothers, prove that Aaron has 4 brothers.
-/
theorem aaron_brothers_count :
  -- Define the number of Bennett's brothers
  let bennett_brothers : ℕ := 6
  -- Define the relationship between Aaron's and Bennett's brothers
  ∀ aaron_brothers : ℕ, bennett_brothers = 2 * aaron_brothers - 2 →
  -- Prove that Aaron has 4 brothers
  aaron_brothers = 4 := by
sorry

end aaron_brothers_count_l4016_401668


namespace consecutive_integers_sum_34_l4016_401646

theorem consecutive_integers_sum_34 :
  ∃! (a : ℕ), a > 0 ∧ (a + (a + 1) + (a + 2) + (a + 3) = 34) := by
  sorry

end consecutive_integers_sum_34_l4016_401646


namespace sector_properties_l4016_401624

/-- Given a sector OAB with central angle 120° and radius 6, 
    prove the length of arc AB and the area of segment AOB -/
theorem sector_properties :
  let angle : Real := 120 * π / 180
  let radius : Real := 6
  let arc_length : Real := radius * angle
  let sector_area : Real := (1 / 2) * radius * arc_length
  let triangle_area : Real := (1 / 2) * radius * radius * Real.sin angle
  let segment_area : Real := sector_area - triangle_area
  arc_length = 4 * π ∧ segment_area = 12 * π - 9 * Real.sqrt 3 := by
  sorry

end sector_properties_l4016_401624


namespace cube_volume_surface_area_l4016_401602

theorem cube_volume_surface_area (x : ℝ) : x > 0 → 
  (∃ (s : ℝ), s > 0 ∧ s^3 = 7*x ∧ 6*s^2 = 2*x) → x = 1323 := by
  sorry

end cube_volume_surface_area_l4016_401602


namespace intersection_equals_target_l4016_401666

-- Define the sets M and N
def M : Set ℝ := {x | Real.log (x - 1) < 0}
def N : Set ℝ := {x | 2 * x^2 - 3 * x ≤ 0}

-- Define the intersection of M and N
def M_intersect_N : Set ℝ := M ∩ N

-- Define the open-closed interval (1, 3/2]
def target_set : Set ℝ := {x | 1 < x ∧ x ≤ 3/2}

-- Theorem statement
theorem intersection_equals_target : M_intersect_N = target_set := by
  sorry

end intersection_equals_target_l4016_401666


namespace work_hours_theorem_l4016_401680

def amber_hours : ℕ := 12

def armand_hours : ℕ := amber_hours / 3

def ella_hours : ℕ := 2 * amber_hours

def total_hours : ℕ := amber_hours + armand_hours + ella_hours

theorem work_hours_theorem : total_hours = 40 := by
  sorry

end work_hours_theorem_l4016_401680


namespace exponential_function_point_l4016_401645

theorem exponential_function_point (a : ℝ) (ha : a > 0) (ha_neq : a ≠ 1) :
  let f : ℝ → ℝ := fun x ↦ a^x + 1
  f 0 = 2 := by sorry

end exponential_function_point_l4016_401645


namespace circle_with_diameter_AB_l4016_401615

-- Define the line segment AB
def line_segment_AB (x y : ℝ) : Prop :=
  x + y - 2 = 0 ∧ 0 ≤ x ∧ x ≤ 2

-- Define the circle equation
def circle_equation (x y : ℝ) : Prop :=
  (x - 1)^2 + (y - 1)^2 = 2

-- Theorem statement
theorem circle_with_diameter_AB :
  ∀ x y : ℝ, line_segment_AB x y →
  ∃ center_x center_y radius : ℝ,
    (∀ p q : ℝ, (p - center_x)^2 + (q - center_y)^2 = radius^2 ↔ circle_equation p q) :=
by sorry

end circle_with_diameter_AB_l4016_401615


namespace f_properties_l4016_401633

open Real

noncomputable def f (x : ℝ) : ℝ := (log (1 + x)) / x

theorem f_properties (x : ℝ) (h : x > 0) :
  (∀ y z, 0 < y ∧ y < z → f y > f z) ∧
  f x > 2 / (x + 2) := by
  sorry

end f_properties_l4016_401633


namespace pentagon_rectangle_intersection_angle_l4016_401697

-- Define the structure of our problem
structure PentagonWithRectangles where
  -- Regular pentagon
  pentagon_angle : ℝ
  -- Right angles from rectangles
  right_angle1 : ℝ
  right_angle2 : ℝ
  -- Reflex angle
  reflex_angle : ℝ
  -- The angle we're solving for
  x : ℝ

-- Define our theorem
theorem pentagon_rectangle_intersection_angle 
  (p : PentagonWithRectangles) 
  (h1 : p.pentagon_angle = 108)
  (h2 : p.right_angle1 = 90)
  (h3 : p.right_angle2 = 90)
  (h4 : p.reflex_angle = 198)
  (h5 : p.pentagon_angle + p.right_angle1 + p.right_angle2 + p.reflex_angle + p.x = 540) :
  p.x = 54 := by
  sorry

end pentagon_rectangle_intersection_angle_l4016_401697


namespace parallel_vectors_x_value_l4016_401663

def a : Fin 2 → ℝ := ![1, 2]
def b (x : ℝ) : Fin 2 → ℝ := ![x, -6]

theorem parallel_vectors_x_value :
  (∃ (k : ℝ), k ≠ 0 ∧ a = k • b x) → x = -3 := by
  sorry

end parallel_vectors_x_value_l4016_401663


namespace polynomial_derivative_bound_l4016_401662

theorem polynomial_derivative_bound (p : ℝ → ℝ) :
  (∀ x, x ∈ Set.Icc (-1 : ℝ) 1 → |p x| ≤ 1) →
  (∃ a b c : ℝ, ∀ x, p x = a * x^2 + b * x + c) →
  ∀ x, x ∈ Set.Icc (-1 : ℝ) 1 → |(deriv p) x| ≤ 4 := by
sorry

end polynomial_derivative_bound_l4016_401662


namespace gabby_savings_l4016_401683

/-- Represents the cost of the makeup set in dollars -/
def makeup_cost : ℕ := 65

/-- Represents the amount Gabby's mom gives her in dollars -/
def mom_gift : ℕ := 20

/-- Represents the additional amount Gabby needs after receiving the gift in dollars -/
def additional_needed : ℕ := 10

/-- Represents Gabby's initial savings in dollars -/
def initial_savings : ℕ := 35

theorem gabby_savings :
  initial_savings + mom_gift + additional_needed = makeup_cost :=
by sorry

end gabby_savings_l4016_401683


namespace problem_statement_l4016_401611

theorem problem_statement (x y : ℚ) (hx : x = 3/4) (hy : y = 4/3) : 
  (1/2 : ℚ) * x^6 * y^7 = 2/3 := by sorry

end problem_statement_l4016_401611


namespace intersection_of_A_and_B_l4016_401651

def A : Set ℝ := {x | x ≤ 1}
def B : Set ℝ := {0, 1, 2}

theorem intersection_of_A_and_B : A ∩ B = {0, 1} := by
  sorry

end intersection_of_A_and_B_l4016_401651


namespace equal_roots_implies_value_l4016_401689

/-- If x^2 + 2kx + k^2 + k + 3 = 0 has two equal real roots with respect to x,
    then k^2 + k + 3 = 9 -/
theorem equal_roots_implies_value (k : ℝ) :
  (∃ x : ℝ, x^2 + 2*k*x + k^2 + k + 3 = 0 ∧
   ∀ y : ℝ, y^2 + 2*k*y + k^2 + k + 3 = 0 → y = x) →
  k^2 + k + 3 = 9 := by
  sorry

end equal_roots_implies_value_l4016_401689


namespace min_value_a_l4016_401676

theorem min_value_a (a : ℝ) (h1 : a > 0) :
  (∀ x y : ℝ, x ≠ 0 → y ≠ 0 → (x + y) * (1 / x + a / y) ≥ 16) →
  a ≥ 9 :=
by sorry

end min_value_a_l4016_401676


namespace cube_root_of_product_l4016_401698

theorem cube_root_of_product (a : ℕ) : a^3 = 21 * 25 * 45 * 49 → a = 105 := by
  sorry

end cube_root_of_product_l4016_401698


namespace mutuallyExclusive_but_not_complementary_l4016_401630

/-- Represents the color of a ball -/
inductive Color
| Red
| White

/-- Represents the outcome of drawing two balls -/
structure DrawOutcome :=
  (first : Color)
  (second : Color)

/-- The set of all possible outcomes when drawing two balls -/
def sampleSpace : Set DrawOutcome := sorry

/-- The event of drawing exactly one white ball -/
def exactlyOneWhite : Set DrawOutcome := sorry

/-- The event of drawing exactly two white balls -/
def exactlyTwoWhite : Set DrawOutcome := sorry

/-- Two events are mutually exclusive if their intersection is empty -/
def mutuallyExclusive (A B : Set DrawOutcome) : Prop :=
  A ∩ B = ∅

/-- Two events are complementary if their union is the entire sample space -/
def complementary (A B : Set DrawOutcome) : Prop :=
  A ∪ B = sampleSpace

theorem mutuallyExclusive_but_not_complementary :
  mutuallyExclusive exactlyOneWhite exactlyTwoWhite ∧
  ¬complementary exactlyOneWhite exactlyTwoWhite :=
sorry

end mutuallyExclusive_but_not_complementary_l4016_401630


namespace no_solution_implies_m_equals_six_l4016_401614

/-- If the equation (3x - m) / (x - 2) = 1 has no solution, then m = 6 -/
theorem no_solution_implies_m_equals_six (m : ℝ) : 
  (∀ x : ℝ, (3 * x - m) / (x - 2) ≠ 1) → m = 6 := by
  sorry

end no_solution_implies_m_equals_six_l4016_401614


namespace periodic_sum_implies_periodic_increasing_sum_not_implies_increasing_l4016_401638

def PeriodicFunction (f : ℝ → ℝ) (T : ℝ) : Prop :=
  ∀ x, f (x + T) = f x

def IncreasingFunction (f : ℝ → ℝ) : Prop :=
  ∀ x y, x < y → f x < f y

theorem periodic_sum_implies_periodic
  (f g h : ℝ → ℝ) (T : ℝ) :
  (PeriodicFunction (fun x ↦ f x + g x) T) →
  (PeriodicFunction (fun x ↦ f x + h x) T) →
  (PeriodicFunction (fun x ↦ g x + h x) T) →
  (PeriodicFunction f T) ∧ (PeriodicFunction g T) ∧ (PeriodicFunction h T) := by
  sorry

theorem increasing_sum_not_implies_increasing :
  ∃ f g h : ℝ → ℝ,
    (IncreasingFunction (fun x ↦ f x + g x)) ∧
    (IncreasingFunction (fun x ↦ f x + h x)) ∧
    (IncreasingFunction (fun x ↦ g x + h x)) ∧
    (¬IncreasingFunction f ∨ ¬IncreasingFunction g ∨ ¬IncreasingFunction h) := by
  sorry

end periodic_sum_implies_periodic_increasing_sum_not_implies_increasing_l4016_401638


namespace odd_square_sum_of_consecutive_b_l4016_401640

def a : ℕ → ℕ
  | n => if n % 2 = 1 then 4 * ((n - 1) / 2) + 2 else 4 * (n / 2 - 1) + 3

def b : ℕ → ℕ
  | n => if n % 2 = 1 then 8 * ((n - 1) / 2) + 3 else 8 * (n / 2 - 1) + 6

theorem odd_square_sum_of_consecutive_b (k : ℕ) (hk : k > 0) :
  ∃ r : ℕ, (2 * k + 1)^2 = b r + b (r + 1) := by
  sorry

end odd_square_sum_of_consecutive_b_l4016_401640


namespace absolute_value_of_z_l4016_401699

theorem absolute_value_of_z (z : ℂ) (h : z^2 = 16 - 30*I) : Complex.abs z = Real.sqrt 34 := by
  sorry

end absolute_value_of_z_l4016_401699


namespace largest_digit_divisible_by_four_l4016_401654

theorem largest_digit_divisible_by_four :
  ∀ n : ℕ, 
    (n = 4969794) → 
    (∀ m : ℕ, m % 4 = 0 ↔ (m % 100) % 4 = 0) → 
    n % 4 = 0 :=
by sorry

end largest_digit_divisible_by_four_l4016_401654


namespace polynomial_equality_l4016_401691

theorem polynomial_equality (x : ℝ) : 
  (x - 1)^4 + 4*(x - 1)^3 + 6*(x - 1)^2 + 4*x - 3 = x^4 := by
  sorry

end polynomial_equality_l4016_401691


namespace total_candies_l4016_401600

/-- The number of jellybeans in a dozen -/
def dozen : ℕ := 12

/-- Caleb's candies -/
def caleb_jellybeans : ℕ := 3 * dozen
def caleb_chocolate_bars : ℕ := 5
def caleb_gummy_bears : ℕ := 8

/-- Sophie's candies -/
def sophie_jellybeans : ℕ := caleb_jellybeans / 2
def sophie_chocolate_bars : ℕ := 3
def sophie_gummy_bears : ℕ := 12

/-- Max's candies -/
def max_jellybeans : ℕ := sophie_jellybeans + 2 * dozen
def max_chocolate_bars : ℕ := 6
def max_gummy_bears : ℕ := 10

/-- Total candies for each person -/
def caleb_total : ℕ := caleb_jellybeans + caleb_chocolate_bars + caleb_gummy_bears
def sophie_total : ℕ := sophie_jellybeans + sophie_chocolate_bars + sophie_gummy_bears
def max_total : ℕ := max_jellybeans + max_chocolate_bars + max_gummy_bears

/-- Theorem: The total number of candies is 140 -/
theorem total_candies : caleb_total + sophie_total + max_total = 140 := by
  sorry

end total_candies_l4016_401600


namespace point_in_fourth_quadrant_l4016_401685

def fourth_quadrant (z : ℂ) : Prop := 
  Complex.re z > 0 ∧ Complex.im z < 0

theorem point_in_fourth_quadrant : 
  fourth_quadrant ((2 - Complex.I) ^ 2) := by
  sorry

end point_in_fourth_quadrant_l4016_401685


namespace tiffany_fastest_l4016_401603

structure Runner where
  name : String
  uphill_blocks : ℕ
  uphill_time : ℕ
  downhill_blocks : ℕ
  downhill_time : ℕ
  flat_blocks : ℕ
  flat_time : ℕ

def total_distance (r : Runner) : ℕ :=
  r.uphill_blocks + r.downhill_blocks + r.flat_blocks

def total_time (r : Runner) : ℕ :=
  r.uphill_time + r.downhill_time + r.flat_time

def average_speed (r : Runner) : ℚ :=
  (total_distance r : ℚ) / (total_time r : ℚ)

def tiffany : Runner :=
  { name := "Tiffany"
    uphill_blocks := 6
    uphill_time := 3
    downhill_blocks := 8
    downhill_time := 5
    flat_blocks := 6
    flat_time := 3 }

def moses : Runner :=
  { name := "Moses"
    uphill_blocks := 5
    uphill_time := 5
    downhill_blocks := 10
    downhill_time := 10
    flat_blocks := 5
    flat_time := 4 }

def morgan : Runner :=
  { name := "Morgan"
    uphill_blocks := 7
    uphill_time := 4
    downhill_blocks := 9
    downhill_time := 6
    flat_blocks := 4
    flat_time := 2 }

theorem tiffany_fastest : 
  average_speed tiffany > average_speed moses ∧ 
  average_speed tiffany > average_speed morgan ∧
  total_distance tiffany = 20 ∧
  total_distance moses = 20 ∧
  total_distance morgan = 20 := by
  sorry

end tiffany_fastest_l4016_401603


namespace min_value_cosine_function_l4016_401605

theorem min_value_cosine_function :
  ∀ x : ℝ, 2 * Real.cos x - 1 ≥ -3 ∧ ∃ x : ℝ, 2 * Real.cos x - 1 = -3 := by
  sorry

end min_value_cosine_function_l4016_401605


namespace mountain_loop_trail_length_l4016_401628

/-- Represents the hiking trip on Mountain Loop Trail -/
structure HikingTrip where
  day1 : ℝ
  day2 : ℝ
  day3 : ℝ
  day4 : ℝ
  day5 : ℝ

/-- The conditions of the hiking trip -/
def validHikingTrip (trip : HikingTrip) : Prop :=
  trip.day1 + trip.day2 + trip.day3 = 45 ∧
  (trip.day2 + trip.day4) / 2 = 18 ∧
  trip.day3 + trip.day4 + trip.day5 = 60 ∧
  trip.day1 + trip.day4 = 32

/-- The theorem stating the total length of the trail -/
theorem mountain_loop_trail_length (trip : HikingTrip) 
  (h : validHikingTrip trip) : 
  trip.day1 + trip.day2 + trip.day3 + trip.day4 + trip.day5 = 69 := by
  sorry


end mountain_loop_trail_length_l4016_401628


namespace supplement_of_complement_of_30_l4016_401643

def complement (α : ℝ) : ℝ := 90 - α

def supplement (α : ℝ) : ℝ := 180 - α

theorem supplement_of_complement_of_30 :
  supplement (complement 30) = 120 := by sorry

end supplement_of_complement_of_30_l4016_401643


namespace remainder_theorem_l4016_401647

theorem remainder_theorem (x : ℂ) : 
  (x^2023 + 1) % (x^10 - x^8 + x^6 - x^4 + x^2 - 1) = -x^7 + 1 := by
  sorry

end remainder_theorem_l4016_401647


namespace subset_condition_l4016_401639

def A : Set ℝ := {x | (x - 3) / (x + 1) ≥ 0}
def B (a : ℝ) : Set ℝ := {x | a * x + 1 ≤ 0}

theorem subset_condition (a : ℝ) : 
  B a ⊆ A ↔ a ∈ Set.Icc (-1/3) 1 := by sorry

end subset_condition_l4016_401639


namespace inequality_proof_l4016_401642

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  (1 + 4*a/(b+c)) * (1 + 4*b/(c+a)) * (1 + 4*c/(a+b)) > 25 := by
  sorry

end inequality_proof_l4016_401642


namespace quadratic_solution_inequality_solution_l4016_401613

-- Define the quadratic equation
def quadratic_equation (x : ℝ) : Prop := 2 * x^2 + x - 2 = 0

-- Define the inequality system
def inequality_system (x : ℝ) : Prop := x + 3 > -2*x ∧ 2*x - 5 < 1

-- Theorem for the quadratic equation solution
theorem quadratic_solution :
  ∃ x1 x2 : ℝ, x1 = (-1 + Real.sqrt 17) / 4 ∧
              x2 = (-1 - Real.sqrt 17) / 4 ∧
              quadratic_equation x1 ∧
              quadratic_equation x2 :=
sorry

-- Theorem for the inequality system solution
theorem inequality_solution :
  ∀ x : ℝ, inequality_system x ↔ -1 < x ∧ x < 3 :=
sorry

end quadratic_solution_inequality_solution_l4016_401613


namespace degree_plus_one_divides_l4016_401658

/-- A polynomial with coefficients in {1, 2022} -/
def SpecialPoly (R : Type*) [CommRing R] := Polynomial R

/-- Predicate to check if a polynomial has coefficients only in {1, 2022} -/
def HasSpecialCoeffs (p : Polynomial ℤ) : Prop :=
  ∀ (i : ℕ), p.coeff i = 1 ∨ p.coeff i = 2022 ∨ p.coeff i = 0

theorem degree_plus_one_divides
  (f g : Polynomial ℤ)
  (hf : HasSpecialCoeffs f)
  (hg : HasSpecialCoeffs g)
  (h_div : f ∣ g) :
  (Polynomial.degree f + 1) ∣ (Polynomial.degree g + 1) :=
sorry

end degree_plus_one_divides_l4016_401658
