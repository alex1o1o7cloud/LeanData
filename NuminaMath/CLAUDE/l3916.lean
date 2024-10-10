import Mathlib

namespace imaginary_part_of_2_plus_i_times_i_l3916_391613

theorem imaginary_part_of_2_plus_i_times_i (i : ℂ) : 
  Complex.im ((2 : ℂ) + i * i) = 2 :=
by
  sorry

end imaginary_part_of_2_plus_i_times_i_l3916_391613


namespace negation_of_universal_proposition_l3916_391609

theorem negation_of_universal_proposition :
  (¬ ∀ x : ℝ, x^2 + 2*x + 5 > 0) ↔ (∃ x : ℝ, x^2 + 2*x + 5 ≤ 0) :=
by sorry

end negation_of_universal_proposition_l3916_391609


namespace cubic_equation_solution_product_l3916_391667

theorem cubic_equation_solution_product (d e f : ℝ) : 
  d^3 + 2*d^2 + 3*d - 5 = 0 ∧ 
  e^3 + 2*e^2 + 3*e - 5 = 0 ∧ 
  f^3 + 2*f^2 + 3*f - 5 = 0 → 
  (d - 1) * (e - 1) * (f - 1) = 3 := by
sorry

end cubic_equation_solution_product_l3916_391667


namespace age_sum_problem_l3916_391626

theorem age_sum_problem (a b c : ℕ+) : 
  a = b ∧ 
  a > c ∧ 
  c < 10 ∧ 
  a * b * c = 162 → 
  a + b + c = 20 := by
sorry

end age_sum_problem_l3916_391626


namespace distance_between_points_l3916_391603

/-- The distance between two points A and B, given the travel time and average speed -/
theorem distance_between_points (time : ℝ) (speed : ℝ) (h1 : time = 4.5) (h2 : speed = 80) :
  time * speed = 360 := by
  sorry

end distance_between_points_l3916_391603


namespace disc_price_calculation_l3916_391697

/-- The price of the other type of compact disc -/
def other_disc_price : ℝ := 10.50

theorem disc_price_calculation (total_discs : ℕ) (total_spent : ℝ) (known_price : ℝ) (known_quantity : ℕ) :
  total_discs = 10 →
  total_spent = 93 →
  known_price = 8.50 →
  known_quantity = 6 →
  other_disc_price = (total_spent - known_price * known_quantity) / (total_discs - known_quantity) :=
by
  sorry

#eval other_disc_price

end disc_price_calculation_l3916_391697


namespace exam_time_allocation_l3916_391605

theorem exam_time_allocation (total_time : ℕ) (total_questions : ℕ) (type_a_questions : ℕ) :
  total_time = 180 →
  total_questions = 200 →
  type_a_questions = 50 →
  let type_b_questions := total_questions - type_a_questions
  let time_ratio := 2
  let total_time_units := type_a_questions * time_ratio + type_b_questions
  let time_per_unit := total_time / total_time_units
  let time_for_type_a := type_a_questions * time_ratio * time_per_unit
  time_for_type_a = 72 :=
by
  sorry

#check exam_time_allocation

end exam_time_allocation_l3916_391605


namespace min_hypotenuse_right_triangle_l3916_391641

theorem min_hypotenuse_right_triangle (a b c : ℝ) 
  (h1 : a > 0) (h2 : b > 0) (h3 : c > 0)
  (h4 : a + b = 10) 
  (h5 : c^2 = a^2 + b^2) : 
  c ≥ 5 * Real.sqrt 2 := by
sorry

end min_hypotenuse_right_triangle_l3916_391641


namespace sum_always_positive_l3916_391637

-- Define a monotonically increasing odd function on ℝ
def MonoIncreasingOddFunction (f : ℝ → ℝ) : Prop :=
  (∀ x y, x < y → f x < f y) ∧ (∀ x, f (-x) = -f x)

-- Define an arithmetic sequence
def ArithmeticSequence (a : ℕ → ℝ) : Prop :=
  ∃ d, ∀ n, a (n + 1) = a n + d

-- Theorem statement
theorem sum_always_positive
  (f : ℝ → ℝ)
  (a : ℕ → ℝ)
  (hf : MonoIncreasingOddFunction f)
  (ha : ArithmeticSequence a)
  (ha3_pos : a 3 > 0) :
  f (a 1) + f (a 3) + f (a 5) > 0 :=
sorry

end sum_always_positive_l3916_391637


namespace gcd_lcm_product_l3916_391652

theorem gcd_lcm_product (a b : ℕ) (h1 : Nat.gcd a b = 12) (h2 : Nat.lcm a b = 168) :
  a * b = 2016 := by
  sorry

end gcd_lcm_product_l3916_391652


namespace angela_deliveries_l3916_391625

/-- Calculates the total number of meals and packages delivered -/
def total_deliveries (meals : ℕ) (package_multiplier : ℕ) : ℕ :=
  meals + meals * package_multiplier

/-- Proves that given 3 meals and 8 times as many packages, the total deliveries is 27 -/
theorem angela_deliveries : total_deliveries 3 8 = 27 := by
  sorry

end angela_deliveries_l3916_391625


namespace abc_divisibility_problem_l3916_391648

theorem abc_divisibility_problem :
  ∀ a b c : ℕ,
    1 < a → a < b → b < c →
    (((a - 1) * (b - 1) * (c - 1)) ∣ (a * b * c - 1)) →
    ((a = 3 ∧ b = 5 ∧ c = 15) ∨ (a = 2 ∧ b = 4 ∧ c = 8)) :=
by sorry

end abc_divisibility_problem_l3916_391648


namespace min_distance_from_point_on_unit_circle_l3916_391668

theorem min_distance_from_point_on_unit_circle (z : ℂ) (h : Complex.abs z = 1) :
  Complex.abs (z - (3 + 4 * Complex.I)) ≥ 4 := by
  sorry

end min_distance_from_point_on_unit_circle_l3916_391668


namespace correct_calculation_l3916_391657

theorem correct_calculation (x : ℚ) (h : x + 7/5 = 81/20) : x - 7/5 = 25/20 := by
  sorry

end correct_calculation_l3916_391657


namespace det_skew_symmetric_nonneg_l3916_391661

/-- A 4x4 real matrix is skew-symmetric if its transpose is equal to its negation. -/
def isSkewSymmetric (A : Matrix (Fin 4) (Fin 4) ℝ) : Prop :=
  A.transpose = -A

/-- The determinant of a 4x4 real skew-symmetric matrix is non-negative. -/
theorem det_skew_symmetric_nonneg (A : Matrix (Fin 4) (Fin 4) ℝ) 
  (h : isSkewSymmetric A) : 0 ≤ A.det := by
  sorry

end det_skew_symmetric_nonneg_l3916_391661


namespace abs_ratio_equality_l3916_391608

theorem abs_ratio_equality (a b : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (h : a^2 + b^2 = 9*a*b) :
  |(a + b) / (a - b)| = Real.sqrt 77 / 7 := by
  sorry

end abs_ratio_equality_l3916_391608


namespace divisibility_property_l3916_391651

theorem divisibility_property (x y a b S : ℤ) 
  (sum_eq : x + y = S) 
  (masha_divisible : S ∣ (a * x + b * y)) : 
  S ∣ (b * x + a * y) := by
  sorry

end divisibility_property_l3916_391651


namespace computer_multiplications_l3916_391691

theorem computer_multiplications (multiplications_per_second : ℕ) (hours : ℕ) : 
  multiplications_per_second = 15000 → 
  hours = 3 → 
  multiplications_per_second * (hours * 3600) = 162000000 := by
  sorry

end computer_multiplications_l3916_391691


namespace absolute_value_inequality_solution_set_l3916_391650

theorem absolute_value_inequality_solution_set :
  {x : ℝ | |x + 3| < 1} = {x : ℝ | -4 < x ∧ x < -2} := by
  sorry

end absolute_value_inequality_solution_set_l3916_391650


namespace rectangle_to_square_l3916_391604

/-- Represents a rectangle with integer dimensions -/
structure Rectangle where
  length : ℕ
  width : ℕ

/-- Represents a square with integer side length -/
structure Square where
  side : ℕ

/-- Represents the area of a shape -/
def area : Rectangle → ℕ
  | ⟨l, w⟩ => l * w

/-- Theorem stating that a 9x4 rectangle can be cut and rearranged into a 6x6 square -/
theorem rectangle_to_square : 
  ∃ (r : Rectangle) (s : Square), 
    r.length = 9 ∧ 
    r.width = 4 ∧ 
    s.side = 6 ∧ 
    area r = s.side * s.side := by
  sorry

end rectangle_to_square_l3916_391604


namespace factorization_proof_l3916_391675

theorem factorization_proof :
  (∀ x : ℝ, 4 * x^2 - 36 = 4 * (x + 3) * (x - 3)) ∧
  (∀ x y : ℝ, x^3 - 2 * x^2 * y + x * y^2 = x * (x - y)^2) := by
sorry

end factorization_proof_l3916_391675


namespace perpendicular_vectors_sum_l3916_391685

/-- Given two perpendicular vectors a and b in ℝ², prove their sum is (3, -1) -/
theorem perpendicular_vectors_sum (a b : ℝ × ℝ) :
  a.1 = x ∧ a.2 = 1 ∧ b = (1, -2) ∧ a.1 * b.1 + a.2 * b.2 = 0 →
  a + b = (3, -1) := by sorry

end perpendicular_vectors_sum_l3916_391685


namespace complex_on_imaginary_axis_l3916_391622

theorem complex_on_imaginary_axis (z : ℂ) : 
  Complex.abs (z - 1) = Complex.abs (z + 1) → z.re = 0 :=
by sorry

end complex_on_imaginary_axis_l3916_391622


namespace columbus_discovery_year_l3916_391636

def is_15th_century (year : ℕ) : Prop := 1400 ≤ year ∧ year ≤ 1499

def sum_of_digits (year : ℕ) : ℕ :=
  (year / 1000) + ((year / 100) % 10) + ((year / 10) % 10) + (year % 10)

def tens_digit (year : ℕ) : ℕ := (year / 10) % 10

def units_digit (year : ℕ) : ℕ := year % 10

theorem columbus_discovery_year :
  ∃! year : ℕ,
    is_15th_century year ∧
    sum_of_digits year = 16 ∧
    tens_digit year / units_digit year = 4 ∧
    tens_digit year % units_digit year = 1 ∧
    year = 1492 :=
by
  sorry

end columbus_discovery_year_l3916_391636


namespace P_bounds_l3916_391681

/-- Represents the minimum number of transformations needed to convert
    any triangulation of a convex n-gon to any other triangulation. -/
def P (n : ℕ) : ℕ := sorry

/-- The main theorem about the bounds of P(n) -/
theorem P_bounds (n : ℕ) : 
  (n ≥ 3 → P n ≥ n - 3) ∧ 
  (n ≥ 3 → P n ≤ 2*n - 7) ∧ 
  (n ≥ 13 → P n ≤ 2*n - 10) := by
  sorry

end P_bounds_l3916_391681


namespace cube_of_sum_fractions_is_three_l3916_391646

theorem cube_of_sum_fractions_is_three (a b c : ℤ) 
  (h : (a : ℚ) / b + (b : ℚ) / c + (c : ℚ) / a = 3) : 
  ∃ n : ℤ, a * b * c = n^3 := by
sorry

end cube_of_sum_fractions_is_three_l3916_391646


namespace max_value_problem_l3916_391610

theorem max_value_problem (a b c : ℝ) 
  (ha : 0 ≤ a ∧ a ≤ 2) 
  (hb : 0 ≤ b ∧ b ≤ 2) 
  (hc : 0 ≤ c ∧ c ≤ 2) : 
  a^2 * b^2 * c^2 + (2 - a)^2 * (2 - b)^2 * (2 - c)^2 ≤ 64 := by
  sorry

end max_value_problem_l3916_391610


namespace number_expression_not_equal_l3916_391618

theorem number_expression_not_equal (x : ℝ) : 5 * x + 7 ≠ 5 * (x + 7) := by
  sorry

end number_expression_not_equal_l3916_391618


namespace inverse_f_at_3_l3916_391619

-- Define the function f
def f (x : ℝ) : ℝ := x^2 + 2

-- Define the domain of f
def f_domain (x : ℝ) : Prop := -2 ≤ x ∧ x < 0

-- State the theorem
theorem inverse_f_at_3 :
  ∃ (f_inv : ℝ → ℝ), (∀ x, f_domain x → f_inv (f x) = x) ∧ f_inv 3 = -1 :=
sorry

end inverse_f_at_3_l3916_391619


namespace jacqueline_initial_plums_l3916_391677

/-- The number of plums Jacqueline had initially -/
def initial_plums : ℕ := 16

/-- The number of guavas Jacqueline had initially -/
def initial_guavas : ℕ := 18

/-- The number of apples Jacqueline had initially -/
def initial_apples : ℕ := 21

/-- The number of fruits Jacqueline gave away -/
def fruits_given_away : ℕ := 40

/-- The number of fruits Jacqueline had left -/
def fruits_left : ℕ := 15

/-- Theorem stating that the initial number of plums is 16 -/
theorem jacqueline_initial_plums :
  initial_plums = 16 ∧
  initial_plums + initial_guavas + initial_apples = fruits_given_away + fruits_left :=
by sorry

end jacqueline_initial_plums_l3916_391677


namespace unique_solution_l3916_391639

def base_6_value (s h e : ℕ) : ℕ := s * 36 + h * 6 + e

theorem unique_solution :
  ∀ (s h e : ℕ),
    s ≠ 0 ∧ h ≠ 0 ∧ e ≠ 0 →
    s < 6 ∧ h < 6 ∧ e < 6 →
    s ≠ h ∧ s ≠ e ∧ h ≠ e →
    base_6_value s h e + base_6_value 0 h e = base_6_value h e s →
    s = 4 ∧ h = 2 ∧ e = 5 ∧ (s + h + e) % 6 = 5 ∧ ((s + h + e) / 6) % 6 = 1 :=
by sorry

end unique_solution_l3916_391639


namespace no_equal_result_from_19_and_98_l3916_391621

/-- Represents the two possible operations: squaring or adding one -/
inductive Operation
  | square
  | addOne

/-- Applies the given operation to a number -/
def applyOperation (n : ℕ) (op : Operation) : ℕ :=
  match op with
  | Operation.square => n * n
  | Operation.addOne => n + 1

/-- Applies a sequence of operations to a number -/
def applyOperations (start : ℕ) (ops : List Operation) : ℕ :=
  ops.foldl applyOperation start

/-- Theorem stating that it's impossible to obtain the same number from 19 and 98
    using the same number of operations -/
theorem no_equal_result_from_19_and_98 :
  ¬ ∃ (ops1 ops2 : List Operation) (result : ℕ),
    ops1.length = ops2.length ∧
    applyOperations 19 ops1 = result ∧
    applyOperations 98 ops2 = result :=
  sorry


end no_equal_result_from_19_and_98_l3916_391621


namespace set_equality_invariant_under_variable_renaming_l3916_391638

theorem set_equality_invariant_under_variable_renaming :
  {x : ℝ | x ≤ 1} = {t : ℝ | t ≤ 1} := by
  sorry

end set_equality_invariant_under_variable_renaming_l3916_391638


namespace smallest_four_digit_non_divisor_l3916_391696

def sum_of_first_n (n : ℕ) : ℕ := n * (n + 1) / 2

def product_of_first_n (n : ℕ) : ℕ := Nat.factorial n

theorem smallest_four_digit_non_divisor :
  (∀ m : ℕ, 1000 ≤ m → m < 1005 → (product_of_first_n m)^2 % (sum_of_first_n m) = 0) ∧
  (product_of_first_n 1005)^2 % (sum_of_first_n 1005) ≠ 0 := by
  sorry

end smallest_four_digit_non_divisor_l3916_391696


namespace six_digit_repeating_divisible_by_11_l3916_391693

/-- A 6-digit integer where the first three digits and the last three digits
    form the same three-digit number in the same order is divisible by 11. -/
theorem six_digit_repeating_divisible_by_11 (N : ℕ) (a b c : ℕ) :
  N = 100000 * a + 10000 * b + 1000 * c + 100 * a + 10 * b + c →
  a < 10 → b < 10 → c < 10 →
  11 ∣ N :=
by sorry

end six_digit_repeating_divisible_by_11_l3916_391693


namespace parking_garage_problem_l3916_391647

theorem parking_garage_problem (first_level : ℕ) (second_level : ℕ) (third_level : ℕ) (fourth_level : ℕ) 
  (h1 : first_level = 90)
  (h2 : second_level = first_level + 8)
  (h3 : third_level = second_level + 12)
  (h4 : fourth_level = third_level - 9)
  (h5 : first_level + second_level + third_level + fourth_level - 299 = 100) : 
  ∃ (cars_parked : ℕ), cars_parked = 100 := by
sorry

end parking_garage_problem_l3916_391647


namespace three_digit_difference_l3916_391654

theorem three_digit_difference (a b c : ℕ) (h1 : a ≥ 1) (h2 : a ≤ 9) (h3 : b ≥ 0) (h4 : b ≤ 9) (h5 : c ≥ 0) (h6 : c ≤ 9) (h7 : a = c + 2) :
  (100 * a + 10 * b + c) - (100 * c + 10 * b + a) = 198 := by
  sorry

end three_digit_difference_l3916_391654


namespace common_measure_of_angles_l3916_391669

-- Define the angles and natural numbers
variable (α β : ℝ)
variable (m n : ℕ)

-- State the theorem
theorem common_measure_of_angles (h : α = β * (m / n)) :
  α / m = β / n ∧ 
  ∃ (k₁ k₂ : ℕ), α = k₁ * (α / m) ∧ β = k₂ * (β / n) :=
sorry

end common_measure_of_angles_l3916_391669


namespace probability_one_of_each_l3916_391664

/-- The number of forks in the drawer -/
def num_forks : ℕ := 7

/-- The number of spoons in the drawer -/
def num_spoons : ℕ := 8

/-- The number of knives in the drawer -/
def num_knives : ℕ := 5

/-- The total number of pieces of silverware -/
def total_pieces : ℕ := num_forks + num_spoons + num_knives

/-- The number of pieces to be selected -/
def num_selected : ℕ := 3

/-- The probability of selecting one fork, one spoon, and one knife -/
theorem probability_one_of_each : 
  (num_forks * num_spoons * num_knives : ℚ) / (Nat.choose total_pieces num_selected) = 14 / 57 := by
  sorry

end probability_one_of_each_l3916_391664


namespace division_problem_l3916_391687

theorem division_problem (L S Q : ℕ) : 
  L - S = 2500 → 
  L = 2982 → 
  L = Q * S + 15 → 
  Q = 6 := by sorry

end division_problem_l3916_391687


namespace problem_1_problem_2_l3916_391617

-- Problem 1
theorem problem_1 : 2 * Real.sqrt 12 - 6 * Real.sqrt (1/3) + 3 * Real.sqrt 48 = 14 * Real.sqrt 3 := by
  sorry

-- Problem 2
theorem problem_2 (x : ℝ) (hx : x > 0) : 
  (2/3) * Real.sqrt (9*x) + 6 * Real.sqrt (x/4) - x * Real.sqrt (1/x) = 4 * Real.sqrt x := by
  sorry

end problem_1_problem_2_l3916_391617


namespace agricultural_experiment_l3916_391690

theorem agricultural_experiment (seeds_second_plot : ℕ) : 
  (300 : ℝ) * 0.30 + seeds_second_plot * 0.35 = (300 + seeds_second_plot) * 0.32 →
  seeds_second_plot = 200 := by
sorry

end agricultural_experiment_l3916_391690


namespace difference_of_squares_example_l3916_391624

theorem difference_of_squares_example : (538 * 538) - (537 * 539) = 1 := by
  sorry

end difference_of_squares_example_l3916_391624


namespace product_from_lcm_gcd_l3916_391679

theorem product_from_lcm_gcd (a b : ℕ+) (h1 : Nat.lcm a b = 120) (h2 : Nat.gcd a b = 8) :
  a * b = 960 := by sorry

end product_from_lcm_gcd_l3916_391679


namespace parentheses_expression_l3916_391695

theorem parentheses_expression (x y : ℝ) (h : x ≠ 0 ∧ y ≠ 0) :
  ∃ z : ℝ, x * y * z = -x^3 * y^2 → z = -x^2 * y :=
sorry

end parentheses_expression_l3916_391695


namespace percentage_problem_l3916_391606

theorem percentage_problem (x : ℝ) (p : ℝ) : 
  x = 230 → 
  p / 100 * x = 20 / 100 * 747.50 → 
  p = 65 := by
sorry

end percentage_problem_l3916_391606


namespace fraction_subtraction_problem_l3916_391663

theorem fraction_subtraction_problem : (1/2 : ℚ) + 5/6 - 2/3 = 2/3 := by
  sorry

end fraction_subtraction_problem_l3916_391663


namespace sum_of_roots_zero_l3916_391656

theorem sum_of_roots_zero (p q a b c : ℝ) : 
  a ≠ b → b ≠ c → a ≠ c →
  a^3 + p*a + q = 0 →
  b^3 + p*b + q = 0 →
  c^3 + p*c + q = 0 →
  a + b + c = 0 := by
sorry

end sum_of_roots_zero_l3916_391656


namespace right_triangle_inequality_l3916_391655

theorem right_triangle_inequality (a b c : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : c > 0) 
    (h4 : a ≥ b) (h5 : c^2 = a^2 + b^2) : 
  a + b / 2 > c ∧ c > 8 / 9 * (a + b / 2) := by
  sorry

end right_triangle_inequality_l3916_391655


namespace darnel_sprint_distance_l3916_391653

theorem darnel_sprint_distance (jogged_distance : Real) (additional_sprint : Real) :
  jogged_distance = 0.75 →
  additional_sprint = 0.125 →
  jogged_distance + additional_sprint = 0.875 := by
  sorry

end darnel_sprint_distance_l3916_391653


namespace incorrect_transformation_l3916_391644

theorem incorrect_transformation (a b : ℝ) :
  ¬(∀ a b : ℝ, a = b → a / b = 1) := by
  sorry

end incorrect_transformation_l3916_391644


namespace rearranged_box_surface_area_l3916_391629

theorem rearranged_box_surface_area :
  let original_length : ℝ := 2
  let original_width : ℝ := 1
  let original_height : ℝ := 1
  let first_cut_height : ℝ := 1/4
  let second_cut_height : ℝ := 1/3
  let piece_A_height : ℝ := first_cut_height
  let piece_B_height : ℝ := second_cut_height
  let piece_C_height : ℝ := original_height - (piece_A_height + piece_B_height)
  let new_length : ℝ := original_width * 3
  let new_width : ℝ := original_length
  let new_height : ℝ := piece_A_height + piece_B_height + piece_C_height
  let top_bottom_area : ℝ := 2 * (new_length * new_width)
  let side_area : ℝ := 2 * (new_height * new_width)
  let front_back_area : ℝ := 2 * (new_length * new_height)
  let total_surface_area : ℝ := top_bottom_area + side_area + front_back_area
  total_surface_area = 12 := by
    sorry

end rearranged_box_surface_area_l3916_391629


namespace min_weighings_for_extremes_l3916_391659

/-- Represents a coin with a weight -/
structure Coin where
  weight : ℕ

/-- Represents a weighing operation that compares two coins -/
def weighing (a b : Coin) : Bool :=
  a.weight > b.weight

theorem min_weighings_for_extremes (coins : List Coin) : 
  coins.length = 68 → (∃ n : ℕ, n = 100 ∧ 
    (∀ m : ℕ, m < n → ¬(∃ heaviest lightest : Coin, 
      heaviest ∈ coins ∧ lightest ∈ coins ∧
      (∀ c : Coin, c ∈ coins → c.weight ≤ heaviest.weight) ∧
      (∀ c : Coin, c ∈ coins → c.weight ≥ lightest.weight) ∧
      (heaviest ≠ lightest)))) :=
by
  sorry

end min_weighings_for_extremes_l3916_391659


namespace gcd_lcm_product_360_l3916_391615

theorem gcd_lcm_product_360 (x y : ℕ+) : 
  (Nat.gcd x y * Nat.lcm x y = 360) → 
  (∃ (s : Finset ℕ), s.card = 8 ∧ ∀ (d : ℕ), d ∈ s ↔ ∃ (a b : ℕ+), Nat.gcd a b * Nat.lcm a b = 360 ∧ Nat.gcd a b = d) :=
sorry

end gcd_lcm_product_360_l3916_391615


namespace eeshas_travel_time_l3916_391684

/-- Eesha's travel time problem -/
theorem eeshas_travel_time 
  (usual_time : ℝ) 
  (usual_speed : ℝ) 
  (late_start : ℝ) 
  (late_arrival : ℝ) 
  (speed_reduction : ℝ) 
  (h1 : late_start = 30) 
  (h2 : late_arrival = 50) 
  (h3 : speed_reduction = 0.25) 
  (h4 : usual_time / (usual_time + late_arrival) = (1 - speed_reduction)) :
  usual_time = 150 := by
sorry

end eeshas_travel_time_l3916_391684


namespace target_breaking_orders_l3916_391602

def factorial (n : ℕ) : ℕ := (List.range n).foldl (· * ·) 1

def multinomial (n : ℕ) (ks : List ℕ) : ℕ :=
  factorial n / (ks.map factorial).prod

theorem target_breaking_orders : 
  let n : ℕ := 9
  let ks : List ℕ := [4, 3, 2]
  multinomial n ks = 1260 := by sorry

end target_breaking_orders_l3916_391602


namespace digit_59_is_4_l3916_391674

/-- The decimal representation of 1/17 as a list of digits -/
def decimal_rep_1_17 : List Nat := [0, 5, 8, 8, 2, 3, 5, 2, 9, 4, 1, 1, 7, 6, 4, 7]

/-- The length of the repeating sequence in the decimal representation of 1/17 -/
def cycle_length : Nat := 16

/-- The 59th digit after the decimal point in the decimal representation of 1/17 -/
def digit_59 : Nat := decimal_rep_1_17[(59 - 1) % cycle_length]

theorem digit_59_is_4 : digit_59 = 4 := by sorry

end digit_59_is_4_l3916_391674


namespace rhombus_area_in_square_l3916_391666

/-- The area of a rhombus inscribed in a circle, which is in turn inscribed in a square -/
theorem rhombus_area_in_square (s : ℝ) (h : s = 16) : 
  let r := s / 2
  let d := s
  let rhombus_area := d * d / 2
  rhombus_area = 128 := by sorry

end rhombus_area_in_square_l3916_391666


namespace cauchy_problem_solution_l3916_391607

noncomputable def y (x : ℝ) : ℝ := x^2/2 + x^3/6 + x^4/12 + x^5/20 + x + 1

theorem cauchy_problem_solution (x : ℝ) :
  (deriv^[2] y) x = 1 + x + x^2 + x^3 ∧
  y 0 = 1 ∧
  (deriv y) 0 = 1 := by sorry

end cauchy_problem_solution_l3916_391607


namespace diego_yearly_savings_l3916_391671

/-- Calculates the yearly savings given monthly deposit, monthly expenses, and number of months in a year. -/
def yearly_savings (monthly_deposit : ℕ) (monthly_expenses : ℕ) (months_in_year : ℕ) : ℕ :=
  (monthly_deposit - monthly_expenses) * months_in_year

/-- Theorem stating that Diego's yearly savings is $4,800 -/
theorem diego_yearly_savings :
  yearly_savings 5000 4600 12 = 4800 := by
  sorry

end diego_yearly_savings_l3916_391671


namespace arithmetic_geometric_sequence_property_l3916_391676

/-- An arithmetic-geometric sequence -/
def ArithmeticGeometricSequence (a : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, ∀ n : ℕ, a (n + 1) = a n * q

/-- The sum of the first five terms of a sequence -/
def SumFirstFive (a : ℕ → ℝ) : ℝ :=
  a 1 + a 2 + a 3 + a 4 + a 5

/-- The sum of the squares of the first five terms of a sequence -/
def SumSquaresFirstFive (a : ℕ → ℝ) : ℝ :=
  a 1^2 + a 2^2 + a 3^2 + a 4^2 + a 5^2

/-- The alternating sum of the first five terms of a sequence -/
def AlternatingSumFirstFive (a : ℕ → ℝ) : ℝ :=
  a 1 - a 2 + a 3 - a 4 + a 5

theorem arithmetic_geometric_sequence_property (a : ℕ → ℝ) :
  ArithmeticGeometricSequence a →
  SumFirstFive a = 3 →
  SumSquaresFirstFive a = 12 →
  AlternatingSumFirstFive a = 4 := by
  sorry

end arithmetic_geometric_sequence_property_l3916_391676


namespace arithmetic_mean_of_reciprocals_of_first_four_primes_l3916_391632

def first_four_primes : List Nat := [2, 3, 5, 7]

theorem arithmetic_mean_of_reciprocals_of_first_four_primes :
  let reciprocals := first_four_primes.map (λ x => (1 : ℚ) / x)
  (reciprocals.sum / reciprocals.length) = 247 / 840 := by
  sorry

end arithmetic_mean_of_reciprocals_of_first_four_primes_l3916_391632


namespace min_value_quadratic_form_l3916_391635

theorem min_value_quadratic_form (x y : ℤ) (h : x ≠ 0 ∨ y ≠ 0) :
  |5 * x^2 + 11 * x * y - 5 * y^2| ≥ 5 := by
  sorry

end min_value_quadratic_form_l3916_391635


namespace cosine_sum_problem_l3916_391612

theorem cosine_sum_problem (x y z : ℝ) : 
  x = Real.cos (π / 13) → 
  y = Real.cos (3 * π / 13) → 
  z = Real.cos (9 * π / 13) → 
  x * y + y * z + z * x = -1/4 := by
sorry

end cosine_sum_problem_l3916_391612


namespace adam_has_more_apple_difference_l3916_391694

/-- The number of apples Adam has -/
def adam_apples : ℕ := 9

/-- The number of apples Jackie has -/
def jackie_apples : ℕ := 6

/-- Adam has more apples than Jackie -/
theorem adam_has_more : adam_apples > jackie_apples := by sorry

/-- The difference in apples between Adam and Jackie is 3 -/
theorem apple_difference : adam_apples - jackie_apples = 3 := by sorry

end adam_has_more_apple_difference_l3916_391694


namespace car_original_price_l3916_391692

/-- 
Given a car sale scenario where:
1. A car is sold at a 10% loss to a friend
2. The friend sells it for Rs. 54000 with a 20% gain

This theorem proves that the original cost price of the car was Rs. 50000.
-/
theorem car_original_price : ℝ → Prop :=
  fun original_price =>
    let friend_buying_price := 0.9 * original_price
    let friend_selling_price := 54000
    (1.2 * friend_buying_price = friend_selling_price) →
    (original_price = 50000)

-- The proof is omitted
example : car_original_price 50000 := by sorry

end car_original_price_l3916_391692


namespace circle_c_and_line_theorem_l3916_391699

/-- Circle C with given properties -/
structure CircleC where
  radius : ℝ
  center : ℝ × ℝ
  chord_length : ℝ
  center_below_x_axis : center.2 < 0
  center_on_y_eq_x : center.1 = center.2
  radius_eq_3 : radius = 3
  chord_eq_2root5 : chord_length = 2 * Real.sqrt 5

/-- Line with slope 1 -/
structure Line where
  b : ℝ
  equation : ℝ → ℝ
  slope_eq_1 : ∀ x, equation x = x + b

/-- Theorem about CircleC and related Line -/
theorem circle_c_and_line_theorem (c : CircleC) :
  (∃ x y, (x + 2)^2 + (y + 2)^2 = 9 ↔ (x - c.center.1)^2 + (y - c.center.2)^2 = c.radius^2) ∧
  (∃ l : Line, (l.b = 1 ∨ l.b = -1) ∧
    ∃ x₁ y₁ x₂ y₂, ((x₁ - c.center.1)^2 + (y₁ - c.center.2)^2 = c.radius^2 ∧
                    (x₂ - c.center.1)^2 + (y₂ - c.center.2)^2 = c.radius^2 ∧
                    y₁ = l.equation x₁ ∧ y₂ = l.equation x₂ ∧
                    x₁ * x₂ + y₁ * y₂ = 0)) :=
sorry

end circle_c_and_line_theorem_l3916_391699


namespace square_difference_l3916_391670

theorem square_difference (a b : ℝ) (h1 : a * b = 2) (h2 : a + b = 3) :
  (a - b)^2 = 1 := by sorry

end square_difference_l3916_391670


namespace two_digit_sqrt_prob_l3916_391682

theorem two_digit_sqrt_prob : 
  let two_digit_numbers := Finset.Icc 10 99
  let satisfying_numbers := two_digit_numbers.filter (λ n => n.sqrt < 8)
  (satisfying_numbers.card : ℚ) / two_digit_numbers.card = 3 / 5 := by
sorry

end two_digit_sqrt_prob_l3916_391682


namespace race_head_start_l3916_391634

/-- Proves that the head start in a race is equal to the difference in distances covered by two runners with different speeds in a given time. -/
theorem race_head_start (cristina_speed nicky_speed : ℝ) (race_time : ℝ) 
  (h1 : cristina_speed > nicky_speed) 
  (h2 : cristina_speed = 4)
  (h3 : nicky_speed = 3)
  (h4 : race_time = 36) :
  cristina_speed * race_time - nicky_speed * race_time = 36 := by
  sorry

#check race_head_start

end race_head_start_l3916_391634


namespace pq_length_is_eight_l3916_391630

/-- A quadrilateral with three equal sides -/
structure ThreeEqualSidesQuadrilateral where
  -- The lengths of the four sides
  pq : ℝ
  qr : ℝ
  rs : ℝ
  sp : ℝ
  -- Three sides are equal
  three_equal : pq = qr ∧ pq = sp
  -- SR length is 16
  sr_length : rs = 16
  -- Perimeter is 40
  perimeter : pq + qr + rs + sp = 40

/-- The length of PQ in a ThreeEqualSidesQuadrilateral is 8 -/
theorem pq_length_is_eight (quad : ThreeEqualSidesQuadrilateral) : quad.pq = 8 :=
by sorry

end pq_length_is_eight_l3916_391630


namespace present_age_ratio_l3916_391600

theorem present_age_ratio (R M : ℝ) (h1 : M - R = 7.5) (h2 : (R + 10) / (M + 10) = 2 / 3) 
  (h3 : R > 0) (h4 : M > 0) : R / M = 2 / 5 := by
  sorry

end present_age_ratio_l3916_391600


namespace coral_population_decline_l3916_391665

/-- The yearly decrease rate of the coral population -/
def decrease_rate : ℝ := 0.25

/-- The threshold below which we consider the population critically low -/
def critical_threshold : ℝ := 0.05

/-- The number of years it takes for the population to fall below the critical threshold -/
def years_to_critical : ℕ := 9

/-- The remaining population after n years -/
def population_after (n : ℕ) : ℝ := (1 - decrease_rate) ^ n

theorem coral_population_decline :
  population_after years_to_critical < critical_threshold :=
sorry

end coral_population_decline_l3916_391665


namespace sqrt_eight_div_sqrt_two_equals_two_l3916_391672

theorem sqrt_eight_div_sqrt_two_equals_two : 
  Real.sqrt 8 / Real.sqrt 2 = 2 := by sorry

end sqrt_eight_div_sqrt_two_equals_two_l3916_391672


namespace number_of_students_in_line_l3916_391642

/-- The number of students in a line with specific conditions -/
theorem number_of_students_in_line :
  ∀ (n : ℕ),
  (∃ (eunjung_position yoojung_position : ℕ),
    eunjung_position = 5 ∧
    yoojung_position = n ∧
    yoojung_position - eunjung_position = 9) →
  n = 14 :=
by
  sorry

end number_of_students_in_line_l3916_391642


namespace x_convergence_to_sqrt2_l3916_391678

-- Define the sequence x_n
def x : ℕ → ℚ
| 0 => 1
| (n+1) => 1 + 1 / (2 + 1 / (x n))

-- Define the bound function
def bound (n : ℕ) : ℚ := 1 / 2^(2^n - 1)

-- State the theorem
theorem x_convergence_to_sqrt2 (n : ℕ) :
  |x n - Real.sqrt 2| < bound n :=
sorry

end x_convergence_to_sqrt2_l3916_391678


namespace construct_triangle_from_symmetric_points_l3916_391623

/-- A point in 2D space -/
structure Point :=
  (x : ℝ)
  (y : ℝ)

/-- A triangle defined by three points -/
structure Triangle :=
  (A : Point)
  (B : Point)
  (C : Point)

/-- The orthocenter of a triangle -/
def orthocenter (t : Triangle) : Point := sorry

/-- Check if a triangle is acute-angled -/
def is_acute_angled (t : Triangle) : Prop := sorry

/-- The symmetric point of a given point with respect to a line segment -/
def symmetric_point (p : Point) (a : Point) (b : Point) : Point := sorry

/-- Theorem: Given three points that are symmetric to the orthocenter of an acute-angled triangle
    with respect to its sides, the triangle can be uniquely constructed -/
theorem construct_triangle_from_symmetric_points
  (A' B' C' : Point) :
  ∃! (t : Triangle),
    is_acute_angled t ∧
    A' = symmetric_point (orthocenter t) t.B t.C ∧
    B' = symmetric_point (orthocenter t) t.C t.A ∧
    C' = symmetric_point (orthocenter t) t.A t.B :=
sorry

end construct_triangle_from_symmetric_points_l3916_391623


namespace octagon_side_length_l3916_391680

/-- The side length of a regular octagon with an area equal to the sum of the areas of three regular octagons with side lengths 3, 4, and 12 units is 13 units. -/
theorem octagon_side_length (a b c d : ℝ) : 
  a > 0 → b > 0 → c > 0 → d > 0 →
  (a = 3) → (b = 4) → (c = 12) →
  (d^2 = a^2 + b^2 + c^2) →
  d = 13 := by sorry

end octagon_side_length_l3916_391680


namespace sum_of_possible_numbers_l3916_391631

/-- The original number from which we remove digits -/
def original_number : ℕ := 112277

/-- The set of all possible three-digit numbers obtained by removing three digits from the original number -/
def possible_numbers : Finset ℕ := {112, 117, 122, 127, 177, 227, 277}

/-- The theorem stating that the sum of all possible three-digit numbers is 1159 -/
theorem sum_of_possible_numbers : 
  (possible_numbers.sum id) = 1159 := by sorry

end sum_of_possible_numbers_l3916_391631


namespace wire_cutting_l3916_391601

theorem wire_cutting (total_length : ℝ) (used_parts : ℕ) (unused_length : ℝ) (n : ℕ) :
  total_length = 50 →
  used_parts = 3 →
  unused_length = 20 →
  total_length = n * (total_length - unused_length) / used_parts →
  n = 5 := by
  sorry

end wire_cutting_l3916_391601


namespace zoo_animal_ratio_l3916_391673

/-- Given the animal counts at San Diego Zoo, prove the ratio of bee-eaters to leopards -/
theorem zoo_animal_ratio :
  let total_animals : ℕ := 670
  let snakes : ℕ := 100
  let arctic_foxes : ℕ := 80
  let leopards : ℕ := 20
  let cheetahs : ℕ := snakes / 2
  let alligators : ℕ := 2 * (arctic_foxes + leopards)
  let bee_eaters : ℕ := total_animals - (snakes + arctic_foxes + leopards + cheetahs + alligators)
  (bee_eaters : ℚ) / leopards = 11 / 1 := by
  sorry

end zoo_animal_ratio_l3916_391673


namespace max_students_distribution_l3916_391628

theorem max_students_distribution (pens pencils : ℕ) 
  (h1 : pens = 1001) (h2 : pencils = 910) : ℕ :=
  Nat.gcd pens pencils

#check max_students_distribution

end max_students_distribution_l3916_391628


namespace octal_sum_equality_l3916_391611

/-- Converts a base-8 number to base-10 --/
def octal_to_decimal (n : ℕ) : ℕ := sorry

/-- Converts a base-10 number to base-8 --/
def decimal_to_octal (n : ℕ) : ℕ := sorry

/-- The sum of three octal numbers is equal to another octal number --/
theorem octal_sum_equality : 
  decimal_to_octal (octal_to_decimal 236 + octal_to_decimal 425 + octal_to_decimal 157) = 1042 := by
  sorry

end octal_sum_equality_l3916_391611


namespace quadratic_sum_abc_l3916_391614

/-- The quadratic function f(x) = -4x^2 + 20x + 196 -/
def f (x : ℝ) : ℝ := -4 * x^2 + 20 * x + 196

/-- The sum of a, b, and c when f(x) is expressed as a(x+b)^2 + c -/
def sum_abc : ℝ := 213.5

theorem quadratic_sum_abc :
  ∃ (a b c : ℝ), (∀ x, f x = a * (x + b)^2 + c) ∧ (a + b + c = sum_abc) := by
  sorry

end quadratic_sum_abc_l3916_391614


namespace frame_interior_edges_sum_l3916_391640

/-- Represents a rectangular picture frame -/
structure Frame where
  outer_length : ℝ
  outer_width : ℝ
  frame_width : ℝ

/-- Calculates the area of the frame -/
def frame_area (f : Frame) : ℝ :=
  f.outer_length * f.outer_width - (f.outer_length - 2 * f.frame_width) * (f.outer_width - 2 * f.frame_width)

/-- Calculates the sum of the lengths of the four interior edges of the frame -/
def interior_edges_sum (f : Frame) : ℝ :=
  2 * ((f.outer_length - 2 * f.frame_width) + (f.outer_width - 2 * f.frame_width))

/-- Theorem stating that for a frame with given conditions, the sum of interior edges is 8 inches -/
theorem frame_interior_edges_sum :
  ∀ (f : Frame),
    f.frame_width = 2 →
    f.outer_length = 8 →
    frame_area f = 32 →
    interior_edges_sum f = 8 := by
  sorry

end frame_interior_edges_sum_l3916_391640


namespace reflection_line_sum_l3916_391683

/-- Given a reflection of point (0,1) across line y = mx + b to point (4,5), prove m + b = 4 -/
theorem reflection_line_sum (m b : ℝ) : 
  (∃ (x y : ℝ), x = 4 ∧ y = 5 ∧ 
    ((x - 0) * (x - 0) + (y - 1) * (y - 1)) / 4 = 
    ((x - 0) * (1 + y) / 2 - (y - 1) * (0 + x) / 2)^2 / ((x - 0)^2 + (y - 1)^2) ∧
    y = m * x + b) →
  m + b = 4 := by
sorry

end reflection_line_sum_l3916_391683


namespace positive_sum_reciprocal_inequality_l3916_391643

theorem positive_sum_reciprocal_inequality (p : ℝ) (hp : p > 0) :
  p + 1/p > 2 ↔ p ≠ 1 := by sorry

end positive_sum_reciprocal_inequality_l3916_391643


namespace percentage_increase_l3916_391616

theorem percentage_increase (initial : ℝ) (final : ℝ) : 
  initial = 350 → final = 525 → (final - initial) / initial * 100 = 50 := by
  sorry

end percentage_increase_l3916_391616


namespace pennsylvania_quarters_l3916_391633

theorem pennsylvania_quarters (total : ℕ) (state_fraction : ℚ) (penn_fraction : ℚ) : 
  total = 35 → 
  state_fraction = 2 / 5 → 
  penn_fraction = 1 / 2 → 
  ⌊total * state_fraction * penn_fraction⌋ = 7 := by
  sorry

end pennsylvania_quarters_l3916_391633


namespace solution_set_of_inequality_l3916_391662

noncomputable def f (x : ℝ) : ℝ := Real.exp x + Real.exp (-x) + Real.log (abs x)

theorem solution_set_of_inequality :
  {x : ℝ | f (x + 1) > f (2 * x - 1)} = {x : ℝ | 0 < x ∧ x < 1/2 ∨ 1/2 < x ∧ x < 2} :=
sorry

end solution_set_of_inequality_l3916_391662


namespace polynomial_expansion_l3916_391627

theorem polynomial_expansion (x : ℝ) : 
  (5 * x^2 + 7 * x - 3) * (3 * x^3 + 4) = 
  15 * x^5 + 21 * x^4 - 9 * x^3 + 20 * x^2 + 28 * x - 12 := by
  sorry

end polynomial_expansion_l3916_391627


namespace quadratic_symmetry_l3916_391686

/-- A quadratic function with axis of symmetry at x = 9.5 -/
def p (d e f : ℝ) (x : ℝ) : ℝ := d * x^2 + e * x + f

theorem quadratic_symmetry (d e f : ℝ) :
  (∀ x, p d e f (9.5 + x) = p d e f (9.5 - x)) →  -- axis of symmetry at x = 9.5
  p d e f (-1) = 1 →  -- p(-1) = 1
  ∃ n : ℤ, p d e f 20 = n →  -- p(20) is an integer
  p d e f 20 = 1 := by  -- prove p(20) = 1
sorry

end quadratic_symmetry_l3916_391686


namespace combined_distance_is_1890_l3916_391660

/-- The combined swimming distance for Jamir, Sarah, and Julien for a week -/
def combined_swimming_distance (julien_distance : ℕ) : ℕ :=
  let sarah_distance := 2 * julien_distance
  let jamir_distance := sarah_distance + 20
  let days_in_week := 7
  (julien_distance + sarah_distance + jamir_distance) * days_in_week

/-- Theorem stating that the combined swimming distance for a week is 1890 meters -/
theorem combined_distance_is_1890 :
  combined_swimming_distance 50 = 1890 := by
  sorry

end combined_distance_is_1890_l3916_391660


namespace ice_skate_profit_maximization_l3916_391658

/-- Ice skate problem -/
theorem ice_skate_profit_maximization
  (cost_A cost_B : ℕ)  -- Cost prices of type A and B
  (sell_A sell_B : ℕ)  -- Selling prices of type A and B
  (total_pairs : ℕ)    -- Total number of pairs to purchase
  : cost_B = 2 * cost_A  -- Condition 1
  → 2 * cost_A + cost_B = 920  -- Condition 2
  → sell_A = 400  -- Condition 3
  → sell_B = 560  -- Condition 4
  → total_pairs = 50  -- Condition 5
  → (∀ x y : ℕ, x + y = total_pairs → x ≤ 2 * y)  -- Condition 6
  → ∃ (x y : ℕ),
      x + y = total_pairs ∧
      x = 33 ∧
      y = 17 ∧
      x * (sell_A - cost_A) + y * (sell_B - cost_B) = 6190 ∧
      ∀ (a b : ℕ), a + b = total_pairs →
        a * (sell_A - cost_A) + b * (sell_B - cost_B) ≤ 6190 :=
by sorry

end ice_skate_profit_maximization_l3916_391658


namespace zoo_field_trip_vans_l3916_391620

/-- The number of vans needed for a field trip --/
def vans_needed (van_capacity : ℕ) (num_students : ℕ) (num_adults : ℕ) : ℕ :=
  (num_students + num_adults + van_capacity - 1) / van_capacity

/-- Theorem: The number of vans needed for the zoo field trip is 6 --/
theorem zoo_field_trip_vans : vans_needed 5 25 5 = 6 := by
  sorry

end zoo_field_trip_vans_l3916_391620


namespace contractor_payment_l3916_391698

/-- Calculates the total amount a contractor receives given the contract terms and attendance. -/
def contractorPay (totalDays : ℕ) (payPerDay : ℚ) (finePerDay : ℚ) (absentDays : ℕ) : ℚ :=
  let workDays := totalDays - absentDays
  let totalPay := (workDays : ℚ) * payPerDay
  let totalFine := (absentDays : ℚ) * finePerDay
  totalPay - totalFine

/-- Proves that under the given conditions, the contractor receives Rs. 425. -/
theorem contractor_payment :
  contractorPay 30 25 7.50 10 = 425 := by
  sorry

end contractor_payment_l3916_391698


namespace solution_set_inequality_l3916_391689

theorem solution_set_inequality (x : ℝ) : 
  (x ≠ -1 ∧ (2*x - 1)/(x + 1) ≤ 1) ↔ -1 < x ∧ x ≤ 2 :=
by sorry

end solution_set_inequality_l3916_391689


namespace percentage_of_returned_books_l3916_391649

/-- Given a library's special collection with initial and final book counts,
    and the number of books loaned out, prove the percentage of returned books. -/
theorem percentage_of_returned_books
  (initial_books : ℕ)
  (final_books : ℕ)
  (loaned_books : ℕ)
  (h1 : initial_books = 75)
  (h2 : final_books = 69)
  (h3 : loaned_books = 30) :
  (initial_books - final_books : ℚ) / loaned_books * 100 = 20 := by
  sorry

#check percentage_of_returned_books

end percentage_of_returned_books_l3916_391649


namespace function_value_at_two_l3916_391645

/-- Given a function f(x) = x^5 + px^3 + qx - 8 where f(-2) = 10, prove that f(2) = -26 -/
theorem function_value_at_two (p q : ℝ) (f : ℝ → ℝ) 
  (h1 : ∀ x, f x = x^5 + p*x^3 + q*x - 8)
  (h2 : f (-2) = 10) : 
  f 2 = -26 := by
  sorry

end function_value_at_two_l3916_391645


namespace sin_160_eq_sin_20_l3916_391688

theorem sin_160_eq_sin_20 : Real.sin (160 * π / 180) = Real.sin (20 * π / 180) := by
  sorry

end sin_160_eq_sin_20_l3916_391688
