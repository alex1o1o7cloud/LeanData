import Mathlib

namespace proportions_sum_l2985_298580

theorem proportions_sum (x y : ℚ) :
  (4 : ℚ) / 7 = x / 63 ∧ (4 : ℚ) / 7 = 84 / y → x + y = 183 := by
  sorry

end proportions_sum_l2985_298580


namespace committee_selection_l2985_298513

theorem committee_selection (n : ℕ) (h : Nat.choose n 3 = 20) : Nat.choose n 3 = 20 := by
  sorry

end committee_selection_l2985_298513


namespace smallest_congruent_difference_l2985_298520

/-- The smallest positive four-digit integer congruent to 7 (mod 13) -/
def p : ℕ := sorry

/-- The smallest positive five-digit integer congruent to 7 (mod 13) -/
def q : ℕ := sorry

theorem smallest_congruent_difference : q - p = 8996 := by sorry

end smallest_congruent_difference_l2985_298520


namespace z_in_fourth_quadrant_l2985_298581

-- Define the complex number i
def i : ℂ := Complex.I

-- Define the equation z(1+i) = 2i
def equation (z : ℂ) : Prop := z * (1 + i) = 2 * i

-- Define the fourth quadrant
def fourth_quadrant (z : ℂ) : Prop :=
  z.re > 0 ∧ z.im < 0

-- Theorem statement
theorem z_in_fourth_quadrant :
  ∃ z : ℂ, equation z ∧ fourth_quadrant z :=
sorry

end z_in_fourth_quadrant_l2985_298581


namespace unique_polynomial_with_given_value_l2985_298543

/-- A polynomial with natural number coefficients less than 10 -/
def PolynomialWithSmallCoeffs (p : Polynomial ℕ) : Prop :=
  ∀ i, (p.coeff i) < 10

theorem unique_polynomial_with_given_value :
  ∀ p : Polynomial ℕ,
  PolynomialWithSmallCoeffs p →
  p.eval 10 = 1248 →
  p = Polynomial.monomial 3 1 + Polynomial.monomial 2 2 + Polynomial.monomial 1 4 + Polynomial.monomial 0 8 :=
by sorry

end unique_polynomial_with_given_value_l2985_298543


namespace limit_proof_l2985_298589

/-- The limit of (2 - e^(arcsin^2(√x)))^(3/x) as x approaches 0 is e^(-3) -/
theorem limit_proof : ∀ ε > 0, ∃ δ > 0, ∀ x : ℝ, 0 < |x| ∧ |x| < δ →
  |(2 - Real.exp (Real.arcsin (Real.sqrt x))^2)^(3/x) - Real.exp (-3)| < ε :=
sorry

end limit_proof_l2985_298589


namespace f_even_and_increasing_l2985_298502

-- Define the function f(x) = |x| + 1
def f (x : ℝ) : ℝ := |x| + 1

-- Theorem statement
theorem f_even_and_increasing :
  (∀ x : ℝ, f (-x) = f x) ∧
  (∀ x y : ℝ, 0 < x → x < y → f x < f y) :=
by sorry

end f_even_and_increasing_l2985_298502


namespace percentage_of_x_l2985_298573

theorem percentage_of_x (x y : ℝ) (P : ℝ) : 
  (P / 100) * x = (20 / 100) * y →
  x / y = 2 →
  P = 10 := by
sorry

end percentage_of_x_l2985_298573


namespace area_of_CALI_l2985_298539

/-- Square BERK with side length 10 -/
def BERK : Set (ℝ × ℝ) := sorry

/-- Points T, O, W, N as midpoints of BE, ER, RK, KB respectively -/
def T : ℝ × ℝ := sorry
def O : ℝ × ℝ := sorry
def W : ℝ × ℝ := sorry
def N : ℝ × ℝ := sorry

/-- Square CALI whose edges contain vertices of BERK -/
def CALI : Set (ℝ × ℝ) := sorry

/-- CA is parallel to BO -/
def CA_parallel_BO : Prop := sorry

/-- The area of a set in ℝ² -/
def area (s : Set (ℝ × ℝ)) : ℝ := sorry

theorem area_of_CALI : area CALI = 180 :=
sorry

end area_of_CALI_l2985_298539


namespace square_not_always_positive_l2985_298578

theorem square_not_always_positive : ¬ (∀ x : ℝ, x^2 > 0) := by
  sorry

end square_not_always_positive_l2985_298578


namespace cost_reduction_equation_l2985_298512

theorem cost_reduction_equation (x : ℝ) : 
  (∀ (total_reduction : ℝ), total_reduction = 0.36 → 
    ((1 - x) ^ 2 = 1 - total_reduction)) ↔ 
  ((1 - x) ^ 2 = 1 - 0.36) :=
sorry

end cost_reduction_equation_l2985_298512


namespace factorization_proof_l2985_298592

theorem factorization_proof (a b : ℝ) : -a^3 + 12*a^2*b - 36*a*b^2 = -a*(a-6*b)^2 := by
  sorry

end factorization_proof_l2985_298592


namespace parallelogram_area_l2985_298550

/-- Given two 2D vectors u and z, this theorem proves that the area of the parallelogram
    formed by u and z + u is 3. -/
theorem parallelogram_area (u z : Fin 2 → ℝ) (hu : u = ![4, -1]) (hz : z = ![9, -3]) :
  let z' := z + u
  abs (u 0 * z' 1 - u 1 * z' 0) = 3 := by
  sorry

end parallelogram_area_l2985_298550


namespace prop_2_correct_prop_4_correct_prop_1_not_necessarily_true_prop_3_not_necessarily_true_l2985_298549

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relations
variable (parallel_line_plane : Line → Plane → Prop)
variable (perpendicular_line_plane : Line → Plane → Prop)
variable (parallel_lines : Line → Line → Prop)
variable (perpendicular_lines : Line → Line → Prop)
variable (perpendicular_planes : Plane → Plane → Prop)

-- Proposition 2
theorem prop_2_correct 
  (a b : Line) (α : Plane) 
  (h1 : parallel_line_plane a α) 
  (h2 : perpendicular_line_plane b α) : 
  perpendicular_lines a b :=
sorry

-- Proposition 4
theorem prop_4_correct 
  (a : Line) (α β : Plane) 
  (h1 : perpendicular_line_plane a α) 
  (h2 : parallel_line_plane a β) : 
  perpendicular_planes α β :=
sorry

-- Proposition 1 is not necessarily true
theorem prop_1_not_necessarily_true :
  ¬ ∀ (a b : Line) (α β : Plane),
    parallel_line_plane a α → parallel_line_plane b β → parallel_lines a b :=
sorry

-- Proposition 3 is not necessarily true
theorem prop_3_not_necessarily_true :
  ¬ ∀ (a b : Line) (α : Plane),
    parallel_lines a b → parallel_line_plane b α → parallel_line_plane a α :=
sorry

end prop_2_correct_prop_4_correct_prop_1_not_necessarily_true_prop_3_not_necessarily_true_l2985_298549


namespace wilson_cola_purchase_wilson_cola_purchase_correct_l2985_298527

theorem wilson_cola_purchase (hamburger_cost : ℕ) (total_cost : ℕ) (discount : ℕ) (cola_cost : ℕ) : ℕ :=
  let hamburgers := 2
  let hamburger_total := hamburgers * hamburger_cost
  let discounted_hamburger_cost := hamburger_total - discount
  let cola_total := total_cost - discounted_hamburger_cost
  cola_total / cola_cost

#check wilson_cola_purchase 5 12 4 2

theorem wilson_cola_purchase_correct : wilson_cola_purchase 5 12 4 2 = 3 := by
  sorry

end wilson_cola_purchase_wilson_cola_purchase_correct_l2985_298527


namespace sin_negative_120_degrees_l2985_298575

theorem sin_negative_120_degrees : Real.sin (-(2 * π / 3)) = Real.sqrt 3 / 2 := by
  sorry

end sin_negative_120_degrees_l2985_298575


namespace count_even_four_digit_is_784_l2985_298586

/-- Count of even integers between 3000 and 6000 with four different digits -/
def count_even_four_digit : ℕ := sorry

/-- An integer is between 3000 and 6000 -/
def is_between_3000_and_6000 (n : ℕ) : Prop :=
  3000 < n ∧ n < 6000

/-- An integer has four different digits -/
def has_four_different_digits (n : ℕ) : Prop := sorry

/-- Theorem stating that the count of even integers between 3000 and 6000
    with four different digits is 784 -/
theorem count_even_four_digit_is_784 :
  count_even_four_digit = 784 := by sorry

end count_even_four_digit_is_784_l2985_298586


namespace bronson_leaf_collection_l2985_298595

theorem bronson_leaf_collection (thursday_leaves : ℕ) (yellow_leaves : ℕ) 
  (h1 : thursday_leaves = 12)
  (h2 : yellow_leaves = 15)
  (h3 : yellow_leaves = (3 / 5 : ℚ) * (thursday_leaves + friday_leaves)) :
  friday_leaves = 13 := by
  sorry

end bronson_leaf_collection_l2985_298595


namespace k_range_l2985_298556

-- Define the propositions p and q
def p (x k : ℝ) : Prop := x ≥ k
def q (x : ℝ) : Prop := x^2 - x > 2

-- Define what it means for p to be sufficient but not necessary for q
def sufficient_not_necessary (k : ℝ) : Prop :=
  (∀ x, p x k → q x) ∧ (∃ x, q x ∧ ¬p x k)

-- Theorem statement
theorem k_range :
  ∀ k : ℝ, sufficient_not_necessary k ↔ k > 2 :=
sorry

end k_range_l2985_298556


namespace day_200_N_minus_1_is_wednesday_l2985_298568

/-- Represents the days of the week -/
inductive DayOfWeek
  | Sunday
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday
  | Saturday

/-- Represents a specific day in a year -/
structure YearDay where
  year : Int
  day : Nat

/-- Function to determine the day of the week for a given YearDay -/
def dayOfWeek (yd : YearDay) : DayOfWeek := sorry

theorem day_200_N_minus_1_is_wednesday 
  (N : Int)
  (h1 : dayOfWeek ⟨N, 400⟩ = DayOfWeek.Wednesday)
  (h2 : dayOfWeek ⟨N + 2, 300⟩ = DayOfWeek.Wednesday) :
  dayOfWeek ⟨N - 1, 200⟩ = DayOfWeek.Wednesday := by
  sorry

end day_200_N_minus_1_is_wednesday_l2985_298568


namespace homework_completion_l2985_298516

theorem homework_completion (total : ℕ) (math : ℕ) (korean : ℕ) 
  (h1 : total = 48) 
  (h2 : math = 37) 
  (h3 : korean = 42) 
  (h4 : math + korean - total ≥ 0) : 
  math + korean - total = 31 := by
  sorry

end homework_completion_l2985_298516


namespace special_function_at_one_seventh_l2985_298504

/-- A function satisfying the given conditions -/
def special_function (f : ℝ → ℝ) : Prop :=
  (∀ x, x ∈ Set.Icc 0 1 → f x ∈ Set.Icc 0 1) ∧
  f 0 = 0 ∧ f 1 = 1 ∧
  ∃ a : ℝ, 0 ≤ a ∧ a ≤ 1 ∧
    ∀ x y, x ∈ Set.Icc 0 1 → y ∈ Set.Icc 0 1 → x ≤ y →
      f ((x + y) / 2) = (1 - a) * f x + a * f y

theorem special_function_at_one_seventh (f : ℝ → ℝ) (h : special_function f) :
  f (1/7) = 1/7 := by
  sorry

end special_function_at_one_seventh_l2985_298504


namespace line_vector_proof_l2985_298598

def line_vector (t : ℝ) : ℝ × ℝ × ℝ := sorry

theorem line_vector_proof :
  (line_vector 0 = (1, 5, 9)) →
  (line_vector 1 = (6, 0, 4)) →
  (line_vector 4 = (21, -15, -11)) := by
  sorry

end line_vector_proof_l2985_298598


namespace line_hyperbola_intersection_range_l2985_298567

/-- The range of k for which the line y = kx intersects the hyperbola x^2/9 - y^2/4 = 1 -/
theorem line_hyperbola_intersection_range :
  ∀ k : ℝ, 
  (∃ x y : ℝ, y = k * x ∧ x^2 / 9 - y^2 / 4 = 1) ↔ 
  -2/3 < k ∧ k < 2/3 :=
by sorry

end line_hyperbola_intersection_range_l2985_298567


namespace line_ellipse_intersection_condition_l2985_298564

/-- The range of m for which the line y = kx + 1 always intersects 
    with the ellipse x²/5 + y²/m = 1 for any real k -/
theorem line_ellipse_intersection_condition (k : ℝ) :
  (∀ x y : ℝ, y = k * x + 1 → x^2 / 5 + y^2 / m = 1) ↔ (m ≥ 1 ∧ m ≠ 5) :=
sorry

end line_ellipse_intersection_condition_l2985_298564


namespace sandwich_combinations_l2985_298536

theorem sandwich_combinations (meat_types : ℕ) (cheese_types : ℕ) (condiment_types : ℕ) :
  meat_types = 12 →
  cheese_types = 11 →
  condiment_types = 5 →
  (meat_types * Nat.choose cheese_types 2 * (condiment_types + 1)) = 3960 :=
by sorry

end sandwich_combinations_l2985_298536


namespace exactly_seven_numbers_l2985_298548

def is_two_digit (n : ℕ) : Prop := 10 ≤ n ∧ n < 100

def swap_digits (n : ℕ) : ℕ :=
  (n % 10) * 10 + (n / 10)

def is_perfect_cube (n : ℤ) : Prop :=
  ∃ m : ℤ, n = m^3

def satisfies_condition (n : ℕ) : Prop :=
  is_two_digit n ∧ is_perfect_cube (n - swap_digits n)

theorem exactly_seven_numbers :
  ∃! (s : Finset ℕ), s.card = 7 ∧ ∀ n, n ∈ s ↔ satisfies_condition n :=
sorry

end exactly_seven_numbers_l2985_298548


namespace unique_solution_abs_equation_l2985_298566

theorem unique_solution_abs_equation :
  ∃! x : ℝ, |x - 2| = |x - 3| + |x - 4| + |x - 5| := by
  sorry

end unique_solution_abs_equation_l2985_298566


namespace money_left_after_purchase_l2985_298561

def initial_amount : ℕ := 120
def hamburger_cost : ℕ := 4
def milkshake_cost : ℕ := 3
def hamburgers_bought : ℕ := 8
def milkshakes_bought : ℕ := 6

theorem money_left_after_purchase : 
  initial_amount - (hamburger_cost * hamburgers_bought + milkshake_cost * milkshakes_bought) = 70 := by
  sorry

end money_left_after_purchase_l2985_298561


namespace decagon_perimeter_decagon_perimeter_30_l2985_298533

/-- The perimeter of a regular decagon with side length 3 units is 30 units. -/
theorem decagon_perimeter : ℝ → ℝ → ℝ → Prop :=
  fun (num_sides : ℝ) (side_length : ℝ) (perimeter : ℝ) =>
    num_sides = 10 ∧ side_length = 3 → perimeter = num_sides * side_length

/-- The theorem applied to our specific case. -/
theorem decagon_perimeter_30 : decagon_perimeter 10 3 30 := by
  sorry

end decagon_perimeter_decagon_perimeter_30_l2985_298533


namespace junipers_bones_l2985_298530

theorem junipers_bones (initial_bones : ℕ) : 
  (2 * initial_bones - 2 = 6) → initial_bones = 4 := by
  sorry

end junipers_bones_l2985_298530


namespace prism_volume_l2985_298510

/-- The volume of a right rectangular prism with specific face areas and dimension ratio -/
theorem prism_volume (l w h : ℝ) : 
  l > 0 → w > 0 → h > 0 →
  l * w = 10 → w * h = 18 → l * h = 36 →
  l = 2 * w →
  l * w * h = 36 * Real.sqrt 5 := by
sorry

end prism_volume_l2985_298510


namespace g_five_equals_248_l2985_298574

theorem g_five_equals_248 (g : ℤ → ℤ) 
  (h1 : g 1 > 1)
  (h2 : ∀ x y : ℤ, g (x + y) + x * g y + y * g x = g x * g y + x + y + x * y)
  (h3 : ∀ x : ℤ, 3 * g x = g (x + 1) + 2 * x - 1) :
  g 5 = 248 := by sorry

end g_five_equals_248_l2985_298574


namespace f_monotone_increasing_on_interval_l2985_298500

/-- The function f(x) = (1/2)^(x^2 - x - 1) is monotonically increasing on (-∞, 1/2) -/
theorem f_monotone_increasing_on_interval :
  ∀ x y : ℝ, x < y → x < (1/2 : ℝ) → y < (1/2 : ℝ) →
  ((1/2 : ℝ) ^ (x^2 - x - 1)) < ((1/2 : ℝ) ^ (y^2 - y - 1)) :=
sorry

end f_monotone_increasing_on_interval_l2985_298500


namespace prob_sum_five_is_one_ninth_l2985_298506

/-- The number of faces on each die -/
def faces : ℕ := 6

/-- The total number of possible outcomes when rolling two dice -/
def total_outcomes : ℕ := faces * faces

/-- The number of favorable outcomes (sum of 5) when rolling two dice -/
def favorable_outcomes : ℕ := 4

/-- The probability of rolling a sum of 5 with two dice -/
def prob_sum_five : ℚ := favorable_outcomes / total_outcomes

theorem prob_sum_five_is_one_ninth :
  prob_sum_five = 1 / 9 := by sorry

end prob_sum_five_is_one_ninth_l2985_298506


namespace f_max_value_l2985_298599

/-- The quadratic function f(x) = -9x^2 + 27x + 15 -/
def f (x : ℝ) : ℝ := -9 * x^2 + 27 * x + 15

/-- The maximum value of f(x) is 35.25 -/
theorem f_max_value : ∃ (M : ℝ), M = 35.25 ∧ ∀ (x : ℝ), f x ≤ M := by
  sorry

end f_max_value_l2985_298599


namespace smallest_additional_airplanes_lucas_airplanes_arrangement_l2985_298591

theorem smallest_additional_airplanes (current_airplanes : ℕ) (row_size : ℕ) : ℕ :=
  let next_multiple := (current_airplanes + row_size - 1) / row_size * row_size
  next_multiple - current_airplanes

theorem lucas_airplanes_arrangement :
  smallest_additional_airplanes 37 8 = 3 := by
  sorry

end smallest_additional_airplanes_lucas_airplanes_arrangement_l2985_298591


namespace exists_term_between_zero_and_one_l2985_298514

/-- An infinite sequence satisfying a_{n+2} = |a_{n+1} - a_n| -/
def SpecialSequence (a : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, a (n + 2) = |a (n + 1) - a n|

/-- Theorem: For any special sequence, there exists a term between 0 and 1 -/
theorem exists_term_between_zero_and_one (a : ℕ → ℝ) (h : SpecialSequence a) :
    ∃ k : ℕ, 0 ≤ a k ∧ a k < 1 := by
  sorry

end exists_term_between_zero_and_one_l2985_298514


namespace rectangle_area_change_l2985_298529

theorem rectangle_area_change 
  (l w : ℝ) 
  (h_pos_l : l > 0) 
  (h_pos_w : w > 0) : 
  let new_length := 1.1 * l
  let new_width := 0.9 * w
  let new_area := new_length * new_width
  let original_area := l * w
  new_area / original_area = 0.99 := by
sorry

end rectangle_area_change_l2985_298529


namespace compute_expression_l2985_298545

theorem compute_expression : 6^3 - 5*7 + 2^4 = 197 := by sorry

end compute_expression_l2985_298545


namespace simplify_fraction_l2985_298524

theorem simplify_fraction (b : ℝ) (h1 : b ≠ -1) (h2 : b ≠ -1/2) :
  1 - 1 / (1 + b / (1 + b)) = b / (1 + 2*b) := by
  sorry

end simplify_fraction_l2985_298524


namespace tonys_walking_speed_l2985_298509

/-- Proves that Tony's walking speed is 2 MPH given the problem conditions -/
theorem tonys_walking_speed :
  let store_distance : ℝ := 4
  let running_speed : ℝ := 10
  let average_time_minutes : ℝ := 56
  let walking_speed : ℝ := 2

  (walking_speed * store_distance + 2 * (store_distance / running_speed) * 60) / 3 = average_time_minutes
  ∧ walking_speed > 0 := by sorry

end tonys_walking_speed_l2985_298509


namespace consecutive_seating_theorem_l2985_298522

/-- Represents the number of people at the table -/
def total_people : ℕ := 12

/-- Represents the number of math majors -/
def math_majors : ℕ := 5

/-- Represents the number of physics majors -/
def physics_majors : ℕ := 4

/-- Represents the number of biology majors -/
def biology_majors : ℕ := 3

/-- The probability of math and physics majors sitting consecutively -/
def consecutive_seating_probability : ℚ := 7 / 240

theorem consecutive_seating_theorem :
  let total := total_people
  let math := math_majors
  let physics := physics_majors
  let bio := biology_majors
  total = math + physics + bio →
  consecutive_seating_probability = 7 / 240 :=
by sorry

end consecutive_seating_theorem_l2985_298522


namespace unique_x_with_three_prime_divisors_l2985_298537

theorem unique_x_with_three_prime_divisors (x n : ℕ) : 
  x = 9^n - 1 →
  (∃ p q r : ℕ, Prime p ∧ Prime q ∧ Prime r ∧ p ≠ q ∧ q ≠ r ∧ p ≠ r ∧ x = p * q * r) →
  13 ∣ x →
  x = 728 :=
sorry

end unique_x_with_three_prime_divisors_l2985_298537


namespace hotel_bill_problem_l2985_298515

theorem hotel_bill_problem (total_bill : ℕ) (equal_share : ℕ) (extra_payment : ℕ) (num_paying_80 : ℕ) :
  (num_paying_80 = 7) →
  (80 * num_paying_80 + 160 = total_bill) →
  (equal_share + 70 = 160) →
  (total_bill / equal_share = 8) :=
by
  sorry

#check hotel_bill_problem

end hotel_bill_problem_l2985_298515


namespace negation_equivalence_l2985_298563

-- Define the original statement
def P : Prop := ∀ n : ℤ, 3 ∣ n → Odd n

-- Define the correct negation
def not_P : Prop := ∃ n : ℤ, 3 ∣ n ∧ ¬(Odd n)

-- Theorem stating that not_P is indeed the negation of P
theorem negation_equivalence : ¬P ↔ not_P := by sorry

end negation_equivalence_l2985_298563


namespace locus_general_case_locus_special_case_l2985_298526

-- Define the triangle PQR
def Triangle (P Q R : ℝ × ℝ) : Prop := sorry

-- Define a point inside a triangle
def InsideTriangle (S : ℝ × ℝ) (P Q R : ℝ × ℝ) : Prop := sorry

-- Define a segment on a side of a triangle
def SegmentOnSide (A B : ℝ × ℝ) (P Q : ℝ × ℝ) : Prop := sorry

-- Define the area of a triangle
def AreaTriangle (A B C : ℝ × ℝ) : ℝ := sorry

-- Define a line segment parallel to another line segment
def ParallelSegment (A B C D : ℝ × ℝ) : Prop := sorry

-- Define the locus of points
def Locus (S : ℝ × ℝ) (P Q R : ℝ × ℝ) (A B C D E F : ℝ × ℝ) (S₀ : ℝ × ℝ) : Prop :=
  InsideTriangle S P Q R ∧
  AreaTriangle S A B + AreaTriangle S C D + AreaTriangle S E F =
  AreaTriangle S₀ A B + AreaTriangle S₀ C D + AreaTriangle S₀ E F

-- Theorem for the general case
theorem locus_general_case 
  (P Q R : ℝ × ℝ) 
  (A B C D E F : ℝ × ℝ) 
  (S₀ : ℝ × ℝ) 
  (h1 : Triangle P Q R)
  (h2 : SegmentOnSide A B P Q)
  (h3 : SegmentOnSide C D Q R)
  (h4 : SegmentOnSide E F R P)
  (h5 : InsideTriangle S₀ P Q R) :
  ∃ D' E' : ℝ × ℝ, 
    ParallelSegment D' E' C D ∧ 
    (∀ S : ℝ × ℝ, Locus S P Q R A B C D E F S₀ ↔ 
      (S = S₀ ∨ ParallelSegment S S₀ D' E')) :=
sorry

-- Theorem for the special case
theorem locus_special_case
  (P Q R : ℝ × ℝ) 
  (A B C D E F : ℝ × ℝ) 
  (S₀ : ℝ × ℝ) 
  (h1 : Triangle P Q R)
  (h2 : SegmentOnSide A B P Q)
  (h3 : SegmentOnSide C D Q R)
  (h4 : SegmentOnSide E F R P)
  (h5 : InsideTriangle S₀ P Q R)
  (h6 : ∃ k : ℝ, k > 0 ∧ 
    ‖A - B‖ / ‖P - Q‖ = k ∧ 
    ‖C - D‖ / ‖Q - R‖ = k ∧ 
    ‖E - F‖ / ‖R - P‖ = k) :
  ∀ S : ℝ × ℝ, InsideTriangle S P Q R → Locus S P Q R A B C D E F S₀ :=
sorry

end locus_general_case_locus_special_case_l2985_298526


namespace charlie_calculation_l2985_298565

theorem charlie_calculation (x : ℝ) : 
  (x / 7 + 20 = 21) → (x * 7 - 20 = 29) := by
sorry

end charlie_calculation_l2985_298565


namespace largest_n_with_unique_k_l2985_298501

theorem largest_n_with_unique_k : 
  ∀ n : ℕ+, n ≤ 112 ↔ 
    (∃! k : ℤ, (8 : ℚ)/15 < (n : ℚ)/(n + k) ∧ (n : ℚ)/(n + k) < 7/13) :=
by sorry

end largest_n_with_unique_k_l2985_298501


namespace outfit_combinations_l2985_298585

/-- The number of red shirts -/
def red_shirts : ℕ := 8

/-- The number of green shirts -/
def green_shirts : ℕ := 7

/-- The number of blue pants -/
def blue_pants : ℕ := 8

/-- The number of red hats -/
def red_hats : ℕ := 10

/-- The number of green hats -/
def green_hats : ℕ := 9

/-- The number of black belts -/
def black_belts : ℕ := 5

/-- The number of brown belts -/
def brown_belts : ℕ := 4

/-- The total number of possible outfits -/
def total_outfits : ℕ := red_shirts * blue_pants * green_hats * brown_belts + 
                         green_shirts * blue_pants * red_hats * black_belts

theorem outfit_combinations : total_outfits = 5104 := by
  sorry

end outfit_combinations_l2985_298585


namespace problem_solution_l2985_298558

theorem problem_solution (x y : ℝ) : 
  (65 / 100 : ℝ) * 900 = (40 / 100 : ℝ) * x → 
  (35 / 100 : ℝ) * 1200 = (25 / 100 : ℝ) * y → 
  x + y = 3142.5 := by
  sorry

end problem_solution_l2985_298558


namespace car_distance_traveled_l2985_298571

theorem car_distance_traveled (time : ℝ) (speed : ℝ) (distance : ℝ) :
  time = 11 →
  speed = 65 →
  distance = speed * time →
  distance = 715 := by sorry

end car_distance_traveled_l2985_298571


namespace difference_of_squares_305_295_l2985_298508

theorem difference_of_squares_305_295 : 305^2 - 295^2 = 6000 := by
  sorry

end difference_of_squares_305_295_l2985_298508


namespace polygon_area_is_six_l2985_298590

/-- The vertices of the polygon -/
def vertices : List (ℤ × ℤ) := [
  (0, 0), (0, 2), (1, 2), (2, 3), (2, 2), (3, 2), (3, 0), (2, 0), (2, 1), (1, 0)
]

/-- Calculate the area of a polygon given its vertices using the Shoelace formula -/
def polygonArea (vs : List (ℤ × ℤ)) : ℚ :=
  let pairs := vs.zip (vs.rotate 1)
  let sum := pairs.foldl (fun acc (p, q) => acc + (p.1 * q.2 - p.2 * q.1)) 0
  (sum.natAbs : ℚ) / 2

/-- The theorem stating that the area of the given polygon is 6 square units -/
theorem polygon_area_is_six :
  polygonArea vertices = 6 := by sorry

end polygon_area_is_six_l2985_298590


namespace floor_of_e_equals_two_l2985_298517

-- Define e as the base of natural logarithms
noncomputable def e : ℝ := Real.exp 1

-- Theorem statement
theorem floor_of_e_equals_two : ⌊e⌋ = 2 := by
  sorry

end floor_of_e_equals_two_l2985_298517


namespace proper_subsets_without_two_eq_l2985_298577

def S : Set ℕ := {1, 2, 3, 4}

def proper_subsets_without_two : Set (Set ℕ) :=
  {A | A ⊂ S ∧ 2 ∉ A}

theorem proper_subsets_without_two_eq :
  proper_subsets_without_two = {∅, {1}, {3}, {4}, {1, 3}, {1, 4}, {3, 4}, {1, 3, 4}} := by
  sorry

end proper_subsets_without_two_eq_l2985_298577


namespace min_value_of_f_l2985_298544

-- Define the function f
def f (x : ℝ) : ℝ := 3*x - 4*x^3

-- State the theorem
theorem min_value_of_f :
  ∃ (x : ℝ), x ∈ Set.Icc 0 1 ∧
  (∀ y ∈ Set.Icc 0 1, f y ≥ f x) ∧
  f x = -1 := by
  sorry

end min_value_of_f_l2985_298544


namespace special_rectangle_side_lengths_l2985_298555

/-- A rectangle with specific properties -/
structure SpecialRectangle where
  AB : ℝ  -- Length of side AB
  BC : ℝ  -- Length of side BC
  ratio_condition : AB / BC = 7 / 5
  square_area : ℝ  -- Area of the common square
  square_area_value : square_area = 72

/-- Theorem stating the side lengths of the special rectangle -/
theorem special_rectangle_side_lengths (rect : SpecialRectangle) : 
  rect.AB = 42 ∧ rect.BC = 30 := by
  sorry

end special_rectangle_side_lengths_l2985_298555


namespace divisibility_equivalence_l2985_298534

theorem divisibility_equivalence (m n : ℕ+) :
  (6 * m.val ∣ (2 * m.val + 3)^n.val + 1) ↔ (4 * m.val ∣ 3^n.val + 1) := by
  sorry

end divisibility_equivalence_l2985_298534


namespace quadratic_inequality_solution_sets_l2985_298570

/-- Given that the solution set of ax² + bx + c > 0 is {x | 1 < x < 2},
    prove that the solution set of cx² + bx + a < 0 is {x | x < 1/2 or x > 1} -/
theorem quadratic_inequality_solution_sets 
  (a b c : ℝ) 
  (h : ∀ x : ℝ, ax^2 + b*x + c > 0 ↔ 1 < x ∧ x < 2) :
  ∀ x : ℝ, c*x^2 + b*x + a < 0 ↔ x < 1/2 ∨ x > 1 := by
  sorry

end quadratic_inequality_solution_sets_l2985_298570


namespace sarah_marriage_age_l2985_298560

/-- The game that predicts marriage age based on name, age, birth month, and siblings' ages -/
def marriage_age_prediction 
  (name_length : ℕ) 
  (age : ℕ) 
  (birth_month : ℕ) 
  (sibling_ages : List ℕ) : ℕ :=
  let step1 := name_length + 2 * age
  let step2 := step1 * (sibling_ages.sum)
  let step3 := step2 / (sibling_ages.length)
  step3 * birth_month

/-- Theorem stating that Sarah's predicted marriage age is 966 -/
theorem sarah_marriage_age : 
  marriage_age_prediction 5 9 7 [5, 7] = 966 := by
  sorry

#eval marriage_age_prediction 5 9 7 [5, 7]

end sarah_marriage_age_l2985_298560


namespace arrangements_with_pair_together_eq_48_l2985_298505

/-- The number of ways to arrange n distinct objects -/
def factorial (n : ℕ) : ℕ := (List.range n).foldl (· * ·) 1

/-- The number of ways to arrange five people in a row, with two specific people always standing together -/
def arrangements_with_pair_together : ℕ :=
  factorial 4 * factorial 2

theorem arrangements_with_pair_together_eq_48 :
  arrangements_with_pair_together = 48 := by
  sorry

end arrangements_with_pair_together_eq_48_l2985_298505


namespace max_value_ratio_l2985_298538

theorem max_value_ratio (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) :
  (x + y + z)^2 / (x^2 + y^2 + z^2) ≤ 3 := by
  sorry

end max_value_ratio_l2985_298538


namespace initial_persons_count_l2985_298572

/-- The number of persons initially in the group -/
def n : ℕ := sorry

/-- The average weight increase when a new person replaces one person -/
def average_increase : ℚ := 5/2

/-- The weight difference between the new person and the replaced person -/
def weight_difference : ℕ := 20

/-- Theorem stating that the initial number of persons is 8 -/
theorem initial_persons_count : n = 8 := by
  sorry

end initial_persons_count_l2985_298572


namespace fencing_cost_calculation_l2985_298511

/-- Calculates the total cost of fencing a rectangular plot. -/
def total_fencing_cost (length : ℝ) (breadth : ℝ) (cost_per_meter : ℝ) : ℝ :=
  2 * (length + breadth) * cost_per_meter

/-- Proves that the total cost of fencing the given rectangular plot is 5300 currency units. -/
theorem fencing_cost_calculation :
  let length : ℝ := 55
  let breadth : ℝ := 45
  let cost_per_meter : ℝ := 26.50
  total_fencing_cost length breadth cost_per_meter = 5300 := by
  sorry

end fencing_cost_calculation_l2985_298511


namespace blackboard_numbers_l2985_298523

theorem blackboard_numbers (a b : ℕ) : 
  (¬ ∃ a b : ℕ, 13 * a + 11 * b = 86) ∧ 
  (∃ a b : ℕ, 13 * a + 11 * b = 2015) := by
  sorry

end blackboard_numbers_l2985_298523


namespace fraction_sum_integer_l2985_298596

theorem fraction_sum_integer (n : ℕ) (hn : n > 0) 
  (h_sum : ∃ (k : ℤ), (1 : ℚ) / 3 + (1 : ℚ) / 4 + (1 : ℚ) / 8 + (1 : ℚ) / n = k) : 
  n = 24 := by
sorry

end fraction_sum_integer_l2985_298596


namespace probability_in_given_scenario_l2985_298557

/-- Represents the probability of drawing a genuine product after drawing a defective one -/
def probability_genuine_after_defective (total : ℕ) (genuine : ℕ) (defective : ℕ) : ℚ :=
  if total = genuine + defective ∧ defective > 0 then
    genuine / (total - 1)
  else
    0

/-- The main theorem about the probability in the given scenario -/
theorem probability_in_given_scenario :
  probability_genuine_after_defective 7 4 3 = 2/3 := by
  sorry

end probability_in_given_scenario_l2985_298557


namespace problem_solution_l2985_298562

theorem problem_solution (x y : ℝ) : 
  ((x + 2)^3 < x^3 + 8*x^2 + 42*x + 27) ∧
  ((x^3 + 8*x^2 + 42*x + 27) < (x + 4)^3) ∧
  (y = x + 3) ∧
  ((x + 3)^3 = x^3 + 9*x^2 + 27*x + 27) ∧
  ((x + 3)^3 = x^3 + 8*x^2 + 42*x + 27) ∧
  (x^2 = 15*x) →
  x = 15 := by
sorry

end problem_solution_l2985_298562


namespace exam_score_calculation_l2985_298583

theorem exam_score_calculation (total_questions : ℕ) (answered_questions : ℕ) (correct_answers : ℕ) (raw_score : ℚ) :
  total_questions = 85 →
  answered_questions = 82 →
  correct_answers = 70 →
  raw_score = 67 →
  ∃ (points_per_correct : ℚ),
    points_per_correct * correct_answers - (answered_questions - correct_answers) * (1/4 : ℚ) = raw_score ∧
    points_per_correct = 1 :=
by sorry

end exam_score_calculation_l2985_298583


namespace margie_change_l2985_298525

/-- Calculates the change received after a purchase -/
def change_received (banana_price : ℚ) (orange_price : ℚ) (num_bananas : ℕ) (num_oranges : ℕ) (paid_amount : ℚ) : ℚ :=
  let total_cost := banana_price * num_bananas + orange_price * num_oranges
  paid_amount - total_cost

/-- Proves that Margie received $7.60 in change -/
theorem margie_change : 
  let banana_price : ℚ := 30/100
  let orange_price : ℚ := 60/100
  let num_bananas : ℕ := 4
  let num_oranges : ℕ := 2
  let paid_amount : ℚ := 10
  change_received banana_price orange_price num_bananas num_oranges paid_amount = 76/10 := by
  sorry

#eval change_received (30/100) (60/100) 4 2 10

end margie_change_l2985_298525


namespace area_of_special_points_triangle_l2985_298593

/-- A triangle with side lengths 18, 24, and 30 -/
structure RightTriangle where
  a : ℝ
  b : ℝ
  c : ℝ
  h_sides : a = 18 ∧ b = 24 ∧ c = 30

/-- The incenter of a triangle -/
def incenter (t : RightTriangle) : ℝ × ℝ := sorry

/-- The circumcenter of a triangle -/
def circumcenter (t : RightTriangle) : ℝ × ℝ := sorry

/-- The centroid of a triangle -/
def centroid (t : RightTriangle) : ℝ × ℝ := sorry

/-- The area of a triangle given its vertices -/
def triangleArea (p1 p2 p3 : ℝ × ℝ) : ℝ := sorry

/-- Theorem: The area of the triangle formed by the incenter, circumcenter, and centroid of a 18-24-30 right triangle is 6 -/
theorem area_of_special_points_triangle (t : RightTriangle) : 
  triangleArea (incenter t) (circumcenter t) (centroid t) = 6 := by sorry

end area_of_special_points_triangle_l2985_298593


namespace triangle_arithmetic_angle_sequence_side_relation_l2985_298535

open Real

/-- Given a triangle ABC with sides a, b, c and angles A, B, C (in radians),
    where A, B, C form an arithmetic sequence, 
    prove that 1/(a+b) + 1/(b+c) = 3/(a+b+c) -/
theorem triangle_arithmetic_angle_sequence_side_relation 
  (a b c A B C : ℝ) 
  (h_triangle : 0 < a ∧ 0 < b ∧ 0 < c)
  (h_angles : 0 < A ∧ 0 < B ∧ 0 < C)
  (h_sum_angles : A + B + C = π)
  (h_arithmetic_seq : ∃ d : ℝ, B = A + d ∧ C = B + d) :
  1 / (a + b) + 1 / (b + c) = 3 / (a + b + c) := by
  sorry

end triangle_arithmetic_angle_sequence_side_relation_l2985_298535


namespace m_plus_one_value_l2985_298546

theorem m_plus_one_value (m n : ℕ) 
  (h1 : m * n = 121) 
  (h2 : (m + 1) * (n + 1) = 1000) : 
  m + 1 = 879 - n := by
sorry

end m_plus_one_value_l2985_298546


namespace meeting_democrat_ratio_l2985_298521

/-- Given a meeting with participants, prove the ratio of male democrats to total male participants -/
theorem meeting_democrat_ratio 
  (total_participants : ℕ) 
  (female_democrats : ℕ) 
  (h_total : total_participants = 780)
  (h_female_dem : female_democrats = 130)
  (h_half_female : female_democrats * 2 ≤ total_participants)
  (h_third_dem : 3 * female_democrats * 2 = total_participants)
  : (total_participants - female_democrats * 2 - female_democrats) / 
    (total_participants - female_democrats * 2) = 1 / 4 := by
  sorry

end meeting_democrat_ratio_l2985_298521


namespace lcm_36_75_l2985_298587

theorem lcm_36_75 : Nat.lcm 36 75 = 900 := by
  sorry

end lcm_36_75_l2985_298587


namespace expansion_coefficient_l2985_298569

theorem expansion_coefficient (a : ℝ) (h1 : a > 0) 
  (h2 : (1 + 1) * (a + 1)^6 = 1458) : 
  (1 + 6 * 4) = 61 := by sorry

end expansion_coefficient_l2985_298569


namespace johnson_volunteers_count_l2985_298547

def total_volunteers (math_classes : ℕ) (students_per_class : ℕ) (teacher_volunteers : ℕ) (additional_needed : ℕ) : ℕ :=
  math_classes * students_per_class + teacher_volunteers + additional_needed

theorem johnson_volunteers_count :
  total_volunteers 6 5 13 7 = 50 := by
  sorry

end johnson_volunteers_count_l2985_298547


namespace number_of_possible_sums_l2985_298579

/-- The set of chips in Bag A -/
def bagA : Finset ℕ := {1, 4, 5}

/-- The set of chips in Bag B -/
def bagB : Finset ℕ := {2, 4, 6}

/-- The set of all possible sums when drawing one chip from each bag -/
def possibleSums : Finset ℕ := (bagA.product bagB).image (fun p => p.1 + p.2)

theorem number_of_possible_sums : Finset.card possibleSums = 8 := by
  sorry

end number_of_possible_sums_l2985_298579


namespace union_of_sets_l2985_298552

theorem union_of_sets : 
  let A : Set Int := {-2, 0}
  let B : Set Int := {-2, 3}
  A ∪ B = {-2, 0, 3} := by
sorry

end union_of_sets_l2985_298552


namespace bricklayer_solution_l2985_298559

/-- Represents the problem of two bricklayers building a wall -/
structure BricklayerProblem where
  -- Total number of bricks in the wall
  total_bricks : ℕ
  -- Time taken by the first bricklayer alone (in hours)
  time_first : ℕ
  -- Time taken by the second bricklayer alone (in hours)
  time_second : ℕ
  -- Reduction in combined output (in bricks per hour)
  output_reduction : ℕ
  -- Time taken when working together (in hours)
  time_together : ℕ

/-- The theorem stating the solution to the bricklayer problem -/
theorem bricklayer_solution (problem : BricklayerProblem) :
  problem.time_first = 8 →
  problem.time_second = 12 →
  problem.output_reduction = 15 →
  problem.time_together = 6 →
  problem.total_bricks = 360 := by
  sorry

#check bricklayer_solution

end bricklayer_solution_l2985_298559


namespace hyperbola_implies_constant_difference_constant_difference_not_sufficient_l2985_298540

-- Define a point in a plane
structure Point :=
  (x : ℝ) (y : ℝ)

-- Define a trajectory as a function from time to point
def Trajectory := ℝ → Point

-- Define the distance between two points
def distance (p q : Point) : ℝ := sorry

-- Define a hyperbola
def isHyperbola (t : Trajectory) : Prop := sorry

-- Define the constant difference property
def hasConstantDifference (t : Trajectory) : Prop :=
  ∃ (F₁ F₂ : Point) (k : ℝ), ∀ (time : ℝ),
    |distance (t time) F₁ - distance (t time) F₂| = k

theorem hyperbola_implies_constant_difference (t : Trajectory) :
  isHyperbola t → hasConstantDifference t := by sorry

theorem constant_difference_not_sufficient (t : Trajectory) :
  ∃ t, hasConstantDifference t ∧ ¬isHyperbola t := by sorry

end hyperbola_implies_constant_difference_constant_difference_not_sufficient_l2985_298540


namespace hyperbola_foci_distance_l2985_298588

/-- The distance between the foci of a hyperbola with equation x^2/32 - y^2/8 = 1 is 4√10 -/
theorem hyperbola_foci_distance :
  ∀ (x y : ℝ),
  x^2 / 32 - y^2 / 8 = 1 →
  ∃ (f₁ f₂ : ℝ × ℝ),
  (f₁.1 - f₂.1)^2 + (f₁.2 - f₂.2)^2 = (4 * Real.sqrt 10)^2 :=
by sorry

end hyperbola_foci_distance_l2985_298588


namespace expression_value_l2985_298503

theorem expression_value (a b : ℝ) (h : 2 * a - 3 * b = 5) :
  10 - 4 * a + 6 * b = 0 := by sorry

end expression_value_l2985_298503


namespace solution_to_system_l2985_298542

theorem solution_to_system : 
  ∀ x y : ℝ, 
  3 * x^2 - 9 * y^2 = 0 → 
  x + y = 5 → 
  ((x = (15 - 5 * Real.sqrt 3) / 2 ∧ y = (5 * Real.sqrt 3 - 5) / 2) ∨
   (x = (-15 - 5 * Real.sqrt 3) / 2 ∧ y = (5 + 5 * Real.sqrt 3) / 2)) :=
by sorry

end solution_to_system_l2985_298542


namespace pipe_A_fill_time_l2985_298541

/-- The time (in minutes) it takes for pipe B to fill the tanker alone -/
def time_B : ℝ := 40

/-- The total time (in minutes) it takes to fill the tanker when B is used for half the time
    and A and B fill it together for the other half -/
def total_time : ℝ := 29.999999999999993

/-- The time (in minutes) it takes for pipe A to fill the tanker alone -/
def time_A : ℝ := 60

/-- Proves that the time taken by pipe A to fill the tanker alone is 60 minutes -/
theorem pipe_A_fill_time :
  (time_B / 2) / time_B + (total_time / 2) * (1 / time_A + 1 / time_B) = 1 :=
sorry

end pipe_A_fill_time_l2985_298541


namespace sandwich_combinations_l2985_298576

theorem sandwich_combinations (num_meats num_cheeses : ℕ) : 
  num_meats = 12 → num_cheeses = 11 → 
  (num_meats * num_cheeses) + (num_meats * (num_cheeses.choose 2)) = 792 := by
  sorry

#check sandwich_combinations

end sandwich_combinations_l2985_298576


namespace base_r_transaction_l2985_298531

def base_r_to_decimal (digits : List Nat) (r : Nat) : Nat :=
  digits.foldl (fun acc d => acc * r + d) 0

theorem base_r_transaction (r : Nat) : r = 8 :=
  by
  have h1 : base_r_to_decimal [4, 4, 0] r + base_r_to_decimal [3, 4, 0] r = base_r_to_decimal [1, 0, 0, 0] r :=
    sorry
  sorry

end base_r_transaction_l2985_298531


namespace simple_interest_rate_percent_l2985_298553

/-- Simple interest calculation -/
theorem simple_interest_rate_percent 
  (principal : ℝ) 
  (interest : ℝ) 
  (time : ℝ) 
  (h1 : principal = 1000) 
  (h2 : interest = 400) 
  (h3 : time = 4) : 
  (interest * 100) / (principal * time) = 10 := by
sorry

end simple_interest_rate_percent_l2985_298553


namespace problem_solution_l2985_298507

theorem problem_solution (a b : ℕ) 
  (sum_eq : a + b = 31462)
  (b_div_20 : b % 20 = 0)
  (a_eq_b_div_10 : a = b / 10) : 
  b - a = 28462 := by
sorry

end problem_solution_l2985_298507


namespace first_number_10th_group_l2985_298554

/-- Sequence term definition -/
def a (n : ℕ) : ℤ := 2 * n - 3

/-- Sum of first n natural numbers -/
def sum_first_n (n : ℕ) : ℕ := n * (n + 1) / 2

/-- The first number in the kth group -/
def first_in_group (k : ℕ) : ℕ := sum_first_n (k - 1) + 1

theorem first_number_10th_group :
  a (first_in_group 10) = 89 :=
sorry

end first_number_10th_group_l2985_298554


namespace third_day_income_l2985_298519

def cab_driver_income (day1 day2 day4 day5 : ℕ) (average : ℚ) : Prop :=
  ∃ day3 : ℕ,
    (day1 + day2 + day3 + day4 + day5 : ℚ) / 5 = average ∧
    day3 = 60

theorem third_day_income :
  cab_driver_income 45 50 65 70 58 :=
sorry

end third_day_income_l2985_298519


namespace A_empty_iff_A_singleton_iff_A_singleton_elements_l2985_298584

def A (a : ℝ) : Set ℝ := {x : ℝ | a * x^2 - 3 * x + 2 = 0}

theorem A_empty_iff (a : ℝ) : A a = ∅ ↔ a > 9/8 := by sorry

theorem A_singleton_iff (a : ℝ) : (∃! x, x ∈ A a) ↔ a = 0 ∨ a = 9/8 := by sorry

theorem A_singleton_elements (a : ℝ) :
  (a = 0 → A a = {2/3}) ∧ (a = 9/8 → A a = {4/3}) := by sorry

end A_empty_iff_A_singleton_iff_A_singleton_elements_l2985_298584


namespace books_added_to_bin_l2985_298518

/-- Proves the number of books added to a bargain bin -/
theorem books_added_to_bin (initial books_sold final : ℕ) 
  (h1 : initial = 4)
  (h2 : books_sold = 3)
  (h3 : final = 11) :
  final - (initial - books_sold) = 10 := by
  sorry

end books_added_to_bin_l2985_298518


namespace expression_quadrupled_l2985_298597

variables (x y : ℝ) (h : x ≠ y)

theorem expression_quadrupled :
  (2*x)^2 * (2*y) / (2*x - 2*y) = 4 * (x^2 * y / (x - y)) :=
sorry

end expression_quadrupled_l2985_298597


namespace pizza_order_proof_l2985_298528

theorem pizza_order_proof (num_people : ℕ) (slices_per_pizza : ℕ) (slices_per_person : ℕ) : 
  num_people = 6 → slices_per_pizza = 8 → slices_per_person = 4 →
  (num_people * slices_per_person) / slices_per_pizza = 3 := by
sorry

end pizza_order_proof_l2985_298528


namespace parallel_lines_condition_l2985_298594

/-- Two lines in the form ax + by + c = 0 are parallel if and only if their slopes are equal -/
def are_parallel (a1 b1 a2 b2 : ℝ) : Prop := a1 * b2 = a2 * b1

/-- The first line equation: x + ay + 3 = 0 -/
def line1 (a : ℝ) (x y : ℝ) : Prop := x + a * y + 3 = 0

/-- The second line equation: (a-2)x + 3y + a = 0 -/
def line2 (a : ℝ) (x y : ℝ) : Prop := (a - 2) * x + 3 * y + a = 0

theorem parallel_lines_condition (a : ℝ) : 
  are_parallel 1 a (a - 2) 3 ↔ a = -1 := by sorry

end parallel_lines_condition_l2985_298594


namespace largest_integer_solution_l2985_298551

theorem largest_integer_solution (x : ℤ) : (∀ y : ℤ, -y + 3 > 1 → y ≤ x) ↔ x = 1 := by
  sorry

end largest_integer_solution_l2985_298551


namespace perpendicular_vectors_l2985_298532

def vector_a (m : ℝ) : Fin 2 → ℝ := ![m, 1]
def vector_b : Fin 2 → ℝ := ![3, 3]

def dot_product (u v : Fin 2 → ℝ) : ℝ := (u 0) * (v 0) + (u 1) * (v 1)

theorem perpendicular_vectors (m : ℝ) :
  dot_product (λ i => vector_a m i - vector_b i) vector_b = 0 → m = 5 := by
  sorry

end perpendicular_vectors_l2985_298532


namespace least_addition_for_divisibility_l2985_298582

theorem least_addition_for_divisibility :
  ∃! x : ℕ, x < 23 ∧ (1053 + x) % 23 = 0 ∧ ∀ y : ℕ, y < x → (1053 + y) % 23 ≠ 0 :=
by sorry

end least_addition_for_divisibility_l2985_298582
