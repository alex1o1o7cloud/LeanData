import Mathlib

namespace NUMINAMATH_CALUDE_water_needed_for_tanks_l2775_277511

/-- Represents a water tank with its capacity and current fill level -/
structure Tank where
  capacity : ℝ
  filled : ℝ

/-- Calculates the amount of water needed to fill a tank completely -/
def amountNeeded (tank : Tank) : ℝ :=
  tank.capacity - tank.filled

/-- The problem statement -/
theorem water_needed_for_tanks : 
  let tank1 : Tank := ⟨800, 300⟩
  let tank2 : Tank := ⟨1000, 450⟩
  let tank3 : Tank := ⟨1200, 1200 * 0.657⟩
  (amountNeeded tank1) + (amountNeeded tank2) + (amountNeeded tank3) = 1461.6 := by
  sorry


end NUMINAMATH_CALUDE_water_needed_for_tanks_l2775_277511


namespace NUMINAMATH_CALUDE_beta_interval_l2775_277557

theorem beta_interval (β : ℝ) : 
  (∃ k : ℤ, β = π/6 + 2*k*π) ∧ -2*π < β ∧ β < 2*π ↔ β = π/6 ∨ β = -11*π/6 := by
  sorry

end NUMINAMATH_CALUDE_beta_interval_l2775_277557


namespace NUMINAMATH_CALUDE_one_negative_root_condition_l2775_277533

/-- A polynomial of the form x^4 + 3px^3 + 6x^2 + 3px + 1 -/
def polynomial (p : ℝ) (x : ℝ) : ℝ := x^4 + 3*p*x^3 + 6*x^2 + 3*p*x + 1

/-- The condition that the polynomial has exactly one negative real root -/
def has_one_negative_root (p : ℝ) : Prop :=
  ∃! x : ℝ, x < 0 ∧ polynomial p x = 0

/-- Theorem stating the condition on p for the polynomial to have exactly one negative real root -/
theorem one_negative_root_condition (p : ℝ) :
  has_one_negative_root p ↔ p ≥ 4/3 := by sorry

end NUMINAMATH_CALUDE_one_negative_root_condition_l2775_277533


namespace NUMINAMATH_CALUDE_ab_range_l2775_277596

theorem ab_range (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : 4 * a * b = a + b) : a * b ≥ 1/4 := by
  sorry

end NUMINAMATH_CALUDE_ab_range_l2775_277596


namespace NUMINAMATH_CALUDE_min_value_theorem_l2775_277560

theorem min_value_theorem (a b : ℝ) (h1 : b > 0) (h2 : a + b = 2) :
  (1 / (2 * |a|)) + (|a| / b) ≥ 3/4 ∧ 
  ∃ (a₀ b₀ : ℝ), b₀ > 0 ∧ a₀ + b₀ = 2 ∧ (1 / (2 * |a₀|)) + (|a₀| / b₀) = 3/4 :=
sorry

end NUMINAMATH_CALUDE_min_value_theorem_l2775_277560


namespace NUMINAMATH_CALUDE_polynomial_factor_implies_d_value_l2775_277591

theorem polynomial_factor_implies_d_value (c d : ℤ) : 
  (∃ k : ℤ, (X^3 - 2*X^2 - X + 2) * (c*X + k) = c*X^4 + d*X^3 - 2*X^2 + 2) → 
  d = -1 :=
by sorry

end NUMINAMATH_CALUDE_polynomial_factor_implies_d_value_l2775_277591


namespace NUMINAMATH_CALUDE_pure_imaginary_modulus_l2775_277550

theorem pure_imaginary_modulus (a : ℝ) : 
  let z : ℂ := (a + Complex.I) / (1 - 2 * Complex.I)
  (z.re = 0 ∧ z.im ≠ 0) → Complex.abs (a + 2 * Complex.I) = 2 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_pure_imaginary_modulus_l2775_277550


namespace NUMINAMATH_CALUDE_sum_of_binary_digits_300_l2775_277510

def binary_representation (n : ℕ) : List ℕ :=
  sorry

def sum_of_digits (l : List ℕ) : ℕ :=
  sorry

theorem sum_of_binary_digits_300 :
  sum_of_digits (binary_representation 300) = 4 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_binary_digits_300_l2775_277510


namespace NUMINAMATH_CALUDE_assignment_calculations_l2775_277528

/-- Represents the number of volunteers -/
def num_volunteers : ℕ := 5

/-- Represents the number of communities -/
def num_communities : ℕ := 4

/-- Total number of assignment schemes -/
def total_assignments : ℕ := num_communities ^ num_volunteers

/-- Number of assignments with restrictions on community A and minimum volunteers -/
def restricted_assignments : ℕ := 150

/-- Number of assignments with each community having at least one volunteer and two specific volunteers not in the same community -/
def specific_restricted_assignments : ℕ := 216

/-- Theorem stating the correctness of the assignment calculations -/
theorem assignment_calculations :
  (total_assignments = 1024) ∧
  (restricted_assignments = 150) ∧
  (specific_restricted_assignments = 216) := by sorry

end NUMINAMATH_CALUDE_assignment_calculations_l2775_277528


namespace NUMINAMATH_CALUDE_positive_expressions_l2775_277525

theorem positive_expressions (x y z : ℝ) 
  (hx : -1 < x ∧ x < 0) 
  (hy : 0 < y ∧ y < 1) 
  (hz : 2 < z ∧ z < 3) : 
  0 < y + x^2 * z ∧ 
  0 < y + x^2 ∧ 
  0 < y + y^2 ∧ 
  0 < y + 2 * z := by
  sorry

end NUMINAMATH_CALUDE_positive_expressions_l2775_277525


namespace NUMINAMATH_CALUDE_exists_fourth_power_product_l2775_277562

def is_not_divisible_by_primes_greater_than_28 (n : ℕ) : Prop :=
  ∀ p : ℕ, Prime p → p > 28 → ¬(p ∣ n)

theorem exists_fourth_power_product 
  (M : Finset ℕ) 
  (h_card : M.card = 2008) 
  (h_distinct : M.card = Finset.card (M.image id))
  (h_positive : ∀ n ∈ M, n > 0)
  (h_not_div : ∀ n ∈ M, is_not_divisible_by_primes_greater_than_28 n) :
  ∃ a b c d : ℕ, a ∈ M ∧ b ∈ M ∧ c ∈ M ∧ d ∈ M ∧ 
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d ∧
  ∃ k : ℕ, a * b * c * d = k^4 :=
sorry

end NUMINAMATH_CALUDE_exists_fourth_power_product_l2775_277562


namespace NUMINAMATH_CALUDE_qinJiushao_v3_value_l2775_277578

-- Define the polynomial f(x)
def f (x : ℝ) : ℝ := 2*x^5 + 5*x^4 + 8*x^3 + 7*x^2 - 6*x + 11

-- Define Qin Jiushao's algorithm for this specific polynomial
def qinJiushao (x : ℝ) : ℝ × ℝ × ℝ × ℝ := 
  let v1 := 2*x + 5
  let v2 := v1*x + 8
  let v3 := v2*x + 7
  let v4 := v3*x - 6
  (v1, v2, v3, v4)

-- Theorem statement
theorem qinJiushao_v3_value : 
  (qinJiushao 3).2.2 = 130 :=
by sorry

end NUMINAMATH_CALUDE_qinJiushao_v3_value_l2775_277578


namespace NUMINAMATH_CALUDE_matrix_determinant_zero_l2775_277519

theorem matrix_determinant_zero (a b c : ℝ) : 
  Matrix.det !![1, a+b, b+c; 1, a+2*b, b+2*c; 1, a+3*b, b+3*c] = 0 := by
  sorry

end NUMINAMATH_CALUDE_matrix_determinant_zero_l2775_277519


namespace NUMINAMATH_CALUDE_new_ratio_first_term_l2775_277567

/-- Given an original ratio of 7:11, when 5 is added to both terms, 
    the first term of the new ratio is 12. -/
theorem new_ratio_first_term : 
  let original_first : ℕ := 7
  let original_second : ℕ := 11
  let added_number : ℕ := 5
  let new_first : ℕ := original_first + added_number
  new_first = 12 := by sorry

end NUMINAMATH_CALUDE_new_ratio_first_term_l2775_277567


namespace NUMINAMATH_CALUDE_not_well_placed_2011_l2775_277500

/-- Represents the first number in a row of the triangular table -/
def first_in_row (k : ℕ) : ℕ := k^2 - k + 1

/-- Represents the first number in a column of the triangular table -/
def first_in_column (k : ℕ) : ℕ := k^2

/-- A number is well-placed if it equals the sum of the first number in its row and the first number in its column -/
def is_well_placed (N : ℕ) : Prop :=
  ∃ (n m : ℕ), N = first_in_row n + first_in_column m

theorem not_well_placed_2011 : ¬ is_well_placed (2^2011) := by sorry

end NUMINAMATH_CALUDE_not_well_placed_2011_l2775_277500


namespace NUMINAMATH_CALUDE_sum_110_terms_l2775_277549

-- Define an arithmetic sequence type
def ArithmeticSequence := ℕ → ℤ

-- Define the sum of the first n terms of an arithmetic sequence
def sum_n_terms (seq : ArithmeticSequence) (n : ℕ) : ℤ :=
  (List.range n).map seq |>.sum

-- Define the properties of our specific arithmetic sequence
def special_arithmetic_sequence (seq : ArithmeticSequence) : Prop :=
  sum_n_terms seq 10 = 100 ∧ sum_n_terms seq 100 = 10

-- State the theorem
theorem sum_110_terms (seq : ArithmeticSequence) 
  (h : special_arithmetic_sequence seq) : 
  sum_n_terms seq 110 = -110 := by
  sorry

end NUMINAMATH_CALUDE_sum_110_terms_l2775_277549


namespace NUMINAMATH_CALUDE_blue_balls_removed_l2775_277590

theorem blue_balls_removed (initial_total : ℕ) (initial_blue : ℕ) (final_probability : ℚ) : ℕ :=
  let removed : ℕ := 3
  have h1 : initial_total = 18 := by sorry
  have h2 : initial_blue = 6 := by sorry
  have h3 : final_probability = 1 / 5 := by sorry
  have h4 : (initial_blue - removed : ℚ) / (initial_total - removed) = final_probability := by sorry
  removed

#check blue_balls_removed

end NUMINAMATH_CALUDE_blue_balls_removed_l2775_277590


namespace NUMINAMATH_CALUDE_parabola_point_distance_l2775_277523

/-- Given a point P(a,0) and a parabola y^2 = 4x, if for every point Q on the parabola |PQ| ≥ |a|, then a ≤ 2 -/
theorem parabola_point_distance (a : ℝ) : 
  (∀ x y : ℝ, y^2 = 4*x → ((x - a)^2 + y^2 ≥ a^2)) → 
  a ≤ 2 := by
sorry

end NUMINAMATH_CALUDE_parabola_point_distance_l2775_277523


namespace NUMINAMATH_CALUDE_center_of_specific_circle_l2775_277522

/-- The center coordinates of a circle given its equation -/
def circle_center (a b r : ℝ) : ℝ × ℝ := (a, -b)

/-- Theorem: The center coordinates of the circle (x-2)^2 + (y+1)^2 = 4 are (2, -1) -/
theorem center_of_specific_circle :
  circle_center 2 (-1) 2 = (2, -1) := by sorry

end NUMINAMATH_CALUDE_center_of_specific_circle_l2775_277522


namespace NUMINAMATH_CALUDE_compute_fraction_power_l2775_277521

theorem compute_fraction_power : 8 * (2 / 7)^4 = 128 / 2401 := by
  sorry

end NUMINAMATH_CALUDE_compute_fraction_power_l2775_277521


namespace NUMINAMATH_CALUDE_surface_area_comparison_l2775_277540

/-- Given a cube, equilateral cylinder, and sphere with equal volumes, prove their surface areas satisfy S₁ > S₂ > S₃ --/
theorem surface_area_comparison 
  (a : ℝ) (r : ℝ) (R : ℝ) 
  (h_pos : a > 0 ∧ r > 0 ∧ R > 0)
  (h_vol_eq : a^3 = 2 * π * r^3 ∧ a^3 = (4/3) * π * R^3)
  (h_cylinder_eq : 2 * r = 2 * r) -- Height equals diameter of base
  (S₁ : ℝ) (S₂ : ℝ) (S₃ : ℝ)
  (h_S₁ : S₁ = 6 * a^2)
  (h_S₂ : S₂ = 2 * π * r^2 + 2 * π * r * (2 * r))
  (h_S₃ : S₃ = 4 * π * R^2) :
  S₁ > S₂ ∧ S₂ > S₃ := by
  sorry

end NUMINAMATH_CALUDE_surface_area_comparison_l2775_277540


namespace NUMINAMATH_CALUDE_turkey_cost_per_kg_turkey_cost_is_two_l2775_277501

/-- Given Dabbie's turkey purchase scenario, prove the cost per kilogram of turkey. -/
theorem turkey_cost_per_kg : ℝ → Prop :=
  fun cost_per_kg =>
    let first_turkey_weight := 6
    let second_turkey_weight := 9
    let third_turkey_weight := 2 * second_turkey_weight
    let total_weight := first_turkey_weight + second_turkey_weight + third_turkey_weight
    let total_cost := 66
    cost_per_kg = total_cost / total_weight

/-- The cost per kilogram of turkey is $2. -/
theorem turkey_cost_is_two : turkey_cost_per_kg 2 := by
  sorry

end NUMINAMATH_CALUDE_turkey_cost_per_kg_turkey_cost_is_two_l2775_277501


namespace NUMINAMATH_CALUDE_line_points_k_value_l2775_277539

/-- A line contains the points (0, 10), (5, k), and (25, 0). The value of k is 8. -/
theorem line_points_k_value :
  ∀ (k : ℝ),
  (∃ (m b : ℝ), 
    (0 : ℝ) * m + b = 10 ∧
    5 * m + b = k ∧
    25 * m + b = 0) →
  k = 8 :=
by
  sorry

end NUMINAMATH_CALUDE_line_points_k_value_l2775_277539


namespace NUMINAMATH_CALUDE_equal_cost_sharing_l2775_277565

theorem equal_cost_sharing (A B C : ℝ) : 
  let total_cost := A + B + C
  let equal_share := total_cost / 3
  let leroy_additional_payment := equal_share - A
  leroy_additional_payment = (B + C - 2 * A) / 3 := by
sorry

end NUMINAMATH_CALUDE_equal_cost_sharing_l2775_277565


namespace NUMINAMATH_CALUDE_whitney_total_spent_l2775_277535

def whale_books : ℕ := 9
def fish_books : ℕ := 7
def magazines : ℕ := 3
def book_cost : ℕ := 11
def magazine_cost : ℕ := 1

theorem whitney_total_spent : 
  (whale_books + fish_books) * book_cost + magazines * magazine_cost = 179 := by
  sorry

end NUMINAMATH_CALUDE_whitney_total_spent_l2775_277535


namespace NUMINAMATH_CALUDE_dog_kennel_problem_l2775_277520

theorem dog_kennel_problem (total long_fur brown neither : ℕ) 
  (h_total : total = 45)
  (h_long_fur : long_fur = 26)
  (h_brown : brown = 30)
  (h_neither : neither = 8)
  : long_fur + brown - (total - neither) = 19 := by
  sorry

end NUMINAMATH_CALUDE_dog_kennel_problem_l2775_277520


namespace NUMINAMATH_CALUDE_weight_of_b_l2775_277552

theorem weight_of_b (a b c : ℝ) : 
  (a + b + c) / 3 = 45 →
  (a + b) / 2 = 40 →
  (b + c) / 2 = 43 →
  b = 31 := by
sorry

end NUMINAMATH_CALUDE_weight_of_b_l2775_277552


namespace NUMINAMATH_CALUDE_computers_needed_for_expanded_class_l2775_277543

/-- Given an initial number of students, a student-to-computer ratio, and additional students,
    calculate the total number of computers needed to maintain the ratio. -/
def total_computers_needed (initial_students : ℕ) (ratio : ℕ) (additional_students : ℕ) : ℕ :=
  (initial_students / ratio) + (additional_students / ratio)

/-- Theorem: Given 82 initial students, a ratio of 2 students per computer, and 16 additional students,
    the total number of computers needed to maintain the same ratio is 49. -/
theorem computers_needed_for_expanded_class : total_computers_needed 82 2 16 = 49 := by
  sorry

end NUMINAMATH_CALUDE_computers_needed_for_expanded_class_l2775_277543


namespace NUMINAMATH_CALUDE_simplify_expression_l2775_277568

theorem simplify_expression (m : ℝ) (h1 : m ≠ -1) (h2 : m ≠ -2) :
  ((4 * m + 5) / (m + 1) + m - 1) / ((m + 2) / (m + 1)) = m + 2 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l2775_277568


namespace NUMINAMATH_CALUDE_tan_theta_value_l2775_277595

theorem tan_theta_value (θ : ℝ) (z₁ z₂ : ℂ) 
  (h1 : z₁ = Complex.mk (Real.sin θ) (-4/5))
  (h2 : z₂ = Complex.mk (3/5) (-Real.cos θ))
  (h3 : (z₁ - z₂).re = 0) : 
  Real.tan θ = -3/4 := by
  sorry

end NUMINAMATH_CALUDE_tan_theta_value_l2775_277595


namespace NUMINAMATH_CALUDE_repeated_root_condition_l2775_277536

theorem repeated_root_condition (a : ℝ) : 
  (∃ x : ℝ, (3 / (x - 3) + a * x / (x^2 - 9) = 4 / (x + 3)) ∧ 
   (∀ ε > 0, ∃ y ≠ x, |y - x| < ε ∧ (3 / (y - 3) + a * y / (y^2 - 9) = 4 / (y + 3)))) ↔ 
  (a = -6 ∨ a = 8) := by
sorry

end NUMINAMATH_CALUDE_repeated_root_condition_l2775_277536


namespace NUMINAMATH_CALUDE_angle_B_measure_l2775_277572

/-- Given an acute triangle ABC with sides a, b, c opposite to angles A, B, C respectively,
    prove that if a = 1, b = 2cos(C), and sin(C)cos(A) - sin(π/4 - B)sin(π/4 + B) = 0,
    then B = π/6. -/
theorem angle_B_measure (A B C : Real) (a b c : Real) : 
  0 < A ∧ A < π/2 →
  0 < B ∧ B < π/2 →
  0 < C ∧ C < π/2 →
  a = 1 →
  b = 2 * Real.cos C →
  Real.sin C * Real.cos A - Real.sin (π/4 - B) * Real.sin (π/4 + B) = 0 →
  B = π/6 := by
  sorry

end NUMINAMATH_CALUDE_angle_B_measure_l2775_277572


namespace NUMINAMATH_CALUDE_symmetric_point_l2775_277563

/-- Given a point (a, b) and a line x + y + 1 = 0, the point symmetric to (a, b) with respect to the line is (-b-1, -a-1) -/
theorem symmetric_point (a b : ℝ) : 
  let original_point := (a, b)
  let line_equation (x y : ℝ) := x + y + 1 = 0
  let symmetric_point := (-b - 1, -a - 1)
  ∀ x y, line_equation x y → 
    (x - a) ^ 2 + (y - b) ^ 2 = (x - (-b - 1)) ^ 2 + (y - (-a - 1)) ^ 2 ∧
    line_equation ((a + (-b - 1)) / 2) ((b + (-a - 1)) / 2) := by
  sorry

end NUMINAMATH_CALUDE_symmetric_point_l2775_277563


namespace NUMINAMATH_CALUDE_conner_average_speed_l2775_277559

/-- The average speed of Conner's dune buggy given different terrain conditions -/
theorem conner_average_speed 
  (flat_speed : ℝ) 
  (downhill_speed_increase : ℝ) 
  (uphill_speed_decrease : ℝ) 
  (h1 : flat_speed = 60) 
  (h2 : downhill_speed_increase = 12) 
  (h3 : uphill_speed_decrease = 18) :
  (flat_speed + (flat_speed + downhill_speed_increase) + (flat_speed - uphill_speed_decrease)) / 3 = 58 := by
  sorry

end NUMINAMATH_CALUDE_conner_average_speed_l2775_277559


namespace NUMINAMATH_CALUDE_yadav_clothes_transport_expenditure_l2775_277514

/-- Represents Mr. Yadav's financial situation --/
structure YadavFinances where
  monthlySalary : ℝ
  consumablePercentage : ℝ
  rentPercentage : ℝ
  utilitiesPercentage : ℝ
  entertainmentPercentage : ℝ
  clothesTransportPercentage : ℝ
  annualSavings : ℝ

/-- Calculates Mr. Yadav's monthly expenditure on clothes and transport --/
def monthlyClothesTransportExpenditure (y : YadavFinances) : ℝ :=
  let totalSpentPercentage := y.consumablePercentage + y.rentPercentage + y.utilitiesPercentage + y.entertainmentPercentage
  let remainingPercentage := 1 - totalSpentPercentage
  let monthlyRemainder := y.monthlySalary * remainingPercentage
  monthlyRemainder * y.clothesTransportPercentage

/-- Theorem stating that Mr. Yadav's monthly expenditure on clothes and transport is 2052 --/
theorem yadav_clothes_transport_expenditure (y : YadavFinances) 
  (h1 : y.consumablePercentage = 0.6)
  (h2 : y.rentPercentage = 0.2)
  (h3 : y.utilitiesPercentage = 0.1)
  (h4 : y.entertainmentPercentage = 0.05)
  (h5 : y.clothesTransportPercentage = 0.5)
  (h6 : y.annualSavings = 24624) :
  monthlyClothesTransportExpenditure y = 2052 := by
  sorry

#check yadav_clothes_transport_expenditure

end NUMINAMATH_CALUDE_yadav_clothes_transport_expenditure_l2775_277514


namespace NUMINAMATH_CALUDE_school_trip_theorem_l2775_277571

/-- The number of school buses -/
def num_buses : ℕ := 95

/-- The number of seats in each school bus -/
def seats_per_bus : ℕ := 118

/-- The number of students in the school -/
def num_students : ℕ := num_buses * seats_per_bus

theorem school_trip_theorem : num_students = 11210 := by
  sorry

end NUMINAMATH_CALUDE_school_trip_theorem_l2775_277571


namespace NUMINAMATH_CALUDE_cubic_equation_real_root_l2775_277558

/-- The cubic equation 5z^3 - 4iz^2 + z - k = 0 has at least one real root for all positive real k -/
theorem cubic_equation_real_root (k : ℝ) (hk : k > 0) : 
  ∃ (z : ℂ), z.im = 0 ∧ 5 * z^3 - 4 * Complex.I * z^2 + z - k = 0 := by sorry

end NUMINAMATH_CALUDE_cubic_equation_real_root_l2775_277558


namespace NUMINAMATH_CALUDE_problem_solution_l2775_277592

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (2^x + 1) / (2^(x+1) - a)

theorem problem_solution :
  ∃ (a : ℝ),
    (∀ (x : ℝ), f a x = (2^x + 1) / (2^(x+1) - a)) ∧
    (a = 2) ∧
    (∀ (x y : ℝ), 0 < x → 0 < y → x < y → f a x > f a y) ∧
    (∀ (k : ℝ), (∃ (x : ℝ), 0 < x ∧ x ≤ 1 ∧ k * f a x = 2) → 0 < k ∧ k ≤ 4/3) :=
by sorry

end NUMINAMATH_CALUDE_problem_solution_l2775_277592


namespace NUMINAMATH_CALUDE_sum_of_decimals_l2775_277599

theorem sum_of_decimals : (7.46 : ℝ) + 4.29 = 11.75 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_decimals_l2775_277599


namespace NUMINAMATH_CALUDE_union_A_B_l2775_277505

def A : Set ℕ := {1, 2}

def B : Set ℕ := {x | ∃ a b, a ∈ A ∧ b ∈ A ∧ x = a + b}

theorem union_A_B : A ∪ B = {1, 2, 3, 4} := by sorry

end NUMINAMATH_CALUDE_union_A_B_l2775_277505


namespace NUMINAMATH_CALUDE_paper_cutting_theorem_l2775_277509

/-- Represents the number of pieces after a series of cutting operations -/
def num_pieces (n : ℕ) : ℕ := 4 * n + 1

/-- The result we want to prove is valid -/
def target_result : ℕ := 1993

theorem paper_cutting_theorem :
  ∃ (n : ℕ), num_pieces n = target_result ∧
  ∀ (m : ℕ), ∃ (k : ℕ), num_pieces k = m → m = target_result ∨ m ≠ target_result :=
by sorry

end NUMINAMATH_CALUDE_paper_cutting_theorem_l2775_277509


namespace NUMINAMATH_CALUDE_cement_mixture_weight_l2775_277546

theorem cement_mixture_weight :
  ∀ (total_weight : ℝ),
  (1/5 : ℝ) * total_weight +     -- Weight of sand
  (3/4 : ℝ) * total_weight +     -- Weight of water
  6 = total_weight →             -- Weight of gravel
  total_weight = 120 := by
sorry

end NUMINAMATH_CALUDE_cement_mixture_weight_l2775_277546


namespace NUMINAMATH_CALUDE_three_points_count_l2775_277508

/-- A configuration of points and lines -/
structure Configuration where
  points : Finset Nat
  lines : Finset (Finset Nat)
  point_count : points.card = 6
  line_count : lines.card = 4
  points_per_line : ∀ l ∈ lines, l.card = 3
  lines_contain_points : ∀ l ∈ lines, l ⊆ points

/-- The number of ways to choose three points on a line in the configuration -/
def three_points_on_line (config : Configuration) : Nat :=
  config.lines.sum (fun l => (l.card.choose 3))

theorem three_points_count (config : Configuration) :
  three_points_on_line config = 24 := by
  sorry

#check three_points_count

end NUMINAMATH_CALUDE_three_points_count_l2775_277508


namespace NUMINAMATH_CALUDE_age_difference_l2775_277506

theorem age_difference (A B : ℕ) : B = 39 → A + 10 = 2 * (B - 10) → A - B = 9 := by
  sorry

end NUMINAMATH_CALUDE_age_difference_l2775_277506


namespace NUMINAMATH_CALUDE_uncool_family_members_l2775_277517

theorem uncool_family_members (total : ℕ) (cool_dad : ℕ) (cool_mom : ℕ) (cool_sibling : ℕ)
  (cool_dad_and_mom : ℕ) (cool_mom_and_sibling : ℕ) (cool_dad_and_sibling : ℕ)
  (cool_all : ℕ) (h1 : total = 40) (h2 : cool_dad = 18) (h3 : cool_mom = 20)
  (h4 : cool_sibling = 10) (h5 : cool_dad_and_mom = 8) (h6 : cool_mom_and_sibling = 4)
  (h7 : cool_dad_and_sibling = 3) (h8 : cool_all = 2) :
  total - (cool_dad + cool_mom + cool_sibling - cool_dad_and_mom - cool_mom_and_sibling - cool_dad_and_sibling + cool_all) = 5 := by
sorry

end NUMINAMATH_CALUDE_uncool_family_members_l2775_277517


namespace NUMINAMATH_CALUDE_player_A_wins_l2775_277551

/-- Represents a player in the game -/
inductive Player : Type
| A : Player
| B : Player

/-- Represents a row of squares on the game board -/
structure Row :=
  (length : ℕ)

/-- Represents the state of the game -/
structure GameState :=
  (tokens : ℕ)
  (row_R : Row)
  (row_S : Row)

/-- Determines if a player has a winning strategy -/
def has_winning_strategy (player : Player) (state : GameState) : Prop :=
  match player with
  | Player.A => state.tokens > 10
  | Player.B => state.tokens ≤ 10

/-- The main theorem stating that Player A has a winning strategy when tokens > 10 -/
theorem player_A_wins (state : GameState) (h1 : state.row_R.length = 1492) (h2 : state.row_S.length = 1989) :
  has_winning_strategy Player.A state ↔ state.tokens > 10 :=
sorry

end NUMINAMATH_CALUDE_player_A_wins_l2775_277551


namespace NUMINAMATH_CALUDE_ellipse_properties_l2775_277545

/-- Given an ellipse with semi-major axis a, semi-minor axis b, and focal distance 2√3 -/
def Ellipse (a b : ℝ) : Prop :=
  a > b ∧ b > 0 ∧ a^2 - b^2 = 3

/-- The equation of the ellipse -/
def EllipseEquation (a b : ℝ) : ℝ × ℝ → Prop :=
  fun (x, y) ↦ x^2 / a^2 + y^2 / b^2 = 1

/-- Line l₁ with slope k intersecting the ellipse at two points -/
def Line1 (k : ℝ) : ℝ × ℝ → Prop :=
  fun (x, y) ↦ y = k * x ∧ k ≠ 0

/-- Line l₂ with slope k/4 passing through a point on the ellipse -/
def Line2 (k : ℝ) (x₀ y₀ : ℝ) : ℝ × ℝ → Prop :=
  fun (x, y) ↦ y - y₀ = (k/4) * (x - x₀)

theorem ellipse_properties (a b : ℝ) (h : Ellipse a b) :
  ∃ (k x₀ y₀ x₁ y₁ : ℝ),
    EllipseEquation a b (x₀, y₀) ∧
    EllipseEquation a b (x₁, y₁) ∧
    Line1 k (x₀, y₀) ∧
    Line2 k x₀ y₀ (x₁, y₁) ∧
    (y₁ - y₀) * (x₁ - x₀) = -1/k ∧
    (∀ (x y : ℝ), EllipseEquation a b (x, y) ↔ x^2/4 + y^2 = 1) ∧
    (∃ (M N : ℝ),
      Line2 k x₀ y₀ (M, 0) ∧
      Line2 k x₀ y₀ (0, N) ∧
      ∀ (M' N' : ℝ),
        Line2 k x₀ y₀ (M', 0) ∧
        Line2 k x₀ y₀ (0, N') →
        abs (M * N) / 2 ≤ 9/8) :=
by sorry

end NUMINAMATH_CALUDE_ellipse_properties_l2775_277545


namespace NUMINAMATH_CALUDE_planes_perpendicular_to_line_are_parallel_lines_perpendicular_to_plane_are_parallel_l2775_277527

-- Define the basic types
variable (Point Line Plane : Type)

-- Define the relationships
variable (perpendicular_line_plane : Line → Plane → Prop)
variable (perpendicular_plane_line : Plane → Line → Prop)
variable (parallel_planes : Plane → Plane → Prop)
variable (parallel_lines : Line → Line → Prop)

-- Theorem 1: Two planes perpendicular to the same line are parallel
theorem planes_perpendicular_to_line_are_parallel 
  (l : Line) (p1 p2 : Plane) 
  (h1 : perpendicular_plane_line p1 l) 
  (h2 : perpendicular_plane_line p2 l) : 
  parallel_planes p1 p2 :=
sorry

-- Theorem 2: Two lines perpendicular to the same plane are parallel
theorem lines_perpendicular_to_plane_are_parallel 
  (p : Plane) (l1 l2 : Line) 
  (h1 : perpendicular_line_plane l1 p) 
  (h2 : perpendicular_line_plane l2 p) : 
  parallel_lines l1 l2 :=
sorry

end NUMINAMATH_CALUDE_planes_perpendicular_to_line_are_parallel_lines_perpendicular_to_plane_are_parallel_l2775_277527


namespace NUMINAMATH_CALUDE_rectangular_plot_length_difference_l2775_277588

theorem rectangular_plot_length_difference (b x : ℝ) : 
  b + x = 64 →                         -- length is 64 meters
  26.5 * (2 * (b + x) + 2 * b) = 5300 →  -- cost of fencing
  x = 28 := by sorry

end NUMINAMATH_CALUDE_rectangular_plot_length_difference_l2775_277588


namespace NUMINAMATH_CALUDE_tan_alpha_on_unit_circle_l2775_277537

theorem tan_alpha_on_unit_circle (α : ℝ) : 
  ((-4/5 : ℝ)^2 + (3/5 : ℝ)^2 = 1) →  -- Point lies on the unit circle
  (∃ (t : ℝ), t > 0 ∧ t * Real.cos α = -4/5 ∧ t * Real.sin α = 3/5) →  -- Point is terminal point of angle α
  Real.tan α = -3/4 := by
sorry

end NUMINAMATH_CALUDE_tan_alpha_on_unit_circle_l2775_277537


namespace NUMINAMATH_CALUDE_triangle_properties_l2775_277585

-- Define the lines of triangle ABC
def line_AB (x y : ℝ) : Prop := 3 * x + 4 * y + 12 = 0
def line_BC (x y : ℝ) : Prop := 4 * x - 3 * y + 16 = 0
def line_CA (x y : ℝ) : Prop := 2 * x + y - 2 = 0

-- Define point B as the intersection of AB and BC
def point_B : ℝ × ℝ := (-4, 0)

-- Define the equation of the altitude from A to BC
def altitude_A_to_BC (x y : ℝ) : Prop := x - 2 * y + 4 = 0

theorem triangle_properties :
  (∀ x y : ℝ, line_AB x y ∧ line_BC x y → (x, y) = point_B) ∧
  (∀ x y : ℝ, altitude_A_to_BC x y ↔ 
    (∃ t : ℝ, x = t * (point_B.1 - (2 / 5)) ∧ 
              y = t * (point_B.2 + (1 / 5)) ∧
              2 * x + y - 2 = 0)) :=
sorry

end NUMINAMATH_CALUDE_triangle_properties_l2775_277585


namespace NUMINAMATH_CALUDE_repeating_decimal_difference_l2775_277504

theorem repeating_decimal_difference : 
  (8 : ℚ) / 11 - 72 / 100 = 2 / 275 := by sorry

end NUMINAMATH_CALUDE_repeating_decimal_difference_l2775_277504


namespace NUMINAMATH_CALUDE_college_selection_ways_l2775_277524

theorem college_selection_ways (n : ℕ) (k : ℕ) (m : ℕ) :
  n = 6 → k = 3 → m = 2 →
  (m * (n - m).choose (k - 1)) + ((n - m).choose k) = 16 := by sorry

end NUMINAMATH_CALUDE_college_selection_ways_l2775_277524


namespace NUMINAMATH_CALUDE_radish_patch_size_proof_l2775_277570

/-- The size of a pea patch in square feet -/
def pea_patch_size : ℝ := 30

/-- The size of a radish patch in square feet -/
def radish_patch_size : ℝ := 15

theorem radish_patch_size_proof :
  (pea_patch_size = 2 * radish_patch_size) ∧
  (pea_patch_size / 6 = 5) →
  radish_patch_size = 15 := by
  sorry

end NUMINAMATH_CALUDE_radish_patch_size_proof_l2775_277570


namespace NUMINAMATH_CALUDE_distance_borya_vasya_l2775_277515

/-- Represents the positions of houses along a road -/
structure HousePositions where
  andrey : ℝ
  borya : ℝ
  vasya : ℝ
  gena : ℝ

/-- The race setup along the road -/
def RaceSetup (h : HousePositions) : Prop :=
  h.gena - h.andrey = 2450 ∧
  h.vasya - h.andrey = h.gena - h.borya ∧
  (h.borya + h.gena) / 2 - (h.andrey + h.vasya) / 2 = 1000

theorem distance_borya_vasya (h : HousePositions) (race : RaceSetup h) :
  h.vasya - h.borya = 450 := by
  sorry

end NUMINAMATH_CALUDE_distance_borya_vasya_l2775_277515


namespace NUMINAMATH_CALUDE_nested_expression_value_l2775_277569

theorem nested_expression_value : (3 * (3 * (3 * (3 * (3 * (3 + 2) + 2) + 2) + 2) + 2) + 2) = 1457 := by
  sorry

end NUMINAMATH_CALUDE_nested_expression_value_l2775_277569


namespace NUMINAMATH_CALUDE_cross_section_distance_l2775_277554

/-- Represents a right hexagonal pyramid -/
structure RightHexagonalPyramid where
  /-- Height of the pyramid -/
  height : ℝ
  /-- Side length of the base hexagon -/
  base_side : ℝ

/-- Represents a cross section of the pyramid -/
structure CrossSection where
  /-- Distance from the apex of the pyramid -/
  distance : ℝ
  /-- Area of the cross section -/
  area : ℝ

/-- 
Theorem: In a right hexagonal pyramid, if two cross sections parallel to the base 
have areas of 300√3 sq ft and 675√3 sq ft, and these planes are 12 feet apart, 
then the distance from the apex to the larger cross section is 36 feet.
-/
theorem cross_section_distance 
  (pyramid : RightHexagonalPyramid) 
  (cs1 cs2 : CrossSection) 
  (h_area1 : cs1.area = 300 * Real.sqrt 3)
  (h_area2 : cs2.area = 675 * Real.sqrt 3)
  (h_distance : cs2.distance - cs1.distance = 12)
  (h_order : cs1.distance < cs2.distance) :
  cs2.distance = 36 := by
  sorry

end NUMINAMATH_CALUDE_cross_section_distance_l2775_277554


namespace NUMINAMATH_CALUDE_triangle_properties_l2775_277516

-- Define the triangle
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

-- Define the theorem
theorem triangle_properties (t : Triangle) 
  (h1 : t.a < t.b ∧ t.b < t.c)
  (h2 : t.a / Real.sin t.A = 2 * t.b / Real.sqrt 3) :
  t.B = π / 3 ∧ 
  (t.a = 2 → t.c = 3 → t.b = Real.sqrt 7 ∧ 
    (1 / 2 : ℝ) * t.a * t.c * Real.sin t.B = (3 * Real.sqrt 3) / 2) :=
by sorry

end NUMINAMATH_CALUDE_triangle_properties_l2775_277516


namespace NUMINAMATH_CALUDE_min_value_of_f_l2775_277513

/-- The quadratic function f(x) = (x-1)^2 - 3 -/
def f (x : ℝ) : ℝ := (x - 1)^2 - 3

/-- The minimum value of f(x) is -3 -/
theorem min_value_of_f :
  ∃ (m : ℝ), m = -3 ∧ ∀ (x : ℝ), f x ≥ m :=
sorry

end NUMINAMATH_CALUDE_min_value_of_f_l2775_277513


namespace NUMINAMATH_CALUDE_quadratic_inequality_properties_l2775_277579

theorem quadratic_inequality_properties (a b c : ℝ) :
  (∀ x : ℝ, a * x^2 + b * x + c ≥ 0 ↔ x ≤ 3 ∨ x ≥ 4) →
  a > 0 ∧ a + b + c > 0 := by
sorry

end NUMINAMATH_CALUDE_quadratic_inequality_properties_l2775_277579


namespace NUMINAMATH_CALUDE_buddy_system_l2775_277575

theorem buddy_system (s n : ℕ) (h1 : n ≠ 0) (h2 : s ≠ 0) : 
  (n / 4 : ℚ) = (s / 2 : ℚ) → 
  ((n / 4 + s / 2) / (n + s) : ℚ) = (1 / 3 : ℚ) := by
  sorry

end NUMINAMATH_CALUDE_buddy_system_l2775_277575


namespace NUMINAMATH_CALUDE_negation_of_product_zero_implies_factor_zero_l2775_277544

theorem negation_of_product_zero_implies_factor_zero (a b c : ℝ) :
  (¬(abc = 0 → a = 0 ∨ b = 0 ∨ c = 0)) ↔ (abc = 0 → a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0) := by
  sorry

end NUMINAMATH_CALUDE_negation_of_product_zero_implies_factor_zero_l2775_277544


namespace NUMINAMATH_CALUDE_palmer_photos_l2775_277542

/-- The number of photos Palmer has after her trip to Bali -/
def total_photos (initial_photos : ℕ) (first_week : ℕ) (third_fourth_week : ℕ) : ℕ :=
  initial_photos + first_week + 2 * first_week + third_fourth_week

/-- Theorem stating the total number of photos Palmer has after her trip -/
theorem palmer_photos : 
  total_photos 100 50 80 = 330 := by
  sorry

#eval total_photos 100 50 80

end NUMINAMATH_CALUDE_palmer_photos_l2775_277542


namespace NUMINAMATH_CALUDE_exactly_one_item_count_l2775_277530

/-- The number of households with exactly one item (car only, bike only, scooter only, or skateboard only) -/
def households_with_one_item (total : ℕ) (none : ℕ) (car_and_bike : ℕ) (car : ℕ) (bike : ℕ) (scooter : ℕ) (skateboard : ℕ) : ℕ :=
  (car - car_and_bike) + (bike - car_and_bike) + scooter + skateboard

theorem exactly_one_item_count :
  households_with_one_item 120 15 28 52 32 18 8 = 54 := by
  sorry

end NUMINAMATH_CALUDE_exactly_one_item_count_l2775_277530


namespace NUMINAMATH_CALUDE_english_score_is_98_l2775_277584

/-- Given the Mathematics score, Korean language score, and average score,
    calculate the English score. -/
def calculate_english_score (math_score : ℕ) (korean_offset : ℕ) (average_score : ℚ) : ℚ :=
  3 * average_score - (math_score : ℚ) - ((math_score : ℚ) + korean_offset)

/-- Theorem stating that under the given conditions, the English score is 98. -/
theorem english_score_is_98 :
  let math_score : ℕ := 82
  let korean_offset : ℕ := 5
  let average_score : ℚ := 89
  calculate_english_score math_score korean_offset average_score = 98 := by
  sorry

#eval calculate_english_score 82 5 89

end NUMINAMATH_CALUDE_english_score_is_98_l2775_277584


namespace NUMINAMATH_CALUDE_box_weight_difference_l2775_277555

theorem box_weight_difference (first_box_weight third_box_weight : ℕ) 
  (h1 : first_box_weight = 2)
  (h2 : third_box_weight = 13) : 
  third_box_weight - first_box_weight = 11 := by
  sorry

end NUMINAMATH_CALUDE_box_weight_difference_l2775_277555


namespace NUMINAMATH_CALUDE_number_divided_by_16_equals_16_times_8_l2775_277518

theorem number_divided_by_16_equals_16_times_8 : 
  2048 / 16 = 16 * 8 := by sorry

end NUMINAMATH_CALUDE_number_divided_by_16_equals_16_times_8_l2775_277518


namespace NUMINAMATH_CALUDE_smallest_marble_count_l2775_277587

def is_valid_marble_count (n : ℕ) : Prop :=
  n > 2 ∧ n % 6 = 2 ∧ n % 7 = 2 ∧ n % 8 = 2 ∧ n % 11 = 2

theorem smallest_marble_count :
  ∃ (n : ℕ), is_valid_marble_count n ∧ ∀ (m : ℕ), is_valid_marble_count m → n ≤ m :=
by
  use 3698
  sorry

end NUMINAMATH_CALUDE_smallest_marble_count_l2775_277587


namespace NUMINAMATH_CALUDE_remainder_problem_l2775_277548

theorem remainder_problem (n : ℤ) (h : n % 11 = 4) : (8 * n + 5) % 11 = 4 := by
  sorry

end NUMINAMATH_CALUDE_remainder_problem_l2775_277548


namespace NUMINAMATH_CALUDE_palmer_photo_ratio_l2775_277566

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

end NUMINAMATH_CALUDE_palmer_photo_ratio_l2775_277566


namespace NUMINAMATH_CALUDE_assignment_theorem_l2775_277529

/-- The number of ways to assign 4 distinct tasks to 4 people selected from 6 volunteers -/
def assignment_ways : ℕ := 360

/-- The total number of volunteers -/
def total_volunteers : ℕ := 6

/-- The number of people to be selected -/
def selected_people : ℕ := 4

/-- The number of tasks to be assigned -/
def number_of_tasks : ℕ := 4

theorem assignment_theorem : 
  assignment_ways = (total_volunteers.factorial) / ((total_volunteers - selected_people).factorial) := by
  sorry

end NUMINAMATH_CALUDE_assignment_theorem_l2775_277529


namespace NUMINAMATH_CALUDE_unique_n_satisfying_equation_l2775_277582

theorem unique_n_satisfying_equation : 
  ∃! n : ℤ, ⌊(n^2 : ℚ) / 9⌋ - ⌊(n : ℚ) / 3⌋^2 = 5 ∧ n = 14 :=
by sorry

end NUMINAMATH_CALUDE_unique_n_satisfying_equation_l2775_277582


namespace NUMINAMATH_CALUDE_volume_sphere_minus_cylinder_l2775_277598

/-- The volume of space inside a sphere and outside an inscribed right cylinder -/
theorem volume_sphere_minus_cylinder (R : ℝ) (r : ℝ) (h : ℝ) :
  R = 7 →
  r = 4 →
  h = 2 * Real.sqrt 33 →
  (4 / 3 * π * R^3 - π * r^2 * h) = ((1372 / 3 : ℝ) - 32 * Real.sqrt 33) * π :=
by sorry

end NUMINAMATH_CALUDE_volume_sphere_minus_cylinder_l2775_277598


namespace NUMINAMATH_CALUDE_factorial_difference_l2775_277561

theorem factorial_difference : Nat.factorial 10 - Nat.factorial 9 = 3265920 := by
  sorry

end NUMINAMATH_CALUDE_factorial_difference_l2775_277561


namespace NUMINAMATH_CALUDE_probability_of_one_each_item_l2775_277586

def drawer_items : ℕ := 8

def total_items : ℕ := 4 * drawer_items

def items_removed : ℕ := 4

def total_combinations : ℕ := Nat.choose total_items items_removed

def favorable_outcomes : ℕ := drawer_items^items_removed

theorem probability_of_one_each_item : 
  (favorable_outcomes : ℚ) / total_combinations = 128 / 1125 := by sorry

end NUMINAMATH_CALUDE_probability_of_one_each_item_l2775_277586


namespace NUMINAMATH_CALUDE_total_cost_calculation_l2775_277512

def shirt_cost : ℕ := 5
def hat_cost : ℕ := 4
def jeans_cost : ℕ := 10

def num_shirts : ℕ := 3
def num_hats : ℕ := 4
def num_jeans : ℕ := 2

theorem total_cost_calculation :
  shirt_cost * num_shirts + hat_cost * num_hats + jeans_cost * num_jeans = 51 := by
  sorry

end NUMINAMATH_CALUDE_total_cost_calculation_l2775_277512


namespace NUMINAMATH_CALUDE_complex_magnitude_product_l2775_277553

theorem complex_magnitude_product : 
  Complex.abs ((5 * Real.sqrt 2 - Complex.I * 3) * (2 * Real.sqrt 3 + Complex.I * 4)) = 2 * Real.sqrt 413 := by
  sorry

end NUMINAMATH_CALUDE_complex_magnitude_product_l2775_277553


namespace NUMINAMATH_CALUDE_clock_angle_at_8_30_l2775_277541

/-- The angle between the hour and minute hands at 8:30 on a standard 12-hour clock -/
def clock_angle : ℝ :=
  let numbers_on_clock : ℕ := 12
  let angle_between_numbers : ℝ := 30
  let hour_hand_position : ℝ := 8.5  -- Between 8 and 9
  let minute_hand_position : ℝ := 6
  angle_between_numbers * (minute_hand_position - hour_hand_position)

theorem clock_angle_at_8_30 : clock_angle = 75 := by
  sorry

end NUMINAMATH_CALUDE_clock_angle_at_8_30_l2775_277541


namespace NUMINAMATH_CALUDE_table_seating_theorem_l2775_277577

/-- Represents the setup of people around a round table -/
structure TableSetup where
  num_men : ℕ
  num_women : ℕ

/-- Calculates the probability of a specific man being satisfied -/
def prob_man_satisfied (setup : TableSetup) : ℚ :=
  1 - (setup.num_men - 1) / (setup.num_men + setup.num_women - 1) *
      (setup.num_men - 2) / (setup.num_men + setup.num_women - 2)

/-- Calculates the expected number of satisfied men -/
def expected_satisfied_men (setup : TableSetup) : ℚ :=
  setup.num_men * prob_man_satisfied setup

/-- Main theorem about the probability and expectation in the given setup -/
theorem table_seating_theorem (setup : TableSetup) 
    (h_men : setup.num_men = 50) (h_women : setup.num_women = 50) : 
    prob_man_satisfied setup = 25 / 33 ∧ 
    expected_satisfied_men setup = 1250 / 33 := by
  sorry

#eval prob_man_satisfied ⟨50, 50⟩
#eval expected_satisfied_men ⟨50, 50⟩

end NUMINAMATH_CALUDE_table_seating_theorem_l2775_277577


namespace NUMINAMATH_CALUDE_unique_prime_B_l2775_277576

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d > 1 → d < n → ¬(n % d = 0)

def number_form (B : ℕ) : ℕ := 1034960 + B

theorem unique_prime_B :
  ∃! B : ℕ, B < 10 ∧ is_prime (number_form B) :=
sorry

end NUMINAMATH_CALUDE_unique_prime_B_l2775_277576


namespace NUMINAMATH_CALUDE_quadratic_roots_problem_l2775_277538

theorem quadratic_roots_problem (c : ℝ) 
  (h : ∃ r : ℝ, r^2 - 3*r + c = 0 ∧ (-r)^2 + 3*(-r) - c = 0) :
  ∃ x y : ℝ, x^2 - 3*x + c = 0 ∧ y^2 - 3*y + c = 0 ∧ x = 0 ∧ y = 3 :=
sorry

end NUMINAMATH_CALUDE_quadratic_roots_problem_l2775_277538


namespace NUMINAMATH_CALUDE_units_digit_theorem_l2775_277593

-- Define a function to get the units digit of a number
def unitsDigit (n : ℕ) : ℕ := n % 10

-- Define the property we want to prove
def propertyHolds (n : ℕ) : Prop :=
  n > 0 → unitsDigit ((35 ^ n) + (93 ^ 45)) = 8

-- The theorem statement
theorem units_digit_theorem :
  ∀ n : ℕ, propertyHolds n :=
sorry

end NUMINAMATH_CALUDE_units_digit_theorem_l2775_277593


namespace NUMINAMATH_CALUDE_min_value_theorem_l2775_277583

theorem min_value_theorem (a b : ℝ) (h1 : a > b) (h2 : b > 0) :
  ∃ (m : ℝ), m = 4 ∧ (∀ x y, x > y ∧ y > 0 → x^2 + 1 / (y * (x - y)) ≥ m) ∧
  (∃ x y, x > y ∧ y > 0 ∧ x^2 + 1 / (y * (x - y)) = m) := by
  sorry

end NUMINAMATH_CALUDE_min_value_theorem_l2775_277583


namespace NUMINAMATH_CALUDE_quadratic_factorization_l2775_277573

theorem quadratic_factorization (a b c : ℤ) : 
  (∀ x, x^2 + 9*x + 14 = (x + a) * (x + b)) →
  (∀ x, x^2 + 7*x - 30 = (x + b) * (x - c)) →
  a + b + c = 15 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_factorization_l2775_277573


namespace NUMINAMATH_CALUDE_parametric_to_standard_equation_l2775_277556

/-- Given parametric equations x = 1 + (1/2)t and y = 5 + (√3/2)t,
    prove they are equivalent to the standard equation √3x - y + 5 - √3 = 0 -/
theorem parametric_to_standard_equation 
  (t x y : ℝ) 
  (h1 : x = 1 + (1/2) * t) 
  (h2 : y = 5 + (Real.sqrt 3 / 2) * t) :
  Real.sqrt 3 * x - y + 5 - Real.sqrt 3 = 0 :=
sorry

end NUMINAMATH_CALUDE_parametric_to_standard_equation_l2775_277556


namespace NUMINAMATH_CALUDE_point_line_distance_constraint_l2775_277580

/-- Given a point P(4, a) and a line 4x - 3y - 1 = 0, if the distance from P to the line
    is no greater than 3, then a is in the range [0, 10]. -/
theorem point_line_distance_constraint (a : ℝ) : 
  let P : ℝ × ℝ := (4, a)
  let line (x y : ℝ) : Prop := 4 * x - 3 * y - 1 = 0
  let distance := |4 * 4 - 3 * a - 1| / 5
  distance ≤ 3 → 0 ≤ a ∧ a ≤ 10 := by
sorry

end NUMINAMATH_CALUDE_point_line_distance_constraint_l2775_277580


namespace NUMINAMATH_CALUDE_hyperbola_axis_ratio_l2775_277507

/-- Given a hyperbola with equation x² - my² = 1, where m is a real number,
    if the length of the conjugate axis is three times that of the transverse axis,
    then m = 1/9 -/
theorem hyperbola_axis_ratio (m : ℝ) : 
  (∀ x y : ℝ, x^2 - m*y^2 = 1) → 
  (∃ a b : ℝ, a > 0 ∧ b > 0 ∧ 2*b = 3*(2*a) ∧ a^2 = 1 ∧ b^2 = 1/m) →
  m = 1/9 := by
sorry

end NUMINAMATH_CALUDE_hyperbola_axis_ratio_l2775_277507


namespace NUMINAMATH_CALUDE_carl_responsibility_l2775_277532

/-- Calculates the amount a person owes in an accident based on their fault percentage and insurance coverage -/
def calculate_personal_responsibility (total_property_damage : ℝ) (total_medical_bills : ℝ)
  (property_insurance_coverage : ℝ) (medical_insurance_coverage : ℝ) (fault_percentage : ℝ) : ℝ :=
  let remaining_property_damage := total_property_damage * (1 - property_insurance_coverage)
  let remaining_medical_bills := total_medical_bills * (1 - medical_insurance_coverage)
  fault_percentage * (remaining_property_damage + remaining_medical_bills)

/-- Theorem stating Carl's personal responsibility in the accident -/
theorem carl_responsibility :
  let total_property_damage : ℝ := 40000
  let total_medical_bills : ℝ := 70000
  let property_insurance_coverage : ℝ := 0.8
  let medical_insurance_coverage : ℝ := 0.75
  let carl_fault_percentage : ℝ := 0.6
  calculate_personal_responsibility total_property_damage total_medical_bills
    property_insurance_coverage medical_insurance_coverage carl_fault_percentage = 15300 := by
  sorry

end NUMINAMATH_CALUDE_carl_responsibility_l2775_277532


namespace NUMINAMATH_CALUDE_book_gain_percent_l2775_277503

theorem book_gain_percent (marked_price : ℝ) (marked_price_pos : marked_price > 0) : 
  let cost_price := 0.64 * marked_price
  let selling_price := 0.88 * marked_price
  let profit := selling_price - cost_price
  let gain_percent := (profit / cost_price) * 100
  gain_percent = 37.5 := by sorry

end NUMINAMATH_CALUDE_book_gain_percent_l2775_277503


namespace NUMINAMATH_CALUDE_perpendicular_bisector_segments_theorem_l2775_277564

structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  n_a : ℝ
  n_b : ℝ
  n_c : ℝ
  h_positive : 0 < a ∧ 0 < b ∧ 0 < c
  h_triangle_inequality : a + b > c ∧ b + c > a ∧ c + a > b
  h_order : a < b ∧ b < c

theorem perpendicular_bisector_segments_theorem (t : Triangle) :
  t.n_a > t.n_b ∧ t.n_c > t.n_b ∧
  ∃ (t1 t2 : Triangle), t1.n_a > t1.n_c ∧ t2.n_c > t2.n_a :=
by sorry

end NUMINAMATH_CALUDE_perpendicular_bisector_segments_theorem_l2775_277564


namespace NUMINAMATH_CALUDE_inequality_solution_set_l2775_277597

theorem inequality_solution_set :
  {x : ℝ | x * (x - 1) > 0} = {x : ℝ | x < 0 ∨ x > 1} := by
sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l2775_277597


namespace NUMINAMATH_CALUDE_max_reciprocal_sum_l2775_277526

theorem max_reciprocal_sum (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : x^2 + y^2 = 1) :
  (1 / x + 1 / y) ≤ 2 * Real.sqrt 2 ∧ 
  ∃ (x₀ y₀ : ℝ), x₀ > 0 ∧ y₀ > 0 ∧ x₀^2 + y₀^2 = 1 ∧ 1 / x₀ + 1 / y₀ = 2 * Real.sqrt 2 :=
sorry

end NUMINAMATH_CALUDE_max_reciprocal_sum_l2775_277526


namespace NUMINAMATH_CALUDE_andre_flowers_l2775_277534

theorem andre_flowers (initial_flowers : ℝ) (total_flowers : ℕ) 
  (h1 : initial_flowers = 67.0)
  (h2 : total_flowers = 157) :
  ↑total_flowers - initial_flowers = 90 := by
  sorry

end NUMINAMATH_CALUDE_andre_flowers_l2775_277534


namespace NUMINAMATH_CALUDE_string_average_length_l2775_277594

theorem string_average_length : 
  let string1 : ℝ := 2
  let string2 : ℝ := 6
  let string3 : ℝ := 9
  let num_strings : ℕ := 3
  (string1 + string2 + string3) / num_strings = 17 / 3 := by
  sorry

end NUMINAMATH_CALUDE_string_average_length_l2775_277594


namespace NUMINAMATH_CALUDE_trapezoid_ratio_l2775_277502

/-- Represents a trapezoid ABCD with a point P inside -/
structure Trapezoid :=
  (AB CD : ℝ)
  (height : ℝ)
  (area_PCD area_PAD area_PBC area_PAB : ℝ)

/-- The theorem stating the ratio of parallel sides in the trapezoid -/
theorem trapezoid_ratio (T : Trapezoid) : 
  T.AB > T.CD →
  T.height = 8 →
  T.area_PCD = 4 →
  T.area_PAD = 6 →
  T.area_PBC = 5 →
  T.area_PAB = 7 →
  T.AB / T.CD = 4 := by
  sorry


end NUMINAMATH_CALUDE_trapezoid_ratio_l2775_277502


namespace NUMINAMATH_CALUDE_number_of_girls_l2775_277574

theorem number_of_girls (total_pupils : ℕ) (boys : ℕ) (girls : ℕ) : 
  total_pupils = 929 → boys = 387 → girls = total_pupils - boys → girls = 542 := by
sorry

end NUMINAMATH_CALUDE_number_of_girls_l2775_277574


namespace NUMINAMATH_CALUDE_cos_seven_pi_sixths_l2775_277581

theorem cos_seven_pi_sixths : Real.cos (7 * π / 6) = -Real.sqrt 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_cos_seven_pi_sixths_l2775_277581


namespace NUMINAMATH_CALUDE_fibonacci_determinant_identity_fibonacci_1002_1004_minus_1003_squared_l2775_277531

def fib : ℕ → ℤ
  | 0 => 0
  | 1 => 1
  | n + 2 => fib (n + 1) + fib n

theorem fibonacci_determinant_identity (n : ℕ) (h : n > 0) :
  fib (n - 1) * fib (n + 1) - fib n ^ 2 = (-1) ^ n := by
  sorry

-- The specific case for n = 1003
theorem fibonacci_1002_1004_minus_1003_squared :
  fib 1002 * fib 1004 - fib 1003 ^ 2 = -1 := by
  sorry

end NUMINAMATH_CALUDE_fibonacci_determinant_identity_fibonacci_1002_1004_minus_1003_squared_l2775_277531


namespace NUMINAMATH_CALUDE_triangle_angle_relation_l2775_277589

theorem triangle_angle_relation (a b c α β γ : ℝ) : 
  b = (a + c) / Real.sqrt 2 →
  β = (α + γ) / 2 →
  c > a →
  γ = α + 90 :=
sorry

end NUMINAMATH_CALUDE_triangle_angle_relation_l2775_277589


namespace NUMINAMATH_CALUDE_vector_operation_result_l2775_277547

theorem vector_operation_result :
  let a : ℝ × ℝ × ℝ := (3, -2, 1)
  let b : ℝ × ℝ × ℝ := (-2, 4, 0)
  let c : ℝ × ℝ × ℝ := (3, 0, 2)
  a - 2 • b + 4 • c = (19, -10, 9) :=
by sorry

end NUMINAMATH_CALUDE_vector_operation_result_l2775_277547
