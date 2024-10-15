import Mathlib

namespace NUMINAMATH_CALUDE_largest_divisor_of_n_squared_divisible_by_18_l2176_217663

theorem largest_divisor_of_n_squared_divisible_by_18 (n : ℕ+) (h : 18 ∣ n^2) :
  6 = Nat.gcd 6 n ∧ ∀ m : ℕ, m ∣ n → m ≤ 6 :=
sorry

end NUMINAMATH_CALUDE_largest_divisor_of_n_squared_divisible_by_18_l2176_217663


namespace NUMINAMATH_CALUDE_unique_solution_linear_system_l2176_217696

theorem unique_solution_linear_system (x y z : ℝ) :
  (3*x + 2*y + 2*z = 13) ∧
  (2*x + 3*y + 2*z = 14) ∧
  (2*x + 2*y + 3*z = 15) ↔
  (x = 1 ∧ y = 2 ∧ z = 3) :=
by sorry

end NUMINAMATH_CALUDE_unique_solution_linear_system_l2176_217696


namespace NUMINAMATH_CALUDE_max_value_product_l2176_217641

theorem max_value_product (x y z : ℝ) (hx : x ≥ 0) (hy : y ≥ 0) (hz : z ≥ 0) (hsum : x + y + z = 3) :
  (x^2 + x*y + y^2) * (y^2 + y*z + z^2) * (z^2 + z*x + x^2) ≤ 27 :=
by sorry

end NUMINAMATH_CALUDE_max_value_product_l2176_217641


namespace NUMINAMATH_CALUDE_investment_scientific_notation_l2176_217659

-- Define the value in billion yuan
def investment : ℝ := 845

-- Define the scientific notation representation
def scientific_notation : ℝ := 8.45 * (10 ^ 3)

-- Theorem statement
theorem investment_scientific_notation : investment = scientific_notation := by
  sorry

end NUMINAMATH_CALUDE_investment_scientific_notation_l2176_217659


namespace NUMINAMATH_CALUDE_least_n_for_g_prime_product_l2176_217690

def g (n : ℕ) : ℕ := n.choose 3

def isArithmeticProgression (p₁ p₂ p₃ : ℕ) (d : ℕ) : Prop :=
  p₂ = p₁ + d ∧ p₃ = p₂ + d

theorem least_n_for_g_prime_product : 
  ∃ (p₁ p₂ p₃ : ℕ),
    p₁.Prime ∧ p₂.Prime ∧ p₃.Prime ∧
    p₁ < p₂ ∧ p₂ < p₃ ∧
    isArithmeticProgression p₁ p₂ p₃ 336 ∧
    g 2019 = p₁ * p₂ * p₃ ∧
    (∀ n < 2019, ¬∃ (q₁ q₂ q₃ : ℕ),
      q₁.Prime ∧ q₂.Prime ∧ q₃.Prime ∧
      q₁ < q₂ ∧ q₂ < q₃ ∧
      isArithmeticProgression q₁ q₂ q₃ 336 ∧
      g n = q₁ * q₂ * q₃) :=
by sorry

end NUMINAMATH_CALUDE_least_n_for_g_prime_product_l2176_217690


namespace NUMINAMATH_CALUDE_min_value_of_f_l2176_217635

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a^(x-2) - 2*a

theorem min_value_of_f (a : ℝ) (ha : a > 0) (ha' : a ≠ 1) (hf : f a 2 = 1/3) :
  ∃ (m : ℝ), IsMinOn (f a) (Set.Icc 0 3) m ∧ m = -1/3 := by sorry

end NUMINAMATH_CALUDE_min_value_of_f_l2176_217635


namespace NUMINAMATH_CALUDE_expression_equality_l2176_217628

theorem expression_equality (a b c : ℤ) 
  (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) 
  (hab : a ≠ b) (hbc : b ≠ c) (hac : a ≠ c) : 
  (a^2 - b^2) / (a * b) - (a * b - b * c) / (a * b - a * c) = (c * a - (c - 1) * b) / b :=
by sorry

end NUMINAMATH_CALUDE_expression_equality_l2176_217628


namespace NUMINAMATH_CALUDE_expression_evaluation_l2176_217672

theorem expression_evaluation :
  (4^4 - 4*(4-2)^4)^(4+1) = 14889702426 := by sorry

end NUMINAMATH_CALUDE_expression_evaluation_l2176_217672


namespace NUMINAMATH_CALUDE_fraction_to_decimal_l2176_217660

theorem fraction_to_decimal : (11 : ℚ) / 125 = (88 : ℚ) / 1000 := by sorry

end NUMINAMATH_CALUDE_fraction_to_decimal_l2176_217660


namespace NUMINAMATH_CALUDE_marble_probability_difference_l2176_217625

/-- The number of red marbles in the box -/
def red_marbles : ℕ := 1001

/-- The number of black marbles in the box -/
def black_marbles : ℕ := 1001

/-- The total number of marbles in the box -/
def total_marbles : ℕ := red_marbles + black_marbles

/-- The probability of drawing two marbles of the same color -/
def P_same : ℚ := (red_marbles.choose 2 + black_marbles.choose 2 : ℚ) / total_marbles.choose 2

/-- The probability of drawing two marbles of different colors -/
def P_diff : ℚ := (red_marbles * black_marbles : ℚ) / total_marbles.choose 2

/-- The theorem stating that the absolute difference between P_same and P_diff is 1/2001 -/
theorem marble_probability_difference : |P_same - P_diff| = 1 / 2001 := by
  sorry

end NUMINAMATH_CALUDE_marble_probability_difference_l2176_217625


namespace NUMINAMATH_CALUDE_average_weight_increase_l2176_217616

/-- Proves that the increase in average weight when including a teacher is 400 grams -/
theorem average_weight_increase (num_students : Nat) (avg_weight_students : ℝ) (teacher_weight : ℝ) :
  num_students = 24 →
  avg_weight_students = 35 →
  teacher_weight = 45 →
  ((num_students + 1) * ((num_students * avg_weight_students + teacher_weight) / (num_students + 1)) -
   (num_students * avg_weight_students)) * 1000 = 400 := by
  sorry

end NUMINAMATH_CALUDE_average_weight_increase_l2176_217616


namespace NUMINAMATH_CALUDE_fraction_equality_l2176_217608

theorem fraction_equality (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) 
  (h : (5*x - 2*y) / (2*x + 3*y) = 1) : 
  (2*x - 5*y) / (5*x + 2*y) = -5/31 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_l2176_217608


namespace NUMINAMATH_CALUDE_no_solution_range_l2176_217624

theorem no_solution_range (a : ℝ) : 
  (∀ x : ℝ, |x - 5| + |x + 3| ≥ a) → a ∈ Set.Iic 8 := by
  sorry

end NUMINAMATH_CALUDE_no_solution_range_l2176_217624


namespace NUMINAMATH_CALUDE_cube_surface_area_increase_l2176_217686

theorem cube_surface_area_increase (L : ℝ) (h : L > 0) :
  let original_area := 6 * L^2
  let new_length := 1.6 * L
  let new_area := 6 * new_length^2
  (new_area - original_area) / original_area * 100 = 156 := by
  sorry

end NUMINAMATH_CALUDE_cube_surface_area_increase_l2176_217686


namespace NUMINAMATH_CALUDE_percent_of_x_l2176_217631

theorem percent_of_x (x : ℝ) (h : x > 0) : (x / 5 + x / 25) / x = 24 / 100 := by
  sorry

end NUMINAMATH_CALUDE_percent_of_x_l2176_217631


namespace NUMINAMATH_CALUDE_yolandas_walking_rate_l2176_217644

/-- Proves that Yolanda's walking rate is 5 miles per hour given the problem conditions -/
theorem yolandas_walking_rate
  (total_distance : ℝ)
  (bobs_rate : ℝ)
  (time_difference : ℝ)
  (bobs_distance : ℝ)
  (h1 : total_distance = 60)
  (h2 : bobs_rate = 6)
  (h3 : time_difference = 1)
  (h4 : bobs_distance = 30) :
  (total_distance - bobs_distance) / (bobs_distance / bobs_rate + time_difference) = 5 :=
by sorry

end NUMINAMATH_CALUDE_yolandas_walking_rate_l2176_217644


namespace NUMINAMATH_CALUDE_expression_evaluation_l2176_217654

theorem expression_evaluation (x y : ℝ) 
  (h : (x + 2)^2 + |y - 2/3| = 0) : 
  1/2 * x - 2 * (x - 1/3 * y^2) + (-3/2 * x + 1/3 * y^2) = 6 + 4/9 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l2176_217654


namespace NUMINAMATH_CALUDE_function_equation_solution_l2176_217680

open Real

theorem function_equation_solution (f : ℝ → ℝ) (h : ∀ x ∈ Set.Ioo (-1) 1, 2 * f x - f (-x) = log (x + 1)) :
  ∀ x ∈ Set.Ioo (-1) 1, f x = (2/3) * log (x + 1) + (1/3) * log (1 - x) := by
  sorry

end NUMINAMATH_CALUDE_function_equation_solution_l2176_217680


namespace NUMINAMATH_CALUDE_fifth_term_value_l2176_217666

def geometric_sequence (a : ℕ → ℝ) (r : ℝ) : Prop :=
  ∀ n : ℕ, a (n + 1) = r * a n

theorem fifth_term_value
  (a : ℕ → ℝ)
  (h_geom : geometric_sequence a 2)
  (h_positive : ∀ n, a n > 0)
  (h_product : a 3 * a 11 = 16) :
  a 5 = 1 :=
sorry

end NUMINAMATH_CALUDE_fifth_term_value_l2176_217666


namespace NUMINAMATH_CALUDE_inequality_solution_set_l2176_217695

/-- The solution set of the inequality -x^2 - 2x + 3 > 0 is the open interval (-3, 1) -/
theorem inequality_solution_set : 
  {x : ℝ | -x^2 - 2*x + 3 > 0} = Set.Ioo (-3) 1 := by sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l2176_217695


namespace NUMINAMATH_CALUDE_quadratic_roots_relation_l2176_217633

theorem quadratic_roots_relation (m n p q : ℝ) (hm : m ≠ 0) (hn : n ≠ 0) (hp : p ≠ 0) (hq : q ≠ 0) :
  (∃ s₁ s₂ : ℝ, s₁ + s₂ = -p ∧ s₁ * s₂ = q ∧
               3 * s₁ + 3 * s₂ = -m ∧ 9 * s₁ * s₂ = n) →
  n / q = 9 := by
sorry

end NUMINAMATH_CALUDE_quadratic_roots_relation_l2176_217633


namespace NUMINAMATH_CALUDE_negation_equivalence_l2176_217683

theorem negation_equivalence :
  (¬ ∀ x : ℝ, x ∈ Set.Icc 1 2 → 2 * x^2 - 3 ≥ 0) ↔
  (∃ x : ℝ, x ∈ Set.Icc 1 2 ∧ 2 * x^2 - 3 < 0) :=
by sorry

end NUMINAMATH_CALUDE_negation_equivalence_l2176_217683


namespace NUMINAMATH_CALUDE_certain_number_value_value_is_232_l2176_217636

theorem certain_number_value : ℤ → ℤ → Prop :=
  fun n value => 5 * n - 28 = value

theorem value_is_232 (n : ℤ) (value : ℤ) 
  (h1 : n = 52) 
  (h2 : certain_number_value n value) : 
  value = 232 := by
sorry

end NUMINAMATH_CALUDE_certain_number_value_value_is_232_l2176_217636


namespace NUMINAMATH_CALUDE_origin_outside_circle_l2176_217698

theorem origin_outside_circle (a : ℝ) (h : 0 < a ∧ a < 1) :
  let f : ℝ × ℝ → ℝ := λ (x, y) ↦ x^2 + y^2 + 2*a*x + 2*y + (a-1)^2
  f (0, 0) > 0 := by sorry

end NUMINAMATH_CALUDE_origin_outside_circle_l2176_217698


namespace NUMINAMATH_CALUDE_triangle_perimeter_after_tripling_l2176_217692

theorem triangle_perimeter_after_tripling (a b c : ℝ) :
  a = 8 → b = 15 → c = 17 →
  (a + b > c) ∧ (a + c > b) ∧ (b + c > a) →
  (3 * a + 3 * b + 3 * c = 120) :=
by
  sorry

end NUMINAMATH_CALUDE_triangle_perimeter_after_tripling_l2176_217692


namespace NUMINAMATH_CALUDE_certain_number_problem_l2176_217646

theorem certain_number_problem (x : ℝ) : 0.75 * x = 0.5 * 900 → x = 600 := by
  sorry

end NUMINAMATH_CALUDE_certain_number_problem_l2176_217646


namespace NUMINAMATH_CALUDE_parabola_intersection_range_l2176_217620

/-- A parabola defined by y = ax^2 -/
def Parabola (a : ℝ) : ℝ → ℝ := λ x ↦ a * x^2

/-- A line passing through (1, -2) with slope k -/
def Line (k : ℝ) : ℝ → ℝ := λ x ↦ k * (x - 1) - 2

/-- Checks if a parabola intersects a line -/
def intersects (p : ℝ → ℝ) (l : ℝ → ℝ) : Prop :=
  ∃ x : ℝ, p x = l x

theorem parabola_intersection_range (a : ℝ) :
  (∃ k : ℝ, k ≠ 0 ∧ (intersects (Parabola a) (Line k) ∨ intersects (Parabola a) (Line (-1/k)))) →
  a < 0 ∨ (0 < a ∧ a ≤ 1/8) := by
  sorry

end NUMINAMATH_CALUDE_parabola_intersection_range_l2176_217620


namespace NUMINAMATH_CALUDE_simple_interest_rate_l2176_217688

/-- Simple interest calculation -/
theorem simple_interest_rate (principal time interest : ℚ) (h1 : principal = 23) 
  (h2 : time = 3) (h3 : interest = 3.45) : 
  interest / (principal * time) = 0.05 := by
  sorry

end NUMINAMATH_CALUDE_simple_interest_rate_l2176_217688


namespace NUMINAMATH_CALUDE_sally_pen_distribution_l2176_217604

/-- Represents the problem of distributing pens to students --/
def pen_distribution (total_pens : ℕ) (num_students : ℕ) (pens_home : ℕ) : ℕ → Prop :=
  λ pens_per_student : ℕ =>
    let pens_given := pens_per_student * num_students
    let remainder := total_pens - pens_given
    let pens_in_locker := remainder / 2
    pens_in_locker + pens_home = remainder

theorem sally_pen_distribution :
  pen_distribution 342 44 17 7 := by
  sorry

#check sally_pen_distribution

end NUMINAMATH_CALUDE_sally_pen_distribution_l2176_217604


namespace NUMINAMATH_CALUDE_composition_of_functions_l2176_217634

theorem composition_of_functions (f g : ℝ → ℝ) :
  (∀ x, f x = 5 - 2 * x) →
  (∀ x, g x = x^2 + x + 1) →
  f (g (Real.sqrt 3)) = -3 - 2 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_composition_of_functions_l2176_217634


namespace NUMINAMATH_CALUDE_factorization_3m_squared_minus_12_l2176_217676

theorem factorization_3m_squared_minus_12 (m : ℝ) : 3 * m^2 - 12 = 3 * (m - 2) * (m + 2) := by
  sorry

end NUMINAMATH_CALUDE_factorization_3m_squared_minus_12_l2176_217676


namespace NUMINAMATH_CALUDE_units_digit_of_product_l2176_217611

theorem units_digit_of_product (a b c : ℕ) : 
  (2^1501 * 5^1602 * 11^1703) % 10 = 0 := by sorry

end NUMINAMATH_CALUDE_units_digit_of_product_l2176_217611


namespace NUMINAMATH_CALUDE_complex_fraction_simplification_l2176_217648

theorem complex_fraction_simplification :
  let i : ℂ := Complex.I
  (2 - 2*i) / (3 + 4*i) = -2/25 - 14/25*i :=
by
  sorry

end NUMINAMATH_CALUDE_complex_fraction_simplification_l2176_217648


namespace NUMINAMATH_CALUDE_bookstore_shipment_problem_l2176_217607

theorem bookstore_shipment_problem :
  ∀ (B : ℕ), 
    (70 : ℚ) / 100 * B = 45 →
    B = 64 :=
by
  sorry

end NUMINAMATH_CALUDE_bookstore_shipment_problem_l2176_217607


namespace NUMINAMATH_CALUDE_probability_not_same_intersection_is_two_thirds_l2176_217650

/-- Represents the number of officers -/
def num_officers : ℕ := 3

/-- Represents the number of intersections -/
def num_intersections : ℕ := 2

/-- The total number of possible arrangements -/
def total_arrangements : ℕ := (num_officers.choose 2) * 2

/-- The number of arrangements where two specific officers are at the same intersection -/
def same_intersection_arrangements : ℕ := 2

/-- The probability that two specific officers are not at the same intersection -/
def probability_not_same_intersection : ℚ := 1 - (same_intersection_arrangements : ℚ) / total_arrangements

theorem probability_not_same_intersection_is_two_thirds :
  probability_not_same_intersection = 2/3 := by sorry

end NUMINAMATH_CALUDE_probability_not_same_intersection_is_two_thirds_l2176_217650


namespace NUMINAMATH_CALUDE_arithmetic_sequence_value_l2176_217623

def is_arithmetic_sequence (x y z : ℝ) : Prop :=
  y - x = z - y

theorem arithmetic_sequence_value :
  ∀ a : ℝ, is_arithmetic_sequence 2 a 10 → a = 6 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_value_l2176_217623


namespace NUMINAMATH_CALUDE_grouping_theorem_l2176_217613

/- Define the number of men and women -/
def num_men : ℕ := 4
def num_women : ℕ := 5

/- Define the size of each group -/
def group_size : ℕ := 3

/- Define the total number of groups -/
def num_groups : ℕ := 3

/- Define the function to calculate the number of ways to group people -/
def group_ways : ℕ :=
  let first_group_men := 1
  let first_group_women := 2
  let second_group_men := 2
  let second_group_women := 1
  (num_men.choose first_group_men * num_women.choose first_group_women) *
  ((num_men - first_group_men).choose second_group_men * (num_women - first_group_women).choose second_group_women)

/- Theorem statement -/
theorem grouping_theorem :
  group_ways = 360 :=
sorry

end NUMINAMATH_CALUDE_grouping_theorem_l2176_217613


namespace NUMINAMATH_CALUDE_tan_ratio_from_sin_sum_diff_l2176_217640

theorem tan_ratio_from_sin_sum_diff (x y : ℝ) 
  (h1 : Real.sin (x + y) = 5/8) 
  (h2 : Real.sin (x - y) = 1/4) : 
  Real.tan x / Real.tan y = 2 := by
  sorry

end NUMINAMATH_CALUDE_tan_ratio_from_sin_sum_diff_l2176_217640


namespace NUMINAMATH_CALUDE_product_of_squared_terms_l2176_217652

theorem product_of_squared_terms (x : ℝ) : 3 * x^2 * (2 * x^2) = 6 * x^4 := by
  sorry

end NUMINAMATH_CALUDE_product_of_squared_terms_l2176_217652


namespace NUMINAMATH_CALUDE_equation_conditions_l2176_217673

theorem equation_conditions (a b c d : ℝ) :
  (2*a + 3*b) / (b + 2*c) = (3*c + 2*d) / (d + 2*a) →
  (2*a = 3*c) ∨ (2*a + 3*b + d + 2*c = 0) :=
by sorry

end NUMINAMATH_CALUDE_equation_conditions_l2176_217673


namespace NUMINAMATH_CALUDE_line_plane_perpendicular_l2176_217667

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relations
variable (perpendicular : Line → Plane → Prop)
variable (parallel : Line → Line → Prop)
variable (subset : Line → Plane → Prop)
variable (plane_perpendicular : Plane → Plane → Prop)

-- State the theorem
theorem line_plane_perpendicular 
  (m n : Line) (α β : Plane) :
  perpendicular m α →
  parallel m n →
  subset n β →
  plane_perpendicular α β :=
sorry

end NUMINAMATH_CALUDE_line_plane_perpendicular_l2176_217667


namespace NUMINAMATH_CALUDE_sin_n_equals_cos_682_l2176_217681

theorem sin_n_equals_cos_682 (n : ℤ) : 
  -180 ≤ n ∧ n ≤ 180 ∧ Real.sin (n * π / 180) = Real.cos (682 * π / 180) → n = 128 := by
  sorry

end NUMINAMATH_CALUDE_sin_n_equals_cos_682_l2176_217681


namespace NUMINAMATH_CALUDE_sin_cos_sixth_power_sum_l2176_217603

theorem sin_cos_sixth_power_sum (θ : Real) (h : Real.cos (2 * θ) = 1 / 5) :
  Real.sin θ ^ 6 + Real.cos θ ^ 6 = 7 / 25 := by
  sorry

end NUMINAMATH_CALUDE_sin_cos_sixth_power_sum_l2176_217603


namespace NUMINAMATH_CALUDE_binomial_coefficient_ratio_l2176_217694

theorem binomial_coefficient_ratio (n k : ℕ) : 
  (Nat.choose n k : ℚ) / (Nat.choose n (k + 1) : ℚ) = 1 / 3 ∧
  (Nat.choose n (k + 1) : ℚ) / (Nat.choose n (k + 2) : ℚ) = 1 / 2 →
  n + k = 9 := by
sorry

end NUMINAMATH_CALUDE_binomial_coefficient_ratio_l2176_217694


namespace NUMINAMATH_CALUDE_min_xy_point_l2176_217619

theorem min_xy_point (x y : ℝ) (h_pos_x : x > 0) (h_pos_y : y > 0)
  (h_eq : 1/x + 1/(2*y) + 3/(2*x*y) = 1) :
  x * y ≥ 9/2 ∧ (x * y = 9/2 ↔ x = 3 ∧ y = 3/2) := by
  sorry

end NUMINAMATH_CALUDE_min_xy_point_l2176_217619


namespace NUMINAMATH_CALUDE_product_pure_imaginary_implies_a_equals_six_l2176_217627

theorem product_pure_imaginary_implies_a_equals_six :
  ∀ (a : ℝ), 
  (∃ (b : ℝ), (a + 2*I) * (1 + 3*I) = b*I ∧ b ≠ 0) →
  a = 6 := by
sorry

end NUMINAMATH_CALUDE_product_pure_imaginary_implies_a_equals_six_l2176_217627


namespace NUMINAMATH_CALUDE_catches_ratio_l2176_217685

theorem catches_ratio (joe_catches tammy_catches derek_catches : ℕ) : 
  joe_catches = 23 →
  tammy_catches = 30 →
  tammy_catches = derek_catches / 3 + 16 →
  derek_catches / joe_catches = 42 / 23 := by
  sorry

end NUMINAMATH_CALUDE_catches_ratio_l2176_217685


namespace NUMINAMATH_CALUDE_abc_bad_theorem_l2176_217682

def is_valid_quadruple (A B C D : ℕ) : Prop :=
  A ≠ 0 ∧ B ≠ 0 ∧ D ≠ 0 ∧ C ≠ 0 ∧
  A ≠ B ∧ A ≠ C ∧ A ≠ D ∧ B ≠ C ∧ B ≠ D ∧ C ≠ D ∧
  A < 10 ∧ B < 10 ∧ C < 10 ∧ D < 10 ∧
  (100 * A + 10 * B + C) * D = (100 * B + 10 * A + D) * C

def solution_set : Set (ℕ × ℕ × ℕ × ℕ) :=
  {(2,1,7,4), (1,2,4,7), (8,1,9,2), (1,8,2,9), (7,2,8,3), (2,7,3,8), (6,3,7,4), (3,6,4,7)}

theorem abc_bad_theorem :
  {q : ℕ × ℕ × ℕ × ℕ | is_valid_quadruple q.1 q.2.1 q.2.2.1 q.2.2.2} = solution_set :=
sorry

end NUMINAMATH_CALUDE_abc_bad_theorem_l2176_217682


namespace NUMINAMATH_CALUDE_fraction_to_decimal_equivalence_l2176_217678

theorem fraction_to_decimal_equivalence : (1 : ℚ) / 4 = (25 : ℚ) / 100 := by sorry

end NUMINAMATH_CALUDE_fraction_to_decimal_equivalence_l2176_217678


namespace NUMINAMATH_CALUDE_limit_f_at_infinity_l2176_217637

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := ((a^x - 1) / (x * (a - 1)))^(1/x)

theorem limit_f_at_infinity (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) :
  (∀ ε > 0, ∃ N, ∀ x ≥ N, |f a x - (if a > 1 then a else 1)| < ε) :=
sorry

end NUMINAMATH_CALUDE_limit_f_at_infinity_l2176_217637


namespace NUMINAMATH_CALUDE_log_inequalities_l2176_217618

-- Define the logarithm functions
noncomputable def log₃ (x : ℝ) := Real.log x / Real.log 3
noncomputable def log₁₃ (x : ℝ) := Real.log x / Real.log (1/3)

-- State the theorem
theorem log_inequalities :
  (∀ x y, x < y → log₃ x < log₃ y) →  -- log₃ is increasing
  (∀ x y, x < y → log₁₃ x > log₁₃ y) →  -- log₁₃ is decreasing
  (1/5)^0 = 1 →
  log₃ 4 > (1/5)^0 ∧ (1/5)^0 > log₁₃ 10 :=
by sorry

end NUMINAMATH_CALUDE_log_inequalities_l2176_217618


namespace NUMINAMATH_CALUDE_f_2017_equals_one_l2176_217600

theorem f_2017_equals_one (f : ℝ → ℝ)
  (h1 : ∀ x, f (x + 2) = f x)
  (h2 : ∀ θ, f (Real.cos θ) = Real.cos (2 * θ)) :
  f 2017 = 1 := by
  sorry

end NUMINAMATH_CALUDE_f_2017_equals_one_l2176_217600


namespace NUMINAMATH_CALUDE_largest_r_is_two_l2176_217610

/-- A sequence of positive integers satisfying the given inequality -/
def ValidSequence (a : ℕ → ℕ) (r : ℝ) : Prop :=
  ∀ n : ℕ, (a n ≤ a (n + 2)) ∧ ((a (n + 2))^2 ≤ (a n)^2 + r * a (n + 1))

/-- The sequence eventually stabilizes -/
def EventuallyStable (a : ℕ → ℕ) : Prop :=
  ∃ M : ℕ, ∀ n ≥ M, a (n + 2) = a n

/-- The main theorem stating that 2 is the largest real number satisfying the condition -/
theorem largest_r_is_two :
  (∀ a : ℕ → ℕ, ValidSequence a 2 → EventuallyStable a) ∧
  (∀ r > 2, ∃ a : ℕ → ℕ, ValidSequence a r ∧ ¬EventuallyStable a) := by
  sorry

end NUMINAMATH_CALUDE_largest_r_is_two_l2176_217610


namespace NUMINAMATH_CALUDE_complex_expansion_l2176_217642

-- Define the complex number i
noncomputable def i : ℂ := Complex.I

-- State the theorem
theorem complex_expansion : (1 - i) * (1 + 2*i)^2 = 1 + 7*i := by
  sorry

end NUMINAMATH_CALUDE_complex_expansion_l2176_217642


namespace NUMINAMATH_CALUDE_f_eight_minus_f_four_l2176_217669

def is_odd (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

def has_period (f : ℝ → ℝ) (p : ℝ) : Prop := ∀ x, f (x + p) = f x

theorem f_eight_minus_f_four (f : ℝ → ℝ) 
  (h_odd : is_odd f)
  (h_period : has_period f 5)
  (h1 : f 1 = 1)
  (h2 : f 2 = 3) :
  f 8 - f 4 = -2 := by
  sorry

end NUMINAMATH_CALUDE_f_eight_minus_f_four_l2176_217669


namespace NUMINAMATH_CALUDE_newspaper_ad_cost_newspaper_ad_cost_proof_l2176_217602

/-- The total cost for three companies purchasing ads in a newspaper -/
theorem newspaper_ad_cost (num_companies : ℕ) (num_ad_spaces : ℕ) 
  (ad_length : ℝ) (ad_width : ℝ) (cost_per_sqft : ℝ) : ℝ :=
  let ad_area := ad_length * ad_width
  let cost_per_ad := ad_area * cost_per_sqft
  let cost_per_company := cost_per_ad * num_ad_spaces
  num_companies * cost_per_company

/-- Proof that the total cost for three companies purchasing 10 ad spaces each, 
    where each ad space is a 12-foot by 5-foot rectangle and costs $60 per square foot, 
    is $108,000 -/
theorem newspaper_ad_cost_proof :
  newspaper_ad_cost 3 10 12 5 60 = 108000 := by
  sorry

end NUMINAMATH_CALUDE_newspaper_ad_cost_newspaper_ad_cost_proof_l2176_217602


namespace NUMINAMATH_CALUDE_root_equation_solution_l2176_217605

theorem root_equation_solution (a b c : ℕ) (ha : a > 1) (hb : b > 1) (hc : c > 1) :
  (∀ N : ℝ, N ≠ 1 → N^(1/a + 1/(a*b) + 1/(a*b*c)) = N^(25/36)) → b = 3 := by
  sorry

end NUMINAMATH_CALUDE_root_equation_solution_l2176_217605


namespace NUMINAMATH_CALUDE_factorization_equality_l2176_217664

theorem factorization_equality (a b : ℝ) : 3*a - 9*a*b = 3*a*(1 - 3*b) := by
  sorry

end NUMINAMATH_CALUDE_factorization_equality_l2176_217664


namespace NUMINAMATH_CALUDE_three_digit_subtraction_l2176_217621

/-- Given a three-digit number of the form 4c1 and another of the form 3d5,
    prove that if 786 - 4c1 = 3d5 and 3d5 is divisible by 7, then c + d = 8 -/
theorem three_digit_subtraction (c d : ℕ) : 
  (786 - (400 + c * 10 + 1) = 300 + d * 10 + 5) →
  (300 + d * 10 + 5) % 7 = 0 →
  c + d = 8 := by
sorry

end NUMINAMATH_CALUDE_three_digit_subtraction_l2176_217621


namespace NUMINAMATH_CALUDE_sequence_properties_l2176_217612

def is_root (a : ℝ) : Prop := a^2 - 3*a - 5 = 0

def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∀ n m : ℕ, a (n + 1) - a n = a (m + 1) - a m

def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∀ n m : ℕ, a (n + 1) / a n = a (m + 1) / a m

theorem sequence_properties (a : ℕ → ℝ) :
  (is_root (a 3) ∧ is_root (a 10) ∧ arithmetic_sequence a → a 5 + a 8 = 3) ∧
  (is_root (a 3) ∧ is_root (a 10) ∧ geometric_sequence a → a 6 * a 7 = -5) :=
sorry

end NUMINAMATH_CALUDE_sequence_properties_l2176_217612


namespace NUMINAMATH_CALUDE_factor_expression_l2176_217657

theorem factor_expression (x : ℝ) : 36 * x^2 + 24 * x = 12 * x * (3 * x + 2) := by
  sorry

end NUMINAMATH_CALUDE_factor_expression_l2176_217657


namespace NUMINAMATH_CALUDE_expression_value_l2176_217606

theorem expression_value (α : Real) (h : Real.tan α = -3/4) :
  (3 * (Real.sin (α/2))^2 + 2 * Real.sin (α/2) * Real.cos (α/2) + (Real.cos (α/2))^2 - 2) /
  (Real.sin (π/2 + α) * Real.tan (-3*π + α) + Real.cos (6*π - α)) = -7 :=
by sorry

end NUMINAMATH_CALUDE_expression_value_l2176_217606


namespace NUMINAMATH_CALUDE_rectangle_area_l2176_217629

theorem rectangle_area (side : ℝ) (h1 : side > 0) :
  let perimeter := 8 * side
  let area := 4 * side^2
  perimeter = 160 → area = 1600 := by
sorry

end NUMINAMATH_CALUDE_rectangle_area_l2176_217629


namespace NUMINAMATH_CALUDE_roots_square_sum_l2176_217691

theorem roots_square_sum (a b : ℝ) : 
  (∀ x, x^2 - 2*x - 1 = 0 ↔ x = a ∨ x = b) → a^2 + b^2 = 6 := by
sorry

end NUMINAMATH_CALUDE_roots_square_sum_l2176_217691


namespace NUMINAMATH_CALUDE_triangle_exists_l2176_217661

/-- Triangle inequality theorem for a triangle with sides a, b, and c -/
def triangle_inequality (a b c : ℝ) : Prop :=
  a + b > c ∧ b + c > a ∧ c + a > b

/-- A theorem stating that a triangle can be formed with side lengths 6, 8, and 13 -/
theorem triangle_exists : triangle_inequality 6 8 13 := by
  sorry

end NUMINAMATH_CALUDE_triangle_exists_l2176_217661


namespace NUMINAMATH_CALUDE_remaining_capacity_theorem_l2176_217674

/-- Represents the meal capacity and consumption for a trekking group --/
structure MealCapacity where
  adult_capacity : ℕ
  child_capacity : ℕ
  adults_eaten : ℕ

/-- Calculates the number of children that can be catered with the remaining food --/
def remaining_child_capacity (m : MealCapacity) : ℕ :=
  sorry

/-- Theorem stating that given the specific meal capacity and consumption, 
    the remaining food can cater to 45 children --/
theorem remaining_capacity_theorem (m : MealCapacity) 
  (h1 : m.adult_capacity = 70)
  (h2 : m.child_capacity = 90)
  (h3 : m.adults_eaten = 35) :
  remaining_child_capacity m = 45 := by
  sorry

end NUMINAMATH_CALUDE_remaining_capacity_theorem_l2176_217674


namespace NUMINAMATH_CALUDE_special_triangle_sides_l2176_217697

/-- A triangle ABC with specific properties -/
structure SpecialTriangle where
  /-- Vertex B of the triangle -/
  B : ℝ × ℝ
  /-- Equation of the altitude on side AB: ax + by + c = 0 -/
  altitude : ℝ × ℝ × ℝ
  /-- Equation of the angle bisector of angle A: dx + ey + f = 0 -/
  angle_bisector : ℝ × ℝ × ℝ

/-- Theorem about the equations of sides in a special triangle -/
theorem special_triangle_sides 
  (t : SpecialTriangle) 
  (h1 : t.B = (-2, 0))
  (h2 : t.altitude = (1, 3, -26))
  (h3 : t.angle_bisector = (1, 1, -2)) :
  ∃ (AB AC : ℝ × ℝ × ℝ),
    AB = (3, -1, 6) ∧ 
    AC = (1, -3, 10) := by
  sorry

end NUMINAMATH_CALUDE_special_triangle_sides_l2176_217697


namespace NUMINAMATH_CALUDE_inscribed_sphere_radius_l2176_217643

/-- Given a regular tetrahedron formed by the centers of four spheres in a
    triangular pyramid of ten equal spheres, prove that the radius of the
    sphere inscribed at the center of the tetrahedron is √6 - 1, given that
    the radius of the circumscribed sphere of the tetrahedron is 5√2 + 5. -/
theorem inscribed_sphere_radius (R : ℝ) (r : ℝ) :
  R = 5 * Real.sqrt 2 + 5 →
  r = Real.sqrt 6 - 1 :=
by sorry

end NUMINAMATH_CALUDE_inscribed_sphere_radius_l2176_217643


namespace NUMINAMATH_CALUDE_trig_expression_simplification_l2176_217693

theorem trig_expression_simplification (α : ℝ) :
  (Real.sin (π - α) * Real.sin (3 * π - α) + Real.sin (-α - π) * Real.sin (α - 2 * π)) /
  (Real.sin (4 * π - α) * Real.sin (5 * π + α)) = -2 := by
  sorry

end NUMINAMATH_CALUDE_trig_expression_simplification_l2176_217693


namespace NUMINAMATH_CALUDE_sue_buttons_l2176_217609

theorem sue_buttons (mari kendra sue : ℕ) : 
  mari = 8 →
  kendra = 5 * mari + 4 →
  sue = kendra / 2 →
  sue = 22 := by
sorry

end NUMINAMATH_CALUDE_sue_buttons_l2176_217609


namespace NUMINAMATH_CALUDE_imaginary_unit_multiplication_l2176_217658

theorem imaginary_unit_multiplication (i : ℂ) : i * i = -1 → i * (1 + i) = -1 + i := by
  sorry

end NUMINAMATH_CALUDE_imaginary_unit_multiplication_l2176_217658


namespace NUMINAMATH_CALUDE_percentage_difference_l2176_217615

theorem percentage_difference (x y : ℝ) (h : x = 3 * y) :
  (x - y) / x * 100 = 200 / 3 := by
sorry

end NUMINAMATH_CALUDE_percentage_difference_l2176_217615


namespace NUMINAMATH_CALUDE_necessary_condition_for_existence_l2176_217651

theorem necessary_condition_for_existence (a : ℝ) :
  (∃ x ∈ Set.Icc 1 2, x^2 - a > 0) → a ≤ 4 := by
  sorry

end NUMINAMATH_CALUDE_necessary_condition_for_existence_l2176_217651


namespace NUMINAMATH_CALUDE_circle_sum_formula_l2176_217622

/-- The sum of numbers on a circle after n divisions -/
def circle_sum (n : ℕ) : ℝ :=
  2 * 3^n

/-- Theorem stating the sum of numbers on the circle after n divisions -/
theorem circle_sum_formula (n : ℕ) : circle_sum n = 2 * 3^n := by
  sorry

end NUMINAMATH_CALUDE_circle_sum_formula_l2176_217622


namespace NUMINAMATH_CALUDE_water_price_problem_l2176_217653

/-- The residential water price problem -/
theorem water_price_problem (last_year_price : ℝ) 
  (h1 : last_year_price > 0)
  (h2 : 30 / (1.2 * last_year_price) - 15 / last_year_price = 5) : 
  1.2 * last_year_price = 6 := by
  sorry

#check water_price_problem

end NUMINAMATH_CALUDE_water_price_problem_l2176_217653


namespace NUMINAMATH_CALUDE_difference_2020th_2010th_term_l2176_217649

-- Define the arithmetic sequence
def arithmeticSequence (n : ℕ) : ℤ :=
  -10 + (n - 1) * 9

-- State the theorem
theorem difference_2020th_2010th_term :
  (arithmeticSequence 2020 - arithmeticSequence 2010).natAbs = 90 := by
  sorry

end NUMINAMATH_CALUDE_difference_2020th_2010th_term_l2176_217649


namespace NUMINAMATH_CALUDE_jelly_bean_problem_l2176_217679

theorem jelly_bean_problem (b c : ℕ) : 
  b = 3 * c →                  -- Initial ratio
  b - 5 = 5 * (c - 15) →       -- Ratio after eating jelly beans
  b = 105 :=                   -- Conclusion
by sorry

end NUMINAMATH_CALUDE_jelly_bean_problem_l2176_217679


namespace NUMINAMATH_CALUDE_inequality_and_equality_condition_l2176_217601

theorem inequality_and_equality_condition (a b c d : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0) :
  a^3 + b^3 + c^3 + d^3 ≥ a^2*b + b^2*c + c^2*d + d^2*a ∧
  (a^3 + b^3 + c^3 + d^3 = a^2*b + b^2*c + c^2*d + d^2*a ↔ a = b ∧ b = c ∧ c = d) :=
by sorry

end NUMINAMATH_CALUDE_inequality_and_equality_condition_l2176_217601


namespace NUMINAMATH_CALUDE_consecutive_squareful_numbers_l2176_217656

/-- A natural number is squareful if it has a square divisor greater than 1 -/
def IsSquareful (n : ℕ) : Prop :=
  ∃ m : ℕ, m > 1 ∧ m * m ∣ n

/-- For any natural number k, there exist k consecutive squareful numbers -/
theorem consecutive_squareful_numbers :
  ∀ k : ℕ, ∃ n : ℕ, ∀ i : ℕ, i < k → IsSquareful (n + i) :=
sorry

end NUMINAMATH_CALUDE_consecutive_squareful_numbers_l2176_217656


namespace NUMINAMATH_CALUDE_equation_solution_l2176_217647

theorem equation_solution : ∃ x : ℝ, (2 / (3 * x) = 1 / (x + 2)) ∧ x = 4 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l2176_217647


namespace NUMINAMATH_CALUDE_building_heights_properties_l2176_217614

/-- Heights of the buildings in meters -/
def burj_khalifa : ℕ := 828
def shanghai_tower : ℕ := 632
def one_world_trade_center : ℕ := 541
def willis_tower : ℕ := 527

/-- List of building heights -/
def building_heights : List ℕ := [burj_khalifa, shanghai_tower, one_world_trade_center, willis_tower]

/-- Theorem stating the total height and average height difference -/
theorem building_heights_properties :
  (building_heights.sum = 2528) ∧
  (((building_heights.map (λ h => h - willis_tower)).sum : ℚ) / 4 = 105) := by
  sorry

end NUMINAMATH_CALUDE_building_heights_properties_l2176_217614


namespace NUMINAMATH_CALUDE_triangle_area_l2176_217665

/-- Given a triangle ABC with sides a, b, c opposite to angles A, B, C respectively,
    prove that if a^2 + b^2 - c^2 = 6√3 - 2ab and C = 60°, then the area of triangle ABC is 3/2. -/
theorem triangle_area (a b c : ℝ) (h1 : a^2 + b^2 - c^2 = 6 * Real.sqrt 3 - 2*a*b) 
  (h2 : Real.cos (Real.pi / 3) = 1/2) : 
  (1/2) * a * b * Real.sin (Real.pi / 3) = 3/2 := by
  sorry

end NUMINAMATH_CALUDE_triangle_area_l2176_217665


namespace NUMINAMATH_CALUDE_syllogism_structure_l2176_217645

-- Define syllogism as a structure in deductive reasoning
structure Syllogism where
  major_premise : Prop
  minor_premise : Prop
  conclusion : Prop

-- Define deductive reasoning
def DeductiveReasoning : Type := Prop → Prop

-- Theorem stating that syllogism in deductive reasoning consists of major premise, minor premise, and conclusion
theorem syllogism_structure (dr : DeductiveReasoning) :
  ∃ (s : Syllogism), dr s.major_premise ∧ dr s.minor_premise ∧ dr s.conclusion :=
sorry

end NUMINAMATH_CALUDE_syllogism_structure_l2176_217645


namespace NUMINAMATH_CALUDE_complex_equation_solution_l2176_217671

theorem complex_equation_solution (i : ℂ) (a : ℝ) :
  i * i = -1 →
  (1 + i) * (a - i) = 3 + i →
  a = 2 := by
sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l2176_217671


namespace NUMINAMATH_CALUDE_inequality_solution_set_l2176_217699

theorem inequality_solution_set (x : ℝ) :
  {x : ℝ | x^4 - 16*x^2 - 36*x > 0} = {x : ℝ | x < -4 ∨ (-4 < x ∧ x < -1) ∨ x > 9} :=
by sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l2176_217699


namespace NUMINAMATH_CALUDE_bullet_speed_difference_wild_bill_scenario_l2176_217689

/-- The speed difference of a bullet fired from a moving horse -/
theorem bullet_speed_difference (v_horse : ℝ) (v_bullet : ℝ) :
  v_horse > 0 → v_bullet > v_horse →
  (v_bullet + v_horse) - (v_bullet - v_horse) = 2 * v_horse := by
  sorry

/-- Wild Bill's scenario -/
theorem wild_bill_scenario :
  let v_horse : ℝ := 20
  let v_bullet : ℝ := 400
  (v_bullet + v_horse) - (v_bullet - v_horse) = 40 := by
  sorry

end NUMINAMATH_CALUDE_bullet_speed_difference_wild_bill_scenario_l2176_217689


namespace NUMINAMATH_CALUDE_trigonometric_product_equals_one_l2176_217639

theorem trigonometric_product_equals_one : 
  let cos30 : ℝ := Real.sqrt 3 / 2
  let sin60 : ℝ := Real.sqrt 3 / 2
  let sin30 : ℝ := 1 / 2
  let cos60 : ℝ := 1 / 2
  (1 - 1 / cos30) * (1 + 1 / sin60) * (1 - 1 / sin30) * (1 + 1 / cos60) = 1 := by
  sorry

end NUMINAMATH_CALUDE_trigonometric_product_equals_one_l2176_217639


namespace NUMINAMATH_CALUDE_no_natural_solution_l2176_217655

theorem no_natural_solution : ¬∃ (m n : ℕ), m * n * (m + n) = 2020 := by
  sorry

end NUMINAMATH_CALUDE_no_natural_solution_l2176_217655


namespace NUMINAMATH_CALUDE_bus_left_seats_count_l2176_217617

/-- Represents the seating arrangement in a bus -/
structure BusSeating where
  left_seats : ℕ
  right_seats : ℕ
  back_seat_capacity : ℕ
  seat_capacity : ℕ
  total_capacity : ℕ

/-- The bus seating arrangement satisfies the given conditions -/
def valid_bus_seating (bus : BusSeating) : Prop :=
  bus.right_seats = bus.left_seats - 3 ∧
  bus.back_seat_capacity = 7 ∧
  bus.seat_capacity = 3 ∧
  bus.total_capacity = 88 ∧
  bus.total_capacity = bus.seat_capacity * (bus.left_seats + bus.right_seats) + bus.back_seat_capacity

/-- The number of seats on the left side of the bus is 15 -/
theorem bus_left_seats_count (bus : BusSeating) (h : valid_bus_seating bus) : bus.left_seats = 15 := by
  sorry


end NUMINAMATH_CALUDE_bus_left_seats_count_l2176_217617


namespace NUMINAMATH_CALUDE_min_value_of_f_l2176_217630

/-- A cubic function with a constant term. -/
def f (m : ℝ) (x : ℝ) : ℝ := 2 * x^3 - 6 * x + m

/-- The theorem stating the minimum value of f on [0, 2] given its maximum value. -/
theorem min_value_of_f (m : ℝ) :
  (∃ x ∈ Set.Icc 0 2, ∀ y ∈ Set.Icc 0 2, f m y ≤ f m x) ∧
  (∀ x ∈ Set.Icc 0 2, f m x ≤ 3) →
  ∃ x ∈ Set.Icc 0 2, f m x = -1 ∧ ∀ y ∈ Set.Icc 0 2, -1 ≤ f m y :=
by sorry

end NUMINAMATH_CALUDE_min_value_of_f_l2176_217630


namespace NUMINAMATH_CALUDE_f_min_value_a_range_characterization_l2176_217677

-- Define the function f(x)
def f (x : ℝ) : ℝ := 2 * abs (x - 2) - x + 5

-- Theorem for the minimum value of f(x)
theorem f_min_value : ∃ (m : ℝ), (∀ x, f x ≥ m) ∧ (∃ x₀, f x₀ = m) ∧ m = 3 := by
  sorry

-- Define the set of valid values for a
def valid_a_set : Set ℝ := {a | a ≤ -5 ∨ a ≥ 1}

-- Theorem for the range of a
theorem a_range_characterization (a : ℝ) : 
  (∀ x : ℝ, abs (x - a) + abs (x + 2) ≥ 3) ↔ a ∈ valid_a_set := by
  sorry

end NUMINAMATH_CALUDE_f_min_value_a_range_characterization_l2176_217677


namespace NUMINAMATH_CALUDE_simplify_expression_l2176_217670

theorem simplify_expression (x : ℝ) : 3 * (5 - 2 * x) - 2 * (4 + 3 * x) = 7 - 12 * x := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l2176_217670


namespace NUMINAMATH_CALUDE_circle_coverage_l2176_217668

-- Define a circle structure
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

-- Define a function to check if one circle can cover another
def canCover (c1 c2 : Circle) : Prop :=
  ∀ p : ℝ × ℝ, (p.1 - c2.center.1)^2 + (p.2 - c2.center.2)^2 ≤ c2.radius^2 →
    (p.1 - c1.center.1)^2 + (p.2 - c1.center.2)^2 ≤ c1.radius^2

-- Theorem statement
theorem circle_coverage (M1 M2 : Circle) (h : M2.radius > M1.radius) :
  canCover M2 M1 ∧ ¬(canCover M1 M2) := by
  sorry

end NUMINAMATH_CALUDE_circle_coverage_l2176_217668


namespace NUMINAMATH_CALUDE_x_equals_one_sufficient_not_necessary_l2176_217684

theorem x_equals_one_sufficient_not_necessary :
  (∃ x : ℝ, x^2 + x - 2 = 0 ∧ x ≠ 1) ∧
  (∀ x : ℝ, x = 1 → x^2 + x - 2 = 0) :=
by sorry

end NUMINAMATH_CALUDE_x_equals_one_sufficient_not_necessary_l2176_217684


namespace NUMINAMATH_CALUDE_ninth_term_is_nine_l2176_217687

/-- An arithmetic sequence with specific properties -/
structure ArithmeticSequence where
  /-- The first term of the sequence -/
  a : ℝ
  /-- The common difference of the sequence -/
  d : ℝ
  /-- The sum of the first six terms is 21 -/
  sum_first_six : a + (a + d) + (a + 2*d) + (a + 3*d) + (a + 4*d) + (a + 5*d) = 21
  /-- The seventh term is 7 -/
  seventh_term : a + 6*d = 7

/-- The ninth term of the arithmetic sequence is 9 -/
theorem ninth_term_is_nine (seq : ArithmeticSequence) : seq.a + 8*seq.d = 9 := by
  sorry

end NUMINAMATH_CALUDE_ninth_term_is_nine_l2176_217687


namespace NUMINAMATH_CALUDE_bees_after_six_days_l2176_217632

/-- Calculates the number of bees in the beehive after n days -/
def bees_in_hive (n : ℕ) : ℕ :=
  let a₁ : ℕ := 4  -- Initial term (1 original bee + 3 companions)
  let q : ℕ := 3   -- Common ratio (each bee brings 3 companions)
  a₁ * (q^n - 1) / (q - 1)

/-- Theorem stating that the number of bees after 6 days is 1456 -/
theorem bees_after_six_days :
  bees_in_hive 6 = 1456 := by
  sorry

end NUMINAMATH_CALUDE_bees_after_six_days_l2176_217632


namespace NUMINAMATH_CALUDE_cindy_marbles_problem_l2176_217638

theorem cindy_marbles_problem (initial_marbles : ℕ) (num_friends : ℕ) (marbles_per_friend : ℕ) :
  initial_marbles = 1000 →
  num_friends = 6 →
  marbles_per_friend = 120 →
  7 * (initial_marbles - num_friends * marbles_per_friend) = 1960 :=
by
  sorry

end NUMINAMATH_CALUDE_cindy_marbles_problem_l2176_217638


namespace NUMINAMATH_CALUDE_square_difference_divided_by_nine_l2176_217675

theorem square_difference_divided_by_nine : (121^2 - 112^2) / 9 = 233 := by sorry

end NUMINAMATH_CALUDE_square_difference_divided_by_nine_l2176_217675


namespace NUMINAMATH_CALUDE_hyperbola_eccentricity_l2176_217662

/-- The eccentricity of the hyperbola 3x^2 - y^2 = 3 is 2 -/
theorem hyperbola_eccentricity : ∃ (e : ℝ), e = 2 ∧ 
  ∀ (x y : ℝ), 3 * x^2 - y^2 = 3 → 
  e = (Real.sqrt ((3 * x^2 + y^2) / 3)) / (Real.sqrt (3 * x^2 / 3)) := by
  sorry

end NUMINAMATH_CALUDE_hyperbola_eccentricity_l2176_217662


namespace NUMINAMATH_CALUDE_odd_function_max_value_l2176_217626

-- Define the function f on (-∞, 0)
def f (x : ℝ) : ℝ := x * (1 + x)

-- State the theorem
theorem odd_function_max_value :
  (∀ x < 0, f x = x * (1 + x)) →  -- f is defined as x(1+x) on (-∞, 0)
  (∀ x : ℝ, f (-x) = -f x) →      -- f is an odd function
  (∃ M : ℝ, M = 1/4 ∧ ∀ x > 0, f x ≤ M) -- Maximum value on (0, +∞) is 1/4
  := by sorry

end NUMINAMATH_CALUDE_odd_function_max_value_l2176_217626
