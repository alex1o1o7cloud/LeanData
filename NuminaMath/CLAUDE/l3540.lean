import Mathlib

namespace polynomial_coefficients_l3540_354075

-- Define the polynomial
def p (x y : ℝ) : ℝ := 3 * x * y^2 - 2 * y - 1

-- State the theorem
theorem polynomial_coefficients :
  (∃ a b c d : ℝ, ∀ x y : ℝ, p x y = a * x * y^2 + b * y + c * x + d) ∧
  (∀ a b c d : ℝ, (∀ x y : ℝ, p x y = a * x * y^2 + b * y + c * x + d) →
    b = -2 ∧ d = -1) :=
by sorry

end polynomial_coefficients_l3540_354075


namespace reverse_digit_increase_l3540_354087

def is_valid_number (n : ℕ) : Prop :=
  ∃ (a b c : ℕ),
    n = 100 * a + 10 * b + c ∧
    a + b + c = 10 ∧
    b = a + c ∧
    n = 253

theorem reverse_digit_increase (n : ℕ) (h : is_valid_number n) :
  ∃ (a b c : ℕ),
    n = 100 * a + 10 * b + c ∧
    100 * c + 10 * b + a - n = 99 := by
  sorry

end reverse_digit_increase_l3540_354087


namespace smallest_percent_increase_l3540_354029

def question_value : Fin 15 → ℕ
  | 0 => 100
  | 1 => 300
  | 2 => 600
  | 3 => 800
  | 4 => 1500
  | 5 => 3000
  | 6 => 4500
  | 7 => 7000
  | 8 => 10000
  | 9 => 15000
  | 10 => 30000
  | 11 => 45000
  | 12 => 75000
  | 13 => 150000
  | 14 => 300000

def percent_increase (a b : ℕ) : ℚ :=
  (b - a : ℚ) / a * 100

def options : List (Fin 15 × Fin 15) :=
  [(1, 2), (3, 4), (6, 7), (11, 12), (13, 14)]

theorem smallest_percent_increase :
  ∀ (pair : Fin 15 × Fin 15),
    pair ∈ options →
    percent_increase (question_value pair.1) (question_value pair.2) ≥ 
    percent_increase (question_value 6) (question_value 7) :=
by sorry

end smallest_percent_increase_l3540_354029


namespace fixed_point_theorem_l3540_354000

/-- The ellipse C -/
def C (x y : ℝ) : Prop := x^2/4 + y^2/3 = 1

/-- The right focus F of the ellipse C -/
def F : ℝ × ℝ := (1, 0)

/-- The line L -/
def L (x : ℝ) : Prop := x = 4

/-- The left vertex A of the ellipse C -/
def A : ℝ × ℝ := (-2, 0)

/-- The ratio condition for any point P on C -/
def ratio_condition (P : ℝ × ℝ) : Prop :=
  C P.1 P.2 → 2 * Real.sqrt ((P.1 - F.1)^2 + (P.2 - F.2)^2) = |P.1 - 4|

/-- The theorem to be proved -/
theorem fixed_point_theorem :
  ∀ D E M N : ℝ × ℝ,
  (∃ t : ℝ, C (F.1 + t * (D.1 - F.1)) (F.2 + t * (D.2 - F.2))) →
  (∃ t : ℝ, C (F.1 + t * (E.1 - F.1)) (F.2 + t * (E.2 - F.2))) →
  (∃ t : ℝ, M = (4, A.2 + t * (D.2 - A.2))) →
  (∃ t : ℝ, N = (4, A.2 + t * (E.2 - A.2))) →
  (∀ P : ℝ × ℝ, ratio_condition P) →
  ∃ O : ℝ × ℝ, O = (1, 0) ∧ 
    (M.1 - O.1)^2 + (M.2 - O.2)^2 = (N.1 - O.1)^2 + (N.2 - O.2)^2 :=
by
  sorry

end fixed_point_theorem_l3540_354000


namespace cosine_inequality_l3540_354049

theorem cosine_inequality (y : ℝ) (hy : 0 ≤ y ∧ y ≤ 2 * Real.pi) :
  ∀ x, 0 ≤ x ∧ x ≤ 2 * Real.pi → Real.cos (x - y) ≥ Real.cos x - Real.cos y := by
  sorry

end cosine_inequality_l3540_354049


namespace sean_patch_selling_price_l3540_354093

/-- Proves that the selling price per patch is $12 given the conditions of Sean's patch business. -/
theorem sean_patch_selling_price
  (num_patches : ℕ)
  (cost_per_patch : ℚ)
  (net_profit : ℚ)
  (h_num_patches : num_patches = 100)
  (h_cost_per_patch : cost_per_patch = 1.25)
  (h_net_profit : net_profit = 1075) :
  (cost_per_patch * num_patches + net_profit) / num_patches = 12 := by
  sorry

end sean_patch_selling_price_l3540_354093


namespace deepak_age_l3540_354073

theorem deepak_age (rahul_age deepak_age : ℕ) : 
  (rahul_age : ℚ) / (deepak_age : ℚ) = 4 / 3 →
  rahul_age + 6 = 26 →
  deepak_age = 15 := by
sorry

end deepak_age_l3540_354073


namespace pineapple_cost_theorem_l3540_354035

/-- The cost of each pineapple before shipping -/
def pineapple_cost_before_shipping (n : ℕ) (shipping_cost total_cost_per_pineapple : ℚ) : ℚ :=
  total_cost_per_pineapple - (shipping_cost / n)

/-- Theorem: The cost of each pineapple before shipping is $1.25 -/
theorem pineapple_cost_theorem (n : ℕ) (shipping_cost total_cost_per_pineapple : ℚ) 
  (h1 : n = 12)
  (h2 : shipping_cost = 21)
  (h3 : total_cost_per_pineapple = 3) :
  pineapple_cost_before_shipping n shipping_cost total_cost_per_pineapple = 5/4 := by
  sorry

end pineapple_cost_theorem_l3540_354035


namespace x4_plus_y4_equals_47_l3540_354041

theorem x4_plus_y4_equals_47 (x y : ℝ) 
  (h1 : x^2 + 1/x^2 = 7) 
  (h2 : x*y = 1) : 
  x^4 + y^4 = 47 := by
sorry

end x4_plus_y4_equals_47_l3540_354041


namespace greatest_three_digit_divisible_by_5_and_10_l3540_354033

theorem greatest_three_digit_divisible_by_5_and_10 : ∃ n : ℕ, 
  (100 ≤ n ∧ n ≤ 999) ∧ 
  n % 5 = 0 ∧ 
  n % 10 = 0 ∧ 
  ∀ m : ℕ, (100 ≤ m ∧ m ≤ 999 ∧ m % 5 = 0 ∧ m % 10 = 0) → m ≤ n :=
by
  -- Proof goes here
  sorry

end greatest_three_digit_divisible_by_5_and_10_l3540_354033


namespace largest_two_digit_divisible_by_six_ending_in_four_l3540_354055

def is_two_digit (n : ℕ) : Prop := 10 ≤ n ∧ n ≤ 99

def ends_in_four (n : ℕ) : Prop := n % 10 = 4

theorem largest_two_digit_divisible_by_six_ending_in_four :
  ∀ n : ℕ, is_two_digit n → n % 6 = 0 → ends_in_four n → n ≤ 84 :=
sorry

end largest_two_digit_divisible_by_six_ending_in_four_l3540_354055


namespace polynomial_factorization_l3540_354026

theorem polynomial_factorization (a x : ℝ) : 
  a * x^3 + x + a + 1 = (x + 1) * (a * x^2 - a * x + a + 1) := by
  sorry

end polynomial_factorization_l3540_354026


namespace statue_weight_theorem_l3540_354046

/-- Calculates the weight of a marble statue after a series of reductions --/
def final_statue_weight (initial_weight : ℝ) : ℝ :=
  let week1 := initial_weight * (1 - 0.35)
  let week2 := week1 * (1 - 0.20)
  let week3 := week2 * (1 - 0.05)^5
  let after_rain := week3 * (1 - 0.02)
  let week4 := after_rain * (1 - 0.08)
  let final := week4 * (1 - 0.25)
  final

/-- The weight of the final statue is approximately 136.04 kg --/
theorem statue_weight_theorem (ε : ℝ) (h : ε > 0) :
  ∃ δ : ℝ, δ > 0 ∧ |final_statue_weight 500 - 136.04| < δ ∧ δ < ε :=
sorry

end statue_weight_theorem_l3540_354046


namespace complex_magnitude_l3540_354013

theorem complex_magnitude (w : ℂ) (h : w^2 = -48 + 14*I) : Complex.abs w = 5 * Real.sqrt 2 := by
  sorry

end complex_magnitude_l3540_354013


namespace range_of_a_l3540_354019

-- Define the sets A and B
def A : Set ℝ := {x : ℝ | -2 < x ∧ x < 4}
def B (a : ℝ) : Set ℝ := {x : ℝ | x < a}

-- Define the theorem
theorem range_of_a (a : ℝ) : 
  (A ∪ B a = {x : ℝ | x < 4}) ↔ (-2 < a ∧ a ≤ 4) :=
sorry

end range_of_a_l3540_354019


namespace solutions_to_z_sixth_eq_neg_64_l3540_354017

theorem solutions_to_z_sixth_eq_neg_64 :
  {z : ℂ | z^6 = -64} =
    {2 * (Complex.cos (π / 6) + Complex.I * Complex.sin (π / 6)),
     2 * (Complex.cos (π / 2) + Complex.I * Complex.sin (π / 2)),
     2 * (Complex.cos (5 * π / 6) + Complex.I * Complex.sin (5 * π / 6)),
     2 * (Complex.cos (7 * π / 6) + Complex.I * Complex.sin (7 * π / 6)),
     2 * (Complex.cos (3 * π / 2) + Complex.I * Complex.sin (3 * π / 2)),
     2 * (Complex.cos (11 * π / 6) + Complex.I * Complex.sin (11 * π / 6))} :=
by sorry

end solutions_to_z_sixth_eq_neg_64_l3540_354017


namespace one_fifth_equals_point_two_l3540_354060

theorem one_fifth_equals_point_two : (1 : ℚ) / 5 = (2 : ℚ) / 10 := by
  sorry

end one_fifth_equals_point_two_l3540_354060


namespace sequence_periodicity_l3540_354071

def is_periodic (a : ℕ → ℕ) : Prop :=
  ∃ (p : ℕ), p > 0 ∧ ∀ (n : ℕ), a (n + p) = a n

theorem sequence_periodicity (a : ℕ → ℕ) 
  (h1 : ∀ n, a n < 1988)
  (h2 : ∀ m n, (a m + a n) % a (m + n) = 0) :
  is_periodic a := by
  sorry

end sequence_periodicity_l3540_354071


namespace completing_square_equivalence_l3540_354032

theorem completing_square_equivalence :
  ∀ x : ℝ, x^2 + 4*x + 1 = 0 ↔ (x + 2)^2 = 3 := by
sorry

end completing_square_equivalence_l3540_354032


namespace min_value_of_x_plus_y_l3540_354094

theorem min_value_of_x_plus_y (x y : ℝ) 
  (hx : x > 0) 
  (hy : y > 0) 
  (h : 9 / x + 1 / y = 1) : 
  x + y ≥ 16 ∧ ∃ (x₀ y₀ : ℝ), x₀ > 0 ∧ y₀ > 0 ∧ 9 / x₀ + 1 / y₀ = 1 ∧ x₀ + y₀ = 16 :=
sorry

end min_value_of_x_plus_y_l3540_354094


namespace only_one_divides_power_minus_one_l3540_354084

theorem only_one_divides_power_minus_one :
  ∀ n : ℕ, n ≥ 1 → (n ∣ 2^n - 1) → n = 1 := by
  sorry

end only_one_divides_power_minus_one_l3540_354084


namespace simplify_and_evaluate_l3540_354048

theorem simplify_and_evaluate (x : ℝ) (h : x = -2) :
  x / (x^2 - 1) / (1 + 1 / (x - 1)) = -1 := by
  sorry

end simplify_and_evaluate_l3540_354048


namespace ribbon_shortage_l3540_354056

theorem ribbon_shortage (total_ribbon : ℝ) (num_gifts : ℕ) (ribbon_per_gift : ℝ) (ribbon_per_bow : ℝ) :
  total_ribbon = 18 →
  num_gifts = 6 →
  ribbon_per_gift = 2 →
  ribbon_per_bow = 1.5 →
  total_ribbon - (num_gifts * ribbon_per_gift + num_gifts * ribbon_per_bow) = -3 := by
  sorry

end ribbon_shortage_l3540_354056


namespace vector_magnitude_problem_l3540_354009

/-- Given vectors a and b in ℝ², where b = (-1, 2) and a + b = (1, 3),
    prove that the magnitude of a - 2b is equal to 5. -/
theorem vector_magnitude_problem (a b : ℝ × ℝ) 
  (h1 : b = (-1, 2))
  (h2 : a + b = (1, 3)) :
  ‖a - 2 • b‖ = 5 := by
  sorry

end vector_magnitude_problem_l3540_354009


namespace lcm_16_24_l3540_354068

theorem lcm_16_24 : Nat.lcm 16 24 = 48 := by
  sorry

end lcm_16_24_l3540_354068


namespace intersection_of_logarithmic_curves_l3540_354062

theorem intersection_of_logarithmic_curves :
  ∃! x : ℝ, x > 0 ∧ 3 * Real.log x = Real.log (3 * x) :=
by sorry

end intersection_of_logarithmic_curves_l3540_354062


namespace line_equation_through_points_l3540_354085

/-- The line passing through points (-1, 0) and (0, 1) is represented by the equation x - y + 1 = 0 -/
theorem line_equation_through_points : 
  ∀ (x y : ℝ), (x = -1 ∧ y = 0) ∨ (x = 0 ∧ y = 1) → x - y + 1 = 0 :=
by sorry

end line_equation_through_points_l3540_354085


namespace existence_of_special_integers_l3540_354099

theorem existence_of_special_integers :
  ∃ (A : Fin 10 → ℕ+),
    (∀ i j : Fin 10, i ≠ j → ¬(A i ∣ A j)) ∧
    (∀ i j : Fin 10, i ≠ j → (A i)^2 ∣ A j) :=
by sorry

end existence_of_special_integers_l3540_354099


namespace binomial_square_constant_l3540_354052

/-- If 9x^2 - 18x + c is the square of a binomial, then c = 9 -/
theorem binomial_square_constant (c : ℝ) : 
  (∃ a b : ℝ, ∀ x, 9*x^2 - 18*x + c = (a*x + b)^2) → c = 9 := by
sorry

end binomial_square_constant_l3540_354052


namespace tangerine_sales_theorem_l3540_354044

/-- Represents the daily sales data for a week -/
def sales_data : List Int := [300, -400, -200, 100, -600, 1200, 500]

/-- The planned daily sales amount in kilograms -/
def planned_daily_sales : Nat := 20000

/-- The selling price per kilogram in yuan -/
def selling_price : Nat := 6

/-- The express delivery cost and other expenses per kilogram in yuan -/
def expenses : Nat := 2

/-- The number of days in a week -/
def days_in_week : Nat := 7

theorem tangerine_sales_theorem :
  (List.maximum? sales_data).isSome ∧ 
  (List.minimum? sales_data).isSome →
  (∃ max min : Int, 
    (List.maximum? sales_data) = some max ∧
    (List.minimum? sales_data) = some min ∧
    max - min = 1800) ∧
  (planned_daily_sales * days_in_week + (List.sum sales_data)) * 
    (selling_price - expenses) = 563600 := by
  sorry

end tangerine_sales_theorem_l3540_354044


namespace fraction_simplification_l3540_354043

theorem fraction_simplification (m : ℝ) (h : m ≠ 3 ∧ m ≠ -3) :
  (m^2 - 3*m) / (9 - m^2) = -m / (m + 3) :=
by sorry

end fraction_simplification_l3540_354043


namespace circle_area_ratio_l3540_354030

theorem circle_area_ratio (R_A R_B R_C : ℝ) 
  (h1 : (60 / 360) * (2 * π * R_A) = (40 / 360) * (2 * π * R_B))
  (h2 : (30 / 360) * (2 * π * R_B) = (90 / 360) * (2 * π * R_C)) :
  (π * R_A^2) / (π * R_C^2) = 2 := by
  sorry

end circle_area_ratio_l3540_354030


namespace nail_multiple_l3540_354003

theorem nail_multiple (violet_nails : ℕ) (total_nails : ℕ) (M : ℕ) : 
  violet_nails = 27 →
  total_nails = 39 →
  violet_nails = M * (total_nails - violet_nails) + 3 →
  M = 2 := by sorry

end nail_multiple_l3540_354003


namespace hyperbola_ellipse_theorem_l3540_354031

-- Define the hyperbola
def hyperbola (x y : ℝ) : Prop := x^2 - y^2/2 = 1

-- Define the ellipse C
def ellipse_C (x y a b : ℝ) : Prop := x^2/a^2 + y^2/b^2 = 1

-- Define the line l
def line_l (x y k : ℝ) : Prop := y = k * (x - 1)

-- Define the complementary angles condition
def complementary_angles (x_A y_A x_B y_B m : ℝ) : Prop :=
  (y_A / (x_A - m)) + (y_B / (x_B - m)) = 0

theorem hyperbola_ellipse_theorem :
  ∀ (a b : ℝ), a > b ∧ b > 0 →
  (∃ (x_F y_F : ℝ), hyperbola x_F y_F ∧ 
    (∀ (x y : ℝ), ellipse_C x y a b ↔ 
      x^2/3 + y^2/2 = 1)) ∧
  (∃ (k x_A y_A x_B y_B : ℝ), k ≠ 0 ∧
    line_l x_A y_A k ∧ line_l x_B y_B k ∧
    ellipse_C x_A y_A 3 2 ∧ ellipse_C x_B y_B 3 2 ∧
    complementary_angles x_A y_A x_B y_B 3) :=
by sorry

end hyperbola_ellipse_theorem_l3540_354031


namespace triangle_area_triangle_area_proof_l3540_354069

/-- The area of a triangle with base 10 and height 5 is 25 -/
theorem triangle_area : ℝ → ℝ → ℝ → Prop :=
  fun base height area =>
    base = 10 ∧ height = 5 → area = (base * height) / 2 → area = 25

#check triangle_area

theorem triangle_area_proof : triangle_area 10 5 25 := by
  sorry

end triangle_area_triangle_area_proof_l3540_354069


namespace min_value_xyz_l3540_354021

/-- Given positive real numbers x, y, and z satisfying 1/x + 1/y + 1/z = 9,
    the minimum value of x^2 * y^3 * z is 729/6912 -/
theorem min_value_xyz (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0)
  (h : 1/x + 1/y + 1/z = 9) :
  ∃ (m : ℝ), m = 729/6912 ∧ ∀ (a b c : ℝ), a > 0 → b > 0 → c > 0 →
    1/a + 1/b + 1/c = 9 → a^2 * b^3 * c ≥ m :=
by sorry

end min_value_xyz_l3540_354021


namespace set_equality_l3540_354080

-- Define the universal set U as ℝ
def U : Set ℝ := Set.univ

-- Define set E
def E : Set ℝ := {x : ℝ | x ≤ -3 ∨ x ≥ 2}

-- Define set F
def F : Set ℝ := {x : ℝ | -1 < x ∧ x < 5}

-- Theorem statement
theorem set_equality : {x : ℝ | -1 < x ∧ x < 2} = (Eᶜ ∩ F) := by sorry

end set_equality_l3540_354080


namespace cube_root_of_negative_27_l3540_354086

theorem cube_root_of_negative_27 : ∃ x : ℝ, x^3 = -27 ∧ x = -3 := by
  sorry

end cube_root_of_negative_27_l3540_354086


namespace pyramid_with_10_edges_has_6_vertices_l3540_354023

-- Define a pyramid structure
structure Pyramid where
  base_sides : ℕ
  edges : ℕ
  vertices : ℕ

-- Theorem statement
theorem pyramid_with_10_edges_has_6_vertices :
  ∀ p : Pyramid, p.edges = 10 → p.vertices = 6 := by
  sorry


end pyramid_with_10_edges_has_6_vertices_l3540_354023


namespace cube_root_equation_solution_l3540_354050

theorem cube_root_equation_solution :
  ∃! x : ℝ, (5 - x / 3) ^ (1/3 : ℝ) = -4 ∧ x = 207 :=
by sorry

end cube_root_equation_solution_l3540_354050


namespace no_prime_for_expression_l3540_354083

theorem no_prime_for_expression (m n : ℕ) : 
  ¬(Nat.Prime (n^2 + 2018*m*n + 2019*m + n - 2019*m^2)) := by
sorry

end no_prime_for_expression_l3540_354083


namespace complex_equation_result_l3540_354002

theorem complex_equation_result (a b : ℝ) (h : (1 + Complex.I) * (1 - b * Complex.I) = a) : a / b = 2 := by
  sorry

end complex_equation_result_l3540_354002


namespace nonzero_terms_count_l3540_354078

def expression (x : ℝ) : ℝ := (2*x + 5)*(3*x^2 + 4*x + 8) - 4*(x^3 - x^2 + 5*x + 2)

theorem nonzero_terms_count : 
  ∃ (a b c d : ℝ), ∀ x : ℝ, 
    expression x = a*x^3 + b*x^2 + c*x + d ∧ 
    a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0 ∧ d ≠ 0 :=
sorry

end nonzero_terms_count_l3540_354078


namespace stratified_by_stage_is_most_reasonable_l3540_354001

-- Define the possible sampling methods
inductive SamplingMethod
| SimpleRandom
| StratifiedByGender
| StratifiedByEducationalStage
| Systematic

-- Define the characteristics of the population
structure PopulationCharacteristics where
  significantDifferenceByStage : Bool
  significantDifferenceByGender : Bool

-- Define the function to determine the most reasonable sampling method
def mostReasonableSamplingMethod (pop : PopulationCharacteristics) : SamplingMethod :=
  sorry

-- Theorem statement
theorem stratified_by_stage_is_most_reasonable 
  (pop : PopulationCharacteristics) 
  (h1 : pop.significantDifferenceByStage = true) 
  (h2 : pop.significantDifferenceByGender = false) :
  mostReasonableSamplingMethod pop = SamplingMethod.StratifiedByEducationalStage :=
sorry

end stratified_by_stage_is_most_reasonable_l3540_354001


namespace infinitely_many_primes_4k_plus_1_any_4m_plus_1_has_prime_factor_4k_plus_1_infinitely_many_primes_4k_plus_1_from_divisibility_l3540_354090

theorem infinitely_many_primes_4k_plus_1 :
  ∀ (S : Set Nat), (∀ p ∈ S, Nat.Prime p ∧ ∃ k, p = 4*k + 1) →
  (∀ n, ∃ p ∈ S, p > n) :=
by
  sorry

theorem any_4m_plus_1_has_prime_factor_4k_plus_1 :
  ∀ m : Nat, ∃ p : Nat, Nat.Prime p ∧ (∃ k : Nat, p = 4*k + 1) ∧ p ∣ (4*m + 1) :=
by
  sorry

theorem infinitely_many_primes_4k_plus_1_from_divisibility 
  (h : ∀ m : Nat, ∃ p : Nat, Nat.Prime p ∧ (∃ k : Nat, p = 4*k + 1) ∧ p ∣ (4*m + 1)) :
  ∀ (S : Set Nat), (∀ p ∈ S, Nat.Prime p ∧ ∃ k, p = 4*k + 1) →
  (∀ n, ∃ p ∈ S, p > n) :=
by
  sorry

end infinitely_many_primes_4k_plus_1_any_4m_plus_1_has_prime_factor_4k_plus_1_infinitely_many_primes_4k_plus_1_from_divisibility_l3540_354090


namespace travel_time_proof_l3540_354008

/-- Given a person traveling at a constant speed, this theorem proves that
    the travel time is 5 hours when the distance is 500 km and the speed is 100 km/hr. -/
theorem travel_time_proof (distance : ℝ) (speed : ℝ) (time : ℝ) :
  distance = 500 ∧ speed = 100 ∧ time = distance / speed → time = 5 := by
  sorry

end travel_time_proof_l3540_354008


namespace pure_imaginary_complex_l3540_354020

theorem pure_imaginary_complex (a : ℝ) : 
  (a - (10 : ℂ) / (3 - I)).im ≠ 0 ∧ (a - (10 : ℂ) / (3 - I)).re = 0 → a = 3 := by
  sorry

end pure_imaginary_complex_l3540_354020


namespace intersection_line_of_circles_l3540_354027

/-- Given two circles in the plane, this theorem states that the line passing through
    their intersection points has a specific equation. -/
theorem intersection_line_of_circles (x y : ℝ) : 
  (x^2 + y^2 + 2*x + 3*y = 0) →
  (x^2 + y^2 - 4*x + 2*y + 1 = 0) →
  (6*x + y - 1 = 0) := by
sorry

end intersection_line_of_circles_l3540_354027


namespace solution_range_l3540_354076

theorem solution_range (a : ℝ) : 
  (∃ x : ℝ, x ≤ 1 ∧ 3^x = a^2 + 2*a) → 
  (a ∈ Set.Icc (-3) (-2) ∪ Set.Ioo 0 1) :=
sorry

end solution_range_l3540_354076


namespace sum_of_squares_130_l3540_354070

theorem sum_of_squares_130 : ∃ (a b : ℕ), 
  a ≠ b ∧ 
  a > 0 ∧ 
  b > 0 ∧ 
  a^2 + b^2 = 130 ∧ 
  a + b = 16 := by
sorry

end sum_of_squares_130_l3540_354070


namespace rectangle_area_problem_l3540_354006

/-- Given a rectangle with initial dimensions 4 × 6 inches, if shortening one side by 2 inches
    results in an area of 12 square inches, then shortening the other side by 1 inch
    results in an area of 20 square inches. -/
theorem rectangle_area_problem :
  ∀ (length width : ℝ),
  length = 4 ∧ width = 6 →
  (∃ (shortened_side : ℝ),
    (shortened_side = length - 2 ∨ shortened_side = width - 2) ∧
    shortened_side * (if shortened_side = length - 2 then width else length) = 12) →
  (if length - 2 < width - 2 then (length * (width - 1)) else ((length - 1) * width)) = 20 := by
sorry

end rectangle_area_problem_l3540_354006


namespace toms_barbados_trip_cost_l3540_354051

/-- The total cost for Tom's trip to Barbados -/
def total_cost (num_vaccines : ℕ) (vaccine_cost : ℚ) (doctor_visit_cost : ℚ) 
                (insurance_coverage : ℚ) (trip_cost : ℚ) : ℚ :=
  let medical_cost := num_vaccines * vaccine_cost + doctor_visit_cost
  let out_of_pocket_medical := medical_cost * (1 - insurance_coverage)
  out_of_pocket_medical + trip_cost

/-- Theorem stating the total cost for Tom's trip to Barbados -/
theorem toms_barbados_trip_cost :
  total_cost 10 45 250 0.8 1200 = 1340 := by
  sorry

end toms_barbados_trip_cost_l3540_354051


namespace muffin_division_l3540_354065

theorem muffin_division (total_muffins : ℕ) (friends : ℕ) (muffins_per_person : ℕ) :
  total_muffins = 20 →
  friends = 4 →
  muffins_per_person * (friends + 1) = total_muffins →
  muffins_per_person = 4 :=
by
  sorry


end muffin_division_l3540_354065


namespace negative_four_cubed_equality_l3540_354039

theorem negative_four_cubed_equality : (-4)^3 = -4^3 := by
  sorry

end negative_four_cubed_equality_l3540_354039


namespace chocolate_gain_percent_l3540_354022

/-- Calculates the gain percent given the number of chocolates at cost price and selling price that are equal in value. -/
def gain_percent (cost_count : ℕ) (sell_count : ℕ) : ℚ :=
  ((cost_count - sell_count) / sell_count) * 100

/-- Theorem stating that when the cost price of 65 chocolates equals the selling price of 50 chocolates, the gain percent is 30%. -/
theorem chocolate_gain_percent :
  gain_percent 65 50 = 30 := by
  sorry

#eval gain_percent 65 50

end chocolate_gain_percent_l3540_354022


namespace simplify_expression_l3540_354088

theorem simplify_expression (a : ℝ) : 2 * (a + 2) - 2 * a = 4 := by
  sorry

end simplify_expression_l3540_354088


namespace cubic_polynomial_sum_l3540_354082

/-- A cubic polynomial with real coefficients -/
structure CubicPolynomial where
  a : ℝ
  b : ℝ
  c : ℝ

/-- The roots of the polynomial -/
structure PolynomialRoots (w : ℂ) where
  root1 : ℂ := w - Complex.I
  root2 : ℂ := w - 3 * Complex.I
  root3 : ℂ := 2 * w + 2

/-- Theorem statement -/
theorem cubic_polynomial_sum (P : CubicPolynomial) (w : ℂ) 
  (h : ∀ z : ℂ, (z - (w - Complex.I)) * (z - (w - 3 * Complex.I)) * (z - (2 * w + 2)) = 
       z^3 + P.a * z^2 + P.b * z + P.c) :
  P.a + P.b + P.c = 22 := by
  sorry

end cubic_polynomial_sum_l3540_354082


namespace sector_arc_length_l3540_354004

theorem sector_arc_length (area : ℝ) (angle : ℝ) (arc_length : ℝ) : 
  area = 4 → angle = 2 → arc_length = 4 := by
  sorry

end sector_arc_length_l3540_354004


namespace batting_average_calculation_l3540_354057

/-- Calculates the new batting average after a match -/
def newBattingAverage (currentAverage : ℚ) (matchesPlayed : ℕ) (runsScored : ℕ) : ℚ :=
  (currentAverage * matchesPlayed + runsScored) / (matchesPlayed + 1)

/-- Theorem: Given the conditions, the new batting average will be 54 -/
theorem batting_average_calculation (currentAverage : ℚ) (matchesPlayed : ℕ) (runsScored : ℕ)
  (h1 : currentAverage = 51)
  (h2 : matchesPlayed = 5)
  (h3 : runsScored = 69) :
  newBattingAverage currentAverage matchesPlayed runsScored = 54 := by
  sorry

#eval newBattingAverage 51 5 69

end batting_average_calculation_l3540_354057


namespace remaining_drivable_distance_l3540_354092

/-- Proves the remaining drivable distance after a trip --/
theorem remaining_drivable_distance
  (fuel_efficiency : ℝ)
  (tank_capacity : ℝ)
  (trip_distance : ℝ)
  (h1 : fuel_efficiency = 20)
  (h2 : tank_capacity = 16)
  (h3 : trip_distance = 220) :
  fuel_efficiency * tank_capacity - trip_distance = 100 :=
by
  sorry

end remaining_drivable_distance_l3540_354092


namespace least_prime_factor_of_N_l3540_354042

def N : ℕ := 10^2011 + 1

theorem least_prime_factor_of_N :
  (Nat.minFac N = 11) := by sorry

end least_prime_factor_of_N_l3540_354042


namespace factorial_equation_l3540_354024

def factorial (n : ℕ) : ℕ := (List.range n).foldl (· * ·) 1

theorem factorial_equation (n : ℕ) : factorial 6 / factorial (6 - n) = 120 → n = 3 := by
  sorry

end factorial_equation_l3540_354024


namespace fraction_sum_l3540_354077

theorem fraction_sum (x y : ℝ) (h : y / x = 3 / 4) : (x + y) / x = 7 / 4 := by
  sorry

end fraction_sum_l3540_354077


namespace equation_solutions_l3540_354005

theorem equation_solutions : 
  {x : ℝ | 3 * x + 6 = |(-10 + 5 * x)|} = {8, (1/2 : ℝ)} := by sorry

end equation_solutions_l3540_354005


namespace smaller_type_pages_l3540_354079

theorem smaller_type_pages 
  (total_words : ℕ) 
  (larger_type_words_per_page : ℕ) 
  (smaller_type_words_per_page : ℕ) 
  (total_pages : ℕ) 
  (h1 : total_words = 48000)
  (h2 : larger_type_words_per_page = 1800)
  (h3 : smaller_type_words_per_page = 2400)
  (h4 : total_pages = 21) :
  ∃ (x y : ℕ), 
    x + y = total_pages ∧ 
    larger_type_words_per_page * x + smaller_type_words_per_page * y = total_words ∧
    y = 17 := by
  sorry

end smaller_type_pages_l3540_354079


namespace possible_distances_l3540_354054

theorem possible_distances (p q r s t : ℝ) 
  (h1 : |p - q| = 3)
  (h2 : |q - r| = 4)
  (h3 : |r - s| = 5)
  (h4 : |s - t| = 6) :
  ∃ (S : Set ℝ), S = {0, 2, 4, 6, 8, 10, 12, 18} ∧ |p - t| ∈ S :=
by sorry

end possible_distances_l3540_354054


namespace sqrt_50_between_consecutive_integers_l3540_354037

theorem sqrt_50_between_consecutive_integers : ∃ (n : ℕ), n > 0 ∧ n^2 < 50 ∧ (n+1)^2 > 50 ∧ n * (n+1) = 56 := by
  sorry

end sqrt_50_between_consecutive_integers_l3540_354037


namespace janet_lives_gained_l3540_354038

theorem janet_lives_gained (initial_lives : ℕ) (lives_lost : ℕ) (final_lives : ℕ) 
  (h1 : initial_lives = 47)
  (h2 : lives_lost = 23)
  (h3 : final_lives = 70) :
  final_lives - (initial_lives - lives_lost) = 46 := by
  sorry

end janet_lives_gained_l3540_354038


namespace functional_equation_solutions_l3540_354011

open Real

-- Define the type of continuous functions from ℝ⁺ to ℝ⁺
def ContinuousPosFun := {f : ℝ → ℝ // Continuous f ∧ ∀ x, x > 0 → f x > 0}

-- Define the property that the function satisfies the given equation
def SatisfiesEquation (f : ContinuousPosFun) : Prop :=
  ∀ x, x > 0 → x + 1/x = f.val x + 1/(f.val x)

-- Define the set of possible solutions
def PossibleSolutions (x : ℝ) : Set ℝ :=
  {x, 1/x, max x (1/x), min x (1/x)}

-- State the theorem
theorem functional_equation_solutions (f : ContinuousPosFun) 
  (h : SatisfiesEquation f) :
  ∀ x, x > 0 → f.val x ∈ PossibleSolutions x := by
  sorry

end functional_equation_solutions_l3540_354011


namespace exists_same_color_unit_apart_l3540_354034

/-- A coloring of the plane using three colors -/
def Coloring := ℝ × ℝ → Fin 3

/-- Two points are one unit apart -/
def one_unit_apart (p q : ℝ × ℝ) : Prop :=
  (p.1 - q.1)^2 + (p.2 - q.2)^2 = 1

/-- Main theorem: In any three-coloring of the plane, there exist two points of the same color that are exactly one unit apart -/
theorem exists_same_color_unit_apart (c : Coloring) : 
  ∃ (p q : ℝ × ℝ), c p = c q ∧ one_unit_apart p q := by
  sorry

end exists_same_color_unit_apart_l3540_354034


namespace chinese_remainder_theorem_example_l3540_354014

theorem chinese_remainder_theorem_example : ∃ n : ℕ, 
  n > 0 ∧ 
  n % 3 = 2 ∧ 
  n % 5 = 3 ∧ 
  n % 7 = 2 ∧ 
  ∀ m : ℕ, m > 0 → m % 3 = 2 → m % 5 = 3 → m % 7 = 2 → m ≥ n :=
by sorry

end chinese_remainder_theorem_example_l3540_354014


namespace probability_of_selection_l3540_354096

/-- Given a group of students where each student has an equal chance of being selected as the group leader,
    prove that the probability of a specific student (Xiao Li) being chosen is 1/5. -/
theorem probability_of_selection (total_students : ℕ) (xiao_li : Fin total_students) :
  total_students = 5 →
  (∀ (student : Fin total_students), ℚ) →
  (∃! (prob : Fin total_students → ℚ), ∀ (student : Fin total_students), prob student = 1 / total_students) →
  (∃ (prob : Fin total_students → ℚ), prob xiao_li = 1 / 5) :=
by sorry

end probability_of_selection_l3540_354096


namespace conference_handshakes_l3540_354047

/-- Represents a group of employees at a conference -/
structure EmployeeGroup where
  size : Nat
  has_closed_loop : Bool

/-- Calculates the number of handshakes in the employee group -/
def count_handshakes (group : EmployeeGroup) : Nat :=
  if group.has_closed_loop && group.size ≥ 3 then
    (group.size * (group.size - 3)) / 2
  else
    0

/-- Theorem: In a group of 10 employees with a closed managerial loop,
    where each person shakes hands with everyone except their direct manager
    and direct subordinate, the total number of handshakes is 35 -/
theorem conference_handshakes :
  let group : EmployeeGroup := { size := 10, has_closed_loop := true }
  count_handshakes group = 35 := by
  sorry

end conference_handshakes_l3540_354047


namespace sum_is_composite_l3540_354028

theorem sum_is_composite (a b c d : ℕ+) (h : a^2 + b^2 = c^2 + d^2) :
  ∃ (k m : ℕ), k > 1 ∧ m > 1 ∧ (a : ℕ) + b + c + d = k * m := by
  sorry

end sum_is_composite_l3540_354028


namespace H_surjective_l3540_354058

def H (x : ℝ) : ℝ := |x^2 + 2*x + 1| - |x^2 - 2*x + 1|

theorem H_surjective : Function.Surjective H := by sorry

end H_surjective_l3540_354058


namespace m_subset_p_subset_n_l3540_354012

/-- Set M definition -/
def M : Set (ℝ × ℝ) := {p | abs p.1 + abs p.2 < 1}

/-- Set N definition -/
def N : Set (ℝ × ℝ) := {p | Real.sqrt ((p.1 - 1/2)^2 + (p.2 + 1/2)^2) + Real.sqrt ((p.1 + 1/2)^2 + (p.2 - 1/2)^2) < 2 * Real.sqrt 2}

/-- Set P definition -/
def P : Set (ℝ × ℝ) := {p | abs (p.1 + p.2) < 1 ∧ abs p.1 < 1 ∧ abs p.2 < 1}

/-- Theorem stating M ⊂ P ⊂ N -/
theorem m_subset_p_subset_n : M ⊆ P ∧ P ⊆ N := by sorry

end m_subset_p_subset_n_l3540_354012


namespace correct_restroom_count_l3540_354098

/-- The number of students in the restroom -/
def students_in_restroom : ℕ := 2

/-- The number of absent students -/
def absent_students : ℕ := 3 * students_in_restroom - 1

/-- The total number of desks -/
def total_desks : ℕ := 4 * 6

/-- The number of occupied desks -/
def occupied_desks : ℕ := (2 * total_desks) / 3

/-- The total number of students Carla teaches -/
def total_students : ℕ := 23

theorem correct_restroom_count :
  students_in_restroom + absent_students + occupied_desks = total_students :=
sorry

end correct_restroom_count_l3540_354098


namespace problem_solution_l3540_354064

-- Define the function f(x)
def f (x : ℝ) : ℝ := |2*x - 9| - |x - 5|

-- Define the function y(x)
def y (x : ℝ) : ℝ := f x + 3*|x - 5|

theorem problem_solution :
  -- Part 1: Solution set of f(x) ≥ 2x-1
  (∀ x : ℝ, f x ≥ 2*x - 1 ↔ x ≤ 5/3) ∧
  -- Part 2: Minimum value of y(x)
  (∀ x : ℝ, y x ≥ 1) ∧
  (∃ x : ℝ, y x = 1) ∧
  -- Part 3: Minimum value of a + 3b given 1/a + 3/b = 1
  (∀ a b : ℝ, a > 0 → b > 0 → 1/a + 3/b = 1 → a + 3*b ≥ 16) ∧
  (∃ a b : ℝ, a > 0 ∧ b > 0 ∧ 1/a + 3/b = 1 ∧ a + 3*b = 16) :=
by sorry

end problem_solution_l3540_354064


namespace repeating_decimal_fractions_l3540_354045

def repeating_decimal_3 : ℚ := 0.333333
def repeating_decimal_56 : ℚ := 0.565656

theorem repeating_decimal_fractions :
  (repeating_decimal_3 = 1 / 3) ∧
  (repeating_decimal_56 = 56 / 99) := by
  sorry

end repeating_decimal_fractions_l3540_354045


namespace judy_shopping_cost_l3540_354061

-- Define the prices and quantities
def carrot_price : ℝ := 1.50
def carrot_quantity : ℕ := 6
def milk_price : ℝ := 3.50
def milk_quantity : ℕ := 4
def pineapple_price : ℝ := 5.00
def pineapple_quantity : ℕ := 3
def pineapple_discount : ℝ := 0.25
def flour_price : ℝ := 6.00
def flour_quantity : ℕ := 3
def flour_discount : ℝ := 0.10
def ice_cream_price : ℝ := 8.00
def coupon_value : ℝ := 10.00
def coupon_threshold : ℝ := 50.00

-- Define the theorem
theorem judy_shopping_cost :
  let carrot_total := carrot_price * carrot_quantity
  let milk_total := milk_price * milk_quantity
  let pineapple_total := pineapple_price * (1 - pineapple_discount) * pineapple_quantity
  let flour_total := flour_price * (1 - flour_discount) * flour_quantity
  let subtotal := carrot_total + milk_total + pineapple_total + flour_total + ice_cream_price
  let final_total := if subtotal ≥ coupon_threshold then subtotal - coupon_value else subtotal
  final_total = 48.45 := by sorry

end judy_shopping_cost_l3540_354061


namespace olympic_mascot_problem_l3540_354091

theorem olympic_mascot_problem (total_items wholesale_cost : ℕ) 
  (wholesale_price_A wholesale_price_B : ℕ) 
  (retail_price_A retail_price_B : ℕ) (min_profit : ℕ) :
  total_items = 100 ∧ 
  wholesale_cost = 5650 ∧
  wholesale_price_A = 60 ∧ 
  wholesale_price_B = 50 ∧
  retail_price_A = 80 ∧
  retail_price_B = 60 ∧
  min_profit = 1400 →
  (∃ (num_A num_B : ℕ),
    num_A + num_B = total_items ∧
    num_A * wholesale_price_A + num_B * wholesale_price_B = wholesale_cost ∧
    num_A = 65 ∧ num_B = 35) ∧
  (∃ (min_A : ℕ),
    min_A ≥ 40 ∧
    ∀ (num_A : ℕ),
      num_A ≥ min_A →
      (num_A * (retail_price_A - wholesale_price_A) + 
       (total_items - num_A) * (retail_price_B - wholesale_price_B)) ≥ min_profit) :=
by sorry

end olympic_mascot_problem_l3540_354091


namespace worker_a_time_l3540_354059

theorem worker_a_time (b_time : ℝ) (combined_time : ℝ) (a_time : ℝ) : 
  b_time = 10 →
  combined_time = 4.444444444444445 →
  (1 / a_time + 1 / b_time = 1 / combined_time) →
  a_time = 8 := by
    sorry

end worker_a_time_l3540_354059


namespace intersect_x_axis_and_derivative_negative_l3540_354016

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.exp x - a * x + a

theorem intersect_x_axis_and_derivative_negative (a : ℝ) (x₁ x₂ : ℝ) :
  a > Real.exp 2 →
  x₁ < x₂ →
  f a x₁ = 0 →
  f a x₂ = 0 →
  let x₀ := Real.sqrt (x₁ * x₂)
  (deriv (f a)) x₀ < 0 :=
by sorry

end intersect_x_axis_and_derivative_negative_l3540_354016


namespace add_preserves_inequality_l3540_354081

theorem add_preserves_inequality (a b : ℝ) (h : a > b) : a + 2 > b + 2 := by
  sorry

end add_preserves_inequality_l3540_354081


namespace writing_utensils_arrangement_l3540_354018

def factorial (n : ℕ) : ℕ := (List.range n).foldl (·*·) 1

def circular_permutations (n a b c d : ℕ) : ℕ :=
  factorial (n - 1) / (factorial a * factorial b * factorial c * factorial d)

def adjacent_arrangements (n a b c d : ℕ) : ℕ :=
  circular_permutations (n - 1) a 1 c d

theorem writing_utensils_arrangement :
  let total_items : ℕ := 5 + 3 + 1 + 1
  let black_pencils : ℕ := 5
  let blue_pens : ℕ := 3
  let red_pen : ℕ := 1
  let green_pen : ℕ := 1
  circular_permutations total_items black_pencils blue_pens red_pen green_pen -
  adjacent_arrangements total_items black_pencils blue_pens red_pen green_pen = 168 := by
sorry

end writing_utensils_arrangement_l3540_354018


namespace range_of_x_range_of_a_l3540_354072

-- Define the conditions
def p (x : ℝ) := x^2 - x - 2 ≤ 0
def q (x : ℝ) := (x - 3) / x < 0
def r (x a : ℝ) := (x - (a + 1)) * (x + (2 * a - 1)) ≤ 0

-- Question 1
theorem range_of_x (x : ℝ) (h1 : p x) (h2 : q x) : x ∈ Set.Ioc 0 2 := by sorry

-- Question 2
theorem range_of_a (a : ℝ) 
  (h1 : ∀ x, p x → r x a) 
  (h2 : ∃ x, r x a ∧ ¬p x) 
  (h3 : a > 0) : 
  a > 1 := by sorry

end range_of_x_range_of_a_l3540_354072


namespace factorization_a4_plus_4_l3540_354025

theorem factorization_a4_plus_4 (a : ℝ) : a^4 + 4 = (a^2 + 2*a + 2)*(a^2 - 2*a + 2) := by
  sorry

end factorization_a4_plus_4_l3540_354025


namespace elena_and_alex_money_l3540_354040

theorem elena_and_alex_money : (5 : ℚ) / 6 + (7 : ℚ) / 15 = (13 : ℚ) / 10 := by
  sorry

end elena_and_alex_money_l3540_354040


namespace wood_length_l3540_354010

/-- The original length of a piece of wood, given the length sawed off and the remaining length -/
theorem wood_length (sawed_off : ℝ) (remaining : ℝ) (h1 : sawed_off = 2.3) (h2 : remaining = 6.6) :
  sawed_off + remaining = 8.9 := by
  sorry

end wood_length_l3540_354010


namespace ellipse_k_range_l3540_354089

/-- An ellipse with equation x^2 / (2-k) + y^2 / (2k-1) = 1 and foci on the y-axis has k in the range (1, 2) -/
theorem ellipse_k_range (k : ℝ) :
  (∀ x y : ℝ, x^2 / (2-k) + y^2 / (2*k-1) = 1) →  -- equation represents an ellipse
  (∃ c : ℝ, c > 0 ∧ ∀ x y : ℝ, x^2 / (2-k) + y^2 / (2*k-1) = 1 → y^2 ≥ c^2) →  -- foci on y-axis
  1 < k ∧ k < 2 :=
by sorry

end ellipse_k_range_l3540_354089


namespace youtube_video_length_l3540_354053

/-- Represents the duration of a YouTube video session in seconds -/
def YouTubeSession (ad1 ad2 video1 video2 pause totalTime : ℕ) (lastTwoEqual : Bool) : Prop :=
  let firstVideoTotal := ad1 + 120  -- 2 minutes = 120 seconds
  let secondVideoTotal := ad2 + 270  -- 4 minutes 30 seconds = 270 seconds
  let remainingTime := totalTime - (firstVideoTotal + secondVideoTotal)
  let lastTwoVideosTime := remainingTime - pause
  lastTwoEqual ∧ 
  (lastTwoVideosTime / 2 = 495) ∧
  (totalTime = 1500)

theorem youtube_video_length 
  (ad1 ad2 video1 video2 pause totalTime : ℕ) 
  (lastTwoEqual : Bool) 
  (h : YouTubeSession ad1 ad2 video1 video2 pause totalTime lastTwoEqual) :
  ∃ (lastVideoLength : ℕ), lastVideoLength = 495 :=
sorry

end youtube_video_length_l3540_354053


namespace q_investment_time_l3540_354007

/-- Represents a partner in the investment problem -/
structure Partner where
  investment : ℝ
  time : ℝ
  profit : ℝ

/-- The investment problem setup -/
def InvestmentProblem (p q : Partner) : Prop :=
  p.investment / q.investment = 7 / 5 ∧
  p.profit / q.profit = 7 / 9 ∧
  p.time = 5 ∧
  p.investment * p.time / (q.investment * q.time) = p.profit / q.profit

/-- Theorem stating that Q's investment time is 9 months -/
theorem q_investment_time (p q : Partner) 
  (h : InvestmentProblem p q) : q.time = 9 := by
  sorry

end q_investment_time_l3540_354007


namespace unique_solution_l3540_354066

theorem unique_solution : ∃! x : ℝ, x * 3 + 3 * 13 + 3 * 16 + 11 = 134 := by
  sorry

end unique_solution_l3540_354066


namespace triangle_inradius_inequality_l3540_354097

-- Define a triangle with sides a, b, c and inradius r
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  r : ℝ
  -- Ensure that a, b, c form a valid triangle
  triangle_inequality : a > 0 ∧ b > 0 ∧ c > 0 ∧ a + b > c ∧ b + c > a ∧ c + a > b
  -- Ensure that r is positive
  positive_inradius : r > 0

-- State the theorem
theorem triangle_inradius_inequality (t : Triangle) :
  1 / t.a^2 + 1 / t.b^2 + 1 / t.c^2 ≤ 1 / (4 * t.r^2) := by
  sorry

end triangle_inradius_inequality_l3540_354097


namespace triangle_angle_measure_l3540_354074

theorem triangle_angle_measure (D E F : ℝ) : 
  D = 75 → 
  E = 4 * F + 15 → 
  D + E + F = 180 → 
  F = 18 := by sorry

end triangle_angle_measure_l3540_354074


namespace hexagram_arrangements_l3540_354063

/-- A regular six-pointed star -/
structure HexagramStar :=
  (points : Fin 6 → Type)

/-- The group of symmetries of a regular six-pointed star -/
def hexagramSymmetries : ℕ := 12

/-- The number of ways to arrange 6 distinct objects -/
def totalArrangements : ℕ := 720

/-- The number of distinct arrangements of 6 objects on a hexagram star -/
def distinctArrangements (star : HexagramStar) : ℕ :=
  totalArrangements / hexagramSymmetries

theorem hexagram_arrangements (star : HexagramStar) :
  distinctArrangements star = 60 := by
  sorry

end hexagram_arrangements_l3540_354063


namespace smallest_average_of_valid_pair_l3540_354015

/-- Sum of digits of a natural number -/
def sumOfDigits (n : ℕ) : ℕ := sorry

/-- Predicate for two numbers differing by 2 and having sum of digits divisible by 4 -/
def validPair (n m : ℕ) : Prop :=
  m = n + 2 ∧ (sumOfDigits n + sumOfDigits m) % 4 = 0

theorem smallest_average_of_valid_pair :
  ∃ (n m : ℕ), validPair n m ∧
  ∀ (k l : ℕ), validPair k l → (n + m : ℚ) / 2 ≤ (k + l : ℚ) / 2 :=
by sorry

end smallest_average_of_valid_pair_l3540_354015


namespace sphere_volume_ratio_l3540_354095

theorem sphere_volume_ratio (S₁ S₂ S₃ V₁ V₂ V₃ : ℝ) :
  S₁ > 0 ∧ S₂ > 0 ∧ S₃ > 0 →
  V₁ > 0 ∧ V₂ > 0 ∧ V₃ > 0 →
  S₂ / S₁ = 4 →
  S₃ / S₁ = 9 →
  (4 * π * (V₁ / (4/3 * π))^(2/3) = S₁) →
  (4 * π * (V₂ / (4/3 * π))^(2/3) = S₂) →
  (4 * π * (V₃ / (4/3 * π))^(2/3) = S₃) →
  V₁ + V₂ = (1/3) * V₃ := by
sorry

end sphere_volume_ratio_l3540_354095


namespace power_of_two_equation_l3540_354067

theorem power_of_two_equation (y : ℤ) : (1 / 8 : ℚ) * 2^36 = 2^y → y = 33 := by
  sorry

end power_of_two_equation_l3540_354067


namespace weight_sum_l3540_354036

/-- Given the weights of four people (a, b, c, d) in pairs,
    prove that the sum of the weights of the first and last person is 310 pounds. -/
theorem weight_sum (a b c d : ℝ) 
  (h1 : a + b = 280) 
  (h2 : b + c = 230) 
  (h3 : c + d = 260) : 
  a + d = 310 := by
  sorry

end weight_sum_l3540_354036
