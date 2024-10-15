import Mathlib

namespace NUMINAMATH_CALUDE_algebraic_expression_value_l3845_384577

theorem algebraic_expression_value (a : ℝ) : 
  (2023 - a)^2 + (a - 2022)^2 = 7 → (2023 - a) * (a - 2022) = -3 := by
sorry

end NUMINAMATH_CALUDE_algebraic_expression_value_l3845_384577


namespace NUMINAMATH_CALUDE_two_roots_implication_l3845_384591

/-- If a quadratic trinomial ax^2 + bx + c has two roots, 
    then the trinomial 3ax^2 + 2(a + b)x + (b + c) also has two roots. -/
theorem two_roots_implication (a b c : ℝ) : 
  (∃ x y : ℝ, x ≠ y ∧ a * x^2 + b * x + c = 0 ∧ a * y^2 + b * y + c = 0) →
  (∃ u v : ℝ, u ≠ v ∧ 3 * a * u^2 + 2 * (a + b) * u + (b + c) = 0 ∧ 
                    3 * a * v^2 + 2 * (a + b) * v + (b + c) = 0) :=
by sorry

end NUMINAMATH_CALUDE_two_roots_implication_l3845_384591


namespace NUMINAMATH_CALUDE_sum_of_squares_l3845_384570

theorem sum_of_squares (a b c : ℝ) : 
  (a * b + b * c + a * c = 131) → (a + b + c = 18) → (a^2 + b^2 + c^2 = 62) := by
  sorry

end NUMINAMATH_CALUDE_sum_of_squares_l3845_384570


namespace NUMINAMATH_CALUDE_arithmetic_sequence_proof_l3845_384555

def arithmetic_sequence (a : ℕ → ℝ) :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_proof (a : ℕ → ℝ) 
  (h1 : arithmetic_sequence a)
  (h2 : a 5 = 10)
  (h3 : a 1 + a 2 + a 3 = 3) :
  a 1 = -2 ∧ ∃ d : ℝ, d = 3 ∧ ∀ n : ℕ, a (n + 1) = a n + d :=
by sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_proof_l3845_384555


namespace NUMINAMATH_CALUDE_trig_expression_equals_neg_sqrt_three_l3845_384504

theorem trig_expression_equals_neg_sqrt_three : 
  (2 * Real.sin (10 * π / 180) - Real.cos (20 * π / 180)) / Real.cos (70 * π / 180) = -Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_trig_expression_equals_neg_sqrt_three_l3845_384504


namespace NUMINAMATH_CALUDE_orangeade_pricing_l3845_384536

/-- Orangeade pricing problem -/
theorem orangeade_pricing
  (orange_juice : ℝ) -- Amount of orange juice (same for both days)
  (water_day1 : ℝ) -- Amount of water on day 1
  (price_day1 : ℝ) -- Price per glass on day 1
  (h1 : water_day1 = orange_juice) -- Equal amounts of orange juice and water on day 1
  (h2 : price_day1 = 0.60) -- Price per glass on day 1 is $0.60
  : -- Price per glass on day 2
    (price_day1 * (orange_juice + water_day1)) / (orange_juice + 2 * water_day1) = 0.40 := by
  sorry

end NUMINAMATH_CALUDE_orangeade_pricing_l3845_384536


namespace NUMINAMATH_CALUDE_probability_of_winning_pair_l3845_384592

def deck_size : ℕ := 10
def red_cards : ℕ := 5
def green_cards : ℕ := 5
def num_letters : ℕ := 5

def winning_pair_count : ℕ := num_letters + 2 * (red_cards.choose 2)

theorem probability_of_winning_pair :
  (winning_pair_count : ℚ) / (deck_size.choose 2) = 5 / 9 := by sorry

end NUMINAMATH_CALUDE_probability_of_winning_pair_l3845_384592


namespace NUMINAMATH_CALUDE_speed_ratio_is_four_fifths_l3845_384520

/-- Two objects A and B moving uniformly along perpendicular paths -/
structure PerpendicularMotion where
  vA : ℝ  -- Speed of object A
  vB : ℝ  -- Speed of object B
  d  : ℝ  -- Initial distance of B from O

/-- Equidistance condition at time t -/
def equidistant (m : PerpendicularMotion) (t : ℝ) : Prop :=
  m.vA * t = |m.d - m.vB * t|

/-- The theorem stating the ratio of speeds given the conditions -/
theorem speed_ratio_is_four_fifths (m : PerpendicularMotion) :
  m.d = 600 ∧ equidistant m 3 ∧ equidistant m 12 → m.vA / m.vB = 4/5 := by
  sorry

#check speed_ratio_is_four_fifths

end NUMINAMATH_CALUDE_speed_ratio_is_four_fifths_l3845_384520


namespace NUMINAMATH_CALUDE_prime_sum_24_l3845_384519

theorem prime_sum_24 (a b c : ℕ) : 
  Nat.Prime a ∧ Nat.Prime b ∧ Nat.Prime c → a * b + b * c = 119 → a + b + c = 24 := by
sorry

end NUMINAMATH_CALUDE_prime_sum_24_l3845_384519


namespace NUMINAMATH_CALUDE_road_signs_total_l3845_384518

/-- The total number of road signs at six intersections -/
def total_road_signs (first second third fourth fifth sixth : ℕ) : ℕ :=
  first + second + third + fourth + fifth + sixth

/-- Theorem stating the total number of road signs at six intersections -/
theorem road_signs_total : ∃ (first second third fourth fifth sixth : ℕ),
  (first = 50) ∧
  (second = first + first / 5) ∧
  (third = 2 * second - 10) ∧
  (fourth = ((first + second) + 1) / 2) ∧
  (fifth = third - second) ∧
  (sixth = first + fourth - 15) ∧
  (total_road_signs first second third fourth fifth sixth = 415) :=
by sorry

end NUMINAMATH_CALUDE_road_signs_total_l3845_384518


namespace NUMINAMATH_CALUDE_line_l_equation_l3845_384505

/-- The fixed point A through which the line mx - y - m + 2 = 0 always passes -/
def A : ℝ × ℝ := (1, 2)

/-- The slope of the line 2x + y - 2 = 0 -/
def k : ℝ := -2

/-- The equation of the line l passing through A and parallel to 2x + y - 2 = 0 -/
def line_l (x y : ℝ) : Prop := 2 * x + y - 4 = 0

theorem line_l_equation : ∀ m : ℝ, 
  (m * A.1 - A.2 - m + 2 = 0) → 
  (∀ x y : ℝ, line_l x y ↔ y - A.2 = k * (x - A.1)) :=
by sorry

end NUMINAMATH_CALUDE_line_l_equation_l3845_384505


namespace NUMINAMATH_CALUDE_fred_car_wash_earnings_l3845_384590

/-- The amount Fred earned by washing cars -/
def fred_earnings (initial_amount final_amount : ℕ) : ℕ :=
  final_amount - initial_amount

/-- Proof that Fred earned $4 by washing cars -/
theorem fred_car_wash_earnings : 
  fred_earnings 111 115 = 4 := by sorry

end NUMINAMATH_CALUDE_fred_car_wash_earnings_l3845_384590


namespace NUMINAMATH_CALUDE_original_ratio_of_boarders_to_day_students_l3845_384509

theorem original_ratio_of_boarders_to_day_students 
  (initial_boarders : ℕ) 
  (new_boarders : ℕ) 
  (final_ratio_boarders : ℕ) 
  (final_ratio_day_students : ℕ) : 
  initial_boarders = 150 →
  new_boarders = 30 →
  final_ratio_boarders = 1 →
  final_ratio_day_students = 2 →
  ∃ (original_ratio_boarders original_ratio_day_students : ℕ),
    original_ratio_boarders = 5 ∧ 
    original_ratio_day_students = 12 ∧
    (initial_boarders : ℚ) / (initial_boarders + new_boarders : ℚ) * final_ratio_day_students = 
      (original_ratio_boarders : ℚ) / (original_ratio_boarders + original_ratio_day_students : ℚ) :=
by sorry


end NUMINAMATH_CALUDE_original_ratio_of_boarders_to_day_students_l3845_384509


namespace NUMINAMATH_CALUDE_certain_number_proof_l3845_384598

theorem certain_number_proof (x : ℝ) : 0.60 * x = (4 / 5) * 25 + 4 → x = 40 := by
  sorry

end NUMINAMATH_CALUDE_certain_number_proof_l3845_384598


namespace NUMINAMATH_CALUDE_circle_line_intersection_properties_l3845_384569

/-- Given a circle and a line in 2D space, prove properties about their intersection and a related circle. -/
theorem circle_line_intersection_properties 
  (x y : ℝ) (m : ℝ) 
  (h_circle : x^2 + y^2 - 2*x - 4*y + m = 0) 
  (h_line : x + 2*y = 4) 
  (h_perpendicular : ∃ (x₁ y₁ x₂ y₂ : ℝ), 
    x₁^2 + y₁^2 - 2*x₁ - 4*y₁ + m = 0 ∧ 
    x₁ + 2*y₁ = 4 ∧
    x₂^2 + y₂^2 - 2*x₂ - 4*y₂ + m = 0 ∧ 
    x₂ + 2*y₂ = 4 ∧
    x₁*x₂ + y₁*y₂ = 0) :
  m = 8/5 ∧ 
  ∀ (x y : ℝ), x^2 + y^2 - (8/5)*x - (16/5)*y = 0 ↔ 
    ∃ (t : ℝ), 0 ≤ t ∧ t ≤ 1 ∧ 
      x = (1-t)*x₁ + t*x₂ ∧ 
      y = (1-t)*y₁ + t*y₂ :=
by sorry


end NUMINAMATH_CALUDE_circle_line_intersection_properties_l3845_384569


namespace NUMINAMATH_CALUDE_arithmetic_sequence_geometric_mean_l3845_384583

/-- 
Given an arithmetic sequence {a_n} with non-zero common difference d, 
where a_1 = 2d, if a_k is the geometric mean of a_1 and a_{2k+1}, then k = 3.
-/
theorem arithmetic_sequence_geometric_mean (d : ℝ) (k : ℕ) (a : ℕ → ℝ) :
  d ≠ 0 →
  (∀ n, a (n + 1) - a n = d) →
  a 1 = 2 * d →
  a k ^ 2 = a 1 * a (2 * k + 1) →
  k = 3 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_geometric_mean_l3845_384583


namespace NUMINAMATH_CALUDE_max_sum_given_sum_of_squares_and_product_l3845_384524

theorem max_sum_given_sum_of_squares_and_product (x y : ℝ) :
  x^2 + y^2 = 130 → xy = 18 → x + y ≤ Real.sqrt 166 := by
  sorry

end NUMINAMATH_CALUDE_max_sum_given_sum_of_squares_and_product_l3845_384524


namespace NUMINAMATH_CALUDE_no_integer_solution_l3845_384585

theorem no_integer_solution : ¬∃ (x y : ℤ), x * y + 4 = 40 ∧ x + y = 14 := by sorry

end NUMINAMATH_CALUDE_no_integer_solution_l3845_384585


namespace NUMINAMATH_CALUDE_complex_fraction_simplification_l3845_384589

theorem complex_fraction_simplification :
  let i : ℂ := Complex.I
  (i^3) / (1 - i) = (1 / 2 : ℂ) - (1 / 2 : ℂ) * i :=
by sorry

end NUMINAMATH_CALUDE_complex_fraction_simplification_l3845_384589


namespace NUMINAMATH_CALUDE_problem_solution_l3845_384508

theorem problem_solution (x y : ℝ) :
  (4 * x + y = 1) →
  (y = 1 - 4 * x) ∧
  (y ≥ 0 → x ≤ 1/4) ∧
  (-1 < y ∧ y ≤ 2 → -1/4 ≤ x ∧ x < 1/2) :=
by sorry

end NUMINAMATH_CALUDE_problem_solution_l3845_384508


namespace NUMINAMATH_CALUDE_not_sufficient_nor_necessary_l3845_384573

/-- The set A defined by a quadratic inequality -/
def A (a₁ b₁ c₁ : ℝ) : Set ℝ := {x | a₁ * x^2 + b₁ * x + c₁ > 0}

/-- The set B defined by a quadratic inequality -/
def B (a₂ b₂ c₂ : ℝ) : Set ℝ := {x | a₂ * x^2 + b₂ * x + c₂ > 0}

/-- The condition for coefficient ratios -/
def ratio_condition (a₁ b₁ c₁ a₂ b₂ c₂ : ℝ) : Prop :=
  a₁ / a₂ = b₁ / b₂ ∧ b₁ / b₂ = c₁ / c₂

theorem not_sufficient_nor_necessary
  (a₁ b₁ c₁ a₂ b₂ c₂ : ℝ) (h₁ : a₁ * b₁ * c₁ ≠ 0) (h₂ : a₂ * b₂ * c₂ ≠ 0) :
  ¬(∀ a₁ b₁ c₁ a₂ b₂ c₂ : ℝ, ratio_condition a₁ b₁ c₁ a₂ b₂ c₂ → A a₁ b₁ c₁ = B a₂ b₂ c₂) ∧
  ¬(∀ a₁ b₁ c₁ a₂ b₂ c₂ : ℝ, A a₁ b₁ c₁ = B a₂ b₂ c₂ → ratio_condition a₁ b₁ c₁ a₂ b₂ c₂) :=
sorry

end NUMINAMATH_CALUDE_not_sufficient_nor_necessary_l3845_384573


namespace NUMINAMATH_CALUDE_grazing_area_difference_l3845_384516

/-- Proves that the area difference between two circular grazing arrangements is 35π square feet -/
theorem grazing_area_difference (rope_length : ℝ) (tank_radius : ℝ) : 
  rope_length = 12 → tank_radius = 10 → 
  π * rope_length^2 - (3/4 * π * rope_length^2 + 1/4 * π * (rope_length - tank_radius)^2) = 35 * π := by
  sorry

end NUMINAMATH_CALUDE_grazing_area_difference_l3845_384516


namespace NUMINAMATH_CALUDE_binomial_expansion_coefficient_l3845_384531

theorem binomial_expansion_coefficient (a : ℝ) : 
  (∃ k : ℝ, k = (Nat.choose 6 3) * a^3 ∧ k = -160) → a = -2 := by
  sorry

end NUMINAMATH_CALUDE_binomial_expansion_coefficient_l3845_384531


namespace NUMINAMATH_CALUDE_tropical_fish_count_l3845_384552

theorem tropical_fish_count (total : ℕ) (koi : ℕ) (h1 : total = 52) (h2 : koi = 37) :
  total - koi = 15 := by
  sorry

end NUMINAMATH_CALUDE_tropical_fish_count_l3845_384552


namespace NUMINAMATH_CALUDE_unique_solution_factorial_equation_l3845_384568

def factorial (n : ℕ) : ℕ := 
  match n with
  | 0 => 1
  | n+1 => (n+1) * factorial n

theorem unique_solution_factorial_equation : 
  ∃! n : ℕ, n * factorial n + 2 * factorial n = 5040 ∧ n = 5 := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_factorial_equation_l3845_384568


namespace NUMINAMATH_CALUDE_symmetric_points_range_l3845_384586

-- Define the functions f and g
def f (a x : ℝ) : ℝ := a - x^2
def g (x : ℝ) : ℝ := x + 1

-- Define the theorem
theorem symmetric_points_range (a : ℝ) :
  (∃ x : ℝ, 1 ≤ x ∧ x ≤ 2 ∧ ∃ y : ℝ, f a x = -g y) →
  -1 ≤ a ∧ a ≤ 1 :=
by sorry

end NUMINAMATH_CALUDE_symmetric_points_range_l3845_384586


namespace NUMINAMATH_CALUDE_exam_score_proof_l3845_384523

/-- Given an examination with the following conditions:
  * There are 60 questions in total
  * Each correct answer scores 4 marks
  * Each wrong answer loses 1 mark
  * The total score is 130 marks
  This theorem proves that the number of correctly answered questions is 38. -/
theorem exam_score_proof (total_questions : ℕ) (correct_score : ℤ) (wrong_score : ℤ) (total_score : ℤ)
  (h1 : total_questions = 60)
  (h2 : correct_score = 4)
  (h3 : wrong_score = -1)
  (h4 : total_score = 130) :
  ∃ (correct_answers : ℕ),
    correct_answers = 38 ∧
    correct_answers ≤ total_questions ∧
    (correct_score * correct_answers + wrong_score * (total_questions - correct_answers) = total_score) :=
by sorry

end NUMINAMATH_CALUDE_exam_score_proof_l3845_384523


namespace NUMINAMATH_CALUDE_geometric_sequence_problem_l3845_384526

theorem geometric_sequence_problem (a : ℕ → ℝ) (q : ℝ) :
  q ≠ 1 →
  (∀ n : ℕ, a (n + 1) = q * a n) →
  a 1 + a 2 + a 3 + a 4 + a 5 = 6 →
  a 1^2 + a 2^2 + a 3^2 + a 4^2 + a 5^2 = 18 →
  a 1 - a 2 + a 3 - a 4 + a 5 = 3 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_problem_l3845_384526


namespace NUMINAMATH_CALUDE_fishermen_distribution_l3845_384596

theorem fishermen_distribution (x y z : ℕ) : 
  x + y + z = 16 →
  13 * x + 5 * y + 4 * z = 113 →
  x = 5 ∧ y = 4 ∧ z = 7 := by
sorry

end NUMINAMATH_CALUDE_fishermen_distribution_l3845_384596


namespace NUMINAMATH_CALUDE_trailing_zeros_500_50_l3845_384588

theorem trailing_zeros_500_50 : ∃ n : ℕ, 500^50 = n * 10^100 ∧ n % 10 ≠ 0 := by
  sorry

end NUMINAMATH_CALUDE_trailing_zeros_500_50_l3845_384588


namespace NUMINAMATH_CALUDE_surface_area_unchanged_surface_area_4x4x4_with_corners_removed_l3845_384574

/-- Represents a cube with given side length -/
structure Cube where
  side_length : ℝ
  side_length_pos : side_length > 0

/-- Calculates the surface area of a cube -/
def surface_area (c : Cube) : ℝ := 6 * c.side_length ^ 2

/-- Represents the process of removing corner cubes from a larger cube -/
structure CornerCubeRemoval where
  original_cube : Cube
  corner_cube : Cube
  corner_cube_fits : corner_cube.side_length ≤ original_cube.side_length / 2

/-- Theorem stating that removing corner cubes does not change the surface area -/
theorem surface_area_unchanged (removal : CornerCubeRemoval) :
  surface_area removal.original_cube = surface_area
    { side_length := removal.original_cube.side_length,
      side_length_pos := removal.original_cube.side_length_pos } := by
  sorry

/-- The main theorem proving that a 4x4x4 cube with 2x2x2 corner cubes removed has the same surface area -/
theorem surface_area_4x4x4_with_corners_removed :
  let original_cube : Cube := ⟨4, by norm_num⟩
  let corner_cube : Cube := ⟨2, by norm_num⟩
  let removal : CornerCubeRemoval := ⟨original_cube, corner_cube, by norm_num⟩
  surface_area original_cube = 96 := by
  sorry

end NUMINAMATH_CALUDE_surface_area_unchanged_surface_area_4x4x4_with_corners_removed_l3845_384574


namespace NUMINAMATH_CALUDE_intersection_of_M_and_N_l3845_384530

-- Define the sets M and N
def M : Set ℝ := {x : ℝ | x^2 < 4}
def N : Set ℝ := {x : ℝ | x < 1}

-- State the theorem
theorem intersection_of_M_and_N :
  M ∩ N = {x : ℝ | -2 < x ∧ x < 1} := by sorry

end NUMINAMATH_CALUDE_intersection_of_M_and_N_l3845_384530


namespace NUMINAMATH_CALUDE_expression_not_equal_77_l3845_384566

theorem expression_not_equal_77 (x y : ℤ) :
  x^5 - 4*x^4*y - 5*y^2*x^3 + 20*y^3*x^2 + 4*y^4*x - 16*y^5 ≠ 77 := by
  sorry

end NUMINAMATH_CALUDE_expression_not_equal_77_l3845_384566


namespace NUMINAMATH_CALUDE_father_eats_four_papayas_l3845_384594

/-- The number of papayas Jake eats in one week -/
def jake_papayas : ℕ := 3

/-- The number of papayas Jake's brother eats in one week -/
def brother_papayas : ℕ := 5

/-- The number of weeks Jake is planning for -/
def weeks : ℕ := 4

/-- The total number of papayas Jake needs to buy for 4 weeks -/
def total_papayas : ℕ := 48

/-- The number of papayas Jake's father eats in one week -/
def father_papayas : ℕ := (total_papayas - (jake_papayas + brother_papayas) * weeks) / weeks

theorem father_eats_four_papayas : father_papayas = 4 := by
  sorry

end NUMINAMATH_CALUDE_father_eats_four_papayas_l3845_384594


namespace NUMINAMATH_CALUDE_servant_payment_proof_l3845_384540

/-- Calculates the cash payment for a servant who leaves early -/
def servant_payment (total_salary : ℚ) (turban_value : ℚ) (months_worked : ℚ) : ℚ :=
  (months_worked / 12) * total_salary - turban_value

/-- Proves that a servant working 9 months with given conditions receives Rs. 60 -/
theorem servant_payment_proof :
  let total_salary : ℚ := 120
  let turban_value : ℚ := 30
  let months_worked : ℚ := 9
  servant_payment total_salary turban_value months_worked = 60 := by
sorry

end NUMINAMATH_CALUDE_servant_payment_proof_l3845_384540


namespace NUMINAMATH_CALUDE_total_cards_l3845_384548

theorem total_cards (deck_a deck_b deck_c deck_d : ℕ)
  (ha : deck_a = 52)
  (hb : deck_b = 40)
  (hc : deck_c = 50)
  (hd : deck_d = 48) :
  deck_a + deck_b + deck_c + deck_d = 190 := by
  sorry

end NUMINAMATH_CALUDE_total_cards_l3845_384548


namespace NUMINAMATH_CALUDE_number_divisibility_l3845_384580

theorem number_divisibility (n m : ℤ) : 
  n = 859622 ∧ m = 859560 → 
  ∃ k : ℤ, k ≠ 0 ∧ m = n + (-62) ∧ m % k = 0 :=
by sorry

end NUMINAMATH_CALUDE_number_divisibility_l3845_384580


namespace NUMINAMATH_CALUDE_p_satisfies_conditions_l3845_384544

/-- The polynomial p(x) that satisfies the given conditions -/
def p (x : ℝ) : ℝ := x^2 + 1

/-- Theorem stating that p(x) satisfies the required conditions -/
theorem p_satisfies_conditions :
  (p 3 = 10) ∧
  (∀ x y : ℝ, p x * p y = p x + p y + p (x * y) - 3) := by
  sorry

end NUMINAMATH_CALUDE_p_satisfies_conditions_l3845_384544


namespace NUMINAMATH_CALUDE_custom_op_two_three_custom_op_nested_l3845_384581

-- Define the custom operation
def custom_op (a b : ℝ) : ℝ := a^2 - b + a*b

-- Theorem 1: 2 * 3 = 7
theorem custom_op_two_three : custom_op 2 3 = 7 := by sorry

-- Theorem 2: (-2) * [2 * (-3)] = 1
theorem custom_op_nested : custom_op (-2) (custom_op 2 (-3)) = 1 := by sorry

end NUMINAMATH_CALUDE_custom_op_two_three_custom_op_nested_l3845_384581


namespace NUMINAMATH_CALUDE_quadratic_range_l3845_384537

-- Define the quadratic function
def f (x : ℝ) : ℝ := (x - 1)^2 + 1

-- State the theorem
theorem quadratic_range :
  ∀ x : ℝ, (2 ≤ f x ∧ f x < 5) ↔ (-1 < x ∧ x ≤ 0) ∨ (2 ≤ x ∧ x < 3) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_range_l3845_384537


namespace NUMINAMATH_CALUDE_power_function_through_point_l3845_384556

theorem power_function_through_point (a : ℝ) :
  (2 : ℝ) ^ a = (1 / 2 : ℝ) → a = -1 := by
  sorry

end NUMINAMATH_CALUDE_power_function_through_point_l3845_384556


namespace NUMINAMATH_CALUDE_no_such_function_exists_l3845_384554

theorem no_such_function_exists :
  ¬∃ (f : ℤ → ℤ), ∀ (x y : ℤ), f (x + f y) = f x - y := by
  sorry

end NUMINAMATH_CALUDE_no_such_function_exists_l3845_384554


namespace NUMINAMATH_CALUDE_expenditure_ratio_l3845_384525

theorem expenditure_ratio (anand_income balu_income anand_expenditure balu_expenditure : ℚ) :
  anand_income / balu_income = 5 / 4 →
  anand_income = 2000 →
  anand_income - anand_expenditure = 800 →
  balu_income - balu_expenditure = 800 →
  anand_expenditure / balu_expenditure = 3 / 2 := by
sorry

end NUMINAMATH_CALUDE_expenditure_ratio_l3845_384525


namespace NUMINAMATH_CALUDE_range_of_fraction_l3845_384587

theorem range_of_fraction (a b : ℝ) 
  (ha : -6 < a ∧ a < 8) 
  (hb : 2 < b ∧ b < 3) : 
  -3 < a/b ∧ a/b < 4 := by
  sorry

end NUMINAMATH_CALUDE_range_of_fraction_l3845_384587


namespace NUMINAMATH_CALUDE_t_in_possible_values_l3845_384512

/-- The set of possible values for t given the conditions -/
def possible_t_values : Set ℝ :=
  {t | 3 < t ∧ t < 4}

/-- The point (1, t) is above the line 2x - y + 1 = 0 -/
def above_line (t : ℝ) : Prop :=
  2 * 1 - t + 1 < 0

/-- The inequality x^2 + (2t-4)x + 4 > 0 always holds -/
def inequality_holds (t : ℝ) : Prop :=
  ∀ x, x^2 + (2*t-4)*x + 4 > 0

/-- Given the conditions, prove that t is in the set of possible values -/
theorem t_in_possible_values (t : ℝ) 
  (h1 : above_line t) 
  (h2 : inequality_holds t) : 
  t ∈ possible_t_values :=
sorry

end NUMINAMATH_CALUDE_t_in_possible_values_l3845_384512


namespace NUMINAMATH_CALUDE_ball_problem_l3845_384597

/-- Represents the contents of a box with balls of two colors -/
structure Box where
  white : ℕ
  red : ℕ

/-- Represents the random variable X (number of red balls drawn from box A) -/
inductive X
  | zero
  | one
  | two

def box_A : Box := { white := 2, red := 2 }
def box_B : Box := { white := 1, red := 3 }

def prob_X (x : X) : ℚ :=
  match x with
  | X.zero => 1/6
  | X.one => 2/3
  | X.two => 1/6

def expected_X : ℚ := 1

def prob_red_from_B : ℚ := 2/3

theorem ball_problem :
  (∀ x : X, prob_X x > 0) ∧ 
  (prob_X X.zero + prob_X X.one + prob_X X.two = 1) ∧
  (0 * prob_X X.zero + 1 * prob_X X.one + 2 * prob_X X.two = expected_X) ∧
  prob_red_from_B = 2/3 := by sorry

end NUMINAMATH_CALUDE_ball_problem_l3845_384597


namespace NUMINAMATH_CALUDE_square_root_sum_equals_ten_l3845_384503

theorem square_root_sum_equals_ten : 
  Real.sqrt ((5 - 4 * Real.sqrt 2) ^ 2) + Real.sqrt ((5 + 4 * Real.sqrt 2) ^ 2) = 10 := by
  sorry

end NUMINAMATH_CALUDE_square_root_sum_equals_ten_l3845_384503


namespace NUMINAMATH_CALUDE_article_cost_price_l3845_384575

/-- The cost price of an article given its marked price and profit percentages -/
theorem article_cost_price (marked_price : ℝ) (discount_percent : ℝ) (profit_percent : ℝ) : 
  marked_price = 87.5 → 
  discount_percent = 5 → 
  profit_percent = 25 → 
  (1 - discount_percent / 100) * marked_price = (1 + profit_percent / 100) * (marked_price * (1 - discount_percent / 100) / (1 + profit_percent / 100)) → 
  marked_price * (1 - discount_percent / 100) / (1 + profit_percent / 100) = 66.5 := by
sorry

end NUMINAMATH_CALUDE_article_cost_price_l3845_384575


namespace NUMINAMATH_CALUDE_chocolate_triangles_l3845_384500

theorem chocolate_triangles (square_side : ℝ) (triangle_width : ℝ) (triangle_height : ℝ)
  (h_square : square_side = 10)
  (h_width : triangle_width = 1)
  (h_height : triangle_height = 3) :
  ⌊(square_side^2) / ((triangle_width * triangle_height) / 2)⌋ = 66 := by
  sorry

end NUMINAMATH_CALUDE_chocolate_triangles_l3845_384500


namespace NUMINAMATH_CALUDE_problem_statement_l3845_384514

theorem problem_statement (n : ℕ+) 
  (h1 : ∃ a : ℕ+, (3 * n + 1 : ℕ) = a ^ 2)
  (h2 : ∃ b : ℕ+, (5 * n - 1 : ℕ) = b ^ 2) :
  (∃ p q : ℕ+, p * q = 7 * n + 13 ∧ p ≠ 1 ∧ q ≠ 1) ∧
  (∃ x y : ℕ, 8 * (17 * n^2 + 3 * n) = x^2 + y^2) :=
by sorry

end NUMINAMATH_CALUDE_problem_statement_l3845_384514


namespace NUMINAMATH_CALUDE_simplify_expression_simplify_and_evaluate_l3845_384579

-- Problem 1
theorem simplify_expression (x y : ℝ) : 
  x - (2*x - y) + (3*x - 2*y) = 2*x - y := by sorry

-- Problem 2
theorem simplify_and_evaluate : 
  let x : ℚ := -2/3
  let y : ℚ := 3/2
  2*x*y + (-3*x^3 + 5*x*y + 2) - 3*(2*x*y - x^3 + 1) = -2 := by sorry

end NUMINAMATH_CALUDE_simplify_expression_simplify_and_evaluate_l3845_384579


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_equality_l3845_384549

-- Define an arithmetic sequence
def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

-- Theorem statement
theorem arithmetic_sequence_sum_equality 
  (a : ℕ → ℝ) (h : is_arithmetic_sequence a) : 
  a 1 + a 8 = a 4 + a 5 := by sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_equality_l3845_384549


namespace NUMINAMATH_CALUDE_trihedral_acute_angles_l3845_384567

/-- A trihedral angle is an angle formed by three planes meeting at a point. -/
structure TrihedralAngle where
  /-- The three plane angles of the trihedral angle -/
  planeAngles : Fin 3 → ℝ
  /-- The three dihedral angles of the trihedral angle -/
  dihedralAngles : Fin 3 → ℝ

/-- A predicate to check if an angle is acute -/
def isAcute (angle : ℝ) : Prop := 0 < angle ∧ angle < Real.pi / 2

/-- The main theorem: if all dihedral angles of a trihedral angle are acute,
    then all its plane angles are also acute -/
theorem trihedral_acute_angles (t : TrihedralAngle) 
  (h : ∀ i : Fin 3, isAcute (t.dihedralAngles i)) :
  ∀ i : Fin 3, isAcute (t.planeAngles i) := by
  sorry

end NUMINAMATH_CALUDE_trihedral_acute_angles_l3845_384567


namespace NUMINAMATH_CALUDE_clean_city_workers_l3845_384560

/-- The number of people in Lizzie's group -/
def lizzies_group : ℕ := 54

/-- The difference in members between Lizzie's group and the other group -/
def difference : ℕ := 17

/-- The total number of people working together to clean the city -/
def total_people : ℕ := lizzies_group + (lizzies_group - difference)

/-- Theorem stating that the total number of people working together is 91 -/
theorem clean_city_workers : total_people = 91 := by sorry

end NUMINAMATH_CALUDE_clean_city_workers_l3845_384560


namespace NUMINAMATH_CALUDE_exponential_decreasing_base_less_than_one_l3845_384534

theorem exponential_decreasing_base_less_than_one
  (m n : ℝ) (h1 : m > n) (h2 : n > 0) :
  (0.3 : ℝ) ^ m < (0.3 : ℝ) ^ n :=
by sorry

end NUMINAMATH_CALUDE_exponential_decreasing_base_less_than_one_l3845_384534


namespace NUMINAMATH_CALUDE_jerry_shelf_comparison_l3845_384582

theorem jerry_shelf_comparison : 
  ∀ (initial_action_figures initial_books added_action_figures : ℕ),
    initial_action_figures = 5 →
    initial_books = 9 →
    added_action_figures = 7 →
    (initial_action_figures + added_action_figures) - initial_books = 3 :=
by
  sorry

end NUMINAMATH_CALUDE_jerry_shelf_comparison_l3845_384582


namespace NUMINAMATH_CALUDE_largest_common_divisor_l3845_384571

def product (n : ℕ) : ℕ := (n+1)*(n+3)*(n+5)*(n+7)*(n+9)*(n+11)*(n+13)*(n+15)

theorem largest_common_divisor :
  ∀ n : ℕ, Even n → n > 0 → (14175 ∣ product n) ∧
    ∀ m : ℕ, m > 14175 → ∃ k : ℕ, Even k ∧ k > 0 ∧ ¬(m ∣ product k) :=
by sorry

end NUMINAMATH_CALUDE_largest_common_divisor_l3845_384571


namespace NUMINAMATH_CALUDE_coin_count_l3845_384545

theorem coin_count (total_sum : ℕ) (coin_type1 coin_type2 : ℕ) (count_type1 : ℕ) :
  total_sum = 7100 →
  coin_type1 = 20 →
  coin_type2 = 25 →
  count_type1 = 290 →
  count_type1 * coin_type1 + (total_sum - count_type1 * coin_type1) / coin_type2 = 342 :=
by sorry

end NUMINAMATH_CALUDE_coin_count_l3845_384545


namespace NUMINAMATH_CALUDE_total_wool_calculation_l3845_384564

/-- The number of scarves Aaron makes -/
def aaron_scarves : ℕ := 10

/-- The number of sweaters Aaron makes -/
def aaron_sweaters : ℕ := 5

/-- The number of sweaters Enid makes -/
def enid_sweaters : ℕ := 8

/-- The number of balls of wool used for one scarf -/
def wool_per_scarf : ℕ := 3

/-- The number of balls of wool used for one sweater -/
def wool_per_sweater : ℕ := 4

/-- The total number of balls of wool used by Enid and Aaron -/
def total_wool_used : ℕ :=
  aaron_scarves * wool_per_scarf +
  aaron_sweaters * wool_per_sweater +
  enid_sweaters * wool_per_sweater

theorem total_wool_calculation : total_wool_used = 82 := by
  sorry

end NUMINAMATH_CALUDE_total_wool_calculation_l3845_384564


namespace NUMINAMATH_CALUDE_valid_factorization_l3845_384501

theorem valid_factorization (x : ℝ) : x^2 - 9 = (x - 3) * (x + 3) := by
  sorry

#check valid_factorization

end NUMINAMATH_CALUDE_valid_factorization_l3845_384501


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l3845_384558

def A : Set ℚ := {1, 2, 1/2}

def B : Set ℚ := {y | ∃ x ∈ A, y = x^2}

theorem intersection_of_A_and_B : A ∩ B = {1} := by
  sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l3845_384558


namespace NUMINAMATH_CALUDE_line_plane_perpendicularity_l3845_384578

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relations between lines and planes
variable (perpendicular : Line → Plane → Prop)
variable (notParallel : Line → Line → Prop)
variable (notParallelToPlane : Line → Plane → Prop)
variable (perpendicularPlanes : Plane → Plane → Prop)

-- State the theorem
theorem line_plane_perpendicularity 
  (m n : Line) (α β : Plane) :
  perpendicular m α → 
  notParallel m n → 
  notParallelToPlane n β → 
  perpendicularPlanes α β :=
sorry

end NUMINAMATH_CALUDE_line_plane_perpendicularity_l3845_384578


namespace NUMINAMATH_CALUDE_election_win_percentage_l3845_384502

/-- In a two-candidate election, if a candidate receives 45% of the total votes,
    they need more than 50% of the total votes to win. -/
theorem election_win_percentage (total_votes : ℕ) (candidate_votes : ℕ) 
    (h1 : candidate_votes = (45 : ℕ) * total_votes / 100) 
    (h2 : total_votes > 0) : 
    ∃ (winning_percentage : ℚ), 
      winning_percentage > (1 : ℚ) / 2 ∧ 
      winning_percentage * total_votes > candidate_votes := by
  sorry

end NUMINAMATH_CALUDE_election_win_percentage_l3845_384502


namespace NUMINAMATH_CALUDE_rectangle_area_l3845_384538

/-- The area of a rectangle with perimeter equal to a triangle with sides 7.3, 9.4, and 11.3,
    and length twice its width, is 392/9 square centimeters. -/
theorem rectangle_area (triangle_side1 triangle_side2 triangle_side3 : ℝ)
  (rectangle_width rectangle_length : ℝ) :
  triangle_side1 = 7.3 →
  triangle_side2 = 9.4 →
  triangle_side3 = 11.3 →
  2 * (rectangle_length + rectangle_width) = triangle_side1 + triangle_side2 + triangle_side3 →
  rectangle_length = 2 * rectangle_width →
  rectangle_length * rectangle_width = 392 / 9 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_area_l3845_384538


namespace NUMINAMATH_CALUDE_solve_linear_equation_l3845_384539

theorem solve_linear_equation (x : ℝ) : 3*x - 5*x + 7*x = 210 → x = 42 := by
  sorry

end NUMINAMATH_CALUDE_solve_linear_equation_l3845_384539


namespace NUMINAMATH_CALUDE_cost_price_of_ball_l3845_384547

/-- The cost price of a single ball -/
def cost_price : ℚ := 200/3

/-- The number of balls sold -/
def num_balls : ℕ := 17

/-- The selling price after discount -/
def selling_price_after_discount : ℚ := 720

/-- The discount rate -/
def discount_rate : ℚ := 1/10

/-- The selling price before discount -/
def selling_price_before_discount : ℚ := selling_price_after_discount / (1 - discount_rate)

/-- The theorem stating the cost price of each ball -/
theorem cost_price_of_ball :
  (num_balls * cost_price - selling_price_before_discount = 5 * cost_price) ∧
  (selling_price_after_discount = selling_price_before_discount * (1 - discount_rate)) ∧
  (cost_price = 200/3) :=
sorry

end NUMINAMATH_CALUDE_cost_price_of_ball_l3845_384547


namespace NUMINAMATH_CALUDE_no_real_solutions_to_equation_l3845_384543

theorem no_real_solutions_to_equation :
  ¬∃ x : ℝ, x ≠ 0 ∧ x ≠ 4 ∧ (3 * x^2 - 15 * x) / (x^2 - 4 * x) = x - 2 := by
  sorry

end NUMINAMATH_CALUDE_no_real_solutions_to_equation_l3845_384543


namespace NUMINAMATH_CALUDE_miriam_marbles_to_brother_l3845_384572

/-- Given information about Miriam's marbles -/
structure MiriamMarbles where
  initial : ℕ  -- Initial number of marbles
  current : ℕ  -- Current number of marbles
  to_friend : ℕ  -- Number of marbles given to friend

/-- Theorem: Miriam gave 60 marbles to her brother -/
theorem miriam_marbles_to_brother (m : MiriamMarbles) 
  (h1 : m.initial = 300)
  (h2 : m.current = 30)
  (h3 : m.to_friend = 90) :
  ∃ (to_brother : ℕ), 
    to_brother = 60 ∧ 
    m.initial = m.current + to_brother + 2 * to_brother + m.to_friend :=
by
  sorry

#check miriam_marbles_to_brother

end NUMINAMATH_CALUDE_miriam_marbles_to_brother_l3845_384572


namespace NUMINAMATH_CALUDE_sqrt_2_irrational_l3845_384546

theorem sqrt_2_irrational : ¬ ∃ (p q : ℤ), q ≠ 0 ∧ Real.sqrt 2 = (p : ℚ) / q := by
  sorry

end NUMINAMATH_CALUDE_sqrt_2_irrational_l3845_384546


namespace NUMINAMATH_CALUDE_blue_preference_percentage_l3845_384533

def total_responses : ℕ := 70 + 80 + 50 + 70 + 30

def blue_responses : ℕ := 80

def percentage_blue : ℚ := blue_responses / total_responses * 100

theorem blue_preference_percentage :
  percentage_blue = 80 / 300 * 100 :=
by sorry

end NUMINAMATH_CALUDE_blue_preference_percentage_l3845_384533


namespace NUMINAMATH_CALUDE_circle_area_when_six_times_reciprocal_circumference_equals_diameter_l3845_384557

theorem circle_area_when_six_times_reciprocal_circumference_equals_diameter :
  ∀ (r : ℝ), r > 0 → (6 * (1 / (2 * π * r)) = 2 * r) → π * r^2 = 3/2 := by
  sorry

end NUMINAMATH_CALUDE_circle_area_when_six_times_reciprocal_circumference_equals_diameter_l3845_384557


namespace NUMINAMATH_CALUDE_p_sufficient_not_necessary_l3845_384513

/-- p is the condition a^2 + a ≠ 0 -/
def p (a : ℝ) : Prop := a^2 + a ≠ 0

/-- q is the condition a ≠ 0 -/
def q (a : ℝ) : Prop := a ≠ 0

/-- p is a sufficient but not necessary condition for q -/
theorem p_sufficient_not_necessary : 
  (∀ a : ℝ, p a → q a) ∧ (∃ a : ℝ, q a ∧ ¬p a) := by
  sorry

end NUMINAMATH_CALUDE_p_sufficient_not_necessary_l3845_384513


namespace NUMINAMATH_CALUDE_sixth_grade_students_l3845_384515

/-- The number of students in the sixth grade -/
def total_students : ℕ := 147

/-- The number of books available -/
def total_books : ℕ := 105

/-- The number of boys in the sixth grade -/
def num_boys : ℕ := 84

/-- The number of girls in the sixth grade -/
def num_girls : ℕ := 63

theorem sixth_grade_students :
  (total_students = num_boys + num_girls) ∧
  (total_books = 105) ∧
  (num_boys + (num_girls / 3) = total_books) ∧
  (num_girls + (num_boys / 2) = total_books) :=
by sorry

end NUMINAMATH_CALUDE_sixth_grade_students_l3845_384515


namespace NUMINAMATH_CALUDE_series_sum_equals_one_fourth_l3845_384599

/-- The sum of the infinite series Σ(n=1 to ∞) [3^n / (1 + 3^n + 3^(n+1) + 3^(2n+2))] is equal to 1/4. -/
theorem series_sum_equals_one_fourth :
  ∑' n : ℕ, (3 : ℝ)^n / (1 + 3^n + 3^(n+1) + 3^(2*n+2)) = 1/4 := by
  sorry

end NUMINAMATH_CALUDE_series_sum_equals_one_fourth_l3845_384599


namespace NUMINAMATH_CALUDE_length_AB_area_OCD_l3845_384551

-- Define the parabola
def parabola (x y : ℝ) : Prop := y^2 = 8*x

-- Define the focus of the parabola
def focus : ℝ × ℝ := (2, 0)

-- Define line l passing through the focus and perpendicular to x-axis
def line_l (x y : ℝ) : Prop := x = 2

-- Define line l1 passing through the focus with slope angle 45°
def line_l1 (x y : ℝ) : Prop := y = x - 2

-- Theorem 1: Length of AB
theorem length_AB : 
  ∃ A B : ℝ × ℝ, 
    parabola A.1 A.2 ∧ 
    parabola B.1 B.2 ∧ 
    line_l A.1 A.2 ∧ 
    line_l B.1 B.2 ∧ 
    Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) = 8 := by sorry

-- Theorem 2: Area of triangle OCD
theorem area_OCD : 
  ∃ C D : ℝ × ℝ, 
    parabola C.1 C.2 ∧ 
    parabola D.1 D.2 ∧ 
    line_l1 C.1 C.2 ∧ 
    line_l1 D.1 D.2 ∧ 
    (1/2) * Real.sqrt (C.1^2 + C.2^2) * Real.sqrt (D.1^2 + D.2^2) * 
    Real.sin (Real.arccos ((C.1*D.1 + C.2*D.2) / (Real.sqrt (C.1^2 + C.2^2) * Real.sqrt (D.1^2 + D.2^2)))) = 8 * Real.sqrt 2 := by sorry

end NUMINAMATH_CALUDE_length_AB_area_OCD_l3845_384551


namespace NUMINAMATH_CALUDE_factorial_ratio_l3845_384563

theorem factorial_ratio : Nat.factorial 10 / (Nat.factorial 7 * Nat.factorial 2) = 360 := by
  sorry

end NUMINAMATH_CALUDE_factorial_ratio_l3845_384563


namespace NUMINAMATH_CALUDE_spherical_coordinate_shift_l3845_384507

/-- Given a point with rectangular coordinates (3, -2, 5) and spherical coordinates (r, α, β),
    prove that the point with spherical coordinates (r, α+π, β) has rectangular coordinates (-3, 2, 5). -/
theorem spherical_coordinate_shift (r α β : ℝ) : 
  (3 = r * Real.sin β * Real.cos α) → 
  (-2 = r * Real.sin β * Real.sin α) → 
  (5 = r * Real.cos β) → 
  ((-3, 2, 5) : ℝ × ℝ × ℝ) = (
    r * Real.sin β * Real.cos (α + Real.pi),
    r * Real.sin β * Real.sin (α + Real.pi),
    r * Real.cos β
  ) := by sorry

end NUMINAMATH_CALUDE_spherical_coordinate_shift_l3845_384507


namespace NUMINAMATH_CALUDE_deposit_equals_3400_l3845_384535

/-- Sheela's monthly income in rupees -/
def monthly_income : ℚ := 22666.67

/-- The percentage of monthly income deposited -/
def deposit_percentage : ℚ := 15

/-- The amount deposited in the bank savings account -/
def deposit_amount : ℚ := (deposit_percentage / 100) * monthly_income

/-- Theorem stating that the deposit amount is equal to 3400 rupees -/
theorem deposit_equals_3400 : deposit_amount = 3400 := by
  sorry

end NUMINAMATH_CALUDE_deposit_equals_3400_l3845_384535


namespace NUMINAMATH_CALUDE_unique_six_digit_number_l3845_384595

theorem unique_six_digit_number : ∃! n : ℕ,
  100000 ≤ n ∧ n < 1000000 ∧
  n / 100000 = 1 ∧
  3 * n = (n % 100000) * 10 + 1 ∧
  n = 142857 := by
  sorry

end NUMINAMATH_CALUDE_unique_six_digit_number_l3845_384595


namespace NUMINAMATH_CALUDE_inequality_theorem_equality_condition_l3845_384559

theorem inequality_theorem (a b c : ℝ) :
  Real.sqrt (a^2 + a*b + b^2) + Real.sqrt (a^2 + a*c + c^2) ≥ Real.sqrt (3*a^2 + (a+b+c)^2) :=
sorry

theorem equality_condition (a b c : ℝ) :
  (Real.sqrt (a^2 + a*b + b^2) + Real.sqrt (a^2 + a*c + c^2) = Real.sqrt (3*a^2 + (a+b+c)^2)) ↔
  (b = c ∨ (a = 0 ∧ b*c ≥ 0)) :=
sorry

end NUMINAMATH_CALUDE_inequality_theorem_equality_condition_l3845_384559


namespace NUMINAMATH_CALUDE_problem_statement_l3845_384532

theorem problem_statement :
  (∀ k : ℕ, (∀ a b : ℕ+, ab + (a + 1) * (b + 1) ≠ 2^k) → Nat.Prime (k + 1)) ∧
  (∃ k : ℕ, Nat.Prime (k + 1) ∧ ∃ a b : ℕ+, ab + (a + 1) * (b + 1) = 2^k) := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l3845_384532


namespace NUMINAMATH_CALUDE_bobs_driving_speed_l3845_384584

/-- Bob's driving problem -/
theorem bobs_driving_speed (initial_speed : ℝ) (initial_time : ℝ) (construction_time : ℝ) (total_time : ℝ) (total_distance : ℝ) :
  initial_speed = 60 →
  initial_time = 1.5 →
  construction_time = 2 →
  total_time = 3.5 →
  total_distance = 180 →
  (initial_speed * initial_time + construction_time * ((total_distance - initial_speed * initial_time) / construction_time) = total_distance) →
  (total_distance - initial_speed * initial_time) / construction_time = 45 :=
by sorry

end NUMINAMATH_CALUDE_bobs_driving_speed_l3845_384584


namespace NUMINAMATH_CALUDE_geometric_sequence_ratio_l3845_384521

/-- Given a geometric sequence with common ratio -1/3, 
    prove that the sum of odd-indexed terms up to a₇ 
    divided by the sum of even-indexed terms up to a₈ equals -3 -/
theorem geometric_sequence_ratio (a : ℕ → ℝ) :
  (∀ n : ℕ, a (n + 1) = a n * (-1/3)) →
  (a 1 + a 3 + a 5 + a 7) / (a 2 + a 4 + a 6 + a 8) = -3 := by
sorry


end NUMINAMATH_CALUDE_geometric_sequence_ratio_l3845_384521


namespace NUMINAMATH_CALUDE_quadratic_solution_product_l3845_384550

theorem quadratic_solution_product (d e : ℝ) : 
  (3 * d^2 + 4 * d - 7 = 0) → 
  (3 * e^2 + 4 * e - 7 = 0) → 
  (d + 1) * (e + 1) = -8/3 := by
sorry

end NUMINAMATH_CALUDE_quadratic_solution_product_l3845_384550


namespace NUMINAMATH_CALUDE_M_equals_N_l3845_384561

def M : Set ℝ := {x | ∃ m : ℤ, x = Real.sin ((2 * m - 3) * Real.pi / 6)}

def N : Set ℝ := {y | ∃ n : ℤ, y = Real.cos (n * Real.pi / 3)}

theorem M_equals_N : M = N := by sorry

end NUMINAMATH_CALUDE_M_equals_N_l3845_384561


namespace NUMINAMATH_CALUDE_perpendicular_line_equation_l3845_384565

-- Define the lines l₁ and l₂
def l₁ (m : ℝ) (x y : ℝ) : Prop := x + (1 + m) * y + m - 2 = 0
def l₂ (m : ℝ) (x y : ℝ) : Prop := m * x + 2 * y + 8 = 0

-- Define the point A
def A : ℝ × ℝ := (3, 2)

-- Define the property of two lines being parallel
def parallel (f g : ℝ → ℝ → Prop) : Prop :=
  ∃ (k : ℝ), ∀ (x y : ℝ), f x y ↔ g (k * x) (k * y)

-- Define the property of two lines being perpendicular
def perpendicular (f g : ℝ → ℝ → Prop) : Prop :=
  ∃ (k : ℝ), ∀ (x y : ℝ), f x y ↔ g y (-x)

-- State the theorem
theorem perpendicular_line_equation :
  ∃ (m : ℝ), parallel (l₁ m) (l₂ m) →
  ∃ (f : ℝ → ℝ → Prop),
    perpendicular (l₁ m) f ∧
    f A.1 A.2 ∧
    ∀ (x y : ℝ), f x y ↔ 2 * x - y - 4 = 0 :=
sorry

end NUMINAMATH_CALUDE_perpendicular_line_equation_l3845_384565


namespace NUMINAMATH_CALUDE_bisection_method_step_l3845_384528

-- Define the function f(x) = x^2 + 3x - 1
def f (x : ℝ) : ℝ := x^2 + 3*x - 1

-- State the theorem
theorem bisection_method_step (h1 : f 0 < 0) (h2 : f 0.5 > 0) :
  ∃ x₀ ∈ Set.Ioo 0 0.5, f x₀ = 0 ∧ 0.25 = (0 + 0.5) / 2 := by
  sorry

#check bisection_method_step

end NUMINAMATH_CALUDE_bisection_method_step_l3845_384528


namespace NUMINAMATH_CALUDE_salary_changes_l3845_384529

theorem salary_changes (initial_salary : ℝ) : 
  initial_salary = 2500 → 
  (initial_salary * (1 + 0.15) * (1 - 0.10)) = 2587.50 := by
sorry

end NUMINAMATH_CALUDE_salary_changes_l3845_384529


namespace NUMINAMATH_CALUDE_parallel_segments_k_value_l3845_384542

/-- Given points A, B, X, and Y on a Cartesian plane, prove that if AB is parallel to XY, then k = -8 -/
theorem parallel_segments_k_value (k : ℝ) : 
  let A : ℝ × ℝ := (-6, 2)
  let B : ℝ × ℝ := (2, -6)
  let X : ℝ × ℝ := (0, 10)
  let Y : ℝ × ℝ := (18, k)
  let slope (p q : ℝ × ℝ) := (q.2 - p.2) / (q.1 - p.1)
  slope A B = slope X Y → k = -8 := by
  sorry

end NUMINAMATH_CALUDE_parallel_segments_k_value_l3845_384542


namespace NUMINAMATH_CALUDE_biology_magnet_problem_l3845_384593

def word : Finset Char := {'B', 'I', 'O', 'L', 'O', 'G', 'Y'}
def vowels : Finset Char := {'I', 'O', 'Y'}
def consonants : Finset Char := {'B', 'L', 'G'}

def distinct_collections : ℕ := sorry

theorem biology_magnet_problem :
  (word.card = 7) →
  (vowels ⊆ word) →
  (consonants ⊆ word) →
  (vowels ∩ consonants = ∅) →
  (vowels ∪ consonants = word) →
  (distinct_collections = 12) := by sorry

end NUMINAMATH_CALUDE_biology_magnet_problem_l3845_384593


namespace NUMINAMATH_CALUDE_first_agency_mile_rate_calculation_l3845_384511

-- Define the constants
def first_agency_daily_rate : ℝ := 20.25
def second_agency_daily_rate : ℝ := 18.25
def second_agency_mile_rate : ℝ := 0.22
def crossover_miles : ℝ := 25.0

-- Define the theorem
theorem first_agency_mile_rate_calculation :
  ∃ (x : ℝ),
    first_agency_daily_rate + crossover_miles * x =
    second_agency_daily_rate + crossover_miles * second_agency_mile_rate ∧
    x = 0.14 := by
  sorry

end NUMINAMATH_CALUDE_first_agency_mile_rate_calculation_l3845_384511


namespace NUMINAMATH_CALUDE_average_temperature_l3845_384506

def temperatures : List ℤ := [-36, 13, -15, -10]

theorem average_temperature : 
  (temperatures.sum : ℚ) / temperatures.length = -12 := by
  sorry

end NUMINAMATH_CALUDE_average_temperature_l3845_384506


namespace NUMINAMATH_CALUDE_store_visitor_count_l3845_384517

/-- The number of people who entered the store in the first hour -/
def first_hour_entry : ℕ := 94

/-- The number of people who left the store in the first hour -/
def first_hour_exit : ℕ := 27

/-- The number of people who left the store in the second hour -/
def second_hour_exit : ℕ := 9

/-- The number of people remaining in the store after 2 hours -/
def remaining_after_two_hours : ℕ := 76

/-- The number of people who entered the store in the second hour -/
def second_hour_entry : ℕ := 18

theorem store_visitor_count :
  (first_hour_entry - first_hour_exit) + second_hour_entry - second_hour_exit = remaining_after_two_hours :=
by sorry

end NUMINAMATH_CALUDE_store_visitor_count_l3845_384517


namespace NUMINAMATH_CALUDE_units_digit_17_squared_times_29_l3845_384510

theorem units_digit_17_squared_times_29 : (17^2 * 29) % 10 = 1 := by
  sorry

end NUMINAMATH_CALUDE_units_digit_17_squared_times_29_l3845_384510


namespace NUMINAMATH_CALUDE_complex_problem_l3845_384562

theorem complex_problem (a : ℝ) (z₁ : ℂ) (h₁ : a < 0) (h₂ : z₁ = 1 + a * Complex.I) 
  (h₃ : Complex.re (z₁^2) = 0) : 
  a = -1 ∧ Complex.abs ((z₁ / (1 + Complex.I)) + 2) = Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_complex_problem_l3845_384562


namespace NUMINAMATH_CALUDE_multiple_problem_l3845_384527

theorem multiple_problem (x m : ℝ) (h1 : x = 69) (h2 : x - 18 = m * (86 - x)) : m = 3 := by
  sorry

end NUMINAMATH_CALUDE_multiple_problem_l3845_384527


namespace NUMINAMATH_CALUDE_final_ratio_is_four_to_one_l3845_384553

/-- Represents the dimensions of a rectangle -/
structure Rectangle where
  length : ℕ
  width : ℕ

/-- Represents the transformations applied to the rectangle -/
def transform (r : Rectangle) : Rectangle :=
  let r1 := Rectangle.mk (2 * r.length) (r.width / 2)
  let r2 := 
    if 2 * r1.length > r1.width
    then Rectangle.mk (r1.length + 1) (r1.width - 4)
    else Rectangle.mk (r1.length - 4) (r1.width + 1)
  let r3 := 
    if r2.length > r2.width
    then Rectangle.mk r2.length (r2.width - 1)
    else Rectangle.mk r2.length (r2.width - 1)
  r3

/-- The theorem stating that after transformations, the ratio of sides is 4:1 -/
theorem final_ratio_is_four_to_one (r : Rectangle) :
  let final := transform r
  (final.length : ℚ) / final.width = 4 :=
sorry

end NUMINAMATH_CALUDE_final_ratio_is_four_to_one_l3845_384553


namespace NUMINAMATH_CALUDE_smallest_prime_factor_of_1879_l3845_384576

theorem smallest_prime_factor_of_1879 :
  Nat.minFac 1879 = 17 := by sorry

end NUMINAMATH_CALUDE_smallest_prime_factor_of_1879_l3845_384576


namespace NUMINAMATH_CALUDE_abs_five_necessary_not_sufficient_l3845_384541

theorem abs_five_necessary_not_sufficient :
  (∀ x : ℝ, x = 5 → |x| = 5) ∧
  ¬(∀ x : ℝ, |x| = 5 → x = 5) :=
by sorry

end NUMINAMATH_CALUDE_abs_five_necessary_not_sufficient_l3845_384541


namespace NUMINAMATH_CALUDE_max_sum_sqrt_inequality_l3845_384522

theorem max_sum_sqrt_inequality (x y z : ℝ) 
  (pos_x : 0 < x) (pos_y : 0 < y) (pos_z : 0 < z) 
  (sum_eq_three : x + y + z = 3) : 
  Real.sqrt (2 * x + 1) + Real.sqrt (2 * y + 1) + Real.sqrt (2 * z + 1) ≤ 3 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_max_sum_sqrt_inequality_l3845_384522
