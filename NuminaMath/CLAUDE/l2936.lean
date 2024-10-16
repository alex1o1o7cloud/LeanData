import Mathlib

namespace NUMINAMATH_CALUDE_overall_pass_rate_l2936_293688

theorem overall_pass_rate (a b : ℝ) (ha : 0 ≤ a ∧ a ≤ 1) (hb : 0 ≤ b ∧ b ≤ 1) :
  (1 - a) * (1 - b) = ab - a - b + 1 := by
  sorry

end NUMINAMATH_CALUDE_overall_pass_rate_l2936_293688


namespace NUMINAMATH_CALUDE_total_books_read_l2936_293644

def books_may : ℕ := 2
def books_june : ℕ := 6
def books_july : ℕ := 10
def books_august : ℕ := 14
def books_september : ℕ := 18

theorem total_books_read : books_may + books_june + books_july + books_august + books_september = 50 := by
  sorry

end NUMINAMATH_CALUDE_total_books_read_l2936_293644


namespace NUMINAMATH_CALUDE_xy_value_l2936_293686

theorem xy_value (x y : ℝ) (h : x * (x + y) = x^2 + 12) : x * y = 12 := by
  sorry

end NUMINAMATH_CALUDE_xy_value_l2936_293686


namespace NUMINAMATH_CALUDE_system_solution_l2936_293628

theorem system_solution (x y : ℝ) (eq1 : x + 5 * y = 6) (eq2 : 3 * x - y = 2) : 
  x + y = 2 := by
  sorry

end NUMINAMATH_CALUDE_system_solution_l2936_293628


namespace NUMINAMATH_CALUDE_intersection_sum_l2936_293651

theorem intersection_sum (a b : ℚ) : 
  (∀ x y : ℚ, x = (1/3) * y + a ↔ y = (1/3) * x + b) →
  (3 : ℚ) = (1/3) * 1 + a →
  (1 : ℚ) = (1/3) * 3 + b →
  a + b = 8/3 := by
sorry

end NUMINAMATH_CALUDE_intersection_sum_l2936_293651


namespace NUMINAMATH_CALUDE_polar_equation_is_circle_l2936_293634

-- Define the polar equation
def polar_equation (ρ θ : ℝ) : Prop := ρ = 2 * Real.cos θ - 4 * Real.sin θ

-- Define a circle in Cartesian coordinates
def is_circle (x y h k r : ℝ) : Prop := (x - h)^2 + (y - k)^2 = r^2

-- Theorem statement
theorem polar_equation_is_circle :
  ∃ (h k r : ℝ), ∀ (x y : ℝ),
    (∃ (ρ θ : ℝ), polar_equation ρ θ ∧ x = ρ * Real.cos θ ∧ y = ρ * Real.sin θ) →
    is_circle x y h k r :=
sorry

end NUMINAMATH_CALUDE_polar_equation_is_circle_l2936_293634


namespace NUMINAMATH_CALUDE_money_division_l2936_293625

/-- The problem of dividing money among A, B, and C -/
theorem money_division (a b c : ℚ) : 
  a + b + c = 720 →  -- Total amount is $720
  a = (1/3) * (b + c) →  -- A gets 1/3 of what B and C get
  b = (2/7) * (a + c) →  -- B gets 2/7 of what A and C get
  a > b →  -- A receives more than B
  a - b = 20 :=  -- Prove that A receives $20 more than B
by sorry

end NUMINAMATH_CALUDE_money_division_l2936_293625


namespace NUMINAMATH_CALUDE_inequality_proof_l2936_293604

theorem inequality_proof (x y z : ℝ) 
  (hx : x ≥ 0) (hy : y ≥ 0) (hz : z ≥ 0) 
  (hsum : x + y + z = 1) : 
  (2 * (x^2 + y^2 + z^2) + 9*x*y*z ≥ 1) ∧ 
  (x*y + y*z + z*x - 3*x*y*z ≤ 1/4) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l2936_293604


namespace NUMINAMATH_CALUDE_wire_length_l2936_293668

/-- Given a wire cut into three pieces in the ratio of 7:3:2, where the shortest piece is 16 cm long,
    the total length of the wire before it was cut is 96 cm. -/
theorem wire_length (ratio_long ratio_medium ratio_short : ℕ) 
  (shortest_piece : ℝ) (h1 : ratio_long = 7) (h2 : ratio_medium = 3) 
  (h3 : ratio_short = 2) (h4 : shortest_piece = 16) : 
  (ratio_long + ratio_medium + ratio_short) * (shortest_piece / ratio_short) = 96 := by
  sorry

end NUMINAMATH_CALUDE_wire_length_l2936_293668


namespace NUMINAMATH_CALUDE_inequality_not_always_preserved_l2936_293692

theorem inequality_not_always_preserved (a b : ℝ) (h : a < b) :
  ∃ m : ℝ, m^2 * a ≤ m^2 * b :=
sorry

end NUMINAMATH_CALUDE_inequality_not_always_preserved_l2936_293692


namespace NUMINAMATH_CALUDE_sum_seventeen_terms_l2936_293682

/-- An arithmetic progression with specific properties -/
structure ArithmeticProgression where
  a : ℚ  -- First term
  d : ℚ  -- Common difference
  sum_first_fifth : a + (a + 4 * d) = 5 / 3
  product_third_fourth : (a + 2 * d) * (a + 3 * d) = 65 / 72

/-- Sum of the first n terms of an arithmetic progression -/
def sum_n_terms (ap : ArithmeticProgression) (n : ℕ) : ℚ :=
  n / 2 * (2 * ap.a + (n - 1) * ap.d)

/-- The main theorem to prove -/
theorem sum_seventeen_terms (ap : ArithmeticProgression) :
  sum_n_terms ap 17 = 119 / 3 := by
  sorry

end NUMINAMATH_CALUDE_sum_seventeen_terms_l2936_293682


namespace NUMINAMATH_CALUDE_ball_count_equality_l2936_293602

-- Define the initial state of the urns
def Urn := ℕ → ℕ

-- m: initial number of black balls in the first urn
-- n: initial number of white balls in the second urn
-- k: number of balls transferred between urns
def initial_state (m n k : ℕ) : Urn × Urn :=
  (λ _ => m, λ _ => n)

-- Function to represent the ball transfer process
def transfer_balls (state : Urn × Urn) (k : ℕ) : Urn × Urn :=
  let (urn1, urn2) := state
  let urn1_after := λ color =>
    if color = 0 then urn1 0 - k + (k - (urn2 1 - (urn2 1 - k)))
    else k - (urn2 1 - (urn2 1 - k))
  let urn2_after := λ color =>
    if color = 0 then k - (k - (urn2 1 - (urn2 1 - k)))
    else urn2 1 - k + (urn2 1 - (urn2 1 - k))
  (urn1_after, urn2_after)

theorem ball_count_equality (m n k : ℕ) :
  let (final_urn1, final_urn2) := transfer_balls (initial_state m n k) k
  final_urn1 1 = final_urn2 0 := by
  sorry

end NUMINAMATH_CALUDE_ball_count_equality_l2936_293602


namespace NUMINAMATH_CALUDE_remainder_of_permutation_number_l2936_293657

-- Define a type for permutations of numbers from 1 to 2018
def Permutation := Fin 2018 → Fin 2018

-- Define a function that creates a number from a permutation
def numberFromPermutation (p : Permutation) : ℕ := sorry

-- Theorem statement
theorem remainder_of_permutation_number (p : Permutation) :
  numberFromPermutation p % 3 = 0 := by sorry

end NUMINAMATH_CALUDE_remainder_of_permutation_number_l2936_293657


namespace NUMINAMATH_CALUDE_f_minus_two_range_l2936_293660

def f (a b x : ℝ) : ℝ := a * x^2 + b * x

theorem f_minus_two_range (a b : ℝ) :
  (1 ≤ f a b (-1) ∧ f a b (-1) ≤ 2) →
  (2 ≤ f a b 1 ∧ f a b 1 ≤ 4) →
  ∃ (y : ℝ), y = f a b (-2) ∧ 3 ≤ y ∧ y ≤ 12 :=
by sorry

end NUMINAMATH_CALUDE_f_minus_two_range_l2936_293660


namespace NUMINAMATH_CALUDE_graph_is_single_line_l2936_293607

-- Define the function representing the equation
def f (x y : ℝ) : Prop := (x - 1)^2 * (x + y - 2) = (y - 1)^2 * (x + y - 2)

-- Theorem stating that the graph of f is a single line
theorem graph_is_single_line :
  ∃! (m b : ℝ), ∀ x y : ℝ, f x y ↔ y = m * x + b :=
sorry

end NUMINAMATH_CALUDE_graph_is_single_line_l2936_293607


namespace NUMINAMATH_CALUDE_system_of_equations_solution_l2936_293695

theorem system_of_equations_solution :
  let x : ℚ := -133 / 57
  let y : ℚ := 64 / 19
  (3 * x - 4 * y = -7) ∧ (7 * x - 3 * y = 5) := by
  sorry

end NUMINAMATH_CALUDE_system_of_equations_solution_l2936_293695


namespace NUMINAMATH_CALUDE_cos_300_deg_l2936_293654

/-- Cosine of 300 degrees is equal to 1/2 -/
theorem cos_300_deg : Real.cos (300 * π / 180) = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_cos_300_deg_l2936_293654


namespace NUMINAMATH_CALUDE_isosceles_hyperbola_l2936_293635

/-- 
Given that C ≠ 0 and A and B do not vanish simultaneously,
the equation A x(x^2 - y^2) - (A^2 - B^2) x y = C represents
an isosceles hyperbola with asymptotes A x + B y = 0 and B x - A y = 0
-/
theorem isosceles_hyperbola (A B C : ℝ) (h1 : C ≠ 0) (h2 : ¬(A = 0 ∧ B = 0)) :
  ∃ (x y : ℝ → ℝ), 
    (∀ t, A * (x t) * ((x t)^2 - (y t)^2) - (A^2 - B^2) * (x t) * (y t) = C) ∧ 
    (∃ (t1 t2 : ℝ), t1 ≠ t2 ∧ 
      A * (x t1) + B * (y t1) = 0 ∧ 
      B * (x t2) - A * (y t2) = 0) :=
by sorry

end NUMINAMATH_CALUDE_isosceles_hyperbola_l2936_293635


namespace NUMINAMATH_CALUDE_multiply_63_57_l2936_293691

theorem multiply_63_57 : 63 * 57 = 3591 := by
  sorry

end NUMINAMATH_CALUDE_multiply_63_57_l2936_293691


namespace NUMINAMATH_CALUDE_vectors_orthogonality_l2936_293670

/-- Given plane vectors a and b, prove that (a + b) is orthogonal to (a - b) -/
theorem vectors_orthogonality (a b : ℝ × ℝ) 
  (ha : a = (-1/2, Real.sqrt 3/2)) 
  (hb : b = (Real.sqrt 3/2, -1/2)) : 
  (a + b) • (a - b) = 0 := by sorry

end NUMINAMATH_CALUDE_vectors_orthogonality_l2936_293670


namespace NUMINAMATH_CALUDE_x_range_for_inequality_l2936_293608

theorem x_range_for_inequality (x : ℝ) : 
  (0 ≤ x ∧ x < (1 + Real.sqrt 13) / 3) ↔ 
  (∀ y : ℝ, y > 0 → (2 * (x * y^2 + x^2 * y + 2 * y^2 + 2 * x * y)) / (x + y) > 3 * x^2 * y) :=
by sorry

end NUMINAMATH_CALUDE_x_range_for_inequality_l2936_293608


namespace NUMINAMATH_CALUDE_complement_union_equals_two_five_l2936_293618

-- Define the universal set U
def U : Set Nat := {1, 2, 3, 4, 5}

-- Define sets A and B
def A : Set Nat := {1, 4}
def B : Set Nat := {3, 4}

-- Theorem statement
theorem complement_union_equals_two_five :
  (U \ (A ∪ B)) = {2, 5} := by sorry

end NUMINAMATH_CALUDE_complement_union_equals_two_five_l2936_293618


namespace NUMINAMATH_CALUDE_f_extrema_on_interval_l2936_293637

noncomputable def f (x : ℝ) := x^3 + 3*x^2 - 9*x + 1

theorem f_extrema_on_interval :
  let a := -4
  let b := 4
  ∃ (x_max x_min : ℝ), x_max ∈ Set.Icc a b ∧ x_min ∈ Set.Icc a b ∧
    (∀ x ∈ Set.Icc a b, f x ≤ f x_max) ∧
    (∀ x ∈ Set.Icc a b, f x_min ≤ f x) ∧
    f x_max = 77 ∧ f x_min = -4 :=
by sorry

end NUMINAMATH_CALUDE_f_extrema_on_interval_l2936_293637


namespace NUMINAMATH_CALUDE_perfect_cube_in_range_l2936_293683

theorem perfect_cube_in_range (Y J : ℤ) : 
  (150 < Y) → (Y < 300) → (Y = J^5) → (∃ n : ℤ, Y = n^3) → J = 3 := by
  sorry

end NUMINAMATH_CALUDE_perfect_cube_in_range_l2936_293683


namespace NUMINAMATH_CALUDE_degrees_to_radians_l2936_293678

theorem degrees_to_radians (degrees : ℝ) (radians : ℝ) : 
  degrees = 12 → radians = degrees * (π / 180) → radians = π / 15 := by
  sorry

end NUMINAMATH_CALUDE_degrees_to_radians_l2936_293678


namespace NUMINAMATH_CALUDE_solve_ages_l2936_293606

/-- Represents the ages of people in the problem -/
structure Ages where
  rehana : ℕ
  phoebe : ℕ
  jacob : ℕ
  xander : ℕ

/-- The conditions of the problem -/
def problem_conditions (ages : Ages) : Prop :=
  ages.rehana = 25 ∧
  ages.rehana + 5 = 3 * (ages.phoebe + 5) ∧
  ages.jacob = 3 * ages.phoebe / 5 ∧
  ages.xander = ages.rehana + ages.jacob - 4

/-- The theorem to prove -/
theorem solve_ages : 
  ∃ (ages : Ages), problem_conditions ages ∧ 
    ages.rehana = 25 ∧ 
    ages.phoebe = 5 ∧ 
    ages.jacob = 3 ∧ 
    ages.xander = 24 := by
  sorry

end NUMINAMATH_CALUDE_solve_ages_l2936_293606


namespace NUMINAMATH_CALUDE_machine_purchase_price_l2936_293655

/-- Represents the purchase price of the machine in rupees -/
def purchase_price : ℕ := sorry

/-- Represents the repair cost in rupees -/
def repair_cost : ℕ := 5000

/-- Represents the transportation charges in rupees -/
def transportation_charges : ℕ := 1000

/-- Represents the profit percentage -/
def profit_percentage : ℚ := 50 / 100

/-- Represents the selling price in rupees -/
def selling_price : ℕ := 27000

/-- Theorem stating that the purchase price is 12000 rupees -/
theorem machine_purchase_price : 
  purchase_price = 12000 ∧
  (purchase_price + repair_cost + transportation_charges) * (1 + profit_percentage) = selling_price :=
sorry

end NUMINAMATH_CALUDE_machine_purchase_price_l2936_293655


namespace NUMINAMATH_CALUDE_magic_8_ball_probability_l2936_293621

/-- The probability of getting exactly k positive answers out of n questions
    when each question has a probability p of getting a positive answer. -/
def binomial_probability (n k : ℕ) (p : ℝ) : ℝ :=
  (n.choose k : ℝ) * p^k * (1 - p)^(n - k)

/-- The probability of getting exactly 3 positive answers when asking 6 questions
    to a Magic 8 Ball, where each question has a 1/2 chance of getting a positive answer. -/
theorem magic_8_ball_probability : binomial_probability 6 3 (1/2) = 5/16 := by
  sorry

end NUMINAMATH_CALUDE_magic_8_ball_probability_l2936_293621


namespace NUMINAMATH_CALUDE_triangle_area_from_squares_l2936_293613

theorem triangle_area_from_squares (a b c : ℝ) (ha : a^2 = 36) (hb : b^2 = 64) (hc : c^2 = 100) :
  (1/2) * a * b = 24 := by
  sorry

end NUMINAMATH_CALUDE_triangle_area_from_squares_l2936_293613


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_l2936_293609

theorem quadratic_inequality_solution (n : ℤ) : 
  n^2 - 13*n + 36 < 0 ↔ n ∈ ({5, 6, 7, 8} : Set ℤ) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_l2936_293609


namespace NUMINAMATH_CALUDE_older_ate_12_pancakes_l2936_293616

/-- Represents the pancake eating scenario --/
structure PancakeScenario where
  initial_pancakes : ℕ
  younger_eats : ℕ
  older_eats : ℕ
  grandma_bakes : ℕ
  final_pancakes : ℕ

/-- Calculates the number of pancakes eaten by the older grandchild --/
def older_grandchild_pancakes (scenario : PancakeScenario) : ℕ :=
  let net_reduction := scenario.younger_eats + scenario.older_eats - scenario.grandma_bakes
  let cycles := (scenario.initial_pancakes - scenario.final_pancakes) / net_reduction
  scenario.older_eats * cycles

/-- Theorem stating that in the given scenario, the older grandchild ate 12 pancakes --/
theorem older_ate_12_pancakes (scenario : PancakeScenario) 
  (h1 : scenario.initial_pancakes = 19)
  (h2 : scenario.younger_eats = 1)
  (h3 : scenario.older_eats = 3)
  (h4 : scenario.grandma_bakes = 2)
  (h5 : scenario.final_pancakes = 11) :
  older_grandchild_pancakes scenario = 12 := by
  sorry

end NUMINAMATH_CALUDE_older_ate_12_pancakes_l2936_293616


namespace NUMINAMATH_CALUDE_equation_solution_l2936_293687

theorem equation_solution : 
  {x : ℝ | -x^2 = (2*x + 4)/(x + 2)} = {-2, -1} :=
by sorry

end NUMINAMATH_CALUDE_equation_solution_l2936_293687


namespace NUMINAMATH_CALUDE_tangent_lines_for_cubic_curve_l2936_293641

def f (x : ℝ) := x^3 - x

theorem tangent_lines_for_cubic_curve :
  let C := f
  -- Tangent line at (2, f(2))
  ∃ (m b : ℝ), (∀ x y, y = m*x + b ↔ m*x - y + b = 0) ∧
               m = 3*2^2 - 1 ∧
               b = f 2 - m*2 ∧
               m*2 - f 2 + b = 0
  ∧
  -- Tangent lines parallel to y = 5x + 3
  ∃ (x₁ x₂ : ℝ), x₁ ≠ x₂ ∧
                 x₁^2 = 2 ∧ x₂^2 = 2 ∧
                 (∀ x y, y - f x₁ = 5*(x - x₁) ↔ 5*x - y - 4*Real.sqrt 2 = 0) ∧
                 (∀ x y, y - f x₂ = 5*(x - x₂) ↔ 5*x - y + 4*Real.sqrt 2 = 0) := by
  sorry

end NUMINAMATH_CALUDE_tangent_lines_for_cubic_curve_l2936_293641


namespace NUMINAMATH_CALUDE_quadratic_factorization_l2936_293671

theorem quadratic_factorization :
  ∃ (a b : ℤ), (∀ y : ℝ, 4 * y^2 - 9 * y - 36 = (4 * y + a) * (y + b)) ∧ (a - b = 13) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_factorization_l2936_293671


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l2936_293653

/-- Given an arithmetic sequence, if the sum of the first n terms is P 
    and the sum of the first 2n terms is q, then the sum of the first 3n terms is 3(2P - q). -/
theorem arithmetic_sequence_sum (n : ℕ) (P q : ℝ) :
  (∃ (a d : ℝ), P = n / 2 * (2 * a + (n - 1) * d) ∧ q = n * (2 * a + (2 * n - 1) * d)) →
  (∃ (S_3n : ℝ), S_3n = 3 * (2 * P - q)) :=
sorry


end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l2936_293653


namespace NUMINAMATH_CALUDE_quadratic_equation_solution_l2936_293663

theorem quadratic_equation_solution : 
  {x : ℝ | x^2 = 4*x} = {0, 4} := by sorry

end NUMINAMATH_CALUDE_quadratic_equation_solution_l2936_293663


namespace NUMINAMATH_CALUDE_inner_rectangle_length_l2936_293679

/-- Represents the dimensions of a rectangular region -/
structure RectDimensions where
  length : ℝ
  width : ℝ

/-- Calculates the area of a rectangle given its dimensions -/
def rectangleArea (dim : RectDimensions) : ℝ :=
  dim.length * dim.width

/-- Represents the three regions of the rug -/
structure RugRegions where
  inner : RectDimensions
  middle : RectDimensions
  outer : RectDimensions

/-- Theorem stating that the length of the inner rectangle is 4 feet -/
theorem inner_rectangle_length (rug : RugRegions) : rug.inner.length = 4 :=
  by
  -- Assuming the following conditions:
  have inner_width : rug.inner.width = 2 := by sorry
  have middle_surround : rug.middle.length = rug.inner.length + 4 ∧ 
                         rug.middle.width = rug.inner.width + 4 := by sorry
  have outer_surround : rug.outer.length = rug.middle.length + 4 ∧ 
                        rug.outer.width = rug.middle.width + 4 := by sorry
  have areas_arithmetic_progression : 
    (rectangleArea rug.middle - rectangleArea rug.inner) = 
    (rectangleArea rug.outer - rectangleArea rug.middle) := by sorry
  
  sorry -- Proof goes here

end NUMINAMATH_CALUDE_inner_rectangle_length_l2936_293679


namespace NUMINAMATH_CALUDE_sum_and_difference_of_numbers_l2936_293667

theorem sum_and_difference_of_numbers : ∃ (a b : ℕ), 
  b = 100 * a ∧ 
  a + b = 36400 ∧ 
  b - a = 35640 := by
sorry

end NUMINAMATH_CALUDE_sum_and_difference_of_numbers_l2936_293667


namespace NUMINAMATH_CALUDE_charles_pictures_before_work_l2936_293697

/-- The number of pictures Charles drew before going to work yesterday -/
def pictures_before_work : ℕ → ℕ → ℕ → ℕ → ℕ
  | total_papers, papers_left, pictures_today, pictures_after_work =>
    total_papers - papers_left - pictures_today - pictures_after_work

theorem charles_pictures_before_work :
  pictures_before_work 20 2 6 6 = 6 := by
  sorry

end NUMINAMATH_CALUDE_charles_pictures_before_work_l2936_293697


namespace NUMINAMATH_CALUDE_opposite_of_negative_seven_thirds_l2936_293661

theorem opposite_of_negative_seven_thirds :
  ∃ y : ℚ, -7/3 + y = 0 ∧ y = 7/3 := by
sorry

end NUMINAMATH_CALUDE_opposite_of_negative_seven_thirds_l2936_293661


namespace NUMINAMATH_CALUDE_find_other_divisor_l2936_293614

theorem find_other_divisor : ∃ (x : ℕ), x > 1 ∧ 261 % 7 = 2 ∧ 261 % x = 2 ∧ ∀ (y : ℕ), y > 1 → 261 % y = 2 → y = 7 ∨ y = x := by
  sorry

end NUMINAMATH_CALUDE_find_other_divisor_l2936_293614


namespace NUMINAMATH_CALUDE_license_plate_difference_l2936_293676

/-- The number of letters in the alphabet -/
def num_letters : ℕ := 26

/-- The number of digits -/
def num_digits : ℕ := 10

/-- The number of possible Sunland license plates -/
def sunland_plates : ℕ := num_letters^4 * num_digits^2

/-- The number of possible Moonland license plates -/
def moonland_plates : ℕ := num_letters^3 * num_digits^3

/-- The difference in the number of possible license plates between Sunland and Moonland -/
theorem license_plate_difference :
  sunland_plates - moonland_plates = 7321600 :=
by sorry

end NUMINAMATH_CALUDE_license_plate_difference_l2936_293676


namespace NUMINAMATH_CALUDE_sin_pi_over_4n_lower_bound_l2936_293648

theorem sin_pi_over_4n_lower_bound (n : ℕ) (hn : n > 0) :
  Real.sin (π / (4 * n)) ≥ Real.sqrt 2 / (2 * n) := by
  sorry

end NUMINAMATH_CALUDE_sin_pi_over_4n_lower_bound_l2936_293648


namespace NUMINAMATH_CALUDE_expansion_properties_l2936_293633

def binomial_sum (n : ℕ) : ℕ := 2^n

def constant_term (n : ℕ) : ℕ := Nat.choose n (n / 2)

theorem expansion_properties :
  ∃ (n : ℕ), 
    binomial_sum n = 64 ∧ 
    constant_term n = 15 := by
  sorry

end NUMINAMATH_CALUDE_expansion_properties_l2936_293633


namespace NUMINAMATH_CALUDE_d_t_eventually_two_exists_n_d_t_two_from_m_l2936_293612

/-- The number of positive divisors of n -/
def d (n : ℕ) : ℕ := (Nat.divisors n).card

/-- The t-th iteration of d applied to n -/
def d_t (t n : ℕ) : ℕ :=
  match t with
  | 0 => n
  | t + 1 => d (d_t t n)

/-- For any n > 1, the sequence d_t(n) eventually becomes 2 -/
theorem d_t_eventually_two (n : ℕ) (h : n > 1) :
  ∃ k, ∀ t, t ≥ k → d_t t n = 2 := by sorry

/-- For any m, there exists an n such that d_t(n) becomes 2 from the m-th term onwards -/
theorem exists_n_d_t_two_from_m (m : ℕ) :
  ∃ n, ∀ t, t ≥ m → d_t t n = 2 := by sorry

end NUMINAMATH_CALUDE_d_t_eventually_two_exists_n_d_t_two_from_m_l2936_293612


namespace NUMINAMATH_CALUDE_arithmetic_expression_equality_l2936_293645

theorem arithmetic_expression_equality : 
  (50 - (4050 - 450)) * (4050 - (450 - 50)) = -12957500 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_expression_equality_l2936_293645


namespace NUMINAMATH_CALUDE_election_majority_l2936_293630

/-- Calculates the majority in an election --/
theorem election_majority (total_votes : ℕ) (winning_percentage : ℚ) : 
  total_votes = 7520 → 
  winning_percentage = 60 / 100 → 
  (winning_percentage * total_votes : ℚ).floor - ((1 - winning_percentage) * total_votes : ℚ).floor = 1504 := by
  sorry


end NUMINAMATH_CALUDE_election_majority_l2936_293630


namespace NUMINAMATH_CALUDE_complex_equation_solution_l2936_293659

theorem complex_equation_solution (z : ℂ) (h : z * (1 - Complex.I) = 1 + Complex.I) : z = Complex.I := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l2936_293659


namespace NUMINAMATH_CALUDE_gasoline_price_change_l2936_293627

/-- The price of gasoline after five months of changes -/
def final_price (initial_price : ℝ) : ℝ :=
  initial_price * 1.30 * 0.75 * 1.10 * 0.85 * 0.80

/-- Theorem stating the relationship between the initial and final price -/
theorem gasoline_price_change (initial_price : ℝ) :
  final_price initial_price = 102.60 → initial_price = 140.67 := by
  sorry

#eval final_price 140.67

end NUMINAMATH_CALUDE_gasoline_price_change_l2936_293627


namespace NUMINAMATH_CALUDE_circle_and_max_distance_l2936_293690

-- Define the circle C
def Circle (center : ℝ × ℝ) (radius : ℝ) :=
  {p : ℝ × ℝ | (p.1 - center.1)^2 + (p.2 - center.2)^2 = radius^2}

-- Define the ray 3x - y = 0 (x ≥ 0)
def Ray := {p : ℝ × ℝ | 3 * p.1 - p.2 = 0 ∧ p.1 ≥ 0}

-- Define the line x = 4
def TangentLine := {p : ℝ × ℝ | p.1 = 4}

-- Define the line 3x + 4y + 10 = 0
def ChordLine := {p : ℝ × ℝ | 3 * p.1 + 4 * p.2 + 10 = 0}

-- Define points A and B
def A : ℝ × ℝ := (1, 1)
def B : ℝ × ℝ := (-2, 0)

-- Define the theorem
theorem circle_and_max_distance :
  ∃ (center : ℝ × ℝ) (radius : ℝ),
    -- Circle C's center is on the ray
    center ∈ Ray ∧
    -- Circle C is tangent to the line x = 4
    (∃ (p : ℝ × ℝ), p ∈ Circle center radius ∧ p ∈ TangentLine) ∧
    -- The chord intercepted by the line has length 4√3
    (∃ (p q : ℝ × ℝ), p ∈ Circle center radius ∧ q ∈ Circle center radius ∧
      p ∈ ChordLine ∧ q ∈ ChordLine ∧
      (p.1 - q.1)^2 + (p.2 - q.2)^2 = 48) ∧
    -- The equation of circle C is x^2 + y^2 = 16
    Circle center radius = {p : ℝ × ℝ | p.1^2 + p.2^2 = 16} ∧
    -- The maximum value of |PA|^2 + |PB|^2 is 38 + 8√2
    (∀ (p : ℝ × ℝ), p ∈ Circle center radius →
      (p.1 - A.1)^2 + (p.2 - A.2)^2 + (p.1 - B.1)^2 + (p.2 - B.2)^2 ≤ 38 + 8 * Real.sqrt 2) ∧
    (∃ (p : ℝ × ℝ), p ∈ Circle center radius ∧
      (p.1 - A.1)^2 + (p.2 - A.2)^2 + (p.1 - B.1)^2 + (p.2 - B.2)^2 = 38 + 8 * Real.sqrt 2) :=
by sorry

end NUMINAMATH_CALUDE_circle_and_max_distance_l2936_293690


namespace NUMINAMATH_CALUDE_money_difference_l2936_293603

/-- Given an initial amount of money and an additional amount received, 
    prove that the difference between the final amount and the initial amount 
    is equal to the additional amount received. -/
theorem money_difference (initial additional : ℕ) : 
  (initial + additional) - initial = additional := by
  sorry

end NUMINAMATH_CALUDE_money_difference_l2936_293603


namespace NUMINAMATH_CALUDE_badminton_survey_k_squared_l2936_293624

/-- Represents the contingency table for the badminton survey --/
structure ContingencyTable :=
  (male_like : ℕ)
  (male_dislike : ℕ)
  (female_like : ℕ)
  (female_dislike : ℕ)

/-- Calculates the K² statistic for a given contingency table --/
def calculate_k_squared (table : ContingencyTable) : ℚ :=
  let n := table.male_like + table.male_dislike + table.female_like + table.female_dislike
  let a := table.male_like
  let b := table.male_dislike
  let c := table.female_like
  let d := table.female_dislike
  (n * (a * d - b * c)^2 : ℚ) / ((a + b) * (c + d) * (a + c) * (b + d))

theorem badminton_survey_k_squared :
  ∀ (table : ContingencyTable),
    table.male_like + table.male_dislike = 100 →
    table.female_like + table.female_dislike = 100 →
    table.male_like = 40 →
    table.female_dislike = 90 →
    calculate_k_squared table = 24 := by
  sorry

end NUMINAMATH_CALUDE_badminton_survey_k_squared_l2936_293624


namespace NUMINAMATH_CALUDE_triangle_angle_problem_l2936_293649

theorem triangle_angle_problem (a b c : ℝ) : 
  a > 0 ∧ b > 0 ∧ c > 0 → -- angles are positive
  b = 2 * a → -- second angle is double the first
  c = a - 40 → -- third angle is 40 less than the first
  a + b + c = 180 → -- sum of angles in a triangle
  a = 55 := by sorry

end NUMINAMATH_CALUDE_triangle_angle_problem_l2936_293649


namespace NUMINAMATH_CALUDE_inequality_proof_l2936_293699

theorem inequality_proof (x y : ℝ) (hx : x < 0) (hy : y < 0) :
  x^4 / y^4 + y^4 / x^4 - x^2 / y^2 - y^2 / x^2 + x / y + y / x ≥ 2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l2936_293699


namespace NUMINAMATH_CALUDE_ratio_proof_l2936_293652

theorem ratio_proof (a b c k : ℕ) (h1 : a < b) (h2 : b < c) (h3 : c = 56) 
  (h4 : c - a = 32) (h5 : a = 3 * k) (h6 : b = 5 * k) : 
  c / b = 7 / 5 := by
  sorry

end NUMINAMATH_CALUDE_ratio_proof_l2936_293652


namespace NUMINAMATH_CALUDE_probability_four_twos_correct_l2936_293610

def num_dice : ℕ := 12
def num_sides : ℕ := 6
def num_success : ℕ := 4

def probability_exactly_four_twos : ℚ :=
  (Nat.choose num_dice num_success) *
  (1 / num_sides) ^ num_success *
  ((num_sides - 1) / num_sides) ^ (num_dice - num_success)

theorem probability_four_twos_correct :
  probability_exactly_four_twos = 
    (Nat.choose num_dice num_success) *
    (1 / num_sides) ^ num_success *
    ((num_sides - 1) / num_sides) ^ (num_dice - num_success) :=
by sorry

end NUMINAMATH_CALUDE_probability_four_twos_correct_l2936_293610


namespace NUMINAMATH_CALUDE_negation_equivalence_l2936_293647

theorem negation_equivalence : 
  (¬ ∃ x : ℝ, x^2 > 1) ↔ (∀ x : ℝ, -1 ≤ x ∧ x ≤ 1) :=
by sorry

end NUMINAMATH_CALUDE_negation_equivalence_l2936_293647


namespace NUMINAMATH_CALUDE_largest_three_digit_special_divisibility_l2936_293638

theorem largest_three_digit_special_divisibility : ∃ n : ℕ,
  (100 ≤ n ∧ n ≤ 999) ∧ 
  (∀ d : ℕ, d ∈ n.digits 10 → d ≠ 0 → n % d = 0) ∧
  (n % 11 = 0) ∧
  (∀ m : ℕ, (100 ≤ m ∧ m ≤ 999) → 
    (∀ d : ℕ, d ∈ m.digits 10 → d ≠ 0 → m % d = 0) →
    (m % 11 = 0) → m ≤ n) ∧
  n = 924 :=
by sorry

end NUMINAMATH_CALUDE_largest_three_digit_special_divisibility_l2936_293638


namespace NUMINAMATH_CALUDE_crabapple_sequences_l2936_293693

/-- The number of students in Mrs. Crabapple's class -/
def num_students : ℕ := 12

/-- The number of times the class meets in a week -/
def meetings_per_week : ℕ := 5

/-- The number of different sequences of crabapple recipients possible in a week -/
def num_sequences : ℕ := num_students ^ meetings_per_week

theorem crabapple_sequences :
  num_sequences = 248832 := by
  sorry

end NUMINAMATH_CALUDE_crabapple_sequences_l2936_293693


namespace NUMINAMATH_CALUDE_all_rooms_on_same_hall_l2936_293642

/-- A type representing a hall in the castle -/
def Hall : Type := ℕ

/-- A function that assigns a hall to each room number -/
def room_to_hall : ℕ → Hall := sorry

/-- The property that room n is on the same hall as rooms 2n+1 and 3n+1 -/
def hall_property (room_to_hall : ℕ → Hall) : Prop :=
  ∀ n : ℕ, (room_to_hall n = room_to_hall (2*n + 1)) ∧ (room_to_hall n = room_to_hall (3*n + 1))

/-- The theorem stating that all rooms must be on the same hall -/
theorem all_rooms_on_same_hall (room_to_hall : ℕ → Hall) 
  (h : hall_property room_to_hall) : 
  ∀ m n : ℕ, room_to_hall m = room_to_hall n :=
by sorry

end NUMINAMATH_CALUDE_all_rooms_on_same_hall_l2936_293642


namespace NUMINAMATH_CALUDE_min_value_theorem_l2936_293658

theorem min_value_theorem (a b c : ℝ) :
  (∀ x y : ℝ, 3*x + 4*y - 5 ≤ a*x + b*y + c ∧ a*x + b*y + c ≤ 3*x + 4*y + 5) →
  2 ≤ a + b - c :=
by sorry

end NUMINAMATH_CALUDE_min_value_theorem_l2936_293658


namespace NUMINAMATH_CALUDE_faculty_reduction_l2936_293694

theorem faculty_reduction (initial_faculty : ℝ) (reduction_percentage : ℝ) : 
  initial_faculty = 243.75 →
  reduction_percentage = 20 →
  initial_faculty * (1 - reduction_percentage / 100) = 195 := by
  sorry

end NUMINAMATH_CALUDE_faculty_reduction_l2936_293694


namespace NUMINAMATH_CALUDE_fraction_to_decimal_l2936_293673

theorem fraction_to_decimal : (58 : ℚ) / 125 = (464 : ℚ) / 1000 := by sorry

end NUMINAMATH_CALUDE_fraction_to_decimal_l2936_293673


namespace NUMINAMATH_CALUDE_felix_trees_chopped_l2936_293611

/-- Calculates the minimum number of trees chopped given the total spent on sharpening,
    cost per sharpening, and trees chopped before resharpening is needed. -/
def min_trees_chopped (total_spent : ℕ) (cost_per_sharpening : ℕ) (trees_per_sharpening : ℕ) : ℕ :=
  (total_spent / cost_per_sharpening) * trees_per_sharpening

/-- Proves that Felix has chopped down at least 150 trees given the problem conditions. -/
theorem felix_trees_chopped :
  let total_spent : ℕ := 48
  let cost_per_sharpening : ℕ := 8
  let trees_per_sharpening : ℕ := 25
  min_trees_chopped total_spent cost_per_sharpening trees_per_sharpening = 150 := by
  sorry

#eval min_trees_chopped 48 8 25  -- Should output 150

end NUMINAMATH_CALUDE_felix_trees_chopped_l2936_293611


namespace NUMINAMATH_CALUDE_min_product_of_three_l2936_293617

def S : Set Int := {-10, -7, -5, 0, 4, 6, 9}

theorem min_product_of_three (a b c : Int) (ha : a ∈ S) (hb : b ∈ S) (hc : c ∈ S) 
  (hab : a ≠ b) (hbc : b ≠ c) (hac : a ≠ c) :
  ∃ (x y z : Int), x ∈ S ∧ y ∈ S ∧ z ∈ S ∧ x ≠ y ∧ y ≠ z ∧ x ≠ z ∧ 
  x * y * z ≤ a * b * c ∧ x * y * z = -540 := by
  sorry

end NUMINAMATH_CALUDE_min_product_of_three_l2936_293617


namespace NUMINAMATH_CALUDE_data_analysis_l2936_293662

def data : List ℝ := [7, 11, 10, 11, 6, 14, 11, 10, 11, 9]

def mode (l : List ℝ) : ℝ := sorry

def median (l : List ℝ) : ℝ := sorry

def mean (l : List ℝ) : ℝ := sorry

def variance (l : List ℝ) : ℝ := sorry

theorem data_analysis (d : List ℝ) (h : d = data) : 
  mode d = 11 ∧ 
  median d ≠ 10 ∧ 
  mean d = 10 ∧ 
  variance d = 4.6 := by sorry

end NUMINAMATH_CALUDE_data_analysis_l2936_293662


namespace NUMINAMATH_CALUDE_divisibility_of_expression_l2936_293698

theorem divisibility_of_expression (a b : ℤ) : 
  ∃ k : ℤ, (2*a + 3)^2 - (2*b + 1)^2 = 8 * k := by
sorry

end NUMINAMATH_CALUDE_divisibility_of_expression_l2936_293698


namespace NUMINAMATH_CALUDE_max_sqrt_sum_l2936_293677

theorem max_sqrt_sum (x y : ℝ) (hx : x ≥ 0) (hy : y ≥ 0) (hsum : x + y = 18) :
  ∃ d : ℝ, d = 6 ∧ ∀ a b : ℝ, a ≥ 0 → b ≥ 0 → a + b = 18 → Real.sqrt a + Real.sqrt b ≤ d :=
by sorry

end NUMINAMATH_CALUDE_max_sqrt_sum_l2936_293677


namespace NUMINAMATH_CALUDE_equality_check_l2936_293619

-- Define the set {-1, 0, 1}
def S : Set ℝ := {-1, 0, 1}

-- Define the equality we're checking
def f (a : ℝ) : Prop := (a^4 - 1)^6 = (a^6 - 1)^4

theorem equality_check :
  (∃ a : ℝ, ¬(f a)) ∧ (∀ a ∈ S, f a) := by
  sorry

end NUMINAMATH_CALUDE_equality_check_l2936_293619


namespace NUMINAMATH_CALUDE_bird_photo_combinations_l2936_293629

/-- Represents the number of pairs of birds -/
def num_pairs : ℕ := 5

/-- Calculates the number of ways to photograph birds with alternating genders -/
def photo_combinations (n : ℕ) : ℕ :=
  let female_choices := List.range n
  let male_choices := List.range (n - 1)
  (female_choices.foldl (· * ·) 1) * (male_choices.foldl (· * ·) 1)

/-- Theorem stating the number of ways to photograph the birds -/
theorem bird_photo_combinations :
  photo_combinations num_pairs = 2880 := by
  sorry

end NUMINAMATH_CALUDE_bird_photo_combinations_l2936_293629


namespace NUMINAMATH_CALUDE_greatest_product_of_three_l2936_293674

def S : Finset Int := {-6, -4, -2, 0, 1, 3, 5, 7}

theorem greatest_product_of_three (a b c : Int) : 
  a ∈ S → b ∈ S → c ∈ S → 
  a ≠ b → b ≠ c → a ≠ c → 
  ∀ x y z : Int, x ∈ S → y ∈ S → z ∈ S → 
  x ≠ y → y ≠ z → x ≠ z → 
  a * b * c ≤ 168 ∧ (∃ p q r : Int, p ∈ S ∧ q ∈ S ∧ r ∈ S ∧ p ≠ q ∧ q ≠ r ∧ p ≠ r ∧ p * q * r = 168) :=
by sorry

end NUMINAMATH_CALUDE_greatest_product_of_three_l2936_293674


namespace NUMINAMATH_CALUDE_similar_right_triangles_l2936_293620

theorem similar_right_triangles (y : ℝ) : 
  y > 0 →  -- ensure y is positive
  (16 : ℝ) / y = 12 / 9 → 
  y = 12 := by
sorry

end NUMINAMATH_CALUDE_similar_right_triangles_l2936_293620


namespace NUMINAMATH_CALUDE_john_post_break_time_l2936_293675

/-- The number of hours John danced before the break -/
def john_pre_break : ℝ := 3

/-- The number of hours John took for break -/
def john_break : ℝ := 1

/-- The total dancing time of both John and James (excluding John's break) -/
def total_dance_time : ℝ := 20

/-- The number of hours John danced after the break -/
def john_post_break : ℝ := 5

theorem john_post_break_time : 
  john_post_break = 
    (total_dance_time - john_pre_break - 
      (john_pre_break + john_break + john_post_break + 
        (1/3) * (john_pre_break + john_break + john_post_break))) / 
    (7/3) := by sorry

end NUMINAMATH_CALUDE_john_post_break_time_l2936_293675


namespace NUMINAMATH_CALUDE_percentage_difference_l2936_293636

theorem percentage_difference (x : ℝ) : 
  (60 / 100 * 50 = 30) →
  (30 = x / 100 * 30 + 17.4) →
  x = 42 := by
sorry

end NUMINAMATH_CALUDE_percentage_difference_l2936_293636


namespace NUMINAMATH_CALUDE_middle_number_proof_l2936_293631

theorem middle_number_proof (a b c d e : ℕ) : 
  ({a, b, c, d, e} : Finset ℕ) = {7, 8, 9, 10, 11} →
  a + b + c = 26 →
  c + d + e = 30 →
  c = 11 := by
  sorry

end NUMINAMATH_CALUDE_middle_number_proof_l2936_293631


namespace NUMINAMATH_CALUDE_remaining_dimes_l2936_293696

def initial_dimes : ℕ := 5
def spent_dimes : ℕ := 2

theorem remaining_dimes : initial_dimes - spent_dimes = 3 := by
  sorry

end NUMINAMATH_CALUDE_remaining_dimes_l2936_293696


namespace NUMINAMATH_CALUDE_g_at_negative_two_l2936_293664

/-- The function g is defined as g(x) = 2x^2 - 3x + 1 for all real x. -/
def g (x : ℝ) : ℝ := 2 * x^2 - 3 * x + 1

/-- Theorem: The value of g(-2) is 15. -/
theorem g_at_negative_two : g (-2) = 15 := by
  sorry

end NUMINAMATH_CALUDE_g_at_negative_two_l2936_293664


namespace NUMINAMATH_CALUDE_sum_of_divisors_231_eq_384_l2936_293615

/-- The sum of the positive whole number divisors of 231 -/
def sum_of_divisors_231 : ℕ := sorry

/-- Theorem stating that the sum of the positive whole number divisors of 231 is 384 -/
theorem sum_of_divisors_231_eq_384 : sum_of_divisors_231 = 384 := by sorry

end NUMINAMATH_CALUDE_sum_of_divisors_231_eq_384_l2936_293615


namespace NUMINAMATH_CALUDE_sector_area_l2936_293684

/-- Given a circular sector with central angle 2 radians and arc length 4, the area of the sector is 4. -/
theorem sector_area (θ : Real) (l : Real) (h1 : θ = 2) (h2 : l = 4) :
  (1/2) * (l/θ)^2 * θ = 4 := by sorry

end NUMINAMATH_CALUDE_sector_area_l2936_293684


namespace NUMINAMATH_CALUDE_class_size_l2936_293639

theorem class_size (n : ℕ) (h1 : n > 0) :
  (∃ (x : ℕ), x > 0 ∧ x = 6 + 7 - 1) →
  3 * n = 33 :=
by
  sorry

end NUMINAMATH_CALUDE_class_size_l2936_293639


namespace NUMINAMATH_CALUDE_min_value_expression_min_value_achievable_l2936_293622

theorem min_value_expression (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0)
  (h_sum : x + y + z = 3) (h_z : z = (x + y) / 2) :
  1 / (x + y) + 1 / (x + z) + 1 / (y + z) ≥ 3 / 2 :=
by sorry

theorem min_value_achievable :
  ∃ (x y z : ℝ), x > 0 ∧ y > 0 ∧ z > 0 ∧ x + y + z = 3 ∧ z = (x + y) / 2 ∧
  1 / (x + y) + 1 / (x + z) + 1 / (y + z) = 3 / 2 :=
by sorry

end NUMINAMATH_CALUDE_min_value_expression_min_value_achievable_l2936_293622


namespace NUMINAMATH_CALUDE_parabola_focus_distance_l2936_293685

-- Define the parabola
def parabola (x y : ℝ) : Prop := y^2 = 8*x

-- Define the focus of the parabola
def focus : ℝ × ℝ := (2, 0)

-- Define a point on the parabola
def point_on_parabola (M : ℝ × ℝ) : Prop :=
  parabola M.1 M.2

-- Define the y-axis intersection point
def y_axis_intersection (N : ℝ × ℝ) : Prop :=
  N.1 = 0

-- Define the midpoint condition
def is_midpoint (F M N : ℝ × ℝ) : Prop :=
  M.1 = (F.1 + N.1) / 2 ∧ M.2 = (F.2 + N.2) / 2

-- Main theorem
theorem parabola_focus_distance (M N : ℝ × ℝ) :
  point_on_parabola M →
  y_axis_intersection N →
  is_midpoint focus M N →
  (focus.1 - N.1)^2 + (focus.2 - N.2)^2 = 36 := by
  sorry

end NUMINAMATH_CALUDE_parabola_focus_distance_l2936_293685


namespace NUMINAMATH_CALUDE_two_numbers_difference_l2936_293623

theorem two_numbers_difference (x y : ℝ) (h1 : x > y) (h2 : x + y = 30) (h3 : x * y = 200) :
  x - y = 10 :=
by sorry

end NUMINAMATH_CALUDE_two_numbers_difference_l2936_293623


namespace NUMINAMATH_CALUDE_P_in_fourth_quadrant_iff_m_gt_two_l2936_293640

/-- A point P in the Cartesian plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- Definition of the fourth quadrant -/
def in_fourth_quadrant (p : Point) : Prop :=
  p.x > 0 ∧ p.y < 0

/-- The point P with coordinates (m, 2-m) -/
def P (m : ℝ) : Point :=
  ⟨m, 2 - m⟩

/-- Theorem stating that for P(m, 2-m) to be in the fourth quadrant, m > 2 -/
theorem P_in_fourth_quadrant_iff_m_gt_two (m : ℝ) :
  in_fourth_quadrant (P m) ↔ m > 2 := by
  sorry


end NUMINAMATH_CALUDE_P_in_fourth_quadrant_iff_m_gt_two_l2936_293640


namespace NUMINAMATH_CALUDE_bianca_candy_count_l2936_293665

def candy_problem (eaten : ℕ) (pieces_per_pile : ℕ) (num_piles : ℕ) : ℕ :=
  eaten + pieces_per_pile * num_piles

theorem bianca_candy_count : candy_problem 12 5 4 = 32 := by
  sorry

end NUMINAMATH_CALUDE_bianca_candy_count_l2936_293665


namespace NUMINAMATH_CALUDE_monotonic_increase_interval_l2936_293656

-- Define the function
def f (x : ℝ) := -x^2

-- State the theorem
theorem monotonic_increase_interval (a b : ℝ) :
  (∀ x y, x < y → x ∈ Set.Iio 0 → y ∈ Set.Iio 0 → f x < f y) ∧
  (∀ x, x ∈ Set.Iic 0 → f x ≤ f 0) ∧
  (∀ x, x > 0 → f x < f 0) :=
sorry

end NUMINAMATH_CALUDE_monotonic_increase_interval_l2936_293656


namespace NUMINAMATH_CALUDE_binomial_unique_parameters_l2936_293605

/-- A random variable following a binomial distribution -/
structure BinomialRV where
  n : ℕ
  p : ℝ
  hp : 0 ≤ p ∧ p ≤ 1

/-- The expectation of a binomial random variable -/
def expectation (X : BinomialRV) : ℝ := X.n * X.p

/-- The variance of a binomial random variable -/
def variance (X : BinomialRV) : ℝ := X.n * X.p * (1 - X.p)

/-- Theorem: For a binomial random variable X with E(X) = 1.6 and D(X) = 1.28, n = 8 and p = 0.2 -/
theorem binomial_unique_parameters :
  ∀ X : BinomialRV,
  expectation X = 1.6 →
  variance X = 1.28 →
  X.n = 8 ∧ X.p = 0.2 := by
  sorry

end NUMINAMATH_CALUDE_binomial_unique_parameters_l2936_293605


namespace NUMINAMATH_CALUDE_money_division_l2936_293681

theorem money_division (a b c : ℝ) : 
  a = (1/3) * (b + c) →
  b = (2/7) * (a + c) →
  a = b + 15 →
  a + b + c = 540 :=
by
  sorry

end NUMINAMATH_CALUDE_money_division_l2936_293681


namespace NUMINAMATH_CALUDE_equal_sets_implies_sum_l2936_293650

-- Define sets A and B
def A (a b : ℝ) : Set ℝ := {a, b/a, 1}
def B (a b : ℝ) : Set ℝ := {a^2, a+b, 0}

-- Theorem statement
theorem equal_sets_implies_sum (a b : ℝ) (h : A a b = B a b) :
  a^2013 + b^2014 = -1 :=
sorry

end NUMINAMATH_CALUDE_equal_sets_implies_sum_l2936_293650


namespace NUMINAMATH_CALUDE_arithmetic_sequence_property_l2936_293666

/-- An arithmetic sequence -/
def arithmetic_sequence (a : ℕ → ℝ) (d : ℝ) :=
  ∀ n, a (n + 1) = a n + d

/-- The proposition to be proved -/
theorem arithmetic_sequence_property
  (a : ℕ → ℝ) (d : ℝ) (h_arith : arithmetic_sequence a d) (h_nonzero : ∃ n, a n ≠ 0)
  (h_eq : 2 * a 4 - (a 7)^2 + 2 * a 10 = 0) :
  a 7 = 4 * d :=
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_property_l2936_293666


namespace NUMINAMATH_CALUDE_fifth_triple_is_pythagorean_l2936_293680

/-- A Pythagorean triple is a tuple of three positive integers (a, b, c) such that a² + b² = c² -/
def is_pythagorean_triple (a b c : ℕ) : Prop :=
  a > 0 ∧ b > 0 ∧ c > 0 ∧ a * a + b * b = c * c

/-- The fifth group in the sequence of Pythagorean triples -/
def fifth_pythagorean_triple : ℕ × ℕ × ℕ := (11, 60, 61)

theorem fifth_triple_is_pythagorean :
  let (a, b, c) := fifth_pythagorean_triple
  is_pythagorean_triple a b c :=
by sorry

end NUMINAMATH_CALUDE_fifth_triple_is_pythagorean_l2936_293680


namespace NUMINAMATH_CALUDE_percentage_difference_l2936_293600

theorem percentage_difference (x y : ℝ) (h : x = 7 * y) :
  (1 - y / x) * 100 = (1 - 1 / 7) * 100 := by
  sorry

end NUMINAMATH_CALUDE_percentage_difference_l2936_293600


namespace NUMINAMATH_CALUDE_intersection_value_l2936_293601

theorem intersection_value (k : ℝ) (x₁ x₂ y₁ y₂ : ℝ) :
  y₁ = k * x₁ ∧ y₁ = 1 / x₁ ∧
  y₂ = k * x₂ ∧ y₂ = 1 / x₂ ∧
  x₁ ≠ 0 ∧ x₂ ≠ 0 ∧ x₁ ≠ x₂ →
  x₁ * y₂ + x₂ * y₁ = -2 :=
by sorry

end NUMINAMATH_CALUDE_intersection_value_l2936_293601


namespace NUMINAMATH_CALUDE_afternoon_sales_l2936_293689

/-- 
Given a salesman who sold pears in the morning and afternoon, 
this theorem proves that if he sold twice as much in the afternoon 
as in the morning, and 420 kilograms in total, then he sold 280 
kilograms in the afternoon.
-/
theorem afternoon_sales 
  (morning_sales : ℕ) 
  (afternoon_sales : ℕ) 
  (h1 : afternoon_sales = 2 * morning_sales) 
  (h2 : morning_sales + afternoon_sales = 420) : 
  afternoon_sales = 280 := by
  sorry

end NUMINAMATH_CALUDE_afternoon_sales_l2936_293689


namespace NUMINAMATH_CALUDE_tangency_quad_area_theorem_l2936_293646

/-- An isosceles trapezoid circumscribed around a circle -/
structure CircumscribedTrapezoid where
  /-- The radius of the inscribed circle -/
  radius : ℝ
  /-- The area of the trapezoid -/
  trapezoidArea : ℝ
  /-- The area of the quadrilateral formed by tangency points -/
  tangencyQuadArea : ℝ
  /-- Assumption that the trapezoid is circumscribed around the circle -/
  isCircumscribed : Prop
  /-- Assumption that the trapezoid is isosceles -/
  isIsosceles : Prop

/-- Theorem stating the relationship between the areas -/
theorem tangency_quad_area_theorem (t : CircumscribedTrapezoid)
  (h1 : t.radius = 1)
  (h2 : t.trapezoidArea = 5)
  : t.tangencyQuadArea = 1.6 := by
  sorry

end NUMINAMATH_CALUDE_tangency_quad_area_theorem_l2936_293646


namespace NUMINAMATH_CALUDE_existence_of_irrational_term_l2936_293632

theorem existence_of_irrational_term (a : ℕ → ℝ) 
  (h_pos : ∀ n, a n > 0)
  (h_rec : ∀ n, (a (n + 1))^2 = a n + 1) :
  ∃ n, Irrational (a n) :=
sorry

end NUMINAMATH_CALUDE_existence_of_irrational_term_l2936_293632


namespace NUMINAMATH_CALUDE_wilson_oldest_child_age_wilson_oldest_child_age_proof_l2936_293669

/-- The age of the oldest Wilson child given the average age and the ages of the two younger children -/
theorem wilson_oldest_child_age 
  (average_age : ℝ) 
  (younger_child1_age : ℕ) 
  (younger_child2_age : ℕ) 
  (h1 : average_age = 8) 
  (h2 : younger_child1_age = 5) 
  (h3 : younger_child2_age = 8) : 
  ℕ := 
  11

theorem wilson_oldest_child_age_proof :
  let oldest_child_age := wilson_oldest_child_age 8 5 8 rfl rfl rfl
  (8 : ℝ) = (5 + 8 + oldest_child_age) / 3 := by
  sorry

end NUMINAMATH_CALUDE_wilson_oldest_child_age_wilson_oldest_child_age_proof_l2936_293669


namespace NUMINAMATH_CALUDE_line_translation_l2936_293626

theorem line_translation (x : ℝ) : 
  let original_line := λ x : ℝ => x / 3
  let translated_line := λ x : ℝ => (x + 5) / 3
  translated_line x - original_line x = 5 / 3 := by
  sorry

end NUMINAMATH_CALUDE_line_translation_l2936_293626


namespace NUMINAMATH_CALUDE_tea_set_combinations_l2936_293672

/-- The number of different cups available -/
def num_cups : ℕ := 5

/-- The number of different saucers available -/
def num_saucers : ℕ := 3

/-- The number of different teaspoons available -/
def num_spoons : ℕ := 4

/-- The number of ways to choose a cup and a saucer -/
def ways_cup_saucer : ℕ := num_cups * num_saucers

/-- The number of ways to choose a cup, a saucer, and a spoon -/
def ways_cup_saucer_spoon : ℕ := ways_cup_saucer * num_spoons

/-- The number of ways to choose two items of different types -/
def ways_two_different : ℕ := num_cups * num_saucers + num_cups * num_spoons + num_saucers * num_spoons

theorem tea_set_combinations :
  ways_cup_saucer = 15 ∧
  ways_cup_saucer_spoon = 60 ∧
  ways_two_different = 47 := by
  sorry

end NUMINAMATH_CALUDE_tea_set_combinations_l2936_293672


namespace NUMINAMATH_CALUDE_ten_hash_four_l2936_293643

/-- Operation # defined on real numbers -/
noncomputable def hash (r s : ℝ) : ℝ :=
  sorry

/-- Properties of the hash operation -/
axiom hash_zero (r : ℝ) : hash r 0 = r
axiom hash_comm (r s : ℝ) : hash r s = hash s r
axiom hash_succ (r s : ℝ) : hash (r + 1) s = hash r s + s + 2

/-- Theorem stating that 10 # 4 = 58 -/
theorem ten_hash_four : hash 10 4 = 58 := by
  sorry

end NUMINAMATH_CALUDE_ten_hash_four_l2936_293643
