import Mathlib

namespace specific_trapezoid_diagonal_l3078_307814

/-- An isosceles trapezoid with given dimensions -/
structure IsoscelesTrapezoid where
  bottom_base : ℝ
  top_base : ℝ
  side : ℝ

/-- The diagonal of an isosceles trapezoid -/
def diagonal (t : IsoscelesTrapezoid) : ℝ :=
  sorry

/-- Theorem: The diagonal of the specific isosceles trapezoid is 12√3 -/
theorem specific_trapezoid_diagonal :
  let t : IsoscelesTrapezoid := {
    bottom_base := 24,
    top_base := 12,
    side := 12
  }
  diagonal t = 12 * Real.sqrt 3 := by
  sorry

end specific_trapezoid_diagonal_l3078_307814


namespace equation_solutions_l3078_307872

theorem equation_solutions : 
  ∃ x₁ x₂ : ℝ, x₁ = 3 ∧ x₂ = 2/3 ∧ 
  (∀ x : ℝ, 2*x - 6 = 3*x*(x - 3) ↔ (x = x₁ ∨ x = x₂)) := by
sorry

end equation_solutions_l3078_307872


namespace french_speaking_percentage_l3078_307881

theorem french_speaking_percentage (total : ℕ) (french_and_english : ℕ) (french_only : ℕ) 
  (h1 : total = 200)
  (h2 : french_and_english = 10)
  (h3 : french_only = 40) :
  (total - (french_and_english + french_only)) / total * 100 = 75 :=
by sorry

end french_speaking_percentage_l3078_307881


namespace unique_number_l3078_307887

def is_valid_number (n : ℕ) : Prop :=
  (n ≥ 1000) ∧ (n < 10000) ∧
  (n % 10 = (n / 100) % 10) ∧
  (n - (n % 10 * 1000 + (n / 10) % 10 * 100 + (n / 100) % 10 * 10 + n / 1000) = 7812)

theorem unique_number : ∃! n : ℕ, is_valid_number n ∧ n = 1979 :=
  sorry

end unique_number_l3078_307887


namespace gym_weights_problem_l3078_307875

/-- Given the conditions of the gym weights problem, prove that each green weight is 3 pounds. -/
theorem gym_weights_problem (blue_weight : ℕ) (num_blue : ℕ) (num_green : ℕ) (bar_weight : ℕ) (total_weight : ℕ) :
  blue_weight = 2 →
  num_blue = 4 →
  num_green = 5 →
  bar_weight = 2 →
  total_weight = 25 →
  ∃ (green_weight : ℕ), green_weight = 3 ∧ total_weight = blue_weight * num_blue + green_weight * num_green + bar_weight :=
by sorry

end gym_weights_problem_l3078_307875


namespace largest_prime_divisor_of_sum_of_squares_l3078_307804

theorem largest_prime_divisor_of_sum_of_squares : 
  ∃ p : ℕ, Prime p ∧ p ∣ (36^2 + 49^2) ∧ ∀ q : ℕ, Prime q → q ∣ (36^2 + 49^2) → q ≤ p :=
by
  -- The proof goes here
  sorry

end largest_prime_divisor_of_sum_of_squares_l3078_307804


namespace degree_of_example_monomial_l3078_307830

/-- Represents a monomial with coefficient and exponents for x and y -/
structure Monomial where
  coeff : ℤ
  x_exp : ℕ
  y_exp : ℕ

/-- Calculates the degree of a monomial -/
def degree (m : Monomial) : ℕ := m.x_exp + m.y_exp

/-- The specific monomial -5x^2y -/
def example_monomial : Monomial := ⟨-5, 2, 1⟩

theorem degree_of_example_monomial :
  degree example_monomial = 3 := by sorry

end degree_of_example_monomial_l3078_307830


namespace arctan_cos_solution_l3078_307816

theorem arctan_cos_solution (x : Real) :
  -π ≤ x ∧ x ≤ π →
  Real.arctan (Real.cos x) = x / 3 →
  x = Real.arccos (Real.sqrt ((Real.sqrt 5 - 1) / 2)) ∨
  x = -Real.arccos (Real.sqrt ((Real.sqrt 5 - 1) / 2)) := by
  sorry

end arctan_cos_solution_l3078_307816


namespace function_intersects_x_axis_l3078_307811

theorem function_intersects_x_axis (a : ℝ) : 
  (∀ m : ℝ, ∃ x : ℝ, m * x^2 + x - m - a = 0) → 
  a ∈ Set.Icc (-1 : ℝ) 1 :=
by sorry

end function_intersects_x_axis_l3078_307811


namespace worker_task_completion_time_l3078_307898

/-- Given two workers who can complete a task together in 35 days,
    and one of them can complete the task alone in 84 days,
    prove that the other worker can complete the task alone in 70 days. -/
theorem worker_task_completion_time 
  (total_time : ℝ) 
  (worker1_time : ℝ) 
  (worker2_time : ℝ) 
  (h1 : total_time = 35) 
  (h2 : worker1_time = 84) 
  (h3 : 1 / total_time = 1 / worker1_time + 1 / worker2_time) : 
  worker2_time = 70 := by
  sorry

end worker_task_completion_time_l3078_307898


namespace skew_edges_count_l3078_307845

/-- Represents a cube in 3D space -/
structure Cube where
  -- Add necessary fields here
  mk :: -- Constructor

/-- Represents a line in 3D space -/
structure Line where
  -- Add necessary fields here
  mk :: -- Constructor

/-- Checks if a line lies on a face of the cube -/
def lineOnFace (c : Cube) (l : Line) : Prop :=
  sorry

/-- Counts the number of edges not in the same plane as the given line -/
def countSkewEdges (c : Cube) (l : Line) : ℕ :=
  sorry

/-- Main theorem: The number of skew edges is either 4, 6, 7, or 8 -/
theorem skew_edges_count (c : Cube) (l : Line) 
  (h : lineOnFace c l) : 
  (countSkewEdges c l = 4) ∨ 
  (countSkewEdges c l = 6) ∨ 
  (countSkewEdges c l = 7) ∨ 
  (countSkewEdges c l = 8) :=
sorry

end skew_edges_count_l3078_307845


namespace friday_dinner_customers_l3078_307810

/-- The number of customers during breakfast on Friday -/
def breakfast_customers : ℕ := 73

/-- The number of customers during lunch on Friday -/
def lunch_customers : ℕ := 127

/-- The predicted number of customers for Saturday -/
def saturday_prediction : ℕ := 574

/-- The number of customers during dinner on Friday -/
def dinner_customers : ℕ := 87

theorem friday_dinner_customers : 
  dinner_customers = saturday_prediction / 2 - breakfast_customers - lunch_customers :=
by sorry

end friday_dinner_customers_l3078_307810


namespace find_k_l3078_307859

theorem find_k : ∃ k : ℝ, ∀ x : ℝ, -x^2 - (k + 7)*x - 8 = -(x - 2)*(x - 4) → k = -13 := by
  sorry

end find_k_l3078_307859


namespace lines_skew_iff_a_neq_4_l3078_307843

/-- Two lines in 3D space -/
structure Line3D where
  point : ℝ × ℝ × ℝ
  direction : ℝ × ℝ × ℝ

/-- Check if two lines are skew -/
def are_skew (l1 l2 : Line3D) : Prop :=
  ¬ (∃ (t u : ℝ), l1.point + t • l1.direction = l2.point + u • l2.direction)

/-- The main theorem -/
theorem lines_skew_iff_a_neq_4 (a : ℝ) :
  let l1 : Line3D := ⟨(2, 3, a), (3, 4, 5)⟩
  let l2 : Line3D := ⟨(5, 2, 1), (6, 3, 2)⟩
  are_skew l1 l2 ↔ a ≠ 4 := by
  sorry


end lines_skew_iff_a_neq_4_l3078_307843


namespace smallest_n_with_common_divisors_l3078_307832

def M : ℕ := 30030

theorem smallest_n_with_common_divisors (n : ℕ) : n = 9440 ↔ 
  (∀ k : ℕ, k ≤ 20 → ∃ d : ℕ, d > 1 ∧ d ∣ (n + k) ∧ d ∣ M) ∧
  (∀ m : ℕ, m < n → ∃ k : ℕ, k ≤ 20 ∧ ∀ d : ℕ, d > 1 → d ∣ (m + k) → ¬(d ∣ M)) :=
sorry

end smallest_n_with_common_divisors_l3078_307832


namespace function_equality_l3078_307849

/-- Given f(x) = 3x - 5, prove that 2 * [f(1)] - 16 = f(7) -/
theorem function_equality (f : ℝ → ℝ) (h : ∀ x, f x = 3 * x - 5) :
  2 * (f 1) - 16 = f 7 := by sorry

end function_equality_l3078_307849


namespace radius_ef_is_sqrt_136_l3078_307895

/-- Triangle DEF with semicircles on its sides -/
structure TriangleWithSemicircles where
  /-- Length of side DE -/
  de : ℝ
  /-- Length of side DF -/
  df : ℝ
  /-- Length of side EF -/
  ef : ℝ
  /-- DEF is a right triangle -/
  right_angle : de^2 + df^2 = ef^2
  /-- Area of semicircle on DE -/
  area_de : (1/2) * Real.pi * (de/2)^2 = 18 * Real.pi
  /-- Arc length of semicircle on DF -/
  arc_df : Real.pi * (df/2) = 10 * Real.pi

/-- The radius of the semicircle on EF is √136 -/
theorem radius_ef_is_sqrt_136 (t : TriangleWithSemicircles) : ef/2 = Real.sqrt 136 := by
  sorry

end radius_ef_is_sqrt_136_l3078_307895


namespace multiply_power_result_l3078_307890

theorem multiply_power_result : 112 * (5^4) = 70000 := by
  sorry

end multiply_power_result_l3078_307890


namespace cos_pi_fourth_plus_alpha_l3078_307844

theorem cos_pi_fourth_plus_alpha (α : ℝ) 
  (h : Real.sin (π / 4 - α) = 1 / 2) : 
  Real.cos (π / 4 + α) = 1 / 2 := by
  sorry

end cos_pi_fourth_plus_alpha_l3078_307844


namespace square_minus_product_equals_one_l3078_307856

theorem square_minus_product_equals_one (a : ℝ) (h : a = -4) : a^2 - (a+1)*(a-1) = 1 := by
  sorry

end square_minus_product_equals_one_l3078_307856


namespace volume_removed_percentage_l3078_307852

/-- Represents the dimensions of a rectangular prism -/
structure PrismDimensions where
  length : ℝ
  width : ℝ
  height : ℝ

/-- Calculates the volume of a rectangular prism -/
def prismVolume (d : PrismDimensions) : ℝ :=
  d.length * d.width * d.height

/-- Represents the side length of the small cube removed from each corner -/
def smallCubeSide : ℝ := 4

/-- Calculates the volume of the small cube -/
def smallCubeVolume : ℝ :=
  smallCubeSide ^ 3

/-- The number of corners in a rectangular prism -/
def numCorners : ℕ := 8

/-- Theorem: The percentage of volume removed from the rectangular prism -/
theorem volume_removed_percentage
  (d : PrismDimensions)
  (h1 : d.length = 20)
  (h2 : d.width = 14)
  (h3 : d.height = 12) :
  (numCorners * smallCubeVolume) / (prismVolume d) * 100 = 512 / 3360 * 100 := by
  sorry

end volume_removed_percentage_l3078_307852


namespace cherries_refund_l3078_307864

def grapes_cost : ℚ := 12.08
def total_spent : ℚ := 2.23

theorem cherries_refund :
  grapes_cost - total_spent = 9.85 := by sorry

end cherries_refund_l3078_307864


namespace exercise_book_cost_l3078_307824

/-- Proves that the total cost of buying 'a' exercise books at 0.8 yuan each is 0.8a yuan -/
theorem exercise_book_cost (a : ℝ) : 
  let cost_per_book : ℝ := 0.8
  let num_books : ℝ := a
  let total_cost : ℝ := cost_per_book * num_books
  total_cost = 0.8 * a := by sorry

end exercise_book_cost_l3078_307824


namespace determinant_transformation_l3078_307863

theorem determinant_transformation (x y z w : ℝ) :
  Matrix.det ![![x, y], ![z, w]] = 6 →
  Matrix.det ![![x, 5*x + 4*y], ![z, 5*z + 4*w]] = 24 := by
  sorry

end determinant_transformation_l3078_307863


namespace pure_imaginary_condition_l3078_307823

theorem pure_imaginary_condition (z : ℂ) (a b : ℝ) : 
  z = Complex.mk a b → z.re = 0 → a = 0 ∧ b ≠ 0 := by
  sorry

end pure_imaginary_condition_l3078_307823


namespace phone_number_revenue_l3078_307869

theorem phone_number_revenue (X Y : ℕ) : 
  125 * X - 64 * Y = 5 ∧ X < 250 ∧ Y < 250 → 
  (X = 41 ∧ Y = 80) ∨ (X = 105 ∧ Y = 205) :=
sorry

end phone_number_revenue_l3078_307869


namespace tea_mixture_price_l3078_307853

theorem tea_mixture_price (price1 price2 mixture_price : ℚ) (ratio1 ratio2 ratio3 : ℚ) :
  price1 = 126 →
  price2 = 135 →
  mixture_price = 153 →
  ratio1 = 1 →
  ratio2 = 1 →
  ratio3 = 2 →
  ∃ price3 : ℚ,
    price3 = 175.5 ∧
    (ratio1 * price1 + ratio2 * price2 + ratio3 * price3) / (ratio1 + ratio2 + ratio3) = mixture_price :=
by sorry

end tea_mixture_price_l3078_307853


namespace solution_pairs_l3078_307855

theorem solution_pairs (x y : ℝ) (h1 : x + y = 1) (h2 : x^3 + y^3 = 19) :
  (x = 3 ∧ y = -2) ∨ (x = -2 ∧ y = 3) := by
  sorry

end solution_pairs_l3078_307855


namespace abc_inequality_l3078_307892

theorem abc_inequality (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  a * b * c * (a + b + c) ≤ a^3 * b + b^3 * c + c^3 * a := by
  sorry

end abc_inequality_l3078_307892


namespace cosine_of_angle_l3078_307802

/-- Given two vectors a and b in ℝ², prove that the cosine of the angle between them is -63/65,
    when a + b = (2, -8) and a - b = (-8, 16). -/
theorem cosine_of_angle (a b : ℝ × ℝ) 
    (sum_eq : a + b = (2, -8)) 
    (diff_eq : a - b = (-8, 16)) : 
  (a.1 * b.1 + a.2 * b.2) / (Real.sqrt (a.1^2 + a.2^2) * Real.sqrt (b.1^2 + b.2^2)) = -63/65 := by
  sorry

end cosine_of_angle_l3078_307802


namespace queen_diamond_probability_l3078_307870

/-- Represents a standard deck of 52 playing cards -/
structure Deck :=
  (cards : Finset (Nat × Nat))
  (size : cards.card = 52)
  (valid : ∀ c ∈ cards, c.1 ∈ Finset.range 13 ∧ c.2 ∈ Finset.range 4)

/-- Represents the event of drawing a Queen as the first card -/
def isFirstCardQueen (d : Deck) : Finset (Nat × Nat) :=
  d.cards.filter (λ c => c.1 = 11)

/-- Represents the event of drawing a diamond as the second card -/
def isSecondCardDiamond (d : Deck) : Finset (Nat × Nat) :=
  d.cards.filter (λ c => c.2 = 1)

/-- The main theorem stating the probability of drawing a Queen first and a diamond second -/
theorem queen_diamond_probability (d : Deck) :
  (isFirstCardQueen d).card / d.cards.card *
  (isSecondCardDiamond d).card / (d.cards.card - 1) = 18 / 221 :=
sorry

end queen_diamond_probability_l3078_307870


namespace bernoulli_inequality_l3078_307888

theorem bernoulli_inequality (x r : ℝ) (hx : x > 0) (hr : r > 1) :
  (1 + x)^r > 1 + r * x := by
  sorry

end bernoulli_inequality_l3078_307888


namespace hyperbola_eccentricity_l3078_307899

/-- Given a parabola and a hyperbola with an intersection point on the hyperbola's asymptote,
    prove that the hyperbola's eccentricity is √5 under specific conditions. -/
theorem hyperbola_eccentricity (p a b : ℝ) (h1 : p > 0) (h2 : a > 0) (h3 : b > 0) :
  let C₁ := {(x, y) : ℝ × ℝ | y^2 = 2*p*x}
  let C₂ := {(x, y) : ℝ × ℝ | x^2/a^2 - y^2/b^2 = 1}
  let asymptote := {(x, y) : ℝ × ℝ | y = (b/a)*x}
  ∃ A : ℝ × ℝ, A ∈ C₁ ∧ A ∈ C₂ ∧ A ∈ asymptote ∧ 
    (let (x, y) := A
     x - p/2 = p) →
  (Real.sqrt ((a^2 + b^2) / a^2) : ℝ) = Real.sqrt 5 :=
sorry

end hyperbola_eccentricity_l3078_307899


namespace smallest_solution_quadratic_l3078_307858

theorem smallest_solution_quadratic (x : ℝ) :
  (6 * x^2 - 37 * x + 48 = 0) → (x ≥ 13/6) :=
by sorry

end smallest_solution_quadratic_l3078_307858


namespace problem_statement_l3078_307894

theorem problem_statement (a b : ℝ) (h1 : a - b = 1) (h2 : a * b = -2) :
  (a + 1) * (b - 1) = -4 := by
  sorry

end problem_statement_l3078_307894


namespace project_hours_difference_l3078_307889

theorem project_hours_difference (total_hours : ℕ) 
  (h1 : total_hours = 216) 
  (kate_hours : ℕ) 
  (pat_hours : ℕ) 
  (mark_hours : ℕ) 
  (h2 : pat_hours = 2 * kate_hours) 
  (h3 : pat_hours * 3 = mark_hours) 
  (h4 : kate_hours + pat_hours + mark_hours = total_hours) : 
  mark_hours - kate_hours = 120 := by
sorry

end project_hours_difference_l3078_307889


namespace percentage_calculation_approximation_l3078_307839

theorem percentage_calculation_approximation : 
  ∃ (ε : ℝ), ε > 0 ∧ ε < 0.01 ∧ 
  abs ((0.47 * 1442 - 0.36 * 1412) + 63 - 232.42) < ε := by
  sorry

end percentage_calculation_approximation_l3078_307839


namespace ratio_problem_l3078_307886

theorem ratio_problem (a b : ℝ) (h : a / b = 3 / 2) : (a + b) / a = 5 / 3 := by
  sorry

end ratio_problem_l3078_307886


namespace quadratic_roots_imply_m_l3078_307847

theorem quadratic_roots_imply_m (m : ℝ) : 
  (∃ x : ℂ, 5 * x^2 - 2 * x + m = 0 ∧ x = (2 + Complex.I * Real.sqrt 78) / 10) ∧
  (∃ x : ℂ, 5 * x^2 - 2 * x + m = 0 ∧ x = (2 - Complex.I * Real.sqrt 78) / 10) →
  m = 41 / 10 := by
sorry

end quadratic_roots_imply_m_l3078_307847


namespace impossible_equal_distribution_l3078_307820

/-- Represents the state of coins on the hexagon vertices -/
def HexagonState := Fin 6 → ℕ

/-- The initial state of the hexagon -/
def initial_state : HexagonState := fun i => if i = 0 then 1 else 0

/-- Represents a valid move in the game -/
def valid_move (s1 s2 : HexagonState) : Prop :=
  ∃ (i j : Fin 6) (n : ℕ), 
    (j = i + 1 ∨ j = i - 1 ∨ (i = 5 ∧ j = 0) ∨ (i = 0 ∧ j = 5)) ∧
    s2 i + 6 * n = s1 i ∧
    s2 j = s1 j + 6 * n ∧
    ∀ k, k ≠ i ∧ k ≠ j → s2 k = s1 k

/-- A sequence of valid moves -/
def valid_sequence (s : ℕ → HexagonState) : Prop :=
  s 0 = initial_state ∧ ∀ n, valid_move (s n) (s (n + 1))

/-- The theorem to be proved -/
theorem impossible_equal_distribution :
  ¬∃ (s : ℕ → HexagonState) (n : ℕ), 
    valid_sequence s ∧ 
    (∀ (i j : Fin 6), s n i = s n j) :=
sorry

end impossible_equal_distribution_l3078_307820


namespace defense_attorney_implication_l3078_307854

-- Define propositions
variable (P : Prop) -- P represents "the defendant is guilty"
variable (Q : Prop) -- Q represents "the defendant had an accomplice"

-- Theorem statement
theorem defense_attorney_implication : ¬(P → Q) → (P ∧ ¬Q) := by
  sorry

end defense_attorney_implication_l3078_307854


namespace arithmetic_sequence_sum_l3078_307806

/-- An arithmetic sequence with a positive common difference -/
structure ArithmeticSequence where
  a : ℕ → ℝ
  d : ℝ
  h_arithmetic : ∀ n, a (n + 1) = a n + d
  h_positive : d > 0

/-- The theorem statement -/
theorem arithmetic_sequence_sum (seq : ArithmeticSequence) 
  (h_sum : seq.a 1 + seq.a 2 + seq.a 3 = 15)
  (h_product : seq.a 1 * seq.a 2 * seq.a 3 = 45) :
  seq.a 2009 + seq.a 2010 + seq.a 2011 = 24111 :=
sorry

end arithmetic_sequence_sum_l3078_307806


namespace simplify_expression_l3078_307846

theorem simplify_expression (x : ℝ) : (3 * x^4)^2 * (2 * x^2)^3 = 72 * x^14 := by
  sorry

end simplify_expression_l3078_307846


namespace remainder_three_to_89_plus_5_mod_7_l3078_307893

theorem remainder_three_to_89_plus_5_mod_7 : (3^89 + 5) % 7 = 3 := by
  sorry

end remainder_three_to_89_plus_5_mod_7_l3078_307893


namespace diamonds_in_G10_num_diamonds_formula_num_diamonds_induction_l3078_307879

/-- Represents the number of diamonds in figure G_n -/
def num_diamonds (n : ℕ) : ℕ :=
  4 * n^2 + 5 * n - 8

theorem diamonds_in_G10 :
  num_diamonds 10 = 442 :=
by sorry

theorem num_diamonds_formula (n : ℕ) :
  n ≥ 1 →
  num_diamonds n =
    1 + -- initial diamond in G_1
    (4 * (n - 1) * n) + -- diamonds added to sides
    (8 * (n - 1)) -- diamonds added to corners
  :=
by sorry

theorem num_diamonds_induction (n : ℕ) :
  n ≥ 1 →
  num_diamonds n =
    (if n = 1 then 1
     else num_diamonds (n - 1) + 8 * (4 * (n - 1) + 1))
  :=
by sorry

end diamonds_in_G10_num_diamonds_formula_num_diamonds_induction_l3078_307879


namespace simple_interest_problem_l3078_307803

/-- Given a sum put at simple interest for 3 years, if increasing the interest
    rate by 1% results in Rs. 69 more interest, then the sum is Rs. 2300. -/
theorem simple_interest_problem (P : ℝ) (R : ℝ) : 
  (P * (R + 1) * 3 / 100 - P * R * 3 / 100 = 69) → P = 2300 := by
  sorry

end simple_interest_problem_l3078_307803


namespace puzzle_solution_l3078_307871

/-- Represents the animals in the puzzle -/
inductive Animal : Type
  | Cat | Chicken | Crab | Bear | Goat

/-- Represents the puzzle grid -/
def Grid := Animal → Nat

/-- Checks if the grid satisfies the sum conditions -/
def satisfies_sums (g : Grid) : Prop :=
  g Animal.Crab + g Animal.Crab + g Animal.Crab + g Animal.Crab + g Animal.Crab = 10 ∧
  g Animal.Goat + g Animal.Goat + g Animal.Crab + g Animal.Bear + g Animal.Bear = 16 ∧
  g Animal.Crab + g Animal.Chicken + g Animal.Chicken + g Animal.Goat + g Animal.Crab = 17 ∧
  g Animal.Cat + g Animal.Bear + g Animal.Goat + g Animal.Goat + g Animal.Crab = 13

/-- Checks if all animals have different values -/
def all_different (g : Grid) : Prop :=
  ∀ a b : Animal, a ≠ b → g a ≠ g b

/-- The main theorem stating the unique solution -/
theorem puzzle_solution :
  ∃! g : Grid, satisfies_sums g ∧ all_different g ∧
    g Animal.Cat = 1 ∧ g Animal.Chicken = 5 ∧ g Animal.Crab = 2 ∧
    g Animal.Bear = 4 ∧ g Animal.Goat = 3 :=
  sorry

end puzzle_solution_l3078_307871


namespace complex_sum_argument_l3078_307851

theorem complex_sum_argument : ∃ (r : ℝ), 
  Complex.exp (11 * Real.pi * Complex.I / 60) + 
  Complex.exp (23 * Real.pi * Complex.I / 60) + 
  Complex.exp (35 * Real.pi * Complex.I / 60) + 
  Complex.exp (47 * Real.pi * Complex.I / 60) + 
  Complex.exp (59 * Real.pi * Complex.I / 60) + 
  Complex.exp (Real.pi * Complex.I / 60) = 
  r * Complex.exp (7 * Real.pi * Complex.I / 12) :=
by sorry

end complex_sum_argument_l3078_307851


namespace interest_rate_calculation_l3078_307812

theorem interest_rate_calculation (principal_B principal_C time_B time_C total_interest : ℕ) 
  (h1 : principal_B = 5000)
  (h2 : principal_C = 3000)
  (h3 : time_B = 2)
  (h4 : time_C = 4)
  (h5 : total_interest = 2640) :
  let rate := total_interest * 100 / (principal_B * time_B + principal_C * time_C)
  rate = 12 := by sorry

end interest_rate_calculation_l3078_307812


namespace shortest_major_axis_ellipse_l3078_307861

-- Define the line l
def line_l (x y : ℝ) : Prop := x - y + 9 = 0

-- Define the ellipse C
def ellipse_C (x y θ : ℝ) : Prop := x = 2 * Real.sqrt 3 * Real.cos θ ∧ y = Real.sqrt 3 * Real.sin θ

-- Define the foci
def F₁ : ℝ × ℝ := (-3, 0)
def F₂ : ℝ × ℝ := (3, 0)

-- Define a point on the line l
def point_on_l (M : ℝ × ℝ) : Prop := line_l M.1 M.2

-- Define the ellipse equation
def ellipse_equation (x y : ℝ) : Prop := x^2 / 45 + y^2 / 36 = 1

-- Theorem statement
theorem shortest_major_axis_ellipse :
  ∀ (M : ℝ × ℝ), point_on_l M →
  ∀ (x y : ℝ), ellipse_equation x y →
  ∃ (a b : ℝ), a > b ∧ b > 0 ∧
    (x^2 / a^2 + y^2 / b^2 = 1) ∧
    (∀ (a' b' : ℝ), a' > b' ∧ b' > 0 →
      (x^2 / a'^2 + y^2 / b'^2 = 1) →
      point_on_l (x, y) →
      a ≤ a') :=
by sorry

end shortest_major_axis_ellipse_l3078_307861


namespace dining_bill_calculation_l3078_307818

/-- Proves that the original bill before tip was $139 given the problem conditions -/
theorem dining_bill_calculation (people : ℕ) (tip_percentage : ℚ) (individual_payment : ℚ) :
  people = 5 ∧ 
  tip_percentage = 1/10 ∧ 
  individual_payment = 3058/100 →
  ∃ (original_bill : ℚ),
    original_bill * (1 + tip_percentage) = people * individual_payment ∧
    original_bill = 139 :=
by sorry

end dining_bill_calculation_l3078_307818


namespace binomial_square_constant_l3078_307821

theorem binomial_square_constant (a : ℝ) : 
  (∃ b : ℝ, ∀ x : ℝ, 9*x^2 + 27*x + a = (3*x + b)^2) → a = 81/4 := by
  sorry

end binomial_square_constant_l3078_307821


namespace fathers_age_l3078_307874

/-- The father's age given the son's age ratio conditions -/
theorem fathers_age (man_age : ℝ) (father_age : ℝ) : 
  man_age = (2 / 5) * father_age ∧ 
  man_age + 14 = (1 / 2) * (father_age + 14) → 
  father_age = 70 := by
sorry

end fathers_age_l3078_307874


namespace equal_roots_quadratic_l3078_307868

/-- If the quadratic equation x^2 + x + m = 0 has two equal real roots,
    then m = 1/4 -/
theorem equal_roots_quadratic (m : ℝ) : 
  (∃ x : ℝ, x^2 + x + m = 0 ∧ 
   ∀ y : ℝ, y^2 + y + m = 0 → y = x) → 
  m = 1/4 :=
by sorry

end equal_roots_quadratic_l3078_307868


namespace complement_intersection_equality_l3078_307826

-- Define the sets P and Q
def P : Set ℝ := {x : ℝ | x ≥ 2}
def Q : Set ℝ := {x : ℝ | 1 < x ∧ x ≤ 2}

-- Define the open interval (1, 2)
def open_interval : Set ℝ := {x : ℝ | 1 < x ∧ x < 2}

-- State the theorem
theorem complement_intersection_equality :
  (Set.univ \ P) ∩ Q = open_interval :=
sorry

end complement_intersection_equality_l3078_307826


namespace larger_box_capacity_l3078_307835

/-- Represents a rectangular box with integer dimensions -/
structure Box where
  length : ℕ
  width : ℕ
  height : ℕ

/-- Calculates the volume of a box -/
def Box.volume (b : Box) : ℕ := b.length * b.width * b.height

/-- Represents the number of marbles a box can hold -/
def marbles_capacity (b : Box) (marbles : ℕ) : Prop :=
  b.volume = marbles

theorem larger_box_capacity 
  (kevin_box : Box)
  (kevin_marbles : ℕ)
  (laura_box : Box)
  (h1 : kevin_box.length = 3 ∧ kevin_box.width = 3 ∧ kevin_box.height = 8)
  (h2 : marbles_capacity kevin_box kevin_marbles)
  (h3 : kevin_marbles = 216)
  (h4 : laura_box.length = 3 * kevin_box.length ∧ 
        laura_box.width = 3 * kevin_box.width ∧ 
        laura_box.height = 3 * kevin_box.height) :
  marbles_capacity laura_box 5832 :=
sorry

end larger_box_capacity_l3078_307835


namespace election_winner_percentage_l3078_307860

theorem election_winner_percentage (total_votes : ℕ) (winner_votes : ℕ) (margin : ℕ) : 
  winner_votes = 930 →
  margin = 360 →
  total_votes = winner_votes + (winner_votes - margin) →
  (winner_votes : ℚ) / (total_votes : ℚ) = 62 / 100 := by
  sorry

end election_winner_percentage_l3078_307860


namespace cube_iff_greater_l3078_307865

theorem cube_iff_greater (a b : ℝ) : a > b ↔ a^3 > b^3 := by sorry

end cube_iff_greater_l3078_307865


namespace smallest_number_l3078_307809

theorem smallest_number (S : Set ℤ) (hS : S = {0, 1, -5, -1}) :
  ∃ m ∈ S, ∀ x ∈ S, m ≤ x ∧ m = -5 := by sorry

end smallest_number_l3078_307809


namespace elbertas_money_l3078_307867

/-- Given that Granny Smith has $45, Elberta has $4 more than Anjou, and Anjou has one-fourth as much as Granny Smith, prove that Elberta has $15.25. -/
theorem elbertas_money (granny_smith : ℝ) (elberta anjou : ℝ) 
  (h1 : granny_smith = 45)
  (h2 : elberta = anjou + 4)
  (h3 : anjou = granny_smith / 4) :
  elberta = 15.25 := by
  sorry

end elbertas_money_l3078_307867


namespace distance_sum_squares_l3078_307857

-- Define the lines l₁ and l₂
def l₁ (m : ℝ) (x y : ℝ) : Prop := x - m * y + 1 = 0
def l₂ (m : ℝ) (x y : ℝ) : Prop := m * x + y - m + 3 = 0

-- Define points A and B
def A : ℝ × ℝ := (-1, 0)
def B : ℝ × ℝ := (1, -3)

-- Theorem statement
theorem distance_sum_squares (m : ℝ) (P : ℝ × ℝ) :
  l₁ m P.1 P.2 ∧ l₂ m P.1 P.2 →
  l₁ m A.1 A.2 ∧ l₂ m B.1 B.2 →
  (P.1 - A.1)^2 + (P.2 - A.2)^2 + (P.1 - B.1)^2 + (P.2 - B.2)^2 = 13 :=
by sorry

end distance_sum_squares_l3078_307857


namespace equilateral_triangle_to_trapezoid_l3078_307828

/-- Represents a paper shape that can be folded -/
structure PaperShape where
  vertices : ℕ
  layers : ℕ

/-- Represents the process of folding a paper shape -/
def fold (initial : PaperShape) (final : PaperShape) : Prop :=
  ∃ (steps : ℕ), steps > 0 ∧ final.layers ≥ initial.layers

/-- An equilateral triangle -/
def equilateralTriangle : PaperShape :=
  { vertices := 3, layers := 1 }

/-- A trapezoid -/
def trapezoid : PaperShape :=
  { vertices := 4, layers := 3 }

theorem equilateral_triangle_to_trapezoid :
  fold equilateralTriangle trapezoid :=
sorry

#check equilateral_triangle_to_trapezoid

end equilateral_triangle_to_trapezoid_l3078_307828


namespace binomial_expansion_arithmetic_sequence_max_coefficient_terms_sqrt_inequality_l3078_307800

-- Part 1
theorem binomial_expansion_arithmetic_sequence (n : ℕ) :
  (∃ r : ℚ, r ≠ 0 ∧ 
    n.choose 0 + (1/4) * n.choose 2 = 2 * (1/2) * n.choose 1) →
  n = 8 :=
sorry

-- Part 2
theorem max_coefficient_terms (n : ℕ) (x : ℝ) :
  n = 8 →
  ∃ c : ℝ, c > 0 ∧
    (∀ k : ℕ, k ≤ n → 
      c * x^(5 : ℝ) ≥ (1/(2^k : ℝ)) * n.choose k * x^((n - k : ℝ)/2)) ∧
    (∀ k : ℕ, k ≤ n → 
      c * x^(7/2 : ℝ) ≥ (1/(2^k : ℝ)) * n.choose k * x^((n - k : ℝ)/2)) :=
sorry

-- Part 3
theorem sqrt_inequality (a : ℝ) :
  a > 1 →
  Real.sqrt (a + 1) - Real.sqrt a < Real.sqrt a - Real.sqrt (a - 1) :=
sorry

end binomial_expansion_arithmetic_sequence_max_coefficient_terms_sqrt_inequality_l3078_307800


namespace log_equality_l3078_307822

theorem log_equality (y : ℝ) (m : ℝ) : 
  (Real.log 5 / Real.log 8 = y) → 
  (Real.log 125 / Real.log 2 = m * y) → 
  m = 9 := by
sorry

end log_equality_l3078_307822


namespace max_area_inscribed_circle_l3078_307801

/-- A quadrilateral with given angles and perimeter -/
structure Quadrilateral where
  angles : Fin 4 → ℝ
  perimeter : ℝ
  sum_angles : (angles 0) + (angles 1) + (angles 2) + (angles 3) = 2 * Real.pi
  positive_perimeter : perimeter > 0

/-- The area of a quadrilateral -/
def area (q : Quadrilateral) : ℝ := sorry

/-- A predicate indicating whether a circle can be inscribed in the quadrilateral -/
def has_inscribed_circle (q : Quadrilateral) : Prop := sorry

/-- The theorem stating that the quadrilateral with an inscribed circle has the largest area -/
theorem max_area_inscribed_circle (q : Quadrilateral) :
  has_inscribed_circle q ↔ ∀ (q' : Quadrilateral), q'.angles = q.angles ∧ q'.perimeter = q.perimeter → area q ≥ area q' :=
sorry

end max_area_inscribed_circle_l3078_307801


namespace seven_digit_nondecreasing_integers_l3078_307817

theorem seven_digit_nondecreasing_integers (n : ℕ) (h : n = 7) :
  (Nat.choose (10 + n - 1) n) % 1000 = 440 := by
  sorry

end seven_digit_nondecreasing_integers_l3078_307817


namespace star_divided_by_square_equals_sixteen_l3078_307813

-- Define the symbols as real numbers
variable (triangle circle square star : ℝ)

-- State the conditions
axiom triangle_plus_triangle : triangle + triangle = star
axiom circle_equals_square_plus_square : circle = square + square
axiom triangle_equals_four_circles : triangle = circle + circle + circle + circle

-- State the theorem to be proved
theorem star_divided_by_square_equals_sixteen :
  star / square = 16 := by sorry

end star_divided_by_square_equals_sixteen_l3078_307813


namespace last_digit_of_189_in_ternary_l3078_307842

theorem last_digit_of_189_in_ternary (n : Nat) : n = 189 → n % 3 = 0 := by
  sorry

end last_digit_of_189_in_ternary_l3078_307842


namespace complex_set_property_l3078_307876

def is_closed_under_multiplication (S : Set ℂ) : Prop :=
  ∀ x y, x ∈ S → y ∈ S → (x * y) ∈ S

theorem complex_set_property (a b c d : ℂ) :
  let S : Set ℂ := {a, b, c, d}
  is_closed_under_multiplication S →
  a = 1 →
  b^2 = 1 →
  c^2 = b →
  b + c + d = -1 := by
  sorry

end complex_set_property_l3078_307876


namespace power_subtraction_l3078_307834

theorem power_subtraction : (81 : ℝ) ^ (1/4) - (16 : ℝ) ^ (1/2) = -1 := by sorry

end power_subtraction_l3078_307834


namespace antonia_pill_box_l3078_307837

def pill_box_problem (num_supplements : ℕ) 
                     (num_large_bottles : ℕ) 
                     (num_small_bottles : ℕ) 
                     (pills_per_large_bottle : ℕ) 
                     (pills_per_small_bottle : ℕ) 
                     (pills_left : ℕ) 
                     (num_weeks : ℕ) : Prop :=
  let total_pills := num_large_bottles * pills_per_large_bottle + 
                     num_small_bottles * pills_per_small_bottle
  let pills_used := total_pills - pills_left
  let days_filled := num_weeks * 7
  pills_used / num_supplements = days_filled

theorem antonia_pill_box : 
  pill_box_problem 5 3 2 120 30 350 2 = true :=
sorry

end antonia_pill_box_l3078_307837


namespace greater_root_of_quadratic_l3078_307883

theorem greater_root_of_quadratic (x : ℝ) :
  x^2 - 5*x - 36 = 0 → x ≤ 9 :=
by sorry

end greater_root_of_quadratic_l3078_307883


namespace min_value_theorem_l3078_307873

-- Define the function f
def f (x : ℝ) : ℝ := |2 * x - 1|

-- Define the function g
def g (x : ℝ) : ℝ := f x + f (x - 1)

-- State the theorem
theorem min_value_theorem (m n : ℝ) (hm : m > 0) (hn : n > 0) (h_sum : m + n = 2) :
  (m^2 + 2) / m + (n^2 + 1) / n ≥ (7 + 2 * Real.sqrt 2) / 2 :=
by sorry

end min_value_theorem_l3078_307873


namespace drummer_trombone_difference_l3078_307819

/-- Represents the number of players for each instrument in the school band --/
structure BandComposition where
  flute : Nat
  trumpet : Nat
  trombone : Nat
  clarinet : Nat
  frenchHorn : Nat
  drummer : Nat

/-- Theorem stating the difference between drummers and trombone players --/
theorem drummer_trombone_difference (band : BandComposition) : 
  band.flute = 5 →
  band.trumpet = 3 * band.flute →
  band.trombone = band.trumpet - 8 →
  band.clarinet = 2 * band.flute →
  band.frenchHorn = band.trombone + 3 →
  band.flute + band.trumpet + band.trombone + band.clarinet + band.frenchHorn + band.drummer = 65 →
  band.drummer > band.trombone →
  band.drummer - band.trombone = 11 := by
sorry


end drummer_trombone_difference_l3078_307819


namespace partial_fraction_decomposition_l3078_307891

theorem partial_fraction_decomposition (x : ℝ) (h1 : x ≠ 1) (h4 : x ≠ 4) (h6 : x ≠ 6) :
  (x^2 - 13) / ((x - 1) * (x - 4) * (x - 6)) =
  (-4/5) / (x - 1) + (-1/2) / (x - 4) + (23/10) / (x - 6) := by
  sorry

end partial_fraction_decomposition_l3078_307891


namespace major_axis_length_major_axis_length_is_eight_l3078_307862

/-- An ellipse with foci at (3, -4 + 2√3) and (3, -4 - 2√3), tangent to both x and y axes -/
structure TangentEllipse where
  /-- The ellipse is tangent to the x-axis -/
  tangent_x : Bool
  /-- The ellipse is tangent to the y-axis -/
  tangent_y : Bool
  /-- The first focus of the ellipse -/
  focus1 : ℝ × ℝ
  /-- The second focus of the ellipse -/
  focus2 : ℝ × ℝ
  /-- Condition that the first focus is at (3, -4 + 2√3) -/
  h1 : focus1 = (3, -4 + 2 * Real.sqrt 3)
  /-- Condition that the second focus is at (3, -4 - 2√3) -/
  h2 : focus2 = (3, -4 - 2 * Real.sqrt 3)
  /-- Condition that the ellipse is tangent to the x-axis -/
  h3 : tangent_x = true
  /-- Condition that the ellipse is tangent to the y-axis -/
  h4 : tangent_y = true

/-- The length of the major axis of the ellipse is 8 -/
theorem major_axis_length (e : TangentEllipse) : ℝ :=
  8

/-- The theorem stating that the major axis length of the given ellipse is 8 -/
theorem major_axis_length_is_eight (e : TangentEllipse) : 
  major_axis_length e = 8 := by sorry

end major_axis_length_major_axis_length_is_eight_l3078_307862


namespace geometric_series_ratio_l3078_307848

theorem geometric_series_ratio (a r : ℝ) (h : r ≠ 1) :
  (a / (1 - r) = 64 * (a * r^4 / (1 - r))) → r = 1/2 := by
  sorry

end geometric_series_ratio_l3078_307848


namespace isosceles_triangle_perimeter_l3078_307884

-- Define an isosceles triangle with side lengths 3 and 6
def IsoscelesTriangle (a b c : ℝ) : Prop :=
  (a = 3 ∧ b = 6 ∧ c = 6) ∨ (a = 6 ∧ b = 3 ∧ c = 3)

-- Define the perimeter of a triangle
def Perimeter (a b c : ℝ) : ℝ := a + b + c

-- Theorem statement
theorem isosceles_triangle_perimeter :
  ∀ a b c : ℝ, IsoscelesTriangle a b c → Perimeter a b c = 15 :=
by
  sorry


end isosceles_triangle_perimeter_l3078_307884


namespace sum_coefficients_x_minus_3y_to_20_l3078_307878

theorem sum_coefficients_x_minus_3y_to_20 :
  (fun x y => (x - 3 * y) ^ 20) 1 1 = 1048576 := by
  sorry

end sum_coefficients_x_minus_3y_to_20_l3078_307878


namespace square_sum_lower_bound_l3078_307882

theorem square_sum_lower_bound (x y : ℝ) (h : |x - 2*y| = 5) : x^2 + y^2 ≥ 5 := by
  sorry

end square_sum_lower_bound_l3078_307882


namespace melissa_driving_hours_l3078_307808

/-- Calculates the total hours Melissa spends driving in a year -/
def total_driving_hours (trips_per_month : ℕ) (hours_per_trip : ℕ) (months_per_year : ℕ) : ℕ :=
  trips_per_month * hours_per_trip * months_per_year

/-- Proves that Melissa spends 72 hours driving in a year -/
theorem melissa_driving_hours :
  total_driving_hours 2 3 12 = 72 := by
  sorry

end melissa_driving_hours_l3078_307808


namespace lines_intersection_l3078_307825

-- Define the two lines
def line1 (t : ℝ) : ℝ × ℝ := (3 * t, 2 + 4 * t)
def line2 (u : ℝ) : ℝ × ℝ := (1 + u, 1 - u)

-- State the theorem
theorem lines_intersection :
  ∃! p : ℝ × ℝ, (∃ t : ℝ, line1 t = p) ∧ (∃ u : ℝ, line2 u = p) ∧ p = (0, 2) := by
  sorry

end lines_intersection_l3078_307825


namespace inequality_proof_l3078_307833

theorem inequality_proof (a b : ℝ) (h1 : 0 < a) (h2 : a < b) (h3 : b < 1) :
  (1 - a) ^ a > (1 - b) ^ b :=
by sorry

end inequality_proof_l3078_307833


namespace valid_liar_counts_l3078_307827

/-- Represents the number of people in the room -/
def total_people : ℕ := 30

/-- Represents the possible numbers of liars in the room -/
def possible_liar_counts : List ℕ := [2, 3, 5, 6, 10, 15, 30]

/-- Predicate to check if a number is a valid liar count -/
def is_valid_liar_count (x : ℕ) : Prop :=
  x > 1 ∧ (total_people % x = 0) ∧
  ∃ (n : ℕ), (n + 1) * x = total_people

/-- Theorem stating that the possible_liar_counts are the only valid liar counts -/
theorem valid_liar_counts :
  ∀ (x : ℕ), is_valid_liar_count x ↔ x ∈ possible_liar_counts :=
by sorry

end valid_liar_counts_l3078_307827


namespace right_triangle_inequality_l3078_307880

theorem right_triangle_inequality (a b c : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : c > 0) 
  (h4 : a^2 + b^2 = c^2) : a^4 + b^4 < c^4 := by
  sorry

end right_triangle_inequality_l3078_307880


namespace hiking_team_gloves_l3078_307885

/-- The minimum number of gloves needed for a hiking team -/
theorem hiking_team_gloves (num_participants : ℕ) (gloves_per_participant : ℕ) 
  (h1 : num_participants = 63)
  (h2 : gloves_per_participant = 3) : 
  num_participants * gloves_per_participant = 189 := by
  sorry

end hiking_team_gloves_l3078_307885


namespace ryan_overall_percentage_l3078_307841

def total_problems : ℕ := 25 + 40 + 10

def correct_problems : ℕ := 
  (25 * 80 / 100) + (40 * 90 / 100) + (10 * 70 / 100)

theorem ryan_overall_percentage : 
  (correct_problems * 100) / total_problems = 84 := by sorry

end ryan_overall_percentage_l3078_307841


namespace sequence_relation_l3078_307877

/-- Given two sequences a and b, where a_n = n^2 and b_n are distinct positive integers,
    and for all n, the a_n-th term of b equals the b_n-th term of a,
    prove that (log(b 1 * b 4 * b 9 * b 16)) / (log(b 1 * b 2 * b 3 * b 4)) = 2 -/
theorem sequence_relation (b : ℕ+ → ℕ+) 
  (h_distinct : ∀ m n : ℕ+, m ≠ n → b m ≠ b n)
  (h_relation : ∀ n : ℕ+, b (n^2) = (b n)^2) :
  (Real.log ((b 1) * (b 4) * (b 9) * (b 16))) / (Real.log ((b 1) * (b 2) * (b 3) * (b 4))) = 2 := by
  sorry

end sequence_relation_l3078_307877


namespace race_time_calculation_race_problem_l3078_307838

theorem race_time_calculation (race_distance : ℝ) (a_time : ℝ) (beat_distance : ℝ) : ℝ :=
  let a_speed := race_distance / a_time
  let b_distance_when_a_finishes := race_distance - beat_distance
  let b_speed := b_distance_when_a_finishes / a_time
  let b_time := race_distance / b_speed
  b_time

theorem race_problem : 
  race_time_calculation 130 20 26 = 25 := by sorry

end race_time_calculation_race_problem_l3078_307838


namespace simplify_expression_l3078_307815

theorem simplify_expression (x y : ℝ) : 3*x + 5*x + 7*x + 2*y = 15*x + 2*y := by
  sorry

end simplify_expression_l3078_307815


namespace smallest_difference_l3078_307866

def Digits : Finset Nat := {1, 3, 4, 6, 7, 8}

def is_valid_subtraction (a b : Nat) : Prop :=
  a ≥ 1000 ∧ a < 10000 ∧ b ≥ 100 ∧ b < 1000 ∧
  (Digits.card = 6) ∧
  (Finset.card (Finset.filter (λ d => d ∈ Digits) (Finset.range 10)) = 6) ∧
  (∀ d ∈ Digits, (d ∈ a.digits 10 ∨ d ∈ b.digits 10)) ∧
  (∀ d ∈ (a.digits 10 ∪ b.digits 10), d ∈ Digits)

theorem smallest_difference : 
  ∀ a b : Nat, is_valid_subtraction a b → a - b ≥ 473 :=
sorry

end smallest_difference_l3078_307866


namespace janet_friday_gym_hours_l3078_307897

/-- Janet's weekly gym schedule -/
structure GymSchedule where
  total_hours : ℝ
  monday_hours : ℝ
  wednesday_hours : ℝ
  tuesday_friday_equal : Bool

/-- Theorem: Janet spends 1 hour at the gym on Friday -/
theorem janet_friday_gym_hours (schedule : GymSchedule) 
  (h1 : schedule.total_hours = 5)
  (h2 : schedule.monday_hours = 1.5)
  (h3 : schedule.wednesday_hours = 1.5)
  (h4 : schedule.tuesday_friday_equal = true) :
  ∃ friday_hours : ℝ, friday_hours = 1 ∧ 
  schedule.total_hours = schedule.monday_hours + schedule.wednesday_hours + 2 * friday_hours :=
by
  sorry

end janet_friday_gym_hours_l3078_307897


namespace evening_ticket_price_l3078_307896

/-- The price of a matinee ticket in dollars -/
def matinee_price : ℚ := 5

/-- The price of a 3D ticket in dollars -/
def three_d_price : ℚ := 20

/-- The number of matinee tickets sold -/
def matinee_count : ℕ := 200

/-- The number of evening tickets sold -/
def evening_count : ℕ := 300

/-- The number of 3D tickets sold -/
def three_d_count : ℕ := 100

/-- The total revenue in dollars -/
def total_revenue : ℚ := 6600

/-- The price of an evening ticket in dollars -/
def evening_price : ℚ := 12

theorem evening_ticket_price :
  matinee_price * matinee_count + evening_price * evening_count + three_d_price * three_d_count = total_revenue :=
by sorry

end evening_ticket_price_l3078_307896


namespace sages_can_succeed_l3078_307850

/-- Represents the color of a hat -/
def HatColor := Fin 1000

/-- Represents the signal a sage can show (white or black card) -/
def Signal := Bool

/-- Represents the configuration of hats on the sages -/
def HatConfiguration := Fin 11 → HatColor

/-- A strategy is a function that takes the colors of the other hats and returns a signal -/
def Strategy := (Fin 10 → HatColor) → Signal

/-- The result of applying a strategy is a function that determines the hat color based on the signals of others -/
def StrategyResult := (Fin 10 → Signal) → HatColor

/-- A successful strategy correctly determines the hat color for all possible configurations -/
def SuccessfulStrategy (strategy : Fin 11 → Strategy) (result : Fin 11 → StrategyResult) : Prop :=
  ∀ (config : HatConfiguration),
    ∀ (i : Fin 11),
      result i (λ j => if j < i then strategy j (λ k => config (k.succ)) 
                       else strategy j.succ (λ k => if k < j then config k else config k.succ)) = config i

theorem sages_can_succeed : ∃ (strategy : Fin 11 → Strategy) (result : Fin 11 → StrategyResult),
  SuccessfulStrategy strategy result := by
  sorry

end sages_can_succeed_l3078_307850


namespace solve_chocolate_problem_l3078_307807

def chocolate_problem (price_per_bar : ℕ) (total_bars : ℕ) (revenue : ℕ) : Prop :=
  let sold_bars : ℕ := revenue / price_per_bar
  let unsold_bars : ℕ := total_bars - sold_bars
  unsold_bars = 4

theorem solve_chocolate_problem :
  chocolate_problem 3 7 9 := by
  sorry

end solve_chocolate_problem_l3078_307807


namespace triangle_inequalities_l3078_307836

theorem triangle_inequalities (a b c A B C : Real) 
  (ha : a > 0) (hb : b > 0) (hc : c > 0)
  (hA : A > 0) (hB : B > 0) (hC : C > 0)
  (h_triangle : A + B + C = π)
  (h_sides : a = BC ∧ b = AC ∧ c = AB) : 
  (1 / a^3 + 1 / b^3 + 1 / c^3 + a*b*c ≥ 2 * Real.sqrt 3) ∧
  (1 / A + 1 / B + 1 / C ≥ 9 / π) := by
  sorry

end triangle_inequalities_l3078_307836


namespace parabola_vertex_l3078_307840

/-- The parabola equation -/
def parabola_equation (x y : ℝ) : Prop := y = -(x - 2)^2 + 3

/-- The vertex of the parabola -/
def vertex : ℝ × ℝ := (2, 3)

/-- Theorem: The vertex of the parabola y = -(x - 2)^2 + 3 is (2, 3) -/
theorem parabola_vertex :
  ∀ x y : ℝ, parabola_equation x y → (x, y) = vertex :=
sorry

end parabola_vertex_l3078_307840


namespace farm_theorem_l3078_307831

def farm_problem (num_pigs num_hens : ℕ) : Prop :=
  let num_heads := num_pigs + num_hens
  let num_legs := 4 * num_pigs + 2 * num_hens
  (num_pigs = 11) ∧ (∃ k : ℕ, num_legs = 2 * num_heads + k) ∧ (num_legs - 2 * num_heads = 22)

theorem farm_theorem : ∃ num_hens : ℕ, farm_problem 11 num_hens :=
by sorry

end farm_theorem_l3078_307831


namespace simplified_expression_terms_l3078_307805

def simplified_terms_count (n : ℕ) : ℕ :=
  (n / 2 + 1)^2

theorem simplified_expression_terms (n : ℕ) (h : n = 2008) :
  simplified_terms_count n = 1010025 :=
by
  sorry

end simplified_expression_terms_l3078_307805


namespace isosceles_triangle_coordinates_l3078_307829

def O : ℝ × ℝ := (0, 0)
def A : ℝ × ℝ := (4, 2)

def is_right_angle (p q r : ℝ × ℝ) : Prop :=
  (p.1 - q.1) * (r.1 - q.1) + (p.2 - q.2) * (r.2 - q.2) = 0

def is_isosceles (p q r : ℝ × ℝ) : Prop :=
  (p.1 - q.1)^2 + (p.2 - q.2)^2 = (p.1 - r.1)^2 + (p.2 - r.2)^2

theorem isosceles_triangle_coordinates :
  ∀ B : ℝ × ℝ,
    is_isosceles O A B →
    is_right_angle O B A →
    (B = (1, 3) ∨ B = (3, -1)) :=
by sorry

end isosceles_triangle_coordinates_l3078_307829
