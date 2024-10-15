import Mathlib

namespace NUMINAMATH_CALUDE_lizzy_money_problem_l2305_230518

/-- The amount of cents Lizzy's father gave her -/
def father_gave : ℕ := 40

/-- The amount of cents Lizzy spent on candy -/
def spent_on_candy : ℕ := 50

/-- The amount of cents Lizzy's uncle gave her -/
def uncle_gave : ℕ := 70

/-- The amount of cents Lizzy has now -/
def current_amount : ℕ := 140

/-- The amount of cents Lizzy's mother gave her -/
def mother_gave : ℕ := 80

theorem lizzy_money_problem :
  mother_gave = current_amount + spent_on_candy - (father_gave + uncle_gave) :=
by sorry

end NUMINAMATH_CALUDE_lizzy_money_problem_l2305_230518


namespace NUMINAMATH_CALUDE_hyperbola_asymptote_l2305_230504

/-- Given a hyperbola with equation x²/a² - y² = 1 where a > 0,
    if one of its asymptotes is √3x + y = 0, then a = √3/3 -/
theorem hyperbola_asymptote (a : ℝ) (h1 : a > 0) :
  (∃ x y : ℝ, x^2 / a^2 - y^2 = 1 ∧ Real.sqrt 3 * x + y = 0) →
  a = Real.sqrt 3 / 3 :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_asymptote_l2305_230504


namespace NUMINAMATH_CALUDE_abs_negative_seven_l2305_230521

theorem abs_negative_seven : |(-7 : ℤ)| = 7 := by
  sorry

end NUMINAMATH_CALUDE_abs_negative_seven_l2305_230521


namespace NUMINAMATH_CALUDE_loan_to_c_amount_lent_to_c_l2305_230545

/-- Calculates the simple interest --/
def simple_interest (principal : ℝ) (rate : ℝ) (time : ℝ) : ℝ :=
  principal * rate * time

theorem loan_to_c (loan_to_b : ℝ) (time_b : ℝ) (time_c : ℝ) (total_interest : ℝ) (rate : ℝ) : ℝ :=
  let interest_b := simple_interest loan_to_b rate time_b
  let interest_c := total_interest - interest_b
  interest_c / (rate * time_c)

/-- The amount A lent to C --/
theorem amount_lent_to_c : loan_to_c 5000 2 4 2640 0.12 = 3000 := by
  sorry

end NUMINAMATH_CALUDE_loan_to_c_amount_lent_to_c_l2305_230545


namespace NUMINAMATH_CALUDE_smallest_x_value_l2305_230580

theorem smallest_x_value (x : ℝ) : 
  ((5 * x - 20) / (4 * x - 5))^2 + ((5 * x - 20) / (4 * x - 5)) = 20 → x ≥ 0 :=
by sorry

end NUMINAMATH_CALUDE_smallest_x_value_l2305_230580


namespace NUMINAMATH_CALUDE_rebus_solution_exists_and_unique_l2305_230523

def is_valid_digit (n : ℕ) : Prop := n ≥ 0 ∧ n ≤ 9

def are_distinct (a b c d e f g h i j : ℕ) : Prop :=
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧ a ≠ f ∧ a ≠ g ∧ a ≠ h ∧ a ≠ i ∧ a ≠ j ∧
  b ≠ c ∧ b ≠ d ∧ b ≠ e ∧ b ≠ f ∧ b ≠ g ∧ b ≠ h ∧ b ≠ i ∧ b ≠ j ∧
  c ≠ d ∧ c ≠ e ∧ c ≠ f ∧ c ≠ g ∧ c ≠ h ∧ c ≠ i ∧ c ≠ j ∧
  d ≠ e ∧ d ≠ f ∧ d ≠ g ∧ d ≠ h ∧ d ≠ i ∧ d ≠ j ∧
  e ≠ f ∧ e ≠ g ∧ e ≠ h ∧ e ≠ i ∧ e ≠ j ∧
  f ≠ g ∧ f ≠ h ∧ f ≠ i ∧ f ≠ j ∧
  g ≠ h ∧ g ≠ i ∧ g ≠ j ∧
  h ≠ i ∧ h ≠ j ∧
  i ≠ j

theorem rebus_solution_exists_and_unique :
  ∃! (a b c d e f g h i j : ℕ),
    is_valid_digit a ∧ is_valid_digit b ∧ is_valid_digit c ∧
    is_valid_digit d ∧ is_valid_digit e ∧ is_valid_digit f ∧
    is_valid_digit g ∧ is_valid_digit h ∧ is_valid_digit i ∧ is_valid_digit j ∧
    are_distinct a b c d e f g h i j ∧
    100 * a + 10 * b + c + 100 * d + 10 * e + f = 1000 * g + 100 * h + 10 * i + j :=
sorry

end NUMINAMATH_CALUDE_rebus_solution_exists_and_unique_l2305_230523


namespace NUMINAMATH_CALUDE_diagonals_in_nonagon_l2305_230503

/-- The number of diagonals in a regular nine-sided polygon -/
theorem diagonals_in_nonagon : 
  let n : ℕ := 9  -- number of sides
  let total_connections := n.choose 2  -- total number of connections between vertices
  let num_sides := n  -- number of sides (which are not diagonals)
  total_connections - num_sides = 27 := by sorry

end NUMINAMATH_CALUDE_diagonals_in_nonagon_l2305_230503


namespace NUMINAMATH_CALUDE_sqrt_equation_solutions_l2305_230527

theorem sqrt_equation_solutions :
  ∀ x : ℝ, (Real.sqrt ((3 + Real.sqrt 5) ^ x) + Real.sqrt ((3 - Real.sqrt 5) ^ x) = 6) ↔ (x = 2 ∨ x = -2) := by
  sorry

end NUMINAMATH_CALUDE_sqrt_equation_solutions_l2305_230527


namespace NUMINAMATH_CALUDE_problem_statement_l2305_230598

theorem problem_statement :
  -- Part 1
  (∀ a b : ℝ, a > 0 ∧ b > 0 ∧ a + b = 1 → 1/a + 1/b ≥ 4) ∧
  -- Part 2
  (∃ min : ℝ, min = 1/14 ∧
    ∀ x y z : ℝ, x + 2*y + 3*z = 1 → x^2 + y^2 + z^2 ≥ min ∧
    ∃ x₀ y₀ z₀ : ℝ, x₀ + 2*y₀ + 3*z₀ = 1 ∧ x₀^2 + y₀^2 + z₀^2 = min) :=
by sorry

end NUMINAMATH_CALUDE_problem_statement_l2305_230598


namespace NUMINAMATH_CALUDE_bubble_pass_probability_specific_l2305_230585

/-- The probability that in a sequence of n distinct terms,
    the kth term ends up in the mth position after one bubble pass -/
def bubble_pass_probability (n k m : ℕ) : ℚ :=
  if k ≤ m ∧ m < n then
    (1 : ℚ) / k * (1 : ℚ) / (m - k + 1) * (1 : ℚ) / (n - m)
  else 0

/-- The main theorem stating the probability for the specific case -/
theorem bubble_pass_probability_specific :
  bubble_pass_probability 50 20 40 = 1 / 4000 := by
  sorry

#eval bubble_pass_probability 50 20 40

end NUMINAMATH_CALUDE_bubble_pass_probability_specific_l2305_230585


namespace NUMINAMATH_CALUDE_range_of_m_for_function_equality_l2305_230597

theorem range_of_m_for_function_equality (m : ℝ) : 
  (∀ x₁ ∈ (Set.Icc (-1 : ℝ) 2), ∃ x₀ ∈ (Set.Icc (-1 : ℝ) 2), 
    m * x₁ + 2 = x₀^2 - 2*x₀) → 
  m ∈ Set.Icc (-1 : ℝ) (1/2) := by
  sorry

end NUMINAMATH_CALUDE_range_of_m_for_function_equality_l2305_230597


namespace NUMINAMATH_CALUDE_problem_solution_l2305_230567

theorem problem_solution : 
  (1/2 - 2/3 - 3/4) * 12 = -11 ∧ 
  -(1^6) + |-2/3| - (1 - 5/9) + 2/3 = -1/9 := by sorry

end NUMINAMATH_CALUDE_problem_solution_l2305_230567


namespace NUMINAMATH_CALUDE_inequality_proof_l2305_230573

theorem inequality_proof (m n : ℕ) (h : m < Real.sqrt 2 * n) :
  (m : ℝ) / n < Real.sqrt 2 * (1 - 1 / (4 * n^2)) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l2305_230573


namespace NUMINAMATH_CALUDE_constant_term_binomial_expansion_l2305_230561

theorem constant_term_binomial_expansion (x : ℝ) : 
  (∃ c : ℝ, c = 1120 ∧ 
   ∃ f : ℝ → ℝ, 
   (∀ y, f y = (y - 2/y)^8) ∧
   (∃ g : ℝ → ℝ, (∀ y, f y = g y + c + y * (g y)))) :=
by sorry

end NUMINAMATH_CALUDE_constant_term_binomial_expansion_l2305_230561


namespace NUMINAMATH_CALUDE_equation_represents_hyperbola_l2305_230560

/-- The equation (x+y)^2 = x^2 + y^2 + 1 represents a hyperbola -/
theorem equation_represents_hyperbola :
  ∃ (a b : ℝ) (k : ℝ), k ≠ 0 ∧
  (∀ x y : ℝ, (x + y)^2 = x^2 + y^2 + 1 ↔ (x * y = k ∧ (x / a)^2 - (y / b)^2 = 1)) :=
by sorry

end NUMINAMATH_CALUDE_equation_represents_hyperbola_l2305_230560


namespace NUMINAMATH_CALUDE_agnes_current_age_l2305_230558

/-- Agnes's current age -/
def agnes_age : ℕ := 25

/-- Jane's current age -/
def jane_age : ℕ := 6

/-- Years into the future when Agnes will be twice Jane's age -/
def years_future : ℕ := 13

theorem agnes_current_age :
  agnes_age = 25 ∧
  jane_age = 6 ∧
  agnes_age + years_future = 2 * (jane_age + years_future) :=
by sorry

end NUMINAMATH_CALUDE_agnes_current_age_l2305_230558


namespace NUMINAMATH_CALUDE_billy_bobbi_probability_zero_l2305_230530

def billy_number (n : ℕ) : Prop := n > 0 ∧ n < 150 ∧ 15 ∣ n
def bobbi_number (n : ℕ) : Prop := n > 0 ∧ n < 150 ∧ 20 ∣ n
def is_square (n : ℕ) : Prop := ∃ m : ℕ, n = m * m

theorem billy_bobbi_probability_zero :
  ∀ (b₁ b₂ : ℕ), 
    billy_number b₁ → 
    bobbi_number b₂ → 
    (is_square b₁ ∨ is_square b₂) →
    b₁ = b₂ → 
    False :=
sorry

end NUMINAMATH_CALUDE_billy_bobbi_probability_zero_l2305_230530


namespace NUMINAMATH_CALUDE_sum_of_compositions_l2305_230534

def r (x : ℝ) : ℝ := |x + 1| - 3

def s (x : ℝ) : ℝ := -|x + 2|

def evaluation_points : List ℝ := [-5, -4, -3, -2, -1, 0, 1, 2, 3]

theorem sum_of_compositions :
  (evaluation_points.map (fun x => s (r x))).sum = -37 := by sorry

end NUMINAMATH_CALUDE_sum_of_compositions_l2305_230534


namespace NUMINAMATH_CALUDE_g_evaluation_l2305_230546

def g (x : ℝ) : ℝ := 3 * x^2 - 5 * x + 7

theorem g_evaluation : 3 * g 2 + 4 * g (-4) = 327 := by
  sorry

end NUMINAMATH_CALUDE_g_evaluation_l2305_230546


namespace NUMINAMATH_CALUDE_octal_5374_to_decimal_l2305_230525

def octal_to_decimal (a b c d : Nat) : Nat :=
  d * 8^0 + c * 8^1 + b * 8^2 + a * 8^3

theorem octal_5374_to_decimal :
  octal_to_decimal 5 3 7 4 = 2812 := by
  sorry

end NUMINAMATH_CALUDE_octal_5374_to_decimal_l2305_230525


namespace NUMINAMATH_CALUDE_polynomial_factorization_sum_l2305_230533

theorem polynomial_factorization_sum (b₁ b₂ b₃ c₁ c₂ c₃ : ℝ) 
  (h : ∀ x : ℝ, x^8 - 3*x^6 + 3*x^4 - x^2 + 2 = 
    (x^2 + b₁*x + c₁) * (x^2 + b₂*x + c₂) * (x^2 + 2*b₃*x + c₃)) : 
  b₁*c₁ + b₂*c₂ + 2*b₃*c₃ = 0 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_factorization_sum_l2305_230533


namespace NUMINAMATH_CALUDE_cosine_sum_theorem_l2305_230554

theorem cosine_sum_theorem : 
  12 * (Real.cos (π / 8)) ^ 4 + 
  (Real.cos (3 * π / 8)) ^ 4 + 
  (Real.cos (5 * π / 8)) ^ 4 + 
  (Real.cos (7 * π / 8)) ^ 4 = 3 / 2 := by
sorry

end NUMINAMATH_CALUDE_cosine_sum_theorem_l2305_230554


namespace NUMINAMATH_CALUDE_symmetric_line_l2305_230520

/-- Given a line L1 with equation x - 2y + 1 = 0 and a line of symmetry x = 1,
    the symmetric line L2 has the equation x + 2y - 3 = 0 -/
theorem symmetric_line (x y : ℝ) :
  (x - 2*y + 1 = 0) →  -- Original line L1
  (x = 1) →            -- Line of symmetry
  (x + 2*y - 3 = 0)    -- Symmetric line L2
:= by sorry

end NUMINAMATH_CALUDE_symmetric_line_l2305_230520


namespace NUMINAMATH_CALUDE_perfect_square_quadratic_l2305_230524

theorem perfect_square_quadratic (m : ℝ) : 
  (∀ x : ℝ, ∃ y : ℝ, x^2 + 2*(m-3)*x + 16 = y^2) → (m = 7 ∨ m = -1) := by
  sorry

end NUMINAMATH_CALUDE_perfect_square_quadratic_l2305_230524


namespace NUMINAMATH_CALUDE_equation_solution_l2305_230506

theorem equation_solution : 
  ∃! x : ℝ, x ≠ 2 ∧ (4 * x^2 + 3 * x + 2) / (x - 2) = 4 * x + 3 :=
by
  use -1
  sorry

end NUMINAMATH_CALUDE_equation_solution_l2305_230506


namespace NUMINAMATH_CALUDE_box_dimensions_l2305_230590

theorem box_dimensions (x : ℕ+) : 
  (((x : ℝ) + 3) * ((x : ℝ) - 4) * ((x : ℝ)^2 + 16) < 800 ∧ 
   (x : ℝ)^2 + 16 > 30) ↔ 
  (x = 4 ∨ x = 5) :=
sorry

end NUMINAMATH_CALUDE_box_dimensions_l2305_230590


namespace NUMINAMATH_CALUDE_total_fruits_shared_l2305_230509

def persimmons_to_yuna : ℕ := 2
def apples_to_minyoung : ℕ := 7

theorem total_fruits_shared : persimmons_to_yuna + apples_to_minyoung = 9 := by
  sorry

end NUMINAMATH_CALUDE_total_fruits_shared_l2305_230509


namespace NUMINAMATH_CALUDE_expression_evaluation_l2305_230549

theorem expression_evaluation (m n : ℤ) (hm : m = 1) (hn : n = -2) :
  -2 * (m * n - 3 * m^2) - (2 * m * n - 5 * (m * n - m^2)) = -1 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l2305_230549


namespace NUMINAMATH_CALUDE_proof_by_contradiction_assumption_l2305_230564

theorem proof_by_contradiction_assumption (a b : ℝ) : 
  (a ≤ b → False) → a > b :=
sorry

end NUMINAMATH_CALUDE_proof_by_contradiction_assumption_l2305_230564


namespace NUMINAMATH_CALUDE_square_formation_theorem_l2305_230512

/-- Function to calculate the sum of integers from 1 to n -/
def sum_to_n (n : ℕ) : ℕ := n * (n + 1) / 2

/-- Function to calculate the minimum number of sticks to break -/
def min_sticks_to_break (n : ℕ) : ℕ :=
  let total_length := sum_to_n n
  if total_length % 4 = 0 then 0
  else 
    let target_length := ((total_length + 3) / 4) * 4
    (target_length - total_length + 1) / 2

theorem square_formation_theorem :
  (min_sticks_to_break 12 = 2) ∧ (min_sticks_to_break 15 = 0) :=
sorry

end NUMINAMATH_CALUDE_square_formation_theorem_l2305_230512


namespace NUMINAMATH_CALUDE_infinitely_many_divisors_of_2_pow_n_plus_1_l2305_230575

theorem infinitely_many_divisors_of_2_pow_n_plus_1 (m : ℕ) :
  (3 ^ m) ∣ (2 ^ (3 ^ m) + 1) :=
sorry

end NUMINAMATH_CALUDE_infinitely_many_divisors_of_2_pow_n_plus_1_l2305_230575


namespace NUMINAMATH_CALUDE_greeting_card_profit_l2305_230515

/-- Represents the greeting card sale problem -/
theorem greeting_card_profit
  (purchase_price : ℚ)
  (total_sale : ℚ)
  (h_purchase : purchase_price = 21 / 100)
  (h_sale : total_sale = 1457 / 100)
  (h_price_limit : ∃ (selling_price : ℚ), 
    selling_price ≤ 2 * purchase_price ∧
    selling_price * (total_sale / selling_price) = total_sale)
  : ∃ (profit : ℚ), profit = 47 / 10 :=
sorry

end NUMINAMATH_CALUDE_greeting_card_profit_l2305_230515


namespace NUMINAMATH_CALUDE_problem_statement_l2305_230508

/-- The number we're looking for -/
def x : ℝ := 640

/-- 50% of x is 190 more than 20% of 650 -/
theorem problem_statement : 0.5 * x = 0.2 * 650 + 190 := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l2305_230508


namespace NUMINAMATH_CALUDE_problem_distribution_l2305_230584

def distribute_problems (n : ℕ) (m : ℕ) : ℕ :=
  n * (n - 1) * n^(m - 2)

theorem problem_distribution :
  distribute_problems 12 5 = 228096 := by
  sorry

end NUMINAMATH_CALUDE_problem_distribution_l2305_230584


namespace NUMINAMATH_CALUDE_reward_function_satisfies_requirements_l2305_230500

theorem reward_function_satisfies_requirements :
  let f : ℝ → ℝ := λ x => 2 * Real.sqrt x - 6
  let domain : Set ℝ := { x | 25 ≤ x ∧ x ≤ 1600 }
  (∀ x ∈ domain, ∀ y ∈ domain, x < y → f x < f y) ∧
  (∀ x ∈ domain, f x ≤ 90) ∧
  (∀ x ∈ domain, f x ≤ x / 5) :=
by sorry

end NUMINAMATH_CALUDE_reward_function_satisfies_requirements_l2305_230500


namespace NUMINAMATH_CALUDE_isosceles_triangle_circumradius_l2305_230529

/-- The radius of a circle circumscribing an isosceles triangle -/
theorem isosceles_triangle_circumradius (a b c : ℝ) (h_isosceles : a = b) (h_sides : a = 13 ∧ c = 10) :
  let s := (a + b + c) / 2
  let area := Real.sqrt (s * (s - a) * (s - b) * (s - c))
  (a * b * c) / (4 * area) = 169 / 24 := by
  sorry

end NUMINAMATH_CALUDE_isosceles_triangle_circumradius_l2305_230529


namespace NUMINAMATH_CALUDE_work_completion_time_l2305_230579

theorem work_completion_time (x_days y_days : ℕ) (x_remaining : ℕ) (y_worked : ℕ) : 
  x_days = 24 →
  y_worked = 10 →
  x_remaining = 9 →
  (y_worked : ℚ) / y_days + (x_remaining : ℚ) / x_days = 1 →
  y_days = 16 := by
sorry

end NUMINAMATH_CALUDE_work_completion_time_l2305_230579


namespace NUMINAMATH_CALUDE_right_triangle_with_incircle_l2305_230593

theorem right_triangle_with_incircle (r c a b : ℝ) : 
  r = 15 →  -- radius of incircle
  c = 73 →  -- hypotenuse
  r = (a + b - c) / 2 →  -- incircle radius formula
  a^2 + b^2 = c^2 →  -- Pythagorean theorem
  ((a = 55 ∧ b = 48) ∨ (a = 48 ∧ b = 55)) := by sorry

end NUMINAMATH_CALUDE_right_triangle_with_incircle_l2305_230593


namespace NUMINAMATH_CALUDE_open_box_volume_l2305_230505

/-- The volume of an open box formed by cutting squares from corners of a rectangular sheet -/
theorem open_box_volume 
  (sheet_length sheet_width cut_size : ℝ) 
  (h_length : sheet_length = 48)
  (h_width : sheet_width = 38)
  (h_cut : cut_size = 8) : 
  (sheet_length - 2 * cut_size) * (sheet_width - 2 * cut_size) * cut_size = 5632 :=
by sorry

end NUMINAMATH_CALUDE_open_box_volume_l2305_230505


namespace NUMINAMATH_CALUDE_cosine_product_inequality_l2305_230553

theorem cosine_product_inequality (a b c x : ℝ) :
  -(Real.sin ((b - c) / 2))^2 ≤ Real.cos (a * x + b) * Real.cos (a * x + c) ∧
  Real.cos (a * x + b) * Real.cos (a * x + c) ≤ (Real.cos ((b - c) / 2))^2 := by
  sorry

end NUMINAMATH_CALUDE_cosine_product_inequality_l2305_230553


namespace NUMINAMATH_CALUDE_range_of_a_l2305_230577

-- Define the sets A and B
def A : Set ℝ := {x : ℝ | 2 * x^2 - 3 * x + 1 ≤ 0}
def B (a : ℝ) : Set ℝ := {x : ℝ | x^2 - (2 * a + 1) * x + a * (a + 1) ≤ 0}

-- State the theorem
theorem range_of_a (a : ℝ) : (A ∩ (Set.compl (B a)) = ∅) → (0 ≤ a ∧ a ≤ 1/2) :=
sorry

end NUMINAMATH_CALUDE_range_of_a_l2305_230577


namespace NUMINAMATH_CALUDE_first_company_fixed_cost_l2305_230569

/-- The fixed amount charged by the first rental company -/
def F : ℝ := 38.95

/-- The cost per mile for the first rental company -/
def cost_per_mile_first : ℝ := 0.31

/-- The fixed amount charged by Safety Rent A Truck -/
def fixed_cost_safety : ℝ := 41.95

/-- The cost per mile for Safety Rent A Truck -/
def cost_per_mile_safety : ℝ := 0.29

/-- The number of miles driven -/
def miles : ℝ := 150.0

theorem first_company_fixed_cost :
  F + cost_per_mile_first * miles = fixed_cost_safety + cost_per_mile_safety * miles :=
by sorry

end NUMINAMATH_CALUDE_first_company_fixed_cost_l2305_230569


namespace NUMINAMATH_CALUDE_triangle_count_segment_count_l2305_230565

/-- Represents a convex polygon divided into triangles -/
structure TriangulatedPolygon where
  p : ℕ  -- number of triangles
  n : ℕ  -- number of vertices on the boundary
  m : ℕ  -- number of vertices inside

/-- The number of triangles in a triangulated polygon satisfies p = n + 2m - 2 -/
theorem triangle_count (poly : TriangulatedPolygon) :
  poly.p = poly.n + 2 * poly.m - 2 := by sorry

/-- The number of segments that are sides of the resulting triangles is 2n + 3m - 3 -/
theorem segment_count (poly : TriangulatedPolygon) :
  2 * poly.n + 3 * poly.m - 3 = poly.p + poly.n + poly.m - 1 := by sorry

end NUMINAMATH_CALUDE_triangle_count_segment_count_l2305_230565


namespace NUMINAMATH_CALUDE_number_divided_by_seven_l2305_230526

theorem number_divided_by_seven : ∃ x : ℚ, x / 7 = 5 / 14 ∧ x = 5 / 2 := by
  sorry

end NUMINAMATH_CALUDE_number_divided_by_seven_l2305_230526


namespace NUMINAMATH_CALUDE_abc_equality_l2305_230516

theorem abc_equality (a b c : ℝ) (h : a + 1/b = b + 1/c ∧ b + 1/c = c + 1/a) :
  a^2 * b^2 * c^2 = 1 ∨ (a = b ∧ b = c) := by
  sorry

end NUMINAMATH_CALUDE_abc_equality_l2305_230516


namespace NUMINAMATH_CALUDE_tank_emptied_in_two_minutes_l2305_230540

/-- Represents the state and properties of a water tank system -/
structure WaterTank where
  initialFill : ℚ  -- Initial fill level of the tank (1/5)
  pipeARate : ℚ    -- Rate at which pipe A fills the tank (1/15 per minute)
  pipeBRate : ℚ    -- Rate at which pipe B empties the tank (1/6 per minute)

/-- Calculates the time to empty or fill the tank completely -/
def timeToEmptyOrFill (tank : WaterTank) : ℚ :=
  tank.initialFill / (tank.pipeBRate - tank.pipeARate)

/-- Theorem stating that the tank will be emptied in 2 minutes -/
theorem tank_emptied_in_two_minutes (tank : WaterTank) 
  (h1 : tank.initialFill = 1/5)
  (h2 : tank.pipeARate = 1/15)
  (h3 : tank.pipeBRate = 1/6) : 
  timeToEmptyOrFill tank = 2 := by
  sorry

#eval timeToEmptyOrFill { initialFill := 1/5, pipeARate := 1/15, pipeBRate := 1/6 }

end NUMINAMATH_CALUDE_tank_emptied_in_two_minutes_l2305_230540


namespace NUMINAMATH_CALUDE_parabola_vertex_problem_parabola_vertex_l2305_230519

/-- The coordinates of the vertex of a parabola in the form y = a(x-h)^2 + k are (h,k) -/
theorem parabola_vertex (a h k : ℝ) : 
  let f : ℝ → ℝ := λ x ↦ a * (x - h)^2 + k
  (∀ x, f x ≥ f h) ∧ f h = k := by sorry

/-- The coordinates of the vertex of the parabola y = 3(x-7)^2 + 5 are (7,5) -/
theorem problem_parabola_vertex : 
  let f : ℝ → ℝ := λ x ↦ 3 * (x - 7)^2 + 5
  (∀ x, f x ≥ f 7) ∧ f 7 = 5 := by sorry

end NUMINAMATH_CALUDE_parabola_vertex_problem_parabola_vertex_l2305_230519


namespace NUMINAMATH_CALUDE_intersection_singleton_l2305_230551

/-- The set A parameterized by a -/
def A (a : ℝ) : Set (ℝ × ℝ) := {p | p.2 = a * p.1 + 1}

/-- The set B -/
def B : Set (ℝ × ℝ) := {p | p.2 = |p.1|}

/-- The theorem stating the condition for A ∩ B to be a singleton -/
theorem intersection_singleton (a : ℝ) :
  (∃! p, p ∈ A a ∩ B) ↔ a ∈ Set.Iic (-1) ∪ Set.Ici 1 :=
sorry

end NUMINAMATH_CALUDE_intersection_singleton_l2305_230551


namespace NUMINAMATH_CALUDE_max_product_sum_2006_l2305_230547

theorem max_product_sum_2006 : 
  (∃ (a b : ℤ), a + b = 2006 ∧ ∀ (x y : ℤ), x + y = 2006 → x * y ≤ a * b) ∧
  (∀ (a b : ℤ), a + b = 2006 → a * b ≤ 1006009) := by
  sorry

end NUMINAMATH_CALUDE_max_product_sum_2006_l2305_230547


namespace NUMINAMATH_CALUDE_repeating_decimal_fraction_sum_l2305_230559

theorem repeating_decimal_fraction_sum : ∃ (n d : ℕ), 
  (n.gcd d = 1) ∧ 
  (n : ℚ) / (d : ℚ) = 7 + 47 / 99 ∧ 
  n + d = 839 := by
  sorry

end NUMINAMATH_CALUDE_repeating_decimal_fraction_sum_l2305_230559


namespace NUMINAMATH_CALUDE_transportation_cost_optimization_l2305_230563

/-- Transportation cost optimization problem -/
theorem transportation_cost_optimization 
  (distance : ℝ) 
  (max_speed : ℝ) 
  (fixed_cost : ℝ) 
  (variable_cost_factor : ℝ) :
  distance = 1000 →
  max_speed = 80 →
  fixed_cost = 400 →
  variable_cost_factor = 1/4 →
  ∃ (optimal_speed : ℝ),
    optimal_speed > 0 ∧ 
    optimal_speed ≤ max_speed ∧
    optimal_speed = 40 ∧
    ∀ (speed : ℝ), 
      speed > 0 → 
      speed ≤ max_speed → 
      distance * (variable_cost_factor * speed + fixed_cost / speed) ≥ 
      distance * (variable_cost_factor * optimal_speed + fixed_cost / optimal_speed) :=
by sorry


end NUMINAMATH_CALUDE_transportation_cost_optimization_l2305_230563


namespace NUMINAMATH_CALUDE_light_2004_is_red_l2305_230528

def light_color (n : ℕ) : String :=
  match n % 6 with
  | 0 => "red"
  | 1 => "green"
  | 2 => "yellow"
  | 3 => "yellow"
  | 4 => "red"
  | 5 => "red"
  | _ => "error" -- This case should never occur

theorem light_2004_is_red : light_color 2004 = "red" := by
  sorry

end NUMINAMATH_CALUDE_light_2004_is_red_l2305_230528


namespace NUMINAMATH_CALUDE_set_S_satisfies_conditions_l2305_230594

def S : Finset Nat := {2, 3, 11, 23, 31}

def P : Nat := S.prod id

theorem set_S_satisfies_conditions :
  (∀ x ∈ S, x > 1) ∧
  (∀ x ∈ S, x ∣ (P / x + 1) ∧ x ≠ (P / x + 1)) := by
  sorry

end NUMINAMATH_CALUDE_set_S_satisfies_conditions_l2305_230594


namespace NUMINAMATH_CALUDE_vector_equation_and_parallel_condition_l2305_230572

/-- Vector in R² -/
structure Vec2 where
  x : ℝ
  y : ℝ

/-- Scalar multiplication for Vec2 -/
def scalarMul (r : ℝ) (v : Vec2) : Vec2 :=
  ⟨r * v.x, r * v.y⟩

/-- Addition for Vec2 -/
def add (v w : Vec2) : Vec2 :=
  ⟨v.x + w.x, v.y + w.y⟩

/-- Two Vec2 are parallel if their cross product is zero -/
def isParallel (v w : Vec2) : Prop :=
  v.x * w.y = v.y * w.x

theorem vector_equation_and_parallel_condition :
  let a : Vec2 := ⟨3, 2⟩
  let b : Vec2 := ⟨-1, 2⟩
  let c : Vec2 := ⟨4, 1⟩
  
  /- Part 1: Vector equation -/
  (a = add (scalarMul (5/9) b) (scalarMul (8/9) c)) ∧
  
  /- Part 2: Parallel condition -/
  (isParallel (add a (scalarMul (-16/13) c)) (add (scalarMul 2 b) (scalarMul (-1) a))) :=
by
  sorry

end NUMINAMATH_CALUDE_vector_equation_and_parallel_condition_l2305_230572


namespace NUMINAMATH_CALUDE_regular_polygon_150_deg_interior_has_12_sides_l2305_230576

/-- A regular polygon with interior angles of 150 degrees has 12 sides -/
theorem regular_polygon_150_deg_interior_has_12_sides :
  ∀ n : ℕ,
  n > 2 →
  (∀ angle : ℝ, angle = 150 → n * angle = (n - 2) * 180) →
  n = 12 := by
sorry

end NUMINAMATH_CALUDE_regular_polygon_150_deg_interior_has_12_sides_l2305_230576


namespace NUMINAMATH_CALUDE_tetrahedron_faces_tetrahedron_has_four_faces_l2305_230582

/-- A tetrahedron is a three-dimensional geometric shape with four triangular faces. -/
structure Tetrahedron where
  -- We don't need to define the internal structure for this problem

/-- The number of faces in a tetrahedron is 4. -/
theorem tetrahedron_faces (t : Tetrahedron) : Nat :=
  4

#check tetrahedron_faces

/-- Proof that a tetrahedron has 4 faces. -/
theorem tetrahedron_has_four_faces (t : Tetrahedron) : tetrahedron_faces t = 4 := by
  sorry

end NUMINAMATH_CALUDE_tetrahedron_faces_tetrahedron_has_four_faces_l2305_230582


namespace NUMINAMATH_CALUDE_manager_salary_calculation_l2305_230557

/-- The daily salary of a manager in a grocery store -/
def manager_salary : ℝ := 5

/-- The daily salary of a clerk in a grocery store -/
def clerk_salary : ℝ := 2

/-- The number of managers in the grocery store -/
def num_managers : ℕ := 2

/-- The number of clerks in the grocery store -/
def num_clerks : ℕ := 3

/-- The total daily salary of all employees in the grocery store -/
def total_salary : ℝ := 16

theorem manager_salary_calculation :
  manager_salary * num_managers + clerk_salary * num_clerks = total_salary :=
by sorry

end NUMINAMATH_CALUDE_manager_salary_calculation_l2305_230557


namespace NUMINAMATH_CALUDE_volume_increase_rectangular_prism_l2305_230542

theorem volume_increase_rectangular_prism 
  (l w h : ℝ) 
  (l_increase : ℝ) 
  (w_increase : ℝ) 
  (h_increase : ℝ) 
  (hl : l_increase = 0.15) 
  (hw : w_increase = 0.20) 
  (hh : h_increase = 0.10) :
  let new_volume := (l * (1 + l_increase)) * (w * (1 + w_increase)) * (h * (1 + h_increase))
  let original_volume := l * w * h
  let volume_increase_percentage := (new_volume - original_volume) / original_volume * 100
  volume_increase_percentage = 51.8 := by
sorry

end NUMINAMATH_CALUDE_volume_increase_rectangular_prism_l2305_230542


namespace NUMINAMATH_CALUDE_parabola_tangent_condition_l2305_230537

/-- A parabola is tangent to a line if and only if their intersection has exactly one solution --/
def is_tangent (a b : ℝ) : Prop :=
  ∃! x, a * x^2 + b * x + 12 = 2 * x + 3

/-- The main theorem stating the conditions for the parabola to be tangent to the line --/
theorem parabola_tangent_condition (a b : ℝ) :
  is_tangent a b ↔ (b = 2 + 6 * Real.sqrt a ∨ b = 2 - 6 * Real.sqrt a) ∧ a ≥ 0 := by
  sorry

end NUMINAMATH_CALUDE_parabola_tangent_condition_l2305_230537


namespace NUMINAMATH_CALUDE_range_of_a_l2305_230588

theorem range_of_a (x y a : ℝ) (h1 : x > y) (h2 : (a + 3) * x < (a + 3) * y) : a < -3 := by
  sorry

end NUMINAMATH_CALUDE_range_of_a_l2305_230588


namespace NUMINAMATH_CALUDE_digit67_is_one_l2305_230541

/-- The sequence of digits formed by concatenating integers from 50 down to 1 -/
def integerSequence : List Nat := sorry

/-- The 67th digit in the sequence -/
def digit67 : Nat := sorry

/-- Theorem stating that the 67th digit in the sequence is 1 -/
theorem digit67_is_one : digit67 = 1 := by sorry

end NUMINAMATH_CALUDE_digit67_is_one_l2305_230541


namespace NUMINAMATH_CALUDE_least_sum_p_q_l2305_230514

theorem least_sum_p_q : ∃ (p q : ℕ), 
  p > 1 ∧ q > 1 ∧ 
  17 * (p + 1) = 21 * (q + 1) ∧
  p + q = 38 ∧
  ∀ (p' q' : ℕ), p' > 1 → q' > 1 → 17 * (p' + 1) = 21 * (q' + 1) → p' + q' ≥ 38 :=
by sorry

end NUMINAMATH_CALUDE_least_sum_p_q_l2305_230514


namespace NUMINAMATH_CALUDE_range_of_y₂_l2305_230552

-- Define the ellipse C₁
def C₁ (x y : ℝ) : Prop := x^2 / 3 + y^2 / 2 = 1

-- Define the foci F₁ and F₂
def F₁ : ℝ × ℝ := (-1, 0)
def F₂ : ℝ × ℝ := (1, 0)

-- Define line l₁
def l₁ (x : ℝ) : Prop := x = -1

-- Define line l₂
def l₂ (y t : ℝ) : Prop := y = t

-- Define point P
def P (t : ℝ) : ℝ × ℝ := (-1, t)

-- Define curve C₂
def C₂ (x y : ℝ) : Prop := y^2 = 4 * x

-- Define points A, B, and C on C₂
def A : ℝ × ℝ := (1, 2)
def B (x₁ y₁ : ℝ) : Prop := C₂ x₁ y₁ ∧ (x₁, y₁) ≠ A
def C (x₂ y₂ : ℝ) : Prop := C₂ x₂ y₂ ∧ (x₂, y₂) ≠ A

-- AB perpendicular to BC
def perpendicular (x₁ y₁ x₂ y₂ : ℝ) : Prop :=
  (x₁ - 1) * (x₂ - x₁) + (y₁ - 2) * (y₂ - y₁) = 0

-- Theorem statement
theorem range_of_y₂ (x₁ y₁ x₂ y₂ : ℝ) :
  B x₁ y₁ → C x₂ y₂ → perpendicular x₁ y₁ x₂ y₂ →
  y₂ ∈ (Set.Iic (-6) \ {-6}) ∪ Set.Ici 10 :=
sorry

end NUMINAMATH_CALUDE_range_of_y₂_l2305_230552


namespace NUMINAMATH_CALUDE_rectangle_area_l2305_230501

-- Define the points
variable (A B C D E F : ℝ × ℝ)

-- Define the properties of the semicircle and rectangle
def is_semicircle (F E : ℝ × ℝ) : Prop := sorry

def is_inscribed_rectangle (A B C D : ℝ × ℝ) (F E : ℝ × ℝ) : Prop := sorry

def is_right_triangle (D F C : ℝ × ℝ) : Prop := sorry

-- Define the distances
def distance (P Q : ℝ × ℝ) : ℝ := sorry

-- Theorem statement
theorem rectangle_area (A B C D E F : ℝ × ℝ) :
  is_semicircle F E →
  is_inscribed_rectangle A B C D F E →
  is_right_triangle D F C →
  distance D A = 12 →
  distance F D = 7 →
  distance A E = 7 →
  distance D A * distance C D = 24 * Real.sqrt 30 := by sorry

end NUMINAMATH_CALUDE_rectangle_area_l2305_230501


namespace NUMINAMATH_CALUDE_smallest_result_l2305_230592

def S : Set Nat := {2, 3, 5, 7, 11, 13}

theorem smallest_result (a b c : Nat) (ha : a ∈ S) (hb : b ∈ S) (hc : c ∈ S) 
  (hdistinct : a ≠ b ∧ b ≠ c ∧ a ≠ c) : 
  (∀ x y z : Nat, x ∈ S → y ∈ S → z ∈ S → x ≠ y ∧ y ≠ z ∧ x ≠ z → 
    22 ≤ (x + x + y) * z) ∧ (∃ x y z : Nat, x ∈ S ∧ y ∈ S ∧ z ∈ S ∧ 
    x ≠ y ∧ y ≠ z ∧ x ≠ z ∧ (x + x + y) * z = 22) := by
  sorry

end NUMINAMATH_CALUDE_smallest_result_l2305_230592


namespace NUMINAMATH_CALUDE_sons_age_l2305_230583

theorem sons_age (son_age father_age : ℕ) : 
  father_age = son_age + 30 →
  father_age + 2 = 2 * (son_age + 2) →
  son_age = 28 := by
sorry

end NUMINAMATH_CALUDE_sons_age_l2305_230583


namespace NUMINAMATH_CALUDE_total_apples_bought_l2305_230544

theorem total_apples_bought (num_men num_women : ℕ) (apples_per_man : ℕ) (extra_apples_per_woman : ℕ) : 
  num_men = 2 → 
  num_women = 3 → 
  apples_per_man = 30 → 
  extra_apples_per_woman = 20 →
  num_men * apples_per_man + num_women * (apples_per_man + extra_apples_per_woman) = 210 := by
  sorry

#check total_apples_bought

end NUMINAMATH_CALUDE_total_apples_bought_l2305_230544


namespace NUMINAMATH_CALUDE_meter_to_skips_l2305_230511

theorem meter_to_skips 
  (a b c d e f g : ℝ) 
  (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0) (he : e > 0) (hf : f > 0) (hg : g > 0)
  (hop_skip : a * 1 = b * 1)  -- a hops = b skips
  (jog_hop : c * 1 = d * 1)   -- c jogs = d hops
  (dash_jog : e * 1 = f * 1)  -- e dashes = f jogs
  (meter_dash : 1 = g * 1)    -- 1 meter = g dashes
  : 1 = (g * f * d * b) / (e * c * a) * 1 := by
  sorry

end NUMINAMATH_CALUDE_meter_to_skips_l2305_230511


namespace NUMINAMATH_CALUDE_parabola_equation_l2305_230548

theorem parabola_equation (p : ℝ) (x₀ y₀ : ℝ) : 
  p > 0 → 
  y₀^2 = 2*p*x₀ → 
  (x₀ + p/2)^2 + y₀^2 = 100 → 
  y₀^2 = 36 → 
  (y^2 = 4*x ∨ y^2 = 36*x) := by
  sorry

end NUMINAMATH_CALUDE_parabola_equation_l2305_230548


namespace NUMINAMATH_CALUDE_right_triangle_345_l2305_230517

theorem right_triangle_345 : ∃ (a b c : ℝ), a = 3 ∧ b = 4 ∧ c = 5 ∧ a^2 + b^2 = c^2 :=
  sorry

end NUMINAMATH_CALUDE_right_triangle_345_l2305_230517


namespace NUMINAMATH_CALUDE_total_time_outside_class_l2305_230578

def recess_break_1 : ℕ := 15
def recess_break_2 : ℕ := 15
def lunch_break : ℕ := 30
def additional_recess : ℕ := 20

theorem total_time_outside_class : 
  2 * recess_break_1 + lunch_break + additional_recess = 80 := by
  sorry

end NUMINAMATH_CALUDE_total_time_outside_class_l2305_230578


namespace NUMINAMATH_CALUDE_fraction_division_proof_l2305_230568

theorem fraction_division_proof : (5 / 4) / (8 / 15) = 75 / 32 := by
  sorry

end NUMINAMATH_CALUDE_fraction_division_proof_l2305_230568


namespace NUMINAMATH_CALUDE_intersection_M_N_l2305_230556

-- Define the sets M and N
def M : Set ℝ := {x | 1 + x ≥ 0}
def N : Set ℝ := {x | (4 : ℝ) / (1 - x) > 0}

-- State the theorem
theorem intersection_M_N :
  M ∩ N = {x : ℝ | -1 ≤ x ∧ x < 1} := by sorry

end NUMINAMATH_CALUDE_intersection_M_N_l2305_230556


namespace NUMINAMATH_CALUDE_new_person_weight_l2305_230522

theorem new_person_weight (initial_count : ℕ) (weight_increase : ℝ) (replaced_weight : ℝ) :
  initial_count = 8 →
  weight_increase = 2.5 →
  replaced_weight = 35 →
  (initial_count : ℝ) * weight_increase + replaced_weight = 55 :=
by sorry

end NUMINAMATH_CALUDE_new_person_weight_l2305_230522


namespace NUMINAMATH_CALUDE_outlet_pipe_emptying_time_l2305_230586

/-- Given an outlet pipe that empties 1/3 of a cistern in 8 minutes,
    prove that it takes 16 minutes to empty 2/3 of the cistern. -/
theorem outlet_pipe_emptying_time
  (emptying_rate : ℝ → ℝ)
  (h1 : emptying_rate 8 = 1/3)
  (h2 : ∀ t : ℝ, emptying_rate t = (t/8) * (1/3)) :
  ∃ t : ℝ, emptying_rate t = 2/3 ∧ t = 16 :=
sorry

end NUMINAMATH_CALUDE_outlet_pipe_emptying_time_l2305_230586


namespace NUMINAMATH_CALUDE_right_triangle_sin_value_l2305_230591

/-- Given a right triangle DEF with �angle E = 90° and 4 sin D = 5 cos D, prove that sin D = (5√41) / 41 -/
theorem right_triangle_sin_value (D E F : ℝ) (h_right_angle : E = 90) 
  (h_sin_cos_relation : 4 * Real.sin D = 5 * Real.cos D) : 
  Real.sin D = (5 * Real.sqrt 41) / 41 := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_sin_value_l2305_230591


namespace NUMINAMATH_CALUDE_evaluate_expression_l2305_230596

theorem evaluate_expression : -25 - 7 * (4 + 2) = -67 := by
  sorry

end NUMINAMATH_CALUDE_evaluate_expression_l2305_230596


namespace NUMINAMATH_CALUDE_frac_greater_than_one_solution_set_l2305_230538

theorem frac_greater_than_one_solution_set (x : ℝ) : 
  (1 / x > 1) ↔ (0 < x ∧ x < 1) :=
sorry

end NUMINAMATH_CALUDE_frac_greater_than_one_solution_set_l2305_230538


namespace NUMINAMATH_CALUDE_perpendicular_lines_parallel_perpendicular_planes_parallel_l2305_230536

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the perpendicular and parallel relations
variable (perp_line_plane : Line → Plane → Prop)
variable (perp_plane_plane : Plane → Plane → Prop)
variable (parallel_line : Line → Line → Prop)
variable (parallel_plane : Plane → Plane → Prop)

-- Theorem 1: If two lines are perpendicular to the same plane, they are parallel
theorem perpendicular_lines_parallel (a b : Line) (α : Plane) :
  perp_line_plane a α → perp_line_plane b α → parallel_line a b :=
sorry

-- Theorem 2: If a line is perpendicular to a plane, and that plane is perpendicular to another plane, then the two planes are parallel
theorem perpendicular_planes_parallel (a : Line) (α β : Plane) :
  perp_line_plane a α → perp_plane_plane α β → parallel_plane α β :=
sorry

end NUMINAMATH_CALUDE_perpendicular_lines_parallel_perpendicular_planes_parallel_l2305_230536


namespace NUMINAMATH_CALUDE_max_digit_sum_l2305_230574

theorem max_digit_sum (d e f z : ℕ) : 
  d ≤ 9 → e ≤ 9 → f ≤ 9 →
  (d * 100 + e * 10 + f : ℚ) / 1000 = 1 / z →
  0 < z → z ≤ 9 →
  d + e + f ≤ 8 := by
sorry

end NUMINAMATH_CALUDE_max_digit_sum_l2305_230574


namespace NUMINAMATH_CALUDE_washington_party_handshakes_l2305_230571

/-- Represents a party with married couples -/
structure Party where
  couples : Nat
  men : Nat
  women : Nat

/-- Calculates the number of handshakes in the party -/
def handshakes (p : Party) : Nat :=
  -- Handshakes among men
  (p.men.choose 2) +
  -- Handshakes between men and women (excluding spouses)
  p.men * (p.women - 1)

/-- Theorem stating the number of handshakes at George Washington's party -/
theorem washington_party_handshakes :
  ∃ (p : Party),
    p.couples = 13 ∧
    p.men = p.couples ∧
    p.women = p.couples ∧
    handshakes p = 234 := by
  sorry

end NUMINAMATH_CALUDE_washington_party_handshakes_l2305_230571


namespace NUMINAMATH_CALUDE_total_groom_time_in_minutes_l2305_230562

/-- The time in hours it takes to groom a dog -/
def dog_groom_time : ℝ := 2.5

/-- The time in hours it takes to groom a cat -/
def cat_groom_time : ℝ := 0.5

/-- The number of dogs to be groomed -/
def num_dogs : ℕ := 5

/-- The number of cats to be groomed -/
def num_cats : ℕ := 3

/-- The number of minutes in an hour -/
def minutes_per_hour : ℕ := 60

/-- Theorem stating that the total time to groom 5 dogs and 3 cats is 840 minutes -/
theorem total_groom_time_in_minutes : 
  (dog_groom_time * num_dogs + cat_groom_time * num_cats) * minutes_per_hour = 840 := by
  sorry

end NUMINAMATH_CALUDE_total_groom_time_in_minutes_l2305_230562


namespace NUMINAMATH_CALUDE_simplify_fraction_l2305_230543

theorem simplify_fraction : 5 * (14 / 3) * (21 / -70) = -35 / 2 := by
  sorry

end NUMINAMATH_CALUDE_simplify_fraction_l2305_230543


namespace NUMINAMATH_CALUDE_journey_solution_l2305_230513

def journey_problem (total_time : ℝ) (speed1 speed2 speed3 speed4 : ℝ) : Prop :=
  let distance := total_time * (speed1 + speed2 + speed3 + speed4) / 4
  total_time = (distance / 4) / speed1 + (distance / 4) / speed2 + (distance / 4) / speed3 + (distance / 4) / speed4 ∧
  distance = 960

theorem journey_solution :
  journey_problem 60 20 10 15 30 := by
  sorry

end NUMINAMATH_CALUDE_journey_solution_l2305_230513


namespace NUMINAMATH_CALUDE_factorization_equality_l2305_230510

theorem factorization_equality (x y : ℝ) : -4 * x^2 + y^2 = (y - 2*x) * (y + 2*x) := by
  sorry

end NUMINAMATH_CALUDE_factorization_equality_l2305_230510


namespace NUMINAMATH_CALUDE_three_digit_number_count_l2305_230502

/-- A three-digit number where the hundreds digit equals the units digit -/
structure ThreeDigitNumber where
  hundreds : Nat
  tens : Nat
  units : Nat
  is_valid : hundreds = units ∧ hundreds ≥ 1 ∧ hundreds ≤ 9 ∧ tens ≤ 9

/-- The value of a ThreeDigitNumber -/
def ThreeDigitNumber.value (n : ThreeDigitNumber) : Nat :=
  100 * n.hundreds + 10 * n.tens + n.units

/-- Predicate for divisibility by 4 -/
def divisible_by_four (n : Nat) : Prop :=
  n % 4 = 0

theorem three_digit_number_count :
  (∃ (s : Finset ThreeDigitNumber), s.card = 90) ∧
  (∃ (s : Finset ThreeDigitNumber), s.card = 20 ∧ ∀ n ∈ s, divisible_by_four n.value) :=
sorry

end NUMINAMATH_CALUDE_three_digit_number_count_l2305_230502


namespace NUMINAMATH_CALUDE_sugar_solution_replacement_l2305_230599

theorem sugar_solution_replacement (W : ℝ) (x : ℝ) : 
  (W > 0) → 
  (0 ≤ x) → (x ≤ 1) →
  ((1 - x) * (0.22 * W) + x * (0.74 * W) = 0.35 * W) ↔ 
  (x = 1/4) :=
by sorry

end NUMINAMATH_CALUDE_sugar_solution_replacement_l2305_230599


namespace NUMINAMATH_CALUDE_absolute_value_of_h_l2305_230539

theorem absolute_value_of_h (h : ℝ) : 
  (∃ x y : ℝ, x^2 + 2*h*x = 8 ∧ y^2 + 2*h*y = 8 ∧ x^2 + y^2 = 18) → 
  |h| = Real.sqrt 2 / 2 := by
sorry

end NUMINAMATH_CALUDE_absolute_value_of_h_l2305_230539


namespace NUMINAMATH_CALUDE_problem_statement_l2305_230595

theorem problem_statement (a b : ℕ+) :
  (18 ^ a.val) * (9 ^ (3 * a.val - 1)) = (2 ^ 6) * (3 ^ b.val) →
  a.val = 6 := by
sorry

end NUMINAMATH_CALUDE_problem_statement_l2305_230595


namespace NUMINAMATH_CALUDE_imaginary_part_of_complex_fraction_l2305_230566

theorem imaginary_part_of_complex_fraction (a : ℝ) : 
  Complex.im ((1 + a * Complex.I) / Complex.I) = -1 := by
  sorry

end NUMINAMATH_CALUDE_imaginary_part_of_complex_fraction_l2305_230566


namespace NUMINAMATH_CALUDE_quadratic_roots_problem_l2305_230550

theorem quadratic_roots_problem (m : ℝ) (α β : ℝ) :
  (α > 0 ∧ β > 0) ∧ 
  (α^2 + (2*m - 1)*α + m^2 = 0) ∧ 
  (β^2 + (2*m - 1)*β + m^2 = 0) →
  ((m ≤ 1/4 ∧ m ≠ 0) ∧
   (α^2 + β^2 = 49 → m = -4)) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_roots_problem_l2305_230550


namespace NUMINAMATH_CALUDE_ten_machines_four_minutes_l2305_230587

/-- The number of bottles produced by a given number of machines in a given time -/
def bottles_produced (machines : ℕ) (minutes : ℕ) : ℕ :=
  let bottles_per_minute := (270 * machines) / 5
  bottles_per_minute * minutes

/-- Theorem stating that 10 machines produce 2160 bottles in 4 minutes -/
theorem ten_machines_four_minutes :
  bottles_produced 10 4 = 2160 := by
  sorry

end NUMINAMATH_CALUDE_ten_machines_four_minutes_l2305_230587


namespace NUMINAMATH_CALUDE_complex_trajectory_l2305_230581

theorem complex_trajectory (x y : ℝ) (z : ℂ) (h1 : x ≥ 1/2) (h2 : z = x + y * I) (h3 : Complex.abs (z - 1) = x) :
  y^2 = 2*x - 1 :=
sorry

end NUMINAMATH_CALUDE_complex_trajectory_l2305_230581


namespace NUMINAMATH_CALUDE_fraction_equation_solution_l2305_230570

theorem fraction_equation_solution :
  ∀ x : ℚ, (1 : ℚ) / 3 + (1 : ℚ) / 4 = 1 / x → x = 12 / 7 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equation_solution_l2305_230570


namespace NUMINAMATH_CALUDE_fraction_equality_l2305_230532

theorem fraction_equality (p q : ℚ) (h : p / q = 4 / 5) :
  18 / 7 + (2 * q - p) / (2 * q + p) = 3 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_l2305_230532


namespace NUMINAMATH_CALUDE_incorrect_assignment_l2305_230535

-- Define valid assignment statements
def valid_assignment (stmt : String) : Prop :=
  stmt = "N = N + 1" ∨ stmt = "K = K * K" ∨ stmt = "C = A / B"

-- Define the statement in question
def questionable_statement : String := "C = A(B + D)"

-- Theorem to prove
theorem incorrect_assignment :
  (∀ stmt, valid_assignment stmt → stmt ≠ questionable_statement) →
  ¬(valid_assignment questionable_statement) :=
by
  sorry

end NUMINAMATH_CALUDE_incorrect_assignment_l2305_230535


namespace NUMINAMATH_CALUDE_livestock_puzzle_l2305_230531

theorem livestock_puzzle :
  ∃! (x y z : ℕ), 
    x + y + z = 100 ∧ 
    10 * x + 3 * y + (1/2) * z = 100 ∧
    x = 5 ∧ y = 1 ∧ z = 94 := by
  sorry

end NUMINAMATH_CALUDE_livestock_puzzle_l2305_230531


namespace NUMINAMATH_CALUDE_johns_shower_duration_l2305_230555

/-- Proves that John's shower duration is 10 minutes given the conditions --/
theorem johns_shower_duration :
  let days_in_four_weeks : ℕ := 28
  let shower_frequency : ℕ := 2  -- every other day
  let water_usage_per_minute : ℚ := 2  -- gallons per minute
  let total_water_usage : ℚ := 280  -- gallons in 4 weeks
  
  let num_showers : ℕ := days_in_four_weeks / shower_frequency
  let water_per_shower : ℚ := total_water_usage / num_showers
  let shower_duration : ℚ := water_per_shower / water_usage_per_minute
  
  shower_duration = 10 := by sorry

end NUMINAMATH_CALUDE_johns_shower_duration_l2305_230555


namespace NUMINAMATH_CALUDE_max_integer_k_for_distinct_roots_l2305_230507

theorem max_integer_k_for_distinct_roots (k : ℤ) : 
  (∃ x y : ℝ, x ≠ y ∧ x^2 - (4*k - 2)*x + 4*k^2 = 0 ∧ y^2 - (4*k - 2)*y + 4*k^2 = 0) →
  k ≤ 0 :=
by sorry

end NUMINAMATH_CALUDE_max_integer_k_for_distinct_roots_l2305_230507


namespace NUMINAMATH_CALUDE_arithmetic_sequence_common_difference_l2305_230589

/-- An arithmetic sequence with 10 terms where the sum of odd-numbered terms is 15
    and the sum of even-numbered terms is 30 has a common difference of 3. -/
theorem arithmetic_sequence_common_difference 
  (a : ℕ → ℚ) -- The arithmetic sequence
  (h1 : a 1 + a 3 + a 5 + a 7 + a 9 = 15) -- Sum of odd-numbered terms
  (h2 : a 2 + a 4 + a 6 + a 8 + a 10 = 30) -- Sum of even-numbered terms
  (h3 : ∀ n : ℕ, n < 10 → a (n + 1) - a n = a 2 - a 1) -- Definition of arithmetic sequence
  : a 2 - a 1 = 3 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_common_difference_l2305_230589
