import Mathlib

namespace NUMINAMATH_CALUDE_smallest_circular_sequence_l479_47919

def is_valid_sequence (s : List Nat) : Prop :=
  ∀ x ∈ s, x = 1 ∨ x = 2

def contains_all_four_digit_sequences (s : List Nat) : Prop :=
  ∀ seq : List Nat, seq.length = 4 → is_valid_sequence seq →
    ∃ i, List.take 4 (List.rotateLeft s i ++ List.rotateLeft s i) = seq ∨
         List.take 4 (List.rotateRight s i ++ List.rotateRight s i) = seq

theorem smallest_circular_sequence :
  ∃ (N : Nat) (s : List Nat),
    N = s.length ∧
    is_valid_sequence s ∧
    contains_all_four_digit_sequences s ∧
    (∀ M < N, ¬∃ t : List Nat, M = t.length ∧ is_valid_sequence t ∧ contains_all_four_digit_sequences t) ∧
    N = 14 := by
  sorry

end NUMINAMATH_CALUDE_smallest_circular_sequence_l479_47919


namespace NUMINAMATH_CALUDE_inequalities_not_always_satisfied_l479_47933

theorem inequalities_not_always_satisfied :
  ∃ (a b c x y z : ℝ), 
    x ≤ a ∧ y ≤ b ∧ z ≤ c ∧
    ((x^2 * y + y^2 * z + z^2 * x ≥ a^2 * b + b^2 * c + c^2 * a) ∨
     (x^3 + y^3 + z^3 ≥ a^3 + b^3 + c^3)) :=
by sorry

end NUMINAMATH_CALUDE_inequalities_not_always_satisfied_l479_47933


namespace NUMINAMATH_CALUDE_trigonometric_equation_l479_47982

theorem trigonometric_equation (α : Real) 
  (h : (Real.tan α - 3) * (Real.sin α + Real.cos α + 3) = 0) : 
  ((4 * Real.sin α + 2 * Real.cos α) / (5 * Real.cos α + 3 * Real.sin α) = 1) ∧ 
  (2 + (2/3) * (Real.sin α)^2 + (1/4) * (Real.cos α)^2 = 21/8) := by
  sorry

end NUMINAMATH_CALUDE_trigonometric_equation_l479_47982


namespace NUMINAMATH_CALUDE_city_population_l479_47920

theorem city_population (known_percentage : ℝ) (known_population : ℕ) (total_population : ℕ) : 
  known_percentage = 96 / 100 →
  known_population = 23040 →
  (known_percentage * total_population : ℝ) = known_population →
  total_population = 24000 := by
  sorry

end NUMINAMATH_CALUDE_city_population_l479_47920


namespace NUMINAMATH_CALUDE_integral_x_plus_x_squared_plus_sin_x_l479_47918

theorem integral_x_plus_x_squared_plus_sin_x : 
  ∫ x in (-1 : ℝ)..1, (x + x^2 + Real.sin x) = 2/3 := by sorry

end NUMINAMATH_CALUDE_integral_x_plus_x_squared_plus_sin_x_l479_47918


namespace NUMINAMATH_CALUDE_bryden_received_is_ten_l479_47922

/-- The amount a collector pays for a state quarter, as a multiple of its face value -/
def collector_rate : ℚ := 5

/-- The face value of a single state quarter in dollars -/
def quarter_value : ℚ := 1/2

/-- The number of state quarters Bryden has -/
def bryden_quarters : ℕ := 4

/-- The amount Bryden will receive from the collector in dollars -/
def bryden_received : ℚ := collector_rate * quarter_value * bryden_quarters

theorem bryden_received_is_ten : bryden_received = 10 := by
  sorry

end NUMINAMATH_CALUDE_bryden_received_is_ten_l479_47922


namespace NUMINAMATH_CALUDE_problem1_l479_47928

theorem problem1 (a b : ℝ) (h1 : a = 1) (h2 : b = -3) :
  (a - b)^2 - 2*a*(a + 3*b) + (a + 2*b)*(a - 2*b) = -3 := by
  sorry

end NUMINAMATH_CALUDE_problem1_l479_47928


namespace NUMINAMATH_CALUDE_max_total_profit_max_avg_annual_profit_l479_47947

/-- The total profit function for a coach operation -/
def total_profit (x : ℕ+) : ℚ := -x^2 + 18*x - 36

/-- The average annual profit function for a coach operation -/
def avg_annual_profit (x : ℕ+) : ℚ := (total_profit x) / x

/-- Theorem stating the year of maximum total profit -/
theorem max_total_profit :
  ∃ (x : ℕ+), ∀ (y : ℕ+), total_profit x ≥ total_profit y ∧ x = 9 :=
sorry

/-- Theorem stating the year of maximum average annual profit -/
theorem max_avg_annual_profit :
  ∃ (x : ℕ+), ∀ (y : ℕ+), avg_annual_profit x ≥ avg_annual_profit y ∧ x = 6 :=
sorry

end NUMINAMATH_CALUDE_max_total_profit_max_avg_annual_profit_l479_47947


namespace NUMINAMATH_CALUDE_max_sum_of_squares_l479_47912

theorem max_sum_of_squares (a b c d : ℝ) (h : a^2 + b^2 + c^2 + d^2 = 5) :
  ∃ M : ℝ, M = 20 ∧ ∀ x y z w : ℝ, x^2 + y^2 + z^2 + w^2 = 5 →
    (x - y)^2 + (x - z)^2 + (x - w)^2 + (y - z)^2 + (y - w)^2 + (z - w)^2 ≤ M :=
by sorry

end NUMINAMATH_CALUDE_max_sum_of_squares_l479_47912


namespace NUMINAMATH_CALUDE_negation_of_universal_proposition_l479_47985

theorem negation_of_universal_proposition :
  (¬ ∀ x : ℝ, x^2 - 2*x - 1 > 0) ↔ (∃ x₀ : ℝ, x₀^2 - 2*x₀ - 1 ≤ 0) := by sorry

end NUMINAMATH_CALUDE_negation_of_universal_proposition_l479_47985


namespace NUMINAMATH_CALUDE_money_distribution_l479_47989

theorem money_distribution (a b c total : ℕ) : 
  (a + b + c = total) →
  (2 * b = 3 * a) →
  (4 * b = 3 * c) →
  (b = 1500) →
  (total = 4500) := by
sorry

end NUMINAMATH_CALUDE_money_distribution_l479_47989


namespace NUMINAMATH_CALUDE_store_promotion_probabilities_l479_47999

/-- A store promotion event with three prizes -/
structure StorePromotion where
  p_first : ℝ  -- Probability of winning first prize
  p_second : ℝ  -- Probability of winning second prize
  p_third : ℝ  -- Probability of winning third prize
  h_first : 0 ≤ p_first ∧ p_first ≤ 1
  h_second : 0 ≤ p_second ∧ p_second ≤ 1
  h_third : 0 ≤ p_third ∧ p_third ≤ 1

/-- The probability of winning a prize in the store promotion -/
def prob_win_prize (sp : StorePromotion) : ℝ :=
  sp.p_first + sp.p_second + sp.p_third

/-- The probability of not winning any prize in the store promotion -/
def prob_no_prize (sp : StorePromotion) : ℝ :=
  1 - prob_win_prize sp

/-- Theorem stating the probabilities for a specific store promotion -/
theorem store_promotion_probabilities (sp : StorePromotion) 
  (h1 : sp.p_first = 0.1) (h2 : sp.p_second = 0.2) (h3 : sp.p_third = 0.4) : 
  prob_win_prize sp = 0.7 ∧ prob_no_prize sp = 0.3 := by
  sorry

end NUMINAMATH_CALUDE_store_promotion_probabilities_l479_47999


namespace NUMINAMATH_CALUDE_binomial_expansion_problem_l479_47952

theorem binomial_expansion_problem (a₀ a₁ a₂ a₃ : ℝ) :
  (∀ x : ℝ, (Real.sqrt 5 * x - 1)^3 = a₀ + a₁ * x + a₂ * x^2 + a₃ * x^3) →
  (a₀ + a₂)^2 - (a₁ + a₃)^2 = -64 := by
  sorry

end NUMINAMATH_CALUDE_binomial_expansion_problem_l479_47952


namespace NUMINAMATH_CALUDE_income_expenditure_ratio_l479_47927

/-- Given a person's income and savings, prove the ratio of income to expenditure -/
theorem income_expenditure_ratio 
  (income : ℕ) 
  (savings : ℕ) 
  (h1 : income = 36000) 
  (h2 : savings = 4000) :
  (income : ℚ) / (income - savings) = 9 / 8 := by
  sorry

end NUMINAMATH_CALUDE_income_expenditure_ratio_l479_47927


namespace NUMINAMATH_CALUDE_five_circles_common_point_l479_47938

-- Define a circle type
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

-- Define a point type
structure Point where
  x : ℝ
  y : ℝ

-- Define a function to check if a point is on a circle
def pointOnCircle (p : Point) (c : Circle) : Prop :=
  (p.x - c.center.1)^2 + (p.y - c.center.2)^2 = c.radius^2

-- Define a function to check if four circles pass through a single point
def fourCirclesCommonPoint (c1 c2 c3 c4 : Circle) : Prop :=
  ∃ p : Point, pointOnCircle p c1 ∧ pointOnCircle p c2 ∧ pointOnCircle p c3 ∧ pointOnCircle p c4

-- Theorem statement
theorem five_circles_common_point 
  (c1 c2 c3 c4 c5 : Circle) 
  (h1234 : fourCirclesCommonPoint c1 c2 c3 c4)
  (h1235 : fourCirclesCommonPoint c1 c2 c3 c5)
  (h1245 : fourCirclesCommonPoint c1 c2 c4 c5)
  (h1345 : fourCirclesCommonPoint c1 c3 c4 c5)
  (h2345 : fourCirclesCommonPoint c2 c3 c4 c5) :
  ∃ p : Point, pointOnCircle p c1 ∧ pointOnCircle p c2 ∧ pointOnCircle p c3 ∧ pointOnCircle p c4 ∧ pointOnCircle p c5 :=
by
  sorry

end NUMINAMATH_CALUDE_five_circles_common_point_l479_47938


namespace NUMINAMATH_CALUDE_percentage_of_women_in_non_union_l479_47925

theorem percentage_of_women_in_non_union (total_employees : ℝ) 
  (h1 : total_employees > 0)
  (h2 : ∃ p : ℝ, 0 ≤ p ∧ p ≤ 1 ∧ p * total_employees = number_of_male_employees)
  (h3 : 0.6 * total_employees = number_of_unionized_employees)
  (h4 : 0.7 * number_of_unionized_employees = number_of_male_unionized_employees)
  (h5 : 0.9 * (total_employees - number_of_unionized_employees) = number_of_female_non_unionized_employees) :
  (number_of_female_non_unionized_employees / (total_employees - number_of_unionized_employees)) = 0.9 := by
sorry


end NUMINAMATH_CALUDE_percentage_of_women_in_non_union_l479_47925


namespace NUMINAMATH_CALUDE_z_in_fourth_quadrant_l479_47958

-- Define the complex number z
def z : ℂ := (1 - Complex.I) * (3 + Complex.I)

-- Theorem stating that z is in the fourth quadrant
theorem z_in_fourth_quadrant :
  Real.sign (z.re) = 1 ∧ Real.sign (z.im) = -1 :=
sorry

end NUMINAMATH_CALUDE_z_in_fourth_quadrant_l479_47958


namespace NUMINAMATH_CALUDE_fraction_sum_equality_l479_47996

theorem fraction_sum_equality (a b : ℝ) (ha : a ≠ 0) :
  1 / (2 * a * b) + b / (4 * a) = (2 + b^2) / (4 * a * b) := by
  sorry

end NUMINAMATH_CALUDE_fraction_sum_equality_l479_47996


namespace NUMINAMATH_CALUDE_circle_division_theorem_l479_47983

/-- The number of regions a circle is divided into by radii and concentric circles -/
def num_regions (num_radii : ℕ) (num_concentric_circles : ℕ) : ℕ :=
  (num_concentric_circles + 1) * num_radii

/-- Theorem: A circle with 16 radii and 10 concentric circles is divided into 176 regions -/
theorem circle_division_theorem :
  num_regions 16 10 = 176 := by
  sorry

end NUMINAMATH_CALUDE_circle_division_theorem_l479_47983


namespace NUMINAMATH_CALUDE_square_calculation_identity_l479_47962

theorem square_calculation_identity (x : ℝ) : ((x + 1)^3 - (x - 1)^3 - 2) / 6 = x^2 := by
  sorry

end NUMINAMATH_CALUDE_square_calculation_identity_l479_47962


namespace NUMINAMATH_CALUDE_perpendicular_vectors_m_l479_47910

-- Define the vectors
def a : Fin 2 → ℝ := ![1, -3]
def b (m : ℝ) : Fin 2 → ℝ := ![m, 2]

-- Define the dot product
def dot_product (u v : Fin 2 → ℝ) : ℝ :=
  (u 0) * (v 0) + (u 1) * (v 1)

-- Define perpendicularity
def perpendicular (u v : Fin 2 → ℝ) : Prop :=
  dot_product u v = 0

-- State the theorem
theorem perpendicular_vectors_m (m : ℝ) :
  perpendicular a (fun i => a i + b m i) → m = -4 := by
  sorry

end NUMINAMATH_CALUDE_perpendicular_vectors_m_l479_47910


namespace NUMINAMATH_CALUDE_hockey_league_face_count_l479_47987

/-- The number of times each team faces all other teams in a hockey league -/
def face_count (num_teams : ℕ) (total_games : ℕ) : ℕ :=
  total_games / (num_teams * (num_teams - 1) / 2)

/-- Theorem: In a hockey league with 18 teams, where each team faces all other teams
    the same number of times, and a total of 1530 games are played in the season,
    each team faces all the other teams 5 times. -/
theorem hockey_league_face_count :
  face_count 18 1530 = 5 := by
  sorry

end NUMINAMATH_CALUDE_hockey_league_face_count_l479_47987


namespace NUMINAMATH_CALUDE_tangent_line_cubic_function_l479_47949

/-- Given a cubic function f(x) = ax³ - 2x passing through the point (-1, 4),
    this theorem states that the equation of the tangent line to y = f(x) at x = -1
    is 8x + y + 4 = 0. -/
theorem tangent_line_cubic_function (a : ℝ) :
  let f : ℝ → ℝ := λ x ↦ a * x^3 - 2*x
  f (-1) = 4 →
  let m : ℝ := (6 * (-1)^2 + 2)  -- Derivative of f at x = -1
  let tangent_line : ℝ → ℝ := λ x ↦ m * (x - (-1)) + f (-1)
  ∀ x y, y = tangent_line x ↔ 8*x + y + 4 = 0 := by
sorry


end NUMINAMATH_CALUDE_tangent_line_cubic_function_l479_47949


namespace NUMINAMATH_CALUDE_division_remainder_l479_47906

/-- The divisor polynomial -/
def divisor (x : ℂ) : ℂ := x^5 + x^4 + x^3 + x^2 + x + 1

/-- The dividend polynomial -/
def dividend (x : ℂ) : ℂ := x^60 + x^45 + x^30 + x^15 + 1

/-- Theorem stating that the remainder of the division is 5 -/
theorem division_remainder : ∃ (q : ℂ → ℂ), ∀ (x : ℂ), 
  dividend x = (divisor x) * (q x) + 5 := by
  sorry

end NUMINAMATH_CALUDE_division_remainder_l479_47906


namespace NUMINAMATH_CALUDE_largest_mersenne_prime_under_500_l479_47930

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d > 1 → d < n → ¬(n % d = 0)

def mersenne_prime (p : ℕ) : Prop := is_prime p ∧ ∃ n : ℕ, is_prime n ∧ p = 2^n - 1

theorem largest_mersenne_prime_under_500 :
  ∃ p : ℕ, mersenne_prime p ∧ p < 500 ∧ ∀ q : ℕ, mersenne_prime q → q < 500 → q ≤ p :=
by sorry

end NUMINAMATH_CALUDE_largest_mersenne_prime_under_500_l479_47930


namespace NUMINAMATH_CALUDE_fraction_sum_inequality_l479_47940

theorem fraction_sum_inequality (a b c d n : ℕ) 
  (h1 : a + c < n) 
  (h2 : (a : ℚ) / b + (c : ℚ) / d < 1) : 
  (a : ℚ) / b + (c : ℚ) / d < 1 - 1 / (n^3 : ℚ) := by
  sorry

end NUMINAMATH_CALUDE_fraction_sum_inequality_l479_47940


namespace NUMINAMATH_CALUDE_derivative_of_y_l479_47929

-- Define the function y
def y (x : ℝ) : ℝ := 2 * x^2 - 2 * x + 1

-- State the theorem
theorem derivative_of_y (x : ℝ) : 
  deriv y x = 4 * x - 2 := by sorry

end NUMINAMATH_CALUDE_derivative_of_y_l479_47929


namespace NUMINAMATH_CALUDE_middle_school_math_club_payment_l479_47988

theorem middle_school_math_club_payment (A : Nat) : 
  A < 10 → (100 + 10 * A + 2) % 11 = 0 ↔ A = 3 := by
  sorry

end NUMINAMATH_CALUDE_middle_school_math_club_payment_l479_47988


namespace NUMINAMATH_CALUDE_vinegar_percentage_second_brand_l479_47960

/-- Calculates the vinegar percentage in the second brand of Italian dressing -/
theorem vinegar_percentage_second_brand 
  (total_volume : ℝ) 
  (desired_vinegar_percentage : ℝ) 
  (first_brand_volume : ℝ) 
  (second_brand_volume : ℝ) 
  (first_brand_vinegar_percentage : ℝ)
  (h1 : total_volume = 320)
  (h2 : desired_vinegar_percentage = 11)
  (h3 : first_brand_volume = 128)
  (h4 : second_brand_volume = 128)
  (h5 : first_brand_vinegar_percentage = 8) :
  ∃ (second_brand_vinegar_percentage : ℝ),
    second_brand_vinegar_percentage = 19.5 ∧
    (first_brand_volume * first_brand_vinegar_percentage / 100 + 
     second_brand_volume * second_brand_vinegar_percentage / 100) / total_volume * 100 = 
    desired_vinegar_percentage :=
by sorry

end NUMINAMATH_CALUDE_vinegar_percentage_second_brand_l479_47960


namespace NUMINAMATH_CALUDE_solve_system_l479_47978

theorem solve_system (y z x : ℚ) 
  (h1 : (2 : ℚ) / 3 = y / 90)
  (h2 : (2 : ℚ) / 3 = (y + z) / 120)
  (h3 : (2 : ℚ) / 3 = (x - z) / 150) : 
  x = 120 := by sorry

end NUMINAMATH_CALUDE_solve_system_l479_47978


namespace NUMINAMATH_CALUDE_multiply_six_and_mixed_number_l479_47934

theorem multiply_six_and_mixed_number : 6 * (8 + 1/3) = 50 := by
  sorry

end NUMINAMATH_CALUDE_multiply_six_and_mixed_number_l479_47934


namespace NUMINAMATH_CALUDE_disinfectant_purchase_theorem_l479_47968

/-- Represents the cost and quantity of disinfectants --/
structure DisinfectantPurchase where
  costA : ℕ  -- Cost of one bottle of Class A disinfectant
  costB : ℕ  -- Cost of one bottle of Class B disinfectant
  quantityA : ℕ  -- Number of bottles of Class A disinfectant
  quantityB : ℕ  -- Number of bottles of Class B disinfectant

/-- Theorem about disinfectant purchase --/
theorem disinfectant_purchase_theorem 
  (purchase : DisinfectantPurchase)
  (total_cost : purchase.costA * purchase.quantityA + purchase.costB * purchase.quantityB = 2250)
  (cost_difference : purchase.costA + 15 = purchase.costB)
  (quantities : purchase.quantityA = 80 ∧ purchase.quantityB = 35)
  (new_total : ℕ)
  (new_budget : new_total * purchase.costA + (50 - new_total) * purchase.costB ≤ 1200)
  : purchase.costA = 15 ∧ purchase.costB = 30 ∧ new_total ≥ 20 := by
  sorry

#check disinfectant_purchase_theorem

end NUMINAMATH_CALUDE_disinfectant_purchase_theorem_l479_47968


namespace NUMINAMATH_CALUDE_square_and_sqrt_problem_l479_47900

theorem square_and_sqrt_problem :
  let a : ℕ := 101
  let b : ℕ := 10101
  let c : ℕ := 102030405060504030201
  (a ^ 2 = 10201) ∧
  (b ^ 2 = 102030201) ∧
  (c = 10101010101 ^ 2) := by
  sorry

end NUMINAMATH_CALUDE_square_and_sqrt_problem_l479_47900


namespace NUMINAMATH_CALUDE_condition_relationship_l479_47921

theorem condition_relationship : 
  (∀ x : ℝ, (0 < x ∧ x < 1) → x^2 < 1) ∧ 
  (∃ x : ℝ, x^2 < 1 ∧ ¬(0 < x ∧ x < 1)) := by
sorry

end NUMINAMATH_CALUDE_condition_relationship_l479_47921


namespace NUMINAMATH_CALUDE_base_twelve_representation_l479_47979

def is_three_digit (n : ℕ) (b : ℕ) : Prop :=
  b ^ 2 ≤ n ∧ n < b ^ 3

def has_odd_final_digit (n : ℕ) (b : ℕ) : Prop :=
  n % b % 2 = 1

theorem base_twelve_representation : 
  is_three_digit 125 12 ∧ has_odd_final_digit 125 12 ∧ 
  ∀ b : ℕ, b ≠ 12 → ¬(is_three_digit 125 b ∧ has_odd_final_digit 125 b) :=
sorry

end NUMINAMATH_CALUDE_base_twelve_representation_l479_47979


namespace NUMINAMATH_CALUDE_inequality_solution_l479_47975

theorem inequality_solution (x : ℝ) : (x + 2) / (x + 4) ≤ 3 ↔ -5 < x ∧ x < -4 := by
  sorry

end NUMINAMATH_CALUDE_inequality_solution_l479_47975


namespace NUMINAMATH_CALUDE_modulus_of_imaginary_unit_l479_47901

theorem modulus_of_imaginary_unit (z : ℂ) (h : z^2 + 1 = 0) : Complex.abs z = 1 := by
  sorry

end NUMINAMATH_CALUDE_modulus_of_imaginary_unit_l479_47901


namespace NUMINAMATH_CALUDE_sports_club_members_l479_47941

theorem sports_club_members (badminton tennis both neither : ℕ) 
  (h1 : badminton = 18)
  (h2 : tennis = 19)
  (h3 : both = 9)
  (h4 : neither = 2) :
  badminton + tennis - both + neither = 30 := by
  sorry

end NUMINAMATH_CALUDE_sports_club_members_l479_47941


namespace NUMINAMATH_CALUDE_xyz_value_is_ten_l479_47916

-- Define the variables
variable (a b c x y z : ℂ)

-- State the theorem
theorem xyz_value_is_ten
  (h1 : a ≠ 0)
  (h2 : b ≠ 0)
  (h3 : c ≠ 0)
  (h4 : x ≠ 0)
  (h5 : y ≠ 0)
  (h6 : z ≠ 0)
  (h7 : a = (b + c) / (x - 3))
  (h8 : b = (a + c) / (y - 3))
  (h9 : c = (a + b) / (z - 3))
  (h10 : x * y + x * z + y * z = 9)
  (h11 : x + y + z = 6) :
  x * y * z = 10 := by
sorry


end NUMINAMATH_CALUDE_xyz_value_is_ten_l479_47916


namespace NUMINAMATH_CALUDE_opposite_numbers_l479_47990

-- Define the concept of opposite numbers
def are_opposite (a b : ℝ) : Prop := a = -b

-- Theorem statement
theorem opposite_numbers : are_opposite (-|-(1/100)|) (-(-1/100)) := by
  sorry

end NUMINAMATH_CALUDE_opposite_numbers_l479_47990


namespace NUMINAMATH_CALUDE_intersection_points_on_circle_l479_47993

/-- The parabolas y = (x - 2)^2 and x - 5 = (y + 1)^2 intersect at four points that lie on a circle with radius squared equal to 1.5 -/
theorem intersection_points_on_circle :
  ∃ (c : ℝ × ℝ) (r : ℝ),
    (∀ (p : ℝ × ℝ), 
      (p.2 = (p.1 - 2)^2 ∧ p.1 - 5 = (p.2 + 1)^2) →
      (p.1 - c.1)^2 + (p.2 - c.2)^2 = r^2) ∧
    r^2 = 1.5 := by
  sorry

end NUMINAMATH_CALUDE_intersection_points_on_circle_l479_47993


namespace NUMINAMATH_CALUDE_perimeter_after_adding_tiles_l479_47926

/-- Represents a configuration of square tiles -/
structure TileConfiguration where
  length : ℕ
  width : ℕ
  perimeter : ℕ

/-- The initial configuration of tiles -/
def initial_config : TileConfiguration :=
  { length := 6, width := 1, perimeter := 14 }

/-- Calculates the new perimeter after adding tiles -/
def new_perimeter (config : TileConfiguration) (added_tiles : ℕ) : ℕ :=
  2 * (config.length + added_tiles) + 2 * config.width

/-- Theorem stating that adding two tiles results in a perimeter of 18 -/
theorem perimeter_after_adding_tiles :
  new_perimeter initial_config 2 = 18 := by
  sorry

#eval new_perimeter initial_config 2

end NUMINAMATH_CALUDE_perimeter_after_adding_tiles_l479_47926


namespace NUMINAMATH_CALUDE_arithmetic_sequence_general_term_l479_47998

/-- An arithmetic sequence with its sum function -/
structure ArithmeticSequence where
  a : ℕ → ℝ  -- The sequence
  S : ℕ → ℝ  -- The sum function
  sum_formula : ∀ n, S n = n * (2 * a 1 + (n - 1) * (a 2 - a 1)) / 2
  arithmetic : ∀ n, a (n + 1) - a n = a 2 - a 1

/-- The theorem stating the general term of the arithmetic sequence -/
theorem arithmetic_sequence_general_term 
  (seq : ArithmeticSequence) 
  (sum10 : seq.S 10 = 10) 
  (sum20 : seq.S 20 = 220) : 
  ∀ n, seq.a n = 2 * n - 10 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_general_term_l479_47998


namespace NUMINAMATH_CALUDE_set_union_problem_l479_47905

theorem set_union_problem (A B : Set ℕ) (a : ℕ) :
  A = {2, 3} →
  B = {1, a} →
  A ∩ B = {2} →
  A ∪ B = {1, 2, 3} := by
sorry

end NUMINAMATH_CALUDE_set_union_problem_l479_47905


namespace NUMINAMATH_CALUDE_is_circle_center_l479_47961

/-- The equation of a circle in the xy-plane -/
def circle_equation (x y : ℝ) : Prop :=
  x^2 + y^2 - 2*x + 4*y + 3 = 0

/-- The center of the circle -/
def circle_center : ℝ × ℝ := (1, -2)

/-- Theorem stating that the given point is the center of the circle -/
theorem is_circle_center :
  ∀ x y : ℝ, circle_equation x y ↔ (x - circle_center.1)^2 + (y - circle_center.2)^2 = 2 :=
by sorry

end NUMINAMATH_CALUDE_is_circle_center_l479_47961


namespace NUMINAMATH_CALUDE_smallest_positive_integer_with_given_remainders_l479_47972

theorem smallest_positive_integer_with_given_remainders :
  ∃ n : ℕ, n > 0 ∧
    n % 3 = 1 ∧
    n % 4 = 2 ∧
    n % 5 = 3 ∧
    ∀ m : ℕ, m > 0 →
      m % 3 = 1 →
      m % 4 = 2 →
      m % 5 = 3 →
      n ≤ m :=
by
  use 58
  sorry

end NUMINAMATH_CALUDE_smallest_positive_integer_with_given_remainders_l479_47972


namespace NUMINAMATH_CALUDE_average_of_x_and_y_is_16_l479_47902

theorem average_of_x_and_y_is_16 (x y : ℝ) : 
  3 = 0.15 * x → 3 = 0.25 * y → (x + y) / 2 = 16 := by
  sorry

end NUMINAMATH_CALUDE_average_of_x_and_y_is_16_l479_47902


namespace NUMINAMATH_CALUDE_square_diff_minus_diff_squares_l479_47991

theorem square_diff_minus_diff_squares (x y : ℝ) :
  (x - y)^2 - (x^2 - y^2) = (x - y)^2 - (x^2 - y^2) := by
  sorry

end NUMINAMATH_CALUDE_square_diff_minus_diff_squares_l479_47991


namespace NUMINAMATH_CALUDE_decorative_band_length_l479_47936

/-- The length of a decorative band for a circular sign -/
theorem decorative_band_length :
  let π : ℚ := 22 / 7
  let area : ℚ := 616
  let extra_length : ℚ := 5
  let radius : ℚ := (area / π).sqrt
  let circumference : ℚ := 2 * π * radius
  let band_length : ℚ := circumference + extra_length
  band_length = 93 := by sorry

end NUMINAMATH_CALUDE_decorative_band_length_l479_47936


namespace NUMINAMATH_CALUDE_negation_equivalence_l479_47924

theorem negation_equivalence :
  (¬ ∃ x₀ : ℝ, x₀^3 - x₀^2 + 1 > 0) ↔ (∀ x : ℝ, x^3 - x^2 + 1 ≤ 0) :=
by sorry

end NUMINAMATH_CALUDE_negation_equivalence_l479_47924


namespace NUMINAMATH_CALUDE_odd_sum_of_odd_square_plus_cube_l479_47943

theorem odd_sum_of_odd_square_plus_cube (n m : ℤ) : 
  Odd (n^2 + m^3) → Odd (n + m) := by
  sorry

end NUMINAMATH_CALUDE_odd_sum_of_odd_square_plus_cube_l479_47943


namespace NUMINAMATH_CALUDE_lucy_groceries_l479_47964

theorem lucy_groceries (cookies : ℕ) (cake : ℕ) : 
  cookies = 2 → cake = 12 → cookies + cake = 14 := by
  sorry

end NUMINAMATH_CALUDE_lucy_groceries_l479_47964


namespace NUMINAMATH_CALUDE_arctg_sum_eq_pi_fourth_l479_47956

theorem arctg_sum_eq_pi_fourth (x : ℝ) (h : x > -1) : 
  Real.arctan x + Real.arctan ((1 - x) / (1 + x)) = π / 4 := by
  sorry

end NUMINAMATH_CALUDE_arctg_sum_eq_pi_fourth_l479_47956


namespace NUMINAMATH_CALUDE_book_pages_problem_l479_47971

theorem book_pages_problem (x : ℕ) (h1 : x > 0) (h2 : x + (x + 1) = 125) : x + 1 = 63 := by
  sorry

end NUMINAMATH_CALUDE_book_pages_problem_l479_47971


namespace NUMINAMATH_CALUDE_sample_size_is_40_l479_47909

/-- Represents a frequency distribution histogram -/
structure Histogram where
  num_bars : ℕ
  central_freq : ℕ
  other_freq : ℕ

/-- Calculates the sample size of a histogram -/
def sample_size (h : Histogram) : ℕ :=
  h.central_freq + h.other_freq

/-- Theorem stating the sample size for the given histogram -/
theorem sample_size_is_40 (h : Histogram) 
  (h_bars : h.num_bars = 7)
  (h_central : h.central_freq = 8)
  (h_ratio : h.central_freq = h.other_freq / 4) :
  sample_size h = 40 := by
  sorry

#check sample_size_is_40

end NUMINAMATH_CALUDE_sample_size_is_40_l479_47909


namespace NUMINAMATH_CALUDE_right_triangle_area_l479_47932

/-- Given a right triangle ABC with legs a and b, and hypotenuse c,
    if a + b = 21 and c = 15, then the area of triangle ABC is 54. -/
theorem right_triangle_area (a b c : ℝ) : 
  a > 0 → b > 0 → c > 0 → 
  a + b = 21 → 
  c = 15 → 
  a^2 + b^2 = c^2 →
  (1/2) * a * b = 54 := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_area_l479_47932


namespace NUMINAMATH_CALUDE_journey_distance_l479_47951

theorem journey_distance (total_time : ℝ) (speed1 : ℝ) (speed2 : ℝ) :
  total_time = 10 ∧ speed1 = 21 ∧ speed2 = 24 →
  ∃ (distance : ℝ), distance = 224 ∧
    total_time = (distance / 2) / speed1 + (distance / 2) / speed2 := by
  sorry

end NUMINAMATH_CALUDE_journey_distance_l479_47951


namespace NUMINAMATH_CALUDE_ac_eq_b_squared_necessary_not_sufficient_l479_47994

/-- Definition of a geometric progression for three real numbers -/
def isGeometricProgression (a b c : ℝ) : Prop :=
  ∃ r : ℝ, b = a * r ∧ c = b * r

/-- The main theorem stating that ac = b^2 is necessary but not sufficient for a, b, c to be in geometric progression -/
theorem ac_eq_b_squared_necessary_not_sufficient :
  (∀ a b c : ℝ, isGeometricProgression a b c → a * c = b^2) ∧
  (∃ a b c : ℝ, a * c = b^2 ∧ ¬isGeometricProgression a b c) := by
  sorry

end NUMINAMATH_CALUDE_ac_eq_b_squared_necessary_not_sufficient_l479_47994


namespace NUMINAMATH_CALUDE_sin_210_degrees_l479_47914

theorem sin_210_degrees : Real.sin (210 * π / 180) = -1/2 := by
  sorry

end NUMINAMATH_CALUDE_sin_210_degrees_l479_47914


namespace NUMINAMATH_CALUDE_complex_modulus_problem_l479_47944

theorem complex_modulus_problem (z : ℂ) (a : ℝ) : 
  z = a * Complex.I → 
  (Complex.re ((1 + z) * (1 + Complex.I)) = (1 + z) * (1 + Complex.I)) → 
  Complex.abs (z + 2) = Real.sqrt 5 := by
sorry

end NUMINAMATH_CALUDE_complex_modulus_problem_l479_47944


namespace NUMINAMATH_CALUDE_solution_set_inequality_l479_47917

theorem solution_set_inequality (x : ℝ) : 
  (x - 1) / (2 * x + 1) ≤ 0 ↔ -1/2 < x ∧ x ≤ 1 := by sorry

end NUMINAMATH_CALUDE_solution_set_inequality_l479_47917


namespace NUMINAMATH_CALUDE_sum_of_coefficients_l479_47954

theorem sum_of_coefficients (a₀ a₁ a₂ a₃ a₄ a₅ : ℝ) :
  (∀ x, (1 - 2*x)^5 = a₀ + a₁*x + a₂*x^2 + a₃*x^3 + a₄*x^4 + a₅*x^5) →
  a₁ + a₂ + a₃ + a₄ + a₅ = -2 := by
sorry

end NUMINAMATH_CALUDE_sum_of_coefficients_l479_47954


namespace NUMINAMATH_CALUDE_cycle_gain_percent_l479_47974

def gain_percent (cost_price selling_price : ℚ) : ℚ :=
  (selling_price - cost_price) / cost_price * 100

theorem cycle_gain_percent :
  let cost_price : ℚ := 900
  let selling_price : ℚ := 1150
  gain_percent cost_price selling_price = (1150 - 900) / 900 * 100 := by
  sorry

end NUMINAMATH_CALUDE_cycle_gain_percent_l479_47974


namespace NUMINAMATH_CALUDE_grid_size_for_2017_colored_squares_l479_47997

/-- Represents a square grid -/
structure SquareGrid where
  size : ℕ

/-- The number of colored squares on the two longest diagonals of a square grid -/
def coloredSquares (grid : SquareGrid) : ℕ := 2 * grid.size - 1

theorem grid_size_for_2017_colored_squares :
  ∃ (grid : SquareGrid), coloredSquares grid = 2017 ∧ grid.size = 1009 :=
sorry

end NUMINAMATH_CALUDE_grid_size_for_2017_colored_squares_l479_47997


namespace NUMINAMATH_CALUDE_jacket_price_reduction_l479_47986

theorem jacket_price_reduction (x : ℝ) : 
  (1 - x / 100) * (1 - 0.15) * (1 + 56.86274509803921 / 100) = 1 → x = 25 := by
  sorry

end NUMINAMATH_CALUDE_jacket_price_reduction_l479_47986


namespace NUMINAMATH_CALUDE_stating_min_connections_for_given_problem_l479_47981

/-- Represents the number of cities -/
def num_cities : Nat := 100

/-- Represents the number of different routes -/
def num_routes : Nat := 1000

/-- 
Given a number of cities and a number of routes, 
calculates the minimum number of flight connections per city 
that allows for the specified number of routes.
-/
def min_connections (cities : Nat) (routes : Nat) : Nat :=
  sorry

/-- 
Theorem stating that given 100 cities and 1000 routes, 
the minimum number of connections per city is 4.
-/
theorem min_connections_for_given_problem : 
  min_connections num_cities num_routes = 4 := by sorry

end NUMINAMATH_CALUDE_stating_min_connections_for_given_problem_l479_47981


namespace NUMINAMATH_CALUDE_days_A_worked_alone_l479_47911

/-- Represents the number of days it takes for A and B to finish the work together -/
def total_days_together : ℝ := 40

/-- Represents the number of days it takes for A to finish the work alone -/
def total_days_A : ℝ := 28

/-- Represents the number of days A and B worked together before B left -/
def days_worked_together : ℝ := 10

/-- Represents the total amount of work to be done -/
def total_work : ℝ := 1

theorem days_A_worked_alone :
  let remaining_work := total_work - (days_worked_together / total_days_together)
  let days_A_alone := remaining_work * total_days_A
  days_A_alone = 21 := by
sorry

end NUMINAMATH_CALUDE_days_A_worked_alone_l479_47911


namespace NUMINAMATH_CALUDE_log_equation_solution_l479_47946

theorem log_equation_solution :
  ∀ x : ℝ, (Real.log x - 3 * Real.log 4 = -3) → x = 0.064 := by
  sorry

end NUMINAMATH_CALUDE_log_equation_solution_l479_47946


namespace NUMINAMATH_CALUDE_square_root_and_abs_simplification_l479_47976

theorem square_root_and_abs_simplification :
  Real.sqrt ((-2)^2) + |Real.sqrt 2 - Real.sqrt 3| - |Real.sqrt 3 - 1| = 3 - Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_square_root_and_abs_simplification_l479_47976


namespace NUMINAMATH_CALUDE_find_number_l479_47957

theorem find_number (A B : ℕ) (h1 : B = 913) (h2 : Nat.lcm A B = 2310) (h3 : Nat.gcd A B = 83) : A = 210 := by
  sorry

end NUMINAMATH_CALUDE_find_number_l479_47957


namespace NUMINAMATH_CALUDE_equal_roots_quadratic_l479_47950

/-- 
Given a quadratic equation 3x^2 + 6x + m = 0, if it has two equal real roots,
then m = 3.
-/
theorem equal_roots_quadratic (m : ℝ) : 
  (∃ x : ℝ, 3 * x^2 + 6 * x + m = 0 ∧ 
   ∀ y : ℝ, 3 * y^2 + 6 * y + m = 0 → y = x) → 
  m = 3 := by sorry

end NUMINAMATH_CALUDE_equal_roots_quadratic_l479_47950


namespace NUMINAMATH_CALUDE_expression_evaluation_l479_47969

theorem expression_evaluation :
  let x : ℝ := (1/2)^2023
  let y : ℝ := 2^2022
  (2*x + y)^2 - (2*x + y)*(2*x - y) - 2*y*(x + y) = 1 := by
sorry

end NUMINAMATH_CALUDE_expression_evaluation_l479_47969


namespace NUMINAMATH_CALUDE_least_reducible_fraction_l479_47953

theorem least_reducible_fraction :
  ∃ (n : ℕ), n > 0 ∧ 
  (∀ m : ℕ, m > 0 → m < n → ¬(∃ k : ℕ, k > 1 ∧ k ∣ (m + 17) ∧ k ∣ (7*m - 9))) ∧
  (∃ k : ℕ, k > 1 ∧ k ∣ (n + 17) ∧ k ∣ (7*n - 9)) ∧
  n = 1 :=
sorry

end NUMINAMATH_CALUDE_least_reducible_fraction_l479_47953


namespace NUMINAMATH_CALUDE_cookies_per_pan_l479_47942

theorem cookies_per_pan (total_pans : ℕ) (total_cookies : ℕ) (h1 : total_pans = 5) (h2 : total_cookies = 40) :
  total_cookies / total_pans = 8 := by
  sorry

end NUMINAMATH_CALUDE_cookies_per_pan_l479_47942


namespace NUMINAMATH_CALUDE_special_cubic_e_value_l479_47963

/-- A cubic polynomial with specific properties -/
structure SpecialCubic where
  d : ℝ
  e : ℝ
  zeros_mean_prod : (- d / 9) = 2 * (-4)
  coeff_sum_y_intercept : 3 + d + e + 12 = 12

/-- The value of e in the special cubic polynomial is -75 -/
theorem special_cubic_e_value (p : SpecialCubic) : p.e = -75 := by
  sorry

end NUMINAMATH_CALUDE_special_cubic_e_value_l479_47963


namespace NUMINAMATH_CALUDE_math_class_size_l479_47908

/-- Proves that the number of students in the mathematics class is 48 given the conditions --/
theorem math_class_size (total : ℕ) (physics : ℕ) (math : ℕ) (both : ℕ) : 
  total = 56 →
  math = 4 * physics →
  both = 8 →
  total = physics + math - both →
  math = 48 := by
sorry

end NUMINAMATH_CALUDE_math_class_size_l479_47908


namespace NUMINAMATH_CALUDE_distance_between_homes_is_40_l479_47945

/-- The distance between Maxwell's and Brad's homes -/
def distance_between_homes : ℝ := 40

/-- Maxwell's walking speed in km/h -/
def maxwell_speed : ℝ := 3

/-- Brad's running speed in km/h -/
def brad_speed : ℝ := 5

/-- The distance Maxwell travels before they meet -/
def maxwell_distance : ℝ := 15

/-- Theorem stating that the distance between homes is 40 km -/
theorem distance_between_homes_is_40 :
  distance_between_homes = maxwell_distance * (maxwell_speed + brad_speed) / maxwell_speed :=
by sorry

end NUMINAMATH_CALUDE_distance_between_homes_is_40_l479_47945


namespace NUMINAMATH_CALUDE_modified_cube_painted_faces_l479_47955

/-- Represents a cube with its 8 corner small cubes removed and its surface painted -/
structure ModifiedCube where
  size : ℕ
  corner_removed : Bool
  surface_painted : Bool

/-- Counts the number of small cubes with a given number of painted faces -/
def count_painted_faces (c : ModifiedCube) (n : ℕ) : ℕ :=
  sorry

/-- The main theorem about the number of painted faces in a modified cube -/
theorem modified_cube_painted_faces (c : ModifiedCube) 
  (h1 : c.size > 2) 
  (h2 : c.corner_removed = true) 
  (h3 : c.surface_painted = true) : 
  (count_painted_faces c 4 = 12) ∧ 
  (count_painted_faces c 1 = 6) ∧ 
  (count_painted_faces c 0 = 1) :=
sorry

end NUMINAMATH_CALUDE_modified_cube_painted_faces_l479_47955


namespace NUMINAMATH_CALUDE_prisoner_selection_l479_47966

/-- Given 25 prisoners, prove the number of ways to choose 3 in order and without order. -/
theorem prisoner_selection (n : ℕ) (h : n = 25) : 
  (n * (n - 1) * (n - 2) = 13800) ∧ (Nat.choose n 3 = 2300) := by
  sorry

end NUMINAMATH_CALUDE_prisoner_selection_l479_47966


namespace NUMINAMATH_CALUDE_abs_value_of_root_l479_47967

theorem abs_value_of_root (z : ℂ) : z^2 - 2*z + 2 = 0 → Complex.abs z = Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_abs_value_of_root_l479_47967


namespace NUMINAMATH_CALUDE_set_equality_implies_value_l479_47903

theorem set_equality_implies_value (a b : ℝ) : 
  ({a, b/a, 1} : Set ℝ) = {a^2, a+b, 0} → a^2012 + b^2012 = 1 :=
by sorry

end NUMINAMATH_CALUDE_set_equality_implies_value_l479_47903


namespace NUMINAMATH_CALUDE_multiple_of_q_in_equation_l479_47995

theorem multiple_of_q_in_equation (p q m : ℚ) 
  (h1 : p / q = 3 / 4)
  (h2 : 3 * p + m * q = 25 / 4) :
  m = 4 := by
sorry

end NUMINAMATH_CALUDE_multiple_of_q_in_equation_l479_47995


namespace NUMINAMATH_CALUDE_nina_total_spent_l479_47935

/-- The total amount spent by Nina on toys, basketball cards, and shirts. -/
def total_spent (toy_quantity : ℕ) (toy_price : ℕ) (card_quantity : ℕ) (card_price : ℕ) (shirt_quantity : ℕ) (shirt_price : ℕ) : ℕ :=
  toy_quantity * toy_price + card_quantity * card_price + shirt_quantity * shirt_price

/-- Theorem stating that Nina's total spent is $70 -/
theorem nina_total_spent :
  total_spent 3 10 2 5 5 6 = 70 := by
  sorry

end NUMINAMATH_CALUDE_nina_total_spent_l479_47935


namespace NUMINAMATH_CALUDE_tomato_seeds_proof_l479_47984

/-- The number of tomato seeds planted by Mike and Ted -/
def total_seeds (mike_morning mike_afternoon ted_morning ted_afternoon : ℕ) : ℕ :=
  mike_morning + mike_afternoon + ted_morning + ted_afternoon

theorem tomato_seeds_proof :
  ∀ (mike_morning mike_afternoon ted_morning ted_afternoon : ℕ),
    mike_morning = 50 →
    ted_morning = 2 * mike_morning →
    mike_afternoon = 60 →
    ted_afternoon = mike_afternoon - 20 →
    total_seeds mike_morning mike_afternoon ted_morning ted_afternoon = 250 := by
  sorry

end NUMINAMATH_CALUDE_tomato_seeds_proof_l479_47984


namespace NUMINAMATH_CALUDE_steve_pencil_theorem_l479_47931

def steve_pencil_problem (boxes : ℕ) (pencils_per_box : ℕ) (lauren_pencils : ℕ) (matt_extra_pencils : ℕ) : Prop :=
  let total_pencils := boxes * pencils_per_box
  let matt_pencils := lauren_pencils + matt_extra_pencils
  let given_away_pencils := lauren_pencils + matt_pencils
  let remaining_pencils := total_pencils - given_away_pencils
  remaining_pencils = 9

theorem steve_pencil_theorem :
  steve_pencil_problem 2 12 6 3 :=
by
  sorry

end NUMINAMATH_CALUDE_steve_pencil_theorem_l479_47931


namespace NUMINAMATH_CALUDE_opposite_of_2023_l479_47948

/-- The opposite of a number is the number that, when added to the original number, results in zero. -/
def opposite (n : ℤ) : ℤ := -n

/-- The opposite of 2023 is -2023. -/
theorem opposite_of_2023 : opposite 2023 = -2023 := by sorry

end NUMINAMATH_CALUDE_opposite_of_2023_l479_47948


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l479_47965

def set_A : Set ℝ := {x | 2 * x + 1 > 0}
def set_B : Set ℝ := {x | |x - 1| < 2}

theorem intersection_of_A_and_B :
  set_A ∩ set_B = {x | -1/2 < x ∧ x < 3} := by sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l479_47965


namespace NUMINAMATH_CALUDE_abs_complex_fraction_equals_sqrt_two_l479_47992

/-- The absolute value of the complex number (1-3i)/(1+2i) is equal to √2 -/
theorem abs_complex_fraction_equals_sqrt_two :
  let z : ℂ := (1 - 3*I) / (1 + 2*I)
  ‖z‖ = Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_abs_complex_fraction_equals_sqrt_two_l479_47992


namespace NUMINAMATH_CALUDE_complex_magnitude_l479_47907

theorem complex_magnitude (z w : ℂ) 
  (h1 : Complex.abs (3 * z - w) = 15)
  (h2 : Complex.abs (z + 3 * w) = 10)
  (h3 : Complex.abs (z - w) = 3) :
  Complex.abs z = 6 := by sorry

end NUMINAMATH_CALUDE_complex_magnitude_l479_47907


namespace NUMINAMATH_CALUDE_function_property_l479_47977

def is_symmetric_about_one (f : ℝ → ℝ) : Prop :=
  ∀ x, f (x + 1) = f (1 - x)

def is_increasing_on_right_of_one (f : ℝ → ℝ) : Prop :=
  ∀ x y, 1 ≤ x → x < y → f x < f y

def satisfies_inequality (f : ℝ → ℝ) (a : ℝ) : Prop :=
  ∀ x, x ∈ Set.Icc (1/2) 1 → f (a * x + 2) ≤ f (x - 1)

theorem function_property (f : ℝ → ℝ) (h1 : is_symmetric_about_one f)
    (h2 : is_increasing_on_right_of_one f) :
    {a : ℝ | satisfies_inequality f a} = Set.Icc (-2) 0 := by
  sorry

end NUMINAMATH_CALUDE_function_property_l479_47977


namespace NUMINAMATH_CALUDE_distance_midway_to_new_city_l479_47939

theorem distance_midway_to_new_city : 
  let new_city : ℂ := 0
  let old_town : ℂ := 3200 * I
  let midway : ℂ := 960 + 1280 * I
  Complex.abs (midway - new_city) = 3200 := by
sorry

end NUMINAMATH_CALUDE_distance_midway_to_new_city_l479_47939


namespace NUMINAMATH_CALUDE_isosceles_triangle_perimeter_isosceles_triangle_perimeter_proof_l479_47937

/-- An isosceles triangle with side lengths 5 and 10 has a perimeter of 25 -/
theorem isosceles_triangle_perimeter : ℝ → ℝ → ℝ → Prop :=
  fun a b c =>
    (a = 5 ∨ a = 10) ∧ 
    (b = 5 ∨ b = 10) ∧ 
    (c = 5 ∨ c = 10) ∧ 
    (a = b ∨ b = c ∨ a = c) ∧ 
    (a + b > c ∧ b + c > a ∧ a + c > b) →
    a + b + c = 25

theorem isosceles_triangle_perimeter_proof : ∃ a b c : ℝ, isosceles_triangle_perimeter a b c := by
  sorry

end NUMINAMATH_CALUDE_isosceles_triangle_perimeter_isosceles_triangle_perimeter_proof_l479_47937


namespace NUMINAMATH_CALUDE_smallest_valid_perfect_square_l479_47913

def is_valid (n : ℕ) : Prop :=
  ∀ k ∈ Finset.range 10, n % (k + 2) = k + 1

theorem smallest_valid_perfect_square : 
  ∃ n : ℕ, n = 2782559 ∧ 
    is_valid n ∧ 
    ∃ m : ℕ, n = m^2 ∧ 
    ∀ k : ℕ, k < n → ¬(is_valid k ∧ ∃ m : ℕ, k = m^2) :=
by sorry

end NUMINAMATH_CALUDE_smallest_valid_perfect_square_l479_47913


namespace NUMINAMATH_CALUDE_chicken_problem_l479_47980

theorem chicken_problem (total chickens_colten : ℕ) 
  (h_total : total = 383)
  (h_colten : chickens_colten = 37) : 
  ∃ (chickens_skylar chickens_quentin : ℕ),
    chickens_quentin = 2 * chickens_skylar + 25 ∧
    chickens_quentin + chickens_skylar + chickens_colten = total ∧
    3 * chickens_colten - chickens_skylar = 4 :=
by sorry

end NUMINAMATH_CALUDE_chicken_problem_l479_47980


namespace NUMINAMATH_CALUDE_fraction_equation_solution_l479_47923

theorem fraction_equation_solution :
  ∃ x : ℚ, (x + 7) / (x - 4) = (x - 1) / (x + 6) ∧ x = -19/9 ∧ x ≠ -6 ∧ x ≠ 4 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equation_solution_l479_47923


namespace NUMINAMATH_CALUDE_age_ratio_proof_l479_47915

theorem age_ratio_proof (man_age wife_age : ℕ) : 
  man_age = 30 →
  wife_age = 30 →
  man_age - 10 = wife_age →
  ∃ k : ℚ, man_age = k * (wife_age - 10) →
  (man_age : ℚ) / (wife_age - 10 : ℚ) = 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_age_ratio_proof_l479_47915


namespace NUMINAMATH_CALUDE_imaginary_part_of_z_l479_47973

theorem imaginary_part_of_z (z : ℂ) (h : (1 + 2*I)*z = 3 - 2*I) : 
  z.im = -8/5 := by sorry

end NUMINAMATH_CALUDE_imaginary_part_of_z_l479_47973


namespace NUMINAMATH_CALUDE_gcd_from_lcm_and_ratio_l479_47904

theorem gcd_from_lcm_and_ratio (A B : ℕ) (h1 : lcm A B = 180) (h2 : A * 5 = B * 2) : 
  gcd A B = 18 := by
  sorry

end NUMINAMATH_CALUDE_gcd_from_lcm_and_ratio_l479_47904


namespace NUMINAMATH_CALUDE_ExistEvenOddComposition_l479_47959

-- Define the property of being an even function
def IsEven (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = f x

-- Define the property of being an odd function
def IsOdd (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

-- Define the property of a function not being identically zero
def NotIdenticallyZero (f : ℝ → ℝ) : Prop := ∃ x, f x ≠ 0

-- State the theorem
theorem ExistEvenOddComposition :
  ∃ (p q : ℝ → ℝ), IsEven p ∧ IsOdd (p ∘ q) ∧ NotIdenticallyZero (p ∘ q) := by
  sorry

end NUMINAMATH_CALUDE_ExistEvenOddComposition_l479_47959


namespace NUMINAMATH_CALUDE_election_winner_votes_l479_47970

theorem election_winner_votes (total_votes : ℕ) : 
  (total_votes : ℚ) * (58 / 100) - (total_votes : ℚ) * (42 / 100) = 288 →
  (total_votes : ℚ) * (58 / 100) = 1044 := by
  sorry

end NUMINAMATH_CALUDE_election_winner_votes_l479_47970
