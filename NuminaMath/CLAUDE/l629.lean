import Mathlib

namespace ice_cream_cost_l629_62964

/-- Given the following conditions:
    - Alok ordered 16 chapatis, 5 plates of rice, 7 plates of mixed vegetable, and 6 ice-cream cups
    - Cost of each chapati is Rs. 6
    - Cost of each plate of rice is Rs. 45
    - Cost of each plate of mixed vegetable is Rs. 70
    - Alok paid the cashier Rs. 985
    Prove that the cost of each ice-cream cup is Rs. 29 -/
theorem ice_cream_cost (chapati_count : ℕ) (rice_count : ℕ) (vegetable_count : ℕ) (ice_cream_count : ℕ)
                       (chapati_cost : ℕ) (rice_cost : ℕ) (vegetable_cost : ℕ) (total_paid : ℕ) :
  chapati_count = 16 →
  rice_count = 5 →
  vegetable_count = 7 →
  ice_cream_count = 6 →
  chapati_cost = 6 →
  rice_cost = 45 →
  vegetable_cost = 70 →
  total_paid = 985 →
  (total_paid - (chapati_count * chapati_cost + rice_count * rice_cost + vegetable_count * vegetable_cost)) / ice_cream_count = 29 := by
  sorry


end ice_cream_cost_l629_62964


namespace boys_to_girls_ratio_l629_62972

/-- In a class of students, where half of the number of girls equals one-third of the total number of students, 
    the ratio of boys to girls is 1:2. -/
theorem boys_to_girls_ratio (S : ℕ) (B G : ℕ) : 
  S > 0 → 
  S = B + G → 
  (G : ℚ) / 2 = (S : ℚ) / 3 → 
  (B : ℚ) / G = 1 / 2 :=
by sorry

end boys_to_girls_ratio_l629_62972


namespace range_of_m_l629_62960

theorem range_of_m (m : ℝ) : 
  (∀ x : ℝ, x^2 + m*x + 2*m - 3 ≥ 0) → m ∈ Set.Icc 2 6 := by
  sorry

end range_of_m_l629_62960


namespace sum_of_edges_is_120_l629_62917

/-- A rectangular solid with specific properties -/
structure RectangularSolid where
  -- The three dimensions of the solid
  a : ℝ
  b : ℝ
  c : ℝ
  -- Volume is 1000 cm³
  volume_eq : a * b * c = 1000
  -- Surface area is 600 cm²
  surface_area_eq : 2 * (a * b + b * c + a * c) = 600
  -- Dimensions are in geometric progression
  geometric_progression : ∃ (r : ℝ), b = a * r ∧ c = b * r

/-- The sum of all edge lengths of a rectangular solid -/
def sum_of_edges (solid : RectangularSolid) : ℝ :=
  4 * (solid.a + solid.b + solid.c)

/-- Theorem stating that the sum of all edge lengths is 120 cm -/
theorem sum_of_edges_is_120 (solid : RectangularSolid) :
  sum_of_edges solid = 120 := by
  sorry

#check sum_of_edges_is_120

end sum_of_edges_is_120_l629_62917


namespace safe_locks_and_keys_l629_62955

/-- Represents the number of committee members -/
def n : ℕ := 11

/-- Represents the size of the smallest group that can open the safe -/
def k : ℕ := 6

/-- Calculates the number of locks needed -/
def num_locks : ℕ := Nat.choose n (k - 1)

/-- Calculates the total number of keys needed -/
def num_keys : ℕ := num_locks * k

/-- Theorem stating the minimum number of locks and keys needed -/
theorem safe_locks_and_keys : num_locks = 462 ∧ num_keys = 2772 := by
  sorry

#eval num_locks -- Should output 462
#eval num_keys  -- Should output 2772

end safe_locks_and_keys_l629_62955


namespace circle_condition_l629_62994

theorem circle_condition (m : ℝ) :
  (∃ (x₀ y₀ r : ℝ), r > 0 ∧ ∀ (x y : ℝ), x^2 + y^2 + 4*m*x - 2*y + 5*m = 0 ↔ (x - x₀)^2 + (y - y₀)^2 = r^2) ↔
  (m < 1/4 ∨ m > 1) :=
sorry

end circle_condition_l629_62994


namespace jakes_weight_ratio_l629_62997

/-- Proves that the ratio of Jake's weight after losing 8 pounds to his sister's weight is 2:1 -/
theorem jakes_weight_ratio :
  let jake_current_weight : ℕ := 188
  let total_weight : ℕ := 278
  let weight_loss : ℕ := 8
  let jake_new_weight : ℕ := jake_current_weight - weight_loss
  let sister_weight : ℕ := total_weight - jake_current_weight
  (jake_new_weight : ℚ) / (sister_weight : ℚ) = 2 / 1 := by
  sorry

end jakes_weight_ratio_l629_62997


namespace min_value_of_expression_l629_62958

theorem min_value_of_expression (m n : ℝ) (hm : m > 0) (hn : n > 0) (h : 3 * m + n = 1) :
  (1 / m + 3 / n) ≥ 12 ∧ ∃ m n, m > 0 ∧ n > 0 ∧ 3 * m + n = 1 ∧ 1 / m + 3 / n = 12 :=
sorry

end min_value_of_expression_l629_62958


namespace rectangular_box_diagonal_sum_l629_62922

theorem rectangular_box_diagonal_sum (a b c : ℝ) 
  (h_surface_area : 2 * (a * b + b * c + c * a) = 112)
  (h_edge_sum : 4 * (a + b + c) = 60) :
  4 * Real.sqrt (a^2 + b^2 + c^2) = 4 * Real.sqrt 113 := by
  sorry

end rectangular_box_diagonal_sum_l629_62922


namespace integer_expression_l629_62935

theorem integer_expression (n : ℕ) : ∃ (k : ℤ), 
  (3^(2*n) : ℚ) / 112 - (4^(2*n) : ℚ) / 63 + (5^(2*n) : ℚ) / 144 = k := by
  sorry

end integer_expression_l629_62935


namespace g_value_at_one_l629_62903

def g_property (g : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, g (g (x - y)) = g x * g y - g x + g y - 2 * x * y

theorem g_value_at_one (g : ℝ → ℝ) (h : g_property g) : g 1 = -Real.sqrt 2 := by
  sorry

end g_value_at_one_l629_62903


namespace trigonometric_system_solution_l629_62923

theorem trigonometric_system_solution :
  let eq1 (x y : Real) := 
    (Real.sin x + Real.cos x) / (Real.sin y + Real.cos y) + 
    (Real.sin y - Real.cos y) / (Real.sin x + Real.cos x) = 
    1 / (Real.sin (x + y) + Real.cos (x - y))
  let eq2 (x y : Real) := 
    2 * (Real.sin x + Real.cos x)^2 - (2 * Real.cos y^2 + 1) = Real.sqrt 3 / 2
  let solutions : List (Real × Real) := 
    [(π/6, π/12), (π/6, 13*π/12), (π/3, 11*π/12), (π/3, 23*π/12)]
  ∀ (x y : Real), (x, y) ∈ solutions → eq1 x y ∧ eq2 x y :=
by
  sorry


end trigonometric_system_solution_l629_62923


namespace absolute_value_ratio_l629_62993

theorem absolute_value_ratio (a b : ℝ) (h1 : a ≠ 0) (h2 : b ≠ 0) (h3 : a^2 + b^2 = 10*a*b) :
  |((a + b) / (a - b))| = Real.sqrt (3/2) := by
sorry

end absolute_value_ratio_l629_62993


namespace complex_division_result_l629_62984

theorem complex_division_result : (3 + Complex.I) / (1 + Complex.I) = 2 - Complex.I := by
  sorry

end complex_division_result_l629_62984


namespace inverse_variation_problem_l629_62956

theorem inverse_variation_problem (a b : ℝ) (k : ℝ) (h1 : a * b^3 = k) (h2 : 8 * 2^3 = k) :
  a * 4^3 = k → a = 1 := by
sorry

end inverse_variation_problem_l629_62956


namespace order_of_abc_l629_62942

noncomputable def a : ℝ := 2 * Real.log 1.01
noncomputable def b : ℝ := Real.log 1.02
noncomputable def c : ℝ := Real.sqrt 1.04 - 1

theorem order_of_abc : a > c ∧ c > b := by sorry

end order_of_abc_l629_62942


namespace complex_roots_of_quadratic_l629_62998

theorem complex_roots_of_quadratic (a b : ℝ) : 
  (Complex.I + 1) ^ 2 + a * (Complex.I + 1) + b = 0 → 
  (a = -2 ∧ b = 2) ∧ (Complex.I - 1) ^ 2 + a * (Complex.I - 1) + b = 0 := by
  sorry

end complex_roots_of_quadratic_l629_62998


namespace remainder_theorem_l629_62945

theorem remainder_theorem (n : ℤ) (k : ℤ) (h : n = 75 * k - 2) :
  (n^2 + 2*n + 3) % 75 = 3 := by
sorry

end remainder_theorem_l629_62945


namespace sin_2x_derivative_at_pi_6_l629_62908

theorem sin_2x_derivative_at_pi_6 (f : ℝ → ℝ) (h : ∀ x, f x = Real.sin (2 * x)) :
  deriv f (π / 6) = 1 := by
  sorry

end sin_2x_derivative_at_pi_6_l629_62908


namespace smallest_five_digit_congruent_to_two_mod_seventeen_l629_62948

theorem smallest_five_digit_congruent_to_two_mod_seventeen : 
  ∀ n : ℕ, 
    10000 ≤ n ∧ n < 100000 ∧ n ≡ 2 [ZMOD 17] → 
    n ≥ 10013 := by
  sorry

end smallest_five_digit_congruent_to_two_mod_seventeen_l629_62948


namespace correct_journey_equation_l629_62963

/-- Represents the journey of a ship between two ports -/
def ship_journey (distance : ℝ) (flow_speed : ℝ) (ship_speed : ℝ) : Prop :=
  distance / (ship_speed + flow_speed) + distance / (ship_speed - flow_speed) = 8

/-- Theorem stating that the given equation correctly represents the ship's journey -/
theorem correct_journey_equation :
  ∀ x : ℝ, x > 4 → ship_journey 50 4 x :=
by
  sorry

end correct_journey_equation_l629_62963


namespace kaleb_toys_l629_62983

def number_of_toys (initial_savings allowance toy_cost : ℕ) : ℕ :=
  (initial_savings + allowance) / toy_cost

theorem kaleb_toys : number_of_toys 21 15 6 = 6 := by
  sorry

end kaleb_toys_l629_62983


namespace apple_pear_equivalence_l629_62941

/-- Given that 3/4 of 12 apples are worth as much as 6 pears,
    prove that 1/3 of 9 apples are worth as much as 2 pears. -/
theorem apple_pear_equivalence (apple pear : ℝ) 
    (h : (3/4 : ℝ) * 12 * apple = 6 * pear) : 
    (1/3 : ℝ) * 9 * apple = 2 * pear := by
  sorry

end apple_pear_equivalence_l629_62941


namespace smaller_number_in_ratio_l629_62973

/-- Given two positive integers a and b in ratio 4:5 with LCM 180, prove that a = 36 -/
theorem smaller_number_in_ratio (a b : ℕ+) : 
  (a : ℚ) / b = 4 / 5 → 
  Nat.lcm a b = 180 → 
  a = 36 := by
sorry

end smaller_number_in_ratio_l629_62973


namespace parallel_lines_in_parallel_planes_not_always_parallel_l629_62918

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relations
variable (subset : Line → Plane → Prop)
variable (parallel_lines : Line → Line → Prop)
variable (parallel_planes : Plane → Plane → Prop)

-- State the theorem
theorem parallel_lines_in_parallel_planes_not_always_parallel 
  (m n : Line) (α β : Plane) : 
  ¬(∀ m n α β, subset m α ∧ subset n β ∧ parallel_planes α β → parallel_lines m n) :=
sorry

end parallel_lines_in_parallel_planes_not_always_parallel_l629_62918


namespace insect_legs_count_l629_62987

theorem insect_legs_count (num_insects : ℕ) (legs_per_insect : ℕ) : 
  num_insects = 5 → legs_per_insect = 6 → num_insects * legs_per_insect = 30 := by
  sorry

end insect_legs_count_l629_62987


namespace largest_non_expressible_l629_62913

def is_composite (n : ℕ) : Prop :=
  ∃ a b, 1 < a ∧ 1 < b ∧ n = a * b

def is_not_multiple_of_four (n : ℕ) : Prop :=
  ¬(∃ k, n = 4 * k)

def is_expressible (n : ℕ) : Prop :=
  ∃ (a b : ℕ), 0 < a ∧ 0 < b ∧ is_composite b ∧ is_not_multiple_of_four b ∧ n = 36 * a + b

theorem largest_non_expressible : 
  (∀ n > 147, is_expressible n) ∧ ¬(is_expressible 147) :=
sorry

end largest_non_expressible_l629_62913


namespace polynomial_divisibility_l629_62900

def p (m : ℝ) (x : ℝ) : ℝ := 4 * x^3 - 16 * x^2 + m * x - 20

theorem polynomial_divisibility (m : ℝ) :
  (∃ q : ℝ → ℝ, ∀ x, p m x = (x - 4) * q x) →
  (m = 5 ∧ ¬∃ r : ℝ → ℝ, ∀ x, p 5 x = (x - 5) * r x) :=
by sorry

end polynomial_divisibility_l629_62900


namespace y_axis_reflection_l629_62916

/-- Given a point P(-2,3) in the Cartesian coordinate system, 
    its coordinates with respect to the y-axis are (2,3). -/
theorem y_axis_reflection :
  let P : ℝ × ℝ := (-2, 3)
  let reflected_P : ℝ × ℝ := (2, 3)
  reflected_P = (-(P.1), P.2) :=
by sorry

end y_axis_reflection_l629_62916


namespace smallest_x_value_l629_62943

theorem smallest_x_value (x : ℝ) : x ≠ 1/4 →
  ((20 * x^2 - 49 * x + 20) / (4 * x - 1) + 7 * x = 3 * x + 2) →
  x ≥ 2/9 ∧ (∃ y : ℝ, y ≠ 1/4 ∧ ((20 * y^2 - 49 * y + 20) / (4 * y - 1) + 7 * y = 3 * y + 2) ∧ y = 2/9) :=
by sorry

end smallest_x_value_l629_62943


namespace correct_division_incorrect_others_l629_62992

theorem correct_division_incorrect_others :
  ((-8) / (-4) = 8 / 4) ∧
  ¬((-5) + 9 = -(9 - 5)) ∧
  ¬(7 - (-10) = 7 - 10) ∧
  ¬((-5) * 0 = -5) := by
  sorry

end correct_division_incorrect_others_l629_62992


namespace arithmetic_progression_theorem_l629_62990

/-- Represents an arithmetic progression of five terms. -/
structure ArithmeticProgression :=
  (a : ℝ)  -- First term
  (d : ℝ)  -- Common difference

/-- Checks if the arithmetic progression is decreasing. -/
def ArithmeticProgression.isDecreasing (ap : ArithmeticProgression) : Prop :=
  ap.d > 0

/-- Calculates the sum of cubes of the terms in the arithmetic progression. -/
def ArithmeticProgression.sumOfCubes (ap : ArithmeticProgression) : ℝ :=
  ap.a^3 + (ap.a - ap.d)^3 + (ap.a - 2*ap.d)^3 + (ap.a - 3*ap.d)^3 + (ap.a - 4*ap.d)^3

/-- Calculates the sum of fourth powers of the terms in the arithmetic progression. -/
def ArithmeticProgression.sumOfFourthPowers (ap : ArithmeticProgression) : ℝ :=
  ap.a^4 + (ap.a - ap.d)^4 + (ap.a - 2*ap.d)^4 + (ap.a - 3*ap.d)^4 + (ap.a - 4*ap.d)^4

/-- The main theorem stating the properties of the required arithmetic progression. -/
theorem arithmetic_progression_theorem (ap : ArithmeticProgression) :
  ap.isDecreasing ∧
  ap.sumOfCubes = 0 ∧
  ap.sumOfFourthPowers = 306 →
  ap.a - 4*ap.d = -2 * Real.sqrt 3 :=
by sorry

end arithmetic_progression_theorem_l629_62990


namespace infinitely_many_composite_numbers_l629_62953

theorem infinitely_many_composite_numbers :
  ∃ (N : Set ℕ), Set.Infinite N ∧
    ∀ n ∈ N, ∃ (a b : ℕ), a > 1 ∧ b > 1 ∧ 50^n + (50*n + 1)^50 = a * b :=
by sorry

end infinitely_many_composite_numbers_l629_62953


namespace equation_solution_l629_62909

theorem equation_solution : ∃ x : ℚ, (3 / 7 + 7 / x = 10 / x + 1 / 10) ∧ x = 210 / 23 := by
  sorry

end equation_solution_l629_62909


namespace jiyoon_sum_l629_62938

theorem jiyoon_sum : 36 + 17 + 32 + 54 + 28 + 3 = 170 := by
  sorry

end jiyoon_sum_l629_62938


namespace mistaken_division_l629_62970

theorem mistaken_division (D : ℕ) (h1 : D % 21 = 0) (h2 : D / 21 = 36) :
  D / 12 = 63 := by
sorry

end mistaken_division_l629_62970


namespace motel_rental_rate_l629_62921

theorem motel_rental_rate (lower_rate higher_rate total_rent : ℚ) 
  (h1 : lower_rate = 40)
  (h2 : total_rent = 400)
  (h3 : total_rent / 2 = total_rent - 10 * (higher_rate - lower_rate)) :
  higher_rate = 60 := by
  sorry

end motel_rental_rate_l629_62921


namespace yellow_peaches_count_l629_62968

/-- The number of yellow peaches in a basket -/
def yellow_peaches (red green yellow total_green_yellow : ℕ) : Prop :=
  red = 5 ∧ green = 6 ∧ total_green_yellow = 20 → yellow = 14

theorem yellow_peaches_count : ∀ (red green yellow total_green_yellow : ℕ),
  yellow_peaches red green yellow total_green_yellow :=
by
  sorry

end yellow_peaches_count_l629_62968


namespace function_derivative_problem_l629_62920

theorem function_derivative_problem (f : ℝ → ℝ) (a : ℝ) 
  (h1 : ∀ x, f x = (2*x + a)^2)
  (h2 : deriv f 2 = 20) : 
  a = 1 := by sorry

end function_derivative_problem_l629_62920


namespace complex_modulus_of_three_plus_i_squared_l629_62914

theorem complex_modulus_of_three_plus_i_squared :
  let z : ℂ := (3 + Complex.I) ^ 2
  ‖z‖ = 10 := by sorry

end complex_modulus_of_three_plus_i_squared_l629_62914


namespace triangle_inequality_l629_62967

theorem triangle_inequality (a b c : ℝ) : 
  (a + b + c = 2) → 
  (a > 0) → (b > 0) → (c > 0) →
  (a + b ≥ c) → (b + c ≥ a) → (c + a ≥ b) →
  abc + 1/27 ≥ ab + bc + ca - 1 ∧ ab + bc + ca - 1 ≥ abc := by
  sorry

end triangle_inequality_l629_62967


namespace second_round_score_l629_62902

/-- Represents the points scored in a round of darts --/
structure DartScore :=
  (points : ℕ)

/-- Represents the scores for three rounds of darts --/
structure ThreeRoundScores :=
  (round1 : DartScore)
  (round2 : DartScore)
  (round3 : DartScore)

/-- Defines the relationship between scores in three rounds --/
def validScores (scores : ThreeRoundScores) : Prop :=
  scores.round2.points = 2 * scores.round1.points ∧
  scores.round3.points = (3 * scores.round1.points : ℕ)

/-- Theorem: Given the conditions, the score in the second round is 48 --/
theorem second_round_score (scores : ThreeRoundScores) 
  (h : validScores scores) : scores.round2.points = 48 := by
  sorry

#check second_round_score

end second_round_score_l629_62902


namespace two_solutions_l629_62949

-- Define the matrix evaluation rule
def matrixEval (a b c d : ℝ) : ℝ := a * b - c * d + c

-- Define the equation
def equation (x : ℝ) : Prop := matrixEval (3 * x) x 2 (2 * x) = 2

-- Theorem statement
theorem two_solutions :
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ equation x₁ ∧ equation x₂) ∧
  (∀ x : ℝ, equation x → x = 0 ∨ x = 4/3) :=
sorry

end two_solutions_l629_62949


namespace sisters_birth_year_l629_62989

/-- Represents the birth years of family members --/
structure FamilyBirthYears where
  brother : Nat
  sister : Nat
  grandmother : Nat

/-- Checks if the birth years satisfy the given conditions --/
def validBirthYears (years : FamilyBirthYears) : Prop :=
  years.brother = 1932 ∧
  years.grandmother = 1944 ∧
  (years.grandmother - years.sister) = 2 * (years.sister - years.brother)

/-- Theorem stating that the grandmother's older sister was born in 1936 --/
theorem sisters_birth_year (years : FamilyBirthYears) 
  (h : validBirthYears years) : years.sister = 1936 := by
  sorry

#check sisters_birth_year

end sisters_birth_year_l629_62989


namespace equilateral_triangle_side_length_l629_62985

theorem equilateral_triangle_side_length 
  (circular_radius : ℝ) 
  (circular_speed : ℝ) 
  (triangular_speed : ℝ) 
  (h1 : circular_radius = 60) 
  (h2 : circular_speed = 6) 
  (h3 : triangular_speed = 5) :
  ∃ x : ℝ, 
    (3 * x = triangular_speed * ((2 * Real.pi * circular_radius) / circular_speed)) ∧ 
    x = 100 * Real.pi / 3 := by
  sorry

end equilateral_triangle_side_length_l629_62985


namespace ticket_distribution_ways_l629_62979

/-- The number of ways to distribute tickets among programs -/
def distribute_tickets (total_tickets : ℕ) (num_programs : ℕ) (min_tickets_a : ℕ) : ℕ :=
  sorry

/-- The number of ways to distribute 6 tickets among 4 programs with program A receiving at least 3 and the most -/
theorem ticket_distribution_ways : distribute_tickets 6 4 3 = 17 := by
  sorry

end ticket_distribution_ways_l629_62979


namespace no_solution_iff_k_equals_nine_l629_62951

theorem no_solution_iff_k_equals_nine :
  ∀ k : ℝ, (∀ x : ℝ, x ≠ 1 ∧ x ≠ 7 → (x - 3) / (x - 1) ≠ (x - k) / (x - 7)) ↔ k = 9 := by
  sorry

end no_solution_iff_k_equals_nine_l629_62951


namespace simplify_expression_1_simplify_expression_2_l629_62940

-- Problem 1
theorem simplify_expression_1 (x : ℝ) :
  (2*x - 1) * (2*x - 3) - (1 - 2*x) * (2 - x) = 2*x^2 - 3*x + 1 := by sorry

-- Problem 2
theorem simplify_expression_2 (a : ℝ) (ha : a ≠ 0) (ha' : a ≠ 1) :
  (a^2 - 1) / a * (1 - (2*a + 1) / (a^2 + 2*a + 1)) / (a - 1) = a / (a + 1) := by sorry

end simplify_expression_1_simplify_expression_2_l629_62940


namespace wax_requirement_l629_62950

theorem wax_requirement (current_wax : ℕ) (additional_wax : ℕ) : 
  current_wax = 11 → additional_wax = 481 → current_wax + additional_wax = 492 := by
  sorry

end wax_requirement_l629_62950


namespace no_double_application_function_l629_62962

theorem no_double_application_function : ¬∃ (f : ℕ → ℕ), ∀ n, f (f n) = n + 1 := by
  sorry

end no_double_application_function_l629_62962


namespace sin_cos_difference_equals_half_l629_62904

theorem sin_cos_difference_equals_half : 
  Real.sin (65 * π / 180) * Real.cos (35 * π / 180) - 
  Real.sin (25 * π / 180) * Real.sin (35 * π / 180) = 1/2 := by
  sorry

end sin_cos_difference_equals_half_l629_62904


namespace childrens_tickets_sold_l629_62906

theorem childrens_tickets_sold 
  (adult_price : ℚ) 
  (child_price : ℚ) 
  (total_tickets : ℕ) 
  (total_revenue : ℚ) 
  (h1 : adult_price = 6)
  (h2 : child_price = 9/2)
  (h3 : total_tickets = 400)
  (h4 : total_revenue = 2100) :
  ∃ (adult_tickets child_tickets : ℕ),
    adult_tickets + child_tickets = total_tickets ∧
    adult_price * adult_tickets + child_price * child_tickets = total_revenue ∧
    child_tickets = 200 := by
  sorry

end childrens_tickets_sold_l629_62906


namespace paint_used_after_four_weeks_l629_62905

/-- Calculates the amount of paint used over 4 weeks given an initial amount and usage fractions --/
def paint_used (initial : ℝ) (w1_frac w2_frac w3_frac w4_frac : ℝ) : ℝ :=
  let w1_used := w1_frac * initial
  let w1_remaining := initial - w1_used
  let w2_used := w2_frac * w1_remaining
  let w2_remaining := w1_remaining - w2_used
  let w3_used := w3_frac * w2_remaining
  let w3_remaining := w2_remaining - w3_used
  let w4_used := w4_frac * w3_remaining
  w1_used + w2_used + w3_used + w4_used

/-- The theorem stating the amount of paint used after 4 weeks --/
theorem paint_used_after_four_weeks :
  let initial_paint := 360
  let week1_fraction := 1/4
  let week2_fraction := 1/3
  let week3_fraction := 2/5
  let week4_fraction := 3/7
  abs (paint_used initial_paint week1_fraction week2_fraction week3_fraction week4_fraction - 298.2857) < 0.0001 := by
  sorry


end paint_used_after_four_weeks_l629_62905


namespace equation_solution_l629_62977

theorem equation_solution : ∃ x : ℝ, (6 + 1.5 * x = 2.5 * x - 30 + Real.sqrt 100) ∧ x = 26 := by
  sorry

end equation_solution_l629_62977


namespace range_of_a_minus_abs_b_l629_62981

theorem range_of_a_minus_abs_b (a b : ℝ) :
  1 < a ∧ a < 8 ∧ -4 < b ∧ b < 2 →
  ∃ x, -3 < x ∧ x < 8 ∧ x = a - |b| :=
sorry

end range_of_a_minus_abs_b_l629_62981


namespace point_in_second_quadrant_l629_62975

def point : ℝ × ℝ := (-2, 3)

def is_in_second_quadrant (p : ℝ × ℝ) : Prop :=
  p.1 < 0 ∧ p.2 > 0

theorem point_in_second_quadrant : is_in_second_quadrant point := by
  sorry

end point_in_second_quadrant_l629_62975


namespace inequality_proof_l629_62919

theorem inequality_proof (a b c : ℝ) (hab : a ≠ b) (hbc : b ≠ c) (hca : c ≠ a) :
  (2*a - b)^2 / (a - b)^2 + (2*b - c)^2 / (b - c)^2 + (2*c - a)^2 / (c - a)^2 ≥ 5 := by
  sorry

end inequality_proof_l629_62919


namespace owls_on_fence_l629_62934

theorem owls_on_fence (initial_owls final_owls joined_owls : ℕ) : 
  final_owls = initial_owls + joined_owls →
  joined_owls = 2 →
  final_owls = 5 →
  initial_owls = 3 := by
  sorry

end owls_on_fence_l629_62934


namespace game_strategy_sum_final_result_l629_62910

theorem game_strategy_sum (R S : ℕ) : R - S = 1010 :=
  by
  have h1 : R = (1010 : ℕ) * 2022 / 2 := by sorry
  have h2 : S = (1010 : ℕ) * 2020 / 2 := by sorry
  sorry

theorem final_result : (R - S) / 10 = 101 :=
  by
  have h : R - S = 1010 := game_strategy_sum R S
  sorry

end game_strategy_sum_final_result_l629_62910


namespace min_value_of_squared_differences_l629_62969

theorem min_value_of_squared_differences (a b c : ℝ) :
  ∃ (min : ℝ), min = ((a - b)^2 + (b - c)^2 + (a - c)^2) / 3 ∧
  ∀ (x : ℝ), (x - a)^2 + (x - b)^2 + (x - c)^2 ≥ min :=
by sorry

end min_value_of_squared_differences_l629_62969


namespace dartboard_angle_l629_62928

theorem dartboard_angle (p : ℝ) (θ : ℝ) : 
  p = 1 / 8 → θ = p * 360 → θ = 45 :=
by sorry

end dartboard_angle_l629_62928


namespace isosceles_triangles_bound_l629_62954

/-- The largest number of isosceles triangles whose vertices belong to some set of n points in the plane without three colinear points -/
noncomputable def f (n : ℕ) : ℕ := sorry

/-- Theorem stating the existence of positive real constants a and b bounding f(n) -/
theorem isosceles_triangles_bound :
  ∃ (a b : ℝ), a > 0 ∧ b > 0 ∧ ∀ n : ℕ, n ≥ 3 → (a * n^2 : ℝ) < f n ∧ (f n : ℝ) < b * n^2 :=
sorry

end isosceles_triangles_bound_l629_62954


namespace largest_c_for_negative_three_in_range_l629_62911

-- Define the function f
def f (x c : ℝ) : ℝ := x^2 + 5*x + c

-- State the theorem
theorem largest_c_for_negative_three_in_range :
  (∃ (c : ℝ), ∀ (d : ℝ), 
    (∃ (x : ℝ), f x c = -3) → 
    (∃ (y : ℝ), f y d = -3) → 
    d ≤ c) ∧
  (∃ (x : ℝ), f x (13/4) = -3) :=
sorry

end largest_c_for_negative_three_in_range_l629_62911


namespace adult_tickets_sold_l629_62946

/-- Represents the number of adult tickets sold -/
def adult_tickets : ℕ := sorry

/-- Represents the number of child tickets sold -/
def child_tickets : ℕ := sorry

/-- The cost of an adult ticket in dollars -/
def adult_cost : ℕ := 12

/-- The cost of a child ticket in dollars -/
def child_cost : ℕ := 4

/-- The total number of tickets sold -/
def total_tickets : ℕ := 130

/-- The total receipts in dollars -/
def total_receipts : ℕ := 840

theorem adult_tickets_sold : 
  adult_tickets = 40 ∧
  adult_tickets + child_tickets = total_tickets ∧
  adult_tickets * adult_cost + child_tickets * child_cost = total_receipts :=
by sorry

end adult_tickets_sold_l629_62946


namespace inequality_proof_l629_62957

theorem inequality_proof (a b c : ℝ) 
  (h1 : a > b) (h2 : b > c) (h3 : a + b + c = 0) : 
  Real.sqrt (b^2 - a*c) > Real.sqrt 3 * a := by
  sorry

end inequality_proof_l629_62957


namespace vacation_pictures_remaining_l629_62924

-- Define the number of pictures taken at each location
def zoo_pictures : ℕ := 49
def museum_pictures : ℕ := 8

-- Define the number of deleted pictures
def deleted_pictures : ℕ := 38

-- Theorem to prove
theorem vacation_pictures_remaining :
  zoo_pictures + museum_pictures - deleted_pictures = 19 := by
  sorry

end vacation_pictures_remaining_l629_62924


namespace sqrt_seven_to_sixth_l629_62932

theorem sqrt_seven_to_sixth : (Real.sqrt 7) ^ 6 = 343 := by
  sorry

end sqrt_seven_to_sixth_l629_62932


namespace M_reflected_y_axis_l629_62959

/-- Reflects a point across the y-axis -/
def reflect_y_axis (p : ℝ × ℝ) : ℝ × ℝ :=
  (-(p.1), p.2)

/-- The coordinates of point M -/
def M : ℝ × ℝ := (1, 2)

theorem M_reflected_y_axis :
  reflect_y_axis M = (-1, 2) := by sorry

end M_reflected_y_axis_l629_62959


namespace tan_150_degrees_l629_62988

theorem tan_150_degrees :
  Real.tan (150 * π / 180) = -1 / Real.sqrt 3 := by
  sorry

end tan_150_degrees_l629_62988


namespace equation_solution_l629_62930

theorem equation_solution : ∃ x : ℚ, (3/4 : ℚ) + 1/x = 7/8 ∧ x = 8 := by
  sorry

end equation_solution_l629_62930


namespace right_triangle_inradius_l629_62991

/-- The inradius of a right triangle with side lengths 7, 24, and 25 is 3 -/
theorem right_triangle_inradius : ∀ (a b c r : ℝ),
  a = 7 ∧ b = 24 ∧ c = 25 →
  a^2 + b^2 = c^2 →
  (a + b + c) / 2 * r = (a * b) / 2 →
  r = 3 := by sorry

end right_triangle_inradius_l629_62991


namespace logarithm_order_comparison_l629_62944

theorem logarithm_order_comparison : 
  Real.log 4 / Real.log 3 > Real.log 3 / Real.log 4 ∧ 
  Real.log 3 / Real.log 4 > Real.log (3/4) / Real.log (4/3) := by
  sorry

end logarithm_order_comparison_l629_62944


namespace a1_value_l629_62965

noncomputable section

def f (x : ℝ) : ℝ :=
  if x ≥ 1 then x * Real.log x else Real.log x / x

def is_geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, r > 0 ∧ ∀ n : ℕ, a (n + 1) = r * a n

theorem a1_value (a : ℕ → ℝ) :
  is_geometric_sequence a →
  a 3 * a 4 * a 5 = 1 →
  f (a 1) + f (a 2) + f (a 3) + f (a 4) + f (a 5) + f (a 6) = 2 * a 1 →
  a 1 = Real.exp 2 := by sorry

end

end a1_value_l629_62965


namespace vector_equation_solution_l629_62982

theorem vector_equation_solution (a b : ℝ × ℝ) (m n : ℝ) : 
  a = (2, 1) → b = (1, -2) → m • a + n • b = (9, -8) → m - n = -3 := by
  sorry

end vector_equation_solution_l629_62982


namespace tapanga_corey_candy_difference_l629_62996

theorem tapanga_corey_candy_difference (total : ℕ) (corey : ℕ) (h1 : total = 66) (h2 : corey = 29) (h3 : corey < total - corey) :
  total - corey - corey = 8 := by
  sorry

end tapanga_corey_candy_difference_l629_62996


namespace infinite_series_sum_equals_one_l629_62976

/-- The sum of the infinite series Σ(n=1 to ∞) (n^5 + 5n^3 + 15n + 15) / (2^n * (n^5 + 5)) is equal to 1 -/
theorem infinite_series_sum_equals_one :
  let f : ℕ → ℝ := λ n => (n^5 + 5*n^3 + 15*n + 15) / (2^n * (n^5 + 5))
  ∑' n, f n = 1 := by sorry

end infinite_series_sum_equals_one_l629_62976


namespace class_test_probabilities_l629_62947

theorem class_test_probabilities (p_first : ℝ) (p_second : ℝ) (p_neither : ℝ)
  (h1 : p_first = 0.7)
  (h2 : p_second = 0.55)
  (h3 : p_neither = 0.2) :
  p_first + p_second - (1 - p_neither) = 0.45 := by
  sorry

end class_test_probabilities_l629_62947


namespace teacher_engineer_ratio_l629_62966

theorem teacher_engineer_ratio 
  (t e : ℕ) -- t is the number of teachers, e is the number of engineers
  (h_group : t + e > 0) -- ensures the group is not empty
  (h_avg : (40 * t + 55 * e) / (t + e) = 45) -- average age of the entire group is 45
  : t = 2 * e := by
sorry

end teacher_engineer_ratio_l629_62966


namespace problem_statement_l629_62974

theorem problem_statement (x : ℝ) (h : 1 - 5/x + 6/x^3 = 0) : 3/x = 3/2 := by
  sorry

end problem_statement_l629_62974


namespace apple_weight_difference_l629_62937

/-- Given two baskets of apples with a total weight and the weight of one basket,
    prove the difference in weight between the baskets. -/
theorem apple_weight_difference (total_weight weight_a : ℕ) 
  (h1 : total_weight = 72)
  (h2 : weight_a = 42) :
  weight_a - (total_weight - weight_a) = 12 := by
  sorry

#check apple_weight_difference

end apple_weight_difference_l629_62937


namespace product_first_three_eq_960_l629_62907

/-- An arithmetic sequence with the given properties -/
structure ArithmeticSequence where
  -- The seventh term is 20
  seventh_term : ℕ
  seventh_term_eq : seventh_term = 20
  -- The common difference is 2
  common_diff : ℕ
  common_diff_eq : common_diff = 2

/-- The product of the first three terms of the arithmetic sequence -/
def product_first_three (seq : ArithmeticSequence) : ℕ :=
  let a := seq.seventh_term - 6 * seq.common_diff -- First term
  let a2 := a + seq.common_diff -- Second term
  let a3 := a + 2 * seq.common_diff -- Third term
  a * a2 * a3

/-- Theorem stating that the product of the first three terms is 960 -/
theorem product_first_three_eq_960 (seq : ArithmeticSequence) :
  product_first_three seq = 960 := by
  sorry

end product_first_three_eq_960_l629_62907


namespace three_elements_satisfy_l629_62980

/-- The set M containing elements A, A₁, A₂, A₃, A₄, A₅ -/
inductive M
  | A
  | A1
  | A2
  | A3
  | A4
  | A5

/-- The operation ⊗ defined on M -/
def otimes : M → M → M
  | M.A, M.A => M.A
  | M.A, M.A1 => M.A1
  | M.A, M.A2 => M.A2
  | M.A, M.A3 => M.A3
  | M.A, M.A4 => M.A4
  | M.A, M.A5 => M.A1
  | M.A1, M.A => M.A1
  | M.A1, M.A1 => M.A2
  | M.A1, M.A2 => M.A3
  | M.A1, M.A3 => M.A4
  | M.A1, M.A4 => M.A1
  | M.A1, M.A5 => M.A2
  | M.A2, M.A => M.A2
  | M.A2, M.A1 => M.A3
  | M.A2, M.A2 => M.A4
  | M.A2, M.A3 => M.A1
  | M.A2, M.A4 => M.A2
  | M.A2, M.A5 => M.A3
  | M.A3, M.A => M.A3
  | M.A3, M.A1 => M.A4
  | M.A3, M.A2 => M.A1
  | M.A3, M.A3 => M.A2
  | M.A3, M.A4 => M.A3
  | M.A3, M.A5 => M.A4
  | M.A4, M.A => M.A4
  | M.A4, M.A1 => M.A1
  | M.A4, M.A2 => M.A2
  | M.A4, M.A3 => M.A3
  | M.A4, M.A4 => M.A4
  | M.A4, M.A5 => M.A1
  | M.A5, M.A => M.A1
  | M.A5, M.A1 => M.A2
  | M.A5, M.A2 => M.A3
  | M.A5, M.A3 => M.A4
  | M.A5, M.A4 => M.A1
  | M.A5, M.A5 => M.A2

/-- The theorem stating that exactly 3 elements in M satisfy (a ⊗ a) ⊗ A₂ = A -/
theorem three_elements_satisfy :
  (∃! (s : Finset M), s.card = 3 ∧ ∀ a ∈ s, otimes (otimes a a) M.A2 = M.A) :=
sorry

end three_elements_satisfy_l629_62980


namespace largest_prime_factor_of_4620_l629_62971

theorem largest_prime_factor_of_4620 : 
  (Nat.factors 4620).maximum? = some 11 := by
  sorry

end largest_prime_factor_of_4620_l629_62971


namespace remaining_money_l629_62939

def initial_amount : ℚ := 3
def purchase_amount : ℚ := 1

theorem remaining_money :
  initial_amount - purchase_amount = 2 := by sorry

end remaining_money_l629_62939


namespace stem_and_leaf_update_l629_62901

/-- Represents a stem-and-leaf diagram --/
structure StemAndLeaf :=
  (stem : List ℕ)
  (leaf : List (List ℕ))

/-- The initial stem-and-leaf diagram --/
def initial_diagram : StemAndLeaf := {
  stem := [0, 1, 2, 3, 4],
  leaf := [[], [0, 0, 1, 2, 2, 3], [1, 5, 6], [0, 2, 4, 6], [1, 6]]
}

/-- Function to update ages in the diagram --/
def update_ages (d : StemAndLeaf) (years : ℕ) : StemAndLeaf :=
  sorry

/-- Theorem stating the time passed and the reconstruction of the new diagram --/
theorem stem_and_leaf_update :
  ∃ (years : ℕ) (new_diagram : StemAndLeaf),
    years = 6 ∧
    new_diagram = update_ages initial_diagram years ∧
    new_diagram.stem = [0, 1, 2, 3, 4] ∧
    new_diagram.leaf = [[],
                        [5, 5],
                        [1, 5, 6],
                        [0, 2, 4, 6],
                        [1, 6]] :=
  sorry

end stem_and_leaf_update_l629_62901


namespace car_efficiency_improvement_l629_62933

/-- Represents the additional miles a car can travel after improving fuel efficiency -/
def additional_miles (initial_efficiency : ℝ) (tank_capacity : ℝ) (efficiency_improvement : ℝ) : ℝ :=
  tank_capacity * (initial_efficiency * (1 + efficiency_improvement) - initial_efficiency)

/-- Theorem stating the additional miles a car can travel after modification -/
theorem car_efficiency_improvement :
  additional_miles 33 16 0.25 = 132 := by
  sorry

end car_efficiency_improvement_l629_62933


namespace handshake_problem_l629_62925

theorem handshake_problem (n : ℕ) : n * (n - 1) / 2 = 78 → n = 13 := by
  sorry

end handshake_problem_l629_62925


namespace possible_values_of_a_l629_62915

def A : Set ℝ := {x | x^2 + x - 6 = 0}
def B (a : ℝ) : Set ℝ := {x | a*x + 1 = 0}

theorem possible_values_of_a (a : ℝ) : A ∪ B a = A → a ∈ ({0, 1/3, -1/2} : Set ℝ) := by
  sorry

end possible_values_of_a_l629_62915


namespace exam_score_difference_l629_62986

def math_exam_proof (bryan_score jen_score sammy_score total_points sammy_mistakes : ℕ) : Prop :=
  bryan_score = 20 ∧
  jen_score > bryan_score ∧
  sammy_score = jen_score - 2 ∧
  total_points = 35 ∧
  sammy_mistakes = 7 ∧
  sammy_score = total_points - sammy_mistakes ∧
  jen_score - bryan_score = 10

theorem exam_score_difference :
  ∀ (bryan_score jen_score sammy_score total_points sammy_mistakes : ℕ),
    math_exam_proof bryan_score jen_score sammy_score total_points sammy_mistakes :=
by
  sorry

end exam_score_difference_l629_62986


namespace cookie_bags_theorem_l629_62931

/-- Given a total number of cookies and cookies per bag, calculate the number of bags. -/
def number_of_bags (total_cookies : ℕ) (cookies_per_bag : ℕ) : ℕ :=
  total_cookies / cookies_per_bag

/-- Theorem: Given 33 cookies in total and 11 cookies per bag, the number of bags is 3. -/
theorem cookie_bags_theorem :
  number_of_bags 33 11 = 3 := by
  sorry

end cookie_bags_theorem_l629_62931


namespace square_sum_from_means_l629_62927

theorem square_sum_from_means (x y : ℝ) 
  (h_arithmetic : (x + y) / 2 = 20) 
  (h_geometric : Real.sqrt (x * y) = Real.sqrt 100) : 
  x^2 + y^2 = 1400 := by
sorry

end square_sum_from_means_l629_62927


namespace sqrt_2x_minus_1_meaningful_l629_62961

theorem sqrt_2x_minus_1_meaningful (x : ℝ) : 
  (∃ y : ℝ, y^2 = 2*x - 1) ↔ x ≥ (1/2) :=
sorry

end sqrt_2x_minus_1_meaningful_l629_62961


namespace girls_not_adjacent_arrangements_l629_62912

def num_boys : ℕ := 3
def num_girls : ℕ := 2

theorem girls_not_adjacent_arrangements :
  (num_boys.factorial * (num_boys + 1).choose num_girls) = 72 :=
by sorry

end girls_not_adjacent_arrangements_l629_62912


namespace shorts_cost_calculation_l629_62929

def jacket_cost : ℝ := 14.82
def shirt_cost : ℝ := 12.51
def total_cost : ℝ := 42.33

theorem shorts_cost_calculation : 
  total_cost - jacket_cost - shirt_cost = 15 :=
by sorry

end shorts_cost_calculation_l629_62929


namespace one_fifth_of_ten_x_plus_five_l629_62995

theorem one_fifth_of_ten_x_plus_five (x : ℝ) : (1 / 5) * (10 * x + 5) = 2 * x + 1 := by
  sorry

end one_fifth_of_ten_x_plus_five_l629_62995


namespace min_value_of_3a_plus_2_l629_62926

theorem min_value_of_3a_plus_2 (a : ℝ) (h : 5 * a^2 + 7 * a + 2 = 1) :
  ∃ (m : ℝ), (3 * a + 2 ≥ m) ∧ (∀ (x : ℝ), 5 * x^2 + 7 * x + 2 = 1 → 3 * x + 2 ≥ m) ∧ m = -1 :=
by sorry

end min_value_of_3a_plus_2_l629_62926


namespace jordan_rectangle_width_l629_62936

/-- Given two rectangles with equal area, where one rectangle measures 8 inches by 15 inches
    and the other has a length of 4 inches, prove that the width of the second rectangle is 30 inches. -/
theorem jordan_rectangle_width (area carol_length carol_width jordan_length jordan_width : ℝ) :
  area = carol_length * carol_width →
  area = jordan_length * jordan_width →
  carol_length = 8 →
  carol_width = 15 →
  jordan_length = 4 →
  jordan_width = 30 := by
  sorry

end jordan_rectangle_width_l629_62936


namespace seventh_group_sample_l629_62978

/-- Represents the systematic sampling method described in the problem -/
def systematicSample (m : ℕ) (k : ℕ) : ℕ :=
  10 * (k - 1) + (m + k) % 10

/-- The problem statement translated to a theorem -/
theorem seventh_group_sample :
  ∀ m : ℕ,
  m = 6 →
  systematicSample m 7 = 73 :=
by
  sorry

end seventh_group_sample_l629_62978


namespace work_done_by_resultant_force_l629_62952

/-- Represents a 2D vector -/
structure Vector2D where
  x : ℝ
  y : ℝ

/-- Calculates the dot product of two 2D vectors -/
def dotProduct (v1 v2 : Vector2D) : ℝ :=
  v1.x * v2.x + v1.y * v2.y

/-- Adds two 2D vectors -/
def addVectors (v1 v2 : Vector2D) : Vector2D :=
  ⟨v1.x + v2.x, v1.y + v2.y⟩

/-- Calculates the work done by a force over a displacement -/
def workDone (force displacement : Vector2D) : ℝ :=
  dotProduct force displacement

theorem work_done_by_resultant_force : 
  let f1 : Vector2D := ⟨3, -4⟩
  let f2 : Vector2D := ⟨2, -5⟩
  let f3 : Vector2D := ⟨3, 1⟩
  let a : Vector2D := ⟨1, 1⟩
  let b : Vector2D := ⟨0, 5⟩
  let resultantForce := addVectors (addVectors f1 f2) f3
  let displacement := ⟨b.x - a.x, b.y - a.y⟩
  workDone resultantForce displacement = -40 := by
  sorry


end work_done_by_resultant_force_l629_62952


namespace tangent_slope_sin_pi_sixth_l629_62999

theorem tangent_slope_sin_pi_sixth :
  let f : ℝ → ℝ := λ x ↦ Real.sin x
  (deriv f) (π / 6) = Real.sqrt 3 / 2 := by
  sorry

end tangent_slope_sin_pi_sixth_l629_62999
