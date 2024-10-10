import Mathlib

namespace union_of_sets_l1016_101610

def A (a : ℝ) : Set ℝ := {1, 2^a}
def B (a b : ℝ) : Set ℝ := {a, b}

theorem union_of_sets (a b : ℝ) :
  A a ∩ B a b = {1/4} →
  A a ∪ B a b = {-2, 1, 1/4} := by
  sorry

end union_of_sets_l1016_101610


namespace equality_proof_l1016_101685

theorem equality_proof (a b c p : ℝ) (h : a + b + c = 2 * p) :
  (2 * a * p + b * c) * (2 * b * p + a * c) * (2 * c * p + a * b) =
  (a + b)^2 * (a + c)^2 * (b + c)^2 := by
  sorry

end equality_proof_l1016_101685


namespace a_range_l1016_101655

theorem a_range (a b c : ℝ) 
  (eq1 : a^2 - b*c - 8*a + 7 = 0)
  (eq2 : b^2 + c^2 + b*c - 6*a + 6 = 0) :
  1 ≤ a ∧ a ≤ 9 := by
sorry

end a_range_l1016_101655


namespace equation_solution_l1016_101672

theorem equation_solution : ∃ x : ℚ, (2 * x + 1 = 0) ∧ (x = -1/2) := by
  sorry

end equation_solution_l1016_101672


namespace sum_704_159_base12_l1016_101678

/-- Represents a number in base 12 --/
def Base12 : Type := List (Fin 12)

/-- Converts a base 10 number to base 12 --/
def toBase12 (n : ℕ) : Base12 :=
  sorry

/-- Converts a base 12 number to base 10 --/
def toBase10 (b : Base12) : ℕ :=
  sorry

/-- Adds two base 12 numbers --/
def addBase12 (a b : Base12) : Base12 :=
  sorry

/-- Theorem: The sum of 704₁₂ and 159₁₂ in base 12 is 861₁₂ --/
theorem sum_704_159_base12 :
  addBase12 (toBase12 704) (toBase12 159) = toBase12 861 :=
sorry

end sum_704_159_base12_l1016_101678


namespace mike_toys_total_cost_l1016_101686

def marbles_cost : ℝ := 9.05
def football_cost : ℝ := 4.95
def baseball_cost : ℝ := 6.52
def toy_car_original_cost : ℝ := 5.50
def toy_car_discount_rate : ℝ := 0.10
def puzzle_cost : ℝ := 2.90
def action_figure_cost : ℝ := 8.80

def total_cost : ℝ :=
  marbles_cost +
  football_cost +
  baseball_cost +
  (toy_car_original_cost * (1 - toy_car_discount_rate)) +
  puzzle_cost +
  action_figure_cost

theorem mike_toys_total_cost :
  total_cost = 36.17 := by
  sorry

end mike_toys_total_cost_l1016_101686


namespace questionnaire_survey_l1016_101634

theorem questionnaire_survey (a₁ a₂ a₃ a₄ : ℕ) : 
  a₂ = 60 →
  a₁ + a₂ + a₃ + a₄ = 300 →
  ∃ d : ℤ, a₁ = a₂ - d ∧ a₃ = a₂ + d ∧ a₄ = a₂ + 2*d →
  a₄ = 120 := by
  sorry

end questionnaire_survey_l1016_101634


namespace car_speed_first_hour_l1016_101695

/-- Given a car's speed in the second hour and its average speed over two hours,
    calculate its speed in the first hour. -/
theorem car_speed_first_hour (second_hour_speed : ℝ) (average_speed : ℝ) :
  second_hour_speed = 60 →
  average_speed = 77.5 →
  (second_hour_speed + (average_speed * 2 - second_hour_speed)) / 2 = average_speed →
  average_speed * 2 - second_hour_speed = 95 := by
  sorry

end car_speed_first_hour_l1016_101695


namespace curve_points_difference_l1016_101652

theorem curve_points_difference : 
  ∀ (a b : ℝ), a ≠ b → 
  (4 + a^2 = 8*a - 5) → 
  (4 + b^2 = 8*b - 5) → 
  |a - b| = 2 * Real.sqrt 7 := by
sorry

end curve_points_difference_l1016_101652


namespace birth_year_problem_l1016_101679

theorem birth_year_problem :
  ∃! x : ℕ, 1750 < x ∧ x < 1954 ∧
  (7 * x) % 13 = 11 ∧
  (13 * x) % 11 = 7 ∧
  1954 - x = 86 := by
  sorry

end birth_year_problem_l1016_101679


namespace unique_triangle_l1016_101694

/-- 
A triple of positive integers (a, a, b) represents an acute-angled isosceles triangle 
with perimeter 31 if and only if it satisfies the following conditions:
1. 2a + b = 31 (perimeter condition)
2. a < b < 2a (acute-angled isosceles condition)
-/
def is_valid_triangle (a b : ℕ) : Prop :=
  2 * a + b = 31 ∧ a < b ∧ b < 2 * a

/-- There exists exactly one triple of positive integers (a, a, b) that represents 
an acute-angled isosceles triangle with perimeter 31. -/
theorem unique_triangle : ∃! p : ℕ × ℕ, is_valid_triangle p.1 p.2 := by
  sorry

end unique_triangle_l1016_101694


namespace meaningful_iff_condition_l1016_101690

def is_meaningful (x : ℝ) : Prop :=
  x ≥ -1 ∧ x ≠ 0

theorem meaningful_iff_condition (x : ℝ) :
  is_meaningful x ↔ (∃ y : ℝ, y^2 = x + 1) ∧ x ≠ 0 :=
by sorry

end meaningful_iff_condition_l1016_101690


namespace smallest_result_l1016_101600

def number_set : Finset ℕ := {3, 4, 7, 11, 13, 14}

def is_prime_greater_than_10 (n : ℕ) : Prop :=
  Nat.Prime n ∧ n > 10

def valid_triple (a b c : ℕ) : Prop :=
  a ∈ number_set ∧ b ∈ number_set ∧ c ∈ number_set ∧
  a ≠ b ∧ b ≠ c ∧ a ≠ c ∧
  (is_prime_greater_than_10 a ∨ is_prime_greater_than_10 b ∨ is_prime_greater_than_10 c)

def process_result (a b c : ℕ) : ℕ :=
  (a + b) * c

theorem smallest_result :
  ∀ a b c : ℕ, valid_triple a b c →
    77 ≤ min (process_result a b c) (min (process_result a c b) (process_result b c a)) :=
by sorry

end smallest_result_l1016_101600


namespace parabola_through_point_l1016_101606

theorem parabola_through_point (a b c : ℤ) : 
  5 = a * 2^2 + b * 2 + c → 2 * a + b - c = 1 := by
  sorry

end parabola_through_point_l1016_101606


namespace component_unqualified_l1016_101614

/-- Determines if a component is qualified based on its diameter -/
def is_qualified (measured_diameter : ℝ) (specified_diameter : ℝ) (tolerance : ℝ) : Prop :=
  measured_diameter ≥ specified_diameter - tolerance ∧ 
  measured_diameter ≤ specified_diameter + tolerance

/-- Theorem stating that the component with measured diameter 19.9 mm is unqualified -/
theorem component_unqualified : 
  ¬ is_qualified 19.9 20 0.02 := by
  sorry

end component_unqualified_l1016_101614


namespace ratio_problem_l1016_101698

theorem ratio_problem (a b c d : ℚ) 
  (h1 : a / b = 5 / 4)
  (h2 : c / d = 4 / 3)
  (h3 : d / b = 1 / 7) :
  a / c = 105 / 16 := by
  sorry

end ratio_problem_l1016_101698


namespace quadratic_equation_always_has_real_root_l1016_101657

theorem quadratic_equation_always_has_real_root (a : ℝ) : ∃ x : ℝ, a * x^2 - x = 0 := by
  sorry

end quadratic_equation_always_has_real_root_l1016_101657


namespace average_weight_B_and_C_l1016_101680

theorem average_weight_B_and_C (A B C : ℝ) : 
  (A + B + C) / 3 = 45 →
  (A + B) / 2 = 40 →
  B = 31 →
  (B + C) / 2 = 43 := by
sorry

end average_weight_B_and_C_l1016_101680


namespace proposition_relation_necessary_not_sufficient_l1016_101609

theorem proposition_relation (a : ℝ) : 
  (∀ x : ℝ, a * x^2 + 2 * a * x + 1 > 0) ↔ (0 ≤ a ∧ a < 1) :=
by sorry

theorem necessary_not_sufficient :
  (∃ a : ℝ, (∀ x : ℝ, a * x^2 + 2 * a * x + 1 > 0) ∧ (a ≤ 0 ∨ a ≥ 1)) :=
by sorry

end proposition_relation_necessary_not_sufficient_l1016_101609


namespace quadrilateral_area_bound_l1016_101630

-- Define a quadrilateral type
structure Quadrilateral where
  a : ℝ
  b : ℝ
  c : ℝ
  d : ℝ

-- Define the area function for a quadrilateral
noncomputable def area (q : Quadrilateral) : ℝ := sorry

-- Theorem statement
theorem quadrilateral_area_bound (q : Quadrilateral) :
  area q ≤ (1/4 : ℝ) * (q.a + q.c)^2 + q.b * q.d := by sorry

end quadrilateral_area_bound_l1016_101630


namespace pyramid_base_area_l1016_101663

theorem pyramid_base_area (slant_height height : ℝ) :
  slant_height = 5 →
  height = 7 →
  ∃ (side_length : ℝ), 
    side_length ^ 2 + slant_height ^ 2 = height ^ 2 ∧
    (side_length ^ 2) * 4 = 24 := by
  sorry

end pyramid_base_area_l1016_101663


namespace tan_theta_minus_pi_fourth_l1016_101656

theorem tan_theta_minus_pi_fourth (θ : Real) (h : Real.tan θ = 3) : 
  Real.tan (θ - π/4) = 1/2 := by
  sorry

end tan_theta_minus_pi_fourth_l1016_101656


namespace probability_three_girls_out_of_six_l1016_101629

def binomial_probability (n k : ℕ) (p : ℚ) : ℚ :=
  (n.choose k : ℚ) * p^k * (1 - p)^(n - k)

theorem probability_three_girls_out_of_six :
  binomial_probability 6 3 (1/2) = 5/16 := by
  sorry

end probability_three_girls_out_of_six_l1016_101629


namespace glycerin_mixture_problem_l1016_101647

theorem glycerin_mixture_problem :
  let total_volume : ℝ := 100
  let final_concentration : ℝ := 0.75
  let solution1_volume : ℝ := 75
  let solution1_concentration : ℝ := 0.30
  let solution2_volume : ℝ := 75
  let solution2_concentration : ℝ := x
  (solution1_volume * solution1_concentration + solution2_volume * solution2_concentration = total_volume * final_concentration) →
  x = 0.70 :=
by sorry

end glycerin_mixture_problem_l1016_101647


namespace gcd_98_63_l1016_101627

theorem gcd_98_63 : Nat.gcd 98 63 = 7 := by
  sorry

end gcd_98_63_l1016_101627


namespace distance_XT_equals_twenty_l1016_101632

/-- Represents a square pyramid -/
structure SquarePyramid where
  baseLength : ℝ
  height : ℝ

/-- Represents a frustum created by cutting a pyramid -/
structure Frustum where
  basePyramid : SquarePyramid
  volumeRatio : ℝ  -- Ratio of original pyramid volume to smaller pyramid volume

/-- The distance from the center of the frustum's circumsphere to the apex of the original pyramid -/
def distanceXT (f : Frustum) : ℝ := sorry

theorem distance_XT_equals_twenty (f : Frustum) 
  (h1 : f.basePyramid.baseLength = 10)
  (h2 : f.basePyramid.height = 20)
  (h3 : f.volumeRatio = 9) :
  distanceXT f = 20 := by sorry

end distance_XT_equals_twenty_l1016_101632


namespace matrix_equation_proof_l1016_101664

def N : Matrix (Fin 2) (Fin 2) ℝ := !![2, 4; 1, 2]

theorem matrix_equation_proof :
  N^3 - 3 • N^2 + 3 • N = !![6, 12; 3, 6] := by sorry

end matrix_equation_proof_l1016_101664


namespace weight_of_BaBr2_l1016_101602

/-- The molecular weight of BaBr2 in grams per mole -/
def molecular_weight_BaBr2 : ℝ := 137.33 + 2 * 79.90

/-- The number of moles of BaBr2 -/
def moles_BaBr2 : ℝ := 8

/-- Calculates the total weight of a given number of moles of BaBr2 -/
def total_weight (mw : ℝ) (moles : ℝ) : ℝ := mw * moles

/-- Theorem stating that the total weight of 8 moles of BaBr2 is 2377.04 grams -/
theorem weight_of_BaBr2 : 
  total_weight molecular_weight_BaBr2 moles_BaBr2 = 2377.04 := by
  sorry

end weight_of_BaBr2_l1016_101602


namespace range_of_f_range_of_f_complete_l1016_101646

def f (x : ℝ) : ℝ := x^2 - 2*x + 3

theorem range_of_f :
  ∀ y ∈ Set.range f,
  (∃ x : ℝ, -1 ≤ x ∧ x ≤ 2 ∧ f x = y) →
  2 ≤ y ∧ y ≤ 6 :=
by sorry

theorem range_of_f_complete :
  ∀ y : ℝ, 2 ≤ y ∧ y ≤ 6 →
  ∃ x : ℝ, -1 ≤ x ∧ x ≤ 2 ∧ f x = y :=
by sorry

end range_of_f_range_of_f_complete_l1016_101646


namespace rectangular_field_area_l1016_101693

theorem rectangular_field_area (L W : ℝ) (h1 : L = 30) (h2 : 2 * W + L = 84) : L * W = 810 := by
  sorry

end rectangular_field_area_l1016_101693


namespace shelter_ratio_l1016_101637

theorem shelter_ratio (initial_cats : ℕ) (initial_dogs : ℕ) (additional_dogs : ℕ) :
  initial_cats = 45 →
  (initial_cats : ℚ) / initial_dogs = 15 / 7 →
  additional_dogs = 12 →
  (initial_cats : ℚ) / (initial_dogs + additional_dogs) = 15 / 11 :=
by sorry

end shelter_ratio_l1016_101637


namespace circle_and_m_value_l1016_101619

-- Define the curve
def curve (x y : ℝ) : Prop := y = x^2 - 4*x + 3

-- Define the circle C
def circle_C (x y : ℝ) : Prop := (x - 2)^2 + (y - 2)^2 = 5

-- Define the line
def line (x y m : ℝ) : Prop := x + y + m = 0

-- Define perpendicularity condition
def perpendicular (x₁ y₁ x₂ y₂ : ℝ) : Prop := x₁ * x₂ + y₁ * y₂ = 0

theorem circle_and_m_value :
  ∃ (x₁ y₁ x₂ y₂ : ℝ),
    (curve 0 3 ∧ curve 1 0 ∧ curve 3 0) ∧  -- Intersection points with axes
    (circle_C 0 3 ∧ circle_C 1 0 ∧ circle_C 3 0) ∧  -- These points lie on circle C
    (∃ m : ℝ, 
      line x₁ y₁ m ∧ line x₂ y₂ m ∧  -- A and B lie on the line
      circle_C x₁ y₁ ∧ circle_C x₂ y₂ ∧  -- A and B lie on circle C
      perpendicular x₁ y₁ x₂ y₂ ∧  -- OA is perpendicular to OB
      (m = -1 ∨ m = -3))  -- The value of m
  :=
sorry

end circle_and_m_value_l1016_101619


namespace ellipse_x_intercept_l1016_101608

-- Define the ellipse
def ellipse (x y : ℝ) : Prop :=
  Real.sqrt (x^2 + y^2) + Real.sqrt ((x - 4)^2 + (y - 3)^2) = 7

-- Define the foci
def F₁ : ℝ × ℝ := (0, 3)
def F₂ : ℝ × ℝ := (4, 0)

-- Theorem statement
theorem ellipse_x_intercept :
  ellipse 0 0 → -- The ellipse passes through (0,0)
  (∃ x : ℝ, x ≠ 0 ∧ ellipse x 0) → -- There exists another x-intercept
  (∃ x : ℝ, x = 56/11 ∧ ellipse x 0) -- The other x-intercept is (56/11, 0)
  := by sorry

end ellipse_x_intercept_l1016_101608


namespace taylor_score_ratio_l1016_101625

/-- Given the conditions for Taylor's score mixture, prove the ratio of white to black scores -/
theorem taylor_score_ratio :
  ∀ (white black : ℕ),
  white + black = 78 →
  2 * (black - white) = 3 * 4 →
  (white : ℚ) / black = 6 / 7 :=
by
  sorry

end taylor_score_ratio_l1016_101625


namespace cosine_product_sqrt_eight_l1016_101651

theorem cosine_product_sqrt_eight : 
  Real.sqrt ((3 - Real.cos (π / 8) ^ 2) * (3 - Real.cos (π / 4) ^ 2) * (3 - Real.cos (3 * π / 8) ^ 2)) = 8 := by
  sorry

end cosine_product_sqrt_eight_l1016_101651


namespace square_sum_equals_one_l1016_101621

theorem square_sum_equals_one (a b : ℝ) (h : a + b = -1) : a^2 + b^2 + 2*a*b = 1 := by
  sorry

end square_sum_equals_one_l1016_101621


namespace perpendicular_points_constant_sum_l1016_101675

/-- The curve E in polar coordinates -/
def curve_E (ρ θ : ℝ) : Prop :=
  ρ^2 * (1/3 * Real.cos θ^2 + 1/2 * Real.sin θ^2) = 1

/-- Theorem: For any two perpendicular points on curve E, the sum of reciprocals of their squared distances from the origin is constant -/
theorem perpendicular_points_constant_sum (ρ₁ ρ₂ θ : ℝ) :
  curve_E ρ₁ θ → curve_E ρ₂ (θ + π/2) → 1/ρ₁^2 + 1/ρ₂^2 = 5/6 := by
  sorry

#check perpendicular_points_constant_sum

end perpendicular_points_constant_sum_l1016_101675


namespace school_students_l1016_101613

/-- The number of students in a school -/
theorem school_students (boys : ℕ) (girls : ℕ) : 
  boys = 272 → girls = boys + 106 → boys + girls = 650 := by
  sorry

end school_students_l1016_101613


namespace decagon_diagonals_l1016_101617

/-- The number of diagonals in a polygon with n sides -/
def num_diagonals (n : ℕ) : ℕ := n * (n - 3) / 2

/-- A decagon has 10 sides -/
def decagon_sides : ℕ := 10

/-- Theorem: The number of diagonals in a regular decagon is 35 -/
theorem decagon_diagonals : num_diagonals decagon_sides = 35 := by
  sorry

end decagon_diagonals_l1016_101617


namespace polynomial_equality_l1016_101659

theorem polynomial_equality (x : ℝ) : 
  (3*x^3 + 2*x^2 + 5*x + 9)*(x - 2) - (x - 2)*(2*x^3 + 5*x^2 - 74) + (4*x - 17)*(x - 2)*(x + 4) 
  = x^4 + 2*x^3 - 5*x^2 + 9*x - 30 := by
sorry

end polynomial_equality_l1016_101659


namespace least_positive_integer_with_remainders_l1016_101616

theorem least_positive_integer_with_remainders : ∃! N : ℕ,
  N > 0 ∧
  N % 4 = 3 ∧
  N % 5 = 4 ∧
  N % 6 = 5 ∧
  N % 7 = 6 ∧
  ∀ M : ℕ, (M > 0 ∧ M % 4 = 3 ∧ M % 5 = 4 ∧ M % 6 = 5 ∧ M % 7 = 6) → N ≤ M :=
by
  -- The proof goes here
  sorry

end least_positive_integer_with_remainders_l1016_101616


namespace cube_coloring_count_dodecahedron_coloring_count_l1016_101689

/-- The number of rotational symmetries of a cube -/
def cube_rotations : ℕ := 24

/-- The number of rotational symmetries of a dodecahedron -/
def dodecahedron_rotations : ℕ := 60

/-- The number of faces of a cube -/
def cube_faces : ℕ := 6

/-- The number of faces of a dodecahedron -/
def dodecahedron_faces : ℕ := 12

/-- Calculates the number of geometrically distinct colorings for a polyhedron -/
def distinct_colorings (faces : ℕ) (rotations : ℕ) : ℕ :=
  (Nat.factorial faces) / rotations

theorem cube_coloring_count :
  distinct_colorings cube_faces cube_rotations = 30 := by sorry

theorem dodecahedron_coloring_count :
  distinct_colorings dodecahedron_faces dodecahedron_rotations = 7983360 := by sorry

end cube_coloring_count_dodecahedron_coloring_count_l1016_101689


namespace max_value_of_f_l1016_101615

/-- The quadratic function f(x) = -2x^2 + 4x - 6 -/
def f (x : ℝ) : ℝ := -2 * x^2 + 4 * x - 6

/-- The maximum value of f(x) is -4 -/
theorem max_value_of_f :
  ∃ (m : ℝ), m = -4 ∧ ∀ (x : ℝ), f x ≤ m :=
sorry

end max_value_of_f_l1016_101615


namespace sqrt_factorial_fraction_l1016_101676

theorem sqrt_factorial_fraction : 
  let factorial_10 : ℕ := 10 * 9 * 8 * 7 * 6 * 5 * 4 * 3 * 2 * 1
  let denominator : ℕ := 2 * 3 * 7 * 7
  Real.sqrt (factorial_10 / denominator) = 120 * Real.sqrt 6 := by
sorry

end sqrt_factorial_fraction_l1016_101676


namespace marigold_sale_ratio_l1016_101604

/-- Proves that the ratio of marigolds sold on the third day to the second day is 2:1 --/
theorem marigold_sale_ratio :
  ∀ (day3 : ℕ),
  14 + 25 + day3 = 89 →
  (day3 : ℚ) / 25 = 2 := by
  sorry

end marigold_sale_ratio_l1016_101604


namespace smallest_number_l1016_101692

/-- Converts a number from base b to decimal --/
def to_decimal (digits : List Nat) (b : Nat) : Nat :=
  digits.reverse.enum.foldl (fun acc (i, d) => acc + d * b^i) 0

theorem smallest_number : 
  let base_9 := to_decimal [5, 8] 9
  let base_4 := to_decimal [0, 0, 0, 1] 4
  let base_2 := to_decimal [1, 1, 1, 1, 1, 1] 2
  base_2 < base_4 ∧ base_2 < base_9 := by
  sorry

end smallest_number_l1016_101692


namespace digit_150_is_5_l1016_101636

/-- The decimal representation of 31/198 -/
def decimal_rep : ℚ := 31 / 198

/-- The period of the decimal representation -/
def period : ℕ := 6

/-- The nth digit after the decimal point in the decimal representation -/
noncomputable def nth_digit (n : ℕ) : ℕ := sorry

/-- The 150th digit after the decimal point in the decimal representation of 31/198 is 5 -/
theorem digit_150_is_5 : nth_digit 150 = 5 := by sorry

end digit_150_is_5_l1016_101636


namespace three_dice_same_number_l1016_101671

/-- A standard six-sided die -/
def StandardDie := Fin 6

/-- The probability of a specific outcome on a standard die -/
def prob_specific_outcome : ℚ := 1 / 6

/-- The probability of all three dice showing the same number -/
def prob_all_same : ℚ := 1 / 36

/-- Theorem: The probability of three standard six-sided dice showing the same number
    when tossed simultaneously is 1/36 -/
theorem three_dice_same_number :
  (1 : ℚ) * prob_specific_outcome * prob_specific_outcome = prob_all_same := by
  sorry

end three_dice_same_number_l1016_101671


namespace golf_distance_l1016_101660

/-- 
Given a golf scenario where:
1. The distance from the starting tee to the hole is 250 yards.
2. On the second turn, the ball traveled half as far as it did on the first turn.
3. After the second turn, the ball landed 20 yards beyond the hole.
This theorem proves that the distance the ball traveled on the first turn is 180 yards.
-/
theorem golf_distance (first_turn : ℝ) (second_turn : ℝ) : 
  (first_turn + second_turn = 250 + 20) →  -- Total distance is to the hole plus 20 yards beyond
  (second_turn = first_turn / 2) →         -- Second turn is half of the first turn
  (first_turn = 180) :=                    -- The distance of the first turn is 180 yards
by sorry

end golf_distance_l1016_101660


namespace father_walking_time_l1016_101623

/-- The time (in minutes) it takes Xiaoming to cycle from the meeting point to B -/
def meeting_to_B : ℝ := 18

/-- Xiaoming's cycling speed is 4 times his father's walking speed -/
def speed_ratio : ℝ := 4

/-- The time (in minutes) it takes Xiaoming's father to walk from the meeting point to A -/
def father_time : ℝ := 288

theorem father_walking_time :
  ∀ (xiaoming_speed father_speed : ℝ),
  xiaoming_speed > 0 ∧ father_speed > 0 →
  xiaoming_speed = speed_ratio * father_speed →
  father_time = 4 * (speed_ratio * meeting_to_B) := by
  sorry

end father_walking_time_l1016_101623


namespace cosine_value_l1016_101603

theorem cosine_value (α : Real) 
  (h1 : 0 < α) (h2 : α < π) 
  (h3 : 3 * Real.sin (2 * α) = Real.sin α) : 
  Real.cos (α - π) = -1/6 := by
sorry

end cosine_value_l1016_101603


namespace smallest_divisible_integer_l1016_101699

theorem smallest_divisible_integer : ∃ (M : ℕ), 
  (M = 362) ∧ 
  (∀ (k : ℕ), k < M → ¬(
    (∃ (i : Fin 3), 2^2 ∣ (k + i)) ∧
    (∃ (i : Fin 3), 3^2 ∣ (k + i)) ∧
    (∃ (i : Fin 3), 7^2 ∣ (k + i)) ∧
    (∃ (i : Fin 3), 11^2 ∣ (k + i))
  )) ∧
  (∃ (i : Fin 3), 2^2 ∣ (M + i)) ∧
  (∃ (i : Fin 3), 3^2 ∣ (M + i)) ∧
  (∃ (i : Fin 3), 7^2 ∣ (M + i)) ∧
  (∃ (i : Fin 3), 11^2 ∣ (M + i)) :=
by sorry

end smallest_divisible_integer_l1016_101699


namespace factorization_equality_l1016_101668

theorem factorization_equality (a b : ℝ) : a^2 * b - b^3 = b * (a + b) * (a - b) := by sorry

end factorization_equality_l1016_101668


namespace sphere_radius_is_sqrt_six_over_four_l1016_101673

/-- A sphere circumscribing a right circular cone -/
structure CircumscribedCone where
  /-- The radius of the circumscribing sphere -/
  sphere_radius : ℝ
  /-- The diameter of the base of the cone -/
  base_diameter : ℝ
  /-- Assertion that the base diameter is 1 -/
  base_diameter_is_one : base_diameter = 1
  /-- Assertion that the apex of the cone is on the sphere -/
  apex_on_sphere : True
  /-- Assertion about the perpendicularity condition -/
  perpendicular_condition : True

/-- Theorem stating that the radius of the circumscribing sphere is √6/4 -/
theorem sphere_radius_is_sqrt_six_over_four (cone : CircumscribedCone) :
  cone.sphere_radius = Real.sqrt 6 / 4 := by
  sorry

end sphere_radius_is_sqrt_six_over_four_l1016_101673


namespace correlation_coefficient_is_one_l1016_101677

/-- A structure representing a set of sample points -/
structure SampleData where
  n : ℕ
  x : Fin n → ℝ
  y : Fin n → ℝ
  n_ge_2 : n ≥ 2
  not_all_x_equal : ∃ i j, i ≠ j ∧ x i ≠ x j

/-- The sample correlation coefficient -/
def sampleCorrelationCoefficient (data : SampleData) : ℝ := sorry

/-- All points lie on the line y = 2x + 1 -/
def allPointsOnLine (data : SampleData) : Prop :=
  ∀ i, data.y i = 2 * data.x i + 1

/-- Theorem stating that if all points lie on y = 2x + 1, then the correlation coefficient is 1 -/
theorem correlation_coefficient_is_one (data : SampleData) 
  (h : allPointsOnLine data) : sampleCorrelationCoefficient data = 1 := by sorry

end correlation_coefficient_is_one_l1016_101677


namespace union_of_M_and_N_l1016_101658

def M : Set ℝ := {x | x^2 - x - 12 = 0}
def N : Set ℝ := {x | x^2 + 3*x = 0}

theorem union_of_M_and_N : M ∪ N = {0, -3, 4} := by sorry

end union_of_M_and_N_l1016_101658


namespace function_property_implies_k_equals_8_l1016_101601

/-- Given a function f: ℝ → ℝ satisfying certain properties, prove that k = 8 -/
theorem function_property_implies_k_equals_8 (f : ℝ → ℝ) (k : ℝ) 
  (h1 : f 1 = 1)
  (h2 : ∀ x y, f (x + y) = f x + f y + k * x * y - 2)
  (h3 : f 7 = 163) :
  k = 8 := by
  sorry

end function_property_implies_k_equals_8_l1016_101601


namespace quadratic_real_roots_l1016_101670

theorem quadratic_real_roots (k : ℝ) :
  (∃ x : ℝ, x^2 - 6*x + 9*k = 0) ↔ k ≤ 1 := by
  sorry

end quadratic_real_roots_l1016_101670


namespace liza_reading_speed_l1016_101642

/-- Given that Suzie reads 15 pages in an hour and Liza reads 15 more pages than Suzie in 3 hours,
    prove that Liza reads 20 pages in an hour. -/
theorem liza_reading_speed (suzie_pages_per_hour : ℕ) (liza_extra_pages : ℕ) :
  suzie_pages_per_hour = 15 →
  liza_extra_pages = 15 →
  ∃ (liza_pages_per_hour : ℕ),
    liza_pages_per_hour * 3 = suzie_pages_per_hour * 3 + liza_extra_pages ∧
    liza_pages_per_hour = 20 :=
by sorry

end liza_reading_speed_l1016_101642


namespace positive_integer_solutions_m_value_when_sum_zero_fixed_solution_integer_m_for_integer_x_l1016_101691

-- Define the system of equations
def system (x y m : ℝ) : Prop :=
  x + 2*y - 6 = 0 ∧ x - 2*y + m*x + 5 = 0

-- Theorem 1: Positive integer solutions
theorem positive_integer_solutions :
  ∀ x y : ℤ, x > 0 ∧ y > 0 ∧ x + 2*y - 6 = 0 → (x = 2 ∧ y = 2) ∨ (x = 4 ∧ y = 1) :=
sorry

-- Theorem 2: Value of m when x + y = 0
theorem m_value_when_sum_zero :
  ∀ x y m : ℝ, system x y m ∧ x + y = 0 → m = -13/6 :=
sorry

-- Theorem 3: Fixed solution regardless of m
theorem fixed_solution :
  ∀ m : ℝ, 0 - 2*2.5 + m*0 + 5 = 0 :=
sorry

-- Theorem 4: Integer values of m for integer x
theorem integer_m_for_integer_x :
  ∀ x : ℤ, ∀ m : ℤ, (∃ y : ℝ, system x y m) → m = -1 ∨ m = -3 :=
sorry

end positive_integer_solutions_m_value_when_sum_zero_fixed_solution_integer_m_for_integer_x_l1016_101691


namespace max_area_isosceles_trapezoidal_canal_l1016_101662

/-- 
Given an isosceles trapezoidal canal where the legs are equal to the smaller base,
this theorem states that the cross-sectional area is maximized when the angle of 
inclination of the legs is π/3 radians.
-/
theorem max_area_isosceles_trapezoidal_canal :
  ∀ (a : ℝ) (α : ℝ), 
  0 < a → 
  0 < α → 
  α < π / 2 →
  let S := a^2 * (1 + Real.cos α) * Real.sin α
  ∀ (β : ℝ), 0 < β → β < π / 2 → 
  a^2 * (1 + Real.cos β) * Real.sin β ≤ S →
  α = π / 3 :=
by sorry


end max_area_isosceles_trapezoidal_canal_l1016_101662


namespace shape_cell_count_l1016_101635

theorem shape_cell_count (n : ℕ) : 
  n < 16 ∧ 
  n % 4 = 0 ∧ 
  n % 3 = 0 → 
  n = 12 := by sorry

end shape_cell_count_l1016_101635


namespace cos_two_theta_value_l1016_101653

theorem cos_two_theta_value (θ : Real) 
  (h : Real.sin (θ / 2) - Real.cos (θ / 2) = Real.sqrt 6 / 3) : 
  Real.cos (2 * θ) = 7 / 9 := by
  sorry

end cos_two_theta_value_l1016_101653


namespace two_person_travel_problem_l1016_101626

/-- The problem of two people traveling between two locations --/
theorem two_person_travel_problem 
  (distance : ℝ) 
  (total_time : ℝ) 
  (speed_difference : ℝ) :
  distance = 25.5 ∧ 
  total_time = 3 ∧ 
  speed_difference = 2 →
  ∃ (speed_A speed_B : ℝ),
    speed_A = 2 * speed_B + speed_difference ∧
    speed_B * total_time + speed_A * total_time = 2 * distance ∧
    speed_A = 12 ∧
    speed_B = 5 := by sorry

end two_person_travel_problem_l1016_101626


namespace triangle_with_arithmetic_sides_l1016_101624

/-- 
A triangle with sides forming an arithmetic sequence with common difference 1 and area 6 
has sides 3, 4, and 5, and one of its angles is a right angle.
-/
theorem triangle_with_arithmetic_sides (a b c : ℝ) (α β γ : ℝ) :
  a > 0 ∧ b > 0 ∧ c > 0 →  -- sides are positive
  b = a + 1 ∧ c = b + 1 →  -- sides form arithmetic sequence with difference 1
  (a + b + c) / 2 * ((a + b + c) / 2 - a) * ((a + b + c) / 2 - b) * ((a + b + c) / 2 - c) = 36 →  -- area is 6 (Heron's formula)
  α + β + γ = π →  -- sum of angles is π
  a * a = b * b + c * c - 2 * b * c * Real.cos α →  -- law of cosines for side a
  b * b = a * a + c * c - 2 * a * c * Real.cos β →  -- law of cosines for side b
  c * c = a * a + b * b - 2 * a * b * Real.cos γ →  -- law of cosines for side c
  (a = 3 ∧ b = 4 ∧ c = 5) ∧ γ = π / 2 := by sorry

end triangle_with_arithmetic_sides_l1016_101624


namespace spinner_product_even_probability_l1016_101682

def spinner1 : Finset Nat := {2, 5, 7, 11}
def spinner2 : Finset Nat := {3, 4, 6, 8, 10}

def isEven (n : Nat) : Bool := n % 2 = 0

theorem spinner_product_even_probability :
  let totalOutcomes := spinner1.card * spinner2.card
  let evenProductOutcomes := (spinner1.card * spinner2.card) - 
    (spinner1.filter (λ x => ¬isEven x)).card * (spinner2.filter (λ x => ¬isEven x)).card
  (evenProductOutcomes : ℚ) / totalOutcomes = 17 / 20 := by
  sorry

end spinner_product_even_probability_l1016_101682


namespace subset_implies_a_zero_l1016_101667

theorem subset_implies_a_zero (a : ℝ) :
  let P : Set ℝ := {x | x^2 ≠ 1}
  let Q : Set ℝ := {x | a * x = 1}
  Q ⊆ P → a = 0 := by
sorry

end subset_implies_a_zero_l1016_101667


namespace triangle_angle_c_l1016_101665

theorem triangle_angle_c (A B C : ℝ) (a b c : ℝ) : 
  A = 80 * π / 180 →
  a^2 = b * (b + c) →
  A + B + C = π →
  a = 2 * Real.sin (A / 2) →
  b = 2 * Real.sin (B / 2) →
  c = 2 * Real.sin (C / 2) →
  C = π / 3 := by
sorry

end triangle_angle_c_l1016_101665


namespace cube_coloring_theorem_l1016_101607

/-- Represents the symmetry group of a cube -/
def CubeSymmetryGroup : Type := Unit

/-- The order of the cube symmetry group -/
def symmetryGroupOrder : ℕ := 24

/-- The total number of ways to color a cube with 6 colors without considering rotations -/
def totalColorings : ℕ := 720

/-- The number of distinct colorings of a cube with 6 colors, considering rotational symmetries -/
def distinctColorings : ℕ := totalColorings / symmetryGroupOrder

theorem cube_coloring_theorem :
  distinctColorings = 30 :=
sorry

end cube_coloring_theorem_l1016_101607


namespace eight_mile_taxi_cost_l1016_101654

/-- Calculates the cost of a taxi ride given the base fare, per-mile charge, and distance traveled. -/
def taxi_cost (base_fare : ℝ) (per_mile_charge : ℝ) (distance : ℝ) : ℝ :=
  base_fare + per_mile_charge * distance

/-- Proves that the cost of an 8-mile taxi ride with a base fare of $2.00 and a per-mile charge of $0.30 is equal to $4.40. -/
theorem eight_mile_taxi_cost :
  taxi_cost 2.00 0.30 8 = 4.40 := by
  sorry

end eight_mile_taxi_cost_l1016_101654


namespace unfair_die_expected_value_l1016_101687

/-- An unfair eight-sided die with specific probabilities -/
structure UnfairDie where
  /-- The probability of rolling an 8 -/
  prob_eight : ℚ
  /-- The probability of rolling any number from 1 to 7 -/
  prob_others : ℚ
  /-- The probability of rolling an 8 is 3/7 -/
  h_prob_eight : prob_eight = 3/7
  /-- The probabilities sum to 1 -/
  h_sum_to_one : prob_eight + 7 * prob_others = 1

/-- The expected value of rolling the unfair die -/
def expected_value (d : UnfairDie) : ℚ :=
  d.prob_others * (1 + 2 + 3 + 4 + 5 + 6 + 7) + 8 * d.prob_eight

/-- Theorem stating the expected value of the unfair die -/
theorem unfair_die_expected_value (d : UnfairDie) :
  expected_value d = 40/7 := by
  sorry

#eval (40 : ℚ) / 7

end unfair_die_expected_value_l1016_101687


namespace xiaohong_journey_time_l1016_101669

/-- Represents Xiaohong's journey to the meeting venue -/
structure Journey where
  initialSpeed : ℝ
  totalTime : ℝ

/-- The conditions of Xiaohong's journey -/
def journeyConditions (j : Journey) : Prop :=
  j.initialSpeed * 30 + (j.initialSpeed * 1.25) * (j.totalTime - 55) = j.initialSpeed * j.totalTime

/-- Theorem stating that the total time of Xiaohong's journey is 155 minutes -/
theorem xiaohong_journey_time :
  ∃ j : Journey, journeyConditions j ∧ j.totalTime = 155 := by
  sorry


end xiaohong_journey_time_l1016_101669


namespace p_sufficient_not_necessary_l1016_101628

-- Define the propositions p and q as functions of x
def p (x : ℝ) : Prop := Real.log (x - 1) < 0
def q (x : ℝ) : Prop := |1 - x| < 2

-- State the theorem
theorem p_sufficient_not_necessary :
  (∀ x, p x → q x) ∧ (∃ x, q x ∧ ¬(p x)) := by sorry

end p_sufficient_not_necessary_l1016_101628


namespace combination_sum_permutation_ratio_l1016_101696

-- Define combination function
def C (n : ℕ) (r : ℕ) : ℕ := 
  if r ≤ n then (Nat.factorial n) / ((Nat.factorial r) * (Nat.factorial (n - r)))
  else 0

-- Define permutation function
def A (n : ℕ) (r : ℕ) : ℕ := 
  if r ≤ n then (Nat.factorial n) / (Nat.factorial (n - r))
  else 0

-- Theorem 1: Combination sum
theorem combination_sum : C 9 2 + C 9 3 = 120 := by sorry

-- Theorem 2: Permutation ratio
theorem permutation_ratio (n m : ℕ) (h : m < n) : 
  (A n m) / (A (n-1) (m-1)) = n := by sorry

end combination_sum_permutation_ratio_l1016_101696


namespace area_triangle_AOB_l1016_101643

/-- Given two points A and B in polar coordinates, prove that the area of triangle AOB is 6 -/
theorem area_triangle_AOB (A B : ℝ × ℝ) : 
  A.1 = 3 ∧ A.2 = π/3 ∧ B.1 = 4 ∧ B.2 = 5*π/6 → 
  (1/2) * A.1 * B.1 * Real.sin (B.2 - A.2) = 6 := by
sorry

end area_triangle_AOB_l1016_101643


namespace tim_additional_water_consumption_l1016_101645

/-- Represents the amount of water Tim drinks -/
structure WaterConsumption where
  bottles_per_day : ℕ
  quarts_per_bottle : ℚ
  total_ounces_per_week : ℕ
  ounces_per_quart : ℕ
  days_per_week : ℕ

/-- Calculates the additional ounces of water Tim drinks daily -/
def additional_daily_ounces (w : WaterConsumption) : ℚ :=
  ((w.total_ounces_per_week : ℚ) - 
   (w.bottles_per_day : ℚ) * w.quarts_per_bottle * (w.ounces_per_quart : ℚ) * (w.days_per_week : ℚ)) / 
  (w.days_per_week : ℚ)

/-- Theorem stating that Tim drinks an additional 20 ounces of water daily -/
theorem tim_additional_water_consumption :
  let w : WaterConsumption := {
    bottles_per_day := 2,
    quarts_per_bottle := 3/2,
    total_ounces_per_week := 812,
    ounces_per_quart := 32,
    days_per_week := 7
  }
  additional_daily_ounces w = 20 := by
  sorry

end tim_additional_water_consumption_l1016_101645


namespace billboard_area_l1016_101620

/-- The area of a rectangular billboard with perimeter 46 feet and width 8 feet is 120 square feet. -/
theorem billboard_area (perimeter width : ℝ) (h1 : perimeter = 46) (h2 : width = 8) :
  let length := (perimeter - 2 * width) / 2
  width * length = 120 :=
by sorry

end billboard_area_l1016_101620


namespace exists_function_with_properties_l1016_101683

-- Define the function type
def RealFunction := ℝ → ℝ

-- Define the properties of the function
def PassesThroughPoint (f : RealFunction) : Prop :=
  f (-2) = 1

def IncreasingInSecondQuadrant (f : RealFunction) : Prop :=
  ∀ x₁ x₂, x₁ < x₂ → x₁ < 0 → x₂ < 0 → f x₁ > 0 → f x₂ > 0 → f x₁ < f x₂

-- Theorem statement
theorem exists_function_with_properties :
  ∃ f : RealFunction, PassesThroughPoint f ∧ IncreasingInSecondQuadrant f :=
sorry

end exists_function_with_properties_l1016_101683


namespace chord_length_l1016_101641

theorem chord_length (r d : ℝ) (hr : r = 5) (hd : d = 4) :
  let chord_length := 2 * Real.sqrt (r^2 - d^2)
  chord_length = 6 := by sorry

end chord_length_l1016_101641


namespace scientific_notation_equivalence_l1016_101605

/-- Represents the value in billion yuan -/
def original_value : ℝ := 8450

/-- Represents the coefficient in scientific notation -/
def coefficient : ℝ := 8.45

/-- Represents the exponent in scientific notation -/
def exponent : ℤ := 3

/-- Theorem stating that the original value is equal to its scientific notation representation -/
theorem scientific_notation_equivalence :
  original_value = coefficient * (10 : ℝ) ^ exponent :=
sorry

end scientific_notation_equivalence_l1016_101605


namespace always_balanced_arrangement_l1016_101684

-- Define the cube type
structure Cube :=
  (blue_faces : Nat)
  (red_faces : Nat)

-- Define the set of 8 cubes
def CubeSet := List Cube

-- Define the property of a valid cube set
def ValidCubeSet (cs : CubeSet) : Prop :=
  cs.length = 8 ∧
  (cs.map (·.blue_faces)).sum = 24 ∧
  (cs.map (·.red_faces)).sum = 24

-- Define the property of a balanced surface
def BalancedSurface (surface_blue : Nat) (surface_red : Nat) : Prop :=
  surface_blue = surface_red ∧ surface_blue + surface_red = 24

-- Main theorem
theorem always_balanced_arrangement (cs : CubeSet) 
  (h : ValidCubeSet cs) : 
  ∃ (surface_blue surface_red : Nat), 
    BalancedSurface surface_blue surface_red :=
sorry

end always_balanced_arrangement_l1016_101684


namespace probability_divisible_by_five_l1016_101631

-- Define a three-digit positive integer with a ones digit of 5
def three_digit_ending_in_five (n : ℕ) : Prop :=
  100 ≤ n ∧ n ≤ 999 ∧ n % 10 = 5

-- Define divisibility by 5
def divisible_by_five (n : ℕ) : Prop :=
  n % 5 = 0

-- Theorem statement
theorem probability_divisible_by_five :
  ∀ n : ℕ, three_digit_ending_in_five n → divisible_by_five n :=
by
  sorry

end probability_divisible_by_five_l1016_101631


namespace paint_needed_to_buy_l1016_101650

theorem paint_needed_to_buy (total_paint : ℕ) (available_paint : ℕ) : 
  total_paint = 333 → available_paint = 157 → total_paint - available_paint = 176 := by
  sorry

end paint_needed_to_buy_l1016_101650


namespace existence_of_special_number_l1016_101640

theorem existence_of_special_number : ∃ N : ℕ, 
  (∃ k : ℕ, k < 150 ∧ k + 1 ≤ 150 ∧ ¬(k ∣ N) ∧ ¬((k + 1) ∣ N)) ∧ 
  (∀ m : ℕ, m ≤ 150 → (∃ k : ℕ, k < 150 ∧ k + 1 ≤ 150 ∧ m ≠ k ∧ m ≠ k + 1) → m ∣ N) :=
sorry

end existence_of_special_number_l1016_101640


namespace point_coordinates_l1016_101639

-- Define a point in 2D space
def Point := ℝ × ℝ

-- Define the second quadrant
def second_quadrant (p : Point) : Prop :=
  p.1 < 0 ∧ p.2 > 0

-- Define distance to x-axis
def distance_to_x_axis (p : Point) : ℝ :=
  |p.2|

-- Define distance to y-axis
def distance_to_y_axis (p : Point) : ℝ :=
  |p.1|

theorem point_coordinates :
  ∀ p : Point,
    second_quadrant p →
    distance_to_x_axis p = 4 →
    distance_to_y_axis p = 3 →
    p = (-3, 4) :=
by
  sorry

end point_coordinates_l1016_101639


namespace max_ratio_APPE_l1016_101612

/-- A point in the Z × Z lattice -/
structure LatticePoint where
  x : ℤ
  y : ℤ

/-- Triangle defined by three lattice points -/
structure LatticeTriangle where
  A : LatticePoint
  B : LatticePoint
  C : LatticePoint

/-- Checks if a point is inside a triangle -/
def isInside (P : LatticePoint) (T : LatticeTriangle) : Prop := sorry

/-- Checks if a point is the unique interior lattice point of a triangle -/
def isUniqueInteriorPoint (P : LatticePoint) (T : LatticeTriangle) : Prop :=
  isInside P T ∧ ∀ Q : LatticePoint, isInside Q T → Q = P

/-- Intersection point of line AP and BC -/
def intersectionPoint (A P B C : LatticePoint) : LatticePoint := sorry

/-- Calculates the ratio AP/PE -/
def ratioAPPE (A P E : LatticePoint) : ℚ := sorry

/-- Main theorem: The maximum value of AP/PE is 5 -/
theorem max_ratio_APPE (T : LatticeTriangle) (P : LatticePoint) 
  (h : isUniqueInteriorPoint P T) :
  ∃ (M : ℚ), (∀ (A B C : LatticePoint),
    T = ⟨A, B, C⟩ → 
    let E := intersectionPoint A P B C
    ratioAPPE A P E ≤ M) ∧
  M = 5 := by
  sorry

end max_ratio_APPE_l1016_101612


namespace sum_of_squares_of_coefficients_l1016_101649

/-- The sum of the squares of the coefficients of the fully simplified expression 
    3(x^3 - 4x + 5) - 5(2x^3 - x^2 + 3x - 2) is equal to 1428 -/
theorem sum_of_squares_of_coefficients : ∃ (a b c d : ℤ),
  (∀ x : ℝ, 3 * (x^3 - 4*x + 5) - 5 * (2*x^3 - x^2 + 3*x - 2) = a*x^3 + b*x^2 + c*x + d) ∧
  a^2 + b^2 + c^2 + d^2 = 1428 := by
  sorry

end sum_of_squares_of_coefficients_l1016_101649


namespace inequality_range_l1016_101618

theorem inequality_range (a : ℝ) : 
  (∀ x > 1, x + 1 / (x - 1) ≥ a) → a ≤ 3 := by
  sorry

end inequality_range_l1016_101618


namespace intersection_empty_implies_m_range_l1016_101638

theorem intersection_empty_implies_m_range (m : ℝ) : 
  let A : Set ℝ := {x | x^2 - 3*x + 2 < 0}
  let B : Set ℝ := {x | x*(x-m) > 0}
  (A ∩ B = ∅) → m ≥ 2 :=
by
  sorry

end intersection_empty_implies_m_range_l1016_101638


namespace train_length_calculation_l1016_101688

theorem train_length_calculation (speed1 speed2 speed3 : ℝ) (time1 time2 time3 : ℝ) :
  let length1 := (speed1 / 3600) * time1
  let length2 := (speed2 / 3600) * time2
  let length3 := (speed3 / 3600) * time3
  speed1 = 300 ∧ speed2 = 250 ∧ speed3 = 350 ∧
  time1 = 33 ∧ time2 = 44 ∧ time3 = 28 →
  length1 + length2 + length3 = 8.52741 :=
by sorry

#check train_length_calculation

end train_length_calculation_l1016_101688


namespace tens_digit_of_2023_pow_2024_minus_2025_l1016_101674

theorem tens_digit_of_2023_pow_2024_minus_2025 :
  (2023^2024 - 2025) % 100 / 10 = 0 :=
by sorry

end tens_digit_of_2023_pow_2024_minus_2025_l1016_101674


namespace english_majors_count_l1016_101661

def choose (n k : ℕ) : ℕ := Nat.choose n k

theorem english_majors_count (bio_majors : ℕ) (engineers : ℕ) (total_selections : ℕ) :
  bio_majors = 6 →
  engineers = 5 →
  total_selections = 200 →
  ∃ (eng_majors : ℕ), 
    choose eng_majors 3 * choose bio_majors 3 * choose engineers 3 = total_selections ∧
    eng_majors = 3 :=
by sorry

end english_majors_count_l1016_101661


namespace carol_to_cathy_ratio_ratio_is_one_to_one_l1016_101644

-- Define the number of cars each person owns
def cathy_cars : ℕ := 5
def carol_cars : ℕ := cathy_cars
def lindsey_cars : ℕ := cathy_cars + 4
def susan_cars : ℕ := carol_cars - 2

-- Theorem to prove
theorem carol_to_cathy_ratio : 
  carol_cars = cathy_cars := by sorry

-- The ratio is 1:1 if the numbers are equal
theorem ratio_is_one_to_one : 
  carol_cars = cathy_cars → (carol_cars : ℚ) / cathy_cars = 1 := by sorry

end carol_to_cathy_ratio_ratio_is_one_to_one_l1016_101644


namespace interior_angle_sum_for_polygon_with_60_degree_exterior_angles_l1016_101622

theorem interior_angle_sum_for_polygon_with_60_degree_exterior_angles :
  ∀ (n : ℕ) (exterior_angle : ℝ),
    n > 2 →
    exterior_angle = 60 →
    n * exterior_angle = 360 →
    (n - 2) * 180 = 720 :=
by
  sorry

end interior_angle_sum_for_polygon_with_60_degree_exterior_angles_l1016_101622


namespace standard_deck_probability_l1016_101697

/-- Represents a standard deck of cards -/
structure Deck :=
  (total_cards : ℕ)
  (black_cards : ℕ)
  (red_cards : ℕ)
  (h_total : total_cards = black_cards + red_cards)

/-- The probability of drawing a black card, then a red card, then a black card -/
def draw_probability (d : Deck) : ℚ :=
  (d.black_cards : ℚ) * d.red_cards * (d.black_cards - 1) / 
  (d.total_cards * (d.total_cards - 1) * (d.total_cards - 2))

/-- Theorem stating the probability for a standard 52-card deck -/
theorem standard_deck_probability :
  let d : Deck := ⟨52, 26, 26, rfl⟩
  draw_probability d = 13 / 102 := by
  sorry

end standard_deck_probability_l1016_101697


namespace susie_rhode_island_reds_l1016_101648

/-- The number of Rhode Island Reds that Susie has -/
def susie_reds : ℕ := sorry

/-- The number of Golden Comets that Susie has -/
def susie_comets : ℕ := 6

/-- The number of Rhode Island Reds that Britney has -/
def britney_reds : ℕ := 2 * susie_reds

/-- The number of Golden Comets that Britney has -/
def britney_comets : ℕ := susie_comets / 2

/-- The total number of chickens Susie has -/
def susie_total : ℕ := susie_reds + susie_comets

/-- The total number of chickens Britney has -/
def britney_total : ℕ := britney_reds + britney_comets

theorem susie_rhode_island_reds :
  (britney_total = susie_total + 8) → susie_reds = 11 := by
  sorry

end susie_rhode_island_reds_l1016_101648


namespace intersection_of_A_and_B_l1016_101633

def A : Set ℝ := {x | x ≤ 0}
def B : Set ℝ := {-2, -1, 0, 1, 2}

theorem intersection_of_A_and_B : A ∩ B = {-2, -1, 0} := by sorry

end intersection_of_A_and_B_l1016_101633


namespace line_parameterization_l1016_101611

def is_valid_parameterization (x₀ y₀ u v : ℝ) : Prop :=
  y₀ = 3 * x₀ - 5 ∧ ∃ (k : ℝ), u = k * 1 ∧ v = k * 3

theorem line_parameterization 
  (x₀ y₀ u v : ℝ) : 
  is_valid_parameterization x₀ y₀ u v ↔ 
  (∀ t : ℝ, (3 * (x₀ + t * u) - 5 = y₀ + t * v)) :=
sorry

end line_parameterization_l1016_101611


namespace largest_increase_2007_2008_l1016_101666

def students : Fin 6 → ℕ
  | 0 => 50  -- 2003
  | 1 => 58  -- 2004
  | 2 => 65  -- 2005
  | 3 => 75  -- 2006
  | 4 => 80  -- 2007
  | 5 => 100 -- 2008

def percentageIncrease (a b : ℕ) : ℚ :=
  (b - a : ℚ) / a * 100

theorem largest_increase_2007_2008 :
  ∀ i : Fin 5, percentageIncrease (students 4) (students 5) ≥ percentageIncrease (students i) (students (i + 1)) :=
by sorry

end largest_increase_2007_2008_l1016_101666


namespace james_missed_two_questions_l1016_101681

/-- Represents the quiz bowl scoring system and James' performance -/
structure QuizBowl where
  points_per_correct : ℕ := 2
  bonus_points : ℕ := 4
  num_rounds : ℕ := 5
  questions_per_round : ℕ := 5
  james_points : ℕ := 66

/-- Calculates the number of questions James missed based on his score -/
def questions_missed (qb : QuizBowl) : ℕ :=
  let max_points := qb.num_rounds * (qb.questions_per_round * qb.points_per_correct + qb.bonus_points)
  (max_points - qb.james_points) / qb.points_per_correct

/-- Theorem stating that James missed exactly 2 questions -/
theorem james_missed_two_questions (qb : QuizBowl) : questions_missed qb = 2 := by
  sorry

end james_missed_two_questions_l1016_101681
