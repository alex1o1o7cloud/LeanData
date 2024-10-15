import Mathlib

namespace NUMINAMATH_CALUDE_geometric_series_sum_l3097_309711

/-- Given a sequence a_n and its partial sum S_n, prove that S_20 = 400 -/
theorem geometric_series_sum (a : ℕ → ℝ) (S : ℕ → ℝ) : 
  (∀ n, S n = (a n + 1)^2 / 4) →
  (∀ n, a n = 2 * n - 1) →
  S 20 = 400 := by
sorry

end NUMINAMATH_CALUDE_geometric_series_sum_l3097_309711


namespace NUMINAMATH_CALUDE_intersection_M_N_l3097_309764

def M : Set ℝ := {x | -4 < x ∧ x < 2}
def N : Set ℝ := {x | x^2 - x - 6 < 0}

theorem intersection_M_N : M ∩ N = {x : ℝ | -2 < x ∧ x < 2} := by sorry

end NUMINAMATH_CALUDE_intersection_M_N_l3097_309764


namespace NUMINAMATH_CALUDE_rectangle_area_l3097_309707

/-- The area of a rectangle with length 2 and width 4 is 8 -/
theorem rectangle_area : ∀ (length width area : ℝ), 
  length = 2 → width = 4 → area = length * width → area = 8 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_area_l3097_309707


namespace NUMINAMATH_CALUDE_raffle_tickets_sold_l3097_309706

theorem raffle_tickets_sold (ticket_price : ℚ) (total_donations : ℚ) (total_raised : ℚ) :
  ticket_price = 2 →
  total_donations = 50 →
  total_raised = 100 →
  (total_raised - total_donations) / ticket_price = 25 := by
  sorry

end NUMINAMATH_CALUDE_raffle_tickets_sold_l3097_309706


namespace NUMINAMATH_CALUDE_discriminant_of_quadratic_l3097_309731

/-- The discriminant of a quadratic polynomial ax^2 + bx + c is b^2 - 4ac -/
def discriminant (a b c : ℚ) : ℚ := b^2 - 4*a*c

/-- The quadratic polynomial 2x^2 + (4 - 1/2)x + 1 -/
def quadratic_polynomial (x : ℚ) : ℚ := 2*x^2 + (4 - 1/2)*x + 1

theorem discriminant_of_quadratic : 
  discriminant 2 (4 - 1/2) 1 = 17/4 := by
  sorry

end NUMINAMATH_CALUDE_discriminant_of_quadratic_l3097_309731


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l3097_309746

-- Define set A as the domain of y = lg x
def A : Set ℝ := {x | x > 0}

-- Define set B
def B : Set ℝ := {0, 1, 2}

-- Theorem statement
theorem intersection_of_A_and_B :
  A ∩ B = {1, 2} := by sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l3097_309746


namespace NUMINAMATH_CALUDE_rectangle_area_l3097_309773

/-- The area of a rectangle with length twice its width and perimeter equal to a triangle with sides 7, 10, and 11 is 392/9 -/
theorem rectangle_area (w : ℝ) (h : 2 * (2 * w + w) = 7 + 10 + 11) : w * (2 * w) = 392 / 9 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_area_l3097_309773


namespace NUMINAMATH_CALUDE_car_speed_problem_l3097_309700

theorem car_speed_problem (distance : ℝ) (original_time : ℝ) (new_time : ℝ) (new_speed : ℝ) :
  original_time = 12 →
  new_time = 4 →
  new_speed = 30 →
  distance = new_speed * new_time →
  distance = (distance / original_time) * original_time →
  distance / original_time = 10 := by
sorry

end NUMINAMATH_CALUDE_car_speed_problem_l3097_309700


namespace NUMINAMATH_CALUDE_ab_equals_zero_l3097_309793

theorem ab_equals_zero (a b : ℤ) (h : |a - b| + |a * b| = 2) : a * b = 0 := by
  sorry

end NUMINAMATH_CALUDE_ab_equals_zero_l3097_309793


namespace NUMINAMATH_CALUDE_min_value_expression_l3097_309789

theorem min_value_expression (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) (h : x * y * z = 3/2) :
  2 * x^2 + 4 * x * y + 9 * y^2 + 10 * y * z + 3 * z^2 ≥ 27 / 2^(4/9) * Real.rpow 90 (1/9) :=
by sorry

end NUMINAMATH_CALUDE_min_value_expression_l3097_309789


namespace NUMINAMATH_CALUDE_cos_seventh_power_decomposition_l3097_309777

theorem cos_seventh_power_decomposition :
  ∃ (b₁ b₂ b₃ b₄ b₅ b₆ b₇ : ℝ),
    (∀ θ : ℝ, (Real.cos θ)^7 = b₁ * Real.cos θ + b₂ * Real.cos (2*θ) + b₃ * Real.cos (3*θ) + 
                               b₄ * Real.cos (4*θ) + b₅ * Real.cos (5*θ) + b₆ * Real.cos (6*θ) + 
                               b₇ * Real.cos (7*θ)) ∧
    (b₁ = 35/64 ∧ b₂ = 0 ∧ b₃ = 21/64 ∧ b₄ = 0 ∧ b₅ = 7/64 ∧ b₆ = 0 ∧ b₇ = 1/64) ∧
    (b₁^2 + b₂^2 + b₃^2 + b₄^2 + b₅^2 + b₆^2 + b₇^2 = 1785/4096) := by
  sorry

end NUMINAMATH_CALUDE_cos_seventh_power_decomposition_l3097_309777


namespace NUMINAMATH_CALUDE_expression_increase_l3097_309703

theorem expression_increase (x y : ℝ) : 
  let original := 3 * x^2 * y
  let new_x := 1.2 * x
  let new_y := 2.4 * y
  let new_expression := 3 * new_x^2 * new_y
  new_expression = 3.456 * original := by
sorry

end NUMINAMATH_CALUDE_expression_increase_l3097_309703


namespace NUMINAMATH_CALUDE_floor_neg_sqrt_64_over_9_l3097_309719

theorem floor_neg_sqrt_64_over_9 : ⌊-Real.sqrt (64 / 9)⌋ = -3 := by
  sorry

end NUMINAMATH_CALUDE_floor_neg_sqrt_64_over_9_l3097_309719


namespace NUMINAMATH_CALUDE_average_of_xyz_l3097_309709

theorem average_of_xyz (x y z : ℝ) : 
  x = 3 → y = 2 * x → z = 3 * y → (x + y + z) / 3 = 9 := by
  sorry

end NUMINAMATH_CALUDE_average_of_xyz_l3097_309709


namespace NUMINAMATH_CALUDE_steven_needs_three_more_seeds_l3097_309727

/-- The number of seeds needed for the assignment -/
def assignment_seeds : ℕ := 60

/-- The average number of seeds in an apple -/
def apple_seeds : ℕ := 6

/-- The average number of seeds in a pear -/
def pear_seeds : ℕ := 2

/-- The average number of seeds in a grape -/
def grape_seeds : ℕ := 3

/-- The number of apples Steven has -/
def steven_apples : ℕ := 4

/-- The number of pears Steven has -/
def steven_pears : ℕ := 3

/-- The number of grapes Steven has -/
def steven_grapes : ℕ := 9

/-- The number of additional seeds Steven needs -/
def additional_seeds_needed : ℕ := 3

theorem steven_needs_three_more_seeds :
  assignment_seeds - (steven_apples * apple_seeds + steven_pears * pear_seeds + steven_grapes * grape_seeds) = additional_seeds_needed := by
  sorry

end NUMINAMATH_CALUDE_steven_needs_three_more_seeds_l3097_309727


namespace NUMINAMATH_CALUDE_jacks_allowance_l3097_309730

/-- Calculates Jack's weekly allowance given the initial amount, number of weeks, and final amount in his piggy bank -/
def calculate_allowance (initial_amount : ℚ) (weeks : ℕ) (final_amount : ℚ) : ℚ :=
  2 * (final_amount - initial_amount) / weeks

/-- Proves that Jack's weekly allowance is $10 given the problem conditions -/
theorem jacks_allowance :
  let initial_amount : ℚ := 43
  let weeks : ℕ := 8
  let final_amount : ℚ := 83
  calculate_allowance initial_amount weeks final_amount = 10 := by
  sorry

#eval calculate_allowance 43 8 83

end NUMINAMATH_CALUDE_jacks_allowance_l3097_309730


namespace NUMINAMATH_CALUDE_a_in_range_l3097_309748

/-- A function f(x) = ax^2 + (a-3)x + 1 that is decreasing on [-1, +∞) -/
def f (a : ℝ) (x : ℝ) : ℝ := a * x^2 + (a - 3) * x + 1

/-- The property that f is decreasing on [-1, +∞) -/
def is_decreasing_on_interval (a : ℝ) : Prop :=
  ∀ x y, -1 ≤ x ∧ x < y → f a y < f a x

/-- The theorem stating that if f is decreasing on [-1, +∞), then a is in [-3, 0) -/
theorem a_in_range (a : ℝ) : is_decreasing_on_interval a → a ∈ Set.Icc (-3) 0 :=
sorry

end NUMINAMATH_CALUDE_a_in_range_l3097_309748


namespace NUMINAMATH_CALUDE_triangle_angle_calculation_l3097_309737

theorem triangle_angle_calculation (A B C : ℝ) : 
  A + B + C = 180 →  -- Sum of angles in a triangle is 180°
  C = 2 * B →        -- Angle C is double angle B
  A = 3 * B →        -- Angle A is thrice angle B
  B = 30 :=          -- Angle B is 30°
by
  sorry

end NUMINAMATH_CALUDE_triangle_angle_calculation_l3097_309737


namespace NUMINAMATH_CALUDE_selling_to_buying_price_ratio_l3097_309701

theorem selling_to_buying_price_ratio 
  (natasha_money : ℕ) 
  (natasha_carla_ratio : ℕ) 
  (carla_cosima_ratio : ℕ) 
  (profit : ℕ) 
  (h1 : natasha_money = 60)
  (h2 : natasha_carla_ratio = 3)
  (h3 : carla_cosima_ratio = 2)
  (h4 : profit = 36) :
  let carla_money := natasha_money / natasha_carla_ratio
  let cosima_money := carla_money / carla_cosima_ratio
  let total_money := natasha_money + carla_money + cosima_money
  let selling_price := total_money + profit
  ∃ (a b : ℕ), a = 7 ∧ b = 5 ∧ selling_price * b = total_money * a :=
by sorry

end NUMINAMATH_CALUDE_selling_to_buying_price_ratio_l3097_309701


namespace NUMINAMATH_CALUDE_shadow_problem_l3097_309779

/-- Given a cube with edge length 2 cm and a point light source x cm above an upper vertex,
    if the shadow area (excluding the area beneath the cube) is 192 sq cm,
    then the greatest integer not exceeding 1000x is 12000. -/
theorem shadow_problem (x : ℝ) : 
  let cube_edge : ℝ := 2
  let shadow_area : ℝ := 192
  let total_shadow_area : ℝ := shadow_area + cube_edge^2
  let shadow_side : ℝ := Real.sqrt total_shadow_area
  x = cube_edge * (shadow_side - cube_edge) / cube_edge →
  Int.floor (1000 * x) = 12000 := by
sorry

end NUMINAMATH_CALUDE_shadow_problem_l3097_309779


namespace NUMINAMATH_CALUDE_tetrahedron_vector_equality_l3097_309765

-- Define the tetrahedron O-ABC
variable (O A B C : EuclideanSpace ℝ (Fin 3))

-- Define vectors a, b, c
variable (a b c : EuclideanSpace ℝ (Fin 3))

-- Define points M and N
variable (M N : EuclideanSpace ℝ (Fin 3))

-- State the theorem
theorem tetrahedron_vector_equality 
  (h1 : A - O = a) 
  (h2 : B - O = b) 
  (h3 : C - O = c) 
  (h4 : M - O = (2/3) • (A - O)) 
  (h5 : N - O = (1/2) • (B - O) + (1/2) • (C - O)) :
  M - N = (1/2) • b + (1/2) • c - (2/3) • a := by sorry

end NUMINAMATH_CALUDE_tetrahedron_vector_equality_l3097_309765


namespace NUMINAMATH_CALUDE_closest_to_product_l3097_309717

def options : List ℝ := [7, 42, 74, 84, 737]

def product : ℝ := 1.8 * (40.3 + 0.07)

theorem closest_to_product : 
  ∃ (x : ℝ), x ∈ options ∧ 
  ∀ (y : ℝ), y ∈ options → |x - product| ≤ |y - product| ∧ 
  x = 74 :=
by sorry

end NUMINAMATH_CALUDE_closest_to_product_l3097_309717


namespace NUMINAMATH_CALUDE_pencil_count_l3097_309771

theorem pencil_count (initial : ℕ) (nancy_added : ℕ) (steven_added : ℕ)
  (h1 : initial = 138)
  (h2 : nancy_added = 256)
  (h3 : steven_added = 97) :
  initial + nancy_added + steven_added = 491 :=
by sorry

end NUMINAMATH_CALUDE_pencil_count_l3097_309771


namespace NUMINAMATH_CALUDE_jessica_seashells_l3097_309762

theorem jessica_seashells (joan_shells jessica_shells total_shells : ℕ) 
  (h1 : joan_shells = 6)
  (h2 : total_shells = 14)
  (h3 : joan_shells + jessica_shells = total_shells) :
  jessica_shells = 8 := by
  sorry

end NUMINAMATH_CALUDE_jessica_seashells_l3097_309762


namespace NUMINAMATH_CALUDE_student_multiplication_problem_l3097_309757

theorem student_multiplication_problem (x : ℚ) : 
  (63 * x) - 142 = 110 → x = 4 := by
  sorry

end NUMINAMATH_CALUDE_student_multiplication_problem_l3097_309757


namespace NUMINAMATH_CALUDE_part_one_part_two_l3097_309713

-- Define the functions f and g
def f (a : ℝ) (x : ℝ) : ℝ := |2 * x - a| + a
def g (x : ℝ) : ℝ := |2 * x - 1|

-- Part I
theorem part_one : 
  {x : ℝ | f 2 x ≤ 6} = {x : ℝ | -1 ≤ x ∧ x ≤ 3} := by sorry

-- Part II
theorem part_two :
  (∀ x : ℝ, f a x + g x ≥ 3) → a ≥ 2 := by sorry

end NUMINAMATH_CALUDE_part_one_part_two_l3097_309713


namespace NUMINAMATH_CALUDE_vector_relation_l3097_309754

-- Define the space
variable {V : Type*} [NormedAddCommGroup V] [InnerProductSpace ℝ V]

-- Define points A, B, and C
variable (A B C : V)

-- Define the theorem
theorem vector_relation (h1 : ∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ C = (1 - t) • A + t • B) 
                        (h2 : C - A = (3/5) • (B - A)) : 
  C - A = -(3/2) • (C - B) := by
  sorry

end NUMINAMATH_CALUDE_vector_relation_l3097_309754


namespace NUMINAMATH_CALUDE_partnership_profit_l3097_309778

/-- Given the investment ratio and B's share of profit, calculate the total profit --/
theorem partnership_profit (a b c : ℕ) (b_share : ℕ) (h1 : a = 6) (h2 : b = 2) (h3 : c = 3) (h4 : b_share = 800) :
  (b_share / b) * (a + b + c) = 4400 := by
  sorry

end NUMINAMATH_CALUDE_partnership_profit_l3097_309778


namespace NUMINAMATH_CALUDE_only_set_A_is_right_triangle_l3097_309726

-- Define a function to check if three numbers form a right triangle
def is_right_triangle (a b c : ℕ) : Prop :=
  a * a + b * b = c * c ∨ a * a + c * c = b * b ∨ b * b + c * c = a * a

-- Define the sets of numbers
def set_A : List ℕ := [5, 12, 13]
def set_B : List ℕ := [3, 4, 6]
def set_C : List ℕ := [4, 5, 6]
def set_D : List ℕ := [5, 7, 9]

-- Theorem to prove
theorem only_set_A_is_right_triangle :
  (is_right_triangle 5 12 13) ∧
  ¬(is_right_triangle 3 4 6) ∧
  ¬(is_right_triangle 4 5 6) ∧
  ¬(is_right_triangle 5 7 9) :=
sorry

end NUMINAMATH_CALUDE_only_set_A_is_right_triangle_l3097_309726


namespace NUMINAMATH_CALUDE_sin_cos_sum_equals_sqrt3_over_2_l3097_309743

theorem sin_cos_sum_equals_sqrt3_over_2 :
  Real.sin (20 * π / 180) * Real.sin (50 * π / 180) + 
  Real.cos (20 * π / 180) * Real.sin (40 * π / 180) = 
  Real.sqrt 3 / 2 := by sorry

end NUMINAMATH_CALUDE_sin_cos_sum_equals_sqrt3_over_2_l3097_309743


namespace NUMINAMATH_CALUDE_division_problem_l3097_309785

theorem division_problem : (120 : ℝ) / (5 / 2.5) = 60 := by sorry

end NUMINAMATH_CALUDE_division_problem_l3097_309785


namespace NUMINAMATH_CALUDE_nell_initial_cards_l3097_309738

/-- The number of baseball cards Nell had initially -/
def initial_cards : ℕ := sorry

/-- The number of cards Jeff gave to Nell -/
def cards_from_jeff : ℝ := 276.0

/-- The total number of cards Nell has now -/
def total_cards : ℕ := 580

/-- Theorem stating that Nell's initial number of cards was 304 -/
theorem nell_initial_cards : 
  initial_cards = 304 :=
by
  sorry

end NUMINAMATH_CALUDE_nell_initial_cards_l3097_309738


namespace NUMINAMATH_CALUDE_function_bound_l3097_309784

theorem function_bound (x : ℝ) : 
  1/2 ≤ (x^2 + x + 1) / (x^2 + 1) ∧ (x^2 + x + 1) / (x^2 + 1) ≤ 3/2 := by
  sorry

end NUMINAMATH_CALUDE_function_bound_l3097_309784


namespace NUMINAMATH_CALUDE_solution_set_inequality_l3097_309753

theorem solution_set_inequality (x : ℝ) :
  (1 / (x - 1) ≥ -1) ↔ (x ≤ 0 ∨ x > 1) := by sorry

end NUMINAMATH_CALUDE_solution_set_inequality_l3097_309753


namespace NUMINAMATH_CALUDE_sum_of_extrema_equals_two_l3097_309782

-- Define the function f(x) = x ln |x| + 1
noncomputable def f (x : ℝ) : ℝ := x * Real.log (abs x) + 1

-- Theorem statement
theorem sum_of_extrema_equals_two :
  ∃ (max_val min_val : ℝ),
    (∀ x, f x ≤ max_val) ∧
    (∃ x, f x = max_val) ∧
    (∀ x, f x ≥ min_val) ∧
    (∃ x, f x = min_val) ∧
    max_val + min_val = 2 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_extrema_equals_two_l3097_309782


namespace NUMINAMATH_CALUDE_exponentiation_distributive_multiplication_multiplication_not_distributive_exponentiation_l3097_309787

theorem exponentiation_distributive_multiplication (a b c : ℝ) :
  (a * b) ^ c = a ^ c * b ^ c :=
sorry

theorem multiplication_not_distributive_exponentiation :
  ∃ a b c : ℝ, (a ^ b) * c ≠ (a * c) ^ (b * c) :=
sorry

end NUMINAMATH_CALUDE_exponentiation_distributive_multiplication_multiplication_not_distributive_exponentiation_l3097_309787


namespace NUMINAMATH_CALUDE_good_bulbs_count_l3097_309780

def total_bulbs : ℕ := 10
def num_lamps : ℕ := 3
def prob_lighted : ℚ := 29/30

def num_good_bulbs : ℕ := 6

theorem good_bulbs_count :
  (1 : ℚ) - (Nat.choose (total_bulbs - num_good_bulbs) num_lamps : ℚ) / (Nat.choose total_bulbs num_lamps) = prob_lighted :=
sorry

end NUMINAMATH_CALUDE_good_bulbs_count_l3097_309780


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l3097_309794

theorem arithmetic_sequence_sum (a : ℕ → ℝ) (h_arith : ∀ n, a (n + 1) - a n = a (n + 2) - a (n + 1)) 
  (h_sum1 : a 1 + a 2 + a 3 = -24) (h_sum2 : a 18 + a 19 + a 20 = 78) : 
  a 1 + a 20 = 18 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l3097_309794


namespace NUMINAMATH_CALUDE_warriors_height_order_l3097_309797

theorem warriors_height_order (heights : Set ℝ) (h : Set.Infinite heights) :
  ∃ (subseq : ℕ → ℝ), (∀ n, subseq n ∈ heights) ∧ 
    (Set.Infinite (Set.range subseq)) ∧ 
    (∀ n m, n < m → subseq n < subseq m) :=
sorry

end NUMINAMATH_CALUDE_warriors_height_order_l3097_309797


namespace NUMINAMATH_CALUDE_complex_absolute_value_l3097_309798

/-- Given that ω = 10 + 3i, prove that |ω² + 10ω + 104| = 212 -/
theorem complex_absolute_value (ω : ℂ) (h : ω = 10 + 3*I) :
  Complex.abs (ω^2 + 10*ω + 104) = 212 := by
  sorry

end NUMINAMATH_CALUDE_complex_absolute_value_l3097_309798


namespace NUMINAMATH_CALUDE_fewer_heads_probability_l3097_309715

/-- The probability of getting fewer heads than tails when flipping 12 fair coins -/
def fewer_heads_prob : ℚ := 1586 / 4096

/-- The number of coins being flipped -/
def num_coins : ℕ := 12

theorem fewer_heads_probability :
  fewer_heads_prob = (2^num_coins - (num_coins.choose (num_coins / 2))) / (2 * 2^num_coins) :=
sorry

end NUMINAMATH_CALUDE_fewer_heads_probability_l3097_309715


namespace NUMINAMATH_CALUDE_polynomial_simplification_l3097_309755

theorem polynomial_simplification (x : ℝ) :
  (3 * x^6 + 2 * x^5 - x^4 + 3 * x^2 + 15) - (x^6 + 4 * x^5 + 3 * x^3 - 2 * x^2 + 20) =
  2 * x^6 - 2 * x^5 - x^4 + 5 * x^2 - 5 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_simplification_l3097_309755


namespace NUMINAMATH_CALUDE_geometric_sequence_304th_term_l3097_309761

/-- Given a geometric sequence with first term 8 and second term -8, the 304th term is -8 -/
theorem geometric_sequence_304th_term :
  ∀ (a : ℕ → ℝ), 
    a 1 = 8 →  -- First term is 8
    a 2 = -8 →  -- Second term is -8
    (∀ n : ℕ, a (n + 1) = a n * (a 2 / a 1)) →  -- Geometric sequence property
    a 304 = -8 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_304th_term_l3097_309761


namespace NUMINAMATH_CALUDE_margie_change_l3097_309749

theorem margie_change : 
  let banana_cost : ℚ := 0.30
  let orange_cost : ℚ := 0.40
  let banana_count : ℕ := 5
  let orange_count : ℕ := 3
  let paid_amount : ℚ := 10.00
  let total_cost : ℚ := banana_cost * banana_count + orange_cost * orange_count
  let change : ℚ := paid_amount - total_cost
  change = 7.30 := by sorry

end NUMINAMATH_CALUDE_margie_change_l3097_309749


namespace NUMINAMATH_CALUDE_horner_method_v4_l3097_309759

def f (x : ℝ) : ℝ := x^7 - 2*x^6 + 3*x^3 - 4*x^2 + 1

def horner_v4 (x : ℝ) : ℝ := (((x - 2) * x + 0) * x + 0) * x + 3

theorem horner_method_v4 :
  horner_v4 2 = 3 :=
by sorry

end NUMINAMATH_CALUDE_horner_method_v4_l3097_309759


namespace NUMINAMATH_CALUDE_min_employees_needed_l3097_309766

/-- The minimum number of employees needed for pollution monitoring -/
theorem min_employees_needed (water air soil water_air air_soil water_soil all_three : ℕ)
  (h1 : water = 120)
  (h2 : air = 150)
  (h3 : soil = 100)
  (h4 : water_air = 50)
  (h5 : air_soil = 30)
  (h6 : water_soil = 20)
  (h7 : all_three = 10) :
  water + air + soil - water_air - air_soil - water_soil + all_three = 280 := by
  sorry

end NUMINAMATH_CALUDE_min_employees_needed_l3097_309766


namespace NUMINAMATH_CALUDE_weight_of_b_l3097_309786

def weight_problem (a b c : ℝ) : Prop :=
  (a + b + c) / 3 = 45 ∧
  (a + b) / 2 = 40 ∧
  (b + c) / 2 = 43

theorem weight_of_b (a b c : ℝ) (h : weight_problem a b c) : b = 31 := by
  sorry

end NUMINAMATH_CALUDE_weight_of_b_l3097_309786


namespace NUMINAMATH_CALUDE_grid_arithmetic_sequence_l3097_309750

theorem grid_arithmetic_sequence (row : Fin 7 → ℚ) (col1 col2 : Fin 5 → ℚ) :
  -- The row forms an arithmetic sequence
  (∀ i : Fin 6, row (i + 1) - row i = row 1 - row 0) →
  -- The first column forms an arithmetic sequence
  (∀ i : Fin 4, col1 (i + 1) - col1 i = col1 1 - col1 0) →
  -- The second column forms an arithmetic sequence
  (∀ i : Fin 4, col2 (i + 1) - col2 i = col2 1 - col2 0) →
  -- Given values
  row 0 = 25 →
  col1 2 = 16 →
  col1 3 = 20 →
  col2 4 = -21 →
  -- The fourth element in the row is the same as the first element in the first column
  row 3 = col1 0 →
  -- The last element in the row is the same as the first element in the second column
  row 6 = col2 0 →
  -- M is the first element in the second column
  col2 0 = 1021 / 12 := by
sorry

end NUMINAMATH_CALUDE_grid_arithmetic_sequence_l3097_309750


namespace NUMINAMATH_CALUDE_complex_sum_of_parts_l3097_309720

theorem complex_sum_of_parts (a b : ℝ) : (Complex.I * (1 - Complex.I) = Complex.mk a b) → a + b = 2 := by
  sorry

end NUMINAMATH_CALUDE_complex_sum_of_parts_l3097_309720


namespace NUMINAMATH_CALUDE_cubic_equation_solution_l3097_309790

theorem cubic_equation_solution (p q : ℝ) : 
  (3 * p^2 - 5 * p - 8 = 0) → 
  (3 * q^2 - 5 * q - 8 = 0) → 
  p ≠ q →
  (5 * p^3 - 5 * q^3) / (p - q) = 245 / 9 := by
  sorry

end NUMINAMATH_CALUDE_cubic_equation_solution_l3097_309790


namespace NUMINAMATH_CALUDE_min_value_fraction_l3097_309763

theorem min_value_fraction (x y z w : ℝ) 
  (pos_x : 0 < x) (pos_y : 0 < y) (pos_z : 0 < z) (pos_w : 0 < w)
  (sum_one : x + y + z + w = 1) :
  (x + y) / (x * y * z * w) ≥ 108 ∧ 
  ∃ x y z w, 0 < x ∧ 0 < y ∧ 0 < z ∧ 0 < w ∧ 
    x + y + z + w = 1 ∧ (x + y) / (x * y * z * w) = 108 := by
  sorry

end NUMINAMATH_CALUDE_min_value_fraction_l3097_309763


namespace NUMINAMATH_CALUDE_triangle_angle_proof_l3097_309704

theorem triangle_angle_proof (α β γ : ℝ) : 
  α + β + γ = 180 →  -- Sum of angles in a triangle is 180°
  α = 100 →          -- One angle is 100°
  β = 2 * γ →        -- One angle is twice the other
  γ = 26 :=          -- The smallest angle is 26°
by sorry

end NUMINAMATH_CALUDE_triangle_angle_proof_l3097_309704


namespace NUMINAMATH_CALUDE_new_rectangle_area_l3097_309747

theorem new_rectangle_area (a b : ℝ) (h : a > b) :
  let base := a^2 + b^2 + a
  let height := a^2 + b^2 - b
  base * height = a^4 + a^3 + 2*a^2*b^2 + a*b^3 - a*b + b^4 - b^3 - b^2 :=
by sorry

end NUMINAMATH_CALUDE_new_rectangle_area_l3097_309747


namespace NUMINAMATH_CALUDE_rain_on_tuesday_l3097_309734

theorem rain_on_tuesday (rain_monday : ℝ) (rain_both : ℝ) (no_rain : ℝ)
  (h1 : rain_monday = 0.7)
  (h2 : rain_both = 0.4)
  (h3 : no_rain = 0.2) :
  ∃ rain_tuesday : ℝ,
    rain_tuesday = 0.5 ∧
    rain_monday + rain_tuesday - rain_both = 1 - no_rain :=
by
  sorry

end NUMINAMATH_CALUDE_rain_on_tuesday_l3097_309734


namespace NUMINAMATH_CALUDE_library_problem_l3097_309714

theorem library_problem (total_books : ℕ) (books_per_student : ℕ) 
  (students_day1 : ℕ) (students_day2 : ℕ) (students_day3 : ℕ) : ℕ :=
  by
  have h1 : total_books = 120 := by sorry
  have h2 : books_per_student = 5 := by sorry
  have h3 : students_day1 = 4 := by sorry
  have h4 : students_day2 = 5 := by sorry
  have h5 : students_day3 = 6 := by sorry
  
  have remaining_books : ℕ := total_books - (students_day1 + students_day2 + students_day3) * books_per_student
  
  exact remaining_books / books_per_student

end NUMINAMATH_CALUDE_library_problem_l3097_309714


namespace NUMINAMATH_CALUDE_jar_weight_percentage_l3097_309728

theorem jar_weight_percentage (jar_weight : ℝ) (full_beans_weight : ℝ) 
  (h1 : jar_weight = 0.2 * (jar_weight + full_beans_weight))
  (h2 : 0.5 * full_beans_weight = full_beans_weight / 2) :
  (jar_weight + full_beans_weight / 2) / (jar_weight + full_beans_weight) = 0.6 := by
  sorry

end NUMINAMATH_CALUDE_jar_weight_percentage_l3097_309728


namespace NUMINAMATH_CALUDE_work_completion_workers_work_completion_workers_proof_l3097_309721

/-- Given a work that can be finished in 12 days by an initial group of workers,
    and is finished in 9 days after 10 more workers join,
    prove that the total number of workers after the addition is 40. -/
theorem work_completion_workers : ℕ → Prop :=
  λ initial_workers =>
    (initial_workers * 12 = (initial_workers + 10) * 9) →
    initial_workers + 10 = 40
  
#check work_completion_workers

/-- Proof of the theorem -/
theorem work_completion_workers_proof : ∃ n : ℕ, work_completion_workers n := by
  sorry

end NUMINAMATH_CALUDE_work_completion_workers_work_completion_workers_proof_l3097_309721


namespace NUMINAMATH_CALUDE_four_digit_sum_l3097_309739

/-- Given four distinct non-zero digits, the sum of all four-digit numbers formed using these digits without repetition is 73,326 if and only if the digits are 1, 2, 3, and 5. -/
theorem four_digit_sum (a b c d : ℕ) : 
  (a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0 ∧ d ≠ 0) →  -- non-zero digits
  (a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d) →  -- distinct digits
  (6 * (a + b + c + d) * 1111 = 73326) →  -- sum condition
  ({a, b, c, d} : Set ℕ) = {1, 2, 3, 5} :=
sorry

end NUMINAMATH_CALUDE_four_digit_sum_l3097_309739


namespace NUMINAMATH_CALUDE_trig_fraction_value_l3097_309722

theorem trig_fraction_value (α : Real) (h : Real.tan α = 2) :
  1 / (2 * Real.sin α * Real.cos α + Real.cos α ^ 2) = 1 := by
  sorry

end NUMINAMATH_CALUDE_trig_fraction_value_l3097_309722


namespace NUMINAMATH_CALUDE_min_colors_tessellation_l3097_309795

/-- Represents a tile in the tessellation -/
inductive Tile
| Triangle
| Trapezoid

/-- Represents a color used in the tessellation -/
inductive Color
| Red
| Green
| Blue

/-- Represents the tessellation as a function from coordinates to tiles -/
def Tessellation := ℕ → ℕ → Tile

/-- A valid tessellation alternates between rows of triangles and trapezoids -/
def isValidTessellation (t : Tessellation) : Prop :=
  ∀ i j, t i j = if i % 2 = 0 then Tile.Triangle else Tile.Trapezoid

/-- A coloring of the tessellation -/
def Coloring := ℕ → ℕ → Color

/-- Checks if two tiles are adjacent -/
def isAdjacent (i1 j1 i2 j2 : ℕ) : Prop :=
  (i1 = i2 ∧ j1 + 1 = j2) ∨ 
  (i1 + 1 = i2 ∧ j1 = j2) ∨ 
  (i1 + 1 = i2 ∧ j1 + 1 = j2)

/-- A valid coloring ensures no adjacent tiles have the same color -/
def isValidColoring (t : Tessellation) (c : Coloring) : Prop :=
  ∀ i1 j1 i2 j2, isAdjacent i1 j1 i2 j2 → c i1 j1 ≠ c i2 j2

/-- The main theorem: 3 colors are sufficient and necessary -/
theorem min_colors_tessellation (t : Tessellation) (h : isValidTessellation t) :
  (∃ c : Coloring, isValidColoring t c) ∧ 
  (∀ c : Coloring, isValidColoring t c → ∃ (x y z : Color), 
    (∀ i j, c i j = x ∨ c i j = y ∨ c i j = z)) :=
sorry

end NUMINAMATH_CALUDE_min_colors_tessellation_l3097_309795


namespace NUMINAMATH_CALUDE_tangent_segment_equality_tangent_line_distance_equality_l3097_309740

-- Define a circle
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

-- Define a line
structure Line where
  slope : ℝ
  intercept : ℝ

-- Define a point
def Point := ℝ × ℝ

-- Define a tangent line to a circle
def is_tangent (l : Line) (c : Circle) : Prop :=
  sorry

-- Define the distance between two points
def distance (p1 p2 : Point) : ℝ :=
  sorry

-- Define the distance between two lines
def line_distance (l1 l2 : Line) : ℝ :=
  sorry

-- Define the point of tangency
def point_of_tangency (l : Line) (c : Circle) : Point :=
  sorry

theorem tangent_segment_equality (c1 c2 : Circle) (l1 l2 l3 l4 : Line) 
  (h1 : is_tangent l1 c1) (h2 : is_tangent l2 c1)
  (h3 : is_tangent l3 c2) (h4 : is_tangent l4 c2) :
  let p1 := point_of_tangency l1 c1
  let p2 := point_of_tangency l2 c1
  let p3 := point_of_tangency l3 c2
  let p4 := point_of_tangency l4 c2
  distance p1 p3 = distance p2 p4 :=
sorry

theorem tangent_line_distance_equality (c1 c2 : Circle) (l1 l2 l3 l4 : Line)
  (h1 : is_tangent l1 c1) (h2 : is_tangent l2 c1)
  (h3 : is_tangent l3 c2) (h4 : is_tangent l4 c2) :
  line_distance l1 l3 = line_distance l2 l4 :=
sorry

end NUMINAMATH_CALUDE_tangent_segment_equality_tangent_line_distance_equality_l3097_309740


namespace NUMINAMATH_CALUDE_no_integer_solutions_l3097_309745

theorem no_integer_solutions :
  ¬ ∃ (x y z : ℤ), x^4 + y^4 + z^4 = 2*x^2*y^2 + 2*y^2*z^2 + 2*z^2*x^2 + 24 :=
by sorry

end NUMINAMATH_CALUDE_no_integer_solutions_l3097_309745


namespace NUMINAMATH_CALUDE_digital_earth_sharing_l3097_309735

/-- Represents the concept of Digital Earth -/
structure DigitalEarth where
  technology : Type
  data : Type
  sharing_method : Type

/-- Represents the internet as a sharing method -/
def Internet : Type := Unit

/-- Axiom: Digital Earth involves digital technology and Earth-related data -/
axiom digital_earth_components : ∀ (de : DigitalEarth), de.technology × de.data

/-- Theorem: Digital Earth can only achieve global information sharing through the internet -/
theorem digital_earth_sharing (de : DigitalEarth) : 
  de.sharing_method = Internet :=
sorry

end NUMINAMATH_CALUDE_digital_earth_sharing_l3097_309735


namespace NUMINAMATH_CALUDE_max_npk_l3097_309742

/-- Represents a single digit integer -/
def SingleDigit := {n : ℕ | 1 ≤ n ∧ n ≤ 9}

/-- Converts a single digit to a two-digit number with repeated digits -/
def toTwoDigit (m : SingleDigit) : ℕ := 11 * m

/-- Checks if a number is three digits -/
def isThreeDigits (n : ℕ) : Prop := 100 ≤ n ∧ n < 1000

/-- The result of multiplying a two-digit number by a single digit -/
def result (m k : SingleDigit) : ℕ := toTwoDigit m * k

theorem max_npk :
  ∀ m k : SingleDigit,
    m ≠ k →
    isThreeDigits (result m k) →
    ∀ m' k' : SingleDigit,
      m' ≠ k' →
      isThreeDigits (result m' k') →
      result m' k' ≤ 891 :=
sorry

end NUMINAMATH_CALUDE_max_npk_l3097_309742


namespace NUMINAMATH_CALUDE_angle_between_vectors_is_acute_l3097_309770

theorem angle_between_vectors_is_acute (A B C : ℝ) (p q : ℝ × ℝ) :
  0 < A ∧ A < π/2 →
  0 < B ∧ B < π/2 →
  0 < C ∧ C < π/2 →
  A + B + C = π →
  p = (Real.cos A, Real.sin A) →
  q = (-Real.cos B, Real.sin B) →
  ∃ α, 0 < α ∧ α < π/2 ∧ Real.cos α = p.1 * q.1 + p.2 * q.2 := by
  sorry

end NUMINAMATH_CALUDE_angle_between_vectors_is_acute_l3097_309770


namespace NUMINAMATH_CALUDE_female_height_calculation_l3097_309752

theorem female_height_calculation (total_avg : ℝ) (male_avg : ℝ) (ratio : ℝ) 
  (h1 : total_avg = 180)
  (h2 : male_avg = 185)
  (h3 : ratio = 2) :
  ∃ female_avg : ℝ, female_avg = 170 ∧ 
  (ratio * female_avg + male_avg) / (ratio + 1) = total_avg :=
by sorry

end NUMINAMATH_CALUDE_female_height_calculation_l3097_309752


namespace NUMINAMATH_CALUDE_fleas_perished_count_l3097_309725

/-- Represents the count of fleas in an ear -/
structure FleaCount where
  adultA : ℕ
  adultB : ℕ
  nymphA : ℕ
  nymphB : ℕ

/-- Represents the survival rates for different flea types -/
structure SurvivalRates where
  adultA : ℚ
  adultB : ℚ
  nymphA : ℚ
  nymphB : ℚ

def rightEar : FleaCount := {
  adultA := 42,
  adultB := 80,
  nymphA := 37,
  nymphB := 67
}

def leftEar : FleaCount := {
  adultA := 29,
  adultB := 64,
  nymphA := 71,
  nymphB := 45
}

def survivalRates : SurvivalRates := {
  adultA := 3/4,
  adultB := 3/5,
  nymphA := 2/5,
  nymphB := 11/20
}

/-- Calculates the number of fleas that perished in an ear -/
def fleaPerished (ear : FleaCount) (rates : SurvivalRates) : ℚ :=
  ear.adultA * (1 - rates.adultA) +
  ear.adultB * (1 - rates.adultB) +
  ear.nymphA * (1 - rates.nymphA) +
  ear.nymphB * (1 - rates.nymphB)

theorem fleas_perished_count :
  ⌊fleaPerished rightEar survivalRates + fleaPerished leftEar survivalRates⌋ = 190 := by
  sorry

end NUMINAMATH_CALUDE_fleas_perished_count_l3097_309725


namespace NUMINAMATH_CALUDE_sqrt_sum_equals_ten_l3097_309702

theorem sqrt_sum_equals_ten : 
  Real.sqrt ((5 - 3 * Real.sqrt 2) ^ 2) + Real.sqrt ((5 + 3 * Real.sqrt 2) ^ 2) = 10 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_sum_equals_ten_l3097_309702


namespace NUMINAMATH_CALUDE_rice_weight_in_pounds_l3097_309708

/-- Given rice divided equally into 4 containers, with each container having 33 ounces,
    and 1 pound being equal to 16 ounces, the total weight of rice in pounds is 8.25. -/
theorem rice_weight_in_pounds :
  let num_containers : ℕ := 4
  let ounces_per_container : ℚ := 33
  let ounces_per_pound : ℚ := 16
  let total_ounces : ℚ := num_containers * ounces_per_container
  let total_pounds : ℚ := total_ounces / ounces_per_pound
  total_pounds = 8.25 := by sorry

end NUMINAMATH_CALUDE_rice_weight_in_pounds_l3097_309708


namespace NUMINAMATH_CALUDE_alternative_configuration_beats_malfatti_l3097_309775

/-- Given an equilateral triangle with side length 1, the total area of three circles
    in an alternative configuration is greater than the total area of Malfatti circles. -/
theorem alternative_configuration_beats_malfatti :
  let malfatti_area : ℝ := 3 * Real.pi * (2 - Real.sqrt 3) / 8
  let alternative_area : ℝ := 11 * Real.pi / 108
  alternative_area > malfatti_area :=
by sorry

end NUMINAMATH_CALUDE_alternative_configuration_beats_malfatti_l3097_309775


namespace NUMINAMATH_CALUDE_opposite_lime_is_black_l3097_309712

-- Define the colors
inductive Color
  | Purple
  | Cyan
  | Magenta
  | Silver
  | Black
  | Lime

-- Define a square with a color
structure Square where
  color : Color

-- Define a cube made of squares
structure Cube where
  squares : List Square
  hinged : squares.length = 6

-- Define the opposite face relation
def oppositeFace (c : Cube) (f1 f2 : Square) : Prop :=
  f1 ∈ c.squares ∧ f2 ∈ c.squares ∧ f1 ≠ f2

-- Theorem statement
theorem opposite_lime_is_black (c : Cube) :
  ∃ (lime_face black_face : Square),
    lime_face.color = Color.Lime ∧
    black_face.color = Color.Black ∧
    oppositeFace c lime_face black_face :=
  sorry


end NUMINAMATH_CALUDE_opposite_lime_is_black_l3097_309712


namespace NUMINAMATH_CALUDE_ages_sum_l3097_309716

theorem ages_sum (a b c : ℕ+) (h1 : b = c) (h2 : a * b * c = 72) : a + b + c = 14 := by
  sorry

end NUMINAMATH_CALUDE_ages_sum_l3097_309716


namespace NUMINAMATH_CALUDE_power_product_equality_l3097_309796

theorem power_product_equality (a b : ℝ) : (2 * a^2 * b)^3 = 8 * a^6 * b^3 := by
  sorry

end NUMINAMATH_CALUDE_power_product_equality_l3097_309796


namespace NUMINAMATH_CALUDE_horse_speed_around_square_field_l3097_309769

/-- Given a square field with area 625 km^2 and a horse that runs around it in 4 hours,
    prove that the speed of the horse is 25 km/hour. -/
theorem horse_speed_around_square_field (area : ℝ) (time : ℝ) (horse_speed : ℝ) : 
  area = 625 → time = 4 → horse_speed = (4 * Real.sqrt area) / time → horse_speed = 25 := by
  sorry

end NUMINAMATH_CALUDE_horse_speed_around_square_field_l3097_309769


namespace NUMINAMATH_CALUDE_simplify_complex_fraction_l3097_309788

theorem simplify_complex_fraction : 
  (1 / ((1 / (Real.sqrt 3 + 1)) + (2 / (Real.sqrt 5 - 2)))) = Real.sqrt 3 - 2 * Real.sqrt 5 - 3 := by
  sorry

end NUMINAMATH_CALUDE_simplify_complex_fraction_l3097_309788


namespace NUMINAMATH_CALUDE_four_number_theorem_l3097_309736

theorem four_number_theorem (a b c d : ℕ+) (h : a * b = c * d) :
  ∃ (p q r s : ℕ+), a = p * q ∧ b = r * s ∧ c = p * s ∧ d = q * r := by
  sorry

end NUMINAMATH_CALUDE_four_number_theorem_l3097_309736


namespace NUMINAMATH_CALUDE_triangle_fence_problem_l3097_309776

theorem triangle_fence_problem (a b c : ℕ) : 
  a ≤ b → b ≤ c → 
  a + b + c = 2022 → 
  c - b = 1 → 
  b - a = 2 → 
  b = 674 := by
sorry

end NUMINAMATH_CALUDE_triangle_fence_problem_l3097_309776


namespace NUMINAMATH_CALUDE_investment_comparison_l3097_309729

def initial_aa : ℝ := 150
def initial_bb : ℝ := 120
def initial_cc : ℝ := 100

def year1_aa_change : ℝ := 1.15
def year1_bb_change : ℝ := 0.70
def year1_cc_change : ℝ := 1.00

def year2_aa_change : ℝ := 0.85
def year2_bb_change : ℝ := 1.20
def year2_cc_change : ℝ := 1.00

def year3_aa_change : ℝ := 1.10
def year3_bb_change : ℝ := 0.95
def year3_cc_change : ℝ := 1.05

def final_aa : ℝ := initial_aa * year1_aa_change * year2_aa_change * year3_aa_change
def final_bb : ℝ := initial_bb * year1_bb_change * year2_bb_change * year3_bb_change
def final_cc : ℝ := initial_cc * year1_cc_change * year2_cc_change * year3_cc_change

theorem investment_comparison : final_bb < final_cc ∧ final_cc < final_aa :=
sorry

end NUMINAMATH_CALUDE_investment_comparison_l3097_309729


namespace NUMINAMATH_CALUDE_hyperbola_eccentricity_l3097_309723

/-- The eccentricity of a hyperbola with specific conditions -/
theorem hyperbola_eccentricity (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  let C := {(x, y) : ℝ × ℝ | x^2 / a^2 - y^2 / b^2 = 1}
  let l := {(x, y) : ℝ × ℝ | x - 2*y + 1 = 0}
  let asymptote_slope := b / a
  let line_slope := 1 / 2
  (asymptote_slope = 2 * line_slope / (1 - line_slope^2)) →
  Real.sqrt (1 + (b/a)^2) = 5/3 :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_eccentricity_l3097_309723


namespace NUMINAMATH_CALUDE_trajectory_is_line_segment_l3097_309799

/-- The trajectory of a point P(x,y) satisfying |PF₁| + |PF₂| = 10, where F₁(-5,0) and F₂(5,0) are fixed points, is a line segment. -/
theorem trajectory_is_line_segment :
  ∀ (x y : ℝ),
  let P : ℝ × ℝ := (x, y)
  let F₁ : ℝ × ℝ := (-5, 0)
  let F₂ : ℝ × ℝ := (5, 0)
  let dist (A B : ℝ × ℝ) := Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2)
  dist P F₁ + dist P F₂ = 10 →
  ∃ (A B : ℝ × ℝ), P ∈ Set.Icc A B :=
by sorry

end NUMINAMATH_CALUDE_trajectory_is_line_segment_l3097_309799


namespace NUMINAMATH_CALUDE_possible_values_of_y_l3097_309751

theorem possible_values_of_y (x y : ℝ) :
  |x - Real.sin (Real.log y)| = x + Real.sin (Real.log y) →
  ∃ n : ℤ, y = Real.exp (2 * π * ↑n) :=
sorry

end NUMINAMATH_CALUDE_possible_values_of_y_l3097_309751


namespace NUMINAMATH_CALUDE_unicorn_rope_length_l3097_309783

theorem unicorn_rope_length (rope_length : ℝ) (tower_radius : ℝ) (rope_end_distance : ℝ) 
  (h1 : rope_length = 24)
  (h2 : tower_radius = 10)
  (h3 : rope_end_distance = 6) :
  rope_length - 2 * Real.sqrt (rope_length^2 - tower_radius^2) = 24 - 2 * Real.sqrt 119 :=
by sorry

end NUMINAMATH_CALUDE_unicorn_rope_length_l3097_309783


namespace NUMINAMATH_CALUDE_middle_number_8th_row_l3097_309791

/-- Represents a number in the array -/
def ArrayNumber (row : ℕ) (position : ℕ) : ℕ := sorry

/-- The number of elements in the nth row -/
def RowLength (n : ℕ) : ℕ := 2 * n - 1

/-- The last number in the nth row -/
def LastNumber (n : ℕ) : ℕ := n ^ 2

/-- The middle position in a row -/
def MiddlePosition (n : ℕ) : ℕ := n

theorem middle_number_8th_row :
  ∀ (row position : ℕ),
  (∀ n : ℕ, LastNumber n = ArrayNumber n (RowLength n)) →
  (∀ n : ℕ, RowLength n = 2 * n - 1) →
  ArrayNumber 8 (MiddlePosition 8) = 57 := by sorry

end NUMINAMATH_CALUDE_middle_number_8th_row_l3097_309791


namespace NUMINAMATH_CALUDE_tree_planting_problem_l3097_309774

theorem tree_planting_problem (a o c m : ℕ) 
  (ha : a = 47)
  (ho : o = 27)
  (hm : m = a * o)
  (hc : c = a - 15) :
  a = 47 ∧ o = 27 ∧ c = 32 ∧ m = 1269 := by
  sorry

end NUMINAMATH_CALUDE_tree_planting_problem_l3097_309774


namespace NUMINAMATH_CALUDE_fenced_area_calculation_l3097_309710

/-- The area of a rectangle with two square cut-outs at opposite corners -/
def fencedArea (length width cutout1 cutout2 : ℝ) : ℝ :=
  length * width - cutout1^2 - cutout2^2

/-- Theorem stating that the area of the fenced region is 340 square feet -/
theorem fenced_area_calculation :
  fencedArea 20 18 4 2 = 340 := by
  sorry

end NUMINAMATH_CALUDE_fenced_area_calculation_l3097_309710


namespace NUMINAMATH_CALUDE_inscribed_rhombus_radius_l3097_309718

/-- A rhombus inscribed in the intersection of two equal circles -/
structure InscribedRhombus where
  /-- The length of one diagonal of the rhombus -/
  diagonal1 : ℝ
  /-- The length of the other diagonal of the rhombus -/
  diagonal2 : ℝ
  /-- The radius of the circles -/
  radius : ℝ
  /-- The diagonals are positive -/
  diagonal1_pos : diagonal1 > 0
  diagonal2_pos : diagonal2 > 0
  /-- The radius is positive -/
  radius_pos : radius > 0
  /-- The relationship between the diagonals and the radius -/
  radius_eq : radius^2 = (radius - diagonal1/2)^2 + (diagonal2/2)^2

/-- The theorem stating that a rhombus with diagonals 12 and 6 inscribed in two equal circles implies the radius is 7.5 -/
theorem inscribed_rhombus_radius (r : InscribedRhombus) (h1 : r.diagonal1 = 6) (h2 : r.diagonal2 = 12) : 
  r.radius = 7.5 := by
  sorry

end NUMINAMATH_CALUDE_inscribed_rhombus_radius_l3097_309718


namespace NUMINAMATH_CALUDE_largest_int_less_100_rem_4_div_7_l3097_309756

theorem largest_int_less_100_rem_4_div_7 : ∃ n : ℕ, n < 100 ∧ n % 7 = 4 ∧ ∀ m : ℕ, m < 100 → m % 7 = 4 → m ≤ n :=
by sorry

end NUMINAMATH_CALUDE_largest_int_less_100_rem_4_div_7_l3097_309756


namespace NUMINAMATH_CALUDE_complex_modulus_equality_l3097_309772

theorem complex_modulus_equality (x y : ℝ) : 
  (Complex.I + 1) * x = Complex.I * y + 1 → Complex.abs (x + Complex.I * y) = Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_complex_modulus_equality_l3097_309772


namespace NUMINAMATH_CALUDE_brianna_marbles_l3097_309781

/-- The number of marbles Brianna lost through the hole in the bag. -/
def L : ℕ := sorry

/-- The total number of marbles Brianna started with. -/
def total : ℕ := 24

/-- The number of marbles Brianna had remaining. -/
def remaining : ℕ := 10

theorem brianna_marbles : 
  L + 2 * L + L / 2 = total - remaining ∧ L = 4 := by sorry

end NUMINAMATH_CALUDE_brianna_marbles_l3097_309781


namespace NUMINAMATH_CALUDE_hyperbola_focal_length_l3097_309705

/-- The focal length of a hyperbola with equation x²- y²/4 = 1 is 2√5 -/
theorem hyperbola_focal_length : 
  let h : Set ((ℝ × ℝ) → Prop) := {f | ∃ (x y : ℝ), f (x, y) ↔ x^2 - y^2/4 = 1}
  ∃ (f : (ℝ × ℝ) → Prop), f ∈ h ∧ 
    (∃ (a b c : ℝ), a^2 = 1 ∧ b^2 = 4 ∧ c^2 = a^2 + b^2 ∧ 2*c = 2*Real.sqrt 5) :=
by
  sorry

end NUMINAMATH_CALUDE_hyperbola_focal_length_l3097_309705


namespace NUMINAMATH_CALUDE_painted_cube_theorem_l3097_309724

theorem painted_cube_theorem (n : ℕ) (h : n > 2) :
  6 * (n - 2)^2 = (n - 2)^3 ↔ n = 8 := by
  sorry

end NUMINAMATH_CALUDE_painted_cube_theorem_l3097_309724


namespace NUMINAMATH_CALUDE_x_seventh_plus_64x_squared_l3097_309741

theorem x_seventh_plus_64x_squared (x : ℝ) (h : x^3 + 4*x = 8) : x^7 + 64*x^2 = 128 := by
  sorry

end NUMINAMATH_CALUDE_x_seventh_plus_64x_squared_l3097_309741


namespace NUMINAMATH_CALUDE_terrys_spending_ratio_l3097_309767

/-- Terry's spending problem -/
theorem terrys_spending_ratio :
  ∀ (monday tuesday wednesday total : ℚ),
    monday = 6 →
    tuesday = 2 * monday →
    total = monday + tuesday + wednesday →
    total = 54 →
    wednesday = 2 * (monday + tuesday) :=
by sorry

end NUMINAMATH_CALUDE_terrys_spending_ratio_l3097_309767


namespace NUMINAMATH_CALUDE_train_overtake_time_specific_overtake_time_l3097_309760

/-- The time it takes for a train to overtake a motorbike -/
theorem train_overtake_time (train_speed : ℝ) (motorbike_speed : ℝ) (train_length : ℝ) : ℝ :=
  let relative_speed_kmph := train_speed - motorbike_speed
  let relative_speed_mps := relative_speed_kmph * 1000 / 3600
  train_length / relative_speed_mps

/-- Proof of the specific overtake time given the problem conditions -/
theorem specific_overtake_time : 
  train_overtake_time 100 64 800.064 = 80.0064 := by
  sorry

end NUMINAMATH_CALUDE_train_overtake_time_specific_overtake_time_l3097_309760


namespace NUMINAMATH_CALUDE_negation_of_forall_proposition_l3097_309792

theorem negation_of_forall_proposition :
  (¬ ∀ x : ℝ, x > 2 → x^2 + 2 > 6) ↔ (∃ x : ℝ, x > 2 ∧ x^2 + 2 ≤ 6) := by
  sorry

end NUMINAMATH_CALUDE_negation_of_forall_proposition_l3097_309792


namespace NUMINAMATH_CALUDE_prime_sum_112_l3097_309733

theorem prime_sum_112 :
  ∃ (S : Finset Nat), 
    (∀ p ∈ S, Nat.Prime p ∧ p > 10) ∧ 
    (S.sum id = 112) ∧ 
    (S.card = 6) := by
  sorry

end NUMINAMATH_CALUDE_prime_sum_112_l3097_309733


namespace NUMINAMATH_CALUDE_eighth_power_sum_l3097_309768

theorem eighth_power_sum (a b : ℝ) 
  (h1 : a + b = 1)
  (h2 : a^2 + b^2 = 3)
  (h3 : a^3 + b^3 = 4)
  (h4 : a^4 + b^4 = 7)
  (h5 : a^5 + b^5 = 11) : 
  a^8 + b^8 = 47 := by
  sorry

end NUMINAMATH_CALUDE_eighth_power_sum_l3097_309768


namespace NUMINAMATH_CALUDE_max_lessons_with_clothing_constraints_l3097_309758

/-- The maximum number of lessons an instructor can conduct given specific clothing constraints -/
theorem max_lessons_with_clothing_constraints :
  ∀ (x y z : ℕ),
  (x > 0) → (y > 0) → (z > 0) →
  (3 * y * z = 18) →
  (3 * x * z = 63) →
  (3 * x * y = 42) →
  (3 * x * y * z = 126) := by
sorry

end NUMINAMATH_CALUDE_max_lessons_with_clothing_constraints_l3097_309758


namespace NUMINAMATH_CALUDE_orange_juice_fraction_l3097_309732

-- Define the capacities of the pitchers
def pitcher1_capacity : ℚ := 800
def pitcher2_capacity : ℚ := 700

-- Define the fractions of orange juice in each pitcher
def pitcher1_juice_fraction : ℚ := 1/4
def pitcher2_juice_fraction : ℚ := 1/3

-- Calculate the amount of orange juice in each pitcher
def pitcher1_juice : ℚ := pitcher1_capacity * pitcher1_juice_fraction
def pitcher2_juice : ℚ := pitcher2_capacity * pitcher2_juice_fraction

-- Calculate the total amount of orange juice
def total_juice : ℚ := pitcher1_juice + pitcher2_juice

-- Calculate the total volume of the mixture
def total_volume : ℚ := pitcher1_capacity + pitcher2_capacity

-- Define the fraction of orange juice in the large container
def juice_fraction : ℚ := total_juice / total_volume

-- Theorem to prove
theorem orange_juice_fraction :
  juice_fraction = 433.33 / 1500 := by sorry

end NUMINAMATH_CALUDE_orange_juice_fraction_l3097_309732


namespace NUMINAMATH_CALUDE_table_satisfies_conditions_l3097_309744

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d > 1 → d < n → ¬(n % d = 0)

def is_perfect_square (n : ℕ) : Prop := ∃ m : ℕ, n = m * m

def is_consecutive_prime_product (n : ℕ) : Prop :=
  ∃ p q : ℕ, is_prime p ∧ is_prime q ∧ q = p + 2 ∧ n = p * q

def table : Matrix (Fin 4) (Fin 4) ℕ :=
  ![![2, 1, 8, 7],
    ![7, 3, 8, 7],
    ![7, 7, 4, 4],
    ![7, 8, 4, 4]]

theorem table_satisfies_conditions :
  (∀ i j, table i j < 10) ∧
  (∀ i, table i 0 ≠ 0) ∧
  (∃ p q : ℕ, is_prime p ∧ is_prime q ∧ 
    1000 * table 0 0 + 100 * table 0 1 + 10 * table 0 2 + table 0 3 = p^q) ∧
  (is_consecutive_prime_product 
    (1000 * table 1 0 + 100 * table 1 1 + 10 * table 1 2 + table 1 3)) ∧
  (is_perfect_square 
    (1000 * table 2 0 + 100 * table 2 1 + 10 * table 2 2 + table 2 3)) ∧
  ((1000 * table 3 0 + 100 * table 3 1 + 10 * table 3 2 + table 3 3) % 37 = 0) :=
by sorry

end NUMINAMATH_CALUDE_table_satisfies_conditions_l3097_309744
