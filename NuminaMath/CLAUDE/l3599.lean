import Mathlib

namespace f_geq_g_for_all_real_l3599_359996

theorem f_geq_g_for_all_real : ∀ x : ℝ, x^2 * Real.exp x ≥ 2 * x^3 := by
  sorry

end f_geq_g_for_all_real_l3599_359996


namespace intersection_quadrilateral_perimeter_bounds_l3599_359920

/-- A regular tetrahedron with edge length a -/
structure RegularTetrahedron where
  a : ℝ
  a_pos : 0 < a

/-- A quadrilateral formed by the intersection of a plane and a regular tetrahedron -/
structure IntersectionQuadrilateral (t : RegularTetrahedron) where
  perimeter : ℝ

/-- The theorem stating that the perimeter of the intersection quadrilateral
    is bounded between 2a and 3a -/
theorem intersection_quadrilateral_perimeter_bounds
  (t : RegularTetrahedron) (q : IntersectionQuadrilateral t) :
  2 * t.a ≤ q.perimeter ∧ q.perimeter ≤ 3 * t.a :=
sorry

end intersection_quadrilateral_perimeter_bounds_l3599_359920


namespace parabola_points_condition_l3599_359993

/-- The parabola equation -/
def parabola (x y k : ℝ) : Prop := y = -2 * (x - 1)^2 + k

theorem parabola_points_condition (m y₁ y₂ k : ℝ) :
  parabola (m - 1) y₁ k →
  parabola m y₂ k →
  y₁ > y₂ →
  m > 3/2 := by sorry

end parabola_points_condition_l3599_359993


namespace trigonometric_expression_value_l3599_359916

theorem trigonometric_expression_value :
  (Real.sin (330 * π / 180) * Real.tan (-13 * π / 3)) /
  (Real.cos (-19 * π / 6) * Real.cos (690 * π / 180)) = -2 * Real.sqrt 3 / 3 := by
  sorry

end trigonometric_expression_value_l3599_359916


namespace max_rooks_on_100x100_board_l3599_359968

/-- Represents a chessboard --/
structure Chessboard :=
  (size : ℕ)

/-- Represents a nearsighted rook --/
structure NearsightedRook :=
  (range : ℕ)

/-- Calculates the maximum number of non-attacking nearsighted rooks on a chessboard --/
def max_non_attacking_rooks (board : Chessboard) (rook : NearsightedRook) : ℕ :=
  sorry

/-- Theorem stating the maximum number of non-attacking nearsighted rooks on a 100x100 board --/
theorem max_rooks_on_100x100_board :
  let board : Chessboard := ⟨100⟩
  let rook : NearsightedRook := ⟨60⟩
  max_non_attacking_rooks board rook = 178 :=
sorry

end max_rooks_on_100x100_board_l3599_359968


namespace m_range_l3599_359985

theorem m_range (m : ℝ) : 
  (∃ x₀ : ℝ, x₀^2 + m ≤ 0) ∧ 
  (∀ x : ℝ, x^2 + m*x + 1 > 0) ∧ 
  ¬((¬∃ x₀ : ℝ, x₀^2 + m ≤ 0) ∨ (∀ x : ℝ, x^2 + m*x + 1 > 0)) → 
  m ≤ -2 := by
sorry

end m_range_l3599_359985


namespace nested_fraction_evaluation_l3599_359963

theorem nested_fraction_evaluation :
  1 + 1 / (2 + 1 / (3 + 2)) = 16 / 11 := by
  sorry

end nested_fraction_evaluation_l3599_359963


namespace certain_number_problem_l3599_359977

theorem certain_number_problem (y : ℝ) : 
  (0.25 * 900 = 0.15 * y - 15) → y = 1600 := by
  sorry

end certain_number_problem_l3599_359977


namespace slope_of_l₃_l3599_359918

-- Define the lines and points
def l₁ (x y : ℝ) : Prop := 4 * x - 3 * y = 2
def l₂ (x y : ℝ) : Prop := y = 2
def A : ℝ × ℝ := (0, -3)

-- Define the existence of point B
def B_exists : Prop := ∃ x y : ℝ, l₁ x y ∧ l₂ x y

-- Define the existence of point C
def C_exists : Prop := ∃ x : ℝ, l₂ x (2 : ℝ)

-- Define the properties of line l₃
def l₃_properties (m : ℝ) : Prop :=
  m > 0 ∧ 
  (∃ b : ℝ, ∀ x : ℝ, m * x + b = -3 → x = 0) ∧
  (∃ x : ℝ, l₂ x (m * x + -3))

-- Define the area of triangle ABC
def triangle_area (m : ℝ) : Prop :=
  ∃ B C : ℝ × ℝ, 
    l₁ B.1 B.2 ∧ l₂ B.1 B.2 ∧
    l₂ C.1 C.2 ∧
    (C.2 - A.2) = m * (C.1 - A.1) ∧
    abs ((B.1 - A.1) * (C.2 - A.2) - (C.1 - A.1) * (B.2 - A.2)) = 10

-- Theorem statement
theorem slope_of_l₃ :
  B_exists → C_exists → ∃ m : ℝ, l₃_properties m ∧ triangle_area m → m = 5/4 := by
  sorry

end slope_of_l₃_l3599_359918


namespace money_left_after_gift_l3599_359927

def gift_package_cost : ℚ := 445
def erika_savings : ℚ := 155
def sam_savings : ℚ := 175
def cake_flowers_skincare_cost : ℚ := 25 + 35 + 45

def rick_savings : ℚ := gift_package_cost / 2
def amy_savings : ℚ := 2 * cake_flowers_skincare_cost

def total_savings : ℚ := erika_savings + rick_savings + sam_savings + amy_savings

theorem money_left_after_gift (h : total_savings - gift_package_cost = 317.5) :
  total_savings - gift_package_cost = 317.5 := by
  sorry

end money_left_after_gift_l3599_359927


namespace chloe_carrot_count_l3599_359917

/-- Given Chloe's carrot picking scenario, prove the final number of carrots. -/
theorem chloe_carrot_count (initial_carrots : ℕ) (thrown_out : ℕ) (picked_next_day : ℕ) 
  (h1 : initial_carrots = 48)
  (h2 : thrown_out = 45)
  (h3 : picked_next_day = 42) :
  initial_carrots - thrown_out + picked_next_day = 45 :=
by sorry

end chloe_carrot_count_l3599_359917


namespace at_least_one_geq_two_l3599_359958

theorem at_least_one_geq_two (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  (a + 1/b ≥ 2) ∨ (b + 1/c ≥ 2) ∨ (c + 1/a ≥ 2) := by
  sorry

end at_least_one_geq_two_l3599_359958


namespace angle_after_folding_is_60_degrees_l3599_359959

/-- An isosceles right triangle -/
structure IsoscelesRightTriangle where
  /-- The two equal sides of the triangle -/
  leg : ℝ
  /-- The hypotenuse of the triangle -/
  hypotenuse : ℝ
  /-- The condition that it's a right triangle -/
  right_angle : hypotenuse^2 = 2 * leg^2

/-- The angle between the legs after folding an isosceles right triangle along its height to the hypotenuse -/
def angle_after_folding (t : IsoscelesRightTriangle) : ℝ := sorry

/-- Theorem stating that the angle between the legs after folding is 60° -/
theorem angle_after_folding_is_60_degrees (t : IsoscelesRightTriangle) :
  angle_after_folding t = 60 * (π / 180) := by sorry

end angle_after_folding_is_60_degrees_l3599_359959


namespace compare_expressions_l3599_359965

theorem compare_expressions (x : ℝ) (h : x ≥ 0) :
  (x > 2 → 5*x^2 - 1 > 3*x^2 + 3*x + 1) ∧
  (x = 2 → 5*x^2 - 1 = 3*x^2 + 3*x + 1) ∧
  (0 ≤ x ∧ x < 2 → 5*x^2 - 1 < 3*x^2 + 3*x + 1) :=
by sorry

end compare_expressions_l3599_359965


namespace min_score_game_12_is_42_l3599_359992

/-- Represents the scores of a football player over a season -/
structure FootballScores where
  first_seven : ℕ  -- Total score for first 7 games
  game_8 : ℕ := 18
  game_9 : ℕ := 25
  game_10 : ℕ := 10
  game_11 : ℕ := 22
  game_12 : ℕ

/-- The minimum score for game 12 that satisfies all conditions -/
def min_score_game_12 (scores : FootballScores) : Prop :=
  let total_8_to_11 := scores.game_8 + scores.game_9 + scores.game_10 + scores.game_11
  let avg_8_to_11 : ℚ := total_8_to_11 / 4
  let total_12_games := scores.first_seven + total_8_to_11 + scores.game_12
  (scores.first_seven / 7 : ℚ) < (total_12_games - scores.game_12) / 11 ∧ 
  (total_12_games : ℚ) / 12 > 20 ∧
  scores.game_12 = 42 ∧
  ∀ x : ℕ, x < 42 → 
    let total_with_x := scores.first_seven + total_8_to_11 + x
    (total_with_x : ℚ) / 12 ≤ 20 ∨ (scores.first_seven / 7 : ℚ) ≥ (total_with_x - x) / 11

theorem min_score_game_12_is_42 (scores : FootballScores) :
  min_score_game_12 scores := by sorry

end min_score_game_12_is_42_l3599_359992


namespace sequence_length_l3599_359982

def arithmetic_sequence (a₁ : ℝ) (d : ℝ) (n : ℕ) : ℝ := a₁ + d * (n - 1 : ℝ)

theorem sequence_length :
  ∃ n : ℕ, n > 0 ∧ arithmetic_sequence 2.5 5 n = 87.5 ∧
  ∀ m : ℕ, m > 0 ∧ m ≠ n → arithmetic_sequence 2.5 5 m ≠ 87.5 :=
by
  use 18
  sorry

end sequence_length_l3599_359982


namespace bank_queue_properties_l3599_359939

/-- Represents a bank queue with simple and long operations -/
structure BankQueue where
  total_people : Nat
  simple_ops : Nat
  long_ops : Nat
  simple_time : Nat
  long_time : Nat

/-- Calculates the minimum wasted person-minutes -/
def min_wasted_time (q : BankQueue) : Nat :=
  sorry

/-- Calculates the maximum wasted person-minutes -/
def max_wasted_time (q : BankQueue) : Nat :=
  sorry

/-- Calculates the expected wasted person-minutes assuming random order -/
def expected_wasted_time (q : BankQueue) : Nat :=
  sorry

/-- Theorem stating the properties of the bank queue problem -/
theorem bank_queue_properties (q : BankQueue) 
  (h1 : q.total_people = 8)
  (h2 : q.simple_ops = 5)
  (h3 : q.long_ops = 3)
  (h4 : q.simple_time = 1)
  (h5 : q.long_time = 5) :
  min_wasted_time q = 40 ∧ 
  max_wasted_time q = 100 ∧
  expected_wasted_time q = 84 :=
by sorry

end bank_queue_properties_l3599_359939


namespace solve_equation_l3599_359987

theorem solve_equation (k l q : ℚ) : 
  (3/4 : ℚ) = k/108 ∧ 
  (3/4 : ℚ) = (l+k)/126 ∧ 
  (3/4 : ℚ) = (q-l)/180 → 
  q = 148.5 := by
sorry

end solve_equation_l3599_359987


namespace arithmetic_sequence_2017_l3599_359936

def arithmetic_sequence (a₁ : ℝ) (d : ℝ) (n : ℕ) : ℝ := a₁ + d * (n - 1)

theorem arithmetic_sequence_2017 :
  arithmetic_sequence 4 3 672 = 2017 := by
  sorry

end arithmetic_sequence_2017_l3599_359936


namespace min_value_theorem_l3599_359971

theorem min_value_theorem (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : a + 1/b = 2) :
  ∀ x y : ℝ, x > 0 → y > 0 → x + 1/y = 2 → 2*x*y + 1/x ≥ 5/2 :=
by sorry

end min_value_theorem_l3599_359971


namespace max_a_for_quadratic_inequality_l3599_359906

theorem max_a_for_quadratic_inequality :
  (∀ x : ℝ, x^2 - a*x + a ≥ 0) → a ≤ 4 :=
by sorry

end max_a_for_quadratic_inequality_l3599_359906


namespace arithmetic_sequence_sum_l3599_359976

/-- An arithmetic sequence with given properties -/
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  (∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d) ∧
  a 1 = 2 ∧
  a 2 + a 3 = 13

/-- The sum of the 4th, 5th, and 6th terms equals 42 -/
theorem arithmetic_sequence_sum (a : ℕ → ℝ) (h : arithmetic_sequence a) :
  a 4 + a 5 + a 6 = 42 := by sorry

end arithmetic_sequence_sum_l3599_359976


namespace family_reunion_food_l3599_359973

/-- The total amount of food Peter buys for the family reunion -/
def total_food (chicken : ℝ) (hamburger_ratio : ℝ) (hotdog_difference : ℝ) (sides_ratio : ℝ) : ℝ :=
  let hamburger := chicken * hamburger_ratio
  let hotdog := hamburger + hotdog_difference
  let sides := hotdog * sides_ratio
  chicken + hamburger + hotdog + sides

/-- Theorem stating the total amount of food Peter will buy -/
theorem family_reunion_food :
  total_food 16 (1/2) 2 (1/2) = 39 := by
  sorry

end family_reunion_food_l3599_359973


namespace triangle_inequality_with_medians_l3599_359901

/-- Given a triangle with sides a, b, c and medians s_a, s_b, s_c, 
    prove the inequality a + b + c > s_a + s_b + s_c > 3/4 * (a + b + c) -/
theorem triangle_inequality_with_medians 
  (a b c s_a s_b s_c : ℝ) 
  (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_pos_c : 0 < c)
  (h_triangle : a + b > c ∧ b + c > a ∧ c + a > b)
  (h_median_a : s_a = (1/2) * Real.sqrt (2*b^2 + 2*c^2 - a^2))
  (h_median_b : s_b = (1/2) * Real.sqrt (2*a^2 + 2*c^2 - b^2))
  (h_median_c : s_c = (1/2) * Real.sqrt (2*a^2 + 2*b^2 - c^2)) :
  a + b + c > s_a + s_b + s_c ∧ s_a + s_b + s_c > (3/4) * (a + b + c) :=
by sorry

end triangle_inequality_with_medians_l3599_359901


namespace function_minimum_l3599_359903

theorem function_minimum (f : ℝ → ℝ) (a : ℝ) 
  (h_diff : Differentiable ℝ f) 
  (h_cond : ∀ x, (x - a) * (deriv f x) ≥ 0) :
  ∀ x, f x ≥ f a :=
by sorry

end function_minimum_l3599_359903


namespace safari_arrangement_l3599_359934

/-- Represents the number of animal pairs in the safari park -/
def num_pairs : ℕ := 6

/-- Calculates the number of ways to arrange animals with alternating genders -/
def arrange_animals : ℕ := sorry

/-- Theorem stating the number of ways to arrange the animals -/
theorem safari_arrangement :
  arrange_animals = 86400 := by sorry

end safari_arrangement_l3599_359934


namespace no_primitive_root_for_multiple_odd_primes_l3599_359933

theorem no_primitive_root_for_multiple_odd_primes (n : ℕ) 
  (h1 : ∃ (p q : ℕ), Prime p ∧ Prime q ∧ p ≠ q ∧ Odd p ∧ Odd q ∧ n % p = 0 ∧ n % q = 0) : 
  ¬ ∃ a : ℕ, IsPrimitiveRoot a n :=
sorry

end no_primitive_root_for_multiple_odd_primes_l3599_359933


namespace inequality_proof_l3599_359966

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (h_abc : a * b * c = 1) :
  (a + b + c + 3) / 4 ≥ 1 / (a + b) + 1 / (b + c) + 1 / (c + a) := by
  sorry

end inequality_proof_l3599_359966


namespace julia_tag_playmates_l3599_359911

/-- The number of kids Julia played tag with on Monday, Tuesday, and in total. -/
structure TagPlaymates where
  monday : ℕ
  tuesday : ℕ
  total : ℕ

/-- Given that Julia played tag with 20 kids in total and 13 kids on Tuesday,
    prove that she played tag with 7 kids on Monday. -/
theorem julia_tag_playmates : ∀ (j : TagPlaymates),
  j.total = 20 → j.tuesday = 13 → j.monday = 7 :=
by
  sorry

end julia_tag_playmates_l3599_359911


namespace function_max_value_solution_l3599_359925

/-- The function f(x) defined in the problem -/
def f (a : ℝ) (x : ℝ) : ℝ := 4 * x^2 - 4 * a * x + a^2 - 2 * a + 2

/-- The maximum value of f(x) on the interval [0, 2] -/
def max_value : ℝ := 3

/-- The theorem stating the solution -/
theorem function_max_value_solution (a : ℝ) :
  (∀ x ∈ Set.Icc 0 2, f a x ≤ max_value) ∧
  (∃ x ∈ Set.Icc 0 2, f a x = max_value) →
  a = 5 - Real.sqrt 10 ∨ a = 1 + Real.sqrt 2 := by
  sorry

end function_max_value_solution_l3599_359925


namespace unique_prime_triplet_l3599_359957

theorem unique_prime_triplet :
  ∀ p q r : ℕ,
    Prime p → Prime q → Prime r →
    p + q = r →
    ∃ n : ℕ, (r - p) * (q - p) - 27 * p = n^2 →
    p = 2 ∧ q = 29 ∧ r = 31 := by
  sorry

end unique_prime_triplet_l3599_359957


namespace tony_haircut_distance_l3599_359995

theorem tony_haircut_distance (total_distance halfway_distance groceries_distance doctor_distance : ℕ)
  (h1 : total_distance = 2 * halfway_distance)
  (h2 : halfway_distance = 15)
  (h3 : groceries_distance = 10)
  (h4 : doctor_distance = 5) :
  total_distance - (groceries_distance + doctor_distance) = 15 := by
  sorry

end tony_haircut_distance_l3599_359995


namespace triangle_abc_theorem_l3599_359998

open Real

theorem triangle_abc_theorem (A B C : ℝ) (a b c : ℝ) :
  0 < A ∧ 0 < B ∧ 0 < C ∧ A + B + C = π →
  a > 0 ∧ b > 0 ∧ c > 0 →
  sin A / a = sin B / b ∧ sin B / b = sin C / c →
  cos (2 * C) - cos (2 * A) = 2 * sin (π / 3 + C) * sin (π / 3 - C) →
  a = sqrt 3 →
  b ≥ a →
  A = π / 3 ∧ sqrt 3 ≤ 2 * b - c ∧ 2 * b - c < 2 * sqrt 3 :=
by sorry

end triangle_abc_theorem_l3599_359998


namespace purely_imaginary_implies_a_eq_three_halves_l3599_359956

/-- A complex number z is purely imaginary if its real part is zero and its imaginary part is non-zero. -/
def IsPurelyImaginary (z : ℂ) : Prop := z.re = 0 ∧ z.im ≠ 0

/-- The complex number z defined as (1+2i)(3+ai) where a is a real number. -/
def z (a : ℝ) : ℂ := (1 + 2*Complex.I) * (3 + a*Complex.I)

/-- If z is purely imaginary, then a = 3/2. -/
theorem purely_imaginary_implies_a_eq_three_halves :
  ∀ a : ℝ, IsPurelyImaginary (z a) → a = 3/2 := by sorry

end purely_imaginary_implies_a_eq_three_halves_l3599_359956


namespace whistle_solution_l3599_359964

/-- The number of whistles Sean, Charles, and Alex have. -/
def whistle_problem (W_Sean W_Charles W_Alex : ℕ) : Prop :=
  W_Sean = 2483 ∧ 
  W_Charles = W_Sean - 463 ∧
  W_Alex = W_Charles - 131

theorem whistle_solution :
  ∀ W_Sean W_Charles W_Alex : ℕ,
  whistle_problem W_Sean W_Charles W_Alex →
  W_Charles = 2020 ∧ 
  W_Alex = 1889 ∧
  W_Sean + W_Charles + W_Alex = 6392 :=
by
  sorry

#check whistle_solution

end whistle_solution_l3599_359964


namespace complement_of_N_in_M_l3599_359913

def M : Set ℝ := {-1, 0, 1}
def N : Set ℝ := {x | ∃ k : ℤ, x = Real.cos (k * Real.pi)}

theorem complement_of_N_in_M : M \ N = {0} := by sorry

end complement_of_N_in_M_l3599_359913


namespace min_domain_length_l3599_359947

open Real

theorem min_domain_length (f : ℝ → ℝ) (m n : ℝ) :
  (∀ x ∈ Set.Icc m n, f x = sin x * sin (x + π/3) - 1/4) →
  m < n →
  Set.range f = Set.Icc (-1/2) (1/4) →
  n - m ≥ 2*π/3 :=
sorry

end min_domain_length_l3599_359947


namespace min_value_f_and_sum_squares_l3599_359952

def f (x : ℝ) : ℝ := |x - 4| + |x - 3|

theorem min_value_f_and_sum_squares :
  (∃ (m : ℝ), ∀ (x : ℝ), f x ≥ m ∧ (∃ (y : ℝ), f y = m) ∧ m = 1) ∧
  (∀ (a b c : ℝ), a + 2*b + 3*c = 1 → a^2 + b^2 + c^2 ≥ 1/14) ∧
  (∃ (a b c : ℝ), a + 2*b + 3*c = 1 ∧ a^2 + b^2 + c^2 = 1/14) := by
  sorry

end min_value_f_and_sum_squares_l3599_359952


namespace shopper_receives_115_l3599_359909

/-- Represents the amount of money each person has -/
structure MoneyDistribution where
  isabella : ℕ
  sam : ℕ
  giselle : ℕ

/-- Calculates the amount each shopper receives when the total is divided equally -/
def amountPerShopper (md : MoneyDistribution) : ℕ :=
  (md.isabella + md.sam + md.giselle) / 3

/-- Theorem stating the amount each shopper receives under the given conditions -/
theorem shopper_receives_115 (md : MoneyDistribution) 
  (h1 : md.isabella = md.sam + 45)
  (h2 : md.isabella = md.giselle + 15)
  (h3 : md.giselle = 120) :
  amountPerShopper md = 115 := by
  sorry

#eval amountPerShopper { isabella := 135, sam := 90, giselle := 120 }

end shopper_receives_115_l3599_359909


namespace smallest_integer_with_remainder_l3599_359932

theorem smallest_integer_with_remainder (n : ℕ) : 
  (n > 1) →
  (n % 3 = 2) →
  (n % 5 = 2) →
  (n % 7 = 2) →
  (∀ m : ℕ, m > 1 → m % 3 = 2 → m % 5 = 2 → m % 7 = 2 → n ≤ m) →
  (n = 107 ∧ 90 < n ∧ n < 119) := by
sorry

end smallest_integer_with_remainder_l3599_359932


namespace extra_flowers_l3599_359900

def tulips : ℕ := 36
def roses : ℕ := 37
def used_flowers : ℕ := 70

theorem extra_flowers :
  tulips + roses - used_flowers = 3 := by sorry

end extra_flowers_l3599_359900


namespace managers_salary_l3599_359912

theorem managers_salary (num_employees : ℕ) (avg_salary : ℝ) (salary_increase : ℝ) :
  num_employees = 20 →
  avg_salary = 1500 →
  salary_increase = 100 →
  (num_employees : ℝ) * avg_salary + (num_employees + 1 : ℝ) * salary_increase = 3600 :=
by sorry

end managers_salary_l3599_359912


namespace cory_fruit_arrangements_l3599_359999

/-- The number of ways to arrange indistinguishable objects of different types -/
def multinomial_coefficient (n : ℕ) (ks : List ℕ) : ℕ :=
  Nat.factorial n / (List.prod (List.map Nat.factorial ks))

/-- The number of distinct arrangements of Cory's fruit -/
theorem cory_fruit_arrangements :
  let total_fruit : ℕ := 7
  let fruit_counts : List ℕ := [3, 2, 2]
  multinomial_coefficient total_fruit fruit_counts = 210 := by
  sorry

end cory_fruit_arrangements_l3599_359999


namespace always_quadratic_l3599_359924

/-- Definition of a quadratic equation -/
def is_quadratic_equation (f : ℝ → ℝ) : Prop :=
  ∃ a b c : ℝ, a ≠ 0 ∧ ∀ x, f x = a * x^2 + b * x + c

/-- The equation (k^2+1)x^2-2x+1=0 is always a quadratic equation -/
theorem always_quadratic (k : ℝ) : 
  is_quadratic_equation (λ x => (k^2 + 1) * x^2 - 2 * x + 1) :=
sorry

end always_quadratic_l3599_359924


namespace product_of_roots_is_4y_squared_l3599_359922

-- Define a quadratic function f
variable (f : ℝ → ℝ)
variable (y : ℝ)

-- Assumptions
axiom f_is_quadratic : ∃ (a b c : ℝ), ∀ x, f x = a * x^2 + b * x + c
axiom root_of_f_x_minus_y : f (2*y - y) = 0
axiom root_of_f_x_plus_y : f (3*y + y) = 0

-- Theorem statement
theorem product_of_roots_is_4y_squared :
  (∃ (r₁ r₂ : ℝ), ∀ x, f x = 0 ↔ (x = r₁ ∨ x = r₂)) →
  (∃ (r₁ r₂ : ℝ), (∀ x, f x = 0 ↔ (x = r₁ ∨ x = r₂)) ∧ r₁ * r₂ = 4 * y^2) :=
by sorry

end product_of_roots_is_4y_squared_l3599_359922


namespace merchant_markup_theorem_l3599_359928

/-- Proves the required markup percentage for a merchant to achieve a specific profit --/
theorem merchant_markup_theorem (list_price : ℝ) (h_list_price_pos : 0 < list_price) :
  let cost_price := 0.7 * list_price
  let selling_price := list_price
  let marked_price := (5/4) * list_price
  (cost_price = 0.7 * selling_price) →
  (selling_price = 0.8 * marked_price) →
  (marked_price = 1.25 * list_price) :=
by
  sorry

#check merchant_markup_theorem

end merchant_markup_theorem_l3599_359928


namespace jerome_money_left_l3599_359937

def jerome_problem (initial_half : ℕ) (meg_amount : ℕ) : Prop :=
  let initial_total := 2 * initial_half
  let after_meg := initial_total - meg_amount
  let bianca_amount := 3 * meg_amount
  let final_amount := after_meg - bianca_amount
  final_amount = 54

theorem jerome_money_left : jerome_problem 43 8 := by
  sorry

end jerome_money_left_l3599_359937


namespace hedgehog_strawberries_l3599_359931

theorem hedgehog_strawberries : 
  ∀ (num_hedgehogs num_baskets strawberries_per_basket : ℕ) 
    (remaining_fraction : ℚ),
  num_hedgehogs = 2 →
  num_baskets = 3 →
  strawberries_per_basket = 900 →
  remaining_fraction = 2 / 9 →
  ∃ (strawberries_eaten_per_hedgehog : ℕ),
    strawberries_eaten_per_hedgehog = 1050 ∧
    (num_baskets * strawberries_per_basket) * (1 - remaining_fraction) = 
      num_hedgehogs * strawberries_eaten_per_hedgehog :=
by sorry

end hedgehog_strawberries_l3599_359931


namespace complex_modulus_l3599_359951

theorem complex_modulus (z : ℂ) (h : (z - 2) * Complex.I = 1 + Complex.I) : Complex.abs z = Real.sqrt 10 := by
  sorry

end complex_modulus_l3599_359951


namespace cylinder_volume_height_relation_l3599_359972

theorem cylinder_volume_height_relation (r h : ℝ) (h_pos : 0 < h) (r_pos : 0 < r) :
  let v := π * r^2 * h
  let r' := 2 * r
  let v' := 2 * v
  ∃ h', v' = π * r'^2 * h' ∧ h' = h / 4 :=
by sorry

end cylinder_volume_height_relation_l3599_359972


namespace total_fruits_l3599_359938

def persimmons : ℕ := 2
def apples : ℕ := 7

theorem total_fruits : persimmons + apples = 9 := by
  sorry

end total_fruits_l3599_359938


namespace stratified_sampling_second_grade_l3599_359942

/-- Represents the number of students in each grade -/
structure GradeDistribution where
  first : ℕ
  second : ℕ
  third : ℕ

/-- Calculates the total number of students -/
def totalStudents (dist : GradeDistribution) : ℕ :=
  dist.first + dist.second + dist.third

/-- Calculates the number of students to be sampled from a specific grade -/
def sampleSize (dist : GradeDistribution) (totalSample : ℕ) (grade : ℕ) : ℕ :=
  match grade with
  | 1 => (dist.first * totalSample) / (totalStudents dist)
  | 2 => (dist.second * totalSample) / (totalStudents dist)
  | 3 => (dist.third * totalSample) / (totalStudents dist)
  | _ => 0

theorem stratified_sampling_second_grade 
  (dist : GradeDistribution) 
  (h1 : dist.first = 1200) 
  (h2 : dist.second = 900) 
  (h3 : dist.third = 1500) 
  (h4 : totalStudents dist = 3600) 
  (h5 : sampleSize dist 720 2 = 480) : 
  sampleSize dist 720 2 = 480 := by
  sorry

end stratified_sampling_second_grade_l3599_359942


namespace trigonometric_fraction_bounds_l3599_359926

theorem trigonometric_fraction_bounds (x : ℝ) : 
  -1/3 ≤ (6 * Real.cos x + Real.sin x - 5) / (2 * Real.cos x - 3 * Real.sin x - 5) ∧ 
  (6 * Real.cos x + Real.sin x - 5) / (2 * Real.cos x - 3 * Real.sin x - 5) ≤ 3 :=
by sorry

end trigonometric_fraction_bounds_l3599_359926


namespace square_point_configuration_l3599_359902

-- Define the square and points
def Square (A B C D : Point) : Prop := sorry

def OnSegment (P Q R : Point) : Prop := sorry

-- Define angle measurement
def AngleMeasure (P Q R : Point) : ℝ := sorry

-- Main theorem
theorem square_point_configuration 
  (A B C D M N P : Point) 
  (x : ℝ) 
  (h_square : Square A B C D)
  (h_M : OnSegment B M C)
  (h_N : OnSegment C N D)
  (h_P : OnSegment D P A)
  (h_angle_AM : AngleMeasure A B M = x)
  (h_angle_MN : AngleMeasure B C N = 2 * x)
  (h_angle_NP : AngleMeasure C D P = 3 * x)
  (h_x_range : 0 ≤ x ∧ x ≤ 22.5) :
  (∃! (M N P : Point), 
    OnSegment B M C ∧ 
    OnSegment C N D ∧ 
    OnSegment D P A ∧
    AngleMeasure A B M = x ∧
    AngleMeasure B C N = 2 * x ∧
    AngleMeasure C D P = 3 * x) ∧
  (∀ Q, OnSegment D Q A → ∃ x, 
    0 ≤ x ∧ x ≤ 22.5 ∧
    AngleMeasure A B M = x ∧
    AngleMeasure B C N = 2 * x ∧
    AngleMeasure C D P = 3 * x ∧
    Q = P) ∧
  (∃ S : Set ℝ, S.Infinite ∧ 
    ∀ y ∈ S, 0 ≤ y ∧ y ≤ 22.5 ∧ 
    AngleMeasure D A B = 4 * y) :=
sorry

end square_point_configuration_l3599_359902


namespace root_values_l3599_359974

theorem root_values (a b c d e k : ℂ) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) (hd : d ≠ 0) (he : e ≠ 0)
  (hk1 : a * k^3 + b * k^2 + c * k + d = e)
  (hk2 : b * k^3 + c * k^2 + d * k + e = a) :
  k = 1 ∨ k = -1 ∨ k = Complex.I ∨ k = -Complex.I :=
sorry

end root_values_l3599_359974


namespace james_bike_ride_l3599_359978

theorem james_bike_ride (first_hour : ℝ) : 
  first_hour > 0 →
  let second_hour := 1.2 * first_hour
  let third_hour := 1.25 * second_hour
  first_hour + second_hour + third_hour = 55.5 →
  second_hour = 18 := by
sorry

end james_bike_ride_l3599_359978


namespace fib_F15_units_digit_l3599_359943

-- Define the Fibonacci sequence
def fib : ℕ → ℕ
| 0 => 0
| 1 => 1
| (n + 2) => fib (n + 1) + fib n

-- Define a function to get the units digit
def unitsDigit (n : ℕ) : ℕ := n % 10

-- Theorem statement
theorem fib_F15_units_digit :
  unitsDigit (fib (fib 15)) = 5 := by sorry

end fib_F15_units_digit_l3599_359943


namespace product_of_primes_between_n_and_2n_l3599_359907

theorem product_of_primes_between_n_and_2n (n : ℤ) :
  (n > 4 → ∃ p : ℕ, Prime p ∧ n < 2*p ∧ 2*p < 2*n) ∧
  (n > 15 → ∃ p : ℕ, Prime p ∧ n < 6*p ∧ 6*p < 2*n) :=
sorry

end product_of_primes_between_n_and_2n_l3599_359907


namespace smallest_b_value_l3599_359949

theorem smallest_b_value (a b : ℝ) : 
  (2 < a ∧ a < b) →
  (2 + a ≤ b) →
  (1 / a + 1 / b ≤ 2) →
  b ≥ (3 + Real.sqrt 7) / 2 :=
by sorry

end smallest_b_value_l3599_359949


namespace car_average_speed_l3599_359955

/-- Proves that the average speed of a car is 72 km/h given specific travel conditions -/
theorem car_average_speed (s : ℝ) (h : s > 0) : 
  let t1 := s / 2 / 60
  let t2 := s / 6 / 120
  let t3 := s / 3 / 80
  s / (t1 + t2 + t3) = 72 := by
  sorry

end car_average_speed_l3599_359955


namespace car_sale_profit_percentage_l3599_359997

/-- Calculates the profit percentage on a car sale given specific conditions --/
theorem car_sale_profit_percentage (P : ℝ) : 
  let discount_rate : ℝ := 0.1
  let discounted_price : ℝ := P * (1 - discount_rate)
  let first_year_expense_rate : ℝ := 0.05
  let second_year_expense_rate : ℝ := 0.04
  let third_year_expense_rate : ℝ := 0.03
  let selling_price_increase_rate : ℝ := 0.8
  
  let first_year_value : ℝ := discounted_price * (1 + first_year_expense_rate)
  let second_year_value : ℝ := first_year_value * (1 + second_year_expense_rate)
  let third_year_value : ℝ := second_year_value * (1 + third_year_expense_rate)
  
  let selling_price : ℝ := discounted_price * (1 + selling_price_increase_rate)
  let profit : ℝ := selling_price - P
  let profit_percentage : ℝ := (profit / P) * 100
  
  profit_percentage = 62 := by sorry

end car_sale_profit_percentage_l3599_359997


namespace bananas_per_box_l3599_359914

theorem bananas_per_box (total_bananas : ℕ) (num_boxes : ℕ) (h1 : total_bananas = 40) (h2 : num_boxes = 10) :
  total_bananas / num_boxes = 4 := by
  sorry

end bananas_per_box_l3599_359914


namespace constant_value_proof_l3599_359969

theorem constant_value_proof :
  ∀ (t : ℝ) (constant : ℝ),
    let x := 1 - 2 * t
    let y := constant * t - 2
    (t = 0.75 → x = y) →
    constant = 2 := by
  sorry

end constant_value_proof_l3599_359969


namespace nested_fraction_equality_l3599_359923

theorem nested_fraction_equality : 
  (1 : ℚ) / (3 - 1 / (3 - 1 / (3 - 1 / 3))) = 8 / 21 := by sorry

end nested_fraction_equality_l3599_359923


namespace b_value_function_comparison_l3599_359961

/-- A quadratic function with the given properties -/
def f (b c : ℝ) (x : ℝ) : ℝ := -x^2 + b*x + c

/-- The symmetry property of the function -/
axiom symmetry_property (b c : ℝ) : ∀ x : ℝ, f b c (2 + x) = f b c (2 - x)

/-- The value of b in the quadratic function -/
theorem b_value : ∃ b : ℝ, (∀ c x : ℝ, f b c (2 + x) = f b c (2 - x)) ∧ b = 4 :=
sorry

/-- Comparison of function values -/
theorem function_comparison (b c : ℝ) 
  (h : ∀ x : ℝ, f b c (2 + x) = f b c (2 - x)) : 
  ∀ a : ℝ, f b c (5/4) < f b c (-a^2 - a + 1) :=
sorry

end b_value_function_comparison_l3599_359961


namespace curve_tangent_parallel_l3599_359941

/-- The curve C: y = ax^3 + bx^2 + d -/
def C (a b d : ℝ) (x : ℝ) : ℝ := a * x^3 + b * x^2 + d

/-- The derivative of C with respect to x -/
def C' (a b : ℝ) (x : ℝ) : ℝ := 3 * a * x^2 + 2 * b * x

theorem curve_tangent_parallel (a b d : ℝ) :
  C a b d 1 = 1 →  -- Point A(1,1) lies on the curve
  C a b d (-1) = -3 →  -- Point B(-1,-3) lies on the curve
  C' a b 1 = C' a b (-1) →  -- Tangents at A and B are parallel
  a^3 + b^2 + d = 7 := by sorry

end curve_tangent_parallel_l3599_359941


namespace honey_servings_l3599_359983

/-- Calculates the number of full servings of honey in a container -/
def fullServings (containerAmount : Rat) (servingSize : Rat) : Rat :=
  containerAmount / servingSize

/-- Proves that a container with 47 2/3 tablespoons of honey provides 14 1/5 full servings when each serving is 3 1/3 tablespoons -/
theorem honey_servings :
  let containerAmount : Rat := 47 + 2/3
  let servingSize : Rat := 3 + 1/3
  fullServings containerAmount servingSize = 14 + 1/5 := by
sorry

#eval fullServings (47 + 2/3) (3 + 1/3)

end honey_servings_l3599_359983


namespace lee_ribbons_left_l3599_359991

/-- The number of ribbons Mr. Lee had left after giving away ribbons in the morning and afternoon -/
def ribbons_left (initial : ℕ) (morning : ℕ) (afternoon : ℕ) : ℕ :=
  initial - (morning + afternoon)

/-- Theorem stating that Mr. Lee had 8 ribbons left -/
theorem lee_ribbons_left : ribbons_left 38 14 16 = 8 := by
  sorry

end lee_ribbons_left_l3599_359991


namespace subset_implies_all_elements_in_l3599_359921

theorem subset_implies_all_elements_in : 
  ∀ (A B : Set α), A.Nonempty → B.Nonempty → A ⊆ B → ∀ x ∈ A, x ∈ B := by
  sorry

end subset_implies_all_elements_in_l3599_359921


namespace biking_jogging_swimming_rates_l3599_359905

theorem biking_jogging_swimming_rates : 
  ∃! (b j s : ℕ+), 
    (3 * b.val + 2 * j.val + 4 * s.val = 80) ∧ 
    (4 * b.val + 3 * j.val + 2 * s.val = 98) ∧ 
    (b.val^2 + j.val^2 + s.val^2 = 536) := by
  sorry

end biking_jogging_swimming_rates_l3599_359905


namespace geometric_series_ratio_l3599_359970

/-- For an infinite geometric series with first term a and common ratio r,
    if the sum of the series is 64 times the sum of the series with the first four terms removed,
    then r = 1/2 -/
theorem geometric_series_ratio (a r : ℝ) (h : r ≠ 1) :
  (a / (1 - r)) = 64 * (a * r^4 / (1 - r)) → r = 1/2 := by
  sorry

end geometric_series_ratio_l3599_359970


namespace sqrt_four_equals_two_l3599_359984

theorem sqrt_four_equals_two : Real.sqrt 4 = 2 := by
  sorry

end sqrt_four_equals_two_l3599_359984


namespace smallest_gcd_multiple_l3599_359948

theorem smallest_gcd_multiple (a b : ℕ+) (h : Nat.gcd a b = 10) :
  (∃ (x y : ℕ+), Nat.gcd x y = 10 ∧ Nat.gcd (8 * x) (14 * y) = 20) ∧
  ∀ (c d : ℕ+), Nat.gcd c d = 10 → Nat.gcd (8 * c) (14 * d) ≥ 20 :=
by sorry

end smallest_gcd_multiple_l3599_359948


namespace sum_of_ages_l3599_359935

/-- The sum of Eunji and Yuna's ages given their age difference -/
theorem sum_of_ages (eunji_age : ℕ) (age_difference : ℕ) : 
  eunji_age = 7 → age_difference = 5 → eunji_age + (eunji_age + age_difference) = 19 := by
  sorry

#check sum_of_ages

end sum_of_ages_l3599_359935


namespace prism_volume_l3599_359953

/-- A right triangular prism with given base area and lateral face areas has volume 12 -/
theorem prism_volume (base_area : ℝ) (lateral_area1 lateral_area2 lateral_area3 : ℝ) 
  (h_base : base_area = 4)
  (h_lateral1 : lateral_area1 = 9)
  (h_lateral2 : lateral_area2 = 10)
  (h_lateral3 : lateral_area3 = 17) :
  base_area * (lateral_area1 / base_area.sqrt) = 12 :=
by sorry

end prism_volume_l3599_359953


namespace booklet_cost_l3599_359944

theorem booklet_cost (b : ℝ) : 
  (10 * b < 15) → (12 * b > 17) → b = 1.42 := by
  sorry

end booklet_cost_l3599_359944


namespace last_digit_power_of_two_divisibility_l3599_359967

theorem last_digit_power_of_two_divisibility (k : ℕ) (N a A : ℕ) :
  k ≥ 3 →
  N = 2^k →
  a = N % 10 →
  A * 10 + a = N →
  6 ∣ a * A :=
by sorry

end last_digit_power_of_two_divisibility_l3599_359967


namespace relationship_between_x_squared_ax_bx_l3599_359986

theorem relationship_between_x_squared_ax_bx
  (x a b : ℝ)
  (h1 : x < a)
  (h2 : a < 0)
  (h3 : b > 0) :
  x^2 > a*x ∧ a*x > b*x :=
by sorry

end relationship_between_x_squared_ax_bx_l3599_359986


namespace greatest_common_multiple_9_15_under_100_l3599_359960

theorem greatest_common_multiple_9_15_under_100 : ∃ n : ℕ, n = 90 ∧ 
  (∀ m : ℕ, m < 100 → m % 9 = 0 → m % 15 = 0 → m ≤ n) :=
by sorry

end greatest_common_multiple_9_15_under_100_l3599_359960


namespace liu_data_correct_l3599_359929

/-- Represents the agricultural data for the Li and Liu families -/
structure FamilyData where
  li_land : ℕ
  li_yield : ℕ
  liu_land_diff : ℕ
  total_production : ℕ

/-- Calculates the Liu family's total production and yield difference -/
def calculate_liu_data (data : FamilyData) : ℕ × ℕ :=
  let liu_land := data.li_land - data.liu_land_diff
  let liu_production := data.total_production
  let liu_yield := liu_production / liu_land
  let li_yield := data.li_yield
  let yield_diff := liu_yield - li_yield
  (liu_production, yield_diff)

/-- Theorem stating the correctness of the calculation -/
theorem liu_data_correct (data : FamilyData) 
  (h1 : data.li_land = 100)
  (h2 : data.li_yield = 600)
  (h3 : data.liu_land_diff = 20)
  (h4 : data.total_production = data.li_land * data.li_yield) :
  calculate_liu_data data = (6000, 15) := by
  sorry

#eval calculate_liu_data ⟨100, 600, 20, 60000⟩

end liu_data_correct_l3599_359929


namespace smallest_fifth_prime_term_l3599_359940

/-- An arithmetic sequence of five prime numbers -/
structure PrimeArithmeticSequence :=
  (a : ℕ)  -- First term
  (d : ℕ)  -- Common difference
  (h1 : 0 < d)  -- Ensure the sequence is increasing
  (h2 : ∀ i : Fin 5, Prime (a + i.val * d))  -- All 5 terms are prime

/-- The fifth term of a prime arithmetic sequence -/
def fifthTerm (seq : PrimeArithmeticSequence) : ℕ :=
  seq.a + 4 * seq.d

theorem smallest_fifth_prime_term :
  (∃ seq : PrimeArithmeticSequence, fifthTerm seq = 29) ∧
  (∀ seq : PrimeArithmeticSequence, 29 ≤ fifthTerm seq) :=
sorry

end smallest_fifth_prime_term_l3599_359940


namespace total_cost_is_correct_l3599_359989

-- Define the number of DVDs and prices for each store
def store_a_dvds : ℕ := 8
def store_a_price : ℚ := 15
def store_b_dvds : ℕ := 12
def store_b_price : ℚ := 12
def online_dvds : ℕ := 5
def online_price : ℚ := 16.99

-- Define the discount percentage
def discount_percent : ℚ := 15

-- Define the total cost function
def total_cost (store_a_dvds store_b_dvds online_dvds : ℕ) 
               (store_a_price store_b_price online_price : ℚ) 
               (discount_percent : ℚ) : ℚ :=
  let physical_store_cost := store_a_dvds * store_a_price + store_b_dvds * store_b_price
  let online_store_cost := online_dvds * online_price
  let discount := physical_store_cost * (discount_percent / 100)
  (physical_store_cost - discount) + online_store_cost

-- Theorem statement
theorem total_cost_is_correct : 
  total_cost store_a_dvds store_b_dvds online_dvds 
             store_a_price store_b_price online_price 
             discount_percent = 309.35 := by
  sorry

end total_cost_is_correct_l3599_359989


namespace random_simulation_approximates_actual_probability_l3599_359980

/-- Random simulation method for estimating probabilities -/
def RandomSimulationMethod : Type := Unit

/-- Estimated probability from random simulation -/
def estimated_probability (method : RandomSimulationMethod) : ℝ := sorry

/-- Actual probability of the event -/
def actual_probability : ℝ := sorry

/-- Definition of approximation -/
def is_approximation (x y : ℝ) : Prop := sorry

theorem random_simulation_approximates_actual_probability 
  (method : RandomSimulationMethod) : 
  is_approximation (estimated_probability method) actual_probability := by
  sorry

end random_simulation_approximates_actual_probability_l3599_359980


namespace flu_spreads_indefinitely_flu_stops_spreading_l3599_359915

-- Define the population as a finite type
variable {Population : Type} [Finite Population]

-- Define the state of a person
inductive State
  | Healthy
  | Infected
  | Immune

-- Define the friendship relation
variable (friends : Population → Population → Prop)

-- Define the state of the population on a given day
variable (state : ℕ → Population → State)

-- Define the condition that each person visits their friends daily
axiom daily_visits : ∀ (d : ℕ) (p q : Population), friends p q → True

-- Define the condition that healthy people become ill after visiting sick friends
axiom infection_spread : ∀ (d : ℕ) (p : Population), 
  state d p = State.Healthy → 
  (∃ (q : Population), friends p q ∧ state d q = State.Infected) → 
  state (d + 1) p = State.Infected

-- Define the condition that illness lasts one day, followed by immunity
axiom illness_duration : ∀ (d : ℕ) (p : Population),
  state d p = State.Infected → state (d + 1) p = State.Immune

-- Define the condition that immunity lasts at least one day
axiom immunity_duration : ∀ (d : ℕ) (p : Population),
  state d p = State.Immune → state (d + 1) p ≠ State.Infected

-- Theorem 1: If some people have immunity initially, the flu can spread indefinitely
theorem flu_spreads_indefinitely (h : ∃ (p : Population), state 0 p = State.Immune) :
  ∀ (n : ℕ), ∃ (d : ℕ) (p : Population), d ≥ n ∧ state d p = State.Infected :=
sorry

-- Theorem 2: If no one has immunity initially, the flu will eventually stop spreading
theorem flu_stops_spreading (h : ∀ (p : Population), state 0 p ≠ State.Immune) :
  ∃ (n : ℕ), ∀ (d : ℕ) (p : Population), d ≥ n → state d p ≠ State.Infected :=
sorry

end flu_spreads_indefinitely_flu_stops_spreading_l3599_359915


namespace system_solution_in_first_quadrant_l3599_359954

theorem system_solution_in_first_quadrant (c : ℝ) :
  (∃ x y : ℝ, x > 0 ∧ y > 0 ∧ x - y = 2 ∧ c * x + y = 3) ↔ -1 < c ∧ c < 3/2 := by
  sorry

end system_solution_in_first_quadrant_l3599_359954


namespace clown_balloons_l3599_359975

/-- The number of balloons a clown had initially, given the number of boys and girls who bought balloons, and the number of balloons remaining. -/
def initial_balloons (boys girls remaining : ℕ) : ℕ :=
  boys + girls + remaining

/-- Theorem stating that the clown initially had 36 balloons -/
theorem clown_balloons : initial_balloons 3 12 21 = 36 := by
  sorry

end clown_balloons_l3599_359975


namespace expansion_properties_l3599_359990

/-- The binomial coefficient -/
def binomial (n k : ℕ) : ℕ := sorry

/-- The coefficient of the kth term in the expansion of (x + 1/(2√x))^n -/
def coefficient (n k : ℕ) : ℚ := (1 / 2^k : ℚ) * binomial n k

/-- The expansion of (x + 1/(2√x))^n has its first three coefficients in arithmetic sequence -/
def first_three_in_arithmetic_sequence (n : ℕ) : Prop :=
  coefficient n 0 + coefficient n 2 = 2 * coefficient n 1

/-- The kth term has the maximum coefficient in the expansion -/
def max_coefficient (n k : ℕ) : Prop :=
  ∀ i, i ≠ k → coefficient n k ≥ coefficient n i

theorem expansion_properties :
  ∃ n : ℕ,
    first_three_in_arithmetic_sequence n ∧
    max_coefficient n 2 ∧
    max_coefficient n 3 ∧
    ∀ k, k ≠ 2 ∧ k ≠ 3 → ¬(max_coefficient n k) :=
  sorry

end expansion_properties_l3599_359990


namespace share_ratio_problem_l3599_359930

theorem share_ratio_problem (total : ℕ) (john_share : ℕ) :
  total = 4800 →
  john_share = 1600 →
  ∃ (jose_share binoy_share : ℕ),
    total = john_share + jose_share + binoy_share ∧
    2 * jose_share = 4 * john_share ∧
    3 * jose_share = 6 * john_share ∧
    binoy_share = 3 * john_share :=
by sorry

end share_ratio_problem_l3599_359930


namespace cosine_rational_values_l3599_359979

theorem cosine_rational_values (α : ℚ) (h : ∃ (r : ℚ), r = Real.cos (α * Real.pi)) :
  Real.cos (α * Real.pi) = 0 ∨
  Real.cos (α * Real.pi) = 1 ∨
  Real.cos (α * Real.pi) = -1 ∨
  Real.cos (α * Real.pi) = (1/2) ∨
  Real.cos (α * Real.pi) = -(1/2) :=
by sorry

end cosine_rational_values_l3599_359979


namespace perimeter_of_modified_square_l3599_359962

/-- The perimeter of the shape ABFCDE formed by cutting a right triangle from a square and
    repositioning it on the left side of the square. -/
theorem perimeter_of_modified_square (
  square_perimeter : ℝ)
  (triangle_leg : ℝ)
  (h1 : square_perimeter = 48)
  (h2 : triangle_leg = 12) : ℝ :=
by
  -- The perimeter of the new shape ABFCDE is 60 inches
  sorry

#check perimeter_of_modified_square

end perimeter_of_modified_square_l3599_359962


namespace min_distance_circle_to_line_l3599_359988

theorem min_distance_circle_to_line : 
  let circle := {(x, y) : ℝ × ℝ | x^2 + y^2 = 1}
  let line := {(x, y) : ℝ × ℝ | 3*x + 4*y + 15 = 0}
  ∃ d : ℝ, d = 2 ∧ 
    ∀ p ∈ circle, ∀ q ∈ line, 
      Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2) ≥ d ∧
      ∃ p' ∈ circle, ∃ q' ∈ line, 
        Real.sqrt ((p'.1 - q'.1)^2 + (p'.2 - q'.2)^2) = d :=
by
  sorry


end min_distance_circle_to_line_l3599_359988


namespace system_solution_l3599_359919

theorem system_solution (a b c x y z : ℝ) : 
  (x + a * y + a^2 * z = a^3) →
  (x + b * y + b^2 * z = b^3) →
  (x + c * y + c^2 * z = c^3) →
  (x = a * b * c ∧ y = -(a * b + b * c + c * a) ∧ z = a + b + c) :=
by sorry

end system_solution_l3599_359919


namespace point_on_line_l3599_359910

theorem point_on_line (n : ℕ) (P : ℕ → ℤ × ℤ) : 
  (P 0 = (0, 1)) →
  (∀ k : ℕ, k ≥ 1 → k ≤ n → (P k).1 - (P (k-1)).1 = 1) →
  (∀ k : ℕ, k ≥ 1 → k ≤ n → (P k).2 - (P (k-1)).2 = 2) →
  (P n).1 = n →
  (P n).2 = 2*n + 1 →
  (P n).2 = 3*(P n).1 - 8 →
  n = 9 := by sorry

end point_on_line_l3599_359910


namespace carter_drum_sticks_l3599_359981

/-- The number of drum stick sets Carter uses per show -/
def sticks_used_per_show : ℕ := 8

/-- The number of drum stick sets Carter tosses to the audience after each show -/
def sticks_tossed_per_show : ℕ := 10

/-- The number of nights Carter performs -/
def number_of_shows : ℕ := 45

/-- The total number of drum stick sets Carter goes through -/
def total_sticks : ℕ := (sticks_used_per_show + sticks_tossed_per_show) * number_of_shows

theorem carter_drum_sticks :
  total_sticks = 810 := by sorry

end carter_drum_sticks_l3599_359981


namespace pizza_theorem_l3599_359908

def pizza_eaten (n : ℕ) : ℚ :=
  if n = 0 then 0
  else if n = 1 then 1/3
  else 1/3 + (1 - 1/3) * (1 - (1/2)^(n-1))

theorem pizza_theorem : pizza_eaten 4 = 11/12 := by
  sorry

end pizza_theorem_l3599_359908


namespace value_of_a_minus_b_l3599_359946

theorem value_of_a_minus_b (a b : ℝ) (h1 : 2 * a - b = 5) (h2 : a - 2 * b = 4) : a - b = 3 := by
  sorry

end value_of_a_minus_b_l3599_359946


namespace yearly_music_expenditure_l3599_359945

def hours_per_month : ℕ := 20
def minutes_per_song : ℕ := 3
def price_per_song : ℚ := 1/2
def months_per_year : ℕ := 12

def yearly_music_cost : ℚ :=
  (hours_per_month * 60 / minutes_per_song) * price_per_song * months_per_year

theorem yearly_music_expenditure :
  yearly_music_cost = 2400 := by
  sorry

end yearly_music_expenditure_l3599_359945


namespace equation_solution_l3599_359950

theorem equation_solution : ∃ x : ℝ, (10 - 2 * x = 14) ∧ (x = -2) := by
  sorry

end equation_solution_l3599_359950


namespace solution_equality_l3599_359994

theorem solution_equality (x₁ x₂ x₃ x₄ x₅ : ℝ) 
  (h_pos : x₁ > 0 ∧ x₂ > 0 ∧ x₃ > 0 ∧ x₄ > 0 ∧ x₅ > 0)
  (h₁ : (x₁^2 - x₃*x₅)*(x₂^2 - x₃*x₅) ≤ 0)
  (h₂ : (x₂^2 - x₄*x₁)*(x₃^2 - x₄*x₁) ≤ 0)
  (h₃ : (x₃^2 - x₅*x₂)*(x₄^2 - x₅*x₂) ≤ 0)
  (h₄ : (x₄^2 - x₁*x₃)*(x₅^2 - x₁*x₃) ≤ 0)
  (h₅ : (x₅^2 - x₂*x₄)*(x₁^2 - x₂*x₄) ≤ 0) :
  x₁ = x₂ ∧ x₂ = x₃ ∧ x₃ = x₄ ∧ x₄ = x₅ :=
by sorry

end solution_equality_l3599_359994


namespace third_number_in_first_set_l3599_359904

theorem third_number_in_first_set (x : ℝ) : 
  (20 + 40 + x) / 3 = (10 + 70 + 13) / 3 + 9 → x = 60 := by
  sorry

end third_number_in_first_set_l3599_359904
