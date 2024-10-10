import Mathlib

namespace mans_speed_with_current_l2317_231785

/-- 
Given a man's speed against a current and the speed of the current,
this theorem proves the man's speed with the current.
-/
theorem mans_speed_with_current 
  (speed_against_current : ℝ) 
  (current_speed : ℝ) 
  (h1 : speed_against_current = 10) 
  (h2 : current_speed = 2.5) : 
  speed_against_current + 2 * current_speed = 15 :=
by
  sorry

#check mans_speed_with_current

end mans_speed_with_current_l2317_231785


namespace sum_of_x_and_y_equals_twice_C_x_plus_y_equals_76_l2317_231730

-- Define the angles
def angle_A : ℝ := 34
def angle_B : ℝ := 80
def angle_C : ℝ := 38

-- Define x and y as real numbers (representing angle measures)
variable (x y : ℝ)

-- State the theorem
theorem sum_of_x_and_y_equals_twice_C :
  x + y = 2 * angle_C := by sorry

-- Prove that x + y equals 76
theorem x_plus_y_equals_76 :
  x + y = 76 := by sorry

end sum_of_x_and_y_equals_twice_C_x_plus_y_equals_76_l2317_231730


namespace dinner_lunch_difference_l2317_231723

-- Define the number of cakes served during lunch
def lunch_cakes : ℕ := 6

-- Define the number of cakes served during dinner
def dinner_cakes : ℕ := 9

-- Theorem stating the difference between dinner and lunch cakes
theorem dinner_lunch_difference : dinner_cakes - lunch_cakes = 3 := by
  sorry

end dinner_lunch_difference_l2317_231723


namespace hexagon_area_l2317_231759

-- Define the hexagon points
def hexagon_points : List (ℤ × ℤ) := [(0, 0), (1, 2), (2, 3), (4, 2), (3, 0), (0, 0)]

-- Function to calculate the area of a polygon given its vertices
def polygon_area (points : List (ℤ × ℤ)) : ℚ :=
  sorry

-- Theorem statement
theorem hexagon_area : polygon_area hexagon_points = 4 := by
  sorry

end hexagon_area_l2317_231759


namespace tan_identities_l2317_231771

theorem tan_identities (α : Real) (h : Real.tan α = 2) :
  (Real.tan (α + π/4) = -3) ∧
  ((Real.sin (2*α)) / (Real.sin α ^ 2 + Real.sin α * Real.cos α - Real.cos (2*α) - 1) = 1) := by
  sorry

end tan_identities_l2317_231771


namespace complex_exp_eleven_pi_over_two_equals_neg_i_l2317_231744

-- Define the complex exponential function
noncomputable def cexp (z : ℂ) : ℂ := Real.exp z.re * (Complex.cos z.im + Complex.I * Complex.sin z.im)

-- State the theorem
theorem complex_exp_eleven_pi_over_two_equals_neg_i :
  cexp (11 * Real.pi / 2 * Complex.I) = -Complex.I :=
sorry

end complex_exp_eleven_pi_over_two_equals_neg_i_l2317_231744


namespace shaded_area_of_intersecting_diameters_l2317_231721

theorem shaded_area_of_intersecting_diameters (r : ℝ) (θ : ℝ) : 
  r = 6 → θ = π / 3 → 2 * (θ / (2 * π)) * (π * r^2) = 12 * π := by
  sorry

end shaded_area_of_intersecting_diameters_l2317_231721


namespace multiply_72516_by_9999_l2317_231767

theorem multiply_72516_by_9999 : 72516 * 9999 = 724787484 := by
  sorry

end multiply_72516_by_9999_l2317_231767


namespace quadratic_inequality_properties_l2317_231754

theorem quadratic_inequality_properties (a b c : ℝ) : 
  (∀ x : ℝ, ax^2 + b*x + c > 0 ↔ -3 < x ∧ x < 2) → 
  (a < 0 ∧ a - b + c > 0) :=
by sorry

end quadratic_inequality_properties_l2317_231754


namespace units_digit_problem_l2317_231718

theorem units_digit_problem : ∃ n : ℕ, (7 * 27 * 1977 + 9) - 7^3 ≡ 9 [ZMOD 10] ∧ n < 10 := by
  sorry

end units_digit_problem_l2317_231718


namespace sum_of_hidden_numbers_l2317_231741

/-- Represents a standard six-sided die with faces numbered 1 through 6 -/
def Die := Fin 6

/-- The sum of all numbers on a standard die -/
def dieTotalSum : ℕ := 21

/-- The visible numbers on the seven sides of the stacked dice -/
def visibleNumbers : List ℕ := [2, 3, 4, 4, 5, 5, 6]

/-- The number of dice stacked -/
def numberOfDice : ℕ := 3

/-- Theorem stating that the sum of numbers not visible on the stacked dice is 34 -/
theorem sum_of_hidden_numbers :
  (numberOfDice * dieTotalSum) - (visibleNumbers.sum) = 34 := by
  sorry

end sum_of_hidden_numbers_l2317_231741


namespace trig_identity_l2317_231743

theorem trig_identity : 
  let tan30 := Real.sqrt 3 / 3
  let tan60 := Real.sqrt 3
  let cos30 := Real.sqrt 3 / 2
  let sin60 := Real.sqrt 3 / 2
  let cot45 := 1
  3 * tan30^2 + tan60^2 - cos30 * sin60 * cot45 = 7/4 := by
sorry

end trig_identity_l2317_231743


namespace inequality_holds_iff_k_equals_6020_l2317_231796

theorem inequality_holds_iff_k_equals_6020 :
  ∃ (k : ℝ), k > 0 ∧
  (∀ (a b c : ℝ), a > 0 → b > 0 → c > 0 →
    (a / (c + k * b) + b / (a + k * c) + c / (b + k * a) ≥ 1 / 2007)) ∧
  k = 6020 := by
  sorry

end inequality_holds_iff_k_equals_6020_l2317_231796


namespace larger_number_problem_l2317_231739

theorem larger_number_problem (L S : ℕ) 
  (h1 : L - S = 1395)
  (h2 : L = 6 * S + 15) :
  L = 1671 := by
sorry

end larger_number_problem_l2317_231739


namespace simplify_radical_product_l2317_231761

theorem simplify_radical_product (x : ℝ) (hx : x ≥ 0) :
  Real.sqrt (12 * x) * Real.sqrt (18 * x) * Real.sqrt (27 * x) = 54 * x * Real.sqrt x :=
by sorry

end simplify_radical_product_l2317_231761


namespace common_root_cubics_theorem_l2317_231712

/-- Two cubic equations with two common roots -/
structure CommonRootCubics where
  A : ℝ
  B : ℝ
  C : ℝ
  root1 : ℝ
  root2 : ℝ
  eq1_holds : ∀ x : ℝ, x^3 + A*x^2 + 20*x + C = 0 ↔ x = root1 ∨ x = root2 ∨ x = -A - root1 - root2
  eq2_holds : ∀ x : ℝ, x^3 + B*x^2 + 100 = 0 ↔ x = root1 ∨ x = root2 ∨ x = -B - root1 - root2

theorem common_root_cubics_theorem (cubics : CommonRootCubics) :
  cubics.C = 100 ∧ cubics.root1 * cubics.root2 = 5 * Real.rpow 5 (1/3) := by sorry

end common_root_cubics_theorem_l2317_231712


namespace problem_solution_l2317_231784

/-- Given the conditions of the problem, prove that x · z = 4.5 -/
theorem problem_solution :
  ∀ x y z : ℝ,
  (∃ x₀ y₀ z₀ : ℝ, x₀ = 2*y₀ ∧ z₀ = x₀ ∧ x₀*y₀ = y₀*z₀) →  -- Initial condition
  z = x/2 →
  x*y = y^2 →
  y = 3 →
  x*z = 4.5 := by
sorry

end problem_solution_l2317_231784


namespace poor_people_distribution_l2317_231727

theorem poor_people_distribution (x : ℕ) : 
  (120 / (x - 10) - 120 / x = 120 / x - 120 / (x + 20)) → x = 40 := by
  sorry

end poor_people_distribution_l2317_231727


namespace fraction_inequality_l2317_231735

theorem fraction_inequality (x : ℝ) : 
  x ∈ Set.Icc (-2 : ℝ) 2 → 
  (6 * x + 1 > 7 - 4 * x ↔ 3 / 5 < x ∧ x ≤ 2) := by
  sorry

end fraction_inequality_l2317_231735


namespace largest_n_unique_k_l2317_231750

theorem largest_n_unique_k : ∃ (n : ℕ), n > 0 ∧ n = 112 ∧
  (∃! (k : ℤ), (8 : ℚ)/15 < (n : ℚ)/(n + k) ∧ (n : ℚ)/(n + k) < 7/13) ∧
  (∀ (m : ℕ), m > n → ¬∃! (k : ℤ), (8 : ℚ)/15 < (m : ℚ)/(m + k) ∧ (m : ℚ)/(m + k) < 7/13) :=
by sorry

end largest_n_unique_k_l2317_231750


namespace exists_permutation_satisfying_average_condition_l2317_231736

/-- A permutation of numbers from 1 to n satisfies the average condition if for any three indices
    i < j < k, the average of the i-th and k-th elements is not equal to the j-th element. -/
def satisfies_average_condition (n : ℕ) (perm : Fin n → ℕ) : Prop :=
  ∀ i j k : Fin n, i < j → j < k →
    (perm i + perm k) / 2 ≠ perm j

/-- For any positive integer n, there exists a permutation of the numbers 1 to n
    that satisfies the average condition. -/
theorem exists_permutation_satisfying_average_condition (n : ℕ+) :
  ∃ perm : Fin n → ℕ, Function.Injective perm ∧ Set.range perm = Finset.range n ∧
    satisfies_average_condition n perm :=
sorry

end exists_permutation_satisfying_average_condition_l2317_231736


namespace sin_cos_sum_13_17_l2317_231715

theorem sin_cos_sum_13_17 :
  Real.sin (13 * π / 180) * Real.cos (17 * π / 180) +
  Real.cos (13 * π / 180) * Real.sin (17 * π / 180) = 1 / 2 := by
  sorry

end sin_cos_sum_13_17_l2317_231715


namespace solve_equation_l2317_231704

-- Define the complex number i
def i : ℂ := Complex.I

-- Define the equation
def equation (a : ℝ) : Prop := (a - i : ℂ) ^ 2 = 2 * i

-- Theorem statement
theorem solve_equation : ∃! (a : ℝ), equation a :=
  sorry

end solve_equation_l2317_231704


namespace odd_function_inverse_range_l2317_231753

/-- An odd function f defined on ℝ with specific properties -/
def OddFunction (f : ℝ → ℝ) : Prop :=
  (∀ x, f (-x) = -f x) ∧ 
  (∃ a b, 0 < a ∧ a < 1 ∧ ∀ x > 0, f x = a^x + b)

/-- Theorem stating the range of b for which f has an inverse -/
theorem odd_function_inverse_range (f : ℝ → ℝ) (h : OddFunction f) 
  (h_inv : Function.Injective f) : 
  ∃ a b, (0 < a ∧ a < 1) ∧ (∀ x > 0, f x = a^x + b) ∧ (b ≤ -1 ∨ b ≥ 0) := by
  sorry

end odd_function_inverse_range_l2317_231753


namespace smallest_two_digit_number_with_conditions_l2317_231776

theorem smallest_two_digit_number_with_conditions : ∃ n : ℕ,
  (n ≥ 10 ∧ n < 100) ∧  -- two-digit number
  (n % 3 = 0) ∧         -- divisible by 3
  (n % 4 = 0) ∧         -- divisible by 4
  (n % 5 = 4) ∧         -- remainder 4 when divided by 5
  (∀ m : ℕ, (m ≥ 10 ∧ m < 100) ∧ (m % 3 = 0) ∧ (m % 4 = 0) ∧ (m % 5 = 4) → n ≤ m) ∧
  n = 24 :=
by
  sorry

#check smallest_two_digit_number_with_conditions

end smallest_two_digit_number_with_conditions_l2317_231776


namespace curve_C_and_m_range_l2317_231713

/-- The curve C defined by the arithmetic sequence property -/
def C : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | let (x, y) := p
               Real.sqrt ((x - 1)^2 + y^2) + Real.sqrt ((x + 1)^2 + y^2) = 4}

/-- The line l₁ that intersects C -/
def l₁ (m : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | let (x, y) := p
               x - y + m = 0}

/-- Predicate for the obtuse angle condition -/
def isObtuseAngle (M N : ℝ × ℝ) : Prop :=
  let (x₁, y₁) := M
  let (x₂, y₂) := N
  x₁ * x₂ + y₁ * y₂ < 0

/-- The main theorem stating the properties of C and the range of m -/
theorem curve_C_and_m_range :
  ∃ (a b : ℝ),
    (a = 2 ∧ b = Real.sqrt 3) ∧
    (C = {p : ℝ × ℝ | let (x, y) := p
                      x^2 / a^2 + y^2 / b^2 = 1}) ∧
    (∀ m : ℝ,
      (∃ M N : ℝ × ℝ, M ∈ C ∧ N ∈ C ∧ M ∈ l₁ m ∧ N ∈ l₁ m ∧ M ≠ N ∧ isObtuseAngle M N) ↔
      -2 * Real.sqrt 42 / 7 < m ∧ m < 2 * Real.sqrt 42 / 7) := by
  sorry

end curve_C_and_m_range_l2317_231713


namespace chocolate_distribution_l2317_231793

def is_valid_distribution (n m : ℕ) : Prop :=
  n ≤ m ∨ (m < n ∧ m ∣ (n - m))

def possible_n_for_m_9 : Set ℕ :=
  {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 18}

theorem chocolate_distribution (n m : ℕ) :
  (m = 9 → n ∈ possible_n_for_m_9) ↔ is_valid_distribution n m :=
sorry

end chocolate_distribution_l2317_231793


namespace vector_difference_magnitude_l2317_231770

/-- Given two vectors a and b in a 2D plane with an angle of 120° between them,
    |a| = 1, and |b| = 3, prove that |a - b| = √13 -/
theorem vector_difference_magnitude (a b : ℝ × ℝ) :
  (a.fst * b.fst + a.snd * b.snd = -3/2) →  -- Dot product for 120° angle
  (a.fst^2 + a.snd^2 = 1) →  -- |a| = 1
  (b.fst^2 + b.snd^2 = 9) →  -- |b| = 3
  ((a.fst - b.fst)^2 + (a.snd - b.snd)^2 = 13) :=
by sorry

end vector_difference_magnitude_l2317_231770


namespace number_difference_l2317_231711

theorem number_difference (x y : ℝ) (h1 : x + y = 147) (h2 : x - 0.375 * y = 4) (h3 : x ≥ y) : x - 0.375 * y = 4 := by
  sorry

end number_difference_l2317_231711


namespace absolute_value_sqrt_three_l2317_231707

theorem absolute_value_sqrt_three : 
  |1 - Real.sqrt 3| - (Real.sqrt 3 - 1)^0 = Real.sqrt 3 - 2 := by sorry

end absolute_value_sqrt_three_l2317_231707


namespace balloon_cost_theorem_l2317_231792

/-- Represents the cost of balloons for a person -/
structure BalloonCost where
  count : ℕ
  price : ℚ

/-- Calculates the total cost for a person's balloons -/
def totalCost (bc : BalloonCost) : ℚ :=
  bc.count * bc.price

theorem balloon_cost_theorem (fred sam dan : BalloonCost)
  (h_fred : fred = ⟨10, 1⟩)
  (h_sam : sam = ⟨46, (3/2)⟩)
  (h_dan : dan = ⟨16, (3/4)⟩) :
  totalCost fred + totalCost sam + totalCost dan = 91 := by
  sorry

end balloon_cost_theorem_l2317_231792


namespace parabola_translation_l2317_231775

/-- Represents a parabola in the xy-plane -/
structure Parabola where
  a : ℝ
  h : ℝ
  k : ℝ

/-- The equation of a parabola in vertex form -/
def Parabola.equation (p : Parabola) (x y : ℝ) : Prop :=
  y = p.a * (x - p.h)^2 + p.k

/-- Vertical translation of a parabola -/
def verticalTranslate (p : Parabola) (dy : ℝ) : Parabola :=
  { a := p.a, h := p.h, k := p.k + dy }

theorem parabola_translation (x y : ℝ) :
  let p := Parabola.mk 3 0 0
  let p_translated := verticalTranslate p 3
  Parabola.equation p_translated x y ↔ y = 3 * x^2 + 3 := by sorry

end parabola_translation_l2317_231775


namespace football_team_progress_l2317_231706

theorem football_team_progress (yards_lost yards_gained : ℤ) : 
  yards_lost = 5 → yards_gained = 9 → yards_gained - yards_lost = 4 := by
  sorry

end football_team_progress_l2317_231706


namespace counterexample_exists_negative_four_is_counterexample_l2317_231758

theorem counterexample_exists : ∃ a : ℝ, a < 3 ∧ a^2 ≥ 9 :=
  by
  use -4
  constructor
  · -- Prove -4 < 3
    sorry
  · -- Prove (-4)^2 ≥ 9
    sorry

theorem negative_four_is_counterexample : -4 < 3 ∧ (-4)^2 ≥ 9 :=
  by
  constructor
  · -- Prove -4 < 3
    sorry
  · -- Prove (-4)^2 ≥ 9
    sorry

end counterexample_exists_negative_four_is_counterexample_l2317_231758


namespace man_speed_against_current_is_10_l2317_231760

/-- Given a man's speed with the current and the speed of the current, 
    calculate the man's speed against the current. -/
def man_speed_against_current (speed_with_current : ℝ) (current_speed : ℝ) : ℝ :=
  speed_with_current - 2 * current_speed

/-- Theorem stating that given the specific conditions, 
    the man's speed against the current is 10 km/hr. -/
theorem man_speed_against_current_is_10 :
  man_speed_against_current 15 2.5 = 10 := by
  sorry

#eval man_speed_against_current 15 2.5

end man_speed_against_current_is_10_l2317_231760


namespace negative_solution_existence_l2317_231733

/-- The inequality x^2 < 4 - |x - a| has at least one negative solution if and only if a ∈ [-17/4, 4). -/
theorem negative_solution_existence (a : ℝ) :
  (∃ x : ℝ, x < 0 ∧ x^2 < 4 - |x - a|) ↔ -17/4 ≤ a ∧ a < 4 := by
  sorry

end negative_solution_existence_l2317_231733


namespace square_center_sum_l2317_231746

-- Define the square ABCD
structure Square where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ
  D : ℝ × ℝ

-- Define the conditions
def square_conditions (s : Square) : Prop :=
  -- Square is in the first quadrant
  s.A.1 ≥ 0 ∧ s.A.2 ≥ 0 ∧
  s.B.1 ≥ 0 ∧ s.B.2 ≥ 0 ∧
  s.C.1 ≥ 0 ∧ s.C.2 ≥ 0 ∧
  s.D.1 ≥ 0 ∧ s.D.2 ≥ 0 ∧
  -- Points on the lines
  (∃ t : ℝ, t ∈ Set.Icc 0 1 ∧ (4, 0) = s.A + t • (s.B - s.A)) ∧
  (∃ t : ℝ, t ∈ Set.Icc 0 1 ∧ (7, 0) = s.C + t • (s.D - s.C)) ∧
  (∃ t : ℝ, t ∈ Set.Icc 0 1 ∧ (9, 0) = s.B + t • (s.C - s.B)) ∧
  (∃ t : ℝ, t ∈ Set.Icc 0 1 ∧ (15, 0) = s.D + t • (s.A - s.D))

-- Theorem statement
theorem square_center_sum (s : Square) (h : square_conditions s) :
  (s.A.1 + s.B.1 + s.C.1 + s.D.1) / 4 + (s.A.2 + s.B.2 + s.C.2 + s.D.2) / 4 = 27 / 4 :=
by sorry

end square_center_sum_l2317_231746


namespace wrong_number_calculation_l2317_231782

theorem wrong_number_calculation (n : ℕ) (initial_avg correct_avg actual_num : ℝ) :
  n = 10 ∧ 
  initial_avg = 14 ∧ 
  correct_avg = 15 ∧ 
  actual_num = 36 →
  ∃ wrong_num : ℝ, 
    n * correct_avg - n * initial_avg = actual_num - wrong_num ∧ 
    wrong_num = 26 := by
  sorry

end wrong_number_calculation_l2317_231782


namespace mutually_exclusive_not_opposite_l2317_231766

-- Define the total number of balls and the number of each color
def total_balls : ℕ := 7
def red_balls : ℕ := 5
def black_balls : ℕ := 2

-- Define the number of balls drawn
def drawn_balls : ℕ := 3

-- Define the events
def exactly_one_black (outcome : Finset ℕ) : Prop :=
  outcome.card = drawn_balls ∧ (outcome.filter (λ x => x > red_balls)).card = 1

def exactly_two_black (outcome : Finset ℕ) : Prop :=
  outcome.card = drawn_balls ∧ (outcome.filter (λ x => x > red_balls)).card = 2

-- Theorem statement
theorem mutually_exclusive_not_opposite :
  (∃ outcome, exactly_one_black outcome ∧ exactly_two_black outcome = False) ∧
  (∃ outcome, ¬(exactly_one_black outcome ∨ exactly_two_black outcome)) := by
  sorry

end mutually_exclusive_not_opposite_l2317_231766


namespace perfect_square_existence_l2317_231755

theorem perfect_square_existence (k : ℕ+) :
  ∃ (n m : ℕ+), n * 2^(k : ℕ) - 7 = m^2 := by
  sorry

end perfect_square_existence_l2317_231755


namespace cubic_roots_product_l2317_231798

theorem cubic_roots_product (r s t : ℝ) : 
  (r^3 - 20*r^2 + 18*r - 7 = 0) ∧ 
  (s^3 - 20*s^2 + 18*s - 7 = 0) ∧ 
  (t^3 - 20*t^2 + 18*t - 7 = 0) →
  (1 + r) * (1 + s) * (1 + t) = 46 := by
sorry


end cubic_roots_product_l2317_231798


namespace project_time_allocation_l2317_231756

theorem project_time_allocation (worker1 worker2 worker3 : ℚ) 
  (h1 : worker1 = 1/2)
  (h2 : worker3 = 1/3)
  (h_total : worker1 + worker2 + worker3 = 1) :
  worker2 = 1/6 := by
sorry

end project_time_allocation_l2317_231756


namespace julias_running_time_l2317_231780

/-- Julia's running time problem -/
theorem julias_running_time 
  (normal_mile_time : ℝ) 
  (extra_time_for_five_miles : ℝ) 
  (h1 : normal_mile_time = 10) 
  (h2 : extra_time_for_five_miles = 15) : 
  (5 * normal_mile_time + extra_time_for_five_miles) / 5 = 13 := by
  sorry

end julias_running_time_l2317_231780


namespace expected_replanted_seeds_l2317_231781

/-- The expected number of replanted seeds when sowing 1000 seeds with a 0.9 germination probability -/
theorem expected_replanted_seeds :
  let germination_prob : ℝ := 0.9
  let total_seeds : ℕ := 1000
  let replant_per_fail : ℕ := 2
  let expected_non_germinating : ℝ := total_seeds * (1 - germination_prob)
  expected_non_germinating * replant_per_fail = 200 := by sorry

end expected_replanted_seeds_l2317_231781


namespace x_range_for_quadratic_inequality_l2317_231752

theorem x_range_for_quadratic_inequality :
  (∀ m : ℝ, |m| ≤ 2 → ∀ x : ℝ, m * x^2 - 2 * x - m + 1 < 0) →
  ∀ x : ℝ, (-1 + Real.sqrt 7) / 2 < x ∧ x < (1 + Real.sqrt 3) / 2 :=
by sorry

end x_range_for_quadratic_inequality_l2317_231752


namespace quadratic_function_properties_l2317_231708

/-- Given a quadratic function f(x) = ax² + bx + c with specific properties,
    prove statements about its coefficients and roots. -/
theorem quadratic_function_properties (a b c : ℝ) (f : ℝ → ℝ) 
    (h1 : ∀ x, f x = a * x^2 + b * x + c)
    (h2 : f 1 = -a / 2)
    (h3 : 3 * a > 2 * c)
    (h4 : 2 * c > 2 * b) : 
  (a > 0 ∧ -3 < b / a ∧ b / a < -3 / 4) ∧ 
  (∃ x : ℝ, 0 < x ∧ x < 2 ∧ f x = 0) ∧
  (∀ x₁ x₂ : ℝ, f x₁ = 0 → f x₂ = 0 → x₁ ≠ x₂ → 
    Real.sqrt 2 ≤ |x₁ - x₂| ∧ |x₁ - x₂| < Real.sqrt 57 / 4) :=
by sorry

end quadratic_function_properties_l2317_231708


namespace reciprocal_equation_solution_l2317_231709

theorem reciprocal_equation_solution (x : ℝ) :
  (2 - (1 / (1 - x)) = 1 / (1 - x)) → x = 0 := by
  sorry

end reciprocal_equation_solution_l2317_231709


namespace max_value_2a_plus_b_l2317_231742

theorem max_value_2a_plus_b (a b : ℝ) 
  (h1 : 4 * a + 3 * b ≤ 10) 
  (h2 : 3 * a + 6 * b ≤ 12) : 
  2 * a + b ≤ 5 ∧ ∃ (a' b' : ℝ), 4 * a' + 3 * b' ≤ 10 ∧ 3 * a' + 6 * b' ≤ 12 ∧ 2 * a' + b' = 5 := by
  sorry

end max_value_2a_plus_b_l2317_231742


namespace circles_are_separate_l2317_231762

-- Define the circles
def C₁ (x y : ℝ) : Prop := (x - 1)^2 + y^2 = 1
def C₂ (x y : ℝ) : Prop := (x + 3)^2 + (y - 2)^2 = 4

-- Define the centers and radii
def center₁ : ℝ × ℝ := (1, 0)
def center₂ : ℝ × ℝ := (-3, 2)
def radius₁ : ℝ := 1
def radius₂ : ℝ := 2

-- Theorem statement
theorem circles_are_separate :
  let d := Real.sqrt ((center₁.1 - center₂.1)^2 + (center₁.2 - center₂.2)^2)
  d > radius₁ + radius₂ :=
by sorry

end circles_are_separate_l2317_231762


namespace units_digit_of_4539_pow_201_l2317_231747

theorem units_digit_of_4539_pow_201 : (4539^201) % 10 = 9 := by sorry

end units_digit_of_4539_pow_201_l2317_231747


namespace decrement_value_proof_l2317_231725

theorem decrement_value_proof (n : ℕ) (original_mean updated_mean : ℚ) 
  (h1 : n = 50)
  (h2 : original_mean = 200)
  (h3 : updated_mean = 153) :
  (n : ℚ) * original_mean - n * updated_mean = n * 47 := by
  sorry

end decrement_value_proof_l2317_231725


namespace newspaper_recycling_profit_l2317_231732

/-- Calculates the amount of money made from recycling stolen newspapers over a period of time. -/
def recycling_profit (weekday_paper_weight : ℚ) (sunday_paper_weight : ℚ) 
  (papers_per_day : ℕ) (num_weeks : ℕ) (recycling_rate : ℚ) : ℚ :=
  let weekly_weight := (6 * weekday_paper_weight + sunday_paper_weight) * papers_per_day
  let total_weight := weekly_weight * num_weeks
  let total_tons := total_weight / 2000
  total_tons * recycling_rate

/-- Theorem stating that under the given conditions, the profit from recycling stolen newspapers is $100. -/
theorem newspaper_recycling_profit :
  recycling_profit (8/16) (16/16) 250 10 20 = 100 := by
  sorry

end newspaper_recycling_profit_l2317_231732


namespace cube_root_function_l2317_231764

/-- Given a function y = kx^(1/3) where y = 4√3 when x = 64, 
    prove that y = 2√3 when x = 8 -/
theorem cube_root_function (k : ℝ) :
  (∀ x : ℝ, x > 0 → k * x^(1/3) = 4 * Real.sqrt 3 → x = 64) →
  k * 8^(1/3) = 2 * Real.sqrt 3 :=
by sorry

end cube_root_function_l2317_231764


namespace power_function_domain_and_oddness_l2317_231769

def is_odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

def has_real_domain (f : ℝ → ℝ) : Prop :=
  ∀ x, ∃ y, f x = y

theorem power_function_domain_and_oddness (a : ℝ) :
  a ∈ ({-1, 0, 1/2, 1, 2, 3} : Set ℝ) →
  (has_real_domain (fun x ↦ x^a) ∧ is_odd_function (fun x ↦ x^a)) ↔ (a = 1 ∨ a = 3) :=
sorry

end power_function_domain_and_oddness_l2317_231769


namespace zeros_product_bound_l2317_231717

/-- Given a > e and f(x) = e^x - a((ln x + x)/x) has two distinct zeros, prove x₁x₂ > e^(2-x₁-x₂) -/
theorem zeros_product_bound (a : ℝ) (x₁ x₂ : ℝ) 
  (ha : a > Real.exp 1)
  (hf : ∀ x : ℝ, x > 0 → Real.exp x - a * ((Real.log x + x) / x) = 0 ↔ x = x₁ ∨ x = x₂)
  (hx : x₁ ≠ x₂) :
  x₁ * x₂ > Real.exp (2 - x₁ - x₂) :=
by sorry

end zeros_product_bound_l2317_231717


namespace speech_contest_probability_l2317_231799

/-- Represents the number of participants in the speech contest -/
def total_participants : ℕ := 10

/-- Represents the number of participants from Class 1 -/
def class1_participants : ℕ := 3

/-- Represents the number of participants from Class 2 -/
def class2_participants : ℕ := 2

/-- Represents the number of participants from other classes -/
def other_participants : ℕ := 5

/-- Calculates the probability of Class 1 students being consecutive and Class 2 students not being consecutive -/
def probability_class1_consecutive_class2_not : ℚ :=
  1 / 20

/-- Theorem stating the probability of the given event -/
theorem speech_contest_probability :
  probability_class1_consecutive_class2_not = 1 / 20 :=
by
  sorry

end speech_contest_probability_l2317_231799


namespace sequence_eleventh_term_l2317_231734

/-- Given a sequence a₁, a₂, ..., where a₁ = 3 and aₙ₊₁ - aₙ = n for n ≥ 1,
    prove that a₁₁ = 58. -/
theorem sequence_eleventh_term (a : ℕ → ℕ) 
  (h1 : a 1 = 3)
  (h2 : ∀ n : ℕ, n ≥ 1 → a (n + 1) - a n = n) : 
  a 11 = 58 := by
  sorry

end sequence_eleventh_term_l2317_231734


namespace inequality_empty_solution_set_l2317_231737

theorem inequality_empty_solution_set (a : ℝ) : 
  (∀ x : ℝ, a * x^2 - |x + 1| + 2*a ≥ 0) → a ≥ (Real.sqrt 3 + 1) / 4 := by
  sorry

end inequality_empty_solution_set_l2317_231737


namespace comic_book_collections_l2317_231765

def kymbrea_initial : ℕ := 50
def kymbrea_rate : ℕ := 1
def lashawn_initial : ℕ := 20
def lashawn_rate : ℕ := 7
def months : ℕ := 33

theorem comic_book_collections : 
  (lashawn_initial + lashawn_rate * months) = 
  3 * (kymbrea_initial + kymbrea_rate * months) :=
by sorry

end comic_book_collections_l2317_231765


namespace exactly_two_even_dice_probability_l2317_231716

def numDice : ℕ := 5
def numFaces : ℕ := 12

def probEven : ℚ := 1 / 2

def probExactlyTwoEven : ℚ := (numDice.choose 2 : ℚ) * probEven ^ 2 * (1 - probEven) ^ (numDice - 2)

theorem exactly_two_even_dice_probability :
  probExactlyTwoEven = 5 / 16 := by
  sorry

end exactly_two_even_dice_probability_l2317_231716


namespace rectangle_area_l2317_231794

theorem rectangle_area (p : ℝ) (h : p > 0) : ∃ (l w : ℝ),
  l > 0 ∧ w > 0 ∧
  l / w = 5 / 2 ∧
  2 * (l + w) = p ∧
  l * w = (5 / 98) * p^2 :=
sorry

end rectangle_area_l2317_231794


namespace third_degree_polynomial_property_l2317_231740

/-- A third-degree polynomial with real coefficients -/
def ThirdDegreePolynomial : Type := ℝ → ℝ

/-- The property that g satisfies the given conditions -/
def SatisfiesConditions (g : ThirdDegreePolynomial) : Prop :=
  ∀ x : ℝ, x ∈ ({-1, 0, 2, 4, 5, 8} : Set ℝ) → |g x| = 10

theorem third_degree_polynomial_property (g : ThirdDegreePolynomial) 
  (h : SatisfiesConditions g) : |g 3| = 11.25 := by
  sorry

end third_degree_polynomial_property_l2317_231740


namespace complex_equation_solution_l2317_231720

theorem complex_equation_solution (z : ℂ) :
  (3 + 4*I) * z = 1 - 2*I → z = -1/5 - 2/5*I := by
  sorry

end complex_equation_solution_l2317_231720


namespace junk_mail_distribution_l2317_231768

theorem junk_mail_distribution (total_mail : ℕ) (total_houses : ℕ) (white_houses : ℕ) (red_houses : ℕ)
  (h1 : total_mail = 48)
  (h2 : total_houses = 8)
  (h3 : white_houses = 2)
  (h4 : red_houses = 3)
  (h5 : total_houses > 0) :
  let colored_houses := white_houses + red_houses
  let mail_per_house := total_mail / total_houses
  mail_per_house = 6 ∧ colored_houses * mail_per_house = colored_houses * 6 :=
by sorry

end junk_mail_distribution_l2317_231768


namespace monotonicity_and_extrema_l2317_231783

def f (x : ℝ) : ℝ := 2*x^3 + 3*x^2 - 12*x + 1

theorem monotonicity_and_extrema :
  (∀ x < -2, (deriv f) x > 0) ∧
  (∀ x ∈ Set.Ioo (-2 : ℝ) 1, (deriv f) x < 0) ∧
  (∀ x > 1, (deriv f) x > 0) ∧
  IsLocalMax f (-2) ∧
  IsLocalMin f 1 ∧
  (∀ x ∈ Set.Icc (-1 : ℝ) 5, f x ≤ f 5) ∧
  (∀ x ∈ Set.Icc (-1 : ℝ) 5, f x ≥ f 1) ∧
  f 5 = 266 ∧
  f 1 = -6 :=
sorry

end monotonicity_and_extrema_l2317_231783


namespace exam_score_calculation_l2317_231745

theorem exam_score_calculation (total_questions : ℕ) (total_marks : ℤ) (correct_answers : ℕ) 
  (h1 : total_questions = 60)
  (h2 : total_marks = 130)
  (h3 : correct_answers = 38)
  (h4 : total_questions = correct_answers + (total_questions - correct_answers)) :
  ∃ (marks_per_correct : ℕ), 
    marks_per_correct * correct_answers - (total_questions - correct_answers) = total_marks ∧ 
    marks_per_correct = 4 := by
  sorry

end exam_score_calculation_l2317_231745


namespace class_size_problem_l2317_231786

/-- Given information about class sizes, prove the size of Class C -/
theorem class_size_problem (size_B : ℕ) (size_A : ℕ) (size_C : ℕ)
  (h1 : size_A = 2 * size_B)
  (h2 : size_A = size_C / 3)
  (h3 : size_B = 20) :
  size_C = 120 := by
  sorry

end class_size_problem_l2317_231786


namespace unique_c_value_l2317_231773

/-- The quadratic equation we're considering -/
def quadratic (b : ℝ) (c : ℝ) (x : ℝ) : Prop :=
  x^2 + (b^2 + 1/b^2) * x + c = 3

/-- The condition for the quadratic to have exactly one solution -/
def has_unique_solution (b : ℝ) (c : ℝ) : Prop :=
  ∃! x, quadratic b c x

/-- The main theorem statement -/
theorem unique_c_value : ∃! c : ℝ, c ≠ 0 ∧ 
  (∃! b : ℕ, b > 0 ∧ has_unique_solution (b : ℝ) c) :=
sorry

end unique_c_value_l2317_231773


namespace irrational_sqrt_7_rational_others_l2317_231731

theorem irrational_sqrt_7_rational_others : 
  (Irrational (Real.sqrt 7)) ∧ 
  (¬ Irrational 3.1415) ∧ 
  (¬ Irrational 3) ∧ 
  (¬ Irrational (1/3 : ℚ)) := by sorry

end irrational_sqrt_7_rational_others_l2317_231731


namespace cube_volume_from_surface_area_l2317_231719

theorem cube_volume_from_surface_area :
  ∀ (s : ℝ), (6 * s^2 = 294) → (s^3 = 343) := by
  sorry

end cube_volume_from_surface_area_l2317_231719


namespace cars_distance_theorem_l2317_231763

/-- The distance between two cars on a straight road -/
def distance_between_cars (initial_distance : ℝ) (car1_distance : ℝ) (car2_distance : ℝ) : ℝ :=
  initial_distance - (car1_distance + car2_distance)

/-- Theorem: The distance between two cars is 28 km -/
theorem cars_distance_theorem (initial_distance car1_distance car2_distance : ℝ) 
  (h1 : initial_distance = 113)
  (h2 : car1_distance = 50)
  (h3 : car2_distance = 35) :
  distance_between_cars initial_distance car1_distance car2_distance = 28 := by
  sorry

#eval distance_between_cars 113 50 35

end cars_distance_theorem_l2317_231763


namespace tan_beta_value_l2317_231728

theorem tan_beta_value (α β : Real) 
  (h1 : Real.tan α = 1/3)
  (h2 : Real.sin β = 2 * Real.cos (α + β) * Real.sin α) : 
  Real.tan β = 1/2 := by
  sorry

end tan_beta_value_l2317_231728


namespace square_sum_eight_l2317_231791

theorem square_sum_eight (a b : ℝ) (h : a^2 * b^2 + a^2 + b^2 + 16 = 10 * a * b) : 
  a^2 + b^2 = 8 := by
sorry

end square_sum_eight_l2317_231791


namespace system_solution_value_l2317_231700

theorem system_solution_value (x y a b : ℝ) : 
  3 * x - 2 * y + 20 = 0 →
  2 * x + 15 * y - 3 = 0 →
  a * x - b * y = 3 →
  6 * a + b = -3 := by
sorry

end system_solution_value_l2317_231700


namespace boat_distance_downstream_l2317_231751

/-- Calculates the distance traveled downstream given boat speed, stream speed, and time -/
def distanceDownstream (boatSpeed streamSpeed time : ℝ) : ℝ :=
  (boatSpeed + streamSpeed) * time

/-- Proves that the distance traveled downstream is 54 km under the given conditions -/
theorem boat_distance_downstream :
  let boatSpeed : ℝ := 10
  let streamSpeed : ℝ := 8
  let time : ℝ := 3
  distanceDownstream boatSpeed streamSpeed time = 54 := by
sorry

#eval distanceDownstream 10 8 3

end boat_distance_downstream_l2317_231751


namespace special_numbers_l2317_231705

/-- A two-digit number is equal to three times the product of its digits -/
def is_special_number (n : ℕ) : Prop :=
  10 ≤ n ∧ n < 100 ∧ ∃ (a b : ℕ), n = 10 * a + b ∧ n = 3 * a * b

/-- The only two-digit numbers that are equal to three times the product of their digits are 15 and 24 -/
theorem special_numbers : ∀ n : ℕ, is_special_number n ↔ (n = 15 ∨ n = 24) :=
sorry

end special_numbers_l2317_231705


namespace margarets_mean_score_l2317_231726

def scores : List ℝ := [82, 85, 88, 90, 95, 97, 98, 100]

theorem margarets_mean_score 
  (h1 : scores.length = 8)
  (h2 : ∃ (cyprian_scores margaret_scores : List ℝ), 
        cyprian_scores.length = 4 ∧ 
        margaret_scores.length = 4 ∧ 
        cyprian_scores ++ margaret_scores = scores)
  (h3 : ∃ (cyprian_scores : List ℝ), 
        cyprian_scores.length = 4 ∧ 
        cyprian_scores.sum / cyprian_scores.length = 91) :
  ∃ (margaret_scores : List ℝ), 
    margaret_scores.length = 4 ∧ 
    margaret_scores.sum / margaret_scores.length = 92.75 := by
  sorry

end margarets_mean_score_l2317_231726


namespace inequality_proof_l2317_231772

theorem inequality_proof (a b c : ℝ) (h_pos : a > 0 ∧ b > 0 ∧ c > 0) 
  (h_sum : a^(3/2) + b^(3/2) + c^(3/2) = 1) : 
  a * b * c ≤ 1/9 ∧ 
  a / (b + c) + b / (a + c) + c / (a + b) ≤ 1 / (2 * Real.sqrt (a * b * c)) := by
sorry

end inequality_proof_l2317_231772


namespace seven_lines_twenty_two_regions_l2317_231749

/-- A configuration of lines in a plane -/
structure LineConfiguration where
  total_lines : ℕ
  parallel_lines : ℕ
  non_parallel_lines : ℕ
  no_concurrency : Prop
  no_other_parallel : Prop

/-- Calculate the number of regions formed by a given line configuration -/
def number_of_regions (config : LineConfiguration) : ℕ :=
  sorry

/-- The theorem stating that the specific configuration of 7 lines creates 22 regions -/
theorem seven_lines_twenty_two_regions :
  ∀ (config : LineConfiguration),
    config.total_lines = 7 ∧
    config.parallel_lines = 2 ∧
    config.non_parallel_lines = 5 ∧
    config.no_concurrency ∧
    config.no_other_parallel →
    number_of_regions config = 22 :=
by sorry

end seven_lines_twenty_two_regions_l2317_231749


namespace hexadecagon_triangles_l2317_231779

/-- The number of vertices in a regular hexadecagon -/
def n : ℕ := 16

/-- A function to calculate the number of triangles in a regular polygon with n vertices -/
def num_triangles (n : ℕ) : ℕ := n.choose 3

/-- Theorem: The number of triangles in a regular hexadecagon is 560 -/
theorem hexadecagon_triangles : num_triangles n = 560 := by
  sorry

#eval num_triangles n

end hexadecagon_triangles_l2317_231779


namespace kate_change_l2317_231738

-- Define the prices of items
def gum_price : ℚ := 89 / 100
def chocolate_price : ℚ := 125 / 100
def chips_price : ℚ := 249 / 100

-- Define the sales tax rate
def sales_tax_rate : ℚ := 6 / 100

-- Define the amount Kate gave to the clerk
def amount_given : ℚ := 10

-- Theorem statement
theorem kate_change (gum : ℚ) (chocolate : ℚ) (chips : ℚ) (tax_rate : ℚ) (given : ℚ) :
  gum = gum_price →
  chocolate = chocolate_price →
  chips = chips_price →
  tax_rate = sales_tax_rate →
  given = amount_given →
  ∃ (change : ℚ), change = 509 / 100 ∧ 
    change = given - (gum + chocolate + chips + (gum + chocolate + chips) * tax_rate) :=
by sorry

end kate_change_l2317_231738


namespace eight_brown_boxes_contain_480_sticks_l2317_231703

/-- Calculates the number of sticks of gum in a given number of brown boxes. -/
def sticksInBrownBoxes (numBoxes : ℕ) : ℕ :=
  let packsPerCarton : ℕ := 5
  let sticksPerPack : ℕ := 3
  let cartonsPerBox : ℕ := 4
  numBoxes * cartonsPerBox * packsPerCarton * sticksPerPack

/-- Theorem stating that 8 brown boxes contain 480 sticks of gum. -/
theorem eight_brown_boxes_contain_480_sticks :
  sticksInBrownBoxes 8 = 480 := by
  sorry


end eight_brown_boxes_contain_480_sticks_l2317_231703


namespace third_number_divisible_by_seven_l2317_231748

theorem third_number_divisible_by_seven (n : ℕ) : 
  (Nat.gcd 35 91 = 7) → (Nat.gcd (Nat.gcd 35 91) n = 7) → (n % 7 = 0) := by
  sorry

end third_number_divisible_by_seven_l2317_231748


namespace hamburger_combinations_l2317_231702

theorem hamburger_combinations (num_condiments : ℕ) (num_patty_options : ℕ) :
  num_condiments = 10 →
  num_patty_options = 3 →
  (2^num_condiments) * num_patty_options = 3072 :=
by
  sorry

end hamburger_combinations_l2317_231702


namespace max_value_f_in_interval_l2317_231729

-- Define the function f
def f (x : ℝ) : ℝ := x^3 - 3*x^2 + 2

-- State the theorem
theorem max_value_f_in_interval :
  ∃ (c : ℝ), c ∈ Set.Icc (-1) 1 ∧ f c = 2 ∧ ∀ x ∈ Set.Icc (-1) 1, f x ≤ f c :=
sorry

end max_value_f_in_interval_l2317_231729


namespace wrapping_paper_distribution_l2317_231789

theorem wrapping_paper_distribution (total : ℚ) (decoration : ℚ) (num_presents : ℕ) :
  total = 5/8 ∧ decoration = 1/24 ∧ num_presents = 4 →
  (total - decoration) / (num_presents - 1) = 7/36 := by
sorry

end wrapping_paper_distribution_l2317_231789


namespace tiling_remainder_l2317_231714

/-- Represents a tiling of an 8x1 board -/
structure Tiling :=
  (pieces : ℕ)
  (red_used : Bool)
  (blue_used : Bool)
  (green_used : Bool)

/-- The number of valid tilings of an 8x1 board -/
def M : ℕ := sorry

/-- Theorem stating the result of the tiling problem -/
theorem tiling_remainder : M % 1000 = 336 := by sorry

end tiling_remainder_l2317_231714


namespace remainder_of_product_l2317_231778

theorem remainder_of_product (a b c : ℕ) (hc : c ≥ 3) 
  (ha : a % c = 1) (hb : b % c = 2) : (a * b) % c = 2 := by
  sorry

end remainder_of_product_l2317_231778


namespace machine_b_time_for_150_copies_l2317_231788

/-- Given two machines A and B with the following properties:
    1. Machine A makes 100 copies in 20 minutes
    2. Machines A and B working simultaneously for 30 minutes produce 600 copies
    This theorem proves that it takes 10 minutes for Machine B to make 150 copies -/
theorem machine_b_time_for_150_copies 
  (rate_a : ℚ) -- rate of machine A in copies per minute
  (rate_b : ℚ) -- rate of machine B in copies per minute
  (h1 : rate_a = 100 / 20) -- condition 1
  (h2 : 30 * (rate_a + rate_b) = 600) -- condition 2
  : 150 / rate_b = 10 := by sorry

end machine_b_time_for_150_copies_l2317_231788


namespace sandys_shopping_l2317_231790

/-- Sandy's shopping problem -/
theorem sandys_shopping (total : ℝ) (spent_percentage : ℝ) (remaining : ℝ) : 
  total = 300 → 
  spent_percentage = 30 → 
  remaining = total * (1 - spent_percentage / 100) → 
  remaining = 210 := by
sorry

end sandys_shopping_l2317_231790


namespace hydrochloric_acid_percentage_l2317_231774

/-- Calculates the percentage of hydrochloric acid in a solution after adding water -/
theorem hydrochloric_acid_percentage
  (initial_volume : ℝ)
  (initial_water_percentage : ℝ)
  (initial_acid_percentage : ℝ)
  (added_water : ℝ)
  (h1 : initial_volume = 300)
  (h2 : initial_water_percentage = 0.60)
  (h3 : initial_acid_percentage = 0.40)
  (h4 : added_water = 100)
  (h5 : initial_water_percentage + initial_acid_percentage = 1) :
  let initial_water := initial_volume * initial_water_percentage
  let initial_acid := initial_volume * initial_acid_percentage
  let final_volume := initial_volume + added_water
  let final_water := initial_water + added_water
  let final_acid := initial_acid
  final_acid / final_volume = 0.30 := by
  sorry

end hydrochloric_acid_percentage_l2317_231774


namespace tank_fill_time_l2317_231710

-- Define the rates of the pipes
def input_pipe_rate : ℚ := 1 / 15
def outlet_pipe_rate : ℚ := 1 / 45

-- Define the combined rate of all pipes
def combined_rate : ℚ := 2 * input_pipe_rate - outlet_pipe_rate

-- State the theorem
theorem tank_fill_time :
  (1 : ℚ) / combined_rate = 9 := by sorry

end tank_fill_time_l2317_231710


namespace quadratic_roots_sum_l2317_231777

theorem quadratic_roots_sum (a b : ℝ) : 
  (∀ x, ax^2 + bx - 2 = 0 ↔ x = -2 ∨ x = -1/4) → 
  a + b = -13 := by
sorry

end quadratic_roots_sum_l2317_231777


namespace fourth_root_of_y_squared_times_sqrt_y_l2317_231787

theorem fourth_root_of_y_squared_times_sqrt_y (y : ℝ) (h : y > 0) :
  (y^2 * y^(1/2))^(1/4) = y^(5/8) := by
  sorry

end fourth_root_of_y_squared_times_sqrt_y_l2317_231787


namespace quadratic_root_difference_l2317_231797

-- Define the quadratic equation coefficients
def a : ℝ := 5
def b : ℝ := -8
def c : ℝ := -7

-- Define the condition for m (not divisible by the square of any prime)
def is_square_free (m : ℕ) : Prop := ∀ p : ℕ, Prime p → (p^2 ∣ m) → False

-- Define the theorem
theorem quadratic_root_difference (m n : ℕ) (h1 : is_square_free m) (h2 : n > 0) :
  (((b^2 - 4*a*c).sqrt / (2*a)) = (m.sqrt / n)) → m + n = 56 :=
sorry

end quadratic_root_difference_l2317_231797


namespace movie_book_difference_l2317_231724

/-- The number of movies in the 'crazy silly school' series -/
def num_movies : ℕ := 47

/-- The number of books in the 'crazy silly school' series -/
def num_books : ℕ := 23

/-- Theorem: The difference between the number of movies and books in the 'crazy silly school' series is 24 -/
theorem movie_book_difference : num_movies - num_books = 24 := by
  sorry

end movie_book_difference_l2317_231724


namespace assignment_validity_l2317_231795

-- Define what constitutes a valid assignment statement
def is_valid_assignment (stmt : String) : Prop :=
  ∃ (var : String) (expr : String),
    stmt = var ++ " = " ++ expr ∧
    var.length > 0 ∧
    expr.length > 0 ∧
    var.all Char.isAlpha

theorem assignment_validity :
  is_valid_assignment "x = x + 1" ∧
  ¬is_valid_assignment "b =" ∧
  ¬is_valid_assignment "x = y = 10" ∧
  ¬is_valid_assignment "x + y = 10" :=
by sorry


end assignment_validity_l2317_231795


namespace equation_solution_l2317_231701

theorem equation_solution : 
  let equation := fun x : ℝ => 3 * x * (x - 2) = x - 2
  ∃ x₁ x₂ : ℝ, x₁ = 2 ∧ x₂ = 1/3 ∧ equation x₁ ∧ equation x₂ ∧ 
  ∀ x : ℝ, equation x → x = x₁ ∨ x = x₂ := by
sorry

end equation_solution_l2317_231701


namespace largest_coefficient_7th_8th_term_l2317_231757

/-- The binomial coefficient -/
def binomial (n k : ℕ) : ℕ := sorry

/-- The coefficient of the r-th term in the expansion of (x^2 + 1/x)^13 -/
def coefficient (r : ℕ) : ℕ := binomial 13 r

theorem largest_coefficient_7th_8th_term :
  ∀ r, r ≠ 6 ∧ r ≠ 7 → coefficient r ≤ coefficient 6 ∧ coefficient r ≤ coefficient 7 :=
sorry

end largest_coefficient_7th_8th_term_l2317_231757


namespace cyclist_speed_solution_l2317_231722

/-- Represents the speeds and distance of two cyclists traveling in opposite directions. -/
structure CyclistProblem where
  slower_speed : ℝ
  time : ℝ
  distance_apart : ℝ
  speed_difference : ℝ

/-- Calculates the total distance traveled by both cyclists. -/
def total_distance (p : CyclistProblem) : ℝ :=
  p.time * (2 * p.slower_speed + p.speed_difference)

/-- Theorem stating the conditions and solution for the cyclist problem. -/
theorem cyclist_speed_solution (p : CyclistProblem) 
  (h1 : p.time = 6)
  (h2 : p.distance_apart = 246)
  (h3 : p.speed_difference = 5) :
  p.slower_speed = 18 ∧ p.slower_speed + p.speed_difference = 23 :=
by
  sorry

#check cyclist_speed_solution

end cyclist_speed_solution_l2317_231722
