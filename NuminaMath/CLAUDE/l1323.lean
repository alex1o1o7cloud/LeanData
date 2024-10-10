import Mathlib

namespace number_of_routes_l1323_132315

/-- Recursive function representing the number of possible routes after n minutes -/
def M : ℕ → ℕ
| 0 => 0
| 1 => 1
| (n + 2) => M (n + 1) + M n

/-- The racing duration in minutes -/
def racingDuration : ℕ := 10

/-- Theorem stating that the number of possible routes after 10 minutes is 34 -/
theorem number_of_routes : M racingDuration = 34 := by
  sorry

end number_of_routes_l1323_132315


namespace power_neg_square_cube_l1323_132389

theorem power_neg_square_cube (b : ℝ) : ((-b)^2)^3 = b^6 := by
  sorry

end power_neg_square_cube_l1323_132389


namespace shaded_area_square_with_circles_l1323_132328

/-- The area of the shaded region in a square with circles at its vertices -/
theorem shaded_area_square_with_circles (s : ℝ) (r : ℝ) 
  (h_s : s = 10) (h_r : r = 3) : 
  let square_area := s^2
  let triangle_area := 8 * (1/2 * s/2 * (r * Real.sqrt 3))
  let sector_area := 4 * (1/12 * Real.pi * r^2)
  square_area - triangle_area - sector_area = 100 - 60 * Real.sqrt 3 - 3 * Real.pi := by
  sorry

end shaded_area_square_with_circles_l1323_132328


namespace division_addition_equality_l1323_132348

theorem division_addition_equality : (-180) / (-45) + (-9) = -5 := by
  sorry

end division_addition_equality_l1323_132348


namespace cos_alpha_value_l1323_132335

theorem cos_alpha_value (α : Real) (h1 : α ∈ Set.Ioo 0 π) 
  (h2 : 1 - 2 * Real.sin (2 * α) = Real.cos (2 * α)) : 
  Real.cos α = Real.sqrt 5 / 5 := by
  sorry

end cos_alpha_value_l1323_132335


namespace truncated_trigonal_pyramid_theorem_l1323_132341

/-- A truncated trigonal pyramid circumscribed around a sphere -/
structure TruncatedTrigonalPyramid where
  /-- The altitude of the pyramid -/
  h : ℝ
  /-- The circumradius of the lower base -/
  R₁ : ℝ
  /-- The circumradius of the upper base -/
  R₂ : ℝ
  /-- The distance from the circumcenter of the lower base to the point where the sphere touches it -/
  O₁T₁ : ℝ
  /-- The distance from the circumcenter of the upper base to the point where the sphere touches it -/
  O₂T₂ : ℝ
  /-- The sphere touches both bases of the pyramid -/
  touches_bases : True

/-- The main theorem about the relationship between the measurements of a truncated trigonal pyramid -/
theorem truncated_trigonal_pyramid_theorem (p : TruncatedTrigonalPyramid) :
  p.R₁ * p.R₂ * p.h^2 = (p.R₁^2 - p.O₁T₁^2) * (p.R₂^2 - p.O₂T₂^2) := by
  sorry

end truncated_trigonal_pyramid_theorem_l1323_132341


namespace subcommittee_formation_count_l1323_132347

theorem subcommittee_formation_count :
  let total_republicans : ℕ := 8
  let total_democrats : ℕ := 10
  let subcommittee_republicans : ℕ := 3
  let subcommittee_democrats : ℕ := 2
  let ways_to_choose_republicans : ℕ := Nat.choose total_republicans subcommittee_republicans
  let ways_to_choose_chair : ℕ := total_democrats
  let ways_to_choose_other_democrat : ℕ := Nat.choose (total_democrats - 1) (subcommittee_democrats - 1)
  ways_to_choose_republicans * ways_to_choose_chair * ways_to_choose_other_democrat = 5040 :=
by
  sorry

end subcommittee_formation_count_l1323_132347


namespace problems_per_worksheet_l1323_132377

theorem problems_per_worksheet
  (total_worksheets : ℕ)
  (graded_worksheets : ℕ)
  (remaining_problems : ℕ)
  (h1 : total_worksheets = 15)
  (h2 : graded_worksheets = 7)
  (h3 : remaining_problems = 24)
  : (remaining_problems / (total_worksheets - graded_worksheets) : ℚ) = 3 :=
by sorry

end problems_per_worksheet_l1323_132377


namespace find_a_l1323_132374

-- Define the universal set U
def U (a : ℝ) : Set ℝ := {2, 4, 3 - a^2}

-- Define set P
def P (a : ℝ) : Set ℝ := {2, a^2 - a + 2}

-- Define the complement of P with respect to U
def complementP (a : ℝ) : Set ℝ := {-1}

-- Theorem statement
theorem find_a : ∃ a : ℝ, 
  (U a = P a ∪ complementP a) ∧ 
  (U a = {2, 4, 3 - a^2}) ∧ 
  (P a = {2, a^2 - a + 2}) ∧ 
  (complementP a = {-1}) →
  a = 2 := by sorry

end find_a_l1323_132374


namespace probability_in_standard_deck_l1323_132326

/-- A standard deck of cards -/
structure Deck :=
  (cards : Nat)
  (red_cards : Nat)
  (black_cards : Nat)

/-- The probability of drawing a red card first and then a black card -/
def probability_red_then_black (d : Deck) : Rat :=
  (d.red_cards * d.black_cards) / (d.cards * (d.cards - 1))

/-- Theorem statement for the probability in a standard 52-card deck -/
theorem probability_in_standard_deck :
  let d : Deck := ⟨52, 26, 26⟩
  probability_red_then_black d = 13 / 51 := by
  sorry

end probability_in_standard_deck_l1323_132326


namespace car_production_increase_l1323_132356

/-- Proves that adding 50 cars to the monthly production of 100 cars
    will result in an annual production of 1800 cars. -/
theorem car_production_increase (current_monthly : ℕ) (target_yearly : ℕ) (increase : ℕ) :
  current_monthly = 100 →
  target_yearly = 1800 →
  increase = 50 →
  (current_monthly + increase) * 12 = target_yearly := by
  sorry

#check car_production_increase

end car_production_increase_l1323_132356


namespace inequality_solution_l1323_132316

theorem inequality_solution (x : ℝ) :
  (2 - 1 / (3 * x + 4) < 5) ↔ (x < -4/3 ∨ x > -13/9) :=
by sorry

end inequality_solution_l1323_132316


namespace smallest_m_for_equal_notebooks_and_pencils_l1323_132334

theorem smallest_m_for_equal_notebooks_and_pencils :
  ∃ (M : ℕ+), (M = 5) ∧
  (∀ (k : ℕ+), k < M → ¬∃ (n : ℕ+), 3 * k = 5 * n) ∧
  (∃ (n : ℕ+), 3 * M = 5 * n) := by
  sorry

end smallest_m_for_equal_notebooks_and_pencils_l1323_132334


namespace positive_difference_54_and_y_l1323_132321

theorem positive_difference_54_and_y (y : ℝ) (h : (54 + y) / 2 = 32) :
  |54 - y| = 44 := by
  sorry

end positive_difference_54_and_y_l1323_132321


namespace cube_surface_area_l1323_132300

theorem cube_surface_area (edge_length : ℝ) (h : edge_length = 20) :
  6 * edge_length^2 = 2400 :=
by sorry

end cube_surface_area_l1323_132300


namespace equation_solution_l1323_132390

theorem equation_solution : 
  ∀ x y z : ℕ+, 
  (x : ℚ) / 21 * (y : ℚ) / 189 + (z : ℚ) = 1 → 
  x = 21 ∧ y = 567 ∧ z = 1 := by
  sorry

end equation_solution_l1323_132390


namespace sin_330_degrees_l1323_132370

theorem sin_330_degrees : 
  Real.sin (330 * Real.pi / 180) = -Real.sqrt 3 / 2 := by sorry

end sin_330_degrees_l1323_132370


namespace consecutive_integers_with_unique_prime_factors_l1323_132364

theorem consecutive_integers_with_unique_prime_factors (n : ℕ) (hn : n > 0) :
  ∃ x : ℤ, ∀ i : ℕ, 1 ≤ i ∧ i ≤ n →
    ∃ (p : ℕ) (k : ℕ), Prime p ∧ (x + i : ℤ) = p * k ∧ ¬(p ∣ k) :=
sorry

end consecutive_integers_with_unique_prime_factors_l1323_132364


namespace number_with_fraction_difference_l1323_132342

theorem number_with_fraction_difference (x : ℤ) : x - (7 : ℤ) * x / (13 : ℤ) = 110 ↔ x = 237 := by
  sorry

end number_with_fraction_difference_l1323_132342


namespace range_of_a_l1323_132360

-- Define set A
def A : Set ℝ := {x : ℝ | x^2 - 4*x + 3 ≤ 0}

-- Define set B
def B (a : ℝ) : Set ℝ := {x : ℝ | x^2 - a*x < x - a}

-- Theorem statement
theorem range_of_a :
  (∀ x : ℝ, x ∈ B a → x ∈ A) ∧ 
  (∃ x : ℝ, x ∈ A ∧ x ∉ B a) →
  a ∈ Set.Icc 1 3 :=
sorry

end range_of_a_l1323_132360


namespace shyne_plants_l1323_132327

/-- The number of eggplants that can be grown from one seed packet -/
def eggplants_per_packet : ℕ := 14

/-- The number of sunflowers that can be grown from one seed packet -/
def sunflowers_per_packet : ℕ := 10

/-- The number of eggplant seed packets Shyne bought -/
def eggplant_packets : ℕ := 4

/-- The number of sunflower seed packets Shyne bought -/
def sunflower_packets : ℕ := 6

/-- The total number of plants Shyne can grow -/
def total_plants : ℕ := eggplants_per_packet * eggplant_packets + sunflowers_per_packet * sunflower_packets

theorem shyne_plants : total_plants = 116 := by
  sorry

end shyne_plants_l1323_132327


namespace hyperbola_asymptote_implies_b_equals_one_l1323_132375

/-- 
Given a hyperbola with equation x²/4 - y²/b² = 1 where b > 0,
if the asymptotes are y = ±(1/2)x, then b = 1.
-/
theorem hyperbola_asymptote_implies_b_equals_one (b : ℝ) : 
  b > 0 → 
  (∀ x y : ℝ, x^2 / 4 - y^2 / b^2 = 1 → 
    (y = (1/2) * x ∨ y = -(1/2) * x)) → 
  b = 1 := by sorry

end hyperbola_asymptote_implies_b_equals_one_l1323_132375


namespace queen_mary_legs_l1323_132346

/-- The total number of legs on a ship with cats and humans -/
def total_legs (total_heads : ℕ) (cat_count : ℕ) (one_legged_human_count : ℕ) : ℕ :=
  let human_count := total_heads - cat_count
  let cat_legs := cat_count * 4
  let human_legs := (human_count - one_legged_human_count) * 2 + one_legged_human_count
  cat_legs + human_legs

/-- Theorem stating the total number of legs on the Queen Mary II -/
theorem queen_mary_legs : total_legs 16 5 1 = 41 := by
  sorry

end queen_mary_legs_l1323_132346


namespace part_one_part_two_l1323_132397

-- Define the function f(x)
def f (a : ℝ) (x : ℝ) : ℝ := |x - a|

-- Part I
theorem part_one (x : ℝ) :
  let a : ℝ := 1
  (f a x ≥ 4 - |x - 1|) ↔ (x ≤ -1 ∨ x ≥ 3) :=
sorry

-- Part II
theorem part_two (m n : ℝ) (hm : m > 0) (hn : n > 0) :
  let a : ℝ := 1
  (∀ x, f a x ≤ 1 ↔ 0 ≤ x ∧ x ≤ 2) →
  (1/m + 1/(2*n) = a) →
  (∀ k l, k > 0 → l > 0 → 1/k + 1/(2*l) = a → m*n ≤ k*l) →
  m*n = 2 :=
sorry

end part_one_part_two_l1323_132397


namespace lighter_box_weight_l1323_132392

/-- Proves that the weight of lighter boxes is 12 pounds given the conditions of the shipment. -/
theorem lighter_box_weight (total_boxes : Nat) (heavier_box_weight : Nat) (initial_average : Nat) 
  (final_average : Nat) (removed_boxes : Nat) :
  total_boxes = 20 →
  heavier_box_weight = 20 →
  initial_average = 18 →
  final_average = 12 →
  removed_boxes = 15 →
  ∃ (lighter_box_weight : Nat), 
    lighter_box_weight = 12 ∧
    lighter_box_weight * (total_boxes - removed_boxes) = 
      final_average * (total_boxes - removed_boxes) :=
by sorry

end lighter_box_weight_l1323_132392


namespace custom_operation_theorem_l1323_132331

/-- Custom operation ⊗ defined for real numbers -/
def otimes (a b : ℝ) : ℝ := 2 * a + b

/-- Theorem stating that if x ⊗ (-y) = 2 and (2y) ⊗ x = 1, then x + y = 1 -/
theorem custom_operation_theorem (x y : ℝ) 
  (h1 : otimes x (-y) = 2) 
  (h2 : otimes (2 * y) x = 1) : 
  x + y = 1 := by
  sorry

end custom_operation_theorem_l1323_132331


namespace x_value_l1323_132312

theorem x_value (x y : ℝ) : 
  (x = y * 0.9) → (y = 125 * 1.1) → x = 123.75 := by
  sorry

end x_value_l1323_132312


namespace arithmetic_progression_polynomial_j_value_l1323_132359

/-- A polynomial of degree 4 with four distinct real zeros in arithmetic progression -/
structure ArithmeticProgressionPolynomial where
  j : ℝ
  k : ℝ
  zeros : Fin 4 → ℝ
  distinct : ∀ i j, i ≠ j → zeros i ≠ zeros j
  arithmetic_progression : ∃ (b d : ℝ), ∀ i, zeros i = b + d * i.val
  is_zero : ∀ x, x^4 + j * x^2 + k * x + 256 = (x - zeros 0) * (x - zeros 1) * (x - zeros 2) * (x - zeros 3)

/-- The value of j in an ArithmeticProgressionPolynomial is -40 -/
theorem arithmetic_progression_polynomial_j_value (p : ArithmeticProgressionPolynomial) : p.j = -40 := by
  sorry

end arithmetic_progression_polynomial_j_value_l1323_132359


namespace rational_equation_solution_l1323_132318

theorem rational_equation_solution (x : ℝ) : 
  (1 / (x^2 + 12*x - 9) + 1 / (x^2 + 3*x - 9) + 1 / (x^2 - 14*x - 9) = 0) ↔ 
  (x = 3 ∨ x = 1 ∨ x = -3 ∨ x = -9) := by
sorry

end rational_equation_solution_l1323_132318


namespace trigonometric_expressions_l1323_132301

theorem trigonometric_expressions (α : Real) (h : Real.tan α = 2) : 
  ((Real.sin α - 4 * Real.cos α) / (5 * Real.sin α + 2 * Real.cos α) = -1/6) ∧ 
  (Real.sin α ^ 2 + Real.sin (2 * α) = 8/5) := by
  sorry

end trigonometric_expressions_l1323_132301


namespace geometric_series_sum_l1323_132380

/-- The sum of an infinite geometric series with first term 1 and common ratio 1/4 is 4/3 -/
theorem geometric_series_sum : 
  let a : ℚ := 1
  let r : ℚ := 1/4
  let S : ℚ := ∑' n, a * r^n
  S = 4/3 := by
sorry

end geometric_series_sum_l1323_132380


namespace smallest_slope_tangent_line_l1323_132329

/-- The function f(x) = x^3 + 3x^2 + 6x - 10 --/
def f (x : ℝ) : ℝ := x^3 + 3*x^2 + 6*x - 10

/-- The derivative of f(x) --/
def f' (x : ℝ) : ℝ := 3*x^2 + 6*x + 6

theorem smallest_slope_tangent_line :
  ∃ (a b c : ℝ), 
    (∀ x y : ℝ, y = f x → (a*x + b*y + c = 0 ↔ y - f x = f' x * (x - x)))  -- Tangent line equation
    ∧ (∀ x₀ : ℝ, f' x₀ ≥ f' (-1))  -- Slope at x = -1 is the smallest
    ∧ a = 3 ∧ b = -1 ∧ c = -11  -- Coefficients of the tangent line equation
:= by sorry

end smallest_slope_tangent_line_l1323_132329


namespace dragon_defeat_probability_is_one_l1323_132391

/-- Represents the state of the dragon's heads -/
structure DragonState where
  heads : ℕ

/-- Represents the possible outcomes after chopping off a head -/
inductive ChopOutcome
  | TwoHeadsGrow
  | OneHeadGrows
  | NoHeadGrows

/-- The probability distribution of chop outcomes -/
def chopProbability : ChopOutcome → ℚ
  | ChopOutcome.TwoHeadsGrow => 1/4
  | ChopOutcome.OneHeadGrows => 1/3
  | ChopOutcome.NoHeadGrows => 5/12

/-- The transition function for the dragon state after a chop -/
def transition (state : DragonState) (outcome : ChopOutcome) : DragonState :=
  match outcome with
  | ChopOutcome.TwoHeadsGrow => ⟨state.heads + 1⟩
  | ChopOutcome.OneHeadGrows => state
  | ChopOutcome.NoHeadGrows => ⟨state.heads - 1⟩

/-- The probability of eventually defeating the dragon -/
noncomputable def defeatProbability (initialState : DragonState) : ℝ :=
  sorry

/-- Theorem stating that the probability of defeating the dragon is 1 -/
theorem dragon_defeat_probability_is_one :
  defeatProbability ⟨3⟩ = 1 := by sorry

end dragon_defeat_probability_is_one_l1323_132391


namespace total_turkey_cost_l1323_132372

def turkey_cost (weight : ℕ) (price_per_kg : ℕ) : ℕ := weight * price_per_kg

theorem total_turkey_cost : 
  let first_turkey := 6
  let second_turkey := 9
  let third_turkey := 2 * second_turkey
  let price_per_kg := 2
  turkey_cost first_turkey price_per_kg + 
  turkey_cost second_turkey price_per_kg + 
  turkey_cost third_turkey price_per_kg = 66 := by
sorry

end total_turkey_cost_l1323_132372


namespace temperature_below_freezing_is_negative_three_l1323_132398

/-- The freezing point of water in degrees Celsius -/
def freezing_point : ℝ := 0

/-- The temperature difference in degrees Celsius -/
def temperature_difference : ℝ := 3

/-- The temperature below freezing point -/
def temperature_below_freezing : ℝ := freezing_point - temperature_difference

theorem temperature_below_freezing_is_negative_three :
  temperature_below_freezing = -3 := by sorry

end temperature_below_freezing_is_negative_three_l1323_132398


namespace unique_eight_times_digit_sum_l1323_132352

def sum_of_digits (n : ℕ) : ℕ := sorry

theorem unique_eight_times_digit_sum :
  ∃! n : ℕ, n < 500 ∧ n > 0 ∧ n = 8 * sum_of_digits n := by sorry

end unique_eight_times_digit_sum_l1323_132352


namespace basketball_team_selection_l1323_132304

/-- The number of ways to choose k elements from n elements -/
def choose (n k : ℕ) : ℕ := sorry

theorem basketball_team_selection :
  let total_players : ℕ := 16
  let quadruplets : ℕ := 4
  let starters : ℕ := 7
  let quadruplets_in_lineup : ℕ := 3
  
  (choose quadruplets quadruplets_in_lineup) * 
  (choose (total_players - quadruplets) (starters - quadruplets_in_lineup)) = 1980 :=
by sorry

end basketball_team_selection_l1323_132304


namespace binary_to_base5_equivalence_l1323_132381

-- Define the binary number
def binary_num : ℕ := 168  -- 10101000 in binary is 168 in decimal

-- Define the base-5 number
def base5_num : List ℕ := [1, 1, 3, 3]  -- 1133 in base-5

-- Theorem to prove the equivalence
theorem binary_to_base5_equivalence :
  (binary_num : ℕ) = (List.foldl (λ acc d => acc * 5 + d) 0 base5_num) :=
sorry

end binary_to_base5_equivalence_l1323_132381


namespace angle_between_hexagon_and_square_diagonal_l1323_132354

/-- A configuration with a square inside a regular hexagon sharing a common vertex. -/
structure SquareInHexagon where
  /-- The measure of an interior angle of the regular hexagon -/
  hexagon_angle : ℝ
  /-- The measure of an interior angle of the square -/
  square_angle : ℝ
  /-- The hexagon is regular -/
  hexagon_regular : hexagon_angle = 120
  /-- The square has right angles -/
  square_right : square_angle = 90

/-- The theorem stating that the angle between the hexagon side and square diagonal is 75° -/
theorem angle_between_hexagon_and_square_diagonal (config : SquareInHexagon) :
  config.hexagon_angle - (config.square_angle / 2) = 75 := by
  sorry

end angle_between_hexagon_and_square_diagonal_l1323_132354


namespace angle_sum_proof_l1323_132322

theorem angle_sum_proof (α β : Real) (h1 : 0 < α ∧ α < π/2) (h2 : 0 < β ∧ β < π/2)
  (h3 : (1 - Real.tan α) * (1 - Real.tan β) = 2) : α + β = 3*π/4 := by
  sorry

end angle_sum_proof_l1323_132322


namespace ratio_problem_l1323_132365

theorem ratio_problem (A B C : ℝ) (h1 : A + B + C = 98) (h2 : A / B = 2 / 3) (h3 : B = 30) :
  B / C = 5 / 8 := by
  sorry

end ratio_problem_l1323_132365


namespace parallel_sides_in_pentagon_l1323_132330

-- Define the pentagon
structure Pentagon (V : Type*) [AddCommGroup V] [Module ℝ V] :=
  (A B C D E : V)

-- Define the parallelism relation
def Parallel (V : Type*) [AddCommGroup V] [Module ℝ V] (v w : V) : Prop :=
  ∃ (t : ℝ), v = t • w

-- State the theorem
theorem parallel_sides_in_pentagon
  (V : Type*) [AddCommGroup V] [Module ℝ V] (p : Pentagon V)
  (h1 : Parallel V (p.B - p.C) (p.A - p.D))
  (h2 : Parallel V (p.C - p.D) (p.B - p.E))
  (h3 : Parallel V (p.D - p.E) (p.A - p.C))
  (h4 : Parallel V (p.A - p.E) (p.B - p.D)) :
  Parallel V (p.A - p.B) (p.C - p.E) :=
sorry

end parallel_sides_in_pentagon_l1323_132330


namespace polynomial_division_remainder_l1323_132320

theorem polynomial_division_remainder : ∃ q : Polynomial ℤ, 
  (X^5 - X^3 + X - 1) * (X^3 - X^2 + 1) = (X^2 + X + 1) * q + (-7 : Polynomial ℤ) := by
  sorry

end polynomial_division_remainder_l1323_132320


namespace sequence_sum_equals_eight_l1323_132303

/-- Given a geometric sequence and an arithmetic sequence with specific properties, 
    prove that the sum of two terms in the arithmetic sequence equals 8. -/
theorem sequence_sum_equals_eight 
  (a : ℕ → ℝ) 
  (b : ℕ → ℝ) 
  (h_geometric : ∀ n m : ℕ, a (n + m) = a n * (a 1) ^ m) 
  (h_arithmetic : ∀ n m : ℕ, b (n + m) = b n + m * (b 1 - b 0)) 
  (h_relation : a 3 * a 11 = 4 * a 7) 
  (h_equal : b 7 = a 7) : 
  b 5 + b 9 = 8 := by
sorry

end sequence_sum_equals_eight_l1323_132303


namespace exists_special_sequence_l1323_132340

def sequence_condition (a : ℕ → ℕ) : Prop :=
  ∀ i j p q r, i ≠ j ∧ p ≠ q ∧ p ≠ r ∧ q ≠ r →
    Nat.gcd (a i + a j) (a p + a q + a r) = 1

theorem exists_special_sequence :
  ∃ a : ℕ → ℕ, sequence_condition a ∧ (∀ n, a n < a (n + 1)) :=
sorry

end exists_special_sequence_l1323_132340


namespace math_evening_students_l1323_132323

theorem math_evening_students (total_rows : ℕ) 
  (h1 : 70 < total_rows * total_rows ∧ total_rows * total_rows < 90)
  (h2 : total_rows = (total_rows - 3) + 3)
  (h3 : 3 * total_rows < 90 ∧ 3 * total_rows > 70) :
  (total_rows * (total_rows - 3) = 54) ∧ (total_rows * 3 = 27) := by
  sorry

end math_evening_students_l1323_132323


namespace min_value_complex_sum_l1323_132339

theorem min_value_complex_sum (z : ℂ) (h : Complex.abs (z - (3 - 3*I)) = 4) :
  Complex.abs (z + (2 - I))^2 + Complex.abs (z - (6 - 5*I))^2 ≥ 76 :=
by sorry

end min_value_complex_sum_l1323_132339


namespace diamond_calculation_l1323_132382

-- Define the diamond operation
def diamond (a b : ℤ) : ℤ := Int.natAbs (a + b - 10)

-- Theorem statement
theorem diamond_calculation : diamond 5 (diamond 3 8) = 4 := by
  sorry

end diamond_calculation_l1323_132382


namespace square_root_equal_self_l1323_132337

theorem square_root_equal_self (a : ℝ) : 
  (Real.sqrt a = a) → (a^2 + 1 = 1 ∨ a^2 + 1 = 2) := by
  sorry

end square_root_equal_self_l1323_132337


namespace building_cost_theorem_l1323_132399

/-- Calculates the total cost of all units in a building -/
def total_cost (total_units : ℕ) (cost_1bed : ℕ) (cost_2bed : ℕ) (num_2bed : ℕ) : ℕ :=
  let num_1bed := total_units - num_2bed
  num_1bed * cost_1bed + num_2bed * cost_2bed

/-- Theorem stating the total cost of all units in the given building configuration -/
theorem building_cost_theorem : 
  total_cost 12 360 450 7 = 4950 := by
  sorry

#eval total_cost 12 360 450 7

end building_cost_theorem_l1323_132399


namespace sufficient_not_necessary_l1323_132378

theorem sufficient_not_necessary (a : ℝ) :
  (∀ a, a > 1 / a^2 → a^2 > 1 / a) ∧
  (∃ a, a^2 > 1 / a ∧ a ≤ 1 / a^2) :=
sorry

end sufficient_not_necessary_l1323_132378


namespace one_mile_in_yards_l1323_132369

-- Define the conversion rates
def mile_to_furlong : ℚ := 5
def furlong_to_rod : ℚ := 50
def rod_to_yard : ℚ := 5

-- Theorem statement
theorem one_mile_in_yards :
  mile_to_furlong * furlong_to_rod * rod_to_yard = 1250 := by
  sorry

end one_mile_in_yards_l1323_132369


namespace bridge_length_l1323_132373

/-- The length of a bridge given a train crossing it -/
theorem bridge_length (train_length : ℝ) (crossing_time : ℝ) (train_speed : ℝ) :
  train_length = 100 →
  crossing_time = 60 →
  train_speed = 5 →
  train_speed * crossing_time - train_length = 200 :=
by
  sorry

end bridge_length_l1323_132373


namespace expression_value_l1323_132311

theorem expression_value : 4 * ((3.6 * 0.48 * 2.50) / (0.12 * 0.09 * 0.5)) = 8000 := by
  sorry

end expression_value_l1323_132311


namespace pizza_meat_calculation_l1323_132386

/-- Represents the number of pieces of each type of meat on a pizza --/
structure PizzaToppings where
  pepperoni : ℕ
  ham : ℕ
  sausage : ℕ

/-- Calculates the total number of pieces of meat on each slice of pizza --/
def meat_per_slice (toppings : PizzaToppings) (slices : ℕ) : ℚ :=
  (toppings.pepperoni + toppings.ham + toppings.sausage : ℚ) / slices

theorem pizza_meat_calculation :
  let toppings : PizzaToppings := {
    pepperoni := 30,
    ham := 30 * 2,
    sausage := 30 + 12
  }
  let slices : ℕ := 6
  meat_per_slice toppings slices = 22 := by
  sorry

#eval meat_per_slice { pepperoni := 30, ham := 30 * 2, sausage := 30 + 12 } 6

end pizza_meat_calculation_l1323_132386


namespace calculate_expression_l1323_132383

theorem calculate_expression : ((28 / (5 + 3 - 6)) * 7) = 98 := by
  sorry

end calculate_expression_l1323_132383


namespace cyrus_shot_percentage_l1323_132367

def total_shots : ℕ := 20
def missed_shots : ℕ := 4

def shots_made : ℕ := total_shots - missed_shots

def percentage_made : ℚ := (shots_made : ℚ) / (total_shots : ℚ) * 100

theorem cyrus_shot_percentage :
  percentage_made = 80 := by
  sorry

end cyrus_shot_percentage_l1323_132367


namespace bacteria_at_8_20_am_l1323_132357

/-- Calculates the bacterial population after a given time period -/
def bacterial_population (initial_population : ℕ) (doubling_time : ℕ) (elapsed_time : ℕ) : ℕ :=
  initial_population * (2 ^ (elapsed_time / doubling_time))

/-- Theorem stating the bacterial population at 8:20 AM -/
theorem bacteria_at_8_20_am : 
  let initial_population : ℕ := 30
  let doubling_time : ℕ := 4  -- in minutes
  let elapsed_time : ℕ := 20  -- in minutes
  bacterial_population initial_population doubling_time elapsed_time = 960 :=
by
  sorry


end bacteria_at_8_20_am_l1323_132357


namespace smallest_class_is_four_l1323_132324

/-- Represents a systematic sampling of classes. -/
structure SystematicSampling where
  total_classes : ℕ
  selected_classes : ℕ
  sum_of_selected : ℕ

/-- The smallest class number in a systematic sampling. -/
def smallest_class (s : SystematicSampling) : ℕ :=
  (s.sum_of_selected - (s.selected_classes * (s.selected_classes - 1) * s.total_classes / s.selected_classes / 2)) / s.selected_classes

/-- Theorem stating that for the given conditions, the smallest class number is 4. -/
theorem smallest_class_is_four (s : SystematicSampling) 
  (h1 : s.total_classes = 24)
  (h2 : s.selected_classes = 4)
  (h3 : s.sum_of_selected = 52) :
  smallest_class s = 4 := by
  sorry

end smallest_class_is_four_l1323_132324


namespace real_part_of_z_is_zero_l1323_132363

theorem real_part_of_z_is_zero :
  let i : ℂ := Complex.I
  let z : ℂ := (2 + i) / (-2*i + 1)
  Complex.re z = 0 := by
sorry

end real_part_of_z_is_zero_l1323_132363


namespace no_roots_composition_l1323_132307

/-- A quadratic polynomial f(x) = ax^2 + bx + c -/
structure QuadraticPolynomial (α : Type*) [Ring α] where
  a : α
  b : α
  c : α

/-- The function represented by a quadratic polynomial -/
def evalQuadratic {α : Type*} [Ring α] (f : QuadraticPolynomial α) (x : α) : α :=
  f.a * x * x + f.b * x + f.c

theorem no_roots_composition {α : Type*} [LinearOrderedField α] (f : QuadraticPolynomial α) :
  (∀ x : α, evalQuadratic f x ≠ x) →
  (∀ x : α, evalQuadratic f (evalQuadratic f x) ≠ x) := by
  sorry

end no_roots_composition_l1323_132307


namespace A_intersect_Z_l1323_132379

def A : Set ℝ := {x : ℝ | |x - 1| < 2}

theorem A_intersect_Z : A ∩ Set.range (Int.cast : ℤ → ℝ) = {0, 1, 2} := by
  sorry

end A_intersect_Z_l1323_132379


namespace binomial_product_l1323_132325

theorem binomial_product : (Nat.choose 10 3) * (Nat.choose 8 3) = 6720 := by
  sorry

end binomial_product_l1323_132325


namespace triangle_right_angled_l1323_132308

/-- Given a triangle ABC with sides a, b, c opposite to angles A, B, C respectively,
    prove that if b*cos(B) + c*cos(C) = a*cos(A), then the triangle is right-angled. -/
theorem triangle_right_angled (A B C a b c : ℝ) : 
  0 < A ∧ A < π ∧ 
  0 < B ∧ B < π ∧ 
  0 < C ∧ C < π ∧
  A + B + C = π ∧
  b * Real.cos B + c * Real.cos C = a * Real.cos A →
  B = π/2 ∨ C = π/2 := by
  sorry

end triangle_right_angled_l1323_132308


namespace total_ways_is_81_l1323_132306

/-- The number of base options available to each student -/
def num_bases : ℕ := 3

/-- The number of students choosing bases -/
def num_students : ℕ := 4

/-- The total number of ways for students to choose bases -/
def total_ways : ℕ := num_bases ^ num_students

/-- Theorem stating that the total number of ways is 81 -/
theorem total_ways_is_81 : total_ways = 81 := by
  sorry

end total_ways_is_81_l1323_132306


namespace largest_temperature_time_l1323_132343

theorem largest_temperature_time (t : ℝ) : 
  (-t^2 + 10*t + 40 = 60) → t ≤ 5 + Real.sqrt 5 :=
by sorry

end largest_temperature_time_l1323_132343


namespace divisibility_by_290304_l1323_132333

theorem divisibility_by_290304 (a b : Nat) (ha : Nat.Prime a) (hb : Nat.Prime b) 
  (ga : a > 7) (gb : b > 7) : 
  290304 ∣ (a^2 - 1) * (b^2 - 1) * (a^6 - b^6) := by
  sorry

end divisibility_by_290304_l1323_132333


namespace smallest_prime_perimeter_triangle_l1323_132310

/-- A function that checks if a number is prime -/
def isPrime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ d : ℕ, d > 1 → d < n → ¬(n % d = 0)

/-- A function that checks if three numbers can form a triangle -/
def canFormTriangle (a b c : ℕ) : Prop :=
  a + b > c ∧ a + c > b ∧ b + c > a

/-- A function that checks if three numbers are consecutive odd primes -/
def areConsecutiveOddPrimes (a b c : ℕ) : Prop :=
  isPrime a ∧ isPrime b ∧ isPrime c ∧
  b = a + 2 ∧ c = b + 2 ∧
  a % 2 = 1 ∧ b % 2 = 1 ∧ c % 2 = 1

/-- The main theorem stating that the smallest perimeter of a scalene triangle
    with consecutive odd prime side lengths and a prime perimeter is 23 -/
theorem smallest_prime_perimeter_triangle :
  ∀ a b c : ℕ,
  areConsecutiveOddPrimes a b c →
  canFormTriangle a b c →
  isPrime (a + b + c) →
  a + b + c ≥ 23 :=
sorry

end smallest_prime_perimeter_triangle_l1323_132310


namespace residue_system_product_condition_l1323_132351

/-- A function that generates a complete residue system modulo n -/
def completeResidueSystem (n : ℕ) : Fin n → ℕ :=
  fun i => i.val

/-- Predicate to check if a list of natural numbers forms a complete residue system modulo n -/
def isCompleteResidueSystem (n : ℕ) (l : List ℕ) : Prop :=
  l.length = n ∧ ∀ k, 0 ≤ k ∧ k < n → ∃ x ∈ l, x % n = k

theorem residue_system_product_condition (n : ℕ) : 
  (∃ (a b : Fin n → ℕ), 
    isCompleteResidueSystem n (List.ofFn a) ∧
    isCompleteResidueSystem n (List.ofFn b) ∧
    isCompleteResidueSystem n (List.ofFn (fun i => (a i * b i) % n))) ↔ 
  n = 1 ∨ n = 2 :=
sorry

end residue_system_product_condition_l1323_132351


namespace quadratic_factorization_l1323_132314

theorem quadratic_factorization (y : ℝ) (A B : ℤ) 
  (h : ∀ y, 12 * y^2 - 65 * y + 42 = (A * y - 14) * (B * y - 3)) : 
  A * B + A = 15 := by
  sorry

end quadratic_factorization_l1323_132314


namespace quadratic_points_order_l1323_132355

/-- The quadratic function f(x) = x² - 6x + c -/
def f (c : ℝ) (x : ℝ) : ℝ := x^2 - 6*x + c

/-- Theorem: Given points A(-1, y₁), B(1, y₂), C(4, y₃) on the graph of f(x) = x² - 6x + c,
    prove that y₁ > y₂ > y₃ -/
theorem quadratic_points_order (c y₁ y₂ y₃ : ℝ) 
  (h₁ : f c (-1) = y₁)
  (h₂ : f c 1 = y₂)
  (h₃ : f c 4 = y₃) :
  y₁ > y₂ ∧ y₂ > y₃ := by
  sorry

end quadratic_points_order_l1323_132355


namespace largest_prime_factor_of_expression_l1323_132344

theorem largest_prime_factor_of_expression : 
  let expression := 18^4 + 3 * 18^2 + 1 - 17^4
  ∃ (p : ℕ), Nat.Prime p ∧ p ∣ expression ∧ ∀ (q : ℕ), Nat.Prime q → q ∣ expression → q ≤ p ∧ p = 307 :=
by sorry

end largest_prime_factor_of_expression_l1323_132344


namespace rectangle_similarity_symmetry_l1323_132366

/-- A rectangle with width and height -/
structure Rectangle where
  width : ℝ
  height : ℝ

/-- Two rectangles are similar if their aspect ratios are equal -/
def similar (r1 r2 : Rectangle) : Prop :=
  r1.width / r1.height = r2.width / r2.height

/-- A rectangle can be formed from congruent copies of another rectangle -/
def can_form (r1 r2 : Rectangle) : Prop :=
  ∃ (m n p q : ℕ), m * r1.width + n * r1.height = r2.width ∧
                   p * r1.width + q * r1.height = r2.height

theorem rectangle_similarity_symmetry (A B : Rectangle) :
  (∃ C : Rectangle, similar C B ∧ can_form C A) →
  (∃ D : Rectangle, similar D A ∧ can_form D B) :=
by sorry

end rectangle_similarity_symmetry_l1323_132366


namespace quadratic_equation_solution_l1323_132313

theorem quadratic_equation_solution : 
  ∃ x₁ x₂ : ℝ, (x₁ = 2 + Real.sqrt 6 ∧ x₁^2 - 4*x₁ = 2) ∧ 
              (x₂ = 2 - Real.sqrt 6 ∧ x₂^2 - 4*x₂ = 2) :=
by sorry

end quadratic_equation_solution_l1323_132313


namespace min_distance_point_to_line_l1323_132309

theorem min_distance_point_to_line (m n : ℝ) (h1 : m > 0) (h2 : n > 0) (h3 : m * n - m - n = 3) :
  let d := |m + n| / Real.sqrt 2
  d ≥ 3 * Real.sqrt 2 := by
sorry

end min_distance_point_to_line_l1323_132309


namespace room_population_theorem_l1323_132332

theorem room_population_theorem (total : ℕ) (under_21 : ℕ) (over_65 : ℕ) :
  (3 : ℚ) / 7 * total = under_21 →
  50 < total →
  total < 100 →
  under_21 = 24 →
  total = 56 ∧ (over_65 : ℚ) / total = over_65 / 56 := by
  sorry

end room_population_theorem_l1323_132332


namespace min_blocks_for_garden_wall_l1323_132396

/-- Represents the configuration of a garden wall --/
structure WallConfig where
  length : ℕ
  height : ℕ
  blockHeight : ℕ
  shortBlockLength : ℕ
  longBlockLength : ℕ

/-- Calculates the minimum number of blocks required for the wall --/
def minBlocksRequired (config : WallConfig) : ℕ :=
  sorry

/-- The specific wall configuration from the problem --/
def gardenWall : WallConfig :=
  { length := 90
  , height := 8
  , blockHeight := 1
  , shortBlockLength := 2
  , longBlockLength := 3 }

/-- Theorem stating that the minimum number of blocks required is 244 --/
theorem min_blocks_for_garden_wall :
  minBlocksRequired gardenWall = 244 :=
sorry

end min_blocks_for_garden_wall_l1323_132396


namespace tsunami_area_theorem_l1323_132353

/-- Regular tetrahedron with edge length 900 km -/
structure Tetrahedron where
  edge_length : ℝ
  regular : edge_length = 900

/-- Tsunami propagation properties -/
structure Tsunami where
  speed : ℝ
  time : ℝ
  speed_is_300 : speed = 300
  time_is_2 : time = 2

/-- Epicenter location -/
inductive EpicenterLocation
  | FaceCenter
  | EdgeMidpoint

/-- Area covered by tsunami -/
noncomputable def tsunami_area (t : Tetrahedron) (w : Tsunami) (loc : EpicenterLocation) : ℝ :=
  match loc with
  | EpicenterLocation.FaceCenter => 180000 * Real.pi + 270000 * Real.sqrt 3
  | EpicenterLocation.EdgeMidpoint => 720000 * Real.arccos (3/4) + 135000 * Real.sqrt 7

/-- Main theorem -/
theorem tsunami_area_theorem (t : Tetrahedron) (w : Tsunami) :
  (tsunami_area t w EpicenterLocation.FaceCenter = 180000 * Real.pi + 270000 * Real.sqrt 3) ∧
  (tsunami_area t w EpicenterLocation.EdgeMidpoint = 720000 * Real.arccos (3/4) + 135000 * Real.sqrt 7) := by
  sorry


end tsunami_area_theorem_l1323_132353


namespace two_never_appears_l1323_132362

/-- Represents a move in the game -/
def Move (s : List Int) : List Int :=
  -- Implementation details omitted
  sorry

/-- Represents the state of the board after any number of moves -/
inductive BoardState
| initial (n : Nat) : BoardState
| after_move (prev : BoardState) : BoardState

/-- The sequence of numbers on the board -/
def board_sequence (state : BoardState) : List Int :=
  match state with
  | BoardState.initial n => List.range (2*n) -- Simplified representation
  | BoardState.after_move prev => Move (board_sequence prev)

/-- Theorem stating that 2 never appears after any number of moves -/
theorem two_never_appears (n : Nat) (state : BoardState) : 
  2 ∉ board_sequence state :=
sorry

end two_never_appears_l1323_132362


namespace honor_students_count_l1323_132371

theorem honor_students_count 
  (total_students : ℕ) 
  (girls : ℕ) 
  (boys : ℕ) 
  (honor_girls : ℕ) 
  (honor_boys : ℕ) : 
  total_students < 30 →
  total_students = girls + boys →
  (honor_girls : ℚ) / girls = 3 / 13 →
  (honor_boys : ℚ) / boys = 4 / 11 →
  honor_girls + honor_boys = 7 :=
by sorry

end honor_students_count_l1323_132371


namespace white_balls_estimate_l1323_132336

theorem white_balls_estimate (total_balls : ℕ) (total_draws : ℕ) (white_draws : ℕ) 
  (h_total_balls : total_balls = 20)
  (h_total_draws : total_draws = 100)
  (h_white_draws : white_draws = 40) :
  (white_draws : ℚ) / total_draws * total_balls = 8 := by
  sorry

end white_balls_estimate_l1323_132336


namespace quadratic_sum_zero_l1323_132319

/-- A quadratic function f(x) = ax^2 + bx + c with specified properties -/
def QuadraticFunction (a b c : ℝ) : ℝ → ℝ := fun x ↦ a * x^2 + b * x + c

theorem quadratic_sum_zero
  (a b c : ℝ)
  (h_min : ∃ x, ∀ y, QuadraticFunction a b c y ≥ QuadraticFunction a b c x ∧ QuadraticFunction a b c x = 36)
  (h_root1 : QuadraticFunction a b c 1 = 0)
  (h_root5 : QuadraticFunction a b c 5 = 0) :
  a + b + c = 0 := by
  sorry

end quadratic_sum_zero_l1323_132319


namespace leila_bought_two_armchairs_l1323_132394

/-- Represents the living room set purchase --/
structure LivingRoomSet where
  sofaCost : ℕ
  armchairCost : ℕ
  coffeeTableCost : ℕ
  totalCost : ℕ

/-- Calculates the number of armchairs in the living room set --/
def numberOfArmchairs (set : LivingRoomSet) : ℕ :=
  (set.totalCost - set.sofaCost - set.coffeeTableCost) / set.armchairCost

/-- Theorem stating that Leila bought 2 armchairs --/
theorem leila_bought_two_armchairs (set : LivingRoomSet)
    (h1 : set.sofaCost = 1250)
    (h2 : set.armchairCost = 425)
    (h3 : set.coffeeTableCost = 330)
    (h4 : set.totalCost = 2430) :
    numberOfArmchairs set = 2 := by
  sorry

#eval numberOfArmchairs {
  sofaCost := 1250,
  armchairCost := 425,
  coffeeTableCost := 330,
  totalCost := 2430
}

end leila_bought_two_armchairs_l1323_132394


namespace equal_roots_equation_l1323_132349

theorem equal_roots_equation : ∃ x : ℝ, (x - 1) * (x - 1) = 0 ∧ 
  (∀ y : ℝ, (y - 1) * (y - 1) = 0 → y = x) :=
by sorry

end equal_roots_equation_l1323_132349


namespace stable_performance_lower_variance_athlete_a_more_stable_l1323_132338

-- Define the structure for an athlete's performance
structure AthletePerformance where
  average_score : ℝ
  variance : ℝ
  variance_positive : variance > 0

-- Define the notion of stability
def more_stable (a b : AthletePerformance) : Prop :=
  a.variance < b.variance

-- Theorem statement
theorem stable_performance_lower_variance 
  (a b : AthletePerformance) 
  (h_equal_avg : a.average_score = b.average_score) :
  more_stable a b ↔ a.variance < b.variance :=
sorry

-- Specific instance for the given problem
def athlete_a : AthletePerformance := {
  average_score := 9
  variance := 1.2
  variance_positive := by norm_num
}

def athlete_b : AthletePerformance := {
  average_score := 9
  variance := 2.4
  variance_positive := by norm_num
}

-- Theorem application to the specific instance
theorem athlete_a_more_stable : more_stable athlete_a athlete_b :=
sorry

end stable_performance_lower_variance_athlete_a_more_stable_l1323_132338


namespace probability_ten_nine_eight_sequence_l1323_132387

theorem probability_ten_nine_eight_sequence (deck : Nat) (tens : Nat) (nines : Nat) (eights : Nat) :
  deck = 52 →
  tens = 4 →
  nines = 4 →
  eights = 4 →
  (tens / deck) * (nines / (deck - 1)) * (eights / (deck - 2)) = 4 / 33150 := by
  sorry

end probability_ten_nine_eight_sequence_l1323_132387


namespace right_triangle_in_segment_sets_l1323_132317

/-- Check if three line segments can form a right-angled triangle -/
def is_right_triangle (a b c : ℝ) : Prop :=
  a > 0 ∧ b > 0 ∧ c > 0 ∧ a^2 + b^2 = c^2

/-- The given sets of line segments -/
def segment_sets : List (ℝ × ℝ × ℝ) :=
  [(1, 2, 4), (3, 4, 5), (4, 6, 8), (5, 7, 11)]

theorem right_triangle_in_segment_sets :
  ∃! (a b c : ℝ), (a, b, c) ∈ segment_sets ∧ is_right_triangle a b c :=
by sorry

end right_triangle_in_segment_sets_l1323_132317


namespace lcm_problem_l1323_132350

theorem lcm_problem (a b : ℕ+) (h1 : Nat.gcd a b = 20) (h2 : a * b = 2560) :
  Nat.lcm a b = 128 := by
  sorry

end lcm_problem_l1323_132350


namespace circle_y_axis_intersection_sum_l1323_132395

/-- A circle with center (a, b) and radius r -/
structure Circle where
  a : ℝ
  b : ℝ
  r : ℝ

/-- The sum of y-coordinates of intersection points between a circle and the y-axis -/
def sumYIntersections (c : Circle) : ℝ :=
  2 * c.b

/-- Theorem: For a circle with center (-3, -4) and radius 7, 
    the sum of y-coordinates of its intersection points with the y-axis is -8 -/
theorem circle_y_axis_intersection_sum :
  ∃ (c : Circle), c.a = -3 ∧ c.b = -4 ∧ c.r = 7 ∧ sumYIntersections c = -8 := by
  sorry


end circle_y_axis_intersection_sum_l1323_132395


namespace clubsuit_calculation_l1323_132385

/-- Custom operation ⊗ for real numbers -/
def clubsuit (a b : ℝ) : ℝ := (a + b)^2 - (a - b)^2

/-- Theorem stating that 5 ⊗ (7 ⊗ 8) = 4480 -/
theorem clubsuit_calculation : clubsuit 5 (clubsuit 7 8) = 4480 := by
  sorry

end clubsuit_calculation_l1323_132385


namespace archer_probability_l1323_132388

theorem archer_probability (p10 p9 p8 : ℝ) (h1 : p10 = 0.2) (h2 : p9 = 0.3) (h3 : p8 = 0.3) :
  1 - p10 - p9 - p8 = 0.2 := by
  sorry

end archer_probability_l1323_132388


namespace ferris_break_length_is_correct_l1323_132302

/-- Represents the job completion scenario with Audrey and Ferris --/
structure JobCompletion where
  audrey_solo_time : ℝ
  ferris_solo_time : ℝ
  collaboration_time : ℝ
  ferris_break_count : ℕ

/-- Calculates the length of each of Ferris' breaks in minutes --/
def ferris_break_length (job : JobCompletion) : ℝ :=
  2.5

/-- Theorem stating that Ferris' break length is 2.5 minutes under the given conditions --/
theorem ferris_break_length_is_correct (job : JobCompletion) 
  (h1 : job.audrey_solo_time = 4)
  (h2 : job.ferris_solo_time = 3)
  (h3 : job.collaboration_time = 2)
  (h4 : job.ferris_break_count = 6) :
  ferris_break_length job = 2.5 := by
  sorry

#eval ferris_break_length { audrey_solo_time := 4, ferris_solo_time := 3, collaboration_time := 2, ferris_break_count := 6 }

end ferris_break_length_is_correct_l1323_132302


namespace balls_in_boxes_l1323_132393

def num_balls : ℕ := 6
def num_boxes : ℕ := 3

theorem balls_in_boxes :
  (num_boxes ^ num_balls : ℕ) = 729 := by
  sorry

end balls_in_boxes_l1323_132393


namespace water_boiling_time_l1323_132368

/-- Time for water to boil away given initial conditions -/
theorem water_boiling_time 
  (T₀ : ℝ) (Tₘ : ℝ) (t : ℝ) (c : ℝ) (L : ℝ)
  (h₁ : T₀ = 20)
  (h₂ : Tₘ = 100)
  (h₃ : t = 10)
  (h₄ : c = 4200)
  (h₅ : L = 2.3e6) :
  ∃ t₁ : ℝ, t₁ ≥ 67.5 ∧ t₁ < 68.5 :=
by sorry

end water_boiling_time_l1323_132368


namespace inverse_proportion_m_value_l1323_132358

theorem inverse_proportion_m_value : 
  ∃! m : ℝ, m^2 - 5 = -1 ∧ m + 2 ≠ 0 :=
by
  sorry

end inverse_proportion_m_value_l1323_132358


namespace triangle_perimeter_l1323_132384

theorem triangle_perimeter (a b c : ℝ) (ha : a = Real.sqrt 8) (hb : b = Real.sqrt 18) (hc : c = Real.sqrt 32) :
  a + b + c = 9 * Real.sqrt 2 := by
  sorry

end triangle_perimeter_l1323_132384


namespace remainder_of_1279_divided_by_89_l1323_132345

theorem remainder_of_1279_divided_by_89 : Nat.mod 1279 89 = 33 := by
  sorry

end remainder_of_1279_divided_by_89_l1323_132345


namespace find_c_l1323_132305

theorem find_c (a b c : ℝ) : 
  (∀ x, (x + 3) * (x + b) = x^2 + c*x + 15) → c = 8 := by
  sorry

end find_c_l1323_132305


namespace greatest_b_not_in_range_l1323_132361

/-- The quadratic function f(x) = x^2 + bx + 20 -/
def f (b : ℤ) (x : ℝ) : ℝ := x^2 + b*x + 20

/-- Predicate that checks if -9 is not in the range of f for a given b -/
def not_in_range (b : ℤ) : Prop := ∀ x : ℝ, f b x ≠ -9

/-- The theorem stating that 10 is the greatest integer b such that -9 is not in the range of f -/
theorem greatest_b_not_in_range : 
  (not_in_range 10 ∧ ∀ b : ℤ, b > 10 → ¬(not_in_range b)) := by sorry

end greatest_b_not_in_range_l1323_132361


namespace hammer_weight_exceeds_ton_on_10th_day_l1323_132376

def hammer_weight (day : ℕ) : ℝ :=
  7 * (2 ^ (day - 1))

theorem hammer_weight_exceeds_ton_on_10th_day :
  (∀ d : ℕ, d < 10 → hammer_weight d ≤ 2000) ∧
  hammer_weight 10 > 2000 :=
by sorry

end hammer_weight_exceeds_ton_on_10th_day_l1323_132376
