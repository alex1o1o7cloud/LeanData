import Mathlib

namespace NUMINAMATH_CALUDE_geometric_sequence_inequality_l2836_283607

theorem geometric_sequence_inequality (a : Fin 8 → ℝ) (q : ℝ) :
  (∀ i : Fin 8, a i > 0) →
  (∀ i : Fin 7, a (i + 1) = a i * q) →
  q ≠ 1 →
  a 0 + a 7 > a 3 + a 4 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_inequality_l2836_283607


namespace NUMINAMATH_CALUDE_batsman_average_is_59_l2836_283623

/-- Calculates the batting average given the total innings, highest score, 
    average excluding highest and lowest scores, and the difference between highest and lowest scores. -/
def battingAverage (totalInnings : ℕ) (highestScore : ℕ) (averageExcludingExtremes : ℕ) (scoreDifference : ℕ) : ℚ :=
  let lowestScore := highestScore - scoreDifference
  let totalScore := (totalInnings - 2) * averageExcludingExtremes + highestScore + lowestScore
  totalScore / totalInnings

/-- Theorem stating that under the given conditions, the batting average is 59 runs. -/
theorem batsman_average_is_59 :
  battingAverage 46 156 58 150 = 59 := by sorry

end NUMINAMATH_CALUDE_batsman_average_is_59_l2836_283623


namespace NUMINAMATH_CALUDE_modular_inverse_of_5_mod_23_l2836_283680

theorem modular_inverse_of_5_mod_23 :
  ∃ a : ℕ, a ≤ 22 ∧ (5 * a) % 23 = 1 ∧ a = 14 := by
  sorry

end NUMINAMATH_CALUDE_modular_inverse_of_5_mod_23_l2836_283680


namespace NUMINAMATH_CALUDE_pi_fourth_in_range_of_g_l2836_283632

noncomputable def g (x : ℝ) : ℝ := Real.arctan (2 * x) + Real.arctan ((2 - x) / (2 + x))

theorem pi_fourth_in_range_of_g : ∃ (x : ℝ), g x = π / 4 := by sorry

end NUMINAMATH_CALUDE_pi_fourth_in_range_of_g_l2836_283632


namespace NUMINAMATH_CALUDE_cyclic_fraction_sum_l2836_283605

theorem cyclic_fraction_sum (x y z w t : ℝ) 
  (h_pos : x > 0 ∧ y > 0 ∧ z > 0 ∧ w > 0)
  (h_diff : x ≠ y ∧ y ≠ z ∧ z ≠ w ∧ w ≠ x)
  (h_eq : x + 1/y = t ∧ y + 1/z = t ∧ z + 1/w = t ∧ w + 1/x = t) :
  t = Real.sqrt 2 := by
sorry

end NUMINAMATH_CALUDE_cyclic_fraction_sum_l2836_283605


namespace NUMINAMATH_CALUDE_unique_prime_triplet_l2836_283697

/-- A function that checks if a number is prime -/
def isPrime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d > 1 → d < n → ¬(n % d = 0)

/-- The theorem stating that 3 is the only integer p such that p, p+2, and p+4 are all prime -/
theorem unique_prime_triplet : 
  ∀ p : ℤ, (isPrime p.natAbs ∧ isPrime (p + 2).natAbs ∧ isPrime (p + 4).natAbs) ↔ p = 3 :=
sorry

end NUMINAMATH_CALUDE_unique_prime_triplet_l2836_283697


namespace NUMINAMATH_CALUDE_min_steps_to_eliminate_zeroes_l2836_283692

/-- Represents the state of the blackboard with zeroes and ones -/
structure BlackboardState where
  zeroes : Nat
  ones : Nat

/-- Defines a step operation on the blackboard state -/
def step (state : BlackboardState) : BlackboardState :=
  { zeroes := state.zeroes - 1, ones := state.ones + 1 }

/-- The initial state of the blackboard -/
def initial_state : BlackboardState := { zeroes := 150, ones := 151 }

/-- Predicate to check if a state has no zeroes -/
def no_zeroes (state : BlackboardState) : Prop := state.zeroes = 0

/-- The theorem to be proved -/
theorem min_steps_to_eliminate_zeroes :
  ∃ (n : Nat), n = 150 ∧ no_zeroes (n.iterate step initial_state) ∧
  ∀ (m : Nat), m < n → ¬no_zeroes (m.iterate step initial_state) :=
sorry

end NUMINAMATH_CALUDE_min_steps_to_eliminate_zeroes_l2836_283692


namespace NUMINAMATH_CALUDE_wyatt_envelopes_l2836_283644

/-- The number of blue envelopes Wyatt has -/
def blue_envelopes : ℕ := 10

/-- The difference between blue and yellow envelopes -/
def envelope_difference : ℕ := 4

/-- The total number of envelopes Wyatt has -/
def total_envelopes : ℕ := blue_envelopes + (blue_envelopes - envelope_difference)

/-- Theorem stating the total number of envelopes Wyatt has -/
theorem wyatt_envelopes : total_envelopes = 16 := by sorry

end NUMINAMATH_CALUDE_wyatt_envelopes_l2836_283644


namespace NUMINAMATH_CALUDE_pond_filling_time_l2836_283678

/-- Proves the time required to fill a pond under drought conditions -/
theorem pond_filling_time (pond_capacity : ℝ) (normal_rate : ℝ) (drought_factor : ℝ) : 
  pond_capacity = 200 →
  normal_rate = 6 →
  drought_factor = 2/3 →
  (pond_capacity / (normal_rate * drought_factor) = 50) :=
by
  sorry

end NUMINAMATH_CALUDE_pond_filling_time_l2836_283678


namespace NUMINAMATH_CALUDE_characterize_satisfying_functions_l2836_283689

/-- A function satisfying the given functional equation -/
def SatisfiesEquation (f : ℤ → ℤ) : Prop :=
  ∀ a b : ℤ, f (2 * a) + 2 * f b = f (f (a + b))

/-- The theorem stating the characterization of functions satisfying the equation -/
theorem characterize_satisfying_functions :
  ∀ f : ℤ → ℤ, SatisfiesEquation f →
    (∀ n : ℤ, f n = 0) ∨ (∃ K : ℤ, ∀ n : ℤ, f n = 2 * n + K) := by
  sorry

end NUMINAMATH_CALUDE_characterize_satisfying_functions_l2836_283689


namespace NUMINAMATH_CALUDE_points_collinear_and_m_values_l2836_283663

noncomputable section

-- Define the points and vectors
def O : ℝ × ℝ := (0, 0)
def A (x : ℝ) : ℝ × ℝ := (1, Real.cos x)
def B (x : ℝ) : ℝ × ℝ := (1 + Real.sin x, Real.cos x)
def OA (x : ℝ) : ℝ × ℝ := A x
def OB (x : ℝ) : ℝ × ℝ := B x
def OC (x : ℝ) : ℝ × ℝ := (1/3 : ℝ) • (OA x) + (2/3 : ℝ) • (OB x)

-- Define the function f
def f (x m : ℝ) : ℝ :=
  (OA x).1 * (OC x).1 + (OA x).2 * (OC x).2 +
  (2*m + 1/3) * Real.sqrt ((B x).1 - (A x).1)^2 + ((B x).2 - (A x).2)^2 +
  m^2

-- Theorem statement
theorem points_collinear_and_m_values (x : ℝ) (h : x ∈ Set.Icc 0 (Real.pi / 2)) :
  (∃ t : ℝ, OC x = t • OA x + (1 - t) • OB x) ∧
  (∃ m : ℝ, (∀ y ∈ Set.Icc 0 (Real.pi / 2), f y m ≥ 5) ∧ f x m = 5 ∧ (m = -3 ∨ m = Real.sqrt 3)) :=
sorry

end NUMINAMATH_CALUDE_points_collinear_and_m_values_l2836_283663


namespace NUMINAMATH_CALUDE_largest_special_number_l2836_283634

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d > 1 → d < n → ¬(n % d = 0)

def digits_distinct (n : ℕ) : Prop :=
  let d1 := n / 1000
  let d2 := (n / 100) % 10
  let d3 := (n / 10) % 10
  let d4 := n % 10
  d1 ≠ d2 ∧ d1 ≠ d3 ∧ d1 ≠ d4 ∧ d2 ≠ d3 ∧ d2 ≠ d4 ∧ d3 ≠ d4

theorem largest_special_number :
  ∀ n : ℕ,
  n ≥ 1000 ∧ n < 10000 →
  digits_distinct n →
  is_prime (n / 100) →
  is_prime ((n / 1000) * 10 + (n % 100) / 10) →
  is_prime ((n / 1000) * 10 + (n % 10)) →
  n % 3 = 0 →
  ¬is_prime n →
  n ≤ 4731 :=
sorry

end NUMINAMATH_CALUDE_largest_special_number_l2836_283634


namespace NUMINAMATH_CALUDE_abc_inequality_l2836_283648

theorem abc_inequality (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c)
  (h : a^2 + b^2 - a*b = c^2) : (a - c) * (b - c) ≤ 0 := by
  sorry

end NUMINAMATH_CALUDE_abc_inequality_l2836_283648


namespace NUMINAMATH_CALUDE_triangle_side_lengths_l2836_283662

/-- Given a triangle with angle α, internal angle bisector length f, and external angle bisector length g,
    calculate the side lengths a, b, and c. -/
theorem triangle_side_lengths
  (α : Real) (f g : ℝ) (h_α : 0 < α ∧ α < π) (h_f : f > 0) (h_g : g > 0) :
  ∃ (a b c : ℝ),
    a = (f * g * Real.sqrt (f^2 + g^2) * Real.sin α) / (g^2 * (Real.cos (α/2))^2 - f^2 * (Real.sin (α/2))^2) ∧
    b = (f * g) / (g * Real.cos (α/2) + f * Real.sin (α/2)) ∧
    c = (f * g) / (g * Real.cos (α/2) - f * Real.sin (α/2)) ∧
    0 < a ∧ 0 < b ∧ 0 < c ∧
    a + b > c ∧ b + c > a ∧ c + a > b :=
by sorry

end NUMINAMATH_CALUDE_triangle_side_lengths_l2836_283662


namespace NUMINAMATH_CALUDE_cricket_team_age_theorem_l2836_283673

def cricket_team_age_problem (team_size : ℕ) (captain_age : ℕ) (wicket_keeper_age_diff : ℕ) 
  (remaining_players_age_diff : ℕ) (bowlers_count : ℕ) 
  (bowlers_min_age : ℕ) (bowlers_max_age : ℕ) : Prop :=
  let wicket_keeper_age := captain_age + wicket_keeper_age_diff
  let total_age := team_size * 30
  let captain_wicket_keeper_age := captain_age + wicket_keeper_age
  let remaining_players := team_size - 2
  total_age = captain_wicket_keeper_age + remaining_players * (30 - remaining_players_age_diff) ∧
  bowlers_min_age * bowlers_count ≤ bowlers_count * 30 ∧
  bowlers_count * 30 ≤ bowlers_max_age * bowlers_count

theorem cricket_team_age_theorem : 
  cricket_team_age_problem 11 24 3 1 5 18 22 := by
  sorry

end NUMINAMATH_CALUDE_cricket_team_age_theorem_l2836_283673


namespace NUMINAMATH_CALUDE_quadrilateral_circle_condition_l2836_283649

-- Define the lines
def line1 (a x y : ℝ) : Prop := (a + 2) * x + (1 - a) * y - 3 = 0
def line2 (a x y : ℝ) : Prop := (a - 1) * x + (2 * a + 3) * y + 2 = 0

-- Define the property of forming a quadrilateral with coordinate axes
def forms_quadrilateral (a : ℝ) : Prop :=
  ∃ x1 y1 x2 y2 : ℝ, 
    line1 a x1 0 ∧ line1 a 0 y1 ∧ line2 a x2 0 ∧ line2 a 0 y2

-- Define the property of having a circumscribed circle
def has_circumscribed_circle (a : ℝ) : Prop :=
  forms_quadrilateral a → 
    (a + 2) * (a - 1) + (1 - a) * (2 * a + 3) = 0

-- The theorem to prove
theorem quadrilateral_circle_condition (a : ℝ) :
  forms_quadrilateral a → has_circumscribed_circle a → (a = 1 ∨ a = -1) :=
sorry

end NUMINAMATH_CALUDE_quadrilateral_circle_condition_l2836_283649


namespace NUMINAMATH_CALUDE_kitchen_hours_theorem_l2836_283653

/-- The minimum number of hours required to produce a given number of large and small cakes -/
def min_hours_required (num_helpers : ℕ) (large_cakes_per_hour : ℕ) (small_cakes_per_hour : ℕ) (large_cakes_needed : ℕ) (small_cakes_needed : ℕ) : ℕ :=
  max 
    (large_cakes_needed / (num_helpers * large_cakes_per_hour))
    (small_cakes_needed / (num_helpers * small_cakes_per_hour))

theorem kitchen_hours_theorem :
  min_hours_required 10 2 35 20 700 = 2 := by
  sorry

end NUMINAMATH_CALUDE_kitchen_hours_theorem_l2836_283653


namespace NUMINAMATH_CALUDE_triangle_inequalities_l2836_283606

/-- 
For any triangle ABC, we define:
- ha, hb, hc as the altitudes
- ra, rb, rc as the exradii
- r as the inradius
-/
theorem triangle_inequalities (A B C : Point) 
  (ha hb hc : ℝ) (ra rb rc : ℝ) (r : ℝ) :
  (ha > 0 ∧ hb > 0 ∧ hc > 0) →
  (ra > 0 ∧ rb > 0 ∧ rc > 0) →
  (r > 0) →
  (ha * hb * hc ≥ 27 * r^3) ∧ (ra * rb * rc ≥ 27 * r^3) := by
  sorry

end NUMINAMATH_CALUDE_triangle_inequalities_l2836_283606


namespace NUMINAMATH_CALUDE_unique_modular_congruence_l2836_283694

theorem unique_modular_congruence :
  ∃! n : ℕ, 0 ≤ n ∧ n ≤ 8 ∧ n ≡ 100000 [ZMOD 9] ∧ n = 1 := by
  sorry

end NUMINAMATH_CALUDE_unique_modular_congruence_l2836_283694


namespace NUMINAMATH_CALUDE_pet_store_snakes_l2836_283631

/-- The number of snakes in a pet store -/
theorem pet_store_snakes (num_cages : ℕ) (snakes_per_cage : ℕ) : 
  num_cages = 2 → snakes_per_cage = 2 → num_cages * snakes_per_cage = 4 := by
  sorry

#check pet_store_snakes

end NUMINAMATH_CALUDE_pet_store_snakes_l2836_283631


namespace NUMINAMATH_CALUDE_no_solution_exists_l2836_283665

theorem no_solution_exists : ¬∃ (x : ℝ), x ≥ 0 ∧ 
  (42 + x = 3 * (8 + x)) ∧ (42 + x = 2 * (10 + x)) := by
  sorry

end NUMINAMATH_CALUDE_no_solution_exists_l2836_283665


namespace NUMINAMATH_CALUDE_parallelogram_d_not_two_neg_two_l2836_283654

/-- Definition of a point in 2D space -/
def Point := ℝ × ℝ

/-- Definition of a parallelogram -/
def is_parallelogram (A B C D : Point) : Prop :=
  (A.1 + C.1 = B.1 + D.1) ∧ (A.2 + C.2 = B.2 + D.2)

/-- Theorem: If ABCD is a parallelogram with A(0,0), B(2,2), C(3,0), then D cannot be (2,-2) -/
theorem parallelogram_d_not_two_neg_two :
  let A : Point := (0, 0)
  let B : Point := (2, 2)
  let C : Point := (3, 0)
  let D : Point := (2, -2)
  ¬(is_parallelogram A B C D) := by
  sorry


end NUMINAMATH_CALUDE_parallelogram_d_not_two_neg_two_l2836_283654


namespace NUMINAMATH_CALUDE_alpha_plus_beta_l2836_283690

theorem alpha_plus_beta (α β : ℝ) : 
  (∀ x : ℝ, (x - α) / (x + β) = (x^2 - 64*x + 975) / (x^2 + 99*x - 2200)) → 
  α + β = 138 := by
  sorry

end NUMINAMATH_CALUDE_alpha_plus_beta_l2836_283690


namespace NUMINAMATH_CALUDE_intersection_nonempty_iff_m_leq_neg_one_l2836_283613

/-- Sets A and B are defined as follows:
    A = {(x, y) | y = x^2 + mx + 2}
    B = {(x, y) | x - y + 1 = 0 and 0 ≤ x ≤ 2}
    This theorem states that A ∩ B is non-empty if and only if m ≤ -1 -/
theorem intersection_nonempty_iff_m_leq_neg_one (m : ℝ) :
  (∃ x y : ℝ, y = x^2 + m*x + 2 ∧ x - y + 1 = 0 ∧ 0 ≤ x ∧ x ≤ 2) ↔ m ≤ -1 := by
  sorry

end NUMINAMATH_CALUDE_intersection_nonempty_iff_m_leq_neg_one_l2836_283613


namespace NUMINAMATH_CALUDE_right_triangle_equations_l2836_283669

/-- A right-angled triangle ABC with specified coordinates -/
structure RightTriangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ
  is_right_angle : B.1 = 1 ∧ B.2 = Real.sqrt 3
  A_on_x_axis : A = (-2, 0)
  C_on_x_axis : C.2 = 0

/-- The equation of line BC in the form ax + by + c = 0 -/
def line_equation (a b c : ℝ) (x y : ℝ) : Prop :=
  a * x + b * y + c = 0

/-- The equation of line OB (median to hypotenuse) in the form y = kx -/
def median_equation (k : ℝ) (x y : ℝ) : Prop :=
  y = k * x

theorem right_triangle_equations (t : RightTriangle) :
  (∃ (a b c : ℝ), a = Real.sqrt 3 ∧ b = 1 ∧ c = -2 * Real.sqrt 3 ∧
    ∀ (x y : ℝ), line_equation a b c x y ↔ (x, y) ∈ ({t.B, t.C} : Set (ℝ × ℝ))) ∧
  (∃ (k : ℝ), k = Real.sqrt 3 ∧
    ∀ (x y : ℝ), median_equation k x y ↔ (x, y) ∈ ({(0, 0), t.B} : Set (ℝ × ℝ))) :=
sorry

end NUMINAMATH_CALUDE_right_triangle_equations_l2836_283669


namespace NUMINAMATH_CALUDE_first_player_strategy_guarantees_six_no_root_equations_l2836_283699

/-- Represents a quadratic equation ax² + bx + c = 0 -/
structure QuadraticEquation where
  a : ℝ
  b : ℝ
  c : ℝ
  a_nonzero : a ≠ 0
  b_nonzero : b ≠ 0
  c_nonzero : c ≠ 0

/-- Represents the game state -/
structure GameState where
  equations : List QuadraticEquation
  num_equations : Nat
  num_equations_eq_11 : num_equations = 11

/-- Represents a player's strategy -/
def Strategy := GameState → QuadraticEquation

/-- Determines if a quadratic equation has no real roots -/
def has_no_real_roots (eq : QuadraticEquation) : Prop :=
  eq.b * eq.b - 4 * eq.a * eq.c < 0

/-- The maximum number of equations without real roots that the first player can guarantee -/
def max_no_root_equations : Nat := 6

/-- The main theorem to be proved -/
theorem first_player_strategy_guarantees_six_no_root_equations
  (initial_state : GameState)
  (first_player_strategy : Strategy)
  (second_player_strategy : Strategy) :
  ∃ (final_state : GameState),
    (final_state.num_equations = initial_state.num_equations) ∧
    (∀ eq ∈ final_state.equations, has_no_real_roots eq) ∧
    (final_state.equations.length ≥ max_no_root_equations) :=
  sorry

#check first_player_strategy_guarantees_six_no_root_equations

end NUMINAMATH_CALUDE_first_player_strategy_guarantees_six_no_root_equations_l2836_283699


namespace NUMINAMATH_CALUDE_equation_solution_l2836_283641

theorem equation_solution : ∃! x : ℝ, x^2 - ⌊x⌋ = 3 ∧ x = (4 : ℝ)^(1/3) := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l2836_283641


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l2836_283685

/-- Given an arithmetic sequence {a_n} with S_n as the sum of its first n terms,
    if -a_{2015} < a_1 < -a_{2016}, then S_{2015} > 0 and S_{2016} < 0. -/
theorem arithmetic_sequence_sum (a : ℕ → ℝ) (S : ℕ → ℝ)
  (h_arithmetic : ∀ n, a (n + 1) - a n = a (n + 2) - a (n + 1))
  (h_sum : ∀ n, S n = n * (a 1 + a n) / 2)
  (h_inequality : -a 2015 < a 1 ∧ a 1 < -a 2016) :
  S 2015 > 0 ∧ S 2016 < 0 := by
  sorry


end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l2836_283685


namespace NUMINAMATH_CALUDE_games_purchase_l2836_283635

theorem games_purchase (initial_amount : ℕ) (spent_amount : ℕ) (game_cost : ℕ) : 
  initial_amount = 42 → spent_amount = 10 → game_cost = 8 → 
  (initial_amount - spent_amount) / game_cost = 4 := by
  sorry

end NUMINAMATH_CALUDE_games_purchase_l2836_283635


namespace NUMINAMATH_CALUDE_largest_c_value_l2836_283687

theorem largest_c_value (c : ℝ) (h : (3 * c + 4) * (c - 2) = 9 * c) : 
  ∀ x : ℝ, (3 * x + 4) * (x - 2) = 9 * x → x ≤ 4 := by sorry

end NUMINAMATH_CALUDE_largest_c_value_l2836_283687


namespace NUMINAMATH_CALUDE_point_in_second_quadrant_exists_l2836_283656

theorem point_in_second_quadrant_exists : ∃ (x y : ℤ), 
  x < 0 ∧ 
  y > 0 ∧ 
  y ≤ x + 4 ∧ 
  x = -1 ∧ 
  y = 3 := by
sorry

end NUMINAMATH_CALUDE_point_in_second_quadrant_exists_l2836_283656


namespace NUMINAMATH_CALUDE_quadratic_roots_sum_product_l2836_283679

theorem quadratic_roots_sum_product (m : ℝ) (x₁ x₂ : ℝ) : 
  (∀ x, x^2 + m*x - 3 = 0 ↔ x = x₁ ∨ x = x₂) →
  x₁ + x₂ - x₁*x₂ = 5 →
  m = -2 := by
sorry

end NUMINAMATH_CALUDE_quadratic_roots_sum_product_l2836_283679


namespace NUMINAMATH_CALUDE_rational_solutions_quadratic_l2836_283675

theorem rational_solutions_quadratic (k : ℕ+) : 
  (∃ x : ℚ, k * x^2 + 22 * x + k = 0) ↔ k = 11 := by
  sorry

end NUMINAMATH_CALUDE_rational_solutions_quadratic_l2836_283675


namespace NUMINAMATH_CALUDE_count_words_l2836_283640

/-- The number of letters in the alphabet -/
def alphabet_size : ℕ := 6

/-- The maximum word length -/
def max_word_length : ℕ := 3

/-- Counts the number of words with exactly one letter -/
def one_letter_words : ℕ := alphabet_size

/-- Counts the number of words with exactly two letters -/
def two_letter_words : ℕ := alphabet_size * alphabet_size

/-- Counts the number of three-letter words with all letters the same -/
def three_same_letter_words : ℕ := alphabet_size

/-- Counts the number of three-letter words with exactly two letters the same -/
def three_two_same_letter_words : ℕ := alphabet_size * (alphabet_size - 1) * 3

/-- The total number of words in the language -/
def total_words : ℕ := one_letter_words + two_letter_words + three_same_letter_words + three_two_same_letter_words

/-- Theorem stating that the total number of words is 138 -/
theorem count_words : total_words = 138 := by
  sorry

end NUMINAMATH_CALUDE_count_words_l2836_283640


namespace NUMINAMATH_CALUDE_z_120_20_bounds_l2836_283622

/-- Z_{2k}^s is the s-th member from the center in the 2k-th row -/
def Z (k : ℕ) (s : ℕ) : ℝ := sorry

/-- w_{2k} is a function of k -/
def w (k : ℕ) : ℝ := sorry

/-- Main theorem: Z_{120}^{20} is bounded between 0.012 and 0.016 -/
theorem z_120_20_bounds :
  0.012 < Z 60 10 ∧ Z 60 10 < 0.016 :=
sorry

end NUMINAMATH_CALUDE_z_120_20_bounds_l2836_283622


namespace NUMINAMATH_CALUDE_negation_of_universal_proposition_l2836_283629

theorem negation_of_universal_proposition :
  (¬ ∀ x : ℝ, x^3 - x^2 + 1 ≤ 0) ↔ (∃ x₀ : ℝ, x₀^3 - x₀^2 + 1 > 0) := by sorry

end NUMINAMATH_CALUDE_negation_of_universal_proposition_l2836_283629


namespace NUMINAMATH_CALUDE_valid_subcommittee_count_l2836_283626

def total_members : ℕ := 12
def teacher_count : ℕ := 6
def subcommittee_size : ℕ := 5
def min_teachers : ℕ := 2

def subcommittee_count : ℕ := 696

theorem valid_subcommittee_count :
  (total_members.choose subcommittee_size) -
  ((teacher_count.choose 0) * ((total_members - teacher_count).choose subcommittee_size) +
   (teacher_count.choose 1) * ((total_members - teacher_count).choose (subcommittee_size - 1)))
  = subcommittee_count :=
by sorry

end NUMINAMATH_CALUDE_valid_subcommittee_count_l2836_283626


namespace NUMINAMATH_CALUDE_trigonometric_simplification_l2836_283658

theorem trigonometric_simplification (α : ℝ) : 
  3.4113 * Real.sin α * Real.cos (3 * α) + 
  9 * Real.sin α * Real.cos α - 
  Real.sin (3 * α) * Real.cos (3 * α) - 
  3 * Real.sin (3 * α) * Real.cos α = 
  2 * (Real.sin (2 * α))^3 := by sorry

end NUMINAMATH_CALUDE_trigonometric_simplification_l2836_283658


namespace NUMINAMATH_CALUDE_height_distribution_study_l2836_283618

-- Define the type for students
def Student : Type := Unit

-- Define the school population
def schoolPopulation : Finset Student := sorry

-- Define the sample of measured students
def measuredSample : Finset Student := sorry

-- State the theorem
theorem height_distribution_study :
  (Finset.card schoolPopulation = 240) ∧
  (∀ s : Student, s ∈ schoolPopulation) ∧
  (measuredSample ⊆ schoolPopulation) ∧
  (Finset.card measuredSample = 40) →
  (Finset.card schoolPopulation = 240) ∧
  (∀ s : Student, s ∈ schoolPopulation → s = s) ∧
  (measuredSample = measuredSample) ∧
  (Finset.card measuredSample = 40) := by
  sorry

end NUMINAMATH_CALUDE_height_distribution_study_l2836_283618


namespace NUMINAMATH_CALUDE_box_third_dimension_l2836_283683

/-- Proves that the third dimension of a rectangular box is 6 cm, given specific conditions -/
theorem box_third_dimension (num_cubes : ℕ) (cube_volume : ℝ) (length width : ℝ) :
  num_cubes = 24 →
  cube_volume = 27 →
  length = 9 →
  width = 12 →
  (num_cubes : ℝ) * cube_volume = length * width * 6 :=
by sorry

end NUMINAMATH_CALUDE_box_third_dimension_l2836_283683


namespace NUMINAMATH_CALUDE_area_of_overlapping_squares_l2836_283630

/-- Represents a square in a 2D plane -/
structure Square where
  sideLength : ℝ
  center : ℝ × ℝ

/-- Calculates the area of the region covered by two overlapping squares -/
def areaCoveredByOverlappingSquares (s1 s2 : Square) : ℝ :=
  sorry

/-- Theorem stating the area covered by two specific overlapping squares -/
theorem area_of_overlapping_squares :
  let s1 := Square.mk 12 (0, 0)
  let s2 := Square.mk 12 (6, 6)
  areaCoveredByOverlappingSquares s1 s2 = 144 := by
  sorry

end NUMINAMATH_CALUDE_area_of_overlapping_squares_l2836_283630


namespace NUMINAMATH_CALUDE_dads_age_l2836_283636

theorem dads_age (son_age : ℕ) (age_difference : ℕ) : 
  son_age = 9 →
  age_difference = 27 →
  (4 : ℕ) * son_age + age_difference = 63 := by
sorry

end NUMINAMATH_CALUDE_dads_age_l2836_283636


namespace NUMINAMATH_CALUDE_solution_values_l2836_283603

def has_55_solutions (n : ℕ+) : Prop :=
  (Finset.filter (fun (x, y, z) => 3 * x + 3 * y + z = n ∧ x > 0 ∧ y > 0 ∧ z > 0)
    (Finset.product (Finset.range n) (Finset.product (Finset.range n) (Finset.range n)))).card = 55

theorem solution_values (n : ℕ+) (h : has_55_solutions n) : n = 34 ∨ n = 37 := by
  sorry

end NUMINAMATH_CALUDE_solution_values_l2836_283603


namespace NUMINAMATH_CALUDE_reciprocal_equal_self_is_set_l2836_283643

def reciprocal_equal_self (x : ℝ) : Prop := x ≠ 0 ∧ 1 / x = x

def reciprocal_equal_self_set : Set ℝ := {x : ℝ | reciprocal_equal_self x}

theorem reciprocal_equal_self_is_set : 
  ∃ (S : Set ℝ), ∀ x : ℝ, x ∈ S ↔ reciprocal_equal_self x :=
sorry

end NUMINAMATH_CALUDE_reciprocal_equal_self_is_set_l2836_283643


namespace NUMINAMATH_CALUDE_min_colors_regular_ngon_l2836_283668

/-- 
Represents a coloring of sides and diagonals in a regular n-gon.
The coloring is valid if any two segments sharing a common point have different colors.
-/
def ValidColoring (n : ℕ) := 
  { coloring : (Fin n × Fin n) → ℕ // 
    ∀ (i j k : Fin n), i ≠ j → i ≠ k → j ≠ k → 
    coloring (i, j) ≠ coloring (i, k) ∧ 
    coloring (i, j) ≠ coloring (j, k) ∧ 
    coloring (i, k) ≠ coloring (j, k) }

/-- 
The minimum number of colors needed for a valid coloring of a regular n-gon 
is equal to n.
-/
theorem min_colors_regular_ngon (n : ℕ) (h : n ≥ 3) : 
  (∃ (c : ValidColoring n), ∀ (i j : Fin n), c.val (i, j) < n) ∧ 
  (∀ (c : ValidColoring n) (m : ℕ), (∀ (i j : Fin n), c.val (i, j) < m) → m ≥ n) :=
sorry

end NUMINAMATH_CALUDE_min_colors_regular_ngon_l2836_283668


namespace NUMINAMATH_CALUDE_min_n_with_three_same_color_l2836_283619

/-- A coloring of an n × n grid using three colors. -/
def Coloring (n : ℕ) := Fin n → Fin n → Fin 3

/-- Checks if a coloring satisfies the condition of having at least three squares
    of the same color in a row or column. -/
def satisfiesCondition (n : ℕ) (c : Coloring n) : Prop :=
  ∃ (i : Fin n) (color : Fin 3),
    (∃ (j₁ j₂ j₃ : Fin n), j₁ ≠ j₂ ∧ j₁ ≠ j₃ ∧ j₂ ≠ j₃ ∧
      c i j₁ = color ∧ c i j₂ = color ∧ c i j₃ = color) ∨
    (∃ (i₁ i₂ i₃ : Fin n), i₁ ≠ i₂ ∧ i₁ ≠ i₃ ∧ i₂ ≠ i₃ ∧
      c i₁ i = color ∧ c i₂ i = color ∧ c i₃ i = color)

/-- The main theorem stating that 7 is the smallest n that satisfies the condition. -/
theorem min_n_with_three_same_color :
  (∀ (c : Coloring 7), satisfiesCondition 7 c) ∧
  (∀ (n : ℕ), n < 7 → ∃ (c : Coloring n), ¬satisfiesCondition n c) :=
sorry

end NUMINAMATH_CALUDE_min_n_with_three_same_color_l2836_283619


namespace NUMINAMATH_CALUDE_henry_games_count_henry_games_count_proof_l2836_283646

theorem henry_games_count : ℕ → ℕ → ℕ → Prop :=
  fun initial_neil initial_henry games_given =>
    -- Neil's initial games count
    let initial_neil_games := 7

    -- Henry initially had 3 times more games than Neil
    (initial_henry = 3 * initial_neil_games + initial_neil_games) →
    
    -- After giving Neil 6 games, Henry has 4 times more games than Neil
    (initial_henry - games_given = 4 * (initial_neil_games + games_given)) →
    
    -- The number of games given to Neil
    (games_given = 6) →
    
    -- Conclusion: Henry's initial game count
    initial_henry = 58

-- The proof of the theorem
theorem henry_games_count_proof : henry_games_count 7 58 6 := by
  sorry

end NUMINAMATH_CALUDE_henry_games_count_henry_games_count_proof_l2836_283646


namespace NUMINAMATH_CALUDE_point_p_position_l2836_283686

/-- Given seven points O, A, B, C, D, E, F on a line, with specified distances from O,
    and a point P between D and E satisfying a ratio condition,
    prove that OP has a specific value. -/
theorem point_p_position
  (a b c d e f : ℝ)  -- Real parameters for distances
  (O A B C D E F P : ℝ)  -- Points on the real line
  (h1 : O = 0)  -- O is the origin
  (h2 : A = 2*a)
  (h3 : B = 5*b)
  (h4 : C = 9*c)
  (h5 : D = 12*d)
  (h6 : E = 15*e)
  (h7 : F = 20*f)
  (h8 : D ≤ P ∧ P ≤ E)  -- P is between D and E
  (h9 : (P - A) / (F - P) = (P - D) / (E - P))  -- Ratio condition
  : P = (300*a*e - 240*d*f) / (2*a - 15*e + 20*f) :=
sorry

end NUMINAMATH_CALUDE_point_p_position_l2836_283686


namespace NUMINAMATH_CALUDE_complex_fraction_squared_complex_fraction_minus_z_l2836_283693

-- Define the complex number i
noncomputable def i : ℂ := Complex.I

-- Theorem 1
theorem complex_fraction_squared :
  ((3 - i) / (1 + i))^2 = -3 - 4*i := by sorry

-- Theorem 2
theorem complex_fraction_minus_z (z : ℂ) (h : z = 1 + i) :
  2 / z - z = -2*i := by sorry

end NUMINAMATH_CALUDE_complex_fraction_squared_complex_fraction_minus_z_l2836_283693


namespace NUMINAMATH_CALUDE_exists_permutation_with_difference_l2836_283624

theorem exists_permutation_with_difference (x y z w : ℝ) 
  (sum_eq : x + y + z + w = 13)
  (sum_squares_eq : x^2 + y^2 + z^2 + w^2 = 43) :
  ∃ (a b c d : ℝ), ({a, b, c, d} : Finset ℝ) = {x, y, z, w} ∧ a * b - c * d ≥ 3 := by
  sorry

end NUMINAMATH_CALUDE_exists_permutation_with_difference_l2836_283624


namespace NUMINAMATH_CALUDE_inequality_equivalence_l2836_283682

theorem inequality_equivalence (x : ℝ) : 5 * x - 12 ≤ 2 * (4 * x - 3) ↔ x ≥ -2 := by sorry

end NUMINAMATH_CALUDE_inequality_equivalence_l2836_283682


namespace NUMINAMATH_CALUDE_swimmers_pass_178_times_l2836_283609

/-- Represents a swimmer in the pool --/
structure Swimmer where
  speed : ℝ
  startPosition : ℝ

/-- Represents the swimming pool scenario --/
structure PoolScenario where
  poolLength : ℝ
  swimmer1 : Swimmer
  swimmer2 : Swimmer
  totalTime : ℝ

/-- Calculates the number of times swimmers pass each other --/
def calculatePassings (scenario : PoolScenario) : ℕ :=
  sorry

/-- The specific pool scenario from the problem --/
def problemScenario : PoolScenario :=
  { poolLength := 100
    swimmer1 := { speed := 4, startPosition := 0 }
    swimmer2 := { speed := 3, startPosition := 100 }
    totalTime := 20 * 60 }  -- 20 minutes in seconds

theorem swimmers_pass_178_times :
  calculatePassings problemScenario = 178 :=
sorry

end NUMINAMATH_CALUDE_swimmers_pass_178_times_l2836_283609


namespace NUMINAMATH_CALUDE_trig_values_equal_for_same_terminal_side_l2836_283681

-- Define what it means for two angles to have the same terminal side
def same_terminal_side (α β : Real) : Prop := sorry

-- Define a general trigonometric function
def trig_function (α : Real) : Real := sorry

theorem trig_values_equal_for_same_terminal_side :
  ∀ (α β : Real) (f : Real → Real),
  same_terminal_side α β →
  f = trig_function →
  f α = f β :=
sorry

end NUMINAMATH_CALUDE_trig_values_equal_for_same_terminal_side_l2836_283681


namespace NUMINAMATH_CALUDE_equation_solution_l2836_283652

theorem equation_solution : ∃! (x : ℝ), x ≠ 0 ∧ (6 * x)^18 = (12 * x)^9 ∧ x = 1/3 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l2836_283652


namespace NUMINAMATH_CALUDE_sum_of_solutions_is_zero_l2836_283659

theorem sum_of_solutions_is_zero (x₁ x₂ : ℝ) :
  (8 : ℝ) = 8 →
  x₁^2 + 8^2 = 144 →
  x₂^2 + 8^2 = 144 →
  x₁ + x₂ = 0 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_solutions_is_zero_l2836_283659


namespace NUMINAMATH_CALUDE_beef_weight_before_processing_l2836_283684

theorem beef_weight_before_processing 
  (weight_after : ℝ) 
  (percent_lost : ℝ) 
  (h1 : weight_after = 750)
  (h2 : percent_lost = 50) : 
  weight_after / (1 - percent_lost / 100) = 1500 :=
by sorry

end NUMINAMATH_CALUDE_beef_weight_before_processing_l2836_283684


namespace NUMINAMATH_CALUDE_construction_work_proof_l2836_283639

/-- Represents the number of men who dropped out -/
def men_dropped_out : ℕ := 1

theorem construction_work_proof :
  let initial_men : ℕ := 5
  let half_job_days : ℕ := 15
  let full_job_days : ℕ := 30
  let completion_days : ℕ := 25
  (initial_men * full_job_days : ℚ) = ((initial_men - men_dropped_out) * completion_days : ℚ) :=
by sorry

end NUMINAMATH_CALUDE_construction_work_proof_l2836_283639


namespace NUMINAMATH_CALUDE_parabola_coefficients_l2836_283614

/-- A parabola passing through (1, 1) with a tangent line of slope 1 at (2, -1) has coefficients a = 3, b = -11, and c = 9. -/
theorem parabola_coefficients : 
  ∀ (a b c : ℝ), 
  (a * 1^2 + b * 1 + c = 1) →  -- Passes through (1, 1)
  (a * 2^2 + b * 2 + c = -1) →  -- Passes through (2, -1)
  (2 * a * 2 + b = 1) →  -- Slope of tangent line at (2, -1) is 1
  (a = 3 ∧ b = -11 ∧ c = 9) := by
sorry

end NUMINAMATH_CALUDE_parabola_coefficients_l2836_283614


namespace NUMINAMATH_CALUDE_percentage_failed_hindi_l2836_283671

theorem percentage_failed_hindi (failed_english : ℝ) (failed_both : ℝ) (passed_both : ℝ)
  (h1 : failed_english = 70)
  (h2 : failed_both = 10)
  (h3 : passed_both = 20) :
  ∃ failed_hindi : ℝ, failed_hindi = 20 ∧ 
    passed_both + (failed_hindi + failed_english - failed_both) = 100 :=
by sorry

end NUMINAMATH_CALUDE_percentage_failed_hindi_l2836_283671


namespace NUMINAMATH_CALUDE_restaurant_bill_calculation_l2836_283698

theorem restaurant_bill_calculation (num_people : ℕ) (cost_per_person : ℝ) (gratuity_percentage : ℝ) :
  num_people = 6 →
  cost_per_person = 100 →
  gratuity_percentage = 0.20 →
  num_people * cost_per_person * (1 + gratuity_percentage) = 720 := by
sorry

end NUMINAMATH_CALUDE_restaurant_bill_calculation_l2836_283698


namespace NUMINAMATH_CALUDE_partnership_profit_share_l2836_283676

/-- 
Given three partners A, B, and C in a partnership where:
- A invests 3 times as much as B
- B invests two-thirds of what C invests
- The total profit is 4400

This theorem proves that B's share of the profit is 1760.
-/
theorem partnership_profit_share 
  (investment_A investment_B investment_C : ℚ) 
  (total_profit : ℚ) 
  (h1 : investment_A = 3 * investment_B)
  (h2 : investment_B = 2/3 * investment_C)
  (h3 : total_profit = 4400) :
  (investment_B / (investment_A + investment_B + investment_C)) * total_profit = 1760 := by
  sorry


end NUMINAMATH_CALUDE_partnership_profit_share_l2836_283676


namespace NUMINAMATH_CALUDE_coal_burning_duration_l2836_283677

theorem coal_burning_duration (total : ℝ) (burned_fraction : ℝ) (burned_days : ℝ) 
  (h1 : total > 0)
  (h2 : burned_fraction = 2 / 9)
  (h3 : burned_days = 6)
  (h4 : burned_fraction < 1) :
  (total - burned_fraction * total) / (burned_fraction * total / burned_days) = 21 := by
  sorry

end NUMINAMATH_CALUDE_coal_burning_duration_l2836_283677


namespace NUMINAMATH_CALUDE_triangle_angle_relation_l2836_283688

theorem triangle_angle_relation (X Y Z Z₁ Z₂ : ℝ) : 
  X = 40 → Y = 50 → X + Y + Z = 180 → Z = Z₁ + Z₂ → Z₁ - Z₂ = 10 := by
  sorry

end NUMINAMATH_CALUDE_triangle_angle_relation_l2836_283688


namespace NUMINAMATH_CALUDE_min_omega_value_l2836_283655

theorem min_omega_value (ω : ℕ+) : 
  (∀ k : ℕ+, 2 * Real.sin (2 * Real.pi * ↑k + Real.pi / 3) = Real.sqrt 3 → ω ≤ k) →
  2 * Real.sin (2 * Real.pi * ↑ω + Real.pi / 3) = Real.sqrt 3 →
  ω = 1 := by sorry

end NUMINAMATH_CALUDE_min_omega_value_l2836_283655


namespace NUMINAMATH_CALUDE_family_weight_l2836_283600

/-- The total weight of a family consisting of a mother, daughter, and grandchild is 160 kg,
    given that:
    1. The daughter and her child weigh 60 kg together.
    2. The child is 1/5th the weight of her grandmother (mother).
    3. The daughter weighs 40 kg. -/
theorem family_weight (mother daughter grandchild : ℝ) : 
  (daughter + grandchild = 60) →
  (grandchild = (1/5) * mother) →
  (daughter = 40) →
  (mother + daughter + grandchild = 160) := by
sorry

end NUMINAMATH_CALUDE_family_weight_l2836_283600


namespace NUMINAMATH_CALUDE_difference_between_number_and_fraction_l2836_283660

theorem difference_between_number_and_fraction (x : ℝ) (h : x = 155) : x - (3/5 * x) = 62 := by
  sorry

end NUMINAMATH_CALUDE_difference_between_number_and_fraction_l2836_283660


namespace NUMINAMATH_CALUDE_base5_division_l2836_283695

/-- Converts a number from base 5 to base 10 -/
def base5ToBase10 (digits : List Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * (5 ^ i)) 0

/-- Converts a number from base 10 to base 5 -/
def base10ToBase5 (n : Nat) : List Nat :=
  if n = 0 then [0] else
  let rec aux (m : Nat) (acc : List Nat) :=
    if m = 0 then acc
    else aux (m / 5) ((m % 5) :: acc)
  aux n []

/-- Theorem: The quotient of 2314₅ divided by 21₅ is equal to 110₅ -/
theorem base5_division :
  base10ToBase5 (base5ToBase10 [4, 1, 3, 2] / base5ToBase10 [1, 2]) = [0, 1, 1] :=
sorry

end NUMINAMATH_CALUDE_base5_division_l2836_283695


namespace NUMINAMATH_CALUDE_zach_stadium_goal_l2836_283615

/-- The number of stadiums Zach wants to visit --/
def num_stadiums : ℕ := 30

/-- The cost per stadium in dollars --/
def cost_per_stadium : ℕ := 900

/-- Zach's yearly savings in dollars --/
def yearly_savings : ℕ := 1500

/-- The number of years to accomplish the goal --/
def years_to_goal : ℕ := 18

/-- Theorem stating that the number of stadiums Zach wants to visit is 30 --/
theorem zach_stadium_goal :
  num_stadiums = (yearly_savings * years_to_goal) / cost_per_stadium :=
by sorry

end NUMINAMATH_CALUDE_zach_stadium_goal_l2836_283615


namespace NUMINAMATH_CALUDE_pencil_count_equality_l2836_283601

theorem pencil_count_equality (jayden marcus dana ella : ℕ) : 
  jayden = 20 →
  dana = jayden + 15 →
  jayden = 2 * marcus →
  ella = 3 * marcus - 5 →
  dana = marcus + ella :=
by
  sorry

end NUMINAMATH_CALUDE_pencil_count_equality_l2836_283601


namespace NUMINAMATH_CALUDE_gasoline_price_increase_l2836_283602

/-- The percentage increase in gasoline price from 1972 to 1992 -/
theorem gasoline_price_increase (initial_price final_price : ℝ) : 
  initial_price = 29.90 →
  final_price = 149.70 →
  (final_price - initial_price) / initial_price * 100 = 400 := by
sorry

end NUMINAMATH_CALUDE_gasoline_price_increase_l2836_283602


namespace NUMINAMATH_CALUDE_divisibility_problem_l2836_283642

theorem divisibility_problem (n m k : ℕ) (h1 : n = 425897) (h2 : m = 456) (h3 : k = 247) :
  (n + k) % m = 0 :=
by sorry

end NUMINAMATH_CALUDE_divisibility_problem_l2836_283642


namespace NUMINAMATH_CALUDE_existence_of_unique_distance_point_l2836_283674

-- Define a lattice point as a pair of integers
def LatticePoint := ℤ × ℤ

-- Define a function to calculate the squared distance between two points
def squaredDistance (x y : ℝ × ℝ) : ℝ :=
  (x.1 - y.1)^2 + (x.2 - y.2)^2

theorem existence_of_unique_distance_point :
  ∃ (P : ℝ × ℝ), 
    (∃ (a b : ℝ), P = (a, b) ∧ Irrational a ∧ Irrational b) ∧
    (∀ (L₁ L₂ : LatticePoint), 
      L₁ ≠ L₂ → squaredDistance (P.1, P.2) (↑L₁.1, ↑L₁.2) ≠ 
                 squaredDistance (P.1, P.2) (↑L₂.1, ↑L₂.2)) :=
by sorry

end NUMINAMATH_CALUDE_existence_of_unique_distance_point_l2836_283674


namespace NUMINAMATH_CALUDE_sum_of_fifth_powers_l2836_283691

theorem sum_of_fifth_powers (ζ₁ ζ₂ ζ₃ : ℂ) 
  (h1 : ζ₁ + ζ₂ + ζ₃ = 2)
  (h2 : ζ₁^2 + ζ₂^2 + ζ₃^2 = 6)
  (h3 : ζ₁^3 + ζ₂^3 + ζ₃^3 = 8) :
  ζ₁^5 + ζ₂^5 + ζ₃^5 = 30 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_fifth_powers_l2836_283691


namespace NUMINAMATH_CALUDE_optimal_threshold_at_intersection_l2836_283628

/-- Represents the height distribution of vehicles for a given class --/
def HeightDistribution := ℝ → ℝ

/-- The cost for class 1 vehicles --/
def class1Cost : ℝ := 200

/-- The cost for class 2 vehicles --/
def class2Cost : ℝ := 300

/-- The height distribution for class 1 vehicles --/
noncomputable def class1Distribution : HeightDistribution := sorry

/-- The height distribution for class 2 vehicles --/
noncomputable def class2Distribution : HeightDistribution := sorry

/-- The intersection point of the two height distributions --/
noncomputable def intersectionPoint : ℝ := sorry

/-- The error function for a given threshold --/
def errorFunction (h : ℝ) : ℝ := sorry

/-- Theorem: The optimal threshold that minimizes classification errors
    is at the intersection point of the two height distributions --/
theorem optimal_threshold_at_intersection :
  ∀ h : ℝ, h ≠ intersectionPoint → errorFunction h > errorFunction intersectionPoint :=
by sorry

end NUMINAMATH_CALUDE_optimal_threshold_at_intersection_l2836_283628


namespace NUMINAMATH_CALUDE_smallest_period_sin_polar_l2836_283667

theorem smallest_period_sin_polar (t : ℝ) : 
  (∀ θ : ℝ, 0 ≤ θ ∧ θ ≤ t → 
    ∃ r : ℝ, r = Real.sin θ ∧ 
    (∃ x y : ℝ, x = r * Real.cos θ ∧ y = r * Real.sin θ)) → 
  (∀ x y : ℝ, x^2 + y^2 ≤ 1 → 
    ∃ θ : ℝ, 0 ≤ θ ∧ θ ≤ t ∧ 
    x = (Real.sin θ) * (Real.cos θ) ∧ 
    y = (Real.sin θ) * (Real.sin θ)) →
  t ≥ π :=
sorry

end NUMINAMATH_CALUDE_smallest_period_sin_polar_l2836_283667


namespace NUMINAMATH_CALUDE_line_y_coordinate_l2836_283617

/-- Given a line that passes through points (-6, y1) and (x2, 3), 
    with an x-intercept at (4, 0), prove that y1 = 7.5 -/
theorem line_y_coordinate (y1 x2 : ℝ) : 
  (∃ m b : ℝ, 
    (y1 = m * (-6) + b) ∧ 
    (3 = m * x2 + b) ∧ 
    (0 = m * 4 + b)) →
  y1 = 7.5 := by
sorry

end NUMINAMATH_CALUDE_line_y_coordinate_l2836_283617


namespace NUMINAMATH_CALUDE_smallest_N_for_P_less_than_half_l2836_283620

/-- The probability that at least 2/3 of the green balls are on the same side of either of the red balls -/
def P (N : ℕ) : ℚ :=
  sorry

/-- N is a multiple of 6 -/
def is_multiple_of_six (N : ℕ) : Prop :=
  ∃ k : ℕ, N = 6 * k

theorem smallest_N_for_P_less_than_half :
  (is_multiple_of_six 18) ∧
  (P 18 < 1/2) ∧
  (∀ N : ℕ, is_multiple_of_six N → N < 18 → P N ≥ 1/2) :=
sorry

end NUMINAMATH_CALUDE_smallest_N_for_P_less_than_half_l2836_283620


namespace NUMINAMATH_CALUDE_percentage_difference_l2836_283604

theorem percentage_difference : 
  (55 / 100 * 40) - (4 / 5 * 25) = 2 := by
  sorry

end NUMINAMATH_CALUDE_percentage_difference_l2836_283604


namespace NUMINAMATH_CALUDE_smallest_lcm_three_digit_gcd_five_l2836_283627

theorem smallest_lcm_three_digit_gcd_five :
  ∃ (m n : ℕ), 
    100 ≤ m ∧ m < 1000 ∧
    100 ≤ n ∧ n < 1000 ∧
    Nat.gcd m n = 5 ∧
    Nat.lcm m n = 2100 ∧
    ∀ (p q : ℕ), 
      100 ≤ p ∧ p < 1000 ∧
      100 ≤ q ∧ q < 1000 ∧
      Nat.gcd p q = 5 →
      Nat.lcm p q ≥ 2100 :=
by sorry

end NUMINAMATH_CALUDE_smallest_lcm_three_digit_gcd_five_l2836_283627


namespace NUMINAMATH_CALUDE_circumscribed_circle_area_l2836_283611

/-- The area of a circle circumscribed about an equilateral triangle with side length 12 units -/
theorem circumscribed_circle_area (s : ℝ) (h : s = 12) : 
  let r := 2 * s * Real.sqrt 3 / 3
  (π : ℝ) * r^2 = 48 * π := by sorry

end NUMINAMATH_CALUDE_circumscribed_circle_area_l2836_283611


namespace NUMINAMATH_CALUDE_cube_sum_divided_by_quadratic_difference_l2836_283638

theorem cube_sum_divided_by_quadratic_difference (a c : ℝ) (h1 : a = 6) (h2 : c = 3) :
  (a^3 + c^3) / (a^2 - a*c + c^2) = 9 := by
  sorry

end NUMINAMATH_CALUDE_cube_sum_divided_by_quadratic_difference_l2836_283638


namespace NUMINAMATH_CALUDE_intersection_line_equation_l2836_283633

/-- Definition of line l1 -/
def l1 (x y : ℝ) : Prop := x - y + 3 = 0

/-- Definition of line l2 -/
def l2 (x y : ℝ) : Prop := 2*x + y = 0

/-- Definition of the intersection point of l1 and l2 -/
def intersection_point (x y : ℝ) : Prop := l1 x y ∧ l2 x y

/-- Definition of a line with inclination angle π/3 passing through a point -/
def line_with_inclination (x₀ y₀ x y : ℝ) : Prop :=
  y - y₀ = Real.sqrt 3 * (x - x₀)

/-- The main theorem -/
theorem intersection_line_equation :
  ∃ x₀ y₀ : ℝ, intersection_point x₀ y₀ ∧
  ∀ x y : ℝ, line_with_inclination x₀ y₀ x y ↔ Real.sqrt 3 * x - y + Real.sqrt 3 + 2 = 0 :=
sorry

end NUMINAMATH_CALUDE_intersection_line_equation_l2836_283633


namespace NUMINAMATH_CALUDE_hyperbola_intersection_theorem_l2836_283664

-- Define the hyperbola C
def hyperbola (x y : ℝ) : Prop := x^2 / 3 - y^2 = 1

-- Define the line l
def line (k m x y : ℝ) : Prop := y = k * x + m

-- Define the focal points
def F₁ : ℝ × ℝ := (-2, 0)
def F₂ : ℝ × ℝ := (2, 0)

-- Define the theorem
theorem hyperbola_intersection_theorem 
  (k m : ℝ) 
  (A B : ℝ × ℝ) 
  (h_focal_length : Real.sqrt 16 = 4)
  (h_imaginary_axis : Real.sqrt 4 = 2)
  (h_m_nonzero : m ≠ 0)
  (h_distinct : A ≠ B)
  (h_on_hyperbola_A : hyperbola A.1 A.2)
  (h_on_hyperbola_B : hyperbola B.1 B.2)
  (h_on_line_A : line k m A.1 A.2)
  (h_on_line_B : line k m B.1 B.2)
  (h_distance : Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) = 2 * Real.sqrt 3)
  (h_passes_F₂ : line k m F₂.1 F₂.2) :
  -- 1. Eccentricity
  (2 * Real.sqrt 3 / 3 = Real.sqrt (1 - 1 / 3)) ∧
  -- 2. Equation of line l
  ((k = 1 ∧ m = -2) ∨ (k = -1 ∧ m = 2) ∨ (k = 0 ∧ m ≠ 0)) ∧
  -- 3. Range of m
  (k ≠ 0 → m ∈ Set.Icc (-1/4) 0 ∪ Set.Ioi 4) ∧
  (k = 0 → m ∈ Set.univ \ {0}) := by
  sorry

end NUMINAMATH_CALUDE_hyperbola_intersection_theorem_l2836_283664


namespace NUMINAMATH_CALUDE_infinitely_many_larger_divisor_sum_ratio_l2836_283610

-- Define the sum of divisors function
def sigma (n : ℕ) : ℕ := sorry

-- Define the theorem
theorem infinitely_many_larger_divisor_sum_ratio :
  ∀ t : ℕ, ∃ n : ℕ, n > t ∧ ∀ k : ℕ, k ∈ Finset.range n → (sigma n : ℚ) / n > (sigma k : ℚ) / k :=
sorry

end NUMINAMATH_CALUDE_infinitely_many_larger_divisor_sum_ratio_l2836_283610


namespace NUMINAMATH_CALUDE_cubic_factorization_l2836_283672

theorem cubic_factorization (x : ℝ) : x^3 - 4*x^2 + 4*x = x*(x-2)^2 := by
  sorry

end NUMINAMATH_CALUDE_cubic_factorization_l2836_283672


namespace NUMINAMATH_CALUDE_derivative_f_at_neg_two_l2836_283696

-- Define the function f
def f (x : ℝ) : ℝ := x^3

-- State the theorem
theorem derivative_f_at_neg_two :
  (deriv f) (-2) = 0 := by sorry

end NUMINAMATH_CALUDE_derivative_f_at_neg_two_l2836_283696


namespace NUMINAMATH_CALUDE_students_not_eating_lunch_l2836_283608

theorem students_not_eating_lunch (total : ℕ) (cafeteria : ℕ) (bring_lunch_multiplier : ℕ) :
  total = 90 →
  bring_lunch_multiplier = 4 →
  cafeteria = 12 →
  total - (cafeteria + bring_lunch_multiplier * cafeteria) = 30 :=
by sorry

end NUMINAMATH_CALUDE_students_not_eating_lunch_l2836_283608


namespace NUMINAMATH_CALUDE_brother_twice_sister_age_l2836_283661

theorem brother_twice_sister_age (brother_age_2010 sister_age_2010 : ℕ) : 
  brother_age_2010 = 16 →
  sister_age_2010 = 10 →
  ∃ (year : ℕ), year = 2006 ∧ 
    brother_age_2010 - (2010 - year) = 2 * (sister_age_2010 - (2010 - year)) :=
by sorry

end NUMINAMATH_CALUDE_brother_twice_sister_age_l2836_283661


namespace NUMINAMATH_CALUDE_katy_brownies_l2836_283651

/-- The number of brownies Katy eats on Monday -/
def monday_brownies : ℕ := 5

/-- The number of brownies Katy makes in total -/
def total_brownies : ℕ := monday_brownies + 2 * monday_brownies

theorem katy_brownies : 
  total_brownies = 15 := by sorry

end NUMINAMATH_CALUDE_katy_brownies_l2836_283651


namespace NUMINAMATH_CALUDE_salad_cost_main_theorem_l2836_283670

/-- The cost of ingredients for Laura's dinner --/
structure DinnerCost where
  salad_price : ℝ
  beef_price : ℝ
  potato_price : ℝ
  juice_price : ℝ

/-- The quantities of ingredients Laura bought --/
structure DinnerQuantities where
  salad_qty : ℕ
  beef_qty : ℕ
  potato_qty : ℕ
  juice_qty : ℕ

/-- The theorem stating the cost of one salad --/
theorem salad_cost (d : DinnerCost) (q : DinnerQuantities) : d.salad_price = 3 :=
  by
    have h1 : d.beef_price = 2 * d.salad_price := sorry
    have h2 : d.potato_price = (1/3) * d.salad_price := sorry
    have h3 : d.juice_price = 1.5 := sorry
    have h4 : q.salad_qty = 2 ∧ q.beef_qty = 2 ∧ q.potato_qty = 1 ∧ q.juice_qty = 2 := sorry
    have h5 : q.salad_qty * d.salad_price + q.beef_qty * d.beef_price + 
              q.potato_qty * d.potato_price + q.juice_qty * d.juice_price = 22 := sorry
    sorry

/-- The main theorem proving the cost of one salad --/
theorem main_theorem : ∃ (d : DinnerCost) (q : DinnerQuantities), d.salad_price = 3 :=
  by
    sorry

end NUMINAMATH_CALUDE_salad_cost_main_theorem_l2836_283670


namespace NUMINAMATH_CALUDE_equation_one_solutions_equation_two_solution_l2836_283666

-- Problem 1
theorem equation_one_solutions (x : ℝ) :
  3 * (x - 2)^2 - 27 = 0 ↔ x = 5 ∨ x = -1 :=
sorry

-- Problem 2
theorem equation_two_solution (x : ℝ) :
  2 * (x + 1)^3 + 54 = 0 ↔ x = -4 :=
sorry

end NUMINAMATH_CALUDE_equation_one_solutions_equation_two_solution_l2836_283666


namespace NUMINAMATH_CALUDE_arithmetic_calculations_l2836_283650

theorem arithmetic_calculations :
  (0.25 + (-9) + (-1/4) - 11 = -20) ∧
  (-15 + 5 + 1/3 * (-6) = -12) ∧
  ((-3/8 - 1/6 + 3/4) * 24 = 5) := by
sorry

end NUMINAMATH_CALUDE_arithmetic_calculations_l2836_283650


namespace NUMINAMATH_CALUDE_multiply_by_twelve_l2836_283616

theorem multiply_by_twelve (x : ℝ) : x / 14 = 42 → 12 * x = 7056 := by
  sorry

end NUMINAMATH_CALUDE_multiply_by_twelve_l2836_283616


namespace NUMINAMATH_CALUDE_complex_division_l2836_283647

theorem complex_division (z : ℂ) (h1 : z.re = 1) (h2 : z.im = -2) : 
  (5 * Complex.I) / z = -2 + Complex.I :=
sorry

end NUMINAMATH_CALUDE_complex_division_l2836_283647


namespace NUMINAMATH_CALUDE_product_of_reciprocals_equals_one_l2836_283657

theorem product_of_reciprocals_equals_one :
  (1 / (2 + Real.sqrt 3)) * (1 / (2 - Real.sqrt 3)) = 1 := by sorry

end NUMINAMATH_CALUDE_product_of_reciprocals_equals_one_l2836_283657


namespace NUMINAMATH_CALUDE_garment_pricing_problem_l2836_283645

-- Define the linear function
def sales_function (x : ℝ) : ℝ := -2 * x + 400

-- Define the profit function without donation
def profit_function (x : ℝ) : ℝ := (x - 60) * (sales_function x)

-- Define the profit function with donation
def profit_function_with_donation (x : ℝ) : ℝ := (x - 70) * (sales_function x)

theorem garment_pricing_problem :
  -- The linear function fits the given data points
  (sales_function 80 = 240) ∧
  (sales_function 90 = 220) ∧
  (sales_function 100 = 200) ∧
  (sales_function 110 = 180) ∧
  -- The smaller solution to the profit equation is 100
  (∃ x₁ x₂ : ℝ, x₁ < x₂ ∧ 
    profit_function x₁ = 8000 ∧ 
    profit_function x₂ = 8000 ∧ 
    x₁ = 100) ∧
  -- The profit function with donation has a maximum at 135
  (∃ max_profit : ℝ, 
    profit_function_with_donation 135 = max_profit ∧
    ∀ x : ℝ, profit_function_with_donation x ≤ max_profit) :=
by sorry

end NUMINAMATH_CALUDE_garment_pricing_problem_l2836_283645


namespace NUMINAMATH_CALUDE_power_of_product_l2836_283621

theorem power_of_product (a b : ℝ) : (-3 * a^3 * b)^2 = 9 * a^6 * b^2 := by
  sorry

end NUMINAMATH_CALUDE_power_of_product_l2836_283621


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_condition_l2836_283637

theorem sufficient_not_necessary_condition (a b : ℝ) :
  (∀ a b, a > b + 1 → a > b) ∧
  (∃ a b, a > b ∧ ¬(a > b + 1)) := by
  sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_condition_l2836_283637


namespace NUMINAMATH_CALUDE_find_C_value_l2836_283612

-- Define the structure of the 8-digit numbers
def FirstNumber (A B : ℕ) : ℕ := 85000000 + A * 100000 + 73000 + B * 100 + 20
def SecondNumber (A B C : ℕ) : ℕ := 41000000 + 700000 + A * 10000 + B * 1000 + 500 + C * 10 + 9

-- Define the condition for being a multiple of 5
def IsMultipleOf5 (n : ℕ) : Prop := ∃ k : ℕ, n = 5 * k

-- State the theorem
theorem find_C_value (A B : ℕ) (h1 : IsMultipleOf5 (FirstNumber A B)) 
  (h2 : ∃ C : ℕ, IsMultipleOf5 (SecondNumber A B C)) : 
  ∃ C : ℕ, C = 1 ∧ IsMultipleOf5 (SecondNumber A B C) :=
sorry

end NUMINAMATH_CALUDE_find_C_value_l2836_283612


namespace NUMINAMATH_CALUDE_count_integers_with_fourth_power_between_negative_hundred_and_hundred_l2836_283625

theorem count_integers_with_fourth_power_between_negative_hundred_and_hundred :
  (∃ (S : Finset Int), (∀ x : Int, x ∈ S ↔ -100 < x^4 ∧ x^4 < 100) ∧ Finset.card S = 7) := by
  sorry

end NUMINAMATH_CALUDE_count_integers_with_fourth_power_between_negative_hundred_and_hundred_l2836_283625
