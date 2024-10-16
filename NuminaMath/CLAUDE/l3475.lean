import Mathlib

namespace NUMINAMATH_CALUDE_intersection_distance_l3475_347554

def C₁ (x y : ℝ) : Prop := x^2 + (y - 2)^2 = 7

def C₂ (ρ θ : ℝ) : Prop := ρ = 2 * Real.cos θ

theorem intersection_distance :
  ∃ (ρ₁ ρ₂ : ℝ),
    C₁ (ρ₁ * Real.cos (π/6)) (ρ₁ * Real.sin (π/6)) ∧
    C₂ ρ₂ (π/6) ∧
    ρ₁ > 0 ∧ ρ₂ > 0 ∧
    ρ₁ - ρ₂ = 3 - Real.sqrt 3 :=
  sorry

end NUMINAMATH_CALUDE_intersection_distance_l3475_347554


namespace NUMINAMATH_CALUDE_pencils_per_child_l3475_347553

theorem pencils_per_child (num_children : ℕ) (total_pencils : ℕ) (h1 : num_children = 2) (h2 : total_pencils = 12) :
  total_pencils / num_children = 6 := by
sorry

end NUMINAMATH_CALUDE_pencils_per_child_l3475_347553


namespace NUMINAMATH_CALUDE_prob_A_wins_four_consecutive_prob_need_fifth_game_prob_C_ultimate_winner_l3475_347511

/-- Represents a player in the badminton game --/
inductive Player : Type
  | A
  | B
  | C

/-- Represents the state of the game --/
structure GameState :=
  (current_players : List Player)
  (bye_player : Player)
  (eliminated_player : Option Player)

/-- The probability of a player winning a single game --/
def win_probability : ℚ := 1/2

/-- The initial game state --/
def initial_state : GameState :=
  { current_players := [Player.A, Player.B],
    bye_player := Player.C,
    eliminated_player := none }

/-- Calculates the probability of a specific game outcome --/
def outcome_probability (num_games : ℕ) : ℚ :=
  (win_probability ^ num_games : ℚ)

/-- Theorem stating the probability of A winning four consecutive games --/
theorem prob_A_wins_four_consecutive :
  outcome_probability 4 = 1/16 := by sorry

/-- Theorem stating the probability of needing a fifth game --/
theorem prob_need_fifth_game :
  1 - 4 * outcome_probability 4 = 3/4 := by sorry

/-- Theorem stating the probability of C being the ultimate winner --/
theorem prob_C_ultimate_winner :
  7/16 = 1 - 2 * (outcome_probability 4 + 7 * outcome_probability 5) := by sorry

end NUMINAMATH_CALUDE_prob_A_wins_four_consecutive_prob_need_fifth_game_prob_C_ultimate_winner_l3475_347511


namespace NUMINAMATH_CALUDE_repeating_decimal_to_fraction_l3475_347536

theorem repeating_decimal_to_fraction :
  ∃ (n : ℕ) (d : ℕ), d ≠ 0 ∧ (n / d : ℚ) = ∑' k, 6 * (1 / 10 : ℚ)^(k + 1) ∧ n / d = 2 / 3 := by
  sorry

end NUMINAMATH_CALUDE_repeating_decimal_to_fraction_l3475_347536


namespace NUMINAMATH_CALUDE_running_days_calculation_l3475_347525

/-- 
Given:
- Peter runs 3 miles more than Andrew per day
- Andrew runs 2 miles per day
- Their total combined distance is 35 miles

Prove that they have been running for 5 days.
-/
theorem running_days_calculation (andrew_miles : ℕ) (peter_miles : ℕ) (total_miles : ℕ) (days : ℕ) :
  andrew_miles = 2 →
  peter_miles = andrew_miles + 3 →
  total_miles = 35 →
  days * (andrew_miles + peter_miles) = total_miles →
  days = 5 := by
  sorry

#check running_days_calculation

end NUMINAMATH_CALUDE_running_days_calculation_l3475_347525


namespace NUMINAMATH_CALUDE_percent_of_y_l3475_347574

theorem percent_of_y (y : ℝ) (h : y > 0) : (1 * y / 20 + 3 * y / 10) / y * 100 = 35 := by
  sorry

end NUMINAMATH_CALUDE_percent_of_y_l3475_347574


namespace NUMINAMATH_CALUDE_ellipse_area_l3475_347584

/-- The area of an ellipse with semi-major axis a and semi-minor axis b -/
theorem ellipse_area (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (∫ x in -a..a, 2 * b * Real.sqrt (1 - x^2 / a^2)) = π * a * b :=
sorry

end NUMINAMATH_CALUDE_ellipse_area_l3475_347584


namespace NUMINAMATH_CALUDE_lindsay_doll_difference_l3475_347518

/-- The number of dolls Lindsay has with different hair colors -/
structure DollCounts where
  blonde : ℕ
  brown : ℕ
  black : ℕ

/-- Lindsay's doll collection satisfying the given conditions -/
def lindsay_dolls : DollCounts where
  blonde := 4
  brown := 4 * 4
  black := 4 * 4 - 2

/-- The difference between the number of dolls with black and brown hair combined
    and the number of dolls with blonde hair -/
def hair_color_difference (d : DollCounts) : ℕ :=
  d.brown + d.black - d.blonde

theorem lindsay_doll_difference :
  hair_color_difference lindsay_dolls = 26 := by
  sorry

end NUMINAMATH_CALUDE_lindsay_doll_difference_l3475_347518


namespace NUMINAMATH_CALUDE_ferry_speed_proof_l3475_347519

/-- The speed of ferry P in km/h -/
def speed_P : ℝ := 6

/-- The speed of ferry Q in km/h -/
def speed_Q : ℝ := speed_P + 3

/-- The time taken by ferry P in hours -/
def time_P : ℝ := 3

/-- The time taken by ferry Q in hours -/
def time_Q : ℝ := time_P + 3

/-- The distance traveled by ferry P in km -/
def distance_P : ℝ := speed_P * time_P

/-- The distance traveled by ferry Q in km -/
def distance_Q : ℝ := 3 * distance_P

theorem ferry_speed_proof :
  speed_P = 6 ∧
  speed_Q = speed_P + 3 ∧
  time_Q = time_P + 3 ∧
  distance_Q = 3 * distance_P ∧
  distance_P = speed_P * time_P ∧
  distance_Q = speed_Q * time_Q :=
by sorry

end NUMINAMATH_CALUDE_ferry_speed_proof_l3475_347519


namespace NUMINAMATH_CALUDE_sum_of_ages_l3475_347555

theorem sum_of_ages (a b c : ℕ) : 
  a = b + c + 16 → 
  a^2 = (b + c)^2 + 1632 → 
  a + b + c = 102 := by
sorry

end NUMINAMATH_CALUDE_sum_of_ages_l3475_347555


namespace NUMINAMATH_CALUDE_product_remainder_l3475_347552

theorem product_remainder (a b c : ℕ) (ha : a = 1234) (hb : b = 1567) (hc : c = 1912) :
  (a * b * c) % 5 = 1 := by
  sorry

end NUMINAMATH_CALUDE_product_remainder_l3475_347552


namespace NUMINAMATH_CALUDE_union_of_sets_l3475_347559

theorem union_of_sets : 
  let A : Set ℕ := {1, 2}
  let B : Set ℕ := {2, 3}
  A ∪ B = {1, 2, 3} := by
sorry

end NUMINAMATH_CALUDE_union_of_sets_l3475_347559


namespace NUMINAMATH_CALUDE_bug_travel_distance_l3475_347589

theorem bug_travel_distance (r : ℝ) (s : ℝ) (h1 : r = 65) (h2 : s = 100) :
  let d := 2 * r
  let x := Real.sqrt (d^2 - s^2)
  d + s + x = 313 :=
by sorry

end NUMINAMATH_CALUDE_bug_travel_distance_l3475_347589


namespace NUMINAMATH_CALUDE_x_value_proof_l3475_347527

theorem x_value_proof : 
  ∀ x : ℝ, x = 143 * (1 + 32.5 / 100) → x = 189.475 := by
  sorry

end NUMINAMATH_CALUDE_x_value_proof_l3475_347527


namespace NUMINAMATH_CALUDE_annika_hiking_rate_l3475_347505

/-- Annika's hiking problem -/
theorem annika_hiking_rate (initial_distance : Real) (total_east_distance : Real) (return_time : Real) :
  initial_distance = 2.75 →
  total_east_distance = 3.625 →
  return_time = 45 →
  let additional_east := total_east_distance - initial_distance
  let total_distance := initial_distance + 2 * additional_east
  total_distance / return_time * 60 = 10 := by
  sorry

end NUMINAMATH_CALUDE_annika_hiking_rate_l3475_347505


namespace NUMINAMATH_CALUDE_arithmetic_sequence_problem_l3475_347562

/-- An arithmetic sequence is a sequence where the difference between each consecutive term is constant. -/
def is_arithmetic_sequence (a : ℕ → ℤ) : Prop :=
  ∃ d : ℤ, ∀ n : ℕ, a (n + 1) = a n + d

/-- Given an arithmetic sequence a where a₃ = -6 and a₇ = a₅ + 4, prove that a₁ = -10 -/
theorem arithmetic_sequence_problem (a : ℕ → ℤ) 
  (h_arith : is_arithmetic_sequence a) 
  (h_a3 : a 3 = -6) 
  (h_a7 : a 7 = a 5 + 4) : 
  a 1 = -10 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_problem_l3475_347562


namespace NUMINAMATH_CALUDE_function_composition_distribution_l3475_347531

-- Define real-valued functions on ℝ
variable (f g h : ℝ → ℝ)

-- Define function composition
def comp (f g : ℝ → ℝ) : ℝ → ℝ := λ x ↦ f (g x)

-- Define pointwise multiplication of functions
def mult (f g : ℝ → ℝ) : ℝ → ℝ := λ x ↦ f x * g x

-- Statement of the theorem
theorem function_composition_distribution :
  ∀ x : ℝ, (comp (mult f g) h) x = (mult (comp f h) (comp g h)) x :=
by sorry

end NUMINAMATH_CALUDE_function_composition_distribution_l3475_347531


namespace NUMINAMATH_CALUDE_vertical_strips_count_l3475_347557

/-- Represents a rectangular grid with a hole -/
structure GridWithHole where
  outer_perimeter : ℕ
  hole_perimeter : ℕ
  horizontal_strips : ℕ

/-- The number of vertical strips in a GridWithHole -/
def vertical_strips (g : GridWithHole) : ℕ :=
  g.outer_perimeter / 2 + g.hole_perimeter / 2 - g.horizontal_strips

theorem vertical_strips_count (g : GridWithHole) 
  (h1 : g.outer_perimeter = 50)
  (h2 : g.hole_perimeter = 32)
  (h3 : g.horizontal_strips = 20) :
  vertical_strips g = 21 := by
  sorry

#eval vertical_strips { outer_perimeter := 50, hole_perimeter := 32, horizontal_strips := 20 }

end NUMINAMATH_CALUDE_vertical_strips_count_l3475_347557


namespace NUMINAMATH_CALUDE_quadratic_equality_l3475_347526

/-- A quadratic function f(x) = ax^2 + bx + 6 -/
def f (a b : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x + 6

/-- Theorem: If f(-1) = f(3) for a quadratic function f(x) = ax^2 + bx + 6, then f(2) = 6 -/
theorem quadratic_equality (a b : ℝ) : f a b (-1) = f a b 3 → f a b 2 = 6 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_equality_l3475_347526


namespace NUMINAMATH_CALUDE_total_carrots_is_twenty_l3475_347569

/-- The number of carrots grown by Sally -/
def sally_carrots : ℕ := 6

/-- The number of carrots grown by Fred -/
def fred_carrots : ℕ := 4

/-- The number of carrots grown by Mary -/
def mary_carrots : ℕ := 10

/-- The total number of carrots grown by Sally, Fred, and Mary -/
def total_carrots : ℕ := sally_carrots + fred_carrots + mary_carrots

theorem total_carrots_is_twenty : total_carrots = 20 := by sorry

end NUMINAMATH_CALUDE_total_carrots_is_twenty_l3475_347569


namespace NUMINAMATH_CALUDE_linear_functions_intersection_l3475_347567

/-- A linear function represented by its slope and y-intercept -/
structure LinearFunction where
  slope : ℝ
  intercept : ℝ

/-- Evaluate a linear function at a given x -/
def LinearFunction.eval (f : LinearFunction) (x : ℝ) : ℝ :=
  f.slope * x + f.intercept

theorem linear_functions_intersection (f₁ f₂ : LinearFunction) :
  (f₁.eval 2 = f₂.eval 2) →
  (|f₁.eval 8 - f₂.eval 8| = 8) →
  ((f₁.eval 20 = 100) ∨ (f₂.eval 20 = 100)) →
  ((f₁.eval 20 = 76 ∧ f₂.eval 20 = 100) ∨ (f₁.eval 20 = 100 ∧ f₂.eval 20 = 124) ∨
   (f₁.eval 20 = 100 ∧ f₂.eval 20 = 76) ∨ (f₁.eval 20 = 124 ∧ f₂.eval 20 = 100)) := by
  sorry

end NUMINAMATH_CALUDE_linear_functions_intersection_l3475_347567


namespace NUMINAMATH_CALUDE_max_abs_sum_on_circle_l3475_347598

theorem max_abs_sum_on_circle (x y : ℝ) (h : x^2 + y^2 = 4) :
  |x| + |y| ≤ 2 * Real.sqrt 2 ∧ ∃ x y : ℝ, x^2 + y^2 = 4 ∧ |x| + |y| = 2 * Real.sqrt 2 :=
by sorry

end NUMINAMATH_CALUDE_max_abs_sum_on_circle_l3475_347598


namespace NUMINAMATH_CALUDE_range_of_m_l3475_347556

theorem range_of_m (m : ℝ) : 
  (∀ x : ℝ, (m - 1 < x ∧ x < m + 1) → (x^2 - 2*x - 3 > 0)) ∧ 
  (∃ x : ℝ, (x^2 - 2*x - 3 > 0) ∧ ¬(m - 1 < x ∧ x < m + 1)) ↔ 
  (m ≤ -2 ∨ m ≥ 4) :=
sorry

end NUMINAMATH_CALUDE_range_of_m_l3475_347556


namespace NUMINAMATH_CALUDE_smallest_prime_with_digit_sum_25_l3475_347532

def digit_sum (n : ℕ) : ℕ :=
  if n < 10 then n else n % 10 + digit_sum (n / 10)

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, 1 < m → m < n → ¬(n % m = 0)

theorem smallest_prime_with_digit_sum_25 :
  ∃ p : ℕ, is_prime p ∧ digit_sum p = 25 ∧
  ∀ q : ℕ, is_prime q ∧ digit_sum q = 25 → p ≤ q :=
by sorry

end NUMINAMATH_CALUDE_smallest_prime_with_digit_sum_25_l3475_347532


namespace NUMINAMATH_CALUDE_inverse_abs_equality_false_l3475_347548

theorem inverse_abs_equality_false : ¬ ∀ a b : ℝ, |a| = |b| → a = b := by
  sorry

end NUMINAMATH_CALUDE_inverse_abs_equality_false_l3475_347548


namespace NUMINAMATH_CALUDE_modulus_of_z_l3475_347547

/-- Given a complex number z satisfying (1-i)z = 2i, prove that its modulus is √2 -/
theorem modulus_of_z (z : ℂ) (h : (1 - Complex.I) * z = 2 * Complex.I) : Complex.abs z = Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_modulus_of_z_l3475_347547


namespace NUMINAMATH_CALUDE_QR_distance_l3475_347545

-- Define the right triangle DEF
structure RightTriangle where
  DE : ℝ
  EF : ℝ
  DF : ℝ
  is_right_triangle : DE^2 + EF^2 = DF^2

-- Define the circles
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

-- Define the problem setup
def problem_setup (t : RightTriangle) (Q R : Circle) : Prop :=
  t.DE = 9 ∧ t.EF = 12 ∧ t.DF = 15 ∧
  Q.center.2 = t.EF ∧ 
  R.center.1 = 0 ∧
  Q.radius = t.DE ∧
  R.radius = t.EF

-- Theorem statement
theorem QR_distance (t : RightTriangle) (Q R : Circle) 
  (h : problem_setup t Q R) : 
  Real.sqrt ((Q.center.1 - R.center.1)^2 + (Q.center.2 - R.center.2)^2) = 15 :=
sorry

end NUMINAMATH_CALUDE_QR_distance_l3475_347545


namespace NUMINAMATH_CALUDE_expected_net_profit_l3475_347582

/-- The expected value of net profit from selling one electronic product -/
theorem expected_net_profit (purchase_price : ℝ) (pass_rate : ℝ) (profit_qualified : ℝ) (loss_defective : ℝ)
  (h1 : purchase_price = 10)
  (h2 : pass_rate = 0.95)
  (h3 : profit_qualified = 2)
  (h4 : loss_defective = 10)
  (h5 : 0 ≤ pass_rate ∧ pass_rate ≤ 1) :
  let X : ℝ → ℝ := λ ω => if ω ≤ pass_rate then profit_qualified else -loss_defective
  let E : (ℝ → ℝ) → ℝ := λ f => pass_rate * (f pass_rate) + (1 - pass_rate) * (f 1)
  E X = 1.4 := by
  sorry

end NUMINAMATH_CALUDE_expected_net_profit_l3475_347582


namespace NUMINAMATH_CALUDE_simplify_trigonometric_expression_l3475_347534

theorem simplify_trigonometric_expression (α : ℝ) :
  (1 - Real.cos (2 * α)) * Real.cos (π / 4 + 2 * α) / (2 * Real.sin (2 * α) ^ 2 - Real.sin (4 * α)) =
  -Real.sqrt 2 / 4 * Real.tan α := by
  sorry

end NUMINAMATH_CALUDE_simplify_trigonometric_expression_l3475_347534


namespace NUMINAMATH_CALUDE_function_decomposition_symmetry_l3475_347579

theorem function_decomposition_symmetry (f : ℝ → ℝ) :
  ∃ (f₁ f₂ : ℝ → ℝ) (a : ℝ), a > 0 ∧
    (∀ x, f x = f₁ x + f₂ x) ∧
    (∀ x, f₁ (-x) = f₁ x) ∧
    (∀ x, f₂ (2 * a - x) = f₂ x) :=
by sorry

end NUMINAMATH_CALUDE_function_decomposition_symmetry_l3475_347579


namespace NUMINAMATH_CALUDE_sum_remainder_zero_l3475_347583

theorem sum_remainder_zero : (9152 + 9153 + 9154 + 9155 + 9156) % 10 = 0 := by
  sorry

end NUMINAMATH_CALUDE_sum_remainder_zero_l3475_347583


namespace NUMINAMATH_CALUDE_A_intersect_B_range_of_a_l3475_347572

-- Define the sets A, B, and C
def A : Set ℝ := {x | x < -2 ∨ (3 < x ∧ x < 4)}
def B : Set ℝ := {x | x^2 - 2*x - 15 ≤ 0}
def C (a : ℝ) : Set ℝ := {x | x ≥ a}

-- Theorem 1: Intersection of A and B
theorem A_intersect_B : A ∩ B = {x : ℝ | 3 < x ∧ x ≤ 5} := by sorry

-- Theorem 2: Range of a given B ∩ C = B
theorem range_of_a (h : B ∩ C a = B) : a ≤ -3 := by sorry

end NUMINAMATH_CALUDE_A_intersect_B_range_of_a_l3475_347572


namespace NUMINAMATH_CALUDE_number_of_girls_in_group_l3475_347592

theorem number_of_girls_in_group (girls_avg_weight : ℝ) (boys_avg_weight : ℝ) 
  (total_avg_weight : ℝ) (num_boys : ℕ) (total_students : ℕ) :
  girls_avg_weight = 45 →
  boys_avg_weight = 55 →
  num_boys = 5 →
  total_students = 10 →
  total_avg_weight = 50 →
  ∃ (num_girls : ℕ), num_girls = 5 ∧ 
    (girls_avg_weight * num_girls + boys_avg_weight * num_boys) / total_students = total_avg_weight :=
by sorry

end NUMINAMATH_CALUDE_number_of_girls_in_group_l3475_347592


namespace NUMINAMATH_CALUDE_matrix_product_is_zero_l3475_347516

open Matrix

/-- Given two 3x3 matrices A and B, prove that their product is the zero matrix --/
theorem matrix_product_is_zero (a b c : ℝ) :
  let A : Matrix (Fin 3) (Fin 3) ℝ := !![0, 2*c, -2*b; -2*c, 0, 2*a; 2*b, -2*a, 0]
  let B : Matrix (Fin 3) (Fin 3) ℝ := !![2*a^2, a^2+b^2, a^2+c^2; a^2+b^2, 2*b^2, b^2+c^2; a^2+c^2, b^2+c^2, 2*c^2]
  A * B = 0 := by
  sorry

#check matrix_product_is_zero

end NUMINAMATH_CALUDE_matrix_product_is_zero_l3475_347516


namespace NUMINAMATH_CALUDE_two_numbers_difference_l3475_347577

theorem two_numbers_difference (a b : ℕ) : 
  a + b = 20000 →
  b % 5 = 0 →
  b / 10 = a →
  (b % 10 = 0 ∨ b % 10 = 5) →
  b - a = 16358 := by
sorry

end NUMINAMATH_CALUDE_two_numbers_difference_l3475_347577


namespace NUMINAMATH_CALUDE_solution_equality_l3475_347510

theorem solution_equality (a : ℝ) : 
  (∃ x, 2 - a - x = 0 ∧ 2*x + 1 = 3) → a = 1 := by
sorry

end NUMINAMATH_CALUDE_solution_equality_l3475_347510


namespace NUMINAMATH_CALUDE_juan_has_498_marbles_l3475_347500

/-- The number of marbles Connie has -/
def connies_marbles : ℕ := 323

/-- The number of additional marbles Juan has compared to Connie -/
def juans_additional_marbles : ℕ := 175

/-- The total number of marbles Juan has -/
def juans_marbles : ℕ := connies_marbles + juans_additional_marbles

/-- Theorem stating that Juan has 498 marbles -/
theorem juan_has_498_marbles : juans_marbles = 498 := by
  sorry

end NUMINAMATH_CALUDE_juan_has_498_marbles_l3475_347500


namespace NUMINAMATH_CALUDE_number_of_subsets_of_three_element_set_l3475_347586

theorem number_of_subsets_of_three_element_set :
  Finset.card (Finset.powerset {1, 2, 3}) = 8 := by
  sorry

end NUMINAMATH_CALUDE_number_of_subsets_of_three_element_set_l3475_347586


namespace NUMINAMATH_CALUDE_circular_arrangement_problem_l3475_347529

/-- Represents a circular arrangement of 6 numbers -/
structure CircularArrangement where
  numbers : Fin 6 → ℕ
  sum_rule : ∀ i : Fin 6, numbers i + numbers (i + 1) = 2 * numbers ((i + 2) % 6)

theorem circular_arrangement_problem 
  (arr : CircularArrangement)
  (h1 : ∃ i : Fin 6, arr.numbers i = 15 ∧ arr.numbers ((i + 1) % 6) + arr.numbers ((i + 5) % 6) = 16)
  (h2 : ∃ j : Fin 6, arr.numbers j + arr.numbers ((j + 2) % 6) = 10) :
  ∃ k : Fin 6, arr.numbers k = 7 ∧ arr.numbers ((k + 1) % 6) + arr.numbers ((k + 5) % 6) = 10 :=
sorry

end NUMINAMATH_CALUDE_circular_arrangement_problem_l3475_347529


namespace NUMINAMATH_CALUDE_gasoline_price_increase_l3475_347564

theorem gasoline_price_increase 
  (spending_increase : Real) 
  (quantity_decrease : Real) 
  (price_increase : Real) : 
  spending_increase = 0.15 → 
  quantity_decrease = 0.08000000000000007 → 
  (1 + price_increase) * (1 - quantity_decrease) = 1 + spending_increase → 
  price_increase = 0.25 := by
sorry

end NUMINAMATH_CALUDE_gasoline_price_increase_l3475_347564


namespace NUMINAMATH_CALUDE_line_in_plane_if_points_in_plane_l3475_347515

-- Define the types for our geometric objects
variable (α : Type) [LinearOrderedField α]
variable (Point : Type) (Line : Type) (Plane : Type)

-- Define the relationships between geometric objects
variable (on_line : Point → Line → Prop)
variable (on_plane : Point → Plane → Prop)
variable (line_in_plane : Line → Plane → Prop)

-- State the theorem
theorem line_in_plane_if_points_in_plane 
  (a b l : Line) (α : Plane) (M N : Point) :
  line_in_plane a α →
  line_in_plane b α →
  on_line M a →
  on_line N b →
  on_line M l →
  on_line N l →
  line_in_plane l α :=
sorry

end NUMINAMATH_CALUDE_line_in_plane_if_points_in_plane_l3475_347515


namespace NUMINAMATH_CALUDE_curve_inequality_l3475_347542

-- Define the logarithm function (base 10)
noncomputable def lg (x : ℝ) : ℝ := Real.log x / Real.log 10

-- Define the curve equation
def curve_equation (a b c x y : ℝ) : Prop :=
  a * (lg x)^2 + 2 * b * (lg x) * (lg y) + c * (lg y)^2 = 1

-- Main theorem
theorem curve_inequality (a b c : ℝ) 
  (h1 : b^2 - a*c < 0) 
  (h2 : curve_equation a b c 10 (1/10)) :
  ∀ x y : ℝ, curve_equation a b c x y →
  -1 / Real.sqrt (a*c - b^2) ≤ lg (x*y) ∧ lg (x*y) ≤ 1 / Real.sqrt (a*c - b^2) := by
  sorry

end NUMINAMATH_CALUDE_curve_inequality_l3475_347542


namespace NUMINAMATH_CALUDE_expression_value_l3475_347593

theorem expression_value (a b c d m : ℝ) 
  (h1 : a = -b)  -- a and b are opposite numbers
  (h2 : c * d = 1)  -- c and d are reciprocals
  (h3 : |m| = 2)  -- absolute value of m is 2
  : (a + b) / (4 * m) + m^2 - 3 * c * d = 1 := by
  sorry

end NUMINAMATH_CALUDE_expression_value_l3475_347593


namespace NUMINAMATH_CALUDE_tesseract_sum_l3475_347514

/-- A tesseract is a 4-dimensional hypercube -/
structure Tesseract where

/-- The number of edges in a tesseract -/
def Tesseract.edges (t : Tesseract) : ℕ := 32

/-- The number of vertices in a tesseract -/
def Tesseract.vertices (t : Tesseract) : ℕ := 16

/-- The number of faces in a tesseract -/
def Tesseract.faces (t : Tesseract) : ℕ := 24

/-- The sum of edges, vertices, and faces in a tesseract is 72 -/
theorem tesseract_sum (t : Tesseract) : 
  t.edges + t.vertices + t.faces = 72 := by sorry

end NUMINAMATH_CALUDE_tesseract_sum_l3475_347514


namespace NUMINAMATH_CALUDE_inscribed_squares_ratio_l3475_347540

theorem inscribed_squares_ratio (a b c x y : ℝ) : 
  a = 5 → b = 12 → c = 13 → 
  a^2 + b^2 = c^2 →
  x * (a + b - x) = a * b →
  y * (c - y) = (a - y) * (b - y) →
  x / y = 5 / 13 := by sorry

end NUMINAMATH_CALUDE_inscribed_squares_ratio_l3475_347540


namespace NUMINAMATH_CALUDE_fraction_to_decimal_l3475_347566

theorem fraction_to_decimal :
  (7 : ℚ) / 12 = 0.5833333333333333333333333333333333 :=
sorry

end NUMINAMATH_CALUDE_fraction_to_decimal_l3475_347566


namespace NUMINAMATH_CALUDE_smallest_number_with_given_remainders_l3475_347573

theorem smallest_number_with_given_remainders :
  ∃ n : ℕ, (n % 2 = 1) ∧ (n % 3 = 2) ∧ (n % 4 = 3) ∧
  (∀ m : ℕ, m < n → ¬((m % 2 = 1) ∧ (m % 3 = 2) ∧ (m % 4 = 3))) ∧
  n = 11 := by
  sorry

end NUMINAMATH_CALUDE_smallest_number_with_given_remainders_l3475_347573


namespace NUMINAMATH_CALUDE_function_properties_l3475_347501

noncomputable def f (a b m x : ℝ) : ℝ := 2 * x^3 + a * x^2 + b * x + m

noncomputable def f' (a b x : ℝ) : ℝ := 6 * x^2 + 2 * a * x + b

theorem function_properties (a b m : ℝ) :
  (∀ x : ℝ, f' a b x = f' a b (-1 - x)) →  -- Symmetry about x = -1/2
  f' a b 1 = 0 →                           -- f'(1) = 0
  a = 3 ∧ b = -12 ∧                        -- Values of a and b
  (∃ x₁ x₂ x₃ : ℝ, x₁ < x₂ ∧ x₂ < x₃ ∧     -- Exactly three zeros
    f a b m x₁ = 0 ∧ f a b m x₂ = 0 ∧ f a b m x₃ = 0 ∧
    (∀ x : ℝ, f a b m x = 0 → x = x₁ ∨ x = x₂ ∨ x = x₃)) →
  -20 < m ∧ m < 7                          -- Range of m
  := by sorry

end NUMINAMATH_CALUDE_function_properties_l3475_347501


namespace NUMINAMATH_CALUDE_percentage_reading_two_novels_l3475_347506

theorem percentage_reading_two_novels
  (total_students : ℕ)
  (three_or_more : ℚ)
  (one_novel : ℚ)
  (no_novels : ℕ)
  (h1 : total_students = 240)
  (h2 : three_or_more = 1 / 6)
  (h3 : one_novel = 5 / 12)
  (h4 : no_novels = 16) :
  (total_students - (three_or_more * total_students).num - (one_novel * total_students).num - no_novels : ℚ) / total_students * 100 = 35 := by
sorry


end NUMINAMATH_CALUDE_percentage_reading_two_novels_l3475_347506


namespace NUMINAMATH_CALUDE_simplify_sqrt_product_l3475_347521

theorem simplify_sqrt_product : 
  Real.sqrt (5 * 3) * Real.sqrt (3^5 * 5^2) = 135 * Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_simplify_sqrt_product_l3475_347521


namespace NUMINAMATH_CALUDE_mortgage_payment_sum_l3475_347575

theorem mortgage_payment_sum (a₁ : ℝ) (r : ℝ) (n : ℕ) (h1 : a₁ = 400) (h2 : r = 2) (h3 : n = 11) :
  a₁ * (1 - r^n) / (1 - r) = 819400 := by
  sorry

end NUMINAMATH_CALUDE_mortgage_payment_sum_l3475_347575


namespace NUMINAMATH_CALUDE_exists_non_identity_same_image_l3475_347565

/-- Given two finite groups G and H, and two surjective but non-injective homomorphisms φ and ψ from G to H,
    there exists a non-identity element g in G such that φ(g) = ψ(g). -/
theorem exists_non_identity_same_image 
  {G H : Type*} [Group G] [Group H] [Fintype G] [Fintype H]
  (φ ψ : G →* H) 
  (hφ_surj : Function.Surjective φ) (hψ_surj : Function.Surjective ψ)
  (hφ_non_inj : ¬Function.Injective φ) (hψ_non_inj : ¬Function.Injective ψ) :
  ∃ g : G, g ≠ 1 ∧ φ g = ψ g := by
  sorry

end NUMINAMATH_CALUDE_exists_non_identity_same_image_l3475_347565


namespace NUMINAMATH_CALUDE_third_number_5_4_l3475_347581

/-- Represents the decomposition of a natural number raised to a power -/
def decomposition (base : ℕ) (exponent : ℕ) : List ℕ :=
  sorry

/-- The third element in a list, if it exists -/
def thirdElement (l : List ℕ) : Option ℕ :=
  match l with
  | _ :: _ :: x :: _ => some x
  | _ => none

/-- Theorem stating that the third number in the decomposition of 5^4 is 125 -/
theorem third_number_5_4 : 
  thirdElement (decomposition 5 4) = some 125 := by sorry

end NUMINAMATH_CALUDE_third_number_5_4_l3475_347581


namespace NUMINAMATH_CALUDE_function_symmetry_and_value_l3475_347597

/-- Given a function f(x) = 2cos(ωx + φ) + m with ω > 0, 
    if f(π/4 - t) = f(t) for all real t and f(π/8) = -1, 
    then m = -3 or m = 1 -/
theorem function_symmetry_and_value (ω φ m : ℝ) (h_ω : ω > 0) 
  (f : ℝ → ℝ) (h_f : ∀ x, f x = 2 * Real.cos (ω * x + φ) + m) 
  (h_sym : ∀ t, f (π/4 - t) = f t) (h_val : f (π/8) = -1) : 
  m = -3 ∨ m = 1 := by
  sorry

end NUMINAMATH_CALUDE_function_symmetry_and_value_l3475_347597


namespace NUMINAMATH_CALUDE_inequality_proof_l3475_347576

theorem inequality_proof (x y z : ℝ) (hx : x ≥ 1) (hy : y ≥ 1) (hz : z ≥ 1) :
  (x^3 + 2*y^2 + 3*z) * (4*y^3 + 5*z^2 + 6*x) * (7*z^3 + 8*x^2 + 9*y) ≥ 720 * (x*y + y*z + x*z) :=
by sorry

end NUMINAMATH_CALUDE_inequality_proof_l3475_347576


namespace NUMINAMATH_CALUDE_election_winner_percentage_l3475_347546

theorem election_winner_percentage (total_votes : ℕ) (winner_votes : ℕ) (margin : ℕ) : 
  winner_votes = 3744 →
  margin = 288 →
  total_votes = winner_votes + (winner_votes - margin) →
  (winner_votes : ℚ) / (total_votes : ℚ) = 52 / 100 := by
sorry

end NUMINAMATH_CALUDE_election_winner_percentage_l3475_347546


namespace NUMINAMATH_CALUDE_special_number_unique_l3475_347503

/-- The unique three-digit positive integer that is one more than a multiple of 3, 4, 5, 6, and 7 -/
def special_number : ℕ := 421

/-- Predicate to check if a number is a three-digit positive integer -/
def is_three_digit (n : ℕ) : Prop := 100 ≤ n ∧ n ≤ 999

/-- Predicate to check if a number is one more than a multiple of 3, 4, 5, 6, and 7 -/
def is_special (n : ℕ) : Prop :=
  ∃ (k : ℕ), n = 3 * k + 1 ∧ n = 4 * k + 1 ∧ n = 5 * k + 1 ∧ n = 6 * k + 1 ∧ n = 7 * k + 1

theorem special_number_unique :
  is_three_digit special_number ∧
  is_special special_number ∧
  ∀ (n : ℕ), is_three_digit n → is_special n → n = special_number :=
sorry

end NUMINAMATH_CALUDE_special_number_unique_l3475_347503


namespace NUMINAMATH_CALUDE_no_constant_term_in_expansion_l3475_347538

theorem no_constant_term_in_expansion :
  let expression := (fun x => (5 * x^2 + 2 / x)^8)
  ∀ c : ℝ, (∀ x : ℝ, x ≠ 0 → expression x = c) → c = 0 :=
by sorry

end NUMINAMATH_CALUDE_no_constant_term_in_expansion_l3475_347538


namespace NUMINAMATH_CALUDE_pucks_not_in_original_position_l3475_347561

/-- Represents the arrangement of three objects -/
inductive Arrangement
  | Clockwise
  | Counterclockwise

/-- Represents a single hit that changes the arrangement -/
def hit (a : Arrangement) : Arrangement :=
  match a with
  | Arrangement.Clockwise => Arrangement.Counterclockwise
  | Arrangement.Counterclockwise => Arrangement.Clockwise

/-- Applies n hits to the initial arrangement -/
def applyHits (initial : Arrangement) (n : Nat) : Arrangement :=
  match n with
  | 0 => initial
  | n + 1 => hit (applyHits initial n)

theorem pucks_not_in_original_position (initial : Arrangement) :
  applyHits initial 25 ≠ initial := by
  sorry


end NUMINAMATH_CALUDE_pucks_not_in_original_position_l3475_347561


namespace NUMINAMATH_CALUDE_hcd_7350_150_minus_12_l3475_347543

theorem hcd_7350_150_minus_12 : Nat.gcd 7350 150 - 12 = 138 := by
  sorry

end NUMINAMATH_CALUDE_hcd_7350_150_minus_12_l3475_347543


namespace NUMINAMATH_CALUDE_xiaogang_dart_game_l3475_347588

theorem xiaogang_dart_game :
  ∀ (x y z : ℕ),
    x + y + z > 11 →
    8 * x + 9 * y + 10 * z = 100 →
    (x + y + z = 12 ∧ (x = 10 ∧ y = 0 ∧ z = 2) ∨ (x = 9 ∧ y = 2 ∧ z = 1) ∨ (x = 8 ∧ y = 4 ∧ z = 0)) :=
by
  sorry

end NUMINAMATH_CALUDE_xiaogang_dart_game_l3475_347588


namespace NUMINAMATH_CALUDE_unique_solution_range_l3475_347587

theorem unique_solution_range (a : ℝ) : 
  (∃! x : ℕ, x^2 - (a+2)*x + 2 - a < 0) → 
  (1/2 < a ∧ a ≤ 2/3) :=
by sorry

end NUMINAMATH_CALUDE_unique_solution_range_l3475_347587


namespace NUMINAMATH_CALUDE_min_value_of_fraction_l3475_347551

theorem min_value_of_fraction (a b : ℝ) (ha : a < 0) (hb : b < 0) :
  (∀ x y : ℝ, x < 0 → y < 0 → a / (a + 2*b) + b / (a + b) ≥ x / (x + 2*y) + y / (x + y)) →
  a / (a + 2*b) + b / (a + b) = 2 * (Real.sqrt 2 - 1) :=
sorry

end NUMINAMATH_CALUDE_min_value_of_fraction_l3475_347551


namespace NUMINAMATH_CALUDE_calculation_part1_calculation_part2_l3475_347599

-- Part 1
theorem calculation_part1 : 
  (1/8)^(-(2/3)) - 4*(-3)^4 + (2 + 1/4)^(1/2) - (1.5)^2 = -320.75 := by sorry

-- Part 2
theorem calculation_part2 : 
  (Real.log 5 / Real.log 10)^2 + (Real.log 2 / Real.log 10) * (Real.log 50 / Real.log 10) - 
  (Real.log 8 / Real.log (1/2)) + (Real.log (427/3) / Real.log 3) = 
  1 - (Real.log 5 / Real.log 10)^2 + (Real.log 5 / Real.log 10) + (Real.log 2 / Real.log 10) + 2 := by sorry

end NUMINAMATH_CALUDE_calculation_part1_calculation_part2_l3475_347599


namespace NUMINAMATH_CALUDE_james_oreos_count_l3475_347568

/-- The number of Oreos Jordan has -/
def jordan_oreos : ℕ := sorry

/-- The number of Oreos James has -/
def james_oreos : ℕ := 4 * jordan_oreos + 7

/-- The total number of Oreos -/
def total_oreos : ℕ := 52

theorem james_oreos_count : james_oreos = 43 := by
  sorry

end NUMINAMATH_CALUDE_james_oreos_count_l3475_347568


namespace NUMINAMATH_CALUDE_days_worked_by_c_l3475_347549

-- Define the problem parameters
def days_a : ℕ := 6
def days_b : ℕ := 9
def wage_ratio_a : ℕ := 3
def wage_ratio_b : ℕ := 4
def wage_ratio_c : ℕ := 5
def daily_wage_c : ℕ := 100
def total_earning : ℕ := 1480

-- Theorem statement
theorem days_worked_by_c :
  ∃ (days_c : ℕ),
    days_c * daily_wage_c +
    days_a * (daily_wage_c * wage_ratio_a / wage_ratio_c) +
    days_b * (daily_wage_c * wage_ratio_b / wage_ratio_c) = total_earning ∧
    days_c = 4 := by
  sorry


end NUMINAMATH_CALUDE_days_worked_by_c_l3475_347549


namespace NUMINAMATH_CALUDE_sum_zero_implies_product_nonpositive_l3475_347537

theorem sum_zero_implies_product_nonpositive (a b c : ℝ) (h : a + b + c = 0) :
  a * b + a * c + b * c ≤ 0 := by sorry

end NUMINAMATH_CALUDE_sum_zero_implies_product_nonpositive_l3475_347537


namespace NUMINAMATH_CALUDE_existence_of_non_divisible_pair_l3475_347528

theorem existence_of_non_divisible_pair (p : Nat) (h_prime : Prime p) (h_p_gt_3 : p > 3) :
  ∃ n : Nat, n > 0 ∧ n < p - 1 ∧
    ¬(p^2 ∣ n^(p-1) - 1) ∧ ¬(p^2 ∣ (n+1)^(p-1) - 1) := by
  sorry

end NUMINAMATH_CALUDE_existence_of_non_divisible_pair_l3475_347528


namespace NUMINAMATH_CALUDE_max_value_of_f_l3475_347533

-- Define the function
def f (x : ℝ) : ℝ := 5 * x - 4 * x^2 + 6

-- State the theorem
theorem max_value_of_f :
  ∃ (max : ℝ), (∀ (x : ℝ), f x ≤ max) ∧ (max = 121 / 16) := by
  sorry

end NUMINAMATH_CALUDE_max_value_of_f_l3475_347533


namespace NUMINAMATH_CALUDE_arithmetic_mean_min_value_l3475_347535

theorem arithmetic_mean_min_value (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : Real.sqrt (a * b) = 1) :
  (a + b) / 2 ≥ 1 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_mean_min_value_l3475_347535


namespace NUMINAMATH_CALUDE_smallest_number_in_special_set_l3475_347512

theorem smallest_number_in_special_set (a b c : ℕ) : 
  a > 0 ∧ b > 0 ∧ c > 0 →
  (a + b + c) / 3 = 26 →
  b = 27 →
  c = b + 5 →
  a < b ∧ b < c →
  a = 19 := by
sorry

end NUMINAMATH_CALUDE_smallest_number_in_special_set_l3475_347512


namespace NUMINAMATH_CALUDE_binomial_coefficient_two_l3475_347578

theorem binomial_coefficient_two (n : ℕ+) : Nat.choose n 2 = n * (n - 1) / 2 := by
  sorry

end NUMINAMATH_CALUDE_binomial_coefficient_two_l3475_347578


namespace NUMINAMATH_CALUDE_sum_digits_base5_588_l3475_347544

/-- Converts a natural number from base 10 to base 5 -/
def toBase5 (n : ℕ) : List ℕ :=
  sorry

/-- Sums the digits in a list -/
def sumDigits (digits : List ℕ) : ℕ :=
  sorry

theorem sum_digits_base5_588 :
  sumDigits (toBase5 588) = 12 := by
  sorry

end NUMINAMATH_CALUDE_sum_digits_base5_588_l3475_347544


namespace NUMINAMATH_CALUDE_min_value_reciprocal_sum_l3475_347585

theorem min_value_reciprocal_sum (x y : ℝ) (hx : x > 0) (hy : y > 0) (hxy : x * y = 2) :
  1 / x + 2 / y ≥ 2 ∧ ∃ (x₀ y₀ : ℝ), x₀ > 0 ∧ y₀ > 0 ∧ x₀ * y₀ = 2 ∧ 1 / x₀ + 2 / y₀ = 2 := by
  sorry

end NUMINAMATH_CALUDE_min_value_reciprocal_sum_l3475_347585


namespace NUMINAMATH_CALUDE_optimal_arrangement_l3475_347596

/-- Represents the housekeeping service company scenario -/
structure CleaningCompany where
  total_cleaners : ℕ
  large_rooms_per_cleaner : ℕ
  small_rooms_per_cleaner : ℕ
  large_room_payment : ℕ
  small_room_payment : ℕ

/-- Calculates the daily income based on the number of cleaners assigned to large rooms -/
def daily_income (company : CleaningCompany) (x : ℕ) : ℕ :=
  company.large_room_payment * company.large_rooms_per_cleaner * x +
  company.small_room_payment * company.small_rooms_per_cleaner * (company.total_cleaners - x)

/-- The main theorem to prove -/
theorem optimal_arrangement (company : CleaningCompany) (x : ℕ) :
  company.total_cleaners = 16 ∧
  company.large_rooms_per_cleaner = 4 ∧
  company.small_rooms_per_cleaner = 5 ∧
  company.large_room_payment = 80 ∧
  company.small_room_payment = 60 ∧
  x = 10 →
  daily_income company x = 5000 := by sorry

end NUMINAMATH_CALUDE_optimal_arrangement_l3475_347596


namespace NUMINAMATH_CALUDE_volleyball_team_selection_l3475_347508

def total_players : ℕ := 18
def quadruplets : ℕ := 4
def starters : ℕ := 8

theorem volleyball_team_selection :
  Nat.choose (total_players - quadruplets) (starters - quadruplets) = 1001 :=
by sorry

end NUMINAMATH_CALUDE_volleyball_team_selection_l3475_347508


namespace NUMINAMATH_CALUDE_final_S_value_l3475_347524

/-- Calculates the final value of S after executing the loop three times -/
def final_S : ℕ → ℕ → ℕ → ℕ
| 0, s, i => s
| (n + 1), s, i => final_S n (s + i) (i + 2)

theorem final_S_value :
  final_S 3 0 1 = 9 := by
sorry

end NUMINAMATH_CALUDE_final_S_value_l3475_347524


namespace NUMINAMATH_CALUDE_school_play_tickets_l3475_347563

theorem school_play_tickets (total_money : ℕ) (adult_price : ℕ) (child_price : ℕ) (total_tickets : ℕ) :
  total_money = 104 →
  adult_price = 6 →
  child_price = 4 →
  total_tickets = 21 →
  ∃ (adult_tickets : ℕ) (child_tickets : ℕ),
    adult_tickets + child_tickets = total_tickets ∧
    adult_tickets * adult_price + child_tickets * child_price = total_money ∧
    child_tickets = 11 :=
by
  sorry

end NUMINAMATH_CALUDE_school_play_tickets_l3475_347563


namespace NUMINAMATH_CALUDE_cos_x_plus_2y_equals_one_l3475_347520

theorem cos_x_plus_2y_equals_one 
  (x y a : ℝ) 
  (h1 : x ∈ Set.Icc (-π/4) (π/4))
  (h2 : y ∈ Set.Icc (-π/4) (π/4))
  (h3 : x^3 + Real.sin x - 2*a = 0)
  (h4 : 4*y^3 + Real.sin y * Real.cos y + a = 0) :
  Real.cos (x + 2*y) = 1 := by
  sorry

end NUMINAMATH_CALUDE_cos_x_plus_2y_equals_one_l3475_347520


namespace NUMINAMATH_CALUDE_hyperbola_intersection_line_l3475_347513

-- Define the hyperbola
def hyperbola (x y : ℝ) : Prop := x^2 - y^2/3 = 1

-- Define the line
def line (x y : ℝ) : Prop := y = x + 2

-- Define the midpoint condition
def is_midpoint (x₁ y₁ x₂ y₂ xₘ yₘ : ℝ) : Prop :=
  xₘ = (x₁ + x₂) / 2 ∧ yₘ = (y₁ + y₂) / 2

-- Main theorem
theorem hyperbola_intersection_line :
  ∀ (x₁ y₁ x₂ y₂ : ℝ),
    hyperbola x₁ y₁ →
    hyperbola x₂ y₂ →
    is_midpoint x₁ y₁ x₂ y₂ 1 3 →
    (∀ x y, line x y ↔ (y - 3 = x - 1)) :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_intersection_line_l3475_347513


namespace NUMINAMATH_CALUDE_unique_integer_solution_l3475_347595

theorem unique_integer_solution : ∃! (n : ℤ), n + 10 > 11 ∧ -4*n > -12 := by
  sorry

end NUMINAMATH_CALUDE_unique_integer_solution_l3475_347595


namespace NUMINAMATH_CALUDE_min_value_inequality_l3475_347509

theorem min_value_inequality (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0)
  (h_sum : x + 2*y + 3*z = 12) :
  9/x + 4/y + 1/z ≥ 49/12 := by
  sorry

end NUMINAMATH_CALUDE_min_value_inequality_l3475_347509


namespace NUMINAMATH_CALUDE_isosceles_triangle_perimeter_isosceles_triangle_perimeter_proof_l3475_347522

/-- An isosceles triangle with two sides of length 12 and one side of length 17 has a perimeter of 41. -/
theorem isosceles_triangle_perimeter : ℝ → Prop :=
  fun perimeter =>
    ∃ (side1 side2 base : ℝ),
      side1 = 12 ∧
      side2 = 12 ∧
      base = 17 ∧
      perimeter = side1 + side2 + base ∧
      perimeter = 41

/-- Proof of the theorem -/
theorem isosceles_triangle_perimeter_proof : isosceles_triangle_perimeter 41 := by
  sorry

end NUMINAMATH_CALUDE_isosceles_triangle_perimeter_isosceles_triangle_perimeter_proof_l3475_347522


namespace NUMINAMATH_CALUDE_brandy_safe_caffeine_l3475_347590

/-- The maximum safe amount of caffeine that can be consumed per day (in mg) -/
def max_safe_caffeine : ℕ := 500

/-- The amount of caffeine in each energy drink (in mg) -/
def caffeine_per_drink : ℕ := 120

/-- The number of energy drinks Brandy consumed -/
def drinks_consumed : ℕ := 4

/-- The remaining amount of caffeine Brandy can safely consume (in mg) -/
def remaining_safe_caffeine : ℕ := max_safe_caffeine - (caffeine_per_drink * drinks_consumed)

theorem brandy_safe_caffeine : remaining_safe_caffeine = 20 := by
  sorry

end NUMINAMATH_CALUDE_brandy_safe_caffeine_l3475_347590


namespace NUMINAMATH_CALUDE_unique_solution_for_k_l3475_347571

/-- The equation (2x + 3)/(kx - 2) = x has exactly one solution when k = -4/3 -/
theorem unique_solution_for_k (k : ℚ) : 
  (∃! x, (2 * x + 3) / (k * x - 2) = x) ↔ k = -4/3 := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_for_k_l3475_347571


namespace NUMINAMATH_CALUDE_limit_equals_one_implies_a_and_b_l3475_347594

/-- Given that a and b are constants such that the limit of (ln(2-x))^2 / (x^2 + ax + b) as x approaches 1 is equal to 1, prove that a = -2 and b = 1. -/
theorem limit_equals_one_implies_a_and_b (a b : ℝ) :
  (∀ ε > 0, ∃ δ > 0, ∀ x : ℝ, 0 < |x - 1| ∧ |x - 1| < δ →
    |((Real.log (2 - x))^2) / (x^2 + a*x + b) - 1| < ε) →
  a = -2 ∧ b = 1 := by
  sorry

end NUMINAMATH_CALUDE_limit_equals_one_implies_a_and_b_l3475_347594


namespace NUMINAMATH_CALUDE_difference_in_tickets_l3475_347517

def tickets_for_toys : ℕ := 31
def tickets_for_clothes : ℕ := 14

theorem difference_in_tickets : tickets_for_toys - tickets_for_clothes = 17 := by
  sorry

end NUMINAMATH_CALUDE_difference_in_tickets_l3475_347517


namespace NUMINAMATH_CALUDE_number_division_problem_l3475_347523

theorem number_division_problem (x y : ℝ) 
  (h1 : (x - 5) / 7 = 7)
  (h2 : (x - 2) / y = 4) : 
  y = 13 := by
sorry

end NUMINAMATH_CALUDE_number_division_problem_l3475_347523


namespace NUMINAMATH_CALUDE_find_A_l3475_347558

theorem find_A : ∀ A : ℕ, (A / 9 = 2 ∧ A % 9 = 6) → A = 24 := by
  sorry

end NUMINAMATH_CALUDE_find_A_l3475_347558


namespace NUMINAMATH_CALUDE_x_plus_reciprocal_three_l3475_347539

theorem x_plus_reciprocal_three (x : ℝ) (h : x ≠ 0) :
  x + 1/x = 3 →
  (x - 1)^2 + 16/(x - 1)^2 = x + 16/x :=
by
  sorry

end NUMINAMATH_CALUDE_x_plus_reciprocal_three_l3475_347539


namespace NUMINAMATH_CALUDE_smallest_non_prime_non_square_no_small_factors_l3475_347530

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d ∣ n → d = 1 ∨ d = n

def is_square (n : ℕ) : Prop := ∃ m : ℕ, n = m * m

def has_prime_factor_less_than_60 (n : ℕ) : Prop :=
  ∃ p : ℕ, is_prime p ∧ p < 60 ∧ p ∣ n

theorem smallest_non_prime_non_square_no_small_factors : 
  (∀ n : ℕ, n < 4087 → is_prime n ∨ is_square n ∨ has_prime_factor_less_than_60 n) ∧
  ¬is_prime 4087 ∧
  ¬is_square 4087 ∧
  ¬has_prime_factor_less_than_60 4087 :=
sorry

end NUMINAMATH_CALUDE_smallest_non_prime_non_square_no_small_factors_l3475_347530


namespace NUMINAMATH_CALUDE_x_equation_implies_zero_l3475_347507

theorem x_equation_implies_zero (x : ℝ) (h : x + 1/x = Real.sqrt 5) :
  x^11 - 7*x^7 + x^3 = 0 := by
  sorry

end NUMINAMATH_CALUDE_x_equation_implies_zero_l3475_347507


namespace NUMINAMATH_CALUDE_ellipse_center_x_coordinate_l3475_347550

/-- An ellipse in the first quadrant tangent to both axes with foci at (3,4) and (3,12) -/
structure Ellipse where
  /-- The ellipse is in the first quadrant -/
  first_quadrant : True
  /-- The ellipse is tangent to both x-axis and y-axis -/
  tangent_to_axes : True
  /-- One focus is at (3,4) -/
  focus1 : ℝ × ℝ := (3, 4)
  /-- The other focus is at (3,12) -/
  focus2 : ℝ × ℝ := (3, 12)

/-- The x-coordinate of the center of the ellipse is 3 -/
theorem ellipse_center_x_coordinate (e : Ellipse) : ∃ (y : ℝ), e.focus1.1 = 3 ∧ e.focus2.1 = 3 := by
  sorry

end NUMINAMATH_CALUDE_ellipse_center_x_coordinate_l3475_347550


namespace NUMINAMATH_CALUDE_historical_fiction_new_releases_fraction_l3475_347591

/-- Represents the inventory composition and new release percentages for a bookstore. -/
structure BookstoreInventory where
  historicalFictionPercentage : Float
  scienceFictionPercentage : Float
  biographiesPercentage : Float
  mysteryNovelsPercentage : Float
  historicalFictionNewReleasePercentage : Float
  scienceFictionNewReleasePercentage : Float
  biographiesNewReleasePercentage : Float
  mysteryNovelsNewReleasePercentage : Float

/-- Calculates the fraction of all new releases that are historical fiction new releases. -/
def historicalFictionNewReleasesFraction (inventory : BookstoreInventory) : Float :=
  let totalNewReleases := 
    inventory.historicalFictionPercentage * inventory.historicalFictionNewReleasePercentage +
    inventory.scienceFictionPercentage * inventory.scienceFictionNewReleasePercentage +
    inventory.biographiesPercentage * inventory.biographiesNewReleasePercentage +
    inventory.mysteryNovelsPercentage * inventory.mysteryNovelsNewReleasePercentage
  let historicalFictionNewReleases := 
    inventory.historicalFictionPercentage * inventory.historicalFictionNewReleasePercentage
  historicalFictionNewReleases / totalNewReleases

/-- Theorem stating that the fraction of all new releases that are historical fiction new releases is 9/20. -/
theorem historical_fiction_new_releases_fraction :
  let inventory := BookstoreInventory.mk 0.40 0.25 0.15 0.20 0.45 0.30 0.50 0.35
  historicalFictionNewReleasesFraction inventory = 9/20 := by
  sorry

end NUMINAMATH_CALUDE_historical_fiction_new_releases_fraction_l3475_347591


namespace NUMINAMATH_CALUDE_garden_transformation_cost_and_area_increase_l3475_347504

/-- Represents a rectangular garden with its dimensions and fence cost -/
structure RectGarden where
  length : ℝ
  width : ℝ
  fence_cost : ℝ

/-- Represents a square garden with its side length and fence cost -/
structure SquareGarden where
  side : ℝ
  fence_cost : ℝ

/-- Calculates the perimeter of a rectangular garden -/
def rect_perimeter (g : RectGarden) : ℝ :=
  2 * (g.length + g.width)

/-- Calculates the area of a rectangular garden -/
def rect_area (g : RectGarden) : ℝ :=
  g.length * g.width

/-- Calculates the total fencing cost of a rectangular garden -/
def rect_fence_cost (g : RectGarden) : ℝ :=
  rect_perimeter g * g.fence_cost

/-- Calculates the area of a square garden -/
def square_area (g : SquareGarden) : ℝ :=
  g.side * g.side

/-- Calculates the total fencing cost of a square garden -/
def square_fence_cost (g : SquareGarden) : ℝ :=
  4 * g.side * g.fence_cost

/-- The main theorem to prove -/
theorem garden_transformation_cost_and_area_increase :
  let rect := RectGarden.mk 60 20 15
  let square := SquareGarden.mk (rect_perimeter rect / 4) 20
  square_fence_cost square - rect_fence_cost rect = 800 ∧
  square_area square - rect_area rect = 400 := by
  sorry


end NUMINAMATH_CALUDE_garden_transformation_cost_and_area_increase_l3475_347504


namespace NUMINAMATH_CALUDE_sugar_amount_in_recipe_l3475_347541

/-- Given a recipe that requires a total of 10 cups of flour, 
    with 2 cups already added, and the remaining flour needed 
    being 5 cups more than the amount of sugar, 
    prove that the recipe calls for 3 cups of sugar. -/
theorem sugar_amount_in_recipe 
  (total_flour : ℕ) 
  (added_flour : ℕ) 
  (sugar : ℕ) : 
  total_flour = 10 → 
  added_flour = 2 → 
  total_flour = added_flour + (sugar + 5) → 
  sugar = 3 := by
  sorry

end NUMINAMATH_CALUDE_sugar_amount_in_recipe_l3475_347541


namespace NUMINAMATH_CALUDE_quadratic_increasing_condition_l3475_347560

/-- The function f(x) = ax^2 - 2x + 1 is increasing on [1, 2] iff a > 0 and 1/a < 1 -/
theorem quadratic_increasing_condition (a : ℝ) :
  (∀ x ∈ Set.Icc 1 2, Monotone (fun x => a * x^2 - 2 * x + 1)) ↔ (a > 0 ∧ 1 / a < 1) := by
  sorry


end NUMINAMATH_CALUDE_quadratic_increasing_condition_l3475_347560


namespace NUMINAMATH_CALUDE_toy_shopping_total_l3475_347580

def calculate_total_spent (prices : List Float) (discount_rate : Float) (tax_rate : Float) : Float :=
  let total_before_discount := prices.sum
  let discount := discount_rate * total_before_discount
  let total_after_discount := total_before_discount - discount
  let sales_tax := tax_rate * total_after_discount
  total_after_discount + sales_tax

theorem toy_shopping_total (prices : List Float) 
  (h1 : prices = [8.25, 6.59, 12.10, 15.29, 23.47])
  (h2 : calculate_total_spent prices 0.10 0.06 = 62.68) : 
  calculate_total_spent prices 0.10 0.06 = 62.68 := by
  sorry

end NUMINAMATH_CALUDE_toy_shopping_total_l3475_347580


namespace NUMINAMATH_CALUDE_comprehensive_score_calculation_l3475_347502

theorem comprehensive_score_calculation (initial_score retest_score : ℝ) 
  (initial_weight retest_weight : ℝ) (h1 : initial_score = 400) 
  (h2 : retest_score = 85) (h3 : initial_weight = 0.4) (h4 : retest_weight = 0.6) :
  initial_score * initial_weight + retest_score * retest_weight = 211 := by
  sorry

end NUMINAMATH_CALUDE_comprehensive_score_calculation_l3475_347502


namespace NUMINAMATH_CALUDE_condition_neither_sufficient_nor_necessary_l3475_347570

def a (x : ℝ) : Fin 2 → ℝ := ![1, 2 - x]
def b (x : ℝ) : Fin 2 → ℝ := ![2 + x, 3]

def vectors_collinear (u v : Fin 2 → ℝ) : Prop :=
  ∃ (k : ℝ), ∀ (i : Fin 2), u i = k * v i

def norm_squared (v : Fin 2 → ℝ) : ℝ :=
  (v 0) ^ 2 + (v 1) ^ 2

theorem condition_neither_sufficient_nor_necessary :
  ¬(∀ x : ℝ, norm_squared (a x) = 2 → vectors_collinear (a x) (b x)) ∧
  ¬(∀ x : ℝ, vectors_collinear (a x) (b x) → norm_squared (a x) = 2) := by
  sorry

end NUMINAMATH_CALUDE_condition_neither_sufficient_nor_necessary_l3475_347570
