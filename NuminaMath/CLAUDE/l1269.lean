import Mathlib

namespace NUMINAMATH_CALUDE_quadratic_radicals_combination_l1269_126915

theorem quadratic_radicals_combination (a : ℝ) : 
  (∃ (k : ℝ), k > 0 ∧ (1 + a) = k * (4 - 2*a)) ↔ a = 1 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_radicals_combination_l1269_126915


namespace NUMINAMATH_CALUDE_starting_lineup_combinations_l1269_126921

/-- The number of members in the basketball team -/
def team_size : ℕ := 12

/-- The number of positions in the starting lineup -/
def lineup_size : ℕ := 5

/-- The number of ways to choose a starting lineup -/
def lineup_choices : ℕ := team_size * (team_size - 1) * (team_size - 2) * (team_size - 3) * (team_size - 4)

theorem starting_lineup_combinations : 
  lineup_choices = 95040 :=
sorry

end NUMINAMATH_CALUDE_starting_lineup_combinations_l1269_126921


namespace NUMINAMATH_CALUDE_weight_problem_l1269_126931

/-- Given the average weights of three people and two pairs, prove the weight of one person. -/
theorem weight_problem (A B C : ℝ) : 
  (A + B + C) / 3 = 45 ∧ 
  (A + B) / 2 = 40 ∧ 
  (B + C) / 2 = 43 → 
  B = 31 := by
  sorry

end NUMINAMATH_CALUDE_weight_problem_l1269_126931


namespace NUMINAMATH_CALUDE_prob_three_odds_eq_4_35_l1269_126963

/-- The set of numbers from which we select -/
def S : Finset ℕ := {1, 2, 3, 4, 5, 6, 7}

/-- The set of odd numbers in S -/
def odds : Finset ℕ := S.filter (fun n => n % 2 = 1)

/-- The number of elements to select -/
def k : ℕ := 3

/-- The probability of selecting three distinct odd numbers from S -/
theorem prob_three_odds_eq_4_35 : 
  (Finset.card (odds.powersetCard k)) / (Finset.card (S.powersetCard k)) = 4 / 35 := by
  sorry

end NUMINAMATH_CALUDE_prob_three_odds_eq_4_35_l1269_126963


namespace NUMINAMATH_CALUDE_unique_factorial_sum_l1269_126901

def factorial (n : ℕ) : ℕ := (List.range n).foldl (· * ·) 1

theorem unique_factorial_sum (x a b c d : ℕ) 
  (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) (h4 : 0 < d) (h5 : 0 < x)
  (h6 : a ≤ b) (h7 : b ≤ c) (h8 : c ≤ d) (h9 : d < x)
  (h10 : factorial x = factorial a + factorial b + factorial c + factorial d) :
  x = 4 ∧ a = 3 ∧ b = 3 ∧ c = 3 ∧ d = 3 :=
by sorry

end NUMINAMATH_CALUDE_unique_factorial_sum_l1269_126901


namespace NUMINAMATH_CALUDE_triangle_properties_l1269_126967

theorem triangle_properties (A B C : ℝ) (a b c : ℝ) :
  a = 2 →
  c = Real.sqrt 2 →
  Real.cos A = -(Real.sqrt 2) / 4 →
  a = b →
  b = c →
  c = a →
  Real.sin C = Real.sqrt 7 / 4 ∧
  b = 1 ∧
  Real.cos (2 * A + π / 3) = (-3 + Real.sqrt 21) / 8 := by
  sorry

end NUMINAMATH_CALUDE_triangle_properties_l1269_126967


namespace NUMINAMATH_CALUDE_pattern_holds_squares_in_figure_150_l1269_126900

/-- The number of unit squares in figure n -/
def f (n : ℕ) : ℕ := 3 * n^2 + 3 * n + 1

/-- The sequence of unit squares follows the given pattern for the first four figures -/
theorem pattern_holds : f 0 = 1 ∧ f 1 = 7 ∧ f 2 = 19 ∧ f 3 = 37 := by sorry

/-- The number of unit squares in figure 150 is 67951 -/
theorem squares_in_figure_150 : f 150 = 67951 := by sorry

end NUMINAMATH_CALUDE_pattern_holds_squares_in_figure_150_l1269_126900


namespace NUMINAMATH_CALUDE_polynomial_factorization_constant_term_l1269_126958

theorem polynomial_factorization_constant_term (a b c d e f : ℝ) 
  (p : ℝ → ℝ) (x₁ x₂ x₃ x₄ x₅ x₆ x₇ x₈ : ℝ) :
  (∀ x, p x = x^8 - 4*x^7 + 7*x^6 + a*x^5 + b*x^4 + c*x^3 + d*x^2 + e*x + f) →
  (∀ x, p x = (x - x₁) * (x - x₂) * (x - x₃) * (x - x₄) * (x - x₅) * (x - x₆) * (x - x₇) * (x - x₈)) →
  (x₁ > 0 ∧ x₂ > 0 ∧ x₃ > 0 ∧ x₄ > 0 ∧ x₅ > 0 ∧ x₆ > 0 ∧ x₇ > 0 ∧ x₈ > 0) →
  f = 1 / 256 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_factorization_constant_term_l1269_126958


namespace NUMINAMATH_CALUDE_correct_installation_time_l1269_126974

/-- Calculates the time needed to install remaining windows in a skyscraper -/
def time_to_install_remaining (total_windows : ℕ) (installed_windows : ℕ) (time_per_window : ℕ) : ℕ :=
  (total_windows - installed_windows) * time_per_window

/-- Theorem stating that the time to install remaining windows is correct -/
theorem correct_installation_time (total_windows installed_windows time_per_window : ℕ)
  (h1 : installed_windows ≤ total_windows) :
  time_to_install_remaining total_windows installed_windows time_per_window =
  (total_windows - installed_windows) * time_per_window :=
by sorry

end NUMINAMATH_CALUDE_correct_installation_time_l1269_126974


namespace NUMINAMATH_CALUDE_sqrt_eighteen_div_sqrt_two_eq_three_l1269_126968

theorem sqrt_eighteen_div_sqrt_two_eq_three : 
  Real.sqrt 18 / Real.sqrt 2 = 3 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_eighteen_div_sqrt_two_eq_three_l1269_126968


namespace NUMINAMATH_CALUDE_ratio_problem_l1269_126944

theorem ratio_problem (a b c d : ℝ) 
  (h1 : a / b = 3)
  (h2 : b / c = 2)
  (h3 : c / d = 5) :
  d / a = 1 / 30 := by
  sorry

end NUMINAMATH_CALUDE_ratio_problem_l1269_126944


namespace NUMINAMATH_CALUDE_arrangement_count_5_2_l1269_126910

/-- The number of ways to arrange n distinct objects and m pairs of 2 distinct objects each in a row,
    where the objects within each pair must be adjacent -/
def arrangementCount (n m : ℕ) : ℕ :=
  Nat.factorial (n + m) * (Nat.factorial 2)^m

/-- Theorem: The number of ways to arrange 5 distinct objects and 2 pairs of 2 distinct objects each
    in a row, where the objects within each pair must be adjacent, is equal to 7! * (2!)^2 -/
theorem arrangement_count_5_2 :
  arrangementCount 5 2 = 20160 := by
  sorry

end NUMINAMATH_CALUDE_arrangement_count_5_2_l1269_126910


namespace NUMINAMATH_CALUDE_problem_solution_l1269_126959

-- Define the solution set
def solution_set (x : ℝ) : Prop := -2 ≤ x ∧ x ≤ 1

-- Define the inequality
def inequality (x m : ℝ) : Prop := |x + 2| + |x - m| ≤ 3

theorem problem_solution :
  (∀ x, solution_set x ↔ inequality x 1) ∧
  ∀ a b c : ℝ, a^2 + 2*b^2 + 3*c^2 = 1 → -Real.sqrt 6 ≤ a + 2*b + 3*c ∧ a + 2*b + 3*c ≤ Real.sqrt 6 :=
sorry

end NUMINAMATH_CALUDE_problem_solution_l1269_126959


namespace NUMINAMATH_CALUDE_circle_symmetry_l1269_126940

def original_circle (x y : ℝ) : Prop :=
  x^2 - 2*x + y^2 = 3

def symmetric_circle (x y : ℝ) : Prop :=
  x^2 + 2*x + y^2 = 3

def symmetric_wrt_y_axis (x₁ y₁ x₂ y₂ : ℝ) : Prop :=
  x₁ = -x₂ ∧ y₁ = y₂

theorem circle_symmetry :
  ∀ (x₁ y₁ x₂ y₂ : ℝ),
    original_circle x₁ y₁ →
    symmetric_wrt_y_axis x₁ y₁ x₂ y₂ →
    symmetric_circle x₂ y₂ :=
by sorry

end NUMINAMATH_CALUDE_circle_symmetry_l1269_126940


namespace NUMINAMATH_CALUDE_equation_solution_l1269_126961

theorem equation_solution : ∃ x : ℝ, 5*x + 9*x = 420 - 12*(x - 4) ∧ x = 18 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l1269_126961


namespace NUMINAMATH_CALUDE_initial_number_proof_l1269_126919

theorem initial_number_proof (N : ℤ) : (N + 3) % 24 = 0 → N = 21 := by
  sorry

end NUMINAMATH_CALUDE_initial_number_proof_l1269_126919


namespace NUMINAMATH_CALUDE_range_of_m_l1269_126954

-- Define the propositions p and q
def p (x : ℝ) : Prop := |2*x - 1| ≤ 5
def q (x m : ℝ) : Prop := x^2 - 4*x + 4 - 9*m^2 ≤ 0

-- Define the set of x that satisfies ¬p
def not_p_set : Set ℝ := {x | ¬(p x)}

-- Define the set of x that satisfies ¬q
def not_q_set (m : ℝ) : Set ℝ := {x | ¬(q x m)}

-- State the theorem
theorem range_of_m :
  ∀ m : ℝ, m > 0 →
  (∀ x : ℝ, x ∈ not_p_set → x ∈ not_q_set m) →
  (∃ x : ℝ, x ∈ not_q_set m ∧ x ∉ not_p_set) →
  m ∈ Set.Ioo 0 (1/3) ∪ {1/3} :=
sorry

end NUMINAMATH_CALUDE_range_of_m_l1269_126954


namespace NUMINAMATH_CALUDE_heaviest_weight_in_geometric_progression_l1269_126982

/-- Given four weights in geometric progression, the heaviest can be found using a balance twice -/
theorem heaviest_weight_in_geometric_progression 
  (b : ℝ) (d : ℝ) (h_b_pos : b > 0) (h_d_gt_one : d > 1) :
  ∃ (n : ℕ), n ≤ 2 ∧ 
    (∀ i : Fin 4, b * d ^ i.val ≤ b * d ^ 3) ∧
    (∀ i : Fin 4, i.val ≠ 3 → b * d ^ i.val < b * d ^ 3) := by
  sorry

end NUMINAMATH_CALUDE_heaviest_weight_in_geometric_progression_l1269_126982


namespace NUMINAMATH_CALUDE_train_average_speed_l1269_126980

/-- Given a train that travels two segments with known distances and times, 
    calculate its average speed. -/
theorem train_average_speed 
  (distance1 : ℝ) (time1 : ℝ) (distance2 : ℝ) (time2 : ℝ) 
  (h1 : distance1 = 325) (h2 : time1 = 3.5) 
  (h3 : distance2 = 470) (h4 : time2 = 4) : 
  (distance1 + distance2) / (time1 + time2) = 106 := by
  sorry

#eval (325 + 470) / (3.5 + 4)

end NUMINAMATH_CALUDE_train_average_speed_l1269_126980


namespace NUMINAMATH_CALUDE_log_75843_bounds_l1269_126936

theorem log_75843_bounds : ∃ (c d : ℤ), (c : ℝ) < Real.log 75843 / Real.log 10 ∧ 
  Real.log 75843 / Real.log 10 < (d : ℝ) ∧ c = 4 ∧ d = 5 ∧ c + d = 9 := by
  sorry

#check log_75843_bounds

end NUMINAMATH_CALUDE_log_75843_bounds_l1269_126936


namespace NUMINAMATH_CALUDE_inequality_holds_only_for_m_equals_negative_four_l1269_126913

theorem inequality_holds_only_for_m_equals_negative_four :
  ∀ m : ℝ, (∀ x : ℝ, |2*x - m| ≤ |3*x + 6|) ↔ m = -4 := by
  sorry

end NUMINAMATH_CALUDE_inequality_holds_only_for_m_equals_negative_four_l1269_126913


namespace NUMINAMATH_CALUDE_committee_probability_l1269_126979

/-- The number of members in the Grammar club -/
def total_members : ℕ := 20

/-- The number of boys in the Grammar club -/
def num_boys : ℕ := 10

/-- The number of girls in the Grammar club -/
def num_girls : ℕ := 10

/-- The size of the committee to be chosen -/
def committee_size : ℕ := 4

/-- The probability of selecting a committee with at least one boy and one girl -/
theorem committee_probability : 
  (Nat.choose total_members committee_size - 
   (Nat.choose num_boys committee_size + Nat.choose num_girls committee_size)) / 
   Nat.choose total_members committee_size = 295 / 323 := by
  sorry

end NUMINAMATH_CALUDE_committee_probability_l1269_126979


namespace NUMINAMATH_CALUDE_max_stamps_purchasable_l1269_126947

def stamp_price : ℕ := 35
def discount_threshold : ℕ := 100
def discount_rate : ℚ := 5 / 100
def budget : ℕ := 3200

theorem max_stamps_purchasable :
  let max_stamps := (budget / stamp_price : ℕ)
  let discounted_price := stamp_price * (1 - discount_rate)
  let max_stamps_with_discount := (budget / discounted_price).floor
  (max_stamps_with_discount ≤ discount_threshold) ∧
  (max_stamps = 91) := by
sorry

end NUMINAMATH_CALUDE_max_stamps_purchasable_l1269_126947


namespace NUMINAMATH_CALUDE_max_m_value_l1269_126932

def M (m : ℝ) : Set (ℝ × ℝ) := {p | p.1 ≤ -1 ∧ p.2 ≤ m}

theorem max_m_value :
  ∃ m : ℝ, m = 1 ∧
  (∀ m' : ℝ, (∀ p ∈ M m', p.1 * 2^p.2 - p.2 - 3*p.1 ≥ 0) →
  m' ≤ m) ∧
  (∀ p ∈ M m, p.1 * 2^p.2 - p.2 - 3*p.1 ≥ 0) :=
sorry

end NUMINAMATH_CALUDE_max_m_value_l1269_126932


namespace NUMINAMATH_CALUDE_arithmetic_sequence_problem_l1269_126972

/-- An arithmetic sequence with its sum sequence -/
structure ArithmeticSequence where
  a : ℕ → ℝ  -- The sequence
  S : ℕ → ℝ  -- The sum sequence
  is_arithmetic : ∀ n, a (n + 1) - a n = a 2 - a 1
  sum_formula : ∀ n, S n = n * a 1 + (n * (n - 1) / 2) * (a 2 - a 1)

/-- Theorem: For an arithmetic sequence with a_2 = 1 and S_4 = 8, a_5 = 7 and S_10 = 80 -/
theorem arithmetic_sequence_problem (seq : ArithmeticSequence) 
    (h1 : seq.a 2 = 1) (h2 : seq.S 4 = 8) : 
    seq.a 5 = 7 ∧ seq.S 10 = 80 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_problem_l1269_126972


namespace NUMINAMATH_CALUDE_largest_initial_number_l1269_126911

theorem largest_initial_number : 
  ∃ (a b c d e : ℕ), 
    189 + a + b + c + d + e = 200 ∧ 
    a ≥ 2 ∧ b ≥ 2 ∧ c ≥ 2 ∧ d ≥ 2 ∧ e ≥ 2 ∧
    189 % a ≠ 0 ∧ 189 % b ≠ 0 ∧ 189 % c ≠ 0 ∧ 189 % d ≠ 0 ∧ 189 % e ≠ 0 ∧
    ∀ (n : ℕ), n > 189 → 
      ¬(∃ (a' b' c' d' e' : ℕ), 
        n + a' + b' + c' + d' + e' = 200 ∧
        a' ≥ 2 ∧ b' ≥ 2 ∧ c' ≥ 2 ∧ d' ≥ 2 ∧ e' ≥ 2 ∧
        n % a' ≠ 0 ∧ n % b' ≠ 0 ∧ n % c' ≠ 0 ∧ n % d' ≠ 0 ∧ n % e' ≠ 0) :=
by sorry

end NUMINAMATH_CALUDE_largest_initial_number_l1269_126911


namespace NUMINAMATH_CALUDE_sufficient_but_not_necessary_l1269_126933

theorem sufficient_but_not_necessary : 
  (∀ x : ℝ, x > 3 → x^2 - 5*x + 6 > 0) ∧ 
  (∃ x : ℝ, x^2 - 5*x + 6 > 0 ∧ ¬(x > 3)) := by
  sorry

end NUMINAMATH_CALUDE_sufficient_but_not_necessary_l1269_126933


namespace NUMINAMATH_CALUDE_certain_number_proof_l1269_126991

theorem certain_number_proof : ∃ x : ℝ, x * 7 = (35 / 100) * 900 ∧ x = 45 := by sorry

end NUMINAMATH_CALUDE_certain_number_proof_l1269_126991


namespace NUMINAMATH_CALUDE_probability_two_yellow_marbles_l1269_126978

/-- The probability of drawing two yellow marbles successively from a jar -/
theorem probability_two_yellow_marbles 
  (blue : ℕ) (yellow : ℕ) (black : ℕ) 
  (h_blue : blue = 3)
  (h_yellow : yellow = 4)
  (h_black : black = 8) :
  let total := blue + yellow + black
  (yellow / total) * ((yellow - 1) / (total - 1)) = 2 / 35 := by
sorry

end NUMINAMATH_CALUDE_probability_two_yellow_marbles_l1269_126978


namespace NUMINAMATH_CALUDE_binomial_expansion_sum_l1269_126997

theorem binomial_expansion_sum (a₀ a₁ a₂ a₃ a₄ a₅ : ℤ) :
  (∀ x : ℝ, (2*x - 1)^5 = a₀*x^5 + a₁*x^4 + a₂*x^3 + a₃*x^2 + a₄*x + a₅) →
  a₂ + a₃ = 40 := by
sorry

end NUMINAMATH_CALUDE_binomial_expansion_sum_l1269_126997


namespace NUMINAMATH_CALUDE_quadratic_inequality_range_l1269_126928

theorem quadratic_inequality_range (a : ℝ) : 
  (∃ x : ℝ, x^2 + a*x + 4 < 0) ↔ (a < -4 ∨ a > 4) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_range_l1269_126928


namespace NUMINAMATH_CALUDE_smallest_n0_for_inequality_l1269_126905

theorem smallest_n0_for_inequality : ∃ (n0 : ℕ), n0 = 5 ∧ 
  (∀ n : ℕ, n ≥ n0 → 2^n > n^2 + 1) ∧ 
  (∀ m : ℕ, m < n0 → ¬(2^m > m^2 + 1)) :=
sorry

end NUMINAMATH_CALUDE_smallest_n0_for_inequality_l1269_126905


namespace NUMINAMATH_CALUDE_max_plain_cupcakes_l1269_126937

structure Cupcakes :=
  (total : ℕ)
  (blueberries : ℕ)
  (sprinkles : ℕ)
  (frosting : ℕ)
  (pecans : ℕ)

def has_no_ingredients (c : Cupcakes) : ℕ :=
  c.total - (c.blueberries + c.sprinkles + c.frosting + c.pecans)

theorem max_plain_cupcakes (c : Cupcakes) 
  (h_total : c.total = 60)
  (h_blueberries : c.blueberries ≥ c.total / 3)
  (h_sprinkles : c.sprinkles ≥ c.total / 4)
  (h_frosting : c.frosting ≥ c.total / 2)
  (h_pecans : c.pecans ≥ c.total / 5) :
  has_no_ingredients c ≤ 0 :=
sorry

end NUMINAMATH_CALUDE_max_plain_cupcakes_l1269_126937


namespace NUMINAMATH_CALUDE_toby_speed_proof_l1269_126926

/-- Toby's speed when pulling an unloaded sled -/
def unloaded_speed : ℝ := 20

/-- Distance of the first loaded part of the journey -/
def loaded_distance1 : ℝ := 180

/-- Distance of the first unloaded part of the journey -/
def unloaded_distance1 : ℝ := 120

/-- Distance of the second loaded part of the journey -/
def loaded_distance2 : ℝ := 80

/-- Distance of the second unloaded part of the journey -/
def unloaded_distance2 : ℝ := 140

/-- Total time of the journey -/
def total_time : ℝ := 39

/-- Toby's speed when pulling a loaded sled -/
def loaded_speed : ℝ := 10

theorem toby_speed_proof :
  (loaded_distance1 / loaded_speed + unloaded_distance1 / unloaded_speed +
   loaded_distance2 / loaded_speed + unloaded_distance2 / unloaded_speed) = total_time :=
by sorry

end NUMINAMATH_CALUDE_toby_speed_proof_l1269_126926


namespace NUMINAMATH_CALUDE_cricket_team_ratio_l1269_126984

theorem cricket_team_ratio (total_players : ℕ) (throwers : ℕ) (right_handed : ℕ) :
  total_players = 70 →
  throwers = 37 →
  right_handed = 59 →
  (total_players - throwers : ℚ) / (total_players - throwers) = 3 / 4 :=
by sorry

end NUMINAMATH_CALUDE_cricket_team_ratio_l1269_126984


namespace NUMINAMATH_CALUDE_ellipse_properties_l1269_126909

-- Define the ellipse C
def ellipse (a b : ℝ) (x y : ℝ) : Prop :=
  x^2 / a^2 + y^2 / b^2 = 1

-- Define the conditions
def short_axis_length (b : ℝ) : Prop :=
  2 * b = 2 * Real.sqrt 3

def slope_product (a : ℝ) (x y : ℝ) : Prop :=
  y^2 / (x^2 - a^2) = 3 / 4

-- Define the theorem
theorem ellipse_properties (a b : ℝ) (h1 : a > b) (h2 : b > 0) 
  (h3 : short_axis_length b) (h4 : ∀ x y, ellipse a b x y → slope_product a x y) :
  (∀ x y, ellipse a b x y ↔ x^2 / 4 + y^2 / 3 = 1) ∧
  (∀ m : ℝ, m ≠ 0 → 
    ∃ Q : ℝ × ℝ, 
      (∃ A B : ℝ × ℝ, 
        ellipse a b A.1 A.2 ∧ 
        ellipse a b B.1 B.2 ∧
        A.1 = m * A.2 + 1 ∧
        B.1 = m * B.2 + 1 ∧
        Q.1 = A.1 + (Q.2 - A.2) * (A.1 + a) / A.2 ∧
        Q.1 = B.1 + (Q.2 - B.2) * (B.1 - a) / B.2) →
      Q.1 = 4) :=
sorry

end NUMINAMATH_CALUDE_ellipse_properties_l1269_126909


namespace NUMINAMATH_CALUDE_probability_nine_heads_in_twelve_flips_l1269_126938

theorem probability_nine_heads_in_twelve_flips : 
  let n : ℕ := 12  -- number of coin flips
  let k : ℕ := 9   -- number of heads we want
  let p : ℚ := 1/2 -- probability of heads for a fair coin
  Nat.choose n k * p^k * (1-p)^(n-k) = 55/1024 :=
by sorry

end NUMINAMATH_CALUDE_probability_nine_heads_in_twelve_flips_l1269_126938


namespace NUMINAMATH_CALUDE_quadratic_range_on_interval_l1269_126914

def g (a b c x : ℝ) : ℝ := a * x^2 + b * x + c

theorem quadratic_range_on_interval
  (a b c : ℝ)
  (ha : a > 0) :
  let range_min := min (g a b c (-1)) (g a b c 2)
  let range_max := max (g a b c (-1)) (max (g a b c 2) (g a b c (-b/(2*a))))
  ∀ x ∈ Set.Icc (-1 : ℝ) 2,
    range_min ≤ g a b c x ∧ g a b c x ≤ range_max :=
by sorry

end NUMINAMATH_CALUDE_quadratic_range_on_interval_l1269_126914


namespace NUMINAMATH_CALUDE_exterior_angle_theorem_l1269_126964

/-- The measure of the exterior angle BAC formed by a square and a regular octagon sharing a common side --/
def exterior_angle_measure : ℝ := 135

/-- A square and a regular octagon are coplanar and share a common side AD --/
axiom share_common_side : True

/-- Theorem: The measure of the exterior angle BAC is 135 degrees --/
theorem exterior_angle_theorem : exterior_angle_measure = 135 := by
  sorry

end NUMINAMATH_CALUDE_exterior_angle_theorem_l1269_126964


namespace NUMINAMATH_CALUDE_total_members_in_math_club_l1269_126971

def math_club (female_members : ℕ) (male_members : ℕ) : Prop :=
  male_members = 2 * female_members

theorem total_members_in_math_club (female_members : ℕ) 
  (h1 : female_members = 6) 
  (h2 : math_club female_members (2 * female_members)) : 
  female_members + 2 * female_members = 18 := by
  sorry

end NUMINAMATH_CALUDE_total_members_in_math_club_l1269_126971


namespace NUMINAMATH_CALUDE_arithmetic_sequence_special_property_l1269_126989

/-- Given an arithmetic sequence {a_n} with common difference d (d ≠ 0) and sum of first n terms S_n,
    if {√(S_n + n)} is also an arithmetic sequence with common difference d, then d = 1/2. -/
theorem arithmetic_sequence_special_property (d : ℝ) (a : ℕ → ℝ) (S : ℕ → ℝ) :
  d ≠ 0 ∧
  (∀ n : ℕ, a (n + 1) - a n = d) ∧
  (∀ n : ℕ, S n = n * a 1 + (n * (n - 1)) / 2 * d) ∧
  (∀ n : ℕ, Real.sqrt (S (n + 1) + (n + 1)) - Real.sqrt (S n + n) = d) →
  d = 1 / 2 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_special_property_l1269_126989


namespace NUMINAMATH_CALUDE_prism_intersection_probability_l1269_126977

/-- A rectangular prism with dimensions 2, 3, and 5 units. -/
structure RectangularPrism where
  length : ℕ := 2
  width : ℕ := 3
  height : ℕ := 5

/-- The probability that three randomly chosen vertices of the prism
    form a plane intersecting the prism's interior. -/
def intersectionProbability (p : RectangularPrism) : ℚ :=
  11/14

/-- Theorem stating that the probability of three randomly chosen vertices
    forming a plane that intersects the interior of the given rectangular prism is 11/14. -/
theorem prism_intersection_probability (p : RectangularPrism) :
  intersectionProbability p = 11/14 := by
  sorry

end NUMINAMATH_CALUDE_prism_intersection_probability_l1269_126977


namespace NUMINAMATH_CALUDE_even_function_property_l1269_126946

def evenFunction (f : ℝ → ℝ) : Prop := ∀ x, f x = f (-x)

theorem even_function_property
  (f : ℝ → ℝ)
  (h_even : evenFunction f)
  (h_increasing : ∀ x y, x < y → x < 0 → f x < f y)
  (x₁ x₂ : ℝ)
  (h_x₁_neg : x₁ < 0)
  (h_x₂_pos : x₂ > 0)
  (h_abs : |x₁| < |x₂|) :
  f (-x₁) > f (-x₂) := by
sorry

end NUMINAMATH_CALUDE_even_function_property_l1269_126946


namespace NUMINAMATH_CALUDE_dance_attendance_l1269_126973

theorem dance_attendance (girls boys : ℕ) : 
  boys = 2 * girls ∧ 
  boys = (girls - 1) + 8 → 
  boys = 14 := by
sorry

end NUMINAMATH_CALUDE_dance_attendance_l1269_126973


namespace NUMINAMATH_CALUDE_square_roots_problem_l1269_126906

theorem square_roots_problem (a m : ℝ) (ha : 0 < a) 
  (h1 : (2 * m - 1)^2 = a) (h2 : (m + 4)^2 = a) : a = 9 := by
  sorry

end NUMINAMATH_CALUDE_square_roots_problem_l1269_126906


namespace NUMINAMATH_CALUDE_largest_negative_congruent_to_one_mod_seventeen_l1269_126948

theorem largest_negative_congruent_to_one_mod_seventeen :
  ∃ (n : ℤ), 
    n = -1002 ∧ 
    n ≡ 1 [ZMOD 17] ∧ 
    n < 0 ∧ 
    -9999 ≤ n ∧
    ∀ (m : ℤ), m ≡ 1 [ZMOD 17] ∧ m < 0 ∧ -9999 ≤ m → m ≤ n :=
by sorry

end NUMINAMATH_CALUDE_largest_negative_congruent_to_one_mod_seventeen_l1269_126948


namespace NUMINAMATH_CALUDE_cricket_team_selection_l1269_126916

/-- The total number of players in the cricket team -/
def total_players : ℕ := 16

/-- The number of quadruplets in the team -/
def num_quadruplets : ℕ := 4

/-- The number of players to be chosen for the training squad -/
def squad_size : ℕ := 5

/-- The number of ways to choose the training squad under the given restrictions -/
def valid_selections : ℕ := 4356

theorem cricket_team_selection :
  (Nat.choose total_players squad_size) - (Nat.choose (total_players - num_quadruplets) 1) = valid_selections :=
sorry

end NUMINAMATH_CALUDE_cricket_team_selection_l1269_126916


namespace NUMINAMATH_CALUDE_cube_third_times_eighth_equals_one_over_216_l1269_126999

theorem cube_third_times_eighth_equals_one_over_216 :
  (1 / 3 : ℚ)^3 * (1 / 8 : ℚ) = 1 / 216 := by
  sorry

end NUMINAMATH_CALUDE_cube_third_times_eighth_equals_one_over_216_l1269_126999


namespace NUMINAMATH_CALUDE_division_of_decimals_l1269_126912

theorem division_of_decimals : (0.45 : ℝ) / (0.005 : ℝ) = 90 := by
  sorry

end NUMINAMATH_CALUDE_division_of_decimals_l1269_126912


namespace NUMINAMATH_CALUDE_roberto_outfits_l1269_126992

/-- The number of different outfits Roberto can create -/
def number_of_outfits (trousers shirts jackets : ℕ) : ℕ := trousers * shirts * jackets

/-- Theorem: Roberto can create 84 different outfits -/
theorem roberto_outfits :
  let trousers : ℕ := 4
  let shirts : ℕ := 7
  let jackets : ℕ := 3
  number_of_outfits trousers shirts jackets = 84 := by
  sorry

#eval number_of_outfits 4 7 3

end NUMINAMATH_CALUDE_roberto_outfits_l1269_126992


namespace NUMINAMATH_CALUDE_water_tank_capacity_l1269_126983

theorem water_tank_capacity : ∀ x : ℚ,
  (5/6 : ℚ) * x - 30 = (4/5 : ℚ) * x → x = 900 := by
  sorry

end NUMINAMATH_CALUDE_water_tank_capacity_l1269_126983


namespace NUMINAMATH_CALUDE_minimum_cars_with_all_characteristics_l1269_126924

theorem minimum_cars_with_all_characteristics 
  (total : ℕ) 
  (zhiguli dark_colored male_drivers with_passengers : ℕ) 
  (h_total : total = 20)
  (h_zhiguli : zhiguli = 14)
  (h_dark : dark_colored = 15)
  (h_male : male_drivers = 17)
  (h_passengers : with_passengers = 18) :
  total - ((total - zhiguli) + (total - dark_colored) + (total - male_drivers) + (total - with_passengers)) = 4 := by
sorry

end NUMINAMATH_CALUDE_minimum_cars_with_all_characteristics_l1269_126924


namespace NUMINAMATH_CALUDE_tan_two_implies_sum_l1269_126960

theorem tan_two_implies_sum (θ : ℝ) (h : Real.tan θ = 2) : 
  2 * Real.sin θ + Real.sin θ * Real.cos θ = 2 := by
  sorry

end NUMINAMATH_CALUDE_tan_two_implies_sum_l1269_126960


namespace NUMINAMATH_CALUDE_households_using_all_brands_l1269_126966

/-- Represents the survey results of household soap usage -/
structure SoapSurvey where
  total : ℕ
  none : ℕ
  only_x : ℕ
  only_y : ℕ
  only_z : ℕ
  ratio_all_to_two : ℕ
  ratio_all_to_one : ℕ

/-- Calculates the number of households using all three brands of soap -/
def households_using_all (survey : SoapSurvey) : ℕ :=
  (survey.only_x + survey.only_y + survey.only_z) / survey.ratio_all_to_one

/-- Theorem stating the number of households using all three brands of soap -/
theorem households_using_all_brands (survey : SoapSurvey) 
  (h1 : survey.total = 5000)
  (h2 : survey.none = 1200)
  (h3 : survey.only_x = 800)
  (h4 : survey.only_y = 600)
  (h5 : survey.only_z = 300)
  (h6 : survey.ratio_all_to_two = 5)
  (h7 : survey.ratio_all_to_one = 10) :
  households_using_all survey = 170 := by
  sorry


end NUMINAMATH_CALUDE_households_using_all_brands_l1269_126966


namespace NUMINAMATH_CALUDE_sallys_garden_area_l1269_126929

/-- Represents a rectangular garden with fence posts. -/
structure GardenFence where
  total_posts : ℕ
  post_spacing : ℕ
  long_side_post_ratio : ℕ

/-- Calculates the area of the garden given its fence configuration. -/
def garden_area (fence : GardenFence) : ℕ :=
  let short_side_posts := (fence.total_posts / 2) / (fence.long_side_post_ratio + 1)
  let long_side_posts := short_side_posts * fence.long_side_post_ratio
  let short_side_length := (short_side_posts - 1) * fence.post_spacing
  let long_side_length := (long_side_posts - 1) * fence.post_spacing
  short_side_length * long_side_length

/-- Theorem stating that Sally's garden has an area of 297 square yards. -/
theorem sallys_garden_area :
  let sally_fence := GardenFence.mk 24 3 3
  garden_area sally_fence = 297 := by
  sorry

end NUMINAMATH_CALUDE_sallys_garden_area_l1269_126929


namespace NUMINAMATH_CALUDE_sqrt_two_f_pi_fourth_plus_f_neg_pi_sixth_pos_l1269_126935

open Real

-- Define f as a function from ℝ to ℝ
variable (f : ℝ → ℝ)

-- Define f' as the derivative of f
variable (f' : ℝ → ℝ)

-- Axiom: f is an odd function
axiom f_odd (x : ℝ) : f (-x) = -f x

-- Axiom: f' is the derivative of f
axiom f'_is_derivative : ∀ x, HasDerivAt f (f' x) x

-- Axiom: For x in (0, π/2) ∪ (π/2, π), f(x) + f'(x)tan(x) > 0
axiom f_plus_f'_tan_pos (x : ℝ) (h1 : 0 < x) (h2 : x < π) (h3 : x ≠ π/2) :
  f x + f' x * tan x > 0

-- Theorem to prove
theorem sqrt_two_f_pi_fourth_plus_f_neg_pi_sixth_pos :
  Real.sqrt 2 * f (π/4) + f (-π/6) > 0 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_two_f_pi_fourth_plus_f_neg_pi_sixth_pos_l1269_126935


namespace NUMINAMATH_CALUDE_transformer_current_load_transformer_current_load_is_700A_l1269_126917

theorem transformer_current_load : ℕ → Prop :=
  fun total_load =>
    let units_40A := 3
    let units_60A := 2
    let units_25A := 1
    let running_current_40A := 40
    let running_current_60A := 60
    let running_current_25A := 25
    let starting_multiplier_40A := 2
    let starting_multiplier_60A := 3
    let starting_multiplier_25A := 4
    let total_start_current_40A := units_40A * running_current_40A * starting_multiplier_40A
    let total_start_current_60A := units_60A * running_current_60A * starting_multiplier_60A
    let total_start_current_25A := units_25A * running_current_25A * starting_multiplier_25A
    total_load = total_start_current_40A + total_start_current_60A + total_start_current_25A

theorem transformer_current_load_is_700A : transformer_current_load 700 := by
  sorry

end NUMINAMATH_CALUDE_transformer_current_load_transformer_current_load_is_700A_l1269_126917


namespace NUMINAMATH_CALUDE_pizza_total_slices_l1269_126987

def pizza_problem (john_slices sam_slices remaining_slices : ℕ) : Prop :=
  john_slices = 3 ∧
  sam_slices = 2 * john_slices ∧
  remaining_slices = 3

theorem pizza_total_slices 
  (john_slices sam_slices remaining_slices : ℕ) 
  (h : pizza_problem john_slices sam_slices remaining_slices) : 
  john_slices + sam_slices + remaining_slices = 12 :=
by
  sorry

#check pizza_total_slices

end NUMINAMATH_CALUDE_pizza_total_slices_l1269_126987


namespace NUMINAMATH_CALUDE_tangent_line_difference_l1269_126943

/-- A curve defined by y = x^3 + ax + b -/
def curve (a b : ℝ) (x : ℝ) : ℝ := x^3 + a*x + b

/-- The derivative of the curve -/
def curve_derivative (a : ℝ) (x : ℝ) : ℝ := 3*x^2 + a

/-- A line defined by y = kx + 1 -/
def line (k : ℝ) (x : ℝ) : ℝ := k*x + 1

theorem tangent_line_difference (a b k : ℝ) :
  (curve a b 1 = 2) →  -- The curve passes through (1, 2)
  (line k 1 = 2) →     -- The line passes through (1, 2)
  (curve_derivative a 1 = k) →  -- The derivative of the curve at x=1 equals the slope of the line
  b - a = 5 := by
    sorry


end NUMINAMATH_CALUDE_tangent_line_difference_l1269_126943


namespace NUMINAMATH_CALUDE_fourth_term_largest_l1269_126918

theorem fourth_term_largest (x : ℝ) : 
  (5/8 < x ∧ x < 20/21) ↔ 
  (∀ k : ℕ, k ≠ 4 → 
    Nat.choose 10 3 * (5^7) * (3*x)^3 ≥ Nat.choose 10 (k-1) * (5^(10-(k-1))) * (3*x)^(k-1)) :=
by sorry

end NUMINAMATH_CALUDE_fourth_term_largest_l1269_126918


namespace NUMINAMATH_CALUDE_two_digit_sum_reverse_l1269_126957

theorem two_digit_sum_reverse : 
  (∃! n : Nat, n = (Finset.filter 
    (fun p : Nat × Nat => 
      let (a, b) := p
      0 < a ∧ a ≤ 9 ∧ 0 ≤ b ∧ b ≤ 9 ∧ (10 * a + b) + (10 * b + a) = 143)
    (Finset.product (Finset.range 10) (Finset.range 10))).card
  ∧ n = 6) := by sorry

end NUMINAMATH_CALUDE_two_digit_sum_reverse_l1269_126957


namespace NUMINAMATH_CALUDE_distance_to_focus_is_six_l1269_126939

/-- A parabola with equation y^2 = 4x -/
structure Parabola where
  equation : ∀ x y, y^2 = 4*x

/-- The focus of the parabola y^2 = 4x -/
def focus : ℝ × ℝ := (1, 0)

/-- A point on the parabola -/
structure PointOnParabola (p : Parabola) where
  x : ℝ
  y : ℝ
  on_parabola : y^2 = 4*x

theorem distance_to_focus_is_six (p : Parabola) (P : PointOnParabola p) 
  (h : |P.x - (-3)| = 5) : 
  Real.sqrt ((P.x - focus.1)^2 + (P.y - focus.2)^2) = 6 := by
  sorry

end NUMINAMATH_CALUDE_distance_to_focus_is_six_l1269_126939


namespace NUMINAMATH_CALUDE_oil_purchase_amount_l1269_126927

/-- Proves that the amount spent on oil is Rs. 600 given the conditions of the problem -/
theorem oil_purchase_amount (original_price : ℝ) (reduced_price : ℝ) (additional_oil : ℝ) 
  (h1 : reduced_price = original_price * 0.75)
  (h2 : reduced_price = 30)
  (h3 : additional_oil = 5) :
  ∃ (amount_spent : ℝ), 
    amount_spent / reduced_price - amount_spent / original_price = additional_oil ∧ 
    amount_spent = 600 := by
  sorry

end NUMINAMATH_CALUDE_oil_purchase_amount_l1269_126927


namespace NUMINAMATH_CALUDE_determinant_equality_l1269_126942

theorem determinant_equality (x y z w : ℝ) :
  Matrix.det ![![x, y], ![z, w]] = 7 →
  Matrix.det ![![x + 2*z, y + 2*w], ![z, w]] = 7 := by
  sorry

end NUMINAMATH_CALUDE_determinant_equality_l1269_126942


namespace NUMINAMATH_CALUDE_solution_when_a_is_3_solution_when_a_is_neg_l1269_126945

-- Define the quadratic inequality
def quadratic_inequality (a : ℝ) (x : ℝ) : Prop :=
  a * x^2 + (1 - 2*a) * x - 2 < 0

-- Define the solution set for a = 3
def solution_set_a3 : Set ℝ :=
  {x | -1/3 < x ∧ x < 2}

-- Define the solution set for a < 0
def solution_set_a_neg (a : ℝ) : Set ℝ :=
  if -1/2 < a ∧ a < 0 then
    {x | x < 2 ∨ x > -1/a}
  else if a = -1/2 then
    {x | x ≠ 2}
  else
    {x | x > 2 ∨ x < -1/a}

-- Theorem for a = 3
theorem solution_when_a_is_3 :
  ∀ x, x ∈ solution_set_a3 ↔ quadratic_inequality 3 x :=
sorry

-- Theorem for a < 0
theorem solution_when_a_is_neg :
  ∀ a, a < 0 → ∀ x, x ∈ solution_set_a_neg a ↔ quadratic_inequality a x :=
sorry

end NUMINAMATH_CALUDE_solution_when_a_is_3_solution_when_a_is_neg_l1269_126945


namespace NUMINAMATH_CALUDE_speeding_percentage_l1269_126990

/-- The percentage of motorists who exceed the speed limit and receive tickets -/
def ticketed_speeders : ℝ := 20

/-- The percentage of speeding motorists who do not receive tickets -/
def unticketed_speeder_percentage : ℝ := 20

/-- The total percentage of motorists who exceed the speed limit -/
def total_speeders : ℝ := 25

theorem speeding_percentage :
  ticketed_speeders * (100 - unticketed_speeder_percentage) / 100 = total_speeders * (100 - unticketed_speeder_percentage) / 100 := by
  sorry

end NUMINAMATH_CALUDE_speeding_percentage_l1269_126990


namespace NUMINAMATH_CALUDE_prob_two_heads_in_three_fair_coin_l1269_126951

/-- A fair coin is a coin with probability 1/2 of landing heads -/
def fairCoin (p : ℝ) : Prop := p = (1 : ℝ) / 2

/-- The probability of getting exactly two heads in three independent coin flips -/
def probTwoHeadsInThree (p : ℝ) : ℝ := 3 * p^2 * (1 - p)

/-- Theorem: The probability of getting exactly two heads in three flips of a fair coin is 3/8 -/
theorem prob_two_heads_in_three_fair_coin :
  ∀ p : ℝ, fairCoin p → probTwoHeadsInThree p = (3 : ℝ) / 8 :=
by sorry

end NUMINAMATH_CALUDE_prob_two_heads_in_three_fair_coin_l1269_126951


namespace NUMINAMATH_CALUDE_max_a4b4_l1269_126904

/-- Given an arithmetic sequence a and a geometric sequence b satisfying
    certain conditions, the maximum value of a₄b₄ is 37/4 -/
theorem max_a4b4 (a b : ℕ → ℝ) 
  (h_arith : ∀ n, a (n + 1) - a n = a (n + 2) - a (n + 1))
  (h_geom : ∀ n, b (n + 1) / b n = b (n + 2) / b (n + 1))
  (h1 : a 1 * b 1 = 20)
  (h2 : a 2 * b 2 = 19)
  (h3 : a 3 * b 3 = 14) :
  (∀ x, a 4 * b 4 ≤ x) → x = 37/4 :=
sorry

end NUMINAMATH_CALUDE_max_a4b4_l1269_126904


namespace NUMINAMATH_CALUDE_reciprocal_of_point_B_is_one_l1269_126988

-- Define the position of point A on the number line
def point_A : ℝ := -3

-- Define the distance between point A and point B
def distance_AB : ℝ := 4

-- Define the position of point B on the number line
def point_B : ℝ := point_A + distance_AB

-- Theorem to prove
theorem reciprocal_of_point_B_is_one : 
  (1 : ℝ) / point_B = 1 := by sorry

end NUMINAMATH_CALUDE_reciprocal_of_point_B_is_one_l1269_126988


namespace NUMINAMATH_CALUDE_jim_tree_planting_l1269_126955

/-- The age at which Jim started planting a new row of trees every year -/
def start_age : ℕ := sorry

/-- The number of trees Jim has initially -/
def initial_trees : ℕ := 2 * 4

/-- The number of trees Jim plants each year -/
def trees_per_year : ℕ := 4

/-- Jim's age when he doubles his trees -/
def final_age : ℕ := 15

/-- The total number of trees Jim has after doubling on his 15th birthday -/
def total_trees : ℕ := 56

theorem jim_tree_planting :
  2 * (initial_trees + trees_per_year * (final_age - start_age)) = total_trees := by sorry

end NUMINAMATH_CALUDE_jim_tree_planting_l1269_126955


namespace NUMINAMATH_CALUDE_two_digit_prime_difference_l1269_126956

theorem two_digit_prime_difference (x y : ℕ) : 
  x < 10 → y < 10 → (10 * x + y) - (10 * y + x) = 90 → x - y = 10 := by
  sorry

end NUMINAMATH_CALUDE_two_digit_prime_difference_l1269_126956


namespace NUMINAMATH_CALUDE_fraction_simplification_l1269_126920

theorem fraction_simplification :
  ((2^2010)^2 - (2^2008)^2) / ((2^2009)^2 - (2^2007)^2) = 4 := by
  sorry

end NUMINAMATH_CALUDE_fraction_simplification_l1269_126920


namespace NUMINAMATH_CALUDE_dice_probability_l1269_126950

-- Define a die
def Die := Fin 6

-- Define the sum of three dice rolls
def diceSum (d1 d2 d3 : Die) : ℕ := d1.val + d2.val + d3.val + 3

-- Define the condition for the sum to be even and greater than 15
def validRoll (d1 d2 d3 : Die) : Prop :=
  Even (diceSum d1 d2 d3) ∧ diceSum d1 d2 d3 > 15

-- Define the total number of possible outcomes
def totalOutcomes : ℕ := 216

-- Define the number of favorable outcomes
def favorableOutcomes : ℕ := 10

-- Theorem statement
theorem dice_probability :
  (favorableOutcomes : ℚ) / totalOutcomes = 5 / 108 := by sorry

end NUMINAMATH_CALUDE_dice_probability_l1269_126950


namespace NUMINAMATH_CALUDE_bess_throws_20_meters_l1269_126970

/-- Represents the Frisbee throwing scenario -/
structure FrisbeeScenario where
  bess_throws : ℕ           -- Number of times Bess throws
  holly_throws : ℕ          -- Number of times Holly throws
  holly_distance : ℕ        -- Distance Holly throws in meters
  total_distance : ℕ        -- Total distance traveled by all Frisbees in meters

/-- Calculates Bess's throwing distance given a FrisbeeScenario -/
def bess_distance (scenario : FrisbeeScenario) : ℕ :=
  (scenario.total_distance - scenario.holly_throws * scenario.holly_distance) / (2 * scenario.bess_throws)

/-- Theorem stating that Bess's throwing distance is 20 meters in the given scenario -/
theorem bess_throws_20_meters (scenario : FrisbeeScenario) 
  (h1 : scenario.bess_throws = 4)
  (h2 : scenario.holly_throws = 5)
  (h3 : scenario.holly_distance = 8)
  (h4 : scenario.total_distance = 200) :
  bess_distance scenario = 20 := by
  sorry

end NUMINAMATH_CALUDE_bess_throws_20_meters_l1269_126970


namespace NUMINAMATH_CALUDE_combined_age_proof_l1269_126925

def jeremy_age : ℕ := 66

theorem combined_age_proof (amy_age chris_age : ℕ) 
  (h1 : amy_age = jeremy_age / 3)
  (h2 : chris_age = 2 * amy_age) :
  amy_age + jeremy_age + chris_age = 132 := by
  sorry

end NUMINAMATH_CALUDE_combined_age_proof_l1269_126925


namespace NUMINAMATH_CALUDE_complex_equation_solution_l1269_126976

theorem complex_equation_solution (a : ℝ) (i : ℂ) (hi : i * i = -1) :
  (a + 2 * i) / (2 + i) = i → a = -1 := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l1269_126976


namespace NUMINAMATH_CALUDE_area_of_bounded_region_l1269_126995

/-- The equation of the graph that partitions the plane -/
def graph_equation (x y : ℝ) : Prop :=
  y^2 + 2*x*y + 50*abs x = 500

/-- The bounded region formed by the graph -/
def bounded_region : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | ∃ (x y : ℝ), p = (x, y) ∧ 
    ((y = 25 - 2*x ∧ x ≥ 0) ∨ (y = -25 - 2*x ∧ x < 0)) ∧
    -25 ≤ y ∧ y ≤ 25}

/-- The area of the bounded region is 1250 -/
theorem area_of_bounded_region :
  MeasureTheory.volume bounded_region = 1250 := by sorry

end NUMINAMATH_CALUDE_area_of_bounded_region_l1269_126995


namespace NUMINAMATH_CALUDE_curve_and_intersection_l1269_126981

/-- The polar equation of curve C -/
def polar_equation (ρ θ a : ℝ) : Prop :=
  ρ * Real.sqrt (a^2 * Real.sin θ^2 + 4 * Real.cos θ^2) = 2 * a

/-- The Cartesian equation of curve C -/
def cartesian_equation (x y a : ℝ) : Prop :=
  4 * x^2 + a^2 * y^2 = 4 * a^2

/-- The parametric equations of line l -/
def line_equation (x y t : ℝ) : Prop :=
  x = Real.sqrt 3 + t ∧ y = 7 + Real.sqrt 3 * t

/-- Point P -/
def point_P : ℝ × ℝ := (0, 4)

/-- The distance product condition -/
def distance_product (a : ℝ) : Prop :=
  ∃ (M N : ℝ × ℝ), line_equation M.1 M.2 (M.1 - Real.sqrt 3) ∧
                   line_equation N.1 N.2 (N.1 - Real.sqrt 3) ∧
                   cartesian_equation M.1 M.2 a ∧
                   cartesian_equation N.1 N.2 a ∧
                   (M.1 - point_P.1)^2 + (M.2 - point_P.2)^2 *
                   (N.1 - point_P.1)^2 + (N.2 - point_P.2)^2 = 14^2

theorem curve_and_intersection :
  (∀ (ρ θ : ℝ), polar_equation ρ θ a ↔ cartesian_equation (ρ * Real.cos θ) (ρ * Real.sin θ) a) ∧
  (distance_product a → a = 2 * Real.sqrt 21 / 3) := by
  sorry

end NUMINAMATH_CALUDE_curve_and_intersection_l1269_126981


namespace NUMINAMATH_CALUDE_circle_radius_is_two_l1269_126941

/-- The equation of the circle -/
def circle_equation (x y : ℝ) : Prop :=
  x^2 - 8*x + y^2 + 4*y + 16 = 0

/-- The radius of the circle -/
def circle_radius : ℝ := 2

theorem circle_radius_is_two :
  ∃ (h k : ℝ), ∀ (x y : ℝ),
    circle_equation x y ↔ (x - h)^2 + (y - k)^2 = circle_radius^2 :=
sorry

end NUMINAMATH_CALUDE_circle_radius_is_two_l1269_126941


namespace NUMINAMATH_CALUDE_count_distinct_tetrahedrons_l1269_126993

/-- The number of distinct tetrahedrons that can be painted with n colors, 
    where each face is painted with exactly one color. -/
def distinctTetrahedrons (n : ℕ) : ℕ :=
  n * ((n-1)*(n-2)*(n-3)/12 + (n-1)*(n-2)/3 + (n-1) + 1)

/-- Theorem stating the number of distinct tetrahedrons that can be painted 
    with n colors, where n ≥ 4 and each face is painted with exactly one color. -/
theorem count_distinct_tetrahedrons (n : ℕ) (h : n ≥ 4) : 
  distinctTetrahedrons n = n * ((n-1)*(n-2)*(n-3)/12 + (n-1)*(n-2)/3 + (n-1) + 1) :=
by
  sorry

#check count_distinct_tetrahedrons

end NUMINAMATH_CALUDE_count_distinct_tetrahedrons_l1269_126993


namespace NUMINAMATH_CALUDE_car_cleaning_ratio_l1269_126975

/-- Given the total time spent cleaning a car and the time spent cleaning the outside,
    calculate the ratio of time spent cleaning the inside to the time spent cleaning the outside. -/
theorem car_cleaning_ratio (total_time outside_time inside_time : ℕ) : 
  total_time = outside_time + inside_time → 
  outside_time = 80 → 
  total_time = 100 → 
  (inside_time : ℚ) / outside_time = 1 / 4 := by
  sorry

end NUMINAMATH_CALUDE_car_cleaning_ratio_l1269_126975


namespace NUMINAMATH_CALUDE_special_triangle_angles_l1269_126934

/-- A triangle with a 90° angle that is three times the smallest angle has angles 90°, 60°, and 30° and is right-angled. -/
theorem special_triangle_angles :
  ∀ (a b c : ℝ),
  a > 0 ∧ b > 0 ∧ c > 0 →
  a + b + c = 180 →
  a = 90 →
  a = 3 * c →
  (a = 90 ∧ b = 60 ∧ c = 30) ∧ (∃ x, x = 90) :=
by sorry

end NUMINAMATH_CALUDE_special_triangle_angles_l1269_126934


namespace NUMINAMATH_CALUDE_lcm_of_12_and_18_l1269_126907

theorem lcm_of_12_and_18 : Nat.lcm 12 18 = 36 := by
  sorry

end NUMINAMATH_CALUDE_lcm_of_12_and_18_l1269_126907


namespace NUMINAMATH_CALUDE_rocket_max_height_l1269_126965

/-- Rocket's maximum height calculation --/
theorem rocket_max_height (a g : ℝ) (τ : ℝ) (h : a > g) (h_a : a = 30) (h_g : g = 10) (h_τ : τ = 30) :
  let v₀ := a * τ
  let y₀ := a * τ^2 / 2
  let t := v₀ / g
  let y_max := y₀ + v₀ * t - g * t^2 / 2
  y_max = 54000 ∧ y_max > 50000 := by
  sorry

#check rocket_max_height

end NUMINAMATH_CALUDE_rocket_max_height_l1269_126965


namespace NUMINAMATH_CALUDE_f_4cos2alpha_equals_4_l1269_126996

/-- A function is odd if f(-x) = -f(x) for all x -/
def IsOdd (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

/-- A function has period T if f(x + T) = f(x) for all x -/
def HasPeriod (f : ℝ → ℝ) (T : ℝ) : Prop := ∀ x, f (x + T) = f x

theorem f_4cos2alpha_equals_4 
  (f : ℝ → ℝ) (α : ℝ) 
  (h_odd : IsOdd f) 
  (h_period : HasPeriod f 5) 
  (h_f_neg3 : f (-3) = 4) 
  (h_sin_alpha : Real.sin α = Real.sqrt 3 / 2) : 
  f (4 * Real.cos (2 * α)) = 4 := by
sorry

end NUMINAMATH_CALUDE_f_4cos2alpha_equals_4_l1269_126996


namespace NUMINAMATH_CALUDE_greatest_divisor_with_remainders_l1269_126923

theorem greatest_divisor_with_remainders (n : ℕ) : 
  (1657 % n = 10 ∧ 2037 % n = 7) → n = 1 :=
by
  sorry

end NUMINAMATH_CALUDE_greatest_divisor_with_remainders_l1269_126923


namespace NUMINAMATH_CALUDE_three_number_problem_l1269_126998

theorem three_number_problem (a b c : ℚ) : 
  a + b + c = 220 ∧ 
  a = 2 * b ∧ 
  c = (1 / 3) * a →
  b = 60 := by sorry

end NUMINAMATH_CALUDE_three_number_problem_l1269_126998


namespace NUMINAMATH_CALUDE_distance_between_points_l1269_126902

theorem distance_between_points (x : ℝ) : 
  |(3 + x) - (3 - x)| = 8 → |x| = 4 := by
sorry

end NUMINAMATH_CALUDE_distance_between_points_l1269_126902


namespace NUMINAMATH_CALUDE_two_correct_statements_l1269_126930

/-- A statement about triangles -/
inductive TriangleStatement
  | altitudes_intersect : TriangleStatement
  | medians_intersect_inside : TriangleStatement
  | right_triangle_one_altitude : TriangleStatement
  | angle_bisectors_intersect : TriangleStatement

/-- Predicate to check if a statement is correct -/
def is_correct (s : TriangleStatement) : Prop :=
  match s with
  | TriangleStatement.altitudes_intersect => true
  | TriangleStatement.medians_intersect_inside => true
  | TriangleStatement.right_triangle_one_altitude => false
  | TriangleStatement.angle_bisectors_intersect => true

/-- The main theorem to prove -/
theorem two_correct_statements :
  ∃ (s1 s2 : TriangleStatement),
    s1 ≠ s2 ∧
    is_correct s1 ∧
    is_correct s2 ∧
    ∀ (s : TriangleStatement),
      s ≠ s1 ∧ s ≠ s2 → ¬(is_correct s) :=
sorry

end NUMINAMATH_CALUDE_two_correct_statements_l1269_126930


namespace NUMINAMATH_CALUDE_complex_modulus_problem_l1269_126949

theorem complex_modulus_problem (z : ℂ) : (1 - Complex.I) * z = 2 * Complex.I → Complex.abs z = Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_complex_modulus_problem_l1269_126949


namespace NUMINAMATH_CALUDE_smallest_sum_of_factors_of_72_l1269_126962

theorem smallest_sum_of_factors_of_72 :
  ∃ (a b c : ℕ), 
    a ≠ b ∧ b ≠ c ∧ a ≠ c ∧
    a * b * c = 72 ∧
    a > 0 ∧ b > 0 ∧ c > 0 ∧
    ∀ (x y z : ℕ), 
      x ≠ y ∧ y ≠ z ∧ x ≠ z →
      x * y * z = 72 →
      x > 0 ∧ y > 0 ∧ z > 0 →
      a + b + c ≤ x + y + z ∧
      a + b + c = 13 :=
by sorry

end NUMINAMATH_CALUDE_smallest_sum_of_factors_of_72_l1269_126962


namespace NUMINAMATH_CALUDE_rectangle_area_l1269_126952

/-- A rectangle with diagonal length x and length twice its width has area (2/5)x^2 -/
theorem rectangle_area (x : ℝ) (h : x > 0) : ∃ (w l : ℝ),
  w > 0 ∧ l > 0 ∧ l = 2 * w ∧ w^2 + l^2 = x^2 ∧ w * l = (2/5) * x^2 := by
  sorry


end NUMINAMATH_CALUDE_rectangle_area_l1269_126952


namespace NUMINAMATH_CALUDE_inverse_proportion_problem_l1269_126922

theorem inverse_proportion_problem (x y : ℝ) (C : ℝ) :
  (x * y = C) →  -- x and y are inversely proportional
  (x + y = 32) →
  (x - y = 8) →
  (x = 4) →
  y = 60 := by sorry

end NUMINAMATH_CALUDE_inverse_proportion_problem_l1269_126922


namespace NUMINAMATH_CALUDE_min_dot_product_l1269_126994

/-- The ellipse equation -/
def ellipse (x y : ℝ) : Prop := x^2 / 5 + y^2 / 4 = 1

/-- The circle C equation -/
def circle_C (x y : ℝ) : Prop := x^2 + y^2 = 9

/-- A point on the circle C -/
structure PointOnC where
  x : ℝ
  y : ℝ
  on_C : circle_C x y

/-- The dot product of tangent vectors PA and PB -/
def dot_product (P : PointOnC) (A B : ℝ × ℝ) : ℝ :=
  let PA := (A.1 - P.x, A.2 - P.y)
  let PB := (B.1 - P.x, B.2 - P.y)
  PA.1 * PB.1 + PA.2 * PB.2

/-- The theorem statement -/
theorem min_dot_product :
  ∀ P : PointOnC, ∃ A B : ℝ × ℝ,
    circle_C A.1 A.2 ∧ circle_C B.1 B.2 ∧
    dot_product P A B ≥ 18 * Real.sqrt 2 - 27 :=
sorry

end NUMINAMATH_CALUDE_min_dot_product_l1269_126994


namespace NUMINAMATH_CALUDE_break_even_items_l1269_126986

/-- The cost of producing each item is inversely proportional to the square root of the number of items produced. -/
def cost_production_relation (C N : ℝ) : Prop :=
  ∃ k : ℝ, C * (N^(1/2 : ℝ)) = k

/-- The cost of producing 10 items is $2100. -/
def cost_10_items : ℝ := 2100

/-- The selling price per item is $30. -/
def selling_price : ℝ := 30

/-- The break-even condition: total revenue equals total cost. -/
def break_even (N : ℝ) : Prop :=
  selling_price * N = cost_10_items * (10^(1/2 : ℝ)) / (N^(1/2 : ℝ))

/-- The number of items needed to break even is 10 * ∛49. -/
theorem break_even_items :
  ∃ N : ℝ, break_even N ∧ N = 10 * (49^(1/3 : ℝ)) :=
sorry

end NUMINAMATH_CALUDE_break_even_items_l1269_126986


namespace NUMINAMATH_CALUDE_line_plane_perpendicularity_l1269_126969

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relations
variable (contained_in : Line → Plane → Prop)
variable (perpendicular : Line → Plane → Prop)
variable (plane_perpendicular : Plane → Plane → Prop)

-- State the theorem
theorem line_plane_perpendicularity 
  (c : Line) (α β : Plane) :
  contained_in c α → perpendicular c β → plane_perpendicular α β :=
by sorry

end NUMINAMATH_CALUDE_line_plane_perpendicularity_l1269_126969


namespace NUMINAMATH_CALUDE_min_value_sum_cubes_l1269_126908

/-- Given positive real numbers x and y satisfying x³ + y³ + 3xy = 1,
    the expression (x + 1/x)³ + (y + 1/y)³ has a minimum value of 125/4. -/
theorem min_value_sum_cubes (x y : ℝ) (hx : x > 0) (hy : y > 0) 
    (h : x^3 + y^3 + 3*x*y = 1) : 
    ∃ m : ℝ, m = 125/4 ∧ ∀ a b : ℝ, a > 0 → b > 0 → a^3 + b^3 + 3*a*b = 1 → 
    (a + 1/a)^3 + (b + 1/b)^3 ≥ m :=
  sorry

end NUMINAMATH_CALUDE_min_value_sum_cubes_l1269_126908


namespace NUMINAMATH_CALUDE_trig_identity_l1269_126985

theorem trig_identity (α : ℝ) : 
  (1 + 1 / Real.cos (2 * α) + Real.tan (2 * α)) * (1 - 1 / Real.cos (2 * α) + Real.tan (2 * α)) = 2 * Real.tan (2 * α) := by
  sorry

end NUMINAMATH_CALUDE_trig_identity_l1269_126985


namespace NUMINAMATH_CALUDE_quadratic_solution_difference_l1269_126903

theorem quadratic_solution_difference (x : ℝ) : 
  (x^2 - 5*x + 12 = 2*x + 60) → 
  ∃ (x1 x2 : ℝ), x1 ≠ x2 ∧ 
    (x1^2 - 5*x1 + 12 = 2*x1 + 60) ∧ 
    (x2^2 - 5*x2 + 12 = 2*x2 + 60) ∧ 
    |x1 - x2| = Real.sqrt 241 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_solution_difference_l1269_126903


namespace NUMINAMATH_CALUDE_nested_fraction_simplification_l1269_126953

theorem nested_fraction_simplification :
  2 + (3 / (4 + (5 / (6 + (7/8))))) = 137/52 := by
  sorry

end NUMINAMATH_CALUDE_nested_fraction_simplification_l1269_126953
