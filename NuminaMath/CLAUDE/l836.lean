import Mathlib

namespace hour_hand_rotation_3_to_6_l836_83698

/-- The number of segments in a clock face. -/
def clock_segments : ℕ := 12

/-- The number of degrees in a full rotation. -/
def full_rotation : ℕ := 360

/-- The number of hours between 3 o'clock and 6 o'clock. -/
def hours_passed : ℕ := 3

/-- The degree measure of the rotation of the hour hand from 3 o'clock to 6 o'clock. -/
def hour_hand_rotation : ℕ := (full_rotation / clock_segments) * hours_passed

theorem hour_hand_rotation_3_to_6 :
  hour_hand_rotation = 90 := by sorry

end hour_hand_rotation_3_to_6_l836_83698


namespace sufficient_not_necessary_condition_l836_83600

theorem sufficient_not_necessary_condition (a : ℝ) :
  (∀ a, a > 1 → a^2 > 1) ∧
  (∃ a, a^2 > 1 ∧ ¬(a > 1)) :=
by sorry

end sufficient_not_necessary_condition_l836_83600


namespace dvd_discount_l836_83682

/-- The discount on each pack of DVDs, given the original price and the discounted price for multiple packs. -/
theorem dvd_discount (original_price : ℕ) (num_packs : ℕ) (total_price : ℕ) : 
  original_price = 107 → num_packs = 93 → total_price = 93 → 
  (original_price - (total_price / num_packs) : ℕ) = 106 :=
by sorry

end dvd_discount_l836_83682


namespace root_sum_reciprocals_l836_83650

theorem root_sum_reciprocals (p q r s : ℂ) : 
  (p^4 - 6*p^3 + 23*p^2 - 72*p + 8 = 0) →
  (q^4 - 6*q^3 + 23*q^2 - 72*q + 8 = 0) →
  (r^4 - 6*r^3 + 23*r^2 - 72*r + 8 = 0) →
  (s^4 - 6*s^3 + 23*s^2 - 72*s + 8 = 0) →
  1/(p*q) + 1/(p*r) + 1/(p*s) + 1/(q*r) + 1/(q*s) + 1/(r*s) = -9 :=
by sorry

end root_sum_reciprocals_l836_83650


namespace bounded_recurrence_sequence_is_constant_two_l836_83636

/-- A sequence of natural numbers satisfying the given recurrence relation -/
def RecurrenceSequence (a : ℕ → ℕ) : Prop :=
  ∀ n : ℕ, n ≥ 3 → a n = (a (n - 1) + a (n - 2)) / Nat.gcd (a (n - 1)) (a (n - 2))

/-- A sequence is bounded if there exists an upper bound for all its terms -/
def BoundedSequence (a : ℕ → ℕ) : Prop :=
  ∃ M : ℕ, ∀ n : ℕ, a n ≤ M

theorem bounded_recurrence_sequence_is_constant_two (a : ℕ → ℕ) 
  (h_recurrence : RecurrenceSequence a) (h_bounded : BoundedSequence a) :
  ∀ n : ℕ, a n = 2 := by
  sorry

end bounded_recurrence_sequence_is_constant_two_l836_83636


namespace smallest_resolvable_debt_l836_83668

theorem smallest_resolvable_debt (pig_value goat_value : ℕ) 
  (pig_value_pos : pig_value > 0) (goat_value_pos : goat_value > 0) :
  ∃ (debt : ℕ), debt > 0 ∧ 
  (∃ (p g : ℤ), debt = pig_value * p + goat_value * g) ∧
  (∀ (d : ℕ), d > 0 → (∃ (p g : ℤ), d = pig_value * p + goat_value * g) → d ≥ debt) :=
by
  sorry

end smallest_resolvable_debt_l836_83668


namespace divisor_count_relation_l836_83612

-- Define a function to count divisors
def count_divisors (x : ℕ) : ℕ := sorry

-- Theorem statement
theorem divisor_count_relation (n : ℕ) :
  n > 0 → count_divisors (210 * n^3) = 210 → count_divisors (64 * n^5) = 22627 :=
by sorry

end divisor_count_relation_l836_83612


namespace alex_is_26_l836_83672

-- Define the ages as natural numbers
def inez_age : ℕ := 18
def zack_age : ℕ := inez_age + 5
def jose_age : ℕ := zack_age - 3
def alex_age : ℕ := jose_age + 6

-- Theorem to prove
theorem alex_is_26 : alex_age = 26 := by
  sorry

end alex_is_26_l836_83672


namespace largest_possible_a_l836_83666

theorem largest_possible_a :
  ∀ (a b c d e : ℕ),
    a > 0 → b > 0 → c > 0 → d > 0 → e > 0 →
    a < 3 * b →
    b < 4 * c →
    c < 5 * d →
    e = d - 10 →
    e < 105 →
    a ≤ 6824 ∧ ∃ (a' b' c' d' e' : ℕ),
      a' = 6824 ∧
      b' > 0 ∧ c' > 0 ∧ d' > 0 ∧ e' > 0 ∧
      a' < 3 * b' ∧
      b' < 4 * c' ∧
      c' < 5 * d' ∧
      e' = d' - 10 ∧
      e' < 105 :=
by
  sorry


end largest_possible_a_l836_83666


namespace ratio_equality_implies_sum_ratio_l836_83688

theorem ratio_equality_implies_sum_ratio (x y z : ℝ) :
  x / 3 = y / (-4) ∧ y / (-4) = z / 7 →
  (3 * x + y + z) / y = -3 := by
sorry

end ratio_equality_implies_sum_ratio_l836_83688


namespace normal_distribution_probability_theorem_l836_83695

/-- A random variable following a normal distribution -/
structure NormalRandomVariable where
  μ : ℝ
  σ : ℝ
  hσ_pos : σ > 0

/-- The probability that a normal random variable is less than a given value -/
noncomputable def prob_less (ξ : NormalRandomVariable) (x : ℝ) : ℝ := sorry

/-- The probability that a normal random variable is greater than a given value -/
noncomputable def prob_greater (ξ : NormalRandomVariable) (x : ℝ) : ℝ := sorry

/-- The probability that a normal random variable is between two given values -/
noncomputable def prob_between (ξ : NormalRandomVariable) (a b : ℝ) : ℝ := sorry

theorem normal_distribution_probability_theorem (ξ : NormalRandomVariable) 
  (h1 : prob_less ξ (-3) = 0.2)
  (h2 : prob_greater ξ 1 = 0.2) :
  prob_between ξ (-1) 1 = 0.3 := by sorry

end normal_distribution_probability_theorem_l836_83695


namespace intersection_equals_B_l836_83673

def A : Set ℝ := {x | x^2 < 4}
def B : Set ℝ := {0, 1}

theorem intersection_equals_B : A ∩ B = B := by sorry

end intersection_equals_B_l836_83673


namespace restaurant_bill_proof_l836_83630

theorem restaurant_bill_proof : 
  ∀ (n : ℕ) (total_friends : ℕ) (paying_friends : ℕ) (extra_amount : ℕ),
    total_friends = 10 →
    paying_friends = 9 →
    extra_amount = 3 →
    n = (paying_friends * (n / total_friends + extra_amount)) →
    n = 270 := by
  sorry

end restaurant_bill_proof_l836_83630


namespace complex_modulus_l836_83613

theorem complex_modulus (z : ℂ) (h : (1 + Complex.I) * z = 2 * Complex.I) :
  Complex.abs z = Real.sqrt 2 := by
  sorry

end complex_modulus_l836_83613


namespace fruit_weight_assignment_is_correct_l836_83684

/-- Represents the fruits in the problem -/
inductive Fruit
  | orange
  | banana
  | mandarin
  | peach
  | apple

/-- Assigns weights to fruits -/
def weight_assignment : Fruit → ℕ
  | Fruit.orange   => 280
  | Fruit.banana   => 170
  | Fruit.mandarin => 100
  | Fruit.peach    => 200
  | Fruit.apple    => 150

/-- The set of possible weights -/
def possible_weights : Set ℕ := {100, 150, 170, 200, 280}

theorem fruit_weight_assignment_is_correct :
  (∀ f : Fruit, weight_assignment f ∈ possible_weights) ∧
  (weight_assignment Fruit.peach < weight_assignment Fruit.orange) ∧
  (weight_assignment Fruit.apple < weight_assignment Fruit.banana) ∧
  (weight_assignment Fruit.banana < weight_assignment Fruit.peach) ∧
  (weight_assignment Fruit.mandarin < weight_assignment Fruit.banana) ∧
  (weight_assignment Fruit.apple + weight_assignment Fruit.banana > weight_assignment Fruit.orange) ∧
  (∀ w : Fruit → ℕ, 
    (∀ f : Fruit, w f ∈ possible_weights) →
    (w Fruit.peach < w Fruit.orange) →
    (w Fruit.apple < w Fruit.banana) →
    (w Fruit.banana < w Fruit.peach) →
    (w Fruit.mandarin < w Fruit.banana) →
    (w Fruit.apple + w Fruit.banana > w Fruit.orange) →
    w = weight_assignment) :=
by sorry

end fruit_weight_assignment_is_correct_l836_83684


namespace gym_member_count_l836_83645

/-- Represents a gym with its pricing and revenue information -/
structure Gym where
  charge_per_half_month : ℕ
  monthly_revenue : ℕ

/-- Calculates the number of members in the gym -/
def member_count (g : Gym) : ℕ :=
  g.monthly_revenue / (2 * g.charge_per_half_month)

/-- Theorem stating that a gym with the given parameters has 300 members -/
theorem gym_member_count :
  ∃ (g : Gym), g.charge_per_half_month = 18 ∧ g.monthly_revenue = 10800 ∧ member_count g = 300 := by
  sorry

end gym_member_count_l836_83645


namespace combined_height_sara_joe_l836_83634

/-- The combined height of Sara and Joe is 120 inches -/
theorem combined_height_sara_joe : 
  ∀ (sara_height joe_height : ℕ),
  joe_height = 2 * sara_height + 6 →
  joe_height = 82 →
  sara_height + joe_height = 120 :=
by
  sorry

end combined_height_sara_joe_l836_83634


namespace part_one_part_two_l836_83653

-- Define the function f
def f (x k m : ℝ) : ℝ := |x^2 - k*x - m|

-- Theorem for part (1)
theorem part_one (k m : ℝ) :
  m = 2 * k^2 →
  (∀ x y, 1 < x ∧ x < y → f x k m < f y k m) →
  -1 ≤ k ∧ k ≤ 1/2 := by sorry

-- Theorem for part (2)
theorem part_two (k m a b : ℝ) :
  (∀ x, x ∈ Set.Icc a b → f x k m ≤ 1) →
  b - a ≤ 2 * Real.sqrt 2 := by sorry

end part_one_part_two_l836_83653


namespace complementary_angle_l836_83622

theorem complementary_angle (A : ℝ) (h : A = 25) : 90 - A = 65 := by
  sorry

end complementary_angle_l836_83622


namespace sphere_volume_increase_l836_83689

theorem sphere_volume_increase (r : ℝ) (h : r > 0) :
  let new_r := r * Real.sqrt 2
  (4 / 3 * Real.pi * new_r^3) / (4 / 3 * Real.pi * r^3) = 2 * Real.sqrt 2 := by
  sorry

end sphere_volume_increase_l836_83689


namespace soccer_team_games_theorem_l836_83640

/-- Represents the ratio of wins, losses, and ties for a soccer team -/
structure GameRatio :=
  (wins : ℕ)
  (losses : ℕ)
  (ties : ℕ)

/-- Calculates the total number of games played given a game ratio and number of losses -/
def totalGames (ratio : GameRatio) (numLosses : ℕ) : ℕ :=
  let gamesPerPart := numLosses / ratio.losses
  (ratio.wins + ratio.losses + ratio.ties) * gamesPerPart

/-- Theorem stating that for a team with a 4:3:1 win:loss:tie ratio and 9 losses, 
    the total number of games played is 24 -/
theorem soccer_team_games_theorem :
  let ratio : GameRatio := ⟨4, 3, 1⟩
  totalGames ratio 9 = 24 := by
  sorry

end soccer_team_games_theorem_l836_83640


namespace equation_solution_l836_83655

theorem equation_solution (x : ℝ) (h : x ≠ -1) :
  (x^2 + x + 1) / (x + 1) = x + 3 ↔ x = -2/3 :=
by sorry

end equation_solution_l836_83655


namespace problem_statement_l836_83659

theorem problem_statement (θ : ℝ) : 
  ((∀ x : ℝ, x^2 - 2*x*Real.sin θ + 1 ≥ 0) ∨ 
   (∀ α β : ℝ, Real.sin (α + β) ≤ Real.sin α + Real.sin β)) := by
  sorry

end problem_statement_l836_83659


namespace arithmetic_sequence_problem_l836_83604

/-- An arithmetic sequence -/
def ArithmeticSequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

/-- The theorem statement -/
theorem arithmetic_sequence_problem (a : ℕ → ℝ) 
    (h_arith : ArithmeticSequence a)
    (h_sum : a 6 + a 8 = 10)
    (h_a3 : a 3 = 1) :
    a 11 = 9 := by
  sorry

end arithmetic_sequence_problem_l836_83604


namespace incorrect_operation_l836_83637

theorem incorrect_operation (a b c d e f : ℝ) 
  (h1 : a = Real.sqrt 2)
  (h2 : b = Real.sqrt 3)
  (h3 : c = Real.sqrt 5)
  (h4 : d = Real.sqrt 6)
  (h5 : e = Real.sqrt (1/2))
  (h6 : f = Real.sqrt 8)
  (prop1 : a * b = d)
  (prop2 : a / e = 2)
  (prop3 : a + f = 3 * a) :
  a + b ≠ c := by
  sorry

end incorrect_operation_l836_83637


namespace subset_implies_m_values_l836_83674

def A (m : ℝ) : Set ℝ := {-1, 3, m^2}
def B (m : ℝ) : Set ℝ := {3, 2*m - 1}

theorem subset_implies_m_values (m : ℝ) : B m ⊆ A m → m = 0 ∨ m = 1 := by
  sorry

end subset_implies_m_values_l836_83674


namespace problem_statement_l836_83691

theorem problem_statement (a b : ℝ) :
  (4 / (Real.sqrt 6 + Real.sqrt 2) - 1 / (Real.sqrt 3 + Real.sqrt 2) = Real.sqrt a - Real.sqrt b) →
  a - b = 3 := by
  sorry

end problem_statement_l836_83691


namespace half_plus_six_equals_eleven_l836_83641

theorem half_plus_six_equals_eleven (n : ℝ) : (1/2 : ℝ) * n + 6 = 11 → n = 10 := by
  sorry

end half_plus_six_equals_eleven_l836_83641


namespace wood_stove_burn_rate_l836_83683

/-- Wood stove burning rate problem -/
theorem wood_stove_burn_rate 
  (morning_duration : ℝ) 
  (afternoon_duration : ℝ)
  (morning_rate : ℝ) 
  (starting_wood : ℝ) 
  (ending_wood : ℝ) : 
  morning_duration = 4 →
  afternoon_duration = 4 →
  morning_rate = 2 →
  starting_wood = 30 →
  ending_wood = 3 →
  ∃ (afternoon_rate : ℝ), 
    afternoon_rate = (starting_wood - ending_wood - morning_duration * morning_rate) / afternoon_duration ∧ 
    afternoon_rate = 4.75 := by
  sorry

end wood_stove_burn_rate_l836_83683


namespace e_value_proof_l836_83617

theorem e_value_proof (a b c : ℕ) (e : ℚ) 
  (h1 : a = 105)
  (h2 : b = 126)
  (h3 : c = 63)
  (h4 : a^3 - b^2 + c^2 = 21 * 25 * 45 * e) :
  e = 47.7 := by
  sorry

end e_value_proof_l836_83617


namespace max_value_d_l836_83625

theorem max_value_d (a b c d : ℝ) 
  (sum_eq : a + b + c + d = 10)
  (sum_prod_eq : a*b + a*c + a*d + b*c + b*d + c*d = 20) :
  d ≤ (5 + Real.sqrt 105) / 2 ∧ 
  ∃ a b c d, a + b + c + d = 10 ∧ 
             a*b + a*c + a*d + b*c + b*d + c*d = 20 ∧
             d = (5 + Real.sqrt 105) / 2 :=
by sorry

end max_value_d_l836_83625


namespace max_min_product_l836_83610

theorem max_min_product (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c)
  (sum_eq : a + b + c = 12) (prod_sum_eq : a * b + b * c + c * a = 30) :
  ∃ (m : ℝ), m = min (a * b) (min (b * c) (c * a)) ∧ m ≤ 9 * Real.sqrt 2 ∧
  ∃ (a' b' c' : ℝ), 0 < a' ∧ 0 < b' ∧ 0 < c' ∧
    a' + b' + c' = 12 ∧ a' * b' + b' * c' + c' * a' = 30 ∧
    min (a' * b') (min (b' * c') (c' * a')) = 9 * Real.sqrt 2 :=
by sorry

end max_min_product_l836_83610


namespace parallel_lines_parameter_l836_83616

/-- Given two lines in the plane, prove that the parameter 'a' must equal 4 for the lines to be parallel -/
theorem parallel_lines_parameter (a : ℝ) : 
  (∀ x y : ℝ, 3 * x + (1 - a) * y + 1 = 0 ↔ x - y + 2 = 0) → a = 4 := by
  sorry

end parallel_lines_parameter_l836_83616


namespace angle_B_value_l836_83670

theorem angle_B_value (a b c : ℝ) (h : a^2 + c^2 - b^2 = Real.sqrt 3 * a * c) :
  let B := Real.arccos ((a^2 + c^2 - b^2) / (2 * a * c))
  B = π / 6 := by
sorry

end angle_B_value_l836_83670


namespace range_of_m_and_n_l836_83647

-- Define sets A and B
def A (m : ℝ) : Set (ℝ × ℝ) := {p : ℝ × ℝ | 2 * p.1 - p.2 + m > 0}
def B (n : ℝ) : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.1 + p.2 - n ≤ 0}

-- Define point P
def P : ℝ × ℝ := (2, 3)

-- Theorem statement
theorem range_of_m_and_n (m n : ℝ) 
  (h1 : P ∈ A m) 
  (h2 : P ∉ B n) : 
  m > -1 ∧ n < 5 := by
  sorry

end range_of_m_and_n_l836_83647


namespace dice_surface_sum_l836_83629

/-- The number of dice in the arrangement -/
def num_dice : Nat := 2012

/-- The sum of points on all faces of a single die -/
def die_sum : Nat := 21

/-- The sum of points on opposite faces of a die -/
def opposite_faces_sum : Nat := 7

/-- A value representing the number of points on one end face of the first die -/
def X : Fin 6 := sorry

/-- The sum of points on the surface of the arranged dice -/
def surface_sum : Nat := 28175 + 2 * X.val

theorem dice_surface_sum :
  surface_sum = num_dice * die_sum - (num_dice - 1) * opposite_faces_sum + 2 * X.val :=
by sorry

end dice_surface_sum_l836_83629


namespace max_profit_at_70_best_selling_price_l836_83628

/-- Represents the profit function for a product with given pricing and demand characteristics -/
def profit (x : ℕ) : ℝ :=
  (50 + x - 40) * (50 - x)

/-- Theorem stating that the maximum profit occurs when the selling price is 70 yuan -/
theorem max_profit_at_70 :
  ∀ x : ℕ, x < 50 → x > 0 → profit x ≤ profit 20 :=
sorry

/-- Corollary stating that the best selling price is 70 yuan -/
theorem best_selling_price :
  ∃ x : ℕ, x < 50 ∧ x > 0 ∧ ∀ y : ℕ, y < 50 → y > 0 → profit y ≤ profit x :=
sorry

end max_profit_at_70_best_selling_price_l836_83628


namespace cyclists_meeting_time_l836_83619

/-- Two cyclists moving in opposite directions on a circular track meet at the starting point -/
theorem cyclists_meeting_time
  (circumference : ℝ)
  (speed1 : ℝ)
  (speed2 : ℝ)
  (h1 : circumference = 675)
  (h2 : speed1 = 7)
  (h3 : speed2 = 8) :
  circumference / (speed1 + speed2) = 45 :=
by sorry

end cyclists_meeting_time_l836_83619


namespace P_on_angle_bisector_PQ_parallel_to_x_axis_l836_83627

-- Define points P and Q
def P (a : ℝ) : ℝ × ℝ := (a + 1, 2 * a - 3)
def Q : ℝ × ℝ := (2, 3)

-- Theorem for the first condition
theorem P_on_angle_bisector :
  ∃ a : ℝ, P a = (5, 5) ∧ (P a).1 = (P a).2 := by sorry

-- Theorem for the second condition
theorem PQ_parallel_to_x_axis :
  ∃ a : ℝ, (P a).2 = Q.2 → |((P a).1 - Q.1)| = 2 := by sorry

end P_on_angle_bisector_PQ_parallel_to_x_axis_l836_83627


namespace exam_score_proof_l836_83646

theorem exam_score_proof (mean : ℝ) (low_score : ℝ) (std_dev_below : ℝ) (std_dev_above : ℝ) :
  mean = 88.8 →
  low_score = 86 →
  std_dev_below = 7 →
  std_dev_above = 3 →
  low_score = mean - std_dev_below * ((mean - low_score) / std_dev_below) →
  mean + std_dev_above * ((mean - low_score) / std_dev_below) = 90 := by
  sorry

end exam_score_proof_l836_83646


namespace committee_combinations_l836_83665

theorem committee_combinations : Nat.choose 8 5 = 56 := by sorry

end committee_combinations_l836_83665


namespace calculation_proof_l836_83694

theorem calculation_proof :
  (9.5 * 101 = 959.5) ∧
  (12.5 * 8.8 = 110) ∧
  (38.4 * 187 - 15.4 * 384 + 3.3 * 16 = 1320) ∧
  (5.29 * 73 + 52.9 * 2.7 = 529) := by
  sorry

end calculation_proof_l836_83694


namespace ordered_pairs_count_l836_83618

theorem ordered_pairs_count : 
  { p : ℤ × ℤ | (p.1 : ℤ) ^ 2019 + (p.2 : ℤ) ^ 2 = 2 * (p.2 : ℤ) }.Finite ∧ 
  { p : ℤ × ℤ | (p.1 : ℤ) ^ 2019 + (p.2 : ℤ) ^ 2 = 2 * (p.2 : ℤ) }.ncard = 3 :=
by sorry

end ordered_pairs_count_l836_83618


namespace circle_center_and_radius_l836_83690

/-- Given a circle with equation x^2 + y^2 + 2x - 4y - 4 = 0, 
    its center is at (-1, 2) and its radius is 3. -/
theorem circle_center_and_radius :
  ∃ (center : ℝ × ℝ) (radius : ℝ),
    center = (-1, 2) ∧ 
    radius = 3 ∧
    ∀ (x y : ℝ), x^2 + y^2 + 2*x - 4*y - 4 = 0 ↔ 
      (x - center.1)^2 + (y - center.2)^2 = radius^2 :=
by sorry

end circle_center_and_radius_l836_83690


namespace winter_spending_calculation_l836_83679

/-- The amount spent by the Surf City government at the end of November 1988, in millions of dollars. -/
def spent_end_november : ℝ := 3.3

/-- The amount spent by the Surf City government at the end of February 1989, in millions of dollars. -/
def spent_end_february : ℝ := 7.0

/-- The amount spent during December, January, and February, in millions of dollars. -/
def winter_spending : ℝ := spent_end_february - spent_end_november

theorem winter_spending_calculation : winter_spending = 3.7 := by
  sorry

end winter_spending_calculation_l836_83679


namespace regression_coefficient_correlation_same_sign_l836_83620

/-- Linear regression model -/
structure LinearRegression where
  a : ℝ
  b : ℝ
  x : ℝ → ℝ
  y : ℝ → ℝ
  equation : ∀ t, y t = a + b * x t

/-- Correlation coefficient -/
def correlation_coefficient (x y : ℝ → ℝ) : ℝ := sorry

theorem regression_coefficient_correlation_same_sign 
  (model : LinearRegression) 
  (r : ℝ) 
  (h_r : r = correlation_coefficient model.x model.y) :
  (r > 0 ∧ model.b > 0) ∨ (r < 0 ∧ model.b < 0) ∨ (r = 0 ∧ model.b = 0) :=
sorry

end regression_coefficient_correlation_same_sign_l836_83620


namespace simplify_expression_l836_83652

theorem simplify_expression :
  (3/2) * Real.sqrt 5 - (1/3) * Real.sqrt 6 + (1/2) * (-Real.sqrt 5 + 2 * Real.sqrt 6) =
  Real.sqrt 5 + (2/3) * Real.sqrt 6 := by
  sorry

end simplify_expression_l836_83652


namespace right_triangle_sets_l836_83608

def is_pythagorean_triple (a b c : ℕ) : Prop :=
  a * a + b * b = c * c ∨ a * a + c * c = b * b ∨ b * b + c * c = a * a

theorem right_triangle_sets :
  is_pythagorean_triple 8 15 17 ∧
  is_pythagorean_triple 7 24 25 ∧
  is_pythagorean_triple 3 4 5 ∧
  ¬ is_pythagorean_triple 2 3 4 :=
by sorry

end right_triangle_sets_l836_83608


namespace acme_cheaper_at_min_shirts_l836_83611

/-- Acme T-Shirt Company's pricing function -/
def acme_price (x : ℕ) : ℕ := 60 + 8 * x

/-- Delta T-shirt Company's pricing function -/
def delta_price (x : ℕ) : ℕ := 12 * x

/-- The minimum number of shirts for which Acme is cheaper than Delta -/
def min_shirts_for_acme_cheaper : ℕ := 16

theorem acme_cheaper_at_min_shirts :
  acme_price min_shirts_for_acme_cheaper < delta_price min_shirts_for_acme_cheaper ∧
  ∀ n : ℕ, n < min_shirts_for_acme_cheaper →
    acme_price n ≥ delta_price n :=
by sorry

end acme_cheaper_at_min_shirts_l836_83611


namespace fifth_month_sale_l836_83639

theorem fifth_month_sale
  (sale1 sale2 sale3 sale4 sale6 : ℕ)
  (average : ℚ)
  (h1 : sale1 = 2500)
  (h2 : sale2 = 6500)
  (h3 : sale3 = 9855)
  (h4 : sale4 = 7230)
  (h6 : sale6 = 11915)
  (h_avg : average = 7500)
  (h_total : (sale1 + sale2 + sale3 + sale4 + sale6 + sale5) / 6 = average) :
  sale5 = 7000 := by
sorry

end fifth_month_sale_l836_83639


namespace quadratic_factorization_l836_83609

theorem quadratic_factorization (x : ℝ) : 2 * x^2 - 4 * x + 2 = 2 * (x - 1)^2 := by
  sorry

end quadratic_factorization_l836_83609


namespace magic_square_sum_l836_83657

/-- Represents a 3x3 magic square with five unknown values -/
structure MagicSquare where
  a : ℕ
  b : ℕ
  c : ℕ
  d : ℕ
  e : ℕ

/-- The magic sum (sum of each row, column, and diagonal) -/
def magicSum (sq : MagicSquare) : ℕ := 15 + sq.b + 27

/-- Conditions for the magic square -/
def isMagicSquare (sq : MagicSquare) : Prop :=
  magicSum sq = 15 + sq.b + 27
  ∧ magicSum sq = 24 + sq.a + sq.d
  ∧ magicSum sq = sq.e + 18 + sq.c
  ∧ magicSum sq = 15 + sq.a + sq.c
  ∧ magicSum sq = sq.b + sq.a + 18
  ∧ magicSum sq = 27 + sq.d + sq.c
  ∧ magicSum sq = 15 + sq.a + sq.c
  ∧ magicSum sq = 27 + sq.a + sq.e

theorem magic_square_sum (sq : MagicSquare) (h : isMagicSquare sq) : sq.d + sq.e = 47 := by
  sorry


end magic_square_sum_l836_83657


namespace gcd_8008_12012_l836_83676

theorem gcd_8008_12012 : Nat.gcd 8008 12012 = 4004 := by
  sorry

end gcd_8008_12012_l836_83676


namespace train_length_l836_83602

/-- The length of a train given its speed and time to cross a pole -/
theorem train_length (speed_kmh : ℝ) (time_s : ℝ) : 
  speed_kmh = 180 → time_s = 7 → speed_kmh * (1000 / 3600) * time_s = 350 := by
  sorry

end train_length_l836_83602


namespace shopkeeper_stock_worth_l836_83681

def item_A_profit_percentage : Real := 0.15
def item_A_loss_percentage : Real := 0.10
def item_A_profit_portion : Real := 0.25
def item_A_loss_portion : Real := 0.75

def item_B_profit_percentage : Real := 0.20
def item_B_loss_percentage : Real := 0.05
def item_B_profit_portion : Real := 0.30
def item_B_loss_portion : Real := 0.70

def item_C_profit_percentage : Real := 0.10
def item_C_loss_percentage : Real := 0.08
def item_C_profit_portion : Real := 0.40
def item_C_loss_portion : Real := 0.60

def tax_rate : Real := 0.12
def net_loss : Real := 750

def cost_price_ratio_A : Real := 2
def cost_price_ratio_B : Real := 3
def cost_price_ratio_C : Real := 4

theorem shopkeeper_stock_worth (x : Real) :
  let cost_A := cost_price_ratio_A * x
  let cost_B := cost_price_ratio_B * x
  let cost_C := cost_price_ratio_C * x
  let profit_loss_A := item_A_profit_portion * cost_A * item_A_profit_percentage - 
                       item_A_loss_portion * cost_A * item_A_loss_percentage
  let profit_loss_B := item_B_profit_portion * cost_B * item_B_profit_percentage - 
                       item_B_loss_portion * cost_B * item_B_loss_percentage
  let profit_loss_C := item_C_profit_portion * cost_C * item_C_profit_percentage - 
                       item_C_loss_portion * cost_C * item_C_loss_percentage
  let total_profit_loss := profit_loss_A + profit_loss_B + profit_loss_C
  total_profit_loss = -net_loss →
  cost_A = 46875 ∧ cost_B = 70312.5 ∧ cost_C = 93750 := by
sorry


end shopkeeper_stock_worth_l836_83681


namespace remaining_three_average_l836_83658

theorem remaining_three_average (total : ℕ) (all_avg first_four_avg next_three_avg following_two_avg : ℚ) :
  total = 12 →
  all_avg = 6.30 →
  first_four_avg = 5.60 →
  next_three_avg = 4.90 →
  following_two_avg = 7.25 →
  (total * all_avg - (4 * first_four_avg + 3 * next_three_avg + 2 * following_two_avg)) / 3 = 8 := by
  sorry

end remaining_three_average_l836_83658


namespace hyperbola_eccentricity_l836_83624

/-- A hyperbola with the property that the distance from its vertex to its asymptote
    is 1/4 of the length of its imaginary axis has eccentricity 2. -/
theorem hyperbola_eccentricity (a b : ℝ) (h : a > 0) (k : b > 0) :
  (a * b / Real.sqrt (a^2 + b^2) = 1/4 * (2*b)) → (Real.sqrt (a^2 + b^2) / a = 2) :=
by sorry

end hyperbola_eccentricity_l836_83624


namespace derivative_at_negative_two_l836_83615

-- Define a real-valued function f
variable (f : ℝ → ℝ)

-- State the theorem
theorem derivative_at_negative_two (h : ∀ ε > 0, ∃ δ > 0, ∀ Δx ≠ 0, |Δx| < δ → 
  |((f (-2 + Δx) - f (-2 - Δx)) / Δx) - (-2)| < ε) : 
  deriv f (-2) = -1 := by
  sorry

end derivative_at_negative_two_l836_83615


namespace solution_set_quadratic_inequality_l836_83680

theorem solution_set_quadratic_inequality :
  {x : ℝ | x^2 + x - 2 < 0} = Set.Ioo (-2 : ℝ) 1 := by sorry

end solution_set_quadratic_inequality_l836_83680


namespace sum_of_bases_equals_1500_l836_83697

/-- Converts a number from base 13 to base 10 -/
def base13ToBase10 (digits : List Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * (13 ^ i)) 0

/-- Converts a number from base 14 to base 10 -/
def base14ToBase10 (digits : List Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * (14 ^ i)) 0

/-- The main theorem to prove -/
theorem sum_of_bases_equals_1500 :
  let num1 := base13ToBase10 [6, 2, 3]
  let num2 := base14ToBase10 [9, 12, 4]
  num1 + num2 = 1500 := by sorry

end sum_of_bases_equals_1500_l836_83697


namespace no_solution_exists_l836_83623

theorem no_solution_exists : ∀ k : ℕ, k^6 + k^4 + k^2 ≠ 10^(k+1) + 9 := by
  sorry

end no_solution_exists_l836_83623


namespace water_in_sport_formulation_l836_83678

/-- Represents the ratio of ingredients in a flavored drink formulation -/
structure DrinkRatio :=
  (flavoring : ℚ)
  (corn_syrup : ℚ)
  (water : ℚ)

/-- The standard formulation ratio -/
def standard_ratio : DrinkRatio :=
  { flavoring := 1, corn_syrup := 12, water := 30 }

/-- The sport formulation ratio -/
def sport_ratio : DrinkRatio :=
  { flavoring := 3 * standard_ratio.flavoring,
    corn_syrup := standard_ratio.corn_syrup,
    water := 60 * standard_ratio.flavoring }

/-- The amount of water in ounces given the amount of corn syrup in the sport formulation -/
def water_amount (corn_syrup_oz : ℚ) : ℚ :=
  corn_syrup_oz * (sport_ratio.water / sport_ratio.corn_syrup)

theorem water_in_sport_formulation :
  water_amount 2 = 120 :=
sorry

end water_in_sport_formulation_l836_83678


namespace x_plus_twice_y_l836_83631

theorem x_plus_twice_y (x y z : ℚ) : 
  x = y / 3 → y = z / 4 → z = 100 → x + 2 * y = 175 / 3 := by
  sorry

end x_plus_twice_y_l836_83631


namespace rectangle_dimensions_l836_83649

theorem rectangle_dimensions (x : ℝ) : 
  (x - 3 > 0) →
  (3 * x + 4 > 0) →
  ((x - 3) * (3 * x + 4) = 12 * x - 9) →
  (x = (17 + 5 * Real.sqrt 13) / 6) :=
by sorry

end rectangle_dimensions_l836_83649


namespace meeting_point_x_coordinate_l836_83662

-- Define the river boundaries
def river_left : ℝ := 0
def river_right : ℝ := 25

-- Define the current speed
def current_speed : ℝ := 2

-- Define the starting positions
def mallard_start : ℝ × ℝ := (0, 0)
def wigeon_start : ℝ × ℝ := (25, 0)

-- Define the meeting point y-coordinate
def meeting_y : ℝ := 22

-- Define the speeds relative to water
def mallard_speed : ℝ := 4
def wigeon_speed : ℝ := 3

-- Theorem statement
theorem meeting_point_x_coordinate :
  ∃ (x : ℝ), 
    x > river_left ∧ 
    x < river_right ∧ 
    (∃ (t : ℝ), t > 0 ∧
      (mallard_start.1 + mallard_speed * t * Real.cos (Real.arctan ((meeting_y - mallard_start.2) / (x - mallard_start.1))) = x) ∧
      (wigeon_start.1 - wigeon_speed * t * Real.cos (Real.arctan ((meeting_y - wigeon_start.2) / (wigeon_start.1 - x))) = x) ∧
      (mallard_start.2 + (mallard_speed * Real.sin (Real.arctan ((meeting_y - mallard_start.2) / (x - mallard_start.1))) + current_speed) * t = meeting_y) ∧
      (wigeon_start.2 + (wigeon_speed * Real.sin (Real.arctan ((meeting_y - wigeon_start.2) / (wigeon_start.1 - x))) + current_speed) * t = meeting_y)) ∧
    x = 100 / 7 := by
  sorry

end meeting_point_x_coordinate_l836_83662


namespace arithmetic_sequence_property_l836_83663

/-- An arithmetic sequence -/
def ArithmeticSequence (a : ℕ → ℝ) : Prop :=
  ∀ n, a (n + 2) - a (n + 1) = a (n + 1) - a n

theorem arithmetic_sequence_property (a : ℕ → ℝ) 
  (h_arithmetic : ArithmeticSequence a)
  (h_condition : a 1 + 3 * a 8 + a 15 = 120) :
  2 * a 6 - a 4 = 24 := by
  sorry

end arithmetic_sequence_property_l836_83663


namespace range_m_prop_p_range_m_prop_p_not_q_l836_83626

/-- Proposition p: For all real x, x²-2mx-3m > 0 -/
def prop_p (m : ℝ) : Prop := ∀ x : ℝ, x^2 - 2*m*x - 3*m > 0

/-- Proposition q: There exists a real x such that x²+4mx+1 < 0 -/
def prop_q (m : ℝ) : Prop := ∃ x : ℝ, x^2 + 4*m*x + 1 < 0

/-- The range of m for which proposition p is true -/
theorem range_m_prop_p : 
  {m : ℝ | prop_p m} = Set.Ioo (-3) 0 :=
sorry

/-- The range of m for which proposition p is true and proposition q is false -/
theorem range_m_prop_p_not_q : 
  {m : ℝ | prop_p m ∧ ¬(prop_q m)} = Set.Ico (-1/2) 0 :=
sorry

end range_m_prop_p_range_m_prop_p_not_q_l836_83626


namespace range_of_m_l836_83656

-- Define the propositions p and q
def p (m : ℝ) : Prop := ∀ x y : ℝ, x^2 / (2*m) - y^2 / (m-2) = 1 → m > 2

def q (m : ℝ) : Prop := ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ 
  x₁^2 + (2*m-3)*x₁ + 1 = 0 ∧ x₂^2 + (2*m-3)*x₂ + 1 = 0

-- State the theorem
theorem range_of_m : 
  (∀ m : ℝ, ¬(p m ∧ q m)) → 
  (∀ m : ℝ, p m ∨ q m) → 
  ∀ m : ℝ, (2 < m ∧ m ≤ 5/2) ∨ m < 1/2 :=
sorry

end range_of_m_l836_83656


namespace isosceles_right_triangle_hypotenuse_length_l836_83632

/-- Represents an isosceles right triangle -/
structure IsoscelesRightTriangle where
  /-- The length of a leg of the triangle -/
  leg : ℝ
  /-- The length of the hypotenuse of the triangle -/
  hypotenuse : ℝ
  /-- Condition that the hypotenuse is √2 times the leg -/
  hyp_leg_relation : hypotenuse = leg * Real.sqrt 2
  /-- Condition that the leg is positive -/
  leg_pos : leg > 0

/-- The theorem stating that for an isosceles right triangle with area 64, its hypotenuse is 16 -/
theorem isosceles_right_triangle_hypotenuse_length
  (t : IsoscelesRightTriangle)
  (area_eq : t.leg * t.leg / 2 = 64) :
  t.hypotenuse = 16 := by
  sorry

end isosceles_right_triangle_hypotenuse_length_l836_83632


namespace max_value_expression_l836_83606

theorem max_value_expression (x y : ℚ) (hx : x ≠ 0) (hy : y ≠ 0) :
  x / abs x + abs y / y - (x * y) / abs (x * y) ≤ 1 := by
  sorry

end max_value_expression_l836_83606


namespace manuscript_has_100_pages_l836_83638

/-- Represents the pricing and revision structure of a typing service --/
structure TypingService where
  initial_cost : ℕ  -- Cost per page for initial typing
  revision_cost : ℕ  -- Cost per page for each revision

/-- Represents the manuscript details --/
structure Manuscript where
  total_pages : ℕ
  once_revised : ℕ
  twice_revised : ℕ

/-- Calculates the total cost for typing and revising a manuscript --/
def total_cost (service : TypingService) (manuscript : Manuscript) : ℕ :=
  service.initial_cost * manuscript.total_pages +
  service.revision_cost * manuscript.once_revised +
  2 * service.revision_cost * manuscript.twice_revised

/-- Theorem stating that given the conditions, the manuscript has 100 pages --/
theorem manuscript_has_100_pages (service : TypingService) (manuscript : Manuscript) :
  service.initial_cost = 5 →
  service.revision_cost = 4 →
  manuscript.once_revised = 30 →
  manuscript.twice_revised = 20 →
  total_cost service manuscript = 780 →
  manuscript.total_pages = 100 :=
by
  sorry


end manuscript_has_100_pages_l836_83638


namespace molecular_weight_8_moles_Al2O3_l836_83686

/-- The atomic weight of Aluminum in g/mol -/
def atomic_weight_Al : ℝ := 26.98

/-- The atomic weight of Oxygen in g/mol -/
def atomic_weight_O : ℝ := 16.00

/-- The number of Aluminum atoms in one molecule of Al2O3 -/
def num_Al_atoms : ℕ := 2

/-- The number of Oxygen atoms in one molecule of Al2O3 -/
def num_O_atoms : ℕ := 3

/-- The number of moles of Al2O3 -/
def num_moles : ℕ := 8

/-- The molecular weight of Al2O3 in g/mol -/
def molecular_weight_Al2O3 : ℝ :=
  num_Al_atoms * atomic_weight_Al + num_O_atoms * atomic_weight_O

/-- Theorem: The molecular weight of 8 moles of Al2O3 is 815.68 grams -/
theorem molecular_weight_8_moles_Al2O3 :
  num_moles * molecular_weight_Al2O3 = 815.68 := by
  sorry


end molecular_weight_8_moles_Al2O3_l836_83686


namespace chairs_to_hall_l836_83642

/-- Calculates the total number of chairs taken to the hall given the number of students,
    chairs per trip, and number of trips. -/
def totalChairs (students : ℕ) (chairsPerTrip : ℕ) (numTrips : ℕ) : ℕ :=
  students * chairsPerTrip * numTrips

/-- Proves that 5 students, each carrying 5 chairs per trip and making 10 trips,
    will take a total of 250 chairs to the hall. -/
theorem chairs_to_hall :
  totalChairs 5 5 10 = 250 := by
  sorry

#eval totalChairs 5 5 10

end chairs_to_hall_l836_83642


namespace orchard_fruit_count_l836_83654

def apple_trees : ℕ := 50
def orange_trees : ℕ := 30
def apple_baskets_per_tree : ℕ := 25
def orange_baskets_per_tree : ℕ := 15
def apples_per_basket : ℕ := 18
def oranges_per_basket : ℕ := 12

theorem orchard_fruit_count :
  let total_apples := apple_trees * apple_baskets_per_tree * apples_per_basket
  let total_oranges := orange_trees * orange_baskets_per_tree * oranges_per_basket
  total_apples = 22500 ∧ total_oranges = 5400 := by
  sorry

end orchard_fruit_count_l836_83654


namespace no_valid_partition_l836_83687

/-- A partition of integers into three subsets -/
def IntPartition := ℤ → Fin 3

/-- Property that n, n-50, and n+1987 belong to different subsets -/
def ValidPartition (p : IntPartition) : Prop :=
  ∀ n : ℤ, p n ≠ p (n - 50) ∧ p n ≠ p (n + 1987) ∧ p (n - 50) ≠ p (n + 1987)

/-- Theorem stating the impossibility of such a partition -/
theorem no_valid_partition : ¬ ∃ p : IntPartition, ValidPartition p := by
  sorry

end no_valid_partition_l836_83687


namespace number_line_problem_l836_83699

theorem number_line_problem (a b c : ℚ) : 
  a = (-4)^2 - 8 →
  b = -c →
  |c - a| = 3 →
  ((b = -5 ∧ c = 5) ∨ (b = -11 ∧ c = 11)) ∧
  (-a^2 + b - c = -74 ∨ -a^2 + b - c = -86) :=
by sorry

end number_line_problem_l836_83699


namespace money_distribution_l836_83603

/-- Given three people A, B, and C with money amounts a, b, and c respectively,
    if their total amount is 500, B and C together have 310, and C has 10,
    then A and C together have 200. -/
theorem money_distribution (a b c : ℕ) : 
  a + b + c = 500 → b + c = 310 → c = 10 → a + c = 200 := by
  sorry

#check money_distribution

end money_distribution_l836_83603


namespace smallest_repeating_block_of_8_11_l836_83660

theorem smallest_repeating_block_of_8_11 : ∃ (n : ℕ), n > 0 ∧ 
  (∀ (k : ℕ), k > 0 → (8 * 10^k) % 11 = (8 * 10^(k + n)) % 11) ∧
  (∀ (m : ℕ), m > 0 → m < n → ∃ (k : ℕ), k > 0 ∧ (8 * 10^k) % 11 ≠ (8 * 10^(k + m)) % 11) ∧
  n = 2 :=
sorry

end smallest_repeating_block_of_8_11_l836_83660


namespace isosceles_base_length_l836_83696

/-- Represents a triangle with sides a, b, and c. -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  pos_a : a > 0
  pos_b : b > 0
  pos_c : c > 0

/-- An equilateral triangle is a triangle with all sides equal. -/
def IsEquilateral (t : Triangle) : Prop :=
  t.a = t.b ∧ t.b = t.c

/-- An isosceles triangle is a triangle with at least two sides equal. -/
def IsIsosceles (t : Triangle) : Prop :=
  t.a = t.b ∨ t.b = t.c ∨ t.a = t.c

/-- The perimeter of a triangle is the sum of its sides. -/
def Perimeter (t : Triangle) : ℝ :=
  t.a + t.b + t.c

theorem isosceles_base_length
  (equi : Triangle)
  (iso : Triangle)
  (h_equi_equilateral : IsEquilateral equi)
  (h_iso_isosceles : IsIsosceles iso)
  (h_equi_perimeter : Perimeter equi = 60)
  (h_iso_perimeter : Perimeter iso = 70)
  (h_shared_side : equi.a = iso.a) :
  iso.c = 30 :=
sorry

end isosceles_base_length_l836_83696


namespace intersection_implies_sum_l836_83644

-- Define the functions
def f (a b x : ℝ) : ℝ := -|x - a| + b
def g (c d x : ℝ) : ℝ := |x - c| + d

-- State the theorem
theorem intersection_implies_sum (a b c d : ℝ) :
  (f a b 2 = 5) ∧ (f a b 8 = 3) ∧ (g c d 2 = 5) ∧ (g c d 8 = 3) →
  a + c = 10 := by
  sorry

end intersection_implies_sum_l836_83644


namespace f_derivative_at_one_l836_83664

-- Define the function f
def f (x : ℝ) : ℝ := x^2 - 2*x + 3

-- State the theorem
theorem f_derivative_at_one : 
  (deriv f) 1 = 0 := by sorry

end f_derivative_at_one_l836_83664


namespace expression_comparison_l836_83605

theorem expression_comparison (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a ≠ b) : 
  (∃ a b : ℝ, (a + 1/a) * (b + 1/b) > (Real.sqrt (a * b) + 1 / Real.sqrt (a * b))^2) ∧
  (∃ a b : ℝ, (a + 1/a) * (b + 1/b) > ((a + b)/2 + 2/(a + b))^2) ∧
  (∃ a b : ℝ, ((a + b)/2 + 2/(a + b))^2 > (a + 1/a) * (b + 1/b)) :=
by sorry

end expression_comparison_l836_83605


namespace tiffany_phone_pictures_l836_83675

theorem tiffany_phone_pictures :
  ∀ (phone_pics camera_pics total_pics num_albums pics_per_album : ℕ),
    camera_pics = 13 →
    num_albums = 5 →
    pics_per_album = 4 →
    total_pics = num_albums * pics_per_album →
    total_pics = phone_pics + camera_pics →
    phone_pics = 7 := by
  sorry

end tiffany_phone_pictures_l836_83675


namespace equation_solutions_l836_83614

-- Define the equations
def equation1 (x : ℚ) : Prop := (1 - x) / 3 - 2 = x / 6

def equation2 (x : ℚ) : Prop := (x + 1) / (1/4) - (x - 2) / (1/2) = 5

-- State the theorem
theorem equation_solutions :
  (∃ x : ℚ, equation1 x ∧ x = -10/3) ∧
  (∃ x : ℚ, equation2 x ∧ x = -3/2) :=
sorry

end equation_solutions_l836_83614


namespace base_k_conversion_l836_83607

theorem base_k_conversion (k : ℕ) : k > 0 ∧ 1 * k^2 + 3 * k + 2 = 42 ↔ k = 5 := by
  sorry

end base_k_conversion_l836_83607


namespace isosceles_triangle_vertex_angle_l836_83601

-- Define an isosceles triangle
structure IsoscelesTriangle where
  -- We only need to define two angles, as the third is determined by these two
  angle1 : ℝ
  angle2 : ℝ
  -- Condition: sum of angles is 180°
  sum_angles : angle1 + angle2 + (180 - angle1 - angle2) = 180
  -- Condition: at least two angles are equal (isosceles property)
  isosceles : angle1 = angle2 ∨ angle1 = (180 - angle1 - angle2) ∨ angle2 = (180 - angle1 - angle2)

-- Theorem statement
theorem isosceles_triangle_vertex_angle (t : IsoscelesTriangle) (h : t.angle1 = 70 ∨ t.angle2 = 70 ∨ (180 - t.angle1 - t.angle2) = 70) :
  t.angle1 = 70 ∨ t.angle2 = 70 ∨ (180 - t.angle1 - t.angle2) = 70 ∨
  t.angle1 = 40 ∨ t.angle2 = 40 ∨ (180 - t.angle1 - t.angle2) = 40 :=
by sorry

end isosceles_triangle_vertex_angle_l836_83601


namespace vegetarian_eaters_l836_83651

/-- Given a family with the following characteristics:
  - Total number of people: 45
  - Number of people who eat only vegetarian: 22
  - Number of people who eat only non-vegetarian: 15
  - Number of people who eat both vegetarian and non-vegetarian: 8
  Prove that the number of people who eat vegetarian meals is 30. -/
theorem vegetarian_eaters (total : ℕ) (only_veg : ℕ) (only_nonveg : ℕ) (both : ℕ)
  (h1 : total = 45)
  (h2 : only_veg = 22)
  (h3 : only_nonveg = 15)
  (h4 : both = 8) :
  only_veg + both = 30 := by
  sorry

end vegetarian_eaters_l836_83651


namespace alkaline_probability_is_two_fifths_l836_83661

/-- The number of total solutions -/
def total_solutions : ℕ := 5

/-- The number of alkaline solutions -/
def alkaline_solutions : ℕ := 2

/-- The probability of selecting an alkaline solution -/
def alkaline_probability : ℚ := alkaline_solutions / total_solutions

theorem alkaline_probability_is_two_fifths :
  alkaline_probability = 2 / 5 := by sorry

end alkaline_probability_is_two_fifths_l836_83661


namespace courtyard_width_l836_83621

/-- The width of a rectangular courtyard given its length and paving stone requirements -/
theorem courtyard_width (length : Real) (num_stones : Nat) (stone_length stone_width : Real) 
  (h1 : length = 40)
  (h2 : num_stones = 132)
  (h3 : stone_length = 2.5)
  (h4 : stone_width = 2) :
  length * (num_stones * stone_length * stone_width / length) = 16.5 := by
  sorry

#check courtyard_width

end courtyard_width_l836_83621


namespace gcd_lcm_sum_l836_83693

theorem gcd_lcm_sum : Nat.gcd 15 45 + Nat.lcm 15 30 = 45 := by
  sorry

end gcd_lcm_sum_l836_83693


namespace rectangle_dimension_change_l836_83677

theorem rectangle_dimension_change (b h : ℝ) (h_pos : 0 < h) (b_pos : 0 < b) :
  let new_base := 1.1 * b
  let new_height := h * (new_base * h) / (b * h) / new_base
  (h - new_height) / h = 1 / 11 := by sorry

end rectangle_dimension_change_l836_83677


namespace ellipse_properties_l836_83685

/-- An ellipse with semi-major axis a and semi-minor axis b -/
structure Ellipse where
  a : ℝ
  b : ℝ
  h_pos : 0 < b ∧ b < a

/-- A hyperbola formed from an ellipse -/
structure Hyperbola (e : Ellipse) where
  is_equilateral : Bool

/-- A triangle formed by the left focus, right focus, and two points on the ellipse -/
structure Triangle (e : Ellipse) where
  perimeter : ℝ

/-- The eccentricity of an ellipse -/
def eccentricity (e : Ellipse) : ℝ := sorry

/-- The maximum area of a triangle formed by the foci and two points on the ellipse -/
def max_triangle_area (e : Ellipse) (t : Triangle e) : ℝ := sorry

theorem ellipse_properties (e : Ellipse) (h : Hyperbola e) (t : Triangle e) :
  h.is_equilateral = true → t.perimeter = 8 →
    eccentricity e = Real.sqrt 2 / 2 ∧ max_triangle_area e t = 2 * Real.sqrt 2 := by
  sorry

end ellipse_properties_l836_83685


namespace parabola_coefficient_l836_83667

def quadratic_function (a b c : ℤ) (x : ℝ) : ℝ := a * x^2 + b * x + c

theorem parabola_coefficient
  (a b c : ℤ)
  (vertex_x vertex_y : ℝ)
  (point_x point_y : ℝ)
  (h_vertex : ∀ x, quadratic_function a b c x ≥ quadratic_function a b c vertex_x)
  (h_vertex_y : quadratic_function a b c vertex_x = vertex_y)
  (h_point : quadratic_function a b c point_x = point_y)
  (h_vertex_coords : vertex_x = 2 ∧ vertex_y = 3)
  (h_point_coords : point_x = 1 ∧ point_y = 0) :
  a = -3 := by
sorry

end parabola_coefficient_l836_83667


namespace ninth_term_is_negative_256_l836_83643

/-- A geometric sequence with specific properties -/
structure GeometricSequence where
  a : ℕ → ℤ
  is_geometric : ∀ n : ℕ, ∃ q : ℤ, a (n + 1) = a n * q
  a2a5 : a 2 * a 5 = -32
  a3a4_sum : a 3 + a 4 = 4

/-- The 9th term of the geometric sequence is -256 -/
theorem ninth_term_is_negative_256 (seq : GeometricSequence) : seq.a 9 = -256 := by
  sorry

#check ninth_term_is_negative_256

end ninth_term_is_negative_256_l836_83643


namespace complex_fraction_sum_l836_83669

theorem complex_fraction_sum : (1 - Complex.I) / (1 + Complex.I)^2 + (1 + Complex.I) / (1 - Complex.I)^2 = -1 := by
  sorry

end complex_fraction_sum_l836_83669


namespace smallest_three_digit_multiple_of_13_l836_83671

theorem smallest_three_digit_multiple_of_13 : 
  ∀ n : ℕ, 100 ≤ n ∧ n < 1000 ∧ 13 ∣ n → 104 ≤ n :=
by sorry

end smallest_three_digit_multiple_of_13_l836_83671


namespace three_digit_numbers_satisfying_condition_l836_83648

def is_valid_number (n : ℕ) : Prop :=
  100 ≤ n ∧ n < 1000

def satisfies_condition (n : ℕ) : Prop :=
  let a := n / 100
  let b := (n / 10) % 10
  let c := n % 10
  n = 2 * ((10 * a + b) + (10 * b + c) + (10 * a + c))

def solution_set : Set ℕ := {134, 144, 150, 288, 294}

theorem three_digit_numbers_satisfying_condition :
  ∀ n : ℕ, is_valid_number n ∧ satisfies_condition n ↔ n ∈ solution_set :=
sorry

end three_digit_numbers_satisfying_condition_l836_83648


namespace ball_distribution_proof_l836_83633

def choose (n k : ℕ) : ℕ := (Nat.factorial n) / (Nat.factorial k * Nat.factorial (n - k))

theorem ball_distribution_proof (total_balls : ℕ) (box1_num box2_num : ℕ) : 
  total_balls = 7 ∧ box1_num = 2 ∧ box2_num = 3 →
  (choose total_balls box1_num.succ.succ) + 
  (choose total_balls box1_num.succ) + 
  (choose total_balls box1_num) = 91 := by
sorry

end ball_distribution_proof_l836_83633


namespace greatest_three_digit_multiple_of_23_l836_83692

theorem greatest_three_digit_multiple_of_23 : ∀ n : ℕ, n ≤ 999 → n ≥ 100 → n % 23 = 0 → n ≤ 991 :=
by
  sorry

end greatest_three_digit_multiple_of_23_l836_83692


namespace lowest_degree_polynomial_l836_83635

/-- A polynomial with coefficients in ℤ -/
def IntPolynomial := ℕ → ℤ

/-- The degree of a polynomial -/
def degree (p : IntPolynomial) : ℕ := sorry

/-- The set of coefficients of a polynomial -/
def coefficients (p : IntPolynomial) : Set ℤ := sorry

/-- Predicate for a polynomial satisfying the given conditions -/
def satisfies_conditions (p : IntPolynomial) : Prop :=
  ∃ b : ℤ, (∃ x ∈ coefficients p, x < b) ∧
            (∃ y ∈ coefficients p, y > b) ∧
            b ∉ coefficients p

/-- The main theorem -/
theorem lowest_degree_polynomial :
  ∃ p : IntPolynomial, satisfies_conditions p ∧
    degree p = 4 ∧
    ∀ q : IntPolynomial, satisfies_conditions q → degree q ≥ 4 := by
  sorry

end lowest_degree_polynomial_l836_83635
