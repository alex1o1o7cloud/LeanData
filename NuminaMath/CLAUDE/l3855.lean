import Mathlib

namespace helga_shoe_shopping_l3855_385549

theorem helga_shoe_shopping (x : ℕ) : 
  (x + (x + 2) + 0 + 2 * (x + (x + 2) + 0) = 48) → x = 7 := by
  sorry

end helga_shoe_shopping_l3855_385549


namespace min_quotient_l3855_385557

/-- A three-digit number with distinct non-zero digits that sum to 10 -/
structure ThreeDigitNumber where
  a : ℕ
  b : ℕ
  c : ℕ
  h1 : a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0
  h2 : a ≠ b ∧ b ≠ c ∧ a ≠ c
  h3 : a + b + c = 10

/-- The quotient of the number divided by the sum of its digits -/
def quotient (n : ThreeDigitNumber) : ℚ :=
  (100 * n.a + 10 * n.b + n.c) / (n.a + n.b + n.c)

/-- The minimum value of the quotient is 12.7 -/
theorem min_quotient :
  ∀ n : ThreeDigitNumber, quotient n ≥ 127/10 ∧ ∃ m : ThreeDigitNumber, quotient m = 127/10 :=
sorry

end min_quotient_l3855_385557


namespace last_digit_3_2004_l3855_385567

/-- The last digit of 3^n -/
def last_digit (n : ℕ) : ℕ :=
  (3^n) % 10

/-- The pattern of last digits repeats every 4 steps -/
axiom last_digit_pattern (n : ℕ) :
  last_digit n = last_digit (n % 4)

/-- The last digits for the first 4 powers of 3 -/
axiom last_digit_base :
  last_digit 0 = 1 ∧ 
  last_digit 1 = 3 ∧ 
  last_digit 2 = 9 ∧ 
  last_digit 3 = 7

theorem last_digit_3_2004 :
  last_digit 2004 = 1 :=
by sorry

end last_digit_3_2004_l3855_385567


namespace digit_sum_property_l3855_385581

def S (k : ℕ) : ℕ := (k.repr.toList.map (λ c => c.toNat - 48)).sum

theorem digit_sum_property (n : ℕ) : 
  (∃ (a b : ℕ), n = S a ∧ n = S b ∧ n = S (a + b)) ↔ 
  (∃ (k : ℕ+), n = 9 * k) :=
sorry

end digit_sum_property_l3855_385581


namespace set_union_problem_l3855_385598

-- Define the sets A and B
def A (a : ℝ) : Set ℝ := {-1, a}
def B (a b : ℝ) : Set ℝ := {2^a, b}

-- Theorem statement
theorem set_union_problem (a b : ℝ) : 
  A a ∩ B a b = {1} → A a ∪ B a b = {-1, 1, 2} := by
  sorry

end set_union_problem_l3855_385598


namespace lcm_equality_pairs_l3855_385584

theorem lcm_equality_pairs (m n : ℕ) : 
  Nat.lcm m n = 3 * m + 2 * n + 1 ↔ (m = 3 ∧ n = 10) ∨ (m = 9 ∧ n = 4) := by
  sorry

end lcm_equality_pairs_l3855_385584


namespace min_cuts_for_daily_payment_min_cuts_for_all_lengths_l3855_385579

/-- Represents a chain of links -/
structure Chain where
  length : ℕ

/-- Represents a cut strategy for a chain -/
structure CutStrategy where
  cuts : ℕ

/-- Checks if a cut strategy is valid for daily payments -/
def is_valid_daily_payment_strategy (chain : Chain) (strategy : CutStrategy) : Prop :=
  ∀ day : ℕ, day ≤ chain.length → ∃ payment : ℕ, payment = day

/-- Checks if a cut strategy can produce any number of links up to the chain length -/
def can_produce_all_lengths (chain : Chain) (strategy : CutStrategy) : Prop :=
  ∀ n : ℕ, n ≤ chain.length → ∃ combination : List ℕ, combination.sum = n

/-- Theorem for the minimum cuts needed for daily payments -/
theorem min_cuts_for_daily_payment (chain : Chain) (strategy : CutStrategy) : 
  chain.length = 7 → 
  strategy.cuts = 1 → 
  is_valid_daily_payment_strategy chain strategy :=
sorry

/-- Theorem for the minimum cuts needed to produce all lengths -/
theorem min_cuts_for_all_lengths (chain : Chain) (strategy : CutStrategy) :
  chain.length = 2000 →
  strategy.cuts = 7 →
  can_produce_all_lengths chain strategy :=
sorry

end min_cuts_for_daily_payment_min_cuts_for_all_lengths_l3855_385579


namespace square_root_difference_squared_l3855_385517

theorem square_root_difference_squared : 
  (Real.sqrt (16 - 8 * Real.sqrt 3) - Real.sqrt (16 + 8 * Real.sqrt 3))^2 = 48 := by
  sorry

end square_root_difference_squared_l3855_385517


namespace students_in_class_l3855_385514

/-- Proves that the number of students in Ms. Leech's class is 30 -/
theorem students_in_class (num_boys : ℕ) (num_girls : ℕ) (cups_per_boy : ℕ) (total_cups : ℕ) :
  num_boys = 10 →
  num_girls = 2 * num_boys →
  cups_per_boy = 5 →
  total_cups = 90 →
  num_boys * cups_per_boy + num_girls * ((total_cups - num_boys * cups_per_boy) / num_girls) = total_cups →
  num_boys + num_girls = 30 := by
  sorry

end students_in_class_l3855_385514


namespace counterexample_exists_l3855_385595

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d > 1 → d < n → ¬(n % d = 0)

def given_numbers : List ℕ := [6, 9, 10, 11, 15]

theorem counterexample_exists : ∃ n : ℕ, 
  n ∈ given_numbers ∧
  ¬(is_prime n) ∧ 
  is_prime (n - 2) ∧ 
  is_prime (n + 2) :=
sorry

end counterexample_exists_l3855_385595


namespace combinatorial_equation_l3855_385519

theorem combinatorial_equation (n : ℕ) : (Nat.choose (n + 1) (n - 1) = 28) → n = 7 := by
  sorry

end combinatorial_equation_l3855_385519


namespace grace_age_l3855_385580

/-- Represents the ages of the people in the problem -/
structure Ages where
  grace : ℕ
  faye : ℕ
  chad : ℕ
  eduardo : ℕ
  diana : ℕ

/-- Defines the age relationships between the people -/
def valid_ages (a : Ages) : Prop :=
  a.faye = a.grace + 6 ∧
  a.faye = a.chad + 2 ∧
  a.eduardo = a.chad + 3 ∧
  a.eduardo = a.diana + 4 ∧
  a.diana = 17

/-- Theorem stating that if the ages are valid, Grace's age is 14 -/
theorem grace_age (a : Ages) : valid_ages a → a.grace = 14 := by
  sorry

end grace_age_l3855_385580


namespace rearrange_three_of_eight_count_l3855_385536

/-- The number of ways to select and rearrange 3 people out of 8 -/
def rearrange_three_of_eight : ℕ :=
  Nat.choose 8 3 * (3 * 2)

/-- Theorem stating that rearranging 3 people out of 8 has C₈₃ * A³₂ ways -/
theorem rearrange_three_of_eight_count :
  rearrange_three_of_eight = Nat.choose 8 3 * (3 * 2) := by
  sorry

end rearrange_three_of_eight_count_l3855_385536


namespace optimal_distribution_maximizes_sum_l3855_385507

/-- Represents the distribution of blue balls between two boxes -/
structure Distribution where
  first_box : ℕ
  second_box : ℕ

/-- Calculates the sum of percentages of blue balls in each box -/
def sum_of_percentages (d : Distribution) : ℚ :=
  d.first_box / 24 + d.second_box / 23

/-- Checks if a distribution is valid given the total number of blue balls -/
def is_valid_distribution (d : Distribution) (total_blue : ℕ) : Prop :=
  d.first_box + d.second_box = total_blue ∧ d.first_box ≤ 24 ∧ d.second_box ≤ 23

theorem optimal_distribution_maximizes_sum :
  ∀ d : Distribution,
  is_valid_distribution d 25 →
  sum_of_percentages d ≤ sum_of_percentages { first_box := 2, second_box := 23 } :=
by sorry

end optimal_distribution_maximizes_sum_l3855_385507


namespace batsman_average_l3855_385582

theorem batsman_average (previous_total : ℕ) (previous_average : ℚ) : 
  previous_total = 10 * previous_average →
  (previous_total + 69) / 11 = previous_average + 1 →
  (previous_total + 69) / 11 = 59 := by
sorry

end batsman_average_l3855_385582


namespace imaginary_part_of_reciprocal_l3855_385588

theorem imaginary_part_of_reciprocal (a : ℝ) (z : ℂ) : 
  z = a^2 - 1 + (a + 1) * Complex.I → 
  z.re = 0 → 
  z ≠ 0 → 
  (Complex.I * ((z + a)⁻¹)).re = -2/5 :=
sorry

end imaginary_part_of_reciprocal_l3855_385588


namespace complex_magnitude_two_thirds_minus_four_fifths_i_l3855_385593

theorem complex_magnitude_two_thirds_minus_four_fifths_i :
  Complex.abs (2/3 - 4/5 * Complex.I) = Real.sqrt 244 / 15 := by
  sorry

end complex_magnitude_two_thirds_minus_four_fifths_i_l3855_385593


namespace free_throw_contest_total_l3855_385537

/-- Given a free throw contest where:
  * Alex made 8 baskets
  * Sandra made three times as many baskets as Alex
  * Hector made two times the number of baskets that Sandra made
  Prove that the total number of baskets made by all three is 80. -/
theorem free_throw_contest_total (alex_baskets : ℕ) (sandra_baskets : ℕ) (hector_baskets : ℕ) 
  (h1 : alex_baskets = 8)
  (h2 : sandra_baskets = 3 * alex_baskets)
  (h3 : hector_baskets = 2 * sandra_baskets) :
  alex_baskets + sandra_baskets + hector_baskets = 80 := by
  sorry

#check free_throw_contest_total

end free_throw_contest_total_l3855_385537


namespace no_natural_solutions_l3855_385505

theorem no_natural_solutions (x y : ℕ) : 
  (1 : ℚ) / (x^2 : ℚ) + (1 : ℚ) / ((x * y) : ℚ) + (1 : ℚ) / (y^2 : ℚ) ≠ 1 := by
  sorry

end no_natural_solutions_l3855_385505


namespace complex_fraction_squared_difference_l3855_385559

theorem complex_fraction_squared_difference (a b : ℝ) :
  (Complex.I : ℂ)^2 = -1 →
  (1 - Complex.I) / (1 + Complex.I) = a + b * Complex.I →
  a^2 - b^2 = -1 := by
  sorry

end complex_fraction_squared_difference_l3855_385559


namespace power_simplification_l3855_385546

theorem power_simplification : 10^6 * (10^2)^3 / 10^4 = 10^8 := by
  sorry

end power_simplification_l3855_385546


namespace system_solution_l3855_385568

theorem system_solution (x y k : ℝ) : 
  (2 * x + 3 * y = k) → 
  (x + 4 * y = k - 16) → 
  (x + y = 8) → 
  k = 12 := by
sorry

end system_solution_l3855_385568


namespace shirt_price_reduction_l3855_385528

/-- Represents the price reduction problem for a shopping mall selling shirts. -/
theorem shirt_price_reduction
  (initial_sales : ℕ)
  (initial_profit : ℝ)
  (price_reduction_effect : ℝ → ℕ)
  (target_profit : ℝ)
  (price_reduction : ℝ)
  (h1 : initial_sales = 20)
  (h2 : initial_profit = 40)
  (h3 : ∀ x, price_reduction_effect x = initial_sales + 2 * ⌊x⌋)
  (h4 : target_profit = 1200)
  (h5 : price_reduction = 20) :
  (initial_profit - price_reduction) * price_reduction_effect price_reduction = target_profit :=
sorry

end shirt_price_reduction_l3855_385528


namespace polynomial_simplification_l3855_385550

theorem polynomial_simplification (x y : ℝ) :
  (10 * x^12 + 8 * x^9 + 5 * x^7) + (11 * x^9 + 3 * x^7 + 4 * x^3 + 6 * y^2 + 7 * x + 9) =
  10 * x^12 + 19 * x^9 + 8 * x^7 + 4 * x^3 + 6 * y^2 + 7 * x + 9 := by
  sorry

end polynomial_simplification_l3855_385550


namespace jorge_total_goals_l3855_385594

/-- The total number of goals Jorge scored over two seasons -/
def total_goals (last_season_goals this_season_goals : ℕ) : ℕ :=
  last_season_goals + this_season_goals

/-- Theorem stating that Jorge's total goals over two seasons is 343 -/
theorem jorge_total_goals :
  total_goals 156 187 = 343 := by
  sorry

end jorge_total_goals_l3855_385594


namespace binomial_60_3_l3855_385521

theorem binomial_60_3 : Nat.choose 60 3 = 34220 := by
  sorry

end binomial_60_3_l3855_385521


namespace onion_harvest_weight_l3855_385574

/-- The total weight of onions harvested by Titan's father -/
def total_weight (bags_per_trip : ℕ) (weight_per_bag : ℕ) (num_trips : ℕ) : ℕ :=
  bags_per_trip * weight_per_bag * num_trips

/-- Theorem stating the total weight of onions harvested -/
theorem onion_harvest_weight :
  total_weight 10 50 20 = 10000 := by
  sorry

#eval total_weight 10 50 20

end onion_harvest_weight_l3855_385574


namespace roots_of_g_l3855_385524

/-- Given that 2 is a root of f(x) = ax + b, prove that the roots of g(x) = bx² - ax are 0 and -1/2 --/
theorem roots_of_g (a b : ℝ) (h : a * 2 + b = 0) :
  ∃ (x y : ℝ), x = 0 ∧ y = -1/2 ∧ ∀ z : ℝ, b * z^2 - a * z = 0 ↔ z = x ∨ z = y :=
by sorry

end roots_of_g_l3855_385524


namespace candy_bar_chocolate_cost_difference_l3855_385547

/-- The problem of calculating the difference in cost between a candy bar and chocolate. -/
theorem candy_bar_chocolate_cost_difference :
  let dans_money : ℕ := 2
  let candy_bar_cost : ℕ := 6
  let chocolate_cost : ℕ := 3
  candy_bar_cost - chocolate_cost = 3 :=
by sorry

end candy_bar_chocolate_cost_difference_l3855_385547


namespace simple_interest_rate_l3855_385555

/-- Given a principal amount and a time period of 10 years, 
    if the simple interest is 7/5 of the principal, 
    then the annual interest rate is 14%. -/
theorem simple_interest_rate (P : ℝ) (P_pos : P > 0) : 
  (P * 14 * 10) / 100 = (7 / 5) * P := by sorry

end simple_interest_rate_l3855_385555


namespace mets_fan_count_l3855_385578

/-- Represents the number of fans for each team -/
structure FanCounts where
  yankees : ℕ
  mets : ℕ
  redsox : ℕ

/-- The conditions of the problem -/
def fan_conditions (f : FanCounts) : Prop :=
  (f.yankees : ℚ) / f.mets = 3 / 2 ∧
  (f.mets : ℚ) / f.redsox = 4 / 5 ∧
  f.yankees + f.mets + f.redsox = 330

/-- The theorem to be proved -/
theorem mets_fan_count (f : FanCounts) :
  fan_conditions f → f.mets = 88 := by
  sorry

end mets_fan_count_l3855_385578


namespace square_construction_l3855_385531

/-- A point in a 2D plane -/
structure Point2D where
  x : ℝ
  y : ℝ

/-- A line in a 2D plane defined by two points -/
structure Line2D where
  p1 : Point2D
  p2 : Point2D

/-- A square in a 2D plane -/
structure Square where
  vertices : Fin 4 → Point2D

/-- Check if a point lies on a line -/
def pointOnLine (p : Point2D) (l : Line2D) : Prop := sorry

/-- Check if two lines are perpendicular -/
def perpendicular (l1 l2 : Line2D) : Prop := sorry

/-- Check if all sides of a square are equal length -/
def equalSides (s : Square) : Prop := sorry

/-- The main theorem -/
theorem square_construction (A B C D : Point2D) :
  ∃ (s : Square),
    (∀ i : Fin 4, ∃ p ∈ [A, B, C, D], pointOnLine (s.vertices i) (Line2D.mk (s.vertices i) (s.vertices ((i + 1) % 4)))) ∧
    (∀ i : Fin 4, perpendicular (Line2D.mk (s.vertices i) (s.vertices ((i + 1) % 4))) (Line2D.mk (s.vertices ((i + 1) % 4)) (s.vertices ((i + 2) % 4)))) ∧
    equalSides s :=
sorry

end square_construction_l3855_385531


namespace sin_theta_value_l3855_385522

theorem sin_theta_value (θ : Real) (h1 : 0 < θ) (h2 : θ < Real.pi / 2) 
  (h3 : 1 + Real.sin θ = 2 * Real.cos θ) : Real.sin θ = 3/5 := by
  sorry

end sin_theta_value_l3855_385522


namespace tangent_line_implies_a_and_b_l3855_385518

/-- Given a function f(x) = x^3 - 3ax^2 + b, prove that if the curve y = f(x) is tangent
    to the line y = 8 at the point (2, f(2)), then a = 1 and b = 12. -/
theorem tangent_line_implies_a_and_b (a b : ℝ) : 
  let f : ℝ → ℝ := λ x ↦ x^3 - 3*a*x^2 + b
  (f 2 = 8) ∧ (deriv f 2 = 0) → a = 1 ∧ b = 12 := by
  sorry

#check tangent_line_implies_a_and_b

end tangent_line_implies_a_and_b_l3855_385518


namespace intersection_and_subset_l3855_385500

def A : Set ℝ := {x | 1 ≤ x ∧ x ≤ 3}
def B : Set ℝ := {x | Real.sqrt (x - 1) ≥ 1}

theorem intersection_and_subset : 
  (A ∩ B = {x | 2 ≤ x ∧ x ≤ 3}) ∧
  (∀ a : ℝ, (A ∩ B) ⊆ {x | x ≥ a} ↔ a ≤ 2) := by
  sorry

end intersection_and_subset_l3855_385500


namespace inequality_solution_and_bound_l3855_385534

def f (x : ℝ) := |x - 3|
def g (x : ℝ) := |x - 2|

theorem inequality_solution_and_bound :
  (∀ x, f x + g x < 2 ↔ x ∈ Set.Ioo (3/2) (7/2)) ∧
  (∀ x y, f x ≤ 1 → g y ≤ 1 → |x - 2*y + 1| ≤ 3) := by sorry

end inequality_solution_and_bound_l3855_385534


namespace cauchy_schwarz_2d_l3855_385592

theorem cauchy_schwarz_2d {a b c d : ℝ} :
  (a^2 + b^2) * (c^2 + d^2) ≥ (a*c + b*d)^2 ∧
  ((a^2 + b^2) * (c^2 + d^2) = (a*c + b*d)^2 ↔ a*d = b*c) :=
sorry

end cauchy_schwarz_2d_l3855_385592


namespace external_tangent_y_intercept_l3855_385525

/-- Given two circles with centers and radii as specified, 
    prove that their common external tangent with positive slope 
    has a y-intercept of 740/171 -/
theorem external_tangent_y_intercept :
  let c1 : ℝ × ℝ := (1, 3)  -- Center of circle 1
  let r1 : ℝ := 3           -- Radius of circle 1
  let c2 : ℝ × ℝ := (15, 8) -- Center of circle 2
  let r2 : ℝ := 10          -- Radius of circle 2
  let m : ℝ := (140 : ℝ) / 171 -- Slope of the tangent line (positive)
  let b : ℝ := (740 : ℝ) / 171 -- y-intercept to be proved
  let tangent_line (x : ℝ) := m * x + b -- Equation of the tangent line
  (∀ x y : ℝ, (x - c1.1)^2 + (y - c1.2)^2 = r1^2 → 
    (tangent_line x - y)^2 ≥ (r1 * m)^2) ∧ 
  (∀ x y : ℝ, (x - c2.1)^2 + (y - c2.2)^2 = r2^2 → 
    (tangent_line x - y)^2 ≥ (r2 * m)^2) ∧
  (∃ x1 y1 x2 y2 : ℝ, 
    (x1 - c1.1)^2 + (y1 - c1.2)^2 = r1^2 ∧
    (x2 - c2.1)^2 + (y2 - c2.2)^2 = r2^2 ∧
    tangent_line x1 = y1 ∧ tangent_line x2 = y2) :=
by sorry

end external_tangent_y_intercept_l3855_385525


namespace edge_probability_is_one_l3855_385573

/-- Represents a position on the 4x4 grid -/
structure Position :=
  (x : Fin 4)
  (y : Fin 4)

/-- Checks if a position is on the edge of the grid -/
def isEdge (p : Position) : Bool :=
  p.x = 0 || p.x = 3 || p.y = 0 || p.y = 3

/-- Represents a single hop direction -/
inductive Direction
  | Up
  | Down
  | Left
  | Right

/-- Applies a hop in the given direction, staying in bounds -/
def hop (p : Position) (d : Direction) : Position :=
  match d with
  | Direction.Up => ⟨min 3 (p.x + 1), p.y⟩
  | Direction.Down => ⟨max 0 (p.x - 1), p.y⟩
  | Direction.Left => ⟨p.x, max 0 (p.y - 1)⟩
  | Direction.Right => ⟨p.x, min 3 (p.y + 1)⟩

/-- The starting position (2,2) -/
def startPos : Position := ⟨2, 2⟩

/-- Theorem: The probability of reaching an edge cell within three hops from (2,2) is 1 -/
theorem edge_probability_is_one :
  ∀ (hops : List Direction),
    hops.length ≤ 3 →
    isEdge (hops.foldl hop startPos) = true :=
by sorry

end edge_probability_is_one_l3855_385573


namespace intersection_of_A_and_B_l3855_385553

def A : Set ℝ := {x : ℝ | |x| ≤ 1}
def B : Set ℝ := {x : ℝ | x ≤ 0}

theorem intersection_of_A_and_B : A ∩ B = {x : ℝ | -1 ≤ x ∧ x ≤ 0} := by sorry

end intersection_of_A_and_B_l3855_385553


namespace geometric_sequence_properties_l3855_385587

def geometric_sequence (a : ℚ) (r : ℚ) (n : ℕ) : ℚ := a * r ^ (n - 1)

def geometric_sum (a : ℚ) (r : ℚ) (n : ℕ) : ℚ := a * (1 - r^n) / (1 - r)

theorem geometric_sequence_properties :
  let a : ℚ := 1/5
  let r : ℚ := 1/2
  let n : ℕ := 8
  (geometric_sequence a r n = 1/640) ∧
  (geometric_sum a r n = 255/320) := by sorry

end geometric_sequence_properties_l3855_385587


namespace arithmetic_sequence_problem_l3855_385589

theorem arithmetic_sequence_problem :
  ∀ (a b c d : ℝ),
    (a + b + c + d = 26) →
    (b * c = 40) →
    (c - b = b - a) →
    (d - c = c - b) →
    ((a = 2 ∧ b = 5 ∧ c = 8 ∧ d = 11) ∨ (a = 11 ∧ b = 8 ∧ c = 5 ∧ d = 2)) :=
by
  sorry

end arithmetic_sequence_problem_l3855_385589


namespace cost_of_one_sandwich_and_juice_l3855_385520

/-- Given the cost of multiple items, calculate the cost of one item and one juice -/
theorem cost_of_one_sandwich_and_juice 
  (juice_cost : ℝ) 
  (juice_count : ℕ) 
  (sandwich_cost : ℝ) 
  (sandwich_count : ℕ) : 
  juice_cost / juice_count + sandwich_cost / sandwich_count = 5 :=
by
  sorry

#check cost_of_one_sandwich_and_juice 10 5 6 2

end cost_of_one_sandwich_and_juice_l3855_385520


namespace simplify_fraction_l3855_385576

theorem simplify_fraction (x : ℝ) (h : x ≠ 1) :
  (x^2 + 1) / (x - 1) - 2 * x / (x - 1) = x - 1 := by
  sorry

end simplify_fraction_l3855_385576


namespace ten_row_triangle_pieces_l3855_385516

/-- Calculates the number of rods in a triangle with given number of rows -/
def num_rods (n : ℕ) : ℕ := n * (n + 1) * 3

/-- Calculates the number of connectors in a triangle with given number of rows -/
def num_connectors (n : ℕ) : ℕ := (n + 1) * (n + 2) / 2

/-- The total number of pieces in a triangle with given number of rows -/
def total_pieces (n : ℕ) : ℕ := num_rods n + num_connectors n

theorem ten_row_triangle_pieces :
  total_pieces 10 = 366 ∧
  num_rods 3 = 18 ∧
  num_connectors 3 = 10 := by
  sorry

end ten_row_triangle_pieces_l3855_385516


namespace sequence_proof_l3855_385572

def arithmetic_sequence (a : ℕ → ℤ) : Prop :=
  ∃ d : ℤ, ∀ n : ℕ, a (n + 1) - a n = d

theorem sequence_proof
  (a : ℕ → ℤ)
  (b : ℕ → ℤ)
  (h_arith : arithmetic_sequence a)
  (h_a2 : a 2 = -1)
  (h_b_diff : ∀ n : ℕ, n ≥ 2 → b n - b (n - 1) = a n)
  (h_b1 : b 1 = 1)
  (h_b3 : b 3 = 1) :
  (a 1 = -3) ∧
  (∀ n : ℕ, n ≥ 1 → b n = n^2 - 4*n + 4) :=
by sorry

end sequence_proof_l3855_385572


namespace stating_work_completion_time_l3855_385535

/-- The time it takes for a man and his son to complete a piece of work together -/
def combined_time : ℝ := 6

/-- The time it takes for the son to complete the work alone -/
def son_time : ℝ := 10

/-- The time it takes for the man to complete the work alone -/
def man_time : ℝ := 15

/-- 
Theorem stating that if a man and his son can complete a piece of work together in 6 days, 
and the son can complete the work alone in 10 days, then the man can complete the work 
alone in 15 days.
-/
theorem work_completion_time : 
  (1 / combined_time) = (1 / man_time) + (1 / son_time) :=
sorry

end stating_work_completion_time_l3855_385535


namespace binary_11011_is_27_l3855_385552

def binary_to_decimal (b : List Bool) : ℕ :=
  (List.enum b).foldl (fun acc (i, bit) => acc + if bit then 2^i else 0) 0

theorem binary_11011_is_27 :
  binary_to_decimal [true, true, false, true, true] = 27 := by
  sorry

end binary_11011_is_27_l3855_385552


namespace square_cutting_existence_l3855_385506

theorem square_cutting_existence : ∃ (a b c S : ℝ), 
  a^2 + 3*b^2 + 5*c^2 = S^2 ∧ 
  a ≠ b ∧ a ≠ c ∧ b ≠ c ∧
  a > 0 ∧ b > 0 ∧ c > 0 ∧ S > 0 := by
  sorry

end square_cutting_existence_l3855_385506


namespace problem_statement_l3855_385575

def p (x : ℝ) : Prop := x^2 - 4*x - 5 ≤ 0

def q (x m : ℝ) : Prop := x^2 - 2*x + 1 - m^2 ≤ 0

theorem problem_statement (m : ℝ) (h : m > 0) :
  (∀ x, p x → q x m) → m ∈ Set.Ici 4 ∧
  (m = 5 → ∀ x, (p x ∨ q x m) ∧ ¬(p x ∧ q x m) → 
    x ∈ Set.Ioc (-4) (-1) ∪ Set.Ioo 5 6) :=
by sorry

end problem_statement_l3855_385575


namespace successive_numbers_product_l3855_385570

theorem successive_numbers_product (n : ℕ) : 
  n > 0 ∧ n * (n + 1) = 4160 → n = 64 := by
  sorry

end successive_numbers_product_l3855_385570


namespace rowing_time_ratio_l3855_385533

/-- Given Ethan's rowing time and the total rowing time for both Ethan and Frank,
    prove that the ratio of Frank's rowing time to Ethan's rowing time is 2:1. -/
theorem rowing_time_ratio 
  (ethan_time : ℕ) 
  (total_time : ℕ) 
  (h1 : ethan_time = 25)
  (h2 : total_time = 75) :
  (total_time - ethan_time) / ethan_time = 2 := by
  sorry

end rowing_time_ratio_l3855_385533


namespace author_earnings_l3855_385562

theorem author_earnings (paper_cover_percentage : ℝ) (hardcover_percentage : ℝ)
  (paper_cover_copies : ℕ) (hardcover_copies : ℕ)
  (paper_cover_price : ℝ) (hardcover_price : ℝ) :
  paper_cover_percentage = 0.06 →
  hardcover_percentage = 0.12 →
  paper_cover_copies = 32000 →
  hardcover_copies = 15000 →
  paper_cover_price = 0.20 →
  hardcover_price = 0.40 →
  (paper_cover_percentage * (paper_cover_copies : ℝ) * paper_cover_price) +
  (hardcover_percentage * (hardcover_copies : ℝ) * hardcover_price) = 1104 :=
by sorry

end author_earnings_l3855_385562


namespace root_sum_product_l3855_385558

theorem root_sum_product (p q : ℝ) : 
  (∃ x, x^4 - 6*x - 2 = 0) → 
  (p^4 - 6*p - 2 = 0) →
  (q^4 - 6*q - 2 = 0) →
  p ≠ q →
  pq + p + q = 1 - 2 * Real.sqrt 2 := by
sorry

end root_sum_product_l3855_385558


namespace diagonals_30_sided_polygon_l3855_385543

/-- The number of diagonals in a convex polygon with n sides -/
def num_diagonals (n : ℕ) : ℕ := n * (n - 3) / 2

/-- Theorem: A convex polygon with 30 sides has 405 diagonals -/
theorem diagonals_30_sided_polygon : num_diagonals 30 = 405 := by
  sorry

end diagonals_30_sided_polygon_l3855_385543


namespace money_ratio_proof_l3855_385585

def josh_money (doug_money : ℚ) : ℚ := (3 / 4) * doug_money

theorem money_ratio_proof (doug_money : ℚ) 
  (h1 : josh_money doug_money + doug_money + 12 = 68) : 
  josh_money doug_money / 12 = 2 := by
  sorry

end money_ratio_proof_l3855_385585


namespace fraction_of_powers_equals_five_fourths_l3855_385554

theorem fraction_of_powers_equals_five_fourths :
  (3^10 + 3^8) / (3^10 - 3^8) = 5/4 := by
  sorry

end fraction_of_powers_equals_five_fourths_l3855_385554


namespace largest_last_digit_l3855_385545

def is_valid_digit_string (s : List Nat) : Prop :=
  s.length = 1001 ∧ 
  s.head? = some 3 ∧
  ∀ i, i < 1000 → (s[i]! * 10 + s[i+1]!) % 17 = 0 ∨ (s[i]! * 10 + s[i+1]!) % 23 = 0

theorem largest_last_digit (s : List Nat) (h : is_valid_digit_string s) : 
  s[1000]! ≤ 2 :=
sorry

end largest_last_digit_l3855_385545


namespace sqrt_sum_comparison_l3855_385566

theorem sqrt_sum_comparison : Real.sqrt 11 + Real.sqrt 7 > Real.sqrt 13 + Real.sqrt 5 := by
  sorry

end sqrt_sum_comparison_l3855_385566


namespace parallel_lines_distance_l3855_385527

/-- The distance between two parallel lines -/
theorem parallel_lines_distance (a b c d e f : ℝ) :
  (a = 1 ∧ b = 2 ∧ c = -1) →
  (d = 2 ∧ e = 4 ∧ f = 3) →
  (∃ (k : ℝ), k ≠ 0 ∧ d = k * a ∧ e = k * b) →
  (abs (f / d - c / a) / Real.sqrt (a^2 + b^2) : ℝ) = Real.sqrt 5 / 2 := by
  sorry

end parallel_lines_distance_l3855_385527


namespace oranges_thrown_away_l3855_385540

theorem oranges_thrown_away (initial : ℕ) (new : ℕ) (final : ℕ) : 
  initial = 34 → new = 13 → final = 27 → initial - (initial - final + new) = 20 := by
  sorry

end oranges_thrown_away_l3855_385540


namespace two_points_on_curve_l3855_385565

def point_on_curve (x y : ℝ) : Prop :=
  x^2 - x*y + 2*y + 1 = 0

def point_A : ℝ × ℝ := (1, -2)
def point_B : ℝ × ℝ := (2, -3)
def point_C : ℝ × ℝ := (3, 10)

theorem two_points_on_curve :
  (point_on_curve point_A.1 point_A.2 ∧
   point_on_curve point_C.1 point_C.2 ∧
   ¬point_on_curve point_B.1 point_B.2) ∨
  (point_on_curve point_A.1 point_A.2 ∧
   point_on_curve point_B.1 point_B.2 ∧
   ¬point_on_curve point_C.1 point_C.2) ∨
  (point_on_curve point_B.1 point_B.2 ∧
   point_on_curve point_C.1 point_C.2 ∧
   ¬point_on_curve point_A.1 point_A.2) :=
sorry

end two_points_on_curve_l3855_385565


namespace inequality_solution_l3855_385596

theorem inequality_solution (x : ℝ) : 
  (x + 3) / (x + 4) > (2 * x + 7) / (3 * x + 12) ↔ x < -4 ∨ x > -4 := by
  sorry

end inequality_solution_l3855_385596


namespace equation_equivalence_l3855_385586

theorem equation_equivalence (x : ℝ) : (5 = 3 * x - 2) ↔ (5 + 2 = 3 * x) := by
  sorry

end equation_equivalence_l3855_385586


namespace difference_of_squares_l3855_385539

theorem difference_of_squares : (23 + 15)^2 - (23 - 15)^2 = 1380 := by
  sorry

end difference_of_squares_l3855_385539


namespace negative_fraction_range_l3855_385532

theorem negative_fraction_range (x : ℝ) : (x - 1) / x^2 < 0 → x < 1 ∧ x ≠ 0 := by
  sorry

end negative_fraction_range_l3855_385532


namespace rectangular_plot_ratio_l3855_385526

/-- Proves that for a rectangular plot with given area and breadth, the ratio of length to breadth is 3:1 -/
theorem rectangular_plot_ratio (area : ℝ) (breadth : ℝ) (length : ℝ) : 
  area = 972 →
  breadth = 18 →
  area = length * breadth →
  length / breadth = 3 := by
  sorry

end rectangular_plot_ratio_l3855_385526


namespace right_triangle_area_l3855_385597

/-- 
  Given a right-angled triangle with legs x and y, and hypotenuse z,
  where x:y = 3:4 and x^2 + y^2 = z^2, prove that the area A of the triangle
  is equal to (2/3)x^2 or (6/25)z^2.
-/
theorem right_triangle_area (x y z : ℝ) (h1 : x > 0) (h2 : y > 0) (h3 : z > 0)
  (h4 : x^2 + y^2 = z^2) (h5 : 3 * y = 4 * x) :
  ∃ A : ℝ, A = (2/3) * x^2 ∧ A = (6/25) * z^2 := by
  sorry

#check right_triangle_area

end right_triangle_area_l3855_385597


namespace complement_of_A_union_B_l3855_385577

open Set

-- Define the sets A and B
def A : Set ℝ := {x : ℝ | -1 < x ∧ x < 5}
def B : Set ℝ := {x : ℝ | x^2 ≥ 4}

-- State the theorem
theorem complement_of_A_union_B :
  (A ∪ B)ᶜ = Ioc (-2) (-1) := by sorry

end complement_of_A_union_B_l3855_385577


namespace percentage_increase_l3855_385591

theorem percentage_increase (original : ℝ) (final : ℝ) (percentage : ℝ) : 
  original = 900 →
  final = 1080 →
  percentage = ((final - original) / original) * 100 →
  percentage = 20 := by
sorry

end percentage_increase_l3855_385591


namespace lcm_gcf_problem_l3855_385563

theorem lcm_gcf_problem (n : ℕ) : 
  Nat.lcm n 14 = 56 → Nat.gcd n 14 = 10 → n = 40 := by
  sorry

end lcm_gcf_problem_l3855_385563


namespace division_problem_l3855_385544

theorem division_problem : ∃ (q : ℕ), 
  220080 = (555 + 445) * q + 80 ∧ 
  q = 2 * (555 - 445) := by
  sorry

end division_problem_l3855_385544


namespace smallest_value_absolute_equation_l3855_385512

theorem smallest_value_absolute_equation :
  ∃ (x : ℝ), x = -5 ∧ |x - 4| = 9 ∧ ∀ (y : ℝ), |y - 4| = 9 → y ≥ x :=
by sorry

end smallest_value_absolute_equation_l3855_385512


namespace banana_count_l3855_385529

/-- Proves that given 8 boxes and 5 bananas per box, the total number of bananas is 40. -/
theorem banana_count (num_boxes : ℕ) (bananas_per_box : ℕ) (total_bananas : ℕ) : 
  num_boxes = 8 → bananas_per_box = 5 → total_bananas = num_boxes * bananas_per_box → total_bananas = 40 :=
by sorry

end banana_count_l3855_385529


namespace factorization_theorem_l3855_385523

theorem factorization_theorem (m n : ℝ) : m^3*n - m*n = m*n*(m-1)*(m+1) := by
  sorry

end factorization_theorem_l3855_385523


namespace total_books_bought_l3855_385583

/-- Proves that the total number of books bought is 90 -/
theorem total_books_bought (total_price : ℕ) (math_book_price : ℕ) (history_book_price : ℕ) (math_books_count : ℕ) :
  total_price = 390 →
  math_book_price = 4 →
  history_book_price = 5 →
  math_books_count = 60 →
  math_books_count + (total_price - math_books_count * math_book_price) / history_book_price = 90 :=
by
  sorry

end total_books_bought_l3855_385583


namespace least_positive_integer_with_four_distinct_primes_l3855_385590

def is_prime (n : ℕ) : Prop := sorry

def distinct_prime_factors (n : ℕ) : Prop := sorry

theorem least_positive_integer_with_four_distinct_primes :
  ∃ (n : ℕ), n > 0 ∧ distinct_prime_factors n ∧ (∀ m : ℕ, m > 0 → distinct_prime_factors m → n ≤ m) :=
sorry

end least_positive_integer_with_four_distinct_primes_l3855_385590


namespace cube_root_comparison_l3855_385515

theorem cube_root_comparison : 2 + Real.rpow 7 (1/3) < Real.rpow 60 (1/3) := by
  sorry

end cube_root_comparison_l3855_385515


namespace max_trig_ratio_l3855_385564

theorem max_trig_ratio (x : ℝ) : 
  (Real.sin x)^2 + (Real.cos x)^2 = 1 → 
  ((Real.sin x)^4 + (Real.cos x)^4 + 1) / ((Real.sin x)^2 + (Real.cos x)^2 + 1) ≤ 7/4 := by
sorry

end max_trig_ratio_l3855_385564


namespace toy_car_energy_comparison_l3855_385501

theorem toy_car_energy_comparison (m : ℝ) (h : m > 0) :
  let KE (v : ℝ) := (1/2) * m * v^2
  (KE 4 - KE 2) = 3 * (KE 2 - KE 0) :=
by
  sorry

end toy_car_energy_comparison_l3855_385501


namespace factorization_equality_l3855_385530

theorem factorization_equality (a b : ℝ) : a^3 - 9*a*b^2 = a*(a+3*b)*(a-3*b) := by
  sorry

end factorization_equality_l3855_385530


namespace three_lines_intersection_l3855_385510

/-- Three lines intersect at a single point if and only if m = 22/7 -/
theorem three_lines_intersection (x y m : ℚ) : 
  (y = 3 * x + 2) ∧ 
  (y = -4 * x + 10) ∧ 
  (y = 2 * x + m) → 
  m = 22 / 7 := by
sorry

end three_lines_intersection_l3855_385510


namespace min_value_d_l3855_385556

def a (n : ℕ+) : ℚ := 1000 / n
def b (n k : ℕ+) : ℚ := 2000 / (k * n)
def c (n k : ℕ+) : ℚ := 1500 / (200 - n - k * n)
def d (n k : ℕ+) : ℚ := max (a n) (max (b n k) (c n k))

theorem min_value_d (n k : ℕ+) (h : n + k * n < 200) :
  ∃ (n₀ k₀ : ℕ+), d n₀ k₀ = 250 / 11 ∧ ∀ (n' k' : ℕ+), n' + k' * n' < 200 → d n' k' ≥ 250 / 11 :=
sorry

end min_value_d_l3855_385556


namespace function_composition_result_l3855_385542

theorem function_composition_result (c d : ℝ) 
  (f : ℝ → ℝ) (g : ℝ → ℝ) 
  (hf : ∀ x, f x = 5*x + c) 
  (hg : ∀ x, g x = c*x + 3) 
  (h_comp : ∀ x, f (g x) = 15*x + d) : d = 18 := by
  sorry

end function_composition_result_l3855_385542


namespace min_blocks_for_specific_wall_l3855_385548

/-- Represents the dimensions of the wall --/
structure WallDimensions where
  length : ℕ
  height : ℕ

/-- Represents the dimensions of a block --/
structure BlockDimensions where
  height : ℕ
  length : ℕ

/-- Calculates the minimum number of blocks required to build the wall --/
def minBlocksRequired (wall : WallDimensions) (block1 : BlockDimensions) (block2 : BlockDimensions) : ℕ :=
  sorry

/-- Theorem stating the minimum number of blocks required for the specific wall --/
theorem min_blocks_for_specific_wall :
  let wall := WallDimensions.mk 120 10
  let block1 := BlockDimensions.mk 1 3
  let block2 := BlockDimensions.mk 1 1
  minBlocksRequired wall block1 block2 = 415 :=
by sorry

end min_blocks_for_specific_wall_l3855_385548


namespace max_b_value_l3855_385508

def is_prime (n : ℕ) : Prop := sorry

theorem max_b_value (a b c : ℕ) : 
  (a * b * c = 720) →
  (1 < c) →
  (c < b) →
  (b < a) →
  (c = 3) →
  is_prime a →
  is_prime b →
  is_prime c →
  (∀ x : ℕ, (1 < x) ∧ (x < b) ∧ is_prime x → x ≤ 3) →
  (b ≤ 5) :=
sorry

end max_b_value_l3855_385508


namespace interval_cardinality_equal_l3855_385560

/-- Two sets are equinumerous if there exists a bijection between them -/
def Equinumerous (α β : Type*) : Prop :=
  ∃ f : α → β, Function.Bijective f

theorem interval_cardinality_equal (a b : ℝ) (h : a < b) :
  Equinumerous (Set.Icc a b) (Set.Ioo a b) ∧
  Equinumerous (Set.Icc a b) (Set.Ico a b) ∧
  Equinumerous (Set.Icc a b) (Set.Ioc a b) :=
sorry

end interval_cardinality_equal_l3855_385560


namespace z_in_fourth_quadrant_l3855_385561

-- Define the operation
def det (a b c d : ℂ) : ℂ := a * d - b * c

-- Define the imaginary unit
def i : ℂ := Complex.I

-- Define z using the operation
def z : ℂ := det 1 2 i (i^4)

-- Define the fourth quadrant
def fourth_quadrant (z : ℂ) : Prop :=
  z.re > 0 ∧ z.im < 0

-- Theorem statement
theorem z_in_fourth_quadrant : fourth_quadrant z := by
  sorry

end z_in_fourth_quadrant_l3855_385561


namespace megan_folders_l3855_385551

theorem megan_folders (initial_files : ℕ) (deleted_files : ℕ) (files_per_folder : ℕ) : 
  initial_files = 93 → 
  deleted_files = 21 → 
  files_per_folder = 8 → 
  (initial_files - deleted_files) / files_per_folder = 9 :=
by
  sorry

end megan_folders_l3855_385551


namespace figure_to_square_approximation_l3855_385504

/-- A figure on a grid of squares -/
structure GridFigure where
  area : ℕ
  is_on_grid : Bool

/-- Represents a division of a figure into parts -/
structure FigureDivision where
  parts : ℕ
  can_rearrange_to_square : Bool

/-- Theorem: A figure with 18 unit squares can be divided into three parts and rearranged to approximate a square -/
theorem figure_to_square_approximation (f : GridFigure) (d : FigureDivision) :
  f.area = 18 ∧ f.is_on_grid = true ∧ d.parts = 3 → d.can_rearrange_to_square = true := by
  sorry

end figure_to_square_approximation_l3855_385504


namespace smallest_four_digit_divisible_by_53_l3855_385571

theorem smallest_four_digit_divisible_by_53 : 
  ∀ n : ℕ, 1000 ≤ n ∧ n < 10000 ∧ n % 53 = 0 → n ≥ 1007 :=
by sorry

end smallest_four_digit_divisible_by_53_l3855_385571


namespace odd_function_property_l3855_385569

def is_odd (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

theorem odd_function_property 
  (f : ℝ → ℝ) 
  (h_odd : is_odd f) 
  (h_prop : ∀ x, f (1 + x) = f (-x)) 
  (h_value : f (-1/3) = 1/3) : 
  f (5/3) = 1/3 := by
sorry

end odd_function_property_l3855_385569


namespace smallest_k_for_single_root_l3855_385513

-- Define the functions f and g
def f (x : ℝ) : ℝ := 41 * x^2 - 4 * x + 4
def g (x : ℝ) : ℝ := -2 * x^2 + x

-- Define the combined function h
def h (k : ℝ) (x : ℝ) : ℝ := f x + k * g x

-- Define the discriminant of h
def discriminant (k : ℝ) : ℝ := (k - 4)^2 - 4 * (41 - 2*k) * 4

-- Theorem statement
theorem smallest_k_for_single_root :
  ∃ d : ℝ, d = -40 ∧ 
  (∀ k : ℝ, (∃ x : ℝ, h k x = 0 ∧ (∀ y : ℝ, h k y = 0 → y = x)) → k ≥ d) ∧
  (∃ x : ℝ, h d x = 0 ∧ (∀ y : ℝ, h d y = 0 → y = x)) :=
sorry

end smallest_k_for_single_root_l3855_385513


namespace simplify_cube_divided_by_base_l3855_385541

theorem simplify_cube_divided_by_base (x y : ℝ) : (x + y)^3 / (x + y) = (x + y)^2 := by
  sorry

end simplify_cube_divided_by_base_l3855_385541


namespace quadratic_inequality_solution_l3855_385503

/-- Given that the solution set of ax^2 + bx + 2 > 0 is {x | -1/2 < x < 1/3}, prove that a - b = -10 -/
theorem quadratic_inequality_solution (a b : ℝ) : 
  (∀ x, ax^2 + b*x + 2 > 0 ↔ -1/2 < x ∧ x < 1/3) → a - b = -10 :=
by sorry

end quadratic_inequality_solution_l3855_385503


namespace equilateral_triangle_properties_l3855_385509

/-- Proves properties of an equilateral triangle with given area and side length -/
theorem equilateral_triangle_properties :
  ∀ (area base altitude perimeter : ℝ),
  area = 450 →
  base = 25 →
  area = (1/2) * base * altitude →
  perimeter = 3 * base →
  altitude = 36 ∧ perimeter = 75 := by
sorry

end equilateral_triangle_properties_l3855_385509


namespace fourth_month_sale_l3855_385511

/-- Given the sales for 5 out of 6 months and the average sale for 6 months, 
    prove that the sale in the fourth month must be 8230. -/
theorem fourth_month_sale 
  (sale1 : ℕ) (sale2 : ℕ) (sale3 : ℕ) (sale5 : ℕ) (sale6 : ℕ) (avg_sale : ℕ)
  (h1 : sale1 = 7435)
  (h2 : sale2 = 7920)
  (h3 : sale3 = 7855)
  (h5 : sale5 = 7560)
  (h6 : sale6 = 6000)
  (h_avg : avg_sale = 7500)
  : ∃ (sale4 : ℕ), sale4 = 8230 ∧ 
    (sale1 + sale2 + sale3 + sale4 + sale5 + sale6) / 6 = avg_sale :=
by sorry

end fourth_month_sale_l3855_385511


namespace farm_animals_l3855_385502

theorem farm_animals (total_legs : ℕ) (total_animals : ℕ) (sheep : ℕ) :
  total_legs = 60 →
  total_animals = 20 →
  total_legs = 2 * (total_animals - sheep) + 4 * sheep →
  sheep = 10 := by
sorry

end farm_animals_l3855_385502


namespace p_necessary_not_sufficient_for_q_l3855_385538

theorem p_necessary_not_sufficient_for_q (a b : ℝ) :
  (∀ a b, a^2 + b^2 ≠ 0 → a * b = 0) ∧
  ¬(∀ a b, a * b = 0 → a^2 + b^2 ≠ 0) :=
by sorry

end p_necessary_not_sufficient_for_q_l3855_385538


namespace inequality_proof_l3855_385599

theorem inequality_proof (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) :
  1/x + 1/y + 1/z + 9/(x+y+z) ≥ 
  3 * ((1/(2*x+y) + 1/(x+2*y)) + (1/(2*y+z) + 1/(y+2*z)) + (1/(2*z+x) + 1/(x+2*z))) := by
  sorry

end inequality_proof_l3855_385599
