import Mathlib

namespace x_142_equals_1995_and_unique_l1811_181155

def p (x : ℕ) : ℕ := sorry

def q (x : ℕ) : ℕ := sorry

def x : ℕ → ℕ
  | 0 => 1
  | n + 1 => (x n * p (x n)) / q (x n)

theorem x_142_equals_1995_and_unique :
  x 142 = 1995 ∧ ∀ n : ℕ, n ≠ 142 → x n ≠ 1995 := by sorry

end x_142_equals_1995_and_unique_l1811_181155


namespace inscribed_quadrilateral_symmetry_l1811_181101

/-- A circle in which the quadrilateral is inscribed -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- A point in the 2D plane -/
def Point := ℝ × ℝ

/-- A line in the 2D plane -/
structure Line where
  point1 : Point
  point2 : Point

/-- A quadrilateral inscribed in a circle -/
structure InscribedQuadrilateral where
  circle : Circle
  A : Point
  B : Point
  C : Point
  D : Point

/-- Intersection point of two lines -/
def intersectionPoint (l1 l2 : Line) : Point := sorry

/-- Check if a point is on a line -/
def isPointOnLine (p : Point) (l : Line) : Prop := sorry

/-- Check if two points are symmetrical with respect to a third point -/
def areSymmetrical (p1 p2 center : Point) : Prop := sorry

/-- The main theorem -/
theorem inscribed_quadrilateral_symmetry 
  (quad : InscribedQuadrilateral)
  (E : Point)
  (t : Line) :
  let AB := Line.mk quad.A quad.B
  let CD := Line.mk quad.C quad.D
  let AC := Line.mk quad.A quad.C
  let BD := Line.mk quad.B quad.D
  let BC := Line.mk quad.B quad.C
  let AD := Line.mk quad.A quad.D
  let O := quad.circle.center
  E = intersectionPoint AB CD →
  isPointOnLine E t →
  (∀ p : Point, isPointOnLine p (Line.mk O E) → isPointOnLine p t → p = E) →
  ∃ (P Q R S : Point),
    P = intersectionPoint AC t ∧
    Q = intersectionPoint BD t ∧
    R = intersectionPoint BC t ∧
    S = intersectionPoint AD t ∧
    areSymmetrical P Q E ∧
    areSymmetrical R S E :=
sorry

end inscribed_quadrilateral_symmetry_l1811_181101


namespace mikes_ride_distance_l1811_181118

theorem mikes_ride_distance (mike_start_fee annie_start_fee : ℚ)
  (annie_bridge_toll : ℚ) (cost_per_mile : ℚ) (annie_distance : ℚ)
  (h1 : mike_start_fee = 2.5)
  (h2 : annie_start_fee = 2.5)
  (h3 : annie_bridge_toll = 5)
  (h4 : cost_per_mile = 0.25)
  (h5 : annie_distance = 22)
  (h6 : ∃ (mike_distance : ℚ),
    mike_start_fee + cost_per_mile * mike_distance =
    annie_start_fee + annie_bridge_toll + cost_per_mile * annie_distance) :
  ∃ (mike_distance : ℚ), mike_distance = 32 :=
by sorry

end mikes_ride_distance_l1811_181118


namespace simplify_sqrt_2_simplify_complex_sqrt_l1811_181115

-- Part 1
theorem simplify_sqrt_2 : 2 * Real.sqrt 2 - Real.sqrt 2 = Real.sqrt 2 := by
  sorry

-- Part 2
theorem simplify_complex_sqrt : 
  Real.sqrt 2 * Real.sqrt 10 / (1 / Real.sqrt 5) = 10 := by
  sorry

end simplify_sqrt_2_simplify_complex_sqrt_l1811_181115


namespace unique_solution_trigonometric_equation_l1811_181160

theorem unique_solution_trigonometric_equation :
  ∃! x : ℝ, 2 * Real.sin (π * x / 2) - 2 * Real.cos (π * x / 2) = x^5 + 10*x - 54 :=
by sorry

end unique_solution_trigonometric_equation_l1811_181160


namespace can_form_123_l1811_181199

/-- A type representing the allowed arithmetic operations -/
inductive Operation
| Add
| Subtract
| Multiply

/-- A type representing an arithmetic expression -/
inductive Expr
| Num (n : ℕ)
| Op (op : Operation) (e1 e2 : Expr)

/-- Evaluates an arithmetic expression -/
def eval : Expr → ℤ
| Expr.Num n => n
| Expr.Op Operation.Add e1 e2 => eval e1 + eval e2
| Expr.Op Operation.Subtract e1 e2 => eval e1 - eval e2
| Expr.Op Operation.Multiply e1 e2 => eval e1 * eval e2

/-- Checks if an expression uses each of the numbers 1, 2, 3, 4, 5 exactly once -/
def usesAllNumbers : Expr → Bool := sorry

/-- The main theorem stating that 123 can be formed using the given rules -/
theorem can_form_123 : ∃ e : Expr, usesAllNumbers e ∧ eval e = 123 := by sorry

end can_form_123_l1811_181199


namespace unique_positive_number_l1811_181134

theorem unique_positive_number : ∃! x : ℝ, x > 0 ∧ x + 17 = 60 / x := by
  sorry

end unique_positive_number_l1811_181134


namespace point_on_line_equal_intercepts_l1811_181138

/-- A line passing through (-2, -3) with equal x and y intercepts -/
def line_with_equal_intercepts (x y : ℝ) : Prop :=
  x + y = 5

/-- The point (-2, -3) lies on the line -/
theorem point_on_line : line_with_equal_intercepts (-2) (-3) := by sorry

/-- The line has equal intercepts on x and y axes -/
theorem equal_intercepts :
  ∃ a : ℝ, a > 0 ∧ line_with_equal_intercepts a 0 ∧ line_with_equal_intercepts 0 a := by sorry

end point_on_line_equal_intercepts_l1811_181138


namespace elevator_problem_l1811_181124

theorem elevator_problem (x y z w v : ℕ) (h : x = 15 ∧ y = 9 ∧ z = 12 ∧ w = 6 ∧ v = 10) :
  x - y + z - w + v = 28 :=
by sorry

end elevator_problem_l1811_181124


namespace students_per_bench_l1811_181145

theorem students_per_bench (male_students : ℕ) (benches : ℕ) : 
  male_students = 29 →
  benches = 29 →
  ∃ (students_per_bench : ℕ), 
    students_per_bench ≥ 5 ∧
    students_per_bench * benches ≥ male_students + 4 * male_students :=
by sorry

end students_per_bench_l1811_181145


namespace square_sum_17_5_l1811_181117

theorem square_sum_17_5 : 17^2 + 2*(17*5) + 5^2 = 484 := by
  sorry

end square_sum_17_5_l1811_181117


namespace contrapositive_example_l1811_181194

theorem contrapositive_example :
  (∀ x : ℝ, x > 2 → x > 0) ↔ (∀ x : ℝ, x ≤ 0 → x ≤ 2) :=
by sorry

end contrapositive_example_l1811_181194


namespace game_ends_in_finite_steps_l1811_181125

/-- Represents the state of a bowl in the game -/
inductive BowlState
| Empty : BowlState
| NonEmpty : BowlState

/-- Represents the game state -/
def GameState (n : ℕ) := Fin n → BowlState

/-- Function to place a bean in a bowl -/
def placeBeanInBowl (k : ℕ) (n : ℕ) : Fin n :=
  ⟨k * (k + 1) / 2, sorry⟩

/-- Predicate to check if a number is a power of 2 -/
def isPowerOfTwo (n : ℕ) : Prop :=
  ∃ k : ℕ, n = 2^k

/-- Theorem stating the condition for the game to end in finite steps -/
theorem game_ends_in_finite_steps (n : ℕ) :
  (∃ k : ℕ, ∀ i : Fin n, (placeBeanInBowl k n).val = i.val → 
    ∃ m : ℕ, m ≤ k ∧ (placeBeanInBowl m n).val = i.val) ↔ 
  isPowerOfTwo n :=
sorry


end game_ends_in_finite_steps_l1811_181125


namespace sqrt_sum_fractions_l1811_181165

theorem sqrt_sum_fractions : Real.sqrt (1/25 + 1/36) = Real.sqrt 61 / 30 := by
  sorry

end sqrt_sum_fractions_l1811_181165


namespace a_cube_gt_b_cube_l1811_181156

theorem a_cube_gt_b_cube (a b : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (h : a * abs a > b * abs b) : a^3 > b^3 := by
  sorry

end a_cube_gt_b_cube_l1811_181156


namespace min_value_of_sum_of_squares_l1811_181150

theorem min_value_of_sum_of_squares (a b : ℝ) : 
  (a > 0) → 
  (b > 0) → 
  ((a - 1)^3 + (b - 1)^3 ≥ 3 * (2 - a - b)) → 
  (a^2 + b^2 ≥ 2) := by
sorry

end min_value_of_sum_of_squares_l1811_181150


namespace gcd_5280_12155_l1811_181123

theorem gcd_5280_12155 : Int.gcd 5280 12155 = 5 := by
  sorry

end gcd_5280_12155_l1811_181123


namespace metal_waste_l1811_181181

/-- Given a rectangle with length l and breadth b (where l > b), from which a maximum-sized
    circular piece is cut and then a maximum-sized square piece is cut from that circle,
    the total amount of metal wasted is equal to l × b - b²/2. -/
theorem metal_waste (l b : ℝ) (h : l > b) (b_pos : b > 0) :
  let circle_area := π * (b/2)^2
  let square_side := b / Real.sqrt 2
  let square_area := square_side^2
  l * b - square_area = l * b - b^2/2 :=
by sorry

end metal_waste_l1811_181181


namespace selection_ways_equal_210_l1811_181166

/-- The number of ways to select at least one boy from a group of 6 boys and G girls is 210 if and only if G = 1 -/
theorem selection_ways_equal_210 (G : ℕ) : (63 * 2^G = 210) ↔ G = 1 := by sorry

end selection_ways_equal_210_l1811_181166


namespace product_mod_25_l1811_181171

theorem product_mod_25 (n : ℕ) : 
  77 * 88 * 99 ≡ n [ZMOD 25] → 0 ≤ n → n < 25 → n = 24 := by
  sorry

end product_mod_25_l1811_181171


namespace son_age_proof_l1811_181136

-- Define the variables
def your_age : ℕ := 45
def son_age : ℕ := 15

-- Define the conditions
theorem son_age_proof :
  (your_age = 3 * son_age) ∧
  (your_age + 5 = (5/2) * (son_age + 5)) →
  son_age = 15 := by
sorry


end son_age_proof_l1811_181136


namespace profit_percentage_unchanged_l1811_181144

/-- Represents a retailer's sales and profit information -/
structure RetailerInfo where
  monthly_sales : ℝ
  profit_percentage : ℝ
  discount_percentage : ℝ
  break_even_sales : ℝ

/-- The retailer's original sales information -/
def original_info : RetailerInfo :=
  { monthly_sales := 100
  , profit_percentage := 0.10
  , discount_percentage := 0
  , break_even_sales := 100 }

/-- The retailer's sales information with discount -/
def discounted_info : RetailerInfo :=
  { monthly_sales := 222.22
  , profit_percentage := 0.10
  , discount_percentage := 0.05
  , break_even_sales := 222.22 }

/-- Calculates the total profit for a given RetailerInfo -/
def total_profit (info : RetailerInfo) (price : ℝ) : ℝ :=
  info.monthly_sales * (info.profit_percentage - info.discount_percentage) * price

/-- Theorem stating that the profit percentage remains the same
    regardless of the discount, given the break-even sales volume -/
theorem profit_percentage_unchanged
  (price : ℝ)
  (h_price_pos : price > 0) :
  original_info.profit_percentage = discounted_info.profit_percentage :=
by
  sorry


end profit_percentage_unchanged_l1811_181144


namespace smallest_n_congruence_l1811_181140

theorem smallest_n_congruence (n : ℕ) : 
  (∀ m : ℕ, 0 < m → m < 15 → ¬(890 * m ≡ 1426 * m [ZMOD 30])) ∧ 
  (890 * 15 ≡ 1426 * 15 [ZMOD 30]) := by
  sorry

end smallest_n_congruence_l1811_181140


namespace cube_surface_area_l1811_181178

/-- The surface area of a cube with edge length 3a is 54a² -/
theorem cube_surface_area (a : ℝ) : 
  6 * (3 * a)^2 = 54 * a^2 := by sorry

end cube_surface_area_l1811_181178


namespace january_display_144_l1811_181126

/-- Rose display sequence with a constant increase -/
structure RoseSequence where
  october : ℕ
  november : ℕ
  december : ℕ
  february : ℕ
  constant_increase : ℕ
  increase_consistent : 
    november - october = constant_increase ∧
    december - november = constant_increase ∧
    february - (december + constant_increase) = constant_increase

/-- The number of roses displayed in January given a rose sequence -/
def january_roses (seq : RoseSequence) : ℕ :=
  seq.december + seq.constant_increase

/-- Theorem stating that for the given rose sequence, January displays 144 roses -/
theorem january_display_144 (seq : RoseSequence) 
  (h_oct : seq.october = 108)
  (h_nov : seq.november = 120)
  (h_dec : seq.december = 132)
  (h_feb : seq.february = 156) :
  january_roses seq = 144 := by
  sorry


end january_display_144_l1811_181126


namespace t_value_l1811_181157

theorem t_value (p j t : ℝ) 
  (hj_p : j = p * (1 - 0.25))
  (hj_t : j = t * (1 - 0.20))
  (ht_p : t = p * (1 - t / 100)) :
  t = 6.25 := by sorry

end t_value_l1811_181157


namespace age_ratio_is_eleven_eighths_l1811_181109

/-- Represents the ages and relationships of Rehana, Phoebe, Jacob, and Xander -/
structure AgeGroup where
  rehana_age : ℕ
  phoebe_age : ℕ
  jacob_age : ℕ
  xander_age : ℕ

/-- Conditions for the age group -/
def valid_age_group (ag : AgeGroup) : Prop :=
  ag.rehana_age = 25 ∧
  ag.rehana_age + 5 = 3 * (ag.phoebe_age + 5) ∧
  ag.jacob_age = (3 * ag.phoebe_age) / 5 ∧
  ag.xander_age = ag.rehana_age + ag.jacob_age - 4

/-- The ratio of combined ages to Xander's age -/
def age_ratio (ag : AgeGroup) : ℚ :=
  (ag.rehana_age + ag.phoebe_age + ag.jacob_age : ℚ) / ag.xander_age

/-- Theorem stating the age ratio is 11/8 for a valid age group -/
theorem age_ratio_is_eleven_eighths (ag : AgeGroup) (h : valid_age_group ag) :
  age_ratio ag = 11/8 := by
  sorry

end age_ratio_is_eleven_eighths_l1811_181109


namespace simplified_fraction_ratio_l1811_181112

theorem simplified_fraction_ratio (k c d : ℤ) : 
  (5 * k + 15) / 5 = c * k + d → c / d = 1 / 3 := by
  sorry

end simplified_fraction_ratio_l1811_181112


namespace ariels_fish_count_l1811_181111

theorem ariels_fish_count (total : ℕ) (male_ratio : ℚ) (female_count : ℕ) 
  (h1 : male_ratio = 2/3)
  (h2 : female_count = 15)
  (h3 : ↑female_count = (1 - male_ratio) * ↑total) : 
  total = 45 := by
  sorry

end ariels_fish_count_l1811_181111


namespace max_product_sum_max_product_sum_achieved_l1811_181161

theorem max_product_sum (A M C : ℕ) (h : A + M + C = 15) :
  (A * M * C + A * M + M * C + C * A) ≤ 200 :=
by sorry

theorem max_product_sum_achieved :
  ∃ A M C : ℕ, A + M + C = 15 ∧ A * M * C + A * M + M * C + C * A = 200 :=
by sorry

end max_product_sum_max_product_sum_achieved_l1811_181161


namespace other_number_from_hcf_lcm_l1811_181106

theorem other_number_from_hcf_lcm (A B : ℕ+) : 
  Nat.gcd A B = 12 → 
  Nat.lcm A B = 396 → 
  A = 24 → 
  B = 198 := by
sorry

end other_number_from_hcf_lcm_l1811_181106


namespace total_games_is_140_l1811_181164

/-- The number of teams in the "High School Ten" basketball conference -/
def num_teams : ℕ := 10

/-- The number of times each team plays every other team in the conference -/
def games_against_each_team : ℕ := 2

/-- The number of games each team plays against non-conference opponents -/
def non_conference_games_per_team : ℕ := 5

/-- The total number of games in a season involving the "High School Ten" teams -/
def total_games : ℕ := (num_teams * (num_teams - 1) / 2) * games_against_each_team + num_teams * non_conference_games_per_team

/-- Theorem stating that the total number of games in a season is 140 -/
theorem total_games_is_140 : total_games = 140 := by
  sorry

end total_games_is_140_l1811_181164


namespace simplify_nested_roots_l1811_181147

theorem simplify_nested_roots : 
  (((1 / 65536)^(1/2))^(1/3))^(1/4) = 1 / (2^(2/3)) := by
  sorry

end simplify_nested_roots_l1811_181147


namespace player_a_strategy_wins_l1811_181174

-- Define the grid as a 3x3 matrix of real numbers
def Grid := Matrix (Fin 3) (Fin 3) ℝ

-- Define a function to calculate the sum of first and third rows
def sumRows (g : Grid) : ℝ := 
  (g 0 0 + g 0 1 + g 0 2) + (g 2 0 + g 2 1 + g 2 2)

-- Define a function to calculate the sum of first and third columns
def sumCols (g : Grid) : ℝ := 
  (g 0 0 + g 1 0 + g 2 0) + (g 0 2 + g 1 2 + g 2 2)

-- Theorem statement
theorem player_a_strategy_wins 
  (cards : Finset ℝ) 
  (h_card_count : cards.card = 9) : 
  ∃ (g : Grid), (∀ i j, g i j ∈ cards) ∧ sumRows g ≥ sumCols g := by
  sorry


end player_a_strategy_wins_l1811_181174


namespace decimal_sum_to_fraction_l1811_181198

theorem decimal_sum_to_fraction :
  (0.2 : ℚ) + 0.03 + 0.004 + 0.0006 = 1173 / 5000 := by
  sorry

end decimal_sum_to_fraction_l1811_181198


namespace triangle_expression_bounds_l1811_181186

theorem triangle_expression_bounds (A B C : Real) (a b c : Real) :
  (0 < A) → (A < π) →
  (0 < B) → (B < π) →
  (0 < C) → (C < π) →
  (A + B + C = π) →
  (c * Real.sin A = -a * Real.cos C) →
  ∃ (x : Real), (1 < x) ∧ 
  (x < (Real.sqrt 6 + Real.sqrt 2) / 2) ∧
  (x = Real.sqrt 3 * Real.sin A - Real.cos (B + 3 * π / 4)) :=
by sorry

end triangle_expression_bounds_l1811_181186


namespace inequality_check_l1811_181182

theorem inequality_check : 
  (¬(0 < -5)) ∧ 
  (¬(7 < -1)) ∧ 
  (¬(10 < (1/4 : ℚ))) ∧ 
  (¬(-1 < -3)) ∧ 
  (-8 < -2) := by
  sorry

end inequality_check_l1811_181182


namespace tan_alpha_beta_eq_three_tan_alpha_l1811_181175

/-- Given that 2 sin β = sin(2α + β), prove that tan(α + β) = 3 tan α -/
theorem tan_alpha_beta_eq_three_tan_alpha (α β : ℝ) 
  (h : 2 * Real.sin β = Real.sin (2 * α + β)) :
  Real.tan (α + β) = 3 * Real.tan α := by
  sorry

end tan_alpha_beta_eq_three_tan_alpha_l1811_181175


namespace inequality_solution_implies_m_l1811_181110

theorem inequality_solution_implies_m (m : ℝ) : 
  (∀ x, mx + 2 > 0 ↔ x < 2) → m = -1 := by
sorry

end inequality_solution_implies_m_l1811_181110


namespace seven_balls_two_boxes_l1811_181129

/-- The number of ways to distribute indistinguishable balls into distinguishable boxes -/
def distribute_balls (n : ℕ) (k : ℕ) : ℕ :=
  n + 1

theorem seven_balls_two_boxes :
  distribute_balls 7 2 = 8 := by
  sorry

end seven_balls_two_boxes_l1811_181129


namespace quadratic_ratio_l1811_181177

/-- Given a quadratic polynomial of the form x^2 + 1800x + 2700,
    prove that when written as (x + b)^2 + c, the ratio c/b equals -897 -/
theorem quadratic_ratio (x : ℝ) :
  let f := fun x => x^2 + 1800*x + 2700
  ∃ b c : ℝ, (∀ x, f x = (x + b)^2 + c) ∧ c / b = -897 := by
  sorry

end quadratic_ratio_l1811_181177


namespace tennis_to_soccer_ratio_l1811_181152

/-- Represents the number of balls of each type -/
structure BallCounts where
  total : ℕ
  soccer : ℕ
  basketball : ℕ
  baseball : ℕ
  volleyball : ℕ
  tennis : ℕ

/-- The ratio of two natural numbers -/
structure Ratio where
  numerator : ℕ
  denominator : ℕ

/-- Theorem stating the ratio of tennis balls to soccer balls -/
theorem tennis_to_soccer_ratio (counts : BallCounts) : 
  counts.total = 145 →
  counts.soccer = 20 →
  counts.basketball = counts.soccer + 5 →
  counts.baseball = counts.soccer + 10 →
  counts.volleyball = 30 →
  counts.total = counts.soccer + counts.basketball + counts.baseball + counts.volleyball + counts.tennis →
  (Ratio.mk counts.tennis counts.soccer) = (Ratio.mk 2 1) := by
  sorry

end tennis_to_soccer_ratio_l1811_181152


namespace z_in_third_quadrant_l1811_181121

/-- The complex number z defined as i(-2 + i) -/
def z : ℂ := Complex.I * (Complex.mk (-2) 1)

/-- Predicate to check if a complex number is in the third quadrant -/
def in_third_quadrant (w : ℂ) : Prop :=
  w.re < 0 ∧ w.im < 0

/-- Theorem stating that z is in the third quadrant -/
theorem z_in_third_quadrant : in_third_quadrant z := by sorry

end z_in_third_quadrant_l1811_181121


namespace polynomial_sum_equality_l1811_181151

/-- Given polynomials p, q, and r, prove their sum is equal to the specified polynomial -/
theorem polynomial_sum_equality (x : ℝ) : 
  let p := fun x : ℝ => -4*x^2 + 2*x - 5
  let q := fun x : ℝ => -6*x^2 + 4*x - 9
  let r := fun x : ℝ => 6*x^2 + 6*x + 2
  p x + q x + r x = -4*x^2 + 12*x - 12 := by
  sorry

end polynomial_sum_equality_l1811_181151


namespace edward_money_theorem_l1811_181116

def edward_money_problem (initial_amount spent1 spent2 remaining : ℕ) : Prop :=
  initial_amount = spent1 + spent2 + remaining

theorem edward_money_theorem :
  ∃ initial_amount : ℕ,
    edward_money_problem initial_amount 9 8 17 ∧ initial_amount = 34 := by
  sorry

end edward_money_theorem_l1811_181116


namespace stratified_sampling_l1811_181187

theorem stratified_sampling (total_employees : ℕ) (administrators : ℕ) (sample_size : ℕ)
  (h1 : total_employees = 160)
  (h2 : administrators = 32)
  (h3 : sample_size = 20) :
  (administrators * sample_size) / total_employees = 4 := by sorry

end stratified_sampling_l1811_181187


namespace darla_electricity_payment_l1811_181184

/-- The number of watts of electricity Darla needs to pay for -/
def watts : ℝ := 300

/-- The cost per watt of electricity in dollars -/
def cost_per_watt : ℝ := 4

/-- The late fee in dollars -/
def late_fee : ℝ := 150

/-- The total payment in dollars -/
def total_payment : ℝ := 1350

theorem darla_electricity_payment :
  cost_per_watt * watts + late_fee = total_payment := by
  sorry

end darla_electricity_payment_l1811_181184


namespace watch_sale_gain_percentage_l1811_181127

/-- Prove the gain percentage for a watch sale --/
theorem watch_sale_gain_percentage 
  (cost_price : ℝ) 
  (initial_loss_percentage : ℝ) 
  (price_increase : ℝ) : 
  cost_price = 875 → 
  initial_loss_percentage = 12 → 
  price_increase = 140 → 
  let initial_selling_price := cost_price * (1 - initial_loss_percentage / 100)
  let new_selling_price := initial_selling_price + price_increase
  let gain := new_selling_price - cost_price
  let gain_percentage := (gain / cost_price) * 100
  gain_percentage = 4 := by
sorry


end watch_sale_gain_percentage_l1811_181127


namespace simplify_expressions_l1811_181143

theorem simplify_expressions :
  (∀ (a b c d : ℝ), 
    a = 4 * Real.sqrt 5 ∧ 
    b = Real.sqrt 45 ∧ 
    c = Real.sqrt 8 ∧ 
    d = 4 * Real.sqrt 2 →
    a + b - c + d = 7 * Real.sqrt 5 + 2 * Real.sqrt 2) ∧
  (∀ (e f g : ℝ),
    e = 2 * Real.sqrt 48 ∧
    f = 3 * Real.sqrt 27 ∧
    g = Real.sqrt 6 →
    (e - f) / g = -(Real.sqrt 2) / 2) :=
by sorry

end simplify_expressions_l1811_181143


namespace food_drive_cans_l1811_181154

theorem food_drive_cans (mark jaydon rachel : ℕ) : 
  mark = 100 ∧ 
  mark = 4 * jaydon ∧ 
  jaydon > 2 * rachel ∧ 
  mark + jaydon + rachel = 135 → 
  jaydon = 2 * rachel + 5 :=
by sorry

end food_drive_cans_l1811_181154


namespace john_marble_weight_l1811_181159

/-- Represents a rectangular prism -/
structure RectangularPrism where
  height : ℝ
  baseLength : ℝ
  baseWidth : ℝ
  density : ℝ

/-- Calculates the volume of a rectangular prism -/
def volume (prism : RectangularPrism) : ℝ :=
  prism.height * prism.baseLength * prism.baseWidth

/-- Calculates the weight of a rectangular prism -/
def weight (prism : RectangularPrism) : ℝ :=
  prism.density * volume prism

/-- The main theorem stating the weight of John's marble prism -/
theorem john_marble_weight :
  let prism : RectangularPrism := {
    height := 8,
    baseLength := 2,
    baseWidth := 2,
    density := 2700
  }
  weight prism = 86400 := by
  sorry


end john_marble_weight_l1811_181159


namespace digit_difference_in_base_d_l1811_181193

/-- 
Given a base d > 8 and digits A and C in base d,
if AC_d + CC_d = 232_d, then A_d - C_d = 1_d.
-/
theorem digit_difference_in_base_d (d : ℕ) (A C : ℕ) 
  (h_base : d > 8) 
  (h_digits : A < d ∧ C < d) 
  (h_sum : A * d + C + C * d + C = 2 * d^2 + 3 * d + 2) : 
  A - C = 1 :=
sorry

end digit_difference_in_base_d_l1811_181193


namespace gcd_459_357_l1811_181176

theorem gcd_459_357 : Nat.gcd 459 357 = 51 := by
  sorry

end gcd_459_357_l1811_181176


namespace intersection_points_inequality_l1811_181195

theorem intersection_points_inequality (a b x₁ x₂ : ℝ) 
  (h₁ : x₁ ≠ x₂) 
  (h₂ : Real.log x₁ / x₁ = a / 2 * x₁ + b) 
  (h₃ : Real.log x₂ / x₂ = a / 2 * x₂ + b) : 
  (x₁ + x₂) * (a / 2 * (x₁ + x₂) + b) > 2 := by
  sorry

end intersection_points_inequality_l1811_181195


namespace inverse_variation_problem_l1811_181100

/-- Given that x² varies inversely with ⁴√w, prove that if x = 3 when w = 16, then x = √6 when w = 81 -/
theorem inverse_variation_problem (x w : ℝ) (h : ∃ k : ℝ, ∀ x w, x^2 * w^(1/4) = k) :
  (x = 3 ∧ w = 16) → (w = 81 → x = Real.sqrt 6) := by
  sorry

end inverse_variation_problem_l1811_181100


namespace probability_no_dessert_l1811_181191

def probability_dessert : ℝ := 0.60
def probability_dessert_no_coffee : ℝ := 0.20

theorem probability_no_dessert :
  1 - probability_dessert = 0.40 :=
sorry

end probability_no_dessert_l1811_181191


namespace fraction_equality_l1811_181169

theorem fraction_equality (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) 
  (h : (3 * x + y) / (x - 3 * y) = -2) : 
  (x + 3 * y) / (3 * x - y) = 2 := by
  sorry

end fraction_equality_l1811_181169


namespace frog_reaches_boundary_in_three_hops_l1811_181131

/-- Represents a position on the 4x4 grid -/
structure Position :=
  (x : Fin 4)
  (y : Fin 4)

/-- Represents the possible directions of movement -/
inductive Direction
  | Up
  | Down
  | Left
  | Right

/-- Defines whether a position is on the boundary of the grid -/
def is_boundary (p : Position) : Bool :=
  p.x = 0 || p.x = 3 || p.y = 0 || p.y = 3

/-- Defines a single hop movement on the grid -/
def hop (p : Position) (d : Direction) : Position :=
  match d with
  | Direction.Up => ⟨min 3 (p.x + 1), p.y⟩
  | Direction.Down => ⟨max 0 (p.x - 1), p.y⟩
  | Direction.Left => ⟨p.x, max 0 (p.y - 1)⟩
  | Direction.Right => ⟨p.x, min 3 (p.y + 1)⟩

/-- Calculates the probability of reaching the boundary within n hops -/
def prob_reach_boundary (start : Position) (n : Nat) : ℝ :=
  sorry

theorem frog_reaches_boundary_in_three_hops :
  prob_reach_boundary ⟨1, 1⟩ 3 = 1 :=
by sorry

end frog_reaches_boundary_in_three_hops_l1811_181131


namespace tom_distance_before_karen_wins_l1811_181133

/-- Represents the race between Karen and Tom -/
structure Race where
  karen_initial_speed : ℝ
  tom_initial_speed : ℝ
  karen_final_speed : ℝ
  tom_final_speed : ℝ
  karen_delay : ℝ
  winning_margin : ℝ

/-- Calculates the distance Tom drives before Karen wins the bet -/
def distance_tom_drives (race : Race) : ℝ :=
  sorry

/-- Theorem stating that Tom drives 21 miles before Karen wins the bet -/
theorem tom_distance_before_karen_wins (race : Race) 
  (h1 : race.karen_initial_speed = 60)
  (h2 : race.tom_initial_speed = 45)
  (h3 : race.karen_final_speed = 70)
  (h4 : race.tom_final_speed = 40)
  (h5 : race.karen_delay = 4/60)  -- 4 minutes converted to hours
  (h6 : race.winning_margin = 4) :
  distance_tom_drives race = 21 :=
sorry

end tom_distance_before_karen_wins_l1811_181133


namespace divisibility_by_three_l1811_181189

theorem divisibility_by_three (B : ℕ) : 
  B < 10 ∧ (5 + 2 + B + 6) % 3 = 0 ↔ B = 2 :=
by sorry

end divisibility_by_three_l1811_181189


namespace polar_to_rectangular_l1811_181146

theorem polar_to_rectangular (r θ : Real) (h : r = 4 ∧ θ = π / 4) :
  (r * Real.cos θ, r * Real.sin θ) = (2 * Real.sqrt 2, 2 * Real.sqrt 2) := by
  sorry

end polar_to_rectangular_l1811_181146


namespace triplet_solution_l1811_181168

def is_valid_triplet (a b c : ℕ) : Prop :=
  a > 0 ∧ b > 0 ∧ c > 0 ∧ a ≤ b ∧ b ≤ c ∧ a^2 + b^2 + c^2 = 2005

def solution_set : Set (ℕ × ℕ × ℕ) :=
  {(24,30,23), (12,30,31), (18,40,9), (15,22,36), (12,30,31)}

theorem triplet_solution :
  ∀ (a b c : ℕ), is_valid_triplet a b c ↔ (a, b, c) ∈ solution_set :=
sorry

end triplet_solution_l1811_181168


namespace sine_shift_equivalence_l1811_181135

theorem sine_shift_equivalence :
  ∀ x : ℝ, Real.sin (2 * x + π / 6) = Real.sin (2 * (x + π / 4) - π / 3) :=
by sorry

end sine_shift_equivalence_l1811_181135


namespace units_digit_of_150_factorial_l1811_181141

theorem units_digit_of_150_factorial (n : ℕ) : n = 150 → n.factorial % 10 = 0 := by
  sorry

end units_digit_of_150_factorial_l1811_181141


namespace segment_length_ratio_l1811_181148

/-- Given a line segment AD with points B and C on it, prove that BC + DE = 5/8 * AD -/
theorem segment_length_ratio (A B C D E : ℝ) : 
  B ∈ Set.Icc A D → -- B is on segment AD
  C ∈ Set.Icc A D → -- C is on segment AD
  B - A = 3 * (D - B) → -- AB = 3 * BD
  C - A = 7 * (D - C) → -- AC = 7 * CD
  E - D = C - B → -- DE = BC
  E - A = D - E → -- E is midpoint of AD
  C - B + E - D = 5/8 * (D - A) := by sorry

end segment_length_ratio_l1811_181148


namespace solve_head_circumference_problem_l1811_181122

def head_circumference_problem (jack_circumference charlie_circumference bill_circumference : ℝ) : Prop :=
  jack_circumference = 12 ∧
  bill_circumference = 10 ∧
  bill_circumference = (2/3) * charlie_circumference ∧
  ∃ x, charlie_circumference = (1/2) * jack_circumference + x ∧
  x = 9

theorem solve_head_circumference_problem :
  ∀ jack_circumference charlie_circumference bill_circumference,
  head_circumference_problem jack_circumference charlie_circumference bill_circumference :=
by
  sorry

end solve_head_circumference_problem_l1811_181122


namespace jungkook_smallest_l1811_181163

def yoongi_collection : ℕ := 4
def jungkook_collection : ℚ := 6 / 3
def yuna_collection : ℕ := 5

theorem jungkook_smallest :
  jungkook_collection < yoongi_collection ∧ jungkook_collection < yuna_collection :=
sorry

end jungkook_smallest_l1811_181163


namespace curve_tangent_product_l1811_181113

/-- Given a curve y = ax³ + bx where the point (2, 2) lies on the curve
    and the slope of the tangent line at this point is 9,
    prove that the product ab equals -3. -/
theorem curve_tangent_product (a b : ℝ) : 
  (2 : ℝ) = a * (2 : ℝ)^3 + b * (2 : ℝ) → -- Point (2, 2) lies on the curve
  (9 : ℝ) = 3 * a * (2 : ℝ)^2 + b →       -- Slope of tangent at (2, 2) is 9
  a * b = -3 := by
sorry

end curve_tangent_product_l1811_181113


namespace book_count_proof_l1811_181162

/-- Proves that given a total of 144 books and a ratio of 7:5 for storybooks to science books,
    the number of storybooks is 84 and the number of science books is 60. -/
theorem book_count_proof (total : ℕ) (storybook_ratio : ℕ) (science_ratio : ℕ)
    (h_total : total = 144)
    (h_ratio : (storybook_ratio : ℚ) / (science_ratio : ℚ) = 7 / 5) :
    ∃ (storybooks science_books : ℕ),
      storybooks = 84 ∧
      science_books = 60 ∧
      storybooks + science_books = total ∧
      (storybooks : ℚ) / (science_books : ℚ) = storybook_ratio / science_ratio :=
by
  sorry

end book_count_proof_l1811_181162


namespace max_sum_of_factors_l1811_181114

theorem max_sum_of_factors (A B C : ℕ+) : 
  A ≠ B ∧ B ≠ C ∧ A ≠ C →
  A * B * C = 3003 →
  A + B + C ≤ 49 :=
by sorry

end max_sum_of_factors_l1811_181114


namespace complementary_angles_can_be_both_acute_l1811_181104

-- Define what it means for two angles to be complementary
def complementary (a b : ℝ) : Prop := a + b = 90

-- Define what it means for an angle to be acute
def acute (a : ℝ) : Prop := 0 < a ∧ a < 90

theorem complementary_angles_can_be_both_acute :
  ∃ (a b : ℝ), complementary a b ∧ acute a ∧ acute b :=
sorry

end complementary_angles_can_be_both_acute_l1811_181104


namespace equation_solution_l1811_181139

theorem equation_solution : ∃ x : ℚ, (5 * x + 12 * x = 540 - 12 * (x - 5)) ∧ (x = 600 / 29) := by
  sorry

end equation_solution_l1811_181139


namespace waiting_room_problem_l1811_181158

theorem waiting_room_problem (initial_waiting : ℕ) (interview_room : ℕ) : 
  initial_waiting = 22 → interview_room = 5 → 
  ∃ (additional : ℕ), initial_waiting + additional = 5 * interview_room ∧ additional = 3 :=
by
  sorry

end waiting_room_problem_l1811_181158


namespace smallest_a_inequality_two_ninths_satisfies_inequality_l1811_181119

theorem smallest_a_inequality (x y z : ℝ) (hx : x ≥ 0) (hy : y ≥ 0) (hz : z ≥ 0) (hsum : x + y + z = 1) :
  ∀ a : ℝ, (∀ x y z : ℝ, x ≥ 0 → y ≥ 0 → z ≥ 0 → x + y + z = 1 → a * (x^2 + y^2 + z^2) + x*y*z ≥ 1/3) →
  a ≥ 2/9 :=
by sorry

theorem two_ninths_satisfies_inequality (x y z : ℝ) (hx : x ≥ 0) (hy : y ≥ 0) (hz : z ≥ 0) (hsum : x + y + z = 1) :
  (2/9 : ℝ) * (x^2 + y^2 + z^2) + x*y*z ≥ 1/3 :=
by sorry

end smallest_a_inequality_two_ninths_satisfies_inequality_l1811_181119


namespace extended_inequality_l1811_181149

theorem extended_inequality (n k : ℕ) (h1 : n ≥ 3) (h2 : 1 ≤ k) (h3 : k ≤ n) :
  2^n + 5^n > 2^(n-k) * 5^k + 2^k * 5^(n-k) := by
  sorry

end extended_inequality_l1811_181149


namespace light_travel_distance_l1811_181183

/-- The distance light travels in one year in kilometers. -/
def light_year_distance : ℝ := 9460800000000

/-- The number of years we want to calculate the light travel distance for. -/
def years : ℕ := 50

/-- Theorem stating the distance light travels in 50 years. -/
theorem light_travel_distance : 
  (light_year_distance * years : ℝ) = 4.7304e14 := by sorry

end light_travel_distance_l1811_181183


namespace max_value_of_product_sum_l1811_181180

theorem max_value_of_product_sum (w x y z : ℝ) 
  (nonneg_w : w ≥ 0) (nonneg_x : x ≥ 0) (nonneg_y : y ≥ 0) (nonneg_z : z ≥ 0)
  (sum_eq_200 : w + x + y + z = 200) : 
  wx + xy + yz ≤ 10000 := by
sorry

end max_value_of_product_sum_l1811_181180


namespace range_of_a_inequality_proof_l1811_181196

-- Question 1
theorem range_of_a (a : ℝ) :
  (∀ x ∈ Set.Icc (1/2 : ℝ) 1, |x - a| + |2*x - 1| ≤ |2*x + 1|) →
  a ∈ Set.Icc (-1 : ℝ) (5/2) :=
sorry

-- Question 2
theorem inequality_proof (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  a^2 * b^2 + a^2 + b^2 ≥ a * b * (a + b + 1) :=
sorry

end range_of_a_inequality_proof_l1811_181196


namespace sum_in_base_9_l1811_181192

/-- Converts a base-9 number to base-10 --/
def base9To10 (digits : List Nat) : Nat :=
  digits.enum.foldr (fun (i, d) acc => acc + d * 9^i) 0

/-- Converts a base-10 number to base-9 --/
def base10To9 (n : Nat) : List Nat :=
  if n = 0 then [0] else
  let rec aux (m : Nat) (acc : List Nat) : List Nat :=
    if m = 0 then acc else aux (m / 9) ((m % 9) :: acc)
  aux n []

/-- The sum of 263₉, 504₉, and 72₉ in base 9 is 850₉ --/
theorem sum_in_base_9 :
  base10To9 (base9To10 [3, 6, 2] + base9To10 [4, 0, 5] + base9To10 [2, 7]) = [0, 5, 8] :=
by sorry

end sum_in_base_9_l1811_181192


namespace adrianna_gum_count_l1811_181130

/-- Calculates the remaining gum count for Adrianna --/
def remaining_gum (initial_gum : ℕ) (additional_gum : ℕ) (friends_given_gum : ℕ) : ℕ :=
  initial_gum + additional_gum - friends_given_gum

/-- Theorem stating that Adrianna has 2 pieces of gum left --/
theorem adrianna_gum_count :
  remaining_gum 10 3 11 = 2 := by
  sorry

end adrianna_gum_count_l1811_181130


namespace expand_expression_l1811_181172

theorem expand_expression (x : ℝ) : 3 * (8 * x^2 - 2 * x + 1) = 24 * x^2 - 6 * x + 3 := by
  sorry

end expand_expression_l1811_181172


namespace inequality_condition_l1811_181170

theorem inequality_condition (a : ℝ) : 
  (∀ x > 1, (Real.exp x) / (x^3) - x - a * Real.log x ≥ 1) ↔ a ≤ -3 := by
  sorry

end inequality_condition_l1811_181170


namespace missed_number_l1811_181105

theorem missed_number (n : ℕ) (incorrect_sum correct_sum missed_number : ℕ) :
  n > 0 →
  incorrect_sum = 575 →
  correct_sum = n * (n + 1) / 2 →
  correct_sum = 595 →
  incorrect_sum + missed_number = correct_sum →
  missed_number = 20 := by
  sorry

end missed_number_l1811_181105


namespace sequence_property_l1811_181128

/-- The function generating the sequence -/
def f (n : ℕ) : ℕ := 2 * (n + 1)^2 * (n + 2)^2

/-- Predicate to check if a number is the sum of two square integers -/
def isSumOfTwoSquares (m : ℕ) : Prop := ∃ a b : ℕ, m = a^2 + b^2

theorem sequence_property :
  (∀ n : ℕ, f n < f (n + 1)) ∧
  (∀ n : ℕ, isSumOfTwoSquares (f n)) ∧
  f 1 = 72 ∧ f 2 = 288 ∧ f 3 = 800 :=
sorry

end sequence_property_l1811_181128


namespace integer_in_range_l1811_181103

theorem integer_in_range : ∃ x : ℤ, -Real.sqrt 2 < x ∧ x < Real.sqrt 5 :=
by
  -- The proof goes here
  sorry

end integer_in_range_l1811_181103


namespace current_average_is_53_l1811_181188

/-- Represents a cricket player's batting statistics -/
structure CricketStats where
  matchesPlayed : ℕ
  totalRuns : ℕ

/-- Calculates the batting average -/
def battingAverage (stats : CricketStats) : ℚ :=
  stats.totalRuns / stats.matchesPlayed

/-- Theorem: If a player's average becomes 58 after scoring 78 in the 5th match,
    then their current average after 4 matches is 53 -/
theorem current_average_is_53
  (player : CricketStats)
  (h1 : player.matchesPlayed = 4)
  (h2 : battingAverage ⟨5, player.totalRuns + 78⟩ = 58) :
  battingAverage player = 53 := by
  sorry

end current_average_is_53_l1811_181188


namespace stan_playlist_sufficient_stan_playlist_sufficient_proof_l1811_181132

theorem stan_playlist_sufficient (total_run_time : ℕ) 
  (songs_3min songs_4min songs_6min : ℕ) 
  (max_songs_per_category : ℕ) 
  (min_favorite_songs : ℕ) 
  (favorite_song_length : ℕ) : Prop :=
  total_run_time = 90 ∧
  songs_3min ≥ 10 ∧
  songs_4min ≥ 12 ∧
  songs_6min ≥ 15 ∧
  max_songs_per_category = 7 ∧
  min_favorite_songs = 3 ∧
  favorite_song_length = 4 →
  ∃ (playlist_3min playlist_4min playlist_6min : ℕ),
    playlist_3min ≤ max_songs_per_category ∧
    playlist_4min ≤ max_songs_per_category ∧
    playlist_6min ≤ max_songs_per_category ∧
    playlist_4min ≥ min_favorite_songs ∧
    playlist_3min * 3 + playlist_4min * 4 + playlist_6min * 6 ≥ total_run_time

theorem stan_playlist_sufficient_proof : stan_playlist_sufficient 90 10 12 15 7 3 4 := by
  sorry

end stan_playlist_sufficient_stan_playlist_sufficient_proof_l1811_181132


namespace transformed_stddev_l1811_181107

variable {n : ℕ}
variable (a : Fin n → ℝ)
variable (S : ℝ)

def variance (x : Fin n → ℝ) : ℝ := sorry

def stdDev (x : Fin n → ℝ) : ℝ := sorry

theorem transformed_stddev 
  (h : variance a = S^2) : 
  stdDev (fun i => 2 * a i - 3) = 2 * S := by sorry

end transformed_stddev_l1811_181107


namespace product_of_first_three_odd_numbers_l1811_181137

theorem product_of_first_three_odd_numbers : 
  (∀ a b c : ℕ, a * b * c = 38 → a = 3 ∧ b = 5 ∧ c = 7) →
  (∀ x y z : ℕ, x * y * z = 268 → x = 13 ∧ y = 15 ∧ z = 17) →
  1 * 3 * 5 = 15 :=
by sorry

end product_of_first_three_odd_numbers_l1811_181137


namespace joan_clothing_expenses_l1811_181102

theorem joan_clothing_expenses : 
  15 + 14.82 + 12.51 = 42.33 := by sorry

end joan_clothing_expenses_l1811_181102


namespace equal_digits_probability_l1811_181179

/-- The number of sides on each die -/
def num_sides : ℕ := 20

/-- The number of dice rolled -/
def num_dice : ℕ := 6

/-- The number of one-digit outcomes on a die -/
def one_digit_outcomes : ℕ := 9

/-- The number of two-digit outcomes on a die -/
def two_digit_outcomes : ℕ := 11

/-- The probability of rolling an equal number of one-digit and two-digit numbers on 6 20-sided dice -/
def equal_digits_prob : ℚ := 4851495 / 16000000

theorem equal_digits_probability : 
  let p_one_digit := one_digit_outcomes / num_sides
  let p_two_digit := two_digit_outcomes / num_sides
  let combinations := Nat.choose num_dice (num_dice / 2)
  combinations * (p_one_digit ^ (num_dice / 2)) * (p_two_digit ^ (num_dice / 2)) = equal_digits_prob := by
  sorry

end equal_digits_probability_l1811_181179


namespace refrigerator_price_l1811_181173

/-- The price Ramesh paid for the refrigerator --/
def price_paid (P : ℝ) : ℝ := 0.80 * P + 375

/-- The theorem stating the price Ramesh paid for the refrigerator --/
theorem refrigerator_price :
  ∃ P : ℝ,
    (1.12 * P = 17920) ∧
    (price_paid P = 13175) := by
  sorry

end refrigerator_price_l1811_181173


namespace domain_of_composite_function_l1811_181120

-- Define the function f with domain (1,3)
def f : Set ℝ := Set.Ioo 1 3

-- Define the composite function g(x) = f(2x-1)
def g (x : ℝ) : ℝ := 2 * x - 1

-- Theorem statement
theorem domain_of_composite_function :
  {x : ℝ | g x ∈ f} = Set.Ioo 1 2 := by sorry

end domain_of_composite_function_l1811_181120


namespace cubic_function_property_l1811_181185

/-- A cubic function g(x) = Ax³ + Bx² - Cx + D -/
def g (A B C D : ℝ) (x : ℝ) : ℝ := A * x^3 + B * x^2 - C * x + D

theorem cubic_function_property (A B C D : ℝ) :
  g A B C D 2 = 5 ∧ g A B C D (-1) = -8 ∧ g A B C D 0 = 2 →
  -12*A + 6*B - 3*C + D = 27.5 := by
  sorry

end cubic_function_property_l1811_181185


namespace solution_is_negative_two_l1811_181153

/-- The equation we want to solve -/
def equation (x : ℝ) : Prop := 2 / x = 1 / (x + 1)

/-- The theorem stating that -2 is the solution to the equation -/
theorem solution_is_negative_two : ∃ x : ℝ, equation x ∧ x = -2 := by
  sorry

end solution_is_negative_two_l1811_181153


namespace hector_siblings_product_l1811_181197

/-- A family where one member has 4 sisters and 7 brothers -/
structure Family :=
  (sisters_of_helen : ℕ)
  (brothers_of_helen : ℕ)
  (helen_is_female : Bool)
  (hector_is_male : Bool)

/-- The number of sisters Hector has in the family -/
def sisters_of_hector (f : Family) : ℕ :=
  f.sisters_of_helen + (if f.helen_is_female then 1 else 0)

/-- The number of brothers Hector has in the family -/
def brothers_of_hector (f : Family) : ℕ :=
  f.brothers_of_helen - 1

theorem hector_siblings_product (f : Family) 
  (h1 : f.sisters_of_helen = 4)
  (h2 : f.brothers_of_helen = 7)
  (h3 : f.helen_is_female = true)
  (h4 : f.hector_is_male = true) :
  (sisters_of_hector f) * (brothers_of_hector f) = 30 :=
sorry

end hector_siblings_product_l1811_181197


namespace seven_power_plus_one_prime_factors_l1811_181190

theorem seven_power_plus_one_prime_factors (n : ℕ) :
  ∃ (primes : Finset ℕ), 
    (∀ p ∈ primes, Nat.Prime p) ∧ 
    primes.card = 2 * n + 3 ∧
    (primes.prod id = 7^(7^(7^(7^2))) + 1) := by
  sorry

end seven_power_plus_one_prime_factors_l1811_181190


namespace R_sufficient_not_necessary_for_Q_l1811_181167

-- Define the propositions
variable (R P Q S : Prop)

-- Define the given conditions
axiom S_necessary_for_R : R → S
axiom S_sufficient_for_P : S → P
axiom Q_necessary_for_P : P → Q
axiom Q_sufficient_for_S : Q → S

-- Theorem to prove
theorem R_sufficient_not_necessary_for_Q :
  (R → Q) ∧ ¬(Q → R) :=
sorry

end R_sufficient_not_necessary_for_Q_l1811_181167


namespace negative_fractions_in_list_l1811_181142

def given_numbers : List ℚ := [5, -1, 0, -6, 125.73, 0.3, -3.5, -0.72, 5.25]

def is_negative_fraction (x : ℚ) : Prop := x < 0 ∧ x ≠ ⌊x⌋

theorem negative_fractions_in_list :
  ∀ x ∈ given_numbers, is_negative_fraction x ↔ x = -3.5 ∨ x = -0.72 := by
  sorry

end negative_fractions_in_list_l1811_181142


namespace correct_average_l1811_181108

theorem correct_average (n : ℕ) (initial_avg : ℚ) 
  (correct_numbers incorrect_numbers : List ℚ) :
  n = 15 ∧ 
  initial_avg = 25 ∧ 
  correct_numbers = [86, 92, 48] ∧ 
  incorrect_numbers = [26, 62, 24] →
  (n * initial_avg + (correct_numbers.sum - incorrect_numbers.sum)) / n = 32.6 := by
  sorry

end correct_average_l1811_181108
