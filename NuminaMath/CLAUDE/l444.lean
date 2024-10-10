import Mathlib

namespace pamphlet_printing_speed_ratio_l444_44449

theorem pamphlet_printing_speed_ratio : 
  ∀ (mike_speed : ℕ) (mike_hours_before_break : ℕ) (mike_hours_after_break : ℕ) 
    (leo_speed_multiplier : ℕ) (total_pamphlets : ℕ),
  mike_speed = 600 →
  mike_hours_before_break = 9 →
  mike_hours_after_break = 2 →
  total_pamphlets = 9400 →
  (mike_speed * mike_hours_before_break + 
   (mike_speed / 3) * mike_hours_after_break + 
   (leo_speed_multiplier * mike_speed) * (mike_hours_before_break / 3) = total_pamphlets) →
  leo_speed_multiplier = 2 := by
sorry

end pamphlet_printing_speed_ratio_l444_44449


namespace ball_prices_theorem_l444_44488

/-- Represents the prices and quantities of soccer balls and volleyballs -/
structure BallPrices where
  soccer_price : ℝ
  volleyball_price : ℝ
  total_balls : ℕ
  max_cost : ℝ

/-- The conditions given in the problem -/
def satisfies_conditions (bp : BallPrices) : Prop :=
  bp.soccer_price = bp.volleyball_price + 15 ∧
  480 / bp.soccer_price = 390 / bp.volleyball_price ∧
  bp.total_balls = 100

/-- The theorem to be proven -/
theorem ball_prices_theorem (bp : BallPrices) 
  (h : satisfies_conditions bp) : 
  bp.soccer_price = 80 ∧ 
  bp.volleyball_price = 65 ∧ 
  ∃ (m : ℕ), m ≤ bp.total_balls ∧ 
             m * bp.soccer_price + (bp.total_balls - m) * bp.volleyball_price ≤ bp.max_cost ∧
             ∀ (n : ℕ), n > m → 
               n * bp.soccer_price + (bp.total_balls - n) * bp.volleyball_price > bp.max_cost :=
by
  sorry

end ball_prices_theorem_l444_44488


namespace intersection_point_l444_44492

-- Define the linear functions
def f1 (a b x : ℝ) : ℝ := a * x + b + 3
def f2 (a b x : ℝ) : ℝ := -b * x + a - 2
def f3 (x : ℝ) : ℝ := 2 * x - 8

-- State the theorem
theorem intersection_point (a b : ℝ) :
  (∃ y, f1 a b 0 = f2 a b 0 ∧ y = f1 a b 0) ∧  -- First and second functions intersect on y-axis
  (∃ x, f2 a b x = f3 x ∧ f2 a b x = 0) →      -- Second and third functions intersect on x-axis
  (∃ x y, f1 a b x = f3 x ∧ y = f1 a b x ∧ x = -3 ∧ y = -14) := by
  sorry

end intersection_point_l444_44492


namespace solve_exponential_equation_l444_44414

theorem solve_exponential_equation :
  ∃ n : ℕ, 16^n * 16^n * 16^n * 16^n = 256^4 ∧ n = 2 := by
sorry

end solve_exponential_equation_l444_44414


namespace parabola_and_bisector_intercept_l444_44434

-- Define the line l
def line_l (x y : ℝ) : Prop := y = (1/2) * (x + 4)

-- Define the parabola G
def parabola_G (p : ℝ) (x y : ℝ) : Prop := x^2 = 2 * p * y

-- Define the intersection points B and C
def intersection_points (xB yB xC yC : ℝ) : Prop :=
  line_l xB yB ∧ line_l xC yC ∧ 
  parabola_G 2 xB yB ∧ parabola_G 2 xC yC

-- Define the midpoint of BC
def midpoint_BC (x y : ℝ) : Prop :=
  ∃ xB yB xC yC, intersection_points xB yB xC yC ∧
  x = (xB + xC) / 2 ∧ y = (yB + yC) / 2

-- Define the perpendicular bisector of BC
def perp_bisector (x y : ℝ) : Prop :=
  ∃ x0 y0, midpoint_BC x0 y0 ∧ y - y0 = -2 * (x - x0)

-- Theorem statement
theorem parabola_and_bisector_intercept :
  (∃ p : ℝ, p > 0 ∧ ∀ x y, parabola_G p x y ↔ x^2 = 4 * y) ∧
  (∃ b : ℝ, b = 9/2 ∧ perp_bisector 0 b) ∧
  (∃ x, x = 1 ∧ midpoint_BC x ((1/2) * (x + 4))) :=
sorry

end parabola_and_bisector_intercept_l444_44434


namespace ellipse_equation_l444_44489

/-- Given an ellipse C with equation x²/a² + y²/b² = 1 where a > b > 0,
    foci at (-2, 0) and (2, 0), and the product of slopes of lines from
    the left vertex to the intersection points of the ellipse with the
    circle having diameter F₁F₂ being 1/3, prove that the standard
    equation of the ellipse C is x²/6 + y²/2 = 1. -/
theorem ellipse_equation (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a > b) :
  (∃ (x y : ℝ → ℝ), ∀ t, (x t)^2 / a^2 + (y t)^2 / b^2 = 1) →
  (∃ (F₁ F₂ : ℝ × ℝ), F₁ = (-2, 0) ∧ F₂ = (2, 0)) →
  (∃ (M N : ℝ × ℝ), M.1 > 0 ∧ M.2 > 0 ∧ N.1 < 0 ∧ N.2 > 0) →
  (∃ (A : ℝ × ℝ), A = (-a, 0)) →
  (∃ (m₁ m₂ : ℝ), m₁ * m₂ = 1/3) →
  (x^2 / 6 + y^2 / 2 = 1) :=
sorry

end ellipse_equation_l444_44489


namespace lizard_eyes_count_l444_44457

theorem lizard_eyes_count :
  ∀ (E W S : ℕ),
  W = 3 * E →
  S = 7 * W →
  E = S + W - 69 →
  E = 3 :=
by
  sorry

end lizard_eyes_count_l444_44457


namespace diagonal_increase_l444_44495

/-- The number of diagonals in a convex polygon with n sides -/
def num_diagonals (n : ℕ) : ℕ := sorry

theorem diagonal_increase (n : ℕ) :
  num_diagonals (n + 1) = num_diagonals n + n - 1 :=
by sorry

end diagonal_increase_l444_44495


namespace cookie_jar_spending_l444_44479

theorem cookie_jar_spending (initial_amount : ℝ) (final_amount : ℝ) (doris_spent : ℝ) :
  initial_amount = 24 →
  final_amount = 15 →
  initial_amount - (doris_spent + doris_spent / 2) = final_amount →
  doris_spent = 6 := by
sorry

end cookie_jar_spending_l444_44479


namespace complex_equation_l444_44499

theorem complex_equation (z : ℂ) (h : z = 1 + I) : z^2 + 2/z = 1 + I := by
  sorry

end complex_equation_l444_44499


namespace big_eighteen_games_l444_44413

/-- Represents a basketball conference with the given structure -/
structure BasketballConference where
  num_divisions : Nat
  teams_per_division : Nat
  intra_division_games : Nat
  inter_division_games : Nat

/-- Calculates the total number of conference games scheduled -/
def total_games (conf : BasketballConference) : Nat :=
  let total_teams := conf.num_divisions * conf.teams_per_division
  let games_per_team := (conf.teams_per_division - 1) * conf.intra_division_games + 
                        (total_teams - conf.teams_per_division) * conf.inter_division_games
  total_teams * games_per_team / 2

/-- The Big Eighteen Basketball Conference -/
def big_eighteen : BasketballConference :=
  { num_divisions := 3
  , teams_per_division := 6
  , intra_division_games := 3
  , inter_division_games := 1 }

theorem big_eighteen_games : total_games big_eighteen = 243 := by
  sorry

end big_eighteen_games_l444_44413


namespace function_domain_constraint_l444_44462

theorem function_domain_constraint (f : ℝ → ℝ) (a : ℝ) : 
  (∀ x : ℝ, f x = (x - 7)^(1/3) / (a * x^2 + 4 * a * x + 3)) →
  (∀ x : ℝ, f x ≠ 0) →
  (0 < a ∧ a < 3/4) :=
sorry

end function_domain_constraint_l444_44462


namespace at_least_one_quadratic_has_solution_l444_44442

theorem at_least_one_quadratic_has_solution (a b c : ℝ) : 
  ∃ x : ℝ, (x^2 + (a - b)*x + (b - c) = 0) ∨ 
            (x^2 + (b - c)*x + (c - a) = 0) ∨ 
            (x^2 + (c - a)*x + (a - b) = 0) := by
  sorry

end at_least_one_quadratic_has_solution_l444_44442


namespace louis_suit_cost_is_141_l444_44409

/-- The cost of Louis's velvet suit materials -/
def louis_suit_cost (fabric_price_per_yard : ℝ) (pattern_price : ℝ) (thread_price_per_spool : ℝ) 
  (fabric_yards : ℝ) (thread_spools : ℕ) : ℝ :=
  fabric_price_per_yard * fabric_yards + pattern_price + thread_price_per_spool * thread_spools

/-- Theorem: The total cost of Louis's suit materials is $141 -/
theorem louis_suit_cost_is_141 : 
  louis_suit_cost 24 15 3 5 2 = 141 := by
  sorry

end louis_suit_cost_is_141_l444_44409


namespace sum_of_x_values_l444_44445

theorem sum_of_x_values (x : ℝ) : 
  (50 < x ∧ x < 150) →
  (Real.cos (2 * x * π / 180))^3 + (Real.cos (6 * x * π / 180))^3 = 
    8 * (Real.cos (4 * x * π / 180))^3 * (Real.cos (x * π / 180))^3 →
  ∃ (s : Finset ℝ), (∀ y ∈ s, 
    (50 < y ∧ y < 150) ∧
    (Real.cos (2 * y * π / 180))^3 + (Real.cos (6 * y * π / 180))^3 = 
      8 * (Real.cos (4 * y * π / 180))^3 * (Real.cos (y * π / 180))^3) ∧
  (s.sum id = 270) := by
  sorry

end sum_of_x_values_l444_44445


namespace seed_germination_percentage_l444_44407

theorem seed_germination_percentage (seeds_plot1 seeds_plot2 : ℕ) 
  (germination_rate1 germination_rate2 : ℚ) :
  seeds_plot1 = 300 →
  seeds_plot2 = 200 →
  germination_rate1 = 30 / 100 →
  germination_rate2 = 35 / 100 →
  (((seeds_plot1 : ℚ) * germination_rate1 + (seeds_plot2 : ℚ) * germination_rate2) / 
   ((seeds_plot1 : ℚ) + (seeds_plot2 : ℚ))) * 100 = 32 := by
  sorry

end seed_germination_percentage_l444_44407


namespace class_average_score_class_average_is_85_l444_44408

theorem class_average_score : ℕ → ℕ → ℕ → ℕ → ℕ → ℕ
  | total_students, students_score_92, students_score_80, students_score_70, score_70 =>
    let total_score := students_score_92 * 92 + students_score_80 * 80 + students_score_70 * score_70
    total_score / total_students

theorem class_average_is_85 :
  class_average_score 10 5 4 1 70 = 85 := by
  sorry

end class_average_score_class_average_is_85_l444_44408


namespace meeting_probability_in_our_tournament_l444_44483

/-- Represents a knockout tournament --/
structure KnockoutTournament where
  total_players : Nat
  num_rounds : Nat
  random_pairing : Bool
  equal_win_chance : Bool

/-- The probability of two specific players meeting in a tournament --/
def meeting_probability (t : KnockoutTournament) : Rat :=
  sorry

/-- Our specific tournament --/
def our_tournament : KnockoutTournament :=
  { total_players := 32
  , num_rounds := 5
  , random_pairing := true
  , equal_win_chance := true }

theorem meeting_probability_in_our_tournament :
  meeting_probability our_tournament = 11097 / 167040 := by
  sorry

end meeting_probability_in_our_tournament_l444_44483


namespace perpendicular_lines_slope_l444_44435

theorem perpendicular_lines_slope (a : ℝ) : 
  (∃ (x y : ℝ), y = a * x - 2) ∧ 
  (∃ (x y : ℝ), y = (a + 2) * x + 1) ∧ 
  (a * (a + 2) = -1) → 
  a = -1 := by
sorry

end perpendicular_lines_slope_l444_44435


namespace john_used_16_bulbs_l444_44471

/-- The number of light bulbs John used -/
def bulbs_used : ℕ := sorry

/-- The initial number of light bulbs -/
def initial_bulbs : ℕ := 40

/-- The number of light bulbs John has left after giving some away -/
def remaining_bulbs : ℕ := 12

theorem john_used_16_bulbs : 
  bulbs_used = 16 ∧ 
  (initial_bulbs - bulbs_used) / 2 = remaining_bulbs :=
sorry

end john_used_16_bulbs_l444_44471


namespace prop_one_prop_two_prop_three_prop_four_l444_44498

-- Define a real-valued function
variable (f : ℝ → ℝ)

-- Define symmetry about x = 1
def symmetric_about_one (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (1 + x) = f (1 - x)

-- Proposition 1
theorem prop_one (h : ∀ x : ℝ, f (1 + 2*x) = f (1 - 2*x)) : 
  symmetric_about_one f := sorry

-- Proposition 2
theorem prop_two : 
  (∀ x : ℝ, f (x - 1) = f (1 - x)) → symmetric_about_one f := sorry

-- Proposition 3
theorem prop_three (h1 : ∀ x : ℝ, f x = f (-x)) 
  (h2 : ∀ x : ℝ, f (1 + x) = -f x) : symmetric_about_one f := sorry

-- Proposition 4
theorem prop_four (h1 : ∀ x : ℝ, f x = -f (-x)) 
  (h2 : ∀ x : ℝ, f x = f (-x - 2)) : symmetric_about_one f := sorry

end prop_one_prop_two_prop_three_prop_four_l444_44498


namespace sum_of_digits_of_product_plus_constant_l444_44469

def repeat_digit (d : Nat) (n : Nat) : Nat :=
  d * (10^n - 1) / 9

def sum_of_digits (n : Nat) : Nat :=
  if n = 0 then 0 else (n % 10) + sum_of_digits (n / 10)

theorem sum_of_digits_of_product_plus_constant : 
  sum_of_digits ((repeat_digit 9 47 * repeat_digit 4 47) + 100000) = 424 := by
  sorry

end sum_of_digits_of_product_plus_constant_l444_44469


namespace custom_mult_equation_solution_l444_44412

-- Define the custom operation
def customMult (a b : ℝ) : ℝ := 4 * a * b

-- State the theorem
theorem custom_mult_equation_solution :
  ∀ x : ℝ, (customMult x x) + (customMult 2 x) - (customMult 2 4) = 0 → x = 2 ∨ x = -4 := by
  sorry

end custom_mult_equation_solution_l444_44412


namespace charity_box_distribution_l444_44402

/-- The charity organization's box distribution problem -/
theorem charity_box_distribution
  (box_cost : ℕ)
  (donation_multiplier : ℕ)
  (total_boxes : ℕ)
  (h1 : box_cost = 245)
  (h2 : donation_multiplier = 4)
  (h3 : total_boxes = 2000) :
  ∃ (initial_boxes : ℕ),
    initial_boxes * box_cost * (1 + donation_multiplier) = total_boxes * box_cost ∧
    initial_boxes = 400 := by
  sorry

end charity_box_distribution_l444_44402


namespace fraction_simplification_l444_44480

theorem fraction_simplification (x : ℝ) (h : 2 * x ≠ 2) :
  (6 * x^3 + 13 * x^2 + 15 * x - 25) / (2 * x^3 + 4 * x^2 + 4 * x - 10) = (6 * x - 5) / (2 * x - 2) :=
by sorry

end fraction_simplification_l444_44480


namespace second_shot_probability_l444_44454

theorem second_shot_probability 
  (p_first : ℝ) 
  (p_consecutive : ℝ) 
  (h1 : p_first = 0.75) 
  (h2 : p_consecutive = 0.6) : 
  p_consecutive / p_first = 0.8 := by
  sorry

end second_shot_probability_l444_44454


namespace range_of_a_l444_44423

-- Define the equation and its roots
def equation (m : ℝ) (x : ℝ) : Prop := x^2 - m*x - 2 = 0

-- Define the inequality condition for a
def inequality_condition (a m : ℝ) (x₁ x₂ : ℝ) : Prop :=
  a^2 - 5*a - 3 ≥ |x₁ - x₂|

-- Define the range for m
def m_range (m : ℝ) : Prop := -1 ≤ m ∧ m ≤ 1

-- Define the condition for the quadratic inequality having no solutions
def no_solutions (a : ℝ) : Prop :=
  ∀ x : ℝ, a*x^2 + 2*x - 1 ≤ 0

theorem range_of_a :
  ∀ m : ℝ, m_range m →
  ∀ x₁ x₂ : ℝ, equation m x₁ ∧ equation m x₂ ∧ x₁ ≠ x₂ →
  ∀ a : ℝ, (∀ m : ℝ, m_range m → inequality_condition a m x₁ x₂) ∧ no_solutions a →
  a ≤ -1 :=
sorry

end range_of_a_l444_44423


namespace radii_of_circles_l444_44477

/-- Two circles lying outside each other -/
structure TwoCircles where
  center_distance : ℝ
  external_tangent : ℝ
  internal_tangent : ℝ

/-- The radii of two circles -/
structure CircleRadii where
  r₁ : ℝ
  r₂ : ℝ

/-- Given the properties of two circles, compute their radii -/
def compute_radii (circles : TwoCircles) : CircleRadii :=
  { r₁ := 38, r₂ := 22 }

/-- Theorem stating that for the given circle properties, the radii are 38 and 22 -/
theorem radii_of_circles (circles : TwoCircles) 
    (h1 : circles.center_distance = 65) 
    (h2 : circles.external_tangent = 63) 
    (h3 : circles.internal_tangent = 25) : 
    compute_radii circles = { r₁ := 38, r₂ := 22 } := by
  sorry

end radii_of_circles_l444_44477


namespace sector_angle_and_area_l444_44490

/-- Given a sector with radius 8 and arc length 12, prove its central angle and area -/
theorem sector_angle_and_area :
  let r : ℝ := 8
  let l : ℝ := 12
  let α : ℝ := l / r
  let S : ℝ := (1 / 2) * l * r
  α = 3 / 2 ∧ S = 48 := by
  sorry

end sector_angle_and_area_l444_44490


namespace cubic_function_extremum_l444_44468

/-- Given a cubic function f with a local extremum at x = -1 -/
def f (a b : ℝ) (x : ℝ) : ℝ := x^3 + 3*a*x^2 + b*x + a^2

/-- The derivative of f -/
def f' (a b : ℝ) (x : ℝ) : ℝ := 3*x^2 + 6*a*x + b

theorem cubic_function_extremum (a b : ℝ) (h1 : a > 1) 
  (h2 : f a b (-1) = 0) (h3 : f' a b (-1) = 0) :
  a = 2 ∧ b = 9 ∧ 
  (∀ x ∈ Set.Icc (-4 : ℝ) 0, 0 ≤ f a b x ∧ f a b x ≤ 4) ∧
  (∃ x ∈ Set.Icc (-4 : ℝ) 0, f a b x = 0) ∧
  (∃ x ∈ Set.Icc (-4 : ℝ) 0, f a b x = 4) := by
  sorry

#check cubic_function_extremum

end cubic_function_extremum_l444_44468


namespace apple_tripling_theorem_l444_44404

theorem apple_tripling_theorem (a b c : ℕ) :
  (3 * a + b + c = (17/10) * (a + b + c)) →
  (a + 3 * b + c = (3/2) * (a + b + c)) →
  (a + b + 3 * c = (9/5) * (a + b + c)) :=
by sorry

end apple_tripling_theorem_l444_44404


namespace magnified_diameter_is_five_l444_44482

/-- The magnification factor of an electron microscope. -/
def magnification : ℝ := 1000

/-- The actual diameter of the tissue in centimeters. -/
def actual_diameter : ℝ := 0.005

/-- The diameter of the magnified image in centimeters. -/
def magnified_diameter : ℝ := actual_diameter * magnification

/-- Theorem stating that the diameter of the magnified image is 5 centimeters. -/
theorem magnified_diameter_is_five :
  magnified_diameter = 5 := by sorry

end magnified_diameter_is_five_l444_44482


namespace alpha_plus_beta_equals_112_l444_44484

theorem alpha_plus_beta_equals_112 
  (α β : ℝ) 
  (h : ∀ x : ℝ, (x - α) / (x + β) = (x^2 - 96*x + 2210) / (x^2 + 65*x - 3510)) : 
  α + β = 112 := by
sorry

end alpha_plus_beta_equals_112_l444_44484


namespace expression_value_l444_44432

theorem expression_value (x y z : ℤ) (hx : x = 3) (hy : y = 2) (hz : z = 1) :
  3 * x - 4 * y + 2 * z = 3 := by
  sorry

end expression_value_l444_44432


namespace practice_paper_percentage_l444_44424

theorem practice_paper_percentage (total_students : ℕ) 
  (passed_all : ℝ) (passed_none : ℝ) (passed_one : ℝ) (passed_four : ℝ) (passed_three : ℕ)
  (h1 : total_students = 2500)
  (h2 : passed_all = 0.1)
  (h3 : passed_none = 0.1)
  (h4 : passed_one = 0.2 * (1 - passed_all - passed_none))
  (h5 : passed_four = 0.24)
  (h6 : passed_three = 500) :
  let remaining := 1 - passed_all - passed_none - passed_one - passed_four - (passed_three : ℝ) / total_students
  let passed_two := (1 - passed_all - passed_none - passed_one - passed_four - (passed_three : ℝ) / total_students) * remaining
  ∃ (ε : ℝ), abs (passed_two - 0.5002) < ε ∧ ε > 0 ∧ ε < 0.0001 :=
by sorry

end practice_paper_percentage_l444_44424


namespace sin_equality_proof_l444_44472

theorem sin_equality_proof (n : ℤ) : 
  -90 ≤ n ∧ n ≤ 90 ∧ Real.sin (n * π / 180) = Real.sin (782 * π / 180) → 
  n = 62 ∨ n = -62 := by
  sorry

end sin_equality_proof_l444_44472


namespace candy_bar_cost_l444_44441

/-- Proves that the cost of each candy bar is $2, given the total spent and number of candy bars. -/
theorem candy_bar_cost (total_spent : ℚ) (num_candy_bars : ℕ) (h1 : total_spent = 4) (h2 : num_candy_bars = 2) :
  total_spent / num_candy_bars = 2 := by
  sorry

#check candy_bar_cost

end candy_bar_cost_l444_44441


namespace arithmetic_seq_sum_l444_44486

/-- An arithmetic sequence with its sum sequence -/
structure ArithmeticSequence where
  a : ℕ → ℝ  -- The sequence
  S : ℕ → ℝ  -- The sum sequence
  is_arithmetic : ∀ n, a (n + 1) - a n = a (n + 2) - a (n + 1)
  sum_def : ∀ n, S n = (n : ℝ) * (a 1 + a n) / 2

/-- Given an arithmetic sequence with S_5 = 20, prove that a_2 + a_3 + a_4 = 12 -/
theorem arithmetic_seq_sum (seq : ArithmeticSequence) (h : seq.S 5 = 20) :
  seq.a 2 + seq.a 3 + seq.a 4 = 12 := by
  sorry

end arithmetic_seq_sum_l444_44486


namespace seastar_arms_l444_44403

theorem seastar_arms (num_starfish : ℕ) (arms_per_starfish : ℕ) (total_arms : ℕ) : 
  num_starfish = 7 → arms_per_starfish = 5 → total_arms = 49 → 
  total_arms - (num_starfish * arms_per_starfish) = 14 := by
sorry

end seastar_arms_l444_44403


namespace cube_with_holes_surface_area_l444_44451

/-- Represents a cube with holes -/
structure CubeWithHoles where
  edge_length : ℝ
  hole_side_length : ℝ

/-- Calculate the total surface area of a cube with holes -/
def total_surface_area (cube : CubeWithHoles) : ℝ :=
  let original_surface_area := 6 * cube.edge_length ^ 2
  let hole_area := 6 * cube.hole_side_length ^ 2
  let exposed_internal_area := 6 * 4 * cube.hole_side_length ^ 2
  original_surface_area - hole_area + exposed_internal_area

/-- Theorem stating the total surface area of the specific cube with holes -/
theorem cube_with_holes_surface_area :
  let cube : CubeWithHoles := { edge_length := 4, hole_side_length := 2 }
  total_surface_area cube = 168 := by
  sorry

end cube_with_holes_surface_area_l444_44451


namespace distance_between_points_l444_44485

/-- The distance between points (3, 24) and (10, 0) is 25. -/
theorem distance_between_points : Real.sqrt ((10 - 3)^2 + (24 - 0)^2) = 25 := by
  sorry

end distance_between_points_l444_44485


namespace min_value_expression_l444_44448

theorem min_value_expression (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : x + 4 * y = 4) :
  (x + 28 * y + 4) / (x * y) ≥ 18 := by
sorry

end min_value_expression_l444_44448


namespace population_sum_theorem_l444_44422

/-- The population of Springfield -/
def springfield_population : ℕ := 482653

/-- The difference in population between Springfield and Greenville -/
def population_difference : ℕ := 119666

/-- The total population of Springfield and Greenville -/
def total_population : ℕ := 845640

/-- Theorem stating that the sum of Springfield's population and a city with 119,666 fewer people equals the total population -/
theorem population_sum_theorem : 
  springfield_population + (springfield_population - population_difference) = total_population := by
  sorry

end population_sum_theorem_l444_44422


namespace range_of_fraction_l444_44446

theorem range_of_fraction (a b : ℝ) (ha : 1 < a ∧ a < 4) (hb : 2 < b ∧ b < 8) :
  1/8 < a/b ∧ a/b < 2 := by
  sorry

end range_of_fraction_l444_44446


namespace slope_movement_l444_44496

theorem slope_movement (hypotenuse : ℝ) (ratio : ℝ) : 
  hypotenuse = 100 * Real.sqrt 5 →
  ratio = 1 / 2 →
  ∃ (x : ℝ), x^2 + (ratio * x)^2 = hypotenuse^2 ∧ x = 100 :=
by sorry

end slope_movement_l444_44496


namespace second_caterer_cheaper_at_50_l444_44410

/-- Represents the cost function for a caterer -/
structure Caterer where
  base_fee : ℕ
  per_person : ℕ
  discount : ℕ → ℕ

/-- Calculate the total cost for a caterer given the number of people -/
def total_cost (c : Caterer) (people : ℕ) : ℕ :=
  c.base_fee + c.per_person * people - c.discount people

/-- First caterer's pricing model -/
def caterer1 : Caterer :=
  { base_fee := 120
  , per_person := 18
  , discount := λ _ => 0 }

/-- Second caterer's pricing model -/
def caterer2 : Caterer :=
  { base_fee := 250
  , per_person := 14
  , discount := λ n => if n ≥ 50 then 50 else 0 }

/-- Theorem stating that 50 is the least number of people for which the second caterer is cheaper -/
theorem second_caterer_cheaper_at_50 :
  (total_cost caterer2 50 < total_cost caterer1 50) ∧
  (∀ n : ℕ, n < 50 → total_cost caterer1 n ≤ total_cost caterer2 n) :=
sorry

end second_caterer_cheaper_at_50_l444_44410


namespace f_properties_l444_44458

-- Define the function f(x) = x^3 - 3x + 1
def f (x : ℝ) : ℝ := x^3 - 3*x + 1

-- Theorem statement
theorem f_properties :
  (¬ (∀ x, f (-x) = -f x) ∧ ¬ (∀ x, f (-x) = f x)) ∧
  (∃ y, f 1 ≤ f y ∧ ∀ x, f 1 ≤ f x) := by
  sorry

end f_properties_l444_44458


namespace probability_of_selection_X_l444_44494

theorem probability_of_selection_X 
  (prob_Y : ℝ) 
  (prob_X_and_Y : ℝ) 
  (h1 : prob_Y = 2/5) 
  (h2 : prob_X_and_Y = 0.05714285714285714) : 
  ∃ (prob_X : ℝ), prob_X = 0.14285714285714285 ∧ prob_X_and_Y = prob_X * prob_Y :=
by
  sorry

end probability_of_selection_X_l444_44494


namespace digit_sum_theorem_l444_44474

/-- Concatenate two digits to form a two-digit number -/
def concatenate (a b : Nat) : Nat := 10 * a + b

/-- Check if a number is prime -/
def isPrime (n : Nat) : Prop := n > 1 ∧ ∀ d : Nat, d > 1 → d < n → ¬(n % d = 0)

theorem digit_sum_theorem (p q r : Nat) : 
  p < 10 → q < 10 → r < 10 →
  p ≠ q → p ≠ r → q ≠ r →
  isPrime (concatenate p q) →
  isPrime (concatenate p r) →
  isPrime (concatenate q r) →
  concatenate p q ≠ concatenate p r →
  concatenate p q ≠ concatenate q r →
  concatenate p r ≠ concatenate q r →
  (concatenate p q) * (concatenate p r) = 221 →
  p + q + r = 11 := by
sorry

end digit_sum_theorem_l444_44474


namespace march_book_sales_l444_44453

theorem march_book_sales (january_sales february_sales : ℕ) 
  (h1 : january_sales = 15)
  (h2 : february_sales = 16)
  (h3 : (january_sales + february_sales + march_sales) / 3 = 16) :
  march_sales = 17 := by
  sorry

end march_book_sales_l444_44453


namespace smallest_number_of_students_l444_44419

/-- Represents the number of students in each grade --/
structure Students :=
  (ninth : ℕ)
  (seventh : ℕ)
  (fifth : ℕ)

/-- The ratio of 9th-graders to 7th-graders is 7:4 --/
def ratio_ninth_seventh (s : Students) : Prop :=
  7 * s.seventh = 4 * s.ninth

/-- The ratio of 7th-graders to 5th-graders is 6:5 --/
def ratio_seventh_fifth (s : Students) : Prop :=
  6 * s.fifth = 5 * s.seventh

/-- The total number of students --/
def total_students (s : Students) : ℕ :=
  s.ninth + s.seventh + s.fifth

/-- The theorem stating the smallest possible number of students --/
theorem smallest_number_of_students :
  ∃ (s : Students),
    ratio_ninth_seventh s ∧
    ratio_seventh_fifth s ∧
    (∀ (t : Students),
      ratio_ninth_seventh t ∧ ratio_seventh_fifth t →
      total_students s ≤ total_students t) ∧
    total_students s = 43 :=
  sorry

end smallest_number_of_students_l444_44419


namespace polynomial_at_negative_two_l444_44447

def polynomial (x : ℝ) : ℝ := 2 * x^4 - 3 * x^3 + x^2 - 2 * x + 4

theorem polynomial_at_negative_two :
  polynomial (-2) = 68 := by sorry

end polynomial_at_negative_two_l444_44447


namespace choose_two_from_three_l444_44433

theorem choose_two_from_three (n : ℕ) (h : n = 3) :
  Nat.choose n 2 = 3 := by
  sorry

end choose_two_from_three_l444_44433


namespace distance_from_origin_to_point_l444_44440

theorem distance_from_origin_to_point (x y : ℝ) :
  x = 8 ∧ y = -15 →
  Real.sqrt (x^2 + y^2) = 17 :=
by sorry

end distance_from_origin_to_point_l444_44440


namespace rogers_app_ratio_l444_44487

/-- Proof that Roger's app ratio is 2 given the problem conditions -/
theorem rogers_app_ratio : 
  let max_apps : ℕ := 50
  let recommended_apps : ℕ := 35
  let delete_apps : ℕ := 20
  let rogers_apps : ℕ := max_apps + delete_apps
  rogers_apps / recommended_apps = 2 := by sorry

end rogers_app_ratio_l444_44487


namespace ratio_of_percentages_l444_44406

theorem ratio_of_percentages (P Q M N : ℝ) 
  (hM : M = 0.4 * Q) 
  (hQ : Q = 0.25 * P) 
  (hN : N = 0.6 * P) 
  (hP : P ≠ 0) : 
  M / N = 1 / 6 := by
  sorry

end ratio_of_percentages_l444_44406


namespace average_visitors_theorem_l444_44461

/-- Calculates the average number of visitors per day in a 30-day month starting on Sunday -/
def averageVisitorsPerDay (sundayVisitors : ℕ) (otherDayVisitors : ℕ) : ℚ :=
  let numSundays := 4
  let numOtherDays := 30 - numSundays
  let totalVisitors := numSundays * sundayVisitors + numOtherDays * otherDayVisitors
  totalVisitors / 30

/-- Proves that the average number of visitors per day is 188 given the specified conditions -/
theorem average_visitors_theorem :
  averageVisitorsPerDay 500 140 = 188 := by
  sorry

end average_visitors_theorem_l444_44461


namespace smallest_digit_change_l444_44439

def original_sum : ℕ := 753 + 946 + 821
def incorrect_result : ℕ := 2420
def correct_result : ℕ := 2520

def change_digit (n : ℕ) (place : ℕ) (new_digit : ℕ) : ℕ := 
  n - (n / 10^place % 10) * 10^place + new_digit * 10^place

theorem smallest_digit_change :
  ∃ (d : ℕ), d < 10 ∧ 
    change_digit 821 2 (d + 1) + 753 + 946 = correct_result ∧
    ∀ (n : ℕ) (p : ℕ) (digit : ℕ), 
      digit < d → 
      change_digit 753 p digit + 946 + 821 ≠ correct_result ∧
      753 + change_digit 946 p digit + 821 ≠ correct_result ∧
      753 + 946 + change_digit 821 p digit ≠ correct_result :=
sorry

#check smallest_digit_change

end smallest_digit_change_l444_44439


namespace initial_player_count_l444_44491

/-- Represents a server in the Minecraft scenario -/
structure Server :=
  (players : ℕ)

/-- Represents the state of the two servers at a given time -/
structure GameState :=
  (server1 : Server)
  (server2 : Server)

/-- Simulates a single step of the game, where a player may switch servers -/
def step (state : GameState) : GameState :=
  if state.server1.players > state.server2.players
  then { server1 := ⟨state.server1.players - 1⟩, server2 := ⟨state.server2.players + 1⟩ }
  else if state.server2.players > state.server1.players
  then { server1 := ⟨state.server1.players + 1⟩, server2 := ⟨state.server2.players - 1⟩ }
  else state

/-- Simulates the entire game for a given number of steps -/
def simulate (initial : GameState) (steps : ℕ) : GameState :=
  match steps with
  | 0 => initial
  | n + 1 => step (simulate initial n)

/-- The theorem stating the possible initial player counts -/
theorem initial_player_count (initial : GameState) :
  (simulate initial 2023).server1.players + (simulate initial 2023).server2.players = initial.server1.players + initial.server2.players →
  (∀ i : ℕ, i ≤ 2023 → (simulate initial i).server1.players ≠ 0) →
  (∀ i : ℕ, i ≤ 2023 → (simulate initial i).server2.players ≠ 0) →
  initial.server1.players = 1011 ∨ initial.server1.players = 1012 :=
sorry

end initial_player_count_l444_44491


namespace odd_multiple_of_three_l444_44456

theorem odd_multiple_of_three (a : ℕ) : 
  Odd (88 * a) → (88 * a) % 3 = 0 → a = 5 := by
  sorry

end odd_multiple_of_three_l444_44456


namespace min_value_expression_l444_44401

theorem min_value_expression (a b c : ℝ) 
  (ha : -0.5 < a ∧ a < 0.5) 
  (hb : -0.5 < b ∧ b < 0.5) 
  (hc : -0.5 < c ∧ c < 0.5) : 
  1 / ((1 - a) * (1 - b) * (1 - c)) + 1 / ((1 + a) * (1 + b) * (1 + c)) ≥ 4.74 :=
sorry

end min_value_expression_l444_44401


namespace complex_product_modulus_l444_44428

theorem complex_product_modulus (a b : ℂ) (t : ℝ) :
  Complex.abs a = 3 →
  Complex.abs b = 5 →
  a * b = t - 3 * Complex.I →
  t = 6 * Real.sqrt 6 := by
sorry

end complex_product_modulus_l444_44428


namespace systematic_sample_correct_l444_44420

/-- Given a total number of students, sample size, and first drawn number,
    returns the list of remaining numbers in the systematic sampling sequence. -/
def systematicSample (totalStudents : Nat) (sampleSize : Nat) (firstNumber : Nat) : List Nat :=
  let interval := totalStudents / sampleSize
  List.range (sampleSize - 1) |>.map (fun i => (firstNumber + (i + 1) * interval) % totalStudents)

/-- Theorem stating that for the given conditions, the systematic sampling
    produces the expected sequence of numbers. -/
theorem systematic_sample_correct :
  systematicSample 60 5 4 = [16, 28, 40, 52] := by
  sorry

#eval systematicSample 60 5 4

end systematic_sample_correct_l444_44420


namespace min_discount_rate_l444_44430

/-- The minimum discount rate for a product with given cost and marked prices, ensuring a minimum profit percentage. -/
theorem min_discount_rate (cost : ℝ) (marked : ℝ) (min_profit_percent : ℝ) :
  cost = 1000 →
  marked = 1500 →
  min_profit_percent = 5 →
  ∃ x : ℝ, x = 0.7 ∧
    ∀ y : ℝ, (marked * y - cost ≥ cost * (min_profit_percent / 100) → y ≥ x) :=
by sorry

end min_discount_rate_l444_44430


namespace power_of_two_plus_one_equals_square_l444_44452

theorem power_of_two_plus_one_equals_square (m n : ℕ) : 2^n + 1 = m^2 ↔ m = 3 ∧ n = 3 :=
by sorry

end power_of_two_plus_one_equals_square_l444_44452


namespace P_in_quadrant_III_l444_44475

-- Define the point P
def P : ℝ × ℝ := (-3, -4)

-- Define the quadrants
def in_quadrant_I (p : ℝ × ℝ) : Prop := p.1 > 0 ∧ p.2 > 0
def in_quadrant_II (p : ℝ × ℝ) : Prop := p.1 < 0 ∧ p.2 > 0
def in_quadrant_III (p : ℝ × ℝ) : Prop := p.1 < 0 ∧ p.2 < 0
def in_quadrant_IV (p : ℝ × ℝ) : Prop := p.1 > 0 ∧ p.2 < 0

-- Theorem: P lies in Quadrant III
theorem P_in_quadrant_III : in_quadrant_III P := by sorry

end P_in_quadrant_III_l444_44475


namespace inequality_solution_set_l444_44455

-- Define the inequality
def inequality (x : ℝ) : Prop := (x - 2) / (x - 1) ≥ 2

-- Define the solution set
def solution_set : Set ℝ := { x | 0 ≤ x ∧ x < 1 }

-- Theorem stating that the solution set is correct
theorem inequality_solution_set :
  ∀ x : ℝ, x ∈ solution_set ↔ inequality x ∧ x ≠ 1 :=
sorry

end inequality_solution_set_l444_44455


namespace hyperbola_m_range_l444_44465

-- Define the hyperbola equation
def is_hyperbola (m : ℝ) : Prop :=
  ∃ (x y : ℝ), x^2 / (m + 2) - y^2 / (m - 1) = 1

-- Define the condition that foci are on x-axis
def foci_on_x_axis (m : ℝ) : Prop :=
  m + 2 > 0 ∧ m - 1 > 0

-- Theorem statement
theorem hyperbola_m_range (m : ℝ) :
  is_hyperbola m ∧ foci_on_x_axis m → m > 1 :=
sorry

end hyperbola_m_range_l444_44465


namespace no_positive_integer_solutions_l444_44444

theorem no_positive_integer_solutions :
  ¬ ∃ (x y : ℕ), x > 0 ∧ y > 0 ∧ 21 * x * y = 7 - 3 * x - 4 * y := by
  sorry

end no_positive_integer_solutions_l444_44444


namespace f_composition_negative_one_l444_44493

-- Define the function f
def f (x : ℝ) : ℝ := x^2 + 1

-- State the theorem
theorem f_composition_negative_one : f (f (-1)) = 5 := by
  sorry

end f_composition_negative_one_l444_44493


namespace number_relations_l444_44473

theorem number_relations : 
  (∃ n : ℤ, 28 = 4 * n) ∧ 
  (∃ n : ℤ, 361 = 19 * n) ∧ 
  (∀ n : ℤ, 63 ≠ 19 * n) ∧ 
  (∃ n : ℤ, 45 = 15 * n) ∧ 
  (∃ n : ℤ, 30 = 15 * n) ∧ 
  (∃ n : ℤ, 144 = 12 * n) := by
sorry

end number_relations_l444_44473


namespace prime_and_even_under_10_composite_and_odd_under_10_l444_44459

def isEven (n : ℕ) : Prop := ∃ k, n = 2 * k
def isOdd (n : ℕ) : Prop := ¬(isEven n)
def isPrime (n : ℕ) : Prop := n > 1 ∧ ∀ m, m ∣ n → m = 1 ∨ m = n
def isComposite (n : ℕ) : Prop := n > 1 ∧ ¬(isPrime n)

theorem prime_and_even_under_10 : ∃! n, n < 10 ∧ isPrime n ∧ isEven n :=
sorry

theorem composite_and_odd_under_10 : ∃! n, n < 10 ∧ isComposite n ∧ isOdd n :=
sorry

end prime_and_even_under_10_composite_and_odd_under_10_l444_44459


namespace complex_roots_sum_of_absolute_values_l444_44466

theorem complex_roots_sum_of_absolute_values (a : ℝ) (x₁ x₂ : ℂ) : 
  (∀ x : ℂ, x^2 - 2*a*x + a^2 - 4*a + 4 = 0 ↔ x = x₁ ∨ x = x₂) →
  Complex.abs x₁ + Complex.abs x₂ = 3 →
  a = 1/2 := by
sorry

end complex_roots_sum_of_absolute_values_l444_44466


namespace buyer_ratio_l444_44405

/-- Represents the number of buyers on a given day -/
structure BuyerCount where
  count : ℕ

/-- Represents the buyer counts for three consecutive days -/
structure ThreeDayBuyers where
  dayBeforeYesterday : BuyerCount
  yesterday : BuyerCount
  today : BuyerCount

/-- The conditions given in the problem -/
def storeConditions (buyers : ThreeDayBuyers) : Prop :=
  buyers.today.count = buyers.yesterday.count + 40 ∧
  buyers.dayBeforeYesterday.count + buyers.yesterday.count + buyers.today.count = 140 ∧
  buyers.dayBeforeYesterday.count = 50

/-- The theorem to prove -/
theorem buyer_ratio (buyers : ThreeDayBuyers) 
  (h : storeConditions buyers) : 
  buyers.yesterday.count * 2 = buyers.dayBeforeYesterday.count := by
  sorry


end buyer_ratio_l444_44405


namespace right_triangle_hypotenuse_segments_l444_44463

/-- Given a right triangle with legs in ratio 3:7 and altitude to hypotenuse of 42,
    prove that the altitude divides the hypotenuse into segments of length 18 and 98 -/
theorem right_triangle_hypotenuse_segments
  (a b c h : ℝ)
  (right_angle : a^2 + b^2 = c^2)
  (leg_ratio : b = 7/3 * a)
  (altitude : h = 42)
  (geo_mean : a * b = h^2) :
  ∃ (x y : ℝ), x + y = c ∧ x * y = h^2 ∧ x = 18 ∧ y = 98 := by
  sorry

end right_triangle_hypotenuse_segments_l444_44463


namespace linear_function_decreasing_iff_k_lt_neg_two_l444_44431

/-- A linear function y = mx + b where m = k + 2 and b = -1 -/
def linear_function (k : ℝ) (x : ℝ) : ℝ := (k + 2) * x - 1

/-- The property that y decreases as x increases -/
def decreasing_function (f : ℝ → ℝ) : Prop :=
  ∀ x₁ x₂, x₁ < x₂ → f x₁ > f x₂

theorem linear_function_decreasing_iff_k_lt_neg_two (k : ℝ) :
  decreasing_function (linear_function k) ↔ k < -2 := by
  sorry

end linear_function_decreasing_iff_k_lt_neg_two_l444_44431


namespace infinite_square_free_sequences_l444_44497

def x_seq (a b n : ℕ) : ℕ := a * n + b
def y_seq (c d n : ℕ) : ℕ := c * n + d

def is_square_free (n : ℕ) : Prop :=
  ∀ p : ℕ, Nat.Prime p → (p * p) ∣ n → False

theorem infinite_square_free_sequences
  (a b c d : ℕ) 
  (h1 : Nat.gcd a b = 1) 
  (h2 : Nat.gcd c d = 1) :
  ∃ S : Set ℕ, Set.Infinite S ∧ 
    ∀ n ∈ S, is_square_free (x_seq a b n) ∧ is_square_free (y_seq c d n) := by
  sorry

end infinite_square_free_sequences_l444_44497


namespace part_1_part_2_l444_44425

-- Define the triangle ABC
structure Triangle where
  A : Real
  B : Real
  C : Real
  a : Real
  b : Real
  c : Real

-- Define the conditions
def is_valid_triangle (t : Triangle) : Prop :=
  t.A + t.B + t.C = 180 ∧ t.A > 0 ∧ t.B > 0 ∧ t.C > 0 ∧
  t.a > 0 ∧ t.b > 0 ∧ t.c > 0

def satisfies_law_of_sines (t : Triangle) : Prop :=
  t.a / Real.sin t.A = t.b / Real.sin t.B ∧
  t.b / Real.sin t.B = t.c / Real.sin t.C

def is_geometric_sequence (a b c : Real) : Prop :=
  b * b = a * c

-- Theorem statements
theorem part_1 (t : Triangle) 
  (h1 : is_valid_triangle t)
  (h2 : satisfies_law_of_sines t)
  (h3 : t.B = 60)
  (h4 : t.b = Real.sqrt 3)
  (h5 : t.A = 45) :
  t.a = Real.sqrt 2 := by sorry

theorem part_2 (t : Triangle)
  (h1 : is_valid_triangle t)
  (h2 : satisfies_law_of_sines t)
  (h3 : t.B = 60)
  (h4 : is_geometric_sequence t.a t.b t.c) :
  t.A = 60 ∧ t.C = 60 := by sorry

end part_1_part_2_l444_44425


namespace remaining_value_proof_l444_44418

theorem remaining_value_proof (x : ℝ) (h : 0.36 * x = 2376) : 4500 - 0.7 * x = -120 := by
  sorry

end remaining_value_proof_l444_44418


namespace largest_number_of_circles_l444_44476

/-- Given a convex quadrilateral BCDE in the plane where lines EB and DC intersect at A,
    this theorem proves that the largest number of nonoverlapping circles that can lie in
    BCDE and are tangent to both BE and CD is 5, given the specified conditions. -/
theorem largest_number_of_circles
  (AB : ℝ) (AC : ℝ) (AD : ℝ) (AE : ℝ) (cos_BAC : ℝ)
  (h_AB : AB = 2)
  (h_AC : AC = 5)
  (h_AD : AD = 200)
  (h_AE : AE = 500)
  (h_cos_BAC : cos_BAC = 7/9)
  : ℕ :=
5

#check largest_number_of_circles

end largest_number_of_circles_l444_44476


namespace probability_of_two_in_three_elevenths_l444_44438

/-- The decimal representation of a rational number -/
def decimal_representation (q : ℚ) : ℕ → ℕ := sorry

/-- The period of the decimal representation of a rational number -/
def decimal_period (q : ℚ) : ℕ := sorry

/-- The count of a specific digit in one period of the decimal representation -/
def digit_count_in_period (q : ℚ) (d : ℕ) : ℕ := sorry

/-- The probability of randomly selecting a specific digit from the decimal representation -/
def digit_probability (q : ℚ) (d : ℕ) : ℚ :=
  (digit_count_in_period q d : ℚ) / (decimal_period q : ℚ)

theorem probability_of_two_in_three_elevenths :
  digit_probability (3/11) 2 = 1/2 := by sorry

end probability_of_two_in_three_elevenths_l444_44438


namespace odd_side_length_l444_44416

/-- A triangle with two known sides and an odd third side -/
structure OddTriangle where
  side1 : ℝ
  side2 : ℝ
  side3 : ℝ
  h1 : side1 = 2
  h2 : side2 = 5
  h3 : ∃ k : ℕ, side3 = 2 * k + 1

/-- The triangle inequality theorem -/
axiom triangle_inequality (t : OddTriangle) : 
  t.side1 + t.side2 > t.side3 ∧ 
  t.side1 + t.side3 > t.side2 ∧ 
  t.side2 + t.side3 > t.side1

theorem odd_side_length (t : OddTriangle) : t.side3 = 5 := by
  sorry

end odd_side_length_l444_44416


namespace right_angle_times_l444_44437

/-- Represents a time on a 12-hour analog clock -/
structure ClockTime where
  hour : Fin 12
  minute : Fin 60

/-- Calculates the angle between hour and minute hands at a given time -/
def angleBetweenHands (time : ClockTime) : ℝ :=
  sorry

/-- Checks if the angle between hands is a right angle (90 degrees) -/
def isRightAngle (time : ClockTime) : Prop :=
  angleBetweenHands time = 90

/-- The theorem stating that when the hands form a right angle, the time is either 3:00 or 9:00 -/
theorem right_angle_times :
  ∀ (time : ClockTime), isRightAngle time →
    (time.hour = 3 ∧ time.minute = 0) ∨ (time.hour = 9 ∧ time.minute = 0) :=
  sorry

end right_angle_times_l444_44437


namespace perfect_fit_implies_r_squared_one_l444_44426

/-- Represents a sample point in a scatter plot -/
structure SamplePoint where
  x : ℝ
  y : ℝ

/-- Represents a linear regression model -/
structure LinearRegression where
  slope : ℝ
  intercept : ℝ

/-- Checks if a point lies on a given line -/
def pointOnLine (p : SamplePoint) (model : LinearRegression) : Prop :=
  p.y = model.slope * p.x + model.intercept

/-- The coefficient of determination (R²) for a regression model -/
def R_squared (data : List SamplePoint) (model : LinearRegression) : ℝ :=
  sorry -- Definition of R² calculation

theorem perfect_fit_implies_r_squared_one
  (data : List SamplePoint)
  (model : LinearRegression)
  (h_non_zero_slope : model.slope ≠ 0)
  (h_all_points_on_line : ∀ p ∈ data, pointOnLine p model) :
  R_squared data model = 1 :=
sorry

end perfect_fit_implies_r_squared_one_l444_44426


namespace power_function_increasing_m_l444_44464

-- Define a power function
def isPowerFunction (f : ℝ → ℝ) : Prop :=
  ∃ (a b : ℝ), ∀ x, f x = a * x^b

-- Define an increasing function on (0, +∞)
def isIncreasingOn (f : ℝ → ℝ) : Prop :=
  ∀ x y, 0 < x ∧ x < y → f x < f y

-- The main theorem
theorem power_function_increasing_m (m : ℝ) :
  let f := fun x : ℝ => (m^2 - m - 1) * x^m
  isPowerFunction f ∧ isIncreasingOn f → m = 2 := by
  sorry

end power_function_increasing_m_l444_44464


namespace sets_operations_l444_44443

-- Define the sets A and B
def A : Set ℝ := {x | -3 < x ∧ x < 2}
def B : Set ℝ := {x | Real.exp (x - 1) ≥ 1}

-- Define the theorem
theorem sets_operations :
  (A ∪ B = {x | x > -3}) ∧
  ((Aᶜ) ∩ B = {x | x ≥ 2}) := by
  sorry

end sets_operations_l444_44443


namespace inequality_proof_l444_44415

theorem inequality_proof (a b c d : ℝ) 
  (nonzero : a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0 ∧ d ≠ 0)
  (distinct : a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d)
  (sum_condition : a/b + b/c + c/d + d/a = 4)
  (product_condition : a*c = b*d) :
  (a/c + b/d + c/a + d/b ≤ -12) ∧
  (∀ k : ℝ, (∀ a' b' c' d' : ℝ, 
    a'/b' + b'/c' + c'/d' + d'/a' = 4 → 
    a'*c' = b'*d' → 
    a'/c' + b'/d' + c'/a' + d'/b' ≤ k) → 
  k ≤ -12) :=
by sorry

end inequality_proof_l444_44415


namespace problem_statement_l444_44411

theorem problem_statement :
  (∀ x : ℝ, 1 + 2 * x^4 ≥ 2 * x^3 + x^2) ∧
  (∀ x y z : ℝ, x + 2*y + 3*z = 6 →
    x^2 + y^2 + z^2 ≥ 18/7 ∧
    ∃ x y z : ℝ, x + 2*y + 3*z = 6 ∧ x^2 + y^2 + z^2 = 18/7) :=
by sorry

end problem_statement_l444_44411


namespace repeating_decimal_sum_l444_44478

theorem repeating_decimal_sum : 
  (1 : ℚ) / 9 + (2 : ℚ) / 99 + (2 : ℚ) / 333 = (503 : ℚ) / 3663 := by sorry

end repeating_decimal_sum_l444_44478


namespace largest_five_digit_congruent_to_17_mod_34_l444_44421

theorem largest_five_digit_congruent_to_17_mod_34 :
  ∀ n : ℕ, 10000 ≤ n ∧ n ≤ 99999 ∧ n % 34 = 17 → n ≤ 99994 :=
by
  sorry

end largest_five_digit_congruent_to_17_mod_34_l444_44421


namespace diet_soda_count_l444_44481

/-- Given a grocery store inventory, calculate the number of diet soda bottles. -/
theorem diet_soda_count (regular_soda : ℕ) (difference : ℕ) : 
  regular_soda = 79 → difference = 26 → regular_soda - difference = 53 := by
  sorry

#check diet_soda_count

end diet_soda_count_l444_44481


namespace sugar_servings_calculation_l444_44400

/-- Calculates the number of servings in a container given the total amount and serving size -/
def number_of_servings (total_amount : ℚ) (serving_size : ℚ) : ℚ :=
  total_amount / serving_size

/-- Proves that a container with 35 2/3 cups of sugar contains 23 7/9 servings when each serving is 1 1/2 cups -/
theorem sugar_servings_calculation :
  let total_sugar : ℚ := 35 + 2/3
  let serving_size : ℚ := 1 + 1/2
  number_of_servings total_sugar serving_size = 23 + 7/9 := by
  sorry

#eval number_of_servings (35 + 2/3) (1 + 1/2)

end sugar_servings_calculation_l444_44400


namespace inverse_variation_problem_l444_44450

theorem inverse_variation_problem (a b : ℝ) (k : ℝ) :
  (∀ a b, a^3 * (b^(1/2)) = k) →  -- The cube of a and square root of b vary inversely
  (3^3 * (64^(1/2)) = k) →        -- a = 3 when b = 64
  (a * b = 36) →                  -- Given condition ab = 36
  (b = 6) :=                      -- Prove that b = 6
by sorry

end inverse_variation_problem_l444_44450


namespace no_integer_solution_l444_44470

theorem no_integer_solution : ¬ ∃ (x y : ℤ), 21 * x - 35 * y = 59 := by
  sorry

end no_integer_solution_l444_44470


namespace part_one_part_two_l444_44467

-- Define sets A and B
def A (a : ℝ) : Set ℝ := {x | a ≤ x ∧ x ≤ a + 2}
def B : Set ℝ := {x | x^2 + x - 2 < 0}

-- Part 1
theorem part_one : A 0 ∩ (Set.univ \ B) = {x | 1 ≤ x ∧ x ≤ 2} := by sorry

-- Part 2
theorem part_two (a : ℝ) : (∀ x ∈ A a, x ∉ B) ↔ (a ≤ -4 ∨ a ≥ 1) := by sorry

end part_one_part_two_l444_44467


namespace reunion_boys_l444_44429

/-- The number of handshakes when n people each shake hands with everyone else exactly once -/
def handshakes (n : ℕ) : ℕ := n * (n - 1) / 2

/-- There were 8 boys at the reunion -/
theorem reunion_boys : ∃ n : ℕ, n > 0 ∧ handshakes n = 28 ∧ n = 8 := by
  sorry

end reunion_boys_l444_44429


namespace intersection_point_l444_44417

-- Define the three lines
def line1 (x y : ℝ) : Prop := y = 2 * x - 1
def line2 (x y : ℝ) : Prop := y = -3 * x + 4
def line3 (x y m : ℝ) : Prop := y = 4 * x + m

-- Theorem statement
theorem intersection_point (m : ℝ) : 
  (∃ x y : ℝ, line1 x y ∧ line2 x y ∧ line3 x y m) → m = -3 := by
  sorry

end intersection_point_l444_44417


namespace division_theorem_l444_44436

theorem division_theorem (dividend divisor remainder quotient : ℕ) : 
  dividend = divisor * quotient + remainder →
  dividend = 167 →
  divisor = 18 →
  remainder = 5 →
  quotient = 9 := by
sorry

end division_theorem_l444_44436


namespace alternating_draw_probability_l444_44427

/-- Represents the number of white balls in the box -/
def white_balls : ℕ := 4

/-- Represents the number of black balls in the box -/
def black_balls : ℕ := 4

/-- Represents the total number of balls in the box -/
def total_balls : ℕ := white_balls + black_balls

/-- Represents the probability of drawing all balls in alternating colors -/
def alternating_probability : ℚ := 1 / 35

/-- Theorem stating that the probability of drawing all balls in alternating colors is 1/35 -/
theorem alternating_draw_probability :
  alternating_probability = 1 / 35 := by sorry

end alternating_draw_probability_l444_44427


namespace parabola_shift_theorem_l444_44460

/-- Represents a parabola of the form y = ax^2 + bx + c -/
structure Parabola where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Shifts a parabola horizontally and vertically -/
def shift_parabola (p : Parabola) (h : ℝ) (k : ℝ) : Parabola :=
  { a := p.a
    b := -2 * p.a * h + p.b
    c := p.a * h^2 - p.b * h + p.c + k }

theorem parabola_shift_theorem (p : Parabola) :
  let original := { a := -2, b := 0, c := 1 : Parabola }
  let shifted := shift_parabola original 1 2
  shifted = { a := -2, b := 4, c := 3 : Parabola } := by
  sorry

#check parabola_shift_theorem

end parabola_shift_theorem_l444_44460
