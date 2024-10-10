import Mathlib

namespace trigonometric_identities_l2007_200738

theorem trigonometric_identities (α : Real) (h : Real.tan α = 2) :
  (Real.cos (π/2 + α) * Real.sin (3*π/2 - α)) / Real.tan (-π + α) = 1/5 ∧
  (1 + 3*Real.sin α*Real.cos α) / (Real.sin α^2 - 2*Real.cos α^2) = 11/2 := by
  sorry

end trigonometric_identities_l2007_200738


namespace inequality_solution_l2007_200710

theorem inequality_solution (a x : ℝ) : ax^2 - ax + x > 0 ↔
  (a = 0 ∧ x > 0) ∨
  (a = 1 ∧ x ≠ 0) ∨
  (a < 0 ∧ 0 < x ∧ x < 1 - 1/a) ∨
  (a > 1 ∧ (x < 0 ∨ x > 1 - 1/a)) ∨
  (0 < a ∧ a < 1 ∧ (x < 1 - 1/a ∨ x > 0)) :=
by sorry

end inequality_solution_l2007_200710


namespace sphere_in_cube_untouchable_area_l2007_200775

/-- The area of a cube's inner surface that a sphere can't touch -/
def untouchableArea (cubeEdge : ℝ) (sphereRadius : ℝ) : ℝ :=
  12 * cubeEdge * sphereRadius - 24 * sphereRadius^2

theorem sphere_in_cube_untouchable_area :
  untouchableArea 5 1 = 96 := by
  sorry

end sphere_in_cube_untouchable_area_l2007_200775


namespace no_real_solution_for_log_equation_l2007_200779

theorem no_real_solution_for_log_equation :
  ¬∃ (x : ℝ), (Real.log (x + 5) + Real.log (x - 2) = Real.log (x^2 - 3*x - 10)) ∧ 
              (x + 5 > 0) ∧ (x - 2 > 0) ∧ (x^2 - 3*x - 10 > 0) := by
  sorry

end no_real_solution_for_log_equation_l2007_200779


namespace jeff_tennis_games_l2007_200745

/-- Calculates the number of games Jeff wins in tennis given the playing time, scoring rate, and points needed to win a match. -/
theorem jeff_tennis_games (playing_time : ℕ) (scoring_rate : ℕ) (points_per_match : ℕ) : 
  playing_time = 120 ∧ scoring_rate = 5 ∧ points_per_match = 8 → 
  (playing_time / scoring_rate) / points_per_match = 3 := by sorry

end jeff_tennis_games_l2007_200745


namespace quadratic_roots_range_l2007_200758

theorem quadratic_roots_range (m : ℝ) : 
  (∃ x y : ℝ, x ≠ y ∧ x^2 + m*x + 1 = 0 ∧ y^2 + m*y + 1 = 0) →
  m ∈ Set.Ioi 2 ∪ Set.Iio (-2) :=
sorry

end quadratic_roots_range_l2007_200758


namespace doctors_visit_cost_l2007_200780

theorem doctors_visit_cost (cast_cost insurance_coverage out_of_pocket : ℝ) :
  cast_cost = 200 →
  insurance_coverage = 0.6 →
  out_of_pocket = 200 →
  ∃ (visit_cost : ℝ),
    visit_cost = 300 ∧
    out_of_pocket = (1 - insurance_coverage) * (visit_cost + cast_cost) :=
by sorry

end doctors_visit_cost_l2007_200780


namespace line_segment_length_l2007_200723

/-- Given points A, B, C, and D on a line in that order, prove that AC = 1 cm -/
theorem line_segment_length (A B C D : ℝ) : 
  (A < B) → (B < C) → (C < D) →  -- Points are in order on the line
  (B - A = 2) →                  -- AB = 2 cm
  (D - B = 6) →                  -- BD = 6 cm
  (D - C = 3) →                  -- CD = 3 cm
  (C - A = 1) :=                 -- AC = 1 cm
by sorry

end line_segment_length_l2007_200723


namespace exterior_bisector_theorem_l2007_200704

/-- Represents a triangle with angles given in degrees -/
structure Triangle where
  angle1 : ℝ
  angle2 : ℝ
  angle3 : ℝ
  sum_180 : angle1 + angle2 + angle3 = 180

/-- The triangle formed by the exterior angle bisectors -/
def exterior_bisector_triangle : Triangle :=
  { angle1 := 52
    angle2 := 61
    angle3 := 67
    sum_180 := by norm_num }

/-- The original triangle whose exterior angle bisectors form the given triangle -/
def original_triangle : Triangle :=
  { angle1 := 76
    angle2 := 58
    angle3 := 46
    sum_180 := by norm_num }

theorem exterior_bisector_theorem (t : Triangle) :
  t = exterior_bisector_triangle →
  ∃ (orig : Triangle), orig = original_triangle ∧
    (90 - orig.angle2 / 2) + (90 - orig.angle3 / 2) = t.angle1 ∧
    (90 - orig.angle1 / 2) + (90 - orig.angle3 / 2) = t.angle2 ∧
    (90 - orig.angle1 / 2) + (90 - orig.angle2 / 2) = t.angle3 :=
by
  sorry

end exterior_bisector_theorem_l2007_200704


namespace tray_height_is_seven_l2007_200759

/-- Represents the dimensions of the rectangular paper --/
structure PaperDimensions where
  length : ℝ
  width : ℝ

/-- Represents the parameters of the cuts made on the paper --/
structure CutParameters where
  distance_from_corner : ℝ
  angle : ℝ

/-- Calculates the height of the tray formed from the paper --/
def tray_height (paper : PaperDimensions) (cut : CutParameters) : ℝ :=
  sorry

/-- Theorem stating that the height of the tray is 7 for the given parameters --/
theorem tray_height_is_seven :
  let paper := PaperDimensions.mk 150 100
  let cut := CutParameters.mk 7 (π / 4)
  tray_height paper cut = 7 := by
  sorry

end tray_height_is_seven_l2007_200759


namespace first_positive_term_is_26_l2007_200743

-- Define the sequence a_n
def a (n : ℕ) : ℤ := 4 * n - 102

-- Define the property of being the first positive term
def is_first_positive (k : ℕ) : Prop :=
  a k > 0 ∧ ∀ m : ℕ, m < k → a m ≤ 0

-- Theorem statement
theorem first_positive_term_is_26 : is_first_positive 26 := by
  sorry

end first_positive_term_is_26_l2007_200743


namespace pizza_pasta_price_difference_l2007_200714

/-- The price difference between a Pizza and a Pasta -/
def price_difference (pizza_price chilli_price pasta_price : ℚ) : ℚ :=
  pizza_price - pasta_price

/-- The total cost of the Smith family's purchase -/
def smith_purchase (pizza_price chilli_price pasta_price : ℚ) : ℚ :=
  2 * pizza_price + 3 * chilli_price + 4 * pasta_price

/-- The total cost of the Patel family's purchase -/
def patel_purchase (pizza_price chilli_price pasta_price : ℚ) : ℚ :=
  5 * pizza_price + 6 * chilli_price + 7 * pasta_price

theorem pizza_pasta_price_difference 
  (pizza_price chilli_price pasta_price : ℚ) 
  (h1 : smith_purchase pizza_price chilli_price pasta_price = 53)
  (h2 : patel_purchase pizza_price chilli_price pasta_price = 107) :
  price_difference pizza_price chilli_price pasta_price = 1 := by
  sorry

end pizza_pasta_price_difference_l2007_200714


namespace arithmetic_sequence_problem_l2007_200771

/-- An arithmetic sequence -/
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

/-- The main theorem -/
theorem arithmetic_sequence_problem (a : ℕ → ℝ) 
  (h_arith : arithmetic_sequence a) 
  (h_sum : a 1 + 3 * a 8 + a 15 = 60) : 
  2 * a 9 - a 10 = 12 := by
sorry

end arithmetic_sequence_problem_l2007_200771


namespace power_fraction_simplification_l2007_200762

theorem power_fraction_simplification :
  (3^2015 + 3^2013) / (3^2015 - 3^2013) = 5/4 := by
  sorry

end power_fraction_simplification_l2007_200762


namespace degree_of_g_given_f_plus_g_l2007_200756

/-- Given two polynomials f and g, where f(x) = -3x^5 + 2x^4 + x^2 - 6 and the degree of f + g is 2, the degree of g is 5. -/
theorem degree_of_g_given_f_plus_g (f g : Polynomial ℝ) : 
  f = -3 * X^5 + 2 * X^4 + X^2 - 6 →
  Polynomial.degree (f + g) = 2 →
  Polynomial.degree g = 5 := by sorry

end degree_of_g_given_f_plus_g_l2007_200756


namespace circumcircle_equation_l2007_200733

-- Define the circle C
def C : Set (ℝ × ℝ) := {p | p.1^2 + p.2^2 = 4}

-- Define point P
def P : ℝ × ℝ := (4, 2)

-- Define that P is outside C
axiom P_outside_C : P ∉ C

-- Define that there are two tangent points A and B
axiom tangent_points_exist : ∃ (A B : ℝ × ℝ), A ∈ C ∧ B ∈ C ∧ A ≠ B

-- Define the circumcircle of triangle ABP
def circumcircle (A B : ℝ × ℝ) : Set (ℝ × ℝ) :=
  {p | (p.1 - 2)^2 + (p.2 - 1)^2 = 5}

-- Theorem statement
theorem circumcircle_equation (A B : ℝ × ℝ) 
  (h1 : A ∈ C) (h2 : B ∈ C) (h3 : A ≠ B) :
  circumcircle A B = {p | (p.1 - 2)^2 + (p.2 - 1)^2 = 5} :=
sorry

end circumcircle_equation_l2007_200733


namespace grandmas_apples_l2007_200768

/-- The problem of Grandma's apple purchase --/
theorem grandmas_apples :
  ∀ (tuesday_price : ℝ) (tuesday_kg : ℝ) (saturday_kg : ℝ),
    tuesday_kg > 0 →
    tuesday_price > 0 →
    tuesday_price * tuesday_kg = 20 →
    saturday_kg = 1.5 * tuesday_kg →
    (tuesday_price - 1) * saturday_kg = 24 →
    saturday_kg = 6 := by
  sorry


end grandmas_apples_l2007_200768


namespace circle_properties_l2007_200783

noncomputable def circle_equation (x y : ℝ) : ℝ := (x - Real.sqrt 3)^2 + (y - 1)^2

theorem circle_properties :
  ∃ (c : ℝ × ℝ),
    (∀ x y : ℝ, circle_equation x y = 1 → ‖(x, y) - c‖ = 1) ∧
    (∃ x : ℝ, circle_equation x 0 = 1) ∧
    (∃ x y : ℝ, circle_equation x y = 1 ∧ y = Real.sqrt 3 * x) :=
sorry

end circle_properties_l2007_200783


namespace product_congruence_l2007_200763

theorem product_congruence : 66 * 77 * 88 ≡ 16 [ZMOD 25] := by sorry

end product_congruence_l2007_200763


namespace train_length_l2007_200720

/-- The length of a train given its speed and time to cross a point -/
theorem train_length (speed : Real) (time : Real) : 
  speed = 72 → time = 4.499640028797696 → 
  ∃ (length : Real), abs (length - 89.99280057595392) < 0.000001 := by
  sorry

end train_length_l2007_200720


namespace elevator_exit_theorem_l2007_200767

/-- The number of ways passengers can exit an elevator -/
def elevator_exit_ways (num_passengers : ℕ) (total_floors : ℕ) (start_floor : ℕ) : ℕ :=
  (total_floors - start_floor + 1) ^ num_passengers

/-- Theorem: 6 passengers exiting an elevator in a 12-story building starting from the 3rd floor -/
theorem elevator_exit_theorem :
  elevator_exit_ways 6 12 3 = 1000000 := by
  sorry

end elevator_exit_theorem_l2007_200767


namespace furniture_production_max_profit_l2007_200722

/-- Represents the problem of maximizing profit in furniture production --/
theorem furniture_production_max_profit :
  let x : ℝ := 16  -- Number of sets of type A furniture
  let y : ℝ := -0.3 * x + 80  -- Total profit function
  let time_constraint : Prop := (5/4) * x + (5/3) * (100 - x) ≤ 160  -- Time constraint
  let total_sets : Prop := x + (100 - x) = 100  -- Total number of sets
  let profit_decreasing : Prop := ∀ x₁ x₂, x₁ < x₂ → (-0.3 * x₁ + 80) > (-0.3 * x₂ + 80)  -- Profit decreases as x increases
  
  -- The following conditions hold:
  time_constraint ∧
  total_sets ∧
  profit_decreasing ∧
  (∀ x' : ℝ, x' ≥ 0 → x' ≤ 100 → (5/4) * x' + (5/3) * (100 - x') ≤ 160 → y ≥ -0.3 * x' + 80) →
  
  -- Then the maximum profit is achieved:
  y = 75.2 := by sorry

end furniture_production_max_profit_l2007_200722


namespace three_times_x_greater_than_four_l2007_200750

theorem three_times_x_greater_than_four (x : ℝ) : 
  (3 * x > 4) ↔ (∀ y : ℝ, y = 3 * x → y > 4) :=
by sorry

end three_times_x_greater_than_four_l2007_200750


namespace polynomial_multiplication_l2007_200735

theorem polynomial_multiplication (a b : ℝ) :
  (3 * a^4 - 7 * b^3) * (9 * a^8 + 21 * a^4 * b^3 + 49 * b^6 + 6 * a^2 * b^2) =
  27 * a^12 + 18 * a^6 * b^2 - 42 * a^2 * b^5 - 343 * b^9 := by
sorry

end polynomial_multiplication_l2007_200735


namespace shaded_area_between_squares_l2007_200770

/-- Given a larger square with area 10 cm² and a smaller square with area 4 cm²,
    where the diagonals of the larger square contain the diagonals of the smaller square,
    prove that the area of one of the four identical regions formed between the squares is 1.5 cm². -/
theorem shaded_area_between_squares (larger_square_area smaller_square_area : ℝ)
  (h1 : larger_square_area = 10)
  (h2 : smaller_square_area = 4)
  (h3 : larger_square_area > smaller_square_area)
  (h4 : ∃ (n : ℕ), n = 4 ∧ n * (larger_square_area - smaller_square_area) / n = 1.5) :
  ∃ (shaded_area : ℝ), shaded_area = 1.5 := by
  sorry

end shaded_area_between_squares_l2007_200770


namespace yellow_to_red_ratio_l2007_200785

/-- Represents the number of marbles Beth has initially -/
def total_marbles : ℕ := 72

/-- Represents the number of colors of marbles -/
def num_colors : ℕ := 3

/-- Represents the number of red marbles Beth loses -/
def red_marbles_lost : ℕ := 5

/-- Represents the number of marbles Beth has left after losing some of each color -/
def marbles_left : ℕ := 42

/-- Theorem stating the ratio of yellow marbles lost to red marbles lost -/
theorem yellow_to_red_ratio :
  let initial_per_color := total_marbles / num_colors
  let blue_marbles_lost := 2 * red_marbles_lost
  let yellow_marbles_lost := initial_per_color - (marbles_left - (2 * initial_per_color - red_marbles_lost - blue_marbles_lost))
  yellow_marbles_lost / red_marbles_lost = 3 := by sorry

end yellow_to_red_ratio_l2007_200785


namespace min_distance_point_to_circle_through_reflection_l2007_200772

/-- The minimum distance from a point to a circle through a reflection point on the x-axis -/
theorem min_distance_point_to_circle_through_reflection (A B P : ℝ × ℝ) : 
  A = (-3, 3) →
  P.2 = 0 →
  (B.1 - 1)^2 + (B.2 - 1)^2 = 2 →
  ∃ (min_dist : ℝ), min_dist = 3 * Real.sqrt 2 ∧ 
    ∀ (P' : ℝ × ℝ), P'.2 = 0 → 
      Real.sqrt ((P'.1 - A.1)^2 + (P'.2 - A.2)^2) + 
      Real.sqrt ((B.1 - P'.1)^2 + (B.2 - P'.2)^2) ≥ min_dist :=
by sorry

end min_distance_point_to_circle_through_reflection_l2007_200772


namespace negation_of_proposition_negation_of_less_than_zero_negation_of_cubic_inequality_l2007_200729

theorem negation_of_proposition (p : ℝ → Prop) :
  (¬∀ x : ℝ, p x) ↔ (∃ x : ℝ, ¬(p x)) :=
by sorry

theorem negation_of_less_than_zero (x : ℝ) :
  ¬(x < 0) ↔ (x ≥ 0) :=
by sorry

theorem negation_of_cubic_inequality :
  (¬∀ x : ℝ, x^3 + 2 < 0) ↔ (∃ x : ℝ, x^3 + 2 ≥ 0) :=
by sorry

end negation_of_proposition_negation_of_less_than_zero_negation_of_cubic_inequality_l2007_200729


namespace sheet_width_calculation_l2007_200709

theorem sheet_width_calculation (paper_length : Real) (margin : Real) (picture_area : Real) :
  paper_length = 10 ∧ margin = 1.5 ∧ picture_area = 38.5 →
  ∃ (paper_width : Real), 
    paper_width = 8.5 ∧
    (paper_width - 2 * margin) * (paper_length - 2 * margin) = picture_area :=
by sorry

end sheet_width_calculation_l2007_200709


namespace train_speed_l2007_200754

/-- The speed of a train given crossing times and platform length -/
theorem train_speed (platform_length : ℝ) (platform_crossing_time : ℝ) (man_crossing_time : ℝ) :
  platform_length = 300 →
  platform_crossing_time = 33 →
  man_crossing_time = 18 →
  (platform_length / (platform_crossing_time - man_crossing_time)) * 3.6 = 72 := by
  sorry

#check train_speed

end train_speed_l2007_200754


namespace race_finish_times_l2007_200766

/-- Represents the time difference at the finish line between two runners -/
def time_difference (distance : ℝ) (speed1 speed2 : ℝ) : ℝ :=
  distance * (speed2 - speed1)

theorem race_finish_times (malcolm_speed joshua_speed alice_speed : ℝ) 
  (h1 : malcolm_speed = 5)
  (h2 : joshua_speed = 7)
  (h3 : alice_speed = 6)
  (race_distance : ℝ)
  (h4 : race_distance = 12) :
  time_difference race_distance malcolm_speed joshua_speed = 24 ∧
  time_difference race_distance malcolm_speed alice_speed = 12 :=
by sorry

end race_finish_times_l2007_200766


namespace frac_one_over_x_is_fraction_l2007_200713

-- Define what a fraction is
def is_fraction (expr : ℚ → ℚ) : Prop :=
  ∃ (n d : ℚ → ℚ), ∀ x, expr x = (n x) / (d x)

-- State the theorem
theorem frac_one_over_x_is_fraction :
  is_fraction (λ x => 1 / x) :=
sorry

end frac_one_over_x_is_fraction_l2007_200713


namespace degree_of_h_is_4_l2007_200753

/-- The polynomial f(x) -/
def f (x : ℝ) : ℝ := 1 - 12*x + 3*x^2 - 4*x^3 + 5*x^4

/-- The polynomial g(x) -/
def g (x : ℝ) : ℝ := 3 - 2*x - 6*x^3 + 8*x^4 + x^5

/-- The combined polynomial h(x) = f(x) + c*g(x) -/
def h (c : ℝ) (x : ℝ) : ℝ := f x + c * g x

/-- The degree of a polynomial -/
def degree (p : ℝ → ℝ) : ℕ := sorry

theorem degree_of_h_is_4 : degree (h 0) = 4 := by sorry

end degree_of_h_is_4_l2007_200753


namespace mittens_per_box_l2007_200747

theorem mittens_per_box (num_boxes : ℕ) (scarves_per_box : ℕ) (total_clothing : ℕ) : 
  num_boxes = 8 → 
  scarves_per_box = 4 → 
  total_clothing = 80 → 
  (total_clothing - num_boxes * scarves_per_box) / num_boxes = 6 := by
sorry

end mittens_per_box_l2007_200747


namespace solve_pocket_money_problem_l2007_200799

def pocket_money_problem (initial_money : ℕ) : Prop :=
  let remaining_money := initial_money / 2
  let total_money := remaining_money + 550
  total_money = 1000 ∧ initial_money = 900

theorem solve_pocket_money_problem :
  ∃ (initial_money : ℕ), pocket_money_problem initial_money :=
sorry

end solve_pocket_money_problem_l2007_200799


namespace extreme_values_and_maximum_b_l2007_200737

noncomputable def f (a b x : ℝ) : ℝ := (a * x + b) / x * Real.exp x

noncomputable def g (a b x : ℝ) : ℝ := a * (x - 1) * Real.exp x - f a b x

theorem extreme_values_and_maximum_b :
  (∀ x : ℝ, x ≠ 0 → f 2 1 x ≤ 1 / Real.exp 1) ∧
  (∀ x : ℝ, x ≠ 0 → f 2 1 x ≥ 4 * Real.sqrt (Real.exp 1)) ∧
  (∃ x : ℝ, x ≠ 0 ∧ f 2 1 x = 1 / Real.exp 1) ∧
  (∃ x : ℝ, x ≠ 0 ∧ f 2 1 x = 4 * Real.sqrt (Real.exp 1)) ∧
  (∀ b : ℝ, (∀ x : ℝ, x > 0 → g 1 b x ≥ 1) → b ≤ -1 - 1 / Real.exp 1) ∧
  (∀ x : ℝ, x > 0 → g 1 (-1 - 1 / Real.exp 1) x ≥ 1) :=
by sorry

end extreme_values_and_maximum_b_l2007_200737


namespace stamp_difference_l2007_200784

theorem stamp_difference (p q : ℕ) (h1 : p * 4 = q * 7) 
  (h2 : (p - 8) * 5 = (q + 8) * 6) : p - q = 8 := by
  sorry

end stamp_difference_l2007_200784


namespace sum_of_digits_base7_squared_expectation_l2007_200734

/-- Sum of digits in base 7 -/
def sum_of_digits_base7 (n : ℕ) : ℕ :=
  sorry

/-- Expected value of a function over a finite range -/
def expected_value {α : Type*} (f : α → ℝ) (range : Finset α) : ℝ :=
  sorry

theorem sum_of_digits_base7_squared_expectation :
  expected_value (λ n => (sum_of_digits_base7 n)^2) (Finset.range (7^20)) = 3680 :=
sorry

end sum_of_digits_base7_squared_expectation_l2007_200734


namespace inequality_proof_l2007_200706

theorem inequality_proof (a b c : ℝ) 
  (h_pos_a : a > 0) (h_pos_b : b > 0) (h_pos_c : c > 0)
  (h_inequality : a * b + b * c + c * a ≥ 1) :
  1 / a^2 + 1 / b^2 + 1 / c^2 ≥ Real.sqrt 3 / (a * b * c) := by
  sorry

end inequality_proof_l2007_200706


namespace cone_base_radius_l2007_200730

/-- Given a cone with surface area 24π cm² and its lateral surface unfolded is a semicircle,
    the radius of the base circle is 2√2 cm. -/
theorem cone_base_radius (r : ℝ) (l : ℝ) : 
  r > 0 → l > 0 → 
  (π * r^2 + π * r * l = 24 * π) → 
  (π * l = 2 * π * r) → 
  r = 2 * Real.sqrt 2 := by
  sorry

end cone_base_radius_l2007_200730


namespace unique_valid_f_l2007_200760

def is_valid_f (f : ℕ → ℕ) : Prop :=
  (∀ m, f m = 1 ↔ m = 1) ∧
  (∀ m n, f (m * n) = f m * f n / f (Nat.gcd m n)) ∧
  (∀ m, (f^[2000]) m = f m)

theorem unique_valid_f :
  ∃! f : ℕ → ℕ, is_valid_f f ∧ ∀ n, f n = n :=
sorry

end unique_valid_f_l2007_200760


namespace section_b_average_weight_l2007_200792

/-- Proves that the average weight of section B is 30 kg given the class composition and weight information -/
theorem section_b_average_weight 
  (num_students_a : ℕ) 
  (num_students_b : ℕ) 
  (avg_weight_a : ℝ) 
  (avg_weight_total : ℝ) :
  num_students_a = 26 →
  num_students_b = 34 →
  avg_weight_a = 50 →
  avg_weight_total = 38.67 →
  (num_students_a * avg_weight_a + num_students_b * 30) / (num_students_a + num_students_b) = avg_weight_total :=
by
  sorry

#eval (26 * 50 + 34 * 30) / (26 + 34) -- Should output approximately 38.67

end section_b_average_weight_l2007_200792


namespace troll_ratio_l2007_200702

/-- Given the number of trolls in different locations, prove the ratio of trolls in the plains to trolls under the bridge -/
theorem troll_ratio (path bridge plains : ℕ) : 
  path = 6 ∧ 
  bridge = 4 * path - 6 ∧ 
  path + bridge + plains = 33 →
  plains * 2 = bridge := by
sorry

end troll_ratio_l2007_200702


namespace triangle_problem_l2007_200744

-- Define the triangle
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

-- Define the theorem
theorem triangle_problem (t : Triangle) : 
  -- Given conditions
  t.c = 2 ∧ 
  t.A = π / 3 ∧ 
  (1/2 * t.b * t.c * Real.sin t.A) = Real.sqrt 3 / 2 →
  -- Conclusion
  t.a = Real.sqrt 3 ∧ 
  t.b = 1 ∧ 
  t.C = π / 2 := by
sorry


end triangle_problem_l2007_200744


namespace fibonacci_ratio_difference_bound_l2007_200782

def fibonacci : ℕ → ℕ
  | 0 => 1
  | 1 => 2
  | (n + 2) => fibonacci (n + 1) + fibonacci n

theorem fibonacci_ratio_difference_bound (n k : ℕ) (hn : n ≥ 1) (hk : k ≥ 1) :
  |((fibonacci (n + 1) : ℝ) / fibonacci n) - ((fibonacci (k + 1) : ℝ) / fibonacci k)| ≤ 1 := by
  sorry

end fibonacci_ratio_difference_bound_l2007_200782


namespace taco_cost_l2007_200757

-- Define the cost of a taco and an enchilada
variable (T E : ℚ)

-- Define the conditions from the problem
def condition1 : Prop := 2 * T + 3 * E = 390 / 50
def condition2 : Prop := 3 * T + 5 * E = 635 / 50

-- Theorem to prove
theorem taco_cost (h1 : condition1 T E) (h2 : condition2 T E) : T = 9 / 10 := by
  sorry

end taco_cost_l2007_200757


namespace daniella_savings_l2007_200787

/-- Daniella's savings amount -/
def D : ℝ := 400

/-- Ariella's initial savings amount -/
def A : ℝ := D + 200

/-- Interest rate per annum (as a decimal) -/
def r : ℝ := 0.1

/-- Time period in years -/
def t : ℝ := 2

/-- Ariella's final amount after interest -/
def F : ℝ := 720

theorem daniella_savings : 
  (A + A * r * t = F) → D = 400 := by
  sorry

end daniella_savings_l2007_200787


namespace f_composition_negative_one_l2007_200764

-- Define the function f
def f (x : ℝ) : ℝ := x + 1

-- State the theorem
theorem f_composition_negative_one : f (f (-1)) = 1 := by
  sorry

end f_composition_negative_one_l2007_200764


namespace anns_shopping_cost_l2007_200778

theorem anns_shopping_cost (shorts_count : ℕ) (shorts_price : ℚ)
                            (shoes_count : ℕ) (shoes_price : ℚ)
                            (tops_count : ℕ) (total_cost : ℚ) :
  shorts_count = 5 →
  shorts_price = 7 →
  shoes_count = 2 →
  shoes_price = 10 →
  tops_count = 4 →
  total_cost = 75 →
  ∃ (top_price : ℚ), top_price = 5 ∧
    total_cost = shorts_count * shorts_price + shoes_count * shoes_price + tops_count * top_price :=
by
  sorry


end anns_shopping_cost_l2007_200778


namespace common_tangents_l2007_200711

/-- The first curve: 9x^2 + 16y^2 = 144 -/
def curve1 (x y : ℝ) : Prop := 9 * x^2 + 16 * y^2 = 144

/-- The second curve: 7x^2 - 32y^2 = 224 -/
def curve2 (x y : ℝ) : Prop := 7 * x^2 - 32 * y^2 = 224

/-- A common tangent line: ax + by + c = 0 -/
def is_tangent (a b c : ℝ) : Prop :=
  ∀ x y : ℝ, (curve1 x y ∨ curve2 x y) → (a * x + b * y + c = 0 → 
    ∀ ε > 0, ∃ δ > 0, ∀ x' y' : ℝ, 
      ((x' - x)^2 + (y' - y)^2 < δ^2) → 
      ((curve1 x' y' ∨ curve2 x' y') → (a * x' + b * y' + c ≠ 0)))

/-- The theorem stating that the given equations are common tangents -/
theorem common_tangents : 
  (is_tangent 1 1 5 ∧ is_tangent 1 1 (-5) ∧ is_tangent 1 (-1) 5 ∧ is_tangent 1 (-1) (-5)) :=
sorry

end common_tangents_l2007_200711


namespace no_prime_solution_l2007_200701

theorem no_prime_solution : 
  ¬∃ (p : ℕ), Nat.Prime p ∧ 
  (p ^ 3 + 7) + (3 * p ^ 2 + 6) + (p ^ 2 + p + 3) + (p ^ 2 + 2 * p + 5) + 6 = 
  (p ^ 2 + 4 * p + 2) + (2 * p ^ 2 + 7 * p + 1) + (3 * p ^ 2 + 6 * p) :=
by sorry

end no_prime_solution_l2007_200701


namespace battery_current_at_12_ohms_l2007_200700

/-- Given a battery with voltage 48V and a relationship between current I and resistance R,
    prove that when R = 12Ω, I = 4A. -/
theorem battery_current_at_12_ohms :
  let voltage : ℝ := 48
  let R : ℝ := 12
  let I : ℝ := voltage / R
  I = 4 := by sorry

end battery_current_at_12_ohms_l2007_200700


namespace hex_B1F4_equals_45556_l2007_200719

/-- Represents a hexadecimal digit --/
inductive HexDigit
| D0 | D1 | D2 | D3 | D4 | D5 | D6 | D7 | D8 | D9
| A | B | C | D | E | F

/-- Convert a HexDigit to its decimal value --/
def hexToDecimal (h : HexDigit) : Nat :=
  match h with
  | HexDigit.D0 => 0
  | HexDigit.D1 => 1
  | HexDigit.D2 => 2
  | HexDigit.D3 => 3
  | HexDigit.D4 => 4
  | HexDigit.D5 => 5
  | HexDigit.D6 => 6
  | HexDigit.D7 => 7
  | HexDigit.D8 => 8
  | HexDigit.D9 => 9
  | HexDigit.A => 10
  | HexDigit.B => 11
  | HexDigit.C => 12
  | HexDigit.D => 13
  | HexDigit.E => 14
  | HexDigit.F => 15

/-- Convert a list of HexDigits to its decimal value --/
def hexListToDecimal (hexList : List HexDigit) : Nat :=
  hexList.foldr (fun digit acc => hexToDecimal digit + 16 * acc) 0

theorem hex_B1F4_equals_45556 :
  hexListToDecimal [HexDigit.B, HexDigit.D1, HexDigit.F, HexDigit.D4] = 45556 := by
  sorry

#eval hexListToDecimal [HexDigit.B, HexDigit.D1, HexDigit.F, HexDigit.D4]

end hex_B1F4_equals_45556_l2007_200719


namespace train_speed_is_45_km_per_hour_l2007_200716

-- Define the given parameters
def train_length : ℝ := 140
def bridge_length : ℝ := 235
def crossing_time : ℝ := 30

-- Define the conversion factor
def meters_per_second_to_km_per_hour : ℝ := 3.6

-- Theorem statement
theorem train_speed_is_45_km_per_hour :
  let total_distance := train_length + bridge_length
  let speed_in_meters_per_second := total_distance / crossing_time
  let speed_in_km_per_hour := speed_in_meters_per_second * meters_per_second_to_km_per_hour
  speed_in_km_per_hour = 45 := by
  sorry

end train_speed_is_45_km_per_hour_l2007_200716


namespace nora_fundraiser_solution_l2007_200725

/-- Represents the fundraising problem for Nora's school trip -/
def muffin_fundraiser (target : ℕ) (muffins_per_pack : ℕ) (packs_per_case : ℕ) (price_per_muffin : ℕ) : Prop :=
  ∃ (cases : ℕ),
    cases * (packs_per_case * muffins_per_pack * price_per_muffin) = target

/-- Theorem stating the solution to Nora's fundraising problem -/
theorem nora_fundraiser_solution :
  muffin_fundraiser 120 4 3 2 → (5 : ℕ) = 5 := by
  sorry

end nora_fundraiser_solution_l2007_200725


namespace income_comparison_l2007_200728

theorem income_comparison (Tim Mary Juan : ℝ) 
  (h1 : Mary = 1.60 * Tim) 
  (h2 : Mary = 1.28 * Juan) : 
  Tim = 0.80 * Juan := by
sorry

end income_comparison_l2007_200728


namespace line_slope_intercept_sum_l2007_200798

/-- Given a line passing through points (-3, 8) and (0, -1), prove that the sum of its slope and y-intercept is -4 -/
theorem line_slope_intercept_sum (m b : ℝ) : 
  (∀ x y : ℝ, (x = -3 ∧ y = 8) ∨ (x = 0 ∧ y = -1) → y = m * x + b) → 
  m + b = -4 := by
  sorry

end line_slope_intercept_sum_l2007_200798


namespace john_learning_alphabets_l2007_200742

/-- The number of alphabets John is learning in the first group -/
def alphabets_learned : ℕ := 15 / 3

/-- The number of days it takes John to learn one alphabet -/
def days_per_alphabet : ℕ := 3

/-- The total number of days John needs to finish learning the alphabets -/
def total_days : ℕ := 15

theorem john_learning_alphabets :
  alphabets_learned = 5 :=
by sorry

end john_learning_alphabets_l2007_200742


namespace hannahs_age_l2007_200765

/-- Given the ages of Eliza, Felipe, Gideon, and Hannah, prove Hannah's age -/
theorem hannahs_age 
  (eliza felipe gideon hannah : ℕ)
  (h1 : eliza = felipe - 4)
  (h2 : felipe = gideon + 6)
  (h3 : hannah = gideon + 2)
  (h4 : eliza = 15) :
  hannah = 15 := by
  sorry

end hannahs_age_l2007_200765


namespace quadratic_two_distinct_roots_l2007_200796

theorem quadratic_two_distinct_roots (a : ℝ) : 
  (∃ x y : ℝ, x ≠ y ∧ a * x^2 - 2*x + 1 = 0 ∧ a * y^2 - 2*y + 1 = 0) ↔ 
  (a < 1 ∧ a ≠ 0) :=
sorry

end quadratic_two_distinct_roots_l2007_200796


namespace solution_system_l2007_200790

theorem solution_system (x y : ℝ) 
  (h1 : x * y = 12)
  (h2 : x^2 * y + x * y^2 + x + y = 120) :
  x^2 + y^2 = 10344 / 169 := by
sorry

end solution_system_l2007_200790


namespace min_distance_intersection_points_l2007_200781

open Real

theorem min_distance_intersection_points (a : ℝ) :
  let f (x : ℝ) := (x - exp x - 3) / 2
  ∃ (x₁ x₂ : ℝ), a = 2 * x₁ - 3 ∧ a = x₂ + exp x₂ ∧ 
    ∀ (y₁ y₂ : ℝ), a = 2 * y₁ - 3 → a = y₂ + exp y₂ → 
      |x₂ - x₁| ≤ |y₂ - y₁| ∧ |x₂ - x₁| = 2 :=
by sorry

end min_distance_intersection_points_l2007_200781


namespace hyperbola_asymptotes_l2007_200731

/-- Given a hyperbola with equation x²/a² - y²/b² = 1, where a > 0, b > 0,
    and eccentricity 5/3, its asymptotes are y = ±(4/3)x -/
theorem hyperbola_asymptotes (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (∀ x y : ℝ, x^2 / a^2 - y^2 / b^2 = 1) →
  (a^2 + b^2) / a^2 = 25 / 9 →
  ∃ k : ℝ, k = 4/3 ∧ (∀ x y : ℝ, y = k * x ∨ y = -k * x) :=
by sorry

end hyperbola_asymptotes_l2007_200731


namespace intersection_of_A_and_B_l2007_200705

def A : Set Nat := {3, 5, 6, 8}
def B : Set Nat := {4, 5, 7, 8}

theorem intersection_of_A_and_B : A ∩ B = {5, 8} := by
  sorry

end intersection_of_A_and_B_l2007_200705


namespace min_wednesday_birthdays_l2007_200748

/-- Represents the number of employees with birthdays on each day of the week -/
structure BirthdayDistribution where
  wednesday : ℕ
  other : ℕ

/-- The conditions of the problem -/
def validDistribution (d : BirthdayDistribution) : Prop :=
  d.wednesday > d.other ∧
  d.wednesday + 6 * d.other = 50

/-- The theorem to prove -/
theorem min_wednesday_birthdays :
  ∀ d : BirthdayDistribution,
  validDistribution d →
  d.wednesday ≥ 8 :=
sorry

end min_wednesday_birthdays_l2007_200748


namespace right_triangle_area_l2007_200726

theorem right_triangle_area (base height : ℝ) (h1 : base = 15) (h2 : height = 10) :
  (1 / 2) * base * height = 75 := by
  sorry

end right_triangle_area_l2007_200726


namespace equivalent_statements_l2007_200795

-- Define the propositions
variable (P Q : Prop)

-- State the theorem
theorem equivalent_statements : 
  ((P → Q) ↔ (¬Q → ¬P)) ∧ ((P → Q) ↔ (¬P ∨ Q)) :=
sorry

end equivalent_statements_l2007_200795


namespace unique_solution_system_l2007_200721

theorem unique_solution_system (x y z : ℝ) :
  x > 0 ∧ y > 0 ∧ z > 0 →
  x^2 + y^2 + x*y = 7 →
  x^2 + z^2 + x*z = 13 →
  y^2 + z^2 + y*z = 19 →
  x = 1 ∧ y = 2 ∧ z = 3 := by sorry

end unique_solution_system_l2007_200721


namespace vicente_shopping_cost_l2007_200703

/-- Calculates the total amount spent in US dollars given the following conditions:
  - 5 kg of rice at €2 per kg with a 10% discount
  - 3 pounds of meat at £5 per pound with a 5% sales tax
  - Exchange rates: €1 = $1.20 and £1 = $1.35
-/
theorem vicente_shopping_cost :
  let rice_kg : ℝ := 5
  let rice_price_euro : ℝ := 2
  let meat_lb : ℝ := 3
  let meat_price_pound : ℝ := 5
  let rice_discount : ℝ := 0.1
  let meat_tax : ℝ := 0.05
  let euro_to_usd : ℝ := 1.20
  let pound_to_usd : ℝ := 1.35
  
  let rice_cost : ℝ := rice_kg * rice_price_euro * (1 - rice_discount) * euro_to_usd
  let meat_cost : ℝ := meat_lb * meat_price_pound * (1 + meat_tax) * pound_to_usd
  let total_cost : ℝ := rice_cost + meat_cost

  total_cost = 32.06 := by sorry

end vicente_shopping_cost_l2007_200703


namespace highest_percentage_increase_city_H_l2007_200797

structure City where
  name : String
  pop1990 : ℕ
  pop2000 : ℕ

def percentageIncrease (c : City) : ℚ :=
  ((c.pop2000 - c.pop1990) : ℚ) / (c.pop1990 : ℚ) * 100

def cities : List City := [
  ⟨"F", 60000, 78000⟩,
  ⟨"G", 80000, 96000⟩,
  ⟨"H", 70000, 91000⟩,
  ⟨"I", 85000, 94500⟩,
  ⟨"J", 95000, 114000⟩
]

theorem highest_percentage_increase_city_H :
  ∃ (c : City), c ∈ cities ∧ c.name = "H" ∧
  ∀ (other : City), other ∈ cities → percentageIncrease c ≥ percentageIncrease other :=
by sorry

end highest_percentage_increase_city_H_l2007_200797


namespace base_10_89_equals_base_4_1121_l2007_200751

/-- Converts a natural number to its base-4 representation as a list of digits -/
def toBase4 (n : ℕ) : List ℕ :=
  if n < 4 then [n]
  else (n % 4) :: toBase4 (n / 4)

/-- Converts a list of base-4 digits to a natural number -/
def fromBase4 (digits : List ℕ) : ℕ :=
  digits.foldr (fun d acc => d + 4 * acc) 0

/-- Theorem stating that 89 in base 10 is equal to 1121 in base 4 -/
theorem base_10_89_equals_base_4_1121 :
  fromBase4 [1, 2, 1, 1] = 89 := by
  sorry

#eval toBase4 89  -- Should output [1, 2, 1, 1]
#eval fromBase4 [1, 2, 1, 1]  -- Should output 89

end base_10_89_equals_base_4_1121_l2007_200751


namespace f_neg_two_eq_nine_l2007_200746

/-- The function f(x) = x^5 + ax^3 + x^2 + bx + 2 -/
def f (a b : ℝ) (x : ℝ) : ℝ := x^5 + a*x^3 + x^2 + b*x + 2

/-- Theorem: If f(2) = 3, then f(-2) = 9 -/
theorem f_neg_two_eq_nine {a b : ℝ} (h : f a b 2 = 3) : f a b (-2) = 9 := by
  sorry

end f_neg_two_eq_nine_l2007_200746


namespace plane_perp_from_line_perp_and_parallel_l2007_200774

-- Define the types for planes and lines
variable (Plane Line : Type)

-- Define the relationships between planes and lines
variable (perpendicular : Plane → Plane → Prop)
variable (parallel : Line → Plane → Prop)
variable (linePerpPlane : Line → Plane → Prop)

-- State the theorem
theorem plane_perp_from_line_perp_and_parallel
  (α β : Plane) (l : Line)
  (h1 : linePerpPlane l α)
  (h2 : parallel l β) :
  perpendicular α β :=
sorry

end plane_perp_from_line_perp_and_parallel_l2007_200774


namespace problem_solution_l2007_200727

theorem problem_solution (x y : ℝ) 
  (h1 : 3 * x + 2 * y = 7) 
  (h2 : 2 * x + 3 * y = 8) : 
  13 * x^2 + 22 * x * y + 13 * y^2 = 113 := by
  sorry

end problem_solution_l2007_200727


namespace count_numbers_with_digit_product_180_l2007_200717

def is_valid_digit (d : ℕ) : Prop := d ≥ 1 ∧ d ≤ 9

def is_five_digit_number (n : ℕ) : Prop :=
  n ≥ 10000 ∧ n < 100000

def digit_product (n : ℕ) : ℕ :=
  let digits := n.digits 10
  digits.prod

def count_valid_numbers : ℕ := sorry

theorem count_numbers_with_digit_product_180 :
  count_valid_numbers = 360 := by sorry

end count_numbers_with_digit_product_180_l2007_200717


namespace circle_selection_theorem_l2007_200755

/-- A figure with circles arranged in a specific pattern -/
structure CircleFigure where
  total_circles : ℕ
  horizontal_lines : ℕ
  diagonal_directions : ℕ

/-- The number of ways to choose three consecutive circles in a given direction -/
def consecutive_choices (n : ℕ) : ℕ := n * (n + 1) / 2

/-- The total number of ways to choose three consecutive circles in the figure -/
def total_choices (fig : CircleFigure) : ℕ :=
  consecutive_choices fig.horizontal_lines +
  fig.diagonal_directions * consecutive_choices (fig.horizontal_lines - 1)

/-- The main theorem stating the number of ways to choose three consecutive circles -/
theorem circle_selection_theorem (fig : CircleFigure) 
  (h1 : fig.total_circles = 33)
  (h2 : fig.horizontal_lines = 7)
  (h3 : fig.diagonal_directions = 2) :
  total_choices fig = 57 := by
  sorry

end circle_selection_theorem_l2007_200755


namespace group_size_problem_l2007_200739

theorem group_size_problem (x : ℕ) : 
  (5 * x + 45 = 7 * x + 3) → x = 21 := by
  sorry

end group_size_problem_l2007_200739


namespace system_solution_l2007_200788

theorem system_solution (w x y z : ℚ) 
  (eq1 : 2*w + x + y + z = 1)
  (eq2 : w + 2*x + y + z = 2)
  (eq3 : w + x + 2*y + z = 2)
  (eq4 : w + x + y + 2*z = 1) :
  w = -1/5 := by
  sorry

end system_solution_l2007_200788


namespace greatest_integer_quadratic_inequality_l2007_200732

theorem greatest_integer_quadratic_inequality :
  ∃ (n : ℤ), n^2 - 13*n + 36 ≤ 0 ∧
  n = 9 ∧
  ∀ (m : ℤ), m^2 - 13*m + 36 ≤ 0 → m ≤ n :=
by sorry

end greatest_integer_quadratic_inequality_l2007_200732


namespace max_value_sine_sum_l2007_200761

theorem max_value_sine_sum : 
  ∀ x : ℝ, 3 * Real.sin (x + π/9) + 5 * Real.sin (x + 4*π/9) ≤ 7 ∧ 
  ∃ x : ℝ, 3 * Real.sin (x + π/9) + 5 * Real.sin (x + 4*π/9) = 7 :=
sorry

end max_value_sine_sum_l2007_200761


namespace two_numbers_difference_l2007_200793

theorem two_numbers_difference (x y : ℝ) : 
  x < y ∧ x + y = 34 ∧ y = 22 → y - x = 10 := by
  sorry

end two_numbers_difference_l2007_200793


namespace product_of_numbers_l2007_200712

theorem product_of_numbers (x y : ℝ) : 
  x + y = 22 → x^2 + y^2 = 460 → x * y = 40 := by
  sorry

end product_of_numbers_l2007_200712


namespace prism_volume_l2007_200708

theorem prism_volume (x y z : Real) (h : Real) :
  x = Real.sqrt 9 →
  y = Real.sqrt 9 →
  h = 6 →
  (1 / 2 : Real) * x * y * h = 27 :=
by
  sorry

end prism_volume_l2007_200708


namespace truck_calculation_l2007_200791

/-- The number of trucks initially requested to transport 60 tons of cargo,
    where reducing each truck's capacity by 0.5 tons required 4 additional trucks. -/
def initial_trucks : ℕ := 20

/-- The total cargo to be transported in tons. -/
def total_cargo : ℝ := 60

/-- The reduction in capacity per truck in tons. -/
def capacity_reduction : ℝ := 0.5

/-- The number of additional trucks required after capacity reduction. -/
def additional_trucks : ℕ := 4

theorem truck_calculation :
  initial_trucks * (total_cargo / initial_trucks - capacity_reduction) = 
  (initial_trucks + additional_trucks) * ((total_cargo / initial_trucks) - capacity_reduction) ∧
  (initial_trucks + additional_trucks) * ((total_cargo / initial_trucks) - capacity_reduction) = total_cargo :=
sorry

end truck_calculation_l2007_200791


namespace inequality_solution_set_l2007_200789

theorem inequality_solution_set (a : ℝ) (h : a > 1) :
  let f := fun x : ℝ => (a - 1) * x^2 - a * x + 1
  (a = 2 → {x : ℝ | f x > 0} = {x : ℝ | x ≠ 1}) ∧
  (1 < a ∧ a < 2 → {x : ℝ | f x > 0} = {x : ℝ | x < 1 ∨ x > 1/(a-1)}) ∧
  (a > 2 → {x : ℝ | f x > 0} = {x : ℝ | x < 1/(a-1) ∨ x > 1}) :=
by sorry

end inequality_solution_set_l2007_200789


namespace min_value_of_expression_l2007_200736

theorem min_value_of_expression (x : ℝ) (h : x > 1) :
  x + 1 / (x - 1) ≥ 3 ∧ ∃ y > 1, y + 1 / (y - 1) = 3 := by
  sorry

end min_value_of_expression_l2007_200736


namespace prob_at_least_3_hits_l2007_200724

-- Define the probability of hitting the target on a single shot
def p_hit : ℝ := 0.8

-- Define the number of shots
def n_shots : ℕ := 4

-- Define the probability of hitting the target at least 3 times out of 4 shots
def p_at_least_3 : ℝ := 
  (Nat.choose n_shots 3 : ℝ) * p_hit^3 * (1 - p_hit) + p_hit^4

-- Theorem statement
theorem prob_at_least_3_hits : p_at_least_3 = 0.8192 := by
  sorry

end prob_at_least_3_hits_l2007_200724


namespace volume_of_one_gram_l2007_200752

/-- Given a substance with the following properties:
  - The mass of 1 cubic meter of the substance is 200 kg.
  - 1 kg = 1,000 grams
  - 1 cubic meter = 1,000,000 cubic centimeters
  
  This theorem proves that the volume of 1 gram of this substance is 5 cubic centimeters. -/
theorem volume_of_one_gram (mass_per_cubic_meter : ℝ) (kg_to_g : ℝ) (cubic_meter_to_cubic_cm : ℝ) :
  mass_per_cubic_meter = 200 →
  kg_to_g = 1000 →
  cubic_meter_to_cubic_cm = 1000000 →
  (1 : ℝ) / (mass_per_cubic_meter * kg_to_g / cubic_meter_to_cubic_cm) = 5 := by
  sorry

#check volume_of_one_gram

end volume_of_one_gram_l2007_200752


namespace chips_left_uneaten_l2007_200769

def cookies_per_dozen : ℕ := 12
def dozens_made : ℕ := 4
def chips_per_cookie : ℕ := 7
def fraction_eaten : ℚ := 1/2

theorem chips_left_uneaten : 
  (dozens_made * cookies_per_dozen * chips_per_cookie) * (1 - fraction_eaten) = 168 := by
  sorry

end chips_left_uneaten_l2007_200769


namespace tangent_slope_perpendicular_lines_l2007_200749

/-- The function f(x) = x^3 + 3ax --/
def f (a : ℝ) (x : ℝ) : ℝ := x^3 + 3*a*x

/-- The derivative of f(x) --/
def f_derivative (a : ℝ) (x : ℝ) : ℝ := 3*x^2 + 3*a

theorem tangent_slope_perpendicular_lines (a : ℝ) :
  (f_derivative a 1 = 6) ↔ ((-1 : ℝ) * (-a) = -1) :=
sorry

end tangent_slope_perpendicular_lines_l2007_200749


namespace tangent_line_y_intercept_l2007_200741

-- Define the function f(x) = x³ + 4x + 5
def f (x : ℝ) : ℝ := x^3 + 4*x + 5

-- Define the derivative of f
def f' (x : ℝ) : ℝ := 3*x^2 + 4

-- Theorem: The y-intercept of the tangent line to f(x) at x = 1 is (0, 3)
theorem tangent_line_y_intercept :
  let x₀ : ℝ := 1
  let y₀ : ℝ := f x₀
  let m : ℝ := f' x₀
  let b : ℝ := y₀ - m * x₀
  b = 3 := by sorry

end tangent_line_y_intercept_l2007_200741


namespace integer_fraction_triples_l2007_200794

theorem integer_fraction_triples :
  ∀ a b c : ℕ+,
    (a = 1 ∧ b = 20 ∧ c = 1) ∨
    (a = 1 ∧ b = 4 ∧ c = 1) ∨
    (a = 3 ∧ b = 4 ∧ c = 1) ↔
    ∃ k : ℤ, (32 * a.val + 3 * b.val + 48 * c.val) = 4 * k * a.val * b.val * c.val := by
  sorry

end integer_fraction_triples_l2007_200794


namespace stratified_sampling_ratio_l2007_200715

-- Define the total number of male and female students
def total_male : ℕ := 500
def total_female : ℕ := 400

-- Define the number of male students selected
def selected_male : ℕ := 25

-- Define the function to calculate the number of female students to be selected
def female_to_select : ℕ := (selected_male * total_female) / total_male

-- Theorem statement
theorem stratified_sampling_ratio :
  female_to_select = 20 := by
  sorry

end stratified_sampling_ratio_l2007_200715


namespace not_p_sufficient_not_necessary_for_not_q_l2007_200740

-- Define the conditions p and q
def p (x : ℝ) : Prop := |x + 1| > 2
def q (x : ℝ) : Prop := 5*x - 6 > x^2

-- State the theorem
theorem not_p_sufficient_not_necessary_for_not_q :
  (∀ x, ¬(p x) → ¬(q x)) ∧ 
  ¬(∀ x, ¬(q x) → ¬(p x)) := by
  sorry

end not_p_sufficient_not_necessary_for_not_q_l2007_200740


namespace sprocket_production_l2007_200707

theorem sprocket_production (machine_p machine_q machine_a : ℕ → ℕ) : 
  (∃ t_q : ℕ, 
    machine_p (t_q + 10) = 550 ∧ 
    machine_q t_q = 550 ∧ 
    (∀ t, machine_q t = (11 * machine_a t) / 10)) → 
  machine_a 1 = 5 := by
sorry

end sprocket_production_l2007_200707


namespace min_people_to_complete_task_l2007_200776

/-- Proves the minimum number of people needed to complete a task on time -/
theorem min_people_to_complete_task
  (total_days : ℕ)
  (days_worked : ℕ)
  (initial_people : ℕ)
  (work_completed : ℚ)
  (h1 : total_days = 40)
  (h2 : days_worked = 10)
  (h3 : initial_people = 12)
  (h4 : work_completed = 2 / 5)
  (h5 : days_worked < total_days) :
  let remaining_days := total_days - days_worked
  let remaining_work := 1 - work_completed
  let work_rate_per_day := work_completed / days_worked / initial_people
  ⌈(remaining_work / (work_rate_per_day * remaining_days))⌉ = 6 :=
by sorry

end min_people_to_complete_task_l2007_200776


namespace positive_real_inequality_l2007_200777

theorem positive_real_inequality (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) :
  (x / (x + 2*y + 3*z)) + (y / (y + 2*z + 3*x)) + (z / (z + 2*x + 3*y)) ≥ 1/2 := by
  sorry

end positive_real_inequality_l2007_200777


namespace xy_value_l2007_200786

theorem xy_value (x y : ℝ) (h : x * (x + y) = x^2 + 18) : x * y = 18 := by
  sorry

end xy_value_l2007_200786


namespace max_y_intercept_of_even_function_l2007_200773

def f (x a b : ℝ) : ℝ := x^2 + (a^2 + b^2 - 1)*x + a^2 + 2*a*b - b^2

theorem max_y_intercept_of_even_function
  (h : ∀ x, f x a b = f (-x) a b) :
  ∃ C, (∀ a b, f 0 a b ≤ C) ∧ (∃ a b, f 0 a b = C) ∧ C = Real.sqrt 2 :=
sorry

end max_y_intercept_of_even_function_l2007_200773


namespace angle_A_measure_l2007_200718

-- Define the triangle ABC
structure Triangle :=
  (a b c : ℝ)
  (A B C : ℝ)

-- State the theorem
theorem angle_A_measure (t : Triangle) (h1 : t.a = 7) (h2 : t.b = 8) (h3 : Real.cos t.B = 1/7) :
  t.A = π/3 := by
  sorry


end angle_A_measure_l2007_200718
