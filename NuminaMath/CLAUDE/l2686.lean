import Mathlib

namespace NUMINAMATH_CALUDE_quadratic_inequality_theorem_l2686_268690

-- Define the quadratic inequality and its solution set
def quadratic_inequality (a b : ℝ) (x : ℝ) : Prop := a * x^2 + x + b > 0
def solution_set (a b : ℝ) : Set ℝ := {x | x < -2 ∨ x > 1}

-- Define the second inequality
def second_inequality (a b c : ℝ) (x : ℝ) : Prop := a * x^2 - (c + b) * x + b * c < 0

-- Theorem statement
theorem quadratic_inequality_theorem :
  ∀ a b : ℝ, (∀ x : ℝ, quadratic_inequality a b x ↔ x ∈ solution_set a b) →
  (a = 1 ∧ b = -2) ∧
  (∀ c : ℝ, 
    (c = -2 → ∀ x : ℝ, ¬(second_inequality a b c x)) ∧
    (c > -2 → ∀ x : ℝ, second_inequality a b c x ↔ -2 < x ∧ x < c) ∧
    (c < -2 → ∀ x : ℝ, second_inequality a b c x ↔ c < x ∧ x < -2)) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_theorem_l2686_268690


namespace NUMINAMATH_CALUDE_juice_cans_bought_l2686_268625

-- Define the original price of ice cream
def original_ice_cream_price : ℚ := 12

-- Define the discount on ice cream
def ice_cream_discount : ℚ := 2

-- Define the price of juice
def juice_price : ℚ := 2

-- Define the number of cans in a set of juice
def cans_per_set : ℕ := 5

-- Define the total cost
def total_cost : ℚ := 24

-- Define the number of ice cream tubs bought
def ice_cream_tubs : ℕ := 2

-- Theorem to prove
theorem juice_cans_bought : ℕ := by
  -- The proof goes here
  sorry

#check juice_cans_bought

end NUMINAMATH_CALUDE_juice_cans_bought_l2686_268625


namespace NUMINAMATH_CALUDE_distribution_problem_l2686_268686

/-- The number of ways to distribute n indistinguishable objects among k distinct groups,
    with each group receiving at least one object. -/
def distribute (n k : ℕ) : ℕ := Nat.choose (n - 1) (k - 1)

/-- The problem statement -/
theorem distribution_problem :
  distribute 12 6 = 462 := by sorry

end NUMINAMATH_CALUDE_distribution_problem_l2686_268686


namespace NUMINAMATH_CALUDE_square_perimeter_problem_l2686_268641

theorem square_perimeter_problem (perimeter_I perimeter_II : ℝ) 
  (h1 : perimeter_I = 16)
  (h2 : perimeter_II = 36)
  (side_I : ℝ) (side_II : ℝ) (side_III : ℝ)
  (h3 : side_I = perimeter_I / 4)
  (h4 : side_II = perimeter_II / 4)
  (h5 : side_III = Real.sqrt (side_I * side_II))
  (perimeter_III : ℝ)
  (h6 : perimeter_III = 4 * side_III) :
  perimeter_III = 24 := by
sorry

end NUMINAMATH_CALUDE_square_perimeter_problem_l2686_268641


namespace NUMINAMATH_CALUDE_first_volume_pages_l2686_268626

/-- Given a two-volume book with a total of 999 digits used for page numbers,
    where the first volume has 9 more pages than the second volume,
    prove that the number of pages in the first volume is 207. -/
theorem first_volume_pages (total_digits : ℕ) (page_difference : ℕ) 
  (h1 : total_digits = 999)
  (h2 : page_difference = 9) :
  ∃ (first_volume second_volume : ℕ),
    first_volume = second_volume + page_difference ∧
    first_volume = 207 :=
by sorry

end NUMINAMATH_CALUDE_first_volume_pages_l2686_268626


namespace NUMINAMATH_CALUDE_triangle_equality_equilateral_is_isosceles_equilateral_is_acute_equilateral_is_oblique_l2686_268654

/-- A triangle with side lengths satisfying a² + b² + c² = ab + bc + ca is equilateral -/
theorem triangle_equality (a b c : ℝ) (h : 0 < a ∧ 0 < b ∧ 0 < c) 
  (eq : a^2 + b^2 + c^2 = a*b + b*c + c*a) : a = b ∧ b = c := by
  sorry

/-- An equilateral triangle is isosceles -/
theorem equilateral_is_isosceles (a b c : ℝ) (h : a = b ∧ b = c) : 
  a = b ∨ b = c ∨ a = c := by
  sorry

/-- An equilateral triangle is acute-angled -/
theorem equilateral_is_acute (a b c : ℝ) (h : a = b ∧ b = c) (pos : 0 < a) : 
  a^2 + b^2 > c^2 ∧ b^2 + c^2 > a^2 ∧ c^2 + a^2 > b^2 := by
  sorry

/-- An equilateral triangle is oblique (not right-angled) -/
theorem equilateral_is_oblique (a b c : ℝ) (h : a = b ∧ b = c) (pos : 0 < a) : 
  a^2 + b^2 ≠ c^2 ∧ b^2 + c^2 ≠ a^2 ∧ c^2 + a^2 ≠ b^2 := by
  sorry

end NUMINAMATH_CALUDE_triangle_equality_equilateral_is_isosceles_equilateral_is_acute_equilateral_is_oblique_l2686_268654


namespace NUMINAMATH_CALUDE_arithmetic_sequence_a2_l2686_268670

/-- An arithmetic sequence {aₙ} -/
def arithmetic_sequence (a : ℕ → ℚ) : Prop :=
  ∃ (a₁ d : ℚ), ∀ n, a n = a₁ + (n - 1) * d

theorem arithmetic_sequence_a2 (a : ℕ → ℚ) 
  (h_arith : arithmetic_sequence a) 
  (h_sum : a 3 + a 5 = 15) 
  (h_a6 : a 6 = 7) : 
  a 2 = 8 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_a2_l2686_268670


namespace NUMINAMATH_CALUDE_simplify_expression_l2686_268621

theorem simplify_expression (x : ℝ) : 105 * x - 58 * x = 47 * x := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l2686_268621


namespace NUMINAMATH_CALUDE_sum_min_max_cubic_quartic_l2686_268652

theorem sum_min_max_cubic_quartic (a b c d : ℝ) 
  (sum_condition : a + b + c + d = 8)
  (sum_squares_condition : a^2 + b^2 + c^2 + d^2 = 18) : 
  let f := fun (x y z w : ℝ) => 3 * (x^3 + y^3 + z^3 + w^3) - 2 * (x^4 + y^4 + z^4 + w^4)
  ∃ (m M : ℝ), (∀ (x y z w : ℝ), (x + y + z + w = 8 ∧ x^2 + y^2 + z^2 + w^2 = 18) → 
    m ≤ f x y z w ∧ f x y z w ≤ M) ∧ m + M = 29 :=
sorry

end NUMINAMATH_CALUDE_sum_min_max_cubic_quartic_l2686_268652


namespace NUMINAMATH_CALUDE_unique_cut_l2686_268636

/-- Represents a cut of the original number -/
structure Cut where
  pos1 : Nat
  pos2 : Nat
  valid : pos1 < pos2 ∧ pos2 < 5

/-- Checks if a given cut produces the required difference -/
def isValidCut (c : Cut) : Prop :=
  let part1 := 12345 / (10^(5 - c.pos1))
  let part2 := (12345 / (10^(5 - c.pos2))) % (10^(c.pos2 - c.pos1))
  let part3 := 12345 % (10^(5 - c.pos2))
  (part1 * 10^4 + part2 * 10^(5 - c.pos2) + part3) -
  (part2 * 10^4 + part3 * 10^c.pos1 + part1) = 28926

theorem unique_cut : 
  ∃! c : Cut, isValidCut c ∧ c.pos1 = 1 ∧ c.pos2 = 4 :=
sorry

end NUMINAMATH_CALUDE_unique_cut_l2686_268636


namespace NUMINAMATH_CALUDE_x_positive_necessary_not_sufficient_for_abs_x_minus_one_less_than_one_l2686_268616

theorem x_positive_necessary_not_sufficient_for_abs_x_minus_one_less_than_one :
  (∃ x : ℝ, x > 0 ∧ ¬(|x - 1| < 1)) ∧
  (∀ x : ℝ, |x - 1| < 1 → x > 0) :=
by sorry

end NUMINAMATH_CALUDE_x_positive_necessary_not_sufficient_for_abs_x_minus_one_less_than_one_l2686_268616


namespace NUMINAMATH_CALUDE_cottage_pie_mince_usage_l2686_268630

/-- Given information about a school cafeteria's use of ground mince for lasagnas and cottage pies,
    prove that each cottage pie uses 3 pounds of ground mince. -/
theorem cottage_pie_mince_usage
  (total_dishes : Nat)
  (lasagna_count : Nat)
  (cottage_pie_count : Nat)
  (total_mince : Nat)
  (mince_per_lasagna : Nat)
  (h1 : total_dishes = lasagna_count + cottage_pie_count)
  (h2 : total_dishes = 100)
  (h3 : lasagna_count = 100)
  (h4 : cottage_pie_count = 100)
  (h5 : total_mince = 500)
  (h6 : mince_per_lasagna = 2) :
  (total_mince - lasagna_count * mince_per_lasagna) / cottage_pie_count = 3 := by
  sorry

end NUMINAMATH_CALUDE_cottage_pie_mince_usage_l2686_268630


namespace NUMINAMATH_CALUDE_pizza_piece_cost_l2686_268637

/-- Represents the cost of pizzas and their division into pieces -/
structure PizzaPurchase where
  totalCost : ℕ        -- Total cost in dollars
  numPizzas : ℕ        -- Number of pizzas
  piecesPerPizza : ℕ   -- Number of pieces each pizza is cut into

/-- Calculates the cost per piece of pizza -/
def costPerPiece (purchase : PizzaPurchase) : ℚ :=
  (purchase.totalCost : ℚ) / (purchase.numPizzas * purchase.piecesPerPizza)

/-- Theorem: Given 4 pizzas cost $80 and each pizza is cut into 5 pieces, 
    the cost per piece is $4 -/
theorem pizza_piece_cost : 
  let purchase := PizzaPurchase.mk 80 4 5
  costPerPiece purchase = 4 := by
  sorry

end NUMINAMATH_CALUDE_pizza_piece_cost_l2686_268637


namespace NUMINAMATH_CALUDE_athlete_running_time_l2686_268698

/-- Proof that an athlete spends 35 minutes running given the conditions -/
theorem athlete_running_time 
  (calories_per_minute_running : ℕ) 
  (calories_per_minute_walking : ℕ)
  (total_calories_burned : ℕ)
  (total_time : ℕ)
  (h1 : calories_per_minute_running = 10)
  (h2 : calories_per_minute_walking = 4)
  (h3 : total_calories_burned = 450)
  (h4 : total_time = 60) :
  ∃ (running_time : ℕ), 
    running_time = 35 ∧ 
    running_time + (total_time - running_time) = total_time ∧
    calories_per_minute_running * running_time + 
    calories_per_minute_walking * (total_time - running_time) = total_calories_burned :=
by
  sorry


end NUMINAMATH_CALUDE_athlete_running_time_l2686_268698


namespace NUMINAMATH_CALUDE_parallel_lines_problem_l2686_268664

/-- The number of parallelograms formed by the intersection of two sets of parallel lines -/
def num_parallelograms (n : ℕ) (m : ℕ) : ℕ := n.choose 2 * m.choose 2

/-- The theorem statement -/
theorem parallel_lines_problem (n : ℕ) :
  num_parallelograms n 8 = 420 ↔ n = 6 := by sorry

end NUMINAMATH_CALUDE_parallel_lines_problem_l2686_268664


namespace NUMINAMATH_CALUDE_parabola_intersection_l2686_268646

/-- First parabola equation -/
def f (x : ℝ) : ℝ := 3 * x^2 - 4 * x + 2

/-- Second parabola equation -/
def g (x : ℝ) : ℝ := -x^2 + 6 * x + 8

/-- Theorem stating that (-0.5, 4.75) and (3, 17) are the only intersection points -/
theorem parabola_intersection :
  (∀ x y : ℝ, f x = g x ∧ y = f x ↔ (x = -0.5 ∧ y = 4.75) ∨ (x = 3 ∧ y = 17)) := by
  sorry

end NUMINAMATH_CALUDE_parabola_intersection_l2686_268646


namespace NUMINAMATH_CALUDE_magician_earnings_proof_l2686_268665

def magician_earnings (price : ℕ) (initial_decks : ℕ) (final_decks : ℕ) : ℕ :=
  (initial_decks - final_decks) * price

theorem magician_earnings_proof (price : ℕ) (initial_decks : ℕ) (final_decks : ℕ) 
  (h1 : price = 2)
  (h2 : initial_decks = 5)
  (h3 : final_decks = 3) :
  magician_earnings price initial_decks final_decks = 4 := by
  sorry

end NUMINAMATH_CALUDE_magician_earnings_proof_l2686_268665


namespace NUMINAMATH_CALUDE_three_digit_divisibility_l2686_268610

theorem three_digit_divisibility (a b : ℕ) (h1 : a < 10) (h2 : b < 10) 
  (h3 : (a + b) % 7 = 0) : (101 * a + 10 * b) % 7 = 0 := by
  sorry

end NUMINAMATH_CALUDE_three_digit_divisibility_l2686_268610


namespace NUMINAMATH_CALUDE_two_in_A_l2686_268634

def A : Set ℝ := {x | x > 1}

theorem two_in_A : 2 ∈ A := by
  sorry

end NUMINAMATH_CALUDE_two_in_A_l2686_268634


namespace NUMINAMATH_CALUDE_tangent_line_perpendicular_and_equation_l2686_268640

/-- The tangent line to y = x^4 at (1, 1) is perpendicular to x + 4y - 8 = 0 and has equation 4x - y - 3 = 0 -/
theorem tangent_line_perpendicular_and_equation (x y : ℝ) : 
  let f : ℝ → ℝ := fun x => x^4
  let tangent_slope : ℝ := (deriv f) 1
  let perpendicular_line_slope : ℝ := -1/4
  let tangent_equation : ℝ → ℝ → Prop := fun x y => 4*x - y - 3 = 0
  tangent_slope * perpendicular_line_slope = -1 ∧
  tangent_equation 1 1 ∧
  (∀ x y, tangent_equation x y ↔ y - 1 = tangent_slope * (x - 1)) := by
sorry

end NUMINAMATH_CALUDE_tangent_line_perpendicular_and_equation_l2686_268640


namespace NUMINAMATH_CALUDE_monotone_f_range_l2686_268672

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x < 1 then a^x else x^2 + 4/x + a * Real.log x

theorem monotone_f_range (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) :
  (∀ x y : ℝ, x < y → f a x < f a y) → 2 ≤ a ∧ a ≤ 5 := by sorry

end NUMINAMATH_CALUDE_monotone_f_range_l2686_268672


namespace NUMINAMATH_CALUDE_largest_common_number_l2686_268608

def is_in_first_sequence (n : ℕ) : Prop := ∃ k : ℕ, n = 1 + 8 * k

def is_in_second_sequence (n : ℕ) : Prop := ∃ m : ℕ, n = 4 + 9 * m

def is_in_range (n : ℕ) : Prop := 1 ≤ n ∧ n ≤ 250

theorem largest_common_number :
  (is_in_first_sequence 193 ∧ is_in_second_sequence 193 ∧ is_in_range 193) ∧
  ∀ n : ℕ, is_in_first_sequence n → is_in_second_sequence n → is_in_range n → n ≤ 193 :=
by sorry

end NUMINAMATH_CALUDE_largest_common_number_l2686_268608


namespace NUMINAMATH_CALUDE_right_triangle_with_hypotenuse_39_l2686_268692

theorem right_triangle_with_hypotenuse_39 :
  ∀ a b c : ℕ,
  a^2 + b^2 = c^2 →  -- Pythagorean theorem for right triangle
  c = 39 →           -- Hypotenuse length is 39
  (a = 15 ∧ b = 36) ∨ (a = 36 ∧ b = 15) := by
sorry

end NUMINAMATH_CALUDE_right_triangle_with_hypotenuse_39_l2686_268692


namespace NUMINAMATH_CALUDE_hayley_stickers_l2686_268651

/-- The number of Hayley's close friends who like stickers. -/
def num_friends : ℕ := 9

/-- The number of stickers each friend would receive if distributed equally. -/
def stickers_per_friend : ℕ := 8

/-- The total number of stickers Hayley has. -/
def total_stickers : ℕ := num_friends * stickers_per_friend

theorem hayley_stickers : total_stickers = 72 := by
  sorry

end NUMINAMATH_CALUDE_hayley_stickers_l2686_268651


namespace NUMINAMATH_CALUDE_hyperbola_equation_l2686_268687

/-- A hyperbola with right focus at (5, 0) and an asymptote with equation 2x - y = 0 
    has the standard equation x²/5 - y²/20 = 1 -/
theorem hyperbola_equation (x y : ℝ) : 
  let right_focus : ℝ × ℝ := (5, 0)
  let asymptote (x y : ℝ) : Prop := 2 * x - y = 0
  x^2 / 5 - y^2 / 20 = 1 :=
by sorry


end NUMINAMATH_CALUDE_hyperbola_equation_l2686_268687


namespace NUMINAMATH_CALUDE_cos_squared_alpha_plus_pi_fourth_l2686_268684

theorem cos_squared_alpha_plus_pi_fourth (α : ℝ) (h : Real.sin (2 * α) = 2 / 3) :
  Real.cos (α + π / 4) ^ 2 = 1 / 6 := by
  sorry

end NUMINAMATH_CALUDE_cos_squared_alpha_plus_pi_fourth_l2686_268684


namespace NUMINAMATH_CALUDE_solve_linear_equations_l2686_268696

theorem solve_linear_equations :
  (∃ y : ℚ, 8 * y - 4 * (3 * y + 2) = 6 ∧ y = -7/2) ∧
  (∃ x : ℚ, 2 - (x + 2) / 3 = x - (x - 1) / 6 ∧ x = 1) :=
by sorry

end NUMINAMATH_CALUDE_solve_linear_equations_l2686_268696


namespace NUMINAMATH_CALUDE_unique_solution_l2686_268675

/-- The number of communications between any n-2 people -/
def communications (n : ℕ) : ℕ := 3^(Nat.succ 0)

/-- The theorem stating that 5 is the only solution -/
theorem unique_solution :
  ∀ n : ℕ,
  (n > 0) →
  (∀ m : ℕ, m = communications n) →
  (∀ i j : Fin n, i ≠ j → (∃! x : ℕ, x ≤ 1)) →
  n = 5 :=
sorry

end NUMINAMATH_CALUDE_unique_solution_l2686_268675


namespace NUMINAMATH_CALUDE_points_difference_is_integer_impossible_score_difference_l2686_268666

/-- Represents the possible outcomes of a chess game -/
inductive GameOutcome
  | Victory
  | Draw
  | Defeat

/-- Calculates the points scored for a given game outcome -/
def points_scored (outcome : GameOutcome) : ℚ :=
  match outcome with
  | GameOutcome.Victory => 1
  | GameOutcome.Draw => 1/2
  | GameOutcome.Defeat => 0

/-- Calculates the points lost for a given game outcome -/
def points_lost (outcome : GameOutcome) : ℚ :=
  match outcome with
  | GameOutcome.Victory => 0
  | GameOutcome.Draw => 1/2
  | GameOutcome.Defeat => 1

/-- Represents a sequence of game outcomes in a chess tournament -/
def Tournament := List GameOutcome

/-- Calculates the total points scored in a tournament -/
def total_points_scored (tournament : Tournament) : ℚ :=
  tournament.map points_scored |>.sum

/-- Calculates the total points lost in a tournament -/
def total_points_lost (tournament : Tournament) : ℚ :=
  tournament.map points_lost |>.sum

/-- Theorem: The difference between points scored and points lost in any chess tournament is always an integer -/
theorem points_difference_is_integer (tournament : Tournament) :
  ∃ n : ℤ, total_points_scored tournament - total_points_lost tournament = n :=
sorry

/-- Corollary: It's impossible to have scored exactly 3.5 points more than lost -/
theorem impossible_score_difference (tournament : Tournament) :
  total_points_scored tournament - total_points_lost tournament ≠ 7/2 :=
sorry

end NUMINAMATH_CALUDE_points_difference_is_integer_impossible_score_difference_l2686_268666


namespace NUMINAMATH_CALUDE_probability_sum_greater_than_six_l2686_268601

def dice_outcomes : ℕ := 6 * 6

def favorable_outcomes : ℕ := 21

theorem probability_sum_greater_than_six :
  (favorable_outcomes : ℚ) / dice_outcomes = 7 / 12 :=
sorry

end NUMINAMATH_CALUDE_probability_sum_greater_than_six_l2686_268601


namespace NUMINAMATH_CALUDE_females_only_in_orchestra_l2686_268691

/-- Represents the membership data for the band and orchestra --/
structure MusicGroups where
  band_females : ℕ
  band_males : ℕ
  orchestra_females : ℕ
  orchestra_males : ℕ
  both_females : ℕ
  total_members : ℕ

/-- The theorem stating the number of females in the orchestra who are not in the band --/
theorem females_only_in_orchestra (mg : MusicGroups)
  (h1 : mg.band_females = 120)
  (h2 : mg.band_males = 100)
  (h3 : mg.orchestra_females = 100)
  (h4 : mg.orchestra_males = 120)
  (h5 : mg.both_females = 80)
  (h6 : mg.total_members = 260) :
  mg.orchestra_females - mg.both_females = 20 := by
  sorry

#check females_only_in_orchestra

end NUMINAMATH_CALUDE_females_only_in_orchestra_l2686_268691


namespace NUMINAMATH_CALUDE_object_with_22_opposite_directions_is_clock_l2686_268689

/-- An object with hands that can show opposite directions -/
structure ObjectWithHands :=
  (oppositeDirectionsPerDay : ℕ)

/-- Definition of a clock based on its behavior -/
def isClock (obj : ObjectWithHands) : Prop :=
  obj.oppositeDirectionsPerDay = 22

/-- Theorem stating that an object with hands showing opposite directions 22 times a day is a clock -/
theorem object_with_22_opposite_directions_is_clock (obj : ObjectWithHands) :
  obj.oppositeDirectionsPerDay = 22 → isClock obj :=
by
  sorry

#check object_with_22_opposite_directions_is_clock

end NUMINAMATH_CALUDE_object_with_22_opposite_directions_is_clock_l2686_268689


namespace NUMINAMATH_CALUDE_circle_region_area_l2686_268632

/-- Given a circle with radius 36 and two chords of length 90 intersecting at a point 12 units from the center,
    the area of one of the regions formed can be expressed as 216π, which is equivalent to aπ - b√c
    where a + b + c = 216 and a, b, c are positive integers with c not divisible by the square of any prime. -/
theorem circle_region_area (r : ℝ) (chord_length : ℝ) (intersection_distance : ℝ)
  (h_radius : r = 36)
  (h_chord : chord_length = 90)
  (h_intersection : intersection_distance = 12) :
  ∃ (a b c : ℕ), 
    (a > 0 ∧ b > 0 ∧ c > 0) ∧
    (∀ (p : ℕ), Prime p → c % (p^2) ≠ 0) ∧
    (a + b + c = 216) ∧
    (Real.pi * (a : ℝ) - (b : ℝ) * Real.sqrt (c : ℝ) = 216 * Real.pi) :=
by sorry

end NUMINAMATH_CALUDE_circle_region_area_l2686_268632


namespace NUMINAMATH_CALUDE_cuboid_volume_l2686_268620

/-- Given a cuboid with face areas 3, 5, and 15 sharing a common vertex, its volume is 15 -/
theorem cuboid_volume (a b c : ℝ) 
  (h1 : a * b = 3) 
  (h2 : a * c = 5) 
  (h3 : b * c = 15) : 
  a * b * c = 15 := by
  sorry

#check cuboid_volume

end NUMINAMATH_CALUDE_cuboid_volume_l2686_268620


namespace NUMINAMATH_CALUDE_unique_valid_n_l2686_268677

def is_valid_n (n : ℕ) : Prop :=
  ∃ (a b c : ℕ),
    1 ≤ a ∧ a ≤ 9 ∧
    0 ≤ b ∧ b ≤ 9 ∧
    0 ≤ c ∧ c ≤ 9 ∧
    n = 10 * a + b ∧
    100 * a + 10 * c + b = 6 * n

theorem unique_valid_n :
  ∃! n : ℕ, n ≥ 10 ∧ is_valid_n n ∧ n = 18 :=
sorry

end NUMINAMATH_CALUDE_unique_valid_n_l2686_268677


namespace NUMINAMATH_CALUDE_inverse_proportion_problem_l2686_268631

/-- Two real numbers are inversely proportional -/
def InverselyProportional (x y : ℝ) : Prop :=
  ∃ k : ℝ, k ≠ 0 ∧ x * y = k

theorem inverse_proportion_problem (x y : ℝ → ℝ) 
  (h1 : InverselyProportional (x 40) (y 5))
  (h2 : x 40 = 40)
  (h3 : y 5 = 5) :
  y 8 = 25 := by
  sorry

end NUMINAMATH_CALUDE_inverse_proportion_problem_l2686_268631


namespace NUMINAMATH_CALUDE_complex_division_l2686_268612

theorem complex_division (i : ℂ) (h : i^2 = -1) : 
  (1 - 2*i) / (1 + i) = -1/2 - 3/2*i := by
  sorry

end NUMINAMATH_CALUDE_complex_division_l2686_268612


namespace NUMINAMATH_CALUDE_problem_solution_l2686_268602

theorem problem_solution : 
  ∃ x : ℝ, (28 + x / 69) * 69 = 1980 ∧ x = 1952 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l2686_268602


namespace NUMINAMATH_CALUDE_min_value_expression_l2686_268642

theorem min_value_expression (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (h_abc : a * b * c = 27) :
  a^2 + 9*a*b + 9*b^2 + 3*c^2 ≥ 60 ∧ 
  ∃ (a₀ b₀ c₀ : ℝ), a₀ > 0 ∧ b₀ > 0 ∧ c₀ > 0 ∧ a₀ * b₀ * c₀ = 27 ∧ 
    a₀^2 + 9*a₀*b₀ + 9*b₀^2 + 3*c₀^2 = 60 :=
by sorry

end NUMINAMATH_CALUDE_min_value_expression_l2686_268642


namespace NUMINAMATH_CALUDE_bill_face_value_l2686_268679

/-- Proves that given a true discount of 360 and a banker's discount of 432, 
    the face value of the bill is 1800. -/
theorem bill_face_value (TD : ℕ) (BD : ℕ) (FV : ℕ) : 
  TD = 360 → BD = 432 → FV = (TD^2) / (BD - TD) → FV = 1800 := by
  sorry

#check bill_face_value

end NUMINAMATH_CALUDE_bill_face_value_l2686_268679


namespace NUMINAMATH_CALUDE_simon_is_10_years_old_l2686_268682

/-- Simon's age given Alvin's age and their relationship -/
def simon_age (alvin_age : ℕ) (age_difference : ℕ) : ℕ :=
  alvin_age / 2 - age_difference

theorem simon_is_10_years_old :
  simon_age 30 5 = 10 := by
  sorry

end NUMINAMATH_CALUDE_simon_is_10_years_old_l2686_268682


namespace NUMINAMATH_CALUDE_sum_of_roots_quadratic_sum_of_roots_specific_quadratic_l2686_268648

theorem sum_of_roots_quadratic (a b c : ℝ) (h : a ≠ 0) :
  let roots := {x : ℝ | a * x^2 + b * x + c = 0}
  (∃ x y : ℝ, roots = {x, y}) →
  (∃ s : ℝ, ∀ z ∈ roots, ∃ w ∈ roots, z + w = s) →
  (∃ s : ℝ, ∀ z ∈ roots, ∃ w ∈ roots, z + w = s) → s = -b / a :=
by sorry

theorem sum_of_roots_specific_quadratic :
  let roots := {x : ℝ | x^2 - 6*x + 8 = 0}
  (∃ x y : ℝ, roots = {x, y}) →
  (∃ s : ℝ, ∀ z ∈ roots, ∃ w ∈ roots, z + w = s) →
  s = 6 :=
by sorry

end NUMINAMATH_CALUDE_sum_of_roots_quadratic_sum_of_roots_specific_quadratic_l2686_268648


namespace NUMINAMATH_CALUDE_cube_split_l2686_268604

/-- The sum of the first n natural numbers -/
def triangular_number (n : ℕ) : ℕ := n * (n + 1) / 2

/-- The nth odd number -/
def nth_odd (n : ℕ) : ℕ := 2 * n - 1

theorem cube_split (m : ℕ) (h1 : m > 1) :
  (∃ k : ℕ, k > 0 ∧ k < m ∧ nth_odd (triangular_number k + 1) = 59) →
  m = 8 := by sorry

end NUMINAMATH_CALUDE_cube_split_l2686_268604


namespace NUMINAMATH_CALUDE_rectangle_square_assembly_l2686_268661

structure Rectangle where
  width : ℝ
  height : ℝ

def Square (s : ℝ) : Rectangle :=
  { width := s, height := s }

def totalArea (rectangles : List Rectangle) : ℝ :=
  rectangles.map (λ r => r.width * r.height) |>.sum

def isSquare (area : ℝ) : Prop :=
  ∃ s : ℝ, s > 0 ∧ s * s = area

theorem rectangle_square_assembly
  (s : ℝ)
  (r1 : Rectangle)
  (r2 : Rectangle)
  (h1 : r1.width = 10 ∧ r1.height = 24)
  (h2 : r2 ∈ [
    Rectangle.mk 2 24,
    Rectangle.mk 19 17.68,
    Rectangle.mk 34 10,
    Rectangle.mk 34 44,
    Rectangle.mk 14 24,
    Rectangle.mk 14 17,
    Rectangle.mk 24 38
  ]) :
  isSquare (totalArea [Square s, Square s, r1, r2]) :=
by sorry

end NUMINAMATH_CALUDE_rectangle_square_assembly_l2686_268661


namespace NUMINAMATH_CALUDE_stream_speed_l2686_268697

/-- Proves that given a boat with a speed of 8 kmph in standing water,
    traveling a round trip of 420 km (210 km each way) in 56 hours,
    the speed of the stream is 2 kmph. -/
theorem stream_speed (boat_speed : ℝ) (distance : ℝ) (total_time : ℝ) :
  boat_speed = 8 →
  distance = 210 →
  total_time = 56 →
  ∃ (stream_speed : ℝ),
    stream_speed = 2 ∧
    (distance / (boat_speed - stream_speed) + distance / (boat_speed + stream_speed) = total_time) :=
by sorry

end NUMINAMATH_CALUDE_stream_speed_l2686_268697


namespace NUMINAMATH_CALUDE_complex_equation_l2686_268650

theorem complex_equation (z : ℂ) (h : z * Complex.I = 1 + Complex.I) : z = 1 - Complex.I := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_l2686_268650


namespace NUMINAMATH_CALUDE_smallest_k_for_sum_4n_plus_1_l2686_268688

/-- Given a positive integer n, M is the set of integers from 1 to 2n -/
def M (n : ℕ+) : Finset ℕ := Finset.range (2 * n) \ {0}

/-- A function that checks if a subset of M contains 4 distinct elements summing to 4n + 1 -/
def has_sum_4n_plus_1 (n : ℕ+) (S : Finset ℕ) : Prop :=
  ∃ (a b c d : ℕ), a ∈ S ∧ b ∈ S ∧ c ∈ S ∧ d ∈ S ∧
    a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d ∧
    a + b + c + d = 4 * n + 1

theorem smallest_k_for_sum_4n_plus_1 (n : ℕ+) :
  (∀ (S : Finset ℕ), S ⊆ M n → S.card = n + 3 → has_sum_4n_plus_1 n S) ∧
  (∃ (T : Finset ℕ), T ⊆ M n ∧ T.card = n + 2 ∧ ¬has_sum_4n_plus_1 n T) :=
sorry

end NUMINAMATH_CALUDE_smallest_k_for_sum_4n_plus_1_l2686_268688


namespace NUMINAMATH_CALUDE_total_students_count_l2686_268635

/-- The number of students per team -/
def students_per_team : ℕ := 18

/-- The number of teams -/
def number_of_teams : ℕ := 9

/-- The total number of students -/
def total_students : ℕ := students_per_team * number_of_teams

theorem total_students_count : total_students = 162 := by
  sorry

end NUMINAMATH_CALUDE_total_students_count_l2686_268635


namespace NUMINAMATH_CALUDE_min_cookies_eaten_l2686_268674

/-- Represents the number of cookies at each stage of the process -/
structure CookieCount where
  initial : ℕ
  after_first : ℕ
  after_second : ℕ
  after_third : ℕ
  evening : ℕ

/-- Defines the cookie distribution process -/
def distribute_cookies (c : CookieCount) : Prop :=
  c.after_first = (2 * (c.initial - 1)) / 3 ∧
  c.after_second = (2 * (c.after_first - 1)) / 3 ∧
  c.after_third = (2 * (c.after_second - 1)) / 3 ∧
  c.evening = c.after_third - 1

/-- Defines the evening distribution condition -/
def evening_distribution (c : CookieCount) (n : ℕ) : Prop :=
  c.evening = 3 * n

/-- Defines the condition that no cookies are broken -/
def no_broken_cookies (c : CookieCount) : Prop :=
  c.initial % 1 = 0 ∧
  c.after_first % 1 = 0 ∧
  c.after_second % 1 = 0 ∧
  c.after_third % 1 = 0 ∧
  c.evening % 1 = 0

/-- Theorem stating the minimum number of cookies Xiao Wang could have eaten -/
theorem min_cookies_eaten (c : CookieCount) (n : ℕ) :
  distribute_cookies c →
  evening_distribution c n →
  no_broken_cookies c →
  (c.initial - c.after_first) = 6 ∧ n = 7 := by
  sorry

#check min_cookies_eaten

end NUMINAMATH_CALUDE_min_cookies_eaten_l2686_268674


namespace NUMINAMATH_CALUDE_triangle_side_value_l2686_268614

theorem triangle_side_value (a b c : ℝ) (A B C : ℝ) (R : ℝ) :
  a > 0 → b > 0 → c > 0 →
  A > 0 → B > 0 → C > 0 →
  a * b = 60 →
  (1 / 2) * a * b * Real.sin C = 15 * Real.sqrt 3 →
  R = Real.sqrt 3 →
  c = 2 * R * Real.sin C →
  c = 3 := by
sorry

end NUMINAMATH_CALUDE_triangle_side_value_l2686_268614


namespace NUMINAMATH_CALUDE_one_third_of_6_3_l2686_268622

theorem one_third_of_6_3 : (6.3 : ℚ) / 3 = 21 / 10 := by
  sorry

end NUMINAMATH_CALUDE_one_third_of_6_3_l2686_268622


namespace NUMINAMATH_CALUDE_x_minus_y_value_l2686_268611

theorem x_minus_y_value (x y : ℝ) (h1 : x^2 = 9) (h2 : |y| = 4) (h3 : x < y) :
  x - y = -7 ∨ x - y = -1 := by
sorry

end NUMINAMATH_CALUDE_x_minus_y_value_l2686_268611


namespace NUMINAMATH_CALUDE_habitable_land_area_l2686_268695

/-- Calculates the area of habitable land in a rectangular field with a circular pond. -/
theorem habitable_land_area (length width diagonal pond_radius : ℝ) 
  (h_length : length = 23)
  (h_diagonal : diagonal = 33)
  (h_width : width^2 = diagonal^2 - length^2)
  (h_pond_radius : pond_radius = 3) : 
  ∃ (area : ℝ), abs (area - 515.91) < 0.01 ∧ 
  area = length * width - π * pond_radius^2 := by
sorry

end NUMINAMATH_CALUDE_habitable_land_area_l2686_268695


namespace NUMINAMATH_CALUDE_roses_in_garden_l2686_268613

theorem roses_in_garden (rows : ℕ) (roses_per_row : ℕ) 
  (red_fraction : ℚ) (white_fraction : ℚ) :
  rows = 10 →
  roses_per_row = 20 →
  red_fraction = 1/2 →
  white_fraction = 3/5 →
  (rows * roses_per_row * (1 - red_fraction) * (1 - white_fraction) : ℚ) = 40 := by
  sorry

end NUMINAMATH_CALUDE_roses_in_garden_l2686_268613


namespace NUMINAMATH_CALUDE_gcf_of_12_and_16_l2686_268619

theorem gcf_of_12_and_16 (n : ℕ) : 
  n = 12 → Nat.lcm n 16 = 48 → Nat.gcd n 16 = 4 := by
  sorry

end NUMINAMATH_CALUDE_gcf_of_12_and_16_l2686_268619


namespace NUMINAMATH_CALUDE_square_cut_diagonal_length_l2686_268669

theorem square_cut_diagonal_length (s x : ℝ) : 
  s > 0 → 
  x > 0 → 
  x^2 = 72 → 
  s^2 = 2 * x^2 → 
  (s - 2*x)^2 + (s - 2*x)^2 = 12^2 := by
sorry

end NUMINAMATH_CALUDE_square_cut_diagonal_length_l2686_268669


namespace NUMINAMATH_CALUDE_dog_food_duration_l2686_268678

/-- Given a dog's feeding schedule and a bag of dog food, calculate how many days the food will last. -/
theorem dog_food_duration (morning_food evening_food bag_size : ℕ) : 
  morning_food = 1 → 
  evening_food = 1 → 
  bag_size = 32 → 
  (bag_size / (morning_food + evening_food) : ℕ) = 16 := by
sorry

end NUMINAMATH_CALUDE_dog_food_duration_l2686_268678


namespace NUMINAMATH_CALUDE_perpendicular_slope_correct_l2686_268618

-- Define the slope of the given line
def given_line_slope : ℚ := 3 / 4

-- Define the slope of the perpendicular line
def perpendicular_slope : ℚ := -4 / 3

-- Theorem stating that the perpendicular slope is correct
theorem perpendicular_slope_correct :
  perpendicular_slope = -1 / given_line_slope :=
by sorry

end NUMINAMATH_CALUDE_perpendicular_slope_correct_l2686_268618


namespace NUMINAMATH_CALUDE_max_fraction_sum_l2686_268658

def DigitSet : Set Nat := {2, 3, 4, 5, 6, 7, 8, 9}

def ValidOptions : Set Rat := {2/17, 3/17, 17/72, 25/72, 13/36}

theorem max_fraction_sum (A B C D : Nat) :
  A ∈ DigitSet → B ∈ DigitSet → C ∈ DigitSet → D ∈ DigitSet →
  A ≠ B → A ≠ C → A ≠ D → B ≠ C → B ≠ D → C ≠ D →
  (A : Rat) / B + (C : Rat) / D ∈ ValidOptions →
  ∀ (X Y Z W : Nat), X ∈ DigitSet → Y ∈ DigitSet → Z ∈ DigitSet → W ∈ DigitSet →
    X ≠ Y → X ≠ Z → X ≠ W → Y ≠ Z → Y ≠ W → Z ≠ W →
    (X : Rat) / Y + (Z : Rat) / W ∈ ValidOptions →
    (X : Rat) / Y + (Z : Rat) / W ≤ (A : Rat) / B + (C : Rat) / D →
  (A : Rat) / B + (C : Rat) / D = 25 / 72 :=
sorry

end NUMINAMATH_CALUDE_max_fraction_sum_l2686_268658


namespace NUMINAMATH_CALUDE_largest_integer_squared_less_than_ten_million_l2686_268624

theorem largest_integer_squared_less_than_ten_million :
  ∃ (n : ℕ), n > 0 ∧ n^2 < 10000000 ∧ ∀ (m : ℕ), m > n → m^2 ≥ 10000000 :=
by sorry

end NUMINAMATH_CALUDE_largest_integer_squared_less_than_ten_million_l2686_268624


namespace NUMINAMATH_CALUDE_partnership_profit_theorem_l2686_268627

/-- Represents the profit distribution in a partnership --/
structure Partnership where
  investment_ratio : ℝ  -- Ratio of A's investment to B's investment
  time_ratio : ℝ        -- Ratio of A's investment time to B's investment time
  b_profit : ℝ          -- B's profit

/-- Calculates the total profit of the partnership --/
def total_profit (p : Partnership) : ℝ :=
  let a_profit := p.b_profit * p.investment_ratio * p.time_ratio
  a_profit + p.b_profit

/-- Theorem stating that under the given conditions, the total profit is 28000 --/
theorem partnership_profit_theorem (p : Partnership) 
  (h1 : p.investment_ratio = 3)
  (h2 : p.time_ratio = 2)
  (h3 : p.b_profit = 4000) :
  total_profit p = 28000 := by
  sorry

#eval total_profit { investment_ratio := 3, time_ratio := 2, b_profit := 4000 }

end NUMINAMATH_CALUDE_partnership_profit_theorem_l2686_268627


namespace NUMINAMATH_CALUDE_expression_simplification_l2686_268668

theorem expression_simplification (x y : ℝ) (hx : x ≥ 0) :
  (3 / 5) * Real.sqrt (x * y^2) / (-(4 / 15) * Real.sqrt (y / x)) * (-(5 / 6) * Real.sqrt (x^3 * y)) =
  (15 * x^2 * y * Real.sqrt x) / 8 :=
by sorry

end NUMINAMATH_CALUDE_expression_simplification_l2686_268668


namespace NUMINAMATH_CALUDE_mutually_exclusive_events_l2686_268655

/-- Represents the outcome of a single shot --/
inductive ShotOutcome
  | Hit
  | Miss

/-- Represents the outcome of two shots --/
def TwoShotOutcome := (ShotOutcome × ShotOutcome)

/-- The event of hitting the target at least once --/
def hitAtLeastOnce (outcome : TwoShotOutcome) : Prop :=
  outcome.1 = ShotOutcome.Hit ∨ outcome.2 = ShotOutcome.Hit

/-- The event of missing the target both times --/
def missBothTimes (outcome : TwoShotOutcome) : Prop :=
  outcome.1 = ShotOutcome.Miss ∧ outcome.2 = ShotOutcome.Miss

/-- Theorem stating that "missing the target both times" is the mutually exclusive event of "hitting the target at least once" --/
theorem mutually_exclusive_events :
  ∀ (outcome : TwoShotOutcome), hitAtLeastOnce outcome ↔ ¬(missBothTimes outcome) :=
sorry

end NUMINAMATH_CALUDE_mutually_exclusive_events_l2686_268655


namespace NUMINAMATH_CALUDE_sum_of_distances_inequality_minimum_value_of_expression_l2686_268600

-- Part 1
theorem sum_of_distances_inequality (x y : ℝ) :
  Real.sqrt (x^2 + y^2) + Real.sqrt ((x-1)^2 + y^2) +
  Real.sqrt (x^2 + (y-1)^2) + Real.sqrt ((x-1)^2 + (y-1)^2) ≥ 2 * Real.sqrt 2 :=
sorry

-- Part 2
theorem minimum_value_of_expression :
  ∃ (min : ℝ), min = 8 ∧
  ∀ (a b : ℝ), abs a ≤ Real.sqrt 2 → b > 0 →
  (a - b)^2 + (Real.sqrt (2 - a^2) - 9 / b)^2 ≥ min :=
sorry

end NUMINAMATH_CALUDE_sum_of_distances_inequality_minimum_value_of_expression_l2686_268600


namespace NUMINAMATH_CALUDE_uncool_parents_count_l2686_268667

theorem uncool_parents_count (total : ℕ) (cool_dads : ℕ) (cool_moms : ℕ) (both_cool : ℕ)
  (h1 : total = 50)
  (h2 : cool_dads = 25)
  (h3 : cool_moms = 30)
  (h4 : both_cool = 15) :
  total - (cool_dads - both_cool + cool_moms - both_cool + both_cool) = 10 := by
  sorry

end NUMINAMATH_CALUDE_uncool_parents_count_l2686_268667


namespace NUMINAMATH_CALUDE_at_least_one_not_less_than_two_l2686_268609

theorem at_least_one_not_less_than_two (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (a + 1/b ≥ 2) ∨ (b + 1/a ≥ 2) := by
  sorry

end NUMINAMATH_CALUDE_at_least_one_not_less_than_two_l2686_268609


namespace NUMINAMATH_CALUDE_triangle_area_l2686_268615

theorem triangle_area (a b c : ℝ) (A : ℝ) :
  A = π / 3 →  -- 60° in radians
  a = Real.sqrt 3 →
  b + c = 3 →
  (1 / 2) * b * c * Real.sin A = Real.sqrt 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_triangle_area_l2686_268615


namespace NUMINAMATH_CALUDE_remainder_problem_l2686_268603

theorem remainder_problem (n : ℕ) : 
  n % 12 = 22 → 
  ((n % 34) + (n % 12)) % 12 = 10 → 
  n % 34 = 10 := by
sorry

end NUMINAMATH_CALUDE_remainder_problem_l2686_268603


namespace NUMINAMATH_CALUDE_sqrt_four_cubes_sum_l2686_268639

theorem sqrt_four_cubes_sum : Real.sqrt (4^3 + 4^3 + 4^3 + 4^3) = 16 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_four_cubes_sum_l2686_268639


namespace NUMINAMATH_CALUDE_circle_problem_l2686_268681

-- Define the circle
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

-- Define the conditions
def passes_through (c : Circle) (p : ℝ × ℝ) : Prop :=
  (c.center.1 - p.1)^2 + (c.center.2 - p.2)^2 = c.radius^2

def tangent_to_line (c : Circle) (a b d : ℝ) : Prop :=
  ∃ (x y : ℝ), a * x + b * y = d ∧ (c.center.1 - x)^2 + (c.center.2 - y)^2 = c.radius^2

def center_on_line (c : Circle) (m b : ℝ) : Prop :=
  c.center.2 = m * c.center.1 + b

-- Define the theorem
theorem circle_problem :
  ∃ (c : Circle),
    passes_through c (2, -1) ∧
    tangent_to_line c 1 1 1 ∧
    center_on_line c (-2) 0 ∧
    c.center = (1, -2) ∧
    c.radius^2 = 2 ∧
    (∀ (x y : ℝ), (x - 1)^2 + (y + 2)^2 = 2 ↔ passes_through c (x, y)) ∧
    (let chord_length := 2 * Real.sqrt (c.radius^2 - (3 * c.center.1 + 4 * c.center.2)^2 / 25);
     chord_length = 2) :=
by sorry

end NUMINAMATH_CALUDE_circle_problem_l2686_268681


namespace NUMINAMATH_CALUDE_square_perimeter_problem_l2686_268643

theorem square_perimeter_problem (A B C : ℝ) : 
  -- A, B, and C represent the side lengths of squares A, B, and C respectively
  (4 * A = 20) →  -- Perimeter of A is 20 units
  (4 * B = 32) →  -- Perimeter of B is 32 units
  (C = A / 2 + 2 * B) →  -- Side length of C definition
  (4 * C = 74) -- Perimeter of C is 74 units
  := by sorry

end NUMINAMATH_CALUDE_square_perimeter_problem_l2686_268643


namespace NUMINAMATH_CALUDE_deck_card_count_l2686_268694

theorem deck_card_count : ∀ (r n : ℕ), 
  (n = 2 * r) →                           -- Initially, black cards are twice red cards
  (n + 4 = 3 * r) →                       -- After adding 4 black cards, black is triple red
  (r + n = 12) :=                         -- Initial total number of cards is 12
by
  sorry

end NUMINAMATH_CALUDE_deck_card_count_l2686_268694


namespace NUMINAMATH_CALUDE_cube_diagonal_length_l2686_268663

theorem cube_diagonal_length (surface_area : ℝ) (h : surface_area = 864) :
  let side_length := Real.sqrt (surface_area / 6)
  let diagonal_length := side_length * Real.sqrt 3
  diagonal_length = 12 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_cube_diagonal_length_l2686_268663


namespace NUMINAMATH_CALUDE_gain_percentage_proof_l2686_268680

/-- Proves that the gain percentage is 20% when selling 20 articles for $60,
    given that selling 29.99999625000047 articles for $60 would result in a 20% loss. -/
theorem gain_percentage_proof (articles_sold : ℝ) (total_price : ℝ) (loss_articles : ℝ) 
  (h1 : articles_sold = 20)
  (h2 : total_price = 60)
  (h3 : loss_articles = 29.99999625000047)
  (h4 : (0.8 * (loss_articles * (total_price / articles_sold))) = total_price) :
  (((total_price / articles_sold) - (total_price / loss_articles)) / (total_price / loss_articles)) * 100 = 20 :=
by sorry

end NUMINAMATH_CALUDE_gain_percentage_proof_l2686_268680


namespace NUMINAMATH_CALUDE_binary_to_octal_conversion_l2686_268647

/-- Convert a binary number represented as a list of bits to decimal --/
def binary_to_decimal (bits : List Bool) : ℕ :=
  bits.enum.foldl (fun acc (i, b) => acc + if b then 2^i else 0) 0

/-- Convert a decimal number to octal --/
def decimal_to_octal (n : ℕ) : List ℕ :=
  if n = 0 then [0] else
    let rec aux (m : ℕ) (acc : List ℕ) :=
      if m = 0 then acc
      else aux (m / 8) ((m % 8) :: acc)
    aux n []

theorem binary_to_octal_conversion :
  let binary := [false, false, true, true, true, false, true, true]
  let decimal := binary_to_decimal binary.reverse
  let octal := decimal_to_octal decimal
  decimal = 220 ∧ octal = [3, 3, 4] := by
  sorry

#eval binary_to_decimal [false, false, true, true, true, false, true, true].reverse
#eval decimal_to_octal 220

end NUMINAMATH_CALUDE_binary_to_octal_conversion_l2686_268647


namespace NUMINAMATH_CALUDE_sum_x_y_equals_negative_one_l2686_268673

theorem sum_x_y_equals_negative_one (x y : ℝ) (h : |x - 1| + (y + 2)^2 = 0) : x + y = -1 := by
  sorry

end NUMINAMATH_CALUDE_sum_x_y_equals_negative_one_l2686_268673


namespace NUMINAMATH_CALUDE_complement_intersection_theorem_l2686_268638

universe u

def U : Set ℕ := {0, 1, 2, 3, 4}
def M : Set ℕ := {0, 1, 2}
def N : Set ℕ := {2, 3}

theorem complement_intersection_theorem :
  (U \ M) ∩ N = {3} := by sorry

end NUMINAMATH_CALUDE_complement_intersection_theorem_l2686_268638


namespace NUMINAMATH_CALUDE_specific_triangle_toothpicks_l2686_268653

/-- Represents the configuration of a large equilateral triangle made of small triangles --/
structure TriangleConfig where
  rows : Nat
  base_triangles : Nat
  double_count_start : Nat

/-- Calculates the total number of toothpicks required for a given triangle configuration --/
def total_toothpicks (config : TriangleConfig) : Nat :=
  sorry

/-- Theorem stating that the specific configuration requires 1617 toothpicks --/
theorem specific_triangle_toothpicks :
  let config : TriangleConfig := {
    rows := 5,
    base_triangles := 100,
    double_count_start := 2
  }
  total_toothpicks config = 1617 := by
  sorry

end NUMINAMATH_CALUDE_specific_triangle_toothpicks_l2686_268653


namespace NUMINAMATH_CALUDE_contrapositive_equality_l2686_268629

theorem contrapositive_equality (a b : ℝ) :
  (¬(a * b = 0) ↔ (a ≠ 0 ∧ b ≠ 0)) ↔
  ((a * b = 0) → (a = 0 ∨ b = 0)) :=
by sorry

end NUMINAMATH_CALUDE_contrapositive_equality_l2686_268629


namespace NUMINAMATH_CALUDE_dans_money_was_three_l2686_268659

/-- Dan's initial amount of money, given he bought a candy bar and has some money left -/
def dans_initial_money (candy_bar_cost : ℝ) (money_left : ℝ) : ℝ :=
  candy_bar_cost + money_left

/-- Theorem stating Dan's initial money was $3 -/
theorem dans_money_was_three :
  dans_initial_money 1 2 = 3 := by
  sorry

end NUMINAMATH_CALUDE_dans_money_was_three_l2686_268659


namespace NUMINAMATH_CALUDE_opposite_of_sqrt_three_l2686_268607

theorem opposite_of_sqrt_three : -(Real.sqrt 3) = -Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_opposite_of_sqrt_three_l2686_268607


namespace NUMINAMATH_CALUDE_christinas_walking_speed_l2686_268644

/-- The problem of finding Christina's walking speed -/
theorem christinas_walking_speed 
  (initial_distance : ℝ) 
  (jacks_speed : ℝ) 
  (lindys_speed : ℝ) 
  (lindys_total_distance : ℝ) 
  (h1 : initial_distance = 270) 
  (h2 : jacks_speed = 4) 
  (h3 : lindys_speed = 8) 
  (h4 : lindys_total_distance = 240) : 
  ∃ (christinas_speed : ℝ), christinas_speed = 5 := by
  sorry

#check christinas_walking_speed

end NUMINAMATH_CALUDE_christinas_walking_speed_l2686_268644


namespace NUMINAMATH_CALUDE_sunglasses_and_caps_probability_l2686_268623

theorem sunglasses_and_caps_probability 
  (total_sunglasses : ℕ) 
  (total_caps : ℕ) 
  (prob_cap_given_sunglasses : ℚ) : 
  total_sunglasses = 60 → 
  total_caps = 40 → 
  prob_cap_given_sunglasses = 1/3 → 
  (total_sunglasses * prob_cap_given_sunglasses : ℚ) / total_caps = 1/2 :=
by sorry

end NUMINAMATH_CALUDE_sunglasses_and_caps_probability_l2686_268623


namespace NUMINAMATH_CALUDE_complex_equation_sum_l2686_268628

theorem complex_equation_sum (a b : ℝ) : 
  (2 : ℂ) / (1 - Complex.I) = Complex.mk a b → a + b = 2 := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_sum_l2686_268628


namespace NUMINAMATH_CALUDE_min_shared_side_length_l2686_268633

/-- Given three triangles ABC, DBC, and EBC sharing side BC, prove that the minimum possible
    integer length of BC is 8 cm. -/
theorem min_shared_side_length
  (AB : ℝ) (AC : ℝ) (DC : ℝ) (BD : ℝ) (EC : ℝ)
  (h_AB : AB = 7)
  (h_AC : AC = 15)
  (h_DC : DC = 9)
  (h_BD : BD = 12)
  (h_EC : EC = 11)
  : ∃ (BC : ℕ), BC ≥ 8 ∧ 
    (∀ (BC' : ℕ), BC' ≥ 8 → BC' > AC - AB) ∧
    (∀ (BC' : ℕ), BC' ≥ 8 → BC' > BD - DC) ∧
    (∀ (BC' : ℕ), BC' ≥ 8 → BC' > 0) ∧
    (∀ (BC'' : ℕ), BC'' < BC → 
      (BC'' ≤ AC - AB ∨ BC'' ≤ BD - DC ∨ BC'' ≤ 0)) :=
by sorry

end NUMINAMATH_CALUDE_min_shared_side_length_l2686_268633


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_set_l2686_268683

theorem quadratic_inequality_solution_set :
  {x : ℝ | x^2 + x - 2 ≤ 0} = {x : ℝ | -2 ≤ x ∧ x ≤ 1} := by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_set_l2686_268683


namespace NUMINAMATH_CALUDE_exists_unsolvable_configuration_l2686_268671

/-- Represents a chessboard with integers -/
def Chessboard := Matrix (Fin 2018) (Fin 2019) ℤ

/-- Represents a set of selected cells on the chessboard -/
def SelectedCells := Set (Fin 2018 × Fin 2019)

/-- Performs one step of the operation on the chessboard -/
def perform_operation (board : Chessboard) (selected : SelectedCells) : Chessboard :=
  sorry

/-- Checks if all numbers on the board are equal -/
def all_equal (board : Chessboard) : Prop :=
  sorry

/-- Theorem stating that there exists a chessboard configuration where it's impossible to make all numbers equal -/
theorem exists_unsolvable_configuration :
  ∃ (initial_board : Chessboard),
    ∀ (operations : List SelectedCells),
      ¬(all_equal (operations.foldl perform_operation initial_board)) :=
sorry

end NUMINAMATH_CALUDE_exists_unsolvable_configuration_l2686_268671


namespace NUMINAMATH_CALUDE_largest_prime_factor_of_1001_l2686_268617

theorem largest_prime_factor_of_1001 : 
  ∃ (p : Nat), Nat.Prime p ∧ p ∣ 1001 ∧ ∀ (q : Nat), Nat.Prime q → q ∣ 1001 → q ≤ p ∧ p = 13 :=
by sorry

end NUMINAMATH_CALUDE_largest_prime_factor_of_1001_l2686_268617


namespace NUMINAMATH_CALUDE_M_congruent_544_mod_1000_l2686_268693

/-- The number of distinguishable arrangements of flags on two flagpoles -/
def M : ℕ :=
  let total_blue := 20
  let total_green := 15
  let total_slots := total_blue + 1
  let ways_to_arrange_greens := Nat.choose total_slots total_green
  let ways_to_divide_poles := total_slots
  let arrangements_with_empty_pole := Nat.choose total_blue total_green
  ways_to_divide_poles * ways_to_arrange_greens - 2 * arrangements_with_empty_pole

/-- The theorem stating that M is congruent to 544 modulo 1000 -/
theorem M_congruent_544_mod_1000 : M ≡ 544 [ZMOD 1000] := by
  sorry

end NUMINAMATH_CALUDE_M_congruent_544_mod_1000_l2686_268693


namespace NUMINAMATH_CALUDE_ravi_overall_profit_l2686_268657

/-- Calculates the overall profit for Ravi's purchases and sales -/
theorem ravi_overall_profit (refrigerator_cost mobile_cost : ℝ)
  (refrigerator_loss_percent mobile_profit_percent : ℝ) :
  refrigerator_cost = 15000 →
  mobile_cost = 8000 →
  refrigerator_loss_percent = 4 →
  mobile_profit_percent = 10 →
  let refrigerator_selling_price := refrigerator_cost * (1 - refrigerator_loss_percent / 100)
  let mobile_selling_price := mobile_cost * (1 + mobile_profit_percent / 100)
  let total_cost := refrigerator_cost + mobile_cost
  let total_selling_price := refrigerator_selling_price + mobile_selling_price
  total_selling_price - total_cost = 200 :=
by sorry

end NUMINAMATH_CALUDE_ravi_overall_profit_l2686_268657


namespace NUMINAMATH_CALUDE_equal_function_values_l2686_268606

/-- Given a function f(x) = ax^2 - 2ax + 1 where a > 1, prove that f(x₁) = f(x₂) when x₁ < x₂ and x₁ + x₂ = 1 + a -/
theorem equal_function_values
  (a : ℝ) (x₁ x₂ : ℝ) 
  (ha : a > 1)
  (hx : x₁ < x₂)
  (hsum : x₁ + x₂ = 1 + a)
  : a * x₁^2 - 2*a*x₁ + 1 = a * x₂^2 - 2*a*x₂ + 1 :=
by sorry

end NUMINAMATH_CALUDE_equal_function_values_l2686_268606


namespace NUMINAMATH_CALUDE_gcd_180_270_l2686_268662

theorem gcd_180_270 : Nat.gcd 180 270 = 90 := by
  sorry

end NUMINAMATH_CALUDE_gcd_180_270_l2686_268662


namespace NUMINAMATH_CALUDE_max_abs_z_given_condition_l2686_268645

theorem max_abs_z_given_condition (z : ℂ) (h : Complex.abs (z + Complex.I) + Complex.abs (z - Complex.I) = 2) : 
  Complex.abs z ≤ 1 := by
  sorry

end NUMINAMATH_CALUDE_max_abs_z_given_condition_l2686_268645


namespace NUMINAMATH_CALUDE_mutually_exclusive_but_not_converse_l2686_268699

/-- Represents the outcome of rolling a single die -/
inductive DieOutcome
  | One
  | Two
  | Three
  | Four
  | Five
  | Six

/-- Represents the outcome of rolling two dice -/
def TwoDiceOutcome := DieOutcome × DieOutcome

/-- Checks if a die outcome is odd -/
def isOdd (outcome : DieOutcome) : Bool :=
  match outcome with
  | DieOutcome.One => true
  | DieOutcome.Three => true
  | DieOutcome.Five => true
  | _ => false

/-- Event: Exactly one odd number -/
def exactlyOneOdd (outcome : TwoDiceOutcome) : Prop :=
  (isOdd outcome.1 && !isOdd outcome.2) || (!isOdd outcome.1 && isOdd outcome.2)

/-- Event: Exactly two odd numbers -/
def exactlyTwoOdd (outcome : TwoDiceOutcome) : Prop :=
  isOdd outcome.1 && isOdd outcome.2

/-- The sample space of all possible outcomes when rolling two fair six-sided dice -/
def sampleSpace : Set TwoDiceOutcome := sorry

theorem mutually_exclusive_but_not_converse :
  (∀ (outcome : TwoDiceOutcome), ¬(exactlyOneOdd outcome ∧ exactlyTwoOdd outcome)) ∧
  (∃ (outcome : TwoDiceOutcome), ¬exactlyOneOdd outcome ∧ ¬exactlyTwoOdd outcome) :=
sorry

end NUMINAMATH_CALUDE_mutually_exclusive_but_not_converse_l2686_268699


namespace NUMINAMATH_CALUDE_factorization_proof_l2686_268605

theorem factorization_proof (x y : ℝ) : 91 * x^7 - 273 * x^14 * y^3 = 91 * x^7 * (1 - 3 * x^7 * y^3) := by
  sorry

end NUMINAMATH_CALUDE_factorization_proof_l2686_268605


namespace NUMINAMATH_CALUDE_correlation_count_correlated_relationships_l2686_268676

/-- Represents a relationship between two quantities -/
structure Relationship where
  name : String
  has_correlation : Bool

/-- The set of relationships given in the problem -/
def relationships : List Relationship := [
  ⟨"cube volume-edge length", false⟩,
  ⟨"yield-fertilizer", true⟩,
  ⟨"height-age", true⟩,
  ⟨"expenses-income", true⟩,
  ⟨"electricity consumption-price", false⟩
]

/-- The correct answer is that exactly three relationships have correlations -/
theorem correlation_count :
  (relationships.filter (fun r => r.has_correlation)).length = 3 := by
  sorry

/-- The relationships with correlations are yield-fertilizer, height-age, and expenses-income -/
theorem correlated_relationships :
  (relationships.filter (fun r => r.has_correlation)).map (fun r => r.name) =
    ["yield-fertilizer", "height-age", "expenses-income"] := by
  sorry

end NUMINAMATH_CALUDE_correlation_count_correlated_relationships_l2686_268676


namespace NUMINAMATH_CALUDE_sum_middle_m_value_l2686_268660

/-- An arithmetic sequence with 3m terms -/
structure ArithmeticSequence (m : ℕ) where
  sum_first_2m : ℝ
  sum_last_2m : ℝ

/-- The sum of the middle m terms in an arithmetic sequence -/
def sum_middle_m (seq : ArithmeticSequence m) : ℝ := sorry

theorem sum_middle_m_value {m : ℕ} (seq : ArithmeticSequence m)
  (h1 : seq.sum_first_2m = 100)
  (h2 : seq.sum_last_2m = 200) :
  sum_middle_m seq = 75 := by sorry

end NUMINAMATH_CALUDE_sum_middle_m_value_l2686_268660


namespace NUMINAMATH_CALUDE_oak_trees_planted_l2686_268649

/-- Given the initial and final number of oak trees in a park, 
    prove that the number of new trees planted is their difference -/
theorem oak_trees_planted (initial final : ℕ) (h : final ≥ initial) :
  final - initial = final - initial :=
by sorry

end NUMINAMATH_CALUDE_oak_trees_planted_l2686_268649


namespace NUMINAMATH_CALUDE_period_of_cosine_l2686_268656

theorem period_of_cosine (x : ℝ) :
  let f : ℝ → ℝ := λ x ↦ Real.cos ((3 * x) / 4)
  ∃ T : ℝ, T > 0 ∧ ∀ x, f (x + T) = f x ∧ T = (8 * Real.pi) / 3 :=
by
  sorry

end NUMINAMATH_CALUDE_period_of_cosine_l2686_268656


namespace NUMINAMATH_CALUDE_opposite_face_of_four_l2686_268685

/-- Represents the six faces of a cube -/
inductive Face
| A | B | C | D | E | F

/-- Assigns numbers to the faces of the cube -/
def face_value : Face → ℕ
| Face.A => 3
| Face.B => 4
| Face.C => 5
| Face.D => 6
| Face.E => 7
| Face.F => 8

/-- Defines the opposite face relation -/
def opposite : Face → Face
| Face.A => Face.F
| Face.B => Face.E
| Face.C => Face.D
| Face.D => Face.C
| Face.E => Face.B
| Face.F => Face.A

theorem opposite_face_of_four (h : ∀ (f : Face), face_value f + face_value (opposite f) = 11) :
  face_value (opposite Face.B) = 7 := by
  sorry

end NUMINAMATH_CALUDE_opposite_face_of_four_l2686_268685
