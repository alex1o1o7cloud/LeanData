import Mathlib

namespace equilateral_triangle_perimeter_l1693_169389

/-- The perimeter of an equilateral triangle whose area is numerically equal to twice its side length is 8√3. -/
theorem equilateral_triangle_perimeter : 
  ∀ s : ℝ, s > 0 → (s^2 * Real.sqrt 3) / 4 = 2 * s → 3 * s = 8 * Real.sqrt 3 := by
  sorry

end equilateral_triangle_perimeter_l1693_169389


namespace unique_intersection_iff_k_eq_22_div_3_l1693_169313

/-- The parabola function -/
def parabola (y : ℝ) : ℝ := -3 * y^2 + 2 * y + 7

/-- The line function -/
def line (k : ℝ) : ℝ := k

/-- The condition for exactly one intersection point -/
def has_unique_intersection (k : ℝ) : Prop :=
  ∃! y, parabola y = line k

theorem unique_intersection_iff_k_eq_22_div_3 :
  ∀ k : ℝ, has_unique_intersection k ↔ k = 22 / 3 := by sorry

end unique_intersection_iff_k_eq_22_div_3_l1693_169313


namespace quotient_problem_l1693_169351

theorem quotient_problem (y : ℝ) (h : 5 / y = 5.3) : y = 26.5 := by
  sorry

end quotient_problem_l1693_169351


namespace fescue_percentage_in_Y_l1693_169309

/-- Represents a seed mixture --/
structure SeedMixture where
  ryegrass : ℝ
  bluegrass : ℝ
  fescue : ℝ

/-- The combined mixture of X and Y --/
def CombinedMixture (X Y : SeedMixture) (xWeight : ℝ) : SeedMixture :=
  { ryegrass := xWeight * X.ryegrass + (1 - xWeight) * Y.ryegrass,
    bluegrass := xWeight * X.bluegrass + (1 - xWeight) * Y.bluegrass,
    fescue := xWeight * X.fescue + (1 - xWeight) * Y.fescue }

theorem fescue_percentage_in_Y
  (X : SeedMixture)
  (Y : SeedMixture)
  (h1 : X.ryegrass = 0.4)
  (h2 : X.bluegrass = 0.6)
  (h3 : Y.ryegrass = 0.25)
  (h4 : (CombinedMixture X Y (1/3)).ryegrass = 0.3)
  : Y.fescue = 0.75 := by
  sorry

end fescue_percentage_in_Y_l1693_169309


namespace arithmetic_sequence_sum_l1693_169336

-- Define an arithmetic sequence
def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

-- State the theorem
theorem arithmetic_sequence_sum (a : ℕ → ℝ) :
  is_arithmetic_sequence a →
  (a 3 + a 4 + a 5 + a 6 + a 7 + a 8 + a 9 = 14) →
  a 6 = 2 := by
sorry

end arithmetic_sequence_sum_l1693_169336


namespace quadratic_roots_abs_less_than_one_l1693_169366

theorem quadratic_roots_abs_less_than_one (a b : ℝ) 
  (h1 : |a| + |b| < 1) 
  (h2 : a^2 - 4*b ≥ 0) : 
  ∀ x, x^2 + a*x + b = 0 → |x| < 1 :=
by sorry

end quadratic_roots_abs_less_than_one_l1693_169366


namespace similar_triangle_shortest_side_l1693_169365

theorem similar_triangle_shortest_side 
  (a b c : ℝ) -- sides of the first triangle
  (k : ℝ) -- scaling factor
  (h1 : a^2 + b^2 = c^2) -- Pythagorean theorem for the first triangle
  (h2 : a ≤ b) -- a is the shortest side of the first triangle
  (h3 : c = 39) -- hypotenuse of the first triangle
  (h4 : a = 15) -- shortest side of the first triangle
  (h5 : k * c = 117) -- hypotenuse of the second triangle
  : k * a = 45 := by sorry

end similar_triangle_shortest_side_l1693_169365


namespace ball_drawing_probabilities_l1693_169328

/-- The probability of drawing a red ball exactly on the fourth draw with replacement -/
def prob_red_fourth_with_replacement (total_balls : ℕ) (red_balls : ℕ) : ℚ :=
  (1 - red_balls / total_balls) ^ 3 * (red_balls / total_balls)

/-- The probability of drawing a red ball exactly on the fourth draw without replacement -/
def prob_red_fourth_without_replacement (total_balls : ℕ) (red_balls : ℕ) (white_balls : ℕ) : ℚ :=
  (white_balls / total_balls) * ((white_balls - 1) / (total_balls - 1)) *
  ((white_balls - 2) / (total_balls - 2)) * (red_balls / (total_balls - 3))

theorem ball_drawing_probabilities :
  let total_balls := 10
  let red_balls := 6
  let white_balls := 4
  prob_red_fourth_with_replacement total_balls red_balls = 24 / 625 ∧
  prob_red_fourth_without_replacement total_balls red_balls white_balls = 1 / 70 := by
  sorry

end ball_drawing_probabilities_l1693_169328


namespace area_of_polygon_AIHFGD_l1693_169317

-- Define the points
variable (A B C D E F G H I : ℝ × ℝ)

-- Define the squares
def is_square (P Q R S : ℝ × ℝ) : Prop := sorry

-- Define the area of a polygon
def area (polygon : List (ℝ × ℝ)) : ℝ := sorry

-- Define midpoint
def is_midpoint (M P Q : ℝ × ℝ) : Prop := sorry

theorem area_of_polygon_AIHFGD :
  is_square A B C D →
  is_square E F G D →
  area [A, B, C, D] = 25 →
  area [E, F, G, D] = 25 →
  is_midpoint H B C →
  is_midpoint H E F →
  is_midpoint I A B →
  area [A, I, H, F, G, D] = 25 := by
  sorry

end area_of_polygon_AIHFGD_l1693_169317


namespace sequence_properties_l1693_169315

-- Define the sequence a_n
def a : ℕ → ℤ
| n => if n ≤ 4 then n - 4 else 2^(n-5)

-- State the theorem
theorem sequence_properties :
  -- Conditions
  (a 2 = -2) ∧
  (a 7 = 4) ∧
  (∀ n ≤ 6, a (n+1) - a n = a (n+2) - a (n+1)) ∧
  (∀ n ≥ 5, (a (n+1))^2 = a n * a (n+2)) ∧
  -- Conclusions
  (∀ n, a n = if n ≤ 4 then n - 4 else 2^(n-5)) ∧
  (∀ m : ℕ, m > 0 → (a m + a (m+1) + a (m+2) = a m * a (m+1) * a (m+2) ↔ m = 1 ∨ m = 3)) :=
by sorry

end sequence_properties_l1693_169315


namespace book_sale_loss_percentage_l1693_169392

/-- Proves that the loss percentage is 10% given the selling prices with loss and with 10% gain --/
theorem book_sale_loss_percentage 
  (sp_loss : ℝ) 
  (sp_gain : ℝ) 
  (h_sp_loss : sp_loss = 450)
  (h_sp_gain : sp_gain = 550)
  (h_gain_percentage : sp_gain = 1.1 * (sp_gain / 1.1)) : 
  (((sp_gain / 1.1) - sp_loss) / (sp_gain / 1.1)) * 100 = 10 := by
  sorry

end book_sale_loss_percentage_l1693_169392


namespace afternoon_morning_difference_l1693_169324

/-- The number of campers who went rowing in the morning -/
def morning_campers : ℕ := 52

/-- The number of campers who went rowing in the afternoon -/
def afternoon_campers : ℕ := 61

/-- The theorem states that the difference between the number of campers
    who went rowing in the afternoon and the number of campers who went
    rowing in the morning is 9 -/
theorem afternoon_morning_difference :
  afternoon_campers - morning_campers = 9 := by
  sorry

end afternoon_morning_difference_l1693_169324


namespace raisins_sum_l1693_169320

-- Define the amounts of yellow and black raisins
def yellow_raisins : ℝ := 0.3
def black_raisins : ℝ := 0.4

-- Define the total amount of raisins
def total_raisins : ℝ := yellow_raisins + black_raisins

-- Theorem statement
theorem raisins_sum : total_raisins = 0.7 := by
  sorry

end raisins_sum_l1693_169320


namespace fraction_to_zero_power_l1693_169395

theorem fraction_to_zero_power (a b : ℤ) (hb : b ≠ 0) : (a / b : ℚ) ^ 0 = 1 := by
  sorry

end fraction_to_zero_power_l1693_169395


namespace lanas_bouquets_l1693_169399

theorem lanas_bouquets (tulips roses extra : ℕ) : 
  tulips = 36 → roses = 37 → extra = 3 → 
  tulips + roses + extra = 76 := by
sorry

end lanas_bouquets_l1693_169399


namespace birds_left_after_week_l1693_169387

/-- Calculates the number of birds left in a poultry farm after a week of disease -/
def birdsLeftAfterWeek (initialChickens initialTurkeys initialGuineaFowls : ℕ)
                       (dailyLossChickens dailyLossTurkeys dailyLossGuineaFowls : ℕ) : ℕ :=
  let daysInWeek : ℕ := 7
  let chickensLeft := initialChickens - daysInWeek * dailyLossChickens
  let turkeysLeft := initialTurkeys - daysInWeek * dailyLossTurkeys
  let guineaFowlsLeft := initialGuineaFowls - daysInWeek * dailyLossGuineaFowls
  chickensLeft + turkeysLeft + guineaFowlsLeft

theorem birds_left_after_week :
  birdsLeftAfterWeek 300 200 80 20 8 5 = 349 := by
  sorry

end birds_left_after_week_l1693_169387


namespace art_probability_correct_l1693_169377

def art_arrangement_probability (total : ℕ) (escher : ℕ) (picasso : ℕ) : ℚ :=
  let other := total - escher - picasso
  let grouped_items := other + 2  -- other items + Escher block + Picasso block
  (grouped_items.factorial * escher.factorial * picasso.factorial : ℚ) / total.factorial

theorem art_probability_correct :
  art_arrangement_probability 12 4 3 = 1 / 660 := by
  sorry

end art_probability_correct_l1693_169377


namespace cube_root_inequality_l1693_169300

theorem cube_root_inequality (a b c d : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hd : 0 < d) :
  Real.rpow (a * b) (1/3) + Real.rpow (c * d) (1/3) ≤ Real.rpow ((a + b + c) * (b + c + d)) (1/3) := by
  sorry

end cube_root_inequality_l1693_169300


namespace small_circles_radius_l1693_169368

theorem small_circles_radius (R : ℝ) (r : ℝ) : 
  R = 10 → 3 * (2 * r) = 2 * R → r = 10 / 3 :=
by sorry

end small_circles_radius_l1693_169368


namespace transport_tax_calculation_l1693_169343

/-- Calculate the transport tax for a vehicle -/
def calculate_transport_tax (horsepower : ℕ) (tax_rate : ℕ) (months_owned : ℕ) : ℕ :=
  (horsepower * tax_rate * months_owned) / 12

theorem transport_tax_calculation (horsepower tax_rate months_owned : ℕ) 
  (h1 : horsepower = 150)
  (h2 : tax_rate = 20)
  (h3 : months_owned = 8) :
  calculate_transport_tax horsepower tax_rate months_owned = 2000 := by
  sorry

#eval calculate_transport_tax 150 20 8

end transport_tax_calculation_l1693_169343


namespace max_log_sum_min_reciprocal_sum_l1693_169303

-- Define the conditions
variable (x y : ℝ)
variable (hx : x > 0)
variable (hy : y > 0)
variable (h_eq : 2 * x + 5 * y = 20)

-- Theorem for the maximum value of log x + log y
theorem max_log_sum :
  ∃ (max : ℝ), ∀ (a b : ℝ), a > 0 → b > 0 → 2 * a + 5 * b = 20 → 
    Real.log a + Real.log b ≤ max ∧ 
    (∃ (c d : ℝ), c > 0 ∧ d > 0 ∧ 2 * c + 5 * d = 20 ∧ Real.log c + Real.log d = max) ∧
    max = 1 :=
sorry

-- Theorem for the minimum value of 1/x + 1/y
theorem min_reciprocal_sum :
  ∃ (min : ℝ), ∀ (a b : ℝ), a > 0 → b > 0 → 2 * a + 5 * b = 20 → 
    1 / a + 1 / b ≥ min ∧ 
    (∃ (c d : ℝ), c > 0 ∧ d > 0 ∧ 2 * c + 5 * d = 20 ∧ 1 / c + 1 / d = min) ∧
    min = (7 + 2 * Real.sqrt 10) / 20 :=
sorry

end max_log_sum_min_reciprocal_sum_l1693_169303


namespace imaginary_unit_expression_l1693_169394

theorem imaginary_unit_expression : Complex.I^7 - 2 / Complex.I = Complex.I := by sorry

end imaginary_unit_expression_l1693_169394


namespace perfect_square_conversion_l1693_169316

theorem perfect_square_conversion (a b : ℝ) : 9 * a^4 * b^2 - 42 * a^2 * b = (3 * a^2 * b - 7)^2 := by
  sorry

end perfect_square_conversion_l1693_169316


namespace popsicle_sticks_given_away_l1693_169310

/-- Given that Gino initially had 63.0 popsicle sticks and now has 13 left,
    prove that he gave away 50 popsicle sticks. -/
theorem popsicle_sticks_given_away 
  (initial_sticks : ℝ) 
  (remaining_sticks : ℕ) 
  (h1 : initial_sticks = 63.0)
  (h2 : remaining_sticks = 13) :
  initial_sticks - remaining_sticks = 50 := by
  sorry

end popsicle_sticks_given_away_l1693_169310


namespace art_supplies_theorem_l1693_169301

def art_supplies_problem (total_spent canvas_cost paint_cost_ratio easel_cost : ℚ) : Prop :=
  let paint_cost := canvas_cost * paint_cost_ratio
  let other_items_cost := canvas_cost + paint_cost + easel_cost
  let paintbrush_cost := total_spent - other_items_cost
  paintbrush_cost = 15

theorem art_supplies_theorem :
  art_supplies_problem 90 40 (1/2) 15 := by
  sorry

end art_supplies_theorem_l1693_169301


namespace mark_solutions_mark_coefficients_l1693_169396

/-- Lauren's equation solutions -/
def lauren_solutions : Set ℝ := {x | |x - 6| = 3}

/-- Mark's equation -/
def mark_equation (b c : ℝ) (x : ℝ) : Prop := x^2 + b*x + c = 0

/-- Mark's equation has Lauren's solutions plus x = -2 -/
theorem mark_solutions (b c : ℝ) : 
  (∀ x ∈ lauren_solutions, mark_equation b c x) ∧ 
  mark_equation b c (-2) :=
sorry

/-- The values of b and c in Mark's equation -/
theorem mark_coefficients : 
  ∃ b c : ℝ, (b = -12 ∧ c = 27) ∧ 
  (∀ x ∈ lauren_solutions, mark_equation b c x) ∧ 
  mark_equation b c (-2) :=
sorry

end mark_solutions_mark_coefficients_l1693_169396


namespace no_two_roots_exist_l1693_169348

-- Define the equation as a function of x, y, and a
def equation (x y a : ℝ) : Prop :=
  x^2 + y^2 + 2*x = |x - a| - 1

-- Theorem statement
theorem no_two_roots_exist :
  ¬ ∃ (a : ℝ), ∃ (x₁ y₁ x₂ y₂ : ℝ),
    x₁ ≠ x₂ ∧ 
    equation x₁ y₁ a ∧ 
    equation x₂ y₂ a ∧ 
    (∀ (x y : ℝ), equation x y a → (x = x₁ ∧ y = y₁) ∨ (x = x₂ ∧ y = y₂)) :=
sorry

end no_two_roots_exist_l1693_169348


namespace sqrt2_minus_1_power_form_l1693_169381

theorem sqrt2_minus_1_power_form (n : ℕ) :
  ∃ k : ℤ, (Real.sqrt 2 - 1) ^ n = Real.sqrt (k + 1) - Real.sqrt k := by
  sorry

end sqrt2_minus_1_power_form_l1693_169381


namespace ten_player_tournament_rounds_l1693_169369

/-- The number of rounds needed for a round-robin tennis tournament -/
def roundsNeeded (players : ℕ) (courts : ℕ) : ℕ :=
  (players * (players - 1) / 2 + courts - 1) / courts

/-- Theorem: A 10-player round-robin tournament on 5 courts needs 9 rounds -/
theorem ten_player_tournament_rounds :
  roundsNeeded 10 5 = 9 := by
  sorry

end ten_player_tournament_rounds_l1693_169369


namespace triangle_with_angle_ratio_1_2_3_is_right_triangle_l1693_169329

theorem triangle_with_angle_ratio_1_2_3_is_right_triangle (A B C : ℝ) :
  A > 0 ∧ B > 0 ∧ C > 0 →
  A + B + C = 180 →
  B = 2 * A →
  C = 3 * A →
  C = 90 :=
sorry

end triangle_with_angle_ratio_1_2_3_is_right_triangle_l1693_169329


namespace volume_of_circumscribed_polyhedron_l1693_169325

/-- A polyhedron circumscribed around a sphere. -/
structure CircumscribedPolyhedron where
  -- The volume of the polyhedron
  volume : ℝ
  -- The radius of the inscribed sphere
  sphereRadius : ℝ
  -- The total surface area of the polyhedron
  surfaceArea : ℝ

/-- 
The volume of a polyhedron circumscribed around a sphere is equal to 
one-third of the product of the sphere's radius and the polyhedron's total surface area.
-/
theorem volume_of_circumscribed_polyhedron (p : CircumscribedPolyhedron) : 
  p.volume = (1 / 3) * p.sphereRadius * p.surfaceArea := by
  sorry

end volume_of_circumscribed_polyhedron_l1693_169325


namespace line_segment_endpoint_l1693_169376

theorem line_segment_endpoint (x : ℝ) : 
  (∃ (y : ℝ), (x - 3)^2 + (y - (-1))^2 = 15^2 ∧ 
               y = 7 ∧ 
               (y - (-1)) / (x - 3) = 1) →
  (x - 3)^2 + 64 = 225 :=
by sorry

end line_segment_endpoint_l1693_169376


namespace largest_base5_is_124_l1693_169322

/-- Represents a three-digit base-5 number -/
structure Base5Number where
  hundreds : Fin 5
  tens : Fin 5
  ones : Fin 5

/-- Converts a Base5Number to its decimal (base 10) representation -/
def toDecimal (n : Base5Number) : ℕ :=
  n.hundreds * 25 + n.tens * 5 + n.ones

/-- The largest three-digit base-5 number -/
def largestBase5 : Base5Number :=
  { hundreds := 4, tens := 4, ones := 4 }

theorem largest_base5_is_124 : toDecimal largestBase5 = 124 := by
  sorry

end largest_base5_is_124_l1693_169322


namespace a_share_in_profit_l1693_169335

/-- Given the investments of A, B, and C, and the total profit, prove A's share in the profit --/
theorem a_share_in_profit 
  (a_investment b_investment c_investment total_profit : ℕ) 
  (h1 : a_investment = 2400)
  (h2 : b_investment = 7200)
  (h3 : c_investment = 9600)
  (h4 : total_profit = 9000) :
  a_investment * total_profit / (a_investment + b_investment + c_investment) = 1125 := by
  sorry

end a_share_in_profit_l1693_169335


namespace ceiling_times_self_equals_156_l1693_169327

theorem ceiling_times_self_equals_156 :
  ∃! (x : ℝ), ⌈x⌉ * x = 156 :=
by sorry

end ceiling_times_self_equals_156_l1693_169327


namespace orange_boxes_needed_l1693_169341

/-- Calculates the number of boxes needed for oranges given the initial conditions --/
theorem orange_boxes_needed (baskets : ℕ) (oranges_per_basket : ℕ) (oranges_eaten : ℕ) (oranges_per_box : ℕ)
  (h1 : baskets = 7)
  (h2 : oranges_per_basket = 31)
  (h3 : oranges_eaten = 3)
  (h4 : oranges_per_box = 17) :
  (baskets * oranges_per_basket - oranges_eaten + oranges_per_box - 1) / oranges_per_box = 13 := by
  sorry

end orange_boxes_needed_l1693_169341


namespace function_and_composition_l1693_169375

def f (x : ℝ) (b c : ℝ) : ℝ := x^2 - b*x + c

theorem function_and_composition 
  (h1 : f 1 b c = 0) 
  (h2 : f 2 b c = -3) :
  (∀ x, f x b c = x^2 - 6*x + 5) ∧ 
  (∀ x > -1, f (1 / Real.sqrt (x + 1)) b c = 1 / (x + 1) - 6 / Real.sqrt (x + 1) + 5) := by
  sorry

end function_and_composition_l1693_169375


namespace solve_linear_equation_l1693_169385

theorem solve_linear_equation (x y : ℝ) : 2 * x + y = 3 → y = 3 - 2 * x := by
  sorry

end solve_linear_equation_l1693_169385


namespace quadrant_crossing_linear_function_y_intercept_positive_l1693_169332

/-- A linear function passing through the first, second, and third quadrants -/
structure QuadrantCrossingLinearFunction where
  b : ℝ
  passes_first_quadrant : ∃ x y, x > 0 ∧ y > 0 ∧ y = x + b
  passes_second_quadrant : ∃ x y, x < 0 ∧ y > 0 ∧ y = x + b
  passes_third_quadrant : ∃ x y, x < 0 ∧ y < 0 ∧ y = x + b

/-- The y-intercept of a quadrant crossing linear function is positive -/
theorem quadrant_crossing_linear_function_y_intercept_positive
  (f : QuadrantCrossingLinearFunction) : f.b > 0 := by
  sorry

end quadrant_crossing_linear_function_y_intercept_positive_l1693_169332


namespace ratio_of_segments_l1693_169358

/-- Given four points A, B, C, and D on a line in that order, with AB = 4, BC = 3, and AD = 20,
    prove that the ratio of AC to BD is 7/16. -/
theorem ratio_of_segments (A B C D : ℝ) : 
  A < B ∧ B < C ∧ C < D → -- Points lie on a line in order
  B - A = 4 →             -- AB = 4
  C - B = 3 →             -- BC = 3
  D - A = 20 →            -- AD = 20
  (C - A) / (D - B) = 7 / 16 := by
sorry

end ratio_of_segments_l1693_169358


namespace product_325_7_4_7_l1693_169340

/-- Converts a base 7 number to base 10 --/
def to_base_10 (digits : List Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * 7^i) 0

/-- Converts a base 10 number to base 7 --/
def to_base_7 (n : Nat) : List Nat :=
  if n = 0 then [0] else
  let rec aux (m : Nat) (acc : List Nat) :=
    if m = 0 then acc else aux (m / 7) ((m % 7) :: acc)
  aux n []

/-- Theorem: The product of 325₇ and 4₇ is equal to 1656₇ in base 7 --/
theorem product_325_7_4_7 : 
  to_base_7 (to_base_10 [5, 2, 3] * to_base_10 [4]) = [6, 5, 6, 1] := by
  sorry

end product_325_7_4_7_l1693_169340


namespace z_in_second_quadrant_l1693_169388

/-- The custom operation ⊗ -/
def tensor (a b c d : ℂ) : ℂ := a * c - b * d

/-- The complex number z satisfying the given equation -/
noncomputable def z : ℂ := sorry

/-- The statement to prove -/
theorem z_in_second_quadrant :
  tensor z (1 - 2*I) (-1) (1 + I) = 0 →
  z.re < 0 ∧ z.im > 0 := by sorry

end z_in_second_quadrant_l1693_169388


namespace min_value_theorem_l1693_169339

theorem min_value_theorem (x y : ℝ) (hx : x > 0) (hy : y > 0) 
  (h : Real.log (2^x) + Real.log (8^y) = Real.log 2) : 
  (x + y) / (x * y) ≥ 2 * Real.sqrt 3 + 4 := by
  sorry

end min_value_theorem_l1693_169339


namespace quadratic_vertex_l1693_169326

/-- The quadratic function f(x) = (x-2)^2 - 1 -/
def f (x : ℝ) : ℝ := (x - 2)^2 - 1

/-- The vertex of the quadratic function f -/
def vertex : ℝ × ℝ := (2, -1)

/-- Theorem: The vertex of the quadratic function f(x) = (x-2)^2 - 1 is at the point (2, -1) -/
theorem quadratic_vertex : 
  (∀ x : ℝ, f x ≥ f (vertex.1)) ∧ vertex.2 = f (vertex.1) := by
  sorry

end quadratic_vertex_l1693_169326


namespace time_for_600_parts_l1693_169384

/-- Linear regression equation relating parts processed to time spent -/
def linear_regression (x : ℝ) : ℝ := 0.01 * x + 0.5

/-- Theorem stating that processing 600 parts takes 6.5 hours -/
theorem time_for_600_parts : linear_regression 600 = 6.5 := by
  sorry

end time_for_600_parts_l1693_169384


namespace inequality_solution_l1693_169346

theorem inequality_solution (x : ℝ) (h : x ≠ 4) :
  (x^2 - 16) / (x - 4) ≤ 0 ↔ x ∈ Set.Iic (-4) := by
  sorry

end inequality_solution_l1693_169346


namespace albert_needs_more_money_l1693_169319

-- Define the costs of items and Albert's current money
def paintbrush_cost : ℚ := 1.50
def paints_cost : ℚ := 4.35
def easel_cost : ℚ := 12.65
def canvas_cost : ℚ := 7.95
def palette_cost : ℚ := 3.75
def albert_current_money : ℚ := 10.60

-- Define the total cost of items
def total_cost : ℚ := paintbrush_cost + paints_cost + easel_cost + canvas_cost + palette_cost

-- Theorem: Albert needs $19.60 more
theorem albert_needs_more_money : total_cost - albert_current_money = 19.60 := by
  sorry

end albert_needs_more_money_l1693_169319


namespace sqrt_seven_less_than_three_l1693_169393

theorem sqrt_seven_less_than_three : Real.sqrt 7 < 3 := by
  sorry

end sqrt_seven_less_than_three_l1693_169393


namespace three_distinct_prime_factors_l1693_169338

theorem three_distinct_prime_factors (p q : ℕ) (hp : Prime p) (hq : Prime q) (h_order : q > p ∧ p > 2) :
  ∃ (a b c : ℕ), Prime a ∧ Prime b ∧ Prime c ∧ a ≠ b ∧ b ≠ c ∧ a ≠ c ∧ (a * b * c ∣ 2^(p*q) - 1) :=
by sorry

end three_distinct_prime_factors_l1693_169338


namespace smallest_k_for_64_power_gt_4_16_l1693_169360

theorem smallest_k_for_64_power_gt_4_16 :
  ∀ k : ℕ, (64 : ℝ) ^ k > (4 : ℝ) ^ 16 ↔ k ≥ 6 :=
by sorry

end smallest_k_for_64_power_gt_4_16_l1693_169360


namespace arithmetic_calculation_l1693_169370

theorem arithmetic_calculation : 2535 + 240 / 30 - 435 = 2108 := by
  sorry

end arithmetic_calculation_l1693_169370


namespace third_part_value_l1693_169349

theorem third_part_value (total : ℚ) (ratio1 ratio2 ratio3 : ℚ) 
  (h_total : total = 782)
  (h_ratio1 : ratio1 = 1/2)
  (h_ratio2 : ratio2 = 2/3)
  (h_ratio3 : ratio3 = 3/4) :
  (ratio3 / (ratio1 + ratio2 + ratio3)) * total = 306 :=
by sorry

end third_part_value_l1693_169349


namespace rectangular_prism_diagonals_l1693_169361

/-- A rectangular prism that is not a cube -/
structure RectangularPrism where
  /-- The number of faces of the rectangular prism -/
  faces : ℕ
  /-- Each face is a rectangle -/
  faces_are_rectangles : True
  /-- The number of diagonals in each rectangular face -/
  diagonals_per_face : ℕ
  /-- The number of space diagonals in the rectangular prism -/
  space_diagonals : ℕ
  /-- The rectangular prism has exactly 6 faces -/
  face_count : faces = 6
  /-- Each rectangular face has exactly 2 diagonals -/
  face_diagonal_count : diagonals_per_face = 2
  /-- The rectangular prism has exactly 4 space diagonals -/
  space_diagonal_count : space_diagonals = 4

/-- The total number of diagonals in a rectangular prism -/
def total_diagonals (rp : RectangularPrism) : ℕ :=
  rp.faces * rp.diagonals_per_face + rp.space_diagonals

/-- Theorem: A rectangular prism (not a cube) has 16 diagonals -/
theorem rectangular_prism_diagonals (rp : RectangularPrism) : total_diagonals rp = 16 := by
  sorry

end rectangular_prism_diagonals_l1693_169361


namespace parallelogram_area_is_36_l1693_169374

-- Define the vectors v and w
def v : ℝ × ℝ := (4, -6)
def w : ℝ × ℝ := (8, -3)

-- Define the area of the parallelogram
def parallelogramArea (a b : ℝ × ℝ) : ℝ :=
  |a.1 * b.2 - a.2 * b.1|

-- Theorem statement
theorem parallelogram_area_is_36 :
  parallelogramArea v w = 36 := by
  sorry

end parallelogram_area_is_36_l1693_169374


namespace coordinates_of_point_B_l1693_169362

def point := ℝ × ℝ

theorem coordinates_of_point_B 
  (A B : point) 
  (length_AB : Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) = 5)
  (parallel_to_x : A.2 = B.2)
  (coord_A : A = (-1, 3)) :
  B = (-6, 3) ∨ B = (4, 3) := by
sorry

end coordinates_of_point_B_l1693_169362


namespace fruit_sales_problem_l1693_169306

/-- Fruit sales problem -/
theorem fruit_sales_problem 
  (apple_price : ℚ)
  (orange_price : ℚ)
  (morning_oranges : ℕ)
  (afternoon_apples : ℕ)
  (afternoon_oranges : ℕ)
  (total_sales : ℚ)
  (h1 : apple_price = 3/2)
  (h2 : orange_price = 1)
  (h3 : morning_oranges = 30)
  (h4 : afternoon_apples = 50)
  (h5 : afternoon_oranges = 40)
  (h6 : total_sales = 205) :
  ∃ (morning_apples : ℕ), 
    apple_price * (morning_apples + afternoon_apples) + 
    orange_price * (morning_oranges + afternoon_oranges) = total_sales ∧
    morning_apples = 40 := by
  sorry

end fruit_sales_problem_l1693_169306


namespace f_strictly_increasing_l1693_169371

noncomputable def f (x : ℝ) : ℝ := Real.exp x - Real.log (x + 1)

theorem f_strictly_increasing :
  ∀ x₁ x₂ : ℝ, 0 < x₁ → x₁ < x₂ → f x₁ < f x₂ := by sorry

end f_strictly_increasing_l1693_169371


namespace average_speed_problem_l1693_169380

theorem average_speed_problem (initial_distance : ℝ) (initial_speed : ℝ) (second_speed : ℝ) (average_speed : ℝ) :
  initial_distance = 24 →
  initial_speed = 40 →
  second_speed = 60 →
  average_speed = 55 →
  ∃ additional_distance : ℝ,
    (initial_distance + additional_distance) / ((initial_distance / initial_speed) + (additional_distance / second_speed)) = average_speed ∧
    additional_distance = 108 := by
  sorry

end average_speed_problem_l1693_169380


namespace expected_difference_coffee_tea_l1693_169398

-- Define the die sides
def dieSides : Nat := 8

-- Define perfect squares and primes up to 8
def perfectSquares : List Nat := [1, 4]
def primes : List Nat := [2, 3, 5, 7]

-- Define probabilities
def probCoffee : ℚ := 1 / 4
def probTea : ℚ := 1 / 2

-- Define number of days in a non-leap year
def daysInYear : Nat := 365

-- State the theorem
theorem expected_difference_coffee_tea :
  (probCoffee * daysInYear : ℚ) - (probTea * daysInYear : ℚ) = -91.25 := by
  sorry

end expected_difference_coffee_tea_l1693_169398


namespace inverse_variation_cube_square_l1693_169344

/-- Given that a³ varies inversely with b², prove that a³ = 125/16 when b = 8,
    given that a = 5 when b = 2. -/
theorem inverse_variation_cube_square (a b : ℝ) (k : ℝ) : 
  (∀ x y : ℝ, x^3 * y^2 = k) →  -- a³ varies inversely with b²
  (5^3 * 2^2 = k) →             -- a = 5 when b = 2
  (a^3 * 8^2 = k) →             -- condition for b = 8
  a^3 = 125/16 := by
sorry

end inverse_variation_cube_square_l1693_169344


namespace missing_figure_proof_l1693_169321

theorem missing_figure_proof (x : ℝ) : (0.1 / 100) * x = 0.24 → x = 240 := by
  sorry

end missing_figure_proof_l1693_169321


namespace set_B_is_empty_l1693_169342

def set_B : Set ℝ := {x | x > 8 ∧ x < 5}

theorem set_B_is_empty : set_B = ∅ := by sorry

end set_B_is_empty_l1693_169342


namespace A_C_mutually_exclusive_l1693_169312

/-- Represents the sample space of three products -/
structure ThreeProducts where
  product1 : Bool  -- True if defective, False if not defective
  product2 : Bool
  product3 : Bool

/-- Event A: All three products are not defective -/
def A (s : ThreeProducts) : Prop :=
  ¬s.product1 ∧ ¬s.product2 ∧ ¬s.product3

/-- Event B: All three products are defective -/
def B (s : ThreeProducts) : Prop :=
  s.product1 ∧ s.product2 ∧ s.product3

/-- Event C: At least one of the three products is defective -/
def C (s : ThreeProducts) : Prop :=
  s.product1 ∨ s.product2 ∨ s.product3

/-- Theorem: A and C are mutually exclusive -/
theorem A_C_mutually_exclusive :
  ∀ s : ThreeProducts, ¬(A s ∧ C s) :=
by sorry

end A_C_mutually_exclusive_l1693_169312


namespace construction_valid_l1693_169378

-- Define the basic types and structures
structure Point where
  x : ℝ
  y : ℝ

structure Quadrilateral where
  A : Point
  B : Point
  C : Point
  D : Point

def isConvex (q : Quadrilateral) : Prop := sorry

def areNotConcyclic (q : Quadrilateral) : Prop := sorry

-- Define the construction steps
def rotateAroundPoint (p : Point) (center : Point) (angle : ℝ) : Point := sorry

def lineIntersection (p1 p2 q1 q2 : Point) : Point := sorry

def circumcircle (p1 p2 p3 : Point) : Set Point := sorry

-- Define the construction method
def constructCD (A B : Point) (angleBCD angleADC angleBCA angleACD : ℝ) : Quadrilateral := sorry

-- The main theorem
theorem construction_valid (A B : Point) (angleBCD angleADC angleBCA angleACD : ℝ) :
  let q := constructCD A B angleBCD angleADC angleBCA angleACD
  isConvex q ∧ areNotConcyclic q →
  ∃ (C D : Point), q = Quadrilateral.mk A B C D :=
sorry

end construction_valid_l1693_169378


namespace beatrix_height_relative_to_georgia_l1693_169311

theorem beatrix_height_relative_to_georgia (B V G : ℝ) 
  (h1 : B = 2 * V) 
  (h2 : V = 2/3 * G) : 
  B = 4/3 * G := by
sorry

end beatrix_height_relative_to_georgia_l1693_169311


namespace smallest_k_for_inequality_l1693_169304

theorem smallest_k_for_inequality : ∃ k : ℕ, k = 4 ∧ 
  (∀ n : ℕ, n > 0 → ∀ a : ℝ, 0 ≤ a ∧ a ≤ 1 → a^k * (1-a)^n < 1 / (n+1)^3) ∧
  (∀ k' : ℕ, k' < k → ∃ n : ℕ, n > 0 ∧ ∃ a : ℝ, 0 ≤ a ∧ a ≤ 1 ∧ a^k' * (1-a)^n ≥ 1 / (n+1)^3) :=
by sorry

end smallest_k_for_inequality_l1693_169304


namespace one_third_of_cake_flour_one_third_of_cake_flour_mixed_number_l1693_169383

theorem one_third_of_cake_flour :
  let full_recipe : ℚ := 19 / 3
  let one_third_recipe : ℚ := full_recipe / 3
  one_third_recipe = 19 / 9 :=
by sorry

-- Convert to mixed number
theorem one_third_of_cake_flour_mixed_number :
  let full_recipe : ℚ := 19 / 3
  let one_third_recipe : ℚ := full_recipe / 3
  ∃ (whole : ℕ) (numerator : ℕ) (denominator : ℕ),
    one_third_recipe = whole + (numerator : ℚ) / denominator ∧
    whole = 2 ∧ numerator = 1 ∧ denominator = 9 :=
by sorry

end one_third_of_cake_flour_one_third_of_cake_flour_mixed_number_l1693_169383


namespace f_at_2_eq_neg_22_l1693_169373

/-- Given a function f(x) = x^5 - ax^3 + bx - 6 where f(-2) = 10, prove that f(2) = -22 -/
theorem f_at_2_eq_neg_22 (a b : ℝ) (f : ℝ → ℝ) 
    (h1 : ∀ x, f x = x^5 - a*x^3 + b*x - 6)
    (h2 : f (-2) = 10) : 
  f 2 = -22 := by sorry

end f_at_2_eq_neg_22_l1693_169373


namespace bike_speed_l1693_169382

/-- Given a bike moving at a constant speed that covers 5400 meters in 9 minutes,
    prove that its speed is 10 meters per second. -/
theorem bike_speed (distance : ℝ) (time_minutes : ℝ) (speed : ℝ) 
    (h1 : distance = 5400)
    (h2 : time_minutes = 9)
    (h3 : speed = distance / (time_minutes * 60)) : 
    speed = 10 := by
  sorry

end bike_speed_l1693_169382


namespace complex_real_condition_l1693_169364

theorem complex_real_condition (a : ℝ) : 
  (((a : ℂ) + Complex.I) / (3 + 4 * Complex.I)).im = 0 → a = 3/4 := by
  sorry

end complex_real_condition_l1693_169364


namespace g_properties_l1693_169345

-- Define g as a function from real numbers to real numbers
variable (g : ℝ → ℝ)

-- Define the properties of g
axiom g_positive : ∀ x, g x > 0
axiom g_sum_property : ∀ a b, g a + g b = g (a + b + 1)

-- State the theorem
theorem g_properties :
  (∃ k : ℝ, k > 0 ∧ g 0 = k) ∧
  (∃ a : ℝ, g (-a) ≠ 1 - g a) :=
sorry

end g_properties_l1693_169345


namespace gcf_of_60_and_75_l1693_169333

theorem gcf_of_60_and_75 : Nat.gcd 60 75 = 15 := by
  sorry

end gcf_of_60_and_75_l1693_169333


namespace abs_negative_seventeen_l1693_169308

theorem abs_negative_seventeen : |(-17 : ℤ)| = 17 := by
  sorry

end abs_negative_seventeen_l1693_169308


namespace arithmetic_progression_theorem_l1693_169390

/-- An arithmetic progression with n terms -/
structure ArithmeticProgression where
  n : ℕ
  a : ℕ → ℕ
  d : ℕ
  progression : ∀ i, i < n → a (i + 1) = a i + d

/-- The sum of an arithmetic progression -/
def sum (ap : ArithmeticProgression) : ℕ :=
  (ap.n * (2 * ap.a 0 + (ap.n - 1) * ap.d)) / 2

theorem arithmetic_progression_theorem (ap : ArithmeticProgression) :
  sum ap = 112 ∧
  ap.a 1 * ap.d = 30 ∧
  ap.a 2 + ap.a 4 = 32 →
  ap.n = 7 ∧
  ((ap.a 0 = 7 ∧ ap.a 1 = 10 ∧ ap.a 2 = 13) ∨
   (ap.a 0 = 1 ∧ ap.a 1 = 6 ∧ ap.a 2 = 11)) :=
by sorry


end arithmetic_progression_theorem_l1693_169390


namespace complex_number_quadrant_l1693_169350

theorem complex_number_quadrant : 
  let z : ℂ := (Complex.I : ℂ) / (1 + Complex.I)
  (z.re > 0) ∧ (z.im > 0) :=
by sorry

end complex_number_quadrant_l1693_169350


namespace number_of_lists_18_4_l1693_169352

/-- The number of elements in the set of balls -/
def n : ℕ := 18

/-- The number of draws -/
def k : ℕ := 4

/-- The number of possible lists when drawing k times with replacement from a set of n elements -/
def number_of_lists (n k : ℕ) : ℕ := n^k

/-- Theorem: The number of possible lists when drawing 4 times with replacement from a set of 18 elements is 104,976 -/
theorem number_of_lists_18_4 : number_of_lists n k = 104976 := by
  sorry

end number_of_lists_18_4_l1693_169352


namespace max_colors_theorem_l1693_169363

/-- Represents a color configuration of an n × n × n cube -/
structure ColorConfig (n : ℕ) where
  colors : Fin n → Fin n → Fin n → ℕ

/-- Represents a set of colors in an n × n × 1 box -/
def ColorSet (n : ℕ) := Set ℕ

/-- Returns the set of colors in an n × n × 1 box for a given configuration and orientation -/
def getColorSet (n : ℕ) (config : ColorConfig n) (orientation : Fin 3) (i : Fin n) : ColorSet n :=
  sorry

/-- Checks if the color configuration satisfies the problem conditions -/
def validConfig (n : ℕ) (config : ColorConfig n) : Prop :=
  ∀ (o1 o2 o3 : Fin 3) (i j : Fin n),
    o1 ≠ o2 ∧ o2 ≠ o3 ∧ o1 ≠ o3 →
    ∃ (k l : Fin n), 
      getColorSet n config o1 i = getColorSet n config o2 k ∧
      getColorSet n config o1 i = getColorSet n config o3 l

/-- The maximal number of colors in a valid configuration -/
def maxColors (n : ℕ) : ℕ :=
  n * (n + 1) * (2 * n + 1) / 6

theorem max_colors_theorem (n : ℕ) (h : n > 1) :
  ∃ (config : ColorConfig n),
    validConfig n config ∧
    (∀ (config' : ColorConfig n), validConfig n config' →
      Finset.card (Finset.image (config.colors) Finset.univ) ≥
      Finset.card (Finset.image (config'.colors) Finset.univ)) ∧
    Finset.card (Finset.image (config.colors) Finset.univ) = maxColors n :=
  sorry

end max_colors_theorem_l1693_169363


namespace percent_of_number_l1693_169323

theorem percent_of_number (percent : ℝ) (number : ℝ) (result : ℝ) :
  percent = 37.5 ∧ number = 725 ∧ result = 271.875 →
  (percent / 100) * number = result :=
by
  sorry

end percent_of_number_l1693_169323


namespace inequality_proof_l1693_169386

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  (a / Real.sqrt (a^2 + 8*b*c)) + (b / Real.sqrt (b^2 + 8*c*a)) + (c / Real.sqrt (c^2 + 8*a*b)) ≥ 1 := by
  sorry

end inequality_proof_l1693_169386


namespace first_worker_load_time_l1693_169307

/-- The time it takes for two workers to load a truck together -/
def combined_time : ℝ := 3.0769230769230766

/-- The time it takes for the second worker to load a truck alone -/
def second_worker_time : ℝ := 8

/-- The time it takes for the first worker to load a truck alone -/
def first_worker_time : ℝ := 5

/-- Theorem stating that given the combined time and the second worker's time, 
    the first worker's time to load the truck alone is 5 hours -/
theorem first_worker_load_time : 
  1 / first_worker_time + 1 / second_worker_time = 1 / combined_time :=
sorry

end first_worker_load_time_l1693_169307


namespace car_journey_speed_l1693_169347

/-- Proves that given specific conditions about a car's journey, 
    the speed for the remaining part of the trip is 60 mph. -/
theorem car_journey_speed (D : ℝ) (h1 : D > 0) : 
  let first_part_distance := 0.4 * D
  let first_part_speed := 40
  let total_average_speed := 50
  let remaining_part_distance := 0.6 * D
  let remaining_part_speed := 
    remaining_part_distance / 
    (D / total_average_speed - first_part_distance / first_part_speed)
  remaining_part_speed = 60 := by
  sorry


end car_journey_speed_l1693_169347


namespace quadratic_two_roots_l1693_169354

/-- The quadratic function f(x) = x^2 - 2x - 3 has exactly two real roots -/
theorem quadratic_two_roots : ∃ (r₁ r₂ : ℝ), r₁ ≠ r₂ ∧ 
  (∀ x, x^2 - 2*x - 3 = 0 ↔ x = r₁ ∨ x = r₂) := by
  sorry

end quadratic_two_roots_l1693_169354


namespace f_even_k_value_g_f_common_point_a_range_l1693_169397

/-- A function is even if f(x) = f(-x) for all x -/
def IsEven (f : ℝ → ℝ) : Prop := ∀ x, f x = f (-x)

/-- The logarithm base 4 -/
noncomputable def log4 (x : ℝ) : ℝ := (Real.log x) / (Real.log 4)

/-- The function f(x) -/
noncomputable def f (k : ℝ) (x : ℝ) : ℝ := log4 (4^x + 1) + k * x

/-- The function g(x) -/
noncomputable def g (a : ℝ) (x : ℝ) : ℝ := log4 (a * 2^x - 4/3 * a)

/-- The number of common points between f and g -/
def CommonPoints (f g : ℝ → ℝ) : Prop := ∃! x, f x = g x

theorem f_even_k_value :
  IsEven (f k) → k = -1/2 :=
sorry

theorem g_f_common_point_a_range :
  CommonPoints (f (-1/2)) (g a) → (a > 1 ∨ a = -3) :=
sorry

end f_even_k_value_g_f_common_point_a_range_l1693_169397


namespace race_distance_l1693_169334

/-- A race between two runners p and q, where p is faster but q gets a head start -/
structure Race where
  /-- The speed of runner q (in meters per second) -/
  q_speed : ℝ
  /-- The speed of runner p (in meters per second) -/
  p_speed : ℝ
  /-- The head start given to runner q (in meters) -/
  head_start : ℝ
  /-- The condition that p is 25% faster than q -/
  speed_ratio : p_speed = 1.25 * q_speed
  /-- The head start is 60 meters -/
  head_start_value : head_start = 60

/-- The theorem stating that if the race ends in a tie, p ran 300 meters -/
theorem race_distance (race : Race) : 
  (∃ t : ℝ, race.q_speed * t = race.p_speed * t - race.head_start) → 
  race.p_speed * (300 / race.p_speed) = 300 := by
  sorry

#check race_distance

end race_distance_l1693_169334


namespace couscous_first_shipment_l1693_169353

theorem couscous_first_shipment (total_shipments : ℕ) 
  (shipment_a shipment_b first_shipment : ℝ) 
  (num_dishes : ℕ) (couscous_per_dish : ℝ) : 
  total_shipments = 3 →
  shipment_a = 13 →
  shipment_b = 45 →
  num_dishes = 13 →
  couscous_per_dish = 5 →
  first_shipment ≠ shipment_b →
  first_shipment = num_dishes * couscous_per_dish :=
by sorry

end couscous_first_shipment_l1693_169353


namespace P_intersect_Q_equals_closed_interval_l1693_169318

-- Define the sets P and Q
def P : Set ℝ := {x | x^2 - 2*x ≤ 0}
def Q : Set ℝ := {y | ∃ x, y = x^2 - 2*x}

-- State the theorem
theorem P_intersect_Q_equals_closed_interval :
  P ∩ Q = Set.Icc 0 2 := by sorry

end P_intersect_Q_equals_closed_interval_l1693_169318


namespace coin_game_probability_l1693_169305

def coin_game (n : ℕ) : ℝ :=
  sorry

theorem coin_game_probability : coin_game 5 = 1521 / 2^15 := by
  sorry

end coin_game_probability_l1693_169305


namespace trigonometric_equation_solution_l1693_169330

theorem trigonometric_equation_solution :
  ∀ x : ℝ, ((7/2 * Real.cos (2*x) + 2) * abs (2 * Real.cos (2*x) - 1) = 
            Real.cos x * (Real.cos x + Real.cos (5*x))) ↔
           (∃ k : ℤ, x = π/6 + k*π/2 ∨ x = -π/6 + k*π/2) :=
by sorry

end trigonometric_equation_solution_l1693_169330


namespace solution_set_of_trig_equation_l1693_169391

theorem solution_set_of_trig_equation :
  let S : Set ℝ := {x | 5 * Real.sin x = 4 + 2 * Real.cos (2 * x)}
  S = {x | ∃ k : ℤ, x = Real.arcsin (3/4) + 2 * k * Real.pi ∨ 
                    x = Real.pi - Real.arcsin (3/4) + 2 * k * Real.pi} :=
by sorry

end solution_set_of_trig_equation_l1693_169391


namespace mixture_replacement_solution_l1693_169302

/-- Represents the mixture replacement problem -/
def MixtureReplacement (initial_A : ℝ) (initial_ratio_A : ℝ) (initial_ratio_B : ℝ) 
                       (final_ratio_A : ℝ) (final_ratio_B : ℝ) : Prop :=
  let initial_B := initial_A * initial_ratio_B / initial_ratio_A
  let replaced_amount := 
    (final_ratio_B * initial_A - final_ratio_A * initial_B) / 
    (final_ratio_A + final_ratio_B)
  replaced_amount = 40

/-- Theorem stating the solution to the mixture replacement problem -/
theorem mixture_replacement_solution :
  MixtureReplacement 32 4 1 2 3 := by
  sorry

end mixture_replacement_solution_l1693_169302


namespace faye_pencils_and_crayons_l1693_169314

/-- Given that Faye arranges her pencils and crayons in 11 rows,
    with 31 pencils and 27 crayons in each row,
    prove that she has 638 pencils and crayons in total. -/
theorem faye_pencils_and_crayons (rows : ℕ) (pencils_per_row : ℕ) (crayons_per_row : ℕ)
    (h1 : rows = 11)
    (h2 : pencils_per_row = 31)
    (h3 : crayons_per_row = 27) :
    rows * pencils_per_row + rows * crayons_per_row = 638 := by
  sorry

end faye_pencils_and_crayons_l1693_169314


namespace ab_max_and_4a2_b2_min_l1693_169372

theorem ab_max_and_4a2_b2_min (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : 2 * a + b = 1) :
  (∀ x y : ℝ, x > 0 → y > 0 → 2 * x + y = 1 → a * b ≥ x * y) ∧
  (∀ x y : ℝ, x > 0 → y > 0 → 2 * x + y = 1 → 4 * a^2 + b^2 ≤ 4 * x^2 + y^2) ∧
  a * b = 1/8 ∧
  4 * a^2 + b^2 = 1/2 :=
by sorry

end ab_max_and_4a2_b2_min_l1693_169372


namespace hawks_score_l1693_169356

theorem hawks_score (total_points margin eagles_three_pointers : ℕ) 
  (h1 : total_points = 82)
  (h2 : margin = 18)
  (h3 : eagles_three_pointers = 12) : 
  total_points - (total_points + margin) / 2 = 32 :=
sorry

end hawks_score_l1693_169356


namespace isosceles_trapezoid_AE_squared_l1693_169331

/-- An isosceles trapezoid with specific properties -/
structure IsoscelesTrapezoid where
  /-- Point A -/
  A : ℝ × ℝ
  /-- Point B -/
  B : ℝ × ℝ
  /-- Point C -/
  C : ℝ × ℝ
  /-- Point D -/
  D : ℝ × ℝ
  /-- Point E on AC -/
  E : ℝ × ℝ
  /-- AB is parallel to CD -/
  parallel_AB_CD : (B.1 - A.1) * (D.2 - C.2) = (B.2 - A.2) * (D.1 - C.1)
  /-- Length of AB is 6 -/
  AB_length : Real.sqrt ((B.1 - A.1)^2 + (B.2 - A.2)^2) = 6
  /-- Length of CD is 14 -/
  CD_length : Real.sqrt ((D.1 - C.1)^2 + (D.2 - C.2)^2) = 14
  /-- ∠AEC is a right angle -/
  AEC_right_angle : (E.1 - A.1) * (E.1 - C.1) + (E.2 - A.2) * (E.2 - C.2) = 0
  /-- CE = CB -/
  CE_eq_CB : (E.1 - C.1)^2 + (E.2 - C.2)^2 = (B.1 - C.1)^2 + (B.2 - C.2)^2

/-- The theorem to be proved -/
theorem isosceles_trapezoid_AE_squared (t : IsoscelesTrapezoid) :
  (t.E.1 - t.A.1)^2 + (t.E.2 - t.A.2)^2 = 84 := by
  sorry

end isosceles_trapezoid_AE_squared_l1693_169331


namespace cheesecake_calories_per_slice_quarter_of_slices_is_two_l1693_169355

/-- Represents a cheesecake with its total calories and number of slices -/
structure Cheesecake where
  totalCalories : ℕ
  numSlices : ℕ

/-- Calculates the number of calories per slice in a cheesecake -/
def caloriesPerSlice (cake : Cheesecake) : ℕ :=
  cake.totalCalories / cake.numSlices

theorem cheesecake_calories_per_slice :
  ∀ (cake : Cheesecake),
    cake.totalCalories = 2800 →
    cake.numSlices = 8 →
    caloriesPerSlice cake = 350 := by
  sorry

/-- Verifies that 25% of the total slices is equal to 2 slices -/
theorem quarter_of_slices_is_two (cake : Cheesecake) :
  cake.numSlices = 8 →
  cake.numSlices / 4 = 2 := by
  sorry

end cheesecake_calories_per_slice_quarter_of_slices_is_two_l1693_169355


namespace negative_rational_power_equality_l1693_169337

theorem negative_rational_power_equality : 
  Real.rpow (-3 * (3/8)) (-(2/3)) = 4/9 := by sorry

end negative_rational_power_equality_l1693_169337


namespace cubic_function_theorem_l1693_169367

/-- Given a function f and its derivative f', g is defined as their sum -/
def g (f : ℝ → ℝ) (f' : ℝ → ℝ) : ℝ → ℝ := λ x => f x + f' x

/-- f is a cubic function with parameters a and b -/
def f (a b : ℝ) : ℝ → ℝ := λ x => a * x^3 + x^2 + b * x

/-- f' is the derivative of f -/
def f' (a b : ℝ) : ℝ → ℝ := λ x => 3 * a * x^2 + 2 * x + b

theorem cubic_function_theorem (a b : ℝ) :
  (∀ x, g (f a b) (f' a b) (-x) = -(g (f a b) (f' a b) x)) →
  (f a b = λ x => -1/3 * x^3 + x^2) ∧
  (∃ x ∈ Set.Icc 1 2, ∀ y ∈ Set.Icc 1 2, g (f a b) (f' a b) y ≤ g (f a b) (f' a b) x) ∧
  (g (f a b) (f' a b) x = 5/3) ∧
  (∃ x ∈ Set.Icc 1 2, ∀ y ∈ Set.Icc 1 2, g (f a b) (f' a b) x ≤ g (f a b) (f' a b) y) ∧
  (g (f a b) (f' a b) x = 4/3) :=
by sorry

end cubic_function_theorem_l1693_169367


namespace selection_problem_l1693_169359

theorem selection_problem (n_sergeants m_soldiers : ℕ) 
  (k_sergeants k_soldiers : ℕ) (factor : ℕ) :
  n_sergeants = 6 →
  m_soldiers = 60 →
  k_sergeants = 2 →
  k_soldiers = 20 →
  factor = 3 →
  (factor * Nat.choose n_sergeants k_sergeants * Nat.choose m_soldiers k_soldiers) = 
  (3 * Nat.choose 6 2 * Nat.choose 60 20) :=
by sorry

end selection_problem_l1693_169359


namespace problem_solution_l1693_169357

theorem problem_solution (x : ℚ) : 
  (1 / 3 : ℚ) - (1 / 4 : ℚ) + (1 / 6 : ℚ) = 4 / x → x = 16 := by
  sorry

end problem_solution_l1693_169357


namespace speed_ratio_l1693_169379

/-- The speed of object A -/
def v_A : ℝ := sorry

/-- The speed of object B -/
def v_B : ℝ := sorry

/-- The initial distance of B from O -/
def initial_distance : ℝ := 800

/-- The time when A and B are first equidistant from O -/
def t1 : ℝ := 3

/-- The additional time until A and B are again equidistant from O -/
def t2 : ℝ := 5

theorem speed_ratio : 
  (∀ t : ℝ, t1 * v_A = |initial_distance - t1 * v_B|) ∧
  (∀ t : ℝ, (t1 + t2) * v_A = |initial_distance - (t1 + t2) * v_B|) →
  v_A / v_B = 1 / 2 := by sorry

end speed_ratio_l1693_169379
