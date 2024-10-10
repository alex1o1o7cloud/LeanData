import Mathlib

namespace quadratic_equation_solution_difference_l3704_370440

theorem quadratic_equation_solution_difference : 
  ∀ x₁ x₂ : ℝ, 
  (x₁^2 - 5*x₁ + 11 = x₁ + 27) → 
  (x₂^2 - 5*x₂ + 11 = x₂ + 27) → 
  x₁ ≠ x₂ →
  |x₁ - x₂| = 10 := by
sorry

end quadratic_equation_solution_difference_l3704_370440


namespace pool_water_volume_l3704_370464

/-- The volume of water in a cylindrical pool with a cylindrical column inside -/
theorem pool_water_volume 
  (pool_diameter : ℝ) 
  (pool_depth : ℝ) 
  (column_diameter : ℝ) 
  (column_depth : ℝ) 
  (h_pool_diameter : pool_diameter = 20)
  (h_pool_depth : pool_depth = 6)
  (h_column_diameter : column_diameter = 4)
  (h_column_depth : column_depth = pool_depth) :
  let pool_radius : ℝ := pool_diameter / 2
  let column_radius : ℝ := column_diameter / 2
  let pool_volume : ℝ := π * pool_radius^2 * pool_depth
  let column_volume : ℝ := π * column_radius^2 * column_depth
  pool_volume - column_volume = 576 * π := by
sorry


end pool_water_volume_l3704_370464


namespace central_cell_only_solution_l3704_370411

/-- Represents a 5x5 grid with boolean values (true for "+", false for "-") -/
def Grid := Fin 5 → Fin 5 → Bool

/-- Represents a subgrid position and size -/
structure Subgrid where
  row : Fin 5
  col : Fin 5
  size : Nat
  size_valid : 2 ≤ size ∧ size ≤ 5

/-- Flips the signs in a subgrid -/
def flip_subgrid (g : Grid) (sg : Subgrid) : Grid :=
  λ i j => if i < sg.row + sg.size ∧ j < sg.col + sg.size
           then !g i j
           else g i j

/-- Checks if all cells in the grid are positive -/
def all_positive (g : Grid) : Prop :=
  ∀ i j, g i j = true

/-- Initial grid with only the specified cell negative -/
def initial_grid (row col : Fin 5) : Grid :=
  λ i j => ¬(i = row ∧ j = col)

/-- Theorem stating that only the central cell as initial negative allows for all positive end state -/
theorem central_cell_only_solution :
  ∀ (row col : Fin 5),
    (∃ (moves : List Subgrid), all_positive (moves.foldl flip_subgrid (initial_grid row col))) ↔
    (row = 2 ∧ col = 2) :=
  sorry

end central_cell_only_solution_l3704_370411


namespace walmart_gift_card_value_l3704_370451

/-- Given information about gift cards and their usage, determine the value of each Walmart gift card -/
theorem walmart_gift_card_value 
  (best_buy_count : ℕ) 
  (best_buy_value : ℕ) 
  (walmart_count : ℕ) 
  (used_best_buy : ℕ) 
  (used_walmart : ℕ) 
  (total_remaining_value : ℕ) :
  best_buy_count = 6 →
  best_buy_value = 500 →
  walmart_count = 9 →
  used_best_buy = 1 →
  used_walmart = 2 →
  total_remaining_value = 3900 →
  (walmart_count - used_walmart) * 
    ((total_remaining_value - (best_buy_count - used_best_buy) * best_buy_value) / 
     (walmart_count - used_walmart)) = 
  (walmart_count - used_walmart) * 200 :=
by sorry

end walmart_gift_card_value_l3704_370451


namespace will_remaining_candy_l3704_370423

/-- Represents the number of pieces in each type of candy box -/
structure CandyBox where
  chocolate : Nat
  mint : Nat
  caramel : Nat

/-- Represents the number of boxes of each type of candy -/
structure CandyInventory where
  chocolate : Nat
  mint : Nat
  caramel : Nat

def initial_inventory : CandyInventory := 
  { chocolate := 7, mint := 5, caramel := 4 }

def pieces_per_box : CandyBox := 
  { chocolate := 12, mint := 15, caramel := 10 }

def boxes_given_away : CandyInventory := 
  { chocolate := 3, mint := 2, caramel := 1 }

/-- Calculates the total number of candy pieces for a given inventory -/
def total_pieces (inventory : CandyInventory) (box : CandyBox) : Nat :=
  inventory.chocolate * box.chocolate + 
  inventory.mint * box.mint + 
  inventory.caramel * box.caramel

/-- Calculates the remaining inventory after giving away boxes -/
def remaining_inventory (initial : CandyInventory) (given_away : CandyInventory) : CandyInventory :=
  { chocolate := initial.chocolate - given_away.chocolate,
    mint := initial.mint - given_away.mint,
    caramel := initial.caramel - given_away.caramel }

theorem will_remaining_candy : 
  total_pieces (remaining_inventory initial_inventory boxes_given_away) pieces_per_box = 123 := by
  sorry

end will_remaining_candy_l3704_370423


namespace decimal_places_of_fraction_l3704_370497

theorem decimal_places_of_fraction : ∃ (n : ℕ), 
  (5^7 : ℚ) / (10^5 * 125) = (n : ℚ) / 10^5 ∧ 
  0 < n ∧ 
  n < 10^5 :=
by sorry

end decimal_places_of_fraction_l3704_370497


namespace arithmetic_calculation_l3704_370436

theorem arithmetic_calculation : 10 - 9 + 8 * 7 + 6 - 5 * 4 + 3 - 2 = 44 := by
  sorry

end arithmetic_calculation_l3704_370436


namespace no_four_digit_n_over_5_and_5n_l3704_370447

theorem no_four_digit_n_over_5_and_5n : 
  ¬ ∃ (n : ℕ), n > 0 ∧ 
    (1000 ≤ n / 5 ∧ n / 5 ≤ 9999) ∧ 
    (1000 ≤ 5 * n ∧ 5 * n ≤ 9999) :=
by sorry

end no_four_digit_n_over_5_and_5n_l3704_370447


namespace test_question_points_l3704_370489

theorem test_question_points (total_points total_questions two_point_questions : ℕ) 
  (h1 : total_points = 100)
  (h2 : total_questions = 40)
  (h3 : two_point_questions = 30) :
  (total_points - 2 * two_point_questions) / (total_questions - two_point_questions) = 4 :=
by
  sorry

end test_question_points_l3704_370489


namespace polyhedron_volume_l3704_370466

theorem polyhedron_volume (prism_volume : ℝ) (pyramid_base_side : ℝ) (pyramid_height : ℝ) :
  prism_volume = Real.sqrt 2 - 1 →
  pyramid_base_side = 1 →
  pyramid_height = 1 / 2 →
  prism_volume + 2 * (1 / 3 * pyramid_base_side^2 * pyramid_height) = Real.sqrt 2 - 2 / 3 := by
  sorry

end polyhedron_volume_l3704_370466


namespace third_derivative_y_l3704_370426

noncomputable def y (x : ℝ) : ℝ := (1 + x^2) * Real.arctan x

theorem third_derivative_y (x : ℝ) :
  HasDerivAt (fun x => (deriv (deriv y)) x) (4 / (1 + x^2)^2) x :=
sorry

end third_derivative_y_l3704_370426


namespace solution_range_l3704_370413

def P (a : ℝ) : Set ℝ := {x : ℝ | (x + 1) / (x + a) < 2}

theorem solution_range (a : ℝ) : (1 ∉ P a) ↔ a ∈ Set.Icc (-1 : ℝ) 0 := by
  sorry

end solution_range_l3704_370413


namespace no_solution_for_equation_l3704_370485

theorem no_solution_for_equation (x y : ℝ) : xy = 1 → ¬(Real.sqrt (x^2 + y^2) = x + y) := by
  sorry

end no_solution_for_equation_l3704_370485


namespace sakshi_investment_dividend_l3704_370460

/-- Calculate the total dividend per annum for Sakshi's investment --/
theorem sakshi_investment_dividend
  (total_investment : ℝ)
  (investment_12_percent : ℝ)
  (price_12_percent : ℝ)
  (price_15_percent : ℝ)
  (dividend_rate_12_percent : ℝ)
  (dividend_rate_15_percent : ℝ)
  (h1 : total_investment = 12000)
  (h2 : investment_12_percent = 4000.000000000002)
  (h3 : price_12_percent = 120)
  (h4 : price_15_percent = 125)
  (h5 : dividend_rate_12_percent = 0.12)
  (h6 : dividend_rate_15_percent = 0.15) :
  ∃ (total_dividend : ℝ), abs (total_dividend - 1680) < 1 :=
sorry

end sakshi_investment_dividend_l3704_370460


namespace original_cat_count_l3704_370462

theorem original_cat_count (original_dogs original_cats current_dogs current_cats : ℕ) :
  original_dogs = original_cats / 2 →
  current_dogs = original_dogs + 20 →
  current_dogs = 2 * current_cats →
  current_cats = 20 →
  original_cats = 40 :=
by
  sorry

end original_cat_count_l3704_370462


namespace ashok_marks_average_l3704_370480

/-- Given a student's average marks and the marks in the last subject, 
    calculate the average marks in the remaining subjects. -/
def average_remaining_subjects (total_subjects : ℕ) (overall_average : ℚ) (last_subject_marks : ℕ) : ℚ :=
  ((overall_average * total_subjects) - last_subject_marks) / (total_subjects - 1)

/-- Theorem stating that given the conditions in the problem, 
    the average of marks in the first 5 subjects is 74. -/
theorem ashok_marks_average : 
  let total_subjects : ℕ := 6
  let overall_average : ℚ := 75
  let last_subject_marks : ℕ := 80
  average_remaining_subjects total_subjects overall_average last_subject_marks = 74 := by
  sorry

#eval average_remaining_subjects 6 75 80

end ashok_marks_average_l3704_370480


namespace arccos_one_half_l3704_370461

theorem arccos_one_half : Real.arccos (1/2) = π/3 := by
  sorry

end arccos_one_half_l3704_370461


namespace ellipse_eccentricity_l3704_370494

/-- The eccentricity of an ellipse with specific properties -/
theorem ellipse_eccentricity (a b : ℝ) (h_ab : a > b) (h_b_pos : b > 0) :
  let e := Real.sqrt 5 / 2 - 1 / 2
  ∃ (F₁ F₂ P : ℝ × ℝ),
    -- F₁ and F₂ are the foci of the ellipse
    F₁.1 = -Real.sqrt (a^2 - b^2) ∧ F₁.2 = 0 ∧
    F₂.1 = Real.sqrt (a^2 - b^2) ∧ F₂.2 = 0 ∧
    -- P is on the ellipse
    P.1^2 / a^2 + P.2^2 / b^2 = 1 ∧
    -- PF₂ is perpendicular to the x-axis
    P.1 = F₂.1 ∧
    -- |F₁F₂| = 2|PF₂|
    (F₁.1 - F₂.1)^2 = 4 * P.2^2 ∧
    -- The eccentricity is e
    e = Real.sqrt (a^2 - b^2) / a := by
  sorry

end ellipse_eccentricity_l3704_370494


namespace gage_skating_problem_l3704_370472

theorem gage_skating_problem (days_75min : ℕ) (days_90min : ℕ) (total_days : ℕ) (avg_minutes : ℕ) :
  days_75min = 5 →
  days_90min = 3 →
  total_days = days_75min + days_90min + 1 →
  avg_minutes = 85 →
  (days_75min * 75 + days_90min * 90 + (total_days * avg_minutes - (days_75min * 75 + days_90min * 90))) / total_days = avg_minutes :=
by sorry

end gage_skating_problem_l3704_370472


namespace problems_left_to_solve_l3704_370416

def math_test (total_problems : ℕ) (first_20min : ℕ) (second_20min : ℕ) : Prop :=
  total_problems = 75 ∧
  first_20min = 10 ∧
  second_20min = 2 * first_20min ∧
  total_problems - (first_20min + second_20min) = 45

theorem problems_left_to_solve :
  ∀ (total_problems first_20min second_20min : ℕ),
    math_test total_problems first_20min second_20min →
    total_problems - (first_20min + second_20min) = 45 :=
by
  sorry

end problems_left_to_solve_l3704_370416


namespace tangent_line_and_max_value_l3704_370418

open Real

noncomputable def f (x : ℝ) := -log x + (1/2) * x^2

theorem tangent_line_and_max_value :
  (∀ x, x ∈ Set.Icc (1/Real.exp 1) (Real.sqrt (Real.exp 1)) →
    f x ≤ 1 + 1 / (2 * (Real.exp 1)^2)) ∧
  (∃ x, x ∈ Set.Icc (1/Real.exp 1) (Real.sqrt (Real.exp 1)) ∧
    f x = 1 + 1 / (2 * (Real.exp 1)^2)) ∧
  (3 * 2 - 2 * f 2 - 2 - 2 * log 2 = 0) :=
by sorry

end tangent_line_and_max_value_l3704_370418


namespace htf_sequence_probability_l3704_370437

/-- A fair coin has equal probability of landing heads or tails -/
def fair_coin (p : ℝ) : Prop := p = 1/2

/-- The probability of a specific sequence of three independent coin flips -/
def prob_sequence (p : ℝ) : ℝ := p * p * p

theorem htf_sequence_probability :
  ∀ p : ℝ, fair_coin p → prob_sequence p = 1/8 := by sorry

end htf_sequence_probability_l3704_370437


namespace factor_calculation_l3704_370408

theorem factor_calculation : ∃ f : ℝ, (2 * 9 + 6) * f = 72 ∧ f = 3 := by
  sorry

end factor_calculation_l3704_370408


namespace difference_in_sums_l3704_370493

def sum_of_integers (n : ℕ) : ℕ := n * (n + 1) / 2

def round_to_nearest_five (n : ℕ) : ℕ :=
  5 * ((n + 2) / 5)

def sum_of_rounded_integers (n : ℕ) : ℕ :=
  (List.range n).map round_to_nearest_five |>.sum

theorem difference_in_sums :
  sum_of_rounded_integers 200 - sum_of_integers 200 = 120 := by
  sorry

end difference_in_sums_l3704_370493


namespace maximize_x_cubed_y_fourth_l3704_370419

theorem maximize_x_cubed_y_fourth (x y : ℝ) (hx : x > 0) (hy : y > 0) (hsum : x + y = 60) :
  x^3 * y^4 ≤ (3/7 * 60)^3 * (4/7 * 60)^4 ∧
  x^3 * y^4 = (3/7 * 60)^3 * (4/7 * 60)^4 ↔ x = 3/7 * 60 ∧ y = 4/7 * 60 :=
by sorry

end maximize_x_cubed_y_fourth_l3704_370419


namespace cube_surface_area_doubles_l3704_370406

/-- Theorem: Doubling the edge length of a cube increases its surface area by a factor of 4 -/
theorem cube_surface_area_doubles (a : ℝ) (h : a > 0) :
  (6 * (2 * a)^2) / (6 * a^2) = 4 := by
  sorry

#check cube_surface_area_doubles

end cube_surface_area_doubles_l3704_370406


namespace tangent_line_circle_a_value_l3704_370468

/-- A line is tangent to a circle if and only if the distance from the center of the circle to the line equals the radius of the circle. -/
axiom line_tangent_to_circle_iff_distance_eq_radius {a b c d e f : ℝ} :
  (∀ x y, a*x + b*y + c = 0 → (x - d)^2 + (y - e)^2 = f^2) ↔
  |a*d + b*e + c| / Real.sqrt (a^2 + b^2) = f

/-- Given that the line 5x + 12y + a = 0 is tangent to the circle x^2 - 2x + y^2 = 0,
    prove that a = 8 or a = -18 -/
theorem tangent_line_circle_a_value :
  (∀ x y, 5*x + 12*y + a = 0 → x^2 - 2*x + y^2 = 0) →
  a = 8 ∨ a = -18 := by
sorry

end tangent_line_circle_a_value_l3704_370468


namespace complex_number_quadrant_l3704_370429

theorem complex_number_quadrant : ∃ (z : ℂ), 
  (z + Complex.I) * (1 - 2 * Complex.I) = 2 ∧ 
  0 < z.re ∧ z.im < 0 := by
  sorry

end complex_number_quadrant_l3704_370429


namespace quadratic_function_properties_l3704_370459

def f (x : ℝ) : ℝ := -x^2 + x + 6

theorem quadratic_function_properties :
  (f (-3) = -6 ∧ f 0 = 6 ∧ f 2 = 4) →
  (∀ x : ℝ, f x = -x^2 + x + 6) ∧
  (∀ x : ℝ, f x ≤ 25/4) ∧
  (f (1/2) = 25/4) ∧
  (∀ x : ℝ, f (x + 1/2) = f (1/2 - x)) :=
by sorry

end quadratic_function_properties_l3704_370459


namespace intersection_A_B_l3704_370420

def A : Set ℝ := {-1, 0, 1}

def B : Set ℝ := {x : ℝ | x * (x - 1) ≤ 0}

theorem intersection_A_B : A ∩ B = {0, 1} := by sorry

end intersection_A_B_l3704_370420


namespace grandfather_grandson_age_relation_l3704_370477

theorem grandfather_grandson_age_relation :
  ∀ (grandfather_age grandson_age : ℕ) (years : ℕ),
    50 < grandfather_age →
    grandfather_age < 90 →
    grandfather_age = 31 * grandson_age →
    (grandfather_age + years = 7 * (grandson_age + years)) →
    years = 8 :=
by sorry

end grandfather_grandson_age_relation_l3704_370477


namespace heather_total_distance_l3704_370483

/-- The distance Heather bicycled per day in kilometers -/
def distance_per_day : ℝ := 40.0

/-- The number of days Heather bicycled -/
def number_of_days : ℝ := 8.0

/-- The total distance Heather bicycled -/
def total_distance : ℝ := distance_per_day * number_of_days

theorem heather_total_distance : total_distance = 320.0 := by
  sorry

end heather_total_distance_l3704_370483


namespace stool_height_is_85_alice_can_reach_light_bulb_l3704_370425

/-- The minimum height of the stool Alice needs to reach the light bulb -/
def stool_height : ℝ :=
  let ceiling_height : ℝ := 280  -- in cm
  let light_bulb_height : ℝ := ceiling_height - 15
  let alice_height : ℝ := 150  -- in cm
  let alice_reach : ℝ := alice_height + 30
  light_bulb_height - alice_reach

theorem stool_height_is_85 :
  stool_height = 85 := by sorry

/-- Alice can reach the light bulb with the calculated stool height -/
theorem alice_can_reach_light_bulb :
  let ceiling_height : ℝ := 280  -- in cm
  let light_bulb_height : ℝ := ceiling_height - 15
  let alice_height : ℝ := 150  -- in cm
  let alice_reach : ℝ := alice_height + 30
  alice_reach + stool_height = light_bulb_height := by sorry

end stool_height_is_85_alice_can_reach_light_bulb_l3704_370425


namespace butter_fraction_for_chocolate_chip_cookies_l3704_370498

theorem butter_fraction_for_chocolate_chip_cookies 
  (total_butter : ℝ)
  (peanut_butter_fraction : ℝ)
  (sugar_cookie_fraction : ℝ)
  (remaining_butter : ℝ)
  (h1 : total_butter = 10)
  (h2 : peanut_butter_fraction = 1/5)
  (h3 : sugar_cookie_fraction = 1/3)
  (h4 : remaining_butter = 2)
  : (total_butter - (peanut_butter_fraction * total_butter) - 
     sugar_cookie_fraction * (total_butter - peanut_butter_fraction * total_butter) - 
     remaining_butter) / total_butter = 1/3 := by
  sorry

#check butter_fraction_for_chocolate_chip_cookies

end butter_fraction_for_chocolate_chip_cookies_l3704_370498


namespace central_diamond_area_l3704_370412

/-- The area of the central diamond-shaped region in a 10x10 square --/
theorem central_diamond_area (square_side : ℝ) (h : square_side = 10) : 
  let diagonal_length : ℝ := square_side * Real.sqrt 2
  let midpoint_distance : ℝ := square_side / 2
  let diamond_area : ℝ := diagonal_length * midpoint_distance / 2
  diamond_area = 50 := by sorry

end central_diamond_area_l3704_370412


namespace dart_game_equations_correct_l3704_370473

/-- Represents the dart throwing game scenario -/
structure DartGame where
  x : ℕ  -- number of times Xiao Hua hits the target
  y : ℕ  -- number of times the father hits the target

/-- The conditions of the dart throwing game -/
def validGame (game : DartGame) : Prop :=
  game.x + game.y = 30 ∧  -- total number of hits
  5 * game.x + 2 = 3 * game.y  -- score difference condition

/-- Theorem stating that the system of equations correctly represents the game -/
theorem dart_game_equations_correct (game : DartGame) :
  validGame game ↔ 
    (game.x + game.y = 30 ∧ 5 * game.x + 2 = 3 * game.y) :=
by sorry

end dart_game_equations_correct_l3704_370473


namespace consecutive_odd_numbers_ratio_l3704_370482

theorem consecutive_odd_numbers_ratio (x : ℝ) (k m : ℝ) : 
  x = 4.2 →                             -- First number is 4.2
  9 * x = k * (x + 4) + m * (x + 2) + 9  -- Equation from the problem
    → (x + 4) / (x + 2) = 41 / 31        -- Ratio of third to second number
  := by sorry

end consecutive_odd_numbers_ratio_l3704_370482


namespace total_amount_is_952_20_l3704_370452

/-- Calculate the total amount paid for three items with given original prices, discounts, and sales taxes. -/
def total_amount_paid (vase_price teacups_price plate_price : ℝ)
                      (vase_discount teacups_discount : ℝ)
                      (vase_tax teacups_tax plate_tax : ℝ) : ℝ :=
  let vase_sale_price := vase_price * (1 - vase_discount)
  let teacups_sale_price := teacups_price * (1 - teacups_discount)
  let vase_total := vase_sale_price * (1 + vase_tax)
  let teacups_total := teacups_sale_price * (1 + teacups_tax)
  let plate_total := plate_price * (1 + plate_tax)
  vase_total + teacups_total + plate_total

/-- The total amount paid for the three porcelain items is $952.20. -/
theorem total_amount_is_952_20 :
  total_amount_paid 200 300 500 0.35 0.20 0.10 0.08 0.10 = 952.20 := by
  sorry

end total_amount_is_952_20_l3704_370452


namespace system_solution_l3704_370401

theorem system_solution (x y : ℝ) (hx : x > 0) (hy : y > 0) :
  (2 * x - Real.sqrt (x * y) - 4 * Real.sqrt (x / y) + 2 = 0 ∧
   2 * x^2 + x^2 * y^4 = 18 * y^2) ↔
  ((x = 2 ∧ y = 2) ∨ (x = Real.rpow 286 (1/4) / 4 ∧ y = Real.rpow 286 (1/4))) :=
sorry

end system_solution_l3704_370401


namespace fifth_term_is_negative_9216_l3704_370469

def alternating_sequence (n : ℕ) : ℤ := 
  if n % 2 = 0 then (101 - n)^2 else -((101 - n)^2)

def sequence_sum (n : ℕ) : ℤ := 
  (List.range n).map alternating_sequence |>.sum

theorem fifth_term_is_negative_9216 (h : sequence_sum 100 = 5050) : 
  alternating_sequence 5 = -9216 := by
  sorry

end fifth_term_is_negative_9216_l3704_370469


namespace planes_parallel_l3704_370456

-- Define the types for our geometric objects
variable (Point Line Plane : Type)

-- Define the geometric relations
variable (belongs_to : Point → Line → Prop)
variable (subset : Line → Plane → Prop)
variable (parallel_line_plane : Line → Plane → Prop)
variable (parallel_planes : Plane → Plane → Prop)
variable (noncoplanar : Line → Line → Prop)

-- State the theorem
theorem planes_parallel 
  (α β : Plane) (a b : Line) :
  noncoplanar a b →
  subset a α →
  subset b β →
  parallel_line_plane a β →
  parallel_line_plane b α →
  parallel_planes α β :=
sorry

end planes_parallel_l3704_370456


namespace fraction_problem_l3704_370427

theorem fraction_problem (x : ℝ) (f : ℝ) (h1 : x = 145) (h2 : x - f * x = 58) : f = 0.6 := by
  sorry

end fraction_problem_l3704_370427


namespace reach_floor_pushups_l3704_370471

/-- Represents the number of push-up variations -/
def numVariations : ℕ := 5

/-- Represents the number of training days per week -/
def trainingDaysPerWeek : ℕ := 6

/-- Represents the number of reps added per day -/
def repsAddedPerDay : ℕ := 1

/-- Represents the target number of reps to progress to the next variation -/
def targetReps : ℕ := 25

/-- Calculates the number of weeks needed to progress through one variation -/
def weeksPerVariation : ℕ := 
  (targetReps + trainingDaysPerWeek - 1) / trainingDaysPerWeek

/-- The total number of weeks needed to reach floor push-ups -/
def totalWeeks : ℕ := numVariations * weeksPerVariation

theorem reach_floor_pushups : totalWeeks = 20 := by
  sorry

end reach_floor_pushups_l3704_370471


namespace phone_bill_ratio_l3704_370449

theorem phone_bill_ratio (jan_total feb_total internet_charge : ℚ)
  (h1 : jan_total = 46)
  (h2 : feb_total = 76)
  (h3 : internet_charge = 16) :
  (feb_total - internet_charge) / (jan_total - internet_charge) = 2 := by
  sorry

end phone_bill_ratio_l3704_370449


namespace x_squared_equals_one_l3704_370445

theorem x_squared_equals_one (x : ℝ) (h1 : x > 0) (h2 : Real.sin (Real.arctan x) = 1 / x) : x^2 = 1 := by
  sorry

end x_squared_equals_one_l3704_370445


namespace equation_equivalence_l3704_370432

theorem equation_equivalence (x y : ℕ) : 
  (∃ (a b c d : ℕ), x = a + 2*b + 3*c + 7*d ∧ y = b + 2*c + 5*d) ↔ 
  5*x ≥ 7*y := by
sorry

end equation_equivalence_l3704_370432


namespace problem_solution_l3704_370417

theorem problem_solution : ∃ x : ℝ, (0.25 * x = 0.15 * 1500 - 15) ∧ (x = 840) := by
  sorry

end problem_solution_l3704_370417


namespace cost_price_correct_l3704_370414

/-- The cost price of a piece of clothing -/
def cost_price : ℝ := 108

/-- The marked price of the clothing -/
def marked_price : ℝ := 132

/-- The discount rate applied to the clothing -/
def discount_rate : ℝ := 0.1

/-- The profit rate after applying the discount -/
def profit_rate : ℝ := 0.1

/-- Theorem stating that the cost price is correct given the conditions -/
theorem cost_price_correct :
  marked_price * (1 - discount_rate) = cost_price * (1 + profit_rate) :=
by sorry

end cost_price_correct_l3704_370414


namespace sum_of_even_coefficients_l3704_370467

theorem sum_of_even_coefficients (a₀ a₁ a₂ a₃ a₄ a₅ : ℝ) :
  (∀ x : ℝ, (x + 1)^4 + (x - 1)^5 = a₀ + a₁*x + a₂*x^2 + a₃*x^3 + a₄*x^4 + a₅*x^5) →
  a₀ + a₂ + a₄ = -8 := by
sorry

end sum_of_even_coefficients_l3704_370467


namespace sum_a_d_l3704_370478

theorem sum_a_d (a b c d : ℤ) 
  (h1 : a + b = 4) 
  (h2 : b + c = 7) 
  (h3 : c + d = 5) : 
  a + d = 4 := by
sorry

end sum_a_d_l3704_370478


namespace solution_set_when_a_is_2_range_of_a_l3704_370446

-- Define the function f
def f (x a : ℝ) : ℝ := |x - a^2| + |x - 2*a + 1|

-- Part 1: When a = 2, prove the solution set of f(x) ≥ 4
theorem solution_set_when_a_is_2 :
  {x : ℝ | f x 2 ≥ 4} = {x : ℝ | x ≤ 3/2 ∨ x ≥ 11/2} :=
sorry

-- Part 2: Prove the range of a for which f(x) ≥ 4
theorem range_of_a :
  {a : ℝ | ∃ x, f x a ≥ 4} = {a : ℝ | a ≤ -1 ∨ a ≥ 3} :=
sorry

end solution_set_when_a_is_2_range_of_a_l3704_370446


namespace valid_three_digit_numbers_count_l3704_370484

/-- The count of three-digit numbers where either all digits are the same or the first and last digits are different -/
def validThreeDigitNumbers : ℕ :=
  -- Total three-digit numbers
  let totalThreeDigitNumbers := 999 - 100 + 1
  -- Numbers to exclude (ABA form where A ≠ B and B ≠ 0)
  let excludedNumbers := 10 * 9
  -- Calculation
  totalThreeDigitNumbers - excludedNumbers

/-- Theorem stating that the count of valid three-digit numbers is 810 -/
theorem valid_three_digit_numbers_count : validThreeDigitNumbers = 810 := by
  sorry


end valid_three_digit_numbers_count_l3704_370484


namespace odd_number_set_characterization_l3704_370405

def OddNumberSet : Set ℤ :=
  {x | -8 < x ∧ x < 20 ∧ ∃ k : ℤ, x = 2 * k + 1}

theorem odd_number_set_characterization :
  OddNumberSet = {x : ℤ | -8 < x ∧ x < 20 ∧ ∃ k : ℤ, x = 2 * k + 1} := by
  sorry

end odd_number_set_characterization_l3704_370405


namespace symmetry_implies_exponential_l3704_370463

-- Define the logarithm function base 3
noncomputable def log3 (x : ℝ) : ℝ := Real.log x / Real.log 3

-- Define the symmetry condition
def symmetric_wrt_y_eq_x (f g : ℝ → ℝ) : Prop :=
  ∀ x y, f x = y ↔ g y = x

-- State the theorem
theorem symmetry_implies_exponential (f : ℝ → ℝ) :
  (∀ x > 0, f (log3 x) = x) →
  symmetric_wrt_y_eq_x f log3 →
  ∀ x, f x = 3^x :=
sorry

end symmetry_implies_exponential_l3704_370463


namespace driver_net_pay_rate_driver_net_pay_is_26_l3704_370442

/-- Calculates the net rate of pay for a driver given specific conditions --/
theorem driver_net_pay_rate (hours : ℝ) (speed : ℝ) (fuel_efficiency : ℝ) 
  (ac_efficiency_decrease : ℝ) (pay_per_mile : ℝ) (gas_price : ℝ) 
  (gas_price_increase : ℝ) : ℝ :=
  let distance := hours * speed
  let adjusted_fuel_efficiency := fuel_efficiency * (1 - ac_efficiency_decrease)
  let gas_used := distance / adjusted_fuel_efficiency
  let earnings := pay_per_mile * distance
  let new_gas_price := gas_price * (1 + gas_price_increase)
  let gas_cost := new_gas_price * gas_used
  let net_earnings := earnings - gas_cost
  let net_rate := net_earnings / hours
  net_rate

/-- Proves that the driver's net rate of pay is $26 per hour under given conditions --/
theorem driver_net_pay_is_26 :
  driver_net_pay_rate 3 50 30 0.1 0.6 2 0.2 = 26 := by
  sorry

end driver_net_pay_rate_driver_net_pay_is_26_l3704_370442


namespace stratified_sample_female_count_l3704_370434

/-- Calculates the number of female athletes in a stratified sample -/
theorem stratified_sample_female_count 
  (total_athletes : ℕ) 
  (female_athletes : ℕ) 
  (male_athletes : ℕ) 
  (sample_size : ℕ) 
  (h1 : total_athletes = female_athletes + male_athletes)
  (h2 : total_athletes = 98)
  (h3 : female_athletes = 42)
  (h4 : male_athletes = 56)
  (h5 : sample_size = 28) :
  (female_athletes : ℚ) * (sample_size : ℚ) / (total_athletes : ℚ) = 12 := by
  sorry

#check stratified_sample_female_count

end stratified_sample_female_count_l3704_370434


namespace rahul_savings_fraction_l3704_370400

def total_savings : ℕ := 180000
def ppf_savings : ℕ := 72000
def nsc_savings : ℕ := total_savings - ppf_savings

def fraction_equality (x : ℚ) : Prop :=
  x * nsc_savings = (1/2 : ℚ) * ppf_savings

theorem rahul_savings_fraction : 
  ∃ (x : ℚ), fraction_equality x ∧ x = (1/3 : ℚ) :=
sorry

end rahul_savings_fraction_l3704_370400


namespace village_language_probability_l3704_370407

/-- Given a village with the following properties:
  - Total population is 1500
  - 800 people speak Tamil
  - 650 people speak English
  - 250 people speak both Tamil and English
  Prove that the probability of a randomly chosen person speaking neither English nor Tamil is 1/5 -/
theorem village_language_probability (total : ℕ) (tamil : ℕ) (english : ℕ) (both : ℕ)
  (h_total : total = 1500)
  (h_tamil : tamil = 800)
  (h_english : english = 650)
  (h_both : both = 250) :
  (total - (tamil + english - both)) / total = 1 / 5 := by
  sorry

end village_language_probability_l3704_370407


namespace complement_of_A_in_U_l3704_370415

def U : Set ℕ := {1, 2, 3, 4, 5, 6, 7}
def A : Set ℕ := {2, 4, 5}

theorem complement_of_A_in_U :
  U \ A = {1, 3, 6, 7} := by sorry

end complement_of_A_in_U_l3704_370415


namespace expression_equality_l3704_370487

theorem expression_equality : 
  Real.sqrt 8 + Real.sqrt (1/2) + (Real.sqrt 3 - 1)^2 + Real.sqrt 6 / (1/2 * Real.sqrt 2) = 
  5/2 * Real.sqrt 2 + 4 := by
  sorry

end expression_equality_l3704_370487


namespace expression_simplification_l3704_370458

theorem expression_simplification (x : ℝ) (h : x ≠ 0) :
  (x * (3 - 4 * x) + 2 * x^2 * (x - 1)) / (-2 * x) = -x^2 + 3 * x - 3/2 := by
  sorry

end expression_simplification_l3704_370458


namespace functional_equation_solution_l3704_370438

theorem functional_equation_solution (g : ℝ → ℝ) 
  (h : ∀ x y : ℝ, (g x * g y - g (x * y)) / 5 = 2 * x + 2 * y + 8) :
  ∀ x : ℝ, g x = -2 * x - 7 := by
  sorry

end functional_equation_solution_l3704_370438


namespace first_month_sale_is_7435_l3704_370488

/-- Calculates the sale in the first month given the sales for months 2 to 6 and the average sale for 6 months -/
def first_month_sale (second_month : ℕ) (third_month : ℕ) (fourth_month : ℕ) (fifth_month : ℕ) (sixth_month : ℕ) (average_sale : ℕ) : ℕ :=
  6 * average_sale - (second_month + third_month + fourth_month + fifth_month + sixth_month)

/-- Theorem stating that the sale in the first month is 7435 given the specified conditions -/
theorem first_month_sale_is_7435 :
  first_month_sale 7927 7855 8230 7562 5991 7500 = 7435 := by
  sorry

end first_month_sale_is_7435_l3704_370488


namespace prob_rain_at_least_one_day_l3704_370454

def prob_rain_friday : ℝ := 0.6
def prob_rain_saturday : ℝ := 0.7
def prob_rain_sunday : ℝ := 0.4

theorem prob_rain_at_least_one_day :
  let prob_no_rain_friday := 1 - prob_rain_friday
  let prob_no_rain_saturday := 1 - prob_rain_saturday
  let prob_no_rain_sunday := 1 - prob_rain_sunday
  let prob_no_rain_all_days := prob_no_rain_friday * prob_no_rain_saturday * prob_no_rain_sunday
  let prob_rain_at_least_one_day := 1 - prob_no_rain_all_days
  prob_rain_at_least_one_day = 0.928 := by
sorry

end prob_rain_at_least_one_day_l3704_370454


namespace quadratic_polynomial_property_l3704_370441

/-- A quadratic polynomial -/
def QuadraticPolynomial (R : Type*) [Field R] := R → R

/-- Property that p(n) = 1/n^2 for n = 1, 2, 3 -/
def SatisfiesCondition (p : QuadraticPolynomial ℝ) : Prop :=
  p 1 = 1 ∧ p 2 = 1/4 ∧ p 3 = 1/9

theorem quadratic_polynomial_property (p : QuadraticPolynomial ℝ) 
  (h : SatisfiesCondition p) : p 4 = -9/16 := by
  sorry

end quadratic_polynomial_property_l3704_370441


namespace parabola_b_value_l3704_370421

/-- A parabola with equation y = x^2 + bx + 3 passing through the points (1, 5), (3, 5), and (0, 3) has b = 1 -/
theorem parabola_b_value : ∃ b : ℝ,
  (∀ x y : ℝ, y = x^2 + b*x + 3 →
    ((x = 1 ∧ y = 5) ∨ (x = 3 ∧ y = 5) ∨ (x = 0 ∧ y = 3))) →
  b = 1 := by
  sorry

end parabola_b_value_l3704_370421


namespace simplify_expression_1_simplify_expression_2_l3704_370433

-- Define the logarithm base 10 function
noncomputable def log10 (x : ℝ) : ℝ := Real.log x / Real.log 10

-- Statement for the first expression
theorem simplify_expression_1 : 
  2 * Real.sqrt 3 * (1.5 : ℝ) ^ (1/3) * 12 ^ (1/6) = 6 := by sorry

-- Statement for the second expression
theorem simplify_expression_2 : 
  log10 25 + (2/3) * log10 8 + log10 5 * log10 20 + (log10 2)^2 = 3 := by sorry

end simplify_expression_1_simplify_expression_2_l3704_370433


namespace savings_problem_l3704_370444

theorem savings_problem (S : ℝ) : 
  (S * 1.1 * (2 / 10) = 44) → S = 200 := by sorry

end savings_problem_l3704_370444


namespace normal_distribution_probability_l3704_370443

-- Define a normal distribution
def normal_distribution (μ σ : ℝ) : Type := ℝ

-- Define the probability function
noncomputable def P {α : Type} (event : Set α) : ℝ := sorry

-- Define the random variable ξ
def ξ : normal_distribution 0 σ := sorry

-- State the theorem
theorem normal_distribution_probability 
  (h : P {x | -2 ≤ x ∧ x ≤ 0} = 0.4) : 
  P {x | x > 2} = 0.1 := by sorry

end normal_distribution_probability_l3704_370443


namespace runner_ends_at_start_l3704_370448

/-- A runner on a circular track --/
structure Runner where
  start : ℝ  -- Starting position on the track (in feet)
  distance : ℝ  -- Total distance run (in feet)

/-- The circular track --/
def track_circumference : ℝ := 60

/-- Theorem: A runner who starts at any point and runs exactly 5400 feet will end at the same point --/
theorem runner_ends_at_start (runner : Runner) (h : runner.distance = 5400) :
  runner.start = (runner.start + runner.distance) % track_circumference := by
  sorry

end runner_ends_at_start_l3704_370448


namespace inequalities_from_sqrt_reciprocal_l3704_370476

theorem inequalities_from_sqrt_reciprocal (a b : ℝ) (h : 1 / Real.sqrt a > 1 / Real.sqrt b) :
  (b / (a + b) + a / (2 * b) ≥ (2 * Real.sqrt 2 - 1) / 2) ∧
  ((b + 1) / (a + 1) < b / a) := by
  sorry

end inequalities_from_sqrt_reciprocal_l3704_370476


namespace decreasing_interval_of_quadratic_l3704_370435

def f (x : ℝ) := x^2 - 2*x - 3

theorem decreasing_interval_of_quadratic :
  ∀ x : ℝ, (∀ y : ℝ, y < x → f y < f x) ↔ x ≤ 1 :=
by sorry

end decreasing_interval_of_quadratic_l3704_370435


namespace segment_movement_area_reduction_l3704_370422

theorem segment_movement_area_reduction (AB d : ℝ) (hAB : AB > 0) (hd : d > 0) :
  ∃ (swept_area : ℝ), swept_area < (AB * d) / 10000 ∧ swept_area ≥ 0 := by
  sorry

end segment_movement_area_reduction_l3704_370422


namespace marbles_shared_proof_l3704_370402

/-- The number of marbles Carolyn started with -/
def initial_marbles : ℕ := 47

/-- The number of marbles Carolyn ended up with after sharing -/
def final_marbles : ℕ := 5

/-- The number of marbles Carolyn shared -/
def shared_marbles : ℕ := initial_marbles - final_marbles

theorem marbles_shared_proof : shared_marbles = 42 := by
  sorry

end marbles_shared_proof_l3704_370402


namespace product_remainder_l3704_370496

theorem product_remainder (n : ℤ) : (12 - 2*n) * (n + 5) ≡ -2*n^2 + 2*n + 5 [ZMOD 11] := by
  sorry

end product_remainder_l3704_370496


namespace triangle_side_length_squared_l3704_370403

theorem triangle_side_length_squared (A B C : ℝ × ℝ) :
  let area := 10
  let tan_ABC := 5
  area = (1/2) * abs (A.1 * B.2 + B.1 * C.2 + C.1 * A.2 - A.2 * B.1 - B.2 * C.1 - C.2 * A.1) →
  tan_ABC = (B.2 - A.2) / (B.1 - A.1) →
  ∃ (AC_squared : ℝ), AC_squared = (C.1 - A.1)^2 + (C.2 - A.2)^2 ∧
    AC_squared ≥ -8 + 8 * Real.sqrt 26 :=
by sorry

#check triangle_side_length_squared

end triangle_side_length_squared_l3704_370403


namespace museum_visitors_scientific_notation_l3704_370491

/-- Represents a number in scientific notation -/
structure ScientificNotation where
  coefficient : ℝ
  exponent : ℤ
  is_valid : 1 ≤ coefficient ∧ coefficient < 10

/-- Converts a real number to scientific notation -/
def toScientificNotation (n : ℝ) : ScientificNotation :=
  sorry

theorem museum_visitors_scientific_notation :
  toScientificNotation 3300000 = ScientificNotation.mk 3.3 6 (by norm_num) :=
sorry

end museum_visitors_scientific_notation_l3704_370491


namespace intersection_empty_union_equals_B_l3704_370404

-- Define set A
def A (a : ℝ) : Set ℝ := {x | (x - a) * (x - (a + 3)) ≤ 0}

-- Define set B
def B : Set ℝ := {x | x^2 - 4*x - 5 > 0}

-- Theorem for the first part
theorem intersection_empty (a : ℝ) : A a ∩ B = ∅ ↔ -1 ≤ a ∧ a ≤ 2 := by sorry

-- Theorem for the second part
theorem union_equals_B (a : ℝ) : A a ∪ B = B ↔ a > 5 ∨ a < -4 := by sorry

end intersection_empty_union_equals_B_l3704_370404


namespace second_level_treasures_is_two_l3704_370428

/-- Represents the number of points scored per treasure -/
def points_per_treasure : ℕ := 4

/-- Represents the number of treasures found on the first level -/
def first_level_treasures : ℕ := 6

/-- Represents the total score -/
def total_score : ℕ := 32

/-- Calculates the number of treasures found on the second level -/
def second_level_treasures : ℕ :=
  (total_score - (first_level_treasures * points_per_treasure)) / points_per_treasure

/-- Theorem stating that the number of treasures found on the second level is 2 -/
theorem second_level_treasures_is_two : second_level_treasures = 2 := by
  sorry

end second_level_treasures_is_two_l3704_370428


namespace staff_discount_percentage_l3704_370457

theorem staff_discount_percentage (d : ℝ) : 
  d > 0 →  -- Assuming the original price is positive
  let discounted_price := 0.85 * d  -- Price after 15% discount
  let final_price := 0.765 * d      -- Price staff member pays
  let staff_discount_percent := (discounted_price - final_price) / discounted_price * 100
  staff_discount_percent = 10 := by
sorry

end staff_discount_percentage_l3704_370457


namespace simplify_expression_l3704_370431

theorem simplify_expression : 3 * (((1 + 2 + 3 + 4) * 3) + ((1 * 4 + 16) / 4)) = 105 := by
  sorry

end simplify_expression_l3704_370431


namespace pyramid_properties_l3704_370492

/-- Pyramid structure with given properties -/
structure Pyramid where
  -- Base is a rhombus
  base_is_rhombus : Prop
  -- Height of the pyramid
  height : ℝ
  -- K lies on diagonal AC
  k_on_diagonal : Prop
  -- KC = KA + AC
  kc_eq_ka_plus_ac : Prop
  -- Length of lateral edge TC
  tc_length : ℝ
  -- Angles of lateral faces to base
  angle1 : ℝ
  angle2 : ℝ

/-- Theorem stating the properties of the pyramid -/
theorem pyramid_properties (p : Pyramid)
  (h_height : p.height = 1)
  (h_tc : p.tc_length = 2 * Real.sqrt 2)
  (h_angles : p.angle1 = π/6 ∧ p.angle2 = π/3) :
  ∃ (base_side angle_ta_tcd : ℝ),
    base_side = 7/6 ∧
    angle_ta_tcd = Real.arcsin (Real.sqrt 3 / 4) :=
by sorry

end pyramid_properties_l3704_370492


namespace f_two_values_l3704_370439

/-- A function satisfying the given property -/
def special_function (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, |f x - f y| = |x - y|

/-- Theorem stating the possible values of f(2) given the conditions -/
theorem f_two_values (f : ℝ → ℝ) (h : special_function f) (h1 : f 1 = 3) :
  f 2 = 2 ∨ f 2 = 4 := by
  sorry

end f_two_values_l3704_370439


namespace price_markup_markdown_l3704_370450

theorem price_markup_markdown (x : ℝ) (h : x > 0) : x * (1 + 0.1) * (1 - 0.1) < x := by
  sorry

end price_markup_markdown_l3704_370450


namespace race_time_comparison_l3704_370453

theorem race_time_comparison (a V : ℝ) (h_a : a > 0) (h_V : V > 0) :
  let planned_time := a / V
  let first_half_time := a / (2 * 1.25 * V)
  let second_half_time := a / (2 * 0.8 * V)
  let actual_time := first_half_time + second_half_time
  actual_time > planned_time := by sorry

end race_time_comparison_l3704_370453


namespace right_triangle_hypotenuse_l3704_370486

theorem right_triangle_hypotenuse (a b c : ℝ) : 
  a > 0 → b > 0 → c > 0 →
  b = 3 * a →
  a^2 + b^2 = c^2 →
  a^2 + b^2 + c^2 = 500 →
  c = 5 * Real.sqrt 10 := by
sorry

end right_triangle_hypotenuse_l3704_370486


namespace negation_of_existence_tan_equals_one_l3704_370495

theorem negation_of_existence_tan_equals_one :
  (¬ ∃ x : ℝ, Real.tan x = 1) ↔ (∀ x : ℝ, Real.tan x ≠ 1) := by sorry

end negation_of_existence_tan_equals_one_l3704_370495


namespace no_integer_solution_l3704_370470

theorem no_integer_solution : ¬∃ (x y : ℤ), x * (x + 1) = 13 * y + 1 := by
  sorry

end no_integer_solution_l3704_370470


namespace probability_of_red_ball_l3704_370410

/-- The probability of drawing a red ball from a bag with red and black balls -/
theorem probability_of_red_ball (red_balls black_balls : ℕ) : 
  red_balls = 3 → black_balls = 9 → 
  (red_balls : ℚ) / (red_balls + black_balls) = 1 / 4 := by
  sorry

end probability_of_red_ball_l3704_370410


namespace f_equality_min_t_value_range_N_minus_n_l3704_370424

-- Define the function f
def f (x : ℝ) : ℝ := 2 * x^2 - x - 3

-- Define the function g
def g (a x : ℝ) : ℝ := f x + 2 * x^3 - (a + 2) * x^2 + x + 5

-- Theorem 1: Prove that f(2x-1) = 8x^2 - 10x
theorem f_equality (x : ℝ) : f (2 * x - 1) = 8 * x^2 - 10 * x := by sorry

-- Theorem 2: Prove the minimum value of t
theorem min_t_value : 
  ∃ t : ℝ, t = 2 * Real.exp 2 - 2 ∧ 
  ∀ x ∈ Set.Icc (-2) 2, f (Real.exp x) ≤ t * Real.exp x - 3 + Real.exp 2 ∧
  ∀ s : ℝ, (∀ x ∈ Set.Icc (-2) 2, f (Real.exp x) ≤ s * Real.exp x - 3 + Real.exp 2) → s ≥ t := by sorry

-- Theorem 3: Prove the range of N-n
theorem range_N_minus_n (a : ℝ) (h : 0 < a ∧ a < 3) :
  let N := max (g a 0) (g a 1)
  let n := min (g a 0) (g a 1)
  ∃ d : ℝ, d = N - n ∧ 8/27 ≤ d ∧ d < 2 := by sorry

end f_equality_min_t_value_range_N_minus_n_l3704_370424


namespace distance_from_origin_l3704_370474

theorem distance_from_origin (P : ℝ × ℝ) (h : P = (5, 12)) : 
  Real.sqrt ((P.1)^2 + (P.2)^2) = 13 := by
  sorry

end distance_from_origin_l3704_370474


namespace clara_quarters_problem_l3704_370479

theorem clara_quarters_problem : ∃! q : ℕ, 
  8 < q ∧ q < 80 ∧ 
  q % 3 = 1 ∧ 
  q % 4 = 1 ∧ 
  q % 5 = 1 ∧ 
  q = 61 := by
sorry

end clara_quarters_problem_l3704_370479


namespace determine_q_investment_l3704_370499

/-- Represents the investment and profit sharing of two business partners -/
structure BusinessPartnership where
  p_investment : ℕ
  q_investment : ℕ
  profit_ratio : Rat

/-- Theorem stating that given P's investment and the profit ratio, Q's investment can be determined -/
theorem determine_q_investment (bp : BusinessPartnership) 
  (h1 : bp.p_investment = 75000)
  (h2 : bp.profit_ratio = 5 / 1) :
  bp.q_investment = 15000 := by
  sorry

#check determine_q_investment

end determine_q_investment_l3704_370499


namespace remaining_food_feeds_children_l3704_370455

/-- Represents the amount of food required for one adult. -/
def adult_meal : ℚ := 1

/-- Represents the amount of food required for one child. -/
def child_meal : ℚ := 7/9

/-- Represents the total amount of food available. -/
def total_food : ℚ := 70 * adult_meal

/-- Theorem stating that if 35 adults have their meal, the remaining food can feed 45 children. -/
theorem remaining_food_feeds_children : 
  total_food - 35 * adult_meal = 45 * child_meal := by
  sorry

#check remaining_food_feeds_children

end remaining_food_feeds_children_l3704_370455


namespace intercept_length_min_distance_l3704_370430

-- Define the family of curves C
def C (m : ℝ) (x y : ℝ) : Prop :=
  4 * x^2 + 5 * y^2 - 8 * m * x - 20 * m * y + 24 * m^2 - 20 = 0

-- Define the circle
def Circle (x y : ℝ) : Prop :=
  x^2 + y^2 - 10 * x + 4 * Real.sqrt 6 * y + 30 = 0

-- Define the lines
def Line1 (x y : ℝ) : Prop := y = 2 * x + 2
def Line2 (x y : ℝ) : Prop := y = 2 * x - 2

-- Theorem for part 1
theorem intercept_length (m : ℝ) :
  ∀ x y, C m x y → (Line1 x y ∨ Line2 x y) →
  ∃ x1 y1 x2 y2, C m x1 y1 ∧ C m x2 y2 ∧
  ((Line1 x1 y1 ∧ Line1 x2 y2) ∨ (Line2 x1 y1 ∧ Line2 x2 y2)) ∧
  Real.sqrt ((x1 - x2)^2 + (y1 - y2)^2) = 5 * Real.sqrt 5 / 3 :=
sorry

-- Theorem for part 2
theorem min_distance :
  ∀ m x1 y1 x2 y2, C m x1 y1 → Circle x2 y2 →
  Real.sqrt ((x1 - x2)^2 + (y1 - y2)^2) ≥ 2 * Real.sqrt 5 - 1 :=
sorry

end intercept_length_min_distance_l3704_370430


namespace min_operations_to_250_l3704_370409

/-- Represents the possible operations: adding 1 or multiplying by 2 -/
inductive Operation
  | addOne
  | multiplyTwo

/-- Applies an operation to a natural number -/
def applyOperation (n : ℕ) (op : Operation) : ℕ :=
  match op with
  | Operation.addOne => n + 1
  | Operation.multiplyTwo => n * 2

/-- Checks if a sequence of operations transforms 1 into the target -/
def isValidSequence (target : ℕ) (ops : List Operation) : Prop :=
  ops.foldl applyOperation 1 = target

/-- The minimum number of operations needed to transform 1 into 250 -/
def minOperations : ℕ := 12

/-- Theorem stating that the minimum number of operations to reach 250 from 1 is 12 -/
theorem min_operations_to_250 :
  (∃ (ops : List Operation), isValidSequence 250 ops ∧ ops.length = minOperations) ∧
  (∀ (ops : List Operation), isValidSequence 250 ops → ops.length ≥ minOperations) :=
sorry

end min_operations_to_250_l3704_370409


namespace harmonious_triplet_from_intersections_l3704_370490

/-- Definition of a harmonious triplet -/
def is_harmonious_triplet (x y z : ℝ) : Prop :=
  x ≠ 0 ∧ y ≠ 0 ∧ z ≠ 0 ∧
  (1/x = 1/y + 1/z ∨ 1/y = 1/x + 1/z ∨ 1/z = 1/x + 1/y)

/-- Theorem about harmonious triplets formed by intersections -/
theorem harmonious_triplet_from_intersections
  (a b c : ℝ) (h₁ : a ≠ 0) (h₂ : b ≠ 0) (h₃ : c ≠ 0) :
  let x₁ := -c / b
  let x₂ := (-b + Real.sqrt (b^2 - 4*a*c)) / (2*a)
  let x₃ := (-b - Real.sqrt (b^2 - 4*a*c)) / (2*a)
  is_harmonious_triplet x₁ x₂ x₃ :=
by sorry

end harmonious_triplet_from_intersections_l3704_370490


namespace two_digit_number_equals_three_times_square_of_units_digit_l3704_370481

theorem two_digit_number_equals_three_times_square_of_units_digit :
  ∀ n : ℕ, 10 ≤ n ∧ n < 100 →
    (n = 3 * (n % 10)^2) ↔ (n = 12 ∨ n = 75) := by
  sorry

end two_digit_number_equals_three_times_square_of_units_digit_l3704_370481


namespace quadratic_equation_solution_l3704_370475

theorem quadratic_equation_solution :
  let f : ℝ → ℝ := fun x ↦ (x - 1)^2 - 4
  ∃ x₁ x₂ : ℝ, x₁ = 3 ∧ x₂ = -1 ∧ f x₁ = 0 ∧ f x₂ = 0 ∧
    ∀ x : ℝ, f x = 0 → x = x₁ ∨ x = x₂ :=
by
  sorry

end quadratic_equation_solution_l3704_370475


namespace standard_deviation_of_data_set_l3704_370465

def data_set : List ℝ := [10, 5, 4, 2, 2, 1]

theorem standard_deviation_of_data_set :
  let x := data_set[2]
  ∀ (mode median : ℝ),
    x ≠ 5 →
    mode = 2 →
    median = (x + 2) / 2 →
    mode = 2/3 * median →
    let mean := (data_set.sum) / (data_set.length : ℝ)
    let variance := (data_set.map (λ y => (y - mean)^2)).sum / (data_set.length : ℝ)
    Real.sqrt variance = 3 := by
  sorry

end standard_deviation_of_data_set_l3704_370465
