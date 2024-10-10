import Mathlib

namespace algae_free_day_l3953_395309

/-- Represents the coverage of algae on the pond for a given day -/
def algaeCoverage (day : ℕ) : ℚ :=
  1 / 2^(30 - day)

/-- The day when the pond is 75% algae-free -/
def targetDay : ℕ := 28

theorem algae_free_day :
  (algaeCoverage targetDay = 1/4) ∧ 
  (∀ d : ℕ, d < targetDay → algaeCoverage d < 1/4) ∧
  (∀ d : ℕ, d > targetDay → algaeCoverage d > 1/4) :=
by sorry

end algae_free_day_l3953_395309


namespace min_width_rectangle_width_satisfies_min_area_min_width_is_four_l3953_395331

/-- The minimum width of a rectangular area with specific constraints -/
theorem min_width_rectangle (w : ℝ) (h1 : w > 0) : 
  w * (w + 20) ≥ 120 → w ≥ 4 := by sorry

/-- The width that satisfies the minimum area requirement -/
theorem width_satisfies_min_area : 
  4 * (4 + 20) ≥ 120 := by sorry

/-- Proof that 4 is the minimum width satisfying the constraints -/
theorem min_width_is_four : 
  ∃ (w : ℝ), w > 0 ∧ w * (w + 20) ≥ 120 ∧ ∀ (x : ℝ), x > 0 ∧ x * (x + 20) ≥ 120 → x ≥ w :=
by
  use 4
  sorry

end min_width_rectangle_width_satisfies_min_area_min_width_is_four_l3953_395331


namespace train_average_speed_l3953_395370

/-- Given a train's travel data, prove its average speed -/
theorem train_average_speed : 
  ∀ (d1 d2 t1 t2 : ℝ),
  d1 = 290 ∧ d2 = 400 ∧ t1 = 4.5 ∧ t2 = 5.5 →
  (d1 + d2) / (t1 + t2) = 69 := by
sorry

end train_average_speed_l3953_395370


namespace chessboard_coverage_l3953_395327

/-- Represents a chessboard -/
structure Chessboard :=
  (rows : Nat)
  (cols : Nat)

/-- Represents a domino -/
structure Domino :=
  (length : Nat)
  (width : Nat)

/-- Checks if a chessboard can be covered by dominoes -/
def can_cover (board : Chessboard) (domino : Domino) : Prop :=
  sorry

/-- Checks if a chessboard with one corner removed can be covered by dominoes -/
def can_cover_one_corner_removed (board : Chessboard) (domino : Domino) : Prop :=
  sorry

/-- Checks if a chessboard with two opposite corners removed can be covered by dominoes -/
def can_cover_two_corners_removed (board : Chessboard) (domino : Domino) : Prop :=
  sorry

theorem chessboard_coverage (board : Chessboard) (domino : Domino) :
  board.rows = 8 ∧ board.cols = 8 ∧ domino.length = 2 ∧ domino.width = 1 →
  (can_cover board domino) ∧
  ¬(can_cover_one_corner_removed board domino) ∧
  ¬(can_cover_two_corners_removed board domino) :=
sorry

end chessboard_coverage_l3953_395327


namespace rogers_pennies_l3953_395367

/-- The number of pennies Roger collected initially -/
def pennies_collected : ℕ := sorry

/-- The number of nickels Roger collected -/
def nickels : ℕ := 36

/-- The number of dimes Roger collected -/
def dimes : ℕ := 15

/-- The number of coins Roger donated -/
def coins_donated : ℕ := 66

/-- The number of coins Roger had left after donating -/
def coins_left : ℕ := 27

/-- The total number of coins Roger had initially -/
def total_coins : ℕ := coins_donated + coins_left

theorem rogers_pennies :
  pennies_collected = total_coins - (nickels + dimes) :=
by sorry

end rogers_pennies_l3953_395367


namespace cost_of_apples_and_bananas_l3953_395350

/-- The cost of apples in dollars per pound -/
def apple_cost : ℚ := 3 / 3

/-- The cost of bananas in dollars per pound -/
def banana_cost : ℚ := 2 / 2

/-- The total cost of apples and bananas -/
def total_cost (apple_pounds banana_pounds : ℚ) : ℚ :=
  apple_pounds * apple_cost + banana_pounds * banana_cost

theorem cost_of_apples_and_bananas :
  total_cost 9 6 = 15 := by sorry

end cost_of_apples_and_bananas_l3953_395350


namespace adjacent_vertices_probability_l3953_395379

/-- A decagon is a polygon with 10 vertices -/
def Decagon : Type := Unit

/-- The number of vertices in a decagon -/
def num_vertices : ℕ := 10

/-- The number of adjacent vertices for each vertex in a decagon -/
def adjacent_vertices : ℕ := 2

/-- The probability of choosing two adjacent vertices when randomly selecting 2 distinct vertices from a decagon -/
theorem adjacent_vertices_probability (d : Decagon) : ℚ :=
  2 / 9

/-- Proof of the theorem -/
lemma adjacent_vertices_probability_proof (d : Decagon) : 
  adjacent_vertices_probability d = 2 / 9 := by
  sorry

end adjacent_vertices_probability_l3953_395379


namespace irrational_sum_equivalence_l3953_395355

theorem irrational_sum_equivalence 
  (a b c d : ℝ) 
  (ha : Irrational a) 
  (hb : Irrational b) 
  (hc : Irrational c) 
  (hd : Irrational d) 
  (hab : a + b = 1) :
  (c + d = 1) ↔ 
  (∀ n : ℕ+, ⌊n * a⌋ + ⌊n * b⌋ = ⌊n * c⌋ + ⌊n * d⌋) :=
by sorry

end irrational_sum_equivalence_l3953_395355


namespace sin_product_18_54_72_36_l3953_395398

theorem sin_product_18_54_72_36 :
  Real.sin (18 * π / 180) * Real.sin (54 * π / 180) *
  Real.sin (72 * π / 180) * Real.sin (36 * π / 180) =
  (Real.sqrt 5 + 1) / 16 := by sorry

end sin_product_18_54_72_36_l3953_395398


namespace ackermann_3_1_l3953_395380

def B : ℕ → ℕ → ℕ
  | 0, n => n + 1
  | m + 1, 0 => B m 1
  | m + 1, n + 1 => B m (B (m + 1) n)

theorem ackermann_3_1 : B 3 1 = 13 := by
  sorry

end ackermann_3_1_l3953_395380


namespace value_of_3x_plus_y_l3953_395365

theorem value_of_3x_plus_y (x y : ℝ) (h : (2*x + y)^3 + x^3 + 3*x + y = 0) : 3*x + y = 0 := by
  sorry

end value_of_3x_plus_y_l3953_395365


namespace salt_solution_volume_l3953_395385

/-- Proves that given a solution with an initial salt concentration of 10%,
    if adding 18 gallons of water reduces the salt concentration to 8%,
    then the initial volume of the solution must be 72 gallons. -/
theorem salt_solution_volume 
  (initial_concentration : ℝ) 
  (final_concentration : ℝ) 
  (water_added : ℝ) 
  (initial_volume : ℝ) :
  initial_concentration = 0.10 →
  final_concentration = 0.08 →
  water_added = 18 →
  initial_concentration * initial_volume = 
    final_concentration * (initial_volume + water_added) →
  initial_volume = 72 :=
by sorry

end salt_solution_volume_l3953_395385


namespace square_and_cube_roots_l3953_395314

theorem square_and_cube_roots :
  (∀ x : ℝ, x ^ 2 = 36 → x = 6 ∨ x = -6) ∧
  (Real.sqrt 16 = 4) ∧
  (∃ x : ℝ, x ^ 2 = 4 ∧ x > 0 ∧ x = 2) ∧
  (∃ x : ℝ, x ^ 3 = -27 ∧ x = -3) := by
  sorry

end square_and_cube_roots_l3953_395314


namespace sqrt_difference_abs_plus_two_sqrt_two_l3953_395356

theorem sqrt_difference_abs_plus_two_sqrt_two :
  |Real.sqrt 2 - Real.sqrt 3| + 2 * Real.sqrt 2 = Real.sqrt 3 + Real.sqrt 2 := by
  sorry

end sqrt_difference_abs_plus_two_sqrt_two_l3953_395356


namespace vector_simplification_l3953_395377

/-- Given points A, B, C, and O in 3D space, 
    prove that AB + OC - OB = AC -/
theorem vector_simplification 
  (A B C O : EuclideanSpace ℝ (Fin 3)) : 
  (B - A) + (C - O) - (B - O) = C - A := by sorry

end vector_simplification_l3953_395377


namespace e_integral_greater_than_ln_integral_l3953_395381

theorem e_integral_greater_than_ln_integral : ∫ (x : ℝ) in (0)..(1), Real.exp x > ∫ (x : ℝ) in (1)..(Real.exp 1), 1 / x := by
  sorry

end e_integral_greater_than_ln_integral_l3953_395381


namespace fixed_point_on_line_l3953_395322

/-- For any real number m, the line mx + y - 1 + 2m = 0 passes through the point (-2, 1) -/
theorem fixed_point_on_line (m : ℝ) : m * (-2) + 1 - 1 + 2 * m = 0 := by
  sorry

end fixed_point_on_line_l3953_395322


namespace infinitely_many_solutions_l3953_395348

theorem infinitely_many_solutions
  (a b c k : ℤ)
  (D : ℤ)
  (hD : D = b^2 - 4*a*c)
  (hD_pos : D > 0)
  (hD_nonsquare : ∀ m : ℤ, D ≠ m^2)
  (hk : k ≠ 0)
  (h_solution : ∃ (x₀ y₀ : ℤ), a*x₀^2 + b*x₀*y₀ + c*y₀^2 = k) :
  ∃ (S : Set (ℤ × ℤ)), (Set.Infinite S) ∧ (∀ (x y : ℤ), (x, y) ∈ S → a*x^2 + b*x*y + c*y^2 = k) :=
sorry

end infinitely_many_solutions_l3953_395348


namespace expression_equals_sum_l3953_395396

theorem expression_equals_sum (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  (a^2 * (1/b - 1/c) + b^2 * (1/c - 1/a) + c^2 * (1/a - 1/b)) / 
  (a * (1/b - 1/c) + b * (1/c - 1/a) + c * (1/a - 1/b)) = a + b + c := by
  sorry

end expression_equals_sum_l3953_395396


namespace smallest_d_for_inverse_l3953_395395

def g (x : ℝ) : ℝ := (x - 3)^2 - 8

theorem smallest_d_for_inverse :
  ∀ d : ℝ, (∀ x y, x ≥ d → y ≥ d → g x = g y → x = y) ↔ d ≥ 3 :=
by sorry

end smallest_d_for_inverse_l3953_395395


namespace annular_ring_area_l3953_395332

/-- Given a circle and a chord AB divided by point C such that AC = a and BC = b,
    the area of the annular ring formed when C traces another circle as AB's position changes
    is π(a + b)²/4. -/
theorem annular_ring_area (a b : ℝ) (h : a > 0 ∧ b > 0) :
  let chord_length := a + b
  ∃ (R : ℝ), R > chord_length / 2 →
    (π * (chord_length ^ 2) / 4 : ℝ) =
      π * R ^ 2 - π * (R ^ 2 - chord_length ^ 2 / 4) :=
by sorry

end annular_ring_area_l3953_395332


namespace james_field_goal_value_l3953_395369

/-- Represents the score of a basketball game -/
structure BasketballScore where
  fieldGoals : ℕ
  fieldGoalValue : ℕ
  twoPointers : ℕ
  totalScore : ℕ

/-- Theorem stating that given the conditions of James' game, each field goal is worth 3 points -/
theorem james_field_goal_value (score : BasketballScore) 
  (h1 : score.fieldGoals = 13)
  (h2 : score.twoPointers = 20)
  (h3 : score.totalScore = 79)
  (h4 : score.totalScore = score.fieldGoals * score.fieldGoalValue + score.twoPointers * 2) :
  score.fieldGoalValue = 3 := by
  sorry


end james_field_goal_value_l3953_395369


namespace bill_fraction_l3953_395378

theorem bill_fraction (total_stickers : ℕ) (andrew_fraction : ℚ) (total_given : ℕ) 
  (h1 : total_stickers = 100)
  (h2 : andrew_fraction = 1 / 5)
  (h3 : total_given = 44) :
  let andrew_stickers := andrew_fraction * total_stickers
  let remaining_after_andrew := total_stickers - andrew_stickers
  let bill_stickers := total_given - andrew_stickers
  bill_stickers / remaining_after_andrew = 3 / 10 := by
  sorry

end bill_fraction_l3953_395378


namespace inequality_solution_l3953_395366

theorem inequality_solution (x : ℝ) : 
  (x^2 - 3*x + 3)^(4*x^3 + 5*x^2) ≤ (x^2 - 3*x + 3)^(2*x^3 + 18*x) ↔ 
  x ≤ -9/2 ∨ (0 ≤ x ∧ x ≤ 1) ∨ x = 2 := by
sorry

end inequality_solution_l3953_395366


namespace diplomats_not_speaking_russian_count_l3953_395339

/-- Represents the diplomats at a summit conference -/
structure DiplomatGroup where
  total : Nat
  latin_speakers : Nat
  neither_latin_nor_russian : Nat
  both_latin_and_russian : Nat

/-- Calculates the number of diplomats who did not speak Russian -/
def diplomats_not_speaking_russian (d : DiplomatGroup) : Nat :=
  d.total - (d.total - d.neither_latin_nor_russian - d.latin_speakers + d.both_latin_and_russian)

/-- Theorem stating the number of diplomats who did not speak Russian -/
theorem diplomats_not_speaking_russian_count :
  ∃ (d : DiplomatGroup),
    d.total = 120 ∧
    d.latin_speakers = 20 ∧
    d.neither_latin_nor_russian = (20 * d.total) / 100 ∧
    d.both_latin_and_russian = (10 * d.total) / 100 ∧
    diplomats_not_speaking_russian d = 20 := by
  sorry

end diplomats_not_speaking_russian_count_l3953_395339


namespace imaginary_part_of_z_l3953_395319

theorem imaginary_part_of_z (z : ℂ) : z = 2 / (-1 + Complex.I) → z.im = -1 := by
  sorry

end imaginary_part_of_z_l3953_395319


namespace tom_car_washing_earnings_l3953_395318

/-- The amount of money Tom had last week -/
def initial_amount : ℕ := 74

/-- The amount of money Tom has now -/
def current_amount : ℕ := 86

/-- The amount of money Tom made washing cars -/
def money_made : ℕ := current_amount - initial_amount

theorem tom_car_washing_earnings : 
  money_made = current_amount - initial_amount :=
by sorry

end tom_car_washing_earnings_l3953_395318


namespace pond_depth_l3953_395360

theorem pond_depth (d : ℝ) 
  (h1 : ¬(d ≥ 10))  -- Adam's statement is false
  (h2 : ¬(d ≤ 8))   -- Ben's statement is false
  (h3 : d ≠ 7)      -- Carla's statement is false
  : 8 < d ∧ d < 10 := by
  sorry

end pond_depth_l3953_395360


namespace y_coordinate_of_P_l3953_395343

/-- The y-coordinate of a point P satisfying certain conditions -/
theorem y_coordinate_of_P (A B C D P : ℝ × ℝ) : 
  A = (-4, 0) →
  B = (-1, 2) →
  C = (1, 2) →
  D = (4, 0) →
  dist P A + dist P D = 10 →
  dist P B + dist P C = 10 →
  P.2 = (-12 + 16 * Real.sqrt 16.5) / 5 :=
by sorry

end y_coordinate_of_P_l3953_395343


namespace unique_n_value_l3953_395371

theorem unique_n_value : ∃! n : ℕ, 
  50 ≤ n ∧ n ≤ 120 ∧ 
  ∃ k : ℕ, n = 8 * k ∧
  n % 7 = 5 ∧
  n % 6 = 3 ∧
  n = 208 := by
sorry

end unique_n_value_l3953_395371


namespace unique_prime_sevens_l3953_395323

def A (n : ℕ+) : ℕ := 1 + 7 * (10^n.val - 1) / 9

def B (n : ℕ+) : ℕ := 3 + 7 * (10^n.val - 1) / 9

theorem unique_prime_sevens : 
  ∃! (n : ℕ+), Nat.Prime (A n) ∧ Nat.Prime (B n) :=
sorry

end unique_prime_sevens_l3953_395323


namespace one_refilling_cost_l3953_395384

/-- Given that Greyson spent 40 dollars on fuel this week and refilled 4 times,
    prove that the cost of one refilling is 10 dollars. -/
theorem one_refilling_cost (total_spent : ℝ) (num_refills : ℕ) 
  (h1 : total_spent = 40)
  (h2 : num_refills = 4) :
  total_spent / num_refills = 10 := by
  sorry

end one_refilling_cost_l3953_395384


namespace inverse_function_problem_l3953_395345

theorem inverse_function_problem (f : ℝ → ℝ) (f_inv : ℝ → ℝ) :
  (∀ x, f_inv x = 2^(x + 1)) → f 1 = -1 := by
  sorry

end inverse_function_problem_l3953_395345


namespace daniels_candies_l3953_395386

theorem daniels_candies (x : ℕ) : 
  (x : ℚ) * 3/8 - 3/2 - 6 = 10 ↔ x = 93 := by
  sorry

end daniels_candies_l3953_395386


namespace point_A_coordinates_l3953_395324

theorem point_A_coordinates :
  ∀ a : ℤ,
  (a + 1 < 0) →
  (2 * a + 6 > 0) →
  (a + 1, 2 * a + 6) = (-1, 2) :=
by
  sorry

end point_A_coordinates_l3953_395324


namespace exists_more_kites_than_points_l3953_395316

/-- A point on a grid --/
structure GridPoint where
  x : ℤ
  y : ℤ

/-- A kite shape formed by four points --/
structure Kite where
  p1 : GridPoint
  p2 : GridPoint
  p3 : GridPoint
  p4 : GridPoint

/-- A configuration of points on a grid --/
structure GridConfiguration where
  points : List GridPoint
  kites : List Kite

/-- Function to count the number of kites in a configuration --/
def countKites (config : GridConfiguration) : ℕ :=
  config.kites.length

/-- Function to count the number of points in a configuration --/
def countPoints (config : GridConfiguration) : ℕ :=
  config.points.length

/-- Theorem stating that there exists a configuration with more kites than points --/
theorem exists_more_kites_than_points :
  ∃ (config : GridConfiguration), countKites config > countPoints config := by
  sorry

end exists_more_kites_than_points_l3953_395316


namespace right_triangle_segment_ratio_l3953_395337

theorem right_triangle_segment_ratio :
  ∀ (a b c r s : ℝ),
  a > 0 → b > 0 → c > 0 → r > 0 → s > 0 →
  a^2 + b^2 = c^2 →  -- Right triangle condition
  a / b = 2 / 5 →    -- Given ratio of sides
  r + s = c →        -- Perpendicular divides hypotenuse
  r * c = a^2 →      -- Geometric mean theorem for r
  s * c = b^2 →      -- Geometric mean theorem for s
  r / s = 4 / 25 := by
sorry

end right_triangle_segment_ratio_l3953_395337


namespace no_valid_coloring_l3953_395363

-- Define a color type
inductive Color
| Blue
| Red
| Green

-- Define a coloring function type
def Coloring := Nat → Color

-- Define the property that all three colors are used
def AllColorsUsed (f : Coloring) : Prop :=
  ∃ (a b c : Nat), a > 1 ∧ b > 1 ∧ c > 1 ∧ 
    f a = Color.Blue ∧ f b = Color.Red ∧ f c = Color.Green

-- Define the property that the product of two differently colored numbers
-- has a different color from both multipliers
def ValidColoring (f : Coloring) : Prop :=
  ∀ (a b : Nat), a > 1 → b > 1 → f a ≠ f b →
    f (a * b) ≠ f a ∧ f (a * b) ≠ f b

-- State the theorem
theorem no_valid_coloring :
  ¬∃ (f : Coloring), AllColorsUsed f ∧ ValidColoring f :=
sorry

end no_valid_coloring_l3953_395363


namespace sum_of_three_polynomials_no_roots_l3953_395362

/-- Given three quadratic polynomials, if the sum of any two has no roots, 
    then the sum of all three has no roots. -/
theorem sum_of_three_polynomials_no_roots 
  (a b c d e f : ℝ) 
  (h1 : ∀ x, (2*x^2 + (a + c)*x + (b + d)) ≠ 0)
  (h2 : ∀ x, (2*x^2 + (c + e)*x + (d + f)) ≠ 0)
  (h3 : ∀ x, (2*x^2 + (e + a)*x + (f + b)) ≠ 0) :
  ∀ x, (3*x^2 + (a + c + e)*x + (b + d + f)) ≠ 0 :=
by sorry

end sum_of_three_polynomials_no_roots_l3953_395362


namespace min_year_exceed_300k_l3953_395306

/-- Represents the linear regression equation for online shoppers --/
def online_shoppers (x : ℤ) : ℝ := 42 * x - 26

/-- Theorem: The minimum integer value of x for which the number of online shoppers exceeds 300 thousand is 8 --/
theorem min_year_exceed_300k :
  ∀ x : ℤ, (x ≥ 8 ↔ online_shoppers x > 300) ∧
  ∀ y : ℤ, y < 8 → online_shoppers y ≤ 300 :=
sorry


end min_year_exceed_300k_l3953_395306


namespace range_of_k_l3953_395333

/-- Given x ∈ (0, 2), prove that x/(e^x) < 1/(k + 2x - x^2) holds if and only if k ∈ [0, e-1) -/
theorem range_of_k (x : ℝ) (hx : x ∈ Set.Ioo 0 2) :
  (∀ k : ℝ, x / Real.exp x < 1 / (k + 2 * x - x^2)) ↔ k ∈ Set.Icc 0 (Real.exp 1 - 1) :=
sorry

end range_of_k_l3953_395333


namespace exam_mean_score_l3953_395387

/-- Given an exam where a score of 42 is 5 standard deviations below the mean
    and a score of 67 is 2.5 standard deviations above the mean,
    prove that the mean score is 440/7.5 -/
theorem exam_mean_score (μ σ : ℝ) 
  (h1 : 42 = μ - 5 * σ)
  (h2 : 67 = μ + 2.5 * σ) : 
  μ = 440 / 7.5 := by
  sorry

end exam_mean_score_l3953_395387


namespace inequality_implication_l3953_395302

theorem inequality_implication (a b : ℝ) (h : a > b) : -5 * a < -5 * b := by
  sorry

end inequality_implication_l3953_395302


namespace trig_inequality_l3953_395349

theorem trig_inequality (a b : Real) (ha : 0 < a ∧ a < π/2) (hb : 0 < b ∧ b < π/2) :
  (Real.sin a)^3 / Real.sin b + (Real.cos a)^3 / Real.cos b ≥ 1 / Real.cos (a - b) := by
  sorry

end trig_inequality_l3953_395349


namespace chicken_feathers_after_crossing_l3953_395330

/-- Represents the number of feathers a chicken has after crossing a road twice -/
def feathers_after_crossing (initial_feathers : ℕ) (cars_dodged : ℕ) : ℕ :=
  initial_feathers - 2 * cars_dodged

/-- Theorem stating the number of feathers remaining after the chicken's adventure -/
theorem chicken_feathers_after_crossing :
  feathers_after_crossing 5263 23 = 5217 := by
  sorry

#eval feathers_after_crossing 5263 23

end chicken_feathers_after_crossing_l3953_395330


namespace gcd_triple_characterization_l3953_395391

theorem gcd_triple_characterization (a b c : ℕ+) :
  (Nat.gcd a 20 = b) →
  (Nat.gcd b 15 = c) →
  (Nat.gcd a c = 5) →
  (∃ t : ℕ+, (a = 20 * t ∧ b = 20 ∧ c = 5) ∨
             (a = 20 * t - 10 ∧ b = 10 ∧ c = 5) ∨
             (a = 10 * t - 5 ∧ b = 5 ∧ c = 5)) :=
by sorry

end gcd_triple_characterization_l3953_395391


namespace set_intersection_equality_l3953_395353

def A : Set ℝ := {x | 0 < x ∧ x ≤ 2}
def B : Set ℝ := {x | x ≤ 3}

theorem set_intersection_equality : A ∩ B = A := by sorry

end set_intersection_equality_l3953_395353


namespace sum_of_cubes_l3953_395338

theorem sum_of_cubes (a b c : ℝ) 
  (sum_eq : a + b + c = 7)
  (sum_products_eq : a * b + a * c + b * c = 11)
  (product_eq : a * b * c = -18) :
  a^3 + b^3 + c^3 = 151 := by
  sorry

end sum_of_cubes_l3953_395338


namespace beka_miles_l3953_395352

/-- The number of miles Jackson flew -/
def jackson_miles : ℕ := 563

/-- The difference in miles between Beka's and Jackson's flights -/
def difference_miles : ℕ := 310

/-- Theorem: Beka flew 873 miles -/
theorem beka_miles : jackson_miles + difference_miles = 873 := by
  sorry

end beka_miles_l3953_395352


namespace raghu_investment_l3953_395310

theorem raghu_investment (raghu trishul vishal : ℝ) : 
  trishul = 0.9 * raghu →
  vishal = 1.1 * trishul →
  raghu + trishul + vishal = 6936 →
  raghu = 2400 := by
sorry

end raghu_investment_l3953_395310


namespace multiply_fractions_l3953_395390

theorem multiply_fractions : 8 * (1 / 11) * 33 = 24 := by sorry

end multiply_fractions_l3953_395390


namespace larger_integer_value_l3953_395364

theorem larger_integer_value (a b : ℕ+) 
  (h1 : (a : ℚ) / (b : ℚ) = 7 / 3) 
  (h2 : (a : ℕ) * b = 189) : 
  a = 21 := by
sorry

end larger_integer_value_l3953_395364


namespace solution_equation1_solution_equation2_l3953_395335

-- Define the quadratic equations
def equation1 (x : ℝ) : Prop := x^2 + 2*x - 3 = 0
def equation2 (x : ℝ) : Prop := 2*x^2 + 4*x - 3 = 0

-- Theorem for the first equation
theorem solution_equation1 : 
  (∃ x₁ x₂ : ℝ, x₁ = 1 ∧ x₂ = -3 ∧ equation1 x₁ ∧ equation1 x₂) ∧
  (∀ x : ℝ, equation1 x → x = 1 ∨ x = -3) :=
sorry

-- Theorem for the second equation
theorem solution_equation2 : 
  (∃ x₁ x₂ : ℝ, x₁ = (-2 + Real.sqrt 10) / 2 ∧ x₂ = (-2 - Real.sqrt 10) / 2 ∧ equation2 x₁ ∧ equation2 x₂) ∧
  (∀ x : ℝ, equation2 x → x = (-2 + Real.sqrt 10) / 2 ∨ x = (-2 - Real.sqrt 10) / 2) :=
sorry

end solution_equation1_solution_equation2_l3953_395335


namespace compound_interest_equation_l3953_395303

/-- The initial sum of money lent out -/
def P : ℝ := sorry

/-- The final amount after 2 years -/
def final_amount : ℝ := 341

/-- The semi-annual interest rate for the first year -/
def r1 : ℝ := 0.025

/-- The semi-annual interest rate for the second year -/
def r2 : ℝ := 0.03

/-- The number of compounding periods per year -/
def n : ℕ := 2

/-- The total number of compounding periods -/
def total_periods : ℕ := 4

theorem compound_interest_equation :
  P * (1 + r1)^n * (1 + r2)^n = final_amount := by sorry

end compound_interest_equation_l3953_395303


namespace line_segment_endpoint_l3953_395376

-- Define the start point of the line segment
def start_point : ℝ × ℝ := (1, 3)

-- Define the end point of the line segment
def end_point (x : ℝ) : ℝ × ℝ := (x, -4)

-- Define the length of the line segment
def segment_length : ℝ := 15

-- Theorem statement
theorem line_segment_endpoint (x : ℝ) : 
  x < 0 → 
  (end_point x).1 - (start_point.1) = -4 * Real.sqrt 11 :=
by sorry

end line_segment_endpoint_l3953_395376


namespace unique_seven_numbers_sum_100_l3953_395383

theorem unique_seven_numbers_sum_100 (a₄ : ℕ) : 
  ∃! (a₁ a₂ a₃ a₅ a₆ a₇ : ℕ), 
    a₁ < a₂ ∧ a₂ < a₃ ∧ a₃ < a₄ ∧ a₄ < a₅ ∧ a₅ < a₆ ∧ a₆ < a₇ ∧
    a₁ + a₂ + a₃ + a₄ + a₅ + a₆ + a₇ = 100 :=
by sorry

end unique_seven_numbers_sum_100_l3953_395383


namespace divisibility_of_f_l3953_395336

def f (x : ℕ) : ℕ := x^3 + 17

theorem divisibility_of_f :
  ∀ n : ℕ, n ≥ 2 →
  ∃ x : ℕ, (3^n ∣ f x) ∧ ¬(3^(n+1) ∣ f x) := by
sorry

end divisibility_of_f_l3953_395336


namespace arithmetic_sequence_theorem_l3953_395399

/-- An arithmetic sequence with sum of first n terms Sn = n^2 + bn + c -/
structure ArithmeticSequence where
  b : ℝ
  c : ℝ
  sum : ℕ+ → ℝ
  sum_eq : ∀ n : ℕ+, sum n = n.val ^ 2 + b * n.val + c

/-- The second term of the arithmetic sequence -/
def ArithmeticSequence.a2 (seq : ArithmeticSequence) : ℝ :=
  seq.sum 2 - seq.sum 1

/-- The third term of the arithmetic sequence -/
def ArithmeticSequence.a3 (seq : ArithmeticSequence) : ℝ :=
  seq.sum 3 - seq.sum 2

theorem arithmetic_sequence_theorem (seq : ArithmeticSequence) 
  (h : seq.a2 + seq.a3 = 4) : 
  seq.c = 0 ∧ seq.b = -2 := by
  sorry

end arithmetic_sequence_theorem_l3953_395399


namespace three_digit_combinations_l3953_395359

def set1 : Finset Nat := {0, 2, 4}
def set2 : Finset Nat := {1, 3, 5}

theorem three_digit_combinations : 
  (Finset.card set1) * (Finset.card set2) * (Finset.card set2 - 1) = 48 := by
  sorry

end three_digit_combinations_l3953_395359


namespace rational_function_equation_l3953_395341

theorem rational_function_equation (f : ℚ → ℚ) :
  (∀ x y : ℚ, f (x + f y) = f x + y) →
  (∀ x : ℚ, f x = x ∨ f x = -x) := by
sorry

end rational_function_equation_l3953_395341


namespace unique_congruence_in_range_l3953_395304

theorem unique_congruence_in_range : ∃! n : ℕ, 0 ≤ n ∧ n ≤ 10 ∧ n ≡ 123456 [ZMOD 11] ∧ n = 3 := by
  sorry

end unique_congruence_in_range_l3953_395304


namespace female_half_marathon_count_half_marathon_probability_no_significant_relation_l3953_395357

/-- Represents the number of students in each category --/
structure StudentCounts where
  male_half_marathon : ℕ
  male_mini_run : ℕ
  female_half_marathon : ℕ
  female_mini_run : ℕ

/-- The given student counts --/
def given_counts : StudentCounts := {
  male_half_marathon := 20,
  male_mini_run := 10,
  female_half_marathon := 10,  -- This is 'a', which we'll prove
  female_mini_run := 10
}

/-- The ratio of male to female students --/
def male_female_ratio : ℚ := 3 / 2

/-- Theorem stating the correct number of female students in half marathon --/
theorem female_half_marathon_count :
  given_counts.female_half_marathon = 10 := by sorry

/-- Theorem stating the probability of choosing half marathon --/
theorem half_marathon_probability :
  (given_counts.male_half_marathon + given_counts.female_half_marathon : ℚ) /
  (given_counts.male_half_marathon + given_counts.male_mini_run +
   given_counts.female_half_marathon + given_counts.female_mini_run) = 3 / 5 := by sorry

/-- Chi-square statistic calculation --/
def chi_square (c : StudentCounts) : ℚ :=
  let n := c.male_half_marathon + c.male_mini_run + c.female_half_marathon + c.female_mini_run
  let ad := c.male_half_marathon * c.female_mini_run
  let bc := c.male_mini_run * c.female_half_marathon
  n * (ad - bc)^2 / ((c.male_half_marathon + c.male_mini_run) *
                     (c.female_half_marathon + c.female_mini_run) *
                     (c.male_half_marathon + c.female_half_marathon) *
                     (c.male_mini_run + c.female_mini_run))

/-- Theorem stating that the chi-square statistic is less than the critical value --/
theorem no_significant_relation :
  chi_square given_counts < 2706 / 1000 := by sorry

end female_half_marathon_count_half_marathon_probability_no_significant_relation_l3953_395357


namespace triangle_gp_length_l3953_395340

-- Define the triangle DEF
structure Triangle :=
  (DE DF EF : ℝ)

-- Define the centroid G and point P
structure TrianglePoints (t : Triangle) :=
  (G P : ℝ × ℝ)

-- Define the length of GP
def lengthGP (t : Triangle) (tp : TrianglePoints t) : ℝ :=
  sorry

-- Theorem statement
theorem triangle_gp_length (t : Triangle) (tp : TrianglePoints t) 
  (h1 : t.DE = 10) (h2 : t.DF = 15) (h3 : t.EF = 17) : 
  lengthGP t tp = 4 * Real.sqrt 154 / 17 := by
  sorry

end triangle_gp_length_l3953_395340


namespace train_distance_theorem_l3953_395317

/-- The distance between two stations given the conditions of two trains meeting --/
theorem train_distance_theorem (v₁ v₂ : ℝ) (d : ℝ) :
  v₁ > 0 → v₂ > 0 →
  v₁ = 20 →
  v₂ = 25 →
  d = 75 →
  (∃ (t : ℝ), t > 0 ∧ v₁ * t + (v₂ * t - d) = v₁ * t + v₂ * t) →
  v₁ * t + v₂ * t = 675 :=
by sorry

end train_distance_theorem_l3953_395317


namespace child_tickets_sold_l3953_395397

theorem child_tickets_sold (adult_price child_price total_tickets total_receipts : ℕ) 
  (h1 : adult_price = 12)
  (h2 : child_price = 4)
  (h3 : total_tickets = 130)
  (h4 : total_receipts = 840) :
  ∃ (adult_tickets child_tickets : ℕ),
    adult_tickets + child_tickets = total_tickets ∧
    adult_price * adult_tickets + child_price * child_tickets = total_receipts ∧
    child_tickets = 90 := by
  sorry

end child_tickets_sold_l3953_395397


namespace min_value_theorem_min_value_equality_l3953_395358

/-- The minimum value of 1/m + 3/n given the conditions -/
theorem min_value_theorem (a m n : ℝ) (ha : a > 0) (ha_neq : a ≠ 1) 
  (hmn : m * n > 0) (h_line : m * 2 + n * 2 = 1) : 
  (1 / m + 3 / n : ℝ) ≥ 5 + 2 * Real.sqrt 6 := by
  sorry

/-- The conditions for equality in the minimum value theorem -/
theorem min_value_equality (a m n : ℝ) (ha : a > 0) (ha_neq : a ≠ 1) 
  (hmn : m * n > 0) (h_line : m * 2 + n * 2 = 1) : 
  (1 / m + 3 / n : ℝ) = 5 + 2 * Real.sqrt 6 ↔ m = Real.sqrt 3 / 3 ∧ n = Real.sqrt 3 / 3 := by
  sorry

end min_value_theorem_min_value_equality_l3953_395358


namespace sum_of_max_min_S_l3953_395325

theorem sum_of_max_min_S (x y z : ℝ) (hx : x ≥ 0) (hy : y ≥ 0) (hz : z ≥ 0)
  (h1 : x + y = 10) (h2 : y + z = 8) : 
  let S := x + z
  ∃ (S_min S_max : ℝ), 
    (∀ s, (∃ x' z', x' ≥ 0 ∧ z' ≥ 0 ∧ ∃ y', y' ≥ 0 ∧ x' + y' = 10 ∧ y' + z' = 8 ∧ s = x' + z') → s ≥ S_min) ∧
    (∀ s, (∃ x' z', x' ≥ 0 ∧ z' ≥ 0 ∧ ∃ y', y' ≥ 0 ∧ x' + y' = 10 ∧ y' + z' = 8 ∧ s = x' + z') → s ≤ S_max) ∧
    S_min + S_max = 20 :=
by sorry

end sum_of_max_min_S_l3953_395325


namespace second_car_speed_theorem_l3953_395326

/-- Two cars traveling on perpendicular roads towards an intersection -/
structure TwoCars where
  s₁ : ℝ  -- Initial distance of first car from intersection
  s₂ : ℝ  -- Initial distance of second car from intersection
  v₁ : ℝ  -- Speed of first car
  s  : ℝ  -- Distance between cars when first car reaches intersection

/-- The speed of the second car in the TwoCars scenario -/
def second_car_speed (cars : TwoCars) : Set ℝ :=
  {v₂ | v₂ = 12 ∨ v₂ = 16}

/-- Theorem stating the possible speeds of the second car -/
theorem second_car_speed_theorem (cars : TwoCars) 
    (h₁ : cars.s₁ = 500)
    (h₂ : cars.s₂ = 700)
    (h₃ : cars.v₁ = 10)  -- 36 km/h converted to m/s
    (h₄ : cars.s = 100) :
  second_car_speed cars = {12, 16} := by
  sorry

end second_car_speed_theorem_l3953_395326


namespace work_to_pump_oil_horizontal_cylinder_l3953_395312

/-- Work required to pump oil from a horizontal cylindrical tank -/
theorem work_to_pump_oil_horizontal_cylinder 
  (δ : ℝ) -- specific weight of oil
  (H : ℝ) -- length of the cylinder
  (R : ℝ) -- radius of the cylinder
  (h : R > 0) -- assumption that radius is positive
  (h' : H > 0) -- assumption that length is positive
  (h'' : δ > 0) -- assumption that specific weight is positive
  : ∃ (Q : ℝ), Q = π * δ * H * R^3 :=
sorry

end work_to_pump_oil_horizontal_cylinder_l3953_395312


namespace difference_of_squares_fraction_l3953_395382

theorem difference_of_squares_fraction :
  (113^2 - 104^2) / 9 = 217 := by sorry

end difference_of_squares_fraction_l3953_395382


namespace square_rectangle_area_ratio_l3953_395368

theorem square_rectangle_area_ratio :
  ∀ (square_perimeter : ℝ) (rect_length rect_width : ℝ),
    square_perimeter = 256 →
    rect_length = 32 →
    rect_width = 64 →
    (square_perimeter / 4)^2 / (rect_length * rect_width) = 2 := by
  sorry

end square_rectangle_area_ratio_l3953_395368


namespace complex_collinearity_l3953_395321

/-- A complex number represented as a point in the plane -/
structure ComplexPoint where
  re : ℝ
  im : ℝ

/-- Check if three ComplexPoints are collinear -/
def areCollinear (p q r : ComplexPoint) : Prop :=
  ∃ k : ℝ, (r.re - p.re, r.im - p.im) = k • (q.re - p.re, q.im - p.im)

theorem complex_collinearity :
  ∃! b : ℝ, areCollinear
    (ComplexPoint.mk 3 (-5))
    (ComplexPoint.mk 1 (-1))
    (ComplexPoint.mk (-2) b) ∧
  b = 5 := by sorry

end complex_collinearity_l3953_395321


namespace expression_evaluation_l3953_395347

theorem expression_evaluation (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (h : x = 1 / y^2) :
  (x - 1 / x^2) * (y + 2 / y) = 2 * x^(5/2) - 1 / x :=
by sorry

end expression_evaluation_l3953_395347


namespace point_q_in_third_quadrant_l3953_395388

/-- A point in the 2D plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- Definition of the second quadrant -/
def is_in_second_quadrant (p : Point) : Prop :=
  p.x < 0 ∧ p.y > 0

/-- Definition of the third quadrant -/
def is_in_third_quadrant (p : Point) : Prop :=
  p.x < 0 ∧ p.y < 0

/-- Theorem: If A is in the second quadrant, then Q is in the third quadrant -/
theorem point_q_in_third_quadrant (A : Point) (h : is_in_second_quadrant A) :
  let Q : Point := ⟨A.x, -A.y⟩
  is_in_third_quadrant Q :=
by
  sorry

end point_q_in_third_quadrant_l3953_395388


namespace sales_amount_is_194_l3953_395373

/-- Represents the sales data for a stationery store --/
structure SalesData where
  eraser_price : ℝ
  regular_price : ℝ
  short_price : ℝ
  eraser_sold : ℕ
  regular_sold : ℕ
  short_sold : ℕ

/-- Calculates the total sales amount --/
def total_sales (data : SalesData) : ℝ :=
  data.eraser_price * data.eraser_sold +
  data.regular_price * data.regular_sold +
  data.short_price * data.short_sold

/-- Theorem stating that the total sales amount is $194 --/
theorem sales_amount_is_194 (data : SalesData) 
  (h1 : data.eraser_price = 0.8)
  (h2 : data.regular_price = 0.5)
  (h3 : data.short_price = 0.4)
  (h4 : data.eraser_sold = 200)
  (h5 : data.regular_sold = 40)
  (h6 : data.short_sold = 35) :
  total_sales data = 194 := by
  sorry

end sales_amount_is_194_l3953_395373


namespace three_layer_rug_area_l3953_395313

theorem three_layer_rug_area (total_area floor_area two_layer_area : ℝ) 
  (h1 : total_area = 200)
  (h2 : floor_area = 140)
  (h3 : two_layer_area = 22) :
  let three_layer_area := (total_area - floor_area - two_layer_area) / 2
  three_layer_area = 19 := by sorry

end three_layer_rug_area_l3953_395313


namespace ahsme_unanswered_questions_l3953_395374

/-- Represents the scoring system for AHSME -/
structure ScoringSystem where
  initial : ℕ
  correct : ℕ
  wrong : ℤ
  unanswered : ℕ

/-- Calculates the score based on the given scoring system and number of questions -/
def calculate_score (system : ScoringSystem) (correct wrong unanswered : ℕ) : ℤ :=
  system.initial + system.correct * correct + system.wrong * wrong + system.unanswered * unanswered

theorem ahsme_unanswered_questions 
  (new_system : ScoringSystem)
  (old_system : ScoringSystem)
  (total_questions : ℕ)
  (new_score : ℕ)
  (old_score : ℕ)
  (h_new_system : new_system = ⟨0, 5, 0, 2⟩)
  (h_old_system : old_system = ⟨30, 4, -1, 0⟩)
  (h_total_questions : total_questions = 30)
  (h_new_score : new_score = 93)
  (h_old_score : old_score = 84) :
  ∃ (correct wrong unanswered : ℕ),
    correct + wrong + unanswered = total_questions ∧
    calculate_score new_system correct wrong unanswered = new_score ∧
    calculate_score old_system correct wrong unanswered = old_score ∧
    unanswered = 9 :=
by sorry


end ahsme_unanswered_questions_l3953_395374


namespace liza_age_is_14_liza_older_than_nastya_liza_triple_nastya_two_years_ago_l3953_395344

/-- The age difference between Liza and Nastya -/
def age_difference : ℕ := 8

/-- Liza's current age -/
def liza_age : ℕ := 14

/-- Nastya's current age -/
def nastya_age : ℕ := liza_age - age_difference

theorem liza_age_is_14 : liza_age = 14 := by sorry

theorem liza_older_than_nastya : liza_age = nastya_age + age_difference := by sorry

theorem liza_triple_nastya_two_years_ago : 
  liza_age - 2 = 3 * (nastya_age - 2) := by sorry

end liza_age_is_14_liza_older_than_nastya_liza_triple_nastya_two_years_ago_l3953_395344


namespace dance_troupe_size_l3953_395328

/-- Represents the number of performers in a dance troupe with various skills -/
structure DanceTroupe where
  singers : ℕ
  dancers : ℕ
  instrumentalists : ℕ
  singer_dancers : ℕ
  singer_instrumentalists : ℕ
  dancer_instrumentalists : ℕ
  all_skilled : ℕ

/-- The conditions of the dance troupe problem -/
def dance_troupe_conditions (dt : DanceTroupe) : Prop :=
  dt.singers = 2 ∧
  dt.dancers = 26 ∧
  dt.instrumentalists = 22 ∧
  dt.singer_dancers = 8 ∧
  dt.singer_instrumentalists = 10 ∧
  dt.dancer_instrumentalists = 11 ∧
  (dt.singers + dt.dancers + dt.instrumentalists
    - dt.singer_dancers - dt.singer_instrumentalists - dt.dancer_instrumentalists + dt.all_skilled
    - (dt.singer_dancers + dt.singer_instrumentalists + dt.dancer_instrumentalists - 2 * dt.all_skilled)) =
  (dt.singer_dancers + dt.singer_instrumentalists + dt.dancer_instrumentalists - 2 * dt.all_skilled)

/-- The total number of performers in the dance troupe -/
def total_performers (dt : DanceTroupe) : ℕ :=
  dt.singers + dt.dancers + dt.instrumentalists
  - dt.singer_dancers - dt.singer_instrumentalists - dt.dancer_instrumentalists
  + dt.all_skilled

/-- Theorem stating that the total number of performers is 46 -/
theorem dance_troupe_size (dt : DanceTroupe) :
  dance_troupe_conditions dt → total_performers dt = 46 := by
  sorry


end dance_troupe_size_l3953_395328


namespace largest_four_digit_sum_20_l3953_395351

def digit_sum (n : ℕ) : ℕ :=
  let digits := n.digits 10
  digits.sum

def is_four_digit (n : ℕ) : Prop :=
  1000 ≤ n ∧ n ≤ 9999

theorem largest_four_digit_sum_20 :
  ∀ n : ℕ, is_four_digit n → digit_sum n = 20 → n ≤ 9920 :=
by sorry

end largest_four_digit_sum_20_l3953_395351


namespace tonys_rope_length_l3953_395334

/-- Represents a rope with its length and knot loss -/
structure Rope where
  length : Float
  knotLoss : Float

/-- Calculates the total length of ropes after tying them together -/
def totalRopeLength (ropes : List Rope) : Float :=
  let totalOriginalLength := ropes.map (·.length) |>.sum
  let totalLossFromKnots := ropes.map (·.knotLoss) |>.sum
  totalOriginalLength - totalLossFromKnots

/-- Theorem stating the total length of Tony's ropes after tying -/
theorem tonys_rope_length :
  let ropes : List Rope := [
    { length := 8, knotLoss := 1.2 },
    { length := 20, knotLoss := 1.5 },
    { length := 2, knotLoss := 1 },
    { length := 2, knotLoss := 1 },
    { length := 2, knotLoss := 1 },
    { length := 7, knotLoss := 0.8 },
    { length := 5, knotLoss := 1.2 },
    { length := 5, knotLoss := 1.2 }
  ]
  totalRopeLength ropes = 42.1 := by
  sorry

end tonys_rope_length_l3953_395334


namespace pizza_cost_l3953_395307

theorem pizza_cost (initial_amount : ℕ) (return_amount : ℕ) (juice_cost : ℕ) (juice_quantity : ℕ) (pizza_quantity : ℕ) :
  initial_amount = 50 ∧
  return_amount = 22 ∧
  juice_cost = 2 ∧
  juice_quantity = 2 ∧
  pizza_quantity = 2 →
  (initial_amount - return_amount - juice_cost * juice_quantity) / pizza_quantity = 12 :=
by sorry

end pizza_cost_l3953_395307


namespace pizza_order_theorem_l3953_395361

def pizza_order_cost (base_price : ℕ) (topping_price : ℕ) (tip : ℕ) : Prop :=
  let pepperoni_cost : ℕ := base_price + topping_price
  let sausage_cost : ℕ := base_price + topping_price
  let olive_mushroom_cost : ℕ := base_price + 2 * topping_price
  let total_before_tip : ℕ := pepperoni_cost + sausage_cost + olive_mushroom_cost
  let total_with_tip : ℕ := total_before_tip + tip
  total_with_tip = 39

theorem pizza_order_theorem :
  pizza_order_cost 10 1 5 :=
by
  sorry

end pizza_order_theorem_l3953_395361


namespace determinant_scaling_l3953_395392

theorem determinant_scaling (a b c d : ℝ) :
  Matrix.det !![a, b; c, d] = 5 →
  Matrix.det !![3*a, 3*b; 2*c, 2*d] = 30 := by
  sorry

end determinant_scaling_l3953_395392


namespace cylinder_height_ratio_l3953_395329

/-- 
Given two cylinders where:
- The first cylinder has height h and is 7/8 full of water
- The second cylinder has a radius 25% larger than the first
- All water from the first cylinder fills 3/5 of the second cylinder
Prove that the height of the second cylinder is 14/15 of h
-/
theorem cylinder_height_ratio (h : ℝ) (h' : ℝ) : 
  (7/8 : ℝ) * π * r^2 * h = (3/5 : ℝ) * π * (1.25 * r)^2 * h' → 
  h' = (14/15 : ℝ) * h :=
by sorry

end cylinder_height_ratio_l3953_395329


namespace cream_strawberry_prices_l3953_395375

/-- Represents the price of a box of cream strawberries in yuan -/
@[ext] structure StrawberryPrice where
  price : ℚ
  price_positive : price > 0

/-- The problem of finding cream strawberry prices -/
theorem cream_strawberry_prices 
  (price_A price_B : StrawberryPrice)
  (price_difference : price_A.price = price_B.price + 10)
  (quantity_equality : 800 / price_A.price = 600 / price_B.price) :
  price_A.price = 40 ∧ price_B.price = 30 := by
  sorry

end cream_strawberry_prices_l3953_395375


namespace smallest_dual_base_palindrome_l3953_395308

/-- A function that checks if a number is a palindrome in a given base -/
def isPalindrome (n : ℕ) (base : ℕ) : Prop :=
  sorry

/-- A function that converts a number from one base to another -/
def baseConvert (n : ℕ) (fromBase toBase : ℕ) : ℕ :=
  sorry

/-- A function that returns the number of digits of a number in a given base -/
def numDigits (n : ℕ) (base : ℕ) : ℕ :=
  sorry

theorem smallest_dual_base_palindrome :
  let n := 51  -- 110011₂ in base 10
  (∀ m < n, numDigits m 2 = 6 → isPalindrome m 2 → 
    ∀ b > 2, ¬(numDigits (baseConvert m 2 b) b = 4 ∧ isPalindrome (baseConvert m 2 b) b)) ∧
  numDigits n 2 = 6 ∧
  isPalindrome n 2 ∧
  numDigits (baseConvert n 2 3) 3 = 4 ∧
  isPalindrome (baseConvert n 2 3) 3 :=
by
  sorry

end smallest_dual_base_palindrome_l3953_395308


namespace window_purchase_savings_l3953_395342

/-- Calculates the cost of windows given the quantity and the store's offer -/
def windowCost (quantity : ℕ) : ℕ :=
  let regularPrice := 100
  let freeWindowsPer4 := quantity / 4
  (quantity - freeWindowsPer4) * regularPrice

/-- Calculates the savings when purchasing windows together vs separately -/
def calculateSavings (dave_windows : ℕ) (doug_windows : ℕ) : ℕ :=
  let separate_cost := windowCost dave_windows + windowCost doug_windows
  let joint_cost := windowCost (dave_windows + doug_windows)
  separate_cost - joint_cost

theorem window_purchase_savings :
  calculateSavings 7 8 = 100 := by
  sorry

end window_purchase_savings_l3953_395342


namespace root_sum_reciprocal_l3953_395315

theorem root_sum_reciprocal (p q r : ℂ) : 
  (p^3 - 2*p^2 - p + 3 = 0) →
  (q^3 - 2*q^2 - q + 3 = 0) →
  (r^3 - 2*r^2 - r + 3 = 0) →
  (p ≠ q) → (q ≠ r) → (p ≠ r) →
  1/(p-2) + 1/(q-2) + 1/(r-2) = -3 := by
sorry

end root_sum_reciprocal_l3953_395315


namespace mixed_number_calculation_l3953_395394

theorem mixed_number_calculation : 
  72 * ((2 + 3/4) - (3 + 1/2)) / ((3 + 1/3) + (1 + 1/4)) = -(13 + 1/11) := by
  sorry

end mixed_number_calculation_l3953_395394


namespace solution_set_equivalence_l3953_395320

theorem solution_set_equivalence (a b : ℝ) :
  (∀ x, ax^2 - b*x - 1 ≥ 0 ↔ x ∈ Set.Icc (-1/2) (-1/3)) →
  (∀ x, x^2 - b*x - a < 0 ↔ x ∈ Set.Ioo 2 3) :=
by sorry

end solution_set_equivalence_l3953_395320


namespace point_on_line_l3953_395389

/-- Given points A and B in the Cartesian plane, if point C satisfies the vector equation,
    then C lies on the line passing through A and B. -/
theorem point_on_line (A B C : ℝ × ℝ) (α β : ℝ) :
  A = (3, 1) →
  B = (-1, 3) →
  α + β = 1 →
  C = (α * A.1 + β * B.1, α * A.2 + β * B.2) →
  C.1 + 2 * C.2 - 5 = 0 := by
  sorry

end point_on_line_l3953_395389


namespace carlas_sunflowers_l3953_395346

/-- The number of sunflowers Carla has -/
def num_sunflowers : ℕ := sorry

/-- The number of dandelions Carla has -/
def num_dandelions : ℕ := 8

/-- The number of seeds per sunflower -/
def seeds_per_sunflower : ℕ := 9

/-- The number of seeds per dandelion -/
def seeds_per_dandelion : ℕ := 12

/-- The percentage of seeds that come from dandelions -/
def dandelion_seed_percentage : ℚ := 64 / 100

theorem carlas_sunflowers : 
  num_sunflowers = 6 ∧
  num_dandelions * seeds_per_dandelion = 
    (dandelion_seed_percentage : ℚ) * 
    (num_sunflowers * seeds_per_sunflower + num_dandelions * seeds_per_dandelion) :=
by sorry

end carlas_sunflowers_l3953_395346


namespace max_perimeter_special_triangle_l3953_395305

theorem max_perimeter_special_triangle :
  ∀ a b c : ℕ,
  (a = 4 * b) →
  (c = 20) →
  (a + b + c > a) →
  (a + b + c > b) →
  (a + b + c > c) →
  (a + b + c ≤ 50) :=
by sorry

end max_perimeter_special_triangle_l3953_395305


namespace tangent_line_equation_l3953_395393

-- Define the circle C
def circle_C (x y : ℝ) : Prop := x^2 + y^2 = 2

-- Define the point P
def point_P : ℝ × ℝ := (1, 2)

-- Define a function to check if a line is tangent to the circle C at a point
def is_tangent_to_C (a b c : ℝ) (x y : ℝ) : Prop :=
  circle_C x y ∧ a*x + b*y + c = 0 ∧
  ∀ x' y', circle_C x' y' → (a*x' + b*y' + c)^2 ≥ (a^2 + b^2) * (x'^2 + y'^2 - 2)

-- Theorem statement
theorem tangent_line_equation :
  ∃ (x₁ y₁ x₂ y₂ : ℝ),
    circle_C x₁ y₁ ∧ circle_C x₂ y₂ ∧
    is_tangent_to_C (point_P.1 - x₁) (point_P.2 - y₁) (x₁*(x₁ - point_P.1) + y₁*(y₁ - point_P.2)) x₁ y₁ ∧
    is_tangent_to_C (point_P.1 - x₂) (point_P.2 - y₂) (x₂*(x₂ - point_P.1) + y₂*(y₂ - point_P.2)) x₂ y₂ ∧
    x₁ + 2*y₁ - 2 = 0 ∧ x₂ + 2*y₂ - 2 = 0 :=
sorry

end tangent_line_equation_l3953_395393


namespace total_animals_is_130_l3953_395354

/-- The total number of animals seen throughout the day -/
def total_animals (initial_beavers initial_chipmunks : ℕ) : ℕ :=
  let morning_total := initial_beavers + initial_chipmunks
  let afternoon_beavers := 2 * initial_beavers
  let afternoon_chipmunks := initial_chipmunks - 10
  morning_total + afternoon_beavers + afternoon_chipmunks

/-- Theorem stating the total number of animals seen is 130 -/
theorem total_animals_is_130 :
  total_animals 20 40 = 130 := by
  sorry

end total_animals_is_130_l3953_395354


namespace remainder_of_1742_base12_div_9_l3953_395301

/-- Converts a base-12 digit to base-10 --/
def base12ToBase10(digit : Nat) : Nat :=
  if digit < 12 then digit else 0

/-- Converts a base-12 number to base-10 --/
def convertBase12ToBase10(n : List Nat) : Nat :=
  n.enum.foldl (fun acc (i, d) => acc + base12ToBase10 d * 12^i) 0

/-- The base-12 representation of 1742₁₂ --/
def base12Num : List Nat := [2, 4, 7, 1]

theorem remainder_of_1742_base12_div_9 :
  (convertBase12ToBase10 base12Num) % 9 = 3 := by
  sorry

end remainder_of_1742_base12_div_9_l3953_395301


namespace alma_carrot_distribution_l3953_395311

/-- Given a number of carrots and goats, calculate the number of carrots left over
    when distributing carrots equally among goats. -/
def carrots_left_over (total_carrots : ℕ) (num_goats : ℕ) : ℕ :=
  total_carrots % num_goats

theorem alma_carrot_distribution :
  carrots_left_over 47 4 = 3 := by
  sorry

end alma_carrot_distribution_l3953_395311


namespace negation_of_implication_l3953_395372

theorem negation_of_implication (a b : ℝ) :
  ¬(ab = 0 → a = 0 ∨ b = 0) ↔ (ab = 0 → a ≠ 0 ∧ b ≠ 0) :=
by sorry

end negation_of_implication_l3953_395372


namespace room_perimeter_is_16_l3953_395300

/-- A rectangular room with specific properties -/
structure Room where
  breadth : ℝ
  length : ℝ
  area : ℝ
  length_eq : length = 3 * breadth
  area_eq : area = length * breadth

/-- The perimeter of a rectangular room -/
def perimeter (r : Room) : ℝ := 2 * (r.length + r.breadth)

/-- Theorem: The perimeter of a room with given properties is 16 meters -/
theorem room_perimeter_is_16 (r : Room) (h : r.area = 12) : perimeter r = 16 := by
  sorry

end room_perimeter_is_16_l3953_395300
