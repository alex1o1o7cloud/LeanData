import Mathlib

namespace yoojeongs_marbles_l2363_236349

theorem yoojeongs_marbles (marbles_given : ℕ) (marbles_left : ℕ) :
  marbles_given = 8 →
  marbles_left = 24 →
  marbles_given + marbles_left = 32 :=
by
  sorry

end yoojeongs_marbles_l2363_236349


namespace solution_equation_1_solution_equation_2_l2363_236306

-- Equation 1
theorem solution_equation_1 (x : ℝ) : 2*x - 3*(2*x - 3) = x + 4 ↔ x = 1 := by sorry

-- Equation 2
theorem solution_equation_2 (x : ℝ) : (3*x - 1)/4 - 1 = (5*x - 7)/6 ↔ x = -1 := by sorry

end solution_equation_1_solution_equation_2_l2363_236306


namespace distance_AB_is_550_l2363_236312

/-- The distance between points A and B --/
def distance_AB : ℝ := 550

/-- Xiaodong's speed in meters per minute --/
def speed_Xiaodong : ℝ := 50

/-- Xiaorong's speed in meters per minute --/
def speed_Xiaorong : ℝ := 60

/-- Time taken for Xiaodong and Xiaorong to meet, in minutes --/
def meeting_time : ℝ := 10

/-- Theorem stating that the distance between points A and B is 550 meters --/
theorem distance_AB_is_550 :
  distance_AB = (speed_Xiaodong + speed_Xiaorong) * meeting_time / 2 :=
by sorry

end distance_AB_is_550_l2363_236312


namespace new_hires_count_l2363_236387

theorem new_hires_count (initial_workers : ℕ) (initial_men_ratio : ℚ) (final_women_percentage : ℚ) : 
  initial_workers = 90 →
  initial_men_ratio = 2/3 →
  final_women_percentage = 40/100 →
  ∃ (new_hires : ℕ), 
    (initial_workers * (1 - initial_men_ratio) + new_hires) / (initial_workers + new_hires) = final_women_percentage ∧
    new_hires = 10 := by
  sorry

end new_hires_count_l2363_236387


namespace at_least_one_non_negative_l2363_236346

theorem at_least_one_non_negative (a₁ a₂ a₃ a₄ a₅ a₆ a₇ a₈ : ℝ) :
  max (a₁*a₃ + a₂*a₄) (max (a₁*a₅ + a₂*a₆) (max (a₁*a₇ + a₂*a₈) 
    (max (a₃*a₅ + a₄*a₆) (max (a₃*a₇ + a₄*a₈) (a₅*a₇ + a₆*a₈))))) ≥ 0 := by
  sorry

end at_least_one_non_negative_l2363_236346


namespace dante_sold_coconuts_l2363_236343

/-- The number of coconuts Paolo has -/
def paolo_coconuts : ℕ := 14

/-- The number of coconuts Dante has relative to Paolo -/
def dante_multiplier : ℕ := 3

/-- The number of coconuts Dante has left after selling -/
def dante_coconuts_left : ℕ := 32

/-- The number of coconuts Dante sold -/
def dante_sold : ℕ := dante_multiplier * paolo_coconuts - dante_coconuts_left

theorem dante_sold_coconuts : dante_sold = 10 := by
  sorry

end dante_sold_coconuts_l2363_236343


namespace johns_allowance_l2363_236360

def weekly_allowance : ℝ → Prop :=
  λ A => 
    let arcade_spent := (3/5) * A
    let remaining_after_arcade := A - arcade_spent
    let toy_store_spent := (1/3) * remaining_after_arcade
    let remaining_after_toy_store := remaining_after_arcade - toy_store_spent
    remaining_after_toy_store = 0.60

theorem johns_allowance : ∃ A : ℝ, weekly_allowance A ∧ A = 2.25 := by sorry

end johns_allowance_l2363_236360


namespace blue_highlighters_count_l2363_236341

/-- The number of pink highlighters in the teacher's desk -/
def pink_highlighters : ℕ := 9

/-- The number of yellow highlighters in the teacher's desk -/
def yellow_highlighters : ℕ := 8

/-- The total number of highlighters in the teacher's desk -/
def total_highlighters : ℕ := 22

/-- The number of blue highlighters in the teacher's desk -/
def blue_highlighters : ℕ := total_highlighters - (pink_highlighters + yellow_highlighters)

theorem blue_highlighters_count : blue_highlighters = 5 := by
  sorry

end blue_highlighters_count_l2363_236341


namespace sum_of_squares_near_n_l2363_236317

theorem sum_of_squares_near_n (n : ℕ) (h : n > 10000) :
  ∃ m : ℕ, ∃ x y : ℕ, 
    m = x^2 + y^2 ∧ 
    0 < m - n ∧
    (m - n : ℝ) < 3 * Real.sqrt (n : ℝ) :=
by sorry

end sum_of_squares_near_n_l2363_236317


namespace trajectory_equation_l2363_236398

/-- 
Given a point P in the plane, if its distance to the line y=-3 is equal to 
its distance to the point (0,3), then the equation of its trajectory is x^2 = 12y.
-/
theorem trajectory_equation (P : ℝ × ℝ) : 
  (∀ (x y : ℝ), P = (x, y) → |y + 3| = ((x - 0)^2 + (y - 3)^2).sqrt) →
  (∃ (x y : ℝ), P = (x, y) ∧ x^2 = 12*y) :=
sorry

end trajectory_equation_l2363_236398


namespace system_solution_l2363_236318

theorem system_solution (a b c x y z : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) :
  (1 / x + 1 / y + 1 / z = 1) ∧ (a * x = b * y) ∧ (b * y = c * z) →
  (x = (a + b + c) / a) ∧ (y = (a + b + c) / b) ∧ (z = (a + b + c) / c) := by
  sorry

end system_solution_l2363_236318


namespace final_concentration_is_correct_l2363_236371

/-- Represents the volume of saline solution in the cup -/
def initial_volume : ℝ := 1

/-- Represents the initial concentration of the saline solution -/
def initial_concentration : ℝ := 0.16

/-- Represents the volume ratio of the large ball -/
def large_ball_ratio : ℝ := 10

/-- Represents the volume ratio of the medium ball -/
def medium_ball_ratio : ℝ := 4

/-- Represents the volume ratio of the small ball -/
def small_ball_ratio : ℝ := 3

/-- Represents the percentage of solution that overflows when the small ball is immersed -/
def overflow_percentage : ℝ := 0.1

/-- Calculates the final concentration of the saline solution after the process -/
def final_concentration : ℝ := sorry

/-- Theorem stating that the final concentration is approximately 10.7% -/
theorem final_concentration_is_correct : 
  ∀ ε > 0, |final_concentration - 0.107| < ε := by sorry

end final_concentration_is_correct_l2363_236371


namespace mass_of_six_moles_l2363_236386

/-- Given a compound with a molecular weight of 444 g/mol, 
    the mass of 6 moles of this compound is 2664 g. -/
theorem mass_of_six_moles (molecular_weight : ℝ) (h : molecular_weight = 444) : 
  6 * molecular_weight = 2664 := by
  sorry

end mass_of_six_moles_l2363_236386


namespace min_output_avoids_losses_l2363_236384

/-- The profit function for a company's product -/
def profit_function (x : ℝ) : ℝ := 0.1 * x - 150

/-- The minimum output to avoid losses -/
def min_output : ℝ := 1500

theorem min_output_avoids_losses :
  ∀ x : ℝ, x ≥ min_output → profit_function x ≥ 0 ∧
  ∀ y : ℝ, y < min_output → ∃ z : ℝ, z ≥ y ∧ profit_function z < 0 :=
by sorry

end min_output_avoids_losses_l2363_236384


namespace last_term_formula_l2363_236313

def u (n : ℕ) : ℕ := 2 + 5 * ((n - 1) % (3 * ((n - 1).sqrt + 1) - 1))

def f (n : ℕ) : ℕ := (15 * n^2 + 10 * n + 4) / 2

theorem last_term_formula (n : ℕ) : 
  u ((n^2 + n) / 2) = f n :=
sorry

end last_term_formula_l2363_236313


namespace extremum_sum_l2363_236331

/-- The function f(x) with parameters a and b -/
def f (a b x : ℝ) : ℝ := x^3 + a*x^2 + b*x + a^2

/-- The derivative of f(x) with respect to x -/
def f' (a b x : ℝ) : ℝ := 3*x^2 + 2*a*x + b

theorem extremum_sum (a b : ℝ) : 
  (f a b 1 = 10) ∧ (f' a b 1 = 0) → a + b = -7 :=
by
  sorry

#check extremum_sum

end extremum_sum_l2363_236331


namespace sarah_boxes_count_l2363_236385

def total_apples : ℕ := 49
def apples_per_box : ℕ := 7

theorem sarah_boxes_count :
  total_apples / apples_per_box = 7 :=
by
  sorry

end sarah_boxes_count_l2363_236385


namespace johns_outfit_cost_l2363_236353

theorem johns_outfit_cost (pants_cost : ℝ) (h1 : pants_cost + 1.6 * pants_cost = 130) : pants_cost = 50 := by
  sorry

end johns_outfit_cost_l2363_236353


namespace triangle_properties_l2363_236326

-- Define the triangle ABC
def Triangle (A B C : ℝ × ℝ) : Prop :=
  let ab := Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2)
  let bc := Real.sqrt ((B.1 - C.1)^2 + (B.2 - C.2)^2)
  let ac := Real.sqrt ((A.1 - C.1)^2 + (A.2 - C.2)^2)
  let s := (ab + bc + ac) / 2
  ab = 6 ∧ bc = 5 ∧ Real.sqrt (s * (s - ab) * (s - bc) * (s - ac)) = 9

-- Define an acute triangle
def AcuteTriangle (A B C : ℝ × ℝ) : Prop :=
  Triangle A B C ∧
  (A.1 - B.1) * (C.1 - B.1) + (A.2 - B.2) * (C.2 - B.2) > 0 ∧
  (B.1 - A.1) * (C.1 - A.1) + (B.2 - A.2) * (C.2 - A.2) > 0 ∧
  (A.1 - C.1) * (B.1 - C.1) + (A.2 - C.2) * (B.2 - C.2) > 0

theorem triangle_properties (A B C : ℝ × ℝ) :
  Triangle A B C →
  (∃ ac : ℝ, (ac = Real.sqrt 13 ∨ ac = Real.sqrt 109) ∧
   ac = Real.sqrt ((A.1 - C.1)^2 + (A.2 - C.2)^2)) ∧
  (AcuteTriangle A B C →
   ∃ angle_A : ℝ,
   Real.cos (2 * angle_A + π / 6) = (-5 * Real.sqrt 3 - 12) / 26) :=
sorry

end triangle_properties_l2363_236326


namespace blake_initial_milk_l2363_236358

/-- The amount of milk needed for one milkshake in ounces -/
def milk_per_milkshake : ℕ := 4

/-- The amount of ice cream needed for one milkshake in ounces -/
def ice_cream_per_milkshake : ℕ := 12

/-- The total amount of ice cream available in ounces -/
def total_ice_cream : ℕ := 192

/-- The amount of milk left over after making milkshakes in ounces -/
def milk_leftover : ℕ := 8

/-- The initial amount of milk Blake had -/
def initial_milk : ℕ := total_ice_cream / ice_cream_per_milkshake * milk_per_milkshake + milk_leftover

theorem blake_initial_milk :
  initial_milk = 72 :=
sorry

end blake_initial_milk_l2363_236358


namespace cricketer_average_after_22nd_inning_l2363_236348

/-- Represents a cricketer's performance --/
structure CricketerPerformance where
  innings : ℕ
  scoreLastInning : ℕ
  averageIncrease : ℚ

/-- Calculates the average score after the last inning --/
def averageAfterLastInning (c : CricketerPerformance) : ℚ :=
  let previousAverage := (c.innings - 1 : ℚ) * (c.averageIncrease + (c.scoreLastInning : ℚ) / c.innings)
  (previousAverage + c.scoreLastInning) / c.innings

/-- Theorem stating the cricketer's average after the 22nd inning --/
theorem cricketer_average_after_22nd_inning 
  (c : CricketerPerformance)
  (h1 : c.innings = 22)
  (h2 : c.scoreLastInning = 134)
  (h3 : c.averageIncrease = 7/2) :
  averageAfterLastInning c = 121/2 := by
  sorry

end cricketer_average_after_22nd_inning_l2363_236348


namespace negation_of_implication_l2363_236389

theorem negation_of_implication (x y : ℝ) : 
  ¬(x^2 + y^2 = 0 → x = 0 ∧ y = 0) ↔ (x^2 + y^2 = 0 ∧ (x ≠ 0 ∨ y ≠ 0)) :=
by sorry

end negation_of_implication_l2363_236389


namespace product_difference_equals_one_l2363_236367

theorem product_difference_equals_one : (527 : ℤ) * 527 - 526 * 528 = 1 := by
  sorry

end product_difference_equals_one_l2363_236367


namespace conveyor_belts_combined_time_l2363_236339

/-- The time taken for two conveyor belts to move one day's coal output together -/
theorem conveyor_belts_combined_time (old_rate new_rate : ℝ) 
  (h1 : old_rate = 1 / 21)
  (h2 : new_rate = 1 / 15) : 
  1 / (old_rate + new_rate) = 35 / 4 := by
  sorry

end conveyor_belts_combined_time_l2363_236339


namespace softball_team_composition_l2363_236330

theorem softball_team_composition :
  ∀ (men women : ℕ),
  men + women = 16 →
  (men : ℚ) / (women : ℚ) = 7 / 9 →
  women - men = 2 :=
by
  sorry

end softball_team_composition_l2363_236330


namespace election_winner_percentage_l2363_236354

theorem election_winner_percentage (winner_votes loser_votes total_votes : ℕ) 
  (h1 : winner_votes = 899)
  (h2 : winner_votes - loser_votes = 348)
  (h3 : total_votes = winner_votes + loser_votes) :
  (winner_votes : ℝ) / (total_votes : ℝ) * 100 = 899 / 1450 * 100 := by
  sorry

end election_winner_percentage_l2363_236354


namespace bottles_left_l2363_236378

theorem bottles_left (initial : Real) (maria_drank : Real) (sister_drank : Real) 
  (h1 : initial = 45.0)
  (h2 : maria_drank = 14.0)
  (h3 : sister_drank = 8.0) :
  initial - maria_drank - sister_drank = 23.0 := by
  sorry

end bottles_left_l2363_236378


namespace max_value_of_ab_l2363_236323

theorem max_value_of_ab (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a + 2*b = 8) :
  ∃ (m : ℝ), m = 8 ∧ ∀ x y, x > 0 → y > 0 → x + 2*y = 8 → x*y ≤ m :=
sorry

end max_value_of_ab_l2363_236323


namespace original_nes_price_l2363_236362

/-- Calculates the original NES sale price before tax given trade-in values and final payment details --/
theorem original_nes_price
  (snes_value : ℝ)
  (snes_credit_rate : ℝ)
  (gameboy_value : ℝ)
  (gameboy_credit_rate : ℝ)
  (ps2_value : ℝ)
  (ps2_credit_rate : ℝ)
  (nes_discount_rate : ℝ)
  (sales_tax_rate : ℝ)
  (cash_paid : ℝ)
  (change_received : ℝ)
  (free_game_value : ℝ)
  (h1 : snes_value = 150)
  (h2 : snes_credit_rate = 0.8)
  (h3 : gameboy_value = 50)
  (h4 : gameboy_credit_rate = 0.75)
  (h5 : ps2_value = 100)
  (h6 : ps2_credit_rate = 0.6)
  (h7 : nes_discount_rate = 0.15)
  (h8 : sales_tax_rate = 0.05)
  (h9 : cash_paid = 80)
  (h10 : change_received = 10)
  (h11 : free_game_value = 30) :
  ∃ (original_price : ℝ), abs (original_price - 289.08) < 0.01 := by
  sorry


end original_nes_price_l2363_236362


namespace rectangle_dimension_increase_l2363_236350

theorem rectangle_dimension_increase (L B : ℝ) (h1 : L > 0) (h2 : B > 0) :
  let L' := 1.3 * L
  let A := L * B
  let A' := 1.885 * A
  ∃ p : ℝ, p > 0 ∧ L' * (B * (1 + p / 100)) = A' ∧ p = 45 :=
by sorry

end rectangle_dimension_increase_l2363_236350


namespace tangent_slope_at_one_l2363_236308

-- Define the function
def f (x : ℝ) : ℝ := x^3 - 2*x^2

-- Define the derivative of the function
def f' (x : ℝ) : ℝ := 3*x^2 - 4*x

-- Theorem statement
theorem tangent_slope_at_one :
  (f' 1) = -1 := by sorry

end tangent_slope_at_one_l2363_236308


namespace certain_number_proof_l2363_236300

theorem certain_number_proof (x : ℕ+) (h : (55 * x.val) % 7 = 6) : x.val % 7 = 1 := by
  sorry

end certain_number_proof_l2363_236300


namespace seven_balls_three_boxes_l2363_236336

/-- The number of ways to distribute indistinguishable balls into distinguishable boxes -/
def distribute_balls (total_balls : ℕ) (num_boxes : ℕ) : ℕ :=
  Nat.choose (total_balls + num_boxes - 1) (num_boxes - 1)

/-- The number of ways to distribute indistinguishable balls into distinguishable boxes
    with at least one ball in each box -/
def distribute_balls_no_empty (total_balls : ℕ) (num_boxes : ℕ) : ℕ :=
  distribute_balls (total_balls - num_boxes) num_boxes

theorem seven_balls_three_boxes :
  distribute_balls_no_empty 7 3 = 15 := by
  sorry

end seven_balls_three_boxes_l2363_236336


namespace unique_arithmetic_grid_solution_l2363_236397

/-- Represents a 5x5 grid of integers -/
def Grid := Matrix (Fin 5) (Fin 5) Int

/-- Checks if a sequence of 5 integers forms an arithmetic progression -/
def isArithmeticSequence (seq : Fin 5 → Int) : Prop :=
  ∃ d : Int, ∀ i : Fin 5, i.val < 4 → seq (i + 1) - seq i = d

/-- The initial grid with given values -/
def initialGrid : Grid :=
  fun i j => if i = 0 ∧ j = 0 then 2
             else if i = 0 ∧ j = 4 then 14
             else if i = 1 ∧ j = 1 then 8
             else if i = 2 ∧ j = 1 then 11
             else if i = 2 ∧ j = 2 then 16
             else if i = 4 ∧ j = 0 then 10
             else 0  -- placeholder for unknown values

/-- Theorem stating the existence and uniqueness of the solution -/
theorem unique_arithmetic_grid_solution :
  ∃! g : Grid,
    (∀ i j, initialGrid i j ≠ 0 → g i j = initialGrid i j) ∧
    (∀ i, isArithmeticSequence (fun j => g i j)) ∧
    (∀ j, isArithmeticSequence (fun i => g i j)) := by
  sorry

end unique_arithmetic_grid_solution_l2363_236397


namespace checkerboard_partition_l2363_236388

theorem checkerboard_partition (n : ℕ) : 
  n % 5 = 0 → n % 7 = 0 → n ≤ 200 → n % 6 ≠ 0 := by
sorry

end checkerboard_partition_l2363_236388


namespace solve_equation_l2363_236337

theorem solve_equation : ∃ x : ℝ, (x - 5)^4 = (1/16)⁻¹ ∧ x = 7 := by
  sorry

end solve_equation_l2363_236337


namespace ball_max_height_l2363_236301

def h (t : ℝ) : ℝ := -20 * t^2 + 100 * t + 36

theorem ball_max_height :
  ∃ (t_max : ℝ), ∀ (t : ℝ), h t ≤ h t_max ∧ h t_max = 161 :=
sorry

end ball_max_height_l2363_236301


namespace quadratic_rational_solutions_product_l2363_236332

theorem quadratic_rational_solutions_product : ∃ (d₁ d₂ : ℕ+),
  (∀ (d : ℕ+), (∃ (x : ℚ), 8 * x^2 + 16 * x + d.val = 0) ↔ (d = d₁ ∨ d = d₂)) ∧
  d₁.val * d₂.val = 48 :=
sorry

end quadratic_rational_solutions_product_l2363_236332


namespace f_composition_one_ninth_l2363_236370

noncomputable def f (x : ℝ) : ℝ :=
  if x ≤ 0 then x^3 + 1 else Real.log x / Real.log 3

theorem f_composition_one_ninth : f (f (1/9)) = -7 := by
  sorry

end f_composition_one_ninth_l2363_236370


namespace arithmetic_sequence_squares_l2363_236375

theorem arithmetic_sequence_squares (k : ℤ) : ∃ (a : ℕ → ℤ), 
  (∀ n, a (n + 1) - a n = a (n + 2) - a (n + 1)) ∧ 
  (a 1)^2 = 36 + k ∧
  (a 2)^2 = 300 + k ∧
  (a 3)^2 = 596 + k ∧
  k = 925 := by
sorry

end arithmetic_sequence_squares_l2363_236375


namespace cindy_calculation_l2363_236395

theorem cindy_calculation (x : ℚ) : 
  ((x - 5) * 3 / 7 = 10) → ((3 * x - 5) / 7 = 80 / 7) := by
  sorry

end cindy_calculation_l2363_236395


namespace correspondence_count_l2363_236321

-- Define the sets and correspondences
def Triangle : Type := sorry
def Circle : Type := sorry
def RealNumber : Type := ℝ

-- Define the correspondences
def correspondence1 : Triangle → Circle := sorry
def correspondence2 : Triangle → RealNumber := sorry
def correspondence3 : RealNumber → RealNumber := sorry
def correspondence4 : RealNumber → RealNumber := sorry

-- Define what it means to be a mapping
def is_mapping (f : α → β) : Prop := ∀ x : α, ∃! y : β, f x = y

-- Define what it means to be a function
def is_function (f : α → β) : Prop := ∀ x : α, ∃ y : β, f x = y

-- The main theorem
theorem correspondence_count :
  (is_mapping correspondence1 ∧
   is_mapping correspondence2 ∧
   is_mapping correspondence3 ∧
   ¬is_mapping correspondence4) ∧
  (¬is_function correspondence1 ∧
   is_function correspondence2 ∧
   is_function correspondence3 ∧
   ¬is_function correspondence4) :=
sorry

end correspondence_count_l2363_236321


namespace sphere_volume_from_surface_area_l2363_236325

/-- Given a sphere with surface area 256π cm², its volume is 2048π/3 cm³. -/
theorem sphere_volume_from_surface_area :
  ∀ r : ℝ,
  (4 : ℝ) * π * r^2 = 256 * π →
  (4 : ℝ) / 3 * π * r^3 = 2048 * π / 3 := by
sorry

end sphere_volume_from_surface_area_l2363_236325


namespace seven_mult_three_equals_sixteen_l2363_236391

-- Define the custom operation *
def custom_mult (a b : ℤ) : ℤ := 4*a + 3*b - a*b

-- State the theorem
theorem seven_mult_three_equals_sixteen : custom_mult 7 3 = 16 := by sorry

end seven_mult_three_equals_sixteen_l2363_236391


namespace monotonicity_f_when_a_is_1_min_a_when_f_has_no_zeros_l2363_236304

noncomputable section

-- Define the function f(x) with parameter a
def f (a : ℝ) (x : ℝ) : ℝ := (2 - a) * (x - 1) - 2 * Real.log x

-- Part 1: Monotonicity of f when a = 1
theorem monotonicity_f_when_a_is_1 :
  ∀ x₁ x₂, 0 < x₁ ∧ x₁ < x₂ ∧ x₂ ≤ 2 → f 1 x₁ > f 1 x₂ ∧
  ∀ x₃ x₄, 2 ≤ x₃ ∧ x₃ < x₄ → f 1 x₃ < f 1 x₄ := by sorry

-- Part 2: Minimum value of a when f has no zeros in (0, 1/2)
theorem min_a_when_f_has_no_zeros :
  (∀ x, 0 < x ∧ x < 1/2 → f a x ≠ 0) →
  a ≥ 2 - 4 * Real.log 2 := by sorry

end

end monotonicity_f_when_a_is_1_min_a_when_f_has_no_zeros_l2363_236304


namespace missing_files_l2363_236369

theorem missing_files (total : ℕ) (morning : ℕ) (afternoon : ℕ) : 
  total = 60 → 
  morning = total / 2 → 
  afternoon = 15 → 
  total - (morning + afternoon) = 15 :=
by
  sorry

end missing_files_l2363_236369


namespace mike_payment_l2363_236302

def medical_costs (x_ray : ℝ) (blood_tests : ℝ) (deductible : ℝ) : ℝ :=
  let mri := 3 * x_ray
  let ct_scan := 2 * mri
  let ultrasound := 0.5 * mri
  let total_cost := x_ray + mri + ct_scan + blood_tests + ultrasound
  let remaining_amount := total_cost - deductible
  let insurance_coverage := 0.8 * x_ray + 0.8 * mri + 0.7 * ct_scan + 0.5 * blood_tests + 0.6 * ultrasound
  remaining_amount - insurance_coverage

theorem mike_payment (x_ray : ℝ) (blood_tests : ℝ) (deductible : ℝ) 
  (h1 : x_ray = 250)
  (h2 : blood_tests = 200)
  (h3 : deductible = 500) :
  medical_costs x_ray blood_tests deductible = 400 := by
  sorry

end mike_payment_l2363_236302


namespace part_one_part_two_l2363_236366

-- Define the function y
def y (x a : ℝ) : ℝ := 2 * x^2 - (a + 2) * x + a

-- Part 1
theorem part_one : 
  ∀ x : ℝ, y x (-1) > 0 ↔ (x > 1 ∨ x < -1/2) := by sorry

-- Part 2
theorem part_two :
  ∀ a x₁ x₂ : ℝ, 
    (x₁ > 0 ∧ x₂ > 0) →
    (2 * x₁^2 - (a + 2) * x₁ + a = x₁ + 1) →
    (2 * x₂^2 - (a + 2) * x₂ + a = x₂ + 1) →
    (∀ x : ℝ, x > 0 → x₂/x₁ + x₁/x₂ ≥ 6) ∧ 
    (∃ a : ℝ, x₂/x₁ + x₁/x₂ = 6) := by sorry

end part_one_part_two_l2363_236366


namespace jacks_stamp_collection_value_l2363_236379

/-- Given a collection of stamps where all stamps have equal value, 
    calculate the total value of the collection. -/
def stamp_collection_value (total_stamps : ℕ) (sample_stamps : ℕ) (sample_value : ℕ) : ℕ :=
  total_stamps * (sample_value / sample_stamps)

/-- Prove that Jack's stamp collection is worth 80 dollars -/
theorem jacks_stamp_collection_value :
  stamp_collection_value 20 4 16 = 80 := by
  sorry

end jacks_stamp_collection_value_l2363_236379


namespace blueberry_muffin_percentage_l2363_236316

/-- Calculates the percentage of blueberry muffins out of the total muffins -/
theorem blueberry_muffin_percentage
  (num_cartons : ℕ)
  (blueberries_per_carton : ℕ)
  (blueberries_per_muffin : ℕ)
  (num_cinnamon_muffins : ℕ)
  (h1 : num_cartons = 3)
  (h2 : blueberries_per_carton = 200)
  (h3 : blueberries_per_muffin = 10)
  (h4 : num_cinnamon_muffins = 60)
  : (((num_cartons * blueberries_per_carton) / blueberries_per_muffin : ℚ) /
     ((num_cartons * blueberries_per_carton) / blueberries_per_muffin + num_cinnamon_muffins)) * 100 = 50 := by
  sorry

end blueberry_muffin_percentage_l2363_236316


namespace remainder_problem_l2363_236399

theorem remainder_problem (d : ℕ) (a b : ℕ) (h1 : d > 0) (h2 : d ≤ a ∧ d ≤ b) 
  (h3 : ∀ k > d, k ∣ a ∨ k ∣ b) (h4 : b % d = 5) : a % d = 6 := by
  sorry

end remainder_problem_l2363_236399


namespace no_linear_term_implies_equal_coefficients_l2363_236311

/-- Given a polynomial (x+p)(x-q) with no linear term in x, prove that p = q -/
theorem no_linear_term_implies_equal_coefficients (p q : ℝ) :
  (∀ x : ℝ, ∃ a b : ℝ, (x + p) * (x - q) = a * x^2 + b) → p = q := by
  sorry

end no_linear_term_implies_equal_coefficients_l2363_236311


namespace division_multiplication_order_matters_l2363_236355

theorem division_multiplication_order_matters : (32 / 0.25) * 4 ≠ 32 / (0.25 * 4) := by
  sorry

end division_multiplication_order_matters_l2363_236355


namespace stick_length_ratio_l2363_236351

/-- Proves that the ratio of the second stick to the first stick is 2:1 given the conditions of the problem -/
theorem stick_length_ratio (stick2 : ℝ) 
  (h1 : 3 + stick2 + (stick2 - 1) = 14) : 
  stick2 / 3 = 2 := by sorry

end stick_length_ratio_l2363_236351


namespace prob_at_least_one_girl_pair_value_l2363_236380

/-- The number of boys in the group -/
def num_boys : ℕ := 4

/-- The number of girls in the group -/
def num_girls : ℕ := 4

/-- The total number of people in the group -/
def total_people : ℕ := num_boys + num_girls

/-- The number of pairs formed -/
def num_pairs : ℕ := total_people / 2

/-- The total number of ways to pair 8 people -/
def total_pairings : ℕ := (total_people.factorial) / (2^num_pairs * num_pairs.factorial)

/-- The number of ways to pair boys with girls (no all-girl pairs) -/
def boy_girl_pairings : ℕ := num_boys.factorial

/-- The probability of at least one pair consisting of two girls -/
def prob_at_least_one_girl_pair : ℚ := 1 - (boy_girl_pairings : ℚ) / total_pairings

theorem prob_at_least_one_girl_pair_value :
  prob_at_least_one_girl_pair = 27 / 35 := by sorry

end prob_at_least_one_girl_pair_value_l2363_236380


namespace factorial_sum_unique_solution_l2363_236319

theorem factorial_sum_unique_solution :
  ∀ w x y z : ℕ+,
  w.val.factorial = x.val.factorial + y.val.factorial + z.val.factorial →
  w = 3 ∧ x = 2 ∧ y = 2 ∧ z = 2 :=
by sorry

end factorial_sum_unique_solution_l2363_236319


namespace marcos_strawberries_weight_l2363_236394

theorem marcos_strawberries_weight (total_weight dad_weight : ℕ) 
  (h1 : total_weight = 23)
  (h2 : dad_weight = 9) :
  total_weight - dad_weight = 14 := by
  sorry

end marcos_strawberries_weight_l2363_236394


namespace arithmetic_mean_difference_l2363_236356

theorem arithmetic_mean_difference (p q r : ℝ) 
  (h1 : (p + q) / 2 = 10) 
  (h2 : (q + r) / 2 = 26) : 
  r - p = 32 := by sorry

end arithmetic_mean_difference_l2363_236356


namespace special_quadrilateral_integer_perimeter_l2363_236307

/-- A quadrilateral with specific properties -/
structure SpecialQuadrilateral where
  -- Points
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ
  D : ℝ × ℝ
  O : ℝ × ℝ
  -- Perpendicular conditions
  ab_perp_bc : (A.1 - B.1) * (B.1 - C.1) + (A.2 - B.2) * (B.2 - C.2) = 0
  bc_perp_cd : (B.1 - C.1) * (C.1 - D.1) + (B.2 - C.2) * (C.2 - D.2) = 0
  -- BC tangent to circle condition
  bc_tangent : (B.1 - O.1) * (C.1 - O.1) + (B.2 - O.2) * (C.2 - O.2) = 0
  -- AD is diameter
  ad_diameter : (A.1 - O.1)^2 + (A.2 - O.2)^2 = (D.1 - O.1)^2 + (D.2 - O.2)^2

/-- Perimeter of the quadrilateral is an integer when AB and CD are integers with AB = 2CD -/
theorem special_quadrilateral_integer_perimeter 
  (q : SpecialQuadrilateral) 
  (ab cd : ℕ) 
  (h_ab : ab = 2 * cd) 
  (h_ab_length : (q.A.1 - q.B.1)^2 + (q.A.2 - q.B.2)^2 = ab^2) 
  (h_cd_length : (q.C.1 - q.D.1)^2 + (q.C.2 - q.D.2)^2 = cd^2) :
  ∃ (n : ℕ), 
    (q.A.1 - q.B.1)^2 + (q.A.2 - q.B.2)^2 +
    (q.B.1 - q.C.1)^2 + (q.B.2 - q.C.2)^2 +
    (q.C.1 - q.D.1)^2 + (q.C.2 - q.D.2)^2 +
    (q.D.1 - q.A.1)^2 + (q.D.2 - q.A.2)^2 = n^2 := by
  sorry

end special_quadrilateral_integer_perimeter_l2363_236307


namespace greatest_3digit_base8_divisible_by_7_l2363_236340

def base_8_to_decimal (n : Nat) : Nat :=
  (n / 100) * 64 + ((n / 10) % 10) * 8 + (n % 10)

theorem greatest_3digit_base8_divisible_by_7 :
  ∀ n : Nat,
  n < 1000 →
  n > 0 →
  base_8_to_decimal n % 7 = 0 →
  n ≤ 777 :=
by sorry

end greatest_3digit_base8_divisible_by_7_l2363_236340


namespace expand_and_simplify_l2363_236376

theorem expand_and_simplify (x y : ℝ) : (x - 2*y)^2 - 2*y*(y - 2*x) = x^2 + 2*y^2 := by
  sorry

end expand_and_simplify_l2363_236376


namespace mina_driving_problem_l2363_236334

theorem mina_driving_problem (initial_distance : ℝ) (initial_speed : ℝ) (second_speed : ℝ) (target_average_speed : ℝ) 
  (h : initial_distance = 20 ∧ initial_speed = 40 ∧ second_speed = 60 ∧ target_average_speed = 55) :
  ∃ additional_distance : ℝ,
    (initial_distance + additional_distance) / ((initial_distance / initial_speed) + (additional_distance / second_speed)) = target_average_speed ∧
    additional_distance = 90 := by
  sorry

end mina_driving_problem_l2363_236334


namespace inequality_solution_set_l2363_236324

theorem inequality_solution_set (x : ℝ) : 3 * x > 2 * x + 4 ↔ x > 4 := by sorry

end inequality_solution_set_l2363_236324


namespace slanted_line_angle_l2363_236328

/-- The angle between a slanted line segment and a plane, given that the slanted line segment
    is twice the length of its projection on the plane. -/
theorem slanted_line_angle (L l : ℝ) (h : L = 2 * l) :
  Real.arccos (l / L) = π / 3 := by
  sorry

end slanted_line_angle_l2363_236328


namespace closest_points_on_hyperbola_l2363_236333

/-- The hyperbola equation -/
def hyperbola (x y : ℝ) : Prop := x^2 - y^2 = 9

/-- The distance squared function from a point (x, y) to A(0, -3) -/
def distance_squared (x y : ℝ) : ℝ := x^2 + (y + 3)^2

/-- The point A -/
def A : ℝ × ℝ := (0, -3)

/-- Theorem stating that the given points are the closest to A on the hyperbola -/
theorem closest_points_on_hyperbola :
  ∃ (x₁ y₁ x₂ y₂ : ℝ),
    hyperbola x₁ y₁ ∧ hyperbola x₂ y₂ ∧
    (x₁ = -3 * Real.sqrt 5 / 2 ∧ y₁ = -3 / 2) ∧
    (x₂ = 3 * Real.sqrt 5 / 2 ∧ y₂ = -3 / 2) ∧
    (∀ (x y : ℝ), hyperbola x y → 
      distance_squared x y ≥ distance_squared x₁ y₁ ∧
      distance_squared x y ≥ distance_squared x₂ y₂) :=
sorry

end closest_points_on_hyperbola_l2363_236333


namespace digit_sum_in_multiplication_l2363_236352

theorem digit_sum_in_multiplication (c d a b : ℕ) : 
  c < 10 → d < 10 → a < 10 → b < 10 →
  (30 + c) * (10 * d + 4) = 100 * a + 10 * b + 8 →
  c + d = 5 := by
sorry

end digit_sum_in_multiplication_l2363_236352


namespace arithmetic_sequence_sum_l2363_236377

/-- An arithmetic sequence is a sequence where the difference between
    any two consecutive terms is constant. -/
def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

/-- Given an arithmetic sequence a_n where a_1 + a_2 = 10 and a_4 = a_3 + 2,
    prove that a_3 + a_4 = 18. -/
theorem arithmetic_sequence_sum (a : ℕ → ℝ) 
  (h_arith : is_arithmetic_sequence a)
  (h_sum : a 1 + a 2 = 10)
  (h_diff : a 4 = a 3 + 2) :
  a 3 + a 4 = 18 := by
  sorry

end arithmetic_sequence_sum_l2363_236377


namespace divisibility_of_polynomial_l2363_236314

theorem divisibility_of_polynomial (x : ℤ) :
  (x^2 + 1) * (x^8 - x^6 + x^4 - x^2 + 1) = x^10 + 1 →
  ∃ k : ℤ, x^2030 + 1 = k * (x^8 - x^6 + x^4 - x^2 + 1) :=
by sorry

end divisibility_of_polynomial_l2363_236314


namespace integer_fraction_pairs_l2363_236305

theorem integer_fraction_pairs : 
  ∀ a b : ℕ+, 
    (∃ k l : ℤ, (a.val^2 + b.val : ℤ) = k * (b.val^2 - a.val) ∧ 
                (b.val^2 + a.val : ℤ) = l * (a.val^2 - b.val)) →
    ((a = 2 ∧ b = 2) ∨ (a = 3 ∧ b = 3) ∨ (a = 1 ∧ b = 2) ∨ (a = 2 ∧ b = 3)) :=
by sorry

end integer_fraction_pairs_l2363_236305


namespace orange_juice_bottles_l2363_236342

theorem orange_juice_bottles (orange_price apple_price total_bottles total_cost : ℚ) 
  (h1 : orange_price = 70/100)
  (h2 : apple_price = 60/100)
  (h3 : total_bottles = 70)
  (h4 : total_cost = 4620/100) :
  ∃ (orange_bottles : ℚ), 
    orange_bottles * orange_price + (total_bottles - orange_bottles) * apple_price = total_cost ∧ 
    orange_bottles = 42 := by
  sorry

end orange_juice_bottles_l2363_236342


namespace projection_problem_l2363_236396

/-- Given that the projection of [2, 5] onto w is [2/5, -1/5],
    prove that the projection of [3, 2] onto w is [8/5, -4/5] -/
theorem projection_problem (w : ℝ × ℝ) :
  let v₁ : ℝ × ℝ := (2, 5)
  let v₂ : ℝ × ℝ := (3, 2)
  let proj₁ : ℝ × ℝ := (2/5, -1/5)
  (∃ (k : ℝ), w = k • proj₁) →
  (v₁ • w / (w • w)) • w = proj₁ →
  (v₂ • w / (w • w)) • w = (8/5, -4/5) := by
sorry

end projection_problem_l2363_236396


namespace sum_of_squares_and_products_l2363_236393

theorem sum_of_squares_and_products : (3 + 5)^2 + (3^2 + 5^2 + 3*5) = 113 := by
  sorry

end sum_of_squares_and_products_l2363_236393


namespace min_handshakes_theorem_l2363_236374

/-- Represents the number of people at the conference -/
def num_people : ℕ := 30

/-- Represents the minimum number of handshakes per person -/
def min_handshakes_per_person : ℕ := 3

/-- Calculates the minimum number of handshakes for the given conditions -/
def min_total_handshakes : ℕ :=
  (num_people * min_handshakes_per_person) / 2

/-- Theorem stating that the minimum number of handshakes is 45 -/
theorem min_handshakes_theorem :
  min_total_handshakes = 45 := by sorry

end min_handshakes_theorem_l2363_236374


namespace quadratic_solution_sum_l2363_236361

theorem quadratic_solution_sum (p q : ℝ) : 
  (∀ x : ℂ, (5 * x^2 + 7 = 2 * x - 6) ↔ (x = p + q * I ∨ x = p - q * I)) →
  p + q^2 = 69/25 := by
sorry

end quadratic_solution_sum_l2363_236361


namespace red_face_probability_l2363_236381

def large_cube_edge : ℕ := 6
def small_cube_edge : ℕ := 1

def total_small_cubes : ℕ := large_cube_edge ^ 3

def corner_cubes : ℕ := 8
def edge_cubes : ℕ := 4 * 12
def face_cubes : ℕ := 4 * 6

def red_faced_cubes : ℕ := corner_cubes + edge_cubes + face_cubes

theorem red_face_probability :
  (red_faced_cubes : ℚ) / total_small_cubes = 10 / 27 := by sorry

end red_face_probability_l2363_236381


namespace boat_rental_solutions_l2363_236363

theorem boat_rental_solutions :
  ∀ (x y : ℕ),
    12 * x + 5 * y = 99 →
    ((x = 2 ∧ y = 15) ∨ (x = 7 ∧ y = 3)) :=
by sorry

end boat_rental_solutions_l2363_236363


namespace cans_per_bag_l2363_236322

theorem cans_per_bag (total_cans : ℕ) (total_bags : ℕ) (h1 : total_cans = 42) (h2 : total_bags = 7) :
  total_cans / total_bags = 6 := by
  sorry

end cans_per_bag_l2363_236322


namespace weed_eater_string_cost_is_seven_l2363_236359

-- Define the number of lawnmower blades
def num_blades : ℕ := 4

-- Define the cost per blade in dollars
def cost_per_blade : ℕ := 8

-- Define the total spent on supplies in dollars
def total_spent : ℕ := 39

-- Define the cost of the weed eater string
def weed_eater_string_cost : ℕ := total_spent - (num_blades * cost_per_blade)

-- Theorem statement
theorem weed_eater_string_cost_is_seven :
  weed_eater_string_cost = 7 := by
  sorry

end weed_eater_string_cost_is_seven_l2363_236359


namespace find_number_to_add_l2363_236303

theorem find_number_to_add : ∃ x : ℝ, (5 * 12) / (180 / 3) + x = 71 := by
  sorry

end find_number_to_add_l2363_236303


namespace cube_face_sum_l2363_236364

theorem cube_face_sum (a₁ a₂ a₃ a₄ a₅ a₆ : ℕ+) : 
  (a₁ * a₂ * a₅ + a₂ * a₃ * a₅ + a₃ * a₄ * a₅ + a₄ * a₁ * a₅ +
   a₁ * a₂ * a₆ + a₂ * a₃ * a₆ + a₃ * a₄ * a₆ + a₄ * a₁ * a₆ = 70) →
  (a₁ + a₂ + a₃ + a₄ + a₅ + a₆ : ℕ) = 14 := by
sorry

end cube_face_sum_l2363_236364


namespace lcm_four_eight_l2363_236329

theorem lcm_four_eight : ∀ n : ℕ,
  (∃ m : ℕ, 4 ∣ m ∧ 8 ∣ m ∧ n ∣ m) →
  n ≥ 8 →
  Nat.lcm 4 8 = 8 :=
by sorry

end lcm_four_eight_l2363_236329


namespace first_jump_over_2km_l2363_236335

def jump_sequence (n : ℕ) : ℕ :=
  2 * 3^(n - 1)

theorem first_jump_over_2km :
  (∀ k < 8, jump_sequence k ≤ 2000) ∧ jump_sequence 8 > 2000 :=
sorry

end first_jump_over_2km_l2363_236335


namespace earnings_difference_l2363_236365

def car_price : ℕ := 5200
def inspection_cost : ℕ := car_price / 10
def headlight_cost : ℕ := 80
def tire_cost : ℕ := 3 * headlight_cost

def first_offer_earnings : ℕ := car_price - inspection_cost
def second_offer_earnings : ℕ := car_price - (headlight_cost + tire_cost)

theorem earnings_difference : second_offer_earnings - first_offer_earnings = 200 := by
  sorry

end earnings_difference_l2363_236365


namespace circle_point_distance_sum_l2363_236345

/-- Given a circle with diameter AB and radius R, and a tangent AT at point A,
    prove that a point M on the circle satisfying the condition that the sum of
    its distances to lines AB and AT is l exists if and only if l ≤ R(√2 + 1). -/
theorem circle_point_distance_sum (R l : ℝ) : 
  ∃ (M : ℝ × ℝ), 
    (M.1^2 + M.2^2 = R^2) ∧ 
    (M.1 + M.2 = l) ↔ 
    l ≤ R * (Real.sqrt 2 + 1) := by
  sorry

end circle_point_distance_sum_l2363_236345


namespace regular_tetrahedron_inradius_l2363_236344

/-- A regular tetrahedron -/
structure RegularTetrahedron where
  /-- The altitude of the regular tetrahedron -/
  altitude : ℝ
  /-- The inradius of the regular tetrahedron -/
  inradius : ℝ

/-- The inradius of a regular tetrahedron is one fourth of its altitude -/
theorem regular_tetrahedron_inradius (t : RegularTetrahedron) :
  t.inradius = (1 / 4) * t.altitude := by
  sorry

end regular_tetrahedron_inradius_l2363_236344


namespace inequality_proof_l2363_236392

theorem inequality_proof (a b : ℝ) (ha : a < 0) (hb : -1 < b ∧ b < 0) : a < a * b^2 ∧ a * b^2 < a * b := by
  sorry

end inequality_proof_l2363_236392


namespace range_of_m_l2363_236372

theorem range_of_m (m : ℝ) : 
  (|m + 3| = m + 3) →
  (|3*m + 9| ≥ 4*m - 3 ↔ -3 ≤ m ∧ m ≤ 12) :=
by sorry

end range_of_m_l2363_236372


namespace seven_x_minus_three_y_equals_thirteen_l2363_236320

theorem seven_x_minus_three_y_equals_thirteen 
  (x y : ℝ) 
  (h1 : 4 * x + y = 8) 
  (h2 : 3 * x - 4 * y = 5) : 
  7 * x - 3 * y = 13 := by
sorry

end seven_x_minus_three_y_equals_thirteen_l2363_236320


namespace min_sum_of_distances_min_sum_of_distances_achievable_l2363_236309

theorem min_sum_of_distances (x y z : ℝ) :
  Real.sqrt (x^2 + y^2 + z^2) + Real.sqrt ((x+1)^2 + (y-2)^2 + (z-1)^2) ≥ Real.sqrt 6 :=
by sorry

theorem min_sum_of_distances_achievable :
  ∃ (x y z : ℝ), Real.sqrt (x^2 + y^2 + z^2) + Real.sqrt ((x+1)^2 + (y-2)^2 + (z-1)^2) = Real.sqrt 6 :=
by sorry

end min_sum_of_distances_min_sum_of_distances_achievable_l2363_236309


namespace window_wood_strip_width_l2363_236390

/-- Represents the dimensions of a glass piece in centimeters -/
structure GlassDimensions where
  width : ℝ
  height : ℝ

/-- Represents the configuration of a window -/
structure WindowConfig where
  glassDimensions : GlassDimensions
  woodStripWidth : ℝ

/-- Calculates the total area of glass in the window -/
def totalGlassArea (config : WindowConfig) : ℝ :=
  4 * config.glassDimensions.width * config.glassDimensions.height

/-- Calculates the total area of the window -/
def totalWindowArea (config : WindowConfig) : ℝ :=
  (2 * config.glassDimensions.width + 3 * config.woodStripWidth) *
  (2 * config.glassDimensions.height + 3 * config.woodStripWidth)

/-- Theorem: If the total area of glass equals the total area of wood,
    then the wood strip width is 20/3 cm -/
theorem window_wood_strip_width
  (config : WindowConfig)
  (h1 : config.glassDimensions.width = 30)
  (h2 : config.glassDimensions.height = 20)
  (h3 : totalGlassArea config = totalWindowArea config - totalGlassArea config) :
  config.woodStripWidth = 20 / 3 := by
  sorry

end window_wood_strip_width_l2363_236390


namespace cellar_water_pumping_time_l2363_236327

/-- Calculates the time needed to pump out water from a flooded cellar. -/
theorem cellar_water_pumping_time
  (length : ℝ)
  (width : ℝ)
  (depth : ℝ)
  (num_pumps : ℕ)
  (pump_rate : ℝ)
  (water_density : ℝ)
  (h_length : length = 30)
  (h_width : width = 40)
  (h_depth : depth = 2)
  (h_num_pumps : num_pumps = 4)
  (h_pump_rate : pump_rate = 10)
  (h_water_density : water_density = 7.5) :
  (length * width * depth * water_density) / (num_pumps * pump_rate) = 450 :=
sorry

end cellar_water_pumping_time_l2363_236327


namespace sin_cos_sum_identity_l2363_236373

theorem sin_cos_sum_identity : 
  Real.sin (70 * π / 180) * Real.sin (10 * π / 180) + 
  Real.cos (10 * π / 180) * Real.cos (70 * π / 180) = 1 / 2 := by
  sorry

end sin_cos_sum_identity_l2363_236373


namespace large_cube_edge_is_one_meter_l2363_236347

/-- The edge length of a cubical box that can contain a given number of smaller cubes -/
def large_cube_edge_length (small_cube_edge : ℝ) (num_small_cubes : ℝ) : ℝ :=
  (small_cube_edge^3 * num_small_cubes)^(1/3)

/-- Theorem: The edge length of a cubical box that can contain 999.9999999999998 cubes 
    with 10 cm edge length is 1 meter -/
theorem large_cube_edge_is_one_meter :
  large_cube_edge_length 0.1 999.9999999999998 = 1 := by
  sorry

end large_cube_edge_is_one_meter_l2363_236347


namespace remainder_three_pow_2040_mod_5_l2363_236357

theorem remainder_three_pow_2040_mod_5 : 3^2040 % 5 = 1 := by
  sorry

end remainder_three_pow_2040_mod_5_l2363_236357


namespace max_red_points_is_13_l2363_236368

/-- Represents a point on the circle -/
inductive Point
| Red : ℕ → Point  -- Red point with number of connections
| Blue : Point

/-- The configuration of points on the circle -/
structure CircleConfig where
  points : Finset Point
  red_count : ℕ
  blue_count : ℕ
  total_count : ℕ
  total_is_25 : total_count = 25
  total_is_sum : total_count = red_count + blue_count
  unique_connections : ∀ p q : Point, p ∈ points → q ∈ points → 
    p ≠ q → (∃ n m : ℕ, p = Point.Red n ∧ q = Point.Red m) → n ≠ m

/-- The maximum number of red points possible -/
def max_red_points : ℕ := 13

/-- Theorem stating that the maximum number of red points is 13 -/
theorem max_red_points_is_13 (config : CircleConfig) : 
  config.red_count ≤ max_red_points :=
sorry

end max_red_points_is_13_l2363_236368


namespace remainder_after_adding_2025_l2363_236315

theorem remainder_after_adding_2025 (n : ℤ) (h : n % 5 = 3) : (n + 2025) % 5 = 3 := by
  sorry

end remainder_after_adding_2025_l2363_236315


namespace painting_job_completion_time_l2363_236382

/-- Represents the time in hours it takes to paint a wall -/
structure PaintingTime where
  hours : ℚ
  is_positive : 0 < hours

/-- Represents a painter's rate in terms of wall painted per hour -/
def painting_rate (time : PaintingTime) : ℚ :=
  1 / time.hours

theorem painting_job_completion_time 
  (gina_time : PaintingTime)
  (tom_time : PaintingTime)
  (joint_work_time : ℚ)
  (h_gina : gina_time.hours = 3)
  (h_tom : tom_time.hours = 5)
  (h_joint : joint_work_time = 2)
  : ∃ (t : ℚ), t = 20/3 ∧ 
    (painting_rate gina_time + painting_rate tom_time) * joint_work_time + 
    painting_rate tom_time * (t - joint_work_time) = 1 :=
sorry

end painting_job_completion_time_l2363_236382


namespace lilies_count_l2363_236310

/-- The cost of a single chrysanthemum in yuan -/
def chrysanthemum_cost : ℕ := 3

/-- The cost of a single lily in yuan -/
def lily_cost : ℕ := 4

/-- The total amount of money Mom wants to spend in yuan -/
def total_money : ℕ := 100

/-- The number of chrysanthemums Mom wants to buy -/
def chrysanthemums_to_buy : ℕ := 16

/-- The number of lilies that can be bought with the remaining money -/
def lilies_to_buy : ℕ := (total_money - chrysanthemum_cost * chrysanthemums_to_buy) / lily_cost

theorem lilies_count : lilies_to_buy = 13 := by
  sorry

end lilies_count_l2363_236310


namespace school_population_l2363_236338

/-- The number of students that each classroom holds -/
def students_per_classroom : ℕ := 30

/-- The number of classrooms needed -/
def number_of_classrooms : ℕ := 13

/-- The total number of students in the school -/
def total_students : ℕ := students_per_classroom * number_of_classrooms

theorem school_population : total_students = 390 := by
  sorry

end school_population_l2363_236338


namespace existence_of_digit_in_power_of_two_l2363_236383

theorem existence_of_digit_in_power_of_two (k d : ℕ) (h1 : k > 1) (h2 : d < 9) :
  ∃ n : ℕ, (2^n : ℕ) % 10^k = d := by
  sorry

end existence_of_digit_in_power_of_two_l2363_236383
