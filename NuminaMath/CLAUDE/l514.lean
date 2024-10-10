import Mathlib

namespace polynomial_factorization_l514_51465

theorem polynomial_factorization (x : ℝ) : 
  x^6 - 4*x^4 + 6*x^2 - 4 = (x-1)*(x+1)*(x-Real.sqrt 2)^2*(x+Real.sqrt 2)^2 := by
  sorry

end polynomial_factorization_l514_51465


namespace professional_doctors_percentage_l514_51434

theorem professional_doctors_percentage 
  (leaders_percent : ℝ)
  (nurses_percent : ℝ)
  (h1 : leaders_percent = 4)
  (h2 : nurses_percent = 56)
  (h3 : ∃ (doctors_percent psychologists_percent : ℝ), 
    leaders_percent + nurses_percent + doctors_percent + psychologists_percent = 100) :
  ∃ (doctors_percent : ℝ), doctors_percent = 40 := by
  sorry

end professional_doctors_percentage_l514_51434


namespace other_solution_of_quadratic_equation_l514_51442

theorem other_solution_of_quadratic_equation :
  let equation := fun (x : ℚ) => 72 * x^2 + 43 = 113 * x - 12
  equation (3/8) → ∃ x : ℚ, x ≠ 3/8 ∧ equation x ∧ x = 43/36 := by
  sorry

end other_solution_of_quadratic_equation_l514_51442


namespace bicycle_stand_stability_l514_51475

/-- A triangle is a geometric shape with three sides and three angles. -/
structure Triangle where
  -- We don't need to define the specifics of a triangle for this problem

/-- A bicycle stand is a device that supports a bicycle. -/
structure BicycleStand where
  -- We don't need to define the specifics of a bicycle stand for this problem

/-- Stability is a property that allows an object to remain balanced and resist toppling. -/
def Stability : Prop := sorry

/-- A property that allows an object to stand firmly on the ground. -/
def AllowsToStandFirmly (prop : Prop) : Prop := sorry

theorem bicycle_stand_stability (t : Triangle) (s : BicycleStand) :
  AllowsToStandFirmly Stability := by sorry

end bicycle_stand_stability_l514_51475


namespace gravel_path_rate_l514_51492

/-- Calculates the rate per square meter for gravelling a path around a rectangular plot. -/
theorem gravel_path_rate (length width path_width total_cost : ℝ) 
  (h1 : length = 110)
  (h2 : width = 65)
  (h3 : path_width = 2.5)
  (h4 : total_cost = 680) : 
  total_cost / ((length * width) - ((length - 2 * path_width) * (width - 2 * path_width))) = 0.8 := by
  sorry

end gravel_path_rate_l514_51492


namespace max_parts_quadratic_trinomials_l514_51491

/-- The maximum number of parts into which the coordinate plane can be divided by n quadratic trinomials -/
def max_parts (n : ℕ) : ℕ := n^2 + 1

/-- Theorem: The maximum number of parts into which the coordinate plane can be divided by n quadratic trinomials is n^2 + 1 -/
theorem max_parts_quadratic_trinomials (n : ℕ) :
  max_parts n = n^2 + 1 := by
  sorry

end max_parts_quadratic_trinomials_l514_51491


namespace unique_solution_l514_51445

/-- Represents a cell in the 5x5 grid --/
structure Cell :=
  (row : Fin 5)
  (col : Fin 5)

/-- Represents the 5x5 grid --/
def Grid := Cell → Fin 5

/-- Check if two cells are in the same row --/
def same_row (c1 c2 : Cell) : Prop := c1.row = c2.row

/-- Check if two cells are in the same column --/
def same_column (c1 c2 : Cell) : Prop := c1.col = c2.col

/-- Check if two cells are in the same block --/
def same_block (c1 c2 : Cell) : Prop :=
  (c1.row / 3 = c2.row / 3) ∧ (c1.col / 3 = c2.col / 3)

/-- Check if two cells are diagonally adjacent --/
def diag_adjacent (c1 c2 : Cell) : Prop :=
  (c1.row = c2.row + 1 ∧ c1.col = c2.col + 1) ∨
  (c1.row = c2.row + 1 ∧ c1.col = c2.col - 1) ∨
  (c1.row = c2.row - 1 ∧ c1.col = c2.col + 1) ∨
  (c1.row = c2.row - 1 ∧ c1.col = c2.col - 1)

/-- Check if a grid is valid according to the rules --/
def valid_grid (g : Grid) : Prop :=
  ∀ c1 c2 : Cell, c1 ≠ c2 →
    (same_row c1 c2 ∨ same_column c1 c2 ∨ same_block c1 c2 ∨ diag_adjacent c1 c2) →
    g c1 ≠ g c2

/-- The unique solution to the puzzle --/
theorem unique_solution (g : Grid) (h : valid_grid g) :
  (g ⟨0, 0⟩ = 5) ∧ (g ⟨0, 1⟩ = 3) ∧ (g ⟨0, 2⟩ = 1) ∧ (g ⟨0, 3⟩ = 2) ∧ (g ⟨0, 4⟩ = 4) := by
  sorry

end unique_solution_l514_51445


namespace student_lecture_assignment_l514_51457

/-- The number of ways to assign students to lectures -/
def assignment_count (num_students : ℕ) (num_lectures : ℕ) : ℕ :=
  num_lectures ^ num_students

/-- Theorem: The number of ways to assign 5 students to 3 lectures is 243 -/
theorem student_lecture_assignment :
  assignment_count 5 3 = 243 := by
  sorry

end student_lecture_assignment_l514_51457


namespace crayons_added_l514_51497

theorem crayons_added (initial : ℕ) (final : ℕ) (added : ℕ) : 
  initial = 7 → final = 10 → initial + added = final → added = 3 := by
  sorry

end crayons_added_l514_51497


namespace corn_kernel_weight_theorem_l514_51499

/-- Calculates the total weight of corn kernels after shucking and accounting for losses -/
def corn_kernel_weight (
  ears_per_stalk : ℕ)
  (total_stalks : ℕ)
  (bad_ear_percentage : ℚ)
  (kernel_distribution : List (ℚ × ℕ))
  (kernel_weight : ℚ)
  (lost_kernel_percentage : ℚ) : ℚ :=
  let total_ears := ears_per_stalk * total_stalks
  let good_ears := total_ears - (bad_ear_percentage * total_ears).floor
  let total_kernels := (kernel_distribution.map (fun (p, k) => 
    ((p * good_ears).floor * k))).sum
  let kernels_after_loss := total_kernels - 
    (lost_kernel_percentage * total_kernels).floor
  kernels_after_loss * kernel_weight

/-- The total weight of corn kernels is approximately 18527.9 grams -/
theorem corn_kernel_weight_theorem : 
  ∃ (ε : ℚ), ε > 0 ∧ ε < 0.1 ∧ 
  |corn_kernel_weight 4 108 (1/5) 
    [(3/5, 500), (3/10, 600), (1/10, 700)] (1/10) (3/200) - 18527.9| < ε :=
sorry

end corn_kernel_weight_theorem_l514_51499


namespace prob_at_least_one_boy_and_girl_l514_51460

/-- The probability of having a boy or a girl -/
def child_probability : ℚ := 1 / 2

/-- The number of children in the family -/
def family_size : ℕ := 4

/-- The probability of having at least one boy and one girl in a family with four children -/
theorem prob_at_least_one_boy_and_girl : 
  (1 : ℚ) - (child_probability ^ family_size + child_probability ^ family_size) = 7 / 8 := by
  sorry

end prob_at_least_one_boy_and_girl_l514_51460


namespace triangle_angle_leq_60_l514_51485

/-- Theorem: In any triangle, at least one angle is less than or equal to 60 degrees. -/
theorem triangle_angle_leq_60 (A B C : ℝ) (h_triangle : A + B + C = 180) :
  A ≤ 60 ∨ B ≤ 60 ∨ C ≤ 60 := by
  sorry

end triangle_angle_leq_60_l514_51485


namespace expression_evaluation_l514_51419

theorem expression_evaluation : (-5)^5 / 5^3 + 3^4 - 6^1 = 50 := by
  sorry

end expression_evaluation_l514_51419


namespace downstream_distance_is_35_l514_51489

-- Define the given constants
def man_speed : ℝ := 5.5
def upstream_distance : ℝ := 20
def swim_time : ℝ := 5

-- Define the theorem
theorem downstream_distance_is_35 :
  let stream_speed := (man_speed - upstream_distance / swim_time) / 2
  let downstream_distance := (man_speed + stream_speed) * swim_time
  downstream_distance = 35 := by sorry

end downstream_distance_is_35_l514_51489


namespace consecutive_integers_product_360_l514_51455

theorem consecutive_integers_product_360 :
  ∃ (n m : ℤ), 
    n * (n + 1) = 360 ∧ 
    (m - 1) * m * (m + 1) = 360 ∧ 
    n + (n + 1) + (m - 1) + m + (m + 1) = 55 := by
  sorry

end consecutive_integers_product_360_l514_51455


namespace min_value_of_expression_l514_51466

theorem min_value_of_expression (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : 1/a + 1/b = 1) :
  ∃ (min : ℝ), min = 4 ∧ ∀ (x : ℝ), x = 1/(a-1) + 4/(b-1) → x ≥ min :=
sorry

end min_value_of_expression_l514_51466


namespace completing_square_quadratic_l514_51416

theorem completing_square_quadratic (x : ℝ) : 
  x^2 - 6*x - 7 = 0 ↔ (x - 3)^2 = 16 := by
  sorry

end completing_square_quadratic_l514_51416


namespace circle_equation_l514_51451

/-- A circle in the 2D plane -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- Check if a point lies on a circle -/
def pointOnCircle (c : Circle) (p : ℝ × ℝ) : Prop :=
  let (x, y) := p
  let (h, k) := c.center
  (x - h)^2 + (y - k)^2 = c.radius^2

/-- Check if a circle is tangent to the y-axis -/
def tangentToYAxis (c : Circle) : Prop :=
  c.center.1 = c.radius

/-- Check if a point lies on the line x - 3y = 0 -/
def pointOnLine (p : ℝ × ℝ) : Prop :=
  p.1 - 3 * p.2 = 0

theorem circle_equation (c : Circle) :
  tangentToYAxis c ∧ 
  pointOnLine c.center ∧ 
  pointOnCircle c (6, 1) →
  c.center = (3, 1) ∧ c.radius = 3 :=
by sorry

#check circle_equation

end circle_equation_l514_51451


namespace sprint_tournament_races_l514_51424

theorem sprint_tournament_races (total_sprinters : ℕ) (lanes_per_race : ℕ) : 
  total_sprinters = 320 →
  lanes_per_race = 8 →
  (∃ (num_races : ℕ), 
    num_races = 46 ∧
    num_races = (total_sprinters - 1) / (lanes_per_race - 1) + 
      (if (total_sprinters - 1) % (lanes_per_race - 1) = 0 then 0 else 1)) :=
by
  sorry

#check sprint_tournament_races

end sprint_tournament_races_l514_51424


namespace max_cuttable_strings_l514_51468

/-- Represents a volleyball net as a graph --/
structure VolleyballNet where
  rows : Nat
  cols : Nat

/-- Calculates the number of nodes in the net --/
def VolleyballNet.nodeCount (net : VolleyballNet) : Nat :=
  (net.rows + 1) * (net.cols + 1)

/-- Calculates the total number of strings in the net --/
def VolleyballNet.stringCount (net : VolleyballNet) : Nat :=
  net.rows * (net.cols + 1) + (net.rows + 1) * net.cols

/-- Theorem: Maximum number of cuttable strings in a 10x100 volleyball net --/
theorem max_cuttable_strings (net : VolleyballNet) 
  (h_rows : net.rows = 10) (h_cols : net.cols = 100) : 
  net.stringCount - (net.nodeCount - 1) = 1000 := by
  sorry

end max_cuttable_strings_l514_51468


namespace x_minus_y_equals_four_l514_51427

theorem x_minus_y_equals_four (x y : ℝ) 
  (sum_eq : x + y = 10) 
  (diff_squares_eq : x^2 - y^2 = 40) : 
  x - y = 4 := by
sorry

end x_minus_y_equals_four_l514_51427


namespace x_gt_y_iff_exp_and_cbrt_l514_51420

theorem x_gt_y_iff_exp_and_cbrt (x y : ℝ) : 
  x > y ↔ (Real.exp x > Real.exp y ∧ x^(1/3) > y^(1/3)) :=
sorry

end x_gt_y_iff_exp_and_cbrt_l514_51420


namespace brick_length_is_125_l514_51473

/-- Represents the dimensions of a rectangular object in centimeters -/
structure Dimensions where
  length : ℝ
  width : ℝ
  height : ℝ

/-- Calculates the volume of a rectangular object given its dimensions -/
def volume (d : Dimensions) : ℝ :=
  d.length * d.width * d.height

/-- The dimensions of the wall in centimeters -/
def wall_dimensions : Dimensions :=
  { length := 800, width := 600, height := 22.5 }

/-- The partial dimensions of a brick in centimeters -/
def brick_partial_dimensions (x : ℝ) : Dimensions :=
  { length := x, width := 11.25, height := 6 }

/-- The number of bricks needed to build the wall -/
def number_of_bricks : ℕ := 1280

/-- Theorem stating that the length of each brick is 125 cm -/
theorem brick_length_is_125 : 
  ∃ x : ℝ, x = 125 ∧ 
  volume wall_dimensions = (number_of_bricks : ℝ) * volume (brick_partial_dimensions x) :=
sorry

end brick_length_is_125_l514_51473


namespace dodecagon_pie_trim_l514_51412

theorem dodecagon_pie_trim (d : ℝ) (h : d = 8) : ∃ (a b : ℤ),
  (π * (d / 2)^2 - 3 * (d / 2)^2 = a * π - b) ∧ (a + b = 64) := by
  sorry

end dodecagon_pie_trim_l514_51412


namespace geometric_sequence_a7_l514_51452

/-- A geometric sequence with positive terms -/
def GeometricSequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, r > 0 ∧ ∀ n : ℕ, a (n + 1) = r * a n

theorem geometric_sequence_a7 (a : ℕ → ℝ) :
  GeometricSequence a →
  (∀ n : ℕ, a n > 0) →
  a 3 * a 11 = 16 →
  a 7 = 4 := by
  sorry

end geometric_sequence_a7_l514_51452


namespace number_exceeding_fraction_l514_51462

theorem number_exceeding_fraction (x : ℚ) : 
  x = (3/8) * x + 35 → x = 56 := by
  sorry

end number_exceeding_fraction_l514_51462


namespace quadratic_roots_condition_l514_51454

-- Define the quadratic equation
def quadratic (m : ℝ) (x : ℝ) : ℝ := (m + 3) * x^2 - 4 * m * x + 2 * m - 1

-- Define the condition for roots having opposite signs
def opposite_signs (x₁ x₂ : ℝ) : Prop := x₁ * x₂ < 0

-- Define the condition for the absolute value of the negative root being greater than the positive root
def negative_root_greater (x₁ x₂ : ℝ) : Prop := 
  (x₁ < 0 ∧ x₂ > 0 ∧ abs x₁ > x₂) ∨ (x₂ < 0 ∧ x₁ > 0 ∧ abs x₂ > x₁)

-- The main theorem
theorem quadratic_roots_condition (m : ℝ) :
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ 
    quadratic m x₁ = 0 ∧ 
    quadratic m x₂ = 0 ∧ 
    opposite_signs x₁ x₂ ∧ 
    negative_root_greater x₁ x₂) →
  -3 < m ∧ m < 0 :=
by sorry

end quadratic_roots_condition_l514_51454


namespace train_bridge_crossing_time_l514_51403

/-- Proves that a train with given length and speed takes the specified time to cross a bridge of given length -/
theorem train_bridge_crossing_time 
  (train_length : Real) 
  (train_speed_kmh : Real) 
  (bridge_length : Real) : 
  train_length = 170 → 
  train_speed_kmh = 45 → 
  bridge_length = 205 → 
  (train_length + bridge_length) / (train_speed_kmh * 1000 / 3600) = 30 := by
  sorry


end train_bridge_crossing_time_l514_51403


namespace system_solution_exists_and_unique_l514_51446

theorem system_solution_exists_and_unique :
  ∃! (x y z : ℝ), 
    8 * (x^3 + y^3 + z^3) = 73 ∧
    2 * (x^2 + y^2 + z^2) = 3 * (x*y + y*z + z*x) ∧
    x * y * z = 1 ∧
    x = 1 ∧ y = 2 ∧ z = (1/2 : ℝ) := by
  sorry

end system_solution_exists_and_unique_l514_51446


namespace contrapositive_equivalence_l514_51486

theorem contrapositive_equivalence (x : ℝ) :
  (¬(x^2 < 1) → ¬(-1 < x ∧ x < 1)) ↔ ((x ≥ 1 ∨ x ≤ -1) → x^2 ≥ 1) :=
by sorry

end contrapositive_equivalence_l514_51486


namespace favorite_sports_survey_l514_51498

/-- Given a survey of students' favorite sports, prove the number of students who like chess or basketball. -/
theorem favorite_sports_survey (total_students : ℕ) 
  (basketball_percent chess_percent soccer_percent badminton_percent : ℚ) :
  basketball_percent = 40/100 →
  chess_percent = 10/100 →
  soccer_percent = 28/100 →
  badminton_percent = 22/100 →
  basketball_percent + chess_percent + soccer_percent + badminton_percent = 1 →
  total_students = 250 →
  ⌊(basketball_percent + chess_percent) * total_students⌋ = 125 := by
  sorry


end favorite_sports_survey_l514_51498


namespace complex_equation_solution_l514_51431

/-- Given that i² = -1, prove that w = -2i/3 is the solution to the equation 3 - iw = 1 + 2iw -/
theorem complex_equation_solution (i : ℂ) (h : i^2 = -1) :
  let w : ℂ := -2*i/3
  3 - i*w = 1 + 2*i*w := by sorry

end complex_equation_solution_l514_51431


namespace swim_club_members_l514_51470

theorem swim_club_members : 
  ∀ (total_members passed_members not_passed_members : ℕ),
  passed_members = (30 * total_members) / 100 →
  not_passed_members = 70 →
  not_passed_members = total_members - passed_members →
  total_members = 100 := by
sorry

end swim_club_members_l514_51470


namespace train_passing_platform_l514_51482

/-- Calculates the time taken for a train to pass a platform -/
theorem train_passing_platform 
  (train_length : ℝ) 
  (platform_length : ℝ) 
  (pole_passing_time : ℝ) 
  (h1 : train_length = 500)
  (h2 : platform_length = 500)
  (h3 : pole_passing_time = 50) :
  (train_length + platform_length) / (train_length / pole_passing_time) = 100 :=
by sorry

end train_passing_platform_l514_51482


namespace complex_subtraction_equality_l514_51435

theorem complex_subtraction_equality : ((1 - 1) - 1) - ((1 - (1 - 1))) = -2 := by
  sorry

end complex_subtraction_equality_l514_51435


namespace same_month_same_gender_exists_l514_51453

/-- Represents a student with their gender and birth month. -/
structure Student where
  gender : Bool  -- True for girl, False for boy
  birthMonth : Fin 12

/-- Theorem: In a class of 25 students, there must be at least two girls
    or two boys born in the same month. -/
theorem same_month_same_gender_exists (students : Finset Student)
    (h_count : students.card = 25) :
    (∃ (m : Fin 12), 2 ≤ (students.filter (fun s => s.gender ∧ s.birthMonth = m)).card) ∨
    (∃ (m : Fin 12), 2 ≤ (students.filter (fun s => ¬s.gender ∧ s.birthMonth = m)).card) :=
  sorry


end same_month_same_gender_exists_l514_51453


namespace fourth_root_over_seventh_root_of_seven_l514_51437

theorem fourth_root_over_seventh_root_of_seven (x : ℝ) (h : x > 0) :
  (x^(1/4)) / (x^(1/7)) = x^(3/28) :=
sorry

end fourth_root_over_seventh_root_of_seven_l514_51437


namespace quadratic_function_properties_l514_51484

-- Define the quadratic function f(x)
def f (b c x : ℝ) : ℝ := x^2 + b*x + c

-- State the theorem
theorem quadratic_function_properties :
  ∀ b c : ℝ,
  (f b c 1 = 0) →
  (f b c 3 = 0) →
  (b = -4 ∧ c = 3) ∧
  (∀ x y : ℝ, 2 < x → x < y → f (-4) 3 x < f (-4) 3 y) :=
by sorry

end quadratic_function_properties_l514_51484


namespace inequality_system_solution_set_l514_51471

theorem inequality_system_solution_set :
  ∀ x : ℝ, (x < -3 ∧ x < 2) ↔ x < -3 := by sorry

end inequality_system_solution_set_l514_51471


namespace average_percentage_decrease_l514_51494

theorem average_percentage_decrease (initial_price final_price : ℝ) 
  (h1 : initial_price = 800)
  (h2 : final_price = 578)
  (h3 : final_price = initial_price * (1 - x)^2)
  : x = 0.15 := by
  sorry

end average_percentage_decrease_l514_51494


namespace divisibility_implies_multiple_of_three_l514_51443

theorem divisibility_implies_multiple_of_three (a b : ℤ) : 
  (9 ∣ a^2 + a*b + b^2) → (3 ∣ a) ∧ (3 ∣ b) := by
  sorry

end divisibility_implies_multiple_of_three_l514_51443


namespace postage_calculation_l514_51474

/-- Calculates the postage for a letter given the base fee, additional fee per ounce, and weight -/
def calculatePostage (baseFee : ℚ) (additionalFeePerOunce : ℚ) (weight : ℚ) : ℚ :=
  baseFee + additionalFeePerOunce * (weight - 1)

/-- Theorem stating that the postage for a 5.3 ounce letter is $1.425 under the given fee structure -/
theorem postage_calculation :
  let baseFee : ℚ := 35 / 100  -- 35 cents in dollars
  let additionalFeePerOunce : ℚ := 25 / 100  -- 25 cents in dollars
  let weight : ℚ := 53 / 10  -- 5.3 ounces
  calculatePostage baseFee additionalFeePerOunce weight = 1425 / 1000 := by
  sorry

#eval calculatePostage (35/100) (25/100) (53/10)

end postage_calculation_l514_51474


namespace complex_equality_and_minimum_distance_l514_51411

open Complex

theorem complex_equality_and_minimum_distance (z : ℂ) :
  (abs z = abs (z + 1 + I)) →
  (∃ (a : ℝ), z = a + a * I) →
  (z = -1 - I) ∧
  (∃ (min_dist : ℝ), min_dist = Real.sqrt 2 ∧
    ∀ (w : ℂ), abs w = abs (w + 1 + I) → abs (w - (2 - I)) ≥ min_dist) :=
by sorry

end complex_equality_and_minimum_distance_l514_51411


namespace james_chocolate_sales_l514_51469

/-- Calculates the number of chocolate bars James sold this week -/
def chocolate_bars_sold_this_week (total : ℕ) (sold_last_week : ℕ) (to_sell : ℕ) : ℕ :=
  total - (sold_last_week + to_sell)

/-- Proves that James sold 2 chocolate bars this week -/
theorem james_chocolate_sales : chocolate_bars_sold_this_week 18 5 6 = 2 := by
  sorry

end james_chocolate_sales_l514_51469


namespace function_equation_solution_l514_51438

theorem function_equation_solution (f : ℤ → ℤ) :
  (∀ a b : ℤ, f (2 * a) + 2 * f b = f (f (a + b))) →
  (∃ c : ℤ, ∀ x : ℤ, f x = 0 ∨ f x = 2 * x + c) :=
by sorry

end function_equation_solution_l514_51438


namespace triple_composition_even_l514_51423

/-- A function f is even if f(-x) = f(x) for all x -/
def IsEven (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = f x

theorem triple_composition_even (f : ℝ → ℝ) (h : IsEven f) :
  ∀ x, f (f (f (-x))) = f (f (f x)) := by sorry

end triple_composition_even_l514_51423


namespace coefficient_of_sixth_power_l514_51444

theorem coefficient_of_sixth_power (x : ℝ) :
  ∃ (a₀ a₁ a₂ a₃ a₄ a₅ a₆ : ℝ),
    (2 - x)^6 = a₀ + a₁*(1+x) + a₂*(1+x)^2 + a₃*(1+x)^3 + a₄*(1+x)^4 + a₅*(1+x)^5 + a₆*(1+x)^6 ∧
    a₆ = 1 :=
by sorry

end coefficient_of_sixth_power_l514_51444


namespace homework_question_not_proposition_l514_51407

-- Define what a proposition is
def is_proposition (s : String) : Prop := 
  ∃ (b : Bool), (s = "true") ∨ (s = "false")

-- Define the statement in question
def homework_question : String :=
  "Have you finished your homework?"

-- Theorem to prove
theorem homework_question_not_proposition :
  ¬ (is_proposition homework_question) := by
  sorry

end homework_question_not_proposition_l514_51407


namespace log_xy_value_l514_51401

theorem log_xy_value (x y : ℝ) (h1 : Real.log (x^3 * y^2) = 2) (h2 : Real.log (x^2 * y^3) = 2) : 
  Real.log (x * y) = 4/5 := by
sorry

end log_xy_value_l514_51401


namespace equality_conditions_l514_51425

theorem equality_conditions (a b c : ℝ) : 
  ((a + (a * b * c) / (a - b * c + b)) / (b + (a * b * c) / (a - a * c + b)) = 
   (a - (a * b) / (a + 2 * b)) / (b - (a * b) / (2 * a + b)) ∧
   (a - (a * b) / (a + 2 * b)) / (b - (a * b) / (2 * a + b)) = 
   ((2 * a * b) / (a - b) + a) / ((2 * a * b) / (a - b) - b) ∧
   ((2 * a * b) / (a - b) + a) / ((2 * a * b) / (a - b) - b) = a / b) ↔
  (a = 0 ∧ b ≠ 0 ∧ c ≠ 1) :=
by sorry


end equality_conditions_l514_51425


namespace linear_function_not_in_first_quadrant_l514_51422

/-- A linear function f(x) = -x - 2 -/
def f (x : ℝ) : ℝ := -x - 2

/-- The first quadrant of the coordinate plane -/
def first_quadrant (x y : ℝ) : Prop := x > 0 ∧ y > 0

theorem linear_function_not_in_first_quadrant :
  ∀ x : ℝ, ¬(first_quadrant x (f x)) :=
by sorry

end linear_function_not_in_first_quadrant_l514_51422


namespace quadratic_function_continuous_l514_51439

/-- A quadratic function f(x) = ax^2 + bx + c is continuous at any point x ∈ ℝ,
    where a, b, and c are real constants. -/
theorem quadratic_function_continuous (a b c : ℝ) :
  Continuous (fun x : ℝ => a * x^2 + b * x + c) :=
sorry

end quadratic_function_continuous_l514_51439


namespace mice_eaten_in_decade_l514_51402

/-- Represents the number of weeks in a year -/
def weeksInYear : ℕ := 52

/-- Represents the eating frequency (in weeks) for the snake in its first year -/
def firstYearFrequency : ℕ := 4

/-- Represents the eating frequency (in weeks) for the snake in its second year -/
def secondYearFrequency : ℕ := 3

/-- Represents the eating frequency (in weeks) for the snake after its second year -/
def laterYearsFrequency : ℕ := 2

/-- Calculates the number of mice eaten in the first year -/
def miceEatenFirstYear : ℕ := weeksInYear / firstYearFrequency

/-- Calculates the number of mice eaten in the second year -/
def miceEatenSecondYear : ℕ := weeksInYear / secondYearFrequency

/-- Calculates the number of mice eaten in one year after the second year -/
def miceEatenPerLaterYear : ℕ := weeksInYear / laterYearsFrequency

/-- Represents the number of years in a decade -/
def yearsInDecade : ℕ := 10

/-- Theorem stating the total number of mice eaten over a decade -/
theorem mice_eaten_in_decade : 
  miceEatenFirstYear + miceEatenSecondYear + (yearsInDecade - 2) * miceEatenPerLaterYear = 238 := by
  sorry

end mice_eaten_in_decade_l514_51402


namespace triangle_inequality_expression_l514_51413

theorem triangle_inequality_expression (a b c : ℝ) 
  (h_positive : a > 0 ∧ b > 0 ∧ c > 0) 
  (h_triangle : a + b > c ∧ b + c > a ∧ c + a > b) : 
  a^2 - 2*a*b + b^2 - c^2 < 0 :=
by sorry

end triangle_inequality_expression_l514_51413


namespace correct_sampling_methods_l514_51496

/-- Represents different types of sampling methods -/
inductive SamplingMethod
  | SimpleRandom
  | Systematic
  | Stratified

/-- Represents a sampling technique -/
structure SamplingTechnique where
  method : SamplingMethod
  description : String

/-- Represents the survey conducted by the school -/
structure Survey where
  totalStudents : Nat
  technique1 : SamplingTechnique
  technique2 : SamplingTechnique

/-- The actual survey conducted by the school -/
def schoolSurvey : Survey :=
  { totalStudents := 200,
    technique1 := 
      { method := SamplingMethod.SimpleRandom,
        description := "Random selection of 20 students by the student council" },
    technique2 := 
      { method := SamplingMethod.Systematic,
        description := "Students numbered from 001 to 200, those with last digit 2 are selected" }
  }

/-- Theorem stating that the sampling methods are correctly identified -/
theorem correct_sampling_methods :
  schoolSurvey.technique1.method = SamplingMethod.SimpleRandom ∧
  schoolSurvey.technique2.method = SamplingMethod.Systematic :=
by sorry


end correct_sampling_methods_l514_51496


namespace sufficient_necessary_equivalence_l514_51477

theorem sufficient_necessary_equivalence (A B : Prop) :
  (A → B) ↔ (¬B → ¬A) :=
sorry

end sufficient_necessary_equivalence_l514_51477


namespace min_value_2sin_x_l514_51408

theorem min_value_2sin_x (x : Real) (h : π/3 ≤ x ∧ x ≤ 5*π/6) : 
  ∃ (y : Real), y = 2 * Real.sin x ∧ y ≥ 1 ∧ ∀ z, (∃ t, π/3 ≤ t ∧ t ≤ 5*π/6 ∧ z = 2 * Real.sin t) → y ≤ z :=
by sorry

end min_value_2sin_x_l514_51408


namespace ellipse_max_ratio_l514_51441

/-- Given an ellipse with semi-major axis a and semi-minor axis b, 
    where a > b > 0, prove that the maximum value of |FA|/|OH| is 1/4, 
    where F is the right focus, A is the right vertex, O is the center, 
    and H is the intersection of the right directrix with the x-axis. -/
theorem ellipse_max_ratio (a b : ℝ) (h : a > b ∧ b > 0) : 
  let e := Real.sqrt (1 - b^2 / a^2)  -- eccentricity
  ∃ (x y : ℝ), x^2 / a^2 + y^2 / b^2 = 1 ∧ 
    (∀ (x' y' : ℝ), x'^2 / a^2 + y'^2 / b^2 = 1 → 
      (a - a * e) / (a^2 / (a * e)) ≤ 1/4) ∧
    (a - a * e) / (a^2 / (a * e)) = 1/4 :=
sorry

end ellipse_max_ratio_l514_51441


namespace prob_all_co_captains_l514_51488

/-- Represents a math team with a certain number of students and co-captains -/
structure MathTeam where
  size : Nat
  coCaptains : Nat

/-- Calculates the probability of selecting all co-captains from a single team -/
def probAllCoCaptains (team : MathTeam) : Rat :=
  1 / (Nat.choose team.size 3)

/-- The set of math teams in the area -/
def mathTeams : List MathTeam := [
  { size := 6, coCaptains := 3 },
  { size := 8, coCaptains := 3 },
  { size := 9, coCaptains := 3 },
  { size := 10, coCaptains := 3 }
]

/-- The main theorem stating the probability of selecting all co-captains -/
theorem prob_all_co_captains : 
  (List.sum (mathTeams.map probAllCoCaptains) / mathTeams.length : Rat) = 53 / 3360 := by
  sorry

end prob_all_co_captains_l514_51488


namespace equal_intercept_line_equation_l514_51433

/-- A line passing through (1, 2) with equal intercepts on both coordinate axes -/
structure EqualInterceptLine where
  -- The slope-intercept form of the line: y = mx + b
  m : ℝ
  b : ℝ
  -- The line passes through (1, 2)
  passes_through : 2 = m * 1 + b
  -- The line has equal intercepts on both axes
  equal_intercepts : m ≠ -1 → b = b / m

/-- The equation of an EqualInterceptLine is either 2x - y = 0 or x + y - 3 = 0 -/
theorem equal_intercept_line_equation (l : EqualInterceptLine) :
  (l.m = 2 ∧ l.b = 0) ∨ (l.m = -1 ∧ l.b = 3) := by
  sorry

end equal_intercept_line_equation_l514_51433


namespace no_real_roots_l514_51495

theorem no_real_roots (c : ℝ) (h : c > 1) : ∀ x : ℝ, x^2 + 2*x + c ≠ 0 := by
  sorry

end no_real_roots_l514_51495


namespace exists_number_with_three_prime_factors_l514_51483

def M (n : ℕ) : Set ℕ := {m | n ≤ m ∧ m ≤ n + 9}

def has_at_least_three_prime_factors (k : ℕ) : Prop :=
  ∃ (p q r : ℕ), Nat.Prime p ∧ Nat.Prime q ∧ Nat.Prime r ∧ 
  p ≠ q ∧ p ≠ r ∧ q ≠ r ∧ k % p = 0 ∧ k % q = 0 ∧ k % r = 0

theorem exists_number_with_three_prime_factors (n : ℕ) (h : n ≥ 93) :
  ∃ k ∈ M n, has_at_least_three_prime_factors k := by
  sorry

end exists_number_with_three_prime_factors_l514_51483


namespace ellipse_triangle_perimeter_l514_51458

/-- Given an ellipse with equation x²/4 + y²/2 = 1, prove that the perimeter of the triangle
    formed by any point on the ellipse and its foci is 4 + 2√2. -/
theorem ellipse_triangle_perimeter (x y : ℝ) (P : ℝ × ℝ) (F₁ F₂ : ℝ × ℝ) :
  x^2 / 4 + y^2 / 2 = 1 →
  P = (x, y) →
  F₁.1^2 / 4 + F₁.2^2 / 2 = 1 →
  F₂.1^2 / 4 + F₂.2^2 / 2 = 1 →
  ∃ c : ℝ, c^2 = 2 ∧ 
    dist P F₁ + dist P F₂ + dist F₁ F₂ = 4 + 2 * Real.sqrt 2 :=
by sorry

end ellipse_triangle_perimeter_l514_51458


namespace quadratic_form_j_value_l514_51432

/-- Given a quadratic expression px^2 + qx + r that can be expressed as 5(x - 3)^2 + 15,
    prove that when 4px^2 + 4qx + 4r is expressed as m(x - j)^2 + k, j = 3. -/
theorem quadratic_form_j_value (p q r : ℝ) :
  (∃ m j k : ℝ, ∀ x : ℝ, 
    px^2 + q*x + r = 5*(x - 3)^2 + 15 ∧ 
    4*p*x^2 + 4*q*x + 4*r = m*(x - j)^2 + k) →
  (∃ m k : ℝ, ∀ x : ℝ, 4*p*x^2 + 4*q*x + 4*r = m*(x - 3)^2 + k) :=
by sorry

end quadratic_form_j_value_l514_51432


namespace inequality_equivalence_l514_51463

theorem inequality_equivalence (a : ℝ) : 
  (∀ x y : ℝ, x > 0 → y > 0 → Real.sqrt x + Real.sqrt y ≤ a * Real.sqrt (x + y)) ↔ 
  a ≥ Real.sqrt 2 := by
  sorry

end inequality_equivalence_l514_51463


namespace function_property_implies_f3_values_l514_51480

-- Define the function type
def FunctionType := ℝ → ℝ

-- Define the property that the function must satisfy
def SatisfiesProperty (f : FunctionType) : Prop :=
  ∀ x y : ℝ, f (x * f y - x) = x * y - f x

-- State the theorem
theorem function_property_implies_f3_values (f : FunctionType) 
  (h : SatisfiesProperty f) : 
  ∃ (a b : ℝ), (∀ z : ℝ, f 3 = z → (z = a ∨ z = b)) ∧ a + b = 0 := by
  sorry

end function_property_implies_f3_values_l514_51480


namespace cheryl_material_usage_l514_51493

theorem cheryl_material_usage
  (material1_bought : ℚ)
  (material2_bought : ℚ)
  (material3_bought : ℚ)
  (material1_left : ℚ)
  (material2_left : ℚ)
  (material3_left : ℚ)
  (h1 : material1_bought = 4/9)
  (h2 : material2_bought = 2/3)
  (h3 : material3_bought = 5/6)
  (h4 : material1_left = 8/18)
  (h5 : material2_left = 3/9)
  (h6 : material3_left = 2/12) :
  (material1_bought - material1_left) + (material2_bought - material2_left) + (material3_bought - material3_left) = 1 := by
  sorry

#check cheryl_material_usage

end cheryl_material_usage_l514_51493


namespace vector_equation_l514_51430

def a : ℝ × ℝ := (3, -2)
def b : ℝ × ℝ := (-2, 1)
def c : ℝ × ℝ := (-12, 7)

theorem vector_equation (m n : ℝ) (h : c = m • a + n • b) : m + n = 1 := by
  sorry

end vector_equation_l514_51430


namespace alex_marbles_l514_51409

theorem alex_marbles (lorin_black : ℕ) (jimmy_yellow : ℕ) 
  (h1 : lorin_black = 4)
  (h2 : jimmy_yellow = 22)
  (alex_black : ℕ) (alex_yellow : ℕ)
  (h3 : alex_black = 2 * lorin_black)
  (h4 : alex_yellow = jimmy_yellow / 2) :
  alex_black + alex_yellow = 19 := by
sorry

end alex_marbles_l514_51409


namespace mystery_number_addition_l514_51450

theorem mystery_number_addition (mystery_number certain_number : ℕ) : 
  mystery_number = 47 → 
  mystery_number + certain_number = 92 → 
  certain_number = 45 := by
sorry

end mystery_number_addition_l514_51450


namespace rational_fraction_representation_l514_51429

def is_rational (f : ℕ+ → ℚ) : Prop :=
  ∀ x : ℕ+, ∃ p q : ℤ, f x = p / q ∧ q ≠ 0

theorem rational_fraction_representation
  (a b : ℚ) (h : is_rational (λ x : ℕ+ => (a * x + b) / x)) :
  ∃ A B C : ℤ, ∀ x : ℕ+, (a * x + b) / x = (A * x + B) / (C * x) :=
sorry

end rational_fraction_representation_l514_51429


namespace school_committee_formation_l514_51447

theorem school_committee_formation (n_students : ℕ) (n_teachers : ℕ) (committee_size : ℕ) : 
  n_students = 11 → n_teachers = 3 → committee_size = 8 →
  (Nat.choose (n_students + n_teachers) committee_size) - (Nat.choose n_students committee_size) = 2838 :=
by sorry

end school_committee_formation_l514_51447


namespace max_value_x_plus_reciprocal_l514_51490

theorem max_value_x_plus_reciprocal (x : ℝ) (h : 9 = x^3 + 1/x^3) :
  ∃ (y : ℝ), y = x + 1/x ∧ y ≤ 3 ∧ y = 3 := by
  sorry

end max_value_x_plus_reciprocal_l514_51490


namespace sqrt_equation_solutions_l514_51472

theorem sqrt_equation_solutions (x : ℝ) :
  (Real.sqrt (5 * x - 4) + 12 / Real.sqrt (5 * x - 4) = 8) ↔ (x = 8 ∨ x = 8/5) :=
by sorry

end sqrt_equation_solutions_l514_51472


namespace average_score_two_classes_l514_51400

theorem average_score_two_classes (n1 n2 : ℕ) (s1 s2 : ℝ) :
  n1 > 0 → n2 > 0 →
  s1 = 80 → s2 = 70 →
  n1 = 20 → n2 = 30 →
  (n1 * s1 + n2 * s2) / (n1 + n2 : ℝ) = 74 := by
  sorry

end average_score_two_classes_l514_51400


namespace initial_points_count_l514_51421

theorem initial_points_count (k : ℕ) : 
  k > 0 → 4 * k - 3 = 101 → k = 26 := by
  sorry

end initial_points_count_l514_51421


namespace fence_painting_fraction_l514_51487

theorem fence_painting_fraction (total_time minutes : ℚ) (fraction : ℚ) :
  total_time = 60 →
  minutes = 15 →
  fraction = minutes / total_time →
  fraction = 1 / 4 := by
sorry

end fence_painting_fraction_l514_51487


namespace tangent_line_to_ellipse_l514_51404

/-- Represents an ellipse with semi-major axis √8 and semi-minor axis √2 -/
def Ellipse (x y : ℝ) : Prop := x^2 / 8 + y^2 / 2 = 1

/-- Represents a point on the ellipse -/
def PointOnEllipse : ℝ × ℝ := (2, 1)

/-- Represents the equation of a line -/
def Line (x y : ℝ) : Prop := x / 4 + y / 2 = 1

theorem tangent_line_to_ellipse :
  Line PointOnEllipse.1 PointOnEllipse.2 ∧
  Ellipse PointOnEllipse.1 PointOnEllipse.2 ∧
  ∀ (x y : ℝ), Ellipse x y → Line x y ∨ (x = PointOnEllipse.1 ∧ y = PointOnEllipse.2) :=
by sorry

end tangent_line_to_ellipse_l514_51404


namespace paint_cost_per_liter_l514_51479

/-- Calculates the cost of paint per liter given the costs of materials and profit --/
theorem paint_cost_per_liter 
  (brush_cost : ℚ) 
  (canvas_cost_multiplier : ℚ) 
  (min_paint_liters : ℚ) 
  (selling_price : ℚ) 
  (profit : ℚ) 
  (h1 : brush_cost = 20)
  (h2 : canvas_cost_multiplier = 3)
  (h3 : min_paint_liters = 5)
  (h4 : selling_price = 200)
  (h5 : profit = 80) :
  (selling_price - profit - (brush_cost + canvas_cost_multiplier * brush_cost)) / min_paint_liters = 8 := by
  sorry

end paint_cost_per_liter_l514_51479


namespace student_sister_weight_ratio_l514_51405

theorem student_sister_weight_ratio : 
  ∀ (student_weight sister_weight : ℝ),
    student_weight = 90 →
    student_weight + sister_weight = 132 →
    (student_weight - 6) / sister_weight = 2 :=
by
  sorry

end student_sister_weight_ratio_l514_51405


namespace diophantine_equation_solution_l514_51467

theorem diophantine_equation_solution : 
  ∀ a b : ℤ, a > 0 ∧ b > 0 → (1 : ℚ) / a - (1 : ℚ) / b = (1 : ℚ) / 37 → (a = 38 ∧ b = 1332) :=
by
  sorry

end diophantine_equation_solution_l514_51467


namespace sunzi_carriage_problem_l514_51426

/-- 
Given a number of carriages and people satisfying the conditions from 
"The Mathematical Classic of Sunzi", prove that the number of carriages 
satisfies the equation 3(x-2) = 2x + 9.
-/
theorem sunzi_carriage_problem (x : ℕ) (people : ℕ) :
  (3 * (x - 2) = people) →  -- Three people per carriage, two empty
  (2 * x + 9 = people) →    -- Two people per carriage, nine walking
  3 * (x - 2) = 2 * x + 9 := by
sorry

end sunzi_carriage_problem_l514_51426


namespace root_sum_and_square_l514_51478

theorem root_sum_and_square (α β : ℝ) : 
  (α^2 - α - 2006 = 0) → 
  (β^2 - β - 2006 = 0) → 
  (α + β = 1) →
  α + β^2 = 2007 := by
sorry

end root_sum_and_square_l514_51478


namespace school_expansion_theorem_l514_51448

/-- Calculates the total number of students after adding classes to a school -/
def total_students_after_adding_classes 
  (initial_classes : ℕ) 
  (students_per_class : ℕ) 
  (added_classes : ℕ) : ℕ :=
  (initial_classes + added_classes) * students_per_class

/-- Theorem: A school with 15 initial classes of 20 students each, 
    after adding 5 more classes, will have 400 students in total -/
theorem school_expansion_theorem : 
  total_students_after_adding_classes 15 20 5 = 400 := by
  sorry

end school_expansion_theorem_l514_51448


namespace game_d_higher_prob_l514_51459

def coin_prob_tails : ℚ := 3/4
def coin_prob_heads : ℚ := 1/4

def game_c_win_prob : ℚ := 2 * (coin_prob_heads * coin_prob_tails)

def game_d_win_prob : ℚ := 
  3 * (coin_prob_tails^2 * coin_prob_heads) + coin_prob_tails^3

theorem game_d_higher_prob : 
  game_d_win_prob - game_c_win_prob = 15/32 :=
sorry

end game_d_higher_prob_l514_51459


namespace total_limes_and_plums_l514_51456

theorem total_limes_and_plums (L M P : ℕ) (hL : L = 25) (hM : M = 32) (hP : P = 12) :
  L + M + P = 69 := by
  sorry

end total_limes_and_plums_l514_51456


namespace parabola_intersection_theorem_l514_51464

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- The parabola y^2 = 4x -/
def Parabola : Set Point :=
  {p : Point | p.y^2 = 4 * p.x}

/-- Distance squared between two points -/
def distanceSquared (p q : Point) : ℝ :=
  (p.x - q.x)^2 + (p.y - q.y)^2

/-- A line in parametric form -/
structure Line where
  a : ℝ  -- x-intercept
  α : ℝ  -- angle of inclination

/-- Get points where a line intersects the parabola -/
def lineParabolaIntersection (l : Line) : Set Point :=
  {p : Point | p ∈ Parabola ∧ ∃ t : ℝ, p.x = l.a + t * Real.cos l.α ∧ p.y = t * Real.sin l.α}

/-- The theorem to be proved -/
theorem parabola_intersection_theorem (a : ℝ) :
  (∀ l : Line, l.a = a →
    let M : Point := ⟨a, 0⟩
    let intersections := lineParabolaIntersection l
    ∃ k : ℝ, ∀ P Q : Point, P ∈ intersections → Q ∈ intersections → P ≠ Q →
      1 / distanceSquared P M + 1 / distanceSquared Q M = k) →
  a = 1 := by
  sorry

end parabola_intersection_theorem_l514_51464


namespace composite_polynomial_l514_51461

theorem composite_polynomial (n : ℕ) : 
  ∃ (a b : ℕ), a > 1 ∧ b > 1 ∧ n^3 + 9*n^2 + 27*n + 35 = a * b :=
by sorry

end composite_polynomial_l514_51461


namespace marks_initial_fries_l514_51476

/-- Given that Sally had 14 fries initially, Mark gave her one-third of his fries,
    and Sally ended up with 26 fries, prove that Mark initially had 36 fries. -/
theorem marks_initial_fries (sally_initial : ℕ) (sally_final : ℕ) (mark_fraction : ℚ) :
  sally_initial = 14 →
  sally_final = 26 →
  mark_fraction = 1 / 3 →
  ∃ (mark_initial : ℕ), 
    mark_initial = 36 ∧
    sally_final = sally_initial + mark_fraction * mark_initial :=
by sorry

end marks_initial_fries_l514_51476


namespace necessary_but_not_sufficient_l514_51417

theorem necessary_but_not_sufficient (x : ℝ) :
  ((x - 5) / (2 - x) > 0 → abs (x - 1) < 4) ∧
  (∃ y : ℝ, abs (y - 1) < 4 ∧ ¬((y - 5) / (2 - y) > 0)) :=
by sorry

end necessary_but_not_sufficient_l514_51417


namespace problem_solution_l514_51440

theorem problem_solution (x y : ℝ) (h1 : x + 2*y = 14) (h2 : y = 3) : 2*x + 3*y = 25 := by
  sorry

end problem_solution_l514_51440


namespace consecutive_color_draw_probability_l514_51449

def num_tan_chips : ℕ := 4
def num_pink_chips : ℕ := 4
def num_violet_chips : ℕ := 3
def total_chips : ℕ := num_tan_chips + num_pink_chips + num_violet_chips

theorem consecutive_color_draw_probability :
  (2 * (num_tan_chips.factorial * num_pink_chips.factorial * num_violet_chips.factorial)) / 
  total_chips.factorial = 1 / 5760 := by sorry

end consecutive_color_draw_probability_l514_51449


namespace pencil_length_l514_51436

theorem pencil_length (length1 length2 total_length : ℕ) : 
  length1 = length2 → 
  length1 + length2 = 24 → 
  length1 = 12 :=
by
  sorry

end pencil_length_l514_51436


namespace pyramid_theorem_l514_51414

structure Pyramid where
  S₁ : ℝ  -- Area of face ABD
  S₂ : ℝ  -- Area of face BCD
  S₃ : ℝ  -- Area of face CAD
  Q : ℝ   -- Area of face ABC
  α : ℝ   -- Dihedral angle at edge AB
  β : ℝ   -- Dihedral angle at edge BC
  γ : ℝ   -- Dihedral angle at edge AC
  h₁ : S₁ > 0
  h₂ : S₂ > 0
  h₃ : S₃ > 0
  h₄ : Q > 0
  h₅ : 0 < α ∧ α < π
  h₆ : 0 < β ∧ β < π
  h₇ : 0 < γ ∧ γ < π
  h₈ : Real.cos α = S₁ / Q
  h₉ : Real.cos β = S₂ / Q
  h₁₀ : Real.cos γ = S₃ / Q

theorem pyramid_theorem (p : Pyramid) : 
  p.S₁^2 + p.S₂^2 + p.S₃^2 = p.Q^2 ∧ 
  Real.cos (2 * p.α) + Real.cos (2 * p.β) + Real.cos (2 * p.γ) = 1 := by
  sorry

end pyramid_theorem_l514_51414


namespace valid_selections_eq_48_l514_51428

/-- The number of ways to select k items from n items -/
def arrangements (n k : ℕ) : ℕ := sorry

/-- The number of valid selections given the problem constraints -/
def valid_selections : ℕ :=
  arrangements 5 3 - arrangements 4 2

theorem valid_selections_eq_48 : valid_selections = 48 := by sorry

end valid_selections_eq_48_l514_51428


namespace max_tangent_segments_2017_l514_51415

/-- Given a number of circles, calculates the maximum number of tangent segments -/
def max_tangent_segments (n : ℕ) : ℕ := 3 * (n * (n - 1)) / 2

/-- Theorem: The maximum number of tangent segments for 2017 circles is 6,051,252 -/
theorem max_tangent_segments_2017 :
  max_tangent_segments 2017 = 6051252 := by
  sorry

#eval max_tangent_segments 2017

end max_tangent_segments_2017_l514_51415


namespace max_min_x_plus_y_l514_51410

theorem max_min_x_plus_y (x y : ℝ) (h : x^2 + y^2 - 4*x + 2*y + 2 = 0) :
  (∃ (a b : ℝ), (∀ (x' y' : ℝ), x'^2 + y'^2 - 4*x' + 2*y' + 2 = 0 → x' + y' ≤ a ∧ b ≤ x' + y') ∧
  a = 1 + Real.sqrt 6 ∧ b = 1 - Real.sqrt 6) :=
by sorry

end max_min_x_plus_y_l514_51410


namespace paint_cost_rectangular_floor_l514_51406

/-- The cost to paint a rectangular floor given its length and the ratio of length to breadth -/
theorem paint_cost_rectangular_floor 
  (length : ℝ) 
  (length_to_breadth_ratio : ℝ) 
  (paint_rate : ℝ) 
  (h1 : length = 15.491933384829668)
  (h2 : length_to_breadth_ratio = 3)
  (h3 : paint_rate = 3) : 
  ⌊length * (length / length_to_breadth_ratio) * paint_rate⌋ = 240 := by
sorry

end paint_cost_rectangular_floor_l514_51406


namespace root_product_theorem_l514_51481

theorem root_product_theorem (x₁ x₂ x₃ : ℝ) : 
  (Real.sqrt 2025 * x₁^3 - 4050 * x₁^2 + 4 = 0) →
  (Real.sqrt 2025 * x₂^3 - 4050 * x₂^2 + 4 = 0) →
  (Real.sqrt 2025 * x₃^3 - 4050 * x₃^2 + 4 = 0) →
  x₁ < x₂ → x₂ < x₃ →
  x₂ * (x₁ + x₃) = 90 := by
sorry

end root_product_theorem_l514_51481


namespace boys_cannot_score_double_l514_51418

/-- Represents a player in the chess tournament -/
inductive Player
| Boy
| Girl

/-- Represents the outcome of a chess game -/
inductive GameResult
| Win
| Draw
| Loss

/-- The number of players in the tournament -/
def numPlayers : Nat := 6

/-- The number of boys in the tournament -/
def numBoys : Nat := 2

/-- The number of girls in the tournament -/
def numGirls : Nat := 4

/-- The number of games each player plays -/
def gamesPerPlayer : Nat := numPlayers - 1

/-- The total number of games in the tournament -/
def totalGames : Nat := (numPlayers * gamesPerPlayer) / 2

/-- The points awarded for each game result -/
def pointsForResult (result : GameResult) : Rat :=
  match result with
  | GameResult.Win => 1
  | GameResult.Draw => 1/2
  | GameResult.Loss => 0

/-- A function representing the total score of a group of players -/
def groupScore (players : List Player) (results : List (Player × Player × GameResult)) : Rat :=
  sorry

/-- The main theorem stating that boys cannot score twice as many points as girls -/
theorem boys_cannot_score_double :
  ¬∃ (results : List (Player × Player × GameResult)),
    (results.length = totalGames) ∧
    (groupScore [Player.Boy, Player.Boy] results = 2 * groupScore [Player.Girl, Player.Girl, Player.Girl, Player.Girl] results) :=
  sorry

end boys_cannot_score_double_l514_51418
