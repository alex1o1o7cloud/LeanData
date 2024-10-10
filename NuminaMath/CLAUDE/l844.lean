import Mathlib

namespace b_value_l844_84447

def p (x : ℝ) : ℝ := 3 * x - 8

def q (x b : ℝ) : ℝ := 4 * x - b

theorem b_value (b : ℝ) : p (q 3 b) = 10 → b = 6 := by
  sorry

end b_value_l844_84447


namespace gcd_654321_543210_l844_84419

theorem gcd_654321_543210 : Nat.gcd 654321 543210 = 3 := by
  sorry

end gcd_654321_543210_l844_84419


namespace expression_equals_two_l844_84498

theorem expression_equals_two :
  (-1)^2023 - 2 * Real.sin (π / 3) + |(-Real.sqrt 3)| + (1/3)⁻¹ = 2 := by
  sorry

end expression_equals_two_l844_84498


namespace apple_pie_cost_per_serving_l844_84482

/-- Calculates the cost per serving of an apple pie --/
theorem apple_pie_cost_per_serving 
  (num_servings : ℕ)
  (apple_pounds : ℝ)
  (apple_cost_per_pound : ℝ)
  (crust_cost : ℝ)
  (lemon_cost : ℝ)
  (butter_cost : ℝ)
  (h1 : num_servings = 8)
  (h2 : apple_pounds = 2)
  (h3 : apple_cost_per_pound = 2)
  (h4 : crust_cost = 2)
  (h5 : lemon_cost = 0.5)
  (h6 : butter_cost = 1.5) :
  (apple_pounds * apple_cost_per_pound + crust_cost + lemon_cost + butter_cost) / num_servings = 1 :=
by sorry

end apple_pie_cost_per_serving_l844_84482


namespace min_value_on_circle_l844_84477

theorem min_value_on_circle (x y : ℝ) : 
  (x + 5)^2 + (y - 12)^2 = 14^2 → 
  ∃ (min : ℝ), (∀ (a b : ℝ), (a + 5)^2 + (b - 12)^2 = 14^2 → x^2 + y^2 ≤ a^2 + b^2) ∧ min = 1 :=
by sorry

end min_value_on_circle_l844_84477


namespace charles_chocolate_milk_l844_84444

/-- The amount of chocolate milk Charles can drink given his supplies -/
def chocolate_milk_total (milk_per_glass : ℚ) (syrup_per_glass : ℚ) 
  (total_milk : ℚ) (total_syrup : ℚ) : ℚ :=
  let glasses_from_milk := total_milk / milk_per_glass
  let glasses_from_syrup := total_syrup / syrup_per_glass
  let glasses := min glasses_from_milk glasses_from_syrup
  glasses * (milk_per_glass + syrup_per_glass)

/-- Theorem stating that Charles will drink 160 ounces of chocolate milk -/
theorem charles_chocolate_milk :
  chocolate_milk_total (6.5) (1.5) (130) (60) = 160 := by
  sorry

end charles_chocolate_milk_l844_84444


namespace square_area_from_perimeter_l844_84424

/-- The area of a square with perimeter 48 cm is 144 cm² -/
theorem square_area_from_perimeter (perimeter : ℝ) (area : ℝ) : 
  perimeter = 48 → area = (perimeter / 4) ^ 2 → area = 144 := by sorry

end square_area_from_perimeter_l844_84424


namespace complex_number_problem_l844_84451

/-- Given a complex number z satisfying z = i(2-z), prove that z = 1 + i and |z-(2-i)| = √5 -/
theorem complex_number_problem (z : ℂ) (h : z = Complex.I * (2 - z)) : 
  z = 1 + Complex.I ∧ Complex.abs (z - (2 - Complex.I)) = Real.sqrt 5 := by
  sorry

end complex_number_problem_l844_84451


namespace fraction_meaningful_l844_84442

theorem fraction_meaningful (x : ℝ) : 
  (∃ y : ℝ, y = 1 / (x - 2)) ↔ x ≠ 2 := by sorry

end fraction_meaningful_l844_84442


namespace computer_factory_month_days_l844_84411

/-- Proves the number of days in a month given computer production rates --/
theorem computer_factory_month_days
  (monthly_production : ℕ)
  (half_hour_production : ℚ)
  (h1 : monthly_production = 3024)
  (h2 : half_hour_production = 225 / 100) :
  (monthly_production : ℚ) / ((half_hour_production * 2 * 24) : ℚ) = 28 := by
  sorry

end computer_factory_month_days_l844_84411


namespace rectangle_area_solution_l844_84431

/-- A rectangle with dimensions (3x - 4) and (4x + 6) has area 12x^2 + 2x - 24 -/
def rectangle_area (x : ℝ) : ℝ := (3*x - 4) * (4*x + 6)

/-- The solution set for x -/
def solution_set : Set ℝ := {x | x > 4/3}

theorem rectangle_area_solution :
  ∀ x : ℝ, x ∈ solution_set ↔ 
    (rectangle_area x = 12*x^2 + 2*x - 24 ∧ 
     3*x - 4 > 0 ∧ 
     4*x + 6 > 0) :=
by sorry

end rectangle_area_solution_l844_84431


namespace polygon_sides_l844_84479

/-- A polygon with side length 4 and perimeter 24 has 6 sides -/
theorem polygon_sides (side_length : ℝ) (perimeter : ℝ) (num_sides : ℕ) : 
  side_length = 4 → perimeter = 24 → num_sides * side_length = perimeter → num_sides = 6 := by
  sorry

end polygon_sides_l844_84479


namespace initial_acorns_l844_84499

def acorns_given_away : ℕ := 7
def acorns_left : ℕ := 9

theorem initial_acorns : 
  acorns_given_away + acorns_left = 16 := by
  sorry

end initial_acorns_l844_84499


namespace base_eight_sum_l844_84425

theorem base_eight_sum (A B C : ℕ) : 
  A ≠ 0 → B ≠ 0 → C ≠ 0 →
  A ≠ B → B ≠ C → A ≠ C →
  A < 8 → B < 8 → C < 8 →
  (8^2 * A + 8 * B + C) + (8^2 * B + 8 * C + A) + (8^2 * C + 8 * A + B) = 8^3 * A + 8^2 * A + 8 * A →
  B + C = 7 := by
sorry

end base_eight_sum_l844_84425


namespace chosen_number_proof_l844_84422

theorem chosen_number_proof (x : ℝ) : (x / 12) - 240 = 8 ↔ x = 2976 := by
  sorry

end chosen_number_proof_l844_84422


namespace circular_arrangements_count_l844_84469

/-- The number of ways to arrange n people in a circle with r people between A and B -/
def circularArrangements (n : ℕ) (r : ℕ) : ℕ :=
  2 * Nat.factorial (n - 2)

/-- Theorem: The number of circular arrangements with r people between A and B -/
theorem circular_arrangements_count (n : ℕ) (r : ℕ) 
  (h₁ : n ≥ 3) 
  (h₂ : r < n / 2 - 1) : 
  circularArrangements n r = 2 * Nat.factorial (n - 2) := by
  sorry

end circular_arrangements_count_l844_84469


namespace spinsters_and_cats_l844_84472

theorem spinsters_and_cats (spinsters : ℕ) (cats : ℕ) : 
  spinsters = 12 →
  (spinsters : ℚ) / cats = 2 / 9 →
  cats > spinsters →
  cats - spinsters = 42 := by
sorry

end spinsters_and_cats_l844_84472


namespace trees_needed_for_road_l844_84464

/-- The number of trees needed to plant on one side of a road -/
def num_trees (road_length : ℕ) (interval : ℕ) : ℕ :=
  road_length / interval + 1

/-- Theorem: The number of trees needed for a 1500m road with 25m intervals is 61 -/
theorem trees_needed_for_road : num_trees 1500 25 = 61 := by
  sorry

end trees_needed_for_road_l844_84464


namespace light_path_length_in_cube_l844_84484

/-- Represents a point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Represents a cube -/
structure Cube where
  sideLength : ℝ

/-- Represents a light beam path in the cube -/
structure LightPath where
  start : Point3D
  reflection : Point3D
  cubeSideLength : ℝ

/-- Calculates the length of the light path -/
def lightPathLength (path : LightPath) : ℝ :=
  sorry

theorem light_path_length_in_cube (c : Cube) (path : LightPath) :
  c.sideLength = 10 ∧
  path.start = Point3D.mk 0 0 0 ∧
  path.reflection = Point3D.mk 10 3 4 ∧
  path.cubeSideLength = c.sideLength →
  lightPathLength path = 50 * Real.sqrt 5 :=
sorry

end light_path_length_in_cube_l844_84484


namespace chess_proficiency_multiple_chess_proficiency_multiple_proof_l844_84437

theorem chess_proficiency_multiple : ℕ → Prop :=
  fun x =>
    let time_learn_rules : ℕ := 2
    let time_get_proficient : ℕ := time_learn_rules * x
    let time_become_master : ℕ := 100 * (time_learn_rules + time_get_proficient)
    let total_time : ℕ := 10100
    total_time = time_learn_rules + time_get_proficient + time_become_master →
    x = 49

theorem chess_proficiency_multiple_proof : chess_proficiency_multiple 49 := by
  sorry

end chess_proficiency_multiple_chess_proficiency_multiple_proof_l844_84437


namespace fraction_equality_l844_84405

theorem fraction_equality (a b c d e f : ℚ) 
  (h1 : a / b = 1 / 3) 
  (h2 : c / d = 1 / 3) 
  (h3 : e / f = 1 / 3) : 
  (3 * a - 2 * c + e) / (3 * b - 2 * d + f) = 1 / 3 := by
  sorry

end fraction_equality_l844_84405


namespace negation_of_forall_not_equal_l844_84450

theorem negation_of_forall_not_equal (x : ℝ) :
  (¬ ∀ x > 0, Real.log x ≠ x - 1) ↔ (∃ x > 0, Real.log x = x - 1) := by
  sorry

end negation_of_forall_not_equal_l844_84450


namespace consecutive_composites_exist_l844_84474

-- Define a function to check if a number is composite
def isComposite (n : ℕ) : Prop := ∃ a b, 1 < a ∧ 1 < b ∧ n = a * b

-- Define a function to check if a sequence of numbers is all composite
def allComposite (start : ℕ) (length : ℕ) : Prop :=
  ∀ i, i ∈ Finset.range length → isComposite (start + i)

-- Theorem statement
theorem consecutive_composites_exist :
  (∃ start : ℕ, start ≤ 500 - 9 + 1 ∧ allComposite start 9) ∧
  (∃ start : ℕ, start ≤ 500 - 11 + 1 ∧ allComposite start 11) := by
  sorry

end consecutive_composites_exist_l844_84474


namespace science_olympiad_participation_l844_84407

theorem science_olympiad_participation 
  (j s : ℕ) -- j: number of juniors, s: number of seniors
  (h1 : (3 : ℚ) / 7 * j = (2 : ℚ) / 7 * s) -- equal number of participants
  : s = 3 * j := by
sorry

end science_olympiad_participation_l844_84407


namespace count_special_numbers_is_792_l844_84448

/-- A function that counts the number of 5-digit numbers beginning with 2 
    and having exactly three identical digits -/
def count_special_numbers : ℕ :=
  let count_with_three_twos := 6 * 9 * 8
  let count_with_three_non_twos := 5 * 8 * 9
  count_with_three_twos + count_with_three_non_twos

/-- Theorem stating that the count of special numbers is 792 -/
theorem count_special_numbers_is_792 : count_special_numbers = 792 := by
  sorry

#eval count_special_numbers

end count_special_numbers_is_792_l844_84448


namespace vertical_shift_proof_l844_84408

/-- Represents a line in slope-intercept form -/
structure Line where
  slope : ℚ
  intercept : ℚ

/-- Shifts a line vertically by a given amount -/
def vertical_shift (l : Line) (shift : ℚ) : Line :=
  { slope := l.slope, intercept := l.intercept + shift }

theorem vertical_shift_proof (x : ℚ) :
  let l1 : Line := { slope := -3/4, intercept := 0 }
  let l2 : Line := { slope := -3/4, intercept := -4 }
  vertical_shift l1 (-4) = l2 := by
  sorry

end vertical_shift_proof_l844_84408


namespace apple_pie_servings_l844_84458

theorem apple_pie_servings 
  (guests : ℕ) 
  (apples_per_guest : ℝ) 
  (num_pies : ℕ) 
  (apples_per_serving : ℝ) 
  (h1 : guests = 12) 
  (h2 : apples_per_guest = 3) 
  (h3 : num_pies = 3) 
  (h4 : apples_per_serving = 1.5) : 
  (guests * apples_per_guest) / (num_pies * apples_per_serving) = 8 := by
  sorry

end apple_pie_servings_l844_84458


namespace overlap_area_is_two_l844_84449

-- Define the 3x3 grid
def Grid := Fin 3 × Fin 3

-- Define the two quadrilaterals
def quad1 : List Grid := [(0,1), (1,2), (2,1), (1,0)]
def quad2 : List Grid := [(0,0), (2,2), (2,0), (0,2)]

-- Define the function to calculate the area of overlap
def overlapArea (q1 q2 : List Grid) : ℝ :=
  sorry  -- The actual calculation would go here

-- State the theorem
theorem overlap_area_is_two :
  overlapArea quad1 quad2 = 2 :=
sorry

end overlap_area_is_two_l844_84449


namespace lemon_juice_per_lemon_l844_84429

/-- The amount of lemon juice needed for one dozen cupcakes, in tablespoons -/
def juice_per_dozen : ℚ := 12

/-- The number of dozens of cupcakes to be made -/
def dozens_to_make : ℚ := 3

/-- The number of lemons needed for the total amount of cupcakes -/
def lemons_needed : ℚ := 9

/-- Proves that each lemon provides 4 tablespoons of juice -/
theorem lemon_juice_per_lemon : 
  (juice_per_dozen * dozens_to_make) / lemons_needed = 4 := by
  sorry

end lemon_juice_per_lemon_l844_84429


namespace hyperbola_focal_length_l844_84470

/-- Given a hyperbola C with equation x²/m - y² = 1 (m > 0) and asymptote √3x + my = 0,
    prove that the focal length of C is 4. -/
theorem hyperbola_focal_length (m : ℝ) (hm : m > 0) :
  let C := {(x, y) : ℝ × ℝ | x^2 / m - y^2 = 1}
  let asymptote := {(x, y) : ℝ × ℝ | Real.sqrt 3 * x + m * y = 0}
  ∃ (a b c : ℝ), a^2 = m ∧ b^2 = m ∧ c^2 = a^2 + b^2 ∧ 2 * c = 4 :=
by sorry

end hyperbola_focal_length_l844_84470


namespace perfect_square_condition_l844_84427

theorem perfect_square_condition (X M : ℕ) : 
  (1000 < X ∧ X < 8000) → 
  (M > 1) → 
  (X = M * M^2) → 
  (∃ k : ℕ, X = k^2) → 
  M = 16 := by
sorry

end perfect_square_condition_l844_84427


namespace sorting_inequality_l844_84488

theorem sorting_inequality (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  2 * (a^3 + b^3 + c^3) ≥ a^2 * (b + c) + b^2 * (a + c) + c^2 * (a + b) := by
  sorry

end sorting_inequality_l844_84488


namespace number_of_cans_l844_84412

/-- Proves the number of cans given space requirements before and after compaction --/
theorem number_of_cans 
  (space_before : ℝ) 
  (compaction_ratio : ℝ) 
  (total_space_after : ℝ) 
  (h1 : space_before = 30) 
  (h2 : compaction_ratio = 0.2) 
  (h3 : total_space_after = 360) : 
  ℕ :=
by
  sorry

#check number_of_cans

end number_of_cans_l844_84412


namespace order_of_expressions_l844_84445

theorem order_of_expressions : 
  let a : ℝ := (0.2 : ℝ)^2
  let b : ℝ := 2^(0.3 : ℝ)
  let c : ℝ := Real.log 2 / Real.log 0.2
  b > a ∧ a > c := by sorry

end order_of_expressions_l844_84445


namespace function_properties_l844_84480

/-- The function f(x) = ax² + bx + 1 -/
def f (a b x : ℝ) : ℝ := a * x^2 + b * x + 1

/-- The function g(x) = f(x) - kx -/
def g (a b k x : ℝ) : ℝ := f a b x - k * x

theorem function_properties (a b k : ℝ) :
  (∀ x, f a b x ≥ 0) ∧  -- Range of f(x) is [0, +∞)
  (f a b (-1) = 0) ∧    -- f(x) has a zero point at x = -1
  (∀ x ∈ Set.Icc (-2) 2, Monotone (g a b k)) -- g(x) is monotonic on [-2, 2]
  →
  (f a b = fun x ↦ x^2 + 2*x + 1) ∧  -- f(x) = x² + 2x + 1
  (k ≥ 6 ∨ k ≤ -2)                   -- Range of k
  := by sorry

end function_properties_l844_84480


namespace point_translation_l844_84417

/-- Given a point B with coordinates (5, -1) that is translated upwards by 2 units
    to obtain point A with coordinates (a+b, a-b), prove that a = 3 and b = 2. -/
theorem point_translation (a b : ℝ) : 
  (5 : ℝ) = a + b ∧ (1 : ℝ) = a - b → a = 3 ∧ b = 2 := by
  sorry

end point_translation_l844_84417


namespace min_value_expression_l844_84414

theorem min_value_expression (x y : ℝ) : (x*y - 2)^2 + (x^2 + y^2)^2 ≥ 4 := by
  sorry

end min_value_expression_l844_84414


namespace hearty_red_packages_l844_84415

/-- The number of beads in each package -/
def beads_per_package : ℕ := 40

/-- The number of packages of blue beads Hearty bought -/
def blue_packages : ℕ := 3

/-- The total number of beads Hearty has -/
def total_beads : ℕ := 320

/-- The number of packages of red beads Hearty bought -/
def red_packages : ℕ := (total_beads - blue_packages * beads_per_package) / beads_per_package

theorem hearty_red_packages : red_packages = 5 := by
  sorry

end hearty_red_packages_l844_84415


namespace alice_has_ball_after_two_turns_l844_84416

/-- Probability of Alice tossing the ball to Bob -/
def alice_toss_prob : ℚ := 1 / 3

/-- Probability of Alice keeping the ball -/
def alice_keep_prob : ℚ := 2 / 3

/-- Probability of Bob tossing the ball to Alice -/
def bob_toss_prob : ℚ := 1 / 4

/-- Probability of Bob keeping the ball -/
def bob_keep_prob : ℚ := 3 / 4

/-- Alice starts with the ball -/
def alice_starts : Prop := True

theorem alice_has_ball_after_two_turns :
  alice_starts →
  (alice_toss_prob * bob_toss_prob + alice_keep_prob * alice_keep_prob : ℚ) = 37 / 108 :=
by sorry

end alice_has_ball_after_two_turns_l844_84416


namespace viewers_of_program_A_l844_84453

theorem viewers_of_program_A (total_viewers : ℕ) (ratio_both ratio_A ratio_B : ℕ) : 
  total_viewers = 560 →
  ratio_both = 1 →
  ratio_A = 2 →
  ratio_B = 3 →
  (ratio_both + ratio_A + ratio_B) * (ratio_both + ratio_A) * total_viewers / ((ratio_both + ratio_A + ratio_B) * (ratio_both + ratio_A + ratio_B)) = 280 :=
by sorry

end viewers_of_program_A_l844_84453


namespace lending_rate_calculation_l844_84468

def borrowed_amount : ℝ := 7000
def borrowed_time : ℝ := 2
def borrowed_rate : ℝ := 4
def gain_per_year : ℝ := 140

theorem lending_rate_calculation :
  let borrowed_interest := borrowed_amount * borrowed_rate * borrowed_time / 100
  let total_gain := gain_per_year * borrowed_time
  let total_interest_earned := borrowed_interest + total_gain
  let lending_rate := (total_interest_earned * 100) / (borrowed_amount * borrowed_time)
  lending_rate = 6 := by sorry

end lending_rate_calculation_l844_84468


namespace arithmetic_sequence_tan_a7_l844_84455

theorem arithmetic_sequence_tan_a7 (a : ℕ → ℝ) :
  (∀ n : ℕ, a (n + 1) - a n = a (n + 2) - a (n + 1)) →  -- arithmetic sequence condition
  a 5 + a 9 = 8 * Real.pi / 3 →                         -- given condition
  Real.tan (a 7) = Real.sqrt 3 :=                       -- conclusion to prove
by sorry

end arithmetic_sequence_tan_a7_l844_84455


namespace remainder_theorem_l844_84462

theorem remainder_theorem : (104 * 106 - 8) % 8 = 0 := by
  sorry

end remainder_theorem_l844_84462


namespace james_delivery_capacity_l844_84487

/-- Given that James takes 20 trips a day and delivers 1000 bags in 5 days,
    prove that he can carry 10 bags on each trip. -/
theorem james_delivery_capacity
  (trips_per_day : ℕ)
  (total_bags : ℕ)
  (total_days : ℕ)
  (h1 : trips_per_day = 20)
  (h2 : total_bags = 1000)
  (h3 : total_days = 5) :
  total_bags / (trips_per_day * total_days) = 10 := by
  sorry

end james_delivery_capacity_l844_84487


namespace exam_score_problem_l844_84483

theorem exam_score_problem (total_questions : ℕ) (correct_score wrong_score total_score : ℤ) 
  (h1 : total_questions = 60)
  (h2 : correct_score = 4)
  (h3 : wrong_score = -1)
  (h4 : total_score = 140) :
  ∃ (correct_answers : ℕ),
    correct_answers ≤ total_questions ∧
    correct_score * correct_answers + wrong_score * (total_questions - correct_answers) = total_score ∧
    correct_answers = 40 := by
  sorry

end exam_score_problem_l844_84483


namespace white_mailbox_houses_l844_84443

theorem white_mailbox_houses (total_mail : ℕ) (total_houses : ℕ) (red_houses : ℕ) (mail_per_house : ℕ)
  (h1 : total_mail = 48)
  (h2 : total_houses = 8)
  (h3 : red_houses = 3)
  (h4 : mail_per_house = 6) :
  total_houses - red_houses = 5 := by
sorry

end white_mailbox_houses_l844_84443


namespace terms_before_five_l844_84404

/-- Given an arithmetic sequence starting with 75 and having a common difference of -5,
    this theorem proves that the number of terms that appear before 5 is 14. -/
theorem terms_before_five (a : ℕ → ℤ) :
  a 0 = 75 ∧ 
  (∀ n : ℕ, a (n + 1) - a n = -5) →
  (∃ k : ℕ, a k = 5 ∧ k = 15) ∧ 
  (∀ m : ℕ, m < 15 → a m > 5) :=
by sorry

end terms_before_five_l844_84404


namespace divisibility_problem_l844_84485

theorem divisibility_problem (a b c : ℤ) (h : 18 ∣ (a^3 + b^3 + c^3)) : 6 ∣ (a * b * c) := by
  sorry

end divisibility_problem_l844_84485


namespace pictures_per_album_l844_84491

theorem pictures_per_album 
  (phone_pics : ℕ) 
  (camera_pics : ℕ) 
  (num_albums : ℕ) 
  (h1 : phone_pics = 23) 
  (h2 : camera_pics = 7) 
  (h3 : num_albums = 5) 
  (h4 : num_albums > 0) :
  (phone_pics + camera_pics) / num_albums = 6 := by
  sorry

end pictures_per_album_l844_84491


namespace solution_difference_l844_84476

theorem solution_difference (r s : ℝ) : 
  r ≠ s →
  (r - 5) * (r + 5) = 25 * r - 125 →
  (s - 5) * (s + 5) = 25 * s - 125 →
  r > s →
  r - s = 15 := by sorry

end solution_difference_l844_84476


namespace square_formation_theorem_l844_84481

def sum_of_naturals (n : ℕ) : ℕ := n * (n + 1) / 2

def can_form_square (n : ℕ) : Bool :=
  sum_of_naturals n % 4 = 0

def min_breaks_to_square (n : ℕ) : ℕ :=
  if can_form_square n then 0
  else
    let total := sum_of_naturals n
    let target := (total + 3) / 4 * 4
    (target - total + 1) / 2

theorem square_formation_theorem :
  (min_breaks_to_square 12 = 2) ∧ (can_form_square 15 = true) := by
  sorry

end square_formation_theorem_l844_84481


namespace cookies_per_child_l844_84473

theorem cookies_per_child (total_cookies : ℕ) (num_adults num_children : ℕ) (adult_fraction : ℚ) : 
  total_cookies = 240 →
  num_adults = 4 →
  num_children = 6 →
  adult_fraction = 1/4 →
  (total_cookies - (adult_fraction * total_cookies).num) / num_children = 30 := by
  sorry

end cookies_per_child_l844_84473


namespace novel_pages_calculation_l844_84471

/-- Calculates the total number of pages in a novel based on a specific reading pattern -/
theorem novel_pages_calculation : 
  let first_four_days := 4
  let next_two_days := 2
  let last_day := 1
  let pages_per_day_first_four := 42
  let pages_per_day_next_two := 50
  let pages_last_day := 30
  
  (first_four_days * pages_per_day_first_four) + 
  (next_two_days * pages_per_day_next_two) + 
  pages_last_day = 298 := by
  sorry

end novel_pages_calculation_l844_84471


namespace negation_of_exp_inequality_l844_84495

theorem negation_of_exp_inequality :
  (¬ ∀ x : ℝ, Real.exp x > x) ↔ (∃ x₀ : ℝ, Real.exp x₀ ≤ x₀) := by
  sorry

end negation_of_exp_inequality_l844_84495


namespace valid_parameterizations_l844_84457

def is_valid_parameterization (p₀ : ℝ × ℝ) (d : ℝ × ℝ) : Prop :=
  ∀ t : ℝ, let (x, y) := p₀ + t • d
           y = x - 1

theorem valid_parameterizations :
  (is_valid_parameterization (1, 0) (1, 1)) ∧
  (is_valid_parameterization (0, -1) (-1, -1)) ∧
  (is_valid_parameterization (2, 1) (0.5, 0.5)) :=
sorry

end valid_parameterizations_l844_84457


namespace F15_triangles_l844_84452

/-- The number of triangles in figure n of the sequence -/
def T (n : ℕ) : ℕ :=
  if n = 0 then 0
  else if n = 1 then 1
  else T (n - 1) + 3 * n + 3

/-- The sequence of figures satisfies the given construction rules -/
axiom construction_rule (n : ℕ) : n ≥ 2 → T n = T (n - 1) + 3 * n + 3

/-- F₂ has 7 triangles -/
axiom F2_triangles : T 2 = 7

/-- The number of triangles in F₁₅ is 400 -/
theorem F15_triangles : T 15 = 400 := by sorry

end F15_triangles_l844_84452


namespace mary_max_earnings_l844_84460

/-- Mary's maximum weekly earnings at the restaurant --/
theorem mary_max_earnings (max_hours : ℕ) (regular_hours : ℕ) (regular_rate : ℚ) 
  (overtime_rate_increase : ℚ) (bonus_hours : ℕ) (bonus_amount : ℚ) :
  max_hours = 80 →
  regular_hours = 20 →
  regular_rate = 8 →
  overtime_rate_increase = 1/4 →
  bonus_hours = 5 →
  bonus_amount = 20 →
  let overtime_hours := max_hours - regular_hours
  let overtime_rate := regular_rate * (1 + overtime_rate_increase)
  let regular_earnings := regular_hours * regular_rate
  let overtime_earnings := overtime_hours * overtime_rate
  let bonus_count := overtime_hours / bonus_hours
  let total_bonus := bonus_count * bonus_amount
  regular_earnings + overtime_earnings + total_bonus = 1000 := by
sorry

end mary_max_earnings_l844_84460


namespace equation_solution_l844_84438

theorem equation_solution :
  let S : Set ℂ := {x | (x - 1)^4 + (x - 1) = 0}
  S = {1, 0, Complex.mk 1 (Real.sqrt 3 / 2), Complex.mk 1 (-Real.sqrt 3 / 2)} := by
  sorry

end equation_solution_l844_84438


namespace circle_condition_l844_84440

-- Define the equation of the curve
def curve_equation (x y m : ℝ) : Prop :=
  x^2 + y^2 + 4*m*x - 2*y + 5*m = 0

-- Define the condition for m
def m_condition (m : ℝ) : Prop :=
  m < 1/4 ∨ m > 1

-- Theorem statement
theorem circle_condition (m : ℝ) :
  (∃ h k r, ∀ x y, curve_equation x y m ↔ (x - h)^2 + (y - k)^2 = r^2) ↔ m_condition m :=
sorry

end circle_condition_l844_84440


namespace binomial_expansion_103_l844_84456

theorem binomial_expansion_103 : 102^3 + 3*(102^2) + 3*102 + 1 = 103^3 := by
  sorry

end binomial_expansion_103_l844_84456


namespace range_of_a_l844_84489

def p (x : ℝ) : Prop := |4 * x - 3| ≤ 1

def q (x a : ℝ) : Prop := x^2 - (2*a + 1)*x + a*(a + 1) ≤ 0

theorem range_of_a :
  (∃ x, ¬(p x) ∧ q x a) ∧
  (∀ x, ¬(q x a) → ¬(p x)) →
  ∃ S : Set ℝ, S = {a : ℝ | 0 ≤ a ∧ a ≤ 1/2} :=
by sorry

end range_of_a_l844_84489


namespace smallest_sum_of_digits_of_sum_l844_84463

def is_three_digit (n : ℕ) : Prop := n ≥ 100 ∧ n ≤ 999

def digits_consecutive (n : ℕ) : Prop :=
  ∃ d₁ d₂ d₃ : ℕ, n = d₁ * 100 + d₂ * 10 + d₃ ∧
  ((d₁ + 1 = d₂ ∧ d₂ + 1 = d₃) ∨
   (d₁ + 1 = d₃ ∧ d₃ + 1 = d₂) ∨
   (d₂ + 1 = d₁ ∧ d₁ + 1 = d₃) ∨
   (d₂ + 1 = d₃ ∧ d₃ + 1 = d₁) ∨
   (d₃ + 1 = d₁ ∧ d₁ + 1 = d₂) ∨
   (d₃ + 1 = d₂ ∧ d₂ + 1 = d₁))

def sum_of_digits (n : ℕ) : ℕ :=
  let digits := n.digits 10
  digits.sum

theorem smallest_sum_of_digits_of_sum :
  ∀ a b : ℕ, is_three_digit a → is_three_digit b →
  digits_consecutive a → digits_consecutive b →
  ∃ S : ℕ, S = a + b ∧ sum_of_digits S ≥ 21 :=
sorry

end smallest_sum_of_digits_of_sum_l844_84463


namespace tire_circumference_l844_84420

/-- Given a tire rotating at 400 revolutions per minute on a car traveling at 120 km/h,
    the circumference of the tire is 5 meters. -/
theorem tire_circumference (rpm : ℝ) (speed : ℝ) (circ : ℝ) : 
  rpm = 400 → speed = 120 → circ * rpm = speed * 1000 / 60 → circ = 5 := by
  sorry

#check tire_circumference

end tire_circumference_l844_84420


namespace percentage_of_female_cows_l844_84493

theorem percentage_of_female_cows (total_cows : ℕ) (pregnant_cows : ℕ) 
  (h1 : total_cows = 44)
  (h2 : pregnant_cows = 11)
  (h3 : pregnant_cows = (female_cows / 2 : ℚ)) :
  (female_cows : ℚ) / total_cows * 100 = 50 :=
by
  sorry

#check percentage_of_female_cows

end percentage_of_female_cows_l844_84493


namespace system_solution_l844_84432

theorem system_solution (x y : ℝ) (eq1 : x + 2*y = 1) (eq2 : 2*x + y = 2) : x + y = 1 := by
  sorry

end system_solution_l844_84432


namespace not_p_and_not_q_l844_84439

-- Define proposition p
def p : Prop := ∃ t : ℝ, t > 0 ∧ t^2 - 2*t + 2 = 0

-- Define proposition q
def q : Prop := ∀ x : ℝ, Real.exp x - x - 1 ≥ -1

-- Theorem statement
theorem not_p_and_not_q : (¬p) ∧ (¬q) := by
  sorry

end not_p_and_not_q_l844_84439


namespace chessboard_coloring_l844_84435

/-- A coloring of an n × n chessboard is valid if for every i ∈ {1,2,...,n}, 
    the 2n-1 cells on i-th row and i-th column have all different colors. -/
def ValidColoring (n : ℕ) (k : ℕ) : Prop :=
  ∃ (coloring : Fin n → Fin n → Fin k),
    ∀ i : Fin n, (∀ j j' : Fin n, j ≠ j' → coloring i j ≠ coloring i j') ∧
                 (∀ i' : Fin n, i ≠ i' → coloring i i' ≠ coloring i' i)

theorem chessboard_coloring :
  (¬ ValidColoring 2001 4001) ∧
  (∀ m : ℕ, ValidColoring (2^m - 1) (2^(m+1) - 1)) :=
sorry

end chessboard_coloring_l844_84435


namespace carols_rectangle_width_l844_84446

theorem carols_rectangle_width (carol_length jordan_length jordan_width : ℝ) 
  (h1 : carol_length = 15)
  (h2 : jordan_length = 6)
  (h3 : jordan_width = 50)
  (h4 : carol_length * carol_width = jordan_length * jordan_width) :
  carol_width = 20 := by
  sorry

end carols_rectangle_width_l844_84446


namespace image_of_four_six_l844_84490

def f (p : ℝ × ℝ) : ℝ × ℝ := (p.1 + p.2, p.1 - p.2)

theorem image_of_four_six : f (4, 6) = (10, -2) := by
  sorry

end image_of_four_six_l844_84490


namespace gcd_20586_58768_l844_84434

theorem gcd_20586_58768 : Nat.gcd 20586 58768 = 2 := by
  sorry

end gcd_20586_58768_l844_84434


namespace book_chapters_l844_84465

/-- A problem about determining the number of chapters in a book based on reading rate. -/
theorem book_chapters (chapters_read : ℕ) (hours_read : ℕ) (hours_remaining : ℕ) : 
  chapters_read = 2 →
  hours_read = 3 →
  hours_remaining = 9 →
  ∃ (total_chapters : ℕ), 
    total_chapters = chapters_read + (chapters_read * hours_remaining / hours_read) ∧
    total_chapters = 8 := by
  sorry


end book_chapters_l844_84465


namespace square_difference_l844_84492

theorem square_difference (x y : ℝ) (h1 : (x + y)^2 = 64) (h2 : x * y = 10) :
  (x - y)^2 = 24 := by sorry

end square_difference_l844_84492


namespace cereal_boxes_purchased_l844_84409

/-- Given the initial price, price reduction, and total payment for monster boxes of cereal,
    prove that the number of boxes purchased is 20. -/
theorem cereal_boxes_purchased
  (initial_price : ℕ)
  (price_reduction : ℕ)
  (total_payment : ℕ)
  (h1 : initial_price = 104)
  (h2 : price_reduction = 24)
  (h3 : total_payment = 1600) :
  total_payment / (initial_price - price_reduction) = 20 :=
by sorry

end cereal_boxes_purchased_l844_84409


namespace max_cars_with_ac_no_stripes_l844_84401

theorem max_cars_with_ac_no_stripes 
  (total_cars : ℕ) 
  (cars_without_ac : ℕ) 
  (cars_with_stripes : ℕ) 
  (h1 : total_cars = 100)
  (h2 : cars_without_ac = 49)
  (h3 : cars_with_stripes ≥ 51) :
  ∃ (max_cars : ℕ), 
    max_cars ≤ total_cars - cars_without_ac ∧
    max_cars ≤ total_cars - cars_with_stripes ∧
    ∀ (n : ℕ), n ≤ total_cars - cars_without_ac ∧ 
               n ≤ total_cars - cars_with_stripes → 
               n ≤ max_cars ∧
    max_cars = 49 := by
  sorry

end max_cars_with_ac_no_stripes_l844_84401


namespace vertex_on_x_axis_l844_84418

/-- The parabola equation -/
def parabola (x d : ℝ) : ℝ := x^2 - 6*x + d

/-- The x-coordinate of the vertex of the parabola -/
def vertex_x : ℝ := 3

/-- The y-coordinate of the vertex of the parabola -/
def vertex_y (d : ℝ) : ℝ := parabola vertex_x d

theorem vertex_on_x_axis (d : ℝ) : vertex_y d = 0 ↔ d = 9 := by sorry

end vertex_on_x_axis_l844_84418


namespace units_digit_of_p_l844_84406

/-- The units digit of a natural number -/
def unitsDigit (n : ℕ) : ℕ := n % 10

/-- Predicate for an integer having a positive units digit -/
def hasPositiveUnitsDigit (n : ℤ) : Prop := unitsDigit n.natAbs ≠ 0

theorem units_digit_of_p (p : ℤ) 
  (hp_pos : hasPositiveUnitsDigit p)
  (hp_cube_square : unitsDigit (p^3).natAbs = unitsDigit (p^2).natAbs)
  (hp_plus_two : unitsDigit ((p + 2).natAbs) = 8) :
  unitsDigit p.natAbs = 6 :=
sorry

end units_digit_of_p_l844_84406


namespace average_customers_per_table_l844_84459

/-- Given a restaurant scenario with tables, women, and men, calculate the average number of customers per table. -/
theorem average_customers_per_table 
  (tables : ℝ) 
  (women : ℝ) 
  (men : ℝ) 
  (h_tables : tables = 9.0) 
  (h_women : women = 7.0) 
  (h_men : men = 3.0) : 
  (women + men) / tables = 10.0 / 9.0 := by
  sorry

end average_customers_per_table_l844_84459


namespace product_percentage_of_x_l844_84413

theorem product_percentage_of_x (w x y z : ℝ) 
  (h1 : 0.45 * z = 1.2 * y)
  (h2 : y = 0.75 * x)
  (h3 : z = 0.8 * w) :
  w * y = 1.875 * x := by
sorry

end product_percentage_of_x_l844_84413


namespace cabbage_area_is_one_square_foot_l844_84467

/-- Represents the cabbage garden --/
structure CabbageGarden where
  side_length : ℝ
  num_cabbages : ℕ

/-- The increase in cabbages from last year to this year --/
def cabbage_increase : ℕ := 211

/-- The number of cabbages grown this year --/
def cabbages_this_year : ℕ := 11236

/-- Calculates the area of a square garden --/
def garden_area (g : CabbageGarden) : ℝ := g.side_length ^ 2

theorem cabbage_area_is_one_square_foot 
  (last_year : CabbageGarden) 
  (this_year : CabbageGarden) 
  (h1 : this_year.num_cabbages = cabbages_this_year)
  (h2 : this_year.num_cabbages = last_year.num_cabbages + cabbage_increase)
  (h3 : garden_area this_year - garden_area last_year = cabbage_increase) :
  (garden_area this_year - garden_area last_year) / cabbage_increase = 1 := by
  sorry

#check cabbage_area_is_one_square_foot

end cabbage_area_is_one_square_foot_l844_84467


namespace equation_has_one_integral_root_l844_84441

theorem equation_has_one_integral_root :
  ∃! (x : ℤ), x - 5 / (x - 4 : ℚ) = 2 - 5 / (x - 4 : ℚ) :=
by
  sorry

end equation_has_one_integral_root_l844_84441


namespace largest_four_digit_divisible_by_six_l844_84402

theorem largest_four_digit_divisible_by_six :
  ∀ n : ℕ, 1000 ≤ n ∧ n ≤ 9999 ∧ n % 6 = 0 → n ≤ 9960 :=
by sorry

end largest_four_digit_divisible_by_six_l844_84402


namespace earring_percentage_l844_84430

theorem earring_percentage :
  ∀ (bella_earrings monica_earrings rachel_earrings : ℕ),
    bella_earrings = 10 →
    monica_earrings = 2 * rachel_earrings →
    bella_earrings + monica_earrings + rachel_earrings = 70 →
    (bella_earrings : ℚ) / (monica_earrings : ℚ) * 100 = 25 :=
by
  sorry

end earring_percentage_l844_84430


namespace problem_solution_l844_84478

theorem problem_solution : ∃ m : ℚ, 
  (∃ x₁ x₂ : ℚ, 5*m + 3*x₁ = 1 + x₁ ∧ 2*x₂ + m = 3*m ∧ x₁ = x₂ + 2) →
  7*m^2 - 1 = 2/7 := by
  sorry

end problem_solution_l844_84478


namespace square_nonnegative_l844_84466

theorem square_nonnegative (x : ℝ) : x ^ 2 ≥ 0 := by
  sorry

end square_nonnegative_l844_84466


namespace rice_dumpling_max_profit_l844_84454

/-- A structure representing the rice dumpling problem -/
structure RiceDumplingProblem where
  purchase_price_A : ℝ
  purchase_price_B : ℝ
  selling_price_A : ℝ
  selling_price_B : ℝ
  total_purchase_cost : ℝ
  total_boxes : ℕ

/-- The profit function for the rice dumpling problem -/
def profit (p : RiceDumplingProblem) (x y : ℕ) : ℝ :=
  (p.selling_price_A - p.purchase_price_A) * x + (p.selling_price_B - p.purchase_price_B) * y

/-- Theorem stating the maximum profit for the rice dumpling problem -/
theorem rice_dumpling_max_profit (p : RiceDumplingProblem) 
  (h1 : p.purchase_price_A = 25)
  (h2 : p.purchase_price_B = 30)
  (h3 : p.selling_price_A = 32)
  (h4 : p.selling_price_B = 40)
  (h5 : p.total_purchase_cost = 1500)
  (h6 : p.total_boxes = 60) :
  ∃ (x y : ℕ), x + y = p.total_boxes ∧ x ≥ 2 * y ∧ profit p x y = 480 ∧ 
  ∀ (a b : ℕ), a + b = p.total_boxes → a ≥ 2 * b → profit p a b ≤ 480 :=
by sorry

#check rice_dumpling_max_profit

end rice_dumpling_max_profit_l844_84454


namespace sum_of_fractions_inequality_l844_84496

theorem sum_of_fractions_inequality (a b c : ℝ) 
  (ha : a > 0) (hb : b > 0) (hc : c > 0) : 
  (a / (b + c)) + (b / (a + c)) + (c / (a + b)) ≥ (3 : ℝ) / 2 := by
  sorry

end sum_of_fractions_inequality_l844_84496


namespace min_rental_cost_l844_84461

/-- Represents the number of buses of type A -/
def x : ℕ := sorry

/-- Represents the number of buses of type B -/
def y : ℕ := sorry

/-- The total number of passengers -/
def total_passengers : ℕ := 900

/-- The capacity of a type A bus -/
def capacity_A : ℕ := 36

/-- The capacity of a type B bus -/
def capacity_B : ℕ := 60

/-- The rental cost of a type A bus -/
def cost_A : ℕ := 1600

/-- The rental cost of a type B bus -/
def cost_B : ℕ := 2400

/-- The maximum total number of buses allowed -/
def max_buses : ℕ := 21

theorem min_rental_cost :
  (∃ x y : ℕ,
    x * capacity_A + y * capacity_B ≥ total_passengers ∧
    x + y ≤ max_buses ∧
    y ≤ x + 7 ∧
    ∀ a b : ℕ,
      (a * capacity_A + b * capacity_B ≥ total_passengers ∧
       a + b ≤ max_buses ∧
       b ≤ a + 7) →
      x * cost_A + y * cost_B ≤ a * cost_A + b * cost_B) ∧
  (∀ x y : ℕ,
    x * capacity_A + y * capacity_B ≥ total_passengers ∧
    x + y ≤ max_buses ∧
    y ≤ x + 7 →
    x * cost_A + y * cost_B ≥ 36800) :=
by sorry

end min_rental_cost_l844_84461


namespace distance_between_points_l844_84423

theorem distance_between_points : 
  let pointA : ℝ × ℝ := (1, -3)
  let pointB : ℝ × ℝ := (4, 6)
  Real.sqrt ((pointB.1 - pointA.1)^2 + (pointB.2 - pointA.2)^2) = Real.sqrt 90 := by
  sorry

end distance_between_points_l844_84423


namespace samirs_age_in_five_years_l844_84486

/-- Given that Samir's age is half of Hania's age 10 years ago,
    and Hania will be 45 years old in 5 years,
    prove that Samir will be 20 years old in 5 years. -/
theorem samirs_age_in_five_years
  (samir_current_age : ℕ)
  (hania_current_age : ℕ)
  (samir_age_condition : samir_current_age = (hania_current_age - 10) / 2)
  (hania_future_age_condition : hania_current_age + 5 = 45) :
  samir_current_age + 5 = 20 := by
  sorry


end samirs_age_in_five_years_l844_84486


namespace unique_four_digit_number_l844_84421

def is_valid_number (n : ℕ) : Prop :=
  ∃ (a b c d : ℕ),
    n = a * 1000 + b * 100 + c * 10 + d ∧
    a > 0 ∧ a < 10 ∧ b < 10 ∧ c < 10 ∧ d < 10 ∧
    a + b + c + d = 18 ∧
    b + c = 7 ∧
    a - d = 3 ∧
    n % 9 = 0

theorem unique_four_digit_number :
  ∃! (n : ℕ), is_valid_number n ∧ n = 6453 := by sorry

end unique_four_digit_number_l844_84421


namespace diane_needs_38_cents_l844_84410

/-- The cost of the cookies in cents -/
def cookie_cost : ℕ := 65

/-- The amount Diane has in cents -/
def diane_has : ℕ := 27

/-- The additional amount Diane needs in cents -/
def additional_amount : ℕ := cookie_cost - diane_has

theorem diane_needs_38_cents : additional_amount = 38 := by
  sorry

end diane_needs_38_cents_l844_84410


namespace quadratic_equation_sum_squares_product_l844_84475

theorem quadratic_equation_sum_squares_product (k : ℚ) : 
  (∃ a b : ℚ, 3 * a^2 + 7 * a + k = 0 ∧ 3 * b^2 + 7 * b + k = 0 ∧ a ≠ b) →
  (∀ a b : ℚ, 3 * a^2 + 7 * a + k = 0 → 3 * b^2 + 7 * b + k = 0 → a^2 + b^2 = 3 * a * b) ↔
  k = 49 / 15 :=
by sorry

end quadratic_equation_sum_squares_product_l844_84475


namespace geometric_sequence_problem_l844_84428

-- Define a geometric sequence with positive common ratio
def is_geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, q > 0 ∧ ∀ n : ℕ, a (n + 1) = q * a n

-- State the theorem
theorem geometric_sequence_problem (a : ℕ → ℝ) 
  (h_geom : is_geometric_sequence a)
  (h_cond : a 4 * a 8 = 2 * (a 5)^2)
  (h_a2 : a 2 = 1) :
  a 1 = Real.sqrt 2 / 2 :=
sorry

end geometric_sequence_problem_l844_84428


namespace sum_reciprocals_equals_one_l844_84436

theorem sum_reciprocals_equals_one 
  (a b c d : ℝ) 
  (ω : ℂ) 
  (ha : a ≠ -1) 
  (hb : b ≠ -1) 
  (hc : c ≠ -1) 
  (hd : d ≠ -1) 
  (hω1 : ω^4 = 1) 
  (hω2 : ω ≠ 1) 
  (h : (1 / (a + ω)) + (1 / (b + ω)) + (1 / (c + ω)) + (1 / (d + ω)) = 3 / (ω + 1)) : 
  (1 / (a + 1)) + (1 / (b + 1)) + (1 / (c + 1)) + (1 / (d + 1)) = 1 := by
  sorry

end sum_reciprocals_equals_one_l844_84436


namespace simplify_fourth_root_l844_84433

theorem simplify_fourth_root : 
  (2^5 * 5^3 : ℝ)^(1/4) = 2 * (250 : ℝ)^(1/4) := by sorry

end simplify_fourth_root_l844_84433


namespace log_equation_implies_relationship_l844_84426

theorem log_equation_implies_relationship (c d y : ℝ) (hc : c > 0) (hd : d > 0) (hy : y > 0) (hy1 : y ≠ 1) :
  9 * (Real.log y / Real.log c)^2 + 5 * (Real.log y / Real.log d)^2 = 18 * (Real.log y)^2 / (Real.log c * Real.log d) →
  d = c^(1/Real.sqrt 3) ∨ d = c^(Real.sqrt 3) ∨ d = c^(1/Real.sqrt 0.6) ∨ d = c^(Real.sqrt 0.6) := by
sorry

end log_equation_implies_relationship_l844_84426


namespace female_democrats_count_l844_84497

theorem female_democrats_count (total : ℕ) (female : ℕ) (male : ℕ) :
  total = 660 →
  female + male = total →
  (female / 2 : ℚ) + (male / 4 : ℚ) = (total : ℚ) / 3 →
  female / 2 = 110 :=
by sorry

end female_democrats_count_l844_84497


namespace concentric_circles_radius_l844_84400

/-- Given a configuration of two concentric circles and four identical circles
    tangent to each other and the concentric circles, if the radius of the smaller
    concentric circle is 1, then the radius of the larger concentric circle is 3 + 2√2. -/
theorem concentric_circles_radius (r : ℝ) : 
  r > 0 ∧ 
  r^2 - 2*r - 1 = 0 → 
  1 + 2*r = 3 + 2*Real.sqrt 2 :=
by sorry

end concentric_circles_radius_l844_84400


namespace inscribed_cube_surface_area_l844_84403

/-- Given a cube with a sphere inscribed within it, and another cube inscribed within the sphere, 
    this theorem relates the surface areas of the outer and inner cubes. -/
theorem inscribed_cube_surface_area 
  (outer_cube_surface_area : ℝ) 
  (inner_cube_surface_area : ℝ) 
  (h_outer : outer_cube_surface_area = 54) :
  inner_cube_surface_area = 18 :=
sorry

#check inscribed_cube_surface_area

end inscribed_cube_surface_area_l844_84403


namespace quadratic_integer_root_l844_84494

/-- The quadratic equation kx^2 - 2(3k - 1)x + 9k - 1 = 0 has at least one integer root
    if and only if k is -3 or -7. -/
theorem quadratic_integer_root (k : ℤ) : 
  (∃ x : ℤ, k * x^2 - 2*(3*k - 1)*x + 9*k - 1 = 0) ↔ (k = -3 ∨ k = -7) :=
sorry

end quadratic_integer_root_l844_84494
