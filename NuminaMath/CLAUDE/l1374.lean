import Mathlib

namespace positive_sum_of_odd_monotonic_increasing_l1374_137434

-- Define a monotonic increasing function
def MonotonicIncreasing (f : ℝ → ℝ) : Prop :=
  ∀ x y, x ≤ y → f x ≤ f y

-- Define an odd function
def OddFunction (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

-- Define an arithmetic sequence
def ArithmeticSequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem positive_sum_of_odd_monotonic_increasing 
  (f : ℝ → ℝ) 
  (a : ℕ → ℝ) 
  (h_mono : MonotonicIncreasing f) 
  (h_odd : OddFunction f) 
  (h_arith : ArithmeticSequence a) 
  (h_a3_pos : a 3 > 0) : 
  f (a 1) + f (a 3) + f (a 5) > 0 :=
by sorry

end positive_sum_of_odd_monotonic_increasing_l1374_137434


namespace right_triangle_segment_ratio_l1374_137495

/-- Given a right triangle with sides a and b, hypotenuse c, and a perpendicular from
    the right angle vertex dividing c into segments r and s, prove that if a : b = 2 : 3,
    then r : s = 4 : 9. -/
theorem right_triangle_segment_ratio (a b c r s : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : c > 0)
    (h4 : r > 0) (h5 : s > 0) (h6 : a^2 + b^2 = c^2) (h7 : r + s = c) (h8 : r * c = a^2)
    (h9 : s * c = b^2) (h10 : a / b = 2 / 3) : r / s = 4 / 9 := by
  sorry

end right_triangle_segment_ratio_l1374_137495


namespace min_distance_parabola_to_line_l1374_137479

/-- The minimum distance from a point on the parabola y = x^2 to the line 2x - y - 11 = 0 is 2√5 -/
theorem min_distance_parabola_to_line : 
  let parabola := {(x, y) : ℝ × ℝ | y = x^2}
  let line := {(x, y) : ℝ × ℝ | 2*x - y - 11 = 0}
  ∃ d : ℝ, d = 2 * Real.sqrt 5 ∧ 
    (∀ p ∈ parabola, ∀ q ∈ line, d ≤ Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)) ∧
    (∃ p ∈ parabola, ∃ q ∈ line, d = Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)) :=
by sorry


end min_distance_parabola_to_line_l1374_137479


namespace parametric_eq_represents_line_l1374_137408

/-- Prove that the given parametric equations represent the line x + y - 2 = 0 --/
theorem parametric_eq_represents_line :
  ∀ (t : ℝ), (3 + t) + (1 - t) - 2 = 0 := by
  sorry

end parametric_eq_represents_line_l1374_137408


namespace chicken_surprise_weight_theorem_l1374_137455

/-- The weight of one serving of Chicken Surprise -/
def chicken_surprise_serving_weight (total_servings : ℕ) (chicken_weight_pounds : ℚ) (stuffing_weight_ounces : ℕ) : ℚ :=
  (chicken_weight_pounds * 16 + stuffing_weight_ounces) / total_servings

/-- Theorem: Given 12 servings of Chicken Surprise, 4.5 pounds of chicken, and 24 ounces of stuffing, one serving of Chicken Surprise is 8 ounces. -/
theorem chicken_surprise_weight_theorem :
  chicken_surprise_serving_weight 12 (9/2) 24 = 8 := by
  sorry

end chicken_surprise_weight_theorem_l1374_137455


namespace product_inequality_l1374_137497

theorem product_inequality (x y : ℝ) (hx : x > 0) (hy : y > 0) (hsum : x + y = 1) :
  (1 + 1/x) * (1 + 1/y) ≥ 9 := by
  sorry

end product_inequality_l1374_137497


namespace isosceles_triangle_perimeter_l1374_137475

/-- An isosceles triangle with side lengths a and b satisfying a certain equation -/
structure IsoscelesTriangle where
  a : ℝ
  b : ℝ
  isosceles : True  -- We don't need to specify which sides are equal for this problem
  equation : Real.sqrt (2 * a - 3 * b + 5) + (2 * a + 3 * b - 13)^2 = 0

/-- The perimeter of an isosceles triangle -/
def perimeter (t : IsoscelesTriangle) : ℝ :=
  t.a + t.a + t.b

/-- Theorem stating that the perimeter is either 7 or 8 -/
theorem isosceles_triangle_perimeter (t : IsoscelesTriangle) :
  perimeter t = 7 ∨ perimeter t = 8 := by
  sorry

end isosceles_triangle_perimeter_l1374_137475


namespace equation_solution_l1374_137496

theorem equation_solution (x : ℚ) : (40 : ℚ) / 60 = Real.sqrt (x / 60) → x = 80 / 3 := by
  sorry

end equation_solution_l1374_137496


namespace tangent_line_at_1_l1374_137403

-- Define the function f
def f (x : ℝ) : ℝ := -(x^3) + x^2

-- Define the derivative of f
def f' (x : ℝ) : ℝ := -3*x^2 + 2*x

-- Theorem statement
theorem tangent_line_at_1 :
  let x₀ : ℝ := 1
  let y₀ : ℝ := f x₀
  let m : ℝ := f' x₀
  ∀ x y : ℝ, y - y₀ = m * (x - x₀) ↔ y = -x + 1 :=
by sorry

end tangent_line_at_1_l1374_137403


namespace trig_identity_proof_l1374_137425

theorem trig_identity_proof :
  Real.cos (10 * π / 180) * Real.cos (20 * π / 180) - 
  Real.cos (80 * π / 180) * Real.sin (20 * π / 180) = 
  Real.sqrt 3 / 2 := by
  sorry

end trig_identity_proof_l1374_137425


namespace cos_rational_angle_irrational_l1374_137423

open Real

theorem cos_rational_angle_irrational (p q : ℤ) (h : q ≠ 0) :
  let x := cos (p / q * π)
  x ≠ 0 ∧ x ≠ 1/2 ∧ x ≠ -1/2 ∧ x ≠ 1 ∧ x ≠ -1 → Irrational x :=
by sorry

end cos_rational_angle_irrational_l1374_137423


namespace sum_of_y_coordinates_is_negative_six_l1374_137404

/-- A circle passes through points (2,0) and (4,0) and is tangent to the line y = x. -/
def CircleThroughPointsAndTangentToLine : Prop :=
  ∃ (center : ℝ × ℝ) (radius : ℝ),
    (center.1 - 2)^2 + center.2^2 = radius^2 ∧
    (center.1 - 4)^2 + center.2^2 = radius^2 ∧
    (|center.1 - center.2| / Real.sqrt 2) = radius

/-- The sum of all possible y-coordinates of the center of the circle is -6. -/
theorem sum_of_y_coordinates_is_negative_six
  (h : CircleThroughPointsAndTangentToLine) :
  ∃ (y₁ y₂ : ℝ), y₁ + y₂ = -6 ∧
    ∀ (center : ℝ × ℝ) (radius : ℝ),
      (center.1 - 2)^2 + center.2^2 = radius^2 →
      (center.1 - 4)^2 + center.2^2 = radius^2 →
      (|center.1 - center.2| / Real.sqrt 2) = radius →
      center.2 = y₁ ∨ center.2 = y₂ :=
sorry

end sum_of_y_coordinates_is_negative_six_l1374_137404


namespace two_times_two_thousand_fifteen_minus_two_thousand_fifteen_l1374_137401

theorem two_times_two_thousand_fifteen_minus_two_thousand_fifteen : 2 * 2015 - 2015 = 2015 := by
  sorry

end two_times_two_thousand_fifteen_minus_two_thousand_fifteen_l1374_137401


namespace fraction_subtraction_l1374_137430

theorem fraction_subtraction : 
  (2 + 4 + 6 + 8) / (1 + 3 + 5 + 7) - (1 + 3 + 5 + 7) / (2 + 4 + 6 + 8) = 9 / 20 := by
  sorry

end fraction_subtraction_l1374_137430


namespace collinear_points_k_value_l1374_137468

/-- A point in 2D space -/
structure Point where
  x : ℚ
  y : ℚ

/-- Check if three points are collinear -/
def collinear (p1 p2 p3 : Point) : Prop :=
  (p2.y - p1.y) * (p3.x - p2.x) = (p3.y - p2.y) * (p2.x - p1.x)

theorem collinear_points_k_value :
  ∀ k : ℚ,
  let p1 : Point := ⟨2, -1⟩
  let p2 : Point := ⟨10, k⟩
  let p3 : Point := ⟨23, 4⟩
  collinear p1 p2 p3 → k = 19 / 21 := by
  sorry

end collinear_points_k_value_l1374_137468


namespace min_value_cube_square_sum_l1374_137441

theorem min_value_cube_square_sum (x y z : ℝ) 
  (h_non_neg : x ≥ 0 ∧ y ≥ 0 ∧ z ≥ 0) 
  (h_constraint : 5*x + 16*y + 33*z ≥ 136) : 
  x^3 + y^3 + z^3 + x^2 + y^2 + z^2 ≥ 50 := by
  sorry

end min_value_cube_square_sum_l1374_137441


namespace inequalities_proof_l1374_137413

theorem inequalities_proof (a b : ℝ) (h1 : a > b) (h2 : b ≥ 2) : 
  (b^2 > 3*b - a) ∧ (a*b > a + b) := by
  sorry

end inequalities_proof_l1374_137413


namespace log_xy_value_l1374_137448

theorem log_xy_value (x y : ℝ) (h1 : Real.log (x * y^2) = 1) (h2 : Real.log (x^2 * y) = 1) :
  Real.log (x * y) = 2/3 := by sorry

end log_xy_value_l1374_137448


namespace necessary_but_not_sufficient_l1374_137442

theorem necessary_but_not_sufficient (a b c d : ℝ) :
  ((a > b ∧ c > d) → a + c > b + d) ∧
  (∃ a b c d : ℝ, a + c > b + d ∧ ¬(a > b ∧ c > d)) :=
by sorry

end necessary_but_not_sufficient_l1374_137442


namespace reciprocal_problems_l1374_137418

theorem reciprocal_problems :
  (1 / 1.5 = 2/3) ∧ (1 / 1 = 1) := by
  sorry

end reciprocal_problems_l1374_137418


namespace room_length_proof_l1374_137402

theorem room_length_proof (width : ℝ) (total_cost : ℝ) (cost_per_sqm : ℝ) :
  width = 3.75 →
  total_cost = 28875 →
  cost_per_sqm = 1400 →
  (total_cost / cost_per_sqm) / width = 5.5 := by
  sorry

end room_length_proof_l1374_137402


namespace height_of_d_l1374_137409

theorem height_of_d (h_abc : ℝ) (h_abcd : ℝ) 
  (avg_abc : (h_abc + h_abc + h_abc) / 3 = 130)
  (avg_abcd : (h_abc + h_abc + h_abc + h_abcd) / 4 = 126) :
  h_abcd = 114 := by
  sorry

end height_of_d_l1374_137409


namespace bob_initial_bushels_bob_extra_ears_l1374_137482

/-- Represents the number of ears of corn in a bushel -/
def ears_per_bushel : ℕ := 14

/-- Represents the number of ears of corn Bob has left after giving some away -/
def ears_left : ℕ := 357

/-- Represents the minimum number of full bushels Bob has left -/
def min_bushels_left : ℕ := ears_left / ears_per_bushel

/-- Theorem stating that Bob initially had at least 25 bushels of corn -/
theorem bob_initial_bushels :
  min_bushels_left ≥ 25 := by
  sorry

/-- Theorem stating that Bob has some extra ears that don't make up a full bushel -/
theorem bob_extra_ears :
  ears_left % ears_per_bushel > 0 := by
  sorry

end bob_initial_bushels_bob_extra_ears_l1374_137482


namespace jim_scuba_diving_bags_l1374_137494

/-- The number of smaller bags Jim found while scuba diving -/
def number_of_smaller_bags : ℕ := by sorry

theorem jim_scuba_diving_bags :
  let hours_diving : ℕ := 8
  let coins_per_hour : ℕ := 25
  let treasure_chest_coins : ℕ := 100
  let total_coins := hours_diving * coins_per_hour
  let remaining_coins := total_coins - treasure_chest_coins
  let coins_per_smaller_bag := treasure_chest_coins / 2
  number_of_smaller_bags = remaining_coins / coins_per_smaller_bag :=
by sorry

end jim_scuba_diving_bags_l1374_137494


namespace lamplighter_monkey_distance_l1374_137498

/-- Represents the speed and duration of a monkey's movement. -/
structure MonkeyMovement where
  speed : ℝ
  duration : ℝ

/-- Calculates the total distance traveled by a Lamplighter monkey. -/
def totalDistance (running : MonkeyMovement) (swinging : MonkeyMovement) : ℝ :=
  running.speed * running.duration + swinging.speed * swinging.duration

/-- Theorem: A Lamplighter monkey travels 175 feet given the specified conditions. -/
theorem lamplighter_monkey_distance :
  let running : MonkeyMovement := ⟨15, 5⟩
  let swinging : MonkeyMovement := ⟨10, 10⟩
  totalDistance running swinging = 175 := by
  sorry

#eval totalDistance ⟨15, 5⟩ ⟨10, 10⟩

end lamplighter_monkey_distance_l1374_137498


namespace original_recipe_eggs_l1374_137457

/-- The number of eggs needed for an eight-person cake -/
def eggs_for_eight : ℕ := 3 + 1

/-- The number of people the original recipe serves -/
def original_servings : ℕ := 4

/-- The number of people Tyler wants to serve -/
def target_servings : ℕ := 8

/-- The number of eggs required for the original recipe -/
def eggs_for_original : ℕ := eggs_for_eight / 2

theorem original_recipe_eggs :
  eggs_for_original = 2 :=
sorry

end original_recipe_eggs_l1374_137457


namespace floor_equation_solution_l1374_137446

theorem floor_equation_solution (x : ℝ) : 
  ⌊⌊3 * x⌋ - 1/2⌋ = ⌊x + 3⌋ ↔ 4/3 ≤ x ∧ x < 7/3 :=
sorry

end floor_equation_solution_l1374_137446


namespace arithmetic_sequence_12th_term_l1374_137474

/-- An arithmetic sequence -/
def ArithmeticSequence (a : ℕ → ℚ) : Prop :=
  ∃ d : ℚ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_12th_term
  (a : ℕ → ℚ)
  (h_arith : ArithmeticSequence a)
  (h_4th : a 4 = 6)
  (h_sum : a 3 + a 5 = a 10) :
  a 12 = 14 := by
sorry

end arithmetic_sequence_12th_term_l1374_137474


namespace sum_of_cubes_l1374_137443

theorem sum_of_cubes (x y : ℝ) (h1 : x + y = 8) (h2 : x * y = 14) : 
  x^3 + y^3 = 176 := by sorry

end sum_of_cubes_l1374_137443


namespace initial_average_calculation_l1374_137407

theorem initial_average_calculation (n : ℕ) (wrong_mark correct_mark : ℝ) (corrected_avg : ℝ) :
  n = 30 ∧ wrong_mark = 90 ∧ correct_mark = 15 ∧ corrected_avg = 57.5 →
  (n * corrected_avg + (wrong_mark - correct_mark)) / n = 60 := by
  sorry

end initial_average_calculation_l1374_137407


namespace speed_difference_l1374_137439

/-- Given distances and times for cycling and walking, prove the speed difference --/
theorem speed_difference (school_distance : ℝ) (cycle_time : ℝ) 
  (park_distance : ℝ) (walk_time : ℝ) 
  (h1 : school_distance = 9.3) 
  (h2 : cycle_time = 0.6)
  (h3 : park_distance = 0.9)
  (h4 : walk_time = 0.2) :
  (school_distance / cycle_time) - (park_distance / walk_time) = 11 := by
  sorry


end speed_difference_l1374_137439


namespace rose_rice_problem_l1374_137426

theorem rose_rice_problem (x : ℚ) : 
  (10000 * (1 - x) * (3/4) = 750) → x = 9/10 := by
  sorry

end rose_rice_problem_l1374_137426


namespace f_upper_bound_f_max_value_condition_l1374_137485

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := a * x + 3 - abs (2 * x - 1)

-- Part 1
theorem f_upper_bound :
  ∀ x : ℝ, (x ≤ 0 ∨ x ≥ 2) → f 1 x ≤ 2 :=
sorry

-- Part 2
theorem f_max_value_condition :
  ∀ a : ℝ, (∃ x : ℝ, ∀ y : ℝ, f a y ≤ f a x) ↔ -2 ≤ a ∧ a ≤ 2 :=
sorry

end f_upper_bound_f_max_value_condition_l1374_137485


namespace sector_properties_l1374_137467

-- Define the sector
def Sector (R : ℝ) (α : ℝ) : Prop :=
  R > 0 ∧ α > 0 ∧ (1 / 2) * R^2 * α = 1 ∧ 2 * R + R * α = 4

-- Theorem statement
theorem sector_properties :
  ∃ (R α : ℝ), Sector R α ∧ α = 2 ∧ 2 * Real.sin 1 = 2 * R * Real.sin (α / 2) :=
sorry

end sector_properties_l1374_137467


namespace x_varies_linearly_with_z_l1374_137471

/-- Given that x varies as the cube of y and y varies as the cube root of z,
    prove that x varies linearly with z. -/
theorem x_varies_linearly_with_z 
  (k : ℝ) (j : ℝ) (x y z : ℝ → ℝ) 
  (h1 : ∀ t, x t = k * (y t)^3) 
  (h2 : ∀ t, y t = j * (z t)^(1/3)) :
  ∃ m : ℝ, ∀ t, x t = m * z t :=
sorry

end x_varies_linearly_with_z_l1374_137471


namespace a_nine_equals_a_three_times_a_seven_l1374_137470

def exponential_sequence (a : ℕ → ℝ) (q : ℝ) : Prop :=
  a 1 = q ∧ ∀ n : ℕ, a (n + 1) = q * a n

theorem a_nine_equals_a_three_times_a_seven
  (a : ℕ → ℝ) (q : ℝ) (h : exponential_sequence a q) :
  a 9 = a 3 * a 7 := by
  sorry

end a_nine_equals_a_three_times_a_seven_l1374_137470


namespace year_2020_is_gengzi_l1374_137480

/-- Represents the Heavenly Stems in the Sexagenary Cycle -/
inductive HeavenlyStem
| Jia | Yi | Bing | Ding | Wu | Ji | Geng | Xin | Ren | Gui

/-- Represents the Earthly Branches in the Sexagenary Cycle -/
inductive EarthlyBranch
| Zi | Chou | Yin | Mao | Chen | Si | Wu | Wei | Shen | You | Xu | Hai

/-- Represents a year in the Sexagenary Cycle -/
structure SexagenaryYear :=
  (stem : HeavenlyStem)
  (branch : EarthlyBranch)

/-- The Sexagenary Cycle system -/
def sexagenaryCycle : List SexagenaryYear := sorry

/-- Function to get the Sexagenary year for a given Gregorian year -/
def getSexagenaryYear (gregorianYear : Nat) : SexagenaryYear := sorry

/-- Theorem stating that 2020 corresponds to the GengZi year in the Sexagenary Cycle -/
theorem year_2020_is_gengzi :
  getSexagenaryYear 2020 = SexagenaryYear.mk HeavenlyStem.Geng EarthlyBranch.Zi :=
sorry

end year_2020_is_gengzi_l1374_137480


namespace negative_inequality_l1374_137459

theorem negative_inequality (m n : ℝ) (h : m > n) : -m < -n := by
  sorry

end negative_inequality_l1374_137459


namespace max_diff_of_squares_l1374_137476

theorem max_diff_of_squares (n : ℕ) (h1 : n > 0) (h2 : n + (n + 1) < 150) :
  (∃ (m : ℕ), m > 0 ∧ m + (m + 1) < 150 ∧ (m + 1)^2 - m^2 > (n + 1)^2 - n^2) →
  (n + 1)^2 - n^2 ≤ 149 :=
sorry

end max_diff_of_squares_l1374_137476


namespace complex_square_root_expression_l1374_137489

theorem complex_square_root_expression : 
  (2 * Real.sqrt 12 - 4 * Real.sqrt 27 + 3 * Real.sqrt 75 + 7 * Real.sqrt 8 - 3 * Real.sqrt 18) * 
  (4 * Real.sqrt 48 - 3 * Real.sqrt 27 - 5 * Real.sqrt 18 + 2 * Real.sqrt 50) = 97 := by
  sorry

end complex_square_root_expression_l1374_137489


namespace swap_values_l1374_137493

/-- Swaps the values of two variables using an intermediate variable -/
theorem swap_values (a b : ℕ) : 
  let a_init := a
  let b_init := b
  let c := a_init
  let a_new := b_init
  let b_new := c
  (a_new = b_init ∧ b_new = a_init) := by sorry

end swap_values_l1374_137493


namespace inequalities_solution_l1374_137415

theorem inequalities_solution (x : ℝ) : 
  (2 * (-x + 2) > -3 * x + 5 → x > 1) ∧
  ((7 - x) / 3 ≤ (x + 2) / 2 + 1 → x ≥ 2 / 5) := by
sorry

end inequalities_solution_l1374_137415


namespace sophies_shopping_l1374_137431

theorem sophies_shopping (total_budget : ℚ) (trouser_cost : ℚ) (additional_items : ℕ) (additional_item_cost : ℚ) (num_shirts : ℕ) :
  total_budget = 260 →
  trouser_cost = 63 →
  additional_items = 4 →
  additional_item_cost = 40 →
  num_shirts = 2 →
  ∃ (shirt_cost : ℚ), 
    shirt_cost * num_shirts + trouser_cost + (additional_items : ℚ) * additional_item_cost = total_budget ∧
    shirt_cost = 37 / 2 := by
  sorry

end sophies_shopping_l1374_137431


namespace math_competition_schools_l1374_137490

/- Define the problem setup -/
structure MathCompetition where
  num_students_per_school : ℕ
  andrea_rank : ℕ
  beth_rank : ℕ
  carla_rank : ℕ

/- Define the conditions -/
def valid_competition (comp : MathCompetition) : Prop :=
  comp.num_students_per_school = 4 ∧
  comp.andrea_rank < comp.beth_rank ∧
  comp.andrea_rank < comp.carla_rank ∧
  comp.beth_rank = 48 ∧
  comp.carla_rank = 75

/- Define Andrea's rank as the median -/
def andrea_is_median (comp : MathCompetition) (total_students : ℕ) : Prop :=
  comp.andrea_rank = (total_students + 1) / 2 ∨
  comp.andrea_rank = (total_students + 2) / 2

/- Theorem statement -/
theorem math_competition_schools (comp : MathCompetition) :
  valid_competition comp →
  ∃ (total_students : ℕ),
    andrea_is_median comp total_students ∧
    total_students % comp.num_students_per_school = 0 ∧
    total_students / comp.num_students_per_school = 23 :=
sorry

end math_competition_schools_l1374_137490


namespace initial_average_production_l1374_137436

/-- Given a company's production data, calculate the initial average daily production. -/
theorem initial_average_production
  (n : ℕ) -- number of days of initial production
  (today_production : ℕ) -- today's production in units
  (new_average : ℚ) -- new average including today's production
  (hn : n = 11)
  (ht : today_production = 110)
  (ha : new_average = 55)
  : (n : ℚ) * (n + 1 : ℚ) * new_average - (n + 1 : ℚ) * today_production = n * 50
  := by sorry

end initial_average_production_l1374_137436


namespace ringbinder_price_decrease_l1374_137499

def original_backpack_price : ℝ := 50
def original_ringbinder_price : ℝ := 20
def backpack_price_increase : ℝ := 5
def num_ringbinders : ℕ := 3
def total_spent : ℝ := 109

theorem ringbinder_price_decrease :
  ∃ (x : ℝ),
    x = 2 ∧
    (original_backpack_price + backpack_price_increase) +
    num_ringbinders * (original_ringbinder_price - x) = total_spent :=
by sorry

end ringbinder_price_decrease_l1374_137499


namespace verify_conditions_max_boxes_A_l1374_137417

/-- Represents the price of a box of paint model A in yuan -/
def price_A : ℕ := 24

/-- Represents the price of a box of paint model B in yuan -/
def price_B : ℕ := 16

/-- Represents the total number of boxes to be purchased -/
def total_boxes : ℕ := 200

/-- Represents the maximum total cost in yuan -/
def max_cost : ℕ := 3920

/-- Verification of the given conditions -/
theorem verify_conditions : 
  price_A + 2 * price_B = 56 ∧ 
  2 * price_A + price_B = 64 := by sorry

/-- Theorem stating the maximum number of boxes of paint A that can be purchased -/
theorem max_boxes_A : 
  (∀ m : ℕ, m ≤ total_boxes → 
    m * price_A + (total_boxes - m) * price_B ≤ max_cost → 
    m ≤ 90) ∧ 
  90 * price_A + (total_boxes - 90) * price_B ≤ max_cost := by sorry

end verify_conditions_max_boxes_A_l1374_137417


namespace system_solution_l1374_137444

theorem system_solution (x y : ℚ) :
  (3 * x - 7 * y = 31) ∧ (5 * x + 2 * y = -10) → x = -336/205 := by
  sorry

end system_solution_l1374_137444


namespace linear_system_solution_l1374_137462

theorem linear_system_solution (m : ℕ) (x y : ℝ) : 
  (2 * x - y = 4 * m - 5) →
  (x + 4 * y = -7 * m + 2) →
  (x + y > -3) →
  (m = 0 ∨ m = 1) :=
by sorry

end linear_system_solution_l1374_137462


namespace cookies_theorem_l1374_137451

def cookies_problem (initial : ℕ) (first_friend : ℕ) (second_friend : ℕ) (eaten : ℕ) (bought : ℕ) (third_friend : ℕ) : Prop :=
  let remaining_after_first := initial - first_friend
  let remaining_after_second := remaining_after_first - second_friend
  let remaining_after_eating := remaining_after_second - eaten
  let remaining_after_buying := remaining_after_eating + bought
  let final_remaining := remaining_after_buying - third_friend
  final_remaining = 67

theorem cookies_theorem : cookies_problem 120 34 29 20 45 15 := by
  sorry

end cookies_theorem_l1374_137451


namespace perpendicular_line_x_intercept_l1374_137437

/-- Given a line L1 defined by 2x + 3y = 9, and another line L2 that is perpendicular to L1
    with a y-intercept of -4, the x-intercept of L2 is 8/3. -/
theorem perpendicular_line_x_intercept :
  ∀ (L1 L2 : Set (ℝ × ℝ)),
  (∀ x y, (x, y) ∈ L1 ↔ 2 * x + 3 * y = 9) →
  (∃ m : ℝ, ∀ x y, (x, y) ∈ L2 ↔ y = m * x - 4) →
  (∀ x y₁ y₂, (x, y₁) ∈ L1 ∧ (x, y₂) ∈ L2 → (y₁ - y₂) * (x - 0) = -(1 : ℝ)) →
  (∃ x : ℝ, (x, 0) ∈ L2 ∧ x = 8 / 3) :=
by sorry

end perpendicular_line_x_intercept_l1374_137437


namespace det_special_matrix_l1374_137438

/-- The determinant of the matrix [[1, a, b], [1, a+b, b+c], [1, a, a+c]] is ab + b^2 + bc -/
theorem det_special_matrix (a b c : ℝ) : 
  Matrix.det ![![1, a, b], ![1, a+b, b+c], ![1, a, a+c]] = a*b + b^2 + b*c := by
  sorry

end det_special_matrix_l1374_137438


namespace stating_initial_amount_is_200_l1374_137427

/-- Represents the exchange rate from U.S. dollars to Canadian dollars -/
def exchange_rate : ℚ := 6 / 5

/-- Represents the amount spent in Canadian dollars -/
def amount_spent : ℚ := 80

/-- 
Given an initial amount of U.S. dollars, calculates the remaining amount 
of Canadian dollars after exchanging and spending
-/
def remaining_amount (d : ℚ) : ℚ := (4 / 5) * d

/-- 
Theorem stating that given the exchange rate and spending conditions, 
the initial amount of U.S. dollars is 200
-/
theorem initial_amount_is_200 : 
  ∃ d : ℚ, d = 200 ∧ 
  exchange_rate * d - amount_spent = remaining_amount d :=
sorry

end stating_initial_amount_is_200_l1374_137427


namespace right_triangle_acute_angle_theorem_l1374_137405

theorem right_triangle_acute_angle_theorem :
  ∀ (x y : ℝ),
  x > 0 ∧ y > 0 →
  x + y = 90 →
  y = 5 * x →
  y = 75 :=
by
  sorry

end right_triangle_acute_angle_theorem_l1374_137405


namespace sum_interior_angles_regular_polygon_l1374_137406

/-- For a regular polygon where each exterior angle measures 20 degrees, 
    the sum of the measures of its interior angles is 2880 degrees. -/
theorem sum_interior_angles_regular_polygon : 
  ∀ (n : ℕ), 
    n > 2 → 
    (360 : ℝ) / n = 20 → 
    (n - 2 : ℝ) * 180 = 2880 := by
  sorry

end sum_interior_angles_regular_polygon_l1374_137406


namespace rebecca_eggs_l1374_137483

theorem rebecca_eggs (num_groups : ℕ) (eggs_per_group : ℕ) 
  (h1 : num_groups = 11) (h2 : eggs_per_group = 2) : 
  num_groups * eggs_per_group = 22 := by
sorry

end rebecca_eggs_l1374_137483


namespace bird_sale_ratio_is_half_l1374_137484

/-- Represents the initial counts of animals in the pet store -/
structure InitialCounts where
  birds : ℕ
  puppies : ℕ
  cats : ℕ
  spiders : ℕ

/-- Represents the changes in animal counts -/
structure Changes where
  puppies_adopted : ℕ
  spiders_loose : ℕ

/-- Calculates the ratio of birds sold to initial birds -/
def bird_sale_ratio (initial : InitialCounts) (changes : Changes) (final_count : ℕ) : ℚ :=
  let total_initial := initial.birds + initial.puppies + initial.cats + initial.spiders
  let birds_sold := total_initial - changes.puppies_adopted - changes.spiders_loose - final_count
  birds_sold / initial.birds

/-- Theorem stating the ratio of birds sold to initial birds is 1:2 -/
theorem bird_sale_ratio_is_half 
  (initial : InitialCounts)
  (changes : Changes)
  (final_count : ℕ)
  (h_initial : initial = ⟨12, 9, 5, 15⟩)
  (h_changes : changes = ⟨3, 7⟩)
  (h_final : final_count = 25) :
  bird_sale_ratio initial changes final_count = 1 / 2 := by
  sorry

end bird_sale_ratio_is_half_l1374_137484


namespace laptop_final_price_l1374_137414

/-- Calculate the final price of a laptop given the original price, discount rate, tax rate, and commission rate. -/
def calculate_final_price (original_price discount_rate tax_rate commission_rate : ℝ) : ℝ :=
  let discounted_price := original_price * (1 - discount_rate)
  let price_after_tax := discounted_price * (1 + tax_rate)
  price_after_tax * (1 + commission_rate)

/-- Theorem stating that the final price of the laptop is 1199.52 dollars given the specified conditions. -/
theorem laptop_final_price :
  calculate_final_price 1200 0.15 0.12 0.05 = 1199.52 := by
  sorry

end laptop_final_price_l1374_137414


namespace youtube_likes_problem_l1374_137411

theorem youtube_likes_problem (likes dislikes : ℕ) : 
  dislikes = likes / 2 + 100 →
  dislikes + 1000 = 2600 →
  likes = 3000 := by
sorry

end youtube_likes_problem_l1374_137411


namespace installation_rate_one_each_possible_solutions_l1374_137465

/-- Represents the number of air conditioners installed by different worker combinations -/
structure InstallationRate where
  skilled : ℕ → ℕ
  new : ℕ → ℕ

/-- Represents the total number of air conditioners to be installed -/
def total_ac : ℕ := 500

/-- Represents the number of days to complete the installation -/
def days : ℕ := 20

/-- Given conditions on installation rates -/
axiom condition1 {r : InstallationRate} : r.skilled 1 + r.new 3 = 11
axiom condition2 {r : InstallationRate} : r.skilled 2 = r.new 5

/-- Theorem stating the installation rate of 1 skilled worker and 1 new worker -/
theorem installation_rate_one_each (r : InstallationRate) : 
  r.skilled 1 + r.new 1 = 7 := by sorry

/-- Theorem stating the possible solutions for m skilled workers and n new workers -/
theorem possible_solutions (m n : ℕ) : 
  (m ≠ 0 ∧ n ≠ 0 ∧ 5 * m + 2 * n = 25) ↔ (m = 1 ∧ n = 10) ∨ (m = 3 ∧ n = 5) := by sorry

end installation_rate_one_each_possible_solutions_l1374_137465


namespace pages_read_day5_l1374_137420

def pages_day1 : ℕ := 63
def pages_day2 : ℕ := 95 -- Rounded up from 94.5
def pages_day3 : ℕ := pages_day2 + 20
def pages_day4 : ℕ := 86 -- Rounded down from 86.25
def total_pages : ℕ := 480

theorem pages_read_day5 : 
  total_pages - (pages_day1 + pages_day2 + pages_day3 + pages_day4) = 121 := by
  sorry

end pages_read_day5_l1374_137420


namespace problem_1_problem_2_l1374_137477

-- Problem 1
theorem problem_1 (a : ℝ) (h : a = Real.sqrt 3 - 1) : 
  (a^2 + a) * ((a + 1) / a) = 3 := by sorry

-- Problem 2
theorem problem_2 (a : ℝ) (h : a = 1/2) : 
  (a + 1) / (a^2 - 1) - (a + 1) / (1 - a) = -5 := by sorry

end problem_1_problem_2_l1374_137477


namespace thabos_book_collection_difference_l1374_137428

/-- Theorem: Thabo's Book Collection Difference --/
theorem thabos_book_collection_difference :
  ∀ (paperback_fiction paperback_nonfiction hardcover_nonfiction : ℕ),
  -- Total number of books is 180
  paperback_fiction + paperback_nonfiction + hardcover_nonfiction = 180 →
  -- More paperback nonfiction than hardcover nonfiction
  paperback_nonfiction > hardcover_nonfiction →
  -- Twice as many paperback fiction as paperback nonfiction
  paperback_fiction = 2 * paperback_nonfiction →
  -- 30 hardcover nonfiction books
  hardcover_nonfiction = 30 →
  -- Prove: Difference between paperback nonfiction and hardcover nonfiction is 20
  paperback_nonfiction - hardcover_nonfiction = 20 := by
  sorry

end thabos_book_collection_difference_l1374_137428


namespace monty_hall_probabilities_l1374_137416

/-- Represents the three doors in the Monty Hall problem -/
inductive Door : Type
  | door1 : Door
  | door2 : Door
  | door3 : Door

/-- Represents the possible contents behind a door -/
inductive Content : Type
  | car : Content
  | goat : Content

/-- The Monty Hall game setup -/
structure MontyHallGame where
  prize_door : Door
  initial_choice : Door
  opened_door : Door
  h_prize_not_opened : opened_door ≠ prize_door
  h_opened_is_goat : opened_door ≠ initial_choice

/-- The probability of winning by sticking with the initial choice -/
def prob_stick_wins (game : MontyHallGame) : ℚ :=
  1 / 3

/-- The probability of winning by switching doors -/
def prob_switch_wins (game : MontyHallGame) : ℚ :=
  2 / 3

theorem monty_hall_probabilities (game : MontyHallGame) :
  prob_stick_wins game = 1 / 3 ∧ prob_switch_wins game = 2 / 3 := by
  sorry

#check monty_hall_probabilities

end monty_hall_probabilities_l1374_137416


namespace special_sequence_2023_l1374_137452

/-- A sequence of positive terms with a special property -/
structure SpecialSequence where
  a : ℕ → ℕ+
  S : ℕ → ℕ
  property : ∀ n, 2 * S n = (a n).val * ((a n).val + 1)

/-- The 2023rd term of a special sequence is 2023 -/
theorem special_sequence_2023 (seq : SpecialSequence) : seq.a 2023 = ⟨2023, by sorry⟩ := by
  sorry

end special_sequence_2023_l1374_137452


namespace distribute_4_balls_3_boxes_l1374_137440

/-- The number of ways to distribute indistinguishable balls into distinguishable boxes -/
def distribute_balls (n : ℕ) (k : ℕ) : ℕ :=
  sorry

/-- Theorem: There are 15 ways to distribute 4 indistinguishable balls into 3 distinguishable boxes -/
theorem distribute_4_balls_3_boxes : distribute_balls 4 3 = 15 := by
  sorry

end distribute_4_balls_3_boxes_l1374_137440


namespace optimal_bus_rental_solution_l1374_137486

/-- Represents a bus rental problem with two types of buses -/
structure BusRental where
  tourists : ℕ
  capacity_A : ℕ
  cost_A : ℕ
  capacity_B : ℕ
  cost_B : ℕ
  max_total_buses : ℕ
  max_B_minus_A : ℕ

/-- Represents a solution to the bus rental problem -/
structure BusRentalSolution where
  buses_A : ℕ
  buses_B : ℕ
  total_cost : ℕ

/-- Check if a solution is valid for a given bus rental problem -/
def is_valid_solution (problem : BusRental) (solution : BusRentalSolution) : Prop :=
  solution.buses_A * problem.capacity_A + solution.buses_B * problem.capacity_B ≥ problem.tourists ∧
  solution.buses_A + solution.buses_B ≤ problem.max_total_buses ∧
  solution.buses_B - solution.buses_A ≤ problem.max_B_minus_A ∧
  solution.total_cost = solution.buses_A * problem.cost_A + solution.buses_B * problem.cost_B

/-- The main theorem stating that the given solution is optimal -/
theorem optimal_bus_rental_solution (problem : BusRental)
  (h_problem : problem = {
    tourists := 900,
    capacity_A := 36,
    cost_A := 1600,
    capacity_B := 60,
    cost_B := 2400,
    max_total_buses := 21,
    max_B_minus_A := 7
  })
  (solution : BusRentalSolution)
  (h_solution : solution = {
    buses_A := 5,
    buses_B := 12,
    total_cost := 36800
  }) :
  is_valid_solution problem solution ∧
  ∀ (other : BusRentalSolution), is_valid_solution problem other → other.total_cost ≥ solution.total_cost :=
by sorry


end optimal_bus_rental_solution_l1374_137486


namespace eighth_root_unity_l1374_137421

theorem eighth_root_unity : ∃ n : ℕ, n ∈ Finset.range 8 ∧
  (Complex.I + Complex.tan (π / 8)) / (Complex.tan (π / 8) - Complex.I) =
  Complex.exp (2 * n * π * Complex.I / 8) := by
  sorry

end eighth_root_unity_l1374_137421


namespace fixed_point_of_quadratic_l1374_137424

/-- The quadratic function y = -x^2 + (m-1)x + m passes through the point (-1, 0) for all real m. -/
theorem fixed_point_of_quadratic (m : ℝ) : 
  let f : ℝ → ℝ := λ x ↦ -x^2 + (m-1)*x + m
  f (-1) = 0 := by
  sorry

end fixed_point_of_quadratic_l1374_137424


namespace parallelogram_area_l1374_137488

theorem parallelogram_area (base height : ℝ) (h1 : base = 14) (h2 : height = 24) :
  base * height = 336 := by
  sorry

end parallelogram_area_l1374_137488


namespace problem_1_l1374_137469

theorem problem_1 (m n : ℕ) (h1 : 3^m = 8) (h2 : 3^n = 2) : 3^(2*m - 3*n + 1) = 24 := by
  sorry

end problem_1_l1374_137469


namespace opposite_def_opposite_of_neg_five_l1374_137491

/-- The opposite of a real number -/
def opposite (x : ℝ) : ℝ := -x

/-- The property that defines the opposite of a number -/
theorem opposite_def (x : ℝ) : x + opposite x = 0 := by sorry

/-- Proof that the opposite of -5 is 5 -/
theorem opposite_of_neg_five : opposite (-5 : ℝ) = 5 := by sorry

end opposite_def_opposite_of_neg_five_l1374_137491


namespace sin_sum_arcsin_arctan_l1374_137429

theorem sin_sum_arcsin_arctan :
  Real.sin (Real.arcsin (4/5) + Real.arctan (1/2)) = 11 * Real.sqrt 5 / 25 := by
  sorry

end sin_sum_arcsin_arctan_l1374_137429


namespace calculation_proof_l1374_137454

theorem calculation_proof : ((4 + 6 + 5) * 2) / 4 - (3 * 2 / 4) = 6 := by
  sorry

end calculation_proof_l1374_137454


namespace tall_min_voters_to_win_l1374_137435

/-- Represents the voting structure and results of the giraffe beauty contest -/
structure GiraffeContest where
  total_voters : Nat
  num_districts : Nat
  sections_per_district : Nat
  voters_per_section : Nat
  winner : String

/-- Calculates the minimum number of voters required for a giraffe to win the contest -/
def min_voters_to_win (contest : GiraffeContest) : Nat :=
  sorry

/-- The theorem stating the minimum number of voters required for Tall to win -/
theorem tall_min_voters_to_win (contest : GiraffeContest) 
  (h1 : contest.total_voters = 105)
  (h2 : contest.num_districts = 5)
  (h3 : contest.sections_per_district = 7)
  (h4 : contest.voters_per_section = 3)
  (h5 : contest.winner = "Tall") :
  min_voters_to_win contest = 24 := by
  sorry

#check tall_min_voters_to_win

end tall_min_voters_to_win_l1374_137435


namespace michaels_number_l1374_137463

theorem michaels_number (m : ℕ) :
  m % 75 = 0 ∧ m % 40 = 0 ∧ 1000 ≤ m ∧ m ≤ 3000 →
  m = 1800 ∨ m = 2400 ∨ m = 3000 := by
sorry

end michaels_number_l1374_137463


namespace fiona_probability_l1374_137453

/-- Represents a lily pad with its number and whether it contains a predator or food -/
structure LilyPad where
  number : Nat
  hasPredator : Bool
  hasFood : Bool

/-- Represents the possible moves Fiona can make -/
inductive Move
  | Forward
  | ForwardTwo
  | Backward

/-- Represents Fiona's current position and the probability of reaching that position -/
structure FionaState where
  position : Nat
  probability : Rat

/-- The probability of each move -/
def moveProbability : Rat := 1 / 3

/-- The total number of lily pads -/
def totalPads : Nat := 15

/-- Creates the initial state of the lily pads -/
def initLilyPads : List LilyPad := sorry

/-- Checks if a move is valid given Fiona's current position -/
def isValidMove (currentPos : Nat) (move : Move) : Bool := sorry

/-- Calculates Fiona's new position after a move -/
def newPosition (currentPos : Nat) (move : Move) : Nat := sorry

/-- Calculates the probability of Fiona reaching pad 13 without landing on pads 4 or 8 -/
def probReachPad13 (initialState : FionaState) (lilyPads : List LilyPad) : Rat := sorry

theorem fiona_probability :
  probReachPad13 ⟨0, 1⟩ initLilyPads = 16 / 177147 := by sorry

end fiona_probability_l1374_137453


namespace class_field_trip_budget_l1374_137456

/-- The class's budget for a field trip to the zoo --/
theorem class_field_trip_budget
  (bus_rental_cost : ℕ)
  (admission_cost_per_student : ℕ)
  (number_of_students : ℕ)
  (h1 : bus_rental_cost = 100)
  (h2 : admission_cost_per_student = 10)
  (h3 : number_of_students = 25) :
  bus_rental_cost + admission_cost_per_student * number_of_students = 350 :=
by sorry

end class_field_trip_budget_l1374_137456


namespace sum_a4_a5_a6_l1374_137460

def arithmetic_sequence (a : ℕ → ℤ) : Prop :=
  ∃ d : ℤ, ∀ n : ℕ, a (n + 1) = a n + d

theorem sum_a4_a5_a6 (a : ℕ → ℤ) :
  arithmetic_sequence a → a 1 = 2 → a 3 = -10 →
  a 4 + a 5 + a 6 = -66 := by
  sorry

end sum_a4_a5_a6_l1374_137460


namespace vector_subtraction_l1374_137400

def a : Fin 3 → ℝ := ![5, -3, 2]
def b : Fin 3 → ℝ := ![-2, 4, 1]

theorem vector_subtraction :
  (fun i => a i - 2 * b i) = ![9, -11, 0] := by sorry

end vector_subtraction_l1374_137400


namespace neg_three_point_fourteen_gt_neg_pi_l1374_137445

theorem neg_three_point_fourteen_gt_neg_pi : -3.14 > -Real.pi := by sorry

end neg_three_point_fourteen_gt_neg_pi_l1374_137445


namespace pencil_cost_is_13_l1374_137447

/-- Represents the data for the pencil purchase problem -/
structure PencilPurchaseData where
  total_students : ℕ
  buyers : ℕ
  total_cost : ℕ
  pencil_cost : ℕ
  pencils_per_student : ℕ

/-- The conditions of the pencil purchase problem -/
def pencil_purchase_conditions (data : PencilPurchaseData) : Prop :=
  data.total_students = 50 ∧
  data.buyers > data.total_students / 2 ∧
  data.pencil_cost > data.pencils_per_student ∧
  data.buyers * data.pencil_cost * data.pencils_per_student = data.total_cost ∧
  data.total_cost = 2275

/-- The theorem stating that under the given conditions, the pencil cost is 13 cents -/
theorem pencil_cost_is_13 (data : PencilPurchaseData) :
  pencil_purchase_conditions data → data.pencil_cost = 13 :=
by sorry

end pencil_cost_is_13_l1374_137447


namespace flowerpot_problem_l1374_137410

/-- Given a row of flowerpots, calculates the number of pots between two specific pots. -/
def pots_between (total : ℕ) (a_from_right : ℕ) (b_from_left : ℕ) : ℕ :=
  b_from_left - (total - a_from_right + 1) - 1

/-- Theorem stating that there are 8 flowerpots between A and B under the given conditions. -/
theorem flowerpot_problem :
  pots_between 33 14 29 = 8 := by
  sorry

end flowerpot_problem_l1374_137410


namespace total_cost_theorem_l1374_137473

def sandwich_cost : ℕ := 3
def soda_cost : ℕ := 2
def num_sandwiches : ℕ := 5
def num_sodas : ℕ := 8

theorem total_cost_theorem : 
  sandwich_cost * num_sandwiches + soda_cost * num_sodas = 31 := by
  sorry

end total_cost_theorem_l1374_137473


namespace popsicle_stick_difference_l1374_137433

theorem popsicle_stick_difference :
  let num_boys : ℕ := 10
  let num_girls : ℕ := 12
  let sticks_per_boy : ℕ := 15
  let sticks_per_girl : ℕ := 12
  let total_boys_sticks : ℕ := num_boys * sticks_per_boy
  let total_girls_sticks : ℕ := num_girls * sticks_per_girl
  total_boys_sticks - total_girls_sticks = 6 := by
  sorry

end popsicle_stick_difference_l1374_137433


namespace triangle_properties_l1374_137487

/-- Triangle ABC with sides a, b, c opposite to angles A, B, C respectively -/
structure Triangle :=
  (a b c : ℝ)
  (A B C : ℝ)

/-- The theorem statement -/
theorem triangle_properties (t : Triangle) 
  (h1 : t.b = 1)
  (h2 : 2 * Real.cos t.C - 2 * t.a - t.c = 0) :
  t.B = 2 * Real.pi / 3 ∧ 
  Real.sqrt 3 / 6 = Real.sqrt (((t.a * t.c) / (4 * Real.sin t.A)) ^ 2 - (t.b / 2) ^ 2) := by
  sorry


end triangle_properties_l1374_137487


namespace complex_modulus_l1374_137461

theorem complex_modulus (z : ℂ) (h : (1 - Complex.I) * z = 2 * Complex.I) : 
  Complex.abs z = Real.sqrt 2 := by
sorry

end complex_modulus_l1374_137461


namespace diagonal_intersections_12x17_l1374_137478

/-- Counts the number of intersection points between a diagonal and grid lines in an m × n grid -/
def countIntersections (m n : ℕ) : ℕ :=
  (n + 1) + (m + 1) - 2

/-- Theorem: In a 12 × 17 grid, the diagonal from A to B intersects the grid at 29 points -/
theorem diagonal_intersections_12x17 :
  countIntersections 12 17 = 29 := by
  sorry

end diagonal_intersections_12x17_l1374_137478


namespace cleo_utility_equality_l1374_137466

/-- Utility function for Cleo's activities -/
def utility (reading : ℝ) (painting : ℝ) : ℝ := reading * painting

/-- Time spent painting on Saturday -/
def saturday_painting (t : ℝ) : ℝ := t

/-- Time spent reading on Saturday -/
def saturday_reading (t : ℝ) : ℝ := 10 - 2 * t

/-- Time spent painting on Sunday -/
def sunday_painting (t : ℝ) : ℝ := 5 - t

/-- Time spent reading on Sunday -/
def sunday_reading (t : ℝ) : ℝ := 2 * t + 4

theorem cleo_utility_equality :
  ∃ t : ℝ, utility (saturday_reading t) (saturday_painting t) = utility (sunday_reading t) (sunday_painting t) ∧ t = 0 := by
  sorry

end cleo_utility_equality_l1374_137466


namespace mass_of_man_sinking_boat_l1374_137464

/-- The mass of a man who causes a boat to sink in water -/
theorem mass_of_man_sinking_boat (length width sinkage : ℝ) (water_density : ℝ) : 
  length = 4 →
  width = 2 →
  sinkage = 0.01 →
  water_density = 1000 →
  length * width * sinkage * water_density = 80 := by
sorry

end mass_of_man_sinking_boat_l1374_137464


namespace solve_system_l1374_137458

theorem solve_system (x y : ℝ) (h1 : x - y = 8) (h2 : x + y = 14) : x = 11 := by
  sorry

end solve_system_l1374_137458


namespace hyperbola_eccentricity_l1374_137419

/-- The eccentricity of a hyperbola with equation x²/4 - y²/12 = 1 is 2 -/
theorem hyperbola_eccentricity : ∃ e : ℝ, e = 2 ∧ 
  ∀ x y : ℝ, x^2/4 - y^2/12 = 1 → 
  ∃ a b c : ℝ, a > 0 ∧ b > 0 ∧ c > 0 ∧
  a^2 = 4 ∧ b^2 = 12 ∧ c^2 = a^2 + b^2 ∧ e = c/a :=
sorry

end hyperbola_eccentricity_l1374_137419


namespace smallest_variance_most_stable_city_D_most_stable_l1374_137481

/-- Represents a city with its cabbage price variance -/
structure City where
  name : String
  variance : ℝ

/-- Defines stability of cabbage prices based on variance -/
def is_most_stable (cities : List City) (c : City) : Prop :=
  ∀ city ∈ cities, c.variance ≤ city.variance

/-- The theorem stating that the city with the smallest variance is the most stable -/
theorem smallest_variance_most_stable (cities : List City) (c : City) 
    (h₁ : c ∈ cities) 
    (h₂ : ∀ city ∈ cities, c.variance ≤ city.variance) : 
    is_most_stable cities c := by
  sorry

/-- The specific problem instance -/
def problem_instance : List City :=
  [⟨"A", 18.3⟩, ⟨"B", 17.4⟩, ⟨"C", 20.1⟩, ⟨"D", 12.5⟩]

/-- The theorem applied to the specific problem instance -/
theorem city_D_most_stable : 
    is_most_stable problem_instance ⟨"D", 12.5⟩ := by
  sorry

end smallest_variance_most_stable_city_D_most_stable_l1374_137481


namespace suv_highway_efficiency_l1374_137492

/-- Represents the fuel efficiency of an SUV -/
structure SUVFuelEfficiency where
  city_mpg : ℝ
  highway_mpg : ℝ
  max_distance : ℝ
  tank_capacity : ℝ

/-- Theorem stating the highway fuel efficiency of the SUV -/
theorem suv_highway_efficiency (suv : SUVFuelEfficiency)
  (h1 : suv.city_mpg = 7.6)
  (h2 : suv.max_distance = 268.4)
  (h3 : suv.tank_capacity = 22) :
  suv.highway_mpg = 12.2 := by
  sorry

#check suv_highway_efficiency

end suv_highway_efficiency_l1374_137492


namespace sum_of_ages_l1374_137449

theorem sum_of_ages (petra_age mother_age : ℕ) : 
  petra_age = 11 → 
  mother_age = 36 → 
  mother_age = 2 * petra_age + 14 → 
  petra_age + mother_age = 47 := by
sorry

end sum_of_ages_l1374_137449


namespace sphere_radius_is_60_37_l1374_137450

/-- A triangular pyramid with perpendicular lateral edges and a sphere touching all lateral faces -/
structure PerpendicularPyramid where
  /-- The side lengths of the triangular base -/
  base_side_1 : ℝ
  base_side_2 : ℝ
  base_side_3 : ℝ
  /-- The radius of the sphere touching all lateral faces -/
  sphere_radius : ℝ
  /-- The lateral edges are pairwise perpendicular -/
  lateral_edges_perpendicular : True
  /-- The center of the sphere lies on the base -/
  sphere_center_on_base : True
  /-- The base side lengths satisfy the given conditions -/
  base_side_1_sq : base_side_1^2 = 61
  base_side_2_sq : base_side_2^2 = 52
  base_side_3_sq : base_side_3^2 = 41

/-- The theorem stating that the radius of the sphere is 60/37 -/
theorem sphere_radius_is_60_37 (p : PerpendicularPyramid) : p.sphere_radius = 60 / 37 := by
  sorry

end sphere_radius_is_60_37_l1374_137450


namespace six_cube_forming_configurations_l1374_137422

/-- Represents a position where an additional square can be attached -/
inductive AttachmentPosition
| TopLeft | TopCenter | TopRight
| MiddleLeft | MiddleRight
| BottomLeft | BottomCenter | BottomRight
| LeftCenter | RightCenter

/-- Represents the cross-shaped arrangement of squares -/
structure CrossArrangement :=
  (center : Square)
  (top : Square)
  (right : Square)
  (bottom : Square)
  (left : Square)

/-- Represents a configuration with an additional square attached -/
structure Configuration :=
  (base : CrossArrangement)
  (attachment : AttachmentPosition)

/-- Predicate to check if a configuration can form a cube with one face missing -/
def can_form_cube (config : Configuration) : Prop :=
  sorry

/-- The main theorem stating that exactly 6 configurations can form a cube -/
theorem six_cube_forming_configurations :
  ∃ (valid_configs : Finset Configuration),
    (∀ c ∈ valid_configs, can_form_cube c) ∧
    (∀ c : Configuration, can_form_cube c → c ∈ valid_configs) ∧
    valid_configs.card = 6 :=
  sorry

end six_cube_forming_configurations_l1374_137422


namespace parallel_vectors_sum_l1374_137472

/-- Given two parallel vectors a and b in R², prove that their linear combination results in (14, 7) -/
theorem parallel_vectors_sum (m : ℝ) : 
  let a : Fin 2 → ℝ := ![2, 1]
  let b : Fin 2 → ℝ := ![m, 2]
  (∃ (k : ℝ), k ≠ 0 ∧ a = k • b) →
  (3 • a + 2 • b : Fin 2 → ℝ) = ![14, 7] := by
sorry

end parallel_vectors_sum_l1374_137472


namespace smallest_n_divisible_sixty_satisfies_smallest_n_is_sixty_l1374_137412

theorem smallest_n_divisible (n : ℕ) : n > 0 ∧ 24 ∣ n^2 ∧ 720 ∣ n^3 → n ≥ 60 :=
by sorry

theorem sixty_satisfies : 24 ∣ 60^2 ∧ 720 ∣ 60^3 :=
by sorry

theorem smallest_n_is_sixty : ∃! n : ℕ, n > 0 ∧ 24 ∣ n^2 ∧ 720 ∣ n^3 ∧ ∀ m : ℕ, (m > 0 ∧ 24 ∣ m^2 ∧ 720 ∣ m^3) → n ≤ m :=
by sorry

end smallest_n_divisible_sixty_satisfies_smallest_n_is_sixty_l1374_137412


namespace unique_solution_l1374_137432

-- Define the color type
inductive Color
| Red
| Blue

-- Define the clothing type
structure Clothing where
  tshirt : Color
  shorts : Color

-- Define the children
structure Children where
  alyna : Clothing
  bohdan : Clothing
  vika : Clothing
  grysha : Clothing

-- Define the conditions
def satisfiesConditions (c : Children) : Prop :=
  c.alyna.tshirt = Color.Red ∧
  c.bohdan.tshirt = Color.Red ∧
  c.alyna.shorts ≠ c.bohdan.shorts ∧
  c.vika.tshirt ≠ c.grysha.tshirt ∧
  c.vika.shorts = Color.Blue ∧
  c.grysha.shorts = Color.Blue ∧
  c.alyna.tshirt ≠ c.vika.tshirt ∧
  c.alyna.shorts ≠ c.vika.shorts

-- Define the correct answer
def correctAnswer : Children :=
  { alyna := { tshirt := Color.Red, shorts := Color.Red }
  , bohdan := { tshirt := Color.Red, shorts := Color.Blue }
  , vika := { tshirt := Color.Blue, shorts := Color.Blue }
  , grysha := { tshirt := Color.Red, shorts := Color.Blue }
  }

-- Theorem statement
theorem unique_solution :
  ∀ c : Children, satisfiesConditions c → c = correctAnswer := by
  sorry

end unique_solution_l1374_137432
