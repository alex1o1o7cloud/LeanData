import Mathlib

namespace circle_symmetry_tangent_length_l235_23576

/-- Circle C with equation x^2 + y^2 + 2x - 4y + 3 = 0 -/
def Circle (x y : ℝ) : Prop := x^2 + y^2 + 2*x - 4*y + 3 = 0

/-- Line of symmetry with equation 2ax + by + 6 = 0 -/
def SymmetryLine (a b x y : ℝ) : Prop := 2*a*x + b*y + 6 = 0

/-- Point (a, b) lies on the symmetry line -/
def PointOnSymmetryLine (a b : ℝ) : Prop := 2*a*a + b*b + 6 = 0

/-- Minimum length of tangent line segment from (a, b) to the circle -/
def MinTangentLength (a b : ℝ) : ℝ := 4

theorem circle_symmetry_tangent_length 
  (a b : ℝ) 
  (h1 : PointOnSymmetryLine a b) :
  MinTangentLength a b = 4 := by sorry

end circle_symmetry_tangent_length_l235_23576


namespace nails_per_plank_l235_23589

/-- Given that John uses 11 nails in total, with 8 additional nails, and needs 1 plank,
    prove that each plank requires 3 nails to be secured. -/
theorem nails_per_plank (total_nails : ℕ) (additional_nails : ℕ) (num_planks : ℕ)
  (h1 : total_nails = 11)
  (h2 : additional_nails = 8)
  (h3 : num_planks = 1) :
  total_nails - additional_nails = 3 := by
sorry

end nails_per_plank_l235_23589


namespace x_gt_one_sufficient_not_necessary_for_x_squared_gt_x_l235_23528

theorem x_gt_one_sufficient_not_necessary_for_x_squared_gt_x :
  (∃ x : ℝ, x > 1 ∧ x^2 > x) ∧
  (∃ x : ℝ, x^2 > x ∧ ¬(x > 1)) ∧
  (∀ x : ℝ, x > 1 → x^2 > x) :=
by sorry

end x_gt_one_sufficient_not_necessary_for_x_squared_gt_x_l235_23528


namespace simplify_expression_l235_23575

theorem simplify_expression (a b c : ℝ) : 
  3*a - (4*a - 6*b - 3*c) - 5*(c - b) = -a + 11*b - 2*c := by
  sorry

end simplify_expression_l235_23575


namespace parabola_focus_construction_l235_23565

/-- Represents a line in 2D space -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a parabola -/
structure Parabola where
  focus : Point
  directrix : Line

def reflect_line (l : Line) (t : Line) : Line :=
  sorry

def intersection_point (l1 : Line) (l2 : Line) : Point :=
  sorry

def is_tangent (p : Parabola) (t : Line) : Prop :=
  sorry

theorem parabola_focus_construction 
  (p : Parabola) (t1 t2 : Line) 
  (h1 : is_tangent p t1) 
  (h2 : is_tangent p t2) :
  p.focus = intersection_point 
    (reflect_line p.directrix t1) 
    (reflect_line p.directrix t2) :=
sorry

end parabola_focus_construction_l235_23565


namespace sheet_area_difference_l235_23513

/-- The difference in combined area of front and back between two rectangular sheets of paper -/
theorem sheet_area_difference (l1 w1 l2 w2 : ℕ) : 
  l1 = 14 ∧ w1 = 12 ∧ l2 = 9 ∧ w2 = 14 → 2 * (l1 * w1) - 2 * (l2 * w2) = 84 := by
  sorry

#check sheet_area_difference

end sheet_area_difference_l235_23513


namespace odd_digits_base7_528_l235_23560

/-- Converts a natural number from base 10 to base 7 -/
def toBase7 (n : ℕ) : List ℕ :=
  sorry

/-- Counts the number of odd digits in a list of natural numbers -/
def countOddDigits (digits : List ℕ) : ℕ :=
  sorry

/-- The number of odd digits in the base-7 representation of 528 (base 10) is 4 -/
theorem odd_digits_base7_528 : countOddDigits (toBase7 528) = 4 := by
  sorry

end odd_digits_base7_528_l235_23560


namespace converse_of_proposition_l235_23508

theorem converse_of_proposition :
  (∀ x : ℝ, x^2 < 1 → -1 < x ∧ x < 1) →
  (∀ x : ℝ, -1 < x ∧ x < 1 → x^2 < 1) :=
by sorry

end converse_of_proposition_l235_23508


namespace tangent_line_to_exp_and_ln_l235_23504

theorem tangent_line_to_exp_and_ln (a b : ℝ) : 
  (∃ x₁ : ℝ, (x₁ + b = Real.exp x₁) ∧ (1 = Real.exp x₁)) →
  (∃ x₂ : ℝ, (x₂ + b = Real.log (x₂ + a)) ∧ (1 = 1 / (x₂ + a))) →
  a = 2 ∧ b = 1 := by
sorry

end tangent_line_to_exp_and_ln_l235_23504


namespace division_base4_correct_l235_23543

/-- Converts a number from base 4 to base 10 --/
def base4ToBase10 (digits : List Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * (4 ^ i)) 0

/-- Converts a number from base 10 to base 4 --/
def base10ToBase4 (n : Nat) : List Nat :=
  if n = 0 then [0] else
    let rec aux (m : Nat) (acc : List Nat) :=
      if m = 0 then acc
      else aux (m / 4) ((m % 4) :: acc)
    aux n []

/-- Performs division in base 4 --/
def divBase4 (a b : List Nat) : (List Nat × List Nat) :=
  let a10 := base4ToBase10 a
  let b10 := base4ToBase10 b
  let q := a10 / b10
  let r := a10 % b10
  (base10ToBase4 q, base10ToBase4 r)

theorem division_base4_correct (a b : List Nat) :
  a = [2, 3, 0, 2] ∧ b = [2, 1] →
  divBase4 a b = ([3, 1, 1], [0, 1]) := by
  sorry

end division_base4_correct_l235_23543


namespace roadwork_pitch_calculation_l235_23595

/-- Calculates the number of barrels of pitch needed to pave the remaining road -/
def barrels_of_pitch_needed (total_road_length : ℕ) (truckloads_per_mile : ℕ) (gravel_bags_per_truckload : ℕ) (gravel_to_pitch_ratio : ℕ) (paved_miles : ℕ) : ℕ :=
  let remaining_miles := total_road_length - paved_miles
  let total_truckloads := remaining_miles * truckloads_per_mile
  let total_gravel_bags := total_truckloads * gravel_bags_per_truckload
  total_gravel_bags / gravel_to_pitch_ratio

theorem roadwork_pitch_calculation :
  barrels_of_pitch_needed 16 3 2 5 11 = 6 := by
  sorry

end roadwork_pitch_calculation_l235_23595


namespace candy_distribution_count_l235_23530

/-- The number of ways to partition n identical objects into at most k parts -/
def partition_count (n k : ℕ) : ℕ := sorry

/-- Theorem: There are 30 ways to partition 10 identical objects into at most 5 parts -/
theorem candy_distribution_count : partition_count 10 5 = 30 := by sorry

end candy_distribution_count_l235_23530


namespace cupcakes_problem_l235_23539

/-- Given the initial number of cupcakes, the number of cupcakes eaten, and the number of packages,
    calculate the number of cupcakes in each package. -/
def cupcakes_per_package (initial : ℕ) (eaten : ℕ) (packages : ℕ) : ℕ :=
  (initial - eaten) / packages

/-- Theorem: Given 38 initial cupcakes, 14 cupcakes eaten, and 3 packages made,
    the number of cupcakes in each package is 8. -/
theorem cupcakes_problem : cupcakes_per_package 38 14 3 = 8 := by
  sorry

end cupcakes_problem_l235_23539


namespace power_sum_difference_l235_23564

theorem power_sum_difference : 2^6 + 2^6 + 2^6 + 2^6 - 4^4 = 0 := by
  sorry

end power_sum_difference_l235_23564


namespace smallest_t_value_l235_23578

theorem smallest_t_value (u v w t : ℤ) : 
  (u^3 + v^3 + w^3 = t^3) →
  (u^3 < v^3) →
  (v^3 < w^3) →
  (w^3 < t^3) →
  (u^3 < 0) →
  (v^3 < 0) →
  (w^3 < 0) →
  (t^3 < 0) →
  (∃ k : ℤ, u = k - 1 ∧ v = k ∧ w = k + 1 ∧ t = k + 2) →
  (∀ s : ℤ, s < 0 ∧ (∃ x y z : ℤ, x^3 + y^3 + z^3 = s^3 ∧ 
    x^3 < y^3 ∧ y^3 < z^3 ∧ z^3 < s^3 ∧ 
    x^3 < 0 ∧ y^3 < 0 ∧ z^3 < 0 ∧ s^3 < 0 ∧
    (∃ j : ℤ, x = j - 1 ∧ y = j ∧ z = j + 1 ∧ s = j + 2)) → 
    8 ≤ |s|) →
  8 = |t| :=
sorry

end smallest_t_value_l235_23578


namespace inscribed_hexagon_area_l235_23540

/-- A rectangle with width and height -/
structure Rectangle where
  width : ℝ
  height : ℝ

/-- A hexagon inscribed in a rectangle, with vertices touching midpoints of the rectangle's edges -/
structure InscribedHexagon (r : Rectangle) where

/-- The area of a rectangle -/
def Rectangle.area (r : Rectangle) : ℝ := r.width * r.height

/-- The area of an inscribed hexagon -/
def InscribedHexagon.area (h : InscribedHexagon r) : ℝ := sorry

theorem inscribed_hexagon_area (r : Rectangle) (h : InscribedHexagon r) 
    (h_width : r.width = 5) (h_height : r.height = 4) : 
    InscribedHexagon.area h = 10 := by sorry

end inscribed_hexagon_area_l235_23540


namespace point_not_in_third_quadrant_or_origin_l235_23503

theorem point_not_in_third_quadrant_or_origin (n : ℝ) : 
  ¬(n ≤ 0 ∧ 1 - n ≤ 0) ∧ ¬(n = 0 ∧ 1 - n = 0) := by
  sorry

end point_not_in_third_quadrant_or_origin_l235_23503


namespace knowledge_competition_probability_l235_23552

/-- The probability of correctly answering a single question -/
def p_correct : ℝ := 0.8

/-- The number of preset questions in the competition -/
def total_questions : ℕ := 5

/-- The probability of answering exactly 4 questions before advancing -/
def prob_four_questions : ℝ := p_correct * p_correct * (1 - p_correct) * p_correct

theorem knowledge_competition_probability :
  prob_four_questions = 0.128 :=
sorry

end knowledge_competition_probability_l235_23552


namespace dog_age_difference_l235_23593

theorem dog_age_difference (
  avg_age_1_5 : ℝ)
  (age_1 : ℝ)
  (age_2 : ℝ)
  (age_3 : ℝ)
  (age_4 : ℝ)
  (age_5 : ℝ)
  (h1 : avg_age_1_5 = 18)
  (h2 : age_1 = 10)
  (h3 : age_2 = age_1 - 2)
  (h4 : age_3 = age_2 + 4)
  (h5 : age_4 = age_3 / 2)
  (h6 : age_5 = age_4 + 20)
  (h7 : avg_age_1_5 = (age_1 + age_5) / 2) :
  age_3 - age_2 = 4 := by
sorry

end dog_age_difference_l235_23593


namespace symmetric_point_xoy_plane_l235_23511

/-- Given a point (1, 2, 3) in a three-dimensional Cartesian coordinate system,
    its symmetric point with respect to the xoy plane is (1, 2, -3). -/
theorem symmetric_point_xoy_plane :
  let original_point : ℝ × ℝ × ℝ := (1, 2, 3)
  let xoy_plane : Set (ℝ × ℝ × ℝ) := {p | p.2.2 = 0}
  let symmetric_point (p : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ := (p.1, p.2.1, -p.2.2)
  symmetric_point original_point = (1, 2, -3) :=
by sorry

end symmetric_point_xoy_plane_l235_23511


namespace python_eating_theorem_l235_23505

/-- The number of days in the given time period -/
def total_days : ℕ := 616

/-- The number of days in a week -/
def days_per_week : ℕ := 7

/-- The rate at which the python eats alligators (alligators per week) -/
def eating_rate : ℕ := 1

/-- The maximum number of alligators the python can eat in the given time period -/
def max_alligators_eaten : ℕ := total_days / days_per_week

theorem python_eating_theorem :
  max_alligators_eaten = eating_rate * (total_days / days_per_week) :=
by sorry

end python_eating_theorem_l235_23505


namespace updated_mean_example_l235_23529

/-- The updated mean of a dataset after corrections -/
def updated_mean (original_mean original_count : ℕ) 
                 (decrement : ℕ) 
                 (missing_obs : List ℕ) 
                 (extra_obs : ℕ) : ℚ :=
  let original_sum := original_mean * original_count
  let corrected_sum := original_sum - decrement * original_count + missing_obs.sum - extra_obs
  let corrected_count := original_count - 1 + missing_obs.length
  (corrected_sum : ℚ) / corrected_count

/-- Theorem stating the updated mean after corrections -/
theorem updated_mean_example : 
  updated_mean 200 50 34 [150, 190, 210] 250 = 8600 / 52 := by
  sorry

end updated_mean_example_l235_23529


namespace converse_propositions_l235_23501

-- Define the basic concepts
def Point : Type := ℝ × ℝ × ℝ
def Line : Type := Point → Prop

-- Define the relationships
def coplanar (a b c d : Point) : Prop := sorry
def collinear (a b c : Point) : Prop := sorry
def have_common_point (l₁ l₂ : Line) : Prop := sorry
def skew_lines (l₁ l₂ : Line) : Prop := sorry

-- State the theorem
theorem converse_propositions :
  (∀ a b c d : Point, (¬collinear a b c ∧ ¬collinear a b d ∧ ¬collinear a c d ∧ ¬collinear b c d) → ¬coplanar a b c d) = false ∧
  (∀ l₁ l₂ : Line, skew_lines l₁ l₂ → ¬have_common_point l₁ l₂) = true :=
sorry

end converse_propositions_l235_23501


namespace prob_same_color_correct_l235_23555

def total_balls : ℕ := 16
def green_balls : ℕ := 8
def red_balls : ℕ := 5
def blue_balls : ℕ := 3

def prob_same_color : ℚ := 98 / 256

theorem prob_same_color_correct :
  (green_balls / total_balls)^2 + (red_balls / total_balls)^2 + (blue_balls / total_balls)^2 = prob_same_color :=
by sorry

end prob_same_color_correct_l235_23555


namespace arithmetic_progression_first_term_l235_23525

theorem arithmetic_progression_first_term
  (a : ℕ → ℝ)
  (h_increasing : ∀ n, a n < a (n + 1))
  (h_arithmetic : ∃ d, ∀ n, a (n + 1) - a n = d)
  (h_sum : a 0 + a 1 + a 2 = 12)
  (h_product : a 0 * a 1 * a 2 = 48) :
  a 0 = 2 :=
sorry

end arithmetic_progression_first_term_l235_23525


namespace student_tape_cost_problem_l235_23542

theorem student_tape_cost_problem :
  ∃ (n : ℕ) (x : ℕ) (price : ℕ),
    Even n ∧
    10 < n ∧ n < 20 ∧
    100 ≤ price ∧ price ≤ 120 ∧
    n * x = price ∧
    (n - 2) * (x + 1) = price ∧
    n = 14 := by
  sorry

end student_tape_cost_problem_l235_23542


namespace expected_balls_in_position_l235_23583

/-- The number of balls arranged in a circle -/
def num_balls : ℕ := 5

/-- The probability that a specific ball is chosen for a swap -/
def prob_chosen : ℚ := 2 / 5

/-- The probability that a specific pair is chosen again -/
def prob_same_pair : ℚ := 1 / 5

/-- The probability that a specific ball is not involved in a swap -/
def prob_not_involved : ℚ := 3 / 5

/-- The number of independent transpositions -/
def num_transpositions : ℕ := 2

/-- 
Theorem: Given 5 balls arranged in a circle, with two independent random transpositions 
of adjacent balls, the expected number of balls in their original positions is 2.2.
-/
theorem expected_balls_in_position : 
  let prob_in_position := prob_chosen * prob_same_pair + prob_not_involved ^ num_transpositions
  num_balls * prob_in_position = 11/5 := by
  sorry

end expected_balls_in_position_l235_23583


namespace average_age_increase_l235_23598

theorem average_age_increase (initial_count : Nat) (replaced_age1 replaced_age2 women_avg_age : ℕ) :
  initial_count = 9 →
  replaced_age1 = 36 →
  replaced_age2 = 32 →
  women_avg_age = 52 →
  (2 * women_avg_age - (replaced_age1 + replaced_age2)) / initial_count = 4 :=
by
  sorry

end average_age_increase_l235_23598


namespace rectangular_field_area_l235_23533

/-- A rectangular field with width half its length and perimeter 54 meters has an area of 162 square meters. -/
theorem rectangular_field_area : ∀ w l : ℝ,
  w > 0 →
  l > 0 →
  w = l / 2 →
  2 * (w + l) = 54 →
  w * l = 162 := by
  sorry

end rectangular_field_area_l235_23533


namespace points_symmetric_about_x_axis_l235_23537

/-- Two points are symmetric about the x-axis if they have the same x-coordinate
    and their y-coordinates are negatives of each other. -/
def symmetric_about_x_axis (p1 p2 : ℝ × ℝ) : Prop :=
  p1.1 = p2.1 ∧ p1.2 = -p2.2

/-- Given points P₁(-4, 3) and P₂(-4, -3), prove they are symmetric about the x-axis. -/
theorem points_symmetric_about_x_axis :
  let p1 : ℝ × ℝ := (-4, 3)
  let p2 : ℝ × ℝ := (-4, -3)
  symmetric_about_x_axis p1 p2 := by
  sorry


end points_symmetric_about_x_axis_l235_23537


namespace range_of_b_l235_23590

-- Define the function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x ≥ a then x^2 - 2*a*x + 1 else -(x^2 - 2*a*x + 1)

-- State the theorem
theorem range_of_b (a : ℝ) (b : ℝ) :
  (a > 0) →
  (∀ x : ℝ, f a (x^3 + a) = -f a (-(x^3 + a))) →
  (∀ x : ℝ, x ∈ Set.Icc (b - 1) (b + 2) → f a (b * x) ≥ 4 * f a (x + 1)) →
  b ∈ Set.Iic (-Real.sqrt 5) ∪ Set.Ici ((3 + Real.sqrt 5) / 2) :=
by sorry

end range_of_b_l235_23590


namespace min_odd_integers_l235_23592

theorem min_odd_integers (a b c d e f : ℤ) : 
  a + b = 28 → 
  a + b + c + d = 45 → 
  a + b + c + d + e + f = 60 → 
  ∃ (odds : Finset ℤ), odds ⊆ {a, b, c, d, e, f} ∧ 
    (∀ x ∈ odds, Odd x) ∧ 
    odds.card = 2 ∧
    (∀ (other_odds : Finset ℤ), other_odds ⊆ {a, b, c, d, e, f} ∧ 
      (∀ x ∈ other_odds, Odd x) → 
      other_odds.card ≥ 2) :=
by sorry

end min_odd_integers_l235_23592


namespace jennys_change_l235_23580

/-- The problem of calculating Jenny's change --/
theorem jennys_change 
  (cost_per_page : ℚ)
  (num_copies : ℕ)
  (pages_per_essay : ℕ)
  (num_pens : ℕ)
  (cost_per_pen : ℚ)
  (payment : ℚ)
  (h1 : cost_per_page = 1/10)
  (h2 : num_copies = 7)
  (h3 : pages_per_essay = 25)
  (h4 : num_pens = 7)
  (h5 : cost_per_pen = 3/2)
  (h6 : payment = 40) :
  payment - (cost_per_page * num_copies * pages_per_essay + cost_per_pen * num_pens) = 12 := by
  sorry


end jennys_change_l235_23580


namespace eight_custom_op_eight_eq_four_l235_23558

/-- Custom operation @ for positive integers -/
def custom_op (a b : ℕ+) : ℚ :=
  (a * b) / (a + b)

/-- Theorem stating that 8 @ 8 = 4 -/
theorem eight_custom_op_eight_eq_four :
  custom_op 8 8 = 4 := by sorry

end eight_custom_op_eight_eq_four_l235_23558


namespace coefficient_x_squared_l235_23546

/-- The coefficient of x^2 in the expansion of (2x^2 + 3x + 4)(5x^2 + 6x + 7) is 52 -/
theorem coefficient_x_squared (x : ℝ) : 
  (2*x^2 + 3*x + 4) * (5*x^2 + 6*x + 7) = 10*x^4 + 27*x^3 + 52*x^2 + 45*x + 28 := by
  sorry

#check coefficient_x_squared

end coefficient_x_squared_l235_23546


namespace pictures_deleted_l235_23544

theorem pictures_deleted (zoo_pics : ℕ) (museum_pics : ℕ) (pics_left : ℕ) : 
  zoo_pics = 50 → museum_pics = 8 → pics_left = 20 → 
  zoo_pics + museum_pics - pics_left = 38 := by
sorry

end pictures_deleted_l235_23544


namespace sequence_properties_l235_23584

theorem sequence_properties :
  (∃ a : ℕ → ℕ, a 1 = 2 ∧ (∀ n, a (n + 1) = a n + n + 1) ∧ a 20 = 211) ∧
  (∃ b : ℕ → ℕ, b 1 = 1 ∧ (∀ n, b (n + 1) = 3 * b n + 2) ∧ b 4 = 53) :=
by sorry

end sequence_properties_l235_23584


namespace typists_letters_time_relation_typists_letters_theorem_l235_23566

/-- The number of letters a single typist can type in one minute -/
def typing_rate (typists : ℕ) (letters : ℕ) (minutes : ℕ) : ℚ :=
  (letters : ℚ) / (typists * minutes)

/-- The theorem stating the relationship between typists, letters, and time -/
theorem typists_letters_time_relation 
  (initial_typists : ℕ) (initial_letters : ℕ) (initial_minutes : ℕ)
  (final_typists : ℕ) (final_minutes : ℕ) :
  initial_typists > 0 → initial_minutes > 0 → final_typists > 0 → final_minutes > 0 →
  (typing_rate initial_typists initial_letters initial_minutes) * 
    (final_typists * final_minutes) = 
  (final_typists * final_minutes * initial_letters : ℚ) / (initial_typists * initial_minutes) :=
by sorry

/-- The main theorem to prove -/
theorem typists_letters_theorem :
  typing_rate 20 42 20 * (30 * 60) = 189 :=
by sorry

end typists_letters_time_relation_typists_letters_theorem_l235_23566


namespace sum_product_bounds_l235_23579

theorem sum_product_bounds (a b c : ℝ) (h : a + b + c = 3) :
  ∃ (lower_bound upper_bound : ℝ),
    lower_bound = -9/2 ∧
    upper_bound = 3 ∧
    (∀ ε > 0, ∃ (x y z : ℝ), x + y + z = 3 ∧ x*y + x*z + y*z < lower_bound + ε) ∧
    (∀ (x y z : ℝ), x + y + z = 3 → x*y + x*z + y*z ≤ upper_bound) :=
by sorry

end sum_product_bounds_l235_23579


namespace power_of_power_l235_23515

theorem power_of_power (a : ℝ) : (a^2)^3 = a^6 := by
  sorry

end power_of_power_l235_23515


namespace A_intersect_B_l235_23518

def A : Set ℕ := {1, 2, 3}

def B : Set ℕ := {y | ∃ x ∈ A, y = 2 * x - 1}

theorem A_intersect_B : A ∩ B = {1, 3} := by sorry

end A_intersect_B_l235_23518


namespace exponential_equation_solution_l235_23586

theorem exponential_equation_solution :
  ∃ x : ℝ, 3^(3*x + 2) = (1:ℝ)/81 ∧ x = -2 := by
  sorry

end exponential_equation_solution_l235_23586


namespace A_enumeration_l235_23563

def A : Set ℤ := {y | ∃ x : ℕ, y = 6 / (x - 2) ∧ 6 % (x - 2) = 0}

theorem A_enumeration : A = {-3, -6, 6, 3, 2, 1} := by
  sorry

end A_enumeration_l235_23563


namespace max_value_theorem_l235_23524

theorem max_value_theorem (x y z : ℝ) (h1 : 0 ≤ x) (h2 : 0 ≤ y) (h3 : 0 ≤ z) 
  (h4 : x^2 + y^2 + z^2 = 1) : 
  2 * x * y * Real.sqrt 6 + 8 * y * z^2 ≤ Real.sqrt 6 :=
by sorry

end max_value_theorem_l235_23524


namespace area_of_inscribed_circle_rectangle_l235_23527

/-- A rectangle with an inscribed circle -/
structure InscribedCircleRectangle where
  /-- The radius of the inscribed circle -/
  r : ℝ
  /-- The width of the rectangle -/
  w : ℝ
  /-- The height of the rectangle -/
  h : ℝ
  /-- The circle is tangent to all sides -/
  tangent_to_sides : w = h
  /-- The circle passes through the midpoint of a diagonal -/
  passes_through_midpoint : w^2 / 4 + h^2 / 4 = r^2

/-- The area of a rectangle with an inscribed circle passing through the midpoint of a diagonal is 2r^2 -/
theorem area_of_inscribed_circle_rectangle (rect : InscribedCircleRectangle) : 
  rect.w * rect.h = 2 * rect.r^2 := by
  sorry

end area_of_inscribed_circle_rectangle_l235_23527


namespace tim_reading_time_l235_23506

/-- Given that Tim spends 1 hour a day meditating and twice as much time reading,
    prove that he spends 14 hours a week reading. -/
theorem tim_reading_time (meditation_time : ℝ) (reading_time : ℝ) (days_in_week : ℕ) :
  meditation_time = 1 →
  reading_time = 2 * meditation_time →
  days_in_week = 7 →
  reading_time * days_in_week = 14 := by
  sorry

end tim_reading_time_l235_23506


namespace antenna_tower_height_l235_23582

/-- Given an antenna tower on flat terrain, if the sum of the angles of elevation
    measured at distances of 100 m, 200 m, and 300 m from its base is 90°,
    then the height of the tower is 100 m. -/
theorem antenna_tower_height (α β γ : Real) (h : Real) :
  (α + β + γ = Real.pi / 2) →
  (h / 100 = Real.tan α) →
  (h / 200 = Real.tan β) →
  (h / 300 = Real.tan γ) →
  h = 100 := by
  sorry

#check antenna_tower_height

end antenna_tower_height_l235_23582


namespace tom_initial_investment_l235_23568

/-- Represents the initial investment of Tom in rupees -/
def tom_investment : ℝ := 3000

/-- Represents Jose's investment in rupees -/
def jose_investment : ℝ := 4500

/-- Represents the total duration of the business in months -/
def total_duration : ℝ := 12

/-- Represents the time after which Jose joined in months -/
def jose_join_time : ℝ := 2

/-- Represents the total profit in rupees -/
def total_profit : ℝ := 6300

/-- Represents Jose's share of the profit in rupees -/
def jose_profit : ℝ := 3500

theorem tom_initial_investment :
  tom_investment * total_duration / (jose_investment * (total_duration - jose_join_time)) =
  (total_profit - jose_profit) / jose_profit :=
sorry

end tom_initial_investment_l235_23568


namespace expansion_coefficient_l235_23512

-- Define the binomial coefficient
def binomial (n k : ℕ) : ℕ := sorry

-- Define the coefficient of x^3 in the expansion of (ax^2 - 1/x)^6
def coefficient (a : ℝ) : ℝ := -a^3 * binomial 6 3

-- Theorem statement
theorem expansion_coefficient (a : ℝ) : coefficient a = 160 → a = -2 := by
  sorry

end expansion_coefficient_l235_23512


namespace lcm_five_equals_lcm_three_l235_23535

def is_subset_prime_factorization (a b : Nat) : Prop :=
  ∀ p : Nat, Prime p → (p^(a.factorization p) ∣ b)

theorem lcm_five_equals_lcm_three
  (a b c d e : Nat)
  (h : a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0 ∧ e > 0)
  (h_lcm : Nat.lcm a (Nat.lcm b (Nat.lcm c (Nat.lcm d e))) = Nat.lcm a (Nat.lcm b c)) :
  (is_subset_prime_factorization d a ∨ is_subset_prime_factorization d b ∨ is_subset_prime_factorization d c) ∧
  (is_subset_prime_factorization e a ∨ is_subset_prime_factorization e b ∨ is_subset_prime_factorization e c) :=
sorry

end lcm_five_equals_lcm_three_l235_23535


namespace base_10_to_base_5_512_l235_23545

/-- Converts a base-10 number to its base-5 representation -/
def toBase5 (n : ℕ) : List ℕ :=
  sorry

theorem base_10_to_base_5_512 :
  toBase5 512 = [4, 0, 2, 2] :=
sorry

end base_10_to_base_5_512_l235_23545


namespace range_of_a_l235_23572

theorem range_of_a (a : ℝ) : 
  (∀ (x : ℝ), x > 3 → x > a) ↔ a ≤ 3 :=
sorry

end range_of_a_l235_23572


namespace inequality_condition_l235_23561

theorem inequality_condition (x : ℝ) : 
  (|x - 1| < 1 → x^2 - 5*x < 0) ∧ 
  ¬(∀ x : ℝ, x^2 - 5*x < 0 → |x - 1| < 1) :=
sorry

end inequality_condition_l235_23561


namespace min_value_fraction_l235_23510

theorem min_value_fraction (x : ℝ) (h : x > 5) : 
  x^2 / (x - 5) ≥ 20 ∧ ∃ y > 5, y^2 / (y - 5) = 20 := by
  sorry

end min_value_fraction_l235_23510


namespace gasoline_tank_capacity_l235_23570

theorem gasoline_tank_capacity : ∃ (capacity : ℝ), 
  capacity > 0 ∧
  (3/4 * capacity - 18 = 1/3 * capacity) ∧
  capacity = 43.2 := by
  sorry

end gasoline_tank_capacity_l235_23570


namespace swim_club_members_swim_club_members_proof_l235_23520

theorem swim_club_members : ℕ → Prop :=
  fun total_members =>
    let passed_test := (30 : ℚ) / 100 * total_members
    let not_passed := total_members - passed_test
    let prep_course := 12
    let no_prep_course := 30
    passed_test + not_passed = total_members ∧
    prep_course + no_prep_course = not_passed ∧
    total_members = 60

-- Proof
theorem swim_club_members_proof : ∃ n : ℕ, swim_club_members n :=
  sorry

end swim_club_members_swim_club_members_proof_l235_23520


namespace fish_caught_fisherman_catch_l235_23532

theorem fish_caught (fish_per_line : ℕ) (initial_lines : ℕ) (broken_lines : ℕ) : ℕ :=
  let usable_lines : ℕ := initial_lines - broken_lines
  usable_lines * fish_per_line

theorem fisherman_catch : fish_caught 3 226 3 = 669 := by
  sorry

end fish_caught_fisherman_catch_l235_23532


namespace polynomial_inequality_l235_23591

theorem polynomial_inequality (x : ℝ) : x^4 + x^3 - 10*x^2 > -25*x ↔ x > 0 := by
  sorry

end polynomial_inequality_l235_23591


namespace motion_equation_l235_23577

theorem motion_equation (g a V V₀ S t : ℝ) 
  (hV : V = (g + a) * t + V₀)
  (hS : S = (1/2) * (g + a) * t^2 + V₀ * t) :
  t = 2 * S / (V + V₀) := by
  sorry

end motion_equation_l235_23577


namespace ab_value_l235_23536

theorem ab_value (a b : ℝ) (h : |a + 1| + (b - 3)^2 = 0) : a^b = -1 := by
  sorry

end ab_value_l235_23536


namespace sales_percentage_other_l235_23557

theorem sales_percentage_other (total_percentage : ℝ) (markers_percentage : ℝ) (notebooks_percentage : ℝ)
  (h1 : total_percentage = 100)
  (h2 : markers_percentage = 42)
  (h3 : notebooks_percentage = 22) :
  total_percentage - markers_percentage - notebooks_percentage = 36 := by
sorry

end sales_percentage_other_l235_23557


namespace smallest_four_digit_divisible_by_first_five_primes_l235_23599

theorem smallest_four_digit_divisible_by_first_five_primes :
  ∃ n : ℕ,
    n ≥ 1000 ∧
    n < 10000 ∧
    2 ∣ n ∧
    3 ∣ n ∧
    5 ∣ n ∧
    7 ∣ n ∧
    11 ∣ n ∧
    (∀ m : ℕ, m ≥ 1000 ∧ m < 10000 ∧ 2 ∣ m ∧ 3 ∣ m ∧ 5 ∣ m ∧ 7 ∣ m ∧ 11 ∣ m → n ≤ m) ∧
    n = 2310 :=
by
  sorry

#eval 2310

end smallest_four_digit_divisible_by_first_five_primes_l235_23599


namespace right_triangle_hypotenuse_l235_23521

theorem right_triangle_hypotenuse (a b c : ℝ) : 
  a = 14 → b = 48 → c^2 = a^2 + b^2 → c = 50 :=
by sorry

end right_triangle_hypotenuse_l235_23521


namespace valid_two_digit_numbers_l235_23522

def digits : Set Nat := {1, 2, 3}

def is_two_digit (n : Nat) : Prop :=
  10 ≤ n ∧ n < 100

def has_no_repeated_digits (n : Nat) : Prop :=
  let tens := n / 10
  let ones := n % 10
  tens ≠ ones

def is_valid_number (n : Nat) : Prop :=
  is_two_digit n ∧
  has_no_repeated_digits n ∧
  (n / 10 ∈ digits) ∧
  (n % 10 ∈ digits)

theorem valid_two_digit_numbers :
  {n : Nat | is_valid_number n} = {12, 13, 21, 23, 31, 32} := by
  sorry

end valid_two_digit_numbers_l235_23522


namespace prob_certain_event_prob_union_l235_23531

-- Define a probability space
variable (Ω : Type*) [MeasurableSpace Ω] (P : Measure Ω)

-- Define a certain event
def certain_event : Set Ω := Set.univ

-- Define the probability of an event
def prob (A : Set Ω) : ℝ := P A

-- Theorem 1: The probability of a certain event is 1
theorem prob_certain_event :
  prob P certain_event = 1 := by sorry

-- Theorem 2: Probability of union of two events
theorem prob_union (A B : Set Ω) :
  prob P (A ∪ B) = prob P A + prob P B - prob P (A ∩ B) := by sorry

end prob_certain_event_prob_union_l235_23531


namespace sector_area_l235_23516

theorem sector_area (r : ℝ) (θ : ℝ) (h : r = 2) (h' : θ = π / 4) :
  (1 / 2) * r^2 * θ = π / 2 := by
  sorry

end sector_area_l235_23516


namespace probability_two_non_defective_pens_l235_23514

theorem probability_two_non_defective_pens 
  (total_pens : ℕ) 
  (defective_pens : ℕ) 
  (selected_pens : ℕ) 
  (h1 : total_pens = 12) 
  (h2 : defective_pens = 4) 
  (h3 : selected_pens = 2) :
  (total_pens - defective_pens : ℚ) / total_pens * 
  ((total_pens - defective_pens - 1) : ℚ) / (total_pens - 1) = 14/33 := by
  sorry

end probability_two_non_defective_pens_l235_23514


namespace arithmetic_sequence_a11_l235_23538

theorem arithmetic_sequence_a11 (a : ℕ → ℚ) 
  (h_arith : ∀ n, (a (n+4) + 1)⁻¹ = ((a n + 1)⁻¹ + (a (n+8) + 1)⁻¹) / 2)
  (h_a3 : a 3 = 2)
  (h_a7 : a 7 = 1) :
  a 11 = 1/2 := by sorry

end arithmetic_sequence_a11_l235_23538


namespace solutions_equation1_solutions_equation2_l235_23581

-- Define the equations
def equation1 (x : ℝ) : Prop := x^2 - 4*x + 2 = 0
def equation2 (x : ℝ) : Prop := (x-3)^2 = 2*x - 6

-- Theorem for the first equation
theorem solutions_equation1 : 
  {x : ℝ | equation1 x} = {2 + Real.sqrt 2, 2 - Real.sqrt 2} :=
sorry

-- Theorem for the second equation
theorem solutions_equation2 :
  {x : ℝ | equation2 x} = {3, 5} :=
sorry

end solutions_equation1_solutions_equation2_l235_23581


namespace factor_expression_l235_23597

theorem factor_expression (y : ℝ) : 3 * y * (y - 4) + 5 * (y - 4) + 2 * (y - 4) = (3 * y + 7) * (y - 4) := by
  sorry

end factor_expression_l235_23597


namespace sum_xyz_l235_23567

theorem sum_xyz (x y z : ℝ) 
  (eq1 : y + z = 20 - 4*x)
  (eq2 : x + z = 10 - 5*y)
  (eq3 : x + y = 15 - 2*z) :
  3*x + 3*y + 3*z = 22.5 := by
sorry

end sum_xyz_l235_23567


namespace tan_x_plus_pi_fourth_l235_23587

theorem tan_x_plus_pi_fourth (x : ℝ) (h : Real.tan x = 2) : 
  Real.tan (x + π / 4) = -3 := by sorry

end tan_x_plus_pi_fourth_l235_23587


namespace m_nonpositive_l235_23541

theorem m_nonpositive (m : ℝ) (h : Real.sqrt (m^2) = -m) : m ≤ 0 := by
  sorry

end m_nonpositive_l235_23541


namespace infinite_inscribed_rectangles_l235_23502

/-- A point in 2D space -/
structure Point :=
  (x : ℝ)
  (y : ℝ)

/-- A rectangle defined by its four vertices -/
structure Rectangle :=
  (A : Point)
  (B : Point)
  (C : Point)
  (D : Point)

/-- Predicate to check if a point lies on a side of a rectangle -/
def PointOnSide (P : Point) (R : Rectangle) : Prop :=
  (P.x = R.A.x ∧ R.A.y ≤ P.y ∧ P.y ≤ R.B.y) ∨
  (P.y = R.B.y ∧ R.B.x ≤ P.x ∧ P.x ≤ R.C.x) ∨
  (P.x = R.C.x ∧ R.C.y ≥ P.y ∧ P.y ≥ R.D.y) ∨
  (P.y = R.D.y ∧ R.D.x ≥ P.x ∧ P.x ≥ R.A.x)

/-- Predicate to check if four points form a rectangle -/
def IsRectangle (E F G H : Point) : Prop :=
  (E.x - F.x) * (G.x - H.x) + (E.y - F.y) * (G.y - H.y) = 0 ∧
  (E.x - H.x) * (F.x - G.x) + (E.y - H.y) * (F.y - G.y) = 0

theorem infinite_inscribed_rectangles (ABCD : Rectangle) :
  ∃ (S : Set (Point × Point × Point × Point)),
    (∀ (E F G H : Point), (E, F, G, H) ∈ S →
      PointOnSide E ABCD ∧ PointOnSide F ABCD ∧
      PointOnSide G ABCD ∧ PointOnSide H ABCD ∧
      IsRectangle E F G H) ∧
    Set.Infinite S :=
  sorry

end infinite_inscribed_rectangles_l235_23502


namespace whole_number_between_l235_23554

theorem whole_number_between : 
  ∀ N : ℤ, (9 < (N : ℚ) / 4 ∧ (N : ℚ) / 4 < 10) → (N = 37 ∨ N = 38 ∨ N = 39) :=
by
  sorry

end whole_number_between_l235_23554


namespace tank_plastering_cost_l235_23550

/-- Calculates the total cost of plastering a rectangular tank. -/
def plastering_cost (length width depth : ℝ) (cost_per_sqm : ℝ) : ℝ :=
  let bottom_area := length * width
  let long_walls_area := 2 * (length * depth)
  let short_walls_area := 2 * (width * depth)
  let total_area := bottom_area + long_walls_area + short_walls_area
  total_area * cost_per_sqm

/-- Theorem stating the cost of plastering a specific tank. -/
theorem tank_plastering_cost :
  plastering_cost 60 25 10 0.9 = 2880 := by
  sorry

#eval plastering_cost 60 25 10 0.9

end tank_plastering_cost_l235_23550


namespace quadratic_roots_real_for_pure_imaginary_k_l235_23509

theorem quadratic_roots_real_for_pure_imaginary_k :
  ∀ (k : ℂ), (∃ (r : ℝ), k = r * I) →
  ∃ (z₁ z₂ : ℝ), (5 : ℂ) * (z₁ : ℂ)^2 + 7 * I * (z₁ : ℂ) - k = 0 ∧
                 (5 : ℂ) * (z₂ : ℂ)^2 + 7 * I * (z₂ : ℂ) - k = 0 :=
by sorry


end quadratic_roots_real_for_pure_imaginary_k_l235_23509


namespace cube_edge_length_l235_23547

theorem cube_edge_length (surface_area : ℝ) (h : surface_area = 24) :
  ∃ edge_length : ℝ, edge_length > 0 ∧ 6 * edge_length^2 = surface_area ∧ edge_length = 2 := by
  sorry

end cube_edge_length_l235_23547


namespace hyperbola_axis_ratio_l235_23556

/-- Given a hyperbola with equation mx^2 + y^2 = 1, if its conjugate axis is twice the length
of its transverse axis, then m = -1/4 -/
theorem hyperbola_axis_ratio (m : ℝ) : 
  (∀ x y : ℝ, m * x^2 + y^2 = 1) →  -- Equation of the hyperbola
  (∃ a b : ℝ, a > 0 ∧ b > 0 ∧       -- Existence of positive a and b
    (∀ x y : ℝ, y^2 / b^2 - x^2 / a^2 = 1) ∧  -- Standard form of hyperbola
    2 * b = 2 * a) →                -- Conjugate axis is twice the transverse axis
  m = -1/4 := by
sorry

end hyperbola_axis_ratio_l235_23556


namespace partial_fraction_decomposition_denominator_factorization_l235_23548

theorem partial_fraction_decomposition (x : ℝ) : 
  let A : ℝ := 1/2
  let B : ℝ := 9/2
  (6*x - 7) / (3*x^2 + 2*x - 8) = A / (x - 2) + B / (3*x + 4) :=
by
  sorry

-- Auxiliary theorem to establish the factorization of the denominator
theorem denominator_factorization (x : ℝ) :
  3*x^2 + 2*x - 8 = (3*x + 4)*(x - 2) :=
by
  sorry

end partial_fraction_decomposition_denominator_factorization_l235_23548


namespace quadratic_nature_l235_23588

/-- A quadratic function g(x) with the condition c = a + b^2 -/
def g (a b : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x + (a + b^2)

theorem quadratic_nature (a b : ℝ) :
  (a < 0 → ∃ x₀, ∀ x, g a b x ≤ g a b x₀) ∧
  (a > 0 → ∃ x₀, ∀ x, g a b x ≥ g a b x₀) :=
sorry

end quadratic_nature_l235_23588


namespace marge_personal_spending_l235_23573

/-- Calculates Marge's personal spending amount after one year --/
def personal_spending_after_one_year (
  lottery_winnings : ℝ)
  (tax_rate : ℝ)
  (mortgage_rate : ℝ)
  (retirement_rate : ℝ)
  (retirement_interest : ℝ)
  (college_rate : ℝ)
  (savings : ℝ)
  (stock_investment_rate : ℝ)
  (stock_return : ℝ) : ℝ :=
  let after_tax := lottery_winnings * (1 - tax_rate)
  let after_mortgage := after_tax * (1 - mortgage_rate)
  let after_retirement := after_mortgage * (1 - retirement_rate)
  let after_college := after_retirement * (1 - college_rate)
  let retirement_growth := after_mortgage * retirement_rate * retirement_interest
  let stock_investment := savings * stock_investment_rate
  let stock_growth := stock_investment * stock_return
  after_college + (savings - stock_investment) + retirement_growth + stock_growth

/-- Theorem stating that Marge's personal spending after one year is $5,363 --/
theorem marge_personal_spending :
  personal_spending_after_one_year 50000 0.6 0.5 0.4 0.05 0.25 1500 0.6 0.07 = 5363 := by
  sorry

end marge_personal_spending_l235_23573


namespace lindas_savings_l235_23551

theorem lindas_savings (savings : ℝ) 
  (h1 : savings * (1/4) = 200)
  (h2 : ∃ (furniture_cost : ℝ), furniture_cost = savings * (3/4) ∧ 
        furniture_cost * 0.8 = savings * (3/4))
  : savings = 800 := by
sorry

end lindas_savings_l235_23551


namespace bucket_leak_problem_l235_23596

/-- Converts gallons to quarts -/
def gallons_to_quarts (g : ℝ) : ℝ := 4 * g

/-- Calculates the amount of water leaked given initial and remaining amounts -/
def water_leaked (initial : ℝ) (remaining : ℝ) : ℝ := initial - remaining

theorem bucket_leak_problem (initial : ℝ) (remaining_gallons : ℝ) 
  (h1 : initial = 4) 
  (h2 : remaining_gallons = 0.33) : 
  water_leaked initial (gallons_to_quarts remaining_gallons) = 2.68 := by
  sorry

#eval water_leaked 4 (gallons_to_quarts 0.33)

end bucket_leak_problem_l235_23596


namespace solve_equation_l235_23526

theorem solve_equation (x : ℝ) : 
  (1 : ℝ) / 7 + 7 / x = 15 / x + (1 : ℝ) / 15 → x = 105 := by
  sorry

end solve_equation_l235_23526


namespace division_problem_l235_23569

theorem division_problem (dividend quotient remainder divisor : ℕ) : 
  dividend = 122 →
  quotient = 6 →
  remainder = 2 →
  dividend = divisor * quotient + remainder →
  divisor = 20 := by
sorry

end division_problem_l235_23569


namespace smallest_multiple_of_42_and_56_not_18_l235_23574

theorem smallest_multiple_of_42_and_56_not_18 : ∃ n : ℕ+, 
  (∀ m : ℕ+, m < n → (42 ∣ m.val ∧ 56 ∣ m.val) → 18 ∣ m.val) ∧ 
  42 ∣ n.val ∧ 56 ∣ n.val ∧ ¬(18 ∣ n.val) ∧ n.val = 168 := by
  sorry

end smallest_multiple_of_42_and_56_not_18_l235_23574


namespace arithmetic_sequence_sum_6_l235_23500

/-- An arithmetic sequence with its sum function -/
structure ArithmeticSequence where
  a : ℕ → ℝ
  d : ℝ
  first_term : a 1 = 2
  is_arithmetic : ∀ n : ℕ, a (n + 1) = a n + d
  sum : ℕ → ℝ
  sum_def : ∀ n : ℕ, sum n = n * (2 * a 1 + (n - 1) * d) / 2

/-- The theorem to be proved -/
theorem arithmetic_sequence_sum_6 (seq : ArithmeticSequence) (h : seq.sum 4 = 20) :
  seq.sum 6 = 42 := by
  sorry

end arithmetic_sequence_sum_6_l235_23500


namespace complex_product_real_l235_23553

theorem complex_product_real (a : ℝ) : 
  let z₁ : ℂ := 3 - 2 * I
  let z₂ : ℂ := 1 + a * I
  (z₁ * z₂).im = 0 → a = 2/3 := by sorry

end complex_product_real_l235_23553


namespace common_difference_is_five_l235_23559

def arithmetic_sequence (a : ℕ → ℝ) := ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem common_difference_is_five 
  (a : ℕ → ℝ) 
  (h_arithmetic : arithmetic_sequence a) 
  (h_sum1 : a 2 + a 6 = 8) 
  (h_sum2 : a 3 + a 4 = 3) : 
  ∃ d : ℝ, d = 5 ∧ ∀ n : ℕ, a (n + 1) = a n + d :=
sorry

end common_difference_is_five_l235_23559


namespace base_subtraction_equals_160_l235_23594

def base_to_decimal (digits : List Nat) (base : Nat) : Nat :=
  digits.foldr (fun d acc => d + base * acc) 0

theorem base_subtraction_equals_160 :
  let base9_to_decimal := base_to_decimal [3, 2, 5] 9
  let base6_to_decimal := base_to_decimal [2, 5, 4] 6
  base9_to_decimal - base6_to_decimal = 160 := by
sorry

end base_subtraction_equals_160_l235_23594


namespace geometric_sequence_third_term_l235_23517

/-- A geometric sequence is a sequence where each term after the first is found by multiplying the previous term by a fixed, non-zero number called the common ratio. -/
def GeometricSequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, r ≠ 0 ∧ ∀ n : ℕ, a (n + 1) = a n * r

theorem geometric_sequence_third_term
  (a : ℕ → ℝ)
  (h_geometric : GeometricSequence a)
  (h_first : a 1 = 1024)
  (h_fifth : a 5 = 128) :
  a 3 = 256 := by
sorry

end geometric_sequence_third_term_l235_23517


namespace ellipse_inequality_l235_23562

/-- An ellipse with equation ax^2 + by^2 = 1 and foci on the x-axis -/
structure Ellipse where
  a : ℝ
  b : ℝ
  is_ellipse : a > 0 ∧ b > 0
  foci_on_x_axis : True  -- We can't directly express this condition in Lean, so we use True as a placeholder

/-- Theorem: For an ellipse ax^2 + by^2 = 1 with foci on the x-axis, 0 < a < b -/
theorem ellipse_inequality (e : Ellipse) : 0 < e.a ∧ e.a < e.b := by
  sorry

end ellipse_inequality_l235_23562


namespace initial_boys_count_l235_23519

theorem initial_boys_count (total : ℕ) : 
  let initial_boys := (60 * total) / 100
  let final_total := total + 2
  let final_boys := initial_boys - 3
  (2 * final_boys = final_total) → initial_boys = 24 :=
by sorry

end initial_boys_count_l235_23519


namespace rationalize_denominator_l235_23523

theorem rationalize_denominator : 
  Real.sqrt (5 / (2 + Real.sqrt 2)) = Real.sqrt 10 / 2 := by
  sorry

end rationalize_denominator_l235_23523


namespace borrowed_sheets_theorem_l235_23534

/-- Represents a collection of lecture notes --/
structure LectureNotes where
  total_pages : Nat
  total_sheets : Nat
  borrowed_sheets : Nat

/-- Calculates the average of remaining page numbers after some sheets are borrowed --/
def averageRemainingPages (notes : LectureNotes) : Rat :=
  let remaining_sheets := notes.total_sheets - notes.borrowed_sheets
  let first_remaining_page := 2 * notes.borrowed_sheets + 1
  let last_remaining_page := notes.total_pages
  ((first_remaining_page + last_remaining_page) * remaining_sheets) / (2 * remaining_sheets)

/-- The theorem to be proved --/
theorem borrowed_sheets_theorem (notes : LectureNotes) :
  notes.total_pages = 72 ∧ 
  notes.total_sheets = 36 ∧ 
  notes.borrowed_sheets = 17 →
  averageRemainingPages notes = 40 := by
  sorry

#eval averageRemainingPages { total_pages := 72, total_sheets := 36, borrowed_sheets := 17 }

end borrowed_sheets_theorem_l235_23534


namespace quadratic_transform_sum_l235_23585

/-- Given a quadratic equation 9x^2 - 54x - 81 = 0, when transformed into (x+q)^2 = p,
    the sum of q and p is 15 -/
theorem quadratic_transform_sum (q p : ℝ) : 
  (∀ x, 9*x^2 - 54*x - 81 = 0 ↔ (x + q)^2 = p) → q + p = 15 := by
  sorry

end quadratic_transform_sum_l235_23585


namespace polynomial_root_sum_l235_23571

theorem polynomial_root_sum (a b : ℝ) : 
  (Complex.I * Real.sqrt 3 + 1 : ℂ) ^ 3 + a * (Complex.I * Real.sqrt 3 + 1) + b = 0 ∧ 
  (-3 : ℂ) ^ 3 + a * (-3) + b = 0 → 
  a + b = 11 := by sorry

end polynomial_root_sum_l235_23571


namespace smallest_possible_students_l235_23507

theorem smallest_possible_students : ∃ (n : ℕ), 
  (5 * n + 2 > 50) ∧ 
  (∀ m : ℕ, m < n → 5 * m + 2 ≤ 50) ∧
  (5 * n + 2 = 52) := by
  sorry

end smallest_possible_students_l235_23507


namespace average_side_length_of_squares_l235_23549

theorem average_side_length_of_squares (a₁ a₂ a₃ : ℝ) (h₁ : a₁ = 25) (h₂ : a₂ = 64) (h₃ : a₃ = 225) :
  (Real.sqrt a₁ + Real.sqrt a₂ + Real.sqrt a₃) / 3 = 28 / 3 := by
sorry

end average_side_length_of_squares_l235_23549
