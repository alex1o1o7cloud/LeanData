import Mathlib

namespace line_through_point_with_equal_intercepts_l3648_364814

-- Define a point in 2D space
structure Point2D where
  x : ℝ
  y : ℝ

-- Define a line in 2D space
structure Line2D where
  a : ℝ
  b : ℝ
  c : ℝ

-- Function to check if a point lies on a line
def pointOnLine (p : Point2D) (l : Line2D) : Prop :=
  l.a * p.x + l.b * p.y + l.c = 0

-- Function to check if a line has equal intercepts on both axes
def equalIntercepts (l : Line2D) : Prop :=
  l.a ≠ 0 ∧ l.b ≠ 0 ∧ l.c / l.a = l.c / l.b

-- The main theorem
theorem line_through_point_with_equal_intercepts :
  ∃ (l1 l2 : Line2D),
    (pointOnLine ⟨2, 3⟩ l1 ∧ equalIntercepts l1) ∧
    (pointOnLine ⟨2, 3⟩ l2 ∧ equalIntercepts l2) ∧
    ((l1.a = 1 ∧ l1.b = 1 ∧ l1.c = -5) ∨ (l2.a = 3 ∧ l2.b = -2 ∧ l2.c = 0)) :=
sorry

end line_through_point_with_equal_intercepts_l3648_364814


namespace cubic_function_symmetry_l3648_364832

/-- Given a cubic function f(x) = ax³ + bx + 5 where f(-9) = -7, prove that f(9) = 17 -/
theorem cubic_function_symmetry (a b : ℝ) (f : ℝ → ℝ) 
  (h1 : ∀ x, f x = a * x^3 + b * x + 5)
  (h2 : f (-9) = -7) : 
  f 9 = 17 := by
sorry

end cubic_function_symmetry_l3648_364832


namespace least_cookies_l3648_364822

theorem least_cookies (n : ℕ) : n = 59 ↔ 
  n > 0 ∧ 
  n % 6 = 5 ∧ 
  n % 8 = 3 ∧ 
  n % 9 = 6 ∧
  ∀ m : ℕ, m > 0 → m % 6 = 5 → m % 8 = 3 → m % 9 = 6 → n ≤ m :=
by sorry

end least_cookies_l3648_364822


namespace trapezoid_to_square_l3648_364880

/-- An isosceles trapezoid with given dimensions can be rearranged into a square -/
theorem trapezoid_to_square (b₁ b₂ h : ℝ) (h₁ : b₁ = 4) (h₂ : b₂ = 12) (h₃ : h = 4) :
  ∃ (s : ℝ), (b₁ + b₂) * h / 2 = s^2 := by
  sorry

end trapezoid_to_square_l3648_364880


namespace equation_solution_l3648_364811

theorem equation_solution : 
  ∃ x : ℚ, (10 * x + 2) / 4 - (3 * x - 6) / 18 = (2 * x + 4) / 3 ∧ x = 3 / 10 := by
  sorry

end equation_solution_l3648_364811


namespace problem_statement_l3648_364862

theorem problem_statement :
  (∀ x : ℝ, x^2 - 8*x + 17 > 0) ∧
  (∀ x : ℝ, (x + 2)^2 - (x - 3)^2 ≥ 0 → x ≥ 1/2) ∧
  (∃ n : ℕ, 11 ∣ 6*n^2 - 7) := by
  sorry

end problem_statement_l3648_364862


namespace laptop_price_l3648_364892

theorem laptop_price (sticker_price : ℝ) : 
  (sticker_price * 0.8 - 100 = sticker_price * 0.7 - 25) → sticker_price = 750 := by
sorry

end laptop_price_l3648_364892


namespace sector_area_l3648_364878

theorem sector_area (arc_length : Real) (central_angle : Real) :
  arc_length = π ∧ central_angle = π / 4 →
  let radius := arc_length / central_angle
  let sector_area := (1 / 2) * radius^2 * central_angle
  sector_area = 2 * π := by
  sorry

end sector_area_l3648_364878


namespace backpack_and_weight_difference_l3648_364877

/-- Given the weights of Bridget and Martha, and their combined weight with a backpack,
    prove the weight of the backpack and the weight difference between Bridget and Martha. -/
theorem backpack_and_weight_difference 
  (bridget_weight : ℕ) 
  (martha_weight : ℕ) 
  (combined_weight_with_backpack : ℕ) 
  (h1 : bridget_weight = 39)
  (h2 : martha_weight = 2)
  (h3 : combined_weight_with_backpack = 60) :
  (∃ backpack_weight : ℕ, 
    backpack_weight = combined_weight_with_backpack - (bridget_weight + martha_weight) ∧ 
    backpack_weight = 19) ∧ 
  (bridget_weight - martha_weight = 37) := by
  sorry

end backpack_and_weight_difference_l3648_364877


namespace doughnut_machine_completion_time_l3648_364804

-- Define the start time (6:00 AM) in minutes since midnight
def start_time : ℕ := 6 * 60

-- Define the time when one-fourth of the job is completed (9:00 AM) in minutes since midnight
def quarter_completion_time : ℕ := 9 * 60

-- Define the maintenance stop duration in minutes
def maintenance_duration : ℕ := 45

-- Define the completion time (6:45 PM) in minutes since midnight
def completion_time : ℕ := 18 * 60 + 45

-- Theorem statement
theorem doughnut_machine_completion_time :
  let working_duration := quarter_completion_time - start_time
  let total_duration := working_duration * 4 + maintenance_duration
  start_time + total_duration = completion_time :=
sorry

end doughnut_machine_completion_time_l3648_364804


namespace second_recipe_amount_is_one_l3648_364816

/-- The amount of lower sodium soy sauce in the second recipe -/
def second_recipe_amount : ℚ :=
  let bottle_ounces : ℚ := 16
  let ounces_per_cup : ℚ := 8
  let first_recipe_cups : ℚ := 2
  let third_recipe_cups : ℚ := 3
  let total_bottles : ℚ := 3
  let total_ounces : ℚ := total_bottles * bottle_ounces
  let total_cups : ℚ := total_ounces / ounces_per_cup
  total_cups - first_recipe_cups - third_recipe_cups

theorem second_recipe_amount_is_one :
  second_recipe_amount = 1 := by sorry

end second_recipe_amount_is_one_l3648_364816


namespace hyperbola_k_range_l3648_364858

theorem hyperbola_k_range (k : ℝ) :
  (∃ x y : ℝ, x^2 / (1 + k) - y^2 / (1 - k) = 1) →
  -1 < k ∧ k < 1 :=
by sorry

end hyperbola_k_range_l3648_364858


namespace pencil_total_length_l3648_364897

/-- The total length of a pencil with colored sections -/
def pencil_length (purple_length black_length blue_length : ℝ) : ℝ :=
  purple_length + black_length + blue_length

/-- Theorem: The total length of a pencil with specific colored sections is 4 cm -/
theorem pencil_total_length :
  pencil_length 1.5 0.5 2 = 4 := by
  sorry

end pencil_total_length_l3648_364897


namespace tom_candy_pieces_l3648_364850

theorem tom_candy_pieces (initial_boxes : ℕ) (given_away : ℕ) (pieces_per_box : ℕ) : 
  initial_boxes = 14 → given_away = 8 → pieces_per_box = 3 →
  (initial_boxes - given_away) * pieces_per_box = 18 := by
sorry

end tom_candy_pieces_l3648_364850


namespace crayon_selection_ways_l3648_364857

/-- The number of ways to choose k items from n items, where order doesn't matter -/
def choose (n k : ℕ) : ℕ := Nat.choose n k

/-- The number of crayons in the box -/
def total_crayons : ℕ := 15

/-- The number of crayons to be selected -/
def selected_crayons : ℕ := 5

theorem crayon_selection_ways : 
  choose total_crayons selected_crayons = 3003 := by sorry

end crayon_selection_ways_l3648_364857


namespace calculation_result_l3648_364824

-- Define the numbers in their respective bases
def num_base_8 : ℚ := 2 * 8^3 + 4 * 8^2 + 6 * 8^1 + 8 * 8^0
def num_base_5 : ℚ := 1 * 5^2 + 2 * 5^1 + 1 * 5^0
def num_base_9 : ℚ := 1 * 9^3 + 3 * 9^2 + 5 * 9^1 + 7 * 9^0
def num_base_10 : ℚ := 2048

-- Define the result of the calculation
def result : ℚ := num_base_8 / num_base_5 - num_base_9 + num_base_10

-- State the theorem
theorem calculation_result : result = 1061.1111 := by sorry

end calculation_result_l3648_364824


namespace lecture_duration_l3648_364879

/-- 
Given a lecture that lasts for 2 hours and m minutes, where the positions of the
hour and minute hands on the clock at the end of the lecture are exactly swapped
from their positions at the beginning, this theorem states that the integer part
of m is 46.
-/
theorem lecture_duration (m : ℝ) : 
  (∀ t : ℝ, 0 ≤ t ∧ t < 120 + m → 
    (5 * (120 + m - t) = (60 * t) % 360 ∨ 5 * (120 + m - t) = ((60 * t) % 360 + 360) % 360)) →
  Int.floor m = 46 := by
  sorry

end lecture_duration_l3648_364879


namespace harrison_croissant_price_l3648_364870

/-- The price of a regular croissant that Harrison buys -/
def regular_croissant_price : ℝ := 3.50

/-- The price of an almond croissant that Harrison buys -/
def almond_croissant_price : ℝ := 5.50

/-- The total amount Harrison spends on croissants in a year -/
def total_spent : ℝ := 468

/-- The number of weeks in a year -/
def weeks_in_year : ℕ := 52

theorem harrison_croissant_price :
  regular_croissant_price * weeks_in_year + almond_croissant_price * weeks_in_year = total_spent :=
by sorry

end harrison_croissant_price_l3648_364870


namespace students_yes_R_is_400_l3648_364891

/-- Given information about student responses to subjects M and R -/
structure StudentResponses where
  total : Nat
  yes_only_M : Nat
  no_both : Nat

/-- Calculate the number of students who answered yes for subject R -/
def students_yes_R (responses : StudentResponses) : Nat :=
  responses.total - responses.yes_only_M - responses.no_both

/-- Theorem stating that the number of students who answered yes for R is 400 -/
theorem students_yes_R_is_400 (responses : StudentResponses)
  (h1 : responses.total = 800)
  (h2 : responses.yes_only_M = 170)
  (h3 : responses.no_both = 230) :
  students_yes_R responses = 400 := by
  sorry

#eval students_yes_R ⟨800, 170, 230⟩

end students_yes_R_is_400_l3648_364891


namespace copper_percentage_bounds_l3648_364840

/-- Represents an alloy composition -/
structure Alloy where
  nickel : ℝ
  copper : ℝ
  manganese : ℝ
  sum_to_one : nickel + copper + manganese = 1

/-- The three given alloys -/
def alloy1 : Alloy := ⟨0.3, 0.7, 0, by norm_num⟩
def alloy2 : Alloy := ⟨0, 0.1, 0.9, by norm_num⟩
def alloy3 : Alloy := ⟨0.15, 0.25, 0.6, by norm_num⟩

/-- The theorem stating the bounds on copper percentage in the new alloy -/
theorem copper_percentage_bounds (x₁ x₂ x₃ : ℝ) 
  (sum_to_one : x₁ + x₂ + x₃ = 1)
  (manganese_constraint : 0.9 * x₂ + 0.6 * x₃ = 0.4) :
  let copper_percentage := 0.7 * x₁ + 0.1 * x₂ + 0.25 * x₃
  0.4 ≤ copper_percentage ∧ copper_percentage ≤ 13/30 := by
  sorry


end copper_percentage_bounds_l3648_364840


namespace min_non_red_surface_fraction_for_specific_cube_l3648_364836

/-- Represents a cube with given edge length and colored subcubes -/
structure ColoredCube where
  edge_length : ℕ
  red_cubes : ℕ
  white_cubes : ℕ
  blue_cubes : ℕ

/-- Calculate the minimum non-red surface area fraction of a ColoredCube -/
def min_non_red_surface_fraction (c : ColoredCube) : ℚ :=
  sorry

/-- The theorem to be proved -/
theorem min_non_red_surface_fraction_for_specific_cube :
  let c := ColoredCube.mk 4 48 12 4
  min_non_red_surface_fraction c = 1/8 := by sorry

end min_non_red_surface_fraction_for_specific_cube_l3648_364836


namespace projection_theorem_l3648_364852

def v : Fin 3 → ℝ := ![1, 2, 3]
def proj_v : Fin 3 → ℝ := ![2, 4, 6]
def u : Fin 3 → ℝ := ![2, 1, -1]

theorem projection_theorem (w : Fin 3 → ℝ) 
  (hw : ∃ (k : ℝ), w = fun i => k * proj_v i) :
  let proj_u := (u • w) / (w • w) • w
  proj_u = fun i => (![1/14, 1/7, 3/14] : Fin 3 → ℝ) i := by
  sorry

#check projection_theorem

end projection_theorem_l3648_364852


namespace permutation_count_mod_500_l3648_364803

/-- Represents the number of ways to arrange letters in specific positions -/
def arrange (n m : ℕ) : ℕ := Nat.choose n m

/-- Calculates the sum of products of arrangements for different k values -/
def sum_arrangements : ℕ :=
  (arrange 5 1 * arrange 6 0 * arrange 7 2) +
  (arrange 5 2 * arrange 6 1 * arrange 7 3) +
  (arrange 5 3 * arrange 6 2 * arrange 7 4) +
  (arrange 5 4 * arrange 6 3 * arrange 7 5) +
  (arrange 5 5 * arrange 6 4 * arrange 7 6)

/-- The main theorem stating the result of the permutation count modulo 500 -/
theorem permutation_count_mod_500 :
  sum_arrangements % 500 = 160 := by sorry

end permutation_count_mod_500_l3648_364803


namespace octagon_pyramid_volume_l3648_364861

/-- A right pyramid with a regular octagon base and one equilateral triangular face --/
structure OctagonPyramid where
  /-- Side length of the equilateral triangular face --/
  side_length : ℝ
  /-- The base is a regular octagon --/
  is_regular_octagon : Bool
  /-- The pyramid is a right pyramid --/
  is_right_pyramid : Bool
  /-- One face is an equilateral triangle --/
  has_equilateral_face : Bool

/-- Calculate the volume of the octagon pyramid --/
noncomputable def volume (p : OctagonPyramid) : ℝ :=
  sorry

/-- Theorem stating the volume of the specific octagon pyramid --/
theorem octagon_pyramid_volume :
  ∀ (p : OctagonPyramid),
    p.side_length = 10 ∧
    p.is_regular_octagon ∧
    p.is_right_pyramid ∧
    p.has_equilateral_face →
    volume p = 1000 * Real.sqrt 2 :=
by
  sorry

end octagon_pyramid_volume_l3648_364861


namespace closest_to_fraction_l3648_364815

def options : List ℝ := [4000, 5000, 6000, 7000, 8000]

theorem closest_to_fraction (x : ℝ) (h : x = 510 / 0.125) :
  4000 ∈ options ∧ ∀ y ∈ options, |x - 4000| ≤ |x - y| :=
by sorry

end closest_to_fraction_l3648_364815


namespace sum_10_with_7_dice_l3648_364813

/-- The number of ways to roll a sum of 10 with 7 fair 6-sided dice -/
def ways_to_roll_10_with_7_dice : ℕ :=
  Nat.choose 9 6

/-- The probability of rolling a sum of 10 with 7 fair 6-sided dice -/
def prob_sum_10_7_dice : ℚ :=
  ways_to_roll_10_with_7_dice / (6^7 : ℚ)

theorem sum_10_with_7_dice :
  ways_to_roll_10_with_7_dice = 84 ∧
  prob_sum_10_7_dice = 84 / (6^7 : ℚ) := by
  sorry

end sum_10_with_7_dice_l3648_364813


namespace M_intersect_P_equals_singleton_l3648_364829

-- Define the sets M and P
def M : Set (ℝ × ℝ) := {(x, y) | 4 * x + y = 6}
def P : Set (ℝ × ℝ) := {(x, y) | 3 * x + 2 * y = 7}

-- Theorem statement
theorem M_intersect_P_equals_singleton : M ∩ P = {(1, 2)} := by
  sorry

end M_intersect_P_equals_singleton_l3648_364829


namespace austins_change_l3648_364885

/-- The amount of change Austin had left after buying robots --/
def change_left (num_robots : ℕ) (robot_cost tax initial_amount : ℚ) : ℚ :=
  initial_amount - (num_robots * robot_cost + tax)

/-- Theorem stating that Austin's change is $11.53 --/
theorem austins_change :
  change_left 7 8.75 7.22 80 = 11.53 := by
  sorry

end austins_change_l3648_364885


namespace rent_split_l3648_364847

theorem rent_split (total_rent : ℕ) (num_people : ℕ) (individual_rent : ℕ) :
  total_rent = 490 →
  num_people = 7 →
  individual_rent = total_rent / num_people →
  individual_rent = 70 := by
  sorry

end rent_split_l3648_364847


namespace volunteer_assignment_l3648_364859

-- Define the number of volunteers
def num_volunteers : ℕ := 5

-- Define the number of venues
def num_venues : ℕ := 3

-- Define the function to calculate the number of ways to assign volunteers
def ways_to_assign (volunteers : ℕ) (venues : ℕ) : ℕ :=
  venues^volunteers - venues * (venues - 1)^volunteers

-- Theorem statement
theorem volunteer_assignment :
  ways_to_assign num_volunteers num_venues = 147 := by
  sorry

end volunteer_assignment_l3648_364859


namespace cone_surface_area_l3648_364868

/-- The surface area of a cone given its slant height and lateral surface central angle -/
theorem cone_surface_area (s : ℝ) (θ : ℝ) (h_s : s = 3) (h_θ : θ = 2 * Real.pi / 3) :
  s * θ / 2 + Real.pi * (s * θ / (2 * Real.pi))^2 = 4 * Real.pi := by
  sorry

end cone_surface_area_l3648_364868


namespace binomial_mode_maximizes_pmf_l3648_364845

/-- The number of trials -/
def n : ℕ := 5

/-- The probability of success -/
def p : ℚ := 3/4

/-- The binomial probability mass function -/
def binomialPMF (k : ℕ) : ℚ :=
  (n.choose k) * p^k * (1-p)^(n-k)

/-- The mode of the binomial distribution -/
def binomialMode : ℕ := 4

/-- Theorem stating that the binomial mode maximizes the probability mass function -/
theorem binomial_mode_maximizes_pmf :
  ∀ k : ℕ, k ≤ n → binomialPMF binomialMode ≥ binomialPMF k :=
sorry

end binomial_mode_maximizes_pmf_l3648_364845


namespace parallel_lines_angle_measure_l3648_364817

-- Define the angle measures as real numbers
variable (angle1 angle2 angle5 : ℝ)

-- State the theorem
theorem parallel_lines_angle_measure :
  -- Conditions
  angle1 = (1 / 4) * angle2 →  -- ∠1 is 1/4 of ∠2
  angle1 = angle5 →            -- ∠1 and ∠5 are alternate angles (implied by parallel lines)
  angle2 + angle5 = 180 →      -- ∠2 and ∠5 form a straight line
  -- Conclusion
  angle5 = 36 := by
sorry

end parallel_lines_angle_measure_l3648_364817


namespace oil_barrel_difference_l3648_364854

theorem oil_barrel_difference :
  ∀ (a b : ℝ),
  a + b = 100 →
  (a + 15) = 4 * (b - 15) →
  a - b = 30 := by
sorry

end oil_barrel_difference_l3648_364854


namespace angle_measure_theorem_l3648_364887

theorem angle_measure_theorem (x : ℝ) : 
  (90 - x) = (180 - x) - 4 → x = 60 := by
  sorry

end angle_measure_theorem_l3648_364887


namespace multiples_of_15_between_25_and_225_l3648_364846

theorem multiples_of_15_between_25_and_225 : 
  (Finset.range 226 
    |>.filter (fun n => n ≥ 25 ∧ n % 15 = 0)
    |>.card) = 14 := by
  sorry

end multiples_of_15_between_25_and_225_l3648_364846


namespace x_plus_y_eq_1_is_linear_l3648_364881

/-- A linear equation with two variables is of the form ax + by = c, where a, b, and c are constants. -/
def IsLinearEquationTwoVars (f : ℝ → ℝ → ℝ) : Prop :=
  ∃ (a b c : ℝ), ∀ x y, f x y = a * x + b * y - c

/-- The function representing x + y = 1 -/
def f (x y : ℝ) : ℝ := x + y - 1

/-- Theorem stating that x + y = 1 is a linear equation with two variables -/
theorem x_plus_y_eq_1_is_linear : IsLinearEquationTwoVars f := by
  sorry


end x_plus_y_eq_1_is_linear_l3648_364881


namespace candy_probability_contradiction_l3648_364843

theorem candy_probability_contradiction :
  ∀ (packet1_blue packet1_total packet2_blue packet2_total : ℕ),
    packet1_blue ≤ packet1_total →
    packet2_blue ≤ packet2_total →
    (3 : ℚ) / 8 ≤ (packet1_blue + packet2_blue : ℚ) / (packet1_total + packet2_total) →
    (packet1_blue + packet2_blue : ℚ) / (packet1_total + packet2_total) ≤ 2 / 5 →
    ¬((17 : ℚ) / 40 ≥ 3 / 8 ∧ 17 / 40 ≤ 2 / 5) :=
by
  sorry

end candy_probability_contradiction_l3648_364843


namespace quadratic_equation_solutions_l3648_364856

theorem quadratic_equation_solutions :
  let f : ℝ → ℝ := λ x ↦ x^2 - 25
  ∃ x₁ x₂ : ℝ, x₁ = 5 ∧ x₂ = -5 ∧ f x₁ = 0 ∧ f x₂ = 0 ∧
  ∀ x : ℝ, f x = 0 → x = x₁ ∨ x = x₂ :=
by
  sorry

end quadratic_equation_solutions_l3648_364856


namespace floor_product_eq_square_l3648_364834

def floor (x : ℚ) : ℤ := Int.floor x

theorem floor_product_eq_square (x : ℤ) : 
  (floor (x / 2 : ℚ)) * (floor (x / 3 : ℚ)) * (floor (x / 4 : ℚ)) = x^2 ↔ x = 0 ∨ x = 24 :=
by sorry

end floor_product_eq_square_l3648_364834


namespace right_cone_diameter_l3648_364882

theorem right_cone_diameter (h : ℝ) (s : ℝ) (d : ℝ) :
  h = 3 →
  s = 5 →
  s^2 = h^2 + (d/2)^2 →
  d = 8 :=
by sorry

end right_cone_diameter_l3648_364882


namespace divisibility_implies_gcd_greater_than_one_l3648_364888

theorem divisibility_implies_gcd_greater_than_one
  (a b c d : ℕ+)
  (h : (a.val * c.val + b.val * d.val) % (a.val^2 + b.val^2) = 0) :
  Nat.gcd (c.val^2 + d.val^2) (a.val^2 + b.val^2) > 1 :=
by sorry

end divisibility_implies_gcd_greater_than_one_l3648_364888


namespace diplomats_not_speaking_russian_l3648_364807

theorem diplomats_not_speaking_russian (total : ℕ) (french : ℕ) (neither_percent : ℚ) (both_percent : ℚ) :
  total = 150 →
  french = 17 →
  neither_percent = 1/5 →
  both_percent = 1/10 →
  ∃ (not_russian : ℕ), not_russian = 32 :=
by sorry

end diplomats_not_speaking_russian_l3648_364807


namespace natasha_quarters_l3648_364801

theorem natasha_quarters (q : ℕ) : 
  (10 < (q : ℚ) * (1/4) ∧ (q : ℚ) * (1/4) < 200) ∧ 
  q % 4 = 2 ∧ q % 5 = 2 ∧ q % 6 = 2 ↔ 
  ∃ k : ℕ, k ≥ 1 ∧ k ≤ 13 ∧ q = 60 * k + 2 :=
by sorry

end natasha_quarters_l3648_364801


namespace optimal_solution_l3648_364894

/-- Represents the factory worker allocation problem -/
structure FactoryProblem where
  total_workers : ℕ
  salary_a : ℕ
  salary_b : ℕ
  job_b_constraint : ℕ → Prop

/-- The specific factory problem instance -/
def factory_instance : FactoryProblem where
  total_workers := 120
  salary_a := 800
  salary_b := 1000
  job_b_constraint := fun x => (120 - x) ≥ 3 * x

/-- Calculate the total monthly salary -/
def total_salary (p : FactoryProblem) (workers_a : ℕ) : ℕ :=
  p.salary_a * workers_a + p.salary_b * (p.total_workers - workers_a)

/-- Theorem stating the optimal solution -/
theorem optimal_solution (p : FactoryProblem) :
  p = factory_instance →
  (∀ x : ℕ, x ≤ p.total_workers → p.job_b_constraint x → total_salary p x ≥ 114000) ∧
  total_salary p 30 = 114000 ∧
  p.job_b_constraint 30 := by
  sorry

end optimal_solution_l3648_364894


namespace edwin_alvin_age_difference_l3648_364855

/-- Represents the age difference between Edwin and Alvin -/
def ageDifference (edwinAge alvinAge : ℝ) : ℝ := edwinAge - alvinAge

/-- Theorem stating the age difference between Edwin and Alvin -/
theorem edwin_alvin_age_difference :
  ∃ (edwinAge alvinAge : ℝ),
    edwinAge > alvinAge ∧
    edwinAge + 2 = (1/3) * (alvinAge + 2) + 20 ∧
    edwinAge + alvinAge = 30.99999999 ∧
    ageDifference edwinAge alvinAge = 12 := by sorry

end edwin_alvin_age_difference_l3648_364855


namespace carl_has_more_stamps_l3648_364886

/-- Given that Carl has 89 stamps and Kevin has 57 stamps, 
    prove that Carl has 32 more stamps than Kevin. -/
theorem carl_has_more_stamps (carl_stamps : ℕ) (kevin_stamps : ℕ) 
  (h1 : carl_stamps = 89) (h2 : kevin_stamps = 57) : 
  carl_stamps - kevin_stamps = 32 := by
  sorry

end carl_has_more_stamps_l3648_364886


namespace even_function_sum_l3648_364819

-- Define the function f
def f (a b x : ℝ) : ℝ := a * x^2 + b * x

-- State the theorem
theorem even_function_sum (a b : ℝ) :
  (∀ x ∈ Set.Icc (a - 1) (2 * a), f a b x = f a b (-x)) →
  a + b = 1/3 := by
  sorry

end even_function_sum_l3648_364819


namespace fraction_equality_l3648_364874

theorem fraction_equality (a b : ℝ) (h : b / a = 3 / 5) : (a - b) / a = 2 / 5 := by
  sorry

end fraction_equality_l3648_364874


namespace ribbon_cuts_l3648_364820

/-- The number of cuts needed to divide ribbon rolls into smaller pieces -/
def cuts_needed (num_rolls : ℕ) (roll_length : ℕ) (piece_length : ℕ) : ℕ :=
  num_rolls * ((roll_length / piece_length) - 1)

/-- Theorem: The number of cuts needed to divide 5 rolls of 50-meter ribbon into 2-meter pieces is 120 -/
theorem ribbon_cuts : cuts_needed 5 50 2 = 120 := by
  sorry

end ribbon_cuts_l3648_364820


namespace tan_graph_property_l3648_364809

theorem tan_graph_property (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (∀ x, x ≠ π / 4 → ∃ y, y = a * Real.tan (b * x)) →
  (3 = a * Real.tan (b * π / 8)) →
  ab = 6 := by
  sorry

end tan_graph_property_l3648_364809


namespace sequence_range_l3648_364835

def is_increasing (a : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, a n < a (n + 1)

theorem sequence_range (a : ℕ → ℝ) 
  (h1 : ∀ n : ℕ, 2 * a (n + 1) * a n + a (n + 1) - 3 * a n = 0)
  (h2 : a 1 > 0)
  (h3 : is_increasing a) :
  0 < a 1 ∧ a 1 < 1 := by
  sorry

end sequence_range_l3648_364835


namespace typist_salary_problem_l3648_364869

theorem typist_salary_problem (S : ℝ) : 
  S * 1.1 * 0.95 = 3135 → S = 3000 := by
  sorry

end typist_salary_problem_l3648_364869


namespace rectangular_solid_surface_area_l3648_364837

theorem rectangular_solid_surface_area (a b c : ℕ) : 
  Prime a ∧ Prime b ∧ Prime c →
  a * b * c = 399 →
  2 * (a * b + b * c + c * a) = 422 := by
  sorry

end rectangular_solid_surface_area_l3648_364837


namespace equal_roots_quadratic_l3648_364805

theorem equal_roots_quadratic (k : ℝ) : 
  (∃ x : ℝ, 3 * x^2 + k * x + 12 = 0 ∧ 
   ∀ y : ℝ, 3 * y^2 + k * y + 12 = 0 → y = x) → 
  k = 12 ∨ k = -12 := by
sorry

end equal_roots_quadratic_l3648_364805


namespace largest_multiple_13_negation_gt_neg150_l3648_364844

theorem largest_multiple_13_negation_gt_neg150 : 
  ∀ n : ℤ, n * 13 > 143 → -(n * 13) ≤ -150 :=
by
  sorry

end largest_multiple_13_negation_gt_neg150_l3648_364844


namespace fencing_calculation_l3648_364848

/-- Calculates the fencing required for a rectangular field -/
theorem fencing_calculation (area : ℝ) (uncovered_side : ℝ) : 
  area > 0 → uncovered_side > 0 → area / uncovered_side + 2 * uncovered_side = 76 := by
  sorry

#check fencing_calculation

end fencing_calculation_l3648_364848


namespace fraction_equivalence_l3648_364841

theorem fraction_equivalence (x : ℝ) : (x + 1) / (x + 3) = 1 / 3 ↔ x = 0 := by
  sorry

end fraction_equivalence_l3648_364841


namespace fraction_addition_simplification_l3648_364849

theorem fraction_addition_simplification : 7/8 + 3/5 = 59/40 := by
  sorry

end fraction_addition_simplification_l3648_364849


namespace sally_pokemon_cards_sally_pokemon_cards_proof_l3648_364812

theorem sally_pokemon_cards : ℕ → Prop :=
  fun x =>
    let sally_initial : ℕ := 27
    let dan_cards : ℕ := 41
    let difference : ℕ := 6
    sally_initial + x = dan_cards + difference →
    x = 20

-- The proof is omitted
theorem sally_pokemon_cards_proof : ∃ x, sally_pokemon_cards x :=
  sorry

end sally_pokemon_cards_sally_pokemon_cards_proof_l3648_364812


namespace min_abs_b_minus_c_l3648_364867

/-- Given real numbers a, b, c satisfying (a - 2b - 1)² + (a - c - ln c)² = 0,
    the minimum value of |b - c| is 1. -/
theorem min_abs_b_minus_c (a b c : ℝ) 
    (h : (a - 2*b - 1)^2 + (a - c - Real.log c)^2 = 0) :
    ∀ x : ℝ, |b - c| ≤ x → 1 ≤ x :=
by sorry

end min_abs_b_minus_c_l3648_364867


namespace greatest_rational_root_l3648_364895

-- Define the quadratic equation type
structure QuadraticEquation where
  a : Nat
  b : Nat
  c : Nat
  h_a : a ≤ 100
  h_b : b ≤ 100
  h_c : c ≤ 100

-- Define a rational root
def RationalRoot (q : QuadraticEquation) (x : ℚ) : Prop :=
  q.a * x^2 + q.b * x + q.c = 0

-- State the theorem
theorem greatest_rational_root (q : QuadraticEquation) :
  ∃ (x : ℚ), RationalRoot q x ∧ 
  ∀ (y : ℚ), RationalRoot q y → y ≤ x ∧ x = -1/99 :=
sorry

end greatest_rational_root_l3648_364895


namespace cube_surface_area_equal_prism_volume_l3648_364883

theorem cube_surface_area_equal_prism_volume (a b c : ℝ) (h : a = 6 ∧ b = 3 ∧ c = 36) :
  let prism_volume := a * b * c
  let cube_edge := (prism_volume) ^ (1/3 : ℝ)
  let cube_surface_area := 6 * cube_edge ^ 2
  cube_surface_area = 216 * 3 ^ (2/3 : ℝ) := by
  sorry

end cube_surface_area_equal_prism_volume_l3648_364883


namespace candy_distribution_l3648_364821

theorem candy_distribution (total_candy : ℕ) (num_students : ℕ) (pieces_per_student : ℕ) 
  (h1 : total_candy = 344)
  (h2 : num_students = 43)
  (h3 : pieces_per_student * num_students = total_candy) :
  pieces_per_student = 8 := by
  sorry

end candy_distribution_l3648_364821


namespace lucky_325th_number_l3648_364866

/-- A positive integer is "lucky" if the sum of its digits is 7. -/
def is_lucky (n : ℕ) : Prop :=
  n > 0 ∧ (Nat.digits 10 n).sum = 7

/-- The sequence of "lucky" numbers in ascending order. -/
def lucky_seq : ℕ → ℕ := sorry

theorem lucky_325th_number : lucky_seq 325 = 52000 := by sorry

end lucky_325th_number_l3648_364866


namespace perfect_square_sum_partition_not_perfect_square_sum_partition_l3648_364890

/-- A partition of a set of natural numbers -/
def Partition (n : ℕ) := Fin 2 → Finset ℕ

/-- Predicate to check if a partition satisfies the perfect square sum property -/
def HasPerfectSquareSum (p : Partition n) : Prop :=
  ∃ (i : Fin 2) (a b : ℕ), a ≠ b ∧ a ∈ p i ∧ b ∈ p i ∧ ∃ (k : ℕ), a + b = k^2

/-- The main theorem stating the property holds for all n ≥ 15 -/
theorem perfect_square_sum_partition (n : ℕ) (h : n ≥ 15) :
  ∀ (p : Partition n), HasPerfectSquareSum p :=
sorry

/-- The property does not hold for n < 15 -/
theorem not_perfect_square_sum_partition (n : ℕ) (h : n < 15) :
  ∃ (p : Partition n), ¬HasPerfectSquareSum p :=
sorry

end perfect_square_sum_partition_not_perfect_square_sum_partition_l3648_364890


namespace quadratic_minimum_l3648_364818

-- Define the quadratic function
def f (x : ℝ) : ℝ := 2 * x^2 - 8 * x + 15

-- State the theorem
theorem quadratic_minimum :
  (∃ (x_min : ℝ), ∀ (x : ℝ), f x ≥ f x_min) ∧
  (∃ (y_min : ℝ), y_min = f 2 ∧ y_min = 7) ∧
  (∀ (x : ℝ), f x ≥ 7) :=
sorry

end quadratic_minimum_l3648_364818


namespace exponent_calculations_l3648_364802

theorem exponent_calculations (a : ℝ) (h : a ≠ 0) : 
  (a^3 + a^3 ≠ a^6) ∧ 
  ((a^2)^3 ≠ a^5) ∧ 
  (a^2 * a^4 ≠ a^8) ∧ 
  (a^4 / a^3 = a) := by sorry

end exponent_calculations_l3648_364802


namespace polynomial_remainder_l3648_364889

theorem polynomial_remainder (x : ℝ) : (x^14 + 1) % (x + 1) = 2 := by
  sorry

end polynomial_remainder_l3648_364889


namespace line_not_in_third_quadrant_l3648_364899

-- Define a line type
structure Line where
  slope : ℝ
  yIntercept : ℝ

-- Define the line from the problem
def problemLine : Line := { slope := -1, yIntercept := 1 }

-- Define the third quadrant
def thirdQuadrant : Set (ℝ × ℝ) :=
  {p | p.1 < 0 ∧ p.2 < 0}

-- Theorem statement
theorem line_not_in_third_quadrant :
  ∀ (x y : ℝ), (y = problemLine.slope * x + problemLine.yIntercept) →
  (x, y) ∉ thirdQuadrant :=
sorry

end line_not_in_third_quadrant_l3648_364899


namespace triangle_top_angle_l3648_364871

theorem triangle_top_angle (total : ℝ) (right : ℝ) (left : ℝ) (top : ℝ) : 
  total = 250 →
  right = 60 →
  left = 2 * right →
  total = left + right + top →
  top = 70 := by
sorry

end triangle_top_angle_l3648_364871


namespace odd_product_pattern_l3648_364827

theorem odd_product_pattern (n : ℕ) (h : Odd n) : n * (n + 2) = (n + 1)^2 - 1 := by
  sorry

end odd_product_pattern_l3648_364827


namespace at_least_one_geq_two_l3648_364839

theorem at_least_one_geq_two (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  (a + 1/b ≥ 2) ∨ (b + 1/c ≥ 2) ∨ (c + 1/a ≥ 2) := by
  sorry

end at_least_one_geq_two_l3648_364839


namespace correct_seating_arrangements_l3648_364830

/-- Represents a seating arrangement in an examination room --/
structure ExamRoom :=
  (rows : Nat)
  (columns : Nat)

/-- Calculates the number of seating arrangements for two students
    who cannot be seated adjacent to each other --/
def countSeatingArrangements (room : ExamRoom) : Nat :=
  sorry

/-- Theorem stating the correct number of seating arrangements --/
theorem correct_seating_arrangements :
  let room : ExamRoom := { rows := 5, columns := 6 }
  countSeatingArrangements room = 772 := by
  sorry

end correct_seating_arrangements_l3648_364830


namespace all_fruits_fallen_by_day_12_l3648_364863

/-- Represents the number of fruits that fall on a given day -/
def fruits_falling (day : ℕ) : ℕ :=
  if day ≤ 10 then day
  else (day - 10)

/-- Represents the total number of fruits that have fallen up to a given day -/
def total_fruits_fallen (day : ℕ) : ℕ :=
  if day ≤ 10 then day * (day + 1) / 2
  else 55 + (day - 10) * (day - 9) / 2

/-- The theorem stating that all fruits will have fallen by the end of the 12th day -/
theorem all_fruits_fallen_by_day_12 :
  total_fruits_fallen 12 = 58 ∧
  ∀ d : ℕ, d < 12 → total_fruits_fallen d < 58 := by
  sorry


end all_fruits_fallen_by_day_12_l3648_364863


namespace clothing_colors_l3648_364806

-- Define the colors
inductive Color
| Red
| Blue

-- Define a structure for a child's clothing
structure Clothing where
  tshirt : Color
  shorts : Color

-- Define the four children
def Alyna : Clothing := sorry
def Bohdan : Clothing := sorry
def Vika : Clothing := sorry
def Grysha : Clothing := sorry

-- Define the theorem
theorem clothing_colors :
  -- Conditions
  (Alyna.tshirt = Color.Red) →
  (Bohdan.tshirt = Color.Red) →
  (Alyna.shorts ≠ Bohdan.shorts) →
  (Vika.tshirt ≠ Grysha.tshirt) →
  (Vika.shorts = Color.Blue) →
  (Grysha.shorts = Color.Blue) →
  (Alyna.tshirt ≠ Vika.tshirt) →
  (Alyna.shorts ≠ Vika.shorts) →
  -- Conclusion
  (Alyna = ⟨Color.Red, Color.Red⟩ ∧
   Bohdan = ⟨Color.Red, Color.Blue⟩ ∧
   Vika = ⟨Color.Blue, Color.Blue⟩ ∧
   Grysha = ⟨Color.Red, Color.Blue⟩) :=
by
  sorry


end clothing_colors_l3648_364806


namespace additional_tickets_needed_l3648_364896

def ferris_wheel_cost : ℕ := 5
def roller_coaster_cost : ℕ := 4
def bumper_cars_cost : ℕ := 4
def current_tickets : ℕ := 5

def total_cost : ℕ := ferris_wheel_cost + roller_coaster_cost + bumper_cars_cost

theorem additional_tickets_needed : 
  (total_cost - current_tickets : ℕ) = 8 := by sorry

end additional_tickets_needed_l3648_364896


namespace rectangle_area_equals_50_l3648_364865

/-- The area of a rectangle with height x and width 2x, whose perimeter is equal to the perimeter of an equilateral triangle with side length 10, is 50. -/
theorem rectangle_area_equals_50 : ∃ x : ℝ,
  let rectangle_height := x
  let rectangle_width := 2 * x
  let rectangle_perimeter := 2 * (rectangle_height + rectangle_width)
  let triangle_side_length := 10
  let triangle_perimeter := 3 * triangle_side_length
  let rectangle_area := rectangle_height * rectangle_width
  rectangle_perimeter = triangle_perimeter ∧ rectangle_area = 50 := by
sorry

end rectangle_area_equals_50_l3648_364865


namespace part_one_part_two_l3648_364872

-- Define the propositions p and q as functions of m
def p (m : ℝ) : Prop := ∃ x y : ℝ, x < 0 ∧ y < 0 ∧ x ≠ y ∧ x^2 + m*x + 1 = 0 ∧ y^2 + m*y + 1 = 0

def q (m : ℝ) : Prop := ∀ x : ℝ, 4*x^2 + 4*(m-2)*x + 1 ≠ 0

-- Theorem for part (1)
theorem part_one (m : ℝ) (h : ¬¬p m) : m > 2 := by
  sorry

-- Theorem for part (2)
theorem part_two (m : ℝ) (h1 : p m ∨ q m) (h2 : ¬(p m ∧ q m)) : m ≥ 3 ∨ (1 < m ∧ m ≤ 2) := by
  sorry

end part_one_part_two_l3648_364872


namespace protein_content_lower_bound_l3648_364838

/-- Represents the protein content of a beverage can -/
structure BeverageCan where
  netWeight : ℝ
  proteinPercentage : ℝ

/-- Theorem: Given a beverage can with net weight 300 grams and protein content ≥ 0.6%,
    the protein content is at least 1.8 grams -/
theorem protein_content_lower_bound (can : BeverageCan)
    (h1 : can.netWeight = 300)
    (h2 : can.proteinPercentage ≥ 0.6) :
    can.netWeight * (can.proteinPercentage / 100) ≥ 1.8 := by
  sorry

#check protein_content_lower_bound

end protein_content_lower_bound_l3648_364838


namespace damage_ratio_is_five_fourths_l3648_364876

/-- The ratio of damages for Winnie-the-Pooh's two falls -/
theorem damage_ratio_is_five_fourths
  (g : ℝ) (H : ℝ) (n : ℝ) (k : ℝ) (M : ℝ) (τ : ℝ)
  (h_pos : 0 < H)
  (n_pos : 0 < n)
  (k_pos : 0 < k)
  (g_pos : 0 < g)
  (h_def : H = n * (H / n)) :
  let h := H / n
  let V_I := Real.sqrt (2 * g * H)
  let V_1 := Real.sqrt (2 * g * h)
  let V_1' := (1 / k) * Real.sqrt (2 * g * h)
  let V_II := Real.sqrt ((1 / k^2) * 2 * g * h + 2 * g * (H - h))
  let I_I := M * V_I * τ
  let I_II := M * τ * ((V_1 - V_1') + V_II)
  I_II / I_I = 5 / 4 := by
sorry

end damage_ratio_is_five_fourths_l3648_364876


namespace tangent_slope_angle_at_one_zero_l3648_364826

/-- The slope angle of the tangent line to y = x^2 - x at (1, 0) is 45° -/
theorem tangent_slope_angle_at_one_zero :
  let f : ℝ → ℝ := λ x => x^2 - x
  let x₀ : ℝ := 1
  let y₀ : ℝ := f x₀
  let slope : ℝ := deriv f x₀
  Real.arctan slope * (180 / Real.pi) = 45 := by sorry

end tangent_slope_angle_at_one_zero_l3648_364826


namespace batsman_average_increase_l3648_364808

/-- Represents a batsman's score statistics -/
structure BatsmanStats where
  innings : ℕ
  totalScore : ℕ
  average : ℚ

/-- Calculates the new average after an additional inning -/
def newAverage (stats : BatsmanStats) (newScore : ℕ) : ℚ :=
  (stats.totalScore + newScore) / (stats.innings + 1)

/-- Theorem: If a batsman's average increases by 1 after scoring 69 in the 11th inning,
    then the new average is 59 -/
theorem batsman_average_increase (stats : BatsmanStats) :
  stats.innings = 10 →
  newAverage stats 69 = stats.average + 1 →
  newAverage stats 69 = 59 := by
  sorry

end batsman_average_increase_l3648_364808


namespace product_in_second_quadrant_l3648_364875

/-- The complex number representing the product (2+i)(-1+i) -/
def z : ℂ := (2 + Complex.I) * (-1 + Complex.I)

/-- The real part of z -/
def real_part : ℝ := z.re

/-- The imaginary part of z -/
def imag_part : ℝ := z.im

/-- Predicate for a complex number being in the second quadrant -/
def in_second_quadrant (w : ℂ) : Prop := w.re < 0 ∧ w.im > 0

theorem product_in_second_quadrant : in_second_quadrant z := by
  sorry

end product_in_second_quadrant_l3648_364875


namespace sufficient_but_not_necessary_l3648_364893

theorem sufficient_but_not_necessary : 
  (∀ x : ℝ, x > 4 → (x > 3 ∨ x < -1)) ∧ 
  (∃ x : ℝ, (x > 3 ∨ x < -1) ∧ ¬(x > 4)) := by
  sorry

end sufficient_but_not_necessary_l3648_364893


namespace work_completion_proof_l3648_364853

/-- The original number of men working on a project -/
def original_men : ℕ := 48

/-- The number of days it takes the original group to complete the work -/
def original_days : ℕ := 60

/-- The number of additional men added to the group -/
def additional_men : ℕ := 8

/-- The number of days it takes the larger group to complete the work -/
def new_days : ℕ := 50

/-- The amount of work to be completed -/
def work : ℝ := 1

theorem work_completion_proof :
  (original_men : ℝ) * work / original_days = 
  ((original_men + additional_men) : ℝ) * work / new_days :=
by sorry

#check work_completion_proof

end work_completion_proof_l3648_364853


namespace algebraic_expression_value_l3648_364828

theorem algebraic_expression_value (x y : ℝ) 
  (h1 : x - y = 5) 
  (h2 : x * y = -3) : 
  x^2 * y - x * y^2 = -15 := by
  sorry

end algebraic_expression_value_l3648_364828


namespace arithmetic_sequence_ninth_term_l3648_364864

/-- Given an arithmetic sequence where the first term is 2/3 and the 17th term is 5/6,
    the 9th term is 3/4. -/
theorem arithmetic_sequence_ninth_term 
  (a : ℚ) 
  (seq : ℕ → ℚ) 
  (h1 : seq 1 = 2/3) 
  (h2 : seq 17 = 5/6) 
  (h3 : ∀ n : ℕ, seq (n + 1) - seq n = seq 2 - seq 1) : 
  seq 9 = 3/4 := by
  sorry

end arithmetic_sequence_ninth_term_l3648_364864


namespace fractional_equation_solution_range_l3648_364884

theorem fractional_equation_solution_range (a x : ℝ) : 
  ((a + 2) / (x + 1) = 1) ∧ 
  (x ≤ 0) ∧ 
  (x + 1 ≠ 0) → 
  (a ≤ -1) ∧ (a ≠ -2) :=
by sorry

end fractional_equation_solution_range_l3648_364884


namespace midpoint_triangle_is_equilateral_l3648_364810

-- Define the points in the plane
variable (A B C D E F G M N P : ℝ × ℝ)

-- Define the conditions
def is_midpoint (M A B : ℝ × ℝ) : Prop := M = ((A.1 + B.1) / 2, (A.2 + B.2) / 2)

def is_equilateral_triangle (X Y Z : ℝ × ℝ) : Prop :=
  (X.1 - Y.1)^2 + (X.2 - Y.2)^2 = (Y.1 - Z.1)^2 + (Y.2 - Z.2)^2 ∧
  (Y.1 - Z.1)^2 + (Y.2 - Z.2)^2 = (Z.1 - X.1)^2 + (Z.2 - X.2)^2

-- State the theorem
theorem midpoint_triangle_is_equilateral
  (h1 : is_midpoint M A B)
  (h2 : is_midpoint P G F)
  (h3 : is_midpoint N E F)
  (h4 : is_equilateral_triangle B C E)
  (h5 : is_equilateral_triangle C D F)
  (h6 : is_equilateral_triangle D A G) :
  is_equilateral_triangle M N P :=
sorry

end midpoint_triangle_is_equilateral_l3648_364810


namespace contractor_problem_l3648_364831

/-- Represents the efficiency of a worker --/
structure WorkerEfficiency where
  value : ℝ
  pos : value > 0

/-- Represents a group of workers with the same efficiency --/
structure WorkerGroup where
  count : ℕ
  efficiency : WorkerEfficiency

/-- Calculates the total work done by a group of workers in a day --/
def dailyWork (group : WorkerGroup) : ℝ :=
  group.count * group.efficiency.value

/-- Calculates the total work done by multiple groups of workers in a day --/
def totalDailyWork (groups : List WorkerGroup) : ℝ :=
  groups.map dailyWork |>.sum

/-- The contractor problem --/
theorem contractor_problem 
  (initialGroups : List WorkerGroup)
  (initialDays : ℕ)
  (totalDays : ℕ)
  (firedLessEfficient : ℕ)
  (firedMoreEfficient : ℕ)
  (h_initial_groups : initialGroups = [
    { count := 15, efficiency := { value := 1, pos := by sorry } },
    { count := 10, efficiency := { value := 1.5, pos := by sorry } }
  ])
  (h_initial_days : initialDays = 40)
  (h_total_days : totalDays = 150)
  (h_fired_less : firedLessEfficient = 4)
  (h_fired_more : firedMoreEfficient = 3)
  (h_one_third_complete : totalDailyWork initialGroups * initialDays = (1/3) * (totalDailyWork initialGroups * totalDays))
  : ∃ (remainingDays : ℕ), remainingDays = 112 ∧ 
    (totalDailyWork initialGroups * totalDays) = 
    (totalDailyWork initialGroups * initialDays + 
     totalDailyWork [
       { count := 15 - firedLessEfficient, efficiency := { value := 1, pos := by sorry } },
       { count := 10 - firedMoreEfficient, efficiency := { value := 1.5, pos := by sorry } }
     ] * remainingDays) := by
  sorry


end contractor_problem_l3648_364831


namespace cube_volume_edge_relation_l3648_364842

theorem cube_volume_edge_relation (a : ℝ) (a' : ℝ) (ha : a > 0) (ha' : a' > 0) :
  (a' ^ 3) = 27 * (a ^ 3) → a' = 3 * a := by
  sorry

end cube_volume_edge_relation_l3648_364842


namespace perfect_square_quadratic_l3648_364860

theorem perfect_square_quadratic (c : ℝ) : 
  (∀ x : ℝ, ∃ y : ℝ, x^2 + 14*x + c = y^2) → c = 49 := by
  sorry

end perfect_square_quadratic_l3648_364860


namespace julia_normal_mile_time_l3648_364800

/-- Represents Julia's running times -/
structure JuliaRunningTimes where
  normalMileTime : ℝ
  newShoesMileTime : ℝ

/-- The conditions of the problem -/
def problemConditions (j : JuliaRunningTimes) : Prop :=
  j.newShoesMileTime = 13 ∧
  5 * j.newShoesMileTime = 5 * j.normalMileTime + 15

/-- The theorem stating Julia's normal mile time -/
theorem julia_normal_mile_time (j : JuliaRunningTimes) 
  (h : problemConditions j) : j.normalMileTime = 10 := by
  sorry


end julia_normal_mile_time_l3648_364800


namespace range_of_t_for_true_proposition_l3648_364833

theorem range_of_t_for_true_proposition (t : ℝ) :
  (∀ x : ℝ, x ≥ 1 → (x^2 + 2*x + t) / x > 0) ↔ t > -3 :=
by sorry

end range_of_t_for_true_proposition_l3648_364833


namespace bobs_remaining_funds_l3648_364851

/-- Converts a number from octal to decimal --/
def octal_to_decimal (n : ℕ) : ℕ := sorry

/-- Calculates the remaining funds after expenses --/
def remaining_funds (savings : ℕ) (ticket_cost : ℕ) (meal_cost : ℕ) : ℕ :=
  savings - (ticket_cost + meal_cost)

theorem bobs_remaining_funds :
  let bobs_savings : ℕ := octal_to_decimal 7777
  let ticket_cost : ℕ := 1500
  let meal_cost : ℕ := 250
  remaining_funds bobs_savings ticket_cost meal_cost = 2345 := by sorry

end bobs_remaining_funds_l3648_364851


namespace smallest_x_for_540x_perfect_square_l3648_364873

theorem smallest_x_for_540x_perfect_square :
  ∃ (x : ℕ+), 
    (∀ (y : ℕ+), ∃ (M : ℤ), 540 * y = M^2 → x ≤ y) ∧
    (∃ (M : ℤ), 540 * x = M^2) ∧
    x = 15 := by
  sorry

end smallest_x_for_540x_perfect_square_l3648_364873


namespace gummy_bears_count_l3648_364823

/-- The number of gummy bears produced per minute -/
def production_rate : ℕ := 300

/-- The time taken to produce enough gummy bears to fill the packets (in minutes) -/
def production_time : ℕ := 40

/-- The number of packets filled with the gummy bears produced -/
def num_packets : ℕ := 240

/-- The number of gummy bears in each packet -/
def gummy_bears_per_packet : ℕ := production_rate * production_time / num_packets

theorem gummy_bears_count : gummy_bears_per_packet = 50 := by
  sorry

end gummy_bears_count_l3648_364823


namespace angle_measure_in_regular_octagon_l3648_364898

/-- A regular octagon is a polygon with 8 sides of equal length and 8 angles of equal measure. -/
structure RegularOctagon where
  vertices : Fin 8 → ℝ × ℝ
  is_regular : sorry

/-- Given three vertices of a regular octagon with one vertex between each pair -/
def skip_one_vertex (o : RegularOctagon) (i j k : Fin 8) : Prop :=
  (j - i) % 8 = 2 ∧ (k - j) % 8 = 2

/-- The angle between three points in a plane -/
def angle_measure (p1 p2 p3 : ℝ × ℝ) : ℝ := sorry

theorem angle_measure_in_regular_octagon (o : RegularOctagon) (i j k : Fin 8) 
  (h : skip_one_vertex o i j k) : 
  angle_measure (o.vertices i) (o.vertices j) (o.vertices k) = 135 := by
  sorry

end angle_measure_in_regular_octagon_l3648_364898


namespace average_of_arithmetic_sequence_l3648_364825

theorem average_of_arithmetic_sequence (z : ℝ) : 
  let seq := [5, 5 + 3*z, 5 + 6*z, 5 + 9*z, 5 + 12*z]
  (seq.sum / seq.length : ℝ) = 5 + 6*z := by sorry

end average_of_arithmetic_sequence_l3648_364825
