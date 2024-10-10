import Mathlib

namespace lg_expression_equals_one_l2108_210802

-- Define the common logarithm (base 10)
noncomputable def lg (x : ℝ) : ℝ := Real.log x / Real.log 10

-- Theorem statement
theorem lg_expression_equals_one :
  lg 2 * lg 2 + lg 2 * lg 5 + lg 5 = 1 := by sorry

end lg_expression_equals_one_l2108_210802


namespace area_of_quadrilateral_l2108_210887

/-- A line with slope -3 intersecting positive x and y axes -/
structure Line1 where
  slope : ℝ
  x_intercept : ℝ
  y_intercept : ℝ

/-- Another line intersecting x and y axes -/
structure Line2 where
  x_intercept : ℝ
  y_intercept : ℝ

/-- Point of intersection of the two lines -/
structure IntersectionPoint where
  x : ℝ
  y : ℝ

/-- The problem setup -/
def ProblemSetup (l1 : Line1) (l2 : Line2) (e : IntersectionPoint) : Prop :=
  l1.slope = -3 ∧
  l1.x_intercept > 0 ∧
  l1.y_intercept > 0 ∧
  l2.x_intercept = 10 ∧
  e.x = 5 ∧
  e.y = 5

/-- The area of quadrilateral OBEC -/
def QuadrilateralArea (l1 : Line1) (l2 : Line2) (e : IntersectionPoint) : ℝ := 
  sorry  -- The actual calculation would go here

theorem area_of_quadrilateral (l1 : Line1) (l2 : Line2) (e : IntersectionPoint) 
  (h : ProblemSetup l1 l2 e) : QuadrilateralArea l1 l2 e = 75 := by
  sorry

end area_of_quadrilateral_l2108_210887


namespace c_minus_d_value_l2108_210823

theorem c_minus_d_value (c d : ℝ) 
  (eq1 : 2020 * c + 2024 * d = 2030)
  (eq2 : 2022 * c + 2026 * d = 2032) : 
  c - d = -4 := by
sorry

end c_minus_d_value_l2108_210823


namespace ellipse_sum_coordinates_and_axes_l2108_210846

/-- Theorem: For an ellipse with center (3, -5), semi-major axis length 7, and semi-minor axis length 4,
    the sum of its center coordinates and axis lengths is 9. -/
theorem ellipse_sum_coordinates_and_axes :
  ∀ (h k a b : ℝ),
    h = 3 →
    k = -5 →
    a = 7 →
    b = 4 →
    h + k + a + b = 9 :=
by
  sorry

end ellipse_sum_coordinates_and_axes_l2108_210846


namespace bus_departure_interval_l2108_210838

/-- Represents the number of minutes between 6:00 AM and 7:00 AM -/
def total_minutes : ℕ := 60

/-- Represents the number of bus departures between 6:00 AM and 7:00 AM -/
def num_departures : ℕ := 11

/-- Calculates the interval between consecutive bus departures -/
def interval (total : ℕ) (departures : ℕ) : ℚ :=
  (total : ℚ) / ((departures - 1) : ℚ)

/-- Proves that the interval between consecutive bus departures is 6 minutes -/
theorem bus_departure_interval :
  interval total_minutes num_departures = 6 := by
  sorry

end bus_departure_interval_l2108_210838


namespace first_term_of_arithmetic_sequence_l2108_210842

/-- An arithmetic sequence with given conditions -/
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem first_term_of_arithmetic_sequence
  (a : ℕ → ℝ)
  (h_arithmetic : arithmetic_sequence a)
  (h_third : a 3 = 30)
  (h_ninth : a 9 = 60) :
  a 1 = 20 := by
  sorry

end first_term_of_arithmetic_sequence_l2108_210842


namespace longest_segment_in_quarter_circle_l2108_210874

theorem longest_segment_in_quarter_circle (d : ℝ) (h : d = 18) :
  let r := d / 2
  let m := (2 * r^2)^(1/2)
  m^2 = 162 := by
  sorry

end longest_segment_in_quarter_circle_l2108_210874


namespace max_value_range_l2108_210879

noncomputable def f (a b x : ℝ) : ℝ :=
  if x ≤ a then -(x + 1) * Real.exp x else b * x - 1

theorem max_value_range (a b : ℝ) :
  ∃ M : ℝ, (∀ x, f a b x ≤ M) ∧ (0 < M) ∧ (M ≤ Real.exp (-2)) :=
sorry

end max_value_range_l2108_210879


namespace product_increase_theorem_l2108_210849

theorem product_increase_theorem :
  ∃ (a b c d e f g : ℕ),
    (a - 3) * (b - 3) * (c - 3) * (d - 3) * (e - 3) * (f - 3) * (g - 3) =
    13 * (a * b * c * d * e * f * g) :=
by sorry

end product_increase_theorem_l2108_210849


namespace pears_left_l2108_210811

theorem pears_left (keith_pears : ℕ) (total_pears : ℕ) : 
  keith_pears = 62 →
  total_pears = 186 →
  total_pears = keith_pears + 2 * keith_pears →
  140 = total_pears - (total_pears / 4) := by
  sorry

#check pears_left

end pears_left_l2108_210811


namespace problem_statement_l2108_210804

-- Define the geometric sequence a_n
def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

-- Define the arithmetic sequence b_n
def arithmetic_sequence (b : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, b (n + 1) = b n + d

-- State the theorem
theorem problem_statement (a b : ℕ → ℝ) :
  geometric_sequence a →
  a 3 * a 11 = 4 * a 7 →
  arithmetic_sequence b →
  a 7 = b 7 →
  b 5 + b 9 = 8 := by
  sorry

end problem_statement_l2108_210804


namespace expression_simplification_l2108_210820

theorem expression_simplification (x : ℝ) : 
  (1 + Real.sin (2 * x) - Real.cos (2 * x)) / (1 + Real.sin (2 * x) + Real.cos (2 * x)) = Real.tan x := by
  sorry

end expression_simplification_l2108_210820


namespace factorization_equality_l2108_210870

theorem factorization_equality (a b : ℝ) : a * b^2 - 4 * a * b + 4 * a = a * (b - 2)^2 := by
  sorry

end factorization_equality_l2108_210870


namespace used_car_selection_l2108_210855

theorem used_car_selection (num_cars : ℕ) (num_clients : ℕ) (selections_per_car : ℕ) :
  num_cars = 18 →
  num_clients = 18 →
  selections_per_car = 3 →
  (num_cars * selections_per_car) / num_clients = 3 := by
  sorry

end used_car_selection_l2108_210855


namespace expand_product_l2108_210836

theorem expand_product (x : ℝ) (hx : x ≠ 0) :
  (3 / 7) * ((7 / x^3) + 14 * x^5) = 3 / x^3 + 6 * x^5 := by
  sorry

end expand_product_l2108_210836


namespace unsupported_attendees_l2108_210872

-- Define the total number of attendees
def total_attendance : ℕ := 500

-- Define the percentage of supporters for each team
def team_a_percentage : ℚ := 35 / 100
def team_b_percentage : ℚ := 25 / 100
def team_c_percentage : ℚ := 20 / 100
def team_d_percentage : ℚ := 15 / 100

-- Define the overlap percentages
def team_ab_overlap_percentage : ℚ := 10 / 100
def team_bc_overlap_percentage : ℚ := 5 / 100
def team_cd_overlap_percentage : ℚ := 7 / 100

-- Define the number of people attending for atmosphere
def atmosphere_attendees : ℕ := 30

-- Theorem to prove
theorem unsupported_attendees :
  ∃ (unsupported : ℕ),
    unsupported = total_attendance -
      (((team_a_percentage + team_b_percentage + team_c_percentage + team_d_percentage) * total_attendance).floor -
       ((team_ab_overlap_percentage * team_a_percentage * total_attendance +
         team_bc_overlap_percentage * team_b_percentage * total_attendance +
         team_cd_overlap_percentage * team_c_percentage * total_attendance).floor) +
       atmosphere_attendees) ∧
    unsupported = 26 := by
  sorry

end unsupported_attendees_l2108_210872


namespace plants_cost_theorem_l2108_210826

/-- Calculates the final cost of plants given the original price, discount rate, tax rate, and delivery surcharge. -/
def finalCost (originalPrice discountRate taxRate deliverySurcharge : ℚ) : ℚ :=
  let discountedPrice := originalPrice * (1 - discountRate)
  let withTax := discountedPrice * (1 + taxRate)
  withTax + deliverySurcharge

/-- Theorem stating that the final cost of the plants is $440.71 given the specified conditions. -/
theorem plants_cost_theorem :
  finalCost 467 0.15 0.08 12 = 440.71 := by
  sorry

#eval finalCost 467 0.15 0.08 12

end plants_cost_theorem_l2108_210826


namespace arithmetic_sequence_10th_term_l2108_210860

/-- An arithmetic sequence with given conditions -/
def arithmetic_sequence (a : ℕ → ℚ) : Prop :=
  (∃ d : ℚ, ∀ n : ℕ, a (n + 1) = a n + d) ∧ 
  a 1 = 4 ∧
  a 2 + a 4 = 4

/-- The 10th term of the arithmetic sequence is -5 -/
theorem arithmetic_sequence_10th_term 
  (a : ℕ → ℚ) 
  (h : arithmetic_sequence a) : 
  a 10 = -5 := by
sorry

end arithmetic_sequence_10th_term_l2108_210860


namespace mystery_number_proof_l2108_210832

theorem mystery_number_proof (mystery : ℕ) : mystery * 24 = 173 * 240 → mystery = 1730 := by
  sorry

end mystery_number_proof_l2108_210832


namespace f_minimum_l2108_210880

def f (x : ℝ) : ℝ := (x^2 + 4*x + 5)*(x^2 + 4*x + 2) + 2*x^2 + 8*x + 1

theorem f_minimum :
  (∀ x : ℝ, f x ≥ -9) ∧ (∃ x : ℝ, f x = -9) :=
sorry

end f_minimum_l2108_210880


namespace rectangular_field_area_l2108_210861

theorem rectangular_field_area (a b c : ℝ) (h1 : a = 13) (h2 : c = 17) (h3 : a^2 + b^2 = c^2) :
  a * b = 26 * Real.sqrt 30 := by
  sorry

end rectangular_field_area_l2108_210861


namespace distinct_number_probability_l2108_210891

-- Define the number of balls of each color and the number to be selected
def total_red_balls : ℕ := 5
def total_black_balls : ℕ := 5
def balls_to_select : ℕ := 4

-- Define the total number of balls
def total_balls : ℕ := total_red_balls + total_black_balls

-- Define the function to calculate binomial coefficient
def binomial (n k : ℕ) : ℕ := sorry

-- Theorem statement
theorem distinct_number_probability :
  (binomial total_balls balls_to_select : ℚ) ≠ 0 →
  (binomial total_red_balls balls_to_select * 2^balls_to_select : ℚ) /
  (binomial total_balls balls_to_select : ℚ) = 8/21 := by sorry

end distinct_number_probability_l2108_210891


namespace pet_store_transactions_l2108_210884

/-- Represents the number of pets of each type -/
structure PetCounts where
  puppies : Nat
  kittens : Nat
  rabbits : Nat
  guineaPigs : Nat
  chameleons : Nat

/-- Calculates the total number of pets -/
def totalPets (counts : PetCounts) : Nat :=
  counts.puppies + counts.kittens + counts.rabbits + counts.guineaPigs + counts.chameleons

/-- Represents the sales and returns of pets -/
structure Transactions where
  puppiesSold : Nat
  kittensSold : Nat
  rabbitsSold : Nat
  guineaPigsSold : Nat
  chameleonsSold : Nat
  kittensReturned : Nat
  guineaPigsReturned : Nat
  chameleonsReturned : Nat

/-- Calculates the remaining pets after transactions -/
def remainingPets (initial : PetCounts) (trans : Transactions) : PetCounts :=
  { puppies := initial.puppies - trans.puppiesSold,
    kittens := initial.kittens - trans.kittensSold + trans.kittensReturned,
    rabbits := initial.rabbits - trans.rabbitsSold,
    guineaPigs := initial.guineaPigs - trans.guineaPigsSold + trans.guineaPigsReturned,
    chameleons := initial.chameleons - trans.chameleonsSold + trans.chameleonsReturned }

theorem pet_store_transactions (initial : PetCounts) (trans : Transactions) :
  initial.puppies = 7 ∧
  initial.kittens = 6 ∧
  initial.rabbits = 4 ∧
  initial.guineaPigs = 5 ∧
  initial.chameleons = 3 ∧
  trans.puppiesSold = 2 ∧
  trans.kittensSold = 3 ∧
  trans.rabbitsSold = 3 ∧
  trans.guineaPigsSold = 3 ∧
  trans.chameleonsSold = 0 ∧
  trans.kittensReturned = 1 ∧
  trans.guineaPigsReturned = 1 ∧
  trans.chameleonsReturned = 1 →
  totalPets (remainingPets initial trans) = 15 := by
  sorry

end pet_store_transactions_l2108_210884


namespace fraction_value_l2108_210843

theorem fraction_value (m n : ℝ) (h : (m - 8)^2 + |n + 6| = 0) : n / m = -3/4 := by
  sorry

end fraction_value_l2108_210843


namespace g_difference_l2108_210867

-- Define the function g
def g (x : ℝ) : ℝ := 6 * x^2 - 3 * x + 4

-- Theorem statement
theorem g_difference (x h : ℝ) : g (x + h) - g x = h * (12 * x + 6 * h - 3) := by
  sorry

end g_difference_l2108_210867


namespace park_road_perimeter_l2108_210837

/-- Given a square park with a road inside, proves that the perimeter of the outer edge of the road is 600 meters -/
theorem park_road_perimeter (side_length : ℝ) : 
  side_length > 0 →
  side_length^2 - (side_length - 6)^2 = 1764 →
  4 * side_length = 600 := by
sorry

end park_road_perimeter_l2108_210837


namespace line_tangent_to_ellipse_l2108_210881

/-- The line equation y = kx + 2 is tangent to the ellipse 2x^2 + 8y^2 = 8 exactly once if and only if k^2 = 3/4 -/
theorem line_tangent_to_ellipse (k : ℝ) : 
  (∃! x y : ℝ, y = k * x + 2 ∧ 2 * x^2 + 8 * y^2 = 8) ↔ k^2 = 3/4 := by
  sorry

end line_tangent_to_ellipse_l2108_210881


namespace digit_150_of_1_13_l2108_210827

/-- The decimal representation of 1/13 -/
def decimal_rep_1_13 : ℕ → Fin 10
  | n => match n % 6 with
    | 0 => 0
    | 1 => 7
    | 2 => 6
    | 3 => 9
    | 4 => 2
    | 5 => 3
    | _ => 0  -- This case should never occur

theorem digit_150_of_1_13 : decimal_rep_1_13 150 = 0 := by
  sorry

end digit_150_of_1_13_l2108_210827


namespace intersection_point_of_lines_l2108_210866

theorem intersection_point_of_lines (x y : ℚ) :
  (5 * x + 2 * y = 8) ∧ (11 * x - 5 * y = 1) ↔ x = 42/47 ∧ y = 83/47 := by
  sorry

end intersection_point_of_lines_l2108_210866


namespace inequality_solution_l2108_210876

-- Define the function f
def f : ℝ → ℝ := sorry

-- Define the properties of f
axiom f_even : ∀ x, f x = f (-x)
axiom f_increasing : ∀ x y, 0 ≤ x → x ≤ y → f x ≤ f y
axiom f_one_eq_zero : f 1 = 0

-- Define the solution set
def solution_set : Set ℝ := {x | x ≥ 3 ∨ x ≤ 1}

-- State the theorem
theorem inequality_solution :
  {x : ℝ | f (x - 2) ≥ 0} = solution_set := by sorry

end inequality_solution_l2108_210876


namespace percentage_with_diploma_l2108_210830

theorem percentage_with_diploma (total : ℝ) 
  (no_diploma_but_choice : ℝ) 
  (no_choice_but_diploma_ratio : ℝ) 
  (job_of_choice : ℝ) :
  no_diploma_but_choice = 0.1 * total →
  no_choice_but_diploma_ratio = 0.15 →
  job_of_choice = 0.4 * total →
  ∃ (with_diploma : ℝ), 
    with_diploma = 0.39 * total ∧
    with_diploma = (job_of_choice - no_diploma_but_choice) + 
                   (no_choice_but_diploma_ratio * (total - job_of_choice)) :=
by sorry

end percentage_with_diploma_l2108_210830


namespace combined_flock_size_is_300_l2108_210800

/-- Calculates the combined flock size after a given number of years -/
def combinedFlockSize (initialSize birthRate deathRate years additionalFlockSize : ℕ) : ℕ :=
  initialSize + (birthRate - deathRate) * years + additionalFlockSize

/-- Theorem: The combined flock size after 5 years is 300 ducks -/
theorem combined_flock_size_is_300 :
  combinedFlockSize 100 30 20 5 150 = 300 := by
  sorry

#eval combinedFlockSize 100 30 20 5 150

end combined_flock_size_is_300_l2108_210800


namespace easter_egg_hunt_l2108_210859

/-- Represents the number of eggs found by each group or individual -/
structure EggCounts where
  kevin : ℕ
  someChildren : ℕ
  george : ℕ
  cheryl : ℕ

/-- The Easter egg hunt problem -/
theorem easter_egg_hunt (counts : EggCounts) 
  (h1 : counts.kevin = 5)
  (h2 : counts.george = 9)
  (h3 : counts.cheryl = 56)
  (h4 : counts.cheryl = counts.kevin + counts.someChildren + counts.george + 29) :
  counts.someChildren = 13 := by
  sorry

end easter_egg_hunt_l2108_210859


namespace negation_of_universal_statement_l2108_210819

theorem negation_of_universal_statement :
  (¬ ∀ x : ℝ, x ∈ Set.Icc 1 2 → x^2 < 4) ↔ (∃ x : ℝ, x ∈ Set.Icc 1 2 ∧ x^2 ≥ 4) := by
  sorry

end negation_of_universal_statement_l2108_210819


namespace circle_tangent_to_line_l2108_210885

/-- A circle with center (-1, 3) that is tangent to the line x - y = 0 -/
def tangentCircle (x y : ℝ) : Prop :=
  (x + 1)^2 + (y - 3)^2 = 8

/-- The line x - y = 0 -/
def tangentLine (x y : ℝ) : Prop :=
  x - y = 0

/-- The center of the circle -/
def circleCenter : ℝ × ℝ := (-1, 3)

theorem circle_tangent_to_line :
  ∃ (x₀ y₀ : ℝ), tangentCircle x₀ y₀ ∧ tangentLine x₀ y₀ ∧
  ∀ (x y : ℝ), tangentCircle x y ∧ tangentLine x y → (x, y) = (x₀, y₀) :=
sorry

end circle_tangent_to_line_l2108_210885


namespace derivative_of_cosine_linear_l2108_210882

/-- Given a function f(x) = cos(2x - π/6), its derivative f'(x) = -2sin(2x - π/6) --/
theorem derivative_of_cosine_linear (x : ℝ) :
  let f : ℝ → ℝ := λ x ↦ Real.cos (2 * x - π / 6)
  deriv f x = -2 * Real.sin (2 * x - π / 6) := by
  sorry

end derivative_of_cosine_linear_l2108_210882


namespace square_difference_mental_calculation_l2108_210853

theorem square_difference (n : ℕ) : 
  ((n + 1) ^ 2 : ℕ) = n ^ 2 + 2 * n + 1 ∧ 
  ((n - 1) ^ 2 : ℕ) = n ^ 2 - 2 * n + 1 := by
  sorry

theorem mental_calculation : 
  (41 ^ 2 : ℕ) = 40 ^ 2 + 81 ∧ 
  (39 ^ 2 : ℕ) = 40 ^ 2 - 79 := by
  sorry

end square_difference_mental_calculation_l2108_210853


namespace shoppingMallMethodIsSystematic_l2108_210896

/-- Represents a sampling method with a fixed interval and starting point -/
structure SamplingMethod where
  interval : ℕ
  start : ℕ

/-- Defines the characteristics of systematic sampling -/
def isSystematicSampling (method : SamplingMethod) : Prop :=
  method.interval > 0 ∧
  method.start > 0 ∧
  method.start ≤ method.interval

/-- The sampling method used by the shopping mall -/
def shoppingMallMethod : SamplingMethod :=
  { interval := 50,
    start := 15 }

/-- Theorem stating that the shopping mall's method is a systematic sampling method -/
theorem shoppingMallMethodIsSystematic :
  isSystematicSampling shoppingMallMethod := by
  sorry

end shoppingMallMethodIsSystematic_l2108_210896


namespace inequality_and_equality_condition_l2108_210810

theorem inequality_and_equality_condition (a b c : ℝ) 
  (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_pos_c : 0 < c)
  (h_sum : a + b + c = 1) : 
  a / (2 * a + 1) + b / (3 * b + 1) + c / (6 * c + 1) ≤ 1 / 2 ∧
  (a / (2 * a + 1) + b / (3 * b + 1) + c / (6 * c + 1) = 1 / 2 ↔ 
   a = 1 / 2 ∧ b = 1 / 3 ∧ c = 1 / 6) :=
by sorry

end inequality_and_equality_condition_l2108_210810


namespace farmer_brown_chickens_l2108_210815

/-- Given the number of sheep, total legs, and legs per animal, calculate the number of chickens -/
def calculate_chickens (num_sheep : ℕ) (total_legs : ℕ) (chicken_legs : ℕ) (sheep_legs : ℕ) : ℕ :=
  (total_legs - num_sheep * sheep_legs) / chicken_legs

/-- Theorem stating that under the given conditions, the number of chickens is 7 -/
theorem farmer_brown_chickens :
  let num_sheep : ℕ := 5
  let total_legs : ℕ := 34
  let chicken_legs : ℕ := 2
  let sheep_legs : ℕ := 4
  calculate_chickens num_sheep total_legs chicken_legs sheep_legs = 7 := by
  sorry

end farmer_brown_chickens_l2108_210815


namespace rectangle_length_l2108_210877

/-- The length of a rectangle with width 4 cm and area equal to a square with sides 4 cm -/
theorem rectangle_length (width : ℝ) (square_side : ℝ) (length : ℝ) : 
  width = 4 →
  square_side = 4 →
  length * width = square_side * square_side →
  length = 4 := by
  sorry

end rectangle_length_l2108_210877


namespace hypotenuse_of_right_triangle_with_medians_l2108_210886

/-- A right triangle with specific median properties -/
structure RightTriangleWithMedians where
  -- The lengths of the two legs
  a : ℝ
  b : ℝ
  -- The medians from acute angles are both 6
  median_a : a^2 + (b/2)^2 = 36
  median_b : b^2 + (a/2)^2 = 36
  -- Ensure positivity of sides
  a_pos : a > 0
  b_pos : b > 0

/-- The hypotenuse of the right triangle with the given median properties is 2√57.6 -/
theorem hypotenuse_of_right_triangle_with_medians (t : RightTriangleWithMedians) :
  Real.sqrt ((2*t.a)^2 + (2*t.b)^2) = 2 * Real.sqrt 57.6 := by
  sorry

end hypotenuse_of_right_triangle_with_medians_l2108_210886


namespace matrix_transformation_l2108_210808

/-- Given a 2x2 matrix M with eigenvector [1, 1] corresponding to eigenvalue 8,
    prove that the transformation of point (-1, 2) by M results in (-2, 4) -/
theorem matrix_transformation (a b : ℝ) : 
  let M : Matrix (Fin 2) (Fin 2) ℝ := !![a, 2; 4, b]
  (M.mulVec ![1, 1] = ![8, 8]) →
  (M.mulVec ![-1, 2] = ![-2, 4]) := by
sorry

end matrix_transformation_l2108_210808


namespace board_numbers_sum_l2108_210871

theorem board_numbers_sum (a b c : ℝ) : 
  ({a, b, c} : Set ℝ) = {a^2 + 2*b*c, b^2 + 2*c*a, c^2 + 2*a*b} → 
  a + b + c = 0 ∨ a + b + c = 1 := by
sorry

end board_numbers_sum_l2108_210871


namespace combination_sum_equality_l2108_210888

theorem combination_sum_equality (n m k : ℕ) (h1 : 1 ≤ k) (h2 : k < m) (h3 : m ≤ n) :
  (Nat.choose n m) + (Finset.range (k + 1)).sum (λ i => (Nat.choose k i) * (Nat.choose n (m - i))) = Nat.choose (n + k) m := by
  sorry

end combination_sum_equality_l2108_210888


namespace ceiling_sum_of_roots_l2108_210892

theorem ceiling_sum_of_roots : ⌈Real.sqrt 3⌉ + ⌈Real.sqrt 33⌉ + ⌈Real.sqrt 333⌉ = 27 := by
  sorry

end ceiling_sum_of_roots_l2108_210892


namespace biker_journey_time_l2108_210822

/-- Given a biker's journey between two towns, prove the time taken for the first half. -/
theorem biker_journey_time (total_distance : ℝ) (initial_speed : ℝ) (speed_increase : ℝ) (second_half_time : ℝ) :
  total_distance = 140 →
  initial_speed = 14 →
  speed_increase = 2 →
  second_half_time = 7/3 →
  (total_distance / 2) / initial_speed = 5 := by
  sorry

end biker_journey_time_l2108_210822


namespace school_boys_count_l2108_210814

theorem school_boys_count :
  ∀ (total boys girls : ℕ),
  total = 900 →
  boys + girls = total →
  girls * total = boys * boys →
  boys = 810 := by
sorry

end school_boys_count_l2108_210814


namespace blue_box_contains_70_blueberries_l2108_210883

/-- Represents the number of blueberries in each blue box -/
def blueberries : ℕ := sorry

/-- Represents the number of strawberries in each red box -/
def strawberries : ℕ := sorry

/-- The increase in total berries when replacing a blue box with a red box -/
def total_increase : ℕ := 30

/-- The increase in difference between strawberries and blueberries when replacing a blue box with a red box -/
def difference_increase : ℕ := 100

theorem blue_box_contains_70_blueberries :
  (strawberries - blueberries = total_increase) ∧
  (strawberries = difference_increase) →
  blueberries = 70 := by sorry

end blue_box_contains_70_blueberries_l2108_210883


namespace min_sum_given_max_product_l2108_210817

theorem min_sum_given_max_product (a b : ℝ) : 
  a > 0 → b > 0 → (∀ x y : ℝ, a * b * x + y ≤ 8) → a + b ≥ 4 * Real.sqrt 2 := by
  sorry

end min_sum_given_max_product_l2108_210817


namespace modulus_of_2_minus_i_l2108_210828

theorem modulus_of_2_minus_i : 
  let z : ℂ := 2 - I
  Complex.abs z = Real.sqrt 5 := by sorry

end modulus_of_2_minus_i_l2108_210828


namespace unique_root_implies_m_equals_one_l2108_210851

def f (x : ℝ) := 2 * x^2 + x - 4

theorem unique_root_implies_m_equals_one (m n : ℤ) :
  (n = m + 1) →
  (∃! x : ℝ, m < x ∧ x < n ∧ f x = 0) →
  m = 1 :=
by sorry

end unique_root_implies_m_equals_one_l2108_210851


namespace number_exceeding_fraction_l2108_210839

theorem number_exceeding_fraction : ∃ x : ℚ, x = (3/8)*x + 20 ∧ x = 32 := by
  sorry

end number_exceeding_fraction_l2108_210839


namespace mismatched_pairs_count_l2108_210816

/-- Represents a sock with a color and a pattern -/
structure Sock :=
  (color : String)
  (pattern : String)

/-- Represents a pair of socks -/
def SockPair := Sock × Sock

/-- Checks if two socks are mismatched (different color and pattern) -/
def isMismatched (s1 s2 : Sock) : Bool :=
  s1.color ≠ s2.color ∧ s1.pattern ≠ s2.pattern

/-- The set of all sock pairs -/
def allPairs : List SockPair := [
  (⟨"Red", "Striped"⟩, ⟨"Red", "Striped"⟩),
  (⟨"Green", "Polka-dotted"⟩, ⟨"Green", "Polka-dotted"⟩),
  (⟨"Blue", "Checked"⟩, ⟨"Blue", "Checked"⟩),
  (⟨"Yellow", "Floral"⟩, ⟨"Yellow", "Floral"⟩),
  (⟨"Purple", "Plaid"⟩, ⟨"Purple", "Plaid"⟩)
]

/-- Theorem: The number of unique mismatched pairs is 10 -/
theorem mismatched_pairs_count :
  (List.length (List.filter
    (fun (p : Sock × Sock) => isMismatched p.1 p.2)
    (List.join (List.map
      (fun (p1 : SockPair) => List.map
        (fun (p2 : SockPair) => (p1.1, p2.2))
        allPairs)
      allPairs)))) = 10 := by
  sorry

end mismatched_pairs_count_l2108_210816


namespace arithmetic_sequence_sum_l2108_210890

def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) - a n = d

theorem arithmetic_sequence_sum 
  (a : ℕ → ℝ) 
  (h_arithmetic : arithmetic_sequence a) 
  (h_a5 : a 5 = 15) : 
  a 2 + a 4 + a 6 + a 8 = 60 := by
sorry

end arithmetic_sequence_sum_l2108_210890


namespace number_120_more_than_third_l2108_210831

theorem number_120_more_than_third : ∃ x : ℚ, x = (1/3) * x + 120 ∧ x = 180 := by
  sorry

end number_120_more_than_third_l2108_210831


namespace hyperbola_chord_length_l2108_210841

/-- Represents a hyperbola with semi-major axis a and semi-minor axis b -/
structure Hyperbola where
  a : ℝ
  b : ℝ

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a chord of the hyperbola -/
structure Chord where
  A : Point
  B : Point

/-- The distance between two points -/
def distance (p q : Point) : ℝ := sorry

/-- Checks if a point lies on the hyperbola -/
def on_hyperbola (h : Hyperbola) (p : Point) : Prop :=
  (p.x^2 / h.a^2) - (p.y^2 / h.b^2) = 1

/-- The foci of the hyperbola -/
def foci (h : Hyperbola) : (Point × Point) := sorry

theorem hyperbola_chord_length 
  (h : Hyperbola) 
  (c : Chord) 
  (F1 F2 : Point) :
  (foci h = (F1, F2)) →
  (on_hyperbola h c.A) →
  (on_hyperbola h c.B) →
  (distance F1 c.A = 0 ∨ distance F1 c.B = 0) →
  (distance c.A F2 + distance c.B F2 = 2 * distance c.A c.B) →
  distance c.A c.B = 4 * h.a :=
sorry

end hyperbola_chord_length_l2108_210841


namespace quadratic_sum_l2108_210875

/-- A quadratic function f(x) = ax^2 - bx + c passing through (1, -1) with vertex at (-1/2, -1/4) -/
def QuadraticFunction (a b c : ℤ) : ℝ → ℝ := λ x ↦ (a : ℝ) * x^2 - (b : ℝ) * x + (c : ℝ)

theorem quadratic_sum (a b c : ℤ) :
  (QuadraticFunction a b c 1 = -1) →
  (QuadraticFunction a b c (-1/2) = -1/4) →
  a + b + c = -1 := by
  sorry

end quadratic_sum_l2108_210875


namespace exponential_inequality_l2108_210857

theorem exponential_inequality (x₁ x₂ : ℝ) (h₁ : 0 < x₁) (h₂ : x₁ < x₂) (h₃ : x₂ < 1) :
  x₂ * Real.exp x₁ > x₁ * Real.exp x₂ := by
  sorry

end exponential_inequality_l2108_210857


namespace cos_240_degrees_l2108_210821

theorem cos_240_degrees : Real.cos (240 * π / 180) = -1/2 := by sorry

end cos_240_degrees_l2108_210821


namespace complex_fraction_simplification_l2108_210813

theorem complex_fraction_simplification :
  (5 - 3 * Complex.I) / (2 - 3 * Complex.I) = -19/5 - 9/5 * Complex.I :=
by sorry

end complex_fraction_simplification_l2108_210813


namespace quadratic_equation_root_quadratic_equation_rational_coefficients_quadratic_equation_leading_coefficient_l2108_210869

theorem quadratic_equation_root (x : ℝ) : x^2 + 6*x + 4 = 0 ↔ x = Real.sqrt 5 - 3 ∨ x = -Real.sqrt 5 - 3 := by sorry

theorem quadratic_equation_rational_coefficients : ∃ a b c : ℚ, a = 1 ∧ ∀ x : ℝ, x^2 + 6*x + 4 = a*x^2 + b*x + c := by sorry

theorem quadratic_equation_leading_coefficient : ∃ a b c : ℝ, a = 1 ∧ ∀ x : ℝ, x^2 + 6*x + 4 = a*x^2 + b*x + c := by sorry

end quadratic_equation_root_quadratic_equation_rational_coefficients_quadratic_equation_leading_coefficient_l2108_210869


namespace hotel_expenditure_l2108_210895

theorem hotel_expenditure (total_expenditure : ℕ) 
  (standard_spenders : ℕ) (standard_amount : ℕ) (extra_amount : ℕ) : 
  total_expenditure = 117 →
  standard_spenders = 8 →
  standard_amount = 12 →
  extra_amount = 8 →
  ∃ (n : ℕ), n = 9 ∧ 
    (standard_spenders * standard_amount + 
    (total_expenditure / n + extra_amount) = total_expenditure) :=
by sorry

end hotel_expenditure_l2108_210895


namespace initial_water_was_six_cups_l2108_210834

/-- Represents the water consumption during a hike --/
structure HikeWaterConsumption where
  total_distance : ℝ
  total_time : ℝ
  remaining_water : ℝ
  leak_rate : ℝ
  last_mile_consumption : ℝ
  first_three_miles_rate : ℝ

/-- Calculates the initial amount of water in the canteen --/
def initial_water (h : HikeWaterConsumption) : ℝ :=
  h.remaining_water + h.leak_rate * h.total_time + 
  h.last_mile_consumption + h.first_three_miles_rate * (h.total_distance - 1)

/-- Theorem stating that the initial amount of water in the canteen was 6 cups --/
theorem initial_water_was_six_cups (h : HikeWaterConsumption) 
  (h_distance : h.total_distance = 4)
  (h_time : h.total_time = 2)
  (h_remaining : h.remaining_water = 1)
  (h_leak : h.leak_rate = 1)
  (h_last_mile : h.last_mile_consumption = 1)
  (h_first_three : h.first_three_miles_rate = 2/3) :
  initial_water h = 6 := by
  sorry


end initial_water_was_six_cups_l2108_210834


namespace carlsons_original_land_size_l2108_210863

/-- Calculates the size of Carlson's original land given the cost and area of new land purchases --/
theorem carlsons_original_land_size
  (cost_land1 : ℝ)
  (cost_land2 : ℝ)
  (cost_per_sqm : ℝ)
  (total_area_after : ℝ)
  (h1 : cost_land1 = 8000)
  (h2 : cost_land2 = 4000)
  (h3 : cost_per_sqm = 20)
  (h4 : total_area_after = 900) :
  total_area_after - (cost_land1 + cost_land2) / cost_per_sqm = 300 :=
by sorry

end carlsons_original_land_size_l2108_210863


namespace certain_number_problem_l2108_210833

theorem certain_number_problem : 
  ∃ x : ℝ, 0.60 * x = 0.30 * 30 + 21 ∧ x = 50 := by
  sorry

end certain_number_problem_l2108_210833


namespace cube_sum_theorem_l2108_210899

theorem cube_sum_theorem (a b c : ℝ) (h_distinct : a ≠ b ∧ b ≠ c ∧ a ≠ c) 
  (h_equality : (a^3 + 8) / a = (b^3 + 8) / b ∧ (b^3 + 8) / b = (c^3 + 8) / c) : 
  a^3 + b^3 + c^3 = -24 := by
  sorry

end cube_sum_theorem_l2108_210899


namespace cookie_sale_total_l2108_210854

/-- Represents the number of cookies sold for each type -/
structure CookieSales where
  raisin : ℕ
  oatmeal : ℕ
  chocolate_chip : ℕ
  peanut_butter : ℕ

/-- Defines the conditions of the cookie sale -/
def cookie_sale_conditions (sales : CookieSales) : Prop :=
  sales.raisin = 42 ∧
  sales.raisin = 6 * sales.oatmeal ∧
  6 * sales.oatmeal = sales.oatmeal + 3 * sales.oatmeal + 2 * sales.oatmeal

theorem cookie_sale_total (sales : CookieSales) :
  cookie_sale_conditions sales →
  sales.raisin + sales.oatmeal + sales.chocolate_chip + sales.peanut_butter = 84 := by
  sorry

end cookie_sale_total_l2108_210854


namespace quadratic_roots_property_l2108_210864

theorem quadratic_roots_property (x₁ x₂ : ℝ) : 
  (x₁^2 - 2*x₁ - 1 = 0) → (x₂^2 - 2*x₂ - 1 = 0) → (x₁ + x₂ - x₁*x₂ = 3) := by
  sorry

end quadratic_roots_property_l2108_210864


namespace square_difference_theorem_l2108_210805

theorem square_difference_theorem (N : ℕ+) : 
  (∃ x : ℤ, 2^(N : ℕ) - 2 * (N : ℤ) = x^2) ↔ N = 1 ∨ N = 2 :=
sorry

end square_difference_theorem_l2108_210805


namespace fox_jeans_price_l2108_210894

/-- The regular price of Fox jeans -/
def F : ℝ := 15

/-- The regular price of Pony jeans -/
def P : ℝ := 18

/-- The discount rate for Fox jeans -/
def discount_rate_fox : ℝ := 0.08

/-- The discount rate for Pony jeans -/
def discount_rate_pony : ℝ := 0.14

/-- The total savings on 5 pairs of jeans (3 Fox, 2 Pony) -/
def total_savings : ℝ := 8.64

theorem fox_jeans_price :
  F = 15 ∧
  P = 18 ∧
  discount_rate_fox + discount_rate_pony = 0.22 ∧
  3 * (F * discount_rate_fox) + 2 * (P * discount_rate_pony) = total_savings :=
by sorry

end fox_jeans_price_l2108_210894


namespace tangent_line_triangle_area_l2108_210856

/-- The area of the triangle formed by the tangent line to y = x^3 at (3, 27) and the axes is 54 -/
theorem tangent_line_triangle_area : 
  let f : ℝ → ℝ := fun x ↦ x^3
  let point : ℝ × ℝ := (3, 27)
  let tangent_line : ℝ → ℝ := fun x ↦ 27 * x - 54
  let triangle_area := 
    let x_intercept := (tangent_line 0) / (-27)
    let y_intercept := tangent_line 0
    (1/2) * x_intercept * (-y_intercept)
  triangle_area = 54 := by
  sorry

end tangent_line_triangle_area_l2108_210856


namespace quadratic_sum_equals_28_l2108_210852

theorem quadratic_sum_equals_28 (a b c : ℝ) 
  (h1 : a - b = 4) 
  (h2 : b + c = 2) : 
  a^2 + b^2 + c^2 - a*b + b*c + c*a = 28 := by
  sorry

end quadratic_sum_equals_28_l2108_210852


namespace distance_to_big_rock_l2108_210844

/-- The distance to Big Rock given the rower's speed, river current, and round trip time -/
theorem distance_to_big_rock 
  (rower_speed : ℝ) 
  (river_current : ℝ) 
  (round_trip_time : ℝ) 
  (h1 : rower_speed = 7) 
  (h2 : river_current = 1) 
  (h3 : round_trip_time = 1) : 
  ∃ (distance : ℝ), distance = 24 / 7 := by
  sorry

end distance_to_big_rock_l2108_210844


namespace arithmetic_sequence_sum_l2108_210873

def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_sum (a : ℕ → ℝ) :
  arithmetic_sequence a →
  (a 3 + a 4 + a 5 + a 6 + a 7 = 450) →
  (a 2 + a 8 = 180) := by
  sorry

end arithmetic_sequence_sum_l2108_210873


namespace chessboard_game_outcomes_l2108_210848

/-- Represents the outcome of the game -/
inductive GameOutcome
  | FirstPlayerWins
  | SecondPlayerWins

/-- Represents the starting position of the piece -/
inductive StartPosition
  | Corner
  | AdjacentToCorner

/-- Defines the game on an n × n chessboard -/
def chessboardGame (n : ℕ) (startPos : StartPosition) : GameOutcome :=
  match n, startPos with
  | n, StartPosition.Corner =>
      if n % 2 = 0 then
        GameOutcome.FirstPlayerWins
      else
        GameOutcome.SecondPlayerWins
  | _, StartPosition.AdjacentToCorner => GameOutcome.FirstPlayerWins

/-- Theorem stating the game outcomes -/
theorem chessboard_game_outcomes :
  (∀ n : ℕ, n > 1 →
    (n % 2 = 0 → chessboardGame n StartPosition.Corner = GameOutcome.FirstPlayerWins) ∧
    (n % 2 = 1 → chessboardGame n StartPosition.Corner = GameOutcome.SecondPlayerWins)) ∧
  (∀ n : ℕ, n > 1 → chessboardGame n StartPosition.AdjacentToCorner = GameOutcome.FirstPlayerWins) :=
sorry

end chessboard_game_outcomes_l2108_210848


namespace min_value_theorem_l2108_210850

theorem min_value_theorem (a b : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : a + 2*b = 1) :
  (2/a + 3/b) ≥ 8 + 4 * Real.sqrt 3 := by
  sorry

end min_value_theorem_l2108_210850


namespace little_red_height_calculation_l2108_210812

/-- Little Ming's height in meters -/
def little_ming_height : ℝ := 1.3

/-- The difference in height between Little Ming and Little Red in meters -/
def height_difference : ℝ := 0.2

/-- Little Red's height in meters -/
def little_red_height : ℝ := little_ming_height - height_difference

theorem little_red_height_calculation :
  little_red_height = 1.1 := by sorry

end little_red_height_calculation_l2108_210812


namespace exists_abc_for_all_n_l2108_210840

def interval (k : ℕ) := Set.Ioo (k^2 : ℝ) (k^2 + k + 3 * Real.sqrt 3)

theorem exists_abc_for_all_n :
  ∀ (n : ℕ), ∃ (a b c : ℝ),
    (∃ (k₁ : ℕ), a ∈ interval k₁) ∧
    (∃ (k₂ : ℕ), b ∈ interval k₂) ∧
    (∃ (k₃ : ℕ), c ∈ interval k₃) ∧
    (n : ℝ) = a * b / c :=
by
  sorry


end exists_abc_for_all_n_l2108_210840


namespace mod_sixteen_equivalence_l2108_210824

theorem mod_sixteen_equivalence : ∃! m : ℤ, 0 ≤ m ∧ m ≤ 15 ∧ m ≡ 12345 [ZMOD 16] ∧ m = 9 := by
  sorry

end mod_sixteen_equivalence_l2108_210824


namespace skateboard_distance_l2108_210898

/-- Calculates the sum of an arithmetic sequence -/
def arithmeticSequenceSum (a₁ : ℕ) (d : ℕ) (n : ℕ) : ℕ :=
  n * (2 * a₁ + (n - 1) * d) / 2

/-- The skateboard problem -/
theorem skateboard_distance : arithmeticSequenceSum 8 9 20 = 1870 := by
  sorry

end skateboard_distance_l2108_210898


namespace min_value_reciprocal_sum_l2108_210878

theorem min_value_reciprocal_sum (x : ℝ) (h1 : 0 < x) (h2 : x < 3) :
  (1 / x) + (1 / (3 - x)) ≥ 4 / 3 ∧
  ((1 / x) + (1 / (3 - x)) = 4 / 3 ↔ x = 3 / 2) :=
by sorry

end min_value_reciprocal_sum_l2108_210878


namespace perpendicular_necessary_not_sufficient_l2108_210806

-- Define the vectors a and b as functions of x
def a (x : ℝ) : ℝ × ℝ := (x - 1, x)
def b (x : ℝ) : ℝ × ℝ := (x + 2, x - 4)

-- Define the perpendicularity condition
def perpendicular (x : ℝ) : Prop :=
  (a x).1 * (b x).1 + (a x).2 * (b x).2 = 0

-- Theorem statement
theorem perpendicular_necessary_not_sufficient :
  (∀ x : ℝ, x = 2 → perpendicular x) ∧
  (∃ x : ℝ, perpendicular x ∧ x ≠ 2) :=
sorry

end perpendicular_necessary_not_sufficient_l2108_210806


namespace cake_pieces_count_l2108_210829

/-- Given 50 friends and 3 pieces of cake per friend, prove that the total number of cake pieces is 150. -/
theorem cake_pieces_count (num_friends : ℕ) (pieces_per_friend : ℕ) : 
  num_friends = 50 → pieces_per_friend = 3 → num_friends * pieces_per_friend = 150 := by
  sorry


end cake_pieces_count_l2108_210829


namespace gcf_of_lcms_l2108_210803

-- Define the GCF (Greatest Common Factor) function
def GCF (a b : ℕ) : ℕ := Nat.gcd a b

-- Define the LCM (Least Common Multiple) function
def LCM (c d : ℕ) : ℕ := Nat.lcm c d

-- Theorem statement
theorem gcf_of_lcms : GCF (LCM 15 21) (LCM 10 14) = 35 := by
  sorry

end gcf_of_lcms_l2108_210803


namespace tenth_term_value_l2108_210865

/-- An arithmetic sequence with specific properties -/
structure ArithmeticSequence where
  a : ℕ → ℝ
  is_arithmetic : ∀ n : ℕ, a (n + 2) - a (n + 1) = a (n + 1) - a n
  sum_property : a 1 + a 3 + a 5 = 9
  product_property : a 3 * (a 4)^2 = 27

/-- The 10th term of the arithmetic sequence is either -39 or 30 -/
theorem tenth_term_value (seq : ArithmeticSequence) : seq.a 10 = -39 ∨ seq.a 10 = 30 := by
  sorry

end tenth_term_value_l2108_210865


namespace discount_rate_for_given_profit_l2108_210801

/-- Given a product with cost price, marked price, and desired profit percentage,
    calculate the discount rate needed to achieve the desired profit. -/
def calculate_discount_rate (cost_price marked_price profit_percentage : ℚ) : ℚ :=
  let selling_price := cost_price * (1 + profit_percentage / 100)
  selling_price / marked_price

theorem discount_rate_for_given_profit :
  let cost_price : ℚ := 200
  let marked_price : ℚ := 300
  let profit_percentage : ℚ := 20
  calculate_discount_rate cost_price marked_price profit_percentage = 4/5 := by
  sorry

end discount_rate_for_given_profit_l2108_210801


namespace x_value_l2108_210868

/-- The value of x is equal to (47% of 1442 - 36% of 1412) + 65 -/
theorem x_value : 
  (0.47 * 1442 - 0.36 * 1412) + 65 = 234.42 := by
  sorry

end x_value_l2108_210868


namespace anna_score_l2108_210897

/-- Calculates the score in a modified contest given the number of correct, incorrect, and unanswered questions -/
def contest_score (correct incorrect unanswered : ℕ) : ℚ :=
  (correct : ℚ) + 0 * (incorrect : ℚ) - 0.5 * (unanswered : ℚ)

theorem anna_score :
  let total_questions : ℕ := 30
  let correct_answers : ℕ := 17
  let incorrect_answers : ℕ := 6
  let unanswered_questions : ℕ := 7
  correct_answers + incorrect_answers + unanswered_questions = total_questions →
  contest_score correct_answers incorrect_answers unanswered_questions = 13.5 := by
  sorry

end anna_score_l2108_210897


namespace guaranteed_pairs_l2108_210807

/-- A color of a candy -/
inductive Color
| Black
| White

/-- A position in the 7x7 grid -/
structure Position where
  x : Fin 7
  y : Fin 7

/-- A configuration of the candy box -/
def Configuration := Position → Color

/-- Two positions are adjacent if they are side-by-side or diagonal -/
def adjacent (p1 p2 : Position) : Prop :=
  (p1.x = p2.x ∧ p1.y.val + 1 = p2.y.val) ∨
  (p1.x = p2.x ∧ p1.y.val = p2.y.val + 1) ∨
  (p1.x.val + 1 = p2.x.val ∧ p1.y = p2.y) ∨
  (p1.x.val = p2.x.val + 1 ∧ p1.y = p2.y) ∨
  (p1.x.val + 1 = p2.x.val ∧ p1.y.val + 1 = p2.y.val) ∨
  (p1.x.val + 1 = p2.x.val ∧ p1.y.val = p2.y.val + 1) ∨
  (p1.x.val = p2.x.val + 1 ∧ p1.y.val + 1 = p2.y.val) ∨
  (p1.x.val = p2.x.val + 1 ∧ p1.y.val = p2.y.val + 1)

/-- A pair of adjacent positions with the same color -/
structure ColoredPair (config : Configuration) where
  p1 : Position
  p2 : Position
  adj : adjacent p1 p2
  same_color : config p1 = config p2

/-- The main theorem: there always exists a set of at least 16 pairs of adjacent cells with the same color -/
theorem guaranteed_pairs (config : Configuration) : 
  ∃ (pairs : Finset (ColoredPair config)), pairs.card ≥ 16 := by
  sorry

end guaranteed_pairs_l2108_210807


namespace approximate_solution_exists_l2108_210847

def f (x : ℝ) := 2 * x^3 + 3 * x - 3

theorem approximate_solution_exists :
  (f 0.625 < 0) →
  (f 0.75 > 0) →
  (f 0.6875 < 0) →
  ∃ x : ℝ, x ∈ Set.Icc 0.6 0.8 ∧ f x = 0 :=
by
  sorry

end approximate_solution_exists_l2108_210847


namespace chess_tournament_games_l2108_210858

theorem chess_tournament_games (n : ℕ) (h : n = 16) : 
  (n * (n - 1)) / 2 = 120 := by
  sorry

end chess_tournament_games_l2108_210858


namespace condition_necessary_not_sufficient_l2108_210835

def is_geometric_sequence_with_ratio_2 (a : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, n ≥ 1 → a (n + 1) = 2 * a n

def condition (a : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, n ≥ 2 → a n = 2 * a (n - 1)

theorem condition_necessary_not_sufficient :
  (∀ a : ℕ → ℝ, is_geometric_sequence_with_ratio_2 a → condition a) ∧
  (∃ a : ℕ → ℝ, condition a ∧ ¬is_geometric_sequence_with_ratio_2 a) :=
sorry

end condition_necessary_not_sufficient_l2108_210835


namespace f_2009_is_zero_l2108_210845

def is_even_function (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = f x

def is_odd_function (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

theorem f_2009_is_zero (f : ℝ → ℝ) 
  (h1 : is_even_function f) 
  (h2 : is_odd_function (fun x ↦ f (x - 1))) : 
  f 2009 = 0 := by
  sorry

end f_2009_is_zero_l2108_210845


namespace product_of_real_parts_l2108_210893

theorem product_of_real_parts (x₁ x₂ : ℂ) : 
  x₁^2 - 4*x₁ = -1 - 3*I → 
  x₂^2 - 4*x₂ = -1 - 3*I → 
  x₁ ≠ x₂ → 
  (x₁.re * x₂.re : ℝ) = (8 - Real.sqrt 6 + Real.sqrt 3) / 2 := by
  sorry

end product_of_real_parts_l2108_210893


namespace second_volume_pages_l2108_210818

/-- Calculates the number of digits used to number pages up to n --/
def digits_used (n : ℕ) : ℕ :=
  if n < 10 then n
  else if n < 100 then 9 + (n - 9) * 2
  else 189 + (n - 99) * 3

/-- Represents the properties of the two volumes --/
structure TwoVolumes :=
  (first : ℕ)
  (second : ℕ)
  (total_digits : ℕ)
  (page_difference : ℕ)

/-- The main theorem about the number of pages in the second volume --/
theorem second_volume_pages (v : TwoVolumes) 
  (h1 : v.total_digits = 888)
  (h2 : v.second = v.first + v.page_difference)
  (h3 : v.page_difference = 8)
  (h4 : digits_used v.first + digits_used v.second = v.total_digits) :
  v.second = 170 := by
  sorry

#check second_volume_pages

end second_volume_pages_l2108_210818


namespace max_trailing_zeros_1003_l2108_210862

/-- Three natural numbers whose sum is 1003 -/
def SumTo1003 (a b c : ℕ) : Prop := a + b + c = 1003

/-- The number of trailing zeros in a natural number -/
def TrailingZeros (n : ℕ) : ℕ := sorry

/-- The product of three natural numbers -/
def ProductOfThree (a b c : ℕ) : ℕ := a * b * c

/-- Theorem stating that the maximum number of trailing zeros in the product of three natural numbers summing to 1003 is 7 -/
theorem max_trailing_zeros_1003 :
  ∀ a b c : ℕ, SumTo1003 a b c →
  ∀ n : ℕ, n = TrailingZeros (ProductOfThree a b c) →
  n ≤ 7 ∧ ∃ x y z : ℕ, SumTo1003 x y z ∧ TrailingZeros (ProductOfThree x y z) = 7 :=
sorry

end max_trailing_zeros_1003_l2108_210862


namespace polygon_interior_angles_sum_l2108_210825

theorem polygon_interior_angles_sum (n : ℕ) (h : n = 9) :
  (n - 2) * 180 = 1260 := by
  sorry

end polygon_interior_angles_sum_l2108_210825


namespace fruit_salad_price_l2108_210889

/-- Represents the cost of the picnic basket items -/
structure PicnicBasket where
  numPeople : Nat
  sandwichPrice : Nat
  sodaPrice : Nat
  snackPrice : Nat
  numSnacks : Nat
  totalCost : Nat

/-- Calculates the cost of fruit salads given the picnic basket information -/
def fruitSaladCost (basket : PicnicBasket) : Nat :=
  basket.totalCost - 
  (basket.numPeople * basket.sandwichPrice + 
   2 * basket.numPeople * basket.sodaPrice + 
   basket.numSnacks * basket.snackPrice)

/-- Theorem stating that the cost of each fruit salad is $3 -/
theorem fruit_salad_price (basket : PicnicBasket) 
  (h1 : basket.numPeople = 4)
  (h2 : basket.sandwichPrice = 5)
  (h3 : basket.sodaPrice = 2)
  (h4 : basket.snackPrice = 4)
  (h5 : basket.numSnacks = 3)
  (h6 : basket.totalCost = 60) :
  fruitSaladCost basket / basket.numPeople = 3 := by
  sorry

end fruit_salad_price_l2108_210889


namespace most_trailing_zeros_l2108_210809

-- Define a function to count trailing zeros
def countTrailingZeros (n : ℕ) : ℕ := sorry

-- Define the arithmetic expressions
def expr1 : ℕ := 300 + 60
def expr2 : ℕ := 22 * 5
def expr3 : ℕ := 25 * 4
def expr4 : ℕ := 400 / 8

-- Theorem statement
theorem most_trailing_zeros :
  countTrailingZeros expr3 ≥ countTrailingZeros expr1 ∧
  countTrailingZeros expr3 ≥ countTrailingZeros expr2 ∧
  countTrailingZeros expr3 ≥ countTrailingZeros expr4 :=
by sorry

end most_trailing_zeros_l2108_210809
