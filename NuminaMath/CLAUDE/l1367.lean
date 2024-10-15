import Mathlib

namespace NUMINAMATH_CALUDE_line_vector_to_slope_intercept_l1367_136772

/-- Given a line in vector form, prove its equivalence to slope-intercept form -/
theorem line_vector_to_slope_intercept :
  let vector_line : ℝ × ℝ → Prop :=
    λ p => (3 : ℝ) * (p.1 + 2) + (7 : ℝ) * (p.2 - 8) = 0
  let slope_intercept_line : ℝ × ℝ → Prop :=
    λ p => p.2 = (-3/7 : ℝ) * p.1 + 50/7
  ∀ p : ℝ × ℝ, vector_line p ↔ slope_intercept_line p :=
by sorry

end NUMINAMATH_CALUDE_line_vector_to_slope_intercept_l1367_136772


namespace NUMINAMATH_CALUDE_exists_angle_leq_90_degrees_l1367_136793

-- Define a type for rays in space
def Ray : Type := ℝ → ℝ × ℝ × ℝ

-- Define a function to calculate the angle between two rays
def angle_between_rays (r1 r2 : Ray) : ℝ := sorry

-- State the theorem
theorem exists_angle_leq_90_degrees (rays : Fin 5 → Ray) 
  (h_distinct : ∀ i j, i ≠ j → rays i ≠ rays j) : 
  ∃ i j, i ≠ j ∧ angle_between_rays (rays i) (rays j) ≤ 90 := by sorry

end NUMINAMATH_CALUDE_exists_angle_leq_90_degrees_l1367_136793


namespace NUMINAMATH_CALUDE_sqrt_pattern_l1367_136727

theorem sqrt_pattern (n : ℕ) (h : n ≥ 1) : 
  Real.sqrt (n + 1 / (n + 2)) = (n + 1) * Real.sqrt (1 / (n + 2)) := by
  sorry

end NUMINAMATH_CALUDE_sqrt_pattern_l1367_136727


namespace NUMINAMATH_CALUDE_cafeteria_students_l1367_136769

theorem cafeteria_students (total : ℕ) (no_lunch : ℕ) (cafeteria : ℕ) : 
  total = 60 → 
  no_lunch = 20 → 
  total = cafeteria + 3 * cafeteria + no_lunch → 
  cafeteria = 10 := by
sorry

end NUMINAMATH_CALUDE_cafeteria_students_l1367_136769


namespace NUMINAMATH_CALUDE_linear_system_solution_l1367_136741

theorem linear_system_solution (a₁ a₂ a₃ a₄ b₁ b₂ b₃ b₄ : ℝ) 
  (eq1 : a₁ * b₁ + a₂ * b₃ = 1)
  (eq2 : a₁ * b₂ + a₂ * b₄ = 0)
  (eq3 : a₃ * b₁ + a₄ * b₃ = 0)
  (eq4 : a₃ * b₂ + a₄ * b₄ = 1)
  (given : a₂ * b₃ = 7) :
  a₄ * b₄ = -6 := by
  sorry

end NUMINAMATH_CALUDE_linear_system_solution_l1367_136741


namespace NUMINAMATH_CALUDE_slope_angle_of_intersecting_line_l1367_136789

/-- The slope angle of a line intersecting a circle -/
theorem slope_angle_of_intersecting_line (α : Real) : 
  (∃ (A B : ℝ × ℝ), 
    (∀ t : ℝ, (1 + t * Real.cos α, t * Real.sin α) ∈ {(x, y) : ℝ × ℝ | (x - 2)^2 + y^2 = 4}) →
    A ∈ {(x, y) : ℝ × ℝ | (x - 2)^2 + y^2 = 4} →
    B ∈ {(x, y) : ℝ × ℝ | (x - 2)^2 + y^2 = 4} →
    Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) = Real.sqrt 14) →
  α = π/4 ∨ α = 3*π/4 := by
sorry

end NUMINAMATH_CALUDE_slope_angle_of_intersecting_line_l1367_136789


namespace NUMINAMATH_CALUDE_leifeng_pagoda_height_l1367_136738

/-- The height of the Leifeng Pagoda problem -/
theorem leifeng_pagoda_height 
  (AC : ℝ) 
  (α β : ℝ) 
  (h1 : AC = 62 * Real.sqrt 2)
  (h2 : α = 45 * π / 180)
  (h3 : β = 15 * π / 180) :
  ∃ BC : ℝ, BC = 62 :=
sorry

end NUMINAMATH_CALUDE_leifeng_pagoda_height_l1367_136738


namespace NUMINAMATH_CALUDE_sunday_cost_theorem_l1367_136709

-- Define the constants
def weekday_discount : ℝ := 0.1
def weekend_increase : ℝ := 0.5
def shaving_cost : ℝ := 10
def styling_cost : ℝ := 15
def monday_total : ℝ := 18

-- Define the theorem
theorem sunday_cost_theorem :
  let weekday_haircut_cost := (monday_total - shaving_cost) / (1 - weekday_discount)
  let weekend_haircut_cost := weekday_haircut_cost * (1 + weekend_increase)
  let sunday_total := weekend_haircut_cost + styling_cost
  ∃ (ε : ℝ), ε > 0 ∧ ε < 0.005 ∧ |sunday_total - 28.34| < ε :=
sorry

end NUMINAMATH_CALUDE_sunday_cost_theorem_l1367_136709


namespace NUMINAMATH_CALUDE_contrapositive_truth_l1367_136735

theorem contrapositive_truth (p q : Prop) : 
  (q → p) → (¬p → ¬q) := by sorry

end NUMINAMATH_CALUDE_contrapositive_truth_l1367_136735


namespace NUMINAMATH_CALUDE_barbara_paper_problem_l1367_136778

/-- The number of sheets in a bundle -/
def sheets_per_bundle : ℕ := 2

/-- The number of sheets in a heap -/
def sheets_per_heap : ℕ := 20

/-- The number of bundles Barbara found -/
def num_bundles : ℕ := 3

/-- The number of bunches Barbara found -/
def num_bunches : ℕ := 2

/-- The number of heaps Barbara found -/
def num_heaps : ℕ := 5

/-- The total number of sheets Barbara removed -/
def total_sheets : ℕ := 114

/-- The number of sheets in a bunch -/
def sheets_per_bunch : ℕ := 4

theorem barbara_paper_problem :
  sheets_per_bunch * num_bunches + sheets_per_bundle * num_bundles + sheets_per_heap * num_heaps = total_sheets :=
by sorry

end NUMINAMATH_CALUDE_barbara_paper_problem_l1367_136778


namespace NUMINAMATH_CALUDE_f_composition_eq_one_fourth_l1367_136743

noncomputable def f (x : ℝ) : ℝ :=
  if x > 0 then Real.log x / Real.log 3
  else 2^x

theorem f_composition_eq_one_fourth :
  f (f (1/9)) = 1/4 := by sorry

end NUMINAMATH_CALUDE_f_composition_eq_one_fourth_l1367_136743


namespace NUMINAMATH_CALUDE_bug_pentagon_probability_l1367_136751

/-- Probability of the bug being at the starting vertex after n moves -/
def Q (n : ℕ) : ℚ :=
  match n with
  | 0 => 1
  | 1 => 0
  | 2 => 1/2
  | n+1 => 1/2 * (1 - Q n)

/-- The probability of returning to the starting vertex on the 12th move in a regular pentagon -/
theorem bug_pentagon_probability : Q 12 = 341/1024 := by
  sorry

end NUMINAMATH_CALUDE_bug_pentagon_probability_l1367_136751


namespace NUMINAMATH_CALUDE_cone_base_circumference_l1367_136782

/-- The circumference of the base of a right circular cone with given volume and height -/
theorem cone_base_circumference (V : ℝ) (h : ℝ) (π : ℝ) :
  V = 36 * π →
  h = 3 →
  π > 0 →
  (2 * π * (3 * V / (π * h))^(1/2) : ℝ) = 12 * π := by sorry

end NUMINAMATH_CALUDE_cone_base_circumference_l1367_136782


namespace NUMINAMATH_CALUDE_rectangle_shorter_side_l1367_136795

theorem rectangle_shorter_side
  (area : ℝ)
  (perimeter : ℝ)
  (h_area : area = 117)
  (h_perimeter : perimeter = 44)
  : ∃ (short_side long_side : ℝ),
    short_side * long_side = area ∧
    2 * (short_side + long_side) = perimeter ∧
    short_side = 9 ∧
    short_side ≤ long_side :=
by
  sorry

end NUMINAMATH_CALUDE_rectangle_shorter_side_l1367_136795


namespace NUMINAMATH_CALUDE_direction_vector_form_l1367_136742

/-- Given a line passing through two points, prove that its direction vector
    has a specific form. -/
theorem direction_vector_form (p1 p2 : ℝ × ℝ) (c : ℝ) : 
  p1 = (-6, 1) →
  p2 = (-1, 5) →
  (p2.1 - p1.1, p2.2 - p1.2) = (5, c) →
  c = 4 := by
  sorry

end NUMINAMATH_CALUDE_direction_vector_form_l1367_136742


namespace NUMINAMATH_CALUDE_fraction_evaluation_l1367_136786

theorem fraction_evaluation : (1 - 1/4) / (1 - 1/3) = 9/8 := by sorry

end NUMINAMATH_CALUDE_fraction_evaluation_l1367_136786


namespace NUMINAMATH_CALUDE_paul_running_time_l1367_136739

/-- Given that Paul watches movies while running on a treadmill, prove that it takes him 12 minutes to run one mile. -/
theorem paul_running_time (num_movies : ℕ) (avg_movie_length : ℝ) (total_miles : ℝ) :
  num_movies = 2 →
  avg_movie_length = 1.5 →
  total_miles = 15 →
  (num_movies * avg_movie_length * 60) / total_miles = 12 := by
  sorry

end NUMINAMATH_CALUDE_paul_running_time_l1367_136739


namespace NUMINAMATH_CALUDE_hyperbola_condition_l1367_136759

-- Define the equation
def is_hyperbola (k : ℝ) : Prop :=
  ∃ (x y : ℝ), x^2 / (k - 1) - y^2 / (k + 2) = 1

-- Define the condition
def condition (k : ℝ) : Prop := 0 < k ∧ k < 1

-- Theorem statement
theorem hyperbola_condition :
  ¬(∀ k : ℝ, is_hyperbola k ↔ condition k) :=
sorry

end NUMINAMATH_CALUDE_hyperbola_condition_l1367_136759


namespace NUMINAMATH_CALUDE_winning_probability_is_five_eighths_l1367_136734

/-- Represents the color of a ball in the lottery bag -/
inductive BallColor
  | Red
  | Yellow
  | White
  | Black

/-- Represents the lottery bag -/
structure LotteryBag where
  total_balls : ℕ
  red_balls : ℕ
  yellow_balls : ℕ
  black_balls : ℕ
  white_balls : ℕ
  h_total : total_balls = red_balls + yellow_balls + black_balls + white_balls

/-- Calculates the probability of winning in the lottery -/
def winning_probability (bag : LotteryBag) : ℚ :=
  (bag.red_balls + bag.yellow_balls + bag.white_balls : ℚ) / bag.total_balls

/-- The lottery bag configuration -/
def lottery_bag : LotteryBag := {
  total_balls := 24
  red_balls := 3
  yellow_balls := 6
  black_balls := 9
  white_balls := 6
  h_total := by rfl
}

/-- Theorem: The probability of winning in the given lottery bag is 5/8 -/
theorem winning_probability_is_five_eighths :
  winning_probability lottery_bag = 5/8 := by
  sorry

end NUMINAMATH_CALUDE_winning_probability_is_five_eighths_l1367_136734


namespace NUMINAMATH_CALUDE_courtyard_width_is_14_l1367_136790

/-- Represents the dimensions of a paving stone -/
structure PavingStone where
  length : ℝ
  width : ℝ

/-- Represents the dimensions of a rectangular courtyard -/
structure Courtyard where
  length : ℝ
  width : ℝ

/-- Calculates the area of a rectangular shape -/
def area (length width : ℝ) : ℝ := length * width

/-- Theorem: The width of the courtyard is 14 meters -/
theorem courtyard_width_is_14 (stone : PavingStone) (yard : Courtyard) 
    (h1 : stone.length = 3)
    (h2 : stone.width = 2)
    (h3 : yard.length = 60)
    (h4 : area yard.length yard.width = 140 * area stone.length stone.width) :
  yard.width = 14 := by
  sorry

#check courtyard_width_is_14

end NUMINAMATH_CALUDE_courtyard_width_is_14_l1367_136790


namespace NUMINAMATH_CALUDE_ellipse_equation_l1367_136707

/-- The standard equation of an ellipse with given eccentricity and major axis length -/
theorem ellipse_equation (e : ℝ) (major_axis : ℝ) :
  e = 2/3 →
  major_axis = 6 →
  ∃ (a b : ℝ),
    a = major_axis / 2 ∧
    b^2 = a^2 * (1 - e^2) ∧
    ((∀ x y : ℝ, x^2/a^2 + y^2/b^2 = 1) ∨
     (∀ x y : ℝ, x^2/b^2 + y^2/a^2 = 1)) :=
by sorry

end NUMINAMATH_CALUDE_ellipse_equation_l1367_136707


namespace NUMINAMATH_CALUDE_age_problem_l1367_136783

theorem age_problem (a b c : ℕ) : 
  a = b + 2 → 
  b = 2 * c → 
  a + b + c = 12 → 
  b = 4 := by
sorry

end NUMINAMATH_CALUDE_age_problem_l1367_136783


namespace NUMINAMATH_CALUDE_investment_percentage_proof_l1367_136758

/-- Proves that given the investment conditions, the percentage at which $3,500 was invested is 4% --/
theorem investment_percentage_proof (total_investment : ℝ) (investment1 : ℝ) (investment2 : ℝ) 
  (rate1 : ℝ) (rate3 : ℝ) (desired_income : ℝ) (x : ℝ) :
  total_investment = 10000 →
  investment1 = 4000 →
  investment2 = 3500 →
  rate1 = 0.05 →
  rate3 = 0.064 →
  desired_income = 500 →
  investment1 * rate1 + investment2 * (x / 100) + (total_investment - investment1 - investment2) * rate3 = desired_income →
  x = 4 := by
sorry

end NUMINAMATH_CALUDE_investment_percentage_proof_l1367_136758


namespace NUMINAMATH_CALUDE_trigonometric_expression_equals_one_l1367_136740

open Real

theorem trigonometric_expression_equals_one : 
  (sin (15 * π / 180) * cos (15 * π / 180) + cos (165 * π / 180) * cos (105 * π / 180)) /
  (sin (19 * π / 180) * cos (11 * π / 180) + cos (161 * π / 180) * cos (101 * π / 180)) = 1 := by
  sorry

end NUMINAMATH_CALUDE_trigonometric_expression_equals_one_l1367_136740


namespace NUMINAMATH_CALUDE_regular_polygon_sides_l1367_136715

theorem regular_polygon_sides (n : ℕ) (n_pos : 0 < n) : 
  (∀ θ : ℝ, θ = 156 → (n : ℝ) * θ = 180 * ((n : ℝ) - 2)) → n = 15 := by
  sorry

end NUMINAMATH_CALUDE_regular_polygon_sides_l1367_136715


namespace NUMINAMATH_CALUDE_triangle_abc_cosine_sine_l1367_136745

theorem triangle_abc_cosine_sine (A B C : ℝ) (cosC_half : ℝ) (BC AC : ℝ) :
  cosC_half = Real.sqrt 5 / 5 →
  BC = 1 →
  AC = 5 →
  (Real.cos C = -3/5 ∧ Real.sin A = Real.sqrt 2 / 10) :=
by sorry

end NUMINAMATH_CALUDE_triangle_abc_cosine_sine_l1367_136745


namespace NUMINAMATH_CALUDE_max_value_sine_cosine_l1367_136763

theorem max_value_sine_cosine (x : Real) : 
  0 ≤ x → x < 2 * Real.pi → 
  ∃ (max_x : Real), max_x = 5 * Real.pi / 6 ∧
    ∀ y : Real, 0 ≤ y → y < 2 * Real.pi → 
      Real.sin x - Real.sqrt 3 * Real.cos x ≤ Real.sin max_x - Real.sqrt 3 * Real.cos max_x :=
by sorry

end NUMINAMATH_CALUDE_max_value_sine_cosine_l1367_136763


namespace NUMINAMATH_CALUDE_store_exit_ways_l1367_136775

/-- The number of different oreo flavors --/
def oreo_flavors : ℕ := 8

/-- The number of different milk types --/
def milk_types : ℕ := 4

/-- The total number of items Charlie can choose from --/
def charlie_choices : ℕ := oreo_flavors + milk_types

/-- The total number of products they leave with --/
def total_products : ℕ := 5

/-- Function to calculate the number of ways Delta can choose n oreos --/
def delta_choices (n : ℕ) : ℕ :=
  if n = 0 then 1
  else if n = 1 then oreo_flavors
  else if n = 2 then (Nat.choose oreo_flavors 2) + oreo_flavors
  else if n = 3 then (Nat.choose oreo_flavors 3) + oreo_flavors * (oreo_flavors - 1) + oreo_flavors
  else if n = 4 then (Nat.choose oreo_flavors 4) + (Nat.choose oreo_flavors 2) * (Nat.choose (oreo_flavors - 2) 2) / 2 + oreo_flavors * (oreo_flavors - 1) + oreo_flavors
  else (Nat.choose oreo_flavors 5) + (Nat.choose oreo_flavors 2) * (Nat.choose (oreo_flavors - 2) 3) + oreo_flavors * (Nat.choose (oreo_flavors - 1) 2) + oreo_flavors

/-- The total number of ways Charlie and Delta could have left the store --/
def total_ways : ℕ :=
  (Nat.choose charlie_choices total_products) +
  (Nat.choose charlie_choices 4) * (delta_choices 1) +
  (Nat.choose charlie_choices 3) * (delta_choices 2) +
  (Nat.choose charlie_choices 2) * (delta_choices 3) +
  (Nat.choose charlie_choices 1) * (delta_choices 4) +
  (delta_choices 5)

theorem store_exit_ways : total_ways = 25512 := by
  sorry

end NUMINAMATH_CALUDE_store_exit_ways_l1367_136775


namespace NUMINAMATH_CALUDE_max_regions_five_lines_l1367_136748

/-- The maximum number of regions a rectangle can be divided into by n line segments -/
def maxRegions (n : ℕ) : ℕ :=
  if n = 0 then 1 else maxRegions (n - 1) + n

/-- Theorem: The maximum number of regions a rectangle can be divided into by 5 line segments is 16 -/
theorem max_regions_five_lines :
  maxRegions 5 = 16 := by sorry

end NUMINAMATH_CALUDE_max_regions_five_lines_l1367_136748


namespace NUMINAMATH_CALUDE_processing_box_function_l1367_136723

-- Define the types of boxes in a flowchart
inductive FlowchartBox
  | Processing
  | Decision
  | Terminal
  | InputOutput

-- Define the functions of boxes in a flowchart
def boxFunction : FlowchartBox → String
  | FlowchartBox.Processing => "assignment and calculation"
  | FlowchartBox.Decision => "determine execution direction"
  | FlowchartBox.Terminal => "start and end of algorithm"
  | FlowchartBox.InputOutput => "handle data input and output"

-- Theorem statement
theorem processing_box_function :
  boxFunction FlowchartBox.Processing = "assignment and calculation" := by
  sorry

end NUMINAMATH_CALUDE_processing_box_function_l1367_136723


namespace NUMINAMATH_CALUDE_jim_purchase_total_l1367_136730

/-- Calculate the total amount Jim paid for lamps and bulbs --/
theorem jim_purchase_total : 
  let lamp_cost : ℚ := 7
  let bulb_cost : ℚ := lamp_cost - 4
  let lamp_quantity : ℕ := 2
  let bulb_quantity : ℕ := 6
  let tax_rate : ℚ := 5 / 100
  let bulb_discount : ℚ := 10 / 100
  let total_lamp_cost : ℚ := lamp_cost * lamp_quantity
  let total_bulb_cost : ℚ := bulb_cost * bulb_quantity
  let discounted_bulb_cost : ℚ := total_bulb_cost * (1 - bulb_discount)
  let subtotal : ℚ := total_lamp_cost + discounted_bulb_cost
  let tax_amount : ℚ := subtotal * tax_rate
  let total_cost : ℚ := subtotal + tax_amount
  total_cost = 3171 / 100 := by sorry

end NUMINAMATH_CALUDE_jim_purchase_total_l1367_136730


namespace NUMINAMATH_CALUDE_friendly_sequences_exist_l1367_136791

/-- Definition of a friendly pair of sequences -/
def is_friendly_pair (a b : ℕ → ℕ) : Prop :=
  (∀ n, a n > 0 ∧ b n > 0) ∧
  (∀ k : ℕ, ∃! (i j : ℕ), a i * b j = k)

/-- Theorem stating the existence of friendly sequences -/
theorem friendly_sequences_exist : ∃ (a b : ℕ → ℕ), is_friendly_pair a b :=
sorry

end NUMINAMATH_CALUDE_friendly_sequences_exist_l1367_136791


namespace NUMINAMATH_CALUDE_solve_equation_l1367_136753

theorem solve_equation : ∃ y : ℝ, (3 * y - 15) / 7 = 18 ∧ y = 47 := by sorry

end NUMINAMATH_CALUDE_solve_equation_l1367_136753


namespace NUMINAMATH_CALUDE_square_plus_one_to_zero_is_one_l1367_136754

theorem square_plus_one_to_zero_is_one (m : ℝ) : (m^2 + 1)^0 = 1 := by
  sorry

end NUMINAMATH_CALUDE_square_plus_one_to_zero_is_one_l1367_136754


namespace NUMINAMATH_CALUDE_witch_cake_votes_l1367_136710

/-- The number of votes for the witch cake -/
def witch_votes : ℕ := sorry

/-- The number of votes for the unicorn cake -/
def unicorn_votes : ℕ := 3 * witch_votes

/-- The number of votes for the dragon cake -/
def dragon_votes : ℕ := witch_votes + 25

/-- The total number of votes cast -/
def total_votes : ℕ := 60

theorem witch_cake_votes :
  witch_votes = 7 ∧
  unicorn_votes = 3 * witch_votes ∧
  dragon_votes = witch_votes + 25 ∧
  witch_votes + unicorn_votes + dragon_votes = total_votes :=
sorry

end NUMINAMATH_CALUDE_witch_cake_votes_l1367_136710


namespace NUMINAMATH_CALUDE_limit_of_a_l1367_136749

def a (n : ℕ) : ℚ := (3 * n - 1) / (5 * n + 1)

theorem limit_of_a : 
  ∀ ε > 0, ∃ N : ℕ, ∀ n ≥ N, |a n - 3/5| < ε :=
sorry

end NUMINAMATH_CALUDE_limit_of_a_l1367_136749


namespace NUMINAMATH_CALUDE_min_sum_abc_l1367_136725

theorem min_sum_abc (a b c : ℕ+) : 
  a * b * c = 2310 →
  (∃ p q : ℕ, Nat.Prime p ∧ Nat.Prime q ∧ p ≠ q ∧ c = p * q) →
  (∀ x y z : ℕ+, x * y * z = 2310 → 
    (∃ p q : ℕ, Nat.Prime p ∧ Nat.Prime q ∧ p ≠ q ∧ z = p * q) →
    a + b + c ≤ x + y + z) →
  a + b + c = 88 :=
by sorry

end NUMINAMATH_CALUDE_min_sum_abc_l1367_136725


namespace NUMINAMATH_CALUDE_geometric_sequences_operations_l1367_136774

/-- A sequence is geometric if the ratio of consecutive terms is constant and non-zero -/
def IsGeometricSequence (s : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, r ≠ 0 ∧ ∀ n : ℕ, s (n + 1) = r * s n

theorem geometric_sequences_operations
  (a b : ℕ → ℝ)
  (ha : IsGeometricSequence a)
  (hb : IsGeometricSequence b) :
  IsGeometricSequence (fun n ↦ a n * b n) ∧
  IsGeometricSequence (fun n ↦ a n / b n) ∧
  ¬ (∀ a b : ℕ → ℝ, IsGeometricSequence a → IsGeometricSequence b → IsGeometricSequence (fun n ↦ a n + b n)) ∧
  ¬ (∀ a b : ℕ → ℝ, IsGeometricSequence a → IsGeometricSequence b → IsGeometricSequence (fun n ↦ a n - b n)) :=
by sorry


end NUMINAMATH_CALUDE_geometric_sequences_operations_l1367_136774


namespace NUMINAMATH_CALUDE_gcf_of_90_and_126_l1367_136757

theorem gcf_of_90_and_126 : Nat.gcd 90 126 = 18 := by
  sorry

end NUMINAMATH_CALUDE_gcf_of_90_and_126_l1367_136757


namespace NUMINAMATH_CALUDE_solutions_nonempty_and_finite_l1367_136704

def solution_set (n : ℕ) : Set (ℕ × ℕ × ℕ) :=
  {(x, y, z) | Real.sqrt ((x^2 : ℝ) + y + n) + Real.sqrt ((y^2 : ℝ) + x + n) = z}

theorem solutions_nonempty_and_finite (n : ℕ) :
  (solution_set n).Nonempty ∧ (solution_set n).Finite :=
sorry

end NUMINAMATH_CALUDE_solutions_nonempty_and_finite_l1367_136704


namespace NUMINAMATH_CALUDE_power_product_rule_l1367_136724

theorem power_product_rule (a : ℝ) : (a * a^3)^2 = a^8 := by
  sorry

end NUMINAMATH_CALUDE_power_product_rule_l1367_136724


namespace NUMINAMATH_CALUDE_correct_calculation_l1367_136781

theorem correct_calculation (m : ℝ) : 2 * m^3 * 3 * m^2 = 6 * m^5 := by
  sorry

end NUMINAMATH_CALUDE_correct_calculation_l1367_136781


namespace NUMINAMATH_CALUDE_trig_expression_equals_sqrt_two_l1367_136779

theorem trig_expression_equals_sqrt_two :
  (Real.cos (-585 * π / 180)) / (Real.tan (495 * π / 180) + Real.sin (-690 * π / 180)) = Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_trig_expression_equals_sqrt_two_l1367_136779


namespace NUMINAMATH_CALUDE_joel_puzzles_l1367_136766

/-- The number of puzzles Joel collected -/
def puzzles : ℕ := sorry

/-- The number of toys Joel's sister donated -/
def sister_toys : ℕ := sorry

/-- The total number of toys Joel donated -/
def total_toys : ℕ := 108

/-- The number of stuffed animals Joel collected -/
def stuffed_animals : ℕ := 18

/-- The number of action figures Joel collected -/
def action_figures : ℕ := 42

/-- The number of board games Joel collected -/
def board_games : ℕ := 2

/-- The number of toys Joel added from his own closet -/
def joel_toys : ℕ := 22

theorem joel_puzzles :
  puzzles = 13 ∧
  sister_toys * 2 = joel_toys ∧
  stuffed_animals + action_figures + board_games + puzzles + sister_toys + joel_toys = total_toys :=
sorry

end NUMINAMATH_CALUDE_joel_puzzles_l1367_136766


namespace NUMINAMATH_CALUDE_smallest_b_is_85_l1367_136796

/-- A pair of integers that multiply to give 1764 -/
def ValidPair : Type := { p : ℤ × ℤ // p.1 * p.2 = 1764 }

/-- Predicate to check if a number is a perfect square -/
def IsPerfectSquare (n : ℤ) : Prop := ∃ m : ℤ, m * m = n

/-- The sum of a valid pair -/
def PairSum (p : ValidPair) : ℤ := p.val.1 + p.val.2

theorem smallest_b_is_85 :
  (∃ (b : ℕ), 
    (∃ (p : ValidPair), PairSum p = b) ∧ 
    (∃ (p : ValidPair), IsPerfectSquare p.val.1 ∨ IsPerfectSquare p.val.2) ∧
    (∀ (b' : ℕ), b' < b → 
      (∀ (p : ValidPair), PairSum p ≠ b' ∨ 
        (¬ IsPerfectSquare p.val.1 ∧ ¬ IsPerfectSquare p.val.2)))) ∧
  (∀ (b : ℕ), 
    ((∃ (p : ValidPair), PairSum p = b) ∧ 
     (∃ (p : ValidPair), IsPerfectSquare p.val.1 ∨ IsPerfectSquare p.val.2) ∧
     (∀ (b' : ℕ), b' < b → 
       (∀ (p : ValidPair), PairSum p ≠ b' ∨ 
         (¬ IsPerfectSquare p.val.1 ∧ ¬ IsPerfectSquare p.val.2))))
    → b = 85) :=
sorry

end NUMINAMATH_CALUDE_smallest_b_is_85_l1367_136796


namespace NUMINAMATH_CALUDE_abs_equation_solution_l1367_136717

theorem abs_equation_solution :
  ∃! x : ℝ, |x + 4| = 3 - x :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_CALUDE_abs_equation_solution_l1367_136717


namespace NUMINAMATH_CALUDE_suraya_kayla_difference_l1367_136701

/-- The number of apples picked by each person -/
structure ApplePicks where
  suraya : ℕ
  caleb : ℕ
  kayla : ℕ

/-- The conditions of the apple-picking scenario -/
def apple_picking_conditions (a : ApplePicks) : Prop :=
  a.suraya = a.caleb + 12 ∧
  a.caleb + 5 = a.kayla ∧
  a.kayla = 20

/-- The theorem stating that Suraya picked 7 more apples than Kayla -/
theorem suraya_kayla_difference (a : ApplePicks) 
  (h : apple_picking_conditions a) : a.suraya - a.kayla = 7 := by
  sorry


end NUMINAMATH_CALUDE_suraya_kayla_difference_l1367_136701


namespace NUMINAMATH_CALUDE_library_visitors_equation_l1367_136777

/-- Represents the equation for library visitors over three months -/
theorem library_visitors_equation 
  (initial_visitors : ℕ) 
  (growth_rate : ℝ) 
  (total_visitors : ℕ) :
  initial_visitors + 
  initial_visitors * (1 + growth_rate) + 
  initial_visitors * (1 + growth_rate)^2 = total_visitors ↔ 
  initial_visitors = 600 ∧ 
  growth_rate > 0 ∧ 
  total_visitors = 2850 :=
by sorry

end NUMINAMATH_CALUDE_library_visitors_equation_l1367_136777


namespace NUMINAMATH_CALUDE_twelve_sided_die_expected_value_l1367_136726

-- Define the number of sides on the die
def n : ℕ := 12

-- Define the expected value function for a fair die with n sides
def expected_value (n : ℕ) : ℚ :=
  (↑n + 1) / 2

-- Theorem statement
theorem twelve_sided_die_expected_value :
  expected_value n = 13/2 := by
  sorry

end NUMINAMATH_CALUDE_twelve_sided_die_expected_value_l1367_136726


namespace NUMINAMATH_CALUDE_sin_105_degrees_l1367_136706

theorem sin_105_degrees : 
  Real.sin (105 * π / 180) = (Real.sqrt 6 + Real.sqrt 2) / 4 := by sorry

end NUMINAMATH_CALUDE_sin_105_degrees_l1367_136706


namespace NUMINAMATH_CALUDE_money_sharing_problem_l1367_136733

theorem money_sharing_problem (amanda_share ben_share carlos_share total : ℕ) : 
  amanda_share = 30 ∧ 
  ben_share = 2 * amanda_share + 10 ∧
  amanda_share + ben_share + carlos_share = total ∧
  3 * ben_share = 4 * amanda_share ∧
  3 * carlos_share = 9 * amanda_share →
  total = 190 := by sorry

end NUMINAMATH_CALUDE_money_sharing_problem_l1367_136733


namespace NUMINAMATH_CALUDE_lamps_with_burnt_bulbs_l1367_136792

/-- Given a set of lamps with some burnt-out bulbs, proves the number of bulbs per lamp -/
theorem lamps_with_burnt_bulbs 
  (total_lamps : ℕ) 
  (burnt_fraction : ℚ) 
  (burnt_per_lamp : ℕ) 
  (working_bulbs : ℕ) : 
  total_lamps = 20 → 
  burnt_fraction = 1/4 → 
  burnt_per_lamp = 2 → 
  working_bulbs = 130 → 
  (total_lamps * (burnt_fraction * burnt_per_lamp + (1 - burnt_fraction) * working_bulbs / total_lamps)) / total_lamps = 7 := by
sorry

end NUMINAMATH_CALUDE_lamps_with_burnt_bulbs_l1367_136792


namespace NUMINAMATH_CALUDE_round_repeating_decimal_to_thousandth_l1367_136736

/-- Represents a repeating decimal where the whole number part is 67 and the repeating part is 836 -/
def repeating_decimal : ℚ := 67 + 836 / 999

/-- Rounding function to the nearest thousandth -/
def round_to_thousandth (x : ℚ) : ℚ := 
  (⌊x * 1000 + 1/2⌋ : ℚ) / 1000

theorem round_repeating_decimal_to_thousandth :
  round_to_thousandth repeating_decimal = 67837 / 1000 := by sorry

end NUMINAMATH_CALUDE_round_repeating_decimal_to_thousandth_l1367_136736


namespace NUMINAMATH_CALUDE_price_restoration_l1367_136705

theorem price_restoration (original_price : ℝ) (reduced_price : ℝ) : 
  reduced_price = 0.8 * original_price → 
  reduced_price * 1.25 = original_price :=
by sorry

end NUMINAMATH_CALUDE_price_restoration_l1367_136705


namespace NUMINAMATH_CALUDE_cookies_per_bag_l1367_136752

/-- Given 26 bags with an equal number of cookies and 52 cookies in total,
    prove that each bag contains 2 cookies. -/
theorem cookies_per_bag :
  ∀ (num_bags : ℕ) (total_cookies : ℕ) (cookies_per_bag : ℕ),
    num_bags = 26 →
    total_cookies = 52 →
    num_bags * cookies_per_bag = total_cookies →
    cookies_per_bag = 2 :=
by
  sorry

end NUMINAMATH_CALUDE_cookies_per_bag_l1367_136752


namespace NUMINAMATH_CALUDE_quadrilateral_weighted_centers_l1367_136776

-- Define a point in 2D space
structure Point :=
  (x : ℝ) (y : ℝ)

-- Define a quadrilateral
structure Quadrilateral :=
  (A B C D : Point)

-- Define a function to calculate the ratio of distances
def distanceRatio (P Q R : Point) : ℝ :=
  sorry

-- Define the weighted center
def weightedCenter (P Q : Point) (m₁ m₂ : ℝ) : Point :=
  sorry

-- Main theorem
theorem quadrilateral_weighted_centers 
  (quad : Quadrilateral) (P Q R S : Point) :
  (∃ (m₁ m₂ m₃ m₄ : ℝ), 
    P = weightedCenter quad.A quad.B m₁ m₂ ∧
    Q = weightedCenter quad.B quad.C m₂ m₃ ∧
    R = weightedCenter quad.C quad.D m₃ m₄ ∧
    S = weightedCenter quad.D quad.A m₄ m₁) ↔
  distanceRatio quad.A P quad.B *
  distanceRatio quad.B Q quad.C *
  distanceRatio quad.C R quad.D *
  distanceRatio quad.D S quad.A = 1 :=
sorry

end NUMINAMATH_CALUDE_quadrilateral_weighted_centers_l1367_136776


namespace NUMINAMATH_CALUDE_parabola_hyperbola_equations_l1367_136767

/-- Given a parabola and a hyperbola with specific properties, prove their equations -/
theorem parabola_hyperbola_equations :
  ∀ (a b : ℝ) (P : ℝ × ℝ),
    a > 0 → b > 0 →
    P = (3/2, Real.sqrt 6) →
    -- Parabola vertex at origin
    -- Directrix of parabola passes through a focus of hyperbola
    -- Directrix perpendicular to line connecting foci of hyperbola
    -- Parabola and hyperbola intersect at P
    ∃ (p : ℝ),
      -- Parabola equation
      (λ (x y : ℝ) => y^2 = 2*p*x) P.1 P.2 ∧
      -- Hyperbola equation
      (λ (x y : ℝ) => x^2/a^2 - y^2/b^2 = 1) P.1 P.2 →
      -- Prove the specific equations
      (λ (x y : ℝ) => y^2 = 4*x) = (λ (x y : ℝ) => y^2 = 2*p*x) ∧
      (λ (x y : ℝ) => 4*x^2 - 4/3*y^2 = 1) = (λ (x y : ℝ) => x^2/a^2 - y^2/b^2 = 1) := by
  sorry

end NUMINAMATH_CALUDE_parabola_hyperbola_equations_l1367_136767


namespace NUMINAMATH_CALUDE_remainder_problem_l1367_136714

theorem remainder_problem : (7 * 7^10 + 1^10) % 11 = 8 := by
  sorry

end NUMINAMATH_CALUDE_remainder_problem_l1367_136714


namespace NUMINAMATH_CALUDE_factorial_sum_square_solutions_l1367_136732

def factorial (n : ℕ) : ℕ := (Finset.range n).prod (λ i => i + 1)

def sum_factorials (n : ℕ) : ℕ := (Finset.range n).sum (λ i => factorial (i + 1))

theorem factorial_sum_square_solutions :
  ∀ n m : ℕ, sum_factorials n = m^2 ↔ (n = 1 ∧ m = 1) ∨ (n = 3 ∧ m = 3) := by
  sorry

end NUMINAMATH_CALUDE_factorial_sum_square_solutions_l1367_136732


namespace NUMINAMATH_CALUDE_decimal_to_percentage_l1367_136711

theorem decimal_to_percentage (x : ℝ) (h : x = 0.005) : x * 100 = 0.5 := by
  sorry

end NUMINAMATH_CALUDE_decimal_to_percentage_l1367_136711


namespace NUMINAMATH_CALUDE_rhombus_area_l1367_136712

/-- The area of a rhombus with side length 4 cm and an angle of 45 degrees between adjacent sides is 8√2 square centimeters. -/
theorem rhombus_area (side_length : ℝ) (angle : ℝ) :
  side_length = 4 →
  angle = π / 4 →
  let area := side_length * side_length * Real.sin angle
  area = 8 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_rhombus_area_l1367_136712


namespace NUMINAMATH_CALUDE_added_amount_l1367_136799

theorem added_amount (n : ℝ) (x : ℝ) (h1 : n = 12) (h2 : n / 2 + x = 11) : x = 5 := by
  sorry

end NUMINAMATH_CALUDE_added_amount_l1367_136799


namespace NUMINAMATH_CALUDE_f_properties_l1367_136746

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (a - x) * Real.exp x - 1

theorem f_properties :
  ∃ (a : ℝ),
    (∀ x ≠ 0, f a x / x < 1) ∧
    (∀ x : ℝ, f 1 x ≤ 0) ∧
    (f 1 0 = 0) :=
sorry

end NUMINAMATH_CALUDE_f_properties_l1367_136746


namespace NUMINAMATH_CALUDE_binary_10111_equals_23_l1367_136718

def binary_to_decimal (b : List Bool) : ℕ :=
  b.enum.foldr (fun (i, bit) acc => acc + if bit then 2^i else 0) 0

theorem binary_10111_equals_23 : 
  binary_to_decimal [true, true, true, false, true] = 23 := by
  sorry

end NUMINAMATH_CALUDE_binary_10111_equals_23_l1367_136718


namespace NUMINAMATH_CALUDE_systematic_sampling_elimination_l1367_136716

/-- The number of individuals randomly eliminated in a systematic sampling -/
def individuals_eliminated (population : ℕ) (sample_size : ℕ) : ℕ :=
  population % sample_size

/-- Theorem: The number of individuals randomly eliminated in a systematic sampling
    of 50 students from a population of 1252 is equal to 2 -/
theorem systematic_sampling_elimination :
  individuals_eliminated 1252 50 = 2 := by
  sorry

end NUMINAMATH_CALUDE_systematic_sampling_elimination_l1367_136716


namespace NUMINAMATH_CALUDE_tangent_line_equation_l1367_136728

-- Define the function f(x) = x^2
def f (x : ℝ) : ℝ := x^2

-- Define the point of tangency
def tangent_point : ℝ × ℝ := (1, f 1)

-- Theorem statement
theorem tangent_line_equation :
  ∃ (m b : ℝ), ∀ (x y : ℝ),
    (x = tangent_point.1 ∧ y = tangent_point.2) ∨
    (y - tangent_point.2 = m * (x - tangent_point.1)) ↔
    (2 * x - y - 1 = 0) :=
sorry

end NUMINAMATH_CALUDE_tangent_line_equation_l1367_136728


namespace NUMINAMATH_CALUDE_two_numbers_solution_l1367_136747

theorem two_numbers_solution : ∃ (x y : ℝ), 
  (2/3 : ℝ) * x + 2 * y = 20 ∧ 
  (1/4 : ℝ) * x - y = 2 ∧ 
  x = 144/7 ∧ 
  y = 22/7 := by
  sorry

end NUMINAMATH_CALUDE_two_numbers_solution_l1367_136747


namespace NUMINAMATH_CALUDE_milk_problem_l1367_136770

theorem milk_problem (initial_milk : ℚ) (rachel_fraction : ℚ) (jack_fraction : ℚ) :
  initial_milk = 3/4 →
  rachel_fraction = 5/8 →
  jack_fraction = 1/2 →
  (initial_milk - rachel_fraction * initial_milk) * jack_fraction = 9/64 := by
  sorry

end NUMINAMATH_CALUDE_milk_problem_l1367_136770


namespace NUMINAMATH_CALUDE_smallest_integer_of_three_l1367_136737

theorem smallest_integer_of_three (a b c : ℕ) : 
  a > 0 ∧ b > 0 ∧ c > 0 →
  a + b + c = 100 →
  2 * b = 3 * a →
  2 * c = 5 * a →
  a = 20 := by sorry

end NUMINAMATH_CALUDE_smallest_integer_of_three_l1367_136737


namespace NUMINAMATH_CALUDE_apple_cost_is_twelve_l1367_136787

/-- The cost of an apple given the total money, number of apples, and number of kids -/
def apple_cost (total_money : ℕ) (num_apples : ℕ) (num_kids : ℕ) : ℚ :=
  (total_money : ℚ) / (num_apples : ℚ)

/-- Theorem stating that the cost of each apple is 12 dollars -/
theorem apple_cost_is_twelve :
  apple_cost 360 30 6 = 12 := by
  sorry

end NUMINAMATH_CALUDE_apple_cost_is_twelve_l1367_136787


namespace NUMINAMATH_CALUDE_tuesday_calls_l1367_136756

/-- Represents the number of calls answered by Jean for each day of the work week -/
structure WeekCalls where
  monday : ℕ
  tuesday : ℕ
  wednesday : ℕ
  thursday : ℕ
  friday : ℕ

/-- Calculates the total number of calls answered in a week -/
def totalCalls (w : WeekCalls) : ℕ :=
  w.monday + w.tuesday + w.wednesday + w.thursday + w.friday

/-- Calculates the average number of calls per day -/
def averageCalls (w : WeekCalls) : ℚ :=
  totalCalls w / 5

theorem tuesday_calls (w : WeekCalls) 
  (h1 : w.monday = 35)
  (h2 : w.wednesday = 27)
  (h3 : w.thursday = 61)
  (h4 : w.friday = 31)
  (h5 : averageCalls w = 40) :
  w.tuesday = 46 := by
  sorry

#check tuesday_calls

end NUMINAMATH_CALUDE_tuesday_calls_l1367_136756


namespace NUMINAMATH_CALUDE_angle_at_point_l1367_136761

theorem angle_at_point (x : ℝ) : 
  (x + x + 160 = 360) → x = 100 := by sorry

end NUMINAMATH_CALUDE_angle_at_point_l1367_136761


namespace NUMINAMATH_CALUDE_greatest_prime_factor_of_341_l1367_136722

theorem greatest_prime_factor_of_341 : ∃ p : ℕ, 
  Nat.Prime p ∧ p ∣ 341 ∧ ∀ q : ℕ, Nat.Prime q → q ∣ 341 → q ≤ p :=
by
  sorry

end NUMINAMATH_CALUDE_greatest_prime_factor_of_341_l1367_136722


namespace NUMINAMATH_CALUDE_quadratic_expression_value_l1367_136755

theorem quadratic_expression_value (a : ℝ) (h : a^2 + 4*a - 5 = 0) : 3*a^2 + 12*a = 15 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_expression_value_l1367_136755


namespace NUMINAMATH_CALUDE_dress_discount_price_l1367_136768

/-- The final price of a dress after applying a discount -/
def final_price (original_price discount_percentage : ℚ) : ℚ :=
  original_price * (1 - discount_percentage / 100)

/-- Theorem stating that a dress originally priced at $350 with a 60% discount costs $140 -/
theorem dress_discount_price : final_price 350 60 = 140 := by
  sorry

end NUMINAMATH_CALUDE_dress_discount_price_l1367_136768


namespace NUMINAMATH_CALUDE_shopping_time_calculation_l1367_136765

-- Define the total shopping trip time in minutes
def total_shopping_time : ℕ := 90

-- Define the waiting times
def wait_for_cart : ℕ := 3
def wait_for_employee : ℕ := 13
def wait_for_restock : ℕ := 14
def wait_in_line : ℕ := 18

-- Define the theorem
theorem shopping_time_calculation :
  total_shopping_time - (wait_for_cart + wait_for_employee + wait_for_restock + wait_in_line) = 42 := by
  sorry

end NUMINAMATH_CALUDE_shopping_time_calculation_l1367_136765


namespace NUMINAMATH_CALUDE_complex_square_roots_l1367_136773

theorem complex_square_roots (z : ℂ) : z^2 = -45 - 54*I ↔ z = 3 - 9*I ∨ z = -3 + 9*I := by
  sorry

end NUMINAMATH_CALUDE_complex_square_roots_l1367_136773


namespace NUMINAMATH_CALUDE_divided_number_problem_l1367_136721

theorem divided_number_problem (x y : ℝ) : 
  x > y ∧ y = 11 ∧ 7 * x + 5 * y = 146 → x + y = 24 := by
  sorry

end NUMINAMATH_CALUDE_divided_number_problem_l1367_136721


namespace NUMINAMATH_CALUDE_arccos_cos_three_pi_half_l1367_136771

theorem arccos_cos_three_pi_half : Real.arccos (Real.cos (3 * π / 2)) = π / 2 := by
  sorry

end NUMINAMATH_CALUDE_arccos_cos_three_pi_half_l1367_136771


namespace NUMINAMATH_CALUDE_weight_lifting_competition_l1367_136798

theorem weight_lifting_competition (total_weight first_lift : ℕ) 
  (h1 : total_weight = 1800)
  (h2 : first_lift = 700) :
  2 * first_lift - (total_weight - first_lift) = 300 := by
  sorry

end NUMINAMATH_CALUDE_weight_lifting_competition_l1367_136798


namespace NUMINAMATH_CALUDE_sum_of_popsicle_sticks_l1367_136784

/-- The number of popsicle sticks Gino has -/
def gino_sticks : ℕ := 63

/-- The number of popsicle sticks I have -/
def my_sticks : ℕ := 50

/-- The sum of Gino's and my popsicle sticks -/
def total_sticks : ℕ := gino_sticks + my_sticks

theorem sum_of_popsicle_sticks : total_sticks = 113 := by sorry

end NUMINAMATH_CALUDE_sum_of_popsicle_sticks_l1367_136784


namespace NUMINAMATH_CALUDE_money_sharing_l1367_136794

theorem money_sharing (total : ℕ) (amanda ben carlos : ℕ) : 
  amanda + ben + carlos = total →
  amanda = 24 →
  2 * ben = 3 * amanda →
  8 * amanda = 3 * carlos →
  total = 156 := by
sorry

end NUMINAMATH_CALUDE_money_sharing_l1367_136794


namespace NUMINAMATH_CALUDE_word_arrangements_count_l1367_136720

def word_length : ℕ := 12
def repeated_letter_1_count : ℕ := 3
def repeated_letter_2_count : ℕ := 2
def repeated_letter_3_count : ℕ := 2
def unique_letters_count : ℕ := 5

def arrangements_count : ℕ := 19958400

theorem word_arrangements_count :
  (word_length.factorial) / 
  (repeated_letter_1_count.factorial * 
   repeated_letter_2_count.factorial * 
   repeated_letter_3_count.factorial) = arrangements_count := by
  sorry

end NUMINAMATH_CALUDE_word_arrangements_count_l1367_136720


namespace NUMINAMATH_CALUDE_connie_blue_markers_l1367_136731

/-- Given the total number of markers and the number of red markers,
    calculate the number of blue markers. -/
def blue_markers (total : ℕ) (red : ℕ) : ℕ := total - red

/-- Theorem stating that Connie has 64 blue markers -/
theorem connie_blue_markers :
  let total_markers : ℕ := 105
  let red_markers : ℕ := 41
  blue_markers total_markers red_markers = 64 := by
  sorry

end NUMINAMATH_CALUDE_connie_blue_markers_l1367_136731


namespace NUMINAMATH_CALUDE_line_passes_through_point_l1367_136750

/-- The line mx + y - m = 0 passes through the point (1, 0) for all real m. -/
theorem line_passes_through_point :
  ∀ (m : ℝ), m * 1 + 0 - m = 0 := by sorry

end NUMINAMATH_CALUDE_line_passes_through_point_l1367_136750


namespace NUMINAMATH_CALUDE_quadrilateral_area_l1367_136764

/-- The area of a quadrilateral with vertices A(1, 3), B(1, 1), C(5, 6), and D(4, 3) is 8.5 square units. -/
theorem quadrilateral_area : 
  let A : ℝ × ℝ := (1, 3)
  let B : ℝ × ℝ := (1, 1)
  let C : ℝ × ℝ := (5, 6)
  let D : ℝ × ℝ := (4, 3)
  let area := abs (A.1 * (B.2 - C.2) + B.1 * (C.2 - A.2) + C.1 * (A.2 - B.2)) / 2 +
              abs (A.1 * (C.2 - D.2) + C.1 * (D.2 - A.2) + D.1 * (A.2 - C.2)) / 2
  area = 8.5 := by
  sorry

end NUMINAMATH_CALUDE_quadrilateral_area_l1367_136764


namespace NUMINAMATH_CALUDE_nested_expression_evaluation_l1367_136760

theorem nested_expression_evaluation : (3 * (3 * (3 * (3 + 2) + 2) + 2) + 2) = 161 := by
  sorry

end NUMINAMATH_CALUDE_nested_expression_evaluation_l1367_136760


namespace NUMINAMATH_CALUDE_calculation_proof_l1367_136700

theorem calculation_proof : -3^2 - (-1)^4 * 5 / (-5/3) = -6 := by
  sorry

end NUMINAMATH_CALUDE_calculation_proof_l1367_136700


namespace NUMINAMATH_CALUDE_ellipse_m_range_l1367_136708

/-- The equation of the curve -/
def curve_equation (x y m : ℝ) : Prop :=
  x^2 / (m - 2) + y^2 / (6 - m) = 1

/-- Definition of an ellipse in terms of its equation -/
def is_ellipse (m : ℝ) : Prop :=
  (∀ x y, curve_equation x y m → x^2 / (m - 2) > 0 ∧ y^2 / (6 - m) > 0) ∧
  m - 2 ≠ 6 - m

/-- Theorem: The range of m for which the curve is an ellipse -/
theorem ellipse_m_range :
  ∀ m : ℝ, is_ellipse m ↔ (2 < m ∧ m < 6 ∧ m ≠ 4) :=
sorry

end NUMINAMATH_CALUDE_ellipse_m_range_l1367_136708


namespace NUMINAMATH_CALUDE_range_of_m_l1367_136762

def is_periodic (f : ℝ → ℝ) (p : ℝ) : Prop :=
  ∀ x, f (x + p) = f x

def smallest_positive_period (f : ℝ → ℝ) (p : ℝ) : Prop :=
  p > 0 ∧ is_periodic f p ∧ ∀ q, 0 < q ∧ q < p → ¬ is_periodic f q

def is_odd (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

theorem range_of_m (f : ℝ → ℝ) (m : ℝ) :
  is_odd f →
  smallest_positive_period f 3 →
  f 1 > -2 →
  f 2 = m^2 - m →
  m ∈ Set.Ioo (-1 : ℝ) 2 :=
by sorry

end NUMINAMATH_CALUDE_range_of_m_l1367_136762


namespace NUMINAMATH_CALUDE_vector_decomposition_l1367_136797

def x : ℝ × ℝ × ℝ := (-5, -5, 5)
def p : ℝ × ℝ × ℝ := (-2, 0, 1)
def q : ℝ × ℝ × ℝ := (1, 3, -1)
def r : ℝ × ℝ × ℝ := (0, 4, 1)

theorem vector_decomposition :
  x = p + (-3 : ℝ) • q + r := by sorry

end NUMINAMATH_CALUDE_vector_decomposition_l1367_136797


namespace NUMINAMATH_CALUDE_condition_equivalence_l1367_136719

-- Define the sets A, B, and C
def A : Set ℝ := {x | x - 2 > 0}
def B : Set ℝ := {x | x < 0}
def C : Set ℝ := {x | x * (x - 2) > 0}

-- State the theorem
theorem condition_equivalence : ∀ x : ℝ, x ∈ A ∪ B ↔ x ∈ C := by
  sorry

end NUMINAMATH_CALUDE_condition_equivalence_l1367_136719


namespace NUMINAMATH_CALUDE_fraction_calculation_l1367_136744

theorem fraction_calculation (N : ℝ) (h : 0.4 * N = 240) : 
  (1/4 : ℝ) * (1/3 : ℝ) * (2/5 : ℝ) * N = 20 := by
  sorry

end NUMINAMATH_CALUDE_fraction_calculation_l1367_136744


namespace NUMINAMATH_CALUDE_specific_prism_surface_area_l1367_136702

/-- A right triangular prism with given dimensions -/
structure RightTriangularPrism where
  leg1 : ℝ
  leg2 : ℝ
  hypotenuse : ℝ
  height : ℝ

/-- Calculate the surface area of a right triangular prism -/
def surfaceArea (prism : RightTriangularPrism) : ℝ :=
  prism.leg1 * prism.leg2 + (prism.leg1 + prism.leg2 + prism.hypotenuse) * prism.height

/-- The surface area of the specific right triangular prism is 72 -/
theorem specific_prism_surface_area :
  let prism : RightTriangularPrism := {
    leg1 := 3,
    leg2 := 4,
    hypotenuse := 5,
    height := 5
  }
  surfaceArea prism = 72 := by sorry

end NUMINAMATH_CALUDE_specific_prism_surface_area_l1367_136702


namespace NUMINAMATH_CALUDE_haley_trees_l1367_136785

theorem haley_trees (initial_trees : ℕ) (dead_trees : ℕ) (final_trees : ℕ) 
  (h1 : initial_trees = 9)
  (h2 : dead_trees = 4)
  (h3 : final_trees = 10) :
  final_trees - (initial_trees - dead_trees) = 5 := by
sorry

end NUMINAMATH_CALUDE_haley_trees_l1367_136785


namespace NUMINAMATH_CALUDE_tile_arrangements_example_l1367_136703

/-- The number of distinguishable arrangements of tiles -/
def tileArrangements (brown purple green yellow : ℕ) : ℕ :=
  Nat.factorial (brown + purple + green + yellow) /
  (Nat.factorial brown * Nat.factorial purple * Nat.factorial green * Nat.factorial yellow)

/-- Theorem stating that the number of distinguishable arrangements
    of 2 brown, 2 purple, 3 green, and 2 yellow tiles is 3780 -/
theorem tile_arrangements_example :
  tileArrangements 2 2 3 2 = 3780 := by
  sorry

end NUMINAMATH_CALUDE_tile_arrangements_example_l1367_136703


namespace NUMINAMATH_CALUDE_brochure_distribution_l1367_136788

theorem brochure_distribution (total_brochures : ℕ) (num_boxes : ℕ) 
  (h1 : total_brochures = 5000) 
  (h2 : num_boxes = 5) : 
  (total_brochures / num_boxes : ℚ) / total_brochures = 1 / 5 := by
  sorry

end NUMINAMATH_CALUDE_brochure_distribution_l1367_136788


namespace NUMINAMATH_CALUDE_functional_equation_solution_l1367_136780

theorem functional_equation_solution (f : ℝ → ℝ) :
  (∀ x y : ℝ, f x * f y = f (x - y)) →
  ((∀ x : ℝ, f x = 0) ∨ (∀ x : ℝ, f x = 1)) :=
by sorry

end NUMINAMATH_CALUDE_functional_equation_solution_l1367_136780


namespace NUMINAMATH_CALUDE_johns_age_l1367_136713

theorem johns_age (john_age father_age : ℕ) : 
  john_age + father_age = 77 →
  father_age = 2 * john_age + 32 →
  john_age = 15 := by
sorry

end NUMINAMATH_CALUDE_johns_age_l1367_136713


namespace NUMINAMATH_CALUDE_tangent_line_curve_n_value_l1367_136729

/-- Given a line and a curve that are tangent at a point, prove the value of n. -/
theorem tangent_line_curve_n_value :
  ∀ (k m n : ℝ),
  (∀ x, k * x + 1 = x^3 + m * x + n → x = 1 ∧ k * x + 1 = 3) →
  (∀ x, (3 * x^2 + m) * (x - 1) + (1^3 + m * 1 + n) = k * x + 1) →
  n = 3 := by
sorry

end NUMINAMATH_CALUDE_tangent_line_curve_n_value_l1367_136729
