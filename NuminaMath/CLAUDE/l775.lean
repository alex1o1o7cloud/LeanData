import Mathlib

namespace tan_150_degrees_l775_77542

theorem tan_150_degrees :
  Real.tan (150 * π / 180) = -Real.sqrt 3 := by
  sorry

end tan_150_degrees_l775_77542


namespace odd_periodic_function_difference_l775_77518

/-- A function f is odd if f(-x) = -f(x) for all x -/
def IsOdd (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

/-- A function f has period 4 if f(x) = f(x + 4) for all x -/
def HasPeriod4 (f : ℝ → ℝ) : Prop := ∀ x, f x = f (x + 4)

theorem odd_periodic_function_difference (f : ℝ → ℝ) 
  (h_odd : IsOdd f) 
  (h_period : HasPeriod4 f) 
  (h_def : ∀ x ∈ Set.Ioo (-2) 0, f x = 2^x) : 
  f 2016 - f 2015 = -1/2 := by
sorry

end odd_periodic_function_difference_l775_77518


namespace arithmetic_geometric_mean_inequality_l775_77572

theorem arithmetic_geometric_mean_inequality
  (a b c : ℝ) (ha : 0 ≤ a) (hb : 0 ≤ b) (hc : 0 ≤ c) :
  (a + b + c) / 3 ≥ (a * b * c) ^ (1/3) :=
by sorry

end arithmetic_geometric_mean_inequality_l775_77572


namespace min_sum_distances_l775_77596

open Real

/-- The minimum sum of distances between four points in a Cartesian plane -/
theorem min_sum_distances :
  let A : ℝ × ℝ := (-2, -3)
  let B : ℝ × ℝ := (4, -1)
  let C : ℝ → ℝ × ℝ := λ m ↦ (m, 0)
  let D : ℝ → ℝ × ℝ := λ n ↦ (n, n)
  let distance (p q : ℝ × ℝ) : ℝ := Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)
  ∃ (m n : ℝ), ∀ (m' n' : ℝ),
    distance A B + distance B (C m) + distance (C m) (D n) + distance (D n) A ≤
    distance A B + distance B (C m') + distance (C m') (D n') + distance (D n') A ∧
    distance A B + distance B (C m) + distance (C m) (D n) + distance (D n) A = 58 + 2 * Real.sqrt 10 :=
by sorry


end min_sum_distances_l775_77596


namespace total_earnings_l775_77500

def weekly_earnings : ℕ := 16
def harvest_duration : ℕ := 76

theorem total_earnings : weekly_earnings * harvest_duration = 1216 := by
  sorry

end total_earnings_l775_77500


namespace unique_solution_linear_equation_l775_77535

theorem unique_solution_linear_equation (a b : ℝ) (ha : a ≠ 0) :
  ∃! x : ℝ, a * x = b ∧ x = b / a := by
  sorry

end unique_solution_linear_equation_l775_77535


namespace simplify_fraction_l775_77584

theorem simplify_fraction : 20 * (9 / 14) * (1 / 18) = 5 / 7 := by
  sorry

end simplify_fraction_l775_77584


namespace quadratic_max_value_l775_77587

/-- A quadratic function that takes specific values for consecutive natural numbers -/
def QuadraticFunction (f : ℝ → ℝ) : Prop :=
  ∃ n : ℕ, f n = 6 ∧ f (n + 1) = 14 ∧ f (n + 2) = 14

/-- The theorem stating the maximum value of the quadratic function -/
theorem quadratic_max_value (f : ℝ → ℝ) (h : QuadraticFunction f) :
  ∃ x : ℝ, ∀ y : ℝ, f y ≤ f x ∧ f x = 15 :=
sorry

end quadratic_max_value_l775_77587


namespace camera_tax_calculation_l775_77550

/-- Calculate the tax amount given the base price and tax rate -/
def calculateTax (basePrice taxRate : ℝ) : ℝ :=
  basePrice * taxRate

/-- Prove that the tax amount for a $200 camera with 15% tax rate is $30 -/
theorem camera_tax_calculation :
  let basePrice : ℝ := 200
  let taxRate : ℝ := 0.15
  calculateTax basePrice taxRate = 30 := by
sorry

#eval calculateTax 200 0.15

end camera_tax_calculation_l775_77550


namespace rectangular_frame_area_l775_77532

theorem rectangular_frame_area : 
  let width : ℚ := 81 / 4
  let depth : ℚ := 148 / 9
  let area : ℚ := width * depth
  ⌊area⌋ = 333 := by
  sorry

end rectangular_frame_area_l775_77532


namespace certain_number_equation_l775_77512

theorem certain_number_equation (x : ℝ) : 13 * x + 14 * x + 17 * x + 11 = 143 ↔ x = 3 := by
  sorry

end certain_number_equation_l775_77512


namespace correct_robes_count_l775_77540

/-- The number of robes a school already has for their choir. -/
def robes_already_have (total_singers : ℕ) (robe_cost : ℕ) (total_spend : ℕ) : ℕ :=
  total_singers - (total_spend / robe_cost)

/-- Theorem stating that the number of robes the school already has is correct. -/
theorem correct_robes_count :
  robes_already_have 30 2 36 = 12 := by sorry

end correct_robes_count_l775_77540


namespace exactlyOneAndTwoBlackMutuallyExclusiveNotContradictory_l775_77591

/-- Represents the outcome of drawing two balls from a bag -/
inductive DrawOutcome
| OneBOne  -- One black, one red
| TwoB     -- Two black
| TwoR     -- Two red

/-- The probability space for drawing two balls from a bag with 2 red and 3 black balls -/
def drawProbSpace : Type := DrawOutcome

/-- The event "Exactly one black ball is drawn" -/
def exactlyOneBlack (outcome : drawProbSpace) : Prop :=
  outcome = DrawOutcome.OneBOne

/-- The event "Exactly two black balls are drawn" -/
def exactlyTwoBlack (outcome : drawProbSpace) : Prop :=
  outcome = DrawOutcome.TwoB

/-- Two events are mutually exclusive if they cannot occur simultaneously -/
def mutuallyExclusive (e1 e2 : drawProbSpace → Prop) : Prop :=
  ∀ (outcome : drawProbSpace), ¬(e1 outcome ∧ e2 outcome)

/-- Two events are contradictory if exactly one of them must occur -/
def contradictory (e1 e2 : drawProbSpace → Prop) : Prop :=
  ∀ (outcome : drawProbSpace), e1 outcome ↔ ¬(e2 outcome)

theorem exactlyOneAndTwoBlackMutuallyExclusiveNotContradictory :
  mutuallyExclusive exactlyOneBlack exactlyTwoBlack ∧
  ¬(contradictory exactlyOneBlack exactlyTwoBlack) := by
  sorry

end exactlyOneAndTwoBlackMutuallyExclusiveNotContradictory_l775_77591


namespace quadratic_intersection_range_l775_77571

/-- The range of a for which the intersection of A and B is non-empty -/
theorem quadratic_intersection_range (a : ℝ) : 
  let f : ℝ → ℝ := λ x => a * x^2 - 2 * x - 2 * a
  let A : Set ℝ := {x | f x > 0}
  let B : Set ℝ := {x | 1 < x ∧ x < 3}
  (A ∩ B).Nonempty → a < -2 ∨ a > 6/7 :=
by sorry

end quadratic_intersection_range_l775_77571


namespace qin_jiushao_area_formula_l775_77531

theorem qin_jiushao_area_formula (a b c : ℝ) (h_pos_a : a > 0) (h_pos_b : b > 0) (h_pos_c : c > 0) :
  let S := Real.sqrt ((c^2 * a^2 - ((c^2 + a^2 - b^2) / 2)^2) / 4)
  a = 25 → b = 24 → c = 14 → S = (105 * Real.sqrt 39) / 4 :=
by sorry

end qin_jiushao_area_formula_l775_77531


namespace stating_max_tulips_in_bouquet_l775_77562

/-- Represents the cost of a yellow tulip in rubles -/
def yellow_cost : ℕ := 50

/-- Represents the cost of a red tulip in rubles -/
def red_cost : ℕ := 31

/-- Represents the maximum budget in rubles -/
def max_budget : ℕ := 600

/-- 
Theorem stating that the maximum number of tulips in a bouquet is 15,
given the specified conditions
-/
theorem max_tulips_in_bouquet :
  ∃ (y r : ℕ),
    -- The total number of tulips is odd
    Odd (y + r) ∧
    -- The difference between yellow and red tulips is exactly 1
    (y = r + 1 ∨ r = y + 1) ∧
    -- The total cost does not exceed the budget
    y * yellow_cost + r * red_cost ≤ max_budget ∧
    -- The total number of tulips is 15
    y + r = 15 ∧
    -- This is the maximum possible number of tulips
    ∀ (y' r' : ℕ),
      Odd (y' + r') →
      (y' = r' + 1 ∨ r' = y' + 1) →
      y' * yellow_cost + r' * red_cost ≤ max_budget →
      y' + r' ≤ 15 :=
by sorry

end stating_max_tulips_in_bouquet_l775_77562


namespace admission_fees_proof_l775_77515

-- Define the given conditions
def child_fee : ℚ := 1.5
def adult_fee : ℚ := 4
def total_people : ℕ := 315
def num_children : ℕ := 180

-- Define the function to calculate total admission fees
def total_admission_fees : ℚ :=
  (child_fee * num_children) + (adult_fee * (total_people - num_children))

-- Theorem to prove
theorem admission_fees_proof : total_admission_fees = 810 := by
  sorry

end admission_fees_proof_l775_77515


namespace triangle_condition_implies_linear_l775_77592

/-- A function satisfying the triangle condition -/
def TriangleCondition (f : ℝ → ℝ) : Prop :=
  ∀ (a b c : ℝ), a > 0 → b > 0 → c > 0 → a ≠ b → b ≠ c → a ≠ c →
    (a + b > c ∧ b + c > a ∧ c + a > b ↔ f a + f b > f c ∧ f b + f c > f a ∧ f c + f a > f b)

/-- The main theorem statement -/
theorem triangle_condition_implies_linear (f : ℝ → ℝ) (h : TriangleCondition f) :
  ∃ (c : ℝ), c > 0 ∧ ∀ (x : ℝ), x > 0 → f x = c * x :=
sorry

end triangle_condition_implies_linear_l775_77592


namespace guessing_game_factor_l775_77563

theorem guessing_game_factor (f : ℚ) : 33 * f = 2 * 51 - 3 → f = 3 := by
  sorry

end guessing_game_factor_l775_77563


namespace regular_polygon_with_36_degree_exterior_angle_is_decagon_l775_77545

/-- A regular polygon with exterior angles measuring 36° has 10 sides -/
theorem regular_polygon_with_36_degree_exterior_angle_is_decagon :
  ∀ (n : ℕ) (exterior_angle : ℝ),
    n > 0 →
    exterior_angle = 36 →
    (n : ℝ) * exterior_angle = 360 →
    n = 10 := by
  sorry

end regular_polygon_with_36_degree_exterior_angle_is_decagon_l775_77545


namespace bird_count_l775_77565

theorem bird_count (cardinals bluebirds swallows : ℕ) : 
  cardinals = 3 * bluebirds ∧ 
  swallows = bluebirds / 2 ∧ 
  swallows = 2 → 
  cardinals + bluebirds + swallows = 18 := by
sorry

end bird_count_l775_77565


namespace us_flag_stars_l775_77560

theorem us_flag_stars (stripes : ℕ) (total_shapes : ℕ) : 
  stripes = 13 → 
  total_shapes = 54 → 
  ∃ (stars : ℕ), 
    (stars / 2 - 3 + 2 * stripes + 6 = total_shapes) ∧ 
    (stars = 50) := by
  sorry

end us_flag_stars_l775_77560


namespace solve_mushroom_problem_l775_77508

def mushroom_pieces_problem (total_mushrooms : ℕ) 
                            (kenny_pieces : ℕ) 
                            (karla_pieces : ℕ) 
                            (remaining_pieces : ℕ) : Prop :=
  let total_pieces := kenny_pieces + karla_pieces + remaining_pieces
  total_pieces / total_mushrooms = 4

theorem solve_mushroom_problem : 
  mushroom_pieces_problem 22 38 42 8 := by
  sorry

end solve_mushroom_problem_l775_77508


namespace no_valid_solution_l775_77576

/-- Represents the conditions of the age problem -/
structure AgeProblem where
  jane_current_age : ℕ
  dick_current_age : ℕ
  n : ℕ
  jane_future_age : ℕ
  dick_future_age : ℕ

/-- Checks if the given ages satisfy the problem conditions -/
def satisfies_conditions (problem : AgeProblem) : Prop :=
  problem.jane_current_age = 30 ∧
  problem.dick_current_age = problem.jane_current_age + 5 ∧
  problem.n > 0 ∧
  problem.jane_future_age = problem.jane_current_age + problem.n ∧
  problem.dick_future_age = problem.dick_current_age + problem.n ∧
  10 ≤ problem.jane_future_age ∧ problem.jane_future_age ≤ 99 ∧
  10 ≤ problem.dick_future_age ∧ problem.dick_future_age ≤ 99 ∧
  (problem.jane_future_age / 10 = problem.dick_future_age % 10) ∧
  (problem.jane_future_age % 10 = problem.dick_future_age / 10)

/-- The main theorem stating that no valid solution exists -/
theorem no_valid_solution : ¬∃ (problem : AgeProblem), satisfies_conditions problem := by
  sorry

end no_valid_solution_l775_77576


namespace percentage_difference_l775_77509

theorem percentage_difference (x y : ℝ) (h : y = x + 0.6667 * x) :
  x = y * (1 - 0.6667) := by
  sorry

end percentage_difference_l775_77509


namespace gomoku_pieces_count_l775_77581

theorem gomoku_pieces_count :
  ∀ (initial_black : ℕ) (added_black : ℕ),
    initial_black > 0 →
    initial_black ≤ 5 →
    initial_black + added_black + (initial_black + (20 - added_black)) ≤ 30 →
    7 * (initial_black + (20 - added_black)) = 8 * (initial_black + added_black) →
    initial_black + added_black = 16 := by
  sorry

end gomoku_pieces_count_l775_77581


namespace probability_of_target_plate_l775_77583

structure LicensePlate where
  first : Char
  second : Char
  third : Char
  fourth : Char

def vowels : List Char := ['A', 'E', 'I', 'O', 'U', 'Y']
def non_vowels : List Char := ['B', 'C', 'D', 'F', 'G', 'H', 'J', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'X', 'Z']
def hex_digits : List Char := ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'D', 'E', 'F']

def is_valid_plate (plate : LicensePlate) : Prop :=
  plate.first ∈ vowels ∧
  plate.second ∈ non_vowels ∧
  plate.third ∈ non_vowels ∧
  plate.second ≠ plate.third ∧
  plate.fourth ∈ hex_digits

def total_valid_plates : ℕ := vowels.length * non_vowels.length * (non_vowels.length - 1) * hex_digits.length

def target_plate : LicensePlate := ⟨'E', 'Y', 'B', '5'⟩

theorem probability_of_target_plate :
  (1 : ℚ) / total_valid_plates = 1 / 44352 :=
sorry

end probability_of_target_plate_l775_77583


namespace conditional_probability_B_given_A_l775_77580

/-- A die is represented as a finite type with 6 elements -/
def Die : Type := Fin 6

/-- The sample space of rolling two dice -/
def SampleSpace : Type := Die × Die

/-- Event A: the number on the first die is a multiple of 3 -/
def EventA (outcome : SampleSpace) : Prop :=
  (outcome.1.val + 1) % 3 = 0

/-- Event B: the sum of the numbers on the two dice is greater than 7 -/
def EventB (outcome : SampleSpace) : Prop :=
  outcome.1.val + outcome.2.val + 2 > 7

/-- The probability measure on the sample space -/
def P : Set SampleSpace → ℝ := sorry

/-- Theorem: The conditional probability P(B|A) is 7/12 -/
theorem conditional_probability_B_given_A :
  P {outcome | EventB outcome ∧ EventA outcome} / P {outcome | EventA outcome} = 7/12 := by
  sorry

end conditional_probability_B_given_A_l775_77580


namespace alpha_plus_beta_is_75_degrees_l775_77519

theorem alpha_plus_beta_is_75_degrees (α β : Real) 
  (h1 : 0 < α ∧ α < π / 2)  -- α is acute
  (h2 : 0 < β ∧ β < π / 2)  -- β is acute
  (h3 : |Real.sin α - 1/2| + Real.sqrt (Real.tan β - 1) = 0) : 
  α + β = π / 2.4 := by  -- π/2.4 is equivalent to 75°
sorry

end alpha_plus_beta_is_75_degrees_l775_77519


namespace cylinder_and_cone_properties_l775_77551

/-- Properties of a cylinder and a cone --/
theorem cylinder_and_cone_properties 
  (base_area : ℝ) 
  (height : ℝ) 
  (cylinder_volume : ℝ) 
  (cone_volume : ℝ) 
  (h1 : base_area = 72) 
  (h2 : height = 6) 
  (h3 : cylinder_volume = base_area * height) 
  (h4 : cone_volume = (1/3) * cylinder_volume) : 
  cylinder_volume = 432 ∧ cone_volume = 144 := by
  sorry

#check cylinder_and_cone_properties

end cylinder_and_cone_properties_l775_77551


namespace square_area_from_diagonal_l775_77553

theorem square_area_from_diagonal (d : ℝ) (h : d = 8 * Real.sqrt 2) :
  let s := d / Real.sqrt 2
  s * s = 64 := by sorry

end square_area_from_diagonal_l775_77553


namespace more_numbers_with_one_l775_77511

def range_upper_bound : ℕ := 10^10

def numbers_without_one (n : ℕ) : ℕ := 9^n - 1

theorem more_numbers_with_one :
  range_upper_bound - numbers_without_one 10 > numbers_without_one 10 := by
  sorry

end more_numbers_with_one_l775_77511


namespace valid_numbers_l775_77549

def is_valid_number (n : ℕ) : Prop :=
  80 ≤ n ∧ n < 100 ∧ ∃ x : ℕ, n = 10 * x + (x - 1) ∧ 1 ≤ x ∧ x ≤ 9

theorem valid_numbers : 
  ∀ n : ℕ, is_valid_number n → n = 87 ∨ n = 98 :=
sorry

end valid_numbers_l775_77549


namespace runner_time_difference_l775_77567

theorem runner_time_difference (total_distance : ℝ) (half_distance : ℝ) (second_half_time : ℝ) :
  total_distance = 40 →
  half_distance = total_distance / 2 →
  second_half_time = 10 →
  ∃ (initial_speed : ℝ),
    initial_speed > 0 ∧
    half_distance / initial_speed + half_distance / (initial_speed / 2) = second_half_time + half_distance / initial_speed ∧
    second_half_time - half_distance / initial_speed = 5 :=
by sorry

end runner_time_difference_l775_77567


namespace extra_bananas_proof_l775_77590

/-- Calculates the number of extra bananas each child receives when some children are absent -/
def extra_bananas (total_children : ℕ) (absent_children : ℕ) : ℕ :=
  absent_children

theorem extra_bananas_proof (total_children : ℕ) (absent_children : ℕ) 
  (h1 : total_children = 700) 
  (h2 : absent_children = 350) :
  extra_bananas total_children absent_children = absent_children :=
by
  sorry

#eval extra_bananas 700 350

end extra_bananas_proof_l775_77590


namespace banana_arrangements_count_l775_77538

/-- The number of unique arrangements of the letters in "BANANA" -/
def banana_arrangements : ℕ :=
  Nat.factorial 6 / (Nat.factorial 3 * Nat.factorial 2)

/-- Theorem stating that the number of unique arrangements of the letters in "BANANA" is 60 -/
theorem banana_arrangements_count : banana_arrangements = 60 := by
  sorry

end banana_arrangements_count_l775_77538


namespace chord_line_equation_l775_77522

/-- Given an ellipse and a point as the midpoint of a chord, find the equation of the line containing the chord -/
theorem chord_line_equation (x y : ℝ) :
  (x^2 / 36 + y^2 / 9 = 1) →  -- Ellipse equation
  ∃ (x₁ y₁ x₂ y₂ : ℝ),
    (x₁^2 / 36 + y₁^2 / 9 = 1) ∧  -- Point (x₁, y₁) is on the ellipse
    (x₂^2 / 36 + y₂^2 / 9 = 1) ∧  -- Point (x₂, y₂) is on the ellipse
    ((x₁ + x₂) / 2 = 1) ∧  -- Midpoint x-coordinate is 1
    ((y₁ + y₂) / 2 = 1) →  -- Midpoint y-coordinate is 1
  ∃ (m : ℝ), m = -1/4 ∧ y - 1 = m * (x - 1) :=  -- Line equation
by sorry

end chord_line_equation_l775_77522


namespace equidistant_point_y_coordinate_l775_77524

/-- The y-coordinate of the point on the y-axis that is equidistant from A(-2, 0) and B(-1, 4) -/
theorem equidistant_point_y_coordinate :
  ∃ y : ℝ, ((-2 : ℝ) - 0)^2 + (0 - y)^2 = ((-1 : ℝ) - 0)^2 + (4 - y)^2 ∧ y = 13/8 := by
  sorry

end equidistant_point_y_coordinate_l775_77524


namespace range_of_x_l775_77573

def p (x : ℝ) : Prop := (x + 2) * (x - 2) ≤ 0
def q (x : ℝ) : Prop := x^2 - 3*x - 4 ≤ 0

theorem range_of_x : 
  (∀ x : ℝ, ¬(p x ∧ q x)) → 
  (∀ x : ℝ, p x ∨ q x) → 
  {x : ℝ | p x ∨ q x} = {x : ℝ | -2 ≤ x ∧ x < -1} ∪ {x : ℝ | 2 < x ∧ x ≤ 4} :=
sorry

end range_of_x_l775_77573


namespace walking_rate_ratio_l775_77536

/-- Given a constant distance and two different walking rates, where one rate
    results in a 14-minute journey and the other in a 12-minute journey,
    prove that the ratio of the faster rate to the slower rate is 7/6. -/
theorem walking_rate_ratio (distance : ℝ) (usual_rate new_rate : ℝ) :
  distance > 0 →
  usual_rate > 0 →
  new_rate > 0 →
  distance = usual_rate * 14 →
  distance = new_rate * 12 →
  new_rate / usual_rate = 7 / 6 := by
  sorry

end walking_rate_ratio_l775_77536


namespace min_gennadys_for_festival_l775_77597

/-- Represents the number of people with a given name -/
structure NameCount where
  alexanders : Nat
  borises : Nat
  vasilies : Nat

/-- Calculates the minimum number of Gennadys required -/
def min_gennadys (counts : NameCount) : Nat :=
  max 0 (counts.borises - 1 - counts.alexanders - counts.vasilies)

/-- The theorem stating the minimum number of Gennadys required -/
theorem min_gennadys_for_festival (counts : NameCount) 
  (h_alex : counts.alexanders = 45)
  (h_boris : counts.borises = 122)
  (h_vasily : counts.vasilies = 27) :
  min_gennadys counts = 49 := by
  sorry

#eval min_gennadys { alexanders := 45, borises := 122, vasilies := 27 }

end min_gennadys_for_festival_l775_77597


namespace max_triangle_area_l775_77521

-- Define the ellipse
def ellipse (x y : ℝ) : Prop := x^2/4 + y^2/3 = 1

-- Define the right focus
def right_focus : ℝ × ℝ := (1, 0)

-- Define a chord passing through the right focus
def chord_through_right_focus (m : ℝ) (y : ℝ) : ℝ := m * y + 1

-- Define the area of triangle PF₁Q
def triangle_area (y₁ y₂ : ℝ) : ℝ := |y₁ - y₂|

-- Theorem statement
theorem max_triangle_area :
  ∃ (max_area : ℝ), max_area = 3 ∧
  ∀ (m : ℝ) (y₁ y₂ : ℝ),
    ellipse (chord_through_right_focus m y₁) y₁ →
    ellipse (chord_through_right_focus m y₂) y₂ →
    triangle_area y₁ y₂ ≤ max_area :=
sorry

end max_triangle_area_l775_77521


namespace factorization_cubic_minus_linear_l775_77558

theorem factorization_cubic_minus_linear (x : ℝ) : x^3 - 9*x = x*(x+3)*(x-3) := by
  sorry

end factorization_cubic_minus_linear_l775_77558


namespace fraction_irreducible_l775_77582

theorem fraction_irreducible (n : ℕ) : Nat.gcd (12 * n + 1) (30 * n + 2) = 1 := by
  sorry

end fraction_irreducible_l775_77582


namespace cube_edge_ratio_l775_77585

theorem cube_edge_ratio (a b : ℝ) (h : a > 0 ∧ b > 0) :
  a^3 / b^3 = 8 / 1 → a / b = 2 / 1 := by
  sorry

end cube_edge_ratio_l775_77585


namespace geometric_sequence_sum_l775_77588

/-- A geometric sequence with positive terms -/
def GeometricSequence (a : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, q > 0 ∧ ∀ n : ℕ, a (n + 1) = a n * q

theorem geometric_sequence_sum (a : ℕ → ℝ) :
  GeometricSequence a →
  (∀ n, a n > 0) →
  a 2 = 8 →
  (2 * a 4 - a 3 = a 3 - 4 * a 5) →
  (a 1 + a 2 + a 3 + a 4 + a 5 = 31) :=
by sorry

end geometric_sequence_sum_l775_77588


namespace parallel_vectors_magnitude_l775_77566

def a : Fin 2 → ℝ := ![1, 2]
def b (y : ℝ) : Fin 2 → ℝ := ![-2, y]

theorem parallel_vectors_magnitude (y : ℝ) 
  (h : a 0 * (b y 1) = a 1 * (b y 0)) : 
  Real.sqrt ((3 * a 0 + b y 0)^2 + (3 * a 1 + b y 1)^2) = Real.sqrt 5 := by
  sorry

end parallel_vectors_magnitude_l775_77566


namespace interval_preserving_linear_l775_77506

-- Define the property that f maps intervals to intervals of the same length
def IntervalPreserving (f : ℝ → ℝ) : Prop :=
  ∀ a b, a < b → ∃ c d, c < d ∧ f '' Set.Icc a b = Set.Icc c d ∧ d - c = b - a

-- State the theorem
theorem interval_preserving_linear (f : ℝ → ℝ) (h : IntervalPreserving f) :
  ∃ c : ℝ, (∀ x, f x = x + c) ∨ (∀ x, f x = -x + c) :=
sorry

end interval_preserving_linear_l775_77506


namespace land_area_scientific_notation_l775_77516

theorem land_area_scientific_notation :
  let land_area : ℝ := 9600000
  9.6 * (10 ^ 6) = land_area := by
  sorry

end land_area_scientific_notation_l775_77516


namespace tangent_problem_l775_77575

theorem tangent_problem (α : Real) 
  (h : Real.tan (π/4 + α) = 1/2) : 
  (Real.tan α = -1/3) ∧ 
  ((Real.sin (2*α) - Real.cos α ^ 2) / (2 + Real.cos (2*α)) = (2 * Real.tan α - 1) / (3 + Real.tan α ^ 2)) := by
  sorry

end tangent_problem_l775_77575


namespace equation_solution_l775_77554

theorem equation_solution :
  let x : ℚ := -21/20
  (Real.sqrt (2 * x + 7) / Real.sqrt (4 * x + 7) = Real.sqrt 7 / 2) := by
  sorry

end equation_solution_l775_77554


namespace eccentricity_of_parametric_ellipse_l775_77569

/-- Eccentricity of an ellipse defined by parametric equations -/
theorem eccentricity_of_parametric_ellipse :
  let x : ℝ → ℝ := λ φ ↦ 3 * Real.cos φ
  let y : ℝ → ℝ := λ φ ↦ Real.sqrt 5 * Real.sin φ
  let a : ℝ := 3
  let b : ℝ := Real.sqrt 5
  let c : ℝ := Real.sqrt (a^2 - b^2)
  c / a = 2 / 3 :=
by sorry

end eccentricity_of_parametric_ellipse_l775_77569


namespace martha_jackets_bought_l775_77523

theorem martha_jackets_bought (J : ℕ) : 
  (J + J / 2 : ℕ) + (9 + 9 / 3 : ℕ) = 18 → J = 4 :=
by sorry

end martha_jackets_bought_l775_77523


namespace incircle_radius_altitude_ratio_l775_77537

/-- An isosceles right triangle with inscribed circle -/
structure IsoscelesRightTriangle where
  -- Side length of the equal sides
  side : ℝ
  -- Radius of the inscribed circle
  incircle_radius : ℝ
  -- Altitude to the hypotenuse
  altitude : ℝ
  -- The triangle is isosceles and right-angled
  is_isosceles : side = altitude * Real.sqrt 2
  -- Relationship between incircle radius and altitude
  radius_altitude_relation : incircle_radius = altitude * (Real.sqrt 2 - 1)

/-- The ratio of the inscribed circle radius to the altitude in an isosceles right triangle is √2 - 1 -/
theorem incircle_radius_altitude_ratio (t : IsoscelesRightTriangle) :
  t.incircle_radius / t.altitude = Real.sqrt 2 - 1 := by
  sorry

end incircle_radius_altitude_ratio_l775_77537


namespace sequence_property_l775_77526

def is_increasing_positive_integer_sequence (a : ℕ → ℕ) : Prop :=
  ∀ n, a n < a (n + 1)

theorem sequence_property (a : ℕ → ℕ) 
  (h1 : is_increasing_positive_integer_sequence a) 
  (h2 : ∀ n : ℕ, a (n + 2) = a (n + 1) + 2 * a n) 
  (h3 : a 5 = 52) : 
  a 7 = 212 := by
  sorry

end sequence_property_l775_77526


namespace cubic_root_polynomial_l775_77534

theorem cubic_root_polynomial (a b c : ℝ) (P : ℝ → ℝ) : 
  (a^3 + 4*a^2 + 7*a + 10 = 0) →
  (b^3 + 4*b^2 + 7*b + 10 = 0) →
  (c^3 + 4*c^2 + 7*c + 10 = 0) →
  (P a = b + c) →
  (P b = a + c) →
  (P c = a + b) →
  (P (a + b + c) = -22) →
  (∀ x, P x = 8/9*x^3 + 44/9*x^2 + 71/9*x + 2/3) :=
by
  sorry

end cubic_root_polynomial_l775_77534


namespace prob_same_color_is_zero_l775_77528

/-- Represents the number of balls of each color in the bag -/
structure BallCounts where
  green : Nat
  white : Nat
  blue : Nat
  red : Nat

/-- Calculates the total number of balls in the bag -/
def totalBalls (counts : BallCounts) : Nat :=
  counts.green + counts.white + counts.blue + counts.red

/-- Represents the number of balls to be drawn -/
def ballsToDraw : Nat := 5

/-- Calculates the probability of drawing all balls of the same color -/
def probSameColor (counts : BallCounts) : ℚ :=
  if counts.green ≥ ballsToDraw ∨ counts.white ≥ ballsToDraw ∨ 
     counts.blue ≥ ballsToDraw ∨ counts.red ≥ ballsToDraw
  then 1 / (totalBalls counts).choose ballsToDraw
  else 0

/-- Theorem: The probability of drawing 5 balls of the same color is 0 -/
theorem prob_same_color_is_zero : 
  probSameColor { green := 10, white := 9, blue := 7, red := 4 } = 0 := by
  sorry

end prob_same_color_is_zero_l775_77528


namespace point_M_coordinates_l775_77520

-- Define the curve
def curve (x : ℝ) : ℝ := 2 * x^2 + 1

-- Define the derivative of the curve
def curve_derivative (x : ℝ) : ℝ := 4 * x

-- Theorem statement
theorem point_M_coordinates :
  ∀ x y : ℝ,
  y = curve x →
  curve_derivative x = -4 →
  (x = -1 ∧ y = 3) :=
by sorry

end point_M_coordinates_l775_77520


namespace fir_trees_count_l775_77505

theorem fir_trees_count :
  ∀ n : ℕ,
  n < 25 →
  n % 11 = 0 →
  n = 11 :=
by
  sorry

end fir_trees_count_l775_77505


namespace path_length_for_73_segment_l775_77559

/-- Represents a segment divided into smaller parts with squares constructed on each part --/
structure SegmentWithSquares where
  length : ℝ
  num_parts : ℕ

/-- Calculates the length of the path along the arrows for a given segment with squares --/
def path_length (s : SegmentWithSquares) : ℝ := 3 * s.length

theorem path_length_for_73_segment : 
  let s : SegmentWithSquares := { length := 73, num_parts := 2 }
  path_length s = 219 := by sorry

end path_length_for_73_segment_l775_77559


namespace initial_dolphins_count_l775_77593

/-- The initial number of dolphins in the ocean -/
def initial_dolphins : ℕ := 65

/-- The number of dolphins joining from the river -/
def joining_dolphins : ℕ := 3 * initial_dolphins

/-- The total number of dolphins after joining -/
def total_dolphins : ℕ := 260

theorem initial_dolphins_count : initial_dolphins = 65 :=
  by sorry

end initial_dolphins_count_l775_77593


namespace point_on_line_with_vector_condition_l775_77555

/-- Given two points in a 2D plane and a third point satisfying certain conditions,
    prove that the third point has specific coordinates. -/
theorem point_on_line_with_vector_condition (P₁ P₂ P : ℝ × ℝ) : 
  P₁ = (1, 3) →
  P₂ = (4, -6) →
  (∃ t : ℝ, P = (1 - t) • P₁ + t • P₂) →  -- P is on the line P₁P₂
  (P.1 - P₁.1, P.2 - P₁.2) = 2 • (P₂.1 - P.1, P₂.2 - P.2) →  -- Vector condition
  P = (3, -3) := by
sorry

end point_on_line_with_vector_condition_l775_77555


namespace ellipse_foci_coordinates_l775_77503

/-- The coordinates of the foci of the ellipse 25x^2 + 16y^2 = 1 are (0, 3/20) and (0, -3/20) -/
theorem ellipse_foci_coordinates :
  let ellipse := {(x, y) : ℝ × ℝ | 25 * x^2 + 16 * y^2 = 1}
  ∃ (f₁ f₂ : ℝ × ℝ), 
    (f₁ ∈ ellipse ∧ f₂ ∈ ellipse) ∧ 
    (∀ p ∈ ellipse, (dist p f₁) + (dist p f₂) = (dist (1/5, 0) (-1/5, 0))) ∧
    f₁ = (0, 3/20) ∧ f₂ = (0, -3/20) :=
by sorry


end ellipse_foci_coordinates_l775_77503


namespace problem_solution_l775_77586

theorem problem_solution (x : ℝ) : (1 / (2 + 3)) * (1 / (3 + 4)) = 1 / (x + 5) → x = 30 := by
  sorry

end problem_solution_l775_77586


namespace pencil_final_price_l775_77530

/-- Given a pencil with an original cost and a discount, calculate the final price. -/
theorem pencil_final_price (original_cost discount : ℚ) 
  (h1 : original_cost = 4)
  (h2 : discount = 63 / 100) :
  original_cost - discount = 337 / 100 := by
  sorry

end pencil_final_price_l775_77530


namespace chris_box_percentage_l775_77504

theorem chris_box_percentage (k c : ℕ) (h : k = 2 * c / 3) : 
  (c : ℚ) / ((k : ℚ) + c) = 3/5 := by
sorry

end chris_box_percentage_l775_77504


namespace amy_photo_upload_l775_77564

theorem amy_photo_upload (num_albums : ℕ) (photos_per_album : ℕ) 
  (h1 : num_albums = 9)
  (h2 : photos_per_album = 20) :
  num_albums * photos_per_album = 180 := by
sorry

end amy_photo_upload_l775_77564


namespace sixth_term_is_three_l775_77577

/-- An arithmetic progression with specific properties -/
structure ArithmeticProgression where
  a : ℕ → ℝ  -- The sequence
  sum_first_three : (a 1) + (a 2) + (a 3) = 168
  diff_two_five : (a 2) - (a 5) = 42

/-- The 6th term of the arithmetic progression is 3 -/
theorem sixth_term_is_three (ap : ArithmeticProgression) : ap.a 6 = 3 := by
  sorry

end sixth_term_is_three_l775_77577


namespace water_filling_canal_is_certain_l775_77527

-- Define the type for events
inductive Event : Type
  | WaitingForRabbit : Event
  | ScoopingMoon : Event
  | WaterFillingCanal : Event
  | SeekingFishByTree : Event

-- Define what it means for an event to be certain
def isCertainEvent (e : Event) : Prop :=
  match e with
  | Event.WaterFillingCanal => true
  | _ => false

-- State the theorem
theorem water_filling_canal_is_certain :
  isCertainEvent Event.WaterFillingCanal :=
sorry

end water_filling_canal_is_certain_l775_77527


namespace exists_universal_transport_l775_77501

/-- A graph where each pair of vertices is connected by exactly one edge of either type A or type B -/
structure TransportGraph (V : Type) :=
  (edges : V → V → Bool)
  (edge_type : V → V → Bool)
  (connect : ∀ (u v : V), u ≠ v → edges u v = true)
  (unique : ∀ (u v : V), edges u v = edges v u)

/-- A path in the graph with at most two intermediate vertices -/
def ShortPath {V : Type} (g : TransportGraph V) (t : Bool) (u v : V) : Prop :=
  ∃ (w x : V), (g.edges u w ∧ g.edge_type u w = t) ∧ 
               (g.edges w x ∧ g.edge_type w x = t) ∧ 
               (g.edges x v ∧ g.edge_type x v = t)

/-- Main theorem: There exists a transport type that allows short paths between all vertices -/
theorem exists_universal_transport {V : Type} (g : TransportGraph V) :
  ∃ (t : Bool), ∀ (u v : V), u ≠ v → ShortPath g t u v :=
sorry

end exists_universal_transport_l775_77501


namespace equality_of_variables_l775_77561

theorem equality_of_variables (x₁ x₂ x₃ x₄ x₅ : ℝ) 
  (h_pos : x₁ > 0 ∧ x₂ > 0 ∧ x₃ > 0 ∧ x₄ > 0 ∧ x₅ > 0)
  (h₁ : (x₁^2 - x₃*x₅)*(x₂^2 - x₃*x₅) ≤ 0)
  (h₂ : (x₂^2 - x₄*x₁)*(x₃^2 - x₄*x₁) ≤ 0)
  (h₃ : (x₃^2 - x₅*x₂)*(x₄^2 - x₅*x₂) ≤ 0)
  (h₄ : (x₄^2 - x₁*x₃)*(x₅^2 - x₁*x₃) ≤ 0)
  (h₅ : (x₅^2 - x₂*x₄)*(x₁^2 - x₂*x₄) ≤ 0) :
  x₁ = x₂ ∧ x₂ = x₃ ∧ x₃ = x₄ ∧ x₄ = x₅ :=
by sorry

end equality_of_variables_l775_77561


namespace cubic_roots_sum_of_squares_l775_77517

theorem cubic_roots_sum_of_squares (a b c t : ℝ) : 
  (∀ x, x^3 - 12*x^2 + 20*x - 2 = 0 ↔ (x = a ∨ x = b ∨ x = c)) →
  t = Real.sqrt a + Real.sqrt b + Real.sqrt c →
  t^4 - 24*t^2 - 16*t = -96 - 8*t := by
  sorry

end cubic_roots_sum_of_squares_l775_77517


namespace trigonometric_expression_equality_l775_77598

theorem trigonometric_expression_equality : 
  (Real.sin (15 * π / 180) * Real.cos (20 * π / 180) + 
   Real.cos (165 * π / 180) * Real.cos (115 * π / 180)) / 
  (Real.sin (25 * π / 180) * Real.cos (5 * π / 180) + 
   Real.cos (155 * π / 180) * Real.cos (95 * π / 180)) = 
  2 * (Real.sin (35 * π / 180) - Real.sin (10 * π / 180)) / 
  (1 - Real.sqrt 3) := by
sorry

end trigonometric_expression_equality_l775_77598


namespace two_distinct_roots_condition_l775_77557

theorem two_distinct_roots_condition (k : ℝ) : 
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ x₁^2 - 2*x₁ + k = 0 ∧ x₂^2 - 2*x₂ + k = 0) → k < 1 :=
by sorry

end two_distinct_roots_condition_l775_77557


namespace inequality_proof_l775_77589

theorem inequality_proof (x y z : ℝ) 
  (pos_x : x > 0) (pos_y : y > 0) (pos_z : z > 0) 
  (sum_one : x + y + z = 1) : 
  (x^2 + y^2) / z + (y^2 + z^2) / x + (z^2 + x^2) / y ≥ 2 := by
  sorry

end inequality_proof_l775_77589


namespace f_strictly_decreasing_iff_a_in_range_l775_77574

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x < 0 then a^x else (a - 4) * x + 3 * a

theorem f_strictly_decreasing_iff_a_in_range (a : ℝ) :
  (∀ x₁ x₂ : ℝ, x₁ ≠ x₂ → (f a x₁ - f a x₂) * (x₁ - x₂) < 0) ↔
  0 < a ∧ a ≤ 1/3 :=
sorry

end f_strictly_decreasing_iff_a_in_range_l775_77574


namespace units_digit_52_cubed_plus_29_cubed_l775_77548

theorem units_digit_52_cubed_plus_29_cubed : (52^3 + 29^3) % 10 = 7 := by
  sorry

end units_digit_52_cubed_plus_29_cubed_l775_77548


namespace sqrt_6_over_3_properties_l775_77502

theorem sqrt_6_over_3_properties : ∃ x : ℝ, x = (Real.sqrt 6) / 3 ∧ 0 < x ∧ x < 1 ∧ Irrational x := by
  sorry

end sqrt_6_over_3_properties_l775_77502


namespace maxwell_twice_sister_age_sister_current_age_l775_77539

/-- Maxwell's current age -/
def maxwell_age : ℕ := 6

/-- Maxwell's sister's current age -/
def sister_age : ℕ := 2

/-- In 2 years, Maxwell will be twice his sister's age -/
theorem maxwell_twice_sister_age : 
  maxwell_age + 2 = 2 * (sister_age + 2) := by sorry

/-- Proof that Maxwell's sister is currently 2 years old -/
theorem sister_current_age : sister_age = 2 := by sorry

end maxwell_twice_sister_age_sister_current_age_l775_77539


namespace bobs_walking_rate_l775_77556

/-- Proves that Bob's walking rate is 7 miles per hour given the conditions of the problem -/
theorem bobs_walking_rate 
  (total_distance : ℝ) 
  (yolanda_rate : ℝ) 
  (bob_distance : ℝ) 
  (head_start : ℝ) 
  (h1 : total_distance = 65) 
  (h2 : yolanda_rate = 5) 
  (h3 : bob_distance = 35) 
  (h4 : head_start = 1) : 
  (bob_distance / (total_distance - yolanda_rate * head_start - bob_distance) * yolanda_rate) = 7 :=
sorry

end bobs_walking_rate_l775_77556


namespace seashells_sum_l775_77541

/-- The number of seashells Joan found on the beach -/
def total_seashells : ℕ := 70

/-- The number of seashells Joan gave to Sam -/
def seashells_given : ℕ := 43

/-- The number of seashells Joan has left -/
def seashells_left : ℕ := 27

/-- Theorem stating that the total number of seashells is the sum of those given away and those left -/
theorem seashells_sum : total_seashells = seashells_given + seashells_left := by
  sorry

end seashells_sum_l775_77541


namespace blood_donor_selection_l775_77525

theorem blood_donor_selection (type_O : Nat) (type_A : Nat) (type_B : Nat) (type_AB : Nat)
  (h1 : type_O = 10)
  (h2 : type_A = 5)
  (h3 : type_B = 8)
  (h4 : type_AB = 3) :
  type_O * type_A * type_B * type_AB = 1200 := by
  sorry

end blood_donor_selection_l775_77525


namespace student_score_l775_77578

theorem student_score (total_questions : Nat) (correct_responses : Nat) : 
  total_questions = 100 →
  correct_responses = 88 →
  let incorrect_responses := total_questions - correct_responses
  let score := correct_responses - 2 * incorrect_responses
  score = 64 := by
sorry

end student_score_l775_77578


namespace sequential_no_conditional_l775_77533

-- Define the structures
inductive FlowchartStructure
  | Sequential
  | Loop
  | If
  | Until

-- Define a predicate for structures that generally contain a conditional judgment box
def hasConditionalJudgment : FlowchartStructure → Prop
  | FlowchartStructure.Sequential => False
  | FlowchartStructure.Loop => True
  | FlowchartStructure.If => True
  | FlowchartStructure.Until => True

theorem sequential_no_conditional : 
  ∀ (s : FlowchartStructure), ¬hasConditionalJudgment s ↔ s = FlowchartStructure.Sequential :=
by sorry

end sequential_no_conditional_l775_77533


namespace wrapping_paper_area_theorem_l775_77595

/-- The area of wrapping paper required to wrap a rectangular box -/
def wrapping_paper_area (l w h : ℝ) : ℝ :=
  l * w + 2 * l * h + 2 * w * h + 4 * h^2

/-- Theorem stating the area of wrapping paper required for a rectangular box -/
theorem wrapping_paper_area_theorem (l w h : ℝ) (h1 : l > w) (h2 : l > 0) (h3 : w > 0) (h4 : h > 0) :
  let box_base_area := l * w
  let box_side_area := 2 * (l * h + w * h)
  let corner_area := 4 * h^2
  box_base_area + box_side_area + corner_area = wrapping_paper_area l w h :=
by
  sorry

#check wrapping_paper_area_theorem

end wrapping_paper_area_theorem_l775_77595


namespace nth_equation_proof_l775_77568

theorem nth_equation_proof (n : ℕ) : (n + 1) * (n^2 - n + 1) - 1 = n^3 := by
  sorry

end nth_equation_proof_l775_77568


namespace abs_eq_sqrt_sq_l775_77514

theorem abs_eq_sqrt_sq (x : ℝ) : |x| = Real.sqrt (x^2) := by sorry

end abs_eq_sqrt_sq_l775_77514


namespace product_and_reciprocal_sum_l775_77599

theorem product_and_reciprocal_sum (a b : ℝ) (ha : a > 0) (hb : b > 0) 
  (h1 : a * b = 4) (h2 : 1 / a = 3 / b) : a + b = 8 * Real.sqrt 3 / 3 := by
  sorry

end product_and_reciprocal_sum_l775_77599


namespace sum_of_numbers_l775_77507

/-- Given three numbers A, B, and C, where A is a three-digit number and B and C are two-digit numbers,
    if the sum of numbers containing the digit seven is 208 and the sum of numbers containing the digit three is 76,
    then the sum of A, B, and C is 247. -/
theorem sum_of_numbers (A B C : ℕ) : 
  100 ≤ A ∧ A < 1000 ∧   -- A is a three-digit number
  10 ≤ B ∧ B < 100 ∧     -- B is a two-digit number
  10 ≤ C ∧ C < 100 ∧     -- C is a two-digit number
  ((A.repr.contains '7' ∨ B.repr.contains '7' ∨ C.repr.contains '7') → A + B + C = 208) ∧  -- Sum of numbers with 7
  (B.repr.contains '3' ∧ C.repr.contains '3' → B + C = 76)  -- Sum of numbers with 3
  → A + B + C = 247 := by
sorry

end sum_of_numbers_l775_77507


namespace largest_divisor_of_n_l775_77594

theorem largest_divisor_of_n (n : ℕ) (h1 : n > 0) (h2 : 450 ∣ n^2) : 
  ∀ d : ℕ, d > 0 ∧ d ∣ n → d ≤ 30 :=
by sorry

end largest_divisor_of_n_l775_77594


namespace painted_cube_theorem_l775_77552

theorem painted_cube_theorem (n : ℕ) (h1 : n > 2) :
  (12 * (n - 2) : ℝ) = (n - 2)^3 → n = 2 * Real.sqrt 3 + 2 := by
  sorry

end painted_cube_theorem_l775_77552


namespace max_eggs_per_basket_l775_77570

def total_red_eggs : ℕ := 30
def total_blue_eggs : ℕ := 45
def min_eggs_per_basket : ℕ := 5

theorem max_eggs_per_basket :
  ∃ (n : ℕ), n ≥ min_eggs_per_basket ∧
             n ∣ total_red_eggs ∧
             n ∣ total_blue_eggs ∧
             ∀ (m : ℕ), m ≥ min_eggs_per_basket ∧
                        m ∣ total_red_eggs ∧
                        m ∣ total_blue_eggs →
                        m ≤ n :=
by sorry

end max_eggs_per_basket_l775_77570


namespace lcm_48_180_l775_77547

theorem lcm_48_180 : Nat.lcm 48 180 = 720 := by sorry

end lcm_48_180_l775_77547


namespace cube_volume_from_diagonal_l775_77544

/-- The volume of a cube with space diagonal 5√3 is 125 -/
theorem cube_volume_from_diagonal : 
  ∀ (s : ℝ), s > 0 → s * Real.sqrt 3 = 5 * Real.sqrt 3 → s^3 = 125 := by
  sorry

end cube_volume_from_diagonal_l775_77544


namespace exists_x_satisfying_conditions_l775_77543

theorem exists_x_satisfying_conditions : ∃ x : ℝ,
  ({1, 3, x^2 - 2*x} : Set ℝ) = {1, 3, 0} ∧
  ({1, |2*x - 1|} : Set ℝ) = {1, 3} := by
sorry

end exists_x_satisfying_conditions_l775_77543


namespace scientific_notation_correct_l775_77513

/-- Scientific notation representation of a number -/
structure ScientificNotation where
  coefficient : ℝ
  exponent : ℤ
  is_valid : 1 ≤ |coefficient| ∧ |coefficient| < 10

/-- The number to be expressed in scientific notation -/
def original_number : ℕ := 17600

/-- The proposed scientific notation representation -/
def proposed_notation : ScientificNotation :=
  { coefficient := 1.76
    exponent := 4
    is_valid := by sorry }

/-- Theorem stating that the proposed notation correctly represents the original number -/
theorem scientific_notation_correct :
  (proposed_notation.coefficient * (10 : ℝ) ^ proposed_notation.exponent) = original_number := by
  sorry

end scientific_notation_correct_l775_77513


namespace four_level_pyramid_books_l775_77529

def pyramid_books (levels : ℕ) (ratio : ℝ) (top_level_books : ℕ) : ℝ :=
  let rec sum_levels (n : ℕ) : ℝ :=
    if n = 0 then 0
    else (top_level_books : ℝ) * (ratio ^ (n - 1)) + sum_levels (n - 1)
  sum_levels levels

theorem four_level_pyramid_books :
  pyramid_books 4 (1 / 0.8) 64 = 369 := by sorry

end four_level_pyramid_books_l775_77529


namespace eight_pow_plus_six_div_seven_l775_77546

theorem eight_pow_plus_six_div_seven (n : ℕ) : 
  7 ∣ (8^n + 6) := by sorry

end eight_pow_plus_six_div_seven_l775_77546


namespace equal_tire_usage_l775_77579

/-- Represents the usage of tires on a car -/
structure TireUsage where
  total_tires : ℕ
  active_tires : ℕ
  total_miles : ℕ
  tire_miles : ℕ

/-- Theorem stating the correct tire usage for the given scenario -/
theorem equal_tire_usage (usage : TireUsage) 
  (h1 : usage.total_tires = 5)
  (h2 : usage.active_tires = 4)
  (h3 : usage.total_miles = 45000)
  (h4 : usage.tire_miles = usage.total_miles * usage.active_tires / usage.total_tires) :
  usage.tire_miles = 36000 := by
  sorry

#check equal_tire_usage

end equal_tire_usage_l775_77579


namespace square_roots_sum_l775_77510

theorem square_roots_sum (x y : ℝ) : 
  x^2 = 16 → y^2 = 9 → x^2 + y^2 + x + 2023 = 2052 := by
  sorry

end square_roots_sum_l775_77510
