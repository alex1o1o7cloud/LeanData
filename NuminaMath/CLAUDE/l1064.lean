import Mathlib

namespace NUMINAMATH_CALUDE_polynomial_sum_theorem_l1064_106488

variable (x : ℝ)

def f (x : ℝ) : ℝ := x^4 - 3*x^2 + x - 1

theorem polynomial_sum_theorem (g : ℝ → ℝ) 
  (h1 : ∀ x, f x + g x = 3*x^2 - 2) :
  g = λ x => -x^4 + 6*x^2 - x - 1 := by
sorry

end NUMINAMATH_CALUDE_polynomial_sum_theorem_l1064_106488


namespace NUMINAMATH_CALUDE_larger_sphere_radius_l1064_106431

theorem larger_sphere_radius (r : ℝ) (n : ℕ) (h : r = 2 ∧ n = 6) :
  (n * (4 / 3 * π * r^3) = 4 / 3 * π * (2 * Real.rpow 3 (1/3))^3) :=
by sorry

end NUMINAMATH_CALUDE_larger_sphere_radius_l1064_106431


namespace NUMINAMATH_CALUDE_q_zero_value_l1064_106424

-- Define polynomials p, q, and r
variable (p q r : ℝ → ℝ)

-- Define the relationship between p, q, and r
axiom relation : ∀ x, r x = p x * q x + 2

-- Define the constant terms of p and r
axiom p_constant : p 0 = 6
axiom r_constant : r 0 = 5

-- Theorem to prove
theorem q_zero_value : q 0 = 1/2 := by sorry

end NUMINAMATH_CALUDE_q_zero_value_l1064_106424


namespace NUMINAMATH_CALUDE_slope_angle_of_line_l1064_106497

/-- The slope angle of the line x + √3y - 3 = 0 is 5π/6 -/
theorem slope_angle_of_line (x y : ℝ) : 
  x + Real.sqrt 3 * y - 3 = 0 → 
  ∃ α : ℝ, α = 5 * Real.pi / 6 ∧ 
    (Real.tan α = -(1 / Real.sqrt 3) ∨ Real.tan α = -(Real.sqrt 3 / 3)) :=
by sorry

end NUMINAMATH_CALUDE_slope_angle_of_line_l1064_106497


namespace NUMINAMATH_CALUDE_sum_of_angles_in_triangle_l1064_106480

-- Define a triangle
structure Triangle where
  A : Point
  B : Point
  C : Point

-- Define angles in a triangle
def angle (t : Triangle) (v : Fin 3) : ℝ :=
  sorry

-- State the theorem
theorem sum_of_angles_in_triangle (t : Triangle) : 
  angle t 0 + angle t 1 + angle t 2 = 180 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_angles_in_triangle_l1064_106480


namespace NUMINAMATH_CALUDE_birthday_problem_l1064_106403

theorem birthday_problem (n : ℕ) (m : ℕ) (h1 : n = 400) (h2 : m = 365) :
  ∃ (i j : ℕ), i ≠ j ∧ i ≤ n ∧ j ≤ n ∧ (i.mod m = j.mod m) :=
sorry

end NUMINAMATH_CALUDE_birthday_problem_l1064_106403


namespace NUMINAMATH_CALUDE_tommy_steak_purchase_l1064_106423

/-- The number of steaks needed for a family meal --/
def steaks_needed (family_members : ℕ) (pounds_per_member : ℕ) (ounces_per_steak : ℕ) : ℕ :=
  let total_pounds := family_members * pounds_per_member
  let total_ounces := total_pounds * 16
  total_ounces / ounces_per_steak

theorem tommy_steak_purchase :
  steaks_needed 5 1 20 = 4 := by
  sorry

end NUMINAMATH_CALUDE_tommy_steak_purchase_l1064_106423


namespace NUMINAMATH_CALUDE_fence_overlap_calculation_l1064_106485

theorem fence_overlap_calculation (num_planks : ℕ) (plank_length : ℝ) (total_length : ℝ) 
  (h1 : num_planks = 25)
  (h2 : plank_length = 30)
  (h3 : total_length = 690) :
  ∃ overlap : ℝ, 
    overlap = 2.5 ∧ 
    total_length = (13 * plank_length) + (12 * (plank_length - 2 * overlap)) :=
by sorry

end NUMINAMATH_CALUDE_fence_overlap_calculation_l1064_106485


namespace NUMINAMATH_CALUDE_power_tower_at_three_l1064_106498

theorem power_tower_at_three : (3^3)^(3^(3^3)) = 27^(3^27) := by sorry

end NUMINAMATH_CALUDE_power_tower_at_three_l1064_106498


namespace NUMINAMATH_CALUDE_specific_rental_cost_l1064_106452

/-- Calculates the total cost of a car rental given the daily rate, per-mile rate, number of days, and miles driven. -/
def carRentalCost (dailyRate perMileRate : ℚ) (days miles : ℕ) : ℚ :=
  dailyRate * days + perMileRate * miles

/-- Theorem stating that for the given rental conditions, the total cost is $162.5 -/
theorem specific_rental_cost :
  carRentalCost 25 0.25 3 350 = 162.5 := by
  sorry

end NUMINAMATH_CALUDE_specific_rental_cost_l1064_106452


namespace NUMINAMATH_CALUDE_max_boxes_theorem_l1064_106468

def lifting_capacities : List Nat := [30, 45, 50, 60, 75, 100, 120]
def box_weights : List Nat := [15, 25, 35, 45, 55, 70, 80, 95, 110]

def max_boxes_lifted (capacities : List Nat) (weights : List Nat) : Nat :=
  sorry

theorem max_boxes_theorem :
  max_boxes_lifted lifting_capacities box_weights = 7 := by
  sorry

end NUMINAMATH_CALUDE_max_boxes_theorem_l1064_106468


namespace NUMINAMATH_CALUDE_given_point_in_fourth_quadrant_l1064_106428

/-- A point in the Cartesian plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- Definition of the fourth quadrant -/
def is_in_fourth_quadrant (p : Point) : Prop :=
  p.x > 0 ∧ p.y < 0

/-- The given point -/
def given_point : Point :=
  { x := 1, y := -2 }

/-- Theorem: The given point is in the fourth quadrant -/
theorem given_point_in_fourth_quadrant :
  is_in_fourth_quadrant given_point := by
  sorry

end NUMINAMATH_CALUDE_given_point_in_fourth_quadrant_l1064_106428


namespace NUMINAMATH_CALUDE_square_sum_existence_l1064_106421

theorem square_sum_existence (k : ℤ) 
  (h1 : 2 * k + 1 > 17) 
  (h2 : ∃ m : ℤ, 6 * k + 1 = m^2) : 
  ∃ b c : ℤ, 
    b > 0 ∧ 
    c > 0 ∧ 
    b ≠ c ∧ 
    (∃ w : ℤ, (2 * k + 1 + b) = w^2) ∧ 
    (∃ x : ℤ, (2 * k + 1 + c) = x^2) ∧ 
    (∃ y : ℤ, (b + c) = y^2) ∧ 
    (∃ z : ℤ, (2 * k + 1 + b + c) = z^2) :=
sorry

end NUMINAMATH_CALUDE_square_sum_existence_l1064_106421


namespace NUMINAMATH_CALUDE_lena_kevin_ratio_l1064_106434

-- Define the initial number of candy bars for Lena
def lena_initial : ℕ := 16

-- Define the number of additional candy bars Lena needs
def additional_candies : ℕ := 5

-- Define the relationship between Lena's and Nicole's candy bars
def lena_nicole_diff : ℕ := 5

-- Define the relationship between Nicole's and Kevin's candy bars
def nicole_kevin_diff : ℕ := 4

-- Calculate Nicole's candy bars
def nicole_candies : ℕ := lena_initial - lena_nicole_diff

-- Calculate Kevin's candy bars
def kevin_candies : ℕ := nicole_candies - nicole_kevin_diff

-- Calculate Lena's final number of candy bars
def lena_final : ℕ := lena_initial + additional_candies

-- Theorem stating the ratio of Lena's final candy bars to Kevin's candy bars
theorem lena_kevin_ratio : 
  lena_final / kevin_candies = 3 ∧ lena_final % kevin_candies = 0 := by
  sorry

end NUMINAMATH_CALUDE_lena_kevin_ratio_l1064_106434


namespace NUMINAMATH_CALUDE_problem_statement_l1064_106478

/-- Given real numbers x and y satisfying x + y/4 = 1, prove:
    1. If |7-y| < 2x+3, then -1 < x < 0
    2. If x > 0 and y > 0, then sqrt(xy) ≥ xy -/
theorem problem_statement (x y : ℝ) (h1 : x + y / 4 = 1) :
  (∀ h2 : |7 - y| < 2*x + 3, -1 < x ∧ x < 0) ∧
  (∀ h3 : x > 0, ∀ h4 : y > 0, Real.sqrt (x * y) ≥ x * y) := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l1064_106478


namespace NUMINAMATH_CALUDE_digit_sum_problem_l1064_106464

theorem digit_sum_problem (p q r : ℕ) : 
  p ≠ q ∧ p ≠ r ∧ q ≠ r ∧
  0 < p ∧ p < 10 ∧ 0 < q ∧ q < 10 ∧ 0 < r ∧ r < 10 ∧
  (10 * p + q) * (10 * p + r) = 221 →
  p + q + r = 11 := by
sorry

end NUMINAMATH_CALUDE_digit_sum_problem_l1064_106464


namespace NUMINAMATH_CALUDE_expand_expression_l1064_106451

theorem expand_expression (x : ℝ) : (x - 3) * (x + 3) * (x^2 + 9) = x^4 - 81 := by
  sorry

end NUMINAMATH_CALUDE_expand_expression_l1064_106451


namespace NUMINAMATH_CALUDE_maintenance_check_increase_l1064_106479

theorem maintenance_check_increase (original_days : ℝ) (new_days : ℝ) 
  (h1 : original_days = 30) 
  (h2 : new_days = 45) : 
  ((new_days - original_days) / original_days) * 100 = 50 := by
  sorry

end NUMINAMATH_CALUDE_maintenance_check_increase_l1064_106479


namespace NUMINAMATH_CALUDE_total_spending_over_four_years_l1064_106469

/-- The annual toy spending of three friends over four years. -/
def annual_toy_spending (trevor_spending : ℕ) (reed_diff : ℕ) (quinn_ratio : ℕ) (years : ℕ) : ℕ :=
  let reed_spending := trevor_spending - reed_diff
  let quinn_spending := reed_spending / quinn_ratio
  (trevor_spending + reed_spending + quinn_spending) * years

/-- Theorem stating the total spending of three friends over four years. -/
theorem total_spending_over_four_years :
  annual_toy_spending 80 20 2 4 = 680 := by
  sorry

#eval annual_toy_spending 80 20 2 4

end NUMINAMATH_CALUDE_total_spending_over_four_years_l1064_106469


namespace NUMINAMATH_CALUDE_max_value_under_constraint_l1064_106496

theorem max_value_under_constraint (x y : ℝ) :
  (|x| + |y| ≤ 1) → (x + 2*y ≤ 2) :=
by sorry

end NUMINAMATH_CALUDE_max_value_under_constraint_l1064_106496


namespace NUMINAMATH_CALUDE_probability_of_C_l1064_106435

-- Define the wheel with four parts
inductive WheelPart : Type
| A
| B
| C
| D

-- Define the probability function
def probability : WheelPart → ℚ
| WheelPart.A => 1/4
| WheelPart.B => 1/3
| WheelPart.C => 1/4  -- This is what we want to prove
| WheelPart.D => 1/6

-- State the theorem
theorem probability_of_C : probability WheelPart.C = 1/4 := by
  -- The sum of all probabilities must equal 1
  have sum_of_probabilities : 
    probability WheelPart.A + probability WheelPart.B + 
    probability WheelPart.C + probability WheelPart.D = 1 := by sorry
  
  -- Proof goes here
  sorry

end NUMINAMATH_CALUDE_probability_of_C_l1064_106435


namespace NUMINAMATH_CALUDE_function_properties_l1064_106495

noncomputable section

-- Define the function f(x)
def f (a : ℝ) (x : ℝ) : ℝ := Real.exp x - a * Real.log x - a

-- State the theorem
theorem function_properties (a : ℝ) (h_a : a > 0) :
  -- Part 1
  (∃ x₀ : ℝ, x₀ > 0 ∧ ∀ x > 0, f e x ≥ f e x₀) ∧
  (∀ M : ℝ, ∃ x > 0, f e x > M) ∧
  -- Part 2
  (∀ x₁ x₂ : ℝ, 0 < x₁ ∧ x₁ < x₂ ∧ f a x₁ = 0 ∧ f a x₂ = 0 →
    1/a < x₁ ∧ x₁ < 1 ∧ 1 < x₂ ∧ x₂ < a) ∧
  -- Part 3
  (∀ x : ℝ, x > 0 → Real.exp (2*x - 2) - Real.exp (x - 1) * Real.log x - x ≥ 0) :=
by sorry

end NUMINAMATH_CALUDE_function_properties_l1064_106495


namespace NUMINAMATH_CALUDE_bryan_has_more_candies_l1064_106456

/-- Given that Bryan has 50 candies and Ben has 20 candies, 
    prove that Bryan has 30 more candies than Ben. -/
theorem bryan_has_more_candies :
  let bryan_candies : ℕ := 50
  let ben_candies : ℕ := 20
  bryan_candies - ben_candies = 30 := by
  sorry

end NUMINAMATH_CALUDE_bryan_has_more_candies_l1064_106456


namespace NUMINAMATH_CALUDE_lukas_points_l1064_106472

/-- Given a basketball player's average points per game and a number of games,
    calculates the total points scored. -/
def total_points (avg_points : ℕ) (num_games : ℕ) : ℕ :=
  avg_points * num_games

/-- Proves that a player averaging 12 points per game scores 60 points in 5 games. -/
theorem lukas_points : total_points 12 5 = 60 := by
  sorry

end NUMINAMATH_CALUDE_lukas_points_l1064_106472


namespace NUMINAMATH_CALUDE_math_club_teams_l1064_106483

/-- The number of ways to choose k items from n items -/
def choose (n k : ℕ) : ℕ := (Nat.factorial n) / (Nat.factorial k * Nat.factorial (n - k))

/-- The number of girls in the math club -/
def num_girls : ℕ := 4

/-- The number of boys in the math club -/
def num_boys : ℕ := 6

/-- The number of girls to be selected for each team -/
def girls_per_team : ℕ := 2

/-- The number of boys to be selected for each team -/
def boys_per_team : ℕ := 2

theorem math_club_teams : 
  (choose num_girls girls_per_team) * (choose num_boys boys_per_team) = 90 := by
  sorry

end NUMINAMATH_CALUDE_math_club_teams_l1064_106483


namespace NUMINAMATH_CALUDE_horner_method_v2_l1064_106405

def horner_polynomial (x : ℝ) : ℝ := x^6 + 6*x^4 + 9*x^2 + 208

def horner_v2 (x : ℝ) : ℝ := 
  let v0 : ℝ := 1
  let v1 : ℝ := v0 * x
  v1 * x + 6

theorem horner_method_v2 : 
  horner_v2 (-4) = 22 :=
sorry

end NUMINAMATH_CALUDE_horner_method_v2_l1064_106405


namespace NUMINAMATH_CALUDE_squarefree_term_existence_l1064_106467

/-- A positive integer is squarefree if it's not divisible by any square number greater than 1 -/
def IsSquarefree (n : ℕ) : Prop :=
  ∀ d : ℕ, d > 1 → d * d ∣ n → d = 1

/-- An arithmetic sequence of positive integers -/
def IsArithmeticSeq (a : ℕ → ℕ) : Prop :=
  ∃ d : ℕ, ∀ n : ℕ, a (n + 1) = a n + d

theorem squarefree_term_existence :
  ∃ C : ℝ, C > 0 ∧
    ∀ a : ℕ → ℕ, IsArithmeticSeq a →
      IsSquarefree (Nat.gcd (a 1) (a 2)) →
        ∃ m : ℕ, m > 0 ∧ m ≤ ⌊C * (a 2)^2⌋ ∧ IsSquarefree (a m) :=
sorry

end NUMINAMATH_CALUDE_squarefree_term_existence_l1064_106467


namespace NUMINAMATH_CALUDE_dogwood_trees_planted_today_l1064_106471

/-- The number of dogwood trees planted today in the park. -/
def trees_planted_today : ℕ := sorry

/-- The current number of dogwood trees in the park. -/
def current_trees : ℕ := 7

/-- The number of dogwood trees to be planted tomorrow. -/
def trees_planted_tomorrow : ℕ := 2

/-- The total number of dogwood trees after planting is finished. -/
def total_trees : ℕ := 12

theorem dogwood_trees_planted_today :
  trees_planted_today = 3 :=
by
  have h : current_trees + trees_planted_today + trees_planted_tomorrow = total_trees := sorry
  sorry

end NUMINAMATH_CALUDE_dogwood_trees_planted_today_l1064_106471


namespace NUMINAMATH_CALUDE_vector_expression_l1064_106474

/-- Given vectors a, b, and c in ℝ², prove that c = 2a - b -/
theorem vector_expression (a b c : ℝ × ℝ) 
  (ha : a = (1, 2)) 
  (hb : b = (-2, 3)) 
  (hc : c = (4, 1)) : 
  c = (2 : ℝ) • a - b := by sorry

end NUMINAMATH_CALUDE_vector_expression_l1064_106474


namespace NUMINAMATH_CALUDE_base_b_not_divisible_by_five_l1064_106455

theorem base_b_not_divisible_by_five (b : ℤ) : b ∈ ({5, 6, 7, 9, 10} : Set ℤ) →
  (3 * b^3 - 2 * b^2 - b - 2) % 5 ≠ 0 ↔ b = 5 ∨ b = 10 := by
  sorry

end NUMINAMATH_CALUDE_base_b_not_divisible_by_five_l1064_106455


namespace NUMINAMATH_CALUDE_square_sum_equals_fifty_l1064_106482

theorem square_sum_equals_fifty (x y : ℝ) 
  (h1 : x + y = -10) 
  (h2 : x = 25 / y) : 
  x^2 + y^2 = 50 := by
sorry

end NUMINAMATH_CALUDE_square_sum_equals_fifty_l1064_106482


namespace NUMINAMATH_CALUDE_m_plus_n_values_l1064_106432

theorem m_plus_n_values (m n : ℤ) (hm : m = 3) (hn : |n| = 1) :
  m + n = 4 ∨ m + n = 2 := by
sorry

end NUMINAMATH_CALUDE_m_plus_n_values_l1064_106432


namespace NUMINAMATH_CALUDE_michaels_subtraction_l1064_106450

theorem michaels_subtraction (a b : ℕ) (h1 : a = 40) (h2 : b = 39) :
  a^2 - b^2 = 79 := by
  sorry

end NUMINAMATH_CALUDE_michaels_subtraction_l1064_106450


namespace NUMINAMATH_CALUDE_expression_evaluation_l1064_106407

theorem expression_evaluation : 101^3 + 3*(101^2) + 3*101 + 1 = 1061208 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l1064_106407


namespace NUMINAMATH_CALUDE_product_of_larger_numbers_l1064_106499

theorem product_of_larger_numbers (A B C : ℝ) 
  (h1 : B - A = C - B) 
  (h2 : A * B = 85) 
  (h3 : B = 10) : 
  B * C = 115 := by
sorry

end NUMINAMATH_CALUDE_product_of_larger_numbers_l1064_106499


namespace NUMINAMATH_CALUDE_championship_outcomes_l1064_106442

theorem championship_outcomes (num_students : ℕ) (num_events : ℕ) : 
  num_students = 4 → num_events = 3 → (num_students ^ num_events : ℕ) = 64 := by
  sorry

#check championship_outcomes

end NUMINAMATH_CALUDE_championship_outcomes_l1064_106442


namespace NUMINAMATH_CALUDE_remaining_income_percentage_l1064_106457

-- Define the percentages as fractions
def food_percent : ℚ := 35 / 100
def education_percent : ℚ := 25 / 100
def transportation_percent : ℚ := 15 / 100
def medical_percent : ℚ := 10 / 100
def rent_percent_of_remaining : ℚ := 80 / 100

-- Theorem statement
theorem remaining_income_percentage :
  let initial_expenses := food_percent + education_percent + transportation_percent + medical_percent
  let remaining_after_initial := 1 - initial_expenses
  let rent_expense := rent_percent_of_remaining * remaining_after_initial
  1 - (initial_expenses + rent_expense) = 3 / 100 := by
  sorry

end NUMINAMATH_CALUDE_remaining_income_percentage_l1064_106457


namespace NUMINAMATH_CALUDE_min_distance_to_line_l1064_106487

/-- Given the line x + 2y = 1, the minimum value of x^2 + y^2 is 1/5 -/
theorem min_distance_to_line (x y : ℝ) (h : x + 2*y = 1) : 
  ∃ (min : ℝ), min = 1/5 ∧ ∀ (x' y' : ℝ), x' + 2*y' = 1 → x'^2 + y'^2 ≥ min :=
sorry

end NUMINAMATH_CALUDE_min_distance_to_line_l1064_106487


namespace NUMINAMATH_CALUDE_miles_traveled_l1064_106486

/-- Represents the efficiency of a car in miles per gallon -/
def miles_per_gallon : ℝ := 25

/-- Represents the cost of gas in dollars per gallon -/
def dollars_per_gallon : ℝ := 5

/-- Represents the amount of money spent on gas in dollars -/
def money_spent : ℝ := 25

/-- Theorem stating that given the efficiency of the car and the cost of gas,
    $25 worth of gas will allow the car to travel 125 miles -/
theorem miles_traveled (mpg : ℝ) (dpg : ℝ) (spent : ℝ) :
  mpg = miles_per_gallon →
  dpg = dollars_per_gallon →
  spent = money_spent →
  (spent / dpg) * mpg = 125 := by
  sorry

end NUMINAMATH_CALUDE_miles_traveled_l1064_106486


namespace NUMINAMATH_CALUDE_fruit_shop_problem_l1064_106441

/-- The price per kilogram of apples in yuan -/
def apple_price : ℝ := 8

/-- The price per kilogram of pears in yuan -/
def pear_price : ℝ := 6

/-- The maximum number of kilograms of apples that can be purchased -/
def max_apples : ℝ := 5

theorem fruit_shop_problem :
  (1 * apple_price + 3 * pear_price = 26) ∧
  (2 * apple_price + 1 * pear_price = 22) ∧
  (∀ x y : ℝ, x + y = 15 → x * apple_price + y * pear_price ≤ 100 → x ≤ max_apples) :=
by sorry

end NUMINAMATH_CALUDE_fruit_shop_problem_l1064_106441


namespace NUMINAMATH_CALUDE_min_value_fraction_l1064_106404

theorem min_value_fraction (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0)
  (h : x - 2*y + 3*z = 0) : y^2 / (x*z) ≥ 3 :=
by sorry

end NUMINAMATH_CALUDE_min_value_fraction_l1064_106404


namespace NUMINAMATH_CALUDE_base_to_lateral_area_ratio_l1064_106419

/-- Represents a cone where the height is equal to the diameter of its circular base -/
structure SpecialCone where
  r : ℝ  -- radius of the base
  h : ℝ  -- height of the cone
  h_eq_diam : h = 2 * r  -- condition that height equals diameter

/-- The ratio of base area to lateral area for a SpecialCone is 1:√5 -/
theorem base_to_lateral_area_ratio (cone : SpecialCone) :
  (π * cone.r^2) / (π * cone.r * Real.sqrt (cone.h^2 + cone.r^2)) = 1 / Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_base_to_lateral_area_ratio_l1064_106419


namespace NUMINAMATH_CALUDE_distance_spain_other_proof_l1064_106411

/-- The distance between Spain and the other country -/
def distance_spain_other : ℕ := 5404

/-- The total distance between two countries -/
def total_distance : ℕ := 7019

/-- The distance between Spain and Germany -/
def distance_spain_germany : ℕ := 1615

/-- Theorem stating that the distance between Spain and the other country
    is equal to the total distance minus the distance between Spain and Germany -/
theorem distance_spain_other_proof :
  distance_spain_other = total_distance - distance_spain_germany :=
by sorry

end NUMINAMATH_CALUDE_distance_spain_other_proof_l1064_106411


namespace NUMINAMATH_CALUDE_first_candle_triple_second_at_correct_time_l1064_106458

/-- The time (in hours) when the first candle is three times the height of the second candle -/
def time_when_first_is_triple_second : ℚ := 40 / 11

/-- The initial height of both candles -/
def initial_height : ℚ := 1

/-- The time (in hours) it takes for the first candle to burn out completely -/
def first_candle_burnout_time : ℚ := 5

/-- The time (in hours) it takes for the second candle to burn out completely -/
def second_candle_burnout_time : ℚ := 4

/-- The height of the first candle at time t -/
def first_candle_height (t : ℚ) : ℚ := initial_height - (t / first_candle_burnout_time)

/-- The height of the second candle at time t -/
def second_candle_height (t : ℚ) : ℚ := initial_height - (t / second_candle_burnout_time)

theorem first_candle_triple_second_at_correct_time :
  first_candle_height time_when_first_is_triple_second = 
  3 * second_candle_height time_when_first_is_triple_second :=
sorry

end NUMINAMATH_CALUDE_first_candle_triple_second_at_correct_time_l1064_106458


namespace NUMINAMATH_CALUDE_parabola_focus_l1064_106453

/-- For a parabola y = ax^2 with focus at (0, 1), a = 1/4 -/
theorem parabola_focus (a : ℝ) : 
  (∀ x y : ℝ, y = a * x^2) →  -- Parabola equation
  (0, 1) = (0, 1 / (4 * a)) →  -- Focus at (0, 1)
  a = 1/4 := by
  sorry

end NUMINAMATH_CALUDE_parabola_focus_l1064_106453


namespace NUMINAMATH_CALUDE_arithmetic_square_root_of_four_l1064_106444

theorem arithmetic_square_root_of_four : ∃ x : ℝ, x > 0 ∧ x^2 = 4 ∧ ∀ y : ℝ, y > 0 ∧ y^2 = 4 → y = x :=
sorry

end NUMINAMATH_CALUDE_arithmetic_square_root_of_four_l1064_106444


namespace NUMINAMATH_CALUDE_max_value_of_f_l1064_106445

/-- The function we're analyzing -/
def f (x : ℝ) : ℝ := -4 * x^2 + 8 * x + 3

/-- The domain of x -/
def X : Set ℝ := Set.Ioo 0 3

theorem max_value_of_f :
  ∃ (M : ℝ), M = 7 ∧ ∀ x ∈ X, f x ≤ M ∧ ∃ x₀ ∈ X, f x₀ = M :=
by sorry

end NUMINAMATH_CALUDE_max_value_of_f_l1064_106445


namespace NUMINAMATH_CALUDE_math_books_count_l1064_106473

theorem math_books_count (total_books : ℕ) (math_cost history_cost total_price : ℕ) :
  total_books = 80 ∧ 
  math_cost = 4 ∧ 
  history_cost = 5 ∧ 
  total_price = 368 →
  ∃ (math_books : ℕ), 
    math_books * math_cost + (total_books - math_books) * history_cost = total_price ∧ 
    math_books = 32 :=
by sorry

end NUMINAMATH_CALUDE_math_books_count_l1064_106473


namespace NUMINAMATH_CALUDE_random_walk_exits_lawn_l1064_106429

/-- A random walk on a 2D plane -/
def RandomWalk2D := ℕ → ℝ × ℝ

/-- The origin (starting point) of the random walk -/
def origin : ℝ × ℝ := (0, 0)

/-- The radius of the circular lawn -/
def lawn_radius : ℝ := 100

/-- The length of each step in the random walk -/
def step_length : ℝ := 1

/-- The expected distance from the origin after n steps in a 2D random walk -/
noncomputable def expected_distance (n : ℕ) : ℝ := Real.sqrt (n : ℝ)

/-- Theorem: For a sufficiently large number of steps, the expected distance 
    from the origin in a 2D random walk exceeds the lawn radius -/
theorem random_walk_exits_lawn :
  ∃ N : ℕ, ∀ n : ℕ, n ≥ N → expected_distance n > lawn_radius :=
sorry

end NUMINAMATH_CALUDE_random_walk_exits_lawn_l1064_106429


namespace NUMINAMATH_CALUDE_apple_purchase_problem_l1064_106463

theorem apple_purchase_problem (x : ℕ) : 
  (12 : ℚ) / x - (12 : ℚ) / (x + 2) = 1 / 12 → x + 2 = 18 := by
  sorry

end NUMINAMATH_CALUDE_apple_purchase_problem_l1064_106463


namespace NUMINAMATH_CALUDE_canoe_kayak_revenue_l1064_106409

/-- Represents the revenue calculation for a canoe and kayak rental business --/
theorem canoe_kayak_revenue
  (canoe_cost : ℕ)
  (kayak_cost : ℕ)
  (canoe_kayak_ratio : ℚ)
  (canoe_kayak_difference : ℕ)
  (h1 : canoe_cost = 12)
  (h2 : kayak_cost = 18)
  (h3 : canoe_kayak_ratio = 3 / 2)
  (h4 : canoe_kayak_difference = 7) :
  ∃ (num_canoes num_kayaks : ℕ),
    num_canoes = num_kayaks + canoe_kayak_difference ∧
    (num_canoes : ℚ) / num_kayaks = canoe_kayak_ratio ∧
    num_canoes * canoe_cost + num_kayaks * kayak_cost = 504 :=
by sorry

end NUMINAMATH_CALUDE_canoe_kayak_revenue_l1064_106409


namespace NUMINAMATH_CALUDE_percentage_calculation_l1064_106476

theorem percentage_calculation (x : ℝ) (h : 0.30 * 0.40 * x = 24) : 0.20 * 0.60 * x = 24 := by
  sorry

end NUMINAMATH_CALUDE_percentage_calculation_l1064_106476


namespace NUMINAMATH_CALUDE_k_value_proof_l1064_106401

theorem k_value_proof (k : ℝ) (h1 : k ≠ 0) :
  (∀ x : ℝ, (x^2 - k) * (x + k) = x^3 + k * (x^2 - x - 5)) → k = 5 := by
  sorry

end NUMINAMATH_CALUDE_k_value_proof_l1064_106401


namespace NUMINAMATH_CALUDE_expression_equality_l1064_106459

theorem expression_equality : 
  (2025^3 - 3 * 2025^2 * 2026 + 5 * 2025 * 2026^2 - 2026^3 + 4) / (2025 * 2026) = 
  4052 + 3 / 2025 := by
sorry

end NUMINAMATH_CALUDE_expression_equality_l1064_106459


namespace NUMINAMATH_CALUDE_rational_function_property_l1064_106461

/-- Represents a rational function -/
structure RationalFunction where
  numerator : Polynomial ℝ
  denominator : Polynomial ℝ

/-- Counts the number of holes in the graph of a rational function -/
def count_holes (f : RationalFunction) : ℕ := sorry

/-- Counts the number of vertical asymptotes in the graph of a rational function -/
def count_vertical_asymptotes (f : RationalFunction) : ℕ := sorry

/-- Counts the number of horizontal asymptotes in the graph of a rational function -/
def count_horizontal_asymptotes (f : RationalFunction) : ℕ := sorry

/-- Counts the number of oblique asymptotes in the graph of a rational function -/
def count_oblique_asymptotes (f : RationalFunction) : ℕ := sorry

/-- The main theorem about the specific rational function -/
theorem rational_function_property : 
  let f : RationalFunction := {
    numerator := Polynomial.monomial 2 1 - Polynomial.monomial 1 5 + Polynomial.monomial 0 6,
    denominator := Polynomial.monomial 3 1 - Polynomial.monomial 2 3 + Polynomial.monomial 1 2
  }
  let p := count_holes f
  let q := count_vertical_asymptotes f
  let r := count_horizontal_asymptotes f
  let s := count_oblique_asymptotes f
  p + 2*q + 3*r + 4*s = 8 := by sorry

end NUMINAMATH_CALUDE_rational_function_property_l1064_106461


namespace NUMINAMATH_CALUDE_expected_value_ten_sided_die_l1064_106491

/-- A fair 10-sided die with faces numbered from 1 to 10 -/
def TenSidedDie : Finset ℕ := Finset.range 10

/-- The expected value of rolling the die -/
def ExpectedValue : ℚ := (Finset.sum TenSidedDie (λ i => i + 1)) / 10

/-- Theorem: The expected value of rolling a fair 10-sided die with faces numbered from 1 to 10 is 5.5 -/
theorem expected_value_ten_sided_die : ExpectedValue = 11/2 := by
  sorry

end NUMINAMATH_CALUDE_expected_value_ten_sided_die_l1064_106491


namespace NUMINAMATH_CALUDE_select_four_from_seven_l1064_106418

theorem select_four_from_seven (n : ℕ) (k : ℕ) : n = 7 ∧ k = 4 → Nat.choose n k = 35 := by
  sorry

end NUMINAMATH_CALUDE_select_four_from_seven_l1064_106418


namespace NUMINAMATH_CALUDE_triangle_area_formula_l1064_106462

variable (m₁ m₂ m₃ : ℝ)
variable (u u₁ u₂ u₃ t : ℝ)

def is_altitude (m : ℝ) : Prop := m > 0

theorem triangle_area_formula 
  (h₁ : is_altitude m₁)
  (h₂ : is_altitude m₂)
  (h₃ : is_altitude m₃)
  (hu : u = 1/2 * (1/m₁ + 1/m₂ + 1/m₃))
  (hu₁ : u₁ = u - 1/m₁)
  (hu₂ : u₂ = u - 1/m₂)
  (hu₃ : u₃ = u - 1/m₃)
  : t = 4 * Real.sqrt (u * u₁ * u₂ * u₃) :=
sorry

end NUMINAMATH_CALUDE_triangle_area_formula_l1064_106462


namespace NUMINAMATH_CALUDE_mapping_count_l1064_106447

-- Define the sets P and Q
variable (P Q : Type)

-- Define the conditions
variable (h1 : Fintype Q)
variable (h2 : Fintype.card Q = 3)
variable (h3 : Fintype P)
variable (h4 : (Fintype.card P) ^ (Fintype.card Q) = 81)

-- The theorem to prove
theorem mapping_count : (Fintype.card Q) ^ (Fintype.card P) = 64 := by
  sorry

end NUMINAMATH_CALUDE_mapping_count_l1064_106447


namespace NUMINAMATH_CALUDE_validBinaryStrings_10_l1064_106425

/-- A function that returns the number of binary strings of length n 
    that do not contain the substrings 101 or 010 -/
def validBinaryStrings (n : ℕ) : ℕ :=
  match n with
  | 0 => 1
  | 1 => 2
  | 2 => 4
  | 3 => 6
  | m + 3 => validBinaryStrings (m + 2) + validBinaryStrings (m + 1)

/-- Theorem stating that the number of binary strings of length 10 
    that do not contain the substrings 101 or 010 is 178 -/
theorem validBinaryStrings_10 : validBinaryStrings 10 = 178 := by
  sorry

end NUMINAMATH_CALUDE_validBinaryStrings_10_l1064_106425


namespace NUMINAMATH_CALUDE_triangle_angle_theorem_l1064_106438

/-- A triple of integers representing the angles of a triangle in degrees. -/
structure TriangleAngles where
  a : ℕ
  b : ℕ
  c : ℕ
  sum_eq_180 : a + b + c = 180
  all_positive : 0 < a ∧ 0 < b ∧ 0 < c
  all_acute : a < 90 ∧ b < 90 ∧ c < 90

/-- The set of valid angle combinations for the triangle. -/
def validCombinations : Set TriangleAngles := {
  ⟨42, 72, 66, by norm_num, by norm_num, by norm_num⟩,
  ⟨49, 54, 77, by norm_num, by norm_num, by norm_num⟩,
  ⟨56, 36, 88, by norm_num, by norm_num, by norm_num⟩,
  ⟨84, 63, 33, by norm_num, by norm_num, by norm_num⟩
}

/-- Theorem stating that the only valid angle combinations for the triangle
    are those in the validCombinations set. -/
theorem triangle_angle_theorem :
  ∀ t : TriangleAngles,
    (∃ k : ℕ, t.a = 7 * k) ∧
    (∃ l : ℕ, t.b = 9 * l) ∧
    (∃ m : ℕ, t.c = 11 * m) →
    t ∈ validCombinations := by
  sorry

end NUMINAMATH_CALUDE_triangle_angle_theorem_l1064_106438


namespace NUMINAMATH_CALUDE_sin_difference_equals_four_l1064_106443

theorem sin_difference_equals_four : 
  1 / Real.sin (10 * π / 180) - Real.sqrt 3 / Real.sin (80 * π / 180) = 4 := by
  sorry

end NUMINAMATH_CALUDE_sin_difference_equals_four_l1064_106443


namespace NUMINAMATH_CALUDE_integral_cube_root_x_squared_plus_sqrt_x_l1064_106481

theorem integral_cube_root_x_squared_plus_sqrt_x (x : ℝ) :
  (deriv (fun x => (3/5) * x * (x^2)^(1/3) + (2/3) * x * x^(1/2))) x = x^(2/3) + x^(1/2) :=
by sorry

end NUMINAMATH_CALUDE_integral_cube_root_x_squared_plus_sqrt_x_l1064_106481


namespace NUMINAMATH_CALUDE_thirteen_thousand_one_hundred_twenty_one_obtainable_twelve_thousand_one_hundred_thirty_one_not_obtainable_l1064_106465

/-- The set of numbers that can be written on the blackboard -/
inductive BoardNumber : ℕ → Prop where
  | one : BoardNumber 1
  | two : BoardNumber 2
  | add (m n : ℕ) : BoardNumber m → BoardNumber n → BoardNumber (m + n + m * n)

/-- A number is obtainable if it's in the set of BoardNumbers -/
def Obtainable (n : ℕ) : Prop := BoardNumber n

theorem thirteen_thousand_one_hundred_twenty_one_obtainable :
  Obtainable 13121 :=
sorry

theorem twelve_thousand_one_hundred_thirty_one_not_obtainable :
  ¬ Obtainable 12131 :=
sorry

end NUMINAMATH_CALUDE_thirteen_thousand_one_hundred_twenty_one_obtainable_twelve_thousand_one_hundred_thirty_one_not_obtainable_l1064_106465


namespace NUMINAMATH_CALUDE_simplify_polynomial_l1064_106494

theorem simplify_polynomial (r : ℝ) :
  (2 * r^3 + 5 * r^2 + 6 * r - 4) - (r^3 + 9 * r^2 + 4 * r - 7) =
  r^3 - 4 * r^2 + 2 * r + 3 := by
  sorry

end NUMINAMATH_CALUDE_simplify_polynomial_l1064_106494


namespace NUMINAMATH_CALUDE_max_surface_area_of_stacked_solids_l1064_106400

/-- Represents the dimensions of a rectangular solid -/
structure Dimensions where
  length : ℝ
  width : ℝ
  height : ℝ

/-- Calculates the surface area of a rectangular solid -/
def surface_area (d : Dimensions) : ℝ :=
  2 * (d.length * d.width + d.length * d.height + d.width * d.height)

/-- Calculates the surface area of two stacked rectangular solids -/
def stacked_surface_area (d : Dimensions) (overlap_dim1 overlap_dim2 : ℝ) : ℝ :=
  2 * (overlap_dim1 * overlap_dim2) + 
  2 * (d.length * d.width + d.length * d.height + d.width * d.height)

/-- The dimensions of the given rectangular solids -/
def solid_dimensions : Dimensions :=
  { length := 5, width := 4, height := 3 }

theorem max_surface_area_of_stacked_solids :
  let d := solid_dimensions
  let sa1 := stacked_surface_area d d.length d.width
  let sa2 := stacked_surface_area d d.length d.height
  let sa3 := stacked_surface_area d d.width d.height
  max sa1 (max sa2 sa3) = 164 := by sorry

end NUMINAMATH_CALUDE_max_surface_area_of_stacked_solids_l1064_106400


namespace NUMINAMATH_CALUDE_quadratic_root_transformation_l1064_106460

theorem quadratic_root_transformation (p q r : ℝ) (u v : ℝ) : 
  (p * u^2 + q * u + r = 0) ∧ (p * v^2 + q * v + r = 0) →
  ((2*p*u + q)^2 - (q/(4*p)) * (2*p*u + q) + r = 0) ∧
  ((2*p*v + q)^2 - (q/(4*p)) * (2*p*v + q) + r = 0) := by
sorry

end NUMINAMATH_CALUDE_quadratic_root_transformation_l1064_106460


namespace NUMINAMATH_CALUDE_angle_sum_is_pi_over_two_l1064_106490

theorem angle_sum_is_pi_over_two (a b : ℝ) 
  (h_acute_a : 0 < a ∧ a < π / 2) 
  (h_acute_b : 0 < b ∧ b < π / 2)
  (h1 : 4 * Real.sin a ^ 2 + 3 * Real.sin b ^ 2 = 1)
  (h2 : 4 * Real.sin (2 * a) - 3 * Real.sin (2 * b) = 0) : 
  2 * a + b = π / 2 := by
sorry

end NUMINAMATH_CALUDE_angle_sum_is_pi_over_two_l1064_106490


namespace NUMINAMATH_CALUDE_first_place_percentage_l1064_106430

/-- 
Given a pot of money where:
- 8 people each contribute $5
- Third place gets $4
- Second and third place split the remaining money after first place
Prove that first place gets 80% of the total money
-/
theorem first_place_percentage (total_people : Nat) (contribution : ℕ) (third_place_prize : ℕ) :
  total_people = 8 →
  contribution = 5 →
  third_place_prize = 4 →
  (((total_people * contribution - 2 * third_place_prize) : ℚ) / (total_people * contribution)) = 4/5 := by
  sorry

#check first_place_percentage

end NUMINAMATH_CALUDE_first_place_percentage_l1064_106430


namespace NUMINAMATH_CALUDE_felix_lifting_capacity_l1064_106426

/-- Felix's lifting capacity problem -/
theorem felix_lifting_capacity 
  (felix_lift_ratio : ℝ) 
  (brother_weight_ratio : ℝ) 
  (brother_lift_ratio : ℝ) 
  (brother_lift_weight : ℝ) 
  (h1 : felix_lift_ratio = 1.5)
  (h2 : brother_weight_ratio = 2)
  (h3 : brother_lift_ratio = 3)
  (h4 : brother_lift_weight = 600) :
  felix_lift_ratio * (brother_lift_weight / (brother_lift_ratio * brother_weight_ratio)) = 150 := by
  sorry


end NUMINAMATH_CALUDE_felix_lifting_capacity_l1064_106426


namespace NUMINAMATH_CALUDE_massager_usage_time_l1064_106410

/-- The number of vibrations per second at the lowest setting -/
def lowest_vibrations_per_second : ℕ := 1600

/-- The percentage increase in vibrations at the highest setting -/
def highest_setting_increase : ℚ := 60 / 100

/-- The total number of vibrations experienced -/
def total_vibrations : ℕ := 768000

/-- The number of seconds in a minute -/
def seconds_per_minute : ℕ := 60

/-- Calculates the number of minutes Matt uses the massager at the highest setting -/
def usage_time_minutes : ℚ :=
  let highest_vibrations_per_second : ℚ := lowest_vibrations_per_second * (1 + highest_setting_increase)
  let usage_time_seconds : ℚ := total_vibrations / highest_vibrations_per_second
  usage_time_seconds / seconds_per_minute

theorem massager_usage_time :
  usage_time_minutes = 5 := by sorry

end NUMINAMATH_CALUDE_massager_usage_time_l1064_106410


namespace NUMINAMATH_CALUDE_complement_of_M_in_U_l1064_106448

def U : Set Nat := {1, 2, 3, 4, 5, 6}
def M : Set Nat := {1, 2, 4}

theorem complement_of_M_in_U :
  (U \ M) = {3, 5, 6} := by sorry

end NUMINAMATH_CALUDE_complement_of_M_in_U_l1064_106448


namespace NUMINAMATH_CALUDE_tan_sum_pi_eighths_l1064_106413

theorem tan_sum_pi_eighths : Real.tan (π / 8) + Real.tan (3 * π / 8) = 2 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_tan_sum_pi_eighths_l1064_106413


namespace NUMINAMATH_CALUDE_smallest_y_theorem_l1064_106440

def x : ℕ := 6 * 18 * 42

def is_perfect_cube (n : ℕ) : Prop := ∃ m : ℕ, n = m^3

def smallest_y_for_perfect_cube : ℕ := 441

theorem smallest_y_theorem :
  (∀ y : ℕ, y < smallest_y_for_perfect_cube → ¬(is_perfect_cube (x * y))) ∧
  (is_perfect_cube (x * smallest_y_for_perfect_cube)) := by sorry

end NUMINAMATH_CALUDE_smallest_y_theorem_l1064_106440


namespace NUMINAMATH_CALUDE_book_arrangement_l1064_106420

theorem book_arrangement (n m : ℕ) (h : n + m = 8) :
  Nat.choose 8 n = Nat.choose 8 m :=
sorry

end NUMINAMATH_CALUDE_book_arrangement_l1064_106420


namespace NUMINAMATH_CALUDE_shared_foci_implies_m_equals_one_l1064_106412

/-- Given an ellipse and a hyperbola that share the same foci, prove that m = 1 -/
theorem shared_foci_implies_m_equals_one (m : ℝ) :
  (∀ x y : ℝ, x^2/4 + y^2/m^2 = 1 ↔ x^2/m - y^2/2 = 1) →
  (∃ c : ℝ, c^2 = 4 - m^2 ∧ c^2 = m + 2) →
  m = 1 := by
  sorry

end NUMINAMATH_CALUDE_shared_foci_implies_m_equals_one_l1064_106412


namespace NUMINAMATH_CALUDE_sector_area_theorem_l1064_106439

/-- A sector is a portion of a circle enclosed by two radii and an arc. -/
structure Sector where
  centralAngle : ℝ
  perimeter : ℝ

/-- The area of a sector. -/
def sectorArea (s : Sector) : ℝ := sorry

theorem sector_area_theorem (s : Sector) :
  s.centralAngle = 2 ∧ s.perimeter = 8 → sectorArea s = 4 := by
  sorry

end NUMINAMATH_CALUDE_sector_area_theorem_l1064_106439


namespace NUMINAMATH_CALUDE_expensive_module_cost_l1064_106422

def total_modules : ℕ := 11
def cheap_modules : ℕ := 10
def cheap_module_cost : ℚ := 3.5
def total_stock_value : ℚ := 45

theorem expensive_module_cost :
  ∃ (expensive_cost : ℚ),
    expensive_cost = total_stock_value - (cheap_modules : ℚ) * cheap_module_cost ∧
    expensive_cost = 10 := by
  sorry

end NUMINAMATH_CALUDE_expensive_module_cost_l1064_106422


namespace NUMINAMATH_CALUDE_transform_standard_deviation_l1064_106477

def standardDeviation (sample : Fin 10 → ℝ) : ℝ := sorry

theorem transform_standard_deviation 
  (x : Fin 10 → ℝ) 
  (h : standardDeviation x = 8) : 
  standardDeviation (fun i => 2 * x i - 1) = 16 := by sorry

end NUMINAMATH_CALUDE_transform_standard_deviation_l1064_106477


namespace NUMINAMATH_CALUDE_greater_number_on_cards_l1064_106427

theorem greater_number_on_cards (x y : ℤ) 
  (sum_eq : x + y = 1443) 
  (diff_eq : x - y = 141) : 
  x = 792 ∧ x > y :=
by sorry

end NUMINAMATH_CALUDE_greater_number_on_cards_l1064_106427


namespace NUMINAMATH_CALUDE_abs_inequality_equivalence_l1064_106433

theorem abs_inequality_equivalence :
  ∀ x : ℝ, |5 - 2*x| < 3 ↔ 1 < x ∧ x < 4 := by sorry

end NUMINAMATH_CALUDE_abs_inequality_equivalence_l1064_106433


namespace NUMINAMATH_CALUDE_lisa_marbles_l1064_106475

/-- The number of marbles each person has -/
structure Marbles where
  connie : ℕ
  juan : ℕ
  mark : ℕ
  lisa : ℕ

/-- The conditions of the marble problem -/
def marble_problem (m : Marbles) : Prop :=
  m.connie = 323 ∧
  m.juan = m.connie + 175 ∧
  m.mark = 3 * m.juan ∧
  m.lisa = m.mark / 2 - 200

/-- The theorem stating that Lisa has 547 marbles -/
theorem lisa_marbles (m : Marbles) (h : marble_problem m) : m.lisa = 547 := by
  sorry

end NUMINAMATH_CALUDE_lisa_marbles_l1064_106475


namespace NUMINAMATH_CALUDE_max_distance_from_origin_l1064_106449

/-- The post position -/
def post : ℝ × ℝ := (2, 5)

/-- The rope length -/
def rope_length : ℝ := 8

/-- The rectangle's vertices -/
def rectangle_vertices : List (ℝ × ℝ) := [(0, 0), (0, 10), (10, 0), (10, 10)]

/-- Check if a point is within the rectangle -/
def in_rectangle (p : ℝ × ℝ) : Prop :=
  0 ≤ p.1 ∧ p.1 ≤ 10 ∧ 0 ≤ p.2 ∧ p.2 ≤ 10

/-- Check if a point is within the rope's reach -/
def in_rope_reach (p : ℝ × ℝ) : Prop :=
  (p.1 - post.1)^2 + (p.2 - post.2)^2 ≤ rope_length^2

/-- The maximum distance from origin theorem -/
theorem max_distance_from_origin :
  ∃ (p : ℝ × ℝ), in_rectangle p ∧ in_rope_reach p ∧
  ∀ (q : ℝ × ℝ), in_rectangle q → in_rope_reach q →
  p.1^2 + p.2^2 ≥ q.1^2 + q.2^2 ∧
  p.1^2 + p.2^2 = 125 :=
sorry

end NUMINAMATH_CALUDE_max_distance_from_origin_l1064_106449


namespace NUMINAMATH_CALUDE_boat_speed_in_still_water_l1064_106446

/-- The speed of a boat in still water, given downstream travel information and current speed. -/
theorem boat_speed_in_still_water (current_speed : ℝ) (downstream_distance : ℝ) (downstream_time : ℝ) :
  current_speed = 3 →
  downstream_distance = 3.6 →
  downstream_time = 1/5 →
  ∃ (boat_speed : ℝ), boat_speed = 15 ∧ downstream_distance = (boat_speed + current_speed) * downstream_time :=
by
  sorry


end NUMINAMATH_CALUDE_boat_speed_in_still_water_l1064_106446


namespace NUMINAMATH_CALUDE_monotonicity_a_eq_zero_monotonicity_a_pos_monotonicity_a_neg_l1064_106402

noncomputable section

-- Define the function f(x) = x^2 * e^(ax)
def f (a : ℝ) (x : ℝ) : ℝ := x^2 * Real.exp (a * x)

-- Define the derivative of f(x)
def f_deriv (a : ℝ) (x : ℝ) : ℝ := (2 * x + a * x^2) * Real.exp (a * x)

-- Theorem for monotonicity when a = 0
theorem monotonicity_a_eq_zero :
  ∀ x : ℝ, x < 0 → (∀ y : ℝ, y < x → f 0 y > f 0 x) ∧
            x > 0 → (∀ y : ℝ, y > x → f 0 y > f 0 x) :=
sorry

-- Theorem for monotonicity when a > 0
theorem monotonicity_a_pos :
  ∀ a : ℝ, a > 0 → 
  ∀ x : ℝ, (x < -2/a → (∀ y : ℝ, y < x → f a y < f a x)) ∧
           (x > 0 → (∀ y : ℝ, y > x → f a y > f a x)) ∧
           (-2/a < x ∧ x < 0 → (∀ y : ℝ, -2/a < y ∧ y < x → f a y > f a x)) :=
sorry

-- Theorem for monotonicity when a < 0
theorem monotonicity_a_neg :
  ∀ a : ℝ, a < 0 → 
  ∀ x : ℝ, (x < 0 → (∀ y : ℝ, y < x → f a y > f a x)) ∧
           (x > -2/a → (∀ y : ℝ, y > x → f a y < f a x)) ∧
           (0 < x ∧ x < -2/a → (∀ y : ℝ, x < y ∧ y < -2/a → f a y > f a x)) :=
sorry

end

end NUMINAMATH_CALUDE_monotonicity_a_eq_zero_monotonicity_a_pos_monotonicity_a_neg_l1064_106402


namespace NUMINAMATH_CALUDE_cone_lateral_surface_area_l1064_106436

/-- Represents a cone with given slant height and lateral surface property -/
structure Cone where
  slant_height : ℝ
  lateral_surface_is_semicircle : Prop

/-- Calculates the lateral surface area of a cone -/
def lateral_surface_area (c : Cone) : ℝ := sorry

theorem cone_lateral_surface_area 
  (c : Cone) 
  (h1 : c.slant_height = 10) 
  (h2 : c.lateral_surface_is_semicircle) : 
  lateral_surface_area c = 50 * Real.pi := by sorry

end NUMINAMATH_CALUDE_cone_lateral_surface_area_l1064_106436


namespace NUMINAMATH_CALUDE_exponent_calculation_l1064_106414

theorem exponent_calculation : 3 * 3^4 - 27^63 / 27^61 = -486 := by
  sorry

end NUMINAMATH_CALUDE_exponent_calculation_l1064_106414


namespace NUMINAMATH_CALUDE_cs_physics_overlap_l1064_106493

/-- Represents the fraction of students in one club who also attend another club -/
def club_overlap (club1 club2 : Type) : ℚ := sorry

theorem cs_physics_overlap :
  let m := club_overlap Mathematics Physics
  let c := club_overlap Mathematics ComputerScience
  let p := club_overlap Physics Mathematics
  let q := club_overlap Physics ComputerScience
  let r := club_overlap ComputerScience Mathematics
  m = 1/6 ∧ c = 1/8 ∧ p = 1/3 ∧ q = 1/5 ∧ r = 1/7 →
  club_overlap ComputerScience Physics = 4/35 :=
sorry

end NUMINAMATH_CALUDE_cs_physics_overlap_l1064_106493


namespace NUMINAMATH_CALUDE_ginger_water_usage_l1064_106416

def water_usage (hours_worked : ℕ) (cups_per_bottle : ℕ) (bottles_for_plants : ℕ) : ℕ :=
  (hours_worked * cups_per_bottle) + (bottles_for_plants * cups_per_bottle)

theorem ginger_water_usage :
  water_usage 8 2 5 = 26 := by
  sorry

end NUMINAMATH_CALUDE_ginger_water_usage_l1064_106416


namespace NUMINAMATH_CALUDE_factor_tree_problem_l1064_106415

theorem factor_tree_problem (H I F G X : ℕ) : 
  H = 7 * 2 →
  I = 11 * 2 →
  F = 7 * H →
  G = 11 * I →
  X = F * G →
  X = 23716 :=
by sorry

end NUMINAMATH_CALUDE_factor_tree_problem_l1064_106415


namespace NUMINAMATH_CALUDE_number_wall_solution_l1064_106454

/-- Represents a block in the Number Wall --/
structure Block where
  value : ℕ

/-- Represents the Number Wall --/
structure NumberWall where
  n : Block
  block1 : Block
  block2 : Block
  block3 : Block
  block4 : Block
  top : Block

/-- The sum of two adjacent blocks equals the block above them --/
def sum_rule (b1 b2 b_above : Block) : Prop :=
  b1.value + b2.value = b_above.value

/-- The Number Wall satisfies all given conditions --/
def valid_wall (w : NumberWall) : Prop :=
  w.block1.value = 4 ∧
  w.block2.value = 8 ∧
  w.block3.value = 7 ∧
  w.block4.value = 15 ∧
  w.top.value = 46 ∧
  sum_rule w.n w.block1 { value := w.n.value + 4 } ∧
  sum_rule { value := w.n.value + 4 } w.block2 w.block4 ∧
  sum_rule w.block4 w.block3 { value := 27 } ∧
  sum_rule { value := w.n.value + 16 } { value := 27 } w.top

theorem number_wall_solution (w : NumberWall) (h : valid_wall w) : w.n.value = 3 := by
  sorry


end NUMINAMATH_CALUDE_number_wall_solution_l1064_106454


namespace NUMINAMATH_CALUDE_product_xyz_equals_25_l1064_106437

/-- Given complex numbers x, y, and z satisfying specific equations, prove that their product is 25. -/
theorem product_xyz_equals_25 
  (x y z : ℂ) 
  (eq1 : 2 * x * y + 5 * y = -20)
  (eq2 : 2 * y * z + 5 * z = -20)
  (eq3 : 2 * z * x + 5 * x = -20) :
  x * y * z = 25 := by
  sorry

end NUMINAMATH_CALUDE_product_xyz_equals_25_l1064_106437


namespace NUMINAMATH_CALUDE_min_value_of_expression_l1064_106470

theorem min_value_of_expression (a b : ℝ) : 
  a > 0 → b > 0 → a + b = 1 → 
  (∀ x y : ℝ, a * x + b * y = 1 → x^2 + y^2 - 2*x - 2*y = 0 → x = 1 ∧ y = 1) →
  (∀ a' b' : ℝ, a' > 0 → b' > 0 → a' + b' = 1 → 1/a' + 2/b' ≥ 3 + 2 * Real.sqrt 2) ∧
  (1/a + 2/b = 3 + 2 * Real.sqrt 2) :=
by sorry

end NUMINAMATH_CALUDE_min_value_of_expression_l1064_106470


namespace NUMINAMATH_CALUDE_rectangular_field_length_l1064_106489

theorem rectangular_field_length (width : ℝ) (length : ℝ) : 
  width = 13.5 → length = 2 * width - 3 → length = 24 := by
  sorry

end NUMINAMATH_CALUDE_rectangular_field_length_l1064_106489


namespace NUMINAMATH_CALUDE_quadratic_factorization_l1064_106406

theorem quadratic_factorization (c d : ℕ) (hc : c > d) 
  (h1 : c + d = 14) (h2 : c * d = 40) : 4 * d - c = 6 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_factorization_l1064_106406


namespace NUMINAMATH_CALUDE_quadrilateral_inequality_quadrilateral_inequality_equality_condition_l1064_106484

/-- Theorem: Quadrilateral Inequality
For any quadrilateral with sides a₁, a₂, a₃, a₄ and semi-perimeter s,
the sum of reciprocals of (aᵢ + s) is less than or equal to 2/9 times
the sum of reciprocals of square roots of (s-aᵢ)(s-aⱼ) for all pairs i,j. -/
theorem quadrilateral_inequality (a₁ a₂ a₃ a₄ s : ℝ) 
  (h₁ : a₁ > 0) (h₂ : a₂ > 0) (h₃ : a₃ > 0) (h₄ : a₄ > 0) (h_s : s > 0)
  (h_perimeter : a₁ + a₂ + a₃ + a₄ = 2 * s) : 
  (1 / (a₁ + s) + 1 / (a₂ + s) + 1 / (a₃ + s) + 1 / (a₄ + s)) ≤ 
  (2 / 9) * (1 / Real.sqrt ((s - a₁) * (s - a₂)) + 
             1 / Real.sqrt ((s - a₁) * (s - a₃)) + 
             1 / Real.sqrt ((s - a₁) * (s - a₄)) + 
             1 / Real.sqrt ((s - a₂) * (s - a₃)) + 
             1 / Real.sqrt ((s - a₂) * (s - a₄)) + 
             1 / Real.sqrt ((s - a₃) * (s - a₄))) :=
by sorry

/-- Corollary: Equality condition for the quadrilateral inequality -/
theorem quadrilateral_inequality_equality_condition (a₁ a₂ a₃ a₄ s : ℝ) 
  (h₁ : a₁ > 0) (h₂ : a₂ > 0) (h₃ : a₃ > 0) (h₄ : a₄ > 0) (h_s : s > 0)
  (h_perimeter : a₁ + a₂ + a₃ + a₄ = 2 * s) : 
  (1 / (a₁ + s) + 1 / (a₂ + s) + 1 / (a₃ + s) + 1 / (a₄ + s) = 
   (2 / 9) * (1 / Real.sqrt ((s - a₁) * (s - a₂)) + 
              1 / Real.sqrt ((s - a₁) * (s - a₃)) + 
              1 / Real.sqrt ((s - a₁) * (s - a₄)) + 
              1 / Real.sqrt ((s - a₂) * (s - a₃)) + 
              1 / Real.sqrt ((s - a₂) * (s - a₄)) + 
              1 / Real.sqrt ((s - a₃) * (s - a₄)))) ↔ 
  (a₁ = a₂ ∧ a₂ = a₃ ∧ a₃ = a₄) :=
by sorry

end NUMINAMATH_CALUDE_quadrilateral_inequality_quadrilateral_inequality_equality_condition_l1064_106484


namespace NUMINAMATH_CALUDE_cupboard_cost_price_correct_l1064_106466

/-- The cost price of a cupboard satisfying the given conditions -/
def cupboard_cost_price : ℝ :=
  let below_cost_percentage : ℝ := 0.12
  let profit_percentage : ℝ := 0.12
  let additional_amount : ℝ := 1650
  
  -- Define the selling price as a function of the cost price
  let selling_price (cost : ℝ) : ℝ := cost * (1 - below_cost_percentage)
  
  -- Define the new selling price (with profit) as a function of the cost price
  let new_selling_price (cost : ℝ) : ℝ := cost * (1 + profit_percentage)
  
  -- The cost price that satisfies the conditions
  6875

/-- Theorem stating that the calculated cost price satisfies the given conditions -/
theorem cupboard_cost_price_correct : 
  let cost := cupboard_cost_price
  let below_cost_percentage : ℝ := 0.12
  let profit_percentage : ℝ := 0.12
  let additional_amount : ℝ := 1650
  let selling_price := cost * (1 - below_cost_percentage)
  let new_selling_price := cost * (1 + profit_percentage)
  (new_selling_price - selling_price = additional_amount) ∧ 
  (cost = 6875) :=
by sorry

#eval cupboard_cost_price

end NUMINAMATH_CALUDE_cupboard_cost_price_correct_l1064_106466


namespace NUMINAMATH_CALUDE_calzone_knead_time_l1064_106408

def calzone_time_problem (total_time onion_time knead_time : ℝ) : Prop :=
  let garlic_pepper_time := onion_time / 4
  let rest_time := 2 * knead_time
  let assemble_time := (knead_time + rest_time) / 10
  total_time = onion_time + garlic_pepper_time + knead_time + rest_time + assemble_time

theorem calzone_knead_time :
  ∃ (knead_time : ℝ), 
    calzone_time_problem 124 20 knead_time ∧ 
    knead_time = 30 := by
  sorry

end NUMINAMATH_CALUDE_calzone_knead_time_l1064_106408


namespace NUMINAMATH_CALUDE_angle_inequality_l1064_106417

theorem angle_inequality (x y z : Real) 
  (h1 : 0 < x ∧ x < π/2)
  (h2 : 0 < y ∧ y < π/2)
  (h3 : 0 < z ∧ z < π/2)
  (h4 : (Real.sin x + Real.cos x) * (Real.sin y + 2 * Real.cos y) * (Real.sin z + 3 * Real.cos z) = 10) :
  x = π/4 ∧ x > y ∧ y > z := by sorry

end NUMINAMATH_CALUDE_angle_inequality_l1064_106417


namespace NUMINAMATH_CALUDE_city_transport_capacity_l1064_106492

/-- Represents the capacity of different public transport vehicles in a small city -/
structure CityTransport where
  train_capacity : ℕ
  bus_capacity : ℕ
  tram_capacity : ℕ

/-- Calculates the total capacity of two buses and a tram given the conditions -/
def total_capacity (ct : CityTransport) : ℕ :=
  2 * ct.bus_capacity + ct.tram_capacity

/-- Theorem stating the total capacity of two buses and a tram in the city -/
theorem city_transport_capacity : ∃ (ct : CityTransport),
  ct.train_capacity = 120 ∧
  ct.bus_capacity = ct.train_capacity / 6 ∧
  ct.tram_capacity = (2 * ct.bus_capacity) * 2 / 3 ∧
  total_capacity ct = 67 := by
  sorry


end NUMINAMATH_CALUDE_city_transport_capacity_l1064_106492
