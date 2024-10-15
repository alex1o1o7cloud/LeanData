import Mathlib

namespace NUMINAMATH_CALUDE_blocks_and_colors_l829_82988

theorem blocks_and_colors (total_blocks : ℕ) (blocks_per_color : ℕ) (colors_used : ℕ) : 
  total_blocks = 49 → blocks_per_color = 7 → colors_used = total_blocks / blocks_per_color →
  colors_used = 7 := by
  sorry

end NUMINAMATH_CALUDE_blocks_and_colors_l829_82988


namespace NUMINAMATH_CALUDE_max_value_of_expression_l829_82903

theorem max_value_of_expression (a b : ℕ+) (ha : a < 6) (hb : b < 10) :
  (∀ x y : ℕ+, x < 6 → y < 10 → 2 * x - x * y ≤ 2 * a - a * b) →
  2 * a - a * b = 5 :=
sorry

end NUMINAMATH_CALUDE_max_value_of_expression_l829_82903


namespace NUMINAMATH_CALUDE_stratified_optimal_survey1_simple_random_optimal_survey2_l829_82928

/-- Represents the income level of a family -/
inductive IncomeLevel
  | High
  | Middle
  | Low

/-- Represents a sampling method -/
inductive SamplingMethod
  | SimpleRandom
  | Stratified
  | Systematic

/-- Structure representing the conditions of Survey 1 -/
structure Survey1 where
  highIncomeFamilies : Nat
  middleIncomeFamilies : Nat
  lowIncomeFamilies : Nat
  sampleSize : Nat

/-- Structure representing the conditions of Survey 2 -/
structure Survey2 where
  totalStudents : Nat
  sampleSize : Nat

/-- Function to determine the optimal sampling method for Survey 1 -/
def optimalMethodSurvey1 (s : Survey1) : SamplingMethod := sorry

/-- Function to determine the optimal sampling method for Survey 2 -/
def optimalMethodSurvey2 (s : Survey2) : SamplingMethod := sorry

/-- Theorem stating that stratified sampling is optimal for Survey 1 -/
theorem stratified_optimal_survey1 (s : Survey1) :
  s.highIncomeFamilies = 125 →
  s.middleIncomeFamilies = 200 →
  s.lowIncomeFamilies = 95 →
  s.sampleSize = 100 →
  optimalMethodSurvey1 s = SamplingMethod.Stratified :=
by sorry

/-- Theorem stating that simple random sampling is optimal for Survey 2 -/
theorem simple_random_optimal_survey2 (s : Survey2) :
  s.totalStudents = 5 →
  s.sampleSize = 3 →
  optimalMethodSurvey2 s = SamplingMethod.SimpleRandom :=
by sorry

end NUMINAMATH_CALUDE_stratified_optimal_survey1_simple_random_optimal_survey2_l829_82928


namespace NUMINAMATH_CALUDE_checkerboard_existence_l829_82953

/-- Represents the color of a cell -/
inductive Color
| Black
| White

/-- Represents a 100x100 board -/
def Board := Fin 100 → Fin 100 → Color

/-- Checks if a cell is on the boundary of the board -/
def isBoundary (i j : Fin 100) : Prop :=
  i = 0 ∨ i = 99 ∨ j = 0 ∨ j = 99

/-- Checks if a 2x2 square is monochromatic -/
def isMonochromatic (b : Board) (i j : Fin 100) : Prop :=
  b i j = b (i+1) j ∧ b i j = b i (j+1) ∧ b i j = b (i+1) (j+1)

/-- Checks if a 2x2 square has a checkerboard pattern -/
def isCheckerboard (b : Board) (i j : Fin 100) : Prop :=
  (b i j = b (i+1) (j+1) ∧ b (i+1) j = b i (j+1) ∧ b i j ≠ b (i+1) j)

theorem checkerboard_existence (b : Board) 
  (h1 : ∀ i j, isBoundary i j → b i j = Color.Black)
  (h2 : ∀ i j, ¬isMonochromatic b i j) :
  ∃ i j, isCheckerboard b i j :=
sorry

end NUMINAMATH_CALUDE_checkerboard_existence_l829_82953


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l829_82942

open Set

def A : Set ℝ := {x | -1 < x ∧ x < 2}
def B : Set ℝ := {-2, -1, 0, 1, 2}

theorem intersection_of_A_and_B : A ∩ B = {0, 1} := by
  sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l829_82942


namespace NUMINAMATH_CALUDE_smallest_fraction_greater_than_three_fifths_l829_82998

theorem smallest_fraction_greater_than_three_fifths :
  ∀ a b : ℕ,
    10 ≤ a ∧ a ≤ 99 →
    10 ≤ b ∧ b ≤ 99 →
    (a : ℚ) / b > 3 / 5 →
    (59 : ℚ) / 98 ≤ (a : ℚ) / b :=
by sorry

end NUMINAMATH_CALUDE_smallest_fraction_greater_than_three_fifths_l829_82998


namespace NUMINAMATH_CALUDE_rod_cutting_l829_82932

/-- Given a rod of length 34 meters that can be cut into 40 equal pieces,
    prove that each piece is 0.85 meters long. -/
theorem rod_cutting (rod_length : ℝ) (num_pieces : ℕ) (piece_length : ℝ) 
  (h1 : rod_length = 34)
  (h2 : num_pieces = 40)
  (h3 : piece_length * num_pieces = rod_length) :
  piece_length = 0.85 := by
  sorry

end NUMINAMATH_CALUDE_rod_cutting_l829_82932


namespace NUMINAMATH_CALUDE_sin_20_cos_10_plus_sin_10_sin_70_l829_82919

theorem sin_20_cos_10_plus_sin_10_sin_70 :
  Real.sin (20 * π / 180) * Real.cos (10 * π / 180) +
  Real.sin (10 * π / 180) * Real.sin (70 * π / 180) = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_sin_20_cos_10_plus_sin_10_sin_70_l829_82919


namespace NUMINAMATH_CALUDE_kiera_muffins_count_l829_82915

/-- Represents the number of items in an order -/
structure Order :=
  (muffins : ℕ)
  (fruitCups : ℕ)

/-- Calculates the cost of an order given the prices -/
def orderCost (order : Order) (muffinPrice fruitCupPrice : ℕ) : ℕ :=
  order.muffins * muffinPrice + order.fruitCups * fruitCupPrice

theorem kiera_muffins_count 
  (muffinPrice fruitCupPrice : ℕ)
  (francis : Order)
  (kiera : Order)
  (h1 : muffinPrice = 2)
  (h2 : fruitCupPrice = 3)
  (h3 : francis.muffins = 2)
  (h4 : francis.fruitCups = 2)
  (h5 : kiera.fruitCups = 1)
  (h6 : orderCost francis muffinPrice fruitCupPrice + 
        orderCost kiera muffinPrice fruitCupPrice = 17) :
  kiera.muffins = 2 := by
sorry

end NUMINAMATH_CALUDE_kiera_muffins_count_l829_82915


namespace NUMINAMATH_CALUDE_lcm_of_incremented_numbers_l829_82904

theorem lcm_of_incremented_numbers : Nat.lcm 5 (Nat.lcm 7 (Nat.lcm 13 19)) = 8645 := by
  sorry

end NUMINAMATH_CALUDE_lcm_of_incremented_numbers_l829_82904


namespace NUMINAMATH_CALUDE_quadratic_equation_roots_l829_82986

theorem quadratic_equation_roots (θ : Real) (m : Real) :
  θ ∈ Set.Ioo 0 (2 * Real.pi) →
  (∃ x, 2 * x^2 - (Real.sqrt 3 + 1) * x + m = 0) →
  (∃ x, x = Real.sin θ ∨ x = Real.cos θ) →
  (Real.sin θ / (1 - Real.cos θ) + Real.cos θ / (1 - Real.tan θ) = (3 + 5 * Real.sqrt 3) / 4) ∧
  (m = Real.sqrt 3 / 4) ∧
  ((Real.sin θ = Real.sqrt 3 / 2 ∧ Real.cos θ = 1 / 2 ∧ θ = Real.pi / 3) ∨
   (Real.sin θ = 1 / 2 ∧ Real.cos θ = Real.sqrt 3 / 2 ∧ θ = Real.pi / 6)) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equation_roots_l829_82986


namespace NUMINAMATH_CALUDE_intersection_sum_l829_82914

/-- Given two lines y = 2x + c and y = -x + d intersecting at (4, 12), prove that c + d = 20 -/
theorem intersection_sum (c d : ℝ) : 
  (∀ x y, y = 2*x + c → y = -x + d → (x = 4 ∧ y = 12)) → 
  c + d = 20 := by sorry

end NUMINAMATH_CALUDE_intersection_sum_l829_82914


namespace NUMINAMATH_CALUDE_highlighter_expense_is_30_l829_82925

/-- The amount of money Heaven's brother spent on highlighters -/
def highlighter_expense (total_money : ℕ) (sharpener_price : ℕ) (notebook_price : ℕ) 
  (eraser_price : ℕ) (num_sharpeners : ℕ) (num_notebooks : ℕ) (num_erasers : ℕ) : ℕ :=
  total_money - (sharpener_price * num_sharpeners + notebook_price * num_notebooks + eraser_price * num_erasers)

/-- Theorem stating the amount spent on highlighters -/
theorem highlighter_expense_is_30 : 
  highlighter_expense 100 5 5 4 2 4 10 = 30 := by
  sorry

end NUMINAMATH_CALUDE_highlighter_expense_is_30_l829_82925


namespace NUMINAMATH_CALUDE_age_of_35th_student_l829_82987

/-- The age of the 35th student in a class, given the following conditions:
  - There are 35 students in total
  - The average age of all 35 students is 16.5 years
  - 10 students have an average age of 15.3 years
  - 17 students have an average age of 16.7 years
  - 6 students have an average age of 18.4 years
  - 1 student has an age of 14.7 years
-/
theorem age_of_35th_student 
  (total_students : Nat) 
  (avg_age_all : ℝ)
  (num_group1 : Nat) (avg_age_group1 : ℝ)
  (num_group2 : Nat) (avg_age_group2 : ℝ)
  (num_group3 : Nat) (avg_age_group3 : ℝ)
  (num_group4 : Nat) (age_group4 : ℝ)
  (h1 : total_students = 35)
  (h2 : avg_age_all = 16.5)
  (h3 : num_group1 = 10)
  (h4 : avg_age_group1 = 15.3)
  (h5 : num_group2 = 17)
  (h6 : avg_age_group2 = 16.7)
  (h7 : num_group3 = 6)
  (h8 : avg_age_group3 = 18.4)
  (h9 : num_group4 = 1)
  (h10 : age_group4 = 14.7)
  (h11 : num_group1 + num_group2 + num_group3 + num_group4 + 1 = total_students) :
  (total_students : ℝ) * avg_age_all - 
  ((num_group1 : ℝ) * avg_age_group1 + 
   (num_group2 : ℝ) * avg_age_group2 + 
   (num_group3 : ℝ) * avg_age_group3 + 
   (num_group4 : ℝ) * age_group4) = 15.5 := by
  sorry

end NUMINAMATH_CALUDE_age_of_35th_student_l829_82987


namespace NUMINAMATH_CALUDE_chicken_rabbit_equations_correct_l829_82959

/-- Represents the "chicken-rabbit in the same cage" problem -/
structure ChickenRabbitProblem where
  total_heads : ℕ
  total_feet : ℕ
  chickens : ℕ
  rabbits : ℕ

/-- The system of equations for the chicken-rabbit problem -/
def correct_equations (problem : ChickenRabbitProblem) : Prop :=
  problem.chickens + problem.rabbits = problem.total_heads ∧
  2 * problem.chickens + 4 * problem.rabbits = problem.total_feet

/-- Theorem stating that the system of equations correctly represents the problem -/
theorem chicken_rabbit_equations_correct (problem : ChickenRabbitProblem) 
  (h1 : problem.total_heads = 35)
  (h2 : problem.total_feet = 94) :
  correct_equations problem :=
sorry

end NUMINAMATH_CALUDE_chicken_rabbit_equations_correct_l829_82959


namespace NUMINAMATH_CALUDE_unique_triple_l829_82944

theorem unique_triple : ∃! (x y z : ℕ), 
  100 ≤ x ∧ x < y ∧ y < z ∧ z < 1000 ∧ 
  (y - x = z - y) ∧ 
  (y * y = x * (z + 1000)) ∧
  x = 160 ∧ y = 560 ∧ z = 960 := by
  sorry

end NUMINAMATH_CALUDE_unique_triple_l829_82944


namespace NUMINAMATH_CALUDE_statue_of_liberty_model_height_l829_82964

/-- The scale ratio of the model to the actual size -/
def scaleRatio : ℚ := 1 / 25

/-- The actual height of the Statue of Liberty in feet -/
def actualHeight : ℕ := 305

/-- Rounds a rational number to the nearest integer -/
def roundToNearest (x : ℚ) : ℤ :=
  ⌊x + 1/2⌋

/-- The height of the scale model in feet -/
def modelHeight : ℚ := actualHeight * scaleRatio

theorem statue_of_liberty_model_height :
  roundToNearest modelHeight = 12 := by
  sorry

end NUMINAMATH_CALUDE_statue_of_liberty_model_height_l829_82964


namespace NUMINAMATH_CALUDE_flowers_used_for_bouquets_l829_82960

theorem flowers_used_for_bouquets (tulips roses extra_flowers : ℕ) :
  tulips = 4 → roses = 11 → extra_flowers = 4 →
  tulips + roses - extra_flowers = 11 := by
  sorry

end NUMINAMATH_CALUDE_flowers_used_for_bouquets_l829_82960


namespace NUMINAMATH_CALUDE_min_distance_midpoint_to_origin_min_distance_midpoint_to_origin_is_5sqrt2_l829_82978

/-- The minimum distance from the midpoint of two points on parallel lines x-y-5=0 and x-y-15=0 to the origin is 5√2. -/
theorem min_distance_midpoint_to_origin : ℝ → Prop := 
  fun d => ∀ (x₁ y₁ x₂ y₂ : ℝ),
    (x₁ - y₁ = 5) →
    (x₂ - y₂ = 15) →
    let midpoint_x := (x₁ + x₂) / 2
    let midpoint_y := (y₁ + y₂) / 2
    d = Real.sqrt 50

-- The proof is omitted
theorem min_distance_midpoint_to_origin_is_5sqrt2 : 
  min_distance_midpoint_to_origin (5 * Real.sqrt 2) :=
sorry

end NUMINAMATH_CALUDE_min_distance_midpoint_to_origin_min_distance_midpoint_to_origin_is_5sqrt2_l829_82978


namespace NUMINAMATH_CALUDE_annular_sector_area_l829_82927

/-- An annulus is the region between two concentric circles. -/
structure Annulus where
  R : ℝ
  r : ℝ
  h : R > r

/-- A point on the larger circle of the annulus. -/
structure PointOnLargerCircle (A : Annulus) where
  P : ℝ × ℝ
  h : (P.1 - 0)^2 + (P.2 - 0)^2 = A.R^2

/-- A point on the smaller circle of the annulus. -/
structure PointOnSmallerCircle (A : Annulus) where
  Q : ℝ × ℝ
  h : (Q.1 - 0)^2 + (Q.2 - 0)^2 = A.r^2

/-- A tangent line to the smaller circle. -/
def IsTangent (A : Annulus) (P : PointOnLargerCircle A) (Q : PointOnSmallerCircle A) : Prop :=
  (P.P.1 - Q.Q.1)^2 + (P.P.2 - Q.Q.2)^2 = A.R^2 - A.r^2

/-- The theorem stating the area of the annular sector. -/
theorem annular_sector_area (A : Annulus) (P : PointOnLargerCircle A) (Q : PointOnSmallerCircle A)
    (θ : ℝ) (t : ℝ) (h_tangent : IsTangent A P Q) (h_t : t^2 = A.R^2 - A.r^2) :
    (θ/2 - π) * A.r^2 + θ * t^2 / 2 = θ * A.R^2 / 2 - π * A.r^2 := by
  sorry

#check annular_sector_area

end NUMINAMATH_CALUDE_annular_sector_area_l829_82927


namespace NUMINAMATH_CALUDE_expand_polynomial_l829_82983

theorem expand_polynomial (x : ℝ) : (5*x^2 + 7*x + 2) * 3*x = 15*x^3 + 21*x^2 + 6*x := by
  sorry

end NUMINAMATH_CALUDE_expand_polynomial_l829_82983


namespace NUMINAMATH_CALUDE_investment_growth_l829_82965

/-- Calculates the final amount after simple interest --/
def final_amount (principal : ℝ) (rate : ℝ) (time : ℝ) : ℝ :=
  principal * (1 + rate * time)

/-- Theorem: Given the conditions, the final amount after 6 years is $380 --/
theorem investment_growth (principal : ℝ) (amount_after_2_years : ℝ) :
  principal = 200 →
  amount_after_2_years = 260 →
  final_amount principal ((amount_after_2_years - principal) / (principal * 2)) 6 = 380 :=
by sorry

end NUMINAMATH_CALUDE_investment_growth_l829_82965


namespace NUMINAMATH_CALUDE_company_price_ratio_l829_82917

/-- Given companies A, B, and KW where:
    - KW's price is 30% more than A's assets
    - KW's price is 100% more than B's assets
    Prove that KW's price is approximately 78.79% of A and B's combined assets -/
theorem company_price_ratio (P A B : ℝ) 
  (h1 : P = A + 0.3 * A) 
  (h2 : P = B + B) : 
  ∃ ε > 0, abs (P / (A + B) - 0.7879) < ε :=
sorry

end NUMINAMATH_CALUDE_company_price_ratio_l829_82917


namespace NUMINAMATH_CALUDE_temperature_rise_l829_82908

theorem temperature_rise (initial_temp final_temp rise : ℤ) : 
  initial_temp = -2 → rise = 3 → final_temp = initial_temp + rise → final_temp = 1 :=
by sorry

end NUMINAMATH_CALUDE_temperature_rise_l829_82908


namespace NUMINAMATH_CALUDE_sqrt_two_plus_pi_irrational_l829_82972

-- Define irrationality
def IsIrrational (x : ℝ) : Prop := ∀ (p q : ℤ), q ≠ 0 → x ≠ (p : ℝ) / (q : ℝ)

-- State the theorem
theorem sqrt_two_plus_pi_irrational :
  IsIrrational (Real.sqrt 2) → IsIrrational π → IsIrrational (Real.sqrt 2 + π) :=
by
  sorry

end NUMINAMATH_CALUDE_sqrt_two_plus_pi_irrational_l829_82972


namespace NUMINAMATH_CALUDE_runners_journey_l829_82985

/-- A runner's journey with changing speeds -/
theorem runners_journey (initial_speed : ℝ) (tired_speed : ℝ) (total_distance : ℝ) (total_time : ℝ)
  (h1 : initial_speed = 15)
  (h2 : tired_speed = 10)
  (h3 : total_distance = 100)
  (h4 : total_time = 9) :
  ∃ (initial_time : ℝ), initial_time = 2 ∧ 
    initial_speed * initial_time + tired_speed * (total_time - initial_time) = total_distance := by
  sorry

end NUMINAMATH_CALUDE_runners_journey_l829_82985


namespace NUMINAMATH_CALUDE_abs_T_equals_128_sqrt_2_l829_82939

-- Define the complex number i
def i : ℂ := Complex.I

-- Define T as in the problem
def T : ℂ := (1 + i)^15 - (1 - i)^15

-- Theorem statement
theorem abs_T_equals_128_sqrt_2 : Complex.abs T = 128 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_abs_T_equals_128_sqrt_2_l829_82939


namespace NUMINAMATH_CALUDE_fruit_drink_volume_l829_82955

/-- Represents a fruit drink composed of grapefruit, lemon, and orange juice -/
structure FruitDrink where
  total : ℝ
  grapefruit : ℝ
  lemon : ℝ
  orange : ℝ

/-- Theorem stating the total volume of the fruit drink -/
theorem fruit_drink_volume (drink : FruitDrink)
  (h1 : drink.grapefruit = 0.25 * drink.total)
  (h2 : drink.lemon = 0.35 * drink.total)
  (h3 : drink.orange = 20)
  (h4 : drink.total = drink.grapefruit + drink.lemon + drink.orange) :
  drink.total = 50 := by
  sorry


end NUMINAMATH_CALUDE_fruit_drink_volume_l829_82955


namespace NUMINAMATH_CALUDE_largest_non_expressible_l829_82984

def is_power_of_two (n : ℕ) : Prop := ∃ k : ℕ, n = 2^k

def expressible (n : ℕ) : Prop :=
  ∃ (a b : ℕ), n = 36 * a + b ∧ a > 0 ∧ is_power_of_two b

theorem largest_non_expressible : 
  (∀ n > 104, expressible n) ∧ ¬(expressible 104) := by sorry

end NUMINAMATH_CALUDE_largest_non_expressible_l829_82984


namespace NUMINAMATH_CALUDE_distribute_five_balls_three_boxes_l829_82936

/-- The number of ways to distribute indistinguishable balls into distinguishable boxes -/
def distribute_balls (n : ℕ) (k : ℕ) : ℕ := Nat.choose (n + k - 1) (k - 1)

/-- Theorem: There are 21 ways to distribute 5 indistinguishable balls into 3 distinguishable boxes -/
theorem distribute_five_balls_three_boxes : distribute_balls 5 3 = 21 := by
  sorry

end NUMINAMATH_CALUDE_distribute_five_balls_three_boxes_l829_82936


namespace NUMINAMATH_CALUDE_equidistant_point_y_coord_l829_82924

/-- The y-coordinate of the point on the y-axis equidistant from A(3, 0) and B(1, -6) is -7/3 -/
theorem equidistant_point_y_coord :
  ∃ y : ℝ, (3^2 + y^2 = 1^2 + (y + 6)^2) ∧ y = -7/3 := by
sorry

end NUMINAMATH_CALUDE_equidistant_point_y_coord_l829_82924


namespace NUMINAMATH_CALUDE_f_decreasing_on_interval_l829_82973

noncomputable def f (x : ℝ) : ℝ := Real.log (2 * x^2 - 3 * x + 1) / Real.log (1/3)

theorem f_decreasing_on_interval : 
  ∀ x y, 1 < x → x < y → f y < f x := by sorry

end NUMINAMATH_CALUDE_f_decreasing_on_interval_l829_82973


namespace NUMINAMATH_CALUDE_forgotten_angles_sum_l829_82909

theorem forgotten_angles_sum (n : ℕ) (partial_sum : ℝ) : 
  n ≥ 3 → 
  partial_sum = 2797 → 
  (n - 2) * 180 - partial_sum = 83 :=
by sorry

end NUMINAMATH_CALUDE_forgotten_angles_sum_l829_82909


namespace NUMINAMATH_CALUDE_nested_fraction_equation_l829_82999

theorem nested_fraction_equation (x : ℚ) : 
  3 + 1 / (2 + 1 / (3 + 3 / (4 + x))) = 225 / 68 → x = -102 / 19 := by
sorry

end NUMINAMATH_CALUDE_nested_fraction_equation_l829_82999


namespace NUMINAMATH_CALUDE_math_team_count_is_480_l829_82970

/-- The number of ways to form a six-member math team with 3 girls and 3 boys 
    from a club of 4 girls and 6 boys, where one team member is selected as captain -/
def math_team_count : ℕ := sorry

/-- The number of girls in the math club -/
def girls_in_club : ℕ := 4

/-- The number of boys in the math club -/
def boys_in_club : ℕ := 6

/-- The number of girls required in the team -/
def girls_in_team : ℕ := 3

/-- The number of boys required in the team -/
def boys_in_team : ℕ := 3

/-- The total number of team members -/
def team_size : ℕ := girls_in_team + boys_in_team

theorem math_team_count_is_480 : 
  math_team_count = (Nat.choose girls_in_club girls_in_team) * 
                    (Nat.choose boys_in_club boys_in_team) * 
                    team_size := by sorry

end NUMINAMATH_CALUDE_math_team_count_is_480_l829_82970


namespace NUMINAMATH_CALUDE_eight_digit_permutations_eq_1680_l829_82923

/-- The number of different positive, eight-digit integers that can be formed
    using the digits 2, 2, 2, 5, 5, 7, 9, and 9 -/
def eight_digit_permutations : ℕ :=
  Nat.factorial 8 / (Nat.factorial 3 * Nat.factorial 2 * Nat.factorial 2)

/-- Theorem stating that the number of different positive, eight-digit integers
    that can be formed using the digits 2, 2, 2, 5, 5, 7, 9, and 9 is 1680 -/
theorem eight_digit_permutations_eq_1680 :
  eight_digit_permutations = 1680 := by
  sorry

end NUMINAMATH_CALUDE_eight_digit_permutations_eq_1680_l829_82923


namespace NUMINAMATH_CALUDE_probability_neither_mix_l829_82947

/-- Represents the set of all buyers -/
def TotalBuyers : ℕ := 100

/-- Represents the number of buyers who purchase cake mix -/
def CakeMixBuyers : ℕ := 50

/-- Represents the number of buyers who purchase muffin mix -/
def MuffinMixBuyers : ℕ := 40

/-- Represents the number of buyers who purchase both cake mix and muffin mix -/
def BothMixesBuyers : ℕ := 19

/-- Theorem stating the probability of selecting a buyer who purchases neither cake mix nor muffin mix -/
theorem probability_neither_mix (TotalBuyers CakeMixBuyers MuffinMixBuyers BothMixesBuyers : ℕ) 
  (h1 : TotalBuyers = 100)
  (h2 : CakeMixBuyers = 50)
  (h3 : MuffinMixBuyers = 40)
  (h4 : BothMixesBuyers = 19) :
  (TotalBuyers - (CakeMixBuyers + MuffinMixBuyers - BothMixesBuyers)) / TotalBuyers = 29 / 100 := by
  sorry

end NUMINAMATH_CALUDE_probability_neither_mix_l829_82947


namespace NUMINAMATH_CALUDE_f_monotonicity_and_minimum_l829_82911

def f (a x : ℝ) := x^2 + 2*a*x + 2

def is_monotonic (f : ℝ → ℝ) (a b : ℝ) :=
  (∀ x y, a ≤ x ∧ x < y ∧ y ≤ b → f x < f y) ∨
  (∀ x y, a ≤ x ∧ x < y ∧ y ≤ b → f x > f y)

theorem f_monotonicity_and_minimum (a : ℝ) :
  (is_monotonic (f a) (-5) 5 ↔ a ≥ 5 ∨ a ≤ -5) ∧
  (∀ x ∈ Set.Icc (-5) 5, f a x ≥
    if a ≥ 5 then 27 - 10*a
    else if a ≥ -5 then 2 - a^2
    else 27 + 10*a) :=
  sorry

end NUMINAMATH_CALUDE_f_monotonicity_and_minimum_l829_82911


namespace NUMINAMATH_CALUDE_pear_problem_l829_82993

theorem pear_problem (alyssa_pears nancy_pears carlos_pears given_away : ℕ) 
  (h1 : alyssa_pears = 42)
  (h2 : nancy_pears = 17)
  (h3 : carlos_pears = 25)
  (h4 : given_away = 5) :
  alyssa_pears + nancy_pears + carlos_pears - 3 * given_away = 69 := by
  sorry

end NUMINAMATH_CALUDE_pear_problem_l829_82993


namespace NUMINAMATH_CALUDE_stuffed_animal_sales_difference_l829_82971

/-- Given the sales of stuffed animals by Quincy, Thor, and Jake, prove the difference between Quincy's and Jake's sales. -/
theorem stuffed_animal_sales_difference 
  (quincy thor jake : ℕ) 
  (h1 : quincy = 100 * thor) 
  (h2 : jake = thor + 15) 
  (h3 : quincy = 2000) : 
  quincy - jake = 1965 := by
sorry

end NUMINAMATH_CALUDE_stuffed_animal_sales_difference_l829_82971


namespace NUMINAMATH_CALUDE_ball_max_height_l829_82991

/-- The height function of the ball's parabolic path -/
def h (t : ℝ) : ℝ := -20 * t^2 + 80 * t + 36

/-- Theorem stating that the maximum height of the ball is 116 feet -/
theorem ball_max_height : 
  ∃ (max : ℝ), max = 116 ∧ ∀ (t : ℝ), h t ≤ max :=
sorry

end NUMINAMATH_CALUDE_ball_max_height_l829_82991


namespace NUMINAMATH_CALUDE_house_of_cards_height_l829_82945

/-- Given a triangular-shaped house of cards with base 40 cm,
    prove that if the total area of three similar houses is 1200 cm²,
    then the height of each house is 20 cm. -/
theorem house_of_cards_height
  (base : ℝ)
  (total_area : ℝ)
  (num_houses : ℕ)
  (h_base : base = 40)
  (h_total_area : total_area = 1200)
  (h_num_houses : num_houses = 3) :
  let area := total_area / num_houses
  let height := 2 * area / base
  height = 20 := by
sorry

end NUMINAMATH_CALUDE_house_of_cards_height_l829_82945


namespace NUMINAMATH_CALUDE_abs_diff_opposite_l829_82902

theorem abs_diff_opposite (x : ℝ) (h : x < 0) : |x - (-x)| = -2*x := by
  sorry

end NUMINAMATH_CALUDE_abs_diff_opposite_l829_82902


namespace NUMINAMATH_CALUDE_ball_hitting_ground_time_l829_82974

/-- The time it takes for a ball to hit the ground when thrown upward -/
theorem ball_hitting_ground_time :
  let initial_speed : ℝ := 5
  let initial_height : ℝ := 10
  let gravity : ℝ := 9.8
  let motion_equation (t : ℝ) : ℝ := -4.9 * t^2 + initial_speed * t + initial_height
  ∃ (t : ℝ), t > 0 ∧ motion_equation t = 0 ∧ t = 10/7 :=
by sorry

end NUMINAMATH_CALUDE_ball_hitting_ground_time_l829_82974


namespace NUMINAMATH_CALUDE_team_a_builds_30m_per_day_l829_82979

/-- Represents the daily road-building rate of Team A in meters -/
def team_a_rate : ℝ := 30

/-- Represents the daily road-building rate of Team B in meters -/
def team_b_rate : ℝ := team_a_rate + 10

/-- Represents the total length of road built by Team A in meters -/
def team_a_total : ℝ := 120

/-- Represents the total length of road built by Team B in meters -/
def team_b_total : ℝ := 160

/-- Theorem stating that Team A's daily rate is 30m, given the problem conditions -/
theorem team_a_builds_30m_per_day :
  (team_a_total / team_a_rate = team_b_total / team_b_rate) ∧
  (team_b_rate = team_a_rate + 10) ∧
  (team_a_rate = 30) := by sorry

end NUMINAMATH_CALUDE_team_a_builds_30m_per_day_l829_82979


namespace NUMINAMATH_CALUDE_train_length_l829_82957

/-- Given a train traveling at 72 km/hr crossing a 270 m long platform in 26 seconds,
    the length of the train is 250 meters. -/
theorem train_length (speed : ℝ) (platform_length : ℝ) (crossing_time : ℝ) :
  speed = 72 * 1000 / 3600 →
  platform_length = 270 →
  crossing_time = 26 →
  speed * crossing_time - platform_length = 250 :=
by sorry

end NUMINAMATH_CALUDE_train_length_l829_82957


namespace NUMINAMATH_CALUDE_solution_equivalence_l829_82976

-- Define the set of real numbers greater than 1
def greaterThanOne : Set ℝ := {x | x > 1}

-- Define the solution set of ax - b > 0
def solutionSet (a b : ℝ) : Set ℝ := {x | a * x - b > 0}

-- Define the set (-∞,-1)∪(2,+∞)
def targetSet : Set ℝ := {x | x < -1 ∨ x > 2}

-- Theorem statement
theorem solution_equivalence (a b : ℝ) :
  solutionSet a b = greaterThanOne →
  {x : ℝ | (a * x + b) / (x - 2) > 0} = targetSet := by
  sorry

end NUMINAMATH_CALUDE_solution_equivalence_l829_82976


namespace NUMINAMATH_CALUDE_pizza_theorem_l829_82954

def pizza_problem (total_pepperoni : ℕ) (fallen_pepperoni : ℕ) : Prop :=
  let half_pizza_pepperoni := total_pepperoni / 2
  let quarter_pizza_pepperoni := half_pizza_pepperoni / 2
  quarter_pizza_pepperoni - fallen_pepperoni = 9

theorem pizza_theorem : pizza_problem 40 1 := by
  sorry

end NUMINAMATH_CALUDE_pizza_theorem_l829_82954


namespace NUMINAMATH_CALUDE_water_and_milk_amounts_l829_82910

/-- Sarah's special bread recipe -/
def special_bread_recipe (flour water milk : ℚ) : Prop :=
  water / flour = 75 / 300 ∧ milk / flour = 60 / 300

/-- The amount of flour Sarah uses -/
def flour_amount : ℚ := 900

/-- The theorem stating the required amounts of water and milk -/
theorem water_and_milk_amounts :
  ∀ water milk : ℚ,
  special_bread_recipe flour_amount water milk →
  water = 225 ∧ milk = 180 := by sorry

end NUMINAMATH_CALUDE_water_and_milk_amounts_l829_82910


namespace NUMINAMATH_CALUDE_radio_show_duration_is_three_hours_l829_82907

/-- Calculates the total duration of a radio show in hours -/
def radio_show_duration (
  talking_segment_duration : ℕ)
  (ad_break_duration : ℕ)
  (num_talking_segments : ℕ)
  (num_ad_breaks : ℕ)
  (song_duration : ℕ) : ℚ :=
  let total_minutes : ℕ := 
    talking_segment_duration * num_talking_segments +
    ad_break_duration * num_ad_breaks +
    song_duration
  (total_minutes : ℚ) / 60

/-- Proves that given the specified conditions, the radio show duration is 3 hours -/
theorem radio_show_duration_is_three_hours :
  radio_show_duration 10 5 3 5 125 = 3 := by
  sorry

end NUMINAMATH_CALUDE_radio_show_duration_is_three_hours_l829_82907


namespace NUMINAMATH_CALUDE_average_age_after_leaving_l829_82937

theorem average_age_after_leaving (initial_people : ℕ) (initial_average : ℚ) 
  (leaving_age1 : ℕ) (leaving_age2 : ℕ) (remaining_people : ℕ) :
  initial_people = 6 →
  initial_average = 25 →
  leaving_age1 = 20 →
  leaving_age2 = 22 →
  remaining_people = 4 →
  (initial_people : ℚ) * initial_average - (leaving_age1 + leaving_age2 : ℚ) = 
    (remaining_people : ℚ) * 27 := by
  sorry

#check average_age_after_leaving

end NUMINAMATH_CALUDE_average_age_after_leaving_l829_82937


namespace NUMINAMATH_CALUDE_circle_center_radius_sum_l829_82996

/-- Given a circle D with equation x^2 + 20x + y^2 + 18y = -36,
    prove that its center coordinates (p, q) and radius s satisfy p + q + s = -19 + Real.sqrt 145 -/
theorem circle_center_radius_sum (x y : ℝ) :
  x^2 + 20*x + y^2 + 18*y = -36 →
  ∃ (p q s : ℝ), (∀ (x y : ℝ), (x - p)^2 + (y - q)^2 = s^2 ↔ x^2 + 20*x + y^2 + 18*y = -36) ∧
                 p + q + s = -19 + Real.sqrt 145 := by
  sorry

end NUMINAMATH_CALUDE_circle_center_radius_sum_l829_82996


namespace NUMINAMATH_CALUDE_range_of_4a_minus_2b_l829_82967

theorem range_of_4a_minus_2b (a b : ℝ) 
  (h1 : 0 ≤ a - b) (h2 : a - b ≤ 1) 
  (h3 : 2 ≤ a + b) (h4 : a + b ≤ 4) : 
  2 ≤ 4 * a - 2 * b ∧ 4 * a - 2 * b ≤ 7 := by
  sorry

end NUMINAMATH_CALUDE_range_of_4a_minus_2b_l829_82967


namespace NUMINAMATH_CALUDE_manager_count_is_two_l829_82989

/-- Represents the daily salary structure and employee count in a grocery store -/
structure GroceryStore where
  managerSalary : ℕ
  clerkSalary : ℕ
  clerkCount : ℕ
  totalSalary : ℕ

/-- Calculates the number of managers in the grocery store -/
def managerCount (store : GroceryStore) : ℕ :=
  (store.totalSalary - store.clerkSalary * store.clerkCount) / store.managerSalary

/-- Theorem stating that the number of managers in the given scenario is 2 -/
theorem manager_count_is_two :
  let store : GroceryStore := {
    managerSalary := 5,
    clerkSalary := 2,
    clerkCount := 3,
    totalSalary := 16
  }
  managerCount store = 2 := by sorry

end NUMINAMATH_CALUDE_manager_count_is_two_l829_82989


namespace NUMINAMATH_CALUDE_smallest_two_digit_with_product_12_l829_82951

def is_two_digit (n : ℕ) : Prop := 10 ≤ n ∧ n ≤ 99

def digit_product (n : ℕ) : ℕ :=
  (n / 10) * (n % 10)

theorem smallest_two_digit_with_product_12 :
  ∃ (n : ℕ), is_two_digit n ∧ digit_product n = 12 ∧
  ∀ (m : ℕ), is_two_digit m → digit_product m = 12 → n ≤ m :=
by
  sorry

end NUMINAMATH_CALUDE_smallest_two_digit_with_product_12_l829_82951


namespace NUMINAMATH_CALUDE_ratio_sum_problem_l829_82990

theorem ratio_sum_problem (a b c : ℕ) : 
  a + b + c = 108 → 
  5 * b = 3 * a → 
  4 * b = 3 * c → 
  b = 27 := by
sorry

end NUMINAMATH_CALUDE_ratio_sum_problem_l829_82990


namespace NUMINAMATH_CALUDE_complex_fraction_simplification_l829_82966

theorem complex_fraction_simplification :
  let i : ℂ := Complex.I
  (2 - i) / (1 + i) = (1 : ℂ) / 2 - (3 : ℂ) / 2 * i := by sorry

end NUMINAMATH_CALUDE_complex_fraction_simplification_l829_82966


namespace NUMINAMATH_CALUDE_solutions_equation1_solutions_equation2_solutions_equation3_l829_82969

-- Define the quadratic equations
def equation1 (x : ℝ) : Prop := 4 * (x - 1)^2 - 36 = 0
def equation2 (x : ℝ) : Prop := x^2 + 2*x - 3 = 0
def equation3 (x : ℝ) : Prop := x*(x - 4) = 8 - 2*x

-- Theorem stating the solutions for equation1
theorem solutions_equation1 : 
  ∀ x : ℝ, equation1 x ↔ (x = 4 ∨ x = -2) :=
sorry

-- Theorem stating the solutions for equation2
theorem solutions_equation2 : 
  ∀ x : ℝ, equation2 x ↔ (x = -3 ∨ x = 1) :=
sorry

-- Theorem stating the solutions for equation3
theorem solutions_equation3 : 
  ∀ x : ℝ, equation3 x ↔ (x = 4 ∨ x = -2) :=
sorry

end NUMINAMATH_CALUDE_solutions_equation1_solutions_equation2_solutions_equation3_l829_82969


namespace NUMINAMATH_CALUDE_johns_presents_worth_l829_82981

/-- The total worth of John's presents to his fiancee -/
def total_worth (ring_cost car_cost brace_cost : ℕ) : ℕ :=
  ring_cost + car_cost + brace_cost

/-- Theorem stating the total worth of John's presents -/
theorem johns_presents_worth :
  ∃ (ring_cost car_cost brace_cost : ℕ),
    ring_cost = 4000 ∧
    car_cost = 2000 ∧
    brace_cost = 2 * ring_cost ∧
    total_worth ring_cost car_cost brace_cost = 14000 := by
  sorry

end NUMINAMATH_CALUDE_johns_presents_worth_l829_82981


namespace NUMINAMATH_CALUDE_converse_of_square_equals_one_l829_82968

theorem converse_of_square_equals_one (a : ℝ) : 
  (∀ a, a = 1 → a^2 = 1) → (∀ a, a^2 = 1 → a = 1) := by
  sorry

end NUMINAMATH_CALUDE_converse_of_square_equals_one_l829_82968


namespace NUMINAMATH_CALUDE_system_solution_l829_82935

theorem system_solution (x y z : ℝ) : 
  (x * y + z = 40 ∧ x * z + y = 51 ∧ x + y + z = 19) ↔ 
  ((x = 12 ∧ y = 3 ∧ z = 4) ∨ (x = 6 ∧ y = 5.4 ∧ z = 7.6)) := by
sorry

end NUMINAMATH_CALUDE_system_solution_l829_82935


namespace NUMINAMATH_CALUDE_correct_sample_l829_82941

def random_number_table : List (List Nat) := [
  [16, 22, 77, 94, 39, 49, 54, 43, 54, 82, 17, 37, 93, 23, 78, 87, 35, 20, 96, 43, 84, 26, 34, 91, 64],
  [84, 42, 17, 53, 31, 57, 24, 55, 06, 88, 77, 04, 74, 47, 67, 21, 76, 33, 50, 25, 83, 92, 12, 06, 76],
  [63, 01, 63, 78, 59, 16, 95, 55, 67, 19, 98, 10, 50, 71, 75, 12, 86, 73, 58, 07, 44, 39, 52, 38, 79],
  [33, 21, 12, 34, 29, 78, 64, 56, 07, 82, 52, 42, 07, 44, 38, 15, 51, 00, 13, 42, 99, 66, 02, 79, 54],
  [57, 60, 86, 32, 44, 09, 47, 27, 96, 54, 49, 17, 46, 09, 62, 90, 52, 84, 77, 27, 08, 02, 73, 43, 28]
]

def start_row : Nat := 5
def start_col : Nat := 4
def total_bottles : Nat := 80
def sample_size : Nat := 6

def is_valid_bottle (n : Nat) : Bool :=
  n < total_bottles

def select_sample (table : List (List Nat)) (row : Nat) (col : Nat) : List Nat :=
  sorry

theorem correct_sample :
  select_sample random_number_table start_row start_col = [77, 39, 49, 54, 43, 17] :=
by sorry

end NUMINAMATH_CALUDE_correct_sample_l829_82941


namespace NUMINAMATH_CALUDE_hemisphere_base_area_l829_82918

theorem hemisphere_base_area (r : ℝ) (h : r > 0) : 3 * Real.pi * r^2 = 9 → Real.pi * r^2 = 3 := by
  sorry

end NUMINAMATH_CALUDE_hemisphere_base_area_l829_82918


namespace NUMINAMATH_CALUDE_smallest_three_square_representations_l829_82977

/-- A function that represents the number of ways a positive integer can be expressed as the sum of three squares -/
def numThreeSquareRepresentations (n : ℕ) : ℕ := sorry

/-- A predicate that checks if a number is expressible as the sum of three squares in three different ways -/
def hasThreeRepresentations (n : ℕ) : Prop :=
  numThreeSquareRepresentations n = 3

/-- Theorem stating that 30 is the smallest positive integer with three different representations as the sum of three squares -/
theorem smallest_three_square_representations :
  (∀ m : ℕ, m > 0 → m < 30 → ¬(hasThreeRepresentations m)) ∧
  hasThreeRepresentations 30 := by sorry

end NUMINAMATH_CALUDE_smallest_three_square_representations_l829_82977


namespace NUMINAMATH_CALUDE_cubic_polynomial_roots_l829_82994

theorem cubic_polynomial_roots (a b c : ℚ) :
  (∃ x : ℝ, x^3 + a*x^2 + b*x + c = 0 ∧ x = 2 - Real.sqrt 5) →
  (∃ x : ℝ, x^3 + a*x^2 + b*x + c = 0 ∧ x = 1) →
  (∃ x : ℝ, x^3 + a*x^2 + b*x + c = 0 ∧ x = 1 ∧ x ≠ 2 - Real.sqrt 5) :=
by sorry

end NUMINAMATH_CALUDE_cubic_polynomial_roots_l829_82994


namespace NUMINAMATH_CALUDE_arithmetic_calculations_l829_82905

theorem arithmetic_calculations :
  (72 * 54 + 28 * 54 = 5400) ∧
  (60 * 25 * 8 = 12000) ∧
  (2790 / (250 * 12 - 2910) = 31) ∧
  ((100 - 1456 / 26) * 78 = 3432) := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_calculations_l829_82905


namespace NUMINAMATH_CALUDE_fraction_equality_l829_82958

theorem fraction_equality (m n r t : ℚ) 
  (h1 : m / n = 7 / 5) 
  (h2 : r / t = 8 / 15) : 
  (5 * m * r - 2 * n * t) / (6 * n * t - 8 * m * r) = 65 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_l829_82958


namespace NUMINAMATH_CALUDE_manuscript_completion_time_l829_82926

theorem manuscript_completion_time 
  (original_computers : ℕ) 
  (original_time : ℚ) 
  (reduced_time_ratio : ℚ) 
  (additional_time : ℚ) :
  (original_computers : ℚ) / (original_computers + 3) = reduced_time_ratio →
  (original_computers : ℚ) / (original_computers - 3) = original_time / (original_time + additional_time) →
  reduced_time_ratio = 3/4 →
  additional_time = 5/6 →
  original_time = 5/3 := by
sorry

end NUMINAMATH_CALUDE_manuscript_completion_time_l829_82926


namespace NUMINAMATH_CALUDE_max_square_plots_l829_82933

/-- Represents the dimensions of the rectangular field -/
structure FieldDimensions where
  length : ℝ
  width : ℝ

/-- Represents the available internal fencing -/
def availableFencing : ℝ := 2400

/-- Calculates the number of square plots given the number of plots along the width -/
def numPlots (n : ℕ) : ℕ := n * n * 2

/-- Calculates the amount of internal fencing needed for a given number of plots along the width -/
def fencingNeeded (n : ℕ) (field : FieldDimensions) : ℝ :=
  (2 * n - 1) * field.width + (n - 1) * field.length

/-- The main theorem stating the maximum number of square plots -/
theorem max_square_plots (field : FieldDimensions) 
    (h_length : field.length = 60) 
    (h_width : field.width = 30) :
    ∃ (n : ℕ), 
      numPlots n = 400 ∧ 
      fencingNeeded n field ≤ availableFencing ∧
      ∀ (m : ℕ), fencingNeeded m field ≤ availableFencing → numPlots m ≤ numPlots n := by
  sorry


end NUMINAMATH_CALUDE_max_square_plots_l829_82933


namespace NUMINAMATH_CALUDE_jessica_age_problem_l829_82952

/-- Jessica's age problem -/
theorem jessica_age_problem (jessica_age_at_death : ℕ) (mother_age_at_death : ℕ) (years_since_death : ℕ) :
  jessica_age_at_death = mother_age_at_death / 2 →
  mother_age_at_death + years_since_death = 70 →
  years_since_death = 10 →
  jessica_age_at_death + years_since_death = 40 :=
by
  sorry

#check jessica_age_problem

end NUMINAMATH_CALUDE_jessica_age_problem_l829_82952


namespace NUMINAMATH_CALUDE_division_problem_l829_82900

theorem division_problem (a b c : ℚ) 
  (h1 : a / b = 3) 
  (h2 : b / c = 2 / 5) : 
  c / a = 5 / 6 := by
  sorry

end NUMINAMATH_CALUDE_division_problem_l829_82900


namespace NUMINAMATH_CALUDE_sequence_a_int_l829_82961

def sequence_a (c : ℕ) : ℕ → ℤ
  | 0 => 2
  | n + 1 => c * sequence_a c n + Int.sqrt ((c^2 - 1) * (sequence_a c n^2 - 4))

theorem sequence_a_int (c : ℕ) (hc : c ≥ 1) :
  ∀ n : ℕ, ∃ k : ℤ, sequence_a c n = k :=
by sorry

end NUMINAMATH_CALUDE_sequence_a_int_l829_82961


namespace NUMINAMATH_CALUDE_sqrt_13_parts_sum_l829_82931

theorem sqrt_13_parts_sum (x y : ℝ) : 
  (x = ⌊Real.sqrt 13⌋) → 
  (y = Real.sqrt 13 - ⌊Real.sqrt 13⌋) → 
  (2 * x - y + Real.sqrt 13 = 9) := by
sorry

end NUMINAMATH_CALUDE_sqrt_13_parts_sum_l829_82931


namespace NUMINAMATH_CALUDE_product_mod_75_l829_82950

theorem product_mod_75 : ∃ m : ℕ, 198 * 864 ≡ m [ZMOD 75] ∧ 0 ≤ m ∧ m < 75 :=
by
  use 72
  sorry

end NUMINAMATH_CALUDE_product_mod_75_l829_82950


namespace NUMINAMATH_CALUDE_repeated_root_fraction_equation_l829_82980

theorem repeated_root_fraction_equation (x m : ℝ) : 
  (∃ x, (x / (x - 3) + 1 = m / (x - 3)) ∧ 
        (∀ y, y ≠ x → y / (y - 3) + 1 ≠ m / (y - 3))) → 
  m = 3 := by
sorry

end NUMINAMATH_CALUDE_repeated_root_fraction_equation_l829_82980


namespace NUMINAMATH_CALUDE_ferry_problem_l829_82922

/-- The ferry problem -/
theorem ferry_problem (speed_p speed_q : ℝ) (time_p distance_q : ℝ) :
  speed_p = 8 →
  time_p = 2 →
  distance_q = 3 * speed_p * time_p →
  speed_q = speed_p + 4 →
  distance_q / speed_q - time_p = 2 := by
  sorry

end NUMINAMATH_CALUDE_ferry_problem_l829_82922


namespace NUMINAMATH_CALUDE_molecular_weight_AlI3_correct_l829_82956

/-- The molecular weight of AlI3 in grams per mole -/
def molecular_weight_AlI3 : ℝ := 408

/-- The number of moles given in the problem -/
def num_moles : ℝ := 8

/-- The total weight of the given number of moles in grams -/
def total_weight : ℝ := 3264

/-- Theorem stating that the molecular weight of AlI3 is correct -/
theorem molecular_weight_AlI3_correct : 
  molecular_weight_AlI3 = total_weight / num_moles :=
sorry

end NUMINAMATH_CALUDE_molecular_weight_AlI3_correct_l829_82956


namespace NUMINAMATH_CALUDE_smallest_divisible_by_8_11_15_l829_82982

theorem smallest_divisible_by_8_11_15 : ∃! n : ℕ+, 
  (∀ m : ℕ+, m < n → ¬(8 ∣ m ∧ 11 ∣ m ∧ 15 ∣ m)) ∧ 
  (8 ∣ n) ∧ (11 ∣ n) ∧ (15 ∣ n) := by
  sorry

end NUMINAMATH_CALUDE_smallest_divisible_by_8_11_15_l829_82982


namespace NUMINAMATH_CALUDE_fractional_equation_root_l829_82962

theorem fractional_equation_root (m : ℝ) : 
  (∃ x : ℝ, x ≠ 4 ∧ (3 / (x - 4) + (x + m) / (4 - x) = 1)) → m = -1 := by
  sorry

end NUMINAMATH_CALUDE_fractional_equation_root_l829_82962


namespace NUMINAMATH_CALUDE_rockets_win_in_7_l829_82901

/-- Probability of Warriors winning a single game -/
def p_warriors : ℚ := 3/4

/-- Probability of Rockets winning a single game -/
def p_rockets : ℚ := 1 - p_warriors

/-- Number of games needed to win the series -/
def games_to_win : ℕ := 4

/-- Maximum number of games in the series -/
def max_games : ℕ := 7

/-- Probability of Rockets winning the series in exactly 7 games -/
def p_rockets_win_in_7 : ℚ := 135/4096

theorem rockets_win_in_7 :
  p_rockets_win_in_7 = (Nat.choose 6 3 : ℚ) * p_rockets^3 * p_warriors^3 * p_rockets :=
by sorry

end NUMINAMATH_CALUDE_rockets_win_in_7_l829_82901


namespace NUMINAMATH_CALUDE_lcm_gcd_product_10_15_l829_82940

theorem lcm_gcd_product_10_15 :
  Nat.lcm 10 15 * Nat.gcd 10 15 = 150 := by
  sorry

end NUMINAMATH_CALUDE_lcm_gcd_product_10_15_l829_82940


namespace NUMINAMATH_CALUDE_functional_equation_solution_l829_82963

/-- A continuous function from positive reals to positive reals satisfying the given functional equation -/
def FunctionalEquation (f : ℝ → ℝ) : Prop :=
  Continuous f ∧ 
  (∀ x, x > 0 → f x > 0) ∧
  ∀ x y, x > 0 → y > 0 → f (x + y) * (f x + f y) = f x * f y

/-- The theorem stating that any function satisfying the functional equation has the form f(x) = 1/(αx) for some α > 0 -/
theorem functional_equation_solution 
  (f : ℝ → ℝ) 
  (h : FunctionalEquation f) :
  ∃ α : ℝ, α > 0 ∧ ∀ x, x > 0 → f x = 1 / (α * x) :=
sorry

end NUMINAMATH_CALUDE_functional_equation_solution_l829_82963


namespace NUMINAMATH_CALUDE_sequence_difference_l829_82948

def sequence_property (a : ℕ+ → ℝ) (S : ℕ+ → ℝ) : Prop :=
  (∀ n : ℕ+, a n ≠ 0) ∧ 
  (∀ n : ℕ+, a n * a (n + 1) = S n)

theorem sequence_difference (a : ℕ+ → ℝ) (S : ℕ+ → ℝ) 
  (h : sequence_property a S) : a 3 - a 1 = 1 := by
  sorry

end NUMINAMATH_CALUDE_sequence_difference_l829_82948


namespace NUMINAMATH_CALUDE_water_needed_to_fill_glasses_l829_82934

theorem water_needed_to_fill_glasses (num_glasses : ℕ) (glass_capacity : ℚ) (current_fullness : ℚ) :
  num_glasses = 10 →
  glass_capacity = 6 →
  current_fullness = 4/5 →
  (num_glasses : ℚ) * glass_capacity * (1 - current_fullness) = 12 := by
  sorry

end NUMINAMATH_CALUDE_water_needed_to_fill_glasses_l829_82934


namespace NUMINAMATH_CALUDE_no_negative_one_in_sequence_l829_82912

def recurrence_sequence (p : ℕ) : ℕ → ℤ
  | 0 => 0
  | 1 => 1
  | (n + 2) => 2 * recurrence_sequence p (n + 1) - p * recurrence_sequence p n

theorem no_negative_one_in_sequence (p : ℕ) (h_prime : Nat.Prime p) (h_odd : Odd p) (h_not_five : p ≠ 5) :
  ∀ n, recurrence_sequence p n ≠ -1 :=
by sorry

end NUMINAMATH_CALUDE_no_negative_one_in_sequence_l829_82912


namespace NUMINAMATH_CALUDE_charlie_data_overage_cost_l829_82997

/-- Charlie's cell phone plan data usage and cost calculation --/
theorem charlie_data_overage_cost
  (data_limit : ℕ)
  (week1_usage week2_usage week3_usage week4_usage : ℕ)
  (overage_charge : ℕ)
  (h1 : data_limit = 8)
  (h2 : week1_usage = 2)
  (h3 : week2_usage = 3)
  (h4 : week3_usage = 5)
  (h5 : week4_usage = 10)
  (h6 : overage_charge = 120) :
  let total_usage := week1_usage + week2_usage + week3_usage + week4_usage
  let overage := total_usage - data_limit
  overage_charge / overage = 10 := by sorry

end NUMINAMATH_CALUDE_charlie_data_overage_cost_l829_82997


namespace NUMINAMATH_CALUDE_inequality_solution_set_l829_82946

theorem inequality_solution_set (a b : ℝ) (d : ℝ) (h1 : a > 0) 
  (h2 : ∀ x : ℝ, x^2 + a*x + b > 0 ↔ x ≠ d) :
  ∃ x₁ x₂ : ℝ, (∀ x : ℝ, x^2 + a*x - b < 0 ↔ x₁ < x ∧ x < x₂) ∧ x₁ * x₂ ≤ 0 := by
  sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l829_82946


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_condition_l829_82921

/-- 
A proposition stating that "a>2 and b>2" is a sufficient but not necessary condition 
for "a+b>4 and ab>4" for real numbers a and b.
-/
theorem sufficient_not_necessary_condition (a b : ℝ) : 
  (∃ x y : ℝ, x > 2 ∧ y > 2 → x + y > 4 ∧ x * y > 4) ∧ 
  (∃ p q : ℝ, p + q > 4 ∧ p * q > 4 ∧ ¬(p > 2 ∧ q > 2)) :=
by sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_condition_l829_82921


namespace NUMINAMATH_CALUDE_weight_difference_l829_82938

/-- Proves that given Robbie's weight, Patty's initial weight relative to Robbie's, and Patty's weight loss, 
    the difference between Patty's current weight and Robbie's weight is 115 pounds. -/
theorem weight_difference (robbie_weight : ℝ) (patty_initial_factor : ℝ) (patty_weight_loss : ℝ) : 
  robbie_weight = 100 → 
  patty_initial_factor = 4.5 → 
  patty_weight_loss = 235 → 
  patty_initial_factor * robbie_weight - patty_weight_loss - robbie_weight = 115 := by
  sorry


end NUMINAMATH_CALUDE_weight_difference_l829_82938


namespace NUMINAMATH_CALUDE_solutions_equation_1_solutions_equation_2_l829_82920

-- Equation 1
theorem solutions_equation_1 : 
  ∀ x : ℝ, x^2 - 2*x - 8 = 0 ↔ x = -2 ∨ x = 4 := by sorry

-- Equation 2
theorem solutions_equation_2 : 
  ∀ x : ℝ, x*(x-1) + 2*(x-1) = 0 ↔ x = 1 ∨ x = -2 := by sorry

end NUMINAMATH_CALUDE_solutions_equation_1_solutions_equation_2_l829_82920


namespace NUMINAMATH_CALUDE_ellipse_hyperbola_same_foci_l829_82992

/-- Given an ellipse and a hyperbola with the same foci, prove that the parameter a of the ellipse is 4 -/
theorem ellipse_hyperbola_same_foci (a : ℝ) : 
  (∀ x y : ℝ, x^2 / a^2 + y^2 / 9 = 1 → a > 0) → -- Ellipse equation condition
  (∀ x y : ℝ, x^2 / 4 - y^2 / 3 = 1) → -- Hyperbola equation
  (∃ c : ℝ, c^2 = 7 ∧ 
    (∀ x y : ℝ, x^2 / a^2 + y^2 / 9 = 1 → x^2 + y^2 = a^2 + c^2) ∧ 
    (∀ x y : ℝ, x^2 / 4 - y^2 / 3 = 1 → x^2 - y^2 = 4 + c^2)) → -- Same foci condition
  a = 4 := by
sorry

end NUMINAMATH_CALUDE_ellipse_hyperbola_same_foci_l829_82992


namespace NUMINAMATH_CALUDE_expression_result_l829_82929

theorem expression_result : 
  (0.66 : ℝ)^3 - (0.1 : ℝ)^3 / (0.66 : ℝ)^2 + 0.066 + (0.1 : ℝ)^2 = 0.3612 := by
  sorry

end NUMINAMATH_CALUDE_expression_result_l829_82929


namespace NUMINAMATH_CALUDE_second_floor_bedrooms_l829_82930

theorem second_floor_bedrooms (total_bedrooms first_floor_bedrooms : ℕ) 
  (h1 : total_bedrooms = 10)
  (h2 : first_floor_bedrooms = 8) :
  total_bedrooms - first_floor_bedrooms = 2 := by
  sorry

end NUMINAMATH_CALUDE_second_floor_bedrooms_l829_82930


namespace NUMINAMATH_CALUDE_sqrt_inequality_l829_82916

theorem sqrt_inequality (x : ℝ) (h : x ≥ 4) :
  Real.sqrt (x - 3) + Real.sqrt (x - 2) > Real.sqrt (x - 4) + Real.sqrt (x - 1) := by
  sorry

end NUMINAMATH_CALUDE_sqrt_inequality_l829_82916


namespace NUMINAMATH_CALUDE_playground_area_l829_82995

/-- The area of a rectangular playground with given conditions -/
theorem playground_area : 
  ∀ (width length : ℝ),
  length = 3 * width + 30 →
  2 * (width + length) = 730 →
  width * length = 23554.6875 := by
sorry

end NUMINAMATH_CALUDE_playground_area_l829_82995


namespace NUMINAMATH_CALUDE_vector_sum_zero_l829_82975

variable {V : Type*} [AddCommGroup V]

theorem vector_sum_zero (A B C : V) : (B - A) + (A - C) - (B - C) = 0 := by
  sorry

end NUMINAMATH_CALUDE_vector_sum_zero_l829_82975


namespace NUMINAMATH_CALUDE_employee_reduction_l829_82949

theorem employee_reduction (original_employees : ℝ) (reduction_percentage : ℝ) : 
  original_employees = 243.75 → 
  reduction_percentage = 0.20 → 
  original_employees * (1 - reduction_percentage) = 195 := by
  sorry


end NUMINAMATH_CALUDE_employee_reduction_l829_82949


namespace NUMINAMATH_CALUDE_triangle_properties_l829_82943

/-- Triangle ABC with sides a, b, c opposite to angles A, B, C respectively -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ
  pos_sides : 0 < a ∧ 0 < b ∧ 0 < c
  pos_angles : 0 < A ∧ 0 < B ∧ 0 < C
  angle_sum : A + B + C = π
  law_of_sines : a / Real.sin A = b / Real.sin B
  law_of_cosines : c^2 = a^2 + b^2 - 2*a*b*Real.cos C

theorem triangle_properties (t : Triangle) :
  (∃ k : ℝ, t.a = 2*k ∧ t.b = 3*k ∧ t.c = 4*k → Real.cos t.C < 0) ∧
  (Real.sin t.A > Real.sin t.B → t.A > t.B) ∧
  (t.C = π/3 ∧ t.b = 10 ∧ t.c = 9 → ∃ t1 t2 : Triangle, t1 ≠ t2 ∧ 
    t1.b = t.b ∧ t1.c = t.c ∧ t1.C = t.C ∧
    t2.b = t.b ∧ t2.c = t.c ∧ t2.C = t.C) :=
by sorry

end NUMINAMATH_CALUDE_triangle_properties_l829_82943


namespace NUMINAMATH_CALUDE_simplify_expression_l829_82913

theorem simplify_expression (a b : ℝ) (ha : a > 0) (hb : b > 0) 
  (h : a^3 + b^3 = a + b - 1) : 
  a / b + b / a - 2 / (a * b) = -1 - 1 / (a * b) := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l829_82913


namespace NUMINAMATH_CALUDE_funfair_visitors_l829_82906

theorem funfair_visitors (a : ℕ) : 
  a > 0 ∧ 
  (50 * a - 40 : ℤ) > 0 ∧ 
  (90 - 20 * a : ℤ) > 0 ∧ 
  (50 * a - 40 : ℤ) > (90 - 20 * a : ℤ) →
  (50 * a - 40 : ℤ) = 60 ∨ (50 * a - 40 : ℤ) = 110 ∨ (50 * a - 40 : ℤ) = 160 :=
by sorry

end NUMINAMATH_CALUDE_funfair_visitors_l829_82906
