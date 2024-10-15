import Mathlib

namespace NUMINAMATH_CALUDE_triangle_properties_l3556_355682

structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ

def altitude_equation (t : Triangle) : ℝ → ℝ → Prop :=
  fun x y => x + 5 * y - 3 = 0

def side_BC_equation (t : Triangle) : ℝ → ℝ → Prop :=
  fun x y => x + 2 * y - 10 = 0

theorem triangle_properties (t : Triangle) :
  t.A = (-2, 1) →
  t.B = (4, 3) →
  (t.C = (3, -2) → altitude_equation t t.A.1 t.A.2) ∧
  (∃ M : ℝ × ℝ, M = (3, 1) ∧ M.1 = (t.A.1 + t.C.1) / 2 ∧ M.2 = (t.A.2 + t.C.2) / 2 →
    side_BC_equation t t.B.1 t.B.2) :=
by
  sorry

end NUMINAMATH_CALUDE_triangle_properties_l3556_355682


namespace NUMINAMATH_CALUDE_unique_solution_quadratic_l3556_355662

/-- A quadratic equation qx^2 - 18x + 8 = 0 has only one solution when q = 81/8 -/
theorem unique_solution_quadratic :
  ∃! (x : ℝ), (81/8 : ℝ) * x^2 - 18 * x + 8 = 0 := by sorry

end NUMINAMATH_CALUDE_unique_solution_quadratic_l3556_355662


namespace NUMINAMATH_CALUDE_max_value_of_n_l3556_355627

theorem max_value_of_n (a b c d n : ℝ) 
  (h1 : a > b) (h2 : b > c) (h3 : c > d)
  (h4 : 1 / (a - b) + 1 / (b - c) + 1 / (c - d) ≥ n / (a - d)) :
  n ≤ 9 ∧ ∃ (a b c d : ℝ), a > b ∧ b > c ∧ c > d ∧ 
    1 / (a - b) + 1 / (b - c) + 1 / (c - d) = 9 / (a - d) :=
sorry

end NUMINAMATH_CALUDE_max_value_of_n_l3556_355627


namespace NUMINAMATH_CALUDE_john_jury_duty_days_l3556_355603

/-- Calculates the total number of days spent on jury duty given the specified conditions. -/
def juryDutyDays (jurySelectionDays : ℕ) (trialMultiplier : ℕ) (deliberationFullDays : ℕ) (deliberationHoursPerDay : ℕ) : ℕ :=
  let trialDays := jurySelectionDays * trialMultiplier
  let deliberationHours := deliberationFullDays * deliberationHoursPerDay
  let deliberationDays := deliberationHours / 24
  jurySelectionDays + trialDays + deliberationDays

/-- Theorem stating that under the given conditions, John spends 14 days on jury duty. -/
theorem john_jury_duty_days :
  juryDutyDays 2 4 6 16 = 14 := by
  sorry

#eval juryDutyDays 2 4 6 16

end NUMINAMATH_CALUDE_john_jury_duty_days_l3556_355603


namespace NUMINAMATH_CALUDE_smallest_non_odd_units_digit_l3556_355624

def OddUnitsDigits : Set Nat := {1, 3, 5, 7, 9}

def Digits : Set Nat := {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}

theorem smallest_non_odd_units_digit :
  ∃ (d : Nat), d ∈ Digits ∧ d ∉ OddUnitsDigits ∧ ∀ (x : Nat), x ∈ Digits ∧ x ∉ OddUnitsDigits → d ≤ x :=
by
  sorry

end NUMINAMATH_CALUDE_smallest_non_odd_units_digit_l3556_355624


namespace NUMINAMATH_CALUDE_least_clock_equivalent_after_10_l3556_355638

def clock_equivalent (h : ℕ) : Prop :=
  (h ^ 2 - h) % 12 = 0

theorem least_clock_equivalent_after_10 :
  ∀ h : ℕ, h > 10 → clock_equivalent h → h ≥ 12 :=
by
  sorry

end NUMINAMATH_CALUDE_least_clock_equivalent_after_10_l3556_355638


namespace NUMINAMATH_CALUDE_forest_to_verdant_green_conversion_l3556_355612

/-- Represents the ratio of blue to yellow paint in forest green -/
def forest_green_ratio : ℚ := 4 / 3

/-- Represents the ratio of yellow to blue paint in verdant green -/
def verdant_green_ratio : ℚ := 4 / 3

/-- The amount of yellow paint added to change forest green to verdant green -/
def yellow_paint_added : ℝ := 2.333333333333333

/-- The original amount of yellow paint in the forest green mixture -/
def original_yellow_paint : ℝ := 3

theorem forest_to_verdant_green_conversion :
  let b := forest_green_ratio * original_yellow_paint
  (original_yellow_paint + yellow_paint_added) / b = verdant_green_ratio :=
by sorry

end NUMINAMATH_CALUDE_forest_to_verdant_green_conversion_l3556_355612


namespace NUMINAMATH_CALUDE_dream_cost_in_illusions_l3556_355698

/-- Represents the price of an item in the dream market -/
structure DreamPrice where
  illusion : ℚ
  nap : ℚ
  nightmare : ℚ
  dream : ℚ

/-- The dream market pricing system satisfies the given conditions -/
def is_valid_pricing (p : DreamPrice) : Prop :=
  7 * p.illusion + 2 * p.nap + p.nightmare = 4 * p.dream ∧
  4 * p.illusion + 4 * p.nap + 2 * p.nightmare = 7 * p.dream

/-- The cost of one dream is equal to 10 illusions -/
theorem dream_cost_in_illusions (p : DreamPrice) : 
  is_valid_pricing p → p.dream = 10 * p.illusion := by
  sorry

end NUMINAMATH_CALUDE_dream_cost_in_illusions_l3556_355698


namespace NUMINAMATH_CALUDE_cricket_team_average_age_l3556_355659

theorem cricket_team_average_age 
  (n : ℕ) 
  (captain_age : ℕ) 
  (wicket_keeper_age_diff : ℕ) 
  (h1 : n = 11) 
  (h2 : captain_age = 28) 
  (h3 : wicket_keeper_age_diff = 3) : 
  ∃ (team_avg : ℚ), 
    team_avg = 25 ∧ 
    (n : ℚ) * team_avg = 
      (captain_age : ℚ) + 
      ((captain_age : ℚ) + wicket_keeper_age_diff) + 
      ((n - 2 : ℚ) * (team_avg - 1)) := by
  sorry

end NUMINAMATH_CALUDE_cricket_team_average_age_l3556_355659


namespace NUMINAMATH_CALUDE_prob_class1_drew_two_mc_correct_expected_rounds_correct_l3556_355649

-- Define the boxes
structure Box where
  multiple_choice : ℕ
  fill_in_blank : ℕ

-- Define the game
structure Game where
  box_a : Box
  box_b : Box
  class_6_first_win_prob : ℚ
  next_win_prob : ℚ

-- Define the problem
def chinese_culture_competition : Game :=
  { box_a := { multiple_choice := 5, fill_in_blank := 3 }
  , box_b := { multiple_choice := 4, fill_in_blank := 3 }
  , class_6_first_win_prob := 3/5
  , next_win_prob := 2/5
  }

-- Part 1: Probability calculation
def prob_class1_drew_two_mc (g : Game) : ℚ :=
  20/49

-- Part 2: Expected value calculation
def expected_rounds (g : Game) : ℚ :=
  537/125

-- Theorem statements
theorem prob_class1_drew_two_mc_correct (g : Game) :
  g = chinese_culture_competition →
  prob_class1_drew_two_mc g = 20/49 := by sorry

theorem expected_rounds_correct (g : Game) :
  g = chinese_culture_competition →
  expected_rounds g = 537/125 := by sorry

end NUMINAMATH_CALUDE_prob_class1_drew_two_mc_correct_expected_rounds_correct_l3556_355649


namespace NUMINAMATH_CALUDE_dans_initial_money_l3556_355699

/-- Dan's initial amount of money -/
def initial_money : ℕ := sorry

/-- Cost of the candy bar -/
def candy_cost : ℕ := 6

/-- Cost of the chocolate -/
def chocolate_cost : ℕ := 3

/-- Theorem stating that Dan's initial money is equal to the total spent -/
theorem dans_initial_money :
  initial_money = candy_cost + chocolate_cost ∧ candy_cost = chocolate_cost + 3 := by
  sorry

end NUMINAMATH_CALUDE_dans_initial_money_l3556_355699


namespace NUMINAMATH_CALUDE_banana_boxes_theorem_l3556_355642

/-- The number of bananas Marilyn has -/
def total_bananas : ℕ := 40

/-- The number of bananas each box must contain -/
def bananas_per_box : ℕ := 5

/-- The number of boxes needed to store all bananas -/
def num_boxes : ℕ := total_bananas / bananas_per_box

theorem banana_boxes_theorem : num_boxes = 8 := by
  sorry

end NUMINAMATH_CALUDE_banana_boxes_theorem_l3556_355642


namespace NUMINAMATH_CALUDE_ad_ratio_l3556_355685

/-- Represents the number of ads on each web page -/
structure WebPages :=
  (page1 : ℕ)
  (page2 : ℕ)
  (page3 : ℕ)
  (page4 : ℕ)

/-- Conditions of the problem -/
def adConditions (w : WebPages) : Prop :=
  w.page1 = 12 ∧
  w.page2 = 2 * w.page1 ∧
  w.page3 = w.page2 + 24 ∧
  2 * 68 = 3 * (w.page1 + w.page2 + w.page3 + w.page4)

/-- The theorem to be proved -/
theorem ad_ratio (w : WebPages) :
  adConditions w →
  (w.page4 : ℚ) / w.page2 = 3 / 4 := by
  sorry

end NUMINAMATH_CALUDE_ad_ratio_l3556_355685


namespace NUMINAMATH_CALUDE_arithmetic_calculations_l3556_355693

theorem arithmetic_calculations :
  (14 - 25 + 12 - 17 = -16) ∧
  ((1/2 + 5/6 - 7/12) / (-1/36) = -27) :=
by sorry

end NUMINAMATH_CALUDE_arithmetic_calculations_l3556_355693


namespace NUMINAMATH_CALUDE_ellipse_C_and_point_T_l3556_355647

/-- The ellipse C -/
def ellipse_C (x y : ℝ) (a b : ℝ) : Prop :=
  x^2 / a^2 + y^2 / b^2 = 1

/-- The circle M -/
def circle_M (x y : ℝ) : Prop :=
  x^2 + y^2 - 4*x - 2*y + 4 = 0

/-- The line l passing through (1,0) and intersecting C at A and B -/
def line_l (m : ℝ) (x y : ℝ) : Prop :=
  y = m * (x - 1)

/-- The angle OTA equals OTB -/
def angle_OTA_eq_OTB (t : ℝ) (xA yA xB yB : ℝ) : Prop :=
  (yA / (xA - t)) + (yB / (xB - t)) = 0

theorem ellipse_C_and_point_T :
  ∃ (a b : ℝ), a > b ∧ b > 0 ∧ b = 1 ∧
  (∃ (c : ℝ), c^2 = a^2 - b^2 ∧
    (∀ x y : ℝ, x + c*y - c = 0 → circle_M x y)) →
  (∀ x y : ℝ, ellipse_C x y a b ↔ x^2/4 + y^2 = 1) ∧
  (∃ t : ℝ, t = 4 ∧
    ∀ m xA yA xB yB : ℝ,
      line_l m xA yA ∧ line_l m xB yB ∧
      ellipse_C xA yA a b ∧ ellipse_C xB yB a b →
      angle_OTA_eq_OTB t xA yA xB yB) :=
by sorry

end NUMINAMATH_CALUDE_ellipse_C_and_point_T_l3556_355647


namespace NUMINAMATH_CALUDE_triangle_count_specific_l3556_355605

/-- The number of triangles formed by points on two sides of a triangle -/
def triangles_from_points (n m : ℕ) : ℕ :=
  Nat.choose (n + m + 1) 3 - Nat.choose (n + 1) 3 - Nat.choose (m + 1) 3

/-- Theorem: The number of triangles formed by 5 points on one side,
    6 points on another side, and 1 shared vertex is 165 -/
theorem triangle_count_specific : triangles_from_points 5 6 = 165 := by
  sorry

end NUMINAMATH_CALUDE_triangle_count_specific_l3556_355605


namespace NUMINAMATH_CALUDE_mathematics_competition_is_good_l3556_355643

theorem mathematics_competition_is_good :
  ∃ (x₁ y₁ x₂ y₂ : ℕ),
    1000 * x₁ + y₁ = 2 * x₁ * y₁ ∧
    1000 * x₂ + y₂ = 2 * x₂ * y₂ ∧
    1000 * x₁ + y₁ = 13520 ∧
    1000 * x₂ + y₂ = 63504 :=
by sorry

end NUMINAMATH_CALUDE_mathematics_competition_is_good_l3556_355643


namespace NUMINAMATH_CALUDE_rationalize_and_simplify_l3556_355641

theorem rationalize_and_simplify :
  (Real.sqrt 18) / (Real.sqrt 9 - Real.sqrt 3) = (3 * Real.sqrt 2 + Real.sqrt 6) / 2 := by
  sorry

end NUMINAMATH_CALUDE_rationalize_and_simplify_l3556_355641


namespace NUMINAMATH_CALUDE_union_of_sets_l3556_355622

theorem union_of_sets : 
  let A : Set ℕ := {2, 5, 6}
  let B : Set ℕ := {3, 5}
  A ∪ B = {2, 3, 5, 6} := by
sorry

end NUMINAMATH_CALUDE_union_of_sets_l3556_355622


namespace NUMINAMATH_CALUDE_irrationality_of_sqrt_7_l3556_355686

theorem irrationality_of_sqrt_7 :
  ¬ (∃ (p q : ℤ), q ≠ 0 ∧ Real.sqrt 7 = (p : ℚ) / q) ∧
  (∃ (p q : ℤ), q ≠ 0 ∧ (-2 : ℚ) / 9 = (p : ℚ) / q) ∧
  (∃ (p q : ℤ), q ≠ 0 ∧ (1 : ℚ) / 2 = (p : ℚ) / q) ∧
  (∃ (p q : ℤ), q ≠ 0 ∧ -4 = (p : ℚ) / q) :=
by sorry

end NUMINAMATH_CALUDE_irrationality_of_sqrt_7_l3556_355686


namespace NUMINAMATH_CALUDE_largest_number_l3556_355669

def to_decimal (digits : List Nat) (base : Nat) : Nat :=
  digits.foldl (fun acc d => acc * base + d) 0

def number_85_9 : Nat := to_decimal [8, 5] 9
def number_210_6 : Nat := to_decimal [2, 1, 0] 6
def number_1000_4 : Nat := to_decimal [1, 0, 0, 0] 4
def number_11111_2 : Nat := to_decimal [1, 1, 1, 1, 1] 2

theorem largest_number :
  number_210_6 > number_85_9 ∧
  number_210_6 > number_1000_4 ∧
  number_210_6 > number_11111_2 := by
  sorry

end NUMINAMATH_CALUDE_largest_number_l3556_355669


namespace NUMINAMATH_CALUDE_mode_most_relevant_for_market_share_l3556_355672

/-- Represents a clothing model with its sales data -/
structure ClothingModel where
  id : ℕ
  sales : ℕ

/-- Represents a collection of clothing models -/
def ClothingModelData := List ClothingModel

/-- Calculates the mode of a list of natural numbers -/
def mode (l : List ℕ) : Option ℕ :=
  sorry

/-- Calculates the mean of a list of natural numbers -/
def mean (l : List ℕ) : ℚ :=
  sorry

/-- Calculates the median of a list of natural numbers -/
def median (l : List ℕ) : ℚ :=
  sorry

/-- Determines the most relevant statistical measure for market share survey -/
def mostRelevantMeasure (data : ClothingModelData) : String :=
  sorry

theorem mode_most_relevant_for_market_share (data : ClothingModelData) :
  mostRelevantMeasure data = "mode" :=
sorry

end NUMINAMATH_CALUDE_mode_most_relevant_for_market_share_l3556_355672


namespace NUMINAMATH_CALUDE_smallest_cube_factor_l3556_355646

theorem smallest_cube_factor (z : ℕ) (hz : z.Prime ∧ z > 7) :
  let y := 19408850
  (∀ k : ℕ, k > 0 ∧ k < y → ¬∃ n : ℕ, (31360 * z) * k = n^3) ∧
  ∃ n : ℕ, (31360 * z) * y = n^3 :=
sorry

end NUMINAMATH_CALUDE_smallest_cube_factor_l3556_355646


namespace NUMINAMATH_CALUDE_remaining_children_meals_l3556_355628

theorem remaining_children_meals (total_children_meals : ℕ) 
  (adults_consumed : ℕ) (child_adult_ratio : ℚ) :
  total_children_meals = 90 →
  adults_consumed = 42 →
  child_adult_ratio = 90 / 70 →
  total_children_meals - (↑adults_consumed * child_adult_ratio).floor = 36 :=
by
  sorry

end NUMINAMATH_CALUDE_remaining_children_meals_l3556_355628


namespace NUMINAMATH_CALUDE_center_shade_ratio_l3556_355690

/-- Represents a square grid -/
structure SquareGrid (n : ℕ) where
  size : ℕ
  total_area : ℝ
  cell_area : ℝ
  h_size : size = n
  h_cell_area : cell_area = total_area / (n^2 : ℝ)

/-- Represents a shaded region in the center of the grid -/
structure CenterShade (grid : SquareGrid 5) where
  area : ℝ
  h_area : area = 4 * (grid.cell_area / 2)

/-- The theorem stating the ratio of the shaded area to the total area -/
theorem center_shade_ratio (grid : SquareGrid 5) (shade : CenterShade grid) :
  shade.area / grid.total_area = 2 / 25 := by
  sorry

end NUMINAMATH_CALUDE_center_shade_ratio_l3556_355690


namespace NUMINAMATH_CALUDE_exists_expression_for_100_l3556_355607

/-- An arithmetic expression using only the number 3, parentheses, and basic arithmetic operations. -/
inductive Expr
  | three : Expr
  | add : Expr → Expr → Expr
  | sub : Expr → Expr → Expr
  | mul : Expr → Expr → Expr
  | div : Expr → Expr → Expr

/-- Evaluate an arithmetic expression. -/
def eval : Expr → ℚ
  | Expr.three => 3
  | Expr.add e1 e2 => eval e1 + eval e2
  | Expr.sub e1 e2 => eval e1 - eval e2
  | Expr.mul e1 e2 => eval e1 * eval e2
  | Expr.div e1 e2 => eval e1 / eval e2

/-- Count the number of threes in an expression. -/
def count_threes : Expr → ℕ
  | Expr.three => 1
  | Expr.add e1 e2 => count_threes e1 + count_threes e2
  | Expr.sub e1 e2 => count_threes e1 + count_threes e2
  | Expr.mul e1 e2 => count_threes e1 + count_threes e2
  | Expr.div e1 e2 => count_threes e1 + count_threes e2

/-- Theorem: There exists an arithmetic expression using fewer than ten threes that evaluates to 100. -/
theorem exists_expression_for_100 : ∃ e : Expr, eval e = 100 ∧ count_threes e < 10 := by
  sorry


end NUMINAMATH_CALUDE_exists_expression_for_100_l3556_355607


namespace NUMINAMATH_CALUDE_simplify_sqrt_expression_l3556_355626

theorem simplify_sqrt_expression : 
  Real.sqrt (28 - 12 * Real.sqrt 2) = 6 - Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_simplify_sqrt_expression_l3556_355626


namespace NUMINAMATH_CALUDE_john_caffeine_consumption_l3556_355671

/-- The amount of caffeine John consumed from two energy drinks and a caffeine pill -/
theorem john_caffeine_consumption (first_drink_oz : ℝ) (first_drink_caffeine : ℝ) 
  (second_drink_oz : ℝ) (second_drink_caffeine_multiplier : ℝ) :
  first_drink_oz = 12 ∧ 
  first_drink_caffeine = 250 ∧ 
  second_drink_oz = 2 ∧ 
  second_drink_caffeine_multiplier = 3 →
  (let first_drink_caffeine_per_oz := first_drink_caffeine / first_drink_oz
   let second_drink_caffeine_per_oz := first_drink_caffeine_per_oz * second_drink_caffeine_multiplier
   let second_drink_caffeine := second_drink_caffeine_per_oz * second_drink_oz
   let total_drinks_caffeine := first_drink_caffeine + second_drink_caffeine
   let pill_caffeine := total_drinks_caffeine
   let total_caffeine := total_drinks_caffeine + pill_caffeine
   total_caffeine = 750) :=
by sorry

end NUMINAMATH_CALUDE_john_caffeine_consumption_l3556_355671


namespace NUMINAMATH_CALUDE_grocery_store_costs_l3556_355692

theorem grocery_store_costs (total_cost : ℝ) (salary_fraction : ℝ) (delivery_fraction : ℝ) 
  (h1 : total_cost = 4000)
  (h2 : salary_fraction = 2/5)
  (h3 : delivery_fraction = 1/4) : 
  total_cost * (1 - salary_fraction) * (1 - delivery_fraction) = 1800 := by
  sorry

end NUMINAMATH_CALUDE_grocery_store_costs_l3556_355692


namespace NUMINAMATH_CALUDE_magician_card_decks_l3556_355652

theorem magician_card_decks (price : ℕ) (decks_left : ℕ) (earnings : ℕ) : 
  price = 7 → decks_left = 8 → earnings = 56 → 
  ∃ (initial_decks : ℕ), initial_decks = decks_left + earnings / price :=
by sorry

end NUMINAMATH_CALUDE_magician_card_decks_l3556_355652


namespace NUMINAMATH_CALUDE_pencil_count_difference_l3556_355621

theorem pencil_count_difference (D J M E : ℕ) : 
  D = J + 15 → 
  J = 2 * M → 
  E = (J - M) / 2 → 
  J = 20 → 
  D - (M + E) = 20 := by
sorry

end NUMINAMATH_CALUDE_pencil_count_difference_l3556_355621


namespace NUMINAMATH_CALUDE_sum_of_digits_next_l3556_355664

/-- S(n) is the sum of the digits of a positive integer n -/
def S (n : ℕ+) : ℕ :=
  sorry

theorem sum_of_digits_next (n : ℕ+) (h : S n = 1274) : S (n + 1) = 1239 :=
  sorry

end NUMINAMATH_CALUDE_sum_of_digits_next_l3556_355664


namespace NUMINAMATH_CALUDE_right_triangle_segment_ratio_l3556_355625

theorem right_triangle_segment_ratio (a b c r s : ℝ) :
  a > 0 ∧ b > 0 ∧ c > 0 ∧ r > 0 ∧ s > 0 →  -- Positive lengths
  c^2 = a^2 + b^2 →  -- Pythagorean theorem
  r * s = a^2 →  -- Geometric mean theorem for r
  r * s = b^2 →  -- Geometric mean theorem for s
  r + s = c →  -- r and s form the hypotenuse
  a / b = 1 / 4 →  -- Given ratio
  r / s = 1 / 16 :=
by sorry

end NUMINAMATH_CALUDE_right_triangle_segment_ratio_l3556_355625


namespace NUMINAMATH_CALUDE_fraction_evaluation_l3556_355683

theorem fraction_evaluation : 
  (20-19+18-17+16-15+14-13+12-11+10-9+8-7+6-5+4-3+2-1) / 
  (2-3+4-5+6-7+8-9+10-11+12-13+14-15+16-17+18-19+20) = 10/11 := by
  sorry

end NUMINAMATH_CALUDE_fraction_evaluation_l3556_355683


namespace NUMINAMATH_CALUDE_person_is_knight_l3556_355635

-- Define the type of person
inductive Person : Type
  | Knight : Person
  | Liar : Person

-- Define the statements
def lovesLinda (p : Person) : Prop := 
  match p with
  | Person.Knight => true
  | Person.Liar => false

def lovesKatie (p : Person) : Prop :=
  match p with
  | Person.Knight => true
  | Person.Liar => false

-- Define the theorem
theorem person_is_knight : 
  ∀ (p : Person), 
    (lovesLinda p = true ∨ lovesLinda p = false) → 
    (lovesLinda p → lovesKatie p) → 
    p = Person.Knight :=
by
  sorry


end NUMINAMATH_CALUDE_person_is_knight_l3556_355635


namespace NUMINAMATH_CALUDE_tv_sales_increase_l3556_355617

theorem tv_sales_increase (original_price : ℝ) (original_quantity : ℝ) 
  (h_positive_price : original_price > 0) (h_positive_quantity : original_quantity > 0) :
  let new_price := 0.9 * original_price
  let new_total_value := 1.665 * (original_price * original_quantity)
  ∃ (new_quantity : ℝ), 
    new_price * new_quantity = new_total_value ∧ 
    (new_quantity - original_quantity) / original_quantity = 0.85 :=
by sorry

end NUMINAMATH_CALUDE_tv_sales_increase_l3556_355617


namespace NUMINAMATH_CALUDE_double_angle_sine_fifteen_degrees_l3556_355636

theorem double_angle_sine_fifteen_degrees :
  2 * Real.sin (15 * π / 180) * Real.cos (15 * π / 180) = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_double_angle_sine_fifteen_degrees_l3556_355636


namespace NUMINAMATH_CALUDE_intersection_A_B_l3556_355648

-- Define set A
def A : Set ℝ := {x | |x - 1| < 2}

-- Define set B
def B : Set ℝ := {x | x^2 + 3*x - 4 < 0}

-- Theorem statement
theorem intersection_A_B : A ∩ B = {x : ℝ | -1 < x ∧ x < 1} := by sorry

end NUMINAMATH_CALUDE_intersection_A_B_l3556_355648


namespace NUMINAMATH_CALUDE_sqrt_inequality_l3556_355657

theorem sqrt_inequality (a : ℝ) (h : a ≥ 3) : 
  Real.sqrt a - Real.sqrt (a - 1) < Real.sqrt (a - 2) - Real.sqrt (a - 3) := by
  sorry

end NUMINAMATH_CALUDE_sqrt_inequality_l3556_355657


namespace NUMINAMATH_CALUDE_coupon_value_l3556_355631

theorem coupon_value (total_spent peaches_after_coupon cherries : ℚ) : 
  total_spent = 23.86 →
  peaches_after_coupon = 12.32 →
  cherries = 11.54 →
  total_spent = peaches_after_coupon + cherries →
  0 = total_spent - (peaches_after_coupon + cherries) := by
sorry

end NUMINAMATH_CALUDE_coupon_value_l3556_355631


namespace NUMINAMATH_CALUDE_newer_car_travels_195_miles_l3556_355687

/-- The distance traveled by the older car -/
def older_car_distance : ℝ := 150

/-- The percentage increase in distance for the newer car -/
def newer_car_percentage : ℝ := 0.30

/-- The distance traveled by the newer car -/
def newer_car_distance : ℝ := older_car_distance * (1 + newer_car_percentage)

/-- Theorem stating that the newer car travels 195 miles -/
theorem newer_car_travels_195_miles :
  newer_car_distance = 195 := by sorry

end NUMINAMATH_CALUDE_newer_car_travels_195_miles_l3556_355687


namespace NUMINAMATH_CALUDE_factorial_plus_one_eq_power_l3556_355695

theorem factorial_plus_one_eq_power (n p : ℕ) : 
  (Nat.factorial (p - 1) + 1 = p ^ n) ↔ 
  ((n = 1 ∧ p = 2) ∨ (n = 1 ∧ p = 3) ∨ (n = 2 ∧ p = 5)) :=
sorry

end NUMINAMATH_CALUDE_factorial_plus_one_eq_power_l3556_355695


namespace NUMINAMATH_CALUDE_machine_present_value_l3556_355616

/-- The present value of a machine given its depreciation rate, selling price after two years, and profit made. -/
theorem machine_present_value
  (depreciation_rate : ℝ)
  (selling_price : ℝ)
  (profit : ℝ)
  (h1 : depreciation_rate = 0.2)
  (h2 : selling_price = 118000.00000000001)
  (h3 : profit = 22000) :
  ∃ (present_value : ℝ),
    present_value = 150000.00000000002 ∧
    present_value * (1 - depreciation_rate)^2 = selling_price - profit :=
by sorry

end NUMINAMATH_CALUDE_machine_present_value_l3556_355616


namespace NUMINAMATH_CALUDE_largest_difference_l3556_355604

def U : ℕ := 3 * 2005^2006
def V : ℕ := 2005^2006
def W : ℕ := 2004 * 2005^2005
def X : ℕ := 3 * 2005^2005
def Y : ℕ := 2005^2005
def Z : ℕ := 2005^2004

theorem largest_difference : 
  (U - V > V - W) ∧ (U - V > W - X) ∧ (U - V > X - Y) ∧ (U - V > Y - Z) :=
by sorry

end NUMINAMATH_CALUDE_largest_difference_l3556_355604


namespace NUMINAMATH_CALUDE_olivia_wallet_proof_l3556_355630

def initial_wallet_amount (amount_spent : ℕ) (amount_left : ℕ) : ℕ :=
  amount_spent + amount_left

theorem olivia_wallet_proof (amount_spent : ℕ) (amount_left : ℕ) 
  (h1 : amount_spent = 38) (h2 : amount_left = 90) :
  initial_wallet_amount amount_spent amount_left = 128 := by
  sorry

end NUMINAMATH_CALUDE_olivia_wallet_proof_l3556_355630


namespace NUMINAMATH_CALUDE_tangent_line_at_zero_f_positive_when_a_eq_two_max_value_on_interval_l3556_355609

/-- The function f(x) = e^x - ax --/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.exp x - a * x

/-- Theorem for the tangent line equation when a = 2 --/
theorem tangent_line_at_zero (x y : ℝ) :
  (f 2) 0 = 1 →
  (∀ h, deriv (f 2) h = Real.exp h - 2) →
  x + y - 1 = 0 ↔ y - 1 = -(x - 0) :=
sorry

/-- Theorem for f(x) > 0 when a = 2 --/
theorem f_positive_when_a_eq_two :
  ∀ x, f 2 x > 0 :=
sorry

/-- Theorem for the maximum value of f(x) when a > 1 --/
theorem max_value_on_interval (a : ℝ) :
  a > 1 →
  ∃ x ∈ Set.Icc 0 a, ∀ y ∈ Set.Icc 0 a, f a x ≥ f a y ∧ f a x = Real.exp a - a^2 :=
sorry

end NUMINAMATH_CALUDE_tangent_line_at_zero_f_positive_when_a_eq_two_max_value_on_interval_l3556_355609


namespace NUMINAMATH_CALUDE_problem_statement_l3556_355620

theorem problem_statement (θ : Real) 
  (h1 : θ ∈ Set.Ioo 0 (π/4)) 
  (h2 : Real.sin θ - Real.cos θ = -Real.sqrt 14 / 4) : 
  (2 * (Real.cos θ)^2 - 1) / Real.cos (π/4 + θ) = 3/2 := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l3556_355620


namespace NUMINAMATH_CALUDE_investment_rate_proof_l3556_355613

/-- Proves that the remaining investment rate is 7% given the specified conditions --/
theorem investment_rate_proof (total_investment : ℝ) (first_investment : ℝ) (second_investment : ℝ)
  (first_rate : ℝ) (second_rate : ℝ) (desired_income : ℝ) :
  total_investment = 12000 →
  first_investment = 5000 →
  second_investment = 4000 →
  first_rate = 0.05 →
  second_rate = 0.035 →
  desired_income = 600 →
  let remaining_investment := total_investment - first_investment - second_investment
  let first_income := first_investment * first_rate
  let second_income := second_investment * second_rate
  let remaining_income := desired_income - first_income - second_income
  remaining_income / remaining_investment = 0.07 :=
by sorry

end NUMINAMATH_CALUDE_investment_rate_proof_l3556_355613


namespace NUMINAMATH_CALUDE_isosceles_triangle_vertex_angle_l3556_355632

-- Define an isosceles triangle
structure IsoscelesTriangle where
  angles : Fin 3 → ℝ
  sum_180 : angles 0 + angles 1 + angles 2 = 180
  isosceles : (angles 0 = angles 1) ∨ (angles 1 = angles 2) ∨ (angles 0 = angles 2)

-- Theorem statement
theorem isosceles_triangle_vertex_angle 
  (t : IsoscelesTriangle) 
  (h : ∃ i, t.angles i = 80) :
  (t.angles 0 = 80 ∨ t.angles 0 = 20) ∨
  (t.angles 1 = 80 ∨ t.angles 1 = 20) ∨
  (t.angles 2 = 80 ∨ t.angles 2 = 20) :=
by sorry

end NUMINAMATH_CALUDE_isosceles_triangle_vertex_angle_l3556_355632


namespace NUMINAMATH_CALUDE_M_is_power_of_three_l3556_355673

/-- Arithmetic sequence with a_n = n -/
def arithmetic_seq (n : ℕ) : ℕ := n

/-- Sequence of t_n values -/
def t_seq : ℕ → ℕ
  | 0 => 0
  | n+1 => (3^(n+1) - 1) / 2

/-- M_n is the sum of terms from (t_{n-1}+1)th to t_n th term -/
def M (n : ℕ) : ℕ :=
  let a := t_seq (n-1)
  let b := t_seq n
  (b * (b + 1) - a * (a + 1)) / 2

/-- Main theorem: M_n = 3^(2n-2) for all n ∈ ℕ -/
theorem M_is_power_of_three (n : ℕ) : M n = 3^(2*n - 2) := by
  sorry


end NUMINAMATH_CALUDE_M_is_power_of_three_l3556_355673


namespace NUMINAMATH_CALUDE_right_triangle_side_values_l3556_355618

theorem right_triangle_side_values (a b x : ℝ) : 
  a = 6 → b = 8 → (x^2 = a^2 + b^2 ∨ b^2 = a^2 + x^2) → (x = 10 ∨ x = 2 * Real.sqrt 7) := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_side_values_l3556_355618


namespace NUMINAMATH_CALUDE_pirate_treasure_distribution_l3556_355645

theorem pirate_treasure_distribution (x : ℕ) : 
  (x * (x + 1)) / 2 = 4 * x → x + 4 * x = 35 := by
  sorry

end NUMINAMATH_CALUDE_pirate_treasure_distribution_l3556_355645


namespace NUMINAMATH_CALUDE_complement_of_A_in_U_l3556_355656

def U : Set ℕ := {1,2,3,4,5,6,7}
def A : Set ℕ := {1,3,5,7}

theorem complement_of_A_in_U : 
  (U \ A) = {2,4,6} := by sorry

end NUMINAMATH_CALUDE_complement_of_A_in_U_l3556_355656


namespace NUMINAMATH_CALUDE_min_value_expression_l3556_355600

open Real

theorem min_value_expression (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  ∃ (min_val : ℝ), min_val = Real.sqrt 10 / 5 ∧
  ∀ (x y : ℝ) (hx : x > 0) (hy : y > 0),
    (abs (x + 3*y - y*(x + 9*y)) + abs (3*y - x + 3*y*(x - y))) / Real.sqrt (x^2 + 9*y^2) ≥ min_val :=
by sorry

end NUMINAMATH_CALUDE_min_value_expression_l3556_355600


namespace NUMINAMATH_CALUDE_correct_operation_l3556_355670

theorem correct_operation (a : ℝ) : (-2 * a^2)^3 = -8 * a^6 := by
  sorry

end NUMINAMATH_CALUDE_correct_operation_l3556_355670


namespace NUMINAMATH_CALUDE_fourth_root_256_times_cube_root_64_times_sqrt_16_l3556_355691

theorem fourth_root_256_times_cube_root_64_times_sqrt_16 :
  (256 : ℝ) ^ (1/4) * (64 : ℝ) ^ (1/3) * (16 : ℝ) ^ (1/2) = 64 := by
  sorry

end NUMINAMATH_CALUDE_fourth_root_256_times_cube_root_64_times_sqrt_16_l3556_355691


namespace NUMINAMATH_CALUDE_line_equation_from_triangle_l3556_355634

/-- Given a line passing through (-a, 0), (b, 0), and (0, h), where the area of the triangle
    formed in the second quadrant is T, prove that the equation of this line is
    2Tx - (b+a)^2y + 2T(b+a) = 0 -/
theorem line_equation_from_triangle (a b h T : ℝ) :
  (∃ (line : ℝ → ℝ → Prop),
    line (-a) 0 ∧
    line b 0 ∧
    line 0 h ∧
    (1/2 : ℝ) * (b + a) * h = T) →
  (∃ (line : ℝ → ℝ → Prop),
    ∀ x y, line x y ↔ 2 * T * x - (b + a)^2 * y + 2 * T * (b + a) = 0) :=
by sorry

end NUMINAMATH_CALUDE_line_equation_from_triangle_l3556_355634


namespace NUMINAMATH_CALUDE_hemisphere_with_cylinder_surface_area_l3556_355666

/-- The total surface area of a hemisphere with a cylindrical protrusion -/
theorem hemisphere_with_cylinder_surface_area (r : ℝ) (h : r > 0) :
  let base_area := π * r^2
  let hemisphere_surface := 2 * π * r^2
  let cylinder_surface := 2 * π * r^2
  base_area + hemisphere_surface + cylinder_surface = 5 * π * r^2 := by
sorry

end NUMINAMATH_CALUDE_hemisphere_with_cylinder_surface_area_l3556_355666


namespace NUMINAMATH_CALUDE_hans_reservation_deposit_l3556_355655

/-- Calculates the total deposit for a restaurant reservation with given guest counts and fees -/
def calculate_deposit (num_kids num_adults num_seniors num_students num_employees : ℕ)
  (flat_fee kid_fee adult_fee senior_fee student_fee employee_fee : ℚ)
  (service_charge_rate : ℚ) : ℚ :=
  let base_deposit := flat_fee + 
    num_kids * kid_fee + 
    num_adults * adult_fee + 
    num_seniors * senior_fee + 
    num_students * student_fee + 
    num_employees * employee_fee
  let service_charge := base_deposit * service_charge_rate
  base_deposit + service_charge

/-- The total deposit for Hans' reservation is $128.63 -/
theorem hans_reservation_deposit :
  calculate_deposit 2 8 5 3 2 30 3 6 4 (9/2) (5/2) (1/20) = 12863/100 := by
  sorry

end NUMINAMATH_CALUDE_hans_reservation_deposit_l3556_355655


namespace NUMINAMATH_CALUDE_math_test_questions_math_test_questions_proof_l3556_355629

theorem math_test_questions : ℕ → Prop :=
  fun total_questions =>
    let word_problems : ℕ := 17
    let addition_subtraction_problems : ℕ := 28
    let steve_answered : ℕ := 38
    let difference : ℕ := 7
    
    (total_questions - steve_answered = difference) ∧
    (word_problems + addition_subtraction_problems ≤ total_questions) ∧
    (steve_answered < total_questions) →
    total_questions = 45

-- The proof is omitted
theorem math_test_questions_proof : math_test_questions 45 := by sorry

end NUMINAMATH_CALUDE_math_test_questions_math_test_questions_proof_l3556_355629


namespace NUMINAMATH_CALUDE_quadratic_factor_l3556_355623

theorem quadratic_factor (k : ℝ) : 
  (∃ b : ℝ, (X + 5) * (X + b) = X^2 - k*X - 15) → 
  (X - 3) * (X + 5) = X^2 - k*X - 15 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_factor_l3556_355623


namespace NUMINAMATH_CALUDE_taxi_theorem_l3556_355668

def taxi_distances : List ℤ := [5, 2, -4, -3, 6]
def fuel_rate : ℚ := 0.3
def base_fare : ℚ := 8
def base_distance : ℚ := 3
def extra_fare_rate : ℚ := 1.6

def final_position (distances : List ℤ) : ℤ :=
  distances.sum

def total_distance (distances : List ℤ) : ℕ :=
  distances.map Int.natAbs |>.sum

def fuel_consumed (distances : List ℤ) (rate : ℚ) : ℚ :=
  rate * (total_distance distances : ℚ)

def fare_for_distance (d : ℚ) : ℚ :=
  if d ≤ base_distance then base_fare
  else base_fare + extra_fare_rate * (d - base_distance)

def total_fare (distances : List ℤ) : ℚ :=
  distances.map (fun d => fare_for_distance (Int.natAbs d : ℚ)) |>.sum

theorem taxi_theorem :
  final_position taxi_distances = 6 ∧
  fuel_consumed taxi_distances fuel_rate = 6 ∧
  total_fare taxi_distances = 49.6 := by
  sorry

end NUMINAMATH_CALUDE_taxi_theorem_l3556_355668


namespace NUMINAMATH_CALUDE_fraction_decrease_l3556_355639

theorem fraction_decrease (m n : ℝ) (h : m ≠ 0 ∧ n ≠ 0) : 
  (3*m + 3*n) / ((3*m) * (3*n)) = (1/3) * ((m + n) / (m * n)) := by
  sorry

end NUMINAMATH_CALUDE_fraction_decrease_l3556_355639


namespace NUMINAMATH_CALUDE_compound_composition_l3556_355675

/-- The atomic weight of Aluminium in g/mol -/
def atomic_weight_Al : ℝ := 26.98

/-- The atomic weight of Sulphur in g/mol -/
def atomic_weight_S : ℝ := 32.06

/-- The number of Sulphur atoms in the compound -/
def num_S_atoms : ℕ := 3

/-- The molecular weight of the compound in g/mol -/
def molecular_weight : ℝ := 150

/-- The number of Aluminium atoms in the compound -/
def num_Al_atoms : ℕ := 2

theorem compound_composition :
  num_Al_atoms * atomic_weight_Al + num_S_atoms * atomic_weight_S = molecular_weight := by
  sorry

end NUMINAMATH_CALUDE_compound_composition_l3556_355675


namespace NUMINAMATH_CALUDE_heptagonal_prism_faces_and_vertices_l3556_355606

/-- A heptagonal prism is a three-dimensional shape with two heptagonal bases and rectangular lateral faces. -/
structure HeptagonalPrism where
  baseFaces : Nat
  lateralFaces : Nat
  baseVertices : Nat

/-- Properties of a heptagonal prism -/
def heptagonalPrismProperties : HeptagonalPrism where
  baseFaces := 2
  lateralFaces := 7
  baseVertices := 7

/-- Theorem: A heptagonal prism has 9 faces and 14 vertices -/
theorem heptagonal_prism_faces_and_vertices :
  let h := heptagonalPrismProperties
  (h.baseFaces + h.lateralFaces = 9) ∧ (h.baseVertices * h.baseFaces = 14) := by
  sorry

end NUMINAMATH_CALUDE_heptagonal_prism_faces_and_vertices_l3556_355606


namespace NUMINAMATH_CALUDE_theta_value_l3556_355653

theorem theta_value : ∃! (Θ : ℕ), Θ ∈ Finset.range 10 ∧ (312 : ℚ) / Θ = 40 + 2 * Θ := by
  sorry

end NUMINAMATH_CALUDE_theta_value_l3556_355653


namespace NUMINAMATH_CALUDE_culture_and_messengers_l3556_355665

-- Define the types
structure Performance :=
  (troupe : String)
  (location : String)
  (impression : String)

-- Define the conditions
def legend_show : Performance :=
  { troupe := "Chinese Acrobatic Troupe",
    location := "United States",
    impression := "favorable" }

-- Define the properties we want to prove
def is_national_and_global (p : Performance) : Prop :=
  p.troupe ≠ p.location ∧ p.impression = "favorable"

def are_cultural_messengers (p : Performance) : Prop :=
  p.troupe = "Chinese Acrobatic Troupe" ∧ p.impression = "favorable"

-- The theorem to prove
theorem culture_and_messengers :
  is_national_and_global legend_show ∧ are_cultural_messengers legend_show :=
by sorry

end NUMINAMATH_CALUDE_culture_and_messengers_l3556_355665


namespace NUMINAMATH_CALUDE_tangent_line_sum_l3556_355667

/-- Given a function f: ℝ → ℝ with a tangent line y = 1/2 * x + 2 at x = 1,
    prove that f(1) + f'(1) = 3 -/
theorem tangent_line_sum (f : ℝ → ℝ) (hf : Differentiable ℝ f) 
    (h_tangent : ∀ x, f 1 + (deriv f 1) * (x - 1) = 1/2 * x + 2) : 
    f 1 + deriv f 1 = 3 := by
  sorry

end NUMINAMATH_CALUDE_tangent_line_sum_l3556_355667


namespace NUMINAMATH_CALUDE_min_value_on_circle_l3556_355661

theorem min_value_on_circle (x y : ℝ) (h : (x - 3)^2 + y^2 = 9) :
  ∃ (m : ℝ), (∀ (a b : ℝ), (a - 3)^2 + b^2 = 9 → -2*b - 3*a ≥ m) ∧
             (∃ (c d : ℝ), (c - 3)^2 + d^2 = 9 ∧ -2*d - 3*c = m) ∧
             m = -3 * Real.sqrt 13 - 9 :=
sorry

end NUMINAMATH_CALUDE_min_value_on_circle_l3556_355661


namespace NUMINAMATH_CALUDE_initial_pigs_l3556_355640

theorem initial_pigs (initial : ℕ) : initial + 22 = 86 → initial = 64 := by
  sorry

end NUMINAMATH_CALUDE_initial_pigs_l3556_355640


namespace NUMINAMATH_CALUDE_factors_of_96_with_square_sum_208_l3556_355614

theorem factors_of_96_with_square_sum_208 :
  ∀ a b : ℕ+,
    a * b = 96 ∧ 
    a^2 + b^2 = 208 →
    (a = 8 ∧ b = 12) ∨ (a = 12 ∧ b = 8) := by
  sorry

end NUMINAMATH_CALUDE_factors_of_96_with_square_sum_208_l3556_355614


namespace NUMINAMATH_CALUDE_min_value_3a_2b_min_value_3a_2b_achieved_l3556_355644

theorem min_value_3a_2b (a b : ℝ) (h1 : a > 0) (h2 : b > 0) 
  (h3 : (a + b)⁻¹ + (a - b)⁻¹ = 1) : 
  ∀ x y : ℝ, x > 0 → y > 0 → (x + y)⁻¹ + (x - y)⁻¹ = 1 → 3*x + 2*y ≥ 3 + Real.sqrt 5 :=
by sorry

theorem min_value_3a_2b_achieved (a b : ℝ) (h1 : a > 0) (h2 : b > 0) 
  (h3 : (a + b)⁻¹ + (a - b)⁻¹ = 1) : 
  ∃ x y : ℝ, x > 0 ∧ y > 0 ∧ (x + y)⁻¹ + (x - y)⁻¹ = 1 ∧ 3*x + 2*y = 3 + Real.sqrt 5 :=
by sorry

end NUMINAMATH_CALUDE_min_value_3a_2b_min_value_3a_2b_achieved_l3556_355644


namespace NUMINAMATH_CALUDE_vector_sum_coords_l3556_355680

def a : ℝ × ℝ := (-2, 3)
def b : ℝ × ℝ := (1, -2)

theorem vector_sum_coords : 
  (2 : ℝ) • a + b = (-3, 4) := by sorry

end NUMINAMATH_CALUDE_vector_sum_coords_l3556_355680


namespace NUMINAMATH_CALUDE_sum_of_integers_problem_l3556_355633

theorem sum_of_integers_problem : ∃ (a b : ℕ), 
  (a > 0) ∧ (b > 0) ∧
  (a * b + a + b - (a - b) = 120) ∧
  (Nat.gcd a b = 1) ∧
  (a < 25) ∧ (b < 25) ∧
  (a + b = 19) := by
  sorry

end NUMINAMATH_CALUDE_sum_of_integers_problem_l3556_355633


namespace NUMINAMATH_CALUDE_hyperbola_focus_l3556_355602

/-- The hyperbola equation -2x^2 + 3y^2 + 8x - 18y - 8 = 0 -/
def hyperbola_equation (x y : ℝ) : Prop :=
  -2 * x^2 + 3 * y^2 + 8 * x - 18 * y - 8 = 0

/-- A point (x, y) is a focus of the hyperbola if it satisfies the focus condition -/
def is_focus (x y : ℝ) : Prop :=
  ∃ (a b : ℝ), a > 0 ∧ b > 0 ∧
  ∀ (p q : ℝ), hyperbola_equation p q →
  (p - x)^2 + (q - y)^2 = ((p - 2)^2 / (2 * b^2) - (q - 3)^2 / (2 * a^2) + 1)^2 * (a^2 + b^2)

theorem hyperbola_focus :
  is_focus 2 7.5 :=
sorry

end NUMINAMATH_CALUDE_hyperbola_focus_l3556_355602


namespace NUMINAMATH_CALUDE_tangent_condition_intersection_condition_l3556_355660

-- Define the ellipse
def ellipse (x y : ℝ) : Prop := x^2 + 4*y^2 = 4

-- Define the line
def line (x y m : ℝ) : Prop := y = x + m

-- Tangent condition
theorem tangent_condition (m : ℝ) : 
  (∃! p : ℝ × ℝ, ellipse p.1 p.2 ∧ line p.1 p.2 m) ↔ m^2 = 5 :=
sorry

-- Intersection condition
theorem intersection_condition (m : ℝ) :
  (∃ p q : ℝ × ℝ, p ≠ q ∧ 
   ellipse p.1 p.2 ∧ ellipse q.1 q.2 ∧ 
   line p.1 p.2 m ∧ line q.1 q.2 m ∧
   (p.1 - q.1)^2 + (p.2 - q.2)^2 = 4) ↔ 16 * m^2 = 30 :=
sorry

end NUMINAMATH_CALUDE_tangent_condition_intersection_condition_l3556_355660


namespace NUMINAMATH_CALUDE_height_on_longest_side_l3556_355637

/-- Given a triangle with side lengths 6, 8, and 10, prove that the height on the longest side is 4.8 -/
theorem height_on_longest_side (a b c h : ℝ) : 
  a = 6 → b = 8 → c = 10 → 
  a^2 + b^2 = c^2 → 
  (1/2) * c * h = (1/2) * a * b → 
  h = 4.8 := by
  sorry

end NUMINAMATH_CALUDE_height_on_longest_side_l3556_355637


namespace NUMINAMATH_CALUDE_computer_contract_probability_l3556_355601

theorem computer_contract_probability 
  (p_hardware : ℝ) 
  (p_not_software : ℝ) 
  (p_at_least_one : ℝ) 
  (h1 : p_hardware = 4/5)
  (h2 : p_not_software = 3/5)
  (h3 : p_at_least_one = 9/10) :
  p_hardware + (1 - p_not_software) - p_at_least_one = 7/10 := by
sorry

end NUMINAMATH_CALUDE_computer_contract_probability_l3556_355601


namespace NUMINAMATH_CALUDE_negative_integer_solution_to_inequality_l3556_355696

theorem negative_integer_solution_to_inequality :
  ∀ x : ℤ, (x < 0 ∧ -2 * x < 4) ↔ x = -1 :=
sorry

end NUMINAMATH_CALUDE_negative_integer_solution_to_inequality_l3556_355696


namespace NUMINAMATH_CALUDE_geometric_concept_word_counts_l3556_355650

/-- A type representing geometric concepts -/
def GeometricConcept : Type := String

/-- A function that counts the number of words in a string -/
def wordCount (s : String) : Nat :=
  s.split (· == ' ') |>.length

/-- Theorem stating that there exist geometric concepts expressible in 1, 2, 3, and 4 words -/
theorem geometric_concept_word_counts :
  ∃ (a b c d : GeometricConcept),
    wordCount a = 1 ∧
    wordCount b = 2 ∧
    wordCount c = 3 ∧
    wordCount d = 4 :=
by sorry


end NUMINAMATH_CALUDE_geometric_concept_word_counts_l3556_355650


namespace NUMINAMATH_CALUDE_complex_modulus_problem_l3556_355610

theorem complex_modulus_problem (z : ℂ) (h : (z - 2*Complex.I) * (1 - Complex.I) = -2) : 
  Complex.abs z = Real.sqrt 2 := by sorry

end NUMINAMATH_CALUDE_complex_modulus_problem_l3556_355610


namespace NUMINAMATH_CALUDE_five_million_times_eight_million_l3556_355651

theorem five_million_times_eight_million :
  (5000000 : ℕ) * 8000000 = 40000000000000 := by
  sorry

end NUMINAMATH_CALUDE_five_million_times_eight_million_l3556_355651


namespace NUMINAMATH_CALUDE_imaginary_part_reciprocal_l3556_355677

theorem imaginary_part_reciprocal (a : ℝ) : Complex.im (1 / (a - Complex.I)) = 1 / (1 + a^2) := by
  sorry

end NUMINAMATH_CALUDE_imaginary_part_reciprocal_l3556_355677


namespace NUMINAMATH_CALUDE_toothpick_grid_theorem_l3556_355608

/-- Represents a rectangular grid of toothpicks -/
structure ToothpickGrid where
  height : ℕ
  width : ℕ
  has_diagonals : Bool

/-- Calculates the total number of toothpicks in the grid -/
def total_toothpicks (grid : ToothpickGrid) : ℕ :=
  let horizontal := (grid.height + 1) * grid.width
  let vertical := (grid.width + 1) * grid.height
  let diagonal := if grid.has_diagonals then grid.height * grid.width else 0
  horizontal + vertical + diagonal

/-- The theorem to be proved -/
theorem toothpick_grid_theorem (grid : ToothpickGrid) :
  grid.height = 15 → grid.width = 12 → grid.has_diagonals = true →
  total_toothpicks grid = 567 := by
  sorry

end NUMINAMATH_CALUDE_toothpick_grid_theorem_l3556_355608


namespace NUMINAMATH_CALUDE_line_equation_proof_l3556_355615

/-- Given two lines in the xy-plane -/
def line1 (x y : ℝ) : Prop := x - 2*y - 3 = 0
def line2 (x y : ℝ) : Prop := 2*x - 3*y - 2 = 0

/-- The intersection point of the two lines -/
def intersection_point : ℝ × ℝ := sorry

/-- The equation of the line passing through (2, 1) and the intersection point -/
def target_line (x y : ℝ) : Prop := 5*x - 7*y - 3 = 0

theorem line_equation_proof :
  (∃ (x y : ℝ), line1 x y ∧ line2 x y ∧ (x, y) = intersection_point) →
  target_line (2 : ℝ) 1 ∧
  target_line (intersection_point.1) (intersection_point.2) :=
by sorry

end NUMINAMATH_CALUDE_line_equation_proof_l3556_355615


namespace NUMINAMATH_CALUDE_clock_angle_l3556_355697

theorem clock_angle (hour_hand_angle hour_hand_movement minute_hand_movement : ℝ) :
  hour_hand_angle = 90 →
  hour_hand_movement = 15 →
  minute_hand_movement = 180 →
  180 - hour_hand_angle - hour_hand_movement = 75 :=
by sorry

end NUMINAMATH_CALUDE_clock_angle_l3556_355697


namespace NUMINAMATH_CALUDE_bakery_flour_usage_l3556_355654

theorem bakery_flour_usage : 
  let wheat_flour : ℝ := 0.2
  let white_flour : ℝ := 0.1
  let rye_flour : ℝ := 0.15
  let almond_flour : ℝ := 0.05
  let rice_flour : ℝ := 0.1
  wheat_flour + white_flour + rye_flour + almond_flour + rice_flour = 0.6 := by
  sorry

end NUMINAMATH_CALUDE_bakery_flour_usage_l3556_355654


namespace NUMINAMATH_CALUDE_max_distance_circle_to_line_l3556_355689

/-- The maximum distance from any point on the circle (x-1)^2 + y^2 = 3 to the line x - y - 1 = 0 is √3 -/
theorem max_distance_circle_to_line :
  let circle := {p : ℝ × ℝ | (p.1 - 1)^2 + p.2^2 = 3}
  let line := {p : ℝ × ℝ | p.1 - p.2 - 1 = 0}
  ∀ p ∈ circle, ∃ q ∈ line,
    ∀ r ∈ circle, ∀ s ∈ line,
      Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2) ≤ Real.sqrt 3 ∧
      ∃ p' ∈ circle, ∃ q' ∈ line,
        Real.sqrt ((p'.1 - q'.1)^2 + (p'.2 - q'.2)^2) = Real.sqrt 3 :=
by sorry

end NUMINAMATH_CALUDE_max_distance_circle_to_line_l3556_355689


namespace NUMINAMATH_CALUDE_cosine_squared_inequality_l3556_355684

theorem cosine_squared_inequality (x y : ℝ) : 
  (Real.cos (x - y))^2 ≤ 4 * (1 - Real.sin x * Real.cos y) * (1 - Real.cos x * Real.sin y) := by
  sorry

end NUMINAMATH_CALUDE_cosine_squared_inequality_l3556_355684


namespace NUMINAMATH_CALUDE_arc_cover_theorem_l3556_355658

/-- Represents an arc on a circle -/
structure Arc where
  start : ℝ  -- Start angle in degrees
  length : ℝ  -- Length of the arc in degrees

/-- A set of arcs covering a circle -/
def ArcCover := Set Arc

/-- Predicate to check if a set of arcs covers the entire circle -/
def covers_circle (cover : ArcCover) : Prop := sorry

/-- Predicate to check if any single arc in the set covers the entire circle -/
def has_complete_arc (cover : ArcCover) : Prop := sorry

/-- Calculate the total measure of a set of arcs -/
def total_measure (arcs : Set Arc) : ℝ := sorry

/-- Main theorem -/
theorem arc_cover_theorem (cover : ArcCover) 
  (h1 : covers_circle cover) 
  (h2 : ¬ has_complete_arc cover) : 
  ∃ (subset : Set Arc), subset ⊆ cover ∧ covers_circle subset ∧ total_measure subset ≤ 720 := by
  sorry

end NUMINAMATH_CALUDE_arc_cover_theorem_l3556_355658


namespace NUMINAMATH_CALUDE_fifty_third_odd_positive_integer_l3556_355678

/-- The nth odd positive integer -/
def nthOddPositiveInteger (n : ℕ) : ℕ := 2 * n - 1

/-- Theorem: The 53rd odd positive integer is 105 -/
theorem fifty_third_odd_positive_integer : nthOddPositiveInteger 53 = 105 := by
  sorry

end NUMINAMATH_CALUDE_fifty_third_odd_positive_integer_l3556_355678


namespace NUMINAMATH_CALUDE_simplify_exponential_fraction_l3556_355663

theorem simplify_exponential_fraction (n : ℕ) :
  (3^(n+3) - 3*(3^n)) / (3*(3^(n+2))) = 8/3 := by
  sorry

end NUMINAMATH_CALUDE_simplify_exponential_fraction_l3556_355663


namespace NUMINAMATH_CALUDE_addition_of_like_terms_l3556_355688

theorem addition_of_like_terms (a : ℝ) : 2 * a + a = 3 * a := by
  sorry

end NUMINAMATH_CALUDE_addition_of_like_terms_l3556_355688


namespace NUMINAMATH_CALUDE_ratio_equality_l3556_355619

theorem ratio_equality (a b c : ℝ) 
  (h1 : a / b = 4 / 3) 
  (h2 : a + c / b - c = 5 / 2) : 
  (3 * a + 2 * b) / (3 * a - 2 * b) = 3 := by
sorry

end NUMINAMATH_CALUDE_ratio_equality_l3556_355619


namespace NUMINAMATH_CALUDE_product_of_differences_divisible_by_twelve_l3556_355674

theorem product_of_differences_divisible_by_twelve 
  (a b c d : ℤ) (h_distinct : a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d) : 
  12 ∣ ((a - b) * (a - c) * (a - d) * (b - c) * (b - d) * (c - d)) :=
by sorry

end NUMINAMATH_CALUDE_product_of_differences_divisible_by_twelve_l3556_355674


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_problem_l3556_355676

def arithmetic_sequence (a₁ d : ℕ) (n : ℕ) : ℕ := a₁ + (n - 1) * d

def sum_of_arithmetic_sequence (a₁ d : ℕ) (n : ℕ) : ℕ :=
  n * (2 * a₁ + (n - 1) * d) / 2

theorem arithmetic_sequence_sum_problem (i : ℕ) (k : ℕ) :
  (k > 0) →
  (k ≤ 10) →
  (sum_of_arithmetic_sequence 3 2 10 - arithmetic_sequence 3 2 (i + k) = 185) →
  (sum_of_arithmetic_sequence 3 2 10 = 200) :=
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_problem_l3556_355676


namespace NUMINAMATH_CALUDE_w_in_terms_of_abc_l3556_355611

theorem w_in_terms_of_abc (w a b c x y z : ℝ) 
  (hdistinct : w ≠ a ∧ w ≠ b ∧ w ≠ c ∧ a ≠ b ∧ a ≠ c ∧ b ≠ c)
  (heq1 : x + y + z = 1)
  (heq2 : x*a^2 + y*b^2 + z*c^2 = w^2)
  (heq3 : x*a^3 + y*b^3 + z*c^3 = w^3)
  (heq4 : x*a^4 + y*b^4 + z*c^4 = w^4) :
  w = -a*b*c / (a*b + b*c + c*a) := by
sorry

end NUMINAMATH_CALUDE_w_in_terms_of_abc_l3556_355611


namespace NUMINAMATH_CALUDE_prob_even_sum_spinners_l3556_355681

/-- Represents a spinner with three sections -/
structure Spinner :=
  (sections : Fin 3 → ℕ)

/-- Calculates the probability of getting an even number on a spinner -/
def probEven (s : Spinner) : ℚ :=
  (Finset.filter (λ i => s.sections i % 2 = 0) Finset.univ).card / 3

/-- Calculates the probability of getting an odd number on a spinner -/
def probOdd (s : Spinner) : ℚ :=
  1 - probEven s

/-- The first spinner with sections 2, 3, and 7 -/
def spinner1 : Spinner :=
  ⟨λ i => [2, 3, 7].get i⟩

/-- The second spinner with sections 5, 3, and 6 -/
def spinner2 : Spinner :=
  ⟨λ i => [5, 3, 6].get i⟩

/-- The probability of getting an even sum when spinning both spinners -/
def probEvenSum (s1 s2 : Spinner) : ℚ :=
  probEven s1 * probEven s2 + probOdd s1 * probOdd s2

theorem prob_even_sum_spinners :
  probEvenSum spinner1 spinner2 = 5 / 9 := by
  sorry

end NUMINAMATH_CALUDE_prob_even_sum_spinners_l3556_355681


namespace NUMINAMATH_CALUDE_xyz_value_l3556_355679

-- Define the complex numbers x, y, and z
variable (x y z : ℂ)

-- Define the conditions
def condition1 : Prop := x * y + 5 * y = -20
def condition2 : Prop := y * z + 5 * z = -20
def condition3 : Prop := z * x + 5 * x = -20
def condition4 : Prop := x + y + z = 3

-- Theorem statement
theorem xyz_value (h1 : condition1 x y) (h2 : condition2 y z) (h3 : condition3 z x) (h4 : condition4 x y z) :
  x * y * z = 105 := by
  sorry

end NUMINAMATH_CALUDE_xyz_value_l3556_355679


namespace NUMINAMATH_CALUDE_infinitely_many_common_terms_l3556_355694

/-- Sequence a_n defined recursively -/
def a : ℕ → ℤ
  | 0 => 2
  | 1 => 14
  | (n + 2) => 14 * a (n + 1) + a n

/-- Sequence b_n defined recursively -/
def b : ℕ → ℤ
  | 0 => 2
  | 1 => 14
  | (n + 2) => 6 * b (n + 1) - b n

/-- There are infinitely many common terms in sequences a and b -/
theorem infinitely_many_common_terms :
  ∀ N : ℕ, ∃ k : ℕ, k > N ∧ a (2 * k + 1) = b (3 * k + 1) :=
by sorry

end NUMINAMATH_CALUDE_infinitely_many_common_terms_l3556_355694
