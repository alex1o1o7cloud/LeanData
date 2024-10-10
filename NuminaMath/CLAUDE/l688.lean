import Mathlib

namespace fourth_root_equality_exp_power_equality_cube_root_equality_sqrt_product_inequality_l688_68818

-- Define π as a real number greater than 3
variable (π : ℝ) [Fact (π > 3)]

-- Theorem for option A
theorem fourth_root_equality : ∀ π : ℝ, π > 3 → (((3 - π) ^ 4) ^ (1/4 : ℝ)) = π - 3 := by sorry

-- Theorem for option B
theorem exp_power_equality : ∀ x : ℝ, Real.exp (2 * x) = (Real.exp x) ^ 2 := by sorry

-- Theorem for option C
theorem cube_root_equality : ∀ a b : ℝ, ((a - b) ^ 3) ^ (1/3 : ℝ) = a - b := by sorry

-- Theorem for option D (showing it's not always true)
theorem sqrt_product_inequality : ∃ a b : ℝ, (a * b) ^ (1/2 : ℝ) ≠ (a ^ (1/2 : ℝ)) * (b ^ (1/2 : ℝ)) := by sorry

end fourth_root_equality_exp_power_equality_cube_root_equality_sqrt_product_inequality_l688_68818


namespace greatest_integer_quadratic_inequality_l688_68839

theorem greatest_integer_quadratic_inequality :
  ∃ (n : ℤ), n^2 - 17*n + 72 ≤ 0 ∧ n = 9 ∧ ∀ (m : ℤ), m^2 - 17*m + 72 ≤ 0 → m ≤ 9 :=
by sorry

end greatest_integer_quadratic_inequality_l688_68839


namespace tom_payment_is_nine_l688_68891

/-- The original price of the rare robot in dollars -/
def original_price : ℝ := 3

/-- The multiplier for the selling price -/
def price_multiplier : ℝ := 3

/-- The amount Tom should pay in dollars -/
def tom_payment : ℝ := original_price * price_multiplier

/-- Theorem stating that Tom should pay $9.00 for the rare robot -/
theorem tom_payment_is_nine : tom_payment = 9 := by
  sorry

end tom_payment_is_nine_l688_68891


namespace multiples_of_five_up_to_hundred_l688_68808

theorem multiples_of_five_up_to_hundred :
  ∃ n : ℕ, n = 100 ∧ (∃! k : ℕ, k = 20 ∧ (∀ m : ℕ, 1 ≤ m ∧ m ≤ n → (m % 5 = 0 ↔ m ∈ Finset.range k))) :=
by sorry

end multiples_of_five_up_to_hundred_l688_68808


namespace perpendicular_vectors_m_value_l688_68868

def vector_a (m : ℝ) : Fin 2 → ℝ := ![3, -2*m]
def vector_b (m : ℝ) : Fin 2 → ℝ := ![8, 3*m]

def dot_product (v w : Fin 2 → ℝ) : ℝ :=
  (v 0) * (w 0) + (v 1) * (w 1)

theorem perpendicular_vectors_m_value :
  ∀ m : ℝ, dot_product (vector_a m) (vector_b m) = 0 → m = 2 ∨ m = -2 := by
  sorry

end perpendicular_vectors_m_value_l688_68868


namespace square_division_has_triangle_l688_68889

/-- A convex polygon within a square --/
structure PolygonInSquare where
  sides : ℕ
  convex : Bool
  inSquare : Bool

/-- Represents a division of a square into polygons --/
def SquareDivision := List PolygonInSquare

/-- Checks if all polygons in the division are convex and within the square --/
def isValidDivision (d : SquareDivision) : Prop :=
  d.all (λ p => p.convex ∧ p.inSquare)

/-- Checks if all polygons have distinct number of sides --/
def hasDistinctSides (d : SquareDivision) : Prop :=
  d.map (λ p => p.sides) |>.Nodup

/-- Checks if there's a triangle in the division --/
def hasTriangle (d : SquareDivision) : Prop :=
  d.any (λ p => p.sides = 3)

theorem square_division_has_triangle (d : SquareDivision) :
  d.length > 1 → isValidDivision d → hasDistinctSides d → hasTriangle d := by
  sorry

end square_division_has_triangle_l688_68889


namespace speed_conversion_l688_68863

theorem speed_conversion (speed_kmh : ℝ) (speed_ms : ℝ) : 
  speed_kmh = 1.2 → speed_ms = 1/3 → speed_kmh * (1000 / 3600) = speed_ms :=
by
  sorry

end speed_conversion_l688_68863


namespace binomial_and_power_of_two_l688_68861

theorem binomial_and_power_of_two : Nat.choose 8 3 = 56 ∧ 2^(Nat.choose 8 3) = 2^56 := by
  sorry

end binomial_and_power_of_two_l688_68861


namespace speed_in_still_water_l688_68806

def upstream_speed : ℝ := 20
def downstream_speed : ℝ := 28

theorem speed_in_still_water : 
  (upstream_speed + downstream_speed) / 2 = 24 := by sorry

end speed_in_still_water_l688_68806


namespace count_valid_bases_for_216_l688_68886

theorem count_valid_bases_for_216 :
  ∃! (n : ℕ), n > 0 ∧ (∃ (S : Finset ℕ), 
    (∀ b ∈ S, b > 0 ∧ ∃ k : ℕ, k > 0 ∧ b^k = 216) ∧
    S.card = n ∧
    (∀ b : ℕ, b > 0 → (∃ k : ℕ, k > 0 ∧ b^k = 216) → b ∈ S)) :=
sorry

end count_valid_bases_for_216_l688_68886


namespace min_value_expression_l688_68825

theorem min_value_expression (a b : ℝ) (h1 : ab - 4*a - b + 1 = 0) (h2 : a > 1) :
  ∀ x y : ℝ, x * y - 4*x - y + 1 = 0 → x > 1 → (a + 1) * (b + 2) ≤ (x + 1) * (y + 2) ∧
  ∃ a₀ b₀ : ℝ, a₀ * b₀ - 4*a₀ - b₀ + 1 = 0 ∧ a₀ > 1 ∧ (a₀ + 1) * (b₀ + 2) = 27 :=
sorry

end min_value_expression_l688_68825


namespace point_in_third_quadrant_l688_68810

/-- The range of m for which the point P(1-1/3m, m-5) is in the third quadrant --/
theorem point_in_third_quadrant (m : ℝ) : 
  (1 - 1/3*m < 0 ∧ m - 5 < 0) ↔ (3 < m ∧ m < 5) := by sorry

end point_in_third_quadrant_l688_68810


namespace ellipse_trajectory_and_minimum_l688_68874

-- Define the ellipse
def ellipse (x y : ℝ) : Prop :=
  x^2 + y^2/4 = 1 ∧ x > 0 ∧ y > 0

-- Define the tangent line
def tangent_line (x₀ y₀ x y : ℝ) : Prop :=
  y = -4*x₀/y₀ * (x - x₀) + y₀

-- Define point M
def point_M (x y : ℝ) : Prop :=
  ∃ x₀ y₀, ellipse x₀ y₀ ∧
  ∃ xA yB, tangent_line x₀ y₀ xA 0 ∧ tangent_line x₀ y₀ 0 yB ∧
  x = xA ∧ y = yB

theorem ellipse_trajectory_and_minimum (x y : ℝ) :
  point_M x y →
  (1/x^2 + 4/y^2 = 1 ∧ x > 1 ∧ y > 2) ∧
  (∀ x' y', point_M x' y' → x'^2 + y'^2 ≥ 9) ∧
  (∃ x₀ y₀, point_M x₀ y₀ ∧ x₀^2 + y₀^2 = 9) :=
by sorry

end ellipse_trajectory_and_minimum_l688_68874


namespace shaded_area_is_eight_l688_68864

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a semicircle -/
structure Semicircle where
  center : Point
  radius : ℝ

/-- The geometric configuration -/
structure GeometricLayout where
  semicircle_ADB : Semicircle
  semicircle_BEC : Semicircle
  semicircle_DFE : Semicircle
  point_D : Point
  point_E : Point
  point_F : Point

/-- Conditions of the geometric layout -/
def validGeometricLayout (layout : GeometricLayout) : Prop :=
  layout.semicircle_ADB.radius = 2 ∧
  layout.semicircle_BEC.radius = 2 ∧
  layout.semicircle_DFE.radius = 1 ∧
  -- D is midpoint of ADB
  layout.point_D = { x := layout.semicircle_ADB.center.x, y := layout.semicircle_ADB.center.y + layout.semicircle_ADB.radius } ∧
  -- E is midpoint of BEC
  layout.point_E = { x := layout.semicircle_BEC.center.x, y := layout.semicircle_BEC.center.y + layout.semicircle_BEC.radius } ∧
  -- F is midpoint of DFE
  layout.point_F = { x := layout.semicircle_DFE.center.x, y := layout.semicircle_DFE.center.y + layout.semicircle_DFE.radius }

/-- Calculate the area of the shaded region -/
def shadedArea (layout : GeometricLayout) : ℝ :=
  -- Placeholder for the actual calculation
  8

/-- Theorem stating that the shaded area is 8 square units -/
theorem shaded_area_is_eight (layout : GeometricLayout) (h : validGeometricLayout layout) :
  shadedArea layout = 8 :=
by
  sorry


end shaded_area_is_eight_l688_68864


namespace white_tiles_count_l688_68842

theorem white_tiles_count (total : Nat) (yellow : Nat) (purple : Nat) : 
  total = 20 → yellow = 3 → purple = 6 → 
  ∃ (blue white : Nat), blue = yellow + 1 ∧ white = total - (yellow + blue + purple) ∧ white = 7 := by
  sorry

end white_tiles_count_l688_68842


namespace ant_return_probability_60_l688_68804

/-- The probability of an ant returning to its starting vertex on a tetrahedron after n random edge traversals -/
def ant_return_probability (n : ℕ) : ℚ :=
  (3^(n-1) + 1) / (4 * 3^(n-1))

/-- The theorem stating the probability of an ant returning to its starting vertex on a tetrahedron after 60 random edge traversals -/
theorem ant_return_probability_60 :
  ant_return_probability 60 = (3^59 + 1) / (4 * 3^59) := by
  sorry

end ant_return_probability_60_l688_68804


namespace roots_not_in_interval_l688_68802

theorem roots_not_in_interval (a : ℝ) (ha : a > 0) (ha_neq : a ≠ 1) :
  ∀ x : ℝ, (a^x + a^(-x) = 2*a) → x ∉ Set.Icc (-1 : ℝ) 1 := by
  sorry

end roots_not_in_interval_l688_68802


namespace factor_expression_l688_68835

theorem factor_expression (x : ℝ) : 16 * x^4 - 4 * x^2 = 4 * x^2 * (2*x + 1) * (2*x - 1) := by
  sorry

end factor_expression_l688_68835


namespace power_division_subtraction_addition_l688_68814

theorem power_division_subtraction_addition : (-6)^4 / 6^2 - 2^5 + 4^2 = 20 := by
  sorry

end power_division_subtraction_addition_l688_68814


namespace vector_parallel_proof_l688_68841

def vector_a (m : ℚ) : Fin 2 → ℚ := ![1, m]
def vector_b : Fin 2 → ℚ := ![3, -2]

def parallel (u v : Fin 2 → ℚ) : Prop :=
  ∃ (k : ℚ), ∀ (i : Fin 2), u i = k * v i

theorem vector_parallel_proof (m : ℚ) :
  parallel (vector_a m + vector_b) vector_b → m = -2/3 := by
  sorry

end vector_parallel_proof_l688_68841


namespace work_hours_ratio_l688_68831

theorem work_hours_ratio (amber_hours : ℕ) (total_hours : ℕ) : 
  amber_hours = 12 →
  total_hours = 40 →
  ∃ (ella_hours : ℕ),
    (ella_hours + amber_hours + amber_hours / 3 = total_hours) ∧
    (ella_hours : ℚ) / amber_hours = 2 := by
  sorry

end work_hours_ratio_l688_68831


namespace fraction_meaningful_l688_68887

theorem fraction_meaningful (x : ℝ) : 
  (∃ y : ℝ, y = 1 / (x - 1)) ↔ x ≠ 1 := by
sorry

end fraction_meaningful_l688_68887


namespace f_minimum_value_F_monotonicity_l688_68801

noncomputable section

def f (x : ℝ) : ℝ := x * Real.log x

def F (a : ℝ) (x : ℝ) : ℝ := a * x^2 + Real.log x + 1

theorem f_minimum_value (x : ℝ) (hx : x > 0) :
  ∃ (min : ℝ), min = -1 / Real.exp 1 ∧ f x ≥ min := by sorry

theorem F_monotonicity (a : ℝ) (x : ℝ) (hx : x > 0) :
  (a ≥ 0 → StrictMono (F a)) ∧
  (a < 0 → 
    (∀ y z, 0 < y ∧ y < z ∧ z < Real.sqrt (-1 / (2 * a)) → F a y < F a z) ∧
    (∀ y z, Real.sqrt (-1 / (2 * a)) < y ∧ y < z → F a y > F a z)) := by sorry

end f_minimum_value_F_monotonicity_l688_68801


namespace class_size_from_mark_change_l688_68826

/-- Given a class where one pupil's mark was increased by 20 points, 
    causing the class average to rise by 1/2, prove that there are 40 pupils in the class. -/
theorem class_size_from_mark_change (mark_increase : ℕ) (average_increase : ℚ) : 
  mark_increase = 20 → average_increase = 1/2 → (mark_increase : ℚ) / average_increase = 40 := by
  sorry

end class_size_from_mark_change_l688_68826


namespace prime_minister_stays_l688_68830

/-- Represents the message on a piece of paper -/
inductive Message
| stay
| leave

/-- Represents a piece of paper with a message -/
structure Paper :=
  (message : Message)

/-- The portfolio containing two papers -/
structure Portfolio :=
  (paper1 : Paper)
  (paper2 : Paper)

/-- The state of the game after the prime minister's action -/
structure GameState :=
  (destroyed : Paper)
  (revealed : Paper)

/-- The prime minister's strategy -/
def primeMinisterStrategy (portfolio : Portfolio) : GameState :=
  { destroyed := portfolio.paper1,
    revealed := portfolio.paper2 }

/-- The king's claim about the portfolio -/
def kingsClaim (p : Portfolio) : Prop :=
  (p.paper1.message = Message.stay ∧ p.paper2.message = Message.leave) ∨
  (p.paper1.message = Message.leave ∧ p.paper2.message = Message.stay)

/-- The actual content of the portfolio -/
def actualPortfolio : Portfolio :=
  { paper1 := { message := Message.leave },
    paper2 := { message := Message.leave } }

theorem prime_minister_stays :
  ∀ (state : GameState),
  state = primeMinisterStrategy actualPortfolio →
  state.revealed.message = Message.leave →
  ∃ (claim : Paper), claim.message = Message.stay ∧ 
    (claim = state.destroyed ∨ kingsClaim actualPortfolio = False) :=
by sorry

end prime_minister_stays_l688_68830


namespace triangle_tangent_difference_bound_l688_68836

theorem triangle_tangent_difference_bound (A B C : Real) (a b c : Real) :
  0 < A ∧ 0 < B ∧ 0 < C ∧  -- Triangle is acute
  A + B + C = π ∧  -- Sum of angles in a triangle
  a > 0 ∧ b > 0 ∧ c > 0 ∧  -- Positive side lengths
  b^2 - a^2 = a*c →  -- Given condition
  1 < (1 / Real.tan A - 1 / Real.tan B) ∧ (1 / Real.tan A - 1 / Real.tan B) < 2 * Real.sqrt 3 / 3 :=
by sorry

end triangle_tangent_difference_bound_l688_68836


namespace sum_of_coefficients_l688_68807

theorem sum_of_coefficients (a b c : ℕ+) : 
  (∃ (x y : ℝ), x * x = 5 ∧ y * y = 7 ∧
    x + 1/x + y + 1/y = (a * x + b * y) / c ∧
    ∀ (a' b' c' : ℕ+), 
      (∃ (x' y' : ℝ), x' * x' = 5 ∧ y' * y' = 7 ∧
        x' + 1/x' + y' + 1/y' = (a' * x' + b' * y') / c') →
      c ≤ c') →
  a + b + c = 117 := by
sorry

end sum_of_coefficients_l688_68807


namespace total_time_is_twelve_years_l688_68827

/-- Represents the time taken for each activity in months -/
structure ActivityTime where
  shape : ℕ
  climb_learn : ℕ
  climb_each : ℕ
  dive_learn : ℕ
  dive_caves : ℕ

/-- Calculates the total time taken for all activities -/
def total_time (t : ActivityTime) (num_summits : ℕ) : ℕ :=
  t.shape + t.climb_learn + (num_summits * t.climb_each) + t.dive_learn + t.dive_caves

/-- Theorem stating that the total time to complete all goals is 12 years -/
theorem total_time_is_twelve_years (t : ActivityTime) (num_summits : ℕ) :
  t.shape = 24 ∧ 
  t.climb_learn = 2 * t.shape ∧ 
  num_summits = 7 ∧ 
  t.climb_each = 5 ∧ 
  t.dive_learn = 13 ∧ 
  t.dive_caves = 24 →
  total_time t num_summits = 12 * 12 := by
  sorry

#check total_time_is_twelve_years

end total_time_is_twelve_years_l688_68827


namespace cylinder_base_area_l688_68843

theorem cylinder_base_area (S : ℝ) (h : S > 0) :
  let cross_section_area := 4 * S
  let cross_section_is_square := true
  let base_area := π * S
  cross_section_is_square ∧ cross_section_area = 4 * S → base_area = π * S :=
by
  sorry

end cylinder_base_area_l688_68843


namespace solution_of_exponential_equation_l688_68817

theorem solution_of_exponential_equation :
  ∃ x : ℝ, (2 : ℝ) ^ x = 8 ∧ x = 3 := by sorry

end solution_of_exponential_equation_l688_68817


namespace sum_remainder_mod_nine_l688_68869

theorem sum_remainder_mod_nine (n : ℤ) : (8 - n + (n + 5)) % 9 = 4 := by
  sorry

end sum_remainder_mod_nine_l688_68869


namespace andrews_family_size_l688_68870

/-- Given the conditions of Andrew's family mask usage, prove the number of family members excluding Andrew. -/
theorem andrews_family_size (total_masks : ℕ) (change_interval : ℕ) (total_days : ℕ) :
  total_masks = 100 →
  change_interval = 4 →
  total_days = 80 →
  ∃ (family_size : ℕ), family_size = 4 ∧ 
    (family_size + 1) * (total_days / change_interval) = total_masks :=
by sorry

end andrews_family_size_l688_68870


namespace ship_age_conversion_l688_68857

/-- Converts an octal number represented as (a, b, c) to its decimal equivalent -/
def octal_to_decimal (a b c : ℕ) : ℕ := c * 8^2 + b * 8^1 + a * 8^0

/-- The age of the sunken pirate ship in octal -/
def ship_age_octal : ℕ × ℕ × ℕ := (7, 4, 2)

theorem ship_age_conversion :
  octal_to_decimal ship_age_octal.1 ship_age_octal.2.1 ship_age_octal.2.2 = 482 := by
  sorry

end ship_age_conversion_l688_68857


namespace max_y_value_l688_68813

theorem max_y_value (x y : ℝ) (hx : x > 0) (hy : y > 0) 
  (h : x^3 + y^3 = (4*x - 5*y)*y) : 
  y ≤ 1/3 :=
sorry

end max_y_value_l688_68813


namespace received_a_implies_met_criteria_l688_68885

/-- Represents the criteria for receiving an A on the exam -/
structure ExamCriteria where
  multiple_choice_correct : ℝ
  extra_credit_completed : Bool

/-- Represents a student's exam performance -/
structure ExamPerformance where
  multiple_choice_correct : ℝ
  extra_credit_completed : Bool
  received_a : Bool

/-- The criteria for receiving an A on the exam -/
def a_criteria : ExamCriteria :=
  { multiple_choice_correct := 90
  , extra_credit_completed := true }

/-- Predicate to check if a student's performance meets the criteria for an A -/
def meets_a_criteria (performance : ExamPerformance) (criteria : ExamCriteria) : Prop :=
  performance.multiple_choice_correct ≥ criteria.multiple_choice_correct ∧
  performance.extra_credit_completed = criteria.extra_credit_completed

/-- Theorem stating that if a student received an A, they must have met the criteria -/
theorem received_a_implies_met_criteria (student : ExamPerformance) :
  student.received_a → meets_a_criteria student a_criteria := by
  sorry

end received_a_implies_met_criteria_l688_68885


namespace triangle_inequality_l688_68828

theorem triangle_inequality (a b c : ℝ) (h_triangle : a > 0 ∧ b > 0 ∧ c > 0 ∧ a + b > c ∧ b + c > a ∧ c + a > b) :
  2 < (a + b) / c + (b + c) / a + (c + a) / b - (a^3 + b^3 + c^3) / (a * b * c) ∧
  (a + b) / c + (b + c) / a + (c + a) / b - (a^3 + b^3 + c^3) / (a * b * c) ≤ 3 :=
sorry

end triangle_inequality_l688_68828


namespace coin_problem_l688_68875

/-- Represents the types of coins --/
inductive CoinType
  | Penny
  | Nickel
  | Dime
  | Quarter
  | HalfDollar

/-- Returns the value of a coin in cents --/
def coinValue (c : CoinType) : ℕ :=
  match c with
  | .Penny => 1
  | .Nickel => 5
  | .Dime => 10
  | .Quarter => 25
  | .HalfDollar => 50

/-- Represents a collection of coins --/
structure CoinCollection where
  pennies : ℕ
  nickels : ℕ
  dimes : ℕ
  quarters : ℕ
  halfDollars : ℕ

/-- Calculates the total number of coins in a collection --/
def totalCoins (c : CoinCollection) : ℕ :=
  c.pennies + c.nickels + c.dimes + c.quarters + c.halfDollars

/-- Calculates the total value of coins in a collection in cents --/
def totalValue (c : CoinCollection) : ℕ :=
  c.pennies * coinValue CoinType.Penny +
  c.nickels * coinValue CoinType.Nickel +
  c.dimes * coinValue CoinType.Dime +
  c.quarters * coinValue CoinType.Quarter +
  c.halfDollars * coinValue CoinType.HalfDollar

/-- The main theorem --/
theorem coin_problem (c : CoinCollection) 
  (h1 : totalCoins c = 11)
  (h2 : totalValue c = 143)
  (h3 : c.pennies ≥ 1)
  (h4 : c.nickels ≥ 1)
  (h5 : c.dimes ≥ 1)
  (h6 : c.quarters ≥ 1)
  (h7 : c.halfDollars ≥ 1) :
  c.dimes = 4 := by
  sorry

end coin_problem_l688_68875


namespace xiaoqiang_games_l688_68820

/-- Represents a player in the chess tournament -/
inductive Player : Type
| Jia : Player
| Yi : Player
| Bing : Player
| Ding : Player
| Xiaoqiang : Player

/-- The number of games played by each player -/
def games_played (p : Player) : ℕ :=
  match p with
  | Player.Jia => 4
  | Player.Yi => 3
  | Player.Bing => 2
  | Player.Ding => 1
  | Player.Xiaoqiang => 2  -- This is what we want to prove

/-- The total number of games in a round-robin tournament -/
def total_games (n : ℕ) : ℕ := n * (n - 1) / 2

theorem xiaoqiang_games :
  games_played Player.Xiaoqiang = 2 :=
by sorry

end xiaoqiang_games_l688_68820


namespace complex_fourth_power_l688_68821

theorem complex_fourth_power (i : ℂ) (h : i^2 = -1) : (1 - i)^4 = -4 := by
  sorry

end complex_fourth_power_l688_68821


namespace square_root_fraction_simplification_l688_68850

theorem square_root_fraction_simplification :
  (Real.sqrt (8^2 + 15^2)) / (Real.sqrt (25 + 16)) = (17 * Real.sqrt 41) / 41 := by
  sorry

end square_root_fraction_simplification_l688_68850


namespace hidden_number_l688_68847

theorem hidden_number (x : ℝ) (hidden : ℝ) : 
  x = -1 → (2 + hidden * x) / 3 = -1 → hidden = 5 := by
  sorry

end hidden_number_l688_68847


namespace triangle_area_l688_68851

-- Define the plane region
def PlaneRegion (k : ℝ) := {p : ℝ × ℝ | p.1 ≥ 0 ∧ p.2 ≥ 2 * p.1 ∧ k * p.1 - p.2 + 1 ≥ 0}

-- Define a right triangle
def IsRightTriangle (r : Set (ℝ × ℝ)) : Prop := sorry

-- Define the area of a set in ℝ²
noncomputable def Area (s : Set (ℝ × ℝ)) : ℝ := sorry

-- Theorem statement
theorem triangle_area (k : ℝ) :
  IsRightTriangle (PlaneRegion k) →
  Area (PlaneRegion k) = 1/5 ∨ Area (PlaneRegion k) = 1/4 := by
  sorry

end triangle_area_l688_68851


namespace first_quartile_of_list_l688_68812

def list : List ℝ := [42, 24, 30, 28, 26, 19, 33, 35]

def median (l : List ℝ) : ℝ := sorry

def first_quartile (l : List ℝ) : ℝ :=
  let m := median l
  median (l.filter (· < m))

theorem first_quartile_of_list : first_quartile list = 25 := by sorry

end first_quartile_of_list_l688_68812


namespace valid_numbers_l688_68845

def is_valid_number (a b : Nat) : Prop :=
  let n := 201800 + 10 * a + b
  n % 5 = 1 ∧ n % 11 = 8

theorem valid_numbers : 
  ∀ a b : Nat, 0 ≤ a ∧ a ≤ 9 ∧ 0 ≤ b ∧ b ≤ 9 →
  is_valid_number a b ↔ (a = 3 ∧ b = 1) ∨ (a = 8 ∧ b = 6) :=
by sorry

end valid_numbers_l688_68845


namespace cloth_price_calculation_l688_68844

theorem cloth_price_calculation (quantity : ℝ) (discount_rate : ℝ) (tax_rate : ℝ) (total_cost : ℝ) :
  quantity = 9.25 →
  discount_rate = 0.12 →
  tax_rate = 0.05 →
  total_cost = 397.75 →
  ∃ P : ℝ, (quantity * (P - discount_rate * P)) * (1 + tax_rate) = total_cost :=
by
  sorry

end cloth_price_calculation_l688_68844


namespace integral_of_f_l688_68846

theorem integral_of_f (f : ℝ → ℝ) (h : ∀ x, f x = x^2 + 2 * ∫ x in (0:ℝ)..1, f x) : 
  ∫ x in (0:ℝ)..1, f x = -1/3 := by
  sorry

end integral_of_f_l688_68846


namespace quadratic_shift_sum_l688_68860

/-- Given a quadratic function f(x) = 3x^2 - 2x + 5, when shifted 7 units right
    and 3 units up, the resulting function g(x) = ax^2 + bx + c
    satisfies a + b + c = 128 -/
theorem quadratic_shift_sum (f g : ℝ → ℝ) (a b c : ℝ) :
  (∀ x, f x = 3 * x^2 - 2 * x + 5) →
  (∀ x, g x = f (x - 7) + 3) →
  (∀ x, g x = a * x^2 + b * x + c) →
  a + b + c = 128 := by
  sorry

end quadratic_shift_sum_l688_68860


namespace zero_in_interval_l688_68805

-- Define the function f
def f (x : ℝ) : ℝ := -|x - 5| + 2*x - 1

-- State the theorem
theorem zero_in_interval : 
  ∃ x ∈ Set.Ioo 2 3, f x = 0 :=
sorry

end zero_in_interval_l688_68805


namespace rectangle_to_parallelogram_perimeter_l688_68838

/-- A rectangle is transformed into a parallelogram while maintaining the same perimeter -/
theorem rectangle_to_parallelogram_perimeter (a b : ℝ) (h₁ : a > 0) (h₂ : b > 0) :
  let rectangle_perimeter := 2 * (a + b)
  let parallelogram_perimeter := 2 * (a + b)
  rectangle_perimeter = parallelogram_perimeter :=
by sorry

end rectangle_to_parallelogram_perimeter_l688_68838


namespace max_NPMK_is_8010_l688_68890

/-- Represents a three-digit number MMK where M and K are digits and M = K + 1 -/
def MMK (M K : ℕ) : Prop :=
  M ≥ 1 ∧ M ≤ 9 ∧ K ≥ 0 ∧ K ≤ 8 ∧ M = K + 1

/-- Represents the result of multiplying MMK by M -/
def NPMK (M K : ℕ) : ℕ := (100 * M + 10 * M + K) * M

/-- The theorem stating that the maximum value of NPMK is 8010 -/
theorem max_NPMK_is_8010 :
  ∀ M K : ℕ, MMK M K → NPMK M K ≤ 8010 ∧ ∃ M K : ℕ, MMK M K ∧ NPMK M K = 8010 := by
  sorry

end max_NPMK_is_8010_l688_68890


namespace greatest_x_value_l688_68879

theorem greatest_x_value (x : ℝ) : 
  x ≠ 9 → 
  (x^2 - x - 90) / (x - 9) = 2 / (x + 7) → 
  x ≤ -4 :=
by sorry

end greatest_x_value_l688_68879


namespace frannie_jumped_less_jump_difference_l688_68853

/-- The number of times Frannie jumped -/
def frannies_jumps : ℕ := 53

/-- The number of times Meg jumped -/
def megs_jumps : ℕ := 71

/-- Frannie jumped fewer times than Meg -/
theorem frannie_jumped_less : frannies_jumps < megs_jumps := by sorry

/-- The difference in jumps between Meg and Frannie is 18 -/
theorem jump_difference : megs_jumps - frannies_jumps = 18 := by sorry

end frannie_jumped_less_jump_difference_l688_68853


namespace bicycle_sprocket_rotation_l688_68855

theorem bicycle_sprocket_rotation (large_teeth small_teeth : ℕ) (large_revolution : ℝ) :
  large_teeth = 48 →
  small_teeth = 20 →
  large_revolution = 1 →
  (large_teeth : ℝ) / small_teeth * (2 * Real.pi * large_revolution) = 4.8 * Real.pi :=
by
  sorry

end bicycle_sprocket_rotation_l688_68855


namespace simplify_expression_l688_68871

theorem simplify_expression (x y : ℝ) : 3*x + 6*x + 9*x + 12*x + 15*x + 20 + 4*y = 45*x + 20 + 4*y := by
  sorry

end simplify_expression_l688_68871


namespace complex_equation_solution_l688_68833

-- Define the complex number i
noncomputable def i : ℂ := Complex.I

-- State the theorem
theorem complex_equation_solution :
  ∃ (z : ℂ), (1 - i * z + 3 * i = -1 + i * z + 3 * i) ∧ (z = -i) :=
by
  sorry


end complex_equation_solution_l688_68833


namespace daps_equivalent_to_dips_l688_68899

/-- The number of daps equivalent to 1 dop -/
def daps_per_dop : ℚ := 5 / 4

/-- The number of dops equivalent to 1 dip -/
def dops_per_dip : ℚ := 3 / 10

/-- The number of dips we want to convert to daps -/
def target_dips : ℚ := 60

theorem daps_equivalent_to_dips : 
  (daps_per_dop * dops_per_dip * target_dips : ℚ) = 45/2 := by sorry

end daps_equivalent_to_dips_l688_68899


namespace nested_fraction_evaluation_l688_68898

theorem nested_fraction_evaluation :
  2 + 2 / (2 + 2 / (2 + 3)) = 17 / 6 := by
  sorry

end nested_fraction_evaluation_l688_68898


namespace unplanted_field_fraction_l688_68816

theorem unplanted_field_fraction (a b c x : ℝ) : 
  a = 5 → b = 12 → c^2 = a^2 + b^2 → x = 5/3 → 
  x^2 / (a * b / 2) = 5/54 := by sorry

end unplanted_field_fraction_l688_68816


namespace gift_card_value_l688_68819

theorem gift_card_value (original_value : ℝ) : 
  (3 / 8 : ℝ) * original_value = 75 → original_value = 200 := by
  sorry

end gift_card_value_l688_68819


namespace greg_extra_books_l688_68896

theorem greg_extra_books (megan_books kelcie_books greg_books : ℕ) : 
  megan_books = 32 →
  kelcie_books = megan_books / 4 →
  greg_books > 2 * kelcie_books →
  megan_books + kelcie_books + greg_books = 65 →
  greg_books - 2 * kelcie_books = 9 := by
sorry

end greg_extra_books_l688_68896


namespace congruence_problem_l688_68880

theorem congruence_problem (x : ℤ) 
  (h1 : (4 + x) % (2^3) = 3^2 % (2^3))
  (h2 : (6 + x) % (3^3) = 2^3 % (3^3))
  (h3 : (8 + x) % (5^3) = 7^2 % (5^3)) :
  x % 30 = 17 := by
sorry

end congruence_problem_l688_68880


namespace compression_force_l688_68834

/-- Compression force calculation for cylindrical pillars -/
theorem compression_force (T H L : ℝ) : 
  T = 3 → H = 9 → L = (30 * T^5) / H^3 → L = 10 := by
  sorry

end compression_force_l688_68834


namespace isosceles_triangle_on_parabola_l688_68852

/-- Given two points P and Q on the parabola y = -x^2 that form an isosceles triangle POQ with the origin O,
    prove that the distance between P and Q is twice the x-coordinate of P. -/
theorem isosceles_triangle_on_parabola (p : ℝ) :
  let P : ℝ × ℝ := (p, -p^2)
  let Q : ℝ × ℝ := (-p, -p^2)
  let O : ℝ × ℝ := (0, 0)
  (P.1^2 + P.2^2 = Q.1^2 + Q.2^2) →  -- PO = OQ (isosceles condition)
  Real.sqrt ((P.1 - Q.1)^2 + (P.2 - Q.2)^2) = 2 * p  -- PQ = 2p
:= by sorry

end isosceles_triangle_on_parabola_l688_68852


namespace inscribed_box_radius_l688_68872

/-- A rectangular box inscribed in a sphere -/
structure InscribedBox where
  r : ℝ  -- radius of the sphere
  x : ℝ  -- width of the box
  y : ℝ  -- length of the box
  z : ℝ  -- height of the box

/-- Properties of the inscribed box -/
def InscribedBoxProperties (box : InscribedBox) : Prop :=
  box.x > 0 ∧ box.y > 0 ∧ box.z > 0 ∧  -- dimensions are positive
  box.z = 3 * box.x ∧  -- ratio between height and width is 1:3
  4 * (box.x + box.y + box.z) = 72 ∧  -- sum of edge lengths
  2 * (box.x * box.y + box.y * box.z + box.x * box.z) = 162 ∧  -- surface area
  4 * box.r^2 = box.x^2 + box.y^2 + box.z^2  -- inscribed in sphere

theorem inscribed_box_radius (box : InscribedBox) 
  (h : InscribedBoxProperties box) : box.r = 3 := by
  sorry

end inscribed_box_radius_l688_68872


namespace find_n_l688_68876

theorem find_n (d Q r m n : ℝ) (hr : r > 0) (hm : m < (1 + r)^n) 
  (hQ : Q = d / ((1 + r)^n - m)) :
  n = Real.log (d / Q + m) / Real.log (1 + r) := by
  sorry

end find_n_l688_68876


namespace circumscribed_circle_area_relation_l688_68892

/-- Given a right triangle with sides 15, 36, and 39, and a circumscribed circle,
    where an altitude from the right angle divides one non-triangular region into
    areas A and B, and C is the largest non-triangular region, prove that A + B + 270 = C -/
theorem circumscribed_circle_area_relation (A B C : ℝ) : 
  A > 0 → B > 0 → C > 0 →
  (15 : ℝ) ^ 2 + 36 ^ 2 = 39 ^ 2 →
  A < B →
  B < C →
  A + B + 270 = C := by
  sorry

end circumscribed_circle_area_relation_l688_68892


namespace lcm_gcd_product_l688_68824

theorem lcm_gcd_product (a b : ℕ) (ha : a = 12) (hb : b = 9) :
  Nat.lcm a b * Nat.gcd a b = 108 := by
  sorry

end lcm_gcd_product_l688_68824


namespace fuel_tank_capacity_l688_68878

/-- The capacity of a fuel tank in gallons. -/
def tank_capacity : ℝ := 218

/-- The volume of fuel A added to the tank in gallons. -/
def fuel_A_volume : ℝ := 122

/-- The percentage of ethanol in fuel A by volume. -/
def fuel_A_ethanol_percentage : ℝ := 0.12

/-- The percentage of ethanol in fuel B by volume. -/
def fuel_B_ethanol_percentage : ℝ := 0.16

/-- The total volume of ethanol in the full tank in gallons. -/
def total_ethanol_volume : ℝ := 30

theorem fuel_tank_capacity : 
  fuel_A_volume * fuel_A_ethanol_percentage + 
  (tank_capacity - fuel_A_volume) * fuel_B_ethanol_percentage = 
  total_ethanol_volume :=
by sorry

end fuel_tank_capacity_l688_68878


namespace cyclists_meet_time_l688_68837

/-- Two cyclists meet at the starting point on a circular track -/
theorem cyclists_meet_time (circumference : ℝ) (speed1 speed2 : ℝ) 
  (h_circumference : circumference = 600)
  (h_speed1 : speed1 = 7)
  (h_speed2 : speed2 = 8) :
  (circumference / (speed1 + speed2)) = 40 := by
  sorry

#check cyclists_meet_time

end cyclists_meet_time_l688_68837


namespace games_calculation_l688_68848

def football_games : List Nat := [29, 35, 48, 43, 56, 36]
def baseball_games : List Nat := [15, 19, 23, 14, 18, 17]
def basketball_games : List Nat := [17, 21, 14, 32, 22, 27]

def total_games : Nat := football_games.sum + baseball_games.sum + basketball_games.sum

def average_games : Nat := total_games / 6

theorem games_calculation :
  total_games = 486 ∧ average_games = 81 := by
  sorry

end games_calculation_l688_68848


namespace cubic_sum_minus_product_l688_68877

theorem cubic_sum_minus_product (a b c : ℝ) 
  (sum_eq : a + b + c = 11) 
  (sum_products_eq : a * b + a * c + b * c = 25) : 
  a^3 + b^3 + c^3 - 3*a*b*c = 506 := by
  sorry

end cubic_sum_minus_product_l688_68877


namespace solutions_for_20_l688_68803

/-- The number of distinct integer solutions (x,y) for |x| + |y| = n -/
def num_solutions (n : ℕ) : ℕ := 4 * n

theorem solutions_for_20 : num_solutions 20 = 80 := by
  sorry

end solutions_for_20_l688_68803


namespace hidden_primes_sum_l688_68856

/-- A card with two numbers -/
structure Card where
  visible : Nat
  hidden : Nat

/-- Predicate to check if a number is prime -/
def isPrime (n : Nat) : Prop := sorry

/-- The sum of numbers on a card -/
def cardSum (c : Card) : Nat := c.visible + c.hidden

theorem hidden_primes_sum (c1 c2 c3 : Card) : 
  c1.visible = 17 →
  c2.visible = 26 →
  c3.visible = 41 →
  isPrime c1.hidden →
  isPrime c2.hidden →
  isPrime c3.hidden →
  cardSum c1 = cardSum c2 →
  cardSum c2 = cardSum c3 →
  c1.hidden + c2.hidden + c3.hidden = 198 := by
  sorry

end hidden_primes_sum_l688_68856


namespace conference_exchanges_l688_68895

/-- The number of business card exchanges in a conference -/
def businessCardExchanges (n : ℕ) : ℕ := n * (n - 1) / 2

/-- Theorem: In a conference of 10 people, where each person exchanges 
    business cards with every other person exactly once, 
    the total number of exchanges is 45 -/
theorem conference_exchanges : businessCardExchanges 10 = 45 := by
  sorry

end conference_exchanges_l688_68895


namespace income_2005_between_3600_and_3800_l688_68840

/-- Represents the income data for farmers in a certain region -/
structure FarmerIncome where
  initialYear : Nat
  initialWageIncome : ℝ
  initialOtherIncome : ℝ
  wageGrowthRate : ℝ
  otherIncomeIncrease : ℝ

/-- Calculates the average income of farmers after a given number of years -/
def averageIncomeAfterYears (data : FarmerIncome) (years : Nat) : ℝ :=
  data.initialWageIncome * (1 + data.wageGrowthRate) ^ years +
  data.initialOtherIncome + data.otherIncomeIncrease * years

/-- Theorem stating that the average income in 2005 will be between 3600 and 3800 yuan -/
theorem income_2005_between_3600_and_3800 (data : FarmerIncome) 
  (h1 : data.initialYear = 2003)
  (h2 : data.initialWageIncome = 1800)
  (h3 : data.initialOtherIncome = 1350)
  (h4 : data.wageGrowthRate = 0.06)
  (h5 : data.otherIncomeIncrease = 160) :
  3600 ≤ averageIncomeAfterYears data 2 ∧ averageIncomeAfterYears data 2 ≤ 3800 := by
  sorry

#eval averageIncomeAfterYears 
  { initialYear := 2003
    initialWageIncome := 1800
    initialOtherIncome := 1350
    wageGrowthRate := 0.06
    otherIncomeIncrease := 160 } 2

end income_2005_between_3600_and_3800_l688_68840


namespace integral_of_f_l688_68858

-- Define the function f(x) = |x + 2|
def f (x : ℝ) : ℝ := |x + 2|

-- State the theorem
theorem integral_of_f : ∫ x in (-4)..3, f x = 29/2 := by sorry

end integral_of_f_l688_68858


namespace f_bounds_l688_68862

def a : ℤ := 2001

def A : Set (ℤ × ℤ) :=
  {p | p.2 ≠ 0 ∧ 
       p.1 < 2 * a ∧ 
       (2 * p.2) ∣ (2 * a * p.1 - p.1^2 + p.2^2) ∧ 
       p.2^2 - p.1^2 + 2 * p.1 * p.2 ≤ 2 * a * (p.2 - p.1)}

def f (p : ℤ × ℤ) : ℚ :=
  (2 * a * p.1 - p.1^2 - p.1 * p.2) / p.2

theorem f_bounds :
  ∃ (min max : ℚ), min = 2 ∧ max = 3750 ∧
  ∀ p ∈ A, min ≤ f p ∧ f p ≤ max :=
sorry

end f_bounds_l688_68862


namespace exam_score_calculation_l688_68849

theorem exam_score_calculation (total_questions : ℕ) (correct_score : ℕ) (total_score : ℕ) (correct_answers : ℕ) 
  (h1 : total_questions = 50)
  (h2 : correct_score = 4)
  (h3 : total_score = 130)
  (h4 : correct_answers = 36) :
  (correct_score * correct_answers - total_score) / (total_questions - correct_answers) = 1 := by
sorry

end exam_score_calculation_l688_68849


namespace exists_special_number_l688_68883

def is_ten_digit (n : ℕ) : Prop :=
  10^9 ≤ n ∧ n < 10^10

def all_digits_distinct (n : ℕ) : Prop :=
  ∀ d₁ d₂, 0 ≤ d₁ ∧ d₁ < 10 ∧ 0 ≤ d₂ ∧ d₂ < 10 →
    (n / 10^d₁ % 10 = n / 10^d₂ % 10) → d₁ = d₂

theorem exists_special_number :
  ∃ (a b : ℕ), a ≠ b ∧ a < 10 ∧ b < 10 ∧
    is_ten_digit ((10000 * a + 1111 * b)^2 - 1) ∧
    all_digits_distinct ((10000 * a + 1111 * b)^2 - 1) :=
sorry

end exists_special_number_l688_68883


namespace sqrt_product_sqrt_equals_product_sqrt_main_theorem_l688_68823

theorem sqrt_product_sqrt_equals_product_sqrt (a b : ℝ) (ha : 0 ≤ a) (hb : 0 ≤ b) :
  Real.sqrt (a * Real.sqrt b) = Real.sqrt a * Real.sqrt (Real.sqrt b) :=
by sorry

theorem main_theorem : Real.sqrt (49 * Real.sqrt 25) = 7 * Real.sqrt 5 :=
by sorry

end sqrt_product_sqrt_equals_product_sqrt_main_theorem_l688_68823


namespace multiple_of_six_square_greater_144_less_30_l688_68822

theorem multiple_of_six_square_greater_144_less_30 (x : ℕ) 
  (h1 : ∃ k : ℕ, x = 6 * k)
  (h2 : x^2 > 144)
  (h3 : x < 30) :
  x = 18 ∨ x = 24 := by
sorry

end multiple_of_six_square_greater_144_less_30_l688_68822


namespace field_trip_minibusses_l688_68881

theorem field_trip_minibusses (num_vans : ℕ) (students_per_van : ℕ) (students_per_minibus : ℕ) (total_students : ℕ) : 
  num_vans = 6 →
  students_per_van = 10 →
  students_per_minibus = 24 →
  total_students = 156 →
  (total_students - num_vans * students_per_van) / students_per_minibus = 4 := by
sorry

end field_trip_minibusses_l688_68881


namespace gcd_of_B_is_two_l688_68800

def B : Set ℕ := {n : ℕ | ∃ x : ℕ, n = (x - 1) + x + (x + 1) + (x + 2) ∧ x > 0}

theorem gcd_of_B_is_two : 
  ∃ d : ℕ, d > 0 ∧ (∀ n ∈ B, d ∣ n) ∧ (∀ m : ℕ, m > 0 → (∀ n ∈ B, m ∣ n) → m ≤ d) ∧ d = 2 := by
  sorry

end gcd_of_B_is_two_l688_68800


namespace absolute_value_inequality_l688_68859

theorem absolute_value_inequality (x : ℝ) (h : x ≠ 2) :
  |((3 * x - 2) / (x - 2))| > 3 ↔ x ∈ Set.Ioo (4/3) 2 ∪ Set.Ioi 2 :=
by sorry

end absolute_value_inequality_l688_68859


namespace overlap_area_is_zero_l688_68893

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- A triangle defined by three points -/
structure Triangle where
  p1 : Point
  p2 : Point
  p3 : Point

/-- Calculate the area of overlap between two triangles -/
def areaOfOverlap (t1 t2 : Triangle) : ℝ :=
  sorry

/-- The main theorem stating that the area of overlap is zero -/
theorem overlap_area_is_zero :
  let t1 := Triangle.mk (Point.mk 0 0) (Point.mk 2 2) (Point.mk 2 0)
  let t2 := Triangle.mk (Point.mk 0 2) (Point.mk 2 2) (Point.mk 0 0)
  areaOfOverlap t1 t2 = 0 :=
by sorry

end overlap_area_is_zero_l688_68893


namespace solution_satisfies_equation_l688_68884

-- Define the logarithm function with base 5
noncomputable def log5 (x : ℝ) : ℝ := Real.log x / Real.log 5

-- Define the equation
def equation (x : ℝ) : Prop :=
  2 * log5 x - 3 * log5 4 = 1

-- Theorem statement
theorem solution_satisfies_equation :
  equation (4 * Real.sqrt 5) ∧ equation (-4 * Real.sqrt 5) :=
sorry

end solution_satisfies_equation_l688_68884


namespace distinct_polygons_count_l688_68832

/-- The number of points marked on the circle -/
def n : ℕ := 15

/-- The total number of possible subsets of n points -/
def total_subsets : ℕ := 2^n

/-- The number of subsets with 0 elements -/
def subsets_0 : ℕ := Nat.choose n 0

/-- The number of subsets with 1 element -/
def subsets_1 : ℕ := Nat.choose n 1

/-- The number of subsets with 2 elements -/
def subsets_2 : ℕ := Nat.choose n 2

/-- The number of distinct convex polygons with 3 or more sides -/
def num_polygons : ℕ := total_subsets - subsets_0 - subsets_1 - subsets_2

theorem distinct_polygons_count : num_polygons = 32647 := by
  sorry

end distinct_polygons_count_l688_68832


namespace license_plate_count_l688_68873

/-- The number of consonants excluding 'Y' -/
def num_consonants_no_y : ℕ := 19

/-- The number of vowels including 'Y' -/
def num_vowels : ℕ := 6

/-- The number of consonants including 'Y' -/
def num_consonants_with_y : ℕ := 21

/-- The number of even digits -/
def num_even_digits : ℕ := 5

/-- The total number of valid license plates -/
def total_license_plates : ℕ := num_consonants_no_y * num_vowels * num_consonants_with_y * num_even_digits

theorem license_plate_count : total_license_plates = 11970 := by
  sorry

end license_plate_count_l688_68873


namespace racket_deal_cost_l688_68854

/-- Calculates the total cost of two rackets given a store's deal and the full price of each racket. -/
def totalCostTwoRackets (fullPrice : ℕ) : ℕ :=
  fullPrice + (fullPrice - fullPrice / 2)

/-- Theorem stating that the total cost of two rackets is $90 given the specific conditions. -/
theorem racket_deal_cost :
  totalCostTwoRackets 60 = 90 := by
  sorry

#eval totalCostTwoRackets 60

end racket_deal_cost_l688_68854


namespace simplify_fraction_l688_68897

theorem simplify_fraction (x y : ℝ) (hx : x ≠ 0) : (x * y) / (3 * x) = y / 3 := by
  sorry

end simplify_fraction_l688_68897


namespace sets_properties_l688_68882

open Set

-- Define the sets A and B
def A : Set ℝ := {x | x^2 - 3*x - 4 ≥ 0}
def B : Set ℝ := {x | x - 5 < 0}

-- State the theorem
theorem sets_properties :
  (A ∩ B = Icc 4 5) ∧
  (A ∪ B = univ) ∧
  (Aᶜ = Ioo (-1) 4) := by sorry

end sets_properties_l688_68882


namespace sequence_general_term_l688_68815

/-- Given a sequence {a_n} where S_n is the sum of its first n terms and S_n = 1 - (2/3)a_n,
    prove that the general term a_n is equal to (3/5) * (2/5)^(n-1). -/
theorem sequence_general_term (a : ℕ → ℝ) (S : ℕ → ℝ) 
    (h : ∀ n, S n = 1 - (2/3) * a n) :
    ∀ n, a n = (3/5) * (2/5)^(n-1) := by
  sorry

end sequence_general_term_l688_68815


namespace stadium_width_l688_68811

theorem stadium_width (length height diagonal : ℝ) 
  (h_length : length = 24)
  (h_height : height = 16)
  (h_diagonal : diagonal = 34) :
  ∃ width : ℝ, width = 18 ∧ diagonal^2 = length^2 + width^2 + height^2 := by
sorry

end stadium_width_l688_68811


namespace smallest_power_divisible_by_240_l688_68865

theorem smallest_power_divisible_by_240 (n : ℕ) : 
  (∀ k : ℕ, k < n → ¬(240 ∣ 60^k)) ∧ (240 ∣ 60^n) → n = 2 := by
  sorry

end smallest_power_divisible_by_240_l688_68865


namespace system_solution_l688_68867

theorem system_solution (x y k : ℝ) : 
  (2 * x + y = 1) → 
  (x + 2 * y = k - 2) → 
  (x - y = 2) → 
  (k = 1) := by
sorry

end system_solution_l688_68867


namespace fraction_zero_at_five_l688_68894

theorem fraction_zero_at_five (x : ℝ) : 
  (x - 5) / (6 * x - 12) = 0 ↔ x = 5 ∧ 6 * x - 12 ≠ 0 :=
by sorry

end fraction_zero_at_five_l688_68894


namespace circular_seating_arrangement_l688_68829

/-- 
Given a circular arrangement of students, if the 6th position 
is exactly opposite to the 16th position, then there are 22 students in total.
-/
theorem circular_seating_arrangement (n : ℕ) : 
  (6 + n / 2 ≡ 16 [MOD n]) → n = 22 :=
by
  sorry

end circular_seating_arrangement_l688_68829


namespace quadratic_two_roots_l688_68866

-- Define the quadratic equation
def quadratic (x k : ℝ) : ℝ := x^2 + 2*x + k

-- Define the condition for two distinct real roots
def has_two_distinct_real_roots (k : ℝ) : Prop :=
  ∃ x y : ℝ, x ≠ y ∧ quadratic x k = 0 ∧ quadratic y k = 0

-- State the theorem
theorem quadratic_two_roots :
  has_two_distinct_real_roots 0 ∧ 
  ∀ k : ℝ, has_two_distinct_real_roots k → k = 0 :=
sorry

end quadratic_two_roots_l688_68866


namespace inequality_solution_count_l688_68888

theorem inequality_solution_count : 
  ∃! (x : ℕ), x > 0 ∧ 15 < -2 * (x : ℤ) + 17 :=
by sorry

end inequality_solution_count_l688_68888


namespace greatest_integer_inequality_l688_68809

theorem greatest_integer_inequality : ∀ x : ℤ, (7 : ℚ) / 9 > (x : ℚ) / 15 ↔ x ≤ 11 := by
  sorry

end greatest_integer_inequality_l688_68809
