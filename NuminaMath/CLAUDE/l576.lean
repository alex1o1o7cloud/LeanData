import Mathlib

namespace NUMINAMATH_CALUDE_wallpaper_coverage_l576_57681

/-- Given information about wallpaper coverage, proves the area covered by three layers. -/
theorem wallpaper_coverage (total_wallpaper : ℝ) (total_wall : ℝ) (two_layer : ℝ)
  (h1 : total_wallpaper = 300)
  (h2 : total_wall = 180)
  (h3 : two_layer = 30) :
  ∃ (one_layer three_layer : ℝ),
    one_layer + 2 * two_layer + 3 * three_layer = total_wallpaper ∧
    one_layer + two_layer + three_layer = total_wall ∧
    three_layer = 90 :=
by sorry

end NUMINAMATH_CALUDE_wallpaper_coverage_l576_57681


namespace NUMINAMATH_CALUDE_prescription_final_cost_l576_57693

/-- Calculates the final cost of a prescription after cashback and rebate --/
theorem prescription_final_cost
  (original_cost : ℝ)
  (cashback_percentage : ℝ)
  (rebate : ℝ)
  (h1 : original_cost = 150)
  (h2 : cashback_percentage = 0.1)
  (h3 : rebate = 25) :
  original_cost - (cashback_percentage * original_cost + rebate) = 110 :=
by sorry

end NUMINAMATH_CALUDE_prescription_final_cost_l576_57693


namespace NUMINAMATH_CALUDE_fruit_bowl_problem_l576_57695

/-- Represents the number of fruits in a bowl -/
structure FruitBowl where
  apples : ℕ
  pears : ℕ
  bananas : ℕ

/-- Defines the conditions of the fruit bowl problem -/
def validFruitBowl (bowl : FruitBowl) : Prop :=
  bowl.pears = bowl.apples + 2 ∧
  bowl.bananas = bowl.pears + 3 ∧
  bowl.apples + bowl.pears + bowl.bananas = 19

/-- Theorem stating that a valid fruit bowl contains 9 bananas -/
theorem fruit_bowl_problem (bowl : FruitBowl) : 
  validFruitBowl bowl → bowl.bananas = 9 := by
  sorry


end NUMINAMATH_CALUDE_fruit_bowl_problem_l576_57695


namespace NUMINAMATH_CALUDE_cricket_bat_cost_price_l576_57636

theorem cricket_bat_cost_price (profit_A_to_B : Real) (profit_B_to_C : Real) (price_C : Real) :
  profit_A_to_B = 0.20 →
  profit_B_to_C = 0.25 →
  price_C = 231 →
  ∃ (cost_price_A : Real), cost_price_A = 154 ∧
    price_C = cost_price_A * (1 + profit_A_to_B) * (1 + profit_B_to_C) := by
  sorry

end NUMINAMATH_CALUDE_cricket_bat_cost_price_l576_57636


namespace NUMINAMATH_CALUDE_lillian_sugar_at_home_l576_57631

/-- The number of cups of sugar needed for cupcake batter -/
def batterSugar (cupcakes : ℕ) : ℕ := cupcakes / 12

/-- The number of cups of sugar needed for cupcake frosting -/
def frostingSugar (cupcakes : ℕ) : ℕ := 2 * (cupcakes / 12)

/-- The total number of cups of sugar needed for cupcakes -/
def totalSugarNeeded (cupcakes : ℕ) : ℕ := batterSugar cupcakes + frostingSugar cupcakes

theorem lillian_sugar_at_home (cupcakes sugarBought sugarAtHome : ℕ) :
  cupcakes = 60 →
  sugarBought = 12 →
  totalSugarNeeded cupcakes = sugarBought + sugarAtHome →
  sugarAtHome = 3 := by
  sorry

#check lillian_sugar_at_home

end NUMINAMATH_CALUDE_lillian_sugar_at_home_l576_57631


namespace NUMINAMATH_CALUDE_average_of_three_l576_57611

theorem average_of_three (y : ℝ) : (15 + 25 + y) / 3 = 20 → y = 20 := by
  sorry

end NUMINAMATH_CALUDE_average_of_three_l576_57611


namespace NUMINAMATH_CALUDE_blue_balls_count_l576_57641

theorem blue_balls_count (total : ℕ) (removed : ℕ) (prob : ℚ) (initial : ℕ) : 
  total = 25 →
  removed = 5 →
  prob = 1/5 →
  (initial - removed : ℚ) / (total - removed : ℚ) = prob →
  initial = 9 := by sorry

end NUMINAMATH_CALUDE_blue_balls_count_l576_57641


namespace NUMINAMATH_CALUDE_nine_integer_chords_l576_57691

/-- Represents a circle with a given radius and a point P at a given distance from the center -/
structure CircleWithPoint where
  radius : ℝ
  distanceToP : ℝ

/-- Counts the number of integer-length chords passing through P -/
def countIntegerChords (c : CircleWithPoint) : ℕ :=
  sorry

/-- The main theorem to be proved -/
theorem nine_integer_chords (c : CircleWithPoint) 
  (h1 : c.radius = 20)
  (h2 : c.distanceToP = 12) : 
  countIntegerChords c = 9 :=
sorry

end NUMINAMATH_CALUDE_nine_integer_chords_l576_57691


namespace NUMINAMATH_CALUDE_quadratic_root_l576_57623

theorem quadratic_root (m : ℝ) : 
  (∃ x : ℝ, 2 * x^2 - (m + 3) * x + m = 0 ∧ x = 1) → 
  (∃ y : ℝ, 2 * y^2 - (m + 3) * y + m = 0 ∧ y = (-m - 5) / 2) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_root_l576_57623


namespace NUMINAMATH_CALUDE_student_tickets_sold_l576_57624

theorem student_tickets_sold (student_price non_student_price total_tickets total_revenue : ℕ)
  (h1 : student_price = 9)
  (h2 : non_student_price = 11)
  (h3 : total_tickets = 2000)
  (h4 : total_revenue = 20960) :
  ∃ student_tickets : ℕ,
    student_tickets * student_price + (total_tickets - student_tickets) * non_student_price = total_revenue ∧
    student_tickets = 520 :=
by sorry

end NUMINAMATH_CALUDE_student_tickets_sold_l576_57624


namespace NUMINAMATH_CALUDE_digit2015_is_8_l576_57642

/-- The function that generates the list of digits of positive even numbers -/
def evenNumberDigits : ℕ → List ℕ := sorry

/-- The 2015th digit in the list of digits of positive even numbers -/
def digit2015 : ℕ := (evenNumberDigits 0).nthLe 2014 sorry

/-- Theorem stating that the 2015th digit is 8 -/
theorem digit2015_is_8 : digit2015 = 8 := by sorry

end NUMINAMATH_CALUDE_digit2015_is_8_l576_57642


namespace NUMINAMATH_CALUDE_certain_number_proof_l576_57685

theorem certain_number_proof : ∃ x : ℝ, 45 * x = 0.6 * 900 ∧ x = 12 := by
  sorry

end NUMINAMATH_CALUDE_certain_number_proof_l576_57685


namespace NUMINAMATH_CALUDE_markup_calculation_l576_57648

-- Define the markup percentage
def markup_percentage : ℝ := 50

-- Define the discount percentage
def discount_percentage : ℝ := 20

-- Define the profit percentage
def profit_percentage : ℝ := 20

-- Define the relationship between cost price, marked price, and selling price
def price_relationship (cost_price marked_price selling_price : ℝ) : Prop :=
  selling_price = marked_price * (1 - discount_percentage / 100) ∧
  selling_price = cost_price * (1 + profit_percentage / 100)

-- Theorem statement
theorem markup_calculation :
  ∀ (cost_price marked_price selling_price : ℝ),
  cost_price > 0 →
  price_relationship cost_price marked_price selling_price →
  (marked_price - cost_price) / cost_price * 100 = markup_percentage :=
by sorry

end NUMINAMATH_CALUDE_markup_calculation_l576_57648


namespace NUMINAMATH_CALUDE_root_properties_l576_57697

-- Define the polynomial f
def f (x : ℝ) : ℝ := x^20 - 123*x^10 + 1

-- Define the polynomial g
def g (x : ℝ) : ℝ := x^4 + 3*x^3 + 4*x^2 + 2*x + 1

theorem root_properties (a β : ℝ) : 
  (f a = 0 → f (-a) = 0 ∧ f (1/a) = 0 ∧ f (-1/a) = 0) ∧
  (g β = 0 → g (-β) ≠ 0 ∧ g (1/β) ≠ 0 ∧ g (-1/β) ≠ 0) :=
by sorry

end NUMINAMATH_CALUDE_root_properties_l576_57697


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_l576_57673

theorem sufficient_not_necessary : 
  (∀ x : ℝ, (|x| < 2 → x^2 - x - 6 < 0)) ∧ 
  (∃ x : ℝ, x^2 - x - 6 < 0 ∧ ¬(|x| < 2)) :=
by sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_l576_57673


namespace NUMINAMATH_CALUDE_expression_evaluation_l576_57616

theorem expression_evaluation :
  let x : ℚ := -2
  let y : ℚ := 1/2
  (x + 2*y)^2 - (x + y)*(x - y) = -11/4 := by
sorry

end NUMINAMATH_CALUDE_expression_evaluation_l576_57616


namespace NUMINAMATH_CALUDE_wristbands_per_spectator_l576_57682

theorem wristbands_per_spectator (total_wristbands : ℕ) (total_spectators : ℕ) 
  (h1 : total_wristbands = 290) 
  (h2 : total_spectators = 145) :
  total_wristbands / total_spectators = 2 := by
  sorry

end NUMINAMATH_CALUDE_wristbands_per_spectator_l576_57682


namespace NUMINAMATH_CALUDE_cubic_difference_equality_l576_57628

theorem cubic_difference_equality (x y : ℝ) : 
  x^2 = 7 + 4 * Real.sqrt 3 ∧ 
  y^2 = 7 - 4 * Real.sqrt 3 → 
  x^3 / y - y^3 / x = 112 * Real.sqrt 3 := by
sorry

end NUMINAMATH_CALUDE_cubic_difference_equality_l576_57628


namespace NUMINAMATH_CALUDE_angle_from_terminal_point_l576_57650

theorem angle_from_terminal_point (θ : Real) :
  θ ∈ Set.Icc 0 (2 * Real.pi) →
  (Real.sin θ = Real.sin (3 * Real.pi / 4) ∧ Real.cos θ = Real.cos (3 * Real.pi / 4)) →
  θ = 7 * Real.pi / 4 := by
  sorry

end NUMINAMATH_CALUDE_angle_from_terminal_point_l576_57650


namespace NUMINAMATH_CALUDE_concert_revenue_calculation_l576_57663

/-- Calculates the total revenue from concert ticket sales given specific conditions --/
theorem concert_revenue_calculation :
  let ticket_price : ℕ := 20
  let first_group_size : ℕ := 10
  let second_group_size : ℕ := 20
  let total_attendees : ℕ := 50
  let first_discount : ℚ := 40 / 100
  let second_discount : ℚ := 15 / 100

  let first_group_revenue := first_group_size * (ticket_price * (1 - first_discount))
  let second_group_revenue := second_group_size * (ticket_price * (1 - second_discount))
  let remaining_group_size := total_attendees - first_group_size - second_group_size
  let remaining_group_revenue := remaining_group_size * ticket_price

  let total_revenue := first_group_revenue + second_group_revenue + remaining_group_revenue

  total_revenue = 860 := by sorry

end NUMINAMATH_CALUDE_concert_revenue_calculation_l576_57663


namespace NUMINAMATH_CALUDE_tangent_point_at_negative_one_slope_l576_57638

-- Define the curve
def f (x : ℝ) : ℝ := x^3 - 3*x^2 + 2*x

-- Define the derivative of the curve
def f' (x : ℝ) : ℝ := 3*x^2 - 6*x + 2

theorem tangent_point_at_negative_one_slope :
  ∃ (x : ℝ), f' x = -1 ∧ (x, f x) = (1, 0) := by
  sorry

end NUMINAMATH_CALUDE_tangent_point_at_negative_one_slope_l576_57638


namespace NUMINAMATH_CALUDE_diane_allison_age_ratio_l576_57694

/-- Proves that the ratio of Diane's age to Allison's age when Diane turns 30 is 2:1 -/
theorem diane_allison_age_ratio :
  -- Diane's current age
  ∀ (diane_current_age : ℕ),
  -- Sum of Alex's and Allison's current ages
  ∀ (alex_allison_sum : ℕ),
  -- Diane's age when she turns 30
  ∀ (diane_future_age : ℕ),
  -- Alex's age when Diane turns 30
  ∀ (alex_future_age : ℕ),
  -- Allison's age when Diane turns 30
  ∀ (allison_future_age : ℕ),
  -- Conditions
  diane_current_age = 16 →
  alex_allison_sum = 47 →
  diane_future_age = 30 →
  alex_future_age = 2 * diane_future_age →
  -- Conclusion
  (diane_future_age : ℚ) / (allison_future_age : ℚ) = 2 := by
  sorry

end NUMINAMATH_CALUDE_diane_allison_age_ratio_l576_57694


namespace NUMINAMATH_CALUDE_root_sum_squares_l576_57622

theorem root_sum_squares (p q r : ℝ) : 
  (p^3 - 24*p^2 + 50*p - 8 = 0) →
  (q^3 - 24*q^2 + 50*q - 8 = 0) →
  (r^3 - 24*r^2 + 50*r - 8 = 0) →
  (p+q)^2 + (q+r)^2 + (r+p)^2 = 1052 := by
sorry

end NUMINAMATH_CALUDE_root_sum_squares_l576_57622


namespace NUMINAMATH_CALUDE_exists_32_chinese_l576_57683

/-- Represents the seating arrangement of businessmen at a round table. -/
structure Seating :=
  (japanese : ℕ)
  (korean : ℕ)
  (chinese : ℕ)
  (total : ℕ)
  (total_eq : japanese + korean + chinese = total)
  (japanese_positive : japanese > 0)

/-- The condition that between any two nearest Japanese, there are exactly as many Chinese as Koreans. -/
def equal_distribution (s : Seating) : Prop :=
  ∃ k : ℕ, s.chinese = k * s.japanese ∧ s.korean = k * s.japanese

/-- The main theorem stating that it's possible to have 32 Chinese in a valid seating arrangement. -/
theorem exists_32_chinese : 
  ∃ s : Seating, s.total = 50 ∧ equal_distribution s ∧ s.chinese = 32 :=
sorry


end NUMINAMATH_CALUDE_exists_32_chinese_l576_57683


namespace NUMINAMATH_CALUDE_min_value_theorem_l576_57629

theorem min_value_theorem (x y : ℝ) (hx : x > 0) (hy : y > 0) 
  (h : 1 / (x + 3) + 1 / (y + 3) = 1 / 4) : 
  ∀ a b : ℝ, a > 0 → b > 0 → 1 / (a + 3) + 1 / (b + 3) = 1 / 4 → 3*x + 4*y ≤ 3*a + 4*b :=
by sorry

end NUMINAMATH_CALUDE_min_value_theorem_l576_57629


namespace NUMINAMATH_CALUDE_optimal_move_is_six_l576_57602

/-- Represents the state of a number in the game -/
inductive NumberState
| Unmarked
| Marked
| Blocked

/-- Represents the game state -/
structure GameState where
  numbers : Fin 17 → NumberState

/-- Checks if a move is valid according to the game rules -/
def isValidMove (state : GameState) (n : Fin 17) : Prop :=
  state.numbers n = NumberState.Unmarked ∧
  ∀ m : Fin 17, state.numbers m = NumberState.Marked →
    n.val ≠ 2 * m.val ∧ n.val ≠ m.val / 2

/-- Applies a move to the game state -/
def applyMove (state : GameState) (n : Fin 17) : GameState :=
  { numbers := λ m =>
      if m = n then NumberState.Marked
      else if m.val = 2 * n.val ∨ 2 * m.val = n.val then NumberState.Blocked
      else state.numbers m }

/-- Checks if the game is over (no more valid moves) -/
def isGameOver (state : GameState) : Prop :=
  ∀ n : Fin 17, ¬isValidMove state n

/-- Defines the initial game state after A marks 8 -/
def initialState : GameState :=
  { numbers := λ n => if n.val = 8 then NumberState.Marked else NumberState.Unmarked }

/-- Theorem: B's optimal move is to mark 6 -/
theorem optimal_move_is_six :
  ∃ (strategy : GameState → Fin 17),
    (∀ state : GameState, isValidMove state (strategy state)) ∧
    (∀ (state : GameState) (n : Fin 17),
      isValidMove state n →
      isGameOver (applyMove (applyMove state (strategy state)) n)) ∧
    strategy initialState = ⟨6, by norm_num⟩ := by
  sorry


end NUMINAMATH_CALUDE_optimal_move_is_six_l576_57602


namespace NUMINAMATH_CALUDE_f_derivative_at_2_l576_57632

def f (x : ℝ) : ℝ := (x - 1) * (x - 2) * (x - 3) * (x - 4) * (x - 5) * (x - 6) * (x - 7) * (x - 8) * (x - 9) * (x - 10)

theorem f_derivative_at_2 : 
  deriv f 2 = 8 * 7 * 6 * 5 * 4 * 3 * 2 * 1 := by
  sorry

end NUMINAMATH_CALUDE_f_derivative_at_2_l576_57632


namespace NUMINAMATH_CALUDE_hexagon_ratio_is_two_l576_57617

/-- Represents a hexagon with specific properties -/
structure Hexagon where
  /-- Total number of unit squares in the hexagon -/
  total_squares : ℕ
  /-- Number of unit squares above the diagonal PQ -/
  squares_above_pq : ℕ
  /-- Base length of the triangle above PQ -/
  triangle_base : ℝ
  /-- Total length of XQ + QY -/
  xq_plus_qy : ℝ
  /-- Condition: The area above PQ is half of the total area -/
  area_condition : squares_above_pq + (triangle_base * triangle_base / 4) = (total_squares + triangle_base * triangle_base / 4) / 2

/-- Theorem: For a hexagon with the given properties, XQ/QY = 2 -/
theorem hexagon_ratio_is_two (h : Hexagon) (h_total : h.total_squares = 8) 
  (h_above : h.squares_above_pq = 3) (h_base : h.triangle_base = 4) (h_xq_qy : h.xq_plus_qy = 4) : 
  ∃ (xq qy : ℝ), xq + qy = h.xq_plus_qy ∧ xq / qy = 2 := by
  sorry

end NUMINAMATH_CALUDE_hexagon_ratio_is_two_l576_57617


namespace NUMINAMATH_CALUDE_shaded_area_between_circles_l576_57640

theorem shaded_area_between_circles (r₁ r₂ : ℝ) (h₁ : r₁ = 4) (h₂ : r₂ = 8) : 
  π * r₂^2 - π * r₁^2 = 48 * π := by
  sorry

end NUMINAMATH_CALUDE_shaded_area_between_circles_l576_57640


namespace NUMINAMATH_CALUDE_yellow_balls_count_l576_57613

theorem yellow_balls_count (white_balls : ℕ) (total_balls : ℕ) 
  (h1 : white_balls = 4)
  (h2 : (white_balls : ℚ) / total_balls = 2 / 3) :
  total_balls - white_balls = 2 := by
  sorry

end NUMINAMATH_CALUDE_yellow_balls_count_l576_57613


namespace NUMINAMATH_CALUDE_quadratic_equation_solution_l576_57601

theorem quadratic_equation_solution : ∃ x₁ x₂ : ℝ, 
  (x₁^2 - x₁ - 2 = 0) ∧ (x₂^2 - x₂ - 2 = 0) ∧ (x₁ = 2) ∧ (x₂ = -1) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equation_solution_l576_57601


namespace NUMINAMATH_CALUDE_workshop_workers_count_l576_57651

/-- Given a workshop with workers including technicians and non-technicians,
    prove that the total number of workers is 22 under the following conditions:
    - The average salary of all workers is 850
    - There are 7 technicians with an average salary of 1000
    - The average salary of non-technicians is 780 -/
theorem workshop_workers_count :
  ∀ (W : ℕ) (avg_salary tech_salary nontech_salary : ℚ),
    avg_salary = 850 →
    tech_salary = 1000 →
    nontech_salary = 780 →
    (W : ℚ) * avg_salary = 7 * tech_salary + (W - 7 : ℚ) * nontech_salary →
    W = 22 := by
  sorry


end NUMINAMATH_CALUDE_workshop_workers_count_l576_57651


namespace NUMINAMATH_CALUDE_range_of_even_quadratic_function_l576_57655

-- Define the function f
def f (a b x : ℝ) : ℝ := x^2 - 2*a*x + b

-- Define the property of being even
def isEven (f : ℝ → ℝ) : Prop := ∀ x, f x = f (-x)

-- Define the domain of the function
def domain (b : ℝ) : Set ℝ := {x | -2*b ≤ x ∧ x ≤ 3*b - 1}

-- State the theorem
theorem range_of_even_quadratic_function (a b : ℝ) :
  (∀ x ∈ domain b, f a b x = f a b (-x)) →
  {y | ∃ x ∈ domain b, f a b x = y} = {y | 1 ≤ y ∧ y ≤ 5} :=
sorry

end NUMINAMATH_CALUDE_range_of_even_quadratic_function_l576_57655


namespace NUMINAMATH_CALUDE_l₂_parallel_and_through_A_B_symmetric_to_A_l576_57679

-- Define the line l₁
def l₁ (x y : ℝ) : Prop := 2 * x + 4 * y - 1 = 0

-- Define point A
def A : ℝ × ℝ := (3, 0)

-- Define the parallel line l₂ passing through A
def l₂ (x y : ℝ) : Prop := x + 2 * y - 3 = 0

-- Define point B
def B : ℝ × ℝ := (2, -2)

-- Theorem 1: l₂ is parallel to l₁ and passes through A
theorem l₂_parallel_and_through_A :
  (∀ x y : ℝ, l₂ x y ↔ ∃ k : ℝ, k ≠ 0 ∧ 2 * x + 4 * y - 1 = k * (2 * A.1 + 4 * A.2 - 1)) ∧
  l₂ A.1 A.2 :=
sorry

-- Theorem 2: B is symmetric to A with respect to l₁
theorem B_symmetric_to_A :
  let midpoint := ((A.1 + B.1) / 2, (A.2 + B.2) / 2)
  l₁ midpoint.1 midpoint.2 ∧
  (B.2 - A.2) / (B.1 - A.1) = - (1 / (2 / 4)) :=
sorry

end NUMINAMATH_CALUDE_l₂_parallel_and_through_A_B_symmetric_to_A_l576_57679


namespace NUMINAMATH_CALUDE_music_festival_group_count_l576_57684

/-- The number of ways to select k items from n items -/
def binomial (n k : ℕ) : ℕ := sorry

/-- The number of ways to select a group for the music festival -/
def musicFestivalGroupWays : ℕ :=
  binomial 6 2 * binomial 4 2 * binomial 4 1

theorem music_festival_group_count :
  musicFestivalGroupWays = 360 := by sorry

end NUMINAMATH_CALUDE_music_festival_group_count_l576_57684


namespace NUMINAMATH_CALUDE_rectangle_segment_relation_l576_57678

-- Define the points and segments
variable (A B C D E F : EuclideanPlane) (BE CD AD AC BC : ℝ)

-- Define the conditions
variable (h1 : IsRectangle C D E F)
variable (h2 : A ∈ SegmentOpen E D)
variable (h3 : B ∈ Line E F)
variable (h4 : B ∈ PerpendicularLine A C C)

-- State the theorem
theorem rectangle_segment_relation :
  BE = CD + (AD / AC) * BC :=
sorry

end NUMINAMATH_CALUDE_rectangle_segment_relation_l576_57678


namespace NUMINAMATH_CALUDE_race_symmetry_l576_57672

/-- Represents a car in the race -/
structure Car where
  speed : ℝ
  direction : Bool -- true for clockwise, false for counterclockwise

/-- Represents the race scenario -/
structure RaceScenario where
  A : Car
  B : Car
  C : Car
  D : Car
  track_length : ℝ
  first_AC_meet_time : ℝ
  first_BD_meet_time : ℝ
  first_AB_meet_time : ℝ

/-- The main theorem statement -/
theorem race_symmetry (race : RaceScenario) :
  race.A.direction = true ∧
  race.B.direction = true ∧
  race.C.direction = false ∧
  race.D.direction = false ∧
  race.A.speed ≠ race.B.speed ∧
  race.A.speed ≠ race.C.speed ∧
  race.A.speed ≠ race.D.speed ∧
  race.B.speed ≠ race.C.speed ∧
  race.B.speed ≠ race.D.speed ∧
  race.C.speed ≠ race.D.speed ∧
  race.first_AC_meet_time = 7 ∧
  race.first_BD_meet_time = 7 ∧
  race.first_AB_meet_time = 53 →
  ∃ (first_CD_meet_time : ℝ), first_CD_meet_time = race.first_AB_meet_time :=
by
  sorry

end NUMINAMATH_CALUDE_race_symmetry_l576_57672


namespace NUMINAMATH_CALUDE_valid_documents_l576_57615

theorem valid_documents (total_papers : ℕ) (invalid_percentage : ℚ) 
  (h1 : total_papers = 400)
  (h2 : invalid_percentage = 40 / 100) :
  (total_papers : ℚ) * (1 - invalid_percentage) = 240 := by
  sorry

end NUMINAMATH_CALUDE_valid_documents_l576_57615


namespace NUMINAMATH_CALUDE_smallest_gcd_of_multiples_l576_57692

theorem smallest_gcd_of_multiples (a b : ℕ+) (h : Nat.gcd a b = 18) :
  ∃ (m n : ℕ+), 12 * m = 12 * a ∧ 20 * n = 20 * b ∧ 
    Nat.gcd (12 * m) (20 * n) = 72 ∧
    ∀ (x y : ℕ+), 12 * x = 12 * a → 20 * y = 20 * b → 
      Nat.gcd (12 * x) (20 * y) ≥ 72 :=
by sorry

end NUMINAMATH_CALUDE_smallest_gcd_of_multiples_l576_57692


namespace NUMINAMATH_CALUDE_remaining_flavors_to_try_l576_57608

def ice_cream_flavors (total : ℕ) (tried_two_years_ago : ℕ) (tried_last_year : ℕ) : Prop :=
  tried_two_years_ago = total / 4 ∧
  tried_last_year = 2 * tried_two_years_ago ∧
  tried_two_years_ago + tried_last_year + 25 = total

theorem remaining_flavors_to_try
  (total : ℕ)
  (tried_two_years_ago : ℕ)
  (tried_last_year : ℕ)
  (h : ice_cream_flavors total tried_two_years_ago tried_last_year)
  (h_total : total = 100) :
  25 = total - (tried_two_years_ago + tried_last_year) :=
by
  sorry

end NUMINAMATH_CALUDE_remaining_flavors_to_try_l576_57608


namespace NUMINAMATH_CALUDE_room_population_lower_limit_l576_57680

theorem room_population_lower_limit :
  ∀ (P : ℕ),
  (P < 100) →
  ((3 : ℚ) / 8 * P = 36) →
  (∃ (n : ℕ), (5 : ℚ) / 12 * P = n) →
  P ≥ 96 :=
by
  sorry

end NUMINAMATH_CALUDE_room_population_lower_limit_l576_57680


namespace NUMINAMATH_CALUDE_fraction_value_given_condition_l576_57647

theorem fraction_value_given_condition (a b : ℝ) 
  (h : |a + 2| + Real.sqrt (b - 4) = 0) : a^2 / b = 1 := by
  sorry

end NUMINAMATH_CALUDE_fraction_value_given_condition_l576_57647


namespace NUMINAMATH_CALUDE_seven_successes_probability_l576_57627

/-- The number of trials -/
def n : ℕ := 7

/-- The probability of success in each trial -/
def p : ℚ := 2/7

/-- The number of successes -/
def k : ℕ := 7

/-- The probability of k successes in n Bernoulli trials with probability p -/
def bernoulli_probability (n k : ℕ) (p : ℚ) : ℚ :=
  (n.choose k) * p^k * (1 - p)^(n - k)

/-- Theorem stating the probability of 7 successes in 7 Bernoulli trials with p = 2/7 -/
theorem seven_successes_probability : 
  bernoulli_probability n k p = 128/823543 := by
  sorry

end NUMINAMATH_CALUDE_seven_successes_probability_l576_57627


namespace NUMINAMATH_CALUDE_xena_head_start_l576_57690

/-- Xena's running speed in feet per second -/
def xena_speed : ℝ := 15

/-- Dragon's flying speed in feet per second -/
def dragon_speed : ℝ := 30

/-- Time Xena has to reach the cave in seconds -/
def time_to_cave : ℝ := 32

/-- Minimum safe distance between Xena and the dragon in feet -/
def safe_distance : ℝ := 120

/-- Theorem stating Xena's head start distance -/
theorem xena_head_start : 
  xena_speed * time_to_cave + 360 = dragon_speed * time_to_cave - safe_distance := by
  sorry

end NUMINAMATH_CALUDE_xena_head_start_l576_57690


namespace NUMINAMATH_CALUDE_ring_with_finite_zero_divisors_is_finite_l576_57687

/-- A ring with at least one non-zero zero divisor and finitely many zero divisors is finite. -/
theorem ring_with_finite_zero_divisors_is_finite (R : Type*) [Ring R]
  (h1 : ∃ (x y : R), x ≠ 0 ∧ y ≠ 0 ∧ x * y = 0)
  (h2 : Set.Finite {x : R | ∃ y, y ≠ 0 ∧ x * y = 0 ∨ y * x = 0}) :
  Set.Finite (Set.univ : Set R) := by
  sorry

end NUMINAMATH_CALUDE_ring_with_finite_zero_divisors_is_finite_l576_57687


namespace NUMINAMATH_CALUDE_isosceles_triangle_m_condition_l576_57689

/-- Represents an isosceles triangle ABC with side BC of length 8 and sides AB and AC as roots of x^2 - 10x + m = 0 --/
structure IsoscelesTriangle where
  m : ℝ
  ab_ac_roots : ∀ x : ℝ, x^2 - 10*x + m = 0 → (x = ab ∨ x = ac)
  isosceles : ab = ac
  bc_length : bc = 8

/-- The value of m in an isosceles triangle satisfies one of two conditions --/
theorem isosceles_triangle_m_condition (t : IsoscelesTriangle) :
  (∃ x : ℝ, x^2 - 10*x + t.m = 0 ∧ (∀ y : ℝ, y^2 - 10*y + t.m = 0 → y = x)) ∨
  (8^2 - 10*8 + t.m = 0) :=
sorry

end NUMINAMATH_CALUDE_isosceles_triangle_m_condition_l576_57689


namespace NUMINAMATH_CALUDE_trees_in_yard_l576_57639

/-- Calculates the number of trees in a yard given the yard length and tree spacing -/
def num_trees (yard_length : ℕ) (tree_spacing : ℕ) : ℕ :=
  (yard_length / tree_spacing) + 1

/-- Theorem stating that the number of trees in a 375-meter yard with 15-meter spacing is 26 -/
theorem trees_in_yard : num_trees 375 15 = 26 := by
  sorry

end NUMINAMATH_CALUDE_trees_in_yard_l576_57639


namespace NUMINAMATH_CALUDE_frog_probability_l576_57654

/-- Represents the probability of ending on a vertical side from a given position -/
def P (x y : ℝ) : ℝ := sorry

/-- The square's dimensions -/
def squareSize : ℝ := 6

theorem frog_probability :
  /- Starting position -/
  let startX : ℝ := 2
  let startY : ℝ := 2

  /- Conditions -/
  (∀ x y, 0 ≤ x ∧ x ≤ squareSize ∧ 0 ≤ y ∧ y ≤ squareSize →
    P x y = (P (x-1) y + P (x+1) y + P x (y-1) + P x (y+1)) / 4) →
  (∀ y, 0 ≤ y ∧ y ≤ squareSize → P 0 y = 1 ∧ P squareSize y = 1) →
  (∀ x, 0 ≤ x ∧ x ≤ squareSize → P x 0 = 0 ∧ P x squareSize = 0) →
  (∀ x y, P x y = P (squareSize - x) (squareSize - y)) →

  /- Conclusion -/
  P startX startY = 2/3 := by sorry

end NUMINAMATH_CALUDE_frog_probability_l576_57654


namespace NUMINAMATH_CALUDE_pipe_cut_theorem_l576_57606

theorem pipe_cut_theorem (total_length : ℝ) (difference : ℝ) (shorter_piece : ℝ) :
  total_length = 68 →
  difference = 12 →
  total_length = shorter_piece + (shorter_piece + difference) →
  shorter_piece = 28 := by
sorry

end NUMINAMATH_CALUDE_pipe_cut_theorem_l576_57606


namespace NUMINAMATH_CALUDE_union_equals_A_l576_57630

def A : Set ℝ := {x | x^2 + x - 2 = 0}
def B (m : ℝ) : Set ℝ := {x | m * x + 1 = 0}

theorem union_equals_A (m : ℝ) : A ∪ B m = A → m = 0 ∨ m = -1 ∨ m = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_union_equals_A_l576_57630


namespace NUMINAMATH_CALUDE_complex_product_with_i_l576_57674

theorem complex_product_with_i : (Complex.I * (-1 + 3 * Complex.I)) = (-3 - Complex.I) := by
  sorry

end NUMINAMATH_CALUDE_complex_product_with_i_l576_57674


namespace NUMINAMATH_CALUDE_rectangle_width_l576_57665

theorem rectangle_width (square_area : ℝ) (rectangle_length : ℝ) (square_perimeter : ℝ) :
  square_area = 5 * (rectangle_length * 10) ∧
  rectangle_length = 50 ∧
  square_perimeter = 200 ∧
  square_area = (square_perimeter / 4) ^ 2 →
  10 = 10 := by
sorry

end NUMINAMATH_CALUDE_rectangle_width_l576_57665


namespace NUMINAMATH_CALUDE_max_silver_tokens_l576_57670

/-- Represents the state of tokens -/
structure TokenState where
  red : ℕ
  blue : ℕ
  silver : ℕ

/-- Represents the exchange rules -/
def exchange1 (state : TokenState) : TokenState :=
  { red := state.red - 3, blue := state.blue + 2, silver := state.silver + 1 }

def exchange2 (state : TokenState) : TokenState :=
  { red := state.red + 1, blue := state.blue - 3, silver := state.silver + 1 }

/-- Checks if an exchange is possible -/
def canExchange (state : TokenState) : Bool :=
  state.red ≥ 3 ∨ state.blue ≥ 3

/-- Theorem: The maximum number of silver tokens obtainable is 95 -/
theorem max_silver_tokens :
  ∃ (finalState : TokenState),
    finalState.red < 3 ∧
    finalState.blue < 3 ∧
    finalState.silver = 95 ∧
    (∀ (otherState : TokenState),
      otherState.red < 3 →
      otherState.blue < 3 →
      otherState.red + otherState.blue + otherState.silver = 150 →
      otherState.silver ≤ 95) := by
  sorry

#check max_silver_tokens

end NUMINAMATH_CALUDE_max_silver_tokens_l576_57670


namespace NUMINAMATH_CALUDE_events_mutually_exclusive_not_complementary_l576_57660

/-- Represents the number of boys in the group -/
def num_boys : Nat := 3

/-- Represents the number of girls in the group -/
def num_girls : Nat := 2

/-- Represents the number of students to be selected -/
def num_selected : Nat := 2

/-- Event: Exactly 1 boy is selected -/
def event_one_boy (selected : Finset (Fin (num_boys + num_girls))) : Prop :=
  (selected.filter (λ i => i.val < num_boys)).card = 1

/-- Event: Exactly 2 girls are selected -/
def event_two_girls (selected : Finset (Fin (num_boys + num_girls))) : Prop :=
  (selected.filter (λ i => i.val ≥ num_boys)).card = 2

/-- Theorem: The events are mutually exclusive but not complementary -/
theorem events_mutually_exclusive_not_complementary :
  (∀ selected : Finset (Fin (num_boys + num_girls)), selected.card = num_selected →
    ¬(event_one_boy selected ∧ event_two_girls selected)) ∧
  (∃ selected : Finset (Fin (num_boys + num_girls)), selected.card = num_selected →
    ¬event_one_boy selected ∧ ¬event_two_girls selected) :=
sorry

end NUMINAMATH_CALUDE_events_mutually_exclusive_not_complementary_l576_57660


namespace NUMINAMATH_CALUDE_excel_in_both_subjects_l576_57656

theorem excel_in_both_subjects 
  (total_students : ℕ) 
  (excel_chinese : ℕ) 
  (excel_math : ℕ) 
  (h1 : total_students = 45)
  (h2 : excel_chinese = 34)
  (h3 : excel_math = 39)
  (h4 : ∀ s, s ≤ total_students → (s ≤ excel_chinese ∨ s ≤ excel_math)) :
  excel_chinese + excel_math - total_students = 28 := by
  sorry

end NUMINAMATH_CALUDE_excel_in_both_subjects_l576_57656


namespace NUMINAMATH_CALUDE_minimum_buses_needed_minimum_buses_for_field_trip_l576_57635

theorem minimum_buses_needed 
  (total_students : ℕ) 
  (regular_capacity : ℕ) 
  (reduced_capacity : ℕ) 
  (reduced_buses : ℕ) : ℕ :=
  let remaining_students := total_students - (reduced_capacity * reduced_buses)
  let regular_buses_needed := (remaining_students + regular_capacity - 1) / regular_capacity
  regular_buses_needed + reduced_buses

theorem minimum_buses_for_field_trip : 
  minimum_buses_needed 1234 45 30 3 = 29 := by
  sorry

end NUMINAMATH_CALUDE_minimum_buses_needed_minimum_buses_for_field_trip_l576_57635


namespace NUMINAMATH_CALUDE_inequality_system_solution_l576_57675

theorem inequality_system_solution :
  ∀ x : ℝ, (3 * x - 1 > 2 * (x + 1) ∧ (x + 2) / 3 > x - 2) ↔ (3 < x ∧ x < 4) :=
by sorry

end NUMINAMATH_CALUDE_inequality_system_solution_l576_57675


namespace NUMINAMATH_CALUDE_equation_solution_l576_57699

theorem equation_solution (x : ℝ) : (24 / 36 : ℝ) = Real.sqrt (x / 36) → x = 16 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l576_57699


namespace NUMINAMATH_CALUDE_ball_piles_problem_l576_57607

theorem ball_piles_problem (x y z a : ℕ) : 
  x + y + z = 2012 →
  y - a = 17 →
  x - a = 2 * (z - a) →
  z = 665 := by
sorry

end NUMINAMATH_CALUDE_ball_piles_problem_l576_57607


namespace NUMINAMATH_CALUDE_base3_to_base10_conversion_l576_57645

/-- Converts a list of digits in base 3 to a natural number in base 10 -/
def base3ToBase10 (digits : List Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * 3^i) 0

/-- The base 3 representation of the number -/
def base3Digits : List Nat := [1, 2, 0, 2, 1]

theorem base3_to_base10_conversion :
  base3ToBase10 base3Digits = 142 := by
  sorry

end NUMINAMATH_CALUDE_base3_to_base10_conversion_l576_57645


namespace NUMINAMATH_CALUDE_gcf_210_286_l576_57619

theorem gcf_210_286 : Nat.gcd 210 286 = 2 := by
  sorry

end NUMINAMATH_CALUDE_gcf_210_286_l576_57619


namespace NUMINAMATH_CALUDE_product_of_real_parts_l576_57614

-- Define the complex number i
def i : ℂ := Complex.I

-- Define the quadratic equation
def quadratic_equation (z : ℂ) : Prop :=
  z^2 + 2*z = 10 - 2*i

-- Define the roots of the quadratic equation
noncomputable def roots : Set ℂ :=
  {z : ℂ | quadratic_equation z}

-- State the theorem
theorem product_of_real_parts :
  ∃ (z₁ z₂ : ℂ), z₁ ∈ roots ∧ z₂ ∈ roots ∧ 
  (z₁.re * z₂.re : ℝ) = -10.25 :=
sorry

end NUMINAMATH_CALUDE_product_of_real_parts_l576_57614


namespace NUMINAMATH_CALUDE_game_end_conditions_l576_57677

/-- Represents a game board of size n × n with k game pieces -/
structure GameBoard (n : ℕ) (k : ℕ) where
  size : n ≥ 2
  pieces : k ≥ 0

/-- Determines if the game never ends for any initial arrangement -/
def never_ends (n : ℕ) (k : ℕ) : Prop :=
  k > 3 * n^2 - 4 * n

/-- Determines if the game always ends for any initial arrangement -/
def always_ends (n : ℕ) (k : ℕ) : Prop :=
  k < 2 * n^2 - 2 * n

/-- Theorem stating the conditions for the game to never end or always end -/
theorem game_end_conditions (n : ℕ) (k : ℕ) (board : GameBoard n k) :
  (never_ends n k ↔ k > 3 * n^2 - 4 * n) ∧
  (always_ends n k ↔ k < 2 * n^2 - 2 * n) :=
sorry

end NUMINAMATH_CALUDE_game_end_conditions_l576_57677


namespace NUMINAMATH_CALUDE_author_writing_speed_l576_57600

/-- Calculates the average words written per hour given total words, total hours, and break hours. -/
def average_words_per_hour (total_words : ℕ) (total_hours : ℕ) (break_hours : ℕ) : ℚ :=
  total_words / (total_hours - break_hours)

/-- Theorem stating that given the specific conditions, the average words per hour is 550. -/
theorem author_writing_speed :
  average_words_per_hour 55000 120 20 = 550 := by
  sorry

end NUMINAMATH_CALUDE_author_writing_speed_l576_57600


namespace NUMINAMATH_CALUDE_water_speed_l576_57646

/-- The speed of water given a swimmer's speed in still water and time taken to swim against the current -/
theorem water_speed (swim_speed : ℝ) (distance : ℝ) (time : ℝ) (h1 : swim_speed = 6)
  (h2 : distance = 14) (h3 : time = 3.5) :
  ∃ (water_speed : ℝ), water_speed = 2 ∧ distance = (swim_speed - water_speed) * time := by
  sorry

end NUMINAMATH_CALUDE_water_speed_l576_57646


namespace NUMINAMATH_CALUDE_range_of_m_l576_57686

-- Define the sets P and S
def P : Set ℝ := {x | x^2 - 8*x - 20 ≤ 0}
def S (m : ℝ) : Set ℝ := {x | 1 - m ≤ x ∧ x ≤ 1 + m}

-- State the theorem
theorem range_of_m (m : ℝ) (h_nonempty : (S m).Nonempty) 
  (h_condition : ∀ x, x ∉ P → (x ∉ S m → True) ∧ (x ∉ S m → x ∉ P → False)) : 
  m ≥ 9 := by
  sorry

end NUMINAMATH_CALUDE_range_of_m_l576_57686


namespace NUMINAMATH_CALUDE_complex_modulus_problem_l576_57676

theorem complex_modulus_problem (x y : ℝ) (h : Complex.I * Complex.mk x y = Complex.mk 3 4) :
  Complex.abs (Complex.mk x y) = 5 := by
  sorry

end NUMINAMATH_CALUDE_complex_modulus_problem_l576_57676


namespace NUMINAMATH_CALUDE_binomial_9_choose_5_l576_57612

theorem binomial_9_choose_5 : Nat.choose 9 5 = 126 := by sorry

end NUMINAMATH_CALUDE_binomial_9_choose_5_l576_57612


namespace NUMINAMATH_CALUDE_divisibility_by_seven_l576_57698

theorem divisibility_by_seven (A a b : ℕ) : A = 100 * a + b →
  (7 ∣ A ↔ 7 ∣ (2 * a + b)) ∧ (7 ∣ A ↔ 7 ∣ (5 * a - b)) := by
  sorry

end NUMINAMATH_CALUDE_divisibility_by_seven_l576_57698


namespace NUMINAMATH_CALUDE_cube_sum_is_90_l576_57652

-- Define the type for the cube faces
def CubeFaces := Fin 6 → ℕ

-- Define the property of consecutive even numbers
def ConsecutiveEven (faces : CubeFaces) : Prop :=
  ∃ n : ℕ, ∀ i : Fin 6, faces i = 2 * (n + i.val)

-- Define the property of opposite face sums being equal
def OppositeFaceSumsEqual (faces : CubeFaces) : Prop :=
  ∃ s : ℕ, 
    faces 0 + faces 5 + 2 = s ∧
    faces 1 + faces 4 + 2 = s ∧
    faces 2 + faces 3 + 2 = s

-- Theorem statement
theorem cube_sum_is_90 (faces : CubeFaces) 
  (h1 : ConsecutiveEven faces) 
  (h2 : OppositeFaceSumsEqual faces) : 
  (faces 0 + faces 1 + faces 2 + faces 3 + faces 4 + faces 5 = 90) :=
sorry

end NUMINAMATH_CALUDE_cube_sum_is_90_l576_57652


namespace NUMINAMATH_CALUDE_tan_negative_405_degrees_l576_57618

theorem tan_negative_405_degrees : Real.tan ((-405 : ℝ) * π / 180) = -1 := by
  sorry

end NUMINAMATH_CALUDE_tan_negative_405_degrees_l576_57618


namespace NUMINAMATH_CALUDE_lunch_cost_per_person_l576_57696

theorem lunch_cost_per_person (total_price : ℝ) (num_people : ℕ) (gratuity_rate : ℝ) : 
  total_price = 207 ∧ num_people = 15 ∧ gratuity_rate = 0.15 →
  (total_price / (1 + gratuity_rate)) / num_people = 12 := by
sorry

end NUMINAMATH_CALUDE_lunch_cost_per_person_l576_57696


namespace NUMINAMATH_CALUDE_intersection_M_N_l576_57657

-- Define the sets M and N
def M : Set ℝ := {x : ℝ | x^2 < 4}
def N : Set ℝ := {x : ℝ | x * Real.log x > 0}

-- State the theorem
theorem intersection_M_N :
  M ∩ N = Set.Ioo 1 2 := by sorry

end NUMINAMATH_CALUDE_intersection_M_N_l576_57657


namespace NUMINAMATH_CALUDE_point_on_negative_x_axis_l576_57653

/-- Given a point A with coordinates (a+1, a^2-4) that lies on the negative half of the x-axis,
    prove that its coordinates are (-1, 0). -/
theorem point_on_negative_x_axis (a : ℝ) : 
  (a + 1 < 0) ∧ (a^2 - 4 = 0) → (a + 1 = -1 ∧ a^2 - 4 = 0) :=
by sorry

end NUMINAMATH_CALUDE_point_on_negative_x_axis_l576_57653


namespace NUMINAMATH_CALUDE_rectangle_area_l576_57659

/-- The area of a rectangle with sides of length (a - b) and (c + d) is equal to ac + ad - bc - bd. -/
theorem rectangle_area (a b c d : ℝ) : 
  let length := a - b
  let width := c + d
  length * width = a*c + a*d - b*c - b*d := by sorry

end NUMINAMATH_CALUDE_rectangle_area_l576_57659


namespace NUMINAMATH_CALUDE_fraction_simplification_l576_57649

theorem fraction_simplification :
  (3 : ℝ) / (2 * Real.sqrt 50 + 3 * Real.sqrt 8 - Real.sqrt 18) = (3 * Real.sqrt 2) / 26 := by
  sorry

end NUMINAMATH_CALUDE_fraction_simplification_l576_57649


namespace NUMINAMATH_CALUDE_total_laundry_time_is_344_l576_57688

/-- Represents a load of laundry with washing and drying times -/
structure LaundryLoad where
  washTime : ℕ
  dryTime : ℕ

/-- Calculates the total time for a single load of laundry -/
def totalLoadTime (load : LaundryLoad) : ℕ :=
  load.washTime + load.dryTime

/-- Calculates the total time for all loads of laundry -/
def totalLaundryTime (loads : List LaundryLoad) : ℕ :=
  loads.map totalLoadTime |>.sum

/-- Theorem: The total laundry time for the given loads is 344 minutes -/
theorem total_laundry_time_is_344 :
  let whites : LaundryLoad := { washTime := 72, dryTime := 50 }
  let darks : LaundryLoad := { washTime := 58, dryTime := 65 }
  let colors : LaundryLoad := { washTime := 45, dryTime := 54 }
  let allLoads : List LaundryLoad := [whites, darks, colors]
  totalLaundryTime allLoads = 344 := by
  sorry


end NUMINAMATH_CALUDE_total_laundry_time_is_344_l576_57688


namespace NUMINAMATH_CALUDE_distance_between_points_l576_57604

def point1 : ℝ × ℝ := (2, -2)
def point2 : ℝ × ℝ := (8, 9)

theorem distance_between_points : 
  Real.sqrt ((point2.1 - point1.1)^2 + (point2.2 - point1.2)^2) = Real.sqrt 157 := by
  sorry

end NUMINAMATH_CALUDE_distance_between_points_l576_57604


namespace NUMINAMATH_CALUDE_largest_radius_special_circle_l576_57644

/-- A circle with the given properties -/
structure SpecialCircle where
  center : ℝ × ℝ
  radius : ℝ
  contains_points : center.1^2 + (center.2 - 11)^2 = radius^2 ∧
                    center.1^2 + (center.2 + 11)^2 = radius^2
  contains_unit_disk : ∀ (x y : ℝ), x^2 + y^2 < 1 →
                       (x - center.1)^2 + (y - center.2)^2 < radius^2

/-- The theorem stating the largest possible radius -/
theorem largest_radius_special_circle :
  ∃ (c : SpecialCircle), ∀ (c' : SpecialCircle), c'.radius ≤ c.radius ∧ c.radius = Real.sqrt 122 :=
sorry

end NUMINAMATH_CALUDE_largest_radius_special_circle_l576_57644


namespace NUMINAMATH_CALUDE_geometric_sequence_common_ratio_l576_57620

/-- A geometric sequence with a_2 = 8 and a_5 = 64 has a common ratio of 2 -/
theorem geometric_sequence_common_ratio : ∀ (a : ℕ → ℝ),
  (∀ n : ℕ, a (n + 1) = a n * a 1)  -- Definition of geometric sequence
  → a 2 = 8                         -- Given condition
  → a 5 = 64                        -- Given condition
  → a 1 = 2                         -- Common ratio q = a_1
  := by sorry

end NUMINAMATH_CALUDE_geometric_sequence_common_ratio_l576_57620


namespace NUMINAMATH_CALUDE_value_of_a_minus_b_l576_57671

theorem value_of_a_minus_b (a b : ℤ) 
  (eq1 : 3015 * a + 3019 * b = 3023)
  (eq2 : 3017 * a + 3021 * b = 3025) : 
  a - b = -3 := by
  sorry

end NUMINAMATH_CALUDE_value_of_a_minus_b_l576_57671


namespace NUMINAMATH_CALUDE_solve_chocolates_problem_l576_57621

def chocolates_problem (nick_chocolates : ℕ) (alix_multiplier : ℕ) (difference_after : ℕ) : Prop :=
  let initial_alix_chocolates := nick_chocolates * alix_multiplier
  let alix_chocolates_after := nick_chocolates + difference_after
  let chocolates_taken := initial_alix_chocolates - alix_chocolates_after
  chocolates_taken = 5

theorem solve_chocolates_problem :
  chocolates_problem 10 3 15 := by
  sorry

end NUMINAMATH_CALUDE_solve_chocolates_problem_l576_57621


namespace NUMINAMATH_CALUDE_planes_parallel_from_skew_lines_l576_57634

-- Define the types for planes and lines
variable (Plane Line : Type)

-- Define the relations
variable (subset : Line → Plane → Prop)
variable (parallel : Line → Plane → Prop)
variable (plane_parallel : Plane → Plane → Prop)
variable (skew : Line → Line → Prop)

-- State the theorem
theorem planes_parallel_from_skew_lines 
  (α β : Plane) (l m : Line) :
  skew l m →
  subset l α →
  parallel l β →
  subset m β →
  parallel m α →
  plane_parallel α β :=
sorry

end NUMINAMATH_CALUDE_planes_parallel_from_skew_lines_l576_57634


namespace NUMINAMATH_CALUDE_gcd_884_1071_l576_57669

theorem gcd_884_1071 : Nat.gcd 884 1071 = 17 := by
  sorry

end NUMINAMATH_CALUDE_gcd_884_1071_l576_57669


namespace NUMINAMATH_CALUDE_expressions_equality_l576_57658

theorem expressions_equality (a b c : ℝ) : a + b * c = (a + b) * (a + c) ↔ a + b + c = 1 := by
  sorry

end NUMINAMATH_CALUDE_expressions_equality_l576_57658


namespace NUMINAMATH_CALUDE_mean_proportional_problem_l576_57661

theorem mean_proportional_problem (x : ℝ) :
  (156 : ℝ)^2 = 234 * x → x = 104 := by
  sorry

end NUMINAMATH_CALUDE_mean_proportional_problem_l576_57661


namespace NUMINAMATH_CALUDE_store_sales_increase_l576_57666

/-- Proves that if a store's sales increased by 25% to $400 million, then the previous year's sales were $320 million. -/
theorem store_sales_increase (current_sales : ℝ) (increase_percent : ℝ) :
  current_sales = 400 ∧ increase_percent = 0.25 →
  (1 + increase_percent) * (current_sales / (1 + increase_percent)) = 320 := by
  sorry

end NUMINAMATH_CALUDE_store_sales_increase_l576_57666


namespace NUMINAMATH_CALUDE_number_equation_solution_l576_57664

theorem number_equation_solution : ∃ x : ℚ, x^2 + 95 = (x - 15)^2 ∧ x = 10/3 := by sorry

end NUMINAMATH_CALUDE_number_equation_solution_l576_57664


namespace NUMINAMATH_CALUDE_total_blue_marbles_l576_57625

/-- The total number of blue marbles owned by Jason, Tom, and Emily is 104. -/
theorem total_blue_marbles (jason_blue : ℕ) (tom_blue : ℕ) (emily_blue : ℕ)
  (h1 : jason_blue = 44)
  (h2 : tom_blue = 24)
  (h3 : emily_blue = 36) :
  jason_blue + tom_blue + emily_blue = 104 :=
by sorry

end NUMINAMATH_CALUDE_total_blue_marbles_l576_57625


namespace NUMINAMATH_CALUDE_total_spider_legs_l576_57662

/-- The number of legs a single spider has -/
def spider_legs : ℕ := 8

/-- The number of spiders in the group -/
def group_size : ℕ := spider_legs / 2 + 10

/-- The total number of spider legs in the group -/
def total_legs : ℕ := group_size * spider_legs

/-- Theorem stating that the total number of spider legs in the group is 112 -/
theorem total_spider_legs : total_legs = 112 := by
  sorry

end NUMINAMATH_CALUDE_total_spider_legs_l576_57662


namespace NUMINAMATH_CALUDE_smallest_number_with_hcf_twelve_l576_57603

/-- The highest common factor of two natural numbers -/
def hcf (a b : ℕ) : ℕ := Nat.gcd a b

/-- Theorem: 48 is the smallest number greater than 36 that has a highest common factor of 12 with 36 -/
theorem smallest_number_with_hcf_twelve : 
  ∀ n : ℕ, n > 36 → hcf 36 n = 12 → n ≥ 48 :=
sorry

end NUMINAMATH_CALUDE_smallest_number_with_hcf_twelve_l576_57603


namespace NUMINAMATH_CALUDE_prism_volume_l576_57610

/-- Given a right rectangular prism with dimensions a, b, and c, where:
    - ab = 24 (area of side face)
    - bc = 8 (area of front face)
    - ca = 3 (area of bottom face)
    Prove that the volume of the prism is 24 cubic inches. -/
theorem prism_volume (a b c : ℝ) (h1 : a * b = 24) (h2 : b * c = 8) (h3 : c * a = 3) :
  a * b * c = 24 := by
  sorry

end NUMINAMATH_CALUDE_prism_volume_l576_57610


namespace NUMINAMATH_CALUDE_sqrt_2_simplest_l576_57643

def is_simplest_quadratic_radical (x : ℝ) : Prop :=
  ∃ (y : ℝ), x = Real.sqrt y ∧ (∀ (z : ℝ), z ^ 2 = y → z = x)

theorem sqrt_2_simplest :
  is_simplest_quadratic_radical (Real.sqrt 2) ∧
  ¬is_simplest_quadratic_radical (3 ^ (1/3 : ℝ)) ∧
  ¬is_simplest_quadratic_radical (Real.sqrt (1/2)) ∧
  ¬is_simplest_quadratic_radical (Real.sqrt 16) :=
by sorry

end NUMINAMATH_CALUDE_sqrt_2_simplest_l576_57643


namespace NUMINAMATH_CALUDE_rectangle_ratio_l576_57667

/-- A configuration of squares and a rectangle forming a larger square -/
structure SquareConfiguration where
  /-- Side length of each small square -/
  s : ℝ
  /-- Side length of the large square -/
  bigSquareSide : ℝ
  /-- Length of the rectangle -/
  rectLength : ℝ
  /-- Width of the rectangle -/
  rectWidth : ℝ
  /-- The large square's side is 5 times the small square's side -/
  bigSquare_eq : bigSquareSide = 5 * s
  /-- The rectangle's length is equal to the large square's side -/
  rectLength_eq : rectLength = bigSquareSide
  /-- The rectangle's width is the large square's side minus 4 small square sides -/
  rectWidth_eq : rectWidth = bigSquareSide - 4 * s

/-- The ratio of the rectangle's length to its width is 5 -/
theorem rectangle_ratio (config : SquareConfiguration) :
    config.rectLength / config.rectWidth = 5 := by
  sorry


end NUMINAMATH_CALUDE_rectangle_ratio_l576_57667


namespace NUMINAMATH_CALUDE_letters_with_dot_only_in_given_alphabet_l576_57605

/-- Represents an alphabet with letters containing dots and/or straight lines -/
structure Alphabet where
  total : ℕ
  both : ℕ
  straight_only : ℕ
  dot_only : ℕ
  all_contain : both + straight_only + dot_only = total

/-- The number of letters containing only a dot in a specific alphabet -/
def letters_with_dot_only (a : Alphabet) : ℕ := a.dot_only

/-- Theorem stating the number of letters with only a dot in the given alphabet -/
theorem letters_with_dot_only_in_given_alphabet :
  ∃ (a : Alphabet), a.total = 60 ∧ a.both = 20 ∧ a.straight_only = 36 ∧ letters_with_dot_only a = 4 := by
  sorry

end NUMINAMATH_CALUDE_letters_with_dot_only_in_given_alphabet_l576_57605


namespace NUMINAMATH_CALUDE_scatter_plot_placement_l576_57668

/-- Represents a variable in a scatter plot -/
inductive Variable
  | Explanatory
  | Forecast

/-- Represents an axis in a scatter plot -/
inductive Axis
  | X
  | Y

/-- Represents the correct placement of variables on axes in a scatter plot -/
def correct_placement (v : Variable) (a : Axis) : Prop :=
  match v, a with
  | Variable.Explanatory, Axis.X => True
  | Variable.Forecast, Axis.Y => True
  | _, _ => False

/-- Theorem stating the correct placement of variables in a scatter plot -/
theorem scatter_plot_placement :
  ∀ (v : Variable) (a : Axis),
    correct_placement v a ↔
      ((v = Variable.Explanatory ∧ a = Axis.X) ∨
       (v = Variable.Forecast ∧ a = Axis.Y)) :=
by sorry

end NUMINAMATH_CALUDE_scatter_plot_placement_l576_57668


namespace NUMINAMATH_CALUDE_cow_profit_is_600_l576_57633

/-- Calculates the profit from selling a cow given the purchase price, daily food cost,
    health care cost, number of days kept, and selling price. -/
def cowProfit (purchasePrice foodCostPerDay healthCareCost numDays sellingPrice : ℕ) : ℕ :=
  sellingPrice - (purchasePrice + foodCostPerDay * numDays + healthCareCost)

/-- Theorem stating that the profit from selling the cow under given conditions is $600. -/
theorem cow_profit_is_600 :
  cowProfit 600 20 500 40 2500 = 600 := by
  sorry

#eval cowProfit 600 20 500 40 2500

end NUMINAMATH_CALUDE_cow_profit_is_600_l576_57633


namespace NUMINAMATH_CALUDE_quartic_equation_roots_l576_57609

theorem quartic_equation_roots : 
  let f (x : ℝ) := 8*x^4 - 47*x^3 + 74*x^2 - 47*x + 8
  let α := (47 + Real.sqrt 353) / 32
  let β := (47 - Real.sqrt 353) / 32
  ∃ (r₁ r₂ r₃ r₄ : ℝ),
    r₁ = α + Real.sqrt ((α*2)^2 - 4) / 2 ∧
    r₂ = α - Real.sqrt ((α*2)^2 - 4) / 2 ∧
    r₃ = β + Real.sqrt ((β*2)^2 - 4) / 2 ∧
    r₄ = β - Real.sqrt ((β*2)^2 - 4) / 2 ∧
    f r₁ = 0 ∧ f r₂ = 0 ∧ f r₃ = 0 ∧ f r₄ = 0 :=
by
  sorry

end NUMINAMATH_CALUDE_quartic_equation_roots_l576_57609


namespace NUMINAMATH_CALUDE_root_product_bound_l576_57637

-- Define the equations
def equation1 (x : ℝ) : Prop := Real.log x / Real.log 4 - (1/4)^x = 0
def equation2 (x : ℝ) : Prop := Real.log x / Real.log (1/4) - (1/4)^x = 0

-- State the theorem
theorem root_product_bound 
  (x₁ x₂ : ℝ) 
  (h1 : equation1 x₁) 
  (h2 : equation2 x₂) : 
  0 < x₁ * x₂ ∧ x₁ * x₂ < 1 := by
  sorry

end NUMINAMATH_CALUDE_root_product_bound_l576_57637


namespace NUMINAMATH_CALUDE_ordering_of_numbers_l576_57626

theorem ordering_of_numbers (a b : ℝ) 
  (ha : a > 0) 
  (hb : b < 0) 
  (hab : a + b < 0) : 
  b < -a ∧ -a < 0 ∧ 0 < a ∧ a < -b :=
sorry

end NUMINAMATH_CALUDE_ordering_of_numbers_l576_57626
