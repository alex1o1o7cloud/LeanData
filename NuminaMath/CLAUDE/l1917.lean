import Mathlib

namespace geometric_sequence_12th_term_l1917_191761

/-- In a geometric sequence, given the 4th and 8th terms, prove the 12th term -/
theorem geometric_sequence_12th_term 
  (a : ℕ → ℝ) -- The sequence
  (h1 : a 4 = 2) -- 4th term is 2
  (h2 : a 8 = 162) -- 8th term is 162
  (h3 : ∀ n : ℕ, n ≥ 1 → ∃ r : ℝ, a (n + 1) = a n * r) -- Definition of geometric sequence
  : a 12 = 13122 :=
by sorry

end geometric_sequence_12th_term_l1917_191761


namespace circle_tangent_to_line_l1917_191792

theorem circle_tangent_to_line (m : ℝ) (h : m ≥ 0) :
  ∃ (x y : ℝ), x^2 + y^2 = m ∧ x + y = Real.sqrt (2 * m) ∧
  ∀ (x' y' : ℝ), x'^2 + y'^2 = m → x' + y' ≤ Real.sqrt (2 * m) :=
sorry

end circle_tangent_to_line_l1917_191792


namespace f_4_3_2_1_l1917_191776

/-- The mapping f from (a₁, a₂, a₃, a₄) to (b₁, b₂, b₃, b₄) based on the equation
    x^4 + a₁x³ + a₂x² + a₃x + a₄ = (x+1)^4 + b₁(x+1)³ + b₂(x+1)² + b₃(x+1) + b₄ -/
def f (a₁ a₂ a₃ a₄ : ℝ) : ℝ × ℝ × ℝ × ℝ := sorry

/-- Theorem stating that f(4, 3, 2, 1) = (0, -3, 4, -1) -/
theorem f_4_3_2_1 : f 4 3 2 1 = (0, -3, 4, -1) := by sorry

end f_4_3_2_1_l1917_191776


namespace subtraction_preserves_inequality_l1917_191782

theorem subtraction_preserves_inequality (a b c : ℝ) (h : a > b) : b - c < a - c := by
  sorry

end subtraction_preserves_inequality_l1917_191782


namespace sum_of_cubes_divisible_by_nine_l1917_191726

theorem sum_of_cubes_divisible_by_nine (x : ℤ) : 
  ∃ k : ℤ, (x - 1)^3 + x^3 + (x + 1)^3 = 9 * k := by
  sorry

end sum_of_cubes_divisible_by_nine_l1917_191726


namespace linear_function_not_in_quadrant_III_l1917_191703

-- Define the linear function
def linear_function (k : ℝ) (x : ℝ) : ℝ := k * x - k

-- Define the condition that y decreases as x increases
def decreasing_y (f : ℝ → ℝ) : Prop :=
  ∀ x₁ x₂, x₁ < x₂ → f x₁ > f x₂

-- Define what it means for a point to be in Quadrant III
def in_quadrant_III (x : ℝ) (y : ℝ) : Prop :=
  x < 0 ∧ y < 0

-- Theorem statement
theorem linear_function_not_in_quadrant_III (k : ℝ) :
  decreasing_y (linear_function k) →
  ¬∃ x, in_quadrant_III x (linear_function k x) :=
by sorry

end linear_function_not_in_quadrant_III_l1917_191703


namespace largest_tank_volume_width_l1917_191750

/-- Represents the dimensions of a rectangular crate -/
structure CrateDimensions where
  length : ℝ
  width : ℝ
  height : ℝ

/-- Represents a cylindrical gas tank -/
structure GasTank where
  radius : ℝ
  height : ℝ

/-- Checks if a gas tank fits inside a crate -/
def tankFitsInCrate (tank : GasTank) (crate : CrateDimensions) : Prop :=
  (2 * tank.radius ≤ crate.length ∧ 2 * tank.radius ≤ crate.width) ∨
  (2 * tank.radius ≤ crate.length ∧ 2 * tank.radius ≤ crate.height) ∨
  (2 * tank.radius ≤ crate.width ∧ 2 * tank.radius ≤ crate.height)

/-- Theorem: The width of the crate must be 8 feet for the largest possible tank volume -/
theorem largest_tank_volume_width (x : ℝ) :
  let crate := CrateDimensions.mk 6 x 10
  let tank := GasTank.mk 4 (min (min 6 x) 10)
  tankFitsInCrate tank crate → x = 8 := by
  sorry

#check largest_tank_volume_width

end largest_tank_volume_width_l1917_191750


namespace parabola_focus_l1917_191709

/-- A parabola with equation y^2 = 2px (p > 0) and directrix x = -1 has its focus at (1, 0) -/
theorem parabola_focus (p : ℝ) (h : p > 0) :
  let parabola := {(x, y) : ℝ × ℝ | y^2 = 2*p*x}
  let directrix := {(x, y) : ℝ × ℝ | x = -1}
  let focus := (1, 0)
  (∀ (point : ℝ × ℝ), point ∈ parabola ↔ 
    Real.sqrt ((point.1 - focus.1)^2 + (point.2 - focus.2)^2) = 
    |point.1 - (-1)|) :=
by sorry

end parabola_focus_l1917_191709


namespace product_of_group_at_least_72_l1917_191784

theorem product_of_group_at_least_72 (group1 group2 group3 : List Nat) : 
  (group1 ++ group2 ++ group3).toFinset = Finset.range 9 →
  (group1.prod ≥ 72) ∨ (group2.prod ≥ 72) ∨ (group3.prod ≥ 72) := by
  sorry

end product_of_group_at_least_72_l1917_191784


namespace expression_is_integer_not_necessarily_natural_l1917_191775

theorem expression_is_integer_not_necessarily_natural : ∃ (n : ℤ), 
  (((1 + Real.sqrt 1991)^100 - (1 - Real.sqrt 1991)^100) / Real.sqrt 1991 = n) ∧ 
  (n ≠ 0 ∨ n < 0) := by
  sorry

end expression_is_integer_not_necessarily_natural_l1917_191775


namespace min_distance_line_curve_l1917_191744

/-- The minimum distance between a point on the line 2x - y + 6 = 0 and
    a point on the curve y = 2ln(x) + 2 is 6√5/5 -/
theorem min_distance_line_curve :
  let line := {(x, y) : ℝ × ℝ | 2 * x - y + 6 = 0}
  let curve := {(x, y) : ℝ × ℝ | y = 2 * Real.log x + 2}
  ∃ (d : ℝ), d = 6 * Real.sqrt 5 / 5 ∧
    ∀ (p : ℝ × ℝ) (q : ℝ × ℝ), p ∈ line → q ∈ curve →
      Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2) ≥ d :=
by sorry

end min_distance_line_curve_l1917_191744


namespace prob_three_same_color_standard_deck_l1917_191711

/-- Represents a standard deck of cards -/
structure Deck :=
  (total_cards : ℕ)
  (suits : ℕ)
  (cards_per_suit : ℕ)
  (red_suits : ℕ)
  (black_suits : ℕ)

/-- A standard 52-card deck -/
def standard_deck : Deck :=
  { total_cards := 52,
    suits := 4,
    cards_per_suit := 13,
    red_suits := 2,
    black_suits := 2 }

/-- The probability of drawing three cards of the same color from a standard deck -/
def prob_three_same_color (d : Deck) : ℚ :=
  let red_cards := d.red_suits * d.cards_per_suit
  let total_ways := d.total_cards * (d.total_cards - 1) * (d.total_cards - 2)
  let same_color_ways := 2 * (red_cards * (red_cards - 1) * (red_cards - 2))
  same_color_ways / total_ways

/-- Theorem: The probability of drawing three cards of the same color from a standard 52-card deck is 40/85 -/
theorem prob_three_same_color_standard_deck :
  prob_three_same_color standard_deck = 40 / 85 := by
  sorry

end prob_three_same_color_standard_deck_l1917_191711


namespace complex_number_with_sqrt3_imaginary_and_modulus_2_l1917_191716

theorem complex_number_with_sqrt3_imaginary_and_modulus_2 :
  ∀ z : ℂ, (z.im = Real.sqrt 3) → (Complex.abs z = 2) →
  (z = Complex.mk 1 (Real.sqrt 3) ∨ z = Complex.mk (-1) (Real.sqrt 3)) :=
by
  sorry

end complex_number_with_sqrt3_imaginary_and_modulus_2_l1917_191716


namespace unique_three_digit_divisible_by_seven_l1917_191771

theorem unique_three_digit_divisible_by_seven : 
  ∃! n : ℕ, 
    100 ≤ n ∧ n < 1000 ∧  -- three-digit number
    n % 10 = 4 ∧          -- units digit is 4
    n / 100 = 6 ∧         -- hundreds digit is 6
    n % 7 = 0 ∧           -- divisible by 7
    n = 658               -- the number is 658
  := by sorry

end unique_three_digit_divisible_by_seven_l1917_191771


namespace farm_field_area_l1917_191719

/-- Represents the farm field ploughing problem -/
structure FarmField where
  plannedDailyArea : ℕ  -- Planned area to plough per day
  actualDailyArea : ℕ   -- Actual area ploughed per day
  extraDays : ℕ         -- Extra days needed
  remainingArea : ℕ     -- Area left to plough

/-- Calculates the total area of the farm field -/
def totalArea (f : FarmField) : ℕ :=
  f.plannedDailyArea * ((f.actualDailyArea * (f.extraDays + 3) + f.remainingArea) / f.plannedDailyArea)

/-- Theorem stating that the total area of the given farm field is 480 hectares -/
theorem farm_field_area (f : FarmField) 
    (h1 : f.plannedDailyArea = 160)
    (h2 : f.actualDailyArea = 85)
    (h3 : f.extraDays = 2)
    (h4 : f.remainingArea = 40) : 
  totalArea f = 480 := by
  sorry

#eval totalArea { plannedDailyArea := 160, actualDailyArea := 85, extraDays := 2, remainingArea := 40 }

end farm_field_area_l1917_191719


namespace work_hours_constant_l1917_191753

/-- Represents the work schedule for a week -/
structure WorkSchedule where
  days_per_week : ℕ
  initial_hours_task1 : ℕ
  initial_hours_task2 : ℕ
  hours_reduction_task1 : ℕ

/-- Calculates the total weekly work hours -/
def total_weekly_hours (schedule : WorkSchedule) : ℕ :=
  schedule.days_per_week * (schedule.initial_hours_task1 + schedule.initial_hours_task2)

/-- Theorem stating that the total weekly work hours remain constant after redistribution -/
theorem work_hours_constant (schedule : WorkSchedule) 
  (h1 : schedule.days_per_week = 5)
  (h2 : schedule.initial_hours_task1 = 5)
  (h3 : schedule.initial_hours_task2 = 3)
  (h4 : schedule.hours_reduction_task1 = 5) :
  total_weekly_hours schedule = 40 := by
  sorry

#eval total_weekly_hours { days_per_week := 5, initial_hours_task1 := 5, initial_hours_task2 := 3, hours_reduction_task1 := 5 }

end work_hours_constant_l1917_191753


namespace first_question_percentage_l1917_191745

/-- Given a class of students taking a test with two questions, this theorem proves
    the percentage of students who answered the first question correctly. -/
theorem first_question_percentage
  (second_correct : ℝ)
  (neither_correct : ℝ)
  (both_correct : ℝ)
  (h1 : second_correct = 65)
  (h2 : neither_correct = 20)
  (h3 : both_correct = 60) :
  ∃ (first_correct : ℝ),
    first_correct = 75 ∧
    first_correct + second_correct - both_correct = 100 - neither_correct :=
by
  sorry


end first_question_percentage_l1917_191745


namespace rice_stock_calculation_l1917_191769

theorem rice_stock_calculation (initial_stock sold restocked : ℕ) : 
  initial_stock = 55 → sold = 23 → restocked = 132 → 
  initial_stock - sold + restocked = 164 := by
  sorry

end rice_stock_calculation_l1917_191769


namespace white_more_probable_l1917_191710

def yellow_balls : ℕ := 3
def white_balls : ℕ := 5
def total_balls : ℕ := yellow_balls + white_balls

def prob_yellow : ℚ := yellow_balls / total_balls
def prob_white : ℚ := white_balls / total_balls

theorem white_more_probable : prob_white > prob_yellow := by
  sorry

end white_more_probable_l1917_191710


namespace parallel_resistors_l1917_191736

/-- Given two resistors connected in parallel with resistances x and y,
    where the combined resistance r satisfies 1/r = 1/x + 1/y,
    prove that when x = 4 ohms and r = 2.4 ohms, y = 6 ohms. -/
theorem parallel_resistors (x y r : ℝ) 
  (hx : x = 4)
  (hr : r = 2.4)
  (h_combined : 1 / r = 1 / x + 1 / y) :
  y = 6 := by
  sorry

end parallel_resistors_l1917_191736


namespace sequence_formula_l1917_191747

-- Define the sequence type
def Sequence := ℕ → ℝ

-- Define the properties of the sequence
def PropertyOne (a : Sequence) : Prop :=
  ∀ m n : ℕ, m > n → a (m - n) = a m - a n

def PropertyTwo (a : Sequence) : Prop :=
  ∀ m n : ℕ, m > n → a m > a n

-- State the theorem
theorem sequence_formula (a : Sequence) 
  (h1 : PropertyOne a) (h2 : PropertyTwo a) : 
  ∃ k : ℝ, k > 0 ∧ ∀ n : ℕ, a n = k * n := by
  sorry

end sequence_formula_l1917_191747


namespace intersection_of_A_and_B_l1917_191725

def A : Set ℝ := {y | ∃ x, y = Real.cos x}
def B : Set ℝ := {x | x^2 < 9}

theorem intersection_of_A_and_B : A ∩ B = Set.Icc (-1) 1 := by sorry

end intersection_of_A_and_B_l1917_191725


namespace c_monthly_income_l1917_191760

/-- The monthly income ratio between A and B -/
def income_ratio : ℚ := 5 / 2

/-- The percentage increase of B's income over C's income -/
def income_increase_percentage : ℚ := 12 / 100

/-- A's annual income in rupees -/
def a_annual_income : ℕ := 470400

/-- The number of months in a year -/
def months_per_year : ℕ := 12

/-- Theorem stating C's monthly income -/
theorem c_monthly_income :
  let a_monthly_income : ℚ := a_annual_income / months_per_year
  let b_monthly_income : ℚ := a_monthly_income / income_ratio
  let c_monthly_income : ℚ := b_monthly_income / (1 + income_increase_percentage)
  c_monthly_income = 14000 := by sorry

end c_monthly_income_l1917_191760


namespace min_perimeter_is_8_meters_l1917_191773

/-- Represents the side length of a square tile in centimeters -/
def tileSideLength : ℕ := 40

/-- Represents the total number of tiles -/
def totalTiles : ℕ := 24

/-- Calculates the perimeter of a rectangle given its length and width in tile units -/
def perimeterInTiles (length width : ℕ) : ℕ := 2 * (length + width)

/-- Checks if the given dimensions form a valid rectangle using all tiles -/
def isValidRectangle (length width : ℕ) : Prop := length * width = totalTiles

/-- Theorem: The minimum perimeter of a rectangular arrangement of 24 square tiles,
    each with side length 40 cm, is 8 meters -/
theorem min_perimeter_is_8_meters :
  ∃ (length width : ℕ),
    isValidRectangle length width ∧
    ∀ (l w : ℕ), isValidRectangle l w →
      perimeterInTiles length width ≤ perimeterInTiles l w ∧
      perimeterInTiles length width * tileSideLength = 800 :=
by sorry

end min_perimeter_is_8_meters_l1917_191773


namespace smallest_cut_length_for_non_triangle_l1917_191730

theorem smallest_cut_length_for_non_triangle : ∃ (x : ℕ), 
  (x > 0) ∧ 
  (∀ (y : ℕ), y < x → (9 - y) + (16 - y) > (18 - y)) ∧
  ((9 - x) + (16 - x) ≤ (18 - x)) ∧ 
  x = 7 := by
  sorry

end smallest_cut_length_for_non_triangle_l1917_191730


namespace corn_acreage_l1917_191783

theorem corn_acreage (total_land : ℕ) (bean_ratio wheat_ratio corn_ratio : ℕ) 
  (h1 : total_land = 1034)
  (h2 : bean_ratio = 5)
  (h3 : wheat_ratio = 2)
  (h4 : corn_ratio = 4) :
  (total_land * corn_ratio) / (bean_ratio + wheat_ratio + corn_ratio) = 376 := by
  sorry

end corn_acreage_l1917_191783


namespace simplify_expression_l1917_191721

theorem simplify_expression : (7^3 * (2^5)^3) / ((7^2) * 2^(3*3)) = 448 := by
  sorry

end simplify_expression_l1917_191721


namespace largest_of_three_numbers_l1917_191786

theorem largest_of_three_numbers (x y z : ℝ) 
  (sum_eq : x + y + z = 3)
  (sum_products_eq : x*y + x*z + y*z = -6)
  (product_eq : x*y*z = -8) :
  ∃ max_val : ℝ, max_val = (1 + Real.sqrt 17) / 2 ∧ 
  max_val = max x (max y z) := by
sorry

end largest_of_three_numbers_l1917_191786


namespace committee_size_is_four_l1917_191766

/-- The number of boys in the total group -/
def num_boys : ℕ := 5

/-- The number of girls in the total group -/
def num_girls : ℕ := 6

/-- The number of boys required in each committee -/
def boys_in_committee : ℕ := 2

/-- The number of girls required in each committee -/
def girls_in_committee : ℕ := 2

/-- The total number of possible committees -/
def total_committees : ℕ := 150

/-- The number of people in each committee -/
def committee_size : ℕ := boys_in_committee + girls_in_committee

theorem committee_size_is_four :
  committee_size = 4 :=
by sorry

end committee_size_is_four_l1917_191766


namespace transylvania_statements_l1917_191789

/-- Represents a statement that can be made by a resident of Transylvania -/
structure Statement :=
  (proposition : Prop)

/-- Defines what it means for one statement to be the converse of another -/
def is_converse (X Y : Statement) : Prop :=
  ∃ P Q : Prop, X.proposition = (P → Q) ∧ Y.proposition = (Q → P)

/-- Defines the property that asserting one statement implies the truth of another -/
def implies_truth (X Y : Statement) : Prop :=
  ∀ (resident : Prop), (resident → X.proposition) → Y.proposition

/-- The main theorem stating the existence of two statements satisfying the given conditions -/
theorem transylvania_statements : ∃ (X Y : Statement),
  is_converse X Y ∧
  (¬ (X.proposition → Y.proposition)) ∧
  (¬ (Y.proposition → X.proposition)) ∧
  implies_truth X Y ∧
  implies_truth Y X := by
  sorry

end transylvania_statements_l1917_191789


namespace competition_participants_l1917_191770

theorem competition_participants : 
  ∀ n : ℕ, 
  (100 < n ∧ n < 200) ∧ 
  (∃ k : ℕ, n = 4 * k - 2) ∧
  (∃ l : ℕ, n = 5 * l - 3) ∧
  (∃ m : ℕ, n = 6 * m - 4) →
  n = 122 ∨ n = 182 := by
sorry

end competition_participants_l1917_191770


namespace not_divisible_by_67_l1917_191758

theorem not_divisible_by_67 (x y : ℕ) 
  (h1 : ¬(67 ∣ x))
  (h2 : ¬(67 ∣ y))
  (h3 : 67 ∣ (7*x + 32*y)) :
  ¬(67 ∣ (10*x + 17*y + 1)) := by
sorry

end not_divisible_by_67_l1917_191758


namespace sin_cos_sum_14_16_l1917_191787

theorem sin_cos_sum_14_16 : 
  Real.sin (14 * π / 180) * Real.cos (16 * π / 180) + 
  Real.cos (14 * π / 180) * Real.sin (16 * π / 180) = 1 / 2 := by
  sorry

end sin_cos_sum_14_16_l1917_191787


namespace cubic_difference_l1917_191759

theorem cubic_difference (x : ℝ) (h : x - 1/x = 5) : x^3 - 1/x^3 = 135 := by
  sorry

end cubic_difference_l1917_191759


namespace cars_with_airbag_l1917_191763

theorem cars_with_airbag (total : ℕ) (power_windows : ℕ) (both : ℕ) (neither : ℕ) :
  total = 65 →
  power_windows = 30 →
  both = 12 →
  neither = 2 →
  total - neither = power_windows + (total - power_windows - neither) - both :=
by sorry

end cars_with_airbag_l1917_191763


namespace manager_chef_wage_difference_l1917_191790

/-- Represents the hourly wages at Joe's Steakhouse -/
structure SteakhouseWages where
  manager : ℝ
  dishwasher : ℝ
  chef : ℝ

/-- Defines the wage relationships at Joe's Steakhouse -/
def valid_steakhouse_wages (w : SteakhouseWages) : Prop :=
  w.manager = 8.50 ∧
  w.dishwasher = w.manager / 2 ∧
  w.chef = w.dishwasher * 1.22

/-- Theorem stating the wage difference between manager and chef -/
theorem manager_chef_wage_difference (w : SteakhouseWages) 
  (h : valid_steakhouse_wages w) : 
  w.manager - w.chef = 3.315 := by
  sorry

end manager_chef_wage_difference_l1917_191790


namespace maria_water_bottles_l1917_191701

/-- Calculates the final number of water bottles Maria has after a series of actions. -/
def final_bottle_count (initial : ℕ) (drunk : ℕ) (given_away : ℕ) (bought : ℕ) : ℕ :=
  initial - drunk - given_away + bought

/-- Theorem stating that Maria ends up with 71 bottles given the initial conditions and actions. -/
theorem maria_water_bottles : final_bottle_count 23 12 5 65 = 71 := by
  sorry

end maria_water_bottles_l1917_191701


namespace inequality_solution_l1917_191788

theorem inequality_solution (x : ℝ) : (10 * x^2 + 1 < 7 * x) ∧ ((2 * x - 7) / (-3 * x + 1) > 0) ↔ x > 1/3 ∧ x < 1/2 := by
  sorry

end inequality_solution_l1917_191788


namespace total_cookies_l1917_191712

def cookie_problem (chris kenny glenn dan anne : ℕ) : Prop :=
  chris = kenny / 3 ∧
  glenn = 4 * chris ∧
  glenn = 24 ∧
  dan = 2 * (chris + kenny) ∧
  anne = kenny / 2

theorem total_cookies :
  ∀ chris kenny glenn dan anne : ℕ,
  cookie_problem chris kenny glenn dan anne →
  chris + kenny + glenn + dan + anne = 105 :=
by
  sorry

end total_cookies_l1917_191712


namespace third_term_is_16_l1917_191746

/-- Geometric sequence with common ratio 2 and sum of first 4 terms equal to 60 -/
def geometric_sequence (a : ℕ → ℝ) : Prop :=
  (∀ n, a (n + 1) = 2 * a n) ∧ 
  (a 1 + a 2 + a 3 + a 4 = 60)

/-- The third term of the geometric sequence is 16 -/
theorem third_term_is_16 (a : ℕ → ℝ) (h : geometric_sequence a) : a 3 = 16 := by
  sorry

end third_term_is_16_l1917_191746


namespace reflection_point_properties_l1917_191751

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- A concave spherical mirror -/
structure SphericalMirror where
  radius : ℝ
  center : Point

/-- The reflection point on a spherical mirror -/
def reflection_point (mirror : SphericalMirror) (A B : Point) : Point :=
  sorry

/-- Theorem: The reflection point satisfies the sphere equation and reflection equation -/
theorem reflection_point_properties (mirror : SphericalMirror) (A B : Point) :
  let X := reflection_point mirror A B
  (X.x^2 + X.y^2 = mirror.radius^2) ∧
  ((A.x * B.y + B.x * A.y) * (X.x^2 - X.y^2) - 
   2 * (A.x * B.x - A.y * B.y) * X.x * X.y + 
   mirror.radius^2 * ((A.x + B.x) * X.y - (A.y + B.y) * X.x) = 0) := by
  sorry


end reflection_point_properties_l1917_191751


namespace sequence_sum_equality_l1917_191794

/-- Given two integer sequences satisfying a specific condition, 
    there exists a positive integer k such that the sum of the k-th terms 
    equals the sum of the (k+2018)-th terms. -/
theorem sequence_sum_equality 
  (a b : ℕ → ℤ) 
  (h : ∀ n ≥ 3, (a n - a (n-1)) * (a n - a (n-2)) + 
                (b n - b (n-1)) * (b n - b (n-2)) = 0) : 
  ∃ k : ℕ+, a k + b k = a (k + 2018) + b (k + 2018) := by
sorry

end sequence_sum_equality_l1917_191794


namespace c_range_l1917_191722

def P (c : ℝ) : Prop := ∀ x y : ℝ, x < y → c^x > c^y

def q (c : ℝ) : Prop := ∀ x y : ℝ, 1/2 < x ∧ x < y → (x^2 - 2*c*x + 1) < (y^2 - 2*c*y + 1)

theorem c_range (c : ℝ) (h1 : c > 0) (h2 : c ≠ 1) :
  (P c ∨ q c) ∧ ¬(P c ∧ q c) → 1/2 < c ∧ c < 1 := by
  sorry

end c_range_l1917_191722


namespace sum_of_powers_l1917_191737

theorem sum_of_powers (n : ℕ) :
  n^5 + n^5 + n^5 + n^5 + n^5 = 5 * n^5 := by
  sorry

end sum_of_powers_l1917_191737


namespace unicorn_tower_rope_length_l1917_191720

/-- Represents the setup of the unicorn and tower problem -/
structure UnicornTowerSetup where
  towerRadius : ℝ
  ropeLength : ℝ
  ropeAngle : ℝ
  ropeDistanceFromTower : ℝ
  unicornHeight : ℝ

/-- Calculates the length of rope touching the tower given the problem setup -/
def ropeTouchingTowerLength (setup : UnicornTowerSetup) : ℝ :=
  sorry

/-- Theorem stating the length of rope touching the tower in the given setup -/
theorem unicorn_tower_rope_length :
  let setup : UnicornTowerSetup := {
    towerRadius := 5,
    ropeLength := 30,
    ropeAngle := 30 * Real.pi / 180,  -- Convert to radians
    ropeDistanceFromTower := 5,
    unicornHeight := 5
  }
  ∃ (ε : ℝ), abs (ropeTouchingTowerLength setup - 19.06) < ε ∧ ε > 0 :=
sorry

end unicorn_tower_rope_length_l1917_191720


namespace olivers_candy_l1917_191762

/-- Oliver's Halloween candy problem -/
theorem olivers_candy (initial_candy : ℕ) : 
  (initial_candy - 10 = 68) → initial_candy = 78 := by
  sorry

end olivers_candy_l1917_191762


namespace geometric_sequence_proof_l1917_191754

theorem geometric_sequence_proof (a : ℕ → ℝ) (q : ℝ) (h_positive : q > 0) :
  (∀ n : ℕ, a (n + 1) = q * a n) →  -- geometric sequence condition
  a 2 * a 6 = 9 * a 4 →             -- given condition
  a 2 = 1 →                         -- given condition
  (q = 3 ∧ ∀ n : ℕ, a n = 3^(n - 2)) := by
  sorry

end geometric_sequence_proof_l1917_191754


namespace square_stack_sums_l1917_191704

theorem square_stack_sums : 
  (¬ ∃ n : ℕ+, 10 * n = 8016) ∧ 
  (∃ n : ℕ+, 10 * n = 8020) := by
  sorry

end square_stack_sums_l1917_191704


namespace ellipse_intersection_fixed_point_l1917_191738

-- Define the ellipse
def ellipse (x y : ℝ) : Prop := x^2/4 + y^2 = 1

-- Define the line
def line (k : ℝ) (x y : ℝ) : Prop := x = k*y - 1

-- Define the intersection points
def intersection_points (k : ℝ) (x₁ y₁ x₂ y₂ : ℝ) : Prop :=
  ellipse x₁ y₁ ∧ ellipse x₂ y₂ ∧ line k x₁ y₁ ∧ line k x₂ y₂

-- Define the reflection point
def reflection_point (x₁ y₁ : ℝ) : ℝ × ℝ := (x₁, -y₁)

-- Theorem statement
theorem ellipse_intersection_fixed_point (k : ℝ) (x₁ y₁ x₂ y₂ : ℝ) :
  intersection_points k x₁ y₁ x₂ y₂ →
  let (x₁', y₁') := reflection_point x₁ y₁
  (x₁' ≠ x₂ ∨ y₁' ≠ y₂) →
  ∃ (t : ℝ), t * (x₂ - x₁') + x₁' = -4 ∧ t * (y₂ - y₁') + y₁' = 0 :=
by sorry

end ellipse_intersection_fixed_point_l1917_191738


namespace problem_statement_l1917_191723

theorem problem_statement (a b c : ℝ) 
  (h1 : a + 2*b + 3*c = 12) 
  (h2 : a^2 + b^2 + c^2 = a*b + a*c + b*c) : 
  a + b^2 + c^3 = 14 := by
  sorry

end problem_statement_l1917_191723


namespace project_hours_l1917_191702

theorem project_hours (kate mark pat : ℕ) 
  (h1 : pat = 2 * kate)
  (h2 : pat * 3 = mark)
  (h3 : mark = kate + 80) :
  kate + mark + pat = 144 := by
sorry

end project_hours_l1917_191702


namespace greatest_number_with_odd_factors_under_200_l1917_191735

theorem greatest_number_with_odd_factors_under_200 :
  ∀ n : ℕ, n < 200 → (∃ k : ℕ, n = k^2) →
  ∀ m : ℕ, m < 200 → (∃ l : ℕ, m = l^2) → m ≤ n →
  n = 196 :=
by sorry

end greatest_number_with_odd_factors_under_200_l1917_191735


namespace diamond_properties_l1917_191791

def diamond (a b : ℤ) : ℤ := a^2 - 2*b

theorem diamond_properties :
  (diamond (-1) 2 = -3) ∧
  (∃ a b : ℤ, diamond a b ≠ diamond b a) := by sorry

end diamond_properties_l1917_191791


namespace unfair_die_theorem_l1917_191795

def unfair_die_expected_value (p_eight : ℚ) (p_other : ℚ) : ℚ :=
  (1 * p_other + 2 * p_other + 3 * p_other + 4 * p_other + 
   5 * p_other + 6 * p_other + 7 * p_other) + (8 * p_eight)

theorem unfair_die_theorem :
  let p_eight : ℚ := 3/8
  let p_other : ℚ := 1/14
  unfair_die_expected_value p_eight p_other = 5 := by
sorry

#eval unfair_die_expected_value (3/8 : ℚ) (1/14 : ℚ)

end unfair_die_theorem_l1917_191795


namespace parallel_implications_l1917_191768

-- Define the types for points and lines
variable (Point Line : Type)

-- Define a function to check if a point is on a line
variable (on_line : Point → Line → Prop)

-- Define a function to check if two lines are parallel
variable (parallel : Line → Line → Prop)

-- Define a function to create a line from two points
variable (line_from_points : Point → Point → Line)

-- Define the theorem
theorem parallel_implications
  (l l' : Line) (O A B C A' B' C' : Point)
  (h1 : on_line A l) (h2 : on_line B l) (h3 : on_line C l)
  (h4 : on_line A' l') (h5 : on_line B' l') (h6 : on_line C' l')
  (h7 : parallel (line_from_points A B') (line_from_points A' B))
  (h8 : parallel (line_from_points A C') (line_from_points A' C)) :
  parallel (line_from_points B C') (line_from_points B' C) :=
sorry

end parallel_implications_l1917_191768


namespace loop_condition_proof_l1917_191728

theorem loop_condition_proof (i₀ S₀ : ℕ) (result : ℕ) : 
  i₀ = 12 → S₀ = 1 → result = 11880 →
  (∃ n : ℕ, result = Nat.factorial n - Nat.factorial (i₀ - 1)) →
  (∀ i S : ℕ, i > 9 ↔ result = S ∧ S = Nat.factorial i - Nat.factorial (i₀ - 1)) :=
by sorry

end loop_condition_proof_l1917_191728


namespace f_is_quadratic_l1917_191798

/-- Definition of a quadratic equation -/
def is_quadratic_equation (f : ℝ → ℝ) : Prop :=
  ∃ a b c : ℝ, a ≠ 0 ∧ ∀ x, f x = a * x^2 + b * x + c

/-- The function representing x^2 - 13 = 0 -/
def f (x : ℝ) : ℝ := x^2 - 13

/-- Theorem stating that f is a quadratic equation -/
theorem f_is_quadratic : is_quadratic_equation f := by
  sorry

end f_is_quadratic_l1917_191798


namespace animal_rescue_donation_l1917_191781

/-- Represents the daily earnings and costs for a 5-day bake sale --/
structure BakeSaleData :=
  (earnings : Fin 5 → ℕ)
  (costs : Fin 5 → ℕ)

/-- Represents the distribution percentages for charities --/
structure CharityDistribution :=
  (homeless_shelter : ℚ)
  (food_bank : ℚ)
  (park_restoration : ℚ)
  (animal_rescue : ℚ)

/-- Calculates the total donation to the Animal Rescue Center --/
def calculateAnimalRescueDonation (data : BakeSaleData) (dist : CharityDistribution) (personal_contribution : ℕ) : ℚ :=
  sorry

/-- Theorem stating the total donation to the Animal Rescue Center --/
theorem animal_rescue_donation
  (data : BakeSaleData)
  (dist : CharityDistribution)
  (h1 : data.earnings = ![450, 550, 400, 600, 500])
  (h2 : data.costs = ![80, 100, 70, 120, 90])
  (h3 : dist.homeless_shelter = 30 / 100)
  (h4 : dist.food_bank = 25 / 100)
  (h5 : dist.park_restoration = 20 / 100)
  (h6 : dist.animal_rescue = 25 / 100)
  (h7 : personal_contribution = 20) :
  calculateAnimalRescueDonation data dist personal_contribution = 535 := by
  sorry

end animal_rescue_donation_l1917_191781


namespace driver_stops_theorem_l1917_191733

/-- Calculates the number of stops a delivery driver needs to make -/
def num_stops (total_boxes : ℕ) (boxes_per_stop : ℕ) : ℕ :=
  total_boxes / boxes_per_stop

theorem driver_stops_theorem :
  let total_boxes : ℕ := 27
  let boxes_per_stop : ℕ := 9
  num_stops total_boxes boxes_per_stop = 3 := by
sorry

end driver_stops_theorem_l1917_191733


namespace garland_arrangement_count_l1917_191707

/-- The number of ways to arrange blue, red, and white bulbs in a garland with no adjacent white bulbs -/
def garland_arrangements (blue red white : ℕ) : ℕ :=
  (Nat.choose (blue + red) blue) * (Nat.choose (blue + red + 1) white)

/-- Theorem stating the number of arrangements for 8 blue, 6 red, and 12 white bulbs -/
theorem garland_arrangement_count : garland_arrangements 8 6 12 = 1366365 := by
  sorry

end garland_arrangement_count_l1917_191707


namespace decimal_expansion_of_prime_reciprocal_l1917_191780

/-- The type of natural numbers greater than 1 -/
def PositiveNatGT1 := { n : ℕ // n > 1 }

/-- The period of a rational number's decimal expansion -/
def decimalPeriod (q : ℚ) : ℕ := sorry

/-- The nth digit in the decimal expansion of a rational number -/
def nthDecimalDigit (q : ℚ) (n : ℕ) : Fin 10 := sorry

theorem decimal_expansion_of_prime_reciprocal (p : PositiveNatGT1) 
  (h_prime : Nat.Prime p.val) 
  (h_period : decimalPeriod (1 / p.val) = 200) : 
  nthDecimalDigit (1 / p.val) 101 = 9 := by sorry

end decimal_expansion_of_prime_reciprocal_l1917_191780


namespace road_trip_equation_correct_l1917_191700

/-- Represents a road trip with a stop -/
structure RoadTrip where
  totalDistance : ℝ
  totalTime : ℝ
  stopDuration : ℝ
  speedBeforeStop : ℝ
  speedAfterStop : ℝ

/-- The equation for the road trip is correct -/
def correctEquation (trip : RoadTrip) (t : ℝ) : Prop :=
  trip.speedBeforeStop * t + trip.speedAfterStop * (trip.totalTime - trip.stopDuration / 60 - t) = trip.totalDistance

theorem road_trip_equation_correct (trip : RoadTrip) (t : ℝ) :
  trip.totalDistance = 300 ∧
  trip.totalTime = 4 ∧
  trip.stopDuration = 30 ∧
  trip.speedBeforeStop = 70 ∧
  trip.speedAfterStop = 90 →
  correctEquation trip t ↔ 70 * t + 90 * (3.5 - t) = 300 := by
  sorry


end road_trip_equation_correct_l1917_191700


namespace power_comparison_l1917_191732

theorem power_comparison : 2^100 < 3^75 := by sorry

end power_comparison_l1917_191732


namespace quadratic_inequality_l1917_191718

theorem quadratic_inequality (x : ℝ) : -8 * x^2 + 6 * x - 1 < 0 ↔ 0.25 < x ∧ x < 0.5 := by
  sorry

end quadratic_inequality_l1917_191718


namespace half_of_number_l1917_191797

theorem half_of_number (N : ℚ) (h : (4/15 : ℚ) * (5/7 : ℚ) * N = (4/9 : ℚ) * (2/5 : ℚ) * N + 8) : 
  N / 2 = 315 := by
  sorry

end half_of_number_l1917_191797


namespace not_p_and_q_is_true_l1917_191705

theorem not_p_and_q_is_true (h1 : ¬(p ∧ q)) (h2 : ¬¬q) : (¬p) ∧ q := by
  sorry

end not_p_and_q_is_true_l1917_191705


namespace quadratic_equation_roots_condition_l1917_191708

/-- Given a quadratic equation k^2x^2 + (4k-1)x + 4 = 0 with two distinct real roots,
    the range of values for k is k < 1/8 and k ≠ 0 -/
theorem quadratic_equation_roots_condition (k : ℝ) : 
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ 
    k^2 * x₁^2 + (4*k - 1) * x₁ + 4 = 0 ∧
    k^2 * x₂^2 + (4*k - 1) * x₂ + 4 = 0) →
  k < 1/8 ∧ k ≠ 0 :=
by sorry

end quadratic_equation_roots_condition_l1917_191708


namespace r_lower_bound_l1917_191767

theorem r_lower_bound (a b c d : ℕ+) (r : ℚ) :
  r = 1 - (a : ℚ) / b - (c : ℚ) / d →
  a + c ≤ 1982 →
  r ≥ 0 →
  r > 1 / (1983 : ℚ)^3 := by
  sorry

end r_lower_bound_l1917_191767


namespace sum_of_two_equals_zero_l1917_191772

theorem sum_of_two_equals_zero (a b c d : ℝ) 
  (h1 : a^3 + b^3 + c^3 + d^3 = 0) 
  (h2 : a + b + c + d = 0) : 
  a + b = 0 ∨ c + d = 0 := by
  sorry

end sum_of_two_equals_zero_l1917_191772


namespace complex_root_equation_l1917_191729

theorem complex_root_equation (p : ℝ) : 
  (Complex.I : ℂ)^2 = -1 →
  (3 - Complex.I : ℂ)^2 + p * (3 - Complex.I) + 10 = 0 →
  p = -6 := by
sorry

end complex_root_equation_l1917_191729


namespace prob_exactly_two_ones_value_l1917_191755

def num_dice : ℕ := 12
def num_sides : ℕ := 6
def target_outcome : ℕ := 1
def num_target : ℕ := 2

def prob_exactly_two_ones : ℚ :=
  (num_dice.choose num_target) *
  (1 / num_sides) ^ num_target *
  ((num_sides - 1) / num_sides) ^ (num_dice - num_target)

theorem prob_exactly_two_ones_value :
  prob_exactly_two_ones = (66 * 5^10) / 6^12 := by
  sorry

end prob_exactly_two_ones_value_l1917_191755


namespace sector_area_from_arc_length_l1917_191749

/-- Given a circle with radius 6cm and an arc length of 25.12cm, 
    the area of the sector formed by this arc is 75.36 cm². -/
theorem sector_area_from_arc_length : 
  let r : ℝ := 6  -- radius in cm
  let arc_length : ℝ := 25.12  -- arc length in cm
  let π : ℝ := Real.pi
  let central_angle : ℝ := arc_length / r  -- angle in radians
  let sector_area : ℝ := 0.5 * r^2 * central_angle
  sector_area = 75.36 := by
  sorry

#check sector_area_from_arc_length

end sector_area_from_arc_length_l1917_191749


namespace jim_apples_count_l1917_191748

theorem jim_apples_count : ∀ (j : ℕ), 
  (j + 60 + 40) / 3 = 2 * j → j = 200 := by
  sorry

end jim_apples_count_l1917_191748


namespace red_ball_probability_l1917_191717

/-- Represents a container with red and green balls -/
structure Container where
  red : Nat
  green : Nat

/-- The probability of selecting a red ball from the given containers -/
def redBallProbability (x y z : Container) : Rat :=
  (1 / 3 : Rat) * (x.red / (x.red + x.green : Rat)) +
  (1 / 3 : Rat) * (y.red / (y.red + y.green : Rat)) +
  (1 / 3 : Rat) * (z.red / (z.red + z.green : Rat))

/-- Theorem stating the probability of selecting a red ball -/
theorem red_ball_probability :
  let x : Container := { red := 3, green := 7 }
  let y : Container := { red := 7, green := 3 }
  let z : Container := { red := 7, green := 3 }
  redBallProbability x y z = 17 / 30 := by
  sorry


end red_ball_probability_l1917_191717


namespace rectangle_area_l1917_191757

theorem rectangle_area (L B : ℝ) (h1 : L - B = 23) (h2 : 2 * L + 2 * B = 226) : L * B = 3060 := by
  sorry

end rectangle_area_l1917_191757


namespace bisection_second_iteration_l1917_191752

def f (x : ℝ) := x^3 + 3*x - 1

theorem bisection_second_iteration
  (h1 : f 0 < 0)
  (h2 : f (1/2) > 0) :
  let second_iteration := (0 + 1/2) / 2
  second_iteration = 1/4 :=
by sorry

end bisection_second_iteration_l1917_191752


namespace coloring_books_total_l1917_191727

theorem coloring_books_total (initial : ℕ) (given_away : ℕ) (bought : ℕ) : 
  initial = 34 → given_away = 3 → bought = 48 → 
  initial - given_away + bought = 79 := by
  sorry

end coloring_books_total_l1917_191727


namespace sector_central_angle_l1917_191734

/-- Given a sector with radius 2 cm and area 4 cm², 
    prove that its central angle measures 2 radians. -/
theorem sector_central_angle (r : ℝ) (S : ℝ) (α : ℝ) : 
  r = 2 → S = 4 → S = (1/2) * r^2 * α → α = 2 := by
  sorry

end sector_central_angle_l1917_191734


namespace largest_package_size_l1917_191713

theorem largest_package_size (hazel_pencils leo_pencils mia_pencils : ℕ) 
  (h1 : hazel_pencils = 36)
  (h2 : leo_pencils = 54)
  (h3 : mia_pencils = 72) :
  Nat.gcd hazel_pencils (Nat.gcd leo_pencils mia_pencils) = 18 := by
  sorry

end largest_package_size_l1917_191713


namespace inverse_proportion_percentage_change_l1917_191742

theorem inverse_proportion_percentage_change 
  (x y p : ℝ) 
  (hx : x > 0) 
  (hy : y > 0) 
  (hp : p > 0) 
  (h_inverse : ∃ k, k > 0 ∧ x * y = k) :
  let x' := x * (1 + 2*p/100)
  let y' := y * 100 / (100 + 2*p)
  (y - y') / y * 100 = 200 * p / (100 + 2*p) := by
  sorry

end inverse_proportion_percentage_change_l1917_191742


namespace yellow_shirts_calculation_l1917_191714

/-- The number of yellow shirts in each pack -/
def yellow_shirts_per_pack : ℕ :=
  let black_packs : ℕ := 3
  let yellow_packs : ℕ := 3
  let black_shirts_per_pack : ℕ := 5
  let total_shirts : ℕ := 21
  (total_shirts - black_packs * black_shirts_per_pack) / yellow_packs

theorem yellow_shirts_calculation :
  yellow_shirts_per_pack = 2 := by
  sorry

end yellow_shirts_calculation_l1917_191714


namespace investment_growth_l1917_191740

def calculate_final_value (initial_investment : ℝ) : ℝ :=
  let year1_increase := 0.75
  let year2_decrease := 0.30
  let year3_increase := 0.45
  let year4_decrease := 0.15
  let tax_rate := 0.20
  let fee_rate := 0.02

  let year1_value := initial_investment * (1 + year1_increase)
  let year1_after_fees := year1_value * (1 - fee_rate)

  let year2_value := year1_after_fees * (1 - year2_decrease)
  let year2_after_fees := year2_value * (1 - fee_rate)

  let year3_value := year2_after_fees * (1 + year3_increase)
  let year3_after_fees := year3_value * (1 - fee_rate)

  let year4_value := year3_after_fees * (1 - year4_decrease)
  let year4_after_fees := year4_value * (1 - fee_rate)

  let capital_gains := year4_after_fees - initial_investment
  let taxes := capital_gains * tax_rate
  year4_after_fees - taxes

theorem investment_growth (initial_investment : ℝ) :
  initial_investment = 100 →
  calculate_final_value initial_investment = 131.408238206 := by
  sorry

end investment_growth_l1917_191740


namespace digit_fraction_statement_l1917_191765

theorem digit_fraction_statement : 
  ∃ (a b c : ℕ) (f g h : ℚ), 
    a ≠ b ∧ b ≠ c ∧ a ≠ c ∧
    a < 10 ∧ b < 10 ∧ c < 10 ∧
    f + 2 * g + h = 1 ∧
    f = 1/2 ∧
    g = 1/5 ∧
    h = 1/10 := by
  sorry

end digit_fraction_statement_l1917_191765


namespace inequality_solution_range_l1917_191799

theorem inequality_solution_range :
  ∀ a : ℝ, (∃ x : ℝ, |x + 1| - |x - 3| < a) ↔ a > -4 :=
by sorry

end inequality_solution_range_l1917_191799


namespace james_downhill_speed_l1917_191731

/-- Proves that James' speed on the downhill trail is 5 miles per hour given the problem conditions. -/
theorem james_downhill_speed :
  ∀ (v : ℝ),
    v > 0 →
    (20 : ℝ) / v = (12 : ℝ) / 3 + 1 - 1 →
    v = 5 := by
  sorry

end james_downhill_speed_l1917_191731


namespace min_value_expression_l1917_191756

theorem min_value_expression (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) 
  (h_sum : x + 2*y + z = 1) :
  ∃ (m : ℝ), m = 3 + 2*Real.sqrt 2 ∧ 
  ∀ x' y' z' : ℝ, x' > 0 → y' > 0 → z' > 0 → x' + 2*y' + z' = 1 → 
    (1 / (x' + y')) + (2 / (y' + z')) ≥ m :=
by sorry

end min_value_expression_l1917_191756


namespace final_value_less_than_original_l1917_191715

theorem final_value_less_than_original (p q M : ℝ) 
  (hp : p > 0) (hq : q > 0) (hq_upper : q < 50) (hM : M > 0) :
  M * (1 - p / 100) * (1 + q / 100) < M ↔ p > (100 * q - q^2) / 100 := by
  sorry

end final_value_less_than_original_l1917_191715


namespace decimal_places_relation_l1917_191743

/-- Represents a decimal number -/
structure Decimal where
  integerPart : ℤ
  fractionalPart : ℕ
  decimalPlaces : ℕ

/-- Represents the result of decimal multiplication -/
structure DecimalMultiplicationResult where
  product : Decimal
  factor1 : Decimal
  factor2 : Decimal

/-- Rules of decimal multiplication -/
axiom decimal_multiplication_rule (result : DecimalMultiplicationResult) :
  result.product.decimalPlaces = result.factor1.decimalPlaces + result.factor2.decimalPlaces

/-- Theorem: The number of decimal places in a product is related to the number of decimal places in its factors -/
theorem decimal_places_relation :
  ∃ (result : DecimalMultiplicationResult),
    result.product.decimalPlaces ≠ result.factor1.decimalPlaces ∨
    result.product.decimalPlaces ≠ result.factor2.decimalPlaces :=
  sorry

end decimal_places_relation_l1917_191743


namespace fenced_rectangle_fence_length_l1917_191777

/-- A rectangular region with a fence on three sides and a wall on the fourth -/
structure FencedRectangle where
  short_side : ℝ
  long_side : ℝ
  area : ℝ
  fence_length : ℝ

/-- Properties of the fenced rectangle -/
def is_valid_fenced_rectangle (r : FencedRectangle) : Prop :=
  r.long_side = 2 * r.short_side ∧
  r.area = r.short_side * r.long_side ∧
  r.fence_length = 2 * r.short_side + r.long_side

theorem fenced_rectangle_fence_length 
  (r : FencedRectangle) 
  (h : is_valid_fenced_rectangle r) 
  (area_eq : r.area = 200) : 
  r.fence_length = 40 := by
  sorry

end fenced_rectangle_fence_length_l1917_191777


namespace min_value_quadratic_l1917_191785

theorem min_value_quadratic (x y : ℝ) : 3 * x^2 + 3 * x * y + y^2 - 3 * x + 3 * y + 9 ≥ 39 / 4 ∧
  (3 * x^2 + 3 * x * y + y^2 - 3 * x + 3 * y + 9 = 39 / 4 ↔ x = 1 / 2 ∧ y = 0) :=
sorry

end min_value_quadratic_l1917_191785


namespace factor_polynomial_l1917_191774

theorem factor_polynomial (k : ℤ) : 
  (∀ x : ℝ, (x + k) ∣ (3 * x^2 + 14 * x + 8)) → k = 4 := by
  sorry

end factor_polynomial_l1917_191774


namespace sqrt_sum_squares_geq_arithmetic_mean_l1917_191706

theorem sqrt_sum_squares_geq_arithmetic_mean (a b : ℝ) :
  Real.sqrt ((a^2 + b^2) / 2) ≥ (a + b) / 2 ∧
  (Real.sqrt ((a^2 + b^2) / 2) = (a + b) / 2 ↔ a = b) := by
  sorry

end sqrt_sum_squares_geq_arithmetic_mean_l1917_191706


namespace range_of_f_l1917_191778

noncomputable def f (x : ℝ) : ℝ := (x^2 + 4*x + 3) / (x + 2)

theorem range_of_f :
  ∀ y : ℝ, ∃ x : ℝ, x ≠ -2 ∧ f x = y :=
sorry

end range_of_f_l1917_191778


namespace polynomial_C_value_l1917_191724

def polynomial (A B C D : ℤ) (x : ℝ) : ℝ := x^6 - 12*x^5 + A*x^4 + B*x^3 + C*x^2 + D*x + 36

theorem polynomial_C_value (A B C D : ℤ) :
  (∃ (r₁ r₂ r₃ r₄ r₅ r₆ : ℕ+), 
    (∀ x : ℝ, polynomial A B C D x = (x - r₁) * (x - r₂) * (x - r₃) * (x - r₄) * (x - r₅) * (x - r₆)) ∧
    (r₁ + r₂ + r₃ + r₄ + r₅ + r₆ = 12)) →
  C = -171 := by
sorry

end polynomial_C_value_l1917_191724


namespace fathers_age_l1917_191741

/-- Proves that the father's age is 30 given the conditions of the problem -/
theorem fathers_age (man_age : ℝ) (father_age : ℝ) : 
  man_age = (2/5) * father_age ∧ 
  man_age + 6 = (1/2) * (father_age + 6) → 
  father_age = 30 := by
  sorry

end fathers_age_l1917_191741


namespace olivia_math_problem_l1917_191793

theorem olivia_math_problem (x : ℝ) 
  (h1 : 7 * x + 3 = 31) : 
  x = 4 := by
sorry

end olivia_math_problem_l1917_191793


namespace arithmetic_sequence_first_term_l1917_191764

/-- Sum of first n terms of an arithmetic sequence -/
def S (a : ℚ) (n : ℕ) : ℚ := n * (2 * a + (n - 1) * 5) / 2

/-- The theorem states that if the ratio of S_{3n} to S_n is constant
    for all positive integers n, then the first term of the sequence is 5/2 -/
theorem arithmetic_sequence_first_term
  (h : ∃ (c : ℚ), ∀ (n : ℕ), n > 0 → S a (3 * n) / S a n = c) :
  a = 5 / 2 := by
  sorry

end arithmetic_sequence_first_term_l1917_191764


namespace lake_depth_for_specific_cone_l1917_191796

/-- Represents a conical hill partially submerged in a lake -/
structure SubmergedCone where
  total_height : ℝ
  volume_ratio_above_water : ℝ

/-- Calculates the depth of the lake at the base of a partially submerged conical hill -/
def lake_depth (cone : SubmergedCone) : ℝ :=
  cone.total_height * (1 - (1 - cone.volume_ratio_above_water) ^ (1/3))

theorem lake_depth_for_specific_cone :
  let cone : SubmergedCone := ⟨5000, 1/5⟩
  lake_depth cone = 660 := by
  sorry

end lake_depth_for_specific_cone_l1917_191796


namespace beads_per_package_l1917_191739

theorem beads_per_package (total_packages : Nat) (total_beads : Nat) : 
  total_packages = 8 → total_beads = 320 → (total_beads / total_packages : Nat) = 40 := by
  sorry

end beads_per_package_l1917_191739


namespace stream_speed_stream_speed_problem_l1917_191779

/-- The speed of a stream given a swimmer's still water speed and relative upstream/downstream times -/
theorem stream_speed (still_speed : ℝ) (upstream_downstream_ratio : ℝ) : ℝ :=
  let stream_speed := (still_speed * (upstream_downstream_ratio - 1)) / (upstream_downstream_ratio + 1)
  stream_speed

/-- Proves that the speed of the stream is 0.5 km/h given the conditions -/
theorem stream_speed_problem : stream_speed 1.5 2 = 0.5 := by
  sorry

end stream_speed_stream_speed_problem_l1917_191779
