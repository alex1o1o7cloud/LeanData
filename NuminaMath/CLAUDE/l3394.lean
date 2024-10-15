import Mathlib

namespace NUMINAMATH_CALUDE_fraction_division_l3394_339482

theorem fraction_division (x : ℚ) : 
  (37 + 1/2 : ℚ) = 450 * x → x = 1/12 ∧ (37 + 1/2 : ℚ) / x = 450 := by
  sorry

end NUMINAMATH_CALUDE_fraction_division_l3394_339482


namespace NUMINAMATH_CALUDE_tree_distance_l3394_339427

/-- Given a yard of length 180 meters with 11 trees planted at equal distances,
    with one tree at each end, the distance between two consecutive trees is 18 meters. -/
theorem tree_distance (yard_length : ℝ) (num_trees : ℕ) :
  yard_length = 180 →
  num_trees = 11 →
  let num_spaces := num_trees - 1
  yard_length / num_spaces = 18 :=
by sorry

end NUMINAMATH_CALUDE_tree_distance_l3394_339427


namespace NUMINAMATH_CALUDE_hyperbola_to_ellipse_l3394_339433

/-- Given a hyperbola with equation x²/4 - y²/12 = 1, prove that the ellipse with the foci of the
hyperbola as its vertices and the vertices of the hyperbola as its foci has the equation
x²/16 + y²/12 = 1 -/
theorem hyperbola_to_ellipse (x y : ℝ) :
  (x^2 / 4 - y^2 / 12 = 1) →
  ∃ (x' y' : ℝ), (x'^2 / 16 + y'^2 / 12 = 1 ∧
    (∀ (f_x f_y : ℝ), (f_x^2 / 4 - f_y^2 / 12 = 1 ∧ f_y = 0) →
      ((x' = f_x ∧ y' = 0) ∨ (x' = -f_x ∧ y' = 0))) ∧
    (∀ (v_x v_y : ℝ), (v_x^2 / 4 - v_y^2 / 12 = 1 ∧ v_y = 0) →
      (∃ (c : ℝ), x'^2 / 16 + y'^2 / 12 = 1 ∧ x'^2 - y'^2 = c^2 ∧ (v_x = c ∨ v_x = -c)))) :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_to_ellipse_l3394_339433


namespace NUMINAMATH_CALUDE_line_arrangement_count_l3394_339496

/-- The number of ways to arrange n distinct objects. -/
def factorial (n : ℕ) : ℕ := (List.range n).foldl (· * ·) 1

/-- The number of people in the line. -/
def totalPeople : ℕ := 7

/-- The number of people in the family that must stay together. -/
def familySize : ℕ := 3

/-- The number of individual entities to arrange (family counts as one entity). -/
def entities : ℕ := totalPeople - familySize + 1

/-- The number of ways to arrange the line of people with the family staying together. -/
def arrangements : ℕ := factorial entities * factorial familySize

theorem line_arrangement_count : arrangements = 720 := by sorry

end NUMINAMATH_CALUDE_line_arrangement_count_l3394_339496


namespace NUMINAMATH_CALUDE_nabla_computation_l3394_339449

-- Define the operation ∇
def nabla (x y : ℕ) : ℕ := x^3 - 2*y

-- State the theorem
theorem nabla_computation :
  (5^(nabla 7 4)) - 2*(2^(nabla 6 9)) = 5^1005 - 2^199 :=
by sorry

end NUMINAMATH_CALUDE_nabla_computation_l3394_339449


namespace NUMINAMATH_CALUDE_deficiency_and_excess_l3394_339422

theorem deficiency_and_excess (people : ℕ) (price : ℕ) : 
  (5 * people + 45 = price) →
  (7 * people + 3 = price) →
  (people = 21 ∧ price = 150) := by
  sorry

end NUMINAMATH_CALUDE_deficiency_and_excess_l3394_339422


namespace NUMINAMATH_CALUDE_phone_number_probability_correct_probability_l3394_339477

theorem phone_number_probability : ℝ → Prop :=
  fun p => (∀ n : ℕ, n ≤ 3 → n > 0 → (1 - (9/10)^n) ≤ p) ∧ p ≤ 3/10

theorem correct_probability : phone_number_probability (3/10) := by
  sorry

end NUMINAMATH_CALUDE_phone_number_probability_correct_probability_l3394_339477


namespace NUMINAMATH_CALUDE_max_container_weight_l3394_339409

def total_goods : ℕ := 1500
def num_platforms : ℕ := 25
def platform_capacity : ℕ := 80

def is_transportable (k : ℕ) : Prop :=
  ∀ (containers : List ℕ),
    (containers.sum = total_goods) →
    (∀ c ∈ containers, c ≤ k ∧ c > 0) →
    ∃ (loading : List (List ℕ)),
      loading.length ≤ num_platforms ∧
      (∀ platform ∈ loading, platform.sum ≤ platform_capacity) ∧
      loading.join.sum = total_goods

theorem max_container_weight :
  (∀ k ≤ 26, is_transportable k) ∧
  ¬(is_transportable 27) := by sorry

end NUMINAMATH_CALUDE_max_container_weight_l3394_339409


namespace NUMINAMATH_CALUDE_double_acute_angle_range_l3394_339472

theorem double_acute_angle_range (α : Real) (h : 0 < α ∧ α < π / 2) : 
  0 < 2 * α ∧ 2 * α < π := by
  sorry

end NUMINAMATH_CALUDE_double_acute_angle_range_l3394_339472


namespace NUMINAMATH_CALUDE_perpendicular_lines_l3394_339465

-- Define the slopes of the lines
def m1 : ℚ := 3/4
def m2 : ℚ := -3/4
def m3 : ℚ := -3/4
def m4 : ℚ := -4/3

-- Define a function to check if two lines are perpendicular
def are_perpendicular (m1 m2 : ℚ) : Prop := m1 * m2 = -1

-- Theorem statement
theorem perpendicular_lines :
  (are_perpendicular m1 m4) ∧
  (¬ are_perpendicular m1 m2) ∧
  (¬ are_perpendicular m1 m3) ∧
  (¬ are_perpendicular m2 m3) ∧
  (¬ are_perpendicular m2 m4) ∧
  (¬ are_perpendicular m3 m4) :=
sorry

end NUMINAMATH_CALUDE_perpendicular_lines_l3394_339465


namespace NUMINAMATH_CALUDE_positive_real_solution_l3394_339456

theorem positive_real_solution (a b : ℝ) (h1 : a > 0) (h2 : b > 0) 
  (h3 : a^b = b^a) (h4 : b = 4*a) : a = Real.rpow 4 (1/3) := by
  sorry

end NUMINAMATH_CALUDE_positive_real_solution_l3394_339456


namespace NUMINAMATH_CALUDE_min_perimeter_triangle_l3394_339481

/-- Represents a triangle with integer side lengths -/
structure Triangle where
  a : ℕ  -- side EF
  b : ℕ  -- side DE and DF
  h : a = 2 * b  -- EF is twice the length of DE and DF

/-- Represents a circle -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- Represents an excircle of a triangle -/
structure Excircle where
  center : ℝ × ℝ
  radius : ℝ

/-- The incenter of a triangle -/
def incenter (t : Triangle) : ℝ × ℝ := sorry

/-- The incircle of a triangle -/
def incircle (t : Triangle) : Circle := sorry

/-- The excircle of a triangle opposite to side EF -/
def excircle_EF (t : Triangle) : Excircle := sorry

/-- The excircles of a triangle opposite to sides DE and DF -/
def excircles_DE_DF (t : Triangle) : Excircle × Excircle := sorry

/-- Checks if two circles are internally tangent -/
def internally_tangent (c1 c2 : Circle) : Prop := sorry

/-- Checks if two circles are externally tangent -/
def externally_tangent (c1 c2 : Circle) : Prop := sorry

/-- The main theorem -/
theorem min_perimeter_triangle (t : Triangle) :
  let χ : Circle := incircle t
  let exc_EF : Excircle := excircle_EF t
  let (exc_DE, exc_DF) := excircles_DE_DF t
  (internally_tangent ⟨exc_EF.center, exc_EF.radius⟩ χ) ∧
  (externally_tangent ⟨exc_DE.center, exc_DE.radius⟩ χ) ∧
  (externally_tangent ⟨exc_DF.center, exc_DF.radius⟩ χ) →
  t.a + 2 * t.b ≥ 40 :=
sorry

end NUMINAMATH_CALUDE_min_perimeter_triangle_l3394_339481


namespace NUMINAMATH_CALUDE_kelsey_sister_age_difference_l3394_339498

/-- Represents the age difference between Kelsey and her older sister -/
def age_difference (kelsey_birth_year : ℕ) (sister_birth_year : ℕ) : ℕ :=
  kelsey_birth_year - sister_birth_year

theorem kelsey_sister_age_difference :
  ∀ (kelsey_birth_year : ℕ) (sister_birth_year : ℕ),
  kelsey_birth_year + 25 = 1999 →
  sister_birth_year + 50 = 2021 →
  age_difference kelsey_birth_year sister_birth_year = 3 := by
  sorry

end NUMINAMATH_CALUDE_kelsey_sister_age_difference_l3394_339498


namespace NUMINAMATH_CALUDE_correct_quadratic_equation_l3394_339434

/-- The correct quadratic equation given the conditions of the problem -/
theorem correct_quadratic_equation :
  ∀ (b c : ℝ),
  (∀ x : ℝ, x^2 + b*x + c = 0 ↔ x = 5 ∨ x = 3 ∨ x = -6 ∨ x = -4) →
  (5 + 3 = -(b)) →
  ((-6) * (-4) = c) →
  (∀ x : ℝ, x^2 - 8*x + 24 = 0 ↔ x^2 + b*x + c = 0) :=
by sorry

end NUMINAMATH_CALUDE_correct_quadratic_equation_l3394_339434


namespace NUMINAMATH_CALUDE_bakery_rolls_combinations_l3394_339442

theorem bakery_rolls_combinations :
  let total_rolls : ℕ := 8
  let num_kinds : ℕ := 4
  let rolls_to_distribute : ℕ := total_rolls - num_kinds
  (Nat.choose (rolls_to_distribute + num_kinds - 1) (num_kinds - 1)) = 35 := by
  sorry

end NUMINAMATH_CALUDE_bakery_rolls_combinations_l3394_339442


namespace NUMINAMATH_CALUDE_correct_regression_equation_l3394_339474

/-- Represents a linear regression equation -/
structure LinearRegression where
  slope : ℝ
  intercept : ℝ

/-- Checks if a linear regression equation passes through a given point -/
def passes_through (eq : LinearRegression) (x y : ℝ) : Prop :=
  eq.slope * x + eq.intercept = y

/-- Represents the properties of the given data -/
structure DataProperties where
  x_mean : ℝ
  y_mean : ℝ
  positively_correlated : Prop

theorem correct_regression_equation 
  (data : DataProperties)
  (h_x_mean : data.x_mean = 2.4)
  (h_y_mean : data.y_mean = 3.2)
  (h_corr : data.positively_correlated) :
  ∃ (eq : LinearRegression), 
    eq.slope = 0.5 ∧ 
    eq.intercept = 2 ∧ 
    passes_through eq data.x_mean data.y_mean :=
sorry

end NUMINAMATH_CALUDE_correct_regression_equation_l3394_339474


namespace NUMINAMATH_CALUDE_peanuts_in_box_l3394_339455

/-- The number of peanuts in a box after adding more -/
def total_peanuts (initial : ℕ) (added : ℕ) : ℕ :=
  initial + added

/-- Theorem: The total number of peanuts is 10 when starting with 4 and adding 6 -/
theorem peanuts_in_box : total_peanuts 4 6 = 10 := by
  sorry

end NUMINAMATH_CALUDE_peanuts_in_box_l3394_339455


namespace NUMINAMATH_CALUDE_trajectory_equation_l3394_339452

/-- Given a fixed point A(1,2) and a moving point P(x,y), if the projection of vector OP on vector OA is -√5,
    then the equation x + 2y + 5 = 0 represents the trajectory of point P. -/
theorem trajectory_equation (x y : ℝ) : 
  let A : ℝ × ℝ := (1, 2)
  let P : ℝ × ℝ := (x, y)
  let OA : ℝ × ℝ := A
  let OP : ℝ × ℝ := P
  (OP.1 * OA.1 + OP.2 * OA.2) / Real.sqrt (OA.1^2 + OA.2^2) = -Real.sqrt 5 →
  x + 2*y + 5 = 0 :=
by sorry

end NUMINAMATH_CALUDE_trajectory_equation_l3394_339452


namespace NUMINAMATH_CALUDE_easter_egg_hunt_l3394_339437

/-- The number of eggs found by Cheryl exceeds the combined total of eggs found by Kevin, Bonnie, and George by 29. -/
theorem easter_egg_hunt (kevin bonnie george cheryl : ℕ) 
  (h1 : kevin = 5) 
  (h2 : bonnie = 13) 
  (h3 : george = 9) 
  (h4 : cheryl = 56) : 
  cheryl - (kevin + bonnie + george) = 29 := by
  sorry

end NUMINAMATH_CALUDE_easter_egg_hunt_l3394_339437


namespace NUMINAMATH_CALUDE_tangent_line_b_value_l3394_339479

/-- A line tangent to a cubic curve -/
structure TangentLine where
  k : ℝ
  a : ℝ
  b : ℝ

/-- The tangent line y = kx + 1 is tangent to the curve y = x^3 + ax + b at the point (1, 3) -/
def is_tangent (t : TangentLine) : Prop :=
  3 = t.k * 1 + 1 ∧
  3 = 1^3 + t.a * 1 + t.b ∧
  t.k = 3 * 1^2 + t.a

theorem tangent_line_b_value (t : TangentLine) (h : is_tangent t) : t.b = 3 := by
  sorry

#check tangent_line_b_value

end NUMINAMATH_CALUDE_tangent_line_b_value_l3394_339479


namespace NUMINAMATH_CALUDE_min_distance_log_circle_l3394_339429

theorem min_distance_log_circle (e : ℝ) (h : e > 0) :
  let f := fun x : ℝ => Real.log x
  let circle := fun (x y : ℝ) => (x - (e + 1/e))^2 + y^2 = 1/4
  ∃ (min_dist : ℝ),
    (∀ (x₁ y₁ x₂ y₂ : ℝ), f x₁ = y₁ → circle x₂ y₂ →
      min_dist ≤ Real.sqrt ((x₁ - x₂)^2 + (y₁ - y₂)^2)) ∧
    min_dist = (2 * Real.sqrt (e^2 + 1) - e) / (2 * e) :=
by sorry

end NUMINAMATH_CALUDE_min_distance_log_circle_l3394_339429


namespace NUMINAMATH_CALUDE_cloth_weaving_problem_l3394_339488

theorem cloth_weaving_problem (a₁ a₃₀ n : ℝ) (h1 : a₁ = 5) (h2 : a₃₀ = 1) (h3 : n = 30) :
  n / 2 * (a₁ + a₃₀) = 90 := by
  sorry

end NUMINAMATH_CALUDE_cloth_weaving_problem_l3394_339488


namespace NUMINAMATH_CALUDE_soccer_games_per_month_l3394_339421

/-- Given a total number of games and number of months in a season,
    calculate the number of games per month assuming equal distribution -/
def games_per_month (total_games : ℕ) (num_months : ℕ) : ℕ :=
  total_games / num_months

/-- Theorem: For 27 games over 3 months, there are 9 games per month -/
theorem soccer_games_per_month :
  games_per_month 27 3 = 9 := by
  sorry

end NUMINAMATH_CALUDE_soccer_games_per_month_l3394_339421


namespace NUMINAMATH_CALUDE_green_pepper_weight_l3394_339497

def hannah_peppers (total_weight red_weight green_weight : Real) : Prop :=
  total_weight = 0.66 ∧ 
  red_weight = 0.33 ∧ 
  green_weight = total_weight - red_weight

theorem green_pepper_weight : 
  ∀ (total_weight red_weight green_weight : Real),
  hannah_peppers total_weight red_weight green_weight →
  green_weight = 0.33 :=
by
  sorry

end NUMINAMATH_CALUDE_green_pepper_weight_l3394_339497


namespace NUMINAMATH_CALUDE_expected_value_Y_l3394_339494

/-- A random variable following a binomial distribution -/
structure BinomialRV (n : ℕ) (p : ℝ) where
  X : ℝ → ℝ

/-- The expected value of a binomial random variable -/
def expectedValue (X : BinomialRV n p) : ℝ := n * p

/-- Two random variables X and Y satisfying X + Y = 8, where X follows B(10, 0.6) -/
structure RandomVariables where
  X : BinomialRV 10 0.6
  Y : ℝ → ℝ
  sum_constraint : ∀ ω, X.X ω + Y ω = 8

/-- The theorem stating that E(Y) = 2 -/
theorem expected_value_Y (rv : RandomVariables) : 
  ∃ (E_Y : ℝ → ℝ), (∀ ω, E_Y ω = rv.Y ω) ∧ (∀ ω, E_Y ω = 2) :=
sorry

end NUMINAMATH_CALUDE_expected_value_Y_l3394_339494


namespace NUMINAMATH_CALUDE_log_cutting_l3394_339473

theorem log_cutting (fallen_pieces fixed_pieces : ℕ) 
  (h1 : fallen_pieces = 10)
  (h2 : fixed_pieces = 2) :
  fallen_pieces + fixed_pieces - 1 = 11 := by
sorry

end NUMINAMATH_CALUDE_log_cutting_l3394_339473


namespace NUMINAMATH_CALUDE_town_friendship_theorem_l3394_339459

structure Town where
  inhabitants : Set Nat
  friendship : inhabitants → inhabitants → Prop
  enemy : inhabitants → inhabitants → Prop

def Town.canBecomeFriends (t : Town) : Prop :=
  ∃ (steps : ℕ), ∀ (a b : t.inhabitants), t.friendship a b

theorem town_friendship_theorem (t : Town) 
  (h1 : ∀ (a b : t.inhabitants), t.friendship a b ∨ t.enemy a b)
  (h2 : ∀ (a b c : t.inhabitants), t.friendship a b → t.friendship b c → t.friendship a c)
  (h3 : ∀ (a b c : t.inhabitants), t.friendship a b ∨ t.friendship a c ∨ t.friendship b c)
  (h4 : ∀ (day : ℕ), ∃ (a : t.inhabitants), 
    ∀ (b : t.inhabitants), 
      (t.friendship a b → t.enemy a b) ∧ 
      (t.enemy a b → t.friendship a b)) :
  t.canBecomeFriends :=
sorry

end NUMINAMATH_CALUDE_town_friendship_theorem_l3394_339459


namespace NUMINAMATH_CALUDE_tax_revenue_consumption_relation_l3394_339464

/-- Proves that a 40% tax reduction and 25% revenue decrease results in a 25% consumption increase -/
theorem tax_revenue_consumption_relation 
  (T : ℝ) -- Original tax rate
  (C : ℝ) -- Original consumption
  (h1 : T > 0) -- Assumption: Original tax rate is positive
  (h2 : C > 0) -- Assumption: Original consumption is positive
  : 
  let new_tax := 0.6 * T -- New tax rate after 40% reduction
  let new_revenue := 0.75 * T * C -- New revenue after 25% decrease
  let new_consumption := new_revenue / new_tax -- New consumption
  new_consumption = 1.25 * C -- Proves 25% increase in consumption
  := by sorry

end NUMINAMATH_CALUDE_tax_revenue_consumption_relation_l3394_339464


namespace NUMINAMATH_CALUDE_doubled_added_tripled_l3394_339400

theorem doubled_added_tripled (y : ℝ) : 3 * (2 * 7 + y) = 69 → y = 9 := by
  sorry

end NUMINAMATH_CALUDE_doubled_added_tripled_l3394_339400


namespace NUMINAMATH_CALUDE_mary_flour_amount_l3394_339438

/-- The amount of flour Mary uses in her cake recipe -/
def flour_recipe : ℝ := 7.0

/-- The extra amount of flour Mary adds -/
def flour_extra : ℝ := 2.0

/-- The total amount of flour Mary uses -/
def flour_total : ℝ := flour_recipe + flour_extra

theorem mary_flour_amount : flour_total = 9.0 := by
  sorry

end NUMINAMATH_CALUDE_mary_flour_amount_l3394_339438


namespace NUMINAMATH_CALUDE_fruit_arrangement_count_l3394_339407

def number_of_arrangements (a o b g : ℕ) : ℕ :=
  Nat.factorial 14 / (Nat.factorial a * Nat.factorial o * Nat.factorial b * Nat.factorial g)

theorem fruit_arrangement_count :
  number_of_arrangements 4 3 3 4 = 4204200 :=
by
  sorry

#eval number_of_arrangements 4 3 3 4

end NUMINAMATH_CALUDE_fruit_arrangement_count_l3394_339407


namespace NUMINAMATH_CALUDE_total_ages_is_56_l3394_339453

/-- Given Craig's age and the age difference with his mother, calculate the total of their ages -/
def total_ages (craig_age : ℕ) (age_difference : ℕ) : ℕ :=
  craig_age + (craig_age + age_difference)

/-- Theorem: The total of Craig and his mother's ages is 56 years -/
theorem total_ages_is_56 : total_ages 16 24 = 56 := by
  sorry

end NUMINAMATH_CALUDE_total_ages_is_56_l3394_339453


namespace NUMINAMATH_CALUDE_shower_tasks_count_l3394_339469

/-- The number of tasks to clean the house -/
def clean_house_tasks : ℕ := 7

/-- The number of tasks to make dinner -/
def make_dinner_tasks : ℕ := 4

/-- The time each task takes in minutes -/
def time_per_task : ℕ := 10

/-- The total time to complete all tasks in hours -/
def total_time_hours : ℕ := 2

/-- The number of minutes in an hour -/
def minutes_per_hour : ℕ := 60

theorem shower_tasks_count : 
  (clean_house_tasks + make_dinner_tasks + 1) * time_per_task = total_time_hours * minutes_per_hour := by
  sorry

end NUMINAMATH_CALUDE_shower_tasks_count_l3394_339469


namespace NUMINAMATH_CALUDE_no_valid_a_exists_l3394_339476

theorem no_valid_a_exists : ¬ ∃ (a n : ℕ), 
  a > 1 ∧ 
  n > 0 ∧ 
  ∃ (k : ℕ), a * (10^n + 1) = k * a^2 := by
sorry

end NUMINAMATH_CALUDE_no_valid_a_exists_l3394_339476


namespace NUMINAMATH_CALUDE_max_enclosure_area_l3394_339491

/-- The number of fence pieces --/
def num_pieces : ℕ := 15

/-- The length of each fence piece in meters --/
def piece_length : ℝ := 2

/-- The total length of fencing available in meters --/
def total_length : ℝ := num_pieces * piece_length

/-- The area of the rectangular enclosure as a function of its width --/
def area (w : ℝ) : ℝ := (total_length - 2 * w) * w

/-- The maximum area of the enclosure, rounded down to the nearest integer --/
def max_area : ℕ := 112

theorem max_enclosure_area :
  ∃ (w : ℝ), 0 < w ∧ w < total_length / 2 ∧
  (∀ (x : ℝ), 0 < x → x < total_length / 2 → area x ≤ area w) ∧
  ⌊area w⌋ = max_area :=
sorry

end NUMINAMATH_CALUDE_max_enclosure_area_l3394_339491


namespace NUMINAMATH_CALUDE_max_area_of_cut_triangle_l3394_339416

/-- Triangle ABC with side lengths -/
structure Triangle :=
  (AB : ℝ)
  (BC : ℝ)
  (CA : ℝ)

/-- The given triangle -/
def givenTriangle : Triangle :=
  { AB := 13, BC := 14, CA := 15 }

/-- A line cutting the triangle -/
structure CuttingLine :=
  (intersectsSide1 : ℝ)
  (intersectsSide2 : ℝ)

/-- The area of the triangle formed by the cutting line -/
def areaOfCutTriangle (t : Triangle) (l : CuttingLine) : ℝ :=
  sorry

/-- The perimeter of the triangle formed by the cutting line -/
def perimeterOfCutTriangle (t : Triangle) (l : CuttingLine) : ℝ :=
  sorry

/-- The perimeter of the quadrilateral formed by the cutting line -/
def perimeterOfCutQuadrilateral (t : Triangle) (l : CuttingLine) : ℝ :=
  sorry

/-- The theorem to be proved -/
theorem max_area_of_cut_triangle (t : Triangle) :
  t = givenTriangle →
  ∃ (l : CuttingLine),
    perimeterOfCutTriangle t l = perimeterOfCutQuadrilateral t l ∧
    ∀ (l' : CuttingLine),
      perimeterOfCutTriangle t l' = perimeterOfCutQuadrilateral t l' →
      areaOfCutTriangle t l' ≤ 1323 / 26 :=
sorry

end NUMINAMATH_CALUDE_max_area_of_cut_triangle_l3394_339416


namespace NUMINAMATH_CALUDE_nuts_cost_to_age_ratio_l3394_339486

/-- The ratio of the cost of a pack of nuts to Betty's age -/
theorem nuts_cost_to_age_ratio : 
  ∀ (doug_age betty_age : ℕ) (num_packs total_cost : ℕ),
  doug_age = 40 →
  doug_age + betty_age = 90 →
  num_packs = 20 →
  total_cost = 2000 →
  (total_cost / num_packs : ℚ) / betty_age = 2 :=
by
  sorry

end NUMINAMATH_CALUDE_nuts_cost_to_age_ratio_l3394_339486


namespace NUMINAMATH_CALUDE_subtracted_value_l3394_339447

theorem subtracted_value (n : ℕ) (x : ℕ) (h1 : n = 121) (h2 : 2 * n - x = 104) : x = 138 := by
  sorry

end NUMINAMATH_CALUDE_subtracted_value_l3394_339447


namespace NUMINAMATH_CALUDE_symmetry_point_l3394_339414

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Checks if a point is the midpoint of two other points -/
def isMidpoint (mid : Point) (p1 : Point) (p2 : Point) : Prop :=
  mid.x = (p1.x + p2.x) / 2 ∧ mid.y = (p1.y + p2.y) / 2

theorem symmetry_point (m n : ℝ) :
  let M : Point := ⟨4, m⟩
  let N : Point := ⟨n, -3⟩
  let P : Point := ⟨6, -9⟩
  isMidpoint N M P → m = 3 ∧ n = 5 := by
  sorry

end NUMINAMATH_CALUDE_symmetry_point_l3394_339414


namespace NUMINAMATH_CALUDE_basketball_lineup_combinations_l3394_339490

/-- The number of ways to choose a starting lineup from a basketball team -/
def number_of_lineups (total_players : ℕ) (center_players : ℕ) (lineup_size : ℕ) : ℕ :=
  center_players * (total_players - 1) * (total_players - 2) * (total_players - 3)

/-- Theorem stating that for a team of 12 players with 4 centers, there are 3960 ways to choose a starting lineup of 4 players -/
theorem basketball_lineup_combinations :
  number_of_lineups 12 4 4 = 3960 := by
  sorry

end NUMINAMATH_CALUDE_basketball_lineup_combinations_l3394_339490


namespace NUMINAMATH_CALUDE_circle_parabola_intersection_l3394_339463

theorem circle_parabola_intersection (b : ℝ) : 
  (∃ (a : ℝ), -- center of the circle (a, b)
    (∃ (r : ℝ), r > 0 ∧ -- radius of the circle
      (∀ (x y : ℝ), (x - a)^2 + (y - b)^2 = r^2 → -- equation of the circle
        ((y = 3/4 * x^2) ∨ (x = 0 ∧ y = 0) ∨ (y = 3/4 * x + b)) -- intersections
      ) ∧
      (∃ (x1 x2 : ℝ), x1 ≠ x2 ∧ -- two distinct intersection points
        (3/4 * x1^2 = 3/4 * x1 + b) ∧ 
        (3/4 * x2^2 = 3/4 * x2 + b)
      )
    )
  ) → b = 25/12 :=
by sorry

end NUMINAMATH_CALUDE_circle_parabola_intersection_l3394_339463


namespace NUMINAMATH_CALUDE_min_value_expression_l3394_339436

theorem min_value_expression (x y : ℝ) (hx : x > 0) (hy : y > 0) :
  (Real.sqrt ((2 * x^2 + y^2) * (4 * x^2 + y^2))) / (x * y) ≥ 2 * Real.rpow 3 (1/4) :=
sorry

end NUMINAMATH_CALUDE_min_value_expression_l3394_339436


namespace NUMINAMATH_CALUDE_raviraj_cycled_20km_l3394_339485

/-- Represents a point in a 2D coordinate system -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents Raviraj's cycling journey -/
def raviraj_journey (final_distance : ℝ) : Prop :=
  ∃ (home : Point) (last_turn : Point) (final : Point),
    -- Initial movements
    last_turn.x = home.x - 10 ∧
    last_turn.y = home.y ∧
    -- Final position
    final.x = last_turn.x - 20 ∧
    final.y = last_turn.y ∧
    -- Distance to home is 30 km
    (final.x - home.x)^2 + (final.y - home.y)^2 = final_distance^2

/-- The theorem stating that Raviraj cycled 20 km after the third turn -/
theorem raviraj_cycled_20km : raviraj_journey 30 → 20 = 20 := by
  sorry

end NUMINAMATH_CALUDE_raviraj_cycled_20km_l3394_339485


namespace NUMINAMATH_CALUDE_geometric_sequence_problem_l3394_339417

-- Define a geometric sequence
def is_geometric (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

-- Define the problem statement
theorem geometric_sequence_problem (a : ℕ → ℝ) 
  (h_geometric : is_geometric a) 
  (h_product : a 2 * a 10 = 4)
  (h_sum_positive : a 2 + a 10 > 0) :
  a 6 = 2 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_problem_l3394_339417


namespace NUMINAMATH_CALUDE_max_xy_on_line_AB_l3394_339435

/-- Given points A(3,0) and B(0,4), prove that the maximum value of xy for any point P(x,y) on the line AB is 3. -/
theorem max_xy_on_line_AB :
  let A : ℝ × ℝ := (3, 0)
  let B : ℝ × ℝ := (0, 4)
  let line_AB (x : ℝ) := -4/3 * x + 4
  ∀ x y : ℝ, y = line_AB x → x * y ≤ 3 :=
by sorry

end NUMINAMATH_CALUDE_max_xy_on_line_AB_l3394_339435


namespace NUMINAMATH_CALUDE_min_sum_products_l3394_339450

theorem min_sum_products (m n : ℕ) : 
  (m * (m - 1)) / ((m + n) * (m + n - 1)) = 1 / 2 →
  m ≥ 1 →
  n ≥ 1 →
  m + n ≥ 4 := by
  sorry

end NUMINAMATH_CALUDE_min_sum_products_l3394_339450


namespace NUMINAMATH_CALUDE_quadratic_composition_theorem_l3394_339468

/-- A quadratic polynomial is a polynomial of degree 2 -/
def QuadraticPolynomial (R : Type*) [CommRing R] := {p : Polynomial R // p.degree = 2}

theorem quadratic_composition_theorem {R : Type*} [CommRing R] :
  ∀ (f : QuadraticPolynomial R),
  ∃ (g h : QuadraticPolynomial R),
  (f.val * (f.val.comp (Polynomial.X + 1))) = g.val.comp h.val :=
sorry

end NUMINAMATH_CALUDE_quadratic_composition_theorem_l3394_339468


namespace NUMINAMATH_CALUDE_demand_proportion_for_constant_income_l3394_339492

theorem demand_proportion_for_constant_income
  (original_price original_demand : ℝ)
  (price_increase_factor : ℝ := 1.20)
  (demand_increase_factor : ℝ := 1.12)
  (h_price_positive : original_price > 0)
  (h_demand_positive : original_demand > 0) :
  let new_price := price_increase_factor * original_price
  let new_demand := (14 / 15) * original_demand
  new_price * new_demand = original_price * original_demand :=
by sorry

end NUMINAMATH_CALUDE_demand_proportion_for_constant_income_l3394_339492


namespace NUMINAMATH_CALUDE_trig_identity_l3394_339431

theorem trig_identity (x y : ℝ) :
  Real.sin x ^ 2 + Real.sin (x + y) ^ 2 + 2 * Real.sin x * Real.sin y * Real.sin (x + y) =
  2 - Real.cos x ^ 2 - Real.cos (x + y) ^ 2 := by
  sorry

end NUMINAMATH_CALUDE_trig_identity_l3394_339431


namespace NUMINAMATH_CALUDE_largest_n_for_equation_l3394_339440

theorem largest_n_for_equation : 
  (∃ n : ℕ, 
    (∀ m : ℕ, m > n → 
      ¬∃ x y z : ℕ+, m^2 = x^2 + y^2 + z^2 + 2*x*y + 2*y*z + 2*z*x + 5*x + 5*y + 5*z - 10) ∧
    (∃ x y z : ℕ+, n^2 = x^2 + y^2 + z^2 + 2*x*y + 2*y*z + 2*z*x + 5*x + 5*y + 5*z - 10)) ∧
  (∀ n : ℕ, 
    (∀ m : ℕ, m > n → 
      ¬∃ x y z : ℕ+, m^2 = x^2 + y^2 + z^2 + 2*x*y + 2*y*z + 2*z*x + 5*x + 5*y + 5*z - 10) ∧
    (∃ x y z : ℕ+, n^2 = x^2 + y^2 + z^2 + 2*x*y + 2*y*z + 2*z*x + 5*x + 5*y + 5*z - 10) →
    n = 4) :=
by sorry

end NUMINAMATH_CALUDE_largest_n_for_equation_l3394_339440


namespace NUMINAMATH_CALUDE_x_over_y_value_l3394_339445

theorem x_over_y_value (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) 
  (h : (5*x + y)^2019 + x^2019 + 30*x + 5*y = 0) : 
  x / y = -1 / 6 := by
sorry

end NUMINAMATH_CALUDE_x_over_y_value_l3394_339445


namespace NUMINAMATH_CALUDE_sqrt_equation_l3394_339426

theorem sqrt_equation (n : ℝ) : Real.sqrt (8 + n) = 9 → n = 73 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_equation_l3394_339426


namespace NUMINAMATH_CALUDE_group_average_l3394_339439

theorem group_average (x : ℝ) : 
  (5 + 5 + x + 6 + 8) / 5 = 6 → x = 6 := by
sorry

end NUMINAMATH_CALUDE_group_average_l3394_339439


namespace NUMINAMATH_CALUDE_lingling_tourist_growth_l3394_339405

/-- The average annual growth rate of tourists visiting Lingling Ancient City from 2018 to 2020 -/
def average_growth_rate : ℝ := 0.125

/-- The number of tourists (in millions) visiting Lingling Ancient City in 2018 -/
def tourists_2018 : ℝ := 6.4

/-- The number of tourists (in millions) visiting Lingling Ancient City in 2020 -/
def tourists_2020 : ℝ := 8.1

/-- The time period in years -/
def years : ℕ := 2

theorem lingling_tourist_growth :
  tourists_2018 * (1 + average_growth_rate) ^ years = tourists_2020 := by
  sorry

end NUMINAMATH_CALUDE_lingling_tourist_growth_l3394_339405


namespace NUMINAMATH_CALUDE_smartphone_price_l3394_339480

theorem smartphone_price :
  ∀ (S : ℝ),
  (∃ (PC Tablet : ℝ),
    PC = S + 500 ∧
    Tablet = S + (S + 500) ∧
    S + PC + Tablet = 2200) →
  S = 300 := by
sorry

end NUMINAMATH_CALUDE_smartphone_price_l3394_339480


namespace NUMINAMATH_CALUDE_k_equals_p_l3394_339461

theorem k_equals_p (k p : ℕ) : 
  (∃ (nums_k : Finset ℕ) (nums_p : Finset ℕ), 
    (Finset.card nums_k = k) ∧ 
    (Finset.card nums_p = p) ∧
    (∀ x ∈ nums_k, x = 2*p + 3) ∧
    (∀ y ∈ nums_p, y = 5 - 2*k) ∧
    ((Finset.sum nums_k id + Finset.sum nums_p id) / (k + p : ℝ) = 4)) →
  k = p :=
sorry

end NUMINAMATH_CALUDE_k_equals_p_l3394_339461


namespace NUMINAMATH_CALUDE_midpoint_property_l3394_339462

/-- Given two points D and E in the plane, and F as their midpoint, 
    prove that 2x - 4y = 14 where F = (x, y) -/
theorem midpoint_property (D E F : ℝ × ℝ) : 
  D = (30, 10) →
  E = (6, 1) →
  F = ((D.1 + E.1) / 2, (D.2 + E.2) / 2) →
  2 * F.1 - 4 * F.2 = 14 := by
sorry

end NUMINAMATH_CALUDE_midpoint_property_l3394_339462


namespace NUMINAMATH_CALUDE_negations_universal_and_true_l3394_339454

-- Define the propositions
def prop_A (x : ℝ) := x^2 - x + 1/4 < 0
def prop_C (x : ℝ) := x^2 + 2*x + 2 ≤ 0
def prop_D (x : ℝ) := x^3 + 1 = 0

-- Define the negations
def neg_A := ∀ x : ℝ, ¬(prop_A x)
def neg_C := ∀ x : ℝ, ¬(prop_C x)
def neg_D := ∀ x : ℝ, ¬(prop_D x)

-- Theorem statement
theorem negations_universal_and_true :
  (neg_A ∧ neg_C) ∧ 
  (∃ x : ℝ, prop_D x) :=
sorry

end NUMINAMATH_CALUDE_negations_universal_and_true_l3394_339454


namespace NUMINAMATH_CALUDE_cricketer_average_difference_is_13_l3394_339443

def cricketer_average_difference (runs_A runs_B : ℕ) (innings_A innings_B : ℕ) 
  (increase_A increase_B : ℚ) : ℚ :=
  let avg_A : ℚ := (runs_A : ℚ) / (innings_A : ℚ)
  let avg_B : ℚ := (runs_B : ℚ) / (innings_B : ℚ)
  (avg_B + increase_B) - (avg_A + increase_A)

theorem cricketer_average_difference_is_13 :
  cricketer_average_difference 125 145 20 18 5 6 = 13 := by
  sorry

end NUMINAMATH_CALUDE_cricketer_average_difference_is_13_l3394_339443


namespace NUMINAMATH_CALUDE_problem_solution_l3394_339419

def A (x y : ℝ) : ℝ := 3 * x^2 + 2 * x * y - 2 * x - 1
def B (x y : ℝ) : ℝ := -x^2 + x * y - 1

theorem problem_solution (x y : ℝ) :
  (A x y + 3 * B x y = 5 * x * y - 2 * x - 4) ∧
  (∀ x, A x y + 3 * B x y = A 0 y + 3 * B 0 y → y = 2/5) := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l3394_339419


namespace NUMINAMATH_CALUDE_not_solution_and_solutions_l3394_339432

def is_solution (x y : ℤ) : Prop := 85 * x - 324 * y = 101

theorem not_solution_and_solutions :
  ¬(is_solution 978 256) ∧
  is_solution 5 1 ∧
  is_solution 329 86 ∧
  is_solution 653 171 ∧
  is_solution 1301 341 := by
  sorry

end NUMINAMATH_CALUDE_not_solution_and_solutions_l3394_339432


namespace NUMINAMATH_CALUDE_tangent_line_equation_l3394_339470

-- Define the circle C
def C (x y : ℝ) : Prop := (x - 1)^2 + (y - 2)^2 = 2

-- Define the point P
def P : ℝ × ℝ := (2, -1)

-- Define the line equation
def line_equation (x y : ℝ) : Prop := x - 3*y + 3 = 0

-- Theorem statement
theorem tangent_line_equation :
  ∃ (A B : ℝ × ℝ),
    C A.1 A.2 ∧ C B.1 B.2 ∧
    (∀ (x y : ℝ), C x y → (x - P.1)*(y - P.2) = (A.1 - P.1)*(A.2 - P.2) ∨ (x - P.1)*(y - P.2) = (B.1 - P.1)*(B.2 - P.2)) →
    (∀ (x y : ℝ), line_equation x y ↔ (∃ t : ℝ, x = A.1 + t*(B.1 - A.1) ∧ y = A.2 + t*(B.2 - A.2))) :=
sorry


end NUMINAMATH_CALUDE_tangent_line_equation_l3394_339470


namespace NUMINAMATH_CALUDE_total_fruits_is_236_l3394_339411

/-- The total number of fruits picked by Sara and Sally -/
def total_fruits (sara_pears sara_apples sara_plums sally_pears sally_apples sally_plums : ℕ) : ℕ :=
  (sara_pears + sally_pears) + (sara_apples + sally_apples) + (sara_plums + sally_plums)

/-- Theorem: The total number of fruits picked by Sara and Sally is 236 -/
theorem total_fruits_is_236 :
  total_fruits 45 22 64 11 38 56 = 236 := by
  sorry

end NUMINAMATH_CALUDE_total_fruits_is_236_l3394_339411


namespace NUMINAMATH_CALUDE_equation_solution_l3394_339478

theorem equation_solution (y : ℝ) : 
  (y / 5) / 3 = 15 / (y / 3) → y = 15 * Real.sqrt 3 ∨ y = -15 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l3394_339478


namespace NUMINAMATH_CALUDE_percentage_of_prize_money_kept_l3394_339413

-- Define the original repair cost
def original_repair_cost : ℝ := 20000

-- Define the discount percentage
def discount_percentage : ℝ := 0.20

-- Define the prize money
def prize_money : ℝ := 70000

-- Define John's profit
def profit : ℝ := 47000

-- Theorem to prove
theorem percentage_of_prize_money_kept (ε : ℝ) (h : ε > 0) :
  ∃ (percentage : ℝ), 
    abs (percentage - (profit / prize_money * 100)) < ε ∧ 
    abs (percentage - 67.14) < ε :=
sorry

end NUMINAMATH_CALUDE_percentage_of_prize_money_kept_l3394_339413


namespace NUMINAMATH_CALUDE_tape_recorder_cost_l3394_339441

theorem tape_recorder_cost :
  ∃ (x : ℕ) (p : ℝ),
    x > 2 ∧
    170 < p ∧ p < 195 ∧
    p / (x - 2) - p / x = 1 ∧
    p = 180 := by
  sorry

end NUMINAMATH_CALUDE_tape_recorder_cost_l3394_339441


namespace NUMINAMATH_CALUDE_lisa_quiz_goal_impossible_l3394_339458

theorem lisa_quiz_goal_impossible (total_quizzes : ℕ) (goal_percentage : ℚ) 
  (completed_quizzes : ℕ) (earned_as : ℕ) : 
  total_quizzes = 60 → 
  goal_percentage = 9/10 → 
  completed_quizzes = 40 → 
  earned_as = 30 → 
  ¬ ∃ (remaining_non_as : ℕ), 
    earned_as + (total_quizzes - completed_quizzes - remaining_non_as) ≥ 
    ⌈goal_percentage * total_quizzes⌉ := by
  sorry

#check lisa_quiz_goal_impossible

end NUMINAMATH_CALUDE_lisa_quiz_goal_impossible_l3394_339458


namespace NUMINAMATH_CALUDE_units_digit_of_product_sequence_l3394_339487

def product_sequence (n : ℕ) : ℕ :=
  (List.range 17).foldl (λ acc i => acc * (2^(2*i) + 1)) 3

theorem units_digit_of_product_sequence :
  (product_sequence 17 + 1) % 10 = 6 := by
  sorry

end NUMINAMATH_CALUDE_units_digit_of_product_sequence_l3394_339487


namespace NUMINAMATH_CALUDE_arithmetic_simplification_l3394_339495

theorem arithmetic_simplification :
  (4 + 6 + 4) / 3 - 4 / 3 = 10 / 3 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_simplification_l3394_339495


namespace NUMINAMATH_CALUDE_product_of_three_numbers_l3394_339402

theorem product_of_three_numbers (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c)
  (hab : a * b = 24 * (3 ^ (1/4)))
  (hac : a * c = 50 * (3 ^ (1/4)))
  (hbc : b * c = 18 * (3 ^ (1/4))) :
  a * b * c = 120 * (3 ^ (1/4)) := by
sorry

end NUMINAMATH_CALUDE_product_of_three_numbers_l3394_339402


namespace NUMINAMATH_CALUDE_equation_one_solutions_l3394_339499

theorem equation_one_solutions (x : ℝ) : 
  x^2 - 6*x - 1 = 0 ↔ x = 3 + Real.sqrt 10 ∨ x = 3 - Real.sqrt 10 := by
sorry

end NUMINAMATH_CALUDE_equation_one_solutions_l3394_339499


namespace NUMINAMATH_CALUDE_man_ownership_fraction_l3394_339425

/-- Proves that the fraction of the business the man owns is 2/3, given the conditions -/
theorem man_ownership_fraction (sold_fraction : ℚ) (sold_value : ℕ) (total_value : ℕ) 
  (h1 : sold_fraction = 3 / 4)
  (h2 : sold_value = 45000)
  (h3 : total_value = 90000) :
  ∃ (x : ℚ), x * sold_fraction * total_value = sold_value ∧ x = 2 / 3 := by
  sorry

#check man_ownership_fraction

end NUMINAMATH_CALUDE_man_ownership_fraction_l3394_339425


namespace NUMINAMATH_CALUDE_temperature_difference_l3394_339489

def highest_temp : ℝ := 10
def lowest_temp : ℝ := -1

theorem temperature_difference : highest_temp - lowest_temp = 11 := by
  sorry

end NUMINAMATH_CALUDE_temperature_difference_l3394_339489


namespace NUMINAMATH_CALUDE_gwen_race_time_l3394_339484

/-- Represents the time Gwen spent jogging and walking during a race. -/
structure RaceTime where
  jogging : ℕ
  walking : ℕ

/-- Calculates if the given race time satisfies the required ratio and walking time. -/
def is_valid_race_time (rt : RaceTime) : Prop :=
  rt.jogging * 3 = rt.walking * 5 ∧ rt.walking = 9

/-- Theorem stating that the race time with 15 minutes of jogging and 9 minutes of walking
    satisfies the required conditions. -/
theorem gwen_race_time : ∃ (rt : RaceTime), is_valid_race_time rt ∧ rt.jogging = 15 := by
  sorry

end NUMINAMATH_CALUDE_gwen_race_time_l3394_339484


namespace NUMINAMATH_CALUDE_twins_age_problem_l3394_339401

theorem twins_age_problem (age : ℕ) : 
  (age + 1) * (age + 1) = age * age + 13 → age = 6 := by sorry

end NUMINAMATH_CALUDE_twins_age_problem_l3394_339401


namespace NUMINAMATH_CALUDE_initial_marbles_l3394_339415

theorem initial_marbles (lost_marbles current_marbles : ℕ) 
  (h1 : lost_marbles = 7)
  (h2 : current_marbles = 9) :
  lost_marbles + current_marbles = 16 := by
  sorry

end NUMINAMATH_CALUDE_initial_marbles_l3394_339415


namespace NUMINAMATH_CALUDE_complex_equality_l3394_339406

theorem complex_equality (z : ℂ) : z = Complex.I ↔ 
  Complex.abs (z - 2) = Complex.abs (z + 1 - Complex.I) ∧ 
  Complex.abs (z - 2) = Complex.abs (z - (1 + 2*Complex.I)) := by
  sorry

end NUMINAMATH_CALUDE_complex_equality_l3394_339406


namespace NUMINAMATH_CALUDE_uncle_dave_nieces_l3394_339423

theorem uncle_dave_nieces (total_sandwiches : ℕ) (sandwiches_per_niece : ℕ) (h1 : total_sandwiches = 143) (h2 : sandwiches_per_niece = 13) :
  total_sandwiches / sandwiches_per_niece = 11 := by
  sorry

end NUMINAMATH_CALUDE_uncle_dave_nieces_l3394_339423


namespace NUMINAMATH_CALUDE_distance_home_to_school_l3394_339428

theorem distance_home_to_school :
  ∀ (d : ℝ) (t : ℝ),
    d = 6 * (t + 7/60) ∧
    d = 12 * (t - 8/60) →
    d = 3 := by
  sorry

end NUMINAMATH_CALUDE_distance_home_to_school_l3394_339428


namespace NUMINAMATH_CALUDE_distance_between_4th_and_26th_red_lights_l3394_339418

/-- The distance in feet between two red lights in a repeating pattern -/
def distance_between_red_lights (n m : ℕ) : ℚ :=
  let inches_between_lights : ℕ := 4
  let pattern_length : ℕ := 5
  let inches_per_foot : ℕ := 12
  let position (k : ℕ) : ℕ := 1 + (k - 1) / 2 * pattern_length + 2 * ((k - 1) % 2)
  let gaps : ℕ := position m - position n
  (gaps * inches_between_lights : ℚ) / inches_per_foot

/-- The theorem stating the distance between the 4th and 26th red lights -/
theorem distance_between_4th_and_26th_red_lights :
  distance_between_red_lights 4 26 = 18.33 :=
sorry

end NUMINAMATH_CALUDE_distance_between_4th_and_26th_red_lights_l3394_339418


namespace NUMINAMATH_CALUDE_function_bound_l3394_339430

noncomputable def f (a x : ℝ) : ℝ := a * Real.sin x - 1/2 * Real.cos (2*x) + a - 3/a + 1/2

theorem function_bound (a : ℝ) (ha : a ≠ 0) :
  (∀ x, f a x ≤ 0) → 0 < a ∧ a ≤ 3 := by sorry

end NUMINAMATH_CALUDE_function_bound_l3394_339430


namespace NUMINAMATH_CALUDE_course_size_l3394_339467

theorem course_size (num_d : ℕ) (h_d : num_d = 25) :
  ∃ (total : ℕ),
    total > 0 ∧
    (total : ℚ) = num_d + (1/5 : ℚ) * total + (1/4 : ℚ) * total + (1/2 : ℚ) * total ∧
    total = 500 := by
  sorry

end NUMINAMATH_CALUDE_course_size_l3394_339467


namespace NUMINAMATH_CALUDE_simplify_polynomial_l3394_339444

theorem simplify_polynomial (x : ℝ) : 
  3*x + 5 - 4*x^2 + 2*x - 7 + x^2 - 3*x + 8 = -3*x^2 + 2*x + 6 := by
sorry

end NUMINAMATH_CALUDE_simplify_polynomial_l3394_339444


namespace NUMINAMATH_CALUDE_geometric_and_arithmetic_sequences_l3394_339424

-- Define the geometric sequence a_n
def a (n : ℕ) : ℝ := 3 * 2^(n - 1)

-- Define the arithmetic sequence b_n
def b (n : ℕ) : ℝ := 6 * n - 6

-- Define the sum of the first n terms of b_n
def S (n : ℕ) : ℝ := 3 * n^2 - 3 * n

theorem geometric_and_arithmetic_sequences :
  (a 1 = 3) ∧ 
  (a 4 = 24) ∧ 
  (b 2 = a 2) ∧ 
  (b 9 = a 5) ∧ 
  (∀ n : ℕ, a n = 3 * 2^(n - 1)) ∧ 
  (∀ n : ℕ, S n = 3 * n^2 - 3 * n) :=
by sorry

end NUMINAMATH_CALUDE_geometric_and_arithmetic_sequences_l3394_339424


namespace NUMINAMATH_CALUDE_bee_multiple_l3394_339412

theorem bee_multiple (bees_day1 bees_day2 : ℕ) (h1 : bees_day1 = 144) (h2 : bees_day2 = 432) :
  bees_day2 / bees_day1 = 3 := by
  sorry

end NUMINAMATH_CALUDE_bee_multiple_l3394_339412


namespace NUMINAMATH_CALUDE_min_reciprocal_sum_l3394_339448

theorem min_reciprocal_sum (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : a + 3 * b = 2) :
  (1 / a + 1 / b) ≥ 2 + Real.sqrt 3 ∧
  ∃ (a₀ b₀ : ℝ), a₀ > 0 ∧ b₀ > 0 ∧ a₀ + 3 * b₀ = 2 ∧ 1 / a₀ + 1 / b₀ = 2 + Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_min_reciprocal_sum_l3394_339448


namespace NUMINAMATH_CALUDE_perpendicular_line_equation_l3394_339410

-- Define the given line
def given_line (x y : ℝ) : Prop := 3*x + y + 5 = 0

-- Define the point P
def point_P : ℝ × ℝ := (2, 1)

-- Define the perpendicular line l
def line_l (x y : ℝ) : Prop := x - 3*y + 1 = 0

-- Theorem statement
theorem perpendicular_line_equation :
  (∀ x y : ℝ, given_line x y → (line_l x y → ¬given_line x y)) ∧
  line_l point_P.1 point_P.2 :=
sorry

end NUMINAMATH_CALUDE_perpendicular_line_equation_l3394_339410


namespace NUMINAMATH_CALUDE_parabola_focus_directrix_distance_l3394_339446

/-- Given a parabola x^2 = (1/4)y, the distance between its focus and directrix is 1/8 -/
theorem parabola_focus_directrix_distance (x y : ℝ) :
  x^2 = (1/4) * y → (distance_focus_directrix : ℝ) = 1/8 := by
  sorry

end NUMINAMATH_CALUDE_parabola_focus_directrix_distance_l3394_339446


namespace NUMINAMATH_CALUDE_rect_to_cylindrical_conversion_l3394_339483

/-- Conversion from rectangular to cylindrical coordinates -/
theorem rect_to_cylindrical_conversion (x y z : ℝ) (r θ : ℝ) 
  (h1 : x = 6)
  (h2 : y = 6)
  (h3 : z = -10)
  (h4 : r > 0)
  (h5 : 0 ≤ θ ∧ θ < 2 * π)
  (h6 : x = r * Real.cos θ)
  (h7 : y = r * Real.sin θ) :
  r = 6 * Real.sqrt 2 ∧ θ = π / 4 ∧ z = -10 := by
  sorry

end NUMINAMATH_CALUDE_rect_to_cylindrical_conversion_l3394_339483


namespace NUMINAMATH_CALUDE_quadratic_condition_l3394_339493

theorem quadratic_condition (m : ℝ) : 
  (∀ x, ∃ a b c : ℝ, a ≠ 0 ∧ (m^2 - 1) * x^2 + x + m = a * x^2 + b * x + c) ↔ 
  (m ≠ 1 ∧ m ≠ -1) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_condition_l3394_339493


namespace NUMINAMATH_CALUDE_fish_filets_count_l3394_339475

/-- The number of fish filets Ben and his family will have -/
def fish_filets : ℕ :=
  let ben_fish := 4
  let judy_fish := 1
  let billy_fish := 3
  let jim_fish := 2
  let susie_fish := 5
  let thrown_back := 3
  let filets_per_fish := 2
  let total_caught := ben_fish + judy_fish + billy_fish + jim_fish + susie_fish
  let fish_kept := total_caught - thrown_back
  fish_kept * filets_per_fish

theorem fish_filets_count : fish_filets = 24 := by
  sorry

end NUMINAMATH_CALUDE_fish_filets_count_l3394_339475


namespace NUMINAMATH_CALUDE_geometric_sequence_sum_property_l3394_339460

/-- A geometric sequence -/
def GeometricSequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

/-- The sum of two consecutive terms in a sequence -/
def ConsecutiveSum (a : ℕ → ℝ) (n : ℕ) : ℝ :=
  a n + a (n + 1)

theorem geometric_sequence_sum_property
  (a : ℕ → ℝ)
  (h_geo : GeometricSequence a)
  (h_sum1 : ConsecutiveSum a 1 = 16)
  (h_sum2 : ConsecutiveSum a 3 = 24) :
  ConsecutiveSum a 7 = 54 :=
sorry

end NUMINAMATH_CALUDE_geometric_sequence_sum_property_l3394_339460


namespace NUMINAMATH_CALUDE_second_draw_probability_l3394_339403

/-- Represents the total number of items -/
def total_items : ℕ := 10

/-- Represents the number of genuine items -/
def genuine_items : ℕ := 6

/-- Represents the number of defective items -/
def defective_items : ℕ := 4

/-- Represents the probability of drawing a genuine item on the second draw,
    given that the first item drawn is genuine -/
def prob_second_genuine : ℚ := 5 / 9

theorem second_draw_probability :
  total_items = genuine_items + defective_items →
  genuine_items > 0 →
  prob_second_genuine = (genuine_items - 1) / (total_items - 1) :=
by sorry

end NUMINAMATH_CALUDE_second_draw_probability_l3394_339403


namespace NUMINAMATH_CALUDE_logarithm_expression_equality_l3394_339451

theorem logarithm_expression_equality : 
  (Real.log 8 / Real.log 5 * Real.log 2 / Real.log 5 + 25 ^ (Real.log 3 / Real.log 5)) / 
  (Real.log 4 + Real.log 25) + 5 * Real.log 2 / Real.log 3 - Real.log (32/9) / Real.log 3 = 8 := by
  sorry

end NUMINAMATH_CALUDE_logarithm_expression_equality_l3394_339451


namespace NUMINAMATH_CALUDE_solution_range_l3394_339457

-- Define the new operation
def otimes (x y : ℝ) : ℝ := x * (1 - y)

-- Theorem statement
theorem solution_range (a : ℝ) :
  (∃ x : ℝ, otimes x (x - a) > 1) ↔ (a < -3 ∨ a > 1) :=
sorry

end NUMINAMATH_CALUDE_solution_range_l3394_339457


namespace NUMINAMATH_CALUDE_vegan_soy_free_fraction_l3394_339408

theorem vegan_soy_free_fraction (total_dishes : ℕ) (vegan_dishes : ℕ) (soy_dishes : ℕ)
  (h1 : vegan_dishes = total_dishes / 4)
  (h2 : soy_dishes = 4 * vegan_dishes / 5)
  (h3 : vegan_dishes > 0)
  (h4 : total_dishes > 0) :
  (vegan_dishes - soy_dishes) / total_dishes = 1 / 20 :=
by sorry

end NUMINAMATH_CALUDE_vegan_soy_free_fraction_l3394_339408


namespace NUMINAMATH_CALUDE_rope_division_l3394_339420

theorem rope_division (rope_length : ℝ) (num_parts : ℕ) (part_length : ℝ) :
  rope_length = 5 →
  num_parts = 4 →
  rope_length = num_parts * part_length →
  part_length = 1.25 := by
sorry

end NUMINAMATH_CALUDE_rope_division_l3394_339420


namespace NUMINAMATH_CALUDE_coprime_35_58_in_base_l3394_339404

/-- Two natural numbers are coprime if their greatest common divisor is 1. -/
def Coprime (a b : ℕ) : Prop := Nat.gcd a b = 1

/-- A numeral system base is valid if it's greater than 1. -/
def ValidBase (base : ℕ) : Prop := base > 1

theorem coprime_35_58_in_base (base : ℕ) (h : ValidBase base) (h_base : base > 8) :
  Coprime 35 58 := by
  sorry

#check coprime_35_58_in_base

end NUMINAMATH_CALUDE_coprime_35_58_in_base_l3394_339404


namespace NUMINAMATH_CALUDE_rectangle_height_from_square_perimeter_l3394_339471

theorem rectangle_height_from_square_perimeter (square_side : ℝ) (rect_width : ℝ) :
  square_side = 20 →
  rect_width = 14 →
  4 * square_side = 2 * (rect_width + (80 - 2 * rect_width) / 2) →
  (80 - 2 * rect_width) / 2 = 26 :=
by sorry

end NUMINAMATH_CALUDE_rectangle_height_from_square_perimeter_l3394_339471


namespace NUMINAMATH_CALUDE_complex_equation_sum_l3394_339466

theorem complex_equation_sum (x y : ℝ) :
  (x / (1 - Complex.I)) + (y / (1 - 2 * Complex.I)) = 5 / (1 - 3 * Complex.I) →
  x + y = 4 := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_sum_l3394_339466
