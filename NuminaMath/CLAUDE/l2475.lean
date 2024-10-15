import Mathlib

namespace NUMINAMATH_CALUDE_original_number_is_27_l2475_247568

theorem original_number_is_27 (x : ℕ) :
  (Odd (3 * x)) →
  (∃ k : ℕ, 3 * x = 9 * k) →
  (∃ y : ℕ, x * y = 108) →
  x = 27 := by
sorry

end NUMINAMATH_CALUDE_original_number_is_27_l2475_247568


namespace NUMINAMATH_CALUDE_medicine_supply_duration_l2475_247569

theorem medicine_supply_duration (pills : ℕ) (consumption_rate : ℚ) (consumption_days : ℕ) (days_per_month : ℕ) : 
  pills = 90 → 
  consumption_rate = 1/3 → 
  consumption_days = 3 → 
  days_per_month = 30 → 
  (pills * consumption_days / consumption_rate) / days_per_month = 27 := by
  sorry

end NUMINAMATH_CALUDE_medicine_supply_duration_l2475_247569


namespace NUMINAMATH_CALUDE_david_crunches_count_l2475_247563

/-- The number of crunches Zachary did -/
def zachary_crunches : ℕ := 62

/-- The difference in crunches between Zachary and David -/
def crunch_difference : ℕ := 17

/-- The number of crunches David did -/
def david_crunches : ℕ := zachary_crunches - crunch_difference

theorem david_crunches_count : david_crunches = 45 := by
  sorry

end NUMINAMATH_CALUDE_david_crunches_count_l2475_247563


namespace NUMINAMATH_CALUDE_score_difference_l2475_247591

/-- Represents the scores of a student in three subjects -/
structure Scores where
  math : ℝ
  physics : ℝ
  chemistry : ℝ

/-- The problem statement -/
theorem score_difference (s : Scores) 
  (h1 : s.math + s.physics = 20)
  (h2 : (s.math + s.chemistry) / 2 = 20)
  (h3 : s.chemistry > s.physics) :
  s.chemistry - s.physics = 20 := by
  sorry

end NUMINAMATH_CALUDE_score_difference_l2475_247591


namespace NUMINAMATH_CALUDE_cylinder_height_given_cone_volume_ratio_l2475_247548

theorem cylinder_height_given_cone_volume_ratio (base_area : ℝ) (cone_height : ℝ) :
  cone_height = 4.5 →
  (1 / 3 * base_area * cone_height) / (base_area * cylinder_height) = 1 / 6 →
  cylinder_height = 9 :=
by
  sorry

end NUMINAMATH_CALUDE_cylinder_height_given_cone_volume_ratio_l2475_247548


namespace NUMINAMATH_CALUDE_trajectory_and_max_dot_product_l2475_247536

/-- Trajectory of point P satisfying given conditions -/
def trajectory (x y : ℝ) : Prop :=
  x^2 / 4 + y^2 = 1

/-- Line segment AB with A on x-axis and B on y-axis -/
def lineSegmentAB (xA yB : ℝ) : Prop :=
  xA^2 + yB^2 = 9 ∧ xA ≥ 0 ∧ yB ≥ 0

/-- Point P satisfies BP = 2PA -/
def pointPCondition (xA yB x y : ℝ) : Prop :=
  (x - 0)^2 + (y - yB)^2 = 4 * ((x - xA)^2 + y^2)

/-- Line passing through (1,0) -/
def lineThroughOneZero (t x y : ℝ) : Prop :=
  x = t * y + 1

/-- Theorem stating the trajectory equation and maximum dot product -/
theorem trajectory_and_max_dot_product :
  ∀ xA yB x y t x1 y1 x2 y2 : ℝ,
  lineSegmentAB xA yB →
  pointPCondition xA yB x y →
  trajectory x y →
  lineThroughOneZero t x1 y1 →
  lineThroughOneZero t x2 y2 →
  trajectory x1 y1 →
  trajectory x2 y2 →
  (∀ x' y' : ℝ, trajectory x' y' → lineThroughOneZero t x' y' → 
    x1 * x2 + y1 * y2 ≥ x' * x' + y' * y') →
  x1 * x2 + y1 * y2 ≤ 1/4 :=
by sorry

end NUMINAMATH_CALUDE_trajectory_and_max_dot_product_l2475_247536


namespace NUMINAMATH_CALUDE_area_difference_equals_target_l2475_247525

/-- A right triangle with side lengths 3, 4, and 5 -/
structure RightTriangle where
  base : Real
  height : Real
  hypotenuse : Real
  is_right : base = 3 ∧ height = 4 ∧ hypotenuse = 5

/-- The set Xₙ as defined in the problem -/
def X (n : ℕ) (t : RightTriangle) : Set (Real × Real) :=
  sorry

/-- The area of the region outside X₂₀ but inside X₂₁ -/
def area_difference (t : RightTriangle) : Real :=
  sorry

/-- The main theorem to prove -/
theorem area_difference_equals_target (t : RightTriangle) :
  area_difference t = (41 * Real.pi / 2) + 12 := by
  sorry

end NUMINAMATH_CALUDE_area_difference_equals_target_l2475_247525


namespace NUMINAMATH_CALUDE_units_digit_of_7_pow_6_pow_5_l2475_247518

-- Define the function to get the units digit
def unitsDigit (n : ℕ) : ℕ := n % 10

-- Define the exponential function
def pow (base exponent : ℕ) : ℕ := base ^ exponent

-- Theorem statement
theorem units_digit_of_7_pow_6_pow_5 : unitsDigit (pow 7 (pow 6 5)) = 1 := by
  sorry

end NUMINAMATH_CALUDE_units_digit_of_7_pow_6_pow_5_l2475_247518


namespace NUMINAMATH_CALUDE_diophantine_equation_solution_l2475_247539

theorem diophantine_equation_solution (x y : ℤ) :
  y^2 = x^3 + 16 → x = 0 ∧ y = 4 := by
  sorry

end NUMINAMATH_CALUDE_diophantine_equation_solution_l2475_247539


namespace NUMINAMATH_CALUDE_math_club_team_selection_l2475_247595

def total_boys : ℕ := 9
def total_girls : ℕ := 10
def experienced_boys : ℕ := 4
def team_size : ℕ := 7
def required_boys : ℕ := 4
def required_girls : ℕ := 3
def required_experienced_boys : ℕ := 2

theorem math_club_team_selection :
  (Nat.choose experienced_boys required_experienced_boys) *
  (Nat.choose (total_boys - experienced_boys) (required_boys - required_experienced_boys)) *
  (Nat.choose total_girls required_girls) = 7200 :=
sorry

end NUMINAMATH_CALUDE_math_club_team_selection_l2475_247595


namespace NUMINAMATH_CALUDE_cheaper_store_difference_l2475_247544

/-- The list price of Book Y in dollars -/
def list_price : ℚ := 24.95

/-- The discount amount at Readers' Delight in dollars -/
def readers_delight_discount : ℚ := 5

/-- The discount percentage at Book Bargains -/
def book_bargains_discount_percent : ℚ := 20

/-- The sale price at Readers' Delight in dollars -/
def readers_delight_price : ℚ := list_price - readers_delight_discount

/-- The sale price at Book Bargains in dollars -/
def book_bargains_price : ℚ := list_price * (1 - book_bargains_discount_percent / 100)

/-- The price difference in cents -/
def price_difference_cents : ℤ := ⌊(book_bargains_price - readers_delight_price) * 100⌋

theorem cheaper_store_difference :
  price_difference_cents = 1 :=
sorry

end NUMINAMATH_CALUDE_cheaper_store_difference_l2475_247544


namespace NUMINAMATH_CALUDE_officer_assignment_count_l2475_247540

def group_size : ℕ := 4
def roles : ℕ := 3

theorem officer_assignment_count : (group_size.choose roles) * (Nat.factorial roles) = 24 := by
  sorry

end NUMINAMATH_CALUDE_officer_assignment_count_l2475_247540


namespace NUMINAMATH_CALUDE_hyperbola_circle_relation_l2475_247586

-- Define the hyperbola
def is_hyperbola (x y : ℝ) : Prop := y^2 - x^2/3 = 1

-- Define a focus of the hyperbola
def is_focus (x y : ℝ) : Prop := x = 0 ∧ (y = 2 ∨ y = -2)

-- Define the eccentricity of the hyperbola
def eccentricity : ℝ := 2

-- Define the circle
def is_circle (x y : ℝ) : Prop := x^2 + (y-2)^2 = 4

-- Theorem statement
theorem hyperbola_circle_relation :
  ∀ (x y cx cy : ℝ),
  is_hyperbola x y →
  is_focus cx cy →
  is_circle (x - cx) (y - cy) :=
sorry

end NUMINAMATH_CALUDE_hyperbola_circle_relation_l2475_247586


namespace NUMINAMATH_CALUDE_solution_set_x_squared_geq_four_l2475_247575

theorem solution_set_x_squared_geq_four :
  {x : ℝ | x^2 ≥ 4} = {x : ℝ | x ≤ -2 ∨ x ≥ 2} := by
sorry

end NUMINAMATH_CALUDE_solution_set_x_squared_geq_four_l2475_247575


namespace NUMINAMATH_CALUDE_floor_times_self_eq_72_l2475_247579

theorem floor_times_self_eq_72 (x : ℝ) :
  x > 0 ∧ ⌊x⌋ * x = 72 → x = 9 := by sorry

end NUMINAMATH_CALUDE_floor_times_self_eq_72_l2475_247579


namespace NUMINAMATH_CALUDE_complex_sum_problem_l2475_247527

theorem complex_sum_problem (p q r s t u : ℝ) : 
  s = 5 →
  t = -p - r →
  (p + q * Complex.I) + (r + s * Complex.I) + (t + u * Complex.I) = -6 * Complex.I →
  u + q = -11 := by
  sorry

end NUMINAMATH_CALUDE_complex_sum_problem_l2475_247527


namespace NUMINAMATH_CALUDE_green_chips_count_l2475_247547

theorem green_chips_count (total : ℕ) (red : ℕ) (h1 : total = 60) (h2 : red = 34) :
  total - (total / 6) - red = 16 := by
  sorry

end NUMINAMATH_CALUDE_green_chips_count_l2475_247547


namespace NUMINAMATH_CALUDE_smallest_n_congruence_l2475_247503

theorem smallest_n_congruence (n : ℕ) : 
  (n > 0 ∧ 5 * n ≡ 980 [ZMOD 33] ∧ ∀ m : ℕ, m > 0 ∧ m < n → ¬(5 * m ≡ 980 [ZMOD 33])) ↔ n = 19 :=
by sorry

end NUMINAMATH_CALUDE_smallest_n_congruence_l2475_247503


namespace NUMINAMATH_CALUDE_hyperbola_eccentricity_l2475_247551

/-- Given a hyperbola with equation x²/m² - y² = 4 where m > 0 and focal distance 8,
    prove that its eccentricity is 2√3/3 -/
theorem hyperbola_eccentricity (m : ℝ) (h1 : m > 0) :
  let focal_distance : ℝ := 8
  let a : ℝ := m * 2
  let b : ℝ := 2
  let c : ℝ := focal_distance / 2
  let eccentricity : ℝ := c / a
  eccentricity = 2 * Real.sqrt 3 / 3 := by sorry

end NUMINAMATH_CALUDE_hyperbola_eccentricity_l2475_247551


namespace NUMINAMATH_CALUDE_bug_path_length_l2475_247561

theorem bug_path_length (a b c : ℝ) (h1 : a = 120) (h2 : b = 90) (h3 : c = 150) : 
  ∃ (d : ℝ), (a^2 + b^2 = c^2) ∧ (c + c + d = 390) ∧ (d = a ∨ d = b) :=
sorry

end NUMINAMATH_CALUDE_bug_path_length_l2475_247561


namespace NUMINAMATH_CALUDE_negation_of_implication_is_false_l2475_247517

theorem negation_of_implication_is_false : 
  ¬(∃ a b : ℝ, (a ≤ 1 ∨ b ≤ 1) ∧ (a + b ≤ 2)) := by
  sorry

end NUMINAMATH_CALUDE_negation_of_implication_is_false_l2475_247517


namespace NUMINAMATH_CALUDE_trigonometric_identity_l2475_247510

theorem trigonometric_identity (α : Real) 
  (h : Real.sqrt 2 * Real.sin (α + π / 4) = 4 * Real.cos α) : 
  2 * Real.sin α ^ 2 - Real.sin α * Real.cos α + Real.cos α ^ 2 = 8 / 5 := by
  sorry

end NUMINAMATH_CALUDE_trigonometric_identity_l2475_247510


namespace NUMINAMATH_CALUDE_optimal_strategy_and_expected_red_balls_l2475_247506

-- Define the contents of A's box
structure BoxA where
  red : ℕ
  white : ℕ
  sum_eq_four : red + white = 4

-- Define the contents of B's box
def BoxB : Finset (Fin 4) := {0, 1, 2, 3}

-- Define the probability of winning for A given their box contents
def win_probability (box : BoxA) : ℚ :=
  (box.red * box.white * 2) / (12 * 6)

-- Define the expected number of red balls drawn
def expected_red_balls (box : BoxA) : ℚ :=
  (box.red * 2 / 6) + (2 / 4)

-- Theorem statement
theorem optimal_strategy_and_expected_red_balls :
  ∃ (box : BoxA),
    (∀ (other : BoxA), win_probability box ≥ win_probability other) ∧
    (box.red = 2 ∧ box.white = 2) ∧
    (expected_red_balls box = 3/2) := by
  sorry

end NUMINAMATH_CALUDE_optimal_strategy_and_expected_red_balls_l2475_247506


namespace NUMINAMATH_CALUDE_smoothie_time_theorem_l2475_247534

/-- Represents the time in minutes to chop each fruit type -/
structure ChoppingTimes where
  apple : ℕ
  banana : ℕ
  strawberry : ℕ
  mango : ℕ
  pineapple : ℕ

/-- Calculates the total time to make smoothies -/
def totalSmoothieTime (ct : ChoppingTimes) (blendTime : ℕ) (numSmoothies : ℕ) : ℕ :=
  (ct.apple + ct.banana + ct.strawberry + ct.mango + ct.pineapple + blendTime) * numSmoothies

/-- Theorem: The total time to make 5 smoothies is 115 minutes -/
theorem smoothie_time_theorem (ct : ChoppingTimes) (blendTime : ℕ) (numSmoothies : ℕ) :
  ct.apple = 2 →
  ct.banana = 3 →
  ct.strawberry = 4 →
  ct.mango = 5 →
  ct.pineapple = 6 →
  blendTime = 3 →
  numSmoothies = 5 →
  totalSmoothieTime ct blendTime numSmoothies = 115 := by
  sorry

end NUMINAMATH_CALUDE_smoothie_time_theorem_l2475_247534


namespace NUMINAMATH_CALUDE_exists_line_with_specified_length_l2475_247589

-- Define the circles
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

-- Define the intersection points
def A : ℝ × ℝ := sorry
def B : ℝ × ℝ := sorry

-- Define the circles
def S₁ : Circle := sorry
def S₂ : Circle := sorry

-- Specify that the circles intersect at A and B
axiom intersect_at_A : A ∈ {p | (p.1 - S₁.center.1)^2 + (p.2 - S₁.center.2)^2 = S₁.radius^2} ∩
                           {p | (p.1 - S₂.center.1)^2 + (p.2 - S₂.center.2)^2 = S₂.radius^2}
axiom intersect_at_B : B ∈ {p | (p.1 - S₁.center.1)^2 + (p.2 - S₁.center.2)^2 = S₁.radius^2} ∩
                           {p | (p.1 - S₂.center.1)^2 + (p.2 - S₂.center.2)^2 = S₂.radius^2}

-- Define a line passing through point A
def line_through_A (m : ℝ) : Set (ℝ × ℝ) :=
  {p | p.2 - A.2 = m * (p.1 - A.1)}

-- Define the segment of a line contained within both circles
def segment_in_circles (l : Set (ℝ × ℝ)) : Set (ℝ × ℝ) :=
  l ∩ {p | (p.1 - S₁.center.1)^2 + (p.2 - S₁.center.2)^2 ≤ S₁.radius^2} ∩
       {p | (p.1 - S₂.center.1)^2 + (p.2 - S₂.center.2)^2 ≤ S₂.radius^2}

-- Define the length of a segment
def segment_length (s : Set (ℝ × ℝ)) : ℝ := sorry

-- Theorem statement
theorem exists_line_with_specified_length (length : ℝ) :
  ∃ m : ℝ, segment_length (segment_in_circles (line_through_A m)) = length :=
sorry

end NUMINAMATH_CALUDE_exists_line_with_specified_length_l2475_247589


namespace NUMINAMATH_CALUDE_apple_cost_l2475_247562

theorem apple_cost (initial_cost : ℝ) (initial_dozen : ℕ) (target_dozen : ℕ) : 
  initial_cost * (target_dozen / initial_dozen) = 54.60 :=
by
  sorry

#check apple_cost 39.00 5 7

end NUMINAMATH_CALUDE_apple_cost_l2475_247562


namespace NUMINAMATH_CALUDE_total_cost_theorem_l2475_247538

def cost_per_package : ℕ := 5
def num_parents : ℕ := 2
def num_brothers : ℕ := 3
def children_per_brother : ℕ := 2

def total_relatives : ℕ := num_parents + num_brothers + num_brothers + (num_brothers * children_per_brother)

theorem total_cost_theorem : cost_per_package * total_relatives = 70 := by
  sorry

end NUMINAMATH_CALUDE_total_cost_theorem_l2475_247538


namespace NUMINAMATH_CALUDE_grocer_coffee_stock_theorem_l2475_247564

/-- Represents the amount of coffee in pounds and its decaffeinated percentage -/
structure CoffeeStock where
  amount : ℝ
  decaf_percent : ℝ

/-- Calculates the new coffee stock after a purchase or sale -/
def update_stock (current : CoffeeStock) (transaction : CoffeeStock) (is_sale : Bool) : CoffeeStock :=
  sorry

/-- Calculates the final percentage of decaffeinated coffee -/
def final_decaf_percentage (transactions : List (CoffeeStock × Bool)) : ℝ :=
  sorry

theorem grocer_coffee_stock_theorem (initial_stock : CoffeeStock) 
  (transactions : List (CoffeeStock × Bool)) : 
  let final_percent := final_decaf_percentage transactions
  ∃ ε > 0, |final_percent - 28.88| < ε :=
by sorry

end NUMINAMATH_CALUDE_grocer_coffee_stock_theorem_l2475_247564


namespace NUMINAMATH_CALUDE_average_marks_proof_l2475_247509

def english_marks : ℕ := 76
def math_marks : ℕ := 65
def physics_marks : ℕ := 82
def chemistry_marks : ℕ := 67
def biology_marks : ℕ := 85
def num_subjects : ℕ := 5

theorem average_marks_proof :
  (english_marks + math_marks + physics_marks + chemistry_marks + biology_marks) / num_subjects = 75 := by
  sorry

end NUMINAMATH_CALUDE_average_marks_proof_l2475_247509


namespace NUMINAMATH_CALUDE_amoeba_count_after_10_days_l2475_247560

def amoeba_count (day : ℕ) : ℕ :=
  if day = 0 then 3
  else if (day % 3 = 0) ∧ (day ≥ 3) then
    amoeba_count (day - 1)
  else
    2 * amoeba_count (day - 1)

theorem amoeba_count_after_10_days :
  amoeba_count 10 = 384 :=
sorry

end NUMINAMATH_CALUDE_amoeba_count_after_10_days_l2475_247560


namespace NUMINAMATH_CALUDE_nonreal_roots_product_l2475_247558

theorem nonreal_roots_product (x : ℂ) : 
  (x^4 - 4*x^3 + 6*x^2 - 4*x + 4 = 4036) → 
  (∃ a b : ℂ, a ≠ b ∧ a.im ≠ 0 ∧ b.im ≠ 0 ∧ 
   (x = a ∨ x = b) ∧ 
   (x^4 - 4*x^3 + 6*x^2 - 4*x + 4 = 4036) ∧
   (a * b = 1 + Real.sqrt 4033)) :=
by sorry

end NUMINAMATH_CALUDE_nonreal_roots_product_l2475_247558


namespace NUMINAMATH_CALUDE_suresh_work_time_l2475_247592

theorem suresh_work_time (S : ℝ) (h1 : S > 0) : 
  (∃ (ashutosh_time : ℝ), 
    ashutosh_time = 35 ∧ 
    (9 / S) + (14 / ashutosh_time) = 1) → 
  S = 15 := by
sorry

end NUMINAMATH_CALUDE_suresh_work_time_l2475_247592


namespace NUMINAMATH_CALUDE_number_125_with_digit_sum_5_l2475_247521

/-- A function that calculates the sum of digits of a natural number -/
def digitSum (n : ℕ) : ℕ := sorry

/-- A function that returns the nth number in the sequence of natural numbers with digit sum 5 -/
def nthNumberWithDigitSum5 (n : ℕ) : ℕ := sorry

/-- The theorem stating that the 125th number in the sequence is 41000 -/
theorem number_125_with_digit_sum_5 : nthNumberWithDigitSum5 125 = 41000 := by sorry

end NUMINAMATH_CALUDE_number_125_with_digit_sum_5_l2475_247521


namespace NUMINAMATH_CALUDE_regular_polygon_on_grid_l2475_247523

/-- A grid in the plane formed by two families of equally spaced parallel lines -/
structure Grid where
  -- We don't need to define the internal structure of the grid

/-- A point in the plane -/
structure Point where
  -- We don't need to define the internal structure of the point

/-- A regular convex n-gon -/
structure RegularPolygon where
  vertices : List Point
  n : Nat
  is_regular : Bool
  is_convex : Bool

/-- Predicate to check if a point is on the grid -/
def Point.on_grid (p : Point) (g : Grid) : Prop := sorry

/-- Predicate to check if all vertices of a polygon are on the grid -/
def RegularPolygon.vertices_on_grid (p : RegularPolygon) (g : Grid) : Prop :=
  ∀ v ∈ p.vertices, v.on_grid g

/-- The main theorem -/
theorem regular_polygon_on_grid (g : Grid) (p : RegularPolygon) :
  p.n ≥ 3 ∧ p.is_regular ∧ p.is_convex ∧ p.vertices_on_grid g → p.n = 3 ∨ p.n = 4 ∨ p.n = 6 := by
  sorry

end NUMINAMATH_CALUDE_regular_polygon_on_grid_l2475_247523


namespace NUMINAMATH_CALUDE_mary_remaining_stickers_l2475_247546

/-- Calculates the number of remaining stickers for Mary --/
def remaining_stickers (initial : ℕ) (front_page : ℕ) (other_pages : ℕ) (stickers_per_page : ℕ) : ℕ :=
  initial - (front_page + other_pages * stickers_per_page)

/-- Proves that Mary has 44 stickers remaining --/
theorem mary_remaining_stickers :
  remaining_stickers 89 3 6 7 = 44 := by
  sorry

#eval remaining_stickers 89 3 6 7

end NUMINAMATH_CALUDE_mary_remaining_stickers_l2475_247546


namespace NUMINAMATH_CALUDE_population_change_l2475_247574

/-- The population change problem --/
theorem population_change (P : ℝ) : 
  P > 0 → 
  P * 1.15 * 0.90 * 1.20 * 0.75 = 7575 → 
  ∃ n : ℕ, n > 0 ∧ (n : ℝ) ≤ P ∧ P < (n : ℝ) + 1 :=
by sorry

end NUMINAMATH_CALUDE_population_change_l2475_247574


namespace NUMINAMATH_CALUDE_solution_set_inequality_l2475_247541

-- Define the determinant function
def det (a b c d : ℝ) : ℝ := a * d - b * c

-- Define the logarithm with base √2
noncomputable def log_sqrt2 (x : ℝ) : ℝ := Real.log x / Real.log (Real.sqrt 2)

-- Theorem statement
theorem solution_set_inequality (x : ℝ) :
  (log_sqrt2 (det 1 1 1 x) < 0) ↔ (1 < x ∧ x < 2) :=
sorry

end NUMINAMATH_CALUDE_solution_set_inequality_l2475_247541


namespace NUMINAMATH_CALUDE_segment_length_is_zero_l2475_247552

/-- Triangle with side lengths and an angle -/
structure Triangle :=
  (a b c : ℝ)
  (angle : ℝ)

/-- The problem setup -/
def problem : Prop :=
  ∃ (ABC DEF : Triangle),
    ABC.a = 8 ∧ ABC.b = 12 ∧ ABC.c = 10 ∧
    DEF.a = 4 ∧ DEF.b = 6 ∧ DEF.c = 5 ∧
    ABC.angle = 100 ∧ DEF.angle = 100 ∧
    ∀ (BD : ℝ), BD = 0

/-- The theorem to be proved -/
theorem segment_length_is_zero : problem := by sorry

end NUMINAMATH_CALUDE_segment_length_is_zero_l2475_247552


namespace NUMINAMATH_CALUDE_teacher_selection_problem_l2475_247593

/-- The number of ways to select k items from n items --/
def permutation (n k : ℕ) : ℕ := 
  Nat.factorial n / Nat.factorial (n - k)

/-- The number of valid selections of teachers --/
def validSelections (totalTeachers maleTeachers femaleTeachers selectCount : ℕ) : ℕ :=
  permutation totalTeachers selectCount - 
  (permutation maleTeachers selectCount + permutation femaleTeachers selectCount)

theorem teacher_selection_problem :
  validSelections 9 5 4 3 = 420 := by
  sorry

end NUMINAMATH_CALUDE_teacher_selection_problem_l2475_247593


namespace NUMINAMATH_CALUDE_fraction_simplification_complex_fraction_simplification_l2475_247596

theorem fraction_simplification (x y : ℝ) (h : 2 * x ≠ y) :
  (3 * x) / (2 * x - y) - (x + y) / (2 * x - y) = 1 := by sorry

theorem complex_fraction_simplification (x : ℝ) (h1 : x ≠ -2) (h2 : x ≠ 2) (h3 : x ≠ -2) :
  (x^2 - 5*x) / (x + 2) / ((x - 5) / (x^2 - 4)) = x^2 - 2*x := by sorry

end NUMINAMATH_CALUDE_fraction_simplification_complex_fraction_simplification_l2475_247596


namespace NUMINAMATH_CALUDE_apple_basket_problem_l2475_247580

theorem apple_basket_problem (small_basket_capacity : ℕ) (small_basket_count : ℕ) 
  (large_basket_count : ℕ) (leftover_weight : ℕ) :
  small_basket_capacity = 25 →
  small_basket_count = 28 →
  large_basket_count = 10 →
  leftover_weight = 50 →
  (small_basket_capacity * small_basket_count - leftover_weight) / large_basket_count = 65 := by
  sorry

end NUMINAMATH_CALUDE_apple_basket_problem_l2475_247580


namespace NUMINAMATH_CALUDE_choir_problem_l2475_247502

/-- A choir problem involving singers joining in different verses -/
theorem choir_problem (total_singers : ℕ) (first_verse_singers : ℕ) 
  (second_verse_joiners : ℕ) (third_verse_joiners : ℕ) : 
  total_singers = 30 →
  first_verse_singers = total_singers / 2 →
  third_verse_joiners = 10 →
  first_verse_singers + second_verse_joiners + third_verse_joiners = total_singers →
  (second_verse_joiners : ℚ) / (total_singers - first_verse_singers : ℚ) = 1 / 3 := by
  sorry

#check choir_problem

end NUMINAMATH_CALUDE_choir_problem_l2475_247502


namespace NUMINAMATH_CALUDE_hours_worked_on_second_job_l2475_247588

/-- Calculates the number of hours worked on the second job given the total earnings and other job details -/
theorem hours_worked_on_second_job
  (hourly_rate_1 hourly_rate_2 hourly_rate_3 : ℚ)
  (hours_1 hours_3 : ℚ)
  (days : ℚ)
  (total_earnings : ℚ)
  (h1 : hourly_rate_1 = 7)
  (h2 : hourly_rate_2 = 10)
  (h3 : hourly_rate_3 = 12)
  (h4 : hours_1 = 3)
  (h5 : hours_3 = 4)
  (h6 : days = 5)
  (h7 : total_earnings = 445)
  : ∃ hours_2 : ℚ, hours_2 = 2 ∧ 
    days * (hourly_rate_1 * hours_1 + hourly_rate_2 * hours_2 + hourly_rate_3 * hours_3) = total_earnings :=
by sorry

end NUMINAMATH_CALUDE_hours_worked_on_second_job_l2475_247588


namespace NUMINAMATH_CALUDE_octagon_lines_l2475_247501

/-- The number of sides in a triangle -/
def triangle_sides : ℕ := 3

/-- The number of sides in a square -/
def square_sides : ℕ := 4

/-- The number of sides in a pentagon -/
def pentagon_sides : ℕ := 5

/-- The number of sides in a hexagon -/
def hexagon_sides : ℕ := 6

/-- The number of sides in an octagon -/
def octagon_sides : ℕ := 8

/-- The number of triangles Bill drew -/
def num_triangles : ℕ := 12

/-- The number of squares Bill drew -/
def num_squares : ℕ := 8

/-- The number of pentagons Bill drew -/
def num_pentagons : ℕ := 4

/-- The number of hexagons Bill drew -/
def num_hexagons : ℕ := 6

/-- The number of octagons Bill drew -/
def num_octagons : ℕ := 2

/-- The number of lines shared between triangles and squares -/
def shared_triangle_square : ℕ := 5

/-- The number of lines shared between pentagons and hexagons -/
def shared_pentagon_hexagon : ℕ := 3

/-- The number of lines shared between hexagons and octagons -/
def shared_hexagon_octagon : ℕ := 1

/-- Theorem: The number of lines drawn with the purple marker (for octagons) is 15 -/
theorem octagon_lines : 
  num_octagons * octagon_sides - shared_hexagon_octagon = 15 := by sorry

end NUMINAMATH_CALUDE_octagon_lines_l2475_247501


namespace NUMINAMATH_CALUDE_negation_of_universal_quantifier_l2475_247566

theorem negation_of_universal_quantifier (a : ℝ) :
  (¬ ∀ x > 0, Real.log x = a) ↔ (∃ x > 0, Real.log x ≠ a) := by sorry

end NUMINAMATH_CALUDE_negation_of_universal_quantifier_l2475_247566


namespace NUMINAMATH_CALUDE_midpoint_locus_l2475_247584

/-- Given real numbers a, b, c forming an arithmetic sequence,
    prove that the locus of the midpoint of the chord of the line
    bx + ay + c = 0 intersecting the parabola y^2 = -1/2 x
    is described by the equation x + 1 = -(2y - 1)^2 -/
theorem midpoint_locus (a b c : ℝ) :
  (2 * b = a + c) →
  ∃ (x y : ℝ), 
    (∃ (x₁ y₁ : ℝ), 
      b * x₁ + a * y₁ + c = 0 ∧ 
      y₁^2 = -1/2 * x₁ ∧
      x = (x₁ - 2) / 2 ∧
      y = (y₁ + 1) / 2) →
    x + 1 = -(2 * y - 1)^2 :=
by sorry

end NUMINAMATH_CALUDE_midpoint_locus_l2475_247584


namespace NUMINAMATH_CALUDE_cube_vertex_sum_difference_l2475_247526

theorem cube_vertex_sum_difference (a b c d e f g h : ℝ) 
  (ha : 3 * a = b + e + d)
  (hb : 3 * b = c + f + a)
  (hc : 3 * c = d + g + b)
  (hd : 3 * d = a + h + c)
  (he : 3 * e = f + a + h)
  (hf : 3 * f = g + b + e)
  (hg : 3 * g = h + c + f)
  (hh : 3 * h = e + d + g) :
  (a + b + c + d) - (e + f + g + h) = 0 := by
  sorry

end NUMINAMATH_CALUDE_cube_vertex_sum_difference_l2475_247526


namespace NUMINAMATH_CALUDE_divisor_problem_l2475_247571

theorem divisor_problem (dividend : ℕ) (quotient : ℕ) (remainder : ℕ) (divisor : ℕ) :
  dividend = 165 →
  quotient = 9 →
  remainder = 3 →
  dividend = divisor * quotient + remainder →
  divisor = 18 := by
sorry

end NUMINAMATH_CALUDE_divisor_problem_l2475_247571


namespace NUMINAMATH_CALUDE_store_a_cheaper_condition_store_b_cheaper_condition_store_a_cheaper_at_100_most_cost_effective_plan_is_best_l2475_247554

/-- Represents the cost of purchasing from Store A or B -/
def store_cost (x : ℝ) (is_store_a : Bool) : ℝ :=
  if is_store_a then 20 * x + 2400 else 18 * x + 2700

/-- Theorem stating the conditions under which Store A is cheaper than Store B -/
theorem store_a_cheaper_condition (x : ℝ) (h1 : x > 30) :
  store_cost x true < store_cost x false ↔ x < 150 :=
sorry

/-- Theorem stating the conditions under which Store B is cheaper than Store A -/
theorem store_b_cheaper_condition (x : ℝ) (h1 : x > 30) :
  store_cost x false < store_cost x true ↔ x > 150 :=
sorry

/-- Theorem proving that for x = 100, Store A is cheaper -/
theorem store_a_cheaper_at_100 :
  store_cost 100 true < store_cost 100 false :=
sorry

/-- Definition of the cost for the most cost-effective plan when x = 100 -/
def most_cost_effective_plan : ℝ := 3000 + 20 * 70 * 0.9

/-- Theorem proving that the most cost-effective plan is cheaper than both Store A and B when x = 100 -/
theorem most_cost_effective_plan_is_best :
  most_cost_effective_plan < store_cost 100 true ∧
  most_cost_effective_plan < store_cost 100 false :=
sorry

end NUMINAMATH_CALUDE_store_a_cheaper_condition_store_b_cheaper_condition_store_a_cheaper_at_100_most_cost_effective_plan_is_best_l2475_247554


namespace NUMINAMATH_CALUDE_favorite_color_survey_l2475_247542

theorem favorite_color_survey (total_students : ℕ) (total_girls : ℕ) 
  (h1 : total_students = 30)
  (h2 : total_girls = 18)
  (h3 : total_students / 2 = total_students - total_girls + total_girls / 3 + 9) :
  9 = total_students - (total_students / 2 + total_girls / 3) :=
by sorry

end NUMINAMATH_CALUDE_favorite_color_survey_l2475_247542


namespace NUMINAMATH_CALUDE_min_product_of_three_l2475_247545

def S : Finset Int := {-9, -7, -1, 2, 4, 6, 8}

theorem min_product_of_three (a b c : Int) (ha : a ∈ S) (hb : b ∈ S) (hc : c ∈ S)
  (hdiff : a ≠ b ∧ b ≠ c ∧ a ≠ c) :
  ∃ (x y z : Int), x ∈ S ∧ y ∈ S ∧ z ∈ S ∧ x ≠ y ∧ y ≠ z ∧ x ≠ z ∧
  x * y * z = -432 ∧ (∀ (p q r : Int), p ∈ S → q ∈ S → r ∈ S →
  p ≠ q → q ≠ r → p ≠ r → p * q * r ≥ -432) :=
sorry

end NUMINAMATH_CALUDE_min_product_of_three_l2475_247545


namespace NUMINAMATH_CALUDE_point_coordinates_wrt_origin_l2475_247590

/-- In a plane rectangular coordinate system, the coordinates of a point
    with respect to the origin are equal to its given coordinates. -/
theorem point_coordinates_wrt_origin (x y : ℝ) :
  let A : ℝ × ℝ := (x, y)
  A = (x, y) := by sorry

end NUMINAMATH_CALUDE_point_coordinates_wrt_origin_l2475_247590


namespace NUMINAMATH_CALUDE_no_solution_range_l2475_247550

theorem no_solution_range (a : ℝ) : 
  (∀ x : ℝ, |x + a + 1| + |x + a^2 - 2| ≥ 3) ↔ 
  (a ≤ -2 ∨ (0 ≤ a ∧ a ≤ 1) ∨ 3 ≤ a) :=
sorry

end NUMINAMATH_CALUDE_no_solution_range_l2475_247550


namespace NUMINAMATH_CALUDE_inverse_mod_53_l2475_247528

theorem inverse_mod_53 (h : (17⁻¹ : ZMod 53) = 13) : (36⁻¹ : ZMod 53) = 40 := by
  sorry

end NUMINAMATH_CALUDE_inverse_mod_53_l2475_247528


namespace NUMINAMATH_CALUDE_thirteenth_digit_of_sum_l2475_247585

def decimal_sum (a b : ℚ) : ℚ := a + b

def nth_digit_after_decimal (q : ℚ) (n : ℕ) : ℕ := sorry

theorem thirteenth_digit_of_sum :
  let sum := decimal_sum (1/8) (1/11)
  nth_digit_after_decimal sum 13 = 9 := by
  sorry

end NUMINAMATH_CALUDE_thirteenth_digit_of_sum_l2475_247585


namespace NUMINAMATH_CALUDE_arrangement_schemes_eq_twelve_l2475_247555

/-- The number of ways to divide 2 teachers and 4 students into 2 groups -/
def arrangement_schemes : ℕ :=
  (Nat.choose 2 1) * (Nat.choose 4 2)

/-- Theorem stating that the number of arrangement schemes is 12 -/
theorem arrangement_schemes_eq_twelve :
  arrangement_schemes = 12 := by
  sorry

end NUMINAMATH_CALUDE_arrangement_schemes_eq_twelve_l2475_247555


namespace NUMINAMATH_CALUDE_proportion_solution_l2475_247537

theorem proportion_solution (x : ℝ) : (0.25 / x = 2 / 6) → x = 0.75 := by
  sorry

end NUMINAMATH_CALUDE_proportion_solution_l2475_247537


namespace NUMINAMATH_CALUDE_employee_payment_percentage_l2475_247587

theorem employee_payment_percentage (total_payment : ℝ) (b_payment : ℝ) :
  total_payment = 450 ∧ b_payment = 180 →
  (total_payment - b_payment) / b_payment * 100 = 150 := by
sorry

end NUMINAMATH_CALUDE_employee_payment_percentage_l2475_247587


namespace NUMINAMATH_CALUDE_log_inequality_l2475_247598

theorem log_inequality (a b c : ℝ) (h1 : 0 < a) (h2 : a < b) (h3 : b < 1) (h4 : 1 < c) :
  Real.log c / Real.log a > Real.log c / Real.log b :=
by sorry

end NUMINAMATH_CALUDE_log_inequality_l2475_247598


namespace NUMINAMATH_CALUDE_at_least_95_buildings_collapsed_l2475_247567

/-- Represents the number of buildings that collapsed in each earthquake --/
structure EarthquakeCollapses where
  first : ℕ
  second : ℕ
  third : ℕ
  fourth : ℕ

/-- Theorem stating that at least 95 buildings collapsed after five earthquakes --/
theorem at_least_95_buildings_collapsed
  (initial_buildings : ℕ)
  (collapses : EarthquakeCollapses)
  (h_initial : initial_buildings = 100)
  (h_first : collapses.first = 5)
  (h_second : collapses.second = 6)
  (h_third : collapses.third = 13)
  (h_fourth : collapses.fourth = 24)
  (h_handful : ∀ n : ℕ, n ≤ 5 → n ≤ initial_buildings - (collapses.first + collapses.second + collapses.third + collapses.fourth)) :
  95 ≤ collapses.first + collapses.second + collapses.third + collapses.fourth :=
sorry

end NUMINAMATH_CALUDE_at_least_95_buildings_collapsed_l2475_247567


namespace NUMINAMATH_CALUDE_count_multiples_count_multiples_equals_1002_l2475_247572

theorem count_multiples : ℕ :=
  let range_start := 1
  let range_end := 2005
  let count_multiples_of_3 := (range_end / 3 : ℕ)
  let count_multiples_of_4 := (range_end / 4 : ℕ)
  let count_multiples_of_12 := (range_end / 12 : ℕ)
  count_multiples_of_3 + count_multiples_of_4 - count_multiples_of_12

theorem count_multiples_equals_1002 : count_multiples = 1002 := by
  sorry

end NUMINAMATH_CALUDE_count_multiples_count_multiples_equals_1002_l2475_247572


namespace NUMINAMATH_CALUDE_quadratic_form_equivalence_l2475_247577

theorem quadratic_form_equivalence (b m : ℝ) : 
  b > 0 → 
  (∀ x, x^2 + b*x + 56 = (x + m)^2 + 20) → 
  b = 12 := by
sorry

end NUMINAMATH_CALUDE_quadratic_form_equivalence_l2475_247577


namespace NUMINAMATH_CALUDE_modulo_equivalence_problem_l2475_247507

theorem modulo_equivalence_problem : ∃! n : ℕ, 0 ≤ n ∧ n ≤ 14 ∧ n ≡ 15478 [MOD 15] ∧ n = 13 := by
  sorry

end NUMINAMATH_CALUDE_modulo_equivalence_problem_l2475_247507


namespace NUMINAMATH_CALUDE_geometric_sequence_sum_l2475_247524

theorem geometric_sequence_sum (a : ℕ) : 
  let seq := [a, 2*a, 4*a, 8*a, 16*a, 32*a]
  ∀ (x y z w : ℕ), x ∈ seq → y ∈ seq → z ∈ seq → w ∈ seq →
  x ≠ y ∧ z ≠ w ∧ x ≠ z ∧ x ≠ w ∧ y ≠ z ∧ y ≠ w →
  x + y = 136 →
  z + w = 272 →
  ∃ (p q : ℕ), p ∈ seq ∧ q ∈ seq ∧ p ≠ q ∧ p + q = 96 :=
by sorry

end NUMINAMATH_CALUDE_geometric_sequence_sum_l2475_247524


namespace NUMINAMATH_CALUDE_regular_polygon_sides_l2475_247519

theorem regular_polygon_sides (n : ℕ) (h : n > 2) : 
  (((n - 2) * 180) / n = 150) → n = 12 := by
  sorry

end NUMINAMATH_CALUDE_regular_polygon_sides_l2475_247519


namespace NUMINAMATH_CALUDE_f_is_quadratic_l2475_247599

/-- Definition of a quadratic equation in one variable -/
def is_quadratic_equation (f : ℝ → ℝ) : Prop :=
  ∃ (a b c : ℝ), a ≠ 0 ∧ ∀ x, f x = a * x^2 + b * x + c

/-- The equation x^2 - 1 = 0 -/
def f (x : ℝ) : ℝ := x^2 - 1

/-- Theorem: f is a quadratic equation in one variable -/
theorem f_is_quadratic : is_quadratic_equation f := by
  sorry

end NUMINAMATH_CALUDE_f_is_quadratic_l2475_247599


namespace NUMINAMATH_CALUDE_inscribed_squares_ratio_l2475_247553

/-- Given two right triangles with sides 5, 12, and 13, where a square of side length x
    is inscribed in the first triangle with a vertex coinciding with the right angle,
    and a square of side length y is inscribed in the second triangle with a side lying
    on the hypotenuse, the ratio x/y equals 12/13. -/
theorem inscribed_squares_ratio (x y : ℝ) : 
  (x > 0 ∧ y > 0) →
  (5^2 + 12^2 = 13^2) →
  (x / 12 = x / 5) →
  ((12 - y) / y = (5 - y) / y) →
  x / y = 12 / 13 := by sorry

end NUMINAMATH_CALUDE_inscribed_squares_ratio_l2475_247553


namespace NUMINAMATH_CALUDE_yuan_yuan_delivery_cost_l2475_247583

def express_delivery_cost (weight : ℕ) : ℕ :=
  let base_fee := 13
  let weight_limit := 5
  let additional_fee := 2
  if weight ≤ weight_limit then
    base_fee
  else
    base_fee + (weight - weight_limit) * additional_fee

theorem yuan_yuan_delivery_cost :
  express_delivery_cost 7 = 17 := by sorry

end NUMINAMATH_CALUDE_yuan_yuan_delivery_cost_l2475_247583


namespace NUMINAMATH_CALUDE_girls_not_adjacent_arrangements_l2475_247578

theorem girls_not_adjacent_arrangements :
  let num_boys : ℕ := 4
  let num_girls : ℕ := 4
  let total_people : ℕ := num_boys + num_girls
  let num_spaces : ℕ := num_boys + 1
  
  (num_boys.factorial * num_spaces.factorial) = 2880 :=
by sorry

end NUMINAMATH_CALUDE_girls_not_adjacent_arrangements_l2475_247578


namespace NUMINAMATH_CALUDE_exists_line_not_through_lattice_points_l2475_247582

-- Define a 2D point
structure Point where
  x : ℝ
  y : ℝ

-- Define a line by its slope and y-intercept
structure Line where
  slope : ℝ
  intercept : ℝ

-- Define a lattice point (grid point)
def isLatticePoint (p : Point) : Prop :=
  ∃ (m n : ℤ), p.x = m ∧ p.y = n

-- Define when a point is on a line
def pointOnLine (p : Point) (l : Line) : Prop :=
  p.y = l.slope * p.x + l.intercept

-- Theorem statement
theorem exists_line_not_through_lattice_points :
  ∃ (l : Line), ∀ (p : Point), isLatticePoint p → ¬ pointOnLine p l :=
sorry

end NUMINAMATH_CALUDE_exists_line_not_through_lattice_points_l2475_247582


namespace NUMINAMATH_CALUDE_arithmetic_calculation_l2475_247511

theorem arithmetic_calculation : (18 / (8 - 2 * 3)) + 4 = 13 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_calculation_l2475_247511


namespace NUMINAMATH_CALUDE_scout_troop_profit_l2475_247515

/-- Calculates the profit of a scout troop selling candy bars -/
theorem scout_troop_profit (num_bars : ℕ) (buy_price : ℚ) (sell_price : ℚ) : 
  num_bars = 1500 → 
  buy_price = 1/3 → 
  sell_price = 2/3 → 
  (sell_price - buy_price) * num_bars = 500 := by
  sorry

#check scout_troop_profit

end NUMINAMATH_CALUDE_scout_troop_profit_l2475_247515


namespace NUMINAMATH_CALUDE_aunt_may_milk_problem_l2475_247573

/-- Aunt May's milk problem -/
theorem aunt_may_milk_problem 
  (morning_milk : ℕ) 
  (evening_milk : ℕ) 
  (sold_milk : ℕ) 
  (leftover_milk : ℕ) 
  (h1 : morning_milk = 365)
  (h2 : evening_milk = 380)
  (h3 : sold_milk = 612)
  (h4 : leftover_milk = 15) :
  morning_milk + evening_milk + leftover_milk - sold_milk = 148 := by
sorry

end NUMINAMATH_CALUDE_aunt_may_milk_problem_l2475_247573


namespace NUMINAMATH_CALUDE_quadratic_always_nonnegative_implies_a_bounded_l2475_247508

theorem quadratic_always_nonnegative_implies_a_bounded (a : ℝ) :
  (∀ x : ℝ, 2 * x^2 - 3 * a * x + 9 ≥ 0) →
  -2 * Real.sqrt 2 ≤ a ∧ a ≤ 2 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_always_nonnegative_implies_a_bounded_l2475_247508


namespace NUMINAMATH_CALUDE_cylinder_volume_from_lateral_surface_l2475_247505

/-- The volume of a cylinder whose lateral surface is a square with side length 2 * (π^(1/3)) is 2 -/
theorem cylinder_volume_from_lateral_surface (π : ℝ) (h : π > 0) :
  let lateral_surface_side := 2 * π^(1/3)
  let cylinder_height := lateral_surface_side
  let cylinder_radius := lateral_surface_side / (2 * π)
  let cylinder_volume := π * cylinder_radius^2 * cylinder_height
  cylinder_volume = 2 := by
  sorry

end NUMINAMATH_CALUDE_cylinder_volume_from_lateral_surface_l2475_247505


namespace NUMINAMATH_CALUDE_existence_of_zero_crossing_l2475_247533

open Function Set

theorem existence_of_zero_crossing (a b : ℝ) (h : a < b) :
  ∃ (f : ℝ → ℝ), ContinuousOn f (Icc a b) ∧ 
  f a * f b > 0 ∧ 
  ∃ c ∈ Ioo a b, f c = 0 := by
  sorry

end NUMINAMATH_CALUDE_existence_of_zero_crossing_l2475_247533


namespace NUMINAMATH_CALUDE_milk_for_six_cookies_l2475_247500

/-- Calculates the amount of milk needed for a given number of cookies. -/
def milk_needed (cookies : ℕ) : ℚ :=
  (5000 : ℚ) * cookies / 24

theorem milk_for_six_cookies :
  milk_needed 6 = 1250 := by sorry


end NUMINAMATH_CALUDE_milk_for_six_cookies_l2475_247500


namespace NUMINAMATH_CALUDE_no_real_solutions_for_equation_l2475_247565

theorem no_real_solutions_for_equation :
  ∀ x : ℝ, x + Real.sqrt (2 * x - 3) ≠ 5 := by
  sorry

end NUMINAMATH_CALUDE_no_real_solutions_for_equation_l2475_247565


namespace NUMINAMATH_CALUDE_pistachio_problem_l2475_247594

theorem pistachio_problem (total : ℕ) (shell_percent : ℚ) (open_percent : ℚ) 
  (h1 : total = 80)
  (h2 : shell_percent = 95 / 100)
  (h3 : open_percent = 75 / 100) :
  ⌊(shell_percent * total : ℚ) * open_percent⌋ = 57 := by
sorry

#eval ⌊(95 / 100 : ℚ) * 80 * (75 / 100 : ℚ)⌋

end NUMINAMATH_CALUDE_pistachio_problem_l2475_247594


namespace NUMINAMATH_CALUDE_expansion_terms_count_l2475_247556

def expandedTerms (N : ℕ) : ℕ := Nat.choose N 4

theorem expansion_terms_count : expandedTerms 14 = 1001 := by
  sorry

end NUMINAMATH_CALUDE_expansion_terms_count_l2475_247556


namespace NUMINAMATH_CALUDE_total_books_collected_l2475_247512

def books_first_week : ℕ := 9
def weeks_collecting : ℕ := 6
def multiplier : ℕ := 10

theorem total_books_collected :
  (books_first_week + (weeks_collecting - 1) * (books_first_week * multiplier)) = 459 :=
by sorry

end NUMINAMATH_CALUDE_total_books_collected_l2475_247512


namespace NUMINAMATH_CALUDE_correct_coverings_8x8_l2475_247559

/-- Represents a square grid -/
structure Grid :=
  (size : ℕ)

/-- Represents a covering of the grid with colored triangles -/
def Covering (g : Grid) := Unit

/-- Checks if a covering is correct (adjacent triangles have different colors) -/
def is_correct_covering (g : Grid) (c : Covering g) : Prop := sorry

/-- Counts the number of correct coverings for a given grid -/
def count_correct_coverings (g : Grid) : ℕ := sorry

/-- Theorem: The number of correct coverings for an 8x8 grid is 2^16 -/
theorem correct_coverings_8x8 :
  let g : Grid := ⟨8⟩
  count_correct_coverings g = 2^16 := by sorry

end NUMINAMATH_CALUDE_correct_coverings_8x8_l2475_247559


namespace NUMINAMATH_CALUDE_emily_age_is_23_l2475_247581

-- Define the ages as natural numbers
def uncle_bob_age : ℕ := 54
def daniel_age : ℕ := uncle_bob_age / 2
def emily_age : ℕ := daniel_age - 4
def zoe_age : ℕ := emily_age * 3 / 2

-- Theorem statement
theorem emily_age_is_23 : emily_age = 23 := by
  sorry

end NUMINAMATH_CALUDE_emily_age_is_23_l2475_247581


namespace NUMINAMATH_CALUDE_geometric_sequence_ratio_sum_l2475_247513

theorem geometric_sequence_ratio_sum 
  (m x y : ℝ) 
  (h_m : m ≠ 0) 
  (h_x_ne_y : x ≠ y) 
  (h_nonconstant : x ≠ 1 ∧ y ≠ 1) 
  (h_eq : m * x^2 - m * y^2 = 3 * (m * x - m * y)) : 
  x + y = 3 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_ratio_sum_l2475_247513


namespace NUMINAMATH_CALUDE_shoe_price_calculation_l2475_247522

theorem shoe_price_calculation (discount_rate : ℝ) (savings : ℝ) (original_price : ℝ) : 
  discount_rate = 0.30 →
  savings = 46 →
  original_price = savings / discount_rate →
  original_price = 153.33 := by
sorry

end NUMINAMATH_CALUDE_shoe_price_calculation_l2475_247522


namespace NUMINAMATH_CALUDE_square_root_plus_square_eq_zero_l2475_247516

theorem square_root_plus_square_eq_zero (x y : ℝ) :
  Real.sqrt (x + 2) + (x + y)^2 = 0 → x^2 - x*y = 8 := by
  sorry

end NUMINAMATH_CALUDE_square_root_plus_square_eq_zero_l2475_247516


namespace NUMINAMATH_CALUDE_odd_periodic_two_at_one_l2475_247514

/-- A function f: ℝ → ℝ is odd if f(-x) = -f(x) for all x ∈ ℝ -/
def IsOdd (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

/-- A function f: ℝ → ℝ has period 2 if f(x + 2) = f(x) for all x ∈ ℝ -/
def HasPeriodTwo (f : ℝ → ℝ) : Prop :=
  ∀ x, f (x + 2) = f x

/-- For a function f: ℝ → ℝ, if f is odd and has a period of 2, then f(1) = 0 -/
theorem odd_periodic_two_at_one (f : ℝ → ℝ) (h_odd : IsOdd f) (h_period : HasPeriodTwo f) :
  f 1 = 0 := by
  sorry

end NUMINAMATH_CALUDE_odd_periodic_two_at_one_l2475_247514


namespace NUMINAMATH_CALUDE_triangle_perimeter_l2475_247520

/-- Given a triangle with inradius 2.5 cm and area 35 cm², its perimeter is 28 cm. -/
theorem triangle_perimeter (r : ℝ) (A : ℝ) (p : ℝ) 
  (h1 : r = 2.5)
  (h2 : A = 35)
  (h3 : A = r * (p / 2)) :
  p = 28 := by
  sorry

end NUMINAMATH_CALUDE_triangle_perimeter_l2475_247520


namespace NUMINAMATH_CALUDE_travel_options_l2475_247543

theorem travel_options (train_services : ℕ) (ferry_services : ℕ) : 
  train_services = 3 → ferry_services = 2 → train_services * ferry_services = 6 := by
  sorry

#check travel_options

end NUMINAMATH_CALUDE_travel_options_l2475_247543


namespace NUMINAMATH_CALUDE_parabola_c_value_l2475_247576

/-- A parabola passing through two specific points has a determined c-value. -/
theorem parabola_c_value (b c : ℝ) : 
  (2 = 1^2 + b*1 + c) ∧ (2 = 5^2 + b*5 + c) → c = 7 := by
  sorry

end NUMINAMATH_CALUDE_parabola_c_value_l2475_247576


namespace NUMINAMATH_CALUDE_hundredth_number_is_hundred_l2475_247532

def counting_sequence (n : ℕ) : ℕ := n

theorem hundredth_number_is_hundred :
  counting_sequence 100 = 100 := by
  sorry

end NUMINAMATH_CALUDE_hundredth_number_is_hundred_l2475_247532


namespace NUMINAMATH_CALUDE_lcm_of_36_and_176_l2475_247597

theorem lcm_of_36_and_176 :
  let a : ℕ := 36
  let b : ℕ := 176
  let hcf : ℕ := 16
  Nat.gcd a b = hcf →
  Nat.lcm a b = 396 := by
sorry

end NUMINAMATH_CALUDE_lcm_of_36_and_176_l2475_247597


namespace NUMINAMATH_CALUDE_convex_lattice_polygon_vertices_l2475_247535

/-- A lattice point in 2D space -/
structure LatticePoint where
  x : ℤ
  y : ℤ

/-- A convex polygon defined by its vertices -/
structure ConvexPolygon where
  vertices : List LatticePoint
  is_convex : Bool  -- Assume this is true for our polygon

/-- Checks if a point is inside or on the sides of a polygon -/
def is_inside_or_on_sides (point : LatticePoint) (polygon : ConvexPolygon) : Bool :=
  sorry  -- Implementation details omitted

theorem convex_lattice_polygon_vertices (polygon : ConvexPolygon) :
  (∀ point : LatticePoint, point ∉ polygon.vertices → ¬(is_inside_or_on_sides point polygon)) →
  List.length polygon.vertices ≤ 4 :=
sorry

end NUMINAMATH_CALUDE_convex_lattice_polygon_vertices_l2475_247535


namespace NUMINAMATH_CALUDE_g_eval_l2475_247549

def g (x : ℝ) : ℝ := 2 * x^2 - 2 * x + 11

theorem g_eval : 3 * g 2 + 2 * g (-4) = 147 := by sorry

end NUMINAMATH_CALUDE_g_eval_l2475_247549


namespace NUMINAMATH_CALUDE_randy_store_trips_l2475_247557

theorem randy_store_trips (initial_amount : ℕ) (final_amount : ℕ) (amount_per_trip : ℕ) (months_per_year : ℕ) :
  initial_amount = 200 →
  final_amount = 104 →
  amount_per_trip = 2 →
  months_per_year = 12 →
  (initial_amount - final_amount) / amount_per_trip / months_per_year = 4 := by
  sorry

end NUMINAMATH_CALUDE_randy_store_trips_l2475_247557


namespace NUMINAMATH_CALUDE_star_equation_solution_l2475_247504

-- Define the star operation
def star (a b : ℝ) : ℝ := 2*a*b + 3*b - 2*a

-- Theorem statement
theorem star_equation_solution :
  ∀ x : ℝ, star 3 x = 60 → x = 22/3 :=
by
  sorry

end NUMINAMATH_CALUDE_star_equation_solution_l2475_247504


namespace NUMINAMATH_CALUDE_triangle_ratio_l2475_247529

-- Define a triangle ABC
structure Triangle where
  A : Real
  B : Real
  C : Real
  a : Real
  b : Real
  c : Real

-- State the theorem
theorem triangle_ratio (t : Triangle) 
  (h1 : t.A = π / 3)
  (h2 : Real.sin (t.B + t.C) = 6 * Real.cos t.B * Real.sin t.C) :
  t.b / t.c = (1 + Real.sqrt 21) / 2 := by
  sorry


end NUMINAMATH_CALUDE_triangle_ratio_l2475_247529


namespace NUMINAMATH_CALUDE_pencils_across_diameter_l2475_247530

theorem pencils_across_diameter (radius : ℝ) (pencil_length : ℝ) : 
  radius = 14 → pencil_length = 0.5 → 
  (2 * radius * 12) / pencil_length = 56 := by
  sorry

end NUMINAMATH_CALUDE_pencils_across_diameter_l2475_247530


namespace NUMINAMATH_CALUDE_additional_monthly_income_l2475_247531

/-- Given a shoe company's current monthly sales and desired annual income,
    calculate the additional monthly income required to reach the annual goal. -/
theorem additional_monthly_income
  (current_monthly_sales : ℕ)
  (desired_annual_income : ℕ)
  (h1 : current_monthly_sales = 4000)
  (h2 : desired_annual_income = 60000) :
  (desired_annual_income - current_monthly_sales * 12) / 12 = 1000 :=
by sorry

end NUMINAMATH_CALUDE_additional_monthly_income_l2475_247531


namespace NUMINAMATH_CALUDE_cube_plus_reciprocal_cube_l2475_247570

theorem cube_plus_reciprocal_cube (a : ℝ) (h : (a + 1/a)^2 = 3) :
  a^3 + 1/a^3 = 0 := by
  sorry

end NUMINAMATH_CALUDE_cube_plus_reciprocal_cube_l2475_247570
