import Mathlib

namespace NUMINAMATH_CALUDE_random_selection_probability_l2736_273698

theorem random_selection_probability (a : ℝ) : a > 0 → (∃ m : ℝ, 0 ≤ m ∧ m ≤ a) → (2 / a = 1 / 3) → a = 6 := by
  sorry

end NUMINAMATH_CALUDE_random_selection_probability_l2736_273698


namespace NUMINAMATH_CALUDE_tangent_line_at_one_l2736_273606

/-- The function f(x) = -x³ + 4x -/
def f (x : ℝ) : ℝ := -x^3 + 4*x

/-- The derivative of f(x) -/
def f_derivative (x : ℝ) : ℝ := -3*x^2 + 4

theorem tangent_line_at_one :
  let x₀ : ℝ := 1
  let y₀ : ℝ := f x₀
  let m : ℝ := f_derivative x₀
  ∀ x y : ℝ, y - y₀ = m * (x - x₀) ↔ y = x + 2 :=
by sorry

end NUMINAMATH_CALUDE_tangent_line_at_one_l2736_273606


namespace NUMINAMATH_CALUDE_intersection_implies_sum_l2736_273649

-- Define the functions
def f (x p q : ℝ) : ℝ := -|x - p| + q
def g (x r s : ℝ) : ℝ := |x - r| + s

-- State the theorem
theorem intersection_implies_sum (p q r s : ℝ) :
  (f 3 p q = g 3 r s) ∧ 
  (f 5 p q = g 5 r s) ∧ 
  (f 3 p q = 6) ∧ 
  (f 5 p q = 2) ∧ 
  (g 3 r s = 6) ∧ 
  (g 5 r s = 2) →
  p + r = 8 := by sorry

end NUMINAMATH_CALUDE_intersection_implies_sum_l2736_273649


namespace NUMINAMATH_CALUDE_specific_ellipse_intercept_l2736_273688

/-- Definition of an ellipse with given foci and one x-intercept -/
structure Ellipse where
  foci1 : ℝ × ℝ
  foci2 : ℝ × ℝ
  x_intercept1 : ℝ × ℝ
  sum_distances : ℝ

/-- The other x-intercept of the ellipse -/
def other_x_intercept (e : Ellipse) : ℝ × ℝ := sorry

/-- Theorem stating the other x-intercept of the specific ellipse -/
theorem specific_ellipse_intercept :
  let e : Ellipse := {
    foci1 := (0, 3),
    foci2 := (4, 0),
    x_intercept1 := (0, 0),
    sum_distances := 7
  }
  other_x_intercept e = (56/11, 0) := by sorry

end NUMINAMATH_CALUDE_specific_ellipse_intercept_l2736_273688


namespace NUMINAMATH_CALUDE_equilateral_triangle_area_decrease_l2736_273682

/-- The decrease in area of an equilateral triangle when its sides are shortened --/
theorem equilateral_triangle_area_decrease :
  ∀ (s : ℝ),
  s > 0 →
  (s^2 * Real.sqrt 3) / 4 = 100 * Real.sqrt 3 →
  let new_s := s - 3
  let original_area := (s^2 * Real.sqrt 3) / 4
  let new_area := (new_s^2 * Real.sqrt 3) / 4
  original_area - new_area = 27.75 * Real.sqrt 3 :=
by sorry

end NUMINAMATH_CALUDE_equilateral_triangle_area_decrease_l2736_273682


namespace NUMINAMATH_CALUDE_intersection_M_N_l2736_273654

def M : Set ℤ := {-1, 0, 1}
def N : Set ℤ := {x | x^2 = x}

theorem intersection_M_N : M ∩ N = {0, 1} := by sorry

end NUMINAMATH_CALUDE_intersection_M_N_l2736_273654


namespace NUMINAMATH_CALUDE_david_rosy_age_difference_l2736_273624

theorem david_rosy_age_difference :
  ∀ (david_age rosy_age : ℕ),
    david_age > rosy_age →
    rosy_age = 8 →
    david_age + 4 = 2 * (rosy_age + 4) →
    david_age - rosy_age = 12 := by
  sorry

end NUMINAMATH_CALUDE_david_rosy_age_difference_l2736_273624


namespace NUMINAMATH_CALUDE_toys_ratio_saturday_to_wednesday_l2736_273646

/-- The number of rabbits Junior has -/
def num_rabbits : ℕ := 16

/-- The number of toys bought on Monday -/
def toys_monday : ℕ := 6

/-- The number of toys bought on Wednesday -/
def toys_wednesday : ℕ := 2 * toys_monday

/-- The number of toys bought on Friday -/
def toys_friday : ℕ := 4 * toys_monday

/-- The number of toys each rabbit has when split evenly -/
def toys_per_rabbit : ℕ := 3

/-- The total number of toys -/
def total_toys : ℕ := num_rabbits * toys_per_rabbit

/-- The number of toys bought on Saturday -/
def toys_saturday : ℕ := total_toys - (toys_monday + toys_wednesday + toys_friday)

theorem toys_ratio_saturday_to_wednesday :
  toys_saturday * 2 = toys_wednesday := by sorry

end NUMINAMATH_CALUDE_toys_ratio_saturday_to_wednesday_l2736_273646


namespace NUMINAMATH_CALUDE_smallest_value_of_sum_of_cubes_l2736_273607

theorem smallest_value_of_sum_of_cubes (a b : ℂ) 
  (h1 : Complex.abs (a + b) = 2)
  (h2 : Complex.abs (a^2 + b^2) = 8) :
  20 ≤ Complex.abs (a^3 + b^3) :=
by sorry

end NUMINAMATH_CALUDE_smallest_value_of_sum_of_cubes_l2736_273607


namespace NUMINAMATH_CALUDE_work_time_for_less_efficient_worker_l2736_273643

/-- Represents the time it takes for a worker to complete a job alone -/
def WorkTime := ℝ

/-- Represents the efficiency of a worker (fraction of job completed per day) -/
def Efficiency := ℝ

theorem work_time_for_less_efficient_worker 
  (total_time : ℝ) 
  (efficiency_ratio : ℝ) :
  total_time > 0 →
  efficiency_ratio > 1 →
  let joint_efficiency := 1 / total_time
  let less_efficient_worker_efficiency := joint_efficiency / (1 + efficiency_ratio)
  let work_time_less_efficient := 1 / less_efficient_worker_efficiency
  (total_time = 36 ∧ efficiency_ratio = 2) → work_time_less_efficient = 108 := by
  sorry

end NUMINAMATH_CALUDE_work_time_for_less_efficient_worker_l2736_273643


namespace NUMINAMATH_CALUDE_point_on_x_axis_l2736_273610

theorem point_on_x_axis (m : ℝ) : (∃ P : ℝ × ℝ, P.1 = m + 5 ∧ P.2 = m - 2 ∧ P.2 = 0) → m = 2 := by
  sorry

end NUMINAMATH_CALUDE_point_on_x_axis_l2736_273610


namespace NUMINAMATH_CALUDE_inequality_solution_l2736_273689

theorem inequality_solution (x : ℝ) : (x^2 - 4) / (x^2 - 9) > 0 ↔ x < -3 ∨ x > 3 :=
by sorry

end NUMINAMATH_CALUDE_inequality_solution_l2736_273689


namespace NUMINAMATH_CALUDE_harry_tomato_packets_l2736_273603

/-- Represents the number of packets of tomato seeds Harry bought -/
def tomato_packets : ℕ := sorry

/-- The price of a packet of pumpkin seeds in dollars -/
def pumpkin_price : ℚ := 5/2

/-- The price of a packet of tomato seeds in dollars -/
def tomato_price : ℚ := 3/2

/-- The price of a packet of chili pepper seeds in dollars -/
def chili_price : ℚ := 9/10

/-- The number of packets of pumpkin seeds Harry bought -/
def pumpkin_bought : ℕ := 3

/-- The number of packets of chili pepper seeds Harry bought -/
def chili_bought : ℕ := 5

/-- The total amount Harry spent in dollars -/
def total_spent : ℚ := 18

theorem harry_tomato_packets : 
  pumpkin_price * pumpkin_bought + tomato_price * tomato_packets + chili_price * chili_bought = total_spent ∧ 
  tomato_packets = 4 := by sorry

end NUMINAMATH_CALUDE_harry_tomato_packets_l2736_273603


namespace NUMINAMATH_CALUDE_als_original_portion_l2736_273628

theorem als_original_portion (a b c : ℝ) : 
  a + b + c = 1200 →
  0.75 * a + 2 * b + 2 * c = 1800 →
  a = 480 :=
by sorry

end NUMINAMATH_CALUDE_als_original_portion_l2736_273628


namespace NUMINAMATH_CALUDE_solution_set_inequality_l2736_273669

theorem solution_set_inequality (x : ℝ) : 
  (x - 2) * (3 - x) > 0 ↔ x ∈ Set.Ioo 2 3 := by sorry

end NUMINAMATH_CALUDE_solution_set_inequality_l2736_273669


namespace NUMINAMATH_CALUDE_union_A_B_complement_A_union_B_l2736_273604

-- Define the universal set U
def U : Set Nat := {1, 2, 3, 4, 5, 6, 7}

-- Define set A
def A : Set Nat := {1, 3, 5, 7}

-- Define set B
def B : Set Nat := {3, 5}

-- Theorem for A ∪ B
theorem union_A_B : A ∪ B = {1, 3, 5, 7} := by sorry

-- Theorem for (∁ₐA) ∪ B
theorem complement_A_union_B : (U \ A) ∪ B = {2, 3, 4, 5, 6} := by sorry

end NUMINAMATH_CALUDE_union_A_B_complement_A_union_B_l2736_273604


namespace NUMINAMATH_CALUDE_reciprocal_of_mixed_number_l2736_273668

def mixed_number_to_fraction (whole : ℤ) (numerator : ℤ) (denominator : ℤ) : ℚ :=
  (whole * denominator + numerator) / denominator

def reciprocal (x : ℚ) : ℚ := 1 / x

theorem reciprocal_of_mixed_number :
  let original : ℚ := mixed_number_to_fraction (-1) 2 3
  let recip : ℚ := -3 / 5
  (reciprocal original = recip) ∧ (original * recip = 1) := by sorry

end NUMINAMATH_CALUDE_reciprocal_of_mixed_number_l2736_273668


namespace NUMINAMATH_CALUDE_smallest_n_not_divisible_by_10_l2736_273602

theorem smallest_n_not_divisible_by_10 :
  ∃ (n : ℕ), n = 2020 ∧ n > 2016 ∧
  ¬(10 ∣ (1^n + 2^n + 3^n + 4^n)) ∧
  ∀ (m : ℕ), 2016 < m ∧ m < n → (10 ∣ (1^m + 2^m + 3^m + 4^m)) :=
by sorry

end NUMINAMATH_CALUDE_smallest_n_not_divisible_by_10_l2736_273602


namespace NUMINAMATH_CALUDE_hyperbola_eccentricity_l2736_273605

/-- The eccentricity of a hyperbola with equation x²/a² - y²/b² = 1,
    where a > 0, b > 0, and one asymptote forms a 60° angle with the y-axis. -/
theorem hyperbola_eccentricity (a b : ℝ) (ha : a > 0) (hb : b > 0) 
    (h_asymptote : b / a = Real.sqrt 3 / 3) : 
    Real.sqrt (1 + (b / a)^2) = 2 * Real.sqrt 3 / 3 := by
  sorry

end NUMINAMATH_CALUDE_hyperbola_eccentricity_l2736_273605


namespace NUMINAMATH_CALUDE_problem_solution_l2736_273686

theorem problem_solution (x y : ℝ) (h : -x + 2*y = 5) :
  5*(x - 2*y)^2 - 3*(x - 2*y) - 60 = 80 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l2736_273686


namespace NUMINAMATH_CALUDE_line_through_point_not_perpendicular_l2736_273665

/-- A line in the form y = k(x-2) passes through (2,0) and is not perpendicular to the x-axis -/
theorem line_through_point_not_perpendicular (k : ℝ) : 
  ∃ (x y : ℝ), y = k * (x - 2) → 
  (x = 2 ∧ y = 0) ∧ 
  (k ≠ 0 → ∃ (m : ℝ), m ≠ 0 ∧ k = 1 / m) :=
sorry

end NUMINAMATH_CALUDE_line_through_point_not_perpendicular_l2736_273665


namespace NUMINAMATH_CALUDE_quadratic_roots_sum_l2736_273692

theorem quadratic_roots_sum (a b c : ℝ) (x₁ x₂ : ℂ) : 
  (∃ (s t : ℝ), x₁ = s + t * I ∧ t ≠ 0) →  -- x₁ is a complex number
  (a * x₁^2 + b * x₁ + c = 0) →  -- x₁ is a root of the quadratic equation
  (a * x₂^2 + b * x₂ + c = 0) →  -- x₂ is a root of the quadratic equation
  (∃ (r : ℝ), x₁^2 / x₂ = r) →  -- x₁²/x₂ is real
  let S := 1 + x₁/x₂ + (x₁/x₂)^2 + (x₁/x₂)^4 + (x₁/x₂)^8 + (x₁/x₂)^16 + (x₁/x₂)^32
  S = -2 := by
sorry

end NUMINAMATH_CALUDE_quadratic_roots_sum_l2736_273692


namespace NUMINAMATH_CALUDE_elizabeth_pencil_purchase_l2736_273666

def pencil_cost : ℕ := 600
def elizabeth_money : ℕ := 500
def borrowed_money : ℕ := 53

theorem elizabeth_pencil_purchase : 
  pencil_cost - (elizabeth_money + borrowed_money) = 47 := by
  sorry

end NUMINAMATH_CALUDE_elizabeth_pencil_purchase_l2736_273666


namespace NUMINAMATH_CALUDE_geometric_sequence_property_l2736_273631

-- Define a geometric sequence
def is_geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

-- Define the theorem
theorem geometric_sequence_property
  (a : ℕ → ℝ)
  (h_geo : is_geometric_sequence a)
  (h_pos : ∀ n : ℕ, a n > 0)
  (h_prod : a 3 * a 11 = 16) :
  a 7 = 4 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_property_l2736_273631


namespace NUMINAMATH_CALUDE_max_a_value_l2736_273672

/-- Given a quadratic trinomial f(x) = x^2 + ax + b, if for any real x there exists a real y 
    such that f(y) = f(x) + y, then the maximum possible value of a is 1/2. -/
theorem max_a_value (a b : ℝ) : 
  (∀ x : ℝ, ∃ y : ℝ, (y^2 + a*y + b) = (x^2 + a*x + b) + y) → 
  a ≤ (1/2 : ℝ) ∧ ∃ a₀ : ℝ, a₀ ≤ (1/2 : ℝ) ∧ 
    (∀ x : ℝ, ∃ y : ℝ, (y^2 + a₀*y + b) = (x^2 + a₀*x + b) + y) :=
sorry

end NUMINAMATH_CALUDE_max_a_value_l2736_273672


namespace NUMINAMATH_CALUDE_wall_breadth_l2736_273699

/-- Proves that the breadth of a wall with given proportions and volume is 0.4 meters -/
theorem wall_breadth (V h l b : ℝ) (hV : V = 12.8) (hh : h = 5 * b) (hl : l = 8 * h) 
  (hvolume : V = l * b * h) : b = 0.4 := by
  sorry

end NUMINAMATH_CALUDE_wall_breadth_l2736_273699


namespace NUMINAMATH_CALUDE_two_by_six_grid_triangles_l2736_273653

/-- Represents a rectangular grid with diagonal lines --/
structure DiagonalGrid :=
  (rows : ℕ)
  (cols : ℕ)
  (has_center_diagonals : Bool)

/-- Counts the number of triangles in a diagonal grid --/
def count_triangles (grid : DiagonalGrid) : ℕ :=
  sorry

/-- Theorem stating that a 2x6 grid with center diagonals has at least 88 triangles --/
theorem two_by_six_grid_triangles :
  ∀ (grid : DiagonalGrid),
    grid.rows = 2 ∧ 
    grid.cols = 6 ∧ 
    grid.has_center_diagonals = true →
    count_triangles grid ≥ 88 :=
by sorry

end NUMINAMATH_CALUDE_two_by_six_grid_triangles_l2736_273653


namespace NUMINAMATH_CALUDE_angle_relation_in_triangle_l2736_273641

/-- Given a triangle XYZ with an interior point E, where a, b, c, p are the measures of angles
    around E in degrees, and t is the exterior angle at vertex Y, prove that p = 180° - a - b + t. -/
theorem angle_relation_in_triangle (a b c p t : ℝ) : 
  (a + b + c + p = 360) →  -- Sum of angles around interior point E
  (t = 180 - c) →          -- Exterior angle relation
  (p = 180 - a - b + t) :=
by sorry

end NUMINAMATH_CALUDE_angle_relation_in_triangle_l2736_273641


namespace NUMINAMATH_CALUDE_solution_set_inequality_l2736_273660

theorem solution_set_inequality (x : ℝ) :
  (x + 5) * (1 - x) ≥ 8 ↔ -3 ≤ x ∧ x ≤ -1 := by
  sorry

end NUMINAMATH_CALUDE_solution_set_inequality_l2736_273660


namespace NUMINAMATH_CALUDE_probability_of_star_is_one_fifth_l2736_273617

/-- Represents a deck of cards -/
structure Deck :=
  (total_cards : ℕ)
  (num_suits : ℕ)
  (cards_per_suit : ℕ)

/-- Calculates the probability of drawing a specific suit from a deck -/
def probability_of_suit (d : Deck) : ℚ :=
  d.cards_per_suit / d.total_cards

/-- The modified deck of cards as described in the problem -/
def modified_deck : Deck :=
  { total_cards := 65,
    num_suits := 5,
    cards_per_suit := 13 }

theorem probability_of_star_is_one_fifth :
  probability_of_suit modified_deck = 1 / 5 := by
  sorry

end NUMINAMATH_CALUDE_probability_of_star_is_one_fifth_l2736_273617


namespace NUMINAMATH_CALUDE_chord_cosine_l2736_273627

theorem chord_cosine (r : ℝ) (θ φ : ℝ) : 
  r > 0 →
  θ > 0 →
  φ > 0 →
  θ + φ < π →
  8^2 = 2 * r^2 * (1 - Real.cos θ) →
  15^2 = 2 * r^2 * (1 - Real.cos φ) →
  17^2 = 2 * r^2 * (1 - Real.cos (θ + φ)) →
  Real.cos θ = 161 / 225 := by
sorry

end NUMINAMATH_CALUDE_chord_cosine_l2736_273627


namespace NUMINAMATH_CALUDE_third_place_winnings_l2736_273638

theorem third_place_winnings (num_people : ℕ) (contribution : ℝ) (first_place_percentage : ℝ) :
  num_people = 8 →
  contribution = 5 →
  first_place_percentage = 0.8 →
  let total_pot := num_people * contribution
  let first_place_amount := first_place_percentage * total_pot
  let remaining_amount := total_pot - first_place_amount
  remaining_amount / 2 = 4 := by sorry

end NUMINAMATH_CALUDE_third_place_winnings_l2736_273638


namespace NUMINAMATH_CALUDE_workout_days_l2736_273633

/-- Represents the number of squats performed on a given day -/
def squats_on_day (initial_squats : ℕ) (daily_increase : ℕ) (day : ℕ) : ℕ :=
  initial_squats + (day - 1) * daily_increase

/-- Represents the problem of determining the number of consecutive workout days -/
theorem workout_days (initial_squats : ℕ) (daily_increase : ℕ) (target_squats : ℕ) : 
  initial_squats = 30 → 
  daily_increase = 5 → 
  target_squats = 45 → 
  ∃ (n : ℕ), n = 4 ∧ squats_on_day initial_squats daily_increase n = target_squats :=
by
  sorry


end NUMINAMATH_CALUDE_workout_days_l2736_273633


namespace NUMINAMATH_CALUDE_trees_in_row_l2736_273642

/-- Given a plot of trees with the following properties:
  1. Trees are planted in rows of 4
  2. Each tree gives 5 apples
  3. Each apple is sold for $0.5
  4. Total revenue is $30
  Prove that the number of trees in one row is 4. -/
theorem trees_in_row (trees_per_row : ℕ) (apples_per_tree : ℕ) (price_per_apple : ℚ) (total_revenue : ℚ)
  (h1 : trees_per_row = 4)
  (h2 : apples_per_tree = 5)
  (h3 : price_per_apple = 1/2)
  (h4 : total_revenue = 30) :
  trees_per_row = 4 := by sorry

end NUMINAMATH_CALUDE_trees_in_row_l2736_273642


namespace NUMINAMATH_CALUDE_sum_is_non_horizontal_line_l2736_273683

/-- Represents a quadratic function of the form ax^2 + bx + c -/
structure Quadratic where
  a : ℝ
  b : ℝ
  c : ℝ

/-- The original parabola -/
def original_parabola : Quadratic → ℝ → ℝ := λ q x => q.a * x^2 + q.b * x + q.c

/-- Reflection of the parabola about the x-axis -/
def reflected_parabola : Quadratic → ℝ → ℝ := λ q x => -q.a * x^2 - q.b * x - q.c

/-- Horizontal translation of a function -/
def translate (f : ℝ → ℝ) (h : ℝ) : ℝ → ℝ := λ x => f (x - h)

/-- The sum of the translated original and reflected parabolas -/
def sum_of_translated_parabolas (q : Quadratic) : ℝ → ℝ :=
  λ x => translate (original_parabola q) 3 x + translate (reflected_parabola q) (-3) x

/-- Theorem stating that the sum of translated parabolas is a non-horizontal line -/
theorem sum_is_non_horizontal_line (q : Quadratic) (h : q.a ≠ 0) :
  ∃ m k : ℝ, m ≠ 0 ∧ ∀ x, sum_of_translated_parabolas q x = m * x + k :=
sorry

end NUMINAMATH_CALUDE_sum_is_non_horizontal_line_l2736_273683


namespace NUMINAMATH_CALUDE_madhav_rank_from_last_l2736_273650

theorem madhav_rank_from_last (total_students : ℕ) (madhav_rank_start : ℕ) 
  (h1 : total_students = 31) (h2 : madhav_rank_start = 17) : 
  total_students - madhav_rank_start + 1 = 15 := by
  sorry

end NUMINAMATH_CALUDE_madhav_rank_from_last_l2736_273650


namespace NUMINAMATH_CALUDE_basketball_count_l2736_273635

theorem basketball_count :
  ∀ (basketballs volleyballs soccerballs : ℕ),
    basketballs + volleyballs + soccerballs = 100 →
    basketballs = 2 * volleyballs →
    volleyballs = soccerballs + 8 →
    basketballs = 54 :=
by
  sorry

end NUMINAMATH_CALUDE_basketball_count_l2736_273635


namespace NUMINAMATH_CALUDE_valid_rectangle_exists_l2736_273611

/-- Represents a point in a triangular grid -/
structure GridPoint where
  x : ℤ
  y : ℤ

/-- Represents a rectangle in the triangular grid -/
structure Rectangle where
  bottomLeft : GridPoint
  width : ℕ
  height : ℕ

/-- Counts the number of grid points on the boundary of a rectangle -/
def boundaryPoints (rect : Rectangle) : ℕ :=
  2 * (rect.width + rect.height)

/-- Counts the number of grid points in the interior of a rectangle -/
def interiorPoints (rect : Rectangle) : ℕ :=
  rect.width * rect.height + (rect.width - 1) * (rect.height - 1)

/-- Checks if a rectangle satisfies the required conditions -/
def isValidRectangle (rect : Rectangle) : Prop :=
  boundaryPoints rect = interiorPoints rect

/-- Main theorem: There exists a valid rectangle in the triangular grid -/
theorem valid_rectangle_exists : ∃ (rect : Rectangle), isValidRectangle rect :=
  sorry

end NUMINAMATH_CALUDE_valid_rectangle_exists_l2736_273611


namespace NUMINAMATH_CALUDE_total_students_is_28_l2736_273629

/-- The number of students taking the AMC 8 in Mrs. Germain's class -/
def germain_students : ℕ := 11

/-- The number of students taking the AMC 8 in Mr. Newton's class -/
def newton_students : ℕ := 8

/-- The number of students taking the AMC 8 in Mrs. Young's class -/
def young_students : ℕ := 9

/-- The total number of students taking the AMC 8 at Euclid Middle School -/
def total_students : ℕ := germain_students + newton_students + young_students

/-- Theorem stating that the total number of students taking the AMC 8 is 28 -/
theorem total_students_is_28 : total_students = 28 := by sorry

end NUMINAMATH_CALUDE_total_students_is_28_l2736_273629


namespace NUMINAMATH_CALUDE_xenia_june_earnings_l2736_273644

/-- Xenia's earnings during the first two weeks of June -/
def xenia_earnings (hours_week1 hours_week2 : ℕ) (wage_difference : ℚ) : ℚ :=
  let hourly_wage := wage_difference / (hours_week2 - hours_week1 : ℚ)
  hourly_wage * (hours_week1 + hours_week2 : ℚ)

/-- Theorem stating Xenia's earnings during the first two weeks of June -/
theorem xenia_june_earnings :
  xenia_earnings 15 22 (47.60 : ℚ) = (251.60 : ℚ) := by
  sorry

#eval xenia_earnings 15 22 (47.60 : ℚ)

end NUMINAMATH_CALUDE_xenia_june_earnings_l2736_273644


namespace NUMINAMATH_CALUDE_twenty_first_term_equals_203_l2736_273639

/-- An arithmetic sequence is defined by its first term and common difference -/
structure ArithmeticSequence where
  first_term : ℝ
  common_difference : ℝ

/-- The nth term of an arithmetic sequence -/
def nth_term (seq : ArithmeticSequence) (n : ℕ) : ℝ :=
  seq.first_term + (n - 1) * seq.common_difference

theorem twenty_first_term_equals_203 :
  ∃ (seq : ArithmeticSequence),
    seq.first_term = 3 ∧
    nth_term seq 2 = 13 ∧
    nth_term seq 3 = 23 ∧
    nth_term seq 21 = 203 := by
  sorry

end NUMINAMATH_CALUDE_twenty_first_term_equals_203_l2736_273639


namespace NUMINAMATH_CALUDE_equation_is_quadratic_l2736_273684

/-- Definition of a quadratic equation -/
def is_quadratic_equation (f : ℝ → ℝ) : Prop :=
  ∃ a b c : ℝ, a ≠ 0 ∧ ∀ x, f x = a * x^2 + b * x + c

/-- The given equation -/
def f (x : ℝ) : ℝ := 3 - 5 * x^2 - x

theorem equation_is_quadratic : is_quadratic_equation f := by
  sorry

end NUMINAMATH_CALUDE_equation_is_quadratic_l2736_273684


namespace NUMINAMATH_CALUDE_sum_of_coefficients_l2736_273609

theorem sum_of_coefficients (d : ℝ) (h : d ≠ 0) : ∃ a b c : ℝ, 
  (16 * d + 17 + 18 * d^3) + (4 * d + 2) = a * d^3 + b * d + c ∧ a + b + c = 57 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_coefficients_l2736_273609


namespace NUMINAMATH_CALUDE_min_cards_sum_eleven_l2736_273618

theorem min_cards_sum_eleven (n : ℕ) (h : n = 10) : 
  ∃ (k : ℕ), k = 6 ∧ 
  (∀ (S : Finset ℕ), S ⊆ Finset.range n → S.card ≥ k → 
    ∃ (a b : ℕ), a ∈ S ∧ b ∈ S ∧ a ≠ b ∧ a + b = 11) ∧
  (∀ (m : ℕ), m < k → 
    ∃ (T : Finset ℕ), T ⊆ Finset.range n ∧ T.card = m ∧
      ∀ (a b : ℕ), a ∈ T → b ∈ T → a ≠ b → a + b ≠ 11) :=
by sorry

end NUMINAMATH_CALUDE_min_cards_sum_eleven_l2736_273618


namespace NUMINAMATH_CALUDE_sqrt_b_minus_a_l2736_273670

theorem sqrt_b_minus_a (a b : ℝ) 
  (h1 : (2 * a - 1).sqrt = 3)
  (h2 : (3 * a + b - 1)^(1/3) = 3) :
  (b - a).sqrt = 2 * Real.sqrt 2 ∨ (b - a).sqrt = -2 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_b_minus_a_l2736_273670


namespace NUMINAMATH_CALUDE_golden_section_length_l2736_273663

/-- Definition of a golden section point -/
def is_golden_section (A B C : ℝ) : Prop :=
  (B - A) / (C - A) = (C - A) / (A - C)

/-- Theorem: Length of AC when C is a golden section point of AB -/
theorem golden_section_length (A B C : ℝ) :
  B - A = 20 →
  is_golden_section A B C →
  (C - A = 10 * Real.sqrt 5 - 10) ∨ (C - A = 30 - 10 * Real.sqrt 5) :=
by sorry

end NUMINAMATH_CALUDE_golden_section_length_l2736_273663


namespace NUMINAMATH_CALUDE_A_intersect_B_l2736_273673

def A : Set ℤ := {1, 2, 3, 4}
def B : Set ℤ := {x | 2 ≤ x ∧ x ≤ 3}

theorem A_intersect_B : A ∩ B = {2, 3} := by sorry

end NUMINAMATH_CALUDE_A_intersect_B_l2736_273673


namespace NUMINAMATH_CALUDE_rational_equation_solution_l2736_273664

theorem rational_equation_solution : 
  ∃! y : ℚ, (y^2 - 12*y + 32) / (y - 2) + (3*y^2 + 11*y - 14) / (3*y - 1) = -5 ∧ y = -17/6 := by
  sorry

end NUMINAMATH_CALUDE_rational_equation_solution_l2736_273664


namespace NUMINAMATH_CALUDE_smallest_bob_number_l2736_273621

def alice_number : ℕ := 30

theorem smallest_bob_number (bob_number : ℕ) 
  (h1 : ∀ p : ℕ, Nat.Prime p → p ∣ alice_number → p ∣ bob_number) 
  (h2 : ∀ n : ℕ, n ≥ bob_number → 
    (∀ p : ℕ, Nat.Prime p → p ∣ alice_number → p ∣ n)) : 
  bob_number = 30 := by
sorry

end NUMINAMATH_CALUDE_smallest_bob_number_l2736_273621


namespace NUMINAMATH_CALUDE_f_properties_l2736_273656

noncomputable def f (x : ℝ) : ℝ := Real.cos x ^ 2 + Real.sqrt 3 * Real.sin x * Real.cos x + 1

def is_periodic (f : ℝ → ℝ) (T : ℝ) : Prop :=
  ∀ x, f (x + T) = f x

def is_smallest_positive_period (f : ℝ → ℝ) (T : ℝ) : Prop :=
  T > 0 ∧ is_periodic f T ∧ ∀ T' > 0, is_periodic f T' → T ≤ T'

theorem f_properties :
  (is_smallest_positive_period f π) ∧
  (∀ x, 1/2 ≤ f x ∧ f x ≤ 5/2) ∧
  (∀ k : ℤ, StrictMonoOn f (Set.Icc (-π/3 + k*π) (π/6 + k*π))) :=
sorry

end NUMINAMATH_CALUDE_f_properties_l2736_273656


namespace NUMINAMATH_CALUDE_distance_to_intersecting_line_l2736_273640

/-- Ellipse G with equation x^2/8 + y^2/4 = 1 -/
def ellipse_G (x y : ℝ) : Prop := x^2/8 + y^2/4 = 1

/-- Left focus F1(-2,0) -/
def F1 : ℝ × ℝ := (-2, 0)

/-- Right focus F2(2,0) -/
def F2 : ℝ × ℝ := (2, 0)

/-- Line l intersects ellipse G at points A and B -/
def intersects_ellipse (l : Set (ℝ × ℝ)) : Prop :=
  ∃ A B : ℝ × ℝ, A ≠ B ∧ A ∈ l ∧ B ∈ l ∧ ellipse_G A.1 A.2 ∧ ellipse_G B.1 B.2

/-- OA is perpendicular to OB -/
def perpendicular (A B : ℝ × ℝ) : Prop :=
  A.1 * B.1 + A.2 * B.2 = 0

/-- Distance from a point to a line -/
def distance_point_to_line (p : ℝ × ℝ) (l : Set (ℝ × ℝ)) : ℝ := sorry

/-- Theorem: For ellipse G, if line l intersects G at A and B with OA ⊥ OB,
    then the distance from O to l is 2√6/3 -/
theorem distance_to_intersecting_line :
  ∀ l : Set (ℝ × ℝ),
  intersects_ellipse l →
  (∃ A B : ℝ × ℝ, A ≠ B ∧ A ∈ l ∧ B ∈ l ∧ ellipse_G A.1 A.2 ∧ ellipse_G B.1 B.2 ∧ perpendicular A B) →
  distance_point_to_line (0, 0) l = 2 * Real.sqrt 6 / 3 :=
sorry

end NUMINAMATH_CALUDE_distance_to_intersecting_line_l2736_273640


namespace NUMINAMATH_CALUDE_negation_of_all_dogs_are_playful_l2736_273674

-- Define the universe of discourse
variable (U : Type)

-- Define predicates for being a dog and being playful
variable (dog : U → Prop)
variable (playful : U → Prop)

-- State the theorem
theorem negation_of_all_dogs_are_playful :
  (¬∀ x, dog x → playful x) ↔ (∃ x, dog x ∧ ¬playful x) :=
sorry

end NUMINAMATH_CALUDE_negation_of_all_dogs_are_playful_l2736_273674


namespace NUMINAMATH_CALUDE_age_ratio_proof_l2736_273612

theorem age_ratio_proof (my_current_age brother_current_age : ℕ) 
  (h1 : my_current_age = 20)
  (h2 : my_current_age + 10 + (brother_current_age + 10) = 45) :
  (my_current_age + 10) / (brother_current_age + 10) = 2 := by
  sorry

end NUMINAMATH_CALUDE_age_ratio_proof_l2736_273612


namespace NUMINAMATH_CALUDE_cube_difference_of_squares_l2736_273636

theorem cube_difference_of_squares (a : ℕ+) :
  ∃ (x y : ℤ), x^2 - y^2 = (a : ℤ)^3 := by
  sorry

end NUMINAMATH_CALUDE_cube_difference_of_squares_l2736_273636


namespace NUMINAMATH_CALUDE_sum_integers_from_neg50_to_75_l2736_273630

def sum_integers (a b : Int) : Int :=
  (b - a + 1) * (a + b) / 2

theorem sum_integers_from_neg50_to_75 :
  sum_integers (-50) 75 = 1575 := by
  sorry

end NUMINAMATH_CALUDE_sum_integers_from_neg50_to_75_l2736_273630


namespace NUMINAMATH_CALUDE_cosine_sine_sum_zero_l2736_273626

theorem cosine_sine_sum_zero (x : ℝ) (h : Real.cos (π / 6 - x) = -Real.sqrt 3 / 3) :
  Real.cos (5 * π / 6 + x) + Real.sin (2 * π / 3 - x) = 0 := by
  sorry

end NUMINAMATH_CALUDE_cosine_sine_sum_zero_l2736_273626


namespace NUMINAMATH_CALUDE_real_z_implies_m_eq_3_modulus_z_eq_sqrt_13_when_m_eq_1_l2736_273623

-- Define the complex number z as a function of m
def z (m : ℝ) : ℂ := Complex.mk (m + 2) (m - 3)

-- Theorem 1: If z is a real number, then m = 3
theorem real_z_implies_m_eq_3 (m : ℝ) : z m = Complex.mk (z m).re 0 → m = 3 := by
  sorry

-- Theorem 2: When m = 1, the modulus of z is √13
theorem modulus_z_eq_sqrt_13_when_m_eq_1 : Complex.abs (z 1) = Real.sqrt 13 := by
  sorry

end NUMINAMATH_CALUDE_real_z_implies_m_eq_3_modulus_z_eq_sqrt_13_when_m_eq_1_l2736_273623


namespace NUMINAMATH_CALUDE_divisible_by_27_l2736_273637

theorem divisible_by_27 (n : ℕ) : ∃ k : ℤ, 2^(2*n - 1) - 9*n^2 + 21*n - 14 = 27*k := by
  sorry

end NUMINAMATH_CALUDE_divisible_by_27_l2736_273637


namespace NUMINAMATH_CALUDE_discretionary_income_ratio_l2736_273679

/-- Jill's financial situation --/
def jill_finances (net_salary : ℚ) (discretionary_income : ℚ) : Prop :=
  net_salary = 3600 ∧
  0.30 * discretionary_income + 0.20 * discretionary_income + 0.35 * discretionary_income + 108 = discretionary_income ∧
  discretionary_income > 0

/-- The ratio of discretionary income to net salary is 1:5 --/
theorem discretionary_income_ratio
  (net_salary discretionary_income : ℚ)
  (h : jill_finances net_salary discretionary_income) :
  discretionary_income / net_salary = 1 / 5 :=
by sorry

end NUMINAMATH_CALUDE_discretionary_income_ratio_l2736_273679


namespace NUMINAMATH_CALUDE_first_child_2019th_number_l2736_273616

/-- Represents the counting game with three children -/
def CountingGame :=
  { n : ℕ | n > 0 ∧ n ≤ 10000 }

/-- The sequence of numbers said by the first child -/
def first_child_sequence (n : ℕ) : ℕ :=
  3 * n * n - 2 * n + 1

/-- The number of complete cycles before the 2019th number -/
def complete_cycles : ℕ := 36

/-- The position of the 2019th number within its cycle -/
def position_in_cycle : ℕ := 93

/-- The 2019th number said by the first child -/
theorem first_child_2019th_number :
  ∃ (game : CountingGame),
    first_child_sequence complete_cycles +
    position_in_cycle = 5979 :=
sorry

end NUMINAMATH_CALUDE_first_child_2019th_number_l2736_273616


namespace NUMINAMATH_CALUDE_f_zero_at_three_l2736_273651

/-- The function f(x) = 3x^3 + 2x^2 - 5x + s -/
def f (s : ℝ) (x : ℝ) : ℝ := 3 * x^3 + 2 * x^2 - 5 * x + s

/-- Theorem: f(3) = 0 if and only if s = -84 -/
theorem f_zero_at_three (s : ℝ) : f s 3 = 0 ↔ s = -84 := by
  sorry

end NUMINAMATH_CALUDE_f_zero_at_three_l2736_273651


namespace NUMINAMATH_CALUDE_units_digit_27_times_64_l2736_273652

/-- The units digit of a natural number -/
def units_digit (n : ℕ) : ℕ := n % 10

/-- The property that the units digit of a product depends only on the units digits of its factors -/
axiom units_digit_product (a b : ℕ) : 
  units_digit (a * b) = units_digit (units_digit a * units_digit b)

/-- The theorem stating that the units digit of 27 · 64 is 8 -/
theorem units_digit_27_times_64 : units_digit (27 * 64) = 8 := by
  sorry

end NUMINAMATH_CALUDE_units_digit_27_times_64_l2736_273652


namespace NUMINAMATH_CALUDE_joe_cars_count_l2736_273678

theorem joe_cars_count (initial_cars additional_cars : ℕ) : 
  initial_cars = 50 → additional_cars = 12 → initial_cars + additional_cars = 62 := by
  sorry

end NUMINAMATH_CALUDE_joe_cars_count_l2736_273678


namespace NUMINAMATH_CALUDE_line_segment_lattice_points_l2736_273614

/-- The number of lattice points on a line segment with given endpoints -/
def latticePointCount (x1 y1 x2 y2 : Int) : Nat :=
  sorry

theorem line_segment_lattice_points :
  latticePointCount 5 10 68 178 = 22 := by sorry

end NUMINAMATH_CALUDE_line_segment_lattice_points_l2736_273614


namespace NUMINAMATH_CALUDE_triangle_area_triangle_area_is_18_l2736_273608

/-- The area of the triangle formed by the lines y = 8, y = 2 + 2x, and y = 2 - 2x -/
theorem triangle_area : ℝ → Prop :=
  fun area =>
    ∃ (A B C : ℝ × ℝ),
      (A.2 = 8 ∧ B.2 = 2 + 2 * B.1 ∧ C.2 = 2 - 2 * C.1) ∧
      (A.2 = 8 ∧ A.2 = 2 + 2 * A.1) ∧
      (B.2 = 8 ∧ B.2 = 2 - 2 * B.1) ∧
      (C.2 = 2 + 2 * C.1 ∧ C.2 = 2 - 2 * C.1) ∧
      area = 18

/-- The area of the triangle is 18 -/
theorem triangle_area_is_18 : triangle_area 18 := by
  sorry

end NUMINAMATH_CALUDE_triangle_area_triangle_area_is_18_l2736_273608


namespace NUMINAMATH_CALUDE_remaining_money_l2736_273681

/-- Calculates the remaining money after spending on sweets and giving to friends -/
theorem remaining_money 
  (initial_amount : ℚ)
  (spent_on_sweets : ℚ)
  (given_to_each_friend : ℚ)
  (number_of_friends : ℕ)
  (h1 : initial_amount = 7.1)
  (h2 : spent_on_sweets = 1.05)
  (h3 : given_to_each_friend = 1)
  (h4 : number_of_friends = 2) :
  initial_amount - spent_on_sweets - (given_to_each_friend * number_of_friends) = 4.05 := by
  sorry

#eval (7.1 : ℚ) - 1.05 - (1 * 2)  -- This should evaluate to 4.05

end NUMINAMATH_CALUDE_remaining_money_l2736_273681


namespace NUMINAMATH_CALUDE_genevieve_coffee_consumption_l2736_273662

-- Define the conversion rate from gallons to pints
def gallons_to_pints (gallons : Real) : Real := gallons * 8

-- Define the total amount of coffee in gallons
def total_coffee_gallons : Real := 4.5

-- Define the number of thermoses
def num_thermoses : Nat := 18

-- Define the number of thermoses Genevieve drank
def genevieve_thermoses : Nat := 3

-- Theorem statement
theorem genevieve_coffee_consumption :
  let total_pints := gallons_to_pints total_coffee_gallons
  let pints_per_thermos := total_pints / num_thermoses
  pints_per_thermos * genevieve_thermoses = 6 := by
  sorry


end NUMINAMATH_CALUDE_genevieve_coffee_consumption_l2736_273662


namespace NUMINAMATH_CALUDE_dice_events_properties_l2736_273661

-- Define the sample space
def Ω : Type := Fin 6 × Fin 6

-- Define the probability measure
def P : Set Ω → ℝ := sorry

-- Define events A, B, C, and D
def A : Set Ω := {ω | ω.1 = 2}
def B : Set Ω := {ω | ω.2 < 5}
def C : Set Ω := {ω | (ω.1.val + ω.2.val) % 2 = 1}
def D : Set Ω := {ω | ω.1.val + ω.2.val = 9}

-- State the theorem
theorem dice_events_properties :
  (¬(A ∩ B = ∅) ∧ P (A ∩ B) = P A * P B) ∧
  (A ∩ D = ∅ ∧ P (A ∩ D) ≠ P A * P D) ∧
  (¬(A ∩ C = ∅) ∧ P (A ∩ C) = P A * P C) :=
sorry

end NUMINAMATH_CALUDE_dice_events_properties_l2736_273661


namespace NUMINAMATH_CALUDE_translate_AB_to_origin_l2736_273690

/-- Given two points A and B in a 2D Cartesian coordinate system, 
    this function returns the coordinates of B after translating 
    the line segment AB so that A coincides with the origin. -/
def translate_to_origin (A B : ℝ × ℝ) : ℝ × ℝ :=
  (B.1 - A.1, B.2 - A.2)

/-- Theorem stating that translating the line segment AB 
    from A(-4, 0) to B(0, 2) so that A coincides with the origin 
    results in B having coordinates (4, 2). -/
theorem translate_AB_to_origin : 
  let A : ℝ × ℝ := (-4, 0)
  let B : ℝ × ℝ := (0, 2)
  translate_to_origin A B = (4, 2) := by
  sorry


end NUMINAMATH_CALUDE_translate_AB_to_origin_l2736_273690


namespace NUMINAMATH_CALUDE_min_value_sum_product_l2736_273620

theorem min_value_sum_product (a b c d : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hd : 0 < d) :
  (a + b + c + d) * (1 / (a + b) + 1 / (a + c) + 1 / (b + d) + 1 / (c + d)) ≥ 8 := by
  sorry

end NUMINAMATH_CALUDE_min_value_sum_product_l2736_273620


namespace NUMINAMATH_CALUDE_different_suit_card_selection_l2736_273675

/-- The number of cards in a standard deck -/
def standard_deck_size : ℕ := 52

/-- The number of suits in a standard deck -/
def num_suits : ℕ := 4

/-- The number of cards per suit in a standard deck -/
def cards_per_suit : ℕ := 13

/-- The number of cards to be chosen -/
def cards_to_choose : ℕ := 4

/-- Theorem: The number of ways to choose 4 cards from a standard deck of 52 cards,
    where all four cards must be of different suits, is equal to 28561. -/
theorem different_suit_card_selection :
  (cards_per_suit ^ cards_to_choose : ℕ) = 28561 := by
  sorry

end NUMINAMATH_CALUDE_different_suit_card_selection_l2736_273675


namespace NUMINAMATH_CALUDE_perpendicular_lines_b_value_l2736_273625

/-- 
If two lines given by the equations 4y + 3x + 6 = 0 and 6y + bx + 5 = 0 are perpendicular,
then b = -8.
-/
theorem perpendicular_lines_b_value (b : ℝ) : 
  (∀ x y, 4 * y + 3 * x + 6 = 0 ↔ y = -3/4 * x - 3/2) →
  (∀ x y, 6 * y + b * x + 5 = 0 ↔ y = -b/6 * x - 5/6) →
  ((-3/4) * (-b/6) = -1) →
  b = -8 := by sorry

end NUMINAMATH_CALUDE_perpendicular_lines_b_value_l2736_273625


namespace NUMINAMATH_CALUDE_train_journey_time_l2736_273655

theorem train_journey_time 
  (D : ℝ) -- Distance in km
  (T : ℝ) -- Original time in hours
  (h1 : D = 48 * T) -- Distance equation for original journey
  (h2 : D = 60 * (40 / 60)) -- Distance equation for faster journey
  : T * 60 = 50 := by
  sorry

end NUMINAMATH_CALUDE_train_journey_time_l2736_273655


namespace NUMINAMATH_CALUDE_store_discount_l2736_273685

theorem store_discount (initial_discount : ℝ) (additional_discount : ℝ) (claimed_discount : ℝ) : 
  initial_discount = 0.40 →
  additional_discount = 0.10 →
  claimed_discount = 0.55 →
  let price_after_first_discount := 1 - initial_discount
  let price_after_second_discount := price_after_first_discount * (1 - additional_discount)
  let actual_discount := 1 - price_after_second_discount
  actual_discount = 0.46 ∧ claimed_discount - actual_discount = 0.09 := by
  sorry

end NUMINAMATH_CALUDE_store_discount_l2736_273685


namespace NUMINAMATH_CALUDE_fraction_sum_l2736_273657

theorem fraction_sum (a b : ℚ) (h : a / b = 3 / 2) : (a + b) / b = 5 / 2 := by
  sorry

end NUMINAMATH_CALUDE_fraction_sum_l2736_273657


namespace NUMINAMATH_CALUDE_lcm_primes_sum_l2736_273647

theorem lcm_primes_sum (x y : ℕ) : 
  Nat.Prime x → Nat.Prime y → x > y → Nat.lcm x y = 10 → 2 * x + y = 12 := by
  sorry

end NUMINAMATH_CALUDE_lcm_primes_sum_l2736_273647


namespace NUMINAMATH_CALUDE_polynomial_sum_of_squares_l2736_273613

/-- A polynomial with real coefficients that is non-negative for all real inputs
    can be expressed as the sum of squares of two polynomials. -/
theorem polynomial_sum_of_squares
  (P : Polynomial ℝ)
  (h : ∀ x : ℝ, 0 ≤ P.eval x) :
  ∃ Q R : Polynomial ℝ, P = Q^2 + R^2 :=
sorry

end NUMINAMATH_CALUDE_polynomial_sum_of_squares_l2736_273613


namespace NUMINAMATH_CALUDE_balloons_rearrangements_l2736_273659

def word : String := "BALLOONS"

def is_vowel (c : Char) : Bool :=
  c ∈ ['A', 'E', 'I', 'O', 'U']

def vowels : List Char :=
  word.toList.filter is_vowel

def consonants : List Char :=
  word.toList.filter (fun c => ¬(is_vowel c))

theorem balloons_rearrangements :
  (vowels.length.factorial / (vowels.countP (· = 'O')).factorial) *
  (consonants.length.factorial / (consonants.countP (· = 'L')).factorial) = 180 := by
  sorry

end NUMINAMATH_CALUDE_balloons_rearrangements_l2736_273659


namespace NUMINAMATH_CALUDE_sqrt_1_minus_x_real_l2736_273676

theorem sqrt_1_minus_x_real (x : ℝ) : (∃ y : ℝ, y ^ 2 = 1 - x) ↔ x ≤ 1 := by sorry

end NUMINAMATH_CALUDE_sqrt_1_minus_x_real_l2736_273676


namespace NUMINAMATH_CALUDE_car_average_speed_l2736_273677

/-- The average speed of a car given its speeds in two consecutive hours -/
theorem car_average_speed (speed1 speed2 : ℝ) (h1 : speed1 = 145) (h2 : speed2 = 60) :
  (speed1 + speed2) / 2 = 102.5 := by
  sorry

end NUMINAMATH_CALUDE_car_average_speed_l2736_273677


namespace NUMINAMATH_CALUDE_max_product_vertical_multiplication_l2736_273671

theorem max_product_vertical_multiplication :
  ∀ a b : ℕ,
  50 ≤ a ∧ a < 100 →
  100 ≤ b ∧ b < 1000 →
  ∃ c d e f g : ℕ,
  a * b = 10000 * c + 1000 * d + 100 * e + 10 * f + g ∧
  c = 2 ∧ d = 0 ∧ e = 1 ∧ f = 5 →
  a * b ≤ 19864 :=
by sorry

end NUMINAMATH_CALUDE_max_product_vertical_multiplication_l2736_273671


namespace NUMINAMATH_CALUDE_gaussland_olympics_l2736_273694

theorem gaussland_olympics (total_students : ℕ) (events_per_student : ℕ) (students_per_event : ℕ) (total_coaches : ℕ) 
  (h1 : total_students = 480)
  (h2 : events_per_student = 4)
  (h3 : students_per_event = 20)
  (h4 : total_coaches = 16)
  : (total_students * events_per_student) / (students_per_event * total_coaches) = 6 := by
  sorry

#check gaussland_olympics

end NUMINAMATH_CALUDE_gaussland_olympics_l2736_273694


namespace NUMINAMATH_CALUDE_combine_like_terms_l2736_273697

theorem combine_like_terms (a : ℝ) : 2 * a - 5 * a = -3 * a := by
  sorry

end NUMINAMATH_CALUDE_combine_like_terms_l2736_273697


namespace NUMINAMATH_CALUDE_regression_slope_effect_l2736_273645

/-- Represents a simple linear regression model -/
structure LinearRegression where
  intercept : ℝ
  slope : ℝ

/-- The predicted value of y given x in a linear regression model -/
def predict (model : LinearRegression) (x : ℝ) : ℝ :=
  model.intercept + model.slope * x

/-- The change in y when x increases by one unit -/
def change_in_y (model : LinearRegression) : ℝ :=
  predict model 1 - predict model 0

theorem regression_slope_effect (model : LinearRegression) 
  (h : model = {intercept := 3, slope := -5}) : 
  change_in_y model = -5 := by
  sorry

end NUMINAMATH_CALUDE_regression_slope_effect_l2736_273645


namespace NUMINAMATH_CALUDE_sum_of_squares_of_roots_l2736_273615

theorem sum_of_squares_of_roots (a b : ℝ) : 
  a^2 - 15*a + 6 = 0 → 
  b^2 - 15*b + 6 = 0 → 
  a^2 + b^2 = 213 := by
sorry

end NUMINAMATH_CALUDE_sum_of_squares_of_roots_l2736_273615


namespace NUMINAMATH_CALUDE_smallest_cut_prevents_triangle_smallest_cut_is_minimal_l2736_273658

/-- The smallest positive integer that, when subtracted from the original lengths,
    prevents the formation of a triangle. -/
def smallest_cut : ℕ := 2

/-- Original lengths of the sticks -/
def original_lengths : List ℕ := [9, 12, 20]

/-- Check if three lengths can form a triangle -/
def can_form_triangle (a b c : ℕ) : Prop :=
  a + b > c ∧ b + c > a ∧ c + a > b

/-- The remaining lengths after cutting -/
def remaining_lengths (x : ℕ) : List ℕ :=
  original_lengths.map (λ l => l - x)

theorem smallest_cut_prevents_triangle :
  ∀ x : ℕ, x < smallest_cut →
    ∃ a b c, a::b::c::[] = remaining_lengths x ∧ can_form_triangle a b c :=
by sorry

theorem smallest_cut_is_minimal :
  ¬∃ a b c, a::b::c::[] = remaining_lengths smallest_cut ∧ can_form_triangle a b c :=
by sorry

end NUMINAMATH_CALUDE_smallest_cut_prevents_triangle_smallest_cut_is_minimal_l2736_273658


namespace NUMINAMATH_CALUDE_carol_birthday_invitations_l2736_273634

/-- The number of friends Carol wants to invite -/
def num_friends : ℕ := sorry

/-- The number of invitations in each package -/
def invitations_per_package : ℕ := 3

/-- The number of packages Carol bought -/
def packages_bought : ℕ := 2

/-- The number of extra invitations Carol needs to buy -/
def extra_invitations : ℕ := 3

/-- Theorem stating that the number of friends Carol wants to invite
    is equal to the sum of invitations in bought packs and extra invitations -/
theorem carol_birthday_invitations :
  num_friends = packages_bought * invitations_per_package + extra_invitations := by
  sorry

end NUMINAMATH_CALUDE_carol_birthday_invitations_l2736_273634


namespace NUMINAMATH_CALUDE_real_number_line_bijection_sqrt_six_representation_l2736_273667

-- Define the number line as a type isomorphic to ℝ
def NumberLine : Type := ℝ

-- Statement 1: There exists a bijective function between real numbers and points on the number line
theorem real_number_line_bijection : ∃ f : ℝ → NumberLine, Function.Bijective f :=
sorry

-- Statement 2: The arithmetic square root of 6 is represented by √6
theorem sqrt_six_representation : Real.sqrt 6 = (6 : ℝ).sqrt :=
sorry

end NUMINAMATH_CALUDE_real_number_line_bijection_sqrt_six_representation_l2736_273667


namespace NUMINAMATH_CALUDE_certain_number_proof_l2736_273693

theorem certain_number_proof (x : ℝ) : 
  0.8 * 170 - 0.35 * x = 31 → x = 300 := by
  sorry

end NUMINAMATH_CALUDE_certain_number_proof_l2736_273693


namespace NUMINAMATH_CALUDE_solve_mushroom_problem_l2736_273622

def mushroom_problem (pieces_per_mushroom : ℕ) (total_mushrooms : ℕ) 
  (kenny_pieces : ℕ) (remaining_pieces : ℕ) : Prop :=
  let total_pieces := pieces_per_mushroom * total_mushrooms
  let karla_pieces := total_pieces - (kenny_pieces + remaining_pieces)
  karla_pieces = 42

theorem solve_mushroom_problem :
  mushroom_problem 4 22 38 8 := by sorry

end NUMINAMATH_CALUDE_solve_mushroom_problem_l2736_273622


namespace NUMINAMATH_CALUDE_max_area_quadrilateral_ellipse_l2736_273696

/-- Given an ellipse with equation x²/a² + y²/b² = 1, where a > 0 and b > 0,
    the maximum area of quadrilateral OAPB is √2/2 * a * b,
    where A is the point on the positive x-axis,
    B is the point on the positive y-axis,
    and P is any point on the ellipse within the first quadrant. -/
theorem max_area_quadrilateral_ellipse (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  let ellipse := {P : ℝ × ℝ | (P.1^2 / a^2) + (P.2^2 / b^2) = 1}
  let A := (a, 0)
  let B := (0, b)
  let valid_P := {P ∈ ellipse | P.1 ≥ 0 ∧ P.2 ≥ 0}
  let area (P : ℝ × ℝ) := (P.1 * P.2 / 2) + ((a - P.1) * b + (0 - P.2) * a) / 2
  (⨆ P ∈ valid_P, area P) = Real.sqrt 2 / 2 * a * b := by
  sorry

end NUMINAMATH_CALUDE_max_area_quadrilateral_ellipse_l2736_273696


namespace NUMINAMATH_CALUDE_square_inscribed_problem_l2736_273691

theorem square_inscribed_problem (inner_perimeter outer_perimeter : ℝ) 
  (h1 : inner_perimeter = 32)
  (h2 : outer_perimeter = 40)
  (h3 : inner_perimeter > 0)
  (h4 : outer_perimeter > 0) :
  let inner_side := inner_perimeter / 4
  let outer_side := outer_perimeter / 4
  let third_side := 2 * inner_side
  (∃ (greatest_distance : ℝ), 
    greatest_distance = Real.sqrt 2 ∧ 
    greatest_distance = (outer_side * Real.sqrt 2 - inner_side * Real.sqrt 2) / 2) ∧
  third_side ^ 2 = 256 := by
sorry

end NUMINAMATH_CALUDE_square_inscribed_problem_l2736_273691


namespace NUMINAMATH_CALUDE_rectangle_length_fraction_of_circle_radius_l2736_273687

theorem rectangle_length_fraction_of_circle_radius 
  (square_area : ℝ) 
  (rectangle_area : ℝ) 
  (rectangle_breadth : ℝ) 
  (h1 : square_area = 3025) 
  (h2 : rectangle_area = 220) 
  (h3 : rectangle_breadth = 10) : 
  (rectangle_area / rectangle_breadth) / Real.sqrt square_area = 2 / 5 := by
sorry

end NUMINAMATH_CALUDE_rectangle_length_fraction_of_circle_radius_l2736_273687


namespace NUMINAMATH_CALUDE_arithmetic_calculation_l2736_273695

theorem arithmetic_calculation : 3 * 11 + 3 * 12 + 3 * 15 + 11 = 125 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_calculation_l2736_273695


namespace NUMINAMATH_CALUDE_sam_speed_l2736_273600

/-- Represents a point on the route --/
structure Point where
  position : ℝ

/-- Represents a person traveling on the route --/
structure Traveler where
  start : Point
  speed : ℝ

/-- The scenario of Sam and Nik's travel --/
structure TravelScenario where
  sam : Traveler
  nik : Traveler
  meetingPoint : Point
  totalTime : ℝ

/-- The given travel scenario --/
def givenScenario : TravelScenario where
  sam := { start := { position := 0 }, speed := 0 }  -- speed will be calculated
  nik := { start := { position := 1000 }, speed := 0 }  -- speed is not needed for the problem
  meetingPoint := { position := 600 }
  totalTime := 20

theorem sam_speed (scenario : TravelScenario) :
  scenario.sam.start.position = 0 ∧
  scenario.nik.start.position = 1000 ∧
  scenario.meetingPoint.position = 600 ∧
  scenario.totalTime = 20 →
  scenario.sam.speed = 50 := by
  sorry

end NUMINAMATH_CALUDE_sam_speed_l2736_273600


namespace NUMINAMATH_CALUDE_odd_2n_plus_1_l2736_273632

theorem odd_2n_plus_1 (n : ℤ) : ¬ (∃ k : ℤ, 2 * n + 1 = 2 * k) := by
  sorry

end NUMINAMATH_CALUDE_odd_2n_plus_1_l2736_273632


namespace NUMINAMATH_CALUDE_half_plus_five_equals_fifteen_l2736_273601

theorem half_plus_five_equals_fifteen (n : ℕ) (value : ℕ) : n = 20 → n / 2 + 5 = value → value = 15 := by
  sorry

end NUMINAMATH_CALUDE_half_plus_five_equals_fifteen_l2736_273601


namespace NUMINAMATH_CALUDE_min_tablets_extracted_l2736_273648

/-- The least number of tablets to extract to ensure at least two of each kind -/
def leastTablets (tabletA : ℕ) (tabletB : ℕ) : ℕ :=
  max (tabletB + 2) (tabletA + 2)

theorem min_tablets_extracted (tabletA tabletB : ℕ) 
  (hA : tabletA = 10) (hB : tabletB = 16) :
  leastTablets tabletA tabletB = 18 := by
  sorry

end NUMINAMATH_CALUDE_min_tablets_extracted_l2736_273648


namespace NUMINAMATH_CALUDE_binomial_distribution_parameters_l2736_273680

variable (ξ : ℕ → ℝ)
variable (n : ℕ)
variable (p : ℝ)

-- ξ follows a binomial distribution B(n, p)
def is_binomial (ξ : ℕ → ℝ) (n : ℕ) (p : ℝ) : Prop := sorry

-- Expected value of ξ
def expectation (ξ : ℕ → ℝ) : ℝ := sorry

-- Variance of ξ
def variance (ξ : ℕ → ℝ) : ℝ := sorry

theorem binomial_distribution_parameters 
  (h1 : is_binomial ξ n p)
  (h2 : expectation ξ = 5/3)
  (h3 : variance ξ = 10/9) :
  n = 5 ∧ p = 1/3 := by sorry

end NUMINAMATH_CALUDE_binomial_distribution_parameters_l2736_273680


namespace NUMINAMATH_CALUDE_imaginary_part_product_l2736_273619

theorem imaginary_part_product : Complex.im ((2 - Complex.I) * (1 - 2 * Complex.I)) = -5 := by sorry

end NUMINAMATH_CALUDE_imaginary_part_product_l2736_273619
