import Mathlib

namespace NUMINAMATH_CALUDE_min_max_sum_sqrt_l1727_172774

theorem min_max_sum_sqrt (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) 
  (h_sum : a^2 + b^2 + c^2 = 6) :
  2 + Real.sqrt 2 ≤ Real.sqrt (4 - a^2) + Real.sqrt (4 - b^2) + Real.sqrt (4 - c^2) ∧
  Real.sqrt (4 - a^2) + Real.sqrt (4 - b^2) + Real.sqrt (4 - c^2) ≤ 3 * Real.sqrt 2 :=
by sorry

end NUMINAMATH_CALUDE_min_max_sum_sqrt_l1727_172774


namespace NUMINAMATH_CALUDE_blackbirds_per_tree_l1727_172795

theorem blackbirds_per_tree (num_trees : ℕ) (num_magpies : ℕ) (total_birds : ℕ) 
  (h1 : num_trees = 7)
  (h2 : num_magpies = 13)
  (h3 : total_birds = 34)
  (h4 : ∃ (blackbirds_per_tree : ℕ), num_trees * blackbirds_per_tree + num_magpies = total_birds) :
  ∃ (blackbirds_per_tree : ℕ), blackbirds_per_tree = 3 ∧ 
    num_trees * blackbirds_per_tree + num_magpies = total_birds :=
by
  sorry

end NUMINAMATH_CALUDE_blackbirds_per_tree_l1727_172795


namespace NUMINAMATH_CALUDE_clothing_price_problem_l1727_172736

theorem clothing_price_problem (total_spent : ℕ) (num_pieces : ℕ) (tax_rate : ℚ)
  (untaxed_piece1 : ℕ) (untaxed_piece2 : ℕ) (h_total : total_spent = 610)
  (h_num : num_pieces = 7) (h_tax : tax_rate = 1/10) (h_untaxed1 : untaxed_piece1 = 49)
  (h_untaxed2 : untaxed_piece2 = 81) :
  ∃ (price : ℕ), price * 5 = (total_spent - untaxed_piece1 - untaxed_piece2) * 10 / 11 ∧
  price % 5 = 0 ∧ price = 87 := by
sorry

end NUMINAMATH_CALUDE_clothing_price_problem_l1727_172736


namespace NUMINAMATH_CALUDE_parallel_line_intercepts_l1727_172775

/-- A line parallel to y = 3x - 2 passing through (5, -1) has y-intercept -16 and x-intercept 16/3 -/
theorem parallel_line_intercepts :
  ∀ (b : ℝ → ℝ),
  (∀ x y, b y = 3 * x + (b 0 - 3 * 0)) →  -- b is parallel to y = 3x - 2
  b (-1) = 5 →  -- b passes through (5, -1)
  b 0 = -16 ∧ b (16/3) = 0 := by
sorry

end NUMINAMATH_CALUDE_parallel_line_intercepts_l1727_172775


namespace NUMINAMATH_CALUDE_rectangle_area_error_l1727_172718

theorem rectangle_area_error (L W : ℝ) (L_pos : L > 0) (W_pos : W > 0) :
  let erroneous_area := (1.02 * L) * (1.03 * W)
  let correct_area := L * W
  let percentage_error := (erroneous_area - correct_area) / correct_area * 100
  percentage_error = 5.06 := by
sorry

end NUMINAMATH_CALUDE_rectangle_area_error_l1727_172718


namespace NUMINAMATH_CALUDE_line_in_plane_theorem_l1727_172783

-- Define the types for our geometric objects
variable (Point Line Plane : Type)

-- Define the relations we need
variable (parallel_line_plane : Line → Plane → Prop)
variable (parallel_line_line : Line → Line → Prop)
variable (passes_through : Line → Point → Prop)
variable (point_in_plane : Point → Plane → Prop)
variable (line_in_plane : Line → Plane → Prop)

-- State the theorem
theorem line_in_plane_theorem 
  (a b : Line) (α : Plane) (M : Point)
  (h1 : parallel_line_plane a α)
  (h2 : parallel_line_line b a)
  (h3 : passes_through b M)
  (h4 : point_in_plane M α) :
  line_in_plane b α :=
sorry

end NUMINAMATH_CALUDE_line_in_plane_theorem_l1727_172783


namespace NUMINAMATH_CALUDE_sine_increases_with_angle_not_always_isosceles_right_angle_condition_obtuse_angle_from_ratio_l1727_172708

-- Define a triangle
structure Triangle where
  A : ℝ
  B : ℝ
  C : ℝ
  a : ℝ
  b : ℝ
  c : ℝ
  sum_angles : A + B + C = Real.pi
  positive_sides : a > 0 ∧ b > 0 ∧ c > 0

-- Statement A
theorem sine_increases_with_angle (t : Triangle) :
  t.A > t.B → Real.sin t.A > Real.sin t.B :=
sorry

-- Statement B
theorem not_always_isosceles (t : Triangle) :
  Real.sin (2 * t.A) = Real.sin (2 * t.B) →
  ¬(t.A = t.B ∧ t.a = t.b) :=
sorry

-- Statement C
theorem right_angle_condition (t : Triangle) :
  t.a * Real.cos t.B - t.b * Real.cos t.A = t.c →
  t.C = Real.pi / 2 :=
sorry

-- Statement D
theorem obtuse_angle_from_ratio (t : Triangle) :
  ∃ (k : ℝ), k > 0 ∧ t.a = 3*k ∧ t.b = 5*k ∧ t.c = 7*k →
  t.C > Real.pi / 2 :=
sorry

end NUMINAMATH_CALUDE_sine_increases_with_angle_not_always_isosceles_right_angle_condition_obtuse_angle_from_ratio_l1727_172708


namespace NUMINAMATH_CALUDE_ratio_of_11th_terms_l1727_172751

def arithmetic_sequence (a₁ d : ℚ) (n : ℕ) : ℚ := a₁ + (n - 1) * d

def sum_arithmetic_sequence (a₁ d : ℚ) (n : ℕ) : ℚ := n * (a₁ + (n - 1) / 2 * d)

theorem ratio_of_11th_terms
  (a₁ d₁ a₂ d₂ : ℚ)
  (h : ∀ n : ℕ, sum_arithmetic_sequence a₁ d₁ n / sum_arithmetic_sequence a₂ d₂ n = (7 * n + 1) / (4 * n + 27)) :
  (arithmetic_sequence a₁ d₁ 11) / (arithmetic_sequence a₂ d₂ 11) = 4 / 3 := by
  sorry

end NUMINAMATH_CALUDE_ratio_of_11th_terms_l1727_172751


namespace NUMINAMATH_CALUDE_point_on_line_l1727_172703

/-- Given a line passing through points (0, 3) and (-8, 0),
    if the point (t, 7) lies on this line, then t = 32/3 -/
theorem point_on_line (t : ℚ) :
  (∀ (x y : ℚ), (y - 3) / (x - 0) = (0 - 3) / (-8 - 0) →
    (7 - 3) / (t - 0) = (0 - 3) / (-8 - 0)) →
  t = 32 / 3 := by sorry

end NUMINAMATH_CALUDE_point_on_line_l1727_172703


namespace NUMINAMATH_CALUDE_g_increasing_on_negative_l1727_172758

-- Define the function f
variable (f : ℝ → ℝ)

-- Define the conditions on f
variable (h1 : ∀ x y, x < y → f x < f y)  -- f is increasing
variable (h2 : ∀ x, f x < 0)  -- f is always negative

-- Define the function g
def g (x : ℝ) : ℝ := x^2 * f x

-- State the theorem
theorem g_increasing_on_negative (x y : ℝ) (hx : x < 0) (hy : y < 0) (hxy : x < y) :
  g f x < g f y := by sorry

end NUMINAMATH_CALUDE_g_increasing_on_negative_l1727_172758


namespace NUMINAMATH_CALUDE_min_value_sqrt_sum_min_value_sqrt_sum_achieved_l1727_172764

theorem min_value_sqrt_sum (x : ℝ) : 
  Real.sqrt (x^2 + (2 - x)^2) + Real.sqrt ((2 - x)^2 + (-x)^2) ≥ 2 * Real.sqrt 2 :=
sorry

theorem min_value_sqrt_sum_achieved : 
  ∃ x : ℝ, Real.sqrt (x^2 + (2 - x)^2) + Real.sqrt ((2 - x)^2 + (-x)^2) = 2 * Real.sqrt 2 :=
sorry

end NUMINAMATH_CALUDE_min_value_sqrt_sum_min_value_sqrt_sum_achieved_l1727_172764


namespace NUMINAMATH_CALUDE_valid_arrangements_count_l1727_172727

def digits : List Nat := [1, 1, 5, 5]

def is_multiple_of_five (n : Nat) : Prop :=
  n % 5 = 0

def is_four_digit (n : Nat) : Prop :=
  n ≥ 1000 ∧ n < 10000

def count_valid_arrangements (ds : List Nat) : Nat :=
  sorry

theorem valid_arrangements_count :
  count_valid_arrangements digits = 3 := by
  sorry

end NUMINAMATH_CALUDE_valid_arrangements_count_l1727_172727


namespace NUMINAMATH_CALUDE_bacteria_growth_relation_l1727_172702

/-- Represents the amount of bacteria at a given time point -/
structure BacteriaAmount where
  amount : ℝ
  time : ℕ

/-- Represents a growth factor between two time points -/
structure GrowthFactor where
  factor : ℝ
  startTime : ℕ
  endTime : ℕ

/-- Theorem stating the relationship between initial, intermediate, and final bacteria amounts
    and their corresponding growth factors -/
theorem bacteria_growth_relation 
  (A₁ : BacteriaAmount) 
  (A₂ : BacteriaAmount) 
  (A₃ : BacteriaAmount) 
  (g : GrowthFactor) 
  (h : GrowthFactor) : 
  A₁.time = 1 →
  A₂.time = 4 →
  A₃.time = 7 →
  g.startTime = 1 →
  g.endTime = 4 →
  h.startTime = 4 →
  h.endTime = 7 →
  A₁.amount = 10 →
  A₃.amount = 12.1 →
  A₂.amount = A₁.amount * g.factor →
  A₃.amount = A₂.amount * h.factor →
  A₃.amount = A₁.amount * g.factor * h.factor :=
by sorry

end NUMINAMATH_CALUDE_bacteria_growth_relation_l1727_172702


namespace NUMINAMATH_CALUDE_shooting_probabilities_l1727_172766

/-- Represents the probability of hitting a specific ring in a shooting event -/
structure ShootingProbability where
  ring10 : ℝ
  ring9 : ℝ
  ring8 : ℝ

/-- Calculates the probability of hitting either the 10-ring or the 9-ring -/
def prob_10_or_9 (p : ShootingProbability) : ℝ :=
  p.ring10 + p.ring9

/-- Calculates the probability of hitting below the 8-ring -/
def prob_below_8 (p : ShootingProbability) : ℝ :=
  1 - (p.ring10 + p.ring9 + p.ring8)

/-- Theorem stating the probabilities for a given shooting event -/
theorem shooting_probabilities (p : ShootingProbability)
  (h1 : p.ring10 = 0.24)
  (h2 : p.ring9 = 0.28)
  (h3 : p.ring8 = 0.19) :
  prob_10_or_9 p = 0.52 ∧ prob_below_8 p = 0.29 := by
  sorry

end NUMINAMATH_CALUDE_shooting_probabilities_l1727_172766


namespace NUMINAMATH_CALUDE_local_arts_students_percentage_l1727_172724

/-- Proves that the percentage of local arts students is 50% --/
theorem local_arts_students_percentage
  (total_arts : ℕ)
  (total_science : ℕ)
  (total_commerce : ℕ)
  (science_local_percentage : ℚ)
  (commerce_local_percentage : ℚ)
  (total_local_percentage : ℕ)
  (h_total_arts : total_arts = 400)
  (h_total_science : total_science = 100)
  (h_total_commerce : total_commerce = 120)
  (h_science_local : science_local_percentage = 25/100)
  (h_commerce_local : commerce_local_percentage = 85/100)
  (h_total_local : total_local_percentage = 327)
  : ∃ (arts_local_percentage : ℚ),
    arts_local_percentage = 50/100 ∧
    arts_local_percentage * total_arts +
    science_local_percentage * total_science +
    commerce_local_percentage * total_commerce =
    total_local_percentage := by
  sorry


end NUMINAMATH_CALUDE_local_arts_students_percentage_l1727_172724


namespace NUMINAMATH_CALUDE_megacorp_fine_l1727_172755

/-- MegaCorp's fine calculation --/
theorem megacorp_fine :
  let daily_mining_profit : ℕ := 3000000
  let daily_oil_profit : ℕ := 5000000
  let monthly_expenses : ℕ := 30000000
  let days_per_year : ℕ := 365
  let months_per_year : ℕ := 12
  let fine_percentage : ℚ := 1 / 100

  let annual_revenue : ℕ := (daily_mining_profit + daily_oil_profit) * days_per_year
  let annual_expenses : ℕ := monthly_expenses * months_per_year
  let annual_profit : ℕ := annual_revenue - annual_expenses
  let fine : ℚ := (annual_profit : ℚ) * fine_percentage

  fine = 25600000 := by sorry

end NUMINAMATH_CALUDE_megacorp_fine_l1727_172755


namespace NUMINAMATH_CALUDE_ellipse_foci_distance_sum_l1727_172780

/-- An ellipse with semi-major axis 5 and semi-minor axis 4 -/
def Ellipse : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | (p.1^2 / 25) + (p.2^2 / 16) = 1}

/-- The foci of the ellipse -/
def Foci : (ℝ × ℝ) × (ℝ × ℝ) := sorry

/-- The distance between two points in ℝ² -/
def distance (p q : ℝ × ℝ) : ℝ := sorry

/-- Theorem: For any point on the ellipse, the sum of distances to the foci is 10 -/
theorem ellipse_foci_distance_sum (p : ℝ × ℝ) (h : p ∈ Ellipse) :
  distance p Foci.1 + distance p Foci.2 = 10 := by sorry

end NUMINAMATH_CALUDE_ellipse_foci_distance_sum_l1727_172780


namespace NUMINAMATH_CALUDE_properties_of_f_l1727_172768

def f (x : ℝ) := -5 * x

theorem properties_of_f :
  (∃ m b : ℝ, ∀ x, f x = m * x + b) ∧  -- f is linear
  (∀ x₁ x₂, x₁ < x₂ → f x₁ > f x₂) ∧   -- f is decreasing
  (f 0 = 0) ∧                          -- f passes through (0,0)
  (∀ x ≠ 0, x * (f x) < 0) :=          -- f is in 2nd and 4th quadrants
by sorry

end NUMINAMATH_CALUDE_properties_of_f_l1727_172768


namespace NUMINAMATH_CALUDE_derivative_of_f_l1727_172756

noncomputable def f (x : ℝ) : ℝ := x + 1 / x

theorem derivative_of_f :
  ∀ x : ℝ, x ≠ 0 → deriv f x = 1 - 1 / x^2 := by sorry

end NUMINAMATH_CALUDE_derivative_of_f_l1727_172756


namespace NUMINAMATH_CALUDE_rectangle_grid_ratio_l1727_172776

/-- Given a 3x2 grid of identical rectangles with height h and width w,
    and a line segment PQ intersecting the grid as described,
    prove that h/w = 3/8 -/
theorem rectangle_grid_ratio (h w : ℝ) (h_pos : h > 0) (w_pos : w > 0) : 
  let grid_width := 3 * w
  let grid_height := 2 * h
  ∃ (X Y Z : ℝ × ℝ),
    X.1 ∈ Set.Icc 0 grid_width ∧
    X.2 ∈ Set.Icc 0 grid_height ∧
    Z.1 ∈ Set.Icc 0 grid_width ∧
    Z.2 ∈ Set.Icc 0 grid_height ∧
    Y.1 = X.1 ∧
    Y.2 = Z.2 ∧
    (Z.1 - X.1)^2 + (Z.2 - X.2)^2 = (Y.1 - X.1)^2 + (Y.2 - X.2)^2 + (Z.1 - Y.1)^2 + (Z.2 - Y.2)^2 ∧
    (Z.1 - Y.1)^2 + (Z.2 - Y.2)^2 = 4 * ((Y.1 - X.1)^2 + (Y.2 - X.2)^2) →
  h / w = 3 / 8 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_grid_ratio_l1727_172776


namespace NUMINAMATH_CALUDE_johns_final_push_time_l1727_172744

/-- The time of John's final push in a race, given specific conditions. -/
theorem johns_final_push_time (john_speed steve_speed initial_distance final_distance : ℝ) 
  (h1 : john_speed = 4.2)
  (h2 : steve_speed = 3.7)
  (h3 : initial_distance = 14)
  (h4 : final_distance = 2)
  : (initial_distance + final_distance) / john_speed = 16 / 4.2 := by
  sorry

#eval (14 + 2) / 4.2

end NUMINAMATH_CALUDE_johns_final_push_time_l1727_172744


namespace NUMINAMATH_CALUDE_smallest_base_for_xyxy_cube_l1727_172721

/-- Represents a number in the form xyxy in base b -/
def xyxy_form (x y b : ℕ) : ℕ := x * b^3 + y * b^2 + x * b + y

/-- Checks if a number is a perfect cube -/
def is_perfect_cube (n : ℕ) : Prop := ∃ m : ℕ, m^3 = n

/-- The statement to be proved -/
theorem smallest_base_for_xyxy_cube : 
  (∀ x y : ℕ, ¬ is_perfect_cube (xyxy_form x y 10)) →
  (∃ x y : ℕ, is_perfect_cube (xyxy_form x y 7)) ∧
  (∀ b : ℕ, 1 < b → b < 7 → ∀ x y : ℕ, ¬ is_perfect_cube (xyxy_form x y b)) :=
sorry

end NUMINAMATH_CALUDE_smallest_base_for_xyxy_cube_l1727_172721


namespace NUMINAMATH_CALUDE_tangent_line_to_ellipse_l1727_172767

/-- Ellipse definition -/
def is_on_ellipse (x y : ℝ) : Prop :=
  x^2 / 16 + y^2 / 4 = 1

/-- Line equation -/
def line_equation (x y x₀ y₀ : ℝ) : Prop :=
  x * x₀ / 16 + y * y₀ / 4 = 1

/-- Tangent line property -/
def is_tangent_line (x₀ y₀ : ℝ) : Prop :=
  is_on_ellipse x₀ y₀ →
  ∀ x y : ℝ, line_equation x y x₀ y₀ →
  (x = x₀ ∧ y = y₀) ∨ ¬(is_on_ellipse x y)

/-- Main theorem -/
theorem tangent_line_to_ellipse :
  ∀ x₀ y₀ : ℝ, is_tangent_line x₀ y₀ :=
sorry

end NUMINAMATH_CALUDE_tangent_line_to_ellipse_l1727_172767


namespace NUMINAMATH_CALUDE_cauchy_schwarz_inequality_l1727_172779

theorem cauchy_schwarz_inequality (a b x y : ℝ) : (a^2 + b^2) * (x^2 + y^2) ≥ (a*x + b*y)^2 := by
  sorry

end NUMINAMATH_CALUDE_cauchy_schwarz_inequality_l1727_172779


namespace NUMINAMATH_CALUDE_abigail_report_time_l1727_172778

/-- The time it takes Abigail to finish her report -/
def report_completion_time (total_words : ℕ) (words_per_half_hour : ℕ) (words_written : ℕ) (proofreading_time : ℕ) : ℕ :=
  let words_left := total_words - words_written
  let half_hour_blocks := (words_left + words_per_half_hour - 1) / words_per_half_hour
  let writing_time := half_hour_blocks * 30
  writing_time + proofreading_time

/-- Theorem stating that Abigail will take 225 minutes to finish her report -/
theorem abigail_report_time :
  report_completion_time 1500 250 200 45 = 225 := by
  sorry

end NUMINAMATH_CALUDE_abigail_report_time_l1727_172778


namespace NUMINAMATH_CALUDE_max_x_value_l1727_172760

/-- Represents the linear relationship between x and y --/
def linear_relation (x y : ℝ) : Prop := y = x - 5

/-- The maximum forecast value for y --/
def max_y : ℝ := 10

/-- Theorem stating the maximum value of x given the conditions --/
theorem max_x_value (h : linear_relation max_y max_x) : max_x = 15 := by
  sorry

end NUMINAMATH_CALUDE_max_x_value_l1727_172760


namespace NUMINAMATH_CALUDE_new_rectangle_area_greater_than_square_l1727_172748

theorem new_rectangle_area_greater_than_square (a b : ℝ) (h1 : a > b) (h2 : b > 0) :
  let new_base := 2 * (a + b)
  let new_height := (2 * b + a) / 3
  let square_side := a + b
  new_base * new_height > square_side * square_side :=
by sorry

end NUMINAMATH_CALUDE_new_rectangle_area_greater_than_square_l1727_172748


namespace NUMINAMATH_CALUDE_slope_of_line_intersecting_hyperbola_l1727_172759

-- Define the hyperbola
def hyperbola (x y : ℝ) : Prop := x^2 - y^2/3 = 1

-- Define the line
def line (k : ℝ) (x y : ℝ) : Prop := y = k*(x - 3)

-- Define the distance from a point to the focus
def dist_to_focus (x y : ℝ) : ℝ := 2*x - 1

theorem slope_of_line_intersecting_hyperbola (k : ℝ) :
  (∃ A B : ℝ × ℝ,
    hyperbola A.1 A.2 ∧
    hyperbola B.1 B.2 ∧
    line k A.1 A.2 ∧
    line k B.1 B.2 ∧
    A.1 > 1 ∧
    B.1 > 1 ∧
    dist_to_focus A.1 A.2 + dist_to_focus B.1 B.2 = 16) →
  k = 3 ∨ k = -3 :=
sorry

end NUMINAMATH_CALUDE_slope_of_line_intersecting_hyperbola_l1727_172759


namespace NUMINAMATH_CALUDE_larger_square_construction_l1727_172797

/-- Represents a square in 2D space -/
structure Square where
  side : ℝ
  deriving Inhabited

/-- Represents the construction of a larger square from two smaller squares -/
def construct_larger_square (s1 s2 : Square) : Square :=
  sorry

/-- Theorem stating that it's possible to construct a larger square from two smaller squares
    without cutting the smaller square -/
theorem larger_square_construction (s1 s2 : Square) :
  ∃ (large : Square), 
    large.side^2 = s1.side^2 + s2.side^2 ∧
    construct_larger_square s1 s2 = large :=
  sorry

end NUMINAMATH_CALUDE_larger_square_construction_l1727_172797


namespace NUMINAMATH_CALUDE_range_of_m_when_p_implies_q_l1727_172707

/-- Represents an ellipse equation with parameter m -/
def is_ellipse_with_foci_on_y_axis (m : ℝ) : Prop :=
  ∃ x y : ℝ, x^2 / (2*m) + y^2 / (1-m) = 1 ∧ 0 < m ∧ m < 1/3

/-- Represents a hyperbola equation with parameter m -/
def is_hyperbola_with_eccentricity_between_1_and_2 (m : ℝ) : Prop :=
  ∃ x y e : ℝ, x^2 / 5 - y^2 / m = 1 ∧ 1 < e ∧ e < 2 ∧ m > 0

/-- The main theorem stating the range of m -/
theorem range_of_m_when_p_implies_q :
  (∀ m : ℝ, is_ellipse_with_foci_on_y_axis m → is_hyperbola_with_eccentricity_between_1_and_2 m) →
  ∃ m : ℝ, 1/3 ≤ m ∧ m < 15 :=
sorry

end NUMINAMATH_CALUDE_range_of_m_when_p_implies_q_l1727_172707


namespace NUMINAMATH_CALUDE_cuboid_surface_area_example_l1727_172765

/-- The surface area of a cuboid with given dimensions -/
def cuboidSurfaceArea (length breadth height : ℝ) : ℝ :=
  2 * (length * height + length * breadth + breadth * height)

/-- Theorem: The surface area of a cuboid with length 10 cm, breadth 8 cm, and height 6 cm is 376 cm² -/
theorem cuboid_surface_area_example : cuboidSurfaceArea 10 8 6 = 376 := by
  sorry

end NUMINAMATH_CALUDE_cuboid_surface_area_example_l1727_172765


namespace NUMINAMATH_CALUDE_max_stamps_with_50_dollars_l1727_172743

/-- The maximum number of stamps that can be purchased with a given budget and stamp price -/
def max_stamps (budget : ℕ) (stamp_price : ℕ) : ℕ :=
  (budget / stamp_price : ℕ)

/-- Theorem: Given a stamp price of 25 cents and a budget of 5000 cents, 
    the maximum number of stamps that can be purchased is 200 -/
theorem max_stamps_with_50_dollars :
  max_stamps 5000 25 = 200 := by
  sorry

end NUMINAMATH_CALUDE_max_stamps_with_50_dollars_l1727_172743


namespace NUMINAMATH_CALUDE_arithmetic_sequence_of_primes_l1727_172793

theorem arithmetic_sequence_of_primes : ∃ (p q r : ℕ), 
  (Nat.Prime p ∧ Nat.Prime q ∧ Nat.Prime r) ∧ 
  (p = 127 ∧ q = 3697 ∧ r = 5527) ∧
  (∃ (d : ℕ), 
    q * (q + 1) - p * (p + 1) = d ∧
    r * (r + 1) - q * (q + 1) = d ∧
    p * (p + 1) < q * (q + 1) ∧ q * (q + 1) < r * (r + 1)) :=
by sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_of_primes_l1727_172793


namespace NUMINAMATH_CALUDE_divisible_by_six_l1727_172714

theorem divisible_by_six (a : ℤ) : 6 ∣ a * (a + 1) * (2 * a + 1) := by
  sorry

end NUMINAMATH_CALUDE_divisible_by_six_l1727_172714


namespace NUMINAMATH_CALUDE_friend_team_assignment_l1727_172715

theorem friend_team_assignment (n : ℕ) (k : ℕ) (h1 : n = 8) (h2 : k = 4) :
  k^n = 65536 := by
  sorry

end NUMINAMATH_CALUDE_friend_team_assignment_l1727_172715


namespace NUMINAMATH_CALUDE_polynomial_division_existence_l1727_172752

theorem polynomial_division_existence :
  ∃ (Q R : Polynomial ℚ),
    4 * X^5 - 7 * X^4 + 3 * X^3 + 9 * X^2 - 23 * X + 8 = (5 * X^2 + 2 * X - 1) * Q + R ∧
    R.degree < (5 * X^2 + 2 * X - 1).degree := by
  sorry

end NUMINAMATH_CALUDE_polynomial_division_existence_l1727_172752


namespace NUMINAMATH_CALUDE_train_to_subway_ratio_l1727_172763

/-- Represents the travel times for Andrew's journey from Manhattan to the Bronx -/
structure TravelTimes where
  total : ℝ
  subway : ℝ
  biking : ℝ
  train : ℝ

/-- Theorem stating the ratio of train time to subway time -/
theorem train_to_subway_ratio (t : TravelTimes) 
  (h1 : t.total = 38)
  (h2 : t.subway = 10)
  (h3 : t.biking = 8)
  (h4 : t.train = t.total - t.subway - t.biking) :
  t.train / t.subway = 2 := by
  sorry

#check train_to_subway_ratio

end NUMINAMATH_CALUDE_train_to_subway_ratio_l1727_172763


namespace NUMINAMATH_CALUDE_line_perp_to_plane_and_line_para_to_plane_implies_lines_perp_l1727_172790

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the perpendicular and parallel relations
variable (perp : Line → Plane → Prop)
variable (para : Line → Plane → Prop)
variable (perpLine : Line → Line → Prop)

-- State the theorem
theorem line_perp_to_plane_and_line_para_to_plane_implies_lines_perp
  (m n : Line) (α : Plane)
  (h1 : perp m α)
  (h2 : para n α) :
  perpLine m n :=
sorry

end NUMINAMATH_CALUDE_line_perp_to_plane_and_line_para_to_plane_implies_lines_perp_l1727_172790


namespace NUMINAMATH_CALUDE_horner_rule_v2_value_l1727_172739

/-- Horner's Rule evaluation function -/
def horner_eval (coeffs : List ℝ) (x : ℝ) : ℝ :=
  coeffs.foldl (fun acc a => acc * x + a) 0

/-- The polynomial f(x) = x⁵ + 4x⁴ + x² + 20x + 16 -/
def f : ℝ → ℝ := fun x => x^5 + 4*x^4 + x^2 + 20*x + 16

/-- Coefficients of f(x) in descending order of degree -/
def f_coeffs : List ℝ := [1, 4, 0, 1, 20, 16]

theorem horner_rule_v2_value :
  let x := -2
  let v₂ := (horner_eval (f_coeffs.take 3) x)
  v₂ = -4 :=
by sorry

end NUMINAMATH_CALUDE_horner_rule_v2_value_l1727_172739


namespace NUMINAMATH_CALUDE_sine_graph_shift_l1727_172733

theorem sine_graph_shift (x : ℝ) :
  3 * Real.sin (2 * x + π / 4) = 3 * Real.sin (2 * (x + π / 8)) :=
by sorry

end NUMINAMATH_CALUDE_sine_graph_shift_l1727_172733


namespace NUMINAMATH_CALUDE_average_cost_is_two_l1727_172770

/-- The average cost of fruit given the prices and quantities of apples, bananas, and oranges. -/
def average_cost (apple_price banana_price orange_price : ℚ) 
                 (apple_qty banana_qty orange_qty : ℕ) : ℚ :=
  let total_cost := apple_price * apple_qty + banana_price * banana_qty + orange_price * orange_qty
  let total_qty := apple_qty + banana_qty + orange_qty
  total_cost / total_qty

/-- Theorem stating that the average cost of fruit is $2 given the specified prices and quantities. -/
theorem average_cost_is_two :
  average_cost 2 1 3 12 4 4 = 2 := by
  sorry

end NUMINAMATH_CALUDE_average_cost_is_two_l1727_172770


namespace NUMINAMATH_CALUDE_linear_functions_properties_l1727_172726

/-- Linear function y₁ -/
def y₁ (x : ℝ) : ℝ := 50 + 2 * x

/-- Linear function y₂ -/
def y₂ (x : ℝ) : ℝ := 5 * x

theorem linear_functions_properties :
  (∃ x : ℝ, y₁ x > y₂ x) ∧ 
  (∃ x : ℝ, y₁ x < y₂ x) ∧
  (∀ x dx : ℝ, y₁ (x + dx) - y₁ x = 2 * dx) ∧
  (∀ x dx : ℝ, y₂ (x + dx) - y₂ x = 5 * dx) ∧
  (∃ x₁ x₂ : ℝ, x₁ ≥ 1 ∧ x₂ ≥ 1 ∧ y₂ x₂ ≥ 100 ∧ y₁ x₁ ≥ 100 ∧ x₂ < x₁ ∧
    ∀ x : ℝ, x ≥ 1 → y₂ x ≥ 100 → x ≥ x₂) :=
by sorry

end NUMINAMATH_CALUDE_linear_functions_properties_l1727_172726


namespace NUMINAMATH_CALUDE_max_sum_of_factors_l1727_172757

theorem max_sum_of_factors (A B C : ℕ+) : 
  A ≠ B → B ≠ C → A ≠ C → A * B * C = 2310 → 
  ∀ (X Y Z : ℕ+), X ≠ Y → Y ≠ Z → X ≠ Z → X * Y * Z = 2310 → 
  A + B + C ≤ X + Y + Z → A + B + C ≤ 390 := by
sorry

end NUMINAMATH_CALUDE_max_sum_of_factors_l1727_172757


namespace NUMINAMATH_CALUDE_symmetric_point_wrt_x_axis_l1727_172700

/-- Given a point P(-2, 1), prove that its symmetric point Q with respect to the x-axis has coordinates (-2, -1) -/
theorem symmetric_point_wrt_x_axis :
  let P : ℝ × ℝ := (-2, 1)
  let Q : ℝ × ℝ := (-2, -1)
  (Q.1 = P.1) ∧ (Q.2 = -P.2) := by sorry

end NUMINAMATH_CALUDE_symmetric_point_wrt_x_axis_l1727_172700


namespace NUMINAMATH_CALUDE_stability_comparison_l1727_172754

/-- Represents a student's performance in standing long jump --/
structure StudentPerformance where
  average_score : ℝ
  variance : ℝ

/-- Defines stability of performance based on variance --/
def more_stable (a b : StudentPerformance) : Prop :=
  a.average_score = b.average_score ∧ a.variance < b.variance

/-- Theorem: Given two students with the same average score, 
    the one with lower variance has more stable performance --/
theorem stability_comparison 
  (student_A student_B : StudentPerformance)
  (h_same_average : student_A.average_score = student_B.average_score)
  (h_A_variance : student_A.variance = 0.6)
  (h_B_variance : student_B.variance = 0.35) :
  more_stable student_B student_A :=
sorry

end NUMINAMATH_CALUDE_stability_comparison_l1727_172754


namespace NUMINAMATH_CALUDE_special_trapezoid_ratio_l1727_172711

/-- Isosceles trapezoid with specific properties -/
structure SpecialTrapezoid where
  /-- Length of the shorter base -/
  a : ℝ
  /-- Length of the longer base -/
  long_base : ℝ
  /-- Length of the altitude -/
  altitude : ℝ
  /-- Length of a diagonal -/
  diagonal : ℝ
  /-- Longer base is square of shorter base -/
  long_base_eq : long_base = a^2
  /-- Shorter base equals altitude -/
  altitude_eq : altitude = a
  /-- Diagonal equals radius of circumscribed circle -/
  diagonal_eq : diagonal = 2

/-- The ratio of shorter base to longer base in the special trapezoid is 3/16 -/
theorem special_trapezoid_ratio (t : SpecialTrapezoid) : t.a / t.long_base = 3/16 := by
  sorry

end NUMINAMATH_CALUDE_special_trapezoid_ratio_l1727_172711


namespace NUMINAMATH_CALUDE_cab_driver_income_l1727_172705

theorem cab_driver_income (income : List ℝ) (average : ℝ) : 
  income.length = 5 →
  income[0]! = 600 →
  income[1]! = 250 →
  income[2]! = 450 →
  income[4]! = 800 →
  average = (income.sum / income.length) →
  average = 500 →
  income[3]! = 400 := by
sorry

end NUMINAMATH_CALUDE_cab_driver_income_l1727_172705


namespace NUMINAMATH_CALUDE_constant_term_equals_twenty_implies_n_equals_three_l1727_172777

/-- The constant term in the expansion of (x + 2 + 1/x)^n -/
def constant_term (n : ℕ) : ℕ := Nat.choose (2 * n) n

/-- The theorem stating that if the constant term is 20, then n = 3 -/
theorem constant_term_equals_twenty_implies_n_equals_three :
  ∃ n : ℕ, constant_term n = 20 ∧ n = 3 :=
sorry

end NUMINAMATH_CALUDE_constant_term_equals_twenty_implies_n_equals_three_l1727_172777


namespace NUMINAMATH_CALUDE_problem_statement_l1727_172773

theorem problem_statement (x : ℝ) (h : x^2 + x = 1) :
  3*x^4 + 3*x^3 + 3*x + 1 = 4 := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l1727_172773


namespace NUMINAMATH_CALUDE_divisibility_of_binomial_difference_l1727_172789

theorem divisibility_of_binomial_difference (p : ℕ) (a b : ℤ) (hp : Nat.Prime p) :
  ∃ k : ℤ, (a + b)^p - a^p - b^p = k * p :=
sorry

end NUMINAMATH_CALUDE_divisibility_of_binomial_difference_l1727_172789


namespace NUMINAMATH_CALUDE_smallest_sum_is_11_l1727_172710

/-- B is a digit in base 4 -/
def is_base_4_digit (B : ℕ) : Prop := B < 4

/-- b is a base greater than 5 -/
def is_base_greater_than_5 (b : ℕ) : Prop := b > 5

/-- BBB₄ = 44ᵦ -/
def equality_condition (B b : ℕ) : Prop := 21 * B = 4 * (b + 1)

/-- The smallest possible sum of B and b is 11 -/
theorem smallest_sum_is_11 :
  ∃ (B b : ℕ), is_base_4_digit B ∧ is_base_greater_than_5 b ∧ equality_condition B b ∧
  B + b = 11 ∧
  ∀ (B' b' : ℕ), is_base_4_digit B' → is_base_greater_than_5 b' → equality_condition B' b' →
  B' + b' ≥ 11 :=
sorry

end NUMINAMATH_CALUDE_smallest_sum_is_11_l1727_172710


namespace NUMINAMATH_CALUDE_factor_expression_l1727_172782

theorem factor_expression (x y : ℝ) : -x^2*y + 6*y^2*x - 9*y^3 = -y*(x-3*y)^2 := by
  sorry

end NUMINAMATH_CALUDE_factor_expression_l1727_172782


namespace NUMINAMATH_CALUDE_binomial_1350_2_l1727_172746

theorem binomial_1350_2 : Nat.choose 1350 2 = 910575 := by sorry

end NUMINAMATH_CALUDE_binomial_1350_2_l1727_172746


namespace NUMINAMATH_CALUDE_range_of_function_l1727_172787

theorem range_of_function : 
  ∀ y : ℝ, ∃ x : ℝ, |x + 3| - |x - 5| + 3 * x = y := by
sorry

end NUMINAMATH_CALUDE_range_of_function_l1727_172787


namespace NUMINAMATH_CALUDE_shirt_price_l1727_172799

theorem shirt_price (total_items : Nat) (dress_count : Nat) (shirt_count : Nat)
  (total_money : ℕ) (dress_price : ℕ) :
  total_items = dress_count + shirt_count →
  total_money = dress_count * dress_price + shirt_count * (total_money - dress_count * dress_price) / shirt_count →
  dress_count = 7 →
  shirt_count = 4 →
  total_money = 69 →
  dress_price = 7 →
  (total_money - dress_count * dress_price) / shirt_count = 5 := by
sorry

end NUMINAMATH_CALUDE_shirt_price_l1727_172799


namespace NUMINAMATH_CALUDE_bill_face_value_is_12250_l1727_172745

/-- Calculates the face value of a bill given the true discount, time period, and annual discount rate. -/
def calculate_face_value (true_discount : ℚ) (time : ℚ) (rate : ℚ) : ℚ :=
  (true_discount * (100 + rate * time)) / (rate * time)

/-- Theorem stating that given the specific conditions, the face value of the bill is 12250. -/
theorem bill_face_value_is_12250 :
  let true_discount : ℚ := 3500
  let time : ℚ := 2
  let rate : ℚ := 20
  calculate_face_value true_discount time rate = 12250 := by
  sorry

#eval calculate_face_value 3500 2 20

end NUMINAMATH_CALUDE_bill_face_value_is_12250_l1727_172745


namespace NUMINAMATH_CALUDE_same_even_on_all_dice_l1727_172771

/-- A standard six-sided die -/
def StandardDie : Type := Fin 6

/-- The probability of rolling an even number on a standard die -/
def probEven : ℚ := 1/2

/-- The probability of rolling a specific number on a standard die -/
def probSpecific : ℚ := 1/6

/-- The number of dice being rolled -/
def numDice : ℕ := 4

/-- Theorem: The probability of all dice showing the same even number -/
theorem same_even_on_all_dice : 
  probEven * probSpecific^(numDice - 1) = 1/432 := by sorry

end NUMINAMATH_CALUDE_same_even_on_all_dice_l1727_172771


namespace NUMINAMATH_CALUDE_unique_items_count_l1727_172729

/-- Represents a Beatles fan's collection --/
structure BeatlesFan where
  albums : ℕ
  memorabilia : ℕ

/-- Given the information about Andrew and John's collections, prove that the number of items
    in either Andrew's or John's collection or memorabilia, but not both, is 24. --/
theorem unique_items_count (andrew john : BeatlesFan) 
  (h1 : andrew.albums = 23)
  (h2 : andrew.memorabilia = 5)
  (h3 : john.albums = andrew.albums - 12 + 8) : 
  (andrew.albums - 12) + (john.albums - (andrew.albums - 12)) + andrew.memorabilia = 24 := by
  sorry

end NUMINAMATH_CALUDE_unique_items_count_l1727_172729


namespace NUMINAMATH_CALUDE_parallel_line_plane_intersection_false_l1727_172723

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the parallel and intersection relations
variable (parallel_line_plane : Line → Plane → Prop)
variable (parallel_lines : Line → Line → Prop)
variable (intersect_planes : Plane → Plane → Line → Prop)

-- Define our specific objects
variable (m n : Line)
variable (α β : Plane)

-- State that m and n are different lines
variable (m_neq_n : m ≠ n)

-- State that α and β are different planes
variable (α_neq_β : α ≠ β)

-- The theorem to be proved
theorem parallel_line_plane_intersection_false :
  ¬(∀ (m n : Line) (α β : Plane), 
    parallel_line_plane m α → intersect_planes α β n → parallel_lines m n) :=
sorry

end NUMINAMATH_CALUDE_parallel_line_plane_intersection_false_l1727_172723


namespace NUMINAMATH_CALUDE_factorization_a_squared_minus_8a_l1727_172735

theorem factorization_a_squared_minus_8a (a : ℝ) : a^2 - 8*a = a*(a - 8) := by
  sorry

end NUMINAMATH_CALUDE_factorization_a_squared_minus_8a_l1727_172735


namespace NUMINAMATH_CALUDE_octal_addition_47_56_l1727_172794

/-- Represents a digit in the octal system -/
def OctalDigit := Fin 8

/-- Represents a number in the octal system as a list of digits -/
def OctalNumber := List OctalDigit

/-- Addition operation for octal numbers -/
def octal_add : OctalNumber → OctalNumber → OctalNumber := sorry

/-- Conversion from a natural number to an octal number -/
def nat_to_octal : ℕ → OctalNumber := sorry

/-- Conversion from an octal number to a natural number -/
def octal_to_nat : OctalNumber → ℕ := sorry

theorem octal_addition_47_56 :
  octal_add (nat_to_octal 47) (nat_to_octal 56) = nat_to_octal 125 := by sorry

end NUMINAMATH_CALUDE_octal_addition_47_56_l1727_172794


namespace NUMINAMATH_CALUDE_equation_solution_l1727_172753

theorem equation_solution :
  ∃! x : ℚ, ∀ y : ℚ, 10 * x * y - 15 * y + 2 * x - 3 = 0 :=
by
  use 3/2
  sorry

end NUMINAMATH_CALUDE_equation_solution_l1727_172753


namespace NUMINAMATH_CALUDE_hyperbola_eccentricity_is_sqrt_5_l1727_172706

/-- Represents a hyperbola with semi-major axis a and semi-minor axis b -/
structure Hyperbola (a b : ℝ) where
  (a_pos : a > 0)
  (b_pos : b > 0)

/-- The eccentricity of a hyperbola -/
def eccentricity (h : Hyperbola a b) : ℝ := sorry

/-- Predicate indicating that the focus of a hyperbola is symmetric with respect to its asymptote -/
def focus_symmetric_to_asymptote (h : Hyperbola a b) : Prop := sorry

/-- Predicate indicating that the focus of a hyperbola lies on the hyperbola -/
def focus_on_hyperbola (h : Hyperbola a b) : Prop := sorry

/-- Theorem stating that if the focus of a hyperbola is symmetric with respect to its asymptote
    and lies on the hyperbola, then its eccentricity is √5 -/
theorem hyperbola_eccentricity_is_sqrt_5 {a b : ℝ} (h : Hyperbola a b)
  (h_sym : focus_symmetric_to_asymptote h) (h_on : focus_on_hyperbola h) :
  eccentricity h = Real.sqrt 5 := by sorry

end NUMINAMATH_CALUDE_hyperbola_eccentricity_is_sqrt_5_l1727_172706


namespace NUMINAMATH_CALUDE_password_guess_probability_l1727_172720

/-- The number of digits in the password -/
def password_length : ℕ := 6

/-- The number of possible digits for each position -/
def digit_options : ℕ := 10

/-- The number of attempts allowed -/
def max_attempts : ℕ := 2

/-- The probability of guessing the correct last digit in no more than 2 attempts -/
theorem password_guess_probability :
  (1 : ℚ) / digit_options + (digit_options - 1 : ℚ) / digit_options * (1 : ℚ) / (digit_options - 1) = 1 / 5 := by
  sorry

end NUMINAMATH_CALUDE_password_guess_probability_l1727_172720


namespace NUMINAMATH_CALUDE_tangent_slope_at_point_A_l1727_172737

-- Define the curve
def f (x : ℝ) : ℝ := 2 * x^2

-- State the theorem
theorem tangent_slope_at_point_A :
  let x₀ : ℝ := 2
  let y₀ : ℝ := f x₀
  (y₀ = 8) →  -- This ensures the point (2,8) is on the curve
  (deriv f x₀ = 8) :=
by sorry

end NUMINAMATH_CALUDE_tangent_slope_at_point_A_l1727_172737


namespace NUMINAMATH_CALUDE_set_a_forms_triangle_l1727_172740

/-- Triangle Inequality Theorem: A set of three line segments can form a triangle
    if and only if the sum of the lengths of any two sides is strictly greater
    than the length of the remaining side. -/
def can_form_triangle (a b c : ℝ) : Prop :=
  a + b > c ∧ b + c > a ∧ c + a > b

/-- Prove that the set of line segments (3, 5, 7) can form a triangle. -/
theorem set_a_forms_triangle : can_form_triangle 3 5 7 := by
  sorry

end NUMINAMATH_CALUDE_set_a_forms_triangle_l1727_172740


namespace NUMINAMATH_CALUDE_isosceles_right_triangles_in_quadrilateral_l1727_172722

-- Define the points
variable (A B C D O₁ O₂ O₃ O₄ : Point)

-- Define the quadrilateral
def is_convex_quadrilateral (A B C D : Point) : Prop := sorry

-- Define isosceles right triangle
def is_isosceles_right_triangle (X Y Z : Point) : Prop := sorry

-- State the theorem
theorem isosceles_right_triangles_in_quadrilateral 
  (h_quad : is_convex_quadrilateral A B C D)
  (h_ABO₁ : is_isosceles_right_triangle A B O₁)
  (h_BCO₂ : is_isosceles_right_triangle B C O₂)
  (h_CDO₃ : is_isosceles_right_triangle C D O₃)
  (h_DAO₄ : is_isosceles_right_triangle D A O₄)
  (h_O₁_O₃ : O₁ = O₃) :
  O₂ = O₄ := by sorry

end NUMINAMATH_CALUDE_isosceles_right_triangles_in_quadrilateral_l1727_172722


namespace NUMINAMATH_CALUDE_solution_set_of_inequality_l1727_172781

theorem solution_set_of_inequality (x : ℝ) :
  (3 * x^2 + 7 * x ≤ 2) ↔ (-2 ≤ x ∧ x ≤ 1/3) := by
  sorry

end NUMINAMATH_CALUDE_solution_set_of_inequality_l1727_172781


namespace NUMINAMATH_CALUDE_complement_of_A_l1727_172785

-- Define the set A
def A : Set ℝ := {x | x^2 - 2*x > 0}

-- State the theorem
theorem complement_of_A : 
  {x : ℝ | ¬ (x ∈ A)} = {x : ℝ | 0 ≤ x ∧ x ≤ 2} := by
  sorry

end NUMINAMATH_CALUDE_complement_of_A_l1727_172785


namespace NUMINAMATH_CALUDE_trigonometric_equation_solution_l1727_172713

theorem trigonometric_equation_solution 
  (a b c : ℝ) 
  (h : a ≠ 0 ∨ b ≠ 0) :
  ∃ (n : ℤ), 
    ∀ (x : ℝ), 
      a * Real.sin x + b * Real.cos x = c → 
        x = Real.arctan (b / a) + n * Real.pi :=
sorry

end NUMINAMATH_CALUDE_trigonometric_equation_solution_l1727_172713


namespace NUMINAMATH_CALUDE_power_equation_solution_l1727_172798

theorem power_equation_solution : ∃! x : ℝ, (5 : ℝ)^3 + (5 : ℝ)^3 + (5 : ℝ)^3 = (15 : ℝ)^x := by
  sorry

end NUMINAMATH_CALUDE_power_equation_solution_l1727_172798


namespace NUMINAMATH_CALUDE_eight_term_sequence_sum_l1727_172738

def sequence_sum (seq : List ℤ) (i : ℕ) : ℤ :=
  if i + 2 < seq.length then
    seq[i]! + seq[i+1]! + seq[i+2]!
  else
    0

theorem eight_term_sequence_sum (P Q R S T U V W : ℤ) : 
  R = 8 →
  (∀ i, i + 2 < 8 → sequence_sum [P, Q, R, S, T, U, V, W] i = 35) →
  P + W = 27 := by
sorry

end NUMINAMATH_CALUDE_eight_term_sequence_sum_l1727_172738


namespace NUMINAMATH_CALUDE_hcf_problem_l1727_172784

theorem hcf_problem (a b : ℕ+) : 
  (∃ h : ℕ+, Nat.lcm a b = h * 13 * 14) →
  max a b = 322 →
  Nat.gcd a b = 7 := by
sorry

end NUMINAMATH_CALUDE_hcf_problem_l1727_172784


namespace NUMINAMATH_CALUDE_xyz_sum_l1727_172734

theorem xyz_sum (x y z : ℕ+) 
  (h1 : x * y = 32)
  (h2 : x * z = 64)
  (h3 : y * z = 96) :
  x + y + z = 28 := by
  sorry

end NUMINAMATH_CALUDE_xyz_sum_l1727_172734


namespace NUMINAMATH_CALUDE_problem_solution_l1727_172772

theorem problem_solution : ∃ n : ℕ+, 
  (24 ∣ n) ∧ 
  (8.2 < (n : ℝ) ^ (1/3 : ℝ)) ∧ 
  ((n : ℝ) ^ (1/3 : ℝ) < 8.3) := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l1727_172772


namespace NUMINAMATH_CALUDE_total_cost_correct_l1727_172730

def makeup_palette_price : ℝ := 15
def lipstick_price : ℝ := 2.5
def hair_color_price : ℝ := 4

def makeup_palette_count : ℕ := 3
def lipstick_count : ℕ := 4
def hair_color_count : ℕ := 3

def makeup_palette_discount : ℝ := 0.2
def hair_color_coupon_discount : ℝ := 0.1
def reward_points_discount : ℝ := 5

def storewide_discount_threshold : ℝ := 50
def storewide_discount_rate : ℝ := 0.1

def sales_tax_threshold : ℝ := 25
def sales_tax_rate_low : ℝ := 0.05
def sales_tax_rate_high : ℝ := 0.08

def calculate_total_cost : ℝ := sorry

theorem total_cost_correct : 
  ∀ ε > 0, |calculate_total_cost - 47.41| < ε := by sorry

end NUMINAMATH_CALUDE_total_cost_correct_l1727_172730


namespace NUMINAMATH_CALUDE_largest_multiple_of_seven_less_than_negative_hundred_l1727_172786

theorem largest_multiple_of_seven_less_than_negative_hundred :
  ∀ n : ℤ, n * 7 < -100 → n * 7 ≤ -105 :=
by
  sorry

end NUMINAMATH_CALUDE_largest_multiple_of_seven_less_than_negative_hundred_l1727_172786


namespace NUMINAMATH_CALUDE_sibling_ages_l1727_172731

theorem sibling_ages (x y z : ℕ) : 
  x - y = 3 →                    -- Age difference between siblings
  z - 1 = 2 * (x + y) →          -- Father's age one year ago
  z + 20 = (x + 20) + (y + 20) → -- Father's age in 20 years
  (x = 11 ∧ y = 8) :=            -- Ages of the siblings
by sorry

end NUMINAMATH_CALUDE_sibling_ages_l1727_172731


namespace NUMINAMATH_CALUDE_vikas_rank_among_boys_l1727_172742

/-- Represents the ranking information of students in a class -/
structure ClassRanking where
  total_students : ℕ
  vikas_overall_rank : ℕ
  tanvi_overall_rank : ℕ
  girls_between : ℕ
  vikas_boys_top_rank : ℕ
  vikas_bottom_rank : ℕ

/-- The theorem to prove Vikas's rank among boys -/
theorem vikas_rank_among_boys (c : ClassRanking) 
  (h1 : c.vikas_overall_rank = 9)
  (h2 : c.tanvi_overall_rank = 17)
  (h3 : c.girls_between = 2)
  (h4 : c.vikas_boys_top_rank = 4)
  (h5 : c.vikas_bottom_rank = 18) :
  c.vikas_boys_top_rank = 4 := by
  sorry


end NUMINAMATH_CALUDE_vikas_rank_among_boys_l1727_172742


namespace NUMINAMATH_CALUDE_min_cuts_for_polygons_l1727_172788

theorem min_cuts_for_polygons (n : ℕ) (k : ℕ) (s : ℕ) : 
  n = 73 ∧ s = 30 ∧ k ≥ n - 1 ∧ 
  (n * ((s - 2) * π) + (k + 1 - n) * π ≤ (k + 1) * 2 * π) →
  k ≥ 1970 :=
sorry

end NUMINAMATH_CALUDE_min_cuts_for_polygons_l1727_172788


namespace NUMINAMATH_CALUDE_inscribed_circle_radius_l1727_172762

/-- The radius of an inscribed circle in a sector -/
theorem inscribed_circle_radius (R : ℝ) (h : R = 5) :
  let sector_angle : ℝ := 2 * Real.pi / 3
  let r : ℝ := R * (Real.sqrt 3 - 1) / 2
  r = (5 * Real.sqrt 3 - 5) / 2 ∧ 
  r > 0 ∧ 
  r * (Real.sqrt 3 + 1) = R := by
  sorry

end NUMINAMATH_CALUDE_inscribed_circle_radius_l1727_172762


namespace NUMINAMATH_CALUDE_room_length_calculation_l1727_172709

/-- The length of a room given carpet and room dimensions -/
theorem room_length_calculation (total_cost carpet_width_cm carpet_price_per_m room_breadth : ℝ)
  (h1 : total_cost = 810)
  (h2 : carpet_width_cm = 75)
  (h3 : carpet_price_per_m = 4.5)
  (h4 : room_breadth = 7.5) :
  total_cost / (carpet_price_per_m * room_breadth * (carpet_width_cm / 100)) = 18 := by
  sorry

end NUMINAMATH_CALUDE_room_length_calculation_l1727_172709


namespace NUMINAMATH_CALUDE_carlas_marbles_l1727_172719

/-- Given that Carla had some marbles and bought more, prove how many she has now. -/
theorem carlas_marbles (initial : ℝ) (bought : ℝ) (total : ℝ) 
  (h1 : initial = 187.0) 
  (h2 : bought = 134.0) 
  (h3 : total = initial + bought) : 
  total = 321.0 := by
  sorry


end NUMINAMATH_CALUDE_carlas_marbles_l1727_172719


namespace NUMINAMATH_CALUDE_nonagon_diagonals_l1727_172769

/-- The number of diagonals in a convex polygon with n sides -/
def num_diagonals (n : ℕ) : ℕ := n * (n - 3) / 2

/-- A nonagon is a polygon with 9 sides -/
def is_nonagon (n : ℕ) : Prop := n = 9

theorem nonagon_diagonals :
  ∀ n : ℕ, is_nonagon n → num_diagonals n = 27 := by sorry

end NUMINAMATH_CALUDE_nonagon_diagonals_l1727_172769


namespace NUMINAMATH_CALUDE_profit_percentage_l1727_172792

theorem profit_percentage (selling_price cost_price : ℝ) (h : cost_price = 0.81 * selling_price) :
  (selling_price - cost_price) / cost_price * 100 = (100/81 - 1) * 100 := by
  sorry

end NUMINAMATH_CALUDE_profit_percentage_l1727_172792


namespace NUMINAMATH_CALUDE_negative_cube_equality_l1727_172717

theorem negative_cube_equality : (-3)^3 = -3^3 := by
  sorry

end NUMINAMATH_CALUDE_negative_cube_equality_l1727_172717


namespace NUMINAMATH_CALUDE_final_highway_length_l1727_172741

def highway_extension (initial_length day1_construction day2_multiplier additional_miles : ℕ) : ℕ := 
  initial_length + day1_construction + day1_construction * day2_multiplier + additional_miles

theorem final_highway_length :
  highway_extension 200 50 3 250 = 650 := by
  sorry

end NUMINAMATH_CALUDE_final_highway_length_l1727_172741


namespace NUMINAMATH_CALUDE_sasha_sticker_collection_l1727_172704

theorem sasha_sticker_collection (m n : ℕ) (t : ℝ) : 
  m < n →
  m > 0 →
  t > 1 →
  m * t + n = 100 →
  m + n * t = 101 →
  (n = 34 ∨ n = 66) ∧ ∀ k : ℕ, (k ≠ 34 ∧ k ≠ 66) → 
    ¬(∃ m' : ℕ, ∃ t' : ℝ, 
      m' < k ∧ 
      m' > 0 ∧ 
      t' > 1 ∧ 
      m' * t' + k = 100 ∧ 
      m' + k * t' = 101) :=
by sorry

end NUMINAMATH_CALUDE_sasha_sticker_collection_l1727_172704


namespace NUMINAMATH_CALUDE_questionnaire_responses_l1727_172796

theorem questionnaire_responses (response_rate : ℝ) (questionnaires_mailed : ℕ) 
  (h1 : response_rate = 0.8)
  (h2 : questionnaires_mailed = 375) :
  ⌊response_rate * questionnaires_mailed⌋ = 300 := by
  sorry

end NUMINAMATH_CALUDE_questionnaire_responses_l1727_172796


namespace NUMINAMATH_CALUDE_spaceship_age_conversion_l1727_172716

/-- Converts an octal number represented as a list of digits to its decimal equivalent -/
def octal_to_decimal (digits : List Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * (8 ^ i)) 0

/-- The octal representation of the spaceship's age -/
def spaceship_age_octal : List Nat := [3, 4, 7]

theorem spaceship_age_conversion :
  octal_to_decimal spaceship_age_octal = 483 := by
  sorry

#eval octal_to_decimal spaceship_age_octal

end NUMINAMATH_CALUDE_spaceship_age_conversion_l1727_172716


namespace NUMINAMATH_CALUDE_differentiable_functions_inequality_l1727_172761

open Set

theorem differentiable_functions_inequality 
  (f g : ℝ → ℝ) 
  (hf : Differentiable ℝ f) 
  (hg : Differentiable ℝ g)
  (h_deriv : ∀ x, deriv f x > deriv g x) 
  (a b x : ℝ) 
  (h_x : x ∈ Ioo a b) : 
  (f x + g b < g x + f b) ∧ (f x + g a > g x + f a) := by
  sorry

end NUMINAMATH_CALUDE_differentiable_functions_inequality_l1727_172761


namespace NUMINAMATH_CALUDE_abs_neg_2023_l1727_172791

theorem abs_neg_2023 : |(-2023 : ℤ)| = 2023 := by
  sorry

end NUMINAMATH_CALUDE_abs_neg_2023_l1727_172791


namespace NUMINAMATH_CALUDE_grill_runtime_proof_l1727_172725

/-- Represents the burning rate of coals in a grill -/
structure BurningRate :=
  (coals : ℕ)
  (minutes : ℕ)

/-- Represents a bag of coals -/
structure CoalBag :=
  (coals : ℕ)

def grill_running_time (rate : BurningRate) (bags : ℕ) (bag : CoalBag) : ℕ :=
  (bags * bag.coals * rate.minutes) / rate.coals

theorem grill_runtime_proof (rate : BurningRate) (bags : ℕ) (bag : CoalBag)
  (h1 : rate.coals = 15)
  (h2 : rate.minutes = 20)
  (h3 : bags = 3)
  (h4 : bag.coals = 60) :
  grill_running_time rate bags bag = 240 :=
by
  sorry

#check grill_runtime_proof

end NUMINAMATH_CALUDE_grill_runtime_proof_l1727_172725


namespace NUMINAMATH_CALUDE_divide_flour_possible_l1727_172747

/-- Represents the result of a weighing operation -/
inductive Weighing
| MeasuredFlour (amount : ℕ)
| CombinedFlour (amount1 amount2 : ℕ)

/-- Represents a weighing operation using the balance scale -/
def weigh (yeast ginger flour : ℕ) : Weighing :=
  sorry

/-- Represents the process of dividing flour using two weighings -/
def divideFlour (totalFlour yeast ginger : ℕ) : Option (ℕ × ℕ) :=
  sorry

/-- Theorem stating that it's possible to divide 500g of flour into 400g and 100g parts
    using only two weighings with 5g of yeast and 30g of ginger -/
theorem divide_flour_possible :
  ∃ (w1 w2 : Weighing),
    let result := divideFlour 500 5 30
    result = some (400, 100) ∧
    (∃ (f1 : ℕ), w1 = Weighing.MeasuredFlour f1) ∧
    (∃ (f2 : ℕ), w2 = Weighing.MeasuredFlour f2) ∧
    f1 + f2 = 100 :=
  sorry

end NUMINAMATH_CALUDE_divide_flour_possible_l1727_172747


namespace NUMINAMATH_CALUDE_fib_equals_tiling_pred_l1727_172749

/-- Fibonacci sequence -/
def fib : ℕ → ℕ
  | 0 => 0
  | 1 => 1
  | n + 2 => fib (n + 1) + fib n

/-- Number of ways to tile a 1 × n rectangle with 1 × 1 squares and 1 × 2 dominos -/
def tiling : ℕ → ℕ
  | 0 => 1
  | 1 => 1
  | n + 2 => tiling (n + 1) + tiling n

/-- Theorem: The n-th Fibonacci number equals the number of ways to tile a 1 × (n-1) rectangle -/
theorem fib_equals_tiling_pred (n : ℕ) : fib n = tiling (n - 1) := by
  sorry

end NUMINAMATH_CALUDE_fib_equals_tiling_pred_l1727_172749


namespace NUMINAMATH_CALUDE_triangle_problem_l1727_172701

theorem triangle_problem (a b c : ℝ) (A B C : ℝ) :
  0 < A ∧ A < π ∧ 0 < B ∧ B < π ∧ 0 < C ∧ C < π →
  A + B + C = π →
  Real.sqrt 3 * a = 2 * b * Real.sin A →
  a = 6 →
  1/2 * a * c * Real.sin B = 6 * Real.sqrt 3 →
  ((B = π/3 ∨ B = 2*π/3) ∧ (b = 2 * Real.sqrt 7 ∨ b = Real.sqrt 76)) := by
  sorry

end NUMINAMATH_CALUDE_triangle_problem_l1727_172701


namespace NUMINAMATH_CALUDE_largest_A_for_divisibility_by_3_l1727_172712

def is_divisible_by_3 (n : ℕ) : Prop := n % 3 = 0

def digit_sum (n : ℕ) : ℕ :=
  let digits := n.digits 10
  digits.sum

theorem largest_A_for_divisibility_by_3 :
  ∀ A : ℕ, A ≤ 9 →
    is_divisible_by_3 (3 * 100000 + A * 10000 + 6 * 1000 + 7 * 100 + 9 * 10 + 2) →
    A ≤ 9 :=
by sorry

end NUMINAMATH_CALUDE_largest_A_for_divisibility_by_3_l1727_172712


namespace NUMINAMATH_CALUDE_maria_savings_l1727_172728

/-- The amount of money Maria will have left after buying sweaters and scarves -/
def money_left (sweater_price scarf_price num_sweaters num_scarves savings : ℕ) : ℕ :=
  savings - (sweater_price * num_sweaters + scarf_price * num_scarves)

/-- Theorem stating that Maria will have $200 left after her purchases -/
theorem maria_savings : money_left 30 20 6 6 500 = 200 := by
  sorry

end NUMINAMATH_CALUDE_maria_savings_l1727_172728


namespace NUMINAMATH_CALUDE_set_equality_implies_sum_zero_l1727_172750

theorem set_equality_implies_sum_zero (x y : ℝ) : 
  ({x, y, x + y} : Set ℝ) = {0, x^2, x*y} → x + y = 0 := by
  sorry

end NUMINAMATH_CALUDE_set_equality_implies_sum_zero_l1727_172750


namespace NUMINAMATH_CALUDE_balls_after_2023_steps_l1727_172732

/-- Converts a natural number to its base-8 representation --/
def toBase8 (n : ℕ) : List ℕ :=
  if n = 0 then [0] else
  let rec go (m : ℕ) (acc : List ℕ) : List ℕ :=
    if m = 0 then acc else go (m / 8) ((m % 8) :: acc)
  go n []

/-- Sums the digits in a list of natural numbers --/
def sumDigits (l : List ℕ) : ℕ :=
  l.sum

/-- The process of placing balls in boxes as described in the problem --/
def ballProcess (steps : ℕ) : ℕ :=
  sumDigits (toBase8 steps)

/-- The theorem stating that the number of balls after 2023 steps is 21 --/
theorem balls_after_2023_steps :
  ballProcess 2023 = 21 := by
  sorry

end NUMINAMATH_CALUDE_balls_after_2023_steps_l1727_172732
