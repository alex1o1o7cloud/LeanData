import Mathlib

namespace NUMINAMATH_CALUDE_smores_per_person_l1046_104692

/-- Proves that given the conditions of the S'mores problem, each person will eat 3 S'mores -/
theorem smores_per_person 
  (total_people : ℕ) 
  (cost_per_set : ℚ) 
  (smores_per_set : ℕ) 
  (total_cost : ℚ) 
  (h1 : total_people = 8)
  (h2 : cost_per_set = 3)
  (h3 : smores_per_set = 4)
  (h4 : total_cost = 18) :
  (total_cost / cost_per_set * smores_per_set) / total_people = 3 := by
sorry

end NUMINAMATH_CALUDE_smores_per_person_l1046_104692


namespace NUMINAMATH_CALUDE_work_relation_l1046_104678

/-- Represents an isothermal process of a gas -/
structure IsothermalProcess where
  pressure : ℝ → ℝ
  volume : ℝ → ℝ
  work : ℝ

/-- The work done on a gas during an isothermal process -/
def work_done (p : IsothermalProcess) : ℝ := p.work

/-- Condition: The volume in process 1-2 is twice the volume in process 3-4 for any given pressure -/
def volume_relation (p₁₂ p₃₄ : IsothermalProcess) : Prop :=
  ∀ t, p₁₂.volume t = 2 * p₃₄.volume t

/-- Theorem: The work done on the gas in process 1-2 is twice the work done in process 3-4 -/
theorem work_relation (p₁₂ p₃₄ : IsothermalProcess) 
  (h : volume_relation p₁₂ p₃₄) : 
  work_done p₁₂ = 2 * work_done p₃₄ := by
  sorry

end NUMINAMATH_CALUDE_work_relation_l1046_104678


namespace NUMINAMATH_CALUDE_a_6_equals_25_l1046_104619

/-- An increasing geometric sequence -/
def is_increasing_geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, q > 1 ∧ ∀ n : ℕ, a (n + 1) = a n * q

/-- The roots of x^2 - 6x + 5 = 0 -/
def is_root_of_equation (x : ℝ) : Prop :=
  x^2 - 6*x + 5 = 0

/-- Theorem: For an increasing geometric sequence where a_2 and a_4 are roots of x^2 - 6x + 5 = 0, a_6 = 25 -/
theorem a_6_equals_25 (a : ℕ → ℝ) 
  (h1 : is_increasing_geometric_sequence a)
  (h2 : is_root_of_equation (a 2))
  (h3 : is_root_of_equation (a 4)) :
  a 6 = 25 :=
sorry

end NUMINAMATH_CALUDE_a_6_equals_25_l1046_104619


namespace NUMINAMATH_CALUDE_cylinder_surface_area_doubling_l1046_104666

theorem cylinder_surface_area_doubling (r h : ℝ) : 
  r > 0 → h > 0 →
  2 * Real.pi * r^2 + 2 * Real.pi * r * h = 300 →
  8 * Real.pi * r^2 + 4 * Real.pi * r * h = 900 →
  h = r →
  2 * Real.pi * r^2 + 2 * Real.pi * r * (2 * h) = 450 :=
by sorry

end NUMINAMATH_CALUDE_cylinder_surface_area_doubling_l1046_104666


namespace NUMINAMATH_CALUDE_sibling_ages_sum_l1046_104651

/-- Given four positive integers representing ages of siblings, prove that their sum is 24 --/
theorem sibling_ages_sum (x y z : ℕ) (h1 : 2 * x^2 + y^2 + z^2 = 194) (h2 : x > y) (h3 : y > z) :
  x + x + y + z = 24 := by
  sorry

end NUMINAMATH_CALUDE_sibling_ages_sum_l1046_104651


namespace NUMINAMATH_CALUDE_camping_trip_percentage_l1046_104690

theorem camping_trip_percentage (total_students : ℕ) 
  (h1 : (22 : ℝ) / 100 * total_students = (25 : ℝ) / 100 * ((88 : ℝ) / 100 * total_students)) 
  (h2 : (75 : ℝ) / 100 * ((88 : ℝ) / 100 * total_students) + (25 : ℝ) / 100 * ((88 : ℝ) / 100 * total_students) = (88 : ℝ) / 100 * total_students) : 
  (88 : ℝ) / 100 * total_students = (22 : ℝ) / (25 / 100) := by
  sorry

end NUMINAMATH_CALUDE_camping_trip_percentage_l1046_104690


namespace NUMINAMATH_CALUDE_rectangle_ratio_l1046_104661

/-- Represents the configuration of rectangles around a quadrilateral -/
structure RectangleConfiguration where
  s : ℝ  -- side length of the inner quadrilateral
  x : ℝ  -- shorter side of each rectangle
  y : ℝ  -- longer side of each rectangle
  h1 : s > 0  -- side length is positive
  h2 : x > 0  -- rectangle sides are positive
  h3 : y > 0
  h4 : (s + 2*x)^2 = 4*s^2  -- area relation
  h5 : s + 2*y = 2*s  -- relation for y sides

/-- The ratio of y to x is 1 in the given configuration -/
theorem rectangle_ratio (config : RectangleConfiguration) : config.y / config.x = 1 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_ratio_l1046_104661


namespace NUMINAMATH_CALUDE_always_ahead_probability_l1046_104629

/-- Represents the probability that candidate A's cumulative vote count 
    always remains ahead of candidate B's during the counting process 
    in an election where A receives n votes and B receives m votes. -/
def election_probability (n m : ℕ) : ℚ :=
  (n - m : ℚ) / (n + m : ℚ)

/-- Theorem stating the probability that candidate A's cumulative vote count 
    always remains ahead of candidate B's during the counting process. -/
theorem always_ahead_probability (n m : ℕ) (h : n > m) :
  election_probability n m = (n - m : ℚ) / (n + m : ℚ) := by
  sorry

end NUMINAMATH_CALUDE_always_ahead_probability_l1046_104629


namespace NUMINAMATH_CALUDE_equation_solutions_l1046_104635

theorem equation_solutions :
  (∀ x : ℝ, x^2 - 1 = 8 ↔ x = 3 ∨ x = -3) ∧
  (∀ x : ℝ, (x + 4)^3 = -64 ↔ x = -8) := by
sorry

end NUMINAMATH_CALUDE_equation_solutions_l1046_104635


namespace NUMINAMATH_CALUDE_five_valid_configurations_l1046_104617

/-- Represents a square in the figure -/
structure Square :=
  (label : Char)

/-- Represents the L-shaped figure -/
structure LShape :=
  (squares : Finset Square)
  (size : Nat)
  (h_size : size = 4)

/-- Represents the set of additional squares -/
structure AdditionalSquares :=
  (squares : Finset Square)
  (size : Nat)
  (h_size : size = 8)

/-- Represents a configuration formed by adding one square to the L-shape -/
structure Configuration :=
  (base : LShape)
  (added : Square)

/-- Predicate to determine if a configuration can be folded into a topless cubical box -/
def canFoldIntoCube (config : Configuration) : Prop :=
  sorry

/-- The main theorem stating that exactly 5 configurations can be folded into a topless cubical box -/
theorem five_valid_configurations
  (l : LShape)
  (extras : AdditionalSquares) :
  ∃! (validConfigs : Finset Configuration),
    validConfigs.card = 5 ∧
    ∀ (config : Configuration),
      config ∈ validConfigs ↔
        (config.base = l ∧
         config.added ∈ extras.squares ∧
         canFoldIntoCube config) :=
sorry

end NUMINAMATH_CALUDE_five_valid_configurations_l1046_104617


namespace NUMINAMATH_CALUDE_inequality_range_l1046_104602

theorem inequality_range (a : ℝ) : 
  (∀ x : ℝ, x > 0 → x - Real.log x - a > 0) ↔ a < 1 := by
  sorry

end NUMINAMATH_CALUDE_inequality_range_l1046_104602


namespace NUMINAMATH_CALUDE_always_even_l1046_104653

theorem always_even (m n : ℤ) : 
  ∃ k : ℤ, (2*m + 1)^2 + 3*(2*m + 1)*(2*n + 1) = 2*k := by
sorry

end NUMINAMATH_CALUDE_always_even_l1046_104653


namespace NUMINAMATH_CALUDE_prob_diff_colors_is_11_18_l1046_104646

def num_blue : ℕ := 6
def num_yellow : ℕ := 4
def num_red : ℕ := 2
def total_chips : ℕ := num_blue + num_yellow + num_red

def prob_diff_colors : ℚ :=
  (num_blue : ℚ) / total_chips * ((num_yellow + num_red) : ℚ) / total_chips +
  (num_yellow : ℚ) / total_chips * ((num_blue + num_red) : ℚ) / total_chips +
  (num_red : ℚ) / total_chips * ((num_blue + num_yellow) : ℚ) / total_chips

theorem prob_diff_colors_is_11_18 : prob_diff_colors = 11 / 18 := by
  sorry

end NUMINAMATH_CALUDE_prob_diff_colors_is_11_18_l1046_104646


namespace NUMINAMATH_CALUDE_digit_sum_problem_l1046_104695

theorem digit_sum_problem (X Y Z : ℕ) : 
  X < 10 → Y < 10 → Z < 10 →
  100 * X + 10 * Y + Z + 100 * X + 10 * Y + Z + 10 * Y + Z = 1675 →
  X + Y + Z = 15 := by
sorry

end NUMINAMATH_CALUDE_digit_sum_problem_l1046_104695


namespace NUMINAMATH_CALUDE_rectangular_solid_volume_l1046_104634

theorem rectangular_solid_volume 
  (side_area front_area bottom_area : ℝ)
  (h_side : side_area = 15)
  (h_front : front_area = 20)
  (h_bottom : bottom_area = 12) :
  ∃ (a b c : ℝ), a * b = side_area ∧ b * c = front_area ∧ c * a = bottom_area ∧ a * b * c = 60 :=
by sorry

end NUMINAMATH_CALUDE_rectangular_solid_volume_l1046_104634


namespace NUMINAMATH_CALUDE_poster_height_l1046_104669

/-- Given a rectangular poster with width 4 inches and area 28 square inches, its height is 7 inches. -/
theorem poster_height (width : ℝ) (area : ℝ) (height : ℝ) 
    (h_width : width = 4)
    (h_area : area = 28)
    (h_rect_area : area = width * height) : height = 7 := by
  sorry

end NUMINAMATH_CALUDE_poster_height_l1046_104669


namespace NUMINAMATH_CALUDE_protein_percentage_of_first_meal_l1046_104605

-- Define the constants
def total_weight : ℝ := 280
def mixture_protein_percentage : ℝ := 13
def cornmeal_protein_percentage : ℝ := 7
def first_meal_weight : ℝ := 240
def cornmeal_weight : ℝ := total_weight - first_meal_weight

-- Define the theorem
theorem protein_percentage_of_first_meal :
  let total_protein := total_weight * mixture_protein_percentage / 100
  let cornmeal_protein := cornmeal_weight * cornmeal_protein_percentage / 100
  let first_meal_protein := total_protein - cornmeal_protein
  first_meal_protein / first_meal_weight * 100 = 14 := by
sorry

end NUMINAMATH_CALUDE_protein_percentage_of_first_meal_l1046_104605


namespace NUMINAMATH_CALUDE_p_equiv_simplified_p_sum_of_squares_of_coefficients_l1046_104677

/-- The polynomial p(x) defined by the given expression -/
def p (x : ℝ) : ℝ := 5 * (x^2 - 3*x + 4) - 8 * (x^3 - x^2 + 2*x - 3)

/-- The simplified form of p(x) -/
def simplified_p (x : ℝ) : ℝ := -8*x^3 + 13*x^2 - 31*x + 44

/-- Theorem stating that p(x) is equivalent to its simplified form -/
theorem p_equiv_simplified_p : p = simplified_p := by sorry

/-- Theorem proving the sum of squares of coefficients of simplified_p is 3130 -/
theorem sum_of_squares_of_coefficients :
  (-8)^2 + 13^2 + (-31)^2 + 44^2 = 3130 := by sorry

end NUMINAMATH_CALUDE_p_equiv_simplified_p_sum_of_squares_of_coefficients_l1046_104677


namespace NUMINAMATH_CALUDE_inequality_proof_l1046_104668

theorem inequality_proof (a b c : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) 
  (h4 : a + b + c = 1) 
  (h5 : ∀ x : ℝ, |x - a| + |x - 1| ≥ (a^2 + b^2 + c^2) / (b + c)) : 
  a ≤ Real.sqrt 2 - 1 := by
sorry

end NUMINAMATH_CALUDE_inequality_proof_l1046_104668


namespace NUMINAMATH_CALUDE_lucky_number_2015_l1046_104685

/-- A function that returns the sum of digits of a positive integer -/
def sumOfDigits (n : ℕ+) : ℕ := sorry

/-- A function that returns true if a positive integer is a "lucky number" (sum of digits is 8) -/
def isLuckyNumber (n : ℕ+) : Prop := sumOfDigits n = 8

/-- A function that returns the nth "lucky number" -/
def nthLuckyNumber (n : ℕ+) : ℕ+ := sorry

theorem lucky_number_2015 : nthLuckyNumber 106 = 2015 := by sorry

end NUMINAMATH_CALUDE_lucky_number_2015_l1046_104685


namespace NUMINAMATH_CALUDE_phil_final_quarters_l1046_104670

/-- Calculates the number of quarters Phil has after four years of collecting and losing some. -/
def phil_quarters : ℕ :=
  let initial := 50
  let after_first_year := initial * 2
  let second_year_collection := 3 * 12
  let third_year_collection := 12 / 3
  let total_before_loss := after_first_year + second_year_collection + third_year_collection
  let quarters_lost := total_before_loss / 4
  total_before_loss - quarters_lost

/-- Theorem stating that Phil ends up with 105 quarters after four years. -/
theorem phil_final_quarters : phil_quarters = 105 := by
  sorry

end NUMINAMATH_CALUDE_phil_final_quarters_l1046_104670


namespace NUMINAMATH_CALUDE_system_solutions_l1046_104688

def is_solution (x y z : ℝ) : Prop :=
  x^3 + y^3 + z^3 = 8 ∧
  x^2 + y^2 + z^2 = 22 ∧
  1/x + 1/y + 1/z + z/(x*y) = 0

theorem system_solutions :
  (is_solution 3 2 (-3)) ∧
  (is_solution (-3) 2 3) ∧
  (is_solution 2 3 (-3)) ∧
  (is_solution 2 (-3) 3) :=
by sorry

end NUMINAMATH_CALUDE_system_solutions_l1046_104688


namespace NUMINAMATH_CALUDE_charlies_data_usage_l1046_104696

/-- Represents Charlie's cell phone data usage problem -/
theorem charlies_data_usage 
  (data_limit : ℝ) 
  (extra_cost_per_gb : ℝ)
  (week1_usage : ℝ)
  (week2_usage : ℝ)
  (week3_usage : ℝ)
  (extra_charge : ℝ)
  (h1 : data_limit = 8)
  (h2 : extra_cost_per_gb = 10)
  (h3 : week1_usage = 2)
  (h4 : week2_usage = 3)
  (h5 : week3_usage = 5)
  (h6 : extra_charge = 120)
  : ∃ (week4_usage : ℝ), 
    week4_usage = 10 ∧ 
    (week1_usage + week2_usage + week3_usage + week4_usage - data_limit) * extra_cost_per_gb = extra_charge :=
sorry

end NUMINAMATH_CALUDE_charlies_data_usage_l1046_104696


namespace NUMINAMATH_CALUDE_special_right_triangle_hypotenuse_l1046_104643

/-- A right triangle with specific properties -/
structure SpecialRightTriangle where
  /-- Length of the shorter leg -/
  short_leg : ℝ
  /-- Length of the longer leg -/
  long_leg : ℝ
  /-- Length of the hypotenuse -/
  hypotenuse : ℝ
  /-- The longer leg is 2 feet longer than twice the shorter leg -/
  leg_relation : long_leg = 2 * short_leg + 2
  /-- The area of the triangle is 96 square feet -/
  area_constraint : (1 / 2) * short_leg * long_leg = 96
  /-- Pythagorean theorem holds -/
  pythagorean : short_leg ^ 2 + long_leg ^ 2 = hypotenuse ^ 2

/-- Theorem: The hypotenuse of the special right triangle is √388 feet -/
theorem special_right_triangle_hypotenuse (t : SpecialRightTriangle) :
  t.hypotenuse = Real.sqrt 388 := by
  sorry

end NUMINAMATH_CALUDE_special_right_triangle_hypotenuse_l1046_104643


namespace NUMINAMATH_CALUDE_tangent_line_at_x_1_l1046_104611

-- Define the function f
def f (x : ℝ) : ℝ := -x^2 + 8*x

-- Define the derivative of f
def f' (x : ℝ) : ℝ := -2*x + 8

-- Theorem statement
theorem tangent_line_at_x_1 :
  let x₀ : ℝ := 1
  let y₀ : ℝ := f x₀
  let m : ℝ := f' x₀
  ∀ x y : ℝ, y - y₀ = m * (x - x₀) ↔ y = 6*x + 1 :=
by sorry

end NUMINAMATH_CALUDE_tangent_line_at_x_1_l1046_104611


namespace NUMINAMATH_CALUDE_blue_part_length_l1046_104633

/-- Proves that the blue part of a pencil is 3.5 cm long given specific conditions -/
theorem blue_part_length (total_length : ℝ) (black_ratio : ℝ) (white_ratio : ℝ)
  (h1 : total_length = 8)
  (h2 : black_ratio = 1 / 8)
  (h3 : white_ratio = 1 / 2)
  (h4 : black_ratio * total_length + white_ratio * (total_length - black_ratio * total_length) +
    (total_length - black_ratio * total_length - white_ratio * (total_length - black_ratio * total_length)) = total_length) :
  total_length - black_ratio * total_length - white_ratio * (total_length - black_ratio * total_length) = 3.5 := by
  sorry

end NUMINAMATH_CALUDE_blue_part_length_l1046_104633


namespace NUMINAMATH_CALUDE_extreme_point_of_f_l1046_104612

-- Define the function f(x)
def f (x : ℝ) : ℝ := (x^2 - 1)^3 + 2

-- State the theorem
theorem extreme_point_of_f :
  ∃! x : ℝ, ∀ y : ℝ, f y ≥ f x :=
  by sorry

end NUMINAMATH_CALUDE_extreme_point_of_f_l1046_104612


namespace NUMINAMATH_CALUDE_stratified_sampling_probability_l1046_104659

/-- The probability of selecting an individual in a sampling method -/
def sampling_probability (m : ℕ) (sample_size : ℕ) : ℚ := 1 / sample_size

theorem stratified_sampling_probability (m : ℕ) (h : m ≥ 3) :
  sampling_probability m 3 = 1 / 3 := by sorry

end NUMINAMATH_CALUDE_stratified_sampling_probability_l1046_104659


namespace NUMINAMATH_CALUDE_complex_equation_product_l1046_104627

theorem complex_equation_product (a b : ℝ) (i : ℂ) :
  i * i = -1 →
  a + b * i = 5 / (1 + 2 * i) →
  a * b = -2 := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_product_l1046_104627


namespace NUMINAMATH_CALUDE_sequence_properties_l1046_104624

def is_arithmetic_sequence (a : ℕ → ℕ) : Prop :=
  ∃ d, ∀ n, a (n + 1) = a n + d

theorem sequence_properties
  (a : ℕ → ℕ)
  (h_increasing : ∀ n, a n < a (n + 1))
  (h_positive : ∀ n, 0 < a n)
  (b : ℕ → ℕ)
  (h_b : ∀ n, b n = a (a n))
  (c : ℕ → ℕ)
  (h_c : ∀ n, c n = a (a (n + 1)))
  (h_b_value : ∀ n, b n = 3 * n)
  (h_c_arithmetic : is_arithmetic_sequence c ∧ ∀ n, c (n + 1) = c n + 1) :
  a 1 = 2 ∧ c 1 = 6 ∧ is_arithmetic_sequence a :=
by sorry

end NUMINAMATH_CALUDE_sequence_properties_l1046_104624


namespace NUMINAMATH_CALUDE_painted_cube_one_third_blue_iff_three_l1046_104665

/-- Represents a cube with side length n, painted blue on all faces and cut into n^3 unit cubes -/
structure PaintedCube where
  n : ℕ

/-- The total number of faces of all unit cubes -/
def PaintedCube.totalFaces (c : PaintedCube) : ℕ := 6 * c.n^3

/-- The number of blue faces among all unit cubes -/
def PaintedCube.blueFaces (c : PaintedCube) : ℕ := 6 * c.n^2

/-- The condition that exactly one-third of the total faces are blue -/
def PaintedCube.oneThirdBlue (c : PaintedCube) : Prop :=
  3 * c.blueFaces = c.totalFaces

theorem painted_cube_one_third_blue_iff_three (c : PaintedCube) :
  c.oneThirdBlue ↔ c.n = 3 := by sorry

end NUMINAMATH_CALUDE_painted_cube_one_third_blue_iff_three_l1046_104665


namespace NUMINAMATH_CALUDE_tims_garden_carrots_l1046_104671

/-- Represents the number of carrots in Tim's garden -/
def carrots : ℕ := sorry

/-- Represents the number of potatoes in Tim's garden -/
def potatoes : ℕ := sorry

/-- The ratio of carrots to potatoes -/
def ratio : Rat := 3 / 4

/-- The initial number of potatoes -/
def initial_potatoes : ℕ := 32

/-- The number of potatoes added -/
def added_potatoes : ℕ := 28

theorem tims_garden_carrots : 
  (ratio = carrots / potatoes) → 
  (potatoes = initial_potatoes + added_potatoes) →
  carrots = 45 := by sorry

end NUMINAMATH_CALUDE_tims_garden_carrots_l1046_104671


namespace NUMINAMATH_CALUDE_union_of_sets_l1046_104682

theorem union_of_sets (S T : Set ℕ) (h1 : S = {0, 1}) (h2 : T = {0}) : 
  S ∪ T = {0, 1} := by
sorry

end NUMINAMATH_CALUDE_union_of_sets_l1046_104682


namespace NUMINAMATH_CALUDE_school_days_per_week_l1046_104676

theorem school_days_per_week 
  (daily_usage_per_class : ℕ) 
  (weekly_total_usage : ℕ) 
  (num_classes : ℕ) 
  (h1 : daily_usage_per_class = 200)
  (h2 : weekly_total_usage = 9000)
  (h3 : num_classes = 9) :
  weekly_total_usage / (daily_usage_per_class * num_classes) = 5 := by
sorry

end NUMINAMATH_CALUDE_school_days_per_week_l1046_104676


namespace NUMINAMATH_CALUDE_negative_four_squared_l1046_104660

theorem negative_four_squared : -4^2 = -16 := by
  sorry

end NUMINAMATH_CALUDE_negative_four_squared_l1046_104660


namespace NUMINAMATH_CALUDE_ellipse_point_properties_l1046_104673

-- Define the ellipse C
def ellipse (x y : ℝ) : Prop := x^2 / 2 + y^2 = 1

-- Define the point P
structure Point (x₀ y₀ : ℝ) where
  inside_ellipse : 0 < x₀^2 / 2 + y₀^2
  inside_ellipse' : x₀^2 / 2 + y₀^2 < 1

-- Define the line passing through P
def line (x₀ y₀ x y : ℝ) : Prop := x₀ * x / 2 + y₀ * y = 1

-- Theorem statement
theorem ellipse_point_properties {x₀ y₀ : ℝ} (P : Point x₀ y₀) :
  -- 1. Range of |PF₁| + |PF₂|
  ∃ (PF₁ PF₂ : ℝ), 2 ≤ PF₁ + PF₂ ∧ PF₁ + PF₂ < 2 * Real.sqrt 2 ∧
  -- 2. No common points between the line and ellipse
  ∀ (x y : ℝ), line x₀ y₀ x y → ¬ ellipse x y :=
sorry

end NUMINAMATH_CALUDE_ellipse_point_properties_l1046_104673


namespace NUMINAMATH_CALUDE_percentage_problem_l1046_104694

theorem percentage_problem (p : ℝ) : 
  (25 / 100 * 840 = p / 100 * 1500 - 15) → p = 15 := by sorry

end NUMINAMATH_CALUDE_percentage_problem_l1046_104694


namespace NUMINAMATH_CALUDE_girls_count_l1046_104645

theorem girls_count (boys girls : ℕ) : 
  (boys : ℚ) / girls = 8 / 5 →
  boys + girls = 351 →
  girls = 135 := by
sorry

end NUMINAMATH_CALUDE_girls_count_l1046_104645


namespace NUMINAMATH_CALUDE_quadratic_equation_properties_l1046_104632

theorem quadratic_equation_properties :
  ∃ (p q : ℝ),
    (∀ (r : ℝ), (∀ (x : ℝ), x^2 - (r+7)*x + r + 87 = 0 →
      (∃ (y : ℝ), y ≠ x ∧ y^2 - (r+7)*y + r + 87 = 0) ∧
      x < 0) ↔ p < r ∧ r < q) ∧
    p^2 + q^2 = 8098 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equation_properties_l1046_104632


namespace NUMINAMATH_CALUDE_negation_equivalence_l1046_104628

theorem negation_equivalence :
  (¬ ∃ x₀ : ℝ, (x₀^2 + 1 > 0 ∨ x₀ > Real.sin x₀)) ↔ 
  (∀ x : ℝ, (x^2 + 1 ≤ 0 ∧ x ≤ Real.sin x)) :=
by sorry

end NUMINAMATH_CALUDE_negation_equivalence_l1046_104628


namespace NUMINAMATH_CALUDE_fourth_intersection_point_l1046_104662

/-- A point in the 2D plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- The curve xy = 2 -/
def onCurve (p : Point) : Prop := p.x * p.y = 2

/-- A circle in the 2D plane -/
structure Circle where
  center : Point
  radius : ℝ

/-- A point lies on a circle -/
def onCircle (p : Point) (c : Circle) : Prop :=
  (p.x - c.center.x)^2 + (p.y - c.center.y)^2 = c.radius^2

/-- The theorem statement -/
theorem fourth_intersection_point
  (c : Circle)
  (h1 : onCurve ⟨2, 1⟩ ∧ onCircle ⟨2, 1⟩ c)
  (h2 : onCurve ⟨-4, -1/2⟩ ∧ onCircle ⟨-4, -1/2⟩ c)
  (h3 : onCurve ⟨1/2, 4⟩ ∧ onCircle ⟨1/2, 4⟩ c)
  (h4 : ∃ (p : Point), onCurve p ∧ onCircle p c ∧ p ≠ ⟨2, 1⟩ ∧ p ≠ ⟨-4, -1/2⟩ ∧ p ≠ ⟨1/2, 4⟩) :
  ∃ (p : Point), p = ⟨-1, -2⟩ ∧ onCurve p ∧ onCircle p c :=
sorry

end NUMINAMATH_CALUDE_fourth_intersection_point_l1046_104662


namespace NUMINAMATH_CALUDE_solution_difference_l1046_104663

-- Define the equation
def equation (r : ℝ) : Prop :=
  (r^2 - 6*r - 20) / (r + 3) = 3*r + 10

-- Define the solutions
def solutions : Set ℝ :=
  {r : ℝ | equation r ∧ r ≠ -3}

-- Theorem statement
theorem solution_difference :
  ∃ (r₁ r₂ : ℝ), r₁ ∈ solutions ∧ r₂ ∈ solutions ∧ r₁ ≠ r₂ ∧ |r₁ - r₂| = 20 :=
by sorry

end NUMINAMATH_CALUDE_solution_difference_l1046_104663


namespace NUMINAMATH_CALUDE_stratified_sampling_most_appropriate_l1046_104606

/-- Represents a sampling method -/
inductive SamplingMethod
| Lottery
| RandomNumber
| Systematic
| Stratified

/-- Represents a population with two equal-sized subgroups -/
structure Population :=
  (size : ℕ)
  (subgroup1_size : ℕ)
  (subgroup2_size : ℕ)
  (h_equal_size : subgroup1_size = subgroup2_size)
  (h_total_size : subgroup1_size + subgroup2_size = size)

/-- Represents the goal of understanding differences between subgroups -/
def UnderstandDifferences : Prop := True

/-- The most appropriate sampling method for a given population and goal -/
def MostAppropriateSamplingMethod (p : Population) (goal : UnderstandDifferences) : SamplingMethod :=
  SamplingMethod.Stratified

/-- Theorem stating that stratified sampling is the most appropriate method 
    for a population with two equal-sized subgroups when the goal is to 
    understand differences between these subgroups -/
theorem stratified_sampling_most_appropriate 
  (p : Population) (goal : UnderstandDifferences) :
  MostAppropriateSamplingMethod p goal = SamplingMethod.Stratified :=
by
  sorry


end NUMINAMATH_CALUDE_stratified_sampling_most_appropriate_l1046_104606


namespace NUMINAMATH_CALUDE_soap_brand_ratio_l1046_104658

def total_households : ℕ := 240
def households_neither : ℕ := 80
def households_only_A : ℕ := 60
def households_both : ℕ := 25

theorem soap_brand_ratio :
  ∃ (households_only_B : ℕ),
    households_only_A + households_only_B + households_both + households_neither = total_households ∧
    households_only_B / households_both = 3 := by
  sorry

end NUMINAMATH_CALUDE_soap_brand_ratio_l1046_104658


namespace NUMINAMATH_CALUDE_gold_asymmetric_probability_l1046_104649

/-- Represents a coin --/
inductive Coin
| Gold
| Silver

/-- Represents the symmetry of a coin --/
inductive Symmetry
| Symmetric
| Asymmetric

/-- The probability of getting heads for the asymmetric coin --/
def asymmetricHeadsProbability : ℝ := 0.6

/-- The result of a coin flip --/
inductive FlipResult
| Heads
| Tails

/-- Represents the sequence of coin flips --/
structure FlipSequence where
  goldResult : FlipResult
  silverResult1 : FlipResult
  silverResult2 : FlipResult

/-- The observed flip sequence --/
def observedFlips : FlipSequence := {
  goldResult := FlipResult.Heads,
  silverResult1 := FlipResult.Tails,
  silverResult2 := FlipResult.Heads
}

/-- The probability that the gold coin is asymmetric given the observed flip sequence --/
def probGoldAsymmetric (flips : FlipSequence) : ℝ := sorry

theorem gold_asymmetric_probability :
  probGoldAsymmetric observedFlips = 6/10 := by sorry

end NUMINAMATH_CALUDE_gold_asymmetric_probability_l1046_104649


namespace NUMINAMATH_CALUDE_impossible_non_eleven_multiple_l1046_104640

/-- Represents a 5x5 board where each cell can be increased along with its adjacent cells. -/
structure Board :=
  (cells : Matrix (Fin 5) (Fin 5) ℕ)

/-- The operation of increasing a cell and its adjacent cells by 1. -/
def increase_cell (b : Board) (i j : Fin 5) : Board := sorry

/-- Checks if all cells in the board have the same value. -/
def all_cells_equal (b : Board) (s : ℕ) : Prop := sorry

/-- Main theorem: It's impossible to obtain a number not divisible by 11 in all cells. -/
theorem impossible_non_eleven_multiple (s : ℕ) (h : ¬ 11 ∣ s) : 
  ¬ ∃ (b : Board), all_cells_equal b s :=
sorry

end NUMINAMATH_CALUDE_impossible_non_eleven_multiple_l1046_104640


namespace NUMINAMATH_CALUDE_equations_hold_l1046_104683

-- Define the equations
def equation1 : ℝ := 6.8 + 4.1 + 1.1
def equation2 : ℝ := 6.2 + 6.2 + 7.6
def equation3 : ℝ := 19.9 - 4.3 - 5.6

-- State the theorem
theorem equations_hold :
  equation1 = 12 ∧ equation2 = 20 ∧ equation3 = 10 := by sorry

end NUMINAMATH_CALUDE_equations_hold_l1046_104683


namespace NUMINAMATH_CALUDE_expression_evaluation_l1046_104616

theorem expression_evaluation (c : ℕ) (h : c = 4) :
  (c^c + c*(c+1)^c)^c = 5750939763536 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l1046_104616


namespace NUMINAMATH_CALUDE_parabola_intersection_length_l1046_104621

-- Define the parabola
def parabola (x y : ℝ) : Prop := y^2 = 4*x

-- Define the focus of the parabola
def focus : ℝ × ℝ := (1, 0)

-- Define a line passing through the focus
def line_through_focus (m b : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.2 = m * p.1 + b ∧ focus ∈ {p : ℝ × ℝ | p.2 = m * p.1 + b}}

-- Define the intersection points A and B
def intersection_points (m b : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | parabola p.1 p.2 ∧ p ∈ line_through_focus m b}

-- State the theorem
theorem parabola_intersection_length 
  (m b : ℝ) 
  (A B : ℝ × ℝ) 
  (h_A : A ∈ intersection_points m b) 
  (h_B : B ∈ intersection_points m b) 
  (h_midpoint : (A.1 + B.1) / 2 = 3) :
  Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) = 8 := by
  sorry

end NUMINAMATH_CALUDE_parabola_intersection_length_l1046_104621


namespace NUMINAMATH_CALUDE_quadratic_solution_difference_squared_l1046_104647

theorem quadratic_solution_difference_squared : 
  ∀ α β : ℝ, α ≠ β → α^2 = 2*α + 2 → β^2 = 2*β + 2 → (α - β)^2 = 12 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_solution_difference_squared_l1046_104647


namespace NUMINAMATH_CALUDE_coin_probability_l1046_104679

theorem coin_probability (p q : ℝ) (hq : q = 1 - p) 
  (h : (Nat.choose 10 5 : ℝ) * p^5 * q^5 = (Nat.choose 10 6 : ℝ) * p^6 * q^4) : 
  p = 6/11 := by
sorry

end NUMINAMATH_CALUDE_coin_probability_l1046_104679


namespace NUMINAMATH_CALUDE_symmetric_point_xOy_l1046_104604

def xOy_plane : Set (ℝ × ℝ × ℝ) := {p | p.2.2 = 0}

def symmetric_point (p : ℝ × ℝ × ℝ) (plane : Set (ℝ × ℝ × ℝ)) : ℝ × ℝ × ℝ :=
  (p.1, p.2.1, -p.2.2)

theorem symmetric_point_xOy : 
  symmetric_point (2, 3, 4) xOy_plane = (2, 3, -4) := by sorry

end NUMINAMATH_CALUDE_symmetric_point_xOy_l1046_104604


namespace NUMINAMATH_CALUDE_chorus_students_l1046_104675

theorem chorus_students (total : ℕ) (band : ℕ) (both : ℕ) (neither : ℕ) :
  total = 50 →
  band = 26 →
  both = 2 →
  neither = 8 →
  ∃ chorus : ℕ, chorus = 18 ∧ chorus + band - both = total - neither :=
by sorry

end NUMINAMATH_CALUDE_chorus_students_l1046_104675


namespace NUMINAMATH_CALUDE_polynomial_factorization_l1046_104691

theorem polynomial_factorization (x : ℤ) :
  5 * (x + 4) * (x + 7) * (x + 9) * (x + 11) - 4 * x^2 =
  (5 * x + 63) * (x + 3) * (x + 5) * (x + 21) := by
sorry

end NUMINAMATH_CALUDE_polynomial_factorization_l1046_104691


namespace NUMINAMATH_CALUDE_pigeon_win_conditions_l1046_104648

/-- The game result for the pigeon -/
inductive GameResult
| Win
| Lose

/-- Determines the game result for the pigeon given the board size, egg count, and seagull's square size -/
def pigeonWins (n : ℕ) (m : ℕ) (k : ℕ) : GameResult :=
  if k ≤ n ∧ n ≤ 2 * k - 1 ∧ m ≥ k^2 then GameResult.Win
  else if n ≥ 2 * k ∧ m ≥ k^2 + 1 then GameResult.Win
  else GameResult.Lose

/-- Theorem stating the conditions for the pigeon to win -/
theorem pigeon_win_conditions (n : ℕ) (m : ℕ) (k : ℕ) (h : n ≥ k) :
  (k ≤ n ∧ n ≤ 2 * k - 1 → (pigeonWins n m k = GameResult.Win ↔ m ≥ k^2)) ∧
  (n ≥ 2 * k → (pigeonWins n m k = GameResult.Win ↔ m ≥ k^2 + 1)) := by
  sorry

end NUMINAMATH_CALUDE_pigeon_win_conditions_l1046_104648


namespace NUMINAMATH_CALUDE_tan_seven_pi_fourths_l1046_104608

theorem tan_seven_pi_fourths : Real.tan (7 * π / 4) = -1 := by
  sorry

end NUMINAMATH_CALUDE_tan_seven_pi_fourths_l1046_104608


namespace NUMINAMATH_CALUDE_same_walking_speed_l1046_104615

-- Define Jack's speed function
def jack_speed (x : ℝ) : ℝ := x^2 - 13*x - 48

-- Define Jill's distance function
def jill_distance (x : ℝ) : ℝ := x^2 - 5*x - 84

-- Define Jill's time function
def jill_time (x : ℝ) : ℝ := x + 8

theorem same_walking_speed : 
  ∃ x : ℝ, x > 0 ∧ 
    jack_speed x = jill_distance x / jill_time x ∧ 
    jack_speed x = 6 := by
  sorry

end NUMINAMATH_CALUDE_same_walking_speed_l1046_104615


namespace NUMINAMATH_CALUDE_marbles_remaining_l1046_104637

theorem marbles_remaining (total_marbles : ℕ) (total_bags : ℕ) (bags_removed : ℕ) : 
  total_marbles = 28 →
  total_bags = 4 →
  bags_removed = 1 →
  total_marbles % total_bags = 0 →
  (total_bags - bags_removed) * (total_marbles / total_bags) = 21 := by
  sorry

end NUMINAMATH_CALUDE_marbles_remaining_l1046_104637


namespace NUMINAMATH_CALUDE_binomial_expansion_theorem_l1046_104638

theorem binomial_expansion_theorem (n k : ℕ) (a b : ℝ) (h1 : n ≥ 2) (h2 : a ≠ 0) (h3 : b ≠ 0) (h4 : a = k * b) (h5 : k > 0) :
  (n * (a - b)^(n-1) * (-b) + n * (n-1) / 2 * (a - b)^(n-2) * (-b)^2 = 0) → n = 2 * k + 1 := by
sorry

end NUMINAMATH_CALUDE_binomial_expansion_theorem_l1046_104638


namespace NUMINAMATH_CALUDE_add_3031_minutes_to_initial_equals_final_l1046_104614

-- Define a structure for date and time
structure DateTime where
  year : ℕ
  month : ℕ
  day : ℕ
  hour : ℕ
  minute : ℕ

-- Define the function to add minutes to a DateTime
def addMinutes (dt : DateTime) (minutes : ℕ) : DateTime :=
  sorry

-- Define the initial and final DateTimes
def initialDateTime : DateTime :=
  { year := 2020, month := 12, day := 31, hour := 17, minute := 0 }

def finalDateTime : DateTime :=
  { year := 2021, month := 1, day := 2, hour := 19, minute := 31 }

-- Theorem to prove
theorem add_3031_minutes_to_initial_equals_final :
  addMinutes initialDateTime 3031 = finalDateTime :=
sorry

end NUMINAMATH_CALUDE_add_3031_minutes_to_initial_equals_final_l1046_104614


namespace NUMINAMATH_CALUDE_conic_sections_decomposition_decomposition_into_ellipse_and_hyperbola_l1046_104622

/-- The equation y^4 - 9x^4 = 3y^2 - 1 represents two conic sections -/
theorem conic_sections_decomposition (x y : ℝ) :
  y^4 - 9*x^4 = 3*y^2 - 1 ↔
  ((y^2 - 3/2 = 3*x^2 + Real.sqrt 5/2) ∨ (y^2 - 3/2 = -(3*x^2 + Real.sqrt 5/2))) :=
by sorry

/-- The first equation represents an ellipse -/
def is_ellipse (x y : ℝ) : Prop :=
  y^2 - 3/2 = 3*x^2 + Real.sqrt 5/2

/-- The second equation represents a hyperbola -/
def is_hyperbola (x y : ℝ) : Prop :=
  y^2 - 3/2 = -(3*x^2 + Real.sqrt 5/2)

/-- The original equation decomposes into an ellipse and a hyperbola -/
theorem decomposition_into_ellipse_and_hyperbola (x y : ℝ) :
  y^4 - 9*x^4 = 3*y^2 - 1 ↔ (is_ellipse x y ∨ is_hyperbola x y) :=
by sorry

end NUMINAMATH_CALUDE_conic_sections_decomposition_decomposition_into_ellipse_and_hyperbola_l1046_104622


namespace NUMINAMATH_CALUDE_squareable_numbers_l1046_104618

def isSquareable (n : ℕ) : Prop :=
  ∃ (p : Fin n → Fin n), Function.Bijective p ∧
    ∀ i : Fin n, ∃ k : ℕ, (p i).val + 1 + i.val = k^2

theorem squareable_numbers : 
  (¬ isSquareable 7) ∧ 
  (isSquareable 9) ∧ 
  (¬ isSquareable 11) ∧ 
  (isSquareable 15) :=
sorry

end NUMINAMATH_CALUDE_squareable_numbers_l1046_104618


namespace NUMINAMATH_CALUDE_brandon_skittles_l1046_104689

/-- 
Given Brandon's initial number of Skittles and the number of Skittles he loses,
prove that his final number of Skittles is equal to the difference between
the initial number and the number lost.
-/
theorem brandon_skittles (initial : ℕ) (lost : ℕ) :
  initial ≥ lost → initial - lost = initial - lost :=
by sorry

end NUMINAMATH_CALUDE_brandon_skittles_l1046_104689


namespace NUMINAMATH_CALUDE_cylinder_fill_cost_l1046_104697

/-- Represents a right circular cylinder -/
structure Cylinder where
  radius : ℝ
  height : ℝ

/-- The cost to fill a cylinder with gasoline -/
def fillCost (c : Cylinder) (price : ℝ) : ℝ := c.radius^2 * c.height * price

/-- The theorem statement -/
theorem cylinder_fill_cost 
  (canB canN : Cylinder) 
  (h_radius : canN.radius = 2 * canB.radius) 
  (h_height : canN.height = canB.height / 2) 
  (h_half_cost : fillCost { radius := canB.radius, height := canB.height / 2 } (8 / (π * canB.radius^2 * canB.height)) = 4) :
  fillCost canN (8 / (π * canB.radius^2 * canB.height)) = 16 := by
  sorry


end NUMINAMATH_CALUDE_cylinder_fill_cost_l1046_104697


namespace NUMINAMATH_CALUDE_marble_problem_l1046_104681

theorem marble_problem (a : ℕ) 
  (angela : ℕ) 
  (brian : ℕ) 
  (caden : ℕ) 
  (daryl : ℕ) 
  (h1 : angela = a) 
  (h2 : brian = 3 * a) 
  (h3 : caden = 2 * brian) 
  (h4 : daryl = 5 * caden) 
  (h5 : angela + brian + caden + daryl = 120) : 
  a = 3 := by
sorry

end NUMINAMATH_CALUDE_marble_problem_l1046_104681


namespace NUMINAMATH_CALUDE_hyperbola_iff_equation_l1046_104641

/-- Represents the condition for a hyperbola given a real number m -/
def is_hyperbola (m : ℝ) : Prop :=
  (m < -1) ∨ (-1 < m ∧ m < 1) ∨ (m > 2)

/-- The equation representing a potential hyperbola -/
def hyperbola_equation (m x y : ℝ) : Prop :=
  x^2 / (|m| - 1) - y^2 / (m - 2) = 1

/-- Theorem stating the equivalence between the hyperbola condition and the equation -/
theorem hyperbola_iff_equation (m : ℝ) :
  is_hyperbola m ↔ ∃ x y : ℝ, hyperbola_equation m x y :=
sorry

end NUMINAMATH_CALUDE_hyperbola_iff_equation_l1046_104641


namespace NUMINAMATH_CALUDE_journey_distance_l1046_104655

/-- The total distance of a journey is the sum of miles driven and miles remaining. -/
theorem journey_distance (miles_driven miles_remaining : ℕ) 
  (h1 : miles_driven = 923)
  (h2 : miles_remaining = 277) :
  miles_driven + miles_remaining = 1200 := by
sorry

end NUMINAMATH_CALUDE_journey_distance_l1046_104655


namespace NUMINAMATH_CALUDE_sum_of_specific_numbers_l1046_104657

theorem sum_of_specific_numbers : 
  15.58 + 21.32 + 642.51 + 51.51 = 730.92 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_specific_numbers_l1046_104657


namespace NUMINAMATH_CALUDE_jellybean_problem_l1046_104642

theorem jellybean_problem (initial_quantity : ℝ) : 
  (0.75^3 * initial_quantity = 27) → initial_quantity = 64 := by
  sorry

end NUMINAMATH_CALUDE_jellybean_problem_l1046_104642


namespace NUMINAMATH_CALUDE_area_between_squares_l1046_104680

/-- The area of the region inside a large square but outside a smaller square -/
theorem area_between_squares (large_side : ℝ) (small_side : ℝ) 
  (h_large : large_side = 10)
  (h_small : small_side = 4)
  (h_placement : ∃ (x y : ℝ), x ^ 2 + y ^ 2 = (large_side / 2) ^ 2 ∧ 
                 0 ≤ x ∧ x ≤ small_side ∧ 0 ≤ y ∧ y ≤ small_side) :
  large_side ^ 2 - small_side ^ 2 = 84 := by
sorry

end NUMINAMATH_CALUDE_area_between_squares_l1046_104680


namespace NUMINAMATH_CALUDE_opposites_and_reciprocals_l1046_104652

theorem opposites_and_reciprocals (a b c d : ℝ) 
  (h1 : a = -b) -- a and b are opposites
  (h2 : c * d = 1) -- c and d are reciprocals
  : 3 * (a + b) - 4 * c * d = -4 := by
  sorry

end NUMINAMATH_CALUDE_opposites_and_reciprocals_l1046_104652


namespace NUMINAMATH_CALUDE_algebraic_expression_value_l1046_104672

theorem algebraic_expression_value (a b : ℝ) : 
  (a * 1^3 + b * 1 + 1 = 5) → (a * (-1)^3 + b * (-1) + 1 = -3) := by
  sorry

end NUMINAMATH_CALUDE_algebraic_expression_value_l1046_104672


namespace NUMINAMATH_CALUDE_ticket_cost_count_l1046_104674

def ticket_cost_possibilities (total_11th : ℕ) (total_12th : ℕ) : ℕ :=
  (Finset.filter (fun x : ℕ => 
    x > 0 ∧ 
    total_11th % x = 0 ∧ 
    total_12th % x = 0 ∧ 
    total_11th / x < total_12th / x) 
    (Finset.range (min total_11th total_12th + 1))).card

theorem ticket_cost_count : ticket_cost_possibilities 108 90 = 6 := by
  sorry

end NUMINAMATH_CALUDE_ticket_cost_count_l1046_104674


namespace NUMINAMATH_CALUDE_factorization_of_3x_squared_minus_12_l1046_104667

theorem factorization_of_3x_squared_minus_12 (x : ℝ) :
  3 * x^2 - 12 = 3 * (x + 2) * (x - 2) := by
  sorry

end NUMINAMATH_CALUDE_factorization_of_3x_squared_minus_12_l1046_104667


namespace NUMINAMATH_CALUDE_x_axis_intercept_l1046_104610

/-- The x-axis intercept of the line x + 2y + 1 = 0 is -1. -/
theorem x_axis_intercept :
  ∃ (x : ℝ), x + 2 * 0 + 1 = 0 ∧ x = -1 := by
  sorry

end NUMINAMATH_CALUDE_x_axis_intercept_l1046_104610


namespace NUMINAMATH_CALUDE_complex_expression_simplification_l1046_104607

theorem complex_expression_simplification :
  (7 - 3 * Complex.I) - (2 - 5 * Complex.I) - (3 + 2 * Complex.I) = (2 : ℂ) := by
  sorry

end NUMINAMATH_CALUDE_complex_expression_simplification_l1046_104607


namespace NUMINAMATH_CALUDE_line_through_points_eq_target_l1046_104698

/-- The equation of a line passing through two points -/
def line_equation (x1 y1 x2 y2 : ℝ) (x y : ℝ) : Prop :=
  (y - y1) * (x2 - x1) = (x - x1) * (y2 - y1)

/-- The specific points given in the problem -/
def point1 : ℝ × ℝ := (-1, 0)
def point2 : ℝ × ℝ := (0, 1)

/-- The equation we want to prove -/
def target_equation (x y : ℝ) : Prop := x - y + 1 = 0

/-- Theorem stating that the line equation passing through the given points
    is equivalent to the target equation -/
theorem line_through_points_eq_target :
  ∀ x y : ℝ, line_equation point1.1 point1.2 point2.1 point2.2 x y ↔ target_equation x y :=
by sorry

end NUMINAMATH_CALUDE_line_through_points_eq_target_l1046_104698


namespace NUMINAMATH_CALUDE_max_books_purchasable_l1046_104636

theorem max_books_purchasable (book_price : ℚ) (budget : ℚ) : 
  book_price = 15 → budget = 200 → 
    ↑(⌊budget / book_price⌋) = (13 : ℤ) := by
  sorry

end NUMINAMATH_CALUDE_max_books_purchasable_l1046_104636


namespace NUMINAMATH_CALUDE_cos_alpha_for_point_l1046_104613

/-- If the terminal side of angle α passes through the point (-1, 6), then cos α = -√37/37 -/
theorem cos_alpha_for_point (α : Real) : 
  (∃ (t : Real), t > 0 ∧ t * Real.cos α = -1 ∧ t * Real.sin α = 6) →
  Real.cos α = -Real.sqrt 37 / 37 := by
sorry

end NUMINAMATH_CALUDE_cos_alpha_for_point_l1046_104613


namespace NUMINAMATH_CALUDE_max_area_at_two_l1046_104664

open Real

noncomputable def tangentArea (m : ℝ) : ℝ :=
  if 1 ≤ m ∧ m ≤ 2 then
    4 * (4 - m) * (exp m)
  else if 2 < m ∧ m ≤ 5 then
    8 * (exp m)
  else
    0

theorem max_area_at_two :
  ∀ m : ℝ, 1 ≤ m ∧ m ≤ 5 → tangentArea m ≤ tangentArea 2 := by
  sorry

#check max_area_at_two

end NUMINAMATH_CALUDE_max_area_at_two_l1046_104664


namespace NUMINAMATH_CALUDE_matrix_M_property_l1046_104603

def matrix_M (x : ℝ × ℝ) : ℝ × ℝ := sorry

theorem matrix_M_property (M : ℝ × ℝ → ℝ × ℝ) 
  (h1 : M (2, -1) = (3, 0))
  (h2 : M (-3, 5) = (-1, -1)) :
  M (5, 1) = (11, -1) := by sorry

end NUMINAMATH_CALUDE_matrix_M_property_l1046_104603


namespace NUMINAMATH_CALUDE_pulley_distance_l1046_104630

theorem pulley_distance (r₁ r₂ t d : ℝ) 
  (hr₁ : r₁ = 14)
  (hr₂ : r₂ = 4)
  (ht : t = 24)
  (hd : d = Real.sqrt ((r₁ - r₂)^2 + t^2)) :
  d = 26 := by
  sorry

end NUMINAMATH_CALUDE_pulley_distance_l1046_104630


namespace NUMINAMATH_CALUDE_tree_planting_schedule_l1046_104687

theorem tree_planting_schedule (total_trees : ℕ) (days_saved : ℕ) : 
  total_trees = 960 →
  days_saved = 4 →
  ∃ (original_plan : ℕ),
    original_plan = 120 ∧
    (total_trees / original_plan) - (total_trees / (2 * original_plan)) = days_saved :=
by
  sorry

end NUMINAMATH_CALUDE_tree_planting_schedule_l1046_104687


namespace NUMINAMATH_CALUDE_job_completion_time_l1046_104644

/-- Represents the time (in minutes) it takes to complete a job when working together,
    given the individual completion times of two workers. -/
def time_working_together (sylvia_time carla_time : ℚ) : ℚ :=
  1 / (1 / sylvia_time + 1 / carla_time)

/-- Theorem stating that if Sylvia takes 45 minutes and Carla takes 30 minutes to complete a job individually,
    then together they will complete the job in 18 minutes. -/
theorem job_completion_time :
  time_working_together 45 30 = 18 := by
  sorry

#eval time_working_together 45 30

end NUMINAMATH_CALUDE_job_completion_time_l1046_104644


namespace NUMINAMATH_CALUDE_fifteenth_entry_is_29_l1046_104699

/-- r₁₁(m) is the remainder when m is divided by 11 -/
def r₁₁ (m : ℕ) : ℕ := m % 11

/-- List of nonnegative integers n that satisfy r₁₁(7n) ≤ 5 -/
def satisfying_list : List ℕ :=
  (List.range (100 : ℕ)).filter (fun n => r₁₁ (7 * n) ≤ 5)

theorem fifteenth_entry_is_29 : satisfying_list[14] = 29 := by
  sorry

end NUMINAMATH_CALUDE_fifteenth_entry_is_29_l1046_104699


namespace NUMINAMATH_CALUDE_tangent_line_triangle_area_l1046_104626

-- Define the function f(x) = x^3 - x + 1
def f (x : ℝ) : ℝ := x^3 - x + 1

-- Define the derivative of f
def f' (x : ℝ) : ℝ := 3 * x^2 - 1

-- Theorem statement
theorem tangent_line_triangle_area :
  let x₀ : ℝ := 0
  let y₀ : ℝ := f x₀
  let m : ℝ := f' x₀
  let tangent_line (x : ℝ) : ℝ := m * (x - x₀) + y₀
  let x_intercept : ℝ := -y₀ / m + x₀
  let y_intercept : ℝ := tangent_line 0
  (1/2) * x_intercept * y_intercept = 1/2 :=
sorry

end NUMINAMATH_CALUDE_tangent_line_triangle_area_l1046_104626


namespace NUMINAMATH_CALUDE_line_slope_and_inclination_l1046_104631

/-- Given a line l passing through points A(1,2) and B(4, 2+√3), 
    prove its slope and angle of inclination. -/
theorem line_slope_and_inclination :
  let A : ℝ × ℝ := (1, 2)
  let B : ℝ × ℝ := (4, 2 + Real.sqrt 3)
  let slope := (B.2 - A.2) / (B.1 - A.1)
  let angle := Real.arctan slope
  slope = Real.sqrt 3 / 3 ∧ angle = π / 6 := by
  sorry


end NUMINAMATH_CALUDE_line_slope_and_inclination_l1046_104631


namespace NUMINAMATH_CALUDE_ball_drawing_properties_l1046_104650

/-- Probability of drawing a red ball on the nth draw -/
def P (n : ℕ) : ℚ :=
  1/2 + 1/(2^(2*n + 1))

/-- Sum of the first n terms of the sequence P_n -/
def S (n : ℕ) : ℚ :=
  (1/6) * (3*n + 1 - (1/4)^n)

theorem ball_drawing_properties :
  (P 2 = 17/32) ∧
  (∀ n : ℕ, 4 * P (n+2) + P n = 5 * P (n+1)) ∧
  (∀ n : ℕ, S n = (1/6) * (3*n + 1 - (1/4)^n)) :=
by sorry

end NUMINAMATH_CALUDE_ball_drawing_properties_l1046_104650


namespace NUMINAMATH_CALUDE_dogs_and_bunnies_total_l1046_104623

/-- Represents the number of animals in a pet shop -/
structure PetShop where
  dogs : ℕ
  cats : ℕ
  bunnies : ℕ

/-- Defines the conditions of the pet shop problem -/
def pet_shop_problem (shop : PetShop) : Prop :=
  shop.dogs = 51 ∧
  shop.dogs * 5 = shop.cats * 3 ∧
  shop.dogs * 9 = shop.bunnies * 3

/-- Theorem stating the total number of dogs and bunnies in the pet shop -/
theorem dogs_and_bunnies_total (shop : PetShop) :
  pet_shop_problem shop → shop.dogs + shop.bunnies = 204 := by
  sorry


end NUMINAMATH_CALUDE_dogs_and_bunnies_total_l1046_104623


namespace NUMINAMATH_CALUDE_base8_653_equals_base10_427_l1046_104601

/-- Converts a number from base 8 to base 10 -/
def base8ToBase10 (n : Nat) : Nat :=
  let hundreds := n / 100
  let tens := (n / 10) % 10
  let ones := n % 10
  hundreds * 8^2 + tens * 8^1 + ones * 8^0

theorem base8_653_equals_base10_427 :
  base8ToBase10 653 = 427 := by
  sorry

end NUMINAMATH_CALUDE_base8_653_equals_base10_427_l1046_104601


namespace NUMINAMATH_CALUDE_bullets_shot_l1046_104693

/-- Proves that given 5 guys with 25 bullets each, if they all shoot an equal number of bullets
    and the remaining bullets equal the initial amount per person,
    then each person must have shot 20 bullets. -/
theorem bullets_shot (num_guys : Nat) (initial_bullets : Nat) (bullets_shot : Nat) : 
  num_guys = 5 →
  initial_bullets = 25 →
  (num_guys * initial_bullets) - (num_guys * bullets_shot) = initial_bullets →
  bullets_shot = 20 := by
  sorry

#check bullets_shot

end NUMINAMATH_CALUDE_bullets_shot_l1046_104693


namespace NUMINAMATH_CALUDE_two_and_half_in_one_and_three_fourths_l1046_104625

theorem two_and_half_in_one_and_three_fourths : 
  (1 + 3/4) / (2 + 1/2) = 7/10 := by sorry

end NUMINAMATH_CALUDE_two_and_half_in_one_and_three_fourths_l1046_104625


namespace NUMINAMATH_CALUDE_lines_do_not_form_triangle_l1046_104620

/-- Three lines in a 2D plane -/
structure ThreeLines where
  line1 : ℝ → ℝ → Prop
  line2 : ℝ → ℝ → ℝ → Prop
  line3 : ℝ → ℝ → ℝ → Prop

/-- The given three lines -/
def givenLines (m : ℝ) : ThreeLines :=
  { line1 := λ x y => 4 * x + y = 4
  , line2 := λ x y m => m * x + y = 0
  , line3 := λ x y m => 2 * x - 3 * m * y = 4 }

/-- Predicate to check if three lines form a triangle -/
def formsTriangle (lines : ThreeLines) : Prop := sorry

/-- The set of m values for which the lines do not form a triangle -/
def noTriangleValues : Set ℝ := {4, -1/6, -1, 2/3}

/-- Theorem stating the condition for the lines to not form a triangle -/
theorem lines_do_not_form_triangle (m : ℝ) :
  ¬(formsTriangle (givenLines m)) ↔ m ∈ noTriangleValues :=
sorry

end NUMINAMATH_CALUDE_lines_do_not_form_triangle_l1046_104620


namespace NUMINAMATH_CALUDE_distance_traveled_l1046_104609

/-- Calculates the distance traveled given speed and time -/
def distance (speed : ℝ) (time : ℝ) : ℝ := speed * time

/-- Theorem: Given a speed of 100 km/hr and a time of 5 hours, the distance traveled is 500 km -/
theorem distance_traveled (speed : ℝ) (time : ℝ) 
  (h_speed : speed = 100) 
  (h_time : time = 5) : 
  distance speed time = 500 := by
  sorry

end NUMINAMATH_CALUDE_distance_traveled_l1046_104609


namespace NUMINAMATH_CALUDE_beach_seashells_l1046_104600

/-- 
Given a person who spends 5 days at the beach and finds 7 seashells each day,
the total number of seashells found during the trip is 35.
-/
theorem beach_seashells : 
  ∀ (days : ℕ) (shells_per_day : ℕ),
  days = 5 → shells_per_day = 7 →
  days * shells_per_day = 35 := by
sorry

end NUMINAMATH_CALUDE_beach_seashells_l1046_104600


namespace NUMINAMATH_CALUDE_four_digit_repeat_count_l1046_104686

theorem four_digit_repeat_count : ∀ n : ℕ, (20 ≤ n ∧ n ≤ 99) → (Finset.range 100 \ Finset.range 20).card = 80 := by
  sorry

end NUMINAMATH_CALUDE_four_digit_repeat_count_l1046_104686


namespace NUMINAMATH_CALUDE_smallest_of_five_consecutive_sum_100_l1046_104639

theorem smallest_of_five_consecutive_sum_100 (n : ℕ) :
  (n + (n + 1) + (n + 2) + (n + 3) + (n + 4) = 100) → n = 18 := by
  sorry

end NUMINAMATH_CALUDE_smallest_of_five_consecutive_sum_100_l1046_104639


namespace NUMINAMATH_CALUDE_johns_next_birthday_age_l1046_104684

/-- Proves that John's age on his next birthday is 9, given the conditions of the problem -/
theorem johns_next_birthday_age (john carl beth : ℝ) 
  (h1 : john = 0.75 * carl)  -- John is 25% younger than Carl
  (h2 : carl = 1.3 * beth)   -- Carl is 30% older than Beth
  (h3 : john + carl + beth = 30) -- Sum of their ages is 30
  : ⌈john⌉ = 9 := by
  sorry

end NUMINAMATH_CALUDE_johns_next_birthday_age_l1046_104684


namespace NUMINAMATH_CALUDE_fraction_equality_l1046_104654

theorem fraction_equality (x : ℚ) : (5 + x) / (8 + x) = (2 + x) / (3 + x) ↔ x = -1/2 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_l1046_104654


namespace NUMINAMATH_CALUDE_projection_of_sum_onto_a_l1046_104656

def a : ℝ × ℝ := (1, 1)
def b : ℝ × ℝ := (-2, 1)

theorem projection_of_sum_onto_a :
  let sum := (a.1 + b.1, a.2 + b.2)
  let dot_product := sum.1 * a.1 + sum.2 * a.2
  let magnitude_a := Real.sqrt (a.1^2 + a.2^2)
  dot_product / magnitude_a = Real.sqrt 2 / 2 := by sorry

end NUMINAMATH_CALUDE_projection_of_sum_onto_a_l1046_104656
