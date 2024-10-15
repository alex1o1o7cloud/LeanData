import Mathlib

namespace NUMINAMATH_CALUDE_max_min_values_of_f_l2573_257393

noncomputable def f (x : ℝ) : ℝ := Real.exp (2 * x^2 - 4 * x - 6)

theorem max_min_values_of_f :
  ∃ (max min : ℝ),
    (∀ x ∈ Set.Icc 0 3, f x ≤ max) ∧
    (∃ x ∈ Set.Icc 0 3, f x = max) ∧
    (∀ x ∈ Set.Icc 0 3, min ≤ f x) ∧
    (∃ x ∈ Set.Icc 0 3, f x = min) ∧
    max = 1 ∧
    min = 1 / Real.exp 8 :=
by sorry

end NUMINAMATH_CALUDE_max_min_values_of_f_l2573_257393


namespace NUMINAMATH_CALUDE_solution_set_theorem_range_of_k_theorem_l2573_257300

-- Define the function f
def f (x k : ℝ) : ℝ := |x + 1| + |2 - x| - k

-- Theorem 1: Solution set of f(x) < 0 when k = 4
theorem solution_set_theorem :
  {x : ℝ | f x 4 < 0} = Set.Ioo (-3/2) (5/2) := by sorry

-- Theorem 2: Range of k for f(x) ≥ √(k+3) for all x ∈ ℝ
theorem range_of_k_theorem :
  ∀ k : ℝ, (∀ x : ℝ, f x k ≥ Real.sqrt (k + 3)) ↔ k ∈ Set.Iic 1 := by sorry

end NUMINAMATH_CALUDE_solution_set_theorem_range_of_k_theorem_l2573_257300


namespace NUMINAMATH_CALUDE_circle_passes_through_points_l2573_257336

def point_A : ℝ × ℝ := (-2, 1)
def point_B : ℝ × ℝ := (9, 3)
def point_C : ℝ × ℝ := (1, 7)

def circle_equation (x y : ℝ) : Prop :=
  (x - 7/2)^2 + (y - 2)^2 = 125/4

theorem circle_passes_through_points :
  circle_equation point_A.1 point_A.2 ∧
  circle_equation point_B.1 point_B.2 ∧
  circle_equation point_C.1 point_C.2 :=
by sorry

end NUMINAMATH_CALUDE_circle_passes_through_points_l2573_257336


namespace NUMINAMATH_CALUDE_integer_sqrt_15_l2573_257385

theorem integer_sqrt_15 (a : ℝ) : 
  (∃ m n : ℤ, (a + Real.sqrt 15 = m) ∧ (1 / (a - Real.sqrt 15) = n)) →
  (a = 4 + Real.sqrt 15 ∨ a = -(4 + Real.sqrt 15)) :=
by sorry

end NUMINAMATH_CALUDE_integer_sqrt_15_l2573_257385


namespace NUMINAMATH_CALUDE_sqrt_two_irrational_l2573_257363

theorem sqrt_two_irrational : Irrational (Real.sqrt 2) := by
  sorry

end NUMINAMATH_CALUDE_sqrt_two_irrational_l2573_257363


namespace NUMINAMATH_CALUDE_brownie_pieces_fit_l2573_257342

/-- Represents the dimensions of a rectangle -/
structure Dimensions where
  length : ℕ
  width : ℕ

/-- Calculates the area of a rectangle given its dimensions -/
def area (d : Dimensions) : ℕ := d.length * d.width

/-- Represents the pan and brownie piece dimensions -/
def pan : Dimensions := ⟨24, 15⟩
def piece : Dimensions := ⟨3, 2⟩

/-- The number of brownie pieces that fit in the pan -/
def num_pieces : ℕ := area pan / area piece

theorem brownie_pieces_fit :
  num_pieces = 60 ∧
  area pan = num_pieces * area piece :=
sorry

end NUMINAMATH_CALUDE_brownie_pieces_fit_l2573_257342


namespace NUMINAMATH_CALUDE_a_plus_b_equals_10_l2573_257345

-- Define the logarithm function (base 10)
noncomputable def log10 (x : ℝ) : ℝ := Real.log x / Real.log 10

-- State the theorem
theorem a_plus_b_equals_10 (a b : ℝ) 
  (ha : a + log10 a = 10) 
  (hb : b + 10^b = 10) : 
  a + b = 10 := by sorry

end NUMINAMATH_CALUDE_a_plus_b_equals_10_l2573_257345


namespace NUMINAMATH_CALUDE_travel_theorem_l2573_257346

/-- Represents the speeds of Butch, Sundance, and Sparky in miles per hour -/
structure Speeds where
  butch : ℝ
  sundance : ℝ
  sparky : ℝ

/-- Represents the distance traveled and time taken -/
structure TravelData where
  distance : ℕ  -- in miles
  time : ℕ      -- in minutes

/-- The main theorem representing the problem -/
theorem travel_theorem (speeds : Speeds) (h1 : speeds.butch = 4)
    (h2 : speeds.sundance = 2.5) (h3 : speeds.sparky = 6) : 
    ∃ (data : TravelData), data.distance = 19 ∧ data.time = 330 ∧ 
    data.distance + data.time = 349 := by
  sorry

#check travel_theorem

end NUMINAMATH_CALUDE_travel_theorem_l2573_257346


namespace NUMINAMATH_CALUDE_asha_granny_gift_l2573_257332

/-- The amount of money Asha was gifted by her granny --/
def granny_gift (brother_loan mother_loan father_loan savings spent_fraction remaining : ℚ) : ℚ :=
  (remaining / (1 - spent_fraction)) - (brother_loan + mother_loan + father_loan + savings)

/-- Theorem stating the amount gifted by Asha's granny --/
theorem asha_granny_gift :
  granny_gift 20 30 40 100 (3/4) 65 = 70 := by sorry

end NUMINAMATH_CALUDE_asha_granny_gift_l2573_257332


namespace NUMINAMATH_CALUDE_cubic_equation_roots_l2573_257371

theorem cubic_equation_roots (a b : ℝ) : 
  (∀ x : ℝ, 2*x^3 + a*x^2 - 13*x + b = 0 ↔ x = 2 ∨ x = -3 ∨ (∃ r : ℝ, x = r ∧ 2*(2-r)*(3+r) = 0)) →
  a = 1 ∧ b = 6 := by
sorry

end NUMINAMATH_CALUDE_cubic_equation_roots_l2573_257371


namespace NUMINAMATH_CALUDE_merchant_articles_l2573_257310

/-- The number of articles a merchant has, given profit percentage and price relationship -/
theorem merchant_articles (N : ℕ) (profit_percentage : ℚ) : 
  profit_percentage = 25 / 400 →
  (N : ℚ) * (1 : ℚ) = 16 * (1 + profit_percentage) →
  N = 17 := by
  sorry

#check merchant_articles

end NUMINAMATH_CALUDE_merchant_articles_l2573_257310


namespace NUMINAMATH_CALUDE_equation_solution_l2573_257321

theorem equation_solution : ∃ x : ℝ, 4*x + 6*x = 360 - 10*(x - 4) ∧ x = 20 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l2573_257321


namespace NUMINAMATH_CALUDE_solution_set_f_non_empty_solution_set_l2573_257379

-- Define the functions f and g
def f (x : ℝ) : ℝ := |x - 2|
def g (m : ℝ) (x : ℝ) : ℝ := -|x + 7| + 3*m

-- Theorem 1: Solution set of f(x) + x^2 - 4 > 0
theorem solution_set_f (x : ℝ) : f x + x^2 - 4 > 0 ↔ x > 2 ∨ x < -1 := by sorry

-- Theorem 2: Condition for non-empty solution set of f(x) < g(x)
theorem non_empty_solution_set (m : ℝ) :
  (∃ x : ℝ, f x < g m x) ↔ m > 3 := by sorry

end NUMINAMATH_CALUDE_solution_set_f_non_empty_solution_set_l2573_257379


namespace NUMINAMATH_CALUDE_steve_monday_pounds_l2573_257389

/-- The amount of money Steve wants to make in total -/
def total_money : ℕ := 100

/-- The pay rate per pound of lingonberries -/
def pay_rate : ℕ := 2

/-- The number of pounds Steve picked on Thursday -/
def thursday_pounds : ℕ := 18

/-- The factor by which Tuesday's harvest was greater than Monday's -/
def tuesday_factor : ℕ := 3

theorem steve_monday_pounds : 
  ∃ (monday_pounds : ℕ), 
    monday_pounds + tuesday_factor * monday_pounds + thursday_pounds = total_money / pay_rate ∧ 
    monday_pounds = 8 := by
  sorry

end NUMINAMATH_CALUDE_steve_monday_pounds_l2573_257389


namespace NUMINAMATH_CALUDE_circle_through_M_same_center_as_C_l2573_257380

-- Define the given circle C
def circle_C (x y : ℝ) : Prop := x^2 + y^2 + 2*x + 4*y - 11 = 0

-- Define the point M
def point_M : ℝ × ℝ := (1, 1)

-- Define the equation of the circle we want to prove
def circle_equation (x y : ℝ) : Prop := (x + 1)^2 + (y + 2)^2 = 13

-- State the theorem
theorem circle_through_M_same_center_as_C :
  ∀ (x y : ℝ),
  (∃ (h k r : ℝ), ∀ (u v : ℝ), circle_C u v ↔ (u - h)^2 + (v - k)^2 = r^2) →
  circle_equation point_M.1 point_M.2 ∧
  (∀ (u v : ℝ), circle_C u v ↔ circle_equation u v) :=
sorry

end NUMINAMATH_CALUDE_circle_through_M_same_center_as_C_l2573_257380


namespace NUMINAMATH_CALUDE_book_loss_percentage_l2573_257303

/-- Given that the cost price of 5 books equals the selling price of 20 books,
    prove that the loss percentage is 75%. -/
theorem book_loss_percentage : ∀ (C S : ℝ), 
  C > 0 → S > 0 →  -- Ensure positive prices
  5 * C = 20 * S →  -- Given condition
  (C - S) / C * 100 = 75 := by  -- Loss percentage formula
  sorry

end NUMINAMATH_CALUDE_book_loss_percentage_l2573_257303


namespace NUMINAMATH_CALUDE_geometric_sequence_first_term_l2573_257377

theorem geometric_sequence_first_term 
  (a : ℕ → ℝ) 
  (q : ℝ) 
  (h_positive : q > 0) 
  (h_geometric : ∀ n, a (n + 1) = a n * q) 
  (h_condition : a 3 * a 9 = 2 * (a 5)^2) 
  (h_second_term : a 2 = 1) : 
  a 1 = Real.sqrt 2 / 2 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_first_term_l2573_257377


namespace NUMINAMATH_CALUDE_area_of_triangle_ABC_l2573_257369

/-- The area of triangle ABC given the total area of small triangles and subtracted areas. -/
theorem area_of_triangle_ABC (total_area : ℝ) (subtracted_area : ℝ) 
  (h1 : total_area = 24)
  (h2 : subtracted_area = 14) :
  total_area - subtracted_area = 10 := by
  sorry

end NUMINAMATH_CALUDE_area_of_triangle_ABC_l2573_257369


namespace NUMINAMATH_CALUDE_original_price_correct_l2573_257313

/-- The original selling price of a shirt before discount -/
def original_price : ℝ := 700

/-- The discount percentage offered by the shop -/
def discount_percentage : ℝ := 20

/-- The price Smith paid for the shirt after discount -/
def discounted_price : ℝ := 560

/-- Theorem stating that the original price is correct given the discount and final price -/
theorem original_price_correct : 
  original_price * (1 - discount_percentage / 100) = discounted_price :=
by sorry

end NUMINAMATH_CALUDE_original_price_correct_l2573_257313


namespace NUMINAMATH_CALUDE_solution_set_is_closed_interval_l2573_257391

-- Define the function representing the left side of the inequality
def f (x : ℝ) : ℝ := -x^2 + 2*x + 3

-- Define the solution set
def solution_set : Set ℝ := {x | f x ≥ 0}

-- Theorem statement
theorem solution_set_is_closed_interval :
  solution_set = Set.Icc (-1) 3 := by sorry

end NUMINAMATH_CALUDE_solution_set_is_closed_interval_l2573_257391


namespace NUMINAMATH_CALUDE_no_real_roots_implies_not_first_quadrant_l2573_257304

theorem no_real_roots_implies_not_first_quadrant (m : ℝ) :
  (∀ x : ℝ, x^2 - 2*x - m ≠ 0) →
  ∀ x y : ℝ, y = (m + 1) * x + (m - 1) → (x > 0 ∧ y > 0 → False) :=
by sorry

end NUMINAMATH_CALUDE_no_real_roots_implies_not_first_quadrant_l2573_257304


namespace NUMINAMATH_CALUDE_arithmetic_sequence_properties_l2573_257312

/-- An arithmetic sequence with specified properties -/
structure ArithmeticSequence where
  a : ℕ → ℝ  -- The sequence
  S : ℕ → ℝ  -- The sum sequence
  is_arithmetic : ∀ n : ℕ, a (n + 1) - a n = a (n + 2) - a (n + 1)
  sum_property : ∀ n : ℕ, S n = (n : ℝ) * (a 1 + a n) / 2
  a4_eq_10 : a 4 = 10
  S6_eq_S3_plus_39 : S 6 = S 3 + 39

/-- The theorem stating the properties of the arithmetic sequence -/
theorem arithmetic_sequence_properties (seq : ArithmeticSequence) :
  seq.a 1 = 1 ∧ ∀ n : ℕ, seq.a n = 3 * n - 2 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_properties_l2573_257312


namespace NUMINAMATH_CALUDE_unique_solution_condition_l2573_257348

theorem unique_solution_condition (a b : ℝ) : 
  (∃! x : ℝ, 4 * x - 6 + a = (b + 1) * x + 2) ↔ b ≠ 3 := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_condition_l2573_257348


namespace NUMINAMATH_CALUDE_intersection_implies_a_value_l2573_257316

def A (a : ℝ) : Set ℝ := {2, 4, a^3 - 2*a^2 - a + 7}

def B (a : ℝ) : Set ℝ := {-4, a + 3, a^2 - 2*a + 2, a^3 + a^2 + 3*a + 7}

theorem intersection_implies_a_value (a : ℝ) :
  A a ∩ B a = {2, 5} → a = 2 := by
  sorry

end NUMINAMATH_CALUDE_intersection_implies_a_value_l2573_257316


namespace NUMINAMATH_CALUDE_oreo_multiple_l2573_257358

theorem oreo_multiple (total : Nat) (jordan : Nat) (m : Nat) : 
  total = 36 → jordan = 11 → jordan + (jordan * m + 3) = total → m = 2 := by
  sorry

end NUMINAMATH_CALUDE_oreo_multiple_l2573_257358


namespace NUMINAMATH_CALUDE_five_balls_four_boxes_l2573_257372

/-- The number of ways to distribute indistinguishable balls into distinguishable boxes -/
def distribute_balls (balls : ℕ) (boxes : ℕ) : ℕ :=
  sorry

/-- Theorem: There are 56 ways to distribute 5 indistinguishable balls into 4 distinguishable boxes -/
theorem five_balls_four_boxes : distribute_balls 5 4 = 56 := by
  sorry

end NUMINAMATH_CALUDE_five_balls_four_boxes_l2573_257372


namespace NUMINAMATH_CALUDE_angle_ABC_measure_l2573_257305

theorem angle_ABC_measure (angle_CBD angle_ABD angle_sum : ℝ) : 
  angle_CBD = 90 → angle_ABD = 60 → angle_sum = 200 → 
  ∃ (angle_ABC : ℝ), angle_ABC = 50 ∧ angle_ABC + angle_ABD + angle_CBD = angle_sum :=
by sorry

end NUMINAMATH_CALUDE_angle_ABC_measure_l2573_257305


namespace NUMINAMATH_CALUDE_l_shape_area_is_55_l2573_257354

/-- The area of an "L" shape formed by cutting a smaller rectangle from a larger rectangle -/
def l_shape_area (large_width large_height small_width small_height : ℕ) : ℕ :=
  large_width * large_height - small_width * small_height

/-- Theorem: The area of the "L" shape is 55 square units -/
theorem l_shape_area_is_55 :
  l_shape_area 10 7 5 3 = 55 := by
  sorry

#eval l_shape_area 10 7 5 3

end NUMINAMATH_CALUDE_l_shape_area_is_55_l2573_257354


namespace NUMINAMATH_CALUDE_employed_females_percentage_l2573_257306

theorem employed_females_percentage (population : ℝ) 
  (h1 : population > 0)
  (employed : ℝ) 
  (h2 : employed = 0.6 * population)
  (employed_males : ℝ) 
  (h3 : employed_males = 0.42 * population) :
  (employed - employed_males) / employed = 0.3 := by
sorry

end NUMINAMATH_CALUDE_employed_females_percentage_l2573_257306


namespace NUMINAMATH_CALUDE_constant_term_g_l2573_257308

-- Define polynomials f, g, and h
variable (f g h : ℝ[X])

-- Define the conditions
axiom h_def : h = f * g
axiom f_constant : f.coeff 0 = 5
axiom h_constant : h.coeff 0 = -10
axiom g_quadratic : g.degree ≤ 2

-- Theorem to prove
theorem constant_term_g : g.coeff 0 = -2 := by sorry

end NUMINAMATH_CALUDE_constant_term_g_l2573_257308


namespace NUMINAMATH_CALUDE_water_pumped_in_30_min_l2573_257330

/-- 
Given a pump that operates at a rate of 540 gallons per hour, 
this theorem proves that the volume of water pumped in 30 minutes is 270 gallons.
-/
theorem water_pumped_in_30_min (pump_rate : ℝ) (time : ℝ) : 
  pump_rate = 540 → time = 0.5 → pump_rate * time = 270 := by
  sorry

#check water_pumped_in_30_min

end NUMINAMATH_CALUDE_water_pumped_in_30_min_l2573_257330


namespace NUMINAMATH_CALUDE_min_n_plus_d_l2573_257364

/-- An arithmetic sequence with positive integer terms -/
structure ArithmeticSequence where
  n : ℕ+  -- number of terms
  d : ℕ+  -- common difference
  first_term : ℕ+ := 1  -- first term
  last_term : ℕ+ := 51  -- last term

/-- The property that the sequence follows the arithmetic sequence formula -/
def is_valid (seq : ArithmeticSequence) : Prop :=
  seq.first_term + (seq.n - 1) * seq.d = seq.last_term

/-- The theorem stating the minimum value of n + d -/
theorem min_n_plus_d (seq : ArithmeticSequence) (h : is_valid seq) : 
  (∀ seq' : ArithmeticSequence, is_valid seq' → seq.n + seq.d ≤ seq'.n + seq'.d) → 
  seq.n + seq.d = 16 := by
  sorry

end NUMINAMATH_CALUDE_min_n_plus_d_l2573_257364


namespace NUMINAMATH_CALUDE_map_coloring_theorem_l2573_257362

/-- Represents a map with regions -/
structure Map where
  regions : Nat
  adjacency : List (Nat × Nat)

/-- The minimum number of colors needed to color a map -/
def minColors (m : Map) : Nat :=
  sorry

/-- Theorem: The minimum number of colors needed for a 26-region map is 4 -/
theorem map_coloring_theorem (m : Map) (h1 : m.regions = 26) 
  (h2 : ∀ (i j : Nat), (i, j) ∈ m.adjacency → i ≠ j) 
  (h3 : minColors m > 3) : 
  minColors m = 4 :=
sorry

end NUMINAMATH_CALUDE_map_coloring_theorem_l2573_257362


namespace NUMINAMATH_CALUDE_sequence_first_term_l2573_257398

theorem sequence_first_term (a : ℕ → ℚ) :
  (∀ n, a (n + 1) = 1 / (1 - a n)) →
  a 2 = 2 →
  a 1 = 1/2 := by
sorry

end NUMINAMATH_CALUDE_sequence_first_term_l2573_257398


namespace NUMINAMATH_CALUDE_prob_odd_divisor_15_factorial_l2573_257326

/-- The factorial function -/
def factorial (n : ℕ) : ℕ := (List.range n).foldl (· * ·) 1

/-- The number of divisors of n -/
def numDivisors (n : ℕ) : ℕ := sorry

/-- The number of odd divisors of n -/
def numOddDivisors (n : ℕ) : ℕ := sorry

/-- The probability of a randomly chosen divisor of n being odd -/
def probOddDivisor (n : ℕ) : ℚ :=
  (numOddDivisors n : ℚ) / (numDivisors n : ℚ)

theorem prob_odd_divisor_15_factorial :
  probOddDivisor (factorial 15) = 1 / 12 := by sorry

end NUMINAMATH_CALUDE_prob_odd_divisor_15_factorial_l2573_257326


namespace NUMINAMATH_CALUDE_similar_walls_length_l2573_257357

/-- Represents the work done to build a wall -/
structure WallWork where
  persons : ℕ
  days : ℕ
  length : ℝ

/-- The theorem stating the relationship between two similar walls -/
theorem similar_walls_length
  (wall1 : WallWork)
  (wall2 : WallWork)
  (h1 : wall1.persons = 18)
  (h2 : wall1.days = 42)
  (h3 : wall2.persons = 30)
  (h4 : wall2.days = 18)
  (h5 : wall2.length = 100)
  (h6 : (wall1.persons * wall1.days) / (wall2.persons * wall2.days) = wall1.length / wall2.length) :
  wall1.length = 140 := by
  sorry

#check similar_walls_length

end NUMINAMATH_CALUDE_similar_walls_length_l2573_257357


namespace NUMINAMATH_CALUDE_min_value_product_min_value_product_achieved_l2573_257387

theorem min_value_product (x y z u : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) (hu : u > 0)
  (h : 1/x + 1/y + 1/z + 1/u = 8) : 
  x^3 * y^2 * z * u^2 ≥ 1/432 :=
sorry

theorem min_value_product_achieved (x y z u : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) (hu : u > 0)
  (h : 1/x + 1/y + 1/z + 1/u = 8) : 
  ∃ (x' y' z' u' : ℝ), x' > 0 ∧ y' > 0 ∧ z' > 0 ∧ u' > 0 ∧ 
    1/x' + 1/y' + 1/z' + 1/u' = 8 ∧ 
    x'^3 * y'^2 * z' * u'^2 = 1/432 :=
sorry

end NUMINAMATH_CALUDE_min_value_product_min_value_product_achieved_l2573_257387


namespace NUMINAMATH_CALUDE_binomial_14_11_l2573_257317

theorem binomial_14_11 : Nat.choose 14 11 = 364 := by
  sorry

end NUMINAMATH_CALUDE_binomial_14_11_l2573_257317


namespace NUMINAMATH_CALUDE_digit_distribution_exists_l2573_257341

theorem digit_distribution_exists : ∃ n : ℕ, 
  n > 0 ∧ 
  n % 2 = 0 ∧ 
  n % 5 = 0 ∧ 
  n % 10 = 0 ∧ 
  n / 2 + 2 * (n / 5) + n / 10 = n :=
sorry

end NUMINAMATH_CALUDE_digit_distribution_exists_l2573_257341


namespace NUMINAMATH_CALUDE_cylinder_height_l2573_257350

/-- Represents a right cylinder with given dimensions -/
structure RightCylinder where
  radius : ℝ
  height : ℝ
  lateralSurfaceArea : ℝ
  endArea : ℝ

/-- Theorem stating the height of a specific cylinder -/
theorem cylinder_height (c : RightCylinder) 
  (h_radius : c.radius = 2)
  (h_lsa : c.lateralSurfaceArea = 16 * Real.pi)
  (h_ea : c.endArea = 8 * Real.pi) :
  c.height = 4 := by
  sorry

end NUMINAMATH_CALUDE_cylinder_height_l2573_257350


namespace NUMINAMATH_CALUDE_tesseract_triangles_l2573_257319

/-- The number of vertices in a tesseract -/
def tesseract_vertices : ℕ := 16

/-- The number of vertices required to form a triangle -/
def triangle_vertices : ℕ := 3

/-- The number of distinct triangles in a tesseract -/
def distinct_triangles : ℕ := Nat.choose tesseract_vertices triangle_vertices

theorem tesseract_triangles : distinct_triangles = 560 := by
  sorry

end NUMINAMATH_CALUDE_tesseract_triangles_l2573_257319


namespace NUMINAMATH_CALUDE_partial_fraction_decomposition_l2573_257324

theorem partial_fraction_decomposition (C D : ℚ) :
  (∀ x : ℚ, x ≠ 11 ∧ x ≠ -5 →
    (7 * x - 4) / (x^2 - 6 * x - 55) = C / (x - 11) + D / (x + 5)) →
  C = 73 / 16 ∧ D = 39 / 16 := by
sorry

end NUMINAMATH_CALUDE_partial_fraction_decomposition_l2573_257324


namespace NUMINAMATH_CALUDE_f_composition_value_l2573_257355

noncomputable def f (x : ℝ) : ℝ :=
  if x > 0 then Real.log x else Real.exp (x + 1) - 2

theorem f_composition_value : f (f (1 / Real.exp 1)) = -1 := by
  sorry

end NUMINAMATH_CALUDE_f_composition_value_l2573_257355


namespace NUMINAMATH_CALUDE_largest_number_with_conditions_l2573_257397

def is_valid_number (n : ℕ) : Prop :=
  ∀ d : ℕ, d ∈ n.digits 10 → d = 2 ∨ d = 3

def digit_sum (n : ℕ) : ℕ :=
  (n.digits 10).sum

theorem largest_number_with_conditions :
  ∀ n : ℕ,
    is_valid_number n ∧
    digit_sum n = 14 →
    n ≤ 333322 :=
by sorry

end NUMINAMATH_CALUDE_largest_number_with_conditions_l2573_257397


namespace NUMINAMATH_CALUDE_a_range_l2573_257374

-- Define the statements p and q
def p (x a : ℝ) : Prop := (x - a) * (x - 3 * a) < 0

def q (x : ℝ) : Prop := (x - 3) / (x - 2) ≤ 0

-- Theorem statement
theorem a_range (a : ℝ) : 
  (a > 0) → 
  (∀ x, q x → p x a) → 
  (∃ x, p x a ∧ ¬q x) → 
  1 ≤ a ∧ a ≤ 2 := by
  sorry

end NUMINAMATH_CALUDE_a_range_l2573_257374


namespace NUMINAMATH_CALUDE_problem_solution_l2573_257352

theorem problem_solution (x : ℝ) : ((x / 4) * 5 + 10 - 12 = 48) → x = 40 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l2573_257352


namespace NUMINAMATH_CALUDE_pie_chart_probability_l2573_257347

theorem pie_chart_probability (prob_D prob_E prob_F : ℚ) : 
  prob_D = 1/4 → prob_E = 1/3 → prob_D + prob_E + prob_F = 1 → prob_F = 5/12 := by
  sorry

end NUMINAMATH_CALUDE_pie_chart_probability_l2573_257347


namespace NUMINAMATH_CALUDE_possible_distances_l2573_257301

/-- Three points on a line -/
structure PointsOnLine where
  A : ℝ
  B : ℝ
  C : ℝ

/-- The distance between two points on a line -/
def distance (x y : ℝ) : ℝ := |x - y|

theorem possible_distances (p : PointsOnLine) 
  (h1 : distance p.A p.B = 1)
  (h2 : distance p.B p.C = 3) :
  distance p.A p.C = 4 ∨ distance p.A p.C = 2 := by
  sorry

end NUMINAMATH_CALUDE_possible_distances_l2573_257301


namespace NUMINAMATH_CALUDE_sqrt_of_sqrt_81_l2573_257382

theorem sqrt_of_sqrt_81 : Real.sqrt (Real.sqrt 81) = 9 := by sorry

end NUMINAMATH_CALUDE_sqrt_of_sqrt_81_l2573_257382


namespace NUMINAMATH_CALUDE_marlington_orchestra_max_members_l2573_257337

theorem marlington_orchestra_max_members :
  ∀ n : ℕ,
  (∃ k : ℕ, 30 * n = 31 * k + 5) →
  30 * n < 1500 →
  30 * n ≤ 780 :=
by sorry

end NUMINAMATH_CALUDE_marlington_orchestra_max_members_l2573_257337


namespace NUMINAMATH_CALUDE_sum_of_squares_and_products_l2573_257318

theorem sum_of_squares_and_products (a b c : ℝ) : 
  0 ≤ a ∧ 0 ≤ b ∧ 0 ≤ c → a^2 + b^2 + c^2 = 39 → a*b + b*c + c*a = 21 → a + b + c = 9 :=
by sorry

end NUMINAMATH_CALUDE_sum_of_squares_and_products_l2573_257318


namespace NUMINAMATH_CALUDE_unique_solution_modular_equation_l2573_257386

theorem unique_solution_modular_equation :
  ∃! n : ℕ, n < 103 ∧ (100 * n) % 103 = 65 % 103 ∧ n = 68 := by sorry

end NUMINAMATH_CALUDE_unique_solution_modular_equation_l2573_257386


namespace NUMINAMATH_CALUDE_bbq_guests_count_l2573_257325

/-- Represents the BBQ scenario with given parameters -/
structure BBQ where
  cook_time_per_side : ℕ  -- cooking time for one side of a burger in minutes
  grill_capacity : ℕ      -- number of burgers that can be cooked simultaneously
  total_cook_time : ℕ     -- total time spent cooking all burgers in minutes

/-- Calculates the number of guests at the BBQ -/
def number_of_guests (bbq : BBQ) : ℕ :=
  let total_burgers := (bbq.total_cook_time / (2 * bbq.cook_time_per_side)) * bbq.grill_capacity
  (2 * total_burgers) / 3

/-- Theorem stating that the number of guests at the BBQ is 30 -/
theorem bbq_guests_count (bbq : BBQ) 
  (h1 : bbq.cook_time_per_side = 4)
  (h2 : bbq.grill_capacity = 5)
  (h3 : bbq.total_cook_time = 72) :
  number_of_guests bbq = 30 := by
  sorry

#eval number_of_guests ⟨4, 5, 72⟩

end NUMINAMATH_CALUDE_bbq_guests_count_l2573_257325


namespace NUMINAMATH_CALUDE_salon_customers_l2573_257339

/-- The number of customers a salon has each day, given their hairspray usage and purchasing. -/
theorem salon_customers (total_cans : ℕ) (extra_cans : ℕ) (cans_per_customer : ℕ) : 
  total_cans = 33 →
  extra_cans = 5 →
  cans_per_customer = 2 →
  (total_cans - extra_cans) / cans_per_customer = 14 :=
by sorry

end NUMINAMATH_CALUDE_salon_customers_l2573_257339


namespace NUMINAMATH_CALUDE_mixed_fruit_juice_cost_l2573_257390

/-- The cost per litre of the superfruit juice cocktail -/
def superfruit_cost : ℝ := 1399.45

/-- The cost per litre of the açaí berry juice -/
def acai_cost : ℝ := 3104.35

/-- The volume of mixed fruit juice used -/
def mixed_fruit_volume : ℝ := 33

/-- The volume of açaí berry juice used -/
def acai_volume : ℝ := 22

/-- The cost per litre of the mixed fruit juice -/
def mixed_fruit_cost : ℝ := 256.79

theorem mixed_fruit_juice_cost : 
  mixed_fruit_volume * mixed_fruit_cost + acai_volume * acai_cost = 
  (mixed_fruit_volume + acai_volume) * superfruit_cost :=
by sorry

end NUMINAMATH_CALUDE_mixed_fruit_juice_cost_l2573_257390


namespace NUMINAMATH_CALUDE_prime_power_sum_l2573_257359

theorem prime_power_sum (w x y z : ℕ) : 
  2^w * 3^x * 5^y * 7^z = 13230 → 3*w + 2*x + 6*y + 4*z = 23 := by
  sorry

end NUMINAMATH_CALUDE_prime_power_sum_l2573_257359


namespace NUMINAMATH_CALUDE_cube_volume_problem_l2573_257394

theorem cube_volume_problem (a : ℝ) : 
  a > 0 → 
  (a - 2) * a * (a + 2) = a^3 - 16 → 
  a^3 = 64 := by
sorry

end NUMINAMATH_CALUDE_cube_volume_problem_l2573_257394


namespace NUMINAMATH_CALUDE_painter_problem_l2573_257309

/-- Calculates the total number of rooms to be painted given the painting time per room,
    number of rooms already painted, and remaining painting time. -/
def total_rooms_to_paint (time_per_room : ℕ) (rooms_painted : ℕ) (remaining_time : ℕ) : ℕ :=
  rooms_painted + remaining_time / time_per_room

/-- Proves that the total number of rooms to be painted is 10 given the specific conditions. -/
theorem painter_problem :
  let time_per_room : ℕ := 8
  let rooms_painted : ℕ := 8
  let remaining_time : ℕ := 16
  total_rooms_to_paint time_per_room rooms_painted remaining_time = 10 := by
  sorry

end NUMINAMATH_CALUDE_painter_problem_l2573_257309


namespace NUMINAMATH_CALUDE_classroom_pairing_probability_l2573_257307

/-- The probability of two specific students being paired in a classroom. -/
def pairProbability (n : ℕ) : ℚ :=
  1 / (n - 1)

/-- Theorem: In a classroom of 24 students where each student is randomly paired
    with another, the probability of a specific student being paired with
    another specific student is 1/23. -/
theorem classroom_pairing_probability :
  pairProbability 24 = 1 / 23 := by
  sorry

#eval pairProbability 24

end NUMINAMATH_CALUDE_classroom_pairing_probability_l2573_257307


namespace NUMINAMATH_CALUDE_altitude_sum_of_specific_triangle_l2573_257388

/-- The sum of altitudes of a triangle formed by the line 10x + 8y = 80 and coordinate axes --/
theorem altitude_sum_of_specific_triangle : 
  let line : ℝ → ℝ → Prop := λ x y => 10 * x + 8 * y = 80
  let triangle := {p : ℝ × ℝ | p.1 ≥ 0 ∧ p.2 ≥ 0 ∧ line p.1 p.2}
  let altitudes := 
    [10, 8, (40 : ℝ) / Real.sqrt 41]  -- altitudes to y-axis, x-axis, and hypotenuse
  (altitudes.sum : ℝ) = 18 + 40 / Real.sqrt 41 := by
  sorry

end NUMINAMATH_CALUDE_altitude_sum_of_specific_triangle_l2573_257388


namespace NUMINAMATH_CALUDE_sum_of_roots_l2573_257356

theorem sum_of_roots (p q : ℝ) : 
  (∀ x, x^2 - p*x + q = 0 ↔ (x = p ∨ x = q)) →
  2*p + 3*q = 6 →
  p + q = p :=
by sorry

end NUMINAMATH_CALUDE_sum_of_roots_l2573_257356


namespace NUMINAMATH_CALUDE_equation_solution_l2573_257331

theorem equation_solution (x : ℝ) : 3 - 1 / (2 - x) = 1 / (2 - x) → x = 4 / 3 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l2573_257331


namespace NUMINAMATH_CALUDE_age_difference_l2573_257353

/-- Given the ages of Patrick, Michael, and Monica satisfying certain ratios and sum, 
    prove that the difference between Monica's and Patrick's ages is 33 years. -/
theorem age_difference (patrick michael monica : ℕ) : 
  patrick * 5 = michael * 3 →  -- Patrick and Michael's ages are in ratio 3:5
  michael * 4 = monica * 3 →   -- Michael and Monica's ages are in ratio 3:4
  patrick + michael + monica = 132 →  -- Sum of their ages is 132
  monica - patrick = 33 := by  -- Difference between Monica's and Patrick's ages is 33
sorry  -- Proof omitted

end NUMINAMATH_CALUDE_age_difference_l2573_257353


namespace NUMINAMATH_CALUDE_correct_probability_open_l2573_257335

/-- Represents a three-digit combination lock -/
structure CombinationLock :=
  (digits : Fin 3 → Fin 10)

/-- The probability of opening the lock by randomly selecting the last digit -/
def probability_open (lock : CombinationLock) : ℚ :=
  1 / 10

theorem correct_probability_open (lock : CombinationLock) :
  probability_open lock = 1 / 10 := by
  sorry

end NUMINAMATH_CALUDE_correct_probability_open_l2573_257335


namespace NUMINAMATH_CALUDE_cosine_sum_specific_angles_l2573_257322

theorem cosine_sum_specific_angles : 
  Real.cos (π/3) * Real.cos (π/6) - Real.sin (π/3) * Real.sin (π/6) = 0 := by
  sorry

end NUMINAMATH_CALUDE_cosine_sum_specific_angles_l2573_257322


namespace NUMINAMATH_CALUDE_not_even_implies_exists_unequal_l2573_257340

-- Define a real-valued function on ℝ
variable (f : ℝ → ℝ)

-- Define what it means for f to be not even
def NotEven (f : ℝ → ℝ) : Prop :=
  ¬(∀ x : ℝ, f (-x) = f x)

-- Theorem statement
theorem not_even_implies_exists_unequal (f : ℝ → ℝ) :
  NotEven f → ∃ x₀ : ℝ, f (-x₀) ≠ f x₀ :=
by
  sorry

end NUMINAMATH_CALUDE_not_even_implies_exists_unequal_l2573_257340


namespace NUMINAMATH_CALUDE_all_numbers_equal_l2573_257378

/-- Represents a grid of positive integers -/
def Grid := ℕ → ℕ → ℕ+

/-- Checks if two polygons are congruent -/
def CongruentPolygons (p q : Set (ℕ × ℕ)) : Prop := sorry

/-- Calculates the sum of numbers in a polygon -/
def PolygonSum (g : Grid) (p : Set (ℕ × ℕ)) : ℕ := sorry

/-- Calculates the area of a polygon -/
def PolygonArea (p : Set (ℕ × ℕ)) : ℕ := sorry

/-- Main theorem -/
theorem all_numbers_equal (g : Grid) (n : ℕ) (h_n : n > 2) :
  (∀ p q : Set (ℕ × ℕ), CongruentPolygons p q → PolygonArea p = n → PolygonArea q = n →
    PolygonSum g p = PolygonSum g q) →
  ∀ i j k l : ℕ, g i j = g k l :=
sorry

end NUMINAMATH_CALUDE_all_numbers_equal_l2573_257378


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l2573_257360

def A : Set Int := {-2, -1}
def B : Set Int := {-1, 2, 3}

theorem intersection_of_A_and_B : A ∩ B = {-1} := by sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l2573_257360


namespace NUMINAMATH_CALUDE_perpendicular_unit_vector_to_BC_l2573_257349

def AB : ℝ × ℝ := (-1, 3)
def BC : ℝ → ℝ × ℝ := λ k => (3, k)
def CD : ℝ → ℝ × ℝ := λ k => (k, 2)

def AC (k : ℝ) : ℝ × ℝ := (AB.1 + (BC k).1, AB.2 + (BC k).2)

def parallel (v w : ℝ × ℝ) : Prop := v.1 * w.2 = v.2 * w.1

def perpendicular (v w : ℝ × ℝ) : Prop := v.1 * w.1 + v.2 * w.2 = 0

def is_unit_vector (v : ℝ × ℝ) : Prop := v.1^2 + v.2^2 = 1

theorem perpendicular_unit_vector_to_BC (k : ℝ) :
  parallel (AC k) (CD k) →
  ∃ v : ℝ × ℝ, perpendicular v (BC k) ∧ is_unit_vector v ∧
    (v = (Real.sqrt 10 / 10, -3 * Real.sqrt 10 / 10) ∨
     v = (-Real.sqrt 10 / 10, 3 * Real.sqrt 10 / 10)) :=
by sorry

end NUMINAMATH_CALUDE_perpendicular_unit_vector_to_BC_l2573_257349


namespace NUMINAMATH_CALUDE_intersection_not_empty_implies_a_value_l2573_257333

theorem intersection_not_empty_implies_a_value (a : ℤ) : 
  let M : Set ℤ := {a, 0}
  let N : Set ℤ := {x | 2 * x^2 - 5 * x < 0}
  (M ∩ N).Nonempty → a = 1 ∨ a = 2 := by
sorry

end NUMINAMATH_CALUDE_intersection_not_empty_implies_a_value_l2573_257333


namespace NUMINAMATH_CALUDE_vector_operation_l2573_257327

def vector_a : ℝ × ℝ := (2, -1)
def vector_b : ℝ × ℝ := (1, 3)

theorem vector_operation :
  (-2 : ℝ) • vector_a + (3 : ℝ) • vector_b = (-1, 11) := by sorry

end NUMINAMATH_CALUDE_vector_operation_l2573_257327


namespace NUMINAMATH_CALUDE_cosine_law_acute_triangle_condition_l2573_257323

-- Define a triangle
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ
  -- Assume the triangle is valid
  valid : a > 0 ∧ b > 0 ∧ c > 0 ∧ A > 0 ∧ B > 0 ∧ C > 0 ∧ A + B + C = π

-- Theorem 1: c = a cos B + b cos A
theorem cosine_law (t : Triangle) : t.c = t.a * Real.cos t.B + t.b * Real.cos t.A := by
  sorry

-- Theorem 2: If a³ + b³ = c³, then a² + b² > c²
theorem acute_triangle_condition (t : Triangle) : 
  t.a^3 + t.b^3 = t.c^3 → t.a^2 + t.b^2 > t.c^2 := by
  sorry

end NUMINAMATH_CALUDE_cosine_law_acute_triangle_condition_l2573_257323


namespace NUMINAMATH_CALUDE_grade_distribution_equals_total_total_students_is_100_l2573_257367

-- Define the total number of students
def total_students : ℕ := 100

-- Define the number of students who received each grade
def a_students : ℕ := total_students / 5
def b_students : ℕ := total_students / 4
def c_students : ℕ := total_students / 2
def d_students : ℕ := 5

-- Theorem stating that the sum of students in each grade category equals the total number of students
theorem grade_distribution_equals_total :
  a_students + b_students + c_students + d_students = total_students := by
  sorry

-- Theorem proving that the total number of students is 100
theorem total_students_is_100 : total_students = 100 := by
  sorry

end NUMINAMATH_CALUDE_grade_distribution_equals_total_total_students_is_100_l2573_257367


namespace NUMINAMATH_CALUDE_iced_tea_price_l2573_257399

/-- The cost of a beverage order --/
def order_cost (cappuccino_price : ℚ) (latte_price : ℚ) (espresso_price : ℚ) (iced_tea_price : ℚ) : ℚ :=
  3 * cappuccino_price + 2 * iced_tea_price + 2 * latte_price + 2 * espresso_price

theorem iced_tea_price (cappuccino_price latte_price espresso_price : ℚ)
  (h1 : cappuccino_price = 2)
  (h2 : latte_price = 3/2)
  (h3 : espresso_price = 1)
  (h4 : ∃ (x : ℚ), order_cost cappuccino_price latte_price espresso_price x = 17) :
  ∃ (x : ℚ), x = 3 ∧ order_cost cappuccino_price latte_price espresso_price x = 17 :=
sorry

end NUMINAMATH_CALUDE_iced_tea_price_l2573_257399


namespace NUMINAMATH_CALUDE_max_students_per_classroom_l2573_257395

/-- Theorem: Maximum students per classroom with equal gender distribution -/
theorem max_students_per_classroom 
  (num_classrooms : ℕ) 
  (num_boys : ℕ) 
  (num_girls : ℕ) 
  (h1 : num_classrooms = 7)
  (h2 : num_boys = 68)
  (h3 : num_girls = 53) :
  ∃ (students_per_classroom : ℕ),
    students_per_classroom = 14 ∧
    students_per_classroom ≤ min num_boys num_girls ∧
    students_per_classroom % 2 = 0 ∧
    (students_per_classroom / 2) * num_classrooms ≤ min num_boys num_girls :=
by
  sorry

#check max_students_per_classroom

end NUMINAMATH_CALUDE_max_students_per_classroom_l2573_257395


namespace NUMINAMATH_CALUDE_probability_not_perfect_power_l2573_257320

/-- A number is a perfect power if it can be expressed as x^y where x and y are integers and y > 1 -/
def IsPerfectPower (n : ℕ) : Prop :=
  ∃ x y : ℕ, y > 1 ∧ n = x^y

/-- The count of numbers from 1 to 200 that are not perfect powers -/
def CountNotPerfectPower : ℕ := 179

theorem probability_not_perfect_power :
  (CountNotPerfectPower : ℚ) / 200 = 179 / 200 :=
sorry

end NUMINAMATH_CALUDE_probability_not_perfect_power_l2573_257320


namespace NUMINAMATH_CALUDE_solution_set_of_inequality_l2573_257365

noncomputable def f (x : ℝ) : ℝ := Real.exp (-x) - Real.exp x - 5 * x

theorem solution_set_of_inequality :
  {x : ℝ | f (x^2) + f (-x-6) < 0} = {x : ℝ | x < -2 ∨ x > 3} := by sorry

end NUMINAMATH_CALUDE_solution_set_of_inequality_l2573_257365


namespace NUMINAMATH_CALUDE_smallest_n_with_6474_l2573_257329

def concatenate (a b c : ℕ) : List ℕ :=
  (a.digits 10) ++ (b.digits 10) ++ (c.digits 10)

def contains_subseq (l : List ℕ) (s : List ℕ) : Prop :=
  ∃ i, l.drop i = s ++ l.drop (i + s.length)

theorem smallest_n_with_6474 :
  ∀ n : ℕ, n < 46 →
    ¬ (contains_subseq (concatenate n (n + 1) (n + 2)) [6, 4, 7, 4]) ∧
  contains_subseq (concatenate 46 47 48) [6, 4, 7, 4] :=
sorry

end NUMINAMATH_CALUDE_smallest_n_with_6474_l2573_257329


namespace NUMINAMATH_CALUDE_maria_water_bottles_l2573_257375

theorem maria_water_bottles (initial_bottles : ℝ) (sister_drank : ℝ) (bottles_left : ℝ) :
  initial_bottles = 45.0 →
  sister_drank = 8.0 →
  bottles_left = 23 →
  initial_bottles - sister_drank - bottles_left = 14.0 :=
by
  sorry

end NUMINAMATH_CALUDE_maria_water_bottles_l2573_257375


namespace NUMINAMATH_CALUDE_quadratic_roots_real_and_equal_l2573_257384

/-- The quadratic equation x^2 + 2x√3 + 3 = 0 has real and equal roots given that its discriminant is zero -/
theorem quadratic_roots_real_and_equal :
  let a : ℝ := 1
  let b : ℝ := 2 * Real.sqrt 3
  let c : ℝ := 3
  let discriminant := b^2 - 4*a*c
  discriminant = 0 →
  ∃! x : ℝ, a * x^2 + b * x + c = 0 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_roots_real_and_equal_l2573_257384


namespace NUMINAMATH_CALUDE_road_distance_ratio_l2573_257314

/-- Represents the distance between two cities --/
structure CityDistance where
  total : ℕ
  deriving Repr

/-- Represents a pole with distances to two cities --/
structure Pole where
  distanceA : ℕ
  distanceB : ℕ
  deriving Repr

/-- The configuration of poles between two cities --/
structure RoadConfiguration where
  distance : CityDistance
  pole1 : Pole
  pole2 : Pole
  pole3 : Pole
  deriving Repr

/-- The theorem to be proved --/
theorem road_distance_ratio 
  (config : RoadConfiguration) 
  (h1 : config.pole1.distanceB = 3 * config.pole1.distanceA)
  (h2 : config.pole2.distanceB = 3 * config.pole2.distanceA)
  (h3 : config.pole1.distanceA + config.pole1.distanceB = config.distance.total)
  (h4 : config.pole2.distanceA + config.pole2.distanceB = config.distance.total)
  (h5 : config.pole2.distanceA = config.pole1.distanceA + 40)
  (h6 : config.pole3.distanceA = config.pole2.distanceA + 10)
  (h7 : config.pole3.distanceB = config.pole2.distanceB - 10) :
  (max config.pole3.distanceA config.pole3.distanceB) / 
  (min config.pole3.distanceA config.pole3.distanceB) = 7 := by
  sorry

end NUMINAMATH_CALUDE_road_distance_ratio_l2573_257314


namespace NUMINAMATH_CALUDE_balloon_problem_solution_l2573_257344

def balloon_problem (initial_balloons : ℕ) (given_to_girl : ℕ) (floated_away : ℕ) (given_away_later : ℕ) (final_balloons : ℕ) : ℕ :=
  final_balloons - (initial_balloons - given_to_girl - floated_away - given_away_later)

theorem balloon_problem_solution :
  balloon_problem 50 1 12 9 39 = 11 := by
  sorry

end NUMINAMATH_CALUDE_balloon_problem_solution_l2573_257344


namespace NUMINAMATH_CALUDE_factor_expression_l2573_257361

theorem factor_expression (x : ℝ) : 63 * x^2 + 54 = 9 * (7 * x^2 + 6) := by
  sorry

end NUMINAMATH_CALUDE_factor_expression_l2573_257361


namespace NUMINAMATH_CALUDE_divisibility_implies_unit_l2573_257396

theorem divisibility_implies_unit (a b c d : ℤ) 
  (h1 : (ab - cd) ∣ a) 
  (h2 : (ab - cd) ∣ b) 
  (h3 : (ab - cd) ∣ c) 
  (h4 : (ab - cd) ∣ d) : 
  ab - cd = 1 ∨ ab - cd = -1 :=
by
  sorry

end NUMINAMATH_CALUDE_divisibility_implies_unit_l2573_257396


namespace NUMINAMATH_CALUDE_plus_sign_square_has_90_degree_symmetry_l2573_257328

/-- Represents a square with markings -/
structure MarkedSquare where
  markings : Set (ℝ × ℝ)

/-- Defines 90-degree rotational symmetry for a marked square -/
def has_90_degree_rotational_symmetry (s : MarkedSquare) : Prop :=
  ∀ (x y : ℝ), (x, y) ∈ s.markings ↔ (-y, x) ∈ s.markings

/-- Represents a square with vertical and horizontal midlines crossed (plus sign) -/
def plus_sign_square : MarkedSquare :=
  { markings := {(x, y) | x = 0 ∨ y = 0} }

/-- Theorem: A square with both vertical and horizontal midlines crossed (plus sign) has 90-degree rotational symmetry -/
theorem plus_sign_square_has_90_degree_symmetry :
  has_90_degree_rotational_symmetry plus_sign_square :=
sorry

end NUMINAMATH_CALUDE_plus_sign_square_has_90_degree_symmetry_l2573_257328


namespace NUMINAMATH_CALUDE_reservoir_crossing_time_l2573_257381

/-- The time it takes to cross a reservoir under specific conditions -/
theorem reservoir_crossing_time
  (b : ℝ)  -- width of the reservoir in km
  (v : ℝ)  -- swimming speed of A and C in km/h
  (h1 : 0 < b)  -- reservoir width is positive
  (h2 : 0 < v)  -- swimming speed is positive
  : ∃ (t : ℝ), t = (31 * b) / (130 * v) ∧ 
    (∃ (x d : ℝ),
      0 < x ∧ 0 < d ∧
      x = (9 * b) / 13 ∧
      d = (b - x) / 2 ∧
      2 * d + x = b ∧
      (b + 3 * x) / 2 / (10 * v) = d / v ∧
      t = ((b + 2 * x) / (10 * v))) :=
sorry

end NUMINAMATH_CALUDE_reservoir_crossing_time_l2573_257381


namespace NUMINAMATH_CALUDE_copper_percentage_in_alloy_l2573_257370

/-- Given the following conditions:
    - 30 ounces of 20% alloy is used
    - 70 ounces of 27% alloy is used
    - Total amount of the desired alloy is 100 ounces
    Prove that the percentage of copper in the desired alloy is 24.9% -/
theorem copper_percentage_in_alloy : 
  let alloy_20_amount : ℝ := 30
  let alloy_27_amount : ℝ := 70
  let total_alloy : ℝ := 100
  let alloy_20_copper_percentage : ℝ := 20
  let alloy_27_copper_percentage : ℝ := 27
  let copper_amount : ℝ := (alloy_20_amount * alloy_20_copper_percentage / 100) + 
                           (alloy_27_amount * alloy_27_copper_percentage / 100)
  copper_amount / total_alloy * 100 = 24.9 := by
  sorry

end NUMINAMATH_CALUDE_copper_percentage_in_alloy_l2573_257370


namespace NUMINAMATH_CALUDE_salary_increase_after_three_years_l2573_257392

theorem salary_increase_after_three_years (annual_raise : Real) (years : Nat) : 
  annual_raise = 0.15 → years = 3 → 
  ((1 + annual_raise) ^ years - 1) * 100 = 52.0875 := by
  sorry

end NUMINAMATH_CALUDE_salary_increase_after_three_years_l2573_257392


namespace NUMINAMATH_CALUDE_jamie_ball_collection_l2573_257368

def total_balls (initial_red : ℕ) (blue_multiplier : ℕ) (lost_red : ℕ) (bought_yellow : ℕ) : ℕ :=
  (initial_red - lost_red) + (initial_red * blue_multiplier) + bought_yellow

theorem jamie_ball_collection : total_balls 16 2 6 32 = 74 := by
  sorry

end NUMINAMATH_CALUDE_jamie_ball_collection_l2573_257368


namespace NUMINAMATH_CALUDE_sum_xy_equals_two_l2573_257373

theorem sum_xy_equals_two (w x y z : ℝ) 
  (eq1 : w + x + y = 3)
  (eq2 : x + y + z = 4)
  (eq3 : w + x + y + z = 5) :
  x + y = 2 := by
sorry

end NUMINAMATH_CALUDE_sum_xy_equals_two_l2573_257373


namespace NUMINAMATH_CALUDE_chips_for_dinner_l2573_257311

theorem chips_for_dinner (dinner : ℕ) (after : ℕ) : 
  dinner > 0 → 
  after > 0 → 
  dinner + after = 3 → 
  dinner = 2 := by
sorry

end NUMINAMATH_CALUDE_chips_for_dinner_l2573_257311


namespace NUMINAMATH_CALUDE_total_toys_given_l2573_257376

theorem total_toys_given (toy_cars : ℕ) (dolls : ℕ) (board_games : ℕ)
  (h1 : toy_cars = 134)
  (h2 : dolls = 269)
  (h3 : board_games = 87) :
  toy_cars + dolls + board_games = 490 := by
  sorry

end NUMINAMATH_CALUDE_total_toys_given_l2573_257376


namespace NUMINAMATH_CALUDE_smallest_four_digit_divisible_by_4_and_5_l2573_257351

theorem smallest_four_digit_divisible_by_4_and_5 :
  ∀ n : ℕ, 1000 ≤ n → n < 10000 → n % 4 = 0 → n % 5 = 0 → 1000 ≤ n :=
by sorry

end NUMINAMATH_CALUDE_smallest_four_digit_divisible_by_4_and_5_l2573_257351


namespace NUMINAMATH_CALUDE_student_admission_price_l2573_257334

theorem student_admission_price
  (total_tickets : ℕ)
  (adult_price : ℕ)
  (total_amount : ℕ)
  (student_attendees : ℕ)
  (h1 : total_tickets = 1500)
  (h2 : adult_price = 12)
  (h3 : total_amount = 16200)
  (h4 : student_attendees = 300) :
  (total_amount - (total_tickets - student_attendees) * adult_price) / student_attendees = 6 :=
by sorry

end NUMINAMATH_CALUDE_student_admission_price_l2573_257334


namespace NUMINAMATH_CALUDE_odometer_puzzle_l2573_257338

theorem odometer_puzzle (a b c : ℕ) : 
  (a ≥ 1) →
  (a + b + c ≤ 9) →
  (100 * b + 10 * a + c - (100 * a + 10 * b + c)) % 60 = 0 →
  a^2 + b^2 + c^2 = 35 := by
sorry

end NUMINAMATH_CALUDE_odometer_puzzle_l2573_257338


namespace NUMINAMATH_CALUDE_inequality_multiplication_l2573_257383

theorem inequality_multiplication (x y : ℝ) : x < y → 2 * x < 2 * y := by
  sorry

end NUMINAMATH_CALUDE_inequality_multiplication_l2573_257383


namespace NUMINAMATH_CALUDE_inequality_solution_set_minimum_m_value_minimum_fraction_value_l2573_257343

def f (x : ℝ) := |x + 1| + |x - 1|

theorem inequality_solution_set :
  {x : ℝ | f x < 2*x + 3} = {x : ℝ | x > -1/2} := by sorry

theorem minimum_m_value :
  (∃ (x₀ : ℝ), f x₀ ≤ 2) ∧ 
  (∀ (m : ℝ), (∃ (x : ℝ), f x ≤ m) → m ≥ 2) := by sorry

theorem minimum_fraction_value :
  ∀ (a b : ℝ), a > 0 → b > 0 → 3*a + b = 2 →
  1/(2*a) + 1/(a+b) ≥ 2 := by sorry

end NUMINAMATH_CALUDE_inequality_solution_set_minimum_m_value_minimum_fraction_value_l2573_257343


namespace NUMINAMATH_CALUDE_negation_false_l2573_257302

/-- A multi-digit number ends in 0 -/
def EndsInZero (n : ℕ) : Prop := n % 10 = 0 ∧ n ≥ 10

/-- A number is a multiple of 5 -/
def MultipleOfFive (n : ℕ) : Prop := ∃ k : ℕ, n = 5 * k

theorem negation_false : 
  ¬(∀ n : ℕ, EndsInZero n → MultipleOfFive n) → 
  (∃ n : ℕ, EndsInZero n ∧ ¬MultipleOfFive n) :=
by sorry

end NUMINAMATH_CALUDE_negation_false_l2573_257302


namespace NUMINAMATH_CALUDE_load_transport_l2573_257315

theorem load_transport (total_load : ℝ) (box_weight_max : ℝ) (num_trucks : ℕ) (truck_capacity : ℝ) :
  total_load = 13.5 →
  box_weight_max ≤ 0.35 →
  num_trucks = 11 →
  truck_capacity = 1.5 →
  ∃ (n : ℕ), n ≤ num_trucks ∧ n * truck_capacity ≥ total_load :=
by sorry

end NUMINAMATH_CALUDE_load_transport_l2573_257315


namespace NUMINAMATH_CALUDE_volumes_not_equal_sufficient_not_necessary_for_areas_not_always_equal_l2573_257366

/-- Represents a geometric body -/
structure GeometricBody where
  volume : ℝ
  crossSectionArea : ℝ → ℝ  -- Function mapping height to cross-sectional area

/-- The Gougu Principle -/
axiom gougu_principle {A B : GeometricBody} (h : ∀ (height : ℝ), A.crossSectionArea height = B.crossSectionArea height) :
  A.volume = B.volume

/-- Two geometric bodies have the same height -/
def same_height (A B : GeometricBody) : Prop := true

theorem volumes_not_equal_sufficient_not_necessary_for_areas_not_always_equal
  (A B : GeometricBody) (h : same_height A B) :
  (∃ (height : ℝ), A.crossSectionArea height ≠ B.crossSectionArea height) ↔ 
  (A.volume ≠ B.volume ∨ (A.volume = B.volume ∧ ∃ (height : ℝ), A.crossSectionArea height ≠ B.crossSectionArea height)) :=
sorry

end NUMINAMATH_CALUDE_volumes_not_equal_sufficient_not_necessary_for_areas_not_always_equal_l2573_257366
