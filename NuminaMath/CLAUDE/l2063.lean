import Mathlib

namespace NUMINAMATH_CALUDE_bus_stop_walk_time_l2063_206345

theorem bus_stop_walk_time (usual_time : ℝ) (usual_speed : ℝ) : 
  usual_speed > 0 →
  (2 / 3 * usual_speed) * (usual_time + 15) = usual_speed * usual_time →
  usual_time = 30 := by
  sorry

end NUMINAMATH_CALUDE_bus_stop_walk_time_l2063_206345


namespace NUMINAMATH_CALUDE_k_upper_bound_l2063_206358

noncomputable section

variables (a k : ℝ)

def f (x : ℝ) : ℝ :=
  if x ≤ a then -(x - a)^2 - k - a else Real.exp x / (a - x)

theorem k_upper_bound :
  (∃ x₀ ≤ a, ∀ x₁ > a, f a k x₁ ≤ f a k x₀) →
  k ≤ 2 := by
  sorry

end

end NUMINAMATH_CALUDE_k_upper_bound_l2063_206358


namespace NUMINAMATH_CALUDE_ellipse_line_intersection_range_l2063_206342

/-- Represents an ellipse with semi-major axis a and semi-minor axis b -/
structure Ellipse where
  a : ℝ
  b : ℝ
  h_pos : 0 < b ∧ b < a

/-- Represents a line with slope 1 and y-intercept m -/
structure Line where
  m : ℝ

/-- Represents the intersection of an ellipse and a line -/
def Intersection (e : Ellipse) (l : Line) :=
  {p : ℝ × ℝ | p.1^2 / e.a^2 + p.2^2 / e.b^2 = 1 ∧ p.2 = p.1 + l.m}

/-- Theorem stating the range of m for which the line intersects the ellipse at two distinct points forming an acute angle at the origin -/
theorem ellipse_line_intersection_range (e : Ellipse) (l : Line) 
  (h_minor : e.b = 1)
  (h_ecc : Real.sqrt (1 - e.b^2 / e.a^2) = Real.sqrt 3 / 2)
  (h_intersect : ∃ A B, A ∈ Intersection e l ∧ B ∈ Intersection e l ∧ A ≠ B)
  (h_acute : ∃ A B, A ∈ Intersection e l ∧ B ∈ Intersection e l ∧ A ≠ B ∧ 
    0 < Real.arccos ((A.1 * B.1 + A.2 * B.2) / (Real.sqrt (A.1^2 + A.2^2) * Real.sqrt (B.1^2 + B.2^2))) ∧
    Real.arccos ((A.1 * B.1 + A.2 * B.2) / (Real.sqrt (A.1^2 + A.2^2) * Real.sqrt (B.1^2 + B.2^2))) < π / 2) :
  (-Real.sqrt 5 < l.m ∧ l.m < -2 * Real.sqrt 10 / 5) ∨ (2 * Real.sqrt 10 / 5 < l.m ∧ l.m < Real.sqrt 5) := by
  sorry

end NUMINAMATH_CALUDE_ellipse_line_intersection_range_l2063_206342


namespace NUMINAMATH_CALUDE_fraction_multiplication_l2063_206397

theorem fraction_multiplication : (2 : ℚ) / 5 * 5 / 7 * 7 / 3 * 3 / 8 = 1 / 4 := by
  sorry

end NUMINAMATH_CALUDE_fraction_multiplication_l2063_206397


namespace NUMINAMATH_CALUDE_polynomial_less_than_factorial_l2063_206328

theorem polynomial_less_than_factorial (A B C : ℝ) : 
  ∃ N : ℕ, ∀ n : ℕ, n ≥ N → (A * n^2 + B * n + C : ℝ) < n! :=
sorry

end NUMINAMATH_CALUDE_polynomial_less_than_factorial_l2063_206328


namespace NUMINAMATH_CALUDE_large_pizza_slices_l2063_206374

/-- Proves that a large pizza has 12 slices given the problem conditions -/
theorem large_pizza_slices :
  ∀ (small_slices medium_slices large_slices : ℕ)
    (total_pizzas small_pizzas medium_pizzas large_pizzas : ℕ)
    (total_slices : ℕ),
  small_slices = 6 →
  medium_slices = 8 →
  total_pizzas = 15 →
  small_pizzas = 4 →
  medium_pizzas = 5 →
  large_pizzas = total_pizzas - small_pizzas - medium_pizzas →
  total_slices = 136 →
  total_slices = small_slices * small_pizzas + medium_slices * medium_pizzas + large_slices * large_pizzas →
  large_slices = 12 := by
sorry

end NUMINAMATH_CALUDE_large_pizza_slices_l2063_206374


namespace NUMINAMATH_CALUDE_other_items_tax_is_ten_percent_l2063_206343

/-- Represents the tax rates and spending percentages in Jill's shopping trip -/
structure ShoppingTax where
  clothing_spend : Rat
  food_spend : Rat
  other_spend : Rat
  clothing_tax : Rat
  food_tax : Rat
  total_tax : Rat

/-- The tax rate on other items given the shopping tax structure -/
def other_items_tax_rate (st : ShoppingTax) : Rat :=
  (st.total_tax - st.clothing_tax * st.clothing_spend) / st.other_spend

/-- Theorem stating that the tax rate on other items is 10% -/
theorem other_items_tax_is_ten_percent (st : ShoppingTax) 
  (h1 : st.clothing_spend = 1/2)
  (h2 : st.food_spend = 1/5)
  (h3 : st.other_spend = 3/10)
  (h4 : st.clothing_tax = 1/20)
  (h5 : st.food_tax = 0)
  (h6 : st.total_tax = 11/200) :
  other_items_tax_rate st = 1/10 := by
  sorry


end NUMINAMATH_CALUDE_other_items_tax_is_ten_percent_l2063_206343


namespace NUMINAMATH_CALUDE_simplify_expression_l2063_206366

theorem simplify_expression (y : ℝ) : (3 - Real.sqrt (y^2 - 9))^2 = y^2 - 6 * Real.sqrt (y^2 - 9) := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l2063_206366


namespace NUMINAMATH_CALUDE_min_value_of_function_l2063_206303

theorem min_value_of_function (x : ℝ) (h : x > -1) :
  x - 4 + 9 / (x + 1) ≥ 1 ∧
  (x - 4 + 9 / (x + 1) = 1 ↔ x = 2) :=
by sorry

end NUMINAMATH_CALUDE_min_value_of_function_l2063_206303


namespace NUMINAMATH_CALUDE_book_pages_sum_l2063_206361

theorem book_pages_sum (chapter1 chapter2 chapter3 : ℕ) 
  (h1 : chapter1 = 66)
  (h2 : chapter2 = 35)
  (h3 : chapter3 = 24) :
  chapter1 + chapter2 + chapter3 = 125 := by
  sorry

end NUMINAMATH_CALUDE_book_pages_sum_l2063_206361


namespace NUMINAMATH_CALUDE_union_of_given_sets_l2063_206355

theorem union_of_given_sets :
  let A : Set Int := {-3, 1, 2}
  let B : Set Int := {0, 1, 2, 3}
  A ∪ B = {-3, 0, 1, 2, 3} := by
sorry

end NUMINAMATH_CALUDE_union_of_given_sets_l2063_206355


namespace NUMINAMATH_CALUDE_least_number_with_digit_product_12_l2063_206357

def is_valid_number (n : ℕ) : Prop :=
  100 ≤ n ∧ n < 1000 ∧ (n / 100) * ((n / 10) % 10) * (n % 10) = 12

theorem least_number_with_digit_product_12 :
  ∀ n : ℕ, is_valid_number n → 134 ≤ n :=
by sorry

end NUMINAMATH_CALUDE_least_number_with_digit_product_12_l2063_206357


namespace NUMINAMATH_CALUDE_pipe_fill_time_l2063_206399

/-- Time to fill tank with leak (in hours) -/
def time_with_leak : ℝ := 15

/-- Time for leak to empty full tank (in hours) -/
def time_leak_empty : ℝ := 30

/-- Time to fill tank without leak (in hours) -/
def time_without_leak : ℝ := 10

theorem pipe_fill_time :
  (1 / time_without_leak) - (1 / time_leak_empty) = (1 / time_with_leak) :=
sorry

end NUMINAMATH_CALUDE_pipe_fill_time_l2063_206399


namespace NUMINAMATH_CALUDE_not_p_necessary_not_sufficient_for_not_p_or_q_l2063_206337

theorem not_p_necessary_not_sufficient_for_not_p_or_q (p q : Prop) :
  (¬(p ∨ q) → ¬p) ∧ ¬(¬p → ¬(p ∨ q)) := by
  sorry

end NUMINAMATH_CALUDE_not_p_necessary_not_sufficient_for_not_p_or_q_l2063_206337


namespace NUMINAMATH_CALUDE_common_chord_of_circles_l2063_206359

/-- The equation of the first circle -/
def circle1 (x y : ℝ) : Prop := x^2 + y^2 - 25 = 0

/-- The equation of the second circle -/
def circle2 (x y : ℝ) : Prop := x^2 + y^2 - 4*x + 3*y - 10 = 0

/-- The equation of the potential common chord -/
def common_chord (x y : ℝ) : Prop := 4*x - 3*y - 15 = 0

/-- Theorem stating that the given equation represents the common chord of the two circles -/
theorem common_chord_of_circles :
  ∀ x y : ℝ, (circle1 x y ∧ circle2 x y) → common_chord x y :=
by sorry

end NUMINAMATH_CALUDE_common_chord_of_circles_l2063_206359


namespace NUMINAMATH_CALUDE_choir_group_ratio_l2063_206394

theorem choir_group_ratio (total_sopranos total_altos num_groups : ℕ) 
  (h1 : total_sopranos = 10)
  (h2 : total_altos = 15)
  (h3 : num_groups = 5)
  (h4 : total_sopranos % num_groups = 0)
  (h5 : total_altos % num_groups = 0) :
  (total_sopranos / num_groups : ℚ) / (total_altos / num_groups : ℚ) = 2 / 3 :=
by sorry

end NUMINAMATH_CALUDE_choir_group_ratio_l2063_206394


namespace NUMINAMATH_CALUDE_distance_post_office_to_home_l2063_206384

/-- The distance Spencer walked from his house to the library -/
def distance_house_to_library : ℝ := 0.3

/-- The distance Spencer walked from the library to the post office -/
def distance_library_to_post_office : ℝ := 0.1

/-- The total distance Spencer walked -/
def total_distance : ℝ := 0.8

/-- Theorem: The distance Spencer walked from the post office back home is 0.4 miles -/
theorem distance_post_office_to_home : 
  total_distance - (distance_house_to_library + distance_library_to_post_office) = 0.4 := by
  sorry

end NUMINAMATH_CALUDE_distance_post_office_to_home_l2063_206384


namespace NUMINAMATH_CALUDE_horner_rule_v2_equals_14_l2063_206300

def f (x : ℝ) : ℝ := 2*x^4 + 3*x^3 + 5*x - 4

def horner_v2 (a b c d e x : ℝ) : ℝ := ((a * x + b) * x + c) * x + d

theorem horner_rule_v2_equals_14 : 
  horner_v2 2 3 0 5 (-4) 2 = 14 := by
  sorry

end NUMINAMATH_CALUDE_horner_rule_v2_equals_14_l2063_206300


namespace NUMINAMATH_CALUDE_gcd_459_357_l2063_206377

theorem gcd_459_357 : Nat.gcd 459 357 = 51 := by
  sorry

end NUMINAMATH_CALUDE_gcd_459_357_l2063_206377


namespace NUMINAMATH_CALUDE_eight_digit_number_divisibility_l2063_206307

/-- Represents an eight-digit number in the form 757AB384 -/
def EightDigitNumber (A B : ℕ) : ℕ := 757000000 + A * 10000 + B * 1000 + 384

/-- The number is divisible by 357 -/
def IsDivisibleBy357 (n : ℕ) : Prop := ∃ k : ℕ, n = 357 * k

theorem eight_digit_number_divisibility :
  ∀ A : ℕ, (A < 10) →
    (IsDivisibleBy357 (EightDigitNumber A 5) ∧
     ∀ B : ℕ, B < 10 → B ≠ 5 → ¬IsDivisibleBy357 (EightDigitNumber A B)) :=
by sorry

end NUMINAMATH_CALUDE_eight_digit_number_divisibility_l2063_206307


namespace NUMINAMATH_CALUDE_quadratic_is_square_of_binomial_l2063_206368

theorem quadratic_is_square_of_binomial :
  ∃ (r s : ℚ), (r * X + s)^2 = (81/16 : ℚ) * X^2 + 18 * X + 16 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_is_square_of_binomial_l2063_206368


namespace NUMINAMATH_CALUDE_distribute_five_balls_four_boxes_l2063_206336

/-- The number of ways to distribute n distinguishable objects into k distinguishable containers -/
def distribute (n k : ℕ) : ℕ := k ^ n

/-- Proof that distributing 5 distinguishable balls into 4 distinguishable boxes results in 1024 ways -/
theorem distribute_five_balls_four_boxes : distribute 5 4 = 1024 := by
  sorry

end NUMINAMATH_CALUDE_distribute_five_balls_four_boxes_l2063_206336


namespace NUMINAMATH_CALUDE_increase_by_percentage_l2063_206340

theorem increase_by_percentage (initial : ℝ) (percentage : ℝ) (final : ℝ) :
  initial = 110 → percentage = 50 → final = initial * (1 + percentage / 100) → final = 165 := by
  sorry

end NUMINAMATH_CALUDE_increase_by_percentage_l2063_206340


namespace NUMINAMATH_CALUDE_technician_salary_l2063_206356

/-- The average salary of technicians in a workshop --/
theorem technician_salary (total_workers : ℝ) (total_avg_salary : ℝ) 
  (num_technicians : ℝ) (non_tech_avg_salary : ℝ) :
  total_workers = 21.11111111111111 →
  total_avg_salary = 1000 →
  num_technicians = 10 →
  non_tech_avg_salary = 820 →
  (total_workers * total_avg_salary - (total_workers - num_technicians) * non_tech_avg_salary) 
    / num_technicians = 1200 := by
  sorry

end NUMINAMATH_CALUDE_technician_salary_l2063_206356


namespace NUMINAMATH_CALUDE_x_in_P_sufficient_not_necessary_for_x_in_Q_l2063_206349

-- Define the sets P and Q
def P : Set ℝ := {x | x^2 - 4*x + 3 ≤ 0}
def Q : Set ℝ := {x | ∃ y, y = Real.sqrt (x + 1) + Real.sqrt (3 - x)}

-- State the theorem
theorem x_in_P_sufficient_not_necessary_for_x_in_Q :
  (∀ x, x ∈ P → x ∈ Q) ∧ (∃ x, x ∈ Q ∧ x ∉ P) := by
  sorry

end NUMINAMATH_CALUDE_x_in_P_sufficient_not_necessary_for_x_in_Q_l2063_206349


namespace NUMINAMATH_CALUDE_perpendicular_parallel_implication_l2063_206375

-- Define a type for lines in 3D space
variable (Line : Type)

-- Define the perpendicular relation between two lines
variable (perpendicular : Line → Line → Prop)

-- Define the parallel relation between two lines
variable (parallel : Line → Line → Prop)

-- State the theorem
theorem perpendicular_parallel_implication (l₁ l₂ l₃ : Line) :
  perpendicular l₁ l₂ → parallel l₂ l₃ → perpendicular l₁ l₃ :=
by sorry

end NUMINAMATH_CALUDE_perpendicular_parallel_implication_l2063_206375


namespace NUMINAMATH_CALUDE_meshed_gears_speed_proportion_l2063_206350

/-- Represents a gear with number of teeth and angular speed -/
structure Gear where
  teeth : ℕ
  speed : ℝ

/-- Proves that for four meshed gears, their angular speeds are proportional to yzw : xzw : xyw : xyz -/
theorem meshed_gears_speed_proportion
  (A B C D : Gear)
  (h_mesh : A.teeth * A.speed = B.teeth * B.speed ∧
            B.teeth * B.speed = C.teeth * C.speed ∧
            C.teeth * C.speed = D.teeth * D.speed) :
  ∃ (k : ℝ), k ≠ 0 ∧
    A.speed = k * (B.teeth * C.teeth * D.teeth) ∧
    B.speed = k * (A.teeth * C.teeth * D.teeth) ∧
    C.speed = k * (A.teeth * B.teeth * D.teeth) ∧
    D.speed = k * (A.teeth * B.teeth * C.teeth) :=
sorry

end NUMINAMATH_CALUDE_meshed_gears_speed_proportion_l2063_206350


namespace NUMINAMATH_CALUDE_dance_group_average_age_l2063_206346

theorem dance_group_average_age 
  (num_females : ℕ) 
  (num_males : ℕ) 
  (avg_female_age : ℝ) 
  (avg_male_age : ℝ) 
  (h1 : num_females = 12)
  (h2 : avg_female_age = 25)
  (h3 : num_males = 18)
  (h4 : avg_male_age = 40)
  (h5 : num_females + num_males = 30) : 
  (num_females * avg_female_age + num_males * avg_male_age) / (num_females + num_males) = 34 := by
  sorry

end NUMINAMATH_CALUDE_dance_group_average_age_l2063_206346


namespace NUMINAMATH_CALUDE_arithmetic_sequence_fifth_term_l2063_206320

/-- An arithmetic sequence is a sequence where the difference between
    any two consecutive terms is constant. -/
def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

/-- Given an arithmetic sequence satisfying certain conditions,
    prove that the 5th term equals 10. -/
theorem arithmetic_sequence_fifth_term
  (a : ℕ → ℝ)
  (h_arithmetic : is_arithmetic_sequence a)
  (h_sum : a 1 + a 5 = 8)
  (h_fourth : a 4 = 7) :
  a 5 = 10 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_fifth_term_l2063_206320


namespace NUMINAMATH_CALUDE_equation1_solutions_equation2_solutions_l2063_206398

-- Define the equations
def equation1 (x : ℝ) : Prop := 3 * x * (x + 3) = 2 * (x + 3)
def equation2 (x : ℝ) : Prop := x^2 - 4*x - 5 = 0

-- Theorem for equation 1
theorem equation1_solutions :
  ∃ (x₁ x₂ : ℝ), x₁ = -3 ∧ x₂ = 2/3 ∧ equation1 x₁ ∧ equation1 x₂ ∧
  ∀ (x : ℝ), equation1 x → x = x₁ ∨ x = x₂ := by sorry

-- Theorem for equation 2
theorem equation2_solutions :
  ∃ (x₁ x₂ : ℝ), x₁ = 5 ∧ x₂ = -1 ∧ equation2 x₁ ∧ equation2 x₂ ∧
  ∀ (x : ℝ), equation2 x → x = x₁ ∨ x = x₂ := by sorry

end NUMINAMATH_CALUDE_equation1_solutions_equation2_solutions_l2063_206398


namespace NUMINAMATH_CALUDE_wall_painting_fraction_l2063_206331

theorem wall_painting_fraction (paint_rate : ℝ) (total_time minutes : ℝ) 
  (h1 : paint_rate * total_time = 1)  -- Can paint whole wall in total_time
  (h2 : minutes / total_time = 1 / 5) -- Minutes is 1/5 of total time
  : paint_rate * minutes = 1 / 5 := by
  sorry

end NUMINAMATH_CALUDE_wall_painting_fraction_l2063_206331


namespace NUMINAMATH_CALUDE_simplify_expression_l2063_206390

theorem simplify_expression (z : ℝ) : (5 - 2 * z^2) - (4 * z^2 - 7) = 12 - 6 * z^2 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l2063_206390


namespace NUMINAMATH_CALUDE_distance_CX_l2063_206389

/-- Given five points A, B, C, D, X on a plane with specific distances between them,
    prove that the distance between C and X is 3. -/
theorem distance_CX (A B C D X : EuclideanSpace ℝ (Fin 2)) 
  (h1 : dist A C = 2)
  (h2 : dist A X = 5)
  (h3 : dist A D = 11)
  (h4 : dist C D = 9)
  (h5 : dist C B = 10)
  (h6 : dist D B = 1)
  (h7 : dist X B = 7) :
  dist C X = 3 := by
  sorry


end NUMINAMATH_CALUDE_distance_CX_l2063_206389


namespace NUMINAMATH_CALUDE_min_three_colors_proof_l2063_206391

/-- Represents the number of balls of each color in the box -/
structure BallCounts where
  red : ℕ
  green : ℕ
  blue : ℕ
  white : ℕ

/-- The total number of balls in the box -/
def total_balls : ℕ := 111

/-- The number of balls that, when drawn, ensures getting all four colors -/
def all_colors_draw : ℕ := 100

/-- Predicate to check if a BallCounts configuration is valid -/
def valid_configuration (counts : BallCounts) : Prop :=
  counts.red + counts.green + counts.blue + counts.white = total_balls ∧
  ∀ (n : ℕ), n ≥ all_colors_draw →
    n - counts.red < all_colors_draw ∧
    n - counts.green < all_colors_draw ∧
    n - counts.blue < all_colors_draw ∧
    n - counts.white < all_colors_draw

/-- The smallest number of balls to draw to ensure at least three colors -/
def min_three_colors_draw : ℕ := 88

theorem min_three_colors_proof :
  ∀ (counts : BallCounts),
    valid_configuration counts →
    (∀ (n : ℕ), n ≥ min_three_colors_draw →
      ∃ (colors : Finset (Fin 4)),
        colors.card ≥ 3 ∧
        (∀ (i : Fin 4),
          i ∈ colors ↔
            (i = 0 ∧ n > total_balls - counts.red) ∨
            (i = 1 ∧ n > total_balls - counts.green) ∨
            (i = 2 ∧ n > total_balls - counts.blue) ∨
            (i = 3 ∧ n > total_balls - counts.white))) ∧
    (∀ (m : ℕ), m < min_three_colors_draw →
      ∃ (counts' : BallCounts),
        valid_configuration counts' ∧
        ∃ (colors : Finset (Fin 4)),
          colors.card < 3 ∧
          (∀ (i : Fin 4),
            i ∈ colors ↔
              (i = 0 ∧ m > total_balls - counts'.red) ∨
              (i = 1 ∧ m > total_balls - counts'.green) ∨
              (i = 2 ∧ m > total_balls - counts'.blue) ∨
              (i = 3 ∧ m > total_balls - counts'.white))) :=
by sorry

end NUMINAMATH_CALUDE_min_three_colors_proof_l2063_206391


namespace NUMINAMATH_CALUDE_line_equations_l2063_206386

-- Define a point in 2D space
structure Point2D where
  x : ℝ
  y : ℝ

-- Define a line in 2D space using slope-intercept form
structure Line2D where
  slope : ℝ
  intercept : ℝ

def isParallel (l1 l2 : Line2D) : Prop :=
  l1.slope = l2.slope

def isPerpendicular (l1 l2 : Line2D) : Prop :=
  l1.slope * l2.slope = -1

def pointOnLine (p : Point2D) (l : Line2D) : Prop :=
  p.y = l.slope * p.x + l.intercept

theorem line_equations (P : Point2D) (l : Line2D) (given_line : Line2D) :
  P.x = -2 ∧ P.y = 1 ∧ given_line.slope = 1/2 ∧ given_line.intercept = -1/2 →
  (isParallel l given_line → l.slope = 1/2 ∧ l.intercept = 3/2) ∧
  (isPerpendicular l given_line → l.slope = -2 ∧ l.intercept = -5/2) :=
sorry

end NUMINAMATH_CALUDE_line_equations_l2063_206386


namespace NUMINAMATH_CALUDE_complex_equation_solution_l2063_206395

theorem complex_equation_solution :
  ∃ z : ℂ, z^2 - 4*z + 21 = 0 :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l2063_206395


namespace NUMINAMATH_CALUDE_range_of_a_l2063_206304

/-- The function f(x) = x^2 - 2x --/
def f (x : ℝ) : ℝ := x^2 - 2*x

/-- The function g(x) = ax + 2, where a > 0 --/
def g (a : ℝ) (x : ℝ) : ℝ := a*x + 2

/-- The theorem stating the range of a --/
theorem range_of_a (a : ℝ) : 
  (a > 0 ∧ 
   ∀ x₁ ∈ Set.Icc (-1 : ℝ) 2, ∃ x₀ ∈ Set.Icc (-1 : ℝ) 2, g a x₁ = f x₀) → 
  (1/2 : ℝ) ≤ a ∧ a ≤ 3 := by sorry

end NUMINAMATH_CALUDE_range_of_a_l2063_206304


namespace NUMINAMATH_CALUDE_golden_state_points_l2063_206324

/-- The total points scored by the Golden State Team -/
def golden_state_total (draymond curry kelly durant klay : ℕ) : ℕ :=
  draymond + curry + kelly + durant + klay

/-- Theorem stating the total points of the Golden State Team -/
theorem golden_state_points :
  ∃ (draymond curry kelly durant klay : ℕ),
    draymond = 12 ∧
    curry = 2 * draymond ∧
    kelly = 9 ∧
    durant = 2 * kelly ∧
    klay = draymond / 2 ∧
    golden_state_total draymond curry kelly durant klay = 69 := by
  sorry

end NUMINAMATH_CALUDE_golden_state_points_l2063_206324


namespace NUMINAMATH_CALUDE_f_range_and_triangle_property_l2063_206335

noncomputable def m (x : ℝ) : ℝ × ℝ := (Real.sqrt 3 * Real.sin x, 1 - Real.sqrt 2 * Real.sin x)
noncomputable def n (x : ℝ) : ℝ × ℝ := (2 * Real.cos x, 1 + Real.sqrt 2 * Real.sin x)

noncomputable def f (x : ℝ) : ℝ := (m x).1 * (n x).1 + (m x).2 * (n x).2

theorem f_range_and_triangle_property :
  (∀ y ∈ Set.Icc (-1 : ℝ) 2, ∃ x ∈ Set.Icc 0 (Real.pi / 2), f x = y) ∧
  (∀ {a b c A B C : ℝ},
    b / a = Real.sqrt 3 →
    (Real.sin B * Real.cos A) / Real.sin A = 2 - Real.cos B →
    f B = 1) :=
sorry

end NUMINAMATH_CALUDE_f_range_and_triangle_property_l2063_206335


namespace NUMINAMATH_CALUDE_arithmetic_sequence_150th_term_l2063_206379

/-- The nth term of an arithmetic sequence -/
def arithmetic_sequence (a₁ : ℝ) (d : ℝ) (n : ℕ) : ℝ :=
  a₁ + (n - 1 : ℝ) * d

/-- The first term of our specific sequence -/
def a₁ : ℝ := 3

/-- The common difference of our specific sequence -/
def d : ℝ := 5

/-- The 150th term of our specific sequence -/
def a₁₅₀ : ℝ := arithmetic_sequence a₁ d 150

theorem arithmetic_sequence_150th_term : a₁₅₀ = 748 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_150th_term_l2063_206379


namespace NUMINAMATH_CALUDE_noah_garden_larger_by_75_l2063_206383

/-- Represents the dimensions of a rectangular garden -/
structure GardenDimensions where
  length : ℝ
  width : ℝ

/-- Calculates the area of a rectangular garden -/
def gardenArea (d : GardenDimensions) : ℝ := d.length * d.width

theorem noah_garden_larger_by_75 (liam_garden noah_garden : GardenDimensions) 
  (h1 : liam_garden.length = 30 ∧ liam_garden.width = 50)
  (h2 : noah_garden.length = 35 ∧ noah_garden.width = 45) : 
  gardenArea noah_garden - gardenArea liam_garden = 75 := by
  sorry

#check noah_garden_larger_by_75

end NUMINAMATH_CALUDE_noah_garden_larger_by_75_l2063_206383


namespace NUMINAMATH_CALUDE_max_temperature_range_l2063_206316

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d > 1 → d < n → ¬(n % d = 0)

theorem max_temperature_range 
  (T1 T2 T3 T4 T5 : ℕ) 
  (avg_temp : (T1 + T2 + T3 + T4 + T5) / 5 = 60)
  (lowest_temp : T1 = 50 ∧ T2 = 50)
  (consecutive : ∃ n : ℕ, T3 = n ∧ T4 = n + 1 ∧ T5 = n + 2)
  (ordered : T3 ≤ T4 ∧ T4 ≤ T5)
  (prime_exists : is_prime T3 ∨ is_prime T4 ∨ is_prime T5) :
  T5 - T1 = 18 :=
sorry

end NUMINAMATH_CALUDE_max_temperature_range_l2063_206316


namespace NUMINAMATH_CALUDE_log_sum_difference_l2063_206314

theorem log_sum_difference (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) :
  Real.log 50 / Real.log 10 + Real.log 20 / Real.log 10 - Real.log 4 / Real.log 10 = 2 + Real.log 2.5 / Real.log 10 := by
  sorry

end NUMINAMATH_CALUDE_log_sum_difference_l2063_206314


namespace NUMINAMATH_CALUDE_calculate_total_income_person_total_income_l2063_206396

/-- Calculates a person's total income based on given distributions --/
theorem calculate_total_income (children_share : Real) (wife_share : Real) 
  (orphan_donation_rate : Real) (final_amount : Real) : Real :=
  let total_distributed := children_share + wife_share
  let remaining_before_donation := 1 - total_distributed
  let orphan_donation := orphan_donation_rate * remaining_before_donation
  let final_share := remaining_before_donation - orphan_donation
  final_amount / final_share

/-- Proves that the person's total income is $150,000 --/
theorem person_total_income : 
  calculate_total_income 0.25 0.35 0.1 45000 = 150000 := by
  sorry

end NUMINAMATH_CALUDE_calculate_total_income_person_total_income_l2063_206396


namespace NUMINAMATH_CALUDE_inequality_proof_l2063_206334

theorem inequality_proof (a b c : ℝ) 
  (h_pos : a > 0 ∧ b > 0 ∧ c > 0) 
  (h_sum : a^(3/2) + b^(3/2) + c^(3/2) = 1) : 
  a * b * c ≤ 1 / 9 ∧ 
  a / (b + c) + b / (a + c) + c / (a + b) ≤ 1 / (2 * Real.sqrt (a * b * c)) := by
sorry

end NUMINAMATH_CALUDE_inequality_proof_l2063_206334


namespace NUMINAMATH_CALUDE_parallelepiped_coverage_l2063_206354

/-- Represents a parallelepiped -/
structure Parallelepiped where
  length : ℕ
  width : ℕ
  height : ℕ

/-- Represents a square -/
structure Square where
  side : ℕ

/-- Calculates the surface area of a parallelepiped -/
def surface_area (p : Parallelepiped) : ℕ :=
  2 * (p.length * p.width + p.length * p.height + p.width * p.height)

/-- Calculates the area of a square -/
def square_area (s : Square) : ℕ :=
  s.side * s.side

/-- Theorem stating that a 1x1x4 parallelepiped can be covered by two 4x4 squares and one 1x1 square -/
theorem parallelepiped_coverage :
  ∃ (p : Parallelepiped) (s1 s2 s3 : Square),
    p.length = 1 ∧ p.width = 1 ∧ p.height = 4 ∧
    s1.side = 4 ∧ s2.side = 4 ∧ s3.side = 1 ∧
    surface_area p = square_area s1 + square_area s2 + square_area s3 :=
sorry

end NUMINAMATH_CALUDE_parallelepiped_coverage_l2063_206354


namespace NUMINAMATH_CALUDE_quadratic_equation_solution_trigonometric_expression_value_l2063_206370

-- Part 1: Quadratic equation
theorem quadratic_equation_solution :
  let f : ℝ → ℝ := λ x => x^2 - 4*x + 3
  ∃ x₁ x₂ : ℝ, x₁ = 3 ∧ x₂ = 1 ∧ f x₁ = 0 ∧ f x₂ = 0 ∧ ∀ x : ℝ, f x = 0 → x = x₁ ∨ x = x₂ :=
by sorry

-- Part 2: Trigonometric expression
theorem trigonometric_expression_value :
  4 * Real.sin (π/6) - Real.sqrt 2 * Real.cos (π/4) + Real.sqrt 3 * Real.tan (π/3) = 4 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equation_solution_trigonometric_expression_value_l2063_206370


namespace NUMINAMATH_CALUDE_parallel_lines_circle_chord_l2063_206306

theorem parallel_lines_circle_chord (r : ℝ) (d : ℝ) : 
  r > 0 ∧ d > 0 ∧ 
  36 * r^2 = 648 + 9 * d^2 ∧ 
  40 * r^2 = 800 + 90 * d^2 → 
  d = 67/10 := by
sorry

end NUMINAMATH_CALUDE_parallel_lines_circle_chord_l2063_206306


namespace NUMINAMATH_CALUDE_smallest_part_in_ratio_l2063_206362

/-- Given a total amount of (3000 + b) divided in the ratio 5:6:8, where the smallest part is c, then c = 100 -/
theorem smallest_part_in_ratio (b : ℝ) (c : ℝ) : 
  (c = (5 : ℝ) / (5 + 6 + 8 : ℝ) * (3000 + b)) → c = 100 :=
by
  sorry

end NUMINAMATH_CALUDE_smallest_part_in_ratio_l2063_206362


namespace NUMINAMATH_CALUDE_min_e_value_l2063_206321

-- Define the circle and points
def Circle : Type := Unit
def Point : Type := Unit

-- Define the diameter of the circle
def diameter : ℝ := 4

-- Define the points
def P : Point := sorry
def Q : Point := sorry
def X : Point := sorry
def Y : Point := sorry
def Z : Point := sorry

-- Define the properties of the points
def is_diameter (p q : Point) : Prop := sorry
def on_semicircle (p : Point) : Prop := sorry
def is_midpoint (x : Point) : Prop := sorry
def distance (p q : Point) : ℝ := sorry
def symmetric_to (z p q : Point) : Prop := sorry

-- Define the intersection points
def A : Point := sorry
def B : Point := sorry

-- Define the length of AB
def e : ℝ := sorry

-- State the theorem
theorem min_e_value (c : Circle) :
  is_diameter P Q →
  on_semicircle X →
  on_semicircle Y →
  is_midpoint X →
  distance P Y = 5 / 4 →
  symmetric_to Z P Q →
  ∃ (min_e : ℝ), (∀ e', e' ≥ min_e) ∧ min_e = 6 - 5 * Real.sqrt 3 :=
sorry

end NUMINAMATH_CALUDE_min_e_value_l2063_206321


namespace NUMINAMATH_CALUDE_passing_mark_is_40_l2063_206317

/-- Represents the exam results of a class -/
structure ExamResults where
  total_students : ℕ
  absent_percentage : ℚ
  failed_percentage : ℚ
  just_passed_percentage : ℚ
  remaining_average : ℚ
  class_average : ℚ
  fail_margin : ℕ

/-- Calculates the passing mark for the exam given the exam results -/
def calculate_passing_mark (results : ExamResults) : ℚ :=
  let absent := results.total_students * results.absent_percentage
  let failed := results.total_students * results.failed_percentage
  let just_passed := results.total_students * results.just_passed_percentage
  let remaining := results.total_students - (absent + failed + just_passed)
  let total_marks := results.class_average * results.total_students
  let remaining_marks := remaining * results.remaining_average
  (total_marks - remaining_marks) / (failed + just_passed) + results.fail_margin

/-- Theorem stating that given the exam results, the passing mark is 40 -/
theorem passing_mark_is_40 (results : ExamResults) 
  (h1 : results.total_students = 100)
  (h2 : results.absent_percentage = 1/5)
  (h3 : results.failed_percentage = 3/10)
  (h4 : results.just_passed_percentage = 1/10)
  (h5 : results.remaining_average = 65)
  (h6 : results.class_average = 36)
  (h7 : results.fail_margin = 20) :
  calculate_passing_mark results = 40 := by
  sorry

end NUMINAMATH_CALUDE_passing_mark_is_40_l2063_206317


namespace NUMINAMATH_CALUDE_system_solution_l2063_206363

theorem system_solution : ∃! (x y : ℝ), 
  (x + y = (7 - x) + (7 - y)) ∧ 
  (x - y = (x - 2) + (y - 2)) ∧ 
  x = 3 ∧ y = 4 := by
  sorry

end NUMINAMATH_CALUDE_system_solution_l2063_206363


namespace NUMINAMATH_CALUDE_nursery_school_fraction_l2063_206323

/-- Given a nursery school with the following conditions:
  1. 20 students are under 3 years old
  2. 50 students are not between 3 and 4 years old
  3. There are 300 children in total
  Prove that the fraction of students who are 4 years old or older is 1/10 -/
theorem nursery_school_fraction (under_three : ℕ) (not_between_three_and_four : ℕ) (total : ℕ)
  (h1 : under_three = 20)
  (h2 : not_between_three_and_four = 50)
  (h3 : total = 300) :
  (not_between_three_and_four - under_three) / total = 1 / 10 := by
  sorry

end NUMINAMATH_CALUDE_nursery_school_fraction_l2063_206323


namespace NUMINAMATH_CALUDE_unique_positive_solution_l2063_206353

theorem unique_positive_solution :
  ∃! x : ℝ, x > 0 ∧ x^10 + 7*x^9 + 14*x^8 + 1729*x^7 - 1379*x^6 = 0 :=
by sorry

end NUMINAMATH_CALUDE_unique_positive_solution_l2063_206353


namespace NUMINAMATH_CALUDE_total_jars_is_72_l2063_206351

/-- Represents the number of jars of each size -/
def num_jars : ℕ := 24

/-- Represents the total volume of water in gallons -/
def total_volume : ℚ := 42

/-- Represents the volume of a quart jar in gallons -/
def quart_volume : ℚ := 1/4

/-- Represents the volume of a half-gallon jar in gallons -/
def half_gallon_volume : ℚ := 1/2

/-- Represents the volume of a one-gallon jar in gallons -/
def gallon_volume : ℚ := 1

/-- The theorem stating that given the conditions, the total number of jars is 72 -/
theorem total_jars_is_72 :
  (num_jars : ℚ) * (quart_volume + half_gallon_volume + gallon_volume) = total_volume ∧
  num_jars * 3 = 72 := by
  sorry

#check total_jars_is_72

end NUMINAMATH_CALUDE_total_jars_is_72_l2063_206351


namespace NUMINAMATH_CALUDE_correct_operations_l2063_206388

theorem correct_operations (x : ℝ) : 
  (x / 9 - 20 = 8) → (x * 9 + 20 = 2288) := by
  sorry

end NUMINAMATH_CALUDE_correct_operations_l2063_206388


namespace NUMINAMATH_CALUDE_two_year_increase_l2063_206301

/-- Calculates the final amount after a given number of years with a fixed annual increase rate. -/
def finalAmount (initialValue : ℝ) (increaseRate : ℝ) (years : ℕ) : ℝ :=
  initialValue * (1 + increaseRate) ^ years

/-- Theorem stating that an initial amount of 59,000, increasing by 1/8 of itself annually, 
    will result in 74,671.875 after 2 years. -/
theorem two_year_increase : 
  let initialValue : ℝ := 59000
  let increaseRate : ℝ := 1/8
  let years : ℕ := 2
  finalAmount initialValue increaseRate years = 74671.875 := by
sorry

end NUMINAMATH_CALUDE_two_year_increase_l2063_206301


namespace NUMINAMATH_CALUDE_car_speed_problem_l2063_206318

theorem car_speed_problem (highway_length : ℝ) (meeting_time : ℝ) (speed_car2 : ℝ) :
  highway_length = 45 →
  meeting_time = 1.5 →
  speed_car2 = 16 →
  ∃ (speed_car1 : ℝ), 
    speed_car1 * meeting_time + speed_car2 * meeting_time = highway_length ∧ 
    speed_car1 = 14 := by
  sorry

end NUMINAMATH_CALUDE_car_speed_problem_l2063_206318


namespace NUMINAMATH_CALUDE_spinster_cat_ratio_l2063_206311

theorem spinster_cat_ratio : 
  ∀ (spinsters cats : ℕ),
  spinsters = 18 →
  cats = spinsters + 63 →
  ∃ (n : ℕ), spinsters * n = 2 * cats →
  (spinsters : ℚ) / (cats : ℚ) = 2 / 9 := by
sorry

end NUMINAMATH_CALUDE_spinster_cat_ratio_l2063_206311


namespace NUMINAMATH_CALUDE_lowest_score_problem_l2063_206364

theorem lowest_score_problem (scores : List ℝ) (highest_score lowest_score : ℝ) : 
  scores.length = 15 →
  scores.sum / scores.length = 75 →
  highest_score ∈ scores →
  lowest_score ∈ scores →
  highest_score = 95 →
  (scores.sum - highest_score - lowest_score) / (scores.length - 2) = 78 →
  lowest_score = 16 :=
by sorry

end NUMINAMATH_CALUDE_lowest_score_problem_l2063_206364


namespace NUMINAMATH_CALUDE_quadratic_inequality_l2063_206367

theorem quadratic_inequality (x : ℝ) : x^2 - 9*x + 14 ≤ 0 ↔ x ∈ Set.Icc 2 7 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_l2063_206367


namespace NUMINAMATH_CALUDE_root_fraction_to_power_l2063_206369

theorem root_fraction_to_power : (81 ^ (1/3)) / (81 ^ (1/4)) = 81 ^ (1/12) := by
  sorry

end NUMINAMATH_CALUDE_root_fraction_to_power_l2063_206369


namespace NUMINAMATH_CALUDE_max_value_expression_l2063_206347

theorem max_value_expression (a b c d : ℝ) 
  (ha : -8 ≤ a ∧ a ≤ 8) 
  (hb : -8 ≤ b ∧ b ≤ 8) 
  (hc : -8 ≤ c ∧ c ≤ 8) 
  (hd : -8 ≤ d ∧ d ≤ 8) : 
  (∀ x y z w, -8 ≤ x ∧ x ≤ 8 → -8 ≤ y ∧ y ≤ 8 → -8 ≤ z ∧ z ≤ 8 → -8 ≤ w ∧ w ≤ 8 → 
    x + 2*y + z + 2*w - x*y - y*z - z*w - w*x ≤ 272) ∧ 
  (∃ x y z w, -8 ≤ x ∧ x ≤ 8 ∧ -8 ≤ y ∧ y ≤ 8 ∧ -8 ≤ z ∧ z ≤ 8 ∧ -8 ≤ w ∧ w ≤ 8 ∧
    x + 2*y + z + 2*w - x*y - y*z - z*w - w*x = 272) := by
  sorry

end NUMINAMATH_CALUDE_max_value_expression_l2063_206347


namespace NUMINAMATH_CALUDE_two_digit_times_eleven_l2063_206326

theorem two_digit_times_eleven (A B : ℕ) (h : A + B ≥ 10) :
  (10 * A + B) * 11 = 100 * (A + 1) + 10 * (A + B - 10) + B := by
  sorry

end NUMINAMATH_CALUDE_two_digit_times_eleven_l2063_206326


namespace NUMINAMATH_CALUDE_solution_set_f_geq_1_max_value_f_minus_quadratic_range_of_m_l2063_206378

-- Define the function f
def f (x : ℝ) : ℝ := |x + 1| - |x - 2|

-- Theorem for the solution set of f(x) ≥ 1
theorem solution_set_f_geq_1 : 
  {x : ℝ | f x ≥ 1} = {x : ℝ | x ≥ 1} := by sorry

-- Theorem for the maximum value of f(x) - x^2 + x
theorem max_value_f_minus_quadratic :
  ∃ (x : ℝ), ∀ (y : ℝ), f y - y^2 + y ≤ f x - x^2 + x ∧ f x - x^2 + x = 5/4 := by sorry

-- Theorem for the range of m
theorem range_of_m (m : ℝ) :
  (∃ (x : ℝ), f x ≥ x^2 - x + m) ↔ m ≤ 5/4 := by sorry

end NUMINAMATH_CALUDE_solution_set_f_geq_1_max_value_f_minus_quadratic_range_of_m_l2063_206378


namespace NUMINAMATH_CALUDE_quadratic_equation_iff_m_eq_neg_one_l2063_206393

/-- The equation is quadratic if and only if m = -1 -/
theorem quadratic_equation_iff_m_eq_neg_one (m : ℝ) : 
  (∀ x, (m - 1) * x^(m^2 + 1) - x - 2 = 0 ↔ ∃ a b c, a ≠ 0 ∧ a * x^2 + b * x + c = 0) ↔ 
  m = -1 := by
sorry

end NUMINAMATH_CALUDE_quadratic_equation_iff_m_eq_neg_one_l2063_206393


namespace NUMINAMATH_CALUDE_fraction_division_l2063_206387

theorem fraction_division : (3 / 4) / (5 / 8) = 6 / 5 := by
  sorry

end NUMINAMATH_CALUDE_fraction_division_l2063_206387


namespace NUMINAMATH_CALUDE_systematic_sampling_interval_2005_20_l2063_206333

/-- Given a population size and a desired sample size, calculate the sampling interval for systematic sampling -/
def systematicSamplingInterval (populationSize : ℕ) (sampleSize : ℕ) : ℕ :=
  (populationSize / sampleSize : ℕ)

/-- Theorem: For a population of 2005 numbers and a sample size of 20, the systematic sampling interval is 100 -/
theorem systematic_sampling_interval_2005_20 :
  systematicSamplingInterval 2005 20 = 100 := by
  sorry

end NUMINAMATH_CALUDE_systematic_sampling_interval_2005_20_l2063_206333


namespace NUMINAMATH_CALUDE_sum_of_coefficients_l2063_206319

theorem sum_of_coefficients (a₀ a₁ a₂ a₃ a₄ a₅ : ℝ) :
  (∀ x, (1 + 2*x)^5 = a₀ + a₁*x + a₂*x^2 + a₃*x^3 + a₄*x^4 + a₅*x^5) →
  a₀ + a₂ + a₄ = 121 := by
sorry

end NUMINAMATH_CALUDE_sum_of_coefficients_l2063_206319


namespace NUMINAMATH_CALUDE_like_terms_imply_xy_value_l2063_206341

theorem like_terms_imply_xy_value (a b : ℝ) (x y : ℝ) :
  (∃ k : ℝ, k ≠ 0 ∧ 2 * a^x * b^3 = k * (-a^2 * b^(1-y))) →
  x * y = -4 :=
sorry

end NUMINAMATH_CALUDE_like_terms_imply_xy_value_l2063_206341


namespace NUMINAMATH_CALUDE_product_increase_l2063_206330

theorem product_increase (a b : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) 
  (h : (a + 1) * (b + 1) = 2 * a * b) : 
  (a^2 - 1) * (b^2 - 1) = 4 * a * b := by
sorry

end NUMINAMATH_CALUDE_product_increase_l2063_206330


namespace NUMINAMATH_CALUDE_functional_equation_solution_l2063_206380

/-- A function satisfying the given functional equation -/
def SatisfiesEquation (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, f (f x ^ 2 + f y) = x * f x + y

/-- The main theorem stating that any function satisfying the equation
    must be either the identity function or the negation function -/
theorem functional_equation_solution (f : ℝ → ℝ) 
    (h : SatisfiesEquation f) : 
    (∀ x : ℝ, f x = x) ∨ (∀ x : ℝ, f x = -x) := by
  sorry

end NUMINAMATH_CALUDE_functional_equation_solution_l2063_206380


namespace NUMINAMATH_CALUDE_snack_package_average_l2063_206309

theorem snack_package_average (cookie_counts : List ℕ)
  (candy_counts : List ℕ) (pie_counts : List ℕ) :
  cookie_counts.length = 4 →
  candy_counts.length = 3 →
  pie_counts.length = 2 →
  cookie_counts.sum + candy_counts.sum + pie_counts.sum = 153 →
  cookie_counts.sum / cookie_counts.length = 17 →
  (cookie_counts.sum + candy_counts.sum + pie_counts.sum) /
    (cookie_counts.length + candy_counts.length + pie_counts.length) = 17 :=
by
  sorry

end NUMINAMATH_CALUDE_snack_package_average_l2063_206309


namespace NUMINAMATH_CALUDE_modular_inverse_13_mod_64_l2063_206332

theorem modular_inverse_13_mod_64 :
  ∃ x : ℕ, x < 64 ∧ (13 * x) % 64 = 1 :=
by
  use 5
  sorry

end NUMINAMATH_CALUDE_modular_inverse_13_mod_64_l2063_206332


namespace NUMINAMATH_CALUDE_alex_escalator_time_l2063_206365

/-- The time it takes Alex to walk down the non-moving escalator -/
def time_not_moving : ℝ := 75

/-- The time it takes Alex to walk down the moving escalator -/
def time_moving : ℝ := 30

/-- The time it takes Alex to ride the escalator without walking -/
def time_riding : ℝ := 50

theorem alex_escalator_time :
  (time_not_moving * time_moving) / (time_not_moving - time_moving) = time_riding := by
  sorry

end NUMINAMATH_CALUDE_alex_escalator_time_l2063_206365


namespace NUMINAMATH_CALUDE_monthly_compound_interest_greater_than_yearly_l2063_206372

theorem monthly_compound_interest_greater_than_yearly :
  1 + 5 / 100 < (1 + 5 / (12 * 100)) ^ 12 := by
  sorry

end NUMINAMATH_CALUDE_monthly_compound_interest_greater_than_yearly_l2063_206372


namespace NUMINAMATH_CALUDE_complex_sum_problem_l2063_206325

theorem complex_sum_problem (a b c d e f g h : ℂ) :
  b = 2 ∧ 
  g = -a - c - e ∧ 
  a + b * Complex.I + c + d * Complex.I + e + f * Complex.I + g + h * Complex.I = -3 * Complex.I →
  d + f + h = -5 := by
  sorry

end NUMINAMATH_CALUDE_complex_sum_problem_l2063_206325


namespace NUMINAMATH_CALUDE_cube_less_than_triple_l2063_206310

theorem cube_less_than_triple : ∃! (x : ℤ), x^3 < 3*x :=
sorry

end NUMINAMATH_CALUDE_cube_less_than_triple_l2063_206310


namespace NUMINAMATH_CALUDE_blocks_left_l2063_206308

/-- Given that Randy had 59 blocks initially and used 36 blocks to build a tower,
    prove that he has 23 blocks left. -/
theorem blocks_left (initial_blocks : ℕ) (used_blocks : ℕ) 
  (h1 : initial_blocks = 59)
  (h2 : used_blocks = 36) :
  initial_blocks - used_blocks = 23 :=
by sorry

end NUMINAMATH_CALUDE_blocks_left_l2063_206308


namespace NUMINAMATH_CALUDE_six_digit_square_numbers_l2063_206312

/-- Represents a 6-digit number as a tuple of its digits -/
def SixDigitNumber := (Nat × Nat × Nat × Nat × Nat × Nat)

/-- Converts a 6-digit number tuple to its numerical value -/
def toNumber (n : SixDigitNumber) : Nat :=
  match n with
  | (a, b, c, d, e, f) => 100000 * a + 10000 * b + 1000 * c + 100 * d + 10 * e + f

/-- Extracts the last three digits of a 6-digit number tuple -/
def lastThreeDigits (n : SixDigitNumber) : Nat :=
  match n with
  | (_, _, _, d, e, f) => 100 * d + 10 * e + f

/-- Checks if a given 6-digit number satisfies the condition (abcdef) = (def)^2 -/
def satisfiesCondition (n : SixDigitNumber) : Prop :=
  toNumber n = (lastThreeDigits n) ^ 2

theorem six_digit_square_numbers :
  ∀ n : SixDigitNumber,
    satisfiesCondition n →
    (toNumber n = 390625 ∨ toNumber n = 141376) :=
by sorry

end NUMINAMATH_CALUDE_six_digit_square_numbers_l2063_206312


namespace NUMINAMATH_CALUDE_ball_hits_ground_time_l2063_206315

theorem ball_hits_ground_time :
  ∃ t : ℝ, t > 0 ∧ -8 * t^2 - 12 * t + 72 = 0 ∧ abs (t - 2.34) < 0.01 := by
  sorry

end NUMINAMATH_CALUDE_ball_hits_ground_time_l2063_206315


namespace NUMINAMATH_CALUDE_inequality_proof_l2063_206376

theorem inequality_proof (a b c : ℝ) 
  (ha : a < 0) 
  (hb : b < 0) 
  (hab : a < b) 
  (hbc : b < c) : 
  a + b < b + c := by
sorry

end NUMINAMATH_CALUDE_inequality_proof_l2063_206376


namespace NUMINAMATH_CALUDE_swimmer_problem_l2063_206329

/-- Swimmer problem -/
theorem swimmer_problem (a s r : ℝ) (ha : a > 0) (hs : s > 0) (hr : r > 0) 
  (h_order : s < r ∧ r < (100 * s) / (50 + s)) :
  ∃ (x z : ℝ),
    x = (100 * s - 50 * r - r * s) / ((3 * s - r) * a) ∧
    z = (100 * s - 50 * r - r * s) / ((r - s) * a) ∧
    x > 0 ∧ z > 0 ∧
    ∃ (y t : ℝ),
      y > 0 ∧ t > 0 ∧
      t * z = (t + a) * y ∧
      t * z = (t + 2 * a) * x ∧
      (50 + r) / z = (50 - r) / x - 2 * a ∧
      (50 + s) / z = (50 - s) / y - a :=
by
  sorry

end NUMINAMATH_CALUDE_swimmer_problem_l2063_206329


namespace NUMINAMATH_CALUDE_vertex_angle_is_45_degrees_l2063_206338

/-- An isosceles triangle with specific properties -/
structure SpecialIsoscelesTriangle where
  a : ℝ  -- Length of congruent sides
  s : ℝ  -- Semi-perimeter
  h : ℝ  -- Height to the base
  b : ℝ  -- Length of the base
  a_pos : 0 < a  -- Side length is positive
  s_pos : 0 < s  -- Semi-perimeter is positive
  h_pos : 0 < h  -- Height is positive
  b_pos : 0 < b  -- Base length is positive
  isosceles : s = a + b/2  -- Definition of semi-perimeter for this triangle
  right_base_angle : h = a  -- One base angle is a right angle
  area_condition : b * (2 * h) = s^2  -- Given condition

/-- The vertex angle at the base of the special isosceles triangle is 45° -/
theorem vertex_angle_is_45_degrees (t : SpecialIsoscelesTriangle) : 
  Real.arccos ((t.b / 2) / t.a) * (180 / Real.pi) = 45 := by
  sorry

end NUMINAMATH_CALUDE_vertex_angle_is_45_degrees_l2063_206338


namespace NUMINAMATH_CALUDE_parabola_properties_l2063_206305

/-- Parabola represented by y = -3x^2 - 6x + 2 -/
def parabola (x : ℝ) : ℝ := -3 * x^2 - 6 * x + 2

theorem parabola_properties :
  (∃ (x_max y_max : ℝ),
    (∀ x, parabola x ≤ parabola x_max) ∧
    parabola x_max = y_max ∧
    x_max = 1 ∧
    y_max = -7) ∧
  (∀ x, parabola (2 - x) = parabola x) := by
  sorry

end NUMINAMATH_CALUDE_parabola_properties_l2063_206305


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l2063_206344

def A : Set Int := {-1, 0, 1}
def B : Set Int := {0, 1, 2}

theorem intersection_of_A_and_B : A ∩ B = {0, 1} := by
  sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l2063_206344


namespace NUMINAMATH_CALUDE_parallelogram_sides_sum_l2063_206373

theorem parallelogram_sides_sum (x y : ℝ) : 
  (4*x + 4 = 18) → (15*y - 3 = 12) → x + y = 4.5 := by
  sorry

end NUMINAMATH_CALUDE_parallelogram_sides_sum_l2063_206373


namespace NUMINAMATH_CALUDE_sum_of_squares_l2063_206313

theorem sum_of_squares (a b c : ℝ) 
  (eq1 : a^2 + 3*b = 14)
  (eq2 : b^2 + 5*c = -13)
  (eq3 : c^2 + 7*a = -26) :
  a^2 + b^2 + c^2 = 20.75 := by
sorry

end NUMINAMATH_CALUDE_sum_of_squares_l2063_206313


namespace NUMINAMATH_CALUDE_gcf_of_4140_and_9920_l2063_206339

theorem gcf_of_4140_and_9920 : Nat.gcd 4140 9920 = 10 := by
  sorry

end NUMINAMATH_CALUDE_gcf_of_4140_and_9920_l2063_206339


namespace NUMINAMATH_CALUDE_removed_digit_not_power_of_two_l2063_206352

def is_power_of_two (n : ℕ) : Prop := ∃ k : ℕ, n = 2^k

def remove_middle_digit (n : ℕ) : ℕ := 
  -- Implementation details omitted
  sorry

theorem removed_digit_not_power_of_two (N : ℕ) (h : is_power_of_two N) :
  ¬ is_power_of_two (remove_middle_digit N) := by
  sorry

end NUMINAMATH_CALUDE_removed_digit_not_power_of_two_l2063_206352


namespace NUMINAMATH_CALUDE_set_operations_l2063_206382

def A : Set ℝ := {x | -5 ≤ x ∧ x ≤ 3}
def B : Set ℝ := {x | x < -2 ∨ x > 4}

theorem set_operations (x : ℝ) :
  (x ∈ A ∩ B ↔ -5 ≤ x ∧ x < -2) ∧
  (x ∈ (Set.univ \ A) ∩ B ↔ x < -5 ∨ x > 4) ∧
  (x ∈ (Set.univ \ A) ∩ (Set.univ \ B) ↔ 3 < x ∧ x ≤ 4) := by
  sorry

end NUMINAMATH_CALUDE_set_operations_l2063_206382


namespace NUMINAMATH_CALUDE_probability_rain_at_least_one_day_l2063_206327

/-- The probability of rain on at least one day given independent probabilities for each day -/
theorem probability_rain_at_least_one_day 
  (p_friday p_saturday p_sunday : ℝ) 
  (h_friday : p_friday = 0.3)
  (h_saturday : p_saturday = 0.45)
  (h_sunday : p_sunday = 0.55)
  (h_independent : True) -- Assumption of independence
  : 1 - (1 - p_friday) * (1 - p_saturday) * (1 - p_sunday) = 0.82675 := by
  sorry

end NUMINAMATH_CALUDE_probability_rain_at_least_one_day_l2063_206327


namespace NUMINAMATH_CALUDE_existence_of_number_without_seven_l2063_206392

/-- A function that checks if a natural number contains the digit 7 in its decimal representation -/
def containsSeven (n : ℕ) : Prop := sorry

/-- A function that represents multiplying a number by 5 k times -/
def multiplyByFiveKTimes (n : ℕ) (k : ℕ) : List ℕ := sorry

/-- A function that represents multiplying a number by 2 k times -/
def multiplyByTwoKTimes (n : ℕ) (k : ℕ) : List ℕ := sorry

theorem existence_of_number_without_seven 
  (n : ℕ) (k : ℕ) 
  (h : ∀ x ∈ multiplyByFiveKTimes n k, ¬containsSeven x) :
  ∃ m : ℕ, ∀ y ∈ multiplyByTwoKTimes m k, ¬containsSeven y :=
sorry

end NUMINAMATH_CALUDE_existence_of_number_without_seven_l2063_206392


namespace NUMINAMATH_CALUDE_complex_equation_solution_l2063_206302

theorem complex_equation_solution : ∃ (x y : ℝ), (2*x - 1 : ℂ) + I = y - (2 - y)*I ∧ x = 2 ∧ y = 3 := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l2063_206302


namespace NUMINAMATH_CALUDE_jessica_bank_account_l2063_206360

theorem jessica_bank_account (B : ℝ) : 
  B > 0 →
  (3/5) * B = B - 200 →
  ∃ x y : ℝ, x > 0 ∧ y > 0 ∧ 
    (3/5) * B + (x/y) * ((3/5) * B) = 450 →
    x/y = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_jessica_bank_account_l2063_206360


namespace NUMINAMATH_CALUDE_cement_theft_proof_l2063_206322

/-- Represents the weight of cement bags in kilograms -/
structure BagWeight where
  small : Nat
  large : Nat

/-- Represents the number of cement bags -/
structure BagCount where
  small : Nat
  large : Nat

/-- Calculates the total weight of cement given bag weights and counts -/
def totalWeight (w : BagWeight) (c : BagCount) : Nat :=
  w.small * c.small + w.large * c.large

/-- Represents the manager's assumption of bag weight -/
def managerAssumedWeight : Nat := 25

theorem cement_theft_proof (w : BagWeight) (c : BagCount) 
  (h1 : w.small = 25)
  (h2 : w.large = 40)
  (h3 : c.small = 2 * c.large)
  (h4 : totalWeight w c - w.large * 60 = managerAssumedWeight * (c.small + c.large)) :
  totalWeight w c - w.large * 60 = 12000 := by
  sorry

end NUMINAMATH_CALUDE_cement_theft_proof_l2063_206322


namespace NUMINAMATH_CALUDE_complete_gear_exists_l2063_206371

/-- Represents a gear with a certain number of teeth and missing teeth positions -/
structure Gear where
  num_teeth : Nat
  missing_teeth : Finset Nat

/-- The problem statement -/
theorem complete_gear_exists (gear1 gear2 : Gear)
  (h1 : gear1.num_teeth = 14)
  (h2 : gear2.num_teeth = 14)
  (h3 : gear1.missing_teeth.card = 4)
  (h4 : gear2.missing_teeth.card = 4) :
  ∃ (rotation : Nat), ∀ (pos : Nat),
    pos ∈ gear1.missing_teeth →
    (pos + rotation) % gear1.num_teeth ∉ gear2.missing_teeth :=
sorry

end NUMINAMATH_CALUDE_complete_gear_exists_l2063_206371


namespace NUMINAMATH_CALUDE_expression_evaluation_l2063_206385

theorem expression_evaluation (x y : ℕ) (h1 : x = 3) (h2 : y = 4) :
  5 * x^y + 6 * y^x = 789 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l2063_206385


namespace NUMINAMATH_CALUDE_holly_blood_pressure_pills_l2063_206381

/-- Represents the number of pills Holly takes daily for each medication type -/
structure DailyPills where
  insulin : ℕ
  bloodPressure : ℕ
  anticonvulsant : ℕ

/-- Calculates the total number of pills taken in a week -/
def weeklyTotal (d : DailyPills) : ℕ :=
  7 * (d.insulin + d.bloodPressure + d.anticonvulsant)

theorem holly_blood_pressure_pills
  (d : DailyPills)
  (h1 : d.insulin = 2)
  (h2 : d.anticonvulsant = 2 * d.bloodPressure)
  (h3 : weeklyTotal d = 77) :
  d.bloodPressure = 3 := by
sorry

end NUMINAMATH_CALUDE_holly_blood_pressure_pills_l2063_206381


namespace NUMINAMATH_CALUDE_number_of_boys_l2063_206348

theorem number_of_boys (total_pupils : ℕ) (number_of_girls : ℕ) 
  (h1 : total_pupils = 929) 
  (h2 : number_of_girls = 542) : 
  total_pupils - number_of_girls = 387 := by
  sorry

end NUMINAMATH_CALUDE_number_of_boys_l2063_206348
