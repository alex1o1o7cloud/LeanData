import Mathlib

namespace NUMINAMATH_CALUDE_median_lengths_l364_36456

/-- Given a triangle with sides a, b, and c, this theorem states the formulas for the lengths of the medians sa, sb, and sc. -/
theorem median_lengths (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) 
  (h_triangle : a + b > c ∧ b + c > a ∧ c + a > b) :
  ∃ (sa sb sc : ℝ),
    sa = Real.sqrt (2 * b^2 + 2 * c^2 - a^2) / 2 ∧
    sb = Real.sqrt (2 * a^2 + 2 * c^2 - b^2) / 2 ∧
    sc = Real.sqrt (2 * a^2 + 2 * b^2 - c^2) / 2 :=
by sorry


end NUMINAMATH_CALUDE_median_lengths_l364_36456


namespace NUMINAMATH_CALUDE_line_segment_lengths_l364_36464

/-- Given points A, B, and C on a line, prove that if AB = 5 and AC = BC + 1, then AC = 3 and BC = 2 -/
theorem line_segment_lengths (A B C : ℝ) (h1 : |B - A| = 5) (h2 : |C - A| = |C - B| + 1) :
  |C - A| = 3 ∧ |C - B| = 2 := by
  sorry

end NUMINAMATH_CALUDE_line_segment_lengths_l364_36464


namespace NUMINAMATH_CALUDE_perpendicular_line_angle_of_inclination_l364_36417

theorem perpendicular_line_angle_of_inclination 
  (line_eq : ℝ → ℝ → Prop) 
  (h_line_eq : ∀ x y, line_eq x y ↔ x + Real.sqrt 3 * y + 2 = 0) :
  ∃ θ : ℝ, 
    0 ≤ θ ∧ 
    θ < π ∧ 
    (∀ x y, line_eq x y → 
      ∃ m : ℝ, m * Real.tan θ = -1 ∧ 
      ∀ x' y', y' - y = m * (x' - x)) ∧ 
    θ = π / 3 :=
sorry

end NUMINAMATH_CALUDE_perpendicular_line_angle_of_inclination_l364_36417


namespace NUMINAMATH_CALUDE_max_intersections_three_polygons_l364_36483

/-- Represents a convex polygon with a given number of sides -/
structure ConvexPolygon where
  sides : ℕ

/-- Theorem stating the maximum number of intersections among three convex polygons -/
theorem max_intersections_three_polygons
  (P1 P2 P3 : ConvexPolygon)
  (h1 : P1.sides ≤ P2.sides)
  (h2 : P2.sides ≤ P3.sides)
  (h_no_shared_segments : True)  -- Represents the condition that polygons don't share line segments
  : ℕ := by
  sorry

end NUMINAMATH_CALUDE_max_intersections_three_polygons_l364_36483


namespace NUMINAMATH_CALUDE_prob_green_ball_l364_36434

-- Define the containers and their contents
structure Container where
  red_balls : Nat
  green_balls : Nat

-- Define the probabilities
def prob_container : Rat := 1 / 3
def prob_green (c : Container) : Rat := c.green_balls / (c.red_balls + c.green_balls)

-- Define the containers
def container_A : Container := ⟨5, 5⟩
def container_B : Container := ⟨7, 3⟩
def container_C : Container := ⟨6, 4⟩

-- State the theorem
theorem prob_green_ball : 
  prob_container * prob_green container_A +
  prob_container * prob_green container_B +
  prob_container * prob_green container_C = 2 / 5 := by
  sorry


end NUMINAMATH_CALUDE_prob_green_ball_l364_36434


namespace NUMINAMATH_CALUDE_rectangle_point_distances_l364_36404

-- Define the rectangle and point P
def Rectangle (A B C D : ℝ × ℝ) : Prop :=
  -- Add conditions for a rectangle here
  True

def InsideRectangle (P : ℝ × ℝ) (A B C D : ℝ × ℝ) : Prop :=
  -- Add conditions for P being inside the rectangle here
  True

-- Define the distance function
def distance (P Q : ℝ × ℝ) : ℝ :=
  -- Add definition for Euclidean distance here
  0

-- Theorem statement
theorem rectangle_point_distances 
  (A B C D P : ℝ × ℝ) 
  (h_rect : Rectangle A B C D) 
  (h_inside : InsideRectangle P A B C D) 
  (h_PA : distance P A = 5)
  (h_PD : distance P D = 12)
  (h_PC : distance P C = 13) :
  distance P B = 5 * Real.sqrt 2 :=
sorry

end NUMINAMATH_CALUDE_rectangle_point_distances_l364_36404


namespace NUMINAMATH_CALUDE_quadratic_sum_l364_36405

/-- Given a quadratic expression 5x^2 - 20x + 8, when expressed in the form a(x - h)^2 + k,
    the sum a + h + k equals -5. -/
theorem quadratic_sum (x : ℝ) :
  ∃ (a h k : ℝ), (5 * x^2 - 20 * x + 8 = a * (x - h)^2 + k) ∧ (a + h + k = -5) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_sum_l364_36405


namespace NUMINAMATH_CALUDE_real_part_of_z_squared_neg_four_l364_36426

theorem real_part_of_z_squared_neg_four (z : ℂ) : z^2 = -4 → Complex.re z = 0 := by
  sorry

end NUMINAMATH_CALUDE_real_part_of_z_squared_neg_four_l364_36426


namespace NUMINAMATH_CALUDE_books_sold_in_garage_sale_l364_36407

theorem books_sold_in_garage_sale :
  ∀ (initial_books given_to_friend remaining_books : ℕ),
    initial_books = 108 →
    given_to_friend = 35 →
    remaining_books = 62 →
    initial_books - given_to_friend - remaining_books = 11 :=
by
  sorry

end NUMINAMATH_CALUDE_books_sold_in_garage_sale_l364_36407


namespace NUMINAMATH_CALUDE_sugar_for_cake_l364_36408

/-- Given a cake recipe that requires a total of 0.8 cups of sugar,
    with 0.6 cups used for frosting, prove that the cake itself
    requires 0.2 cups of sugar. -/
theorem sugar_for_cake
  (total_sugar : ℝ)
  (frosting_sugar : ℝ)
  (h1 : total_sugar = 0.8)
  (h2 : frosting_sugar = 0.6) :
  total_sugar - frosting_sugar = 0.2 :=
by sorry

end NUMINAMATH_CALUDE_sugar_for_cake_l364_36408


namespace NUMINAMATH_CALUDE_unique_prime_solution_l364_36441

theorem unique_prime_solution : 
  ∀ p q r : ℕ, 
    Prime p ∧ Prime q ∧ Prime r →
    p^2 + 1 = 74 * (q^2 + r^2) →
    p = 31 ∧ q = 2 ∧ r = 3 :=
by
  sorry

end NUMINAMATH_CALUDE_unique_prime_solution_l364_36441


namespace NUMINAMATH_CALUDE_max_value_problem_l364_36491

theorem max_value_problem (a b : ℝ) (ha : a > 0) (hb : b > 0) (h_sum : 2 * a + b = 1) :
  (∃ (x y : ℝ), x > 0 ∧ y > 0 ∧ 2 * x + y = 1 ∧
    2 * Real.sqrt (a * b) - 4 * a^2 - b^2 ≤ 2 * Real.sqrt (x * y) - 4 * x^2 - y^2) ∧
  2 * Real.sqrt (a * b) - 4 * a^2 - b^2 ≤ (Real.sqrt 2 - 1) / 2 :=
sorry

end NUMINAMATH_CALUDE_max_value_problem_l364_36491


namespace NUMINAMATH_CALUDE_solution_set_of_inequality_l364_36490

open Set

theorem solution_set_of_inequality (f : ℝ → ℝ) (f' : ℝ → ℝ) :
  (∀ x, f (-x) = -f x) →  -- f is odd
  (∀ x, HasDerivAt f (f' x) x) →  -- f' is the derivative of f
  (∀ x > 0, x * f' x + 3 * f x > 0) →  -- given condition
  {x : ℝ | x^3 * f x + (2*x - 1)^3 * f (1 - 2*x) < 0} = Iic (1/3) ∪ Ioi 1 :=
sorry

end NUMINAMATH_CALUDE_solution_set_of_inequality_l364_36490


namespace NUMINAMATH_CALUDE_area_of_region_l364_36419

/-- The region defined by the inequality |4x-24|+|3y+10| ≤ 6 -/
def Region : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | |4 * p.1 - 24| + |3 * p.2 + 10| ≤ 6}

/-- The area of a set in ℝ² -/
noncomputable def area (S : Set (ℝ × ℝ)) : ℝ := sorry

/-- The theorem stating that the area of the region is 12 -/
theorem area_of_region : area Region = 12 := by sorry

end NUMINAMATH_CALUDE_area_of_region_l364_36419


namespace NUMINAMATH_CALUDE_hyperbola_equation_l364_36448

-- Define the hyperbola
def hyperbola (x y : ℝ) : Prop := x^2 - y^2 / 3 = 1

-- Define the asymptotes
def asymptotes (x y : ℝ) : Prop := y = Real.sqrt 3 * x ∨ y = -Real.sqrt 3 * x

-- Theorem statement
theorem hyperbola_equation :
  (∀ x y : ℝ, asymptotes x y → (∃ a b : ℝ, x^2 / a^2 - y^2 / b^2 = 1 ∧ b^2 = 3 * a^2)) →
  hyperbola 2 3 →
  ∀ x y : ℝ, hyperbola x y ↔ x^2 - y^2 / 3 = 1 :=
sorry

end NUMINAMATH_CALUDE_hyperbola_equation_l364_36448


namespace NUMINAMATH_CALUDE_alpha_plus_beta_eq_107_l364_36454

theorem alpha_plus_beta_eq_107 :
  ∃ α β : ℝ, (∀ x : ℝ, (x - α) / (x + β) = (x^2 - 64*x + 992) / (x^2 + 72*x - 2184)) →
  α + β = 107 := by
  sorry

end NUMINAMATH_CALUDE_alpha_plus_beta_eq_107_l364_36454


namespace NUMINAMATH_CALUDE_max_b_value_l364_36421

theorem max_b_value (y : ℤ) (b : ℕ+) (h : y^2 + b * y = -21) :
  b ≤ 22 ∧ ∃ y : ℤ, y^2 + 22 * y = -21 :=
sorry

end NUMINAMATH_CALUDE_max_b_value_l364_36421


namespace NUMINAMATH_CALUDE_smartphone_savings_smartphone_savings_proof_l364_36440

/-- Calculates the weekly savings required to purchase a smartphone --/
theorem smartphone_savings (smartphone_cost current_savings : ℚ) : ℚ :=
  let remaining_amount := smartphone_cost - current_savings
  let weeks_in_two_months := 2 * (52 / 12 : ℚ)
  remaining_amount / weeks_in_two_months

/-- Proves that the weekly savings for the given scenario is approximately $13.86 --/
theorem smartphone_savings_proof :
  ∃ (ε : ℚ), ε > 0 ∧ ε < 0.01 ∧ |smartphone_savings 160 40 - (13.86 : ℚ)| < ε :=
sorry

end NUMINAMATH_CALUDE_smartphone_savings_smartphone_savings_proof_l364_36440


namespace NUMINAMATH_CALUDE_five_dollar_neg_three_l364_36495

-- Define the $ operation
def dollar_op (a b : Int) : Int := a * (b - 1) + a * b

-- Theorem statement
theorem five_dollar_neg_three : dollar_op 5 (-3) = -35 := by
  sorry

end NUMINAMATH_CALUDE_five_dollar_neg_three_l364_36495


namespace NUMINAMATH_CALUDE_triangle_perimeter_range_l364_36445

open Real

theorem triangle_perimeter_range (A B C a b c : ℝ) : 
  -- Triangle ABC is acute
  0 < A ∧ A < π/2 ∧ 0 < B ∧ B < π/2 ∧ 0 < C ∧ C < π/2 ∧
  -- Sum of angles in a triangle
  A + B + C = π ∧
  -- Given equation
  cos B^2 + cos B * cos (A - C) = sin A * sin C ∧
  -- Side length a
  a = 2 * Real.sqrt 3 ∧
  -- Derived value of B
  B = π/3 ∧
  -- Definition of sides using sine rule
  b = a * sin B / sin A ∧
  c = a * sin C / sin A
  →
  -- Perimeter range
  3 + 3 * Real.sqrt 3 < a + b + c ∧ a + b + c < 6 + 6 * Real.sqrt 3 :=
by sorry

end NUMINAMATH_CALUDE_triangle_perimeter_range_l364_36445


namespace NUMINAMATH_CALUDE_special_line_equation_l364_36482

/-- A line passing through (6, -2) with x-intercept 1 greater than y-intercept -/
structure SpecialLine where
  /-- The equation of the line in the form ax + by + c = 0 -/
  a : ℝ
  b : ℝ
  c : ℝ
  /-- The line passes through (6, -2) -/
  point_condition : a * 6 + b * (-2) + c = 0
  /-- The x-intercept is 1 greater than the y-intercept -/
  intercept_condition : -c/a = -c/b + 1

/-- The equation of the special line is either x + 2y - 2 = 0 or 2x + 3y - 6 = 0 -/
theorem special_line_equation (l : SpecialLine) :
  (l.a = 1 ∧ l.b = 2 ∧ l.c = -2) ∨ (l.a = 2 ∧ l.b = 3 ∧ l.c = -6) :=
sorry

end NUMINAMATH_CALUDE_special_line_equation_l364_36482


namespace NUMINAMATH_CALUDE_work_left_fraction_l364_36457

theorem work_left_fraction (days_A days_B days_together : ℕ) (h1 : days_A = 20) (h2 : days_B = 30) (h3 : days_together = 4) :
  1 - (days_together : ℚ) * (1 / days_A + 1 / days_B) = 2/3 := by
  sorry

end NUMINAMATH_CALUDE_work_left_fraction_l364_36457


namespace NUMINAMATH_CALUDE_completing_square_result_l364_36415

theorem completing_square_result (x : ℝ) : 
  (x^2 - 4*x - 1 = 0) ↔ ((x - 2)^2 = 5) :=
by sorry

end NUMINAMATH_CALUDE_completing_square_result_l364_36415


namespace NUMINAMATH_CALUDE_max_xy_value_l364_36475

theorem max_xy_value (x y : ℝ) (hx : x > 0) (hy : y > 0) (h_sum : x + 4*y = 12) :
  xy ≤ 9 ∧ ∃ (x₀ y₀ : ℝ), x₀ > 0 ∧ y₀ > 0 ∧ x₀ + 4*y₀ = 12 ∧ x₀*y₀ = 9 :=
by sorry

end NUMINAMATH_CALUDE_max_xy_value_l364_36475


namespace NUMINAMATH_CALUDE_cloth_cost_price_l364_36461

/-- Represents the cost and profit scenario for cloth selling --/
structure ClothSelling where
  total_length : ℕ
  first_half : ℕ
  second_half : ℕ
  total_price : ℚ
  profit_first : ℚ
  profit_second : ℚ

/-- The theorem stating the cost price per meter if it's the same for both halves --/
theorem cloth_cost_price (cs : ClothSelling)
  (h_total : cs.total_length = 120)
  (h_half : cs.first_half = cs.second_half)
  (h_length : cs.first_half + cs.second_half = cs.total_length)
  (h_price : cs.total_price = 15360)
  (h_profit1 : cs.profit_first = 1/10)
  (h_profit2 : cs.profit_second = 1/5)
  (h_equal_cost : ∃ (c : ℚ), 
    cs.first_half * (1 + cs.profit_first) * c + 
    cs.second_half * (1 + cs.profit_second) * c = cs.total_price) :
  ∃ (c : ℚ), c = 11130 / 100 := by
  sorry

end NUMINAMATH_CALUDE_cloth_cost_price_l364_36461


namespace NUMINAMATH_CALUDE_problem_solution_l364_36438

def f (x : ℝ) : ℝ := |2*x + 3| + |x - 1|

theorem problem_solution :
  (∀ x : ℝ, f x > 4 ↔ x ∈ Set.Iio (-2) ∪ Set.Ioi 0) ∧
  (∀ m : ℝ, (∃ x₀ : ℝ, ∀ t : ℝ, f x₀ < |m + t| + |t - m|) ↔ 
    m ∈ Set.Iio (-5/4) ∪ Set.Ioi (5/4)) :=
by sorry

end NUMINAMATH_CALUDE_problem_solution_l364_36438


namespace NUMINAMATH_CALUDE_orange_caterpillar_length_l364_36432

theorem orange_caterpillar_length 
  (green_length : ℝ) 
  (length_difference : ℝ) 
  (h1 : green_length = 3) 
  (h2 : length_difference = 1.83) 
  (h3 : green_length = length_difference + orange_length) : 
  orange_length = 1.17 := by
  sorry

end NUMINAMATH_CALUDE_orange_caterpillar_length_l364_36432


namespace NUMINAMATH_CALUDE_smallest_consecutive_product_seven_consecutive_product_seven_is_smallest_l364_36460

theorem smallest_consecutive_product (n : ℕ) : n > 0 ∧ n * (n + 1) * (n + 2) * (n + 3) = 5040 → n ≥ 7 :=
by sorry

theorem seven_consecutive_product : 7 * 8 * 9 * 10 = 5040 :=
by sorry

theorem seven_is_smallest : ∃ (n : ℕ), n > 0 ∧ n * (n + 1) * (n + 2) * (n + 3) = 5040 ∧ n = 7 :=
by sorry

end NUMINAMATH_CALUDE_smallest_consecutive_product_seven_consecutive_product_seven_is_smallest_l364_36460


namespace NUMINAMATH_CALUDE_max_sum_of_factors_l364_36496

theorem max_sum_of_factors (A B C : ℕ+) : 
  A ≠ B → B ≠ C → A ≠ C → A * B * C = 3003 → 
  ∀ (X Y Z : ℕ+), X ≠ Y → Y ≠ Z → X ≠ Z → X * Y * Z = 3003 → 
  A + B + C ≤ X + Y + Z → A + B + C ≤ 117 :=
by sorry

end NUMINAMATH_CALUDE_max_sum_of_factors_l364_36496


namespace NUMINAMATH_CALUDE_tom_seashells_l364_36427

/-- The number of seashells Tom found -/
def total_seashells (broken : ℕ) (unbroken : ℕ) : ℕ :=
  broken + unbroken

/-- Theorem stating that Tom found 7 seashells in total -/
theorem tom_seashells : total_seashells 4 3 = 7 := by
  sorry

end NUMINAMATH_CALUDE_tom_seashells_l364_36427


namespace NUMINAMATH_CALUDE_partnership_investment_timing_l364_36416

theorem partnership_investment_timing 
  (x : ℝ) 
  (m : ℝ) 
  (total_gain : ℝ) 
  (a_share : ℝ) 
  (h1 : total_gain = 18600) 
  (h2 : a_share = 6200) 
  (h3 : a_share / total_gain = 1 / 3) 
  (h4 : x * 12 = (1 / 3) * (x * 12 + 2 * x * (12 - m) + 3 * x * 4)) : 
  m = 6 := by
  sorry

end NUMINAMATH_CALUDE_partnership_investment_timing_l364_36416


namespace NUMINAMATH_CALUDE_union_of_I_is_odd_integers_l364_36485

def I : ℕ → Set ℤ
  | 0 => {-1, 1}
  | n + 1 => {x : ℤ | ∃ y ∈ I n, x^2 - 2*x*y + y^2 = 4^(n + 1)}

def OddIntegers : Set ℤ := {x : ℤ | ∃ k : ℤ, x = 2*k + 1}

theorem union_of_I_is_odd_integers :
  (⋃ n : ℕ, I n) = OddIntegers :=
sorry

end NUMINAMATH_CALUDE_union_of_I_is_odd_integers_l364_36485


namespace NUMINAMATH_CALUDE_pure_imaginary_complex_number_l364_36451

theorem pure_imaginary_complex_number (a : ℝ) : 
  (Complex.I + 1) * (a + 2 * Complex.I) * Complex.I = Complex.I * (a + 2 : ℝ) → a = 2 := by
  sorry

end NUMINAMATH_CALUDE_pure_imaginary_complex_number_l364_36451


namespace NUMINAMATH_CALUDE_min_value_theorem_l364_36486

theorem min_value_theorem (a b : ℝ) (ha : 0 < a) (hb : 0 < b) (h : (1 / a) + (1 / b) = 1) :
  6 ≤ (4 / (a - 1)) + (9 / (b - 1)) ∧ 
  ∃ (a₀ b₀ : ℝ), 0 < a₀ ∧ 0 < b₀ ∧ (1 / a₀) + (1 / b₀) = 1 ∧ (4 / (a₀ - 1)) + (9 / (b₀ - 1)) = 6 := by
  sorry

end NUMINAMATH_CALUDE_min_value_theorem_l364_36486


namespace NUMINAMATH_CALUDE_quadratic_equation_roots_l364_36442

theorem quadratic_equation_roots (a b c : ℝ) : 
  (∀ x, a * x * (x + 1) + b * x * (x + 2) + c * (x + 1) * (x + 2) = 0 ↔ x = 1 ∨ x = 2) →
  a + b + c = 2 →
  a = 12 := by sorry

end NUMINAMATH_CALUDE_quadratic_equation_roots_l364_36442


namespace NUMINAMATH_CALUDE_savings_multiple_l364_36401

/-- Given two people's savings A and K satisfying certain conditions,
    prove that doubling K results in 3 times A -/
theorem savings_multiple (A K : ℚ) 
  (h1 : A + K = 750)
  (h2 : A - 150 = (1/3) * K) :
  2 * K = 3 * A := by
  sorry

end NUMINAMATH_CALUDE_savings_multiple_l364_36401


namespace NUMINAMATH_CALUDE_jason_initial_money_l364_36481

theorem jason_initial_money (jason_current : ℕ) (jason_earned : ℕ) 
  (h1 : jason_current = 63) 
  (h2 : jason_earned = 60) : 
  jason_current - jason_earned = 3 := by
sorry

end NUMINAMATH_CALUDE_jason_initial_money_l364_36481


namespace NUMINAMATH_CALUDE_simplify_nested_radicals_l364_36463

theorem simplify_nested_radicals : 
  2 * Real.sqrt (3 + Real.sqrt (5 - Real.sqrt (13 + Real.sqrt 48))) = Real.sqrt 6 + Real.sqrt 2 := by
sorry

end NUMINAMATH_CALUDE_simplify_nested_radicals_l364_36463


namespace NUMINAMATH_CALUDE_final_worker_bees_count_l364_36494

/-- Represents the state of the bee hive --/
structure BeeHive where
  workers : ℕ
  drones : ℕ
  queens : ℕ
  guards : ℕ

/-- Applies the series of events to the bee hive --/
def applyEvents (hive : BeeHive) : BeeHive :=
  let hive1 := { hive with 
    workers := hive.workers - 28,
    drones := hive.drones - 12,
    guards := hive.guards - 5 }
  let hive2 := { hive1 with 
    workers := hive1.workers - 30,
    guards := hive1.guards + 30 }
  let hive3 := { hive2 with 
    workers := hive2.workers + 15 }
  { hive3 with 
    workers := 0 }

/-- The theorem to be proved --/
theorem final_worker_bees_count (initialHive : BeeHive) 
  (h1 : initialHive.workers = 400)
  (h2 : initialHive.drones = 75)
  (h3 : initialHive.queens = 1)
  (h4 : initialHive.guards = 50) :
  (applyEvents initialHive).workers = 0 := by
  sorry

#check final_worker_bees_count

end NUMINAMATH_CALUDE_final_worker_bees_count_l364_36494


namespace NUMINAMATH_CALUDE_stream_speed_l364_36409

/-- The speed of a stream given boat travel times and distances -/
theorem stream_speed (downstream_distance upstream_distance : ℝ) 
  (time : ℝ) (h1 : downstream_distance = 84) (h2 : upstream_distance = 48) 
  (h3 : time = 2) : ∃ (stream_speed : ℝ), stream_speed = 9 ∧ 
  ∃ (boat_speed : ℝ), 
    downstream_distance = (boat_speed + stream_speed) * time ∧
    upstream_distance = (boat_speed - stream_speed) * time :=
by
  sorry


end NUMINAMATH_CALUDE_stream_speed_l364_36409


namespace NUMINAMATH_CALUDE_intersection_equals_interval_l364_36471

-- Define the sets M and N
def M : Set ℝ := {x | Real.log x > 0}
def N : Set ℝ := {x | x^2 ≤ 4}

-- Define the interval (1, 2]
def interval : Set ℝ := {x | 1 < x ∧ x ≤ 2}

-- Statement to prove
theorem intersection_equals_interval : M ∩ N = interval := by sorry

end NUMINAMATH_CALUDE_intersection_equals_interval_l364_36471


namespace NUMINAMATH_CALUDE_four_solutions_gg_eq_3_l364_36468

def g (x : ℝ) : ℝ := x^2 - 4*x + 3

def domain (x : ℝ) : Prop := -2 ≤ x ∧ x ≤ 5

theorem four_solutions_gg_eq_3 :
  ∃! (s : Finset ℝ), s.card = 4 ∧ 
  (∀ x ∈ s, domain x ∧ g (g x) = 3) ∧
  (∀ x, domain x → g (g x) = 3 → x ∈ s) :=
sorry

end NUMINAMATH_CALUDE_four_solutions_gg_eq_3_l364_36468


namespace NUMINAMATH_CALUDE_stating_perpendicular_bisector_correct_l364_36437

/-- The perpendicular bisector of a line segment. -/
def perpendicular_bisector (line_eq : ℝ → ℝ → Prop) (x_range : Set ℝ) : ℝ → ℝ → Prop :=
  fun x y => 2 * x + y - 3 = 0

/-- The original line segment equation. -/
def original_line (x y : ℝ) : Prop := x - 2 * y + 1 = 0

/-- The range of x for the original line segment. -/
def x_range : Set ℝ := {x | -1 ≤ x ∧ x ≤ 3}

/-- 
Theorem stating that the perpendicular_bisector function correctly defines 
the perpendicular bisector of the line segment given by the original_line 
equation within the specified x_range.
-/
theorem perpendicular_bisector_correct : 
  perpendicular_bisector original_line x_range = 
    fun x y => 2 * x + y - 3 = 0 := by sorry

end NUMINAMATH_CALUDE_stating_perpendicular_bisector_correct_l364_36437


namespace NUMINAMATH_CALUDE_total_profit_equation_l364_36462

/-- Represents the initial investment of person A in rupees -/
def initial_investment_A : ℚ := 2000

/-- Represents the initial investment of person B in rupees -/
def initial_investment_B : ℚ := 4000

/-- Represents the number of months before investment change -/
def months_before_change : ℕ := 8

/-- Represents the number of months after investment change -/
def months_after_change : ℕ := 4

/-- Represents the amount A withdrew after 8 months in rupees -/
def amount_A_withdrew : ℚ := 1000

/-- Represents the amount B added after 8 months in rupees -/
def amount_B_added : ℚ := 1000

/-- Represents A's share of the profit in rupees -/
def A_profit_share : ℚ := 175

/-- Calculates the total investment of A over the year -/
def total_investment_A : ℚ :=
  initial_investment_A * months_before_change +
  (initial_investment_A - amount_A_withdrew) * months_after_change

/-- Calculates the total investment of B over the year -/
def total_investment_B : ℚ :=
  initial_investment_B * months_before_change +
  (initial_investment_B + amount_B_added) * months_after_change

/-- Theorem stating that the total profit P satisfies the equation (5/18) * P = 175 -/
theorem total_profit_equation (P : ℚ) :
  total_investment_A / (total_investment_A + total_investment_B) * P = A_profit_share := by
  sorry

end NUMINAMATH_CALUDE_total_profit_equation_l364_36462


namespace NUMINAMATH_CALUDE_consecutive_products_sum_l364_36447

theorem consecutive_products_sum : ∃ (a b c d e : ℕ), 
  (b = a + 1) ∧ 
  (d = c + 1) ∧ 
  (e = d + 1) ∧ 
  (a * b = 210) ∧ 
  (c * d * e = 210) ∧ 
  (a + b + c + d + e = 47) := by
  sorry

end NUMINAMATH_CALUDE_consecutive_products_sum_l364_36447


namespace NUMINAMATH_CALUDE_f_is_quadratic_l364_36458

/-- Definition of a quadratic equation -/
def is_quadratic_equation (f : ℝ → ℝ) : Prop :=
  ∃ (a b c : ℝ), a ≠ 0 ∧ ∀ x, f x = a * x^2 + b * x + c

/-- The equation x² - 5 = 0 -/
def f (x : ℝ) : ℝ := x^2 - 5

/-- Theorem: f is a quadratic equation -/
theorem f_is_quadratic : is_quadratic_equation f := by
  sorry

end NUMINAMATH_CALUDE_f_is_quadratic_l364_36458


namespace NUMINAMATH_CALUDE_f_derivative_at_negative_one_l364_36459

noncomputable def f (x : ℝ) : ℝ := -x^3 + 1/x

theorem f_derivative_at_negative_one :
  (deriv f) (-1) = -4 :=
sorry

end NUMINAMATH_CALUDE_f_derivative_at_negative_one_l364_36459


namespace NUMINAMATH_CALUDE_min_value_theorem_l364_36410

/-- A monotonically increasing function on ℝ of the form f(x) = a^x + b -/
noncomputable def f (a b : ℝ) : ℝ → ℝ := λ x ↦ a^x + b

/-- The theorem stating the minimum value of the expression -/
theorem min_value_theorem (a b : ℝ) (h1 : a > 1) (h2 : b > 0) (h3 : f a b 1 = 3) :
  (4 / (a - 1) + 1 / b) ≥ 9/2 :=
sorry

end NUMINAMATH_CALUDE_min_value_theorem_l364_36410


namespace NUMINAMATH_CALUDE_hyperbola_equation_theorem_l364_36406

/-- A hyperbola with focal length 4√3 and one branch intersected by the line y = x - 3 at two points -/
structure Hyperbola where
  /-- The focal length of the hyperbola -/
  focal_length : ℝ
  /-- The line that intersects one branch of the hyperbola at two points -/
  intersecting_line : ℝ → ℝ
  /-- Condition that the focal length is 4√3 -/
  focal_length_cond : focal_length = 4 * Real.sqrt 3
  /-- Condition that the line y = x - 3 intersects one branch at two points -/
  intersecting_line_cond : intersecting_line = fun x => x - 3

/-- The equation of the hyperbola -/
def hyperbola_equation (h : Hyperbola) (x y : ℝ) : Prop :=
  x^2 / 6 - y^2 / 6 = 1

/-- Theorem stating that the given hyperbola has the equation x²/6 - y²/6 = 1 -/
theorem hyperbola_equation_theorem (h : Hyperbola) :
  ∀ x y, hyperbola_equation h x y ↔ x^2 / 6 - y^2 / 6 = 1 :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_equation_theorem_l364_36406


namespace NUMINAMATH_CALUDE_parabola_directrix_l364_36446

/-- The directrix of the parabola x = -1/4 * y^2 is the line x = 1 -/
theorem parabola_directrix (x y : ℝ) : 
  (x = -1/4 * y^2) → (∃ (p : ℝ), p = 1 ∧ 
    (∀ (x₀ y₀ : ℝ), x₀ = -1/4 * y₀^2 → 
      ((x₀ + 1)^2 + y₀^2 = (x₀ - p)^2))) := by
  sorry

end NUMINAMATH_CALUDE_parabola_directrix_l364_36446


namespace NUMINAMATH_CALUDE_intersects_x_axis_once_iff_m_range_l364_36444

/-- The function f(x) = x³ - x² - x + m -/
def f (m : ℝ) (x : ℝ) : ℝ := x^3 - x^2 - x + m

/-- The derivative of f(x) -/
def f_derivative (x : ℝ) : ℝ := 3*x^2 - 2*x - 1

theorem intersects_x_axis_once_iff_m_range (m : ℝ) :
  (∃! x, f m x = 0) ↔ (m < -5/27 ∨ m > 0) := by sorry

end NUMINAMATH_CALUDE_intersects_x_axis_once_iff_m_range_l364_36444


namespace NUMINAMATH_CALUDE_divisible_by_seven_l364_36480

theorem divisible_by_seven (k : ℕ) : ∃ m : ℤ, 2^(6*k+1) + 3^(6*k+1) + 5^(6*k) + 1 = 7*m := by
  sorry

end NUMINAMATH_CALUDE_divisible_by_seven_l364_36480


namespace NUMINAMATH_CALUDE_multiply_power_equals_power_sum_problem_solution_l364_36487

theorem multiply_power_equals_power_sum (a : ℕ) (m n : ℕ) : 
  a * (a^n) = a^(n + 1) := by sorry

theorem problem_solution : 
  3000 * (3000^3000) = 3000^3001 := by sorry

end NUMINAMATH_CALUDE_multiply_power_equals_power_sum_problem_solution_l364_36487


namespace NUMINAMATH_CALUDE_equal_derivatives_of_quadratic_functions_l364_36443

theorem equal_derivatives_of_quadratic_functions :
  let f (x : ℝ) := 1 - 2 * x^2
  let g (x : ℝ) := -2 * x^2 + 3
  ∀ x, deriv f x = deriv g x := by
sorry

end NUMINAMATH_CALUDE_equal_derivatives_of_quadratic_functions_l364_36443


namespace NUMINAMATH_CALUDE_angle_EFG_value_l364_36484

/-- A configuration of a square inside a regular octagon sharing one side -/
structure SquareInOctagon where
  /-- The measure of an internal angle of the regular octagon -/
  octagon_angle : ℝ
  /-- The measure of an internal angle of the square -/
  square_angle : ℝ
  /-- The measure of angle EFH -/
  angle_EFH : ℝ
  /-- The measure of angle EFG -/
  angle_EFG : ℝ

/-- Properties of the SquareInOctagon configuration -/
axiom octagon_angle_value (config : SquareInOctagon) : config.octagon_angle = 135
axiom square_angle_value (config : SquareInOctagon) : config.square_angle = 90
axiom angle_EFH_value (config : SquareInOctagon) : config.angle_EFH = config.octagon_angle - config.square_angle
axiom isosceles_triangle (config : SquareInOctagon) : config.angle_EFG = (180 - config.angle_EFH) / 2

/-- The main theorem: angle EFG measures 67.5° -/
theorem angle_EFG_value (config : SquareInOctagon) : config.angle_EFG = 67.5 := by
  sorry

end NUMINAMATH_CALUDE_angle_EFG_value_l364_36484


namespace NUMINAMATH_CALUDE_inequality_solution_l364_36425

theorem inequality_solution (x : ℝ) :
  x ≥ -14 → (x + 2 < Real.sqrt (x + 14) ↔ -14 ≤ x ∧ x < 2) := by
  sorry

end NUMINAMATH_CALUDE_inequality_solution_l364_36425


namespace NUMINAMATH_CALUDE_seating_arrangements_l364_36423

/-- Represents the number of people sitting around the table -/
def total_people : ℕ := 8

/-- Represents the number of people in the special block (leader, vice leader, recorder) -/
def special_block : ℕ := 3

/-- Represents the number of units to arrange (treating the special block as one unit) -/
def units_to_arrange : ℕ := total_people - special_block + 1

/-- Represents the number of ways to arrange the people within the special block -/
def internal_arrangements : ℕ := 2

/-- Calculates the number of unique circular arrangements for n elements -/
def circular_arrangements (n : ℕ) : ℕ := Nat.factorial (n - 1)

/-- The main theorem stating the number of unique seating arrangements -/
theorem seating_arrangements : 
  circular_arrangements units_to_arrange * internal_arrangements = 240 := by
  sorry

end NUMINAMATH_CALUDE_seating_arrangements_l364_36423


namespace NUMINAMATH_CALUDE_certain_number_of_seconds_l364_36488

/-- Converts minutes to seconds -/
def minutes_to_seconds (m : ℕ) : ℕ := m * 60

/-- The proportion given in the problem -/
def proportion (x : ℕ) : Prop :=
  15 / x = 30 / minutes_to_seconds 10

theorem certain_number_of_seconds : ∃ x : ℕ, proportion x ∧ x = 300 :=
  sorry

end NUMINAMATH_CALUDE_certain_number_of_seconds_l364_36488


namespace NUMINAMATH_CALUDE_solution_set_min_value_min_value_achieved_l364_36478

-- Define the function f
def f (x : ℝ) : ℝ := |2 * x - 1|

-- Part 1: Theorem for the solution set of f(2x) ≤ f(x + 1)
theorem solution_set (x : ℝ) : f (2 * x) ≤ f (x + 1) ↔ 0 ≤ x ∧ x ≤ 1 := by sorry

-- Part 2: Theorem for the minimum value of f(a²) + f(b²)
theorem min_value (a b : ℝ) (h : a + b = 2) : 
  ∃ (m : ℝ), m = 2 ∧ ∀ (a' b' : ℝ), a' + b' = 2 → f (a' ^ 2) + f (b' ^ 2) ≥ m := by sorry

-- Corollary: The minimum is achieved when a = b = 1
theorem min_value_achieved (a b : ℝ) (h : a + b = 2) : 
  f (a ^ 2) + f (b ^ 2) = 2 ↔ a = 1 ∧ b = 1 := by sorry

end NUMINAMATH_CALUDE_solution_set_min_value_min_value_achieved_l364_36478


namespace NUMINAMATH_CALUDE_probability_is_five_twelfths_l364_36479

/-- Represents a person with 6 differently colored blocks -/
structure Person :=
  (blocks : Fin 6 → Color)

/-- Represents the colors of blocks -/
inductive Color
  | Red | Blue | Yellow | White | Green | Purple

/-- Represents a box with placed blocks -/
structure Box :=
  (blocks : Fin 3 → Color)

/-- The probability of at least one box receiving blocks of the same color from at least two different people -/
def probability_same_color (people : Fin 3 → Person) (boxes : Fin 5 → Box) : ℚ :=
  sorry

/-- The main theorem stating that the probability is 5/12 -/
theorem probability_is_five_twelfths :
  ∃ (people : Fin 3 → Person) (boxes : Fin 5 → Box),
    probability_same_color people boxes = 5 / 12 :=
  sorry

end NUMINAMATH_CALUDE_probability_is_five_twelfths_l364_36479


namespace NUMINAMATH_CALUDE_staircase_has_31_steps_l364_36489

/-- Represents a staircase with a middle step and specific movement rules -/
structure Staircase where
  total_steps : ℕ
  middle_step : ℕ
  (middle_property : middle_step * 2 - 1 = total_steps)
  (movement_property : middle_step + 7 - 15 = 8)

/-- Theorem stating that a staircase satisfying the given conditions has 31 steps -/
theorem staircase_has_31_steps (s : Staircase) : s.total_steps = 31 := by
  sorry

#check staircase_has_31_steps

end NUMINAMATH_CALUDE_staircase_has_31_steps_l364_36489


namespace NUMINAMATH_CALUDE_square_root_of_sixteen_l364_36400

theorem square_root_of_sixteen : 
  {x : ℝ | x^2 = 16} = {4, -4} := by sorry

end NUMINAMATH_CALUDE_square_root_of_sixteen_l364_36400


namespace NUMINAMATH_CALUDE_mersenne_coprime_iff_exponents_coprime_l364_36439

theorem mersenne_coprime_iff_exponents_coprime (p q : ℕ+) :
  Nat.gcd ((2 : ℕ) ^ p.val - 1) ((2 : ℕ) ^ q.val - 1) = 1 ↔ Nat.gcd p.val q.val = 1 := by
  sorry

end NUMINAMATH_CALUDE_mersenne_coprime_iff_exponents_coprime_l364_36439


namespace NUMINAMATH_CALUDE_magic_card_price_l364_36422

/-- The initial price of a Magic card that triples in value and results in a $200 profit when sold -/
def initial_price : ℝ := 100

/-- The value of the card after tripling -/
def tripled_value (price : ℝ) : ℝ := 3 * price

/-- The profit made from selling the card -/
def profit (initial_price : ℝ) : ℝ := tripled_value initial_price - initial_price

theorem magic_card_price :
  profit initial_price = 200 ∧ tripled_value initial_price = 3 * initial_price := by
  sorry

end NUMINAMATH_CALUDE_magic_card_price_l364_36422


namespace NUMINAMATH_CALUDE_lee_cookies_l364_36449

/-- Given that Lee can make 24 cookies with 3 cups of flour, 
    this theorem proves he can make 36 cookies with 4.5 cups of flour. -/
theorem lee_cookies (cookies_per_3_cups : ℕ) (cookies_per_4_5_cups : ℕ) 
  (h1 : cookies_per_3_cups = 24) :
  cookies_per_4_5_cups = 36 :=
by
  sorry

#check lee_cookies

end NUMINAMATH_CALUDE_lee_cookies_l364_36449


namespace NUMINAMATH_CALUDE_unique_function_solution_l364_36420

/-- A function from non-negative reals to non-negative reals -/
def NonnegFunction := {f : ℝ → ℝ // ∀ x, 0 ≤ x → 0 ≤ f x}

theorem unique_function_solution (f : NonnegFunction) 
  (h : ∀ x : ℝ, 0 ≤ x → f.val (f.val x) + f.val x = 12 * x) :
  ∀ x : ℝ, 0 ≤ x → f.val x = 3 * x := by
  sorry

end NUMINAMATH_CALUDE_unique_function_solution_l364_36420


namespace NUMINAMATH_CALUDE_min_value_xy_l364_36433

theorem min_value_xy (x y : ℝ) (hx : x > 1) (hy : y > 1) 
  (h_seq : (1/4 * Real.log x) * (Real.log y) = 1/16) : 
  x * y ≥ Real.exp 1 ∧ ∃ x y, x > 1 ∧ y > 1 ∧ (1/4 * Real.log x) * (Real.log y) = 1/16 ∧ x * y = Real.exp 1 := by
  sorry

end NUMINAMATH_CALUDE_min_value_xy_l364_36433


namespace NUMINAMATH_CALUDE_total_precious_stones_l364_36498

theorem total_precious_stones :
  let agate : ℕ := 24
  let olivine : ℕ := agate + 5
  let sapphire : ℕ := 2 * olivine
  let diamond : ℕ := olivine + 11
  let amethyst : ℕ := sapphire + diamond
  let ruby : ℕ := (5 * olivine + 1) / 2  -- Rounded up
  let garnet : ℕ := amethyst - ruby - 5
  let topaz : ℕ := garnet / 2
  agate + olivine + sapphire + diamond + amethyst + ruby + garnet + topaz = 352 := by
sorry

end NUMINAMATH_CALUDE_total_precious_stones_l364_36498


namespace NUMINAMATH_CALUDE_log_equation_solution_l364_36493

theorem log_equation_solution (b x : ℝ) (hb_pos : b > 0) (hb_neq_one : b ≠ 1) (hx_neq_one : x ≠ 1) :
  (Real.log x) / (Real.log (b^3)) + (Real.log b) / (Real.log (x^3)) = 1 →
  x = b^((3 + Real.sqrt 5) / 2) ∨ x = b^((3 - Real.sqrt 5) / 2) := by
sorry

end NUMINAMATH_CALUDE_log_equation_solution_l364_36493


namespace NUMINAMATH_CALUDE_natasha_quarters_l364_36455

theorem natasha_quarters : 
  ∃ (n : ℕ), 8 < n ∧ n < 80 ∧ 
  n % 4 = 1 ∧ n % 5 = 1 ∧ n % 6 = 1 ∧
  (∀ (m : ℕ), 8 < m ∧ m < 80 ∧ 
   m % 4 = 1 ∧ m % 5 = 1 ∧ m % 6 = 1 → m = n) ∧
  n = 61 :=
by sorry

end NUMINAMATH_CALUDE_natasha_quarters_l364_36455


namespace NUMINAMATH_CALUDE_standard_form_of_given_equation_l364_36429

/-- Standard form of a quadratic equation -/
def standard_form (a b c : ℝ) : ℝ → Prop :=
  fun x ↦ a * x^2 + b * x + c = 0

/-- The given quadratic equation -/
def given_equation (x : ℝ) : Prop :=
  3 * x^2 + 1 = 7 * x

/-- Theorem stating that the standard form of 3x^2 + 1 = 7x is 3x^2 - 7x + 1 = 0 -/
theorem standard_form_of_given_equation :
  ∃ a b c : ℝ, a ≠ 0 ∧ (∀ x, given_equation x ↔ standard_form a b c x) ∧ 
  a = 3 ∧ b = -7 ∧ c = 1 :=
sorry

end NUMINAMATH_CALUDE_standard_form_of_given_equation_l364_36429


namespace NUMINAMATH_CALUDE_work_completion_days_l364_36435

/-- Calculates the initial planned days to complete a work given the original number of workers,
    the number of absent workers, and the days taken by the remaining workers. -/
def initialPlannedDays (originalWorkers : ℕ) (absentWorkers : ℕ) (daysWithFewerWorkers : ℕ) : ℚ :=
  (originalWorkers - absentWorkers) * daysWithFewerWorkers / originalWorkers

/-- Proves that given 15 original workers, 5 absent workers, and 60 days taken by the remaining workers,
    the initial planned days to complete the work is 40. -/
theorem work_completion_days : initialPlannedDays 15 5 60 = 40 := by
  sorry

end NUMINAMATH_CALUDE_work_completion_days_l364_36435


namespace NUMINAMATH_CALUDE_solution_set_is_positive_reals_l364_36474

open Set
open Function
open Real

noncomputable section

variables {f : ℝ → ℝ} (hf : Differentiable ℝ f)

def condition_1 (f : ℝ → ℝ) : Prop :=
  ∀ x, f x + deriv f x > 1

def condition_2 (f : ℝ → ℝ) : Prop :=
  f 0 = 2018

def solution_set (f : ℝ → ℝ) : Set ℝ :=
  {x | Real.exp x * f x - Real.exp x > 2017}

theorem solution_set_is_positive_reals
  (h1 : condition_1 f) (h2 : condition_2 f) :
  solution_set f = Ioi 0 :=
sorry

end

end NUMINAMATH_CALUDE_solution_set_is_positive_reals_l364_36474


namespace NUMINAMATH_CALUDE_dragon_jewels_l364_36476

theorem dragon_jewels (D : ℕ) : 
  6 = D / 3 →  -- The new jewels (6) are one-third of the original count
  21 = D - 3 + 6 -- The final count is the original count minus 3 (stolen) plus 6 (taken from king)
  := by sorry

end NUMINAMATH_CALUDE_dragon_jewels_l364_36476


namespace NUMINAMATH_CALUDE_three_digit_numbers_with_properties_l364_36418

def satisfiesConditions (N : ℕ) : Prop :=
  N % 2 = 1 ∧ N % 3 = 2 ∧ N % 4 = 3 ∧ N % 5 = 4 ∧ N % 6 = 5

def isThreeDigit (N : ℕ) : Prop :=
  100 ≤ N ∧ N ≤ 999

def solutionSet : Set ℕ :=
  {119, 179, 239, 299, 359, 419, 479, 539, 599, 659, 719, 779, 839, 899, 959}

theorem three_digit_numbers_with_properties :
  {N : ℕ | isThreeDigit N ∧ satisfiesConditions N} = solutionSet := by sorry

end NUMINAMATH_CALUDE_three_digit_numbers_with_properties_l364_36418


namespace NUMINAMATH_CALUDE_min_value_sum_of_squares_l364_36430

theorem min_value_sum_of_squares (u v w : ℝ) 
  (h_pos_u : u > 0) (h_pos_v : v > 0) (h_pos_w : w > 0)
  (h_sum_squares : u^2 + v^2 + w^2 = 8) :
  (u^4 / 9) + (v^4 / 16) + (w^4 / 25) ≥ 32/25 := by
  sorry

end NUMINAMATH_CALUDE_min_value_sum_of_squares_l364_36430


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_condition_l364_36412

open Real

-- Define the logarithm base 10
noncomputable def log10 (x : ℝ) : ℝ := log x / log 10

-- Define the set {1, 2}
def set_1_2 : Set ℝ := {1, 2}

-- Theorem statement
theorem sufficient_not_necessary_condition :
  (∀ m ∈ set_1_2, log10 m < 1) ∧
  (∃ m : ℝ, log10 m < 1 ∧ m ∉ set_1_2) :=
by sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_condition_l364_36412


namespace NUMINAMATH_CALUDE_trig_expression_approx_value_l364_36477

theorem trig_expression_approx_value : 
  let expr := (2 * Real.sin (30 * π / 180) * Real.cos (10 * π / 180) + 
               3 * Real.cos (150 * π / 180) * Real.cos (110 * π / 180)) /
              (4 * Real.sin (40 * π / 180) * Real.cos (20 * π / 180) + 
               5 * Real.cos (140 * π / 180) * Real.cos (100 * π / 180))
  ∃ ε > 0, abs (expr - 0.465) < ε := by
  sorry

end NUMINAMATH_CALUDE_trig_expression_approx_value_l364_36477


namespace NUMINAMATH_CALUDE_students_present_l364_36452

theorem students_present (total : ℕ) (absent_percentage : ℚ) : 
  total = 50 ∧ absent_percentage = 14/100 → 
  total - (total * absent_percentage).floor = 43 := by
  sorry

end NUMINAMATH_CALUDE_students_present_l364_36452


namespace NUMINAMATH_CALUDE_binomial_coefficient_seven_four_l364_36492

theorem binomial_coefficient_seven_four : Nat.choose 7 4 = 35 := by
  sorry

end NUMINAMATH_CALUDE_binomial_coefficient_seven_four_l364_36492


namespace NUMINAMATH_CALUDE_quadratic_equation_solution_l364_36497

theorem quadratic_equation_solution :
  ∀ x : ℝ, x^2 - 5*x + 6 = 0 ↔ x = 3 ∨ x = 2 := by
sorry

end NUMINAMATH_CALUDE_quadratic_equation_solution_l364_36497


namespace NUMINAMATH_CALUDE_quadratic_roots_and_integer_case_l364_36428

-- Define the quadratic equation
def quadratic_equation (k x : ℝ) : ℝ := k * x^2 + (2*k + 1) * x + 2

-- Theorem statement
theorem quadratic_roots_and_integer_case :
  (∀ k : ℝ, k ≠ 0 → ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ quadratic_equation k x₁ = 0 ∧ quadratic_equation k x₂ = 0) ∧
  (∀ k : ℕ+, (∃ x₁ x₂ : ℤ, x₁ ≠ x₂ ∧ quadratic_equation k x₁ = 0 ∧ quadratic_equation k x₂ = 0) → k = 1) :=
by sorry


end NUMINAMATH_CALUDE_quadratic_roots_and_integer_case_l364_36428


namespace NUMINAMATH_CALUDE_school_visit_arrangements_l364_36472

/-- Represents the number of days in a week -/
def week_days : Nat := 7

/-- Represents the number of consecutive days School A visits -/
def school_a_days : Nat := 2

/-- Represents the number of days School B visits -/
def school_b_days : Nat := 1

/-- Represents the number of days School C visits -/
def school_c_days : Nat := 1

/-- Calculates the number of arrangements for the school visits -/
def calculate_arrangements : Nat :=
  (week_days - school_a_days - school_b_days - school_c_days + 1) *
  (week_days - school_a_days - school_b_days - school_c_days)

/-- Theorem stating that the number of arrangements is 40 -/
theorem school_visit_arrangements :
  calculate_arrangements = 40 := by
  sorry

end NUMINAMATH_CALUDE_school_visit_arrangements_l364_36472


namespace NUMINAMATH_CALUDE_right_triangle_existence_l364_36403

theorem right_triangle_existence (a b c d : ℕ+) 
  (h1 : a * b = c * d) 
  (h2 : a + b = c - d) : 
  ∃ x y z : ℕ+, x^2 + y^2 = z^2 ∧ (1/2 : ℚ) * x * y = a * b := by
sorry

end NUMINAMATH_CALUDE_right_triangle_existence_l364_36403


namespace NUMINAMATH_CALUDE_wall_passing_skill_l364_36411

theorem wall_passing_skill (n : ℕ) : 
  8 * Real.sqrt (8 / n) = Real.sqrt (8 * (8 / n)) → n = 63 :=
by sorry

end NUMINAMATH_CALUDE_wall_passing_skill_l364_36411


namespace NUMINAMATH_CALUDE_fan_sales_theorem_l364_36467

/-- Represents the sales data for a week -/
structure WeeklySales where
  modelA : ℕ
  modelB : ℕ
  revenue : ℕ

/-- Represents the fan models and their properties -/
structure FanModels where
  purchasePriceA : ℕ
  purchasePriceB : ℕ
  sellingPriceA : ℕ
  sellingPriceB : ℕ

def totalUnits : ℕ := 40
def maxBudget : ℕ := 7650

def weekOneSales : WeeklySales := {
  modelA := 3
  modelB := 5
  revenue := 2150
}

def weekTwoSales : WeeklySales := {
  modelA := 4
  modelB := 10
  revenue := 3700
}

def fanModels : FanModels := {
  purchasePriceA := 210
  purchasePriceB := 180
  sellingPriceA := 300
  sellingPriceB := 250
}

theorem fan_sales_theorem (w1 : WeeklySales) (w2 : WeeklySales) (f : FanModels) :
  w1 = weekOneSales ∧ w2 = weekTwoSales ∧ f.purchasePriceA = 210 ∧ f.purchasePriceB = 180 →
  (f.sellingPriceA * w1.modelA + f.sellingPriceB * w1.modelB = w1.revenue ∧
   f.sellingPriceA * w2.modelA + f.sellingPriceB * w2.modelB = w2.revenue) →
  f.sellingPriceA = 300 ∧ f.sellingPriceB = 250 ∧
  (∀ a : ℕ, a ≤ totalUnits →
    f.purchasePriceA * a + f.purchasePriceB * (totalUnits - a) ≤ maxBudget →
    (f.sellingPriceA - f.purchasePriceA) * a + (f.sellingPriceB - f.purchasePriceB) * (totalUnits - a) ≤ 3100) ∧
  ∃ a : ℕ, a ≤ totalUnits ∧
    f.purchasePriceA * a + f.purchasePriceB * (totalUnits - a) ≤ maxBudget ∧
    (f.sellingPriceA - f.purchasePriceA) * a + (f.sellingPriceB - f.purchasePriceB) * (totalUnits - a) = 3100 :=
by sorry

end NUMINAMATH_CALUDE_fan_sales_theorem_l364_36467


namespace NUMINAMATH_CALUDE_tiles_for_monica_room_l364_36413

/-- Calculates the total number of tiles needed to cover a rectangular room
    with a border of larger tiles and inner area of smaller tiles. -/
def total_tiles (room_length room_width border_tile_size inner_tile_size : ℕ) : ℕ :=
  let border_area := room_length * room_width - (room_length - 2 * border_tile_size) * (room_width - 2 * border_tile_size)
  let border_tiles := border_area / (border_tile_size * border_tile_size)
  let inner_area := (room_length - 2 * border_tile_size) * (room_width - 2 * border_tile_size)
  let inner_tiles := inner_area / (inner_tile_size * inner_tile_size)
  border_tiles + inner_tiles

/-- Theorem stating that the total number of tiles for the given room dimensions and tile sizes is 318. -/
theorem tiles_for_monica_room : total_tiles 24 18 2 1 = 318 := by
  sorry

end NUMINAMATH_CALUDE_tiles_for_monica_room_l364_36413


namespace NUMINAMATH_CALUDE_multiply_by_three_l364_36450

theorem multiply_by_three (x : ℤ) : x + 14 = 56 → 3 * x = 126 := by
  sorry

end NUMINAMATH_CALUDE_multiply_by_three_l364_36450


namespace NUMINAMATH_CALUDE_danny_wrappers_found_l364_36465

/-- Represents the number of wrappers Danny found at the park -/
def wrappers_found : ℕ := 46

/-- Represents the number of bottle caps Danny found at the park -/
def bottle_caps_found : ℕ := 50

/-- Represents the difference between bottle caps and wrappers found -/
def difference : ℕ := 4

theorem danny_wrappers_found :
  bottle_caps_found = wrappers_found + difference →
  wrappers_found = 46 := by
  sorry

end NUMINAMATH_CALUDE_danny_wrappers_found_l364_36465


namespace NUMINAMATH_CALUDE_cubic_function_theorem_l364_36414

-- Define the cubic function f(x)
def f (a b c x : ℝ) : ℝ := x^3 + a*x^2 + b*x + c

-- Define the derivative of f(x)
def f' (a b x : ℝ) : ℝ := 3*x^2 + 2*a*x + b

theorem cubic_function_theorem (a b c : ℝ) :
  (∀ x ∈ Set.Icc (-3) 2, f a b c x > 1/c - 1/2) →
  (f' a b 1 = 0 ∧ f' a b (-2) = 0) →
  (a = 3/2 ∧ b = -6 ∧ ((3 - Real.sqrt 13)/2 < c ∧ c < 0) ∨ c > (3 + Real.sqrt 13)/2) :=
sorry

end NUMINAMATH_CALUDE_cubic_function_theorem_l364_36414


namespace NUMINAMATH_CALUDE_total_rubber_bands_l364_36466

def harper_rubber_bands : ℕ := 100
def brother_difference : ℕ := 56
def sister_difference : ℕ := 47

theorem total_rubber_bands :
  harper_rubber_bands +
  (harper_rubber_bands - brother_difference) +
  (harper_rubber_bands - brother_difference + sister_difference) = 235 := by
  sorry

end NUMINAMATH_CALUDE_total_rubber_bands_l364_36466


namespace NUMINAMATH_CALUDE_prob_two_good_in_four_draws_l364_36470

/-- Represents the number of light bulbs in the box -/
def total_bulbs : ℕ := 10

/-- Represents the number of good quality bulbs -/
def good_bulbs : ℕ := 8

/-- Represents the number of defective bulbs -/
def defective_bulbs : ℕ := 2

/-- Represents the number of draws -/
def num_draws : ℕ := 4

/-- Represents the number of good quality bulbs to be drawn -/
def target_good_bulbs : ℕ := 2

/-- Calculates the probability of drawing exactly 2 good quality bulbs in 4 draws -/
theorem prob_two_good_in_four_draws :
  (defective_bulbs * (defective_bulbs - 1) * good_bulbs * (good_bulbs - 1)) / 
  (total_bulbs * (total_bulbs - 1) * (total_bulbs - 2) * (total_bulbs - 3)) = 1 / 15 := by
  sorry

end NUMINAMATH_CALUDE_prob_two_good_in_four_draws_l364_36470


namespace NUMINAMATH_CALUDE_cody_dumplings_l364_36436

/-- The number of dumplings Cody cooked -/
def dumplings_cooked : ℕ := sorry

/-- The number of dumplings Cody ate -/
def dumplings_eaten : ℕ := 7

/-- The number of dumplings Cody has left -/
def dumplings_left : ℕ := 7

theorem cody_dumplings : dumplings_cooked = dumplings_eaten + dumplings_left := by
  sorry

end NUMINAMATH_CALUDE_cody_dumplings_l364_36436


namespace NUMINAMATH_CALUDE_remainder_sum_l364_36402

theorem remainder_sum (n : ℤ) : n % 12 = 7 → (n % 3 + n % 4 = 4) := by
  sorry

end NUMINAMATH_CALUDE_remainder_sum_l364_36402


namespace NUMINAMATH_CALUDE_lcm_28_72_l364_36473

theorem lcm_28_72 : Nat.lcm 28 72 = 504 := by
  sorry

end NUMINAMATH_CALUDE_lcm_28_72_l364_36473


namespace NUMINAMATH_CALUDE_existence_of_representation_l364_36453

theorem existence_of_representation (m : ℤ) :
  ∃ (a b k : ℤ), Odd a ∧ Odd b ∧ k ≥ 0 ∧ 2 * m = a^19 + b^99 + k * 2^1999 := by
  sorry

end NUMINAMATH_CALUDE_existence_of_representation_l364_36453


namespace NUMINAMATH_CALUDE_max_S_at_7_or_8_l364_36431

/-- Represents the sum of the first n terms of the arithmetic sequence -/
def S (n : ℕ) : ℚ :=
  5 * n - (5 / 14) * n * (n - 1)

/-- The maximum value of S occurs when n is 7 or 8 -/
theorem max_S_at_7_or_8 :
  ∀ k : ℕ, (S k ≤ S 7 ∧ S k ≤ S 8) ∧
  (S 7 = S 8 ∨ (∀ m : ℕ, m ≠ 7 → m ≠ 8 → S m < max (S 7) (S 8))) := by
  sorry

end NUMINAMATH_CALUDE_max_S_at_7_or_8_l364_36431


namespace NUMINAMATH_CALUDE_hyperbola_eccentricity_l364_36469

-- Define the hyperbola
def hyperbola (a b : ℝ) (x y : ℝ) : Prop :=
  x^2 / a^2 - y^2 / b^2 = 1

-- Define the asymptote
def asymptote (x y : ℝ) : Prop :=
  2 * x - y = 0

-- Theorem statement
theorem hyperbola_eccentricity (a b : ℝ) (h1 : a > 0) (h2 : b > 0) :
  (∃ x y, hyperbola a b x y ∧ asymptote x y) →
  let c := Real.sqrt (a^2 + b^2)
  c / a = Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_hyperbola_eccentricity_l364_36469


namespace NUMINAMATH_CALUDE_quadratic_roots_complex_l364_36424

theorem quadratic_roots_complex (x : ℂ) : 
  x^2 + 6*x + 13 = 0 ↔ (x + 3*I) * (x - 3*I) = 0 :=
sorry

end NUMINAMATH_CALUDE_quadratic_roots_complex_l364_36424


namespace NUMINAMATH_CALUDE_partner_investment_time_l364_36499

/-- Given two partners P and Q with investments and profits, prove Q's investment time -/
theorem partner_investment_time 
  (investment_ratio : ℚ) -- Ratio of P's investment to Q's investment
  (profit_ratio : ℚ) -- Ratio of P's profit to Q's profit
  (p_time : ℕ) -- Time P invested in months
  (h1 : investment_ratio = 7 / 5)
  (h2 : profit_ratio = 7 / 14)
  (h3 : p_time = 5) :
  ∃ (q_time : ℕ), q_time = 14 := by
  sorry


end NUMINAMATH_CALUDE_partner_investment_time_l364_36499
