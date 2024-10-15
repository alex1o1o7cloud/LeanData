import Mathlib

namespace NUMINAMATH_CALUDE_like_terms_exponent_sum_l1644_164459

theorem like_terms_exponent_sum (a b : ℝ) (x y : ℤ) : 
  (∃ k : ℝ, k ≠ 0 ∧ -3 * a^(x + 2*y) * b^9 = k * (2 * a^3 * b^(2*x + y))) → 
  x + y = 4 := by sorry

end NUMINAMATH_CALUDE_like_terms_exponent_sum_l1644_164459


namespace NUMINAMATH_CALUDE_connie_tickets_l1644_164405

def ticket_distribution (total_tickets : ℕ) : Prop :=
  let koala := total_tickets * 20 / 100
  let earbuds := 30
  let car := earbuds * 2
  let bracelets := total_tickets * 15 / 100
  let remaining := total_tickets - (koala + earbuds + car + bracelets)
  let poster := (remaining * 4) / 7
  let keychain := (remaining * 3) / 7
  koala = 100 ∧ 
  earbuds = 30 ∧ 
  car = 60 ∧ 
  bracelets = 75 ∧ 
  poster = 135 ∧ 
  keychain = 100 ∧
  koala + earbuds + car + bracelets + poster + keychain = total_tickets

theorem connie_tickets : ticket_distribution 500 := by
  sorry

end NUMINAMATH_CALUDE_connie_tickets_l1644_164405


namespace NUMINAMATH_CALUDE_baxter_earnings_l1644_164487

structure School where
  name : String
  students : ℕ
  days : ℕ
  bonus : ℚ

def total_student_days (schools : List School) : ℕ :=
  schools.foldl (fun acc s => acc + s.students * s.days) 0

def total_bonus (schools : List School) : ℚ :=
  schools.foldl (fun acc s => acc + s.students * s.bonus) 0

theorem baxter_earnings (schools : List School) 
  (h_schools : schools = [
    ⟨"Ajax", 5, 4, 0⟩, 
    ⟨"Baxter", 3, 6, 5⟩, 
    ⟨"Colton", 6, 8, 0⟩
  ])
  (h_total_paid : 920 = (total_student_days schools) * (daily_wage : ℚ) + total_bonus schools)
  (daily_wage : ℚ) :
  ∃ (baxter_earnings : ℚ), baxter_earnings = 204.42 ∧ 
    baxter_earnings = 3 * 6 * daily_wage + 3 * 5 :=
by sorry

end NUMINAMATH_CALUDE_baxter_earnings_l1644_164487


namespace NUMINAMATH_CALUDE_largest_n_is_correct_l1644_164458

/-- The largest value of n for which 3x^2 + nx + 90 can be factored as the product of two linear factors with integer coefficients -/
def largest_n : ℕ := 271

/-- A function representing the quadratic expression 3x^2 + nx + 90 -/
def quadratic (n : ℕ) (x : ℚ) : ℚ := 3 * x^2 + n * x + 90

/-- A predicate that checks if a quadratic expression can be factored into two linear factors with integer coefficients -/
def has_integer_linear_factors (n : ℕ) : Prop :=
  ∃ (a b c d : ℤ), ∀ (x : ℚ), quadratic n x = (a * x + b) * (c * x + d)

theorem largest_n_is_correct :
  (∀ n : ℕ, n > largest_n → ¬(has_integer_linear_factors n)) ∧
  (has_integer_linear_factors largest_n) :=
by sorry


end NUMINAMATH_CALUDE_largest_n_is_correct_l1644_164458


namespace NUMINAMATH_CALUDE_rectangle_area_l1644_164492

theorem rectangle_area (square_area : ℝ) (rectangle_width rectangle_length rectangle_area : ℝ) : 
  square_area = 36 →
  rectangle_width = Real.sqrt square_area →
  rectangle_length = 3 * rectangle_width →
  rectangle_area = rectangle_width * rectangle_length →
  rectangle_area = 108 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_area_l1644_164492


namespace NUMINAMATH_CALUDE_cheryl_material_usage_l1644_164437

/-- The amount of material Cheryl used for her project -/
def material_used (material1 material2 leftover : ℚ) : ℚ :=
  material1 + material2 - leftover

/-- Theorem stating the total amount of material Cheryl used -/
theorem cheryl_material_usage :
  let material1 : ℚ := 4/9
  let material2 : ℚ := 2/3
  let leftover : ℚ := 8/18
  material_used material1 material2 leftover = 2/3 :=
by
  sorry

#check cheryl_material_usage

end NUMINAMATH_CALUDE_cheryl_material_usage_l1644_164437


namespace NUMINAMATH_CALUDE_expression_evaluation_l1644_164496

theorem expression_evaluation (x y : ℕ) (hx : x = 3) (hy : y = 2) :
  y + y * (y^x + Nat.factorial x) = 30 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l1644_164496


namespace NUMINAMATH_CALUDE_count_hollow_circles_l1644_164400

/-- The length of the repeating sequence of circles -/
def sequence_length : ℕ := 24

/-- The number of hollow circles in each repetition of the sequence -/
def hollow_circles_per_sequence : ℕ := 5

/-- The total number of circles we're considering -/
def total_circles : ℕ := 2003

/-- The number of hollow circles in the first 2003 circles -/
def hollow_circles_count : ℕ := 446

theorem count_hollow_circles :
  (total_circles / sequence_length) * hollow_circles_per_sequence +
  (hollow_circles_per_sequence * (total_circles % sequence_length) / sequence_length) =
  hollow_circles_count :=
sorry

end NUMINAMATH_CALUDE_count_hollow_circles_l1644_164400


namespace NUMINAMATH_CALUDE_initial_liquid_A_amount_l1644_164419

/-- Proves that the initial amount of liquid A in a can is 36.75 litres given the specified conditions -/
theorem initial_liquid_A_amount
  (initial_ratio_A : ℚ)
  (initial_ratio_B : ℚ)
  (drawn_off_amount : ℚ)
  (new_ratio_A : ℚ)
  (new_ratio_B : ℚ)
  (h1 : initial_ratio_A = 7)
  (h2 : initial_ratio_B = 5)
  (h3 : drawn_off_amount = 18)
  (h4 : new_ratio_A = 7)
  (h5 : new_ratio_B = 9) :
  ∃ (initial_A : ℚ),
    initial_A = 36.75 ∧
    (initial_A / (initial_A * initial_ratio_B / initial_ratio_A) = initial_ratio_A / initial_ratio_B) ∧
    ((initial_A - drawn_off_amount * initial_ratio_A / (initial_ratio_A + initial_ratio_B)) /
     (initial_A * initial_ratio_B / initial_ratio_A - drawn_off_amount * initial_ratio_B / (initial_ratio_A + initial_ratio_B) + drawn_off_amount) =
     new_ratio_A / new_ratio_B) :=
by sorry

end NUMINAMATH_CALUDE_initial_liquid_A_amount_l1644_164419


namespace NUMINAMATH_CALUDE_gift_cost_per_parent_l1644_164401

-- Define the given values
def total_spent : ℝ := 150
def siblings_count : ℕ := 3
def cost_per_sibling : ℝ := 30

-- Define the theorem
theorem gift_cost_per_parent :
  let spent_on_siblings := siblings_count * cost_per_sibling
  let spent_on_parents := total_spent - spent_on_siblings
  let cost_per_parent := spent_on_parents / 2
  cost_per_parent = 30 := by sorry

end NUMINAMATH_CALUDE_gift_cost_per_parent_l1644_164401


namespace NUMINAMATH_CALUDE_intersection_point_l1644_164447

def L₁ (x : ℝ) : ℝ := 3 * x + 9
def L₂ (x : ℝ) : ℝ := -x + 6

def parameterization_L₁ (t : ℝ) : ℝ × ℝ := (t, 3 * t + 9)
def parameterization_L₂ (s : ℝ) : ℝ × ℝ := (s, -s + 6)

theorem intersection_point :
  ∃ (x y : ℝ), L₁ x = y ∧ L₂ x = y ∧ x = -3/4 ∧ y = 15/4 :=
by sorry

end NUMINAMATH_CALUDE_intersection_point_l1644_164447


namespace NUMINAMATH_CALUDE_intersection_range_l1644_164429

-- Define the points M and N
def M : ℝ × ℝ := (1, 0)
def N : ℝ × ℝ := (-1, 0)

-- Define the line equation
def line (x y b : ℝ) : Prop := 2 * x + y = b

-- Define the line segment MN
def on_segment (x y : ℝ) : Prop :=
  x ≥ -1 ∧ x ≤ 1 ∧ y = 0

-- Theorem statement
theorem intersection_range :
  ∀ b : ℝ,
  (∃ x y : ℝ, line x y b ∧ on_segment x y) ↔
  b ≥ -2 ∧ b ≤ 2 :=
sorry

end NUMINAMATH_CALUDE_intersection_range_l1644_164429


namespace NUMINAMATH_CALUDE_reflection_problem_l1644_164427

/-- Reflection of a point across a line --/
def reflect (x y m b : ℝ) : ℝ × ℝ := sorry

/-- The problem statement --/
theorem reflection_problem (m b : ℝ) :
  reflect (-4) 2 m b = (6, 0) → m + b = 1 := by sorry

end NUMINAMATH_CALUDE_reflection_problem_l1644_164427


namespace NUMINAMATH_CALUDE_polynomial_identity_l1644_164409

/-- For any real numbers a, b, and c, 
    a(b - c)^4 + b(c - a)^4 + c(a - b)^4 = (a - b)(b - c)(c - a)(a + b + c) -/
theorem polynomial_identity (a b c : ℝ) : 
  a * (b - c)^4 + b * (c - a)^4 + c * (a - b)^4 = (a - b) * (b - c) * (c - a) * (a + b + c) := by
  sorry

end NUMINAMATH_CALUDE_polynomial_identity_l1644_164409


namespace NUMINAMATH_CALUDE_travel_probability_is_two_thirds_l1644_164448

/-- Represents the probability of a bridge being destroyed in an earthquake -/
def p : ℝ := 0.5

/-- Represents the probability of a bridge surviving an earthquake -/
def q : ℝ := 1 - p

/-- Represents the probability of traveling from the first island to the shore after an earthquake -/
noncomputable def travel_probability : ℝ := q / (1 - p * q)

/-- Theorem stating that the probability of traveling from the first island to the shore
    after an earthquake is 2/3 -/
theorem travel_probability_is_two_thirds :
  travel_probability = 2 / 3 := by sorry

end NUMINAMATH_CALUDE_travel_probability_is_two_thirds_l1644_164448


namespace NUMINAMATH_CALUDE_peaches_in_knapsack_l1644_164480

/-- Given a total of 60 peaches distributed among two identical bags and a knapsack,
    where the knapsack contains half as many peaches as each bag,
    prove that the number of peaches in the knapsack is 12. -/
theorem peaches_in_knapsack :
  let total_peaches : ℕ := 60
  let knapsack_peaches : ℕ := x
  let bag_peaches : ℕ := 2 * x
  x + bag_peaches + bag_peaches = total_peaches →
  x = 12 :=
by
  sorry

end NUMINAMATH_CALUDE_peaches_in_knapsack_l1644_164480


namespace NUMINAMATH_CALUDE_equation_solution_l1644_164443

theorem equation_solution :
  ∃ x : ℝ, (5 + 3.5 * x = 2 * x - 25) ∧ (x = -20) := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l1644_164443


namespace NUMINAMATH_CALUDE_three_digit_cubes_divisible_by_27_l1644_164469

theorem three_digit_cubes_divisible_by_27 :
  (∃! (s : Finset ℕ), ∀ n : ℕ, n ∈ s ↔ 
    (100 ≤ n^3 ∧ n^3 ≤ 999 ∧ n^3 % 27 = 0)) ∧
  (∃ s : Finset ℕ, (∀ n : ℕ, n ∈ s ↔ 
    (100 ≤ n^3 ∧ n^3 ≤ 999 ∧ n^3 % 27 = 0)) ∧ 
    s.card = 2) :=
by sorry

end NUMINAMATH_CALUDE_three_digit_cubes_divisible_by_27_l1644_164469


namespace NUMINAMATH_CALUDE_watch_cost_price_l1644_164432

/-- Proves that the cost price of a watch is 2000, given specific selling conditions. -/
theorem watch_cost_price : 
  ∀ (cost_price : ℝ),
  (cost_price * 0.8 + 520 = cost_price * 1.06) →
  cost_price = 2000 := by
sorry

end NUMINAMATH_CALUDE_watch_cost_price_l1644_164432


namespace NUMINAMATH_CALUDE_minimum_yellow_balls_l1644_164420

theorem minimum_yellow_balls
  (g : ℕ) -- number of green balls
  (y : ℕ) -- number of yellow balls
  (o : ℕ) -- number of orange balls
  (h1 : o ≥ g / 3)  -- orange balls at least one-third of green balls
  (h2 : o ≤ y / 4)  -- orange balls at most one-fourth of yellow balls
  (h3 : g + o ≥ 75) -- combined green and orange balls at least 75
  : y ≥ 76 := by
  sorry

#check minimum_yellow_balls

end NUMINAMATH_CALUDE_minimum_yellow_balls_l1644_164420


namespace NUMINAMATH_CALUDE_parabola_intersection_l1644_164403

theorem parabola_intersection :
  let f (x : ℝ) := 3 * x^2 - 4 * x + 2
  let g (x : ℝ) := 9 * x^2 + 6 * x + 2
  ∀ x y : ℝ, f x = y ∧ g x = y ↔ (x = 0 ∧ y = 2) ∨ (x = -5/3 ∧ y = 17) :=
by sorry

end NUMINAMATH_CALUDE_parabola_intersection_l1644_164403


namespace NUMINAMATH_CALUDE_largest_non_36multiple_composite_sum_l1644_164446

def is_composite (n : ℕ) : Prop := ∃ (a b : ℕ), a > 1 ∧ b > 1 ∧ n = a * b

def is_sum_of_36multiple_and_composite (n : ℕ) : Prop :=
  ∃ (k m : ℕ), k > 0 ∧ is_composite m ∧ n = 36 * k + m

theorem largest_non_36multiple_composite_sum :
  (∀ n > 304, is_sum_of_36multiple_and_composite n) ∧
  ¬is_sum_of_36multiple_and_composite 304 := by
  sorry

end NUMINAMATH_CALUDE_largest_non_36multiple_composite_sum_l1644_164446


namespace NUMINAMATH_CALUDE_max_q_minus_r_l1644_164445

theorem max_q_minus_r (q r : ℕ) (h : 1025 = 23 * q + r) (hq : q > 0) (hr : r > 0) :
  q - r ≤ 31 ∧ ∃ (q' r' : ℕ), 1025 = 23 * q' + r' ∧ q' > 0 ∧ r' > 0 ∧ q' - r' = 31 :=
sorry

end NUMINAMATH_CALUDE_max_q_minus_r_l1644_164445


namespace NUMINAMATH_CALUDE_function_symmetry_l1644_164422

/-- A function f : ℝ → ℝ is symmetric with respect to the point (a, b) if f(x) + f(2a - x) = 2b for all x ∈ ℝ -/
def SymmetricAboutPoint (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x, f x + f (2 * a - x) = 2 * b

/-- The function property given in the problem -/
def FunctionProperty (f : ℝ → ℝ) : Prop :=
  ∀ x, f (x + 2) = -f (-x)

theorem function_symmetry (f : ℝ → ℝ) (h : FunctionProperty f) :
  SymmetricAboutPoint f 1 0 := by
  sorry


end NUMINAMATH_CALUDE_function_symmetry_l1644_164422


namespace NUMINAMATH_CALUDE_bus_children_difference_l1644_164466

/-- Given the initial number of children on a bus, the number of children who got on,
    and the final number of children on the bus, this theorem proves that
    2 more children got on than got off. -/
theorem bus_children_difference (initial : ℕ) (got_on : ℕ) (final : ℕ)
    (h1 : initial = 28)
    (h2 : got_on = 82)
    (h3 : final = 30)
    (h4 : final = initial + got_on - (initial + got_on - final)) :
  got_on - (initial + got_on - final) = 2 :=
by sorry

end NUMINAMATH_CALUDE_bus_children_difference_l1644_164466


namespace NUMINAMATH_CALUDE_problem_solution_l1644_164439

theorem problem_solution (x y : ℝ) (hx : x = 1/2) (hy : y = 2) :
  (1/3) * x^8 * y^9 = 2/3 := by
sorry

end NUMINAMATH_CALUDE_problem_solution_l1644_164439


namespace NUMINAMATH_CALUDE_matrix_with_unequal_rank_and_square_rank_l1644_164402

theorem matrix_with_unequal_rank_and_square_rank
  (n : ℕ)
  (h_n : n ≥ 2)
  (A : Matrix (Fin n) (Fin n) ℂ)
  (h_rank : Matrix.rank A ≠ Matrix.rank (A * A)) :
  ∃ (B : Matrix (Fin n) (Fin n) ℂ), B ≠ 0 ∧ A * B = 0 ∧ B * A = 0 ∧ B * B = 0 := by
sorry

end NUMINAMATH_CALUDE_matrix_with_unequal_rank_and_square_rank_l1644_164402


namespace NUMINAMATH_CALUDE_jones_family_probability_l1644_164491

theorem jones_family_probability :
  let n : ℕ := 8  -- total number of children
  let k : ℕ := 4  -- number of sons (or daughters)
  let p : ℚ := 1/2  -- probability of a child being a son (or daughter)
  Nat.choose n k * p^k * (1-p)^(n-k) = 35/128 :=
by sorry

end NUMINAMATH_CALUDE_jones_family_probability_l1644_164491


namespace NUMINAMATH_CALUDE_triangle_area_proof_l1644_164457

theorem triangle_area_proof (a b c : ℝ) (h1 : a + b + c = 10 + 2 * Real.sqrt 7) 
  (h2 : a / 2 = b / 3) (h3 : a / 2 = c / Real.sqrt 7) : 
  Real.sqrt ((1 / 4) * (c^2 * a^2 - ((c^2 + a^2 - b^2) / 2)^2)) = 6 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_triangle_area_proof_l1644_164457


namespace NUMINAMATH_CALUDE_initial_marbles_l1644_164461

theorem initial_marbles (initial : ℕ) (lost : ℕ) (remaining : ℕ) : 
  lost = 5 → remaining = 4 → initial = lost + remaining :=
by
  sorry

end NUMINAMATH_CALUDE_initial_marbles_l1644_164461


namespace NUMINAMATH_CALUDE_sum_of_first_cards_theorem_l1644_164415

/-- The sum of points of the first cards in card piles -/
def sum_of_first_cards (a b c d : ℕ) : ℕ :=
  b * (c + 1) + d - a

/-- Theorem stating the sum of points of the first cards in card piles -/
theorem sum_of_first_cards_theorem (a b c d : ℕ) :
  ∃ x : ℕ, x = sum_of_first_cards a b c d :=
by
  sorry

#check sum_of_first_cards_theorem

end NUMINAMATH_CALUDE_sum_of_first_cards_theorem_l1644_164415


namespace NUMINAMATH_CALUDE_complex_modulus_equality_l1644_164411

theorem complex_modulus_equality (x : ℝ) (h : x > 0) :
  Complex.abs (10 + Complex.I * x) = 15 ↔ x = 5 * Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_complex_modulus_equality_l1644_164411


namespace NUMINAMATH_CALUDE_minimum_cans_needed_l1644_164494

/-- The number of ounces in each can -/
def can_capacity : ℕ := 10

/-- The minimum number of ounces required -/
def min_ounces : ℕ := 120

/-- The minimum number of cans needed to provide at least the required ounces -/
def min_cans : ℕ := 12

theorem minimum_cans_needed :
  (min_cans * can_capacity ≥ min_ounces) ∧
  (∀ n : ℕ, n * can_capacity ≥ min_ounces → n ≥ min_cans) :=
by sorry

end NUMINAMATH_CALUDE_minimum_cans_needed_l1644_164494


namespace NUMINAMATH_CALUDE_arithmetic_sequence_problem_l1644_164488

/-- Arithmetic sequence with first term 4 and common difference 2 -/
def a (n : ℕ) : ℕ := 4 + 2 * (n - 1)

/-- Sum of first n terms of the arithmetic sequence -/
def S (n : ℕ) : ℕ := n * (2 * 4 + (n - 1) * 2) / 2

/-- The proposition to be proved -/
theorem arithmetic_sequence_problem :
  ∃ (k : ℕ), k > 0 ∧ S k - a (k + 5) = 44 ∧ k = 7 := by
  sorry


end NUMINAMATH_CALUDE_arithmetic_sequence_problem_l1644_164488


namespace NUMINAMATH_CALUDE_megaTek_circle_graph_error_l1644_164410

theorem megaTek_circle_graph_error :
  let total_degrees : ℕ := 360
  let manufacturing_degrees : ℕ := 252
  let administration_degrees : ℕ := 68
  let research_degrees : ℕ := 40
  manufacturing_degrees + administration_degrees + research_degrees = total_degrees :=
by
  sorry

end NUMINAMATH_CALUDE_megaTek_circle_graph_error_l1644_164410


namespace NUMINAMATH_CALUDE_sqrt_two_squared_times_three_l1644_164453

theorem sqrt_two_squared_times_three : 4 - (Real.sqrt 2)^2 * 3 = -2 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_two_squared_times_three_l1644_164453


namespace NUMINAMATH_CALUDE_shaded_area_is_63_l1644_164430

/-- Represents a rectangle with width and height -/
structure Rectangle where
  width : ℝ
  height : ℝ

/-- Calculates the area of a rectangle -/
def Rectangle.area (r : Rectangle) : ℝ := r.width * r.height

/-- Represents the configuration of two intersecting rectangles -/
structure IntersectingRectangles where
  rect1 : Rectangle
  rect2 : Rectangle
  overlap : Rectangle

/-- Calculates the shaded area formed by two intersecting rectangles -/
def IntersectingRectangles.shadedArea (ir : IntersectingRectangles) : ℝ :=
  ir.rect1.area + ir.rect2.area - ir.overlap.area

/-- The main theorem stating that the shaded area is 63 square units -/
theorem shaded_area_is_63 (ir : IntersectingRectangles)
  (h1 : ir.rect1 = { width := 4, height := 12 })
  (h2 : ir.rect2 = { width := 5, height := 7 })
  (h3 : ir.overlap = { width := 4, height := 5 }) :
  ir.shadedArea = 63 := by
  sorry

#check shaded_area_is_63

end NUMINAMATH_CALUDE_shaded_area_is_63_l1644_164430


namespace NUMINAMATH_CALUDE_binary_11011000_equals_quaternary_3120_l1644_164495

/-- Converts a binary (base 2) number to its decimal representation -/
def binary_to_decimal (binary : List Bool) : ℕ :=
  binary.foldr (λ b acc => 2 * acc + if b then 1 else 0) 0

/-- Converts a decimal number to its quaternary (base 4) representation -/
def decimal_to_quaternary (n : ℕ) : List ℕ :=
  if n = 0 then [0] else
    let rec aux (m : ℕ) (acc : List ℕ) : List ℕ :=
      if m = 0 then acc else aux (m / 4) ((m % 4) :: acc)
    aux n []

/-- The binary representation of 11011000₂ -/
def binary_11011000 : List Bool := [true, true, false, true, true, false, false, false]

theorem binary_11011000_equals_quaternary_3120 :
  decimal_to_quaternary (binary_to_decimal binary_11011000) = [3, 1, 2, 0] := by
  sorry

#eval decimal_to_quaternary (binary_to_decimal binary_11011000)

end NUMINAMATH_CALUDE_binary_11011000_equals_quaternary_3120_l1644_164495


namespace NUMINAMATH_CALUDE_min_value_2x_plus_y_compare_expressions_l1644_164435

-- Problem 1
theorem min_value_2x_plus_y (x y : ℝ) (hx : x > 0) (hy : y > 0) 
  (h : 1/x + 2/(y+1) = 2) : 
  ∀ x' y' : ℝ, x' > 0 → y' > 0 → 1/x' + 2/(y'+1) = 2 → 2*x + y ≤ 2*x' + y' :=
sorry

-- Problem 2
theorem compare_expressions (a b : ℝ) (ha : a > 0) (hb : b > 0) 
  (hab : a + b = 1) : 
  8 - 1/a ≤ 1/b + 1/(a*b) :=
sorry

end NUMINAMATH_CALUDE_min_value_2x_plus_y_compare_expressions_l1644_164435


namespace NUMINAMATH_CALUDE_binomial_inequality_l1644_164407

theorem binomial_inequality (n k : ℕ) (h1 : n > k) (h2 : k > 0) :
  (1 : ℝ) / (n + 1 : ℝ) * (n^n : ℝ) / ((k^k : ℝ) * ((n - k)^(n - k) : ℝ)) <
  (n.factorial : ℝ) / ((k.factorial : ℝ) * ((n - k).factorial : ℝ)) ∧
  (n.factorial : ℝ) / ((k.factorial : ℝ) * ((n - k).factorial : ℝ)) <
  (n^n : ℝ) / ((k^k : ℝ) * ((n - k)^(n - k) : ℝ)) :=
by sorry

end NUMINAMATH_CALUDE_binomial_inequality_l1644_164407


namespace NUMINAMATH_CALUDE_m_range_l1644_164498

def p (x : ℝ) : Prop := x^2 - 2*x - 8 ≤ 0

def q (x m : ℝ) : Prop := (x - (1 - m)) * (x - (1 + m)) ≤ 0

theorem m_range (m : ℝ) :
  (m < 0) →
  (∀ x, p x → q x m) →
  m ≤ -3 :=
sorry

end NUMINAMATH_CALUDE_m_range_l1644_164498


namespace NUMINAMATH_CALUDE_volunteer_distribution_count_l1644_164473

def distribute_volunteers (n : ℕ) (k : ℕ) : ℕ :=
  sorry

theorem volunteer_distribution_count : 
  distribute_volunteers 5 4 = 216 :=
sorry

end NUMINAMATH_CALUDE_volunteer_distribution_count_l1644_164473


namespace NUMINAMATH_CALUDE_ring_binder_price_l1644_164493

/-- Proves that the original price of each ring-binder was $20 given the problem conditions -/
theorem ring_binder_price : 
  ∀ (original_backpack_price backpack_price_increase 
     ring_binder_price_decrease num_ring_binders total_spent : ℕ),
  original_backpack_price = 50 →
  backpack_price_increase = 5 →
  ring_binder_price_decrease = 2 →
  num_ring_binders = 3 →
  total_spent = 109 →
  ∃ (original_ring_binder_price : ℕ),
    original_ring_binder_price = 20 ∧
    (original_backpack_price + backpack_price_increase) + 
    num_ring_binders * (original_ring_binder_price - ring_binder_price_decrease) = total_spent :=
by
  sorry

end NUMINAMATH_CALUDE_ring_binder_price_l1644_164493


namespace NUMINAMATH_CALUDE_product_of_seven_and_sum_l1644_164489

theorem product_of_seven_and_sum (x : ℝ) : 27 - 7 = x * 5 → 7 * (x + 5) = 63 := by
  sorry

end NUMINAMATH_CALUDE_product_of_seven_and_sum_l1644_164489


namespace NUMINAMATH_CALUDE_existence_of_numbers_l1644_164474

-- Define the sum of digits function
def sum_of_digits (n : ℕ) : ℕ := sorry

-- Theorem statement
theorem existence_of_numbers : 
  ∃ (a b c : ℕ), 
    sum_of_digits (a + b) < 5 ∧ 
    sum_of_digits (a + c) < 5 ∧ 
    sum_of_digits (b + c) < 5 ∧ 
    sum_of_digits (a + b + c) > 50 := by
  sorry

end NUMINAMATH_CALUDE_existence_of_numbers_l1644_164474


namespace NUMINAMATH_CALUDE_total_pizza_slices_l1644_164418

theorem total_pizza_slices : 
  let number_of_pizzas : ℕ := 17
  let slices_per_pizza : ℕ := 4
  number_of_pizzas * slices_per_pizza = 68 := by
sorry

end NUMINAMATH_CALUDE_total_pizza_slices_l1644_164418


namespace NUMINAMATH_CALUDE_range_of_a_l1644_164499

def p (a : ℝ) : Prop := ∃ x : ℝ, x ∈ Set.Icc (-1) 1 ∧ a^2 * x^2 + a * x - 2 = 0

def q (a : ℝ) : Prop := ∃ x : ℝ, x < 0 ∧ a * x^2 + 2 * x + 1 = 0

theorem range_of_a (a : ℝ) : 
  (p a ∨ q a) ∧ ¬(p a ∧ q a) → a > 1 ∨ (-1 < a ∧ a < 1) :=
by sorry

end NUMINAMATH_CALUDE_range_of_a_l1644_164499


namespace NUMINAMATH_CALUDE_vhs_trade_in_value_proof_l1644_164479

/-- The number of movies John has -/
def num_movies : ℕ := 100

/-- The cost of each DVD in dollars -/
def dvd_cost : ℚ := 10

/-- The total cost to replace all movies in dollars -/
def total_replacement_cost : ℚ := 800

/-- The trade-in value of each VHS in dollars -/
def vhs_trade_in_value : ℚ := 2

theorem vhs_trade_in_value_proof :
  vhs_trade_in_value * num_movies + total_replacement_cost = dvd_cost * num_movies :=
sorry

end NUMINAMATH_CALUDE_vhs_trade_in_value_proof_l1644_164479


namespace NUMINAMATH_CALUDE_smallest_stair_count_l1644_164465

theorem smallest_stair_count : ∃ n : ℕ, n = 71 ∧ n > 15 ∧ 
  n % 3 = 2 ∧ n % 7 = 1 ∧ n % 4 = 3 ∧
  ∀ m : ℕ, m > 15 → m % 3 = 2 → m % 7 = 1 → m % 4 = 3 → m ≥ n :=
by sorry

end NUMINAMATH_CALUDE_smallest_stair_count_l1644_164465


namespace NUMINAMATH_CALUDE_train_speed_proof_l1644_164433

/-- The speed of the second train in km/hr -/
def second_train_speed : ℝ := 40

/-- The additional distance traveled by the first train in km -/
def additional_distance : ℝ := 100

/-- The total distance between P and Q in km -/
def total_distance : ℝ := 900

/-- The speed of the first train in km/hr -/
def first_train_speed : ℝ := 50

theorem train_speed_proof :
  ∃ (t : ℝ), 
    t > 0 ∧
    first_train_speed * t = second_train_speed * t + additional_distance ∧
    first_train_speed * t + second_train_speed * t = total_distance :=
by sorry

end NUMINAMATH_CALUDE_train_speed_proof_l1644_164433


namespace NUMINAMATH_CALUDE_unique_plane_through_and_parallel_l1644_164413

-- Define the concept of skew lines
def SkewLines (l₁ l₂ : Set (Point)) : Prop := sorry

-- Define a plane passing through a line and parallel to another line
def PlaneThroughAndParallel (π : Set (Point)) (l₁ l₂ : Set (Point)) : Prop := sorry

theorem unique_plane_through_and_parallel 
  (l₁ l₂ : Set (Point)) 
  (h : SkewLines l₁ l₂) : 
  ∃! π, PlaneThroughAndParallel π l₁ l₂ := by sorry

end NUMINAMATH_CALUDE_unique_plane_through_and_parallel_l1644_164413


namespace NUMINAMATH_CALUDE_q_array_sum_formula_l1644_164486

/-- Definition of a 1/q-array sum -/
def qArraySum (q : ℚ) : ℚ :=
  (2 * q^2) / ((2*q - 1) * (q - 1))

/-- Theorem: The sum of all terms in a 1/q-array with the given properties is (2q^2) / ((2q-1)(q-1)) -/
theorem q_array_sum_formula (q : ℚ) (hq : q ≠ 0) (hq1 : q ≠ 1/2) (hq2 : q ≠ 1) : 
  qArraySum q = ∑' (r : ℕ) (c : ℕ), (1 / (2*q)^r) * (1 / q^c) :=
sorry

#eval (qArraySum 1220).num % 1220 + (qArraySum 1220).den % 1220

end NUMINAMATH_CALUDE_q_array_sum_formula_l1644_164486


namespace NUMINAMATH_CALUDE_third_side_length_l1644_164467

/-- A triangle with known perimeter and two side lengths -/
structure Triangle where
  side1 : ℝ
  side2 : ℝ
  side3 : ℝ
  perimeter : ℝ
  perimeter_eq : side1 + side2 + side3 = perimeter

/-- The theorem stating that for a triangle with two sides 7 and 15, and perimeter 32, the third side is 10 -/
theorem third_side_length (t : Triangle) 
    (h1 : t.side1 = 7)
    (h2 : t.side2 = 15)
    (h3 : t.perimeter = 32) : 
  t.side3 = 10 := by
  sorry


end NUMINAMATH_CALUDE_third_side_length_l1644_164467


namespace NUMINAMATH_CALUDE_largest_c_for_negative_two_in_range_l1644_164408

/-- The function f(x) with parameter c -/
def f (c : ℝ) (x : ℝ) : ℝ := x^2 + 4*x + c

/-- The largest value of c such that -2 is in the range of f(x) -/
theorem largest_c_for_negative_two_in_range :
  ∃ (c_max : ℝ), 
    (∃ (x : ℝ), f c_max x = -2) ∧ 
    (∀ (c : ℝ), c > c_max → ¬∃ (x : ℝ), f c x = -2) ∧
    c_max = 2 := by
  sorry

end NUMINAMATH_CALUDE_largest_c_for_negative_two_in_range_l1644_164408


namespace NUMINAMATH_CALUDE_quadratic_real_roots_condition_l1644_164412

theorem quadratic_real_roots_condition (k : ℝ) : 
  (∃ x : ℝ, (k - 1) * x^2 + 2 * x + 1 = 0) ↔ (k ≤ 2 ∧ k ≠ 1) :=
sorry

end NUMINAMATH_CALUDE_quadratic_real_roots_condition_l1644_164412


namespace NUMINAMATH_CALUDE_art_gallery_problem_l1644_164476

theorem art_gallery_problem (total_pieces : ℕ) 
  (h1 : total_pieces / 3 = total_pieces - (total_pieces * 2 / 3))  -- 1/3 of pieces are displayed
  (h2 : (total_pieces / 3) / 6 = (total_pieces / 3) - (5 * total_pieces / 18))  -- 1/6 of displayed pieces are sculptures
  (h3 : (total_pieces * 2 / 3) / 3 = (total_pieces * 2 / 3) - (2 * total_pieces / 3))  -- 1/3 of not displayed pieces are paintings
  (h4 : 2 * (total_pieces * 2 / 3) / 3 = 1200)  -- 1200 sculptures are not on display
  : total_pieces = 2700 := by
  sorry

end NUMINAMATH_CALUDE_art_gallery_problem_l1644_164476


namespace NUMINAMATH_CALUDE_smallest_greater_perfect_square_l1644_164468

theorem smallest_greater_perfect_square (a : ℕ) (h : ∃ k : ℕ, a = k^2) :
  (∀ n : ℕ, n > a ∧ (∃ m : ℕ, n = m^2) → n ≥ a + 2*Int.sqrt a + 1) ∧
  (∃ m : ℕ, a + 2*Int.sqrt a + 1 = m^2) :=
sorry

end NUMINAMATH_CALUDE_smallest_greater_perfect_square_l1644_164468


namespace NUMINAMATH_CALUDE_d_necessary_not_sufficient_for_a_l1644_164424

-- Define propositions
variable (A B C D : Prop)

-- Define the relationships between propositions
variable (h1 : A → B ∧ ¬(B → A))
variable (h2 : B → C ∧ ¬(C → B))
variable (h3 : D ↔ C)

-- Theorem to prove
theorem d_necessary_not_sufficient_for_a :
  (D → A) ∧ ¬(A → D) :=
sorry

end NUMINAMATH_CALUDE_d_necessary_not_sufficient_for_a_l1644_164424


namespace NUMINAMATH_CALUDE_special_sequence_sum_l1644_164450

/-- A sequence where the sum of two terms with a term between them increases by a constant amount -/
def SpecialSequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 2) + a (n + 3) = a n + a (n + 1) + d

theorem special_sequence_sum (a : ℕ → ℝ) (h : SpecialSequence a)
    (h1 : a 2 + a 3 = 4) (h2 : a 4 + a 5 = 6) :
  a 9 + a 10 = 10 := by
  sorry

end NUMINAMATH_CALUDE_special_sequence_sum_l1644_164450


namespace NUMINAMATH_CALUDE_quadratic_inequality_l1644_164460

theorem quadratic_inequality (x : ℝ) : x^2 - 6*x - 16 > 0 ↔ x < -2 ∨ x > 8 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_l1644_164460


namespace NUMINAMATH_CALUDE_polar_to_cartesian_line_l1644_164416

/-- The curve defined by the polar equation r = 1 / (2sin(θ) - cos(θ)) is a line. -/
theorem polar_to_cartesian_line :
  ∃ (a b c : ℝ), a ≠ 0 ∨ b ≠ 0 ∧
  ∀ (θ : ℝ), ∀ (r : ℝ), r > 0 →
  r = 1 / (2 * Real.sin θ - Real.cos θ) →
  a * (r * Real.cos θ) + b * (r * Real.sin θ) + c = 0 :=
sorry

end NUMINAMATH_CALUDE_polar_to_cartesian_line_l1644_164416


namespace NUMINAMATH_CALUDE_number_of_officers_l1644_164470

/-- Prove the number of officers in an office given average salaries and number of non-officers -/
theorem number_of_officers
  (avg_salary : ℝ)
  (avg_salary_officers : ℝ)
  (avg_salary_non_officers : ℝ)
  (num_non_officers : ℕ)
  (h1 : avg_salary = 120)
  (h2 : avg_salary_officers = 430)
  (h3 : avg_salary_non_officers = 110)
  (h4 : num_non_officers = 465) :
  ∃ (num_officers : ℕ),
    avg_salary * (num_officers + num_non_officers) =
    avg_salary_officers * num_officers + avg_salary_non_officers * num_non_officers ∧
    num_officers = 15 :=
by sorry

end NUMINAMATH_CALUDE_number_of_officers_l1644_164470


namespace NUMINAMATH_CALUDE_debate_team_group_size_l1644_164442

theorem debate_team_group_size 
  (num_boys : ℕ) 
  (num_girls : ℕ) 
  (num_groups : ℕ) 
  (h1 : num_boys = 31)
  (h2 : num_girls = 32)
  (h3 : num_groups = 7) :
  (num_boys + num_girls) / num_groups = 9 := by
sorry

end NUMINAMATH_CALUDE_debate_team_group_size_l1644_164442


namespace NUMINAMATH_CALUDE_alexander_buckwheat_investment_l1644_164434

theorem alexander_buckwheat_investment (initial_price : ℝ) (final_price : ℝ)
  (one_year_rate_2020 : ℝ) (two_year_rate : ℝ) (one_year_rate_2021 : ℝ)
  (h1 : initial_price = 70)
  (h2 : final_price = 100)
  (h3 : one_year_rate_2020 = 0.1)
  (h4 : two_year_rate = 0.08)
  (h5 : one_year_rate_2021 = 0.05) :
  (initial_price * (1 + one_year_rate_2020) * (1 + one_year_rate_2021) < final_price) ∧
  (initial_price * (1 + two_year_rate)^2 < final_price) :=
by sorry

end NUMINAMATH_CALUDE_alexander_buckwheat_investment_l1644_164434


namespace NUMINAMATH_CALUDE_f_properties_l1644_164414

/-- A function f from positive integers to positive integers with a parameter k -/
def f (k : ℕ+) : ℕ+ → ℕ+ :=
  fun n => if n > k then n - k else sorry

/-- The number of different functions f when k = 5 and 1 ≤ f(n) ≤ 2 for n ≤ 5 -/
def count_functions : ℕ := sorry

theorem f_properties :
  (∃ (a : ℕ+), f 1 1 = a) ∧
  count_functions = 32 := by sorry

end NUMINAMATH_CALUDE_f_properties_l1644_164414


namespace NUMINAMATH_CALUDE_goldfish_sales_l1644_164464

theorem goldfish_sales (buy_price sell_price tank_cost shortfall_percent : ℚ) 
  (h1 : buy_price = 25 / 100)
  (h2 : sell_price = 75 / 100)
  (h3 : tank_cost = 100)
  (h4 : shortfall_percent = 45 / 100) :
  (tank_cost * (1 - shortfall_percent)) / (sell_price - buy_price) = 110 := by
sorry

end NUMINAMATH_CALUDE_goldfish_sales_l1644_164464


namespace NUMINAMATH_CALUDE_exists_number_of_1_and_2_divisible_by_2_pow_l1644_164490

/-- A function that checks if a natural number is composed of only digits 1 and 2 -/
def isComposedOf1And2 (x : ℕ) : Prop :=
  ∀ d, d ∈ x.digits 10 → d = 1 ∨ d = 2

/-- Theorem stating that for all natural numbers n, there exists a number x
    composed of only digits 1 and 2 such that x is divisible by 2^n -/
theorem exists_number_of_1_and_2_divisible_by_2_pow (n : ℕ) :
  ∃ x : ℕ, isComposedOf1And2 x ∧ (2^n ∣ x) := by
  sorry

end NUMINAMATH_CALUDE_exists_number_of_1_and_2_divisible_by_2_pow_l1644_164490


namespace NUMINAMATH_CALUDE_cube_configurations_l1644_164441

/-- Represents a rotation in 3D space -/
structure Rotation :=
  (fixedConfigurations : ℕ)

/-- The group of rotations for a cube -/
def rotationGroup : Finset Rotation := sorry

/-- The number of white unit cubes -/
def numWhiteCubes : ℕ := 5

/-- The number of blue unit cubes -/
def numBlueCubes : ℕ := 3

/-- The total number of unit cubes -/
def totalCubes : ℕ := numWhiteCubes + numBlueCubes

/-- Calculates the number of fixed configurations for a given rotation -/
def fixedConfigurations (r : Rotation) : ℕ := r.fixedConfigurations

/-- Applies Burnside's Lemma to calculate the number of distinct configurations -/
def distinctConfigurations : ℕ :=
  (rotationGroup.sum fixedConfigurations) / rotationGroup.card

theorem cube_configurations :
  distinctConfigurations = 3 := by sorry

end NUMINAMATH_CALUDE_cube_configurations_l1644_164441


namespace NUMINAMATH_CALUDE_decision_box_two_exits_l1644_164484

-- Define the types of program blocks
inductive ProgramBlock
  | TerminationBox
  | InputOutputBox
  | ProcessingBox
  | DecisionBox

-- Define a function that returns the number of exit directions for each program block
def exitDirections (block : ProgramBlock) : Nat :=
  match block with
  | ProgramBlock.TerminationBox => 1
  | ProgramBlock.InputOutputBox => 1
  | ProgramBlock.ProcessingBox => 1
  | ProgramBlock.DecisionBox => 2

-- Theorem statement
theorem decision_box_two_exits :
  ∀ (block : ProgramBlock), exitDirections block = 2 ↔ block = ProgramBlock.DecisionBox :=
by sorry


end NUMINAMATH_CALUDE_decision_box_two_exits_l1644_164484


namespace NUMINAMATH_CALUDE_distribution_problem_l1644_164421

theorem distribution_problem (total : ℕ) (a b c : ℕ) : 
  total = 370 →
  total = a + b + c →
  b + c = a + 50 →
  (a : ℚ) / b = (b : ℚ) / c →
  a = 160 ∧ b = 120 ∧ c = 90 := by
  sorry

end NUMINAMATH_CALUDE_distribution_problem_l1644_164421


namespace NUMINAMATH_CALUDE_second_term_is_plus_minus_one_l1644_164406

/-- A geometric sequence with a_1 = 1/5 and a_3 = 5 -/
def geometric_sequence (a : ℕ → ℚ) : Prop :=
  a 1 = 1/5 ∧ a 3 = 5 ∧ ∃ r : ℚ, ∀ n : ℕ, a (n + 1) = a n * r

/-- The second term of the geometric sequence is either 1 or -1 -/
theorem second_term_is_plus_minus_one (a : ℕ → ℚ) (h : geometric_sequence a) :
  a 2 = 1 ∨ a 2 = -1 :=
sorry

end NUMINAMATH_CALUDE_second_term_is_plus_minus_one_l1644_164406


namespace NUMINAMATH_CALUDE_perfect_square_5ab4_l1644_164452

def is_four_digit (n : ℕ) : Prop := 1000 ≤ n ∧ n ≤ 9999

def ends_with_four (n : ℕ) : Prop := n % 10 = 4

def starts_with_five (n : ℕ) : Prop := 5000 ≤ n ∧ n < 6000

def is_5ab4_form (n : ℕ) : Prop :=
  ∃ (a b : ℕ), a < 10 ∧ b < 10 ∧ n = 5000 + 100 * a + 10 * b + 4

theorem perfect_square_5ab4 (n : ℕ) :
  is_four_digit n →
  ends_with_four n →
  starts_with_five n →
  is_5ab4_form n →
  ∃ (m : ℕ), n = m^2 →
  ∃ (a b : ℕ), a < 10 ∧ b < 10 ∧ n = 5000 + 100 * a + 10 * b + 4 ∧ a + b = 9 :=
sorry

end NUMINAMATH_CALUDE_perfect_square_5ab4_l1644_164452


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_l1644_164483

theorem quadratic_inequality_solution (a b : ℝ) :
  (∀ x : ℝ, ax^2 + x + b > 0 ↔ 1 < x ∧ x < 2) →
  a + b = -1 := by
sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_l1644_164483


namespace NUMINAMATH_CALUDE_least_positive_integer_congruence_l1644_164478

theorem least_positive_integer_congruence :
  ∃ (x : ℕ), x > 0 ∧ (x + 5419 : ℤ) ≡ 3789 [ZMOD 15] ∧
  ∀ (y : ℕ), y > 0 ∧ (y + 5419 : ℤ) ≡ 3789 [ZMOD 15] → x ≤ y :=
by sorry

end NUMINAMATH_CALUDE_least_positive_integer_congruence_l1644_164478


namespace NUMINAMATH_CALUDE_grandma_olga_grandchildren_l1644_164444

/-- Represents the number of grandchildren Grandma Olga has -/
def total_grandchildren (num_daughters num_sons : ℕ) (sons_per_daughter daughters_per_son : ℕ) : ℕ :=
  num_daughters * sons_per_daughter + num_sons * daughters_per_son

/-- Proves that Grandma Olga has 33 grandchildren given the specified conditions -/
theorem grandma_olga_grandchildren :
  total_grandchildren 3 3 6 5 = 33 := by
  sorry

end NUMINAMATH_CALUDE_grandma_olga_grandchildren_l1644_164444


namespace NUMINAMATH_CALUDE_smallest_number_divisible_l1644_164440

theorem smallest_number_divisible (a b c d e f : ℕ) (h1 : a = 35 ∧ b = 66)
  (h2 : c = 28 ∧ d = 165) (h3 : e = 25 ∧ f = 231) :
  ∃ (n : ℚ), n = 700 / 33 ∧
  (∃ (k1 k2 k3 : ℕ), n / (a / b) = k1 ∧ n / (c / d) = k2 ∧ n / (e / f) = k3) ∧
  ∀ (m : ℚ), m < n →
  ¬(∃ (l1 l2 l3 : ℕ), m / (a / b) = l1 ∧ m / (c / d) = l2 ∧ m / (e / f) = l3) :=
sorry

end NUMINAMATH_CALUDE_smallest_number_divisible_l1644_164440


namespace NUMINAMATH_CALUDE_jellybean_count_jellybean_count_proof_l1644_164449

theorem jellybean_count : ℕ → ℕ → ℕ → ℕ → Prop :=
  fun black green orange total =>
    (black = 8) →
    (green = black + 2) →
    (orange = green - 1) →
    (total = black + green + orange) →
    (total = 27)

-- The proof is omitted
theorem jellybean_count_proof : jellybean_count 8 10 9 27 := by sorry

end NUMINAMATH_CALUDE_jellybean_count_jellybean_count_proof_l1644_164449


namespace NUMINAMATH_CALUDE_area_difference_l1644_164423

/-- A right isosceles triangle with base length 1 -/
structure RightIsoscelesTriangle where
  base : ℝ
  base_eq_one : base = 1

/-- Configuration of two identical squares in the triangle (Figure 2) -/
structure SquareConfig2 (t : RightIsoscelesTriangle) where
  side_length : ℝ
  side_length_eq : side_length = 1 / 4

/-- Configuration of two identical squares in the triangle (Figure 3) -/
structure SquareConfig3 (t : RightIsoscelesTriangle) where
  side_length : ℝ
  side_length_eq : side_length = Real.sqrt 2 / 6

/-- Total area of squares in Configuration 2 -/
def totalArea2 (t : RightIsoscelesTriangle) (c : SquareConfig2 t) : ℝ :=
  2 * c.side_length ^ 2

/-- Total area of squares in Configuration 3 -/
def totalArea3 (t : RightIsoscelesTriangle) (c : SquareConfig3 t) : ℝ :=
  2 * c.side_length ^ 2

/-- The main theorem stating the difference in areas -/
theorem area_difference (t : RightIsoscelesTriangle) 
  (c2 : SquareConfig2 t) (c3 : SquareConfig3 t) : 
  totalArea2 t c2 - totalArea3 t c3 = 1 / 72 := by
  sorry

end NUMINAMATH_CALUDE_area_difference_l1644_164423


namespace NUMINAMATH_CALUDE_solution_satisfies_equation_l1644_164428

theorem solution_satisfies_equation :
  let x : ℝ := 1
  let y : ℝ := -1
  x - 2 * y = 3 := by sorry

end NUMINAMATH_CALUDE_solution_satisfies_equation_l1644_164428


namespace NUMINAMATH_CALUDE_S_intersect_T_l1644_164438

noncomputable def S : Set ℝ := {y | ∃ x, y = 2^x}
def T : Set ℝ := {x | Real.log (x - 1) < 0}

theorem S_intersect_T : S ∩ T = {x | 1 < x ∧ x < 2} := by sorry

end NUMINAMATH_CALUDE_S_intersect_T_l1644_164438


namespace NUMINAMATH_CALUDE_problem_solution_l1644_164404

noncomputable def f (x : ℝ) : ℝ := Real.exp x * Real.log x + (2 / x) * Real.exp (x - 1)

noncomputable def g (x : ℝ) : ℝ := x * Real.exp (-x) - 2 / Real.exp 1

theorem problem_solution (x : ℝ) (hx : x > 0) :
  f 1 = 2 ∧ 
  (deriv f) 1 = Real.exp 1 ∧
  (∀ y > 0, g y ≤ -1 / Real.exp 1) ∧
  (∀ y > 0, g y = -1 / Real.exp 1 → y = 1) ∧
  f x > 1 :=
sorry

end NUMINAMATH_CALUDE_problem_solution_l1644_164404


namespace NUMINAMATH_CALUDE_prop1_prop2_prop3_l1644_164426

-- Define the function f(x)
def f (p q x : ℝ) : ℝ := x * abs x + p * x + q

-- Proposition 1: When q = 0, f(x) is an odd function
theorem prop1 (p : ℝ) : 
  ∀ x : ℝ, f p 0 (-x) = -(f p 0 x) := by sorry

-- Proposition 2: The graph of y = f(x) is symmetric with respect to the point (0,q)
theorem prop2 (p q : ℝ) :
  ∀ x : ℝ, f p q x - q = -(f p q (-x) - q) := by sorry

-- Proposition 3: When p = 0 and q > 0, the equation f(x) = 0 has exactly one real root
theorem prop3 (q : ℝ) (hq : q > 0) :
  ∃! x : ℝ, f 0 q x = 0 := by sorry

end NUMINAMATH_CALUDE_prop1_prop2_prop3_l1644_164426


namespace NUMINAMATH_CALUDE_a_equals_3_sufficient_not_necessary_for_a_squared_9_l1644_164485

theorem a_equals_3_sufficient_not_necessary_for_a_squared_9 :
  (∀ a : ℝ, a = 3 → a^2 = 9) ∧
  (∃ a : ℝ, a ≠ 3 ∧ a^2 = 9) :=
by sorry

end NUMINAMATH_CALUDE_a_equals_3_sufficient_not_necessary_for_a_squared_9_l1644_164485


namespace NUMINAMATH_CALUDE_saturday_sales_proof_l1644_164451

/-- The number of caricatures sold on Saturday -/
def saturday_sales : ℕ := 24

/-- The price of each caricature in dollars -/
def price_per_caricature : ℚ := 20

/-- The number of caricatures sold on Sunday -/
def sunday_sales : ℕ := 16

/-- The total revenue for the weekend in dollars -/
def total_revenue : ℚ := 800

theorem saturday_sales_proof : 
  saturday_sales = 24 ∧ 
  price_per_caricature * (saturday_sales + sunday_sales : ℚ) = total_revenue :=
by sorry

end NUMINAMATH_CALUDE_saturday_sales_proof_l1644_164451


namespace NUMINAMATH_CALUDE_abc_divisible_by_four_l1644_164463

theorem abc_divisible_by_four (a b c d : ℤ) (h : a^2 + b^2 + c^2 = d^2) : 
  4 ∣ (a * b * c) := by
  sorry

end NUMINAMATH_CALUDE_abc_divisible_by_four_l1644_164463


namespace NUMINAMATH_CALUDE_absolute_value_equation_solution_l1644_164456

theorem absolute_value_equation_solution :
  ∀ x : ℝ, (|2*x - 5| = 3*x + 2) ↔ (x = 3/5) := by
  sorry

end NUMINAMATH_CALUDE_absolute_value_equation_solution_l1644_164456


namespace NUMINAMATH_CALUDE_expression_evaluation_l1644_164425

theorem expression_evaluation (x y : ℕ) (hx : x = 3) (hy : y = 2) :
  3 * x^(y + 1) + 4 * y^(x + 1) = 145 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l1644_164425


namespace NUMINAMATH_CALUDE_sum_remainder_mod_13_l1644_164462

theorem sum_remainder_mod_13 : (9001 + 9002 + 9003 + 9004) % 13 = 7 := by
  sorry

end NUMINAMATH_CALUDE_sum_remainder_mod_13_l1644_164462


namespace NUMINAMATH_CALUDE_cori_age_relation_l1644_164431

theorem cori_age_relation (cori_age aunt_age : ℕ) (years : ℕ) : 
  cori_age = 3 → aunt_age = 19 → 
  (cori_age + years : ℚ) = (1 / 3) * (aunt_age + years : ℚ) → 
  years = 5 := by sorry

end NUMINAMATH_CALUDE_cori_age_relation_l1644_164431


namespace NUMINAMATH_CALUDE_parabola_angle_theorem_l1644_164472

/-- The parabola y² = 4x with focus F -/
structure Parabola where
  F : ℝ × ℝ
  equation : ℝ → ℝ → Prop

/-- A point on the parabola -/
structure PointOnParabola (p : Parabola) where
  M : ℝ × ℝ
  on_parabola : p.equation M.1 M.2

/-- The foot of the perpendicular from a point to the directrix -/
def footOfPerpendicular (p : Parabola) (point : PointOnParabola p) : ℝ × ℝ := sorry

/-- The angle between two vectors -/
def angle (v1 v2 : ℝ × ℝ) : ℝ := sorry

theorem parabola_angle_theorem (p : Parabola) (M : PointOnParabola p) :
  p.F = (1, 0) →
  p.equation = fun x y ↦ y^2 = 4*x →
  ‖M.M - p.F‖ = 4/3 →
  angle (footOfPerpendicular p M - M.M) (p.F - M.M) = 2*π/3 := by sorry

end NUMINAMATH_CALUDE_parabola_angle_theorem_l1644_164472


namespace NUMINAMATH_CALUDE_greatest_prime_divisor_digit_sum_l1644_164436

def n : ℕ := 2^15 - 1

theorem greatest_prime_divisor_digit_sum :
  (∃ p : ℕ, Nat.Prime p ∧ p ∣ n ∧
    (∀ q : ℕ, Nat.Prime q → q ∣ n → q ≤ p) ∧
    (Nat.digits 10 p).sum = 8) :=
by sorry

end NUMINAMATH_CALUDE_greatest_prime_divisor_digit_sum_l1644_164436


namespace NUMINAMATH_CALUDE_min_vertices_is_six_l1644_164477

/-- A graph where each vertex knows exactly three others -/
def KnowledgeGraph (V : Type*) := V → Finset V

/-- Predicate to check if a vertex has exactly 3 neighbors -/
def has_three_neighbors (G : KnowledgeGraph V) (v : V) : Prop :=
  (G v).card = 3

/-- Predicate to check if among any three vertices, two are not connected -/
def has_non_connected_pair (G : KnowledgeGraph V) : Prop :=
  ∀ (a b c : V), a ≠ b ∧ b ≠ c ∧ a ≠ c →
    ¬(a ∈ G b ∧ b ∈ G c ∧ c ∈ G a)

/-- The main theorem stating the minimum number of vertices is 6 -/
theorem min_vertices_is_six (V : Type*) [Fintype V] :
  (∃ (G : KnowledgeGraph V), (∀ v, has_three_neighbors G v) ∧ has_non_connected_pair G) →
  Fintype.card V ≥ 6 :=
sorry

end NUMINAMATH_CALUDE_min_vertices_is_six_l1644_164477


namespace NUMINAMATH_CALUDE_sara_peaches_theorem_l1644_164481

/-- The number of peaches Sara picked initially -/
def initial_peaches : ℝ := 61

/-- The number of peaches Sara picked at the orchard -/
def orchard_peaches : ℝ := 24.0

/-- The total number of peaches Sara picked -/
def total_peaches : ℝ := 85

/-- Theorem stating that the initial number of peaches plus the orchard peaches equals the total peaches -/
theorem sara_peaches_theorem : initial_peaches + orchard_peaches = total_peaches := by
  sorry

end NUMINAMATH_CALUDE_sara_peaches_theorem_l1644_164481


namespace NUMINAMATH_CALUDE_linear_equation_condition_l1644_164417

/-- The equation (a-2)x^|a-1| + 3 = 9 is linear in x if and only if a = 0 -/
theorem linear_equation_condition (a : ℝ) : 
  (∃ b c : ℝ, ∀ x : ℝ, (a - 2) * x^(|a - 1|) + 3 = b * x + c) ↔ a = 0 :=
sorry

end NUMINAMATH_CALUDE_linear_equation_condition_l1644_164417


namespace NUMINAMATH_CALUDE_quadratic_roots_ratio_l1644_164475

theorem quadratic_roots_ratio (x₁ x₂ : ℝ) : 
  x₁^2 - 2*x₁ - 8 = 0 → x₂^2 - 2*x₂ - 8 = 0 → (x₁ + x₂) / (x₁ * x₂) = -1/4 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_roots_ratio_l1644_164475


namespace NUMINAMATH_CALUDE_function_inequality_condition_l1644_164471

theorem function_inequality_condition (f : ℝ → ℝ) (a b : ℝ) :
  (f = fun x ↦ x^2 + 5*x + 6) →
  (a > 0) →
  (b > 0) →
  (∀ x, |x + 1| < b → |f x + 3| < a) ↔ (a > 11/4 ∧ b > 3/2) :=
by sorry

end NUMINAMATH_CALUDE_function_inequality_condition_l1644_164471


namespace NUMINAMATH_CALUDE_inequality_proof_l1644_164497

theorem inequality_proof (x y z : ℝ) (h1 : x ≥ 0) (h2 : y ≥ 0) (h3 : z ≥ 0) (h4 : x^2 + y^2 + z^2 = 3) :
  Real.sqrt (3 - ((x + y) / 2)^2) + Real.sqrt (3 - ((y + z) / 2)^2) + Real.sqrt (3 - ((z + x) / 2)^2) ≥ 3 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l1644_164497


namespace NUMINAMATH_CALUDE_xyz_mod_seven_l1644_164455

theorem xyz_mod_seven (x y z : ℕ) (hx : x < 7) (hy : y < 7) (hz : z < 7)
  (h1 : x + 3*y + 2*z ≡ 0 [ZMOD 7])
  (h2 : 3*x + 2*y + z ≡ 2 [ZMOD 7])
  (h3 : 2*x + y + 3*z ≡ 3 [ZMOD 7]) :
  x * y * z ≡ 1 [ZMOD 7] := by
  sorry

end NUMINAMATH_CALUDE_xyz_mod_seven_l1644_164455


namespace NUMINAMATH_CALUDE_fraction_simplification_l1644_164482

theorem fraction_simplification :
  ((5^1004)^4 - (5^1002)^4) / ((5^1003)^4 - (5^1001)^4) = 25 := by
  sorry

end NUMINAMATH_CALUDE_fraction_simplification_l1644_164482


namespace NUMINAMATH_CALUDE_simplify_expression_l1644_164454

theorem simplify_expression : ((4 + 6) * 2) / 4 - 3 / 4 = 17 / 4 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l1644_164454
