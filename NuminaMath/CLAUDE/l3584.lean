import Mathlib

namespace NUMINAMATH_CALUDE_rotate_angle_result_l3584_358486

/-- Given an initial angle of 30 degrees and a 450-degree counterclockwise rotation,
    the resulting acute angle measures 60 degrees. -/
theorem rotate_angle_result (initial_angle rotation : ℝ) (h1 : initial_angle = 30)
    (h2 : rotation = 450) : 
    (initial_angle + rotation) % 360 = 60 ∨ 360 - (initial_angle + rotation) % 360 = 60 :=
by sorry

end NUMINAMATH_CALUDE_rotate_angle_result_l3584_358486


namespace NUMINAMATH_CALUDE_product_mod_seventeen_l3584_358452

theorem product_mod_seventeen : (2022 * 2023 * 2024 * 2025) % 17 = 0 := by
  sorry

end NUMINAMATH_CALUDE_product_mod_seventeen_l3584_358452


namespace NUMINAMATH_CALUDE_quadratic_inequality_implies_m_range_l3584_358497

theorem quadratic_inequality_implies_m_range :
  (∀ x : ℝ, x^2 + m*x + 2*m - 3 ≥ 0) → m ∈ Set.Icc 2 6 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_implies_m_range_l3584_358497


namespace NUMINAMATH_CALUDE_polyhedron_sum_l3584_358449

/-- A convex polyhedron with triangular and pentagonal faces -/
structure ConvexPolyhedron where
  V : ℕ  -- number of vertices
  E : ℕ  -- number of edges
  F : ℕ  -- number of faces
  T : ℕ  -- number of triangular faces
  P : ℕ  -- number of pentagonal faces
  euler : V - E + F = 2
  faces : F = 32
  face_types : F = T + P
  vertex_edges : 2 * E = V * (T + P)
  edge_count : 3 * T + 5 * P = 2 * E

/-- Theorem stating that P + T + V = 34 for the given convex polyhedron -/
theorem polyhedron_sum (poly : ConvexPolyhedron) : poly.P + poly.T + poly.V = 34 := by
  sorry


end NUMINAMATH_CALUDE_polyhedron_sum_l3584_358449


namespace NUMINAMATH_CALUDE_expr1_simplification_expr1_evaluation_expr2_simplification_expr2_evaluation_l3584_358416

-- Define variables
variable (x y : ℝ)

-- First expression
def expr1 (x y : ℝ) : ℝ := 3*x^2*y - (2*x*y^2 - 2*(x*y - 1.5*x^2*y) + x*y) + 3*x*y^2

-- Second expression
def expr2 (x y : ℝ) : ℝ := (2*x + 3*y) - 4*y - (3*x - 2*y)

-- Theorem for the first expression
theorem expr1_simplification : 
  expr1 x y = x*y^2 + x*y := by sorry

-- Theorem for the evaluation of the first expression
theorem expr1_evaluation : 
  expr1 (-3) (-2) = -6 := by sorry

-- Theorem for the second expression
theorem expr2_simplification :
  expr2 x y = -x + y := by sorry

-- Theorem for the evaluation of the second expression
theorem expr2_evaluation :
  expr2 (-3) 2 = 5 := by sorry

end NUMINAMATH_CALUDE_expr1_simplification_expr1_evaluation_expr2_simplification_expr2_evaluation_l3584_358416


namespace NUMINAMATH_CALUDE_oak_trees_in_park_l3584_358444

/-- The number of oak trees remaining in a park after some are cut down -/
def remaining_oak_trees (initial : ℕ) (cut_down : ℕ) : ℕ :=
  initial - cut_down

/-- Theorem stating that 7 oak trees remain after cutting down 2 from an initial 9 -/
theorem oak_trees_in_park : remaining_oak_trees 9 2 = 7 := by
  sorry

end NUMINAMATH_CALUDE_oak_trees_in_park_l3584_358444


namespace NUMINAMATH_CALUDE_quadratic_function_bounds_l3584_358430

/-- Given a quadratic function f and its derivative g, 
    prove bounds on c and g(x) when f is bounded on [-1, 1] -/
theorem quadratic_function_bounds 
  (a b c : ℝ) 
  (f : ℝ → ℝ) 
  (g : ℝ → ℝ) 
  (hf : ∀ x, f x = a * x^2 + b * x + c) 
  (hg : ∀ x, g x = a * x + b) 
  (hbound : ∀ x ∈ Set.Icc (-1) 1, |f x| ≤ 1) : 
  (|c| ≤ 1) ∧ (∀ x ∈ Set.Icc (-1) 1, |g x| ≤ 2) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_function_bounds_l3584_358430


namespace NUMINAMATH_CALUDE_range_of_a_l3584_358480

-- Define the sets A and B
def A : Set ℝ := Set.Ioo 1 4
def B (a : ℝ) : Set ℝ := Set.Ioo (2 * a) (a + 1)

-- State the theorem
theorem range_of_a (a : ℝ) (h : a < 1) :
  B a ⊆ A → 1/2 ≤ a ∧ a < 1 := by
sorry

end NUMINAMATH_CALUDE_range_of_a_l3584_358480


namespace NUMINAMATH_CALUDE_opposite_seven_is_nine_or_eleven_l3584_358422

def DieNumbers : Finset ℕ := {6, 7, 8, 9, 10, 11}

def isValidOpposite (n : ℕ) : Prop :=
  n ∈ DieNumbers ∧ n ≠ 7 ∧
  ∃ (a b c d e : ℕ),
    {a, b, c, d, e, 7} = DieNumbers ∧
    (a + b + c + d = 33 ∨ a + b + c + d = 35) ∧
    (e + 7 = 16 ∨ e + 7 = 17 ∨ e + 7 = 18)

theorem opposite_seven_is_nine_or_eleven :
  ∀ n, isValidOpposite n → n = 9 ∨ n = 11 := by sorry

end NUMINAMATH_CALUDE_opposite_seven_is_nine_or_eleven_l3584_358422


namespace NUMINAMATH_CALUDE_christmas_to_january_10_l3584_358473

-- Define the days of the week
inductive DayOfWeek
  | Sunday
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday
  | Saturday

-- Define a function to get the next day
def nextDay (d : DayOfWeek) : DayOfWeek :=
  match d with
  | DayOfWeek.Sunday => DayOfWeek.Monday
  | DayOfWeek.Monday => DayOfWeek.Tuesday
  | DayOfWeek.Tuesday => DayOfWeek.Wednesday
  | DayOfWeek.Wednesday => DayOfWeek.Thursday
  | DayOfWeek.Thursday => DayOfWeek.Friday
  | DayOfWeek.Friday => DayOfWeek.Saturday
  | DayOfWeek.Saturday => DayOfWeek.Sunday

-- Define a function to advance a day by n days
def advanceDay (d : DayOfWeek) (n : Nat) : DayOfWeek :=
  match n with
  | 0 => d
  | n + 1 => advanceDay (nextDay d) n

-- Theorem statement
theorem christmas_to_january_10 :
  advanceDay DayOfWeek.Wednesday 16 = DayOfWeek.Friday := by
  sorry

end NUMINAMATH_CALUDE_christmas_to_january_10_l3584_358473


namespace NUMINAMATH_CALUDE_retailer_profit_percentage_l3584_358496

theorem retailer_profit_percentage 
  (cost_price : ℝ) 
  (markup_percentage : ℝ) 
  (discount_percentage : ℝ) 
  (h1 : markup_percentage = 60) 
  (h2 : discount_percentage = 25) 
  (h3 : cost_price > 0) : 
  let marked_price := cost_price * (1 + markup_percentage / 100)
  let selling_price := marked_price * (1 - discount_percentage / 100)
  let profit := selling_price - cost_price
  let profit_percentage := (profit / cost_price) * 100
  profit_percentage = 20 := by
sorry

end NUMINAMATH_CALUDE_retailer_profit_percentage_l3584_358496


namespace NUMINAMATH_CALUDE_fiftieth_term_of_specific_sequence_l3584_358407

/-- The nth term of an arithmetic sequence -/
def arithmeticSequenceTerm (a₁ : ℝ) (d : ℝ) (n : ℕ) : ℝ :=
  a₁ + (n - 1 : ℝ) * d

/-- Theorem: The 50th term of the arithmetic sequence with first term 3 and common difference 7 is 346 -/
theorem fiftieth_term_of_specific_sequence : 
  arithmeticSequenceTerm 3 7 50 = 346 := by
  sorry

#check fiftieth_term_of_specific_sequence

end NUMINAMATH_CALUDE_fiftieth_term_of_specific_sequence_l3584_358407


namespace NUMINAMATH_CALUDE_smallest_four_digit_mod_5_l3584_358471

theorem smallest_four_digit_mod_5 :
  ∀ n : ℕ, n ≥ 1000 ∧ n < 10000 ∧ n % 5 = 4 → n ≥ 1004 := by
  sorry

end NUMINAMATH_CALUDE_smallest_four_digit_mod_5_l3584_358471


namespace NUMINAMATH_CALUDE_coefficient_sum_l3584_358483

theorem coefficient_sum (b₀ b₁ b₂ b₃ b₄ b₅ b₆ : ℝ) :
  (∀ x : ℝ, (5*x - 2)^6 = b₆*x^6 + b₅*x^5 + b₄*x^4 + b₃*x^3 + b₂*x^2 + b₁*x + b₀) →
  b₆ + b₅ + b₄ + b₃ + b₂ + b₁ + b₀ = 729 := by
sorry

end NUMINAMATH_CALUDE_coefficient_sum_l3584_358483


namespace NUMINAMATH_CALUDE_min_sum_of_weights_l3584_358472

theorem min_sum_of_weights (S : ℕ) : 
  S > 280 ∧ S % 70 = 30 → S ≥ 310 :=
by sorry

end NUMINAMATH_CALUDE_min_sum_of_weights_l3584_358472


namespace NUMINAMATH_CALUDE_problem_solution_l3584_358443

theorem problem_solution (x y : ℝ) (h1 : x > 0) (h2 : y > 0) 
  (h3 : 6 * x^3 + 12 * x * y = 2 * x^4 + 3 * x^3 * y) (h4 : y = x^2) : 
  x = (-1 + Real.sqrt 55) / 3 := by
sorry

end NUMINAMATH_CALUDE_problem_solution_l3584_358443


namespace NUMINAMATH_CALUDE_complement_A_intersect_B_l3584_358414

-- Define the sets A and B
def A : Set ℝ := {x | 2^x ≤ 2 * Real.sqrt 2}
def B : Set ℝ := {x | Real.log (2 - x) < 0}

-- State the theorem
theorem complement_A_intersect_B :
  (Aᶜ ∩ B) = Set.Ioo (3/2) 2 := by sorry

end NUMINAMATH_CALUDE_complement_A_intersect_B_l3584_358414


namespace NUMINAMATH_CALUDE_wills_jogging_time_l3584_358481

/-- Calculates the jogging time given initial calories, burn rate, and final calories -/
def joggingTime (initialCalories : ℕ) (burnRate : ℕ) (finalCalories : ℕ) : ℕ :=
  (initialCalories - finalCalories) / burnRate

/-- Theorem stating that Will's jogging time is 30 minutes -/
theorem wills_jogging_time :
  let initialCalories : ℕ := 900
  let burnRate : ℕ := 10
  let finalCalories : ℕ := 600
  joggingTime initialCalories burnRate finalCalories = 30 := by
  sorry

end NUMINAMATH_CALUDE_wills_jogging_time_l3584_358481


namespace NUMINAMATH_CALUDE_storage_unit_blocks_l3584_358491

/-- Represents the dimensions of the storage unit --/
def storage_unit_side : ℕ := 8

/-- Represents the thickness of the walls, floor, and ceiling --/
def wall_thickness : ℕ := 1

/-- Calculates the number of blocks required for the storage unit construction --/
def blocks_required : ℕ :=
  storage_unit_side ^ 3 - (storage_unit_side - 2 * wall_thickness) ^ 3

/-- Theorem stating that 296 blocks are required for the storage unit construction --/
theorem storage_unit_blocks : blocks_required = 296 := by
  sorry

end NUMINAMATH_CALUDE_storage_unit_blocks_l3584_358491


namespace NUMINAMATH_CALUDE_power_of_eleven_l3584_358499

/-- Given an expression (11)^n * (4)^11 * (7)^5 where the total number of prime factors is 29,
    prove that the value of n (the power of 11) is 2. -/
theorem power_of_eleven (n : ℕ) : 
  (n + 22 + 5 = 29) → n = 2 := by
sorry

end NUMINAMATH_CALUDE_power_of_eleven_l3584_358499


namespace NUMINAMATH_CALUDE_functional_equation_solution_l3584_358441

/-- A function satisfying the given functional equation. -/
def SatisfiesEquation (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, f (x * f y - y * f x) = f (x * y) - x * y

/-- The main theorem stating that functions satisfying the equation are either the identity or absolute value function. -/
theorem functional_equation_solution (f : ℝ → ℝ) (h : SatisfiesEquation f) :
    (∀ x : ℝ, f x = x) ∨ (∀ x : ℝ, f x = |x|) := by
  sorry


end NUMINAMATH_CALUDE_functional_equation_solution_l3584_358441


namespace NUMINAMATH_CALUDE_fruit_brought_to_school_l3584_358433

/-- 
Given:
- Mark had an initial number of fruit pieces for the week
- Mark ate a certain number of fruit pieces in the first four days
- Mark decided to keep some pieces for next week

Prove that the number of fruit pieces Mark brought to school on Friday
is equal to the initial number minus the number eaten minus the number kept for next week
-/
theorem fruit_brought_to_school (initial_fruit pieces_eaten pieces_kept : ℕ) :
  initial_fruit - pieces_eaten - pieces_kept = initial_fruit - (pieces_eaten + pieces_kept) :=
by sorry

end NUMINAMATH_CALUDE_fruit_brought_to_school_l3584_358433


namespace NUMINAMATH_CALUDE_ellipse_cosine_theorem_l3584_358470

-- Define the ellipse
def ellipse (x y : ℝ) : Prop := x^2 / 4 + y^2 = 1

-- Define the foci
def foci (F₁ F₂ : ℝ × ℝ) : Prop :=
  let a := 2
  let c := Real.sqrt 3
  F₁ = (-c, 0) ∧ F₂ = (c, 0)

-- Define a point on the ellipse
def point_on_ellipse (P : ℝ × ℝ) : Prop :=
  let (x, y) := P
  ellipse x y

-- Define the distance ratio condition
def distance_ratio (P F₁ F₂ : ℝ × ℝ) : Prop :=
  let d₁ := Real.sqrt ((P.1 - F₁.1)^2 + (P.2 - F₁.2)^2)
  let d₂ := Real.sqrt ((P.2 - F₂.1)^2 + (P.2 - F₂.2)^2)
  d₁ = 3 * d₂

-- Theorem statement
theorem ellipse_cosine_theorem (P F₁ F₂ : ℝ × ℝ) :
  foci F₁ F₂ →
  point_on_ellipse P →
  distance_ratio P F₁ F₂ →
  let d₁ := Real.sqrt ((P.1 - F₁.1)^2 + (P.2 - F₁.2)^2)
  let d₂ := Real.sqrt ((P.2 - F₂.1)^2 + (P.2 - F₂.2)^2)
  let d₃ := Real.sqrt ((F₁.1 - F₂.1)^2 + (F₁.2 - F₂.2)^2)
  (d₁^2 + d₂^2 - d₃^2) / (2 * d₁ * d₂) = -1/3 :=
by sorry

end NUMINAMATH_CALUDE_ellipse_cosine_theorem_l3584_358470


namespace NUMINAMATH_CALUDE_max_value_sqrt_sum_l3584_358432

theorem max_value_sqrt_sum (x y z : ℝ) 
  (sum_eq_one : x + y + z = 1)
  (x_ge : x ≥ -1/3)
  (y_ge : y ≥ -1)
  (z_ge : z ≥ -5/3) :
  ∃ (max : ℝ), max = 6 ∧ 
    ∀ (a b c : ℝ), a + b + c = 1 → a ≥ -1/3 → b ≥ -1 → c ≥ -5/3 →
      Real.sqrt (3 * a + 1) + Real.sqrt (3 * b + 3) + Real.sqrt (3 * c + 5) ≤ max ∧
      Real.sqrt (3 * x + 1) + Real.sqrt (3 * y + 3) + Real.sqrt (3 * z + 5) = max :=
by
  sorry

end NUMINAMATH_CALUDE_max_value_sqrt_sum_l3584_358432


namespace NUMINAMATH_CALUDE_trig_equation_iff_equal_l3584_358446

theorem trig_equation_iff_equal (a b : Real) 
  (ha : 0 ≤ a ∧ a ≤ π/2) (hb : 0 ≤ b ∧ b ≤ π/2) : 
  (Real.sin a)^6 + 3*(Real.sin a)^2*(Real.cos b)^2 + (Real.cos b)^6 = 1 ↔ a = b := by
  sorry

end NUMINAMATH_CALUDE_trig_equation_iff_equal_l3584_358446


namespace NUMINAMATH_CALUDE_percentage_50_59_is_100_over_9_l3584_358494

/-- Represents the frequency distribution of test scores -/
structure ScoreDistribution :=
  (score_90_100 : ℕ)
  (score_80_89 : ℕ)
  (score_70_79 : ℕ)
  (score_60_69 : ℕ)
  (score_50_59 : ℕ)
  (score_below_50 : ℕ)

/-- The actual score distribution from Ms. Garcia's geometry class -/
def garcia_distribution : ScoreDistribution :=
  { score_90_100 := 5,
    score_80_89 := 7,
    score_70_79 := 9,
    score_60_69 := 8,
    score_50_59 := 4,
    score_below_50 := 3 }

/-- Calculate the total number of students -/
def total_students (d : ScoreDistribution) : ℕ :=
  d.score_90_100 + d.score_80_89 + d.score_70_79 + d.score_60_69 + d.score_50_59 + d.score_below_50

/-- Calculate the percentage of students in the 50%-59% range -/
def percentage_50_59 (d : ScoreDistribution) : ℚ :=
  (d.score_50_59 : ℚ) / (total_students d : ℚ) * 100

/-- Theorem stating that the percentage of students who scored in the 50%-59% range is 100/9 -/
theorem percentage_50_59_is_100_over_9 :
  percentage_50_59 garcia_distribution = 100 / 9 := by
  sorry


end NUMINAMATH_CALUDE_percentage_50_59_is_100_over_9_l3584_358494


namespace NUMINAMATH_CALUDE_gcd_204_85_f_at_2_l3584_358495

-- Part 1: GCD of 204 and 85
theorem gcd_204_85 : Nat.gcd 204 85 = 17 := by sorry

-- Part 2: Value of polynomial at x = 2
def f (x : ℝ) : ℝ := 2*x^4 + 3*x^3 + 5*x - 4

theorem f_at_2 : f 2 = 62 := by sorry

end NUMINAMATH_CALUDE_gcd_204_85_f_at_2_l3584_358495


namespace NUMINAMATH_CALUDE_x_over_y_value_l3584_358425

theorem x_over_y_value (x y z : ℝ) 
  (eq1 : x + y = 2 * x + z)
  (eq2 : x - 2 * y = 4 * z)
  (eq3 : x + y + z = 21)
  (eq4 : y / z = 6) :
  x / y = 1 / 3 := by
sorry

end NUMINAMATH_CALUDE_x_over_y_value_l3584_358425


namespace NUMINAMATH_CALUDE_area_between_curves_l3584_358442

-- Define the two functions
def f (y : ℝ) : ℝ := 4 - (y - 1)^2
def g (y : ℝ) : ℝ := y^2 - 4*y + 3

-- Define the bounds of integration
def lower_bound : ℝ := 0
def upper_bound : ℝ := 3

-- State the theorem
theorem area_between_curves : 
  (∫ y in lower_bound..upper_bound, f y - g y) = 9 := by sorry

end NUMINAMATH_CALUDE_area_between_curves_l3584_358442


namespace NUMINAMATH_CALUDE_garden_tulips_percentage_l3584_358410

theorem garden_tulips_percentage :
  ∀ (total_flowers : ℕ) (pink_flowers red_flowers pink_roses red_roses pink_tulips red_tulips lilies : ℕ),
    pink_flowers + red_flowers + lilies = total_flowers →
    pink_roses + pink_tulips = pink_flowers →
    red_roses + red_tulips = red_flowers →
    2 * pink_roses = pink_flowers →
    3 * red_tulips = 2 * red_flowers →
    4 * pink_flowers = 3 * total_flowers →
    10 * lilies = total_flowers →
    100 * (pink_tulips + red_tulips) = 61 * total_flowers :=
by
  sorry

end NUMINAMATH_CALUDE_garden_tulips_percentage_l3584_358410


namespace NUMINAMATH_CALUDE_child_attraction_fee_is_two_l3584_358456

/-- Represents the cost of various tickets and the family composition --/
structure ParkCosts where
  entrance_fee : ℕ
  adult_attraction_fee : ℕ
  child_attraction_fee : ℕ
  num_children : ℕ
  num_parents : ℕ
  num_grandparents : ℕ
  total_cost : ℕ

/-- Theorem stating that given the conditions, the child attraction fee is $2 --/
theorem child_attraction_fee_is_two (c : ParkCosts)
  (h1 : c.entrance_fee = 5)
  (h2 : c.adult_attraction_fee = 4)
  (h3 : c.num_children = 4)
  (h4 : c.num_parents = 2)
  (h5 : c.num_grandparents = 1)
  (h6 : c.total_cost = 55)
  (h7 : c.total_cost = (c.num_children + c.num_parents + c.num_grandparents) * c.entrance_fee +
                       (c.num_parents + c.num_grandparents) * c.adult_attraction_fee +
                       c.num_children * c.child_attraction_fee) :
  c.child_attraction_fee = 2 :=
by sorry

end NUMINAMATH_CALUDE_child_attraction_fee_is_two_l3584_358456


namespace NUMINAMATH_CALUDE_jujube_sales_theorem_l3584_358493

/-- Represents the daily sales deviation from the planned amount -/
def daily_deviations : List Int := [4, -3, -5, 14, -8, 21, -6]

/-- The planned daily sales amount in pounds -/
def planned_daily_sales : Nat := 100

/-- The selling price per pound in yuan -/
def selling_price : Nat := 8

/-- The freight cost per pound in yuan -/
def freight_cost : Nat := 3

theorem jujube_sales_theorem :
  /- Total amount sold in first three days -/
  (List.take 3 daily_deviations).sum + 3 * planned_daily_sales = 296 ∧
  /- Total earnings for the week -/
  (daily_deviations.sum + 7 * planned_daily_sales) * (selling_price - freight_cost) = 3585 := by
  sorry

end NUMINAMATH_CALUDE_jujube_sales_theorem_l3584_358493


namespace NUMINAMATH_CALUDE_solution_set_g_range_of_m_l3584_358409

-- Define the functions f and g
def f (x : ℝ) := x^2 - 2*x - 8
def g (x : ℝ) := 2*x^2 - 4*x - 16

-- Theorem for the solution set of g(x) < 0
theorem solution_set_g (x : ℝ) : g x < 0 ↔ -2 < x ∧ x < 4 := by sorry

-- Theorem for the range of m
theorem range_of_m (m : ℝ) :
  (∀ x > 2, f x ≥ (m + 2) * x - m - 15) → m ≤ 2 := by sorry

end NUMINAMATH_CALUDE_solution_set_g_range_of_m_l3584_358409


namespace NUMINAMATH_CALUDE_sum_of_numbers_l3584_358419

theorem sum_of_numbers (t a : ℝ) 
  (h1 : t = a + 12) 
  (h2 : t^2 + a^2 = 169/2) 
  (h3 : t^4 = a^4 + 5070) : 
  t + a = 5 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_numbers_l3584_358419


namespace NUMINAMATH_CALUDE_womens_doubles_handshakes_l3584_358468

/-- The number of handshakes in a women's doubles tennis tournament -/
theorem womens_doubles_handshakes (n : ℕ) (k : ℕ) (h1 : n = 4) (h2 : k = 2) : 
  let total_women := n * k
  let handshakes_per_woman := total_women - k
  (total_women * handshakes_per_woman) / 2 = 24 := by
sorry

end NUMINAMATH_CALUDE_womens_doubles_handshakes_l3584_358468


namespace NUMINAMATH_CALUDE_minimum_h_10_l3584_358489

def IsTenuous (h : ℕ+ → ℤ) : Prop :=
  ∀ x y : ℕ+, h x + h y > 2 * y.val ^ 2

def SumH (h : ℕ+ → ℤ) : ℤ :=
  (Finset.range 15).sum (fun i => h ⟨i + 1, Nat.succ_pos i⟩)

theorem minimum_h_10 (h : ℕ+ → ℤ) 
  (tenuous : IsTenuous h) 
  (min_sum : ∀ g : ℕ+ → ℤ, IsTenuous g → SumH g ≥ SumH h) : 
  h ⟨10, Nat.succ_pos 9⟩ ≥ 137 := by
  sorry

end NUMINAMATH_CALUDE_minimum_h_10_l3584_358489


namespace NUMINAMATH_CALUDE_smallest_value_of_expression_l3584_358413

theorem smallest_value_of_expression (a b c : ℝ) (h1 : a > b) (h2 : b > c) (h3 : a ≠ 0) :
  ((a + b)^2 + (b + c)^2 + (c + a)^2) / a^2 ≥ 6 ∧
  ∃ (a' b' c' : ℝ), a' > b' ∧ b' > c' ∧ a' ≠ 0 ∧
    ((a' + b')^2 + (b' + c')^2 + (c' + a')^2) / a'^2 = 6 :=
by sorry

end NUMINAMATH_CALUDE_smallest_value_of_expression_l3584_358413


namespace NUMINAMATH_CALUDE_soap_scrap_parts_l3584_358463

/-- The number of parts used to manufacture one soap -/
def soap_parts : ℕ := 11

/-- The total number of scraps at the end of the day -/
def total_scraps : ℕ := 251

/-- The number of additional soaps that can be manufactured from the scraps -/
def additional_soaps : ℕ := 25

/-- The number of scrap parts obtained for making one soap -/
def scrap_parts_per_soap : ℕ := 10

theorem soap_scrap_parts :
  scrap_parts_per_soap * additional_soaps = total_scraps ∧
  scrap_parts_per_soap < soap_parts :=
by sorry

end NUMINAMATH_CALUDE_soap_scrap_parts_l3584_358463


namespace NUMINAMATH_CALUDE_solve_linear_equation_l3584_358435

theorem solve_linear_equation (x : ℝ) : 3 * x - 5 = 4 * x + 10 → x = -15 := by
  sorry

end NUMINAMATH_CALUDE_solve_linear_equation_l3584_358435


namespace NUMINAMATH_CALUDE_parabola_focus_l3584_358455

/-- A parabola is defined by the equation x^2 = 4y -/
structure Parabola where
  eq : ∀ x y : ℝ, x^2 = 4*y

/-- The focus of a parabola is a point (h, k) where h and k are real numbers -/
structure Focus where
  h : ℝ
  k : ℝ

/-- Theorem: The focus of the parabola x^2 = 4y has coordinates (0, 1) -/
theorem parabola_focus (p : Parabola) : ∃ f : Focus, f.h = 0 ∧ f.k = 1 :=
sorry

end NUMINAMATH_CALUDE_parabola_focus_l3584_358455


namespace NUMINAMATH_CALUDE_sum_product_inequality_l3584_358465

theorem sum_product_inequality (a b c : ℝ) (h : a + b + c = 0) : a * b + b * c + c * a ≤ 0 := by
  sorry

end NUMINAMATH_CALUDE_sum_product_inequality_l3584_358465


namespace NUMINAMATH_CALUDE_eight_bead_bracelet_arrangements_l3584_358438

-- Define the number of beads
def n : ℕ := 8

-- Define the function to calculate the number of distinct arrangements
def bracelet_arrangements (m : ℕ) : ℕ :=
  (Nat.factorial m) / (m * 2)

-- Theorem statement
theorem eight_bead_bracelet_arrangements :
  bracelet_arrangements n = 2520 := by
  sorry

end NUMINAMATH_CALUDE_eight_bead_bracelet_arrangements_l3584_358438


namespace NUMINAMATH_CALUDE_perimeter_is_24_l3584_358459

/-- A right triangle ABC with specific properties -/
structure RightTriangleABC where
  -- Points A, B, and C
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ
  -- ABC is a right triangle with right angle at B
  is_right_triangle : (B.1 - A.1) * (C.1 - B.1) + (B.2 - A.2) * (C.2 - B.2) = 0
  -- Angle BAC equals angle BCA
  angle_equality : (A.1 - B.1) * (A.1 - C.1) + (A.2 - B.2) * (A.2 - C.2) = 
                   (C.1 - B.1) * (C.1 - A.1) + (C.2 - B.2) * (C.2 - A.2)
  -- Length of AB is 9
  AB_length : Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) = 9
  -- Length of BC is 6
  BC_length : Real.sqrt ((B.1 - C.1)^2 + (B.2 - C.2)^2) = 6

/-- The perimeter of the right triangle ABC is 24 -/
theorem perimeter_is_24 (t : RightTriangleABC) : 
  Real.sqrt ((t.A.1 - t.B.1)^2 + (t.A.2 - t.B.2)^2) +
  Real.sqrt ((t.B.1 - t.C.1)^2 + (t.B.2 - t.C.2)^2) +
  Real.sqrt ((t.C.1 - t.A.1)^2 + (t.C.2 - t.A.2)^2) = 24 := by
  sorry


end NUMINAMATH_CALUDE_perimeter_is_24_l3584_358459


namespace NUMINAMATH_CALUDE_four_distinct_roots_iff_q_16_l3584_358476

/-- The function f(x) = x^2 + 8x + q -/
def f (q : ℝ) (x : ℝ) : ℝ := x^2 + 8*x + q

/-- The composition of f with itself -/
def f_comp (q : ℝ) (x : ℝ) : ℝ := f q (f q x)

/-- The number of distinct real roots of f(f(x)) -/
noncomputable def num_distinct_roots (q : ℝ) : ℕ := sorry

theorem four_distinct_roots_iff_q_16 :
  ∀ q : ℝ, num_distinct_roots q = 4 ↔ q = 16 :=
sorry

end NUMINAMATH_CALUDE_four_distinct_roots_iff_q_16_l3584_358476


namespace NUMINAMATH_CALUDE_intersection_of_M_and_N_l3584_358474

def M : Set ℕ := {1, 2, 3}
def N : Set ℕ := {0, 1, 2}

theorem intersection_of_M_and_N : M ∩ N = {1, 2} := by
  sorry

end NUMINAMATH_CALUDE_intersection_of_M_and_N_l3584_358474


namespace NUMINAMATH_CALUDE_min_value_trig_expression_l3584_358440

theorem min_value_trig_expression (x : ℝ) :
  (Real.sin x)^8 + (Real.cos x)^8 + 2 ≥ 5/4 * ((Real.sin x)^6 + (Real.cos x)^6 + 2) := by
  sorry

end NUMINAMATH_CALUDE_min_value_trig_expression_l3584_358440


namespace NUMINAMATH_CALUDE_modulus_of_z_l3584_358453

theorem modulus_of_z (a : ℝ) (i : ℂ) (h1 : i * i = -1) 
  (h2 : ∃ (b : ℝ), (2 - i) / (a + i) = b * i) :
  Complex.abs (2 * a + Complex.I * Real.sqrt 3) = 2 := by
  sorry

end NUMINAMATH_CALUDE_modulus_of_z_l3584_358453


namespace NUMINAMATH_CALUDE_cube_root_equal_self_l3584_358426

theorem cube_root_equal_self (a : ℝ) : a^(1/3) = a → a = 1 ∨ a = 0 ∨ a = -1 := by
  sorry

end NUMINAMATH_CALUDE_cube_root_equal_self_l3584_358426


namespace NUMINAMATH_CALUDE_intersection_equals_C_l3584_358464

-- Define the set of angles less than 90°
def A : Set ℝ := {α | α < 90}

-- Define the set of angles in the first quadrant
def B : Set ℝ := {α | ∃ k : ℤ, k * 360 < α ∧ α < k * 360 + 90}

-- Define the set of angles α such that k · 360° < α < k · 360° + 90° for some integer k ≤ 0
def C : Set ℝ := {α | ∃ k : ℤ, k ≤ 0 ∧ k * 360 < α ∧ α < k * 360 + 90}

-- Theorem statement
theorem intersection_equals_C : A ∩ B = C := by sorry

end NUMINAMATH_CALUDE_intersection_equals_C_l3584_358464


namespace NUMINAMATH_CALUDE_library_visitors_l3584_358466

/-- Proves that the average number of visitors on Sundays is 540 given the specified conditions --/
theorem library_visitors (total_days : Nat) (non_sunday_visitors : Nat) (avg_visitors : Nat) :
  total_days = 30 ∧
  non_sunday_visitors = 240 ∧
  avg_visitors = 290 →
  (5 * (((avg_visitors * total_days) - (25 * non_sunday_visitors)) / 5) + 25 * non_sunday_visitors) / total_days = avg_visitors :=
by sorry

end NUMINAMATH_CALUDE_library_visitors_l3584_358466


namespace NUMINAMATH_CALUDE_problem_statement_l3584_358454

theorem problem_statement (a b c : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0)
  (sum_zero : a + b + c = 0) (sum_prod_nonzero : a * b + a * c + b * c ≠ 0) :
  (a^7 + b^7 + c^7) / (a * b * c * (a * b + a * c + b * c)) = -7 := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l3584_358454


namespace NUMINAMATH_CALUDE_largest_square_area_l3584_358423

-- Define the triangle and its properties
structure RightTriangle where
  XY : ℝ
  YZ : ℝ
  XZ : ℝ
  right_angle : XZ^2 = XY^2 + YZ^2
  hypotenuse_relation : XZ^2 = 2 * XY^2

-- Define the theorem
theorem largest_square_area
  (triangle : RightTriangle)
  (total_area : ℝ)
  (h_total_area : XY^2 + YZ^2 + XZ^2 = total_area)
  (h_total_area_value : total_area = 450) :
  XZ^2 = 225 := by
  sorry

#check largest_square_area

end NUMINAMATH_CALUDE_largest_square_area_l3584_358423


namespace NUMINAMATH_CALUDE_wire_cutting_l3584_358439

theorem wire_cutting (total_length : ℝ) (cut_fraction : ℝ) (remaining_length : ℝ) : 
  total_length = 3 → 
  cut_fraction = 1/3 → 
  remaining_length = total_length * (1 - cut_fraction) → 
  remaining_length = 2 := by
sorry

end NUMINAMATH_CALUDE_wire_cutting_l3584_358439


namespace NUMINAMATH_CALUDE_student_weight_difference_l3584_358429

/-- Proves the difference in average weights given specific conditions about a group of students --/
theorem student_weight_difference (n : ℕ) (initial_avg : ℝ) (joe_weight : ℝ) (new_avg : ℝ) :
  initial_avg = 30 →
  joe_weight = 42 →
  new_avg = 31 →
  (n * initial_avg + joe_weight) / (n + 1) = new_avg →
  ((n + 1) * new_avg - 2 * initial_avg) / (n - 1) = initial_avg →
  abs (((n + 1) * new_avg - n * initial_avg) / 2 - joe_weight) = 6 := by
  sorry

end NUMINAMATH_CALUDE_student_weight_difference_l3584_358429


namespace NUMINAMATH_CALUDE_quadratic_function_nonnegative_constraint_l3584_358418

theorem quadratic_function_nonnegative_constraint (a : ℝ) : 
  (∀ x : ℝ, x ∈ Set.Icc (-2) 2 → x^2 + a*x + 3 - a ≥ 0) → 
  a ∈ Set.Icc (-7) 2 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_function_nonnegative_constraint_l3584_358418


namespace NUMINAMATH_CALUDE_sum_of_reciprocal_roots_l3584_358405

theorem sum_of_reciprocal_roots (x₁ x₂ : ℝ) : 
  x₁^2 + 2*x₁ - 3 = 0 → x₂^2 + 2*x₂ - 3 = 0 → x₁ ≠ x₂ → 
  (1/x₁ + 1/x₂ : ℝ) = 2/3 := by sorry

end NUMINAMATH_CALUDE_sum_of_reciprocal_roots_l3584_358405


namespace NUMINAMATH_CALUDE_min_triangle_area_l3584_358488

open Complex

-- Define the equation solutions
def solutions : Set ℂ := {z : ℂ | (z - 3)^10 = 32}

-- Define the property that solutions form a regular decagon
def is_regular_decagon (s : Set ℂ) : Prop := sorry

-- Define the function to calculate the area of a triangle given three complex points
def triangle_area (a b c : ℂ) : ℝ := sorry

-- Theorem statement
theorem min_triangle_area (h : is_regular_decagon solutions) :
  ∃ (a b c : ℂ), a ∈ solutions ∧ b ∈ solutions ∧ c ∈ solutions ∧
  (∀ (x y z : ℂ), x ∈ solutions → y ∈ solutions → z ∈ solutions →
    triangle_area a b c ≤ triangle_area x y z) ∧
  triangle_area a b c = 2 * Real.sin (18 * π / 180) * Real.sin (36 * π / 180) :=
sorry

end NUMINAMATH_CALUDE_min_triangle_area_l3584_358488


namespace NUMINAMATH_CALUDE_ams_sequence_results_in_14_l3584_358484

/-- Milly's operation: multiply by 3 -/
def milly (x : ℤ) : ℤ := 3 * x

/-- Abby's operation: add 2 -/
def abby (x : ℤ) : ℤ := x + 2

/-- Sam's operation: subtract 1 -/
def sam (x : ℤ) : ℤ := x - 1

/-- The theorem stating that applying Abby's, Milly's, and Sam's operations in order to 3 results in 14 -/
theorem ams_sequence_results_in_14 : sam (milly (abby 3)) = 14 := by
  sorry

end NUMINAMATH_CALUDE_ams_sequence_results_in_14_l3584_358484


namespace NUMINAMATH_CALUDE_remainder_theorem_l3584_358412

theorem remainder_theorem (n : ℤ) (k : ℤ) (h : n = 7 * k - 1) :
  (n^2 + 3*n + 4) % 7 = 2 := by sorry

end NUMINAMATH_CALUDE_remainder_theorem_l3584_358412


namespace NUMINAMATH_CALUDE_coefficient_x_squared_in_expansion_l3584_358485

theorem coefficient_x_squared_in_expansion :
  let expansion := (fun x => (x - 2/x)^8)
  let coefficient_x_squared (f : ℝ → ℝ) := 
    (1/2) * (deriv (deriv f) 0)
  coefficient_x_squared expansion = -Nat.choose 8 3 * 2^3 := by
  sorry

end NUMINAMATH_CALUDE_coefficient_x_squared_in_expansion_l3584_358485


namespace NUMINAMATH_CALUDE_vector_decomposition_l3584_358482

/-- Given vectors in ℝ³ -/
def x : Fin 3 → ℝ := ![6, 5, -14]
def p : Fin 3 → ℝ := ![1, 1, 4]
def q : Fin 3 → ℝ := ![0, -3, 2]
def r : Fin 3 → ℝ := ![2, 1, -1]

/-- Theorem: x can be expressed as a linear combination of p, q, and r -/
theorem vector_decomposition :
  x = (-2 : ℝ) • p + (-1 : ℝ) • q + (4 : ℝ) • r := by
  sorry

end NUMINAMATH_CALUDE_vector_decomposition_l3584_358482


namespace NUMINAMATH_CALUDE_conditional_probability_B_given_A_l3584_358457

-- Define the sample space
def sample_space : ℕ := 36

-- Define event A
def event_A : ℕ := 6

-- Define event B
def event_B : ℕ := 5

-- Define the intersection of events A and B
def event_AB : ℕ := 1

-- Theorem statement
theorem conditional_probability_B_given_A :
  (event_AB : ℚ) / event_A = 1 / 6 :=
sorry

end NUMINAMATH_CALUDE_conditional_probability_B_given_A_l3584_358457


namespace NUMINAMATH_CALUDE_q_is_correct_l3584_358451

-- Define the polynomial q(x)
def q (x : ℝ) : ℝ := -2*x^5 + 5*x^4 + 27*x^3 + 50*x^2 + 63*x + 5

-- Theorem statement
theorem q_is_correct :
  ∀ x : ℝ, q x + (2*x^5 + 5*x^4 + 8*x^3 + 9*x) = 10*x^4 + 35*x^3 + 50*x^2 + 72*x + 5 :=
by
  sorry

end NUMINAMATH_CALUDE_q_is_correct_l3584_358451


namespace NUMINAMATH_CALUDE_gem_selection_count_l3584_358458

/-- The number of ways to select gems under given constraints -/
def select_gems (red blue green : ℕ) (total_select : ℕ) (min_red min_blue max_green : ℕ) : ℕ :=
  (Finset.range (min green max_green + 1)).sum (λ g =>
    (Finset.range (red + 1)).sum (λ r =>
      if r ≥ min_red ∧ total_select - r - g ≥ min_blue ∧ total_select - r - g ≤ blue
      then 1
      else 0))

/-- Theorem stating the number of ways to select gems under given constraints -/
theorem gem_selection_count : select_gems 9 5 6 10 2 2 3 = 22 := by
  sorry

end NUMINAMATH_CALUDE_gem_selection_count_l3584_358458


namespace NUMINAMATH_CALUDE_typing_task_correct_characters_l3584_358492

/-- The total number of characters in the typing task -/
def total_characters : ℕ := 10000

/-- Xiaoyuan's error rate: 1 mistake per 10 characters -/
def xiaoyuan_error_rate : ℚ := 1 / 10

/-- Xiaofang's error rate: 2 mistakes per 10 characters -/
def xiaofang_error_rate : ℚ := 2 / 10

/-- The ratio of correct characters typed by Xiaoyuan to Xiaofang -/
def correct_ratio : ℕ := 2

theorem typing_task_correct_characters :
  ∃ (xiaoyuan_correct xiaofang_correct : ℕ),
    xiaoyuan_correct + xiaofang_correct = 8640 ∧
    xiaoyuan_correct = 2 * xiaofang_correct ∧
    xiaoyuan_correct = total_characters * (1 - xiaoyuan_error_rate) ∧
    xiaofang_correct = total_characters * (1 - xiaofang_error_rate) :=
sorry

end NUMINAMATH_CALUDE_typing_task_correct_characters_l3584_358492


namespace NUMINAMATH_CALUDE_methane_hydrate_density_scientific_notation_l3584_358428

theorem methane_hydrate_density_scientific_notation :
  0.00092 = 9.2 * 10^(-4) := by sorry

end NUMINAMATH_CALUDE_methane_hydrate_density_scientific_notation_l3584_358428


namespace NUMINAMATH_CALUDE_reciprocal_of_sum_divided_l3584_358450

theorem reciprocal_of_sum_divided : 
  (((1 : ℚ) / 4 + (1 : ℚ) / 5) / ((1 : ℚ) / 3))⁻¹ = 20 / 27 := by
  sorry

end NUMINAMATH_CALUDE_reciprocal_of_sum_divided_l3584_358450


namespace NUMINAMATH_CALUDE_unique_corresponding_point_l3584_358460

-- Define a square as a structure with a side length and a position
structure Square where
  sideLength : ℝ
  position : ℝ × ℝ

-- Define the problem setup
axiom larger_square : Square
axiom smaller_square : Square

-- The smaller square is entirely within the larger square
axiom smaller_inside_larger :
  smaller_square.position.1 ≥ larger_square.position.1 ∧
  smaller_square.position.1 + smaller_square.sideLength ≤ larger_square.position.1 + larger_square.sideLength ∧
  smaller_square.position.2 ≥ larger_square.position.2 ∧
  smaller_square.position.2 + smaller_square.sideLength ≤ larger_square.position.2 + larger_square.sideLength

-- The squares have the same area
axiom same_area : larger_square.sideLength^2 = smaller_square.sideLength^2

-- Define a point as a pair of real numbers
def Point := ℝ × ℝ

-- The theorem to be proved
theorem unique_corresponding_point :
  ∃! p : Point,
    (p.1 - larger_square.position.1) / larger_square.sideLength =
    (p.1 - smaller_square.position.1) / smaller_square.sideLength ∧
    (p.2 - larger_square.position.2) / larger_square.sideLength =
    (p.2 - smaller_square.position.2) / smaller_square.sideLength :=
  sorry

end NUMINAMATH_CALUDE_unique_corresponding_point_l3584_358460


namespace NUMINAMATH_CALUDE_sum_remainder_five_l3584_358404

theorem sum_remainder_five (n : ℤ) : ((5 - n) + (n + 4)) % 5 = 4 := by
  sorry

end NUMINAMATH_CALUDE_sum_remainder_five_l3584_358404


namespace NUMINAMATH_CALUDE_farm_has_two_fields_l3584_358427

/-- Represents a corn field -/
structure CornField where
  rows : ℕ
  cobs_per_row : ℕ

/-- Calculates the total number of corn cobs in a field -/
def total_cobs (field : CornField) : ℕ :=
  field.rows * field.cobs_per_row

/-- Represents the farm's corn production -/
structure FarmProduction where
  field1 : CornField
  field2 : CornField
  total_cobs : ℕ

/-- Theorem: The farm is growing corn in 2 fields -/
theorem farm_has_two_fields (farm : FarmProduction) : 
  farm.field1.rows = 13 ∧ 
  farm.field2.rows = 16 ∧ 
  farm.field1.cobs_per_row = 4 ∧ 
  farm.field2.cobs_per_row = 4 ∧ 
  farm.total_cobs = 116 → 
  2 = (if total_cobs farm.field1 + total_cobs farm.field2 = farm.total_cobs then 2 else 1) :=
by sorry


end NUMINAMATH_CALUDE_farm_has_two_fields_l3584_358427


namespace NUMINAMATH_CALUDE_consecutive_integers_sqrt_3_l3584_358478

theorem consecutive_integers_sqrt_3 (a b : ℤ) : 
  (b = a + 1) → (↑a < Real.sqrt 3 ∧ Real.sqrt 3 < ↑b) → a + b = 3 := by
  sorry

end NUMINAMATH_CALUDE_consecutive_integers_sqrt_3_l3584_358478


namespace NUMINAMATH_CALUDE_positive_integer_expression_l3584_358424

def factorial (n : ℕ) : ℕ := if n = 0 then 1 else n * factorial (n - 1)

theorem positive_integer_expression (m n : ℕ) :
  ∃ k : ℕ+, k = (factorial (2 * m) * factorial (2 * n)) / (factorial m * factorial n * factorial (m + n)) :=
by sorry

end NUMINAMATH_CALUDE_positive_integer_expression_l3584_358424


namespace NUMINAMATH_CALUDE_daughter_age_in_three_years_l3584_358402

/-- Given a mother's current age and the fact that she was twice her daughter's age 5 years ago,
    this function calculates the daughter's age in 3 years. -/
def daughters_future_age (mothers_current_age : ℕ) : ℕ :=
  let mothers_past_age := mothers_current_age - 5
  let daughters_past_age := mothers_past_age / 2
  let daughters_current_age := daughters_past_age + 5
  daughters_current_age + 3

/-- Theorem stating that given the problem conditions, the daughter will be 26 years old in 3 years. -/
theorem daughter_age_in_three_years :
  daughters_future_age 41 = 26 := by
  sorry

end NUMINAMATH_CALUDE_daughter_age_in_three_years_l3584_358402


namespace NUMINAMATH_CALUDE_circle_points_theorem_l3584_358437

/-- The number of points on the circumference of the circle -/
def n : ℕ := 9

/-- Represents the property that no three points are collinear along any line passing through the circle's center -/
def no_three_collinear (points : Fin n → ℝ × ℝ) : Prop := sorry

/-- The number of different triangles that can be formed -/
def num_triangles : ℕ := Nat.choose n 3

/-- The number of distinct straight lines that can be drawn -/
def num_lines : ℕ := Nat.choose n 2

/-- Main theorem stating the number of triangles and lines -/
theorem circle_points_theorem (points : Fin n → ℝ × ℝ) 
  (h : no_three_collinear points) : 
  num_triangles = 84 ∧ num_lines = 36 := by sorry

end NUMINAMATH_CALUDE_circle_points_theorem_l3584_358437


namespace NUMINAMATH_CALUDE_min_value_exponential_sum_l3584_358479

theorem min_value_exponential_sum (x : ℝ) : 16^x + 4^x - 2^x + 1 ≥ (3/4 : ℝ) := by
  sorry

end NUMINAMATH_CALUDE_min_value_exponential_sum_l3584_358479


namespace NUMINAMATH_CALUDE_playground_slide_total_l3584_358462

theorem playground_slide_total (boys_first_10min : ℕ) (boys_next_5min : ℕ) (boys_last_20min : ℕ)
  (h1 : boys_first_10min = 22)
  (h2 : boys_next_5min = 13)
  (h3 : boys_last_20min = 35) :
  boys_first_10min + boys_next_5min + boys_last_20min = 70 :=
by sorry

end NUMINAMATH_CALUDE_playground_slide_total_l3584_358462


namespace NUMINAMATH_CALUDE_inequality_not_preserved_after_subtraction_of_squares_l3584_358436

theorem inequality_not_preserved_after_subtraction_of_squares : 
  ∃ (a b : ℝ), a > 0 ∧ b > 0 ∧ a > b ∧ (a - a^2) ≤ (b - b^2) := by
  sorry

end NUMINAMATH_CALUDE_inequality_not_preserved_after_subtraction_of_squares_l3584_358436


namespace NUMINAMATH_CALUDE_ann_age_l3584_358445

/-- The complex age relationship between Ann and Barbara --/
def age_relationship (a b : ℕ) : Prop :=
  ∃ y : ℕ, b = a / 2 + 2 * y ∧ y = a - b

/-- The theorem stating Ann's age given the conditions --/
theorem ann_age :
  ∀ a b : ℕ,
  age_relationship a b →
  a + b = 54 →
  a = 29 :=
by
  sorry

end NUMINAMATH_CALUDE_ann_age_l3584_358445


namespace NUMINAMATH_CALUDE_complex_equation_solution_l3584_358415

theorem complex_equation_solution (a b c x y z : ℂ) 
  (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) (hx : x ≠ 0) (hy : y ≠ 0) (hz : z ≠ 0)
  (eq1 : a = (b + c) / (x - 1))
  (eq2 : b = (a + c) / (y - 1))
  (eq3 : c = (a + b) / (z - 1))
  (eq4 : x * y + x * z + y * z = 7)
  (eq5 : x + y + z = 3) :
  x * y * z = 9 := by
  sorry


end NUMINAMATH_CALUDE_complex_equation_solution_l3584_358415


namespace NUMINAMATH_CALUDE_max_value_constraint_l3584_358469

theorem max_value_constraint (x y : ℝ) (h : 5 * x^2 + 4 * y^2 = 10 * x) :
  x^2 + y^2 ≤ 4 ∧ ∃ (x₀ y₀ : ℝ), 5 * x₀^2 + 4 * y₀^2 = 10 * x₀ ∧ x₀^2 + y₀^2 = 4 :=
by sorry

end NUMINAMATH_CALUDE_max_value_constraint_l3584_358469


namespace NUMINAMATH_CALUDE_chef_cakes_problem_l3584_358487

def chef_cakes (total_eggs : ℕ) (fridge_eggs : ℕ) (eggs_per_cake : ℕ) : ℕ :=
  (total_eggs - fridge_eggs) / eggs_per_cake

theorem chef_cakes_problem :
  chef_cakes 60 10 5 = 10 := by
  sorry

end NUMINAMATH_CALUDE_chef_cakes_problem_l3584_358487


namespace NUMINAMATH_CALUDE_special_polynomial_characterization_l3584_358421

/-- A polynomial that satisfies the given functional equation -/
structure SpecialPolynomial where
  P : Polynomial ℝ
  eq : ∀ (X : ℝ), 16 * (P.eval (X^2)) = (P.eval (2*X))^2

/-- The characterization of polynomials satisfying the functional equation -/
theorem special_polynomial_characterization (sp : SpecialPolynomial) :
  ∃ (n : ℕ), sp.P = Polynomial.monomial n (16 * (1/4)^n) := by
  sorry

end NUMINAMATH_CALUDE_special_polynomial_characterization_l3584_358421


namespace NUMINAMATH_CALUDE_salary_increase_comparison_l3584_358403

theorem salary_increase_comparison (initial_salary : ℝ) (h : initial_salary > 0) :
  let first_worker_new_salary := 2 * initial_salary
  let second_worker_new_salary := 1.5 * initial_salary
  (first_worker_new_salary - second_worker_new_salary) / second_worker_new_salary = 1 / 3 := by
sorry

end NUMINAMATH_CALUDE_salary_increase_comparison_l3584_358403


namespace NUMINAMATH_CALUDE_irreducible_fractions_divisibility_l3584_358448

theorem irreducible_fractions_divisibility (a n : ℕ) (ha : a > 1) (hn : n > 1) :
  ∃ k : ℕ, Nat.totient (a^n - 1) = n * k := by
  sorry

end NUMINAMATH_CALUDE_irreducible_fractions_divisibility_l3584_358448


namespace NUMINAMATH_CALUDE_age_sum_proof_l3584_358406

theorem age_sum_proof (child_age mother_age : ℕ) : 
  child_age = 10 →
  mother_age = 3 * child_age →
  child_age + mother_age = 40 := by
sorry

end NUMINAMATH_CALUDE_age_sum_proof_l3584_358406


namespace NUMINAMATH_CALUDE_exactly_one_line_through_6_5_l3584_358447

/-- Represents a line in the xy-plane with given x and y intercepts -/
structure Line where
  x_intercept : ℝ
  y_intercept : ℝ

/-- Checks if a real number is a positive even integer -/
def is_positive_even (n : ℝ) : Prop :=
  n > 0 ∧ ∃ k : ℤ, n = 2 * k

/-- Checks if a real number is a positive odd integer -/
def is_positive_odd (n : ℝ) : Prop :=
  n > 0 ∧ ∃ k : ℤ, n = 2 * k + 1

/-- Checks if a line passes through the point (6,5) -/
def passes_through_6_5 (l : Line) : Prop :=
  6 / l.x_intercept + 5 / l.y_intercept = 1

/-- The main theorem to be proved -/
theorem exactly_one_line_through_6_5 :
  ∃! l : Line,
    is_positive_even l.x_intercept ∧
    is_positive_odd l.y_intercept ∧
    passes_through_6_5 l :=
  sorry

end NUMINAMATH_CALUDE_exactly_one_line_through_6_5_l3584_358447


namespace NUMINAMATH_CALUDE_pigeonhole_trees_leaves_l3584_358401

theorem pigeonhole_trees_leaves (n : ℕ) (L : ℕ → ℕ) 
  (h1 : ∀ i, i < n → L i < n) : 
  ∃ i j, i < n ∧ j < n ∧ i ≠ j ∧ L i = L j :=
by sorry

end NUMINAMATH_CALUDE_pigeonhole_trees_leaves_l3584_358401


namespace NUMINAMATH_CALUDE_existence_of_special_number_l3584_358461

theorem existence_of_special_number : ∃ A : ℕ, 
  (100000 ≤ A ∧ A < 1000000) ∧ 
  ∀ k : ℕ, 1 ≤ k ∧ k ≤ 500000 → 
    ¬(∃ d : ℕ, d < 10 ∧ (k * A) % 1000000 = d * 111111) :=
by sorry

end NUMINAMATH_CALUDE_existence_of_special_number_l3584_358461


namespace NUMINAMATH_CALUDE_min_a_squared_over_area_l3584_358400

theorem min_a_squared_over_area (a b c : ℝ) (A B C : ℝ) (S : ℝ) :
  a > 0 → b > 0 → c > 0 →
  A > 0 → B > 0 → C > 0 →
  A + B + C = π →
  b * Real.cos C + c * Real.cos B = 3 * a * Real.cos A →
  S = (1 / 2) * b * c * Real.sin A →
  a^2 / S ≥ 2 * Real.sqrt 2 := by
  sorry

#check min_a_squared_over_area

end NUMINAMATH_CALUDE_min_a_squared_over_area_l3584_358400


namespace NUMINAMATH_CALUDE_derivative_of_f_l3584_358477

noncomputable def f (x : ℝ) : ℝ := x^2 * Real.exp (2*x)

theorem derivative_of_f :
  deriv f = λ x => Real.exp (2*x) * (2*x + 2*x^2) := by sorry

end NUMINAMATH_CALUDE_derivative_of_f_l3584_358477


namespace NUMINAMATH_CALUDE_selling_price_is_twenty_l3584_358408

/-- Calculates the selling price per phone given the total number of phones, 
    total cost, and desired profit ratio. -/
def selling_price_per_phone (total_phones : ℕ) (total_cost : ℚ) (profit_ratio : ℚ) : ℚ :=
  let cost_per_phone := total_cost / total_phones
  let profit_per_phone := (total_cost * profit_ratio) / total_phones
  cost_per_phone + profit_per_phone

/-- Theorem stating that the selling price per phone is $20 given the problem conditions. -/
theorem selling_price_is_twenty :
  selling_price_per_phone 200 3000 (1/3) = 20 := by
  sorry

end NUMINAMATH_CALUDE_selling_price_is_twenty_l3584_358408


namespace NUMINAMATH_CALUDE_inverse_proportion_point_order_l3584_358498

/-- Prove that for points on an inverse proportion function, their y-coordinates follow a specific order -/
theorem inverse_proportion_point_order (k : ℝ) (y₁ y₂ y₃ : ℝ) 
  (h_pos : k > 0)
  (h_A : y₁ = k / (-1))
  (h_B : y₂ = k / 2)
  (h_C : y₃ = k / 3) :
  y₁ < y₃ ∧ y₃ < y₂ := by
  sorry

end NUMINAMATH_CALUDE_inverse_proportion_point_order_l3584_358498


namespace NUMINAMATH_CALUDE_max_value_when_a_zero_range_of_a_for_nonpositive_f_l3584_358420

/-- The function f(x) defined in the problem -/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (2 * x) / Real.exp x + a * Real.log (x + 1)

/-- Theorem for the maximum value of f when a = 0 -/
theorem max_value_when_a_zero :
  ∃ (x : ℝ), ∀ (y : ℝ), f 0 y ≤ f 0 x ∧ f 0 x = 2 / Real.exp 1 :=
sorry

/-- Theorem for the range of a when f(x) ≤ 0 for x ∈ [0, +∞) -/
theorem range_of_a_for_nonpositive_f :
  ∀ (a : ℝ), (∀ (x : ℝ), x ≥ 0 → f a x ≤ 0) ↔ a ≤ -2 :=
sorry

end NUMINAMATH_CALUDE_max_value_when_a_zero_range_of_a_for_nonpositive_f_l3584_358420


namespace NUMINAMATH_CALUDE_sum_of_ages_l3584_358417

theorem sum_of_ages (age1 age2 age3 : ℕ) : 
  age1 = 9 → age2 = 9 → age3 = 11 → age1 + age2 + age3 = 29 :=
by
  sorry

end NUMINAMATH_CALUDE_sum_of_ages_l3584_358417


namespace NUMINAMATH_CALUDE_smallest_value_of_a_l3584_358475

def a (n : ℕ+) : ℤ := 2 * n.val ^ 2 - 10 * n.val + 3

theorem smallest_value_of_a (n : ℕ+) :
  a n ≥ a 2 ∧ a n ≥ a 3 ∧ (a 2 = a 3) :=
sorry

end NUMINAMATH_CALUDE_smallest_value_of_a_l3584_358475


namespace NUMINAMATH_CALUDE_largest_prime_factors_difference_l3584_358490

theorem largest_prime_factors_difference (n : Nat) (h : n = 195195) :
  ∃ (p q : Nat), 
    Nat.Prime p ∧ 
    Nat.Prime q ∧ 
    p > q ∧
    p ∣ n ∧
    q ∣ n ∧
    (∀ r : Nat, Nat.Prime r → r ∣ n → r ≤ p) ∧
    (∀ r : Nat, Nat.Prime r → r ∣ n → r ≠ p → r ≤ q) ∧
    p - q = 2 := by
  sorry

end NUMINAMATH_CALUDE_largest_prime_factors_difference_l3584_358490


namespace NUMINAMATH_CALUDE_geometric_sequence_sum_l3584_358411

def geometric_sequence (a : ℕ → ℝ) (q : ℝ) : Prop :=
  ∀ n : ℕ, a (n + 1) = a n * q

theorem geometric_sequence_sum (a : ℕ → ℝ) (q : ℝ) :
  geometric_sequence a q → a 1 = 1 → q = -2 →
  a 1 + |a 2| + a 3 + |a 4| = 15 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_sum_l3584_358411


namespace NUMINAMATH_CALUDE_point_on_transformed_plane_l3584_358467

/-- Represents a point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Represents a plane in 3D space -/
structure Plane where
  a : ℝ
  b : ℝ
  c : ℝ
  d : ℝ

/-- Applies a similarity transformation to a plane -/
def transformPlane (p : Plane) (k : ℝ) : Plane :=
  { a := p.a, b := p.b, c := p.c, d := k * p.d }

/-- Checks if a point lies on a plane -/
def pointOnPlane (point : Point3D) (plane : Plane) : Prop :=
  plane.a * point.x + plane.b * point.y + plane.c * point.z + plane.d = 0

theorem point_on_transformed_plane (A : Point3D) (a : Plane) (k : ℝ) :
  A.x = 2 ∧ A.y = 5 ∧ A.z = 1 ∧
  a.a = 5 ∧ a.b = -2 ∧ a.c = 1 ∧ a.d = -3 ∧
  k = 1/3 →
  pointOnPlane A (transformPlane a k) :=
by sorry

end NUMINAMATH_CALUDE_point_on_transformed_plane_l3584_358467


namespace NUMINAMATH_CALUDE_norma_laundry_ratio_l3584_358431

/-- Proves the ratio of sweaters to T-shirts Norma left in the washer -/
theorem norma_laundry_ratio : 
  ∀ (S : ℕ), -- S is the number of sweaters Norma left
  -- Given conditions:
  (9 : ℕ) + S = 3 + 3 * 9 + 15 → -- Total items left = Total items found + Missing items
  (S : ℚ) / 9 = 2 / 1 := by
    sorry

end NUMINAMATH_CALUDE_norma_laundry_ratio_l3584_358431


namespace NUMINAMATH_CALUDE_average_daily_income_l3584_358434

def earnings : List ℝ := [620, 850, 760, 950, 680, 890, 720, 900, 780, 830, 800, 880]

theorem average_daily_income :
  (earnings.sum / earnings.length : ℝ) = 805 := by
  sorry

end NUMINAMATH_CALUDE_average_daily_income_l3584_358434
