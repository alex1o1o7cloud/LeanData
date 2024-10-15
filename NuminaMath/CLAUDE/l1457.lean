import Mathlib

namespace NUMINAMATH_CALUDE_inequality_holds_iff_l1457_145726

theorem inequality_holds_iff (m : ℝ) :
  (∀ x : ℝ, (x^2 - m*x - 2) / (x^2 - 3*x + 4) > -1) ↔ -7 < m ∧ m < 1 := by
  sorry

end NUMINAMATH_CALUDE_inequality_holds_iff_l1457_145726


namespace NUMINAMATH_CALUDE_groomer_problem_l1457_145708

/-- The number of full-haired dogs a groomer has to dry --/
def num_full_haired_dogs : ℕ := by sorry

theorem groomer_problem :
  let time_short_haired : ℕ := 10  -- minutes to dry a short-haired dog
  let time_full_haired : ℕ := 2 * time_short_haired  -- minutes to dry a full-haired dog
  let num_short_haired : ℕ := 6  -- number of short-haired dogs
  let total_time : ℕ := 4 * 60  -- total time in minutes (4 hours)
  
  num_full_haired_dogs = 
    (total_time - num_short_haired * time_short_haired) / time_full_haired :=
by sorry

end NUMINAMATH_CALUDE_groomer_problem_l1457_145708


namespace NUMINAMATH_CALUDE_isosceles_right_triangle_congruence_l1457_145714

/-- Given two congruent isosceles right triangles sharing a common base,
    if one leg of one triangle is 12, then the corresponding leg of the other triangle is 6√2 -/
theorem isosceles_right_triangle_congruence (a b c d : ℝ) :
  a = b ∧                    -- Triangle 1 is isosceles
  c = d ∧                    -- Triangle 2 is isosceles
  a^2 + a^2 = b^2 ∧          -- Triangle 1 is right-angled (Pythagorean theorem)
  c^2 + c^2 = d^2 ∧          -- Triangle 2 is right-angled (Pythagorean theorem)
  b = d ∧                    -- Triangles share a common base
  a = 12                     -- Given leg length in Triangle 1
  → c = 6 * Real.sqrt 2      -- To prove: corresponding leg in Triangle 2
:= by sorry

end NUMINAMATH_CALUDE_isosceles_right_triangle_congruence_l1457_145714


namespace NUMINAMATH_CALUDE_divisibility_implies_equality_l1457_145755

/-- For natural numbers a, b, and n, if a - k^n is divisible by b - k for all natural k ≠ b, 
    then a = b^n -/
theorem divisibility_implies_equality (a b n : ℕ) 
  (h : ∀ k : ℕ, k ≠ b → (b - k) ∣ (a - k^n)) : 
  a = b^n := by
sorry

end NUMINAMATH_CALUDE_divisibility_implies_equality_l1457_145755


namespace NUMINAMATH_CALUDE_quadratic_always_real_roots_triangle_perimeter_l1457_145733

-- Define the quadratic equation
def quadratic_equation (x k : ℝ) : Prop :=
  x^2 - (k + 2) * x + 2 * k = 0

-- Theorem 1: The equation always has real roots
theorem quadratic_always_real_roots (k : ℝ) :
  ∃ x : ℝ, quadratic_equation x k := by sorry

-- Define a right triangle with hypotenuse 3 and other sides as roots of the equation
def right_triangle_from_equation (k : ℝ) : Prop :=
  ∃ b c : ℝ,
    quadratic_equation b k ∧
    quadratic_equation c k ∧
    b^2 + c^2 = 3^2

-- Theorem 2: The perimeter of the triangle is 5 + √5
theorem triangle_perimeter (k : ℝ) :
  right_triangle_from_equation k →
  ∃ b c : ℝ, b + c + 3 = 5 + Real.sqrt 5 := by sorry

end NUMINAMATH_CALUDE_quadratic_always_real_roots_triangle_perimeter_l1457_145733


namespace NUMINAMATH_CALUDE_parabola_y_relationship_l1457_145707

/-- A parabola defined by y = -x² + 6x + c -/
def parabola (x : ℝ) (c : ℝ) : ℝ := -x^2 + 6*x + c

/-- Three points on the parabola -/
structure PointsOnParabola (c : ℝ) where
  y₁ : ℝ
  y₂ : ℝ
  y₃ : ℝ
  h₁ : parabola 1 c = y₁
  h₂ : parabola 3 c = y₂
  h₃ : parabola 4 c = y₃

/-- Theorem stating the relationship between y₁, y₂, and y₃ -/
theorem parabola_y_relationship (c : ℝ) (p : PointsOnParabola c) :
  p.y₁ < p.y₃ ∧ p.y₃ < p.y₂ := by sorry

end NUMINAMATH_CALUDE_parabola_y_relationship_l1457_145707


namespace NUMINAMATH_CALUDE_imaginary_part_of_reciprocal_l1457_145723

theorem imaginary_part_of_reciprocal (z : ℂ) : z = 1 / (2 - I) → z.im = 1 / 5 := by
  sorry

end NUMINAMATH_CALUDE_imaginary_part_of_reciprocal_l1457_145723


namespace NUMINAMATH_CALUDE_tank_cost_l1457_145711

theorem tank_cost (buy_price sell_price : ℚ) (num_sold : ℕ) (profit_percentage : ℚ) :
  buy_price = 0.25 →
  sell_price = 0.75 →
  num_sold = 110 →
  profit_percentage = 0.55 →
  (sell_price - buy_price) * num_sold = profit_percentage * 100 :=
by sorry

end NUMINAMATH_CALUDE_tank_cost_l1457_145711


namespace NUMINAMATH_CALUDE_line_through_point_with_equal_intercepts_l1457_145765

-- Define a line in 2D space
structure Line2D where
  a : ℝ
  b : ℝ
  c : ℝ

-- Define a point in 2D space
structure Point2D where
  x : ℝ
  y : ℝ

-- Function to check if a point is on a line
def pointOnLine (p : Point2D) (l : Line2D) : Prop :=
  l.a * p.x + l.b * p.y + l.c = 0

-- Function to check if a line has equal intercepts
def hasEqualIntercepts (l : Line2D) : Prop :=
  l.a ≠ 0 ∧ l.b ≠ 0 ∧ l.c / l.a = l.c / l.b

-- Theorem statement
theorem line_through_point_with_equal_intercepts :
  ∃ (l : Line2D), pointOnLine ⟨1, 1⟩ l ∧ hasEqualIntercepts l ∧
    ((l.a = 1 ∧ l.b = 1 ∧ l.c = -2) ∨ (l.a = -1 ∧ l.b = 1 ∧ l.c = 0)) :=
sorry

end NUMINAMATH_CALUDE_line_through_point_with_equal_intercepts_l1457_145765


namespace NUMINAMATH_CALUDE_odd_sum_probability_l1457_145777

theorem odd_sum_probability (n : Nat) (h : n = 16) :
  let grid_size := 4
  let total_arrangements := n.factorial
  let valid_arrangements := (grid_size.choose 2) * (n / 2).factorial * (n / 2).factorial
  (valid_arrangements : ℚ) / total_arrangements = 1 / 2150 := by
  sorry

end NUMINAMATH_CALUDE_odd_sum_probability_l1457_145777


namespace NUMINAMATH_CALUDE_right_triangle_345_l1457_145741

def is_right_triangle (a b c : ℕ) : Prop :=
  a * a + b * b = c * c ∨ a * a + c * c = b * b ∨ b * b + c * c = a * a

theorem right_triangle_345 :
  is_right_triangle 3 4 5 ∧
  ¬is_right_triangle 1 2 3 ∧
  ¬is_right_triangle 2 3 4 ∧
  ¬is_right_triangle 4 5 6 :=
by sorry

end NUMINAMATH_CALUDE_right_triangle_345_l1457_145741


namespace NUMINAMATH_CALUDE_power_of_x_in_product_l1457_145713

theorem power_of_x_in_product (x y z : ℕ) (hx : Prime x) (hy : Prime y) (hz : Prime z) 
  (hdiff : x ≠ y ∧ y ≠ z ∧ x ≠ z) :
  ∃ (a b c : ℕ), (a + 1) * (b + 1) * (c + 1) = 12 ∧ a = 1 := by
  sorry

end NUMINAMATH_CALUDE_power_of_x_in_product_l1457_145713


namespace NUMINAMATH_CALUDE_netflix_series_episodes_l1457_145735

/-- A TV series with the given properties -/
structure TVSeries where
  seasons : ℕ
  episodes_per_day : ℕ
  days_to_complete : ℕ

/-- Calculate the number of episodes per season -/
def episodes_per_season (series : TVSeries) : ℕ :=
  (series.episodes_per_day * series.days_to_complete) / series.seasons

/-- Theorem stating that for the given TV series, each season has 20 episodes -/
theorem netflix_series_episodes (series : TVSeries) 
  (h1 : series.seasons = 3)
  (h2 : series.episodes_per_day = 2)
  (h3 : series.days_to_complete = 30) :
  episodes_per_season series = 20 := by
  sorry

#check netflix_series_episodes

end NUMINAMATH_CALUDE_netflix_series_episodes_l1457_145735


namespace NUMINAMATH_CALUDE_infinite_product_l1457_145742

open Set Filter

-- Define the concept of a function being infinite at a point
def IsInfiniteAt (f : ℝ → ℝ) (x₀ : ℝ) : Prop :=
  ∀ K > 0, ∃ δ > 0, ∀ x, 0 < |x - x₀| ∧ |x - x₀| < δ → |f x| > K

-- Define the theorem
theorem infinite_product (f g : ℝ → ℝ) (x₀ M : ℝ) (hM : M > 0)
    (hg : ∀ x, |x - x₀| > 0 → |g x| ≥ M)
    (hf : IsInfiniteAt f x₀) :
    IsInfiniteAt (fun x ↦ f x * g x) x₀ := by
  sorry


end NUMINAMATH_CALUDE_infinite_product_l1457_145742


namespace NUMINAMATH_CALUDE_fraction_meaningfulness_l1457_145739

theorem fraction_meaningfulness (x : ℝ) : 
  (∃ y : ℝ, y = 1 / (x + 2)) ↔ x ≠ -2 := by
  sorry

end NUMINAMATH_CALUDE_fraction_meaningfulness_l1457_145739


namespace NUMINAMATH_CALUDE_inequality_proof_l1457_145784

theorem inequality_proof (a b c : ℝ) (h1 : a > b) (h2 : b > 0) (h3 : c < 0) :
  a / (a - c) > b / (b - c) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l1457_145784


namespace NUMINAMATH_CALUDE_min_value_fraction_l1457_145783

theorem min_value_fraction (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0)
  (h : x - 2*y + 3*z = 0) : y^2 / (x*z) ≥ 3 :=
by sorry

end NUMINAMATH_CALUDE_min_value_fraction_l1457_145783


namespace NUMINAMATH_CALUDE_calculate_expression_l1457_145756

theorem calculate_expression : |-3| + 8 / (-2) + Real.sqrt 16 = 3 := by
  sorry

end NUMINAMATH_CALUDE_calculate_expression_l1457_145756


namespace NUMINAMATH_CALUDE_analysis_time_l1457_145799

/-- The number of bones in the human body -/
def num_bones : ℕ := 206

/-- The time in minutes spent analyzing each bone -/
def minutes_per_bone : ℕ := 45

/-- The number of minutes in an hour -/
def minutes_per_hour : ℕ := 60

/-- The total time in hours required to analyze all bones in the human body -/
theorem analysis_time : (num_bones * minutes_per_bone : ℚ) / minutes_per_hour = 154.5 := by
  sorry

end NUMINAMATH_CALUDE_analysis_time_l1457_145799


namespace NUMINAMATH_CALUDE_perpendicular_vectors_x_value_l1457_145745

def vector_a : ℝ × ℝ := (3, -1)
def vector_b (x : ℝ) : ℝ × ℝ := (1, x)

theorem perpendicular_vectors_x_value :
  ∀ x : ℝ, (vector_a.1 * (vector_b x).1 + vector_a.2 * (vector_b x).2 = 0) → x = 3 := by
  sorry

end NUMINAMATH_CALUDE_perpendicular_vectors_x_value_l1457_145745


namespace NUMINAMATH_CALUDE_one_true_statement_proves_normal_one_false_statement_proves_normal_one_statement_proves_normal_l1457_145769

-- Define the types of people on the island
inductive PersonType
  | Knight
  | Liar
  | Normal

-- Define a statement as either true or false
inductive Statement
  | True
  | False

-- Define a function to determine if a person can make a given statement
def canMakeStatement (person : PersonType) (statement : Statement) : Prop :=
  match person, statement with
  | PersonType.Knight, Statement.True => True
  | PersonType.Knight, Statement.False => False
  | PersonType.Liar, Statement.True => False
  | PersonType.Liar, Statement.False => True
  | PersonType.Normal, _ => True

-- Theorem: One true statement is sufficient to prove one is a normal person
theorem one_true_statement_proves_normal (person : PersonType) :
  canMakeStatement person Statement.True → person = PersonType.Normal :=
sorry

-- Theorem: One false statement is sufficient to prove one is a normal person
theorem one_false_statement_proves_normal (person : PersonType) :
  canMakeStatement person Statement.False → person = PersonType.Normal :=
sorry

-- Main theorem: Either one true or one false statement is sufficient to prove one is a normal person
theorem one_statement_proves_normal (person : PersonType) :
  (canMakeStatement person Statement.True ∨ canMakeStatement person Statement.False) →
  person = PersonType.Normal :=
sorry

end NUMINAMATH_CALUDE_one_true_statement_proves_normal_one_false_statement_proves_normal_one_statement_proves_normal_l1457_145769


namespace NUMINAMATH_CALUDE_total_logs_cut_l1457_145788

/-- The number of logs produced by cutting different types of trees -/
theorem total_logs_cut (pine_logs maple_logs walnut_logs oak_logs birch_logs : ℕ)
  (pine_trees maple_trees walnut_trees oak_trees birch_trees : ℕ)
  (h1 : pine_logs = 80)
  (h2 : maple_logs = 60)
  (h3 : walnut_logs = 100)
  (h4 : oak_logs = 90)
  (h5 : birch_logs = 55)
  (h6 : pine_trees = 8)
  (h7 : maple_trees = 3)
  (h8 : walnut_trees = 4)
  (h9 : oak_trees = 7)
  (h10 : birch_trees = 5) :
  pine_logs * pine_trees + maple_logs * maple_trees + walnut_logs * walnut_trees +
  oak_logs * oak_trees + birch_logs * birch_trees = 2125 := by
  sorry

end NUMINAMATH_CALUDE_total_logs_cut_l1457_145788


namespace NUMINAMATH_CALUDE_motorcycle_meeting_distance_l1457_145768

/-- The distance traveled by a constant speed motorcyclist when meeting an accelerating motorcyclist on a circular track -/
theorem motorcycle_meeting_distance (v : ℝ) (a : ℝ) : 
  v > 0 → a > 0 →
  v * (1 / v) = 1 →
  (1/2) * a * (1 / v)^2 = 1 →
  ∃ (T : ℝ), T > 0 ∧ v * T + (1/2) * a * T^2 = 1 →
  v * T = (-1 + Real.sqrt 5) / 2 := by
sorry

end NUMINAMATH_CALUDE_motorcycle_meeting_distance_l1457_145768


namespace NUMINAMATH_CALUDE_quadratic_integer_roots_l1457_145790

theorem quadratic_integer_roots (n : ℕ+) :
  (∃ x : ℤ, x^2 - 4*x + n = 0) ↔ (n = 3 ∨ n = 4) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_integer_roots_l1457_145790


namespace NUMINAMATH_CALUDE_max_cola_bottles_30_yuan_l1457_145747

/-- Calculates the maximum number of cola bottles that can be consumed given an initial amount of money, the cost per bottle, and the exchange rate of empty bottles for full bottles. -/
def max_cola_bottles (initial_money : ℕ) (cost_per_bottle : ℕ) (exchange_rate : ℕ) : ℕ :=
  sorry

/-- Theorem stating that given 30 yuan, a cola cost of 2 yuan per bottle, and the ability to exchange 2 empty bottles for 1 full bottle, the maximum number of cola bottles that can be consumed is 29. -/
theorem max_cola_bottles_30_yuan :
  max_cola_bottles 30 2 2 = 29 :=
sorry

end NUMINAMATH_CALUDE_max_cola_bottles_30_yuan_l1457_145747


namespace NUMINAMATH_CALUDE_only_expr1_is_inequality_l1457_145727

-- Define the type for mathematical expressions
inductive MathExpression
  | LessThan : ℝ → ℝ → MathExpression
  | LinearExpr : ℝ → ℝ → MathExpression
  | Equation : ℝ → ℝ → ℝ → ℝ → MathExpression
  | Monomial : ℝ → ℕ → MathExpression

-- Define what it means for an expression to be an inequality
def isInequality : MathExpression → Prop
  | MathExpression.LessThan _ _ => True
  | _ => False

-- Define the given expressions
def expr1 : MathExpression := MathExpression.LessThan 0 19
def expr2 : MathExpression := MathExpression.LinearExpr 1 (-2)
def expr3 : MathExpression := MathExpression.Equation 2 3 (-1) 0
def expr4 : MathExpression := MathExpression.Monomial 1 2

-- Theorem statement
theorem only_expr1_is_inequality :
  isInequality expr1 ∧
  ¬isInequality expr2 ∧
  ¬isInequality expr3 ∧
  ¬isInequality expr4 :=
by sorry

end NUMINAMATH_CALUDE_only_expr1_is_inequality_l1457_145727


namespace NUMINAMATH_CALUDE_inequality_solution_l1457_145702

theorem inequality_solution (x : ℝ) : 
  (x^2 + 2*x^3 - 3*x^4) / (2*x + 3*x^2 - 4*x^3) ≥ -1 ↔ 
  (x ∈ Set.Icc (-1) ((-3 - Real.sqrt 41) / -8) ∪ 
   Set.Ioo ((-3 - Real.sqrt 41) / -8) ((-3 + Real.sqrt 41) / -8) ∪
   Set.Ioo ((-3 + Real.sqrt 41) / -8) 0 ∪
   Set.Ioi 0) ∧
  (x ≠ 0) ∧ (x ≠ ((-3 - Real.sqrt 41) / -8)) ∧ (x ≠ ((-3 + Real.sqrt 41) / -8)) :=
by sorry

end NUMINAMATH_CALUDE_inequality_solution_l1457_145702


namespace NUMINAMATH_CALUDE_blueberry_muffin_percentage_l1457_145780

/-- The number of cartons of blueberries Mason has -/
def num_cartons : ℕ := 8

/-- The number of blueberries in each carton -/
def blueberries_per_carton : ℕ := 300

/-- The number of blueberries used per muffin -/
def blueberries_per_muffin : ℕ := 18

/-- The number of blueberries left after making blueberry muffins -/
def blueberries_left : ℕ := 54

/-- The number of cinnamon muffins made -/
def cinnamon_muffins : ℕ := 80

/-- The number of chocolate muffins made -/
def chocolate_muffins : ℕ := 40

/-- The number of cranberry muffins made -/
def cranberry_muffins : ℕ := 50

/-- The number of lemon muffins made -/
def lemon_muffins : ℕ := 30

/-- Theorem stating that the percentage of blueberry muffins is approximately 39.39% -/
theorem blueberry_muffin_percentage :
  let total_blueberries := num_cartons * blueberries_per_carton
  let used_blueberries := total_blueberries - blueberries_left
  let blueberry_muffins := used_blueberries / blueberries_per_muffin
  let total_muffins := blueberry_muffins + cinnamon_muffins + chocolate_muffins + cranberry_muffins + lemon_muffins
  let percentage := (blueberry_muffins : ℚ) / (total_muffins : ℚ) * 100
  abs (percentage - 39.39) < 0.01 :=
by sorry

end NUMINAMATH_CALUDE_blueberry_muffin_percentage_l1457_145780


namespace NUMINAMATH_CALUDE_garden_furniture_cost_l1457_145766

/-- The combined cost of a garden table and bench, given their price relationship -/
theorem garden_furniture_cost (bench_price : ℕ) (table_price : ℕ) : 
  bench_price = 150 → 
  table_price = 2 * bench_price → 
  bench_price + table_price = 450 := by
  sorry

end NUMINAMATH_CALUDE_garden_furniture_cost_l1457_145766


namespace NUMINAMATH_CALUDE_sum_of_digits_of_greatest_prime_divisor_l1457_145720

def number : ℕ := 32767

/-- The greatest prime divisor of a natural number -/
def greatest_prime_divisor (n : ℕ) : ℕ := sorry

/-- Sum of digits of a natural number -/
def sum_of_digits (n : ℕ) : ℕ := sorry

theorem sum_of_digits_of_greatest_prime_divisor :
  sum_of_digits (greatest_prime_divisor number) = 14 := by sorry

end NUMINAMATH_CALUDE_sum_of_digits_of_greatest_prime_divisor_l1457_145720


namespace NUMINAMATH_CALUDE_task_assignment_count_l1457_145724

/-- The number of ways to assign tasks to volunteers -/
def task_assignments (num_volunteers : ℕ) (num_tasks : ℕ) : ℕ :=
  -- Number of ways to divide tasks into groups
  (num_tasks.choose (num_tasks - num_volunteers)) *
  -- Number of ways to permute volunteers
  (num_volunteers.factorial)

/-- Theorem: There are 36 ways to assign 4 tasks to 3 volunteers -/
theorem task_assignment_count :
  task_assignments 3 4 = 36 := by
sorry

end NUMINAMATH_CALUDE_task_assignment_count_l1457_145724


namespace NUMINAMATH_CALUDE_gcf_of_4620_and_10780_l1457_145748

theorem gcf_of_4620_and_10780 : Nat.gcd 4620 10780 = 1540 := by
  sorry

end NUMINAMATH_CALUDE_gcf_of_4620_and_10780_l1457_145748


namespace NUMINAMATH_CALUDE_vector_expression_l1457_145761

/-- Given vectors a, b, and c in ℝ², prove that c = 2a - b -/
theorem vector_expression (a b c : ℝ × ℝ) 
  (ha : a = (1, 2)) 
  (hb : b = (-2, 3)) 
  (hc : c = (4, 1)) : 
  c = (2 : ℝ) • a - b := by sorry

end NUMINAMATH_CALUDE_vector_expression_l1457_145761


namespace NUMINAMATH_CALUDE_eat_cereal_together_l1457_145794

/-- The time taken for two people to eat a certain amount of cereal together -/
def time_to_eat_together (rate1 rate2 amount : ℚ) : ℚ :=
  amount / (rate1 + rate2)

/-- Theorem: Mr. Fat and Mr. Thin will take 96 minutes to eat 4 pounds of cereal together -/
theorem eat_cereal_together : 
  let fat_rate : ℚ := 1 / 40
  let thin_rate : ℚ := 1 / 15
  let amount : ℚ := 4
  time_to_eat_together fat_rate thin_rate amount = 96 := by
  sorry

#eval time_to_eat_together (1 / 40) (1 / 15) 4

end NUMINAMATH_CALUDE_eat_cereal_together_l1457_145794


namespace NUMINAMATH_CALUDE_canoe_kayak_revenue_l1457_145771

/-- Represents the revenue calculation for a canoe and kayak rental business --/
theorem canoe_kayak_revenue
  (canoe_cost : ℕ)
  (kayak_cost : ℕ)
  (canoe_kayak_ratio : ℚ)
  (canoe_kayak_difference : ℕ)
  (h1 : canoe_cost = 12)
  (h2 : kayak_cost = 18)
  (h3 : canoe_kayak_ratio = 3 / 2)
  (h4 : canoe_kayak_difference = 7) :
  ∃ (num_canoes num_kayaks : ℕ),
    num_canoes = num_kayaks + canoe_kayak_difference ∧
    (num_canoes : ℚ) / num_kayaks = canoe_kayak_ratio ∧
    num_canoes * canoe_cost + num_kayaks * kayak_cost = 504 :=
by sorry

end NUMINAMATH_CALUDE_canoe_kayak_revenue_l1457_145771


namespace NUMINAMATH_CALUDE_inequality_statements_l1457_145701

theorem inequality_statements :
  (∃ (a b : ℝ) (c : ℝ), c < 0 ∧ a < b ∧ c * a > c * b) ∧
  (∀ (a b : ℝ), a > 0 → b > 0 → a ≠ b → (2 * a * b) / (a + b) < Real.sqrt (a * b)) ∧
  (∀ (k : ℝ), k > 0 → ∀ (a b : ℝ), a > 0 → b > 0 → a * b = k → 
    (∀ (x y : ℝ), x > 0 → y > 0 → x * y = k → a + b ≤ x + y)) ∧
  (∀ (a b : ℝ), a > 0 → b > 0 → (a^2 + b^2) / 2 < (a + b)^2) ∧
  (∀ (a b : ℝ), a > 0 → b > 0 → (a + b)^2 ≥ a^2 + b^2) := by
  sorry

end NUMINAMATH_CALUDE_inequality_statements_l1457_145701


namespace NUMINAMATH_CALUDE_min_value_of_P_l1457_145759

/-- The polynomial function P(x,y) -/
def P (x y : ℝ) : ℝ := 4 + x^2 * y^4 + x^4 * y^2 - 3 * x^2 * y^2

/-- Theorem stating that the minimal value of P(x,y) is 3 -/
theorem min_value_of_P :
  (∀ x y : ℝ, P x y ≥ 3) ∧ (∃ x y : ℝ, P x y = 3) := by
  sorry

end NUMINAMATH_CALUDE_min_value_of_P_l1457_145759


namespace NUMINAMATH_CALUDE_hot_dogs_per_pack_l1457_145738

theorem hot_dogs_per_pack (total_hot_dogs : ℕ) (buns_per_pack : ℕ) (hot_dogs_per_pack : ℕ) : 
  total_hot_dogs = 36 →
  buns_per_pack = 9 →
  total_hot_dogs % buns_per_pack = 0 →
  total_hot_dogs % hot_dogs_per_pack = 0 →
  total_hot_dogs / buns_per_pack = total_hot_dogs / hot_dogs_per_pack →
  hot_dogs_per_pack = 9 := by
  sorry

end NUMINAMATH_CALUDE_hot_dogs_per_pack_l1457_145738


namespace NUMINAMATH_CALUDE_consecutive_four_product_plus_one_is_square_l1457_145757

theorem consecutive_four_product_plus_one_is_square (n : ℕ) :
  ∃ k : ℕ, n * (n + 1) * (n + 2) * (n + 3) + 1 = k ^ 2 := by
  sorry

end NUMINAMATH_CALUDE_consecutive_four_product_plus_one_is_square_l1457_145757


namespace NUMINAMATH_CALUDE_star_neg_two_three_l1457_145704

/-- The "star" operation for rational numbers -/
def star (a b : ℚ) : ℚ := a * b^2 + a

/-- Theorem: The result of (-2)☆3 is -20 -/
theorem star_neg_two_three : star (-2) 3 = -20 := by
  sorry

end NUMINAMATH_CALUDE_star_neg_two_three_l1457_145704


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_l1457_145762

/-- The function f(x) = ax + 3 -/
def f (a : ℝ) (x : ℝ) : ℝ := a * x + 3

/-- The condition a < -2 is sufficient but not necessary for f to have a zero in [-1, 2] -/
theorem sufficient_not_necessary (a : ℝ) :
  (a < -2 → ∃ x ∈ Set.Icc (-1) 2, f a x = 0) ∧
  ¬(∃ x ∈ Set.Icc (-1) 2, f a x = 0 → a < -2) :=
sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_l1457_145762


namespace NUMINAMATH_CALUDE_campsite_coordinates_l1457_145797

/-- Calculates the coordinates of a point that divides a line segment in a given ratio -/
def divideLineSegment (x1 y1 x2 y2 m n : ℚ) : ℚ × ℚ :=
  ((m * x2 + n * x1) / (m + n), (m * y2 + n * y1) / (m + n))

/-- The campsite coordinates problem -/
theorem campsite_coordinates :
  let annaStart : ℚ × ℚ := (3, -5)
  let bobStart : ℚ × ℚ := (7, 4)
  let campsite := divideLineSegment annaStart.1 annaStart.2 bobStart.1 bobStart.2 2 1
  campsite = (17/3, 1) := by
  sorry


end NUMINAMATH_CALUDE_campsite_coordinates_l1457_145797


namespace NUMINAMATH_CALUDE_trigonometric_equation_solution_l1457_145764

theorem trigonometric_equation_solution :
  ∃! x : ℝ, 0 < x ∧ x < 180 ∧
  Real.tan ((150 - x) * π / 180) = 
    (Real.sin (150 * π / 180) - Real.sin (x * π / 180)) /
    (Real.cos (150 * π / 180) - Real.cos (x * π / 180)) ∧
  x = 110 := by
sorry

end NUMINAMATH_CALUDE_trigonometric_equation_solution_l1457_145764


namespace NUMINAMATH_CALUDE_sqrt_equation_solution_l1457_145786

theorem sqrt_equation_solution :
  ∃ y : ℚ, (40 : ℚ) / 60 = Real.sqrt (y / 60) → y = 80 / 3 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_equation_solution_l1457_145786


namespace NUMINAMATH_CALUDE_pole_top_distance_difference_l1457_145732

theorem pole_top_distance_difference 
  (h₁ : ℝ) (h₂ : ℝ) (d : ℝ)
  (height_pole1 : h₁ = 6)
  (height_pole2 : h₂ = 11)
  (distance_between_feet : d = 12) :
  Real.sqrt ((h₂ - h₁)^2 + d^2) - d = Real.sqrt 13 :=
by sorry

end NUMINAMATH_CALUDE_pole_top_distance_difference_l1457_145732


namespace NUMINAMATH_CALUDE_sum_of_integers_l1457_145729

theorem sum_of_integers (p q r s : ℤ) 
  (eq1 : p - q + r = 4)
  (eq2 : q - r + s = 5)
  (eq3 : r - s + p = 7)
  (eq4 : s - p + q = 3) :
  p + q + r + s = 19 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_integers_l1457_145729


namespace NUMINAMATH_CALUDE_amoeba_count_after_week_l1457_145767

/-- The number of amoebas after n days, given an initial population of 1 and a tripling rate each day -/
def amoeba_count (n : ℕ) : ℕ := 3^n

/-- Theorem: After 7 days, the number of amoebas is 2187 -/
theorem amoeba_count_after_week : amoeba_count 7 = 2187 := by
  sorry

end NUMINAMATH_CALUDE_amoeba_count_after_week_l1457_145767


namespace NUMINAMATH_CALUDE_polaroid_photo_length_l1457_145725

/-- The circumference of a rectangle given its length and width -/
def rectangleCircumference (length width : ℝ) : ℝ :=
  2 * (length + width)

/-- Theorem: A rectangular Polaroid photo with circumference 40 cm and width 8 cm has a length of 12 cm -/
theorem polaroid_photo_length (circumference width : ℝ) 
    (h_circumference : circumference = 40)
    (h_width : width = 8)
    (h_rect : rectangleCircumference length width = circumference) :
    length = 12 := by
  sorry


end NUMINAMATH_CALUDE_polaroid_photo_length_l1457_145725


namespace NUMINAMATH_CALUDE_coupon_value_l1457_145749

/-- Calculates the value of each coupon given the original price, discount percentage, number of bottles, and total cost after discounts and coupons. -/
theorem coupon_value (original_price discount_percent bottles total_cost : ℝ) : 
  original_price = 15 →
  discount_percent = 20 →
  bottles = 3 →
  total_cost = 30 →
  (original_price * (1 - discount_percent / 100) * bottles - total_cost) / bottles = 2 := by
  sorry

end NUMINAMATH_CALUDE_coupon_value_l1457_145749


namespace NUMINAMATH_CALUDE_inscribed_sphere_in_cone_l1457_145796

/-- Given a right cone with base radius 15 cm and height 30 cm, 
    and an inscribed sphere with radius r = b√d - b cm, 
    prove that b + d = 12.5 -/
theorem inscribed_sphere_in_cone (b d : ℝ) : 
  let cone_base_radius : ℝ := 15
  let cone_height : ℝ := 30
  let sphere_radius : ℝ := b * (d.sqrt - 1)
  sphere_radius = (cone_height * cone_base_radius) / (cone_base_radius + (cone_base_radius^2 + cone_height^2).sqrt) →
  b + d = 12.5 := by sorry

end NUMINAMATH_CALUDE_inscribed_sphere_in_cone_l1457_145796


namespace NUMINAMATH_CALUDE_goldfish_to_pretzel_ratio_l1457_145763

/-- Given the number of pretzels, suckers, kids, and items per baggie, 
    prove that the ratio of goldfish to pretzels is 4:1 -/
theorem goldfish_to_pretzel_ratio 
  (pretzels : ℕ) 
  (suckers : ℕ) 
  (kids : ℕ) 
  (items_per_baggie : ℕ) 
  (h1 : pretzels = 64) 
  (h2 : suckers = 32) 
  (h3 : kids = 16) 
  (h4 : items_per_baggie = 22) : 
  (kids * items_per_baggie - pretzels - suckers) / pretzels = 4 := by
  sorry

end NUMINAMATH_CALUDE_goldfish_to_pretzel_ratio_l1457_145763


namespace NUMINAMATH_CALUDE_collinear_vectors_x_value_l1457_145772

/-- Two vectors are collinear if one is a scalar multiple of the other -/
def collinear (a b : ℝ × ℝ) : Prop :=
  ∃ k : ℝ, b.1 = k * a.1 ∧ b.2 = k * a.2

theorem collinear_vectors_x_value :
  let a : ℝ × ℝ := (1, 2)
  let b : ℝ × ℝ := (x, -4)
  collinear a b → x = -2 :=
by
  sorry

end NUMINAMATH_CALUDE_collinear_vectors_x_value_l1457_145772


namespace NUMINAMATH_CALUDE_integral_special_function_l1457_145719

theorem integral_special_function : 
  ∫ x in (0 : ℝ)..(Real.pi / 2), (1 - 5 * x^2) * Real.sin x = 11 - 5 * Real.pi := by
  sorry

end NUMINAMATH_CALUDE_integral_special_function_l1457_145719


namespace NUMINAMATH_CALUDE_max_value_sin_cos_l1457_145787

theorem max_value_sin_cos (x y : ℝ) (h : Real.sin x + Real.sin y = 1/3) :
  ∃ (M : ℝ), M = 4/9 ∧ ∀ (a b : ℝ), Real.sin a + Real.sin b = 1/3 →
  Real.sin b - Real.cos a ^ 2 ≤ M :=
by sorry

end NUMINAMATH_CALUDE_max_value_sin_cos_l1457_145787


namespace NUMINAMATH_CALUDE_least_n_radios_l1457_145753

/-- Represents the problem of finding the least number of radios bought by a dealer. -/
def RadioProblem (n d : ℕ) : Prop :=
  d > 0 ∧  -- d is a positive integer
  (2 * d + (n - 4) * (d + 10 * n)) = n * (d + 100)  -- profit equation

/-- The least possible value of n that satisfies the RadioProblem. -/
theorem least_n_radios : 
  ∀ n d, RadioProblem n d → n ≥ 14 :=
sorry

end NUMINAMATH_CALUDE_least_n_radios_l1457_145753


namespace NUMINAMATH_CALUDE_james_passenger_count_l1457_145798

/-- Calculates the total number of passengers James has seen --/
def total_passengers (total_vehicles : ℕ) (trucks : ℕ) (buses : ℕ) (cars : ℕ) 
  (truck_passengers : ℕ) (bus_passengers : ℕ) (taxi_passengers : ℕ) 
  (motorbike_passengers : ℕ) (car_passengers : ℕ) : ℕ :=
  let taxis := 2 * buses
  let motorbikes := total_vehicles - trucks - buses - taxis - cars
  trucks * truck_passengers + 
  buses * bus_passengers + 
  taxis * taxi_passengers + 
  motorbikes * motorbike_passengers + 
  cars * car_passengers

theorem james_passenger_count : 
  total_passengers 52 12 2 30 2 15 2 1 3 = 156 := by
  sorry

end NUMINAMATH_CALUDE_james_passenger_count_l1457_145798


namespace NUMINAMATH_CALUDE_ellipse_axis_ratio_l1457_145710

theorem ellipse_axis_ratio (k : ℝ) : 
  (∀ x y : ℝ, x^2 + k*y^2 = 1 → ∃ a b : ℝ, a > 0 ∧ b > 0 ∧ x^2/a^2 + y^2/b^2 = 1) →
  (∃ a b : ℝ, a > 0 ∧ b > 0 ∧ x^2/a^2 + y^2/b^2 = 1 ∧ a^2 = 1/k ∧ b^2 = 1) →
  (∃ a b : ℝ, a > 0 ∧ b > 0 ∧ 2*a = 2*b) →
  k = 1/4 :=
sorry

end NUMINAMATH_CALUDE_ellipse_axis_ratio_l1457_145710


namespace NUMINAMATH_CALUDE_inverse_power_of_three_l1457_145728

theorem inverse_power_of_three : 3⁻¹ = (1 : ℚ) / 3 := by sorry

end NUMINAMATH_CALUDE_inverse_power_of_three_l1457_145728


namespace NUMINAMATH_CALUDE_marks_additional_height_l1457_145778

/-- Proves that Mark is 3 inches tall in addition to his height in feet given the conditions -/
theorem marks_additional_height :
  -- Define constants
  let feet_to_inches : ℕ := 12
  let marks_feet : ℕ := 5
  let mikes_feet : ℕ := 6
  let mikes_additional_inches : ℕ := 1
  let height_difference : ℕ := 10

  -- Calculate Mike's height in inches
  let mikes_height : ℕ := mikes_feet * feet_to_inches + mikes_additional_inches

  -- Calculate Mark's height in inches
  let marks_height : ℕ := mikes_height - height_difference

  -- Calculate Mark's additional inches
  let marks_additional_inches : ℕ := marks_height - (marks_feet * feet_to_inches)

  -- Theorem statement
  marks_additional_inches = 3 := by
  sorry

end NUMINAMATH_CALUDE_marks_additional_height_l1457_145778


namespace NUMINAMATH_CALUDE_sum_of_fractions_l1457_145789

theorem sum_of_fractions (p q r : ℝ) 
  (h1 : p + q + r = 5) 
  (h2 : 1 / (p + q) + 1 / (q + r) + 1 / (p + r) = 9) : 
  r / (p + q) + p / (q + r) + q / (p + r) = 42 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_fractions_l1457_145789


namespace NUMINAMATH_CALUDE_scale_division_l1457_145760

/-- Proves that dividing a scale of 80 inches into 5 equal parts results in parts of 16 inches each -/
theorem scale_division (scale_length : ℕ) (num_parts : ℕ) (part_length : ℕ) :
  scale_length = 80 ∧ num_parts = 5 → part_length = scale_length / num_parts → part_length = 16 := by
  sorry

end NUMINAMATH_CALUDE_scale_division_l1457_145760


namespace NUMINAMATH_CALUDE_a_all_positive_l1457_145754

/-- Sequence a_n defined recursively -/
def a (α : ℝ) : ℕ → ℝ
  | 0 => α
  | n + 1 => 2 * a α n - n^2

/-- Theorem stating the condition for all terms of a_n to be positive -/
theorem a_all_positive (α : ℝ) : (∀ n : ℕ, a α n > 0) ↔ α ≥ 3 := by
  sorry

end NUMINAMATH_CALUDE_a_all_positive_l1457_145754


namespace NUMINAMATH_CALUDE_a_range_when_A_B_disjoint_l1457_145779

-- Define the sets A and B
def A (a : ℝ) : Set ℝ := {x : ℝ | |x - 1| ≤ a ∧ a > 0}
def B : Set ℝ := {x : ℝ | x^2 - 6*x - 7 > 0}

-- State the theorem
theorem a_range_when_A_B_disjoint :
  ∀ a : ℝ, (A a ∩ B = ∅) ↔ (0 < a ∧ a ≤ 6) :=
by sorry

end NUMINAMATH_CALUDE_a_range_when_A_B_disjoint_l1457_145779


namespace NUMINAMATH_CALUDE_angle_of_inclination_cosine_l1457_145773

theorem angle_of_inclination_cosine (θ : Real) :
  (∃ (m : Real), m = 2 ∧ θ = Real.arctan m) →
  Real.cos θ = Real.sqrt 5 / 5 := by
  sorry

end NUMINAMATH_CALUDE_angle_of_inclination_cosine_l1457_145773


namespace NUMINAMATH_CALUDE_rob_baseball_cards_l1457_145785

theorem rob_baseball_cards (rob_total : ℕ) (rob_doubles : ℕ) (jess_doubles : ℕ) :
  rob_doubles * 3 = rob_total →
  jess_doubles = rob_doubles * 5 →
  jess_doubles = 40 →
  rob_total = 24 := by
sorry

end NUMINAMATH_CALUDE_rob_baseball_cards_l1457_145785


namespace NUMINAMATH_CALUDE_cubic_function_symmetry_l1457_145734

/-- Given a cubic function f(x) = ax³ + bx + 1 where ab ≠ 0,
    if f(2016) = k, then f(-2016) = 2 - k -/
theorem cubic_function_symmetry (a b k : ℝ) (h1 : a * b ≠ 0) :
  let f : ℝ → ℝ := λ x ↦ a * x^3 + b * x + 1
  f 2016 = k → f (-2016) = 2 - k := by
  sorry

end NUMINAMATH_CALUDE_cubic_function_symmetry_l1457_145734


namespace NUMINAMATH_CALUDE_range_of_a_l1457_145791

-- Define the propositions p and q
def p (x : ℝ) : Prop := x^2 - 8*x - 20 < 0
def q (x a : ℝ) : Prop := x^2 - 2*x + 1 - a^2 ≤ 0

-- State the theorem
theorem range_of_a (a : ℝ) :
  (a > 0) →
  (∀ x, ¬(p x) → ¬(q x a)) →
  (∃ x, ¬(p x) ∧ (q x a)) →
  a ≥ 9 :=
by sorry

end NUMINAMATH_CALUDE_range_of_a_l1457_145791


namespace NUMINAMATH_CALUDE_mrs_hilt_travel_l1457_145715

/-- Calculates the total miles traveled given the number of books read and miles per book -/
def total_miles (books_read : ℕ) (miles_per_book : ℕ) : ℕ :=
  books_read * miles_per_book

/-- Proves that Mrs. Hilt traveled 6750 miles to Japan -/
theorem mrs_hilt_travel : total_miles 15 450 = 6750 := by
  sorry

end NUMINAMATH_CALUDE_mrs_hilt_travel_l1457_145715


namespace NUMINAMATH_CALUDE_bowling_ball_weight_l1457_145700

/-- Given that four identical canoes weigh the same as nine identical bowling balls,
    and one canoe weighs 36 pounds, prove that one bowling ball weighs 16 pounds. -/
theorem bowling_ball_weight (canoe_weight : ℝ) (ball_weight : ℝ) : 
  canoe_weight = 36 →  -- One canoe weighs 36 pounds
  4 * canoe_weight = 9 * ball_weight →  -- Four canoes weigh the same as nine bowling balls
  ball_weight = 16 :=  -- One bowling ball weighs 16 pounds
by
  sorry

#check bowling_ball_weight

end NUMINAMATH_CALUDE_bowling_ball_weight_l1457_145700


namespace NUMINAMATH_CALUDE_expression_equality_l1457_145751

theorem expression_equality : 
  (2025^3 - 3 * 2025^2 * 2026 + 5 * 2025 * 2026^2 - 2026^3 + 4) / (2025 * 2026) = 
  4052 + 3 / 2025 := by
sorry

end NUMINAMATH_CALUDE_expression_equality_l1457_145751


namespace NUMINAMATH_CALUDE_sum_of_circle_areas_l1457_145716

/-- The sum of the areas of an infinite sequence of circles with decreasing radii -/
theorem sum_of_circle_areas : 
  ∀ (π : ℝ), π > 0 → 
  (∑' n, π * (3 / (3 ^ n : ℝ))^2) = 9 * π / 8 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_circle_areas_l1457_145716


namespace NUMINAMATH_CALUDE_unique_monic_polynomial_l1457_145722

/-- A monic polynomial of degree 2 satisfying f(0) = 10 and f(1) = 14 -/
def f (x : ℝ) : ℝ := x^2 + 3*x + 10

/-- The theorem stating that f is the unique monic polynomial of degree 2 satisfying the given conditions -/
theorem unique_monic_polynomial :
  ∀ g : ℝ → ℝ, (∃ a b : ℝ, ∀ x, g x = x^2 + a*x + b) →
  g 0 = 10 → g 1 = 14 → g = f :=
by sorry

end NUMINAMATH_CALUDE_unique_monic_polynomial_l1457_145722


namespace NUMINAMATH_CALUDE_sequence_equation_proof_l1457_145744

/-- Given a sequence of equations, prove the value of (b+1)/a^2 -/
theorem sequence_equation_proof (a b : ℕ) (h : ∀ (n : ℕ), 32 ≤ n → n ≤ 32016 → 
  ∃ (m : ℕ), n + m / n = (n - 32 + 3) * (3 + m / n)) 
  (h_last : 32016 + a / b = 2016 * 3 * (a / b)) : 
  (b + 1) / (a^2 : ℚ) = 2016 := by
  sorry

end NUMINAMATH_CALUDE_sequence_equation_proof_l1457_145744


namespace NUMINAMATH_CALUDE_max_crate_weight_l1457_145706

/-- Proves that the maximum weight each crate can hold is 20 kg given the problem conditions --/
theorem max_crate_weight (num_crates : ℕ) (nail_bags : ℕ) (hammer_bags : ℕ) (plank_bags : ℕ)
  (nail_weight : ℝ) (hammer_weight : ℝ) (plank_weight : ℝ) (left_out_weight : ℝ) :
  num_crates = 15 →
  nail_bags = 4 →
  hammer_bags = 12 →
  plank_bags = 10 →
  nail_weight = 5 →
  hammer_weight = 5 →
  plank_weight = 30 →
  left_out_weight = 80 →
  (nail_bags * nail_weight + hammer_bags * hammer_weight + plank_bags * plank_weight - left_out_weight) / num_crates = 20 := by
  sorry

#check max_crate_weight

end NUMINAMATH_CALUDE_max_crate_weight_l1457_145706


namespace NUMINAMATH_CALUDE_curve_tangent_to_line_l1457_145717

/-- The curve y = x^2 - x + a is tangent to the line y = x + 1 if and only if a = 2 -/
theorem curve_tangent_to_line (a : ℝ) : 
  (∃ x y : ℝ, y = x^2 - x + a ∧ y = x + 1 ∧ 2*x - 1 = 1) ↔ a = 2 := by
  sorry

end NUMINAMATH_CALUDE_curve_tangent_to_line_l1457_145717


namespace NUMINAMATH_CALUDE_max_min_f_on_interval_l1457_145737

noncomputable def f (x : ℝ) : ℝ := x - Real.sqrt 2 * Real.sin x

theorem max_min_f_on_interval :
  let a := 0
  let b := Real.pi
  ∃ (x_max x_min : ℝ), x_max ∈ Set.Icc a b ∧ x_min ∈ Set.Icc a b ∧
    (∀ x ∈ Set.Icc a b, f x ≤ f x_max) ∧
    (∀ x ∈ Set.Icc a b, f x_min ≤ f x) ∧
    f x_max = Real.pi ∧
    f x_min = Real.pi / 4 - 1 :=
sorry

end NUMINAMATH_CALUDE_max_min_f_on_interval_l1457_145737


namespace NUMINAMATH_CALUDE_calzone_knead_time_l1457_145770

def calzone_time_problem (total_time onion_time knead_time : ℝ) : Prop :=
  let garlic_pepper_time := onion_time / 4
  let rest_time := 2 * knead_time
  let assemble_time := (knead_time + rest_time) / 10
  total_time = onion_time + garlic_pepper_time + knead_time + rest_time + assemble_time

theorem calzone_knead_time :
  ∃ (knead_time : ℝ), 
    calzone_time_problem 124 20 knead_time ∧ 
    knead_time = 30 := by
  sorry

end NUMINAMATH_CALUDE_calzone_knead_time_l1457_145770


namespace NUMINAMATH_CALUDE_no_valid_arrangement_with_odd_sums_l1457_145731

def Grid := Matrix (Fin 4) (Fin 4) Nat

def validArrangement (g : Grid) : Prop :=
  (∀ i j, g i j ∈ Finset.range 17) ∧
  (∀ i j k l, i ≠ k ∨ j ≠ l → g i j ≠ g k l)

def rowSum (g : Grid) (i : Fin 4) : Nat :=
  (Finset.range 4).sum (λ j => g i j)

def colSum (g : Grid) (j : Fin 4) : Nat :=
  (Finset.range 4).sum (λ i => g i j)

def mainDiagSum (g : Grid) : Nat :=
  (Finset.range 4).sum (λ i => g i i)

def antiDiagSum (g : Grid) : Nat :=
  (Finset.range 4).sum (λ i => g i (3 - i))

def allSumsOdd (g : Grid) : Prop :=
  (∀ i, Odd (rowSum g i)) ∧
  (∀ j, Odd (colSum g j)) ∧
  Odd (mainDiagSum g) ∧
  Odd (antiDiagSum g)

theorem no_valid_arrangement_with_odd_sums :
  ¬∃ g : Grid, validArrangement g ∧ allSumsOdd g :=
sorry

end NUMINAMATH_CALUDE_no_valid_arrangement_with_odd_sums_l1457_145731


namespace NUMINAMATH_CALUDE_cubic_roots_sum_cubes_l1457_145775

theorem cubic_roots_sum_cubes (p q r : ℝ) : 
  (3 * p^3 - 4 * p^2 + 220 * p - 7 = 0) →
  (3 * q^3 - 4 * q^2 + 220 * q - 7 = 0) →
  (3 * r^3 - 4 * r^2 + 220 * r - 7 = 0) →
  (p + q - 2)^3 + (q + r - 2)^3 + (r + p - 2)^3 = 64.556 := by
  sorry

end NUMINAMATH_CALUDE_cubic_roots_sum_cubes_l1457_145775


namespace NUMINAMATH_CALUDE_banana_distribution_exists_l1457_145750

-- Define the number of bananas and boxes
def total_bananas : ℕ := 40
def num_boxes : ℕ := 8

-- Define a valid distribution
def is_valid_distribution (dist : List ℕ) : Prop :=
  dist.length = num_boxes ∧
  dist.sum = total_bananas ∧
  dist.Nodup

-- Theorem statement
theorem banana_distribution_exists : 
  ∃ (dist : List ℕ), is_valid_distribution dist :=
sorry

end NUMINAMATH_CALUDE_banana_distribution_exists_l1457_145750


namespace NUMINAMATH_CALUDE_counterexample_exists_l1457_145793

theorem counterexample_exists : ∃ (a b c : ℝ), 
  (a^2 + b^2) / (b^2 + c^2) = a / c ∧ a / b ≠ b / c := by
  sorry

end NUMINAMATH_CALUDE_counterexample_exists_l1457_145793


namespace NUMINAMATH_CALUDE_trigonometric_equation_solution_l1457_145712

open Real

theorem trigonometric_equation_solution :
  ∀ x : ℝ, 2 * sin (2 * x) - cos (π / 2 + 3 * x) - cos (3 * x) * arccos (5 * x) * cos (π / 2 - 5 * x) = 0 ↔
  (∃ k : ℤ, x = k * π) ∨ (∃ n : ℤ, x = π / 15 + 2 * n * π / 5) ∨ (∃ n : ℤ, x = -π / 15 + 2 * n * π / 5) :=
by sorry

end NUMINAMATH_CALUDE_trigonometric_equation_solution_l1457_145712


namespace NUMINAMATH_CALUDE_dennis_floor_number_l1457_145730

theorem dennis_floor_number :
  ∀ (frank_floor charlie_floor bob_floor dennis_floor : ℕ),
    frank_floor = 16 →
    charlie_floor = frank_floor / 4 →
    charlie_floor = bob_floor + 1 →
    dennis_floor = charlie_floor + 2 →
    dennis_floor = 6 := by
  sorry

end NUMINAMATH_CALUDE_dennis_floor_number_l1457_145730


namespace NUMINAMATH_CALUDE_validBinaryStrings_10_l1457_145758

/-- A function that returns the number of binary strings of length n 
    that do not contain the substrings 101 or 010 -/
def validBinaryStrings (n : ℕ) : ℕ :=
  match n with
  | 0 => 1
  | 1 => 2
  | 2 => 4
  | 3 => 6
  | m + 3 => validBinaryStrings (m + 2) + validBinaryStrings (m + 1)

/-- Theorem stating that the number of binary strings of length 10 
    that do not contain the substrings 101 or 010 is 178 -/
theorem validBinaryStrings_10 : validBinaryStrings 10 = 178 := by
  sorry

end NUMINAMATH_CALUDE_validBinaryStrings_10_l1457_145758


namespace NUMINAMATH_CALUDE_prime_fourth_powers_sum_l1457_145721

theorem prime_fourth_powers_sum (p q r s : ℕ) : 
  Nat.Prime p ∧ Nat.Prime q ∧ Nat.Prime r ∧ Nat.Prime s →
  p ≤ q ∧ q ≤ r →
  p^4 + q^4 + r^4 + 119 = s^2 →
  p = 2 ∧ q = 3 ∧ r = 5 ∧ s = 29 := by
  sorry

end NUMINAMATH_CALUDE_prime_fourth_powers_sum_l1457_145721


namespace NUMINAMATH_CALUDE_subtraction_multiplication_addition_l1457_145743

theorem subtraction_multiplication_addition (x : ℤ) : 
  423 - x = 421 → (x * 423) + 421 = 1267 := by
  sorry

end NUMINAMATH_CALUDE_subtraction_multiplication_addition_l1457_145743


namespace NUMINAMATH_CALUDE_park_layout_diameter_l1457_145781

/-- The diameter of the outer boundary of a circular park layout -/
def outer_boundary_diameter (statue_diameter bench_width path_width : ℝ) : ℝ :=
  statue_diameter + 2 * (bench_width + path_width)

/-- Theorem: The diameter of the outer boundary of the jogging path is 46 feet -/
theorem park_layout_diameter :
  outer_boundary_diameter 12 10 7 = 46 := by
  sorry

end NUMINAMATH_CALUDE_park_layout_diameter_l1457_145781


namespace NUMINAMATH_CALUDE_triangle_property_l1457_145776

theorem triangle_property (a b c : ℝ) (A B C : ℝ) :
  -- Triangle ABC exists
  0 < a ∧ 0 < b ∧ 0 < c ∧ 
  0 < A ∧ A < π ∧ 0 < B ∧ B < π ∧ 0 < C ∧ C < π ∧
  A + B + C = π →
  -- Given conditions
  2 * b * Real.cos B = c * Real.cos A + a * Real.cos C →
  b = 2 →
  a + c = 4 →
  -- Conclusions
  Real.sin B = Real.sqrt 3 / 2 ∧ 
  (1/2 : ℝ) * a * c * Real.sin B = Real.sqrt 3 := by
sorry

end NUMINAMATH_CALUDE_triangle_property_l1457_145776


namespace NUMINAMATH_CALUDE_quadratic_root_transformation_l1457_145752

theorem quadratic_root_transformation (p q r : ℝ) (u v : ℝ) : 
  (p * u^2 + q * u + r = 0) ∧ (p * v^2 + q * v + r = 0) →
  ((2*p*u + q)^2 - (q/(4*p)) * (2*p*u + q) + r = 0) ∧
  ((2*p*v + q)^2 - (q/(4*p)) * (2*p*v + q) + r = 0) := by
sorry

end NUMINAMATH_CALUDE_quadratic_root_transformation_l1457_145752


namespace NUMINAMATH_CALUDE_total_digits_first_2500_even_integers_l1457_145792

/-- The number of digits in a positive integer -/
def numDigits (n : ℕ) : ℕ := sorry

/-- The sum of digits for all even numbers from 2 to n -/
def sumDigitsEven (n : ℕ) : ℕ := sorry

/-- The 2500th positive even integer -/
def nthEvenInteger : ℕ := 5000

theorem total_digits_first_2500_even_integers :
  sumDigitsEven nthEvenInteger = 9448 := by sorry

end NUMINAMATH_CALUDE_total_digits_first_2500_even_integers_l1457_145792


namespace NUMINAMATH_CALUDE_exists_polygon_with_area_16_l1457_145709

/-- A polygon represented by a list of points in 2D space -/
def Polygon := List (Real × Real)

/-- Calculate the area of a polygon given its vertices -/
def polygonArea (p : Polygon) : Real := sorry

/-- Check if a polygon can be formed from given line segments -/
def canFormPolygon (segments : List Real) (p : Polygon) : Prop := sorry

/-- The main theorem stating that a polygon with area 16 can be formed from 12 segments of length 2 -/
theorem exists_polygon_with_area_16 :
  ∃ (p : Polygon), 
    polygonArea p = 16 ∧ 
    canFormPolygon (List.replicate 12 2) p :=
sorry

end NUMINAMATH_CALUDE_exists_polygon_with_area_16_l1457_145709


namespace NUMINAMATH_CALUDE_zero_in_interval_l1457_145746

def f (x : ℝ) := x^3 + 3*x - 1

theorem zero_in_interval :
  (f 0 < 0) →
  (f 0.5 > 0) →
  (f 0.25 < 0) →
  ∃ x : ℝ, x ∈ Set.Ioo 0.25 0.5 ∧ f x = 0 :=
by
  sorry


end NUMINAMATH_CALUDE_zero_in_interval_l1457_145746


namespace NUMINAMATH_CALUDE_gwen_race_results_l1457_145705

/-- Represents the race details --/
structure RaceDetails where
  jogging_time : ℕ
  jogging_elevation : ℕ
  jogging_ratio : ℕ
  walking_ratio : ℕ

/-- Calculates the walking time based on race details --/
def walking_time (race : RaceDetails) : ℕ :=
  (race.jogging_time / race.jogging_ratio) * race.walking_ratio

/-- Calculates the total elevation gain based on race details --/
def total_elevation_gain (race : RaceDetails) : ℕ :=
  (race.jogging_elevation * (race.jogging_time + walking_time race)) / race.jogging_time

/-- Theorem stating the walking time and total elevation gain for Gwen's race --/
theorem gwen_race_results (race : RaceDetails) 
  (h1 : race.jogging_time = 15)
  (h2 : race.jogging_elevation = 500)
  (h3 : race.jogging_ratio = 5)
  (h4 : race.walking_ratio = 3) :
  walking_time race = 9 ∧ total_elevation_gain race = 800 := by
  sorry


end NUMINAMATH_CALUDE_gwen_race_results_l1457_145705


namespace NUMINAMATH_CALUDE_wood_per_table_is_12_l1457_145736

/-- The number of pieces of wood required to make a table -/
def wood_per_table : ℕ := sorry

/-- The total number of pieces of wood available -/
def total_wood : ℕ := 672

/-- The number of pieces of wood required to make a chair -/
def wood_per_chair : ℕ := 8

/-- The number of tables that can be made -/
def num_tables : ℕ := 24

/-- The number of chairs that can be made -/
def num_chairs : ℕ := 48

theorem wood_per_table_is_12 :
  wood_per_table = 12 :=
by sorry

end NUMINAMATH_CALUDE_wood_per_table_is_12_l1457_145736


namespace NUMINAMATH_CALUDE_inequality_proof_l1457_145718

theorem inequality_proof (x : ℝ) : 
  -2 < (x^2 - 10*x + 21) / (x^2 - 6*x + 10) ∧ 
  (x^2 - 10*x + 21) / (x^2 - 6*x + 10) < 3 ↔ 
  3/2 < x ∧ x < 3 :=
by sorry

end NUMINAMATH_CALUDE_inequality_proof_l1457_145718


namespace NUMINAMATH_CALUDE_magnitude_of_p_l1457_145703

def is_unit_vector (v : ℝ × ℝ) : Prop := v.1^2 + v.2^2 = 1

theorem magnitude_of_p (a b p : ℝ × ℝ) 
  (ha : is_unit_vector a) 
  (hb : is_unit_vector b) 
  (hab : a.1 * b.1 + a.2 * b.2 = -1/2) 
  (hpa : p.1 * a.1 + p.2 * a.2 = 1/2) 
  (hpb : p.1 * b.1 + p.2 * b.2 = 1/2) : 
  p.1^2 + p.2^2 = 1 := by
  sorry

end NUMINAMATH_CALUDE_magnitude_of_p_l1457_145703


namespace NUMINAMATH_CALUDE_element_in_set_l1457_145740

def U : Set ℕ := {1, 2, 3, 4, 5}

theorem element_in_set (M : Set ℕ) (h : Set.compl M = {1, 3}) : 2 ∈ M := by
  sorry

end NUMINAMATH_CALUDE_element_in_set_l1457_145740


namespace NUMINAMATH_CALUDE_birthday_problem_l1457_145782

theorem birthday_problem (n : ℕ) (m : ℕ) (h1 : n = 400) (h2 : m = 365) :
  ∃ (i j : ℕ), i ≠ j ∧ i ≤ n ∧ j ≤ n ∧ (i.mod m = j.mod m) :=
sorry

end NUMINAMATH_CALUDE_birthday_problem_l1457_145782


namespace NUMINAMATH_CALUDE_inscribed_square_side_length_l1457_145774

theorem inscribed_square_side_length (a b : ℝ) (ha : a = 7) (hb : b = 24) :
  let c := Real.sqrt (a^2 + b^2)
  let s := (a * b * c) / (a^2 + b^2)
  s = 525 / 96 := by sorry

end NUMINAMATH_CALUDE_inscribed_square_side_length_l1457_145774


namespace NUMINAMATH_CALUDE_quadratic_inequality_l1457_145795

theorem quadratic_inequality (x : ℝ) : -3*x^2 + 6*x + 9 > 0 ↔ -1 < x ∧ x < 3 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_l1457_145795
