import Mathlib

namespace NUMINAMATH_CALUDE_lines_are_skew_l1776_177630

/-- Two lines in 3D space are skew if and only if b is not equal to -79/19 -/
theorem lines_are_skew (b : ℚ) : 
  (∀ (t u : ℚ), (2 : ℚ) + 3*t ≠ 3 + 7*u ∨ 1 + 4*t ≠ 5 + 3*u ∨ b + 5*t ≠ 2 + u) ↔ 
  b ≠ -79/19 := by
sorry

end NUMINAMATH_CALUDE_lines_are_skew_l1776_177630


namespace NUMINAMATH_CALUDE_correct_number_of_pretzels_l1776_177662

/-- The number of pretzels in Mille's snack packs. -/
def pretzels : ℕ := 64

/-- The number of kids in the class. -/
def kids : ℕ := 16

/-- The number of items in each baggie. -/
def items_per_baggie : ℕ := 22

/-- The number of suckers. -/
def suckers : ℕ := 32

/-- Theorem stating that the number of pretzels is correct given the conditions. -/
theorem correct_number_of_pretzels :
  pretzels * 5 + suckers = kids * items_per_baggie :=
by sorry

end NUMINAMATH_CALUDE_correct_number_of_pretzels_l1776_177662


namespace NUMINAMATH_CALUDE_smallest_among_four_l1776_177600

theorem smallest_among_four (a b c d : ℚ) (h1 : a = -1) (h2 : b = 0) (h3 : c = 1) (h4 : d = -3) :
  d ≤ a ∧ d ≤ b ∧ d ≤ c := by
  sorry

end NUMINAMATH_CALUDE_smallest_among_four_l1776_177600


namespace NUMINAMATH_CALUDE_log_product_plus_exp_equals_seven_l1776_177666

theorem log_product_plus_exp_equals_seven :
  Real.log 9 / Real.log 2 * (Real.log 4 / Real.log 3) + (2 : ℝ) ^ (Real.log 3 / Real.log 2) = 7 := by
  sorry

end NUMINAMATH_CALUDE_log_product_plus_exp_equals_seven_l1776_177666


namespace NUMINAMATH_CALUDE_inequality_system_solution_l1776_177692

theorem inequality_system_solution :
  ∀ x : ℝ, (x - 7 < 5 * (x - 1) ∧ 4/3 * x + 3 ≥ 1 - 2/3 * x) ↔ x > -1/2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_system_solution_l1776_177692


namespace NUMINAMATH_CALUDE_intersection_of_M_and_N_l1776_177640

def M : Set ℕ := {1, 2}
def N : Set ℕ := {2, 3, 4}

theorem intersection_of_M_and_N : M ∩ N = {2} := by sorry

end NUMINAMATH_CALUDE_intersection_of_M_and_N_l1776_177640


namespace NUMINAMATH_CALUDE_function_extrema_sum_l1776_177678

def f (m : ℝ) (x : ℝ) : ℝ := x^3 - 3*x + m

theorem function_extrema_sum (m : ℝ) :
  (∃ (max min : ℝ), 
    (∀ x ∈ Set.Icc (-3 : ℝ) 0, f m x ≤ max) ∧ 
    (∃ x ∈ Set.Icc (-3 : ℝ) 0, f m x = max) ∧
    (∀ x ∈ Set.Icc (-3 : ℝ) 0, f m x ≥ min) ∧ 
    (∃ x ∈ Set.Icc (-3 : ℝ) 0, f m x = min) ∧
    (max + min = -1)) →
  m = 7.5 := by
sorry

end NUMINAMATH_CALUDE_function_extrema_sum_l1776_177678


namespace NUMINAMATH_CALUDE_unfair_coin_flip_probability_l1776_177606

/-- The probability of flipping exactly 3 heads in 8 flips of an unfair coin -/
theorem unfair_coin_flip_probability (p : ℚ) (h : p = 1/3) :
  let n : ℕ := 8
  let k : ℕ := 3
  let q : ℚ := 1 - p
  Nat.choose n k * p^k * q^(n-k) = 1792/6561 := by
  sorry

end NUMINAMATH_CALUDE_unfair_coin_flip_probability_l1776_177606


namespace NUMINAMATH_CALUDE_vector_subtraction_l1776_177659

theorem vector_subtraction (a b : ℝ × ℝ) :
  a = (3, 5) → b = (-2, 1) → a - 2 • b = (7, 3) := by
  sorry

end NUMINAMATH_CALUDE_vector_subtraction_l1776_177659


namespace NUMINAMATH_CALUDE_jacket_cost_calculation_l1776_177651

/-- The amount spent on clothes in cents -/
def total_spent : ℕ := 1428

/-- The amount spent on shorts in cents -/
def shorts_cost : ℕ := 954

/-- The amount spent on the jacket in cents -/
def jacket_cost : ℕ := total_spent - shorts_cost

theorem jacket_cost_calculation : jacket_cost = 474 := by
  sorry

end NUMINAMATH_CALUDE_jacket_cost_calculation_l1776_177651


namespace NUMINAMATH_CALUDE_inscribed_triangle_theorem_l1776_177634

/-- A triangle with an inscribed circle -/
structure InscribedTriangle where
  -- The sides of the triangle
  a : ℝ
  b : ℝ
  c : ℝ
  -- The radius of the inscribed circle
  r : ℝ
  -- The segments of one side divided by the point of tangency
  s₁ : ℝ
  s₂ : ℝ
  -- Conditions
  side_division : a = s₁ + s₂
  radius_positive : r > 0
  sides_positive : a > 0 ∧ b > 0 ∧ c > 0

/-- The theorem stating the relationship between the sides and radius -/
theorem inscribed_triangle_theorem (t : InscribedTriangle) 
  (h₁ : t.s₁ = 10 ∧ t.s₂ = 14)
  (h₂ : t.r = 5)
  (h₃ : t.b = 30) :
  t.c = 36 := by
  sorry

end NUMINAMATH_CALUDE_inscribed_triangle_theorem_l1776_177634


namespace NUMINAMATH_CALUDE_initial_weight_of_beef_l1776_177648

/-- The weight of a side of beef after five stages of processing --/
def final_weight (W : ℝ) : ℝ :=
  ((((W * 0.8) * 0.7) * 0.75) - 15) * 0.88

/-- Theorem stating the initial weight of the side of beef --/
theorem initial_weight_of_beef :
  ∃ W : ℝ, W > 0 ∧ final_weight W = 570 ∧ W = 1578 := by
  sorry

end NUMINAMATH_CALUDE_initial_weight_of_beef_l1776_177648


namespace NUMINAMATH_CALUDE_batsman_average_increase_l1776_177614

/-- Represents a batsman's cricket statistics -/
structure Batsman where
  innings : ℕ
  totalScore : ℕ
  notOutCount : ℕ

/-- Calculate the batting average -/
def battingAverage (b : Batsman) : ℚ :=
  b.totalScore / (b.innings - b.notOutCount)

/-- The increase in average after a new innings -/
def averageIncrease (before after : Batsman) : ℚ :=
  battingAverage after - battingAverage before

theorem batsman_average_increase :
  ∀ (before : Batsman),
    before.innings = 19 →
    before.notOutCount = 0 →
    let after : Batsman :=
      { innings := 20
      , totalScore := before.totalScore + 90
      , notOutCount := 0
      }
    battingAverage after = 52 →
    averageIncrease before after = 2 := by
  sorry

end NUMINAMATH_CALUDE_batsman_average_increase_l1776_177614


namespace NUMINAMATH_CALUDE_sin_monotone_increasing_l1776_177615

open Real

theorem sin_monotone_increasing (t : ℝ) (h : 0 < t ∧ t < π / 6) :
  StrictMonoOn (fun x => sin (2 * x + π / 6)) (Set.Ioo (-t) t) := by
  sorry

end NUMINAMATH_CALUDE_sin_monotone_increasing_l1776_177615


namespace NUMINAMATH_CALUDE_repair_cost_is_5000_l1776_177665

/-- Calculates the repair cost of a machine given its purchase price, transportation charges,
    profit percentage, and final selling price. -/
def repair_cost (purchase_price : ℤ) (transportation_charges : ℤ) (profit_percentage : ℚ)
                (selling_price : ℤ) : ℚ :=
  ((selling_price : ℚ) - (1 + profit_percentage) * ((purchase_price + transportation_charges) : ℚ)) /
  (1 + profit_percentage)

/-- Theorem stating that the repair cost is 5000 given the specific conditions -/
theorem repair_cost_is_5000 :
  repair_cost 13000 1000 (1/2) 28500 = 5000 := by
  sorry

end NUMINAMATH_CALUDE_repair_cost_is_5000_l1776_177665


namespace NUMINAMATH_CALUDE_boa_constrictor_length_l1776_177695

/-- The length of a boa constrictor given the length of a garden snake and their relative sizes -/
theorem boa_constrictor_length 
  (garden_snake_length : ℝ) 
  (relative_size : ℝ) 
  (h1 : garden_snake_length = 10.0)
  (h2 : relative_size = 7.0) : 
  garden_snake_length / relative_size = 10.0 / 7.0 := by sorry

end NUMINAMATH_CALUDE_boa_constrictor_length_l1776_177695


namespace NUMINAMATH_CALUDE_equation_solutions_l1776_177663

def solution_set : Set (ℤ × ℤ) :=
  {(6, 3), (6, -9), (1, 1), (1, -2), (2, -1)}

def satisfies_equation (p : ℤ × ℤ) : Prop :=
  let (x, y) := p
  y * (x + y) = x^3 - 7*x^2 + 11*x - 3

theorem equation_solutions :
  ∀ p : ℤ × ℤ, satisfies_equation p ↔ p ∈ solution_set :=
sorry

end NUMINAMATH_CALUDE_equation_solutions_l1776_177663


namespace NUMINAMATH_CALUDE_minimum_cats_with_stripes_and_black_ear_l1776_177649

theorem minimum_cats_with_stripes_and_black_ear (total_cats : ℕ) (mice_catchers : ℕ) 
  (striped_cats : ℕ) (black_ear_cats : ℕ) 
  (h1 : total_cats = 66) (h2 : mice_catchers = 21) 
  (h3 : striped_cats = 32) (h4 : black_ear_cats = 27) : 
  ∃ (x : ℕ), x = 14 ∧ 
  x ≤ striped_cats ∧ 
  x ≤ black_ear_cats ∧
  x ≤ total_cats - mice_catchers ∧
  ∀ (y : ℕ), y < x → 
    y > striped_cats + black_ear_cats - (total_cats - mice_catchers) := by
  sorry

end NUMINAMATH_CALUDE_minimum_cats_with_stripes_and_black_ear_l1776_177649


namespace NUMINAMATH_CALUDE_fraction_simplification_l1776_177631

theorem fraction_simplification (b y : ℝ) (h : b^2 + y^3 ≠ 0) :
  (Real.sqrt (b^2 + y^3) - (y^3 - b^2) / Real.sqrt (b^2 + y^3)) / (b^2 + y^3) = 
  2 * b^2 / (b^2 + y^3)^(3/2) := by
  sorry

end NUMINAMATH_CALUDE_fraction_simplification_l1776_177631


namespace NUMINAMATH_CALUDE_jennys_grade_is_95_l1776_177617

-- Define the grades as natural numbers
def jennys_grade : ℕ := sorry
def jasons_grade : ℕ := sorry
def bobs_grade : ℕ := sorry

-- State the conditions
axiom condition1 : jasons_grade = jennys_grade - 25
axiom condition2 : bobs_grade = jasons_grade / 2
axiom condition3 : bobs_grade = 35

-- Theorem to prove
theorem jennys_grade_is_95 : jennys_grade = 95 := by sorry

end NUMINAMATH_CALUDE_jennys_grade_is_95_l1776_177617


namespace NUMINAMATH_CALUDE_ellipse_area_l1776_177661

/-- The area of an ellipse with semi-major axis a and semi-minor axis b is k*π where k = a*b -/
theorem ellipse_area (a b : ℝ) (h1 : a = 12) (h2 : b = 6) : ∃ k : ℝ, k = 72 ∧ a * b * π = k * π := by
  sorry

end NUMINAMATH_CALUDE_ellipse_area_l1776_177661


namespace NUMINAMATH_CALUDE_inverse_proportion_example_l1776_177642

/-- Two real numbers are inversely proportional if their product is constant -/
def InverselyProportional (x y : ℝ → ℝ) : Prop :=
  ∃ k : ℝ, ∀ t : ℝ, x t * y t = k

theorem inverse_proportion_example :
  ∀ x y : ℝ → ℝ,
  InverselyProportional x y →
  x 1 = 40 →
  y 1 = 8 →
  y 2 = 20 →
  x 2 = 16 := by
sorry

end NUMINAMATH_CALUDE_inverse_proportion_example_l1776_177642


namespace NUMINAMATH_CALUDE_collinearity_condition_acute_angle_condition_l1776_177607

-- Define the vectors
def OA : ℝ × ℝ := (3, -4)
def OB : ℝ × ℝ := (6, -3)
def OC (m : ℝ) : ℝ × ℝ := (5 - m, -3 - m)

-- Define collinearity
def collinear (A B C : ℝ × ℝ) : Prop :=
  ∃ t : ℝ, C.1 - A.1 = t * (B.1 - A.1) ∧ C.2 - A.2 = t * (B.2 - A.2)

-- Define acute angle
def acute_angle (A B C : ℝ × ℝ) : Prop :=
  (B.1 - A.1) * (C.1 - B.1) + (B.2 - A.2) * (C.2 - B.2) > 0

-- Theorem for collinearity
theorem collinearity_condition (m : ℝ) :
  collinear OA OB (OC m) ↔ m = 1/2 := by sorry

-- Theorem for acute angle
theorem acute_angle_condition (m : ℝ) :
  acute_angle OA OB (OC m) ↔ m ∈ Set.Ioo (-3/4) (1/2) ∪ Set.Ioi (1/2) := by sorry

end NUMINAMATH_CALUDE_collinearity_condition_acute_angle_condition_l1776_177607


namespace NUMINAMATH_CALUDE_max_value_when_m_3_solution_f_geq_0_l1776_177699

-- Define the function f(x, m)
def f (x m : ℝ) : ℝ := |x - m| - 2 * |x - 1|

-- Theorem for the maximum value when m = 3
theorem max_value_when_m_3 :
  ∃ (max : ℝ), max = 2 ∧ ∀ x, f x 3 ≤ max :=
sorry

-- Theorem for the solution of f(x) ≥ 0
theorem solution_f_geq_0 (m : ℝ) :
  (m > 1 → ∀ x, f x m ≥ 0 ↔ 2 - m ≤ x ∧ x ≤ (2 + m) / 3) ∧
  (m = 1 → ∀ x, f x m ≥ 0 ↔ x = 1) ∧
  (m < 1 → ∀ x, f x m ≥ 0 ↔ (2 + m) / 3 ≤ x ∧ x ≤ 2 - m) :=
sorry

end NUMINAMATH_CALUDE_max_value_when_m_3_solution_f_geq_0_l1776_177699


namespace NUMINAMATH_CALUDE_average_speed_calculation_l1776_177694

theorem average_speed_calculation (distance1 : ℝ) (distance2 : ℝ) (time1 : ℝ) (time2 : ℝ) 
  (h1 : distance1 = 100) 
  (h2 : distance2 = 60) 
  (h3 : time1 = 1) 
  (h4 : time2 = 1) : 
  (distance1 + distance2) / (time1 + time2) = 80 := by
  sorry

end NUMINAMATH_CALUDE_average_speed_calculation_l1776_177694


namespace NUMINAMATH_CALUDE_hyperbola_equation_l1776_177668

/-- The equation of a hyperbola with foci at (-3, 0) and (3, 0), and |MA| - |MB| = 4 -/
theorem hyperbola_equation (x y : ℝ) :
  let A : ℝ × ℝ := (-3, 0)
  let B : ℝ × ℝ := (3, 0)
  let M : ℝ × ℝ := (x, y)
  let dist (p q : ℝ × ℝ) := Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)
  (dist M A - dist M B = 4) → (x > 0) →
  (x^2 / 4 - y^2 / 5 = 1) := by
sorry

end NUMINAMATH_CALUDE_hyperbola_equation_l1776_177668


namespace NUMINAMATH_CALUDE_area_triangle_ABC_area_DEFGH_area_triangle_JKL_l1776_177655

-- Define the grid unit
def grid_unit : ℝ := 1

-- Define the dimensions of triangle ABC
def triangle_ABC_base : ℝ := 2 * grid_unit
def triangle_ABC_height : ℝ := 3 * grid_unit

-- Define the dimensions of the square for DEFGH and JKL
def square_side : ℝ := 5 * grid_unit

-- Theorem for the area of triangle ABC
theorem area_triangle_ABC : 
  (1/2) * triangle_ABC_base * triangle_ABC_height = 3 := by sorry

-- Theorem for the area of figure DEFGH
theorem area_DEFGH : 
  square_side^2 - (1/2) * triangle_ABC_base * triangle_ABC_height = 22 := by sorry

-- Theorem for the area of triangle JKL
theorem area_triangle_JKL : 
  square_side^2 - ((1/2) * triangle_ABC_base * triangle_ABC_height + 
  (1/2) * square_side * (square_side - triangle_ABC_height) + 
  (1/2) * square_side * triangle_ABC_base) = 19/2 := by sorry

end NUMINAMATH_CALUDE_area_triangle_ABC_area_DEFGH_area_triangle_JKL_l1776_177655


namespace NUMINAMATH_CALUDE_johns_umbrella_cost_l1776_177672

/-- The total cost of John's umbrellas -/
def total_cost (house_umbrellas car_umbrellas cost_per_umbrella : ℕ) : ℕ :=
  (house_umbrellas + car_umbrellas) * cost_per_umbrella

/-- Proof that John's total cost for umbrellas is $24 -/
theorem johns_umbrella_cost :
  total_cost 2 1 8 = 24 := by
  sorry

end NUMINAMATH_CALUDE_johns_umbrella_cost_l1776_177672


namespace NUMINAMATH_CALUDE_rational_number_problems_l1776_177610

theorem rational_number_problems :
  (∀ (a b : ℚ), a * b = -2 ∧ a = 1/7 → b = -14) ∧
  (∀ (x y z : ℚ), x + y + z = -5 ∧ x = 1 ∧ y = -4 → z = -2) := by sorry

end NUMINAMATH_CALUDE_rational_number_problems_l1776_177610


namespace NUMINAMATH_CALUDE_lineup_combinations_l1776_177688

def total_members : ℕ := 12
def offensive_linemen : ℕ := 5
def positions_to_fill : ℕ := 5

def choose_lineup : ℕ := offensive_linemen * (total_members - 1) * (total_members - 2) * (total_members - 3) * (total_members - 4)

theorem lineup_combinations :
  choose_lineup = 39600 := by
  sorry

end NUMINAMATH_CALUDE_lineup_combinations_l1776_177688


namespace NUMINAMATH_CALUDE_exists_valid_statement_l1776_177633

-- Define the types of people on the island
inductive PersonType
  | Knight
  | Liar
  | Normal

-- Define a statement type
structure Statement where
  content : String
  canBeMadeBy : PersonType → Prop
  truthValueKnown : Prop

-- Define the property of a valid statement
def validStatement (s : Statement) : Prop :=
  (s.canBeMadeBy PersonType.Normal) ∧
  (¬s.canBeMadeBy PersonType.Knight) ∧
  (¬s.canBeMadeBy PersonType.Liar) ∧
  (¬s.truthValueKnown)

-- Theorem: There exists a valid statement
theorem exists_valid_statement : ∃ s : Statement, validStatement s := by
  sorry

end NUMINAMATH_CALUDE_exists_valid_statement_l1776_177633


namespace NUMINAMATH_CALUDE_initial_roses_l1776_177698

theorem initial_roses (thrown_away : ℕ) (final_count : ℕ) :
  thrown_away = 33 →
  final_count = 17 →
  ∃ (initial : ℕ) (new_cut : ℕ),
    initial - thrown_away + new_cut = final_count ∧
    new_cut = thrown_away + 2 ∧
    initial = 15 := by
  sorry

end NUMINAMATH_CALUDE_initial_roses_l1776_177698


namespace NUMINAMATH_CALUDE_range_of_a_l1776_177658

theorem range_of_a (a : ℝ) : 
  (∀ x : ℝ, |x - 3| + |x - a| ≥ 3) ↔ (a ≤ 0 ∨ a ≥ 6) :=
by sorry

end NUMINAMATH_CALUDE_range_of_a_l1776_177658


namespace NUMINAMATH_CALUDE_remainder_3572_div_49_l1776_177624

theorem remainder_3572_div_49 : 3572 % 49 = 44 := by
  sorry

end NUMINAMATH_CALUDE_remainder_3572_div_49_l1776_177624


namespace NUMINAMATH_CALUDE_quadratic_real_root_condition_l1776_177690

-- Define the quadratic equation
def quadratic (b x : ℝ) : ℝ := x^2 + b*x + 25

-- Define the condition for real roots
def has_real_root (b : ℝ) : Prop := ∃ x : ℝ, quadratic b x = 0

-- Theorem statement
theorem quadratic_real_root_condition (b : ℝ) :
  has_real_root b ↔ b ≤ -10 ∨ b ≥ 10 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_real_root_condition_l1776_177690


namespace NUMINAMATH_CALUDE_original_price_correct_l1776_177622

/-- The original price of a single article before discounts and taxes -/
def original_price : ℝ := 669.99

/-- The discount rate for purchases of 2 or more articles -/
def discount_rate : ℝ := 0.24

/-- The sales tax rate -/
def sales_tax_rate : ℝ := 0.08

/-- The number of articles purchased -/
def num_articles : ℕ := 3

/-- The total cost after discount and tax -/
def total_cost : ℝ := 1649.43

/-- Theorem stating that the original price is correct given the conditions -/
theorem original_price_correct : 
  num_articles * (original_price * (1 - discount_rate)) * (1 + sales_tax_rate) = total_cost := by
  sorry

end NUMINAMATH_CALUDE_original_price_correct_l1776_177622


namespace NUMINAMATH_CALUDE_shaded_area_calculation_l1776_177609

/-- The area of the shaded region in a grid with given dimensions and an unshaded triangle -/
theorem shaded_area_calculation (grid_width grid_height triangle_base triangle_height : ℝ) 
  (hw : grid_width = 15)
  (hh : grid_height = 5)
  (hb : triangle_base = grid_width)
  (ht : triangle_height = 3) :
  grid_width * grid_height - (1/2 * triangle_base * triangle_height) = 52.5 := by
  sorry

#check shaded_area_calculation

end NUMINAMATH_CALUDE_shaded_area_calculation_l1776_177609


namespace NUMINAMATH_CALUDE_camping_trip_percentage_l1776_177636

theorem camping_trip_percentage 
  (total_students : ℕ) 
  (students_more_than_100 : ℕ) 
  (h1 : students_more_than_100 = (15 * total_students) / 100)
  (h2 : (75 * (students_more_than_100 * 100 / 25)) / 100 + students_more_than_100 = (60 * total_students) / 100) :
  (students_more_than_100 * 100 / 25) * 100 / total_students = 60 :=
sorry

end NUMINAMATH_CALUDE_camping_trip_percentage_l1776_177636


namespace NUMINAMATH_CALUDE_circle_and_lines_theorem_l1776_177618

/-- Represents a circle with center (a, b) and radius r -/
structure Circle where
  a : ℝ
  b : ℝ
  r : ℝ

/-- Represents a line y = k(x - 2) -/
structure Line where
  k : ℝ

/-- Checks if a circle satisfies the given conditions -/
def satisfiesConditions (c : Circle) : Prop :=
  c.r > 0 ∧
  2 * c.a + c.b = 0 ∧
  (2 - c.a)^2 + (-1 - c.b)^2 = c.r^2 ∧
  |c.a + c.b - 1| / Real.sqrt 2 = c.r

/-- Checks if a line divides the circle into arcs with length ratio 1:2 -/
def dividesCircle (c : Circle) (l : Line) : Prop :=
  ∃ (θ : ℝ), θ = Real.arccos ((1 - l.k * (c.a - 2) - c.b) / (c.r * Real.sqrt (1 + l.k^2))) ∧
              θ / (2 * Real.pi - θ) = 1 / 2

/-- The main theorem stating the properties of the circle and lines -/
theorem circle_and_lines_theorem (c : Circle) (l : Line) :
  satisfiesConditions c →
  dividesCircle c l →
  (c.a = 1 ∧ c.b = -2 ∧ c.r = Real.sqrt 2) ∧
  (l.k = 1 ∨ l.k = 7) := by
  sorry

end NUMINAMATH_CALUDE_circle_and_lines_theorem_l1776_177618


namespace NUMINAMATH_CALUDE_y_derivative_l1776_177691

/-- The function y in terms of x -/
def y (x : ℝ) : ℝ := (3 * x - 2) ^ 2

/-- The derivative of y with respect to x -/
def y' (x : ℝ) : ℝ := 6 * (3 * x - 2)

/-- Theorem stating that y' is the derivative of y -/
theorem y_derivative : ∀ x, deriv y x = y' x := by sorry

end NUMINAMATH_CALUDE_y_derivative_l1776_177691


namespace NUMINAMATH_CALUDE_solution_set_f_geq_3_range_of_a_l1776_177628

-- Define the function f(x)
def f (x : ℝ) : ℝ := |x - 1| + |x - 2|

-- Theorem for the solution set of f(x) ≥ 3
theorem solution_set_f_geq_3 :
  {x : ℝ | f x ≥ 3} = {x : ℝ | x ≤ 0 ∨ x ≥ 3} := by sorry

-- Theorem for the range of a
theorem range_of_a (a : ℝ) :
  (∃ x : ℝ, f x ≤ -a^2 + a + 7) → a ∈ Set.Icc (-2) 3 := by sorry


end NUMINAMATH_CALUDE_solution_set_f_geq_3_range_of_a_l1776_177628


namespace NUMINAMATH_CALUDE_derivative_f_at_pi_fourth_l1776_177603

noncomputable def f (x : ℝ) : ℝ := 2 * Real.sin x + 1

theorem derivative_f_at_pi_fourth : 
  (deriv f) (π/4) = Real.sqrt 2 := by sorry

end NUMINAMATH_CALUDE_derivative_f_at_pi_fourth_l1776_177603


namespace NUMINAMATH_CALUDE_milk_jug_problem_l1776_177604

theorem milk_jug_problem (x y : ℝ) : 
  x + y = 70 ∧ 
  0.875 * x = y + 0.125 * x → 
  x = 40 ∧ y = 30 := by
sorry

end NUMINAMATH_CALUDE_milk_jug_problem_l1776_177604


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l1776_177653

-- Define set A
def A : Set ℝ := {x | x^2 - 5*x + 6 ≤ 0}

-- Define set B
def B : Set ℝ := {x | |2*x - 1| > 3}

-- Theorem statement
theorem intersection_of_A_and_B :
  A ∩ B = {x : ℝ | 2 < x ∧ x ≤ 3} := by sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l1776_177653


namespace NUMINAMATH_CALUDE_generalized_schur_inequality_l1776_177667

theorem generalized_schur_inequality (a b c t : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  a^t * (a - b) * (a - c) + b^t * (b - c) * (b - a) + c^t * (c - a) * (c - b) ≥ 0 := by
  sorry

end NUMINAMATH_CALUDE_generalized_schur_inequality_l1776_177667


namespace NUMINAMATH_CALUDE_equivalent_discount_l1776_177652

theorem equivalent_discount (original_price : ℝ) (discount1 : ℝ) (discount2 : ℝ) 
  (h1 : original_price = 50)
  (h2 : discount1 = 0.3)
  (h3 : discount2 = 0.2) : 
  original_price * (1 - discount1) * (1 - discount2) = original_price * (1 - 0.44) :=
by sorry

end NUMINAMATH_CALUDE_equivalent_discount_l1776_177652


namespace NUMINAMATH_CALUDE_product_not_divisible_by_prime_l1776_177638

theorem product_not_divisible_by_prime (p a b : ℕ) : 
  Prime p → a > 0 → b > 0 → a < p → b < p → ¬(p ∣ (a * b)) := by
  sorry

end NUMINAMATH_CALUDE_product_not_divisible_by_prime_l1776_177638


namespace NUMINAMATH_CALUDE_unique_solution_sin_system_l1776_177671

theorem unique_solution_sin_system (a b c d : Real) 
  (h_sum : a + b + c + d = Real.pi) :
  ∃! (x y z w : Real),
    x = Real.sin (a + b) ∧
    y = Real.sin (b + c) ∧
    z = Real.sin (c + d) ∧
    w = Real.sin (d + a) :=
by sorry

end NUMINAMATH_CALUDE_unique_solution_sin_system_l1776_177671


namespace NUMINAMATH_CALUDE_hours_to_seconds_l1776_177675

-- Define the conversion factors
def minutes_per_hour : ℕ := 60
def seconds_per_minute : ℕ := 60

-- Define the problem
def hours : ℚ := 3.5

-- Theorem to prove
theorem hours_to_seconds : 
  (hours * minutes_per_hour * seconds_per_minute : ℚ) = 12600 := by
  sorry

end NUMINAMATH_CALUDE_hours_to_seconds_l1776_177675


namespace NUMINAMATH_CALUDE_gcf_of_lcms_l1776_177620

theorem gcf_of_lcms : Nat.gcd (Nat.lcm 9 15) (Nat.lcm 10 21) = 15 := by
  sorry

end NUMINAMATH_CALUDE_gcf_of_lcms_l1776_177620


namespace NUMINAMATH_CALUDE_digit_sum_puzzle_l1776_177683

theorem digit_sum_puzzle (c d : ℕ) : 
  c < 10 → d < 10 → 
  (40 + c) * (10 * d + 5) = 215 →
  (40 + c) * 5 = 20 →
  (40 + c) * d * 10 = 180 →
  c + d = 5 := by
sorry

end NUMINAMATH_CALUDE_digit_sum_puzzle_l1776_177683


namespace NUMINAMATH_CALUDE_simplify_expression_l1776_177660

theorem simplify_expression (b : ℝ) : 3*b*(3*b^2 + 2*b) - 2*b^2 = 9*b^3 + 4*b^2 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l1776_177660


namespace NUMINAMATH_CALUDE_bryan_mineral_samples_l1776_177613

/-- The number of mineral samples Bryan has left after rearrangement -/
def samples_left (initial_samples_per_shelf : ℕ) (num_shelves : ℕ) (removed_per_shelf : ℕ) : ℕ :=
  (initial_samples_per_shelf - removed_per_shelf) * num_shelves

/-- Theorem stating the number of samples left after Bryan's rearrangement -/
theorem bryan_mineral_samples :
  samples_left 128 13 2 = 1638 := by
  sorry

end NUMINAMATH_CALUDE_bryan_mineral_samples_l1776_177613


namespace NUMINAMATH_CALUDE_symmetry_composition_iff_intersection_l1776_177682

-- Define a type for lines in a plane
structure Line where
  -- Add necessary properties for a line

-- Define a type for points in a plane
structure Point where
  -- Add necessary properties for a point

-- Define a symmetry operation
def symmetry (l : Line) : Point → Point := sorry

-- Define composition of symmetries
def compose_symmetries (a b c : Line) : Point → Point :=
  symmetry c ∘ symmetry b ∘ symmetry a

-- Define a predicate for three lines intersecting at a single point
def intersect_at_single_point (a b c : Line) : Prop := sorry

-- The main theorem
theorem symmetry_composition_iff_intersection (a b c : Line) :
  (∃ l : Line, compose_symmetries a b c = symmetry l) ↔ intersect_at_single_point a b c := by
  sorry

end NUMINAMATH_CALUDE_symmetry_composition_iff_intersection_l1776_177682


namespace NUMINAMATH_CALUDE_paper_fold_ratio_is_four_fifths_l1776_177681

/-- Represents the dimensions and folding of a rectangular piece of paper. -/
structure PaperFold where
  length : ℝ
  width : ℝ
  fold_ratio : ℝ
  division_parts : ℕ

/-- Calculates the ratio of the new visible area to the original area after folding. -/
def visible_area_ratio (paper : PaperFold) : ℝ :=
  -- Implementation details would go here
  sorry

/-- Theorem stating that for a specific paper folding scenario, the visible area ratio is 8/10. -/
theorem paper_fold_ratio_is_four_fifths :
  let paper : PaperFold := {
    length := 5,
    width := 2,
    fold_ratio := 1/2,
    division_parts := 3
  }
  visible_area_ratio paper = 8/10 := by
  sorry

end NUMINAMATH_CALUDE_paper_fold_ratio_is_four_fifths_l1776_177681


namespace NUMINAMATH_CALUDE_product_ab_l1776_177677

theorem product_ab (a b : ℕ) (h1 : a / 3 = 16) (h2 : b = a - 1) : a * b = 2256 := by
  sorry

end NUMINAMATH_CALUDE_product_ab_l1776_177677


namespace NUMINAMATH_CALUDE_circumscribed_circle_equation_l1776_177619

/-- The equation of a line in the form y = mx + b -/
structure Line where
  m : ℝ
  b : ℝ

/-- A triangle defined by three lines -/
structure Triangle where
  side1 : Line
  side2 : Line
  side3 : Line

/-- The equation of a circle in the form (x - h)^2 + (y - k)^2 = r^2 -/
structure Circle where
  h : ℝ
  k : ℝ
  r : ℝ

/-- The circumscribed circle of a triangle -/
def circumscribedCircle (t : Triangle) : Circle := sorry

/-- Theorem: The circumscribed circle of the given triangle has the equation (x - 2)^2 + (y - 2)^2 = 25 -/
theorem circumscribed_circle_equation (t : Triangle) 
  (h1 : t.side1 = ⟨1, 1⟩) 
  (h2 : t.side2 = ⟨-1/2, -2⟩) 
  (h3 : t.side3 = ⟨3, -9⟩) : 
  circumscribedCircle t = ⟨2, 2, 5⟩ := by sorry

end NUMINAMATH_CALUDE_circumscribed_circle_equation_l1776_177619


namespace NUMINAMATH_CALUDE_leftHandedLikeMusicalCount_l1776_177611

/-- Represents the Drama Club -/
structure DramaClub where
  total : Nat
  leftHanded : Nat
  likeMusical : Nat
  rightHandedDislike : Nat

/-- The number of left-handed people who like musical theater in the Drama Club -/
def leftHandedLikeMusical (club : DramaClub) : Nat :=
  club.leftHanded + club.likeMusical - (club.total - club.rightHandedDislike)

/-- Theorem stating the number of left-handed musical theater lovers in the specific Drama Club -/
theorem leftHandedLikeMusicalCount : leftHandedLikeMusical { 
  total := 25,
  leftHanded := 10,
  likeMusical := 18,
  rightHandedDislike := 3
} = 6 := by sorry

end NUMINAMATH_CALUDE_leftHandedLikeMusicalCount_l1776_177611


namespace NUMINAMATH_CALUDE_parallelogram_area_l1776_177644

/-- The area of a parallelogram is the product of two adjacent sides and the sine of the angle between them. -/
theorem parallelogram_area (a b : ℝ) (γ : ℝ) (ha : 0 < a) (hb : 0 < b) (hγ : 0 < γ ∧ γ < π) :
  ∃ (S : ℝ), S = a * b * Real.sin γ ∧ S > 0 := by
  sorry

end NUMINAMATH_CALUDE_parallelogram_area_l1776_177644


namespace NUMINAMATH_CALUDE_sequence_formula_l1776_177602

def S (n : ℕ+) (a : ℕ+ → ℚ) : ℚ := 2 * n.val - a n

theorem sequence_formula (a : ℕ+ → ℚ) 
  (h : ∀ n : ℕ+, S n a = 2 * n.val - a n) : 
  ∀ n : ℕ+, a n = (2^n.val - 1) / 2^(n.val - 1) := by
sorry

end NUMINAMATH_CALUDE_sequence_formula_l1776_177602


namespace NUMINAMATH_CALUDE_equation_transformation_l1776_177664

theorem equation_transformation (x y : ℚ) : 
  5 * x - 6 * y = 4 → y = (5/6) * x - 2/3 := by
  sorry

end NUMINAMATH_CALUDE_equation_transformation_l1776_177664


namespace NUMINAMATH_CALUDE_cube_split_59_l1776_177605

/-- The number of odd terms in the split of m³ -/
def split_terms (m : ℕ) : ℕ := (m + 2) * (m - 1) / 2

/-- The nth odd number starting from 3 -/
def nth_odd (n : ℕ) : ℕ := 2 * n + 1

theorem cube_split_59 (m : ℕ) (h1 : m > 1) :
  (∃ k, k ≤ split_terms m ∧ nth_odd k = 59) → m = 8 := by
  sorry

end NUMINAMATH_CALUDE_cube_split_59_l1776_177605


namespace NUMINAMATH_CALUDE_abs_diff_inequality_l1776_177696

theorem abs_diff_inequality (x : ℝ) : |x + 3| - |x - 1| > 0 ↔ x > -1 := by
  sorry

end NUMINAMATH_CALUDE_abs_diff_inequality_l1776_177696


namespace NUMINAMATH_CALUDE_marcos_dad_strawberries_l1776_177680

theorem marcos_dad_strawberries (marco_weight : ℕ) (total_weight : ℕ) 
  (h1 : marco_weight = 15)
  (h2 : total_weight = 37) :
  total_weight - marco_weight = 22 := by
  sorry

end NUMINAMATH_CALUDE_marcos_dad_strawberries_l1776_177680


namespace NUMINAMATH_CALUDE_unique_solution_l1776_177637

theorem unique_solution (x y z : ℝ) : 
  x + 3 * y = 33 ∧ 
  y = 10 ∧ 
  2 * x - y + z = 15 → 
  x = 3 ∧ y = 10 ∧ z = 19 := by
sorry

end NUMINAMATH_CALUDE_unique_solution_l1776_177637


namespace NUMINAMATH_CALUDE_equation_one_solution_equation_two_no_solution_l1776_177685

/-- The first equation has a unique solution x = 4 -/
theorem equation_one_solution :
  ∃! x : ℝ, (5 / (x + 1) = 1 / (x - 3)) ∧ (x + 1 ≠ 0) ∧ (x - 3 ≠ 0) :=
sorry

/-- The second equation has no solution -/
theorem equation_two_no_solution :
  ¬∃ x : ℝ, ((3 - x) / (x - 4) = 1 / (4 - x) - 2) ∧ (x - 4 ≠ 0) ∧ (4 - x ≠ 0) :=
sorry

end NUMINAMATH_CALUDE_equation_one_solution_equation_two_no_solution_l1776_177685


namespace NUMINAMATH_CALUDE_trains_meet_at_11am_l1776_177608

/-- The distance between stations A and B in kilometers -/
def distance_between_stations : ℝ := 155

/-- The speed of the first train in km/h -/
def speed_train1 : ℝ := 20

/-- The speed of the second train in km/h -/
def speed_train2 : ℝ := 25

/-- The time difference between the trains' departures in hours -/
def time_difference : ℝ := 1

/-- The meeting time of the trains after the second train's departure -/
def meeting_time : ℝ := 3

theorem trains_meet_at_11am :
  speed_train1 * (time_difference + meeting_time) +
  speed_train2 * meeting_time = distance_between_stations :=
sorry

end NUMINAMATH_CALUDE_trains_meet_at_11am_l1776_177608


namespace NUMINAMATH_CALUDE_f_increasing_f_odd_range_l1776_177626

noncomputable section

-- Define the function f(x)
def f (a : ℝ) (x : ℝ) : ℝ := a - 1 / (2^x + 1)

-- Theorem 1: f(x) is an increasing function on ℝ
theorem f_increasing (a : ℝ) : 
  ∀ x₁ x₂ : ℝ, x₁ < x₂ → f a x₁ < f a x₂ :=
sorry

-- Theorem 2: When f(x) is an odd function, its range on [-1, 2] is [-1/6, 3/10]
theorem f_odd_range (a : ℝ) 
  (h_odd : ∀ x : ℝ, f a (-x) = -(f a x)) : 
  Set.range (fun x => f a x) ∩ Set.Icc (-1 : ℝ) 2 = Set.Icc (-1/6 : ℝ) (3/10) :=
sorry

end

end NUMINAMATH_CALUDE_f_increasing_f_odd_range_l1776_177626


namespace NUMINAMATH_CALUDE_p_is_power_of_two_l1776_177632

def is_power_of_two (x : ℕ) : Prop := ∃ k : ℕ, x = 2^k

theorem p_is_power_of_two (p : ℕ) (h1 : p > 2) (h2 : ∃! d : ℕ, Odd d ∧ (32 * p) % d = 0) :
  is_power_of_two p := by
sorry

end NUMINAMATH_CALUDE_p_is_power_of_two_l1776_177632


namespace NUMINAMATH_CALUDE_bread_price_calculation_bread_price_proof_l1776_177650

theorem bread_price_calculation (initial_price : ℝ) 
  (thursday_increase : ℝ) (saturday_discount : ℝ) : ℝ :=
  let thursday_price := initial_price * (1 + thursday_increase)
  let saturday_price := thursday_price * (1 - saturday_discount)
  saturday_price

theorem bread_price_proof :
  bread_price_calculation 50 0.2 0.15 = 51 := by
  sorry

end NUMINAMATH_CALUDE_bread_price_calculation_bread_price_proof_l1776_177650


namespace NUMINAMATH_CALUDE_area_T_prime_l1776_177693

/-- A transformation matrix -/
def A : Matrix (Fin 2) (Fin 2) ℝ := !![3, 4; -2, 5]

/-- The area of the original region T -/
def area_T : ℝ := 9

/-- The theorem stating the area of the transformed region T' -/
theorem area_T_prime : 
  let det_A := Matrix.det A
  area_T * det_A = 207 := by sorry

end NUMINAMATH_CALUDE_area_T_prime_l1776_177693


namespace NUMINAMATH_CALUDE_impossible_to_cover_modified_chessboard_l1776_177687

/-- Represents a chessboard with some squares removed -/
structure ModifiedChessboard where
  size : Nat
  removed : Finset (Nat × Nat)

/-- Represents a domino that covers two squares -/
structure Domino where
  square1 : Nat × Nat
  square2 : Nat × Nat

/-- Checks if a given set of dominos covers the modified chessboard -/
def covers (board : ModifiedChessboard) (dominos : Finset Domino) : Prop :=
  sorry

/-- The color of a square on a chessboard (assuming top-left is white) -/
def squareColor (pos : Nat × Nat) : Bool :=
  (pos.1 + pos.2) % 2 == 0

theorem impossible_to_cover_modified_chessboard :
  ∀ (dominos : Finset Domino),
    let board := ModifiedChessboard.mk 8 {(0, 0), (7, 7)}
    ¬ covers board dominos := by
  sorry

end NUMINAMATH_CALUDE_impossible_to_cover_modified_chessboard_l1776_177687


namespace NUMINAMATH_CALUDE_song_listens_after_three_months_l1776_177646

/-- Calculates the total listens for a song that doubles in popularity each month -/
def totalListens (initialListens : ℕ) (months : ℕ) : ℕ :=
  let doublingSequence := List.range months |>.map (fun i => initialListens * 2^(i + 1))
  initialListens + doublingSequence.sum

/-- Theorem: The total listens after 3 months of doubling is 900,000 given 60,000 initial listens -/
theorem song_listens_after_three_months :
  totalListens 60000 3 = 900000 := by
  sorry

end NUMINAMATH_CALUDE_song_listens_after_three_months_l1776_177646


namespace NUMINAMATH_CALUDE_isabel_homework_problems_l1776_177657

/-- The total number of problems Isabel has to complete -/
def total_problems (math_pages reading_pages science_pages history_pages : ℕ)
  (math_problems_per_page reading_problems_per_page : ℕ)
  (science_problems_per_page history_problems : ℕ) : ℕ :=
  math_pages * math_problems_per_page +
  reading_pages * reading_problems_per_page +
  science_pages * science_problems_per_page +
  history_pages * history_problems

/-- Theorem stating that Isabel has to complete 61 problems in total -/
theorem isabel_homework_problems :
  total_problems 2 4 3 1 5 5 7 10 = 61 := by
  sorry

end NUMINAMATH_CALUDE_isabel_homework_problems_l1776_177657


namespace NUMINAMATH_CALUDE_people_to_left_of_kolya_l1776_177625

/-- Represents a person in the line -/
structure Person where
  name : String

/-- Represents the arrangement of people in a line -/
structure Arrangement where
  people : List Person
  kolya_index : Nat
  sasha_index : Nat

/-- The number of people to the right of a person at a given index -/
def peopleToRight (arr : Arrangement) (index : Nat) : Nat :=
  arr.people.length - index - 1

/-- The number of people to the left of a person at a given index -/
def peopleToLeft (arr : Arrangement) (index : Nat) : Nat :=
  index

theorem people_to_left_of_kolya (arr : Arrangement) 
  (h1 : peopleToRight arr arr.kolya_index = 12)
  (h2 : peopleToLeft arr arr.sasha_index = 20)
  (h3 : peopleToRight arr arr.sasha_index = 8) :
  peopleToLeft arr arr.kolya_index = 16 := by
  sorry

end NUMINAMATH_CALUDE_people_to_left_of_kolya_l1776_177625


namespace NUMINAMATH_CALUDE_symmetric_axis_of_quadratic_function_l1776_177621

/-- The symmetric axis of a quadratic function -/
def symmetric_axis (f : ℝ → ℝ) : ℝ := sorry

/-- A quadratic function in factored form -/
def quadratic_function (a b : ℝ) : ℝ → ℝ := λ x ↦ (x - a) * (x + b)

theorem symmetric_axis_of_quadratic_function :
  ∀ (f : ℝ → ℝ), f = quadratic_function 3 5 →
  symmetric_axis f = -1 := by sorry

end NUMINAMATH_CALUDE_symmetric_axis_of_quadratic_function_l1776_177621


namespace NUMINAMATH_CALUDE_geometry_book_pages_difference_l1776_177686

/-- Given that a new edition of a Geometry book has 450 pages and the old edition has 340 pages,
    prove that the new edition has 230 pages less than twice the number of pages in the old edition. -/
theorem geometry_book_pages_difference (new_edition : ℕ) (old_edition : ℕ)
  (h1 : new_edition = 450)
  (h2 : old_edition = 340) :
  2 * old_edition - new_edition = 230 := by
  sorry

end NUMINAMATH_CALUDE_geometry_book_pages_difference_l1776_177686


namespace NUMINAMATH_CALUDE_rectangular_prism_edge_sum_l1776_177689

theorem rectangular_prism_edge_sum (l w h : ℝ) : 
  l * w * h = 8 →                   -- Volume condition
  2 * (l * w + w * h + h * l) = 32 → -- Surface area condition
  ∃ q : ℝ, l = 2 / q ∧ w = 2 ∧ h = 2 * q → -- Geometric progression condition
  4 * (l + w + h) = 28 :=           -- Conclusion: sum of edge lengths
by sorry

end NUMINAMATH_CALUDE_rectangular_prism_edge_sum_l1776_177689


namespace NUMINAMATH_CALUDE_range_of_a_l1776_177656

/-- A function satisfying the given properties -/
def special_function (f : ℝ → ℝ) : Prop :=
  (∀ x, f x + f (-x) = x^2) ∧
  (∀ x ≥ 0, deriv f x - x - 1 < 0)

/-- The main theorem -/
theorem range_of_a (f : ℝ → ℝ) (h : special_function f) :
  ∀ a, (f (2 - a) ≥ f a + 4 - 4*a) → a ≥ 1 :=
by sorry

end NUMINAMATH_CALUDE_range_of_a_l1776_177656


namespace NUMINAMATH_CALUDE_babysitting_age_ratio_l1776_177647

theorem babysitting_age_ratio : 
  ∀ (jane_start_age jane_current_age jane_stop_years_ago oldest_babysat_current_age : ℕ),
    jane_start_age = 16 →
    jane_current_age = 32 →
    jane_stop_years_ago = 10 →
    oldest_babysat_current_age = 24 →
    ∃ (child_age jane_age : ℕ),
      child_age = oldest_babysat_current_age - jane_stop_years_ago ∧
      jane_age = jane_current_age - jane_stop_years_ago ∧
      child_age * 11 = jane_age * 7 := by
  sorry

end NUMINAMATH_CALUDE_babysitting_age_ratio_l1776_177647


namespace NUMINAMATH_CALUDE_parallel_vectors_x_value_l1776_177623

/-- Two vectors are parallel if one is a scalar multiple of the other -/
def parallel (a b : ℝ × ℝ) : Prop :=
  ∃ k : ℝ, b = (k * a.1, k * a.2)

theorem parallel_vectors_x_value :
  let a : ℝ × ℝ := (2, 1)
  let b : ℝ × ℝ := (4, x + 1)
  parallel a b → x = 1 :=
by
  sorry

end NUMINAMATH_CALUDE_parallel_vectors_x_value_l1776_177623


namespace NUMINAMATH_CALUDE_four_common_tangents_l1776_177639

-- Define the circle type
structure Circle where
  radius : ℝ
  center : ℝ × ℝ

-- Define the function to count common tangents
def countCommonTangents (c1 c2 : Circle) : ℕ := sorry

-- Theorem statement
theorem four_common_tangents (c1 c2 : Circle) 
  (h1 : c1.radius = 3)
  (h2 : c2.radius = 5)
  (h3 : Real.sqrt ((c1.center.1 - c2.center.1)^2 + (c1.center.2 - c2.center.2)^2) = 10) :
  countCommonTangents c1 c2 = 4 := by sorry

end NUMINAMATH_CALUDE_four_common_tangents_l1776_177639


namespace NUMINAMATH_CALUDE_circle_area_ratio_l1776_177629

/-- Given two circles X and Y where an arc of 90° on circle X has the same length as an arc of 60° on circle Y, 
    the ratio of the area of circle X to the area of circle Y is 9/4 -/
theorem circle_area_ratio (X Y : Real) (hX : X > 0) (hY : Y > 0) 
  (h : (π / 2) * X = (π / 3) * Y) : 
  (π * X^2) / (π * Y^2) = 9 / 4 := by
  sorry

end NUMINAMATH_CALUDE_circle_area_ratio_l1776_177629


namespace NUMINAMATH_CALUDE_problem_1_problem_2_l1776_177612

-- Problem 1
theorem problem_1 : 
  Real.sqrt 12 + |(-4)| - (2003 - Real.pi)^0 - 2 * Real.cos (30 * π / 180) = Real.sqrt 3 + 3 := by
  sorry

-- Problem 2
theorem problem_2 (a : ℤ) (h1 : 0 < a) (h2 : a < 4) (h3 : a ≠ 2) : 
  (a + 2 - 5 / (a - 2)) / ((3 - a) / (2 * a - 4)) = -2 * a - 6 := by
  sorry

end NUMINAMATH_CALUDE_problem_1_problem_2_l1776_177612


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l1776_177670

def A : Set ℝ := {x : ℝ | 0 ≤ x ∧ x ≤ 2}
def B : Set ℝ := {x : ℝ | -1 < x ∧ x ≤ 1}

theorem intersection_of_A_and_B :
  A ∩ B = {x : ℝ | 0 ≤ x ∧ x ≤ 1} := by sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l1776_177670


namespace NUMINAMATH_CALUDE_problem_solution_l1776_177684

theorem problem_solution (a b m n : ℚ) 
  (ha_neg : a < 0) 
  (ha_abs : |a| = 7/4)
  (hb_recip : 1/b = -3/2)
  (hmn_opp : m = -n) :
  4*a / b + 3*(m + n) = 21/2 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l1776_177684


namespace NUMINAMATH_CALUDE_lunks_needed_for_twenty_apples_l1776_177679

/-- Represents the number of lunks needed to purchase a given number of apples,
    given the exchange rates between lunks, kunks, and apples. -/
def lunks_for_apples (lunks_per_kunk : ℚ) (kunks_per_apple : ℚ) (num_apples : ℕ) : ℚ :=
  num_apples * kunks_per_apple * lunks_per_kunk

/-- Theorem stating that 21 lunks are needed to purchase 20 apples,
    given the specified exchange rates. -/
theorem lunks_needed_for_twenty_apples :
  lunks_for_apples (7/4) (3/5) 20 = 21 := by
  sorry

end NUMINAMATH_CALUDE_lunks_needed_for_twenty_apples_l1776_177679


namespace NUMINAMATH_CALUDE_bus_stop_time_l1776_177674

/-- Calculates the time a bus stops per hour given its speeds with and without stoppages -/
theorem bus_stop_time (speed_without_stops speed_with_stops : ℝ) : 
  speed_without_stops = 60 → speed_with_stops = 50 → 
  (60 - (60 * speed_with_stops / speed_without_stops)) = 10 := by
  sorry

#check bus_stop_time

end NUMINAMATH_CALUDE_bus_stop_time_l1776_177674


namespace NUMINAMATH_CALUDE_horner_method_v3_l1776_177643

-- Define the polynomial f(x) = x^5 + 2x^3 + 3x^2 + x + 1
def f (x : ℝ) : ℝ := x^5 + 2*x^3 + 3*x^2 + x + 1

-- Define Horner's method for this specific polynomial
def horner (x : ℝ) : ℝ := ((((x + 0)*x + 2)*x + 3)*x + 1)*x + 1

-- Define v_3 as the result of Horner's method at x = 3
def v_3 : ℝ := horner 3

-- Theorem statement
theorem horner_method_v3 : v_3 = 36 := by sorry

end NUMINAMATH_CALUDE_horner_method_v3_l1776_177643


namespace NUMINAMATH_CALUDE_circle_ratio_l1776_177641

theorem circle_ratio (R r a b : ℝ) (h1 : 0 < r) (h2 : r < R) (h3 : 0 < a) (h4 : 0 < b) 
  (h5 : π * R^2 = (b/a) * (π * R^2 - π * r^2)) : 
  R / r = (b / (a - b))^(1/2) := by
  sorry

end NUMINAMATH_CALUDE_circle_ratio_l1776_177641


namespace NUMINAMATH_CALUDE_sum_of_digits_next_l1776_177654

def S (n : ℕ+) : ℕ := sorry

theorem sum_of_digits_next (n : ℕ+) (h : S n = 876) : S (n + 1) = 877 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_digits_next_l1776_177654


namespace NUMINAMATH_CALUDE_triangle_angle_measure_l1776_177673

theorem triangle_angle_measure (A B C : ℝ) (a b c : ℝ) :
  0 < A ∧ 0 < B ∧ 0 < C ∧ 
  A + B + C = π ∧
  0 < a ∧ 0 < b ∧ 0 < c ∧
  a / Real.cos A = b / (2 * Real.cos B) ∧
  a / Real.cos A = c / (3 * Real.cos C) →
  A = π / 4 := by
sorry

end NUMINAMATH_CALUDE_triangle_angle_measure_l1776_177673


namespace NUMINAMATH_CALUDE_largest_integer_negative_quadratic_seven_satisfies_inequality_eight_does_not_satisfy_inequality_l1776_177676

theorem largest_integer_negative_quadratic :
  ∀ n : ℤ, n^2 - 13*n + 40 < 0 → n ≤ 7 :=
by
  sorry

theorem seven_satisfies_inequality :
  7^2 - 13*7 + 40 < 0 :=
by
  sorry

theorem eight_does_not_satisfy_inequality :
  8^2 - 13*8 + 40 ≥ 0 :=
by
  sorry

end NUMINAMATH_CALUDE_largest_integer_negative_quadratic_seven_satisfies_inequality_eight_does_not_satisfy_inequality_l1776_177676


namespace NUMINAMATH_CALUDE_jackson_collection_l1776_177645

/-- Calculates the total number of souvenirs collected by Jackson -/
def total_souvenirs (hermit_crabs : ℕ) (shells_per_crab : ℕ) (starfish_per_shell : ℕ) (dollars_per_starfish : ℕ) : ℕ :=
  let spiral_shells := hermit_crabs * shells_per_crab
  let starfish := spiral_shells * starfish_per_shell
  let sand_dollars := starfish * dollars_per_starfish
  hermit_crabs + spiral_shells + starfish + sand_dollars

/-- Theorem stating that Jackson's collection totals 3672 souvenirs -/
theorem jackson_collection :
  total_souvenirs 72 5 3 2 = 3672 := by
  sorry


end NUMINAMATH_CALUDE_jackson_collection_l1776_177645


namespace NUMINAMATH_CALUDE_boat_speed_ratio_l1776_177697

theorem boat_speed_ratio (boat_speed : ℝ) (current_speed : ℝ) 
  (h1 : boat_speed = 15)
  (h2 : current_speed = 5)
  (h3 : boat_speed > current_speed) :
  let downstream_speed := boat_speed + current_speed
  let upstream_speed := boat_speed - current_speed
  let avg_speed := 2 / (1 / downstream_speed + 1 / upstream_speed)
  avg_speed / boat_speed = 8 / 9 := by
sorry

end NUMINAMATH_CALUDE_boat_speed_ratio_l1776_177697


namespace NUMINAMATH_CALUDE_third_coaster_speed_l1776_177601

/-- Theorem: Given 5 rollercoasters with specified speeds and average, prove the speed of the third coaster -/
theorem third_coaster_speed 
  (v1 v2 v3 v4 v5 : ℝ) 
  (h1 : v1 = 50)
  (h2 : v2 = 62)
  (h4 : v4 = 70)
  (h5 : v5 = 40)
  (h_avg : (v1 + v2 + v3 + v4 + v5) / 5 = 59) :
  v3 = 73 := by
sorry

end NUMINAMATH_CALUDE_third_coaster_speed_l1776_177601


namespace NUMINAMATH_CALUDE_greatest_common_divisor_360_90_under_60_l1776_177635

theorem greatest_common_divisor_360_90_under_60 : 
  ∃ (n : ℕ), n = 30 ∧ 
  n ∣ 360 ∧ 
  n < 60 ∧ 
  n ∣ 90 ∧
  ∀ (m : ℕ), m ∣ 360 → m < 60 → m ∣ 90 → m ≤ n :=
by sorry

end NUMINAMATH_CALUDE_greatest_common_divisor_360_90_under_60_l1776_177635


namespace NUMINAMATH_CALUDE_exceed_permutations_l1776_177627

def factorial (n : ℕ) : ℕ := (List.range n).foldl (· * ·) 1

def word_length : ℕ := 6
def repeated_letter_count : ℕ := 3

theorem exceed_permutations :
  factorial word_length / factorial repeated_letter_count = 120 := by
  sorry

end NUMINAMATH_CALUDE_exceed_permutations_l1776_177627


namespace NUMINAMATH_CALUDE_quadratic_symmetry_l1776_177669

/-- A quadratic function with axis of symmetry at x = 9.5 and p(1) = 2 -/
def p (a b c : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x + c

theorem quadratic_symmetry (a b c : ℝ) :
  (∀ x : ℝ, p a b c x = p a b c (19 - x)) →  -- symmetry about x = 9.5
  p a b c 1 = 2 →                            -- p(1) = 2
  p a b c 18 = 2 :=                          -- p(18) = 2
by
  sorry

#check quadratic_symmetry

end NUMINAMATH_CALUDE_quadratic_symmetry_l1776_177669


namespace NUMINAMATH_CALUDE_ac_plus_bd_value_l1776_177616

theorem ac_plus_bd_value (a b c d : ℝ) 
  (eq1 : a + b + c = 10)
  (eq2 : a + b + d = -6)
  (eq3 : a + c + d = 0)
  (eq4 : b + c + d = 15) :
  a * c + b * d = -1171 / 9 := by
  sorry

end NUMINAMATH_CALUDE_ac_plus_bd_value_l1776_177616
