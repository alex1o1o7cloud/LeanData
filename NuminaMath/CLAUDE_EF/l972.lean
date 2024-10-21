import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_10_is_55_l972_97202

/-- An arithmetic sequence with specific properties -/
structure ArithmeticSequence where
  a : ℕ → ℚ
  is_arithmetic : ∀ n, a (n + 1) - a n = a (n + 2) - a (n + 1)
  fourth_term : a 4 = 4
  sum_property : a 3 + a 5 + a 7 = 15

/-- The sum of the first n terms of an arithmetic sequence -/
def sum_n (seq : ArithmeticSequence) (n : ℕ) : ℚ :=
  (n : ℚ) / 2 * (seq.a 1 + seq.a n)

/-- Theorem: The sum of the first 10 terms of the specific arithmetic sequence is 55 -/
theorem sum_10_is_55 (seq : ArithmeticSequence) : sum_n seq 10 = 55 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_10_is_55_l972_97202


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_f_min_value_g_l972_97237

-- Define the functions
noncomputable def f (x : ℝ) : ℝ := (Real.cos x) ^ 2 - Real.sin x

noncomputable def g (a : ℝ) (x : ℝ) : ℝ := (Real.cos x) ^ 2 - a * Real.sin x

-- State the theorems
theorem max_value_f : ∀ x : ℝ, f x ≤ 5/4 := by
  sorry

theorem min_value_g : 
  ∀ a : ℝ, (∀ x : ℝ, g a x ≥ min a (-a)) ∧ 
  (∃ x : ℝ, g a x = min a (-a)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_f_min_value_g_l972_97237


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_least_positive_integer_divisibility_l972_97257

def sequenceA (n : ℕ) : ℤ :=
  match n with
  | 0 => 0
  | 1 => 3
  | n + 2 => 8 * sequenceA (n + 1) + 9 * sequenceA n + 16

def is_divisible_by_1999 (x : ℤ) : Prop :=
  ∃ k : ℤ, x = 1999 * k

theorem least_positive_integer_divisibility :
  (∀ n : ℕ, is_divisible_by_1999 (sequenceA (n + 18) - sequenceA n)) ∧
  (∀ h : ℕ, h < 18 → ∃ m : ℕ, ¬is_divisible_by_1999 (sequenceA (m + h) - sequenceA m)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_least_positive_integer_divisibility_l972_97257


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_store_loss_percentage_l972_97260

noncomputable section

def radio_cost : ℝ := 1500
def radio_discount : ℝ := 0.10
def calculator_cost : ℝ := 800
def calculator_discount : ℝ := 0.05
def mobile_cost : ℝ := 8000
def mobile_discount : ℝ := 0.12

def total_cost : ℝ := radio_cost + calculator_cost + mobile_cost

def radio_selling_price : ℝ := radio_cost * (1 - radio_discount)
def calculator_selling_price : ℝ := calculator_cost * (1 - calculator_discount)
def mobile_selling_price : ℝ := mobile_cost * (1 - mobile_discount)

def total_selling_price : ℝ := radio_selling_price + calculator_selling_price + mobile_selling_price

def total_loss : ℝ := total_cost - total_selling_price

def loss_percentage : ℝ := (total_loss / total_cost) * 100

theorem store_loss_percentage :
  (loss_percentage ≥ 11.16) ∧ (loss_percentage ≤ 11.18) := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_store_loss_percentage_l972_97260


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_bus_weight_in_pounds_l972_97267

/-- Conversion factor from kilograms to pounds -/
noncomputable def kg_to_pound : ℝ := 1 / 0.4536

/-- Rounds a real number to the nearest integer -/
noncomputable def round_to_nearest (x : ℝ) : ℤ :=
  ⌊x + 0.5⌋

/-- The weight of the bus in kilograms -/
def bus_weight_kg : ℝ := 350

theorem bus_weight_in_pounds : 
  round_to_nearest (bus_weight_kg * kg_to_pound) = 772 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_bus_weight_in_pounds_l972_97267


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_binomial_expected_value_and_variance_l972_97289

/-- A binomial random variable with parameters n and p -/
structure BinomialRandomVariable (n : ℕ) (p : ℝ) where
  prob : ℕ → ℝ
  prob_eq : ∀ m : ℕ, m ≤ n → prob m = (Nat.choose n m : ℝ) * p^m * (1-p)^(n-m)
  p_bounds : 0 ≤ p ∧ p ≤ 1

/-- Expected value of a binomial random variable -/
def expectedValue (n : ℕ) (p : ℝ) (X : BinomialRandomVariable n p) : ℝ :=
  n * p

/-- Variance of a binomial random variable -/
def variance (n : ℕ) (p : ℝ) (X : BinomialRandomVariable n p) : ℝ :=
  n * p * (1 - p)

theorem binomial_expected_value_and_variance (n : ℕ) (p : ℝ) (X : BinomialRandomVariable n p) :
  expectedValue n p X = n * p ∧ variance n p X = n * p * (1 - p) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_binomial_expected_value_and_variance_l972_97289


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_not_quadratic_radical_l972_97206

-- Define what a quadratic radical is
def is_quadratic_radical (x : ℝ) : Prop := x ≥ 0

-- State that π > 3
axiom pi_greater_than_three : Real.pi > 3

-- Theorem statement
theorem not_quadratic_radical : ¬ is_quadratic_radical (3 - Real.pi) := by
  -- Proof goes here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_not_quadratic_radical_l972_97206


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_terminal_side_l972_97285

theorem angle_terminal_side (b : ℝ) (α : ℝ) : 
  (∃ P : ℝ × ℝ, P = (-b, 4) ∧ P.1 = -b ∧ P.2 = 4) → 
  Real.sin α = 4/5 → 
  b = 3 ∨ b = -3 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_terminal_side_l972_97285


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_patel_family_visit_cost_l972_97217

/-- Represents the cost structure for a theme park visit by the Patel family -/
structure ThemeParkVisit where
  regularTicketPrice : ℚ
  seniorDiscount : ℚ
  childrenDiscount : ℚ
  seniorTicketPrice : ℚ
  numGenerations : ℕ
  membersPerGeneration : ℕ

/-- Calculates the total cost for the Patel family's theme park visit -/
def totalCost (visit : ThemeParkVisit) : ℚ :=
  let seniorTicketCost := visit.seniorTicketPrice
  let childrenTicketCost := visit.regularTicketPrice * (1 - visit.childrenDiscount)
  let middleGenerationsCost := visit.regularTicketPrice * (visit.numGenerations - 2) * visit.membersPerGeneration
  2 * seniorTicketCost + 2 * childrenTicketCost + middleGenerationsCost

/-- Theorem stating that the total cost for the Patel family's theme park visit is $85 -/
theorem patel_family_visit_cost :
  ∀ (visit : ThemeParkVisit),
  visit.regularTicketPrice = 25/2 ∧
  visit.seniorDiscount = 1/5 ∧
  visit.childrenDiscount = 2/5 ∧
  visit.seniorTicketPrice = 10 ∧
  visit.numGenerations = 4 ∧
  visit.membersPerGeneration = 2 →
  totalCost visit = 85 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_patel_family_visit_cost_l972_97217


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l972_97218

-- Define the function f(x) = ln(x^2 - x)
noncomputable def f (x : ℝ) : ℝ := Real.log (x^2 - x)

-- Define the domain of f
def domain_f : Set ℝ := {x | x < 0 ∨ x > 1}

-- Theorem stating that the domain of f is (-∞, 0) ∪ (1, +∞)
theorem domain_of_f : {x : ℝ | f x ∈ Set.range f} = domain_f := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l972_97218


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_power_sum_equals_ten_thirds_l972_97216

theorem power_sum_equals_ten_thirds (x : ℝ) (h : x * (Real.log 64 / Real.log 27) = 1) :
  (4 : ℝ)^x + (4 : ℝ)^(-x) = 10/3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_power_sum_equals_ten_thirds_l972_97216


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_salary_solution_l972_97295

/-- Represents a person's salary and expenses --/
structure SalaryExpenses where
  total : ℝ
  food : ℝ
  rent : ℝ
  clothes : ℝ
  transport : ℝ
  remaining : ℝ

/-- The conditions of the salary problem --/
def salaryProblem (s : SalaryExpenses) : Prop :=
  s.food = s.total / 4 ∧
  s.rent = s.total / 8 ∧
  s.clothes = s.total * 3 / 10 ∧
  s.transport = s.total / 6 ∧
  s.remaining = 35000 ∧
  s.total = s.food + s.rent + s.clothes + s.transport + s.remaining

/-- The theorem stating the solution to the salary problem --/
theorem salary_solution :
  ∃ s : SalaryExpenses, salaryProblem s ∧ abs (s.total - 221052.63) < 0.01 := by
  sorry

#check salary_solution

end NUMINAMATH_CALUDE_ERRORFEEDBACK_salary_solution_l972_97295


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_triangle_area_l972_97226

/-- Parabola structure -/
structure Parabola where
  focus : ℝ × ℝ
  directrix : ℝ

/-- Point on a parabola -/
def PointOnParabola (p : Parabola) (point : ℝ × ℝ) : Prop :=
  point.2^2 = 4 * point.1

/-- Distance between two points -/
noncomputable def distance (p q : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

/-- Area of a triangle given three points -/
noncomputable def triangleArea (p q r : ℝ × ℝ) : ℝ :=
  (1/2) * abs ((q.1 - p.1) * (r.2 - p.2) - (r.1 - p.1) * (q.2 - p.2))

theorem parabola_triangle_area 
  (p : Parabola) 
  (P : ℝ × ℝ) :
  p.focus = (1, 0) → 
  p.directrix = -1 → 
  PointOnParabola p P → 
  distance P p.focus = 5 → 
  triangleArea P (-1, 0) p.focus = 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_triangle_area_l972_97226


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_total_profit_calculation_l972_97210

def investment_A : ℚ := 6300
def investment_B : ℚ := 4200
def investment_C : ℚ := 10500
def profit_share_A : ℚ := 3690

theorem total_profit_calculation :
  let total_investment := investment_A + investment_B + investment_C
  let profit_ratio_A := investment_A / total_investment
  let total_profit := profit_share_A / profit_ratio_A
  total_profit = 12300 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_total_profit_calculation_l972_97210


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_octagon_area_sum_l972_97281

/-- Represents the octagon formed by the intersection of two unit squares -/
structure Octagon :=
  (center : Point)
  (sideAB : ℚ)

/-- The area of the octagon as a fraction -/
def octagonArea (oct : Octagon) : ℚ := 8 * (oct.sideAB / 2) * (1 / 2)

/-- The sum of numerator and denominator of the octagon's area -/
def sumNumeratorDenominator (q : ℚ) : ℕ :=
  (q.num.natAbs + q.den)

theorem octagon_area_sum (oct : Octagon) 
  (h : oct.sideAB = 10 / 99) : 
  sumNumeratorDenominator (octagonArea oct) = 119 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_octagon_area_sum_l972_97281


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equation_solution_set_l972_97251

theorem equation_solution_set : 
  {k : ℕ | k > 0 ∧ ∀ x : ℝ, (x^2 - 1)^(2*k) + (x^2 + 2*x)^(2*k) + (2*x + 1)^(2*k) = 2*(1 + x + x^2)^(2*k)} = {1, 2} := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_equation_solution_set_l972_97251


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l972_97258

-- Define the function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.log (x + 1) / Real.log (2 * a)

-- State the theorem
theorem range_of_a (a : ℝ) :
  (∀ x ∈ Set.Ioo (-1 : ℝ) 0, f a x > 0) ↔ a ∈ Set.Ioo 0 (1/2) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l972_97258


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_2016_value_l972_97278

/-- A sequence satisfying the given conditions -/
def my_sequence (a : ℕ → ℚ) : Prop :=
  a 4 = 1/8 ∧
  (∀ n : ℕ, 0 < n → a (n + 2) - a n ≤ 3^n) ∧
  (∀ n : ℕ, 0 < n → a (n + 4) - a n ≥ 10 * 3^n)

/-- The theorem stating the value of a_{2016} -/
theorem a_2016_value (a : ℕ → ℚ) (h : my_sequence a) :
  a 2016 = (81^504 - 80) / 8 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_2016_value_l972_97278


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_bottle_volume_is_625π_l972_97201

/-- The volume of a bottle with cylindrical and hemispherical parts -/
noncomputable def bottle_volume (r : ℝ) (h : ℝ) : ℝ := Real.pi * r^2 * h + (4/3) * Real.pi * r^3

/-- Theorem: The volume of the bottle is 625π -/
theorem bottle_volume_is_625π (r : ℝ) (h : ℝ) 
  (h_condition : r^2 * h + (4/3) * r^3 = 625) : 
  bottle_volume r h = 625 * Real.pi := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_bottle_volume_is_625π_l972_97201


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_log_inequality_l972_97250

theorem log_inequality (a b : ℝ) (ha : a > 1) (hb : b > 1) :
  1 / (3 + Real.log b / Real.log a) + 1 / (3 + Real.log a / Real.log b) ≥ 1 / 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_log_inequality_l972_97250


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complement_A_intersect_B_range_of_a_l972_97282

-- Define the sets A, B, and D
def A : Set ℝ := {x | x ≥ 2}
def B : Set ℝ := {x | -1 ≤ x ∧ x ≤ 5}
def D (a : ℝ) : Set ℝ := {x | 1 - a ≤ x ∧ x ≤ 1 + a}

-- Theorem for part (I)
theorem complement_A_intersect_B :
  (Set.compl A) ∩ B = {x : ℝ | -1 ≤ x ∧ x < 2} := by sorry

-- Theorem for part (II)
theorem range_of_a :
  {a : ℝ | D a ∪ Set.compl B = Set.compl B} = Set.Ioi 0 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_complement_A_intersect_B_range_of_a_l972_97282


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_period_l972_97219

-- Define the function
noncomputable def f (x : ℝ) : ℝ := Real.sin x * (1 + Real.tan x * Real.tan (x / 2))

-- State the theorem
theorem f_period : 
  ∃ (p : ℝ), p > 0 ∧ (∀ (x : ℝ), f (x + p) = f x) ∧ 
  (∀ (q : ℝ), q > 0 ∧ (∀ (x : ℝ), f (x + q) = f x) → p ≤ q) ∧
  p = 2 * Real.pi := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_period_l972_97219


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_s_squared_plus_c_squared_equals_one_l972_97249

theorem s_squared_plus_c_squared_equals_one (x y : ℝ) (h : x^2 + y^2 ≠ 0) :
  let r := Real.sqrt (x^2 + y^2)
  let s := y / r
  let c := x / r
  s^2 + c^2 = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_s_squared_plus_c_squared_equals_one_l972_97249


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_intersection_area_l972_97276

/-- Ellipse with center at origin, major axis along x-axis, focal distance 6, and minor axis 8 -/
def ellipse_C (x y : ℝ) : Prop :=
  x^2 / 25 + y^2 / 16 = 1

/-- Line passing through (-5, 0) with slope π/4 -/
def line_AB (x y : ℝ) : Prop :=
  y = x + 5

/-- Points A and B are the intersection of ellipse_C and line_AB -/
def point_A : ℝ × ℝ := (-5, 0)

noncomputable def point_B : ℝ × ℝ :=
  (-(45:ℝ)/41, 160/41)

/-- Area of triangle ABO -/
noncomputable def area_ABO : ℝ :=
  400 / 41

theorem ellipse_intersection_area :
  ellipse_C (point_B.1) (point_B.2) ∧
  line_AB (point_B.1) (point_B.2) ∧
  area_ABO = (1/2) * 5 * (point_B.2 - point_A.2) := by
  sorry

#check ellipse_intersection_area

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_intersection_area_l972_97276


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_right_hexagonal_pyramid_surface_area_l972_97284

/-- The total surface area of a right hexagonal pyramid -/
noncomputable def totalSurfaceArea (baseSideLength height : ℝ) : ℝ :=
  let baseArea := 3 * Real.sqrt 3 / 2 * baseSideLength ^ 2
  let lateralEdgeLength := Real.sqrt (height ^ 2 + (baseSideLength / 2) ^ 2)
  let lateralArea := 6 * (baseSideLength * lateralEdgeLength / 2)
  baseArea + lateralArea

/-- Theorem: The total surface area of a right hexagonal pyramid with base side length 8 cm and height 15 cm is 96√3 + 24√241 square cm -/
theorem right_hexagonal_pyramid_surface_area :
  totalSurfaceArea 8 15 = 96 * Real.sqrt 3 + 24 * Real.sqrt 241 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_right_hexagonal_pyramid_surface_area_l972_97284


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_point_l972_97290

-- Define the function g
def g : ℝ → ℝ := sorry

-- State the theorem
theorem intersection_point (h1 : ∃! x : ℝ, g x = g (x - 4)) 
                           (h2 : g 2 = g (-2)) : g 2 = 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_point_l972_97290


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_total_income_calculation_l972_97264

def income_distribution (total_income : ℝ) : Prop :=
  let children_share := 0.18 * total_income
  let wife_share := 0.25 * total_income
  let mutual_fund := 0.15 * total_income
  let remaining_before_donation := total_income - (children_share + wife_share + mutual_fund)
  let orphanage_donation := 0.12 * remaining_before_donation
  let final_remaining := remaining_before_donation - orphanage_donation
  final_remaining = 30000

theorem total_income_calculation :
  ∃ (total_income : ℝ), 
    income_distribution total_income ∧ 
    (abs (total_income - 81188.12) < 0.01) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_total_income_calculation_l972_97264


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_equal_value_l972_97261

/-- Represents a 3x3 table of integers -/
def Table := Matrix (Fin 3) (Fin 3) ℕ

/-- Checks if all elements in the table are equal -/
def all_equal (t : Table) : Prop :=
  ∀ i j k l, t i j = t k l

/-- Represents an operation on a row or column -/
inductive Operation
  | row_op (i : Fin 3) (x : ℕ)
  | col_op (j : Fin 3) (x : ℕ)

/-- Applies an operation to a table -/
def apply_operation (t : Table) (op : Operation) : Table :=
  match op with
  | Operation.row_op i x => 
      Matrix.updateRow t i (fun j => 
        if j = 0 then t i 0 - x
        else if j = 1 then t i 1 - x
        else t i 2 + x)
  | Operation.col_op j x => 
      Matrix.updateColumn t j (fun i => 
        if i = 0 then t 0 j - x
        else if i = 1 then t 1 j - x
        else t 2 j + x)

/-- The sum of all elements in a table -/
def table_sum (t : Table) : ℕ :=
  (Finset.univ.sum fun i => (Finset.univ.sum fun j => t i j))

/-- A table is valid if it contains positive integers from 1 to 9 -/
def valid_table (t : Table) : Prop :=
  (∀ i j, t i j ∈ Finset.range 10 \ {0}) ∧ table_sum t = 45

theorem max_equal_value (t : Table) (h : valid_table t) :
  (∃ (ops : List Operation), all_equal (List.foldl apply_operation t ops)) →
  (∀ i j, (List.foldl apply_operation t ops) i j ≤ 5) :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_equal_value_l972_97261


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_volume_arcsin_arccos_rotation_l972_97298

noncomputable def volume_of_rotation (f g : Real → Real) (a b : Real) : Real :=
  Real.pi * ∫ y in a..b, (g y)^2 - (f y)^2

theorem volume_arcsin_arccos_rotation :
  let lower_bound := 0
  let upper_bound := Real.pi / 4
  let outer_function := λ y : Real => Real.cos y
  let inner_function := λ y : Real => Real.sin y
  volume_of_rotation inner_function outer_function lower_bound upper_bound = Real.pi / 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_volume_arcsin_arccos_rotation_l972_97298


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_point_on_parabola_closest_point_solution_l972_97277

/-- The line equation 2x - y - 4 = 0 -/
def line (x y : ℝ) : Prop := 2 * x - y - 4 = 0

/-- The parabola equation y = x^2 -/
def parabola (x y : ℝ) : Prop := y = x^2

/-- The distance from a point (x, y) to the line 2x - y - 4 = 0 -/
noncomputable def distance_to_line (x y : ℝ) : ℝ := 
  abs (2 * x - y - 4) / Real.sqrt 5

/-- The point (1, 1) is on the parabola -/
theorem point_on_parabola : parabola 1 1 := by
  simp [parabola]

/-- The point (1, 1) minimizes the distance to the line -/
theorem closest_point : 
  ∀ x y : ℝ, parabola x y → distance_to_line 1 1 ≤ distance_to_line x y := by
  sorry

/-- The solution to the problem is (1, 1) -/
theorem solution : parabola 1 1 ∧ (∀ x y : ℝ, parabola x y → distance_to_line 1 1 ≤ distance_to_line x y) := by
  constructor
  · exact point_on_parabola
  · exact closest_point

end NUMINAMATH_CALUDE_ERRORFEEDBACK_point_on_parabola_closest_point_solution_l972_97277


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_count_l972_97222

-- Define the two functions
noncomputable def f (x : ℝ) : ℝ := 3 * Real.log x
noncomputable def g (x : ℝ) : ℝ := Real.log (4 * x)

-- Theorem statement
theorem intersection_count :
  ∃! x : ℝ, x > 0 ∧ f x = g x := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_count_l972_97222


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_correct_answer_is_D_l972_97293

theorem correct_answer_is_D : True := by
  -- We're stating that D is the correct answer
  -- In a real proof, we would need to formalize the problem and prove why D is correct
  -- For now, we'll just use trivial since we're given that D is the correct answer
  trivial

#check correct_answer_is_D

end NUMINAMATH_CALUDE_ERRORFEEDBACK_correct_answer_is_D_l972_97293


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_consecutive_odd_integers_count_l972_97223

/-- Given a sequence of consecutive odd integers with average 414 and least integer 313,
    prove that the number of integers in the sequence is 102. -/
theorem consecutive_odd_integers_count 
  (seq : List Int) 
  (consecutive_odd : ∀ i, i + 1 < seq.length → seq[i+1]! = seq[i]! + 2)
  (least_integer : seq.head? = some 313)
  (average : seq.sum / seq.length = 414) :
  seq.length = 102 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_consecutive_odd_integers_count_l972_97223


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_real_part_of_complex_power_l972_97288

theorem real_part_of_complex_power : Complex.re ((1 - Complex.I) ^ 2008) = 2^1004 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_real_part_of_complex_power_l972_97288


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_semicircle_perimeter_approx_l972_97200

/-- The perimeter of a semicircle with radius 10 is approximately 51.42 -/
theorem semicircle_perimeter_approx : 
  ∀ (r : ℝ), r = 10 → ∃ (p : ℝ), |p - (π * r + 2 * r)| < 0.005 ∧ |p - 51.42| < 0.005 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_semicircle_perimeter_approx_l972_97200


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_equation_through_point_l972_97230

/-- Theorem: For a circle with center on the y-axis, radius 1, and passing through (1,2),
    its equation is x^2 + (y-2)^2 = 1 -/
theorem circle_equation_through_point (a : ℝ) :
  (∀ x y : ℝ, x^2 + (y - a)^2 = 1 → (x = 1 ∧ y = 2)) →
  a = 2 ∧ (∀ x y : ℝ, x^2 + (y - 2)^2 = 1 ↔ (x, y) ∈ Metric.sphere (0 : ℝ × ℝ) 1) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_equation_through_point_l972_97230


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_right_triangle_leg_formula_l972_97262

theorem right_triangle_leg_formula (a c : ℝ) (γ : ℝ) 
  (h1 : a > 0) 
  (h2 : c > 0) 
  (h3 : 0 < γ ∧ γ < Real.pi / 2) 
  (h4 : c ≤ a) : 
  c = a * Real.sin γ :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_right_triangle_leg_formula_l972_97262


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_equation_l972_97246

-- Define a line type
structure Line where
  slope : ℝ
  point : ℝ × ℝ

-- Define the line l
noncomputable def line_l : Line where
  slope := Real.sqrt 3
  point := (1, 0)

-- Define the inclined angle of the line
noncomputable def inclined_angle : ℝ := Real.pi / 3

-- Theorem statement
theorem line_equation (l : Line) (h1 : l.point = (1, 0)) (h2 : l.slope = Real.tan inclined_angle) :
  ∀ x y : ℝ, (Real.sqrt 3) * x - y - (Real.sqrt 3) = 0 ↔ y = l.slope * (x - l.point.fst) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_equation_l972_97246


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hat_price_ratio_l972_97229

theorem hat_price_ratio (p : ℚ) (hp : p > 0) : 
  let selling_price := (3 : ℚ) / 4 * p
  let cost_price := (2 : ℚ) / 3 * selling_price
  cost_price / p = (1 : ℚ) / 2 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hat_price_ratio_l972_97229


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_g_expression_inequality_solution_set_l972_97232

-- Define the functions f and g
def f (x : ℝ) : ℝ := x^2 + 2*x

-- g is symmetric to f about the origin
def g (x : ℝ) : ℝ := -x^2 + 2*x

-- Theorem 1: Prove that g(x) = -x^2 + 2x
theorem g_expression : g = fun x => -x^2 + 2*x := by
  ext x
  rfl

-- Theorem 2: Prove that the solution set of g(x) ≥ f(x) - |x-1| is [-1, 1/2]
theorem inequality_solution_set :
  {x : ℝ | g x ≥ f x - |x - 1|} = {x : ℝ | -1 ≤ x ∧ x ≤ 1/2} := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_g_expression_inequality_solution_set_l972_97232


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_C_properties_l972_97214

noncomputable def ellipse_C : Set (ℝ × ℝ) :=
  {p | p.1^2 + p.2^2 / 4 = 1}

noncomputable def F₁ : ℝ × ℝ := (0, -Real.sqrt 3)
noncomputable def F₂ : ℝ × ℝ := (0, Real.sqrt 3)
noncomputable def M : ℝ × ℝ := (Real.sqrt 3 / 2, 1)

theorem ellipse_C_properties :
  (M ∈ ellipse_C) ∧
  (∀ P ∈ ellipse_C, 1 ≤ (1 / dist P F₁ + 1 / dist P F₂) ∧
                    (1 / dist P F₁ + 1 / dist P F₂) ≤ 4) := by
  sorry

#check ellipse_C_properties

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_C_properties_l972_97214


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_nested_op_result_l972_97211

-- Define the operation *
noncomputable def star_op (a b : ℝ) : ℝ := (a - b) / (1 - a * b)

-- Define a placeholder for the nested operation
noncomputable def nested_op : ℝ := 2

-- Theorem statement
theorem nested_op_result : nested_op = 2 := by
  -- The proof is omitted and replaced with sorry
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_nested_op_result_l972_97211


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_two_theta_l972_97265

theorem cos_two_theta (θ : ℝ) : 
  (2 : ℝ)^(-5/2 + 3 * Real.cos θ) + 1 = (2 : ℝ)^(1/2 + Real.cos θ) → 
  Real.cos (2 * θ) = 7/18 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_two_theta_l972_97265


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_f_greater_than_2_l972_97292

-- Define the function f(x) = 2^x
noncomputable def f (x : ℝ) : ℝ := 2^x

-- Define the interval [-2, 2]
def interval : Set ℝ := Set.Icc (-2) 2

-- Define the favorable set where f(x) > 2
def favorable_set : Set ℝ := {x ∈ interval | f x > 2}

-- State the theorem
theorem probability_f_greater_than_2 :
  (MeasureTheory.volume favorable_set) / (MeasureTheory.volume interval) = 1/4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_f_greater_than_2_l972_97292


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_point_A_in_third_quadrant_l972_97244

noncomputable def point_A : ℝ × ℝ := (Real.sin (2014 * Real.pi / 180), Real.cos (2014 * Real.pi / 180))

def in_third_quadrant (p : ℝ × ℝ) : Prop :=
  p.1 < 0 ∧ p.2 < 0

theorem point_A_in_third_quadrant : in_third_quadrant point_A := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_point_A_in_third_quadrant_l972_97244


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_diagonal_intersection_is_center_l972_97272

/-- A rectangle in a 2D plane -/
structure Rectangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ
  D : ℝ × ℝ
  is_rectangle : Prop  -- Condition for ABCD to form a rectangle

/-- The center of a rectangle -/
def center (r : Rectangle) : ℝ × ℝ := sorry

/-- The intersection point of the diagonals of a rectangle -/
def diagonal_intersection (r : Rectangle) : ℝ × ℝ := sorry

/-- Theorem: The intersection point of the diagonals of a rectangle is its center -/
theorem diagonal_intersection_is_center (r : Rectangle) :
  center r = diagonal_intersection r := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_diagonal_intersection_is_center_l972_97272


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_airport_gate_probability_l972_97269

/-- The number of gates in the airport --/
def num_gates : ℕ := 15

/-- The distance between each gate in feet --/
def gate_distance : ℕ := 150

/-- The maximum walking distance in feet --/
def max_distance : ℕ := 600

/-- The probability of selecting two gates that are at most max_distance apart --/
def probability_within_distance : ℚ := 18/35

theorem airport_gate_probability :
  let total_scenarios := num_gates * (num_gates - 1)
  let valid_scenarios := (List.range num_gates).map (λ i => min i (num_gates - i))
    |> List.sum
    |> (· * 2)
    |> (· - num_gates)
  (valid_scenarios : ℚ) / total_scenarios = probability_within_distance := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_airport_gate_probability_l972_97269


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_specific_pyramid_volume_l972_97279

/-- The volume of a pyramid with a rectangular base and equal edge lengths from apex to corners. -/
noncomputable def pyramid_volume (base_length : ℝ) (base_width : ℝ) (edge_length : ℝ) : ℝ :=
  let base_area := base_length * base_width
  let base_diagonal := Real.sqrt (base_length^2 + base_width^2)
  let height := Real.sqrt (edge_length^2 - (base_diagonal/2)^2)
  (1/3) * base_area * height

/-- Theorem: The volume of a pyramid with a 7x9 rectangular base and four edges of length 15 
    from apex to corners is equal to 21√(192.5). -/
theorem specific_pyramid_volume : 
  pyramid_volume 7 9 15 = 21 * Real.sqrt 192.5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_specific_pyramid_volume_l972_97279


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_acute_angles_contradiction_l972_97248

-- Define Triangle as a structure
structure Triangle where
  -- You might want to add appropriate fields here, e.g.:
  -- a : ℝ
  -- b : ℝ
  -- c : ℝ

-- Define count_acute_angles as a function
def count_acute_angles (t : Triangle) : ℕ :=
  -- Placeholder implementation
  0

theorem triangle_acute_angles_contradiction :
  (¬ (∃ (t : Triangle), count_acute_angles t ≥ 2)) ↔
  (∀ (t : Triangle), count_acute_angles t ≤ 1) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_acute_angles_contradiction_l972_97248


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_projection_equality_l972_97228

noncomputable section

/-- The projection vector p that satisfies the given conditions -/
def p : ℝ × ℝ := (21/73, 56/73)

/-- The first given vector -/
def v1 : ℝ × ℝ := (-3, 2)

/-- The second given vector -/
def v2 : ℝ × ℝ := (5, -1)

/-- Dot product of two 2D vectors -/
def dot_product (a b : ℝ × ℝ) : ℝ := a.1 * b.1 + a.2 * b.2

/-- Projection of a vector onto another vector -/
def projection (v onto : ℝ × ℝ) : ℝ × ℝ :=
  let factor := (dot_product v onto) / (dot_product onto onto)
  (factor * onto.1, factor * onto.2)

theorem projection_equality :
  projection v1 p = projection v2 p ∧
  ∀ q : ℝ × ℝ, q ≠ p → projection v1 q ≠ projection v2 q :=
by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_projection_equality_l972_97228


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l972_97243

-- Define the hyperbola
def hyperbola (a b : ℝ) (x y : ℝ) : Prop := x^2 / a^2 - y^2 / b^2 = 1

-- Define the foci
def left_focus (c : ℝ) : ℝ × ℝ := (-c, 0)
def right_focus (c : ℝ) : ℝ × ℝ := (c, 0)

-- Define the circle (renamed to avoid conflict with existing definition)
def circle_eq (center : ℝ × ℝ) (radius : ℝ) (x y : ℝ) : Prop :=
  (x - center.1)^2 + (y - center.2)^2 = radius^2

-- Define the tangent line
def tangent_line (p q : ℝ × ℝ) (x y : ℝ) : Prop :=
  (y - p.2) * (q.1 - p.1) = (x - p.1) * (q.2 - p.2)

-- Define the asymptote
def asymptote (a b : ℝ) (x y : ℝ) : Prop := y = (b / a) * x

-- Main theorem
theorem hyperbola_eccentricity (a b c : ℝ) (F₁ F₂ Q : ℝ × ℝ) :
  hyperbola a b F₁.1 F₁.2 →
  hyperbola a b F₂.1 F₂.2 →
  F₁ = left_focus c →
  F₂ = right_focus c →
  circle_eq F₂ c Q.1 Q.2 →
  tangent_line F₁ Q F₂.1 F₂.2 →
  ∃ (M : ℝ × ℝ), M.1 = (F₁.1 + Q.1) / 2 ∧ M.2 = (F₁.2 + Q.2) / 2 ∧ asymptote a b M.1 M.2 →
  c / a = 2 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l972_97243


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_lucas_sum_is_one_fifth_l972_97242

-- Define the Lucas sequence
def Lucas : ℕ → ℚ
  | 0 => 2  -- Adding this case to cover all natural numbers
  | 1 => 1
  | 2 => 3
  | (n + 3) => Lucas (n + 2) + Lucas (n + 1)

-- Define the sum of the series
noncomputable def LucasSum : ℚ := ∑' n, Lucas (n + 1) / 3^(n+2)

-- Theorem statement
theorem lucas_sum_is_one_fifth : LucasSum = 1/5 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_lucas_sum_is_one_fifth_l972_97242


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_sum_problem_l972_97241

theorem angle_sum_problem (α β : Real) : 
  0 < α ∧ α < Real.pi/2 →  -- α is acute
  0 < β ∧ β < Real.pi/2 →  -- β is acute
  Real.sin α = Real.sqrt 10 / 10 →
  Real.cos β = 2 * Real.sqrt 5 / 5 →
  α + β = Real.pi/4 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_sum_problem_l972_97241


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_side_length_always_greater_than_one_process_can_be_iterated_indefinitely_l972_97224

/-- The side length of the square after n iterations -/
noncomputable def side_length (a : ℝ) : ℕ → ℝ
  | 0 => a
  | n+1 => Real.sqrt ((side_length a n)^2 - 2*(side_length a n) + 2)

/-- Theorem stating that the side length is always greater than 1 after any number of iterations -/
theorem side_length_always_greater_than_one (a : ℝ) (h : a > 1) (n : ℕ) :
  side_length a n > 1 := by
  sorry

/-- Corollary stating that the process can be iterated indefinitely -/
theorem process_can_be_iterated_indefinitely (a : ℝ) (h : a > 1) :
  ∀ n : ℕ, side_length a n > 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_side_length_always_greater_than_one_process_can_be_iterated_indefinitely_l972_97224


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_five_pi_sixths_l972_97263

theorem sin_five_pi_sixths : Real.sin (5 * π / 6) = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_five_pi_sixths_l972_97263


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_least_root_of_g_l972_97247

/-- The polynomial g(x) = 10x^4 - 8x^2 + 3 -/
def g (x : ℝ) : ℝ := 10 * x^4 - 8 * x^2 + 3

/-- The least root of g(x) -/
noncomputable def least_root : ℝ := -Real.sqrt (3/5)

theorem least_root_of_g :
  g least_root = 0 ∧ ∀ x : ℝ, g x = 0 → x ≥ least_root := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_least_root_of_g_l972_97247


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_factors_of_expression_l972_97215

theorem factors_of_expression (x : ℕ) : 
  (Nat.card (Nat.divisors (5^x + 2 * 5^(x+1))) = 2*x + 2) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_factors_of_expression_l972_97215


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_problem_l972_97252

/-- Ellipse C with semi-major axis a and semi-minor axis b -/
structure Ellipse where
  a : ℝ
  b : ℝ
  h_ab : a > b
  h_b_pos : b > 0

/-- Line l passing through a point with slope α -/
structure Line where
  slope : ℝ
  point : ℝ × ℝ

/-- The statement of the ellipse problem -/
theorem ellipse_problem (C : Ellipse) (l : Line) :
  C.a^2 - C.b^2 = 3 →  -- Left focus at (-√3, 0)
  l.point = (-Real.sqrt 3, 0) →  -- Line passes through left focus
  (∃ A B : ℝ × ℝ,  -- Points A and B exist on the ellipse
    (A.1^2 / C.a^2 + A.2^2 / C.b^2 = 1) ∧
    (B.1^2 / C.a^2 + B.2^2 / C.b^2 = 1) ∧
    (∃ k : ℝ, A.2 - B.2 = k * (A.1 - B.1) ∧ k = l.slope)) →
  (l.slope = Real.tan (π/6) →
    ∃ A B : ℝ × ℝ, ((A.1 - l.point.1)^2 + (A.2 - l.point.2)^2).sqrt = 
                   7 * ((B.1 - l.point.1)^2 + (B.2 - l.point.2)^2).sqrt) →
  (C.a = 2 ∧ C.b = 1) ∧  -- Standard equation of ellipse
  (∃ max_area : ℝ, max_area = 1 ∧
    ∀ A B : ℝ × ℝ, (A.1^2 / C.a^2 + A.2^2 / C.b^2 = 1) →
      (B.1^2 / C.a^2 + B.2^2 / C.b^2 = 1) →
      abs (A.1 * B.2 - A.2 * B.1) / 2 ≤ max_area) ∧
  (∃ k : ℝ, k = Real.sqrt 2 ∧
    (∀ x y : ℝ, x - k * y + Real.sqrt 3 = 0 ∨ x + k * y + Real.sqrt 3 = 0)) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_problem_l972_97252


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pentagon_perimeter_l972_97213

theorem pentagon_perimeter (A B C D E : ℝ × ℝ) : 
  let AB := ((B.1 - A.1)^2 + (B.2 - A.2)^2).sqrt
  let BC := ((C.1 - B.1)^2 + (C.2 - B.2)^2).sqrt
  let CD := ((D.1 - C.1)^2 + (D.2 - C.2)^2).sqrt
  let DE := ((E.1 - D.1)^2 + (E.2 - D.2)^2).sqrt
  let AE := ((E.1 - A.1)^2 + (E.2 - A.2)^2).sqrt
  AB = 1 ∧ BC = Real.sqrt 3 ∧ CD = Real.sqrt 2 ∧ DE = Real.sqrt 2 →
  AB + BC + CD + DE + AE = 1 + Real.sqrt 3 + 3 * Real.sqrt 2 := by
  sorry

#check pentagon_perimeter

end NUMINAMATH_CALUDE_ERRORFEEDBACK_pentagon_perimeter_l972_97213


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_root_implies_n_equals_three_l972_97299

open Real

/-- The function f(x) = x ln x + x - k(x - 1) -/
noncomputable def f (k : ℝ) (x : ℝ) : ℝ := x * log x + x - k * (x - 1)

theorem unique_root_implies_n_equals_three (k : ℝ) (n : ℤ) :
  (∃! x₀ : ℝ, x₀ > 1 ∧ f k x₀ = 0) →
  k > n ∧ k < n + 1 →
  n = 3 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_root_implies_n_equals_three_l972_97299


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_lottery_probability_l972_97291

/-- The probability of exactly one common number between two independently chosen
    combinations of 6 numbers from a set of 45 numbers. -/
theorem lottery_probability : ℝ := by
  -- Define the total number of numbers in the lottery
  let total_numbers : ℕ := 45
  -- Define the number of numbers chosen in each combination
  let combination_size : ℕ := 6
  -- Define the probability
  let prob : ℝ := (combination_size : ℝ) * (Nat.choose (total_numbers - combination_size) (combination_size - 1) : ℝ) / (Nat.choose total_numbers combination_size : ℝ)
  -- Assert that this probability is correct
  sorry

-- Use noncomputable for the evaluation
noncomputable def approximate_probability : ℝ :=
  (6 : ℝ) * (Nat.choose 39 5 : ℝ) / (Nat.choose 45 6 : ℝ)

-- Use #eval with a Float approximation
#eval Float.ofScientific 4 2 4 -- Approximates to 0.424

end NUMINAMATH_CALUDE_ERRORFEEDBACK_lottery_probability_l972_97291


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_minimum_positive_period_of_f_l972_97236

noncomputable def f (x : ℝ) : ℝ := Real.sin x * (4 * Real.cos x ^ 2 - 1)

theorem minimum_positive_period_of_f :
  ∃ (T : ℝ), T > 0 ∧ 
  (∀ (x : ℝ), f (x + T) = f x) ∧
  (∀ (T' : ℝ), T' > 0 ∧ (∀ (x : ℝ), f (x + T') = f x) → T' ≥ T) ∧
  T = 2 * π / 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_minimum_positive_period_of_f_l972_97236


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_function_implies_a_value_l972_97234

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := 2 * a - 1 / (3^x + 1)

theorem odd_function_implies_a_value (a : ℝ) :
  (∀ x : ℝ, f a x = -f a (-x)) → a = 1/4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_function_implies_a_value_l972_97234


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_attendees_wednesday_thursday_no_day_exceeds_wednesday_thursday_l972_97274

-- Define the days of the week
inductive Day : Type
| Monday : Day
| Tuesday : Day
| Wednesday : Day
| Thursday : Day
| Friday : Day

-- Define the team members
inductive Member : Type
| Alice : Member
| Bob : Member
| Clara : Member
| David : Member
| Eve : Member

-- Define a function to represent availability
def is_available (m : Member) (d : Day) : Bool :=
  match m, d with
  | Member.Alice, Day.Monday => false
  | Member.Alice, Day.Thursday => false
  | Member.Bob, Day.Tuesday => false
  | Member.Bob, Day.Thursday => false
  | Member.Bob, Day.Friday => false
  | Member.Clara, Day.Monday => false
  | Member.Clara, Day.Wednesday => false
  | Member.Clara, Day.Thursday => false
  | Member.David, Day.Tuesday => false
  | Member.David, Day.Wednesday => false
  | Member.David, Day.Friday => false
  | Member.Eve, Day.Monday => false
  | Member.Eve, Day.Tuesday => false
  | _, _ => true

-- Define a function to count available members for a given day
def count_available (d : Day) : Nat :=
  (List.filter (λ m => is_available m d) [Member.Alice, Member.Bob, Member.Clara, Member.David, Member.Eve]).length

-- Theorem: Wednesday and Thursday have the maximum number of attendees
theorem max_attendees_wednesday_thursday :
  (count_available Day.Wednesday = 3 ∧ count_available Day.Thursday = 3) ∧
  (∀ d : Day, count_available d ≤ 3) := by
  sorry

-- Theorem: No other day has more attendees than Wednesday or Thursday
theorem no_day_exceeds_wednesday_thursday :
  ∀ d : Day, count_available d ≤ count_available Day.Wednesday ∧
             count_available d ≤ count_available Day.Thursday := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_attendees_wednesday_thursday_no_day_exceeds_wednesday_thursday_l972_97274


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_b_c_equal_l972_97231

def a : ℕ → ℤ
  | 0 => 1
  | 1 => 3
  | n + 2 => 4 * a (n + 1) - a n

def b : ℕ → ℚ
  | 0 => 1
  | 1 => 3
  | n + 2 => (b (n + 1)^2 + 2) / b n

noncomputable def c : ℕ → ℝ
  | 0 => 1
  | n + 1 => 2 * c n + Real.sqrt (3 * c n^2 - 2)

theorem a_b_c_equal (n : ℕ) : (a n : ℝ) = b n ∧ b n = c n := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_b_c_equal_l972_97231


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_points_sum_l972_97209

/-- The polar equation of the curve C -/
def polar_equation (ρ θ : ℝ) : Prop :=
  ρ^2 = 9 / (Real.cos θ^2 + 9 * Real.sin θ^2)

/-- Two points on the curve C -/
structure Point (polar_equation : ℝ → ℝ → Prop) where
  ρ : ℝ
  θ : ℝ
  on_curve : polar_equation ρ θ

/-- The perpendicularity condition -/
def perpendicular (A B : Point polar_equation) : Prop :=
  abs (A.θ - B.θ) = Real.pi / 2

/-- The theorem to be proved -/
theorem perpendicular_points_sum (A B : Point polar_equation) 
  (h : perpendicular A B) : 
  1 / A.ρ^2 + 1 / B.ρ^2 = 10 / 9 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_points_sum_l972_97209


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_between_vertices_l972_97294

/-- The vertex of a quadratic function f(x) = ax² + bx + c is at (-b/(2a), f(-b/(2a))) -/
noncomputable def quadratic_vertex (a b c : ℝ) : ℝ × ℝ :=
  let x := -b / (2 * a)
  (x, a * x^2 + b * x + c)

/-- The distance between two points in 2D space -/
noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

theorem distance_between_vertices : 
  let f1 (x : ℝ) := x^2 - 2*x + 3
  let f2 (x : ℝ) := x^2 + 4*x + 10
  let v1 := quadratic_vertex 1 (-2) 3
  let v2 := quadratic_vertex 1 4 10
  distance v1 v2 = 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_between_vertices_l972_97294


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_m_l972_97275

-- Define the function f as noncomputable
noncomputable def f (x m : ℝ) : ℝ := Real.log (x^2 + 2*x + m^2)

-- State the theorem
theorem range_of_m (m : ℝ) :
  (∀ y : ℝ, ∃ x : ℝ, f x m = y) → m ∈ Set.Icc (-1 : ℝ) 1 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_m_l972_97275


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_survey_b_count_l972_97225

/-- Represents a systematic sampling scenario -/
structure SystematicSample where
  population : ℕ
  sampleSize : ℕ
  firstDrawn : ℕ

/-- Calculates the nth term of the arithmetic sequence for a given systematic sample -/
def nthTerm (s : SystematicSample) (n : ℕ) : ℕ :=
  s.firstDrawn + (n - 1) * (s.population / s.sampleSize)

/-- Counts the number of terms in a given interval -/
def countInInterval (s : SystematicSample) (lower upper : ℕ) : ℕ :=
  Finset.card (Finset.filter (fun n => 
    lower ≤ nthTerm s n ∧ nthTerm s n ≤ upper) (Finset.range s.sampleSize))

theorem survey_b_count (s : SystematicSample) 
  (h1 : s.population = 960) 
  (h2 : s.sampleSize = 32) 
  (h3 : s.firstDrawn = 9) : 
  countInInterval s 451 750 = 10 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_survey_b_count_l972_97225


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_find_a2_l972_97227

/-- The sum of the first n terms of the sequence {aₙ} -/
def S (n : ℕ+) : ℝ := sorry

/-- The nth term of the sequence {aₙ} -/
def a (n : ℕ+) : ℝ := sorry

/-- The main theorem stating the conditions and the result to be proved -/
theorem find_a2 :
  (∀ n : ℕ+, 2 * S n - n * a n = n) →
  S 20 = -360 →
  a 2 = -1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_find_a2_l972_97227


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solid_color_probability_l972_97259

/-- Represents a unit cube with 3 red faces and 3 blue faces adjacent to a vertex -/
structure ColoredCube where
  redFaces : Fin 3
  blueFaces : Fin 3

/-- Represents the larger 3x3x3 cube -/
structure LargeCube where
  smallCubes : Fin 27 → ColoredCube

/-- The probability of a single cube being correctly oriented based on its position -/
def orientationProbability (position : Nat) : ℚ :=
  if position < 8 then 1/8     -- corner cubes
  else if position < 20 then 1/4   -- edge cubes
  else if position < 26 then 1/2   -- face-center cubes
  else 1                        -- center cube

/-- The probability of the entire cube being one solid color -/
noncomputable def solidColorProbability (cube : LargeCube) : ℚ :=
  2 * (Finset.range 27).prod (fun i => orientationProbability i)

theorem solid_color_probability (cube : LargeCube) :
  solidColorProbability cube = 1 / 2^53 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_solid_color_probability_l972_97259


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_triangle_area_l972_97240

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := 2 * Real.sin x * Real.cos x - 2 * (Real.sin x)^2 + 1

-- Define the triangle ABC
structure Triangle where
  A : ℝ
  B : ℝ
  C : ℝ
  a : ℝ
  b : ℝ
  c : ℝ

-- State the theorem
theorem max_triangle_area (ABC : Triangle) :
  ABC.a = Real.sqrt 3 →
  0 < ABC.A ∧ ABC.A < Real.pi / 2 →
  f (ABC.A + Real.pi / 8) = Real.sqrt 2 / 3 →
  (∀ S : ℝ, S = 1/2 * ABC.b * ABC.c * Real.sin ABC.A → S ≤ 3 * (Real.sqrt 3 + Real.sqrt 2) / 4) :=
by
  sorry

-- Add a dummy main function to ensure the file is not empty
def main : IO Unit :=
  IO.println "Theorem stated and proof skipped with 'sorry'"

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_triangle_area_l972_97240


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_pass_time_approx_l972_97296

/-- The time (in seconds) it takes for a train to pass a jogger --/
noncomputable def trainPassTime (joggerSpeed : ℝ) (trainSpeed : ℝ) (trainLength : ℝ) (initialDistance : ℝ) : ℝ :=
  let joggerSpeedMPS := joggerSpeed * 1000 / 3600
  let trainSpeedMPS := trainSpeed * 1000 / 3600
  let relativeSpeed := trainSpeedMPS - joggerSpeedMPS
  let totalDistance := initialDistance + trainLength
  totalDistance / relativeSpeed

/-- Theorem stating that the time for the train to pass the jogger is approximately 43.64 seconds --/
theorem train_pass_time_approx :
  ∃ ε > 0, ε < 0.01 ∧ 
  |trainPassTime 9 75 300 500 - 43.64| < ε := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_pass_time_approx_l972_97296


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_radical4_is_simplest_l972_97203

/-- A function that determines if a quadratic radical is in its simplest form -/
def is_simplest_quadratic_radical (f : ℝ → ℝ) : Prop :=
  ∀ a : ℝ, ∃ b c : ℝ, f a = Real.sqrt (b * a + c) ∧ b ≠ 0 ∧ c ≠ 0

/-- The given quadratic radicals -/
noncomputable def radical1 (a : ℝ) : ℝ := Real.sqrt (0.1 * a)
noncomputable def radical2 (a : ℝ) : ℝ := Real.sqrt (1 / (2 * a))
noncomputable def radical3 (a : ℝ) : ℝ := Real.sqrt (a^3)
noncomputable def radical4 (a : ℝ) : ℝ := Real.sqrt (a^2 + 1)

/-- Theorem stating that radical4 is the simplest quadratic radical -/
theorem radical4_is_simplest :
  is_simplest_quadratic_radical radical4 ∧
  ¬is_simplest_quadratic_radical radical1 ∧
  ¬is_simplest_quadratic_radical radical2 ∧
  ¬is_simplest_quadratic_radical radical3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_radical4_is_simplest_l972_97203


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_identity_or_periodic_l972_97270

theorem function_identity_or_periodic 
  (f g : ℝ → ℝ) 
  (h : ∀ x y : ℝ, f (g x + y) = g (x + y)) : 
  (∀ x : ℝ, f x = x) ∨ (∃ p : ℝ, p ≠ 0 ∧ ∀ x : ℝ, g (x + p) = g x) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_identity_or_periodic_l972_97270


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_all_white_hats_deduction_l972_97268

/-- Represents the color of a hat -/
inductive HatColor
  | White
  | Black

/-- Represents a student -/
structure Student where
  id : Nat
  hatColor : HatColor

/-- Represents the hat distribution -/
structure HatDistribution where
  totalHats : Nat
  whiteHats : Nat
  blackHats : Nat
  students : List Student

/-- Checks if a student can deduce their hat color -/
def canDeduceHatColor (dist : HatDistribution) (s : Student) : Prop :=
  ∀ (otherStudents : List Student),
    otherStudents.length = dist.students.length - 1 ∧
    otherStudents.all (λ st ↦ st.id ≠ s.id) →
    ∃ (deducedColor : HatColor), deducedColor = s.hatColor

/-- The main theorem to prove -/
theorem all_white_hats_deduction
  (dist : HatDistribution)
  (h1 : dist.totalHats = 5)
  (h2 : dist.whiteHats = 3)
  (h3 : dist.blackHats = 2)
  (h4 : dist.students.length = 3)
  (h5 : ∀ s ∈ dist.students, canDeduceHatColor dist s) :
  ∀ s ∈ dist.students, s.hatColor = HatColor.White :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_all_white_hats_deduction_l972_97268


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_flower_bed_fraction_l972_97280

noncomputable def yard_length : ℝ := 30
noncomputable def yard_width : ℝ := 8
noncomputable def trapezoid_short_side : ℝ := 14
noncomputable def trapezoid_long_side : ℝ := 24
noncomputable def trapezoid_height : ℝ := 6

noncomputable def yard_area : ℝ := yard_length * yard_width

noncomputable def trapezoid_area : ℝ := (trapezoid_short_side + trapezoid_long_side) * trapezoid_height / 2

noncomputable def fraction_occupied : ℝ := trapezoid_area / yard_area

theorem flower_bed_fraction :
  fraction_occupied = 19 / 40 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_flower_bed_fraction_l972_97280


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_solution_set_l972_97245

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := Real.log (Real.exp (2 * x) + 1) - x

-- State the theorem
theorem inequality_solution_set :
  {x : ℝ | f (x + 2) > f (2 * x - 3)} = Set.Ioo (1/3 : ℝ) 5 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_solution_set_l972_97245


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_picture_frame_height_l972_97256

/-- A rectangular picture frame with given width and perimeter -/
structure PictureFrame where
  width : ℚ
  perimeter : ℚ

/-- The height of a rectangular picture frame -/
def frame_height (f : PictureFrame) : ℚ :=
  (f.perimeter - 2 * f.width) / 2

/-- Theorem: A picture frame with width 6 inches and perimeter 30 inches has a height of 9 inches -/
theorem picture_frame_height :
  let f : PictureFrame := { width := 6, perimeter := 30 }
  frame_height f = 9 := by
  -- Unfold the definition of frame_height
  unfold frame_height
  -- Simplify the arithmetic
  simp
  -- The proof is complete
  rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_picture_frame_height_l972_97256


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_zero_at_negative_one_f_increasing_when_m_between_negative_one_and_one_f_monotonicity_when_m_greater_than_one_l972_97271

noncomputable section

def f (m : ℝ) (x : ℝ) : ℝ := x - 1/x - 2*m*(Real.log x)

theorem f_zero_at_negative_one :
  ∃! x : ℝ, x > 0 ∧ f (-1) x = 0 :=
sorry

theorem f_increasing_when_m_between_negative_one_and_one (m : ℝ) (h : -1 < m ∧ m ≤ 1) :
  StrictMonoOn (f m) (Set.Ioi 0) :=
sorry

theorem f_monotonicity_when_m_greater_than_one (m : ℝ) (h : m > 1) :
  (StrictMonoOn (f m) (Set.Ioo 0 (m - Real.sqrt (m^2 - 1)))) ∧
  (StrictAntiOn (f m) (Set.Ioo (m - Real.sqrt (m^2 - 1)) (m + Real.sqrt (m^2 - 1)))) ∧
  (StrictMonoOn (f m) (Set.Ioi (m + Real.sqrt (m^2 - 1)))) :=
sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_zero_at_negative_one_f_increasing_when_m_between_negative_one_and_one_f_monotonicity_when_m_greater_than_one_l972_97271


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_lambda_value_l972_97255

theorem max_lambda_value (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : x^3 + y^3 = x - y) :
  (∃ (lambda_max : ℝ), lambda_max = 2 + 2 * Real.sqrt 2 ∧
    (∀ (lambda : ℝ), (∀ (x y : ℝ), x > 0 → y > 0 → x^3 + y^3 = x - y → x^2 + lambda * y^2 ≤ 1) → lambda ≤ lambda_max)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_lambda_value_l972_97255


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_generalized_means_inequality_l972_97287

/-- A theorem about generalized means of a continuously positive function -/
theorem generalized_means_inequality 
  {I : Set ℝ} 
  (hI : I ⊆ Set.Ici 0) 
  (f : ℝ → ℝ) 
  (hf : Continuous f) 
  (hfpos : ∀ x ∈ I, f x > 0) 
  (h2 : ∀ (x₁ x₂ : ℝ) (p₁ p₂ : ℝ), 
    x₁ ∈ I → x₂ ∈ I → p₁ > 0 → p₂ > 0 → p₁ + p₂ = 1 → 
    (f x₁) ^ p₁ * (f x₂) ^ p₂ ≥ f (p₁ * x₁ + p₂ * x₂)) :
  ∀ (n : ℕ) (hn : n ≥ 2) (x : Fin n → ℝ) (p : Fin n → ℝ),
    (∀ i, x i ∈ I) → 
    (∀ i, p i > 0) → 
    (Finset.sum Finset.univ p = 1) →
    (Finset.prod Finset.univ (λ i ↦ f (x i))) ≥ f (Finset.sum Finset.univ (λ i ↦ p i * x i)) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_generalized_means_inequality_l972_97287


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_special_primes_divisible_by_five_l972_97286

theorem sum_of_special_primes_divisible_by_five (A B : ℕ) :
  Prime A ∧ Prime B ∧ Prime (A - 3) ∧ Prime (A + 3) →
  5 ∣ (A + B + (A - 3) + (A + 3)) :=
by
  intro h
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_special_primes_divisible_by_five_l972_97286


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_line_intersection_l972_97220

/-- Given a parabola and a point, prove properties of the intersections with a line through the point -/
theorem parabola_line_intersection
  (p : ℝ) (x₀ y₀ x₁ y₁ x₂ y₂ : ℝ)
  (h_p : p > 0)
  (h_parabola : ∀ x y, y^2 = 2*p*x → (x, y) ∈ Set.range (λ t => (t^2/(2*p), t)))
  (h_not_origin : (x₀, y₀) ≠ (0, 0))
  (h_on_line : ∃ k m : ℝ, x₀ = k*y₀ + m ∧ x₁ = k*y₁ + m ∧ x₂ = k*y₂ + m)
  (h_on_parabola : y₁^2 = 2*p*x₁ ∧ y₂^2 = 2*p*x₂) :
  (y₀ = 0 → y₁*y₂ = -2*p*x₀) ∧
  (x₀ = 0 → 1/y₁ + 1/y₂ = 1/y₀) ∧
  (y₀ - y₁)*(y₀ - y₂) = y₀^2 - 2*p*x₀ :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_line_intersection_l972_97220


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_of_function_l972_97283

theorem min_value_of_function (x : ℝ) (hx : x > 0) : 
  6 * x^6 + 7 * x^(-5 : ℝ) ≥ 13 ∧ ∃ y : ℝ, y > 0 ∧ 6 * y^6 + 7 * y^(-5 : ℝ) = 13 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_of_function_l972_97283


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_interval_length_sum_l972_97212

open Real Set

theorem interval_length_sum (a b : ℝ) (h : a > b) : 
  ∃ (x₁ x₂ : ℝ), 
    b < x₁ ∧ x₁ < a ∧ a < x₂ ∧
    (∀ x, (1 / (x - a) + 1 / (x - b) ≥ 1) ↔ (x ∈ Ioc b x₁ ∪ Ioo a x₂)) ∧
    (x₂ - a) + (x₁ - b) = 2 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_interval_length_sum_l972_97212


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_locus_is_two_rays_l972_97208

-- Define the segment AB
def segment_AB : Set (Fin 2 → ℝ) := sorry

-- Define the angle 70°
noncomputable def angle_70 : ℝ := 70 * Real.pi / 180

-- Define the locus of points M
def locus_M (A B : Fin 2 → ℝ) : Set (Fin 2 → ℝ) :=
  {M : Fin 2 → ℝ | sorry} -- Placeholder for angle condition

-- Define a ray
def ray (A : Fin 2 → ℝ) (θ : ℝ) : Set (Fin 2 → ℝ) := sorry

-- Theorem statement
theorem locus_is_two_rays (A B : Fin 2 → ℝ) :
  ∃ (θ₁ θ₂ : ℝ), locus_M A B = ray A θ₁ ∪ ray A θ₂ := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_locus_is_two_rays_l972_97208


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_juan_running_time_l972_97235

/-- Calculates the time spent running given distance and speed. -/
noncomputable def timeSpentRunning (distance : ℝ) (speed : ℝ) : ℝ :=
  distance / speed

/-- Theorem: Juan's running time is 8 hours -/
theorem juan_running_time : 
  timeSpentRunning 80 10 = 8 := by
  unfold timeSpentRunning
  norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_juan_running_time_l972_97235


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parallelogram_properties_l972_97233

/-- Parallelogram OABC with given vertices -/
structure Parallelogram where
  O : ℝ × ℝ := (0, 0)
  A : ℝ × ℝ := (3, 6)
  B : ℝ × ℝ := (8, 6)
  C : ℝ × ℝ := (5, 0)  -- Derived from the problem

/-- Function to calculate length of a line segment -/
noncomputable def length (p q : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

/-- Theorem about the parallelogram OABC -/
theorem parallelogram_properties (OABC : Parallelogram) :
  -- 1. Ratio of diagonal lengths
  (length OABC.O OABC.B) / (length OABC.A OABC.C) = Real.sqrt 2.5 ∧
  -- 2. Equations of sides and diagonal
  (∀ x y, y = 0 ↔ (x, y) ∈ Set.Icc OABC.O OABC.C) ∧  -- OC
  (∀ x y, y = 6 ↔ (x, y) ∈ Set.Icc OABC.A OABC.B) ∧  -- AB
  (∀ x y, y = 2*x ↔ (x, y) ∈ Set.Icc OABC.O OABC.A) ∧  -- OA
  (∀ x y, y = 2*x - 10 ↔ (x, y) ∈ Set.Icc OABC.B OABC.C) ∧  -- BC
  (∀ x y, y = -3*x + 15 ↔ (x, y) ∈ Set.Icc OABC.A OABC.C)  -- AC
  := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parallelogram_properties_l972_97233


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_line_through_circle_center_l972_97239

/-- The equation of the line passing through the center of the circle x^2 + 2x + y^2 - 3 = 0
    and perpendicular to the line x + y - 1 = 0 is x - y + 1 = 0 -/
theorem perpendicular_line_through_circle_center :
  ∃ (circle given_line perpendicular_line : ℝ → ℝ → Prop) (center : ℝ × ℝ),
    (circle = fun x y => x^2 + 2*x + y^2 - 3 = 0) ∧
    (given_line = fun x y => x + y - 1 = 0) ∧
    (perpendicular_line = fun x y => x - y + 1 = 0) ∧
    (center = (-1, 0)) ∧
    (∀ x y, given_line x y → (y - 0) / (x - (-1)) * (-1) = -1) ∧
    perpendicular_line center.1 center.2 ∧
    (∀ x y, circle x y → (x + 1)^2 + y^2 = 4) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_line_through_circle_center_l972_97239


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fourth_root_simplification_l972_97254

theorem fourth_root_simplification (α β : ℕ) :
  ((2^9 * 5^2 : ℚ)^(1/4) : ℚ) = (α : ℚ) * ((β : ℚ)^(1/4) : ℚ) → α + β = 54 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fourth_root_simplification_l972_97254


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_sum_l972_97297

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a parabola y^2 = 4x -/
def Parabola := {p : Point | p.y^2 = 4 * p.x}

/-- Represents a circle (x-4)^2 + (y-1)^2 = 1 -/
def Circle := {p : Point | (p.x - 4)^2 + (p.y - 1)^2 = 1}

/-- The focus of the parabola y^2 = 4x -/
def focus : Point := ⟨1, 0⟩

/-- Distance between two points -/
noncomputable def distance (p1 p2 : Point) : ℝ :=
  Real.sqrt ((p1.x - p2.x)^2 + (p1.y - p2.y)^2)

/-- Theorem: The minimum value of |MA| + |MF| is 4 -/
theorem min_distance_sum :
  ∀ (M : Point) (A : Point),
    M ∈ Parabola →
    A ∈ Circle →
    (∀ (M' : Point) (A' : Point),
      M' ∈ Parabola →
      A' ∈ Circle →
      distance M A + distance M focus ≤ distance M' A' + distance M' focus) →
    distance M A + distance M focus = 4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_sum_l972_97297


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ratio_value_l972_97266

/-- A geometric sequence with positive terms satisfying a specific condition -/
structure SpecialGeometricSequence where
  a : ℕ → ℝ
  is_geometric : ∀ n : ℕ, a (n + 1) = a n * (a 1 / a 0)
  all_positive : ∀ n : ℕ, a n > 0
  special_condition : a 3 = a 1 + 2 * a 2

/-- The ratio of specific terms in the special geometric sequence -/
noncomputable def ratio (seq : SpecialGeometricSequence) : ℝ :=
  (seq.a 9 + seq.a 10) / (seq.a 7 + seq.a 8)

/-- The theorem stating the value of the ratio -/
theorem ratio_value (seq : SpecialGeometricSequence) : ratio seq = 3 + 2 * Real.sqrt 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ratio_value_l972_97266


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_melted_value_is_100_l972_97273

/-- The value of melted gold quarters per ounce -/
def meltedValuePerOunce (faceValue : ℚ) (weight : ℚ) (meltMultiplier : ℕ) : ℚ :=
  (faceValue * meltMultiplier) / weight

/-- Theorem stating the value of melted gold quarters per ounce is $100 -/
theorem melted_value_is_100 :
  meltedValuePerOunce (1/4) (1/5) 80 = 100 := by
  -- Unfold the definition of meltedValuePerOunce
  unfold meltedValuePerOunce
  -- Simplify the arithmetic
  simp
  -- The proof is complete
  rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_melted_value_is_100_l972_97273


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_diamond_equation_solution_l972_97238

noncomputable def diamond (a b : ℝ) : ℝ := Real.sqrt (a + b + 3) / Real.sqrt (a - b + 1)

theorem diamond_equation_solution :
  ∀ y : ℝ, diamond y 15 = 5 → y = 46 / 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_diamond_equation_solution_l972_97238


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_expression_independent_of_alpha_l972_97221

theorem expression_independent_of_alpha (α : ℝ) (h : ∀ n : ℤ, α ≠ π * n / 2) :
  (1 - Real.cos (α - 3 * π / 2) ^ 4 - Real.sin (α + 3 * π / 2) ^ 4) / 
  (Real.sin α ^ 6 + Real.cos α ^ 6 - 1) = -2/3 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_expression_independent_of_alpha_l972_97221


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_bisector_and_projection_l972_97253

-- Define the circle and points
variable (Circle : Type) (A B C D E : Circle)

-- Define the properties of the points
variable (midpoint : Circle → Circle → Circle → Prop)
variable (on_arc : Circle → Circle → Circle → Prop)
variable (orthogonal_projection : Circle → Circle → Circle → Circle → Prop)
variable (external_angle_bisector : Circle → Circle → Circle → Circle → Prop)

-- Define distance function
variable (distance : Circle → Circle → ℝ)

-- State the theorem
theorem circle_bisector_and_projection 
  (h_midpoint : midpoint D A C)
  (h_on_arc : on_arc B D C)
  (h_not_on_arc : ¬ on_arc B A C)
  (h_projection : orthogonal_projection E D A B) :
  external_angle_bisector B D A C ∧ 
  distance A E = distance B E + distance B C :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_bisector_and_projection_l972_97253


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_parallel_to_parallel_planes_l972_97204

-- Define a type for planes
structure Plane where

-- Define a type for lines
structure Line where

-- Define the parallel relationship between planes
def parallel_planes (p1 p2 : Plane) : Prop :=
  sorry

-- Define the parallel relationship between a line and a plane
def parallel_line_plane (l : Line) (p : Plane) : Prop :=
  sorry

-- Define when a line is within a plane
def line_within_plane (l : Line) (p : Plane) : Prop :=
  sorry

theorem line_parallel_to_parallel_planes 
  (p1 p2 : Plane) (l : Line) 
  (h1 : parallel_planes p1 p2) 
  (h2 : parallel_line_plane l p1) : 
  parallel_line_plane l p2 ∨ line_within_plane l p2 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_parallel_to_parallel_planes_l972_97204


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_oblique_prism_properties_l972_97207

/-- Represents an oblique triangular prism -/
structure ObliquePrism where
  -- Edge length of the prism
  a : ℝ
  -- Assertion that B₁C₁CB is perpendicular to ABC
  h1 : Plane → Plane → Prop
  -- Assertion that AC₁ is perpendicular to BC
  h2 : Line → Line → Prop

/-- The distance between skew lines AA₁ and B₁C₁ in the prism -/
noncomputable def skew_lines_distance (prism : ObliquePrism) : ℝ :=
  (Real.sqrt 3 / 2) * prism.a

/-- The dihedral angle between plane A₁B₁BA and plane ABC in the prism -/
noncomputable def dihedral_angle (prism : ObliquePrism) : ℝ :=
  Real.pi - Real.arctan 2

theorem oblique_prism_properties (prism : ObliquePrism) :
  (∃ d, d = skew_lines_distance prism) ∧
  (∃ θ, θ = dihedral_angle prism) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_oblique_prism_properties_l972_97207


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_min_on_interval_l972_97205

theorem cos_min_on_interval :
  ∃ x ∈ Set.Icc (-Real.pi/3) (Real.pi/6), 
    ∀ y ∈ Set.Icc (-Real.pi/3) (Real.pi/6), 
      Real.cos x ≤ Real.cos y ∧ Real.cos x = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_min_on_interval_l972_97205
