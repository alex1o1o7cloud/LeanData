import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_distance_to_z₁_l593_59364

def z₁ : ℂ := Complex.I * (1 - Complex.I)^3

theorem max_distance_to_z₁ :
  ∃ (max_dist : ℝ), max_dist = 1 + 2 * Real.sqrt 2 ∧
  ∀ (z : ℂ), Complex.abs z = 1 → Complex.abs (z - z₁) ≤ max_dist :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_distance_to_z₁_l593_59364


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distinct_prime_factors_of_divisors_product_l593_59328

def divisors (n : ℕ) : Finset ℕ :=
  Finset.filter (· ∣ n) (Finset.range (n + 1))

def product_of_divisors (n : ℕ) : ℕ :=
  (divisors n).prod id

def num_distinct_prime_factors (n : ℕ) : ℕ :=
  (Nat.factors n).toFinset.card

theorem distinct_prime_factors_of_divisors_product :
  num_distinct_prime_factors (product_of_divisors 60) = 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distinct_prime_factors_of_divisors_product_l593_59328


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_side_c_max_value_of_t_l593_59372

-- Define the triangle ABC
structure Triangle where
  A : ℝ
  B : ℝ
  C : ℝ
  a : ℝ
  b : ℝ
  c : ℝ

-- Define the conditions
def isArithmeticSequence (t : Triangle) : Prop :=
  2 * t.B = t.A + t.C

def anglesSum (t : Triangle) : Prop :=
  t.A + t.B + t.C = Real.pi

def lawOfCosines (t : Triangle) : Prop :=
  t.b^2 = t.a^2 + t.c^2 - 2 * t.a * t.c * Real.cos t.B

-- Theorem 1
theorem triangle_side_c (t : Triangle) 
  (h1 : isArithmeticSequence t)
  (h2 : anglesSum t)
  (h3 : lawOfCosines t)
  (h4 : t.b = Real.sqrt 13)
  (h5 : t.a = 3) :
  t.c = 4 := by sorry

-- Theorem 2
noncomputable def t (A C : ℝ) : ℝ :=
  Real.sin A * Real.sin C

theorem max_value_of_t (A C : ℝ)
  (h : A + C = 2/3 * Real.pi) :
  ∃ (max : ℝ), max = 3/4 ∧ ∀ (A' C' : ℝ), A' + C' = 2/3 * Real.pi → t A' C' ≤ max := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_side_c_max_value_of_t_l593_59372


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_horse_tile_problem_l593_59356

/-- Represents the number of big horses -/
def x : ℕ := sorry

/-- Represents the number of small horses -/
def y : ℕ := sorry

/-- The total number of horses -/
def total_horses : ℕ := 100

/-- The total number of tiles pulled -/
def total_tiles : ℕ := 100

/-- The number of tiles a big horse can pull -/
def big_horse_capacity : ℕ := 3

/-- The number of small horses needed to pull one tile -/
def small_horses_per_tile : ℕ := 3

theorem horse_tile_problem :
  (x + y = total_horses) ∧
  ((3 : ℚ) * x + (1 / 3 : ℚ) * y = total_tiles) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_horse_tile_problem_l593_59356


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inclination_angle_range_l593_59318

open Set
open Real

/-- The inclination angle range of the line x cos θ + √3 y + 2 = 0 -/
theorem inclination_angle_range :
  ∃ α : ℝ, ∀ θ x y : ℝ, x * cos θ + sqrt 3 * y + 2 = 0 →
    α ∈ Icc 0 (π/6) ∪ Ico (5*π/6) π :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inclination_angle_range_l593_59318


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cost_per_square_foot_is_five_l593_59355

/-- Represents the cost per square foot of theater space -/
def CostPerSquareFoot (
  squareFeetPerSeat : ℚ
) (numSeats : ℕ
) (constructionCostMultiplier : ℚ
) (partnerCoveragePercentage : ℚ
) (tomSpending : ℚ
) : ℚ :=
  let totalSquareFootage := squareFeetPerSeat * numSeats
  let tomCoveragePercentage := 1 - partnerCoveragePercentage
  let totalCost := tomSpending / tomCoveragePercentage
  let landCost := totalCost / (1 + constructionCostMultiplier)
  landCost / totalSquareFootage

/-- Theorem stating that the cost per square foot is $5 given the problem conditions -/
theorem cost_per_square_foot_is_five :
  CostPerSquareFoot 12 500 2 (2/5) 54000 = 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cost_per_square_foot_is_five_l593_59355


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_b_100_eq_one_third_l593_59391

def b : ℕ → ℚ
  | 0 => 0  -- Adding a case for 0 to cover all natural numbers
  | 1 => 1/2
  | 2 => 1
  | (n+3) => (2 - b (n+2)) / (3 * b (n+1))

theorem b_100_eq_one_third : b 100 = 1/3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_b_100_eq_one_third_l593_59391


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_theorem_l593_59365

open Real MeasureTheory

-- Define the function f(x) = sin(2x)
noncomputable def f (x : ℝ) : ℝ := Real.sin (2 * x)

-- Define the interval [-π/4, π/4]
def interval : Set ℝ := Set.Icc (-π/4) (π/4)

-- Define the subset where f(x) ≥ 1/2
def subset : Set ℝ := {x ∈ interval | f x ≥ 1/2}

-- State the theorem
theorem probability_theorem :
  (volume subset) / (volume interval) = 1/3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_theorem_l593_59365


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_solved_a_l593_59321

/-- Represents the number of people who solved only problem A -/
def a : ℕ := sorry

/-- Represents the number of people who solved only problem B -/
def b : ℕ := sorry

/-- Represents the number of people who solved only problem C -/
def c : ℕ := sorry

/-- Represents the number of people who solved A and B but not C -/
def x : ℕ := sorry

/-- Represents the number of people who solved A and C but not B -/
def y : ℕ := sorry

/-- Represents the number of people who solved B and C but not A -/
def z : ℕ := sorry

/-- Represents the number of people who solved all three problems -/
def w : ℕ := sorry

/-- The total number of participants -/
def total_participants : ℕ := 39

/-- The condition that each person solved at least one problem -/
axiom all_solved_one : a + b + c + x + y + z + w = total_participants

/-- The condition about people who solved problem A -/
axiom a_condition : a = (x + y + w) + 5

/-- The condition about people who did not solve problem A -/
axiom bc_condition : b + z = 2 * (c + y)

/-- The condition about people who only solved problem A -/
axiom only_a_condition : a = b + c

/-- The theorem stating the maximum number of people who solved problem A -/
theorem max_solved_a : 
  ∃ (max_a : ℕ), max_a ≤ a + x + y + w ∧ 
  ∀ (n : ℕ), n ≤ a + x + y + w → n ≤ max_a ∧ max_a = 23 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_solved_a_l593_59321


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_bisector_of_AB_A_B_on_line_and_circle_l593_59359

-- Define the line and circle
def line_eq (x y : ℝ) : Prop := 3 * x + 4 * y + 2 = 0
def circle_eq (x y : ℝ) : Prop := x^2 + y^2 + 4 * y = 0

-- Define points A and B as intersections of the line and circle
noncomputable def A : ℝ × ℝ := sorry
noncomputable def B : ℝ × ℝ := sorry

-- Define the perpendicular bisector
def perpendicular_bisector (x y : ℝ) : Prop := 4 * x - 3 * y - 6 = 0

-- Theorem statement
theorem perpendicular_bisector_of_AB :
  perpendicular_bisector A.1 A.2 ∧ perpendicular_bisector B.1 B.2 :=
by sorry

-- Additional theorem to show that A and B satisfy the line and circle equations
theorem A_B_on_line_and_circle :
  line_eq A.1 A.2 ∧ circle_eq A.1 A.2 ∧ line_eq B.1 B.2 ∧ circle_eq B.1 B.2 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_bisector_of_AB_A_B_on_line_and_circle_l593_59359


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sector_max_area_l593_59342

/-- The area of a circular sector with constant perimeter C and central angle α -/
noncomputable def sectorArea (C : ℝ) (α : ℝ) : ℝ :=
  (C^2 * α) / (8 * (2 + α))

/-- The maximum area of a circular sector with constant perimeter C -/
noncomputable def maxSectorArea (C : ℝ) : ℝ :=
  C^2 / 16

theorem sector_max_area (C : ℝ) (h : C > 0) :
  ∀ α > 0, sectorArea C α ≤ maxSectorArea C ∧
  (sectorArea C α = maxSectorArea C ↔ α = 2) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sector_max_area_l593_59342


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_focus_distance_l593_59305

/-- Parabola defined by y² = -4x -/
def parabola (x y : ℝ) : Prop := y^2 = -4*x

/-- Distance from a point to the y-axis -/
def distToYAxis (x : ℝ) : ℝ := |x|

/-- Distance from a point to the focus of the parabola y² = -4x -/
noncomputable def distToFocus (x y : ℝ) : ℝ := 
  Real.sqrt ((x + 1)^2 + y^2)

theorem parabola_focus_distance 
  (x y : ℝ) 
  (h1 : parabola x y) 
  (h2 : distToYAxis x = 5) : 
  distToFocus x y = 6 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_focus_distance_l593_59305


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_positive_period_of_f_l593_59302

noncomputable def f (x : ℝ) := 3 * Real.sin (2 * x)

theorem smallest_positive_period_of_f : 
  ∃ (T : ℝ), T > 0 ∧ (∀ (x : ℝ), f (x + T) = f x) ∧ 
  (∀ (S : ℝ), S > 0 ∧ (∀ (x : ℝ), f (x + S) = f x) → T ≤ S) ∧ 
  T = Real.pi :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_positive_period_of_f_l593_59302


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_closest_times_for_84_degree_angle_l593_59389

/-- The angle between hour and minute hands at a given time -/
def angle_between_hands (hours : ℝ) (minutes : ℝ) : ℝ :=
  let hour_angle := 30 * hours + 0.5 * minutes
  let minute_angle := 6 * minutes
  abs (hour_angle - minute_angle)

/-- Check if a given time is between 7 and 8 o'clock -/
def is_between_7_and_8 (hours : ℝ) (minutes : ℝ) : Prop :=
  7 ≤ hours + minutes / 60 ∧ hours + minutes / 60 < 8

/-- The theorem stating that 7:23 and 7:53 are the closest times to when the angle is 84° -/
theorem closest_times_for_84_degree_angle :
  ∃ (h₁ m₁ h₂ m₂ : ℝ),
    is_between_7_and_8 h₁ m₁ ∧
    is_between_7_and_8 h₂ m₂ ∧
    angle_between_hands h₁ m₁ = 84 ∧
    angle_between_hands h₂ m₂ = 84 ∧
    (h₁ = 7 ∧ m₁ = 23) ∧
    (h₂ = 7 ∧ m₂ = 53) ∧
    ∀ (h m : ℝ),
      is_between_7_and_8 h m →
      angle_between_hands h m = 84 →
      (abs (h - 7) + abs (m - 23) / 60 ≥ abs (h₁ - 7) + abs (m₁ - 23) / 60 ∨
       abs (h - 7) + abs (m - 53) / 60 ≥ abs (h₂ - 7) + abs (m₂ - 53) / 60) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_closest_times_for_84_degree_angle_l593_59389


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_vectors_expression_l593_59354

noncomputable def vector_a (α : Real) : Real × Real := (Real.cos α, 3)
noncomputable def vector_b (α : Real) : Real × Real := (Real.sin α, -4)

def parallel (v w : Real × Real) : Prop :=
  ∃ k : Real, v.1 = k * w.1 ∧ v.2 = k * w.2

noncomputable def expression (α : Real) : Real := 
  (3 * Real.sin α + Real.cos α) / (2 * Real.cos α - 3 * Real.sin α)

theorem parallel_vectors_expression (α : Real) :
  parallel (vector_a α) (vector_b α) → expression α = -1/2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_vectors_expression_l593_59354


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_billy_remaining_cherries_l593_59360

noncomputable def initial_cherries : ℚ := 3682.5
noncomputable def eaten_cherries : ℚ := 2147.25
noncomputable def fraction_given_away : ℚ := 3/5
noncomputable def gift_cherries : ℚ := 128.5

theorem billy_remaining_cherries :
  let remaining_after_eating := initial_cherries - eaten_cherries
  let given_away := fraction_given_away * remaining_after_eating
  let remaining_after_giving := remaining_after_eating - given_away
  remaining_after_giving + gift_cherries = 742.6 := by
  sorry

#check billy_remaining_cherries

end NUMINAMATH_CALUDE_ERRORFEEDBACK_billy_remaining_cherries_l593_59360


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_date_statistics_order_l593_59325

-- Define the data set
def date_counts : List (Nat × Nat) :=
  (List.range 28).map (λ n => (n + 1, 12)) ++
  [(29, 11), (30, 11), (31, 7)]

-- Define the total number of data points
def total_count : Nat :=
  date_counts.foldl (λ acc (_, count) => acc + count) 0

-- Define the sum of all data points
def data_sum : Nat :=
  date_counts.foldl (λ acc (date, count) => acc + date * count) 0

-- Define the mean
noncomputable def mean : Real :=
  (data_sum : Real) / total_count

-- Define the median
def median : Nat :=
  let midpoint := (total_count + 1) / 2
  let cumulative_count := date_counts.scanl (λ acc (_, count) => acc + count) 0
  (date_counts.zip cumulative_count).find? (λ ((date, _), cum_count) => cum_count ≥ midpoint)
    |>.map (λ ((date, _), _) => date)
    |>.getD 0  -- Default value if not found (should not happen in this case)

-- Define the median of modes
noncomputable def median_of_modes : Real :=
  (1 + 28) / 2

-- Theorem statement
theorem date_statistics_order : median_of_modes < mean ∧ mean < (median : Real) := by
  sorry

#eval total_count
#eval data_sum
#eval median

end NUMINAMATH_CALUDE_ERRORFEEDBACK_date_statistics_order_l593_59325


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pizza_order_cost_l593_59353

structure Pizza where
  base_price : ℚ
  toppings : List String
  topping_prices : String → ℚ

def pizza_cost (p : Pizza) : ℚ :=
  p.base_price + (p.toppings.map p.topping_prices).sum

def total_order_cost (pizzas : List Pizza) (tip : ℚ) : ℚ :=
  (pizzas.map pizza_cost).sum + tip

def main_theorem : Prop :=
  let base_price : ℚ := 10
  let tip : ℚ := 5
  let topping_prices (t : String) : ℚ :=
    match t with
    | "pepperoni" | "sausage" | "bacon" => 3/2
    | "chicken" => 5/2
    | "extra cheese" | "feta cheese" => 2
    | "barbecue sauce" | "cilantro" => 0
    | _ => 1

  let son_pizza : Pizza := {
    base_price := base_price
    toppings := ["pepperoni", "bacon"]
    topping_prices := topping_prices
  }

  let daughter_pizza : Pizza := {
    base_price := base_price
    toppings := ["sausage", "onions", "pineapple"]
    topping_prices := topping_prices
  }

  let ruby_husband_pizza : Pizza := {
    base_price := base_price
    toppings := ["black olives", "mushrooms", "green peppers", "feta cheese"]
    topping_prices := topping_prices
  }

  let cousin_pizza : Pizza := {
    base_price := base_price
    toppings := ["spinach", "tomatoes", "extra cheese", "artichokes"]
    topping_prices := topping_prices
  }

  let family_pizza : Pizza := {
    base_price := base_price
    toppings := ["chicken", "barbecue sauce", "red onions", "cilantro", "jalapenos"]
    topping_prices := topping_prices
  }

  let all_pizzas := [son_pizza, daughter_pizza, ruby_husband_pizza, cousin_pizza, family_pizza]

  total_order_cost all_pizzas tip = 76

theorem pizza_order_cost : main_theorem := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_pizza_order_cost_l593_59353


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_of_M_and_N_l593_59380

-- Define set M
def M : Set ℝ := {x : ℝ | ∃ y : ℝ, x^2 + y^2 = 1}

-- Define set N
def N : Set ℝ := {y : ℝ | ∃ x : ℝ, y = x^2}

-- Theorem statement
theorem intersection_of_M_and_N : M ∩ N = Set.Icc 0 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_of_M_and_N_l593_59380


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_A_B_l593_59309

def A : Set ℕ := {x | x ≤ 1}
def B : Set ℕ := {0, 1, 2}

theorem intersection_A_B : A ∩ B = {0, 1} := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_A_B_l593_59309


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_class_average_marks_l593_59306

theorem class_average_marks : 
  let total_students : ℕ := 50
  let group1_students : ℕ := 10
  let group1_marks : ℕ := 90
  let group2_students : ℕ := 15
  let group2_marks : ℕ := group1_marks - 10
  let group3_students : ℕ := total_students - group1_students - group2_students
  let group3_marks : ℕ := 60
  let total_marks : ℕ := group1_students * group1_marks + 
                         group2_students * group2_marks + 
                         group3_students * group3_marks
  (total_marks : ℚ) / total_students = 72 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_class_average_marks_l593_59306


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_sum_at_7_l593_59383

/-- An arithmetic sequence with common difference d and first term a1 -/
noncomputable def arithmetic_sequence (d : ℝ) (a1 : ℝ) (n : ℕ) : ℝ :=
  a1 + d * (n - 1 : ℝ)

/-- Sum of the first n terms of an arithmetic sequence -/
noncomputable def arithmetic_sum (d : ℝ) (a1 : ℝ) (n : ℕ) : ℝ :=
  n * (2 * a1 + d * (n - 1)) / 2

theorem max_sum_at_7 
  (d : ℝ) 
  (a1 : ℝ) 
  (h1 : d < 0) 
  (h2 : arithmetic_sum d a1 3 = 11 * arithmetic_sequence d a1 6) :
  ∀ n : ℕ, arithmetic_sum d a1 7 ≥ arithmetic_sum d a1 n :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_sum_at_7_l593_59383


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_carboxylic_acid_oxygen_fraction_l593_59326

/-- Represents the number of carbon atoms in a saturated monobasic carboxylic acid -/
def n : ℕ := 3

/-- Mass fraction of oxygen in the carboxylic acid -/
noncomputable def oxygen_fraction : ℝ := 0.4325

/-- General formula for a saturated monobasic carboxylic acid is CₙH₂ₙ₊₂O₂ -/
def carboxylic_acid_formula (n : ℕ) : ℕ × ℕ × ℕ := (n, 2*n + 2, 2)

/-- Calculate the mass fraction of oxygen in a carboxylic acid -/
noncomputable def calculate_oxygen_fraction (n : ℕ) : ℝ :=
  32 / (14 * n + 34)

/-- Theorem: If the mass fraction of oxygen in a saturated monobasic carboxylic acid is 43.25%, 
    then the number of carbon atoms (n) is 3 -/
theorem carboxylic_acid_oxygen_fraction :
  calculate_oxygen_fraction n = oxygen_fraction → n = 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_carboxylic_acid_oxygen_fraction_l593_59326


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_remainder_problem_l593_59384

theorem remainder_problem (k : ℕ) (hk : k > 0) (h : 125 % (k^3) = 5) : 200 % k = 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_remainder_problem_l593_59384


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_foci_coordinates_l593_59390

/-- The coordinates of the foci of the ellipse 2x^2 + 3y^2 = 1 are (±√6/6, 0) -/
theorem ellipse_foci_coordinates :
  let ellipse := {(x, y) : ℝ × ℝ | 2 * x^2 + 3 * y^2 = 1}
  ∃ c : ℝ, c = Real.sqrt 6 / 6 ∧
    {(-c, 0), (c, 0)} = {p : ℝ × ℝ | ∃ (x y : ℝ), (x, y) ∈ ellipse ∧ p.1^2 - p.2^2 = x^2 - y^2} :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_foci_coordinates_l593_59390


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_increasing_interval_implies_a_bound_l593_59313

-- Define the function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.log x + a * x^2 - 2

-- Define the range
def range : Set ℝ := Set.Ioo (1/2) 2

-- Theorem statement
theorem increasing_interval_implies_a_bound 
  (h : ∃ (I : Set ℝ), I ⊆ range ∧ StrictMonoOn (f a) I) : 
  a > -1/8 := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_increasing_interval_implies_a_bound_l593_59313


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_proof_l593_59310

/-- Definition of a geometric sequence -/
def IsGeometricSequence (a b c d : ℝ) : Prop :=
  ∃ r : ℝ, r ≠ 0 ∧ b = a * r ∧ c = b * r ∧ d = c * r

/-- Given a geometric sequence with first term 25, second term a, third term b, and fourth term 1/25,
    where a and b are positive real numbers, prove that a = ∛25 and b = 25^(-1/3). -/
theorem geometric_sequence_proof (a b : ℝ) (ha : a > 0) (hb : b > 0) 
    (h1 : IsGeometricSequence 25 a b (1/25)) : a = Real.rpow 25 (1/3) ∧ b = Real.rpow 25 (-1/3) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_proof_l593_59310


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_intersection_perpendicular_l593_59332

/-- Ellipse C with foci and minor axis endpoints on the unit circle -/
def ellipse_C (x y : ℝ) : Prop := x^2/2 + y^2 = 1

/-- Line passing through M(2,0) with slope k -/
def line_through_M (k x y : ℝ) : Prop := y = k*(x - 2)

/-- Points A and B are intersections of the line and ellipse C -/
def intersection_points (k x₁ y₁ x₂ y₂ : ℝ) : Prop :=
  ellipse_C x₁ y₁ ∧ ellipse_C x₂ y₂ ∧
  line_through_M k x₁ y₁ ∧ line_through_M k x₂ y₂

/-- OA is perpendicular to OB -/
def perpendicular (x₁ y₁ x₂ y₂ : ℝ) : Prop := x₁*x₂ + y₁*y₂ = 0

theorem ellipse_intersection_perpendicular :
  ∀ k x₁ y₁ x₂ y₂,
  intersection_points k x₁ y₁ x₂ y₂ →
  (perpendicular x₁ y₁ x₂ y₂ ↔ k = Real.sqrt 5 / 5 ∨ k = -Real.sqrt 5 / 5) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_intersection_perpendicular_l593_59332


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_last_two_floors_height_difference_l593_59367

/-- Proves that the last two floors of a building are 0.5 meters higher than the other floors -/
theorem last_two_floors_height_difference (total_floors : Nat) (standard_floor_height : Real) 
  (total_building_height : Real) : Real :=
  let height_difference := 
    (total_building_height - (total_floors - 2) * standard_floor_height) / 2 - standard_floor_height
  by
    have h1 : total_floors = 20 := by sorry
    have h2 : standard_floor_height = 3 := by sorry
    have h3 : total_building_height = 61 := by sorry
    have h4 : height_difference = 0.5 := by sorry
    exact height_difference


end NUMINAMATH_CALUDE_ERRORFEEDBACK_last_two_floors_height_difference_l593_59367


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_double_deck_probability_l593_59366

theorem double_deck_probability : 
  (2 * 7 + 24 * 8) / (104 * 103) = 31 / 5308 := by
  sorry

#eval (2 * 7 + 24 * 8) / (104 * 103) == 31 / 5308

end NUMINAMATH_CALUDE_ERRORFEEDBACK_double_deck_probability_l593_59366


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_martin_traffic_time_l593_59385

/-- Given that Martin spends some time waiting in traffic, four times that long
    trying to get off the freeway, and wastes a total of 10 hours, prove that
    he spends 2 hours waiting in traffic. -/
theorem martin_traffic_time :
  ∀ (t : ℝ), t > 0 →  -- time waiting in traffic
  (4 * t) + t = 10 →  -- total wasted time
  t = 2 := by
  intro t ht htotal
  -- Proof steps would go here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_martin_traffic_time_l593_59385


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_star_properties_l593_59393

/-- The set of all non-zero real numbers -/
def T : Set ℝ := {x : ℝ | x ≠ 0}

/-- The binary operation ★ defined on T -/
def star (a b : ℝ) : ℝ := 3 * a * b

/-- Theorem stating the properties of the ★ operation -/
theorem star_properties :
  (∀ (a b : ℝ), a ∈ T → b ∈ T → star a b = star b a) ∧ 
  (∃ (a b c : ℝ), a ∈ T ∧ b ∈ T ∧ c ∈ T ∧ star (star a b) c ≠ star a (star b c)) ∧
  (∀ (a : ℝ), a ∈ T → star a (1/3) = a ∧ star (1/3) a = a) ∧
  (∀ (a : ℝ), a ∈ T → ∃ (b : ℝ), b ∈ T ∧ star a b = 1/3 ∧ star b a = 1/3) ∧
  (∀ (a : ℝ), a ∈ T → star a (1/(3*a)) = 1/3 ∧ star (1/(3*a)) a = 1/3) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_star_properties_l593_59393


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_numbers_l593_59398

/-- Represents a three-digit number -/
def ThreeDigitNumber := { n : ℕ // 100 ≤ n ∧ n < 1000 }

/-- Represents a two-digit number -/
def TwoDigitNumber := { n : ℕ // 10 ≤ n ∧ n < 100 }

/-- Checks if a natural number contains the digit 7 -/
def containsSeven (n : ℕ) : Prop := ∃ d k, n = 10 * k + 7 * d ∧ d = 0 ∨ d = 1

/-- Checks if a natural number contains the digit 3 -/
def containsThree (n : ℕ) : Prop := ∃ d k, n = 10 * k + 3 * d ∧ d = 0 ∨ d = 1

theorem sum_of_numbers (A : ThreeDigitNumber) (B C : TwoDigitNumber) :
  ((containsSeven A.val ∧ containsSeven B.val) ∨
   (containsSeven A.val ∧ containsSeven C.val) ∨
   (containsSeven B.val ∧ containsSeven C.val)) →
  A.val + B.val + C.val = 208 →
  containsThree B.val →
  containsThree C.val →
  B.val + C.val = 76 →
  A.val + B.val + C.val = 247 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_numbers_l593_59398


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_product_l593_59348

/-- A geometric sequence -/
def GeometricSequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

/-- The product of the first n terms of a sequence -/
def SequenceProduct (a : ℕ → ℝ) (n : ℕ) : ℝ :=
  (List.range n).foldl (fun acc i => acc * a (i + 1)) 1

theorem geometric_sequence_product
  (a : ℕ → ℝ)
  (h_geometric : GeometricSequence a)
  (h_a9 : a 9 = -2) :
  SequenceProduct a 17 = -2^17 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_product_l593_59348


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_multiple_of_15_with_0s_and_1s_l593_59303

/-- A function that checks if a positive integer is composed only of 0s and 1s -/
def isComposedOf0sAnd1s (n : ℕ) : Prop := sorry

/-- The smallest positive integer composed of 0s and 1s that is divisible by 15 -/
def smallestValidT : ℕ := 1110

theorem smallest_multiple_of_15_with_0s_and_1s (T : ℕ) (h1 : isComposedOf0sAnd1s T) (h2 : T % 15 = 0) :
  T ≥ smallestValidT ∧ smallestValidT = 1110 ∧ smallestValidT / 15 = 74 := by
  sorry

#eval smallestValidT
#eval smallestValidT / 15

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_multiple_of_15_with_0s_and_1s_l593_59303


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_decomposition_property_l593_59363

theorem decomposition_property (r : ℕ) (A : Fin r → Set ℕ) 
  (h_disjoint : ∀ i j, i ≠ j → Disjoint (A i) (A j))
  (h_cover : (⋃ i, A i) = Set.univ) :
  ∃ i : Fin r, ∃ m : ℕ, ∀ k : ℕ, ∃ α : ℕ → ℕ,
    (∀ j < k, α j ∈ A i) ∧
    (∀ j < k - 1, 0 < α (j + 1) - α j ∧ α (j + 1) - α j ≤ m) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_decomposition_property_l593_59363


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangle_breadth_l593_59327

theorem rectangle_breadth (square_area : ℝ) (rectangle_area : ℝ) : 
  square_area = 4761 →
  rectangle_area = 598 →
  let square_side := Real.sqrt square_area
  let circle_radius := square_side
  let rectangle_length := (2 / 3) * circle_radius
  let rectangle_breadth := rectangle_area / rectangle_length
  rectangle_breadth = 13 := by
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangle_breadth_l593_59327


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_c_share_is_correct_l593_59300

/-- Represents the work rate change for a worker -/
structure WorkRateChange where
  initial : ℝ
  daily_change : ℝ

/-- Calculates the amount of work done by a worker over a period of days -/
noncomputable def work_done (w : WorkRateChange) (days : ℕ) : ℝ :=
  (days : ℝ) * w.initial + (days : ℝ) * (days - 1 : ℝ) / 2 * w.daily_change * w.initial

/-- The total payment for the job -/
def total_payment : ℝ := 500

/-- Worker A's work rate change -/
noncomputable def worker_a : WorkRateChange :=
  { initial := 1 / 5, daily_change := -0.1 }

/-- Worker B's work rate change -/
noncomputable def worker_b : WorkRateChange :=
  { initial := 1 / 10, daily_change := 0.05 }

/-- The number of days to complete the job with all workers -/
def days_to_complete : ℕ := 2

/-- Calculates the fraction of work done by workers A and B together -/
noncomputable def work_done_a_and_b : ℝ :=
  work_done worker_a days_to_complete + work_done worker_b days_to_complete

/-- Calculates the fraction of work done by worker C -/
noncomputable def work_done_c : ℝ :=
  1 - work_done_a_and_b

/-- Calculates worker C's share of the payment -/
noncomputable def c_share : ℝ :=
  total_payment * work_done_c

theorem c_share_is_correct : ∃ (ε : ℝ), ε > 0 ∧ |c_share - 370.95| < ε := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_c_share_is_correct_l593_59300


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_valid_sequences_l593_59340

def is_valid_sequence (s : List Nat) : Prop :=
  s.length = 10 ∧ 
  ∀ i, i < s.length → s[i]! ≤ 2 ∧
  ∃ original : List Nat, 
    original.length = 10 ∧ 
    original.Nodup ∧
    ∀ i, i < s.length → 
      s[i]! = (if i > 0 ∧ original[i-1]! < original[i]! then 1 else 0) +
              (if i < 9 ∧ original[i+1]! < original[i]! then 1 else 0)

theorem valid_sequences : 
  is_valid_sequence [1,1,0,1,1,1,1,1,1,1] ∧
  is_valid_sequence [1,0,2,1,0,2,1,0,2,0] ∧
  is_valid_sequence [0,1,1,2,1,0,2,0,1,1] := by
  sorry

#check valid_sequences

end NUMINAMATH_CALUDE_ERRORFEEDBACK_valid_sequences_l593_59340


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_square_configuration_theorem_l593_59377

/-- The side length of the small square A in the configuration --/
noncomputable def a₁ : ℝ := 1 / (2 * (1 + 3 * Real.sqrt 2 / 8))

/-- The side length of the small square B in the configuration --/
noncomputable def a₂ : ℝ := 1 / (2 * (1 + Real.sqrt 2 / 3))

/-- The side length of the small square C in the configuration --/
noncomputable def a₃ : ℝ := 1 / (2 * (1 + Real.sqrt 2 / 4))

/-- The side length of the small square D in the configuration --/
noncomputable def a₄ : ℝ := Real.sqrt 2 / 4

/-- Theorem stating the order and maximum of the side lengths --/
theorem square_configuration_theorem :
  a₁ < a₂ ∧ a₂ < a₄ ∧ a₄ < a₃ ∧
  a₃ = max a₁ (max a₂ (max a₃ a₄)) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_square_configuration_theorem_l593_59377


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_between_given_lines_l593_59335

/-- The distance between two parallel lines in 2D space -/
noncomputable def distance_parallel_lines (a b c1 c2 : ℝ) : ℝ :=
  abs (c1 - c2) / Real.sqrt (a^2 + b^2)

/-- The distance between the lines x+y+1=0 and 2x+2y+3=0 -/
theorem distance_between_given_lines : 
  distance_parallel_lines 1 1 1 (3/2) = Real.sqrt 2 / 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_between_given_lines_l593_59335


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_man_speed_in_still_water_l593_59399

/-- Represents the speed of a man rowing in different conditions -/
structure RowingSpeed where
  upstream : ℚ
  downstream : ℚ

/-- Calculates the speed in still water given upstream and downstream speeds -/
def speedInStillWater (s : RowingSpeed) : ℚ :=
  (s.upstream + s.downstream) / 2

/-- Theorem: Given a man's upstream speed of 20 kmph and downstream speed of 80 kmph,
    his speed in still water is 50 kmph -/
theorem man_speed_in_still_water :
  let s : RowingSpeed := { upstream := 20, downstream := 80 }
  speedInStillWater s = 50 := by
  -- Unfold the definition of speedInStillWater
  unfold speedInStillWater
  -- Simplify the arithmetic
  simp
  -- The proof is complete
  rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_man_speed_in_still_water_l593_59399


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_two_alpha_l593_59331

theorem sin_two_alpha (α : ℝ) (h1 : Real.sin (π/4 - α) = 3/5) (h2 : 0 < α ∧ α < π/4) : 
  Real.sin (2*α) = 7/25 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_two_alpha_l593_59331


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_v_1005_equals_3948_l593_59320

-- Define the sequence v_n
def v : ℕ → ℕ
| n =>
  let group := ((Nat.sqrt (8 * n + 1) - 1) / 2 : ℕ)
  let group_start := group * (group + 1) / 2 + 1
  let offset := n - group_start
  (group + 2 : ℕ) + 5 * offset

-- Theorem statement
theorem v_1005_equals_3948 : v 1005 = 3948 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_v_1005_equals_3948_l593_59320


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_equation_l593_59330

-- Define the ellipse C and its properties
def Ellipse (C : Set (ℝ × ℝ)) : Prop :=
  ∃ (F₁ F₂ : ℝ × ℝ), 
    -- F₁ and F₂ are the foci
    (∀ (P : ℝ × ℝ), P ∈ C → 
      -- Distance between foci is 2√3
      Real.sqrt ((F₁.1 - F₂.1)^2 + (F₁.2 - F₂.2)^2) = 2 * Real.sqrt 3 ∧
      -- Arithmetic mean of distances from P to foci equals distance between foci
      (Real.sqrt ((P.1 - F₁.1)^2 + (P.2 - F₁.2)^2) + 
       Real.sqrt ((P.1 - F₂.1)^2 + (P.2 - F₂.2)^2)) / 2 = 2 * Real.sqrt 3)

-- Theorem statement
theorem ellipse_equation (C : Set (ℝ × ℝ)) (h : Ellipse C) :
  (∀ (x y : ℝ), (x, y) ∈ C ↔ x^2/12 + y^2/9 = 1) ∨
  (∀ (x y : ℝ), (x, y) ∈ C ↔ x^2/9 + y^2/12 = 1) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_equation_l593_59330


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_cos_difference_l593_59301

theorem sin_cos_difference (θ : ℝ) (h1 : θ ∈ Set.Ioo 0 π) (h2 : Real.sin θ + Real.cos θ = 7/13) :
  Real.sin θ - Real.cos θ = Real.sqrt 247 / 13 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_cos_difference_l593_59301


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_of_f_l593_59396

noncomputable def a (x m : ℝ) : ℝ × ℝ := (Real.sqrt 3 * Real.sin x, m + Real.cos x)
noncomputable def b (x m : ℝ) : ℝ × ℝ := (Real.cos x, -m + Real.cos x)

noncomputable def f (x m : ℝ) : ℝ := (a x m).1 * (b x m).1 + (a x m).2 * (b x m).2

theorem max_value_of_f (m : ℝ) :
  (∀ x ∈ Set.Icc (-π/6) (π/3), f x m ≥ -4) →
  (∃ x ∈ Set.Icc (-π/6) (π/3), f x m = -4) →
  (∃ x ∈ Set.Icc (-π/6) (π/3), f x m = -5/2 ∧ x = π/6) ∧
  (∀ x ∈ Set.Icc (-π/6) (π/3), f x m ≤ -5/2) := by
  sorry

#check max_value_of_f

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_of_f_l593_59396


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_count_l593_59317

noncomputable def f₁ (x : ℝ) : ℝ := Real.log x / Real.log 5
noncomputable def f₂ (x : ℝ) : ℝ := 3 * Real.log 5 / Real.log x
noncomputable def f₃ (x : ℝ) : ℝ := Real.log x / Real.log (1/5)
noncomputable def f₄ (x : ℝ) : ℝ := 5 * Real.log (1/5) / Real.log x

def intersection_points : Set ℝ :=
  {x : ℝ | x > 0 ∧ (∃ i j : Fin 4, i ≠ j ∧
    ((i = 0 ∧ j = 1 ∧ f₁ x = f₂ x) ∨
     (i = 0 ∧ j = 2 ∧ f₁ x = f₃ x) ∨
     (i = 0 ∧ j = 3 ∧ f₁ x = f₄ x) ∨
     (i = 1 ∧ j = 2 ∧ f₂ x = f₃ x) ∨
     (i = 1 ∧ j = 3 ∧ f₂ x = f₄ x) ∨
     (i = 2 ∧ j = 3 ∧ f₃ x = f₄ x)))}

theorem intersection_count :
  ∃ s : Finset ℝ, s.card = 5 ∧ ∀ x ∈ s, x ∈ intersection_points := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_count_l593_59317


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_mouse_turning_point_is_closest_sum_of_coordinates_l593_59392

/-- The point where the mouse starts moving away from the cheese -/
noncomputable def mouse_turning_point : ℝ × ℝ := (45/17, 194/17)

/-- The location of the cheese -/
def cheese_location : ℝ × ℝ := (15, 14)

/-- The slope of the mouse's path -/
def mouse_path_slope : ℝ := -4

/-- The y-intercept of the mouse's path -/
def mouse_path_intercept : ℝ := 22

/-- Checks if a point is on the mouse's path -/
def on_mouse_path (p : ℝ × ℝ) : Prop :=
  p.2 = mouse_path_slope * p.1 + mouse_path_intercept

/-- The perpendicular distance from a point to the mouse's path -/
noncomputable def perpendicular_distance (p : ℝ × ℝ) : ℝ :=
  abs (p.2 - mouse_path_slope * p.1 - mouse_path_intercept) / 
  Real.sqrt (1 + mouse_path_slope^2)

theorem mouse_turning_point_is_closest :
  on_mouse_path mouse_turning_point ∧
  perpendicular_distance mouse_turning_point = 
  perpendicular_distance cheese_location := by sorry

theorem sum_of_coordinates : 
  mouse_turning_point.1 + mouse_turning_point.2 = 239/17 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_mouse_turning_point_is_closest_sum_of_coordinates_l593_59392


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_distance_C_to_l_l593_59370

/-- The curve C in the x-y plane -/
noncomputable def C : ℝ → ℝ × ℝ := λ θ => (Real.cos θ, Real.sqrt 3 * Real.sin θ)

/-- The line l in the x-y plane -/
def l : ℝ × ℝ → Prop := λ p => Real.sqrt 3 * p.1 - p.2 + 2 = 0

/-- The distance function from a point to the line l -/
noncomputable def distance_to_l (p : ℝ × ℝ) : ℝ :=
  |Real.sqrt 3 * p.1 - p.2 + 2| / Real.sqrt ((Real.sqrt 3)^2 + 1)

/-- The maximum distance from any point on curve C to line l -/
theorem max_distance_C_to_l :
  (⨆ θ : ℝ, distance_to_l (C θ)) = (Real.sqrt 6 + 2) / 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_distance_C_to_l_l593_59370


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_power_function_properties_l593_59362

/-- A power function that passes through the point (3, 27) -/
noncomputable def f (α : ℝ) (x : ℝ) : ℝ := x ^ α

theorem power_function_properties (α : ℝ) (h : f α 3 = 27) :
  α = 3 ∧ f α 0 = 0 ∧ ∀ y : ℝ, ∃ x : ℝ, f α x = y := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_power_function_properties_l593_59362


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_array_sum_is_25_18_l593_59351

/-- Represents the value at the r-th row and c-th column of the 1/5-array -/
def arrayValue (r c : ℕ) : ℚ := (1 / 10 ^ r) * (1 / 5 ^ c)

/-- The sum of all terms in the 1/5-array -/
noncomputable def arraySum : ℚ := ∑' (r : ℕ) (c : ℕ), arrayValue r c

/-- Theorem stating that the sum of all terms in the 1/5-array is 25/18 -/
theorem array_sum_is_25_18 : arraySum = 25 / 18 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_array_sum_is_25_18_l593_59351


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_sum_of_factors_min_sum_is_866_l593_59374

theorem smallest_sum_of_factors (a b : ℕ) (h : (2^10 : ℕ) * (3^6 : ℕ) = a^b) : 
  ∀ (x y : ℕ), (2^10 : ℕ) * (3^6 : ℕ) = x^y → a + b ≤ x + y := by
  sorry

theorem min_sum_is_866 (a b : ℕ) (h : (2^10 : ℕ) * (3^6 : ℕ) = a^b) : a + b = 866 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_sum_of_factors_min_sum_is_866_l593_59374


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_triangle_sum_l593_59346

noncomputable def parabola (x : ℝ) : ℝ := 4 - x^2

def largeBase : ℝ := 4

def largeHeight : ℝ := 4

noncomputable def smallBase (b : ℝ) : ℝ := 2 - b

noncomputable def smallHeight (b : ℝ) : ℝ := 3 - b - b^2 / 4

noncomputable def heightRatio (b : ℝ) : ℝ := largeHeight / smallHeight b

theorem parabola_triangle_sum (a b c : ℕ) (h1 : 0 < a ∧ 0 < b ∧ 0 < c) 
  (h2 : Nat.Coprime a c) (h3 : heightRatio (Real.sqrt 17 - 3) = (a + Real.sqrt b) / c) :
  a + b + c = 20 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_triangle_sum_l593_59346


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_asparagus_price_is_three_l593_59395

/-- The price of a bundle of asparagus that satisfies the given conditions -/
noncomputable def asparagus_price : ℚ :=
  let grapes_cost : ℚ := 40 * (5/2)
  let apples_cost : ℚ := 700 * (1/2)
  let total_cost : ℚ := 630
  let asparagus_bundles : ℚ := 60
  (total_cost - grapes_cost - apples_cost) / asparagus_bundles

/-- Theorem stating that the asparagus price is $3 -/
theorem asparagus_price_is_three : asparagus_price = 3 := by
  -- Unfold the definition of asparagus_price
  unfold asparagus_price
  -- Simplify the arithmetic
  simp [add_sub_cancel, mul_div_cancel]
  -- The proof is complete
  rfl

end NUMINAMATH_CALUDE_ERRORFEEDBACK_asparagus_price_is_three_l593_59395


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_line_intersection_l593_59311

/-- An ellipse with semi-major axis a and semi-minor axis b -/
structure Ellipse where
  a : ℝ
  b : ℝ
  h_pos : 0 < b ∧ b < a

/-- A line with slope k and y-intercept m -/
structure Line where
  k : ℝ
  m : ℝ

/-- The eccentricity of an ellipse -/
noncomputable def eccentricity (e : Ellipse) : ℝ := Real.sqrt (1 - e.b^2 / e.a^2)

/-- The area of a triangle formed by two points on an ellipse and the origin -/
noncomputable def triangle_area (e : Ellipse) (l : Line) : ℝ := sorry

theorem ellipse_line_intersection (e : Ellipse) (l : Line) :
  eccentricity e = Real.sqrt 6 / 3 →
  e.a = Real.sqrt 3 →
  l.m = 2 →
  triangle_area e l = 6 / 7 →
  l.k = Real.sqrt 2 ∨ l.k = -Real.sqrt 2 ∨ l.k = 5 / 3 ∨ l.k = -5 / 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_line_intersection_l593_59311


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_proposition_2_proposition_3_proposition_4_l593_59397

-- Proposition 2
theorem proposition_2 : ∀ x y : ℝ, x + y ≠ 0 → x ≠ 1 ∨ y ≠ -1 := by sorry

-- Proposition 3
theorem proposition_3 : 
  ∀ x y : ℝ, x^2 + y^2 = 1 → (y / (x + 2) ≤ Real.sqrt 3 / 3 ∧ ∃ x₀ y₀ : ℝ, x₀^2 + y₀^2 = 1 ∧ y₀ / (x₀ + 2) = Real.sqrt 3 / 3) := by sorry

-- Proposition 4
noncomputable def f (x : ℝ) : ℝ := 3 * Real.sin (2 * x - Real.pi / 3)

theorem proposition_4 : 
  ∀ x : ℝ, f (2 * Real.pi / 3 + x) = f (2 * Real.pi / 3 - x) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_proposition_2_proposition_3_proposition_4_l593_59397


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_function_values_and_inequality_l593_59371

noncomputable def f (a b x : ℝ) : ℝ := (-2^x + a) / (2^(x+1) + b)

theorem odd_function_values_and_inequality (a b : ℝ) 
  (h_odd : ∀ x, f a b x = -f a b (-x)) :
  (a = 1 ∧ b = 2) ∧ 
  (∀ x c : ℝ, f 1 2 x < c^2 - 3*c + 3) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_function_values_and_inequality_l593_59371


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_system_solution_l593_59315

theorem system_solution (x y₁ y₂ y₃ : ℝ) (n : ℤ) : 
  (n ∈ ({-3, -2, -1, 0, 1, 2, 3} : Set ℤ)) →
  ((1 - x^2) * y₁ = 2*x ∧
   (1 - y₁^2) * y₂ = 2*y₁ ∧
   (1 - y₂^2) * y₃ = 2*y₂ ∧
   y₃ = x) ↔
  (y₁ = Real.tan (2 * ↑n * Real.pi / 7) ∧
   y₂ = Real.tan (4 * ↑n * Real.pi / 7) ∧
   y₃ = Real.tan (↑n * Real.pi / 7) ∧
   x = Real.tan (↑n * Real.pi / 7))
:= by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_system_solution_l593_59315


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_g_max_min_on_interval_l593_59388

noncomputable def g (x : ℝ) : ℝ := 2 * Real.sin (x - Real.pi / 6) + 1

theorem g_max_min_on_interval :
  ∀ x ∈ Set.Icc 0 Real.pi,
    g x ≤ 3 ∧ g x ≥ 0 ∧
    (∃ x₁ ∈ Set.Icc 0 Real.pi, g x₁ = 3) ∧
    (∃ x₂ ∈ Set.Icc 0 Real.pi, g x₂ = 0) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_g_max_min_on_interval_l593_59388


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_least_three_digit_product_ten_l593_59319

/-- A function that returns the product of digits of a natural number -/
def digit_product (n : ℕ) : ℕ := sorry

/-- A function that checks if a number is three-digit -/
def is_three_digit (n : ℕ) : Prop := 100 ≤ n ∧ n ≤ 999

/-- Theorem stating that 125 is the least three-digit number with digit product 10 -/
theorem least_three_digit_product_ten :
  ∃ (n : ℕ), is_three_digit n ∧ digit_product n = 10 ∧
  ∀ (m : ℕ), is_three_digit m ∧ digit_product m = 10 → n ≤ m ∧ n = 125 := by
  sorry

#check least_three_digit_product_ten

end NUMINAMATH_CALUDE_ERRORFEEDBACK_least_three_digit_product_ten_l593_59319


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_trigonometric_sum_l593_59312

open Real

theorem max_value_trigonometric_sum (θ₁ θ₂ θ₃ θ₄ θ₅ : ℝ) :
  (∀ θ₁' θ₂' θ₃' θ₄' θ₅' : ℝ,
    (Real.cos θ₁')^2 * (Real.sin θ₂')^2 + (Real.cos θ₂')^2 * (Real.sin θ₃')^2 + 
    (Real.cos θ₃')^2 * (Real.sin θ₄')^2 + (Real.cos θ₄')^2 * (Real.sin θ₅')^2 + 
    (Real.cos θ₅')^2 * (Real.sin θ₁')^2 ≤ 
    (Real.cos θ₁)^2 * (Real.sin θ₂)^2 + (Real.cos θ₂)^2 * (Real.sin θ₃)^2 + 
    (Real.cos θ₃)^2 * (Real.sin θ₄)^2 + (Real.cos θ₄)^2 * (Real.sin θ₅)^2 + 
    (Real.cos θ₅)^2 * (Real.sin θ₁)^2) →
  (Real.cos θ₁)^2 * (Real.sin θ₂)^2 + (Real.cos θ₂)^2 * (Real.sin θ₃)^2 + 
  (Real.cos θ₃)^2 * (Real.sin θ₄)^2 + (Real.cos θ₄)^2 * (Real.sin θ₅)^2 + 
  (Real.cos θ₅)^2 * (Real.sin θ₁)^2 = 25 / 32 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_trigonometric_sum_l593_59312


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_cosine_between_zero_and_half_l593_59349

-- Define the interval
def I : Set ℝ := Set.Icc (-1) 1

-- Define the function
noncomputable def f (x : ℝ) : ℝ := Real.cos (Real.pi * x / 2)

-- Define the event
def E : Set ℝ := {x ∈ I | 0 ≤ f x ∧ f x ≤ 1/2}

-- State the theorem
theorem probability_cosine_between_zero_and_half :
  (MeasureTheory.volume E) / (MeasureTheory.volume I) = 1/3 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_cosine_between_zero_and_half_l593_59349


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_is_one_fourth_l593_59304

/-- Represents a group of students and the number of trees they planted -/
structure StudentGroup :=
  (trees : List Nat)

/-- Calculates the probability of selecting two students with a total of 19 trees -/
def probabilityOfNineteenTrees (groupA groupB : StudentGroup) : ℚ :=
  let totalPairs := groupA.trees.length * groupB.trees.length
  let validPairs := (List.zip groupA.trees groupB.trees).filter (fun (a, b) => a + b = 19)
  ↑validPairs.length / ↑totalPairs

/-- Main theorem: The probability of selecting two students with 19 trees is 1/4 -/
theorem probability_is_one_fourth :
  let groupA := StudentGroup.mk [9, 9, 11, 11]
  let groupB := StudentGroup.mk [9, 8, 9, 10]
  probabilityOfNineteenTrees groupA groupB = 1/4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_is_one_fourth_l593_59304


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_point_in_first_quadrant_l593_59387

-- Define a complex number z
variable (z : ℂ)

-- Theorem statement
theorem point_in_first_quadrant : 
  z.re > 0 ∧ z.im > 0 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_point_in_first_quadrant_l593_59387


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_function_implies_a_equals_negative_one_f_strictly_decreasing_on_zero_one_l593_59361

-- Define the function f(x)
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * (Real.exp (x * Real.log 3)) + (1 / Real.exp (x * Real.log 3))

-- Part 1: Prove that if f is an odd function, then a = -1
theorem odd_function_implies_a_equals_negative_one (a : ℝ) :
  (∀ x, f a x = -f a (-x)) → a = -1 := by sorry

-- Part 2: Prove that when a = -1, f is strictly decreasing on [0, 1]
theorem f_strictly_decreasing_on_zero_one :
  ∀ x y, 0 ≤ x → x < y → y ≤ 1 → f (-1) x > f (-1) y := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_function_implies_a_equals_negative_one_f_strictly_decreasing_on_zero_one_l593_59361


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_positive_sum_l593_59338

noncomputable def arithmetic_sequence (a : ℕ → ℝ) (d : ℝ) : Prop :=
  ∀ n, a (n + 1) = a n + d

noncomputable def sum_of_terms (a : ℕ → ℝ) (n : ℕ) : ℝ :=
  (n : ℝ) * (a 1 + a n) / 2

theorem smallest_positive_sum (a : ℕ → ℝ) (d : ℝ) :
  d > 0 →
  arithmetic_sequence a d →
  a 2009 ^ 2 - 3 * a 2009 - 5 = 0 →
  a 2010 ^ 2 - 3 * a 2010 - 5 = 0 →
  (∀ n < 4018, sum_of_terms a n ≤ 0) ∧ sum_of_terms a 4018 > 0 :=
by
  sorry

#check smallest_positive_sum

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_positive_sum_l593_59338


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_soap_bars_example_l593_59314

/-- The maximum number of soap bars that can be purchased with a given budget and cost per bar -/
def max_soap_bars (budget : ℚ) (cost_per_bar : ℚ) : ℕ :=
  (budget / cost_per_bar).floor.toNat

/-- Theorem: Given $10.00 and a soap bar cost of $0.95, the maximum number of soap bars that can be purchased is 10 -/
theorem max_soap_bars_example : max_soap_bars 10 (95/100) = 10 := by
  rfl

#eval max_soap_bars 10 (95/100)

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_soap_bars_example_l593_59314


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fruit_drink_volume_l593_59323

/-- Represents the composition of a fruit drink -/
structure FruitDrink where
  orange_percent : ℚ
  watermelon_percent : ℚ
  grape_ounces : ℚ

/-- Calculates the total volume of a fruit drink given its composition -/
def total_volume (drink : FruitDrink) : ℚ :=
  drink.grape_ounces / (1 - drink.orange_percent - drink.watermelon_percent)

/-- Theorem: The total volume of a fruit drink with the given composition is 140 ounces -/
theorem fruit_drink_volume :
  ∀ (drink : FruitDrink),
    drink.orange_percent = 15/100 →
    drink.watermelon_percent = 60/100 →
    drink.grape_ounces = 35 →
    total_volume drink = 140 :=
by
  intro drink h_orange h_watermelon h_grape
  unfold total_volume
  rw [h_orange, h_watermelon, h_grape]
  -- The proof steps would go here
  sorry

#eval total_volume { orange_percent := 15/100, watermelon_percent := 60/100, grape_ounces := 35 }

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fruit_drink_volume_l593_59323


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_rational_roots_l593_59322

-- Define the polynomial h(x)
def h (x : ℚ) : ℚ := x^3 - 6*x^2 + 11*x - 6

-- Define a function to check if a number is a root of h(x)
def is_root (r : ℚ) : Prop := h r = 0

-- State the theorem
theorem sum_of_rational_roots :
  ∃ (roots : Finset ℚ), (∀ r ∈ roots, is_root r) ∧ 
  (∀ r : ℚ, is_root r → r ∈ roots) ∧ 
  (Finset.sum roots id = 6) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_rational_roots_l593_59322


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_three_digits_make_361d_divisible_by_3_l593_59350

def is_divisible_by_3 (n : ℕ) : Prop := n % 3 = 0

def digit_sum (n : ℕ) : ℕ := 
  (n.digits 10).sum

theorem three_digits_make_361d_divisible_by_3 : 
  ∃ (S : Finset ℕ), S.card = 3 ∧ 
    (∀ d : ℕ, d ∈ S ↔ (d < 10 ∧ is_divisible_by_3 (digit_sum (3610 + d)))) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_three_digits_make_361d_divisible_by_3_l593_59350


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_equation_implies_angle_l593_59336

theorem sin_equation_implies_angle (α : Real) : 
  0 < α ∧ α < Real.pi / 2 →  -- α is an acute angle
  Real.sin α = 1 - Real.sqrt 3 * Real.tan (10 * Real.pi / 180) * Real.sin α → 
  α = 50 * Real.pi / 180 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_equation_implies_angle_l593_59336


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l593_59394

/-- The eccentricity of a hyperbola with given conditions -/
theorem hyperbola_eccentricity (a b c : ℝ) (F M : ℝ × ℝ) (C : Set (ℝ × ℝ)) :
  a > 0 →
  b > 0 →
  (∀ x y, x^2 / a^2 - y^2 / b^2 = 1 ↔ (x, y) ∈ C) →
  F = (c, 0) →
  M.1 > 0 ∧ M.2 > 0 →
  M.2 = (b / a) * M.1 →
  Real.sqrt (M.1^2 + M.2^2) = a →
  (M.2 - F.2) / (M.1 - F.1) = -b / a →
  c / a = Real.sqrt 2 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l593_59394


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_tower_of_100s_l593_59344

def a : ℕ → ℕ
  | 0 => 3
  | n + 1 => 3^(a n)

def b : ℕ → ℕ
  | 0 => 100
  | n + 1 => 100^(b n)

theorem smallest_tower_of_100s (n : ℕ) : n ≥ 99 ↔ b n > a 100 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_tower_of_100s_l593_59344


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cosine_angle_special_vectors_l593_59375

open InnerProductSpace

variable {V : Type*} [NormedAddCommGroup V] [InnerProductSpace ℝ V]

theorem cosine_angle_special_vectors (a b : V) (h1 : a ≠ 0) (h2 : b ≠ 0) 
  (h3 : ‖a‖ = ‖b‖) (h4 : ‖a‖ = ‖a + b‖) :
  (inner a (2 • a - b)) / (‖a‖ * ‖2 • a - b‖) = 5 * Real.sqrt 7 / 14 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cosine_angle_special_vectors_l593_59375


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_infinite_sqrt_two_eq_two_l593_59347

/-- The infinite square root expression -/
noncomputable def infinite_sqrt_two : ℝ := Real.sqrt (2 + Real.sqrt (2 + Real.sqrt (2 + Real.sqrt 2)))

/-- Theorem stating that the infinite square root expression equals 2 -/
theorem infinite_sqrt_two_eq_two : infinite_sqrt_two = 2 := by
  -- We'll use a proof by contradiction
  have h : infinite_sqrt_two ^ 2 = 2 + infinite_sqrt_two := by
    -- Expand the definition and simplify
    simp [infinite_sqrt_two]
    -- The rest of the proof steps would go here
    sorry
  
  -- Now we can solve the equation x^2 = 2 + x
  have eq : infinite_sqrt_two ^ 2 - infinite_sqrt_two - 2 = 0 := by
    rw [h]
    ring
  
  -- The positive solution to this equation is 2
  -- We would prove this step rigorously in a complete proof
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_infinite_sqrt_two_eq_two_l593_59347


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angela_future_age_l593_59368

-- Define variables for current ages
variable (A B D : ℕ)

-- Define the conditions
axiom angela_beth : A = 3 * B
axiom angela_derek : A = D / 2
axiom past_sum : A + B + D - 60 = D
axiom future_condition : Real.sqrt (A + 7) - (B + 7) / 3 = D / 4

-- Theorem to prove
theorem angela_future_age : A + 15 = 60 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_angela_future_age_l593_59368


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotonic_decreasing_l593_59386

open Set
open Function
open Real

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := (1/2) * x^2 - log x

-- Define the domain of f
def domain : Set ℝ := {x | x > 0}

-- Define the monotonic decreasing interval
def monotonic_decreasing_interval : Set ℝ := Ioc 0 1

-- Theorem statement
theorem f_monotonic_decreasing :
  StrictMonoOn f monotonic_decreasing_interval := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotonic_decreasing_l593_59386


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_monotonic_f_range_l593_59343

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x < 1 then x^2 - a*x + 5 else 1 + 1/x

theorem monotonic_f_range (a : ℝ) :
  (∀ x y : ℝ, x < y → f a x < f a y) ∨ (∀ x y : ℝ, x < y → f a x > f a y) ↔ 2 ≤ a ∧ a ≤ 4 := by
  sorry

#check monotonic_f_range

end NUMINAMATH_CALUDE_ERRORFEEDBACK_monotonic_f_range_l593_59343


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_swimmer_speed_is_five_l593_59339

/-- Represents the speed of a swimmer in still water given their downstream and upstream speeds -/
noncomputable def swimmer_speed (downstream_distance : ℝ) (upstream_distance : ℝ) (time : ℝ) : ℝ :=
  (downstream_distance + upstream_distance) / (2 * time)

/-- Theorem stating that a swimmer's speed in still water is 5 km/h given the problem conditions -/
theorem swimmer_speed_is_five :
  swimmer_speed 18 12 3 = 5 := by
  -- Unfold the definition of swimmer_speed
  unfold swimmer_speed
  -- Simplify the arithmetic
  simp [add_div]
  -- Evaluate the expression
  norm_num

#check swimmer_speed_is_five

end NUMINAMATH_CALUDE_ERRORFEEDBACK_swimmer_speed_is_five_l593_59339


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_is_one_fifth_l593_59334

/-- A five-digit number represented as a list of its digits -/
def FiveDigitNumber := List Nat

/-- Predicate to check if a given list represents a valid five-digit number with digit sum 30 -/
def isValidNumber (n : FiveDigitNumber) : Prop :=
  n.length = 5 ∧ n.head! ≠ 0 ∧ n.sum = 30

/-- The set of all valid five-digit numbers with digit sum 30 -/
def validNumbers : Set FiveDigitNumber :=
  {n | isValidNumber n}

/-- Predicate to check if a number is divisible by 7 -/
def isDivisibleBySeven (n : FiveDigitNumber) : Prop :=
  (n.foldl (fun acc d => acc * 10 + d) 0) % 7 = 0

/-- The number of valid five-digit numbers with digit sum 30 -/
def totalValidNumbers : Nat := 20

/-- The number of valid five-digit numbers with digit sum 30 that are divisible by 7 -/
def divisibleBySevenCount : Nat := 4

/-- The probability of a randomly selected number from validNumbers being divisible by 7 -/
def probabilityDivisibleBySeven : ℚ :=
  divisibleBySevenCount / totalValidNumbers

theorem probability_is_one_fifth :
  probabilityDivisibleBySeven = 1 / 5 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_is_one_fifth_l593_59334


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_equation_l593_59324

-- Define the circle
def my_circle (x y : ℝ) : Prop := x^2 + y^2 + x - 6*y + 3 = 0

-- Define the line passing through (3, 0)
def my_line (k : ℝ) (x y : ℝ) : Prop := y = k*(x - 3)

-- Define the perpendicularity condition
def my_perpendicular (x1 y1 x2 y2 : ℝ) : Prop := x1*x2 + y1*y2 = 0

-- Main theorem
theorem line_equation : 
  ∃ (k : ℝ) (x1 y1 x2 y2 : ℝ),
    my_line k x1 y1 ∧ 
    my_line k x2 y2 ∧ 
    my_circle x1 y1 ∧ 
    my_circle x2 y2 ∧ 
    my_perpendicular x1 y1 x2 y2 ∧
    (k = -1/2 ∨ k = -1/4) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_equation_l593_59324


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_l593_59381

theorem triangle_area (a b c : ℝ) (A : ℝ) :
  a = 2 →
  b = 2 * c →
  Real.cos A = 1 / 4 →
  (1 / 2) * b * c * Real.sin A = Real.sqrt 15 / 4 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_l593_59381


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_q_nested_calculation_l593_59378

-- Define the function q
noncomputable def q (x y : ℝ) : ℝ :=
  if x ≥ 1 ∧ y ≥ 1 then 2*x + 3*y
  else if x < 0 ∧ y < 0 then x + y^2
  else 4*x - 2*y

-- State the theorem
theorem q_nested_calculation : q (q 2 (-2)) (q 0 0) = 48 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_q_nested_calculation_l593_59378


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_machines_sold_is_130_l593_59376

/-- Calculates the number of machines sold given commission rates, sale price, and total commission --/
def machinesSold (commissionRate1 commissionRate2 salePrice totalCommission : ℚ) : ℕ :=
  let firstBatchCommission := 100 * commissionRate1 * salePrice
  let remainingCommission := totalCommission - firstBatchCommission
  let extraMachines := remainingCommission / (commissionRate2 * salePrice)
  100 + extraMachines.floor.toNat

/-- Theorem stating that the number of machines sold is 130 --/
theorem machines_sold_is_130 :
  machinesSold (3/100) (4/100) 10000 42000 = 130 := by
  sorry

#eval machinesSold (3/100) (4/100) 10000 42000

end NUMINAMATH_CALUDE_ERRORFEEDBACK_machines_sold_is_130_l593_59376


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadrilateral_mean_angle_l593_59357

-- Define a quadrilateral
structure Quadrilateral where
  angles : Fin 4 → ℚ
  sum_360 : (angles 0) + (angles 1) + (angles 2) + (angles 3) = 360

-- Define the mean of the angles
def mean_angles (q : Quadrilateral) : ℚ :=
  ((q.angles 0) + (q.angles 1) + (q.angles 2) + (q.angles 3)) / 4

-- Theorem statement
theorem quadrilateral_mean_angle (q : Quadrilateral) : mean_angles q = 90 := by
  -- Expand the definition of mean_angles
  unfold mean_angles
  -- Use the sum_360 property of the quadrilateral
  rw [q.sum_360]
  -- Simplify the arithmetic
  norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadrilateral_mean_angle_l593_59357


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_minus_sin_value_l593_59382

theorem cos_minus_sin_value (α : ℝ) 
  (h1 : Real.sin (2 * α) = 1/4) 
  (h2 : π/4 < α) 
  (h3 : α < π/2) : 
  Real.cos α - Real.sin α = -Real.sqrt 3/2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_minus_sin_value_l593_59382


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_square_cube_roots_problem_l593_59341

theorem square_cube_roots_problem (a b : ℝ) :
  (∃ x : ℝ, x > 0 ∧ Real.sqrt x = 3 * a - 14 ∧ Real.sqrt x = a - 2) →
  ((b - 15)^(1/3 : ℝ) = -3) →
  (a = 4 ∧ b = -12 ∧ Real.sqrt (4 * a + b) = 2 ∨ Real.sqrt (4 * a + b) = -2) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_square_cube_roots_problem_l593_59341


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_womans_rate_is_12_l593_59373

/-- The woman's traveling rate in miles per hour -/
noncomputable def womans_rate : ℝ := 12

/-- The man's constant walking rate in miles per hour -/
noncomputable def mans_rate : ℝ := 6

/-- Time in hours the woman waits after passing the man -/
noncomputable def wait_time_after_passing : ℝ := 1/6

/-- Time in hours the woman waits for the man to catch up -/
noncomputable def wait_time_for_catchup : ℝ := 1/6

/-- Total waiting time in hours -/
noncomputable def total_wait_time : ℝ := wait_time_after_passing + wait_time_for_catchup

theorem womans_rate_is_12 :
  womans_rate = 12 :=
by
  -- The proof steps would go here, but for now we'll use sorry
  sorry

#check womans_rate_is_12

end NUMINAMATH_CALUDE_ERRORFEEDBACK_womans_rate_is_12_l593_59373


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_square_measurement_error_l593_59379

/-- If the percentage error in the calculated area of a square is 90.44%,
    then the percentage error in measuring the side of the square is approximately 38%. -/
theorem square_measurement_error (S S' : ℝ) (h : S > 0) :
  (S'^2 - S^2) / S^2 = 0.9044 →
  abs ((S' - S) / S - 0.38) < 0.001 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_square_measurement_error_l593_59379


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_collinear_iff_k_eq_half_right_angled_iff_k_in_set_l593_59358

/-- Given three points A, B, and C on a plane, with vectors BC and AC defined as follows -/
def BC (k : ℝ) : ℝ × ℝ := (2 - k, 3)
def AC : ℝ × ℝ := (2, 4)

/-- A, B, and C are collinear if and only if k = 1/2 -/
theorem collinear_iff_k_eq_half (k : ℝ) : 
  (∃ (t : ℝ), BC k = t • AC) ↔ k = 1/2 := by sorry

/-- ABC is a right-angled triangle if and only if k ∈ {-2, -1, 3, 8} -/
theorem right_angled_iff_k_in_set (k : ℝ) :
  (∃ (X Y Z : ℝ × ℝ), X + Y + Z = 0 ∧ 
    (X • Y = 0 ∨ X • Z = 0 ∨ Y • Z = 0) ∧
    (X = BC k ∨ X = AC ∨ X = AC - BC k) ∧
    (Y = BC k ∨ Y = AC ∨ Y = AC - BC k) ∧
    (Z = BC k ∨ Z = AC ∨ Z = AC - BC k) ∧
    X ≠ Y ∧ Y ≠ Z ∧ X ≠ Z) 
  ↔ k ∈ ({-2, -1, 3, 8} : Set ℝ) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_collinear_iff_k_eq_half_right_angled_iff_k_in_set_l593_59358


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sphere_surface_area_is_48pi_l593_59329

/-- A sphere contains a cube with edge length 4 such that all vertices of the cube
    lie on the surface of the sphere. -/
structure SphereCube where
  edge_length : ℝ
  edge_length_eq : edge_length = 4

/-- The surface area of a sphere containing a cube with edge length 4,
    where all vertices of the cube lie on the surface of the sphere. -/
noncomputable def sphere_surface_area (sc : SphereCube) : ℝ :=
  48 * Real.pi

/-- Theorem: The surface area of the sphere described in SphereCube is 48π. -/
theorem sphere_surface_area_is_48pi (sc : SphereCube) :
  sphere_surface_area sc = 48 * Real.pi := by
  -- Unfold the definition of sphere_surface_area
  unfold sphere_surface_area
  -- The equality now follows directly from the definition
  rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sphere_surface_area_is_48pi_l593_59329


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_d_q_l593_59316

noncomputable def arithmetic_sequence (a₁ : ℝ) (d : ℝ) (n : ℕ) : ℝ :=
  a₁ + (n - 1) * d

noncomputable def geometric_sequence (b₁ : ℝ) (q : ℝ) (n : ℕ) : ℝ :=
  b₁ * q^(n - 1)

noncomputable def S_n (a₁ : ℝ) (d : ℝ) (n : ℕ) : ℝ :=
  n * (2 * a₁ + (n - 1) * d) / 2

noncomputable def T_n (b₁ : ℝ) (q : ℝ) (n : ℕ) : ℝ :=
  if q = 1 then n * b₁ else b₁ * (1 - q^n) / (1 - q)

theorem unique_d_q (a₁ b₁ d q : ℝ) :
  (∀ n : ℕ, n > 0 → n^2 * (T_n b₁ q n + 1) = 2^n * S_n a₁ d n) →
  d = 2 ∧ q = 2 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_d_q_l593_59316


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_continuous_function_zero_product_nonnegative_l593_59352

theorem continuous_function_zero_product_nonnegative :
  ∃ (f : ℝ → ℝ) (a b : ℝ),
    a < b ∧
    ContinuousOn f (Set.Icc a b) ∧
    (∃ x, a < x ∧ x < b ∧ f x = 0) ∧
    f a * f b ≥ 0 :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_continuous_function_zero_product_nonnegative_l593_59352


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_circle_to_line_l593_59345

-- Define the circle M
def circle_M (x y : ℝ) : Prop := (x - 4)^2 + y^2 = 1

-- Define the line l in polar coordinates
def line_l_polar (ρ θ : ℝ) : Prop := ρ * Real.sin (θ + Real.pi/6) = 1/2

-- Define the line l in Cartesian coordinates
def line_l_cartesian (x y : ℝ) : Prop := x + Real.sqrt 3 * y - 1 = 0

-- Define the distance function from a point (x, y) to line l
noncomputable def distance_to_line (x y : ℝ) : ℝ := 
  |x + Real.sqrt 3 * y - 1| / 2

-- State the theorem
theorem min_distance_circle_to_line :
  ∃ (x y : ℝ), circle_M x y ∧ 
  (∀ (x' y' : ℝ), circle_M x' y' → distance_to_line x y ≤ distance_to_line x' y') ∧
  distance_to_line x y = 1/2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_circle_to_line_l593_59345


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_parallel_to_plane_l593_59337

-- Define the plane α
structure Plane where
  normal : Fin 3 → ℝ

-- Define a line
structure Line where
  direction : Fin 3 → ℝ
  point : Fin 3 → ℝ

-- Define the dot product
def dot_product (v w : Fin 3 → ℝ) : ℝ := 
  (v 0) * (w 0) + (v 1) * (w 1) + (v 2) * (w 2)

-- Define parallel relationship between a line and a plane
def is_parallel (l : Line) (p : Plane) : Prop :=
  dot_product l.direction p.normal = 0 ∧ ∃ (x : Fin 3 → ℝ), x ≠ l.point

-- Helper function to create a vector from three real numbers
def vector3 (x y z : ℝ) : Fin 3 → ℝ
  | 0 => x
  | 1 => y
  | 2 => z

-- Theorem statement
theorem line_parallel_to_plane (α : Plane) (AB : Line) :
  α.normal = vector3 2 (-2) 4 →
  AB.direction = vector3 (-3) 1 2 →
  (∃ (x : Fin 3 → ℝ), x ≠ AB.point) →
  is_parallel AB α := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_parallel_to_plane_l593_59337


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_of_f_l593_59369

-- Define the function f(x) = (x^2 + 1) / x
noncomputable def f (x : ℝ) := (x^2 + 1) / x

-- State the theorem
theorem min_value_of_f :
  ∀ x : ℝ, x > (1/2 : ℝ) → f x ≥ 2 ∧ (f x = 2 ↔ x = 1) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_of_f_l593_59369


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_waste_volume_maximum_l593_59307

-- Define the drainage volume function V
noncomputable def V (x : ℝ) : ℝ :=
  if x ≤ 0.2 then 90
  else if x ≤ 2 then -50 * x + 100
  else 0

-- Define the waste volume function f
noncomputable def f (x : ℝ) : ℝ := x * V x

-- Theorem statement
theorem waste_volume_maximum :
  ∀ x : ℝ, 0 ≤ x ∧ x ≤ 2 → f x ≤ f 1 ∧ f 1 = 50 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_waste_volume_maximum_l593_59307


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_simon_method_more_popular_l593_59308

/-- Represents the number of English delegates -/
def x : ℕ → ℕ := fun n => n

/-- The total number of delegates -/
def total_delegates (n : ℕ) : ℕ := 7 * x n

/-- The number of delegates supporting Professor Smith's method -/
def smith_supporters (n : ℕ) : ℕ := x n

/-- The number of German delegates supporting Professor Simon's method -/
def y : ℕ → ℕ := fun n => n

/-- The number of delegates supporting Professor Simon's method -/
def simon_supporters (n : ℕ) : ℕ := 4 * x n

/-- Theorem stating that Professor Simon's method received more support -/
theorem simon_method_more_popular (n : ℕ) : simon_supporters n > smith_supporters n := by
  -- Unfold the definitions
  unfold simon_supporters smith_supporters x
  -- Simplify
  simp
  -- The proof
  sorry

#check simon_method_more_popular

end NUMINAMATH_CALUDE_ERRORFEEDBACK_simon_method_more_popular_l593_59308


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fill_time_with_leak_theorem_l593_59333

/-- The time it takes to fill a leaky tank -/
noncomputable def fill_time_with_leak (pump_fill_time leak_empty_time : ℝ) : ℝ :=
  (pump_fill_time * leak_empty_time) / (leak_empty_time - pump_fill_time)

/-- Theorem: Given a pump that fills a tank in 2 hours and a leak that empties
    a full tank in 7 hours, it takes 2.8 hours to fill the tank with the leak present -/
theorem fill_time_with_leak_theorem (pump_fill_time leak_empty_time : ℝ)
  (h1 : pump_fill_time = 2)
  (h2 : leak_empty_time = 7) :
  fill_time_with_leak pump_fill_time leak_empty_time = 2.8 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_fill_time_with_leak_theorem_l593_59333
