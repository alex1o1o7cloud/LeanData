import Mathlib

namespace NUMINAMATH_CALUDE_f_not_prime_l1364_136459

def f (n : ℕ+) : ℤ := n.val^6 - 550 * n.val^3 + 324

theorem f_not_prime : ∀ n : ℕ+, ¬ Nat.Prime (Int.natAbs (f n)) := by
  sorry

end NUMINAMATH_CALUDE_f_not_prime_l1364_136459


namespace NUMINAMATH_CALUDE_problem_statement_l1364_136457

open Real

noncomputable def f (x : ℝ) : ℝ := log x - (1/4) * x + (3/(4*x)) - 1

def g (b : ℝ) (x : ℝ) : ℝ := x^2 - 2*b*x + 4

theorem problem_statement (b : ℝ) :
  (∀ x₁ ∈ Set.Ioo 0 2, ∃ x₂ ∈ Set.Icc 1 2, f x₁ ≥ g b x₂) ↔ b ≥ 17/8 :=
sorry

end NUMINAMATH_CALUDE_problem_statement_l1364_136457


namespace NUMINAMATH_CALUDE_adams_forgotten_lawns_l1364_136456

/-- Adam's lawn mowing problem -/
theorem adams_forgotten_lawns (dollars_per_lawn : ℕ) (total_lawns : ℕ) (actual_earnings : ℕ) :
  dollars_per_lawn = 9 →
  total_lawns = 12 →
  actual_earnings = 36 →
  total_lawns - (actual_earnings / dollars_per_lawn) = 8 := by
  sorry

end NUMINAMATH_CALUDE_adams_forgotten_lawns_l1364_136456


namespace NUMINAMATH_CALUDE_fencing_cost_calculation_l1364_136486

/-- Represents a rectangular plot with given dimensions and fencing cost -/
structure RectangularPlot where
  length : ℝ
  breadth : ℝ
  fencing_cost_per_meter : ℝ

/-- Calculates the total cost of fencing for a rectangular plot -/
def total_fencing_cost (plot : RectangularPlot) : ℝ :=
  2 * (plot.length + plot.breadth) * plot.fencing_cost_per_meter

/-- Theorem stating the total fencing cost for a specific rectangular plot -/
theorem fencing_cost_calculation :
  let plot : RectangularPlot :=
    { length := 58
    , breadth := 58 - 16
    , fencing_cost_per_meter := 26.5 }
  total_fencing_cost plot = 5300 := by
  sorry


end NUMINAMATH_CALUDE_fencing_cost_calculation_l1364_136486


namespace NUMINAMATH_CALUDE_defective_pens_count_l1364_136400

/-- The number of pens in the box -/
def total_pens : ℕ := 16

/-- The probability of selecting two non-defective pens -/
def prob_two_non_defective : ℚ := 65/100

/-- The number of defective pens in the box -/
def defective_pens : ℕ := 3

/-- Theorem stating that given the total number of pens and the probability of
    selecting two non-defective pens, the number of defective pens is 3 -/
theorem defective_pens_count (n : ℕ) (h1 : n = total_pens) 
  (h2 : (n - defective_pens : ℚ) / n * ((n - defective_pens - 1) : ℚ) / (n - 1) = prob_two_non_defective) :
  defective_pens = 3 := by
  sorry

#eval defective_pens

end NUMINAMATH_CALUDE_defective_pens_count_l1364_136400


namespace NUMINAMATH_CALUDE_parallel_transitivity_counterexample_l1364_136468

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the parallel relation for lines and planes
variable (parallelLine : Line → Line → Prop)
variable (parallelLinePlane : Line → Plane → Prop)

-- State the theorem
theorem parallel_transitivity_counterexample 
  (l n : Line) (α : Plane) : 
  ¬(∀ l n α, parallelLinePlane l α → parallelLinePlane n α → parallelLine l n) :=
sorry

end NUMINAMATH_CALUDE_parallel_transitivity_counterexample_l1364_136468


namespace NUMINAMATH_CALUDE_tan_40_plus_6_sin_40_l1364_136487

theorem tan_40_plus_6_sin_40 : 
  Real.tan (40 * π / 180) + 6 * Real.sin (40 * π / 180) = 
    Real.sqrt 3 + Real.cos (10 * π / 180) / Real.cos (40 * π / 180) := by sorry

end NUMINAMATH_CALUDE_tan_40_plus_6_sin_40_l1364_136487


namespace NUMINAMATH_CALUDE_prob_red_ball_l1364_136497

/-- The probability of drawing a red ball from a bag with 2 red balls and 1 white ball is 2/3 -/
theorem prob_red_ball (total_balls : ℕ) (red_balls : ℕ) (white_balls : ℕ) :
  total_balls = 3 →
  red_balls = 2 →
  white_balls = 1 →
  red_balls + white_balls = total_balls →
  (red_balls : ℚ) / total_balls = 2 / 3 := by
  sorry

#check prob_red_ball

end NUMINAMATH_CALUDE_prob_red_ball_l1364_136497


namespace NUMINAMATH_CALUDE_als_original_portion_l1364_136415

theorem als_original_portion (a b c : ℝ) : 
  a + b + c = 1200 →
  a - 150 + 3*b + 3*c = 1800 →
  a = 825 := by
sorry

end NUMINAMATH_CALUDE_als_original_portion_l1364_136415


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l1364_136472

open Set

-- Define the sets A and B
def A : Set ℝ := { x | 2 < x ∧ x < 4 }
def B : Set ℝ := { x | x < 3 ∨ x > 5 }

-- State the theorem
theorem intersection_of_A_and_B : A ∩ B = { x | 2 < x ∧ x < 3 } := by
  sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l1364_136472


namespace NUMINAMATH_CALUDE_compound_molecular_weight_l1364_136448

/-- The atomic weight of Hydrogen in g/mol -/
def atomic_weight_H : ℝ := 1.01

/-- The atomic weight of Bromine in g/mol -/
def atomic_weight_Br : ℝ := 79.90

/-- The atomic weight of Oxygen in g/mol -/
def atomic_weight_O : ℝ := 16.00

/-- The number of Hydrogen atoms in the compound -/
def num_H : ℕ := 1

/-- The number of Bromine atoms in the compound -/
def num_Br : ℕ := 1

/-- The number of Oxygen atoms in the compound -/
def num_O : ℕ := 3

/-- The molecular weight of the compound in g/mol -/
def molecular_weight : ℝ := 
  num_H * atomic_weight_H + num_Br * atomic_weight_Br + num_O * atomic_weight_O

/-- Theorem stating that the molecular weight of the compound is 128.91 g/mol -/
theorem compound_molecular_weight : molecular_weight = 128.91 := by
  sorry

end NUMINAMATH_CALUDE_compound_molecular_weight_l1364_136448


namespace NUMINAMATH_CALUDE_M_subset_N_l1364_136479

def M : Set ℝ := { x | ∃ k : ℤ, x = (k * Real.pi / 2) + (Real.pi / 4) }

def N : Set ℝ := { x | ∃ k : ℤ, x = (k * Real.pi / 4) + (Real.pi / 2) }

theorem M_subset_N : M ⊆ N := by
  sorry

end NUMINAMATH_CALUDE_M_subset_N_l1364_136479


namespace NUMINAMATH_CALUDE_arithmetic_computation_l1364_136491

theorem arithmetic_computation : 3 + 8 * 3 - 4 + 2^3 * 5 / 2 = 43 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_computation_l1364_136491


namespace NUMINAMATH_CALUDE_square_of_cube_third_smallest_prime_l1364_136467

-- Define the third smallest prime number
def third_smallest_prime : ℕ := 5

-- State the theorem
theorem square_of_cube_third_smallest_prime :
  (third_smallest_prime ^ 3) ^ 2 = 15625 := by sorry

end NUMINAMATH_CALUDE_square_of_cube_third_smallest_prime_l1364_136467


namespace NUMINAMATH_CALUDE_carol_initial_cupcakes_l1364_136463

/-- The number of cupcakes Carol initially made -/
def initial_cupcakes : ℕ := 30

/-- The number of cupcakes Carol sold -/
def sold_cupcakes : ℕ := 9

/-- The number of additional cupcakes Carol made -/
def additional_cupcakes : ℕ := 28

/-- The total number of cupcakes Carol had at the end -/
def total_cupcakes : ℕ := 49

/-- Theorem: Carol initially made 30 cupcakes -/
theorem carol_initial_cupcakes : 
  initial_cupcakes = total_cupcakes - additional_cupcakes + sold_cupcakes :=
by sorry

end NUMINAMATH_CALUDE_carol_initial_cupcakes_l1364_136463


namespace NUMINAMATH_CALUDE_committee_seating_arrangements_l1364_136478

/-- The number of distinct arrangements of chairs and benches -/
def distinct_arrangements (total_positions : ℕ) (bench_count : ℕ) : ℕ :=
  Nat.choose total_positions bench_count

theorem committee_seating_arrangements :
  distinct_arrangements 14 4 = 1001 := by
  sorry

end NUMINAMATH_CALUDE_committee_seating_arrangements_l1364_136478


namespace NUMINAMATH_CALUDE_factorization_x_squared_minus_2x_l1364_136445

theorem factorization_x_squared_minus_2x (x : ℝ) : x^2 - 2*x = x*(x - 2) := by
  sorry

end NUMINAMATH_CALUDE_factorization_x_squared_minus_2x_l1364_136445


namespace NUMINAMATH_CALUDE_age_sum_proof_l1364_136424

theorem age_sum_proof (uncle_age : ℕ) (yuna_eunji_diff : ℕ) (uncle_eunji_diff : ℕ) 
  (h1 : uncle_age = 41)
  (h2 : yuna_eunji_diff = 3)
  (h3 : uncle_eunji_diff = 25) :
  uncle_age - uncle_eunji_diff + (uncle_age - uncle_eunji_diff + yuna_eunji_diff) = 35 := by
  sorry

end NUMINAMATH_CALUDE_age_sum_proof_l1364_136424


namespace NUMINAMATH_CALUDE_bonus_threshold_correct_l1364_136449

/-- The sales amount that triggers the bonus commission --/
def bonus_threshold : ℝ := 10000

/-- The total sales amount --/
def total_sales : ℝ := 14000

/-- The commission rate on total sales --/
def commission_rate : ℝ := 0.09

/-- The bonus commission rate on excess sales --/
def bonus_rate : ℝ := 0.03

/-- The total commission received --/
def total_commission : ℝ := 1380

/-- The bonus commission received --/
def bonus_commission : ℝ := 120

theorem bonus_threshold_correct :
  commission_rate * total_sales = total_commission - bonus_commission ∧
  bonus_rate * (total_sales - bonus_threshold) = bonus_commission :=
by sorry

end NUMINAMATH_CALUDE_bonus_threshold_correct_l1364_136449


namespace NUMINAMATH_CALUDE_problem_statement_l1364_136412

theorem problem_statement (x : ℝ) (h : x + 1/x = 3) : 
  (x - 3)^2 + 16/((x - 3)^2) = 7 := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l1364_136412


namespace NUMINAMATH_CALUDE_triangle_geometric_sequence_l1364_136474

theorem triangle_geometric_sequence (a b c : ℝ) (A B C : ℝ) :
  a > 0 → b > 0 → c > 0 →
  -- a, b, c form a geometric sequence
  b^2 = a * c →
  -- cos B = 1/3
  Real.cos B = 1/3 →
  -- a/c = 1/2
  a / c = 1/2 →
  -- k is the first term of the geometric sequence
  ∃ k : ℝ, k > 0 ∧ a = k ∧ b = 2*k ∧ c = 4*k ∧ a + c = 5*k := by
sorry

end NUMINAMATH_CALUDE_triangle_geometric_sequence_l1364_136474


namespace NUMINAMATH_CALUDE_probability_of_five_ones_l1364_136490

def num_dice : ℕ := 15
def num_ones : ℕ := 5
def sides_on_die : ℕ := 6

theorem probability_of_five_ones :
  (Nat.choose num_dice num_ones : ℚ) * (1 / sides_on_die : ℚ)^num_ones * (1 - 1 / sides_on_die : ℚ)^(num_dice - num_ones) =
  (Nat.choose 15 5 : ℚ) * (1 / 6 : ℚ)^5 * (5 / 6 : ℚ)^10 := by
  sorry

end NUMINAMATH_CALUDE_probability_of_five_ones_l1364_136490


namespace NUMINAMATH_CALUDE_only_345_is_right_triangle_pythagoras_345_l1364_136452

/-- A function that checks if three numbers can form a right triangle --/
def isRightTriangle (a b c : ℝ) : Prop :=
  a > 0 ∧ b > 0 ∧ c > 0 ∧ a^2 + b^2 = c^2

/-- The theorem stating that only one of the given sets forms a right triangle --/
theorem only_345_is_right_triangle :
  ¬ isRightTriangle 1 1 (Real.sqrt 3) ∧
  isRightTriangle 3 4 5 ∧
  ¬ isRightTriangle 2 3 4 ∧
  ¬ isRightTriangle 5 7 9 :=
by sorry

/-- The specific theorem for the (3, 4, 5) right triangle --/
theorem pythagoras_345 : 3^2 + 4^2 = 5^2 :=
by sorry

end NUMINAMATH_CALUDE_only_345_is_right_triangle_pythagoras_345_l1364_136452


namespace NUMINAMATH_CALUDE_lions_mortality_rate_l1364_136460

/-- The number of lions that die each month in Londolozi -/
def lions_die_per_month : ℕ := 1

/-- The initial number of lions in Londolozi -/
def initial_lions : ℕ := 100

/-- The number of lion cubs born per month in Londolozi -/
def cubs_born_per_month : ℕ := 5

/-- The number of months in a year -/
def months_in_year : ℕ := 12

/-- The number of lions after one year in Londolozi -/
def lions_after_year : ℕ := 148

theorem lions_mortality_rate :
  lions_after_year = initial_lions + (cubs_born_per_month - lions_die_per_month) * months_in_year :=
sorry

end NUMINAMATH_CALUDE_lions_mortality_rate_l1364_136460


namespace NUMINAMATH_CALUDE_factorization_equality_l1364_136453

theorem factorization_equality (m n : ℝ) : 8*m*n - 2*m*n^3 = 2*m*n*(2+n)*(2-n) := by sorry

end NUMINAMATH_CALUDE_factorization_equality_l1364_136453


namespace NUMINAMATH_CALUDE_javier_first_throw_l1364_136458

/-- Represents the distance of Javier's second throw before adjustments -/
def second_throw : ℝ := sorry

/-- Calculates the adjusted distance of a throw -/
def adjusted_distance (base : ℝ) (wind_reduction : ℝ) (incline : ℝ) : ℝ :=
  base * (1 - wind_reduction) - incline

theorem javier_first_throw :
  let first_throw := 2 * second_throw
  let third_throw := 2 * first_throw
  adjusted_distance first_throw 0.05 2 +
  adjusted_distance second_throw 0.08 4 +
  adjusted_distance third_throw 0 1 = 1050 →
  first_throw = 310 := by sorry

end NUMINAMATH_CALUDE_javier_first_throw_l1364_136458


namespace NUMINAMATH_CALUDE_watermelon_seed_requirement_l1364_136481

/-- Represents the minimum number of watermelon seeds required -/
def min_seeds : ℕ := 10041

/-- Represents the number of watermelons to be sold each year -/
def watermelons_to_sell : ℕ := 10000

/-- Represents the number of seeds produced by each watermelon -/
def seeds_per_watermelon : ℕ := 250

theorem watermelon_seed_requirement (S : ℕ) :
  S ≥ min_seeds →
  ∃ (x : ℕ), S = watermelons_to_sell + x ∧
             seeds_per_watermelon * x ≥ S ∧
             ∀ (S' : ℕ), S' < S →
               ¬∃ (x' : ℕ), S' = watermelons_to_sell + x' ∧
                             seeds_per_watermelon * x' ≥ S' :=
by sorry

#check watermelon_seed_requirement

end NUMINAMATH_CALUDE_watermelon_seed_requirement_l1364_136481


namespace NUMINAMATH_CALUDE_fourth_square_area_l1364_136493

-- Define the triangles and their properties
def triangle_ABC (AB BC AC : ℝ) : Prop :=
  AB^2 + BC^2 = AC^2 ∧ AB^2 = 25 ∧ BC^2 = 49

def triangle_ACD (AC CD AD : ℝ) : Prop :=
  AC^2 + CD^2 = AD^2 ∧ CD^2 = 64

-- Theorem statement
theorem fourth_square_area 
  (AB BC AC CD AD : ℝ) 
  (h1 : triangle_ABC AB BC AC) 
  (h2 : triangle_ACD AC CD AD) :
  AD^2 = 138 := by sorry

end NUMINAMATH_CALUDE_fourth_square_area_l1364_136493


namespace NUMINAMATH_CALUDE_cosine_from_tangent_third_quadrant_l1364_136439

theorem cosine_from_tangent_third_quadrant (α : Real) :
  α ∈ Set.Ioo (π) (3*π/2) →  -- α is in the third quadrant
  Real.tan α = 1/2 →         -- tan(α) = 1/2
  Real.cos α = -2*Real.sqrt 5/5 := by
sorry

end NUMINAMATH_CALUDE_cosine_from_tangent_third_quadrant_l1364_136439


namespace NUMINAMATH_CALUDE_painted_fraction_is_three_eighths_l1364_136407

/-- Represents a square plate with sides of length 4 meters -/
structure Plate :=
  (side_length : ℝ)
  (area : ℝ)
  (h_side : side_length = 4)
  (h_area : area = side_length * side_length)

/-- Represents the number of equal parts the plate is divided into -/
def total_parts : ℕ := 16

/-- Represents the number of painted parts -/
def painted_parts : ℕ := 6

/-- The theorem to be proved -/
theorem painted_fraction_is_three_eighths (plate : Plate) :
  (painted_parts : ℝ) / total_parts = 3 / 8 := by
  sorry


end NUMINAMATH_CALUDE_painted_fraction_is_three_eighths_l1364_136407


namespace NUMINAMATH_CALUDE_second_sea_fields_medalist_from_vietnam_l1364_136454

/-- Represents a mathematician -/
structure Mathematician where
  name : String
  country : String

/-- Represents the Fields Medal award -/
inductive FieldsMedal
  | recipient : Mathematician → FieldsMedal

/-- Predicate to check if a country is in South East Asia -/
def is_south_east_asian (country : String) : Prop := sorry

/-- Predicate to check if a mathematician is a Fields Medal recipient -/
def is_fields_medalist (m : Mathematician) : Prop := sorry

/-- The second South East Asian Fields Medal recipient -/
def second_sea_fields_medalist : Mathematician := sorry

/-- Theorem stating that the second South East Asian Fields Medal recipient is from Vietnam -/
theorem second_sea_fields_medalist_from_vietnam :
  second_sea_fields_medalist.country = "Vietnam" := by sorry

end NUMINAMATH_CALUDE_second_sea_fields_medalist_from_vietnam_l1364_136454


namespace NUMINAMATH_CALUDE_sequence_difference_l1364_136441

theorem sequence_difference (a : ℕ → ℕ) : 
  (∀ n m : ℕ, n < m → a n < a m) →  -- strictly increasing
  (∀ n : ℕ, n ≥ 1 → a n ≥ 1) →     -- a_n ≥ 1 for n ≥ 1
  (∀ n : ℕ, n ≥ 1 → a (a n) = 3 * n) →  -- a_{a_n} = 3n for n ≥ 1
  a 2021 - a 1999 = 66 := by
sorry

end NUMINAMATH_CALUDE_sequence_difference_l1364_136441


namespace NUMINAMATH_CALUDE_student_weight_loss_l1364_136429

/-- The weight the student needs to lose to weigh twice as much as his sister -/
def weight_to_lose (total_weight student_weight : ℝ) : ℝ :=
  let sister_weight := total_weight - student_weight
  student_weight - 2 * sister_weight

theorem student_weight_loss (total_weight student_weight : ℝ) 
  (h1 : total_weight = 110)
  (h2 : student_weight = 75) :
  weight_to_lose total_weight student_weight = 5 := by
  sorry

end NUMINAMATH_CALUDE_student_weight_loss_l1364_136429


namespace NUMINAMATH_CALUDE_trig_identity_l1364_136499

theorem trig_identity : Real.sin (68 * π / 180) * Real.sin (67 * π / 180) - 
  Real.sin (23 * π / 180) * Real.cos (68 * π / 180) = Real.sqrt 2 / 2 := by
  sorry

end NUMINAMATH_CALUDE_trig_identity_l1364_136499


namespace NUMINAMATH_CALUDE_pizza_order_l1364_136455

theorem pizza_order (slices_per_pizza : ℕ) (total_slices : ℕ) (h1 : slices_per_pizza = 2) (h2 : total_slices = 28) :
  total_slices / slices_per_pizza = 14 := by
  sorry

end NUMINAMATH_CALUDE_pizza_order_l1364_136455


namespace NUMINAMATH_CALUDE_pinball_spending_l1364_136432

def half_dollar : ℚ := 0.5

def wednesday_spent : ℕ := 4
def next_day_spent : ℕ := 14

def total_spent : ℚ := (wednesday_spent * half_dollar) + (next_day_spent * half_dollar)

theorem pinball_spending : total_spent = 9 := by sorry

end NUMINAMATH_CALUDE_pinball_spending_l1364_136432


namespace NUMINAMATH_CALUDE_complement_of_A_l1364_136423

-- Define the universal set U
def U : Finset Nat := {1, 2, 3, 4}

-- Define set A
def A : Finset Nat := {1, 3}

-- Theorem statement
theorem complement_of_A :
  (U \ A) = {2, 4} := by sorry

end NUMINAMATH_CALUDE_complement_of_A_l1364_136423


namespace NUMINAMATH_CALUDE_remainder_6n_mod_4_l1364_136469

theorem remainder_6n_mod_4 (n : ℤ) (h : n % 4 = 1) : (6 * n) % 4 = 2 := by
  sorry

end NUMINAMATH_CALUDE_remainder_6n_mod_4_l1364_136469


namespace NUMINAMATH_CALUDE_count_pairs_20_l1364_136442

def count_pairs (n : ℕ) : ℕ :=
  (n - 11) * (n - 11 + 1) / 2

theorem count_pairs_20 :
  count_pairs 20 = 45 :=
sorry

end NUMINAMATH_CALUDE_count_pairs_20_l1364_136442


namespace NUMINAMATH_CALUDE_ellipse_properties_l1364_136430

/-- Definition of the ellipse C -/
def ellipse_C (x y : ℝ) (a b : ℝ) : Prop :=
  x^2 / a^2 + y^2 / b^2 = 1

/-- Definition of point M -/
def point_M : ℝ × ℝ := (0, 2)

/-- Theorem stating the properties of the ellipse and its related points -/
theorem ellipse_properties :
  ∀ (a b : ℝ), a > b ∧ b > 0 →
  (∀ x y, ellipse_C x y a b ↔ x^2 / 8 + y^2 / 4 = 1) ∧
  (∀ x y, (∃ x₁ y₁, ellipse_C x₁ y₁ a b ∧ x = (x₁ + point_M.1) / 2 ∧ y = (y₁ + point_M.2) / 2) ↔
    x^2 / 2 + (y - 1)^2 = 1) ∧
  (∀ k₁ k₂ x₁ y₁ x₂ y₂,
    ellipse_C x₁ y₁ a b ∧ ellipse_C x₂ y₂ a b ∧
    k₁ = (y₁ - point_M.2) / (x₁ - point_M.1) ∧
    k₂ = (y₂ - point_M.2) / (x₂ - point_M.1) ∧
    k₁ + k₂ = 8 →
    ∃ t, (1 - t) * x₁ + t * x₂ = -1/2 ∧ (1 - t) * y₁ + t * y₂ = -2) :=
by sorry

end NUMINAMATH_CALUDE_ellipse_properties_l1364_136430


namespace NUMINAMATH_CALUDE_max_roses_is_316_l1364_136447

/-- Represents the pricing options and budget for purchasing roses -/
structure RosePurchase where
  single_price : ℚ  -- Price of a single rose
  dozen_price : ℚ   -- Price of a dozen roses
  two_dozen_price : ℚ -- Price of two dozen roses
  budget : ℚ        -- Total budget

/-- Calculates the maximum number of roses that can be purchased given the pricing options and budget -/
def max_roses (rp : RosePurchase) : ℕ :=
  sorry

/-- Theorem stating that the maximum number of roses that can be purchased is 316 -/
theorem max_roses_is_316 (rp : RosePurchase) 
  (h1 : rp.single_price = 63/10)
  (h2 : rp.dozen_price = 36)
  (h3 : rp.two_dozen_price = 50)
  (h4 : rp.budget = 680) :
  max_roses rp = 316 :=
by sorry

end NUMINAMATH_CALUDE_max_roses_is_316_l1364_136447


namespace NUMINAMATH_CALUDE_normal_distribution_std_dev_l1364_136480

/-- For a normal distribution with mean 14.5, if the value that is exactly 2 standard deviations
    less than the mean is 11.5, then the standard deviation is 1.5. -/
theorem normal_distribution_std_dev (μ σ : ℝ) :
  μ = 14.5 →
  μ - 2 * σ = 11.5 →
  σ = 1.5 := by
sorry

end NUMINAMATH_CALUDE_normal_distribution_std_dev_l1364_136480


namespace NUMINAMATH_CALUDE_intersection_complement_empty_l1364_136427

/-- The set of all non-zero real numbers -/
def P : Set ℝ := {x : ℝ | x ≠ 0}

/-- The set of all positive real numbers -/
def Q : Set ℝ := {x : ℝ | x > 0}

/-- Theorem stating that the intersection of Q and the complement of P in ℝ is empty -/
theorem intersection_complement_empty : Q ∩ (Set.univ \ P) = ∅ := by sorry

end NUMINAMATH_CALUDE_intersection_complement_empty_l1364_136427


namespace NUMINAMATH_CALUDE_snow_volume_to_clear_l1364_136417

/-- Calculates the volume of snow to be cleared from a driveway -/
theorem snow_volume_to_clear (length width : Real) (depth : Real) (melt_percentage : Real) : 
  length = 30 ∧ width = 3 ∧ depth = 0.5 ∧ melt_percentage = 0.1 → 
  (1 - melt_percentage) * (length * width * depth) / 27 = 1.5 := by
  sorry

#check snow_volume_to_clear

end NUMINAMATH_CALUDE_snow_volume_to_clear_l1364_136417


namespace NUMINAMATH_CALUDE_usamo_page_count_l1364_136496

theorem usamo_page_count (a₁ a₂ a₃ a₄ a₅ a₆ : ℕ+) :
  (((a₁ : ℝ) + 1) / 2 + ((a₂ : ℝ) + 1) / 2 + ((a₃ : ℝ) + 1) / 2 + 
   ((a₄ : ℝ) + 1) / 2 + ((a₅ : ℝ) + 1) / 2 + ((a₆ : ℝ) + 1) / 2) = 2017 →
  (a₁ : ℕ) + a₂ + a₃ + a₄ + a₅ + a₆ = 4028 := by
  sorry

end NUMINAMATH_CALUDE_usamo_page_count_l1364_136496


namespace NUMINAMATH_CALUDE_problem_solution_l1364_136438

-- Define the function f
def f (x : ℝ) : ℝ := |x| - |x - 1|

-- Theorem statement
theorem problem_solution :
  (∃ (m : ℝ), ∀ (x : ℝ), f x ≥ |m - 1| → m ≤ 2) ∧
  (∀ (a b : ℝ), a > 0 → b > 0 → a^2 + b^2 = 2 → a + b ≥ 2*a*b) :=
by sorry

end NUMINAMATH_CALUDE_problem_solution_l1364_136438


namespace NUMINAMATH_CALUDE_area_of_triangle_ACF_l1364_136475

/-- Right triangle with sides a, b, and c -/
structure RightTriangle where
  a : ℝ
  b : ℝ
  c : ℝ
  right_angle : a^2 + b^2 = c^2

/-- Rectangle with width w and height h -/
structure Rectangle where
  w : ℝ
  h : ℝ

theorem area_of_triangle_ACF (ABC : RightTriangle) (ABD : RightTriangle) (BCEF : Rectangle) :
  ABC.a = 8 →
  ABC.c = 12 →
  ABD.a = 8 →
  ABD.b = 8 →
  BCEF.w = 8 →
  BCEF.h = 8 →
  ABC.a = ABD.a →
  (1/2) * ABC.a * ABC.c = 24 := by
  sorry

#check area_of_triangle_ACF

end NUMINAMATH_CALUDE_area_of_triangle_ACF_l1364_136475


namespace NUMINAMATH_CALUDE_factor_sum_l1364_136404

theorem factor_sum (P Q : ℝ) : 
  (∃ b c : ℝ, (X^2 + 4*X + 3) * (X^2 + b*X + c) = X^4 + P*X^2 + Q) →
  P + Q = -1 :=
by sorry

end NUMINAMATH_CALUDE_factor_sum_l1364_136404


namespace NUMINAMATH_CALUDE_tan_double_angle_l1364_136444

theorem tan_double_angle (θ : Real) (h : Real.tan θ = 2) : Real.tan (2 * θ) = -4/3 := by
  sorry

end NUMINAMATH_CALUDE_tan_double_angle_l1364_136444


namespace NUMINAMATH_CALUDE_pencil_case_cost_solution_l1364_136422

/-- Calculates the amount spent on a pencil case given the initial amount,
    the amount spent on a toy truck, and the remaining amount. -/
def pencil_case_cost (initial : ℝ) (toy_truck : ℝ) (remaining : ℝ) : ℝ :=
  initial - toy_truck - remaining

theorem pencil_case_cost_solution :
  pencil_case_cost 10 3 5 = 2 := by
  sorry

end NUMINAMATH_CALUDE_pencil_case_cost_solution_l1364_136422


namespace NUMINAMATH_CALUDE_amusement_park_line_count_l1364_136402

theorem amusement_park_line_count : 
  ∀ (eunji_position : ℕ) (people_behind : ℕ),
    eunji_position = 6 →
    people_behind = 7 →
    eunji_position + people_behind = 13 := by
  sorry

end NUMINAMATH_CALUDE_amusement_park_line_count_l1364_136402


namespace NUMINAMATH_CALUDE_negation_of_existence_logarithm_equality_negation_l1364_136470

theorem negation_of_existence (P : ℝ → Prop) :
  (¬ ∃ x : ℝ, x > 0 ∧ P x) ↔ (∀ x : ℝ, x > 0 → ¬ P x) :=
by sorry

theorem logarithm_equality_negation :
  (¬ ∃ x : ℝ, x > 0 ∧ Real.log x = x - 1) ↔
  (∀ x : ℝ, x > 0 → Real.log x ≠ x - 1) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_existence_logarithm_equality_negation_l1364_136470


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_set_l1364_136420

theorem quadratic_inequality_solution_set :
  {x : ℝ | x^2 - 6*x > 20} = {x : ℝ | x < -2 ∨ x > 10} := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_set_l1364_136420


namespace NUMINAMATH_CALUDE_equation_solution_l1364_136419

theorem equation_solution : 
  {x : ℝ | x * (x - 3)^2 * (5 - x) = 0} = {0, 3, 5} := by
sorry

end NUMINAMATH_CALUDE_equation_solution_l1364_136419


namespace NUMINAMATH_CALUDE_grandson_age_l1364_136461

theorem grandson_age (grandmother_age grandson_age : ℕ) : 
  grandmother_age = grandson_age * 12 →
  grandmother_age + grandson_age = 65 →
  grandson_age = 5 := by
sorry

end NUMINAMATH_CALUDE_grandson_age_l1364_136461


namespace NUMINAMATH_CALUDE_least_exponent_for_divisibility_l1364_136416

/-- The function that calculates the sum of powers for the given exponent -/
def sumOfPowers (a : ℕ+) : ℕ :=
  (1995 : ℕ) ^ a.val + (1996 : ℕ) ^ a.val + (1997 : ℕ) ^ a.val

/-- The property that the sum is divisible by 10 -/
def isDivisibleBy10 (n : ℕ) : Prop :=
  ∃ k : ℕ, n = 10 * k

/-- The main theorem statement -/
theorem least_exponent_for_divisibility :
  (∀ a : ℕ+, a < 2 → ¬(isDivisibleBy10 (sumOfPowers a))) ∧
  isDivisibleBy10 (sumOfPowers 2) := by
  sorry

#check least_exponent_for_divisibility

end NUMINAMATH_CALUDE_least_exponent_for_divisibility_l1364_136416


namespace NUMINAMATH_CALUDE_oliver_final_amount_l1364_136406

/-- Calculates the final amount of money Oliver has after all transactions. -/
def olivers_money (initial : ℕ) (saved : ℕ) (frisbee_cost : ℕ) (puzzle_cost : ℕ) (gift : ℕ) : ℕ :=
  initial + saved - frisbee_cost - puzzle_cost + gift

/-- Theorem stating that Oliver's final amount of money is $15. -/
theorem oliver_final_amount :
  olivers_money 9 5 4 3 8 = 15 := by
  sorry

#eval olivers_money 9 5 4 3 8

end NUMINAMATH_CALUDE_oliver_final_amount_l1364_136406


namespace NUMINAMATH_CALUDE_wire_cutting_l1364_136471

theorem wire_cutting (total_length : ℝ) (shorter_length : ℝ) : 
  total_length = 50 →
  shorter_length + (5/2 * shorter_length) = total_length →
  shorter_length = 100/7 :=
by sorry

end NUMINAMATH_CALUDE_wire_cutting_l1364_136471


namespace NUMINAMATH_CALUDE_marbles_bet_thirteen_l1364_136425

/-- Calculates the number of marbles bet per game -/
def marbles_bet_per_game (friend_start : ℕ) (total_games : ℕ) (reggie_end : ℕ) (reggie_losses : ℕ) : ℕ :=
  ((friend_start - reggie_end) / (total_games - 2 * reggie_losses)).succ

/-- Proves that under the given conditions, 13 marbles were bet per game -/
theorem marbles_bet_thirteen (friend_start : ℕ) (total_games : ℕ) (reggie_end : ℕ) (reggie_losses : ℕ)
  (h1 : friend_start = 100)
  (h2 : total_games = 9)
  (h3 : reggie_end = 90)
  (h4 : reggie_losses = 1) :
  marbles_bet_per_game friend_start total_games reggie_end reggie_losses = 13 := by
  sorry

#eval marbles_bet_per_game 100 9 90 1

end NUMINAMATH_CALUDE_marbles_bet_thirteen_l1364_136425


namespace NUMINAMATH_CALUDE_line_equation_slope_intercept_l1364_136410

/-- Given a line equation, prove its slope and y-intercept -/
theorem line_equation_slope_intercept :
  ∀ (x y : ℝ), 
  2 * (x - 3) + (-1) * (y - (-4)) = 0 →
  ∃ (m b : ℝ), y = m * x + b ∧ m = 2 ∧ b = -10 := by
  sorry

end NUMINAMATH_CALUDE_line_equation_slope_intercept_l1364_136410


namespace NUMINAMATH_CALUDE_market_prices_l1364_136434

/-- The cost of one pound of rice in dollars -/
def rice_cost : ℝ := 0.33

/-- The number of eggs that cost the same as one pound of rice -/
def eggs_per_rice : ℕ := 1

/-- The number of eggs that cost the same as half a liter of kerosene -/
def eggs_per_half_liter : ℕ := 8

/-- The cost of one liter of kerosene in cents -/
def kerosene_cost : ℕ := 528

theorem market_prices :
  (rice_cost = rice_cost / eggs_per_rice) ∧
  (kerosene_cost = 2 * eggs_per_half_liter * rice_cost * 100) := by
sorry

end NUMINAMATH_CALUDE_market_prices_l1364_136434


namespace NUMINAMATH_CALUDE_right_angle_vector_condition_l1364_136492

/-- Given two vectors OA and OB in a Cartesian coordinate plane, 
    if the angle ABO is 90 degrees, then the t-coordinate of OA is 5. -/
theorem right_angle_vector_condition (t : ℝ) : 
  let OA : ℝ × ℝ := (-1, t)
  let OB : ℝ × ℝ := (2, 2)
  (OB.1 * (OB.1 - OA.1) + OB.2 * (OB.2 - OA.2) = 0) →
  t = 5 := by
sorry

end NUMINAMATH_CALUDE_right_angle_vector_condition_l1364_136492


namespace NUMINAMATH_CALUDE_intersection_empty_range_necessary_not_sufficient_range_l1364_136446

-- Define sets A and B
def A (a : ℝ) : Set ℝ := {x | |x - a| ≤ 4}
def B : Set ℝ := {x | (x - 2) * (x - 3) ≤ 0}

-- Theorem 1: If A ∩ B = ∅, then a ∈ (-∞, -2) ∪ (7, ∞)
theorem intersection_empty_range (a : ℝ) : 
  A a ∩ B = ∅ → a < -2 ∨ a > 7 := by sorry

-- Theorem 2: If B is a necessary but not sufficient condition for A, then a ∈ [1, 6]
theorem necessary_not_sufficient_range (a : ℝ) :
  (B ⊆ A a ∧ ¬(A a ⊆ B)) → 1 ≤ a ∧ a ≤ 6 := by sorry

end NUMINAMATH_CALUDE_intersection_empty_range_necessary_not_sufficient_range_l1364_136446


namespace NUMINAMATH_CALUDE_bank_checks_total_amount_l1364_136433

theorem bank_checks_total_amount : 
  let million_won_checks : ℕ := 25
  let hundred_thousand_won_checks : ℕ := 8
  let million_won_value : ℕ := 1000000
  let hundred_thousand_won_value : ℕ := 100000
  (million_won_checks * million_won_value + hundred_thousand_won_checks * hundred_thousand_won_value : ℕ) = 25800000 :=
by
  sorry

end NUMINAMATH_CALUDE_bank_checks_total_amount_l1364_136433


namespace NUMINAMATH_CALUDE_shaded_cubes_count_shaded_cubes_count_proof_l1364_136403

/-- Represents a 3x3x3 cube with a specific shading pattern -/
structure ShadedCube where
  /-- The number of smaller cubes in each dimension of the large cube -/
  size : Nat
  /-- The number of shaded squares on each face -/
  shaded_per_face : Nat
  /-- The total number of smaller cubes in the large cube -/
  total_cubes : Nat
  /-- Assertion that the cube is 3x3x3 -/
  size_is_three : size = 3
  /-- Assertion that the total number of cubes is correct -/
  total_is_correct : total_cubes = size ^ 3
  /-- Assertion that each face has 5 shaded squares -/
  five_shaded_per_face : shaded_per_face = 5

/-- Theorem stating that the number of shaded cubes is 20 -/
theorem shaded_cubes_count (c : ShadedCube) : Nat :=
  20

#check shaded_cubes_count

/-- Proof of the theorem -/
theorem shaded_cubes_count_proof (c : ShadedCube) : shaded_cubes_count c = 20 := by
  sorry

end NUMINAMATH_CALUDE_shaded_cubes_count_shaded_cubes_count_proof_l1364_136403


namespace NUMINAMATH_CALUDE_sector_arc_length_l1364_136495

/-- Given a circular sector with a central angle of 40° and a radius of 18,
    the arc length is equal to 4π. -/
theorem sector_arc_length (θ : ℝ) (r : ℝ) (h1 : θ = 40) (h2 : r = 18) :
  (θ / 360) * (2 * π * r) = 4 * π := by
  sorry

end NUMINAMATH_CALUDE_sector_arc_length_l1364_136495


namespace NUMINAMATH_CALUDE_a_squared_plus_a_negative_l1364_136428

theorem a_squared_plus_a_negative (a : ℝ) (h : a^2 + a < 0) : -a > a^2 ∧ a^2 > -a^2 ∧ -a^2 > a := by
  sorry

end NUMINAMATH_CALUDE_a_squared_plus_a_negative_l1364_136428


namespace NUMINAMATH_CALUDE_jason_money_unchanged_l1364_136436

/-- Represents the money situation of Fred and Jason -/
structure MoneySituation where
  fred_initial : ℕ
  jason_initial : ℕ
  fred_final : ℕ
  total_earned : ℕ

/-- The theorem stating that Jason's final money is equal to his initial money -/
theorem jason_money_unchanged (situation : MoneySituation) 
  (h1 : situation.fred_initial = 111)
  (h2 : situation.jason_initial = 40)
  (h3 : situation.fred_final = 115)
  (h4 : situation.total_earned = 4) :
  situation.jason_initial = 40 := by
  sorry

#check jason_money_unchanged

end NUMINAMATH_CALUDE_jason_money_unchanged_l1364_136436


namespace NUMINAMATH_CALUDE_max_min_product_l1364_136498

def digits : List Nat := [2, 4, 6, 8]

def makeNumber (a b c : Nat) : Nat := 100 * a + 10 * b + c

def product (a b c d : Nat) : Nat := (makeNumber a b c) * d

theorem max_min_product :
  (∀ (a b c d : Nat), a ∈ digits ∧ b ∈ digits ∧ c ∈ digits ∧ d ∈ digits →
    a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d →
    product a b c d ≤ product 8 6 4 2) ∧
  (∀ (a b c d : Nat), a ∈ digits ∧ b ∈ digits ∧ c ∈ digits ∧ d ∈ digits →
    a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d →
    product 2 4 6 8 ≤ product a b c d) :=
by sorry

end NUMINAMATH_CALUDE_max_min_product_l1364_136498


namespace NUMINAMATH_CALUDE_alex_money_left_l1364_136411

def main_job_income : ℝ := 900
def side_job_income : ℝ := 300
def main_job_tax_rate : ℝ := 0.15
def side_job_tax_rate : ℝ := 0.20
def water_bill : ℝ := 75
def main_job_tithe_rate : ℝ := 0.10
def side_job_tithe_rate : ℝ := 0.15
def groceries : ℝ := 150
def transportation : ℝ := 50

theorem alex_money_left :
  let main_job_after_tax := main_job_income * (1 - main_job_tax_rate)
  let side_job_after_tax := side_job_income * (1 - side_job_tax_rate)
  let total_income_after_tax := main_job_after_tax + side_job_after_tax
  let total_tithe := main_job_income * main_job_tithe_rate + side_job_income * side_job_tithe_rate
  let total_deductions := water_bill + groceries + transportation + total_tithe
  total_income_after_tax - total_deductions = 595 := by
  sorry

end NUMINAMATH_CALUDE_alex_money_left_l1364_136411


namespace NUMINAMATH_CALUDE_diophantine_equation_solution_l1364_136440

theorem diophantine_equation_solution : ∃ (a b c : ℕ), a^3 + b^4 = c^5 ∧ a = 4 ∧ b = 16 ∧ c = 18 := by
  sorry

end NUMINAMATH_CALUDE_diophantine_equation_solution_l1364_136440


namespace NUMINAMATH_CALUDE_tournament_committee_count_l1364_136414

/-- The number of teams in the league -/
def num_teams : ℕ := 4

/-- The number of members in each team -/
def team_size : ℕ := 8

/-- The size of the tournament committee -/
def committee_size : ℕ := 10

/-- The number of members selected from the host team -/
def host_selection : ℕ := 4

/-- The number of members selected from each non-host team -/
def non_host_selection : ℕ := 2

/-- The number of possible tournament committees -/
def num_committees : ℕ := 6146560

theorem tournament_committee_count :
  (num_teams : ℕ) *
  (Nat.choose team_size host_selection) *
  (Nat.choose team_size non_host_selection ^ (num_teams - 1)) =
  num_committees := by sorry

end NUMINAMATH_CALUDE_tournament_committee_count_l1364_136414


namespace NUMINAMATH_CALUDE_starters_count_l1364_136450

-- Define the total number of players
def total_players : ℕ := 15

-- Define the number of quadruplets
def num_quadruplets : ℕ := 4

-- Define the number of starters to choose
def num_starters : ℕ := 6

-- Define the number of quadruplets that must be in the starting lineup
def quadruplets_in_lineup : ℕ := 3

-- Define the function to calculate the number of ways to choose the starting lineup
def choose_starters : ℕ :=
  (Nat.choose num_quadruplets quadruplets_in_lineup) *
  (Nat.choose (total_players - num_quadruplets) (num_starters - quadruplets_in_lineup))

-- Theorem statement
theorem starters_count : choose_starters = 660 := by
  sorry

end NUMINAMATH_CALUDE_starters_count_l1364_136450


namespace NUMINAMATH_CALUDE_power_difference_equality_l1364_136484

theorem power_difference_equality (a b : ℕ) (h1 : a = 3) (h2 : b = 2) :
  (a^b)^a - (b^a)^b = 665 := by
  sorry

end NUMINAMATH_CALUDE_power_difference_equality_l1364_136484


namespace NUMINAMATH_CALUDE_largest_number_with_digit_sum_20_l1364_136426

def is_valid_number (n : ℕ) : Prop :=
  ∀ d, d ∈ n.digits 10 → d = 3 ∨ d = 4

def digit_sum (n : ℕ) : ℕ :=
  (n.digits 10).sum

theorem largest_number_with_digit_sum_20 :
  ∀ n : ℕ, is_valid_number n → digit_sum n = 20 → n ≤ 443333 :=
sorry

end NUMINAMATH_CALUDE_largest_number_with_digit_sum_20_l1364_136426


namespace NUMINAMATH_CALUDE_gcd_765432_654321_l1364_136488

theorem gcd_765432_654321 : Nat.gcd 765432 654321 = 111111 := by
  sorry

end NUMINAMATH_CALUDE_gcd_765432_654321_l1364_136488


namespace NUMINAMATH_CALUDE_two_red_two_blue_probability_l1364_136431

/-- The probability of selecting two red and two blue marbles from a bag -/
theorem two_red_two_blue_probability (total_marbles : ℕ) (red_marbles : ℕ) (blue_marbles : ℕ)
  (h1 : total_marbles = red_marbles + blue_marbles)
  (h2 : red_marbles = 12)
  (h3 : blue_marbles = 8) :
  (Nat.choose red_marbles 2 * Nat.choose blue_marbles 2) / Nat.choose total_marbles 4 = 3696 / 9690 := by
  sorry

end NUMINAMATH_CALUDE_two_red_two_blue_probability_l1364_136431


namespace NUMINAMATH_CALUDE_monic_quadratic_with_complex_root_l1364_136489

theorem monic_quadratic_with_complex_root :
  ∃ (a b : ℝ), ∀ (x : ℂ), x^2 + a*x + b = 0 ↔ x = 2 - 3*I ∨ x = 2 + 3*I :=
by
  -- The proof would go here
  sorry

end NUMINAMATH_CALUDE_monic_quadratic_with_complex_root_l1364_136489


namespace NUMINAMATH_CALUDE_optimal_play_result_l1364_136405

/-- Represents a square on the chessboard --/
structure Square where
  x : Fin 8
  y : Fin 8

/-- Represents the state of the chessboard --/
def Chessboard := Square → Bool

/-- Checks if two squares are neighbors --/
def are_neighbors (s1 s2 : Square) : Bool :=
  (s1.x = s2.x ∧ s1.y.val + 1 = s2.y.val) ∨
  (s1.x = s2.x ∧ s1.y.val = s2.y.val + 1) ∨
  (s1.x.val + 1 = s2.x.val ∧ s1.y = s2.y) ∨
  (s1.x.val = s2.x.val + 1 ∧ s1.y = s2.y)

/-- Counts the number of black connected components on the board --/
def count_black_components (board : Chessboard) : Nat :=
  sorry

/-- Represents a move in the game --/
inductive Move
| alice : Square → Move
| bob : Option Square → Move

/-- Applies a move to the chessboard --/
def apply_move (board : Chessboard) (move : Move) : Chessboard :=
  sorry

/-- Represents the game state --/
structure GameState where
  board : Chessboard
  alice_turn : Bool

/-- Checks if the game is over --/
def is_game_over (state : GameState) : Bool :=
  sorry

/-- Returns the optimal move for the current player --/
def optimal_move (state : GameState) : Move :=
  sorry

/-- Plays the game optimally from the given state until it's over --/
def play_game (state : GameState) : Nat :=
  sorry

/-- The main theorem: optimal play results in 16 black connected components --/
theorem optimal_play_result :
  let initial_state : GameState := {
    board := λ _ => false,  -- All squares are initially white
    alice_turn := true
  }
  play_game initial_state = 16 := by
  sorry

end NUMINAMATH_CALUDE_optimal_play_result_l1364_136405


namespace NUMINAMATH_CALUDE_five_volunteers_four_projects_l1364_136409

/-- The number of ways to allocate volunteers to projects -/
def allocation_schemes (n : ℕ) (k : ℕ) : ℕ :=
  (n.choose 2) * (k.factorial)

/-- Theorem stating the number of allocation schemes for 5 volunteers and 4 projects -/
theorem five_volunteers_four_projects :
  allocation_schemes 5 4 = 240 :=
sorry

end NUMINAMATH_CALUDE_five_volunteers_four_projects_l1364_136409


namespace NUMINAMATH_CALUDE_positive_difference_of_numbers_l1364_136451

theorem positive_difference_of_numbers (a b : ℝ) 
  (sum_eq : a + b = 10) 
  (square_diff_eq : a^2 - b^2 = 40) : 
  |a - b| = 4 := by
sorry

end NUMINAMATH_CALUDE_positive_difference_of_numbers_l1364_136451


namespace NUMINAMATH_CALUDE_circle_equation_l1364_136485

-- Define the circle C
def Circle (center : ℝ × ℝ) (radius : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | (p.1 - center.1)^2 + (p.2 - center.2)^2 = radius^2}

-- Define the line l: 3x - y - 3 = 0
def Line : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | 3 * p.1 - p.2 - 3 = 0}

theorem circle_equation :
  ∃ (center : ℝ × ℝ) (radius : ℝ),
    (center ∈ Line) ∧
    ((2, 5) ∈ Circle center radius) ∧
    ((4, 3) ∈ Circle center radius) ∧
    (Circle center radius = {p : ℝ × ℝ | (p.1 - 2)^2 + (p.2 - 3)^2 = 4}) := by
  sorry

end NUMINAMATH_CALUDE_circle_equation_l1364_136485


namespace NUMINAMATH_CALUDE_abs_lt_one_sufficient_not_necessary_for_lt_one_l1364_136494

theorem abs_lt_one_sufficient_not_necessary_for_lt_one :
  (∃ x : ℝ, (|x| < 1 → x < 1) ∧ ¬(x < 1 → |x| < 1)) := by
  sorry

end NUMINAMATH_CALUDE_abs_lt_one_sufficient_not_necessary_for_lt_one_l1364_136494


namespace NUMINAMATH_CALUDE_function_properties_l1364_136435

noncomputable section

-- Define the function f(x)
def f (a : ℝ) (x : ℝ) : ℝ := x / 4 + a / x - Real.log x - 3 / 2

-- Define the derivative of f(x)
def f' (a : ℝ) (x : ℝ) : ℝ := 1 / 4 - a / (x^2) - 1 / x

theorem function_properties (a : ℝ) :
  (∀ x > 0, f a x = x / 4 + a / x - Real.log x - 3 / 2) →
  (f' a 1 = -2) →
  ∃ (x_min : ℝ),
    (a = 5 / 4) ∧
    (x_min = 5) ∧
    (∀ x ∈ Set.Ioo 0 x_min, (f' (5/4)) x < 0) ∧
    (∀ x ∈ Set.Ioi x_min, (f' (5/4)) x > 0) ∧
    (∀ x > 0, f (5/4) x ≥ f (5/4) x_min) ∧
    (f (5/4) x_min = -Real.log 5) :=
by sorry

end

end NUMINAMATH_CALUDE_function_properties_l1364_136435


namespace NUMINAMATH_CALUDE_rowing_distance_l1364_136477

theorem rowing_distance (v_man : ℝ) (v_river : ℝ) (total_time : ℝ) :
  v_man = 8 →
  v_river = 2 →
  total_time = 1 →
  ∃ (distance : ℝ),
    distance / (v_man - v_river) + distance / (v_man + v_river) = total_time ∧
    2 * distance = 7.5 :=
by sorry

end NUMINAMATH_CALUDE_rowing_distance_l1364_136477


namespace NUMINAMATH_CALUDE_youngest_brother_age_l1364_136464

/-- Represents the ages of Rick and his brothers -/
structure FamilyAges where
  rick : ℕ
  oldest : ℕ
  middle : ℕ
  smallest : ℕ
  youngest : ℕ

/-- Defines the relationships between the ages in the family -/
def validFamilyAges (ages : FamilyAges) : Prop :=
  ages.rick = 15 ∧
  ages.oldest = 2 * ages.rick ∧
  ages.middle = ages.oldest / 3 ∧
  ages.smallest = ages.middle / 2 ∧
  ages.youngest = ages.smallest - 2

/-- Theorem stating that given the family age relationships, the youngest brother is 3 years old -/
theorem youngest_brother_age (ages : FamilyAges) (h : validFamilyAges ages) : ages.youngest = 3 := by
  sorry

end NUMINAMATH_CALUDE_youngest_brother_age_l1364_136464


namespace NUMINAMATH_CALUDE_addition_is_unique_solution_l1364_136465

-- Define the possible operations
inductive Operation
| Add
| Subtract
| Multiply
| Divide

-- Define a function to apply the operation
def applyOperation (op : Operation) (a b : Int) : Int :=
  match op with
  | Operation.Add => a + b
  | Operation.Subtract => a - b
  | Operation.Multiply => a * b
  | Operation.Divide => a / b

-- Theorem statement
theorem addition_is_unique_solution :
  ∃! op : Operation, applyOperation op 5 (-5) = 0 :=
sorry

end NUMINAMATH_CALUDE_addition_is_unique_solution_l1364_136465


namespace NUMINAMATH_CALUDE_sams_shirts_l1364_136473

theorem sams_shirts (short_sleeve : ℕ) (long_sleeve : ℕ) (washed : ℕ) (unwashed : ℕ) : 
  long_sleeve = 23 →
  washed = 29 →
  unwashed = 34 →
  short_sleeve + long_sleeve = washed + unwashed →
  short_sleeve = 40 := by
sorry

end NUMINAMATH_CALUDE_sams_shirts_l1364_136473


namespace NUMINAMATH_CALUDE_circle_radius_is_one_l1364_136483

/-- The equation of a circle is x^2 + y^2 + 2x + 2y + 1 = 0. This theorem proves that the radius of this circle is 1. -/
theorem circle_radius_is_one :
  ∃ (h : ℝ → ℝ → Prop),
    (∀ x y : ℝ, h x y ↔ x^2 + y^2 + 2*x + 2*y + 1 = 0) →
    (∃ c : ℝ × ℝ, ∃ r : ℝ, r = 1 ∧ ∀ x y : ℝ, h x y ↔ (x - c.1)^2 + (y - c.2)^2 = r^2) :=
by sorry

end NUMINAMATH_CALUDE_circle_radius_is_one_l1364_136483


namespace NUMINAMATH_CALUDE_total_fruits_picked_l1364_136408

def mike_pears : ℕ := 8
def jason_pears : ℕ := 7
def fred_apples : ℕ := 6
def sarah_apples : ℕ := 12

theorem total_fruits_picked : mike_pears + jason_pears + fred_apples + sarah_apples = 33 := by
  sorry

end NUMINAMATH_CALUDE_total_fruits_picked_l1364_136408


namespace NUMINAMATH_CALUDE_red_apples_count_l1364_136413

theorem red_apples_count (total green yellow : ℕ) (h1 : total = 19) (h2 : green = 2) (h3 : yellow = 14) :
  total - (green + yellow) = 3 := by
  sorry

end NUMINAMATH_CALUDE_red_apples_count_l1364_136413


namespace NUMINAMATH_CALUDE_total_length_l1364_136443

def problem (rubber pen pencil marker ruler : ℝ) : Prop :=
  pen = rubber + 3 ∧
  pencil = pen + 2 ∧
  pencil = 12 ∧
  ruler = 3 * rubber ∧
  marker = (pen + rubber + pencil) / 3 ∧
  marker = ruler / 2

theorem total_length (rubber pen pencil marker ruler : ℝ) 
  (h : problem rubber pen pencil marker ruler) : 
  rubber + pen + pencil + marker + ruler = 60.5 := by
  sorry

end NUMINAMATH_CALUDE_total_length_l1364_136443


namespace NUMINAMATH_CALUDE_sector_angle_l1364_136418

/-- Given a circular sector with radius 2 and area 8, prove that its central angle is 4 radians -/
theorem sector_angle (r : ℝ) (area : ℝ) (θ : ℝ) 
  (h_radius : r = 2)
  (h_area : area = 8)
  (h_sector_area : area = 1/2 * r^2 * θ) : θ = 4 := by
  sorry

end NUMINAMATH_CALUDE_sector_angle_l1364_136418


namespace NUMINAMATH_CALUDE_left_handed_fraction_is_four_ninths_l1364_136482

/-- Represents the ratio of red to blue participants -/
def red_to_blue_ratio : ℚ := 2

/-- Fraction of left-handed red participants -/
def left_handed_red_fraction : ℚ := 1/3

/-- Fraction of left-handed blue participants -/
def left_handed_blue_fraction : ℚ := 2/3

/-- Theorem stating the fraction of left-handed participants -/
theorem left_handed_fraction_is_four_ninths 
  (h1 : red_to_blue_ratio = 2)
  (h2 : left_handed_red_fraction = 1/3)
  (h3 : left_handed_blue_fraction = 2/3) :
  (red_to_blue_ratio * left_handed_red_fraction + left_handed_blue_fraction) / 
  (red_to_blue_ratio + 1) = 4/9 := by
  sorry

#check left_handed_fraction_is_four_ninths

end NUMINAMATH_CALUDE_left_handed_fraction_is_four_ninths_l1364_136482


namespace NUMINAMATH_CALUDE_ferris_wheel_capacity_l1364_136421

theorem ferris_wheel_capacity (total_seats broken_seats people_riding : ℕ) 
  (h1 : total_seats = 18)
  (h2 : broken_seats = 10)
  (h3 : people_riding = 120) :
  people_riding / (total_seats - broken_seats) = 15 := by
  sorry

end NUMINAMATH_CALUDE_ferris_wheel_capacity_l1364_136421


namespace NUMINAMATH_CALUDE_hyperbola_vertex_to_asymptote_distance_l1364_136437

/-- Given a hyperbola with equation x²/a² - y²/3 = 1 and eccentricity 2,
    the distance from its vertex to its asymptote is √3/2 -/
theorem hyperbola_vertex_to_asymptote_distance
  (a : ℝ) -- Semi-major axis
  (h1 : a > 0) -- a is positive
  (h2 : (a^2 + 3) / a^2 = 4) -- Eccentricity condition
  : Real.sqrt 3 / 2 = 
    abs (-Real.sqrt 3 * a) / Real.sqrt (3 + 1) := by sorry

end NUMINAMATH_CALUDE_hyperbola_vertex_to_asymptote_distance_l1364_136437


namespace NUMINAMATH_CALUDE_sector_radius_l1364_136466

theorem sector_radius (l a : ℝ) (hl : l = 10 * Real.pi) (ha : a = 60 * Real.pi) :
  ∃ r : ℝ, r = 12 ∧ a = (1 / 2) * l * r := by
  sorry

end NUMINAMATH_CALUDE_sector_radius_l1364_136466


namespace NUMINAMATH_CALUDE_max_sum_nonnegative_reals_l1364_136476

theorem max_sum_nonnegative_reals (a b c : ℝ) :
  a ≥ 0 ∧ b ≥ 0 ∧ c ≥ 0 →
  a^2 + b^2 + c^2 = 52 →
  a*b + b*c + c*a = 28 →
  a + b + c ≤ 6 * Real.sqrt 3 :=
by sorry

end NUMINAMATH_CALUDE_max_sum_nonnegative_reals_l1364_136476


namespace NUMINAMATH_CALUDE_negation_of_universal_proposition_l1364_136462

open Real

theorem negation_of_universal_proposition :
  (¬ (∀ x : ℝ, 0 < x ∧ x < π / 2 → x < tan x)) ↔
  (∃ x : ℝ, 0 < x ∧ x < π / 2 ∧ x ≥ tan x) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_universal_proposition_l1364_136462


namespace NUMINAMATH_CALUDE_units_digit_of_quotient_l1364_136401

theorem units_digit_of_quotient (h : 7 ∣ (4^1985 + 7^1985)) :
  (4^1985 + 7^1985) / 7 % 10 = 2 := by
sorry

end NUMINAMATH_CALUDE_units_digit_of_quotient_l1364_136401
