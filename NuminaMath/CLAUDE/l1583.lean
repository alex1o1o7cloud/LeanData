import Mathlib

namespace NUMINAMATH_CALUDE_evaluate_g_l1583_158381

def g (x : ℝ) : ℝ := 3 * x^2 - 5 * x + 7

theorem evaluate_g : 3 * g 2 - 4 * g (-2) = -89 := by
  sorry

end NUMINAMATH_CALUDE_evaluate_g_l1583_158381


namespace NUMINAMATH_CALUDE_intersection_point_is_on_line_and_plane_l1583_158366

/-- The line equation represented as a function of a parameter t -/
def line_equation (t : ℝ) : ℝ × ℝ × ℝ :=
  (-3, 2 - 3*t, -5 + 11*t)

/-- The plane equation -/
def plane_equation (p : ℝ × ℝ × ℝ) : Prop :=
  let (x, y, z) := p
  5*x + 7*y + 9*z - 32 = 0

/-- The intersection point -/
def intersection_point : ℝ × ℝ × ℝ := (-3, -1, 6)

theorem intersection_point_is_on_line_and_plane :
  ∃ t : ℝ, line_equation t = intersection_point ∧ plane_equation intersection_point := by
  sorry

#check intersection_point_is_on_line_and_plane

end NUMINAMATH_CALUDE_intersection_point_is_on_line_and_plane_l1583_158366


namespace NUMINAMATH_CALUDE_fourth_root_is_four_l1583_158376

/-- The polynomial with coefficients c and d -/
def polynomial (c d x : ℝ) : ℝ :=
  c * x^4 + (c + 3*d) * x^3 + (d - 4*c) * x^2 + (10 - c) * x + (5 - 2*d)

/-- The theorem stating that if -1, 2, and -3 are roots of the polynomial,
    then 4 is the fourth root -/
theorem fourth_root_is_four (c d : ℝ) :
  polynomial c d (-1) = 0 →
  polynomial c d 2 = 0 →
  polynomial c d (-3) = 0 →
  ∃ x : ℝ, x ≠ -1 ∧ x ≠ 2 ∧ x ≠ -3 ∧ polynomial c d x = 0 ∧ x = 4 := by
  sorry

end NUMINAMATH_CALUDE_fourth_root_is_four_l1583_158376


namespace NUMINAMATH_CALUDE_fixed_point_of_exponential_function_l1583_158322

/-- The function f(x) = a^(2x-1) + 1 passes through (1/2, 2) for all a > 0 and a ≠ 1 -/
theorem fixed_point_of_exponential_function (a : ℝ) (ha : a > 0) (ha_ne_one : a ≠ 1) :
  let f : ℝ → ℝ := λ x => a^(2*x - 1) + 1
  f (1/2) = 2 := by
  sorry

end NUMINAMATH_CALUDE_fixed_point_of_exponential_function_l1583_158322


namespace NUMINAMATH_CALUDE_polynomial_form_l1583_158331

/-- A real-coefficient polynomial function -/
def RealPolynomial := ℝ → ℝ

/-- The condition that needs to be satisfied by the polynomial -/
def SatisfiesCondition (P : RealPolynomial) : Prop :=
  ∀ (a b c : ℝ), a * b + b * c + c * a = 0 →
    P (a - b) + P (b - c) + P (c - a) = 2 * P (a + b + c)

/-- The theorem stating the form of polynomials satisfying the condition -/
theorem polynomial_form (P : RealPolynomial) 
    (h : SatisfiesCondition P) : 
    ∃ (α β : ℝ), ∀ (x : ℝ), P x = α * x^4 + β * x^2 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_form_l1583_158331


namespace NUMINAMATH_CALUDE_hyperbola_real_axis_length_l1583_158360

/-- The length of the real axis of the hyperbola x²/4 - y² = 1 is 4. -/
theorem hyperbola_real_axis_length :
  let h : ℝ → ℝ → Prop := λ x y ↦ x^2/4 - y^2 = 1
  ∃ a : ℝ, a > 0 ∧ (∀ x y, h x y ↔ x^2/a^2 - y^2 = 1) ∧ 2*a = 4 :=
sorry

end NUMINAMATH_CALUDE_hyperbola_real_axis_length_l1583_158360


namespace NUMINAMATH_CALUDE_quadratic_always_positive_l1583_158350

theorem quadratic_always_positive (k : ℝ) : 
  (∀ x : ℝ, x^2 - (k - 5) * x + k + 2 > 0) ↔ 
  (k > 7 - 4 * Real.sqrt 2 ∧ k < 7 + 4 * Real.sqrt 2) :=
sorry

end NUMINAMATH_CALUDE_quadratic_always_positive_l1583_158350


namespace NUMINAMATH_CALUDE_factorization_proof_l1583_158351

theorem factorization_proof (x : ℝ) : 5*x*(x-5) + 7*(x-5) = (x-5)*(5*x+7) := by
  sorry

end NUMINAMATH_CALUDE_factorization_proof_l1583_158351


namespace NUMINAMATH_CALUDE_sequence_sum_l1583_158340

/-- Given an 8-term sequence where C = 10 and the sum of any three consecutive terms is 40,
    prove that A + H = 30 -/
theorem sequence_sum (A B C D E F G H : ℝ) 
  (hC : C = 10)
  (hABC : A + B + C = 40)
  (hBCD : B + C + D = 40)
  (hCDE : C + D + E = 40)
  (hDEF : D + E + F = 40)
  (hEFG : E + F + G = 40)
  (hFGH : F + G + H = 40) :
  A + H = 30 := by
  sorry

end NUMINAMATH_CALUDE_sequence_sum_l1583_158340


namespace NUMINAMATH_CALUDE_pie_eating_contest_l1583_158307

/-- Pie-eating contest problem -/
theorem pie_eating_contest (bill : ℕ) (adam sierra taylor total : ℕ) : 
  adam = bill + 3 →
  sierra = 2 * bill →
  sierra = 12 →
  taylor = (adam + bill + sierra) / 3 →
  total = adam + bill + sierra + taylor →
  total = 36 := by
sorry

end NUMINAMATH_CALUDE_pie_eating_contest_l1583_158307


namespace NUMINAMATH_CALUDE_sum_of_reciprocals_l1583_158312

theorem sum_of_reciprocals (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (h : x + y = x * y - 1) :
  1 / x + 1 / y = 1 - 1 / (x * y) := by
  sorry

end NUMINAMATH_CALUDE_sum_of_reciprocals_l1583_158312


namespace NUMINAMATH_CALUDE_investment_value_change_l1583_158305

theorem investment_value_change (k m : ℝ) : 
  let increase_factor := 1 + k / 100
  let decrease_factor := 1 - m / 100
  let overall_factor := increase_factor * decrease_factor
  overall_factor = 1 + (k - m - k * m / 100) / 100 :=
by sorry

end NUMINAMATH_CALUDE_investment_value_change_l1583_158305


namespace NUMINAMATH_CALUDE_recliner_sales_increase_l1583_158386

/-- Proves that a 20% price reduction and 28% gross revenue increase results in a 60% increase in sales volume -/
theorem recliner_sales_increase 
  (original_price : ℝ) 
  (original_quantity : ℝ) 
  (new_price : ℝ) 
  (new_quantity : ℝ) 
  (h1 : new_price = 0.80 * original_price) 
  (h2 : new_price * new_quantity = 1.28 * (original_price * original_quantity)) : 
  (new_quantity - original_quantity) / original_quantity = 0.60 := by
sorry

end NUMINAMATH_CALUDE_recliner_sales_increase_l1583_158386


namespace NUMINAMATH_CALUDE_sqrt3_minus_1_power_equation_solution_is_16_l1583_158346

theorem sqrt3_minus_1_power_equation : ∃ (N : ℕ), (Real.sqrt 3 - 1) ^ N = 4817152 - 2781184 * Real.sqrt 3 :=
by sorry

theorem solution_is_16 : ∃! (N : ℕ), (Real.sqrt 3 - 1) ^ N = 4817152 - 2781184 * Real.sqrt 3 ∧ N = 16 :=
by sorry

end NUMINAMATH_CALUDE_sqrt3_minus_1_power_equation_solution_is_16_l1583_158346


namespace NUMINAMATH_CALUDE_quadratic_form_k_value_l1583_158306

theorem quadratic_form_k_value (x : ℝ) : 
  ∃ (a h k : ℝ), x^2 - 7*x = a*(x - h)^2 + k ∧ k = -49/4 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_form_k_value_l1583_158306


namespace NUMINAMATH_CALUDE_final_value_is_four_l1583_158375

def increment_sequence (initial : ℕ) : ℕ :=
  let step1 := initial + 1
  let step2 := step1 + 2
  step2

theorem final_value_is_four :
  increment_sequence 1 = 4 := by
  sorry

end NUMINAMATH_CALUDE_final_value_is_four_l1583_158375


namespace NUMINAMATH_CALUDE_ellipse_focal_length_l1583_158374

/-- Given an ellipse with equation x^2/23 + y^2/32 = 1, its focal length is 6. -/
theorem ellipse_focal_length : ∀ (x y : ℝ), x^2/23 + y^2/32 = 1 → ∃ (c : ℝ), c = 3 ∧ 2*c = 6 :=
by
  sorry

end NUMINAMATH_CALUDE_ellipse_focal_length_l1583_158374


namespace NUMINAMATH_CALUDE_geometric_sequence_increasing_condition_l1583_158333

/-- A geometric sequence with first term a₁ and common ratio q -/
def GeometricSequence (a₁ q : ℝ) : ℕ → ℝ :=
  fun n ↦ a₁ * q^(n - 1)

theorem geometric_sequence_increasing_condition (a₁ q : ℝ) :
  (a₁ < 0 ∧ 0 < q ∧ q < 1 →
    ∀ n : ℕ, n > 0 → GeometricSequence a₁ q (n + 1) > GeometricSequence a₁ q n) ∧
  (∃ a₁' q' : ℝ, (∀ n : ℕ, n > 0 → GeometricSequence a₁' q' (n + 1) > GeometricSequence a₁' q' n) ∧
    ¬(a₁' < 0 ∧ 0 < q' ∧ q' < 1)) :=
by sorry

end NUMINAMATH_CALUDE_geometric_sequence_increasing_condition_l1583_158333


namespace NUMINAMATH_CALUDE_max_EH_value_l1583_158393

/-- A cyclic quadrilateral with integer side lengths --/
structure CyclicQuadrilateral where
  EF : ℕ
  FG : ℕ
  GH : ℕ
  EH : ℕ
  distinct : EF ≠ FG ∧ EF ≠ GH ∧ EF ≠ EH ∧ FG ≠ GH ∧ FG ≠ EH ∧ GH ≠ EH
  less_than_20 : EF < 20 ∧ FG < 20 ∧ GH < 20 ∧ EH < 20
  cyclic_property : EF * GH = FG * EH

/-- The maximum possible value of EH in a cyclic quadrilateral with given constraints --/
theorem max_EH_value (q : CyclicQuadrilateral) :
  (∀ q' : CyclicQuadrilateral, q'.EH ≤ q.EH) → q.EH^2 = 394 :=
sorry

end NUMINAMATH_CALUDE_max_EH_value_l1583_158393


namespace NUMINAMATH_CALUDE_nearest_integer_to_three_plus_sqrt_five_fourth_power_l1583_158319

theorem nearest_integer_to_three_plus_sqrt_five_fourth_power :
  ∃ (n : ℤ), ∀ (m : ℤ), |((3 : ℝ) + Real.sqrt 5)^4 - n| ≤ |((3 : ℝ) + Real.sqrt 5)^4 - m| ∧ n = 752 :=
sorry

end NUMINAMATH_CALUDE_nearest_integer_to_three_plus_sqrt_five_fourth_power_l1583_158319


namespace NUMINAMATH_CALUDE_time_difference_walk_bike_l1583_158303

/-- The number of blocks between Youseff's home and office -/
def B : ℕ := 21

/-- The time in minutes it takes to walk one block -/
def walk_time_per_block : ℚ := 1

/-- The time in minutes it takes to bike one block -/
def bike_time_per_block : ℚ := 20 / 60

/-- The total time in minutes it takes to walk to work -/
def total_walk_time : ℚ := B * walk_time_per_block

/-- The total time in minutes it takes to bike to work -/
def total_bike_time : ℚ := B * bike_time_per_block

theorem time_difference_walk_bike : 
  total_walk_time - total_bike_time = 14 := by
  sorry

end NUMINAMATH_CALUDE_time_difference_walk_bike_l1583_158303


namespace NUMINAMATH_CALUDE_aquafaba_to_egg_white_ratio_l1583_158326

/-- Given the number of cakes, egg whites per cake, and total tablespoons of aquafaba,
    prove that the ratio of tablespoons of aquafaba to egg whites is 2:1 -/
theorem aquafaba_to_egg_white_ratio
  (num_cakes : ℕ)
  (egg_whites_per_cake : ℕ)
  (total_aquafaba : ℕ)
  (h1 : num_cakes = 2)
  (h2 : egg_whites_per_cake = 8)
  (h3 : total_aquafaba = 32) :
  (total_aquafaba : ℚ) / (num_cakes * egg_whites_per_cake) = 2 := by
  sorry

end NUMINAMATH_CALUDE_aquafaba_to_egg_white_ratio_l1583_158326


namespace NUMINAMATH_CALUDE_product_of_sum_and_cube_sum_l1583_158367

theorem product_of_sum_and_cube_sum (x y : ℝ) 
  (sum_eq : x + y = 10) 
  (cube_sum_eq : x^3 + y^3 = 172) : 
  x * y = 41.4 := by
sorry

end NUMINAMATH_CALUDE_product_of_sum_and_cube_sum_l1583_158367


namespace NUMINAMATH_CALUDE_triangle_angle_measure_l1583_158365

theorem triangle_angle_measure (X Y Z : Real) (h1 : X = 72) 
  (h2 : Y = 4 * Z + 10) (h3 : X + Y + Z = 180) : Z = 19.6 := by
  sorry

end NUMINAMATH_CALUDE_triangle_angle_measure_l1583_158365


namespace NUMINAMATH_CALUDE_min_value_a_l1583_158390

theorem min_value_a (a : ℝ) (ha : a > 0) : 
  (∀ x y : ℝ, x > 0 → y > 0 → (x + y) * (a / x + 4 / y) ≥ 16) → 
  a ≥ 4 :=
sorry

end NUMINAMATH_CALUDE_min_value_a_l1583_158390


namespace NUMINAMATH_CALUDE_jenny_distance_relationship_l1583_158384

/-- Given Jenny's running and walking speeds and times, prove the relationship between distances -/
theorem jenny_distance_relationship 
  (x : ℝ) -- Jenny's running speed in miles per hour
  (y : ℝ) -- Jenny's walking speed in miles per hour
  (r : ℝ) -- Time spent running in minutes
  (w : ℝ) -- Time spent walking in minutes
  (d : ℝ) -- Difference in distance (running - walking) in miles
  (hx : x > 0) -- Assumption: running speed is positive
  (hy : y > 0) -- Assumption: walking speed is positive
  (hr : r ≥ 0) -- Assumption: time spent running is non-negative
  (hw : w ≥ 0) -- Assumption: time spent walking is non-negative
  : x * r - y * w = 60 * d :=
by sorry

end NUMINAMATH_CALUDE_jenny_distance_relationship_l1583_158384


namespace NUMINAMATH_CALUDE_unique_solution_sin_cos_equation_l1583_158355

theorem unique_solution_sin_cos_equation :
  ∃! x : ℝ, 0 ≤ x ∧ x ≤ π / 2 ∧ Real.sin (Real.cos x) = Real.cos (Real.sin x) := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_sin_cos_equation_l1583_158355


namespace NUMINAMATH_CALUDE_max_value_polynomial_l1583_158371

theorem max_value_polynomial (x y : ℝ) (h : x + y = 4) :
  x^4*y + x^3*y + x^2*y + x*y + x*y^2 + x*y^3 + x*y^4 ≤ 7225/28 := by
  sorry

end NUMINAMATH_CALUDE_max_value_polynomial_l1583_158371


namespace NUMINAMATH_CALUDE_banana_arrangements_l1583_158394

theorem banana_arrangements :
  let total_letters : ℕ := 6
  let a_count : ℕ := 3
  let n_count : ℕ := 2
  let b_count : ℕ := 1
  (total_letters! / (a_count! * n_count! * b_count!)) = 60 :=
by sorry

end NUMINAMATH_CALUDE_banana_arrangements_l1583_158394


namespace NUMINAMATH_CALUDE_factorization_equality_l1583_158389

theorem factorization_equality (a : ℝ) : a^2 - 3*a = a*(a - 3) := by
  sorry

end NUMINAMATH_CALUDE_factorization_equality_l1583_158389


namespace NUMINAMATH_CALUDE_solution_set_correct_inequality_factorization_l1583_158323

/-- The solution set of the quadratic inequality ax^2 + (a-2)x - 2 ≤ 0 -/
def solution_set (a : ℝ) : Set ℝ :=
  if a = 0 then Set.Ici (-1)
  else if a > 0 then Set.Icc (-1) (2/a)
  else if -2 < a ∧ a < 0 then Set.Iic (2/a) ∪ Set.Ici (-1)
  else if a < -2 then Set.Iic (-1) ∪ Set.Ici (2/a)
  else Set.univ

/-- Theorem stating that the solution_set function correctly solves the quadratic inequality -/
theorem solution_set_correct (a : ℝ) (x : ℝ) :
  x ∈ solution_set a ↔ a * x^2 + (a - 2) * x - 2 ≤ 0 := by
  sorry

/-- Theorem stating that the quadratic inequality can be rewritten as a product of linear factors -/
theorem inequality_factorization (a : ℝ) (x : ℝ) :
  a * x^2 + (a - 2) * x - 2 = (a * x - 2) * (x + 1) := by
  sorry

end NUMINAMATH_CALUDE_solution_set_correct_inequality_factorization_l1583_158323


namespace NUMINAMATH_CALUDE_total_weight_of_pets_l1583_158314

/-- The total weight of four pets given specific weight relationships -/
theorem total_weight_of_pets (evan_dog : ℝ) (ivan_dog : ℝ) (kara_cat : ℝ) (lisa_parrot : ℝ) 
  (h1 : evan_dog = 63)
  (h2 : evan_dog = 7 * ivan_dog)
  (h3 : kara_cat = 5 * (evan_dog + ivan_dog))
  (h4 : lisa_parrot = 3 * (evan_dog + ivan_dog + kara_cat)) :
  evan_dog + ivan_dog + kara_cat + lisa_parrot = 1728 := by
  sorry

#check total_weight_of_pets

end NUMINAMATH_CALUDE_total_weight_of_pets_l1583_158314


namespace NUMINAMATH_CALUDE_newspaper_probability_l1583_158397

-- Define the time intervals
def delivery_start : ℝ := 6.5
def delivery_end : ℝ := 7.5
def departure_start : ℝ := 7.0
def departure_end : ℝ := 8.0

-- Define the probability function
def probability_of_getting_newspaper : ℝ := sorry

-- Theorem statement
theorem newspaper_probability :
  probability_of_getting_newspaper = 7 / 8 := by sorry

end NUMINAMATH_CALUDE_newspaper_probability_l1583_158397


namespace NUMINAMATH_CALUDE_average_weight_problem_l1583_158325

/-- Given the average weights of three people and two of them, prove the average weight of two of them. -/
theorem average_weight_problem (a b c : ℝ) : 
  (a + b + c) / 3 = 30 →  -- average weight of a, b, and c is 30 kg
  (a + b) / 2 = 25 →      -- average weight of a and b is 25 kg
  b = 16 →                -- weight of b is 16 kg
  (b + c) / 2 = 28        -- average weight of b and c is 28 kg
:= by sorry

end NUMINAMATH_CALUDE_average_weight_problem_l1583_158325


namespace NUMINAMATH_CALUDE_f_2018_equals_neg_2018_l1583_158357

def is_even_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = f x

def satisfies_equation (f : ℝ → ℝ) : Prop :=
  ∀ x, f x = -1 / f (x + 3)

theorem f_2018_equals_neg_2018
  (f : ℝ → ℝ)
  (h_even : is_even_function f)
  (h_eq : satisfies_equation f)
  (h_f4 : f 4 = -2018) :
  f 2018 = -2018 :=
sorry

end NUMINAMATH_CALUDE_f_2018_equals_neg_2018_l1583_158357


namespace NUMINAMATH_CALUDE_perpendicular_line_through_point_l1583_158308

/-- Given a line L1 with equation x - 2y - 2 = 0, prove that the line L2 with equation 2x + y - 2 = 0
    passes through the point (1,0) and is perpendicular to L1. -/
theorem perpendicular_line_through_point (x y : ℝ) : 
  (x - 2*y - 2 = 0) →  -- Equation of line L1
  (2*x + y - 2 = 0) →  -- Equation of line L2
  (2*1 + 0 - 2 = 0) ∧  -- L2 passes through (1,0)
  (1 * 2 = -1) -- Slopes of L1 and L2 are negative reciprocals
  := by sorry

end NUMINAMATH_CALUDE_perpendicular_line_through_point_l1583_158308


namespace NUMINAMATH_CALUDE_midpoint_coordinate_product_l1583_158387

/-- Given a line segment CD where C(5,4) is one endpoint and M(4,8) is the midpoint,
    the product of the coordinates of the other endpoint D is 36. -/
theorem midpoint_coordinate_product (C D M : ℝ × ℝ) : 
  C = (5, 4) →
  M = (4, 8) →
  M.1 = (C.1 + D.1) / 2 →
  M.2 = (C.2 + D.2) / 2 →
  D.1 * D.2 = 36 := by
sorry

end NUMINAMATH_CALUDE_midpoint_coordinate_product_l1583_158387


namespace NUMINAMATH_CALUDE_cats_vasyas_equality_l1583_158362

variable {α : Type*}
variable (C V : Set α)

theorem cats_vasyas_equality : C ∩ V = V ∩ C := by
  sorry

end NUMINAMATH_CALUDE_cats_vasyas_equality_l1583_158362


namespace NUMINAMATH_CALUDE_caffeine_in_coffee_l1583_158332

/-- The amount of caffeine in a cup of coffee -/
def caffeine_per_cup : ℝ := 80

/-- Lisa's daily caffeine limit in milligrams -/
def daily_limit : ℝ := 200

/-- The number of cups Lisa drinks -/
def cups_drunk : ℝ := 3

/-- The amount Lisa exceeds her limit by in milligrams -/
def excess_amount : ℝ := 40

theorem caffeine_in_coffee :
  caffeine_per_cup * cups_drunk = daily_limit + excess_amount :=
by sorry

end NUMINAMATH_CALUDE_caffeine_in_coffee_l1583_158332


namespace NUMINAMATH_CALUDE_four_rows_with_eight_people_l1583_158316

/-- Represents a seating arrangement with rows of 7 or 8 people -/
structure SeatingArrangement where
  total_people : ℕ
  rows_with_eight : ℕ
  rows_with_seven : ℕ

/-- Conditions for a valid seating arrangement -/
def is_valid_arrangement (s : SeatingArrangement) : Prop :=
  s.total_people = 46 ∧
  s.total_people = 8 * s.rows_with_eight + 7 * s.rows_with_seven

/-- Theorem stating that in a valid arrangement with 46 people, 
    there are exactly 4 rows with 8 people -/
theorem four_rows_with_eight_people 
  (s : SeatingArrangement) (h : is_valid_arrangement s) : 
  s.rows_with_eight = 4 := by
  sorry

end NUMINAMATH_CALUDE_four_rows_with_eight_people_l1583_158316


namespace NUMINAMATH_CALUDE_line_through_point_with_equal_intercepts_l1583_158330

-- Define a point in 2D space
structure Point where
  x : ℝ
  y : ℝ

-- Define a line in 2D space
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

def passes_through (l : Line) (p : Point) : Prop :=
  l.a * p.x + l.b * p.y + l.c = 0

def has_equal_intercepts (l : Line) : Prop :=
  l.a = l.b ∨ l.a = -l.b

theorem line_through_point_with_equal_intercepts :
  ∃ (l1 l2 : Line),
    (passes_through l1 ⟨1, 2⟩ ∧ has_equal_intercepts l1) ∧
    (passes_through l2 ⟨1, 2⟩ ∧ has_equal_intercepts l2) ∧
    ((l1.a = 1 ∧ l1.b = 1 ∧ l1.c = -3) ∨ (l2.a = 2 ∧ l2.b = -1 ∧ l2.c = 0)) :=
sorry

end NUMINAMATH_CALUDE_line_through_point_with_equal_intercepts_l1583_158330


namespace NUMINAMATH_CALUDE_mode_estimate_is_tallest_rectangle_midpoint_l1583_158378

/-- Represents a rectangle in a frequency distribution histogram --/
structure HistogramRectangle where
  height : ℝ
  base_midpoint : ℝ

/-- Represents a sample frequency distribution histogram --/
structure FrequencyHistogram where
  rectangles : List HistogramRectangle

/-- Finds the tallest rectangle in a frequency histogram --/
def tallestRectangle (h : FrequencyHistogram) : HistogramRectangle :=
  sorry

/-- Estimates the mode of a dataset from a frequency histogram --/
def estimateMode (h : FrequencyHistogram) : ℝ :=
  (tallestRectangle h).base_midpoint

theorem mode_estimate_is_tallest_rectangle_midpoint (h : FrequencyHistogram) :
  estimateMode h = (tallestRectangle h).base_midpoint :=
sorry

end NUMINAMATH_CALUDE_mode_estimate_is_tallest_rectangle_midpoint_l1583_158378


namespace NUMINAMATH_CALUDE_sampling_theorem_l1583_158321

/-- Staff distribution in departments A and B -/
structure StaffDistribution where
  maleA : ℕ
  femaleA : ℕ
  maleB : ℕ
  femaleB : ℕ

/-- Sampling method for selecting staff members -/
inductive SamplingMethod
  | Stratified : SamplingMethod

/-- Result of the sampling process -/
structure SamplingResult where
  fromA : ℕ
  fromB : ℕ
  totalSelected : ℕ

/-- Theorem stating the probability of selecting at least one female from A
    and the expectation of the number of males selected -/
theorem sampling_theorem (sd : StaffDistribution) (sm : SamplingMethod) (sr : SamplingResult) :
  sd.maleA = 6 ∧ sd.femaleA = 4 ∧ sd.maleB = 3 ∧ sd.femaleB = 2 ∧
  sm = SamplingMethod.Stratified ∧
  sr.fromA = 2 ∧ sr.fromB = 1 ∧ sr.totalSelected = 3 →
  (ProbabilityAtLeastOneFemaleFromA = 2/3) ∧
  (ExpectationOfMalesSelected = 9/5) := by
  sorry

end NUMINAMATH_CALUDE_sampling_theorem_l1583_158321


namespace NUMINAMATH_CALUDE_stamp_collection_value_l1583_158335

theorem stamp_collection_value (total_stamps : ℕ) (sample_stamps : ℕ) (sample_value : ℚ) 
  (h1 : total_stamps = 18)
  (h2 : sample_stamps = 6)
  (h3 : sample_value = 15) : 
  (total_stamps : ℚ) * (sample_value / sample_stamps) = 45 := by
  sorry

end NUMINAMATH_CALUDE_stamp_collection_value_l1583_158335


namespace NUMINAMATH_CALUDE_acute_angles_are_first_quadrant_l1583_158349

-- Define acute angle
def is_acute_angle (θ : ℝ) : Prop := 0 < θ ∧ θ < Real.pi / 2

-- Define first quadrant angle
def is_first_quadrant_angle (θ : ℝ) : Prop := 0 < θ ∧ θ < Real.pi / 2

-- Theorem: All acute angles are first quadrant angles
theorem acute_angles_are_first_quadrant :
  ∀ θ : ℝ, is_acute_angle θ → is_first_quadrant_angle θ :=
by
  sorry


end NUMINAMATH_CALUDE_acute_angles_are_first_quadrant_l1583_158349


namespace NUMINAMATH_CALUDE_digit_replacement_theorem_l1583_158300

def first_number : ℕ := 631927
def second_number : ℕ := 590265
def given_sum : ℕ := 1192192

def replace_digit (n : ℕ) (d e : ℕ) : ℕ := 
  sorry

theorem digit_replacement_theorem :
  ∃ (d e : ℕ), d ≠ e ∧ d < 10 ∧ e < 10 ∧
  (replace_digit first_number d e) + (replace_digit second_number d e) = 
    replace_digit given_sum d e ∧
  d + e = 6 := by
  sorry

end NUMINAMATH_CALUDE_digit_replacement_theorem_l1583_158300


namespace NUMINAMATH_CALUDE_tangent_length_specific_circle_l1583_158391

/-- A circle passing through three points -/
structure Circle where
  p1 : ℝ × ℝ
  p2 : ℝ × ℝ
  p3 : ℝ × ℝ

/-- The length of the tangent from a point to a circle -/
def tangentLength (p : ℝ × ℝ) (c : Circle) : ℝ :=
  sorry  -- Definition omitted as it's not given in the problem conditions

/-- The theorem stating the length of the tangent from the origin to the specific circle -/
theorem tangent_length_specific_circle :
  let origin : ℝ × ℝ := (0, 0)
  let c : Circle := { p1 := (3, 4), p2 := (6, 8), p3 := (5, 13) }
  tangentLength origin c = 5 * Real.sqrt 2 := by
  sorry


end NUMINAMATH_CALUDE_tangent_length_specific_circle_l1583_158391


namespace NUMINAMATH_CALUDE_decoration_cost_theorem_l1583_158315

/-- Calculates the total cost of decorations for a wedding reception. -/
def total_decoration_cost (num_tables : ℕ) 
                          (tablecloth_cost : ℕ) 
                          (place_setting_cost : ℕ) 
                          (place_settings_per_table : ℕ) 
                          (roses_per_centerpiece : ℕ) 
                          (rose_cost : ℕ) 
                          (lilies_per_centerpiece : ℕ) 
                          (lily_cost : ℕ) 
                          (daisies_per_centerpiece : ℕ) 
                          (daisy_cost : ℕ) 
                          (sunflowers_per_centerpiece : ℕ) 
                          (sunflower_cost : ℕ) 
                          (lighting_cost : ℕ) : ℕ :=
  let tablecloth_total := num_tables * tablecloth_cost
  let place_setting_total := num_tables * place_settings_per_table * place_setting_cost
  let centerpiece_cost := roses_per_centerpiece * rose_cost + 
                          lilies_per_centerpiece * lily_cost + 
                          daisies_per_centerpiece * daisy_cost + 
                          sunflowers_per_centerpiece * sunflower_cost
  let centerpiece_total := num_tables * centerpiece_cost
  tablecloth_total + place_setting_total + centerpiece_total + lighting_cost

theorem decoration_cost_theorem : 
  total_decoration_cost 30 25 12 6 15 6 20 5 5 3 3 4 450 = 9870 := by
  sorry

end NUMINAMATH_CALUDE_decoration_cost_theorem_l1583_158315


namespace NUMINAMATH_CALUDE_parabola_shift_left_2_l1583_158363

/-- Represents a parabola in the form y = (x - h)^2 + k, where (h, k) is the vertex -/
structure Parabola where
  h : ℝ
  k : ℝ

/-- The equation of a parabola given x -/
def parabola_equation (p : Parabola) (x : ℝ) : ℝ :=
  (x - p.h)^2 + p.k

/-- Shifts a parabola horizontally -/
def shift_parabola (p : Parabola) (shift : ℝ) : Parabola :=
  { h := p.h + shift, k := p.k }

theorem parabola_shift_left_2 :
  let original := Parabola.mk 0 0
  let shifted := shift_parabola original (-2)
  ∀ x, parabola_equation shifted x = (x + 2)^2 := by
  sorry

end NUMINAMATH_CALUDE_parabola_shift_left_2_l1583_158363


namespace NUMINAMATH_CALUDE_sqrt_3_sum_square_l1583_158359

theorem sqrt_3_sum_square (x y : ℝ) : x = Real.sqrt 3 + 1 → y = Real.sqrt 3 - 1 → x^2 + x*y + y^2 = 10 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_3_sum_square_l1583_158359


namespace NUMINAMATH_CALUDE_fifteenth_triangular_number_l1583_158343

/-- The nth triangular number -/
def triangularNumber (n : ℕ) : ℕ := n * (n + 1) / 2

/-- The 15th triangular number is 120 -/
theorem fifteenth_triangular_number : triangularNumber 15 = 120 := by
  sorry

end NUMINAMATH_CALUDE_fifteenth_triangular_number_l1583_158343


namespace NUMINAMATH_CALUDE_intersection_empty_iff_intersection_equals_A_iff_l1583_158388

-- Define sets A and B
def A (a : ℝ) : Set ℝ := { x | a ≤ x ∧ x ≤ a + 2 }
def B : Set ℝ := { x | x ≤ 0 ∨ x ≥ 4 }

-- Theorem 1
theorem intersection_empty_iff (a : ℝ) : A a ∩ B = ∅ ↔ 0 < a ∧ a < 2 := by sorry

-- Theorem 2
theorem intersection_equals_A_iff (a : ℝ) : A a ∩ B = A a ↔ a ≤ -2 ∨ a ≥ 4 := by sorry

end NUMINAMATH_CALUDE_intersection_empty_iff_intersection_equals_A_iff_l1583_158388


namespace NUMINAMATH_CALUDE_condition_analysis_l1583_158318

theorem condition_analysis (a b c : ℝ) (h : a > b ∧ b > c) :
  (∀ a b c, a + b + c = 0 → a * b > a * c) ∧
  (∃ a b c, a * b > a * c ∧ a + b + c ≠ 0) := by
  sorry

end NUMINAMATH_CALUDE_condition_analysis_l1583_158318


namespace NUMINAMATH_CALUDE_stewart_farm_sheep_count_l1583_158327

theorem stewart_farm_sheep_count :
  ∀ (sheep horses : ℕ) (horse_food_per_day total_food : ℝ),
    sheep * 7 = horses * 6 →
    horse_food_per_day = 230 →
    total_food = 12880 →
    horses * horse_food_per_day = total_food →
    sheep = 48 := by
  sorry

end NUMINAMATH_CALUDE_stewart_farm_sheep_count_l1583_158327


namespace NUMINAMATH_CALUDE_log_2_base_10_bounds_l1583_158345

theorem log_2_base_10_bounds :
  (10^2 = 100) →
  (10^3 = 1000) →
  (2^7 = 128) →
  (2^10 = 1024) →
  (2 / 7 : ℝ) < Real.log 2 / Real.log 10 ∧ Real.log 2 / Real.log 10 < (3 / 10 : ℝ) :=
by sorry

end NUMINAMATH_CALUDE_log_2_base_10_bounds_l1583_158345


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l1583_158328

/-- An arithmetic sequence -/
def ArithmeticSequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

/-- The sum of five consecutive terms equals 450 -/
def SumCondition (a : ℕ → ℝ) : Prop :=
  a 3 + a 4 + a 5 + a 6 + a 7 = 450

theorem arithmetic_sequence_sum (a : ℕ → ℝ) :
  ArithmeticSequence a → SumCondition a → a 2 + a 8 = 180 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l1583_158328


namespace NUMINAMATH_CALUDE_complex_absolute_value_l1583_158304

theorem complex_absolute_value (t : ℝ) (h : t > 0) :
  Complex.abs (-5 + t * Complex.I) = 3 * Real.sqrt 13 → t = 2 * Real.sqrt 23 := by
  sorry

end NUMINAMATH_CALUDE_complex_absolute_value_l1583_158304


namespace NUMINAMATH_CALUDE_diagonals_25_sided_polygon_l1583_158385

/-- The number of diagonals in a convex polygon with n sides -/
def numDiagonals (n : ℕ) : ℕ := n * (n - 3) / 2

/-- Theorem: The number of diagonals in a convex polygon with 25 sides is 275 -/
theorem diagonals_25_sided_polygon :
  numDiagonals 25 = 275 := by
  sorry

end NUMINAMATH_CALUDE_diagonals_25_sided_polygon_l1583_158385


namespace NUMINAMATH_CALUDE_average_difference_l1583_158334

theorem average_difference (x : ℝ) : (10 + x + 50) / 3 = (20 + 40 + 6) / 3 + 8 ↔ x = 30 := by
  sorry

end NUMINAMATH_CALUDE_average_difference_l1583_158334


namespace NUMINAMATH_CALUDE_seahawks_score_is_37_l1583_158302

/-- The final score of the Seattle Seahawks in the football game -/
def seahawks_final_score : ℕ :=
  let touchdowns : ℕ := 4
  let field_goals : ℕ := 3
  let touchdown_points : ℕ := 7
  let field_goal_points : ℕ := 3
  touchdowns * touchdown_points + field_goals * field_goal_points

/-- Theorem stating that the Seattle Seahawks' final score is 37 points -/
theorem seahawks_score_is_37 : seahawks_final_score = 37 := by
  sorry

end NUMINAMATH_CALUDE_seahawks_score_is_37_l1583_158302


namespace NUMINAMATH_CALUDE_number_of_divisors_32_l1583_158313

theorem number_of_divisors_32 : Finset.card (Nat.divisors 32) = 6 := by
  sorry

end NUMINAMATH_CALUDE_number_of_divisors_32_l1583_158313


namespace NUMINAMATH_CALUDE_sum_sqrt_inequality_l1583_158399

theorem sum_sqrt_inequality (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (habc : a * b * c = 1) :
  Real.sqrt (a / (a + 8)) + Real.sqrt (b / (b + 8)) + Real.sqrt (c / (c + 8)) ≥ 1 := by
  sorry

end NUMINAMATH_CALUDE_sum_sqrt_inequality_l1583_158399


namespace NUMINAMATH_CALUDE_tangent_slope_at_one_two_l1583_158361

/-- The slope of the tangent line to y = 2x^2 at (1, 2) is 4 -/
theorem tangent_slope_at_one_two : 
  let f : ℝ → ℝ := fun x ↦ 2 * x^2
  let x₀ : ℝ := 1
  let y₀ : ℝ := f x₀
  (deriv f) x₀ = 4 := by sorry

end NUMINAMATH_CALUDE_tangent_slope_at_one_two_l1583_158361


namespace NUMINAMATH_CALUDE_no_natural_squares_l1583_158369

theorem no_natural_squares (x y : ℕ) : ¬(∃ a b : ℕ, x^2 + y = a^2 ∧ x - y = b^2) := by
  sorry

end NUMINAMATH_CALUDE_no_natural_squares_l1583_158369


namespace NUMINAMATH_CALUDE_female_leader_probability_l1583_158338

theorem female_leader_probability (female_count male_count : ℕ) 
  (h1 : female_count = 4) 
  (h2 : male_count = 6) : 
  (female_count : ℚ) / (female_count + male_count) = 2 / 5 := by
  sorry

end NUMINAMATH_CALUDE_female_leader_probability_l1583_158338


namespace NUMINAMATH_CALUDE_cooking_and_yoga_count_l1583_158392

/-- Represents the number of people in different curriculum groups -/
structure CurriculumGroups where
  yoga : ℕ
  cooking : ℕ
  weaving : ℕ
  cookingOnly : ℕ
  allCurriculums : ℕ
  cookingAndWeaving : ℕ

/-- Theorem stating the number of people who study both cooking and yoga -/
theorem cooking_and_yoga_count (g : CurriculumGroups) 
  (h1 : g.yoga = 25)
  (h2 : g.cooking = 15)
  (h3 : g.weaving = 8)
  (h4 : g.cookingOnly = 2)
  (h5 : g.allCurriculums = 3)
  (h6 : g.cookingAndWeaving = 3) :
  g.cooking - g.cookingOnly - g.cookingAndWeaving + g.allCurriculums = 10 := by
  sorry


end NUMINAMATH_CALUDE_cooking_and_yoga_count_l1583_158392


namespace NUMINAMATH_CALUDE_planes_parallel_implies_lines_parallel_l1583_158383

-- Define the types for planes and lines
variable (Plane Line : Type)

-- Define the intersection operation
variable (intersect : Plane → Plane → Line)

-- Define the parallel relation for planes and lines
variable (parallel_planes : Plane → Plane → Prop)
variable (parallel_lines : Line → Line → Prop)

-- State the theorem
theorem planes_parallel_implies_lines_parallel
  (α β γ : Plane) (m n : Line)
  (h1 : α ≠ β ∧ α ≠ γ ∧ β ≠ γ)
  (h2 : intersect α γ = m)
  (h3 : intersect β γ = n)
  (h4 : parallel_planes α β) :
  parallel_lines m n :=
sorry

end NUMINAMATH_CALUDE_planes_parallel_implies_lines_parallel_l1583_158383


namespace NUMINAMATH_CALUDE_exactly_one_true_proposition_l1583_158310

-- Define parallel vectors
def parallel (a b : ℝ × ℝ) : Prop := ∃ k : ℝ, a = k • b ∨ b = k • a

theorem exactly_one_true_proposition : ∃! n : Fin 4, match n with
  | 0 => ∀ a b : ℝ, (a * b)^2 = a^2 * b^2
  | 1 => ∀ a b : ℝ, |a + b| > |a - b|
  | 2 => ∀ a b : ℝ, |a + b|^2 = (a + b)^2
  | 3 => ∀ a b : ℝ × ℝ, parallel a b → a.1 * b.1 + a.2 * b.2 = Real.sqrt (a.1^2 + a.2^2) * Real.sqrt (b.1^2 + b.2^2)
  := by sorry

end NUMINAMATH_CALUDE_exactly_one_true_proposition_l1583_158310


namespace NUMINAMATH_CALUDE_five_Y_three_equals_four_l1583_158342

-- Define the Y operation
def Y (a b : ℝ) : ℝ := a^2 - 2*a*b + b^2

-- Theorem statement
theorem five_Y_three_equals_four : Y 5 3 = 4 := by
  sorry

end NUMINAMATH_CALUDE_five_Y_three_equals_four_l1583_158342


namespace NUMINAMATH_CALUDE_carson_seed_fertilizer_problem_l1583_158379

/-- The problem of calculating the total amount of seed and fertilizer used by Carson. -/
theorem carson_seed_fertilizer_problem :
  ∀ (seed fertilizer : ℝ),
  seed = 45 →
  seed = 3 * fertilizer →
  seed + fertilizer = 60 :=
by
  sorry

end NUMINAMATH_CALUDE_carson_seed_fertilizer_problem_l1583_158379


namespace NUMINAMATH_CALUDE_woman_age_difference_l1583_158324

/-- The age of the son -/
def son_age : ℕ := 27

/-- The age of the woman -/
def woman_age : ℕ := 84 - son_age

/-- The difference between the woman's age and twice her son's age -/
def age_difference : ℕ := woman_age - 2 * son_age

theorem woman_age_difference : age_difference = 3 := by
  sorry

end NUMINAMATH_CALUDE_woman_age_difference_l1583_158324


namespace NUMINAMATH_CALUDE_sphere_volume_from_inscribed_cube_l1583_158301

theorem sphere_volume_from_inscribed_cube (s : Real) (r : Real) : 
  (6 * s^2 = 32) →  -- surface area of cube is 32
  (r = s * Real.sqrt 3 / 2) →  -- radius of sphere in terms of cube side length
  (4 / 3 * Real.pi * r^3 = 32 * Real.pi / 3) :=  -- volume of sphere
by sorry

end NUMINAMATH_CALUDE_sphere_volume_from_inscribed_cube_l1583_158301


namespace NUMINAMATH_CALUDE_max_sum_given_constraints_l1583_158354

theorem max_sum_given_constraints (a b : ℝ) 
  (h1 : a^2 + b^2 = 130) 
  (h2 : a * b = 45) : 
  a + b ≤ 2 * Real.sqrt 55 := by
sorry

end NUMINAMATH_CALUDE_max_sum_given_constraints_l1583_158354


namespace NUMINAMATH_CALUDE_jemma_grasshoppers_l1583_158395

/-- The number of grasshoppers Jemma saw on her African daisy plant -/
def grasshoppers_on_plant : ℕ := 7

/-- The number of dozens of baby grasshoppers Jemma found under the plant -/
def dozens_of_baby_grasshoppers : ℕ := 2

/-- The number of grasshoppers in a dozen -/
def grasshoppers_per_dozen : ℕ := 12

/-- The total number of grasshoppers Jemma found -/
def total_grasshoppers : ℕ := grasshoppers_on_plant + dozens_of_baby_grasshoppers * grasshoppers_per_dozen

theorem jemma_grasshoppers : total_grasshoppers = 31 := by
  sorry

end NUMINAMATH_CALUDE_jemma_grasshoppers_l1583_158395


namespace NUMINAMATH_CALUDE_parabola_directrix_l1583_158364

/-- The parabola equation -/
def parabola_equation (x y : ℝ) : Prop :=
  x = -1/4 * y^2 + 1

/-- The directrix equation -/
def directrix_equation (x : ℝ) : Prop :=
  x = 2

/-- Theorem stating that the directrix of the given parabola is x = 2 -/
theorem parabola_directrix :
  ∀ (x y : ℝ), parabola_equation x y → ∃ (d : ℝ), directrix_equation d :=
sorry

end NUMINAMATH_CALUDE_parabola_directrix_l1583_158364


namespace NUMINAMATH_CALUDE_nicky_running_time_l1583_158356

-- Define the race parameters
def race_distance : ℝ := 400
def head_start : ℝ := 12
def cristina_speed : ℝ := 5
def nicky_speed : ℝ := 3

-- Define the theorem
theorem nicky_running_time (t : ℝ) : 
  t * cristina_speed = head_start * nicky_speed + (t + head_start) * nicky_speed → 
  t + head_start = 48 :=
by sorry

end NUMINAMATH_CALUDE_nicky_running_time_l1583_158356


namespace NUMINAMATH_CALUDE_quadratic_polynomial_theorem_l1583_158358

/-- A point in the plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- A quadratic polynomial -/
structure QuadraticPolynomial where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Check if a point lies on the graph of a quadratic polynomial -/
def pointLiesOnPolynomial (p : Point) (q : QuadraticPolynomial) : Prop :=
  p.y = q.a * p.x^2 + q.b * p.x + q.c

/-- The main theorem -/
theorem quadratic_polynomial_theorem 
  (points : Finset Point) 
  (h_count : points.card = 100)
  (h_four_points : ∀ (p₁ p₂ p₃ p₄ : Point), p₁ ∈ points → p₂ ∈ points → p₃ ∈ points → p₄ ∈ points →
    p₁ ≠ p₂ → p₁ ≠ p₃ → p₁ ≠ p₄ → p₂ ≠ p₃ → p₂ ≠ p₄ → p₃ ≠ p₄ →
    ∃ (q : QuadraticPolynomial), pointLiesOnPolynomial p₁ q ∧ pointLiesOnPolynomial p₂ q ∧
      pointLiesOnPolynomial p₃ q ∧ pointLiesOnPolynomial p₄ q) :
  ∃ (q : QuadraticPolynomial), ∀ (p : Point), p ∈ points → pointLiesOnPolynomial p q :=
by sorry

end NUMINAMATH_CALUDE_quadratic_polynomial_theorem_l1583_158358


namespace NUMINAMATH_CALUDE_digit_subtraction_result_l1583_158309

def three_digit_number (a b c : ℕ) : Prop :=
  a ≥ 1 ∧ a ≤ 9 ∧ b ≥ 0 ∧ b ≤ 9 ∧ c ≥ 0 ∧ c ≤ 9

theorem digit_subtraction_result (a b c : ℕ) 
  (h1 : three_digit_number a b c) 
  (h2 : a = c + 2) : 
  (100 * a + 10 * b + c) - (100 * c + 10 * b + a) ≡ 8 [ZMOD 10] := by
  sorry

end NUMINAMATH_CALUDE_digit_subtraction_result_l1583_158309


namespace NUMINAMATH_CALUDE_downstream_distance_is_16_l1583_158337

/-- Represents the swimming scenario with given conditions -/
structure SwimmingScenario where
  upstream_distance : ℝ
  swim_time : ℝ
  still_water_speed : ℝ

/-- Calculates the downstream distance given a swimming scenario -/
def downstream_distance (s : SwimmingScenario) : ℝ :=
  sorry

/-- Theorem stating that under the given conditions, the downstream distance is 16 km -/
theorem downstream_distance_is_16 (s : SwimmingScenario) 
  (h1 : s.upstream_distance = 10)
  (h2 : s.swim_time = 2)
  (h3 : s.still_water_speed = 6.5) :
  downstream_distance s = 16 := by
  sorry

end NUMINAMATH_CALUDE_downstream_distance_is_16_l1583_158337


namespace NUMINAMATH_CALUDE_right_triangle_solution_l1583_158329

-- Define the right triangle
def RightTriangle (a b c h : ℝ) : Prop :=
  a > 0 ∧ b > 0 ∧ c > 0 ∧ h > 0 ∧ a^2 + b^2 = c^2 ∧ h^2 = (a^2 * b^2) / c^2

-- Define the conditions
def TriangleConditions (a b c h e d : ℝ) : Prop :=
  RightTriangle a b c h ∧ 
  (c^2 / (2*h) - h/2 = e) ∧  -- Difference between hypotenuse segments
  (a - b = d)                -- Difference between legs

-- Theorem statement
theorem right_triangle_solution (e d : ℝ) (he : e = 37.0488) (hd : d = 31) :
  ∃ (a b c h : ℝ), TriangleConditions a b c h e d ∧ 
    (a = 40 ∧ b = 9 ∧ c = 41) := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_solution_l1583_158329


namespace NUMINAMATH_CALUDE_equal_charges_at_300_minutes_l1583_158373

/-- Represents a mobile phone plan -/
structure PhonePlan where
  monthly_fee : ℝ
  call_rate : ℝ

/-- Calculates the monthly bill for a given plan and call duration -/
def monthly_bill (plan : PhonePlan) (duration : ℝ) : ℝ :=
  plan.monthly_fee + plan.call_rate * duration

/-- The Unicom company's phone plans -/
def plan_a : PhonePlan := { monthly_fee := 15, call_rate := 0.1 }
def plan_b : PhonePlan := { monthly_fee := 0, call_rate := 0.15 }

theorem equal_charges_at_300_minutes : 
  ∃ (duration : ℝ), duration = 300 ∧ 
    monthly_bill plan_a duration = monthly_bill plan_b duration := by
  sorry

end NUMINAMATH_CALUDE_equal_charges_at_300_minutes_l1583_158373


namespace NUMINAMATH_CALUDE_solution_set_inequality_l1583_158336

/-- Given that the solution set of x^2 + ax + b > 0 is (-∞, -2) ∪ (-1/2, +∞),
    prove that the solution set of bx^2 + ax + 1 < 0 is (-2, -1/2) -/
theorem solution_set_inequality (a b : ℝ) : 
  (∀ x, x^2 + a*x + b > 0 ↔ x < -2 ∨ x > -1/2) →
  (∀ x, b*x^2 + a*x + 1 < 0 ↔ -2 < x ∧ x < -1/2) :=
by sorry

end NUMINAMATH_CALUDE_solution_set_inequality_l1583_158336


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_l1583_158396

theorem quadratic_inequality_solution (a b : ℝ) :
  (∀ x : ℝ, ax^2 - (a+1)*x + b < 0 ↔ 1 < x ∧ x < 5) →
  a + b = 6/5 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_l1583_158396


namespace NUMINAMATH_CALUDE_cubic_function_coefficients_l1583_158380

/-- Given a cubic function f(x) = ax³ - bx + 4, prove that if f(2) = -4/3 and f'(2) = 0,
    then a = 1/3 and b = 4 -/
theorem cubic_function_coefficients (a b : ℝ) :
  let f : ℝ → ℝ := λ x ↦ a * x^3 - b * x + 4
  let f' : ℝ → ℝ := λ x ↦ 3 * a * x^2 - b
  f 2 = -4/3 ∧ f' 2 = 0 → a = 1/3 ∧ b = 4 := by
  sorry

end NUMINAMATH_CALUDE_cubic_function_coefficients_l1583_158380


namespace NUMINAMATH_CALUDE_perfect_square_base_9_l1583_158370

/-- Represents a number in base 9 of the form ab5c where a ≠ 0 -/
structure Base9Number where
  a : ℕ
  b : ℕ
  c : ℕ
  a_nonzero : a ≠ 0
  b_less_than_9 : b < 9
  c_less_than_9 : c < 9

/-- Converts a Base9Number to its decimal representation -/
def toDecimal (n : Base9Number) : ℕ :=
  729 * n.a + 81 * n.b + 45 + n.c

theorem perfect_square_base_9 (n : Base9Number) :
  ∃ (k : ℕ), toDecimal n = k^2 → n.c = 0 ∨ n.c = 7 := by
  sorry

end NUMINAMATH_CALUDE_perfect_square_base_9_l1583_158370


namespace NUMINAMATH_CALUDE_cube_root_of_negative_27_l1583_158311

theorem cube_root_of_negative_27 :
  let S : Set ℂ := {z : ℂ | z^3 = -27}
  S = {-3, (3/2 : ℂ) + (3*Complex.I*Real.sqrt 3)/2, (3/2 : ℂ) - (3*Complex.I*Real.sqrt 3)/2} := by
  sorry

end NUMINAMATH_CALUDE_cube_root_of_negative_27_l1583_158311


namespace NUMINAMATH_CALUDE_sum_of_squares_theorem_l1583_158353

theorem sum_of_squares_theorem (p q r : ℝ) (h_distinct : p ≠ q ∧ q ≠ r ∧ r ≠ p) 
  (h_sum : p / (q - r) + q / (r - p) + r / (p - q) = 3) :
  p^2 / (q - r)^2 + q^2 / (r - p)^2 + r^2 / (p - q)^2 = 3 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_squares_theorem_l1583_158353


namespace NUMINAMATH_CALUDE_opposite_numbers_sum_l1583_158347

/-- If a and b are opposite numbers, then a + b + 3 = 3 -/
theorem opposite_numbers_sum (a b : ℝ) : a + b = 0 → a + b + 3 = 3 := by
  sorry

end NUMINAMATH_CALUDE_opposite_numbers_sum_l1583_158347


namespace NUMINAMATH_CALUDE_log_expression_equality_l1583_158339

theorem log_expression_equality : 2 * Real.log 10 / Real.log 5 + Real.log 0.25 / Real.log 5 + 8^(1/4) * 2^(1/4) = 4 := by
  sorry

end NUMINAMATH_CALUDE_log_expression_equality_l1583_158339


namespace NUMINAMATH_CALUDE_M_is_hypersquared_l1583_158320

def n : ℕ := 1000

def is_perfect_square (x : ℕ) : Prop := ∃ y : ℕ, x = y^2

def first_n_digits (x : ℕ) (n : ℕ) : ℕ := x / 10^n

def last_n_digits (x : ℕ) (n : ℕ) : ℕ := x % 10^n

def is_hypersquared (x : ℕ) : Prop :=
  ∃ n : ℕ,
    (x ≥ 10^(2*n - 1)) ∧
    (x < 10^(2*n)) ∧
    is_perfect_square x ∧
    is_perfect_square (first_n_digits x n) ∧
    is_perfect_square (last_n_digits x n) ∧
    (last_n_digits x n ≥ 10^(n-1))

def M : ℕ := ((5 * 10^(n-1) - 1) * 10^n + (10^n - 1))^2

theorem M_is_hypersquared : is_hypersquared M := by
  sorry

end NUMINAMATH_CALUDE_M_is_hypersquared_l1583_158320


namespace NUMINAMATH_CALUDE_max_value_xyz_l1583_158372

theorem max_value_xyz (x y z : ℝ) 
  (sum_zero : x + y + z = 0) 
  (sum_squares_six : x^2 + y^2 + z^2 = 6) : 
  x^2*y + y^2*z + z^2*x ≤ 6 := by
  sorry

end NUMINAMATH_CALUDE_max_value_xyz_l1583_158372


namespace NUMINAMATH_CALUDE_cake_mix_buyers_l1583_158341

/-- Proof of the number of buyers purchasing cake mix -/
theorem cake_mix_buyers (total : ℕ) (muffin : ℕ) (both : ℕ) (neither_prob : ℚ) 
  (h1 : total = 100)
  (h2 : muffin = 40)
  (h3 : both = 17)
  (h4 : neither_prob = 27 / 100) : 
  ∃ cake : ℕ, cake = 50 ∧ 
    cake + muffin - both = total - (neither_prob * total).num := by
  sorry

end NUMINAMATH_CALUDE_cake_mix_buyers_l1583_158341


namespace NUMINAMATH_CALUDE_smallest_dual_base_representation_l1583_158344

theorem smallest_dual_base_representation :
  ∃ (n : ℕ) (a b : ℕ),
    a > 2 ∧ b > 2 ∧
    n = 2 * a + 1 ∧
    n = 1 * b + 2 ∧
    (∀ (m : ℕ) (c d : ℕ),
      c > 2 → d > 2 →
      m = 2 * c + 1 →
      m = 1 * d + 2 →
      n ≤ m) ∧
    n = 7 :=
by sorry

end NUMINAMATH_CALUDE_smallest_dual_base_representation_l1583_158344


namespace NUMINAMATH_CALUDE_max_quadrilateral_area_l1583_158368

/-- The ellipse equation -/
def ellipse (x y : ℝ) : Prop := x^2/4 + y^2 = 1

/-- Points on the x-axis that the ellipse passes through -/
def P : ℝ × ℝ := (1, 0)
def Q : ℝ × ℝ := (-1, 0)

/-- A function representing parallel lines passing through a point with slope k -/
def parallelLine (p : ℝ × ℝ) (k : ℝ) (x y : ℝ) : Prop :=
  y = k * (x - p.1) + p.2

/-- The quadrilateral formed by the intersection of parallel lines and the ellipse -/
def quadrilateral (k : ℝ) : Set (ℝ × ℝ) :=
  {p | ∃ x y, ellipse x y ∧ (parallelLine P k x y ∨ parallelLine Q k x y)}

/-- The area of the quadrilateral as a function of the slope k -/
noncomputable def quadrilateralArea (k : ℝ) : ℝ :=
  sorry  -- Actual computation of area

theorem max_quadrilateral_area :
  ∃ (max_area : ℝ), max_area = 2 * Real.sqrt 3 ∧
    ∀ k, quadrilateralArea k ≤ max_area :=
  sorry

#check max_quadrilateral_area

end NUMINAMATH_CALUDE_max_quadrilateral_area_l1583_158368


namespace NUMINAMATH_CALUDE_bankers_discount_problem_l1583_158317

theorem bankers_discount_problem (bankers_discount sum_due : ℚ) : 
  bankers_discount = 80 → sum_due = 560 → 
  (bankers_discount / (1 + bankers_discount / sum_due)) = 70 := by
sorry

end NUMINAMATH_CALUDE_bankers_discount_problem_l1583_158317


namespace NUMINAMATH_CALUDE_lentil_dishes_count_l1583_158352

/-- Represents the menu of a vegan restaurant -/
structure VeganMenu :=
  (total_dishes : ℕ)
  (beans_and_lentils : ℕ)
  (beans_and_seitan : ℕ)
  (only_beans : ℕ)
  (only_seitan : ℕ)
  (only_lentils : ℕ)

/-- The number of dishes including lentils in a vegan menu -/
def dishes_with_lentils (menu : VeganMenu) : ℕ :=
  menu.beans_and_lentils + menu.only_lentils

/-- Theorem stating the number of dishes including lentils in the given vegan menu -/
theorem lentil_dishes_count (menu : VeganMenu) 
  (h1 : menu.total_dishes = 10)
  (h2 : menu.beans_and_lentils = 2)
  (h3 : menu.beans_and_seitan = 2)
  (h4 : menu.only_beans = (menu.total_dishes - menu.beans_and_lentils - menu.beans_and_seitan) / 2)
  (h5 : menu.only_beans = 3 * menu.only_seitan)
  (h6 : menu.only_lentils = menu.total_dishes - menu.beans_and_lentils - menu.beans_and_seitan - menu.only_beans - menu.only_seitan) :
  dishes_with_lentils menu = 4 := by
  sorry


end NUMINAMATH_CALUDE_lentil_dishes_count_l1583_158352


namespace NUMINAMATH_CALUDE_chocolate_eggs_weight_l1583_158348

theorem chocolate_eggs_weight (total_eggs : ℕ) (egg_weight : ℕ) (num_boxes : ℕ) (discarded_boxes : ℕ) : 
  total_eggs = 12 →
  egg_weight = 10 →
  num_boxes = 4 →
  discarded_boxes = 1 →
  (total_eggs / num_boxes) * egg_weight * (num_boxes - discarded_boxes) = 90 := by
sorry

end NUMINAMATH_CALUDE_chocolate_eggs_weight_l1583_158348


namespace NUMINAMATH_CALUDE_power_value_from_condition_l1583_158382

theorem power_value_from_condition (x y : ℝ) : 
  |x - 3| + (y + 3)^2 = 0 → y^x = -27 := by sorry

end NUMINAMATH_CALUDE_power_value_from_condition_l1583_158382


namespace NUMINAMATH_CALUDE_library_avg_megabytes_per_hour_l1583_158398

/-- Calculates the average megabytes per hour of music in a digital library, rounded to the nearest whole number -/
def avgMegabytesPerHour (days : ℕ) (totalMB : ℕ) : ℕ :=
  let hoursPerDay : ℕ := 24
  let totalHours : ℕ := days * hoursPerDay
  let exactAvg : ℚ := totalMB / totalHours
  (exactAvg + 1/2).floor.toNat

/-- Theorem stating that for a 15-day library with 20,000 MB, the average is 56 MB/hour -/
theorem library_avg_megabytes_per_hour :
  avgMegabytesPerHour 15 20000 = 56 := by
  sorry

end NUMINAMATH_CALUDE_library_avg_megabytes_per_hour_l1583_158398


namespace NUMINAMATH_CALUDE_circle_equation_valid_a_l1583_158377

def circle_equation (x y a : ℝ) : Prop :=
  x^2 + y^2 + 2*a*x + 2*a*y + 2*a^2 + a - 1 = 0

def is_valid_a (a : ℝ) : Prop :=
  a < 1

def given_options : List ℝ := [-1, 1, 0, 3]

theorem circle_equation_valid_a :
  ∀ a ∈ given_options, circle_equation x y a → is_valid_a a ↔ (a = -1 ∨ a = 0) :=
sorry

end NUMINAMATH_CALUDE_circle_equation_valid_a_l1583_158377
