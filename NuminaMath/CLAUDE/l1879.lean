import Mathlib

namespace NUMINAMATH_CALUDE_copy_pages_proof_l1879_187915

/-- The cost in cents to copy a single page -/
def cost_per_page : ℚ := 25/10

/-- The amount of money available in dollars -/
def available_money : ℚ := 20

/-- The number of cents in a dollar -/
def cents_per_dollar : ℕ := 100

/-- The number of pages that can be copied with the available money -/
def pages_copied : ℕ := 800

theorem copy_pages_proof : 
  (available_money * cents_per_dollar) / cost_per_page = pages_copied := by
  sorry

end NUMINAMATH_CALUDE_copy_pages_proof_l1879_187915


namespace NUMINAMATH_CALUDE_parabola_angle_theorem_l1879_187985

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

end NUMINAMATH_CALUDE_parabola_angle_theorem_l1879_187985


namespace NUMINAMATH_CALUDE_adas_original_seat_l1879_187926

/-- Represents the seats in the movie theater. -/
inductive Seat
| one
| two
| three
| four
| five
| six

/-- Represents the friends. -/
inductive Friend
| Ada
| Bea
| Ceci
| Dee
| Edie
| Fran

/-- Represents a seating arrangement. -/
def Arrangement := Friend → Seat

/-- Defines what it means for a seat to be an end seat. -/
def is_end_seat (s : Seat) : Prop :=
  s = Seat.one ∨ s = Seat.six

/-- Defines the movement of friends after Ada left. -/
def moved (initial final : Arrangement) : Prop :=
  (∃ s, initial Friend.Bea = s ∧ final Friend.Bea = Seat.six) ∧
  (initial Friend.Ceci = final Friend.Ceci) ∧
  (initial Friend.Dee = final Friend.Edie ∧ initial Friend.Edie = final Friend.Dee) ∧
  (∃ s t, initial Friend.Fran = s ∧ final Friend.Fran = t ∧ 
    (s = Seat.two ∧ t = Seat.one ∨
     s = Seat.three ∧ t = Seat.two ∨
     s = Seat.four ∧ t = Seat.three ∨
     s = Seat.five ∧ t = Seat.four ∨
     s = Seat.six ∧ t = Seat.five))

/-- The main theorem stating Ada's original seat. -/
theorem adas_original_seat (initial final : Arrangement) :
  moved initial final →
  is_end_seat (final Friend.Ada) →
  initial Friend.Ada = Seat.three :=
sorry

end NUMINAMATH_CALUDE_adas_original_seat_l1879_187926


namespace NUMINAMATH_CALUDE_function_composition_equality_l1879_187955

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x < 1 then 2^x + 1 else x^2 + a*x

theorem function_composition_equality (a : ℝ) :
  f a (f a 0) = 4*a → a = 2 := by
  sorry

end NUMINAMATH_CALUDE_function_composition_equality_l1879_187955


namespace NUMINAMATH_CALUDE_sum_of_distance_and_reciprocal_l1879_187931

theorem sum_of_distance_and_reciprocal (a b : ℝ) : 
  (|a| = 5 ∧ b = -(1 / (-1/3))) → (a + b = 2 ∨ a + b = -8) :=
by sorry

end NUMINAMATH_CALUDE_sum_of_distance_and_reciprocal_l1879_187931


namespace NUMINAMATH_CALUDE_stolen_newspapers_weight_l1879_187925

/-- Calculates the total weight of stolen newspapers in tons over a given number of weeks. -/
def total_weight_in_tons (weeks : ℕ) : ℚ :=
  let daily_papers : ℕ := 250
  let weekday_paper_weight : ℚ := 8 / 16
  let sunday_paper_weight : ℚ := 2 * weekday_paper_weight
  let weekdays_per_week : ℕ := 6
  let sundays_per_week : ℕ := 1
  let weekday_weight : ℚ := (weeks * weekdays_per_week * daily_papers * weekday_paper_weight)
  let sunday_weight : ℚ := (weeks * sundays_per_week * daily_papers * sunday_paper_weight)
  (weekday_weight + sunday_weight) / 2000

/-- Theorem stating that the total weight of stolen newspapers over 10 weeks is 5 tons. -/
theorem stolen_newspapers_weight : total_weight_in_tons 10 = 5 := by
  sorry

end NUMINAMATH_CALUDE_stolen_newspapers_weight_l1879_187925


namespace NUMINAMATH_CALUDE_quadratic_function_properties_l1879_187959

def f (a x : ℝ) := -x^2 + 2*a*x - 3

theorem quadratic_function_properties :
  (∃ x : ℝ, -3 < x ∧ x < -1 ∧ f (-2) x < 0) ∧
  (∀ x : ℝ, x ∈ Set.Icc 1 5 → ∀ a : ℝ, a > -2 * Real.sqrt 3 → f a x < 3 * a * x) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_function_properties_l1879_187959


namespace NUMINAMATH_CALUDE_B_set_given_A_l1879_187964

def f (a b x : ℝ) : ℝ := x^2 + a*x + b

def A (a b : ℝ) : Set ℝ := {x | f a b x = x}

def B (a b : ℝ) : Set ℝ := {x | f a b (f a b x) = x}

theorem B_set_given_A (a b : ℝ) :
  A a b = {-1, 3} → B a b = {-1, Real.sqrt 3, -Real.sqrt 3, 3} := by
  sorry

end NUMINAMATH_CALUDE_B_set_given_A_l1879_187964


namespace NUMINAMATH_CALUDE_right_triangle_condition_l1879_187938

/-- If in a triangle ABC, angle A equals the sum of angles B and C, then angle A is a right angle -/
theorem right_triangle_condition (A B C : Real) (h1 : A + B + C = Real.pi) (h2 : A = B + C) : A = Real.pi / 2 := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_condition_l1879_187938


namespace NUMINAMATH_CALUDE_equation_root_implies_a_value_l1879_187934

theorem equation_root_implies_a_value (x a : ℝ) : 
  ((x - 2) / (x + 4) = a / (x + 4)) → (∃ x, (x - 2) / (x + 4) = a / (x + 4)) → a = -6 :=
by sorry

end NUMINAMATH_CALUDE_equation_root_implies_a_value_l1879_187934


namespace NUMINAMATH_CALUDE_sqrt_sum_problem_l1879_187996

theorem sqrt_sum_problem (y : ℝ) (h : Real.sqrt (64 - y^2) - Real.sqrt (36 - y^2) = 4) :
  Real.sqrt (64 - y^2) + Real.sqrt (36 - y^2) = 7 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_sum_problem_l1879_187996


namespace NUMINAMATH_CALUDE_triangle_side_length_l1879_187972

/-- Given a triangle ABC with side lengths a = 2, b = 1, and angle C = 60°, 
    the length of side c is √3. -/
theorem triangle_side_length (a b c : ℝ) (C : ℝ) : 
  a = 2 → b = 1 → C = Real.pi / 3 → c = Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_triangle_side_length_l1879_187972


namespace NUMINAMATH_CALUDE_production_average_l1879_187930

/-- Proves that given the conditions in the problem, n = 1 --/
theorem production_average (n : ℕ) : 
  (n * 50 + 60) / (n + 1) = 55 → n = 1 := by
  sorry


end NUMINAMATH_CALUDE_production_average_l1879_187930


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l1879_187939

theorem arithmetic_sequence_sum (a₁ d : ℕ) (n : ℕ) :
  a₁ = 3 → d = 3 → n = 10 →
  (n : ℝ) / 2 * (a₁ + (a₁ + (n - 1) * d)) = 165 := by sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l1879_187939


namespace NUMINAMATH_CALUDE_intersection_point_l1879_187900

def line1 (x y : ℚ) : Prop := y = 3 * x + 1
def line2 (x y : ℚ) : Prop := y + 1 = -7 * x

theorem intersection_point :
  ∃! p : ℚ × ℚ, line1 p.1 p.2 ∧ line2 p.1 p.2 ∧ p = (-1/5, 2/5) := by
  sorry

end NUMINAMATH_CALUDE_intersection_point_l1879_187900


namespace NUMINAMATH_CALUDE_thickness_after_13_folds_l1879_187992

/-- The thickness of a paper after n folds, given an initial thickness of a millimeters -/
def paper_thickness (a : ℝ) (n : ℕ) : ℝ :=
  a * 2^n

/-- Theorem: The thickness of a paper after 13 folds is 2^13 times its initial thickness -/
theorem thickness_after_13_folds (a : ℝ) :
  paper_thickness a 13 = a * 2^13 := by
  sorry

#check thickness_after_13_folds

end NUMINAMATH_CALUDE_thickness_after_13_folds_l1879_187992


namespace NUMINAMATH_CALUDE_solution_set_quadratic_inequality_l1879_187929

theorem solution_set_quadratic_inequality :
  ∀ x : ℝ, x * (x - 2) ≤ 0 ↔ 0 ≤ x ∧ x ≤ 2 := by sorry

end NUMINAMATH_CALUDE_solution_set_quadratic_inequality_l1879_187929


namespace NUMINAMATH_CALUDE_max_square_plots_l1879_187969

/-- Represents the dimensions of the park and available fencing --/
structure ParkData where
  width : ℕ
  length : ℕ
  fencing : ℕ

/-- Represents a potential partitioning of the park --/
structure Partitioning where
  sideLength : ℕ
  numPlots : ℕ

/-- Checks if a partitioning is valid for the given park data --/
def isValidPartitioning (park : ParkData) (part : Partitioning) : Prop :=
  part.sideLength > 0 ∧
  park.width % part.sideLength = 0 ∧
  park.length % part.sideLength = 0 ∧
  part.numPlots = (park.width / part.sideLength) * (park.length / part.sideLength) ∧
  (park.width / part.sideLength - 1) * park.length + (park.length / part.sideLength - 1) * park.width ≤ park.fencing

/-- Theorem stating that the maximum number of square plots is 2 --/
theorem max_square_plots (park : ParkData) 
  (h_width : park.width = 30)
  (h_length : park.length = 60)
  (h_fencing : park.fencing = 2400) :
  (∀ p : Partitioning, isValidPartitioning park p → p.numPlots ≤ 2) ∧
  (∃ p : Partitioning, isValidPartitioning park p ∧ p.numPlots = 2) :=
sorry

end NUMINAMATH_CALUDE_max_square_plots_l1879_187969


namespace NUMINAMATH_CALUDE_complex_fraction_simplification_l1879_187924

theorem complex_fraction_simplification :
  (7 + 18 * Complex.I) / (4 - 5 * Complex.I) = -62/41 + (107/41) * Complex.I :=
by sorry

end NUMINAMATH_CALUDE_complex_fraction_simplification_l1879_187924


namespace NUMINAMATH_CALUDE_stephanie_distance_l1879_187912

/-- Calculates the distance traveled given time and speed -/
def distance (time : ℝ) (speed : ℝ) : ℝ := time * speed

/-- Proves that running for 3 hours at 5 miles per hour results in a distance of 15 miles -/
theorem stephanie_distance :
  let time : ℝ := 3
  let speed : ℝ := 5
  distance time speed = 15 := by
  sorry

end NUMINAMATH_CALUDE_stephanie_distance_l1879_187912


namespace NUMINAMATH_CALUDE_greatest_divisor_four_consecutive_integers_l1879_187943

theorem greatest_divisor_four_consecutive_integers :
  ∃ (d : ℕ), d = 12 ∧ 
  (∀ (n : ℕ), n > 0 → d ∣ (n * (n + 1) * (n + 2) * (n + 3))) ∧
  (∀ (k : ℕ), k > d → ∃ (m : ℕ), m > 0 ∧ ¬(k ∣ (m * (m + 1) * (m + 2) * (m + 3)))) :=
by sorry

end NUMINAMATH_CALUDE_greatest_divisor_four_consecutive_integers_l1879_187943


namespace NUMINAMATH_CALUDE_composite_with_large_smallest_prime_divisor_l1879_187909

theorem composite_with_large_smallest_prime_divisor 
  (N : ℕ) 
  (h_composite : ¬ Prime N) 
  (h_smallest_divisor : ∀ p : ℕ, Prime p → p ∣ N → p > N^(1/3)) : 
  ∃ p q : ℕ, Prime p ∧ Prime q ∧ N = p * q :=
sorry

end NUMINAMATH_CALUDE_composite_with_large_smallest_prime_divisor_l1879_187909


namespace NUMINAMATH_CALUDE_johns_purchase_price_l1879_187970

/-- Calculate the final price after rebate and tax -/
def finalPrice (originalPrice rebatePercent taxPercent : ℚ) : ℚ :=
  let priceAfterRebate := originalPrice * (1 - rebatePercent / 100)
  let salesTax := priceAfterRebate * (taxPercent / 100)
  priceAfterRebate + salesTax

/-- Theorem stating the final price for John's purchase -/
theorem johns_purchase_price :
  finalPrice 6650 6 10 = 6876.1 :=
sorry

end NUMINAMATH_CALUDE_johns_purchase_price_l1879_187970


namespace NUMINAMATH_CALUDE_action_figure_value_l1879_187975

theorem action_figure_value (n : ℕ) (known_value : ℕ) (discount : ℕ) (total_earned : ℕ) :
  n = 5 →
  known_value = 20 →
  discount = 5 →
  total_earned = 55 →
  ∃ (other_value : ℕ),
    other_value * (n - 1) + known_value = total_earned + n * discount ∧
    other_value = 15 := by
  sorry

end NUMINAMATH_CALUDE_action_figure_value_l1879_187975


namespace NUMINAMATH_CALUDE_difference_of_squares_example_l1879_187953

theorem difference_of_squares_example : (15 + 12)^2 - (15 - 12)^2 = 720 := by
  sorry

end NUMINAMATH_CALUDE_difference_of_squares_example_l1879_187953


namespace NUMINAMATH_CALUDE_triangle_xz_interval_l1879_187913

-- Define the triangle XYZ
structure Triangle :=
  (X Y Z : ℝ × ℝ)

-- Define the point W on YZ
def W (t : Triangle) : ℝ × ℝ := sorry

-- Define the length of a line segment
def length (p q : ℝ × ℝ) : ℝ := sorry

-- Define the perimeter of a triangle
def perimeter (t : Triangle) : ℝ := sorry

-- Define the angle bisector
def is_angle_bisector (t : Triangle) (w : ℝ × ℝ) : Prop := sorry

-- State the theorem
theorem triangle_xz_interval (t : Triangle) :
  length t.X t.Y = 8 →
  is_angle_bisector t (W t) →
  length (W t) t.Z = 5 →
  perimeter t = 24 →
  ∃ m n : ℝ, m < n ∧ 
    (∀ xz : ℝ, m < xz ∧ xz < n ↔ length t.X t.Z = xz) ∧
    m + n = 13 := by
  sorry

end NUMINAMATH_CALUDE_triangle_xz_interval_l1879_187913


namespace NUMINAMATH_CALUDE_log_product_problem_l1879_187989

theorem log_product_problem (c d : ℕ+) : 
  (d.val - c.val - 1 = 435) →  -- Number of terms is 435
  (Real.log d.val / Real.log c.val = 3) →  -- Value of the product is 3
  (c.val + d.val = 130) := by
sorry

end NUMINAMATH_CALUDE_log_product_problem_l1879_187989


namespace NUMINAMATH_CALUDE_alberto_bjorn_distance_difference_l1879_187950

/-- Given two cyclists, Alberto and Bjorn, with different speeds, 
    prove the difference in distance traveled after a certain time. -/
theorem alberto_bjorn_distance_difference 
  (alberto_speed : ℝ) 
  (bjorn_speed : ℝ) 
  (time : ℝ) 
  (h1 : alberto_speed = 18) 
  (h2 : bjorn_speed = 17) 
  (h3 : time = 5) : 
  alberto_speed * time - bjorn_speed * time = 5 := by
  sorry

#check alberto_bjorn_distance_difference

end NUMINAMATH_CALUDE_alberto_bjorn_distance_difference_l1879_187950


namespace NUMINAMATH_CALUDE_last_digit_of_one_over_three_to_ten_l1879_187954

theorem last_digit_of_one_over_three_to_ten (n : ℕ) :
  n = 10 →
  ∃ (k : ℕ), (1 : ℚ) / 3^n = k / 10^10 ∧ k % 10 = 3 :=
by sorry

end NUMINAMATH_CALUDE_last_digit_of_one_over_three_to_ten_l1879_187954


namespace NUMINAMATH_CALUDE_p_min_value_l1879_187958

/-- The quadratic function p(x) = x^2 + 6x + 5 -/
def p (x : ℝ) : ℝ := x^2 + 6*x + 5

/-- The minimum value of p(x) is -4 -/
theorem p_min_value : ∀ x : ℝ, p x ≥ -4 := by sorry

end NUMINAMATH_CALUDE_p_min_value_l1879_187958


namespace NUMINAMATH_CALUDE_seventh_term_of_geometric_sequence_l1879_187962

-- Define a geometric sequence
def is_geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, ∀ n : ℕ, a (n + 1) = a n * q

-- State the theorem
theorem seventh_term_of_geometric_sequence
  (a : ℕ → ℝ)
  (h_geometric : is_geometric_sequence a)
  (h_eq : a 2 * a 4 * a 5 = a 3 * a 6)
  (h_prod : a 9 * a 10 = -8) :
  a 7 = -2 := by
  sorry

end NUMINAMATH_CALUDE_seventh_term_of_geometric_sequence_l1879_187962


namespace NUMINAMATH_CALUDE_no_real_roots_composite_l1879_187997

-- Define the quadratic function f
def f (a b c x : ℝ) : ℝ := a * x^2 + b * x + c

-- State the theorem
theorem no_real_roots_composite (a b c : ℝ) :
  (∀ x : ℝ, f a b c x ≠ x) →
  (∀ x : ℝ, f a b c (f a b c x) ≠ x) :=
by sorry

end NUMINAMATH_CALUDE_no_real_roots_composite_l1879_187997


namespace NUMINAMATH_CALUDE_shirt_price_reduction_l1879_187907

/-- Represents the price reduction problem for a shopping mall selling shirts. -/
theorem shirt_price_reduction
  (initial_sales : ℕ)
  (initial_profit : ℝ)
  (price_reduction_effect : ℝ → ℕ)
  (target_profit : ℝ)
  (price_reduction : ℝ)
  (h1 : initial_sales = 20)
  (h2 : initial_profit = 40)
  (h3 : ∀ x, price_reduction_effect x = initial_sales + 2 * ⌊x⌋)
  (h4 : target_profit = 1200)
  (h5 : price_reduction = 20) :
  (initial_profit - price_reduction) * price_reduction_effect price_reduction = target_profit :=
sorry

end NUMINAMATH_CALUDE_shirt_price_reduction_l1879_187907


namespace NUMINAMATH_CALUDE_factorial_squared_gt_power_l1879_187993

theorem factorial_squared_gt_power (n : ℕ) (h : n > 2) : (n.factorial ^ 2 : ℕ) > n ^ n := by
  sorry

end NUMINAMATH_CALUDE_factorial_squared_gt_power_l1879_187993


namespace NUMINAMATH_CALUDE_quadratic_inequality_l1879_187982

theorem quadratic_inequality (x : ℝ) : x^2 - 6*x - 16 > 0 ↔ x < -2 ∨ x > 8 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_l1879_187982


namespace NUMINAMATH_CALUDE_inverse_of_P_l1879_187923

def P (a : ℕ) : Prop := Odd a → Prime a

theorem inverse_of_P : 
  (∀ a : ℕ, P a) ↔ (∀ a : ℕ, Prime a → Odd a) :=
sorry

end NUMINAMATH_CALUDE_inverse_of_P_l1879_187923


namespace NUMINAMATH_CALUDE_sqrt_abs_sum_zero_implies_power_l1879_187940

theorem sqrt_abs_sum_zero_implies_power (a b : ℝ) :
  Real.sqrt (a + 2) + |b - 1| = 0 → (a + b) ^ 2017 = -1 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_abs_sum_zero_implies_power_l1879_187940


namespace NUMINAMATH_CALUDE_sqrt_sum_bounds_l1879_187944

theorem sqrt_sum_bounds : 
  let n : ℝ := Real.sqrt 4 + Real.sqrt 7
  4 < n ∧ n < 5 := by sorry

end NUMINAMATH_CALUDE_sqrt_sum_bounds_l1879_187944


namespace NUMINAMATH_CALUDE_banana_count_l1879_187908

/-- Proves that given 8 boxes and 5 bananas per box, the total number of bananas is 40. -/
theorem banana_count (num_boxes : ℕ) (bananas_per_box : ℕ) (total_bananas : ℕ) : 
  num_boxes = 8 → bananas_per_box = 5 → total_bananas = num_boxes * bananas_per_box → total_bananas = 40 :=
by sorry

end NUMINAMATH_CALUDE_banana_count_l1879_187908


namespace NUMINAMATH_CALUDE_toaster_sales_at_promo_price_l1879_187960

-- Define the inverse proportionality constant
def k : ℝ := 15 * 600

-- Define the original price and number of customers
def original_price : ℝ := 600
def original_customers : ℝ := 15

-- Define the promotional price
def promo_price : ℝ := 450

-- Define the additional sales increase factor
def promo_factor : ℝ := 1.1

-- Theorem statement
theorem toaster_sales_at_promo_price :
  let normal_sales := k / promo_price
  let promo_sales := normal_sales * promo_factor
  promo_sales = 22 := by sorry

end NUMINAMATH_CALUDE_toaster_sales_at_promo_price_l1879_187960


namespace NUMINAMATH_CALUDE_uncool_parents_in_two_classes_l1879_187937

/-- Represents a math class with information about cool parents -/
structure MathClass where
  total_students : ℕ
  cool_dads : ℕ
  cool_moms : ℕ
  both_cool : ℕ

/-- Calculates the number of students with uncool parents in a class -/
def uncool_parents (c : MathClass) : ℕ :=
  c.total_students - (c.cool_dads + c.cool_moms - c.both_cool)

/-- The problem statement -/
theorem uncool_parents_in_two_classes 
  (class1 : MathClass)
  (class2 : MathClass)
  (h1 : class1.total_students = 45)
  (h2 : class1.cool_dads = 22)
  (h3 : class1.cool_moms = 25)
  (h4 : class1.both_cool = 11)
  (h5 : class2.total_students = 35)
  (h6 : class2.cool_dads = 15)
  (h7 : class2.cool_moms = 18)
  (h8 : class2.both_cool = 7) :
  uncool_parents class1 + uncool_parents class2 = 18 := by
  sorry

end NUMINAMATH_CALUDE_uncool_parents_in_two_classes_l1879_187937


namespace NUMINAMATH_CALUDE_uncovered_side_length_l1879_187902

/-- Proves that for a rectangular field with given area and fencing length, 
    the length of the uncovered side is as specified. -/
theorem uncovered_side_length 
  (area : ℝ) 
  (fencing_length : ℝ) 
  (h_area : area = 680) 
  (h_fencing : fencing_length = 178) : 
  ∃ (length width : ℝ), 
    length * width = area ∧ 
    2 * width + length = fencing_length ∧ 
    length = 170 := by
  sorry

end NUMINAMATH_CALUDE_uncovered_side_length_l1879_187902


namespace NUMINAMATH_CALUDE_angle_with_double_supplement_is_60_degrees_l1879_187961

theorem angle_with_double_supplement_is_60_degrees (α : Real) :
  (180 - α = 2 * α) → α = 60 := by
  sorry

end NUMINAMATH_CALUDE_angle_with_double_supplement_is_60_degrees_l1879_187961


namespace NUMINAMATH_CALUDE_tangent_circles_radii_l1879_187981

/-- Two externally tangent circles with specific properties -/
structure TangentCircles where
  r₁ : ℝ  -- radius of the smaller circle
  r₂ : ℝ  -- radius of the larger circle
  h₁ : r₂ = r₁ + 5  -- difference between radii is 5
  h₂ : ∃ (d : ℝ), d = 2.4 * r₁ ∧ d^2 + r₁^2 = (r₂ - r₁)^2  -- distance property

/-- The radii of the two circles are 4 and 9 -/
theorem tangent_circles_radii (c : TangentCircles) : c.r₁ = 4 ∧ c.r₂ = 9 :=
  sorry

end NUMINAMATH_CALUDE_tangent_circles_radii_l1879_187981


namespace NUMINAMATH_CALUDE_existence_of_numbers_l1879_187987

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

end NUMINAMATH_CALUDE_existence_of_numbers_l1879_187987


namespace NUMINAMATH_CALUDE_road_length_proof_l1879_187976

/-- The length of a road given round trip conditions -/
theorem road_length_proof (total_time : ℝ) (walking_speed : ℝ) (bus_speed : ℝ)
  (h1 : total_time = 2)
  (h2 : walking_speed = 5)
  (h3 : bus_speed = 20) :
  ∃ (road_length : ℝ), road_length / walking_speed + road_length / bus_speed = total_time ∧ road_length = 8 := by
  sorry

end NUMINAMATH_CALUDE_road_length_proof_l1879_187976


namespace NUMINAMATH_CALUDE_library_books_before_grant_l1879_187984

theorem library_books_before_grant (books_purchased : ℕ) (total_books_now : ℕ) 
  (h1 : books_purchased = 2647)
  (h2 : total_books_now = 8582) :
  total_books_now - books_purchased = 5935 :=
by sorry

end NUMINAMATH_CALUDE_library_books_before_grant_l1879_187984


namespace NUMINAMATH_CALUDE_tom_brick_cost_l1879_187922

/-- The total cost for Tom's bricks -/
def total_cost (total_bricks : ℕ) (original_price : ℚ) (discount_percent : ℚ) : ℚ :=
  let discounted_bricks := total_bricks / 2
  let full_price_bricks := total_bricks - discounted_bricks
  let discounted_price := original_price * (1 - discount_percent)
  (discounted_bricks : ℚ) * discounted_price + (full_price_bricks : ℚ) * original_price

/-- Theorem stating that the total cost for Tom's bricks is $375 -/
theorem tom_brick_cost :
  total_cost 1000 (1/2) (1/2) = 375 := by
  sorry

end NUMINAMATH_CALUDE_tom_brick_cost_l1879_187922


namespace NUMINAMATH_CALUDE_blackboard_final_product_l1879_187966

/-- Represents the count of each number on the blackboard -/
structure BoardState :=
  (ones : ℕ)
  (twos : ℕ)
  (threes : ℕ)
  (fours : ℕ)

/-- Represents a single operation on the board -/
inductive Operation
  | erase_123_add_4
  | erase_124_add_3
  | erase_134_add_2
  | erase_234_add_1

/-- Applies an operation to a board state -/
def apply_operation (state : BoardState) (op : Operation) : BoardState :=
  match op with
  | Operation.erase_123_add_4 => ⟨state.ones - 1, state.twos - 1, state.threes - 1, state.fours + 2⟩
  | Operation.erase_124_add_3 => ⟨state.ones - 1, state.twos - 1, state.threes + 2, state.fours - 1⟩
  | Operation.erase_134_add_2 => ⟨state.ones - 1, state.twos + 2, state.threes - 1, state.fours - 1⟩
  | Operation.erase_234_add_1 => ⟨state.ones + 2, state.twos - 1, state.threes - 1, state.fours - 1⟩

/-- Checks if a board state is in the final condition (only 3 numbers left) -/
def is_final_state (state : BoardState) : Prop :=
  (state.ones = 0 ∧ state.twos = 2 ∧ state.threes = 1 ∧ state.fours = 0) ∨
  (state.ones = 0 ∧ state.twos = 1 ∧ state.threes = 2 ∧ state.fours = 0) ∨
  (state.ones = 0 ∧ state.twos = 2 ∧ state.threes = 0 ∧ state.fours = 1) ∨
  (state.ones = 1 ∧ state.twos = 0 ∧ state.threes = 2 ∧ state.fours = 0) ∨
  (state.ones = 1 ∧ state.twos = 2 ∧ state.threes = 0 ∧ state.fours = 0) ∨
  (state.ones = 2 ∧ state.twos = 0 ∧ state.threes = 1 ∧ state.fours = 0) ∨
  (state.ones = 2 ∧ state.twos = 1 ∧ state.threes = 0 ∧ state.fours = 0)

/-- Calculates the product of the last three remaining numbers -/
def final_product (state : BoardState) : ℕ :=
  if state.ones > 0 then state.ones else 1 *
  if state.twos > 0 then state.twos else 1 *
  if state.threes > 0 then state.threes else 1 *
  if state.fours > 0 then state.fours else 1

/-- The main theorem to prove -/
theorem blackboard_final_product :
  ∀ (operations : List Operation),
  let initial_state : BoardState := ⟨11, 22, 33, 44⟩
  let final_state := operations.foldl apply_operation initial_state
  is_final_state final_state → final_product final_state = 12 :=
sorry

end NUMINAMATH_CALUDE_blackboard_final_product_l1879_187966


namespace NUMINAMATH_CALUDE_test_results_l1879_187983

/-- Given a class with the following properties:
  * 30 students enrolled
  * 25 students answered question 1 correctly
  * 22 students answered question 2 correctly
  * 18 students answered question 3 correctly
  * 5 students did not take the test
Prove that 18 students answered all three questions correctly. -/
theorem test_results (total_students : ℕ) (q1_correct : ℕ) (q2_correct : ℕ) (q3_correct : ℕ) (absent : ℕ)
  (h1 : total_students = 30)
  (h2 : q1_correct = 25)
  (h3 : q2_correct = 22)
  (h4 : q3_correct = 18)
  (h5 : absent = 5) :
  q3_correct = 18 ∧ q3_correct = (total_students - absent - (total_students - absent - q1_correct) - (total_students - absent - q2_correct)) :=
by sorry

end NUMINAMATH_CALUDE_test_results_l1879_187983


namespace NUMINAMATH_CALUDE_negation_of_universal_proposition_l1879_187904

theorem negation_of_universal_proposition :
  (¬ ∀ x : ℝ, x < 0 → x^3 - x^2 + 1 ≤ 0) ↔ (∃ x : ℝ, x < 0 ∧ x^3 - x^2 + 1 > 0) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_universal_proposition_l1879_187904


namespace NUMINAMATH_CALUDE_eighth_iteration_is_zero_l1879_187946

-- Define the function g based on the graph
def g : ℕ → ℕ
| 0 => 0
| 1 => 8
| 2 => 5
| 3 => 0
| 4 => 7
| 5 => 3
| 6 => 9
| 7 => 2
| 8 => 1
| 9 => 4
| _ => 0  -- Default case for numbers not explicitly shown in the graph

-- Define the iteration of g
def iterate_g (n : ℕ) (x : ℕ) : ℕ :=
  match n with
  | 0 => x
  | n + 1 => iterate_g n (g x)

-- Theorem statement
theorem eighth_iteration_is_zero : iterate_g 8 5 = 0 := by
  sorry

end NUMINAMATH_CALUDE_eighth_iteration_is_zero_l1879_187946


namespace NUMINAMATH_CALUDE_largest_number_l1879_187999

theorem largest_number (a b c d : ℝ) (h1 : a = 1/2) (h2 : b = -1) (h3 : c = |(-2)|) (h4 : d = -3) :
  c ≥ a ∧ c ≥ b ∧ c ≥ d :=
by sorry

end NUMINAMATH_CALUDE_largest_number_l1879_187999


namespace NUMINAMATH_CALUDE_isosceles_triangle_angle_l1879_187932

-- Define the triangle ABC
structure Triangle (A B C : ℝ × ℝ) : Prop where
  isosceles : dist A B = dist B C

-- Define the angles
def angle (A B C : ℝ × ℝ) : ℝ := sorry

-- State the theorem
theorem isosceles_triangle_angle (A B C O : ℝ × ℝ) 
  (h1 : Triangle A B C) 
  (h2 : angle A B C = 80) 
  (h3 : angle O A C = 10) 
  (h4 : angle O C A = 30) : 
  angle A O B = 70 := by sorry

end NUMINAMATH_CALUDE_isosceles_triangle_angle_l1879_187932


namespace NUMINAMATH_CALUDE_independence_implies_a_minus_b_eq_neg_two_l1879_187903

theorem independence_implies_a_minus_b_eq_neg_two :
  ∀ (a b : ℝ), 
  (∀ x : ℝ, ∃ c : ℝ, ∀ y : ℝ, x^2 + a*x - (b*y^2 - y - 3) = c) →
  a - b = -2 :=
by sorry

end NUMINAMATH_CALUDE_independence_implies_a_minus_b_eq_neg_two_l1879_187903


namespace NUMINAMATH_CALUDE_part_one_part_two_l1879_187941

-- Define the propositions p and q
def p (t a : ℝ) : Prop := t^2 - 5*a*t + 4*a^2 < 0

def q (t : ℝ) : Prop := ∃ (x y : ℝ), x^2/(t-2) + y^2/(t-6) = 1 ∧ (t-2)*(t-6) < 0

-- Part I
theorem part_one (t : ℝ) : p t 1 ∧ q t → 2 < t ∧ t < 4 := by sorry

-- Part II
theorem part_two (a : ℝ) : (∀ t : ℝ, q t → p t a) → 3/2 ≤ a ∧ a ≤ 2 := by sorry

end NUMINAMATH_CALUDE_part_one_part_two_l1879_187941


namespace NUMINAMATH_CALUDE_product_from_hcf_lcm_l1879_187951

theorem product_from_hcf_lcm (a b : ℕ+) (h1 : Nat.gcd a b = 12) (h2 : Nat.lcm a b = 205) :
  a * b = 2460 := by
  sorry

end NUMINAMATH_CALUDE_product_from_hcf_lcm_l1879_187951


namespace NUMINAMATH_CALUDE_geometric_progression_sum_inequality_l1879_187948

/-- An increasing positive geometric progression -/
def IsIncreasingPositiveGP (b : ℕ → ℝ) : Prop :=
  ∃ (q : ℝ), q > 1 ∧ ∀ n, b n > 0 ∧ b (n + 1) = b n * q

theorem geometric_progression_sum_inequality 
  (b : ℕ → ℝ) 
  (h_gp : IsIncreasingPositiveGP b) 
  (h_sum : b 4 + b 3 - b 2 - b 1 = 5) : 
  b 6 + b 5 ≥ 20 := by
sorry

end NUMINAMATH_CALUDE_geometric_progression_sum_inequality_l1879_187948


namespace NUMINAMATH_CALUDE_x1_value_l1879_187968

theorem x1_value (x1 x2 x3 x4 : Real) 
  (h_order : 0 ≤ x4 ∧ x4 ≤ x3 ∧ x3 ≤ x2 ∧ x2 ≤ x1 ∧ x1 ≤ 1) 
  (h_eq : (1-x1)^2 + (x1-x2)^2 + (x2-x3)^2 + (x3-x4)^2 + x4^2 = 1/5) :
  x1 = 4/5 := by
  sorry

end NUMINAMATH_CALUDE_x1_value_l1879_187968


namespace NUMINAMATH_CALUDE_simplify_expression_l1879_187918

theorem simplify_expression (a : ℝ) : 6*a - 5*a + 4*a - 3*a + 2*a - a = 3*a := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l1879_187918


namespace NUMINAMATH_CALUDE_meat_for_hamburgers_l1879_187967

/-- Given that 5 pounds of meat make 10 hamburgers, prove that 15 pounds of meat are needed for 30 hamburgers -/
theorem meat_for_hamburgers (meat_per_10 : ℕ) (hamburgers_per_5 : ℕ) 
  (h1 : meat_per_10 = 5) 
  (h2 : hamburgers_per_5 = 10) :
  (meat_per_10 * 3 : ℕ) = 15 ∧ (hamburgers_per_5 * 3 : ℕ) = 30 := by
  sorry

end NUMINAMATH_CALUDE_meat_for_hamburgers_l1879_187967


namespace NUMINAMATH_CALUDE_quadratic_inequality_range_sufficient_condition_range_l1879_187998

-- Part I
theorem quadratic_inequality_range (a : ℝ) :
  (∀ x : ℝ, x^2 - 3*a*x + 9 > 0) ↔ -2 ≤ a ∧ a ≤ 2 :=
sorry

-- Part II
theorem sufficient_condition_range (m : ℝ) :
  ((∀ x : ℝ, x^2 + 2*x - 8 < 0 → x - m > 0) ∧
   (∃ x : ℝ, x - m > 0 ∧ x^2 + 2*x - 8 ≥ 0)) →
  m ≤ -4 :=
sorry

end NUMINAMATH_CALUDE_quadratic_inequality_range_sufficient_condition_range_l1879_187998


namespace NUMINAMATH_CALUDE_taxi_charge_theorem_l1879_187910

-- Define the parameters of the taxi service
def initial_fee : ℚ := 235 / 100
def charge_per_increment : ℚ := 35 / 100
def miles_per_increment : ℚ := 2 / 5
def trip_distance : ℚ := 36 / 10

-- Define the total charge function
def total_charge (initial_fee charge_per_increment miles_per_increment trip_distance : ℚ) : ℚ :=
  initial_fee + (trip_distance / miles_per_increment) * charge_per_increment

-- State the theorem
theorem taxi_charge_theorem :
  total_charge initial_fee charge_per_increment miles_per_increment trip_distance = 865 / 100 := by
  sorry

end NUMINAMATH_CALUDE_taxi_charge_theorem_l1879_187910


namespace NUMINAMATH_CALUDE_base_subtraction_l1879_187917

/-- Convert a number from base b to base 10 -/
def to_base_10 (digits : List Nat) (b : Nat) : Nat :=
  digits.foldr (fun d acc => d + b * acc) 0

/-- Express 343₈ - 265₇ as a base 10 integer -/
theorem base_subtraction : 
  let base_8_num := to_base_10 [3, 4, 3] 8
  let base_7_num := to_base_10 [2, 6, 5] 7
  base_8_num - base_7_num = 82 := by
sorry

end NUMINAMATH_CALUDE_base_subtraction_l1879_187917


namespace NUMINAMATH_CALUDE_arithmetic_sequence_range_l1879_187920

def arithmetic_sequence (a₁ : ℝ) (d : ℝ) (n : ℕ) : ℝ := a₁ + d * (n - 1 : ℝ)

theorem arithmetic_sequence_range :
  ∀ d : ℝ,
  (arithmetic_sequence (-24) d 1 = -24) →
  (arithmetic_sequence (-24) d 10 > 0) →
  (arithmetic_sequence (-24) d 9 ≤ 0) →
  (8/3 < d ∧ d ≤ 3) :=
by sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_range_l1879_187920


namespace NUMINAMATH_CALUDE_first_knife_price_is_50_l1879_187947

/-- Represents the daily sales data for a door-to-door salesman --/
structure SalesData where
  houses_visited : ℕ
  purchase_rate : ℚ
  expensive_knife_price : ℕ
  weekly_revenue : ℕ
  work_days : ℕ

/-- Calculates the price of the first set of knives based on the given sales data --/
def calculate_knife_price (data : SalesData) : ℚ :=
  let buyers := data.houses_visited * data.purchase_rate
  let expensive_knife_buyers := buyers / 2
  let weekly_expensive_knife_revenue := expensive_knife_buyers * data.expensive_knife_price * data.work_days
  let weekly_first_knife_revenue := data.weekly_revenue - weekly_expensive_knife_revenue
  let weekly_first_knife_sales := expensive_knife_buyers * data.work_days
  weekly_first_knife_revenue / weekly_first_knife_sales

/-- Theorem stating that the price of the first set of knives is $50 --/
theorem first_knife_price_is_50 (data : SalesData)
  (h1 : data.houses_visited = 50)
  (h2 : data.purchase_rate = 1/5)
  (h3 : data.expensive_knife_price = 150)
  (h4 : data.weekly_revenue = 5000)
  (h5 : data.work_days = 5) :
  calculate_knife_price data = 50 := by
  sorry

end NUMINAMATH_CALUDE_first_knife_price_is_50_l1879_187947


namespace NUMINAMATH_CALUDE_limit_implies_a_and_b_limit_implies_a_range_l1879_187933

-- Problem 1
theorem limit_implies_a_and_b (a b : ℝ) :
  (∀ ε > 0, ∃ N, ∀ n ≥ N, |2*n^2 / (n+2) - n*a - b| < ε) →
  a = 2 ∧ b = 4 := by sorry

-- Problem 2
theorem limit_implies_a_range (a : ℝ) :
  (∀ ε > 0, ∃ N, ∀ n ≥ N, |3^n / (3^(n+1) + (a+1)^n) - 1/3| < ε) →
  -4 < a ∧ a < 2 := by sorry

end NUMINAMATH_CALUDE_limit_implies_a_and_b_limit_implies_a_range_l1879_187933


namespace NUMINAMATH_CALUDE_simplify_expression_l1879_187995

theorem simplify_expression : ((4 + 6) * 2) / 4 - 3 / 4 = 17 / 4 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l1879_187995


namespace NUMINAMATH_CALUDE_school_store_problem_l1879_187973

/-- Represents the cost of pencils and notebooks given certain pricing conditions -/
def school_store_cost (pencil_price notebook_price : ℚ) : Prop :=
  -- 10 pencils and 6 notebooks cost $3.50
  10 * pencil_price + 6 * notebook_price = (3.50 : ℚ) ∧
  -- 4 pencils and 9 notebooks cost $2.70
  4 * pencil_price + 9 * notebook_price = (2.70 : ℚ)

/-- Calculates the total cost including the fixed fee -/
def total_cost (pencil_price notebook_price : ℚ) (pencil_count notebook_count : ℕ) : ℚ :=
  let base_cost := pencil_count * pencil_price + notebook_count * notebook_price
  if pencil_count + notebook_count > 15 then base_cost + (0.50 : ℚ) else base_cost

/-- Theorem stating the cost of 24 pencils and 15 notebooks -/
theorem school_store_problem :
  ∃ (pencil_price notebook_price : ℚ),
    school_store_cost pencil_price notebook_price →
    total_cost pencil_price notebook_price 24 15 = (9.02 : ℚ) := by
  sorry


end NUMINAMATH_CALUDE_school_store_problem_l1879_187973


namespace NUMINAMATH_CALUDE_greatest_divisor_630_under_60_and_factor_90_l1879_187916

def is_greatest_divisor (n : ℕ) : Prop :=
  n ∣ 630 ∧ n < 60 ∧ n ∣ 90 ∧
  ∀ m : ℕ, m ∣ 630 → m < 60 → m ∣ 90 → m ≤ n

theorem greatest_divisor_630_under_60_and_factor_90 :
  is_greatest_divisor 45 := by
  sorry

end NUMINAMATH_CALUDE_greatest_divisor_630_under_60_and_factor_90_l1879_187916


namespace NUMINAMATH_CALUDE_first_player_winning_strategy_l1879_187952

/-- Represents the state of the game -/
structure GameState where
  score : ℕ
  remainingCards : List ℕ

/-- Defines a valid move in the game -/
def validMove (state : GameState) (card : ℕ) : Prop :=
  card ∈ state.remainingCards ∧ card ≥ 1 ∧ card ≤ 4

/-- Defines the winning condition -/
def isWinningMove (state : GameState) (card : ℕ) : Prop :=
  validMove state card ∧ (state.score + card = 22 ∨ state.score + card > 22)

/-- Theorem stating that the first player has a winning strategy -/
theorem first_player_winning_strategy :
  ∃ (initialCards : List ℕ),
    (initialCards.length = 16) ∧
    (∀ c ∈ initialCards, c ≥ 1 ∧ c ≤ 4) ∧
    (∃ (strategy : GameState → ℕ),
      ∀ (opponentStrategy : GameState → ℕ),
        let initialState : GameState := { score := 0, remainingCards := initialCards }
        let firstMove := strategy initialState
        validMove initialState firstMove ∧ firstMove = 1 →
        ∃ (finalState : GameState),
          isWinningMove finalState (strategy finalState)) :=
sorry

end NUMINAMATH_CALUDE_first_player_winning_strategy_l1879_187952


namespace NUMINAMATH_CALUDE_graph_not_in_first_quadrant_l1879_187945

-- Define the function
def f (k x : ℝ) : ℝ := k * (x - k)

-- Theorem statement
theorem graph_not_in_first_quadrant (k : ℝ) (h : k < 0) :
  ∀ x y : ℝ, f k x = y → ¬(x > 0 ∧ y > 0) :=
by sorry

end NUMINAMATH_CALUDE_graph_not_in_first_quadrant_l1879_187945


namespace NUMINAMATH_CALUDE_cube_sum_over_product_equals_thirteen_l1879_187935

theorem cube_sum_over_product_equals_thirteen
  (a b c : ℂ)
  (nonzero_a : a ≠ 0)
  (nonzero_b : b ≠ 0)
  (nonzero_c : c ≠ 0)
  (sum_equals_ten : a + b + c = 10)
  (squared_diff_sum : (a - b)^2 + (a - c)^2 + (b - c)^2 = 2*a*b*c) :
  (a^3 + b^3 + c^3) / (a*b*c) = 13 := by
sorry

end NUMINAMATH_CALUDE_cube_sum_over_product_equals_thirteen_l1879_187935


namespace NUMINAMATH_CALUDE_count_distinct_n_values_l1879_187936

/-- Given a quadratic equation x² - nx + 36 = 0 with integer roots,
    there are exactly 10 distinct possible values for n. -/
theorem count_distinct_n_values : ∃ (S : Finset ℤ),
  (∀ n ∈ S, ∃ x₁ x₂ : ℤ, x₁ * x₂ = 36 ∧ x₁ + x₂ = n) ∧
  (∀ n : ℤ, (∃ x₁ x₂ : ℤ, x₁ * x₂ = 36 ∧ x₁ + x₂ = n) → n ∈ S) ∧
  Finset.card S = 10 :=
sorry

end NUMINAMATH_CALUDE_count_distinct_n_values_l1879_187936


namespace NUMINAMATH_CALUDE_water_students_l1879_187979

theorem water_students (total : ℕ) (juice_percent : ℚ) (water_percent : ℚ) (juice_students : ℕ) : ℕ :=
  sorry

end NUMINAMATH_CALUDE_water_students_l1879_187979


namespace NUMINAMATH_CALUDE_square_root_sum_of_squares_l1879_187911

theorem square_root_sum_of_squares (x y : ℝ) : 
  (∃ (s : ℝ), s^2 = x - 2 ∧ (s = 2 ∨ s = -2)) →
  (2*x + y + 7)^(1/3) = 3 →
  ∃ (t : ℝ), t^2 = x^2 + y^2 ∧ (t = 10 ∨ t = -10) := by
  sorry

end NUMINAMATH_CALUDE_square_root_sum_of_squares_l1879_187911


namespace NUMINAMATH_CALUDE_count_primes_with_digit_sum_10_l1879_187914

def is_two_digit (n : ℕ) : Prop := 10 ≤ n ∧ n < 100

def digit_sum (n : ℕ) : ℕ := (n / 10) + (n % 10)

def satisfies_condition (n : ℕ) : Prop :=
  is_two_digit n ∧ Nat.Prime n ∧ digit_sum n = 10

theorem count_primes_with_digit_sum_10 :
  ∃ (S : Finset ℕ), (∀ n ∈ S, satisfies_condition n) ∧ S.card = 3 :=
sorry

end NUMINAMATH_CALUDE_count_primes_with_digit_sum_10_l1879_187914


namespace NUMINAMATH_CALUDE_overall_rate_relation_l1879_187978

/-- Given three deposit amounts and their respective interest rates, 
    this theorem proves the relation for the overall annual percentage rate. -/
theorem overall_rate_relation 
  (P1 P2 P3 : ℝ) 
  (R1 R2 R3 : ℝ) 
  (h1 : P1 * (1 + R1)^2 + P2 * (1 + R2)^2 + P3 * (1 + R3)^2 = 2442)
  (h2 : P1 * (1 + R1)^3 + P2 * (1 + R2)^3 + P3 * (1 + R3)^3 = 2926) :
  ∃ R : ℝ, (1 + R)^3 / (1 + R)^2 = 2926 / 2442 :=
sorry

end NUMINAMATH_CALUDE_overall_rate_relation_l1879_187978


namespace NUMINAMATH_CALUDE_mean_of_five_numbers_with_sum_one_third_l1879_187980

theorem mean_of_five_numbers_with_sum_one_third :
  ∀ (a b c d e : ℚ), 
    a + b + c + d + e = 1/3 →
    (a + b + c + d + e) / 5 = 1/15 := by
sorry

end NUMINAMATH_CALUDE_mean_of_five_numbers_with_sum_one_third_l1879_187980


namespace NUMINAMATH_CALUDE_second_smallest_prime_perimeter_l1879_187901

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m > 1 → m < n → ¬(m ∣ n)

def is_scalene_triangle (a b c : ℕ) : Prop := a ≠ b ∧ b ≠ c ∧ a ≠ c

def is_valid_triangle (a b c : ℕ) : Prop :=
  a + b > c ∧ b + c > a ∧ c + a > b

def prime_perimeter_triangle (a b c : ℕ) : Prop :=
  is_prime a ∧ is_prime b ∧ is_prime c ∧
  is_scalene_triangle a b c ∧
  is_valid_triangle a b c ∧
  is_prime (a + b + c)

theorem second_smallest_prime_perimeter :
  ∃ (a b c : ℕ),
    prime_perimeter_triangle a b c ∧
    (a + b + c = 29) ∧
    (∀ (x y z : ℕ),
      prime_perimeter_triangle x y z →
      (x + y + z ≠ 23) →
      (x + y + z ≥ 29)) :=
sorry

end NUMINAMATH_CALUDE_second_smallest_prime_perimeter_l1879_187901


namespace NUMINAMATH_CALUDE_cyclic_sum_squares_identity_l1879_187921

theorem cyclic_sum_squares_identity (a b c x y z : ℝ) :
  (a * x + b * y + c * z)^2 + (b * x + c * y + a * z)^2 + (c * x + a * y + b * z)^2 =
  (c * x + b * y + a * z)^2 + (b * x + a * y + c * z)^2 + (a * x + c * y + b * z)^2 :=
by sorry

end NUMINAMATH_CALUDE_cyclic_sum_squares_identity_l1879_187921


namespace NUMINAMATH_CALUDE_ryegrass_percentage_in_mixture_X_l1879_187905

/-- Proves that the percentage of ryegrass in seed mixture X is 40% -/
theorem ryegrass_percentage_in_mixture_X : ∀ (x : ℝ),
  -- Seed mixture X has x% ryegrass and 60% bluegrass
  x + 60 = 100 →
  -- A mixture of 86.67% X and 13.33% Y contains 38% ryegrass
  0.8667 * x + 0.1333 * 25 = 38 →
  -- The percentage of ryegrass in seed mixture X is 40%
  x = 40 :=
by
  sorry


end NUMINAMATH_CALUDE_ryegrass_percentage_in_mixture_X_l1879_187905


namespace NUMINAMATH_CALUDE_min_value_theorem_l1879_187957

theorem min_value_theorem (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : a + 3*b = 1) :
  (1/a + 1/(3*b)) ≥ 4 ∧ ∃ (a₀ b₀ : ℝ), a₀ > 0 ∧ b₀ > 0 ∧ a₀ + 3*b₀ = 1 ∧ 1/a₀ + 1/(3*b₀) = 4 :=
by sorry

end NUMINAMATH_CALUDE_min_value_theorem_l1879_187957


namespace NUMINAMATH_CALUDE_election_result_theorem_l1879_187928

/-- Represents the result of a mayoral election. -/
structure ElectionResult where
  total_votes : ℕ
  candidates : ℕ
  winner_votes : ℕ
  second_place_votes : ℕ
  third_place_votes : ℕ
  fourth_place_votes : ℕ
  winner_third_diff : ℕ
  winner_fourth_diff : ℕ

/-- Theorem stating the conditions and the result to be proved. -/
theorem election_result_theorem (e : ElectionResult) 
  (h1 : e.total_votes = 979)
  (h2 : e.candidates = 4)
  (h3 : e.winner_votes = e.fourth_place_votes + e.winner_fourth_diff)
  (h4 : e.winner_votes = e.third_place_votes + e.winner_third_diff)
  (h5 : e.fourth_place_votes = 199)
  (h6 : e.winner_fourth_diff = 105)
  (h7 : e.winner_third_diff = 79)
  (h8 : e.total_votes = e.winner_votes + e.second_place_votes + e.third_place_votes + e.fourth_place_votes) :
  e.winner_votes - e.second_place_votes = 53 := by
  sorry


end NUMINAMATH_CALUDE_election_result_theorem_l1879_187928


namespace NUMINAMATH_CALUDE_basketball_match_probabilities_l1879_187988

/-- Represents the probability of a team winning a single game -/
structure GameProbability where
  teamA : ℝ
  teamB : ℝ
  sum_to_one : teamA + teamB = 1

/-- Calculates the probability of team A winning by a score of 2 to 1 -/
def prob_A_wins_2_1 (p : GameProbability) : ℝ :=
  2 * p.teamA * p.teamB * p.teamA

/-- Calculates the probability of team B winning the match -/
def prob_B_wins (p : GameProbability) : ℝ :=
  p.teamB * p.teamB + 2 * p.teamA * p.teamB * p.teamB

/-- The main theorem stating the probabilities for the given scenario -/
theorem basketball_match_probabilities (p : GameProbability) 
  (hA : p.teamA = 0.6) (hB : p.teamB = 0.4) :
  prob_A_wins_2_1 p = 0.288 ∧ prob_B_wins p = 0.352 := by
  sorry


end NUMINAMATH_CALUDE_basketball_match_probabilities_l1879_187988


namespace NUMINAMATH_CALUDE_total_weight_of_baskets_l1879_187990

def basket_weight : ℕ := 30
def num_baskets : ℕ := 8

theorem total_weight_of_baskets : basket_weight * num_baskets = 240 := by
  sorry

end NUMINAMATH_CALUDE_total_weight_of_baskets_l1879_187990


namespace NUMINAMATH_CALUDE_vector_difference_norm_l1879_187949

variable {V : Type*} [NormedAddCommGroup V] [InnerProductSpace ℝ V]

theorem vector_difference_norm (a b : V)
  (ha : ‖a‖ = 6)
  (hb : ‖b‖ = 8)
  (hab : ‖a + b‖ = ‖a - b‖) :
  ‖a - b‖ = 10 := by
  sorry

end NUMINAMATH_CALUDE_vector_difference_norm_l1879_187949


namespace NUMINAMATH_CALUDE_inscribed_squares_ratio_l1879_187965

/-- Given a right triangle with sides 5, 12, and 13, let x be the side length of a square
    inscribed with one vertex at the right angle, and y be the side length of a square
    inscribed with one side on a leg of the triangle. Then x/y = 20/17. -/
theorem inscribed_squares_ratio (x y : ℝ) : 
  x > 0 ∧ y > 0 ∧
  x^2 + (12 - x)^2 = 13^2 ∧
  x^2 + (5 - x)^2 = 12^2 ∧
  y^2 + (5 - y)^2 = (12 - y)^2 →
  x / y = 20 / 17 := by
sorry

end NUMINAMATH_CALUDE_inscribed_squares_ratio_l1879_187965


namespace NUMINAMATH_CALUDE_chip_ratio_l1879_187977

/-- Represents the number of chips each person has -/
structure ChipCount where
  susana_chocolate : ℕ
  susana_vanilla : ℕ
  viviana_chocolate : ℕ
  viviana_vanilla : ℕ

/-- The conditions given in the problem -/
def problem_conditions (c : ChipCount) : Prop :=
  c.viviana_chocolate = c.susana_chocolate + 5 ∧
  c.viviana_vanilla = 20 ∧
  c.susana_chocolate = 25 ∧
  c.susana_chocolate + c.susana_vanilla + c.viviana_chocolate + c.viviana_vanilla = 90

/-- The theorem to prove -/
theorem chip_ratio (c : ChipCount) :
  problem_conditions c → c.susana_vanilla * 4 = c.viviana_vanilla * 3 :=
by
  sorry

end NUMINAMATH_CALUDE_chip_ratio_l1879_187977


namespace NUMINAMATH_CALUDE_reflection_line_sum_l1879_187906

/-- Given a line y = mx + b, if the reflection of point (2,2) across this line is (10,6), then m + b = 14 -/
theorem reflection_line_sum (m b : ℝ) : 
  (∃ (x y : ℝ), (x, y) = (10, 6) ∧ 
    (x - 2, y - 2) = (2 * (m * (x + 2) / 2 + b - y) / (1 + m^2), 
                      2 * (m * (y + 2) / 2 - (x + 2) / 2 + b) / (1 + m^2))) →
  m + b = 14 := by
sorry


end NUMINAMATH_CALUDE_reflection_line_sum_l1879_187906


namespace NUMINAMATH_CALUDE_sqrt_two_squared_times_three_l1879_187994

theorem sqrt_two_squared_times_three : 4 - (Real.sqrt 2)^2 * 3 = -2 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_two_squared_times_three_l1879_187994


namespace NUMINAMATH_CALUDE_no_solution_sqrt_plus_one_l1879_187971

theorem no_solution_sqrt_plus_one :
  ∀ x : ℝ, ¬(Real.sqrt (x + 4) + 1 = 0) := by
  sorry

end NUMINAMATH_CALUDE_no_solution_sqrt_plus_one_l1879_187971


namespace NUMINAMATH_CALUDE_square_equality_l1879_187919

theorem square_equality (a b c : ℤ) 
  (h : (a + 3)^2 + (b + 4)^2 - (c + 5)^2 = a^2 + b^2 - c^2) : 
  ∃ k : ℤ, a^2 + b^2 - c^2 = k^2 := by
  sorry

end NUMINAMATH_CALUDE_square_equality_l1879_187919


namespace NUMINAMATH_CALUDE_rook_paths_bound_l1879_187942

def ChessboardPaths (n : ℕ) : ℕ := sorry

theorem rook_paths_bound (n : ℕ) :
  ChessboardPaths n ≤ 9^n ∧ ∀ k < 9, ∃ m : ℕ, ChessboardPaths m > k^m :=
by sorry

end NUMINAMATH_CALUDE_rook_paths_bound_l1879_187942


namespace NUMINAMATH_CALUDE_special_ellipse_properties_l1879_187956

/-- An ellipse with the given properties -/
structure SpecialEllipse where
  a : ℝ
  b : ℝ
  h_ab : a > b ∧ b > 0
  P : ℝ × ℝ
  h_P_on_ellipse : (P.1 / a) ^ 2 + (P.2 / b) ^ 2 = 1
  F₁ : ℝ × ℝ
  F₂ : ℝ × ℝ
  h_PF₂_eq_F₁F₂ : dist P F₂ = dist F₁ F₂
  A : ℝ × ℝ
  B : ℝ × ℝ
  h_AB_on_ellipse : (A.1 / a) ^ 2 + (A.2 / b) ^ 2 = 1 ∧ (B.1 / a) ^ 2 + (B.2 / b) ^ 2 = 1
  h_AB_on_PF₂ : ∃ (t : ℝ), A = (1 - t) • P + t • F₂ ∧ B = (1 - t) • P + t • F₂
  M : ℝ × ℝ
  N : ℝ × ℝ
  h_MN_on_circle : (M.1 + 1) ^ 2 + (M.2 - Real.sqrt 3) ^ 2 = 16 ∧
                   (N.1 + 1) ^ 2 + (N.2 - Real.sqrt 3) ^ 2 = 16
  h_MN_on_PF₂ : ∃ (s : ℝ), M = (1 - s) • P + s • F₂ ∧ N = (1 - s) • P + s • F₂
  h_MN_AB_ratio : dist M N = (5 / 8) * dist A B

/-- The eccentricity and equation of the special ellipse -/
theorem special_ellipse_properties (e : SpecialEllipse) :
  (∃ (c : ℝ), c > 0 ∧ c ^ 2 = a ^ 2 - b ^ 2 ∧ c / a = 1 / 2) ∧
  (∃ (k : ℝ), k > 0 ∧ e.a ^ 2 = 16 * k ∧ e.b ^ 2 = 12 * k) :=
sorry

end NUMINAMATH_CALUDE_special_ellipse_properties_l1879_187956


namespace NUMINAMATH_CALUDE_fraction_zero_implies_x_equals_two_l1879_187963

theorem fraction_zero_implies_x_equals_two (x : ℝ) :
  (|x| - 2) / (x + 2) = 0 ∧ x + 2 ≠ 0 → x = 2 := by
sorry

end NUMINAMATH_CALUDE_fraction_zero_implies_x_equals_two_l1879_187963


namespace NUMINAMATH_CALUDE_tangent_perpendicular_implies_negative_a_l1879_187974

/-- Given a real-valued function f(x) = ax³ + ln x, prove that if there exists a positive real number x
    such that the derivative of f at x is zero, then a is negative. -/
theorem tangent_perpendicular_implies_negative_a (a : ℝ) :
  (∃ x : ℝ, x > 0 ∧ 3 * a * x^2 + 1 / x = 0) → a < 0 := by
  sorry

end NUMINAMATH_CALUDE_tangent_perpendicular_implies_negative_a_l1879_187974


namespace NUMINAMATH_CALUDE_volunteer_distribution_count_l1879_187986

def distribute_volunteers (n : ℕ) (k : ℕ) : ℕ :=
  sorry

theorem volunteer_distribution_count : 
  distribute_volunteers 5 4 = 216 :=
sorry

end NUMINAMATH_CALUDE_volunteer_distribution_count_l1879_187986


namespace NUMINAMATH_CALUDE_complex_fraction_simplification_l1879_187927

-- Define the complex number i
noncomputable def i : ℂ := Complex.I

-- State the problem
theorem complex_fraction_simplification :
  (2 + 4 * i) / (1 - 5 * i) = -9/13 + (7/13) * i :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_CALUDE_complex_fraction_simplification_l1879_187927


namespace NUMINAMATH_CALUDE_min_difference_l1879_187991

open Real

noncomputable def f (x : ℝ) : ℝ := 1 - Real.sqrt (1 - 2 * x)

noncomputable def g (x : ℝ) : ℝ := Real.log x

theorem min_difference (m n : ℝ) (h1 : m ≤ 1/2) (h2 : n > 0) (h3 : f m = g n) :
  ∃ (diff : ℝ), diff = n - m ∧ diff ≥ 1 ∧ 
  ∀ (m' n' : ℝ), m' ≤ 1/2 → n' > 0 → f m' = g n' → n' - m' ≥ diff :=
sorry

end NUMINAMATH_CALUDE_min_difference_l1879_187991
