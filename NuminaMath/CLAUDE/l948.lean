import Mathlib

namespace NUMINAMATH_CALUDE_flower_arrangement_theorem_l948_94869

/-- The number of ways to arrange flowers of three different hues -/
def flower_arrangements (X : ℕ+) : ℕ :=
  30

/-- Theorem stating that the number of valid flower arrangements is always 30 -/
theorem flower_arrangement_theorem (X : ℕ+) :
  flower_arrangements X = 30 :=
by
  sorry

end NUMINAMATH_CALUDE_flower_arrangement_theorem_l948_94869


namespace NUMINAMATH_CALUDE_sum_of_roots_l948_94863

theorem sum_of_roots (x : ℝ) : 
  let f : ℝ → ℝ := λ x => 3*x^3 - 7*x^2 + 2*x
  (∃ r₁ r₂ : ℝ, r₁ ≠ 0 ∧ r₂ ≠ 0 ∧ f r₁ = 0 ∧ f r₂ = 0 ∧ r₁ + r₂ = 7/3) :=
by sorry

end NUMINAMATH_CALUDE_sum_of_roots_l948_94863


namespace NUMINAMATH_CALUDE_no_rational_roots_l948_94860

theorem no_rational_roots (p q : ℤ) (hp : Odd p) (hq : Odd q) :
  ∀ (x : ℚ), x^2 + 2*p*x + 2*q ≠ 0 := by
  sorry

end NUMINAMATH_CALUDE_no_rational_roots_l948_94860


namespace NUMINAMATH_CALUDE_intersection_line_equation_l948_94859

-- Define the circle (x-1)^2 + y^2 = 1
def circle1 (x y : ℝ) : Prop := (x - 1)^2 + y^2 = 1

-- Define point P
def P : ℝ × ℝ := (2, 3)

-- Define point C (center of circle1)
def C : ℝ × ℝ := (1, 0)

-- Define the circle with diameter PC
def circle2 (x y : ℝ) : Prop := (x - (P.1 + C.1)/2)^2 + (y - (P.2 + C.2)/2)^2 = ((P.1 - C.1)^2 + (P.2 - C.2)^2) / 4

-- Theorem statement
theorem intersection_line_equation : 
  ∀ x y : ℝ, circle1 x y ∧ circle2 x y → x + 3*y - 2 = 0 := by sorry

end NUMINAMATH_CALUDE_intersection_line_equation_l948_94859


namespace NUMINAMATH_CALUDE_f_plus_g_at_2_l948_94885

-- Define f and g as functions from ℝ to ℝ
variable (f g : ℝ → ℝ)

-- Define the properties of f and g
def is_even (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = f x
def is_odd (g : ℝ → ℝ) : Prop := ∀ x, g (-x) = -g x

-- State the theorem
theorem f_plus_g_at_2 (hf : is_even f) (hg : is_odd g) 
  (h : ∀ x, f x - g x = x^3 + 2^(-x)) : 
  f 2 + g 2 = -4 := by
  sorry

end NUMINAMATH_CALUDE_f_plus_g_at_2_l948_94885


namespace NUMINAMATH_CALUDE_walking_time_calculation_l948_94842

/-- A person walks at a constant rate. They cover 36 yards in 18 minutes and have 120 feet left to walk. -/
theorem walking_time_calculation (distance_covered : ℝ) (time_taken : ℝ) (distance_left : ℝ) :
  distance_covered = 36 * 3 →
  time_taken = 18 →
  distance_left = 120 →
  distance_left / (distance_covered / time_taken) = 20 := by
  sorry

end NUMINAMATH_CALUDE_walking_time_calculation_l948_94842


namespace NUMINAMATH_CALUDE_count_distinguishable_triangles_l948_94897

/-- Represents the number of available colors for small triangles -/
def num_colors : ℕ := 8

/-- Represents a large equilateral triangle constructed from four smaller triangles -/
structure LargeTriangle where
  corner1 : Fin num_colors
  corner2 : Fin num_colors
  corner3 : Fin num_colors
  center : Fin num_colors

/-- Two large triangles are considered equivalent if they can be matched by rotations or reflections -/
def equivalent (t1 t2 : LargeTriangle) : Prop :=
  ∃ (perm : Fin 3 → Fin 3), 
    (t1.corner1 = t2.corner1 ∧ t1.corner2 = t2.corner2 ∧ t1.corner3 = t2.corner3) ∨
    (t1.corner1 = t2.corner2 ∧ t1.corner2 = t2.corner3 ∧ t1.corner3 = t2.corner1) ∨
    (t1.corner1 = t2.corner3 ∧ t1.corner2 = t2.corner1 ∧ t1.corner3 = t2.corner2)

/-- The set of all distinguishable large triangles -/
def distinguishable_triangles : Finset LargeTriangle :=
  sorry

theorem count_distinguishable_triangles : 
  Finset.card distinguishable_triangles = 960 := by
  sorry

end NUMINAMATH_CALUDE_count_distinguishable_triangles_l948_94897


namespace NUMINAMATH_CALUDE_subset_implies_bound_l948_94836

theorem subset_implies_bound (A B : Set ℝ) (a : ℝ) : 
  A = {x : ℝ | 1 < x ∧ x < 2} →
  B = {x : ℝ | x < a} →
  A ⊆ B →
  a ≥ 2 := by
sorry

end NUMINAMATH_CALUDE_subset_implies_bound_l948_94836


namespace NUMINAMATH_CALUDE_a_power_sum_l948_94829

theorem a_power_sum (a : ℂ) (h : a^2 - a + 1 = 0) : a^10 + a^20 + a^30 = 3 := by
  sorry

end NUMINAMATH_CALUDE_a_power_sum_l948_94829


namespace NUMINAMATH_CALUDE_repetend_of_five_seventeenths_l948_94809

theorem repetend_of_five_seventeenths :
  ∃ (n : ℕ), (5 : ℚ) / 17 = (n : ℚ) / 999999999999 ∧ 
  n = 294117647058 :=
sorry

end NUMINAMATH_CALUDE_repetend_of_five_seventeenths_l948_94809


namespace NUMINAMATH_CALUDE_units_digit_of_sum_of_powers_l948_94816

theorem units_digit_of_sum_of_powers : ∃ n : ℕ, n < 10 ∧ (42^4 + 24^4) % 10 = n ∧ n = 2 := by
  sorry

end NUMINAMATH_CALUDE_units_digit_of_sum_of_powers_l948_94816


namespace NUMINAMATH_CALUDE_gcd_lcm_a_b_l948_94894

-- Define a and b
def a : Nat := 2 * 3 * 7
def b : Nat := 2 * 3 * 3 * 5

-- State the theorem
theorem gcd_lcm_a_b : Nat.gcd a b = 6 ∧ Nat.lcm a b = 630 := by
  sorry

end NUMINAMATH_CALUDE_gcd_lcm_a_b_l948_94894


namespace NUMINAMATH_CALUDE_journey_to_the_west_readers_l948_94857

theorem journey_to_the_west_readers (total : ℕ) (either : ℕ) (dream : ℕ) (both : ℕ) 
  (h1 : total = 100)
  (h2 : either = 90)
  (h3 : dream = 80)
  (h4 : both = 60)
  (h5 : either ≤ total)
  (h6 : dream ≤ total)
  (h7 : both ≤ dream)
  (h8 : both ≤ either) : 
  ∃ (journey : ℕ), journey = 70 ∧ journey = either + both - dream := by
  sorry

end NUMINAMATH_CALUDE_journey_to_the_west_readers_l948_94857


namespace NUMINAMATH_CALUDE_tops_and_chudis_problem_l948_94888

/-- The price of tops and chudis problem -/
theorem tops_and_chudis_problem (C T : ℚ) : 
  (3 * C + 6 * T = 1500) →  -- Price of 3 chudis and 6 tops
  (C + 12 * T = 1500) →     -- Price of 1 chudi and 12 tops
  (500 / T = 5) :=          -- Number of tops for Rs. 500
by
  sorry

#check tops_and_chudis_problem

end NUMINAMATH_CALUDE_tops_and_chudis_problem_l948_94888


namespace NUMINAMATH_CALUDE_rectangular_solid_volume_l948_94800

theorem rectangular_solid_volume (a b c : ℝ) 
  (h_top : a * b = 15)
  (h_front : b * c = 10)
  (h_side : c * a = 6) :
  a * b * c = 30 := by
sorry

end NUMINAMATH_CALUDE_rectangular_solid_volume_l948_94800


namespace NUMINAMATH_CALUDE_det_trig_matrix_zero_l948_94871

/-- The determinant of the matrix
    [1, cos(a-b), sin(a);
     cos(a-b), 1, sin(b);
     sin(a), sin(b), 1]
    is equal to 0 for any real numbers a and b. -/
theorem det_trig_matrix_zero (a b : ℝ) : 
  let M : Matrix (Fin 3) (Fin 3) ℝ := !![1, Real.cos (a - b), Real.sin a;
                                        Real.cos (a - b), 1, Real.sin b;
                                        Real.sin a, Real.sin b, 1]
  Matrix.det M = 0 := by sorry

end NUMINAMATH_CALUDE_det_trig_matrix_zero_l948_94871


namespace NUMINAMATH_CALUDE_cost_of_six_books_cost_of_six_books_proof_l948_94880

/-- Given that two identical books cost $36, prove that six of these books cost $108. -/
theorem cost_of_six_books : ℝ → Prop :=
  fun (cost_of_two_books : ℝ) =>
    cost_of_two_books = 36 →
    6 * (cost_of_two_books / 2) = 108

-- The proof goes here
theorem cost_of_six_books_proof : cost_of_six_books 36 := by
  sorry

end NUMINAMATH_CALUDE_cost_of_six_books_cost_of_six_books_proof_l948_94880


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l948_94817

theorem arithmetic_sequence_sum (a : ℕ → ℝ) :
  (∀ n, a (n + 1) = a n + 1) →  -- arithmetic sequence with common difference 1
  (a 2 + a 4 + a 6 = 9) →       -- given condition
  (a 5 + a 7 + a 9 = 18) :=     -- conclusion to prove
by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l948_94817


namespace NUMINAMATH_CALUDE_derivative_of_complex_function_l948_94895

/-- The derivative of ln(4x - 1 + √(16x^2 - 8x + 2)) - √(16x^2 - 8x + 2) * arctan(4x - 1) -/
theorem derivative_of_complex_function (x : ℝ) 
  (h1 : 16 * x^2 - 8 * x + 2 ≥ 0) 
  (h2 : 4 * x - 1 + Real.sqrt (16 * x^2 - 8 * x + 2) > 0) :
  deriv (fun x => Real.log (4 * x - 1 + Real.sqrt (16 * x^2 - 8 * x + 2)) - 
    Real.sqrt (16 * x^2 - 8 * x + 2) * Real.arctan (4 * x - 1)) x = 
  (4 * (1 - 4 * x) / Real.sqrt (16 * x^2 - 8 * x + 2)) * Real.arctan (4 * x - 1) := by
  sorry

end NUMINAMATH_CALUDE_derivative_of_complex_function_l948_94895


namespace NUMINAMATH_CALUDE_currant_yield_increase_l948_94813

theorem currant_yield_increase (initial_yield_per_bush : ℝ) : 
  let total_yield := 15 * initial_yield_per_bush
  let new_yield_per_bush := total_yield / 12
  (new_yield_per_bush - initial_yield_per_bush) / initial_yield_per_bush * 100 = 25 := by
sorry

end NUMINAMATH_CALUDE_currant_yield_increase_l948_94813


namespace NUMINAMATH_CALUDE_subtract_negative_self_l948_94808

theorem subtract_negative_self (a : ℤ) : -a - (-a) = 0 := by sorry

end NUMINAMATH_CALUDE_subtract_negative_self_l948_94808


namespace NUMINAMATH_CALUDE_triangle_not_isosceles_l948_94892

/-- A triangle with sides a, b, c is not isosceles if a, b, c are distinct -/
theorem triangle_not_isosceles (a b c : ℝ) (h₁ : a > 0) (h₂ : b > 0) (h₃ : c > 0)
  (h₄ : a ≠ b) (h₅ : b ≠ c) (h₆ : a ≠ c)
  (h₇ : a + b > c) (h₈ : b + c > a) (h₉ : a + c > b) :
  ¬(a = b ∨ b = c ∨ a = c) := by
  sorry

end NUMINAMATH_CALUDE_triangle_not_isosceles_l948_94892


namespace NUMINAMATH_CALUDE_max_tshirts_purchased_l948_94899

def tshirt_cost : ℚ := 915 / 100
def total_spent : ℚ := 201

theorem max_tshirts_purchased : 
  ⌊total_spent / tshirt_cost⌋ = 21 := by sorry

end NUMINAMATH_CALUDE_max_tshirts_purchased_l948_94899


namespace NUMINAMATH_CALUDE_tan_thirty_degrees_l948_94878

theorem tan_thirty_degrees : Real.tan (30 * π / 180) = Real.sqrt 3 / 3 := by
  sorry

end NUMINAMATH_CALUDE_tan_thirty_degrees_l948_94878


namespace NUMINAMATH_CALUDE_peanut_butter_price_increase_peanut_butter_problem_l948_94875

/-- Calculates the new average price of returned peanut butter cans after a price increase -/
theorem peanut_butter_price_increase (initial_avg_price : ℚ) (num_cans : ℕ) 
  (price_increase : ℚ) (num_returned : ℕ) (remaining_avg_price : ℚ) : ℚ :=
  let total_initial_cost := initial_avg_price * num_cans
  let new_price_per_can := initial_avg_price * (1 + price_increase)
  let total_new_cost := new_price_per_can * num_cans
  let num_remaining := num_cans - num_returned
  let total_remaining_cost := remaining_avg_price * num_remaining
  let total_returned_cost := total_new_cost - total_remaining_cost
  let new_avg_returned_price := total_returned_cost / num_returned
  new_avg_returned_price

/-- The new average price of the two returned peanut butter cans is 65.925 cents -/
theorem peanut_butter_problem : 
  peanut_butter_price_increase (36.5 / 100) 6 (15 / 100) 2 (30 / 100) = 65925 / 100000 := by
  sorry

end NUMINAMATH_CALUDE_peanut_butter_price_increase_peanut_butter_problem_l948_94875


namespace NUMINAMATH_CALUDE_probability_sum_seven_is_one_sixth_l948_94837

/-- The number of possible outcomes for each die -/
def dice_outcomes : ℕ := 6

/-- The total number of possible outcomes when throwing two dice -/
def total_outcomes : ℕ := dice_outcomes * dice_outcomes

/-- The number of ways to get a sum of 7 with two dice -/
def favorable_outcomes : ℕ := 6

/-- The probability of getting a sum of 7 when throwing two fair dice -/
def probability_sum_seven : ℚ := favorable_outcomes / total_outcomes

theorem probability_sum_seven_is_one_sixth :
  probability_sum_seven = 1 / 6 := by sorry

end NUMINAMATH_CALUDE_probability_sum_seven_is_one_sixth_l948_94837


namespace NUMINAMATH_CALUDE_f_of_negative_sqrt_three_equals_four_l948_94843

-- Define the function f
def f (x : ℝ) : ℝ := x^2 + 1

-- State the theorem
theorem f_of_negative_sqrt_three_equals_four :
  (∀ x, f (Real.tan x) = 1 / (Real.cos x)^2) →
  f (-Real.sqrt 3) = 4 := by
sorry

end NUMINAMATH_CALUDE_f_of_negative_sqrt_three_equals_four_l948_94843


namespace NUMINAMATH_CALUDE_correct_change_amount_and_composition_l948_94820

def initial_money : ℚ := 20.40
def avocado_prices : List ℚ := [1.50, 2.25, 3.00]
def water_price : ℚ := 1.75
def water_quantity : ℕ := 2
def apple_price : ℚ := 0.75
def apple_quantity : ℕ := 4

def total_cost : ℚ := (List.sum avocado_prices) + (water_price * water_quantity) + (apple_price * apple_quantity)

def change : ℚ := initial_money - total_cost

theorem correct_change_amount_and_composition :
  change = 7.15 ∧
  ∃ (five_dollar : ℕ) (one_dollar : ℕ) (dime : ℕ) (nickel : ℕ),
    five_dollar = 1 ∧
    one_dollar = 2 ∧
    dime = 1 ∧
    nickel = 1 ∧
    5 * five_dollar + one_dollar + 0.1 * dime + 0.05 * nickel = change :=
by sorry

end NUMINAMATH_CALUDE_correct_change_amount_and_composition_l948_94820


namespace NUMINAMATH_CALUDE_power_product_equals_two_l948_94882

theorem power_product_equals_two :
  (-1/2)^2022 * 2^2023 = 2 := by sorry

end NUMINAMATH_CALUDE_power_product_equals_two_l948_94882


namespace NUMINAMATH_CALUDE_model_n_time_proof_l948_94848

/-- Represents the time (in minutes) taken by a model N computer to complete the task -/
def model_n_time : ℝ := 12

/-- Represents the time (in minutes) taken by a model M computer to complete the task -/
def model_m_time : ℝ := 24

/-- Represents the number of model M computers used -/
def num_model_m : ℕ := 8

/-- Represents the total time (in minutes) taken by both models working together -/
def total_time : ℝ := 1

theorem model_n_time_proof :
  (num_model_m : ℝ) / model_m_time + (num_model_m : ℝ) / model_n_time = 1 / total_time :=
sorry

end NUMINAMATH_CALUDE_model_n_time_proof_l948_94848


namespace NUMINAMATH_CALUDE_largest_non_representable_integer_l948_94891

theorem largest_non_representable_integer 
  (a b c : ℕ+) 
  (h1 : Nat.gcd a b = 1) 
  (h2 : Nat.gcd b c = 1) 
  (h3 : Nat.gcd c a = 1) :
  ∀ n : ℕ, n > 2*a*b*c - a*b - b*c - c*a → 
  ∃ (x y z : ℕ), n = b*c*x + c*a*y + a*b*z ∧
  ¬∃ (x y z : ℕ), 2*a*b*c - a*b - b*c - c*a = b*c*x + c*a*y + a*b*z :=
sorry

end NUMINAMATH_CALUDE_largest_non_representable_integer_l948_94891


namespace NUMINAMATH_CALUDE_johns_earnings_l948_94844

theorem johns_earnings (new_earnings : ℝ) (increase_percentage : ℝ) 
  (h1 : new_earnings = 55) 
  (h2 : increase_percentage = 37.5) : 
  ∃ original_earnings : ℝ, 
    original_earnings * (1 + increase_percentage / 100) = new_earnings ∧ 
    original_earnings = 40 := by
  sorry

end NUMINAMATH_CALUDE_johns_earnings_l948_94844


namespace NUMINAMATH_CALUDE_specific_ellipse_area_l948_94815

/-- An ellipse with given properties --/
structure Ellipse where
  major_axis_endpoint1 : ℝ × ℝ
  major_axis_endpoint2 : ℝ × ℝ
  point_on_ellipse : ℝ × ℝ

/-- The area of an ellipse with the given properties --/
def ellipse_area (e : Ellipse) : ℝ := sorry

/-- Theorem stating that the area of the specific ellipse is 50π --/
theorem specific_ellipse_area :
  let e : Ellipse := {
    major_axis_endpoint1 := (-5, 2),
    major_axis_endpoint2 := (15, 2),
    point_on_ellipse := (11, 6)
  }
  ellipse_area e = 50 * Real.pi := by sorry

end NUMINAMATH_CALUDE_specific_ellipse_area_l948_94815


namespace NUMINAMATH_CALUDE_sequence_problem_l948_94839

def arithmetic_sequence (a b c d : ℝ) : Prop :=
  (b - a = c - b) ∧ (c - b = d - c)

def geometric_sequence (a b c d e : ℝ) : Prop :=
  (b / a = c / b) ∧ (c / b = d / c) ∧ (d / c = e / d)

theorem sequence_problem (a₁ a₂ b₁ b₂ b₃ : ℝ) :
  arithmetic_sequence (-7) a₁ a₂ (-1) →
  geometric_sequence (-4) b₁ b₂ b₃ (-1) →
  (a₂ - a₁) / b₂ = -1 := by
  sorry

end NUMINAMATH_CALUDE_sequence_problem_l948_94839


namespace NUMINAMATH_CALUDE_rectangle_circle_overlap_area_l948_94870

/-- The area of overlap between a rectangle and a circle with shared center -/
theorem rectangle_circle_overlap_area 
  (rect_length : ℝ) 
  (rect_width : ℝ) 
  (circle_radius : ℝ) 
  (h_length : rect_length = 10) 
  (h_width : rect_width = 4) 
  (h_radius : circle_radius = 3) : 
  ∃ (overlap_area : ℝ), 
    overlap_area = 9 * Real.pi - 8 * Real.sqrt 5 + 12 :=
sorry

end NUMINAMATH_CALUDE_rectangle_circle_overlap_area_l948_94870


namespace NUMINAMATH_CALUDE_phil_initial_books_l948_94830

def initial_book_count (pages_per_book : ℕ) (books_lost : ℕ) (pages_left : ℕ) : ℕ :=
  (pages_left / pages_per_book) + books_lost

theorem phil_initial_books :
  initial_book_count 100 2 800 = 10 :=
by
  sorry

end NUMINAMATH_CALUDE_phil_initial_books_l948_94830


namespace NUMINAMATH_CALUDE_geometric_sequence_sum_l948_94841

/-- Given a geometric sequence with sum of first n terms Sn = k · 3^n + 1, k = -1 -/
theorem geometric_sequence_sum (n : ℕ) (k : ℝ) :
  (∀ n, ∃ Sn : ℝ, Sn = k * 3^n + 1) →
  (∃ a : ℕ → ℝ, ∀ i j, i < j → a i * a j = (a i)^2) →
  k = -1 :=
sorry

end NUMINAMATH_CALUDE_geometric_sequence_sum_l948_94841


namespace NUMINAMATH_CALUDE_total_money_is_900_l948_94803

/-- The amount of money Sam has -/
def sam_money : ℕ := 200

/-- The amount of money Billy has -/
def billy_money : ℕ := 3 * sam_money - 150

/-- The amount of money Lila has -/
def lila_money : ℕ := billy_money - sam_money

/-- The total amount of money they have together -/
def total_money : ℕ := sam_money + billy_money + lila_money

theorem total_money_is_900 : total_money = 900 := by
  sorry

end NUMINAMATH_CALUDE_total_money_is_900_l948_94803


namespace NUMINAMATH_CALUDE_geometric_sequence_sum_l948_94852

theorem geometric_sequence_sum (a : ℕ → ℝ) (r : ℝ) :
  (∀ n, a n > 0) →  -- Each term is positive
  (∀ n, a (n + 1) = a n * r) →  -- Geometric sequence definition
  (a 0 + a 1 = 6) →  -- Sum of first two terms is 6
  (a 0 + a 1 + a 2 + a 3 + a 4 + a 5 = 126) →  -- Sum of first six terms is 126
  (a 0 + a 1 + a 2 + a 3 = 30) :=  -- Sum of first four terms is 30
by sorry

end NUMINAMATH_CALUDE_geometric_sequence_sum_l948_94852


namespace NUMINAMATH_CALUDE_cube_surface_area_l948_94834

theorem cube_surface_area (volume : ℝ) (side : ℝ) (surface_area : ℝ) : 
  volume = 512 → 
  volume = side^3 → 
  surface_area = 6 * side^2 → 
  surface_area = 384 := by
  sorry

end NUMINAMATH_CALUDE_cube_surface_area_l948_94834


namespace NUMINAMATH_CALUDE_seashells_count_l948_94884

theorem seashells_count (joan_shells jessica_shells : ℕ) 
  (h1 : joan_shells = 6) 
  (h2 : jessica_shells = 8) : 
  joan_shells + jessica_shells = 14 := by
sorry

end NUMINAMATH_CALUDE_seashells_count_l948_94884


namespace NUMINAMATH_CALUDE_sum_of_squares_of_roots_l948_94850

theorem sum_of_squares_of_roots (x₁ x₂ : ℝ) : 
  (10 * x₁^2 - 14 * x₁ - 24 = 0) → 
  (10 * x₂^2 - 14 * x₂ - 24 = 0) → 
  (x₁ ≠ x₂) →
  x₁^2 + x₂^2 = 169/25 := by
sorry

end NUMINAMATH_CALUDE_sum_of_squares_of_roots_l948_94850


namespace NUMINAMATH_CALUDE_break_room_tables_l948_94889

/-- The number of people each table can seat -/
def seating_capacity_per_table : ℕ := 8

/-- The total seating capacity of the break room -/
def total_seating_capacity : ℕ := 32

/-- The number of tables in the break room -/
def number_of_tables : ℕ := total_seating_capacity / seating_capacity_per_table

theorem break_room_tables : number_of_tables = 4 := by
  sorry

end NUMINAMATH_CALUDE_break_room_tables_l948_94889


namespace NUMINAMATH_CALUDE_certain_number_problem_l948_94851

theorem certain_number_problem (x y : ℝ) (h1 : 0.25 * x = 0.15 * y - 15) (h2 : x = 840) : y = 1500 := by
  sorry

end NUMINAMATH_CALUDE_certain_number_problem_l948_94851


namespace NUMINAMATH_CALUDE_centroid_perpendicular_distance_l948_94849

-- Define a triangle
structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ

-- Define a line
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

-- Define the perpendicular distance from a point to a line
def perpendicularDistance (p : ℝ × ℝ) (l : Line) : ℝ :=
  sorry

-- Define the centroid of a triangle
def centroid (t : Triangle) : ℝ × ℝ :=
  sorry

-- Theorem statement
theorem centroid_perpendicular_distance (t : Triangle) (l : Line) :
  perpendicularDistance (centroid t) l =
    (perpendicularDistance t.A l + perpendicularDistance t.B l + perpendicularDistance t.C l) / 3 :=
  sorry

end NUMINAMATH_CALUDE_centroid_perpendicular_distance_l948_94849


namespace NUMINAMATH_CALUDE_simplify_expression_l948_94805

theorem simplify_expression (x y : ℝ) :
  3 * x - 5 * (2 - x + y) + 4 * (1 - x - 2 * y) - 6 * (2 + 3 * x - y) = -14 * x - 7 * y - 18 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l948_94805


namespace NUMINAMATH_CALUDE_fiona_cleaning_time_proof_l948_94876

/-- Calculates Fiona's cleaning time in minutes given the total cleaning time and Lilly's fraction of work -/
def fiona_cleaning_time (total_time : ℝ) (lilly_fraction : ℝ) : ℝ :=
  (total_time - lilly_fraction * total_time) * 60

/-- Theorem: Given a total cleaning time of 8 hours and Lilly spending 1/4 of the total time, 
    Fiona's cleaning time in minutes is equal to 360. -/
theorem fiona_cleaning_time_proof :
  fiona_cleaning_time 8 (1/4) = 360 := by
  sorry

end NUMINAMATH_CALUDE_fiona_cleaning_time_proof_l948_94876


namespace NUMINAMATH_CALUDE_solution_to_equation_l948_94810

theorem solution_to_equation (x : ℝ) : 
  (Real.sqrt ((3 + 2 * Real.sqrt 2) ^ x) + Real.sqrt ((3 - 2 * Real.sqrt 2) ^ x) = 6) ↔ 
  (x = 2 ∨ x = -2) := by
sorry

end NUMINAMATH_CALUDE_solution_to_equation_l948_94810


namespace NUMINAMATH_CALUDE_regression_line_not_exact_l948_94818

-- Define the regression line equation
def regression_line (x : ℝ) : ℝ := 0.5 * x - 85

-- Define the specific x value
def x_value : ℝ := 200

-- Theorem statement
theorem regression_line_not_exact (ε : ℝ) (h : ε > 0) :
  ∃ y : ℝ, y ≠ 15 ∧ |y - regression_line x_value| < ε :=
sorry

end NUMINAMATH_CALUDE_regression_line_not_exact_l948_94818


namespace NUMINAMATH_CALUDE_no_integer_solutions_for_equation_l948_94821

theorem no_integer_solutions_for_equation : 
  ¬ ∃ (x y z : ℤ), 4 * x^2 + 77 * y^2 = 487 * z^2 := by
  sorry

end NUMINAMATH_CALUDE_no_integer_solutions_for_equation_l948_94821


namespace NUMINAMATH_CALUDE_imaginary_unit_power_l948_94832

-- Define the imaginary unit i
noncomputable def i : ℂ := Complex.I

-- Theorem statement
theorem imaginary_unit_power (n : ℤ) : i^n = 1 ↔ 4 ∣ n :=
sorry

end NUMINAMATH_CALUDE_imaginary_unit_power_l948_94832


namespace NUMINAMATH_CALUDE_min_value_inequality_l948_94873

def f (x : ℝ) : ℝ := |3*x - 1| + |x + 1|

def g (x : ℝ) : ℝ := f x + 2*|x + 1|

theorem min_value_inequality (a b : ℝ) 
  (h1 : ∀ x, g x ≥ a^2 + b^2) 
  (h2 : ∃ x, g x = a^2 + b^2) : 
  1 / (a^2 + 1) + 4 / (b^2 + 1) ≥ 3/2 := by
  sorry

end NUMINAMATH_CALUDE_min_value_inequality_l948_94873


namespace NUMINAMATH_CALUDE_two_std_dev_below_mean_l948_94853

/-- For a normal distribution with mean 10.5 and standard deviation 1,
    the value that is exactly 2 standard deviations less than the mean is 8.5. -/
theorem two_std_dev_below_mean (μ σ : ℝ) (hμ : μ = 10.5) (hσ : σ = 1) :
  μ - 2 * σ = 8.5 := by
  sorry

end NUMINAMATH_CALUDE_two_std_dev_below_mean_l948_94853


namespace NUMINAMATH_CALUDE_correct_evolution_process_l948_94868

-- Define the types of population growth models
inductive PopulationGrowthModel
| Primitive
| Traditional
| Modern

-- Define the characteristics of each model
structure ModelCharacteristics where
  productiveForces : ℕ
  disasterResistance : ℕ
  birthRate : ℕ
  deathRate : ℕ
  economicLevel : ℕ
  socialSecurity : ℕ

-- Define the evolution process
def evolutionProcess : List PopulationGrowthModel :=
  [PopulationGrowthModel.Primitive, PopulationGrowthModel.Traditional, PopulationGrowthModel.Modern]

-- Define the characteristics for each model
def primitiveCharacteristics : ModelCharacteristics :=
  { productiveForces := 1, disasterResistance := 1, birthRate := 3, deathRate := 3,
    economicLevel := 1, socialSecurity := 1 }

def traditionalCharacteristics : ModelCharacteristics :=
  { productiveForces := 2, disasterResistance := 2, birthRate := 3, deathRate := 1,
    economicLevel := 2, socialSecurity := 2 }

def modernCharacteristics : ModelCharacteristics :=
  { productiveForces := 3, disasterResistance := 3, birthRate := 1, deathRate := 1,
    economicLevel := 3, socialSecurity := 3 }

-- Theorem stating that the evolution process is correct
theorem correct_evolution_process :
  evolutionProcess = [PopulationGrowthModel.Primitive, PopulationGrowthModel.Traditional, PopulationGrowthModel.Modern] :=
by sorry

end NUMINAMATH_CALUDE_correct_evolution_process_l948_94868


namespace NUMINAMATH_CALUDE_rearrangements_of_13358_l948_94802

/-- The number of different five-digit numbers that can be formed by rearranging the digits in 13358 -/
def rearrangements : ℕ :=
  Nat.factorial 5 / (Nat.factorial 2 * Nat.factorial 1 * Nat.factorial 1 * Nat.factorial 1)

/-- Theorem stating that the number of rearrangements is 60 -/
theorem rearrangements_of_13358 : rearrangements = 60 := by
  sorry

end NUMINAMATH_CALUDE_rearrangements_of_13358_l948_94802


namespace NUMINAMATH_CALUDE_doras_stickers_solve_l948_94811

/-- The number of packs of stickers Dora gets -/
def doras_stickers (allowance : ℕ) (card_cost : ℕ) (sticker_box_cost : ℕ) : ℕ :=
  let total_money := 2 * allowance
  let remaining_money := total_money - card_cost
  let boxes_bought := remaining_money / sticker_box_cost
  boxes_bought / 2

theorem doras_stickers_solve :
  doras_stickers 9 10 2 = 2 := by
  sorry

#eval doras_stickers 9 10 2

end NUMINAMATH_CALUDE_doras_stickers_solve_l948_94811


namespace NUMINAMATH_CALUDE_trivia_team_score_l948_94866

theorem trivia_team_score (total_members : ℕ) (absent_members : ℕ) (total_points : ℕ) 
  (h1 : total_members = 15)
  (h2 : absent_members = 6)
  (h3 : total_points = 27) :
  total_points / (total_members - absent_members) = 3 := by
  sorry

end NUMINAMATH_CALUDE_trivia_team_score_l948_94866


namespace NUMINAMATH_CALUDE_quadratic_equation_solution_l948_94801

theorem quadratic_equation_solution (a : ℝ) : 
  ((-1)^2 - 2*(-1) + a = 0) → 
  (3^2 - 2*3 + a = 0) ∧ 
  (∀ x : ℝ, x^2 - 2*x + a = 0 → (x = -1 ∨ x = 3)) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equation_solution_l948_94801


namespace NUMINAMATH_CALUDE_average_speed_three_segments_l948_94823

/-- Calculate the average speed of a journey with three segments -/
theorem average_speed_three_segments 
  (d₁ d₂ d₃ : ℝ) 
  (v₁ v₂ v₃ : ℝ) 
  (h₁ : d₁ = 2) 
  (h₂ : d₂ = 4) 
  (h₃ : d₃ = 6) 
  (h₄ : v₁ = 3) 
  (h₅ : v₂ = 5) 
  (h₆ : v₃ = 12) :
  let total_distance := d₁ + d₂ + d₃
  let total_time := d₁ / v₁ + d₂ / v₂ + d₃ / v₃
  total_distance / total_time = 360 / 59 := by
sorry

end NUMINAMATH_CALUDE_average_speed_three_segments_l948_94823


namespace NUMINAMATH_CALUDE_smallest_composite_no_small_factors_l948_94854

def is_composite (n : ℕ) : Prop := ∃ a b, 1 < a ∧ 1 < b ∧ n = a * b

def has_no_prime_factors_less_than_20 (n : ℕ) : Prop :=
  ∀ p, p < 20 → p.Prime → ¬(p ∣ n)

theorem smallest_composite_no_small_factors :
  (is_composite 667) ∧
  (has_no_prime_factors_less_than_20 667) ∧
  (∀ m : ℕ, m < 667 →
    ¬(is_composite m ∧ has_no_prime_factors_less_than_20 m)) :=
sorry

end NUMINAMATH_CALUDE_smallest_composite_no_small_factors_l948_94854


namespace NUMINAMATH_CALUDE_decimal_to_binary_13_l948_94896

theorem decimal_to_binary_13 : (13 : ℕ) = 
  (1 * 2^3 + 1 * 2^2 + 0 * 2^1 + 1 * 2^0 : ℕ) := by
  sorry

end NUMINAMATH_CALUDE_decimal_to_binary_13_l948_94896


namespace NUMINAMATH_CALUDE_michael_twice_jacob_age_l948_94807

/-- Given that:
    - Jacob is 12 years younger than Michael
    - Jacob will be 13 years old in 4 years
    - At some point in the future, Michael will be twice as old as Jacob
    This theorem proves that Michael will be twice as old as Jacob in 3 years. -/
theorem michael_twice_jacob_age (jacob_age : ℕ) (michael_age : ℕ) (years_until_twice : ℕ) :
  michael_age = jacob_age + 12 →
  jacob_age + 4 = 13 →
  michael_age + years_until_twice = 2 * (jacob_age + years_until_twice) →
  years_until_twice = 3 := by
  sorry

end NUMINAMATH_CALUDE_michael_twice_jacob_age_l948_94807


namespace NUMINAMATH_CALUDE_special_geometric_sequence_ratio_l948_94898

/-- A geometric sequence with positive terms where a₁, (1/2)a₃, 2a₂ form an arithmetic sequence -/
structure SpecialGeometricSequence where
  a : ℕ → ℝ
  positive : ∀ n, a n > 0
  geometric : ∀ n, a (n + 1) / a n = a 2 / a 1
  arithmetic : a 1 + 2 * a 2 = a 3

/-- The ratio of (a₁₁ + a₁₂) to (a₉ + a₁₀) equals 3 + 2√2 -/
theorem special_geometric_sequence_ratio 
  (seq : SpecialGeometricSequence) :
  (seq.a 11 + seq.a 12) / (seq.a 9 + seq.a 10) = 3 + 2 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_special_geometric_sequence_ratio_l948_94898


namespace NUMINAMATH_CALUDE_dougs_age_l948_94877

/-- Given the ages of Qaddama, Jack, and Doug, prove Doug's age --/
theorem dougs_age (qaddama jack doug : ℕ) 
  (h1 : qaddama = jack + 6)
  (h2 : doug = jack + 3)
  (h3 : qaddama = 19) : 
  doug = 16 := by
  sorry

end NUMINAMATH_CALUDE_dougs_age_l948_94877


namespace NUMINAMATH_CALUDE_water_depth_l948_94838

/-- The depth of water given heights of two people -/
theorem water_depth (ron_height dean_height water_depth : ℕ) : 
  ron_height = 13 →
  dean_height = ron_height + 4 →
  water_depth = 15 * dean_height →
  water_depth = 255 :=
by
  sorry

#check water_depth

end NUMINAMATH_CALUDE_water_depth_l948_94838


namespace NUMINAMATH_CALUDE_downstream_speed_l948_94858

/-- Represents the speed of a man rowing in different conditions -/
structure RowingSpeed where
  upstream : ℝ
  stillWater : ℝ
  downstream : ℝ

/-- Theorem stating the downstream speed of a man given his upstream and still water speeds -/
theorem downstream_speed (s : RowingSpeed) (h1 : s.upstream = 25) (h2 : s.stillWater = 40) :
  s.downstream = 55 := by
  sorry

#check downstream_speed

end NUMINAMATH_CALUDE_downstream_speed_l948_94858


namespace NUMINAMATH_CALUDE_unique_number_property_l948_94846

theorem unique_number_property : ∃! x : ℝ, x / 2 = x - 2 := by sorry

end NUMINAMATH_CALUDE_unique_number_property_l948_94846


namespace NUMINAMATH_CALUDE_rectangle_diagonal_pythagorean_l948_94867

/-- A rectangle with side lengths a and b, and diagonal c -/
structure Rectangle where
  a : ℝ
  b : ℝ
  c : ℝ
  ha : a > 0
  hb : b > 0
  hc : c > 0

/-- The Pythagorean theorem holds for the rectangle's diagonal -/
theorem rectangle_diagonal_pythagorean (rect : Rectangle) : 
  rect.c^2 = rect.a^2 + rect.b^2 := by
  sorry

#check rectangle_diagonal_pythagorean

end NUMINAMATH_CALUDE_rectangle_diagonal_pythagorean_l948_94867


namespace NUMINAMATH_CALUDE_pentagon_rectangle_ratio_l948_94893

/-- Given a regular pentagon and a rectangle with the same perimeter and the rectangle's length
    being twice its width, the ratio of the pentagon's side length to the rectangle's width is 6/5 -/
theorem pentagon_rectangle_ratio (p w l : ℝ) : 
  p > 0 → w > 0 → l > 0 →
  5 * p = 30 →  -- Pentagon perimeter
  2 * w + 2 * l = 30 →  -- Rectangle perimeter
  l = 2 * w →  -- Rectangle length is twice the width
  p / w = 6 / 5 := by
  sorry

end NUMINAMATH_CALUDE_pentagon_rectangle_ratio_l948_94893


namespace NUMINAMATH_CALUDE_facebook_employees_l948_94822

-- Define the given constants
def annual_earnings : ℝ := 5000000
def bonus_percentage : ℝ := 0.25
def non_mother_women : ℕ := 1200
def bonus_per_mother : ℝ := 1250

-- Define the theorem
theorem facebook_employees : 
  ∃ (total_employees : ℕ),
    -- One-third of employees are men
    (total_employees : ℝ) / 3 = (total_employees - (2 * total_employees / 3) : ℝ) ∧
    -- The number of female mother employees
    (2 * total_employees / 3 : ℝ) - non_mother_women = 
      bonus_percentage * annual_earnings / bonus_per_mother ∧
    -- The total number of employees is 3300
    total_employees = 3300 := by
  sorry

end NUMINAMATH_CALUDE_facebook_employees_l948_94822


namespace NUMINAMATH_CALUDE_coffee_stock_proof_l948_94812

/-- Represents the initial stock of coffee in pounds -/
def initial_stock : ℝ := 400

/-- Represents the percentage of decaffeinated coffee in the initial stock -/
def initial_decaf_percent : ℝ := 0.25

/-- Represents the additional coffee purchase in pounds -/
def additional_purchase : ℝ := 100

/-- Represents the percentage of decaffeinated coffee in the additional purchase -/
def additional_decaf_percent : ℝ := 0.60

/-- Represents the final percentage of decaffeinated coffee in the total stock -/
def final_decaf_percent : ℝ := 0.32

theorem coffee_stock_proof :
  initial_stock * initial_decaf_percent + additional_purchase * additional_decaf_percent =
  final_decaf_percent * (initial_stock + additional_purchase) :=
by sorry

end NUMINAMATH_CALUDE_coffee_stock_proof_l948_94812


namespace NUMINAMATH_CALUDE_intersection_right_triangle_l948_94826

/-- Given a line and a circle that intersect, and the triangle formed by the
    intersection points and the circle's center is right-angled, prove the value of a. -/
theorem intersection_right_triangle (a : ℝ) : 
  -- Line equation
  (∃ x y : ℝ, a * x - y + 6 = 0) →
  -- Circle equation
  (∃ x y : ℝ, (x + 1)^2 + (y - a)^2 = 16) →
  -- Circle center
  let C : ℝ × ℝ := (-1, a)
  -- Intersection points exist
  (∃ A B : ℝ × ℝ, A ≠ B ∧ 
    (a * A.1 - A.2 + 6 = 0) ∧ ((A.1 + 1)^2 + (A.2 - a)^2 = 16) ∧
    (a * B.1 - B.2 + 6 = 0) ∧ ((B.1 + 1)^2 + (B.2 - a)^2 = 16)) →
  -- Triangle ABC is right-angled
  (∃ A B : ℝ × ℝ, (A - C) • (B - C) = 0) →
  -- Conclusion
  a = 3 - Real.sqrt 2 :=
by sorry

end NUMINAMATH_CALUDE_intersection_right_triangle_l948_94826


namespace NUMINAMATH_CALUDE_dishonest_dealer_profit_l948_94840

/-- The profit percentage of a dishonest dealer who uses a weight of 800 grams instead of 1000 grams per kg. -/
theorem dishonest_dealer_profit (actual_weight : ℝ) (full_weight : ℝ) :
  actual_weight = 800 →
  full_weight = 1000 →
  (full_weight - actual_weight) / full_weight * 100 = 20 := by
  sorry

end NUMINAMATH_CALUDE_dishonest_dealer_profit_l948_94840


namespace NUMINAMATH_CALUDE_sample_size_is_80_l948_94890

/-- Represents the ratio of product models A, B, and C -/
def productRatio : Fin 3 → ℕ
  | 0 => 2  -- Model A
  | 1 => 3  -- Model B
  | 2 => 5  -- Model C
  | _ => 0  -- This case should never occur due to Fin 3

/-- Calculates the total ratio sum -/
def totalRatio : ℕ := (productRatio 0) + (productRatio 1) + (productRatio 2)

/-- Represents the number of units of model A in the sample -/
def modelAUnits : ℕ := 16

/-- Theorem stating that the sample size is 80 given the conditions -/
theorem sample_size_is_80 :
  ∃ (n : ℕ), n * (productRatio 0) / totalRatio = modelAUnits ∧ n = 80 :=
sorry

end NUMINAMATH_CALUDE_sample_size_is_80_l948_94890


namespace NUMINAMATH_CALUDE_exam_scores_difference_l948_94828

/-- Given five exam scores with specific properties, prove that the absolute difference between two of them is 18. -/
theorem exam_scores_difference (x y : ℝ) : 
  (x + y + 105 + 109 + 110) / 5 = 108 →
  ((x - 108)^2 + (y - 108)^2 + (105 - 108)^2 + (109 - 108)^2 + (110 - 108)^2) / 5 = 35.2 →
  |x - y| = 18 := by
sorry

end NUMINAMATH_CALUDE_exam_scores_difference_l948_94828


namespace NUMINAMATH_CALUDE_characterize_valid_functions_l948_94874

def is_valid_function (f : ℝ → ℝ) : Prop :=
  ∀ a b c : ℝ, a + f b + f (f c) = 0 →
    f a ^ 3 + b * (f b) ^ 2 + c ^ 2 * f c = 3 * a * b * c

theorem characterize_valid_functions :
  ∀ f : ℝ → ℝ, is_valid_function f →
    (∀ x : ℝ, f x = x) ∨ (∀ x : ℝ, f x = -x) ∨ (∀ x : ℝ, f x = 0) :=
sorry

end NUMINAMATH_CALUDE_characterize_valid_functions_l948_94874


namespace NUMINAMATH_CALUDE_arithmetic_to_geometric_sequence_l948_94835

theorem arithmetic_to_geometric_sequence :
  ∀ (a b c : ℝ),
  (∃ (x : ℝ), a = 3*x ∧ b = 4*x ∧ c = 5*x) →
  (b - a = c - b) →
  ((a + 1) * c = b^2) →
  (a = 15 ∧ b = 20 ∧ c = 25) :=
by
  sorry

end NUMINAMATH_CALUDE_arithmetic_to_geometric_sequence_l948_94835


namespace NUMINAMATH_CALUDE_geometric_series_sum_l948_94856

def geometricSum (a : ℚ) (r : ℚ) (n : ℕ) : ℚ :=
  a * (1 - r^n) / (1 - r)

theorem geometric_series_sum :
  let a : ℚ := 1/4
  let r : ℚ := 1/4
  let n : ℕ := 7
  geometricSum a r n = 4/3 := by
  sorry

end NUMINAMATH_CALUDE_geometric_series_sum_l948_94856


namespace NUMINAMATH_CALUDE_pool_cover_radius_increase_l948_94887

/-- Theorem: When a circular pool cover's circumference increases from 30 inches to 40 inches, 
    the radius increases by 5/π inches. -/
theorem pool_cover_radius_increase (r₁ r₂ : ℝ) : 
  2 * Real.pi * r₁ = 30 → 
  2 * Real.pi * r₂ = 40 → 
  r₂ - r₁ = 5 / Real.pi := by
sorry

end NUMINAMATH_CALUDE_pool_cover_radius_increase_l948_94887


namespace NUMINAMATH_CALUDE_quadratic_equations_solutions_l948_94861

theorem quadratic_equations_solutions :
  (∃ x₁ x₂ : ℝ, x₁ = 2/3 ∧ x₂ = 2 ∧ 3*x₁^2 - 8*x₁ + 4 = 0 ∧ 3*x₂^2 - 8*x₂ + 4 = 0) ∧
  (∃ y₁ y₂ : ℝ, y₁ = 4/3 ∧ y₂ = -2 ∧ (2*y₁ - 1)^2 = (y₁ - 3)^2 ∧ (2*y₂ - 1)^2 = (y₂ - 3)^2) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equations_solutions_l948_94861


namespace NUMINAMATH_CALUDE_bert_profit_is_correct_l948_94879

/-- Represents a product in Bert's shop -/
structure Product where
  price : ℝ
  tax_rate : ℝ

/-- Represents a customer's purchase -/
structure Purchase where
  products : List Product
  discount_rate : ℝ

/-- Calculates Bert's profit given the purchases -/
def calculate_profit (purchases : List Purchase) : ℝ :=
  sorry

/-- The actual purchases made by customers -/
def actual_purchases : List Purchase :=
  [
    { products := [
        { price := 90, tax_rate := 0.1 },
        { price := 50, tax_rate := 0.05 }
      ], 
      discount_rate := 0.1
    },
    { products := [
        { price := 30, tax_rate := 0.12 },
        { price := 20, tax_rate := 0.03 }
      ], 
      discount_rate := 0.15
    },
    { products := [
        { price := 15, tax_rate := 0.09 }
      ], 
      discount_rate := 0
    }
  ]

/-- Bert's profit per item -/
def profit_per_item : ℝ := 10

theorem bert_profit_is_correct : 
  calculate_profit actual_purchases = 50.05 :=
sorry

end NUMINAMATH_CALUDE_bert_profit_is_correct_l948_94879


namespace NUMINAMATH_CALUDE_tea_mixture_price_l948_94855

/-- Given two teas mixed in equal proportions, proves that if one tea costs 74 rupees per kg
    and the mixture costs 69 rupees per kg, then the other tea costs 64 rupees per kg. -/
theorem tea_mixture_price (price_tea2 mixture_price : ℝ) 
  (h1 : price_tea2 = 74)
  (h2 : mixture_price = 69) :
  ∃ (price_tea1 : ℝ), 
    price_tea1 = 64 ∧ 
    (price_tea1 + price_tea2) / 2 = mixture_price :=
by
  sorry

end NUMINAMATH_CALUDE_tea_mixture_price_l948_94855


namespace NUMINAMATH_CALUDE_jason_tom_blue_difference_l948_94819

/-- Represents the number of marbles a person has -/
structure MarbleCount where
  blue : ℕ
  red : ℕ
  green : ℕ
  yellow : ℕ

/-- Calculates the difference in blue marbles between two MarbleCounts -/
def blueDifference (a b : MarbleCount) : ℕ :=
  if a.blue ≥ b.blue then a.blue - b.blue else b.blue - a.blue

theorem jason_tom_blue_difference :
  let jason : MarbleCount := { blue := 44, red := 16, green := 8, yellow := 0 }
  let tom : MarbleCount := { blue := 24, red := 0, green := 7, yellow := 10 }
  blueDifference jason tom = 20 := by
  sorry

end NUMINAMATH_CALUDE_jason_tom_blue_difference_l948_94819


namespace NUMINAMATH_CALUDE_hiker_speed_difference_l948_94872

/-- A hiker's journey over three days -/
def hiker_journey (v : ℝ) : Prop :=
  let day1_distance : ℝ := 18
  let day1_speed : ℝ := 3
  let day1_hours : ℝ := day1_distance / day1_speed
  let day2_hours : ℝ := day1_hours - 1
  let day2_distance : ℝ := day2_hours * v
  let day3_distance : ℝ := 5 * 3
  day1_distance + day2_distance + day3_distance = 53

theorem hiker_speed_difference : ∃ v : ℝ, hiker_journey v ∧ v - 3 = 1 := by
  sorry

#check hiker_speed_difference

end NUMINAMATH_CALUDE_hiker_speed_difference_l948_94872


namespace NUMINAMATH_CALUDE_smallest_number_with_sum_l948_94847

/-- Calculates the sum of all unique permutations of digits in a number -/
def sumOfPermutations (n : ℕ) : ℕ := sorry

/-- Checks if a number is the smallest with a given sum of permutations -/
def isSmallestWithSum (n : ℕ) (sum : ℕ) : Prop :=
  (sumOfPermutations n = sum) ∧ 
  (∀ m : ℕ, m < n → sumOfPermutations m ≠ sum)

/-- The main theorem stating that 47899 is the smallest number 
    whose sum of digit permutations is 4,933,284 -/
theorem smallest_number_with_sum :
  isSmallestWithSum 47899 4933284 := by sorry

end NUMINAMATH_CALUDE_smallest_number_with_sum_l948_94847


namespace NUMINAMATH_CALUDE_fraction_square_sum_l948_94862

theorem fraction_square_sum (a b c d : ℚ) (h : a / b + c / d = 1) :
  (a / b)^2 + c / d = (c / d)^2 + a / b := by sorry

end NUMINAMATH_CALUDE_fraction_square_sum_l948_94862


namespace NUMINAMATH_CALUDE_sum_2_4_6_8_l948_94865

/-- An arithmetic sequence -/
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

/-- The sum of the 3rd and 7th terms is 37 -/
def sum_3_7 (a : ℕ → ℝ) : Prop :=
  a 3 + a 7 = 37

theorem sum_2_4_6_8 (a : ℕ → ℝ) (h1 : arithmetic_sequence a) (h2 : sum_3_7 a) :
  a 2 + a 4 + a 6 + a 8 = 74 := by
  sorry

end NUMINAMATH_CALUDE_sum_2_4_6_8_l948_94865


namespace NUMINAMATH_CALUDE_point_on_y_axis_l948_94845

/-- If a point P(a-3, 2-a) lies on the y-axis, then P = (0, -1) -/
theorem point_on_y_axis (a : ℝ) :
  (a - 3 = 0) →  -- P lies on y-axis (x-coordinate is 0)
  (a - 3, 2 - a) = (0, -1) :=
by
  sorry

end NUMINAMATH_CALUDE_point_on_y_axis_l948_94845


namespace NUMINAMATH_CALUDE_scientific_notation_equality_l948_94833

theorem scientific_notation_equality : ∃ (a : ℝ) (n : ℤ), 
  0.0000023 = a * (10 : ℝ) ^ n ∧ 1 ≤ |a| ∧ |a| < 10 ∧ a = 2.3 ∧ n = -6 := by
  sorry

end NUMINAMATH_CALUDE_scientific_notation_equality_l948_94833


namespace NUMINAMATH_CALUDE_unique_integer_sum_l948_94827

theorem unique_integer_sum (C y M A : ℕ) : 
  C > 0 ∧ y > 0 ∧ M > 0 ∧ A > 0 →
  C ≠ y ∧ C ≠ M ∧ C ≠ A ∧ y ≠ M ∧ y ≠ A ∧ M ≠ A →
  C + y + M + M + A = 11 →
  M = 1 := by
sorry

end NUMINAMATH_CALUDE_unique_integer_sum_l948_94827


namespace NUMINAMATH_CALUDE_different_color_pairs_count_l948_94886

def white_socks : ℕ := 4
def brown_socks : ℕ := 4
def blue_socks : ℕ := 2
def gray_socks : ℕ := 5

def total_socks : ℕ := white_socks + brown_socks + blue_socks + gray_socks

def different_color_pairs : ℕ := 
  white_socks * brown_socks + 
  white_socks * blue_socks + 
  white_socks * gray_socks + 
  brown_socks * blue_socks + 
  brown_socks * gray_socks + 
  blue_socks * gray_socks

theorem different_color_pairs_count : different_color_pairs = 82 := by
  sorry

end NUMINAMATH_CALUDE_different_color_pairs_count_l948_94886


namespace NUMINAMATH_CALUDE_divisible_by_nine_l948_94881

theorem divisible_by_nine (a b : ℤ) : ∃ k : ℤ, (3*a + 2)^2 - (3*b + 2)^2 = 9*k := by
  sorry

end NUMINAMATH_CALUDE_divisible_by_nine_l948_94881


namespace NUMINAMATH_CALUDE_exist_congruent_triangles_l948_94883

/-- Represents a vertex in a regular polygon --/
structure Vertex where
  index : Nat
  color : Bool  -- True for red, False for blue

/-- Represents a regular 21-sided polygon with colored vertices --/
structure Polygon where
  vertices : Finset Vertex
  red_count : Nat
  blue_count : Nat
  h_vertex_count : vertices.card = 21
  h_red_count : red_count = 6
  h_blue_count : blue_count = 7
  h_color_sum : red_count + blue_count = vertices.card

/-- Represents a triangle in the polygon --/
structure Triangle where
  v1 : Vertex
  v2 : Vertex
  v3 : Vertex

/-- Two triangles are congruent if they have the same shape --/
def CongruentTriangles (t1 t2 : Triangle) : Prop :=
  sorry

/-- Main theorem: There exist two congruent triangles of different colors --/
theorem exist_congruent_triangles (p : Polygon) :
  ∃ (t1 t2 : Triangle),
    (t1.v1.color ∧ t1.v2.color ∧ t1.v3.color) ∧
    (¬t2.v1.color ∧ ¬t2.v2.color ∧ ¬t2.v3.color) ∧
    CongruentTriangles t1 t2 := by
  sorry

end NUMINAMATH_CALUDE_exist_congruent_triangles_l948_94883


namespace NUMINAMATH_CALUDE_kitchen_width_proof_l948_94825

/-- Proves that the width of a rectangular kitchen floor is 8 inches, given the specified conditions. -/
theorem kitchen_width_proof (tile_area : ℝ) (kitchen_length : ℝ) (total_tiles : ℕ) 
  (h1 : tile_area = 6)
  (h2 : kitchen_length = 72)
  (h3 : total_tiles = 96) :
  (tile_area * total_tiles) / kitchen_length = 8 := by
  sorry

end NUMINAMATH_CALUDE_kitchen_width_proof_l948_94825


namespace NUMINAMATH_CALUDE_vector_magnitude_problem_l948_94806

/-- Given two vectors a and b in a real inner product space, 
    if |a| = 2, |b| = 3, and |a + b| = √19, then |a - b| = √7. -/
theorem vector_magnitude_problem 
  (V : Type*) [NormedAddCommGroup V] [InnerProductSpace ℝ V] 
  (a b : V) 
  (h1 : ‖a‖ = 2) 
  (h2 : ‖b‖ = 3) 
  (h3 : ‖a + b‖ = Real.sqrt 19) : 
  ‖a - b‖ = Real.sqrt 7 := by
  sorry

end NUMINAMATH_CALUDE_vector_magnitude_problem_l948_94806


namespace NUMINAMATH_CALUDE_additional_marbles_needed_l948_94831

/-- The number of friends James has -/
def num_friends : ℕ := 15

/-- The initial number of marbles James has -/
def initial_marbles : ℕ := 80

/-- The function to calculate the sum of first n natural numbers -/
def sum_first_n (n : ℕ) : ℕ := n * (n + 1) / 2

/-- The theorem stating the number of additional marbles needed -/
theorem additional_marbles_needed : 
  sum_first_n num_friends - initial_marbles = 40 := by
  sorry

end NUMINAMATH_CALUDE_additional_marbles_needed_l948_94831


namespace NUMINAMATH_CALUDE_quadratic_range_l948_94804

/-- The quadratic function f(x) = x^2 - 2x - 3 -/
def f (x : ℝ) := x^2 - 2*x - 3

/-- The theorem states that for x in [-2, 2], the range of f(x) is [-4, 5] -/
theorem quadratic_range :
  ∀ x ∈ Set.Icc (-2 : ℝ) 2,
  ∃ y ∈ Set.Icc (-4 : ℝ) 5,
  f x = y ∧
  (∀ z, f z ∈ Set.Icc (-4 : ℝ) 5) :=
sorry

end NUMINAMATH_CALUDE_quadratic_range_l948_94804


namespace NUMINAMATH_CALUDE_harolds_money_l948_94824

theorem harolds_money (x : ℚ) : 
  (x / 2 + 5) +  -- Ticket and candies
  ((x / 2 - 5) / 2 + 10) +  -- Newspaper
  (((x / 2 - 5) / 2 - 10) / 2) +  -- Bus fare
  15 +  -- Beggar
  5  -- Remaining money
  = x  -- Total initial money
  → x = 210 := by sorry

end NUMINAMATH_CALUDE_harolds_money_l948_94824


namespace NUMINAMATH_CALUDE_firecracker_sales_profit_l948_94814

/-- Electronic firecracker sales model -/
structure FirecrackerSales where
  cost : ℝ
  price : ℝ
  volume : ℝ
  profit : ℝ
  h1 : cost = 80
  h2 : 80 ≤ price ∧ price ≤ 160
  h3 : volume = -2 * price + 320
  h4 : profit = (price - cost) * volume

/-- Theorem about firecracker sales profit -/
theorem firecracker_sales_profit (model : FirecrackerSales) :
  -- 1. Profit function
  model.profit = -2 * model.price^2 + 480 * model.price - 25600 ∧
  -- 2. Maximum profit
  (∃ max_profit : ℝ, max_profit = 3200 ∧
    ∀ p, 80 ≤ p ∧ p ≤ 160 → 
      -2 * p^2 + 480 * p - 25600 ≤ max_profit) ∧
  (∃ max_price : ℝ, max_price = 120 ∧
    -2 * max_price^2 + 480 * max_price - 25600 = 3200) ∧
  -- 3. Profit of 2400 at lower price
  (∃ lower_price : ℝ, lower_price = 100 ∧
    -2 * lower_price^2 + 480 * lower_price - 25600 = 2400 ∧
    ∀ p, 80 ≤ p ∧ p ≤ 160 ∧ p ≠ lower_price ∧
      -2 * p^2 + 480 * p - 25600 = 2400 → p > lower_price) := by
  sorry

end NUMINAMATH_CALUDE_firecracker_sales_profit_l948_94814


namespace NUMINAMATH_CALUDE_smallest_n_for_inequality_l948_94864

theorem smallest_n_for_inequality : ∃ (n : ℕ), (∀ (w x y z : ℝ), (w^2 + x^2 + y^2 + z^2)^2 ≤ n * (w^4 + x^4 + y^4 + z^4)) ∧
  (∀ (m : ℕ), m < n → ∃ (w x y z : ℝ), (w^2 + x^2 + y^2 + z^2)^2 > m * (w^4 + x^4 + y^4 + z^4)) ∧
  n = 4 := by
  sorry

end NUMINAMATH_CALUDE_smallest_n_for_inequality_l948_94864
