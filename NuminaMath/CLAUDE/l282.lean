import Mathlib

namespace NUMINAMATH_CALUDE_dice_sum_not_23_l282_28208

theorem dice_sum_not_23 (a b c d e : ℕ) : 
  a ≥ 1 ∧ a ≤ 6 ∧
  b ≥ 1 ∧ b ≤ 6 ∧
  c ≥ 1 ∧ c ≤ 6 ∧
  d ≥ 1 ∧ d ≤ 6 ∧
  e ≥ 1 ∧ e ≤ 6 ∧
  a * b * c * d * e = 720 →
  a + b + c + d + e ≠ 23 :=
by sorry

end NUMINAMATH_CALUDE_dice_sum_not_23_l282_28208


namespace NUMINAMATH_CALUDE_square_sum_difference_l282_28252

theorem square_sum_difference (a b : ℝ) 
  (h1 : (a + b)^2 = 17) 
  (h2 : (a - b)^2 = 11) : 
  a^2 + b^2 = 14 := by sorry

end NUMINAMATH_CALUDE_square_sum_difference_l282_28252


namespace NUMINAMATH_CALUDE_bill_omelet_time_l282_28291

/-- Calculates the total time Bill spends preparing for and cooking omelets -/
def total_omelet_time (num_omelets : ℕ) (num_peppers : ℕ) (num_onions : ℕ) (num_mushrooms : ℕ) (num_tomatoes : ℕ) 
  (pepper_time : ℕ) (onion_time : ℕ) (mushroom_time : ℕ) (tomato_time : ℕ) (cheese_time : ℕ) (cook_time : ℕ) : ℕ :=
  (num_peppers * pepper_time) + 
  (num_onions * onion_time) + 
  (num_mushrooms * mushroom_time) + 
  (num_tomatoes * tomato_time) + 
  (num_omelets * cheese_time) + 
  (num_omelets * cook_time)

theorem bill_omelet_time : 
  total_omelet_time 10 8 4 6 6 3 4 2 3 1 6 = 140 := by
  sorry

end NUMINAMATH_CALUDE_bill_omelet_time_l282_28291


namespace NUMINAMATH_CALUDE_trisected_right_triangle_product_l282_28206

/-- A right triangle with trisected angle -/
structure TrisectedRightTriangle where
  -- The length of side XY
  xy : ℝ
  -- The length of side YZ
  yz : ℝ
  -- Point P on XZ
  p : ℝ × ℝ
  -- Point Q on XZ
  q : ℝ × ℝ
  -- The angle at Y is trisected
  angle_trisected : Bool
  -- X, P, Q, Z lie on XZ in that order
  point_order : Bool

/-- The main theorem -/
theorem trisected_right_triangle_product (t : TrisectedRightTriangle)
  (h_xy : t.xy = 228)
  (h_yz : t.yz = 2004)
  (h_trisected : t.angle_trisected = true)
  (h_order : t.point_order = true) :
  (Real.sqrt ((t.p.1 - 0)^2 + (t.p.2 - t.yz)^2) + t.yz) *
  (Real.sqrt ((t.q.1 - 0)^2 + (t.q.2 - t.yz)^2) + t.xy) = 1370736 := by
  sorry

end NUMINAMATH_CALUDE_trisected_right_triangle_product_l282_28206


namespace NUMINAMATH_CALUDE_quadratic_points_relationship_l282_28212

-- Define the quadratic function
def f (x : ℝ) : ℝ := 3 * (x + 1)^2 - 8

-- Define the points on the graph
def y₁ : ℝ := f 1
def y₂ : ℝ := f 2
def y₃ : ℝ := f (-2)

-- Theorem statement
theorem quadratic_points_relationship : y₂ > y₁ ∧ y₁ > y₃ := by
  sorry

end NUMINAMATH_CALUDE_quadratic_points_relationship_l282_28212


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_l282_28210

theorem sufficient_not_necessary (x y : ℝ) :
  (((x < 0 ∧ y < 0) → (x + y - 4 < 0)) ∧
   ∃ x y : ℝ, (x + y - 4 < 0) ∧ ¬(x < 0 ∧ y < 0)) :=
by sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_l282_28210


namespace NUMINAMATH_CALUDE_union_of_A_and_B_l282_28217

def A : Set ℕ := {1, 2, 3, 5, 7}
def B : Set ℕ := {3, 4, 5}

theorem union_of_A_and_B : A ∪ B = {1, 2, 3, 4, 5, 7} := by sorry

end NUMINAMATH_CALUDE_union_of_A_and_B_l282_28217


namespace NUMINAMATH_CALUDE_root_difference_implies_k_value_l282_28235

theorem root_difference_implies_k_value (k : ℝ) :
  (∃ r s : ℝ, r^2 + k*r + 12 = 0 ∧ s^2 + k*s + 12 = 0) →
  (∃ r s : ℝ, (r+7)^2 - k*(r+7) + 12 = 0 ∧ (s+7)^2 - k*(s+7) + 12 = 0) →
  (∀ r s : ℝ, r^2 + k*r + 12 = 0 ∧ s^2 + k*s + 12 = 0 →
              (r+7)^2 - k*(r+7) + 12 = 0 ∧ (s+7)^2 - k*(s+7) + 12 = 0) →
  k = 7 :=
by sorry

end NUMINAMATH_CALUDE_root_difference_implies_k_value_l282_28235


namespace NUMINAMATH_CALUDE_cubic_expression_evaluation_l282_28201

theorem cubic_expression_evaluation (x y : ℚ) (hx : x = 3) (hy : y = 4) :
  (x^3 + 3*y^3) / 7 = 219 / 7 := by sorry

end NUMINAMATH_CALUDE_cubic_expression_evaluation_l282_28201


namespace NUMINAMATH_CALUDE_inequality_comparison_l282_28223

theorem inequality_comparison (x y : ℝ) (h : x > y) : -3*x + 5 < -3*y + 5 := by
  sorry

end NUMINAMATH_CALUDE_inequality_comparison_l282_28223


namespace NUMINAMATH_CALUDE_simplify_expression_l282_28202

variables (x y : ℝ)

def A (x y : ℝ) : ℝ := x^2 + 3*x*y + y^2
def B (x y : ℝ) : ℝ := x^2 - 3*x*y + y^2

theorem simplify_expression (x y : ℝ) : 
  A x y - (B x y + 2 * B x y - (A x y + B x y)) = 12 * x * y := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l282_28202


namespace NUMINAMATH_CALUDE_ellipse_property_l282_28211

-- Define the ellipse
def is_on_ellipse (x y : ℝ) : Prop := x^2/36 + y^2/9 = 1

-- Define the foci of the ellipse
def F1 : ℝ × ℝ := (-3, 0)
def F2 : ℝ × ℝ := (3, 0)

-- Define the angle between PF1 and PF2
def angle_F1PF2 (P : ℝ × ℝ) : ℝ := 120

-- Theorem statement
theorem ellipse_property (P : ℝ × ℝ) 
  (h1 : is_on_ellipse P.1 P.2) 
  (h2 : angle_F1PF2 P = 120) : 
  Real.sqrt ((P.1 - F1.1)^2 + (P.2 - F1.2)^2) * 
  Real.sqrt ((P.1 - F2.1)^2 + (P.2 - F2.2)^2) = 36 := by
  sorry

end NUMINAMATH_CALUDE_ellipse_property_l282_28211


namespace NUMINAMATH_CALUDE_lisa_challenge_time_l282_28284

/-- The time remaining for Lisa to complete the hotdog-eating challenge -/
def timeRemaining (totalHotdogs : ℕ) (hotdogsEaten : ℕ) (eatingRate : ℕ) : ℚ :=
  (totalHotdogs - hotdogsEaten : ℚ) / eatingRate

/-- Theorem stating that Lisa has 5 minutes to complete the challenge -/
theorem lisa_challenge_time : 
  timeRemaining 75 20 11 = 5 := by sorry

end NUMINAMATH_CALUDE_lisa_challenge_time_l282_28284


namespace NUMINAMATH_CALUDE_min_length_shared_side_l282_28281

/-- Given two triangles PQR and SQR that share side QR, with PQ = 7, PR = 15, SR = 10, and QS = 25,
    prove that the length of QR is at least 15. -/
theorem min_length_shared_side (PQ PR SR QS : ℝ) (hPQ : PQ = 7) (hPR : PR = 15) (hSR : SR = 10) (hQS : QS = 25) :
  ∃ (QR : ℝ), QR ≥ 15 ∧ QR > PR - PQ ∧ QR > QS - SR :=
by sorry

end NUMINAMATH_CALUDE_min_length_shared_side_l282_28281


namespace NUMINAMATH_CALUDE_keith_pears_l282_28234

theorem keith_pears (jason_pears mike_ate remaining_pears : ℕ) 
  (h1 : jason_pears = 46)
  (h2 : mike_ate = 12)
  (h3 : remaining_pears = 81) :
  ∃ keith_pears : ℕ, jason_pears + keith_pears - mike_ate = remaining_pears ∧ keith_pears = 47 :=
by sorry

end NUMINAMATH_CALUDE_keith_pears_l282_28234


namespace NUMINAMATH_CALUDE_x_limit_properties_l282_28231

noncomputable def x : ℕ → ℝ
  | 0 => Real.sqrt 6
  | n + 1 => x n + 3 * Real.sqrt (x n) + (n + 1 : ℝ) / Real.sqrt (x n)

theorem x_limit_properties :
  (∀ ε > 0, ∃ N, ∀ n ≥ N, |n / x n| < ε) ∧
  (∀ ε > 0, ∃ N, ∀ n ≥ N, |n^2 / x n - 4/9| < ε) := by
  sorry

end NUMINAMATH_CALUDE_x_limit_properties_l282_28231


namespace NUMINAMATH_CALUDE_walking_speed_proof_l282_28299

/-- The walking speed of a man who covers the same distance in 9 hours walking
    and in 3 hours running at 24 kmph. -/
def walking_speed : ℝ := 8

theorem walking_speed_proof :
  walking_speed * 9 = 24 * 3 := by
  sorry

end NUMINAMATH_CALUDE_walking_speed_proof_l282_28299


namespace NUMINAMATH_CALUDE_diameters_intersect_l282_28225

-- Define a convex set in a plane
def ConvexSet (S : Set (Real × Real)) : Prop :=
  ∀ x y : Real × Real, x ∈ S → y ∈ S → ∀ t : Real, 0 ≤ t ∧ t ≤ 1 →
    (t * x.1 + (1 - t) * y.1, t * x.2 + (1 - t) * y.2) ∈ S

-- Define a diameter of a convex set
def Diameter (S : Set (Real × Real)) (d : Set (Real × Real)) : Prop :=
  ConvexSet S ∧ d ⊆ S ∧ ∀ x y : Real × Real, x ∈ S → y ∈ S →
    ∃ a b : Real × Real, a ∈ d ∧ b ∈ d ∧ 
      (a.1 - b.1)^2 + (a.2 - b.2)^2 ≥ (x.1 - y.1)^2 + (x.2 - y.2)^2

-- Theorem statement
theorem diameters_intersect (S : Set (Real × Real)) (d1 d2 : Set (Real × Real)) :
  ConvexSet S → Diameter S d1 → Diameter S d2 → (d1 ∩ d2).Nonempty := by
  sorry

end NUMINAMATH_CALUDE_diameters_intersect_l282_28225


namespace NUMINAMATH_CALUDE_sum_of_roots_eq_seven_halves_l282_28276

theorem sum_of_roots_eq_seven_halves :
  let f : ℝ → ℝ := λ x => (2*x + 3)*(x - 5) - 27
  ∃ x₁ x₂ : ℝ, (f x₁ = 0 ∧ f x₂ = 0 ∧ x₁ ≠ x₂) ∧ x₁ + x₂ = 7/2 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_roots_eq_seven_halves_l282_28276


namespace NUMINAMATH_CALUDE_equal_revenue_both_options_l282_28205

/-- Represents the fishing company's financial model -/
structure FishingCompany where
  initial_cost : ℕ
  first_year_expenses : ℕ
  annual_expense_increase : ℕ
  annual_revenue : ℕ

/-- Calculates the net profit for a given number of years -/
def net_profit (company : FishingCompany) (years : ℕ) : ℤ :=
  (company.annual_revenue * years : ℤ) -
  ((company.first_year_expenses + (years - 1) * company.annual_expense_increase / 2) * years : ℤ) -
  company.initial_cost

/-- Calculates the total revenue when selling at maximum average annual profit -/
def revenue_max_avg_profit (company : FishingCompany) (sell_price : ℕ) : ℤ :=
  net_profit company 7 + sell_price

/-- Calculates the total revenue when selling at maximum total net profit -/
def revenue_max_total_profit (company : FishingCompany) (sell_price : ℕ) : ℤ :=
  net_profit company 10 + sell_price

/-- Theorem stating that both selling options result in the same total revenue -/
theorem equal_revenue_both_options (company : FishingCompany) :
  revenue_max_avg_profit company 2600000 = revenue_max_total_profit company 800000 :=
by sorry

end NUMINAMATH_CALUDE_equal_revenue_both_options_l282_28205


namespace NUMINAMATH_CALUDE_stationery_cost_l282_28271

/-- Given the cost of stationery items, prove the total cost of a specific combination. -/
theorem stationery_cost (E P M : ℕ) : 
  (E + 3 * P + 2 * M = 240) →
  (2 * E + 5 * P + 4 * M = 440) →
  (3 * E + 4 * P + 6 * M = 520) :=
by sorry

end NUMINAMATH_CALUDE_stationery_cost_l282_28271


namespace NUMINAMATH_CALUDE_expression_meaningful_iff_l282_28294

def meaningful_expression (x : ℝ) : Prop :=
  x ≠ -5

theorem expression_meaningful_iff (x : ℝ) :
  meaningful_expression x ↔ x ≠ -5 :=
by
  sorry

end NUMINAMATH_CALUDE_expression_meaningful_iff_l282_28294


namespace NUMINAMATH_CALUDE_cistern_emptying_l282_28239

/-- If a pipe can empty 3/4 of a cistern in 12 minutes, then it will empty 1/2 of the cistern in 8 minutes. -/
theorem cistern_emptying (empty_rate : ℚ) (empty_time : ℕ) (target_time : ℕ) :
  empty_rate = 3/4 ∧ empty_time = 12 ∧ target_time = 8 →
  (target_time : ℚ) * (empty_rate / empty_time) = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_cistern_emptying_l282_28239


namespace NUMINAMATH_CALUDE_line_only_count_l282_28290

/-- Represents an alphabet with letters containing dots and straight lines -/
structure Alphabet :=
  (total_letters : ℕ)
  (dot_and_line : ℕ)
  (dot_only : ℕ)
  (h_total : total_letters = 40)
  (h_dot_and_line : dot_and_line = 9)
  (h_dot_only : dot_only = 7)
  (h_all_contain : total_letters = dot_and_line + dot_only + (total_letters - (dot_and_line + dot_only)))

/-- The number of letters containing a straight line but not a dot -/
def line_only (α : Alphabet) : ℕ := α.total_letters - (α.dot_and_line + α.dot_only)

theorem line_only_count (α : Alphabet) : line_only α = 24 := by
  sorry

end NUMINAMATH_CALUDE_line_only_count_l282_28290


namespace NUMINAMATH_CALUDE_remaining_movies_to_watch_l282_28248

theorem remaining_movies_to_watch 
  (total_movies : ℕ) 
  (watched_movies : ℕ) 
  (total_books : ℕ) 
  (read_books : ℕ) 
  (h1 : total_movies = 12) 
  (h2 : watched_movies = 6) 
  (h3 : total_books = 21) 
  (h4 : read_books = 7) 
  (h5 : watched_movies ≤ total_movies) : 
  total_movies - watched_movies = 6 := by
sorry

end NUMINAMATH_CALUDE_remaining_movies_to_watch_l282_28248


namespace NUMINAMATH_CALUDE_shaded_area_in_grid_l282_28241

/-- The area of a shape in a 3x3 grid formed by a 3x1 rectangle with one 1x1 square removed -/
theorem shaded_area_in_grid (grid_size : Nat) (square_side_length : ℝ) 
  (h1 : grid_size = 3) 
  (h2 : square_side_length = 1) : ℝ := by
  sorry

#check shaded_area_in_grid

end NUMINAMATH_CALUDE_shaded_area_in_grid_l282_28241


namespace NUMINAMATH_CALUDE_not_sum_of_two_rational_squares_168_l282_28297

theorem not_sum_of_two_rational_squares_168 : ¬ ∃ (a b : ℚ), a^2 + b^2 = 168 := by
  sorry

end NUMINAMATH_CALUDE_not_sum_of_two_rational_squares_168_l282_28297


namespace NUMINAMATH_CALUDE_digit_sum_problem_l282_28251

theorem digit_sum_problem (a b c d : ℕ) : 
  a < 10 ∧ b < 10 ∧ c < 10 ∧ d < 10 ∧  -- Digits are less than 10
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d ∧  -- All digits are different
  100 * a + 10 * b + c + 100 * d + 10 * c + a = 1100  -- The equation
  → a + b + c + d = 19 := by
sorry

end NUMINAMATH_CALUDE_digit_sum_problem_l282_28251


namespace NUMINAMATH_CALUDE_polynomial_division_remainder_l282_28253

theorem polynomial_division_remainder (x : ℂ) : 
  (x^6 - 1) * (x^3 - 1) % (x^2 + x + 1) = 0 := by
sorry

end NUMINAMATH_CALUDE_polynomial_division_remainder_l282_28253


namespace NUMINAMATH_CALUDE_tangent_product_l282_28260

theorem tangent_product (A B : ℝ) (h1 : A + B = 5 * Real.pi / 4) 
  (h2 : ∀ k : ℤ, A + B ≠ k * Real.pi + Real.pi / 2) : 
  (1 + Real.tan A) * (1 + Real.tan B) = 2 := by
  sorry

end NUMINAMATH_CALUDE_tangent_product_l282_28260


namespace NUMINAMATH_CALUDE_electronic_shop_purchase_cost_l282_28288

/-- The price of a smartphone in dollars -/
def smartphone_price : ℕ := 300

/-- The price difference between a personal computer and a smartphone in dollars -/
def pc_price_difference : ℕ := 500

/-- The price of a personal computer in dollars -/
def pc_price : ℕ := smartphone_price + pc_price_difference

/-- The price of an advanced tablet in dollars -/
def tablet_price : ℕ := smartphone_price + pc_price

/-- The total cost of buying one of each product in dollars -/
def total_cost : ℕ := smartphone_price + pc_price + tablet_price

theorem electronic_shop_purchase_cost : total_cost = 2200 := by
  sorry

end NUMINAMATH_CALUDE_electronic_shop_purchase_cost_l282_28288


namespace NUMINAMATH_CALUDE_gcd_765432_654321_l282_28262

theorem gcd_765432_654321 : Nat.gcd 765432 654321 = 6 := by
  sorry

end NUMINAMATH_CALUDE_gcd_765432_654321_l282_28262


namespace NUMINAMATH_CALUDE_imaginary_part_of_complex_product_l282_28282

theorem imaginary_part_of_complex_product : Complex.im ((1 + Complex.I)^2 * (2 + Complex.I)) = 4 := by
  sorry

end NUMINAMATH_CALUDE_imaginary_part_of_complex_product_l282_28282


namespace NUMINAMATH_CALUDE_max_subway_employees_l282_28203

theorem max_subway_employees (total_employees : ℕ) 
  (h_total : total_employees = 48) 
  (part_time full_time : ℕ) 
  (h_sum : part_time + full_time = total_employees)
  (subway_part_time subway_full_time : ℕ)
  (h_part_time : subway_part_time * 3 ≤ part_time)
  (h_full_time : subway_full_time * 4 ≤ full_time) :
  subway_part_time + subway_full_time ≤ 15 :=
sorry

end NUMINAMATH_CALUDE_max_subway_employees_l282_28203


namespace NUMINAMATH_CALUDE_correct_observation_value_l282_28237

/-- Proves that the correct value of a misrecorded observation is 48, given the conditions of the problem. -/
theorem correct_observation_value (n : ℕ) (original_mean corrected_mean wrong_value : ℚ)
  (h_n : n = 50)
  (h_original_mean : original_mean = 32)
  (h_corrected_mean : corrected_mean = 32.5)
  (h_wrong_value : wrong_value = 23) :
  let correct_value := (n : ℚ) * corrected_mean - ((n : ℚ) * original_mean - wrong_value)
  correct_value = 48 := by
  sorry

end NUMINAMATH_CALUDE_correct_observation_value_l282_28237


namespace NUMINAMATH_CALUDE_decimal_point_problem_l282_28219

theorem decimal_point_problem (x : ℝ) (h1 : x > 0) (h2 : 1000 * x = 9 * (1 / x)) : 
  x = 3 * Real.sqrt 10 / 100 := by
sorry

end NUMINAMATH_CALUDE_decimal_point_problem_l282_28219


namespace NUMINAMATH_CALUDE_quadratic_rewrite_l282_28269

theorem quadratic_rewrite :
  ∃ (p q r : ℤ), 
    (∀ x, 8 * x^2 - 24 * x - 56 = (p * x + q)^2 + r) ∧
    p * q = -12 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_rewrite_l282_28269


namespace NUMINAMATH_CALUDE_rational_equation_implies_c_zero_l282_28221

theorem rational_equation_implies_c_zero (a b c : ℚ) 
  (h : (a + b + c) * (a + b - c) = 2 * c^2) : c = 0 := by
  sorry

end NUMINAMATH_CALUDE_rational_equation_implies_c_zero_l282_28221


namespace NUMINAMATH_CALUDE_range_of_a_l282_28233

-- Define the propositions p and q
def p (a x : ℝ) : Prop := a - 4 < x ∧ x < a + 4
def q (x : ℝ) : Prop := (x - 2) * (x - 3) > 0

-- Define the theorem
theorem range_of_a (a : ℝ) :
  (∀ x, p a x → q x) →
  (a ≤ -2 ∨ a ≥ 7) :=
sorry

end NUMINAMATH_CALUDE_range_of_a_l282_28233


namespace NUMINAMATH_CALUDE_calculation_proof_l282_28220

theorem calculation_proof : 5^2 * 3 + (7 * 2 - 15) / 3 = 74 + 2/3 := by
  sorry

end NUMINAMATH_CALUDE_calculation_proof_l282_28220


namespace NUMINAMATH_CALUDE_otimes_h_otimes_h_l282_28279

-- Define the ⊗ operation
def otimes (x y : ℝ) : ℝ := x^2 - y

-- Theorem statement
theorem otimes_h_otimes_h (h : ℝ) : otimes h (otimes h h) = h := by
  sorry

end NUMINAMATH_CALUDE_otimes_h_otimes_h_l282_28279


namespace NUMINAMATH_CALUDE_train_distance_problem_l282_28246

/-- The distance between two points A and B, given train speeds and time difference --/
theorem train_distance_problem (v_ab v_ba : ℝ) (time_diff : ℝ) : 
  v_ab = 160 → v_ba = 120 → time_diff = 1 → 
  ∃ D : ℝ, D / v_ba = D / v_ab + time_diff ∧ D = 480 := by
  sorry

#check train_distance_problem

end NUMINAMATH_CALUDE_train_distance_problem_l282_28246


namespace NUMINAMATH_CALUDE_even_perfect_square_factors_count_l282_28254

def num_factors (n : ℕ) : ℕ := sorry

def is_even_perfect_square (n : ℕ) : Prop := sorry

theorem even_perfect_square_factors_count : 
  ∃ (f : ℕ → ℕ), 
    (∀ x, is_even_perfect_square (f x)) ∧ 
    (∀ x, f x ∣ (2^6 * 5^3 * 7^8)) ∧ 
    (num_factors (2^6 * 5^3 * 7^8) = 30) := by
  sorry

end NUMINAMATH_CALUDE_even_perfect_square_factors_count_l282_28254


namespace NUMINAMATH_CALUDE_solution_characterization_l282_28293

theorem solution_characterization (x y z : ℝ) :
  (x - y + z)^2 = x^2 - y^2 + z^2 ↔ (x = y ∧ z = 0) ∨ (x = 0 ∧ y = z) := by
  sorry

end NUMINAMATH_CALUDE_solution_characterization_l282_28293


namespace NUMINAMATH_CALUDE_marias_workday_ends_at_5pm_l282_28264

/-- Represents time in 24-hour format -/
structure Time where
  hour : Nat
  minute : Nat
  deriving Repr

/-- Represents a workday -/
structure Workday where
  start : Time
  totalWorkHours : Nat
  lunchBreakStart : Time
  lunchBreakDuration : Nat
  deriving Repr

def addHours (t : Time) (h : Nat) : Time :=
  { hour := (t.hour + h) % 24, minute := t.minute }

def addMinutes (t : Time) (m : Nat) : Time :=
  let totalMinutes := t.hour * 60 + t.minute + m
  { hour := totalMinutes / 60, minute := totalMinutes % 60 }

def mariasWorkday : Workday :=
  { start := { hour := 8, minute := 0 },
    totalWorkHours := 8,
    lunchBreakStart := { hour := 13, minute := 0 },
    lunchBreakDuration := 1 }

theorem marias_workday_ends_at_5pm :
  let endTime := addHours (addMinutes mariasWorkday.lunchBreakStart mariasWorkday.lunchBreakDuration)
                          (mariasWorkday.totalWorkHours - (mariasWorkday.lunchBreakStart.hour - mariasWorkday.start.hour))
  endTime = { hour := 17, minute := 0 } :=
by sorry

end NUMINAMATH_CALUDE_marias_workday_ends_at_5pm_l282_28264


namespace NUMINAMATH_CALUDE_value_of_a_l282_28255

/-- Proves that if 0.5% of a equals 70 paise, then a equals 140 rupees. -/
theorem value_of_a (a : ℝ) : (0.5 / 100) * a = 70 / 100 → a = 140 := by
  sorry

end NUMINAMATH_CALUDE_value_of_a_l282_28255


namespace NUMINAMATH_CALUDE_no_solution_exists_l282_28236

theorem no_solution_exists : ¬∃ (x : ℝ), Real.arccos (4/5) - Real.arccos (-4/5) = Real.arcsin x := by
  sorry

end NUMINAMATH_CALUDE_no_solution_exists_l282_28236


namespace NUMINAMATH_CALUDE_mandy_shirts_l282_28256

/-- The number of black shirt packs bought -/
def black_packs : ℕ := 6

/-- The number of shirts in each black shirt pack -/
def black_per_pack : ℕ := 7

/-- The number of yellow shirt packs bought -/
def yellow_packs : ℕ := 8

/-- The number of shirts in each yellow shirt pack -/
def yellow_per_pack : ℕ := 4

/-- The total number of shirts bought -/
def total_shirts : ℕ := black_packs * black_per_pack + yellow_packs * yellow_per_pack

theorem mandy_shirts : total_shirts = 74 := by
  sorry

end NUMINAMATH_CALUDE_mandy_shirts_l282_28256


namespace NUMINAMATH_CALUDE_total_flowers_l282_28257

def flower_problem (yoojung_flowers namjoon_flowers : ℕ) : Prop :=
  (yoojung_flowers = 4 * namjoon_flowers) ∧ 
  (yoojung_flowers = 32)

theorem total_flowers : 
  ∀ yoojung_flowers namjoon_flowers : ℕ, 
  flower_problem yoojung_flowers namjoon_flowers → 
  yoojung_flowers + namjoon_flowers = 40 :=
by
  sorry

end NUMINAMATH_CALUDE_total_flowers_l282_28257


namespace NUMINAMATH_CALUDE_not_p_and_q_implies_not_p_or_not_q_l282_28227

theorem not_p_and_q_implies_not_p_or_not_q (p q : Prop) :
  ¬(p ∧ q) → (¬p ∨ ¬q) := by
  sorry

end NUMINAMATH_CALUDE_not_p_and_q_implies_not_p_or_not_q_l282_28227


namespace NUMINAMATH_CALUDE_tan_eq_two_solution_set_l282_28240

theorem tan_eq_two_solution_set :
  {x : ℝ | ∃ k : ℤ, x = k * Real.pi + (-1)^k * Real.arctan 2} =
  {x : ℝ | Real.tan x = 2} := by sorry

end NUMINAMATH_CALUDE_tan_eq_two_solution_set_l282_28240


namespace NUMINAMATH_CALUDE_physics_value_l282_28273

def letterValue (n : Nat) : Int :=
  match n % 9 with
  | 0 => 0
  | 1 => 2
  | 2 => 3
  | 3 => 2
  | 4 => 0
  | 5 => -1
  | 6 => -2
  | 7 => -3
  | 8 => -2
  | _ => -1

def wordValue (word : List Nat) : Int :=
  List.sum (List.map letterValue word)

theorem physics_value :
  wordValue [16, 8, 25, 19, 9, 3, 19] = 1 := by
  sorry

end NUMINAMATH_CALUDE_physics_value_l282_28273


namespace NUMINAMATH_CALUDE_teachers_gathering_problem_l282_28218

theorem teachers_gathering_problem (male_teachers female_teachers : ℕ) 
  (h1 : female_teachers = male_teachers + 12)
  (h2 : (male_teachers : ℚ) / (male_teachers + female_teachers) = 9 / 20) :
  male_teachers + female_teachers = 120 := by
sorry

end NUMINAMATH_CALUDE_teachers_gathering_problem_l282_28218


namespace NUMINAMATH_CALUDE_unique_fraction_decomposition_l282_28261

theorem unique_fraction_decomposition (p : ℕ) (h_prime : Nat.Prime p) (h_odd : Odd p) :
  ∃! (m n : ℕ), m ≠ n ∧ 2 / p = 1 / n + 1 / m ∧
  n = (p + 1) / 2 ∧ m = p * (p + 1) / 2 := by
  sorry

end NUMINAMATH_CALUDE_unique_fraction_decomposition_l282_28261


namespace NUMINAMATH_CALUDE_ball_returns_in_five_throws_l282_28215

/-- The number of elements in the circular arrangement -/
def n : ℕ := 13

/-- The number of elements skipped in each throw -/
def skip : ℕ := 4

/-- The number of throws needed to return to the starting element -/
def throws : ℕ := 5

/-- Function to calculate the next position after a throw -/
def nextPosition (current : ℕ) : ℕ :=
  (current + skip + 1) % n

/-- Theorem stating that it takes 5 throws to return to the starting position -/
theorem ball_returns_in_five_throws :
  (throws.iterate nextPosition 0) % n = 0 := by sorry

end NUMINAMATH_CALUDE_ball_returns_in_five_throws_l282_28215


namespace NUMINAMATH_CALUDE_not_perfect_square_l282_28277

theorem not_perfect_square (n : ℕ+) : 
  (n^2 + n)^2 < n^4 + 2*n^3 + 2*n^2 + 2*n + 1 ∧ 
  n^4 + 2*n^3 + 2*n^2 + 2*n + 1 < (n^2 + n + 1)^2 :=
by sorry

end NUMINAMATH_CALUDE_not_perfect_square_l282_28277


namespace NUMINAMATH_CALUDE_all_real_roots_condition_l282_28249

theorem all_real_roots_condition (k : ℝ) : 
  (∀ x : ℂ, x^4 - 4*x^3 + 4*x^2 + k*x - 4 = 0 → x.im = 0) ↔ k = -8 := by
sorry

end NUMINAMATH_CALUDE_all_real_roots_condition_l282_28249


namespace NUMINAMATH_CALUDE_fire_in_city_a_l282_28216

-- Define the cities
inductive City
| A
| B
| C

-- Define the possible statements
inductive Statement
| Fire
| LocationC

-- Define the behavior of residents in each city
def always_truth (c : City) : Prop :=
  c = City.A

def always_lie (c : City) : Prop :=
  c = City.B

def alternate (c : City) : Prop :=
  c = City.C

-- Define the caller's statements
def caller_statements : List Statement :=
  [Statement.Fire, Statement.LocationC]

-- Define the property of the actual fire location
def is_actual_fire_location (c : City) : Prop :=
  ∀ (s : Statement), s ∈ caller_statements → 
    (always_truth c → s = Statement.Fire) ∧
    (always_lie c → s ≠ Statement.LocationC) ∧
    (alternate c → (s = Statement.Fire ↔ s ≠ Statement.LocationC))

-- Theorem: The actual fire location is City A
theorem fire_in_city_a :
  is_actual_fire_location City.A :=
sorry

end NUMINAMATH_CALUDE_fire_in_city_a_l282_28216


namespace NUMINAMATH_CALUDE_quadratic_trinomial_square_l282_28207

theorem quadratic_trinomial_square (a b c : ℝ) :
  (∀ x : ℤ, ∃ y : ℤ, a * x^2 + b * x + c = y^2) →
  (∃ m n k : ℤ, 2 * a = m ∧ 2 * b = n ∧ c = k^2) ∧
  (∃ p q r : ℤ, a = p ∧ b = q ∧ c = r^2) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_trinomial_square_l282_28207


namespace NUMINAMATH_CALUDE_nine_integer_lengths_l282_28200

/-- Represents a right triangle with integer leg lengths -/
structure RightTriangle where
  de : ℕ
  ef : ℕ

/-- Counts the number of distinct integer lengths of line segments
    from vertex E to the hypotenuse DF in a right triangle DEF -/
def countIntegerLengths (t : RightTriangle) : ℕ :=
  sorry

theorem nine_integer_lengths (t : RightTriangle) 
  (h1 : t.de = 24) (h2 : t.ef = 25) : 
  countIntegerLengths t = 9 :=
sorry

end NUMINAMATH_CALUDE_nine_integer_lengths_l282_28200


namespace NUMINAMATH_CALUDE_meal_combinations_count_l282_28230

/-- The number of items in Menu A -/
def menu_a_items : ℕ := 15

/-- The number of items in Menu B -/
def menu_b_items : ℕ := 12

/-- The total number of possible meal combinations -/
def total_combinations : ℕ := menu_a_items * menu_b_items

/-- Theorem stating that the total number of meal combinations is 180 -/
theorem meal_combinations_count : total_combinations = 180 := by
  sorry

end NUMINAMATH_CALUDE_meal_combinations_count_l282_28230


namespace NUMINAMATH_CALUDE_complex_square_minus_i_l282_28267

theorem complex_square_minus_i (z : ℂ) : z = 1 + I → z^2 - I = I := by
  sorry

end NUMINAMATH_CALUDE_complex_square_minus_i_l282_28267


namespace NUMINAMATH_CALUDE_no_fixed_points_iff_a_in_range_l282_28259

/-- A function f: ℝ → ℝ has no fixed points if for all x: ℝ, f x ≠ x -/
def has_no_fixed_points (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f x ≠ x

/-- The quadratic function f(x) = x^2 + 2ax + 1 -/
def f (a : ℝ) (x : ℝ) : ℝ :=
  x^2 + 2*a*x + 1

/-- Theorem stating that f(x) = x^2 + 2ax + 1 has no fixed points iff a ∈ (-1/2, 3/2) -/
theorem no_fixed_points_iff_a_in_range (a : ℝ) :
  has_no_fixed_points (f a) ↔ -1/2 < a ∧ a < 3/2 :=
sorry

end NUMINAMATH_CALUDE_no_fixed_points_iff_a_in_range_l282_28259


namespace NUMINAMATH_CALUDE_translated_minimum_point_l282_28224

def f (x : ℝ) : ℝ := |x| - 4

def g (x : ℝ) : ℝ := f (x - 3) - 4

theorem translated_minimum_point :
  ∃ (x : ℝ), ∀ (y : ℝ), g y ≥ g x ∧ g x = -8 ∧ x = 3 := by
  sorry

end NUMINAMATH_CALUDE_translated_minimum_point_l282_28224


namespace NUMINAMATH_CALUDE_quadratic_equation_solution_l282_28289

theorem quadratic_equation_solution :
  ∀ x : ℝ, (x - 2)^2 - 4 = 0 ↔ x = 4 ∨ x = 0 := by
sorry

end NUMINAMATH_CALUDE_quadratic_equation_solution_l282_28289


namespace NUMINAMATH_CALUDE_isosceles_right_triangle_hypotenuse_l282_28238

/-- An isosceles right triangle with perimeter 8 + 8√2 has a hypotenuse of length 8 -/
theorem isosceles_right_triangle_hypotenuse (a c : ℝ) : 
  a > 0 → -- side length is positive
  c > 0 → -- hypotenuse length is positive
  a + a + c = 8 + 8 * Real.sqrt 2 → -- perimeter condition
  a * a + a * a = c * c → -- Pythagorean theorem for isosceles right triangle
  c = 8 := by
sorry

end NUMINAMATH_CALUDE_isosceles_right_triangle_hypotenuse_l282_28238


namespace NUMINAMATH_CALUDE_cube_sum_divided_l282_28229

theorem cube_sum_divided (x y : ℝ) (hx : x = 3) (hy : y = 4) : 
  (x^3 + 3*y^3) / 9 = 73/3 := by
sorry

end NUMINAMATH_CALUDE_cube_sum_divided_l282_28229


namespace NUMINAMATH_CALUDE_network_connections_l282_28274

/-- 
Given a network of switches where:
- There are 30 switches
- Each switch is directly connected to exactly 4 other switches
This theorem states that the total number of connections in the network is 60.
-/
theorem network_connections (n : ℕ) (c : ℕ) : 
  n = 30 → c = 4 → (n * c) / 2 = 60 := by
  sorry

end NUMINAMATH_CALUDE_network_connections_l282_28274


namespace NUMINAMATH_CALUDE_fence_nailing_problem_l282_28283

theorem fence_nailing_problem (petrov_two : ℕ) (petrov_three : ℕ) 
  (vasechkin_three : ℕ) (vasechkin_five : ℕ) : 
  (2 * petrov_two + 3 * petrov_three = 87) →
  (3 * vasechkin_three + 5 * vasechkin_five = 94) →
  (petrov_two + petrov_three = vasechkin_three + vasechkin_five) →
  (petrov_two + petrov_three = 30) := by
  sorry

#check fence_nailing_problem

end NUMINAMATH_CALUDE_fence_nailing_problem_l282_28283


namespace NUMINAMATH_CALUDE_time_to_write_rearrangements_l282_28287

/-- The time required to write all rearrangements of a name -/
theorem time_to_write_rearrangements 
  (num_letters : ℕ) 
  (rearrangements_per_minute : ℕ) 
  (h1 : num_letters = 5) 
  (h2 : rearrangements_per_minute = 15) : 
  (Nat.factorial num_letters : ℚ) / (rearrangements_per_minute * 60 : ℚ) = 2 / 15 := by
  sorry

end NUMINAMATH_CALUDE_time_to_write_rearrangements_l282_28287


namespace NUMINAMATH_CALUDE_percentage_problem_l282_28242

theorem percentage_problem (x : ℝ) (p : ℝ) 
  (h1 : (p / 100) * x = 400)
  (h2 : (120 / 100) * x = 2400) : 
  p = 20 := by sorry

end NUMINAMATH_CALUDE_percentage_problem_l282_28242


namespace NUMINAMATH_CALUDE_photographer_profit_percentage_l282_28292

/-- Calculates the profit percentage for a photographer's business --/
theorem photographer_profit_percentage
  (selling_price : ℝ)
  (production_cost : ℝ)
  (sale_probability : ℝ)
  (h1 : selling_price = 600)
  (h2 : production_cost = 100)
  (h3 : sale_probability = 1/4)
  : (((sale_probability * selling_price - production_cost) / production_cost) * 100 = 50) :=
by sorry

end NUMINAMATH_CALUDE_photographer_profit_percentage_l282_28292


namespace NUMINAMATH_CALUDE_series_end_probability_l282_28226

/-- Probability of Mathletes winning a single game -/
def p : ℚ := 2/3

/-- Probability of the opponent winning a single game -/
def q : ℚ := 1 - p

/-- Number of games in the series before the final game -/
def n : ℕ := 6

/-- Number of wins required to end the series -/
def k : ℕ := 5

/-- Probability of the series ending in exactly 7 games -/
def prob_series_end_7 : ℚ := 
  (Nat.choose n (k-1)) * (p^(k-1) * q^(n-(k-1)) * p + p^(n-(k-1)) * q^(k-1) * q)

theorem series_end_probability :
  prob_series_end_7 = 20/81 := by
  sorry

end NUMINAMATH_CALUDE_series_end_probability_l282_28226


namespace NUMINAMATH_CALUDE_polygon_25_sides_l282_28214

/-- A convex polygon with n sides -/
structure ConvexPolygon (n : ℕ) where
  sides : n > 2

/-- Number of diagonals in a polygon with n sides -/
def numDiagonals (n : ℕ) : ℕ := n * (n - 3) / 2

/-- Number of triangles formed by choosing any three vertices of a polygon with n sides -/
def numTriangles (n : ℕ) : ℕ := n.choose 3

theorem polygon_25_sides (P : ConvexPolygon 25) : 
  numDiagonals 25 = 275 ∧ numTriangles 25 = 2300 := by
  sorry


end NUMINAMATH_CALUDE_polygon_25_sides_l282_28214


namespace NUMINAMATH_CALUDE_multiple_of_six_between_14_and_30_l282_28228

theorem multiple_of_six_between_14_and_30 (x : ℕ) 
  (h1 : ∃ k : ℕ, x = 6 * k)
  (h2 : x^2 > 196)
  (h3 : x < 30) :
  x = 18 ∨ x = 24 := by
sorry

end NUMINAMATH_CALUDE_multiple_of_six_between_14_and_30_l282_28228


namespace NUMINAMATH_CALUDE_four_digit_number_with_sum_14_divisible_by_14_l282_28295

theorem four_digit_number_with_sum_14_divisible_by_14 :
  ∃ n : ℕ,
    1000 ≤ n ∧ n ≤ 9999 ∧
    (n / 1000 + (n / 100 % 10) + (n / 10 % 10) + (n % 10) = 14) ∧
    n % 14 = 0 := by
  sorry

end NUMINAMATH_CALUDE_four_digit_number_with_sum_14_divisible_by_14_l282_28295


namespace NUMINAMATH_CALUDE_exam_problem_solution_l282_28247

theorem exam_problem_solution (pA pB pC : ℝ) 
  (hA : pA = 1/3) 
  (hB : pB = 1/4) 
  (hC : pC = 1/5) 
  (h_independent : True) -- Assumption of independence
  : 1 - (1 - pA) * (1 - pB) * (1 - pC) = 3/5 := by
  sorry

end NUMINAMATH_CALUDE_exam_problem_solution_l282_28247


namespace NUMINAMATH_CALUDE_quadratic_two_real_roots_l282_28222

/-- 
For a quadratic equation kx^2 - 2x - 1 = 0, where k is a real number,
the equation has two real roots if and only if k ≥ -1 and k ≠ 0.
-/
theorem quadratic_two_real_roots (k : ℝ) : 
  (∃ x y : ℝ, x ≠ y ∧ k * x^2 - 2*x - 1 = 0 ∧ k * y^2 - 2*y - 1 = 0) ↔ 
  (k ≥ -1 ∧ k ≠ 0) :=
sorry

end NUMINAMATH_CALUDE_quadratic_two_real_roots_l282_28222


namespace NUMINAMATH_CALUDE_expression_evaluation_l282_28268

theorem expression_evaluation :
  let x : ℚ := 1/25
  let y : ℚ := -25
  x * (x + 2*y) - (x + 1)^2 + 2*x = -3 := by sorry

end NUMINAMATH_CALUDE_expression_evaluation_l282_28268


namespace NUMINAMATH_CALUDE_b_completion_time_l282_28272

/- Define the work completion time for A -/
def a_completion_time : ℝ := 9

/- Define B's efficiency relative to A -/
def b_efficiency_factor : ℝ := 1.5

/- Theorem statement -/
theorem b_completion_time :
  let a_rate := 1 / a_completion_time
  let b_rate := b_efficiency_factor * a_rate
  (1 / b_rate) = 6 := by sorry

end NUMINAMATH_CALUDE_b_completion_time_l282_28272


namespace NUMINAMATH_CALUDE_car_speed_problem_l282_28296

/-- Given two cars leaving town A at the same time in the same direction,
    prove that if one car travels at 55 mph and they are 45 miles apart after 3 hours,
    then the speed of the other car must be 70 mph. -/
theorem car_speed_problem (v : ℝ) : 
  v * 3 - 55 * 3 = 45 → v = 70 := by
  sorry

end NUMINAMATH_CALUDE_car_speed_problem_l282_28296


namespace NUMINAMATH_CALUDE_sum_of_numbers_l282_28213

theorem sum_of_numbers (x y : ℝ) (h1 : x > 0) (h2 : y > 0) 
  (h3 : x * y = 375) (h4 : 1 / x + 1 / y = 0.10666666666666667) : 
  x + y = 40 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_numbers_l282_28213


namespace NUMINAMATH_CALUDE_trigonometric_simplification_l282_28244

theorem trigonometric_simplification :
  let f : ℝ → ℝ := λ x => Real.sin (x * π / 180)
  let g : ℝ → ℝ := λ x => Real.cos (x * π / 180)
  (f 15 + f 25 + f 35 + f 45 + f 55 + f 65 + f 75 + f 85) / (g 10 * g 15 * g 30) =
  8 * Real.sqrt 3 * g 40 * g 5 :=
by sorry

end NUMINAMATH_CALUDE_trigonometric_simplification_l282_28244


namespace NUMINAMATH_CALUDE_green_peaches_per_basket_l282_28275

theorem green_peaches_per_basket (num_baskets : ℕ) (red_per_basket : ℕ) (total_peaches : ℕ) :
  num_baskets = 11 →
  red_per_basket = 10 →
  total_peaches = 308 →
  ∃ green_per_basket : ℕ, 
    green_per_basket * num_baskets + red_per_basket * num_baskets = total_peaches ∧
    green_per_basket = 18 := by
  sorry

end NUMINAMATH_CALUDE_green_peaches_per_basket_l282_28275


namespace NUMINAMATH_CALUDE_fruit_basket_ratio_l282_28243

/-- Fruit basket problem -/
theorem fruit_basket_ratio : 
  ∀ (oranges apples bananas peaches : ℕ),
  oranges = 6 →
  apples = oranges - 2 →
  bananas = 3 * apples →
  oranges + apples + bananas + peaches = 28 →
  peaches * 2 = bananas :=
by
  sorry

end NUMINAMATH_CALUDE_fruit_basket_ratio_l282_28243


namespace NUMINAMATH_CALUDE_renovation_project_material_l282_28280

theorem renovation_project_material (sand dirt cement : ℝ) 
  (h_sand : sand = 0.17)
  (h_dirt : dirt = 0.33)
  (h_cement : cement = 0.17) :
  sand + dirt + cement = 0.67 := by
  sorry

end NUMINAMATH_CALUDE_renovation_project_material_l282_28280


namespace NUMINAMATH_CALUDE_fast_food_cost_l282_28232

/-- Represents the cost of items at a fast food restaurant -/
structure FastFoodCost where
  hamburger : ℝ
  milkshake : ℝ
  fries : ℝ

/-- Given the costs of different combinations, prove the cost of 2 hamburgers, 2 milkshakes, and 2 fries -/
theorem fast_food_cost (c : FastFoodCost) 
  (eq1 : 3 * c.hamburger + 5 * c.milkshake + c.fries = 23.5)
  (eq2 : 5 * c.hamburger + 9 * c.milkshake + c.fries = 39.5) :
  2 * c.hamburger + 2 * c.milkshake + 2 * c.fries = 15 := by
  sorry

end NUMINAMATH_CALUDE_fast_food_cost_l282_28232


namespace NUMINAMATH_CALUDE_negation_of_or_statement_l282_28204

theorem negation_of_or_statement (x y : ℝ) :
  ¬(x > 1 ∨ y > 1) ↔ x ≤ 1 ∧ y ≤ 1 := by
  sorry

end NUMINAMATH_CALUDE_negation_of_or_statement_l282_28204


namespace NUMINAMATH_CALUDE_complex_polynomial_root_l282_28298

theorem complex_polynomial_root (a b c : ℤ) : 
  (a * (1 + Complex.I * Real.sqrt 3)^3 + b * (1 + Complex.I * Real.sqrt 3)^2 + c * (1 + Complex.I * Real.sqrt 3) + b + a = 0) →
  (Int.gcd a (Int.gcd b c) = 1) →
  (abs c = 9) := by
  sorry

end NUMINAMATH_CALUDE_complex_polynomial_root_l282_28298


namespace NUMINAMATH_CALUDE_floor_sqrt_101_l282_28278

theorem floor_sqrt_101 : ⌊Real.sqrt 101⌋ = 10 := by
  sorry

end NUMINAMATH_CALUDE_floor_sqrt_101_l282_28278


namespace NUMINAMATH_CALUDE_linear_function_through_minus_one_zero_l282_28286

/-- A linear function passing through (-1, 0) has slope 1 -/
theorem linear_function_through_minus_one_zero (k : ℝ) : 
  (∀ x y : ℝ, y = k * x + 1) → 0 = k * (-1) + 1 → k = 1 := by
  sorry

end NUMINAMATH_CALUDE_linear_function_through_minus_one_zero_l282_28286


namespace NUMINAMATH_CALUDE_second_white_given_first_white_probability_l282_28266

/-- Represents the color of a ball -/
inductive Color
| White
| Black

/-- Represents the pocket containing balls -/
structure Pocket where
  white : Nat
  black : Nat

/-- Represents the result of two consecutive draws -/
structure TwoDraws where
  first : Color
  second : Color

/-- Calculates the probability of drawing a white ball on the second draw
    given that the first ball drawn is white -/
def probSecondWhiteGivenFirstWhite (p : Pocket) : Rat :=
  if p.white > 0 then
    (p.white - 1) / (p.white + p.black - 1)
  else
    0

theorem second_white_given_first_white_probability 
  (p : Pocket) (h1 : p.white = 3) (h2 : p.black = 2) :
  probSecondWhiteGivenFirstWhite p = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_second_white_given_first_white_probability_l282_28266


namespace NUMINAMATH_CALUDE_words_with_vowel_count_l282_28209

/-- The set of all letters used to construct words -/
def letters : Finset Char := {'A', 'B', 'C', 'D', 'E', 'F'}

/-- The set of vowels -/
def vowels : Finset Char := {'A', 'E'}

/-- The set of consonants -/
def consonants : Finset Char := letters \ vowels

/-- The length of words we're considering -/
def wordLength : Nat := 5

/-- The number of 5-letter words with at least one vowel -/
def numWordsWithVowel : Nat :=
  letters.card ^ wordLength - consonants.card ^ wordLength

theorem words_with_vowel_count :
  numWordsWithVowel = 6752 := by
  sorry

end NUMINAMATH_CALUDE_words_with_vowel_count_l282_28209


namespace NUMINAMATH_CALUDE_arithmetic_progression_polynomial_j_eq_neg_40_l282_28245

/-- A polynomial of degree 4 with four distinct real zeros in arithmetic progression -/
structure ArithmeticProgressionPolynomial where
  j : ℝ
  k : ℝ
  roots : Fin 4 → ℝ
  distinct : ∀ i j, i ≠ j → roots i ≠ roots j
  arithmetic_progression : ∃ (b d : ℝ), ∀ i, roots i = b + i * d

/-- The coefficient j of an ArithmeticProgressionPolynomial is -40 -/
theorem arithmetic_progression_polynomial_j_eq_neg_40 (p : ArithmeticProgressionPolynomial) :
  p.j = -40 := by sorry

end NUMINAMATH_CALUDE_arithmetic_progression_polynomial_j_eq_neg_40_l282_28245


namespace NUMINAMATH_CALUDE_books_for_girls_l282_28285

theorem books_for_girls (num_girls num_boys total_books : ℕ) : 
  num_girls = 15 → 
  num_boys = 10 → 
  total_books = 375 → 
  (num_girls * (total_books / (num_girls + num_boys))) = 225 := by
  sorry

end NUMINAMATH_CALUDE_books_for_girls_l282_28285


namespace NUMINAMATH_CALUDE_solve_for_s_l282_28258

theorem solve_for_s (t : ℚ) (h1 : 7 * ((t / 2) + 3) + 6 * t = 156) : (t / 2) + 3 = 192 / 19 := by
  sorry

end NUMINAMATH_CALUDE_solve_for_s_l282_28258


namespace NUMINAMATH_CALUDE_polynomial_coefficient_sum_l282_28263

theorem polynomial_coefficient_sum : ∀ P Q R S : ℝ,
  (∀ x : ℝ, (x + 3) * (4 * x^2 - 2 * x + 7) = P * x^3 + Q * x^2 + R * x + S) →
  P + Q + R + S = 36 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_coefficient_sum_l282_28263


namespace NUMINAMATH_CALUDE_dynaco_price_is_44_l282_28270

/-- Calculates the price per share of Dynaco stock given the total number of shares sold,
    number of Dynaco shares sold, price per share of Microtron stock, and average price
    per share of all stocks sold. -/
def dynaco_price_per_share (total_shares : ℕ) (dynaco_shares : ℕ) 
    (microtron_price : ℚ) (average_price : ℚ) : ℚ :=
  let microtron_shares := total_shares - dynaco_shares
  let total_revenue := (total_shares : ℚ) * average_price
  let microtron_revenue := (microtron_shares : ℚ) * microtron_price
  (total_revenue - microtron_revenue) / (dynaco_shares : ℚ)

/-- Theorem stating that given the specific conditions from the problem,
    the price per share of Dynaco stock is $44. -/
theorem dynaco_price_is_44 :
  dynaco_price_per_share 300 150 36 40 = 44 := by
  sorry

end NUMINAMATH_CALUDE_dynaco_price_is_44_l282_28270


namespace NUMINAMATH_CALUDE_new_library_capacity_l282_28250

theorem new_library_capacity 
  (M : ℚ) -- Millicent's books
  (H : ℚ) -- Harold's books
  (G : ℚ) -- Gertrude's books
  (h1 : H = (1/2) * M) -- Harold has 1/2 as many books as Millicent
  (h2 : G = 3 * H) -- Gertrude has 3 times more books than Harold
  : (1/3) * H + (2/5) * G + (1/2) * M = (29/30) * M := by
  sorry

end NUMINAMATH_CALUDE_new_library_capacity_l282_28250


namespace NUMINAMATH_CALUDE_animal_path_distance_l282_28265

/-- The total distance traveled by an animal along a specific path between two concentric circles -/
theorem animal_path_distance (r₁ r₂ : ℝ) (h₁ : r₁ = 15) (h₂ : r₂ = 25) :
  let outer_arc := (1/4) * 2 * Real.pi * r₂
  let radial_line := r₂ - r₁
  let inner_circle := 2 * Real.pi * r₁
  outer_arc + radial_line + inner_circle + radial_line = 42.5 * Real.pi + 20 := by
  sorry

end NUMINAMATH_CALUDE_animal_path_distance_l282_28265
