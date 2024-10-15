import Mathlib

namespace NUMINAMATH_CALUDE_line_plane_perpendicularity_l4128_412874

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the parallel and perpendicular relations
variable (parallel : Line → Plane → Prop)
variable (perpendicular : Line → Plane → Prop)
variable (plane_perpendicular : Plane → Plane → Prop)

-- State the theorem
theorem line_plane_perpendicularity 
  (l : Line) (α β : Plane) 
  (h1 : parallel l α) 
  (h2 : perpendicular l β) : 
  plane_perpendicular α β :=
sorry

end NUMINAMATH_CALUDE_line_plane_perpendicularity_l4128_412874


namespace NUMINAMATH_CALUDE_max_value_of_f_l4128_412809

-- Define the function f(x) = ln(x) / x
noncomputable def f (x : ℝ) : ℝ := (Real.log x) / x

-- State the theorem
theorem max_value_of_f :
  ∃ (c : ℝ), c > 0 ∧ 
  (∀ x > 0, f x ≤ f c) ∧
  f c = 1 / Real.exp 1 := by
  sorry

end NUMINAMATH_CALUDE_max_value_of_f_l4128_412809


namespace NUMINAMATH_CALUDE_prob_all_red_4th_draw_eq_l4128_412859

/-- The number of white balls initially in the bag -/
def white_balls : ℕ := 8

/-- The number of red balls initially in the bag -/
def red_balls : ℕ := 2

/-- The total number of balls in the bag -/
def total_balls : ℕ := white_balls + red_balls

/-- The probability of drawing all red balls exactly by the 4th draw -/
def prob_all_red_4th_draw : ℚ :=
  -- Event A: Red on 1st, White on 2nd and 3rd, Red on 4th
  (red_balls / total_balls) * ((white_balls + 1) / total_balls) * ((white_balls + 1) / total_balls) * (1 / total_balls) +
  -- Event B: White on 1st, Red on 2nd, White on 3rd, Red on 4th
  (white_balls / total_balls) * (red_balls / total_balls) * ((white_balls + 1) / total_balls) * (1 / total_balls) +
  -- Event C: White on 1st and 2nd, Red on 3rd and 4th
  (white_balls / total_balls) * (white_balls / total_balls) * (red_balls / total_balls) * (1 / total_balls)

theorem prob_all_red_4th_draw_eq : prob_all_red_4th_draw = 353 / 5000 := by
  sorry

end NUMINAMATH_CALUDE_prob_all_red_4th_draw_eq_l4128_412859


namespace NUMINAMATH_CALUDE_joe_savings_l4128_412899

def flight_cost : ℕ := 1200
def hotel_cost : ℕ := 800
def food_cost : ℕ := 3000
def money_left : ℕ := 1000

theorem joe_savings : 
  flight_cost + hotel_cost + food_cost + money_left = 6000 := by
  sorry

end NUMINAMATH_CALUDE_joe_savings_l4128_412899


namespace NUMINAMATH_CALUDE_regular_pentagon_side_length_l4128_412875

/-- The length of a side of a regular pentagon with perimeter 125 is 25 -/
theorem regular_pentagon_side_length :
  ∀ (side_length : ℝ),
    side_length > 0 →
    side_length * 5 = 125 →
    side_length = 25 := by
  sorry

end NUMINAMATH_CALUDE_regular_pentagon_side_length_l4128_412875


namespace NUMINAMATH_CALUDE_scarf_cost_proof_l4128_412803

def sweater_cost : ℕ := 30
def num_items : ℕ := 6
def total_savings : ℕ := 500
def remaining_savings : ℕ := 200

theorem scarf_cost_proof :
  ∃ (scarf_cost : ℕ),
    scarf_cost * num_items = total_savings - remaining_savings - (sweater_cost * num_items) :=
by sorry

end NUMINAMATH_CALUDE_scarf_cost_proof_l4128_412803


namespace NUMINAMATH_CALUDE_unique_m_value_l4128_412885

def A (m : ℝ) : Set ℝ := {0, m, m^2 - 3*m + 2}

theorem unique_m_value : ∃! m : ℝ, 2 ∈ A m ∧ m = 3 := by sorry

end NUMINAMATH_CALUDE_unique_m_value_l4128_412885


namespace NUMINAMATH_CALUDE_symmetric_point_wrt_origin_l4128_412804

/-- Given a point P(-2, 3) in a Cartesian coordinate system, 
    its symmetric point with respect to the origin has coordinates (2, -3). -/
theorem symmetric_point_wrt_origin : 
  let P : ℝ × ℝ := (-2, 3)
  let symmetric_point (p : ℝ × ℝ) : ℝ × ℝ := (-p.1, -p.2)
  symmetric_point P = (2, -3) := by sorry

end NUMINAMATH_CALUDE_symmetric_point_wrt_origin_l4128_412804


namespace NUMINAMATH_CALUDE_f_range_l4128_412813

-- Define the function f
def f (x : ℝ) : ℝ := 3 * (x - 4)

-- State the theorem
theorem f_range :
  Set.range f = {y : ℝ | y < -27 ∨ y > -27} :=
by
  sorry

end NUMINAMATH_CALUDE_f_range_l4128_412813


namespace NUMINAMATH_CALUDE_annas_age_at_marriage_l4128_412838

/-- Proves Anna's age at marriage given the conditions of the problem -/
theorem annas_age_at_marriage
  (josh_age_at_marriage : ℕ)
  (years_of_marriage : ℕ)
  (combined_age_factor : ℕ)
  (h1 : josh_age_at_marriage = 22)
  (h2 : years_of_marriage = 30)
  (h3 : combined_age_factor = 5)
  (h4 : josh_age_at_marriage + years_of_marriage + (josh_age_at_marriage + years_of_marriage + anna_age_at_marriage) = combined_age_factor * josh_age_at_marriage) :
  anna_age_at_marriage = 28 :=
by
  sorry

#check annas_age_at_marriage

end NUMINAMATH_CALUDE_annas_age_at_marriage_l4128_412838


namespace NUMINAMATH_CALUDE_x_plus_y_value_l4128_412865

theorem x_plus_y_value (x y : ℝ) 
  (eq1 : x + Real.sin y = 2008)
  (eq2 : x + 2008 * Real.cos y = 2007)
  (h : 0 ≤ y ∧ y ≤ Real.pi / 2) :
  x + y = 2007 + Real.pi / 2 := by
  sorry

end NUMINAMATH_CALUDE_x_plus_y_value_l4128_412865


namespace NUMINAMATH_CALUDE_stating_danny_bottle_caps_l4128_412883

/-- Represents the number of bottle caps Danny had initially. -/
def initial_caps : ℕ := 69

/-- Represents the number of bottle caps Danny threw away. -/
def thrown_away_caps : ℕ := 60

/-- Represents the number of new bottle caps Danny found. -/
def new_caps : ℕ := 58

/-- Represents the number of bottle caps Danny has now. -/
def current_caps : ℕ := 67

/-- 
Theorem stating that the initial number of bottle caps minus the thrown away caps,
plus the new caps found, equals the current number of caps.
-/
theorem danny_bottle_caps : 
  initial_caps - thrown_away_caps + new_caps = current_caps := by
  sorry

#check danny_bottle_caps

end NUMINAMATH_CALUDE_stating_danny_bottle_caps_l4128_412883


namespace NUMINAMATH_CALUDE_distance_walked_l4128_412870

/-- Represents the walking pace in miles per hour -/
def pace : ℝ := 4

/-- Represents the time walked in hours -/
def time : ℝ := 2

/-- Theorem stating that the distance walked is the product of pace and time -/
theorem distance_walked : pace * time = 8 := by sorry

end NUMINAMATH_CALUDE_distance_walked_l4128_412870


namespace NUMINAMATH_CALUDE_circle_radius_square_tangents_l4128_412819

theorem circle_radius_square_tangents (side_length : ℝ) (angle : ℝ) (sin_half_angle : ℝ) :
  side_length = Real.sqrt (2 + Real.sqrt 2) →
  angle = π / 4 →
  sin_half_angle = (Real.sqrt (2 - Real.sqrt 2)) / 2 →
  ∃ (radius : ℝ), radius = Real.sqrt 2 + Real.sqrt (2 - Real.sqrt 2) :=
by sorry

end NUMINAMATH_CALUDE_circle_radius_square_tangents_l4128_412819


namespace NUMINAMATH_CALUDE_min_value_of_expression_l4128_412807

def S : Finset Int := {-8, -6, -4, -1, 3, 5, 7, 14}

theorem min_value_of_expression (p q r s t u v w : Int) 
  (h_distinct : p ≠ q ∧ p ≠ r ∧ p ≠ s ∧ p ≠ t ∧ p ≠ u ∧ p ≠ v ∧ p ≠ w ∧
                q ≠ r ∧ q ≠ s ∧ q ≠ t ∧ q ≠ u ∧ q ≠ v ∧ q ≠ w ∧
                r ≠ s ∧ r ≠ t ∧ r ≠ u ∧ r ≠ v ∧ r ≠ w ∧
                s ≠ t ∧ s ≠ u ∧ s ≠ v ∧ s ≠ w ∧
                t ≠ u ∧ t ≠ v ∧ t ≠ w ∧
                u ≠ v ∧ u ≠ w ∧
                v ≠ w)
  (h_in_S : p ∈ S ∧ q ∈ S ∧ r ∈ S ∧ s ∈ S ∧ t ∈ S ∧ u ∈ S ∧ v ∈ S ∧ w ∈ S) :
  ∃ (x : Int), 3 * (p + q + r + s)^2 + (t + u + v + w)^2 ≥ 300 ∧
               3 * x^2 + (20 - x)^2 = 300 :=
by sorry

end NUMINAMATH_CALUDE_min_value_of_expression_l4128_412807


namespace NUMINAMATH_CALUDE_problem_1_problem_2_problem_3_problem_4_l4128_412879

-- Problem 1
theorem problem_1 : -7 + (-3) - 4 - |(-8)| = -22 := by sorry

-- Problem 2
theorem problem_2 : (1/2 - 5/9 + 7/12) * (-36) = -19 := by sorry

-- Problem 3
theorem problem_3 : -3^2 + 16 / (-2) * (1/2) - (-1)^2023 = -14 := by sorry

-- Problem 4
theorem problem_4 (a b : ℝ) : 3*a^2 - 2*a*b - a^2 + 5*a*b = 2*a^2 + 3*a*b := by sorry

end NUMINAMATH_CALUDE_problem_1_problem_2_problem_3_problem_4_l4128_412879


namespace NUMINAMATH_CALUDE_min_value_2x_plus_y_l4128_412805

theorem min_value_2x_plus_y (x y : ℝ) (hx : x > 0) (hy : y > 0) (heq : x * (x + y) = 5 * x + y) :
  ∃ (m : ℝ), m = 9 ∧ ∀ z, z = 2 * x + y → z ≥ m :=
by sorry

end NUMINAMATH_CALUDE_min_value_2x_plus_y_l4128_412805


namespace NUMINAMATH_CALUDE_sin_value_fourth_quadrant_l4128_412891

theorem sin_value_fourth_quadrant (α : Real) (h1 : α ∈ Set.Icc (3 * Real.pi / 2) (2 * Real.pi)) 
  (h2 : Real.tan α = -5/12) : Real.sin α = -5/13 := by
  sorry

end NUMINAMATH_CALUDE_sin_value_fourth_quadrant_l4128_412891


namespace NUMINAMATH_CALUDE_M_subset_N_l4128_412816

def M : Set ℕ := {x | ∃ a : ℕ, x = a^2 + 2*a + 2}
def N : Set ℕ := {y | ∃ b : ℕ, y = b^2 - 4*b + 5}

theorem M_subset_N : M ⊆ N := by sorry

end NUMINAMATH_CALUDE_M_subset_N_l4128_412816


namespace NUMINAMATH_CALUDE_particle_movement_probability_reach_origin_l4128_412826

/-- Probability of reaching (0,0) from (x,y) before hitting any other point on the axes -/
def P (x y : ℕ) : ℚ :=
  if x = 0 ∧ y = 0 then 1
  else if x = 0 ∨ y = 0 then 0
  else (P (x-1) y + P x (y-1) + P (x-1) (y-1)) / 3

/-- The particle's movement rules and starting position -/
theorem particle_movement (x y : ℕ) (h : x > 0 ∧ y > 0) :
  P x y = (P (x-1) y + P x (y-1) + P (x-1) (y-1)) / 3 :=
sorry

/-- The probability of reaching (0,0) from (5,5) -/
theorem probability_reach_origin : P 5 5 = 381 / 2187 :=
sorry

end NUMINAMATH_CALUDE_particle_movement_probability_reach_origin_l4128_412826


namespace NUMINAMATH_CALUDE_power_composition_l4128_412877

theorem power_composition (x a b : ℝ) (h1 : x^a = 2) (h2 : x^b = 3) : x^(3*a + 2*b) = 72 := by
  sorry

end NUMINAMATH_CALUDE_power_composition_l4128_412877


namespace NUMINAMATH_CALUDE_unique_solution_l4128_412842

theorem unique_solution : ∃! (a b c d : ℤ),
  (a^2 - b^2 - c^2 - d^2 = c - b - 2) ∧
  (2*a*b = a - d - 32) ∧
  (2*a*c = 28 - a - d) ∧
  (2*a*d = b + c + 31) ∧
  (a = 5) ∧ (b = -3) ∧ (c = 2) ∧ (d = 3) := by
sorry

end NUMINAMATH_CALUDE_unique_solution_l4128_412842


namespace NUMINAMATH_CALUDE_max_sales_on_day_40_l4128_412824

def salesVolume (t : ℕ) : ℝ := -t + 110

def price (t : ℕ) : ℝ :=
  if t ≤ 40 then t + 8 else -0.5 * t + 69

def salesAmount (t : ℕ) : ℝ := salesVolume t * price t

theorem max_sales_on_day_40 :
  ∀ t : ℕ, 1 ≤ t ∧ t ≤ 100 → salesAmount t ≤ salesAmount 40 ∧ salesAmount 40 = 3360 :=
by sorry

end NUMINAMATH_CALUDE_max_sales_on_day_40_l4128_412824


namespace NUMINAMATH_CALUDE_alpha_beta_range_l4128_412837

theorem alpha_beta_range (α β : ℝ) (h1 : 1 < α) (h2 : α < 3) (h3 : -4 < β) (h4 : β < 2) :
  -12 < α * (-abs β) ∧ α * (-abs β) < -2 := by
  sorry

end NUMINAMATH_CALUDE_alpha_beta_range_l4128_412837


namespace NUMINAMATH_CALUDE_cubic_equation_sum_l4128_412881

theorem cubic_equation_sum (p q r : ℝ) : 
  p^3 - 6*p^2 + 11*p = 12 →
  q^3 - 6*q^2 + 11*q = 12 →
  r^3 - 6*r^2 + 11*r = 12 →
  p * q / r + q * r / p + r * p / q = -23/12 := by
  sorry

end NUMINAMATH_CALUDE_cubic_equation_sum_l4128_412881


namespace NUMINAMATH_CALUDE_largest_prime_factor_of_3401_l4128_412896

theorem largest_prime_factor_of_3401 :
  ∃ (p : ℕ), Nat.Prime p ∧ p ∣ 3401 ∧ ∀ (q : ℕ), Nat.Prime q → q ∣ 3401 → q ≤ p :=
by sorry

end NUMINAMATH_CALUDE_largest_prime_factor_of_3401_l4128_412896


namespace NUMINAMATH_CALUDE_two_digit_number_digit_difference_l4128_412876

/-- 
Given a two-digit number where the difference between the original number 
and the number with interchanged digits is 27, prove that the difference 
between its two digits is 3.
-/
theorem two_digit_number_digit_difference (x y : ℕ) : 
  x < 10 → y < 10 → (10 * x + y) - (10 * y + x) = 27 → x - y = 3 := by
sorry

end NUMINAMATH_CALUDE_two_digit_number_digit_difference_l4128_412876


namespace NUMINAMATH_CALUDE_missing_angle_measure_l4128_412860

/-- A convex polygon with n sides --/
structure ConvexPolygon where
  n : ℕ
  n_ge_3 : n ≥ 3

/-- The sum of interior angles of a convex polygon --/
def interior_angle_sum (p : ConvexPolygon) : ℝ :=
  (p.n - 2) * 180

/-- The theorem to prove --/
theorem missing_angle_measure (p : ConvexPolygon) 
  (sum_without_one : ℝ) 
  (h_sum : sum_without_one = 3025) :
  interior_angle_sum p - sum_without_one = 35 := by
  sorry

end NUMINAMATH_CALUDE_missing_angle_measure_l4128_412860


namespace NUMINAMATH_CALUDE_largest_n_divisibility_l4128_412894

theorem largest_n_divisibility : 
  ∀ n : ℕ, n > 926 → ¬(n + 10 ∣ n^3 + 64) ∧ (926 + 10 ∣ 926^3 + 64) := by
  sorry

end NUMINAMATH_CALUDE_largest_n_divisibility_l4128_412894


namespace NUMINAMATH_CALUDE_january_oil_bill_l4128_412835

/-- Proves that January's oil bill is $120 given the specified conditions --/
theorem january_oil_bill (feb_bill jan_bill : ℚ) : 
  (feb_bill / jan_bill = 3 / 2) → 
  ((feb_bill + 20) / jan_bill = 5 / 3) →
  jan_bill = 120 := by
  sorry

end NUMINAMATH_CALUDE_january_oil_bill_l4128_412835


namespace NUMINAMATH_CALUDE_product_of_sums_equal_difference_of_powers_l4128_412850

theorem product_of_sums_equal_difference_of_powers : 
  (2^1 + 3^1) * (2^2 + 3^2) * (2^4 + 3^4) * (2^8 + 3^8) * 
  (2^16 + 3^16) * (2^32 + 3^32) * (2^64 + 3^64) = 3^128 - 2^128 := by
  sorry

end NUMINAMATH_CALUDE_product_of_sums_equal_difference_of_powers_l4128_412850


namespace NUMINAMATH_CALUDE_sin_2alpha_value_l4128_412882

theorem sin_2alpha_value (α : Real) 
  (h1 : α ∈ Set.Ioo (π / 2) π) 
  (h2 : 3 * Real.cos (2 * α) = Real.sin (π / 4 - α)) : 
  Real.sin (2 * α) = -17 / 18 := by
  sorry

end NUMINAMATH_CALUDE_sin_2alpha_value_l4128_412882


namespace NUMINAMATH_CALUDE_cost_per_bottle_l4128_412854

/-- Given that 3 bottles cost €1.50 and 4 bottles cost €2, prove that the cost per bottle is €0.50 -/
theorem cost_per_bottle (cost_three : ℝ) (cost_four : ℝ) 
  (h1 : cost_three = 1.5) 
  (h2 : cost_four = 2) : 
  cost_three / 3 = 0.5 ∧ cost_four / 4 = 0.5 := by
  sorry


end NUMINAMATH_CALUDE_cost_per_bottle_l4128_412854


namespace NUMINAMATH_CALUDE_factors_of_2_pow_96_minus_1_l4128_412878

theorem factors_of_2_pow_96_minus_1 :
  ∃ (a b : ℕ), 60 < a ∧ a < 70 ∧ 60 < b ∧ b < 70 ∧
  a ≠ b ∧
  (2^96 - 1) % a = 0 ∧ (2^96 - 1) % b = 0 ∧
  (∀ c : ℕ, 60 < c → c < 70 → c ≠ a → c ≠ b → (2^96 - 1) % c ≠ 0) ∧
  a = 63 ∧ b = 65 :=
by sorry

end NUMINAMATH_CALUDE_factors_of_2_pow_96_minus_1_l4128_412878


namespace NUMINAMATH_CALUDE_extreme_value_implies_a_equals_one_l4128_412867

/-- The function f reaching an extreme value at x = 1 implies a = 1 -/
theorem extreme_value_implies_a_equals_one (a : ℝ) :
  let f : ℝ → ℝ := λ x => a * x^2 + 2 * Real.sqrt x - 3 * Real.log x
  (∃ ε > 0, ∀ x ∈ Set.Ioo (1 - ε) (1 + ε), f x ≤ f 1 ∨ f x ≥ f 1) →
  a = 1 := by
  sorry


end NUMINAMATH_CALUDE_extreme_value_implies_a_equals_one_l4128_412867


namespace NUMINAMATH_CALUDE_problem_solution_l4128_412830

def f (a x : ℝ) := |2*x + a| - |2*x + 3|
def g (x : ℝ) := |x - 1| - 3

theorem problem_solution :
  (∀ x : ℝ, |g x| < 2 ↔ (2 < x ∧ x < 6) ∨ (-4 < x ∧ x < 0)) ∧
  (∀ a : ℝ, (∀ x₁ : ℝ, ∃ x₂ : ℝ, f a x₁ = g x₂) ↔ (0 ≤ a ∧ a ≤ 6)) :=
by sorry

end NUMINAMATH_CALUDE_problem_solution_l4128_412830


namespace NUMINAMATH_CALUDE_counterexample_exists_l4128_412898

theorem counterexample_exists : ∃ n : ℕ, 
  (∀ m : ℕ, m * m ≠ n) ∧ ¬(Nat.Prime (n + 4)) := by
  sorry

end NUMINAMATH_CALUDE_counterexample_exists_l4128_412898


namespace NUMINAMATH_CALUDE_water_needed_to_fill_tanks_l4128_412848

/-- Proves that the total amount of water needed to fill three tanks with equal capacity is 1593 liters, 
    given the specified conditions. -/
theorem water_needed_to_fill_tanks (capacity : ℝ) 
  (h1 : capacity * 0.45 = 450)
  (h2 : capacity > 0) : 
  (capacity - 300) + (capacity - 450) + (capacity - (capacity * 0.657)) = 1593 := by
  sorry

end NUMINAMATH_CALUDE_water_needed_to_fill_tanks_l4128_412848


namespace NUMINAMATH_CALUDE_alice_commission_percentage_l4128_412818

/-- Proves that Alice's commission percentage is 2% given her sales, salary, and savings information --/
theorem alice_commission_percentage (sales : ℝ) (basic_salary : ℝ) (savings : ℝ) 
  (h1 : sales = 2500)
  (h2 : basic_salary = 240)
  (h3 : savings = 29)
  (h4 : savings = 0.1 * (basic_salary + sales * commission_rate)) :
  commission_rate = 0.02 := by
  sorry

#check alice_commission_percentage

end NUMINAMATH_CALUDE_alice_commission_percentage_l4128_412818


namespace NUMINAMATH_CALUDE_shooting_range_problem_l4128_412836

theorem shooting_range_problem :
  ∀ n k : ℕ,
  (10 < n) →
  (n < 20) →
  (5 * k = 3 * (n - k)) →
  (n = 16 ∧ k = 6) :=
by sorry

end NUMINAMATH_CALUDE_shooting_range_problem_l4128_412836


namespace NUMINAMATH_CALUDE_jellybean_count_l4128_412800

/-- The number of jellybeans in a bag with black, green, and orange beans -/
def total_jellybeans (black green orange : ℕ) : ℕ := black + green + orange

/-- Theorem: The total number of jellybeans in the bag is 27 -/
theorem jellybean_count :
  ∀ (black green orange : ℕ),
  black = 8 →
  green = black + 2 →
  orange = green - 1 →
  total_jellybeans black green orange = 27 := by
sorry

end NUMINAMATH_CALUDE_jellybean_count_l4128_412800


namespace NUMINAMATH_CALUDE_one_third_of_nine_times_seven_l4128_412831

theorem one_third_of_nine_times_seven : (1 / 3 : ℚ) * (9 * 7) = 21 := by
  sorry

end NUMINAMATH_CALUDE_one_third_of_nine_times_seven_l4128_412831


namespace NUMINAMATH_CALUDE_quadratic_two_distinct_nonnegative_solutions_l4128_412897

theorem quadratic_two_distinct_nonnegative_solutions (a : ℝ) :
  (6 - 3 * a > 0) →
  (a > 0) →
  (3 * a^2 + a - 2 ≥ 0) →
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ x₁ ≥ 0 ∧ x₂ ≥ 0 ∧
    3 * x₁^2 - 3 * a * x₁ + a = 0 ∧
    3 * x₂^2 - 3 * a * x₂ + a = 0) ↔
  (2/3 ≤ a ∧ a < 5/3) ∨ (5/3 < a ∧ a < 2) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_two_distinct_nonnegative_solutions_l4128_412897


namespace NUMINAMATH_CALUDE_square_measurement_error_l4128_412832

theorem square_measurement_error (S : ℝ) (S' : ℝ) (h : S' > 0) :
  S'^2 = S^2 * (1 + 0.0404) → (S' - S) / S * 100 = 2 := by
  sorry

end NUMINAMATH_CALUDE_square_measurement_error_l4128_412832


namespace NUMINAMATH_CALUDE_hannah_kids_stockings_l4128_412828

theorem hannah_kids_stockings (total_stuffers : ℕ) 
  (candy_canes_per_kid : ℕ) (beanie_babies_per_kid : ℕ) (books_per_kid : ℕ) :
  total_stuffers = 21 ∧ 
  candy_canes_per_kid = 4 ∧ 
  beanie_babies_per_kid = 2 ∧ 
  books_per_kid = 1 →
  ∃ (num_kids : ℕ), 
    num_kids * (candy_canes_per_kid + beanie_babies_per_kid + books_per_kid) = total_stuffers ∧
    num_kids = 3 := by
  sorry

end NUMINAMATH_CALUDE_hannah_kids_stockings_l4128_412828


namespace NUMINAMATH_CALUDE_volume_weight_proportion_l4128_412829

/-- Given a substance where volume is directly proportional to weight,
    if 48 cubic inches of the substance weigh 112 ounces,
    then 56 ounces of the substance will have a volume of 24 cubic inches. -/
theorem volume_weight_proportion (volume weight : ℝ → ℝ) :
  (∀ w₁ w₂, volume w₁ / volume w₂ = w₁ / w₂) →  -- volume is directly proportional to weight
  volume 112 = 48 →                            -- 48 cubic inches weigh 112 ounces
  volume 56 = 24                               -- 56 ounces have a volume of 24 cubic inches
:= by sorry

end NUMINAMATH_CALUDE_volume_weight_proportion_l4128_412829


namespace NUMINAMATH_CALUDE_parallelogram_area_example_l4128_412855

/-- The area of a parallelogram with given base and height -/
def parallelogram_area (base height : ℝ) : ℝ := base * height

/-- Theorem: The area of a parallelogram with base 24 cm and height 16 cm is 384 square centimeters -/
theorem parallelogram_area_example : parallelogram_area 24 16 = 384 := by
  sorry

end NUMINAMATH_CALUDE_parallelogram_area_example_l4128_412855


namespace NUMINAMATH_CALUDE_zero_points_of_f_l4128_412814

def f (x : ℝ) := 2 * x^2 + 3 * x + 1

theorem zero_points_of_f :
  ∃ x y : ℝ, x ≠ y ∧ f x = 0 ∧ f y = 0 ∧ 
  ∀ z : ℝ, f z = 0 → z = x ∨ z = y ∧
  x = -1/2 ∧ y = -1 :=
sorry

end NUMINAMATH_CALUDE_zero_points_of_f_l4128_412814


namespace NUMINAMATH_CALUDE_vector_operation_proof_l4128_412851

def a : Fin 2 → ℝ := ![3, 2]
def b : Fin 2 → ℝ := ![0, -1]

theorem vector_operation_proof :
  (3 • b - a) = ![(-3 : ℝ), -5] := by sorry

end NUMINAMATH_CALUDE_vector_operation_proof_l4128_412851


namespace NUMINAMATH_CALUDE_line_parallel_plane_iff_no_common_points_l4128_412873

-- Define a type for points in 3D space
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

-- Define a type for lines in 3D space
structure Line3D where
  point : Point3D
  direction : Point3D

-- Define a type for planes in 3D space
structure Plane3D where
  point : Point3D
  normal : Point3D

-- Define what it means for a line to be parallel to a plane
def is_parallel (l : Line3D) (p : Plane3D) : Prop :=
  l.direction.x * p.normal.x + l.direction.y * p.normal.y + l.direction.z * p.normal.z = 0

-- Define what it means for a line and a plane to have no common points
def no_common_points (l : Line3D) (p : Plane3D) : Prop :=
  ∀ t : ℝ, 
    (l.point.x + t * l.direction.x - p.point.x) * p.normal.x +
    (l.point.y + t * l.direction.y - p.point.y) * p.normal.y +
    (l.point.z + t * l.direction.z - p.point.z) * p.normal.z ≠ 0

-- State the theorem
theorem line_parallel_plane_iff_no_common_points (l : Line3D) (p : Plane3D) :
  is_parallel l p ↔ no_common_points l p :=
sorry

end NUMINAMATH_CALUDE_line_parallel_plane_iff_no_common_points_l4128_412873


namespace NUMINAMATH_CALUDE_roots_sum_l4128_412817

theorem roots_sum (m₁ m₂ : ℝ) : 
  (∃ a b : ℝ, (m₁ * a^2 - (3 * m₁ - 2) * a + 7 = 0) ∧ 
              (m₁ * b^2 - (3 * m₁ - 2) * b + 7 = 0) ∧ 
              (a / b + b / a = 3 / 2)) ∧
  (∃ a b : ℝ, (m₂ * a^2 - (3 * m₂ - 2) * a + 7 = 0) ∧ 
              (m₂ * b^2 - (3 * m₂ - 2) * b + 7 = 0) ∧ 
              (a / b + b / a = 3 / 2)) →
  m₁ + m₂ = 73 / 18 := by
sorry

end NUMINAMATH_CALUDE_roots_sum_l4128_412817


namespace NUMINAMATH_CALUDE_sumAreaVolume_specific_l4128_412880

/-- Represents a point in 3D space with integer coordinates -/
structure Point3D where
  x : Int
  y : Int
  z : Int

/-- Represents a parallelepiped in 3D space -/
structure Parallelepiped where
  base1 : Point3D
  base2 : Point3D
  base3 : Point3D
  base4 : Point3D
  height : Int

/-- Calculates the sum of surface area and volume of a parallelepiped -/
def sumAreaVolume (p : Parallelepiped) : Int :=
  sorry -- Actual calculation would go here

/-- The specific parallelepiped from the problem -/
def specificParallelepiped : Parallelepiped :=
  { base1 := { x := 0, y := 0, z := 0 },
    base2 := { x := 3, y := 4, z := 0 },
    base3 := { x := 7, y := 0, z := 0 },
    base4 := { x := 10, y := 4, z := 0 },
    height := 5 }

theorem sumAreaVolume_specific : sumAreaVolume specificParallelepiped = 365 := by
  sorry

end NUMINAMATH_CALUDE_sumAreaVolume_specific_l4128_412880


namespace NUMINAMATH_CALUDE_sum_of_four_numbers_l4128_412820

theorem sum_of_four_numbers : 2468 + 8642 + 6824 + 4286 = 22220 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_four_numbers_l4128_412820


namespace NUMINAMATH_CALUDE_absolute_difference_of_solution_l4128_412823

theorem absolute_difference_of_solution (x y : ℝ) : 
  (Int.floor x + (y - Int.floor y) = 3.7) →
  ((x - Int.floor x) + Int.floor y = 6.2) →
  |x - y| = 3.5 := by
sorry

end NUMINAMATH_CALUDE_absolute_difference_of_solution_l4128_412823


namespace NUMINAMATH_CALUDE_cos_alpha_values_l4128_412802

theorem cos_alpha_values (α : Real) (h : Real.sin (Real.pi + α) = -3/5) :
  Real.cos α = 4/5 ∨ Real.cos α = -4/5 := by
  sorry

end NUMINAMATH_CALUDE_cos_alpha_values_l4128_412802


namespace NUMINAMATH_CALUDE_students_in_sports_l4128_412872

theorem students_in_sports (total : ℕ) (basketball soccer baseball cricket : ℕ)
  (basketball_soccer basketball_baseball basketball_cricket : ℕ)
  (soccer_baseball cricket_soccer cricket_baseball : ℕ)
  (basketball_cricket_soccer : ℕ) (no_sport : ℕ)
  (h1 : total = 200)
  (h2 : basketball = 50)
  (h3 : soccer = 60)
  (h4 : baseball = 35)
  (h5 : cricket = 80)
  (h6 : basketball_soccer = 10)
  (h7 : basketball_baseball = 15)
  (h8 : basketball_cricket = 20)
  (h9 : soccer_baseball = 25)
  (h10 : cricket_soccer = 30)
  (h11 : cricket_baseball = 5)
  (h12 : basketball_cricket_soccer = 10)
  (h13 : no_sport = 30) :
  basketball + soccer + baseball + cricket -
  basketball_soccer - basketball_baseball - basketball_cricket -
  soccer_baseball - cricket_soccer - cricket_baseball +
  basketball_cricket_soccer = 130 := by
  sorry

end NUMINAMATH_CALUDE_students_in_sports_l4128_412872


namespace NUMINAMATH_CALUDE_sum_three_x_square_y_correct_l4128_412825

/-- The sum of three times x and the square of y -/
def sum_three_x_square_y (x y : ℝ) : ℝ := 3 * x + y^2

theorem sum_three_x_square_y_correct (x y : ℝ) : 
  sum_three_x_square_y x y = 3 * x + y^2 := by
  sorry

end NUMINAMATH_CALUDE_sum_three_x_square_y_correct_l4128_412825


namespace NUMINAMATH_CALUDE_largest_tile_size_l4128_412849

-- Define the courtyard dimensions in centimeters
def courtyard_length : ℕ := 378
def courtyard_width : ℕ := 525

-- Define the tile size in centimeters
def tile_size : ℕ := 21

-- Theorem statement
theorem largest_tile_size :
  (courtyard_length % tile_size = 0) ∧
  (courtyard_width % tile_size = 0) ∧
  (∀ s : ℕ, s > tile_size →
    (courtyard_length % s ≠ 0) ∨ (courtyard_width % s ≠ 0)) :=
by sorry

end NUMINAMATH_CALUDE_largest_tile_size_l4128_412849


namespace NUMINAMATH_CALUDE_equation_solutions_l4128_412846

theorem equation_solutions :
  (∀ x : ℝ, x^2 - 25 = 0 ↔ x = 5 ∨ x = -5) ∧
  (∀ x : ℝ, 8 * (x - 1)^3 = 27 ↔ x = 5/2) := by
  sorry

end NUMINAMATH_CALUDE_equation_solutions_l4128_412846


namespace NUMINAMATH_CALUDE_complex_roots_theorem_l4128_412895

theorem complex_roots_theorem (x y : ℝ) : 
  (Complex.I : ℂ).re = 0 ∧ (Complex.I : ℂ).im = 1 →
  (x + 3 * Complex.I) * (x + 3 * Complex.I) - (13 + 12 * Complex.I) * (x + 3 * Complex.I) + (15 + 72 * Complex.I) = 0 ∧
  (y + 6 * Complex.I) * (y + 6 * Complex.I) - (13 + 12 * Complex.I) * (y + 6 * Complex.I) + (15 + 72 * Complex.I) = 0 →
  x = 11 ∧ y = 2 := by
sorry

end NUMINAMATH_CALUDE_complex_roots_theorem_l4128_412895


namespace NUMINAMATH_CALUDE_inequality_solution_l4128_412890

theorem inequality_solution (x : ℝ) : 
  (x^2 + x^3 - x^4) / (x + x^2 - x^3) ≥ -1 ↔ 
  (x ∈ Set.Icc (-1) ((1 - Real.sqrt 5) / 2) ∪ 
   Set.Ioo ((1 - Real.sqrt 5) / 2) 0 ∪ 
   Set.Ioo 0 ((1 + Real.sqrt 5) / 2) ∪ 
   Set.Ioi ((1 + Real.sqrt 5) / 2)) := by
sorry

end NUMINAMATH_CALUDE_inequality_solution_l4128_412890


namespace NUMINAMATH_CALUDE_parabola_line_intersection_l4128_412886

/-- Parabola defined by y² = 8x -/
def Parabola (x y : ℝ) : Prop := y^2 = 8*x

/-- Focus of the parabola -/
def Focus : ℝ × ℝ := (2, 0)

/-- Line passing through the focus with slope k -/
def Line (k : ℝ) (x y : ℝ) : Prop := y = k*(x - 2)

/-- Point M -/
def M : ℝ × ℝ := (-2, 2)

/-- Intersection points of the line and the parabola -/
def Intersects (k : ℝ) (A B : ℝ × ℝ) : Prop :=
  Parabola A.1 A.2 ∧ Parabola B.1 B.2 ∧
  Line k A.1 A.2 ∧ Line k B.1 B.2

/-- Vector dot product -/
def DotProduct (v w : ℝ × ℝ) : ℝ :=
  v.1 * w.1 + v.2 * w.2

theorem parabola_line_intersection (k : ℝ) :
  ∃ A B : ℝ × ℝ, Intersects k A B ∧
  DotProduct (A.1 + 2, A.2 - 2) (B.1 + 2, B.2 - 2) = 0 →
  k = 2 :=
sorry

end NUMINAMATH_CALUDE_parabola_line_intersection_l4128_412886


namespace NUMINAMATH_CALUDE_largest_solution_of_equation_l4128_412844

theorem largest_solution_of_equation :
  let f : ℝ → ℝ := λ x => (3*x)/7 + 2/(7*x)
  ∃ x : ℝ, x > 0 ∧ f x = 3/4 ∧ ∀ y : ℝ, y > 0 → f y = 3/4 → y ≤ x ∧ x = (21 + Real.sqrt 345) / 12 :=
by sorry

end NUMINAMATH_CALUDE_largest_solution_of_equation_l4128_412844


namespace NUMINAMATH_CALUDE_max_value_of_expression_l4128_412810

theorem max_value_of_expression (x : ℝ) (h : -1 ≤ x ∧ x ≤ 2) :
  ∃ (max : ℝ), max = 5 ∧ ∀ y, -1 ≤ y ∧ y ≤ 2 → 2 + |y - 2| ≤ max :=
by sorry

end NUMINAMATH_CALUDE_max_value_of_expression_l4128_412810


namespace NUMINAMATH_CALUDE_man_downstream_speed_l4128_412840

/-- Calculates the downstream speed given upstream and still water speeds -/
def downstream_speed (upstream_speed still_water_speed : ℝ) : ℝ :=
  2 * still_water_speed - upstream_speed

/-- Theorem stating that given the specified upstream and still water speeds, 
    the downstream speed is 80 kmph -/
theorem man_downstream_speed :
  downstream_speed 20 50 = 80 := by
  sorry

end NUMINAMATH_CALUDE_man_downstream_speed_l4128_412840


namespace NUMINAMATH_CALUDE_kylie_picks_220_apples_l4128_412858

/-- The number of apples Kylie picks in the first hour -/
def first_hour_apples : ℕ := 66

/-- The number of apples Kylie picks in the second hour -/
def second_hour_apples : ℕ := 2 * first_hour_apples

/-- The number of apples Kylie picks in the third hour -/
def third_hour_apples : ℕ := first_hour_apples / 3

/-- The total number of apples Kylie picks over three hours -/
def total_apples : ℕ := first_hour_apples + second_hour_apples + third_hour_apples

/-- Theorem stating that the total number of apples Kylie picks is 220 -/
theorem kylie_picks_220_apples : total_apples = 220 := by
  sorry

end NUMINAMATH_CALUDE_kylie_picks_220_apples_l4128_412858


namespace NUMINAMATH_CALUDE_arithmetic_progression_problem_l4128_412821

theorem arithmetic_progression_problem (a d : ℝ) : 
  (a - 2*d)^3 + (a - d)^3 + a^3 + (a + d)^3 + (a + 2*d)^3 = 0 ∧
  (a - 2*d)^4 + (a - d)^4 + a^4 + (a + d)^4 + (a + 2*d)^4 = 136 →
  a - 2*d = -2 * Real.sqrt 2 :=
by sorry

end NUMINAMATH_CALUDE_arithmetic_progression_problem_l4128_412821


namespace NUMINAMATH_CALUDE_shooting_competition_sequences_l4128_412861

/-- The number of ways to arrange a multiset with 4 A's, 3 B's, 2 C's, and 1 D -/
def shooting_sequences : ℕ := 12600

/-- The total number of targets -/
def total_targets : ℕ := 10

/-- The number of targets in column A -/
def targets_A : ℕ := 4

/-- The number of targets in column B -/
def targets_B : ℕ := 3

/-- The number of targets in column C -/
def targets_C : ℕ := 2

/-- The number of targets in column D -/
def targets_D : ℕ := 1

theorem shooting_competition_sequences :
  shooting_sequences = (total_targets.factorial) / 
    (targets_A.factorial * targets_B.factorial * 
     targets_C.factorial * targets_D.factorial) :=
by sorry

end NUMINAMATH_CALUDE_shooting_competition_sequences_l4128_412861


namespace NUMINAMATH_CALUDE_race_finish_difference_l4128_412888

/-- The time difference between two runners in a race -/
def time_difference (race_distance : ℕ) (speed1 speed2 : ℕ) : ℕ :=
  race_distance * speed2 - race_distance * speed1

/-- Theorem: In a 12-mile race, a runner with 7 min/mile speed finishes 24 minutes 
    after a runner with 5 min/mile speed -/
theorem race_finish_difference :
  time_difference 12 5 7 = 24 := by sorry

end NUMINAMATH_CALUDE_race_finish_difference_l4128_412888


namespace NUMINAMATH_CALUDE_vs_length_l4128_412857

/-- A square piece of paper PQRS with side length 8 cm is folded so that corner R 
    coincides with T, the midpoint of PS. The crease UV intersects RS at V. -/
structure FoldedSquare where
  /-- Side length of the square -/
  side_length : ℝ
  /-- Point P -/
  P : ℝ × ℝ
  /-- Point Q -/
  Q : ℝ × ℝ
  /-- Point R -/
  R : ℝ × ℝ
  /-- Point S -/
  S : ℝ × ℝ
  /-- Point T (midpoint of PS) -/
  T : ℝ × ℝ
  /-- Point V (intersection of UV and RS) -/
  V : ℝ × ℝ
  /-- PQRS forms a square with side length 8 -/
  square_constraint : 
    P.1 = Q.1 ∧ Q.2 = R.2 ∧ R.1 = S.1 ∧ S.2 = P.2 ∧
    (Q.1 - P.1)^2 + (Q.2 - P.2)^2 = side_length^2 ∧
    side_length = 8
  /-- T is the midpoint of PS -/
  midpoint_constraint : T = ((P.1 + S.1) / 2, (P.2 + S.2) / 2)
  /-- V is on RS -/
  v_on_rs : ∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ V = (R.1 * (1 - t) + S.1 * t, R.2 * (1 - t) + S.2 * t)
  /-- Distance RV equals distance TV (fold constraint) -/
  fold_constraint : (R.1 - V.1)^2 + (R.2 - V.2)^2 = (T.1 - V.1)^2 + (T.2 - V.2)^2

/-- The length of VS in the folded square is 3 cm -/
theorem vs_length (fs : FoldedSquare) : 
  ((fs.V.1 - fs.S.1)^2 + (fs.V.2 - fs.S.2)^2)^(1/2) = 3 := by
  sorry

end NUMINAMATH_CALUDE_vs_length_l4128_412857


namespace NUMINAMATH_CALUDE_subset_implies_C_C_complete_l4128_412815

def A (a : ℝ) : Set ℝ := {x | a * x - 2 = 0}
def B : Set ℝ := {x | x^2 - 3*x + 2 = 0}
def C : Set ℝ := {1, 2}

theorem subset_implies_C (a : ℝ) (h : A a ⊆ B) : a ∈ C := by
  sorry

theorem C_complete : ∀ a ∈ C, A a ⊆ B := by
  sorry

end NUMINAMATH_CALUDE_subset_implies_C_C_complete_l4128_412815


namespace NUMINAMATH_CALUDE_parallel_vectors_subtraction_l4128_412892

/-- Given vectors a and b where a is parallel to b, prove that 2a - b = (4, -8) -/
theorem parallel_vectors_subtraction (m : ℝ) : 
  let a : Fin 2 → ℝ := ![1, -2]
  let b : Fin 2 → ℝ := ![m, 4]
  (∃ (k : ℝ), a = k • b) →
  (2 • a - b) = ![4, -8] := by
sorry

end NUMINAMATH_CALUDE_parallel_vectors_subtraction_l4128_412892


namespace NUMINAMATH_CALUDE_white_balls_count_l4128_412847

/-- Given a bag with 10 balls where the probability of drawing a white ball is 30%,
    prove that the number of white balls in the bag is 3. -/
theorem white_balls_count (total_balls : ℕ) (prob_white : ℚ) (white_balls : ℕ) :
  total_balls = 10 →
  prob_white = 3/10 →
  white_balls = (total_balls : ℚ) * prob_white →
  white_balls = 3 :=
by sorry

end NUMINAMATH_CALUDE_white_balls_count_l4128_412847


namespace NUMINAMATH_CALUDE_problem_1997_2000_l4128_412887

theorem problem_1997_2000 : 1997 * (2000 / 2000) - 2000 * (1997 / 1997) = 0 := by
  sorry

end NUMINAMATH_CALUDE_problem_1997_2000_l4128_412887


namespace NUMINAMATH_CALUDE_rectangle_width_length_ratio_l4128_412864

theorem rectangle_width_length_ratio (w : ℝ) : 
  w > 0 ∧ 2 * w + 2 * 10 = 30 → w / 10 = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_width_length_ratio_l4128_412864


namespace NUMINAMATH_CALUDE_ball_count_in_bag_l4128_412812

/-- Given a bag with red, black, and white balls, prove that the total number of balls is 7
    when the probability of drawing a red ball equals the probability of drawing a white ball. -/
theorem ball_count_in_bag (x : ℕ) : 
  (3 : ℚ) / (4 + x) = (x : ℚ) / (4 + x) → 3 + 1 + x = 7 := by
  sorry

end NUMINAMATH_CALUDE_ball_count_in_bag_l4128_412812


namespace NUMINAMATH_CALUDE_nearest_integer_to_power_l4128_412839

theorem nearest_integer_to_power : ∃ n : ℤ, 
  n = 376 ∧ ∀ m : ℤ, |((3 : ℝ) + Real.sqrt 5)^4 - (n : ℝ)| ≤ |((3 : ℝ) + Real.sqrt 5)^4 - (m : ℝ)| := by
  sorry

end NUMINAMATH_CALUDE_nearest_integer_to_power_l4128_412839


namespace NUMINAMATH_CALUDE_survey_selection_count_l4128_412811

/-- Represents the total number of households selected in a stratified sampling survey. -/
def total_selected (total_households : ℕ) (middle_income : ℕ) (low_income : ℕ) (high_income_selected : ℕ) : ℕ :=
  (high_income_selected * total_households) / (total_households - middle_income - low_income)

/-- Theorem stating that the total number of households selected in the survey is 24. -/
theorem survey_selection_count :
  total_selected 480 200 160 6 = 24 := by
  sorry

end NUMINAMATH_CALUDE_survey_selection_count_l4128_412811


namespace NUMINAMATH_CALUDE_cos_96_cos_24_minus_sin_96_sin_24_l4128_412853

theorem cos_96_cos_24_minus_sin_96_sin_24 : 
  Real.cos (96 * π / 180) * Real.cos (24 * π / 180) - 
  Real.sin (96 * π / 180) * Real.sin (24 * π / 180) = -1/2 := by
  sorry

end NUMINAMATH_CALUDE_cos_96_cos_24_minus_sin_96_sin_24_l4128_412853


namespace NUMINAMATH_CALUDE_incorrect_factorization_l4128_412866

theorem incorrect_factorization (x : ℝ) : x^2 + x - 2 ≠ (x - 2) * (x + 1) := by
  sorry

end NUMINAMATH_CALUDE_incorrect_factorization_l4128_412866


namespace NUMINAMATH_CALUDE_hyperbola_eccentricity_sqrt_6_l4128_412852

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a parabola with equation y² = 4x -/
def Parabola := {p : Point | p.y^2 = 4 * p.x}

/-- Represents a hyperbola with equation x²/a² - y² = 1 -/
def Hyperbola (a : ℝ) := {p : Point | p.x^2 / a^2 - p.y^2 = 1}

/-- The directrix of the parabola y² = 4x -/
def directrix : Set Point := {p : Point | p.x = -1}

/-- The focus of the parabola y² = 4x -/
def focus : Point := ⟨1, 0⟩

/-- Predicate to check if three points form a right-angled triangle -/
def isRightTriangle (p q r : Point) : Prop := sorry

/-- The eccentricity of a hyperbola -/
def hyperbolaEccentricity (a : ℝ) : ℝ := sorry

theorem hyperbola_eccentricity_sqrt_6 (a : ℝ) (A B : Point) :
  A ∈ Hyperbola a →
  B ∈ Hyperbola a →
  A ∈ directrix →
  B ∈ directrix →
  isRightTriangle A B focus →
  hyperbolaEccentricity a = Real.sqrt 6 := by
  sorry

end NUMINAMATH_CALUDE_hyperbola_eccentricity_sqrt_6_l4128_412852


namespace NUMINAMATH_CALUDE_moving_circle_trajectory_l4128_412801

/-- The equation of the fixed circle -/
def fixed_circle (x y : ℝ) : Prop := x^2 + y^2 - 4*x = 0

/-- The equation of the y-axis -/
def y_axis (x : ℝ) : Prop := x = 0

/-- The trajectory of the center of the moving circle -/
def trajectory (x y : ℝ) : Prop := (x > 0 ∧ y^2 = 8*x) ∨ (x ≤ 0 ∧ y = 0)

/-- Theorem stating the trajectory of the center of the moving circle -/
theorem moving_circle_trajectory :
  ∀ (x y : ℝ), 
  (∃ (r : ℝ), r > 0 ∧ 
    (∃ (x₀ y₀ : ℝ), fixed_circle x₀ y₀ ∧ (x - x₀)^2 + (y - y₀)^2 = r^2) ∧
    (∃ (x₁ : ℝ), y_axis x₁ ∧ (x - x₁)^2 + y^2 = r^2)) →
  trajectory x y :=
sorry

end NUMINAMATH_CALUDE_moving_circle_trajectory_l4128_412801


namespace NUMINAMATH_CALUDE_tangent_line_condition_l4128_412863

-- Define the curves
def curve1 (x : ℝ) : ℝ := x^3
def curve2 (a x : ℝ) : ℝ := a*x^2 + x - 9

-- Define the tangent line condition
def is_tangent_to_both (a : ℝ) : Prop :=
  ∃ (m : ℝ), ∃ (x₀ : ℝ),
    -- The line passes through (1,0)
    m * (1 - x₀) = -curve1 x₀ ∧
    -- The line is tangent to y = x^3
    m = 3 * x₀^2 ∧
    -- The line is tangent to y = ax^2 + x - 9
    m = 2 * a * x₀ + 1 ∧
    -- The point (x₀, curve1 x₀) is on both curves
    curve1 x₀ = curve2 a x₀

-- The main theorem
theorem tangent_line_condition (a : ℝ) :
  is_tangent_to_both a → a = -1 ∨ a = -7 :=
sorry

end NUMINAMATH_CALUDE_tangent_line_condition_l4128_412863


namespace NUMINAMATH_CALUDE_perpendicular_vectors_l4128_412845

theorem perpendicular_vectors (x : ℝ) :
  let a : Fin 2 → ℝ := ![1, -2]
  let b : Fin 2 → ℝ := ![-2, x]
  (∀ i, i < 2 → a i * b i = 0) → x = -1 := by
sorry

end NUMINAMATH_CALUDE_perpendicular_vectors_l4128_412845


namespace NUMINAMATH_CALUDE_disaster_relief_team_selection_l4128_412808

/-- The number of internal medicine doctors -/
def internal_doctors : ℕ := 12

/-- The number of surgeons -/
def surgeons : ℕ := 8

/-- The size of the disaster relief medical team -/
def team_size : ℕ := 5

/-- Doctor A is an internal medicine doctor -/
def doctor_A : Fin internal_doctors := sorry

/-- Doctor B is a surgeon -/
def doctor_B : Fin surgeons := sorry

/-- The number of ways to select 5 doctors including A and B -/
def selection_with_A_and_B : ℕ := sorry

/-- The number of ways to select 5 doctors excluding both A and B -/
def selection_without_A_and_B : ℕ := sorry

/-- The number of ways to select 5 doctors including at least one of A or B -/
def selection_with_A_or_B : ℕ := sorry

/-- The number of ways to select 5 doctors with at least one internal medicine doctor and one surgeon -/
def selection_with_both_specialties : ℕ := sorry

theorem disaster_relief_team_selection :
  selection_with_A_and_B = 816 ∧
  selection_without_A_and_B = 8568 ∧
  selection_with_A_or_B = 6936 ∧
  selection_with_both_specialties = 14656 := by sorry

end NUMINAMATH_CALUDE_disaster_relief_team_selection_l4128_412808


namespace NUMINAMATH_CALUDE_triangle_side_and_area_l4128_412806

/-- Given a triangle ABC with side lengths a, b, c and angle A, prove the length of side a and the area of the triangle. -/
theorem triangle_side_and_area 
  (b c : ℝ) 
  (A : ℝ) 
  (hb : b = 4) 
  (hc : c = 2) 
  (hA : Real.cos A = 1/4) :
  ∃ (a : ℝ), 
    a = 4 ∧ 
    (1/2 * b * c * Real.sin A : ℝ) = Real.sqrt 15 :=
by sorry

end NUMINAMATH_CALUDE_triangle_side_and_area_l4128_412806


namespace NUMINAMATH_CALUDE_binary_sum_equals_1945_l4128_412841

/-- Converts a binary number represented as a list of bits to its decimal equivalent -/
def binary_to_decimal (bits : List Bool) : Nat :=
  bits.enum.foldl (fun acc (i, b) => acc + if b then 2^i else 0) 0

theorem binary_sum_equals_1945 :
  let num1 := binary_to_decimal [true, true, true, true, true, true, true, true, true, true]
  let num2 := binary_to_decimal [false, true, false, true, false, true, false, true, false, true]
  let num3 := binary_to_decimal [false, false, false, false, true, true, true, true]
  num1 + num2 + num3 = 1945 := by
  sorry

end NUMINAMATH_CALUDE_binary_sum_equals_1945_l4128_412841


namespace NUMINAMATH_CALUDE_complex_magnitude_l4128_412889

/-- Given that (1+2i)/(a+bi) = 1 - i, where i is the imaginary unit and a and b are real numbers,
    prove that |a+bi| = √10/2 -/
theorem complex_magnitude (a b : ℝ) (i : ℂ) (h1 : i^2 = -1) 
  (h2 : (1 + 2*i) / (a + b*i) = 1 - i) : 
  Complex.abs (a + b*i) = Real.sqrt 10 / 2 := by
  sorry

end NUMINAMATH_CALUDE_complex_magnitude_l4128_412889


namespace NUMINAMATH_CALUDE_acid_mixture_percentage_l4128_412834

theorem acid_mixture_percentage (a w : ℚ) :
  a > 0 ∧ w > 0 →
  (a + 1) / (a + w + 1) = 1/4 →
  (a + 1) / (a + w + 2) = 1/5 →
  a / (a + w) = 2/11 :=
by sorry

end NUMINAMATH_CALUDE_acid_mixture_percentage_l4128_412834


namespace NUMINAMATH_CALUDE_square_gt_of_abs_gt_l4128_412884

theorem square_gt_of_abs_gt (a b : ℝ) : a > |b| → a^2 > b^2 := by
  sorry

end NUMINAMATH_CALUDE_square_gt_of_abs_gt_l4128_412884


namespace NUMINAMATH_CALUDE_second_set_cost_l4128_412856

/-- The cost of a set of footballs and soccer balls -/
def cost_of_set (football_price : ℝ) (soccer_price : ℝ) (num_footballs : ℕ) (num_soccers : ℕ) : ℝ :=
  football_price * (num_footballs : ℝ) + soccer_price * (num_soccers : ℝ)

/-- The theorem stating the cost of the second set of balls -/
theorem second_set_cost :
  ∀ (football_price : ℝ),
  cost_of_set football_price 50 3 1 = 155 →
  cost_of_set football_price 50 2 3 = 220 :=
by
  sorry

end NUMINAMATH_CALUDE_second_set_cost_l4128_412856


namespace NUMINAMATH_CALUDE_percentage_of_non_science_majors_l4128_412822

theorem percentage_of_non_science_majors
  (women_science_percentage : Real)
  (men_class_percentage : Real)
  (men_science_percentage : Real)
  (h1 : women_science_percentage = 0.1)
  (h2 : men_class_percentage = 0.4)
  (h3 : men_science_percentage = 0.8500000000000001) :
  1 - (women_science_percentage * (1 - men_class_percentage) +
       men_science_percentage * men_class_percentage) = 0.6 := by
  sorry

end NUMINAMATH_CALUDE_percentage_of_non_science_majors_l4128_412822


namespace NUMINAMATH_CALUDE_total_digits_of_powers_l4128_412833

theorem total_digits_of_powers : ∃ m n : ℕ,
  (10^(m-1) < 2^2019 ∧ 2^2019 < 10^m) ∧
  (10^(n-1) < 5^2019 ∧ 5^2019 < 10^n) ∧
  m + n = 2020 :=
by sorry

end NUMINAMATH_CALUDE_total_digits_of_powers_l4128_412833


namespace NUMINAMATH_CALUDE_solutions_equation1_solutions_equation2_l4128_412893

-- Define the equations
def equation1 (x : ℝ) : Prop := x^2 - 10*x + 16 = 0
def equation2 (x : ℝ) : Prop := 2*x*(x-1) = x-1

-- Theorem for the first equation
theorem solutions_equation1 : 
  ∀ x : ℝ, equation1 x ↔ (x = 2 ∨ x = 8) :=
by sorry

-- Theorem for the second equation
theorem solutions_equation2 : 
  ∀ x : ℝ, equation2 x ↔ (x = 1 ∨ x = 1/2) :=
by sorry

end NUMINAMATH_CALUDE_solutions_equation1_solutions_equation2_l4128_412893


namespace NUMINAMATH_CALUDE_inequality_constraint_l4128_412871

theorem inequality_constraint (m : ℝ) : 
  (∀ x : ℝ, x ∈ Set.Icc 0 1 → x^2 - 4*x ≥ m) → m ≤ -3 := by
  sorry

end NUMINAMATH_CALUDE_inequality_constraint_l4128_412871


namespace NUMINAMATH_CALUDE_circle_bisection_l4128_412869

/-- A circle in the xy-plane -/
structure Circle where
  equation : ℝ → ℝ → Prop

/-- A line in the xy-plane -/
structure Line where
  equation : ℝ → ℝ → Prop

/-- The center of a circle -/
def Circle.center (c : Circle) : ℝ × ℝ := sorry

/-- A line bisects a circle if it passes through the circle's center -/
def bisects (l : Line) (c : Circle) : Prop :=
  let (x₀, y₀) := c.center
  l.equation x₀ y₀

/-- The main theorem -/
theorem circle_bisection (c : Circle) (l : Line) (a : ℝ) :
  c.equation = (fun x y ↦ x^2 + y^2 + 2*x - 4*y = 0) →
  l.equation = (fun x y ↦ 3*x + y + a = 0) →
  bisects l c →
  a = 1 := by sorry

end NUMINAMATH_CALUDE_circle_bisection_l4128_412869


namespace NUMINAMATH_CALUDE_triangle_area_l4128_412868

theorem triangle_area (a b c : ℝ) (h_a : a = 39) (h_b : b = 36) (h_c : c = 15) :
  let s := (a + b + c) / 2
  Real.sqrt (s * (s - a) * (s - b) * (s - c)) = 270 :=
by sorry

end NUMINAMATH_CALUDE_triangle_area_l4128_412868


namespace NUMINAMATH_CALUDE_light_pulse_reflections_l4128_412862

theorem light_pulse_reflections :
  ∃ (n : ℕ), n > 0 ∧
  (∃ (a b : ℕ), (a + 2) * (b + 2) = 4042 ∧ Nat.gcd (a + 1) (b + 1) = 1 ∧ n = a + b) ∧
  (∀ (m : ℕ), m > 0 →
    (∃ (a b : ℕ), (a + 2) * (b + 2) = 4042 ∧ Nat.gcd (a + 1) (b + 1) = 1 ∧ m = a + b) →
    m ≥ n) ∧
  n = 129 :=
by sorry

end NUMINAMATH_CALUDE_light_pulse_reflections_l4128_412862


namespace NUMINAMATH_CALUDE_reciprocal_power_2014_l4128_412827

theorem reciprocal_power_2014 (a : ℚ) (h : a ≠ 0) : (a = a⁻¹) → a^2014 = 1 := by
  sorry

end NUMINAMATH_CALUDE_reciprocal_power_2014_l4128_412827


namespace NUMINAMATH_CALUDE_coin_problem_l4128_412843

/-- Represents the number and denomination of coins -/
structure CoinCount where
  twenties : Nat
  fifteens : Nat

/-- Calculates the total value of coins in kopecks -/
def totalValue (coins : CoinCount) : Nat :=
  20 * coins.twenties + 15 * coins.fifteens

/-- Represents the conditions of the problem -/
structure ProblemConditions where
  coins : CoinCount
  moreTwenties : coins.twenties > coins.fifteens
  fifthSpentWithTwoCoins : ∃ (a b : Nat), a + b = 2 ∧ 
    (a * 20 + b * 15 = totalValue coins / 5)
  halfRemainingSpentWithThreeCoins : ∃ (c d : Nat), c + d = 3 ∧ 
    (c * 20 + d * 15 = (4 * totalValue coins / 5) / 2)

/-- The theorem to be proved -/
theorem coin_problem (conditions : ProblemConditions) : 
  conditions.coins = CoinCount.mk 6 2 := by
  sorry


end NUMINAMATH_CALUDE_coin_problem_l4128_412843
