import Mathlib

namespace NUMINAMATH_CALUDE_average_first_14_even_numbers_l738_73870

theorem average_first_14_even_numbers :
  let first_14_even : List ℕ := List.range 14 |>.map (fun n => 2 * (n + 1))
  (first_14_even.sum / first_14_even.length : ℚ) = 15 := by
  sorry

end NUMINAMATH_CALUDE_average_first_14_even_numbers_l738_73870


namespace NUMINAMATH_CALUDE_intersection_when_m_is_2_sufficient_not_necessary_condition_l738_73860

def p (x : ℝ) := x^2 + 2*x - 8 < 0

def q (x m : ℝ) := (x - 1 + m)*(x - 1 - m) ≤ 0

def A := {x : ℝ | p x}

def B (m : ℝ) := {x : ℝ | q x m}

theorem intersection_when_m_is_2 :
  B 2 ∩ A = {x : ℝ | -1 ≤ x ∧ x < 2} :=
sorry

theorem sufficient_not_necessary_condition (m : ℝ) :
  (∀ x : ℝ, p x → q x m) ∧ (∃ x : ℝ, q x m ∧ ¬p x) ↔ m ≥ 5 :=
sorry

end NUMINAMATH_CALUDE_intersection_when_m_is_2_sufficient_not_necessary_condition_l738_73860


namespace NUMINAMATH_CALUDE_sara_gave_dan_28_pears_l738_73809

/-- The number of pears Sara initially picked -/
def initial_pears : ℕ := 35

/-- The number of pears Sara has left -/
def remaining_pears : ℕ := 7

/-- The number of pears Sara gave to Dan -/
def pears_given_to_dan : ℕ := initial_pears - remaining_pears

theorem sara_gave_dan_28_pears : pears_given_to_dan = 28 := by
  sorry

end NUMINAMATH_CALUDE_sara_gave_dan_28_pears_l738_73809


namespace NUMINAMATH_CALUDE_f_extrema_and_functional_equation_l738_73892

noncomputable def f (x : ℝ) := Real.sin x + Real.sqrt 3 * Real.cos x + 1

theorem f_extrema_and_functional_equation :
  (∃ (xmax xmin : ℝ), xmax ∈ Set.Icc 0 (Real.pi / 2) ∧ xmin ∈ Set.Icc 0 (Real.pi / 2) ∧
    (∀ x, x ∈ Set.Icc 0 (Real.pi / 2) → f x ≤ f xmax) ∧
    (∀ x, x ∈ Set.Icc 0 (Real.pi / 2) → f xmin ≤ f x) ∧
    f xmax = 3 ∧ f xmin = 2) ∧
  (∃ (a b c : ℝ), (∀ x, a * f x + b * f (x - c) = 1) → b * Real.cos c / a = -1) :=
by sorry

end NUMINAMATH_CALUDE_f_extrema_and_functional_equation_l738_73892


namespace NUMINAMATH_CALUDE_scientific_notation_of_2150_l738_73839

def scientific_notation (x : ℝ) : ℝ × ℤ :=
  sorry

theorem scientific_notation_of_2150 :
  scientific_notation 2150 = (2.15, 3) := by
  sorry

end NUMINAMATH_CALUDE_scientific_notation_of_2150_l738_73839


namespace NUMINAMATH_CALUDE_absolute_value_equality_l738_73816

theorem absolute_value_equality (x : ℝ) : |x + 6| = -(x + 6) ↔ x ≤ -6 := by
  sorry

end NUMINAMATH_CALUDE_absolute_value_equality_l738_73816


namespace NUMINAMATH_CALUDE_bellas_to_annes_height_ratio_l738_73862

/-- Proves that given the conditions in the problem, the ratio of Bella's height to Anne's height is 3:1 -/
theorem bellas_to_annes_height_ratio : 
  ∀ (anne_height bella_height sister_height : ℝ),
  anne_height = 2 * sister_height →
  anne_height = 80 →
  bella_height - sister_height = 200 →
  bella_height / anne_height = 3 := by
sorry

end NUMINAMATH_CALUDE_bellas_to_annes_height_ratio_l738_73862


namespace NUMINAMATH_CALUDE_distance_between_4th_and_28th_red_lights_l738_73873

/-- Represents the color of a light -/
inductive LightColor
| Blue
| Red

/-- Calculates the position of the nth red light in the sequence -/
def redLightPosition (n : ℕ) : ℕ :=
  let groupSize := 5
  let redLightsPerGroup := 3
  let completeGroups := (n - 1) / redLightsPerGroup
  let positionInGroup := (n - 1) % redLightsPerGroup + 1
  completeGroups * groupSize + positionInGroup + 2

/-- The distance between lights in inches -/
def lightDistance : ℕ := 8

/-- The number of inches in a foot -/
def inchesPerFoot : ℕ := 12

/-- The main theorem stating the distance between the 4th and 28th red lights -/
theorem distance_between_4th_and_28th_red_lights :
  (redLightPosition 28 - redLightPosition 4) * lightDistance / inchesPerFoot = 26 := by
  sorry


end NUMINAMATH_CALUDE_distance_between_4th_and_28th_red_lights_l738_73873


namespace NUMINAMATH_CALUDE_symmetry_of_f_l738_73890

def is_symmetric_about (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x, f x + f (2 * a - x) = 2 * b

theorem symmetry_of_f (f : ℝ → ℝ) (h : ∀ x, f (x + 2) = -f (2 - x)) :
  is_symmetric_about f 2 0 := by
  sorry

end NUMINAMATH_CALUDE_symmetry_of_f_l738_73890


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l738_73800

-- Define the sets A and B
def A : Set (ℝ × ℝ) := {p | 4 * p.1 + p.2 = 6}
def B : Set (ℝ × ℝ) := {p | p.1 - p.2 = 4}

-- State the theorem
theorem intersection_of_A_and_B : A ∩ B = {(2, -2)} := by
  sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l738_73800


namespace NUMINAMATH_CALUDE_salary_increase_percentage_l738_73891

theorem salary_increase_percentage (S : ℝ) (P : ℝ) : 
  (S + (P / 100) * S) * 0.9 = S * 1.15 → P = 100 * (15 / 0.9) := by
  sorry

end NUMINAMATH_CALUDE_salary_increase_percentage_l738_73891


namespace NUMINAMATH_CALUDE_triangle_side_length_l738_73818

-- Define the triangle and circle
structure Triangle :=
  (A B C O : ℝ × ℝ)
  (circumscribed : Bool)

-- Define the properties of the triangle
def is_valid_triangle (t : Triangle) : Prop :=
  t.circumscribed ∧ 
  dist t.B t.C = 5 ∧
  dist t.A t.B = 4 ∧
  norm (3 • (t.A - t.O) - 4 • (t.B - t.O) + (t.C - t.O)) = 10

-- Theorem statement
theorem triangle_side_length (t : Triangle) :
  is_valid_triangle t → dist t.A t.C = 4 := by
  sorry

end NUMINAMATH_CALUDE_triangle_side_length_l738_73818


namespace NUMINAMATH_CALUDE_parabola_directrix_l738_73865

/-- The directrix of a parabola with equation x = -1/4 * y^2 is the line x = 1 -/
theorem parabola_directrix : 
  ∀ (x y : ℝ), x = -(1/4) * y^2 → (∃ (k : ℝ), k = 1 ∧ 
    ∀ (p : ℝ × ℝ), p.1 = k ↔ 
      ∀ (q : ℝ × ℝ), q.1 = x ∧ q.2 = y → 
        (p.1 - q.1)^2 + (p.2 - q.2)^2 = (q.1 - (-1))^2 + q.2^2) :=
by
  sorry

end NUMINAMATH_CALUDE_parabola_directrix_l738_73865


namespace NUMINAMATH_CALUDE_average_expenditure_feb_to_jul_l738_73879

def average_expenditure_jan_to_jun : ℝ := 4200
def expenditure_january : ℝ := 1200
def expenditure_july : ℝ := 1500
def num_months : ℕ := 6

theorem average_expenditure_feb_to_jul :
  let total_jan_to_jun := average_expenditure_jan_to_jun * num_months
  let total_feb_to_jun := total_jan_to_jun - expenditure_january
  let total_feb_to_jul := total_feb_to_jun + expenditure_july
  total_feb_to_jul / num_months = 4250 := by
sorry

end NUMINAMATH_CALUDE_average_expenditure_feb_to_jul_l738_73879


namespace NUMINAMATH_CALUDE_division_simplification_l738_73801

theorem division_simplification (m : ℝ) (h : m ≠ 0) :
  (4 * m^2 - 2 * m) / (2 * m) = 2 * m - 1 := by
  sorry

end NUMINAMATH_CALUDE_division_simplification_l738_73801


namespace NUMINAMATH_CALUDE_a_investment_amount_verify_profit_share_l738_73880

/-- Represents the investment scenario described in the problem -/
structure Investment where
  a_amount : ℝ  -- A's investment amount
  b_amount : ℝ := 200  -- B's investment amount
  a_months : ℝ := 12  -- Duration of A's investment in months
  b_months : ℝ := 6  -- Duration of B's investment in months
  total_profit : ℝ := 100  -- Total profit
  a_profit : ℝ := 50  -- A's share of the profit

/-- Theorem stating that A's investment amount must be $100 given the conditions -/
theorem a_investment_amount (inv : Investment) : 
  (inv.a_amount * inv.a_months) / (inv.b_amount * inv.b_months) = 1 →
  inv.a_amount = 100 := by
  sorry

/-- Corollary confirming that the calculated investment satisfies the profit sharing condition -/
theorem verify_profit_share (inv : Investment) : 
  inv.a_amount = 100 →
  (inv.a_amount * inv.a_months) / (inv.b_amount * inv.b_months) = 1 := by
  sorry

end NUMINAMATH_CALUDE_a_investment_amount_verify_profit_share_l738_73880


namespace NUMINAMATH_CALUDE_election_invalid_votes_l738_73881

theorem election_invalid_votes 
  (total_polled : ℕ) 
  (losing_percentage : ℚ) 
  (vote_difference : ℕ) 
  (h_total : total_polled = 90083)
  (h_losing : losing_percentage = 45 / 100)
  (h_difference : vote_difference = 9000) :
  total_polled - (vote_difference / (1/2 - losing_percentage)) = 83 := by
sorry

end NUMINAMATH_CALUDE_election_invalid_votes_l738_73881


namespace NUMINAMATH_CALUDE_quadratic_intercept_l738_73851

/-- A quadratic function with vertex (4, 9) and x-intercept (0, 0) has its other x-intercept at x = 8 -/
theorem quadratic_intercept (a b c : ℝ) : 
  (∀ x, a * x^2 + b * x + c = a * (x - 4)^2 + 9) →  -- vertex form
  (a * 0^2 + b * 0 + c = 0) →                       -- (0, 0) is an x-intercept
  (∃ x ≠ 0, a * x^2 + b * x + c = 0 ∧ x = 8) :=     -- other x-intercept is at x = 8
by sorry

end NUMINAMATH_CALUDE_quadratic_intercept_l738_73851


namespace NUMINAMATH_CALUDE_max_value_a_plus_sqrt3b_l738_73877

theorem max_value_a_plus_sqrt3b (a b : ℝ) (ha : a > 0) (hb : b > 0)
  (h : Real.sqrt 3 * b = Real.sqrt ((1 - a) * (1 + a))) :
  ∃ (x : ℝ), ∀ (y : ℝ), (∃ (a' b' : ℝ), a' > 0 ∧ b' > 0 ∧
    Real.sqrt 3 * b' = Real.sqrt ((1 - a') * (1 + a')) ∧
    y = a' + Real.sqrt 3 * b') →
  y ≤ x ∧ x = Real.sqrt 2 :=
by sorry

end NUMINAMATH_CALUDE_max_value_a_plus_sqrt3b_l738_73877


namespace NUMINAMATH_CALUDE_alcohol_mixture_proof_l738_73883

theorem alcohol_mixture_proof (x y z final_volume final_alcohol : ℝ) :
  x = 300 ∧ y = 600 ∧ z = 300 ∧
  (0.1 * x + 0.3 * y + 0.4 * z) / (x + y + z) = 0.22 ∧
  y = 2 * z →
  final_volume = x + y + z ∧
  final_alcohol = 0.22 * final_volume :=
by sorry

end NUMINAMATH_CALUDE_alcohol_mixture_proof_l738_73883


namespace NUMINAMATH_CALUDE_pear_juice_percentage_l738_73876

def pears_for_juice : ℕ := 4
def oranges_for_juice : ℕ := 3
def pear_juice_yield : ℚ := 12
def orange_juice_yield : ℚ := 6
def pears_in_blend : ℕ := 8
def oranges_in_blend : ℕ := 6

theorem pear_juice_percentage :
  let pear_juice_per_fruit : ℚ := pear_juice_yield / pears_for_juice
  let orange_juice_per_fruit : ℚ := orange_juice_yield / oranges_for_juice
  let total_pear_juice : ℚ := pear_juice_per_fruit * pears_in_blend
  let total_orange_juice : ℚ := orange_juice_per_fruit * oranges_in_blend
  let total_juice : ℚ := total_pear_juice + total_orange_juice
  (total_pear_juice / total_juice) * 100 = 200 / 3 := by sorry

end NUMINAMATH_CALUDE_pear_juice_percentage_l738_73876


namespace NUMINAMATH_CALUDE_red_tickets_for_yellow_l738_73861

/-- The number of yellow tickets needed to win a Bible -/
def yellow_tickets_needed : ℕ := 10

/-- The number of blue tickets needed to obtain one red ticket -/
def blue_per_red : ℕ := 10

/-- Tom's current yellow tickets -/
def tom_yellow : ℕ := 8

/-- Tom's current red tickets -/
def tom_red : ℕ := 3

/-- Tom's current blue tickets -/
def tom_blue : ℕ := 7

/-- Additional blue tickets Tom needs -/
def additional_blue : ℕ := 163

/-- The number of red tickets required to obtain one yellow ticket -/
def red_per_yellow : ℕ := 7

theorem red_tickets_for_yellow : 
  (yellow_tickets_needed - tom_yellow) * red_per_yellow = 
  (additional_blue + tom_blue) / blue_per_red - tom_red := by
  sorry

end NUMINAMATH_CALUDE_red_tickets_for_yellow_l738_73861


namespace NUMINAMATH_CALUDE_quadratic_root_range_l738_73868

theorem quadratic_root_range (a : ℝ) : 
  (∃ x y : ℝ, x ≠ y ∧ x > 1 ∧ y < 1 ∧ 
   x^2 + (a^2 - 1)*x + a - 2 = 0 ∧
   y^2 + (a^2 - 1)*y + a - 2 = 0) →
  a > -2 ∧ a < 1 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_root_range_l738_73868


namespace NUMINAMATH_CALUDE_figure2_total_length_l738_73866

/-- The total length of segments in Figure 2 -/
def total_length (left right top bottom : ℝ) : ℝ :=
  left + right + top + bottom

/-- The theorem stating that the total length of segments in Figure 2 is 19 units -/
theorem figure2_total_length :
  ∃ (left right top bottom : ℝ),
    left = 8 ∧
    right = 6 ∧
    top = 1 ∧
    bottom = 2 + 1 + 1 ∧
    total_length left right top bottom = 19 := by
  sorry

end NUMINAMATH_CALUDE_figure2_total_length_l738_73866


namespace NUMINAMATH_CALUDE_freddy_calls_cost_l738_73812

/-- Calculates the total cost of phone calls in dollars -/
def total_cost_dollars (local_rate : ℚ) (intl_rate : ℚ) (local_duration : ℕ) (intl_duration : ℕ) : ℚ :=
  ((local_rate * local_duration + intl_rate * intl_duration) / 100 : ℚ)

theorem freddy_calls_cost :
  let local_rate : ℚ := 5
  let intl_rate : ℚ := 25
  let local_duration : ℕ := 45
  let intl_duration : ℕ := 31
  total_cost_dollars local_rate intl_rate local_duration intl_duration = 10 := by
sorry

#eval total_cost_dollars 5 25 45 31

end NUMINAMATH_CALUDE_freddy_calls_cost_l738_73812


namespace NUMINAMATH_CALUDE_percentage_of_B_grades_l738_73897

def scores : List ℕ := [88, 73, 55, 95, 76, 91, 86, 73, 76, 64, 85, 79, 72, 81, 89, 77]

def is_B_grade (score : ℕ) : Bool :=
  87 ≤ score ∧ score ≤ 94

def count_B_grades (scores : List ℕ) : ℕ :=
  scores.filter is_B_grade |>.length

theorem percentage_of_B_grades :
  (count_B_grades scores : ℚ) / scores.length * 100 = 12.5 := by sorry

end NUMINAMATH_CALUDE_percentage_of_B_grades_l738_73897


namespace NUMINAMATH_CALUDE_truck_tire_usage_l738_73826

/-- Calculates the number of miles each tire is used on a truck -/
def miles_per_tire (total_miles : ℕ) (total_tires : ℕ) (active_tires : ℕ) : ℚ :=
  (total_miles * active_tires : ℚ) / total_tires

theorem truck_tire_usage :
  let total_miles : ℕ := 36000
  let total_tires : ℕ := 6
  let active_tires : ℕ := 5
  miles_per_tire total_miles total_tires active_tires = 30000 := by
sorry

#eval miles_per_tire 36000 6 5

end NUMINAMATH_CALUDE_truck_tire_usage_l738_73826


namespace NUMINAMATH_CALUDE_forgot_lawns_count_l738_73882

def lawn_problem (total_lawns : ℕ) (earnings_per_lawn : ℕ) (actual_earnings : ℕ) : ℕ :=
  total_lawns - (actual_earnings / earnings_per_lawn)

theorem forgot_lawns_count :
  lawn_problem 17 4 32 = 9 := by
  sorry

end NUMINAMATH_CALUDE_forgot_lawns_count_l738_73882


namespace NUMINAMATH_CALUDE_isosceles_right_triangle_not_regular_polygon_l738_73835

-- Define an isosceles right triangle
structure IsoscelesRightTriangle where
  side : ℝ
  side_positive : side > 0

-- Define a regular polygon
structure RegularPolygon where
  sides : ℕ
  side_length : ℝ
  side_positive : side_length > 0

-- Theorem: Isosceles right triangles are not regular polygons
theorem isosceles_right_triangle_not_regular_polygon :
  ∀ (t : IsoscelesRightTriangle), ¬∃ (p : RegularPolygon), true :=
sorry

end NUMINAMATH_CALUDE_isosceles_right_triangle_not_regular_polygon_l738_73835


namespace NUMINAMATH_CALUDE_angle_sum_in_special_pentagon_l738_73813

theorem angle_sum_in_special_pentagon (x y : ℝ) 
  (h1 : 0 ≤ x ∧ x < 180) 
  (h2 : 0 ≤ y ∧ y < 180) : 
  x + y = 50 := by
  sorry

end NUMINAMATH_CALUDE_angle_sum_in_special_pentagon_l738_73813


namespace NUMINAMATH_CALUDE_expression_simplification_l738_73829

theorem expression_simplification (a₁ a₂ a₃ a₄ : ℝ) :
  1 + a₁ / (1 - a₁) + a₂ / ((1 - a₁) * (1 - a₂)) + 
  a₃ / ((1 - a₁) * (1 - a₂) * (1 - a₃)) + 
  (a₄ - a₁) / ((1 - a₁) * (1 - a₂) * (1 - a₃) * (1 - a₄)) = 
  1 / ((1 - a₂) * (1 - a₃) * (1 - a₄)) :=
by sorry

end NUMINAMATH_CALUDE_expression_simplification_l738_73829


namespace NUMINAMATH_CALUDE_third_degree_equation_roots_l738_73838

theorem third_degree_equation_roots (x : ℝ) : 
  let f : ℝ → ℝ := λ x => 8 * x^3 - 4 * x^2 - 4 * x - 1
  let root1 := Real.sin (π / 14)
  let root2 := Real.sin (5 * π / 14)
  let root3 := Real.sin (-3 * π / 14)
  (f root1 = 0) ∧ (f root2 = 0) ∧ (f root3 = 0) := by
  sorry

end NUMINAMATH_CALUDE_third_degree_equation_roots_l738_73838


namespace NUMINAMATH_CALUDE_three_integer_sum_l738_73857

theorem three_integer_sum (a b c : ℕ) : 
  a > 1 → b > 1 → c > 1 →
  a * b * c = 343000 →
  Nat.gcd a b = 1 → Nat.gcd b c = 1 → Nat.gcd a c = 1 →
  a + b + c = 476 := by
sorry

end NUMINAMATH_CALUDE_three_integer_sum_l738_73857


namespace NUMINAMATH_CALUDE_temperature_80_degrees_l738_73808

-- Define the temperature function
def temperature (t : ℝ) : ℝ := -t^2 + 10*t + 60

-- State the theorem
theorem temperature_80_degrees :
  ∃ t₁ t₂ : ℝ, 
    t₁ = 5 + 3 * Real.sqrt 5 ∧ 
    t₂ = 5 - 3 * Real.sqrt 5 ∧ 
    temperature t₁ = 80 ∧ 
    temperature t₂ = 80 ∧ 
    (∀ t : ℝ, temperature t = 80 → t = t₁ ∨ t = t₂) := by
  sorry

end NUMINAMATH_CALUDE_temperature_80_degrees_l738_73808


namespace NUMINAMATH_CALUDE_imaginary_part_of_i_l738_73855

theorem imaginary_part_of_i : Complex.im i = 1 := by
  sorry

end NUMINAMATH_CALUDE_imaginary_part_of_i_l738_73855


namespace NUMINAMATH_CALUDE_triangle_properties_l738_73834

open Real

/-- Triangle ABC with side lengths a, b, c opposite to angles A, B, C respectively -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- The theorem stating properties of the triangle -/
theorem triangle_properties (t : Triangle) 
  (h1 : t.A ≠ π / 2)
  (h2 : 3 * sin t.A * cos t.B + (1/2) * t.b * sin (2 * t.A) = 3 * sin t.C) :
  t.a = 3 ∧ 
  (t.A = 2 * π / 3 → 
    ∃ (p : ℝ), p ≤ t.a + t.b + t.c ∧ p = 3 + 2 * Real.sqrt 3) := by
  sorry


end NUMINAMATH_CALUDE_triangle_properties_l738_73834


namespace NUMINAMATH_CALUDE_chorus_group_size_l738_73884

theorem chorus_group_size :
  let S := {n : ℕ | 100 < n ∧ n < 200 ∧
                    n % 3 = 1 ∧
                    n % 4 = 2 ∧
                    n % 6 = 4 ∧
                    n % 8 = 6}
  S = {118, 142, 166, 190} := by
  sorry

end NUMINAMATH_CALUDE_chorus_group_size_l738_73884


namespace NUMINAMATH_CALUDE_root_implies_t_value_l738_73864

theorem root_implies_t_value (t : ℝ) : 
  (3 * (((-15 - Real.sqrt 145) / 6) ^ 2) + 15 * ((-15 - Real.sqrt 145) / 6) + t = 0) → 
  t = 20 / 3 := by
sorry

end NUMINAMATH_CALUDE_root_implies_t_value_l738_73864


namespace NUMINAMATH_CALUDE_half_perimeter_area_rectangle_existence_l738_73807

/-- Given a rectangle with sides a and b, this theorem proves the existence of another rectangle
    with half the perimeter and half the area, based on the discriminant of the resulting quadratic equation. -/
theorem half_perimeter_area_rectangle_existence (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  ∃ (x y : ℝ), x > 0 ∧ y > 0 ∧ x + y = (a + b) / 2 ∧ x * y = (a * b) / 2 ↔
  ((a + b)^2 - 4 * (a * b)) ≥ 0 :=
by sorry

end NUMINAMATH_CALUDE_half_perimeter_area_rectangle_existence_l738_73807


namespace NUMINAMATH_CALUDE_lcm_problem_l738_73856

theorem lcm_problem (e n : ℕ) : 
  e > 0 ∧ 
  100 ≤ n ∧ n < 1000 ∧ 
  Nat.lcm e n = 690 ∧ 
  ¬(3 ∣ n) ∧ 
  ¬(2 ∣ e) →
  n = 230 := by
sorry

end NUMINAMATH_CALUDE_lcm_problem_l738_73856


namespace NUMINAMATH_CALUDE_glorias_cash_was_150_l738_73824

/-- Calculates Gloria's initial cash given the cabin cost, tree counts, tree prices, and leftover cash --/
def glorias_initial_cash (cabin_cost : ℕ) (cypress_count pine_count maple_count : ℕ) 
  (cypress_price pine_price maple_price : ℕ) (leftover_cash : ℕ) : ℕ :=
  cabin_cost + leftover_cash - (cypress_count * cypress_price + pine_count * pine_price + maple_count * maple_price)

/-- Theorem stating that Gloria's initial cash was $150 --/
theorem glorias_cash_was_150 : 
  glorias_initial_cash 129000 20 600 24 100 200 300 350 = 150 := by
  sorry


end NUMINAMATH_CALUDE_glorias_cash_was_150_l738_73824


namespace NUMINAMATH_CALUDE_a_5_equals_20_l738_73875

def S (n : ℕ) : ℕ := 2 * n * (n + 1)

def a (n : ℕ) : ℕ := S n - S (n - 1)

theorem a_5_equals_20 : a 5 = 20 := by
  sorry

end NUMINAMATH_CALUDE_a_5_equals_20_l738_73875


namespace NUMINAMATH_CALUDE_parabola_equation_correct_l738_73837

-- Define the parabola
structure Parabola where
  focus : ℝ × ℝ
  directrix : ℝ → ℝ → ℝ

-- Define the equation of the parabola
def parabola_equation (p : Parabola) (x y : ℝ) : ℝ :=
  16 * x^2 + 25 * y^2 + 36 * x + 242 * y - 195

-- Theorem statement
theorem parabola_equation_correct (p : Parabola) :
  p.focus = (2, -1) ∧ p.directrix = (fun x y ↦ 5*x + 4*y - 20) →
  ∀ x y : ℝ, (x - 2)^2 + (y + 1)^2 = ((5*x + 4*y - 20)^2) / 41 ↔
  parabola_equation p x y = 0 :=
sorry

end NUMINAMATH_CALUDE_parabola_equation_correct_l738_73837


namespace NUMINAMATH_CALUDE_smallest_number_with_special_sums_l738_73846

/-- A function that returns the sum of digits of a natural number -/
def digit_sum (n : ℕ) : ℕ := sorry

/-- A function that checks if a list of natural numbers all have the same digit sum -/
def same_digit_sum (list : List ℕ) : Prop := sorry

/-- Theorem stating that 10010 is the smallest natural number satisfying the given conditions -/
theorem smallest_number_with_special_sums : 
  ∀ n : ℕ, n < 10010 → 
  ¬(∃ (list1 : List ℕ) (list2 : List ℕ), 
    list1.length = 2002 ∧ 
    list2.length = 2003 ∧ 
    same_digit_sum list1 ∧ 
    same_digit_sum list2 ∧ 
    list1.sum = n ∧ 
    list2.sum = n) ∧ 
  ∃ (list1 : List ℕ) (list2 : List ℕ), 
    list1.length = 2002 ∧ 
    list2.length = 2003 ∧ 
    same_digit_sum list1 ∧ 
    same_digit_sum list2 ∧ 
    list1.sum = 10010 ∧ 
    list2.sum = 10010 := by
  sorry

end NUMINAMATH_CALUDE_smallest_number_with_special_sums_l738_73846


namespace NUMINAMATH_CALUDE_eight_bead_necklace_arrangements_l738_73863

/-- The number of distinct arrangements of n beads on a necklace,
    considering rotational and reflectional symmetry -/
def necklace_arrangements (n : ℕ) : ℕ :=
  Nat.factorial n / (n * 2)

/-- Theorem stating that the number of distinct arrangements
    of 8 beads on a necklace is 2520 -/
theorem eight_bead_necklace_arrangements :
  necklace_arrangements 8 = 2520 := by
  sorry

end NUMINAMATH_CALUDE_eight_bead_necklace_arrangements_l738_73863


namespace NUMINAMATH_CALUDE_odot_four_three_l738_73819

-- Define the binary operation ⊙
def odot (a b : ℝ) : ℝ := 5 * a + 2 * b

-- Theorem statement
theorem odot_four_three : odot 4 3 = 26 := by
  sorry

end NUMINAMATH_CALUDE_odot_four_three_l738_73819


namespace NUMINAMATH_CALUDE_video_subscription_cost_l738_73811

def monthly_cost : ℚ := 14
def num_people : ℕ := 2
def months_in_year : ℕ := 12

theorem video_subscription_cost :
  (monthly_cost / num_people) * months_in_year = 84 := by
sorry

end NUMINAMATH_CALUDE_video_subscription_cost_l738_73811


namespace NUMINAMATH_CALUDE_water_left_in_bucket_l738_73842

/-- Converts milliliters to liters -/
def ml_to_l (ml : ℚ) : ℚ := ml / 1000

/-- Calculates the remaining water in a bucket after some is removed -/
def remaining_water (initial : ℚ) (removed_ml : ℚ) (removed_l : ℚ) : ℚ :=
  initial - (ml_to_l removed_ml + removed_l)

theorem water_left_in_bucket : 
  remaining_water 30 150 1.65 = 28.20 := by sorry

end NUMINAMATH_CALUDE_water_left_in_bucket_l738_73842


namespace NUMINAMATH_CALUDE_least_integer_with_12_factors_l738_73832

/-- The number of positive factors of a positive integer -/
def num_factors (n : ℕ+) : ℕ := sorry

/-- Checks if a positive integer has exactly 12 factors -/
def has_12_factors (n : ℕ+) : Prop := num_factors n = 12

theorem least_integer_with_12_factors :
  ∃ (k : ℕ+), has_12_factors k ∧ ∀ (m : ℕ+), has_12_factors m → k ≤ m :=
by
  use 108
  sorry

end NUMINAMATH_CALUDE_least_integer_with_12_factors_l738_73832


namespace NUMINAMATH_CALUDE_sin_cos_ratio_simplification_l738_73827

theorem sin_cos_ratio_simplification :
  (Real.sin (40 * π / 180) + Real.sin (80 * π / 180)) /
  (Real.cos (40 * π / 180) + Real.cos (80 * π / 180)) = Real.sqrt 3 :=
by sorry

end NUMINAMATH_CALUDE_sin_cos_ratio_simplification_l738_73827


namespace NUMINAMATH_CALUDE_parallel_iff_a_eq_one_l738_73898

/-- Two lines in the form ax - y + b = 0 and cx - y + d = 0 are parallel if and only if a = c -/
def are_parallel (a c : ℝ) : Prop := a = c

/-- The condition for the given lines to be parallel -/
def parallel_condition (a : ℝ) : Prop := are_parallel a (1/a)

theorem parallel_iff_a_eq_one (a : ℝ) : 
  parallel_condition a ↔ a = 1 :=
sorry

end NUMINAMATH_CALUDE_parallel_iff_a_eq_one_l738_73898


namespace NUMINAMATH_CALUDE_uniform_cost_calculation_l738_73843

/-- Calculates the total cost of uniforms for a student --/
def uniformCost (
  numUniforms : ℕ
) (
  pantsCost : ℚ
) (
  shirtCostMultiplier : ℚ
) (
  tieCostFraction : ℚ
) (
  socksCost : ℚ
) (
  jacketCostMultiplier : ℚ
) (
  shoesCost : ℚ
) (
  discountRate : ℚ
) (
  discountThreshold : ℕ
) : ℚ :=
  sorry

theorem uniform_cost_calculation :
  uniformCost 5 20 2 (1/5) 3 3 40 (1/10) 3 = 1039.5 := by
  sorry

end NUMINAMATH_CALUDE_uniform_cost_calculation_l738_73843


namespace NUMINAMATH_CALUDE_sqrt_eighteen_minus_three_sqrt_half_plus_sqrt_two_l738_73889

theorem sqrt_eighteen_minus_three_sqrt_half_plus_sqrt_two : 
  Real.sqrt 18 - 3 * Real.sqrt (1/2) + Real.sqrt 2 = (5 * Real.sqrt 2) / 2 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_eighteen_minus_three_sqrt_half_plus_sqrt_two_l738_73889


namespace NUMINAMATH_CALUDE_cat_cleaner_amount_l738_73803

/-- The amount of cleaner used for a dog stain in ounces -/
def dog_cleaner : ℝ := 6

/-- The amount of cleaner used for a rabbit stain in ounces -/
def rabbit_cleaner : ℝ := 1

/-- The total amount of cleaner used for all stains in ounces -/
def total_cleaner : ℝ := 49

/-- The number of dogs -/
def num_dogs : ℕ := 6

/-- The number of cats -/
def num_cats : ℕ := 3

/-- The number of rabbits -/
def num_rabbits : ℕ := 1

/-- The amount of cleaner used for a cat stain in ounces -/
def cat_cleaner : ℝ := 4

theorem cat_cleaner_amount :
  dog_cleaner * num_dogs + cat_cleaner * num_cats + rabbit_cleaner * num_rabbits = total_cleaner :=
by sorry

end NUMINAMATH_CALUDE_cat_cleaner_amount_l738_73803


namespace NUMINAMATH_CALUDE_fraction_product_simplification_l738_73878

theorem fraction_product_simplification :
  (2 / 3) * (3 / 4) * (4 / 5) * (5 / 6) * (6 / 7) = 2 / 7 := by
  sorry

end NUMINAMATH_CALUDE_fraction_product_simplification_l738_73878


namespace NUMINAMATH_CALUDE_greatest_product_sum_300_l738_73895

theorem greatest_product_sum_300 : 
  ∀ x y : ℤ, x + y = 300 → x * y ≤ 22500 := by
  sorry

end NUMINAMATH_CALUDE_greatest_product_sum_300_l738_73895


namespace NUMINAMATH_CALUDE_sequences_properties_l738_73858

def sequence_a (n : ℕ) : ℤ := (-2) ^ n
def sequence_b (n : ℕ) : ℤ := (-2) ^ (n - 1)
def sequence_c (n : ℕ) : ℕ := 3 * 2 ^ (n - 1)

theorem sequences_properties :
  (sequence_a 6 = 64) ∧
  (sequence_b 7 = 64) ∧
  (sequence_c 7 = 192) ∧
  (sequence_c 11 = 3072) := by
sorry

end NUMINAMATH_CALUDE_sequences_properties_l738_73858


namespace NUMINAMATH_CALUDE_lcm_gcd_1560_1040_l738_73852

theorem lcm_gcd_1560_1040 :
  (Nat.lcm 1560 1040 = 1560) ∧ (Nat.gcd 1560 1040 = 520) := by
  sorry

end NUMINAMATH_CALUDE_lcm_gcd_1560_1040_l738_73852


namespace NUMINAMATH_CALUDE_probability_white_then_red_l738_73845

def total_marbles : ℕ := 14
def red_marbles : ℕ := 6
def white_marbles : ℕ := 8

theorem probability_white_then_red :
  (white_marbles / total_marbles) * (red_marbles / (total_marbles - 1)) = 24 / 91 := by
sorry

end NUMINAMATH_CALUDE_probability_white_then_red_l738_73845


namespace NUMINAMATH_CALUDE_consecutive_non_multiple_of_five_product_l738_73823

theorem consecutive_non_multiple_of_five_product (k : ℤ) :
  (∃ m : ℤ, (5*k + 1) * (5*k + 2) * (5*k + 3) = 5*m + 1) ∨
  (∃ n : ℤ, (5*k + 2) * (5*k + 3) * (5*k + 4) = 5*n - 1) :=
by sorry

end NUMINAMATH_CALUDE_consecutive_non_multiple_of_five_product_l738_73823


namespace NUMINAMATH_CALUDE_no_representation_as_sum_of_squares_and_ninth_power_l738_73871

theorem no_representation_as_sum_of_squares_and_ninth_power (p : ℕ) (m : ℤ) 
  (h_prime : Nat.Prime p) (h_form : p = 4 * m + 1) :
  ¬ ∃ (x y z : ℤ), 216 * (p : ℤ)^3 = x^2 + y^2 + z^9 := by
  sorry

end NUMINAMATH_CALUDE_no_representation_as_sum_of_squares_and_ninth_power_l738_73871


namespace NUMINAMATH_CALUDE_inequality_implications_l738_73872

theorem inequality_implications (a b : ℝ) (h : a > b) : 
  (a + 2 > b + 2) ∧ 
  (a / 4 > b / 4) ∧ 
  ¬(-3 * a > -3 * b) ∧ 
  ¬(a - 1 < b - 1) := by
sorry

end NUMINAMATH_CALUDE_inequality_implications_l738_73872


namespace NUMINAMATH_CALUDE_cube_tetrahedron_surface_area_ratio_l738_73849

/-- A cube with side length s contains a regular tetrahedron with vertices
    (0,0,0), (s,s,0), (s,0,s), and (0,s,s). The ratio of the surface area of
    the cube to the surface area of the tetrahedron is √3. -/
theorem cube_tetrahedron_surface_area_ratio (s : ℝ) (h : s > 0) :
  let cube_vertices : Fin 8 → ℝ × ℝ × ℝ := fun i =>
    ((i : ℕ) % 2 * s, ((i : ℕ) / 2) % 2 * s, ((i : ℕ) / 4) * s)
  let tetra_vertices : Fin 4 → ℝ × ℝ × ℝ := fun i =>
    match i with
    | 0 => (0, 0, 0)
    | 1 => (s, s, 0)
    | 2 => (s, 0, s)
    | 3 => (0, s, s)
  let cube_surface_area := 6 * s^2
  let tetra_surface_area := 2 * Real.sqrt 3 * s^2
  cube_surface_area / tetra_surface_area = Real.sqrt 3 := by
  sorry

#check cube_tetrahedron_surface_area_ratio

end NUMINAMATH_CALUDE_cube_tetrahedron_surface_area_ratio_l738_73849


namespace NUMINAMATH_CALUDE_equation_linear_iff_a_eq_plus_minus_two_l738_73802

-- Define the equation
def equation (a x y : ℝ) : ℝ := (a^2 - 4) * x^2 + (2 - 3*a) * x + (a + 1) * y + 3*a

-- Define what it means for the equation to be linear in two variables
def is_linear_two_var (a : ℝ) : Prop :=
  (a^2 - 4 = 0) ∧ (2 - 3*a ≠ 0 ∨ a + 1 ≠ 0)

-- State the theorem
theorem equation_linear_iff_a_eq_plus_minus_two :
  ∀ a : ℝ, is_linear_two_var a ↔ (a = 2 ∨ a = -2) :=
by sorry

end NUMINAMATH_CALUDE_equation_linear_iff_a_eq_plus_minus_two_l738_73802


namespace NUMINAMATH_CALUDE_jacksons_grade_calculation_l738_73841

/-- Calculates Jackson's grade based on his time allocation and point system. -/
def jacksons_grade (video_game_hours : ℝ) (study_ratio : ℝ) (kindness_ratio : ℝ) 
  (study_points_per_hour : ℝ) (kindness_points_per_hour : ℝ) : ℝ :=
  let study_hours := video_game_hours * study_ratio
  let kindness_hours := video_game_hours * kindness_ratio
  study_hours * study_points_per_hour + kindness_hours * kindness_points_per_hour

theorem jacksons_grade_calculation :
  jacksons_grade 12 (1/3) (1/4) 20 40 = 200 := by
  sorry

end NUMINAMATH_CALUDE_jacksons_grade_calculation_l738_73841


namespace NUMINAMATH_CALUDE_brownie_pieces_l738_73830

theorem brownie_pieces (pan_length pan_width piece_length piece_width : ℕ) 
  (h1 : pan_length = 30)
  (h2 : pan_width = 24)
  (h3 : piece_length = 3)
  (h4 : piece_width = 4) :
  (pan_length * pan_width) / (piece_length * piece_width) = 60 := by
  sorry

end NUMINAMATH_CALUDE_brownie_pieces_l738_73830


namespace NUMINAMATH_CALUDE_quadratic_equation_solutions_l738_73820

def f (a b c : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x + c

theorem quadratic_equation_solutions 
  (a b c : ℝ) 
  (h1 : f a b c (-2) = 3)
  (h2 : f a b c (-1) = 4)
  (h3 : f a b c 0 = 3)
  (h4 : f a b c 1 = 0)
  (h5 : f a b c 2 = -5) :
  ∃ (x₁ x₂ : ℝ), x₁ = 2 ∧ x₂ = -4 ∧ f a b c x₁ = -5 ∧ f a b c x₂ = -5 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equation_solutions_l738_73820


namespace NUMINAMATH_CALUDE_cPass_max_entries_aPass_cost_effective_l738_73804

-- Define the ticketing options
structure TicketOption where
  initialCost : ℕ
  entryCost : ℕ

-- Define the budget
def budget : ℕ := 80

-- Define the ticketing options
def noPass : TicketOption := ⟨0, 10⟩
def aPass : TicketOption := ⟨120, 0⟩
def bPass : TicketOption := ⟨60, 2⟩
def cPass : TicketOption := ⟨40, 3⟩

-- Function to calculate the number of entries for a given option and budget
def numEntries (option : TicketOption) (budget : ℕ) : ℕ :=
  if option.initialCost > budget then 0
  else (budget - option.initialCost) / option.entryCost

-- Theorem 1: C pass allows for the maximum number of entries with 80 yuan budget
theorem cPass_max_entries :
  ∀ option : TicketOption, numEntries cPass budget ≥ numEntries option budget :=
sorry

-- Theorem 2: A pass becomes more cost-effective when entering more than 30 times
theorem aPass_cost_effective (n : ℕ) (h : n > 30) :
  ∀ option : TicketOption, option.initialCost + n * option.entryCost > aPass.initialCost :=
sorry

end NUMINAMATH_CALUDE_cPass_max_entries_aPass_cost_effective_l738_73804


namespace NUMINAMATH_CALUDE_visual_illusion_occurs_l738_73822

/-- Represents the structure of the cardboard disc -/
structure Disc where
  inner_sectors : Nat
  outer_sectors : Nat
  inner_white : Nat
  outer_white : Nat

/-- Represents the properties of electric lighting -/
structure Lighting where
  flicker_frequency : Real
  flicker_interval : Real

/-- Defines the rotation speeds that create the visual illusion -/
def illusion_speeds (d : Disc) (l : Lighting) : Prop :=
  let inner_speed := 25
  let outer_speed := 20
  inner_speed * l.flicker_interval = 0.25 ∧
  outer_speed * l.flicker_interval = 0.2

theorem visual_illusion_occurs (d : Disc) (l : Lighting) :
  d.inner_sectors = 8 ∧
  d.outer_sectors = 10 ∧
  d.inner_white = 4 ∧
  d.outer_white = 5 ∧
  l.flicker_frequency = 100 ∧
  l.flicker_interval = 0.01 →
  illusion_speeds d l :=
by sorry


end NUMINAMATH_CALUDE_visual_illusion_occurs_l738_73822


namespace NUMINAMATH_CALUDE_sum_of_roots_quadratic_undefined_values_sum_l738_73840

theorem sum_of_roots_quadratic : 
  ∀ (a b c : ℝ) (y₁ y₂ : ℝ), 
  a ≠ 0 → 
  a * y₁^2 + b * y₁ + c = 0 → 
  a * y₂^2 + b * y₂ + c = 0 → 
  y₁ + y₂ = -b / a :=
sorry

theorem undefined_values_sum : 
  let y₁ := (3 + Real.sqrt 49) / 2
  let y₂ := (3 - Real.sqrt 49) / 2
  y₁^2 - 3*y₁ - 10 = 0 ∧ 
  y₂^2 - 3*y₂ - 10 = 0 ∧ 
  y₁ + y₂ = 3 :=
sorry

end NUMINAMATH_CALUDE_sum_of_roots_quadratic_undefined_values_sum_l738_73840


namespace NUMINAMATH_CALUDE_valid_coloring_exists_l738_73888

/-- A type representing a coloring of regions in a plane --/
def Coloring (n : ℕ) := Fin n → Bool

/-- A predicate that checks if a coloring is valid for n lines --/
def IsValidColoring (n : ℕ) (c : Coloring n) : Prop :=
  ∀ (i j : Fin n), i ≠ j → c i ≠ c j

/-- The main theorem stating that a valid coloring exists for any number of lines --/
theorem valid_coloring_exists (n : ℕ) (h : n > 0) : 
  ∃ (c : Coloring n), IsValidColoring n c := by
  sorry

#check valid_coloring_exists

end NUMINAMATH_CALUDE_valid_coloring_exists_l738_73888


namespace NUMINAMATH_CALUDE_symmetry_y_axis_coordinates_l738_73859

/-- A point in a 2D Cartesian coordinate system -/
structure Point where
  x : ℝ
  y : ℝ

/-- Symmetry with respect to the y-axis -/
def symmetricYAxis (p : Point) : Point :=
  { x := -p.x, y := p.y }

theorem symmetry_y_axis_coordinates :
  let A : Point := { x := -1, y := 2 }
  let B : Point := symmetricYAxis A
  B.x = 1 ∧ B.y = 2 := by
  sorry

end NUMINAMATH_CALUDE_symmetry_y_axis_coordinates_l738_73859


namespace NUMINAMATH_CALUDE_point_difference_l738_73825

/-- Represents a basketball player with their score and penalties. -/
structure Player where
  score : ℕ
  penalties : List ℕ

/-- Calculates the final score of a player after subtracting penalties. -/
def finalScore (p : Player) : ℤ :=
  p.score - p.penalties.sum

/-- Represents a basketball team with a list of players. -/
structure Team where
  players : List Player

/-- Calculates the total score of a team. -/
def teamScore (t : Team) : ℤ :=
  t.players.map finalScore |>.sum

/-- The given data for Team A. -/
def teamA : Team := {
  players := [
    { score := 12, penalties := [2] },
    { score := 18, penalties := [2, 2, 2] },
    { score := 5,  penalties := [] },
    { score := 7,  penalties := [3, 3] },
    { score := 6,  penalties := [1] }
  ]
}

/-- The given data for Team B. -/
def teamB : Team := {
  players := [
    { score := 10, penalties := [1, 1] },
    { score := 9,  penalties := [2] },
    { score := 12, penalties := [] },
    { score := 8,  penalties := [1, 1, 1] },
    { score := 5,  penalties := [3] },
    { score := 4,  penalties := [] }
  ]
}

/-- The main theorem stating the point difference between Team B and Team A. -/
theorem point_difference : teamScore teamB - teamScore teamA = 5 := by
  sorry


end NUMINAMATH_CALUDE_point_difference_l738_73825


namespace NUMINAMATH_CALUDE_perpendicular_line_implies_parallel_planes_perpendicular_lines_to_plane_implies_parallel_lines_l738_73874

-- Define the types for lines and planes in space
variable (Line Plane : Type)

-- Define the parallel relation between lines and planes
variable (parallel_line_plane : Line → Plane → Prop)

-- Define the parallel relation between planes
variable (parallel_plane_plane : Plane → Plane → Prop)

-- Define the parallel relation between lines
variable (parallel_line_line : Line → Line → Prop)

-- Define the perpendicular relation between lines and planes
variable (perpendicular_line_plane : Line → Plane → Prop)

-- Theorem 1: If a line is perpendicular to two planes, then those two planes are parallel
theorem perpendicular_line_implies_parallel_planes
  (m : Line) (α β : Plane) :
  perpendicular_line_plane m α →
  perpendicular_line_plane m β →
  parallel_plane_plane α β :=
sorry

-- Theorem 2: If two lines are both perpendicular to the same plane, then those two lines are parallel
theorem perpendicular_lines_to_plane_implies_parallel_lines
  (m n : Line) (α : Plane) :
  perpendicular_line_plane m α →
  perpendicular_line_plane n α →
  parallel_line_line m n :=
sorry

end NUMINAMATH_CALUDE_perpendicular_line_implies_parallel_planes_perpendicular_lines_to_plane_implies_parallel_lines_l738_73874


namespace NUMINAMATH_CALUDE_m_range_l738_73853

theorem m_range : 
  (∀ x, (|x - m| < 1 ↔ 1/3 < x ∧ x < 1/2)) → 
  (-1/2 ≤ m ∧ m ≤ 4/3) :=
by sorry

end NUMINAMATH_CALUDE_m_range_l738_73853


namespace NUMINAMATH_CALUDE_set_intersection_empty_implies_m_range_l738_73828

-- Define the sets M and N
def M : Set ℝ := {x | x^2 + 5*x - 14 < 0}
def N (m : ℝ) : Set ℝ := {x | m < x ∧ x < m + 3}

-- State the theorem
theorem set_intersection_empty_implies_m_range :
  ∀ m : ℝ, (M ∩ N m = ∅) → (m ≤ -10 ∨ m ≥ 2) :=
by sorry

end NUMINAMATH_CALUDE_set_intersection_empty_implies_m_range_l738_73828


namespace NUMINAMATH_CALUDE_arithmetic_progression_not_power_l738_73894

theorem arithmetic_progression_not_power (n : ℕ) (k : ℕ) : 
  let a : ℕ → ℕ := λ i => 4 * i - 2
  ∀ i : ℕ, ∀ r : ℕ, 2 ≤ r → r ≤ n → ¬ ∃ m : ℕ, a i = m ^ r :=
by sorry

end NUMINAMATH_CALUDE_arithmetic_progression_not_power_l738_73894


namespace NUMINAMATH_CALUDE_sophie_bought_five_cupcakes_l738_73887

/-- The number of cupcakes Sophie bought -/
def num_cupcakes : ℕ := sorry

/-- The price of each cupcake in dollars -/
def cupcake_price : ℚ := 2

/-- The number of doughnuts Sophie bought -/
def num_doughnuts : ℕ := 6

/-- The price of each doughnut in dollars -/
def doughnut_price : ℚ := 1

/-- The number of apple pie slices Sophie bought -/
def num_pie_slices : ℕ := 4

/-- The price of each apple pie slice in dollars -/
def pie_slice_price : ℚ := 2

/-- The number of cookies Sophie bought -/
def num_cookies : ℕ := 15

/-- The price of each cookie in dollars -/
def cookie_price : ℚ := 3/5

/-- The total amount Sophie spent in dollars -/
def total_spent : ℚ := 33

/-- Theorem stating that Sophie bought 5 cupcakes -/
theorem sophie_bought_five_cupcakes :
  num_cupcakes = 5 ∧
  (num_cupcakes : ℚ) * cupcake_price +
  (num_doughnuts : ℚ) * doughnut_price +
  (num_pie_slices : ℚ) * pie_slice_price +
  (num_cookies : ℚ) * cookie_price = total_spent :=
by sorry

end NUMINAMATH_CALUDE_sophie_bought_five_cupcakes_l738_73887


namespace NUMINAMATH_CALUDE_rectangular_solid_volume_l738_73896

theorem rectangular_solid_volume 
  (side_area front_area bottom_area : ℝ)
  (h_side : side_area = 20)
  (h_front : front_area = 15)
  (h_bottom : bottom_area = 12) :
  ∃ (a b c : ℝ), 
    a * b = side_area ∧ 
    b * c = front_area ∧ 
    a * c = bottom_area ∧ 
    a * b * c = 60 := by
  sorry

end NUMINAMATH_CALUDE_rectangular_solid_volume_l738_73896


namespace NUMINAMATH_CALUDE_combination_ratio_l738_73817

def combination (n k : ℕ) : ℚ :=
  (Nat.factorial n) / ((Nat.factorial k) * (Nat.factorial (n - k)))

theorem combination_ratio :
  (combination 5 2) / (combination 7 3) = 2 / 7 := by
  sorry

end NUMINAMATH_CALUDE_combination_ratio_l738_73817


namespace NUMINAMATH_CALUDE_largest_common_divisor_540_315_l738_73815

theorem largest_common_divisor_540_315 : Nat.gcd 540 315 = 45 := by
  sorry

end NUMINAMATH_CALUDE_largest_common_divisor_540_315_l738_73815


namespace NUMINAMATH_CALUDE_cylinder_height_relationship_l738_73886

/-- Represents a right circular cylinder --/
structure Cylinder where
  radius : ℝ
  height : ℝ

/-- Given two cylinders with equal volumes and the second radius 20% larger than the first,
    prove that the height of the first cylinder is 44% more than the height of the second --/
theorem cylinder_height_relationship (c1 c2 : Cylinder) 
    (h_volume : c1.radius^2 * c1.height = c2.radius^2 * c2.height)
    (h_radius : c2.radius = 1.2 * c1.radius) :
    c1.height = 1.44 * c2.height := by
  sorry

end NUMINAMATH_CALUDE_cylinder_height_relationship_l738_73886


namespace NUMINAMATH_CALUDE_deepthi_material_usage_l738_73831

theorem deepthi_material_usage
  (material1 : ℚ)
  (material2 : ℚ)
  (leftover : ℚ)
  (h1 : material1 = 4 / 17)
  (h2 : material2 = 3 / 10)
  (h3 : leftover = 9 / 30)
  : material1 + material2 - leftover = 4 / 17 := by
  sorry

end NUMINAMATH_CALUDE_deepthi_material_usage_l738_73831


namespace NUMINAMATH_CALUDE_circle_area_when_radius_equals_three_times_reciprocal_circumference_l738_73806

theorem circle_area_when_radius_equals_three_times_reciprocal_circumference :
  ∀ r : ℝ, r > 0 →
  (3 * (1 / (2 * π * r)) = r) →
  (π * r^2 = 3/2) := by
sorry

end NUMINAMATH_CALUDE_circle_area_when_radius_equals_three_times_reciprocal_circumference_l738_73806


namespace NUMINAMATH_CALUDE_quadratic_real_roots_l738_73854

theorem quadratic_real_roots (k m : ℝ) : 
  (∃ x : ℝ, x^2 + (2*k - 3*m)*x + (k^2 - 5*k*m + 6*m^2) = 0) ↔ k ≥ (15/8)*m :=
by sorry

end NUMINAMATH_CALUDE_quadratic_real_roots_l738_73854


namespace NUMINAMATH_CALUDE_complex_magnitude_squared_l738_73899

theorem complex_magnitude_squared (z : ℂ) (h : z^2 + Complex.abs z ^ 2 = 3 - 5*I) : 
  Complex.abs z ^ 2 = 17/3 := by
  sorry

end NUMINAMATH_CALUDE_complex_magnitude_squared_l738_73899


namespace NUMINAMATH_CALUDE_simplify_and_rationalize_l738_73893

theorem simplify_and_rationalize :
  (Real.sqrt 2 / Real.sqrt 3) * (Real.sqrt 4 / Real.sqrt 5) * (Real.sqrt 6 / Real.sqrt 7) = 
  (4 * Real.sqrt 35) / 35 := by
  sorry

end NUMINAMATH_CALUDE_simplify_and_rationalize_l738_73893


namespace NUMINAMATH_CALUDE_linear_function_composition_l738_73833

/-- Given f(x) = x^2 - 2x + 1 and g(x) is a linear function such that f[g(x)] = 4x^2,
    prove that g(x) = 2x + 1 or g(x) = -2x + 1 -/
theorem linear_function_composition (f g : ℝ → ℝ) :
  (∀ x, f x = x^2 - 2*x + 1) →
  (∃ a b : ℝ, ∀ x, g x = a*x + b) →
  (∀ x, f (g x) = 4 * x^2) →
  (∀ x, g x = 2*x + 1 ∨ g x = -2*x + 1) := by
  sorry

end NUMINAMATH_CALUDE_linear_function_composition_l738_73833


namespace NUMINAMATH_CALUDE_square_area_from_perimeter_l738_73885

/-- Given a square with perimeter 40 meters, its area is 100 square meters. -/
theorem square_area_from_perimeter : 
  ∀ s : Real, 
  (4 * s = 40) → -- perimeter is 40 meters
  (s * s = 100)  -- area is 100 square meters
:= by
  sorry

end NUMINAMATH_CALUDE_square_area_from_perimeter_l738_73885


namespace NUMINAMATH_CALUDE_quadratic_symmetry_implies_ordering_l738_73805

def f (x : ℝ) (b c : ℝ) : ℝ := x^2 + b*x + c

theorem quadratic_symmetry_implies_ordering (b c : ℝ) 
  (h : ∀ t : ℝ, f (2 + t) b c = f (2 - t) b c) : 
  f 2 b c < f 1 b c ∧ f 1 b c < f 4 b c := by
  sorry

end NUMINAMATH_CALUDE_quadratic_symmetry_implies_ordering_l738_73805


namespace NUMINAMATH_CALUDE_expression_value_l738_73847

theorem expression_value (a b c : ℝ) (ha : a = 20) (hb : b = 40) (hc : c = 10) :
  (a - (b - c)) - ((a - b) - c) = 20 := by
  sorry

end NUMINAMATH_CALUDE_expression_value_l738_73847


namespace NUMINAMATH_CALUDE_pyramid_surface_area_l738_73814

/-- The total surface area of a pyramid formed from a cube --/
theorem pyramid_surface_area (a : ℝ) (h : a > 0) : 
  let cube_edge := a
  let base_side := a * Real.sqrt 2 / 2
  let slant_height := 3 * a * Real.sqrt 2 / 4
  let lateral_area := 4 * (base_side * slant_height / 2)
  let base_area := base_side ^ 2
  lateral_area + base_area = 2 * a ^ 2 := by
  sorry

#check pyramid_surface_area

end NUMINAMATH_CALUDE_pyramid_surface_area_l738_73814


namespace NUMINAMATH_CALUDE_coin_and_die_probability_l738_73821

theorem coin_and_die_probability :
  let coin_prob : ℚ := 2/3  -- Probability of heads for the biased coin
  let die_prob : ℚ := 1/6   -- Probability of rolling a 5 on a fair six-sided die
  coin_prob * die_prob = 1/9 := by sorry

end NUMINAMATH_CALUDE_coin_and_die_probability_l738_73821


namespace NUMINAMATH_CALUDE_slope_of_OP_l738_73869

-- Define the ellipse
def is_on_ellipse (x y : ℝ) : Prop := x^2 + 2*y^2 = 2

-- Define the line
def is_on_line (x y k : ℝ) : Prop := x + y = k

-- Define the intersection points
def are_intersection_points (M N : ℝ × ℝ) (k : ℝ) : Prop :=
  is_on_ellipse M.1 M.2 ∧ is_on_ellipse N.1 N.2 ∧
  is_on_line M.1 M.2 k ∧ is_on_line N.1 N.2 k

-- Define the midpoint
def is_midpoint (P M N : ℝ × ℝ) : Prop :=
  P.1 = (M.1 + N.1) / 2 ∧ P.2 = (M.2 + N.2) / 2

-- Theorem statement
theorem slope_of_OP (k : ℝ) (M N P : ℝ × ℝ) :
  are_intersection_points M N k →
  is_midpoint P M N →
  P.2 / P.1 = 1 / 2 :=
sorry

end NUMINAMATH_CALUDE_slope_of_OP_l738_73869


namespace NUMINAMATH_CALUDE_polynomial_factor_implies_coefficients_l738_73867

/-- Given a polynomial px^4 + qx^3 + 40x^2 - 20x + 8 with a factor of 4x^2 - 3x + 2,
    prove that p = 0 and q = -32 -/
theorem polynomial_factor_implies_coefficients
  (p q : ℚ)
  (h : ∃ (r s : ℚ), px^4 + qx^3 + 40*x^2 - 20*x + 8 = (4*x^2 - 3*x + 2) * (r*x^2 + s*x + 4)) :
  p = 0 ∧ q = -32 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_factor_implies_coefficients_l738_73867


namespace NUMINAMATH_CALUDE_remainder_of_expression_l738_73848

theorem remainder_of_expression (n : ℕ) : (1 - 90)^10 % 88 = 1 := by
  sorry

end NUMINAMATH_CALUDE_remainder_of_expression_l738_73848


namespace NUMINAMATH_CALUDE_length_AB_l738_73836

-- Define the parabola
def parabola (x y : ℝ) : Prop := y^2 = 4*x

-- Define the focus of the parabola
def focus : ℝ × ℝ := (1, 0)

-- Define the line with slope 1 passing through the focus
def line (x y : ℝ) : Prop := y = x - 1

-- Define the intersection points A and B
def intersection_points (A B : ℝ × ℝ) : Prop :=
  parabola A.1 A.2 ∧ line A.1 A.2 ∧ parabola B.1 B.2 ∧ line B.1 B.2

-- Theorem statement
theorem length_AB (A B : ℝ × ℝ) (h : intersection_points A B) : 
  Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) = 8 := by sorry

end NUMINAMATH_CALUDE_length_AB_l738_73836


namespace NUMINAMATH_CALUDE_isosceles_triangle_leg_range_l738_73844

theorem isosceles_triangle_leg_range (x : ℝ) : 
  (∃ (base : ℝ), base > 0 ∧ x + x + base = 10 ∧ x + x > base ∧ x + base > x) ↔ 
  (5/2 < x ∧ x < 5) :=
by sorry

end NUMINAMATH_CALUDE_isosceles_triangle_leg_range_l738_73844


namespace NUMINAMATH_CALUDE_min_value_of_f_l738_73850

/-- The function f(x) = 4x^2 - 12x + 9 -/
def f (x : ℝ) : ℝ := 4 * x^2 - 12 * x + 9

/-- The minimum value of f(x) is 0 -/
theorem min_value_of_f : ∃ (x_min : ℝ), ∀ (x : ℝ), f x ≥ f x_min ∧ f x_min = 0 := by
  sorry

end NUMINAMATH_CALUDE_min_value_of_f_l738_73850


namespace NUMINAMATH_CALUDE_negation_of_exists_proposition_l738_73810

theorem negation_of_exists_proposition :
  (¬ ∃ n : ℕ, n^2 > 2^n) ↔ (∀ n : ℕ, n^2 ≤ 2^n) := by sorry

end NUMINAMATH_CALUDE_negation_of_exists_proposition_l738_73810
