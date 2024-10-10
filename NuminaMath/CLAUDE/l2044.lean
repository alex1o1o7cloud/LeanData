import Mathlib

namespace floor_paving_cost_l2044_204443

-- Define the room dimensions and cost per square meter
def room_length : ℝ := 5.5
def room_width : ℝ := 3.75
def cost_per_sq_meter : ℝ := 700

-- Define the function to calculate the total cost
def total_cost (length width cost_per_unit : ℝ) : ℝ :=
  length * width * cost_per_unit

-- Theorem statement
theorem floor_paving_cost :
  total_cost room_length room_width cost_per_sq_meter = 14437.50 := by
  sorry

end floor_paving_cost_l2044_204443


namespace circle_center_l2044_204460

/-- Given a circle with equation x^2 + y^2 - 2x + 4y + 1 = 0, its center is (1, -2) -/
theorem circle_center (x y : ℝ) : 
  (x^2 + y^2 - 2*x + 4*y + 1 = 0) → (∃ r : ℝ, (x - 1)^2 + (y + 2)^2 = r^2) :=
by sorry

end circle_center_l2044_204460


namespace probability_point_in_sphere_l2044_204444

/-- The probability that a randomly selected point (x, y, z) in a cube with side length 2
    centered at the origin lies within a unit sphere centered at the origin. -/
theorem probability_point_in_sphere : 
  let cube_volume : ℝ := 8
  let sphere_volume : ℝ := (4 / 3) * Real.pi
  let prob : ℝ := sphere_volume / cube_volume
  prob = Real.pi / 6 := by sorry

end probability_point_in_sphere_l2044_204444


namespace cubic_inequality_l2044_204471

theorem cubic_inequality (a b : ℝ) (h1 : a ≥ b) (h2 : b > 0) :
  3 * a^3 + 2 * b^3 ≥ 3 * a^2 * b + 2 * a * b^2 := by
  sorry

end cubic_inequality_l2044_204471


namespace diamonds_sequence_property_diamonds_10th_figure_l2044_204415

/-- The number of diamonds in the nth figure of the sequence -/
def diamonds (n : ℕ) : ℕ :=
  if n = 1 then 1
  else if n = 2 then 5
  else 1 + 8 * (n - 1) * n

/-- The sequence satisfies the given conditions -/
theorem diamonds_sequence_property (n : ℕ) (h : n ≥ 3) :
  diamonds n = diamonds (n-1) + 8 * (n-1) :=
sorry

/-- The total number of diamonds in the 10th figure is 721 -/
theorem diamonds_10th_figure :
  diamonds 10 = 721 :=
sorry

end diamonds_sequence_property_diamonds_10th_figure_l2044_204415


namespace integral_x_plus_sqrt_one_minus_x_squared_l2044_204402

open Set
open MeasureTheory
open Interval
open Real

theorem integral_x_plus_sqrt_one_minus_x_squared : 
  ∫ x in (-1 : ℝ)..1, (x + Real.sqrt (1 - x^2)) = π / 2 := by
  sorry

end integral_x_plus_sqrt_one_minus_x_squared_l2044_204402


namespace equal_roots_quadratic_l2044_204436

/-- 
For a quadratic equation x^2 + kx + 1 = 0 to have two equal real roots,
k must equal ±2.
-/
theorem equal_roots_quadratic (k : ℝ) : 
  (∃ x : ℝ, x^2 + k*x + 1 = 0 ∧ 
   ∀ y : ℝ, y^2 + k*y + 1 = 0 → y = x) ↔ 
  k = 2 ∨ k = -2 := by
sorry

end equal_roots_quadratic_l2044_204436


namespace geometric_sequence_log_sum_l2044_204431

/-- A geometric sequence -/
def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

/-- The logarithm function (base 10) -/
noncomputable def lg (x : ℝ) : ℝ := Real.log x / Real.log 10

/-- Theorem: In a geometric sequence where a₂ * a₅ * a₈ = 1, lg(a₄) + lg(a₆) = 0 -/
theorem geometric_sequence_log_sum (a : ℕ → ℝ) 
  (h_geom : geometric_sequence a) 
  (h_prod : a 2 * a 5 * a 8 = 1) : 
  lg (a 4) + lg (a 6) = 0 :=
sorry

end geometric_sequence_log_sum_l2044_204431


namespace work_speed_ratio_is_two_to_one_l2044_204449

def work_speed_ratio (a b : ℚ) : Prop :=
  b = 1 / 12 ∧ a + b = 1 / 4 → a / b = 2

theorem work_speed_ratio_is_two_to_one :
  ∃ a b : ℚ, work_speed_ratio a b :=
by
  sorry

end work_speed_ratio_is_two_to_one_l2044_204449


namespace hd_ha_ratio_specific_triangle_l2044_204456

/-- Represents a triangle with sides a, b, and c -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  positive_a : 0 < a
  positive_b : 0 < b
  positive_c : 0 < c
  triangle_inequality_ab : a + b > c
  triangle_inequality_bc : b + c > a
  triangle_inequality_ca : c + a > b

/-- The orthocenter of a triangle -/
def orthocenter (t : Triangle) : ℝ × ℝ := sorry

/-- The foot of the altitude from a vertex to the opposite side -/
def altitude_foot (t : Triangle) (vertex : ℕ) : ℝ × ℝ := sorry

/-- The vertex of a triangle -/
def vertex (t : Triangle) (v : ℕ) : ℝ × ℝ := sorry

/-- The ratio of distances HD:HA in the triangle -/
def hd_ha_ratio (t : Triangle) : ℝ × ℝ := sorry

theorem hd_ha_ratio_specific_triangle :
  let t : Triangle := ⟨11, 13, 20, sorry, sorry, sorry, sorry, sorry, sorry⟩
  let h := orthocenter t
  let d := altitude_foot t 0  -- Assuming 0 represents vertex A
  let a := vertex t 0
  hd_ha_ratio t = (0, 6.6) := by sorry

end hd_ha_ratio_specific_triangle_l2044_204456


namespace boat_distance_along_stream_l2044_204438

-- Define the speed of the boat in still water
def boat_speed : ℝ := 10

-- Define the distance traveled against the stream in one hour
def distance_against_stream : ℝ := 5

-- Define the time of travel
def travel_time : ℝ := 1

-- Theorem statement
theorem boat_distance_along_stream :
  let stream_speed := boat_speed - distance_against_stream / travel_time
  (boat_speed + stream_speed) * travel_time = 15 := by sorry

end boat_distance_along_stream_l2044_204438


namespace mortgage_payment_l2044_204424

theorem mortgage_payment (total : ℝ) (months : ℕ) (ratio : ℝ) (first_payment : ℝ) :
  total = 109300 ∧ 
  months = 7 ∧ 
  ratio = 3 ∧ 
  total = first_payment * (1 - ratio^months) / (1 - ratio) →
  first_payment = 100 := by
sorry

end mortgage_payment_l2044_204424


namespace common_number_in_list_l2044_204453

theorem common_number_in_list (list : List ℝ) : 
  list.length = 7 →
  (list.take 4).sum / 4 = 7 →
  (list.drop 3).sum / 4 = 9 →
  list.sum / 7 = 8 →
  ∃ x ∈ list.take 4 ∩ list.drop 3, x = 8 := by
sorry

end common_number_in_list_l2044_204453


namespace angle_D_measure_l2044_204454

-- Define the geometric figure
def geometric_figure (B C D E F : Real) : Prop :=
  -- Angle B measures 120°
  B = 120 ∧
  -- Angle B and C form a linear pair
  B + C = 180 ∧
  -- In triangle DEF, angle E = 45°
  E = 45 ∧
  -- Angle F is vertically opposite to angle C
  F = C ∧
  -- Triangle DEF sum of angles
  D + E + F = 180

-- Theorem statement
theorem angle_D_measure (B C D E F : Real) :
  geometric_figure B C D E F → D = 75 := by
  sorry

end angle_D_measure_l2044_204454


namespace lewis_total_earnings_l2044_204482

/-- Lewis's weekly earnings in dollars -/
def weekly_earnings : ℕ := 92

/-- Number of weeks Lewis works during the harvest -/
def weeks_worked : ℕ := 5

/-- Theorem stating Lewis's total earnings during the harvest -/
theorem lewis_total_earnings : weekly_earnings * weeks_worked = 460 := by
  sorry

end lewis_total_earnings_l2044_204482


namespace solve_bath_towels_problem_l2044_204493

def bath_towels_problem (kylie_towels husband_towels : ℕ) 
  (towels_per_load loads : ℕ) : Prop :=
  let total_towels := towels_per_load * loads
  let daughters_towels := total_towels - (kylie_towels + husband_towels)
  daughters_towels = 6

theorem solve_bath_towels_problem : 
  bath_towels_problem 3 3 4 3 := by
  sorry

end solve_bath_towels_problem_l2044_204493


namespace ratio_problem_l2044_204433

/-- Given three positive real numbers A, B, and C with specified ratios,
    prove the fraction of C to A and the ratio of A to C. -/
theorem ratio_problem (A B C : ℝ) (hA : A > 0) (hB : B > 0) (hC : C > 0)
  (hAB : A / B = 7 / 3) (hBC : B / C = 6 / 5) :
  C / A = 5 / 14 ∧ A / C = 14 / 5 := by
  sorry

end ratio_problem_l2044_204433


namespace puppies_ratio_l2044_204481

/-- Puppies problem -/
theorem puppies_ratio (total : ℕ) (kept : ℕ) (price : ℕ) (stud_fee : ℕ) (profit : ℕ) :
  total = 8 →
  kept = 1 →
  price = 600 →
  stud_fee = 300 →
  profit = 1500 →
  (total - kept - (profit + stud_fee) / price : ℚ) / total = 1 / 2 := by
  sorry

end puppies_ratio_l2044_204481


namespace chord_intersection_probability_for_1996_points_chord_intersection_probability_general_l2044_204422

/-- The number of points on the circle -/
def n : ℕ := 1996

/-- The probability that two chords formed by four randomly selected points intersect -/
def chord_intersection_probability (n : ℕ) : ℚ :=
  if n ≥ 4 then 1 / 4 else 0

/-- Theorem stating that the probability of chord intersection is 1/4 for 1996 points -/
theorem chord_intersection_probability_for_1996_points :
  chord_intersection_probability n = 1 / 4 := by
  sorry

/-- Theorem stating that the probability of chord intersection is always 1/4 for n ≥ 4 -/
theorem chord_intersection_probability_general (n : ℕ) (h : n ≥ 4) :
  chord_intersection_probability n = 1 / 4 := by
  sorry

end chord_intersection_probability_for_1996_points_chord_intersection_probability_general_l2044_204422


namespace mollys_age_l2044_204487

/-- Given the ratio of Sandy's age to Molly's age and Sandy's future age, 
    prove Molly's current age -/
theorem mollys_age (sandy_age molly_age : ℕ) : 
  (sandy_age : ℚ) / molly_age = 4 / 3 →
  sandy_age + 6 = 34 →
  molly_age = 21 := by
sorry

end mollys_age_l2044_204487


namespace max_at_neg_two_l2044_204458

/-- The function f(x) that we're analyzing -/
def f (m : ℝ) (x : ℝ) : ℝ := x * (x - m)^2

/-- The derivative of f(x) -/
def f_deriv (m : ℝ) (x : ℝ) : ℝ := (x - m)^2 + 2*x*(x - m)

theorem max_at_neg_two (m : ℝ) :
  (∀ x : ℝ, f m x ≤ f m (-2)) → m = -2 :=
sorry

end max_at_neg_two_l2044_204458


namespace cosine_range_in_geometric_progression_triangle_l2044_204459

theorem cosine_range_in_geometric_progression_triangle (a b c : ℝ) (hpos : a > 0 ∧ b > 0 ∧ c > 0)
  (hacute : 0 < a ^ 2 + b ^ 2 - c ^ 2 ∧ 0 < b ^ 2 + c ^ 2 - a ^ 2 ∧ 0 < c ^ 2 + a ^ 2 - b ^ 2)
  (hgeo : b ^ 2 = a * c) : 1 / 2 ≤ (a ^ 2 + c ^ 2 - b ^ 2) / (2 * a * c) ∧ (a ^ 2 + c ^ 2 - b ^ 2) / (2 * a * c) < 1 := by
  sorry

end cosine_range_in_geometric_progression_triangle_l2044_204459


namespace integer_less_than_sqrt_10_l2044_204401

theorem integer_less_than_sqrt_10 : ∃ n : ℤ, (n : ℝ) < Real.sqrt 10 := by
  sorry

end integer_less_than_sqrt_10_l2044_204401


namespace loan_period_duration_l2044_204435

/-- The amount of money lent (in Rs.) -/
def loanAmount : ℝ := 3150

/-- The interest rate A charges B (as a decimal) -/
def rateAtoB : ℝ := 0.08

/-- The interest rate B charges C (as a decimal) -/
def rateBtoC : ℝ := 0.125

/-- B's total gain over the period (in Rs.) -/
def totalGain : ℝ := 283.5

/-- The duration of the period in years -/
def periodYears : ℝ := 2

theorem loan_period_duration :
  periodYears * (rateBtoC * loanAmount - rateAtoB * loanAmount) = totalGain :=
by sorry

end loan_period_duration_l2044_204435


namespace proposition_b_is_false_l2044_204462

theorem proposition_b_is_false : ¬(∀ x : ℝ, 
  (1 / x < 1 → ¬(-1 ≤ x ∧ x ≤ 1)) ∧ 
  (∃ y : ℝ, ¬(1 / y < 1) ∧ ¬(-1 ≤ y ∧ y ≤ 1))) := by
  sorry

end proposition_b_is_false_l2044_204462


namespace old_man_gold_coins_l2044_204474

theorem old_man_gold_coins (x y : ℕ) (h : x^2 - y^2 = 81 * (x - y)) : x + y = 81 := by
  sorry

end old_man_gold_coins_l2044_204474


namespace new_average_mark_l2044_204400

theorem new_average_mark (n : ℕ) (initial_avg : ℚ) (excluded_n : ℕ) (excluded_avg : ℚ) :
  n = 9 →
  initial_avg = 60 →
  excluded_n = 5 →
  excluded_avg = 44 →
  let total_marks := n * initial_avg
  let excluded_marks := excluded_n * excluded_avg
  let remaining_marks := total_marks - excluded_marks
  let remaining_n := n - excluded_n
  (remaining_marks / remaining_n : ℚ) = 80 := by
  sorry

end new_average_mark_l2044_204400


namespace locus_and_fixed_point_l2044_204407

-- Define the circle M
def circle_M (x y : ℝ) : Prop := (x - 2)^2 + y^2 = 4

-- Define point N
def point_N : ℝ × ℝ := (-2, 0)

-- Define the curve C
def curve_C (x y : ℝ) : Prop := x^2 - y^2 / 3 = 1

-- Define points A₁ and A₂
def point_A1 : ℝ × ℝ := (-1, 0)
def point_A2 : ℝ × ℝ := (1, 0)

-- Define the line x = 2
def line_x_2 (x y : ℝ) : Prop := x = 2

-- Theorem statement
theorem locus_and_fixed_point :
  ∀ (P : ℝ × ℝ) (Q : ℝ × ℝ) (E : ℝ × ℝ) (F : ℝ × ℝ) (A : ℝ × ℝ) (B : ℝ × ℝ),
  circle_M P.1 P.2 →
  line_x_2 E.1 E.2 ∧ line_x_2 F.1 F.2 →
  E.2 = -F.2 →
  curve_C Q.1 Q.2 →
  curve_C A.1 A.2 ∧ curve_C B.1 B.2 →
  (∃ (m : ℝ), A.2 - point_A1.2 = m * (A.1 - point_A1.1) ∧
               E.2 - point_A1.2 = m * (E.1 - point_A1.1)) →
  (∃ (n : ℝ), B.2 - point_A2.2 = n * (B.1 - point_A2.1) ∧
               F.2 - point_A2.2 = n * (F.1 - point_A2.1)) →
  (∃ (k : ℝ), B.2 - A.2 = k * (B.1 - A.1) ∧ 0 = k * (2 - A.1) + A.2) :=
sorry

end locus_and_fixed_point_l2044_204407


namespace f_at_one_l2044_204439

/-- Given a polynomial g(x) with three distinct roots, where each root is also a root of f(x),
    prove that f(1) = -217 -/
theorem f_at_one (a b c : ℝ) : 
  (∃ r₁ r₂ r₃ : ℝ, r₁ ≠ r₂ ∧ r₁ ≠ r₃ ∧ r₂ ≠ r₃ ∧ 
    (∀ x : ℝ, x^3 + a*x^2 + x + 20 = 0 ↔ x = r₁ ∨ x = r₂ ∨ x = r₃)) →
  (∀ x : ℝ, x^3 + a*x^2 + x + 20 = 0 → x^4 + x^3 + b*x^2 + 50*x + c = 0) →
  (1 : ℝ)^4 + (1 : ℝ)^3 + b*(1 : ℝ)^2 + 50*(1 : ℝ) + c = -217 :=
by sorry

end f_at_one_l2044_204439


namespace smallest_base_for_repeating_decimal_l2044_204427

/-- Represents a repeating decimal in base k -/
def RepeatingDecimal (k : ℕ) (n : ℕ) := (k : ℚ) ^ 2 / ((k : ℚ) ^ 2 - 1) * (4 * k + 1)

/-- The smallest integer k > 10 such that 17/85 has a repeating decimal representation of 0.414141... in base k -/
theorem smallest_base_for_repeating_decimal :
  ∃ (k : ℕ), k > 10 ∧ RepeatingDecimal k 2 = 17 / 85 ∧
  ∀ (m : ℕ), m > 10 ∧ m < k → RepeatingDecimal m 2 ≠ 17 / 85 := by
  sorry

end smallest_base_for_repeating_decimal_l2044_204427


namespace circle_ratio_l2044_204478

theorem circle_ratio (r R : ℝ) (h1 : r > 0) (h2 : R > 0) (h3 : r ≤ R) : 
  π * R^2 = 3 * (π * R^2 - π * r^2) → R / r = Real.sqrt (3 / 2) := by
  sorry

end circle_ratio_l2044_204478


namespace circle_radius_reduction_l2044_204408

/-- Given a circle with an initial radius of 5 cm, if its area is reduced by 36%, the new radius will be 4 cm. -/
theorem circle_radius_reduction (π : ℝ) (h_π_pos : π > 0) : 
  let r₁ : ℝ := 5
  let A₁ : ℝ := π * r₁^2
  let A₂ : ℝ := 0.64 * A₁
  let r₂ : ℝ := Real.sqrt (A₂ / π)
  r₂ = 4 := by sorry

end circle_radius_reduction_l2044_204408


namespace ellipse_axis_lengths_l2044_204455

/-- Given an ellipse with equation x²/16 + y²/25 = 1, prove that its major axis length is 10 and its minor axis length is 8 -/
theorem ellipse_axis_lengths :
  let ellipse := {(x, y) : ℝ × ℝ | x^2/16 + y^2/25 = 1}
  ∃ (major_axis minor_axis : ℝ),
    major_axis = 10 ∧
    minor_axis = 8 ∧
    (∀ (p : ℝ × ℝ), p ∈ ellipse →
      (p.1^2 + p.2^2 ≤ (major_axis/2)^2 ∧
       p.1^2 + p.2^2 ≥ (minor_axis/2)^2)) :=
by sorry

end ellipse_axis_lengths_l2044_204455


namespace age_ratio_problem_l2044_204447

/-- Sam's current age -/
def s : ℕ := by sorry

/-- Anna's current age -/
def a : ℕ := by sorry

/-- The number of years until the ratio of their ages is 3:2 -/
def x : ℕ := by sorry

theorem age_ratio_problem :
  (s - 3 = 4 * (a - 3)) ∧ 
  (s - 5 = 6 * (a - 5)) →
  (x = 22 ∧ (s + x) * 2 = (a + x) * 3) := by sorry

end age_ratio_problem_l2044_204447


namespace no_small_order_of_two_l2044_204442

theorem no_small_order_of_two (p : ℕ) (h1 : Prime p) (h2 : ∃ k : ℕ, p = 4 * k + 1) (h3 : Prime (2 * p + 1)) :
  ¬ ∃ k : ℕ, k < 2 * p ∧ (2 : ZMod (2 * p + 1))^k = 1 := by
  sorry

end no_small_order_of_two_l2044_204442


namespace amount_after_two_years_l2044_204413

/-- Calculate the amount after n years with a given initial value and annual increase rate -/
def amountAfterYears (initialValue : ℝ) (rate : ℝ) (years : ℕ) : ℝ :=
  initialValue * (1 + rate) ^ years

/-- Theorem: Given an initial amount of 6400 and an annual increase rate of 1/8,
    the amount after 2 years will be 8100 -/
theorem amount_after_two_years :
  let initialValue : ℝ := 6400
  let rate : ℝ := 1/8
  let years : ℕ := 2
  amountAfterYears initialValue rate years = 8100 := by
  sorry

end amount_after_two_years_l2044_204413


namespace purely_imaginary_condition_l2044_204440

theorem purely_imaginary_condition (m : ℝ) : 
  (((2 * m ^ 2 - 3 * m - 2) : ℂ) + (m ^ 2 - 3 * m + 2) * Complex.I).im ≠ 0 ∧ 
  (((2 * m ^ 2 - 3 * m - 2) : ℂ) + (m ^ 2 - 3 * m + 2) * Complex.I).re = 0 ↔ 
  m = -1/2 := by sorry

end purely_imaginary_condition_l2044_204440


namespace bank_deposit_years_l2044_204477

/-- Proves that the number of years for the second bank deposit is 5 given the problem conditions. -/
theorem bank_deposit_years (principal : ℚ) (rate : ℚ) (years1 : ℚ) (interest_diff : ℚ) 
  (h1 : principal = 640)
  (h2 : rate = 15 / 100)
  (h3 : years1 = 7 / 2)
  (h4 : interest_diff = 144) :
  ∃ (years2 : ℚ), 
    principal * rate * years2 - principal * rate * years1 = interest_diff ∧ 
    years2 = 5 := by
  sorry


end bank_deposit_years_l2044_204477


namespace pyramid_intersection_volume_l2044_204411

/-- The length of each edge of the pyramids -/
def edge_length : ℝ := 12

/-- The volume of the solid of intersection of two regular square pyramids -/
def intersection_volume : ℝ := 72

/-- Theorem stating the volume of the solid of intersection of two regular square pyramids -/
theorem pyramid_intersection_volume :
  let pyramids : ℕ := 2
  let base_parallel : Prop := True  -- Represents that bases are parallel
  let edges_parallel : Prop := True  -- Represents that edges are parallel
  let apex_at_center : Prop := True  -- Represents that each apex is at the center of the other base
  intersection_volume = 72 :=
by sorry

end pyramid_intersection_volume_l2044_204411


namespace min_distance_sum_parabola_l2044_204423

/-- The minimum distance sum for a point on the parabola x = (1/4)y^2 -/
theorem min_distance_sum_parabola :
  let parabola := {P : ℝ × ℝ | P.1 = (1/4) * P.2^2}
  let dist_to_A (P : ℝ × ℝ) := Real.sqrt ((P.1 - 0)^2 + (P.2 - 1)^2)
  let dist_to_y_axis (P : ℝ × ℝ) := |P.1|
  ∃ (min_val : ℝ), min_val = Real.sqrt 2 - 1 ∧
    ∀ P ∈ parabola, dist_to_A P + dist_to_y_axis P ≥ min_val :=
by sorry

end min_distance_sum_parabola_l2044_204423


namespace connie_red_markers_l2044_204412

/-- The number of red markers Connie has -/
def red_markers : ℕ := 3343 - 1028

/-- The total number of markers Connie has -/
def total_markers : ℕ := 3343

/-- The number of blue markers Connie has -/
def blue_markers : ℕ := 1028

theorem connie_red_markers :
  red_markers = 2315 ∧ total_markers = red_markers + blue_markers :=
sorry

end connie_red_markers_l2044_204412


namespace first_digit_base_9_of_21221122211112211111_base_3_l2044_204432

def base_3_num : List Nat := [2, 1, 2, 2, 1, 1, 2, 2, 2, 1, 1, 1, 1, 2, 2, 1, 1, 1, 1, 1]

def to_base_10 (digits : List Nat) : Nat :=
  digits.enum.foldr (fun (i, d) acc => acc + d * 3^(digits.length - 1 - i)) 0

def first_digit_base_9 (n : Nat) : Nat :=
  if n = 0 then 0 else
  let log_9_n := (Nat.log n 9)
  n / 9^log_9_n

theorem first_digit_base_9_of_21221122211112211111_base_3 :
  first_digit_base_9 (to_base_10 base_3_num) = 3 := by
  sorry

end first_digit_base_9_of_21221122211112211111_base_3_l2044_204432


namespace nina_running_distance_l2044_204425

/-- Conversion factor from kilometers to miles -/
def km_to_miles : ℝ := 0.621371

/-- Conversion factor from yards to miles -/
def yard_to_miles : ℝ := 0.000568182

/-- Distance Nina ran in miles for her initial run -/
def initial_run : ℝ := 0.08

/-- Distance Nina ran in kilometers for her second run (done twice) -/
def second_run_km : ℝ := 3

/-- Distance Nina ran in yards for her third run -/
def third_run_yards : ℝ := 1200

/-- Distance Nina ran in kilometers for her final run -/
def final_run_km : ℝ := 6

/-- Total distance Nina ran in miles -/
def total_distance : ℝ := 
  initial_run + 
  2 * (second_run_km * km_to_miles) + 
  (third_run_yards * yard_to_miles) + 
  (final_run_km * km_to_miles)

theorem nina_running_distance : 
  ∃ ε > 0, |total_distance - 8.22| < ε :=
sorry

end nina_running_distance_l2044_204425


namespace trigonometric_simplification_l2044_204448

theorem trigonometric_simplification (α : ℝ) : 
  (1 + Real.tan (2 * α))^2 - 2 * (Real.tan (2 * α))^2 / (1 + (Real.tan (2 * α))^2) - 
  Real.sin (4 * α) - 1 = -2 * (Real.sin (2 * α))^2 := by sorry

end trigonometric_simplification_l2044_204448


namespace lemon_pie_angle_l2044_204484

theorem lemon_pie_angle (total : ℕ) (chocolate : ℕ) (apple : ℕ) (blueberry : ℕ) 
  (h_total : total = 40)
  (h_chocolate : chocolate = 15)
  (h_apple : apple = 10)
  (h_blueberry : blueberry = 5)
  (h_remaining : total - (chocolate + apple + blueberry) = 2 * (total - (chocolate + apple + blueberry)) / 2) :
  (((total - (chocolate + apple + blueberry)) / 2) : ℚ) / total * 360 = 45 := by
sorry

end lemon_pie_angle_l2044_204484


namespace dodecagon_diagonals_l2044_204405

/-- The number of diagonals in a convex polygon with n sides -/
def num_diagonals (n : ℕ) : ℕ := n * (n - 3) / 2

/-- A convex dodecagon has 54 diagonals -/
theorem dodecagon_diagonals : num_diagonals 12 = 54 := by
  sorry

end dodecagon_diagonals_l2044_204405


namespace circles_radius_order_l2044_204434

noncomputable def circle_A_radius : ℝ := 3

noncomputable def circle_B_area : ℝ := 12 * Real.pi

noncomputable def circle_C_area : ℝ := 28 * Real.pi

noncomputable def circle_B_radius : ℝ := Real.sqrt (circle_B_area / Real.pi)

noncomputable def circle_C_radius : ℝ := Real.sqrt (circle_C_area / Real.pi)

theorem circles_radius_order :
  circle_A_radius < circle_B_radius ∧ circle_B_radius < circle_C_radius := by
  sorry

end circles_radius_order_l2044_204434


namespace parabola_properties_l2044_204416

/-- Given a parabola y = x^2 - 8x + 12, prove its properties -/
theorem parabola_properties :
  let f (x : ℝ) := x^2 - 8*x + 12
  ∃ (axis vertex_x vertex_y x1 x2 : ℝ),
    -- The axis of symmetry
    axis = 4 ∧
    -- The vertex coordinates
    f vertex_x = vertex_y ∧
    vertex_x = 4 ∧
    vertex_y = -4 ∧
    -- The x-axis intersection points
    f x1 = 0 ∧
    f x2 = 0 ∧
    x1 = 2 ∧
    x2 = 6 := by
  sorry

end parabola_properties_l2044_204416


namespace min_cubes_for_box_l2044_204461

/-- Proves that the minimum number of 5 cubic cm cubes required to build a box
    with dimensions 10 cm × 13 cm × 5 cm is 130. -/
theorem min_cubes_for_box (box_length box_width box_height cube_volume : ℕ)
  (h1 : box_length = 10)
  (h2 : box_width = 13)
  (h3 : box_height = 5)
  (h4 : cube_volume = 5) :
  (box_length * box_width * box_height) / cube_volume = 130 := by
  sorry

end min_cubes_for_box_l2044_204461


namespace quadratic_completion_sum_l2044_204419

theorem quadratic_completion_sum (x : ℝ) : ∃ (m n : ℝ), 
  (x^2 - 8*x + 3 = 0 ↔ (x - m)^2 = n) ∧ m + n = 17 := by
  sorry

end quadratic_completion_sum_l2044_204419


namespace num_distinct_lines_is_seven_l2044_204465

/-- A right triangle with two 45-degree angles at the base -/
structure RightIsoscelesTriangle where
  /-- The right angle of the triangle -/
  right_angle : Angle
  /-- One of the 45-degree angles at the base -/
  base_angle1 : Angle
  /-- The other 45-degree angle at the base -/
  base_angle2 : Angle
  /-- The right angle is 90 degrees -/
  right_angle_is_right : right_angle = 90
  /-- The base angles are each 45 degrees -/
  base_angles_are_45 : base_angle1 = 45 ∧ base_angle2 = 45

/-- The number of distinct lines formed by altitudes, medians, and angle bisectors -/
def num_distinct_lines (t : RightIsoscelesTriangle) : ℕ := sorry

/-- Theorem stating that the number of distinct lines is 7 -/
theorem num_distinct_lines_is_seven (t : RightIsoscelesTriangle) : 
  num_distinct_lines t = 7 := by sorry

end num_distinct_lines_is_seven_l2044_204465


namespace product_remainder_mod_five_l2044_204473

theorem product_remainder_mod_five : (1657 * 2024 * 1953 * 1865) % 5 = 0 := by
  sorry

end product_remainder_mod_five_l2044_204473


namespace library_books_l2044_204417

theorem library_books (initial_books : ℕ) : 
  (initial_books : ℚ) * (2 / 6) = 3300 → initial_books = 9900 := by
  sorry

end library_books_l2044_204417


namespace gas_cost_per_gallon_l2044_204492

theorem gas_cost_per_gallon (miles_per_gallon : ℝ) (total_miles : ℝ) (total_cost : ℝ) :
  miles_per_gallon = 32 →
  total_miles = 336 →
  total_cost = 42 →
  (total_cost / (total_miles / miles_per_gallon)) = 4 := by
sorry

end gas_cost_per_gallon_l2044_204492


namespace area_difference_sheets_l2044_204468

/-- The difference in combined area (front and back) between a square sheet of paper
    with side length 11 inches and a rectangular sheet of paper measuring 5.5 inches
    by 11 inches is equal to 121 square inches. -/
theorem area_difference_sheets : 
  let square_sheet_side : ℝ := 11
  let rect_sheet_length : ℝ := 11
  let rect_sheet_width : ℝ := 5.5
  let square_sheet_area : ℝ := 2 * square_sheet_side * square_sheet_side
  let rect_sheet_area : ℝ := 2 * rect_sheet_length * rect_sheet_width
  square_sheet_area - rect_sheet_area = 121 := by
sorry

end area_difference_sheets_l2044_204468


namespace triangle_inequality_l2044_204451

theorem triangle_inequality (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c)
  (htri : a + b > c ∧ b + c > a ∧ c + a > b) :
  (c + a - b)^4 / (a * (a + b - c)) +
  (a + b - c)^4 / (b * (b + c - a)) +
  (b + c - a)^4 / (c * (c + a - b)) ≥
  a^2 + b^2 + c^2 := by
sorry

end triangle_inequality_l2044_204451


namespace gps_primary_benefit_l2044_204403

/-- Represents the capabilities of GPS technology -/
structure GPSTechnology where
  navigation : Bool
  routeOptimization : Bool
  costReduction : Bool

/-- Represents the uses of GPS in mobile phones -/
structure GPSUses where
  travel : Bool
  tourism : Bool
  exploration : Bool

/-- Represents the primary benefit of GPS technology in daily life -/
def primaryBenefit (tech : GPSTechnology) : Prop :=
  tech.routeOptimization ∧ tech.costReduction

/-- The theorem stating that given GPS is used for travel, tourism, and exploration,
    its primary benefit is route optimization and cost reduction -/
theorem gps_primary_benefit (uses : GPSUses) (tech : GPSTechnology) 
  (h1 : uses.travel = true)
  (h2 : uses.tourism = true)
  (h3 : uses.exploration = true)
  (h4 : tech.navigation = true) :
  primaryBenefit tech :=
sorry

end gps_primary_benefit_l2044_204403


namespace mod_inverse_sum_17_l2044_204446

theorem mod_inverse_sum_17 :
  ∃ (a b : ℤ), (2 * a) % 17 = 1 ∧ (4 * b) % 17 = 1 ∧ (a + b) % 17 = 5 := by
  sorry

end mod_inverse_sum_17_l2044_204446


namespace eric_white_marbles_l2044_204479

theorem eric_white_marbles (total : ℕ) (blue : ℕ) (green : ℕ) (white : ℕ) 
  (h1 : total = 20) 
  (h2 : blue = 6) 
  (h3 : green = 2) 
  (h4 : total = white + blue + green) : 
  white = 12 := by
  sorry

end eric_white_marbles_l2044_204479


namespace find_t_l2044_204441

/-- The number of hours I worked -/
def my_hours (t : ℝ) : ℝ := 2*t + 2

/-- My hourly rate in dollars -/
def my_rate (t : ℝ) : ℝ := 4*t - 4

/-- The number of hours Emily worked -/
def emily_hours (t : ℝ) : ℝ := 4*t - 2

/-- Emily's hourly rate in dollars -/
def emily_rate (t : ℝ) : ℝ := t + 3

/-- My total earnings -/
def my_earnings (t : ℝ) : ℝ := my_hours t * my_rate t

/-- Emily's total earnings -/
def emily_earnings (t : ℝ) : ℝ := emily_hours t * emily_rate t

theorem find_t : ∃ t : ℝ, t > 0 ∧ my_earnings t = emily_earnings t + 6 := by
  sorry

end find_t_l2044_204441


namespace triangle_with_same_color_and_unit_area_l2044_204406

-- Define a color type
inductive Color
  | Red
  | Green
  | Blue

-- Define a point in a plane
structure Point where
  x : ℝ
  y : ℝ

-- Define a function that assigns a color to each point in the plane
def colorFunction : Point → Color := sorry

-- Define a function to calculate the area of a triangle given three points
def triangleArea (p1 p2 p3 : Point) : ℝ := sorry

-- Theorem statement
theorem triangle_with_same_color_and_unit_area :
  ∃ (p1 p2 p3 : Point),
    colorFunction p1 = colorFunction p2 ∧
    colorFunction p2 = colorFunction p3 ∧
    triangleArea p1 p2 p3 = 1 := by
  sorry

end triangle_with_same_color_and_unit_area_l2044_204406


namespace circumcircle_tangency_l2044_204496

-- Define the points
variable (A B C D E F : EuclideanPlane)

-- Define the parallelogram ABCD
def is_parallelogram (A B C D : EuclideanPlane) : Prop := sorry

-- Define that E is on BC
def point_on_segment (P Q R : EuclideanPlane) : Prop := sorry

-- Define that F is on AD
-- (We can reuse the point_on_segment definition)

-- Define the circumcircle of a triangle
def circumcircle (P Q R : EuclideanPlane) : Set EuclideanPlane := sorry

-- Define a line being tangent to a circle
def is_tangent (line : Set EuclideanPlane) (circle : Set EuclideanPlane) : Prop := sorry

-- Define a line segment
def line_segment (P Q : EuclideanPlane) : Set EuclideanPlane := sorry

-- The main theorem
theorem circumcircle_tangency 
  (h_parallelogram : is_parallelogram A B C D)
  (h_E_on_BC : point_on_segment B E C)
  (h_F_on_AD : point_on_segment A F D)
  (h_ABE_tangent_CF : is_tangent (line_segment C F) (circumcircle A B E)) :
  is_tangent (line_segment A E) (circumcircle C D F) := by
  sorry

end circumcircle_tangency_l2044_204496


namespace near_integer_intervals_l2044_204495

-- Definition of "near-integer interval"
def near_integer_interval (T : ℝ) : Set ℝ :=
  {x | ∃ (m n : ℤ), m < T ∧ T < n ∧ x ∈ Set.Ioo (↑m : ℝ) (↑n : ℝ) ∧
    ∀ (k : ℤ), k ≤ m ∨ n ≤ k}

-- Theorem statement
theorem near_integer_intervals :
  (near_integer_interval (Real.sqrt 5) = Set.Ioo 2 3) ∧
  (near_integer_interval (-Real.sqrt 10) = Set.Ioo (-4) (-3)) ∧
  (∀ (x y : ℝ), y = Real.sqrt (x - 2023) + Real.sqrt (2023 - x) →
    near_integer_interval (Real.sqrt (x + y)) = Set.Ioo 44 45) :=
by sorry

end near_integer_intervals_l2044_204495


namespace molecular_weight_calculation_l2044_204466

/-- Given 5 moles of a compound with a total molecular weight of 1170,
    prove that the molecular weight of 1 mole of the compound is 234. -/
theorem molecular_weight_calculation (total_weight : ℝ) (num_moles : ℝ) :
  total_weight = 1170 →
  num_moles = 5 →
  total_weight / num_moles = 234 := by
sorry

end molecular_weight_calculation_l2044_204466


namespace brownies_in_container_l2044_204486

/-- Represents the problem of calculating the fraction of remaining brownies in the container --/
theorem brownies_in_container (batches : ℕ) (brownies_per_batch : ℕ) 
  (bake_sale_fraction : ℚ) (given_out : ℕ) : 
  batches = 10 →
  brownies_per_batch = 20 →
  bake_sale_fraction = 3/4 →
  given_out = 20 →
  let total_brownies := batches * brownies_per_batch
  let bake_sale_brownies := (bake_sale_fraction * brownies_per_batch) * batches
  let remaining_after_bake_sale := total_brownies - bake_sale_brownies
  let remaining_after_given_out := remaining_after_bake_sale - given_out
  (remaining_after_given_out : ℚ) / remaining_after_bake_sale = 3/5 := by
  sorry

end brownies_in_container_l2044_204486


namespace quadratic_inequality_solution_sets_l2044_204426

theorem quadratic_inequality_solution_sets 
  (a b : ℝ) 
  (h : Set.Ioo 2 3 = {x | x^2 + a*x + b < 0}) : 
  {x : ℝ | b*x^2 + a*x + 1 > 0} = Set.Iic (1/3) ∪ Set.Ioi (1/2) :=
sorry

end quadratic_inequality_solution_sets_l2044_204426


namespace line_parabola_intersection_l2044_204445

/-- Given a line intersecting y = x^2 at (x₁, x₁²) and (x₂, x₂²), and the x-axis at (x₃, 0),
    prove that 1/x₁ + 1/x₂ = 1/x₃ -/
theorem line_parabola_intersection (x₁ x₂ x₃ : ℝ) (h₁ : x₁ ≠ 0) (h₂ : x₂ ≠ 0) (h₃ : x₃ ≠ 0)
  (h_line : ∃ (k m : ℝ), x₁^2 = k * x₁ + m ∧ x₂^2 = k * x₂ + m ∧ 0 = k * x₃ + m) :
  1 / x₁ + 1 / x₂ = 1 / x₃ := by
  sorry

end line_parabola_intersection_l2044_204445


namespace negation_equivalence_l2044_204467

theorem negation_equivalence :
  (¬ ∀ a b : ℝ, a = b → a^2 = a*b) ↔ (∀ a b : ℝ, a ≠ b → a^2 ≠ a*b) := by
  sorry

end negation_equivalence_l2044_204467


namespace holey_iff_presentable_l2044_204490

/-- A function is holey if there exists an interval free of its values -/
def IsHoley (f : ℝ → ℝ) : Prop :=
  ∃ (a b : ℝ), a < b ∧ ∀ x, a < x ∧ x < b → ∀ y, f y ≠ x

/-- A function is presentable if it can be represented as a composition of linear, inverse, and quadratic functions -/
inductive Presentable : (ℝ → ℝ) → Prop
  | linear (k b : ℝ) : Presentable (fun x ↦ k * x + b)
  | inverse : Presentable (fun x ↦ 1 / x)
  | square : Presentable (fun x ↦ x ^ 2)
  | comp {f g : ℝ → ℝ} (hf : Presentable f) (hg : Presentable g) : Presentable (f ∘ g)

/-- The main theorem statement -/
theorem holey_iff_presentable (a b c d : ℝ) 
    (h : ∀ x, x^2 + a*x + b ≠ 0 ∨ x^2 + c*x + d ≠ 0) : 
    IsHoley (fun x ↦ (x^2 + a*x + b) / (x^2 + c*x + d)) ↔ 
    Presentable (fun x ↦ (x^2 + a*x + b) / (x^2 + c*x + d)) :=
  sorry

end holey_iff_presentable_l2044_204490


namespace complement_M_intersect_N_l2044_204483

-- Define the sets M and N
def M : Set ℝ := {x : ℝ | -2 ≤ x ∧ x ≤ 2}
def N : Set ℝ := {x : ℝ | x < 1}

-- State the theorem
theorem complement_M_intersect_N :
  (Set.compl M) ∩ N = {x : ℝ | x < -2} := by sorry

end complement_M_intersect_N_l2044_204483


namespace min_C_for_inequality_l2044_204485

/-- The minimum value of C that satisfies the given inequality for all x and any α, β where |α| ≤ 1 and |β| ≤ 1 -/
theorem min_C_for_inequality : 
  (∃ (C : ℝ), ∀ (x α β : ℝ), |α| ≤ 1 → |β| ≤ 1 → |α * Real.sin x + β * Real.cos (4 * x)| ≤ C) ∧ 
  (∀ (C : ℝ), (∀ (x α β : ℝ), |α| ≤ 1 → |β| ≤ 1 → |α * Real.sin x + β * Real.cos (4 * x)| ≤ C) → C ≥ 2) :=
by sorry

end min_C_for_inequality_l2044_204485


namespace pictures_on_back_l2044_204450

theorem pictures_on_back (total : ℕ) (front : ℕ) (back : ℕ) 
  (h1 : total = 15) 
  (h2 : front = 6) 
  (h3 : total = front + back) : 
  back = 9 := by
  sorry

end pictures_on_back_l2044_204450


namespace paving_rate_calculation_l2044_204480

-- Define the room dimensions and total cost
def roomLength : Real := 6.5
def roomWidth : Real := 2.75
def totalCost : Real := 10725

-- Define the theorem
theorem paving_rate_calculation :
  let area := roomLength * roomWidth
  let ratePerSqMetre := totalCost / area
  ratePerSqMetre = 600 := by
  sorry


end paving_rate_calculation_l2044_204480


namespace coin_value_difference_l2044_204497

/-- Represents the number of coins of each type -/
structure CoinCount where
  pennies : ℕ
  nickels : ℕ
  dimes : ℕ

/-- Calculates the total value in cents for a given coin count -/
def totalValue (coins : CoinCount) : ℕ :=
  coins.pennies + 5 * coins.nickels + 10 * coins.dimes

/-- The total number of coins Maria has -/
def totalCoins : ℕ := 3030

/-- Predicate to check if a coin count is valid for Maria -/
def isValidCount (coins : CoinCount) : Prop :=
  coins.pennies ≥ 1 ∧ coins.nickels ≥ 1 ∧ coins.dimes ≥ 1 ∧
  coins.pennies + coins.nickels + coins.dimes = totalCoins

/-- Theorem stating the difference between max and min possible values -/
theorem coin_value_difference :
  ∃ (maxCoins minCoins : CoinCount),
    isValidCount maxCoins ∧ isValidCount minCoins ∧
    (∀ c, isValidCount c → totalValue c ≤ totalValue maxCoins) ∧
    (∀ c, isValidCount c → totalValue c ≥ totalValue minCoins) ∧
    totalValue maxCoins - totalValue minCoins = 27243 :=
  sorry

end coin_value_difference_l2044_204497


namespace exactly_four_intersections_l2044_204491

-- Define the graphs
def graph1 (B : ℝ) (x y : ℝ) : Prop := y = B * x^2
def graph2 (x y : ℝ) : Prop := y^2 + 2 * x^2 = 5 + 6 * y

-- Define an intersection point
def is_intersection (B : ℝ) (x y : ℝ) : Prop :=
  graph1 B x y ∧ graph2 x y

-- Theorem statement
theorem exactly_four_intersections (B : ℝ) (h : B > 0) :
  ∃ (x₁ y₁ x₂ y₂ x₃ y₃ x₄ y₄ : ℝ),
    is_intersection B x₁ y₁ ∧
    is_intersection B x₂ y₂ ∧
    is_intersection B x₃ y₃ ∧
    is_intersection B x₄ y₄ ∧
    (x₁ ≠ x₂ ∨ y₁ ≠ y₂) ∧
    (x₁ ≠ x₃ ∨ y₁ ≠ y₃) ∧
    (x₁ ≠ x₄ ∨ y₁ ≠ y₄) ∧
    (x₂ ≠ x₃ ∨ y₂ ≠ y₃) ∧
    (x₂ ≠ x₄ ∨ y₂ ≠ y₄) ∧
    (x₃ ≠ x₄ ∨ y₃ ≠ y₄) ∧
    ∀ (x y : ℝ), is_intersection B x y →
      ((x = x₁ ∧ y = y₁) ∨ (x = x₂ ∧ y = y₂) ∨
       (x = x₃ ∧ y = y₃) ∨ (x = x₄ ∧ y = y₄)) :=
by sorry

end exactly_four_intersections_l2044_204491


namespace closest_to_sqrt_difference_l2044_204464

theorem closest_to_sqrt_difference : 
  let diff := Real.sqrt 101 - Real.sqrt 99
  let options := [0.10, 0.12, 0.14, 0.16, 0.18]
  ∀ x ∈ options, x ≠ 0.10 → |diff - 0.10| < |diff - x| :=
by sorry

end closest_to_sqrt_difference_l2044_204464


namespace violet_percentage_l2044_204404

/-- Represents a flower bouquet with yellow and purple flowers -/
structure Bouquet where
  total : ℕ
  yellow : ℕ
  purple : ℕ
  yellow_daisies : ℕ
  purple_violets : ℕ

/-- Conditions for the flower bouquet -/
def bouquet_conditions (b : Bouquet) : Prop :=
  b.total > 0 ∧
  b.yellow + b.purple = b.total ∧
  b.yellow = b.total / 2 ∧
  b.yellow_daisies = b.yellow / 5 ∧
  b.purple_violets = b.purple / 2

/-- Theorem: The percentage of violets in the bouquet is 25% -/
theorem violet_percentage (b : Bouquet) (h : bouquet_conditions b) :
  (b.purple_violets : ℚ) / b.total = 1/4 := by
  sorry

#check violet_percentage

end violet_percentage_l2044_204404


namespace apple_bags_theorem_l2044_204494

def is_valid_total (n : ℕ) : Prop :=
  70 ≤ n ∧ n ≤ 80 ∧ ∃ (a b : ℕ), n = 12 * a + 6 * b

theorem apple_bags_theorem :
  ∀ n : ℕ, is_valid_total n ↔ (n = 72 ∨ n = 78) :=
by sorry

end apple_bags_theorem_l2044_204494


namespace age_difference_l2044_204421

/-- Given that the sum of X and Y is 15 years greater than the sum of Y and Z,
    prove that Z is 1.5 decades younger than X. -/
theorem age_difference (X Y Z : ℕ) (h : X + Y = Y + Z + 15) :
  (X - Z : ℚ) / 10 = 3/2 := by
  sorry

end age_difference_l2044_204421


namespace no_x_squared_term_l2044_204489

theorem no_x_squared_term (m : ℚ) : 
  (∀ x, (x + 1) * (x^2 + 5*m*x + 3) = x^3 + (3 + 5*m)*x + 3) → m = -1/5 := by
  sorry

end no_x_squared_term_l2044_204489


namespace work_completion_time_l2044_204418

/-- Worker rates and work completion time -/
theorem work_completion_time
  (rate_a rate_b rate_c rate_d : ℝ)
  (total_work : ℝ)
  (h1 : rate_a = 1.5 * rate_b)
  (h2 : rate_a * 30 = total_work)
  (h3 : rate_c = 2 * rate_b)
  (h4 : rate_d = 0.5 * rate_a)
  : ∃ (days : ℕ), days = 12 ∧ 
    (1.25 * rate_b + 2.75 * rate_b) * (days : ℝ) ≥ total_work ∧
    (1.25 * rate_b + 2.75 * rate_b) * ((days - 1) : ℝ) < total_work :=
by sorry


end work_completion_time_l2044_204418


namespace conner_needs_27_rocks_l2044_204414

/-- Calculates the number of rocks Conner needs to collect on day 3 to at least tie with Sydney -/
def rocks_conner_needs_day3 (sydney_initial : ℕ) (conner_initial : ℕ) 
  (sydney_day1 : ℕ) (conner_day1_multiplier : ℕ) 
  (sydney_day2 : ℕ) (conner_day2 : ℕ) 
  (sydney_day3_multiplier : ℕ) : ℕ :=
  let conner_day1 := sydney_day1 * conner_day1_multiplier
  let sydney_day3 := conner_day1 * sydney_day3_multiplier
  let sydney_total := sydney_initial + sydney_day1 + sydney_day2 + sydney_day3
  let conner_before_day3 := conner_initial + conner_day1 + conner_day2
  sydney_total - conner_before_day3

/-- Theorem stating that Conner needs to collect 27 rocks on day 3 to at least tie with Sydney -/
theorem conner_needs_27_rocks : 
  rocks_conner_needs_day3 837 723 4 8 0 123 2 = 27 := by
  sorry

end conner_needs_27_rocks_l2044_204414


namespace five_and_half_hours_in_seconds_l2044_204488

/-- Converts hours to seconds -/
def hours_to_seconds (hours : ℝ) : ℝ :=
  hours * 60 * 60

/-- Theorem: 5.5 hours is equal to 19800 seconds -/
theorem five_and_half_hours_in_seconds : 
  hours_to_seconds 5.5 = 19800 := by sorry

end five_and_half_hours_in_seconds_l2044_204488


namespace min_value_expression_min_value_equality_l2044_204472

theorem min_value_expression (x y z : ℝ) 
  (hx : 0 ≤ x ∧ x ≤ 1) (hy : 0 ≤ y ∧ y ≤ 1) (hz : 0 ≤ z ∧ z ≤ 1) :
  (1 / ((1 - x) * (1 - y) * (1 - z))) + (1 / ((1 + x) * (1 + y) * (1 + z))) ≥ 2 :=
by sorry

theorem min_value_equality :
  (1 / ((1 - 0) * (1 - 0) * (1 - 0))) + (1 / ((1 + 0) * (1 + 0) * (1 + 0))) = 2 :=
by sorry

end min_value_expression_min_value_equality_l2044_204472


namespace absolute_value_equation_one_root_l2044_204470

theorem absolute_value_equation_one_root :
  ∃! x : ℝ, (abs x - 4 / x = 3 * abs x / x) ∧ (x ≠ 0) :=
sorry

end absolute_value_equation_one_root_l2044_204470


namespace largest_x_and_fraction_l2044_204499

theorem largest_x_and_fraction (x : ℝ) (a b c d : ℤ) : 
  (7 * x / 5 - 2 = 4 / x) →
  (x = (a + b * Real.sqrt c) / d) →
  (∀ y : ℝ, (7 * y / 5 - 2 = 4 / y) → y ≤ x) →
  (x = (5 + 5 * Real.sqrt 66) / 7 ∧ a * c * d / b = 462) := by
  sorry

end largest_x_and_fraction_l2044_204499


namespace train_length_l2044_204410

/-- The length of a train given its speed and time to cross a pole -/
theorem train_length (speed_kmh : ℝ) (time_s : ℝ) : 
  speed_kmh = 72 → time_s = 8 → speed_kmh * (1000 / 3600) * time_s = 160 := by
  sorry

#check train_length

end train_length_l2044_204410


namespace max_ab_value_l2044_204457

theorem max_ab_value (a b : ℝ) (ha : a > 0) (hb : b > 0) (heq : a^2 + b^2 - 6*a = 0) :
  ∃ (max_ab : ℝ), max_ab = (27 * Real.sqrt 3) / 4 ∧ ∀ (x y : ℝ), x > 0 → y > 0 → x^2 + y^2 - 6*x = 0 → x*y ≤ max_ab :=
sorry

end max_ab_value_l2044_204457


namespace geometric_sequence_ninth_term_l2044_204428

/-- A geometric sequence with the given properties -/
def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, ∀ n : ℕ, a (n + 1) = a n * q

theorem geometric_sequence_ninth_term
  (a : ℕ → ℝ)
  (h_geometric : geometric_sequence a)
  (h_third : a 3 = 1)
  (h_product : a 5 * a 6 * a 7 = 8) :
  a 9 = 4 := by
sorry

end geometric_sequence_ninth_term_l2044_204428


namespace wire_cutting_l2044_204437

theorem wire_cutting (total_length : ℝ) (ratio : ℝ) (shorter_piece : ℝ) : 
  total_length = 49 →
  ratio = 2 / 5 →
  shorter_piece + ratio⁻¹ * shorter_piece = total_length →
  shorter_piece = 14 := by
sorry

end wire_cutting_l2044_204437


namespace mike_plants_cost_l2044_204475

def rose_bush_price : ℝ := 75
def tiger_tooth_aloe_price : ℝ := 100
def total_rose_bushes : ℕ := 6
def friend_rose_bushes : ℕ := 2
def mike_rose_bushes : ℕ := total_rose_bushes - friend_rose_bushes
def tiger_tooth_aloes : ℕ := 2
def rose_bush_tax_rate : ℝ := 0.05
def tiger_tooth_aloe_tax_rate : ℝ := 0.07

def mike_total_cost : ℝ :=
  (mike_rose_bushes : ℝ) * rose_bush_price * (1 + rose_bush_tax_rate) +
  (tiger_tooth_aloes : ℝ) * tiger_tooth_aloe_price * (1 + tiger_tooth_aloe_tax_rate)

theorem mike_plants_cost :
  mike_total_cost = 529 := by sorry

end mike_plants_cost_l2044_204475


namespace vendor_division_l2044_204409

theorem vendor_division (account_balance : Nat) (min_addition : Nat) (num_vendors : Nat) : 
  account_balance = 329864 →
  min_addition = 4 →
  num_vendors = 20 →
  (∀ k < num_vendors, account_balance % k ≠ 0 ∨ (account_balance + min_addition) % k ≠ 0) ∧
  account_balance % num_vendors ≠ 0 ∧
  (account_balance + min_addition) % num_vendors = 0 :=
by sorry

end vendor_division_l2044_204409


namespace profit_share_ratio_l2044_204420

theorem profit_share_ratio (total_profit : ℕ) (difference : ℕ) : 
  total_profit = 700 → difference = 140 → 
  ∃ (x y : ℕ), x + y = total_profit ∧ x - y = difference ∧ 
  (y : ℚ) / total_profit = 2 / 5 := by
  sorry

end profit_share_ratio_l2044_204420


namespace difference_of_squares_division_l2044_204429

theorem difference_of_squares_division : (121^2 - 112^2) / 9 = 233 := by
  sorry

end difference_of_squares_division_l2044_204429


namespace smallest_gcd_yz_l2044_204430

theorem smallest_gcd_yz (x y z : ℕ+) (h1 : Nat.gcd x.val y.val = 360) (h2 : Nat.gcd x.val z.val = 1176) :
  ∃ (k : ℕ+), (∀ (w : ℕ+), Nat.gcd y.val z.val ≥ k.val) ∧ Nat.gcd y.val z.val = k.val :=
by sorry

end smallest_gcd_yz_l2044_204430


namespace four_by_four_min_cuts_five_by_five_min_cuts_l2044_204452

/-- Represents a square grid of size n x n -/
structure Square (n : ℕ) where
  size : ℕ
  size_eq : size = n

/-- Minimum number of cuts required to divide a square into unit squares -/
def min_cuts (s : Square n) : ℕ :=
  sorry

/-- Pieces can be overlapped during cutting -/
axiom overlap_allowed : ∀ (n : ℕ) (s : Square n), True

theorem four_by_four_min_cuts :
  ∀ (s : Square 4), min_cuts s = 4 :=
sorry

theorem five_by_five_min_cuts :
  ∀ (s : Square 5), min_cuts s = 6 :=
sorry

end four_by_four_min_cuts_five_by_five_min_cuts_l2044_204452


namespace quadratic_roots_property_l2044_204498

theorem quadratic_roots_property (p q : ℝ) : 
  (3 * p^2 + 5 * p - 8 = 0) → 
  (3 * q^2 + 5 * q - 8 = 0) → 
  (p - 2) * (q - 2) = 14/3 := by
sorry

end quadratic_roots_property_l2044_204498


namespace fifth_term_value_l2044_204476

theorem fifth_term_value (a : ℕ → ℤ) (S : ℕ → ℤ) 
  (h : ∀ n : ℕ, S n = 2 * n^2 + 3 * n - 1) :
  a 5 = 21 :=
sorry

end fifth_term_value_l2044_204476


namespace sin_330_degrees_l2044_204463

theorem sin_330_degrees : Real.sin (330 * π / 180) = -1/2 := by
  sorry

end sin_330_degrees_l2044_204463


namespace arithmetic_sequence_sum_l2044_204469

/-- An arithmetic sequence -/
def ArithmeticSequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_sum (a : ℕ → ℝ) :
  ArithmeticSequence a → a 2 = 4 → a 6 = 16 → a 3 + a 5 = 20 := by
  sorry

end arithmetic_sequence_sum_l2044_204469
