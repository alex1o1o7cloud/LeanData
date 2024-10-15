import Mathlib

namespace NUMINAMATH_CALUDE_cathy_commission_l511_51141

theorem cathy_commission (x : ℝ) : 
  0.15 * (x - 15) = 0.25 * (x - 25) → 
  0.1 * (x - 10) = 3 := by
sorry

end NUMINAMATH_CALUDE_cathy_commission_l511_51141


namespace NUMINAMATH_CALUDE_square_area_given_equal_perimeter_triangle_l511_51189

theorem square_area_given_equal_perimeter_triangle (s : ℝ) (a : ℝ) : 
  s > 0 → -- side length of equilateral triangle is positive
  a > 0 → -- side length of square is positive
  3 * s = 4 * a → -- equal perimeters
  s^2 * Real.sqrt 3 / 4 = 9 → -- area of equilateral triangle is 9
  a^2 = 27 * Real.sqrt 3 / 4 := by
sorry

end NUMINAMATH_CALUDE_square_area_given_equal_perimeter_triangle_l511_51189


namespace NUMINAMATH_CALUDE_sphere_surface_area_l511_51134

theorem sphere_surface_area (r : ℝ) (d : ℝ) (h1 : r = 1) (h2 : d = Real.sqrt 3) :
  4 * Real.pi * (r^2 + d^2) = 16 * Real.pi := by
  sorry

end NUMINAMATH_CALUDE_sphere_surface_area_l511_51134


namespace NUMINAMATH_CALUDE_imaginary_part_of_z_l511_51165

theorem imaginary_part_of_z (z : ℂ) (h : 1 + z * Complex.I = z - 2 * Complex.I) :
  z.im = 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_imaginary_part_of_z_l511_51165


namespace NUMINAMATH_CALUDE_square_of_negative_product_l511_51147

theorem square_of_negative_product (a b : ℝ) : (-3 * a^2 * b)^2 = 9 * a^4 * b^2 := by
  sorry

end NUMINAMATH_CALUDE_square_of_negative_product_l511_51147


namespace NUMINAMATH_CALUDE_field_trip_bus_occupancy_l511_51185

theorem field_trip_bus_occupancy
  (num_vans : ℕ)
  (num_buses : ℕ)
  (people_per_van : ℕ)
  (total_people : ℕ)
  (h1 : num_vans = 6)
  (h2 : num_buses = 8)
  (h3 : people_per_van = 6)
  (h4 : total_people = 180)
  : (total_people - num_vans * people_per_van) / num_buses = 18 := by
  sorry

end NUMINAMATH_CALUDE_field_trip_bus_occupancy_l511_51185


namespace NUMINAMATH_CALUDE_ellipse_hyperbola_foci_coincide_l511_51109

/-- Given an ellipse and a hyperbola, if the endpoints of the major axis of the ellipse
    coincide with the foci of the hyperbola, then m = 2 -/
theorem ellipse_hyperbola_foci_coincide (m : ℝ) : 
  (∀ x y : ℝ, x^2/3 + y^2/4 = 1 → 
    (∃ a : ℝ, a > 0 ∧ (x = 0 ∧ y = a ∨ x = 0 ∧ y = -a) ∧
    ∀ x' y' : ℝ, y'^2/2 - x'^2/m = 1 → 
      (x' = 0 ∧ y' = a ∨ x' = 0 ∧ y' = -a))) → 
  m = 2 := by
sorry

end NUMINAMATH_CALUDE_ellipse_hyperbola_foci_coincide_l511_51109


namespace NUMINAMATH_CALUDE_final_sum_of_numbers_l511_51171

theorem final_sum_of_numbers (n : ℕ) (h1 : n = 2013) : 
  ∃ (a b c d : ℕ), 
    (a * b * c * d = 27) ∧ 
    (a + b + c + d ≡ (n * (n + 1) / 2) [MOD 9]) ∧
    (a + b + c + d = 30) := by
  sorry

end NUMINAMATH_CALUDE_final_sum_of_numbers_l511_51171


namespace NUMINAMATH_CALUDE_jake_has_more_apples_l511_51170

def steven_apples : ℕ := 14

theorem jake_has_more_apples (jake_apples : ℕ) (h : jake_apples > steven_apples) :
  jake_apples > steven_apples := by sorry

end NUMINAMATH_CALUDE_jake_has_more_apples_l511_51170


namespace NUMINAMATH_CALUDE_maryann_work_time_l511_51145

theorem maryann_work_time (total_time calling_time accounting_time report_time : ℕ) : 
  total_time = 1440 ∧
  accounting_time = 2 * calling_time ∧
  report_time = 3 * accounting_time ∧
  total_time = calling_time + accounting_time + report_time →
  report_time = 960 := by
  sorry

end NUMINAMATH_CALUDE_maryann_work_time_l511_51145


namespace NUMINAMATH_CALUDE_negative_fraction_comparison_l511_51181

theorem negative_fraction_comparison : -1/3 < -1/5 := by
  sorry

end NUMINAMATH_CALUDE_negative_fraction_comparison_l511_51181


namespace NUMINAMATH_CALUDE_jm_length_l511_51197

/-- Triangle DEF with medians and centroid -/
structure TriangleWithCentroid where
  -- Define the triangle
  DE : ℝ
  DF : ℝ
  EF : ℝ
  -- Define the centroid
  J : ℝ × ℝ
  -- Define M as the foot of the altitude from J to EF
  M : ℝ × ℝ

/-- The theorem stating the length of JM in the given triangle -/
theorem jm_length (t : TriangleWithCentroid) 
  (h1 : t.DE = 14) 
  (h2 : t.DF = 15) 
  (h3 : t.EF = 21) : 
  Real.sqrt ((t.J.1 - t.M.1)^2 + (t.J.2 - t.M.2)^2) = 8/3 := by
  sorry


end NUMINAMATH_CALUDE_jm_length_l511_51197


namespace NUMINAMATH_CALUDE_reducible_fraction_implies_divisibility_l511_51101

theorem reducible_fraction_implies_divisibility 
  (a b c d l k p q : ℤ) 
  (h1 : a * l + b = k * p) 
  (h2 : c * l + d = k * q) : 
  k ∣ (a * d - b * c) := by
  sorry

end NUMINAMATH_CALUDE_reducible_fraction_implies_divisibility_l511_51101


namespace NUMINAMATH_CALUDE_f_max_value_l511_51155

noncomputable def f (x : ℝ) : ℝ := Real.log 2 * Real.log 5 - Real.log (2 * x) * Real.log (5 * x)

theorem f_max_value :
  ∃ (max : ℝ), (∀ (x : ℝ), x > 0 → f x ≤ max) ∧ (∃ (x : ℝ), x > 0 ∧ f x = max) ∧ max = 1/4 := by
  sorry

end NUMINAMATH_CALUDE_f_max_value_l511_51155


namespace NUMINAMATH_CALUDE_line_circle_intersection_l511_51175

-- Define the line l
def line_l (k : ℝ) (x y : ℝ) : Prop :=
  (2*k+1)*x + (k-1)*y - (4*k-1) = 0

-- Define the circle C
def circle_C (x y : ℝ) : Prop :=
  x^2 + y^2 - 4*x - 2*y + 1 = 0

-- Define the point P
def point_P : ℝ × ℝ := (4, 4)

-- Define the minimum |AB| line
def min_AB_line (x y : ℝ) : Prop :=
  x - y + 1 = 0

-- Define the tangent lines
def tangent_line_1 (x : ℝ) : Prop := x = 4
def tangent_line_2 (x y : ℝ) : Prop := y = (5/12)*x + (28/12)

theorem line_circle_intersection :
  ∀ k : ℝ,
  (∀ x y : ℝ, line_l k x y ∧ circle_C x y → 
    (∀ x' y' : ℝ, min_AB_line x' y' → 
      (x - x')^2 + (y - y')^2 ≤ (x - x')^2 + (y - y')^2)) ∧
  (∃ x y : ℝ, min_AB_line x y ∧ circle_C x y ∧
    ∃ x' y' : ℝ, min_AB_line x' y' ∧ circle_C x' y' ∧
    (x - x')^2 + (y - y')^2 = 8) ∧
  (∀ x y : ℝ, (tangent_line_1 x ∨ tangent_line_2 x y) →
    (x - 4)^2 + (y - 4)^2 = ((x - 2)^2 + (y - 1)^2 - 4)^2 / ((x - 2)^2 + (y - 1)^2)) :=
by sorry

end NUMINAMATH_CALUDE_line_circle_intersection_l511_51175


namespace NUMINAMATH_CALUDE_sqrt_equation_solution_l511_51193

theorem sqrt_equation_solution (a : ℝ) :
  Real.sqrt 3 * (a * Real.sqrt 6) = 6 * Real.sqrt 2 → a = 2 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_equation_solution_l511_51193


namespace NUMINAMATH_CALUDE_hyperbola_tangent_orthogonal_l511_51106

/-- Hyperbola C: 2x^2 - y^2 = 1 -/
def Hyperbola (x y : ℝ) : Prop := 2 * x^2 - y^2 = 1

/-- Circle: x^2 + y^2 = 1 -/
def UnitCircle (x y : ℝ) : Prop := x^2 + y^2 = 1

/-- Line with slope k passing through (x, y) -/
def Line (k b x y : ℝ) : Prop := y = k * x + b

/-- Tangent condition for a line to the unit circle -/
def IsTangent (k b : ℝ) : Prop := b^2 = k^2 + 1

/-- Perpendicularity condition for two vectors -/
def IsOrthogonal (x1 y1 x2 y2 : ℝ) : Prop := x1 * x2 + y1 * y2 = 0

/-- Main theorem -/
theorem hyperbola_tangent_orthogonal (k b x1 y1 x2 y2 : ℝ) :
  |k| < Real.sqrt 2 →
  Hyperbola x1 y1 →
  Hyperbola x2 y2 →
  Line k b x1 y1 →
  Line k b x2 y2 →
  IsTangent k b →
  IsOrthogonal x1 y1 x2 y2 := by sorry

end NUMINAMATH_CALUDE_hyperbola_tangent_orthogonal_l511_51106


namespace NUMINAMATH_CALUDE_inverse_difference_equals_negative_reciprocal_l511_51157

theorem inverse_difference_equals_negative_reciprocal (a b : ℝ) 
  (ha : a ≠ 0) (hb : b ≠ 0) (hab : 3 * a - b / 3 ≠ 0) :
  (3 * a - b / 3)⁻¹ * ((3 * a)⁻¹ - (b / 3)⁻¹) = -(a * b)⁻¹ := by
  sorry

end NUMINAMATH_CALUDE_inverse_difference_equals_negative_reciprocal_l511_51157


namespace NUMINAMATH_CALUDE_intersection_S_complement_T_l511_51135

-- Define the universal set U as ℝ
def U := ℝ

-- Define set S
def S : Set ℝ := {x : ℝ | x^2 - x ≤ 0}

-- Define set T
def T : Set ℝ := {y : ℝ | ∃ x : ℝ, y = 2^x ∧ x ≤ 0}

-- State the theorem
theorem intersection_S_complement_T : S ∩ (Set.univ \ T) = {0} := by sorry

end NUMINAMATH_CALUDE_intersection_S_complement_T_l511_51135


namespace NUMINAMATH_CALUDE_alpha_value_l511_51177

theorem alpha_value (α γ : ℂ) 
  (h1 : (α + γ).re > 0)
  (h2 : (Complex.I * (α - 3 * γ)).re > 0)
  (h3 : γ = 4 + 3 * Complex.I) :
  α = 10.5 + 0.5 * Complex.I :=
by sorry

end NUMINAMATH_CALUDE_alpha_value_l511_51177


namespace NUMINAMATH_CALUDE_orange_picking_fraction_l511_51116

/-- Proves that the fraction of oranges picked from each tree is 2/5 --/
theorem orange_picking_fraction
  (num_trees : ℕ)
  (fruits_per_tree : ℕ)
  (remaining_fruits : ℕ)
  (h1 : num_trees = 8)
  (h2 : fruits_per_tree = 200)
  (h3 : remaining_fruits = 960)
  (h4 : remaining_fruits < num_trees * fruits_per_tree) :
  (num_trees * fruits_per_tree - remaining_fruits) / (num_trees * fruits_per_tree) = 2 / 5 :=
by sorry

end NUMINAMATH_CALUDE_orange_picking_fraction_l511_51116


namespace NUMINAMATH_CALUDE_expand_expression_l511_51187

theorem expand_expression (x : ℝ) : (17 * x + 12) * (3 * x) = 51 * x^2 + 36 * x := by
  sorry

end NUMINAMATH_CALUDE_expand_expression_l511_51187


namespace NUMINAMATH_CALUDE_rectangle_triangle_altitude_l511_51192

theorem rectangle_triangle_altitude (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a ≥ b) :
  let rectangle_area := a * b
  let triangle_leg := 2 * b
  let triangle_hypotenuse := Real.sqrt (a^2 + triangle_leg^2)
  let triangle_area := (1/2) * a * triangle_leg
  triangle_area = rectangle_area →
  (2 * rectangle_area) / triangle_hypotenuse = (2 * a * b) / Real.sqrt (a^2 + 4 * b^2) :=
by sorry

end NUMINAMATH_CALUDE_rectangle_triangle_altitude_l511_51192


namespace NUMINAMATH_CALUDE_functional_equation_solutions_l511_51180

/-- A real-valued function that satisfies the given functional equation. -/
def SatisfyingFunction (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (2002 * x - f 0) = 2002 * x^2

/-- The theorem stating the only two functions that satisfy the functional equation. -/
theorem functional_equation_solutions :
  ∀ f : ℝ → ℝ, SatisfyingFunction f ↔ 
    (∀ x : ℝ, f x = x^2 / 2002) ∨ 
    (∀ x : ℝ, f x = x^2 / 2002 + 2 * x + 2002) :=
sorry

end NUMINAMATH_CALUDE_functional_equation_solutions_l511_51180


namespace NUMINAMATH_CALUDE_parabola_circle_fixed_points_l511_51198

/-- Parabola C: x^2 = -4y -/
def parabola (x y : ℝ) : Prop := x^2 = -4*y

/-- Focus of the parabola -/
def focus : ℝ × ℝ := (0, -1)

/-- Line l with non-zero slope k passing through the focus -/
def line_l (k : ℝ) (x y : ℝ) : Prop := y = k*x - 1 ∧ k ≠ 0

/-- Intersection points M and N of line l with parabola C -/
def intersection_points (k : ℝ) : ℝ × ℝ × ℝ × ℝ := sorry

/-- Points A and B where y = -1 intersects OM and ON -/
def points_AB (k : ℝ) : ℝ × ℝ × ℝ × ℝ := sorry

/-- Circle with diameter AB -/
def circle_AB (k : ℝ) (x y : ℝ) : Prop :=
  ∃ (xA yA xB yB : ℝ), (points_AB k = (xA, yA, xB, yB)) ∧
  (x - (xA + xB)/2)^2 + (y - (yA + yB)/2)^2 = ((xA - xB)^2 + (yA - yB)^2) / 4

theorem parabola_circle_fixed_points (k : ℝ) :
  (∀ x y, circle_AB k x y → (x = 0 ∧ y = 1) ∨ (x = 0 ∧ y = -3)) ∧
  (circle_AB k 0 1 ∧ circle_AB k 0 (-3)) :=
sorry

end NUMINAMATH_CALUDE_parabola_circle_fixed_points_l511_51198


namespace NUMINAMATH_CALUDE_expression_equals_24_times_30_to_1001_l511_51186

theorem expression_equals_24_times_30_to_1001 :
  (5^1001 + 6^1002)^2 - (5^1001 - 6^1002)^2 = 24 * 30^1001 :=
by sorry

end NUMINAMATH_CALUDE_expression_equals_24_times_30_to_1001_l511_51186


namespace NUMINAMATH_CALUDE_f_properties_l511_51112

noncomputable def f (x : ℝ) : ℝ := Real.cos (2 * x) + 2 * Real.sqrt 3 * Real.sin x * Real.cos x

theorem f_properties :
  (∃ (x : ℝ), 0 ≤ x ∧ x ≤ Real.pi / 2 ∧ f x = 2 ∧ ∀ y, 0 ≤ y ∧ y ≤ Real.pi / 2 → f y ≤ f x) ∧
  (∀ θ : ℝ, 0 < θ ∧ θ < Real.pi / 6 ∧ f θ = 4 / 3 → Real.cos (2 * θ) = (Real.sqrt 15 + 2) / 6) :=
by sorry

end NUMINAMATH_CALUDE_f_properties_l511_51112


namespace NUMINAMATH_CALUDE_min_squared_distance_to_origin_l511_51154

/-- The minimum value of x^2 + y^2 for points on the line x + y - 4 = 0 is 8 -/
theorem min_squared_distance_to_origin (x y : ℝ) : 
  x + y - 4 = 0 → (∀ a b : ℝ, a + b - 4 = 0 → x^2 + y^2 ≤ a^2 + b^2) → x^2 + y^2 = 8 := by
  sorry

end NUMINAMATH_CALUDE_min_squared_distance_to_origin_l511_51154


namespace NUMINAMATH_CALUDE_circle_properties_l511_51150

-- Define the circle equation
def circle_equation (x y : ℝ) : Prop := (x - 1)^2 + (y + 2)^2 = 4

-- Define the center of the circle
def center : ℝ × ℝ := (1, -2)

-- Define the radius of the circle
def radius : ℝ := 2

-- Theorem statement
theorem circle_properties :
  ∀ x y : ℝ, circle_equation x y ↔ ((x - center.1)^2 + (y - center.2)^2 = radius^2) :=
sorry

end NUMINAMATH_CALUDE_circle_properties_l511_51150


namespace NUMINAMATH_CALUDE_graveyard_bones_problem_l511_51176

theorem graveyard_bones_problem :
  let total_skeletons : ℕ := 20
  let adult_women : ℕ := total_skeletons / 2
  let adult_men : ℕ := (total_skeletons - adult_women) / 2
  let children : ℕ := total_skeletons - adult_women - adult_men
  let total_bones : ℕ := 375
  let woman_bones : ℕ → ℕ := λ x => x
  let man_bones : ℕ → ℕ := λ x => x + 5
  let child_bones : ℕ → ℕ := λ x => x / 2

  ∃ (w : ℕ), 
    adult_women * (woman_bones w) + 
    adult_men * (man_bones w) + 
    children * (child_bones w) = total_bones ∧ 
    w = 20 :=
by
  sorry

end NUMINAMATH_CALUDE_graveyard_bones_problem_l511_51176


namespace NUMINAMATH_CALUDE_power_equation_solution_l511_51199

theorem power_equation_solution : ∃ y : ℕ, (2^10 + 2^10 + 2^10 + 2^10 : ℕ) = 4^y ∧ y = 6 := by
  sorry

end NUMINAMATH_CALUDE_power_equation_solution_l511_51199


namespace NUMINAMATH_CALUDE_solution_set_f_leq_x_plus_2_range_f_geq_expr_l511_51142

-- Define the function f(x)
def f (x : ℝ) : ℝ := |x - 1| + |x + 1|

-- Theorem for the solution set of f(x) ≤ x + 2
theorem solution_set_f_leq_x_plus_2 :
  {x : ℝ | f x ≤ x + 2} = {x : ℝ | 0 ≤ x ∧ x ≤ 2} := by sorry

-- Theorem for the range of x satisfying f(x) ≥ (|a+1| - |2a-1|)/|a| for all non-zero real a
theorem range_f_geq_expr :
  {x : ℝ | ∀ a : ℝ, a ≠ 0 → f x ≥ (|a + 1| - |2*a - 1|) / |a|} =
  {x : ℝ | x ≤ -3/2 ∨ x ≥ 3/2} := by sorry

end NUMINAMATH_CALUDE_solution_set_f_leq_x_plus_2_range_f_geq_expr_l511_51142


namespace NUMINAMATH_CALUDE_average_apples_sold_example_l511_51168

/-- Calculates the average number of kg of apples sold per hour given the sales in two hours -/
def average_apples_sold (first_hour_sales second_hour_sales : ℕ) : ℚ :=
  (first_hour_sales + second_hour_sales : ℚ) / 2

theorem average_apples_sold_example : average_apples_sold 10 2 = 6 := by
  sorry

end NUMINAMATH_CALUDE_average_apples_sold_example_l511_51168


namespace NUMINAMATH_CALUDE_athlete_stability_l511_51162

/-- Represents an athlete's shooting performance -/
structure Athlete where
  average_score : ℝ
  variance : ℝ
  shot_count : ℕ

/-- Defines when one athlete's performance is more stable than another's -/
def more_stable (a b : Athlete) : Prop :=
  a.variance < b.variance

theorem athlete_stability 
  (A B : Athlete)
  (h1 : A.average_score = B.average_score)
  (h2 : A.shot_count = 10)
  (h3 : B.shot_count = 10)
  (h4 : A.variance = 0.4)
  (h5 : B.variance = 2)
  : more_stable A B :=
sorry

end NUMINAMATH_CALUDE_athlete_stability_l511_51162


namespace NUMINAMATH_CALUDE_mike_weekly_spending_l511_51174

/-- Given that Mike made $14 mowing lawns and $26 weed eating, and the money would last him 8 weeks,
    prove that he spent $5 per week. -/
theorem mike_weekly_spending (lawn_money : ℕ) (weed_money : ℕ) (weeks : ℕ) 
  (h1 : lawn_money = 14)
  (h2 : weed_money = 26)
  (h3 : weeks = 8) :
  (lawn_money + weed_money) / weeks = 5 := by
  sorry

end NUMINAMATH_CALUDE_mike_weekly_spending_l511_51174


namespace NUMINAMATH_CALUDE_consecutive_integers_sum_of_powers_l511_51102

theorem consecutive_integers_sum_of_powers (n : ℕ) : 
  (n > 0) →
  ((n - 1)^2 + n^2 + (n + 1)^2 = 9458) →
  ((n - 1)^4 + n^4 + (n + 1)^4 = 30212622) :=
by sorry

end NUMINAMATH_CALUDE_consecutive_integers_sum_of_powers_l511_51102


namespace NUMINAMATH_CALUDE_adjacent_knights_probability_l511_51103

/-- The number of knights at the round table -/
def total_knights : ℕ := 30

/-- The number of knights selected for the quest -/
def selected_knights : ℕ := 4

/-- The probability that at least two of the selected knights were sitting next to each other -/
def adjacent_probability : ℚ := 943 / 1023

/-- Theorem stating the probability of at least two out of four randomly selected knights 
    sitting next to each other in a round table of 30 knights -/
theorem adjacent_knights_probability : 
  let total_ways := Nat.choose total_knights selected_knights
  let non_adjacent_ways := (total_knights - selected_knights) * 
                           (total_knights - selected_knights - 3) * 
                           (total_knights - selected_knights - 6) * 
                           (total_knights - selected_knights - 9)
  (1 : ℚ) - (non_adjacent_ways : ℚ) / total_ways = adjacent_probability := by
  sorry

#eval adjacent_probability.num + adjacent_probability.den

end NUMINAMATH_CALUDE_adjacent_knights_probability_l511_51103


namespace NUMINAMATH_CALUDE_students_remaining_in_school_l511_51146

theorem students_remaining_in_school (total_students : ℕ) 
  (h1 : total_students = 1000)
  (h2 : ∃ trip_students : ℕ, trip_students = total_students / 2)
  (h3 : ∃ remaining_after_trip : ℕ, remaining_after_trip = total_students - (total_students / 2))
  (h4 : ∃ sent_home : ℕ, sent_home = remaining_after_trip / 2)
  : total_students - (total_students / 2) - ((total_students - (total_students / 2)) / 2) = 250 := by
  sorry

end NUMINAMATH_CALUDE_students_remaining_in_school_l511_51146


namespace NUMINAMATH_CALUDE_same_side_probability_is_seven_twentyfourths_l511_51139

/-- Represents a 12-sided die with specific colored sides. -/
structure TwelveSidedDie :=
  (maroon : Nat)
  (teal : Nat)
  (cyan : Nat)
  (sparkly : Nat)
  (total_sides : Nat)
  (side_sum : maroon + teal + cyan + sparkly = total_sides)

/-- The probability of two dice showing the same side when rolled. -/
def same_side_probability (d : TwelveSidedDie) : Rat :=
  (d.maroon^2 + d.teal^2 + d.cyan^2 + d.sparkly^2) / d.total_sides^2

/-- The specific die used in the problem. -/
def problem_die : TwelveSidedDie :=
  { maroon := 3
    teal := 4
    cyan := 4
    sparkly := 1
    total_sides := 12
    side_sum := by decide }

/-- Theorem stating that the probability of two problem dice showing the same side is 7/24. -/
theorem same_side_probability_is_seven_twentyfourths :
  same_side_probability problem_die = 7 / 24 := by
  sorry

end NUMINAMATH_CALUDE_same_side_probability_is_seven_twentyfourths_l511_51139


namespace NUMINAMATH_CALUDE_simple_interest_problem_l511_51111

/-- Simple interest calculation problem -/
theorem simple_interest_problem (rate : ℚ) (principal : ℚ) (interest_diff : ℚ) (years : ℚ) :
  rate = 4 / 100 →
  principal = 2400 →
  principal * rate * years = principal - interest_diff →
  interest_diff = 1920 →
  years = 5 := by
  sorry

end NUMINAMATH_CALUDE_simple_interest_problem_l511_51111


namespace NUMINAMATH_CALUDE_john_max_books_l511_51152

/-- The maximum number of books John can buy given his money and the price per book -/
def max_books_buyable (total_money : ℕ) (price_per_book : ℕ) : ℕ :=
  total_money / price_per_book

/-- Proof that John can buy at most 14 books -/
theorem john_max_books :
  let john_money : ℕ := 4575  -- 45 dollars and 75 cents in cents
  let book_price : ℕ := 325   -- 3 dollars and 25 cents in cents
  max_books_buyable john_money book_price = 14 := by
sorry

end NUMINAMATH_CALUDE_john_max_books_l511_51152


namespace NUMINAMATH_CALUDE_collinear_points_sum_l511_51143

/-- Three points in 3D space are collinear if they lie on the same line. -/
def collinear (p1 p2 p3 : ℝ × ℝ × ℝ) : Prop := sorry

/-- The main theorem: If the given points are collinear, then c + d = 6. -/
theorem collinear_points_sum (c d : ℝ) : 
  collinear (2, c, d) (c, 3, d) (c, d, 4) → c + d = 6 := by
  sorry

end NUMINAMATH_CALUDE_collinear_points_sum_l511_51143


namespace NUMINAMATH_CALUDE_hyperbola_condition_l511_51110

/-- A curve of the form ax^2 + by^2 = 1 is a hyperbola if ab < 0 -/
def is_hyperbola (a b : ℝ) : Prop := a * b < 0

/-- The curve mx^2 - (m-2)y^2 = 1 -/
def curve (m : ℝ) : (ℝ → ℝ → Prop) := λ x y => m * x^2 - (m - 2) * y^2 = 1

theorem hyperbola_condition (m : ℝ) :
  (∀ m > 3, is_hyperbola m (2 - m)) ∧
  (∃ m ≤ 3, is_hyperbola m (2 - m)) :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_condition_l511_51110


namespace NUMINAMATH_CALUDE_probability_three_primes_l511_51173

-- Define a 12-sided die
def Die := Finset (Fin 12)

-- Define the set of prime numbers on a 12-sided die
def PrimeNumbers : Finset (Fin 12) := {2, 3, 5, 7, 11}

-- Define the probability of rolling a prime number on a single die
def ProbPrime : ℚ := (PrimeNumbers.card : ℚ) / 12

-- Define the probability of not rolling a prime number on a single die
def ProbNotPrime : ℚ := 1 - ProbPrime

-- Define the number of dice
def NumDice : ℕ := 4

-- Define the number of dice that should show a prime
def NumPrimeDice : ℕ := 3

-- Theorem statement
theorem probability_three_primes :
  (NumDice.choose NumPrimeDice : ℚ) * ProbPrime ^ NumPrimeDice * ProbNotPrime ^ (NumDice - NumPrimeDice) = 875 / 5184 :=
sorry

end NUMINAMATH_CALUDE_probability_three_primes_l511_51173


namespace NUMINAMATH_CALUDE_intersection_nonempty_range_union_equals_B_l511_51163

-- Define sets A and B
def A : Set ℝ := {x | x^2 + 4*x = 0}
def B (m : ℝ) : Set ℝ := {x | x^2 + 2*(m+1)*x + m^2 - 1 = 2}

-- Theorem for part (1)
theorem intersection_nonempty_range (m : ℝ) : 
  (A ∩ B m).Nonempty → m = -Real.sqrt 3 :=
sorry

-- Theorem for part (2)
theorem union_equals_B (m : ℝ) : 
  A ∪ B m = B m → m = -Real.sqrt 3 :=
sorry

end NUMINAMATH_CALUDE_intersection_nonempty_range_union_equals_B_l511_51163


namespace NUMINAMATH_CALUDE_cubic_decomposition_sum_l511_51179

theorem cubic_decomposition_sum :
  ∃ (a b c d e : ℝ),
    (∀ x : ℝ, 512 * x^3 + 27 = (a * x + b) * (c * x^2 + d * x + e)) ∧
    (a + b + c + d + e = 60) := by
  sorry

end NUMINAMATH_CALUDE_cubic_decomposition_sum_l511_51179


namespace NUMINAMATH_CALUDE_sum_of_real_and_imag_parts_l511_51118

theorem sum_of_real_and_imag_parts (z : ℂ) (h : z * (2 + Complex.I) = 2 * Complex.I - 1) :
  z.re + z.im = 1/5 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_real_and_imag_parts_l511_51118


namespace NUMINAMATH_CALUDE_polygon_is_trapezoid_l511_51125

/-- A line in 2D space represented by its slope and y-intercept -/
structure Line where
  slope : ℝ
  intercept : ℝ

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Definition of a trapezoid: a quadrilateral with at least one pair of parallel sides -/
def is_trapezoid (p1 p2 p3 p4 : Point) : Prop :=
  ∃ (l1 l2 l3 l4 : Line),
    (p1.y = l1.slope * p1.x + l1.intercept) ∧
    (p2.y = l2.slope * p2.x + l2.intercept) ∧
    (p3.y = l3.slope * p3.x + l3.intercept) ∧
    (p4.y = l4.slope * p4.x + l4.intercept) ∧
    ((l1.slope = l2.slope ∧ l1.slope ≠ l3.slope ∧ l1.slope ≠ l4.slope) ∨
     (l1.slope = l3.slope ∧ l1.slope ≠ l2.slope ∧ l1.slope ≠ l4.slope) ∨
     (l1.slope = l4.slope ∧ l1.slope ≠ l2.slope ∧ l1.slope ≠ l3.slope) ∨
     (l2.slope = l3.slope ∧ l2.slope ≠ l1.slope ∧ l2.slope ≠ l4.slope) ∨
     (l2.slope = l4.slope ∧ l2.slope ≠ l1.slope ∧ l2.slope ≠ l3.slope) ∨
     (l3.slope = l4.slope ∧ l3.slope ≠ l1.slope ∧ l3.slope ≠ l2.slope))

theorem polygon_is_trapezoid :
  let l1 : Line := ⟨2, 3⟩
  let l2 : Line := ⟨-2, 3⟩
  let l3 : Line := ⟨2, -1⟩
  let l4 : Line := ⟨0, -1⟩
  ∃ (p1 p2 p3 p4 : Point),
    (p1.y = l1.slope * p1.x + l1.intercept ∨ p1.y = l2.slope * p1.x + l2.intercept ∨
     p1.y = l3.slope * p1.x + l3.intercept ∨ p1.y = l4.slope * p1.x + l4.intercept) ∧
    (p2.y = l1.slope * p2.x + l1.intercept ∨ p2.y = l2.slope * p2.x + l2.intercept ∨
     p2.y = l3.slope * p2.x + l3.intercept ∨ p2.y = l4.slope * p2.x + l4.intercept) ∧
    (p3.y = l1.slope * p3.x + l1.intercept ∨ p3.y = l2.slope * p3.x + l2.intercept ∨
     p3.y = l3.slope * p3.x + l3.intercept ∨ p3.y = l4.slope * p3.x + l4.intercept) ∧
    (p4.y = l1.slope * p4.x + l1.intercept ∨ p4.y = l2.slope * p4.x + l2.intercept ∨
     p4.y = l3.slope * p4.x + l3.intercept ∨ p4.y = l4.slope * p4.x + l4.intercept) ∧
    is_trapezoid p1 p2 p3 p4 :=
by sorry

end NUMINAMATH_CALUDE_polygon_is_trapezoid_l511_51125


namespace NUMINAMATH_CALUDE_angle_half_in_fourth_quadrant_l511_51158

/-- Represents the four quadrants of the coordinate plane. -/
inductive Quadrant
  | first
  | second
  | third
  | fourth

/-- Determines if an angle is in a specific quadrant. -/
def in_quadrant (angle : ℝ) (q : Quadrant) : Prop :=
  match q with
  | Quadrant.first => 0 < angle ∧ angle < Real.pi / 2
  | Quadrant.second => Real.pi / 2 < angle ∧ angle < Real.pi
  | Quadrant.third => Real.pi < angle ∧ angle < 3 * Real.pi / 2
  | Quadrant.fourth => 3 * Real.pi / 2 < angle ∧ angle < 2 * Real.pi

theorem angle_half_in_fourth_quadrant (α : ℝ) 
  (h1 : in_quadrant α Quadrant.third) 
  (h2 : |Real.sin (α/2)| = -Real.sin (α/2)) : 
  in_quadrant (α/2) Quadrant.fourth :=
sorry

end NUMINAMATH_CALUDE_angle_half_in_fourth_quadrant_l511_51158


namespace NUMINAMATH_CALUDE_line_tangent_to_circle_l511_51136

theorem line_tangent_to_circle 
  (x₀ y₀ r : ℝ) 
  (h_outside : x₀^2 + y₀^2 > r^2) :
  ∃ (x y : ℝ), 
    x₀*x + y₀*y = r^2 ∧ 
    x^2 + y^2 = r^2 ∧
    ∀ (x' y' : ℝ), x₀*x' + y₀*y' = r^2 ∧ x'^2 + y'^2 = r^2 → (x', y') = (x, y) :=
by sorry

end NUMINAMATH_CALUDE_line_tangent_to_circle_l511_51136


namespace NUMINAMATH_CALUDE_xyz_value_l511_51123

theorem xyz_value (x y z : ℝ) 
  (h1 : (x + y + z) * (x * y + x * z + y * z) = 24)
  (h2 : x^2 * (y + z) + y^2 * (x + z) + z^2 * (x + y) = 8) :
  x * y * z = 16 / 3 := by
sorry

end NUMINAMATH_CALUDE_xyz_value_l511_51123


namespace NUMINAMATH_CALUDE_quadratic_real_roots_l511_51113

theorem quadratic_real_roots (k : ℝ) :
  (∃ x : ℝ, k * x^2 - 2 * x - 1 = 0) ↔ k ≥ -1 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_real_roots_l511_51113


namespace NUMINAMATH_CALUDE_min_distance_theorem_l511_51126

/-- Line l in polar form -/
def line_l (ρ θ : ℝ) : Prop := ρ * Real.sin (θ + Real.pi / 4) = 2 * Real.sqrt 2

/-- Circle C in Cartesian form -/
def circle_C (x y : ℝ) : Prop := x^2 + y^2 = 1

/-- Scaling transformation -/
def scaling (x y x' y' : ℝ) : Prop := x' = 2 * x ∧ y' = 3 * y

/-- Curve C' after scaling transformation -/
def curve_C' (x' y' : ℝ) : Prop := x'^2 / 4 + y'^2 / 9 = 1

theorem min_distance_theorem :
  (∀ ρ θ, line_l ρ θ) →
  (∀ x y, circle_C x y) →
  (∃ d : ℝ, d = 2 * Real.sqrt 2 - 1 ∧
    (∀ x y, circle_C x y → (x + y - 4)^2 / 2 ≥ d^2)) ∧
  (∃ d' : ℝ, d' = 2 * Real.sqrt 2 - 2 ∧
    (∀ x' y', curve_C' x' y' → (x' + y' - 4)^2 / 2 ≥ d'^2)) := by
  sorry

end NUMINAMATH_CALUDE_min_distance_theorem_l511_51126


namespace NUMINAMATH_CALUDE_milk_water_ratio_l511_51191

theorem milk_water_ratio (initial_volume : ℚ) (initial_milk_ratio : ℚ) (initial_water_ratio : ℚ) (added_water : ℚ) :
  initial_volume = 45 ∧
  initial_milk_ratio = 4 ∧
  initial_water_ratio = 1 ∧
  added_water = 3 →
  let total_parts := initial_milk_ratio + initial_water_ratio
  let initial_milk_volume := (initial_milk_ratio / total_parts) * initial_volume
  let initial_water_volume := (initial_water_ratio / total_parts) * initial_volume
  let new_water_volume := initial_water_volume + added_water
  let new_milk_ratio := initial_milk_volume
  let new_water_ratio := new_water_volume
  (new_milk_ratio : ℚ) / new_water_ratio = 3 / 1 :=
by sorry

end NUMINAMATH_CALUDE_milk_water_ratio_l511_51191


namespace NUMINAMATH_CALUDE_find_b_value_l511_51140

theorem find_b_value (a b c : ℤ) : 
  a + b + c = 111 → 
  (a + 10 = b - 10) ∧ (b - 10 = 3 * c) → 
  b = 58 :=
by sorry

end NUMINAMATH_CALUDE_find_b_value_l511_51140


namespace NUMINAMATH_CALUDE_sum_of_integers_l511_51169

theorem sum_of_integers (x y : ℕ+) 
  (sum_of_squares : x^2 + y^2 = 245)
  (product : x * y = 120) : 
  (x : ℝ) + y = Real.sqrt 485 := by
sorry

end NUMINAMATH_CALUDE_sum_of_integers_l511_51169


namespace NUMINAMATH_CALUDE_intersection_with_complement_l511_51132

def U : Set ℤ := Set.univ

def A : Set ℤ := {-1, 1, 2}

def B : Set ℤ := {-1, 1}

theorem intersection_with_complement :
  A ∩ (Set.compl B) = {2} := by sorry

end NUMINAMATH_CALUDE_intersection_with_complement_l511_51132


namespace NUMINAMATH_CALUDE_smallest_n_divisible_l511_51144

theorem smallest_n_divisible (n : ℕ) : 
  (∀ m : ℕ, m > 0 ∧ m < 12 → (¬(72 ∣ m^2) ∨ ¬(1728 ∣ m^3))) ∧ 
  (72 ∣ 12^2) ∧ (1728 ∣ 12^3) := by
  sorry

end NUMINAMATH_CALUDE_smallest_n_divisible_l511_51144


namespace NUMINAMATH_CALUDE_profit_distribution_l511_51195

/-- Profit distribution in a business partnership --/
theorem profit_distribution (a b c : ℕ) (profit_b : ℕ) : 
  a = 8000 → b = 10000 → c = 12000 → profit_b = 4000 →
  ∃ (profit_a profit_c : ℕ),
    profit_a * b = profit_b * a ∧
    profit_c * b = profit_b * c ∧
    profit_c - profit_a = 1600 :=
by sorry

end NUMINAMATH_CALUDE_profit_distribution_l511_51195


namespace NUMINAMATH_CALUDE_angle_complement_quadrant_l511_51108

/-- An angle is in the first quadrant if it's between 0 and π/2 radians -/
def is_first_quadrant (α : ℝ) : Prop := 0 < α ∧ α < Real.pi / 2

/-- An angle is in the second quadrant if it's between π/2 and π radians -/
def is_second_quadrant (α : ℝ) : Prop := Real.pi / 2 < α ∧ α < Real.pi

theorem angle_complement_quadrant (α : ℝ) 
  (h : is_first_quadrant α) : is_second_quadrant (Real.pi - α) := by
  sorry

end NUMINAMATH_CALUDE_angle_complement_quadrant_l511_51108


namespace NUMINAMATH_CALUDE_area_ratio_theorem_l511_51128

-- Define the triangles
structure Triangle :=
  (A B C : ℝ × ℝ)

-- Define the properties of the triangles
def is_30_60_90 (t : Triangle) : Prop := sorry

def is_right_angled_isosceles (t : Triangle) : Prop := sorry

-- Define the area of a triangle
def area (t : Triangle) : ℝ := sorry

-- Define the theorem
theorem area_ratio_theorem (FGH IFG EGH IEH : Triangle) :
  is_30_60_90 FGH →
  is_right_angled_isosceles EGH →
  (area IFG) / (area IEH) = 1 / 2 :=
sorry

end NUMINAMATH_CALUDE_area_ratio_theorem_l511_51128


namespace NUMINAMATH_CALUDE_bob_sandwich_options_l511_51120

/-- Represents the number of different types of bread available. -/
def num_breads : ℕ := 5

/-- Represents the number of different types of meat available. -/
def num_meats : ℕ := 7

/-- Represents the number of different types of cheese available. -/
def num_cheeses : ℕ := 6

/-- Represents the number of sandwiches with turkey and mozzarella cheese. -/
def turkey_mozzarella_combos : ℕ := num_breads

/-- Represents the number of sandwiches with rye bread and beef. -/
def rye_beef_combos : ℕ := num_cheeses

/-- Calculates the total number of possible sandwich combinations. -/
def total_combos : ℕ := num_breads * num_meats * num_cheeses

/-- Theorem stating the number of different sandwiches Bob could order. -/
theorem bob_sandwich_options : 
  total_combos - turkey_mozzarella_combos - rye_beef_combos = 199 := by
  sorry

end NUMINAMATH_CALUDE_bob_sandwich_options_l511_51120


namespace NUMINAMATH_CALUDE_equation_solution_l511_51137

theorem equation_solution : ∃ x : ℝ, x ≠ 2 ∧ x + 2 = 2 / (x - 2) ↔ x = Real.sqrt 6 ∨ x = -Real.sqrt 6 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l511_51137


namespace NUMINAMATH_CALUDE_cube_root_simplification_l511_51159

theorem cube_root_simplification :
  (2^9 * 5^3 * 7^3 : ℝ)^(1/3) = 280 := by sorry

end NUMINAMATH_CALUDE_cube_root_simplification_l511_51159


namespace NUMINAMATH_CALUDE_min_value_x_plus_y_l511_51117

theorem min_value_x_plus_y (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : 1/x + 9/y = 1) :
  x + y ≥ 16 ∧ ∃ x y, x > 0 ∧ y > 0 ∧ 1/x + 9/y = 1 ∧ x + y = 16 := by
  sorry

end NUMINAMATH_CALUDE_min_value_x_plus_y_l511_51117


namespace NUMINAMATH_CALUDE_boat_race_distance_l511_51100

/-- The distance between two points A and B traveled by two boats with different speeds and start times -/
theorem boat_race_distance 
  (a b d n : ℝ) 
  (ha : a > 0) 
  (hb : b > 0) 
  (hd : d ≥ 0) 
  (hn : n > 0) 
  (hab : a > b) :
  ∃ x : ℝ, x > 0 ∧ x = (a * (d + b * n)) / (a - b) ∧
    x / a + n = (x - d) / b :=
by sorry

end NUMINAMATH_CALUDE_boat_race_distance_l511_51100


namespace NUMINAMATH_CALUDE_scooter_price_l511_51196

/-- Given an upfront payment of 20% of the total cost, which amounts to $240, prove that the total price of the scooter is $1200. -/
theorem scooter_price (upfront_percentage : ℝ) (upfront_amount : ℝ) (total_price : ℝ) : 
  upfront_percentage = 0.20 → 
  upfront_amount = 240 → 
  upfront_percentage * total_price = upfront_amount → 
  total_price = 1200 := by
sorry

end NUMINAMATH_CALUDE_scooter_price_l511_51196


namespace NUMINAMATH_CALUDE_water_left_in_ml_l511_51153

-- Define the total amount of water in liters
def total_water : ℝ := 135.1

-- Define the size of each bucket in liters
def bucket_size : ℝ := 7

-- Define the conversion factor from liters to milliliters
def liters_to_ml : ℝ := 1000

-- Theorem statement
theorem water_left_in_ml :
  (total_water - bucket_size * ⌊total_water / bucket_size⌋) * liters_to_ml = 2100 := by
  sorry


end NUMINAMATH_CALUDE_water_left_in_ml_l511_51153


namespace NUMINAMATH_CALUDE_negation_relationship_l511_51119

theorem negation_relationship (x : ℝ) :
  (¬(|x + 1| > 2) → ¬(5*x - 6 > x^2)) ∧
  ¬(¬(5*x - 6 > x^2) → ¬(|x + 1| > 2)) :=
by sorry

end NUMINAMATH_CALUDE_negation_relationship_l511_51119


namespace NUMINAMATH_CALUDE_rectangular_prism_width_l511_51156

/-- Given a rectangular prism with length 4 units, height 10 units, and diagonal 14 units,
    prove that its width is 4√5 units. -/
theorem rectangular_prism_width (l h d w : ℝ) : 
  l = 4 → h = 10 → d = 14 → d^2 = l^2 + w^2 + h^2 → w = 4 * Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_rectangular_prism_width_l511_51156


namespace NUMINAMATH_CALUDE_largest_m_binomial_sum_l511_51164

theorem largest_m_binomial_sum (m : ℕ) : (Nat.choose 10 4 + Nat.choose 10 5 = Nat.choose 11 m) → m ≤ 6 :=
sorry

end NUMINAMATH_CALUDE_largest_m_binomial_sum_l511_51164


namespace NUMINAMATH_CALUDE_bowling_ball_weight_l511_51130

theorem bowling_ball_weight :
  ∀ (b c : ℝ),
  (5 * b = 3 * c) →
  (2 * c = 56) →
  (b = 16.8) :=
by
  sorry

end NUMINAMATH_CALUDE_bowling_ball_weight_l511_51130


namespace NUMINAMATH_CALUDE_equal_sums_exist_l511_51172

/-- Represents the direction a recruit is facing -/
inductive Direction
  | Left : Direction
  | Right : Direction
  | Around : Direction

/-- A line of recruits is represented as a list of their facing directions -/
def RecruitLine := List Direction

/-- Converts a Direction to an integer value -/
def directionToInt (d : Direction) : Int :=
  match d with
  | Direction.Left => -1
  | Direction.Right => 1
  | Direction.Around => 0

/-- Calculates the sum of directions to the left of a given index -/
def leftSum (line : RecruitLine) (index : Nat) : Int :=
  (line.take index).map directionToInt |>.sum

/-- Calculates the sum of directions to the right of a given index -/
def rightSum (line : RecruitLine) (index : Nat) : Int :=
  (line.drop (index + 1)).map directionToInt |>.sum

/-- Theorem: There always exists a position where the left sum equals the right sum -/
theorem equal_sums_exist (line : RecruitLine) :
  ∃ (index : Nat), leftSum line index = rightSum line index :=
  sorry

end NUMINAMATH_CALUDE_equal_sums_exist_l511_51172


namespace NUMINAMATH_CALUDE_maximal_closely_related_interval_l511_51122

/-- Two functions are closely related on an interval if their difference is bounded by 1 -/
def closely_related (f g : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x ∈ Set.Icc a b, |f x - g x| ≤ 1

/-- The given functions f and g -/
def f (x : ℝ) : ℝ := x^2 - 3*x + 4
def g (x : ℝ) : ℝ := 2*x - 3

/-- The theorem stating that [2, 3] is the maximal closely related interval for f and g -/
theorem maximal_closely_related_interval :
  closely_related f g 2 3 ∧
  ∀ a b : ℝ, a < 2 ∨ b > 3 → ¬(closely_related f g a b) :=
sorry

end NUMINAMATH_CALUDE_maximal_closely_related_interval_l511_51122


namespace NUMINAMATH_CALUDE_jellybean_average_increase_l511_51133

theorem jellybean_average_increase (initial_bags : ℕ) (initial_average : ℚ) (additional_jellybeans : ℕ) : 
  initial_bags = 34 →
  initial_average = 117 →
  additional_jellybeans = 362 →
  (((initial_bags : ℚ) * initial_average + additional_jellybeans) / (initial_bags + 1 : ℚ)) - initial_average = 7 :=
by sorry

end NUMINAMATH_CALUDE_jellybean_average_increase_l511_51133


namespace NUMINAMATH_CALUDE_rabbit_distribution_count_l511_51178

/-- Represents the number of stores -/
def num_stores : ℕ := 5

/-- Represents the number of parent rabbits -/
def num_parents : ℕ := 2

/-- Represents the number of child rabbits -/
def num_children : ℕ := 4

/-- Represents the total number of rabbits -/
def total_rabbits : ℕ := num_parents + num_children

/-- 
Represents the number of ways to distribute rabbits to stores 
such that no store has both a parent and a child 
-/
def distribution_ways : ℕ := sorry

theorem rabbit_distribution_count : distribution_ways = 380 := by sorry

end NUMINAMATH_CALUDE_rabbit_distribution_count_l511_51178


namespace NUMINAMATH_CALUDE_jake_debt_work_hours_l511_51182

def total_hours_worked (initial_debt_A initial_debt_B initial_debt_C : ℕ)
                       (payment_A payment_B payment_C : ℕ)
                       (rate_A rate_B rate_C : ℕ) : ℕ :=
  let remaining_debt_A := initial_debt_A - payment_A
  let remaining_debt_B := initial_debt_B - payment_B
  let remaining_debt_C := initial_debt_C - payment_C
  let hours_A := remaining_debt_A / rate_A
  let hours_B := remaining_debt_B / rate_B
  let hours_C := remaining_debt_C / rate_C
  hours_A + hours_B + hours_C

theorem jake_debt_work_hours :
  total_hours_worked 150 200 250 60 80 100 15 20 25 = 18 := by
  sorry

end NUMINAMATH_CALUDE_jake_debt_work_hours_l511_51182


namespace NUMINAMATH_CALUDE_sam_paul_study_difference_l511_51121

def average_difference (differences : List Int) : Int :=
  (differences.sum / differences.length)

theorem sam_paul_study_difference : 
  let differences : List Int := [20, 5, -5, 0, 15, -10, 10]
  average_difference differences = 5 := by
  sorry

end NUMINAMATH_CALUDE_sam_paul_study_difference_l511_51121


namespace NUMINAMATH_CALUDE_cost_price_per_metre_values_l511_51190

-- Define the cloth types
inductive ClothType
  | A
  | B
  | C

-- Define the properties for each cloth type
def metres_sold (t : ClothType) : ℕ :=
  match t with
  | ClothType.A => 200
  | ClothType.B => 150
  | ClothType.C => 100

def selling_price (t : ClothType) : ℕ :=
  match t with
  | ClothType.A => 10000
  | ClothType.B => 6000
  | ClothType.C => 4000

def loss (t : ClothType) : ℕ :=
  match t with
  | ClothType.A => 1000
  | ClothType.B => 450
  | ClothType.C => 200

-- Define the cost price per metre function
def cost_price_per_metre (t : ClothType) : ℚ :=
  (selling_price t + loss t : ℚ) / metres_sold t

-- State the theorem
theorem cost_price_per_metre_values :
  cost_price_per_metre ClothType.A = 55 ∧
  cost_price_per_metre ClothType.B = 43 ∧
  cost_price_per_metre ClothType.C = 42 := by
  sorry


end NUMINAMATH_CALUDE_cost_price_per_metre_values_l511_51190


namespace NUMINAMATH_CALUDE_complex_simplification_l511_51115

-- Define the complex number i
def i : ℂ := Complex.I

-- State the theorem
theorem complex_simplification :
  7 * (4 - 2*i) + 4*i * (3 - 2*i) = 36 - 2*i :=
by sorry

end NUMINAMATH_CALUDE_complex_simplification_l511_51115


namespace NUMINAMATH_CALUDE_average_weight_proof_l511_51127

theorem average_weight_proof (a b c : ℝ) : 
  (a + b + c) / 3 = 45 →
  (b + c) / 2 = 43 →
  b = 35 →
  (a + b) / 2 = 42 := by
sorry

end NUMINAMATH_CALUDE_average_weight_proof_l511_51127


namespace NUMINAMATH_CALUDE_line_inclination_theorem_l511_51114

theorem line_inclination_theorem (a b c : ℝ) (α : ℝ) : 
  (∃ (x y : ℝ), a * x + b * y + c = 0) →  -- Line exists
  (Real.tan α = -a / b) →  -- Angle of inclination
  (Real.sin α + Real.cos α = 0) →  -- Given condition
  (a - b = 0) :=  -- Conclusion to prove
by sorry

end NUMINAMATH_CALUDE_line_inclination_theorem_l511_51114


namespace NUMINAMATH_CALUDE_fly_speed_fly_speed_problem_l511_51131

/-- The speed of a fly moving between two cyclists --/
theorem fly_speed (cyclist_speed : ℝ) (initial_distance : ℝ) (fly_distance : ℝ) : ℝ :=
  let relative_speed := 2 * cyclist_speed
  let meeting_time := initial_distance / relative_speed
  fly_distance / meeting_time

/-- Given the conditions of the problem, prove that the fly's speed is 15 miles/hour --/
theorem fly_speed_problem : fly_speed 10 50 37.5 = 15 := by
  sorry

end NUMINAMATH_CALUDE_fly_speed_fly_speed_problem_l511_51131


namespace NUMINAMATH_CALUDE_orange_bags_count_l511_51138

theorem orange_bags_count (weight_per_bag : ℕ) (total_weight : ℕ) (h1 : weight_per_bag = 23) (h2 : total_weight = 1035) :
  total_weight / weight_per_bag = 45 := by
  sorry

end NUMINAMATH_CALUDE_orange_bags_count_l511_51138


namespace NUMINAMATH_CALUDE_difference_of_prime_squares_can_be_perfect_square_l511_51188

theorem difference_of_prime_squares_can_be_perfect_square :
  ∃ (p q : ℕ) (n : ℕ), Prime p ∧ Prime q ∧ p^2 - q^2 = n^2 := by
  sorry

end NUMINAMATH_CALUDE_difference_of_prime_squares_can_be_perfect_square_l511_51188


namespace NUMINAMATH_CALUDE_ellipse_focus_l511_51129

/-- An ellipse with semi-major axis 5 and semi-minor axis m has its left focus at (-3,0) -/
theorem ellipse_focus (m : ℝ) (h1 : m > 0) : 
  (∀ x y : ℝ, x^2 / 25 + y^2 / m^2 = 1) → (-3 : ℝ)^2 = 25 - m^2 → m = 4 :=
by sorry

end NUMINAMATH_CALUDE_ellipse_focus_l511_51129


namespace NUMINAMATH_CALUDE_largest_valid_number_l511_51105

def is_valid_number (n : ℕ) : Prop :=
  n ≥ 1000 ∧ n ≤ 9999 ∧
  (n % 100 = (n / 100 % 10 + n / 1000) % 10 * 10 + (n / 10 % 10 + n / 100 % 10) % 10)

theorem largest_valid_number : 
  is_valid_number 9099 ∧ ∀ m : ℕ, is_valid_number m → m ≤ 9099 :=
sorry

end NUMINAMATH_CALUDE_largest_valid_number_l511_51105


namespace NUMINAMATH_CALUDE_dima_numbers_l511_51194

def is_valid_pair (a b : ℕ) : Prop :=
  (a = 1 ∧ b = 1) ∨ (a = 1 ∧ b = 2) ∨ (a = 2 ∧ b = 1) ∨
  (a = 3 ∧ b = 6) ∨ (a = 4 ∧ b = 4) ∨ (a = 6 ∧ b = 3)

theorem dima_numbers (a b : ℕ) :
  (4 * a = b + (a + b) + (a * b)) ∨ (4 * b = a + (a + b) + (a * b)) ∨
  (4 * (a + b) = a + b + (a * b)) →
  is_valid_pair a b := by
  sorry

end NUMINAMATH_CALUDE_dima_numbers_l511_51194


namespace NUMINAMATH_CALUDE_grandmothers_age_is_52_l511_51183

/-- The age of the grandmother given the average age of the family and the ages of the children -/
def grandmothers_age (average_age : ℝ) (child1_age child2_age child3_age : ℕ) : ℝ :=
  4 * average_age - (child1_age + child2_age + child3_age)

/-- Theorem stating that the grandmother's age is 52 given the problem conditions -/
theorem grandmothers_age_is_52 :
  grandmothers_age 20 5 10 13 = 52 := by
  sorry

end NUMINAMATH_CALUDE_grandmothers_age_is_52_l511_51183


namespace NUMINAMATH_CALUDE_percentage_problem_l511_51151

theorem percentage_problem (x : ℝ) (p : ℝ) 
  (h1 : 0.2 * x = 80) 
  (h2 : p / 100 * x = 160) : 
  p = 40 := by
  sorry

end NUMINAMATH_CALUDE_percentage_problem_l511_51151


namespace NUMINAMATH_CALUDE_min_value_theorem_l511_51149

theorem min_value_theorem (x y : ℝ) (hx : x > 0) (hy : y > 0) (hsum : x + y = 1) :
  ∃ (min_val : ℝ), min_val = 3 + 2 * Real.sqrt 2 ∧
  ∀ (z : ℝ), z = 2 / x + 1 / y → z ≥ min_val :=
by sorry

end NUMINAMATH_CALUDE_min_value_theorem_l511_51149


namespace NUMINAMATH_CALUDE_solution_value_l511_51160

theorem solution_value (x a : ℝ) : x = 2 ∧ 2 * x + a = 3 → a = -1 := by
  sorry

end NUMINAMATH_CALUDE_solution_value_l511_51160


namespace NUMINAMATH_CALUDE_last_digit_of_sum_l511_51161

theorem last_digit_of_sum (n : ℕ) : 
  (54^2020 + 28^2022) % 10 = 0 := by sorry

end NUMINAMATH_CALUDE_last_digit_of_sum_l511_51161


namespace NUMINAMATH_CALUDE_compare_expressions_l511_51148

theorem compare_expressions (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (a + b) * (a^2 + b^2) ≤ 2 * (a^3 + b^3) := by
  sorry

end NUMINAMATH_CALUDE_compare_expressions_l511_51148


namespace NUMINAMATH_CALUDE_prob_even_sum_half_l511_51167

/-- Represents a die with a specified number of faces -/
structure Die where
  faces : ℕ
  face_range : faces > 0

/-- The probability of getting an even sum when rolling two dice -/
def prob_even_sum (d1 d2 : Die) : ℚ :=
  let even_outcomes := (d1.faces.div 2) * (d2.faces.div 2) + 
                       ((d1.faces + 1).div 2) * ((d2.faces + 1).div 2)
  even_outcomes / (d1.faces * d2.faces)

/-- Theorem stating that the probability of an even sum with the specified dice is 1/2 -/
theorem prob_even_sum_half :
  let d1 : Die := ⟨8, by norm_num⟩
  let d2 : Die := ⟨6, by norm_num⟩
  prob_even_sum d1 d2 = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_prob_even_sum_half_l511_51167


namespace NUMINAMATH_CALUDE_throne_identity_l511_51104

/-- Represents the types of beings in this problem -/
inductive Being
| Knight  -- Always tells the truth
| Liar    -- Always lies
| Human   -- Can either tell the truth or lie
| Monkey  -- An animal

/-- Represents a statement made by a being -/
structure Statement where
  content : Prop
  speaker : Being

/-- The statement made by the being on the throne -/
def throneStatement : Statement :=
  { content := (∃ x : Being, x = Being.Liar ∧ x = Being.Monkey),
    speaker := Being.Human }

/-- Theorem stating that the being on the throne must be a human who is lying -/
theorem throne_identity :
  throneStatement.speaker = Being.Human ∧ 
  ¬throneStatement.content :=
sorry

end NUMINAMATH_CALUDE_throne_identity_l511_51104


namespace NUMINAMATH_CALUDE_seven_twelfths_decimal_l511_51124

theorem seven_twelfths_decimal : 
  (7 : ℚ) / 12 = 0.5833333333333333 := by sorry

end NUMINAMATH_CALUDE_seven_twelfths_decimal_l511_51124


namespace NUMINAMATH_CALUDE_reduced_price_is_three_l511_51107

/-- Represents the price reduction and quantity increase for apples -/
structure ApplePriceReduction where
  reduction_percent : ℝ
  additional_apples : ℕ
  fixed_price : ℝ

/-- Calculates the reduced price per dozen apples given the price reduction information -/
def reduced_price_per_dozen (info : ApplePriceReduction) : ℝ :=
  sorry

/-- Theorem stating that for the given conditions, the reduced price per dozen is 3 Rs -/
theorem reduced_price_is_three (info : ApplePriceReduction) 
  (h1 : info.reduction_percent = 40)
  (h2 : info.additional_apples = 64)
  (h3 : info.fixed_price = 40) : 
  reduced_price_per_dozen info = 3 :=
sorry

end NUMINAMATH_CALUDE_reduced_price_is_three_l511_51107


namespace NUMINAMATH_CALUDE_apple_consumption_duration_l511_51184

theorem apple_consumption_duration (apples_per_box : ℕ) (num_boxes : ℕ) (num_people : ℕ) (apples_per_person_per_day : ℕ) :
  apples_per_box = 14 →
  num_boxes = 3 →
  num_people = 2 →
  apples_per_person_per_day = 1 →
  (apples_per_box * num_boxes) / (num_people * apples_per_person_per_day * 7) = 3 := by
  sorry

end NUMINAMATH_CALUDE_apple_consumption_duration_l511_51184


namespace NUMINAMATH_CALUDE_rectangle_area_l511_51166

theorem rectangle_area (L W : ℝ) (h1 : 2 * L + W = 34) (h2 : L + 2 * W = 38) :
  L * W = 140 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_area_l511_51166
