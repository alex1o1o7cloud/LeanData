import Mathlib

namespace NUMINAMATH_CALUDE_triangle_inradius_l2091_209161

/-- Given a triangle with perimeter 48 and area 60, prove that its inradius is 2.5 -/
theorem triangle_inradius (P : ℝ) (A : ℝ) (r : ℝ) 
    (h1 : P = 48) 
    (h2 : A = 60) 
    (h3 : A = r * (P / 2)) : r = 2.5 := by
  sorry

end NUMINAMATH_CALUDE_triangle_inradius_l2091_209161


namespace NUMINAMATH_CALUDE_students_on_field_trip_l2091_209104

def total_budget : ℕ := 350
def bus_rental_cost : ℕ := 100
def admission_cost_per_student : ℕ := 10

theorem students_on_field_trip : 
  (total_budget - bus_rental_cost) / admission_cost_per_student = 25 := by
  sorry

end NUMINAMATH_CALUDE_students_on_field_trip_l2091_209104


namespace NUMINAMATH_CALUDE_a_approximation_l2091_209149

/-- For large x, the value of a that makes (a * x) / (0.5x - 406) closest to 3 is approximately 1.5 -/
theorem a_approximation (x : ℝ) (hx : x > 3000) :
  ∃ (a : ℝ), ∀ (ε : ℝ), ε > 0 → 
    ∃ (δ : ℝ), δ > 0 ∧ 
      ∀ (y : ℝ), y > x → 
        |((a * y) / (0.5 * y - 406) - 3)| < ε ∧ 
        |a - 1.5| < δ :=
sorry

end NUMINAMATH_CALUDE_a_approximation_l2091_209149


namespace NUMINAMATH_CALUDE_max_perimeter_is_nine_l2091_209101

/-- Represents a configuration of three regular polygons meeting at a point -/
structure PolygonConfiguration where
  p : ℕ
  q : ℕ
  r : ℕ
  p_gt_two : p > 2
  q_gt_two : q > 2
  r_gt_two : r > 2
  distinct : p ≠ q ∧ q ≠ r ∧ p ≠ r
  angle_sum : (p - 2) / p + (q - 2) / q + (r - 2) / r = 2

/-- The perimeter of the resulting polygon -/
def perimeter (config : PolygonConfiguration) : ℕ :=
  config.p + config.q + config.r - 6

/-- Theorem stating that the maximum perimeter is 9 -/
theorem max_perimeter_is_nine :
  ∀ config : PolygonConfiguration, perimeter config ≤ 9 ∧ ∃ config : PolygonConfiguration, perimeter config = 9 :=
sorry

end NUMINAMATH_CALUDE_max_perimeter_is_nine_l2091_209101


namespace NUMINAMATH_CALUDE_translated_line_y_intercept_l2091_209172

/-- Represents a line in 2D space -/
structure Line where
  slope : ℝ
  yIntercept : ℝ

/-- Translates a line horizontally and vertically -/
def translateLine (l : Line) (dx dy : ℝ) : Line :=
  { slope := l.slope,
    yIntercept := l.yIntercept - dy + l.slope * dx }

/-- The original line y = x -/
def originalLine : Line :=
  { slope := 1,
    yIntercept := 0 }

/-- The translated line -/
def translatedLine : Line :=
  translateLine originalLine 3 (-2)

theorem translated_line_y_intercept :
  translatedLine.yIntercept = -5 := by
  sorry

end NUMINAMATH_CALUDE_translated_line_y_intercept_l2091_209172


namespace NUMINAMATH_CALUDE_max_log_sum_l2091_209152

theorem max_log_sum (a b : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : a + 2*b = 6) :
  (∀ x y : ℝ, x > 0 → y > 0 → x + 2*y = 6 → Real.log x + 2 * Real.log y ≤ 3 * Real.log 2) ∧
  (∃ x y : ℝ, x > 0 ∧ y > 0 ∧ x + 2*y = 6 ∧ Real.log x + 2 * Real.log y = 3 * Real.log 2) :=
by sorry

end NUMINAMATH_CALUDE_max_log_sum_l2091_209152


namespace NUMINAMATH_CALUDE_square_area_ratio_l2091_209155

theorem square_area_ratio (a b : ℝ) (h : a > 0 ∧ b > 0) (h_perimeter : 4 * a = 4 * (4 * b)) :
  a^2 = 16 * b^2 := by
  sorry

end NUMINAMATH_CALUDE_square_area_ratio_l2091_209155


namespace NUMINAMATH_CALUDE_power_congruence_l2091_209151

theorem power_congruence (h : 5^200 ≡ 1 [ZMOD 1000]) :
  5^6000 ≡ 1 [ZMOD 1000] := by
  sorry

end NUMINAMATH_CALUDE_power_congruence_l2091_209151


namespace NUMINAMATH_CALUDE_cube_edge_length_l2091_209176

theorem cube_edge_length (V : ℝ) (s : ℝ) (h : V = 7) (h2 : V = s^3) : s = (7 : ℝ)^(1/3) := by
  sorry

end NUMINAMATH_CALUDE_cube_edge_length_l2091_209176


namespace NUMINAMATH_CALUDE_shirts_to_wash_l2091_209126

def washing_machine_capacity : ℕ := 7
def number_of_sweaters : ℕ := 33
def number_of_loads : ℕ := 5

theorem shirts_to_wash (shirts : ℕ) : 
  shirts = number_of_loads * washing_machine_capacity - number_of_sweaters :=
by sorry

end NUMINAMATH_CALUDE_shirts_to_wash_l2091_209126


namespace NUMINAMATH_CALUDE_constant_value_proof_l2091_209120

theorem constant_value_proof (t : ℝ) (constant : ℝ) : 
  let x := 1 - 3 * t
  let y := constant * t - 3
  (t = 0.8 → x = y) → constant = 2 := by
sorry

end NUMINAMATH_CALUDE_constant_value_proof_l2091_209120


namespace NUMINAMATH_CALUDE_parabola_equation_l2091_209181

/-- Given a parabola y^2 = 2px (p > 0) and a line with slope 1 passing through its focus,
    intersecting the parabola at points A and B, if |AB| = 8, then the equation of the parabola is y^2 = 4x -/
theorem parabola_equation (p : ℝ) (A B : ℝ × ℝ) (h_p : p > 0) : 
  (∀ x y, y^2 = 2*p*x → (∃ t, y = x - p/2 + t)) →  -- Line passes through focus (p/2, 0) with slope 1
  (A.2^2 = 2*p*A.1 ∧ B.2^2 = 2*p*B.1) →            -- A and B are on the parabola
  (A.2 = A.1 - p/2 ∧ B.2 = B.1 - p/2) →            -- A and B are on the line
  (A.1 - B.1)^2 + (A.2 - B.2)^2 = 64 →             -- |AB|^2 = 8^2 = 64
  (∀ x y, y^2 = 4*x ↔ y^2 = 2*p*x) :=               -- The parabola equation is y^2 = 4x
by sorry

end NUMINAMATH_CALUDE_parabola_equation_l2091_209181


namespace NUMINAMATH_CALUDE_fourth_rectangle_area_l2091_209119

theorem fourth_rectangle_area (total_area : ℝ) (area1 area2 area3 : ℝ) :
  total_area = 168 ∧ 
  area1 = 33 ∧ 
  area2 = 45 ∧ 
  area3 = 20 →
  total_area - (area1 + area2 + area3) = 70 :=
by sorry

end NUMINAMATH_CALUDE_fourth_rectangle_area_l2091_209119


namespace NUMINAMATH_CALUDE_pat_to_mark_ratio_project_hours_ratio_l2091_209142

/-- Represents the hours charged by each person --/
structure ProjectHours where
  kate : ℕ
  pat : ℕ
  mark : ℕ

/-- Defines the conditions of the problem --/
def satisfiesConditions (hours : ProjectHours) : Prop :=
  hours.pat + hours.kate + hours.mark = 117 ∧
  hours.pat = 2 * hours.kate ∧
  hours.mark = hours.kate + 65

/-- Theorem stating the ratio of Pat's hours to Mark's hours --/
theorem pat_to_mark_ratio (hours : ProjectHours) 
  (h : satisfiesConditions hours) : 
  hours.pat * 3 = hours.mark * 1 := by
  sorry

/-- Main theorem proving the ratio is 1:3 --/
theorem project_hours_ratio : 
  ∃ hours : ProjectHours, satisfiesConditions hours ∧ hours.pat * 3 = hours.mark * 1 := by
  sorry

end NUMINAMATH_CALUDE_pat_to_mark_ratio_project_hours_ratio_l2091_209142


namespace NUMINAMATH_CALUDE_tile_border_ratio_l2091_209179

theorem tile_border_ratio :
  ∀ (n s d : ℝ),
  n > 0 →
  s > 0 →
  d > 0 →
  n = 24 →
  (24 * s)^2 / (24 * s + 25 * d)^2 = 64 / 100 →
  d / s = 6 / 25 := by
sorry

end NUMINAMATH_CALUDE_tile_border_ratio_l2091_209179


namespace NUMINAMATH_CALUDE_units_digit_sum_base_8_l2091_209189

/-- The units digit of a number in a given base -/
def unitsDigit (n : ℕ) (base : ℕ) : ℕ :=
  n % base

/-- Addition in a given base -/
def baseAddition (a b base : ℕ) : ℕ :=
  (a + b) % base^2

theorem units_digit_sum_base_8 :
  unitsDigit (baseAddition 35 47 8) 8 = 4 := by sorry

end NUMINAMATH_CALUDE_units_digit_sum_base_8_l2091_209189


namespace NUMINAMATH_CALUDE_axis_of_symmetry_is_three_l2091_209178

/-- A quadratic function passing through three points -/
structure QuadraticFunction where
  f : ℝ → ℝ
  point1 : f 1 = 8
  point2 : f 3 = -1
  point3 : f 5 = 8

/-- The axis of symmetry of a quadratic function -/
def axisOfSymmetry (q : QuadraticFunction) : ℝ := 3

/-- Theorem: The axis of symmetry of the given quadratic function is x = 3 -/
theorem axis_of_symmetry_is_three (q : QuadraticFunction) :
  axisOfSymmetry q = 3 := by sorry

end NUMINAMATH_CALUDE_axis_of_symmetry_is_three_l2091_209178


namespace NUMINAMATH_CALUDE_perpendicular_vectors_x_value_l2091_209148

-- Define the vectors
def a : Fin 2 → ℝ := ![2, 3]
def b (x : ℝ) : Fin 2 → ℝ := ![x, 1]

-- Define perpendicularity condition
def perpendicular (u v : Fin 2 → ℝ) : Prop :=
  u 0 * v 0 + u 1 * v 1 = 0

theorem perpendicular_vectors_x_value :
  ∀ x : ℝ, perpendicular a (b x) → x = -3/2 := by
  sorry

end NUMINAMATH_CALUDE_perpendicular_vectors_x_value_l2091_209148


namespace NUMINAMATH_CALUDE_puzzle_solution_l2091_209180

def puzzle_problem (total_pieces : ℕ) (num_sons : ℕ) (reyn_pieces : ℕ) : ℕ :=
  let pieces_per_son := total_pieces / num_sons
  let rhys_pieces := 2 * reyn_pieces
  let rory_pieces := 3 * reyn_pieces
  let placed_pieces := reyn_pieces + rhys_pieces + rory_pieces
  total_pieces - placed_pieces

theorem puzzle_solution :
  puzzle_problem 300 3 25 = 150 := by
  sorry

end NUMINAMATH_CALUDE_puzzle_solution_l2091_209180


namespace NUMINAMATH_CALUDE_price_drop_percentage_l2091_209102

/-- Proves that a 50% increase in quantity sold and a 20.000000000000014% increase in gross revenue
    implies a 20% decrease in price -/
theorem price_drop_percentage (P N : ℝ) (P' N' : ℝ) 
    (h_quantity_increase : N' = 1.5 * N)
    (h_revenue_increase : P' * N' = 1.20000000000000014 * (P * N)) : 
    P' = 0.8 * P := by
  sorry

end NUMINAMATH_CALUDE_price_drop_percentage_l2091_209102


namespace NUMINAMATH_CALUDE_soft_drink_bottles_l2091_209125

theorem soft_drink_bottles (small_bottles : ℕ) : 
  (small_bottles : ℝ) * 0.89 + 15000 * 0.88 = 18540 → 
  small_bottles = 6000 := by
  sorry

end NUMINAMATH_CALUDE_soft_drink_bottles_l2091_209125


namespace NUMINAMATH_CALUDE_only_prime_square_difference_pair_l2091_209160

theorem only_prime_square_difference_pair : 
  ∀ p q : ℕ, 
    Prime p → 
    Prime q → 
    p > q → 
    Prime (p^2 - q^2) → 
    p = 3 ∧ q = 2 :=
by sorry

end NUMINAMATH_CALUDE_only_prime_square_difference_pair_l2091_209160


namespace NUMINAMATH_CALUDE_point_on_line_product_of_y_coordinates_l2091_209110

theorem point_on_line_product_of_y_coordinates :
  ∀ y₁ y₂ : ℝ,
  ((-3 - 3)^2 + (-1 - y₁)^2 = 13^2) →
  ((-3 - 3)^2 + (-1 - y₂)^2 = 13^2) →
  y₁ ≠ y₂ →
  y₁ * y₂ = -132 := by sorry

end NUMINAMATH_CALUDE_point_on_line_product_of_y_coordinates_l2091_209110


namespace NUMINAMATH_CALUDE_unique_solution_for_inequalities_l2091_209144

theorem unique_solution_for_inequalities :
  ∀ (x y z : ℝ),
    (1 + x^4 ≤ 2*(y - z)^2) ∧
    (1 + y^4 ≤ 2*(z - x)^2) ∧
    (1 + z^4 ≤ 2*(x - y)^2) →
    ((x = 1 ∧ y = 0 ∧ z = -1) ∨
     (x = 1 ∧ y = -1 ∧ z = 0) ∨
     (x = 0 ∧ y = 1 ∧ z = -1) ∨
     (x = 0 ∧ y = -1 ∧ z = 1) ∨
     (x = -1 ∧ y = 1 ∧ z = 0) ∨
     (x = -1 ∧ y = 0 ∧ z = 1)) :=
by sorry

end NUMINAMATH_CALUDE_unique_solution_for_inequalities_l2091_209144


namespace NUMINAMATH_CALUDE_locus_is_ellipse_l2091_209175

-- Define the two given circles
def circle1 (x y : ℝ) : Prop := x^2 + y^2 + 6*x + 5 = 0
def circle2 (x y : ℝ) : Prop := x^2 + y^2 - 6*x - 91 = 0

-- Define the property of being externally tangent
def externally_tangent (x y : ℝ) : Prop :=
  ∃ (R : ℝ), R > 0 ∧ (x + 3)^2 + y^2 = (R + 2)^2

-- Define the property of being internally tangent
def internally_tangent (x y : ℝ) : Prop :=
  ∃ (R : ℝ), R > 0 ∧ (x - 3)^2 + y^2 = (10 - R)^2

-- Define the locus of points
def locus (x y : ℝ) : Prop :=
  externally_tangent x y ∧ internally_tangent x y

-- Theorem stating that the locus forms an ellipse
theorem locus_is_ellipse :
  ∀ (x y : ℝ), locus x y → (x + 3)^2 / 36 + y^2 / 27 = 1 :=
sorry

end NUMINAMATH_CALUDE_locus_is_ellipse_l2091_209175


namespace NUMINAMATH_CALUDE_f_extreme_values_l2091_209139

noncomputable def f (x : ℝ) : ℝ := Real.sin (Real.cos x ^ 2) + Real.sin (Real.sin x ^ 2)

theorem f_extreme_values (k : ℤ) :
  ∃ (x : ℝ), x = (k : ℝ) * Real.pi / 4 ∧ (∀ y : ℝ, f y ≤ f x ∨ f y ≥ f x) :=
sorry

end NUMINAMATH_CALUDE_f_extreme_values_l2091_209139


namespace NUMINAMATH_CALUDE_longest_line_segment_in_pie_slice_l2091_209184

theorem longest_line_segment_in_pie_slice (d : ℝ) (n : ℕ) (h_d : d = 16) (h_n : n = 4) : 
  let r := d / 2
  let θ := 2 * Real.pi / n
  let m := 2 * r * Real.sin (θ / 2)
  m ^ 2 = 128 := by sorry

end NUMINAMATH_CALUDE_longest_line_segment_in_pie_slice_l2091_209184


namespace NUMINAMATH_CALUDE_zhang_income_ratio_l2091_209141

/-- Represents the per capita income of a village at a given time -/
structure Income where
  amount : ℝ

/-- Represents the state of two villages' incomes at two different times -/
structure VillageIncomes where
  li_past : Income
  li_present : Income
  zhang_past : Income
  zhang_present : Income

/-- The conditions of the problem -/
def income_conditions (v : VillageIncomes) : Prop :=
  v.zhang_past.amount = 0.4 * v.li_past.amount ∧
  v.zhang_present.amount = 0.8 * v.li_present.amount ∧
  v.li_present.amount = 3 * v.li_past.amount

/-- The theorem to be proved -/
theorem zhang_income_ratio (v : VillageIncomes) 
  (h : income_conditions v) : 
  v.zhang_present.amount / v.zhang_past.amount = 6 := by
  sorry


end NUMINAMATH_CALUDE_zhang_income_ratio_l2091_209141


namespace NUMINAMATH_CALUDE_convention_handshakes_l2091_209190

/-- The number of handshakes in a convention with multiple companies -/
def number_of_handshakes (num_companies : ℕ) (reps_per_company : ℕ) : ℕ :=
  let total_people := num_companies * reps_per_company
  let handshakes_per_person := total_people - reps_per_company
  (total_people * handshakes_per_person) / 2

/-- Theorem: In a convention with 5 companies, each having 4 representatives,
    where every person shakes hands once with every person except those
    from their own company, the total number of handshakes is 160. -/
theorem convention_handshakes :
  number_of_handshakes 5 4 = 160 := by
  sorry

end NUMINAMATH_CALUDE_convention_handshakes_l2091_209190


namespace NUMINAMATH_CALUDE_set_intersection_problem_l2091_209140

theorem set_intersection_problem (M N : Set ℤ) (a : ℤ) 
  (hM : M = {a, 0})
  (hN : N = {1, 2})
  (hIntersection : M ∩ N = {1}) :
  a = 1 := by
  sorry

end NUMINAMATH_CALUDE_set_intersection_problem_l2091_209140


namespace NUMINAMATH_CALUDE_square_eq_four_solutions_l2091_209187

theorem square_eq_four_solutions (x : ℝ) : (x - 1)^2 = 4 ↔ x = 3 ∨ x = -1 := by
  sorry

end NUMINAMATH_CALUDE_square_eq_four_solutions_l2091_209187


namespace NUMINAMATH_CALUDE_custom_op_result_l2091_209165

-- Define the custom operation ⊗
def customOp (A B : Set ℝ) : Set ℝ := (A ∪ B) \ (A ∩ B)

-- Define sets M and N
def M : Set ℝ := {x | -2 < x ∧ x < 2}
def N : Set ℝ := {x | 1 < x ∧ x < 3}

-- Theorem statement
theorem custom_op_result :
  customOp M N = {x : ℝ | (-2 < x ∧ x ≤ 1) ∨ (2 ≤ x ∧ x < 3)} := by sorry

end NUMINAMATH_CALUDE_custom_op_result_l2091_209165


namespace NUMINAMATH_CALUDE_max_profit_at_max_price_l2091_209174

/-- Represents the monthly sales and profit of eye-protection lamps --/
structure LampSales where
  cost_price : ℝ
  selling_price : ℝ
  monthly_sales : ℝ
  profit : ℝ

/-- The conditions and constraints of the lamp sales problem --/
def lamp_sales_constraints (s : LampSales) : Prop :=
  s.cost_price = 40 ∧
  s.selling_price ≥ s.cost_price ∧
  s.selling_price ≤ 2 * s.cost_price ∧
  s.monthly_sales = -s.selling_price + 140 ∧
  s.profit = (s.selling_price - s.cost_price) * s.monthly_sales

/-- Theorem stating that the maximum monthly profit is achieved at the highest allowed selling price --/
theorem max_profit_at_max_price (s : LampSales) :
  lamp_sales_constraints s →
  ∃ (max_s : LampSales),
    lamp_sales_constraints max_s ∧
    max_s.selling_price = 80 ∧
    max_s.profit = 2400 ∧
    ∀ (other_s : LampSales), lamp_sales_constraints other_s → other_s.profit ≤ max_s.profit :=
by sorry

end NUMINAMATH_CALUDE_max_profit_at_max_price_l2091_209174


namespace NUMINAMATH_CALUDE_binomial_coefficient_sum_l2091_209150

theorem binomial_coefficient_sum (a₀ a₁ a₂ a₃ a₄ a₅ a₆ a₇ a₈ a₉ : ℝ) :
  (∀ x, (1 - x)^9 = a₀ + a₁*x + a₂*x^2 + a₃*x^3 + a₄*x^4 + a₅*x^5 + a₆*x^6 + a₇*x^7 + a₈*x^8 + a₉*x^9) →
  |a₁| + |a₂| + |a₃| + |a₄| + |a₅| + |a₆| + |a₇| + |a₈| + |a₉| = 511 := by
  sorry

end NUMINAMATH_CALUDE_binomial_coefficient_sum_l2091_209150


namespace NUMINAMATH_CALUDE_cheryl_material_problem_l2091_209121

theorem cheryl_material_problem (x : ℚ) :
  -- Cheryl buys x square yards of first material and 1/3 of second
  -- After project, 15/40 square yards left unused
  -- Total amount used is 1/3 square yards
  (x + 1/3 - 15/40 = 1/3) →
  -- The amount of first material needed is 3/8 square yards
  x = 3/8 := by
  sorry

end NUMINAMATH_CALUDE_cheryl_material_problem_l2091_209121


namespace NUMINAMATH_CALUDE_opposite_number_pairs_l2091_209153

theorem opposite_number_pairs : 
  (-(-(3 : ℤ)) = -(-|(-(3 : ℤ))|)) ∧ 
  ((-(2 : ℤ))^4 = -(2^4)) ∧ 
  ¬((-(2 : ℤ))^3 = -((-(3 : ℤ))^2)) ∧ 
  ¬((-(2 : ℤ))^3 = -(2^3)) := by
  sorry

end NUMINAMATH_CALUDE_opposite_number_pairs_l2091_209153


namespace NUMINAMATH_CALUDE_honey_harvest_increase_l2091_209193

def last_year_harvest : ℕ := 2479
def this_year_harvest : ℕ := 8564

theorem honey_harvest_increase :
  this_year_harvest - last_year_harvest = 6085 :=
by sorry

end NUMINAMATH_CALUDE_honey_harvest_increase_l2091_209193


namespace NUMINAMATH_CALUDE_maggis_cupcakes_l2091_209166

theorem maggis_cupcakes (cupcakes_per_package : ℕ) (cupcakes_eaten : ℕ) (cupcakes_left : ℕ) :
  cupcakes_per_package = 4 →
  cupcakes_eaten = 5 →
  cupcakes_left = 12 →
  ∃ (initial_packages : ℕ), 
    initial_packages * cupcakes_per_package = cupcakes_left + cupcakes_eaten ∧
    initial_packages = 4 :=
by sorry

end NUMINAMATH_CALUDE_maggis_cupcakes_l2091_209166


namespace NUMINAMATH_CALUDE_average_of_remaining_numbers_l2091_209167

theorem average_of_remaining_numbers
  (total_count : Nat)
  (total_average : ℚ)
  (subset_count : Nat)
  (subset_average : ℚ)
  (h1 : total_count = 50)
  (h2 : total_average = 76)
  (h3 : subset_count = 40)
  (h4 : subset_average = 80)
  (h5 : subset_count < total_count) :
  let remaining_count := total_count - subset_count
  let remaining_sum := total_count * total_average - subset_count * subset_average
  remaining_sum / remaining_count = 60 := by
sorry

end NUMINAMATH_CALUDE_average_of_remaining_numbers_l2091_209167


namespace NUMINAMATH_CALUDE_smallest_n_with_properties_l2091_209122

theorem smallest_n_with_properties : ∃ (n : ℕ), 
  (n > 0) ∧ 
  (∃ (a : ℕ), 3 * n = a^2) ∧ 
  (∃ (b : ℕ), 2 * n = b^3) ∧ 
  (∃ (c : ℕ), 5 * n = c^5) ∧ 
  (∀ (m : ℕ), m > 0 → 
    ((∃ (x : ℕ), 3 * m = x^2) ∧ 
     (∃ (y : ℕ), 2 * m = y^3) ∧ 
     (∃ (z : ℕ), 5 * m = z^5)) → 
    m ≥ 7500) ∧
  n = 7500 :=
by sorry

end NUMINAMATH_CALUDE_smallest_n_with_properties_l2091_209122


namespace NUMINAMATH_CALUDE_cubic_equation_root_l2091_209169

theorem cubic_equation_root (a b : ℚ) : 
  (3 + Real.sqrt 5)^3 + a * (3 + Real.sqrt 5)^2 + b * (3 + Real.sqrt 5) - 40 = 0 → b = 64 := by
  sorry

end NUMINAMATH_CALUDE_cubic_equation_root_l2091_209169


namespace NUMINAMATH_CALUDE_ramanujan_number_l2091_209185

def hardy : ℂ := Complex.mk 7 4

theorem ramanujan_number (r : ℂ) : r * hardy = Complex.mk 60 (-18) → r = Complex.mk (174/65) (-183/65) := by
  sorry

end NUMINAMATH_CALUDE_ramanujan_number_l2091_209185


namespace NUMINAMATH_CALUDE_min_total_cost_l2091_209145

/-- Represents a dish with its price and quantity -/
structure Dish where
  price : ℕ
  quantity : ℕ

/-- Calculates the total price of a dish -/
def dishTotal (d : Dish) : ℕ := d.price * d.quantity

/-- Applies discount to an order based on its total -/
def applyDiscount (total : ℕ) : ℕ :=
  if total > 100 then total - 45
  else if total > 60 then total - 30
  else if total > 30 then total - 12
  else total

/-- Calculates the final cost of an order including delivery fee -/
def orderCost (total : ℕ) : ℕ := applyDiscount total + 3

/-- Theorem: The minimum total cost for Xiaoyu's order is 54 -/
theorem min_total_cost (dishes : List Dish) 
  (h1 : dishes = [
    ⟨30, 1⟩, -- Boiled Beef
    ⟨12, 1⟩, -- Vinegar Potatoes
    ⟨30, 1⟩, -- Spare Ribs in Black Bean Sauce
    ⟨12, 1⟩, -- Hand-Torn Cabbage
    ⟨3, 2⟩   -- Rice
  ]) :
  (dishes.map dishTotal).sum = 90 →
  ∃ (order1 order2 : ℕ), 
    order1 + order2 = 90 ∧ 
    orderCost order1 + orderCost order2 = 54 ∧
    ∀ (split1 split2 : ℕ), 
      split1 + split2 = 90 → 
      orderCost split1 + orderCost split2 ≥ 54 := by
  sorry

end NUMINAMATH_CALUDE_min_total_cost_l2091_209145


namespace NUMINAMATH_CALUDE_garden_spaces_per_row_l2091_209143

/-- Represents a vegetable garden with given properties --/
structure Garden where
  tomatoes : Nat
  cucumbers : Nat
  potatoes : Nat
  rows : Nat
  additional_capacity : Nat

/-- Calculates the number of spaces in each row of the garden --/
def spaces_per_row (g : Garden) : Nat :=
  ((g.tomatoes + g.cucumbers + g.potatoes + g.additional_capacity) / g.rows)

/-- Theorem stating that for the given garden configuration, there are 15 spaces per row --/
theorem garden_spaces_per_row :
  let g : Garden := {
    tomatoes := 3 * 5,
    cucumbers := 5 * 4,
    potatoes := 30,
    rows := 10,
    additional_capacity := 85
  }
  spaces_per_row g = 15 := by
  sorry

end NUMINAMATH_CALUDE_garden_spaces_per_row_l2091_209143


namespace NUMINAMATH_CALUDE_science_club_enrollment_l2091_209192

theorem science_club_enrollment (total : ℕ) (biology : ℕ) (chemistry : ℕ) (both : ℕ) 
  (h1 : total = 60)
  (h2 : biology = 40)
  (h3 : chemistry = 35)
  (h4 : both = 25) :
  total - (biology + chemistry - both) = 10 := by
  sorry

end NUMINAMATH_CALUDE_science_club_enrollment_l2091_209192


namespace NUMINAMATH_CALUDE_sector_area_l2091_209168

/-- The area of a circular sector with central angle 54° and radius 20 cm is 60π cm² -/
theorem sector_area (θ : Real) (r : Real) : 
  θ = 54 * π / 180 → r = 20 → (1/2) * r^2 * θ = 60 * π := by
  sorry

end NUMINAMATH_CALUDE_sector_area_l2091_209168


namespace NUMINAMATH_CALUDE_line_equation_l2091_209109

-- Define the parabola
def parabola (x y : ℝ) : Prop := y^2 = 8*x

-- Define a line passing through point P(-2, 0)
def line_through_P (k b : ℝ) (x y : ℝ) : Prop := y = k * (x + 2) + b

-- Define the condition of intersection at only one point
def intersects_once (k b : ℝ) : Prop :=
  (∃! p : ℝ × ℝ, parabola p.1 p.2 ∧ line_through_P k b p.1 p.2)

-- The main theorem
theorem line_equation :
  ∀ k b : ℝ, intersects_once k b →
    (k = 0 ∧ b = 0) ∨ (k = 1 ∧ b = -2) ∨ (k = -1 ∧ b = 2) :=
sorry

end NUMINAMATH_CALUDE_line_equation_l2091_209109


namespace NUMINAMATH_CALUDE_base8_digit_product_l2091_209188

/-- Converts a natural number from base 10 to base 8 --/
def toBase8 (n : ℕ) : List ℕ :=
  sorry

/-- Calculates the product of a list of natural numbers --/
def productList (l : List ℕ) : ℕ :=
  sorry

/-- Theorem: The product of the digits in the base 8 representation of 7254 (base 10) is 72 --/
theorem base8_digit_product :
  productList (toBase8 7254) = 72 := by
  sorry

end NUMINAMATH_CALUDE_base8_digit_product_l2091_209188


namespace NUMINAMATH_CALUDE_profit_function_and_maximum_profit_constraint_and_price_l2091_209195

/-- Weekly profit function -/
def W (x : ℝ) : ℝ := -10 * x^2 + 200 * x + 15000

/-- Initial cost per box in yuan -/
def initial_cost : ℝ := 70

/-- Initial selling price per box in yuan -/
def initial_price : ℝ := 120

/-- Initial weekly sales volume in boxes -/
def initial_sales : ℝ := 300

/-- Sales increase per yuan of price reduction -/
def sales_increase_rate : ℝ := 10

theorem profit_function_and_maximum (x : ℝ) :
  W x = -10 * x^2 + 200 * x + 15000 ∧
  (∀ y : ℝ, W y ≤ W 10) ∧
  W 10 = 16000 := by sorry

theorem profit_constraint_and_price (x : ℝ) :
  W x = 15960 →
  x ≤ 12 →
  initial_price - 12 = 108 := by sorry

end NUMINAMATH_CALUDE_profit_function_and_maximum_profit_constraint_and_price_l2091_209195


namespace NUMINAMATH_CALUDE_division_remainder_proof_l2091_209170

theorem division_remainder_proof (dividend : ℕ) (divisor : ℕ) (quotient : ℕ) (remainder : ℕ) :
  dividend = 689 →
  divisor = 36 →
  quotient = 19 →
  dividend = divisor * quotient + remainder →
  remainder = 5 := by
  sorry

end NUMINAMATH_CALUDE_division_remainder_proof_l2091_209170


namespace NUMINAMATH_CALUDE_parallel_vectors_problem_l2091_209194

/-- Given vectors a, b, and c in ℝ², prove that if a is parallel to c and c = a + 3b, then x = 4 -/
theorem parallel_vectors_problem (x : ℝ) : 
  let a : Fin 2 → ℝ := ![1, -4]
  let b : Fin 2 → ℝ := ![-1, x]
  let c : Fin 2 → ℝ := a + 3 • b
  (∃ (k : ℝ), c = k • a) → x = 4 := by
  sorry

end NUMINAMATH_CALUDE_parallel_vectors_problem_l2091_209194


namespace NUMINAMATH_CALUDE_rope_cut_theorem_l2091_209111

theorem rope_cut_theorem (total_length : ℝ) (ratio_short : ℕ) (ratio_long : ℕ) 
  (h1 : total_length = 40)
  (h2 : ratio_short = 2)
  (h3 : ratio_long = 3) :
  (total_length * ratio_short) / (ratio_short + ratio_long) = 16 := by
  sorry

end NUMINAMATH_CALUDE_rope_cut_theorem_l2091_209111


namespace NUMINAMATH_CALUDE_intersecting_rectangles_area_l2091_209108

/-- Represents a rectangle with width and length -/
structure Rectangle where
  width : ℝ
  length : ℝ

/-- Calculates the area of a rectangle -/
def Rectangle.area (r : Rectangle) : ℝ := r.width * r.length

/-- The total shaded area of two intersecting rectangles -/
def totalShadedArea (r1 r2 overlap : Rectangle) : ℝ :=
  r1.area + r2.area - overlap.area

theorem intersecting_rectangles_area :
  let r1 : Rectangle := ⟨4, 12⟩
  let r2 : Rectangle := ⟨5, 10⟩
  let overlap : Rectangle := ⟨4, 5⟩
  totalShadedArea r1 r2 overlap = 78 := by
  sorry

end NUMINAMATH_CALUDE_intersecting_rectangles_area_l2091_209108


namespace NUMINAMATH_CALUDE_total_flowers_l2091_209130

theorem total_flowers (num_pots : ℕ) (flowers_per_pot : ℕ) 
  (h1 : num_pots = 2150) 
  (h2 : flowers_per_pot = 128) : 
  num_pots * flowers_per_pot = 275200 := by
  sorry

end NUMINAMATH_CALUDE_total_flowers_l2091_209130


namespace NUMINAMATH_CALUDE_angles_with_same_terminal_side_l2091_209113

def same_terminal_side (θ : ℝ) : Set ℝ :=
  {α | ∃ k : ℤ, α = k * 360 + θ}

def angle_range : Set ℝ :=
  {β | -360 ≤ β ∧ β < 720}

theorem angles_with_same_terminal_side :
  (same_terminal_side 60 ∩ angle_range) ∪ (same_terminal_side (-21) ∩ angle_range) =
  {-300, 60, 420, -21, 339, 699} := by
  sorry

end NUMINAMATH_CALUDE_angles_with_same_terminal_side_l2091_209113


namespace NUMINAMATH_CALUDE_gcd_count_for_product_360_l2091_209147

theorem gcd_count_for_product_360 (a b : ℕ+) : 
  (Nat.gcd a b * Nat.lcm a b = 360) → 
  (∃ (S : Finset ℕ+), (∀ d ∈ S, ∃ (x y : ℕ+), Nat.gcd x y = d ∧ Nat.gcd x y * Nat.lcm x y = 360) ∧ 
                       (∀ d : ℕ+, (∃ (x y : ℕ+), Nat.gcd x y = d ∧ Nat.gcd x y * Nat.lcm x y = 360) → d ∈ S) ∧ 
                       S.card = 9) := by
  sorry

end NUMINAMATH_CALUDE_gcd_count_for_product_360_l2091_209147


namespace NUMINAMATH_CALUDE_lcm_gcf_relation_l2091_209116

theorem lcm_gcf_relation (n : ℕ) (h1 : Nat.lcm n 14 = 56) (h2 : Nat.gcd n 14 = 10) : n = 40 := by
  sorry

end NUMINAMATH_CALUDE_lcm_gcf_relation_l2091_209116


namespace NUMINAMATH_CALUDE_dealer_profit_percentage_l2091_209131

/-- The profit percentage of a dealer who sells 900 grams of goods for the price of 1000 grams -/
theorem dealer_profit_percentage : 
  let actual_weight : ℝ := 900
  let claimed_weight : ℝ := 1000
  let profit_percentage := (claimed_weight / actual_weight - 1) * 100
  profit_percentage = (1 / 9) * 100 := by sorry

end NUMINAMATH_CALUDE_dealer_profit_percentage_l2091_209131


namespace NUMINAMATH_CALUDE_shyne_plants_l2091_209106

/-- The number of plants Shyne can grow from her seed packets -/
def total_plants (eggplant_per_packet : ℕ) (sunflower_per_packet : ℕ) 
                 (eggplant_packets : ℕ) (sunflower_packets : ℕ) : ℕ :=
  eggplant_per_packet * eggplant_packets + sunflower_per_packet * sunflower_packets

/-- Proof that Shyne can grow 116 plants -/
theorem shyne_plants : 
  total_plants 14 10 4 6 = 116 := by
  sorry

end NUMINAMATH_CALUDE_shyne_plants_l2091_209106


namespace NUMINAMATH_CALUDE_zero_not_in_range_of_g_l2091_209182

-- Define the function g(x)
noncomputable def g (x : ℝ) : ℤ :=
  if x > -3 then Int.ceil (1 / (x + 3))
  else if x < -3 then Int.floor (1 / (x + 3))
  else 0  -- arbitrary value for x = -3, as g is not defined there

-- Theorem statement
theorem zero_not_in_range_of_g : ∀ x : ℝ, g x ≠ 0 :=
sorry

end NUMINAMATH_CALUDE_zero_not_in_range_of_g_l2091_209182


namespace NUMINAMATH_CALUDE_exponential_inequality_l2091_209158

theorem exponential_inequality : 
  (2/5 : ℝ)^(3/5) < (2/5 : ℝ)^(2/5) ∧ (2/5 : ℝ)^(2/5) < (3/5 : ℝ)^(3/5) := by
  sorry

end NUMINAMATH_CALUDE_exponential_inequality_l2091_209158


namespace NUMINAMATH_CALUDE_intersection_and_union_subset_condition_l2091_209134

-- Define the sets A and B
def A : Set ℝ := {x | x < -4 ∨ x > 1}
def B : Set ℝ := {x | -3 ≤ x - 1 ∧ x - 1 ≤ 2}

-- Theorem for part (I)
theorem intersection_and_union :
  (A ∩ B = {x | 1 < x ∧ x ≤ 3}) ∧
  ((Aᶜ ∪ Bᶜ) = {x | x ≤ 1 ∨ x > 3}) := by sorry

-- Theorem for part (II)
theorem subset_condition (k : ℝ) :
  {x : ℝ | 2*k - 1 ≤ x ∧ x ≤ 2*k + 1} ⊆ A ↔ k > 1 ∨ k < -5/2 := by sorry

end NUMINAMATH_CALUDE_intersection_and_union_subset_condition_l2091_209134


namespace NUMINAMATH_CALUDE_alpha_beta_inequality_l2091_209105

theorem alpha_beta_inequality (α β : ℝ) (h1 : -1 < α) (h2 : α < β) (h3 : β < 1) :
  -2 < α - β ∧ α - β < 0 := by
  sorry

end NUMINAMATH_CALUDE_alpha_beta_inequality_l2091_209105


namespace NUMINAMATH_CALUDE_smallest_seating_arrangement_l2091_209107

/-- Represents a circular seating arrangement -/
structure CircularSeating :=
  (total_chairs : ℕ)
  (seated_people : ℕ)

/-- Checks if the seating arrangement satisfies the condition -/
def satisfies_condition (seating : CircularSeating) : Prop :=
  seating.seated_people > 0 ∧
  seating.seated_people ≤ seating.total_chairs ∧
  ∀ (new_seat : ℕ), new_seat < seating.total_chairs →
    ∃ (occupied_seat : ℕ), occupied_seat < seating.total_chairs ∧
      (new_seat = (occupied_seat + 1) % seating.total_chairs ∨
       new_seat = (occupied_seat + seating.total_chairs - 1) % seating.total_chairs)

/-- The main theorem to be proved -/
theorem smallest_seating_arrangement :
  ∃ (n : ℕ), n = 18 ∧
    satisfies_condition ⟨72, n⟩ ∧
    ∀ (m : ℕ), m < n → ¬satisfies_condition ⟨72, m⟩ :=
sorry

end NUMINAMATH_CALUDE_smallest_seating_arrangement_l2091_209107


namespace NUMINAMATH_CALUDE_binary_predecessor_and_successor_l2091_209191

def binary_number : ℕ := 84  -- 1010100₂ in decimal

theorem binary_predecessor_and_successor :
  (binary_number - 1 = 83) ∧ (binary_number + 1 = 85) := by
  sorry

-- Helper function to convert decimal to binary string (for reference)
def to_binary (n : ℕ) : String :=
  if n = 0 then "0"
  else
    let rec aux (m : ℕ) (acc : String) : String :=
      if m = 0 then acc
      else aux (m / 2) (toString (m % 2) ++ acc)
    aux n ""

-- These computations are to verify the binary representations
#eval to_binary binary_number        -- Should output "1010100"
#eval to_binary (binary_number - 1)  -- Should output "1010011"
#eval to_binary (binary_number + 1)  -- Should output "1010101"

end NUMINAMATH_CALUDE_binary_predecessor_and_successor_l2091_209191


namespace NUMINAMATH_CALUDE_max_a_when_a_squared_plus_100a_prime_l2091_209137

theorem max_a_when_a_squared_plus_100a_prime (a : ℕ+) :
  Nat.Prime (a^2 + 100*a) → a ≤ 1 := by
  sorry

end NUMINAMATH_CALUDE_max_a_when_a_squared_plus_100a_prime_l2091_209137


namespace NUMINAMATH_CALUDE_symmetry_of_lines_l2091_209196

-- Define the line l₁
def l₁ (x y : ℝ) : Prop := 2 * x - y + 1 = 0

-- Define the line of symmetry
def line_of_symmetry (x y : ℝ) : Prop := y = -x

-- Define the symmetric line l₂
def l₂ (x y : ℝ) : Prop := x - 2 * y + 1 = 0

-- Theorem statement
theorem symmetry_of_lines :
  ∀ (x₁ y₁ x₂ y₂ : ℝ),
    l₁ x₁ y₁ →
    line_of_symmetry ((x₁ + x₂) / 2) ((y₁ + y₂) / 2) →
    (x₂ = 2 * ((x₁ + x₂) / 2) - x₁ ∧ y₂ = 2 * ((y₁ + y₂) / 2) - y₁) →
    l₂ x₂ y₂ :=
sorry

end NUMINAMATH_CALUDE_symmetry_of_lines_l2091_209196


namespace NUMINAMATH_CALUDE_cubic_expansion_property_l2091_209146

theorem cubic_expansion_property (a₀ a₁ a₂ a₃ : ℝ) :
  (∀ x, (Real.sqrt 3 * x - 1)^3 = a₀ + a₁ * x + a₂ * x^2 + a₃ * x^3) →
  (a₀ + a₂)^2 - (a₁ + a₃)^2 = -8 := by
  sorry

end NUMINAMATH_CALUDE_cubic_expansion_property_l2091_209146


namespace NUMINAMATH_CALUDE_equal_chicken_wing_distribution_l2091_209132

theorem equal_chicken_wing_distribution 
  (num_friends : ℕ)
  (pre_cooked_wings : ℕ)
  (additional_wings : ℕ)
  (h1 : num_friends = 4)
  (h2 : pre_cooked_wings = 9)
  (h3 : additional_wings = 7) :
  (pre_cooked_wings + additional_wings) / num_friends = 4 :=
by sorry

end NUMINAMATH_CALUDE_equal_chicken_wing_distribution_l2091_209132


namespace NUMINAMATH_CALUDE_product_with_9999_l2091_209127

theorem product_with_9999 (n : ℕ) : n * 9999 = 4691130840 → n = 469200 := by
  sorry

end NUMINAMATH_CALUDE_product_with_9999_l2091_209127


namespace NUMINAMATH_CALUDE_solve_equation_l2091_209197

/-- Proves that the solution to the equation 4.7 × 13.26 + 4.7 × 9.43 + 4.7 × x = 470 is x = 77.31 -/
theorem solve_equation : 
  ∃ x : ℝ, 4.7 * 13.26 + 4.7 * 9.43 + 4.7 * x = 470 ∧ x = 77.31 := by
  sorry

end NUMINAMATH_CALUDE_solve_equation_l2091_209197


namespace NUMINAMATH_CALUDE_deal_or_no_deal_elimination_l2091_209128

theorem deal_or_no_deal_elimination (total_boxes : ℕ) (high_value_boxes : ℕ) 
  (elimination_target : ℚ) :
  total_boxes = 30 →
  high_value_boxes = 9 →
  elimination_target = 1/3 →
  ∃ (boxes_to_eliminate : ℕ),
    boxes_to_eliminate = 3 ∧
    (total_boxes - boxes_to_eliminate : ℚ) * elimination_target ≤ high_value_boxes ∧
    ∀ (n : ℕ), n < boxes_to_eliminate →
      (total_boxes - n : ℚ) * elimination_target > high_value_boxes :=
by sorry

end NUMINAMATH_CALUDE_deal_or_no_deal_elimination_l2091_209128


namespace NUMINAMATH_CALUDE_probability_sum_15_l2091_209138

def is_valid_roll (a b c : ℕ) : Prop :=
  1 ≤ a ∧ a ≤ 6 ∧ 1 ≤ b ∧ b ≤ 6 ∧ 1 ≤ c ∧ c ≤ 6

def sum_is_15 (a b c : ℕ) : Prop :=
  a + b + c = 15

def count_valid_rolls : ℕ := 216

def count_sum_15_rolls : ℕ := 10

theorem probability_sum_15 :
  (count_sum_15_rolls : ℚ) / count_valid_rolls = 5 / 108 :=
sorry

end NUMINAMATH_CALUDE_probability_sum_15_l2091_209138


namespace NUMINAMATH_CALUDE_min_apples_in_basket_l2091_209173

theorem min_apples_in_basket (N : ℕ) : 
  N ≥ 67 ∧ 
  N % 3 = 1 ∧ 
  N % 4 = 3 ∧ 
  N % 5 = 2 ∧
  (∀ m : ℕ, m < N → ¬(m % 3 = 1 ∧ m % 4 = 3 ∧ m % 5 = 2)) := by
  sorry

end NUMINAMATH_CALUDE_min_apples_in_basket_l2091_209173


namespace NUMINAMATH_CALUDE_arithmetic_sequence_first_term_range_l2091_209162

theorem arithmetic_sequence_first_term_range (a : ℕ → ℝ) (d : ℝ) (h1 : d = π / 8) :
  (∀ n : ℕ, a (n + 1) = a n + d) →
  (a 10 ≤ 0) →
  (a 11 ≥ 0) →
  -5 * π / 4 ≤ a 1 ∧ a 1 ≤ -9 * π / 8 :=
by sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_first_term_range_l2091_209162


namespace NUMINAMATH_CALUDE_star_operation_result_l2091_209100

-- Define the sets M and N
def M : Set ℝ := {y | y ≥ 0}
def N : Set ℝ := {y | -3 ≤ y ∧ y ≤ 3}

-- Define the set difference operation
def setDifference (A B : Set ℝ) : Set ℝ := {x | x ∈ A ∧ x ∉ B}

-- Define the * operation
def starOperation (A B : Set ℝ) : Set ℝ := (setDifference A B) ∪ (setDifference B A)

-- State the theorem
theorem star_operation_result :
  starOperation M N = {x : ℝ | -3 ≤ x ∧ x < 0 ∨ x > 3} := by
  sorry

end NUMINAMATH_CALUDE_star_operation_result_l2091_209100


namespace NUMINAMATH_CALUDE_max_cake_pieces_l2091_209123

/-- The size of the large cake in inches -/
def large_cake_size : ℕ := 20

/-- The size of the small piece in inches -/
def small_piece_size : ℕ := 2

/-- The area of the large cake in square inches -/
def large_cake_area : ℕ := large_cake_size * large_cake_size

/-- The area of a small piece in square inches -/
def small_piece_area : ℕ := small_piece_size * small_piece_size

/-- The maximum number of small pieces that can be cut from the large cake -/
def max_pieces : ℕ := large_cake_area / small_piece_area

theorem max_cake_pieces : max_pieces = 100 := by
  sorry

end NUMINAMATH_CALUDE_max_cake_pieces_l2091_209123


namespace NUMINAMATH_CALUDE_quadratic_inequality_bounds_l2091_209118

theorem quadratic_inequality_bounds (x : ℝ) (h : x^2 - 6*x + 8 < 0) :
  25 < x^2 + 6*x + 9 ∧ x^2 + 6*x + 9 < 49 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_bounds_l2091_209118


namespace NUMINAMATH_CALUDE_house_painting_time_l2091_209117

theorem house_painting_time (total_time joint_time john_time : ℝ) 
  (h1 : joint_time = 2.4)
  (h2 : john_time = 6)
  (h3 : 1 / total_time + 1 / john_time = 1 / joint_time) :
  total_time = 4 := by sorry

end NUMINAMATH_CALUDE_house_painting_time_l2091_209117


namespace NUMINAMATH_CALUDE_inequality_proof_l2091_209124

theorem inequality_proof (x y z : ℝ) 
  (hx : x > 0) (hy : y > 0) (hz : z > 0) 
  (hsum : x + y + z = 1) : 
  1 / Real.sqrt (x + y) + 1 / Real.sqrt (y + z) + 1 / Real.sqrt (z + x) 
  ≤ 1 / Real.sqrt (2 * x * y * z) := by
sorry

end NUMINAMATH_CALUDE_inequality_proof_l2091_209124


namespace NUMINAMATH_CALUDE_largest_angle_of_special_quadrilateral_l2091_209114

/-- A convex quadrilateral is rude if there exists a convex quadrilateral inside or on its sides
    with a larger sum of diagonals. -/
def IsRude (Q : Set (ℝ × ℝ)) : Prop := sorry

/-- The largest angle of a quadrilateral -/
def LargestAngle (Q : Set (ℝ × ℝ)) : ℝ := sorry

/-- A convex quadrilateral -/
def ConvexQuadrilateral (Q : Set (ℝ × ℝ)) : Prop := sorry

theorem largest_angle_of_special_quadrilateral 
  (A B C D : ℝ × ℝ) 
  (r : ℝ)
  (h_convex : ConvexQuadrilateral {A, B, C, D})
  (h_not_rude : ¬IsRude {A, B, C, D})
  (h_r_positive : r > 0)
  (h_nearby_rude : ∀ A', A' ≠ A → dist A' A ≤ r → IsRude {A', B, C, D}) :
  LargestAngle {A, B, C, D} = 150 * π / 180 := by sorry

end NUMINAMATH_CALUDE_largest_angle_of_special_quadrilateral_l2091_209114


namespace NUMINAMATH_CALUDE_unique_solution_l2091_209154

def is_two_digit (n : ℕ) : Prop := 10 ≤ n ∧ n ≤ 99

theorem unique_solution :
  ∃! A : ℕ, ∃ B : ℕ,
    4 * A + (10 * B + 3) = 68 ∧
    is_two_digit (4 * A) ∧
    is_two_digit (10 * B + 3) ∧
    A ≤ 9 ∧ B ≤ 9 :=
by sorry

end NUMINAMATH_CALUDE_unique_solution_l2091_209154


namespace NUMINAMATH_CALUDE_decagon_interior_intersections_l2091_209198

/-- The number of distinct interior intersection points of diagonals in a regular decagon -/
def interior_intersection_points (n : ℕ) : ℕ :=
  Nat.choose n 4

/-- A regular decagon has 10 sides -/
def decagon_sides : ℕ := 10

theorem decagon_interior_intersections :
  interior_intersection_points decagon_sides = 210 := by
  sorry

end NUMINAMATH_CALUDE_decagon_interior_intersections_l2091_209198


namespace NUMINAMATH_CALUDE_rectangle_arrangement_probability_l2091_209164

/-- Represents a rectangle with width and height -/
structure Rectangle where
  width : ℕ
  height : ℕ

/-- Represents a square -/
structure Square where
  side_length : ℕ

/-- Represents a line segment connecting midpoints of opposite sides of a square -/
structure MidpointLine where
  square : Square

/-- Represents an arrangement of rectangles in a square -/
structure Arrangement where
  square : Square
  rectangles : List Rectangle

/-- Checks if an arrangement is valid (no overlapping rectangles) -/
def is_valid_arrangement (arr : Arrangement) : Prop := sorry

/-- Checks if an arrangement crosses the midpoint line -/
def crosses_midpoint_line (arr : Arrangement) (line : MidpointLine) : Prop := sorry

/-- Counts the number of valid arrangements -/
def count_valid_arrangements (square : Square) (rect_type : Rectangle) (num_rect : ℕ) : ℕ := sorry

/-- Counts the number of valid arrangements that don't cross the midpoint line -/
def count_non_crossing_arrangements (square : Square) (rect_type : Rectangle) (num_rect : ℕ) (line : MidpointLine) : ℕ := sorry

theorem rectangle_arrangement_probability :
  let square := Square.mk 4
  let rect_type := Rectangle.mk 1 2
  let num_rect := 8
  let line := MidpointLine.mk square
  let total_arrangements := count_valid_arrangements square rect_type num_rect
  let non_crossing_arrangements := count_non_crossing_arrangements square rect_type num_rect line
  (non_crossing_arrangements : ℚ) / total_arrangements = 25 / 36 := by sorry

end NUMINAMATH_CALUDE_rectangle_arrangement_probability_l2091_209164


namespace NUMINAMATH_CALUDE_inequality_proof_l2091_209135

theorem inequality_proof (a b c d : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0) :
  (a^3 + b^3 + c^3) / (a + b + c) + (b^3 + c^3 + d^3) / (b + c + d) +
  (c^3 + d^3 + a^3) / (c + d + a) + (d^3 + a^3 + b^3) / (d + a + b) ≥
  a^2 + b^2 + c^2 + d^2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l2091_209135


namespace NUMINAMATH_CALUDE_hyperbola_real_axis_length_eq_4_div_sqrt5_l2091_209183

noncomputable def hyperbola_real_axis_length 
  (a b : ℝ) 
  (ha : a > 0) 
  (hb : b > 0) 
  (P : ℝ × ℝ) 
  (hP : P.1^2 / a^2 - P.2^2 / b^2 = 1) 
  (hP_right : P.1 > 0) 
  (A B : ℝ × ℝ) 
  (hA : A.1 > 0 ∧ A.2 > 0) 
  (hB : B.1 > 0 ∧ B.2 < 0) 
  (hAP_PB : (A.1 - P.1, A.2 - P.2) = (-1/3) • (B.1 - P.1, B.2 - P.2)) 
  (hAOB_area : (1/2) * abs (A.1 * B.2 - A.2 * B.1) = 2 * b) : 
  ℝ :=
2 * a

theorem hyperbola_real_axis_length_eq_4_div_sqrt5 
  (a b : ℝ) 
  (ha : a > 0) 
  (hb : b > 0) 
  (P : ℝ × ℝ) 
  (hP : P.1^2 / a^2 - P.2^2 / b^2 = 1) 
  (hP_right : P.1 > 0) 
  (A B : ℝ × ℝ) 
  (hA : A.1 > 0 ∧ A.2 > 0) 
  (hB : B.1 > 0 ∧ B.2 < 0) 
  (hAP_PB : (A.1 - P.1, A.2 - P.2) = (-1/3) • (B.1 - P.1, B.2 - P.2)) 
  (hAOB_area : (1/2) * abs (A.1 * B.2 - A.2 * B.1) = 2 * b) : 
  hyperbola_real_axis_length a b ha hb P hP hP_right A B hA hB hAP_PB hAOB_area = 4 / Real.sqrt 5 :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_real_axis_length_eq_4_div_sqrt5_l2091_209183


namespace NUMINAMATH_CALUDE_women_fraction_in_room_l2091_209133

theorem women_fraction_in_room (total_people : ℕ) (married_fraction : ℚ) 
  (max_unmarried_women : ℕ) (h1 : total_people = 80) (h2 : married_fraction = 1/2) 
  (h3 : max_unmarried_women = 32) : 
  (max_unmarried_women + (married_fraction * total_people / 2)) / total_people = 1/2 :=
sorry

end NUMINAMATH_CALUDE_women_fraction_in_room_l2091_209133


namespace NUMINAMATH_CALUDE_function_minimum_and_integer_bound_l2091_209115

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := x * (a + Real.log x)

theorem function_minimum_and_integer_bound :
  (∃ a : ℝ, ∀ x > 0, f a x ≥ -Real.exp (-2) ∧ ∃ x₀ > 0, f a x₀ = -Real.exp (-2)) →
  (∃ a : ℝ, a = 1 ∧
    ∀ k : ℤ, (∀ x > 1, ↑k < (f a x) / (x - 1)) →
      k ≤ 3 ∧ (∃ x > 1, 3 < (f a x) / (x - 1))) :=
by sorry

end NUMINAMATH_CALUDE_function_minimum_and_integer_bound_l2091_209115


namespace NUMINAMATH_CALUDE_imaginary_part_of_z_l2091_209157

theorem imaginary_part_of_z (i : ℂ) (z : ℂ) : 
  i * i = -1 → (1 : ℂ) + i = z * ((1 : ℂ) - i) → Complex.im z = 1 := by
  sorry

end NUMINAMATH_CALUDE_imaginary_part_of_z_l2091_209157


namespace NUMINAMATH_CALUDE_divisibility_by_36_l2091_209112

theorem divisibility_by_36 : ∃! n : ℕ, n < 10 ∧ (6130 + n) % 36 = 0 :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_divisibility_by_36_l2091_209112


namespace NUMINAMATH_CALUDE_jia_incorrect_questions_l2091_209103

-- Define the type for questions
inductive Question
| Q1 | Q2 | Q3 | Q4 | Q5 | Q6 | Q7

-- Define a person's answers
def Answers := Question → Bool

-- Define the correct answers
def correct_answers : Answers := sorry

-- Define Jia's answers
def jia_answers : Answers := sorry

-- Define Yi's answers
def yi_answers : Answers := sorry

-- Define Bing's answers
def bing_answers : Answers := sorry

-- Function to count correct answers
def count_correct (answers : Answers) : Nat := sorry

-- Theorem stating the problem conditions and the conclusion to be proved
theorem jia_incorrect_questions :
  (count_correct jia_answers = 5) →
  (count_correct yi_answers = 5) →
  (count_correct bing_answers = 5) →
  (jia_answers Question.Q1 ≠ correct_answers Question.Q1) ∧
  (jia_answers Question.Q3 ≠ correct_answers Question.Q3) :=
by sorry

end NUMINAMATH_CALUDE_jia_incorrect_questions_l2091_209103


namespace NUMINAMATH_CALUDE_no_natural_product_l2091_209163

theorem no_natural_product (n : ℕ) : ¬∃ (a b : ℕ), 3 * n + 1 = a * b := by
  sorry

end NUMINAMATH_CALUDE_no_natural_product_l2091_209163


namespace NUMINAMATH_CALUDE_parabola_one_y_intercept_l2091_209199

-- Define the parabola function
def f (x : ℝ) : ℝ := 3 * x^2 - 4 * x + 2

-- Define what a y-intercept is
def is_y_intercept (y : ℝ) : Prop := f 0 = y

-- Theorem: The parabola has exactly one y-intercept
theorem parabola_one_y_intercept :
  ∃! y : ℝ, is_y_intercept y :=
sorry

end NUMINAMATH_CALUDE_parabola_one_y_intercept_l2091_209199


namespace NUMINAMATH_CALUDE_younger_person_age_l2091_209136

/-- Given two persons whose ages differ by 20 years, and 5 years ago the elder one was 5 times as old as the younger one, the present age of the younger person is 10 years. -/
theorem younger_person_age (y e : ℕ) : 
  e = y + 20 →                  -- The ages differ by 20 years
  e - 5 = 5 * (y - 5) →         -- 5 years ago, elder was 5 times younger
  y = 10                        -- The younger person's age is 10
  := by sorry

end NUMINAMATH_CALUDE_younger_person_age_l2091_209136


namespace NUMINAMATH_CALUDE_root_sum_fraction_l2091_209159

theorem root_sum_fraction (α β γ : ℂ) : 
  α^3 - α - 1 = 0 → β^3 - β - 1 = 0 → γ^3 - γ - 1 = 0 →
  (1 + α) / (1 - α) + (1 + β) / (1 - β) + (1 + γ) / (1 - γ) = -7 := by
  sorry

end NUMINAMATH_CALUDE_root_sum_fraction_l2091_209159


namespace NUMINAMATH_CALUDE_food_drive_problem_l2091_209129

/-- Represents the food drive problem in Ms. Perez's class -/
theorem food_drive_problem (total_students : ℕ) (total_cans : ℕ) 
  (students_with_four_cans : ℕ) (students_with_zero_cans : ℕ) :
  total_students = 30 →
  total_cans = 232 →
  students_with_four_cans = 13 →
  students_with_zero_cans = 2 →
  2 * (total_students - students_with_four_cans - students_with_zero_cans) = total_students →
  (total_cans - 4 * students_with_four_cans) / 
    (total_students - students_with_four_cans - students_with_zero_cans) = 12 :=
by sorry

end NUMINAMATH_CALUDE_food_drive_problem_l2091_209129


namespace NUMINAMATH_CALUDE_ways_to_soccer_field_l2091_209177

theorem ways_to_soccer_field (walk : ℕ) (drive : ℕ) (total : ℕ) : 
  walk = 3 → drive = 4 → total = walk + drive → total = 7 := by
  sorry

end NUMINAMATH_CALUDE_ways_to_soccer_field_l2091_209177


namespace NUMINAMATH_CALUDE_lcm_15_18_l2091_209171

theorem lcm_15_18 : Nat.lcm 15 18 = 90 := by
  sorry

end NUMINAMATH_CALUDE_lcm_15_18_l2091_209171


namespace NUMINAMATH_CALUDE_trig_identity_l2091_209186

theorem trig_identity (α : ℝ) (h : Real.tan α = 3) :
  2 * (Real.sin α)^2 - (Real.sin α) * (Real.cos α) + (Real.cos α)^2 = 8/5 := by
  sorry

end NUMINAMATH_CALUDE_trig_identity_l2091_209186


namespace NUMINAMATH_CALUDE_committee_probability_l2091_209156

def total_members : ℕ := 30
def num_boys : ℕ := 12
def num_girls : ℕ := 18
def committee_size : ℕ := 6

theorem committee_probability :
  let total_combinations := Nat.choose total_members committee_size
  let all_boys_combinations := Nat.choose num_boys committee_size
  let all_girls_combinations := Nat.choose num_girls committee_size
  let prob_at_least_one_each := 1 - (all_boys_combinations + all_girls_combinations : ℚ) / total_combinations
  prob_at_least_one_each = 574287 / 593775 := by
  sorry

end NUMINAMATH_CALUDE_committee_probability_l2091_209156
