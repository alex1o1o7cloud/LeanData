import Mathlib

namespace NUMINAMATH_CALUDE_right_triangle_perimeter_l2208_220889

theorem right_triangle_perimeter (area : ℝ) (leg1 : ℝ) (leg2 : ℝ) (hypotenuse : ℝ) :
  area = 150 →
  leg1 = 30 →
  area = (1 / 2) * leg1 * leg2 →
  hypotenuse^2 = leg1^2 + leg2^2 →
  leg1 + leg2 + hypotenuse = 40 + 10 * Real.sqrt 10 :=
by sorry

end NUMINAMATH_CALUDE_right_triangle_perimeter_l2208_220889


namespace NUMINAMATH_CALUDE_circle_area_tangent_to_hyperbola_and_xaxis_l2208_220853

/-- A hyperbola in the xy-plane -/
def Hyperbola (x y : ℝ) : Prop := x^2 - 20*y^2 = 24

/-- A circle in the xy-plane -/
def Circle (center : ℝ × ℝ) (radius : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | (p.1 - center.1)^2 + (p.2 - center.2)^2 = radius^2}

/-- A point is on the x-axis if its y-coordinate is 0 -/
def OnXAxis (p : ℝ × ℝ) : Prop := p.2 = 0

/-- A circle is tangent to the hyperbola if there exists a point that satisfies both equations -/
def TangentToHyperbola (c : Set (ℝ × ℝ)) : Prop :=
  ∃ p : ℝ × ℝ, p ∈ c ∧ Hyperbola p.1 p.2

/-- A circle is tangent to the x-axis if there exists a point on the circle that is also on the x-axis -/
def TangentToXAxis (c : Set (ℝ × ℝ)) : Prop :=
  ∃ p : ℝ × ℝ, p ∈ c ∧ OnXAxis p

theorem circle_area_tangent_to_hyperbola_and_xaxis :
  ∃ (center : ℝ × ℝ) (radius : ℝ),
    let c := Circle center radius
    TangentToHyperbola c ∧ TangentToXAxis c ∧ π * radius^2 = 504 * π :=
by sorry

end NUMINAMATH_CALUDE_circle_area_tangent_to_hyperbola_and_xaxis_l2208_220853


namespace NUMINAMATH_CALUDE_simplify_expression_l2208_220834

theorem simplify_expression (a b c : ℝ) :
  (18 * a + 72 * b + 30 * c) + (15 * a + 40 * b - 20 * c) - (12 * a + 60 * b + 25 * c) = 21 * a + 52 * b - 15 * c := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l2208_220834


namespace NUMINAMATH_CALUDE_rhombus_longer_diagonal_l2208_220814

/-- A rhombus with given side length and shorter diagonal has a specific longer diagonal length -/
theorem rhombus_longer_diagonal (side_length shorter_diagonal : ℝ) 
  (h1 : side_length = 65)
  (h2 : shorter_diagonal = 72) : 
  ∃ longer_diagonal : ℝ, longer_diagonal = 108 ∧ 
  longer_diagonal^2 = 4 * (side_length^2 - (shorter_diagonal/2)^2) := by
  sorry

end NUMINAMATH_CALUDE_rhombus_longer_diagonal_l2208_220814


namespace NUMINAMATH_CALUDE_fraction_difference_l2208_220881

def fractions : List ℚ := [2/3, 3/4, 4/5, 5/7, 7/10, 11/13, 14/19]

theorem fraction_difference : 
  (List.maximum fractions).get! - (List.minimum fractions).get! = 11/13 - 2/3 := by
  sorry

end NUMINAMATH_CALUDE_fraction_difference_l2208_220881


namespace NUMINAMATH_CALUDE_chairs_to_remove_l2208_220873

theorem chairs_to_remove (initial_chairs : Nat) (chairs_per_row : Nat) (participants : Nat) 
  (chairs_to_remove : Nat) :
  initial_chairs = 169 →
  chairs_per_row = 13 →
  participants = 95 →
  chairs_to_remove = 65 →
  (initial_chairs - chairs_to_remove) % chairs_per_row = 0 ∧
  initial_chairs - chairs_to_remove ≥ participants ∧
  ∀ n : Nat, n < chairs_to_remove → 
    (initial_chairs - n) % chairs_per_row ≠ 0 ∨ 
    initial_chairs - n < participants := by
  sorry

end NUMINAMATH_CALUDE_chairs_to_remove_l2208_220873


namespace NUMINAMATH_CALUDE_price_reduction_equation_l2208_220832

theorem price_reduction_equation (x : ℝ) : 
  (∀ (original_price final_price : ℝ),
    original_price = 100 ∧ 
    final_price = 81 ∧ 
    final_price = original_price * (1 - x)^2) →
  100 * (1 - x)^2 = 81 :=
by sorry

end NUMINAMATH_CALUDE_price_reduction_equation_l2208_220832


namespace NUMINAMATH_CALUDE_problem_statement_l2208_220847

theorem problem_statement (x y z : ℝ) 
  (h_pos_x : x > 0) (h_pos_y : y > 0) (h_pos_z : z > 0)
  (h1 : x * (y + z) = 132)
  (h2 : z * (x + y) = 180)
  (h3 : x * y * z = 160) : 
  y * (z + x) = 160 := by
sorry

end NUMINAMATH_CALUDE_problem_statement_l2208_220847


namespace NUMINAMATH_CALUDE_original_average_calculation_l2208_220857

theorem original_average_calculation (total_pupils : ℕ) 
  (removed_pupils : ℕ) (removed_total : ℕ) (new_average : ℕ) : 
  total_pupils = 21 →
  removed_pupils = 4 →
  removed_total = 71 →
  new_average = 44 →
  (total_pupils * (total_pupils - removed_pupils) * new_average + 
   total_pupils * removed_total) / (total_pupils * total_pupils) = 39 :=
by sorry

end NUMINAMATH_CALUDE_original_average_calculation_l2208_220857


namespace NUMINAMATH_CALUDE_log_sum_equality_l2208_220899

theorem log_sum_equality : Real.log 3 / Real.log 2 * (Real.log 4 / Real.log 3) + Real.log 8 / Real.log 4 + (5 : ℝ) ^ (Real.log 2 / Real.log 5) = 11 / 2 := by
  sorry

end NUMINAMATH_CALUDE_log_sum_equality_l2208_220899


namespace NUMINAMATH_CALUDE_parentheses_removal_l2208_220878

theorem parentheses_removal (a b c : ℝ) : 3*a - (2*b - c) = 3*a - 2*b + c := by
  sorry

end NUMINAMATH_CALUDE_parentheses_removal_l2208_220878


namespace NUMINAMATH_CALUDE_total_amount_calculation_l2208_220872

/-- Calculates the total amount after simple interest is applied -/
def total_amount_after_interest (principal : ℝ) (rate : ℝ) (time : ℝ) : ℝ :=
  principal + principal * rate * time

/-- Theorem: The total amount after interest for the given conditions -/
theorem total_amount_calculation :
  let principal : ℝ := 979.0209790209791
  let rate : ℝ := 0.06
  let time : ℝ := 2.4
  total_amount_after_interest principal rate time = 1120.0649350649352 :=
by sorry

end NUMINAMATH_CALUDE_total_amount_calculation_l2208_220872


namespace NUMINAMATH_CALUDE_gcd_seven_eight_factorial_l2208_220888

theorem gcd_seven_eight_factorial : Nat.gcd (Nat.factorial 7) (Nat.factorial 8) = Nat.factorial 7 := by
  sorry

end NUMINAMATH_CALUDE_gcd_seven_eight_factorial_l2208_220888


namespace NUMINAMATH_CALUDE_sand_dune_probability_l2208_220861

/-- The probability that a sand dune remains -/
def P_remain : ℚ := 1 / 3

/-- The probability that a blown-out sand dune has a treasure -/
def P_treasure : ℚ := 1 / 5

/-- The probability that a sand dune has lucky coupons -/
def P_coupons : ℚ := 2 / 3

/-- The probability that a dune is formed in the evening -/
def P_evening : ℚ := 70 / 100

/-- The probability that a dune is formed in the morning -/
def P_morning : ℚ := 1 - P_evening

/-- The combined probability that a blown-out sand dune contains both the treasure and lucky coupons -/
def P_combined : ℚ := P_treasure * P_morning * P_coupons

theorem sand_dune_probability : P_combined = 2 / 25 := by
  sorry

end NUMINAMATH_CALUDE_sand_dune_probability_l2208_220861


namespace NUMINAMATH_CALUDE_lines_skew_iff_b_ne_18_l2208_220807

/-- Two lines in 3D space -/
structure Line3D where
  point : ℝ × ℝ × ℝ
  direction : ℝ × ℝ × ℝ

/-- Determines if two lines are skew -/
def are_skew (l1 l2 : Line3D) : Prop := sorry

/-- The first line -/
def line1 (b : ℝ) : Line3D :=
  { point := (3, 2, b),
    direction := (2, 3, 4) }

/-- The second line -/
def line2 : Line3D :=
  { point := (4, 1, 0),
    direction := (3, 4, 2) }

/-- Main theorem: The lines are skew if and only if b ≠ 18 -/
theorem lines_skew_iff_b_ne_18 (b : ℝ) :
  are_skew (line1 b) line2 ↔ b ≠ 18 := by sorry

end NUMINAMATH_CALUDE_lines_skew_iff_b_ne_18_l2208_220807


namespace NUMINAMATH_CALUDE_exam_problem_l2208_220885

/-- Proves that given the conditions of the exam problem, the number of students is 56 -/
theorem exam_problem (N : ℕ) (T : ℕ) : 
  T = 80 * N →                        -- The total marks equal 80 times the number of students
  (T - 160) / (N - 8) = 90 →          -- After excluding 8 students, the new average is 90
  N = 56 :=                           -- The number of students is 56
by
  sorry

#check exam_problem

end NUMINAMATH_CALUDE_exam_problem_l2208_220885


namespace NUMINAMATH_CALUDE_veronica_photos_l2208_220825

def choose (n k : ℕ) : ℕ := Nat.choose n k

theorem veronica_photos (x : ℕ) 
  (h1 : choose x 3 + choose x 4 = 15) : x = 7 := by
  sorry

end NUMINAMATH_CALUDE_veronica_photos_l2208_220825


namespace NUMINAMATH_CALUDE_ellipse_sum_range_l2208_220820

theorem ellipse_sum_range (x y : ℝ) (h : 9 * x^2 + 16 * y^2 = 144) :
  5 ≤ x + y + 10 ∧ x + y + 10 ≤ 15 := by
  sorry

end NUMINAMATH_CALUDE_ellipse_sum_range_l2208_220820


namespace NUMINAMATH_CALUDE_circle_radius_bounds_l2208_220826

/-- Given a quadrilateral ABCD circumscribed around a circle, where the
    tangency points divide AB into segments a and b, and AD into segments a and c,
    prove that the radius r of the circle satisfies the given inequality. -/
theorem circle_radius_bounds (a b c r : ℝ) 
    (ha : a > 0) (hb : b > 0) (hc : c > 0) (hr : r > 0) : 
  Real.sqrt ((a * b * c) / (a + b + c)) < r ∧ 
  r < Real.sqrt (a * b + b * c + c * a) := by
  sorry

end NUMINAMATH_CALUDE_circle_radius_bounds_l2208_220826


namespace NUMINAMATH_CALUDE_coin_division_problem_l2208_220800

theorem coin_division_problem (n : ℕ) : 
  (n > 0) →
  (∀ m : ℕ, m > 0 → m < n → (m % 8 ≠ 6 ∨ m % 7 ≠ 5)) →
  (n % 8 = 6) →
  (n % 7 = 5) →
  (n % 9 = 0) :=
by sorry

end NUMINAMATH_CALUDE_coin_division_problem_l2208_220800


namespace NUMINAMATH_CALUDE_compare_expressions_l2208_220864

theorem compare_expressions (m x : ℝ) : x^2 - x + 1 > -2*m^2 - 2*m*x := by
  sorry

end NUMINAMATH_CALUDE_compare_expressions_l2208_220864


namespace NUMINAMATH_CALUDE_sqrt_x_squared_nonnegative_l2208_220884

theorem sqrt_x_squared_nonnegative (x : ℝ) : 0 ≤ Real.sqrt (x^2) := by sorry

end NUMINAMATH_CALUDE_sqrt_x_squared_nonnegative_l2208_220884


namespace NUMINAMATH_CALUDE_sum_a_d_equals_ten_l2208_220831

theorem sum_a_d_equals_ten (a b c d : ℝ) 
  (h1 : a + b = 16) 
  (h2 : b + c = 9) 
  (h3 : c + d = 3) : 
  a + d = 10 := by
  sorry

end NUMINAMATH_CALUDE_sum_a_d_equals_ten_l2208_220831


namespace NUMINAMATH_CALUDE_fraction_equation_solution_l2208_220863

theorem fraction_equation_solution (x y z : ℝ) (h : x ≠ 0 ∧ y ≠ 0 ∧ y ≠ x) :
  1/x - 1/y = 1/z → z = x*y/(y-x) := by
  sorry

end NUMINAMATH_CALUDE_fraction_equation_solution_l2208_220863


namespace NUMINAMATH_CALUDE_no_real_sqrt_negative_quadratic_l2208_220839

theorem no_real_sqrt_negative_quadratic :
  ∀ x : ℝ, ¬ ∃ y : ℝ, y ^ 2 = -(x ^ 2 + 2 * x + 5) :=
by sorry

end NUMINAMATH_CALUDE_no_real_sqrt_negative_quadratic_l2208_220839


namespace NUMINAMATH_CALUDE_valid_base5_number_l2208_220843

def is_base5_digit (d : Nat) : Prop := d ≤ 4

def is_base5_number (n : Nat) : Prop :=
  ∀ d, d ∈ n.digits 5 → is_base5_digit d

theorem valid_base5_number : is_base5_number 2134 := by sorry

end NUMINAMATH_CALUDE_valid_base5_number_l2208_220843


namespace NUMINAMATH_CALUDE_shaded_area_fraction_l2208_220844

theorem shaded_area_fraction (length width : ℕ) (quarter_shaded_fraction : ℚ) (unshaded_squares : ℕ) :
  length = 15 →
  width = 20 →
  quarter_shaded_fraction = 1/4 →
  unshaded_squares = 9 →
  (quarter_shaded_fraction * (1/4 * (length * width)) - unshaded_squares) / (length * width) = 13/400 :=
by sorry

end NUMINAMATH_CALUDE_shaded_area_fraction_l2208_220844


namespace NUMINAMATH_CALUDE_antifreeze_concentration_proof_l2208_220801

-- Define the constants
def total_volume : ℝ := 55
def pure_antifreeze_volume : ℝ := 6.11
def other_mixture_concentration : ℝ := 0.1

-- Define the theorem
theorem antifreeze_concentration_proof :
  let other_mixture_volume : ℝ := total_volume - pure_antifreeze_volume
  let total_pure_antifreeze : ℝ := pure_antifreeze_volume + other_mixture_concentration * other_mixture_volume
  let final_concentration : ℝ := total_pure_antifreeze / total_volume
  ∃ ε > 0, |final_concentration - 0.2| < ε := by
  sorry

end NUMINAMATH_CALUDE_antifreeze_concentration_proof_l2208_220801


namespace NUMINAMATH_CALUDE_power_sum_equality_l2208_220865

theorem power_sum_equality : (-2 : ℤ) ^ (4 ^ 2) + 2 ^ (3 ^ 2) = 66048 := by
  sorry

end NUMINAMATH_CALUDE_power_sum_equality_l2208_220865


namespace NUMINAMATH_CALUDE_car_dealership_problem_l2208_220894

/- Define the prices of models A and B -/
def price_A : ℝ := 20
def price_B : ℝ := 15

/- Define the sales data for two weeks -/
def week1_sales : ℝ := 65
def week1_units_A : ℕ := 1
def week1_units_B : ℕ := 3

def week2_sales : ℝ := 155
def week2_units_A : ℕ := 4
def week2_units_B : ℕ := 5

/- Define the company's purchase constraints -/
def total_units : ℕ := 8
def min_cost : ℝ := 145
def max_cost : ℝ := 153

/- Define a function to calculate the cost of a purchase plan -/
def purchase_cost (units_A : ℕ) : ℝ :=
  price_A * units_A + price_B * (total_units - units_A)

/- Define a function to check if a purchase plan is valid -/
def is_valid_plan (units_A : ℕ) : Prop :=
  units_A ≤ total_units ∧ 
  min_cost ≤ purchase_cost units_A ∧ 
  purchase_cost units_A ≤ max_cost

/- Theorem statement -/
theorem car_dealership_problem :
  /- Prices satisfy the sales data -/
  (price_A * week1_units_A + price_B * week1_units_B = week1_sales) ∧
  (price_A * week2_units_A + price_B * week2_units_B = week2_sales) ∧
  /- Exactly two valid purchase plans exist -/
  (∃ (plan1 plan2 : ℕ), 
    plan1 ≠ plan2 ∧ 
    is_valid_plan plan1 ∧ 
    is_valid_plan plan2 ∧
    (∀ (plan : ℕ), is_valid_plan plan → plan = plan1 ∨ plan = plan2)) ∧
  /- The most cost-effective plan is 5 units of A and 3 units of B -/
  (∀ (plan : ℕ), is_valid_plan plan → purchase_cost 5 ≤ purchase_cost plan) :=
by sorry

end NUMINAMATH_CALUDE_car_dealership_problem_l2208_220894


namespace NUMINAMATH_CALUDE_haley_laundry_loads_l2208_220869

/-- The number of loads required to wash a given number of clothing items with a fixed-capacity washing machine. -/
def loads_required (machine_capacity : ℕ) (total_items : ℕ) : ℕ :=
  (total_items + machine_capacity - 1) / machine_capacity

theorem haley_laundry_loads :
  let machine_capacity : ℕ := 7
  let shirts : ℕ := 2
  let sweaters : ℕ := 33
  let total_items : ℕ := shirts + sweaters
  loads_required machine_capacity total_items = 5 := by
sorry

end NUMINAMATH_CALUDE_haley_laundry_loads_l2208_220869


namespace NUMINAMATH_CALUDE_smallest_delightful_integer_l2208_220837

/-- Definition of a delightful integer -/
def IsDelightful (B : ℤ) : Prop :=
  ∃ n : ℕ, (n + 1) * (2 * B + n) = 6100

/-- The smallest delightful integer -/
theorem smallest_delightful_integer :
  IsDelightful (-38) ∧ ∀ B : ℤ, B < -38 → ¬IsDelightful B :=
by sorry

end NUMINAMATH_CALUDE_smallest_delightful_integer_l2208_220837


namespace NUMINAMATH_CALUDE_quarrel_between_opposite_houses_l2208_220895

/-- Represents a house in the square yard -/
inductive House : Type
| A : House
| B : House
| C : House
| D : House

/-- Represents a quarrel between two houses -/
structure Quarrel :=
  (house1 : House)
  (house2 : House)

/-- Checks if two houses are neighbors -/
def are_neighbors (h1 h2 : House) : Prop :=
  (h1 = House.A ∧ (h2 = House.B ∨ h2 = House.D)) ∨
  (h1 = House.B ∧ (h2 = House.A ∨ h2 = House.C)) ∨
  (h1 = House.C ∧ (h2 = House.B ∨ h2 = House.D)) ∨
  (h1 = House.D ∧ (h2 = House.A ∨ h2 = House.C))

/-- Checks if two houses are opposite -/
def are_opposite (h1 h2 : House) : Prop :=
  (h1 = House.A ∧ h2 = House.C) ∨ (h1 = House.C ∧ h2 = House.A) ∨
  (h1 = House.B ∧ h2 = House.D) ∨ (h1 = House.D ∧ h2 = House.B)

theorem quarrel_between_opposite_houses 
  (total_friends : Nat)
  (quarrels : List Quarrel)
  (h_total_friends : total_friends = 77)
  (h_quarrels_count : quarrels.length = 365)
  (h_different_houses : ∀ q ∈ quarrels, q.house1 ≠ q.house2)
  (h_no_neighbor_friends : ∀ h1 h2, are_neighbors h1 h2 → 
    ∃ q ∈ quarrels, (q.house1 = h1 ∧ q.house2 = h2) ∨ (q.house1 = h2 ∧ q.house2 = h1))
  : ∃ q ∈ quarrels, are_opposite q.house1 q.house2 :=
by sorry

end NUMINAMATH_CALUDE_quarrel_between_opposite_houses_l2208_220895


namespace NUMINAMATH_CALUDE_line_slope_l2208_220836

theorem line_slope (α : Real) (h : Real.sin α + Real.cos α = 1/5) :
  Real.tan α = -4/3 := by
  sorry

end NUMINAMATH_CALUDE_line_slope_l2208_220836


namespace NUMINAMATH_CALUDE_negation_of_existence_negation_of_quadratic_equation_l2208_220883

theorem negation_of_existence (P : ℝ → Prop) : 
  (¬ ∃ x : ℝ, P x) ↔ (∀ x : ℝ, ¬ P x) := by sorry

theorem negation_of_quadratic_equation : 
  (¬ ∃ x : ℝ, x^2 - 2*x = 0) ↔ (∀ x : ℝ, x^2 - 2*x ≠ 0) := by sorry

end NUMINAMATH_CALUDE_negation_of_existence_negation_of_quadratic_equation_l2208_220883


namespace NUMINAMATH_CALUDE_ellipse_properties_l2208_220803

/-- An ellipse with center at the origin, foci on the x-axis, left focus at (-2,0), and passing through (2,√2) -/
structure Ellipse :=
  (equation : ℝ → ℝ → Prop)
  (center_origin : equation 0 0)
  (foci_on_x_axis : ∀ y, y ≠ 0 → ¬ equation (-2) y ∧ ¬ equation 2 y)
  (left_focus : equation (-2) 0)
  (passes_through : equation 2 (Real.sqrt 2))

/-- The intersection points of a line y=kx with the ellipse -/
def intersect (C : Ellipse) (k : ℝ) : Set (ℝ × ℝ) :=
  {p | C.equation p.1 p.2 ∧ p.2 = k * p.1}

/-- The y-intercepts of lines from A to intersection points -/
def y_intercepts (C : Ellipse) (k : ℝ) : Set ℝ :=
  {y | ∃ p ∈ intersect C k, y = (p.2 / (p.1 + 2*Real.sqrt 2)) * (2*Real.sqrt 2)}

/-- The theorem to be proved -/
theorem ellipse_properties (C : Ellipse) :
  (∀ x y, C.equation x y ↔ x^2/8 + y^2/4 = 1) ∧
  (∀ k ≠ 0, ∀ y ∈ y_intercepts C k,
    (0^2 + y^2 + 2*Real.sqrt 2/k*y = 4) ∧
    (2^2 + 0^2 + 2*Real.sqrt 2/k*0 = 4) ∧
    ((-2)^2 + 0^2 + 2*Real.sqrt 2/k*0 = 4)) :=
sorry

end NUMINAMATH_CALUDE_ellipse_properties_l2208_220803


namespace NUMINAMATH_CALUDE_heels_savings_per_month_l2208_220880

theorem heels_savings_per_month 
  (months_saved : ℕ) 
  (sister_contribution : ℕ) 
  (total_spent : ℕ) : 
  months_saved = 3 → 
  sister_contribution = 50 → 
  total_spent = 260 → 
  (total_spent - sister_contribution) / months_saved = 70 :=
by sorry

end NUMINAMATH_CALUDE_heels_savings_per_month_l2208_220880


namespace NUMINAMATH_CALUDE_smaller_solution_quadratic_l2208_220849

theorem smaller_solution_quadratic (x : ℝ) : 
  x^2 - 9*x + 20 = 0 → x = 4 ∨ x = 5 → min x (15 - x) = 4 := by
  sorry

end NUMINAMATH_CALUDE_smaller_solution_quadratic_l2208_220849


namespace NUMINAMATH_CALUDE_complex_fraction_equality_l2208_220845

theorem complex_fraction_equality : ∃ (i : ℂ), i * i = -1 ∧ (2 * i) / (1 - i) = -1 + i := by
  sorry

end NUMINAMATH_CALUDE_complex_fraction_equality_l2208_220845


namespace NUMINAMATH_CALUDE_det_B_squared_minus_3B_l2208_220812

def B : Matrix (Fin 2) (Fin 2) ℝ := !![2, 4; 3, 2]

theorem det_B_squared_minus_3B : 
  Matrix.det ((B ^ 2) - 3 • B) = 88 := by sorry

end NUMINAMATH_CALUDE_det_B_squared_minus_3B_l2208_220812


namespace NUMINAMATH_CALUDE_inverse_of_A_l2208_220859

def A : Matrix (Fin 2) (Fin 2) ℝ := !![5, -3; 4, -2]

theorem inverse_of_A : 
  (A⁻¹) = !![(-1 : ℝ), (3/2 : ℝ); (-2 : ℝ), (5/2 : ℝ)] := by sorry

end NUMINAMATH_CALUDE_inverse_of_A_l2208_220859


namespace NUMINAMATH_CALUDE_complement_of_P_in_U_l2208_220874

def U : Finset Int := {-1, 0, 1, 2}

def P : Set Int := {x | -Real.sqrt 2 < x ∧ x < Real.sqrt 2}

theorem complement_of_P_in_U : 
  (U.toSet \ P) = {2} := by sorry

end NUMINAMATH_CALUDE_complement_of_P_in_U_l2208_220874


namespace NUMINAMATH_CALUDE_book_selection_combinations_l2208_220876

theorem book_selection_combinations (n k : ℕ) (hn : n = 13) (hk : k = 3) :
  Nat.choose n k = 286 := by
  sorry

end NUMINAMATH_CALUDE_book_selection_combinations_l2208_220876


namespace NUMINAMATH_CALUDE_sin_150_degrees_l2208_220886

theorem sin_150_degrees : Real.sin (150 * Real.pi / 180) = 1 / 2 := by sorry

end NUMINAMATH_CALUDE_sin_150_degrees_l2208_220886


namespace NUMINAMATH_CALUDE_largest_binomial_coefficient_seventh_term_l2208_220858

theorem largest_binomial_coefficient_seventh_term :
  let n : ℕ := 8
  let k : ℕ := 6  -- 7th term corresponds to choosing 6 out of 8
  ∀ i : ℕ, i ≤ n → (n.choose k) ≥ (n.choose i) :=
by sorry

end NUMINAMATH_CALUDE_largest_binomial_coefficient_seventh_term_l2208_220858


namespace NUMINAMATH_CALUDE_parabola_sum_l2208_220802

/-- A parabola with equation y = px^2 + qx + r, vertex (3, 7), vertical axis of symmetry, and containing the point (0, 10) has p + q + r = 8 1/3 -/
theorem parabola_sum (p q r : ℝ) : 
  (∀ x y : ℝ, y = p * x^2 + q * x + r) →  -- Equation of the parabola
  (∃ a : ℝ, ∀ x y : ℝ, y = a * (x - 3)^2 + 7) →  -- Vertex form with (3, 7)
  (10 : ℝ) = p * 0^2 + q * 0 + r →  -- Point (0, 10) on the parabola
  p + q + r = 8 + 1/3 := by
sorry

end NUMINAMATH_CALUDE_parabola_sum_l2208_220802


namespace NUMINAMATH_CALUDE_strawberry_loss_l2208_220898

theorem strawberry_loss (total_weight : ℕ) (marco_weight : ℕ) (dad_weight : ℕ) 
  (h1 : total_weight = 36)
  (h2 : marco_weight = 12)
  (h3 : dad_weight = 16) :
  total_weight - (marco_weight + dad_weight) = 8 := by
  sorry

end NUMINAMATH_CALUDE_strawberry_loss_l2208_220898


namespace NUMINAMATH_CALUDE_translate_line_upward_l2208_220870

/-- Represents a line in 2D space --/
structure Line where
  slope : ℝ
  intercept : ℝ

/-- Translates a line vertically --/
def translateLine (l : Line) (shift : ℝ) : Line :=
  { slope := l.slope, intercept := l.intercept + shift }

theorem translate_line_upward (original : Line) (shift : ℝ) :
  original.slope = -2 ∧ shift = 4 →
  translateLine original shift = { slope := -2, intercept := 4 } := by
  sorry

#check translate_line_upward

end NUMINAMATH_CALUDE_translate_line_upward_l2208_220870


namespace NUMINAMATH_CALUDE_probability_not_all_same_dice_probability_not_all_same_five_eight_sided_dice_l2208_220896

theorem probability_not_all_same_dice (n : ℕ) (s : ℕ) (hn : n > 0) (hs : s > 0) : 
  1 - (s : ℚ) / (s ^ n : ℚ) = (s ^ n - s : ℚ) / (s ^ n : ℚ) :=
by sorry

-- The probability of not getting all the same numbers when rolling five fair 8-sided dice
theorem probability_not_all_same_five_eight_sided_dice : 
  1 - (8 : ℚ) / (8^5 : ℚ) = 4095 / 4096 :=
by sorry

end NUMINAMATH_CALUDE_probability_not_all_same_dice_probability_not_all_same_five_eight_sided_dice_l2208_220896


namespace NUMINAMATH_CALUDE_cups_in_first_stack_l2208_220806

theorem cups_in_first_stack (s : Fin 5 → ℕ) 
  (h1 : s 1 = 21)
  (h2 : s 2 = 25)
  (h3 : s 3 = 29)
  (h4 : s 4 = 33)
  (h_arithmetic : ∃ d : ℕ, ∀ i : Fin 4, s (i + 1) = s i + d) :
  s 0 = 17 := by
sorry

end NUMINAMATH_CALUDE_cups_in_first_stack_l2208_220806


namespace NUMINAMATH_CALUDE_total_people_needed_l2208_220841

def people_per_car : ℕ := 5

def people_per_truck (people_per_car : ℕ) : ℕ := 2 * people_per_car

def people_for_cars (num_cars : ℕ) (people_per_car : ℕ) : ℕ :=
  num_cars * people_per_car

def people_for_trucks (num_trucks : ℕ) (people_per_truck : ℕ) : ℕ :=
  num_trucks * people_per_truck

theorem total_people_needed (num_cars num_trucks : ℕ) :
  people_for_cars num_cars people_per_car +
  people_for_trucks num_trucks (people_per_truck people_per_car) = 60 :=
by
  sorry

end NUMINAMATH_CALUDE_total_people_needed_l2208_220841


namespace NUMINAMATH_CALUDE_juans_number_problem_l2208_220891

theorem juans_number_problem (n : ℝ) : 
  (2 * (n + 3)^2 - 3) / 2 = 49 → n = Real.sqrt (101 / 2) - 3 := by sorry

end NUMINAMATH_CALUDE_juans_number_problem_l2208_220891


namespace NUMINAMATH_CALUDE_special_ellipse_equation_l2208_220823

/-- An ellipse with specific properties -/
structure SpecialEllipse where
  /-- The first focus of the ellipse -/
  F₁ : ℝ × ℝ
  /-- The second focus of the ellipse -/
  F₂ : ℝ × ℝ
  /-- A point on the ellipse -/
  P : ℝ × ℝ
  /-- The first focus is at (-4, 0) -/
  h_F₁ : F₁ = (-4, 0)
  /-- The second focus is at (4, 0) -/
  h_F₂ : F₂ = (4, 0)
  /-- The dot product of PF₁ and PF₂ is zero -/
  h_perpendicular : (P.1 - F₁.1) * (P.1 - F₂.1) + (P.2 - F₁.2) * (P.2 - F₂.2) = 0
  /-- The area of triangle PF₁F₂ is 9 -/
  h_area : abs ((P.1 - F₁.1) * (P.2 - F₂.2) - (P.2 - F₁.2) * (P.1 - F₂.1)) / 2 = 9

/-- The standard equation of the special ellipse -/
def standardEquation (e : SpecialEllipse) : Prop :=
  ∀ x y : ℝ, (x^2 / 25 + y^2 / 9 = 1) ↔ (x, y) ∈ {p : ℝ × ℝ | (p.1 - e.F₁.1)^2 + (p.2 - e.F₁.2)^2 + (p.1 - e.F₂.1)^2 + (p.2 - e.F₂.2)^2 = 100}

/-- The main theorem: The standard equation of the special ellipse is x²/25 + y²/9 = 1 -/
theorem special_ellipse_equation (e : SpecialEllipse) : standardEquation e := by
  sorry

end NUMINAMATH_CALUDE_special_ellipse_equation_l2208_220823


namespace NUMINAMATH_CALUDE_lcm_gcf_product_24_36_l2208_220892

theorem lcm_gcf_product_24_36 : Nat.lcm 24 36 * Nat.gcd 24 36 = 864 := by
  sorry

end NUMINAMATH_CALUDE_lcm_gcf_product_24_36_l2208_220892


namespace NUMINAMATH_CALUDE_inequality_implication_l2208_220855

theorem inequality_implication (m n : ℝ) : -m/2 < -n/6 → 3*m > n := by
  sorry

end NUMINAMATH_CALUDE_inequality_implication_l2208_220855


namespace NUMINAMATH_CALUDE_correct_number_of_bills_l2208_220805

/-- The total amount of money in dollars -/
def total_amount : ℕ := 10000

/-- The denomination of each bill in dollars -/
def bill_denomination : ℕ := 50

/-- The number of bills -/
def number_of_bills : ℕ := total_amount / bill_denomination

theorem correct_number_of_bills : number_of_bills = 200 := by
  sorry

end NUMINAMATH_CALUDE_correct_number_of_bills_l2208_220805


namespace NUMINAMATH_CALUDE_exists_positive_integer_solution_l2208_220842

theorem exists_positive_integer_solution :
  ∃ (x y : ℕ), x > 0 ∧ y > 0 ∧ x + 2 * y = 7 :=
by sorry

end NUMINAMATH_CALUDE_exists_positive_integer_solution_l2208_220842


namespace NUMINAMATH_CALUDE_percent_relation_l2208_220804

theorem percent_relation (a b : ℝ) (h : a = 2 * b) : 5 * b = (5/2) * a := by sorry

end NUMINAMATH_CALUDE_percent_relation_l2208_220804


namespace NUMINAMATH_CALUDE_smallest_solution_comparison_l2208_220848

theorem smallest_solution_comparison (p p' q q' : ℝ) (hp : p ≠ 0) (hp' : p' ≠ 0) :
  (∃ x y : ℝ, x < y ∧ p * x^2 + q = 0 ∧ p' * y^2 + q' = 0 ∧
    (∀ z : ℝ, p * z^2 + q = 0 → x ≤ z) ∧
    (∀ w : ℝ, p' * w^2 + q' = 0 → y ≤ w)) ↔
  Real.sqrt (q' / p') < Real.sqrt (q / p) :=
sorry

end NUMINAMATH_CALUDE_smallest_solution_comparison_l2208_220848


namespace NUMINAMATH_CALUDE_smallest_valid_fourth_number_l2208_220810

def is_valid_fourth_number (n : ℕ) : Prop :=
  n ≥ 10 ∧ n < 100 ∧
  let sum_of_numbers := 42 + 25 + 56 + n
  let sum_of_digits := (4 + 2 + 2 + 5 + 5 + 6 + (n / 10) + (n % 10))
  4 * sum_of_digits = sum_of_numbers

theorem smallest_valid_fourth_number :
  ∀ n : ℕ, is_valid_fourth_number n → n ≥ 79 :=
sorry

end NUMINAMATH_CALUDE_smallest_valid_fourth_number_l2208_220810


namespace NUMINAMATH_CALUDE_A_minus_B_equality_A_minus_B_at_negative_two_l2208_220897

-- Define A and B as functions of x
def A (x : ℝ) : ℝ := 2 * x^2 - 3 * x + 2
def B (x : ℝ) : ℝ := x^2 - 3 * x - 2

-- Theorem 1: A - B = x² + 4 for all real x
theorem A_minus_B_equality (x : ℝ) : A x - B x = x^2 + 4 := by
  sorry

-- Theorem 2: A - B = 8 when x = -2
theorem A_minus_B_at_negative_two : A (-2) - B (-2) = 8 := by
  sorry

end NUMINAMATH_CALUDE_A_minus_B_equality_A_minus_B_at_negative_two_l2208_220897


namespace NUMINAMATH_CALUDE_evenResultCombinations_l2208_220866

def Operation : Type := Nat → Nat

def increaseBy2 : Operation := λ n => n + 2
def increaseBy3 : Operation := λ n => n + 3
def multiplyBy2 : Operation := λ n => n * 2

def applyOperations (ops : List Operation) (initial : Nat) : Nat :=
  ops.foldl (λ acc op => op acc) initial

def isEven (n : Nat) : Bool := n % 2 = 0

def allCombinations (n : Nat) : List (List Operation) :=
  sorry -- Implementation of all combinations of 6 operations

theorem evenResultCombinations :
  let initial := 1
  let operations := [increaseBy2, increaseBy3, multiplyBy2]
  let combinations := allCombinations 6
  (combinations.filter (λ ops => isEven (applyOperations ops initial))).length = 486 := by
  sorry

end NUMINAMATH_CALUDE_evenResultCombinations_l2208_220866


namespace NUMINAMATH_CALUDE_calculate_expression_l2208_220817

theorem calculate_expression : 3000 * (3000 ^ 3000) + 3000 ^ 2 = 3000 ^ 3001 := by
  sorry

end NUMINAMATH_CALUDE_calculate_expression_l2208_220817


namespace NUMINAMATH_CALUDE_total_water_volume_l2208_220879

theorem total_water_volume (num_boxes : ℕ) (bottles_per_box : ℕ) (bottle_capacity : ℝ) (fill_ratio : ℝ) : 
  num_boxes = 10 →
  bottles_per_box = 50 →
  bottle_capacity = 12 →
  fill_ratio = 3/4 →
  (num_boxes * bottles_per_box * bottle_capacity * fill_ratio : ℝ) = 4500 := by
  sorry

end NUMINAMATH_CALUDE_total_water_volume_l2208_220879


namespace NUMINAMATH_CALUDE_expression_factorization_l2208_220821

theorem expression_factorization (a b c : ℝ) : 
  a^4 * (b^3 - c^3) + b^4 * (c^3 - a^3) + c^4 * (a^3 - b^3) = 
  (a - b) * (b - c) * (c - a) * (-(a + b + c) * (a^2 + b^2 + c^2 + a*b + b*c + a*c)) := by
  sorry

end NUMINAMATH_CALUDE_expression_factorization_l2208_220821


namespace NUMINAMATH_CALUDE_dave_winfield_home_runs_l2208_220871

/-- Dave Winfield's career home run count -/
def dave_winfield_hr : ℕ := 465

/-- Hank Aaron's career home run count -/
def hank_aaron_hr : ℕ := 755

/-- Theorem stating Dave Winfield's home run count based on the given conditions -/
theorem dave_winfield_home_runs :
  dave_winfield_hr = 465 ∧
  hank_aaron_hr = 2 * dave_winfield_hr - 175 :=
by sorry

end NUMINAMATH_CALUDE_dave_winfield_home_runs_l2208_220871


namespace NUMINAMATH_CALUDE_geometric_sequence_a7_l2208_220838

/-- Geometric sequence with a_3 = 1 -/
def geometric_sequence (a : ℕ → ℝ) : Prop :=
  (∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n) ∧ a 3 = 1

theorem geometric_sequence_a7 (a : ℕ → ℝ) (h : geometric_sequence a) 
  (h_prod : a 6 * a 8 = 64) : a 7 = 8 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_a7_l2208_220838


namespace NUMINAMATH_CALUDE_rectangles_cover_interior_l2208_220808

-- Define the basic structures
structure Triangle where
  vertices : Fin 3 → ℝ × ℝ

structure Rectangle where
  corners : Fin 4 → ℝ × ℝ

-- Define the given line
def given_line : Set (ℝ × ℝ) := sorry

-- Define the property of covering the sides of a triangle
def covers_sides (rectangles : Fin 3 → Rectangle) (triangle : Triangle) : Prop := sorry

-- Define the property of having a side parallel to the given line
def has_parallel_side (rectangle : Rectangle) : Prop := sorry

-- Define the property of covering the interior of a triangle
def covers_interior (rectangles : Fin 3 → Rectangle) (triangle : Triangle) : Prop := sorry

-- The main theorem
theorem rectangles_cover_interior 
  (triangle : Triangle) 
  (rectangles : Fin 3 → Rectangle) 
  (h1 : covers_sides rectangles triangle)
  (h2 : ∀ i : Fin 3, has_parallel_side (rectangles i)) :
  covers_interior rectangles triangle := by sorry

end NUMINAMATH_CALUDE_rectangles_cover_interior_l2208_220808


namespace NUMINAMATH_CALUDE_count_satisfying_numbers_l2208_220818

/-- Represents a four-digit number -/
structure FourDigitNumber where
  thousands : Nat
  hundreds : Nat
  tens : Nat
  units : Nat
  thousands_nonzero : thousands > 0
  all_digits : thousands < 10 ∧ hundreds < 10 ∧ tens < 10 ∧ units < 10

/-- Checks if a four-digit number satisfies the given conditions -/
def satisfiesConditions (n : FourDigitNumber) : Prop :=
  n.thousands = 2 ∧
  n.hundreds % 2 = 0 ∧
  n.units = n.thousands + n.hundreds + n.tens

theorem count_satisfying_numbers :
  (∃ (s : Finset FourDigitNumber),
    (∀ n ∈ s, satisfiesConditions n) ∧
    s.card = 16 ∧
    (∀ n : FourDigitNumber, satisfiesConditions n → n ∈ s)) := by
  sorry

#check count_satisfying_numbers

end NUMINAMATH_CALUDE_count_satisfying_numbers_l2208_220818


namespace NUMINAMATH_CALUDE_triangle_side_value_l2208_220851

/-- In a triangle ABC, given specific conditions, prove that a = 2√3 -/
theorem triangle_side_value (A B C a b c : ℝ) : 
  0 < A ∧ 0 < B ∧ 0 < C ∧  -- Angles are positive
  A + B + C = π ∧  -- Sum of angles in a triangle
  A = 2 * C ∧  -- Given condition
  c = 2 ∧  -- Given condition
  a^2 = 4*b - 4 ∧  -- Given condition
  a / (Real.sin A) = b / (Real.sin B) ∧  -- Sine law
  a / (Real.sin A) = c / (Real.sin C) ∧  -- Sine law
  a^2 = b^2 + c^2 - 2*b*c*(Real.cos A) ∧  -- Cosine law
  b^2 = a^2 + c^2 - 2*a*c*(Real.cos B) ∧  -- Cosine law
  c^2 = a^2 + b^2 - 2*a*b*(Real.cos C)  -- Cosine law
  → a = 2 * Real.sqrt 3 := by sorry

end NUMINAMATH_CALUDE_triangle_side_value_l2208_220851


namespace NUMINAMATH_CALUDE_tangent_slope_at_one_l2208_220882

-- Define the function
def f (x : ℝ) : ℝ := x^3 + x - 2

-- State the theorem
theorem tangent_slope_at_one : 
  (deriv f) 1 = 3 := by sorry

end NUMINAMATH_CALUDE_tangent_slope_at_one_l2208_220882


namespace NUMINAMATH_CALUDE_absolute_value_ratio_l2208_220816

theorem absolute_value_ratio (a b : ℝ) (h1 : a ≠ 0) (h2 : b ≠ 0) (h3 : a^2 + b^2 = 10*a*b) :
  |((a + b) / (a - b))| = Real.sqrt 6 / 2 := by
  sorry

end NUMINAMATH_CALUDE_absolute_value_ratio_l2208_220816


namespace NUMINAMATH_CALUDE_camera_price_difference_l2208_220856

/-- The list price of Camera Y in dollars -/
def list_price : ℚ := 52.99

/-- The discount amount at Best Deals in dollars -/
def best_deals_discount : ℚ := 12

/-- The discount percentage at Market Value -/
def market_value_discount_percent : ℚ := 20

/-- The sale price at Best Deals in dollars -/
def best_deals_price : ℚ := list_price - best_deals_discount

/-- The sale price at Market Value in dollars -/
def market_value_price : ℚ := list_price * (1 - market_value_discount_percent / 100)

/-- The price difference between Market Value and Best Deals in cents -/
def price_difference_cents : ℤ := 
  ⌊(market_value_price - best_deals_price) * 100⌋

theorem camera_price_difference : price_difference_cents = 140 := by
  sorry

end NUMINAMATH_CALUDE_camera_price_difference_l2208_220856


namespace NUMINAMATH_CALUDE_sqrt_less_than_3y_iff_y_greater_than_one_ninth_l2208_220854

theorem sqrt_less_than_3y_iff_y_greater_than_one_ninth (y : ℝ) (h : y > 0) :
  Real.sqrt y < 3 * y ↔ y > 1/9 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_less_than_3y_iff_y_greater_than_one_ninth_l2208_220854


namespace NUMINAMATH_CALUDE_boxes_with_neither_l2208_220840

theorem boxes_with_neither (total : ℕ) (with_stickers : ℕ) (with_cards : ℕ) (with_both : ℕ) :
  total = 15 →
  with_stickers = 8 →
  with_cards = 5 →
  with_both = 3 →
  total - (with_stickers + with_cards - with_both) = 5 :=
by
  sorry

end NUMINAMATH_CALUDE_boxes_with_neither_l2208_220840


namespace NUMINAMATH_CALUDE_greatest_integer_satisfying_inequality_l2208_220862

theorem greatest_integer_satisfying_inequality :
  ∀ x : ℕ+, (x : ℝ)^6 / (x : ℝ)^3 < 18 → x ≤ 2 :=
by
  sorry

#check greatest_integer_satisfying_inequality

end NUMINAMATH_CALUDE_greatest_integer_satisfying_inequality_l2208_220862


namespace NUMINAMATH_CALUDE_margarets_mean_score_l2208_220868

def scores : List ℕ := [85, 87, 92, 93, 94, 98]

theorem margarets_mean_score 
  (h1 : scores.length = 6)
  (h2 : ∃ (cyprian_scores margaret_scores : List ℕ), 
        cyprian_scores.length = 3 ∧ 
        margaret_scores.length = 3 ∧ 
        cyprian_scores ++ margaret_scores = scores)
  (h3 : ∃ (cyprian_scores : List ℕ), 
        cyprian_scores.length = 3 ∧ 
        cyprian_scores.sum / cyprian_scores.length = 90) :
  ∃ (margaret_scores : List ℕ), 
    margaret_scores.length = 3 ∧ 
    margaret_scores.sum / margaret_scores.length = 93 :=
sorry

end NUMINAMATH_CALUDE_margarets_mean_score_l2208_220868


namespace NUMINAMATH_CALUDE_modulus_of_complex_number_l2208_220830

theorem modulus_of_complex_number (x : ℝ) (i : ℂ) : 
  i * i = -1 →
  (∃ (y : ℝ), (x + i) * (2 + i) = y * i) →
  Complex.abs (2 * x - i) = Real.sqrt 2 := by
sorry

end NUMINAMATH_CALUDE_modulus_of_complex_number_l2208_220830


namespace NUMINAMATH_CALUDE_solid_color_marbles_l2208_220819

theorem solid_color_marbles (total_marbles : ℕ) (solid_color_percent : ℚ) (solid_yellow_percent : ℚ)
  (h1 : solid_color_percent = 90 / 100)
  (h2 : solid_yellow_percent = 5 / 100) :
  solid_color_percent - solid_yellow_percent = 85 / 100 := by
  sorry

end NUMINAMATH_CALUDE_solid_color_marbles_l2208_220819


namespace NUMINAMATH_CALUDE_complex_magnitude_sum_reciprocal_l2208_220813

theorem complex_magnitude_sum_reciprocal (z w : ℂ) 
  (hz : Complex.abs z = 2)
  (hw : Complex.abs w = 4)
  (hzw : Complex.abs (z + w) = 3) :
  Complex.abs (1 / z + 1 / w) = 3 / 8 := by
  sorry

end NUMINAMATH_CALUDE_complex_magnitude_sum_reciprocal_l2208_220813


namespace NUMINAMATH_CALUDE_project_budget_increase_l2208_220846

/-- Proves that the annual increase in budget for project Q is $50,000 --/
theorem project_budget_increase (initial_q initial_v annual_decrease_v : ℕ) 
  (h1 : initial_q = 540000)
  (h2 : initial_v = 780000)
  (h3 : annual_decrease_v = 10000)
  (h4 : ∃ (annual_increase_q : ℕ), 
    initial_q + 4 * annual_increase_q = initial_v - 4 * annual_decrease_v) :
  ∃ (annual_increase_q : ℕ), annual_increase_q = 50000 := by
  sorry

end NUMINAMATH_CALUDE_project_budget_increase_l2208_220846


namespace NUMINAMATH_CALUDE_vector_computation_l2208_220828

theorem vector_computation :
  4 • !![3, -9] - 3 • !![2, -8] + 2 • !![1, -6] = !![8, -24] := by
  sorry

end NUMINAMATH_CALUDE_vector_computation_l2208_220828


namespace NUMINAMATH_CALUDE_smallest_lcm_with_gcd_5_l2208_220875

theorem smallest_lcm_with_gcd_5 (k l : ℕ) : 
  k ≥ 1000 ∧ k < 10000 ∧ l ≥ 1000 ∧ l < 10000 ∧ Nat.gcd k l = 5 →
  Nat.lcm k l ≥ 201000 := by
sorry

end NUMINAMATH_CALUDE_smallest_lcm_with_gcd_5_l2208_220875


namespace NUMINAMATH_CALUDE_p_sufficient_not_necessary_l2208_220890

-- Define the real number x
variable (x : ℝ)

-- Define condition p
def p (x : ℝ) : Prop := |x - 2| < 1

-- Define condition q
def q (x : ℝ) : Prop := 1 < x ∧ x < 5

-- Theorem stating that p is sufficient but not necessary for q
theorem p_sufficient_not_necessary :
  (∀ x, p x → q x) ∧ ¬(∀ x, q x → p x) := by
  sorry


end NUMINAMATH_CALUDE_p_sufficient_not_necessary_l2208_220890


namespace NUMINAMATH_CALUDE_multiplication_value_l2208_220850

theorem multiplication_value : ∃ x : ℚ, (5 / 6) * x = 10 ∧ x = 12 := by
  sorry

end NUMINAMATH_CALUDE_multiplication_value_l2208_220850


namespace NUMINAMATH_CALUDE_binary_addition_subtraction_l2208_220860

def binary_to_decimal (b : List Bool) : ℕ :=
  b.foldl (fun acc x => 2 * acc + if x then 1 else 0) 0

theorem binary_addition_subtraction :
  let a := binary_to_decimal [true, true, false, true, true]
  let b := binary_to_decimal [true, false, true, true]
  let c := binary_to_decimal [true, true, true, false, false]
  let d := binary_to_decimal [true, false, true, false, true]
  let e := binary_to_decimal [true, false, false, true]
  let result := binary_to_decimal [true, true, true, true, false]
  a + b - c + d - e = result :=
by sorry

end NUMINAMATH_CALUDE_binary_addition_subtraction_l2208_220860


namespace NUMINAMATH_CALUDE_geometric_sequence_property_l2208_220833

def is_geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = a n * r

theorem geometric_sequence_property (a : ℕ → ℝ) :
  is_geometric_sequence a →
  a 2 * a 4 = 16 →
  (a 2 * a 3 * a 4 = 64 ∨ a 2 * a 3 * a 4 = -64) :=
by sorry

end NUMINAMATH_CALUDE_geometric_sequence_property_l2208_220833


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_set_l2208_220893

theorem quadratic_inequality_solution_set (x : ℝ) : 
  x^2 + x - 12 ≥ 0 ↔ x ≤ -4 ∨ x ≥ 3 := by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_set_l2208_220893


namespace NUMINAMATH_CALUDE_colorings_count_l2208_220815

/-- Represents the colors available for coloring the cells -/
inductive Color
| Blue
| Red
| White

/-- Represents a cell in the figure -/
structure Cell :=
  (x : ℕ)
  (y : ℕ)

/-- Represents the entire figure to be colored -/
structure Figure :=
  (cells : List Cell)
  (neighbors : Cell → Cell → Bool)

/-- A coloring of the figure -/
def Coloring := Cell → Color

/-- Checks if a coloring is valid for the given figure -/
def is_valid_coloring (f : Figure) (c : Coloring) : Prop :=
  ∀ cell1 cell2, f.neighbors cell1 cell2 → c cell1 ≠ c cell2

/-- The specific figure described in the problem -/
def problem_figure : Figure := sorry

/-- The number of valid colorings for the problem figure -/
def num_valid_colorings (f : Figure) : ℕ := sorry

/-- The main theorem to be proved -/
theorem colorings_count :
  num_valid_colorings problem_figure = 3 * 48^4 := by sorry

end NUMINAMATH_CALUDE_colorings_count_l2208_220815


namespace NUMINAMATH_CALUDE_house_cost_is_280k_l2208_220835

/-- Calculates the total cost of a house given the initial deposit, mortgage duration, and monthly payment. -/
def house_cost (deposit : ℕ) (duration_years : ℕ) (monthly_payment : ℕ) : ℕ :=
  deposit + duration_years * 12 * monthly_payment

/-- Proves that the total cost of the house is $280,000 given the specified conditions. -/
theorem house_cost_is_280k :
  house_cost 40000 10 2000 = 280000 :=
by sorry

end NUMINAMATH_CALUDE_house_cost_is_280k_l2208_220835


namespace NUMINAMATH_CALUDE_complement_union_theorem_l2208_220827

-- Define the universe
def U : Set Nat := {1, 2, 3, 4, 5}

-- Define sets A and B
def A : Set Nat := {1, 2}
def B : Set Nat := {2, 3}

-- State the theorem
theorem complement_union_theorem :
  (U \ A) ∪ B = {2, 3, 4, 5} := by sorry

end NUMINAMATH_CALUDE_complement_union_theorem_l2208_220827


namespace NUMINAMATH_CALUDE_power_product_squared_l2208_220852

theorem power_product_squared : (3^5 * 6^5)^2 = 3570467226624 := by
  sorry

end NUMINAMATH_CALUDE_power_product_squared_l2208_220852


namespace NUMINAMATH_CALUDE_sum_of_common_terms_equal_1472_l2208_220867

def arithmetic_sequence (a₁ : ℕ) (d : ℕ) (n : ℕ) : List ℕ :=
  List.range n |>.map (fun i => a₁ + i * d)

def common_terms (seq1 seq2 : List ℕ) : List ℕ :=
  seq1.filter (fun x => seq2.contains x)

def sum_list (l : List ℕ) : ℕ :=
  l.foldl (· + ·) 0

theorem sum_of_common_terms_equal_1472 :
  let seq1 := arithmetic_sequence 2 4 48
  let seq2 := arithmetic_sequence 2 6 34
  let common := common_terms seq1 seq2
  sum_list common = 1472 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_common_terms_equal_1472_l2208_220867


namespace NUMINAMATH_CALUDE_circular_table_dice_probability_l2208_220822

/-- The number of people seated around the table -/
def num_people : ℕ := 5

/-- The number of sides on the die -/
def die_sides : ℕ := 8

/-- The probability of adjacent people not rolling the same number -/
def prob_not_same : ℚ := 7 / 8

/-- The probability that no two adjacent people roll the same number -/
def prob_no_adjacent_same : ℚ := (prob_not_same ^ (num_people - 1))

theorem circular_table_dice_probability :
  prob_no_adjacent_same = 2401 / 4096 := by sorry

end NUMINAMATH_CALUDE_circular_table_dice_probability_l2208_220822


namespace NUMINAMATH_CALUDE_shorter_container_radius_l2208_220829

-- Define the containers
structure Container where
  radius : ℝ
  height : ℝ

-- Define the problem
theorem shorter_container_radius 
  (c1 c2 : Container) -- Two containers
  (h_volume : c1.radius ^ 2 * c1.height = c2.radius ^ 2 * c2.height) -- Equal volume
  (h_height : c2.height = 2 * c1.height) -- One height is double the other
  (h_tall_radius : c2.radius = 10) -- Radius of taller container is 10
  : c1.radius = 10 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_shorter_container_radius_l2208_220829


namespace NUMINAMATH_CALUDE_trigonometric_identity_l2208_220887

theorem trigonometric_identity : 
  (Real.sin (47 * π / 180) - Real.sin (17 * π / 180) * Real.cos (30 * π / 180)) / 
  Real.cos (17 * π / 180) = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_trigonometric_identity_l2208_220887


namespace NUMINAMATH_CALUDE_olivia_total_time_l2208_220809

/-- The total time Olivia spent on her math problems -/
def total_time (
  num_problems : ℕ)
  (time_first_three : ℕ)
  (time_next_three : ℕ)
  (time_last : ℕ)
  (break_time : ℕ)
  (checking_time : ℕ) : ℕ :=
  3 * time_first_three + 3 * time_next_three + time_last + break_time + checking_time

/-- Theorem stating that Olivia spent 43 minutes in total on her math problems -/
theorem olivia_total_time :
  total_time 7 4 6 8 2 3 = 43 :=
by sorry

end NUMINAMATH_CALUDE_olivia_total_time_l2208_220809


namespace NUMINAMATH_CALUDE_expression_equals_one_l2208_220824

theorem expression_equals_one (x : ℝ) (h1 : x^3 ≠ 2) (h2 : x^3 ≠ -2) :
  ((x+2)^3 * (x^2-x+2)^3 / (x^3+2)^3)^3 * ((x-2)^3 * (x^2+x+2)^3 / (x^3-2)^3)^3 = 1 := by
  sorry

end NUMINAMATH_CALUDE_expression_equals_one_l2208_220824


namespace NUMINAMATH_CALUDE_hyperbola_eccentricity_l2208_220811

/-- Given a hyperbola with standard equation (x²/a² - y²/b² = 1) where a > 0 and b > 0,
    if one of its asymptotes has equation y = 3x, then its eccentricity is √10. -/
theorem hyperbola_eccentricity (a b : ℝ) (ha : a > 0) (hb : b > 0) 
  (h_asymptote : b / a = 3) : 
  Real.sqrt (1 + (b / a)^2) = Real.sqrt 10 := by
  sorry

end NUMINAMATH_CALUDE_hyperbola_eccentricity_l2208_220811


namespace NUMINAMATH_CALUDE_simplification_exponent_sum_l2208_220877

-- Define the expression
def original_expression (a b c : ℝ) : ℝ := (40 * a^5 * b^8 * c^14) ^ (1/3)

-- Define the simplified expression
def simplified_expression (a b c : ℝ) : ℝ := 2 * a * b^2 * c^4 * ((5 * a * b^2 * c^2) ^ (1/3))

-- State the theorem
theorem simplification_exponent_sum :
  ∀ a b c : ℝ, a > 0 ∧ b > 0 ∧ c > 0 →
  original_expression a b c = simplified_expression a b c ∧
  (1 + 2 + 4 = 7) := by sorry

end NUMINAMATH_CALUDE_simplification_exponent_sum_l2208_220877
