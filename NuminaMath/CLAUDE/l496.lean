import Mathlib

namespace class_average_problem_l496_49641

theorem class_average_problem (x : ℝ) :
  (0.45 * x + 0.50 * 78 + 0.05 * 60 = 84.75) →
  x = 95 := by
  sorry

end class_average_problem_l496_49641


namespace smallest_c_value_l496_49650

-- Define the polynomial
def polynomial (c d x : ℤ) : ℤ := x^3 - c*x^2 + d*x - 2310

-- Define the property that the polynomial has three positive integer roots
def has_three_positive_integer_roots (c d : ℤ) : Prop :=
  ∃ (r₁ r₂ r₃ : ℤ), r₁ > 0 ∧ r₂ > 0 ∧ r₃ > 0 ∧
    ∀ x, polynomial c d x = (x - r₁) * (x - r₂) * (x - r₃)

-- State the theorem
theorem smallest_c_value (c d : ℤ) :
  has_three_positive_integer_roots c d →
  (∀ c' d', has_three_positive_integer_roots c' d' → c ≤ c') →
  c = 52 := by
  sorry

end smallest_c_value_l496_49650


namespace count_rectangles_l496_49633

/-- The number of rectangles with sides parallel to the axes in an n×n grid -/
def num_rectangles (n : ℕ) : ℕ :=
  n^2 * (n-1)^2 / 4

/-- Theorem stating the number of rectangles in an n×n grid -/
theorem count_rectangles (n : ℕ) (h : n > 0) :
  num_rectangles n = (n.choose 2)^2 :=
by sorry

end count_rectangles_l496_49633


namespace wire_radius_from_sphere_l496_49677

/-- The radius of a wire's cross section when a sphere is melted and drawn into a wire -/
theorem wire_radius_from_sphere (r_sphere : ℝ) (l_wire : ℝ) (r_wire : ℝ) : 
  r_sphere = 12 →
  l_wire = 144 →
  (4 / 3) * Real.pi * r_sphere^3 = Real.pi * r_wire^2 * l_wire →
  r_wire = 4 := by
  sorry

#check wire_radius_from_sphere

end wire_radius_from_sphere_l496_49677


namespace inverse_24_mod_53_l496_49628

theorem inverse_24_mod_53 (h : (19⁻¹ : ZMod 53) = 31) : (24⁻¹ : ZMod 53) = 22 := by
  sorry

end inverse_24_mod_53_l496_49628


namespace zoo_visit_cost_l496_49665

/-- Calculates the total cost of a zoo visit for a group with a discount applied -/
theorem zoo_visit_cost 
  (num_children num_adults num_seniors : ℕ)
  (child_price adult_price senior_price : ℚ)
  (discount_rate : ℚ)
  (h_children : num_children = 6)
  (h_adults : num_adults = 10)
  (h_seniors : num_seniors = 4)
  (h_child_price : child_price = 12)
  (h_adult_price : adult_price = 20)
  (h_senior_price : senior_price = 15)
  (h_discount : discount_rate = 0.15) :
  (num_children : ℚ) * child_price + 
  (num_adults : ℚ) * adult_price + 
  (num_seniors : ℚ) * senior_price - 
  ((num_children : ℚ) * child_price + 
   (num_adults : ℚ) * adult_price + 
   (num_seniors : ℚ) * senior_price) * discount_rate = 282.20 := by
sorry

end zoo_visit_cost_l496_49665


namespace function_bound_l496_49688

theorem function_bound (f : ℝ → ℝ) (a : ℝ) : 
  (∀ x : ℝ, f x = Real.sqrt 3 * Real.sin (3 * x) + Real.cos (3 * x)) →
  (∀ x : ℝ, |f x| ≤ a) →
  a ≥ 2 := by
  sorry

end function_bound_l496_49688


namespace quadratic_max_value_l496_49630

theorem quadratic_max_value :
  ∃ (max : ℝ), max = 216 ∧ ∀ (s : ℝ), -3 * s^2 + 54 * s - 27 ≤ max :=
by sorry

end quadratic_max_value_l496_49630


namespace man_mass_on_boat_l496_49605

/-- The mass of a man who causes a boat to sink by a certain amount -/
def man_mass (boat_length boat_width boat_sink_depth water_density : ℝ) : ℝ :=
  boat_length * boat_width * boat_sink_depth * water_density

/-- Theorem stating that the mass of the man is 140 kg -/
theorem man_mass_on_boat :
  let boat_length : ℝ := 7
  let boat_width : ℝ := 2
  let boat_sink_depth : ℝ := 0.01  -- 1 cm in meters
  let water_density : ℝ := 1000    -- kg/m³
  man_mass boat_length boat_width boat_sink_depth water_density = 140 := by
  sorry

#eval man_mass 7 2 0.01 1000

end man_mass_on_boat_l496_49605


namespace second_month_sale_proof_l496_49646

/-- Calculates the sale in the second month given the sales of other months and the required total sales. -/
def second_month_sale (first_month : ℕ) (third_month : ℕ) (fourth_month : ℕ) (fifth_month : ℕ) (sixth_month : ℕ) (total_sales : ℕ) : ℕ :=
  total_sales - (first_month + third_month + fourth_month + fifth_month + sixth_month)

/-- Proves that the sale in the second month is 11690 given the specific sales figures. -/
theorem second_month_sale_proof :
  second_month_sale 5266 5678 6029 5922 4937 33600 = 11690 := by
  sorry

end second_month_sale_proof_l496_49646


namespace expression_simplification_l496_49684

variables (a b : ℝ)

theorem expression_simplification :
  (3*a + 2*b - 5*a - b = -2*a + b) ∧
  (5*(3*a^2*b - a*b^2) - (a*b^2 + 3*a^2*b) = 12*a^2*b - 6*a*b^2) := by
  sorry

end expression_simplification_l496_49684


namespace triangle_side_length_l496_49649

theorem triangle_side_length (A B C : Real) (a b c : Real) :
  -- Conditions
  a = Real.sqrt 3 →
  b = 1 →
  A = 2 * B →
  -- Triangle properties
  0 < A ∧ A < π →
  0 < B ∧ B < π →
  0 < C ∧ C < π →
  A + B + C = π →
  -- Sine law
  a / Real.sin A = b / Real.sin B →
  b / Real.sin B = c / Real.sin C →
  -- Question/Conclusion
  c = 2 := by
sorry

end triangle_side_length_l496_49649


namespace system_solution_l496_49692

theorem system_solution : ∃ (x y : ℝ), x + y = 0 ∧ 2*x + 3*y = 3 ∧ x = -3 ∧ y = 3 := by
  sorry

end system_solution_l496_49692


namespace three_digit_equation_solution_l496_49697

/-- Represents a three-digit number ABC --/
def threeDigitNumber (A B C : ℕ) : ℕ := 100 * A + 10 * B + C

/-- Represents a two-digit number AB --/
def twoDigitNumber (A B : ℕ) : ℕ := 10 * A + B

/-- Checks if three numbers are distinct digits --/
def areDistinctDigits (A B C : ℕ) : Prop :=
  A ≠ B ∧ B ≠ C ∧ A ≠ C ∧ A ≤ 9 ∧ B ≤ 9 ∧ C ≤ 9 ∧ A ≥ 1 ∧ B ≥ 1 ∧ C ≥ 1

theorem three_digit_equation_solution :
  ∀ A B C : ℕ,
    areDistinctDigits A B C →
    threeDigitNumber A B C = twoDigitNumber A B * C + twoDigitNumber B C * A + twoDigitNumber C A * B →
    ((A = 7 ∧ B = 8 ∧ C = 1) ∨ (A = 5 ∧ B = 1 ∧ C = 7)) := by
  sorry

end three_digit_equation_solution_l496_49697


namespace marys_number_l496_49636

theorem marys_number (n : ℕ) : 
  150 ∣ n → 
  45 ∣ n → 
  1000 ≤ n → 
  n ≤ 3000 → 
  n = 1350 ∨ n = 1800 ∨ n = 2250 ∨ n = 2700 := by
sorry

end marys_number_l496_49636


namespace second_agency_per_mile_charge_l496_49640

theorem second_agency_per_mile_charge : 
  let first_agency_daily_charge : ℝ := 20.25
  let first_agency_per_mile_charge : ℝ := 0.14
  let second_agency_daily_charge : ℝ := 18.25
  let miles_at_equal_cost : ℝ := 25
  let second_agency_per_mile_charge : ℝ := 
    (first_agency_daily_charge + first_agency_per_mile_charge * miles_at_equal_cost - second_agency_daily_charge) / miles_at_equal_cost
  second_agency_per_mile_charge = 0.22 := by
sorry

end second_agency_per_mile_charge_l496_49640


namespace hyperbolas_same_asymptotes_l496_49629

/-- Given two hyperbolas with equations x^2/9 - y^2/16 = 1 and y^2/25 - x^2/M = 1,
    if they have the same asymptotes, then M = 225/16 -/
theorem hyperbolas_same_asymptotes (M : ℝ) :
  (∀ x y : ℝ, x^2/9 - y^2/16 = 1 ↔ y^2/25 - x^2/M = 1) →
  (∀ x y : ℝ, |y| = (4/3) * |x| ↔ |y| = (5/Real.sqrt M) * |x|) →
  M = 225/16 := by
  sorry

end hyperbolas_same_asymptotes_l496_49629


namespace three_fourths_cubed_l496_49608

theorem three_fourths_cubed : (3 / 4 : ℚ) ^ 3 = 27 / 64 := by sorry

end three_fourths_cubed_l496_49608


namespace eulers_formula_3d_l496_49670

/-- A space convex polyhedron -/
structure ConvexPolyhedron where
  faces : ℕ
  vertices : ℕ
  edges : ℕ

/-- Euler's formula for space convex polyhedra -/
theorem eulers_formula_3d (p : ConvexPolyhedron) : p.faces + p.vertices - p.edges = 2 := by
  sorry

end eulers_formula_3d_l496_49670


namespace coefficient_of_x_cubed_l496_49634

theorem coefficient_of_x_cubed (x : ℝ) : 
  let expression := 2*(x^2 - 2*x^3 + x) + 4*(x + 3*x^3 - 2*x^2 + 2*x^5 + 2*x^3) - 3*(2 + x - 5*x^3 - x^2)
  ∃ (a b c d : ℝ), expression = a*x^5 + b*x^4 + 31*x^3 + c*x^2 + d*x + (2 * 1 - 3 * 2) :=
by sorry

#check coefficient_of_x_cubed

end coefficient_of_x_cubed_l496_49634


namespace gcd_cube_plus_three_cubed_l496_49689

theorem gcd_cube_plus_three_cubed (n : ℕ) (h : n > 3) :
  Nat.gcd (n^3 + 3^3) (n + 4) = 1 := by
sorry

end gcd_cube_plus_three_cubed_l496_49689


namespace sophie_uses_one_sheet_per_load_l496_49644

-- Define the given conditions
def loads_per_week : ℕ := 4
def box_cost : ℚ := 5.5
def sheets_per_box : ℕ := 104
def yearly_savings : ℚ := 11

-- Define the function to calculate the number of dryer sheets per load
def dryer_sheets_per_load : ℚ :=
  (yearly_savings / box_cost) * sheets_per_box / (loads_per_week * 52)

-- Theorem statement
theorem sophie_uses_one_sheet_per_load : 
  dryer_sheets_per_load = 1 := by sorry

end sophie_uses_one_sheet_per_load_l496_49644


namespace quadratic_equation_roots_and_k_l496_49607

/-- Given a quadratic equation x^2 + 3x + k = 0 where x = -3 is a root, 
    prove that the other root is 0 and k = 0. -/
theorem quadratic_equation_roots_and_k (k : ℝ) : 
  ((-3 : ℝ)^2 + 3*(-3) + k = 0) → 
  (∃ (r : ℝ), r ≠ -3 ∧ r^2 + 3*r + k = 0 ∧ r = 0) ∧ 
  (k = 0) := by
sorry

end quadratic_equation_roots_and_k_l496_49607


namespace sally_initial_peaches_l496_49606

/-- The number of peaches Sally picked from the orchard -/
def picked_peaches : ℕ := 42

/-- The final number of peaches at Sally's stand -/
def final_peaches : ℕ := 55

/-- The initial number of peaches at Sally's stand -/
def initial_peaches : ℕ := final_peaches - picked_peaches

theorem sally_initial_peaches : initial_peaches = 13 := by
  sorry

end sally_initial_peaches_l496_49606


namespace cubic_root_sum_cubes_l496_49687

theorem cubic_root_sum_cubes (p q r : ℝ) : 
  (p^3 - 2*p^2 + 3*p - 4 = 0) ∧ 
  (q^3 - 2*q^2 + 3*q - 4 = 0) ∧ 
  (r^3 - 2*r^2 + 3*r - 4 = 0) →
  p^3 + q^3 + r^3 = 2 := by
sorry

end cubic_root_sum_cubes_l496_49687


namespace sqrt_sum_equals_2sqrt14_l496_49679

theorem sqrt_sum_equals_2sqrt14 : 
  Real.sqrt (20 - 8 * Real.sqrt 5) + Real.sqrt (20 + 8 * Real.sqrt 5) = 2 * Real.sqrt 14 := by
  sorry

end sqrt_sum_equals_2sqrt14_l496_49679


namespace rectangle_triangle_equal_area_l496_49619

theorem rectangle_triangle_equal_area (b h : ℝ) : 
  b > 0 → 
  h > 0 → 
  h ≤ 2 → 
  b * h = (1/2) * b * (1 - h/2) → 
  h = 2/5 := by
sorry

end rectangle_triangle_equal_area_l496_49619


namespace employee_savings_l496_49660

/-- Calculate the combined savings of three employees after four weeks -/
theorem employee_savings (hourly_wage : ℚ) (hours_per_day : ℕ) (days_per_week : ℕ) (num_weeks : ℕ)
  (robby_save_ratio : ℚ) (jaylen_save_ratio : ℚ) (miranda_save_ratio : ℚ)
  (h1 : hourly_wage = 10)
  (h2 : hours_per_day = 10)
  (h3 : days_per_week = 5)
  (h4 : num_weeks = 4)
  (h5 : robby_save_ratio = 2/5)
  (h6 : jaylen_save_ratio = 3/5)
  (h7 : miranda_save_ratio = 1/2) :
  (hourly_wage * hours_per_day * days_per_week * num_weeks) *
  (robby_save_ratio + jaylen_save_ratio + miranda_save_ratio) = 3000 := by
  sorry


end employee_savings_l496_49660


namespace intersection_exists_l496_49671

-- Define a structure for a point in 3D space
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

-- Define a type for a set of 5 points
def FivePoints := Fin 5 → Point3D

-- Define a predicate for 4 points being non-coplanar
def nonCoplanar (p1 p2 p3 p4 : Point3D) : Prop := sorry

-- Define a predicate for a line intersecting a triangle
def lineIntersectsTriangle (l1 l2 p1 p2 p3 : Point3D) : Prop := sorry

-- Main theorem
theorem intersection_exists (points : FivePoints) 
  (h : ∀ i j k l : Fin 5, i ≠ j → i ≠ k → i ≠ l → j ≠ k → j ≠ l → k ≠ l → 
       nonCoplanar (points i) (points j) (points k) (points l)) :
  ∃ i j k l m : Fin 5, i ≠ j ∧ i ≠ k ∧ i ≠ l ∧ i ≠ m ∧ j ≠ k ∧ j ≠ l ∧ j ≠ m ∧ k ≠ l ∧ k ≠ m ∧ l ≠ m ∧
    lineIntersectsTriangle (points i) (points j) (points k) (points l) (points m) :=
  sorry

end intersection_exists_l496_49671


namespace no_integer_solution_l496_49681

theorem no_integer_solution (P : Polynomial ℤ) (a b c d : ℤ) 
  (h_distinct : a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d)
  (h_values : P.eval a = 5 ∧ P.eval b = 5 ∧ P.eval c = 5 ∧ P.eval d = 5) :
  ¬∃ k : ℤ, P.eval k = 8 := by
  sorry

end no_integer_solution_l496_49681


namespace max_a_for_nonpositive_f_existence_of_m_for_a_eq_1_max_a_equals_one_l496_49620

theorem max_a_for_nonpositive_f (a : ℝ) : 
  (∃ m : ℝ, m > 0 ∧ m^3 - a*m^2 + (a^2 - 2)*m + 1 ≤ 0) → a ≤ 1 :=
by sorry

theorem existence_of_m_for_a_eq_1 : 
  ∃ m : ℝ, m > 0 ∧ m^3 - m^2 + (1^2 - 2)*m + 1 ≤ 0 :=
by sorry

theorem max_a_equals_one : 
  (∃ a : ℝ, (∃ m : ℝ, m > 0 ∧ m^3 - a*m^2 + (a^2 - 2)*m + 1 ≤ 0) ∧ 
    ∀ b : ℝ, (∃ n : ℝ, n > 0 ∧ n^3 - b*n^2 + (b^2 - 2)*n + 1 ≤ 0) → b ≤ a) ∧
  (∃ m : ℝ, m > 0 ∧ m^3 - 1*m^2 + (1^2 - 2)*m + 1 ≤ 0) :=
by sorry

end max_a_for_nonpositive_f_existence_of_m_for_a_eq_1_max_a_equals_one_l496_49620


namespace note_count_l496_49635

theorem note_count (total_amount : ℕ) (denomination_1 : ℕ) (denomination_5 : ℕ) (denomination_10 : ℕ) :
  total_amount = 192 ∧
  denomination_1 = 1 ∧
  denomination_5 = 5 ∧
  denomination_10 = 10 ∧
  (∃ (x : ℕ), x * denomination_1 + x * denomination_5 + x * denomination_10 = total_amount) →
  (∃ (x : ℕ), x * 3 = 36 ∧ x * denomination_1 + x * denomination_5 + x * denomination_10 = total_amount) :=
by sorry

end note_count_l496_49635


namespace problem_solution_l496_49601

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := |a * x + 1|

-- State the theorem
theorem problem_solution :
  -- Part I
  (∃ a : ℝ, ∀ x : ℝ, f a x ≤ 3 ↔ -2 ≤ x ∧ x ≤ 1) ∧
  (∀ a : ℝ, (∀ x : ℝ, f a x ≤ 3 ↔ -2 ≤ x ∧ x ≤ 1) → a = 2) ∧
  -- Part II
  (∀ x : ℝ, |f 2 x - 2 * f 2 (x/2)| ≤ 1) ∧
  (∀ k : ℝ, (∀ x : ℝ, |f 2 x - 2 * f 2 (x/2)| ≤ k) → k ≥ 1) :=
by sorry

end problem_solution_l496_49601


namespace polynomial_division_quotient_l496_49661

theorem polynomial_division_quotient :
  ∀ (x : ℝ),
  ∃ (r : ℝ),
  8 * x^3 + 5 * x^2 - 4 * x - 7 = (x + 3) * (8 * x^2 - 19 * x + 53) + r :=
by
  sorry

end polynomial_division_quotient_l496_49661


namespace olivia_remaining_money_l496_49642

def initial_amount : ℝ := 200
def grocery_cost : ℝ := 65
def shoe_original_price : ℝ := 75
def shoe_discount_rate : ℝ := 0.15
def belt_cost : ℝ := 25

def remaining_money : ℝ :=
  initial_amount - (grocery_cost + (shoe_original_price * (1 - shoe_discount_rate)) + belt_cost)

theorem olivia_remaining_money :
  remaining_money = 46.25 := by sorry

end olivia_remaining_money_l496_49642


namespace gold_coin_problem_l496_49693

theorem gold_coin_problem (c : ℕ+) (h1 : 8 * (c - 1) = 5 * c + 4) : 
  ∃ n : ℕ, n = 24 ∧ 8 * (c - 1) = n ∧ 5 * c + 4 = n :=
sorry

end gold_coin_problem_l496_49693


namespace fabian_accessories_cost_l496_49653

def mouse_cost : ℕ := 16

def keyboard_cost (m : ℕ) : ℕ := 3 * m

def total_cost (m k : ℕ) : ℕ := m + k

theorem fabian_accessories_cost :
  total_cost mouse_cost (keyboard_cost mouse_cost) = 64 := by
  sorry

end fabian_accessories_cost_l496_49653


namespace max_ratio_two_digit_integers_l496_49669

theorem max_ratio_two_digit_integers (x y : ℕ) : 
  x ≥ 10 ∧ x ≤ 99 ∧ y ≥ 10 ∧ y ≤ 99 → -- x and y are two-digit positive integers
  (x + y) / 2 = 65 → -- mean is 65
  x * y = 1950 → -- product is 1950
  ∀ (a b : ℕ), a ≥ 10 ∧ a ≤ 99 ∧ b ≥ 10 ∧ b ≤ 99 ∧ (a + b) / 2 = 65 ∧ a * b = 1950 →
    (a : ℚ) / b ≤ 99 / 31 :=
by sorry

#check max_ratio_two_digit_integers

end max_ratio_two_digit_integers_l496_49669


namespace negative_two_cubed_minus_squared_l496_49654

theorem negative_two_cubed_minus_squared : (-2)^3 - (-2)^2 = -12 := by
  sorry

end negative_two_cubed_minus_squared_l496_49654


namespace johns_new_height_l496_49647

/-- Calculates the new height in feet after a growth spurt -/
def new_height_in_feet (initial_height_inches : ℕ) (growth_rate_inches_per_month : ℕ) (growth_duration_months : ℕ) : ℚ :=
  (initial_height_inches + growth_rate_inches_per_month * growth_duration_months) / 12

/-- Proves that John's new height is 6 feet -/
theorem johns_new_height :
  new_height_in_feet 66 2 3 = 6 := by
  sorry

end johns_new_height_l496_49647


namespace problem_solution_l496_49603

theorem problem_solution : 
  let M : ℚ := 2007 / 3
  let N : ℚ := M / 3
  let X : ℚ := M - N
  X = 446 := by sorry

end problem_solution_l496_49603


namespace equation_proof_l496_49682

theorem equation_proof : 144 + 2 * 12 * 7 + 49 = 361 := by
  sorry

end equation_proof_l496_49682


namespace arthur_walk_distance_l496_49626

/-- Calculates the total distance walked given the number of blocks and distance per block -/
def total_distance (blocks_east blocks_north : ℕ) (miles_per_block : ℚ) : ℚ :=
  (blocks_east + blocks_north : ℚ) * miles_per_block

/-- Proves that walking 8 blocks east and 10 blocks north, with each block being 1/4 mile, results in a total distance of 4.5 miles -/
theorem arthur_walk_distance :
  total_distance 8 10 (1/4) = 4.5 := by
  sorry


end arthur_walk_distance_l496_49626


namespace samsung_start_is_15_l496_49632

/-- The number of Samsung cell phones left at the end of the day -/
def samsung_end : ℕ := 10

/-- The number of iPhones left at the end of the day -/
def iphone_end : ℕ := 5

/-- The number of damaged Samsung cell phones thrown out during the day -/
def samsung_thrown : ℕ := 2

/-- The number of defective iPhones thrown out during the day -/
def iphone_thrown : ℕ := 1

/-- The total number of cell phones sold today -/
def total_sold : ℕ := 4

/-- The number of Samsung cell phones David started the day with -/
def samsung_start : ℕ := samsung_end + samsung_thrown + (total_sold - iphone_thrown)

theorem samsung_start_is_15 : samsung_start = 15 := by
  sorry

end samsung_start_is_15_l496_49632


namespace max_vertex_sum_l496_49639

-- Define a cube with numbered faces
def Cube := Fin 6 → ℕ

-- Define a function to get the sum of three faces at a vertex
def vertexSum (c : Cube) (v : Fin 8) : ℕ :=
  sorry -- Implementation details omitted as per instructions

-- Theorem statement
theorem max_vertex_sum (c : Cube) : 
  (∀ v : Fin 8, vertexSum c v ≤ 14) ∧ (∃ v : Fin 8, vertexSum c v = 14) :=
sorry

end max_vertex_sum_l496_49639


namespace pizza_burger_cost_ratio_l496_49676

/-- The cost ratio of pizza to burger given certain conditions -/
theorem pizza_burger_cost_ratio :
  let burger_cost : ℚ := 9
  let pizza_cost : ℚ → ℚ := λ k => k * burger_cost
  ∀ k : ℚ, pizza_cost k + 3 * burger_cost = 45 →
  pizza_cost k / burger_cost = 2 := by
sorry

end pizza_burger_cost_ratio_l496_49676


namespace preserve_inequality_arithmetic_harmonic_mean_max_product_fixed_sum_inequality_squares_not_always_max_sum_fixed_product_l496_49610

-- Statement A
theorem preserve_inequality (a b c : ℝ) (h : a < b) (k : ℝ) (hk : k > 0) :
  k * a < k * b ∧ a / k < b / k := by sorry

-- Statement B
theorem arithmetic_harmonic_mean (a b : ℝ) (ha : a > 0) (hb : b > 0) (hne : a ≠ b) :
  (a + b) / 2 > 2 * a * b / (a + b) := by sorry

-- Statement C
theorem max_product_fixed_sum (a b : ℝ) (ha : a > 0) (hb : b > 0) (s : ℝ) (hs : s = a + b) :
  a * b ≤ (s / 2) * (s / 2) := by sorry

-- Statement D
theorem inequality_squares (a b : ℝ) (ha : a > 0) (hb : b > 0) (hne : a ≠ b) :
  (1 / 3) * (a^2 + b^2) > ((1 / 3) * (a + b))^2 := by sorry

-- Statement E (incorrect)
theorem not_always_max_sum_fixed_product (P : ℝ → ℝ → Prop) :
  (∃ a b k, a > 0 ∧ b > 0 ∧ a * b = k ∧ a + b > 2 * Real.sqrt k) → 
  ¬(∀ x y, x > 0 → y > 0 → x * y = k → x + y ≤ 2 * Real.sqrt k) := by sorry

end preserve_inequality_arithmetic_harmonic_mean_max_product_fixed_sum_inequality_squares_not_always_max_sum_fixed_product_l496_49610


namespace bead_purchase_cost_l496_49659

/-- Calculate the total cost of bead sets after discounts and taxes --/
theorem bead_purchase_cost (crystal_price metal_price glass_price : ℚ)
  (crystal_sets metal_sets glass_sets : ℕ)
  (crystal_discount metal_tax glass_discount : ℚ) :
  let crystal_cost := crystal_price * crystal_sets * (1 - crystal_discount)
  let metal_cost := metal_price * metal_sets * (1 + metal_tax)
  let glass_cost := glass_price * glass_sets * (1 - glass_discount)
  crystal_cost + metal_cost + glass_cost = 11028 / 100 →
  crystal_price = 12 →
  metal_price = 15 →
  glass_price = 8 →
  crystal_sets = 3 →
  metal_sets = 4 →
  glass_sets = 2 →
  crystal_discount = 1 / 10 →
  metal_tax = 1 / 20 →
  glass_discount = 7 / 100 →
  true := by sorry

end bead_purchase_cost_l496_49659


namespace inequality_solution_l496_49678

def inequality_solution_set : Set ℝ :=
  {x | x < -3 ∨ (-3 < x ∧ x < 3)}

theorem inequality_solution :
  {x : ℝ | (x^2 - 9) / (x + 3) < 0} = inequality_solution_set :=
by sorry

end inequality_solution_l496_49678


namespace inequality_proof_l496_49668

theorem inequality_proof (a b c d : ℝ) 
  (h1 : a + b > |c - d|) 
  (h2 : c + d > |a - b|) : 
  a + c > |b - d| := by
  sorry

end inequality_proof_l496_49668


namespace father_son_walk_l496_49691

/-- The distance traveled when two people with different step lengths walk together -/
def distanceTraveled (fatherStepLength sonStepLength : ℕ) (coincidences : ℕ) : ℕ :=
  let lcm := Nat.lcm fatherStepLength sonStepLength
  (coincidences - 1) * lcm

theorem father_son_walk (fatherStepLength sonStepLength coincidences : ℕ)
  (h1 : fatherStepLength = 80)
  (h2 : sonStepLength = 60)
  (h3 : coincidences = 601) :
  distanceTraveled fatherStepLength sonStepLength coincidences = 144000 := by
  sorry

#eval distanceTraveled 80 60 601

end father_son_walk_l496_49691


namespace quadratic_equations_solutions_l496_49674

theorem quadratic_equations_solutions :
  (∀ x : ℝ, x^2 - 4*x - 12 = 0 ↔ x = 6 ∨ x = -2) ∧
  (∀ x : ℝ, (2*x - 1)^2 = 3*(2*x - 1) ↔ x = 1/2 ∨ x = 2) := by
  sorry

end quadratic_equations_solutions_l496_49674


namespace project_completion_l496_49673

theorem project_completion (a b : ℕ) (ha : a > 0) (hb : b > 0) 
  (h : (1 : ℚ) / a + (1 : ℚ) / b * 4 = 1) : 
  a + b = 9 ∨ a + b = 10 := by
sorry

end project_completion_l496_49673


namespace at_least_one_geq_two_l496_49614

theorem at_least_one_geq_two (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  (a + 1/b ≥ 2) ∨ (b + 1/c ≥ 2) ∨ (c + 1/a ≥ 2) := by
  sorry

end at_least_one_geq_two_l496_49614


namespace matrix_equation_solution_l496_49658

def A : Matrix (Fin 2) (Fin 2) ℝ := !![2, 3; 5, 7]

def B (x y z w : ℝ) : Matrix (Fin 2) (Fin 2) ℝ := !![x, y; z, w]

theorem matrix_equation_solution (x y z w : ℝ) 
  (h1 : A * B x y z w = B x y z w * A)
  (h2 : 2 * z ≠ 5 * y) :
  ∃ x y z w, (x - w) / (z - 2 * y) = 0 :=
by sorry

end matrix_equation_solution_l496_49658


namespace line_l_theorem_circle_M_theorem_l496_49663

-- Define the circle C
def circle_C (x y : ℝ) : Prop := x^2 + y^2 + 2*x - 6*y + 5 = 0

-- Define point P
def point_P : ℝ × ℝ := (-2, -1)

-- Define point Q
def point_Q : ℝ × ℝ := (0, 1)

-- Define the line l
def line_l (x y : ℝ) : Prop := x = -2 ∨ y = (15/8)*x + 11/4

-- Define the circle M
def circle_M (x y : ℝ) : Prop := x^2 + y^2 - 4*x + 6*y - 7 = 0

-- Theorem for line l
theorem line_l_theorem : 
  ∃ (A B : ℝ × ℝ), 
    (∀ (x y : ℝ), line_l x y ↔ (∃ t : ℝ, (x, y) = (1-t) • point_P + t • A ∨ (x, y) = (1-t) • point_P + t • B)) ∧
    circle_C A.1 A.2 ∧ 
    circle_C B.1 B.2 ∧
    (A.1 - B.1)^2 + (A.2 - B.2)^2 = 16 :=
sorry

-- Theorem for circle M
theorem circle_M_theorem :
  (∀ x y : ℝ, circle_M x y → (x = point_P.1 ∧ y = point_P.2) ∨ (x = point_Q.1 ∧ y = point_Q.2)) ∧
  (∃ t : ℝ, ∀ x y : ℝ, circle_M x y → circle_C x y ∨ (x, y) = (1-t) • point_Q + t • point_P) :=
sorry

end line_l_theorem_circle_M_theorem_l496_49663


namespace dogwood_trees_after_planting_l496_49627

/-- The number of dogwood trees in the park after planting is equal to the sum of 
    the initial number of trees and the number of trees planted. -/
theorem dogwood_trees_after_planting (initial_trees planted_trees : ℕ) :
  initial_trees = 34 → planted_trees = 49 → initial_trees + planted_trees = 83 := by
  sorry

end dogwood_trees_after_planting_l496_49627


namespace proportion_fourth_term_l496_49685

theorem proportion_fourth_term (x y : ℝ) : 
  (0.75 : ℝ) / x = 5 / y ∧ x = 1.2 → y = 8 := by
  sorry

end proportion_fourth_term_l496_49685


namespace max_min_difference_f_l496_49625

def f (x : ℝ) : ℝ := x^3 - 3*x + 1

theorem max_min_difference_f : 
  (⨆ x ∈ (Set.Icc (-3) 0), f x) - (⨅ x ∈ (Set.Icc (-3) 0), f x) = 20 := by
  sorry

end max_min_difference_f_l496_49625


namespace ice_cream_line_problem_l496_49624

def Line := Fin 5 → Fin 5

def is_valid_line (l : Line) : Prop :=
  (∀ i j, i ≠ j → l i ≠ l j) ∧
  (∃ i, l i = 0) ∧
  (∃ i, l i = 1) ∧
  (∃ i, l i = 2) ∧
  (∃ i, l i = 3) ∧
  (∃ i, l i = 4)

theorem ice_cream_line_problem (l : Line) 
  (h_valid : is_valid_line l)
  (h_A_first : ∃ i, l i = 0)
  (h_B_next_to_A : ∃ i j, l i = 0 ∧ l j = 1 ∧ (i.val + 1 = j.val ∨ j.val + 1 = i.val))
  (h_C_second_to_last : ∃ i, l i = 3)
  (h_D_last : ∃ i, l i = 4)
  (h_E_remaining : ∃ i, l i = 2) :
  ∃ i j, l i = 2 ∧ l j = 1 ∧ (i.val = j.val + 1 ∨ j.val = i.val + 1) := by
  sorry

end ice_cream_line_problem_l496_49624


namespace nancy_bead_purchase_cost_l496_49662

/-- The total cost of Nancy's purchase given the prices of crystal and metal beads and the quantities she buys. -/
theorem nancy_bead_purchase_cost (crystal_price metal_price : ℕ) (crystal_qty metal_qty : ℕ) : 
  crystal_price = 9 → metal_price = 10 → crystal_qty = 1 → metal_qty = 2 →
  crystal_price * crystal_qty + metal_price * metal_qty = 29 := by
sorry

end nancy_bead_purchase_cost_l496_49662


namespace two_p_plus_q_value_l496_49652

theorem two_p_plus_q_value (p q : ℚ) (h : p / q = 2 / 7) : 2 * p + q = (11 / 2) * p := by
  sorry

end two_p_plus_q_value_l496_49652


namespace pyramid_volume_l496_49611

/-- Given a pyramid with a rhombus base (diagonals d₁ and d₂, where d₁ > d₂) and height passing
    through the vertex of the acute angle of the rhombus, if the area of the diagonal cross-section
    made through the smaller diagonal is Q, then the volume of the pyramid is
    (d₁/12) * √(16Q² - d₁²d₂²). -/
theorem pyramid_volume (d₁ d₂ Q : ℝ) (h₁ : d₁ > d₂) (h₂ : d₂ > 0) (h₃ : Q > 0) :
  let V := d₁ / 12 * Real.sqrt (16 * Q^2 - d₁^2 * d₂^2)
  ∃ (height : ℝ), height > 0 ∧ 
    (V = (1/3) * (1/2 * d₁ * d₂) * height) ∧
    (Q = (1/2) * d₂ * (2 * Q / d₂)) ∧
    (height = Real.sqrt ((2 * Q / d₂)^2 - (d₁ / 2)^2)) :=
by
  sorry

end pyramid_volume_l496_49611


namespace max_value_of_expression_max_value_achievable_l496_49664

theorem max_value_of_expression (x : ℝ) : 
  x^6 / (x^10 + 3*x^8 - 5*x^6 + 10*x^4 + 25) ≤ 1 / (5 + 2 * Real.sqrt 30) :=
sorry

theorem max_value_achievable : 
  ∃ x : ℝ, x^6 / (x^10 + 3*x^8 - 5*x^6 + 10*x^4 + 25) = 1 / (5 + 2 * Real.sqrt 30) :=
sorry

end max_value_of_expression_max_value_achievable_l496_49664


namespace division_theorem_problem_1999_division_l496_49675

theorem division_theorem (n d q r : ℕ) (h : n = d * q + r) (h_r : r < d) :
  (n / d = q ∧ n % d = r) :=
sorry

theorem problem_1999_division :
  1999 / 40 = 49 ∧ 1999 % 40 = 39 :=
sorry

end division_theorem_problem_1999_division_l496_49675


namespace angle_B_value_side_lengths_l496_49609

noncomputable section

-- Define the triangle ABC
variable (A B C : ℝ) -- Angles
variable (a b c : ℝ) -- Side lengths

-- Define the conditions
axiom triangle_condition : 2 * a * c * Real.sin B = Real.sqrt 3 * (a^2 + b^2 - c^2)
axiom side_b : b = 3
axiom angle_relation : Real.sin C = 2 * Real.sin A

-- Theorem 1: Prove that B = π/3
theorem angle_B_value : 
  2 * a * c * Real.sin B = Real.sqrt 3 * (a^2 + b^2 - c^2) → B = π/3 := by sorry

-- Theorem 2: Prove that a = √3 and c = 2√3
theorem side_lengths : 
  b = 3 → 
  Real.sin C = 2 * Real.sin A → 
  2 * a * c * Real.sin B = Real.sqrt 3 * (a^2 + b^2 - c^2) → 
  a = Real.sqrt 3 ∧ c = 2 * Real.sqrt 3 := by sorry

end angle_B_value_side_lengths_l496_49609


namespace f_max_value_l496_49645

/-- The function f(x) = 5x - x^2 -/
def f (x : ℝ) : ℝ := 5 * x - x^2

/-- The maximum value of f(x) is 6.25 -/
theorem f_max_value : ∃ (c : ℝ), ∀ (x : ℝ), f x ≤ c ∧ ∃ (x₀ : ℝ), f x₀ = c :=
  sorry

end f_max_value_l496_49645


namespace odd_digits_181_base4_l496_49648

/-- Converts a natural number from base 10 to base 8 --/
def toBase8 (n : ℕ) : List ℕ :=
  sorry

/-- Converts a natural number from base 8 to base 4 --/
def base8ToBase4 (n : List ℕ) : List ℕ :=
  sorry

/-- Counts the number of odd digits in a list of natural numbers --/
def countOddDigits (n : List ℕ) : ℕ :=
  sorry

/-- Theorem stating that the number of odd digits in the base 4 representation of 181 (base 10),
    when converted through base 8, is equal to 5 --/
theorem odd_digits_181_base4 : 
  countOddDigits (base8ToBase4 (toBase8 181)) = 5 :=
by sorry

end odd_digits_181_base4_l496_49648


namespace rational_function_property_l496_49615

theorem rational_function_property (f : ℚ → ℚ) 
  (h : ∀ x y : ℚ, f (x + y) = 2 * f (x / 2) + 3 * f (y / 3)) :
  ∃ a : ℚ, ∀ x : ℚ, f x = a * x :=
by sorry

end rational_function_property_l496_49615


namespace quarter_capacity_at_6_l496_49617

/-- Represents the volume of water in the pool as a fraction of its full capacity -/
def PoolVolume := Fin 9 → ℚ

/-- The pool's volume doubles every hour -/
def doubles (v : PoolVolume) : Prop :=
  ∀ i : Fin 8, v (i + 1) = 2 * v i

/-- The pool is full after 8 hours -/
def full_at_8 (v : PoolVolume) : Prop :=
  v 8 = 1

/-- The main theorem: If the pool's volume doubles every hour and is full after 8 hours,
    then it was at one quarter capacity after 6 hours -/
theorem quarter_capacity_at_6 (v : PoolVolume) 
  (h1 : doubles v) (h2 : full_at_8 v) : v 6 = 1/4 := by
  sorry

end quarter_capacity_at_6_l496_49617


namespace expression_simplification_l496_49604

theorem expression_simplification (x y z w : ℝ) :
  (x - (y - (z - w))) - ((x - y) - (z - w)) = 2*z - 2*w := by
  sorry

end expression_simplification_l496_49604


namespace jills_salary_l496_49667

/-- Represents a person's monthly finances -/
structure MonthlyFinances where
  netSalary : ℝ
  discretionaryIncome : ℝ
  giftAmount : ℝ

/-- Conditions for Jill's monthly finances -/
def jillsFinances (f : MonthlyFinances) : Prop :=
  f.discretionaryIncome = f.netSalary / 5 ∧
  f.giftAmount = f.discretionaryIncome * 0.2 ∧
  f.giftAmount = 111

/-- Theorem: If Jill's finances meet the given conditions, her net monthly salary is $2775 -/
theorem jills_salary (f : MonthlyFinances) (h : jillsFinances f) : f.netSalary = 2775 := by
  sorry

end jills_salary_l496_49667


namespace imaginary_part_of_complex_fraction_l496_49600

theorem imaginary_part_of_complex_fraction : 
  let z : ℂ := ((Complex.I - 1)^2 + 4) / (Complex.I + 1)
  Complex.im z = -3 := by sorry

end imaginary_part_of_complex_fraction_l496_49600


namespace total_markers_l496_49612

theorem total_markers (red_markers blue_markers : ℕ) :
  red_markers = 41 → blue_markers = 64 → red_markers + blue_markers = 105 :=
by
  sorry

end total_markers_l496_49612


namespace boxes_in_case_l496_49666

/-- Given that George has 12 blocks in total, each box holds 6 blocks,
    and George has 2 boxes of blocks, prove that there are 2 boxes in a case. -/
theorem boxes_in_case (total_blocks : ℕ) (blocks_per_box : ℕ) (boxes_of_blocks : ℕ) : 
  total_blocks = 12 → blocks_per_box = 6 → boxes_of_blocks = 2 → 
  (total_blocks / blocks_per_box : ℕ) = boxes_of_blocks := by
  sorry

#check boxes_in_case

end boxes_in_case_l496_49666


namespace ella_seventh_test_score_l496_49672

def is_valid_score_set (scores : List ℤ) : Prop :=
  scores.length = 8 ∧
  scores.all (λ s => 88 ≤ s ∧ s ≤ 97) ∧
  scores.Nodup ∧
  (∀ k : ℕ, 1 ≤ k ∧ k ≤ 8 → (scores.take k).sum % k = 0) ∧
  scores.get! 7 = 90

theorem ella_seventh_test_score (scores : List ℤ) :
  is_valid_score_set scores → scores.get! 6 = 95 := by
  sorry

#check ella_seventh_test_score

end ella_seventh_test_score_l496_49672


namespace stock_discount_calculation_l496_49683

/-- Calculates the discount on a stock given its original price, brokerage fee, and final cost price. -/
theorem stock_discount_calculation (original_price brokerage_rate final_cost_price : ℝ) : 
  original_price = 100 →
  brokerage_rate = 1 / 500 →
  final_cost_price = 95.2 →
  ∃ (discount : ℝ), 
    (original_price - discount) * (1 + brokerage_rate) = final_cost_price ∧
    abs (discount - 4.99) < 0.01 := by
  sorry

end stock_discount_calculation_l496_49683


namespace angle_rewrite_and_terminal_sides_l496_49618

theorem angle_rewrite_and_terminal_sides (α : Real) (h : α = 1200 * π / 180) :
  ∃ (β k : Real),
    α = β + 2 * k * π ∧
    0 ≤ β ∧ β < 2 * π ∧
    β = 2 * π / 3 ∧
    k = 3 ∧
    (2 * π / 3 ∈ Set.Icc (-2 * π) (2 * π)) ∧
    (-4 * π / 3 ∈ Set.Icc (-2 * π) (2 * π)) ∧
    ∃ (m n : ℤ),
      2 * π / 3 = α + 2 * m * π ∧
      -4 * π / 3 = α + 2 * n * π :=
by sorry

end angle_rewrite_and_terminal_sides_l496_49618


namespace mike_pens_l496_49613

theorem mike_pens (initial_pens : ℕ) (sharon_pens : ℕ) (final_pens : ℕ) :
  initial_pens = 7 →
  sharon_pens = 19 →
  final_pens = 39 →
  ∃ M : ℕ, 2 * (initial_pens + M) - sharon_pens = final_pens ∧ M = 22 :=
by sorry

end mike_pens_l496_49613


namespace mike_shortfall_l496_49638

def max_marks : ℕ := 780
def mike_score : ℕ := 212
def passing_percentage : ℚ := 30 / 100

theorem mike_shortfall :
  (↑max_marks * passing_percentage).floor - mike_score = 22 := by
  sorry

end mike_shortfall_l496_49638


namespace magic_square_x_value_l496_49699

/-- Represents a 3x3 magic square -/
structure MagicSquare where
  a : ℕ
  b : ℕ
  c : ℕ
  d : ℕ
  e : ℕ
  f : ℕ
  g : ℕ
  h : ℕ
  i : ℕ

/-- The sum of each row, column, and diagonal in a magic square -/
def magicSum (s : MagicSquare) : ℕ := s.a + s.b + s.c

/-- Predicate for a valid magic square -/
def isMagicSquare (s : MagicSquare) : Prop :=
  -- All rows have the same sum
  s.a + s.b + s.c = magicSum s ∧
  s.d + s.e + s.f = magicSum s ∧
  s.g + s.h + s.i = magicSum s ∧
  -- All columns have the same sum
  s.a + s.d + s.g = magicSum s ∧
  s.b + s.e + s.h = magicSum s ∧
  s.c + s.f + s.i = magicSum s ∧
  -- Both diagonals have the same sum
  s.a + s.e + s.i = magicSum s ∧
  s.c + s.e + s.g = magicSum s

theorem magic_square_x_value (s : MagicSquare) 
  (h1 : isMagicSquare s)
  (h2 : s.b = 19 ∧ s.e = 15 ∧ s.h = 11)  -- Second column condition
  (h3 : s.b = 19 ∧ s.c = 14)  -- First row condition
  (h4 : s.e = 15 ∧ s.i = 12)  -- Diagonal condition
  : s.g = 18 := by
  sorry

end magic_square_x_value_l496_49699


namespace max_sum_smallest_angles_l496_49680

/-- A line in a plane --/
structure Line where
  -- We don't need to define the structure of a line for this statement

/-- Represents the configuration of lines in a plane --/
structure LineConfiguration where
  lines : Finset Line
  general_position : Bool

/-- Calculates the sum of smallest angles at intersections --/
def sum_smallest_angles (config : LineConfiguration) : ℝ :=
  sorry

/-- The theorem statement --/
theorem max_sum_smallest_angles :
  ∀ (config : LineConfiguration),
    config.lines.card = 10 ∧ config.general_position →
    ∃ (max_sum : ℝ), 
      (∀ (c : LineConfiguration), c.lines.card = 10 ∧ c.general_position → 
        sum_smallest_angles c ≤ max_sum) ∧
      max_sum = 2250 := by
  sorry

end max_sum_smallest_angles_l496_49680


namespace even_square_operation_l496_49657

theorem even_square_operation (x : ℕ) (h : x > 0) : ∃ k : ℕ, (2 * x)^2 = 2 * k := by
  sorry

end even_square_operation_l496_49657


namespace football_cost_l496_49623

theorem football_cost (total_spent marbles_cost baseball_cost : ℚ)
  (h1 : total_spent = 20.52)
  (h2 : marbles_cost = 9.05)
  (h3 : baseball_cost = 6.52) :
  total_spent - marbles_cost - baseball_cost = 4.95 := by
  sorry

end football_cost_l496_49623


namespace diophantine_equation_solution_l496_49694

theorem diophantine_equation_solution (n m : ℤ) : 
  n^4 + 2*n^3 + 2*n^2 + 2*n + 1 = m^2 ↔ (n = 0 ∧ (m = 1 ∨ m = -1)) ∨ (n = -1 ∧ m = 0) :=
by sorry

end diophantine_equation_solution_l496_49694


namespace ghost_castle_paths_l496_49616

theorem ghost_castle_paths (n : ℕ) (h : n = 8) : n * (n - 1) = 56 := by
  sorry

end ghost_castle_paths_l496_49616


namespace lines_no_common_points_implies_a_equals_negative_one_l496_49621

/-- Two lines in the plane -/
structure TwoLines where
  a : ℝ
  line1 : ℝ → ℝ := λ x => (a^2 - a) * x + 1 - a
  line2 : ℝ → ℝ := λ x => 2 * x - 1

/-- The property that two lines have no points in common -/
def NoCommonPoints (l : TwoLines) : Prop :=
  ∀ x : ℝ, l.line1 x ≠ l.line2 x

/-- The theorem statement -/
theorem lines_no_common_points_implies_a_equals_negative_one (l : TwoLines) :
  NoCommonPoints l → l.a = -1 := by
  sorry

end lines_no_common_points_implies_a_equals_negative_one_l496_49621


namespace equation_solutions_l496_49602

theorem equation_solutions : 
  ∀ x : ℝ, (4 * (3 * x)^2 + 3 * x + 5 = 3 * (8 * x^2 + 3 * x + 3)) ↔ 
  (x = (1 + Real.sqrt 19) / 4 ∨ x = (1 - Real.sqrt 19) / 4) :=
by sorry

end equation_solutions_l496_49602


namespace quadratic_properties_l496_49690

def f (b c x : ℝ) : ℝ := x^2 + b*x + c

theorem quadratic_properties (b c : ℝ) 
  (h1 : f b c 1 = 0) 
  (h2 : f b c 3 = 0) : 
  (f b c (-1) = 8) ∧ 
  (∀ x ∈ Set.Icc 2 4, f b c x ≤ 3) ∧
  (∃ x ∈ Set.Icc 2 4, f b c x = 3) ∧
  (∀ x ∈ Set.Icc 2 4, -1 ≤ f b c x) ∧
  (∃ x ∈ Set.Icc 2 4, f b c x = -1) ∧
  (∀ x y, 2 ≤ x ∧ x ≤ y → f b c x ≤ f b c y) :=
by sorry

end quadratic_properties_l496_49690


namespace k_value_l496_49622

theorem k_value (x : ℝ) (h1 : x ≠ 0) (h2 : 24 / x = k) : k = 24 / x := by
  sorry

end k_value_l496_49622


namespace triangle_abc_properties_l496_49698

theorem triangle_abc_properties (A B C : Real) (a b c : Real) :
  B = π / 3 → a = Real.sqrt 2 →
  (b = Real.sqrt 3 → A = π / 4) ∧
  (1 / 2 * a * c * Real.sin B = 3 * Real.sqrt 3 / 2 → b = Real.sqrt 14) := by
  sorry

end triangle_abc_properties_l496_49698


namespace mike_seashells_l496_49655

/-- The number of seashells Mike found -/
def total_seashells (unbroken_seashells broken_seashells : ℕ) : ℕ :=
  unbroken_seashells + broken_seashells

/-- Theorem stating that Mike found 6 seashells in total -/
theorem mike_seashells : total_seashells 2 4 = 6 := by
  sorry

end mike_seashells_l496_49655


namespace b_hire_charges_l496_49696

/-- Calculates the hire charges for a specific person given the total cost,
    and the hours used by each person. -/
def hireCharges (totalCost : ℚ) (hoursA hoursB hoursC : ℚ) : ℚ :=
  let totalHours := hoursA + hoursB + hoursC
  let costPerHour := totalCost / totalHours
  costPerHour * hoursB

theorem b_hire_charges :
  hireCharges 720 9 10 13 = 225 := by
  sorry

end b_hire_charges_l496_49696


namespace lg_ratio_theorem_l496_49686

theorem lg_ratio_theorem (m n : ℝ) (hm : Real.log 2 = m) (hn : Real.log 3 = n) :
  (Real.log 12) / (Real.log 15) = (2*m + n) / (1 - m + n) := by
  sorry

end lg_ratio_theorem_l496_49686


namespace dice_hidden_sum_l496_49651

/-- The sum of numbers on a single die -/
def die_sum : ℕ := 21

/-- The number of dice -/
def num_dice : ℕ := 4

/-- The sum of visible numbers -/
def visible_sum : ℕ := 1 + 2 + 3 + 4 + 4 + 5 + 5 + 6

/-- The number of visible faces -/
def num_visible : ℕ := 8

theorem dice_hidden_sum :
  (num_dice * die_sum) - visible_sum = 54 :=
by sorry

end dice_hidden_sum_l496_49651


namespace rhombus_properties_l496_49656

-- Define the rhombus ABCD
def Rhombus (A B C D : ℝ × ℝ) : Prop :=
  let dist := λ p q : ℝ × ℝ => Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)
  dist A B = 4 ∧ dist B C = 4 ∧ dist C D = 4 ∧ dist D A = 4

-- Define the origin O
def O : ℝ × ℝ := (0, 0)

-- Define the condition for point A on the semicircle
def OnSemicircle (A : ℝ × ℝ) : Prop :=
  (A.1 - 2)^2 + A.2^2 = 4 ∧ 2 ≤ A.1 ∧ A.1 ≤ 4

-- Main theorem
theorem rhombus_properties
  (A B C D : ℝ × ℝ)
  (h_rhombus : Rhombus A B C D)
  (h_OB : dist O B = 6)
  (h_OD : dist O D = 6)
  (h_A_semicircle : OnSemicircle A) :
  (∃ k, dist O A * dist O B = k) ∧
  (∃ y, -5 ≤ y ∧ y ≤ 5 ∧ C = (5, y)) :=
sorry

#check rhombus_properties

end rhombus_properties_l496_49656


namespace prime_squared_minus_one_divisible_by_24_l496_49631

theorem prime_squared_minus_one_divisible_by_24 (n : ℕ) 
  (h_prime : Nat.Prime n) (h_gt_3 : n > 3) : 
  24 ∣ (n^2 - 1) := by
  sorry

end prime_squared_minus_one_divisible_by_24_l496_49631


namespace monotonic_f_implies_a_range_l496_49695

theorem monotonic_f_implies_a_range (a : ℝ) :
  (∀ x : ℝ, StrictMono (fun x => x - (1/3) * Real.sin (2*x) + a * Real.sin x)) →
  -1/3 ≤ a ∧ a ≤ 1/3 := by
sorry

end monotonic_f_implies_a_range_l496_49695


namespace f_definition_f_2019_l496_49637

def a_n (n : ℕ) : ℕ := Nat.sqrt n

def b_n (n : ℕ) : ℕ := n - (a_n n)^2

def f (n : ℕ) : ℕ :=
  if b_n n ≤ a_n n then
    (a_n n)^2 + 1
  else if a_n n < b_n n ∧ b_n n ≤ 2 * (a_n n) + 1 then
    (a_n n)^2 + a_n n + 1
  else
    0  -- This case should never occur based on the problem definition

theorem f_definition (n : ℕ) :
  f n = if b_n n ≤ a_n n then
          (a_n n)^2 + 1
        else
          (a_n n)^2 + a_n n + 1 :=
by sorry

theorem f_2019 : f 2019 = 1981 :=
by sorry

end f_definition_f_2019_l496_49637


namespace not_both_rational_l496_49643

theorem not_both_rational (x : ℝ) : ¬(∃ (a b : ℚ), (x + Real.sqrt 3 : ℝ) = a ∧ (x^3 + 5 * Real.sqrt 3 : ℝ) = b) :=
sorry

end not_both_rational_l496_49643
