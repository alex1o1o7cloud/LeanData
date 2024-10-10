import Mathlib

namespace unique_intersection_l1615_161567

/-- The value of b for which the graphs of y = bx^2 + 5x + 3 and y = -2x - 3 intersect at exactly one point -/
def b : ℚ := 49 / 24

/-- The first function: f(x) = bx^2 + 5x + 3 -/
def f (x : ℝ) : ℝ := b * x^2 + 5 * x + 3

/-- The second function: g(x) = -2x - 3 -/
def g (x : ℝ) : ℝ := -2 * x - 3

/-- Theorem stating that the graphs of f and g intersect at exactly one point -/
theorem unique_intersection : ∃! x : ℝ, f x = g x := by sorry

end unique_intersection_l1615_161567


namespace initial_doctors_count_l1615_161530

theorem initial_doctors_count (initial_nurses : ℕ) (remaining_staff : ℕ) : initial_nurses = 18 → remaining_staff = 22 → ∃ initial_doctors : ℕ, initial_doctors = 11 ∧ initial_doctors + initial_nurses - 5 - 2 = remaining_staff :=
by
  sorry

end initial_doctors_count_l1615_161530


namespace second_divisor_problem_l1615_161552

theorem second_divisor_problem (x : ℚ) : 
  (((377 / 13) / x) * (1 / 4)) / 2 = 0.125 → x = 29 := by
  sorry

end second_divisor_problem_l1615_161552


namespace goods_train_speed_l1615_161564

/-- Proves that the speed of a goods train is 100 km/h given specific conditions --/
theorem goods_train_speed (man_train_speed : ℝ) (passing_time : ℝ) (goods_train_length : ℝ) :
  man_train_speed = 80 →
  passing_time = 8 →
  goods_train_length = 400 →
  ∃ (goods_train_speed : ℝ),
    goods_train_speed = 100 ∧
    (goods_train_speed + man_train_speed) * (5 / 18) * passing_time = goods_train_length :=
by sorry

end goods_train_speed_l1615_161564


namespace complex_fraction_simplification_l1615_161544

theorem complex_fraction_simplification :
  (1 - 2*Complex.I) / (2 + Complex.I) = -Complex.I :=
by sorry

end complex_fraction_simplification_l1615_161544


namespace brookes_science_problems_l1615_161512

/-- Represents the number of problems and time for each subject in Brooke's homework --/
structure Homework where
  math_problems : ℕ
  social_studies_problems : ℕ
  science_problems : ℕ
  math_time_per_problem : ℚ
  social_studies_time_per_problem : ℚ
  science_time_per_problem : ℚ
  total_time : ℚ

/-- Calculates the total time spent on homework --/
def total_homework_time (hw : Homework) : ℚ :=
  hw.math_problems * hw.math_time_per_problem +
  hw.social_studies_problems * hw.social_studies_time_per_problem +
  hw.science_problems * hw.science_time_per_problem

/-- Theorem stating that Brooke has 10 science problems --/
theorem brookes_science_problems (hw : Homework)
  (h1 : hw.math_problems = 15)
  (h2 : hw.social_studies_problems = 6)
  (h3 : hw.math_time_per_problem = 2)
  (h4 : hw.social_studies_time_per_problem = 1/2)
  (h5 : hw.science_time_per_problem = 3/2)
  (h6 : hw.total_time = 48)
  (h7 : total_homework_time hw = hw.total_time) :
  hw.science_problems = 10 := by
  sorry


end brookes_science_problems_l1615_161512


namespace exists_digit_sum_div_11_l1615_161575

def digit_sum (n : ℕ) : ℕ := sorry

theorem exists_digit_sum_div_11 (n : ℕ) : ∃ k, n ≤ k ∧ k < n + 39 ∧ (digit_sum k) % 11 = 0 := by
  sorry

end exists_digit_sum_div_11_l1615_161575


namespace triangle_inequality_l1615_161562

theorem triangle_inequality (x y z : ℝ) : 
  x > 0 → y > 0 → z > 0 → 
  Real.sqrt x + Real.sqrt y > Real.sqrt z →
  Real.sqrt y + Real.sqrt z > Real.sqrt x →
  Real.sqrt z + Real.sqrt x > Real.sqrt y →
  x / y + y / z + z / x = 5 →
  x * (y^2 - 2*z^2) / z + y * (z^2 - 2*x^2) / x + z * (x^2 - 2*y^2) / y ≥ 0 := by
sorry

end triangle_inequality_l1615_161562


namespace total_selections_l1615_161578

/-- Represents a hexagonal arrangement of circles -/
structure HexCircleArrangement :=
  (total_circles : ℕ)
  (side_length : ℕ)

/-- Calculates the number of ways to select three consecutive circles in one direction -/
def consecutive_selections (n : ℕ) : ℕ :=
  if n < 3 then 0 else n - 2

/-- Calculates the number of ways to select three consecutive circles in a diagonal direction -/
def diagonal_selections (n : ℕ) : ℕ :=
  if n < 3 then 0 else (n - 2) * (n - 1) / 2

/-- The main theorem stating the total number of ways to select three consecutive circles -/
theorem total_selections (h : HexCircleArrangement) 
  (h_total : h.total_circles = 33) 
  (h_side : h.side_length = 7) : 
  consecutive_selections h.side_length + 2 * diagonal_selections h.side_length = 57 := by
  sorry


end total_selections_l1615_161578


namespace sqrt_81_equals_9_l1615_161548

theorem sqrt_81_equals_9 : Real.sqrt 81 = 9 := by
  sorry

end sqrt_81_equals_9_l1615_161548


namespace sin_600_degrees_l1615_161546

theorem sin_600_degrees : Real.sin (600 * Real.pi / 180) = -Real.sqrt 3 / 2 := by
  sorry

end sin_600_degrees_l1615_161546


namespace floor_sqrt_120_l1615_161556

theorem floor_sqrt_120 : ⌊Real.sqrt 120⌋ = 10 := by sorry

end floor_sqrt_120_l1615_161556


namespace power_function_quadrants_l1615_161571

-- Define a power function
def isPowerFunction (f : ℝ → ℝ) : Prop :=
  ∃ a : ℝ, ∀ x : ℝ, f x = x ^ a

-- Define the condition f(1/3) = 9
def satisfiesCondition (f : ℝ → ℝ) : Prop :=
  f (1/3) = 9

-- Define the property of being in first and second quadrants
def isInFirstAndSecondQuadrants (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, x > 0 → f x > 0

-- Theorem statement
theorem power_function_quadrants (f : ℝ → ℝ) 
  (h1 : isPowerFunction f) 
  (h2 : satisfiesCondition f) : 
  isInFirstAndSecondQuadrants f :=
sorry

end power_function_quadrants_l1615_161571


namespace purely_imaginary_complex_l1615_161591

theorem purely_imaginary_complex (m : ℝ) : 
  (Complex.mk (m^2 - m) m).im ≠ 0 ∧ (Complex.mk (m^2 - m) m).re = 0 → m = 1 := by
  sorry

end purely_imaginary_complex_l1615_161591


namespace ellipse_isosceles_triangle_existence_l1615_161505

/-- Ellipse C with equation x²/9 + y²/8 = 1 -/
def ellipse_C (x y : ℝ) : Prop := x^2/9 + y^2/8 = 1

/-- Line l passing through point P(0, 2) with slope k -/
def line_l (k x y : ℝ) : Prop := y = k*x + 2

/-- Point D on the x-axis -/
def point_D (m : ℝ) : Prop := ∃ y, y = 0

/-- Isosceles triangle condition -/
def isosceles_triangle (xA yA xB yB xD : ℝ) : Prop :=
  (xA - xD)^2 + yA^2 = (xB - xD)^2 + yB^2

theorem ellipse_isosceles_triangle_existence :
  ∀ k > 0,
  ∃ xA yA xB yB m,
    ellipse_C xA yA ∧
    ellipse_C xB yB ∧
    line_l k xA yA ∧
    line_l k xB yB ∧
    point_D m ∧
    isosceles_triangle xA yA xB yB m ∧
    -Real.sqrt 2 / 12 ≤ m ∧
    m < 0 :=
sorry

end ellipse_isosceles_triangle_existence_l1615_161505


namespace tangent_line_to_cubic_curve_l1615_161572

theorem tangent_line_to_cubic_curve (k : ℝ) : 
  (∃ x y : ℝ, y = x^3 ∧ y = k*x + 2 ∧ (3 * x^2 = k)) → k = 3 := by
  sorry

end tangent_line_to_cubic_curve_l1615_161572


namespace expression_value_at_three_l1615_161525

theorem expression_value_at_three :
  let x : ℝ := 3
  let expr := (Real.sqrt (x - 2 * Real.sqrt 2) / Real.sqrt (x^2 - 4 * Real.sqrt 2 * x + 8)) -
              (Real.sqrt (x + 2 * Real.sqrt 2) / Real.sqrt (x^2 + 4 * Real.sqrt 2 * x + 8))
  expr = 2 := by
sorry

end expression_value_at_three_l1615_161525


namespace intersection_polar_coords_l1615_161566

-- Define the curves
def C₁ (x y : ℝ) : Prop := x^2 + y^2 = 2
def C₂ (t x y : ℝ) : Prop := x = 2 - t ∧ y = t

-- Define the intersection point
def intersection_point (x y : ℝ) : Prop :=
  C₁ x y ∧ ∃ t, C₂ t x y

-- Define polar coordinates
def polar_coords (x y ρ θ : ℝ) : Prop :=
  x = ρ * Real.cos θ ∧ y = ρ * Real.sin θ

-- Theorem statement
theorem intersection_polar_coords :
  ∃ x y : ℝ, intersection_point x y ∧ 
  polar_coords x y (Real.sqrt 2) (Real.pi / 4) :=
sorry

end intersection_polar_coords_l1615_161566


namespace largest_five_digit_negative_congruent_to_one_mod_23_l1615_161517

theorem largest_five_digit_negative_congruent_to_one_mod_23 :
  ∀ n : ℤ, -99999 ≤ n ∧ n < -9999 ∧ n ≡ 1 [ZMOD 23] → n ≤ -9993 :=
by sorry

end largest_five_digit_negative_congruent_to_one_mod_23_l1615_161517


namespace prob_two_non_defective_pens_l1615_161594

/-- Given a box of 16 pens with 3 defective pens, prove that the probability
    of selecting 2 non-defective pens at random is 13/20. -/
theorem prob_two_non_defective_pens :
  let total_pens : ℕ := 16
  let defective_pens : ℕ := 3
  let non_defective_pens : ℕ := total_pens - defective_pens
  let prob_first_non_defective : ℚ := non_defective_pens / total_pens
  let prob_second_non_defective : ℚ := (non_defective_pens - 1) / (total_pens - 1)
  prob_first_non_defective * prob_second_non_defective = 13 / 20 := by
  sorry


end prob_two_non_defective_pens_l1615_161594


namespace min_number_for_triangle_l1615_161542

/-- A function that checks if three numbers can form a triangle -/
def can_form_triangle (a b c : ℕ) : Prop :=
  a + b > c ∧ b + c > a ∧ c + a > b

/-- The property that any 17 numbers chosen from 1 to 2005 always contain a triangle -/
def always_contains_triangle (n : ℕ) : Prop :=
  ∀ (s : Finset ℕ), s.card = n → (∀ x ∈ s, 1 ≤ x ∧ x ≤ 2005) →
    ∃ a b c, a ∈ s ∧ b ∈ s ∧ c ∈ s ∧ a ≠ b ∧ b ≠ c ∧ c ≠ a ∧ can_form_triangle a b c

/-- The theorem stating that 17 is the minimum number for which the property holds -/
theorem min_number_for_triangle :
  always_contains_triangle 17 ∧ ¬(always_contains_triangle 16) :=
sorry

end min_number_for_triangle_l1615_161542


namespace max_value_of_roots_squared_sum_l1615_161577

theorem max_value_of_roots_squared_sum (k : ℝ) (x₁ x₂ : ℝ) : 
  x₁^2 - (k-2)*x₁ + (k^2 + 3*k + 5) = 0 →
  x₂^2 - (k-2)*x₂ + (k^2 + 3*k + 5) = 0 →
  x₁ ≠ x₂ →
  ∃ (max : ℝ), max = 18 ∧ x₁^2 + x₂^2 ≤ max :=
by sorry

end max_value_of_roots_squared_sum_l1615_161577


namespace distance_calculation_l1615_161551

/-- The distance between Maxwell's and Brad's homes -/
def distance_between_homes : ℝ := 74

/-- Maxwell's walking speed in km/h -/
def maxwell_speed : ℝ := 4

/-- Brad's running speed in km/h -/
def brad_speed : ℝ := 6

/-- Time difference between Maxwell's start and Brad's start in hours -/
def time_difference : ℝ := 1

/-- Total time until Maxwell and Brad meet in hours -/
def total_time : ℝ := 8

theorem distance_calculation :
  distance_between_homes = 
    maxwell_speed * total_time + 
    brad_speed * (total_time - time_difference) :=
by sorry

end distance_calculation_l1615_161551


namespace triangle_formation_range_l1615_161539

theorem triangle_formation_range (x : ℝ) : 
  let circle := {p : ℝ × ℝ | p.1^2 + p.2^2 = 1}
  let A : ℝ × ℝ := (-1, 0)
  let B : ℝ × ℝ := (1, 0)
  let D : ℝ × ℝ := (x, 0)
  let C : ℝ × ℝ := (x, Real.sqrt (1 - x^2))
  let AD := Real.sqrt ((D.1 - A.1)^2 + (D.2 - A.2)^2)
  let BD := Real.sqrt ((D.1 - B.1)^2 + (D.2 - B.2)^2)
  let CD := Real.sqrt ((C.1 - D.1)^2 + (C.2 - D.2)^2)
  (AD + BD > CD ∧ AD + CD > BD ∧ BD + CD > AD) ↔ 
  (x > 2 - Real.sqrt 5 ∧ x < Real.sqrt 5 - 2) :=
by sorry

end triangle_formation_range_l1615_161539


namespace rectangle_area_l1615_161589

theorem rectangle_area (r : ℝ) (ratio : ℝ) : 
  r = 3 → ratio = 3 → 2 * r * (ratio + 1) = 108 := by
  sorry

end rectangle_area_l1615_161589


namespace base_10_to_12_256_l1615_161531

/-- Converts a base-10 number to its base-12 representation -/
def toBase12 (n : ℕ) : List ℕ :=
  sorry

/-- Converts a list of digits in base-12 to a natural number -/
def fromBase12 (digits : List ℕ) : ℕ :=
  sorry

theorem base_10_to_12_256 :
  toBase12 256 = [1, 9, 4] ∧ fromBase12 [1, 9, 4] = 256 := by
  sorry

end base_10_to_12_256_l1615_161531


namespace quadratic_polynomial_divisibility_l1615_161518

theorem quadratic_polynomial_divisibility (p : ℕ) (a b c : ℕ) (h_prime : Nat.Prime p)
  (h_a : 0 < a ∧ a ≤ p) (h_b : 0 < b ∧ b ≤ p) (h_c : 0 < c ∧ c ≤ p)
  (h_divisible : ∀ x : ℕ, x > 0 → p ∣ (a * x^2 + b * x + c)) :
  (p = 2 ∧ a + b + c = 4) ∨ (p > 2 ∧ a + b + c = 3 * p) := by
sorry

end quadratic_polynomial_divisibility_l1615_161518


namespace honey_harvest_increase_l1615_161543

theorem honey_harvest_increase (last_year harvest_this_year increase : ℕ) : 
  last_year = 2479 → 
  harvest_this_year = 8564 → 
  increase = harvest_this_year - last_year → 
  increase = 6085 := by
  sorry

end honey_harvest_increase_l1615_161543


namespace xin_xin_family_stay_l1615_161527

/-- Represents a date with a month and day -/
structure Date where
  month : Nat
  day : Nat

/-- Calculates the number of nights between two dates -/
def nights_between (arrival : Date) (departure : Date) : Nat :=
  sorry

theorem xin_xin_family_stay :
  let arrival : Date := ⟨5, 30⟩  -- May 30
  let departure : Date := ⟨6, 4⟩  -- June 4
  nights_between arrival departure = 5 := by
  sorry

end xin_xin_family_stay_l1615_161527


namespace total_presents_l1615_161522

theorem total_presents (christmas : ℕ) (easter : ℕ) (birthday : ℕ) (halloween : ℕ) : 
  christmas = 60 →
  birthday = 3 * easter →
  easter = christmas / 2 - 10 →
  halloween = birthday - easter →
  christmas + easter + birthday + halloween = 180 := by
sorry

end total_presents_l1615_161522


namespace function_decomposition_l1615_161532

theorem function_decomposition (f : ℝ → ℝ) :
  ∃ (f_even f_odd : ℝ → ℝ),
    (∀ x, f x = f_even x + f_odd x) ∧
    (∀ x, f_even (-x) = f_even x) ∧
    (∀ x, f_odd (-x) = -f_odd x) :=
by
  sorry

end function_decomposition_l1615_161532


namespace problem_solution_l1615_161590

-- Define the ⊕ operation
def circleplus (a b : ℤ) : ℤ := (a + b) * (a - b)

-- State the theorem
theorem problem_solution :
  (circleplus 7 4 - 12) * 5 = 105 := by
  sorry

end problem_solution_l1615_161590


namespace cost_of_pens_l1615_161514

/-- Given a pack of 150 pens costs $45, prove that the cost of 3600 pens is $1080 -/
theorem cost_of_pens (pack_size : ℕ) (pack_cost : ℝ) (total_pens : ℕ) :
  pack_size = 150 →
  pack_cost = 45 →
  total_pens = 3600 →
  (total_pens : ℝ) * (pack_cost / pack_size) = 1080 := by
sorry

end cost_of_pens_l1615_161514


namespace min_value_3a_plus_2_l1615_161568

theorem min_value_3a_plus_2 (a : ℝ) (h : 8 * a^2 + 10 * a + 6 = 2) :
  ∃ (m : ℝ), (3 * a + 2 ≥ m) ∧ (∀ (x : ℝ), 8 * x^2 + 10 * x + 6 = 2 → 3 * x + 2 ≥ m) ∧ m = -1 :=
sorry

end min_value_3a_plus_2_l1615_161568


namespace sugar_price_increase_l1615_161501

theorem sugar_price_increase (initial_price : ℝ) (consumption_reduction : ℝ) (new_price : ℝ) :
  initial_price = 2 →
  consumption_reduction = 0.6 →
  (1 - consumption_reduction) * new_price = initial_price →
  new_price = 5 := by
  sorry

end sugar_price_increase_l1615_161501


namespace optimal_import_quantity_l1615_161504

/-- Represents the annual import volume in units -/
def annual_volume : ℕ := 10000

/-- Represents the shipping cost per import in yuan -/
def shipping_cost : ℕ := 100

/-- Represents the rent cost per unit in yuan -/
def rent_cost_per_unit : ℕ := 2

/-- Calculates the number of imports per year given the quantity per import -/
def imports_per_year (quantity_per_import : ℕ) : ℕ :=
  annual_volume / quantity_per_import

/-- Calculates the total annual shipping cost -/
def annual_shipping_cost (quantity_per_import : ℕ) : ℕ :=
  shipping_cost * imports_per_year quantity_per_import

/-- Calculates the total annual rent cost -/
def annual_rent_cost (quantity_per_import : ℕ) : ℕ :=
  rent_cost_per_unit * (quantity_per_import / 2)

/-- Calculates the total annual cost (shipping + rent) -/
def total_annual_cost (quantity_per_import : ℕ) : ℕ :=
  annual_shipping_cost quantity_per_import + annual_rent_cost quantity_per_import

/-- Theorem stating that 1000 units per import minimizes the total annual cost -/
theorem optimal_import_quantity :
  ∀ q : ℕ, q > 0 → q ≤ annual_volume → total_annual_cost 1000 ≤ total_annual_cost q :=
sorry

end optimal_import_quantity_l1615_161504


namespace mango_purchase_quantity_l1615_161503

/-- Calculates the quantity of mangoes purchased given the total payment, apple quantity, apple price, and mango price -/
def mango_quantity (total_payment : ℕ) (apple_quantity : ℕ) (apple_price : ℕ) (mango_price : ℕ) : ℕ :=
  ((total_payment - apple_quantity * apple_price) / mango_price)

/-- Theorem stating that the quantity of mangoes purchased is 9 kg -/
theorem mango_purchase_quantity :
  mango_quantity 1055 8 70 55 = 9 := by
  sorry

end mango_purchase_quantity_l1615_161503


namespace tangent_and_normal_equations_l1615_161573

-- Define the function f(x) = x^3
def f (x : ℝ) : ℝ := x^3

-- Define the point M₀
def M₀ : ℝ × ℝ := (2, 8)

-- Theorem statement
theorem tangent_and_normal_equations :
  let (x₀, y₀) := M₀
  let f' := λ x => 3 * x^2  -- Derivative of f
  let m_tangent := f' x₀    -- Slope of tangent line
  let m_normal := -1 / m_tangent  -- Slope of normal line
  -- Equation of tangent line
  (∀ x y, 12 * x - y - 16 = 0 ↔ y - y₀ = m_tangent * (x - x₀)) ∧
  -- Equation of normal line
  (∀ x y, x + 12 * y - 98 = 0 ↔ y - y₀ = m_normal * (x - x₀)) :=
by sorry

end tangent_and_normal_equations_l1615_161573


namespace complex_modulus_reciprocal_l1615_161520

theorem complex_modulus_reciprocal (z : ℂ) (h : (1 + z) / (1 + Complex.I) = 2 - Complex.I) :
  Complex.abs (1 / z) = Real.sqrt 5 / 5 := by
  sorry

end complex_modulus_reciprocal_l1615_161520


namespace square_area_from_rectangles_l1615_161508

/-- The area of a square formed by three identical rectangles -/
theorem square_area_from_rectangles (width : ℝ) (h1 : width = 4) : 
  let length := 3 * width
  let square_side := length + width
  square_side ^ 2 = 256 := by sorry

end square_area_from_rectangles_l1615_161508


namespace largest_solution_is_two_l1615_161599

theorem largest_solution_is_two :
  ∃ (x : ℝ), x > 0 ∧ (x / 4 + 2 / (3 * x) = 5 / 6) ∧
  (∀ (y : ℝ), y > 0 → y / 4 + 2 / (3 * y) = 5 / 6 → y ≤ x) ∧
  x = 2 :=
sorry

end largest_solution_is_two_l1615_161599


namespace quadratic_maximum_value_l1615_161576

theorem quadratic_maximum_value :
  ∃ (max : ℝ), max = 111 / 4 ∧ ∀ (x : ℝ), -3 * x^2 + 15 * x + 9 ≤ max :=
by
  sorry

end quadratic_maximum_value_l1615_161576


namespace cds_per_rack_l1615_161598

/-- Given a shelf that can hold 4 racks and 32 CDs, prove that each rack can hold 8 CDs. -/
theorem cds_per_rack (racks_per_shelf : ℕ) (cds_per_shelf : ℕ) (h1 : racks_per_shelf = 4) (h2 : cds_per_shelf = 32) :
  cds_per_shelf / racks_per_shelf = 8 := by
  sorry


end cds_per_rack_l1615_161598


namespace intersection_points_on_hyperbola_l1615_161574

/-- The intersection points of the lines 2tx - 3y - 5t = 0 and x - 3ty + 5 = 0,
    where t is a real number, lie on a hyperbola. -/
theorem intersection_points_on_hyperbola :
  ∀ (t x y : ℝ),
    (2 * t * x - 3 * y - 5 * t = 0) →
    (x - 3 * t * y + 5 = 0) →
    ∃ (a b : ℝ), x^2 / a^2 - y^2 / b^2 = 1 :=
by sorry

end intersection_points_on_hyperbola_l1615_161574


namespace circle_intersection_range_l1615_161519

/-- The circle described by (x-a)^2 + (y-a)^2 = 4 always has two distinct points
    at distance 1 from the origin if and only if a is in the given range -/
theorem circle_intersection_range (a : ℝ) : 
  (∃ (x₁ y₁ x₂ y₂ : ℝ), 
    (x₁ - a)^2 + (y₁ - a)^2 = 4 ∧ 
    (x₂ - a)^2 + (y₂ - a)^2 = 4 ∧
    x₁^2 + y₁^2 = 1 ∧
    x₂^2 + y₂^2 = 1 ∧
    (x₁, y₁) ≠ (x₂, y₂)) ↔ 
  (a ∈ Set.Ioo (-3 * Real.sqrt 2 / 2) (-Real.sqrt 2 / 2) ∪ 
       Set.Ioo (Real.sqrt 2 / 2) (3 * Real.sqrt 2 / 2)) :=
by sorry


end circle_intersection_range_l1615_161519


namespace noodles_given_correct_daniel_noodles_l1615_161545

/-- The number of noodles Daniel gave to William -/
def noodles_given (initial current : ℕ) : ℕ := initial - current

theorem noodles_given_correct (initial current : ℕ) (h : current ≤ initial) :
  noodles_given initial current = initial - current :=
by
  sorry

/-- The specific problem instance -/
theorem daniel_noodles :
  noodles_given 66 54 = 12 :=
by
  sorry

end noodles_given_correct_daniel_noodles_l1615_161545


namespace percentage_of_120_to_50_l1615_161521

theorem percentage_of_120_to_50 : 
  (120 : ℝ) / 50 * 100 = 240 :=
by sorry

end percentage_of_120_to_50_l1615_161521


namespace one_meeting_before_first_lap_l1615_161570

/-- Represents a runner on a circular track -/
structure Runner where
  speed : ℝ
  direction : Bool  -- True for clockwise, False for counterclockwise

/-- Calculates the number of meetings between two runners on a circular track -/
def meetings (track_length : ℝ) (runner1 runner2 : Runner) : ℕ :=
  sorry

theorem one_meeting_before_first_lap (track_length : ℝ) (runner1 runner2 : Runner) :
  track_length = 190 →
  runner1.speed = 7 →
  runner2.speed = 12 →
  runner1.direction ≠ runner2.direction →
  meetings track_length runner1 runner2 = 1 :=
sorry

end one_meeting_before_first_lap_l1615_161570


namespace problem_statement_l1615_161596

theorem problem_statement (a b : ℝ) (h_pos_a : 0 < a) (h_pos_b : 0 < b)
  (h_condition : (a / (1 + a)) + (b / (1 + b)) = 1) :
  (a / (1 + b^2)) - (b / (1 + a^2)) = a - b := by
  sorry

end problem_statement_l1615_161596


namespace third_root_of_cubic_l1615_161536

theorem third_root_of_cubic (a b : ℚ) :
  (∀ x : ℚ, a * x^3 + (a + 2*b) * x^2 + (b - 3*a) * x + (10 - a) = 0 ↔ x = -1 ∨ x = 4 ∨ x = -3/17) :=
by sorry

end third_root_of_cubic_l1615_161536


namespace digit_equation_sum_l1615_161560

theorem digit_equation_sum : 
  ∃ (Y M E T : ℕ), 
    Y < 10 ∧ M < 10 ∧ E < 10 ∧ T < 10 ∧  -- digits are less than 10
    Y ≠ M ∧ Y ≠ E ∧ Y ≠ T ∧ M ≠ E ∧ M ≠ T ∧ E ≠ T ∧  -- digits are unique
    (10 * Y + E) * (10 * M + E) = T * T * T ∧  -- (YE) * (ME) = T * T * T
    T % 2 = 0 ∧  -- T is even
    E + M + T + Y = 10 :=  -- sum equals 10
by sorry

end digit_equation_sum_l1615_161560


namespace cistern_fill_time_l1615_161507

/-- Represents the time (in hours) it takes to fill a cistern when three pipes are opened simultaneously. -/
def fill_time (rate_A rate_B rate_C : ℚ) : ℚ :=
  1 / (rate_A + rate_B + rate_C)

/-- Theorem stating that given the specific fill/empty rates of pipes A, B, and C,
    the cistern will be filled in 12 hours when all pipes are opened simultaneously. -/
theorem cistern_fill_time :
  fill_time (1/10) (1/15) (-1/12) = 12 := by
  sorry

end cistern_fill_time_l1615_161507


namespace max_value_xyz_l1615_161582

theorem max_value_xyz (x y z : ℝ) 
  (sum_zero : x + y + z = 0) 
  (sum_squares_six : x^2 + y^2 + z^2 = 6) : 
  x^2*y + y^2*z + z^2*x ≤ 6 := by
  sorry

end max_value_xyz_l1615_161582


namespace distribute_7_4_l1615_161540

/-- The number of ways to distribute n indistinguishable balls into k distinguishable boxes -/
def distribute (n : ℕ) (k : ℕ) : ℕ := sorry

/-- The number of ways to distribute 7 indistinguishable balls into 4 distinguishable boxes is 128 -/
theorem distribute_7_4 : distribute 7 4 = 128 := by sorry

end distribute_7_4_l1615_161540


namespace stratified_sampling_girls_l1615_161535

theorem stratified_sampling_girls (total_students : ℕ) (sample_size : ℕ) (girls_in_sample : ℕ) 
  (h1 : total_students = 2000)
  (h2 : sample_size = 200)
  (h3 : girls_in_sample = 103) :
  (girls_in_sample : ℚ) / sample_size * total_students = 970 := by
  sorry

end stratified_sampling_girls_l1615_161535


namespace min_value_of_expression_l1615_161555

theorem min_value_of_expression (x : ℝ) : 4^x - 2^x + 2 ≥ (3/2 : ℝ) := by
  sorry

end min_value_of_expression_l1615_161555


namespace cooking_and_yoga_count_l1615_161584

/-- Represents the number of people in different curriculum groups -/
structure CurriculumGroups where
  yoga : ℕ
  cooking : ℕ
  weaving : ℕ
  cookingOnly : ℕ
  allCurriculums : ℕ
  cookingAndWeaving : ℕ

/-- Theorem stating the number of people who study both cooking and yoga -/
theorem cooking_and_yoga_count (g : CurriculumGroups) 
  (h1 : g.yoga = 25)
  (h2 : g.cooking = 15)
  (h3 : g.weaving = 8)
  (h4 : g.cookingOnly = 2)
  (h5 : g.allCurriculums = 3)
  (h6 : g.cookingAndWeaving = 3) :
  g.cooking - g.cookingOnly - g.cookingAndWeaving + g.allCurriculums = 10 := by
  sorry


end cooking_and_yoga_count_l1615_161584


namespace marble_bag_count_l1615_161538

theorem marble_bag_count :
  ∀ (total white : ℕ),
  (6 : ℝ) + 9 + white = total →
  (9 + white : ℝ) / total = 0.7 →
  total = 20 :=
by
  sorry

end marble_bag_count_l1615_161538


namespace scientific_notation_of_small_number_l1615_161534

theorem scientific_notation_of_small_number :
  ∃ (a : ℝ) (n : ℤ), 0.000000007 = a * (10 : ℝ) ^ n ∧ 1 ≤ a ∧ a < 10 ∧ n = -9 :=
sorry

end scientific_notation_of_small_number_l1615_161534


namespace vector_sum_l1615_161529

-- Define the vectors
def a (x : ℝ) : ℝ × ℝ := (x, 1)
def b (y : ℝ) : ℝ × ℝ := (1, y)
def c : ℝ × ℝ := (2, -4)

-- Define perpendicularity for 2D vectors
def perpendicular (v w : ℝ × ℝ) : Prop :=
  v.1 * w.1 + v.2 * w.2 = 0

-- Define parallelism for 2D vectors
def parallel (v w : ℝ × ℝ) : Prop :=
  ∃ (k : ℝ), v.1 = k * w.1 ∧ v.2 = k * w.2

-- Theorem statement
theorem vector_sum (x y : ℝ) 
  (h1 : perpendicular (a x) c) 
  (h2 : parallel (b y) c) : 
  a x + b y = (3, -1) := by
  sorry

end vector_sum_l1615_161529


namespace smallest_positive_integer_with_remainders_l1615_161561

theorem smallest_positive_integer_with_remainders : 
  ∃ (x : ℕ), x > 0 ∧ 
  x % 5 = 4 ∧ 
  x % 7 = 6 ∧ 
  x % 8 = 7 ∧ 
  ∀ (y : ℕ), y > 0 → 
    y % 5 = 4 → 
    y % 7 = 6 → 
    y % 8 = 7 → 
    x ≤ y :=
by
  sorry

end smallest_positive_integer_with_remainders_l1615_161561


namespace max_value_polynomial_l1615_161581

theorem max_value_polynomial (x y : ℝ) (h : x + y = 4) :
  x^4*y + x^3*y + x^2*y + x*y + x*y^2 + x*y^3 + x*y^4 ≤ 7225/28 := by
  sorry

end max_value_polynomial_l1615_161581


namespace quadratic_inequality_range_l1615_161509

theorem quadratic_inequality_range (a : ℝ) : 
  (∀ x : ℝ, x^2 + a*x + 1 ≥ 0) ↔ a ∈ Set.Icc (-2) 2 :=
sorry

end quadratic_inequality_range_l1615_161509


namespace system_of_equations_sum_l1615_161549

theorem system_of_equations_sum (a b c x y z : ℝ) 
  (eq1 : 17 * x + b * y + c * z = 0)
  (eq2 : a * x + 29 * y + c * z = 0)
  (eq3 : a * x + b * y + 37 * z = 0)
  (ha : a ≠ 17)
  (hx : x ≠ 0) :
  a / (a - 17) + b / (b - 29) + c / (c - 37) = 1 := by
  sorry

end system_of_equations_sum_l1615_161549


namespace arithmetic_polynomial_root_count_l1615_161563

/-- Represents a polynomial of degree 5 with integer coefficients forming an arithmetic sequence. -/
structure ArithmeticPolynomial where
  a : ℤ
  d : ℤ  -- Common difference of the arithmetic sequence

/-- The number of integer roots (counting multiplicity) of an ArithmeticPolynomial. -/
def integerRootCount (p : ArithmeticPolynomial) : ℕ :=
  sorry

/-- Theorem stating the possible values for the number of integer roots. -/
theorem arithmetic_polynomial_root_count (p : ArithmeticPolynomial) :
  integerRootCount p ∈ ({0, 1, 2, 3, 5} : Set ℕ) := by
  sorry

end arithmetic_polynomial_root_count_l1615_161563


namespace fraction_addition_l1615_161511

theorem fraction_addition (a : ℝ) (h : a ≠ 0) : 3 / a + 2 / a = 5 / a := by
  sorry

end fraction_addition_l1615_161511


namespace bowling_ball_volume_l1615_161513

/-- The volume of a sphere with cylindrical holes drilled into it -/
theorem bowling_ball_volume (d : ℝ) (r1 r2 r3 h1 h2 h3 : ℝ) : 
  d = 36 → r1 = 1 → r2 = 1 → r3 = 2 → h1 = 9 → h2 = 10 → h3 = 9 → 
  (4 / 3 * π * (d / 2)^3) - (π * r1^2 * h1) - (π * r2^2 * h2) - (π * r3^2 * h3) = 7721 * π := by
  sorry

end bowling_ball_volume_l1615_161513


namespace triangle_properties_l1615_161506

/-- Represents a triangle with sides a, b, c and angles A, B, C -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- Theorem stating properties of triangles -/
theorem triangle_properties (t : Triangle) :
  (t.A > t.B → Real.sin t.A > Real.sin t.B) ∧
  (t.A = π / 6 ∧ t.b = 4 ∧ t.a = 3 → ∃ (t1 t2 : Triangle), t1 ≠ t2 ∧ 
    t1.A = t.A ∧ t1.b = t.b ∧ t1.a = t.a ∧
    t2.A = t.A ∧ t2.b = t.b ∧ t2.a = t.a) :=
by
  sorry


end triangle_properties_l1615_161506


namespace additional_savings_when_combined_l1615_161500

/-- The regular price of a window -/
def window_price : ℕ := 120

/-- The number of windows that need to be bought to get one free -/
def windows_for_free : ℕ := 6

/-- The number of windows Dave needs -/
def dave_windows : ℕ := 9

/-- The number of windows Doug needs -/
def doug_windows : ℕ := 10

/-- Calculate the cost of windows with the offer -/
def cost_with_offer (n : ℕ) : ℕ :=
  ((n + windows_for_free - 1) / windows_for_free * windows_for_free) * window_price

/-- Calculate the savings for a given number of windows -/
def savings (n : ℕ) : ℕ :=
  n * window_price - cost_with_offer n

/-- The theorem to be proved -/
theorem additional_savings_when_combined :
  savings (dave_windows + doug_windows) - (savings dave_windows + savings doug_windows) = 240 := by
  sorry

end additional_savings_when_combined_l1615_161500


namespace wood_cutting_problem_l1615_161593

theorem wood_cutting_problem (original_length : ℚ) (first_cut : ℚ) (second_cut : ℚ) :
  original_length = 35/8 ∧ first_cut = 5/3 ∧ second_cut = 9/4 →
  (original_length - first_cut - second_cut) / 3 = 11/72 := by
  sorry

end wood_cutting_problem_l1615_161593


namespace angle_A_range_l1615_161523

theorem angle_A_range (a b c : ℝ) (A : ℝ) :
  a = 2 →
  b = 2 * Real.sqrt 2 →
  c^2 = a^2 + b^2 - 2*a*b * Real.cos A →
  0 < A ∧ A ≤ π/4 :=
by sorry

end angle_A_range_l1615_161523


namespace min_value_expression_l1615_161550

theorem min_value_expression (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) :
  (x + y) / z + (x + z) / y + (y + z) / x + 3 ≥ 9 ∧
  ((x + y) / z + (x + z) / y + (y + z) / x + 3 = 9 ↔ x = y ∧ y = z) :=
by sorry

end min_value_expression_l1615_161550


namespace union_of_M_and_N_l1615_161569

def M : Set ℝ := {x : ℝ | -3 < x ∧ x < 1}
def N : Set ℝ := {x : ℝ | x ≤ -3}

theorem union_of_M_and_N : M ∪ N = {x : ℝ | x < 1} := by sorry

end union_of_M_and_N_l1615_161569


namespace smaller_circle_area_l1615_161559

/-- Two externally tangent circles with common tangents -/
structure TangentCircles where
  r : ℝ  -- radius of smaller circle
  center_small : ℝ × ℝ  -- center of smaller circle
  center_large : ℝ × ℝ  -- center of larger circle
  P : ℝ × ℝ  -- point P
  A : ℝ × ℝ  -- point A on smaller circle
  B : ℝ × ℝ  -- point B on larger circle
  externally_tangent : (center_small.1 - center_large.1)^2 + (center_small.2 - center_large.2)^2 = (r + 3*r)^2
  on_smaller_circle : (A.1 - center_small.1)^2 + (A.2 - center_small.2)^2 = r^2
  on_larger_circle : (B.1 - center_large.1)^2 + (B.2 - center_large.2)^2 = (3*r)^2
  PA_tangent : ((P.1 - A.1)*(A.1 - center_small.1) + (P.2 - A.2)*(A.2 - center_small.2))^2 = 
               ((P.1 - A.1)^2 + (P.2 - A.2)^2)*r^2
  AB_tangent : ((A.1 - B.1)*(B.1 - center_large.1) + (A.2 - B.2)*(B.2 - center_large.2))^2 = 
               ((A.1 - B.1)^2 + (A.2 - B.2)^2)*(3*r)^2
  PA_length : (P.1 - A.1)^2 + (P.2 - A.2)^2 = 36
  AB_length : (A.1 - B.1)^2 + (A.2 - B.2)^2 = 36

theorem smaller_circle_area (tc : TangentCircles) : Real.pi * tc.r^2 = 36 * Real.pi :=
sorry

end smaller_circle_area_l1615_161559


namespace total_tires_changed_l1615_161579

/-- The number of tires on a motorcycle -/
def motorcycle_tires : ℕ := 2

/-- The number of tires on a car -/
def car_tires : ℕ := 4

/-- The number of motorcycles Mike changed tires on -/
def num_motorcycles : ℕ := 12

/-- The number of cars Mike changed tires on -/
def num_cars : ℕ := 10

/-- Theorem: The total number of tires Mike changed is 64 -/
theorem total_tires_changed : 
  num_motorcycles * motorcycle_tires + num_cars * car_tires = 64 := by
  sorry

end total_tires_changed_l1615_161579


namespace no_solution_when_n_negative_one_l1615_161595

-- Define the system of equations
def system (n x y z : ℝ) : Prop :=
  n * x^2 + y = 2 ∧ n * y^2 + z = 2 ∧ n * z^2 + x = 2

-- Theorem stating that the system has no solution when n = -1
theorem no_solution_when_n_negative_one :
  ¬ ∃ (x y z : ℝ), system (-1) x y z :=
sorry

end no_solution_when_n_negative_one_l1615_161595


namespace odd_function_extension_l1615_161592

def is_odd_function (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

theorem odd_function_extension 
  (f : ℝ → ℝ) 
  (h_odd : is_odd_function f) 
  (h_pos : ∀ x > 0, f x = x^2 - 2*x) : 
  ∀ x ≤ 0, f x = -x^2 - 2*x := by
  sorry

end odd_function_extension_l1615_161592


namespace stickers_for_square_window_l1615_161528

/-- Calculates the number of stickers needed to decorate a square window -/
theorem stickers_for_square_window (side_length interval : ℕ) : 
  side_length = 90 → interval = 3 → (4 * side_length) / interval = 120 := by
  sorry

end stickers_for_square_window_l1615_161528


namespace playground_fence_length_l1615_161565

/-- The side length of the square fence around the playground -/
def playground_side_length : ℝ := 27

/-- The length of the garden -/
def garden_length : ℝ := 12

/-- The width of the garden -/
def garden_width : ℝ := 9

/-- The total fencing for both the playground and the garden -/
def total_fencing : ℝ := 150

/-- Theorem stating that the side length of the square fence around the playground is 27 yards -/
theorem playground_fence_length :
  4 * playground_side_length + 2 * (garden_length + garden_width) = total_fencing :=
sorry

end playground_fence_length_l1615_161565


namespace sugar_remaining_l1615_161526

/-- Given 24 kilos of sugar divided into 4 bags with specified losses, 
    prove that 19.8 kilos of sugar remain. -/
theorem sugar_remaining (total_sugar : ℝ) (num_bags : ℕ) 
  (loss1 loss2 loss3 loss4 : ℝ) :
  total_sugar = 24 ∧ 
  num_bags = 4 ∧ 
  loss1 = 0.1 ∧ 
  loss2 = 0.15 ∧ 
  loss3 = 0.2 ∧ 
  loss4 = 0.25 → 
  (total_sugar / num_bags) * 
    ((1 - loss1) + (1 - loss2) + (1 - loss3) + (1 - loss4)) = 19.8 :=
by sorry

end sugar_remaining_l1615_161526


namespace prime_cube_plus_one_l1615_161524

theorem prime_cube_plus_one (p : ℕ) (h_prime : Nat.Prime p) :
  (∃ (x y : ℕ), x > 0 ∧ y > 0 ∧ p^x = y^3 + 1) ↔ p = 2 ∨ p = 3 := by
  sorry

end prime_cube_plus_one_l1615_161524


namespace string_longest_piece_fraction_l1615_161587

theorem string_longest_piece_fraction (L : ℝ) (x : ℝ) (h1 : x > 0) : 
  x + 2*x + 4*x + 8*x = L → 8*x / L = 8/15 := by
  sorry

end string_longest_piece_fraction_l1615_161587


namespace range_of_a_for_subset_l1615_161547

-- Define the sets A and B
def A (a : ℝ) := { x : ℝ | |x - (a+1)^2/2| ≤ (a-1)^2/2 }
def B (a : ℝ) := { x : ℝ | x^2 - 3*(a+1)*x + 2*(3*a+1) ≤ 0 }

-- Define the subset relation
def is_subset (S T : Set ℝ) := ∀ x, x ∈ S → x ∈ T

-- State the theorem
theorem range_of_a_for_subset : 
  { a : ℝ | is_subset (A a) (B a) } = Set.union (Set.Icc 1 3) {-1} :=
sorry

end range_of_a_for_subset_l1615_161547


namespace ball_drawing_probability_l1615_161558

-- Define the sample space
def Ω : Type := Fin 4 × Fin 3

-- Define the events
def A : Set Ω := {ω | (ω.1 < 2 ∧ ω.2 < 1) ∨ (ω.1 ≥ 2 ∧ ω.2 ≥ 1)}
def B : Set Ω := {ω | ω.1 < 2}
def C : Set Ω := {ω | (ω.1 < 2 ∧ ω.2 < 1) ∨ (ω.1 ≥ 2 ∧ ω.2 = 1)}
def D : Set Ω := {ω | (ω.1 < 2 ∧ ω.2 ≥ 1) ∨ (ω.1 ≥ 2 ∧ ω.2 < 1)}

-- Define the probability measure
noncomputable def P : Set Ω → ℝ := sorry

-- State the theorem
theorem ball_drawing_probability :
  (P A + P D = 1) ∧
  (P (A ∩ B) = P A * P B) ∧
  (P (C ∩ D) = P C * P D) := by
  sorry

end ball_drawing_probability_l1615_161558


namespace second_table_trays_count_l1615_161515

/-- Represents the number of trays Jerry picked up -/
structure TrayPickup where
  capacity : Nat
  firstTable : Nat
  trips : Nat
  total : Nat

/-- Calculates the number of trays picked up from the second table -/
def secondTableTrays (pickup : TrayPickup) : Nat :=
  pickup.total - pickup.firstTable

/-- Theorem stating the number of trays picked up from the second table -/
theorem second_table_trays_count (pickup : TrayPickup) 
  (h1 : pickup.capacity = 8)
  (h2 : pickup.firstTable = 9)
  (h3 : pickup.trips = 2)
  (h4 : pickup.total = pickup.capacity * pickup.trips) :
  secondTableTrays pickup = 7 := by
  sorry

#check second_table_trays_count

end second_table_trays_count_l1615_161515


namespace fraction_evaluation_l1615_161537

theorem fraction_evaluation : (10^7 : ℝ) / (5 * 10^4) = 200 := by sorry

end fraction_evaluation_l1615_161537


namespace social_media_time_theorem_l1615_161533

/-- Calculates the weekly time spent on social media given daily phone usage and social media ratio -/
def weekly_social_media_time (daily_phone_time : ℝ) (social_media_ratio : ℝ) : ℝ :=
  daily_phone_time * social_media_ratio * 7

/-- Theorem: Given 6 hours of daily phone usage and half spent on social media, 
    the weekly social media time is 21 hours -/
theorem social_media_time_theorem :
  weekly_social_media_time 6 0.5 = 21 := by
  sorry

end social_media_time_theorem_l1615_161533


namespace perfect_square_base_9_l1615_161580

/-- Represents a number in base 9 of the form ab5c where a ≠ 0 -/
structure Base9Number where
  a : ℕ
  b : ℕ
  c : ℕ
  a_nonzero : a ≠ 0
  b_less_than_9 : b < 9
  c_less_than_9 : c < 9

/-- Converts a Base9Number to its decimal representation -/
def toDecimal (n : Base9Number) : ℕ :=
  729 * n.a + 81 * n.b + 45 + n.c

theorem perfect_square_base_9 (n : Base9Number) :
  ∃ (k : ℕ), toDecimal n = k^2 → n.c = 0 ∨ n.c = 7 := by
  sorry

end perfect_square_base_9_l1615_161580


namespace largest_number_l1615_161586

theorem largest_number : 
  let a := 0.938
  let b := 0.9389
  let c := 0.93809
  let d := 0.839
  let e := 0.8909
  b > a ∧ b > c ∧ b > d ∧ b > e := by
  sorry

end largest_number_l1615_161586


namespace f_domain_l1615_161597

noncomputable def f (x : ℝ) : ℝ := Real.sqrt (x + 1) + 1 / (2 - x)

def domain (f : ℝ → ℝ) : Set ℝ := {x | ∃ y, f x = y}

theorem f_domain : domain f = {x : ℝ | x ≥ -1 ∧ x ≠ 2} := by
  sorry

end f_domain_l1615_161597


namespace not_expressible_as_difference_of_squares_l1615_161510

theorem not_expressible_as_difference_of_squares (k x y : ℤ) : 
  ¬ (∃ n : ℤ, (n = 8*k + 3 ∨ n = 8*k + 5) ∧ n = x^2 - 2*y^2) :=
sorry

end not_expressible_as_difference_of_squares_l1615_161510


namespace probability_all_female_finalists_l1615_161541

def total_contestants : ℕ := 7
def female_contestants : ℕ := 4
def male_contestants : ℕ := 3
def finalists : ℕ := 3

theorem probability_all_female_finalists :
  (Nat.choose female_contestants finalists : ℚ) / (Nat.choose total_contestants finalists : ℚ) = 4 / 35 :=
sorry

end probability_all_female_finalists_l1615_161541


namespace roses_sold_l1615_161554

theorem roses_sold (initial : ℕ) (picked : ℕ) (final : ℕ) (sold : ℕ) : 
  initial = 37 → picked = 19 → final = 40 → 
  initial - sold + picked = final → 
  sold = 16 := by
sorry

end roses_sold_l1615_161554


namespace angle_sum_quarter_range_l1615_161553

-- Define acute and obtuse angles
def acute_angle (α : Real) : Prop := 0 < α ∧ α < Real.pi / 2
def obtuse_angle (β : Real) : Prop := Real.pi / 2 < β ∧ β < Real.pi

-- Theorem statement
theorem angle_sum_quarter_range (α β : Real) 
  (h_acute : acute_angle α) (h_obtuse : obtuse_angle β) :
  Real.pi / 8 < (α + β) / 4 ∧ (α + β) / 4 < 3 * Real.pi / 8 := by
  sorry

#check angle_sum_quarter_range

end angle_sum_quarter_range_l1615_161553


namespace prob_level_b_part1_prob_not_qualifying_part2_l1615_161588

-- Define the probability of success for a single attempt
def p_success : ℚ := 1/2

-- Define the number of attempts for part 1
def attempts_part1 : ℕ := 4

-- Define the number of successes required for level B
def level_b_successes : ℕ := 3

-- Define the maximum number of attempts for part 2
def max_attempts_part2 : ℕ := 5

-- Part 1: Probability of exactly 3 successes in 4 attempts
theorem prob_level_b_part1 :
  (Nat.choose attempts_part1 level_b_successes : ℚ) * p_success^level_b_successes * (1 - p_success)^(attempts_part1 - level_b_successes) = 3/16 := by
  sorry

-- Part 2: Probability of not qualifying as level B or A player
theorem prob_not_qualifying_part2 :
  let seq := List.cons p_success (List.cons p_success (List.cons (1 - p_success) (List.cons (1 - p_success) [])))
  let p_exactly_3 := (Nat.choose 4 2 : ℚ) * p_success^3 * (1 - p_success)^2
  let p_exactly_2 := p_success^2 * (1 - p_success)^2 + 3 * p_success^2 * (1 - p_success)^3
  let p_exactly_1 := p_success * (1 - p_success)^2 + p_success * (1 - p_success)^3
  let p_exactly_0 := (1 - p_success)^2
  p_exactly_3 + p_exactly_2 + p_exactly_1 + p_exactly_0 = 25/32 := by
  sorry

end prob_level_b_part1_prob_not_qualifying_part2_l1615_161588


namespace sum_of_absolute_coefficients_l1615_161502

theorem sum_of_absolute_coefficients (a₀ a₁ a₂ a₃ a₄ a₅ a₆ : ℝ) : 
  (∀ x : ℝ, (2 - x)^6 = a₀ + a₁*x + a₂*x^2 + a₃*x^3 + a₄*x^4 + a₅*x^5 + a₆*x^6) →
  |a₁| + |a₂| + |a₃| + |a₄| + |a₅| + |a₆| = 665 := by
sorry

end sum_of_absolute_coefficients_l1615_161502


namespace negation_of_universal_proposition_l1615_161516

theorem negation_of_universal_proposition :
  (¬ ∀ x : ℝ, x^2 ≥ 2) ↔ (∃ x : ℝ, x^2 < 2) := by sorry

end negation_of_universal_proposition_l1615_161516


namespace max_EH_value_l1615_161585

/-- A cyclic quadrilateral with integer side lengths --/
structure CyclicQuadrilateral where
  EF : ℕ
  FG : ℕ
  GH : ℕ
  EH : ℕ
  distinct : EF ≠ FG ∧ EF ≠ GH ∧ EF ≠ EH ∧ FG ≠ GH ∧ FG ≠ EH ∧ GH ≠ EH
  less_than_20 : EF < 20 ∧ FG < 20 ∧ GH < 20 ∧ EH < 20
  cyclic_property : EF * GH = FG * EH

/-- The maximum possible value of EH in a cyclic quadrilateral with given constraints --/
theorem max_EH_value (q : CyclicQuadrilateral) :
  (∀ q' : CyclicQuadrilateral, q'.EH ≤ q.EH) → q.EH^2 = 394 :=
sorry

end max_EH_value_l1615_161585


namespace cost_to_replace_movies_l1615_161557

/-- The cost to replace VHS movies with DVDs -/
theorem cost_to_replace_movies 
  (num_movies : ℕ) 
  (vhs_trade_in : ℚ) 
  (dvd_cost : ℚ) : 
  (num_movies : ℚ) * (dvd_cost - vhs_trade_in) = 800 :=
by
  sorry

#check cost_to_replace_movies 100 2 10

end cost_to_replace_movies_l1615_161557


namespace tangent_length_specific_circle_l1615_161583

/-- A circle passing through three points -/
structure Circle where
  p1 : ℝ × ℝ
  p2 : ℝ × ℝ
  p3 : ℝ × ℝ

/-- The length of the tangent from a point to a circle -/
def tangentLength (p : ℝ × ℝ) (c : Circle) : ℝ :=
  sorry  -- Definition omitted as it's not given in the problem conditions

/-- The theorem stating the length of the tangent from the origin to the specific circle -/
theorem tangent_length_specific_circle :
  let origin : ℝ × ℝ := (0, 0)
  let c : Circle := { p1 := (3, 4), p2 := (6, 8), p3 := (5, 13) }
  tangentLength origin c = 5 * Real.sqrt 2 := by
  sorry


end tangent_length_specific_circle_l1615_161583
