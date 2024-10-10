import Mathlib

namespace total_difference_l4078_407876

def sales_tax_rate : ℝ := 0.08
def original_price : ℝ := 120.00
def correct_discount : ℝ := 0.25
def charlie_discount : ℝ := 0.15

def anne_total : ℝ := original_price * (1 + sales_tax_rate) * (1 - correct_discount)
def ben_total : ℝ := original_price * (1 - correct_discount) * (1 + sales_tax_rate)
def charlie_total : ℝ := original_price * (1 - charlie_discount) * (1 + sales_tax_rate)

theorem total_difference : anne_total - ben_total - charlie_total = -12.96 := by
  sorry

end total_difference_l4078_407876


namespace domain_of_f_l4078_407827

noncomputable def f (x : ℝ) : ℝ := Real.sqrt (2 - x)

theorem domain_of_f :
  {x : ℝ | ∃ y, f x = y} = Set.Iic 2 :=
sorry

end domain_of_f_l4078_407827


namespace inequality_proof_l4078_407835

theorem inequality_proof (x y z : ℝ) (h1 : 0 ≤ x) (h2 : 0 ≤ y) (h3 : 0 ≤ z) (h4 : x^2 + y^2 + z^2 = 1) :
  1 ≤ (x / (1 + y*z)) + (y / (1 + z*x)) + (z / (1 + x*y)) ∧
  (x / (1 + y*z)) + (y / (1 + z*x)) + (z / (1 + x*y)) ≤ Real.sqrt 2 := by
  sorry

end inequality_proof_l4078_407835


namespace vector_parallelism_l4078_407850

theorem vector_parallelism (t : ℝ) : 
  let a : Fin 2 → ℝ := ![1, -1]
  let b : Fin 2 → ℝ := ![t, 1]
  (∃ k : ℝ, k ≠ 0 ∧ (a + b) = k • (a - b)) → t = -1 := by
  sorry

end vector_parallelism_l4078_407850


namespace other_communities_count_l4078_407859

theorem other_communities_count (total : ℕ) (muslim_percent hindu_percent sikh_percent : ℚ) : 
  total = 850 →
  muslim_percent = 44 / 100 →
  hindu_percent = 28 / 100 →
  sikh_percent = 10 / 100 →
  ↑total * (1 - (muslim_percent + hindu_percent + sikh_percent)) = 153 := by
  sorry

end other_communities_count_l4078_407859


namespace positive_integer_solutions_count_l4078_407856

theorem positive_integer_solutions_count : 
  (Finset.filter (fun (x : ℕ × ℕ × ℕ × ℕ) => x.1 + x.2.1 + x.2.2.1 + x.2.2.2 = 10) (Finset.product (Finset.range 10) (Finset.product (Finset.range 10) (Finset.product (Finset.range 10) (Finset.range 10))))).card = 84 := by
  sorry

end positive_integer_solutions_count_l4078_407856


namespace savings_to_earnings_ratio_l4078_407894

/-- Proves that the ratio of combined savings to total earnings is 1:2 --/
theorem savings_to_earnings_ratio
  (kimmie_earnings : ℚ)
  (zahra_earnings : ℚ)
  (combined_savings : ℚ)
  (h1 : kimmie_earnings = 450)
  (h2 : zahra_earnings = kimmie_earnings - kimmie_earnings / 3)
  (h3 : combined_savings = 375) :
  combined_savings / (kimmie_earnings + zahra_earnings) = 1 / 2 := by
sorry


end savings_to_earnings_ratio_l4078_407894


namespace total_wheels_in_parking_lot_l4078_407892

/-- The number of wheels on each car -/
def wheels_per_car : ℕ := 4

/-- The number of cars brought by guests -/
def guest_cars : ℕ := 10

/-- The number of cars belonging to Dylan's parents -/
def parent_cars : ℕ := 2

/-- The total number of cars in the parking lot -/
def total_cars : ℕ := guest_cars + parent_cars

/-- Theorem stating the total number of car wheels in the parking lot -/
theorem total_wheels_in_parking_lot : 
  (total_cars * wheels_per_car) = 48 := by
sorry

end total_wheels_in_parking_lot_l4078_407892


namespace fixed_point_on_line_l4078_407824

/-- The line (m-1)x + (2m-1)y = m-5 passes through the point (9, -4) for any real m -/
theorem fixed_point_on_line (m : ℝ) : (m - 1) * 9 + (2 * m - 1) * (-4) = m - 5 := by
  sorry

end fixed_point_on_line_l4078_407824


namespace bowling_ball_weight_l4078_407880

theorem bowling_ball_weight (canoe_weight : ℕ) (num_canoes num_balls : ℕ) :
  canoe_weight = 35 →
  num_canoes = 4 →
  num_balls = 10 →
  num_canoes * canoe_weight = num_balls * (num_canoes * canoe_weight / num_balls) →
  (num_canoes * canoe_weight / num_balls : ℕ) = 14 :=
by
  sorry

end bowling_ball_weight_l4078_407880


namespace fifth_term_geometric_l4078_407819

/-- Given a geometric sequence with first term a and common ratio r,
    the nth term is given by a * r^(n-1) -/
def geometric_term (a : ℝ) (r : ℝ) (n : ℕ) : ℝ := a * r^(n-1)

/-- The fifth term of a geometric sequence with first term 5 and common ratio 3y is 405y^4 -/
theorem fifth_term_geometric (y : ℝ) :
  geometric_term 5 (3*y) 5 = 405 * y^4 := by
  sorry

end fifth_term_geometric_l4078_407819


namespace smallest_value_of_complex_sum_l4078_407890

theorem smallest_value_of_complex_sum (a b c d : ℤ) (ω : ℂ) 
  (h_distinct : a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d)
  (h_omega_power : ω^4 = 1)
  (h_omega_not_one : ω ≠ 1) :
  ∃ (x y z w : ℤ), 
    x ≠ y ∧ x ≠ z ∧ x ≠ w ∧ y ≠ z ∧ y ≠ w ∧ z ≠ w ∧
    ∀ (p q r s : ℤ), p ≠ q ∧ p ≠ r ∧ p ≠ s ∧ q ≠ r ∧ q ≠ s ∧ r ≠ s →
    Complex.abs (x + y*ω + z*ω^2 + w*ω^3) ≤ Complex.abs (p + q*ω + r*ω^2 + s*ω^3) ∧
    Complex.abs (x + y*ω + z*ω^2 + w*ω^3) = 1 :=
by sorry

end smallest_value_of_complex_sum_l4078_407890


namespace line_passes_through_point_l4078_407889

theorem line_passes_through_point :
  ∀ (t : ℝ), (t + 1) * (-4) - (2 * t + 5) * (-2) - 6 = 0 := by
sorry

end line_passes_through_point_l4078_407889


namespace min_value_theorem_l4078_407878

theorem min_value_theorem (a b c : ℝ) 
  (ha : a > 0) (hb : b > 0) (hc : c > 0)
  (hmin : ∀ x, |x + a| + |x - b| + c ≥ 4) :
  (a + b + c = 4) ∧ 
  (∀ a' b' c', a' > 0 → b' > 0 → c' > 0 → 1/a' + 4/b' + 9/c' ≥ 9) ∧
  (∃ a' b' c', a' > 0 ∧ b' > 0 ∧ c' > 0 ∧ 1/a' + 4/b' + 9/c' = 9) := by
  sorry

end min_value_theorem_l4078_407878


namespace intersection_of_three_lines_l4078_407844

/-- Given three lines that intersect at one point, prove the value of a -/
theorem intersection_of_three_lines (a : ℝ) : 
  (∃! p : ℝ × ℝ, a * p.1 + 2 * p.2 + 8 = 0 ∧ 
                  4 * p.1 + 3 * p.2 = 10 ∧ 
                  2 * p.1 - p.2 = 10) → 
  a = -1 := by
sorry

end intersection_of_three_lines_l4078_407844


namespace cone_volume_increase_l4078_407802

/-- The volume of a cone increases by 612.8% when its height is increased by 120% and its radius is increased by 80% -/
theorem cone_volume_increase (r h : ℝ) (hr : r > 0) (hh : h > 0) : 
  let v := (1/3) * Real.pi * r^2 * h
  let r_new := 1.8 * r
  let h_new := 2.2 * h
  let v_new := (1/3) * Real.pi * r_new^2 * h_new
  (v_new - v) / v * 100 = 612.8 := by
  sorry


end cone_volume_increase_l4078_407802


namespace arithmetic_equality_l4078_407830

theorem arithmetic_equality : 5 * 7 + 6 * 12 + 7 * 4 + 2 * 9 = 153 := by
  sorry

end arithmetic_equality_l4078_407830


namespace closest_to_sqrt_diff_l4078_407807

def options : List ℝ := [0.18, 0.19, 0.20, 0.21, 0.22]

theorem closest_to_sqrt_diff (x : ℝ) (hx : x ∈ options) :
  x = 0.21 ↔ ∀ y ∈ options, |Real.sqrt 68 - Real.sqrt 64 - x| ≤ |Real.sqrt 68 - Real.sqrt 64 - y| :=
sorry

end closest_to_sqrt_diff_l4078_407807


namespace circle_m_range_chord_length_m_neg_two_m_value_circle_through_origin_l4078_407841

-- Define the circle equation
def circle_equation (x y m : ℝ) : Prop :=
  x^2 + y^2 - 2*m*x - 4*y + 5*m = 0

-- Define the line equation
def line_equation (x y : ℝ) : Prop :=
  2*x - y + 1 = 0

-- Theorem 1: Range of m
theorem circle_m_range :
  ∀ m : ℝ, (∃ x y : ℝ, circle_equation x y m) ↔ (m < 1 ∨ m > 4) :=
sorry

-- Theorem 2: Chord length when m = -2
theorem chord_length_m_neg_two :
  ∃ x₁ y₁ x₂ y₂ : ℝ,
    circle_equation x₁ y₁ (-2) ∧ circle_equation x₂ y₂ (-2) ∧
    line_equation x₁ y₁ ∧ line_equation x₂ y₂ ∧
    (x₁ - x₂)^2 + (y₁ - y₂)^2 = 26 :=
sorry

-- Theorem 3: Value of m when circle with MN as diameter passes through origin
theorem m_value_circle_through_origin :
  ∃ m x₁ y₁ x₂ y₂ : ℝ,
    circle_equation x₁ y₁ m ∧ circle_equation x₂ y₂ m ∧
    line_equation x₁ y₁ ∧ line_equation x₂ y₂ ∧
    x₁ * x₂ + y₁ * y₂ = 0 ∧
    m = 2/29 :=
sorry

end circle_m_range_chord_length_m_neg_two_m_value_circle_through_origin_l4078_407841


namespace sam_investment_result_l4078_407865

/-- Calculates the final amount of an investment given initial conditions and interest rates --/
def calculate_investment (initial_investment : ℝ) (first_rate : ℝ) (first_years : ℕ) 
  (multiplier : ℝ) (second_rate : ℝ) : ℝ :=
  let first_phase := initial_investment * (1 + first_rate) ^ first_years
  let second_phase := first_phase * multiplier
  let final_amount := second_phase * (1 + second_rate)
  final_amount

/-- Theorem stating the final amount of Sam's investment --/
theorem sam_investment_result : 
  calculate_investment 10000 0.20 3 3 0.15 = 59616 := by
  sorry

#eval calculate_investment 10000 0.20 3 3 0.15

end sam_investment_result_l4078_407865


namespace angle_C_is_105_degrees_l4078_407809

-- Define the triangle ABC
structure Triangle where
  A : ℝ
  B : ℝ
  C : ℝ
  a : ℝ
  b : ℝ
  c : ℝ

-- Define the theorem
theorem angle_C_is_105_degrees (t : Triangle) 
  (h1 : t.a = 3)
  (h2 : t.b = 3 * Real.sqrt 2)
  (h3 : t.B = π / 4) : -- 45° in radians
  t.C = 7 * π / 12 := -- 105° in radians
by sorry

end angle_C_is_105_degrees_l4078_407809


namespace diagonal_passes_810_cubes_l4078_407813

/-- The number of unit cubes an internal diagonal passes through in a rectangular solid -/
def cubes_passed_by_diagonal (l w h : ℕ) : ℕ :=
  l + w + h - (Nat.gcd l w + Nat.gcd w h + Nat.gcd h l) + Nat.gcd l (Nat.gcd w h)

/-- Theorem: The number of unit cubes an internal diagonal passes through
    in a 160 × 330 × 380 rectangular solid is 810 -/
theorem diagonal_passes_810_cubes :
  cubes_passed_by_diagonal 160 330 380 = 810 := by
  sorry

end diagonal_passes_810_cubes_l4078_407813


namespace C_closest_to_one_l4078_407825

def A : ℝ := 0.959595
def B : ℝ := 1.05555
def C : ℝ := 0.960960
def D : ℝ := 1.040040
def E : ℝ := 0.955555

theorem C_closest_to_one :
  |1 - C| < |1 - A| ∧
  |1 - C| < |1 - B| ∧
  |1 - C| < |1 - D| ∧
  |1 - C| < |1 - E| := by
sorry

end C_closest_to_one_l4078_407825


namespace chocolate_sales_theorem_l4078_407881

/-- Represents the chocolate sales problem -/
structure ChocolateSales where
  total_customers : ℕ
  price_A : ℕ
  price_B : ℕ
  max_B_ratio : ℚ
  price_increase_step : ℕ
  A_decrease_rate : ℕ
  B_decrease_rate : ℕ

/-- The main theorem for the chocolate sales problem -/
theorem chocolate_sales_theorem (cs : ChocolateSales)
  (h_total : cs.total_customers = 480)
  (h_price_A : cs.price_A = 90)
  (h_price_B : cs.price_B = 50)
  (h_max_B_ratio : cs.max_B_ratio = 3/5)
  (h_price_increase_step : cs.price_increase_step = 3)
  (h_A_decrease_rate : cs.A_decrease_rate = 5)
  (h_B_decrease_rate : cs.B_decrease_rate = 3) :
  ∃ (min_A : ℕ) (women_day_price_A : ℕ),
    min_A = 300 ∧
    women_day_price_A = 150 ∧
    min_A + (cs.total_customers - min_A) ≤ cs.total_customers ∧
    (cs.total_customers - min_A) ≤ cs.max_B_ratio * min_A ∧
    (min_A - (women_day_price_A - cs.price_A) / cs.price_increase_step * cs.A_decrease_rate) *
      women_day_price_A +
    ((cs.total_customers - min_A) - (women_day_price_A - cs.price_A) / cs.price_increase_step * cs.B_decrease_rate) *
      cs.price_B =
    min_A * cs.price_A + (cs.total_customers - min_A) * cs.price_B :=
by sorry


end chocolate_sales_theorem_l4078_407881


namespace complex_fraction_simplification_l4078_407840

theorem complex_fraction_simplification :
  let i : ℂ := Complex.I
  (i : ℂ) / (3 + i) = (1 : ℂ) / 10 + (3 : ℂ) / 10 * i :=
by sorry

end complex_fraction_simplification_l4078_407840


namespace solution_set_inequality_l4078_407805

theorem solution_set_inequality (x : ℝ) : 
  (3 - 2*x) * (x + 1) ≤ 0 ↔ -1 ≤ x ∧ x ≤ 3/2 := by sorry

end solution_set_inequality_l4078_407805


namespace median_line_equation_circle_equation_l4078_407887

/-- Triangle ABC with vertices A(-3,0), B(2,0), and C(0,-4) -/
structure Triangle :=
  (A : ℝ × ℝ)
  (B : ℝ × ℝ)
  (C : ℝ × ℝ)

/-- Define the specific triangle ABC -/
def triangleABC : Triangle :=
  { A := (-3, 0),
    B := (2, 0),
    C := (0, -4) }

/-- General form of a line equation: ax + by + c = 0 -/
structure Line :=
  (a : ℝ)
  (b : ℝ)
  (c : ℝ)

/-- General form of a circle equation: x^2 + y^2 + dx + ey + f = 0 -/
structure Circle :=
  (d : ℝ)
  (e : ℝ)
  (f : ℝ)

/-- Theorem: The median line of side BC in triangle ABC has equation x + 2y + 3 = 0 -/
theorem median_line_equation (t : Triangle) (l : Line) : t = triangleABC → l = { a := 1, b := 2, c := 3 } := by sorry

/-- Theorem: The circle passing through points A, B, and C has equation x^2 + y^2 + x + (5/2)y - 6 = 0 -/
theorem circle_equation (t : Triangle) (c : Circle) : t = triangleABC → c = { d := 1, e := 5/2, f := -6 } := by sorry

end median_line_equation_circle_equation_l4078_407887


namespace boss_contribution_l4078_407862

def gift_cost : ℝ := 100
def employee_contribution : ℝ := 11
def num_employees : ℕ := 5

theorem boss_contribution :
  ∃ (boss_amount : ℝ),
    boss_amount = 15 ∧
    ∃ (todd_amount : ℝ),
      todd_amount = 2 * boss_amount ∧
      boss_amount + todd_amount + (num_employees : ℝ) * employee_contribution = gift_cost :=
by sorry

end boss_contribution_l4078_407862


namespace y_in_terms_of_x_l4078_407863

theorem y_in_terms_of_x (x y : ℝ) (h : x - 2 = 4 * y + 3) : y = (x - 5) / 4 := by
  sorry

end y_in_terms_of_x_l4078_407863


namespace novel_reading_time_difference_l4078_407888

/-- The number of pages in the novel -/
def pages : ℕ := 760

/-- The time in seconds Bob takes to read one page -/
def bob_time : ℕ := 45

/-- The time in seconds Chandra takes to read one page -/
def chandra_time : ℕ := 30

/-- The difference in reading time between Bob and Chandra for the entire novel -/
def reading_time_difference : ℕ := pages * bob_time - pages * chandra_time

theorem novel_reading_time_difference :
  reading_time_difference = 11400 := by
  sorry

end novel_reading_time_difference_l4078_407888


namespace amount_less_than_five_times_number_l4078_407815

theorem amount_less_than_five_times_number (N : ℕ) (A : ℕ) : 
  N = 52 → A < 5 * N → A = 232 → A = A 
:= by sorry

end amount_less_than_five_times_number_l4078_407815


namespace division_37_by_8_l4078_407855

theorem division_37_by_8 (A B : ℕ) : 37 = 8 * A + B ∧ B < 8 → A = 4 := by
  sorry

end division_37_by_8_l4078_407855


namespace equation_solution_l4078_407823

theorem equation_solution : ∃ c : ℚ, (c - 23) / 2 = (2 * c + 5) / 7 ∧ c = 57 := by
  sorry

end equation_solution_l4078_407823


namespace milburg_population_l4078_407851

/-- The number of grown-ups in Milburg -/
def grown_ups : ℕ := 5256

/-- The number of children in Milburg -/
def children : ℕ := 2987

/-- The total population of Milburg -/
def total_population : ℕ := grown_ups + children

theorem milburg_population : total_population = 8243 := by
  sorry

end milburg_population_l4078_407851


namespace james_age_l4078_407857

theorem james_age (dan_age james_age : ℕ) : 
  (dan_age : ℚ) / james_age = 6 / 5 →
  dan_age + 4 = 28 →
  james_age = 20 := by
sorry

end james_age_l4078_407857


namespace add_preserves_inequality_l4078_407804

theorem add_preserves_inequality (a b : ℝ) (h : a < b) : 3 + a < 3 + b := by
  sorry

end add_preserves_inequality_l4078_407804


namespace population_growth_l4078_407800

theorem population_growth (x : ℝ) : 
  (((1 + x / 100) * 4) - 1) * 100 = 1100 → x = 200 := by
  sorry

end population_growth_l4078_407800


namespace sufficient_condition_for_monotonic_decrease_l4078_407869

-- Define the derivative of f
def f' (x : ℝ) : ℝ := x^2 - 4*x + 3

-- Define the property of being monotonic decreasing on an interval
def monotonic_decreasing_on (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x y, a < x ∧ x < y ∧ y < b → f y ≤ f x

-- Theorem statement
theorem sufficient_condition_for_monotonic_decrease :
  ∃ (f : ℝ → ℝ), (∀ x, deriv f x = f' x) →
    (monotonic_decreasing_on (fun x ↦ f (x + 1)) 0 1) ∧
    ¬(∀ g : ℝ → ℝ, (∀ x, deriv g x = f' x) → 
      monotonic_decreasing_on (fun x ↦ g (x + 1)) 0 1) :=
by sorry

end sufficient_condition_for_monotonic_decrease_l4078_407869


namespace power_simplification_l4078_407833

theorem power_simplification :
  (8^5 / 8^2) * 2^10 - 2^2 = 2^19 - 4 := by
  sorry

end power_simplification_l4078_407833


namespace sector_area_l4078_407868

/-- The area of a circular sector with radius 6 cm and central angle 30° is 3π cm². -/
theorem sector_area : 
  let r : ℝ := 6
  let α : ℝ := 30 * π / 180  -- Convert degrees to radians
  (1/2) * r^2 * α = 3 * π := by sorry

end sector_area_l4078_407868


namespace quadrilateral_existence_l4078_407883

theorem quadrilateral_existence : ∃ (a b c d : ℝ), 
  0 < a ∧ 0 < b ∧ 0 < c ∧ 0 < d ∧
  a ≤ b ∧ b ≤ c ∧ c ≤ d ∧
  d = 2 * a ∧
  a + b + c + d = 2 ∧
  a + b + c > d ∧
  a + b + d > c ∧
  a + c + d > b ∧
  b + c + d > a := by
  sorry

end quadrilateral_existence_l4078_407883


namespace multiplier_value_l4078_407832

theorem multiplier_value (n : ℝ) (x : ℝ) (h1 : n = 1) (h2 : 3 * n - 1 = x * n) : x = 2 := by
  sorry

end multiplier_value_l4078_407832


namespace product_and_difference_imply_sum_l4078_407838

theorem product_and_difference_imply_sum (x y : ℕ+) : 
  x * y = 24 → x - y = 5 → x + y = 11 := by
  sorry

end product_and_difference_imply_sum_l4078_407838


namespace min_value_theorem_l4078_407899

/-- Given real numbers a, b, c, d satisfying the given conditions, 
    the minimum value of (a - c)^2 + (b - d)^2 is 1/10 -/
theorem min_value_theorem (a b c d : ℝ) 
    (h1 : (2 * a^2 - Real.log a) / b = 1) 
    (h2 : (3 * c - 2) / d = 1) : 
  ∃ (x y : ℝ), ∀ (a' b' c' d' : ℝ), 
    (2 * a'^2 - Real.log a') / b' = 1 → 
    (3 * c' - 2) / d' = 1 → 
    (a' - c')^2 + (b' - d')^2 ≥ (1 : ℝ) / 10 ∧
    (x - y)^2 + ((2 * x^2 - Real.log x) - (3 * y - 2))^2 = (1 : ℝ) / 10 :=
by sorry

end min_value_theorem_l4078_407899


namespace trees_along_road_l4078_407891

theorem trees_along_road (road_length : ℕ) (tree_spacing : ℕ) (h1 : road_length = 1000) (h2 : tree_spacing = 5) :
  road_length / tree_spacing + 1 = 201 := by
  sorry

end trees_along_road_l4078_407891


namespace bricks_A_is_40_l4078_407853

/-- Represents the number of bricks of type A -/
def bricks_A : ℕ := sorry

/-- Represents the number of bricks of type B -/
def bricks_B : ℕ := sorry

/-- The number of bricks of type B is half the number of bricks of type A -/
axiom half_relation : bricks_B = bricks_A / 2

/-- The total number of bricks of type A and B is 60 -/
axiom total_bricks : bricks_A + bricks_B = 60

/-- Theorem stating that the number of bricks of type A is 40 -/
theorem bricks_A_is_40 : bricks_A = 40 := by sorry

end bricks_A_is_40_l4078_407853


namespace xy_value_l4078_407826

theorem xy_value (x y : ℝ) (h : x * (x + y) = x^2 + 12) : x * y = 12 := by
  sorry

end xy_value_l4078_407826


namespace thomas_blocks_count_l4078_407871

/-- The number of wooden blocks Thomas used in total -/
def total_blocks (stack1 stack2 stack3 stack4 stack5 : ℕ) : ℕ :=
  stack1 + stack2 + stack3 + stack4 + stack5

/-- Theorem stating the total number of blocks Thomas used -/
theorem thomas_blocks_count :
  ∃ (stack1 stack2 stack3 stack4 stack5 : ℕ),
    stack1 = 7 ∧
    stack2 = stack1 + 3 ∧
    stack3 = stack2 - 6 ∧
    stack4 = stack3 + 10 ∧
    stack5 = 2 * stack2 ∧
    total_blocks stack1 stack2 stack3 stack4 stack5 = 55 :=
by
  sorry


end thomas_blocks_count_l4078_407871


namespace weight_ratio_john_to_mary_l4078_407849

/-- Proves that the ratio of John's weight to Mary's weight is 5:4 given the specified conditions -/
theorem weight_ratio_john_to_mary :
  ∀ (john_weight mary_weight jamison_weight : ℕ),
    mary_weight = 160 →
    mary_weight + 20 = jamison_weight →
    john_weight + mary_weight + jamison_weight = 540 →
    (john_weight : ℚ) / mary_weight = 5 / 4 := by
  sorry

end weight_ratio_john_to_mary_l4078_407849


namespace sum_of_coefficients_l4078_407821

theorem sum_of_coefficients (a a₁ a₂ a₃ a₄ a₅ a₆ : ℝ) :
  (∀ x, (3*x - 1)^6 = a₆*x^6 + a₅*x^5 + a₄*x^4 + a₃*x^3 + a₂*x^2 + a₁*x + a) →
  a₆ + a₅ + a₄ + a₃ + a₂ + a₁ + a = 64 := by
sorry

end sum_of_coefficients_l4078_407821


namespace presentation_length_appropriate_l4078_407808

/-- Represents the duration of a presentation in minutes -/
def PresentationDuration := { d : ℝ // 45 ≤ d ∧ d ≤ 60 }

/-- The recommended speech rate in words per minute -/
def SpeechRate : ℝ := 160

/-- Checks if a given number of words is appropriate for the presentation -/
def isAppropriateLength (duration : PresentationDuration) (words : ℕ) : Prop :=
  (↑words : ℝ) ≥ SpeechRate * duration.val ∧ (↑words : ℝ) ≤ SpeechRate * 60

theorem presentation_length_appropriate :
  ∀ (duration : PresentationDuration), isAppropriateLength duration 9400 := by
  sorry

end presentation_length_appropriate_l4078_407808


namespace multiplicative_inverse_203_mod_317_l4078_407834

theorem multiplicative_inverse_203_mod_317 :
  ∃ x : ℕ, x < 317 ∧ (203 * x) % 317 = 1 :=
by
  use 46
  sorry

end multiplicative_inverse_203_mod_317_l4078_407834


namespace water_left_in_bucket_l4078_407866

/-- Converts milliliters to liters -/
def ml_to_l (ml : ℚ) : ℚ := ml / 1000

/-- Calculates the remaining water in a bucket after some is removed -/
def remaining_water (initial : ℚ) (removed_ml : ℚ) (removed_l : ℚ) : ℚ :=
  initial - (ml_to_l removed_ml + removed_l)

theorem water_left_in_bucket : 
  remaining_water 30 150 1.65 = 28.20 := by sorry

end water_left_in_bucket_l4078_407866


namespace quadratic_properties_l4078_407820

def f (x : ℝ) := -2 * x^2 + 4 * x + 3

theorem quadratic_properties :
  (∀ x y : ℝ, x < y → f x > f y) ∧
  (∀ x : ℝ, f (x + 1) = f (1 - x)) ∧
  (f 1 = 5) ∧
  (∀ x : ℝ, x > 1 → ∀ y : ℝ, y > x → f y < f x) ∧
  (∀ x : ℝ, x < 1 → ∀ y : ℝ, x < y → f x < f y) :=
by sorry

end quadratic_properties_l4078_407820


namespace luke_total_points_l4078_407875

/-- 
Given that Luke gained 11 points in each round and played 14 rounds,
prove that the total points he scored is 154.
-/
theorem luke_total_points : 
  let points_per_round : ℕ := 11
  let number_of_rounds : ℕ := 14
  points_per_round * number_of_rounds = 154 := by sorry

end luke_total_points_l4078_407875


namespace equivalence_of_inequalities_l4078_407845

theorem equivalence_of_inequalities (a : ℝ) : a - 1 > 0 ↔ a > 1 := by
  sorry

end equivalence_of_inequalities_l4078_407845


namespace fraction_simplification_l4078_407867

theorem fraction_simplification :
  ((3^1005)^2 - (3^1003)^2) / ((3^1004)^2 - (3^1002)^2) = 3 := by
  sorry

end fraction_simplification_l4078_407867


namespace total_money_is_75_l4078_407893

/-- Represents the money distribution and orange selling scenario -/
structure MoneyDistribution where
  x : ℝ  -- The common factor in the money distribution
  cara_money : ℝ := 4 * x
  janet_money : ℝ := 5 * x
  jerry_money : ℝ := 6 * x
  total_money : ℝ := cara_money + janet_money + jerry_money
  combined_money : ℝ := cara_money + janet_money
  selling_price_ratio : ℝ := 0.8
  loss : ℝ := combined_money - (selling_price_ratio * combined_money)

/-- Theorem stating the total amount of money given the conditions -/
theorem total_money_is_75 (d : MoneyDistribution) 
  (h_loss : d.loss = 9) : d.total_money = 75 := by
  sorry


end total_money_is_75_l4078_407893


namespace polygon_perimeter_sum_tan_greater_than_x_l4078_407895

theorem polygon_perimeter_sum (R : ℝ) (h : R > 0) :
  let n : ℕ := 1985
  let θ : ℝ := 2 * Real.pi / n
  let inner_side := 2 * R * Real.sin (θ / 2)
  let outer_side := 2 * R * Real.tan (θ / 2)
  let inner_perimeter := n * inner_side
  let outer_perimeter := n * outer_side
  inner_perimeter + outer_perimeter ≥ 4 * Real.pi * R :=
by
  sorry

theorem tan_greater_than_x (x : ℝ) (h : 0 ≤ x ∧ x < Real.pi / 2) :
  Real.tan x ≥ x :=
by
  sorry

end polygon_perimeter_sum_tan_greater_than_x_l4078_407895


namespace job_completion_time_l4078_407803

/-- The time taken for a, b, and c to finish a job together, given the conditions. -/
theorem job_completion_time (a b c : ℝ) : 
  (a + b = 1 / 15) →  -- a and b finish the job in 15 days
  (c = 1 / 7.5) →     -- c alone finishes the job in 7.5 days
  (1 / (a + b + c) = 5) :=  -- a, b, and c together finish the job in 5 days
by sorry

end job_completion_time_l4078_407803


namespace smallest_perimeter_is_78_l4078_407854

-- Define the triangle PQR
structure Triangle :=
  (P Q R : ℝ × ℝ)

-- Define the point J (intersection of angle bisectors)
def J : ℝ × ℝ := sorry

-- Define the condition that PQR has positive integer side lengths
def has_positive_integer_sides (t : Triangle) : Prop :=
  ∃ (a b c : ℕ+), 
    dist t.P t.Q = a ∧ 
    dist t.Q t.R = b ∧ 
    dist t.R t.P = c

-- Define the condition that PQR is isosceles with PQ = PR
def is_isosceles (t : Triangle) : Prop :=
  dist t.P t.Q = dist t.P t.R

-- Define the condition that J is on the angle bisectors of ∠Q and ∠R
def J_on_angle_bisectors (t : Triangle) : Prop :=
  sorry

-- Define the condition that QJ = 10
def QJ_equals_10 (t : Triangle) : Prop :=
  dist t.Q J = 10

-- Define the perimeter of the triangle
def perimeter (t : Triangle) : ℝ :=
  dist t.P t.Q + dist t.Q t.R + dist t.R t.P

-- Theorem statement
theorem smallest_perimeter_is_78 :
  ∀ t : Triangle,
    has_positive_integer_sides t →
    is_isosceles t →
    J_on_angle_bisectors t →
    QJ_equals_10 t →
    ∀ t' : Triangle,
      has_positive_integer_sides t' →
      is_isosceles t' →
      J_on_angle_bisectors t' →
      QJ_equals_10 t' →
      perimeter t ≤ perimeter t' →
      perimeter t = 78 :=
sorry

end smallest_perimeter_is_78_l4078_407854


namespace raft_travel_time_l4078_407842

theorem raft_travel_time (downstream_time upstream_time : ℝ) 
  (h1 : downstream_time = 5)
  (h2 : upstream_time = 7) :
  let steamer_speed := (1 / downstream_time + 1 / upstream_time) / 2
  let current_speed := (1 / downstream_time - 1 / upstream_time) / 2
  1 / current_speed = 35 := by sorry

end raft_travel_time_l4078_407842


namespace exists_uncolored_diameter_l4078_407873

/-- Represents a circle with some arcs colored black -/
structure BlackArcCircle where
  /-- The total circumference of the circle -/
  circumference : ℝ
  /-- The total length of black arcs -/
  blackArcLength : ℝ
  /-- Assumption that the black arc length is less than half the circumference -/
  blackArcLengthLessThanHalf : blackArcLength < circumference / 2

/-- A point on the circle -/
structure CirclePoint where
  /-- The angle of the point relative to a fixed reference point -/
  angle : ℝ

/-- Represents a diameter of the circle -/
structure Diameter where
  /-- One endpoint of the diameter -/
  point1 : CirclePoint
  /-- The other endpoint of the diameter -/
  point2 : CirclePoint
  /-- Assumption that the points are opposite each other on the circle -/
  oppositePoints : point2.angle = point1.angle + π

/-- Function to determine if a point is on a black arc -/
def isOnBlackArc (c : BlackArcCircle) (p : CirclePoint) : Prop := sorry

/-- Theorem stating that there exists a diameter with both ends uncolored -/
theorem exists_uncolored_diameter (c : BlackArcCircle) : 
  ∃ d : Diameter, ¬isOnBlackArc c d.point1 ∧ ¬isOnBlackArc c d.point2 := by sorry

end exists_uncolored_diameter_l4078_407873


namespace earliest_meeting_time_l4078_407836

def ben_lap_time : ℕ := 5
def clara_lap_time : ℕ := 9
def david_lap_time : ℕ := 8

theorem earliest_meeting_time :
  let meeting_time := Nat.lcm (Nat.lcm ben_lap_time clara_lap_time) david_lap_time
  meeting_time = 360 := by sorry

end earliest_meeting_time_l4078_407836


namespace fifth_month_sale_l4078_407847

def sale_month1 : ℕ := 6535
def sale_month2 : ℕ := 6927
def sale_month3 : ℕ := 6855
def sale_month4 : ℕ := 7230
def sale_month6 : ℕ := 4891
def average_sale : ℕ := 6500
def num_months : ℕ := 6

theorem fifth_month_sale :
  ∃ (sale_month5 : ℕ),
    sale_month5 = average_sale * num_months - (sale_month1 + sale_month2 + sale_month3 + sale_month4 + sale_month6) ∧
    sale_month5 = 6562 := by
  sorry

end fifth_month_sale_l4078_407847


namespace not_iff_right_angle_and_equation_l4078_407812

/-- Definition of a triangle with sides a, b, c and altitude m from vertex C -/
structure Triangle :=
  (a b c m : ℝ)
  (positive_sides : 0 < a ∧ 0 < b ∧ 0 < c)
  (positive_altitude : 0 < m)

/-- The equation in question -/
def satisfies_equation (t : Triangle) : Prop :=
  1 / t.m^2 = 1 / t.a^2 + 1 / t.b^2

/-- Theorem stating that the original statement is not true in general -/
theorem not_iff_right_angle_and_equation :
  ∃ (t : Triangle), satisfies_equation t ∧ ¬(t.a^2 + t.b^2 = t.c^2) :=
sorry

end not_iff_right_angle_and_equation_l4078_407812


namespace gcd_lcm_product_36_210_l4078_407806

theorem gcd_lcm_product_36_210 : Nat.gcd 36 210 * Nat.lcm 36 210 = 7560 := by
  sorry

end gcd_lcm_product_36_210_l4078_407806


namespace intersection_equals_specific_set_l4078_407870

-- Define the set P
def P : Set ℝ := {x | ∃ k : ℤ, 2 * k * Real.pi ≤ x ∧ x ≤ (2 * k + 1) * Real.pi}

-- Define the set Q
def Q : Set ℝ := {α | -4 ≤ α ∧ α ≤ 4}

-- Define the intersection set
def intersection_set : Set ℝ := {α | (-4 ≤ α ∧ α ≤ -Real.pi) ∨ (0 ≤ α ∧ α ≤ Real.pi)}

-- Theorem statement
theorem intersection_equals_specific_set : P ∩ Q = intersection_set := by sorry

end intersection_equals_specific_set_l4078_407870


namespace california_texas_plate_difference_l4078_407814

/-- The number of possible digits (0-9) -/
def num_digits : ℕ := 10

/-- The number of possible letters (A-Z) -/
def num_letters : ℕ := 26

/-- The number of possible California license plates -/
def california_plates : ℕ := num_letters^5 * num_digits^2

/-- The number of possible Texas license plates -/
def texas_plates : ℕ := num_digits^2 * num_letters^4

/-- The difference in the number of possible license plates between California and Texas -/
def plate_difference : ℕ := california_plates - texas_plates

theorem california_texas_plate_difference :
  plate_difference = 1142440000 := by
  sorry

end california_texas_plate_difference_l4078_407814


namespace apples_eaten_l4078_407852

theorem apples_eaten (total : ℕ) (eaten : ℕ) : 
  total = 6 → 
  eaten + 2 * eaten = total → 
  eaten = 2 := by
sorry

end apples_eaten_l4078_407852


namespace product_digit_sum_l4078_407828

def number1 : ℕ := 707070707070707070707070707070707070707070707070707070707070707070707070707070707070707070707070707
def number2 : ℕ := 909090909090909090909090909090909090909090909090909090909090909090909090909090909090909090909090909

theorem product_digit_sum :
  let product := number1 * number2
  let thousands_digit := (product / 1000) % 10
  let units_digit := product % 10
  thousands_digit + units_digit = 5 := by
sorry

end product_digit_sum_l4078_407828


namespace gcd_13924_32451_l4078_407861

theorem gcd_13924_32451 : Nat.gcd 13924 32451 = 1 := by
  sorry

end gcd_13924_32451_l4078_407861


namespace fireworks_display_total_l4078_407817

/-- The number of fireworks used in a New Year's Eve display -/
def fireworks_display (fireworks_per_number : ℕ) (fireworks_per_letter : ℕ) 
  (year_digits : ℕ) (phrase_letters : ℕ) (additional_boxes : ℕ) (fireworks_per_box : ℕ) : ℕ :=
  (fireworks_per_number * year_digits) + 
  (fireworks_per_letter * phrase_letters) + 
  (additional_boxes * fireworks_per_box)

/-- Theorem stating the total number of fireworks used in the display -/
theorem fireworks_display_total : 
  fireworks_display 6 5 4 12 50 8 = 484 := by
  sorry

end fireworks_display_total_l4078_407817


namespace max_term_of_sequence_l4078_407874

theorem max_term_of_sequence (n : ℕ) : 
  let a : ℕ → ℤ := λ k => -2 * k^2 + 9 * k + 3
  ∀ k, a k ≤ a 2 := by
  sorry

end max_term_of_sequence_l4078_407874


namespace exists_rational_triangle_l4078_407829

/-- A triangle with integer sides, height, and median, all less than 100 -/
structure RationalTriangle where
  a : ℕ
  b : ℕ
  c : ℕ
  height : ℕ
  median : ℕ
  a_lt_100 : a < 100
  b_lt_100 : b < 100
  c_lt_100 : c < 100
  height_lt_100 : height < 100
  median_lt_100 : median < 100
  triangle_inequality : a + b > c ∧ b + c > a ∧ c + a > b
  not_right_triangle : a^2 + b^2 ≠ c^2 ∧ b^2 + c^2 ≠ a^2 ∧ c^2 + a^2 ≠ b^2

/-- There exists a triangle with integer sides, height, and median, all less than 100, that is not a right triangle -/
theorem exists_rational_triangle : ∃ t : RationalTriangle, True := by
  sorry

end exists_rational_triangle_l4078_407829


namespace even_odd_sum_difference_l4078_407872

/-- Sum of arithmetic sequence -/
def arithmeticSum (a₁ : ℕ) (aₙ : ℕ) (n : ℕ) : ℕ :=
  n * (a₁ + aₙ) / 2

/-- Sum of even integers from 2 to 120 -/
def a : ℕ := arithmeticSum 2 120 60

/-- Sum of odd integers from 1 to 119 -/
def b : ℕ := arithmeticSum 1 119 60

/-- The difference between the sum of even integers from 2 to 120 and
    the sum of odd integers from 1 to 119 is 60 -/
theorem even_odd_sum_difference : a - b = 60 := by sorry

end even_odd_sum_difference_l4078_407872


namespace expression_value_l4078_407886

theorem expression_value (a b c d m : ℝ) 
  (h1 : a + b = 0)  -- a and b are opposite numbers
  (h2 : c * d = 1)  -- c and d are reciprocals
  (h3 : |m| = 2)    -- absolute value of m is 2
  : m + c * d + (a + b) / m = 3 ∨ m + c * d + (a + b) / m = -1 :=
by sorry

end expression_value_l4078_407886


namespace initial_books_l4078_407818

theorem initial_books (x : ℚ) : 
  (1/2 * x + 3 = 23) → x = 40 := by
  sorry

end initial_books_l4078_407818


namespace barbara_wins_iff_odd_sum_l4078_407831

/-- Newspaper cutting game -/
def newspaper_game_winner (a b d : ℝ) : Prop :=
  let x := ⌊a / d⌋
  let y := ⌊b / d⌋
  Odd (x + y)

/-- Barbara wins the newspaper cutting game if and only if the sum of the floor divisions is odd -/
theorem barbara_wins_iff_odd_sum (a b d : ℝ) (h : d > 0) :
  newspaper_game_winner a b d ↔ Barbara_wins :=
sorry

end barbara_wins_iff_odd_sum_l4078_407831


namespace quadratic_inequality_always_negative_l4078_407810

theorem quadratic_inequality_always_negative :
  ∀ x : ℝ, -8 * x^2 + 4 * x - 3 < 0 := by
sorry

end quadratic_inequality_always_negative_l4078_407810


namespace integer_sum_problem_l4078_407811

theorem integer_sum_problem (x y : ℕ) (h1 : x > y) (h2 : x - y = 8) (h3 : x * y = 240) : x + y = 32 := by
  sorry

end integer_sum_problem_l4078_407811


namespace frog_arrangement_count_l4078_407898

/-- Represents the number of frogs of each color -/
structure FrogCounts where
  green : Nat
  red : Nat
  blue : Nat

/-- Represents the arrangement rules for frogs -/
structure FrogRules where
  green_red_adjacent : Bool
  green_blue_adjacent : Bool
  red_blue_adjacent : Bool
  blue_blue_adjacent : Bool

/-- Calculates the number of valid frog arrangements -/
def countFrogArrangements (counts : FrogCounts) (rules : FrogRules) : Nat :=
  sorry

/-- The main theorem stating the number of valid frog arrangements -/
theorem frog_arrangement_count :
  let counts : FrogCounts := ⟨2, 3, 2⟩
  let rules : FrogRules := ⟨false, true, true, true⟩
  countFrogArrangements counts rules = 72 := by sorry

end frog_arrangement_count_l4078_407898


namespace business_income_calculation_l4078_407879

theorem business_income_calculation 
  (spending income : ℕ) 
  (spending_income_ratio : spending * 9 = income * 5) 
  (profit : ℕ) 
  (profit_equation : profit = income - spending) 
  (profit_value : profit = 48000) : income = 108000 := by
sorry

end business_income_calculation_l4078_407879


namespace find_number_l4078_407837

theorem find_number : ∃ x : ℝ, (5 * x) / (180 / 3) + 70 = 71 ∧ x = 12 := by sorry

end find_number_l4078_407837


namespace total_students_l4078_407858

theorem total_students (boys girls : ℕ) (h1 : boys * 5 = girls * 8) (h2 : girls = 300) :
  boys + girls = 780 :=
by sorry

end total_students_l4078_407858


namespace dans_final_limes_l4078_407843

def initial_limes : ℕ := 9
def sara_gift : ℕ := 4
def juice_used : ℕ := 5
def neighbor_gift : ℕ := 3

theorem dans_final_limes : 
  initial_limes + sara_gift - juice_used - neighbor_gift = 5 := by
  sorry

end dans_final_limes_l4078_407843


namespace same_grade_probability_l4078_407882

-- Define the number of student volunteers in each grade
def grade_A_volunteers : ℕ := 240
def grade_B_volunteers : ℕ := 160
def grade_C_volunteers : ℕ := 160

-- Define the total number of student volunteers
def total_volunteers : ℕ := grade_A_volunteers + grade_B_volunteers + grade_C_volunteers

-- Define the number of students to be selected using stratified sampling
def selected_students : ℕ := 7

-- Define the number of students to be chosen for sanitation work
def sanitation_workers : ℕ := 2

-- Define the function to calculate the number of students selected from each grade
def students_per_grade (grade_volunteers : ℕ) : ℕ :=
  (grade_volunteers * selected_students) / total_volunteers

-- Theorem: The probability of selecting 2 students from the same grade is 5/21
theorem same_grade_probability :
  (students_per_grade grade_A_volunteers) * (students_per_grade grade_A_volunteers - 1) / 2 +
  (students_per_grade grade_B_volunteers) * (students_per_grade grade_B_volunteers - 1) / 2 +
  (students_per_grade grade_C_volunteers) * (students_per_grade grade_C_volunteers - 1) / 2 =
  5 * (selected_students * (selected_students - 1) / 2) / 21 :=
by sorry

end same_grade_probability_l4078_407882


namespace smallest_n_for_milly_victory_l4078_407896

def is_valid_coloring (n : ℕ) (coloring : ℕ → Bool) : Prop :=
  ∀ a b c d : ℕ, a ≤ n ∧ b ≤ n ∧ c ≤ n ∧ d ≤ n →
    (coloring a = coloring b ∧ coloring b = coloring c ∧ coloring c = coloring d) →
    a + b + c ≠ d

theorem smallest_n_for_milly_victory : 
  (∀ n < 11, ∃ coloring : ℕ → Bool, is_valid_coloring n coloring) ∧
  (∀ coloring : ℕ → Bool, ¬ is_valid_coloring 11 coloring) :=
sorry

end smallest_n_for_milly_victory_l4078_407896


namespace difference_of_x_and_y_l4078_407848

theorem difference_of_x_and_y (x y : ℝ) (h1 : x + y = 9) (h2 : x^2 - y^2 = 27) : x - y = 3 := by
  sorry

end difference_of_x_and_y_l4078_407848


namespace problem_solution_l4078_407822

def even_squared_sum : ℕ := (2^2) + (4^2) + (6^2) + (8^2) + (10^2)

def prime_count : ℕ := 4

def odd_product : ℕ := 1 * 3 * 5 * 7 * 9

theorem problem_solution :
  let x := even_squared_sum
  let y := prime_count
  let z := odd_product
  x - y + z = 1161 :=
by sorry

end problem_solution_l4078_407822


namespace minimize_distance_sum_l4078_407884

/-- Given points P and Q in the xy-plane, and R on the line segment PQ, 
    prove that R(2, -1/9) minimizes the sum of distances PR + RQ -/
theorem minimize_distance_sum (P Q R : ℝ × ℝ) : 
  P = (-3, -4) → 
  Q = (6, 3) → 
  R.1 = 2 → 
  R.2 = -1/9 → 
  (∃ (t : ℝ), 0 ≤ t ∧ t ≤ 1 ∧ R = (1 - t) • P + t • Q) →
  ∀ (S : ℝ × ℝ), (∃ (u : ℝ), 0 ≤ u ∧ u ≤ 1 ∧ S = (1 - u) • P + u • Q) →
    Real.sqrt ((R.1 - P.1)^2 + (R.2 - P.2)^2) + Real.sqrt ((Q.1 - R.1)^2 + (Q.2 - R.2)^2) ≤ 
    Real.sqrt ((S.1 - P.1)^2 + (S.2 - P.2)^2) + Real.sqrt ((Q.1 - S.1)^2 + (Q.2 - S.2)^2) :=
by sorry


end minimize_distance_sum_l4078_407884


namespace negation_of_universal_proposition_l4078_407864

theorem negation_of_universal_proposition :
  (¬ ∀ x : ℝ, x^2 - 3*x + 1 ≤ 0) ↔ (∃ x : ℝ, x^2 - 3*x + 1 > 0) := by
  sorry

end negation_of_universal_proposition_l4078_407864


namespace xy_difference_squared_l4078_407885

theorem xy_difference_squared (x y : ℝ) 
  (h1 : x * y = 3) 
  (h2 : x - y = -2) : 
  x^2 * y - x * y^2 = -6 := by
sorry

end xy_difference_squared_l4078_407885


namespace marions_score_l4078_407801

theorem marions_score (total_items : ℕ) (ellas_incorrect : ℕ) (marions_additional : ℕ) : 
  total_items = 40 →
  ellas_incorrect = 4 →
  marions_additional = 6 →
  (total_items - ellas_incorrect) / 2 + marions_additional = 24 :=
by sorry

end marions_score_l4078_407801


namespace base_9_conversion_l4078_407839

/-- Converts a list of digits in base 9 to its decimal (base 10) representation -/
def base9ToDecimal (digits : List Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * (9 ^ i)) 0

/-- The problem statement -/
theorem base_9_conversion :
  base9ToDecimal [1, 3, 3, 2] = 1729 := by
  sorry

end base_9_conversion_l4078_407839


namespace not_right_triangle_A_right_triangle_B_right_triangle_C_right_triangle_D_main_result_l4078_407846

/-- A function to check if three numbers can form a right triangle --/
def isRightTriangle (a b c : ℝ) : Prop :=
  a > 0 ∧ b > 0 ∧ c > 0 ∧ (a^2 + b^2 = c^2 ∨ a^2 + c^2 = b^2 ∨ b^2 + c^2 = a^2)

/-- Theorem stating that √3, 2, √5 cannot form a right triangle --/
theorem not_right_triangle_A : ¬ isRightTriangle (Real.sqrt 3) 2 (Real.sqrt 5) := by
  sorry

/-- Theorem stating that 3, 4, 5 can form a right triangle --/
theorem right_triangle_B : isRightTriangle 3 4 5 := by
  sorry

/-- Theorem stating that 0.6, 0.8, 1 can form a right triangle --/
theorem right_triangle_C : isRightTriangle 0.6 0.8 1 := by
  sorry

/-- Theorem stating that 130, 120, 50 can form a right triangle --/
theorem right_triangle_D : isRightTriangle 130 120 50 := by
  sorry

/-- Main theorem combining all the above results --/
theorem main_result : 
  ¬ isRightTriangle (Real.sqrt 3) 2 (Real.sqrt 5) ∧
  isRightTriangle 3 4 5 ∧
  isRightTriangle 0.6 0.8 1 ∧
  isRightTriangle 130 120 50 := by
  sorry

end not_right_triangle_A_right_triangle_B_right_triangle_C_right_triangle_D_main_result_l4078_407846


namespace paint_mixture_ratio_l4078_407816

/-- Given a paint mixture with a ratio of blue:green:yellow as 4:3:5,
    if 15 quarts of yellow paint are used, then 9 quarts of green paint should be used. -/
theorem paint_mixture_ratio (blue green yellow : ℚ) :
  blue / green = 4 / 3 →
  green / yellow = 3 / 5 →
  yellow = 15 →
  green = 9 := by
sorry

end paint_mixture_ratio_l4078_407816


namespace max_value_x_sqrt_1_minus_4x_squared_l4078_407897

theorem max_value_x_sqrt_1_minus_4x_squared (x : ℝ) :
  0 < x → x < 1/2 → x * Real.sqrt (1 - 4 * x^2) ≤ 1/4 :=
sorry

end max_value_x_sqrt_1_minus_4x_squared_l4078_407897


namespace intersection_of_M_and_N_l4078_407877

-- Define the sets M and N
def M : Set ℝ := {x : ℝ | -4 < x ∧ x < 2}
def N : Set ℝ := {x : ℝ | x^2 - x - 6 < 0}

-- State the theorem
theorem intersection_of_M_and_N : M ∩ N = {x : ℝ | -2 < x ∧ x < 2} := by sorry

end intersection_of_M_and_N_l4078_407877


namespace contractor_average_wage_l4078_407860

def average_wage (male_count female_count child_count : ℕ)
                 (male_wage female_wage child_wage : ℚ) : ℚ :=
  let total_workers := male_count + female_count + child_count
  let total_wage := male_count * male_wage + female_count * female_wage + child_count * child_wage
  total_wage / total_workers

theorem contractor_average_wage :
  average_wage 20 15 5 25 20 8 = 21 := by
  sorry

end contractor_average_wage_l4078_407860
