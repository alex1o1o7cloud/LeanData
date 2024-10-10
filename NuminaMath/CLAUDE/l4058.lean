import Mathlib

namespace function_zero_point_l4058_405806

theorem function_zero_point
  (f : ℝ → ℝ)
  (h_mono : Monotone f)
  (h_prop : ∀ x : ℝ, f (f x - 2^x) = -1/2) :
  ∃! x : ℝ, f x = 0 ∧ x = 0 := by
  sorry

end function_zero_point_l4058_405806


namespace age_difference_l4058_405854

theorem age_difference (masc_age sam_age : ℕ) : 
  masc_age > sam_age →
  masc_age + sam_age = 27 →
  masc_age = 17 →
  sam_age = 10 →
  masc_age - sam_age = 7 := by
sorry

end age_difference_l4058_405854


namespace vector_magnitude_l4058_405827

def a (m : ℝ) : ℝ × ℝ := (2, m)
def b (m : ℝ) : ℝ × ℝ := (-1, m)

def parallel (v w : ℝ × ℝ) : Prop :=
  ∃ (k : ℝ), v = k • w

theorem vector_magnitude (m : ℝ) :
  parallel (2 • (a m) + b m) (b m) → ‖a m‖ = 2 := by
  sorry

end vector_magnitude_l4058_405827


namespace cubic_sum_theorem_l4058_405893

theorem cubic_sum_theorem (x : ℝ) (h : x^3 + 1/x^3 = 110) : x + 1/x = 5 := by
  sorry

end cubic_sum_theorem_l4058_405893


namespace no_integer_sqrt_representation_l4058_405832

theorem no_integer_sqrt_representation : ¬ ∃ (A B : ℤ), 99999 + 111111 * Real.sqrt 3 = (A + B * Real.sqrt 3) ^ 2 := by
  sorry

end no_integer_sqrt_representation_l4058_405832


namespace problem_1_problem_2_l4058_405889

-- Problem 1
theorem problem_1 (α : Real) (h : Real.tan (π/4 + α) = 2) :
  Real.sin (2*α) + Real.cos α ^ 2 = 3/2 := by sorry

-- Problem 2
theorem problem_2 (x₁ y₁ x₂ y₂ α : Real) 
  (h1 : x₁^2 + y₁^2 = 1) 
  (h2 : x₂^2 + y₂^2 = 1) 
  (h3 : Real.sin α + Real.cos α = 7/17) 
  (h4 : 0 < α) (h5 : α < π) :
  x₁*x₂ + y₁*y₂ = -8/17 := by sorry

end problem_1_problem_2_l4058_405889


namespace partial_fraction_decomposition_l4058_405872

theorem partial_fraction_decomposition :
  ∀ x : ℝ, x ≠ 0 → x ≠ 1 → x ≠ -1 →
  (-x^2 + 5*x - 6) / (x^3 - x) = 6 / x + (-7*x + 5) / (x^2 - 1) :=
by sorry

end partial_fraction_decomposition_l4058_405872


namespace train_speed_l4058_405820

/-- The speed of a train given its length and time to cross a fixed point. -/
theorem train_speed (length time : ℝ) (h1 : length = 400) (h2 : time = 16) :
  length / time = 25 := by
  sorry

end train_speed_l4058_405820


namespace parallel_lines_d_value_l4058_405860

/-- Two lines are parallel if their slopes are equal -/
def parallel (m₁ m₂ : ℝ) : Prop := m₁ = m₂

/-- The slope of the first line -/
def slope₁ : ℝ := -3

/-- The slope of the second line -/
def slope₂ (d : ℝ) : ℝ := -6 * d

theorem parallel_lines_d_value :
  ∀ d : ℝ, parallel slope₁ (slope₂ d) → d = 1/2 := by
  sorry

end parallel_lines_d_value_l4058_405860


namespace solution_satisfies_system_l4058_405898

theorem solution_satisfies_system :
  let x₁ : ℚ := 1
  let x₂ : ℚ := -1
  let x₃ : ℚ := 1
  let x₄ : ℚ := -1
  let x₅ : ℚ := 1
  (x₁ + 2*x₂ + 2*x₃ + 2*x₄ + 2*x₅ = 1) ∧
  (x₁ + 3*x₂ + 4*x₃ + 4*x₄ + 4*x₅ = 2) ∧
  (x₁ + 3*x₂ + 5*x₃ + 6*x₄ + 6*x₅ = 3) ∧
  (x₁ + 3*x₂ + 5*x₃ + 7*x₄ + 8*x₅ = 4) ∧
  (x₁ + 3*x₂ + 5*x₃ + 7*x₄ + 9*x₅ = 5) := by
  sorry

end solution_satisfies_system_l4058_405898


namespace sin_3phi_value_l4058_405810

theorem sin_3phi_value (φ : ℝ) (h : Complex.exp (Complex.I * φ) = (3 + Complex.I * Real.sqrt 8) / 5) :
  Real.sin (3 * φ) = 19 * Real.sqrt 8 / 125 := by
  sorry

end sin_3phi_value_l4058_405810


namespace nested_fourth_root_l4058_405821

theorem nested_fourth_root (M : ℝ) (h : M > 1) :
  (M * (M * (M^(1/4))^(1/4))^(1/4))^(1/4) = M^(21/64) := by
  sorry

end nested_fourth_root_l4058_405821


namespace line_parallel_to_plane_l4058_405859

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relationships between lines and planes
variable (parallel : Line → Plane → Prop)
variable (perpendicular : Line → Plane → Prop)
variable (contains : Plane → Line → Prop)
variable (perp_planes : Plane → Plane → Prop)

-- State the theorem
theorem line_parallel_to_plane
  (m : Line) (α β : Plane)
  (h1 : perp_planes α β)
  (h2 : perpendicular m β)
  (h3 : ¬ contains α m) :
  parallel m α :=
sorry

end line_parallel_to_plane_l4058_405859


namespace absolute_value_inequality_l4058_405873

theorem absolute_value_inequality (x a : ℝ) (h1 : |x - 4| + |x - 3| < a) (h2 : a > 0) : a > 1 := by
  sorry

end absolute_value_inequality_l4058_405873


namespace range_of_a_equiv_l4058_405885

/-- Proposition p: The equation x² + 2ax + 1 = 0 has two real roots greater than -1 -/
def prop_p (a : ℝ) : Prop :=
  ∃ x y : ℝ, x > -1 ∧ y > -1 ∧ x ≠ y ∧ x^2 + 2*a*x + 1 = 0 ∧ y^2 + 2*a*y + 1 = 0

/-- Proposition q: The solution set of the inequality ax² - ax + 1 > 0 with respect to x is ℝ -/
def prop_q (a : ℝ) : Prop :=
  ∀ x : ℝ, a*x^2 - a*x + 1 > 0

/-- The main theorem stating the equivalence of the conditions and the range of a -/
theorem range_of_a_equiv (a : ℝ) :
  (prop_p a ∨ prop_q a) ∧ ¬prop_q a ↔ a ≤ -1 :=
sorry

end range_of_a_equiv_l4058_405885


namespace toy_store_problem_l4058_405862

/-- Toy store problem -/
theorem toy_store_problem 
  (cost_sum : ℝ) 
  (budget_A budget_B : ℝ) 
  (total_toys : ℕ) 
  (max_A : ℕ) 
  (total_budget : ℝ) 
  (sell_price_A sell_price_B : ℝ) :
  cost_sum = 40 →
  budget_A = 90 →
  budget_B = 150 →
  total_toys = 48 →
  max_A = 23 →
  total_budget = 1000 →
  sell_price_A = 30 →
  sell_price_B = 45 →
  ∃ (cost_A cost_B : ℝ) (num_plans : ℕ) (profit_function : ℕ → ℝ) (max_profit : ℝ),
    cost_A + cost_B = cost_sum ∧
    budget_A / cost_A = budget_B / cost_B ∧
    cost_A = 15 ∧
    cost_B = 25 ∧
    num_plans = 4 ∧
    (∀ m : ℕ, profit_function m = -5 * m + 960) ∧
    max_profit = 860 :=
by sorry

end toy_store_problem_l4058_405862


namespace constant_remainder_implies_b_value_l4058_405813

/-- The dividend polynomial -/
def dividend (b x : ℚ) : ℚ := 12 * x^4 - 14 * x^3 + b * x^2 + 7 * x + 9

/-- The divisor polynomial -/
def divisor (x : ℚ) : ℚ := 3 * x^2 - 4 * x + 2

/-- The remainder polynomial -/
def remainder (b x : ℚ) : ℚ := dividend b x - divisor x * (4 * x^2 + 2/3 * x)

theorem constant_remainder_implies_b_value :
  (∃ (r : ℚ), ∀ (x : ℚ), remainder b x = r) ↔ b = 16/3 := by sorry

end constant_remainder_implies_b_value_l4058_405813


namespace subset_range_m_l4058_405825

theorem subset_range_m (m : ℝ) : 
  let A : Set ℝ := {x | -2 ≤ x ∧ x ≤ 7}
  let B : Set ℝ := {x | m + 1 ≤ x ∧ x ≤ 2*m - 1}
  B ⊆ A → m ≤ 4 := by
  sorry

end subset_range_m_l4058_405825


namespace seventh_root_of_unity_product_l4058_405884

theorem seventh_root_of_unity_product (z : ℂ) (h1 : z^7 = 1) (h2 : z ≠ 1) :
  (z - 1) * (z^2 - 1) * (z^3 - 1) * (z^4 - 1) * (z^5 - 1) * (z^6 - 1) = 8 := by
  sorry

end seventh_root_of_unity_product_l4058_405884


namespace circle_and_tangent_line_l4058_405896

/-- A circle passing through three points -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- A line in 2D space -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- The circle E passing through (0,0), (1,1), and (2,0) -/
def circle_E : Circle :=
  { center := (1, 0), radius := 1 }

/-- Point P -/
def point_P : ℝ × ℝ := (2, 3)

/-- Theorem stating the properties of circle E and line l -/
theorem circle_and_tangent_line :
  (∀ (x y : ℝ), (x - circle_E.center.1)^2 + (y - circle_E.center.2)^2 = circle_E.radius^2 ↔ 
    (x = 0 ∧ y = 0) ∨ (x = 1 ∧ y = 1) ∨ (x = 2 ∧ y = 0)) ∧
  (∃ (l : Line), 
    (l.a * point_P.1 + l.b * point_P.2 + l.c = 0) ∧
    (∀ (x y : ℝ), (x - circle_E.center.1)^2 + (y - circle_E.center.2)^2 = circle_E.radius^2 →
      (l.a * x + l.b * y + l.c)^2 ≥ (l.a^2 + l.b^2) * circle_E.radius^2) ∧
    ((l.a = 1 ∧ l.b = 0 ∧ l.c = -2) ∨ (l.a = 4 ∧ l.b = -3 ∧ l.c = 1))) :=
sorry


end circle_and_tangent_line_l4058_405896


namespace chocolate_chip_cookie_coverage_l4058_405844

theorem chocolate_chip_cookie_coverage : 
  let cookie_radius : ℝ := 3
  let chip_radius : ℝ := 0.3
  let cookie_area : ℝ := π * cookie_radius^2
  let chip_area : ℝ := π * chip_radius^2
  let coverage_ratio : ℝ := 1/4
  let num_chips : ℕ := 25
  (↑num_chips * chip_area = coverage_ratio * cookie_area) ∧ 
  (∀ k : ℕ, k ≠ num_chips → ↑k * chip_area ≠ coverage_ratio * cookie_area) :=
by sorry

end chocolate_chip_cookie_coverage_l4058_405844


namespace black_white_area_ratio_l4058_405808

/-- The ratio of black to white area in concentric circles -/
theorem black_white_area_ratio :
  let radii : Fin 5 → ℝ := ![2, 4, 6, 8, 10]
  let circle_area (r : ℝ) := π * r^2
  let ring_area (i : Fin 4) := circle_area (radii (i + 1)) - circle_area (radii i)
  let black_area := circle_area (radii 0) + ring_area 1 + ring_area 3
  let white_area := ring_area 0 + ring_area 2
  black_area / white_area = 3 / 2 := by
sorry

end black_white_area_ratio_l4058_405808


namespace residue_of_negative_thousand_mod_33_l4058_405865

theorem residue_of_negative_thousand_mod_33 :
  ∃ (k : ℤ), -1000 = 33 * k + 23 ∧ (0 ≤ 23 ∧ 23 < 33) := by
  sorry

end residue_of_negative_thousand_mod_33_l4058_405865


namespace min_value_x_plus_y_l4058_405847

theorem min_value_x_plus_y (x y : ℝ) (h1 : x > 0) (h2 : y > 0) (h3 : 4 * x + y = x * y) :
  x + y ≥ 13 := by
sorry

end min_value_x_plus_y_l4058_405847


namespace carmen_earnings_l4058_405834

-- Define the sales for each house
def green_house_sales : ℕ := 3
def green_house_price : ℚ := 4

def yellow_house_thin_mints : ℕ := 2
def yellow_house_thin_mints_price : ℚ := 3.5
def yellow_house_fudge_delights : ℕ := 1
def yellow_house_fudge_delights_price : ℚ := 5

def brown_house_sales : ℕ := 9
def brown_house_price : ℚ := 2

-- Define the total earnings
def total_earnings : ℚ := 
  green_house_sales * green_house_price +
  yellow_house_thin_mints * yellow_house_thin_mints_price +
  yellow_house_fudge_delights * yellow_house_fudge_delights_price +
  brown_house_sales * brown_house_price

-- Theorem statement
theorem carmen_earnings : total_earnings = 42 := by
  sorry

end carmen_earnings_l4058_405834


namespace geometric_sequence_reciprocal_sum_l4058_405846

/-- Given a geometric sequence {a_n} with a₁ = 2 and a₁ + a₃ + a₅ = 14,
    prove that 1/a₁ + 1/a₃ + 1/a₅ = 7/8 -/
theorem geometric_sequence_reciprocal_sum (a : ℕ → ℝ) :
  (∀ n : ℕ, a (n + 1) / a n = a 2 / a 1) →  -- geometric sequence condition
  a 1 = 2 →
  a 1 + a 3 + a 5 = 14 →
  1 / a 1 + 1 / a 3 + 1 / a 5 = 7 / 8 := by
sorry

end geometric_sequence_reciprocal_sum_l4058_405846


namespace treasure_chest_problem_l4058_405840

theorem treasure_chest_problem (n : ℕ) : 
  (n > 0 ∧ n % 8 = 6 ∧ n % 9 = 5) → 
  (∀ m : ℕ, m > 0 ∧ m % 8 = 6 ∧ m % 9 = 5 → n ≤ m) → 
  (n = 14 ∧ n % 11 = 3) := by
sorry

end treasure_chest_problem_l4058_405840


namespace words_per_page_l4058_405817

theorem words_per_page (total_pages : Nat) (total_words_mod : Nat) (modulus : Nat) 
  (h1 : total_pages = 154)
  (h2 : total_words_mod = 145)
  (h3 : modulus = 221)
  (h4 : ∃ (words_per_page : Nat), words_per_page ≤ 120 ∧ 
        (total_pages * words_per_page) % modulus = total_words_mod) :
  ∃ (words_per_page : Nat), words_per_page = 96 ∧
    (total_pages * words_per_page) % modulus = total_words_mod := by
  sorry

end words_per_page_l4058_405817


namespace motorcycles_sold_is_eight_l4058_405835

/-- Represents the monthly production and sales data for a vehicle factory -/
structure VehicleProduction where
  car_material_cost : ℕ
  cars_produced : ℕ
  car_price : ℕ
  motorcycle_material_cost : ℕ
  motorcycle_price : ℕ
  profit_increase : ℕ

/-- Calculates the number of motorcycles sold per month -/
def motorcycles_sold (data : VehicleProduction) : ℕ :=
  sorry

/-- Theorem stating that the number of motorcycles sold is 8 -/
theorem motorcycles_sold_is_eight (data : VehicleProduction) 
  (h1 : data.car_material_cost = 100)
  (h2 : data.cars_produced = 4)
  (h3 : data.car_price = 50)
  (h4 : data.motorcycle_material_cost = 250)
  (h5 : data.motorcycle_price = 50)
  (h6 : data.profit_increase = 50) :
  motorcycles_sold data = 8 := by
  sorry

end motorcycles_sold_is_eight_l4058_405835


namespace allans_balloons_l4058_405874

theorem allans_balloons (total : ℕ) (jakes_balloons : ℕ) (h1 : total = 3) (h2 : jakes_balloons = 1) :
  total - jakes_balloons = 2 :=
by sorry

end allans_balloons_l4058_405874


namespace card_distribution_l4058_405823

theorem card_distribution (total_cards : ℕ) (num_people : ℕ) 
  (h1 : total_cards = 60) (h2 : num_people = 9) : 
  ∃ (people_with_fewer : ℕ), people_with_fewer = 3 ∧ 
  people_with_fewer = num_people - (total_cards % num_people) :=
by
  sorry

end card_distribution_l4058_405823


namespace total_eggs_calculation_l4058_405891

theorem total_eggs_calculation (eggs_per_omelet : ℕ) (num_people : ℕ) (omelets_per_person : ℕ)
  (h1 : eggs_per_omelet = 4)
  (h2 : num_people = 3)
  (h3 : omelets_per_person = 3) :
  eggs_per_omelet * num_people * omelets_per_person = 36 := by
  sorry

end total_eggs_calculation_l4058_405891


namespace trigonometric_identities_l4058_405841

theorem trigonometric_identities (α : Real) (h : Real.tan (π / 4 + α) = 3) :
  (Real.tan α = 1 / 2) ∧ 
  (Real.tan (2 * α) = 4 / 3) ∧ 
  ((2 * Real.sin α * Real.cos α + 3 * Real.cos (2 * α)) / 
   (5 * Real.cos (2 * α) - 3 * Real.sin (2 * α)) = 13 / 3) := by
  sorry

end trigonometric_identities_l4058_405841


namespace triangle_area_fraction_l4058_405871

-- Define the grid dimensions
def grid_width : ℕ := 8
def grid_height : ℕ := 6

-- Define the triangle vertices
def point_A : ℚ × ℚ := (2, 5)
def point_B : ℚ × ℚ := (7, 2)
def point_C : ℚ × ℚ := (6, 6)

-- Function to calculate the area of a triangle using the Shoelace formula
def triangle_area (p1 p2 p3 : ℚ × ℚ) : ℚ :=
  let (x1, y1) := p1
  let (x2, y2) := p2
  let (x3, y3) := p3
  (1/2) * abs (x1*(y2-y3) + x2*(y3-y1) + x3*(y1-y2))

-- Theorem statement
theorem triangle_area_fraction :
  (triangle_area point_A point_B point_C) / (grid_width * grid_height : ℚ) = 17/96 := by
  sorry

end triangle_area_fraction_l4058_405871


namespace sequences_theorem_l4058_405892

-- Define the arithmetic sequence a_n
def a (n : ℕ) : ℚ := 2 * n + 1

-- Define the sum S_n of the first n terms of a_n
def S (n : ℕ) : ℚ := n * (n + 2)

-- Define the geometric sequence b_n
def b (n : ℕ) : ℚ := 3^n

-- Define T_n as the sum of the first n terms of 1/S_n
def T (n : ℕ) : ℚ := 3/4 - (2*n + 3) / (2 * (n+1) * (n+2))

-- State the theorem
theorem sequences_theorem (n : ℕ) : 
  (a n = 2 * n + 1) ∧ 
  (b n = 3^n) ∧ 
  (T n = 3/4 - (2*n + 3) / (2 * (n+1) * (n+2))) ∧
  (a 1 = b 1) ∧ 
  (a 4 = b 2) ∧ 
  (a 13 = b 3) :=
by sorry

end sequences_theorem_l4058_405892


namespace secretary_donuts_donut_problem_l4058_405800

theorem secretary_donuts (initial : ℕ) (bill_eaten : ℕ) (final : ℕ) : ℕ :=
  let remaining_after_bill := initial - bill_eaten
  let remaining_after_coworkers := final * 2
  let secretary_taken := remaining_after_bill - remaining_after_coworkers
  secretary_taken

theorem donut_problem :
  secretary_donuts 50 2 22 = 4 := by sorry

end secretary_donuts_donut_problem_l4058_405800


namespace problem_solution_l4058_405843

theorem problem_solution : (2010^2 - 2010) / 2010^2 = 2009 / 2010 := by
  sorry

end problem_solution_l4058_405843


namespace jan_skips_proof_l4058_405870

def initial_speed : ℕ := 70
def time_period : ℕ := 5

theorem jan_skips_proof (doubled_speed : ℕ) (total_skips : ℕ) 
  (h1 : doubled_speed = 2 * initial_speed) 
  (h2 : total_skips = doubled_speed * time_period) : 
  total_skips = 700 := by sorry

end jan_skips_proof_l4058_405870


namespace prime_power_plus_one_prime_l4058_405853

theorem prime_power_plus_one_prime (x y z : ℕ) : 
  Prime x ∧ Prime y ∧ Prime z ∧ x^y + 1 = z → (x = 2 ∧ y = 2 ∧ z = 5) :=
by sorry

end prime_power_plus_one_prime_l4058_405853


namespace function_properties_l4058_405863

noncomputable def f (b c : ℝ) (x : ℝ) : ℝ := (x^2 + 1) / (b * x + c)

theorem function_properties (b c : ℝ) :
  (∀ x : ℝ, x ≠ 0 → b * x + c ≠ 0) →
  f b c 1 = 2 →
  (∃ g : ℝ → ℝ, (∀ x : ℝ, x ≠ 0 → f b c x = g x) ∧ 
                (∀ x : ℝ, x ≠ 0 → g x = x + 1/x) ∧
                (∀ x y : ℝ, 1 ≤ x ∧ x < y → g x < g y) ∧
                (∀ x : ℝ, 1 ≤ x ∧ x ≤ 2 → g x ≤ 5/2) ∧
                (∀ x : ℝ, 1 ≤ x ∧ x ≤ 2 → 2 ≤ g x) ∧
                g 2 = 5/2 ∧
                g 1 = 2) :=
by sorry

end function_properties_l4058_405863


namespace no_formula_matches_l4058_405851

/-- Represents the given formulas --/
inductive Formula
  | A
  | B
  | C
  | D

/-- Evaluates a formula for a given x --/
def evaluate (f : Formula) (x : ℝ) : ℝ :=
  match f with
  | .A => x^3 + 3*x + 3
  | .B => x^2 + 4*x + 3
  | .C => x^3 + x^2 + 2*x + 1
  | .D => 2*x^3 - x + 5

/-- The set of given (x, y) pairs --/
def pairs : List (ℝ × ℝ) := [(1, 7), (2, 17), (3, 31), (4, 49), (5, 71)]

/-- Checks if a formula matches all given pairs --/
def matchesAll (f : Formula) : Prop :=
  ∀ (p : ℝ × ℝ), p ∈ pairs → evaluate f p.1 = p.2

theorem no_formula_matches : ∀ (f : Formula), ¬(matchesAll f) := by
  sorry

end no_formula_matches_l4058_405851


namespace fraction_equality_l4058_405816

theorem fraction_equality (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) 
  (h : (4 * x - 2 * y) / (2 * x + 6 * y) = 3) : 
  (2 * x - 6 * y) / (4 * x + y) = 2 / 3 := by
  sorry

end fraction_equality_l4058_405816


namespace popsicle_melting_speed_l4058_405899

theorem popsicle_melting_speed (n : ℕ) (a : ℕ → ℝ) :
  n = 6 →
  (∀ i, 1 ≤ i → i < n → a (i + 1) = 2 * a i) →
  a n = 32 * a 1 := by
  sorry

end popsicle_melting_speed_l4058_405899


namespace pet_store_cages_l4058_405882

/-- Given a pet store with an initial number of puppies, some sold, and a fixed number per cage,
    calculate the number of cages needed for the remaining puppies. -/
theorem pet_store_cages (initial_puppies : ℕ) (sold_puppies : ℕ) (puppies_per_cage : ℕ) 
    (h1 : initial_puppies = 102)
    (h2 : sold_puppies = 21)
    (h3 : puppies_per_cage = 9)
    (h4 : sold_puppies < initial_puppies) :
  (initial_puppies - sold_puppies) / puppies_per_cage = 9 :=
by sorry

end pet_store_cages_l4058_405882


namespace solutions_absolute_value_equation_l4058_405858

theorem solutions_absolute_value_equation :
  (∀ x : ℝ, |x| = 1 ↔ x = 1 ∨ x = -1) :=
by sorry

end solutions_absolute_value_equation_l4058_405858


namespace only_ball_draw_is_classical_l4058_405809

/-- Represents a probability experiment -/
inductive Experiment
| ballDraw
| busWait
| coinToss
| waterTest

/-- Checks if an experiment has a finite number of outcomes -/
def isFinite (e : Experiment) : Prop :=
  match e with
  | Experiment.ballDraw => true
  | Experiment.busWait => false
  | Experiment.coinToss => true
  | Experiment.waterTest => false

/-- Checks if an experiment has equally likely outcomes -/
def isEquallyLikely (e : Experiment) : Prop :=
  match e with
  | Experiment.ballDraw => true
  | Experiment.busWait => false
  | Experiment.coinToss => false
  | Experiment.waterTest => false

/-- Defines a classical probability model -/
def isClassicalProbabilityModel (e : Experiment) : Prop :=
  isFinite e ∧ isEquallyLikely e

/-- Theorem stating that only the ball draw experiment is a classical probability model -/
theorem only_ball_draw_is_classical : 
  ∀ e : Experiment, isClassicalProbabilityModel e ↔ e = Experiment.ballDraw :=
by sorry

end only_ball_draw_is_classical_l4058_405809


namespace first_project_depth_l4058_405836

-- Define the parameters for the first digging project
def length1 : ℝ := 25
def breadth1 : ℝ := 30
def days1 : ℝ := 12

-- Define the parameters for the second digging project
def length2 : ℝ := 20
def breadth2 : ℝ := 50
def depth2 : ℝ := 75
def days2 : ℝ := 12

-- Define the function to calculate volume
def volume (length : ℝ) (breadth : ℝ) (depth : ℝ) : ℝ :=
  length * breadth * depth

-- Theorem statement
theorem first_project_depth :
  ∃ (depth1 : ℝ),
    volume length1 breadth1 depth1 = volume length2 breadth2 depth2 ∧
    depth1 = 100 := by
  sorry

end first_project_depth_l4058_405836


namespace min_value_expression_l4058_405838

theorem min_value_expression (a b c : ℝ) (h1 : 2 ≤ a) (h2 : a ≤ b) (h3 : b ≤ c) (h4 : c ≤ 3) :
  (a - 2)^2 + (b/a - 1)^2 + (c/b - 1)^2 + (3/c - 1)^2 ≥ 4 * (9^(1/4) - 5/4)^2 := by
  sorry

end min_value_expression_l4058_405838


namespace min_value_ab_l4058_405886

theorem min_value_ab (b a : ℝ) (h1 : b > 0)
  (h2 : (b^2 + 1) * (-1 / a) = 1 / b^2) : 
  ∀ x : ℝ, a * b ≥ 2 ∧ (a * b = 2 ↔ b = 1) :=
by sorry

end min_value_ab_l4058_405886


namespace als_original_portion_l4058_405897

theorem als_original_portion
  (total_initial : ℝ)
  (total_final : ℝ)
  (h_total_initial : total_initial = 1200)
  (h_total_final : total_final = 1800)
  (a b c : ℝ)
  (h_initial_sum : a + b + c = total_initial)
  (h_final_sum : (a - 150) + (2 * b) + (3 * c) = total_final) :
  a = 550 := by
sorry

end als_original_portion_l4058_405897


namespace farmer_apples_l4058_405879

/-- The number of apples given to the neighbor -/
def apples_given (initial current : ℕ) : ℕ := initial - current

/-- Theorem: The number of apples given to the neighbor is the difference between
    the initial number of apples and the current number of apples -/
theorem farmer_apples (initial current : ℕ) (h : initial ≥ current) :
  apples_given initial current = initial - current :=
by sorry

end farmer_apples_l4058_405879


namespace equidistant_point_y_coordinate_l4058_405852

/-- The y-coordinate of the point on the y-axis equidistant from (1, 0) and (4, 3) -/
theorem equidistant_point_y_coordinate : ∃ y : ℝ, 
  (Real.sqrt ((1 - 0)^2 + (0 - y)^2) = Real.sqrt ((4 - 0)^2 + (3 - y)^2)) ∧ y = 4 := by
  sorry

end equidistant_point_y_coordinate_l4058_405852


namespace complement_of_A_wrt_U_l4058_405833

def U : Set Int := {-2, -1, 1, 3, 5}
def A : Set Int := {-1, 3}

theorem complement_of_A_wrt_U :
  U \ A = {-2, 1, 5} := by sorry

end complement_of_A_wrt_U_l4058_405833


namespace john_initial_payment_l4058_405845

def soda_cost : ℕ := 2
def num_sodas : ℕ := 3
def change_received : ℕ := 14

theorem john_initial_payment :
  num_sodas * soda_cost + change_received = 20 := by
  sorry

end john_initial_payment_l4058_405845


namespace fifth_term_ratio_l4058_405814

/-- Two arithmetic sequences and their sum ratios -/
structure ArithmeticSequences where
  a : ℕ → ℝ  -- First arithmetic sequence
  b : ℕ → ℝ  -- Second arithmetic sequence
  S : ℕ → ℝ  -- Sum of first n terms of sequence a
  T : ℕ → ℝ  -- Sum of first n terms of sequence b
  sum_ratio : ∀ n : ℕ, S n / T n = (2 * n - 3) / (3 * n - 2)

/-- The ratio of the 5th terms of the sequences is 3/5 -/
theorem fifth_term_ratio (seq : ArithmeticSequences) : seq.a 5 / seq.b 5 = 3 / 5 := by
  sorry

end fifth_term_ratio_l4058_405814


namespace price_difference_year_l4058_405881

/-- 
Given:
- The price of commodity X increases by 45 cents every year
- The price of commodity Y increases by 20 cents every year
- In 2001, the price of commodity X was $4.20
- The price of commodity Y in 2001 is Y dollars

Prove that the number of years n after 2001 when the price of X is 65 cents more than 
the price of Y is given by n = (Y - 3.55) / 0.25
-/
theorem price_difference_year (Y : ℝ) : 
  let n : ℝ := (Y - 3.55) / 0.25
  let price_X (t : ℝ) : ℝ := 4.20 + 0.45 * t
  let price_Y (t : ℝ) : ℝ := Y + 0.20 * t
  price_X n = price_Y n + 0.65 :=
by sorry

end price_difference_year_l4058_405881


namespace sin_330_degrees_l4058_405804

theorem sin_330_degrees : Real.sin (330 * π / 180) = -1/2 := by
  sorry

end sin_330_degrees_l4058_405804


namespace non_redundant_password_count_l4058_405895

/-- A password is a string of characters. -/
def Password := String

/-- The set of available characters for passwords. -/
def AvailableChars : Finset Char := sorry

/-- A password is redundant if it contains a block of consecutive characters
    that can be colored red and blue such that the red and blue substrings are identical. -/
def IsRedundant (p : Password) : Prop := sorry

/-- The number of non-redundant passwords of length n. -/
def NonRedundantCount (n : ℕ) : ℕ := sorry

/-- There are at least 18^n non-redundant passwords of length n for any n ≥ 1. -/
theorem non_redundant_password_count (n : ℕ) (h : n ≥ 1) :
  NonRedundantCount n ≥ 18^n := by sorry

end non_redundant_password_count_l4058_405895


namespace stating_circle_symmetry_l4058_405828

/-- Given circle -/
def given_circle (x y : ℝ) : Prop :=
  x^2 + y^2 - 2*x - 6*y + 9 = 0

/-- Line of symmetry -/
def symmetry_line (x y : ℝ) : Prop :=
  2*x + y + 5 = 0

/-- Symmetric circle -/
def symmetric_circle (x y : ℝ) : Prop :=
  (x + 7)^2 + (y + 1)^2 = 1

/-- 
Theorem stating that the symmetric_circle is indeed symmetric to the given_circle
with respect to the symmetry_line
-/
theorem circle_symmetry :
  ∀ (x₁ y₁ x₂ y₂ : ℝ),
  given_circle x₁ y₁ →
  symmetric_circle x₂ y₂ →
  (∃ (x_mid y_mid : ℝ),
    symmetry_line x_mid y_mid ∧
    x_mid = (x₁ + x₂) / 2 ∧
    y_mid = (y₁ + y₂) / 2) :=
sorry

end stating_circle_symmetry_l4058_405828


namespace pig_count_l4058_405824

theorem pig_count (P H : ℕ) : 
  4 * P + 2 * H = 2 * (P + H) + 22 → P = 11 := by
sorry

end pig_count_l4058_405824


namespace sum_of_zeros_greater_than_one_l4058_405857

open Real

theorem sum_of_zeros_greater_than_one (a : ℝ) :
  let f := fun x : ℝ => log x - a * x + 1 / (2 * x)
  let g := fun x : ℝ => f x + a * (x - 1)
  ∀ x₁ x₂ : ℝ, x₁ < x₂ → g x₁ = 0 → g x₂ = 0 → x₁ + x₂ > 1 := by
  sorry

end sum_of_zeros_greater_than_one_l4058_405857


namespace distance_to_origin_l4058_405826

theorem distance_to_origin (z : ℂ) (h : z = 1 - 2*I) : Complex.abs z = Real.sqrt 5 := by
  sorry

end distance_to_origin_l4058_405826


namespace division_problem_l4058_405819

theorem division_problem : (144 / 6) / 3 = 8 := by
  sorry

end division_problem_l4058_405819


namespace arctan_equation_solution_l4058_405878

theorem arctan_equation_solution (x : ℝ) : 
  x = Real.sqrt ((1 - 3 * Real.sqrt 3) / 13) ∨ 
  x = -Real.sqrt ((1 - 3 * Real.sqrt 3) / 13) → 
  Real.arctan (2 / x) + Real.arctan (1 / (2 * x^2)) = π / 3 :=
by sorry

end arctan_equation_solution_l4058_405878


namespace division_problem_l4058_405815

theorem division_problem : (((120 / 5) / 2) / 3) = 4 := by
  sorry

end division_problem_l4058_405815


namespace set_equality_implies_y_zero_l4058_405876

theorem set_equality_implies_y_zero (x y : ℝ) :
  ({0, 1, x} : Set ℝ) = {x^2, y, -1} → y = 0 := by
  sorry

end set_equality_implies_y_zero_l4058_405876


namespace railway_ticket_types_l4058_405868

/-- The number of stations on the railway --/
def num_stations : ℕ := 25

/-- The number of different types of tickets needed for a railway with n stations --/
def num_ticket_types (n : ℕ) : ℕ := n * (n - 1) / 2

/-- Theorem: The number of different types of tickets needed for a railway with 25 stations is 300 --/
theorem railway_ticket_types : num_ticket_types num_stations = 300 := by
  sorry

end railway_ticket_types_l4058_405868


namespace simplify_expression_l4058_405875

theorem simplify_expression (y : ℝ) : 3*y + 5*y + 2*y + 7*y = 17*y := by
  sorry

end simplify_expression_l4058_405875


namespace bud_uncle_age_ratio_l4058_405802

/-- The ratio of Bud's age to his uncle's age -/
def age_ratio (bud_age uncle_age : ℕ) : ℚ :=
  bud_age / uncle_age

/-- Bud's age -/
def bud_age : ℕ := 8

/-- Bud's uncle's age -/
def uncle_age : ℕ := 24

theorem bud_uncle_age_ratio :
  age_ratio bud_age uncle_age = 1 / 3 := by
  sorry


end bud_uncle_age_ratio_l4058_405802


namespace unique_paintable_number_l4058_405839

def isPaintable (s b a : ℕ+) : Prop :=
  -- Sarah's sequence doesn't overlap with Bob's or Alice's
  ∀ k l : ℕ, k * s.val ≠ l * b.val ∧ k * s.val ≠ 4 + l * a.val
  -- Bob's sequence doesn't overlap with Sarah's or Alice's
  ∧ ∀ k l : ℕ, 2 + k * b.val ≠ l * s.val ∧ 2 + k * b.val ≠ 4 + l * a.val
  -- Alice's sequence doesn't overlap with Sarah's or Bob's
  ∧ ∀ k l : ℕ, 4 + k * a.val ≠ l * s.val ∧ 4 + k * a.val ≠ 2 + l * b.val
  -- Every picket is painted
  ∧ ∀ n : ℕ, n > 0 → (∃ k : ℕ, n = k * s.val ∨ n = 2 + k * b.val ∨ n = 4 + k * a.val)

theorem unique_paintable_number :
  ∃! n : ℕ, ∃ s b a : ℕ+, isPaintable s b a ∧ n = 1000 * s.val + 100 * b.val + 10 * a.val :=
by sorry

end unique_paintable_number_l4058_405839


namespace investment_options_count_l4058_405880

/-- The number of ways to distribute 3 distinct projects among 5 cities, 
    with no more than 2 projects per city. -/
def investmentOptions : ℕ := 120

/-- The number of cities available for investment. -/
def numCities : ℕ := 5

/-- The number of projects to be distributed. -/
def numProjects : ℕ := 3

/-- The maximum number of projects allowed in a single city. -/
def maxProjectsPerCity : ℕ := 2

theorem investment_options_count :
  investmentOptions = 
    (numCities.factorial / (numCities - numProjects).factorial) +
    (numCities.choose 1) * (numProjects.choose 2) * ((numCities - 1).choose 1) :=
by sorry

end investment_options_count_l4058_405880


namespace blueberries_count_l4058_405864

/-- Represents the number of blueberries in each blue box -/
def blueberries : ℕ := sorry

/-- Represents the number of strawberries in each red box -/
def strawberries : ℕ := sorry

/-- The increase in total berries when replacing a blue box with a red box -/
def total_increase : ℕ := 20

/-- The increase in difference between strawberries and blueberries after replacement -/
def difference_increase : ℕ := 80

theorem blueberries_count : blueberries = 60 :=
  by sorry

end blueberries_count_l4058_405864


namespace jerry_tickets_l4058_405877

theorem jerry_tickets (initial_tickets spent_tickets later_won_tickets current_tickets : ℕ) :
  spent_tickets = 2 →
  later_won_tickets = 47 →
  current_tickets = 49 →
  initial_tickets = current_tickets - later_won_tickets + spent_tickets →
  initial_tickets = 4 := by
sorry

end jerry_tickets_l4058_405877


namespace sum_of_exponents_l4058_405866

theorem sum_of_exponents (x y z : ℕ) 
  (h : 800670 = 8 * 10^x + 6 * 10^y + 7 * 10^z) : 
  x + y + z = 8 := by
sorry

end sum_of_exponents_l4058_405866


namespace triangle_area_ratio_l4058_405811

/-- A rectangle on a coordinate grid with vertices at (0,0), (x,0), (0,y), and (x,y) -/
structure Rectangle where
  x : ℝ
  y : ℝ

/-- The number of parts the diagonals are divided into -/
structure DiagonalDivisions where
  n : ℕ  -- number of parts for diagonal from (0,0) to (x,y)
  m : ℕ  -- number of parts for diagonal from (x,0) to (0,y)

/-- Triangle formed by joining a point on a diagonal to the rectangle's center -/
inductive Triangle
  | A  -- formed from diagonal (0,0) to (x,y)
  | B  -- formed from diagonal (x,0) to (0,y)

/-- The area of a triangle -/
def triangleArea (t : Triangle) (r : Rectangle) (d : DiagonalDivisions) : ℝ :=
  sorry  -- definition omitted as it's not directly given in the problem conditions

/-- The theorem to be proved -/
theorem triangle_area_ratio (r : Rectangle) (d : DiagonalDivisions) :
  triangleArea Triangle.A r d / triangleArea Triangle.B r d = d.m / d.n :=
sorry

end triangle_area_ratio_l4058_405811


namespace equal_variance_implies_arithmetic_square_alternating_sequence_is_equal_variance_equal_variance_subsequence_l4058_405818

-- Define the equal variance sequence property
def is_equal_variance_sequence (a : ℕ+ → ℝ) : Prop :=
  ∃ p : ℝ, ∀ n : ℕ+, a n ^ 2 - a (n + 1) ^ 2 = p

-- Define arithmetic sequence property
def is_arithmetic_sequence (b : ℕ+ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ+, b (n + 1) - b n = d

-- Statement 1
theorem equal_variance_implies_arithmetic_square (a : ℕ+ → ℝ) :
  is_equal_variance_sequence a → is_arithmetic_sequence (λ n => a n ^ 2) := by sorry

-- Statement 2
theorem alternating_sequence_is_equal_variance :
  is_equal_variance_sequence (λ n => (-1) ^ (n : ℕ)) := by sorry

-- Statement 3
theorem equal_variance_subsequence (a : ℕ+ → ℝ) (k : ℕ+) :
  is_equal_variance_sequence a → is_equal_variance_sequence (λ n => a (k * n)) := by sorry

end equal_variance_implies_arithmetic_square_alternating_sequence_is_equal_variance_equal_variance_subsequence_l4058_405818


namespace unique_number_property_l4058_405803

theorem unique_number_property : ∃! x : ℝ, x / 3 = x - 5 := by sorry

end unique_number_property_l4058_405803


namespace paperclip_capacity_l4058_405831

theorem paperclip_capacity (box_volume : ℝ) (box_capacity : ℕ) (cube_volume : ℝ) : 
  box_volume = 24 → 
  box_capacity = 75 → 
  cube_volume = 64 → 
  (cube_volume / box_volume * box_capacity : ℝ) = 200 := by
  sorry

end paperclip_capacity_l4058_405831


namespace fourth_number_value_l4058_405894

theorem fourth_number_value (numbers : List ℝ) 
  (h1 : numbers.length = 6)
  (h2 : numbers.sum / numbers.length = 30)
  (h3 : (numbers.take 4).sum / 4 = 25)
  (h4 : (numbers.drop 3).sum / 3 = 35) :
  numbers[3] = 25 := by
sorry

end fourth_number_value_l4058_405894


namespace emmanuel_december_charges_l4058_405887

/-- Emmanuel's total charges for December -/
def total_charges (regular_plan_cost : ℝ) (days_in_guam : ℕ) (international_data_cost : ℝ) : ℝ :=
  regular_plan_cost + (days_in_guam : ℝ) * international_data_cost

/-- Theorem: Emmanuel's total charges for December are $210 -/
theorem emmanuel_december_charges :
  total_charges 175 10 3.5 = 210 := by
  sorry

end emmanuel_december_charges_l4058_405887


namespace equation_solution_l4058_405861

theorem equation_solution : ∃ x : ℝ, x ≠ -2 ∧ (4*x^2 - 3*x + 2) / (x + 2) = 4*x - 3 → x = 1 := by
  sorry

end equation_solution_l4058_405861


namespace deposit_ratio_is_one_fifth_l4058_405848

/-- Represents the financial transaction of Lulu --/
structure LuluFinances where
  initial_amount : ℕ
  ice_cream_cost : ℕ
  cash_left : ℕ

/-- Calculates the ratio of deposited money to money left after buying the t-shirt --/
def deposit_ratio (finances : LuluFinances) : Rat :=
  let after_ice_cream := finances.initial_amount - finances.ice_cream_cost
  let after_tshirt := after_ice_cream / 2
  let deposited := after_tshirt - finances.cash_left
  deposited / after_tshirt

/-- Theorem stating that the deposit ratio is 1:5 given the initial conditions --/
theorem deposit_ratio_is_one_fifth (finances : LuluFinances) 
  (h1 : finances.initial_amount = 65)
  (h2 : finances.ice_cream_cost = 5)
  (h3 : finances.cash_left = 24) : 
  deposit_ratio finances = 1 / 5 := by
  sorry

#eval deposit_ratio ⟨65, 5, 24⟩

end deposit_ratio_is_one_fifth_l4058_405848


namespace compute_m_3v_minus_2w_l4058_405888

variable (M : Matrix (Fin 2) (Fin 2) ℝ)
variable (v w : Fin 2 → ℝ)

def Mv : Fin 2 → ℝ := ![3, -1]
def Mw : Fin 2 → ℝ := ![4, 3]

axiom mv_eq : M.mulVec v = Mv
axiom mw_eq : M.mulVec w = Mw

theorem compute_m_3v_minus_2w : M.mulVec (3 • v - 2 • w) = ![1, -9] := by sorry

end compute_m_3v_minus_2w_l4058_405888


namespace contradiction_proof_l4058_405822

theorem contradiction_proof (a b c d : ℝ) 
  (sum1 : a + b = 1) 
  (sum2 : c + d = 1) 
  (prod_sum : a * c + b * d > 1) 
  (all_nonneg : a ≥ 0 ∧ b ≥ 0 ∧ c ≥ 0 ∧ d ≥ 0) : 
  False :=
sorry

end contradiction_proof_l4058_405822


namespace line_plane_relationship_l4058_405869

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the parallelism relation between a line and a plane
variable (parallel : Line → Plane → Prop)

-- Define the containment relation between a line and a plane
variable (contains : Plane → Line → Prop)

-- Define the parallelism relation between two lines
variable (parallel_lines : Line → Line → Prop)

-- Define the "on different planes" relation between two lines
variable (different_planes : Line → Line → Prop)

-- State the theorem
theorem line_plane_relationship (m n : Line) (α : Plane) 
  (h1 : parallel m α) (h2 : contains α n) :
  parallel_lines m n ∨ different_planes m n :=
sorry

end line_plane_relationship_l4058_405869


namespace train_length_problem_l4058_405850

/-- The length of two trains passing each other --/
theorem train_length_problem (speed1 speed2 : ℝ) (passing_time : ℝ) (h1 : speed1 = 65) (h2 : speed2 = 50) (h3 : passing_time = 11.895652173913044) :
  let relative_speed := (speed1 + speed2) * (1000 / 3600)
  let total_distance := relative_speed * passing_time
  let train_length := total_distance / 2
  train_length = 190 := by sorry

end train_length_problem_l4058_405850


namespace correct_calculation_l4058_405805

theorem correct_calculation : 
  (-2 + 3 = 1) ∧ 
  (-2 - 3 ≠ 1) ∧ 
  (-2 / (-1/2) ≠ 1) ∧ 
  ((-2)^3 ≠ -6) := by
  sorry

end correct_calculation_l4058_405805


namespace sum_of_solutions_quadratic_l4058_405801

theorem sum_of_solutions_quadratic (x : ℝ) : 
  (∃ r s : ℝ, (25 - 10*r - r^2 = 0) ∧ (25 - 10*s - s^2 = 0) ∧ (r + s = -10)) :=
by sorry

end sum_of_solutions_quadratic_l4058_405801


namespace power_of_power_l4058_405842

theorem power_of_power : (3^4)^2 = 6561 := by
  sorry

end power_of_power_l4058_405842


namespace subcommittees_with_coach_count_l4058_405890

def total_members : ℕ := 12
def coach_members : ℕ := 5
def subcommittee_size : ℕ := 5

def subcommittees_with_coach : ℕ := Nat.choose total_members subcommittee_size - Nat.choose (total_members - coach_members) subcommittee_size

theorem subcommittees_with_coach_count : subcommittees_with_coach = 771 := by
  sorry

end subcommittees_with_coach_count_l4058_405890


namespace fraction_inequality_l4058_405867

theorem fraction_inequality (a b c d : ℝ) 
  (h1 : a > b) (h2 : b > 0) (h3 : 0 > c) (h4 : c > d) : 
  c / a - d / b > 0 := by
  sorry

end fraction_inequality_l4058_405867


namespace collinear_vectors_x_value_l4058_405856

/-- Two vectors are collinear if one is a scalar multiple of the other -/
def collinear (a b : ℝ × ℝ) : Prop :=
  ∃ k : ℝ, b = (k * a.1, k * a.2)

/-- Given that vector a = (2, 4) is collinear with vector b = (x, 6), prove that x = 3 -/
theorem collinear_vectors_x_value :
  ∀ x : ℝ, collinear (2, 4) (x, 6) → x = 3 :=
by
  sorry

end collinear_vectors_x_value_l4058_405856


namespace smallest_solution_of_quartic_l4058_405830

theorem smallest_solution_of_quartic (x : ℝ) :
  x^4 - 50*x^2 + 576 = 0 →
  x ≥ -Real.sqrt 26 ∧ 
  ∃ y : ℝ, y^4 - 50*y^2 + 576 = 0 ∧ y = -Real.sqrt 26 :=
by sorry

end smallest_solution_of_quartic_l4058_405830


namespace trigonometric_simplification_l4058_405812

theorem trigonometric_simplification (x y : ℝ) :
  Real.sin x ^ 2 + Real.sin (x + y) ^ 2 - 2 * Real.sin x * Real.sin y * Real.sin (x + y) = Real.sin y ^ 2 := by
  sorry

end trigonometric_simplification_l4058_405812


namespace polygon_triangulation_l4058_405849

/-- Given an n-sided polygon divided into triangles by non-intersecting diagonals,
    this theorem states that the number of triangles with exactly two sides
    as edges of the original polygon is at least 2. -/
theorem polygon_triangulation (n : ℕ) (h : n ≥ 3) :
  ∃ (k₀ k₁ k₂ : ℕ),
    k₀ + k₁ + k₂ = n - 2 ∧
    k₁ + 2 * k₂ = n ∧
    k₂ ≥ 2 :=
by sorry

end polygon_triangulation_l4058_405849


namespace log_27_3_l4058_405807

theorem log_27_3 : Real.log 3 / Real.log 27 = 1 / 3 := by
  -- Define 27 as 3³
  have h : 27 = 3^3 := by norm_num
  -- Proof goes here
  sorry

end log_27_3_l4058_405807


namespace fourth_animal_is_sheep_l4058_405829

/-- Represents the different types of animals -/
inductive Animal
  | Horse
  | Cow
  | Pig
  | Sheep
  | Rabbit
  | Squirrel

/-- The sequence of animals entering the fence -/
def animalSequence : List Animal :=
  [Animal.Horse, Animal.Cow, Animal.Pig, Animal.Sheep, Animal.Rabbit, Animal.Squirrel]

/-- Theorem stating that the 4th animal in the sequence is a sheep -/
theorem fourth_animal_is_sheep :
  animalSequence[3] = Animal.Sheep := by sorry

end fourth_animal_is_sheep_l4058_405829


namespace vectors_parallel_opposite_l4058_405837

/-- Given vectors a = (-1, 2) and b = (2, -4), prove that they are parallel and in opposite directions. -/
theorem vectors_parallel_opposite (a b : ℝ × ℝ) : 
  a = (-1, 2) → b = (2, -4) → ∃ k : ℝ, k < 0 ∧ b = k • a := by sorry

end vectors_parallel_opposite_l4058_405837


namespace factorial_ratio_l4058_405883

def factorial (n : ℕ) : ℕ := Nat.factorial n

theorem factorial_ratio : factorial 4 / factorial (4 - 3) = 24 := by
  sorry

end factorial_ratio_l4058_405883


namespace A_times_B_equals_result_l4058_405855

-- Define the sets A and B
def A : Set ℝ := {x | |x - 1/2| < 1}
def B : Set ℝ := {x | 1/x ≥ 1}

-- Define the operation ×
def times (X Y : Set ℝ) : Set ℝ := {x | x ∈ X ∪ Y ∧ x ∉ X ∩ Y}

-- State the theorem
theorem A_times_B_equals_result : 
  times A B = {x | -1/2 < x ∧ x ≤ 0 ∨ 1 < x ∧ x < 3/2} := by sorry

end A_times_B_equals_result_l4058_405855
