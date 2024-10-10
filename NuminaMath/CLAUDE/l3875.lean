import Mathlib

namespace rectangle_length_l3875_387574

theorem rectangle_length (square_side : ℝ) (rectangle_area : ℝ) : 
  square_side = 15 →
  rectangle_area = 216 →
  ∃ (rectangle_length rectangle_width : ℝ),
    4 * square_side = 2 * (rectangle_length + rectangle_width) ∧
    rectangle_length * rectangle_width = rectangle_area ∧
    rectangle_length = 18 := by
  sorry

end rectangle_length_l3875_387574


namespace square_area_larger_than_circle_l3875_387569

theorem square_area_larger_than_circle (R : ℝ) (h : R > 0) : 
  let AB := 2 * R * Real.sin (3 * Real.pi / 8)
  (AB ^ 2) > Real.pi * R ^ 2 := by
  sorry

end square_area_larger_than_circle_l3875_387569


namespace find_x_l3875_387555

theorem find_x (y : ℝ) (x : ℝ) (h1 : (12 : ℝ)^3 * 6^3 / x = y) (h2 : y = 864) : x = 432 := by
  sorry

end find_x_l3875_387555


namespace complement_of_A_in_U_l3875_387575

def U : Set Int := {-1, 0, 1, 2}
def A : Set Int := {-1, 1, 2}

theorem complement_of_A_in_U :
  (U \ A) = {0} := by sorry

end complement_of_A_in_U_l3875_387575


namespace problem_2_l3875_387516

def f (x a : ℝ) : ℝ := |x - a|

theorem problem_2 (a m n : ℝ) (hm : m > 0) (hn : n > 0) 
  (h_solution_set : ∀ x, f x a ≤ 2 ↔ -1 ≤ x ∧ x ≤ 3)
  (h_equation : m + 2*n = 2*m*n - 3*a) : 
  m + 2*n ≥ 6 := by
sorry

end problem_2_l3875_387516


namespace no_linear_term_implies_m_equals_12_l3875_387594

theorem no_linear_term_implies_m_equals_12 (m : ℝ) : 
  (∃ a b c : ℝ, (mx + 8) * (2 - 3*x) = a*x^2 + b*x + c ∧ b = 0) → m = 12 := by
  sorry

end no_linear_term_implies_m_equals_12_l3875_387594


namespace square_area_after_cut_l3875_387591

theorem square_area_after_cut (x : ℝ) : 
  x > 0 → 
  x^2 - 2*x = 80 → 
  x^2 = 100 := by
sorry

end square_area_after_cut_l3875_387591


namespace plate_cup_cost_l3875_387500

/-- Given that 100 paper plates and 200 paper cups cost $6.00 in total,
    prove that 20 paper plates and 40 paper cups cost $1.20. -/
theorem plate_cup_cost (plate_cost cup_cost : ℝ) 
  (h : 100 * plate_cost + 200 * cup_cost = 6) :
  20 * plate_cost + 40 * cup_cost = 1.2 := by
  sorry

end plate_cup_cost_l3875_387500


namespace hula_hoop_problem_l3875_387582

/-- Hula hoop problem -/
theorem hula_hoop_problem (nancy casey morgan alex : ℕ) : 
  nancy = 10 →
  casey = nancy - 3 →
  morgan = 3 * casey →
  alex = (nancy + casey + morgan) / 2 →
  alex = 19 := by sorry

end hula_hoop_problem_l3875_387582


namespace problem_solution_l3875_387583

theorem problem_solution (x y : ℝ) 
  (h1 : x ≠ 0) 
  (h2 : x / 3 = y^2) 
  (h3 : x / 5 = 5*y) : 
  x = 625 / 3 := by
  sorry

end problem_solution_l3875_387583


namespace bakery_customers_l3875_387512

theorem bakery_customers (total_pastries : ℕ) (regular_customers : ℕ) (pastry_difference : ℕ) :
  total_pastries = 392 →
  regular_customers = 28 →
  pastry_difference = 6 →
  ∃ (actual_customers : ℕ),
    actual_customers * (total_pastries / regular_customers - pastry_difference) = total_pastries ∧
    actual_customers = 49 := by
  sorry

end bakery_customers_l3875_387512


namespace father_reaches_mom_age_in_three_years_l3875_387560

/-- Represents the ages and time in the problem -/
structure AgesProblem where
  talia_future_age : ℕ      -- Talia's age in 7 years
  talia_future_years : ℕ    -- Years until Talia reaches future_age
  father_current_age : ℕ    -- Talia's father's current age
  mom_age_ratio : ℕ         -- Ratio of mom's age to Talia's current age

/-- Calculates the years until Talia's father reaches Talia's mom's current age -/
def years_until_father_reaches_mom_age (p : AgesProblem) : ℕ :=
  let talia_current_age := p.talia_future_age - p.talia_future_years
  let mom_current_age := talia_current_age * p.mom_age_ratio
  mom_current_age - p.father_current_age

/-- Theorem stating the solution to the problem -/
theorem father_reaches_mom_age_in_three_years (p : AgesProblem) 
    (h1 : p.talia_future_age = 20)
    (h2 : p.talia_future_years = 7)
    (h3 : p.father_current_age = 36)
    (h4 : p.mom_age_ratio = 3) :
  years_until_father_reaches_mom_age p = 3 := by
  sorry


end father_reaches_mom_age_in_three_years_l3875_387560


namespace arbor_day_tree_planting_l3875_387557

theorem arbor_day_tree_planting 
  (original_trees_per_row : ℕ) 
  (original_rows : ℕ) 
  (new_rows : ℕ) 
  (h1 : original_trees_per_row = 20) 
  (h2 : original_rows = 18) 
  (h3 : new_rows = 10) : 
  (original_trees_per_row * original_rows) / new_rows = 36 := by
sorry

end arbor_day_tree_planting_l3875_387557


namespace A_intersect_B_l3875_387564

def A : Set ℕ := {0, 2, 4}

def B : Set ℕ := {y | ∃ x ∈ A, y = 2^x}

theorem A_intersect_B : A ∩ B = {4} := by sorry

end A_intersect_B_l3875_387564


namespace least_possible_value_a_2008_l3875_387540

theorem least_possible_value_a_2008 (a : ℕ → ℤ) 
  (h_increasing : ∀ n : ℕ, n ≥ 1 → a n < a (n + 1))
  (h_inequality : ∀ i j k l : ℕ, 1 ≤ i ∧ i < j ∧ j ≤ k ∧ k < l ∧ i + l = j + k → 
    a i + a l > a j + a k) :
  a 2008 ≥ 2015029 := by
sorry

end least_possible_value_a_2008_l3875_387540


namespace worker_ant_ratio_l3875_387587

theorem worker_ant_ratio (total_ants : ℕ) (female_worker_ants : ℕ) 
  (h1 : total_ants = 110)
  (h2 : female_worker_ants = 44)
  (h3 : (female_worker_ants : ℚ) / (female_worker_ants / 0.8 : ℚ) = 0.8) :
  (female_worker_ants / 0.8 : ℚ) / (total_ants : ℚ) = 1 / 2 := by
  sorry

end worker_ant_ratio_l3875_387587


namespace geometric_sequence_second_term_l3875_387552

theorem geometric_sequence_second_term :
  ∀ (a : ℕ+) (r : ℕ+),
    a = 5 →
    a * r^4 = 1280 →
    a * r = 20 :=
by
  sorry

end geometric_sequence_second_term_l3875_387552


namespace star_four_six_l3875_387507

-- Define the star operation
def star (a b : ℝ) : ℝ := a^2 + 2*a*b + b^2

-- Theorem statement
theorem star_four_six : star 4 6 = 100 := by
  sorry

end star_four_six_l3875_387507


namespace earth_surface_area_scientific_notation_l3875_387515

/-- The surface area of the Earth in square kilometers -/
def earth_surface_area : ℝ := 510000000

/-- Scientific notation representation of a real number -/
structure ScientificNotation where
  coefficient : ℝ
  exponent : ℤ
  valid : 1 ≤ |coefficient| ∧ |coefficient| < 10

/-- Conversion of a real number to scientific notation -/
def to_scientific_notation (x : ℝ) : ScientificNotation :=
  sorry

theorem earth_surface_area_scientific_notation :
  to_scientific_notation earth_surface_area = ScientificNotation.mk 5.1 8 sorry := by
  sorry

end earth_surface_area_scientific_notation_l3875_387515


namespace seashell_count_l3875_387544

theorem seashell_count (joan_shells jessica_shells : ℕ) 
  (h1 : joan_shells = 6) 
  (h2 : jessica_shells = 8) : 
  joan_shells + jessica_shells = 14 := by
  sorry

end seashell_count_l3875_387544


namespace wedding_decoration_cost_l3875_387588

/-- Calculates the total cost of decorations for a wedding reception --/
def total_decoration_cost (num_tables : ℕ) 
                          (tablecloth_cost : ℕ) 
                          (place_setting_cost : ℕ) 
                          (place_settings_per_table : ℕ) 
                          (roses_per_centerpiece : ℕ) 
                          (rose_cost : ℕ) 
                          (lilies_per_centerpiece : ℕ) 
                          (lily_cost : ℕ) : ℕ :=
  num_tables * (tablecloth_cost + 
                place_settings_per_table * place_setting_cost + 
                roses_per_centerpiece * rose_cost + 
                lilies_per_centerpiece * lily_cost)

/-- Theorem stating that the total decoration cost for the given conditions is $3500 --/
theorem wedding_decoration_cost : 
  total_decoration_cost 20 25 10 4 10 5 15 4 = 3500 := by
  sorry

end wedding_decoration_cost_l3875_387588


namespace parallelogram_area_example_l3875_387578

/-- The area of a parallelogram with given base and height -/
def parallelogram_area (base height : ℝ) : ℝ := base * height

theorem parallelogram_area_example : 
  parallelogram_area 10 20 = 200 := by sorry

end parallelogram_area_example_l3875_387578


namespace volleyball_not_basketball_l3875_387519

theorem volleyball_not_basketball (total : Nat) (basketball : Nat) (volleyball : Nat) (neither : Nat)
  (h1 : total = 40)
  (h2 : basketball = 15)
  (h3 : volleyball = 20)
  (h4 : neither = 10)
  (h5 : total = basketball + volleyball - (basketball + volleyball - (total - neither)) + neither) :
  volleyball - (basketball + volleyball - (total - neither)) = 15 := by
  sorry

end volleyball_not_basketball_l3875_387519


namespace interest_rate_calculation_l3875_387523

def total_sum : ℝ := 2743
def second_part : ℝ := 1688
def second_rate : ℝ := 0.05
def first_time : ℝ := 8
def second_time : ℝ := 3

theorem interest_rate_calculation (first_rate : ℝ) : 
  (total_sum - second_part) * first_rate * first_time = second_part * second_rate * second_time →
  first_rate = 0.03 := by
  sorry

end interest_rate_calculation_l3875_387523


namespace similar_triangles_side_length_l3875_387506

/-- Given two similar triangles PQR and STU, prove that if PQ = 12, QR = 10, and ST = 18, then TU = 15 -/
theorem similar_triangles_side_length 
  (PQ QR ST TU : ℝ) 
  (h_similar : ∃ k : ℝ, k > 0 ∧ PQ = k * ST ∧ QR = k * TU) 
  (h_PQ : PQ = 12) 
  (h_QR : QR = 10) 
  (h_ST : ST = 18) : 
  TU = 15 := by
sorry

end similar_triangles_side_length_l3875_387506


namespace cookies_left_l3875_387567

theorem cookies_left (total_cookies : ℕ) (num_neighbors : ℕ) (intended_per_neighbor : ℕ) 
  (sarah_cookies : ℕ) (h1 : total_cookies = 150) (h2 : num_neighbors = 15) 
  (h3 : intended_per_neighbor = 10) (h4 : sarah_cookies = 12) : 
  total_cookies - (intended_per_neighbor * (num_neighbors - 1)) - sarah_cookies = 8 := by
  sorry

#check cookies_left

end cookies_left_l3875_387567


namespace chicken_problem_model_l3875_387576

/-- Represents the system of equations for the chicken buying problem -/
def chicken_equations (x y : ℕ) : Prop :=
  (8 * x - y = 3) ∧ (y - 7 * x = 4)

/-- Proves that the system of equations correctly models the given conditions -/
theorem chicken_problem_model (x y : ℕ) :
  (x > 0 ∧ y > 0) →
  (chicken_equations x y ↔
    (8 * x = y + 3 ∧ 7 * x + 4 = y)) :=
by sorry

end chicken_problem_model_l3875_387576


namespace square_of_negative_product_l3875_387568

theorem square_of_negative_product (a b : ℝ) : (-2 * a * b)^2 = 4 * a^2 * b^2 := by
  sorry

end square_of_negative_product_l3875_387568


namespace greatest_multiple_of_four_under_cube_root_2000_l3875_387585

theorem greatest_multiple_of_four_under_cube_root_2000 :
  ∃ (x : ℕ), x > 0 ∧ 4 ∣ x ∧ x^3 < 2000 ∧
  ∀ (y : ℕ), y > 0 → 4 ∣ y → y^3 < 2000 → y ≤ x :=
by
  -- The proof goes here
  sorry

end greatest_multiple_of_four_under_cube_root_2000_l3875_387585


namespace x_cubed_coefficient_is_one_l3875_387556

-- Define the polynomial expression
def poly (x : ℝ) : ℝ := 2 * (x^3 - 2*x^2) + 3 * (x^2 - x^3 + x^4) - (5*x^4 - 2*x^3)

-- Theorem stating that the coefficient of x^3 in the expanded form of poly is 1
theorem x_cubed_coefficient_is_one :
  ∃ a b c d, ∀ x, poly x = a*x^4 + b*x^3 + c*x^2 + d*x ∧ b = 1 :=
by sorry

end x_cubed_coefficient_is_one_l3875_387556


namespace tangent_circle_circumference_l3875_387596

-- Define the geometric configuration
structure GeometricConfig where
  -- Centers of the arcs
  A : Point
  B : Point
  -- Points on the arcs
  C : Point
  -- Radii of the arcs
  r1 : ℝ
  r2 : ℝ
  -- Angle subtended by arc AC at center B
  angle_ACB : ℝ
  -- Length of arc BC
  length_BC : ℝ
  -- Radius of the tangent circle
  r : ℝ

-- State the theorem
theorem tangent_circle_circumference (config : GeometricConfig) 
  (h1 : config.angle_ACB = 75 * π / 180)
  (h2 : config.length_BC = 18)
  (h3 : config.r1 = 54 / π)
  (h4 : config.r2 = 216 / (5 * π))
  (h5 : config.r = 30 / π) : 
  2 * π * config.r = 60 := by
  sorry


end tangent_circle_circumference_l3875_387596


namespace polynomial_product_expansion_l3875_387573

theorem polynomial_product_expansion (x : ℝ) : 
  (7 * x^2 + 5 * x - 3) * (3 * x^3 + 2 * x^2 - x + 4) = 
  21 * x^5 + 29 * x^4 - 6 * x^3 + 17 * x^2 + 23 * x - 12 := by
  sorry

end polynomial_product_expansion_l3875_387573


namespace festival_profit_margin_is_five_percent_l3875_387534

/-- Represents the pricing and profit information for an item -/
structure ItemPricing where
  regular_discount : ℝ
  regular_profit_margin : ℝ

/-- Calculates the profit margin during a "buy one get one free" offer -/
def festival_profit_margin (item : ItemPricing) : ℝ :=
  -- Definition to be filled
  sorry

/-- Theorem stating the profit margin during the shopping festival -/
theorem festival_profit_margin_is_five_percent (item : ItemPricing) 
  (h1 : item.regular_discount = 0.3)
  (h2 : item.regular_profit_margin = 0.47) :
  festival_profit_margin item = 0.05 := by
  sorry

end festival_profit_margin_is_five_percent_l3875_387534


namespace purely_imaginary_complex_number_l3875_387546

theorem purely_imaginary_complex_number (m : ℝ) :
  (3 * m ^ 2 - 8 * m - 3 : ℂ) + (m ^ 2 - 4 * m + 3 : ℂ) * Complex.I = Complex.I * ((m ^ 2 - 4 * m + 3 : ℝ) : ℂ) →
  m = -1/3 :=
by sorry

end purely_imaginary_complex_number_l3875_387546


namespace f_24_18_mod_89_l3875_387528

/-- The function f(x) = x^2 - 2 -/
def f (x : ℤ) : ℤ := x^2 - 2

/-- f^n denotes f applied n times -/
def f_iter (n : ℕ) : ℤ → ℤ :=
  match n with
  | 0 => id
  | n + 1 => f ∘ f_iter n

/-- The main theorem stating that f^24(18) ≡ 47 (mod 89) -/
theorem f_24_18_mod_89 : f_iter 24 18 ≡ 47 [ZMOD 89] := by
  sorry


end f_24_18_mod_89_l3875_387528


namespace locust_jump_symmetry_l3875_387539

/-- A locust on a line -/
structure Locust where
  position : ℝ

/-- A configuration of locusts on a line -/
def LocustConfiguration := List Locust

/-- A property that can be achieved by a locust configuration -/
def ConfigurationProperty := LocustConfiguration → Prop

/-- Jumping to the right -/
def jumpRight (config : LocustConfiguration) : LocustConfiguration := sorry

/-- Jumping to the left -/
def jumpLeft (config : LocustConfiguration) : LocustConfiguration := sorry

/-- Two locusts are 1 mm apart -/
def twoLocustsOneMillimeterApart (config : LocustConfiguration) : Prop := sorry

theorem locust_jump_symmetry 
  (initial_config : LocustConfiguration) 
  (h : ∃ (final_config : LocustConfiguration), 
       twoLocustsOneMillimeterApart final_config ∧ 
       ∃ (n : ℕ), final_config = (jumpRight^[n]) initial_config) :
  ∃ (left_final_config : LocustConfiguration), 
    twoLocustsOneMillimeterApart left_final_config ∧ 
    ∃ (m : ℕ), left_final_config = (jumpLeft^[m]) initial_config := 
by sorry

end locust_jump_symmetry_l3875_387539


namespace circle_properties_l3875_387530

-- Define the points
def O : ℝ × ℝ := (0, 0)
def M : ℝ × ℝ := (1, 1)
def N : ℝ × ℝ := (4, 2)

-- Define the circle equation
def circle_equation (x y : ℝ) := x^2 + y^2 - 4*x + 3*y

-- Define the center and radius
def center : ℝ × ℝ := (4, -3)
def radius : ℝ := 5

theorem circle_properties :
  (circle_equation O.1 O.2 = 0) ∧
  (circle_equation M.1 M.2 = 0) ∧
  (circle_equation N.1 N.2 = 0) ∧
  (∀ (x y : ℝ), circle_equation x y = 0 ↔ (x - center.1)^2 + (y - center.2)^2 = radius^2) :=
by sorry

end circle_properties_l3875_387530


namespace simplify_complex_fraction_l3875_387558

theorem simplify_complex_fraction (a b : ℝ) : 
  ((a - b)^2 + a*b) / ((a + b)^2 - a*b) / 
  ((a^5 + b^5 + a^2*b^3 + a^3*b^2) / 
   ((a^3 + b^3 + a^2*b + a*b^2) * (a^3 - b^3))) = a - b :=
by sorry

end simplify_complex_fraction_l3875_387558


namespace f_equals_g_l3875_387551

-- Define the functions
def f (x : ℝ) : ℝ := x
def g (x : ℝ) : ℝ := (x^3)^(1/3)

-- Statement to prove
theorem f_equals_g : ∀ x : ℝ, f x = g x := by
  sorry

end f_equals_g_l3875_387551


namespace average_tree_height_l3875_387545

def tree_heights : List ℝ := [1000, 500, 500, 1200]

theorem average_tree_height : (tree_heights.sum / tree_heights.length : ℝ) = 800 := by
  sorry

end average_tree_height_l3875_387545


namespace equation_satisfied_at_one_l3875_387538

/-- The function f(x) = 3x - 5 -/
def f (x : ℝ) : ℝ := 3 * x - 5

/-- Theorem stating that the equation 2 * [f(x)] - 16 = f(x - 6) is satisfied when x = 1 -/
theorem equation_satisfied_at_one :
  2 * (f 1) - 16 = f (1 - 6) := by sorry

end equation_satisfied_at_one_l3875_387538


namespace wednesday_sales_proof_l3875_387541

def initial_stock : ℕ := 1300
def monday_sales : ℕ := 75
def tuesday_sales : ℕ := 50
def thursday_sales : ℕ := 78
def friday_sales : ℕ := 135
def unsold_percentage : ℚ := 69.07692307692308

theorem wednesday_sales_proof :
  ∃ (wednesday_sales : ℕ),
    (initial_stock : ℚ) * (1 - unsold_percentage / 100) =
    (monday_sales + tuesday_sales + wednesday_sales + thursday_sales + friday_sales : ℚ) ∧
    wednesday_sales = 64 := by sorry

end wednesday_sales_proof_l3875_387541


namespace smallest_distance_to_i_l3875_387532

theorem smallest_distance_to_i (w : ℂ) (h : Complex.abs (w^2 - 3) = Complex.abs (w * (2*w + 3*Complex.I))) :
  ∃ (min_dist : ℝ), 
    (∀ w', Complex.abs (w'^2 - 3) = Complex.abs (w' * (2*w' + 3*Complex.I)) → 
      Complex.abs (w' - Complex.I) ≥ min_dist) ∧
    min_dist = Complex.abs ((Real.sqrt 3 - Real.sqrt 6) / 3) :=
by sorry

end smallest_distance_to_i_l3875_387532


namespace coordinates_not_on_C_do_not_satisfy_F_l3875_387537

-- Define the curve C as a set of points in R²
def C : Set (ℝ × ℝ) := sorry

-- Define the function F
def F : ℝ → ℝ → ℝ := sorry

-- Theorem statement
theorem coordinates_not_on_C_do_not_satisfy_F :
  (∀ x y, F x y = 0 → (x, y) ∈ C) →
  ∀ x y, (x, y) ∉ C → F x y ≠ 0 := by
  sorry

end coordinates_not_on_C_do_not_satisfy_F_l3875_387537


namespace sum_of_x_solutions_is_zero_l3875_387503

theorem sum_of_x_solutions_is_zero :
  ∀ (x₁ x₂ : ℝ),
  (∃ y : ℝ, y = 8 ∧ x₁^2 + y^2 = 145) ∧
  (∃ y : ℝ, y = 8 ∧ x₂^2 + y^2 = 145) ∧
  (∀ x : ℝ, (∃ y : ℝ, y = 8 ∧ x^2 + y^2 = 145) → (x = x₁ ∨ x = x₂)) →
  x₁ + x₂ = 0 := by
sorry

end sum_of_x_solutions_is_zero_l3875_387503


namespace trig_identity_l3875_387508

theorem trig_identity (θ : Real) (h : θ ≠ 0) (h2 : θ ≠ π/2) : 
  (Real.tan θ)^2 - (Real.sin θ)^2 = (Real.tan θ)^2 * (Real.sin θ)^2 := by
  sorry

end trig_identity_l3875_387508


namespace ab_equals_zero_l3875_387593

theorem ab_equals_zero (a b : ℝ) 
  (h1 : (2 : ℝ) ^ a = (2 : ℝ) ^ (2 * (b + 1)))
  (h2 : (7 : ℝ) ^ b = (7 : ℝ) ^ (a - 2)) : 
  a * b = 0 := by
sorry

end ab_equals_zero_l3875_387593


namespace special_function_properties_l3875_387531

/-- A function satisfying the given properties -/
def special_function (f : ℝ → ℝ) : Prop :=
  (∃ x, f x ≠ 0) ∧
  (∀ a b : ℝ, f (a * b) = a * f b + b * f a) ∧
  f (1 / 2) = 1

theorem special_function_properties (f : ℝ → ℝ) (h : special_function f) :
  f (1 / 4) = 1 ∧
  f (1 / 8) = 3 / 4 ∧
  f (1 / 16) = 1 / 2 ∧
  ∀ n : ℕ, n > 0 → f (2 ^ (-n : ℝ)) = n * (1 / 2) ^ (n - 1) :=
by sorry

end special_function_properties_l3875_387531


namespace smallest_natural_numbers_for_nested_root_l3875_387529

theorem smallest_natural_numbers_for_nested_root (a b : ℕ) : 
  (b > 1) → 
  (Real.sqrt (a * Real.sqrt (a * Real.sqrt a)) = b) → 
  (∀ a' b' : ℕ, b' > 1 → Real.sqrt (a' * Real.sqrt (a' * Real.sqrt a')) = b' → a ≤ a' ∧ b ≤ b') →
  a = 256 ∧ b = 128 := by
sorry

end smallest_natural_numbers_for_nested_root_l3875_387529


namespace max_planes_with_six_points_l3875_387520

/-- A point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- A plane in 3D space -/
structure Plane3D where
  a : ℝ
  b : ℝ
  c : ℝ
  d : ℝ

/-- Check if a point lies on a plane -/
def pointOnPlane (pt : Point3D) (pl : Plane3D) : Prop :=
  pl.a * pt.x + pl.b * pt.y + pl.c * pt.z + pl.d = 0

/-- Check if four points are collinear -/
def areCollinear (p1 p2 p3 p4 : Point3D) : Prop :=
  ∃ (a b c d : ℝ), ∀ (x y z : ℝ),
    a*x + b*y + c*z + d = 0 ↔ (x = p1.x ∧ y = p1.y ∧ z = p1.z) ∨
                            (x = p2.x ∧ y = p2.y ∧ z = p2.z) ∨
                            (x = p3.x ∧ y = p3.y ∧ z = p3.z) ∨
                            (x = p4.x ∧ y = p4.y ∧ z = p4.z)

/-- Main theorem -/
theorem max_planes_with_six_points
  (points : Fin 6 → Point3D)
  (h_not_collinear : ∀ (i j k l : Fin 6), i ≠ j → i ≠ k → i ≠ l → j ≠ k → j ≠ l → k ≠ l →
                     ¬ areCollinear (points i) (points j) (points k) (points l)) :
  ∃ (planes : Fin 6 → Plane3D),
    (∀ (i : Fin 6), ∃ (p1 p2 p3 p4 : Fin 6),
      p1 ≠ p2 ∧ p1 ≠ p3 ∧ p1 ≠ p4 ∧ p2 ≠ p3 ∧ p2 ≠ p4 ∧ p3 ≠ p4 ∧
      pointOnPlane (points p1) (planes i) ∧
      pointOnPlane (points p2) (planes i) ∧
      pointOnPlane (points p3) (planes i) ∧
      pointOnPlane (points p4) (planes i)) ∧
    (∀ (newPlane : Plane3D),
      (∃ (p1 p2 p3 p4 : Fin 6),
        p1 ≠ p2 ∧ p1 ≠ p3 ∧ p1 ≠ p4 ∧ p2 ≠ p3 ∧ p2 ≠ p4 ∧ p3 ≠ p4 ∧
        pointOnPlane (points p1) newPlane ∧
        pointOnPlane (points p2) newPlane ∧
        pointOnPlane (points p3) newPlane ∧
        pointOnPlane (points p4) newPlane) →
      ∃ (i : Fin 6), newPlane = planes i) :=
by
  sorry


end max_planes_with_six_points_l3875_387520


namespace maria_students_l3875_387586

/-- The number of students in Maria's high school -/
def M : ℕ := sorry

/-- The number of students in Jackson's high school -/
def J : ℕ := sorry

/-- Maria's high school has 4 times as many students as Jackson's high school -/
axiom maria_jackson_ratio : M = 4 * J

/-- The total number of students in both high schools is 3600 -/
axiom total_students : M + J = 3600

/-- Theorem: Maria's high school has 2880 students -/
theorem maria_students : M = 2880 := by sorry

end maria_students_l3875_387586


namespace rotation_transformation_l3875_387548

structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ

def ABC : Triangle := ⟨(0, 0), (0, 10), (20, 0)⟩
def DEF : Triangle := ⟨(20, 10), (30, 10), (20, 2)⟩

def rotatePoint (center : ℝ × ℝ) (angle : ℝ) (point : ℝ × ℝ) : ℝ × ℝ := sorry

def rotateTriangle (center : ℝ × ℝ) (angle : ℝ) (t : Triangle) : Triangle := sorry

theorem rotation_transformation (n x y : ℝ) 
  (h1 : 0 < n ∧ n < 180) 
  (h2 : rotateTriangle (x, y) n ABC = DEF) : 
  n + x + y = 92 := by sorry

end rotation_transformation_l3875_387548


namespace clives_box_balls_l3875_387525

/-- The number of balls in Clive's box -/
def total_balls (blue red green yellow : ℕ) : ℕ := blue + red + green + yellow

/-- Theorem: The total number of balls in Clive's box is 36 -/
theorem clives_box_balls : 
  ∃ (blue red green yellow : ℕ),
    blue = 6 ∧ 
    red = 4 ∧ 
    green = 3 * blue ∧ 
    yellow = 2 * red ∧ 
    total_balls blue red green yellow = 36 := by
  sorry

end clives_box_balls_l3875_387525


namespace division_reciprocal_l3875_387584

theorem division_reciprocal (a b c d e : ℝ) (ha : a ≠ 0) (hbcde : b - c + d - e ≠ 0) :
  a / (b - c + d - e) = 1 / ((b - c + d - e) / a) := by
  sorry

end division_reciprocal_l3875_387584


namespace equation_root_l3875_387549

theorem equation_root : ∃ x : ℝ, (18 / (x^3 - 8) - 2 / (x - 2) = 1) ∧ x = 2 := by
  sorry

end equation_root_l3875_387549


namespace area_difference_of_square_fields_l3875_387553

/-- Given two square fields where the second field's side length is 1% longer than the first,
    and the area of the first field is 1 hectare (10,000 square meters),
    prove that the difference in area between the two fields is 201 square meters. -/
theorem area_difference_of_square_fields (a : ℝ) : 
  a^2 = 10000 → (1.01 * a)^2 - a^2 = 201 := by
  sorry

end area_difference_of_square_fields_l3875_387553


namespace absolute_value_inequality_l3875_387595

theorem absolute_value_inequality (a : ℝ) :
  (∀ x : ℝ, |x + 3| - |x + 1| ≤ a) → a ≥ 2 := by
  sorry

end absolute_value_inequality_l3875_387595


namespace fish_pond_estimate_l3875_387563

theorem fish_pond_estimate (initial_marked : ℕ) (second_catch : ℕ) (marked_in_second : ℕ) :
  initial_marked = 40 →
  second_catch = 100 →
  marked_in_second = 5 →
  (second_catch : ℚ) / marked_in_second = (800 : ℚ) / initial_marked :=
by sorry

end fish_pond_estimate_l3875_387563


namespace tank_full_time_l3875_387501

/-- Represents the state of a water tank system -/
structure TankSystem where
  capacity : ℕ
  pipeA_rate : ℕ
  pipeB_rate : ℕ
  pipeC_rate : ℤ

/-- Calculates the time needed to fill the tank -/
def time_to_fill (system : TankSystem) : ℕ :=
  sorry

/-- Theorem stating that the tank will be full after 56 minutes -/
theorem tank_full_time (system : TankSystem) 
  (h1 : system.capacity = 950)
  (h2 : system.pipeA_rate = 40)
  (h3 : system.pipeB_rate = 30)
  (h4 : system.pipeC_rate = -20) : 
  time_to_fill system = 56 :=
  sorry

end tank_full_time_l3875_387501


namespace triangle_abc_properties_l3875_387572

theorem triangle_abc_properties (a b c : ℝ) (A B C : ℝ) :
  a = 4 →
  c = 2 * Real.sqrt 2 →
  Real.cos A = -(Real.sqrt 2) / 4 →
  b = 2 ∧
  Real.sin C = (Real.sqrt 7) / 4 ∧
  Real.cos (2 * A + π / 6) = (Real.sqrt 7 - 3 * Real.sqrt 3) / 8 :=
by sorry


end triangle_abc_properties_l3875_387572


namespace adam_books_before_shopping_l3875_387589

/-- Calculates the number of books Adam had before his shopping trip -/
def books_before_shopping (shelves : ℕ) (avg_books_per_shelf : ℕ) (new_books : ℕ) (leftover : ℕ) : ℕ :=
  shelves * avg_books_per_shelf - (new_books - leftover)

/-- Theorem stating that Adam had 56 books before his shopping trip -/
theorem adam_books_before_shopping :
  books_before_shopping 4 20 26 2 = 56 := by
  sorry

#eval books_before_shopping 4 20 26 2

end adam_books_before_shopping_l3875_387589


namespace line_direction_vector_l3875_387571

-- Define the two points on the line
def point1 : ℝ × ℝ := (-3, 0)
def point2 : ℝ × ℝ := (0, 3)

-- Define the direction vector
def direction_vector : ℝ × ℝ := (3, 3)

-- Theorem statement
theorem line_direction_vector :
  (point2.1 - point1.1, point2.2 - point1.2) = direction_vector :=
sorry

end line_direction_vector_l3875_387571


namespace upward_translation_4_units_l3875_387562

/-- A point in 2D space -/
structure Point2D where
  x : ℝ
  y : ℝ

/-- A translation in 2D space -/
structure Translation2D where
  dx : ℝ
  dy : ℝ

/-- Apply a translation to a point -/
def applyTranslation (p : Point2D) (t : Translation2D) : Point2D :=
  { x := p.x + t.dx, y := p.y + t.dy }

theorem upward_translation_4_units 
  (M : Point2D)
  (N : Point2D)
  (h1 : M.x = -1 ∧ M.y = -1)
  (h2 : N.x = -1 ∧ N.y = 3) :
  ∃ (t : Translation2D), t.dx = 0 ∧ t.dy = 4 ∧ applyTranslation M t = N :=
sorry

end upward_translation_4_units_l3875_387562


namespace rent_increase_problem_l3875_387554

theorem rent_increase_problem (initial_average : ℝ) (new_average : ℝ) (num_friends : ℕ) 
  (increase_percentage : ℝ) (h1 : initial_average = 800) (h2 : new_average = 880) 
  (h3 : num_friends = 4) (h4 : increase_percentage = 0.2) : 
  ∃ (original_rent : ℝ), 
    (num_friends * new_average - num_friends * initial_average) / increase_percentage = original_rent ∧ 
    original_rent = 1600 := by
  sorry

end rent_increase_problem_l3875_387554


namespace ellipse_area_l3875_387517

def ellipse_equation (x y : ℝ) : Prop :=
  2 * x^2 + 8 * x + 3 * y^2 - 9 * y + 12 = 0

theorem ellipse_area : 
  ∃ (A : ℝ), A = Real.pi * Real.sqrt 6 / 6 ∧ 
  ∀ (x y : ℝ), ellipse_equation x y → A = Real.pi * Real.sqrt ((1 / 2) * (1 / 3)) :=
by sorry

end ellipse_area_l3875_387517


namespace velocity_zero_at_two_l3875_387547

-- Define the displacement function
def s (t : ℝ) : ℝ := -2 * t^2 + 8 * t

-- Define the velocity function (derivative of displacement)
def v (t : ℝ) : ℝ := -4 * t + 8

-- Theorem: The time when velocity is 0 is equal to 2
theorem velocity_zero_at_two :
  ∃ t : ℝ, v t = 0 ∧ t = 2 :=
sorry

end velocity_zero_at_two_l3875_387547


namespace max_product_sum_2000_l3875_387513

theorem max_product_sum_2000 : 
  ∃ (a b : ℤ), a + b = 2000 ∧ 
    ∀ (x y : ℤ), x + y = 2000 → x * y ≤ a * b ∧ 
    a * b = 1000000 :=
sorry

end max_product_sum_2000_l3875_387513


namespace sum_x_z_equals_4036_l3875_387559

theorem sum_x_z_equals_4036 (x y z : ℝ) 
  (eq1 : x + y + z = 0)
  (eq2 : 2016 * x + 2017 * y + 2018 * z = 0)
  (eq3 : 2016^2 * x + 2017^2 * y + 2018^2 * z = 2018) :
  x + z = 4036 := by sorry

end sum_x_z_equals_4036_l3875_387559


namespace rectangle_to_square_cut_l3875_387598

theorem rectangle_to_square_cut (length width : ℝ) (h1 : length = 16) (h2 : width = 9) :
  ∃ (side : ℝ), side = 12 ∧ 
  2 * (side * side) = length * width ∧
  side ≤ length ∧ side ≤ width + (length - side) :=
by sorry

end rectangle_to_square_cut_l3875_387598


namespace total_apples_buyable_l3875_387579

def apple_cost : ℕ := 2
def emmy_money : ℕ := 200
def gerry_money : ℕ := 100

theorem total_apples_buyable : 
  (emmy_money + gerry_money) / apple_cost = 150 := by
sorry

end total_apples_buyable_l3875_387579


namespace smallest_shift_l3875_387514

-- Define a function that repeats every 15 units horizontally
def is_periodic_15 (f : ℝ → ℝ) : Prop :=
  ∀ x, f (x - 15) = f x

-- Define the property we're looking for
def satisfies_shift (f : ℝ → ℝ) (b : ℝ) : Prop :=
  ∀ x, f ((x - b) / 3) = f (x / 3)

-- State the theorem
theorem smallest_shift (f : ℝ → ℝ) (h : is_periodic_15 f) :
  ∃ b : ℝ, b > 0 ∧ satisfies_shift f b ∧ ∀ b' : ℝ, b' > 0 ∧ satisfies_shift f b' → b ≤ b' :=
sorry

end smallest_shift_l3875_387514


namespace square_field_area_l3875_387526

/-- The area of a square field with side length 20 meters is 400 square meters. -/
theorem square_field_area : ∀ (side_length : ℝ), side_length = 20 → side_length * side_length = 400 :=
by
  sorry

end square_field_area_l3875_387526


namespace simplify_polynomial_l3875_387570

theorem simplify_polynomial (x : ℝ) : 
  2*x*(4*x^3 + 3*x - 5) - 7*(2*x^2 + x - 8) = 8*x^4 - 8*x^2 - 17*x + 56 := by
  sorry

end simplify_polynomial_l3875_387570


namespace inverse_matrices_sum_l3875_387543

def A (a b c d e : ℝ) : Matrix (Fin 4) (Fin 4) ℝ :=
  !![a, 1, 0, b;
     0, 3, 2, 0;
     c, 4, d, 5;
     6, 0, 7, e]

def B (f g h : ℝ) : Matrix (Fin 4) (Fin 4) ℝ :=
  !![-7, f,  0, -15;
      g, -20, h,   0;
      0,  2,  5,   0;
      3,  0,  8,   6]

theorem inverse_matrices_sum (a b c d e f g h : ℝ) :
  A a b c d e * B f g h = 1 →
  a + b + c + d + e + f + g + h = 27 := by
  sorry

end inverse_matrices_sum_l3875_387543


namespace remainder_of_large_number_l3875_387518

theorem remainder_of_large_number (M : ℕ) (d : ℕ) (h : M = 123456789012 ∧ d = 252) :
  M % d = 228 := by
sorry

end remainder_of_large_number_l3875_387518


namespace subset_size_bound_l3875_387527

/-- Given a natural number n ≥ 2, we define a set A and a family of subsets S with certain properties. -/
theorem subset_size_bound (n : ℕ) (h_n : n ≥ 2) :
  ∃ (A : Finset ℕ) (S : Finset (Finset ℕ)),
    (A = Finset.range (2^(n+1) + 1)) ∧
    (S.card = 2^n) ∧
    (∀ s ∈ S, s ⊆ A) ∧
    (∀ (a b : Finset ℕ) (x y z : ℕ),
      a ∈ S → b ∈ S → x ∈ A → y ∈ A → z ∈ A →
      x < y → y < z → y ∈ a → z ∈ a → x ∈ b → z ∈ b →
      a.card < b.card) →
    ∃ s ∈ S, s.card ≤ 4 * n :=
by
  sorry


end subset_size_bound_l3875_387527


namespace exchange_count_l3875_387566

def number_of_people : ℕ := 10

def business_card_exchanges (n : ℕ) : ℕ := n.choose 2

theorem exchange_count : business_card_exchanges number_of_people = 45 := by
  sorry

end exchange_count_l3875_387566


namespace ellipse_chord_theorem_l3875_387510

/-- The ellipse type -/
structure Ellipse where
  a : ℝ
  b : ℝ

/-- The point type -/
structure Point where
  x : ℝ
  y : ℝ

/-- The chord type -/
structure Chord where
  p1 : Point
  p2 : Point

/-- The theorem statement -/
theorem ellipse_chord_theorem (e : Ellipse) (c : Chord) (F1 F2 : Point) :
  e.a = 5 →
  e.b = 4 →
  F1.x = -3 →
  F1.y = 0 →
  F2.x = 3 →
  F2.y = 0 →
  (c.p1.x^2 / 25 + c.p1.y^2 / 16 = 1) →
  (c.p2.x^2 / 25 + c.p2.y^2 / 16 = 1) →
  (c.p1.x - F1.x) * (c.p2.y - F1.y) = (c.p1.y - F1.y) * (c.p2.x - F1.x) →
  (Real.pi = 2 * Real.pi * (Real.sqrt (5 * 5 / 36))) →
  |c.p1.y - c.p2.y| = 5/3 := by
sorry

end ellipse_chord_theorem_l3875_387510


namespace tens_digit_of_6_pow_19_l3875_387542

-- Define a function to get the tens digit of a natural number
def tens_digit (n : ℕ) : ℕ := (n / 10) % 10

-- State the theorem
theorem tens_digit_of_6_pow_19 : tens_digit (6^19) = 9 := by
  sorry

end tens_digit_of_6_pow_19_l3875_387542


namespace solve_jury_duty_problem_l3875_387505

def jury_duty_problem (jury_selection_days : ℕ) (trial_multiplier : ℕ) (deliberation_full_days : ℕ) (total_days : ℕ) : Prop :=
  let trial_days : ℕ := trial_multiplier * jury_selection_days
  let selection_and_trial_days : ℕ := jury_selection_days + trial_days
  let actual_deliberation_days : ℕ := total_days - selection_and_trial_days
  let deliberation_hours : ℕ := deliberation_full_days * 24
  deliberation_hours / actual_deliberation_days = 16

theorem solve_jury_duty_problem : 
  jury_duty_problem 2 4 6 19 := by
  sorry

end solve_jury_duty_problem_l3875_387505


namespace puppies_difference_l3875_387550

/-- The number of puppies Yuri adopted in the first week -/
def first_week : ℕ := 20

/-- The number of puppies Yuri adopted in the second week -/
def second_week : ℕ := (2 * first_week) / 5

/-- The number of puppies Yuri adopted in the third week -/
def third_week : ℕ := 2 * second_week

/-- The total number of puppies Yuri has after four weeks -/
def total_puppies : ℕ := 74

/-- The number of puppies Yuri adopted in the fourth week -/
def fourth_week : ℕ := total_puppies - (first_week + second_week + third_week)

theorem puppies_difference : fourth_week - first_week = 10 := by
  sorry

end puppies_difference_l3875_387550


namespace stratified_sampling_model_c_l3875_387504

theorem stratified_sampling_model_c (total_units : ℕ) (model_c_units : ℕ) (sample_size : ℕ) :
  total_units = 1000 →
  model_c_units = 300 →
  sample_size = 60 →
  (model_c_units * sample_size) / total_units = 18 := by
  sorry

end stratified_sampling_model_c_l3875_387504


namespace gold_balance_fraction_is_one_third_l3875_387577

/-- Represents a credit card with a spending limit and balance. -/
structure CreditCard where
  limit : ℝ
  balance : ℝ

/-- Represents Sally's credit cards and their properties. -/
structure SallysCards where
  gold : CreditCard
  platinum : CreditCard
  gold_balance_fraction : ℝ
  platinum_balance_fraction : ℝ
  remaining_platinum_fraction : ℝ

/-- Theorem stating the fraction of the gold card's limit that represents the current balance. -/
theorem gold_balance_fraction_is_one_third
  (cards : SallysCards)
  (h1 : cards.platinum.limit = 2 * cards.gold.limit)
  (h2 : cards.gold.balance = cards.gold_balance_fraction * cards.gold.limit)
  (h3 : cards.platinum.balance = (1/4) * cards.platinum.limit)
  (h4 : cards.remaining_platinum_fraction = 0.5833333333333334)
  (h5 : cards.platinum.limit - (cards.platinum.balance + cards.gold.balance) =
        cards.remaining_platinum_fraction * cards.platinum.limit) :
  cards.gold_balance_fraction = 1/3 := by
  sorry

end gold_balance_fraction_is_one_third_l3875_387577


namespace max_area_enclosure_l3875_387511

/-- Represents a rectangular enclosure with length and width. -/
structure Enclosure where
  length : ℝ
  width : ℝ

/-- The perimeter of the enclosure is 500 feet. -/
def perimeterConstraint (e : Enclosure) : Prop :=
  2 * e.length + 2 * e.width = 500

/-- The length of the enclosure is at least 100 feet. -/
def minLengthConstraint (e : Enclosure) : Prop :=
  e.length ≥ 100

/-- The width of the enclosure is at least 60 feet. -/
def minWidthConstraint (e : Enclosure) : Prop :=
  e.width ≥ 60

/-- The area of the enclosure. -/
def area (e : Enclosure) : ℝ :=
  e.length * e.width

/-- Theorem stating that the maximum area of the enclosure satisfying all constraints is 15625 square feet. -/
theorem max_area_enclosure :
  ∃ (e : Enclosure),
    perimeterConstraint e ∧
    minLengthConstraint e ∧
    minWidthConstraint e ∧
    (∀ (e' : Enclosure),
      perimeterConstraint e' ∧
      minLengthConstraint e' ∧
      minWidthConstraint e' →
      area e' ≤ area e) ∧
    area e = 15625 :=
  sorry

end max_area_enclosure_l3875_387511


namespace N_q_odd_iff_prime_power_l3875_387524

/-- The number of integers a such that 0 < a < q/4 and gcd(a,q) = 1 -/
def N_q (q : ℕ) : ℕ :=
  (Finset.filter (fun a => a > 0 ∧ a < q / 4 ∧ Nat.gcd a q = 1) (Finset.range q)).card

/-- A prime p is congruent to 5 or 7 modulo 8 -/
def is_prime_5_or_7_mod_8 (p : ℕ) : Prop :=
  Nat.Prime p ∧ (p % 8 = 5 ∨ p % 8 = 7)

theorem N_q_odd_iff_prime_power (q : ℕ) (h_odd : Odd q) :
  Odd (N_q q) ↔ ∃ (p k : ℕ), q = p^k ∧ k > 0 ∧ is_prime_5_or_7_mod_8 p :=
sorry

end N_q_odd_iff_prime_power_l3875_387524


namespace lines_no_common_points_parallel_or_skew_l3875_387592

/-- Two lines in 3D space -/
structure Line3D where
  point : ℝ × ℝ × ℝ
  direction : ℝ × ℝ × ℝ

/-- Predicate to check if two lines have no common points -/
def NoCommonPoints (l1 l2 : Line3D) : Prop :=
  ∀ t s : ℝ, l1.point + t • l1.direction ≠ l2.point + s • l2.direction

/-- Predicate to check if two lines are parallel -/
def Parallel (l1 l2 : Line3D) : Prop :=
  ∃ k : ℝ, l1.direction = k • l2.direction

/-- Predicate to check if two lines are skew -/
def Skew (l1 l2 : Line3D) : Prop :=
  ¬ Parallel l1 l2 ∧ NoCommonPoints l1 l2

/-- Theorem stating that if two lines have no common points, they are either parallel or skew -/
theorem lines_no_common_points_parallel_or_skew (l1 l2 : Line3D) :
  NoCommonPoints l1 l2 → Parallel l1 l2 ∨ Skew l1 l2 :=
by
  sorry


end lines_no_common_points_parallel_or_skew_l3875_387592


namespace find_M_and_N_convex_polygon_diagonals_calculate_y_l3875_387597

-- Part 1 and 2
theorem find_M_and_N :
  ∃ (M N : ℕ),
    M < 10 ∧ N < 10 ∧
    258024 * 10 + M * 10 + 8 * 9 = 2111110 * N * 11 ∧
    M = 9 ∧ N = 2 := by sorry

-- Part 3
theorem convex_polygon_diagonals (n : ℕ) (h : n = 20) :
  (n * (n - 3)) / 2 = 170 := by sorry

-- Part 4
theorem calculate_y (a b : ℕ) (h1 : a = 99) (h2 : b = 49) :
  a * b + a + b + 1 = 4999 := by sorry

end find_M_and_N_convex_polygon_diagonals_calculate_y_l3875_387597


namespace min_value_expression_lower_bound_achievable_l3875_387561

theorem min_value_expression (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  (a + 3*c) / (a + 2*b + c) + 4*b / (a + b + 2*c) - 8*c / (a + b + 3*c) ≥ -17 + 12 * Real.sqrt 2 :=
by sorry

theorem lower_bound_achievable :
  ∃ (a b c : ℝ), a > 0 ∧ b > 0 ∧ c > 0 ∧
    (a + 3*c) / (a + 2*b + c) + 4*b / (a + b + 2*c) - 8*c / (a + b + 3*c) = -17 + 12 * Real.sqrt 2 :=
by sorry

end min_value_expression_lower_bound_achievable_l3875_387561


namespace exponential_decreasing_inequality_l3875_387533

theorem exponential_decreasing_inequality (m n : ℝ) (h1 : m > n) (h2 : n > 0) : 
  (0.3 : ℝ) ^ m < (0.3 : ℝ) ^ n := by
  sorry

end exponential_decreasing_inequality_l3875_387533


namespace john_buys_three_boxes_l3875_387536

/-- The number of times John plays paintball per month. -/
def plays_per_month : ℕ := 3

/-- The cost of one box of paintballs in dollars. -/
def cost_per_box : ℕ := 25

/-- The total amount John spends on paintballs per month in dollars. -/
def monthly_spending : ℕ := 225

/-- The number of boxes of paintballs John buys each time he plays. -/
def boxes_per_play : ℚ := monthly_spending / (plays_per_month * cost_per_box)

theorem john_buys_three_boxes : boxes_per_play = 3 := by
  sorry

end john_buys_three_boxes_l3875_387536


namespace distributive_property_l3875_387502

theorem distributive_property (x y : ℝ) : x * (1 + y) = x + x * y := by
  sorry

end distributive_property_l3875_387502


namespace correct_equation_l3875_387599

theorem correct_equation (x y : ℝ) : x * y - 2 * (x * y) = -(x * y) := by
  sorry

end correct_equation_l3875_387599


namespace installation_cost_is_6255_l3875_387581

/-- Calculates the installation cost for a refrigerator purchase --/
def calculate_installation_cost (purchase_price : ℚ) (discount_rate : ℚ) 
  (transport_cost : ℚ) (profit_rate : ℚ) (selling_price : ℚ) : ℚ :=
  let labelled_price := purchase_price / (1 - discount_rate)
  let total_cost := selling_price / (1 + profit_rate)
  total_cost - purchase_price - transport_cost

/-- Proves that the installation cost is 6255, given the problem conditions --/
theorem installation_cost_is_6255 :
  calculate_installation_cost 12500 0.20 125 0.18 18880 = 6255 := by
  sorry

end installation_cost_is_6255_l3875_387581


namespace item_list_price_equality_l3875_387580

theorem item_list_price_equality (list_price : ℝ) : 
  (0.15 * (list_price - 15) = 0.25 * (list_price - 25)) → list_price = 40 := by
  sorry

#check item_list_price_equality

end item_list_price_equality_l3875_387580


namespace polynomial_factorization_l3875_387590

theorem polynomial_factorization (x : ℝ) :
  (x^2 + 5*x + 4) * (x^2 + 11*x + 30) + (x^2 + 8*x - 10) =
  (x^2 + 8*x + 7) * (x^2 + 8*x + 19) := by
sorry

end polynomial_factorization_l3875_387590


namespace octagon_area_l3875_387535

theorem octagon_area (r : ℝ) (h : r = 3) : 
  let s := 2 * r * Real.sin (π / 8)
  let triangle_area := (1 / 2) * s^2 * Real.sin (π / 4)
  8 * triangle_area = 8 * (1 / 2) * (6 * Real.sin (π / 8))^2 * Real.sin (π / 4) := by
sorry

end octagon_area_l3875_387535


namespace f_fixed_points_l3875_387522

def f (x : ℝ) : ℝ := x^3 - 4*x

theorem f_fixed_points (x : ℝ) : 
  (x = 0 ∨ x = 2 ∨ x = -2) → f (f x) = f x :=
by
  sorry

end f_fixed_points_l3875_387522


namespace expected_socks_theorem_l3875_387565

/-- The expected number of socks picked to retrieve both favorite socks -/
def expected_socks_picked (n : ℕ) : ℚ :=
  2 * (n + 1) / 3

/-- Theorem: The expected number of socks picked to retrieve both favorite socks is 2(n+1)/3 -/
theorem expected_socks_theorem (n : ℕ) (h : n ≥ 2) :
  expected_socks_picked n = 2 * (n + 1) / 3 := by
  sorry

#check expected_socks_theorem

end expected_socks_theorem_l3875_387565


namespace geometric_progression_ratio_l3875_387509

theorem geometric_progression_ratio (x y z w r : ℂ) : 
  x ≠ 0 ∧ y ≠ 0 ∧ z ≠ 0 ∧ w ≠ 0 ∧
  x * (y - w) ≠ 0 ∧ y * (z - x) ≠ 0 ∧ z * (w - y) ≠ 0 ∧ w * (x - z) ≠ 0 ∧
  x * (y - w) ≠ y * (z - x) ∧ y * (z - x) ≠ z * (w - y) ∧ z * (w - y) ≠ w * (x - z) ∧
  ∃ (a : ℂ), a ≠ 0 ∧
    y * (z - x) = r * (x * (y - w)) ∧
    z * (w - y) = r * (y * (z - x)) ∧
    w * (x - z) = r * (z * (w - y)) →
  r^3 + r^2 + r + 1 = 0 := by
sorry

end geometric_progression_ratio_l3875_387509


namespace hyperbola_equation_l3875_387521

/-- Given a hyperbola with asymptote equation 4x - 3y = 0 and sharing foci with the ellipse x²/30 + y²/5 = 1, 
    its equation is x²/9 - y²/16 = 1 -/
theorem hyperbola_equation (x y : ℝ) : 
  (∃ k : ℝ, 4 * x - 3 * y = k) ∧ 
  (∃ c : ℝ, c > 0 ∧ c^2 = 25 ∧ (∀ x y : ℝ, x^2 / 30 + y^2 / 5 = 1 → x^2 ≤ c^2)) →
  (x^2 / 9 - y^2 / 16 = 1) :=
by sorry

end hyperbola_equation_l3875_387521
