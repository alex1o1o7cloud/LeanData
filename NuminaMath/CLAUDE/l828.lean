import Mathlib

namespace volume_to_surface_area_ratio_l828_82869

/-- Represents a cube structure made of unit cubes -/
structure CubeStructure where
  side_length : ℕ
  removed_cubes : ℕ

/-- Calculates the volume of the cube structure -/
def volume (c : CubeStructure) : ℕ :=
  c.side_length^3 - c.removed_cubes

/-- Calculates the surface area of the cube structure -/
def surface_area (c : CubeStructure) : ℕ :=
  6 * c.side_length^2 - 4 * c.removed_cubes

/-- The specific cube structure described in the problem -/
def hollow_cube : CubeStructure :=
  { side_length := 3
  , removed_cubes := 1 }

/-- Theorem stating the ratio of volume to surface area for the hollow cube -/
theorem volume_to_surface_area_ratio :
  (volume hollow_cube : ℚ) / (surface_area hollow_cube : ℚ) = 2/7 := by
  sorry

end volume_to_surface_area_ratio_l828_82869


namespace product_selection_proof_l828_82835

def total_products : ℕ := 100
def qualified_products : ℕ := 98
def defective_products : ℕ := 2
def selected_products : ℕ := 3

theorem product_selection_proof :
  (Nat.choose total_products selected_products = 161700) ∧
  (Nat.choose defective_products 1 * Nat.choose qualified_products 2 = 9506) ∧
  (Nat.choose total_products selected_products - Nat.choose qualified_products selected_products = 9604) :=
by sorry

end product_selection_proof_l828_82835


namespace hannahs_speed_l828_82867

/-- 
Given two drivers, Glen and Hannah, driving towards each other and then away,
prove that Hannah's speed is 15 km/h under the following conditions:
- Glen drives at a constant speed of 37 km/h
- They are 130 km apart at 6 am and 11 am
- They pass each other at some point between 6 am and 11 am
-/
theorem hannahs_speed 
  (glen_speed : ℝ) 
  (initial_distance final_distance : ℝ)
  (time_interval : ℝ) :
  glen_speed = 37 →
  initial_distance = 130 →
  final_distance = 130 →
  time_interval = 5 →
  ∃ (hannah_speed : ℝ), hannah_speed = 15 := by
  sorry

end hannahs_speed_l828_82867


namespace unique_quadratic_root_l828_82804

theorem unique_quadratic_root (m : ℝ) : 
  (∃! x : ℝ, m * x^2 + 2 * x - 1 = 0) → (m = 0 ∨ m = -1) :=
by sorry

end unique_quadratic_root_l828_82804


namespace rectangle_ratio_golden_ratio_l828_82891

/-- Given a unit square AEFD and rectangles ABCD and BCFE, where the ratio of length to width
    of ABCD equals the ratio of length to width of BCFE, and AB has length W,
    prove that W = (1 + √5) / 2. -/
theorem rectangle_ratio_golden_ratio (W : ℝ) : 
  (W > 0) →  -- W is positive
  (W / 1 = 1 / (W - 1)) →  -- ratio equality condition
  W = (1 + Real.sqrt 5) / 2 := by
sorry

end rectangle_ratio_golden_ratio_l828_82891


namespace least_positive_angle_theorem_l828_82878

theorem least_positive_angle_theorem : ∃ θ : Real,
  θ > 0 ∧
  θ < 360 ∧
  Real.cos (15 * π / 180) = Real.sin (45 * π / 180) + Real.sin θ ∧
  θ = 195 * π / 180 ∧
  ∀ φ, 0 < φ ∧ φ < θ → Real.cos (15 * π / 180) ≠ Real.sin (45 * π / 180) + Real.sin φ :=
by sorry

end least_positive_angle_theorem_l828_82878


namespace expression_equality_l828_82826

theorem expression_equality : 784 + 2 * 28 * 7 + 49 = 1225 := by sorry

end expression_equality_l828_82826


namespace smarties_leftover_l828_82830

theorem smarties_leftover (m : ℕ) (h : m % 7 = 5) : (4 * m) % 7 = 6 := by
  sorry

end smarties_leftover_l828_82830


namespace parallel_line_k_value_l828_82894

/-- Given a line passing through points (4, -7) and (k, 25) that is parallel to the line 3x + 4y = 12, 
    the value of k is -116/3. -/
theorem parallel_line_k_value : 
  ∀ k : ℚ, 
  (∃ m b : ℚ, (∀ x y : ℚ, y = m * x + b → (x = 4 ∧ y = -7) ∨ (x = k ∧ y = 25)) ∧ 
               m = -(3 / 4)) → 
  k = -116 / 3 :=
by sorry

end parallel_line_k_value_l828_82894


namespace alloy_mixture_chromium_balance_l828_82857

/-- Represents the composition of an alloy mixture -/
structure AlloyMixture where
  first_alloy_amount : ℝ
  first_alloy_chromium_percent : ℝ
  second_alloy_amount : ℝ
  second_alloy_chromium_percent : ℝ
  new_alloy_chromium_percent : ℝ

/-- The alloy mixture satisfies the chromium balance equation -/
def satisfies_chromium_balance (mixture : AlloyMixture) : Prop :=
  mixture.first_alloy_chromium_percent * mixture.first_alloy_amount +
  mixture.second_alloy_chromium_percent * mixture.second_alloy_amount =
  mixture.new_alloy_chromium_percent * (mixture.first_alloy_amount + mixture.second_alloy_amount)

/-- Theorem: The alloy mixture satisfies the chromium balance equation -/
theorem alloy_mixture_chromium_balance 
  (mixture : AlloyMixture)
  (h1 : mixture.second_alloy_amount = 35)
  (h2 : mixture.second_alloy_chromium_percent = 0.08)
  (h3 : mixture.new_alloy_chromium_percent = 0.101) :
  satisfies_chromium_balance mixture :=
sorry

end alloy_mixture_chromium_balance_l828_82857


namespace sine_squared_equality_l828_82851

theorem sine_squared_equality (α β : ℝ) 
  (h : (Real.cos α)^4 / (Real.cos β)^2 + (Real.sin α)^4 / (Real.sin β)^2 = 1) :
  (Real.sin α)^2 = (Real.sin β)^2 := by
  sorry

end sine_squared_equality_l828_82851


namespace jerry_age_l828_82822

/-- Given that Mickey's age is 17 years and Mickey's age is 3 years less than 250% of Jerry's age,
    prove that Jerry's age is 8 years. -/
theorem jerry_age (mickey_age jerry_age : ℕ) : 
  mickey_age = 17 → 
  mickey_age = (250 * jerry_age) / 100 - 3 → 
  jerry_age = 8 :=
by sorry

end jerry_age_l828_82822


namespace cubic_equation_root_a_value_l828_82847

theorem cubic_equation_root_a_value :
  ∀ a b : ℚ,
  (∃ x : ℝ, x^3 + a*x^2 + b*x - 48 = 0 ∧ x = 2 - 5*Real.sqrt 3) →
  a = -332/71 := by
sorry

end cubic_equation_root_a_value_l828_82847


namespace quadratic_equation_roots_l828_82801

theorem quadratic_equation_roots (k : ℝ) : 
  ∃ x₁ x₂ : ℝ, x₁^2 + k*x₁ + k - 1 = 0 ∧ x₂^2 + k*x₂ + k - 1 = 0 := by
  sorry

end quadratic_equation_roots_l828_82801


namespace park_area_change_l828_82803

theorem park_area_change (original_area : ℝ) (length_decrease_percent : ℝ) (width_increase_percent : ℝ) :
  original_area = 600 →
  length_decrease_percent = 20 →
  width_increase_percent = 30 →
  let new_length_factor := 1 - length_decrease_percent / 100
  let new_width_factor := 1 + width_increase_percent / 100
  let new_area := original_area * new_length_factor * new_width_factor
  new_area = 624 := by sorry

end park_area_change_l828_82803


namespace fixed_point_on_line_l828_82883

/-- The line equation passes through a fixed point for all values of k -/
theorem fixed_point_on_line (k : ℝ) : 
  (2 * k - 1) * 2 - (k - 2) * 3 - (k + 4) = 0 := by sorry

end fixed_point_on_line_l828_82883


namespace series_sum_l828_82843

def series_term (n : ℕ) : ℚ := (2^n : ℚ) / ((3^(3^n) : ℚ) + 1)

theorem series_sum : ∑' n, series_term n = 1/2 := by sorry

end series_sum_l828_82843


namespace sum_of_coefficients_l828_82802

theorem sum_of_coefficients (a a₁ a₂ a₃ a₄ a₅ : ℝ) :
  (∀ x, (1 - 2*x)^5 = a + a₁*x + a₂*x^2 + a₃*x^3 + a₄*x^4 + a₅*x^5) →
  a₁ + a₂ + a₃ + a₄ + a₅ = -2 := by
sorry

end sum_of_coefficients_l828_82802


namespace tree_structure_equation_l828_82840

/-- Represents the structure of a tree with branches and small branches. -/
structure TreeStructure where
  branches : ℕ
  total_count : ℕ

/-- The equation for the tree structure is correct if it satisfies the given conditions. -/
def is_correct_equation (t : TreeStructure) : Prop :=
  1 + t.branches + t.branches^2 = t.total_count

/-- Theorem stating that the equation correctly represents the tree structure. -/
theorem tree_structure_equation (t : TreeStructure) 
  (h : t.total_count = 57) : is_correct_equation t := by
  sorry

end tree_structure_equation_l828_82840


namespace hyperbola_standard_equation_l828_82888

/-- A hyperbola with given properties -/
structure Hyperbola where
  a : ℝ
  b : ℝ
  equation : (x y : ℝ) → x^2 / a^2 - y^2 / b^2 = 1
  left_focus_x : ℝ
  left_focus_on_directrix : left_focus_x = -5  -- directrix of y^2 = 20x is x = -5
  asymptote_slope : b / a = 4 / 3

/-- The standard equation of the hyperbola is x^2/9 - y^2/16 = 1 -/
theorem hyperbola_standard_equation (h : Hyperbola) : 
  h.a = 3 ∧ h.b = 4 := by sorry

end hyperbola_standard_equation_l828_82888


namespace right_triangle_hypotenuse_l828_82861

theorem right_triangle_hypotenuse (a b c : ℝ) : 
  a > 0 → b > 0 → c > 0 →
  a^2 + b^2 = c^2 →  -- right-angled triangle condition
  a^2 + b^2 + c^2 = 2500 →  -- sum of squares condition
  c = 25 * Real.sqrt 2 := by sorry

end right_triangle_hypotenuse_l828_82861


namespace impossibility_of_arrangement_l828_82893

/-- Represents a 6x7 grid of natural numbers -/
def Grid := Fin 6 → Fin 7 → ℕ

/-- Checks if a given grid is a valid arrangement of numbers 1 to 42 -/
def is_valid_arrangement (g : Grid) : Prop :=
  (∀ i j, g i j ≥ 1 ∧ g i j ≤ 42) ∧
  (∀ i j k l, (i ≠ k ∨ j ≠ l) → g i j ≠ g k l)

/-- Checks if the sum of numbers in each 1x2 vertical rectangle is even -/
def has_even_vertical_sums (g : Grid) : Prop :=
  ∀ i j, Even (g i j + g (i.succ) j)

theorem impossibility_of_arrangement :
  ¬∃ (g : Grid), is_valid_arrangement g ∧ has_even_vertical_sums g :=
sorry

end impossibility_of_arrangement_l828_82893


namespace leading_coefficient_of_P_l828_82859

/-- The polynomial in question -/
def P (x : ℝ) : ℝ := 5*(x^5 - 2*x^4 + 3*x^3) - 6*(x^5 + x^3 + x) + 3*(3*x^5 - x^4 + 4*x^2 + 2)

/-- The leading coefficient of a polynomial -/
def leading_coefficient (p : ℝ → ℝ) : ℝ := 
  sorry

theorem leading_coefficient_of_P : leading_coefficient P = 8 := by
  sorry

end leading_coefficient_of_P_l828_82859


namespace min_bills_for_payment_l828_82838

/-- Represents the number of bills of each denomination --/
structure Bills :=
  (tens : ℕ)
  (fives : ℕ)
  (ones : ℕ)

/-- Calculates the total value of the bills --/
def billsValue (b : Bills) : ℕ :=
  10 * b.tens + 5 * b.fives + b.ones

/-- Checks if a given number of bills is valid for the payment --/
def isValidPayment (b : Bills) (amount : ℕ) : Prop :=
  b.tens ≤ 13 ∧ b.fives ≤ 11 ∧ b.ones ≤ 17 ∧ billsValue b = amount

/-- Counts the total number of bills --/
def totalBills (b : Bills) : ℕ :=
  b.tens + b.fives + b.ones

/-- The main theorem stating that the minimum number of bills required is 16 --/
theorem min_bills_for_payment :
  ∃ (b : Bills), isValidPayment b 128 ∧
  ∀ (b' : Bills), isValidPayment b' 128 → totalBills b ≤ totalBills b' :=
by sorry

end min_bills_for_payment_l828_82838


namespace polynomial_factorization_l828_82827

theorem polynomial_factorization (x : ℝ) : 
  x^9 - 6*x^6 + 12*x^3 - 8 = (x^3 - 2)^3 := by
  sorry

end polynomial_factorization_l828_82827


namespace avery_donation_l828_82875

theorem avery_donation (shirts : ℕ) 
  (h1 : shirts + 2 * shirts + shirts = 16) : shirts = 4 := by
  sorry

end avery_donation_l828_82875


namespace rationalize_sqrt_five_twelfths_l828_82815

theorem rationalize_sqrt_five_twelfths : 
  Real.sqrt (5 / 12) = Real.sqrt 15 / 6 := by
  sorry

end rationalize_sqrt_five_twelfths_l828_82815


namespace weekend_rain_probability_l828_82854

theorem weekend_rain_probability (prob_saturday prob_sunday : ℝ) 
  (h1 : prob_saturday = 0.3)
  (h2 : prob_sunday = 0.6)
  (h3 : 0 ≤ prob_saturday ∧ prob_saturday ≤ 1)
  (h4 : 0 ≤ prob_sunday ∧ prob_sunday ≤ 1) :
  1 - (1 - prob_saturday) * (1 - prob_sunday) = 0.72 := by
  sorry

end weekend_rain_probability_l828_82854


namespace car_dealership_count_l828_82865

theorem car_dealership_count :
  ∀ (total_cars : ℕ),
    (total_cars : ℝ) * 0.6 * 0.6 = 216 →
    total_cars = 600 := by
  sorry

end car_dealership_count_l828_82865


namespace trigonometric_identity_l828_82858

theorem trigonometric_identity (α β γ x : ℝ) : 
  (Real.sin (x - β) * Real.sin (x - γ)) / (Real.sin (α - β) * Real.sin (α - γ)) +
  (Real.sin (x - γ) * Real.sin (x - α)) / (Real.sin (β - γ) * Real.sin (β - α)) +
  (Real.sin (x - α) * Real.sin (x - β)) / (Real.sin (γ - α) * Real.sin (γ - β)) = 1 :=
by sorry

end trigonometric_identity_l828_82858


namespace dye_making_water_amount_l828_82876

/-- Given a dye-making process where:
  * The total mixture is 27 liters
  * 5/6 of 18 liters of vinegar is used
  * The water used is 3/5 of the total water available
  Prove that the amount of water used is 12 liters -/
theorem dye_making_water_amount (total_mixture : ℝ) (vinegar_amount : ℝ) (water_fraction : ℝ) :
  total_mixture = 27 →
  vinegar_amount = 5 / 6 * 18 →
  water_fraction = 3 / 5 →
  total_mixture - vinegar_amount = 12 :=
by sorry

end dye_making_water_amount_l828_82876


namespace complex_equation_solution_l828_82870

theorem complex_equation_solution (z : ℂ) : z * (2 - I) = 3 + I → z = 1 + I := by
  sorry

end complex_equation_solution_l828_82870


namespace machine_production_rate_l828_82821

/-- Given an industrial machine that made 8 shirts in 4 minutes today,
    prove that it can make 2 shirts per minute. -/
theorem machine_production_rate (shirts_today : ℕ) (minutes_today : ℕ) 
  (h1 : shirts_today = 8) (h2 : minutes_today = 4) :
  shirts_today / minutes_today = 2 := by
sorry

end machine_production_rate_l828_82821


namespace exponent_multiplication_l828_82881

theorem exponent_multiplication (a : ℝ) : a^2 * a^3 = a^5 := by
  sorry

end exponent_multiplication_l828_82881


namespace cube_sum_inequality_l828_82809

theorem cube_sum_inequality (n : ℕ) : 
  (∀ a b c : ℕ, (a + b + c)^3 ≤ n * (a^3 + b^3 + c^3)) ↔ n ≥ 9 := by sorry

end cube_sum_inequality_l828_82809


namespace mushroom_trip_theorem_l828_82811

def mushroom_trip_earnings (day1_earnings day2_price day3_price day4_price day5_price day6_price day7_price : ℝ)
  (day2_mushrooms : ℕ) (day3_increase day4_increase day5_mushrooms day6_decrease day7_mushrooms : ℝ)
  (expenses : ℝ) : Prop :=
  let day2_earnings := day2_mushrooms * day2_price
  let day3_mushrooms := day2_mushrooms + day3_increase
  let day3_earnings := day3_mushrooms * day3_price
  let day4_mushrooms := day3_mushrooms * (1 + day4_increase)
  let day4_earnings := day4_mushrooms * day4_price
  let day5_earnings := day5_mushrooms * day5_price
  let day6_mushrooms := day5_mushrooms * (1 - day6_decrease)
  let day6_earnings := day6_mushrooms * day6_price
  let day7_earnings := day7_mushrooms * day7_price
  let total_earnings := day1_earnings + day2_earnings + day3_earnings + day4_earnings + day5_earnings + day6_earnings + day7_earnings
  total_earnings - expenses = 703.40

theorem mushroom_trip_theorem : 
  mushroom_trip_earnings 120 2.50 1.75 1.30 2.00 2.50 1.80 20 18 0.40 72 0.25 80 25 := by
  sorry

end mushroom_trip_theorem_l828_82811


namespace largest_y_value_l828_82831

theorem largest_y_value (y : ℝ) : 
  (y / 3 + 2 / (3 * y) = 1) → y ≤ 2 :=
by sorry

end largest_y_value_l828_82831


namespace cora_cookie_purchase_l828_82836

/-- The number of cookies Cora purchased each day in April -/
def cookies_per_day : ℕ := 3

/-- The cost of each cookie in dollars -/
def cookie_cost : ℕ := 18

/-- The total amount Cora spent on cookies in April in dollars -/
def total_spent : ℕ := 1620

/-- The number of days in April -/
def days_in_april : ℕ := 30

/-- Theorem stating that Cora purchased 3 cookies each day in April -/
theorem cora_cookie_purchase :
  cookies_per_day * days_in_april * cookie_cost = total_spent :=
by sorry

end cora_cookie_purchase_l828_82836


namespace polynomial_simplification_l828_82832

theorem polynomial_simplification (q : ℝ) :
  (4 * q^4 - 7 * q^3 + 3 * q + 8) + (5 - 9 * q^3 + 4 * q^2 - 2 * q) =
  4 * q^4 - 16 * q^3 + 4 * q^2 + q + 13 :=
by sorry

end polynomial_simplification_l828_82832


namespace triangle_circumcircle_l828_82879

/-- Given a triangle with sides defined by three linear equations, 
    prove that its circumscribed circle has the specified equation. -/
theorem triangle_circumcircle 
  (line1 : ℝ → ℝ → Prop) 
  (line2 : ℝ → ℝ → Prop)
  (line3 : ℝ → ℝ → Prop)
  (h1 : ∀ x y, line1 x y ↔ x - 3*y = 2)
  (h2 : ∀ x y, line2 x y ↔ 7*x - y = 34)
  (h3 : ∀ x y, line3 x y ↔ x + 2*y = -8) :
  ∃ (center : ℝ × ℝ) (radius : ℝ),
    center = (1, -2) ∧ 
    radius = 5 ∧
    (∀ x y, (x - center.1)^2 + (y - center.2)^2 = radius^2 ↔ 
      (x - 1)^2 + (y + 2)^2 = 25) :=
sorry

end triangle_circumcircle_l828_82879


namespace total_cost_calculation_l828_82837

def tshirt_cost : ℝ := 9.95
def number_of_tshirts : ℕ := 20

theorem total_cost_calculation : 
  tshirt_cost * (number_of_tshirts : ℝ) = 199.00 := by sorry

end total_cost_calculation_l828_82837


namespace smallest_value_l828_82806

theorem smallest_value (x : ℝ) (h1 : 0 < x) (h2 : x < 1) :
  x^3 < x^2 ∧ x^3 < 3*x ∧ x^3 < Real.sqrt x ∧ x^3 < 1/x := by
  sorry

end smallest_value_l828_82806


namespace sally_monday_seashells_l828_82890

/-- The number of seashells Sally picked on Monday -/
def monday_seashells : ℕ := sorry

/-- The number of seashells Sally picked on Tuesday -/
def tuesday_seashells : ℕ := sorry

/-- The price of each seashell in dollars -/
def seashell_price : ℚ := 6/5

/-- The total amount Sally can make by selling all seashells in dollars -/
def total_amount : ℕ := 54

/-- Theorem stating the number of seashells Sally picked on Monday -/
theorem sally_monday_seashells : 
  monday_seashells = 30 ∧
  tuesday_seashells = monday_seashells / 2 ∧
  seashell_price * (monday_seashells + tuesday_seashells : ℚ) = total_amount := by
  sorry

end sally_monday_seashells_l828_82890


namespace common_rational_root_exists_l828_82868

theorem common_rational_root_exists :
  ∃ (k : ℚ) (a b c d e f g : ℚ),
    k = -1/3 ∧
    k < 0 ∧
    ¬(∃ n : ℤ, k = n) ∧
    90 * k^4 + a * k^3 + b * k^2 + c * k + 18 = 0 ∧
    18 * k^5 + d * k^4 + e * k^3 + f * k^2 + g * k + 90 = 0 :=
by sorry

end common_rational_root_exists_l828_82868


namespace product_of_primes_l828_82829

theorem product_of_primes : 3^2 * 5 * 7^2 * 11 = 24255 := by
  sorry

end product_of_primes_l828_82829


namespace symmetry_implies_k_and_b_values_l828_82895

/-- A linear function f(x) = mx + c is symmetric with respect to the y-axis if f(x) = f(-x) for all x -/
def SymmetricToYAxis (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f x = f (-x)

/-- The first linear function f(x) = kx - 5 -/
def f (k : ℝ) (x : ℝ) : ℝ := k * x - 5

/-- The second linear function g(x) = 2x + b -/
def g (b : ℝ) (x : ℝ) : ℝ := 2 * x + b

theorem symmetry_implies_k_and_b_values :
  ∀ k b : ℝ, 
    SymmetricToYAxis (f k) ∧ 
    SymmetricToYAxis (g b) →
    k = -2 ∧ b = -5 := by
  sorry

end symmetry_implies_k_and_b_values_l828_82895


namespace dog_count_l828_82814

theorem dog_count (num_puppies : ℕ) (dog_meal_frequency : ℕ) (dog_meal_amount : ℕ) (total_food : ℕ) : 
  num_puppies = 4 →
  dog_meal_frequency = 3 →
  dog_meal_amount = 4 →
  total_food = 108 →
  (∃ (num_dogs : ℕ),
    num_dogs * (dog_meal_frequency * dog_meal_amount) + 
    num_puppies * (3 * dog_meal_frequency) * (dog_meal_amount / 2) = total_food ∧
    num_dogs = 3) := by
  sorry

end dog_count_l828_82814


namespace space_diagonals_of_specific_polyhedron_l828_82882

structure ConvexPolyhedron where
  vertices : ℕ
  edges : ℕ
  faces : ℕ
  triangular_faces : ℕ
  quadrilateral_faces : ℕ
  pentagonal_faces : ℕ

def space_diagonals (Q : ConvexPolyhedron) : ℕ :=
  let total_segments := (Q.vertices.choose 2)
  let face_diagonals := 2 * Q.quadrilateral_faces + 5 * Q.pentagonal_faces
  total_segments - Q.edges - face_diagonals

theorem space_diagonals_of_specific_polyhedron :
  let Q : ConvexPolyhedron := {
    vertices := 30,
    edges := 70,
    faces := 40,
    triangular_faces := 20,
    quadrilateral_faces := 15,
    pentagonal_faces := 5
  }
  space_diagonals Q = 310 := by sorry

end space_diagonals_of_specific_polyhedron_l828_82882


namespace find_B_value_l828_82834

theorem find_B_value (A B : Nat) (h1 : A ≤ 9) (h2 : B ≤ 9) 
  (h3 : 32 + A * 100 + 70 + B = 705) : B = 3 := by
  sorry

end find_B_value_l828_82834


namespace sequence_sum_problem_l828_82810

/-- Given a sequence {aₙ} with sum Sn = (a₁(4ⁿ - 1)) / 3 and a₄ = 32, prove a₁ = 1/2 -/
theorem sequence_sum_problem (a : ℕ → ℚ) (S : ℕ → ℚ) 
  (h1 : ∀ n, S n = (a 1 * (4^n - 1)) / 3)
  (h2 : a 4 = 32) :
  a 1 = 1/2 := by sorry

end sequence_sum_problem_l828_82810


namespace find_m_max_sum_squares_max_sum_squares_achievable_l828_82892

-- Define the condition for the unique integer solution
def uniqueIntegerSolution (m : ℤ) : Prop :=
  ∃! (x : ℤ), |2 * x - m| ≤ 1

-- Define the condition for a, b, c
def abcCondition (a b c : ℝ) : Prop :=
  4 * a^4 + 4 * b^4 + 4 * c^4 = 6

-- Theorem 1: Prove m = 6
theorem find_m (m : ℤ) (h : uniqueIntegerSolution m) : m = 6 := by
  sorry

-- Theorem 2: Prove the maximum value of a^2 + b^2 + c^2
theorem max_sum_squares (a b c : ℝ) (h : abcCondition a b c) :
  a^2 + b^2 + c^2 ≤ 3 * Real.sqrt 2 / 2 := by
  sorry

-- Theorem 3: Prove the maximum value is achievable
theorem max_sum_squares_achievable :
  ∃ a b c : ℝ, abcCondition a b c ∧ a^2 + b^2 + c^2 = 3 * Real.sqrt 2 / 2 := by
  sorry

end find_m_max_sum_squares_max_sum_squares_achievable_l828_82892


namespace average_draw_is_n_plus_one_div_two_l828_82817

/-- Represents a deck of cards -/
structure Deck :=
  (n : ℕ)  -- Total number of cards
  (ace_count : ℕ)  -- Number of aces in the deck
  (h1 : n > 0)  -- The deck has at least one card
  (h2 : ace_count = 3)  -- There are exactly three aces in the deck

/-- The average number of cards drawn until the second ace -/
def average_draw (d : Deck) : ℚ :=
  (d.n + 1) / 2

/-- Theorem stating that the average number of cards drawn until the second ace is (n + 1) / 2 -/
theorem average_draw_is_n_plus_one_div_two (d : Deck) :
  average_draw d = (d.n + 1) / 2 := by
  sorry

end average_draw_is_n_plus_one_div_two_l828_82817


namespace inequality_proof_l828_82871

theorem inequality_proof (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : a^3 + b^3 = 2) :
  1/a + 1/b ≥ 2*(a^2 - a + 1)*(b^2 - b + 1) := by
  sorry

end inequality_proof_l828_82871


namespace age_problem_l828_82896

theorem age_problem (a b c : ℕ) : 
  a = b + 2 →
  b = 2 * c →
  a + b + c = 47 →
  b = 18 := by
sorry

end age_problem_l828_82896


namespace max_sum_of_counts_l828_82820

/-- Represents the color of a card -/
inductive CardColor
  | White
  | Black
  | Red

/-- Represents a stack of cards -/
structure CardStack :=
  (cards : List CardColor)
  (white_count : Nat)
  (black_count : Nat)
  (red_count : Nat)

/-- Calculates the sum of counts for a given card stack -/
def calculate_sum (stack : CardStack) : Nat :=
  sorry

/-- Theorem stating the maximum possible sum of counts -/
theorem max_sum_of_counts (stack : CardStack) 
  (h1 : stack.cards.length = 300)
  (h2 : stack.white_count = 100)
  (h3 : stack.black_count = 100)
  (h4 : stack.red_count = 100) :
  (∀ s : CardStack, calculate_sum s ≤ calculate_sum stack) →
  calculate_sum stack = 20000 :=
sorry

end max_sum_of_counts_l828_82820


namespace hyperbola_b_value_l828_82862

/-- Given a hyperbola with equation x^2 - my^2 = 3m (where m > 0),
    prove that the value of b in its standard form is √3. -/
theorem hyperbola_b_value (m : ℝ) (h : m > 0) :
  let C : ℝ × ℝ → Prop := λ (x, y) ↦ x^2 - m*y^2 = 3*m
  ∃ (a b : ℝ), (∀ (x y : ℝ), C (x, y) ↔ (x^2 / (a^2) - y^2 / (b^2) = 1)) ∧ b = Real.sqrt 3 :=
sorry

end hyperbola_b_value_l828_82862


namespace rectangle_area_ratio_l828_82818

theorem rectangle_area_ratio (large_horizontal small_horizontal large_vertical small_vertical large_area : ℝ)
  (h_horizontal_ratio : large_horizontal / small_horizontal = 8 / 7)
  (h_vertical_ratio : large_vertical / small_vertical = 9 / 4)
  (h_large_area : large_horizontal * large_vertical = large_area)
  (h_large_area_value : large_area = 108) :
  small_horizontal * small_vertical = 42 := by
  sorry

end rectangle_area_ratio_l828_82818


namespace sufficient_but_not_necessary_l828_82852

theorem sufficient_but_not_necessary (x : ℝ) : 
  (∀ x, x > 2 → x^2 > 4) ∧ 
  (∃ x, x^2 > 4 ∧ ¬(x > 2)) := by
  sorry

end sufficient_but_not_necessary_l828_82852


namespace arithmetic_sequence_property_l828_82853

/-- An arithmetic sequence -/
def ArithmeticSequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_property
  (a : ℕ → ℝ)
  (h_arith : ArithmeticSequence a)
  (h_sum : a 6 + a 8 + a 10 = 72) :
  2 * a 10 - a 12 = 24 :=
by sorry

end arithmetic_sequence_property_l828_82853


namespace percentage_failed_english_l828_82860

theorem percentage_failed_english (total_percentage : ℝ) (failed_hindi : ℝ) (failed_both : ℝ) (passed_both : ℝ) :
  total_percentage = 100 ∧
  failed_hindi = 30 ∧
  failed_both = 28 ∧
  passed_both = 56 →
  ∃ failed_english : ℝ,
    failed_english = 42 ∧
    total_percentage - passed_both = failed_hindi + failed_english - failed_both :=
by sorry

end percentage_failed_english_l828_82860


namespace telephone_pole_height_l828_82833

-- Define the problem parameters
def base_height : Real := 1
def cable_ground_distance : Real := 5
def leah_distance : Real := 4
def leah_height : Real := 1.8

-- Define the theorem
theorem telephone_pole_height :
  let total_ground_distance : Real := cable_ground_distance + base_height
  let remaining_distance : Real := total_ground_distance - leah_distance
  let pole_height : Real := (leah_height * total_ground_distance) / remaining_distance
  pole_height = 5.4 := by
  sorry

end telephone_pole_height_l828_82833


namespace tv_purchase_price_l828_82884

/-- The purchase price of a TV -/
def purchase_price : ℝ := 2250

/-- The profit made on each TV -/
def profit : ℝ := 270

/-- The price increase percentage -/
def price_increase : ℝ := 0.4

/-- The discount percentage -/
def discount : ℝ := 0.2

theorem tv_purchase_price :
  (purchase_price + purchase_price * price_increase) * (1 - discount) - purchase_price = profit :=
by sorry

end tv_purchase_price_l828_82884


namespace egg_production_increase_l828_82824

/-- The number of eggs produced last year -/
def last_year_production : ℕ := 1416

/-- The number of eggs produced this year -/
def this_year_production : ℕ := 4636

/-- The increase in egg production -/
def production_increase : ℕ := this_year_production - last_year_production

theorem egg_production_increase :
  production_increase = 3220 := by sorry

end egg_production_increase_l828_82824


namespace line_circle_intersection_l828_82866

/-- Given a line and a circle, prove that the coefficient of x in the line equation is 2 when the chord length is 4 -/
theorem line_circle_intersection (a : ℝ) : 
  (∃ x y : ℝ, x^2 + y^2 - 4*x - 2*y + 1 = 0 ∧ a*x + y - 5 = 0) → -- Circle and line intersect
  (∃ x1 y1 x2 y2 : ℝ, 
    x1^2 + y1^2 - 4*x1 - 2*y1 + 1 = 0 ∧ 
    x2^2 + y2^2 - 4*x2 - 2*y2 + 1 = 0 ∧ 
    a*x1 + y1 - 5 = 0 ∧ 
    a*x2 + y2 - 5 = 0 ∧ 
    (x1 - x2)^2 + (y1 - y2)^2 = 16) → -- Chord length is 4
  a = 2 := by
  sorry

#check line_circle_intersection

end line_circle_intersection_l828_82866


namespace range_of_m_for_positive_f_range_of_m_for_zero_in_interval_l828_82886

/-- The function f(x) = x^2 - (m-1)x + 2m -/
def f (m : ℝ) (x : ℝ) : ℝ := x^2 - (m-1)*x + 2*m

/-- Theorem 1: f(x) > 0 for all x in (0, +∞) iff -2√6 + 5 ≤ m ≤ 2√6 + 5 -/
theorem range_of_m_for_positive_f (m : ℝ) :
  (∀ x > 0, f m x > 0) ↔ -2*Real.sqrt 6 + 5 ≤ m ∧ m ≤ 2*Real.sqrt 6 + 5 :=
sorry

/-- Theorem 2: f(x) has a zero point in (0, 1) iff m ∈ (-2, 0) -/
theorem range_of_m_for_zero_in_interval (m : ℝ) :
  (∃ x ∈ Set.Ioo 0 1, f m x = 0) ↔ m > -2 ∧ m < 0 :=
sorry

end range_of_m_for_positive_f_range_of_m_for_zero_in_interval_l828_82886


namespace simplify_fraction_l828_82823

theorem simplify_fraction (a : ℝ) (h : a ≠ 3) :
  1 / (a - 3) - 6 / (a^2 - 9) = 1 / (a + 3) := by sorry

end simplify_fraction_l828_82823


namespace bowtie_equation_solution_l828_82880

-- Define the operation ⊗
noncomputable def bowtie (a b : ℝ) : ℝ :=
  a + Real.sqrt (b + 3 + Real.sqrt (b + 3 + Real.sqrt (b + 3 + Real.sqrt (b + 3))))

-- Theorem statement
theorem bowtie_equation_solution (x : ℝ) :
  bowtie 3 x = 12 → x = 69 := by
  sorry

end bowtie_equation_solution_l828_82880


namespace p_value_l828_82856

/-- The maximum value of x satisfying the inequality |x^2-4x+p|+|x-3|≤5 is 3 -/
def max_x_condition (p : ℝ) : Prop :=
  ∀ x : ℝ, |x^2 - 4*x + p| + |x - 3| ≤ 5 → x ≤ 3

/-- Theorem stating that p = 8 given the condition -/
theorem p_value : ∃ p : ℝ, max_x_condition p ∧ p = 8 :=
sorry

end p_value_l828_82856


namespace decimal_digit_13_14_l828_82889

def decimal_cycle (n d : ℕ) (cycle : List ℕ) : Prop :=
  ∀ k : ℕ, (n * 10^k) % d = (cycle.take ((k - 1) % cycle.length + 1)).foldl (λ acc x => (10 * acc + x) % d) 0

theorem decimal_digit_13_14 :
  decimal_cycle 13 14 [9, 2, 8, 5, 7, 1] →
  (13 * 10^150) / 14 % 10 = 1 := by
sorry

end decimal_digit_13_14_l828_82889


namespace roof_shingle_width_l828_82885

/-- The width of a rectangular roof shingle with length 10 inches and area 70 square inches is 7 inches. -/
theorem roof_shingle_width :
  ∀ (width : ℝ), 
    (10 : ℝ) * width = 70 → width = 7 := by
  sorry

end roof_shingle_width_l828_82885


namespace tamika_always_wins_l828_82845

def tamika_set : Finset ℕ := {7, 11, 14}
def carlos_set : Finset ℕ := {2, 4, 7}

theorem tamika_always_wins :
  ∀ (a b : ℕ), a ∈ tamika_set → b ∈ tamika_set → a ≠ b →
    ∀ (c d : ℕ), c ∈ carlos_set → d ∈ carlos_set → c ≠ d →
      a * b > c + d :=
by sorry

end tamika_always_wins_l828_82845


namespace vector_collinearity_l828_82816

theorem vector_collinearity (a b : ℝ × ℝ) : 
  a = (-1, 2) → b = (1, -2) → ∃ k : ℝ, b = k • a :=
by sorry

end vector_collinearity_l828_82816


namespace polynomial_product_expansion_l828_82813

theorem polynomial_product_expansion :
  ∀ x : ℝ, (3*x^2 + 2*x + 1) * (2*x^2 + 3*x + 4) = 6*x^4 + 13*x^3 + 20*x^2 + 11*x + 4 := by
  sorry

end polynomial_product_expansion_l828_82813


namespace perfect_square_trinomial_k_l828_82844

/-- A trinomial ax^2 + bx + c is a perfect square if there exist p and q such that ax^2 + bx + c = (px + q)^2 -/
def IsPerfectSquareTrinomial (a b c : ℝ) : Prop :=
  ∃ p q : ℝ, ∀ x : ℝ, a * x^2 + b * x + c = (p * x + q)^2

theorem perfect_square_trinomial_k (k : ℝ) :
  IsPerfectSquareTrinomial 1 (-k) 4 → k = 4 ∨ k = -4 :=
by
  sorry

end perfect_square_trinomial_k_l828_82844


namespace sum_of_squares_bound_l828_82819

theorem sum_of_squares_bound 
  (a b c d x y z t : ℝ) 
  (ha : 0 ≤ a ∧ a ≤ 1) 
  (hb : 0 ≤ b ∧ b ≤ 1) 
  (hc : 0 ≤ c ∧ c ≤ 1) 
  (hd : 0 ≤ d ∧ d ≤ 1) 
  (hx : x ≥ 1) 
  (hy : y ≥ 1) 
  (hz : z ≥ 1) 
  (ht : t ≥ 1) 
  (hsum : a + b + c + d + x + y + z + t = 8) : 
  a^2 + b^2 + c^2 + d^2 + x^2 + y^2 + z^2 + t^2 ≤ 28 := by
  sorry

end sum_of_squares_bound_l828_82819


namespace sector_area_given_arc_length_l828_82808

/-- Given a circular sector where the arc length corresponding to a central angle of 2 radians is 4 cm, 
    the area of this sector is 4 cm². -/
theorem sector_area_given_arc_length (r : ℝ) (h : 2 * r = 4) : r * r = 4 := by
  sorry

end sector_area_given_arc_length_l828_82808


namespace polynomial_expansion_properties_l828_82807

theorem polynomial_expansion_properties (a₀ a₁ a₂ a₃ a₄ : ℝ) :
  (∀ x, (x - 2)^4 = a₀ + a₁*x + a₂*x^2 + a₃*x^3 + a₄*x^4) →
  (a₀ = 16 ∧ a₂ = 24 ∧ a₁ + a₂ + a₃ + a₄ = -15) := by
  sorry

end polynomial_expansion_properties_l828_82807


namespace counterexample_exists_l828_82874

theorem counterexample_exists : ∃ n : ℝ, n < 1 ∧ n^2 - 1 ≥ 0 := by
  sorry

end counterexample_exists_l828_82874


namespace intersection_points_count_l828_82848

theorem intersection_points_count : ∃! (points : Finset (ℝ × ℝ)),
  (∀ (x y : ℝ), (x, y) ∈ points ↔ (9*x^2 + 4*y^2 = 36 ∧ 4*x^2 + 9*y^2 = 36)) ∧
  points.card = 4 := by
sorry

end intersection_points_count_l828_82848


namespace average_rate_round_trip_l828_82846

/-- Calculates the average rate of a round trip given the distance, running speed, and swimming speed. -/
theorem average_rate_round_trip 
  (distance : ℝ) 
  (running_speed : ℝ) 
  (swimming_speed : ℝ) 
  (h1 : distance = 4) 
  (h2 : running_speed = 10) 
  (h3 : swimming_speed = 6) : 
  (2 * distance) / (distance / running_speed + distance / swimming_speed) / 60 = 0.125 := by
  sorry

end average_rate_round_trip_l828_82846


namespace right_angled_triangle_l828_82887

theorem right_angled_triangle (a b c : ℝ) (h1 : a = 1) (h2 : b = Real.sqrt 3) (h3 : c = 2) :
  a ^ 2 + b ^ 2 = c ^ 2 :=
by sorry

end right_angled_triangle_l828_82887


namespace double_reflection_F_l828_82872

/-- Reflects a point over the y-axis -/
def reflect_y (p : ℝ × ℝ) : ℝ × ℝ :=
  (-p.1, p.2)

/-- Reflects a point over the x-axis -/
def reflect_x (p : ℝ × ℝ) : ℝ × ℝ :=
  (p.1, -p.2)

/-- The original point F -/
def F : ℝ × ℝ := (-2, -3)

theorem double_reflection_F :
  (reflect_x (reflect_y F)) = (2, 3) := by
  sorry

end double_reflection_F_l828_82872


namespace unique_distance_l828_82842

/-- A two-digit number is represented as 10a + b where a and b are single digits -/
def two_digit_number (a b : ℕ) : Prop := 
  a ≥ 1 ∧ a ≤ 9 ∧ b ≥ 0 ∧ b ≤ 9

/-- Inserting a zero between digits of a two-digit number -/
def insert_zero (a b : ℕ) : ℕ := 100 * a + b

/-- The property that inserting a zero results in 9 times the original number -/
def nine_times_property (a b : ℕ) : Prop :=
  insert_zero a b = 9 * (10 * a + b)

theorem unique_distance : 
  ∀ a b : ℕ, two_digit_number a b → nine_times_property a b → a = 4 ∧ b = 5 := by
  sorry

#check unique_distance

end unique_distance_l828_82842


namespace china_gdp_growth_l828_82897

/-- China's GDP growth model from 2011 to 2016 -/
theorem china_gdp_growth (a r : ℝ) (h : a > 0) (h2 : r > 0) :
  let initial_gdp := a
  let growth_rate := r / 100
  let years := 5
  let final_gdp := initial_gdp * (1 + growth_rate) ^ years
  final_gdp = a * (1 + r / 100) ^ 5 := by sorry

end china_gdp_growth_l828_82897


namespace adjacent_combinations_l828_82828

def number_of_people : ℕ := 9
def number_of_friends : ℕ := 8
def adjacent_positions : ℕ := 2

theorem adjacent_combinations :
  Nat.choose number_of_friends adjacent_positions = 28 := by
  sorry

end adjacent_combinations_l828_82828


namespace possible_k_values_l828_82812

def M : Set ℝ := {x | x^2 + x - 6 = 0}
def N (k : ℝ) : Set ℝ := {x | k*x + 1 = 0}

theorem possible_k_values :
  ∀ k : ℝ, (N k ⊆ M) ↔ (k = 0 ∨ k = -1/2 ∨ k = 1/3) := by sorry

end possible_k_values_l828_82812


namespace flour_for_hundred_cookies_l828_82899

-- Define the recipe's ratio
def recipe_cookies : ℕ := 40
def recipe_flour : ℚ := 3

-- Define the desired number of cookies
def desired_cookies : ℕ := 100

-- Define the function to calculate required flour
def required_flour (cookies : ℕ) : ℚ :=
  (recipe_flour / recipe_cookies) * cookies

-- Theorem statement
theorem flour_for_hundred_cookies :
  required_flour desired_cookies = 7.5 := by
  sorry

end flour_for_hundred_cookies_l828_82899


namespace max_value_in_equation_max_value_achievable_l828_82841

/-- Represents a three-digit number composed of different non-zero digits from 1 to 9 -/
def ThreeDigitNumber := { n : ℕ // 100 ≤ n ∧ n < 1000 ∧ (∃ x y z, x ≠ y ∧ y ≠ z ∧ x ≠ z ∧ 1 ≤ x ∧ x ≤ 9 ∧ 1 ≤ y ∧ y ≤ 9 ∧ 1 ≤ z ∧ z ≤ 9 ∧ n = 100 * x + 10 * y + z) }

/-- The main theorem stating the maximum value of a in the given equation -/
theorem max_value_in_equation (a b c d : ThreeDigitNumber) 
  (h : 1984 - a.val = 2015 - b.val - c.val - d.val) : 
  a.val ≤ 214 := by
  sorry

/-- The theorem proving that 214 is achievable -/
theorem max_value_achievable : 
  ∃ (a b c d : ThreeDigitNumber), 1984 - a.val = 2015 - b.val - c.val - d.val ∧ a.val = 214 := by
  sorry

end max_value_in_equation_max_value_achievable_l828_82841


namespace problem_solution_l828_82849

theorem problem_solution :
  ∀ (a b : ℝ), a > 0 → b > 0 →
  (∃ (max_value : ℝ), (a + 3*b + 3/a + 4/b = 18) → max_value = 9 + 3*Real.sqrt 6 ∧ a + 3*b ≤ max_value) ∧
  (a > b → ∃ (min_value : ℝ), min_value = 32 ∧ a^2 + 64 / (b*(a-b)) ≥ min_value) ∧
  (∃ (min_value : ℝ), (1/(a+1) + 1/(b+2) = 1/3) → min_value = 14 + 6*Real.sqrt 6 ∧ a*b + a + b ≥ min_value) :=
by sorry

end problem_solution_l828_82849


namespace max_strips_from_sheet_l828_82839

/-- Represents a rectangular sheet of paper --/
structure Sheet where
  length : ℕ
  width : ℕ

/-- Represents a rectangular strip of paper --/
structure Strip where
  length : ℕ
  width : ℕ

/-- Calculates the maximum number of strips that can be cut from a sheet --/
def maxStrips (sheet : Sheet) (strip : Strip) : ℕ :=
  max
    ((sheet.length / strip.length) * (sheet.width / strip.width))
    ((sheet.length / strip.width) * (sheet.width / strip.length))

theorem max_strips_from_sheet :
  let sheet := Sheet.mk 14 11
  let strip := Strip.mk 4 1
  maxStrips sheet strip = 33 := by sorry

end max_strips_from_sheet_l828_82839


namespace two_heart_three_l828_82825

/-- The ♥ operation defined as a ♥ b = ab³ - 2b + 3 -/
def heart (a b : ℝ) : ℝ := a * b^3 - 2*b + 3

/-- Theorem stating that 2 ♥ 3 = 51 -/
theorem two_heart_three : heart 2 3 = 51 := by
  sorry

end two_heart_three_l828_82825


namespace women_fair_hair_percentage_l828_82800

/-- Represents the percentage of fair-haired employees who are women -/
def percent_fair_haired_women : ℝ := 0.40

/-- Represents the percentage of employees who have fair hair -/
def percent_fair_haired : ℝ := 0.25

/-- Represents the percentage of employees who are women with fair hair -/
def percent_women_fair_hair : ℝ := percent_fair_haired_women * percent_fair_haired

theorem women_fair_hair_percentage :
  percent_women_fair_hair = 0.10 := by sorry

end women_fair_hair_percentage_l828_82800


namespace at_least_three_prime_factors_l828_82898

theorem at_least_three_prime_factors (n : ℕ) 
  (h1 : n > 0) 
  (h2 : n < 200) 
  (h3 : ∃ k : ℤ, (14 * n) / 60 = k) : 
  ∃ p q r : ℕ, Prime p ∧ Prime q ∧ Prime r ∧ p ≠ q ∧ p ≠ r ∧ q ≠ r ∧ p ∣ n ∧ q ∣ n ∧ r ∣ n :=
sorry

end at_least_three_prime_factors_l828_82898


namespace circular_garden_radius_l828_82877

/-- 
Theorem: For a circular garden with radius r, if the length of the fence (circumference) 
is 1/4 of the area of the garden, then r = 8.
-/
theorem circular_garden_radius (r : ℝ) (h : r > 0) : 2 * π * r = (1/4) * π * r^2 → r = 8 := by
  sorry

end circular_garden_radius_l828_82877


namespace inequalities_proof_l828_82850

theorem inequalities_proof :
  (∀ x : ℝ, 2 * x^2 + 5 * x + 3 > x^2 + 3 * x + 1) ∧
  (∀ a b : ℝ, a > b ∧ b > 0 → Real.sqrt a > Real.sqrt b) := by
  sorry

end inequalities_proof_l828_82850


namespace specific_polygon_perimeter_l828_82805

/-- A polygon that forms part of a square -/
structure PartialSquarePolygon where
  /-- The length of each visible side of the polygon -/
  visible_side_length : ℝ
  /-- The fraction of the square that the polygon occupies -/
  occupied_fraction : ℝ
  /-- Assumption that the visible side length is positive -/
  visible_side_positive : visible_side_length > 0
  /-- Assumption that the occupied fraction is between 0 and 1 -/
  occupied_fraction_valid : 0 < occupied_fraction ∧ occupied_fraction ≤ 1

/-- The perimeter of a polygon that forms part of a square -/
def perimeter (p : PartialSquarePolygon) : ℝ :=
  4 * p.visible_side_length * p.occupied_fraction

/-- Theorem stating that a polygon occupying three-fourths of a square with visible sides of 5 units has a perimeter of 15 units -/
theorem specific_polygon_perimeter :
  ∀ (p : PartialSquarePolygon),
  p.visible_side_length = 5 →
  p.occupied_fraction = 3/4 →
  perimeter p = 15 := by
  sorry

end specific_polygon_perimeter_l828_82805


namespace no_natural_number_power_of_two_l828_82864

theorem no_natural_number_power_of_two : 
  ¬ ∃ (n : ℕ), ∃ (k : ℕ), n^2012 - 1 = 2^k := by
  sorry

end no_natural_number_power_of_two_l828_82864


namespace quadratic_inequality_solution_set_l828_82855

theorem quadratic_inequality_solution_set 
  (a b : ℝ) 
  (h : Set.Ioo (-1/2 : ℝ) (1/3 : ℝ) = {x : ℝ | a * x^2 + b * x + 2 > 0}) :
  {x : ℝ | 2 * x^2 + b * x + a < 0} = Set.Ioo (-2 : ℝ) (3 : ℝ) := by
  sorry

end quadratic_inequality_solution_set_l828_82855


namespace coin_flip_game_properties_l828_82863

/-- Represents the coin-flipping game where a player wins if heads come up on an even-numbered throw
    or loses if tails come up on an odd-numbered throw. -/
def CoinFlipGame :=
  { win_prob : ℝ // win_prob = 1/3 } × { expected_flips : ℝ // expected_flips = 2 }

/-- The probability of winning the coin-flipping game is 1/3, and the expected number of flips is 2. -/
theorem coin_flip_game_properties : ∃ (game : CoinFlipGame), True :=
sorry

end coin_flip_game_properties_l828_82863


namespace no_even_integers_satisfying_conditions_l828_82873

theorem no_even_integers_satisfying_conditions : 
  ¬ ∃ (n : ℤ), 
    (n % 2 = 0) ∧ 
    (100 ≤ n) ∧ (n ≤ 1000) ∧ 
    (∃ (k : ℕ), n = 3 * k + 4) ∧ 
    (∃ (m : ℕ), n = 5 * m + 2) := by
  sorry

end no_even_integers_satisfying_conditions_l828_82873
