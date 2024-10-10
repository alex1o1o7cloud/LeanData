import Mathlib

namespace evaluate_expression_l191_19111

theorem evaluate_expression (a : ℝ) : 
  let x : ℝ := a + 9
  (x - a + 5) = 14 := by sorry

end evaluate_expression_l191_19111


namespace sequence_problem_solution_l191_19154

def sequence_problem (a : ℕ → ℝ) : Prop :=
  (∀ n : ℕ, a n * a (n + 1) = 1 - a (n + 1)) ∧ 
  (a 2010 = 2) ∧
  (a 2008 = -3)

theorem sequence_problem_solution :
  ∃ a : ℕ → ℝ, sequence_problem a :=
sorry

end sequence_problem_solution_l191_19154


namespace kyles_rose_expense_l191_19146

def roses_last_year : ℕ := 12
def roses_this_year : ℕ := roses_last_year / 2
def roses_needed : ℕ := 2 * roses_last_year
def price_per_rose : ℕ := 3

theorem kyles_rose_expense : 
  (roses_needed - roses_this_year) * price_per_rose = 54 := by sorry

end kyles_rose_expense_l191_19146


namespace percentage_relationship_l191_19192

theorem percentage_relationship (x y z : ℝ) (h1 : y = 0.7 * z) (h2 : x = 0.84 * z) :
  x = y * 1.2 :=
sorry

end percentage_relationship_l191_19192


namespace solution_set_characterization_l191_19153

def is_even (f : ℝ → ℝ) : Prop := ∀ x, f x = f (-x)

def monotone_decreasing_on (f : ℝ → ℝ) (s : Set ℝ) : Prop :=
  ∀ {x y}, x ∈ s → y ∈ s → x ≤ y → f y ≤ f x

theorem solution_set_characterization
  (f : ℝ → ℝ)
  (h_even : is_even f)
  (h_decreasing : monotone_decreasing_on f (Set.Ici 0))
  (h_f1 : f 1 = 0) :
  {x | f x > 0} = {x | -1 < x ∧ x < 1} := by sorry

end solution_set_characterization_l191_19153


namespace problem_solution_l191_19145

theorem problem_solution (x : ℝ) (h : |x| = x + 2) :
  19 * x^99 + 3 * x + 27 = 5 := by
  sorry

end problem_solution_l191_19145


namespace train_distance_l191_19175

/-- Proves that a train traveling at a rate of 1 mile per 2 minutes will cover 15 miles in 30 minutes. -/
theorem train_distance (rate : ℚ) (time : ℚ) (distance : ℚ) : 
  rate = 1 / 2 →  -- The train travels 1 mile in 2 minutes
  time = 30 →     -- We want to know the distance traveled in 30 minutes
  distance = rate * time →  -- Distance is calculated as rate times time
  distance = 15 :=  -- The train will travel 15 miles
by
  sorry

end train_distance_l191_19175


namespace ellipse_axis_endpoint_distance_l191_19165

/-- Given an ellipse defined by the equation 16(x+2)^2 + 4(y-3)^2 = 64,
    prove that the distance between an endpoint of its major axis
    and an endpoint of its minor axis is 2√5. -/
theorem ellipse_axis_endpoint_distance :
  ∀ (C D : ℝ × ℝ),
  (∀ (x y : ℝ), 16 * (x + 2)^2 + 4 * (y - 3)^2 = 64 →
    (C.1 + 2)^2 / 4 + (C.2 - 3)^2 / 16 = 1 ∧
    (D.1 + 2)^2 / 4 + (D.2 - 3)^2 / 16 = 1 ∧
    ((C.1 + 2)^2 / 4 = 1 ∨ (C.2 - 3)^2 / 16 = 1) ∧
    ((D.1 + 2)^2 / 4 = 1 ∨ (D.2 - 3)^2 / 16 = 1) ∧
    (C.1 + 2)^2 / 4 ≠ (D.1 + 2)^2 / 4) →
  Real.sqrt ((C.1 - D.1)^2 + (C.2 - D.2)^2) = 2 * Real.sqrt 5 := by
sorry

end ellipse_axis_endpoint_distance_l191_19165


namespace circle_C_equation_l191_19102

/-- A circle C with the following properties:
  - The center is on the positive x-axis
  - The radius is √2
  - The circle is tangent to the line x + y = 0
-/
structure CircleC where
  center : ℝ × ℝ
  radius : ℝ
  center_on_x_axis : center.2 = 0
  center_positive : center.1 > 0
  radius_is_sqrt2 : radius = Real.sqrt 2
  tangent_to_line : ∃ (p : ℝ × ℝ), p.1 + p.2 = 0 ∧ 
    (center.1 - p.1)^2 + (center.2 - p.2)^2 = radius^2

/-- The standard equation of circle C is (x-2)² + y² = 2 -/
theorem circle_C_equation (c : CircleC) : 
  ∀ (x y : ℝ), (x - 2)^2 + y^2 = 2 ↔ (x - c.center.1)^2 + (y - c.center.2)^2 = c.radius^2 :=
by sorry

end circle_C_equation_l191_19102


namespace coefficient_x2y2_in_expansion_l191_19113

def binomial_coefficient (n k : ℕ) : ℕ := Nat.choose n k

theorem coefficient_x2y2_in_expansion :
  let n : ℕ := 4
  let k : ℕ := 2
  let coefficient : ℤ := binomial_coefficient n k * (-2)^k
  coefficient = 24 := by sorry

end coefficient_x2y2_in_expansion_l191_19113


namespace base_b_square_l191_19184

theorem base_b_square (b : ℕ) (hb : b > 1) : 
  (∃ n : ℕ, b^2 + 4*b + 4 = n^2) ↔ b > 4 := by
  sorry

end base_b_square_l191_19184


namespace fixed_points_of_specific_quadratic_min_value_of_ratio_sum_range_of_a_for_always_fixed_point_l191_19157

-- Definition of a quadratic function
def quadratic (m n t : ℝ) (x : ℝ) : ℝ := m * x^2 + n * x + t

-- Definition of a fixed point
def is_fixed_point (f : ℝ → ℝ) (x : ℝ) : Prop := f x = x

-- Part 1
theorem fixed_points_of_specific_quadratic :
  let f := quadratic 1 (-1) (-3)
  ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ is_fixed_point f x₁ ∧ is_fixed_point f x₂ ∧ x₁ = -1 ∧ x₂ = 3 := by sorry

-- Part 2
theorem min_value_of_ratio_sum :
  ∀ a : ℝ, a > 1 →
  let f := quadratic 2 (-(3+a)) (a-1)
  ∃ x₁ x₂ : ℝ, x₁ > 0 ∧ x₂ > 0 ∧ x₁ ≠ x₂ ∧ is_fixed_point f x₁ ∧ is_fixed_point f x₂ →
  (∀ y₁ y₂ : ℝ, y₁ > 0 ∧ y₂ > 0 ∧ y₁ ≠ y₂ ∧ is_fixed_point f y₁ ∧ is_fixed_point f y₂ →
    y₁ / y₂ + y₂ / y₁ ≥ 8) := by sorry

-- Part 3
theorem range_of_a_for_always_fixed_point :
  ∀ a : ℝ, a ≠ 0 →
  (∀ b : ℝ, ∃ x : ℝ, is_fixed_point (quadratic a (b+1) (b-1)) x) ↔
  0 < a ∧ a ≤ 1 := by sorry

end fixed_points_of_specific_quadratic_min_value_of_ratio_sum_range_of_a_for_always_fixed_point_l191_19157


namespace lisa_walking_distance_l191_19104

/-- Lisa's walking problem -/
theorem lisa_walking_distance
  (walking_speed : ℕ)  -- Lisa's walking speed in meters per minute
  (daily_duration : ℕ)  -- Lisa's daily walking duration in minutes
  (days : ℕ)  -- Number of days
  (h1 : walking_speed = 10)  -- Lisa walks 10 meters each minute
  (h2 : daily_duration = 60)  -- Lisa walks for an hour (60 minutes) every day
  (h3 : days = 2)  -- We're considering two days
  : walking_speed * daily_duration * days = 1200 :=
by
  sorry

#check lisa_walking_distance

end lisa_walking_distance_l191_19104


namespace six_tricycles_l191_19193

/-- Represents the number of children riding each type of vehicle -/
structure VehicleCounts where
  bicycles : ℕ
  tricycles : ℕ
  unicycles : ℕ

/-- The total number of children -/
def total_children : ℕ := 10

/-- The total number of wheels -/
def total_wheels : ℕ := 26

/-- Calculates the total number of children based on vehicle counts -/
def count_children (v : VehicleCounts) : ℕ :=
  v.bicycles + v.tricycles + v.unicycles

/-- Calculates the total number of wheels based on vehicle counts -/
def count_wheels (v : VehicleCounts) : ℕ :=
  2 * v.bicycles + 3 * v.tricycles + v.unicycles

/-- Theorem stating that there are 6 tricycles -/
theorem six_tricycles : 
  ∃ v : VehicleCounts, 
    count_children v = total_children ∧ 
    count_wheels v = total_wheels ∧ 
    v.tricycles = 6 := by
  sorry

end six_tricycles_l191_19193


namespace quadratic_roots_condition_l191_19195

theorem quadratic_roots_condition (m : ℝ) : 
  (∃ x y : ℝ, x ≠ y ∧ x^2 + m*x + m + 3 = 0 ∧ y^2 + m*y + m + 3 = 0) ↔ 
  (m < -2 ∨ m > 6) := by
sorry

end quadratic_roots_condition_l191_19195


namespace billy_restaurant_bill_l191_19158

/-- The total bill at Billy's Restaurant for a group with given characteristics -/
def total_bill (num_adults num_children : ℕ) (adult_meal_cost child_meal_cost : ℚ) 
  (num_fries_baskets : ℕ) (fries_basket_cost : ℚ) (drink_cost : ℚ) : ℚ :=
  num_adults * adult_meal_cost + 
  num_children * child_meal_cost + 
  num_fries_baskets * fries_basket_cost + 
  drink_cost

/-- Theorem stating that the total bill for the given group is $89 -/
theorem billy_restaurant_bill : 
  total_bill 4 3 12 7 2 5 10 = 89 := by
  sorry

end billy_restaurant_bill_l191_19158


namespace diagonals_bisect_angles_and_parallel_implies_parallelogram_diagonals_bisect_area_implies_parallelogram_l191_19191

-- Define a quadrilateral
structure Quadrilateral :=
  (A B C D : Point)

-- Define a parallelogram
def is_parallelogram (q : Quadrilateral) : Prop := sorry

-- Define diagonals
def diagonal1 (q : Quadrilateral) : Line := sorry
def diagonal2 (q : Quadrilateral) : Line := sorry

-- Define the property of diagonals bisecting each other's interior angles
def diagonals_bisect_angles (q : Quadrilateral) : Prop := sorry

-- Define the property of diagonals being parallel
def diagonals_parallel (q : Quadrilateral) : Prop := sorry

-- Define the property of diagonals bisecting the area
def diagonals_bisect_area (q : Quadrilateral) : Prop := sorry

-- Theorem 1
theorem diagonals_bisect_angles_and_parallel_implies_parallelogram 
  (q : Quadrilateral) (h1 : diagonals_bisect_angles q) (h2 : diagonals_parallel q) : 
  is_parallelogram q := sorry

-- Theorem 2
theorem diagonals_bisect_area_implies_parallelogram 
  (q : Quadrilateral) (h : diagonals_bisect_area q) : 
  is_parallelogram q := sorry

end diagonals_bisect_angles_and_parallel_implies_parallelogram_diagonals_bisect_area_implies_parallelogram_l191_19191


namespace imaginary_part_of_z_is_zero_l191_19134

theorem imaginary_part_of_z_is_zero (z : ℂ) (h : z / (1 + 2 * I) = 1 - 2 * I) : 
  z.im = 0 := by
  sorry

end imaginary_part_of_z_is_zero_l191_19134


namespace max_power_under_500_l191_19112

theorem max_power_under_500 :
  ∃ (a b : ℕ), 
    a > 0 ∧ b > 1 ∧ 
    a^b < 500 ∧
    (∀ (c d : ℕ), c > 0 → d > 1 → c^d < 500 → c^d ≤ a^b) ∧
    a = 22 ∧ b = 2 ∧ 
    a + b = 24 := by
  sorry

end max_power_under_500_l191_19112


namespace quadratic_inequality_l191_19127

theorem quadratic_inequality : ∀ x : ℝ, 2*x^2 + 5*x + 3 > x^2 + 4*x + 2 := by
  sorry

end quadratic_inequality_l191_19127


namespace anns_skating_speed_l191_19143

/-- Proves that Ann's skating speed is 6 miles per hour given the problem conditions. -/
theorem anns_skating_speed :
  ∀ (ann_speed : ℝ),
  let glenda_speed : ℝ := 8
  let time : ℝ := 3
  let total_distance : ℝ := 42
  (ann_speed * time + glenda_speed * time = total_distance) →
  ann_speed = 6 := by
sorry

end anns_skating_speed_l191_19143


namespace sally_lost_cards_l191_19166

def pokemon_cards_lost (initial : ℕ) (received : ℕ) (current : ℕ) : ℕ :=
  initial + received - current

theorem sally_lost_cards (initial : ℕ) (received : ℕ) (current : ℕ)
  (h1 : initial = 27)
  (h2 : received = 41)
  (h3 : current = 48) :
  pokemon_cards_lost initial received current = 20 := by
  sorry

end sally_lost_cards_l191_19166


namespace cone_sphere_ratio_l191_19182

/-- Proof that the ratio of cone height to base radius is 4/3 when cone volume is 1/3 of sphere volume --/
theorem cone_sphere_ratio (r h : ℝ) (hr : r > 0) :
  (1 / 3) * ((4 / 3) * Real.pi * r^3) = (1 / 3) * Real.pi * r^2 * h → h / r = 4 / 3 := by
  sorry

end cone_sphere_ratio_l191_19182


namespace jerry_added_two_figures_l191_19141

/-- Represents the number of action figures Jerry added to the shelf. -/
def added_figures : ℕ := sorry

/-- The initial number of books on the shelf. -/
def initial_books : ℕ := 7

/-- The initial number of action figures on the shelf. -/
def initial_figures : ℕ := 3

/-- The difference between the number of books and action figures after adding. -/
def book_figure_difference : ℕ := 2

theorem jerry_added_two_figures : 
  added_figures = 2 ∧ 
  initial_books = (initial_figures + added_figures) + book_figure_difference :=
sorry

end jerry_added_two_figures_l191_19141


namespace shortest_distance_is_one_l191_19106

/-- Curve C₁ parameterized by θ -/
noncomputable def C₁ (θ : ℝ) : ℝ × ℝ × ℝ := sorry

/-- Curve C₂ parameterized by t -/
noncomputable def C₂ (t : ℝ) : ℝ × ℝ × ℝ := sorry

/-- Distance function between points on C₁ and C₂ -/
noncomputable def D (θ t : ℝ) : ℝ :=
  let (x₁, y₁, z₁) := C₁ θ
  let (x₂, y₂, z₂) := C₂ t
  Real.sqrt ((x₁ - x₂)^2 + (y₁ - y₂)^2 + (z₁ - z₂)^2)

/-- The shortest distance between C₁ and C₂ is 1 -/
theorem shortest_distance_is_one : ∃ θ₀ t₀, ∀ θ t, D θ₀ t₀ ≤ D θ t ∧ D θ₀ t₀ = 1 := by
  sorry

end shortest_distance_is_one_l191_19106


namespace prob_at_least_one_male_l191_19147

/-- The number of male students -/
def num_male : ℕ := 3

/-- The number of female students -/
def num_female : ℕ := 2

/-- The total number of students -/
def total_students : ℕ := num_male + num_female

/-- The number of students to be chosen -/
def num_chosen : ℕ := 2

/-- The probability of choosing at least one male student -/
theorem prob_at_least_one_male :
  (1 : ℚ) - (Nat.choose num_female num_chosen : ℚ) / (Nat.choose total_students num_chosen : ℚ) = 9/10 :=
sorry

end prob_at_least_one_male_l191_19147


namespace cube_surface_area_l191_19144

theorem cube_surface_area (volume : ℝ) (side : ℝ) (surface_area : ℝ) : 
  volume = 729 → 
  volume = side^3 → 
  surface_area = 6 * side^2 → 
  surface_area = 486 :=
by sorry

end cube_surface_area_l191_19144


namespace factorization_equality_l191_19187

theorem factorization_equality (a b : ℝ) : a^2 * b - a^3 = a^2 * (b - a) := by
  sorry

end factorization_equality_l191_19187


namespace units_digit_of_5_to_12_l191_19101

theorem units_digit_of_5_to_12 : ∃ n : ℕ, 5^12 ≡ 5 [ZMOD 10] :=
  sorry

end units_digit_of_5_to_12_l191_19101


namespace angle_300_in_fourth_quadrant_l191_19130

/-- An angle is in the fourth quadrant if it's between 270° and 360° (exclusive) -/
def is_in_fourth_quadrant (angle : ℝ) : Prop :=
  270 < angle ∧ angle < 360

/-- Prove that 300° is in the fourth quadrant -/
theorem angle_300_in_fourth_quadrant :
  is_in_fourth_quadrant 300 := by
  sorry

end angle_300_in_fourth_quadrant_l191_19130


namespace combined_selling_price_theorem_l191_19181

/-- Calculates the selling price of an article including profit and tax -/
def sellingPrice (cost : ℚ) (profitPercent : ℚ) (taxRate : ℚ) : ℚ :=
  let priceBeforeTax := cost * (1 + profitPercent)
  priceBeforeTax * (1 + taxRate)

/-- Calculates the combined selling price of three articles -/
def combinedSellingPrice (cost1 cost2 cost3 : ℚ) (profit1 profit2 profit3 : ℚ) (taxRate : ℚ) : ℚ :=
  sellingPrice cost1 profit1 taxRate +
  sellingPrice cost2 profit2 taxRate +
  sellingPrice cost3 profit3 taxRate

theorem combined_selling_price_theorem (cost1 cost2 cost3 : ℚ) (profit1 profit2 profit3 : ℚ) (taxRate : ℚ) :
  combinedSellingPrice cost1 cost2 cost3 profit1 profit2 profit3 taxRate =
  sellingPrice 500 (45/100) (12/100) +
  sellingPrice 300 (30/100) (12/100) +
  sellingPrice 1000 (20/100) (12/100) := by
  sorry

end combined_selling_price_theorem_l191_19181


namespace pentagonal_prism_faces_l191_19168

/-- A polyhedron with pentagonal bases and lateral faces -/
structure PentagonalPrism where
  base_edges : ℕ
  base_count : ℕ
  lateral_faces : ℕ

/-- The total number of faces in a pentagonal prism -/
def total_faces (p : PentagonalPrism) : ℕ :=
  p.base_count + p.lateral_faces

/-- Theorem: A pentagonal prism has 7 faces in total -/
theorem pentagonal_prism_faces :
  ∀ (p : PentagonalPrism), 
    p.base_edges = 5 → 
    p.base_count = 2 → 
    p.lateral_faces = 5 → 
    total_faces p = 7 := by
  sorry

#check pentagonal_prism_faces

end pentagonal_prism_faces_l191_19168


namespace tan_105_degrees_l191_19170

theorem tan_105_degrees :
  Real.tan (105 * π / 180) = -2 - Real.sqrt 3 := by
  sorry

end tan_105_degrees_l191_19170


namespace survey_result_survey_result_proof_l191_19189

theorem survey_result (total_surveyed : ℕ) 
  (electrical_fire_believers : ℕ) 
  (hantavirus_believers : ℕ) : Prop :=
  let electrical_fire_percentage : ℚ := 754 / 1000
  let hantavirus_percentage : ℚ := 523 / 1000
  (electrical_fire_believers : ℚ) / (total_surveyed : ℚ) = electrical_fire_percentage ∧
  (hantavirus_believers : ℚ) / (electrical_fire_believers : ℚ) = hantavirus_percentage ∧
  hantavirus_believers = 31 →
  total_surveyed = 78

theorem survey_result_proof : survey_result 78 59 31 :=
sorry

end survey_result_survey_result_proof_l191_19189


namespace min_value_sqrt_expression_l191_19156

theorem min_value_sqrt_expression (x : ℝ) (hx : x > 0) :
  (Real.sqrt (x^4 + x^2 + 2*x + 1) + Real.sqrt (x^4 - 2*x^3 + 5*x^2 - 4*x + 1)) / x ≥ Real.sqrt 10 :=
by sorry

end min_value_sqrt_expression_l191_19156


namespace value_of_c_l191_19129

theorem value_of_c (k a b c : ℝ) (hk : k ≠ 0) (ha : a ≠ 0) (hb : b ≠ 0) 
  (h : 1 / (k * a) - 1 / (k * b) = 1 / c) : c = k * a * b / (b - a) := by
  sorry

end value_of_c_l191_19129


namespace simplify_and_evaluate_l191_19110

theorem simplify_and_evaluate (a : ℚ) (h : a = -2) :
  (2 / (a - 1) - 1 / a) / ((a^2 + a) / (a^2 - 2*a + 1)) = -3/4 := by
  sorry

end simplify_and_evaluate_l191_19110


namespace sufficient_not_necessary_condition_l191_19178

/-- The quadratic function f(x) = x^2 + tx - t -/
def f (t : ℝ) (x : ℝ) : ℝ := x^2 + t*x - t

/-- Theorem stating that t ≥ 0 is a sufficient but not necessary condition for f to have a root -/
theorem sufficient_not_necessary_condition (t : ℝ) :
  (∀ t ≥ 0, ∃ x, f t x = 0) ∧
  (∃ t < 0, ∃ x, f t x = 0) :=
sorry

end sufficient_not_necessary_condition_l191_19178


namespace problem_solution_l191_19122

theorem problem_solution : 3 * 3^4 + 9^30 / 9^28 = 324 := by
  sorry

end problem_solution_l191_19122


namespace selling_prices_correct_l191_19120

def calculate_selling_price (cost : ℚ) (profit_percent : ℚ) (tax_percent : ℚ) : ℚ :=
  let pre_tax_price := cost * (1 + profit_percent)
  pre_tax_price * (1 + tax_percent)

theorem selling_prices_correct : 
  let cost_A : ℚ := 650
  let cost_B : ℚ := 1200
  let cost_C : ℚ := 800
  let profit_A : ℚ := 1/10
  let profit_B : ℚ := 3/20
  let profit_C : ℚ := 1/5
  let tax : ℚ := 1/20
  
  (calculate_selling_price cost_A profit_A tax = 75075/100) ∧
  (calculate_selling_price cost_B profit_B tax = 1449) ∧
  (calculate_selling_price cost_C profit_C tax = 1008) :=
by sorry

end selling_prices_correct_l191_19120


namespace root_expression_equality_l191_19109

/-- Given a cubic polynomial f(t) with roots p, q, r, and expressions for x, y, z,
    prove that xyz - qrx - rpy - pqz = -674 -/
theorem root_expression_equality (p q r : ℝ) : 
  let f : ℝ → ℝ := fun t ↦ t^3 - 2022*t^2 + 2022*t - 337
  let x := (q-1)*((2022 - q)/(r-1) + (2022 - r)/(p-1))
  let y := (r-1)*((2022 - r)/(p-1) + (2022 - p)/(q-1))
  let z := (p-1)*((2022 - p)/(q-1) + (2022 - q)/(r-1))
  f p = 0 ∧ f q = 0 ∧ f r = 0 →
  x*y*z - q*r*x - r*p*y - p*q*z = -674 := by
sorry

end root_expression_equality_l191_19109


namespace A_intersect_B_is_singleton_one_l191_19100

def A : Set ℝ := {x | x ≤ 1}
def B : Set ℝ := {y | ∃ x, y = x^2 + 2*x + 2}

theorem A_intersect_B_is_singleton_one : A ∩ B = {1} := by sorry

end A_intersect_B_is_singleton_one_l191_19100


namespace bus_trip_speed_l191_19173

theorem bus_trip_speed (distance : ℝ) (speed_increase : ℝ) (time_decrease : ℝ) :
  distance = 210 ∧ speed_increase = 5 ∧ time_decrease = 1 →
  ∃ (original_speed : ℝ),
    distance / original_speed - time_decrease = distance / (original_speed + speed_increase) ∧
    original_speed = 30 :=
by sorry

end bus_trip_speed_l191_19173


namespace chameleon_distance_theorem_l191_19149

/-- A chameleon is a sequence of 3n letters, with exactly n occurrences of each of the letters a, b, and c -/
def Chameleon (n : ℕ) := { s : List Char // s.length = 3*n ∧ s.count 'a' = n ∧ s.count 'b' = n ∧ s.count 'c' = n }

/-- The number of swaps required to transform one chameleon into another -/
def swaps_required (n : ℕ) (X Y : Chameleon n) : ℕ := sorry

theorem chameleon_distance_theorem (n : ℕ) (hn : n > 0) (X : Chameleon n) :
  ∃ Y : Chameleon n, swaps_required n X Y ≥ (3 * n^2) / 2 := by
  sorry

end chameleon_distance_theorem_l191_19149


namespace removed_ball_number_l191_19105

theorem removed_ball_number (n : ℕ) (h1 : n > 0) :
  (n * (n + 1)) / 2 - 5048 = 2 :=
by
  sorry

end removed_ball_number_l191_19105


namespace f_difference_l191_19151

-- Define a linear function f
variable (f : ℝ → ℝ)

-- Define the linearity of f
variable (hf : ∀ x y : ℝ, ∀ c : ℝ, f (x + c * y) = f x + c * f y)

-- Define the condition f(d+1) - f(d) = 3 for all real numbers d
variable (h : ∀ d : ℝ, f (d + 1) - f d = 3)

-- State the theorem
theorem f_difference (f : ℝ → ℝ) (hf : ∀ x y : ℝ, ∀ c : ℝ, f (x + c * y) = f x + c * f y) 
  (h : ∀ d : ℝ, f (d + 1) - f d = 3) : 
  f 3 - f 5 = -6 := by sorry

end f_difference_l191_19151


namespace cistern_width_is_four_l191_19123

/-- Represents the dimensions and properties of a cistern --/
structure Cistern where
  length : ℝ
  width : ℝ
  depth : ℝ
  wetSurfaceArea : ℝ

/-- Calculates the total wet surface area of a cistern --/
def totalWetSurfaceArea (c : Cistern) : ℝ :=
  c.length * c.width + 2 * c.length * c.depth + 2 * c.width * c.depth

/-- Theorem stating that a cistern with given dimensions has a width of 4 meters --/
theorem cistern_width_is_four :
  ∃ (c : Cistern),
    c.length = 6 ∧
    c.depth = 1.25 ∧
    c.wetSurfaceArea = 49 ∧
    totalWetSurfaceArea c = c.wetSurfaceArea ∧
    c.width = 4 := by
  sorry


end cistern_width_is_four_l191_19123


namespace consecutive_sum_largest_l191_19126

theorem consecutive_sum_largest (n : ℕ) : 
  (n + (n+1) + (n+2) + (n+3) + (n+4) = 180) → (n+4 = 38) :=
by
  sorry

#check consecutive_sum_largest

end consecutive_sum_largest_l191_19126


namespace candy_store_spending_correct_l191_19180

/-- John's weekly allowance in dollars -/
def weekly_allowance : ℚ := 345 / 100

/-- Fraction of allowance spent at the arcade -/
def arcade_fraction : ℚ := 3 / 5

/-- Fraction of remaining allowance spent at the toy store -/
def toy_store_fraction : ℚ := 1 / 3

/-- Amount spent at the candy store -/
def candy_store_spending : ℚ := 
  weekly_allowance * (1 - arcade_fraction) * (1 - toy_store_fraction)

theorem candy_store_spending_correct : 
  candy_store_spending = 92 / 100 := by sorry

end candy_store_spending_correct_l191_19180


namespace quadratic_root_implies_k_l191_19176

theorem quadratic_root_implies_k (p k : ℝ) : 
  (∃ x : ℂ, 3 * x^2 + p * x + k = 0 ∧ x = 4 + 3*I) → k = 75 := by
  sorry

end quadratic_root_implies_k_l191_19176


namespace f_range_l191_19186

-- Define the function f
def f (x : ℝ) : ℝ := 3 * (x + 4)

-- State the theorem
theorem f_range :
  Set.range f = {y : ℝ | y < 18 ∨ y > 18} :=
by
  sorry


end f_range_l191_19186


namespace third_player_games_l191_19198

/-- Represents a table tennis game with three players -/
structure TableTennisGame where
  total_games : ℕ
  player1_games : ℕ
  player2_games : ℕ
  player3_games : ℕ

/-- The rules and conditions of the game -/
def valid_game (g : TableTennisGame) : Prop :=
  g.total_games = g.player1_games ∧
  g.total_games = g.player2_games + g.player3_games ∧
  g.player1_games = 21 ∧
  g.player2_games = 10

/-- Theorem stating that under the given conditions, the third player must have played 11 games -/
theorem third_player_games (g : TableTennisGame) (h : valid_game g) : 
  g.player3_games = 11 := by
  sorry

end third_player_games_l191_19198


namespace valid_arrangements_l191_19128

/-- The number of ways to arrange plates on a circular table. -/
def circularArrangements (blue red green orange yellow : ℕ) : ℕ :=
  (Nat.factorial (blue + red + green + orange + yellow)) /
  (Nat.factorial blue * Nat.factorial red * Nat.factorial green * 
   Nat.factorial orange * Nat.factorial yellow * 
   (blue + red + green + orange + yellow))

/-- The number of arrangements with green plates adjacent. -/
def greenAdjacentArrangements (blue red green orange yellow : ℕ) : ℕ :=
  (Nat.factorial (blue + red + 1 + orange + yellow)) /
  (Nat.factorial blue * Nat.factorial red * Nat.factorial 1 * 
   Nat.factorial orange * Nat.factorial yellow * 
   (blue + red + 1 + orange + yellow))

/-- The number of arrangements with orange plates adjacent. -/
def orangeAdjacentArrangements (blue red green orange yellow : ℕ) : ℕ :=
  (Nat.factorial (blue + red + green + 1 + yellow)) /
  (Nat.factorial blue * Nat.factorial red * Nat.factorial green * 
   Nat.factorial 1 * Nat.factorial yellow * 
   (blue + red + green + 1 + yellow))

/-- The number of arrangements with both green and orange plates adjacent. -/
def bothAdjacentArrangements (blue red green orange yellow : ℕ) : ℕ :=
  (Nat.factorial (blue + red + 1 + 1 + yellow)) /
  (Nat.factorial blue * Nat.factorial red * Nat.factorial 1 * 
   (blue + red + 1 + 1 + yellow))

/-- The main theorem stating the number of valid arrangements. -/
theorem valid_arrangements (blue red green orange yellow : ℕ) 
  (h_blue : blue = 6) (h_red : red = 3) (h_green : green = 3) 
  (h_orange : orange = 2) (h_yellow : yellow = 1) :
  circularArrangements blue red green orange yellow - 
  (greenAdjacentArrangements blue red green orange yellow + 
   orangeAdjacentArrangements blue red green orange yellow - 
   bothAdjacentArrangements blue red green orange yellow) =
  circularArrangements 6 3 3 2 1 - 
  (greenAdjacentArrangements 6 3 3 2 1 + 
   orangeAdjacentArrangements 6 3 3 2 1 - 
   bothAdjacentArrangements 6 3 3 2 1) := by
  sorry

end valid_arrangements_l191_19128


namespace range_of_g_l191_19117

def f (x : ℝ) : ℝ := 2 * x + 1

def g (x : ℝ) : ℝ := f (x^2 + 1)

theorem range_of_g :
  Set.range g = Set.Ici 3 :=
sorry

end range_of_g_l191_19117


namespace distance_to_school_l191_19172

/-- The distance to school given travel conditions -/
theorem distance_to_school : 
  ∀ (total_time speed_to speed_from : ℝ),
  total_time = 1 →
  speed_to = 5 →
  speed_from = 25 →
  ∃ (distance : ℝ),
    distance / speed_to + distance / speed_from = total_time ∧
    distance = 25 / 6 :=
by sorry

end distance_to_school_l191_19172


namespace kylie_final_coins_l191_19196

/-- Calculates the number of US coins Kylie has left after converting all coins and giving some away --/
def kylie_coins_left (initial_us : ℝ) (euro : ℝ) (canadian : ℝ) (given_away : ℝ) 
  (euro_to_us : ℝ) (canadian_to_us : ℝ) : ℝ :=
  initial_us + euro * euro_to_us + canadian * canadian_to_us - given_away

/-- Theorem stating that Kylie is left with 15.58 US coins --/
theorem kylie_final_coins : 
  kylie_coins_left 15 13 8 21 1.18 0.78 = 15.58 := by
  sorry

end kylie_final_coins_l191_19196


namespace triangle_theorem_l191_19171

/-- Triangle ABC with sides a, b, c opposite to angles A, B, C respectively --/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- The given condition a/(√3 * cos A) = c/sin C --/
def condition (t : Triangle) : Prop :=
  t.a / (Real.sqrt 3 * Real.cos t.A) = t.c / Real.sin t.C

/-- The theorem to be proved --/
theorem triangle_theorem (t : Triangle) (h : condition t) (ha : t.a = 6) :
  t.A = π / 3 ∧ 6 < t.b + t.c ∧ t.b + t.c ≤ 12 := by
  sorry

end triangle_theorem_l191_19171


namespace permutations_of_six_distinct_objects_l191_19160

theorem permutations_of_six_distinct_objects : Nat.factorial 6 = 720 := by
  sorry

end permutations_of_six_distinct_objects_l191_19160


namespace camel_cost_l191_19185

/-- The cost of animals in rupees -/
structure AnimalCosts where
  camel : ℚ
  horse : ℚ
  ox : ℚ
  elephant : ℚ

/-- The conditions given in the problem -/
def problem_conditions (costs : AnimalCosts) : Prop :=
  10 * costs.camel = 24 * costs.horse ∧
  16 * costs.horse = 4 * costs.ox ∧
  6 * costs.ox = 4 * costs.elephant ∧
  10 * costs.elephant = 130000

/-- The theorem stating that given the problem conditions, the cost of a camel is 5200 rupees -/
theorem camel_cost (costs : AnimalCosts) :
  problem_conditions costs → costs.camel = 5200 := by
  sorry

end camel_cost_l191_19185


namespace product_of_binary_and_ternary_l191_19163

/-- Converts a list of digits in a given base to its decimal representation -/
def to_decimal (digits : List Nat) (base : Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * base ^ i) 0

/-- The problem statement -/
theorem product_of_binary_and_ternary : 
  let binary := [1, 0, 1, 1]  -- 1101 in binary, least significant digit first
  let ternary := [2, 0, 2]    -- 202 in ternary, least significant digit first
  (to_decimal binary 2) * (to_decimal ternary 3) = 260 := by
  sorry

end product_of_binary_and_ternary_l191_19163


namespace jewelry_ensemble_orders_l191_19135

theorem jewelry_ensemble_orders (necklace_price bracelet_price earring_price ensemble_price : ℚ)
  (necklaces_sold bracelets_sold earrings_sold : ℕ)
  (total_amount : ℚ)
  (h1 : necklace_price = 25)
  (h2 : bracelet_price = 15)
  (h3 : earring_price = 10)
  (h4 : ensemble_price = 45)
  (h5 : necklaces_sold = 5)
  (h6 : bracelets_sold = 10)
  (h7 : earrings_sold = 20)
  (h8 : total_amount = 565) :
  (total_amount - (necklace_price * necklaces_sold + bracelet_price * bracelets_sold + earring_price * earrings_sold)) / ensemble_price = 2 := by
  sorry

end jewelry_ensemble_orders_l191_19135


namespace yogurt_combinations_l191_19121

/-- The number of yogurt flavors -/
def num_flavors : ℕ := 5

/-- The number of toppings -/
def num_toppings : ℕ := 7

/-- The number of toppings to choose -/
def toppings_to_choose : ℕ := 2

/-- The number of doubling options (double first, double second, or no doubling) -/
def doubling_options : ℕ := 3

/-- Calculates the number of ways to choose k items from n items -/
def choose (n k : ℕ) : ℕ :=
  if k > n then 0
  else (Nat.factorial n) / (Nat.factorial k * Nat.factorial (n - k))

theorem yogurt_combinations :
  num_flavors * choose num_toppings toppings_to_choose * doubling_options = 315 := by
  sorry

end yogurt_combinations_l191_19121


namespace kelly_games_theorem_l191_19114

/-- The number of games Kelly gives away -/
def games_given_away : ℕ := 15

/-- The number of games Kelly has left after giving some away -/
def games_left : ℕ := 35

/-- The initial number of games Kelly has -/
def initial_games : ℕ := games_given_away + games_left

theorem kelly_games_theorem : initial_games = 50 := by
  sorry

end kelly_games_theorem_l191_19114


namespace triangle_areas_l191_19103

-- Define the triangle ABC
structure Triangle :=
  (BC : ℝ)
  (AC : ℝ)
  (AB : ℝ)

-- Define the areas of the triangles formed by altitude and median
def AreaTriangles (t : Triangle) : (ℝ × ℝ × ℝ) :=
  sorry

-- Theorem statement
theorem triangle_areas (t : Triangle) 
  (h1 : t.BC = 3)
  (h2 : t.AC = 4)
  (h3 : t.AB = 5) :
  AreaTriangles t = (3, 0.84, 2.16) :=
sorry

end triangle_areas_l191_19103


namespace data_groups_is_six_l191_19142

/-- Given a dataset, calculate the number of groups it should be divided into -/
def calculateGroups (maxValue minValue interval : ℕ) : ℕ :=
  let range := maxValue - minValue
  let preliminaryGroups := (range + interval - 1) / interval
  preliminaryGroups

/-- Theorem stating that for the given conditions, the number of groups is 6 -/
theorem data_groups_is_six :
  calculateGroups 36 15 4 = 6 := by
  sorry

#eval calculateGroups 36 15 4

end data_groups_is_six_l191_19142


namespace B_max_at_45_l191_19148

/-- The binomial coefficient function -/
def choose (n k : ℕ) : ℕ := sorry

/-- The B_k function as defined in the problem -/
def B (k : ℕ) : ℝ := (choose 500 k) * (0.1 ^ k)

/-- Theorem stating that B(k) is maximum when k = 45 -/
theorem B_max_at_45 : ∀ k : ℕ, k ≤ 500 → B k ≤ B 45 := by sorry

end B_max_at_45_l191_19148


namespace range_of_a_l191_19139

theorem range_of_a (a : ℝ) : 2 * a ≠ a^2 ↔ a ≠ 0 ∧ a ≠ 2 := by
  sorry

end range_of_a_l191_19139


namespace isosceles_trapezoid_angles_l191_19124

/-- An isosceles trapezoid with inscribed and circumscribed circles -/
structure IsoscelesTrapezoid where
  height : ℝ
  circum_radius : ℝ
  height_ratio : height / circum_radius = Real.sqrt (2/3)

/-- The angles of an isosceles trapezoid -/
def trapezoid_angles (t : IsoscelesTrapezoid) : ℝ × ℝ := sorry

theorem isosceles_trapezoid_angles (t : IsoscelesTrapezoid) :
  trapezoid_angles t = (45, 135) := by sorry

end isosceles_trapezoid_angles_l191_19124


namespace seats_needed_l191_19177

/-- Given 58 children and 2 children per seat, prove that 29 seats are needed. -/
theorem seats_needed (total_children : ℕ) (children_per_seat : ℕ) (h1 : total_children = 58) (h2 : children_per_seat = 2) :
  total_children / children_per_seat = 29 := by
  sorry

end seats_needed_l191_19177


namespace michelle_candy_sugar_l191_19137

/-- The total grams of sugar in Michelle's candy purchase -/
def total_sugar (num_bars : ℕ) (sugar_per_bar : ℕ) (lollipop_sugar : ℕ) : ℕ :=
  num_bars * sugar_per_bar + lollipop_sugar

/-- Theorem: The total sugar in Michelle's candy purchase is 177 grams -/
theorem michelle_candy_sugar :
  total_sugar 14 10 37 = 177 := by sorry

end michelle_candy_sugar_l191_19137


namespace f_5_equals_56_l191_19107

def f (x : ℝ) : ℝ := 2*x^7 - 9*x^6 + 5*x^5 - 49*x^4 - 5*x^3 + 2*x^2 + x + 1

def horner_eval (coeffs : List ℝ) (x : ℝ) : ℝ :=
  coeffs.foldl (fun acc a => acc * x + a) 0

theorem f_5_equals_56 :
  f 5 = horner_eval [2, -9, 5, -49, -5, 2, 1, 1] 5 ∧
  horner_eval [2, -9, 5, -49, -5, 2, 1, 1] 5 = 56 := by
  sorry

end f_5_equals_56_l191_19107


namespace gasoline_spending_increase_l191_19155

theorem gasoline_spending_increase (P Q : ℝ) (P_increase : ℝ) (Q_decrease : ℝ) :
  P > 0 ∧ Q > 0 ∧ P_increase = 0.25 ∧ Q_decrease = 0.16 →
  (1 + 0.05) * (P * Q) = (P * (1 + P_increase)) * (Q * (1 - Q_decrease)) :=
by sorry

end gasoline_spending_increase_l191_19155


namespace dinner_price_problem_l191_19164

theorem dinner_price_problem (original_price : ℝ) : 
  -- John's payment (after discount and tip)
  (0.90 * original_price + 0.15 * original_price) -
  -- Jane's payment (after discount and tip)
  (0.90 * original_price + 0.15 * (0.90 * original_price)) = 0.54 →
  original_price = 36 := by
sorry

end dinner_price_problem_l191_19164


namespace smallest_right_triangle_area_l191_19190

theorem smallest_right_triangle_area :
  let a : ℝ := 7
  let b : ℝ := 8
  let c : ℝ := Real.sqrt (a^2 + b^2)
  let area : ℝ := (1/2) * a * Real.sqrt (c^2 - a^2)
  area = (7 * Real.sqrt 15) / 2 :=
by sorry

end smallest_right_triangle_area_l191_19190


namespace fractional_equation_solution_l191_19125

theorem fractional_equation_solution : 
  ∃ x : ℝ, (2 / (x - 1) = 1 / x) ∧ (x = -1) := by
  sorry

end fractional_equation_solution_l191_19125


namespace kelly_initial_games_kelly_initial_games_proof_l191_19150

theorem kelly_initial_games : ℕ → Prop :=
  fun initial : ℕ =>
    let found : ℕ := 31
    let give_away : ℕ := 105
    let remaining : ℕ := 6
    initial + found - give_away = remaining →
    initial = 80

-- The proof is omitted
theorem kelly_initial_games_proof : kelly_initial_games 80 := by sorry

end kelly_initial_games_kelly_initial_games_proof_l191_19150


namespace y_axis_intersection_l191_19152

/-- The line equation 4y + 3x = 24 -/
def line_equation (x y : ℝ) : Prop := 4 * y + 3 * x = 24

/-- The y-axis is defined as the set of points with x-coordinate equal to 0 -/
def on_y_axis (x y : ℝ) : Prop := x = 0

theorem y_axis_intersection :
  ∃ (x y : ℝ), line_equation x y ∧ on_y_axis x y ∧ x = 0 ∧ y = 6 := by
  sorry

end y_axis_intersection_l191_19152


namespace tangent_line_at_one_l191_19167

/-- The function f(x) = x^4 - 2x^3 -/
def f (x : ℝ) : ℝ := x^4 - 2*x^3

/-- The derivative of f(x) -/
def f_deriv (x : ℝ) : ℝ := 4*x^3 - 6*x^2

theorem tangent_line_at_one :
  let x₀ : ℝ := 1
  let y₀ : ℝ := f x₀
  let m : ℝ := f_deriv x₀
  ∀ x y : ℝ, y - y₀ = m * (x - x₀) ↔ y = -2*x + 1 :=
by sorry

end tangent_line_at_one_l191_19167


namespace essay_section_length_l191_19108

theorem essay_section_length 
  (intro_length : ℕ) 
  (conclusion_multiplier : ℕ) 
  (num_body_sections : ℕ) 
  (total_length : ℕ) 
  (h1 : intro_length = 450)
  (h2 : conclusion_multiplier = 3)
  (h3 : num_body_sections = 4)
  (h4 : total_length = 5000)
  : (total_length - (intro_length + intro_length * conclusion_multiplier)) / num_body_sections = 800 := by
  sorry

end essay_section_length_l191_19108


namespace tetrahedron_sphere_radii_relation_l191_19197

/-- Theorem about the relationship between radii of spheres in a tetrahedron -/
theorem tetrahedron_sphere_radii_relation 
  (r r_a r_b r_c r_d : ℝ) 
  (S_a S_b S_c S_d V : ℝ) 
  (h_r : r = 3 * V / (S_a + S_b + S_c + S_d))
  (h_r_a : 1 / r_a = (-S_a + S_b + S_c + S_d) / (3 * V))
  (h_r_b : 1 / r_b = (S_a - S_b + S_c + S_d) / (3 * V))
  (h_r_c : 1 / r_c = (S_a + S_b - S_c + S_d) / (3 * V))
  (h_r_d : 1 / r_d = (S_a + S_b + S_c - S_d) / (3 * V))
  (h_positive : r > 0 ∧ r_a > 0 ∧ r_b > 0 ∧ r_c > 0 ∧ r_d > 0) :
  1 / r_a + 1 / r_b + 1 / r_c + 1 / r_d = 2 / r :=
by
  sorry

end tetrahedron_sphere_radii_relation_l191_19197


namespace lincoln_county_houses_l191_19162

/-- The number of houses in Lincoln County after a housing boom -/
def houses_after_boom (original : ℕ) (new_built : ℕ) : ℕ :=
  original + new_built

/-- Theorem stating the total number of houses after the housing boom -/
theorem lincoln_county_houses :
  houses_after_boom 20817 97741 = 118558 := by
  sorry

end lincoln_county_houses_l191_19162


namespace total_balls_in_box_l191_19116

theorem total_balls_in_box (black_balls : ℕ) (white_balls : ℕ) : 
  black_balls = 8 →
  white_balls = 6 * black_balls →
  black_balls + white_balls = 56 := by
  sorry

end total_balls_in_box_l191_19116


namespace sum_lower_bound_l191_19131

theorem sum_lower_bound (a b : ℝ) (ha : 0 < a) (hb : 0 < b) (h : a + b + 3 = a * b) :
  a + b ≥ 6 := by
  sorry

end sum_lower_bound_l191_19131


namespace symmetric_points_sum_power_l191_19169

theorem symmetric_points_sum_power (m n : ℤ) : 
  (2*n - m = -14) → (m = 4) → (m + n)^2023 = -1 := by
  sorry

end symmetric_points_sum_power_l191_19169


namespace f_difference_l191_19140

/-- The function f(x) = x^4 + 3x^3 + 2x^2 + 7x -/
def f (x : ℝ) : ℝ := x^4 + 3*x^3 + 2*x^2 + 7*x

/-- Theorem: f(6) - f(-6) = 1380 -/
theorem f_difference : f 6 - f (-6) = 1380 := by
  sorry

end f_difference_l191_19140


namespace expression_simplification_and_evaluation_l191_19119

theorem expression_simplification_and_evaluation (a : ℕ) 
  (h1 : 2 * a + 1 < 3 * a + 3) 
  (h2 : 2 / 3 * (a - 1) ≤ 1 / 2 * (a + 1 / 3)) 
  (h3 : a ≠ 0) 
  (h4 : a ≠ 1) 
  (h5 : a ≠ 2) : 
  ∃ (result : ℕ), 
    ((a + 1 - (4 * a - 5) / (a - 1)) / (1 / a - 1 / (a^2 - a)) = a * (a - 2)) ∧ 
    (result = a * (a - 2)) ∧ 
    (result = 3 ∨ result = 8 ∨ result = 15) :=
sorry

end expression_simplification_and_evaluation_l191_19119


namespace arithmetic_expression_equality_l191_19194

theorem arithmetic_expression_equality : 2 + 3 * 4 - 5 + 6 = 15 := by
  sorry

end arithmetic_expression_equality_l191_19194


namespace unique_solution_equation_l191_19118

theorem unique_solution_equation : ∃! x : ℝ, (28 + 48 / x) * x = 1980 := by
  sorry

end unique_solution_equation_l191_19118


namespace inverse_81_mod_103_l191_19161

theorem inverse_81_mod_103 (h : (9⁻¹ : ZMod 103) = 65) : (81⁻¹ : ZMod 103) = 2 := by
  sorry

end inverse_81_mod_103_l191_19161


namespace square_divisibility_l191_19174

theorem square_divisibility (k : ℕ) (n : ℕ) : 
  (∃ m : ℕ, k ^ 2 = n * m) →  -- k^2 is divisible by n
  (∀ j : ℕ, j < k → ¬(∃ m : ℕ, j ^ 2 = n * m)) →  -- k is the least possible value
  k = 60 →  -- the least possible value of k is 60
  n = 3600 :=  -- the number that k^2 is divisible by is 3600
by
  sorry

end square_divisibility_l191_19174


namespace student_count_l191_19159

theorem student_count (n : ℕ) (ella : ℕ) : 
  (ella = 60) → -- Ella's position from best
  (n + 1 - ella = 60) → -- Ella's position from worst (n is total students minus 1)
  (n + 1 = 119) := by
sorry

end student_count_l191_19159


namespace function_positivity_condition_equiv_a_range_l191_19179

/-- The function f(x) = ax² - (2-a)x + 1 --/
def f (a x : ℝ) : ℝ := a * x^2 - (2 - a) * x + 1

/-- The function g(x) = x --/
def g (x : ℝ) : ℝ := x

/-- The theorem stating the equivalence of the condition and the range of a --/
theorem function_positivity_condition_equiv_a_range :
  ∀ a : ℝ, (∀ x : ℝ, max (f a x) (g x) > 0) ↔ (0 ≤ a ∧ a < 4 + 2 * Real.sqrt 3) := by
  sorry

end function_positivity_condition_equiv_a_range_l191_19179


namespace stickers_per_page_l191_19136

theorem stickers_per_page (total_stickers : ℕ) (total_pages : ℕ) 
  (h1 : total_stickers = 220) 
  (h2 : total_pages = 22) 
  (h3 : total_stickers > 0) 
  (h4 : total_pages > 0) : 
  total_stickers / total_pages = 10 :=
sorry

end stickers_per_page_l191_19136


namespace stack_height_is_3_meters_l191_19183

/-- The number of packages in a stack -/
def packages_per_stack : ℕ := 60

/-- The number of sheets in a package -/
def sheets_per_package : ℕ := 500

/-- The thickness of a single sheet in millimeters -/
def sheet_thickness : ℚ := 1/10

/-- The height of a stack in meters -/
def stack_height : ℚ := 3

/-- Theorem stating that the height of a stack of packages is 3 meters -/
theorem stack_height_is_3_meters :
  (packages_per_stack : ℚ) * sheets_per_package * sheet_thickness / 1000 = stack_height :=
by sorry

end stack_height_is_3_meters_l191_19183


namespace tuesday_monday_ratio_l191_19199

/-- Represents the number of visitors to a library on different days of the week -/
structure LibraryVisitors where
  monday : ℕ
  tuesday : ℕ
  remainingDaysAverage : ℕ
  totalWeek : ℕ

/-- The ratio of Tuesday visitors to Monday visitors is 2:1 -/
theorem tuesday_monday_ratio (v : LibraryVisitors) 
  (h1 : v.monday = 50)
  (h2 : v.remainingDaysAverage = 20)
  (h3 : v.totalWeek = 250)
  (h4 : v.totalWeek = v.monday + v.tuesday + 5 * v.remainingDaysAverage) :
  v.tuesday / v.monday = 2 := by
  sorry

#check tuesday_monday_ratio

end tuesday_monday_ratio_l191_19199


namespace max_area_rectangle_l191_19188

/-- The perimeter of the rectangle formed by matches --/
def perimeter : ℕ := 22

/-- Function to calculate the area of a rectangle given its length and width --/
def area (length width : ℕ) : ℕ := length * width

/-- Theorem stating that the rectangle with dimensions 6 × 5 has the maximum area
    among all rectangles with a perimeter of 22 units --/
theorem max_area_rectangle :
  ∀ l w : ℕ, 
    2 * (l + w) = perimeter → 
    area l w ≤ area 6 5 :=
by sorry

end max_area_rectangle_l191_19188


namespace exists_valid_coloring_l191_19138

/-- A coloring of positive integers -/
def Coloring := ℕ+ → Fin 2009

/-- Predicate for a valid coloring satisfying the problem conditions -/
def ValidColoring (f : Coloring) : Prop :=
  (∀ c : Fin 2009, Set.Infinite {n : ℕ+ | f n = c}) ∧
  (∀ a b c : ℕ+, ∀ i j k : Fin 2009,
    i ≠ j ∧ j ≠ k ∧ i ≠ k → f a = i ∧ f b = j ∧ f c = k → a * b ≠ c)

/-- Theorem stating the existence of a valid coloring -/
theorem exists_valid_coloring : ∃ f : Coloring, ValidColoring f := by
  sorry

end exists_valid_coloring_l191_19138


namespace symmetric_function_theorem_l191_19132

/-- A function is symmetric to another function with respect to the origin -/
def SymmetricToOrigin (f g : ℝ → ℝ) : Prop :=
  ∀ x y, f x = y ↔ g (-x) = -y

/-- The main theorem -/
theorem symmetric_function_theorem (f : ℝ → ℝ) :
  SymmetricToOrigin f (λ x ↦ 3 - 2*x) → ∀ x, f x = -2*x - 3 := by
  sorry

end symmetric_function_theorem_l191_19132


namespace nikita_produces_two_per_hour_l191_19133

-- Define the productivity of Ivan and Nikita
def ivan_productivity : ℝ := sorry
def nikita_productivity : ℝ := sorry

-- Define the conditions from the problem
axiom monday_condition : 3 * ivan_productivity + 2 * nikita_productivity = 7
axiom tuesday_condition : 5 * ivan_productivity + 3 * nikita_productivity = 11

-- Theorem to prove
theorem nikita_produces_two_per_hour : nikita_productivity = 2 := by
  sorry

end nikita_produces_two_per_hour_l191_19133


namespace set_A_properties_l191_19115

def A : Set ℝ := {x | x^2 - 1 = 0}

theorem set_A_properties : 
  (1 ∈ A) ∧ (∅ ⊆ A) ∧ ({1, -1} ⊆ A) := by sorry

end set_A_properties_l191_19115
