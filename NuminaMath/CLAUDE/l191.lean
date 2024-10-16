import Mathlib

namespace NUMINAMATH_CALUDE_workshop_inspection_problem_l191_19144

-- Define the number of products produced each day
variable (n : ℕ)

-- Define the probability of passing inspection on the first day
def prob_pass_first_day : ℚ := 3/5

-- Define the probability of passing inspection on the second day
def prob_pass_second_day (n : ℕ) : ℚ := (n - 2).choose 4 / n.choose 4

-- Define the probability of passing inspection on both days
def prob_pass_both_days (n : ℕ) : ℚ := prob_pass_first_day * prob_pass_second_day n

-- Define the probability of passing inspection on at least one day
def prob_pass_at_least_one_day (n : ℕ) : ℚ := 1 - (1 - prob_pass_first_day) * (1 - prob_pass_second_day n)

-- Theorem statement
theorem workshop_inspection_problem (n : ℕ) :
  (prob_pass_first_day = (n - 1).choose 4 / n.choose 4) →
  (n = 10) ∧
  (prob_pass_both_days n = 1/5) ∧
  (prob_pass_at_least_one_day n = 11/15) := by
  sorry


end NUMINAMATH_CALUDE_workshop_inspection_problem_l191_19144


namespace NUMINAMATH_CALUDE_optimal_price_maximizes_profit_l191_19178

/-- Represents the profit function for a product sale scenario -/
def profit_function (x : ℝ) : ℝ := -10 * x^2 + 220 * x - 960

/-- Represents the optimal selling price that maximizes profit -/
def optimal_price : ℝ := 11

theorem optimal_price_maximizes_profit :
  ∀ x : ℝ, profit_function x ≤ profit_function optimal_price :=
sorry

#check optimal_price_maximizes_profit

end NUMINAMATH_CALUDE_optimal_price_maximizes_profit_l191_19178


namespace NUMINAMATH_CALUDE_win_sector_area_l191_19129

theorem win_sector_area (r : ℝ) (p : ℝ) (h1 : r = 12) (h2 : p = 1/3) :
  p * π * r^2 = 48 * π := by
  sorry

end NUMINAMATH_CALUDE_win_sector_area_l191_19129


namespace NUMINAMATH_CALUDE_sams_age_l191_19158

theorem sams_age (sam drew : ℕ) 
  (h1 : sam + drew = 54)
  (h2 : sam = drew / 2) :
  sam = 18 := by
sorry

end NUMINAMATH_CALUDE_sams_age_l191_19158


namespace NUMINAMATH_CALUDE_arthurs_walk_distance_l191_19116

/-- Represents the distance walked in each direction --/
structure WalkDistance where
  east : ℕ
  north : ℕ
  west : ℕ

/-- Calculates the total distance walked in miles --/
def total_distance (walk : WalkDistance) (block_length : ℚ) : ℚ :=
  ((walk.east + walk.north + walk.west) : ℚ) * block_length

/-- Theorem: Arthur's walk totals 6.5 miles --/
theorem arthurs_walk_distance :
  let walk := WalkDistance.mk 8 15 3
  let block_length : ℚ := 1/4
  total_distance walk block_length = 13/2 := by
  sorry

end NUMINAMATH_CALUDE_arthurs_walk_distance_l191_19116


namespace NUMINAMATH_CALUDE_shekar_science_score_l191_19172

def average_marks : ℝ := 77
def num_subjects : ℕ := 5
def math_score : ℝ := 76
def social_studies_score : ℝ := 82
def english_score : ℝ := 67
def biology_score : ℝ := 95

theorem shekar_science_score :
  ∃ (science_score : ℝ),
    (math_score + social_studies_score + english_score + biology_score + science_score) / num_subjects = average_marks ∧
    science_score = 65 := by
  sorry

end NUMINAMATH_CALUDE_shekar_science_score_l191_19172


namespace NUMINAMATH_CALUDE_carl_personal_share_l191_19175

/-- Carl's car accident costs and insurance coverage -/
structure AccidentCost where
  propertyDamage : ℝ
  medicalBills : ℝ
  insuranceCoverage : ℝ

/-- Calculate Carl's personal share of the accident costs -/
def calculatePersonalShare (cost : AccidentCost) : ℝ :=
  (cost.propertyDamage + cost.medicalBills) * (1 - cost.insuranceCoverage)

/-- Theorem stating that Carl's personal share is $22,000 -/
theorem carl_personal_share :
  let cost : AccidentCost := {
    propertyDamage := 40000,
    medicalBills := 70000,
    insuranceCoverage := 0.8
  }
  calculatePersonalShare cost = 22000 := by
  sorry


end NUMINAMATH_CALUDE_carl_personal_share_l191_19175


namespace NUMINAMATH_CALUDE_lcm_gcd_problem_l191_19146

theorem lcm_gcd_problem (a b : ℕ+) 
  (h1 : Nat.lcm a b = 8820)
  (h2 : Nat.gcd a b = 36)
  (h3 : a = 360) :
  b = 882 := by
  sorry

end NUMINAMATH_CALUDE_lcm_gcd_problem_l191_19146


namespace NUMINAMATH_CALUDE_square_sum_reciprocal_l191_19155

theorem square_sum_reciprocal (x : ℝ) (h : x + 1/x = 8) : x^2 + 1/x^2 = 62 := by
  sorry

end NUMINAMATH_CALUDE_square_sum_reciprocal_l191_19155


namespace NUMINAMATH_CALUDE_intersection_of_M_and_N_l191_19179

-- Define the sets M and N
def M : Set ℝ := {x | -3 ≤ x ∧ x < 4}
def N : Set ℝ := {x | x^2 - 2*x - 8 ≤ 0}

-- State the theorem
theorem intersection_of_M_and_N : M ∩ N = {x : ℝ | -2 ≤ x ∧ x < 4} := by sorry

end NUMINAMATH_CALUDE_intersection_of_M_and_N_l191_19179


namespace NUMINAMATH_CALUDE_car_profit_percentage_l191_19118

theorem car_profit_percentage (original_price : ℝ) (h : original_price > 0) :
  let discount_rate := 0.20
  let purchase_price := original_price * (1 - discount_rate)
  let sale_price := purchase_price * 2
  let profit := sale_price - original_price
  let profit_percentage := (profit / original_price) * 100
  profit_percentage = 60 := by sorry

end NUMINAMATH_CALUDE_car_profit_percentage_l191_19118


namespace NUMINAMATH_CALUDE_solve_salt_merchant_problem_l191_19169

def salt_merchant_problem (initial_purchase : ℝ) (profit1 : ℝ) (profit2 : ℝ) : Prop :=
  let revenue1 := initial_purchase + profit1
  let profit_rate := profit2 / revenue1
  profit_rate * initial_purchase = profit1 ∧ profit1 = 100 ∧ profit2 = 120

theorem solve_salt_merchant_problem :
  ∃ (initial_purchase : ℝ), salt_merchant_problem initial_purchase 100 120 ∧ initial_purchase = 500 :=
by
  sorry

end NUMINAMATH_CALUDE_solve_salt_merchant_problem_l191_19169


namespace NUMINAMATH_CALUDE_equation_solution_l191_19139

theorem equation_solution : ∃ (x₁ x₂ : ℝ), x₁ = 1 ∧ x₂ = 2/3 ∧ 
  (∀ x : ℝ, 3*x*(x-1) = 2*x-2 ↔ (x = x₁ ∨ x = x₂)) := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l191_19139


namespace NUMINAMATH_CALUDE_existence_of_x0_l191_19101

-- Define the function f
variable (f : ℝ → ℝ)

-- State the theorem
theorem existence_of_x0 
  (hcont : ContinuousOn f (Set.Icc 0 1))
  (hdiff : DifferentiableOn ℝ f (Set.Ioo 0 1))
  (hf0 : f 0 = 1)
  (hf1 : f 1 = 0) :
  ∃ x0 : ℝ, 0 < x0 ∧ x0 < 1 ∧ 
    |deriv f x0| ≥ 2018 * (f x0)^2018 := by
  sorry

end NUMINAMATH_CALUDE_existence_of_x0_l191_19101


namespace NUMINAMATH_CALUDE_initial_mixture_volume_l191_19190

/-- Given a mixture of wine and water where:
  * The initial mixture contains 20% water
  * Adding 8 liters of water increases the water percentage to 25%
  This theorem proves that the initial volume of the mixture is 120 liters. -/
theorem initial_mixture_volume (V : ℝ) : 
  (0.20 * V = 0.25 * (V + 8)) → V = 120 := by sorry

end NUMINAMATH_CALUDE_initial_mixture_volume_l191_19190


namespace NUMINAMATH_CALUDE_max_value_under_constraints_l191_19145

theorem max_value_under_constraints (x y : ℝ) 
  (h1 : 4 * x + 3 * y ≤ 9) 
  (h2 : 3 * x + 5 * y ≤ 12) : 
  2 * x + y ≤ 39 / 11 := by
  sorry

end NUMINAMATH_CALUDE_max_value_under_constraints_l191_19145


namespace NUMINAMATH_CALUDE_escalator_speed_l191_19191

/-- The speed of an escalator given its length, a person's walking speed, and the time taken to cover the entire length. -/
theorem escalator_speed (escalator_length : ℝ) (walking_speed : ℝ) (time_taken : ℝ) :
  escalator_length = 126 ∧ walking_speed = 3 ∧ time_taken = 9 →
  ∃ (escalator_speed : ℝ), 
    escalator_speed = 11 ∧ 
    (escalator_speed + walking_speed) * time_taken = escalator_length :=
by sorry

end NUMINAMATH_CALUDE_escalator_speed_l191_19191


namespace NUMINAMATH_CALUDE_max_value_4x_3y_l191_19111

theorem max_value_4x_3y (x y : ℝ) :
  x^2 + y^2 = 16*x + 8*y + 8 →
  4*x + 3*y ≤ Real.sqrt (5184 - 173.33) - 72 :=
by sorry

end NUMINAMATH_CALUDE_max_value_4x_3y_l191_19111


namespace NUMINAMATH_CALUDE_peaches_in_basket_c_l191_19119

theorem peaches_in_basket_c (total_baskets : ℕ) (avg_fruits : ℕ) 
  (fruits_a : ℕ) (fruits_b : ℕ) (fruits_d : ℕ) (fruits_e : ℕ) :
  total_baskets = 5 →
  avg_fruits = 25 →
  fruits_a = 15 →
  fruits_b = 30 →
  fruits_d = 25 →
  fruits_e = 35 →
  (total_baskets * avg_fruits) - (fruits_a + fruits_b + fruits_d + fruits_e) = 20 :=
by
  sorry

end NUMINAMATH_CALUDE_peaches_in_basket_c_l191_19119


namespace NUMINAMATH_CALUDE_max_planes_for_10_points_l191_19106

/-- The number of points in space -/
def n : ℕ := 10

/-- The number of points required to determine a plane -/
def k : ℕ := 3

/-- Assumption that no three points are collinear -/
axiom no_collinear : True

/-- The maximum number of planes determined by n points in space -/
def max_planes (n : ℕ) : ℕ := Nat.choose n k

theorem max_planes_for_10_points : max_planes n = 120 := by sorry

end NUMINAMATH_CALUDE_max_planes_for_10_points_l191_19106


namespace NUMINAMATH_CALUDE_cross_product_perpendicular_l191_19174

def v1 : ℝ × ℝ × ℝ := (4, 3, -5)
def v2 : ℝ × ℝ × ℝ := (2, -1, 4)

def cross_product (a b : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  let (a₁, a₂, a₃) := a
  let (b₁, b₂, b₃) := b
  (a₂ * b₃ - a₃ * b₂, a₃ * b₁ - a₁ * b₃, a₁ * b₂ - a₂ * b₁)

def dot_product (a b : ℝ × ℝ × ℝ) : ℝ :=
  let (a₁, a₂, a₃) := a
  let (b₁, b₂, b₃) := b
  a₁ * b₁ + a₂ * b₂ + a₃ * b₃

theorem cross_product_perpendicular :
  let result := cross_product v1 v2
  result = (7, -26, -10) ∧
  dot_product v1 result = 0 ∧
  dot_product v2 result = 0 := by
  sorry

end NUMINAMATH_CALUDE_cross_product_perpendicular_l191_19174


namespace NUMINAMATH_CALUDE_cos_90_degrees_l191_19134

theorem cos_90_degrees : Real.cos (π / 2) = 0 := by
  sorry

end NUMINAMATH_CALUDE_cos_90_degrees_l191_19134


namespace NUMINAMATH_CALUDE_largest_coefficients_in_expansion_l191_19177

def binomial_coefficient (n k : ℕ) : ℕ := sorry

def expansion_term (n r : ℕ) : ℕ := 2^r * binomial_coefficient n r

theorem largest_coefficients_in_expansion (n : ℕ) (h : n = 11) :
  (∀ k, k ≠ 5 ∧ k ≠ 6 → expansion_term n 5 ≥ expansion_term n k) ∧
  (∀ k, k ≠ 5 ∧ k ≠ 6 → expansion_term n 6 ≥ expansion_term n k) ∧
  expansion_term n 7 = expansion_term n 8 ∧
  expansion_term n 7 = 42240 ∧
  (∀ k, k ≠ 7 ∧ k ≠ 8 → expansion_term n 7 > expansion_term n k) :=
sorry

end NUMINAMATH_CALUDE_largest_coefficients_in_expansion_l191_19177


namespace NUMINAMATH_CALUDE_great_fourteen_soccer_league_games_l191_19157

theorem great_fourteen_soccer_league_games (teams_per_division : ℕ) 
  (intra_division_games : ℕ) (inter_division_games : ℕ) : 
  teams_per_division = 7 →
  intra_division_games = 3 →
  inter_division_games = 1 →
  (teams_per_division * (
    (teams_per_division - 1) * intra_division_games + 
    teams_per_division * inter_division_games
  )) / 2 = 175 := by
  sorry

end NUMINAMATH_CALUDE_great_fourteen_soccer_league_games_l191_19157


namespace NUMINAMATH_CALUDE_sum_difference_absolute_values_l191_19154

theorem sum_difference_absolute_values : 
  (3 + (-4) + (-5)) - (|3| + |-4| + |-5|) = -18 := by
  sorry

end NUMINAMATH_CALUDE_sum_difference_absolute_values_l191_19154


namespace NUMINAMATH_CALUDE_ellipse_standard_equation_l191_19196

/-- Given an ellipse with specific properties, prove its standard equation -/
theorem ellipse_standard_equation (a b c : ℝ) (h1 : a > b) (h2 : b > 0) 
  (h3 : c / a = Real.sqrt 3 / 3) 
  (h4 : 2 * b^2 / a = 4 * Real.sqrt 3 / 3) 
  (h5 : a^2 = b^2 + c^2) :
  ∃ (x y : ℝ), x^2 / 3 + y^2 / 2 = 1 ∧ 
    x^2 / a^2 + y^2 / b^2 = 1 := by sorry

end NUMINAMATH_CALUDE_ellipse_standard_equation_l191_19196


namespace NUMINAMATH_CALUDE_youngest_child_age_l191_19152

def arithmetic_progression (a₁ : ℕ) (d : ℕ) (n : ℕ) : List ℕ :=
  List.range n |>.map (λ i => a₁ + i * d)

theorem youngest_child_age 
  (children : ℕ) 
  (ages : List ℕ) 
  (h_children : children = 8)
  (h_ages : ages = arithmetic_progression 2 3 7)
  : ages.head? = some 2 := by
  sorry

#eval arithmetic_progression 2 3 7

end NUMINAMATH_CALUDE_youngest_child_age_l191_19152


namespace NUMINAMATH_CALUDE_optimal_circular_sector_radius_l191_19163

/-- The radius that maximizes the area of a circular sector with given constraints -/
theorem optimal_circular_sector_radius : 
  ∀ (r : ℝ) (s : ℝ),
  -- Total perimeter is 32 meters
  2 * r + s = 32 →
  -- Ratio of radius to arc length is at least 2:3
  r / s ≥ 2 / 3 →
  -- Area of the sector is maximized
  ∀ (r' : ℝ) (s' : ℝ),
  2 * r' + s' = 32 →
  r' / s' ≥ 2 / 3 →
  r * s ≥ r' * s' →
  -- The optimal radius is 64/7
  r = 64 / 7 :=
by sorry

end NUMINAMATH_CALUDE_optimal_circular_sector_radius_l191_19163


namespace NUMINAMATH_CALUDE_triangle_parallelepiped_analogy_inappropriate_l191_19138

/-- A shape in a geometric space -/
inductive GeometricShape
  | Triangle
  | Parallelepiped
  | TriangularPyramid

/-- The dimension of a geometric space -/
inductive Dimension
  | Plane
  | Space

/-- A function that determines if two shapes form an appropriate analogy across dimensions -/
def appropriateAnalogy (shape1 : GeometricShape) (dim1 : Dimension) 
                       (shape2 : GeometricShape) (dim2 : Dimension) : Prop :=
  sorry

/-- Theorem stating that comparing a triangle in a plane to a parallelepiped in space 
    is not an appropriate analogy -/
theorem triangle_parallelepiped_analogy_inappropriate :
  ¬(appropriateAnalogy GeometricShape.Triangle Dimension.Plane 
                       GeometricShape.Parallelepiped Dimension.Space) :=
by sorry

end NUMINAMATH_CALUDE_triangle_parallelepiped_analogy_inappropriate_l191_19138


namespace NUMINAMATH_CALUDE_parallel_vectors_magnitude_l191_19131

/-- Given vectors a and b in ℝ³, where a is parallel to b, prove that the magnitude of b is 3√6 -/
theorem parallel_vectors_magnitude (a b : ℝ × ℝ × ℝ) : 
  a = (-1, 2, 1) → 
  b.1 = 3 → 
  ∃ (k : ℝ), b = k • a → 
  ‖b‖ = 3 * Real.sqrt 6 := by
  sorry


end NUMINAMATH_CALUDE_parallel_vectors_magnitude_l191_19131


namespace NUMINAMATH_CALUDE_sale_price_lower_than_original_l191_19105

theorem sale_price_lower_than_original : ∀ x : ℝ, x > 0 → 0.75 * (1.3 * x) < x := by
  sorry

end NUMINAMATH_CALUDE_sale_price_lower_than_original_l191_19105


namespace NUMINAMATH_CALUDE_smiths_age_problem_l191_19185

/-- Represents a 4-digit number in the form abba -/
def mirroredNumber (a b : Nat) : Nat :=
  1000 * a + 100 * b + 10 * b + a

theorem smiths_age_problem :
  ∃! n : Nat,
    59 < n ∧ n < 100 ∧
    (∃ b : Nat, b < 10 ∧ (mirroredNumber (n / 10) b) % 7 = 0) ∧
    n = 67 := by
  sorry

end NUMINAMATH_CALUDE_smiths_age_problem_l191_19185


namespace NUMINAMATH_CALUDE_sum_of_coefficients_is_27_l191_19164

-- Define the polynomial
def p (x : ℝ) : ℝ := 5 * (2 * x^8 - 3 * x^5 + 4) + 6 * (x^6 + 9 * x^3 - 8)

-- Theorem statement
theorem sum_of_coefficients_is_27 : 
  p 1 = 27 := by sorry

end NUMINAMATH_CALUDE_sum_of_coefficients_is_27_l191_19164


namespace NUMINAMATH_CALUDE_derivative_zero_at_one_l191_19125

theorem derivative_zero_at_one (a : ℝ) : 
  let f : ℝ → ℝ := λ x => (x^2 + a) / (x + 1)
  (deriv f 1 = 0) → a = 3 := by
  sorry

end NUMINAMATH_CALUDE_derivative_zero_at_one_l191_19125


namespace NUMINAMATH_CALUDE_super_vcd_cost_price_l191_19193

theorem super_vcd_cost_price (x : ℝ) : 
  x * (1 + 0.4) * 0.9 - 50 = x + 340 → x = 1500 := by sorry

end NUMINAMATH_CALUDE_super_vcd_cost_price_l191_19193


namespace NUMINAMATH_CALUDE_remaining_water_l191_19147

/-- The amount of distilled water remaining after two experiments -/
theorem remaining_water (initial : ℚ) (used1 : ℚ) (used2 : ℚ) :
  initial = 3 →
  used1 = 5/4 →
  used2 = 1/3 →
  initial - (used1 + used2) = 17/12 := by
sorry

end NUMINAMATH_CALUDE_remaining_water_l191_19147


namespace NUMINAMATH_CALUDE_greatest_average_speed_l191_19108

/-- Checks if a number is a palindrome -/
def is_palindrome (n : ℕ) : Prop := sorry

/-- Finds the greatest palindrome less than or equal to a given number -/
def greatest_palindrome_le (n : ℕ) : ℕ := sorry

theorem greatest_average_speed (initial_reading : ℕ) (trip_duration : ℕ) (max_speed : ℕ) :
  is_palindrome initial_reading →
  initial_reading = 13831 →
  trip_duration = 5 →
  max_speed = 80 →
  let max_distance := max_speed * trip_duration
  let max_final_reading := initial_reading + max_distance
  let actual_final_reading := greatest_palindrome_le max_final_reading
  let distance_traveled := actual_final_reading - initial_reading
  let average_speed := distance_traveled / trip_duration
  average_speed = 62 := by sorry

end NUMINAMATH_CALUDE_greatest_average_speed_l191_19108


namespace NUMINAMATH_CALUDE_probability_of_same_group_l191_19127

def card_count : ℕ := 20
def people_count : ℕ := 4
def first_drawn : ℕ := 5
def second_drawn : ℕ := 14

def same_group_probability : ℚ := 7 / 51

theorem probability_of_same_group :
  let remaining_cards := card_count - people_count + 2
  let favorable_outcomes := (card_count - second_drawn) * (card_count - second_drawn - 1) +
                            (first_drawn - 1) * (first_drawn - 2)
  let total_outcomes := remaining_cards * (remaining_cards - 1)
  (favorable_outcomes : ℚ) / total_outcomes = same_group_probability :=
sorry

end NUMINAMATH_CALUDE_probability_of_same_group_l191_19127


namespace NUMINAMATH_CALUDE_sixteenth_root_of_sixteen_l191_19140

theorem sixteenth_root_of_sixteen (n : ℝ) : (16 : ℝ) ^ (1/4 : ℝ) = 2^n → n = 1 := by
  sorry

end NUMINAMATH_CALUDE_sixteenth_root_of_sixteen_l191_19140


namespace NUMINAMATH_CALUDE_quadratic_inequality_l191_19143

theorem quadratic_inequality (x : ℝ) : x^2 + 5*x + 6 > 0 ↔ x < -3 ∨ x > -2 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_l191_19143


namespace NUMINAMATH_CALUDE_right_angled_triangle_sides_l191_19109

theorem right_angled_triangle_sides : 
  (∃ (a b c : ℕ), (a = 5 ∧ b = 3 ∧ c = 4) ∧ a^2 = b^2 + c^2) ∧
  (∀ (a b c : ℕ), (a = 2 ∧ b = 3 ∧ c = 4) → a^2 ≠ b^2 + c^2) ∧
  (∀ (a b c : ℕ), (a = 4 ∧ b = 6 ∧ c = 9) → a^2 ≠ b^2 + c^2) ∧
  (∀ (a b c : ℕ), (a = 5 ∧ b = 11 ∧ c = 13) → a^2 ≠ b^2 + c^2) :=
by sorry

end NUMINAMATH_CALUDE_right_angled_triangle_sides_l191_19109


namespace NUMINAMATH_CALUDE_opposite_numbers_example_l191_19181

theorem opposite_numbers_example : -(-(5 : ℤ)) = -(-|5|) → -(-(5 : ℤ)) + (-|5|) = 0 := by
  sorry

end NUMINAMATH_CALUDE_opposite_numbers_example_l191_19181


namespace NUMINAMATH_CALUDE_polygon_diagonals_equal_sides_l191_19188

theorem polygon_diagonals_equal_sides : ∃ (n : ℕ), n > 0 ∧ n * (n - 3) / 2 = n := by
  sorry

end NUMINAMATH_CALUDE_polygon_diagonals_equal_sides_l191_19188


namespace NUMINAMATH_CALUDE_parking_lot_cars_l191_19126

theorem parking_lot_cars (initial_cars : ℕ) (cars_left : ℕ) (extra_cars_entered : ℕ) :
  initial_cars = 80 →
  cars_left = 13 →
  extra_cars_entered = 5 →
  initial_cars - cars_left + (cars_left + extra_cars_entered) = 85 :=
by
  sorry

end NUMINAMATH_CALUDE_parking_lot_cars_l191_19126


namespace NUMINAMATH_CALUDE_corn_purchase_proof_l191_19137

/-- The cost of corn in dollars per pound -/
def corn_cost : ℝ := 0.99

/-- The cost of beans in dollars per pound -/
def bean_cost : ℝ := 0.75

/-- The total weight of corn and beans in pounds -/
def total_weight : ℝ := 20

/-- The total cost in dollars -/
def total_cost : ℝ := 16.80

/-- The number of pounds of corn purchased -/
def corn_weight : ℝ := 7.5

theorem corn_purchase_proof :
  ∃ (bean_weight : ℝ),
    bean_weight ≥ 0 ∧
    corn_weight ≥ 0 ∧
    bean_weight + corn_weight = total_weight ∧
    bean_cost * bean_weight + corn_cost * corn_weight = total_cost :=
by sorry

end NUMINAMATH_CALUDE_corn_purchase_proof_l191_19137


namespace NUMINAMATH_CALUDE_system_solution_l191_19156

theorem system_solution :
  ∃! (x y : ℚ), (4 * x - 3 * y = -7) ∧ (5 * x + 4 * y = -2) ∧ x = -34/31 ∧ y = 27/31 := by
  sorry

end NUMINAMATH_CALUDE_system_solution_l191_19156


namespace NUMINAMATH_CALUDE_average_rounds_is_three_l191_19124

/-- Represents the distribution of rounds played by golfers -/
structure GolfDistribution where
  rounds1 : Nat
  rounds2 : Nat
  rounds3 : Nat
  rounds4 : Nat
  rounds5 : Nat

/-- Calculates the average number of rounds played, rounded to the nearest whole number -/
def averageRounds (dist : GolfDistribution) : Nat :=
  let totalRounds := dist.rounds1 * 1 + dist.rounds2 * 2 + dist.rounds3 * 3 + dist.rounds4 * 4 + dist.rounds5 * 5
  let totalGolfers := dist.rounds1 + dist.rounds2 + dist.rounds3 + dist.rounds4 + dist.rounds5
  (totalRounds + totalGolfers / 2) / totalGolfers

theorem average_rounds_is_three (dist : GolfDistribution) 
  (h1 : dist.rounds1 = 4)
  (h2 : dist.rounds2 = 3)
  (h3 : dist.rounds3 = 3)
  (h4 : dist.rounds4 = 2)
  (h5 : dist.rounds5 = 6) :
  averageRounds dist = 3 := by
  sorry

#eval averageRounds { rounds1 := 4, rounds2 := 3, rounds3 := 3, rounds4 := 2, rounds5 := 6 }

end NUMINAMATH_CALUDE_average_rounds_is_three_l191_19124


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_condition_l191_19168

/-- A quadratic function f(x) = x^2 + 2ax - 3 -/
def f (a : ℝ) (x : ℝ) : ℝ := x^2 + 2*a*x - 3

/-- The condition a > -1 is sufficient but not necessary for f to be monotonically increasing on (1, +∞) -/
theorem sufficient_not_necessary_condition (a : ℝ) :
  (a > -1 → ∀ x y, 1 < x → x < y → f a x < f a y) ∧
  ¬(∀ x y, 1 < x → x < y → f a x < f a y → a > -1) :=
sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_condition_l191_19168


namespace NUMINAMATH_CALUDE_middle_number_is_seven_l191_19110

/-- Given three consecutive integers where the sums of these integers taken in pairs are 18, 20, and 23, prove that the middle number is 7. -/
theorem middle_number_is_seven (x : ℤ) 
  (h1 : x + (x + 1) = 18) 
  (h2 : x + (x + 2) = 20) 
  (h3 : (x + 1) + (x + 2) = 23) : 
  x + 1 = 7 := by
  sorry

end NUMINAMATH_CALUDE_middle_number_is_seven_l191_19110


namespace NUMINAMATH_CALUDE_triangular_prism_width_l191_19102

theorem triangular_prism_width
  (l h : ℝ)
  (longest_edge : ℝ)
  (h_l : l = 5)
  (h_h : h = 13)
  (h_longest : longest_edge = 14)
  (h_longest_edge : longest_edge = Real.sqrt (l^2 + w^2 + h^2))
  : w = Real.sqrt 2 :=
by
  sorry

end NUMINAMATH_CALUDE_triangular_prism_width_l191_19102


namespace NUMINAMATH_CALUDE_mod_equivalence_unique_solution_l191_19151

theorem mod_equivalence_unique_solution :
  ∃! n : ℕ, 0 ≤ n ∧ n ≤ 7 ∧ n ≡ -2753 [ZMOD 8] ∧ n = 7 := by
  sorry

end NUMINAMATH_CALUDE_mod_equivalence_unique_solution_l191_19151


namespace NUMINAMATH_CALUDE_chooseAndAssignTheorem_l191_19120

-- Define the set of members
inductive Member : Type
| Alice : Member
| Bob : Member
| Carol : Member
| Dave : Member

-- Define the set of officer roles
inductive Role : Type
| President : Role
| Secretary : Role
| Treasurer : Role

-- Define a function to calculate the number of ways to choose and assign roles
def waysToChooseAndAssign : ℕ :=
  -- Number of ways to choose 3 out of 4 members
  (Nat.choose 4 3) *
  -- Number of ways to assign 3 roles to 3 chosen members
  (Nat.factorial 3)

-- Theorem statement
theorem chooseAndAssignTheorem : waysToChooseAndAssign = 24 := by
  sorry


end NUMINAMATH_CALUDE_chooseAndAssignTheorem_l191_19120


namespace NUMINAMATH_CALUDE_tangent_line_property_l191_19159

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := (2/3) * x^(3/2) - Real.log x - 2/3

-- Define the derivative of f
noncomputable def f' (x : ℝ) : ℝ := (1/x) * (x^(3/2) - 1)

-- Define the function g
noncomputable def g (x : ℝ) : ℝ := |f x + f' x|

-- State the theorem
theorem tangent_line_property (x₁ x₂ : ℝ) (h₁ : x₁ > 0) (h₂ : x₂ > 0) (h₃ : x₁ ≠ x₂) (h₄ : g x₁ = g x₂) : x₁ * x₂ < 1 := by
  sorry

end NUMINAMATH_CALUDE_tangent_line_property_l191_19159


namespace NUMINAMATH_CALUDE_max_k_logarithm_inequality_l191_19192

theorem max_k_logarithm_inequality (x₀ x₁ x₂ x₃ : ℝ) 
  (h₀ : x₀ > x₁) (h₁ : x₁ > x₂) (h₂ : x₂ > x₃) (h₃ : x₃ > 0) :
  let log_base (b a : ℝ) := Real.log a / Real.log b
  9 * log_base (x₀ / x₃) 1993 ≤ 
    log_base (x₀ / x₁) 1993 + log_base (x₁ / x₂) 1993 + log_base (x₂ / x₃) 1993 ∧
  ∀ k > 9, ∃ x₀' x₁' x₂' x₃' : ℝ, x₀' > x₁' ∧ x₁' > x₂' ∧ x₂' > x₃' ∧ x₃' > 0 ∧
    k * log_base (x₀' / x₃') 1993 > 
      log_base (x₀' / x₁') 1993 + log_base (x₁' / x₂') 1993 + log_base (x₂' / x₃') 1993 :=
by sorry

end NUMINAMATH_CALUDE_max_k_logarithm_inequality_l191_19192


namespace NUMINAMATH_CALUDE_select_two_from_four_l191_19114

theorem select_two_from_four (n : ℕ) (k : ℕ) : n = 4 → k = 2 → Nat.choose n k = 6 := by
  sorry

end NUMINAMATH_CALUDE_select_two_from_four_l191_19114


namespace NUMINAMATH_CALUDE_exists_node_not_on_line_l191_19187

/-- Represents a node on the grid --/
structure Node :=
  (x : Nat) (y : Nat)

/-- Represents a polygonal line on the grid --/
structure Line :=
  (nodes : List Node)

/-- The grid size --/
def gridSize : Nat := 100

/-- Checks if a node is on the boundary of the grid --/
def isOnBoundary (n : Node) : Bool :=
  n.x = 0 || n.x = gridSize || n.y = 0 || n.y = gridSize

/-- Checks if a node is a corner of the grid --/
def isCorner (n : Node) : Bool :=
  (n.x = 0 && n.y = 0) || (n.x = 0 && n.y = gridSize) ||
  (n.x = gridSize && n.y = 0) || (n.x = gridSize && n.y = gridSize)

/-- Theorem: There exists a non-corner node not on any line --/
theorem exists_node_not_on_line (lines : List Line) : 
  ∃ (n : Node), !isCorner n ∧ ∀ (l : Line), l ∈ lines → n ∉ l.nodes :=
sorry


end NUMINAMATH_CALUDE_exists_node_not_on_line_l191_19187


namespace NUMINAMATH_CALUDE_cars_served_4pm_to_6pm_l191_19198

def peak_service_rate : ℕ := 12
def off_peak_service_rate : ℕ := 8
def blocks_per_hour : ℕ := 4

def cars_served_peak_hour : ℕ := peak_service_rate * blocks_per_hour
def cars_served_off_peak_hour : ℕ := off_peak_service_rate * blocks_per_hour

def total_cars_served : ℕ := cars_served_peak_hour + cars_served_off_peak_hour

theorem cars_served_4pm_to_6pm : total_cars_served = 80 := by
  sorry

end NUMINAMATH_CALUDE_cars_served_4pm_to_6pm_l191_19198


namespace NUMINAMATH_CALUDE_second_train_length_second_train_length_solution_l191_19100

/-- Calculates the length of the second train given the speeds of two trains, 
    the time they take to clear each other, and the length of the first train. -/
theorem second_train_length 
  (speed1 : ℝ) 
  (speed2 : ℝ) 
  (clear_time : ℝ) 
  (length1 : ℝ) : ℝ :=
  let relative_speed := speed1 + speed2
  let relative_speed_ms := relative_speed * 1000 / 3600
  let total_distance := relative_speed_ms * clear_time
  total_distance - length1

/-- The length of the second train is approximately 165.12 meters. -/
theorem second_train_length_solution :
  ∃ ε > 0, abs (second_train_length 80 65 7.0752960452818945 120 - 165.12) < ε :=
by sorry

end NUMINAMATH_CALUDE_second_train_length_second_train_length_solution_l191_19100


namespace NUMINAMATH_CALUDE_min_value_f_l191_19117

/-- Given that 2x^2 + 3xy + 2y^2 = 1, the minimum value of f(x, y) = x + y + xy is -9/8 -/
theorem min_value_f (x y : ℝ) (h : 2*x^2 + 3*x*y + 2*y^2 = 1) :
  ∃ (m : ℝ), m = -9/8 ∧ ∀ (a b : ℝ), 2*a^2 + 3*a*b + 2*b^2 = 1 → a + b + a*b ≥ m :=
by sorry

end NUMINAMATH_CALUDE_min_value_f_l191_19117


namespace NUMINAMATH_CALUDE_certain_number_proof_l191_19148

theorem certain_number_proof : ∃ n : ℝ, n = 36 ∧ n + 3 * 4.0 = 48 := by
  sorry

end NUMINAMATH_CALUDE_certain_number_proof_l191_19148


namespace NUMINAMATH_CALUDE_min_shift_for_symmetry_l191_19197

noncomputable def f (x : ℝ) : ℝ := Real.sin (2 * x)

theorem min_shift_for_symmetry :
  let g (φ : ℝ) (x : ℝ) := f (x - φ)
  ∃ (φ : ℝ), φ > 0 ∧
    (∀ x, g φ (π/6 + x) = g φ (π/6 - x)) ∧
    (∀ ψ, ψ > 0 ∧ (∀ x, g ψ (π/6 + x) = g ψ (π/6 - x)) → ψ ≥ φ) ∧
    φ = 5*π/12 :=
sorry

end NUMINAMATH_CALUDE_min_shift_for_symmetry_l191_19197


namespace NUMINAMATH_CALUDE_relationship_between_a_and_b_l191_19189

theorem relationship_between_a_and_b (a b : ℝ) 
  (ha : a > 0) (hb : b < 0) (hab : a + b < 0) : 
  b < -a ∧ -a < a ∧ a < -b := by
  sorry

end NUMINAMATH_CALUDE_relationship_between_a_and_b_l191_19189


namespace NUMINAMATH_CALUDE_range_of_a_l191_19182

theorem range_of_a (a : ℝ) : 
  (∀ x ∈ Set.Icc 1 2, x^2 - a ≥ 0) ∧ 
  (∃ x₀ : ℝ, x₀^2 + 2*a*x₀ + 2 - a = 0) → 
  a = 1 ∨ a ≤ -2 :=
by sorry

end NUMINAMATH_CALUDE_range_of_a_l191_19182


namespace NUMINAMATH_CALUDE_point_in_plane_region_l191_19128

def in_plane_region (x y : ℝ) : Prop := 2*x + y - 6 < 0

theorem point_in_plane_region :
  in_plane_region 0 1 ∧
  ¬(in_plane_region 5 0) ∧
  ¬(in_plane_region 0 7) ∧
  ¬(in_plane_region 2 3) :=
by sorry

end NUMINAMATH_CALUDE_point_in_plane_region_l191_19128


namespace NUMINAMATH_CALUDE_revenue_decrease_percentage_l191_19176

def old_revenue : ℝ := 72.0
def new_revenue : ℝ := 48.0

theorem revenue_decrease_percentage :
  (old_revenue - new_revenue) / old_revenue * 100 = 33.33 := by
  sorry

end NUMINAMATH_CALUDE_revenue_decrease_percentage_l191_19176


namespace NUMINAMATH_CALUDE_total_goals_theorem_l191_19199

/-- Represents the number of goals scored by Louie in his last match -/
def louie_last_match_goals : ℕ := 4

/-- Represents the number of goals scored by Louie in previous matches -/
def louie_previous_goals : ℕ := 40

/-- Represents the number of seasons Donnie has played -/
def donnie_seasons : ℕ := 3

/-- Represents the number of games in each season -/
def games_per_season : ℕ := 50

/-- Represents the initial number of goals scored by Annie in her first game -/
def annie_initial_goals : ℕ := 2

/-- Represents the increase in Annie's goals per game -/
def annie_goal_increase : ℕ := 2

/-- Represents the number of seasons Annie has played -/
def annie_seasons : ℕ := 2

/-- Theorem stating that the total number of goals scored by all siblings is 11,344 -/
theorem total_goals_theorem :
  let louie_total := louie_last_match_goals + louie_previous_goals
  let donnie_total := 2 * louie_last_match_goals * donnie_seasons * games_per_season
  let annie_games := annie_seasons * games_per_season
  let annie_total := annie_games * (annie_initial_goals + annie_initial_goals + (annie_games - 1) * annie_goal_increase) / 2
  louie_total + donnie_total + annie_total = 11344 := by
  sorry

end NUMINAMATH_CALUDE_total_goals_theorem_l191_19199


namespace NUMINAMATH_CALUDE_fixed_point_on_line_l191_19115

theorem fixed_point_on_line (a : ℝ) : (a + 1) * (-4) - (2 * a + 5) * (-2) - 6 = 0 := by
  sorry

end NUMINAMATH_CALUDE_fixed_point_on_line_l191_19115


namespace NUMINAMATH_CALUDE_village_seniors_l191_19149

/-- Proves the number of seniors in a village given the population distribution -/
theorem village_seniors (total_population : ℕ) 
  (h1 : total_population * 60 / 100 = 23040)  -- 60% of population are adults
  (h2 : total_population * 30 / 100 = total_population * 3 / 10) -- 30% are children
  : total_population * 10 / 100 = 3840 := by
  sorry

end NUMINAMATH_CALUDE_village_seniors_l191_19149


namespace NUMINAMATH_CALUDE_solve_equation_l191_19112

theorem solve_equation (y : ℤ) : 7 + y = 3 ↔ y = -4 := by
  sorry

end NUMINAMATH_CALUDE_solve_equation_l191_19112


namespace NUMINAMATH_CALUDE_overall_discount_percentage_l191_19195

/-- Calculate the overall discount percentage for three items given their cost prices, markups, and sale prices. -/
theorem overall_discount_percentage
  (cost_A cost_B cost_C : ℝ)
  (markup_A markup_B markup_C : ℝ)
  (sale_A sale_B sale_C : ℝ)
  (h_cost_A : cost_A = 540)
  (h_cost_B : cost_B = 620)
  (h_cost_C : cost_C = 475)
  (h_markup_A : markup_A = 0.15)
  (h_markup_B : markup_B = 0.20)
  (h_markup_C : markup_C = 0.25)
  (h_sale_A : sale_A = 462)
  (h_sale_B : sale_B = 558)
  (h_sale_C : sale_C = 405) :
  let marked_A := cost_A * (1 + markup_A)
  let marked_B := cost_B * (1 + markup_B)
  let marked_C := cost_C * (1 + markup_C)
  let total_marked := marked_A + marked_B + marked_C
  let total_sale := sale_A + sale_B + sale_C
  let discount_percentage := (total_marked - total_sale) / total_marked * 100
  ∃ ε > 0, |discount_percentage - 27.26| < ε :=
by sorry


end NUMINAMATH_CALUDE_overall_discount_percentage_l191_19195


namespace NUMINAMATH_CALUDE_path_time_equality_implies_distance_ratio_l191_19103

/-- Given two points A and B, and a point P between them, 
    if the time to go directly from P to B equals the time to go from P to A 
    and then from A to B at 6 times the speed, 
    then the ratio of PA to PB is 5/7 -/
theorem path_time_equality_implies_distance_ratio 
  (A B P : ℝ × ℝ) -- A, B, and P are points in 2D space
  (h_between : ∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ P = (1 - t) • A + t • B) -- P is between A and B
  (speed : ℝ) -- walking speed
  (h_speed_pos : speed > 0) -- speed is positive
  : (dist P B / speed = dist P A / speed + (dist A B) / (6 * speed)) → 
    (dist P A / dist P B = 5 / 7) :=
by sorry

#check path_time_equality_implies_distance_ratio

end NUMINAMATH_CALUDE_path_time_equality_implies_distance_ratio_l191_19103


namespace NUMINAMATH_CALUDE_oak_trees_in_park_l191_19150

theorem oak_trees_in_park (x : ℕ) : x + 4 = 9 → x = 5 := by
  sorry

end NUMINAMATH_CALUDE_oak_trees_in_park_l191_19150


namespace NUMINAMATH_CALUDE_root_value_theorem_l191_19194

theorem root_value_theorem (m : ℝ) : 
  (2 * m^2 - 3 * m - 1 = 0) → (3 * m * (2 * m - 3) - 1 = 2) := by
  sorry

end NUMINAMATH_CALUDE_root_value_theorem_l191_19194


namespace NUMINAMATH_CALUDE_factorization_equality_l191_19136

theorem factorization_equality (a b : ℝ) : a^2 - 2*a*b = a*(a - 2*b) := by
  sorry

end NUMINAMATH_CALUDE_factorization_equality_l191_19136


namespace NUMINAMATH_CALUDE_no_solution_exists_l191_19186

theorem no_solution_exists (x y z : ℕ) (hx : x > 2) (hy : y > 1) (heq : x^y + 1 = z^2) : False :=
sorry

end NUMINAMATH_CALUDE_no_solution_exists_l191_19186


namespace NUMINAMATH_CALUDE_slope_angle_range_l191_19132

-- Define Circle C
def CircleC (x y : ℝ) : Prop := x^2 + (y - 1)^2 = 4

-- Define Line l
def LineL (k : ℝ) (x y : ℝ) : Prop := y = k * x + 2

-- Define the condition that O is inside circle with diameter AB
def OInsideAB (k : ℝ) : Prop := 4 * (k^2 + 1) > 4 * k^2 + 3

-- Main theorem
theorem slope_angle_range :
  ∀ k : ℝ,
  (∃ x y : ℝ, CircleC x y ∧ LineL k x y) →  -- Line l intersects Circle C
  OInsideAB k →                            -- O is inside circle with diameter AB
  Real.arctan (1/2) < Real.arctan k ∧ Real.arctan k < π - Real.arctan (1/2) :=
by sorry

end NUMINAMATH_CALUDE_slope_angle_range_l191_19132


namespace NUMINAMATH_CALUDE_taxicab_distance_properties_l191_19184

/-- Taxicab distance between two points -/
def taxicab_distance (p q : ℝ × ℝ) : ℝ :=
  |p.1 - q.1| + |p.2 - q.2|

/-- Check if a point is on a line segment -/
def on_segment (a b c : ℝ × ℝ) : Prop :=
  ∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ c = (a.1 + t * (b.1 - a.1), a.2 + t * (b.2 - a.2))

/-- The set of points equidistant from two given points -/
def equidistant_set (m n : ℝ × ℝ) : Set (ℝ × ℝ) :=
  {p | taxicab_distance p m = taxicab_distance p n}

theorem taxicab_distance_properties :
  (∀ a b c : ℝ × ℝ, on_segment a b c → taxicab_distance a c + taxicab_distance c b = taxicab_distance a b) ∧
  ¬(∀ a b c : ℝ × ℝ, taxicab_distance a c + taxicab_distance c b > taxicab_distance a b) ∧
  equidistant_set (-1, 0) (1, 0) = {p : ℝ × ℝ | p.1 = 0} ∧
  (∀ p : ℝ × ℝ, p.1 + p.2 = 2 * Real.sqrt 5 → taxicab_distance (0, 0) p ≥ 2 * Real.sqrt 5) ∧
  (∃ p : ℝ × ℝ, p.1 + p.2 = 2 * Real.sqrt 5 ∧ taxicab_distance (0, 0) p = 2 * Real.sqrt 5) := by
  sorry

end NUMINAMATH_CALUDE_taxicab_distance_properties_l191_19184


namespace NUMINAMATH_CALUDE_monomial_properties_l191_19173

/-- Represents a monomial with coefficient and variables -/
structure Monomial where
  coeff : ℚ
  a_exp : ℕ
  b_exp : ℕ

/-- The given monomial 3a²b/2 -/
def given_monomial : Monomial := { coeff := 3/2, a_exp := 2, b_exp := 1 }

/-- The coefficient of a monomial -/
def coefficient (m : Monomial) : ℚ := m.coeff

/-- The degree of a monomial -/
def degree (m : Monomial) : ℕ := m.a_exp + m.b_exp

theorem monomial_properties :
  coefficient given_monomial = 3/2 ∧ degree given_monomial = 3 := by
  sorry

end NUMINAMATH_CALUDE_monomial_properties_l191_19173


namespace NUMINAMATH_CALUDE_sixth_term_is_negative_four_l191_19166

/-- An arithmetic sequence with specific properties -/
structure ArithmeticSequence where
  -- First term
  a : ℝ
  -- Common difference
  d : ℝ
  -- Sum of first 3 terms is 12
  sum_first_three : a + (a + d) + (a + 2*d) = 12
  -- Fourth term is 0
  fourth_term_zero : a + 3*d = 0

/-- The sixth term of the arithmetic sequence is -4 -/
theorem sixth_term_is_negative_four (seq : ArithmeticSequence) : 
  seq.a + 5*seq.d = -4 := by
  sorry

#check sixth_term_is_negative_four

end NUMINAMATH_CALUDE_sixth_term_is_negative_four_l191_19166


namespace NUMINAMATH_CALUDE_cubic_derivative_root_existence_l191_19180

/-- A cubic polynomial with real coefficients -/
structure CubicPolynomial where
  p : ℝ
  q : ℝ
  r : ℝ

/-- The roots of a cubic polynomial -/
structure CubicRoots where
  a : ℝ
  b : ℝ
  c : ℝ
  h_order : a ≤ b ∧ b ≤ c

/-- Theorem: The derivative of a cubic polynomial has a root in the specified interval -/
theorem cubic_derivative_root_existence (f : CubicPolynomial) (roots : CubicRoots) :
  ∃ x : ℝ, x ∈ Set.Icc ((roots.b + roots.c) / 2) ((roots.b + 2 * roots.c) / 3) ∧
    (3 * x^2 + 2 * f.p * x + f.q) = 0 :=
sorry

end NUMINAMATH_CALUDE_cubic_derivative_root_existence_l191_19180


namespace NUMINAMATH_CALUDE_library_book_count_l191_19122

theorem library_book_count (initial_books : ℕ) (loaned_books : ℕ) (return_rate : ℚ) : 
  initial_books = 75 →
  loaned_books = 40 →
  return_rate = 4/5 →
  initial_books - loaned_books + (return_rate * loaned_books).floor = 67 := by
sorry

end NUMINAMATH_CALUDE_library_book_count_l191_19122


namespace NUMINAMATH_CALUDE_intersection_range_l191_19170

-- Define the endpoints of the line segment
def A : ℝ × ℝ := (3, 2)
def B : ℝ × ℝ := (2, 3)

-- Define the line equation
def line_equation (k : ℝ) (x : ℝ) : ℝ := k * (x - 1)

-- Define the condition for intersection
def intersects (k : ℝ) : Prop :=
  ∃ x y, x ≥ min A.1 B.1 ∧ x ≤ max A.1 B.1 ∧
         y ≥ min A.2 B.2 ∧ y ≤ max A.2 B.2 ∧
         y = line_equation k x

-- Theorem statement
theorem intersection_range :
  {k : ℝ | intersects k} = {k : ℝ | 1 ≤ k ∧ k ≤ 3} :=
sorry

end NUMINAMATH_CALUDE_intersection_range_l191_19170


namespace NUMINAMATH_CALUDE_trapezoid_area_l191_19133

/-- The area of a trapezoid with bases 3h and 5h, and height h, is equal to 4h² -/
theorem trapezoid_area (h : ℝ) : h * ((3 * h + 5 * h) / 2) = 4 * h^2 := by
  sorry

end NUMINAMATH_CALUDE_trapezoid_area_l191_19133


namespace NUMINAMATH_CALUDE_approximation_accuracy_l191_19165

/-- Represents a decimal number with its integer and fractional parts -/
structure DecimalNumber where
  integerPart : Int
  fractionalPart : Nat
  decimalPlaces : Nat

/-- Defines the accuracy of a decimal number -/
def isAccurateTo (n : DecimalNumber) (place : Nat) : Prop :=
  n.decimalPlaces ≥ place

/-- The given decimal number 3.72 -/
def number : DecimalNumber :=
  { integerPart := 3
    fractionalPart := 72
    decimalPlaces := 2 }

/-- Tenths place represented as a natural number -/
def tenthsPlace : Nat := 1

theorem approximation_accuracy :
  isAccurateTo number tenthsPlace := by sorry

end NUMINAMATH_CALUDE_approximation_accuracy_l191_19165


namespace NUMINAMATH_CALUDE_childrens_tickets_sold_l191_19153

theorem childrens_tickets_sold
  (adult_price senior_price children_price : ℚ)
  (total_tickets : ℕ)
  (total_revenue : ℚ)
  (h1 : adult_price = 6)
  (h2 : children_price = 9/2)
  (h3 : senior_price = 5)
  (h4 : total_tickets = 600)
  (h5 : total_revenue = 3250)
  : ∃ (A C S : ℕ),
    A + C + S = total_tickets ∧
    adult_price * A + children_price * C + senior_price * S = total_revenue ∧
    C = (350 - S) / (3/2) :=
sorry

end NUMINAMATH_CALUDE_childrens_tickets_sold_l191_19153


namespace NUMINAMATH_CALUDE_complex_3_minus_i_in_fourth_quadrant_l191_19130

/-- A complex number z is in the fourth quadrant if its real part is positive and its imaginary part is negative -/
def in_fourth_quadrant (z : ℂ) : Prop := 0 < z.re ∧ z.im < 0

/-- The complex number 3 - i is in the fourth quadrant -/
theorem complex_3_minus_i_in_fourth_quadrant : 
  in_fourth_quadrant (3 - I) := by sorry

end NUMINAMATH_CALUDE_complex_3_minus_i_in_fourth_quadrant_l191_19130


namespace NUMINAMATH_CALUDE_portrait_in_silver_box_l191_19107

-- Define the possible box locations
inductive Box
| Gold
| Silver
| Lead

-- Define the propositions
def p (portrait_location : Box) : Prop := portrait_location = Box.Gold
def q (portrait_location : Box) : Prop := portrait_location ≠ Box.Silver
def r (portrait_location : Box) : Prop := portrait_location ≠ Box.Gold

-- Theorem statement
theorem portrait_in_silver_box :
  ∃! (portrait_location : Box),
    (p portrait_location ∨ q portrait_location ∨ r portrait_location) ∧
    (¬(p portrait_location ∧ q portrait_location) ∧
     ¬(p portrait_location ∧ r portrait_location) ∧
     ¬(q portrait_location ∧ r portrait_location)) →
  portrait_location = Box.Silver :=
by sorry

end NUMINAMATH_CALUDE_portrait_in_silver_box_l191_19107


namespace NUMINAMATH_CALUDE_fraction_equality_l191_19104

theorem fraction_equality (y : ℝ) (h : y > 0) : (9 * y) / 20 + (3 * y) / 10 = 0.75 * y := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_l191_19104


namespace NUMINAMATH_CALUDE_sum_of_divisors_of_11_squared_l191_19123

theorem sum_of_divisors_of_11_squared (a b c : ℕ+) : 
  a * b * c = 11^2 →
  a ∣ 11^2 ∧ b ∣ 11^2 ∧ c ∣ 11^2 →
  a ≠ b ∧ b ≠ c ∧ a ≠ c →
  (a : ℕ) + (b : ℕ) + (c : ℕ) = 23 := by
sorry

end NUMINAMATH_CALUDE_sum_of_divisors_of_11_squared_l191_19123


namespace NUMINAMATH_CALUDE_petya_and_anya_ages_l191_19167

/-- Given that Petya is three times older than Anya and Anya is 8 years younger than Petya,
    prove that Petya is 12 years old and Anya is 4 years old. -/
theorem petya_and_anya_ages :
  ∀ (petya_age anya_age : ℕ),
    petya_age = 3 * anya_age →
    petya_age - anya_age = 8 →
    petya_age = 12 ∧ anya_age = 4 := by
  sorry

end NUMINAMATH_CALUDE_petya_and_anya_ages_l191_19167


namespace NUMINAMATH_CALUDE_inequality_proof_l191_19161

theorem inequality_proof (x : ℝ) (n : ℕ) (h1 : x > 0) (h2 : n > 1) :
  (x^(n-1) - 1) / (n-1 : ℝ) ≤ (x^n - 1) / n :=
by sorry

end NUMINAMATH_CALUDE_inequality_proof_l191_19161


namespace NUMINAMATH_CALUDE_nine_div_repeating_third_eq_twentyseven_l191_19121

/-- The repeating decimal 0.3333... --/
def repeating_third : ℚ := 1 / 3

/-- Theorem stating that 9 divided by 0.3333... equals 27 --/
theorem nine_div_repeating_third_eq_twentyseven :
  9 / repeating_third = 27 := by sorry

end NUMINAMATH_CALUDE_nine_div_repeating_third_eq_twentyseven_l191_19121


namespace NUMINAMATH_CALUDE_prime_factorization_of_large_number_l191_19141

theorem prime_factorization_of_large_number :
  1007021035035021007001 = 7^7 * 11^7 * 13^7 := by
  sorry

end NUMINAMATH_CALUDE_prime_factorization_of_large_number_l191_19141


namespace NUMINAMATH_CALUDE_shortest_major_axis_ellipse_l191_19171

/-- The line l: y = x + 9 -/
def line_l (x : ℝ) : ℝ := x + 9

/-- The first focus of the ellipse -/
def F₁ : ℝ × ℝ := (-3, 0)

/-- The second focus of the ellipse -/
def F₂ : ℝ × ℝ := (3, 0)

/-- Definition of the ellipse equation -/
def is_ellipse_equation (a b : ℝ) (x y : ℝ) : Prop :=
  x^2 / a^2 + y^2 / b^2 = 1

/-- The theorem stating the equation of the ellipse with shortest major axis -/
theorem shortest_major_axis_ellipse :
  ∃ (P : ℝ × ℝ),
    (P.2 = line_l P.1) ∧
    is_ellipse_equation 45 36 P.1 P.2 ∧
    ∀ (Q : ℝ × ℝ),
      (Q.2 = line_l Q.1) →
      (Q.1 - F₁.1)^2 + (Q.2 - F₁.2)^2 + (Q.1 - F₂.1)^2 + (Q.2 - F₂.2)^2 ≥
      (P.1 - F₁.1)^2 + (P.2 - F₁.2)^2 + (P.1 - F₂.1)^2 + (P.2 - F₂.2)^2 :=
by sorry

end NUMINAMATH_CALUDE_shortest_major_axis_ellipse_l191_19171


namespace NUMINAMATH_CALUDE_laura_friends_count_l191_19162

def total_blocks : ℕ := 28
def blocks_per_friend : ℕ := 7

theorem laura_friends_count : total_blocks / blocks_per_friend = 4 := by
  sorry

end NUMINAMATH_CALUDE_laura_friends_count_l191_19162


namespace NUMINAMATH_CALUDE_right_angled_triangle_345_l191_19135

theorem right_angled_triangle_345 (a b c : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : c > 0)
  (h4 : a / b = 3 / 4) (h5 : b / c = 4 / 5) : a^2 + b^2 = c^2 := by
sorry


end NUMINAMATH_CALUDE_right_angled_triangle_345_l191_19135


namespace NUMINAMATH_CALUDE_nancy_hula_hoop_time_l191_19113

theorem nancy_hula_hoop_time (morgan_time casey_time nancy_time : ℕ) : 
  morgan_time = 21 →
  morgan_time = 3 * casey_time →
  nancy_time = casey_time + 3 →
  nancy_time = 10 := by
sorry

end NUMINAMATH_CALUDE_nancy_hula_hoop_time_l191_19113


namespace NUMINAMATH_CALUDE_product_of_roots_equals_32_l191_19160

theorem product_of_roots_equals_32 : 
  (256 : ℝ) ^ (1/4) * (8 : ℝ) ^ (1/3) * (16 : ℝ) ^ (1/2) = 32 := by sorry

end NUMINAMATH_CALUDE_product_of_roots_equals_32_l191_19160


namespace NUMINAMATH_CALUDE_circle_and_ngons_inequalities_l191_19183

/-- Given a circle and two regular n-gons (one inscribed, one circumscribed),
    prove the relationships between their areas and perimeters. -/
theorem circle_and_ngons_inequalities 
  (n : ℕ) 
  (S : ℝ) 
  (L : ℝ) 
  (S₁ : ℝ) 
  (S₂ : ℝ) 
  (P₁ : ℝ) 
  (P₂ : ℝ) 
  (h_n : n ≥ 3) 
  (h_S : S > 0) 
  (h_L : L > 0) 
  (h_S₁ : S₁ > 0) 
  (h_S₂ : S₂ > 0) 
  (h_P₁ : P₁ > 0) 
  (h_P₂ : P₂ > 0) 
  (h_inscribed : S₁ < S) 
  (h_circumscribed : S₂ > S) : 
  (S^2 > S₁ * S₂) ∧ (L^2 < P₁ * P₂) := by
  sorry


end NUMINAMATH_CALUDE_circle_and_ngons_inequalities_l191_19183


namespace NUMINAMATH_CALUDE_lily_pad_coverage_l191_19142

/-- Represents the size of the lily pad patch as a fraction of the lake -/
def LilyPadSize := ℚ

/-- The number of days it takes for the patch to cover the entire lake -/
def TotalDays : ℕ := 37

/-- The fraction of the lake that is covered after a given number of days -/
def coverage (days : ℕ) : LilyPadSize :=
  (1 : ℚ) / (2 ^ (TotalDays - days))

/-- Theorem stating that it takes 36 days to cover three-fourths of the lake -/
theorem lily_pad_coverage :
  coverage 36 = (3 : ℚ) / 4 := by sorry

end NUMINAMATH_CALUDE_lily_pad_coverage_l191_19142
