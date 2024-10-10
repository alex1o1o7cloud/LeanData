import Mathlib

namespace house_painting_cost_l242_24201

/-- The total cost of painting a house -/
def total_cost (area : ℝ) (price_per_sqft : ℝ) : ℝ :=
  area * price_per_sqft

/-- Theorem: The total cost of painting a house with an area of 484 sq ft
    at a price of Rs. 20 per sq ft is Rs. 9680 -/
theorem house_painting_cost :
  total_cost 484 20 = 9680 := by
  sorry

end house_painting_cost_l242_24201


namespace opposite_of_neg_three_l242_24291

/-- The opposite number of a real number -/
def opposite (x : ℝ) : ℝ := -x

/-- Theorem: The opposite number of -3 is 3 -/
theorem opposite_of_neg_three : opposite (-3) = 3 := by
  sorry

end opposite_of_neg_three_l242_24291


namespace selling_price_calculation_l242_24220

/-- Given a sale where the gain is $20 and the gain percentage is 25%, 
    prove that the selling price is $100. -/
theorem selling_price_calculation (gain : ℝ) (gain_percentage : ℝ) :
  gain = 20 →
  gain_percentage = 25 →
  ∃ (cost_price selling_price : ℝ),
    gain = gain_percentage / 100 * cost_price ∧
    selling_price = cost_price + gain ∧
    selling_price = 100 := by
  sorry

end selling_price_calculation_l242_24220


namespace range_of_sum_l242_24287

def f (x : ℝ) := |2 - x^2|

theorem range_of_sum (a b : ℝ) (h1 : 0 < a) (h2 : a < b) (h3 : f a = f b) :
  ∃ (y : ℝ), 2 < y ∧ y < 2 * Real.sqrt 2 ∧ y = a + b :=
sorry

end range_of_sum_l242_24287


namespace gina_money_problem_l242_24203

theorem gina_money_problem (initial_amount : ℚ) : 
  initial_amount = 400 → 
  initial_amount - (initial_amount * (1/4 + 1/8 + 1/5)) = 170 := by
  sorry

end gina_money_problem_l242_24203


namespace possible_values_of_expression_l242_24214

theorem possible_values_of_expression (a b : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) :
  (a / abs a + b / abs b + (a * b) / abs (a * b)) = 3 ∨
  (a / abs a + b / abs b + (a * b) / abs (a * b)) = -1 :=
sorry

end possible_values_of_expression_l242_24214


namespace cubic_expression_evaluation_l242_24241

theorem cubic_expression_evaluation : 
  3000^3 - 2999 * 3000^2 - 2999^2 * 3000 + 2999^3 = 26991001 := by
sorry

end cubic_expression_evaluation_l242_24241


namespace cab_driver_income_l242_24224

/-- Cab driver's income problem -/
theorem cab_driver_income 
  (income : Fin 5 → ℕ) 
  (h1 : income 0 = 600)
  (h2 : income 1 = 250)
  (h3 : income 2 = 450)
  (h4 : income 3 = 400)
  (h_avg : (income 0 + income 1 + income 2 + income 3 + income 4) / 5 = 500) :
  income 4 = 800 := by
sorry


end cab_driver_income_l242_24224


namespace joeys_reading_assignment_l242_24215

/-- The number of pages Joey must read after his break -/
def pages_after_break : ℕ := 9

/-- The percentage of pages Joey reads before taking a break -/
def percentage_before_break : ℚ := 70 / 100

theorem joeys_reading_assignment :
  ∃ (total_pages : ℕ),
    (1 - percentage_before_break) * total_pages = pages_after_break ∧
    total_pages = 30 := by
  sorry

end joeys_reading_assignment_l242_24215


namespace f_max_min_l242_24270

-- Define the function f(x) = 2x² - x⁴
def f (x : ℝ) : ℝ := 2 * x^2 - x^4

-- Theorem statement
theorem f_max_min :
  (∃ x : ℝ, f x = 1 ∧ ∀ y : ℝ, f y ≤ 1) ∧
  (∃ x : ℝ, f x = 0 ∧ ∀ y : ℝ, f y ≥ 0) :=
sorry

end f_max_min_l242_24270


namespace triangle_equivalence_l242_24216

/-- A triangle with vertices A, B, and C -/
structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ

/-- The angles of a triangle -/
def Triangle.angles (t : Triangle) : ℝ × ℝ × ℝ := sorry

/-- The Nine-point circle of a triangle -/
def Triangle.ninePointCircle (t : Triangle) : Circle := sorry

/-- The Incircle of a triangle -/
def Triangle.incircle (t : Triangle) : Circle := sorry

/-- The Euler Line of a triangle -/
def Triangle.eulerLine (t : Triangle) : Line := sorry

/-- Check if a triangle is equilateral -/
def Triangle.isEquilateral (t : Triangle) : Prop := sorry

/-- Check if one of the angles is 60° -/
def Triangle.hasAngle60 (t : Triangle) : Prop := sorry

/-- Check if the angles are in arithmetic progression -/
def Triangle.anglesInArithmeticProgression (t : Triangle) : Prop := sorry

/-- Check if the common tangent to the Nine-point circle and Incircle is parallel to the Euler Line -/
def Triangle.commonTangentParallelToEulerLine (t : Triangle) : Prop := sorry

/-- Main theorem -/
theorem triangle_equivalence (t : Triangle) (h : ¬ t.isEquilateral) :
  t.hasAngle60 ↔ t.anglesInArithmeticProgression ∧ t.commonTangentParallelToEulerLine :=
sorry

end triangle_equivalence_l242_24216


namespace square_sum_value_l242_24221

theorem square_sum_value (x y : ℝ) (h1 : x * y = 16) (h2 : x^2 + y^2 = 34) : 
  (x + y)^2 = 66 := by sorry

end square_sum_value_l242_24221


namespace pirate_loot_sum_is_correct_l242_24282

/-- Converts a base 5 number to base 10 -/
def base5ToBase10 (digits : List Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * (5^i)) 0

/-- The sum of the pirate's loot in base 10 -/
def pirateLootSum : Nat :=
  base5ToBase10 [2, 3, 1, 4] + 
  base5ToBase10 [2, 3, 4, 1] + 
  base5ToBase10 [4, 2, 0, 2] + 
  base5ToBase10 [4, 2, 2]

theorem pirate_loot_sum_is_correct : pirateLootSum = 1112 := by
  sorry

end pirate_loot_sum_is_correct_l242_24282


namespace polynomial_value_at_3_l242_24275

-- Define a monic polynomial of degree 4
def is_monic_degree_4 (p : ℝ → ℝ) : Prop :=
  ∃ a b c d : ℝ, ∀ x, p x = x^4 + a*x^3 + b*x^2 + c*x + d

-- State the theorem
theorem polynomial_value_at_3 
  (p : ℝ → ℝ) 
  (h_monic : is_monic_degree_4 p) 
  (h1 : p 1 = 1) 
  (h2 : p (-1) = -1) 
  (h3 : p 2 = 2) 
  (h4 : p (-2) = -2) : 
  p 3 = 43 := by
  sorry

end polynomial_value_at_3_l242_24275


namespace hyperbola_properties_l242_24250

-- Define the hyperbola equation
def hyperbola_eq (x y : ℝ) : Prop := 16 * x^2 - 9 * y^2 = 144

-- Define the parabola C
def parabola_C (x y : ℝ) : Prop := y^2 = -12 * x

-- Theorem statement
theorem hyperbola_properties :
  -- Length of real axis
  (∃ a : ℝ, a = 3 ∧ 2 * a = 6) ∧
  -- Length of imaginary axis
  (∃ b : ℝ, b = 4 ∧ 2 * b = 8) ∧
  -- Eccentricity
  (∃ e : ℝ, e = 5/3) ∧
  -- Parabola C equation
  (∀ x y : ℝ, hyperbola_eq x y → 
    (x = -3 → parabola_C x y) ∧ 
    (x = 0 ∧ y = 0 → parabola_C x y)) :=
by sorry

end hyperbola_properties_l242_24250


namespace turtle_speed_specific_turtle_speed_l242_24255

/-- Given a race with a hare and a turtle, calculate the turtle's speed -/
theorem turtle_speed (race_distance : ℝ) (hare_speed : ℝ) (head_start : ℝ) : ℝ :=
  let turtle_speed := race_distance / (race_distance / hare_speed + head_start)
  turtle_speed

/-- The turtle's speed in the specific race scenario -/
theorem specific_turtle_speed : 
  turtle_speed 20 10 18 = 1 := by sorry

end turtle_speed_specific_turtle_speed_l242_24255


namespace sqrt_four_equals_two_l242_24258

theorem sqrt_four_equals_two : Real.sqrt 4 = 2 := by sorry

end sqrt_four_equals_two_l242_24258


namespace tax_reduction_percentage_l242_24284

/-- Given a commodity with tax and consumption, proves that if the tax is reduced by a certain percentage,
    consumption increases by 15%, and revenue decreases by 8%, then the tax reduction percentage is 20%. -/
theorem tax_reduction_percentage
  (T : ℝ) -- Original tax
  (C : ℝ) -- Original consumption
  (X : ℝ) -- Percentage by which tax is diminished
  (h1 : T > 0)
  (h2 : C > 0)
  (h3 : X ≥ 0)
  (h4 : X ≤ 100)
  (h5 : T * (1 - X / 100) * C * 1.15 = 0.92 * T * C) -- Revenue equation
  : X = 20 := by sorry

end tax_reduction_percentage_l242_24284


namespace power_sum_equals_power_implies_exponent_one_l242_24276

theorem power_sum_equals_power_implies_exponent_one (p a n : ℕ) : 
  Prime p → a > 0 → n > 0 → (2^p + 3^p = a^n) → n = 1 := by sorry

end power_sum_equals_power_implies_exponent_one_l242_24276


namespace equation_one_solution_l242_24296

theorem equation_one_solution (m : ℝ) : 
  (∃! x : ℝ, (3*x+4)*(x-8) = -50 + m*x) ↔ 
  (m = -20 + 6*Real.sqrt 6 ∨ m = -20 - 6*Real.sqrt 6) := by
  sorry

end equation_one_solution_l242_24296


namespace ship_speed_in_still_water_l242_24232

theorem ship_speed_in_still_water :
  let downstream_distance : ℝ := 81
  let upstream_distance : ℝ := 69
  let water_flow_speed : ℝ := 2
  let ship_speed : ℝ := 25
  (downstream_distance / (ship_speed + water_flow_speed) =
   upstream_distance / (ship_speed - water_flow_speed)) →
  ship_speed = 25 := by
  sorry

end ship_speed_in_still_water_l242_24232


namespace parabola_equation_l242_24231

/-- A parabola with vertex at the origin, axis of symmetry along the x-axis,
    and focus on the line 3x - 4y - 12 = 0 has the equation y² = 16x -/
theorem parabola_equation (x y : ℝ) :
  (∀ a b : ℝ, 3 * a - 4 * b - 12 = 0 → (a = 4 ∧ b = 0)) →
  y^2 = 16 * x := by
  sorry

end parabola_equation_l242_24231


namespace correct_distribution_l242_24262

/-- Represents the jellybean distribution problem --/
structure JellybeanDistribution where
  total_jellybeans : ℕ
  num_nephews : ℕ
  num_nieces : ℕ
  nephew_ratio : ℕ
  niece_ratio : ℕ

/-- Calculates the maximum number of jellybeans each nephew and niece can receive --/
def max_distribution (jd : JellybeanDistribution) : ℕ × ℕ :=
  let total_parts := jd.num_nephews * jd.nephew_ratio + jd.num_nieces * jd.niece_ratio
  let jellybeans_per_part := jd.total_jellybeans / total_parts
  (jellybeans_per_part * jd.nephew_ratio, jellybeans_per_part * jd.niece_ratio)

/-- Theorem stating the correct distribution for the given problem --/
theorem correct_distribution (jd : JellybeanDistribution) 
  (h1 : jd.total_jellybeans = 537)
  (h2 : jd.num_nephews = 4)
  (h3 : jd.num_nieces = 3)
  (h4 : jd.nephew_ratio = 2)
  (h5 : jd.niece_ratio = 1) :
  max_distribution jd = (96, 48) ∧ 
  96 * jd.num_nephews + 48 * jd.num_nieces ≤ jd.total_jellybeans :=
by
  sorry

#eval max_distribution {
  total_jellybeans := 537,
  num_nephews := 4,
  num_nieces := 3,
  nephew_ratio := 2,
  niece_ratio := 1
}

end correct_distribution_l242_24262


namespace female_fraction_l242_24212

theorem female_fraction (total_students : ℕ) (non_foreign_males : ℕ) :
  total_students = 300 →
  non_foreign_males = 90 →
  (2 : ℚ) / 3 = (total_students - (non_foreign_males / (9 : ℚ) / 10)) / total_students :=
by sorry

end female_fraction_l242_24212


namespace sum_of_four_digit_odd_and_multiples_of_five_l242_24225

/-- The number of four-digit odd numbers -/
def A : ℕ := 4500

/-- The number of four-digit multiples of 5 -/
def B : ℕ := 1800

/-- The sum of four-digit odd numbers and four-digit multiples of 5 is 6300 -/
theorem sum_of_four_digit_odd_and_multiples_of_five : A + B = 6300 := by
  sorry

end sum_of_four_digit_odd_and_multiples_of_five_l242_24225


namespace system_solution_unique_l242_24205

/-- Proves that x = 2 and y = 1 is the unique solution to the given system of equations -/
theorem system_solution_unique :
  ∃! (x y : ℝ), (2 * x - 5 * y = -1) ∧ (-4 * x + y = -7) := by
  sorry

end system_solution_unique_l242_24205


namespace regular_polygon_with_120_degree_angles_has_6_sides_l242_24236

theorem regular_polygon_with_120_degree_angles_has_6_sides :
  ∀ n : ℕ, n > 2 →
  (∀ θ : ℝ, θ = 120 → θ * n = 180 * (n - 2)) →
  n = 6 := by
sorry

end regular_polygon_with_120_degree_angles_has_6_sides_l242_24236


namespace number_operation_result_l242_24233

theorem number_operation_result : ∃ (x : ℝ), x = 295 ∧ (x / 5 + 6 = 65) := by sorry

end number_operation_result_l242_24233


namespace solution_l242_24207

/-- The set of points satisfying the given equation -/
def S : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | 2 * p.1^2 + p.2^2 + 3 * p.1 * p.2 + 3 * p.1 + p.2 = 2}

/-- The first line -/
def L₁ : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.2 = -p.1 - 2}

/-- The second line -/
def L₂ : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.2 = -2 * p.1 + 1}

/-- The union of the two lines -/
def U : Set (ℝ × ℝ) :=
  L₁ ∪ L₂

theorem solution : S = U := by
  sorry

end solution_l242_24207


namespace proposition_b_l242_24210

theorem proposition_b (a b c : ℝ) : a < b → a * c^2 ≤ b * c^2 := by
  sorry

end proposition_b_l242_24210


namespace hannah_age_problem_l242_24265

/-- Hannah's age problem -/
theorem hannah_age_problem :
  let num_brothers : ℕ := 3
  let brother_age : ℕ := 8
  let hannah_age_factor : ℕ := 2
  let hannah_age := hannah_age_factor * (num_brothers * brother_age)
  hannah_age = 48 := by sorry

end hannah_age_problem_l242_24265


namespace monomial_sum_condition_l242_24202

/-- 
If the sum of the monomials x^2 * y^(m+2) and x^n * y is still a monomial, 
then m + n = 1.
-/
theorem monomial_sum_condition (m n : ℤ) : 
  (∃ (x y : ℚ), x ≠ 0 ∧ y ≠ 0 ∧ ∃ (k : ℚ), x^2 * y^(m+2) + x^n * y = k * (x^2 * y^(m+2))) → 
  m + n = 1 := by
sorry

end monomial_sum_condition_l242_24202


namespace train_passing_pole_l242_24240

/-- Proves that a train of given length and speed takes a specific time to pass a pole -/
theorem train_passing_pole (train_length : ℝ) (train_speed_kmh : ℝ) (time : ℝ) : 
  train_length = 500 → 
  train_speed_kmh = 90 → 
  time = train_length / (train_speed_kmh * (1000 / 3600)) → 
  time = 20 := by
  sorry

end train_passing_pole_l242_24240


namespace y1_gt_y2_l242_24290

/-- A linear function that does not pass through the third quadrant -/
structure LinearFunctionNotInThirdQuadrant where
  k : ℝ
  b : ℝ
  not_in_third_quadrant : k < 0

/-- The function corresponding to the LinearFunctionNotInThirdQuadrant -/
def f (l : LinearFunctionNotInThirdQuadrant) (x : ℝ) : ℝ :=
  l.k * x + l.b

/-- Theorem stating that y₁ > y₂ for the given conditions -/
theorem y1_gt_y2 (l : LinearFunctionNotInThirdQuadrant) (y₁ y₂ : ℝ)
    (h1 : f l (-1) = y₁)
    (h2 : f l 1 = y₂) :
    y₁ > y₂ := by
  sorry

end y1_gt_y2_l242_24290


namespace suresh_work_hours_l242_24244

theorem suresh_work_hours (suresh_rate ashutosh_rate : ℚ) 
  (ashutosh_remaining_time : ℚ) : 
  suresh_rate = 1 / 15 →
  ashutosh_rate = 1 / 25 →
  ashutosh_remaining_time = 10 →
  ∃ (suresh_time : ℚ), 
    suresh_time * suresh_rate + ashutosh_remaining_time * ashutosh_rate = 1 ∧
    suresh_time = 9 := by
sorry

end suresh_work_hours_l242_24244


namespace star_interior_angle_sum_l242_24211

/-- An n-pointed star constructed from an n-sided convex polygon -/
structure StarPolygon where
  n : ℕ
  n_ge_6 : n ≥ 6

/-- The sum of interior angles at the vertices of the star -/
def interior_angle_sum (star : StarPolygon) : ℝ :=
  180 * (star.n - 2)

/-- Theorem: The sum of interior angles at the vertices of an n-pointed star
    constructed by extending every third side of an n-sided convex polygon (n ≥ 6)
    is equal to 180° * (n - 2) -/
theorem star_interior_angle_sum (star : StarPolygon) :
  interior_angle_sum star = 180 * (star.n - 2) := by
  sorry

end star_interior_angle_sum_l242_24211


namespace prop_truth_values_l242_24234

theorem prop_truth_values (p q : Prop) :
  ¬(p ∨ (¬q)) → (¬p ∧ q) := by
  sorry

end prop_truth_values_l242_24234


namespace ferry_travel_time_l242_24278

/-- Represents the travel time of Ferry P in hours -/
def t : ℝ := 2

/-- The speed of Ferry P in kilometers per hour -/
def speed_p : ℝ := 8

/-- The speed of Ferry Q in kilometers per hour -/
def speed_q : ℝ := speed_p + 4

/-- The distance traveled by Ferry P in kilometers -/
def distance_p : ℝ := speed_p * t

/-- The distance traveled by Ferry Q in kilometers -/
def distance_q : ℝ := 3 * distance_p

/-- The travel time of Ferry Q in hours -/
def time_q : ℝ := t + 2

theorem ferry_travel_time :
  speed_q * time_q = distance_q ∧
  t = 2 := by sorry

end ferry_travel_time_l242_24278


namespace fraction_equality_l242_24251

theorem fraction_equality (w z : ℝ) (h : (1/w + 1/z)/(1/w - 1/z) = 1008) :
  (w + z)/(w - z) = 1008 := by
  sorry

end fraction_equality_l242_24251


namespace exists_n_with_constant_term_l242_24253

/-- A function that checks if the expansion of (x - 1/x³)ⁿ contains a constant term -/
def has_constant_term (n : ℕ) : Prop :=
  ∃ r : ℕ, n = 4 * r

/-- Theorem stating that there exists an n between 3 and 16 (inclusive) 
    such that the expansion of (x - 1/x³)ⁿ contains a constant term -/
theorem exists_n_with_constant_term : 
  ∃ n : ℕ, 3 ≤ n ∧ n ≤ 16 ∧ has_constant_term n :=
sorry

end exists_n_with_constant_term_l242_24253


namespace binomial_coefficient_third_term_2x_minus_y_power_8_l242_24227

/-- The binomial coefficient of the 3rd term in the expansion of (2x-y)^8 is 28 -/
theorem binomial_coefficient_third_term_2x_minus_y_power_8 :
  Nat.choose 8 2 = 28 := by
  sorry

end binomial_coefficient_third_term_2x_minus_y_power_8_l242_24227


namespace lollipop_count_l242_24217

theorem lollipop_count (total_cost : ℝ) (single_cost : ℝ) (h1 : total_cost = 90) (h2 : single_cost = 0.75) :
  total_cost / single_cost = 120 := by
  sorry

end lollipop_count_l242_24217


namespace alcohol_percentage_in_original_solution_l242_24293

theorem alcohol_percentage_in_original_solution 
  (original_volume : ℝ)
  (added_water : ℝ)
  (new_mixture_percentage : ℝ)
  (h1 : original_volume = 3)
  (h2 : added_water = 1)
  (h3 : new_mixture_percentage = 24.75) :
  let new_volume := original_volume + added_water
  let alcohol_amount := (new_mixture_percentage / 100) * new_volume
  (alcohol_amount / original_volume) * 100 = 33 := by
sorry


end alcohol_percentage_in_original_solution_l242_24293


namespace coefficient_x_cubed_expansion_l242_24242

/-- The coefficient of x^3 in the expansion of (1+2x^2)(1+x)^4 is 12 -/
theorem coefficient_x_cubed_expansion : ∃ (p : Polynomial ℝ),
  p = (1 + 2 * X^2) * (1 + X)^4 ∧ p.coeff 3 = 12 := by
  sorry

end coefficient_x_cubed_expansion_l242_24242


namespace dads_dimes_l242_24272

theorem dads_dimes (initial : ℕ) (from_mother : ℕ) (total : ℕ) : 
  initial = 7 → from_mother = 4 → total = 19 → 
  total - (initial + from_mother) = 8 := by
sorry

end dads_dimes_l242_24272


namespace rectangular_prism_volume_l242_24267

/-- 
Given a rectangular prism with edges in the ratio 3:2:1 and 
the sum of all edge lengths equal to 72 cm, its volume is 162 cubic centimeters.
-/
theorem rectangular_prism_volume (l w h : ℝ) : 
  l / w = 3 / 2 → 
  w / h = 2 / 1 → 
  4 * (l + w + h) = 72 → 
  l * w * h = 162 := by
sorry

end rectangular_prism_volume_l242_24267


namespace speed_with_stream_is_ten_l242_24288

/-- The speed of a man rowing a boat with and against a stream. -/
structure BoatSpeed where
  /-- Speed against the stream in km/h -/
  against_stream : ℝ
  /-- Speed in still water in km/h -/
  still_water : ℝ

/-- Calculate the speed with the stream given speeds against stream and in still water -/
def speed_with_stream (bs : BoatSpeed) : ℝ :=
  2 * bs.still_water - bs.against_stream

/-- Theorem stating that given the specified conditions, the speed with the stream is 10 km/h -/
theorem speed_with_stream_is_ten (bs : BoatSpeed) 
    (h1 : bs.against_stream = 10) 
    (h2 : bs.still_water = 7) : 
    speed_with_stream bs = 10 := by
  sorry

end speed_with_stream_is_ten_l242_24288


namespace symmetry_oyz_coordinates_l242_24228

/-- A point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- The Oyz plane -/
def Oyz : Set Point3D :=
  {p : Point3D | p.x = 0}

/-- Symmetry with respect to the Oyz plane -/
def symmetricOyz (a b : Point3D) : Prop :=
  b.x = -a.x ∧ b.y = a.y ∧ b.z = a.z

theorem symmetry_oyz_coordinates :
  let a : Point3D := ⟨3, 4, 5⟩
  let b : Point3D := ⟨-3, 4, 5⟩
  symmetricOyz a b := by sorry

end symmetry_oyz_coordinates_l242_24228


namespace trailingZeros_50_factorial_l242_24268

/-- The number of trailing zeros in n! -/
def trailingZeros (n : ℕ) : ℕ :=
  (n / 5) + (n / 25)

/-- Theorem: The number of trailing zeros in 50! is 12 -/
theorem trailingZeros_50_factorial : trailingZeros 50 = 12 := by
  sorry

end trailingZeros_50_factorial_l242_24268


namespace inequality_proof_l242_24204

theorem inequality_proof (a b c d : ℝ) 
  (non_neg_a : 0 ≤ a) (non_neg_b : 0 ≤ b) (non_neg_c : 0 ≤ c) (non_neg_d : 0 ≤ d)
  (sum_condition : a * b + b * c + c * d + d * a = 1) : 
  a^3 / (b + c + d) + b^3 / (c + d + a) + c^3 / (a + b + d) + d^3 / (a + b + c) ≥ 1/3 := by
  sorry

end inequality_proof_l242_24204


namespace min_value_of_f_l242_24254

/-- The function f(x) = e^x - e^(2x) has a minimum value of -e^2 -/
theorem min_value_of_f (x : ℝ) : 
  let f : ℝ → ℝ := λ x => Real.exp x - Real.exp (2 * x)
  ∃ m : ℝ, (∀ x, f x ≥ m) ∧ (∃ x₀, f x₀ = m) ∧ m = -Real.exp 2 := by
  sorry

end min_value_of_f_l242_24254


namespace coefficient_of_x_squared_l242_24213

theorem coefficient_of_x_squared (x : ℝ) : 
  let expr := 2 * (x^2 - 5) + 6 * (3*x^2 - 2*x + 4) - 4 * (x^2 - 3*x)
  ∃ (a b c : ℝ), expr = 16 * x^2 + a * x + b * x + c :=
by sorry

end coefficient_of_x_squared_l242_24213


namespace frog_jump_distance_l242_24259

/-- The jumping contest between a grasshopper and a frog -/
theorem frog_jump_distance (grasshopper_jump : ℕ) (frog_extra_distance : ℕ) : 
  grasshopper_jump = 9 → frog_extra_distance = 3 → 
  grasshopper_jump + frog_extra_distance = 12 := by
  sorry

end frog_jump_distance_l242_24259


namespace max_value_2q_minus_r_l242_24237

theorem max_value_2q_minus_r : 
  ∃ (q r : ℕ+), 1024 = 23 * q + r ∧ 
  ∀ (q' r' : ℕ+), 1024 = 23 * q' + r' → 2 * q - r ≥ 2 * q' - r' ∧
  2 * q - r = 76 := by
sorry

end max_value_2q_minus_r_l242_24237


namespace room_width_calculation_l242_24235

/-- Given a rectangular room with known length, paving cost per square meter, and total paving cost,
    calculate the width of the room. -/
theorem room_width_calculation (length : ℝ) (cost_per_sqm : ℝ) (total_cost : ℝ) 
    (h1 : length = 5.5)
    (h2 : cost_per_sqm = 300)
    (h3 : total_cost = 6187.5) :
    total_cost / cost_per_sqm / length = 3.75 := by
  sorry

end room_width_calculation_l242_24235


namespace certain_number_proof_l242_24281

theorem certain_number_proof (p q : ℝ) 
  (h1 : 3 / p = 8) 
  (h2 : p - q = 0.20833333333333334) : 
  3 / q = 18 := by
  sorry

end certain_number_proof_l242_24281


namespace complex_equation_solution_l242_24283

theorem complex_equation_solution (x y : ℝ) : 
  (Complex.mk (2 * x - 1) 1 = Complex.mk y (y - 2)) → x = 2 ∧ y = 3 := by
  sorry

end complex_equation_solution_l242_24283


namespace geometric_sequence_property_l242_24280

def is_geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = a n * r

theorem geometric_sequence_property
  (a : ℕ → ℝ)
  (h_pos : ∀ n, a n > 0)
  (h_geom : is_geometric_sequence a)
  (h_prod1 : a 1 * a 2 * a 3 = 5)
  (h_prod2 : a 4 * a 8 * a 9 = 10) :
  a 4 * a 5 * a 6 = 5 * Real.sqrt 2 :=
sorry

end geometric_sequence_property_l242_24280


namespace remainder_4523_div_32_l242_24248

theorem remainder_4523_div_32 : 4523 % 32 = 11 := by
  sorry

end remainder_4523_div_32_l242_24248


namespace max_ranked_participants_l242_24286

/-- The maximum number of participants that can be awarded a rank in a chess tournament -/
theorem max_ranked_participants (n : ℕ) (rank_threshold : ℚ) : 
  n = 30 →
  rank_threshold = 60 / 100 →
  ∃ (max_ranked : ℕ), max_ranked = 23 ∧ 
    (∀ (ranked : ℕ), 
      ranked ≤ n ∧
      (ranked : ℚ) * rank_threshold * (n - 1 : ℚ) ≤ (n * (n - 1) / 2 : ℚ) →
      ranked ≤ max_ranked) :=
by sorry

end max_ranked_participants_l242_24286


namespace n_has_four_digits_l242_24206

def n : ℕ := 9376

theorem n_has_four_digits :
  (∃ k : ℕ, n^2 % 10000 = n) →
  (∃ m : ℕ, 10^3 ≤ n ∧ n < 10^4) :=
by sorry

end n_has_four_digits_l242_24206


namespace repeating_decimal_as_fraction_l242_24269

/-- The repeating decimal 0.4747... expressed as a real number -/
def repeating_decimal : ℚ :=
  (0.47 : ℚ) + (0.0047 : ℚ) / (1 - (0.01 : ℚ))

theorem repeating_decimal_as_fraction :
  repeating_decimal = 47 / 99 := by
  sorry

end repeating_decimal_as_fraction_l242_24269


namespace parallelogram_side_length_l242_24245

/-- Given a parallelogram with adjacent sides of lengths 3s and s units, 
    forming a 60-degree angle, and having an area of 9√3 square units, 
    prove that s = √6. -/
theorem parallelogram_side_length (s : ℝ) : 
  s > 0 →
  let adjacent_side1 := 3 * s
  let adjacent_side2 := s
  let angle := Real.pi / 3  -- 60 degrees in radians
  let area := 9 * Real.sqrt 3
  area = adjacent_side1 * adjacent_side2 * Real.sin angle →
  s = Real.sqrt 6 := by
sorry

end parallelogram_side_length_l242_24245


namespace corner_sum_9x9_board_l242_24271

/- Define the size of the checkerboard -/
def boardSize : Nat := 9

/- Define the total number of squares -/
def totalSquares : Nat := boardSize * boardSize

/- Define the positions of the corner and adjacent numbers -/
def cornerPositions : List Nat := [1, 2, 8, 9, 73, 74, 80, 81]

/- Theorem statement -/
theorem corner_sum_9x9_board :
  (List.sum cornerPositions) = 328 := by
  sorry

end corner_sum_9x9_board_l242_24271


namespace parallel_line_slope_l242_24263

/-- The slope of a line parallel to 3x - 6y = 12 is 1/2 -/
theorem parallel_line_slope (x y : ℝ) :
  (3 * x - 6 * y = 12) → 
  (∃ (m b : ℝ), y = m * x + b ∧ m = 1/2) :=
by sorry

end parallel_line_slope_l242_24263


namespace friend_lunch_cost_l242_24264

theorem friend_lunch_cost (total : ℕ) (difference : ℕ) (friend_cost : ℕ) : 
  total = 19 →
  difference = 3 →
  friend_cost = total / 2 + difference →
  friend_cost = 11 := by
sorry

end friend_lunch_cost_l242_24264


namespace chicken_wings_distribution_l242_24223

theorem chicken_wings_distribution (num_friends : ℕ) (total_wings : ℕ) :
  num_friends = 9 →
  total_wings = 27 →
  ∃ (wings_per_person : ℕ), 
    wings_per_person * num_friends = total_wings ∧
    wings_per_person = 3 :=
by
  sorry

end chicken_wings_distribution_l242_24223


namespace race_result_l242_24239

/-- Represents a runner in the race -/
structure Runner where
  speed : ℝ
  (speed_pos : 0 < speed)

/-- The race setup -/
structure Race where
  anton : Runner
  seryozha : Runner
  tolya : Runner
  (different_speeds : anton.speed ≠ seryozha.speed ∧ seryozha.speed ≠ tolya.speed ∧ anton.speed ≠ tolya.speed)

/-- The race conditions -/
def race_conditions (r : Race) : Prop :=
  let t_anton := 100 / r.anton.speed
  let d_seryozha := r.seryozha.speed * t_anton
  let t_seryozha := 100 / r.seryozha.speed
  let d_tolya := r.tolya.speed * t_seryozha
  d_seryozha = 90 ∧ d_tolya = 90

theorem race_result (r : Race) (h : race_conditions r) :
  r.tolya.speed * (100 / r.anton.speed) = 81 := by
  sorry

#check race_result

end race_result_l242_24239


namespace prob_red_or_green_l242_24261

/-- The probability of drawing a red or green marble from a bag -/
theorem prob_red_or_green (red green yellow : ℕ) (h : red = 4 ∧ green = 3 ∧ yellow = 6) :
  (red + green : ℚ) / (red + green + yellow) = 7 / 13 := by
  sorry

end prob_red_or_green_l242_24261


namespace radio_loss_percentage_l242_24230

/-- Calculates the loss percentage given the cost price and selling price. -/
def loss_percentage (cost_price selling_price : ℚ) : ℚ :=
  (cost_price - selling_price) / cost_price * 100

/-- Theorem stating that for a radio with cost price 1500 and selling price 1275,
    the loss percentage is 15%. -/
theorem radio_loss_percentage :
  loss_percentage 1500 1275 = 15 := by sorry

end radio_loss_percentage_l242_24230


namespace f_continuous_iff_b_eq_12_l242_24209

-- Define the piecewise function f(x)
noncomputable def f (b : ℝ) (x : ℝ) : ℝ :=
  if x > -2 then 3 * x + b else -x + 4

-- Theorem statement
theorem f_continuous_iff_b_eq_12 (b : ℝ) :
  Continuous (f b) ↔ b = 12 := by
  sorry

end f_continuous_iff_b_eq_12_l242_24209


namespace relationship_abc_l242_24219

theorem relationship_abc (a b c : Real) 
  (ha : a = 3^(0.3 : Real))
  (hb : b = Real.log 3 / Real.log π)
  (hc : c = Real.log 2 / Real.log 0.3) :
  c < b ∧ b < a := by
  sorry

end relationship_abc_l242_24219


namespace prob_both_truth_l242_24208

/-- The probability that A speaks the truth -/
def prob_A_truth : ℝ := 0.75

/-- The probability that B speaks the truth -/
def prob_B_truth : ℝ := 0.60

/-- The theorem stating the probability of A and B both telling the truth simultaneously -/
theorem prob_both_truth : prob_A_truth * prob_B_truth = 0.45 := by sorry

end prob_both_truth_l242_24208


namespace area_between_circles_l242_24299

-- Define the circles
def Circle (center : ℝ × ℝ) (radius : ℝ) := {p : ℝ × ℝ | (p.1 - center.1)^2 + (p.2 - center.2)^2 = radius^2}

-- Define the centers of the circles
def center_X : ℝ × ℝ := (0, 0)
def center_Y : ℝ × ℝ := (2, 0)
def center_Z : ℝ × ℝ := (0, 2)

-- Define the circles
def X := Circle center_X 1
def Y := Circle center_Y 1
def Z := Circle center_Z 1

-- Define the area function
def area (c : Set (ℝ × ℝ)) : ℝ := sorry

-- Theorem statement
theorem area_between_circles :
  (∀ p ∈ X ∩ Y, p = (1, 0)) →  -- X and Y are tangent
  (∃ p, p ∈ X ∩ Z ∧ p ≠ center_X) →  -- Z is tangent to X
  (∀ p, p ∉ Y ∩ Z) →  -- Z does not intersect Y
  area (Z \ X) = π / 2 := by sorry

end area_between_circles_l242_24299


namespace isosceles_triangle_base_l242_24273

/-- An isosceles triangle with side lengths 3 and 7 has a base of length 3. -/
theorem isosceles_triangle_base (a b : ℝ) (h1 : a = 3 ∨ a = 7) (h2 : b = 3 ∨ b = 7) (h3 : a ≠ b) :
  ∃ (x y : ℝ), x = y ∧ x + y > b ∧ x = 7 ∧ b = 3 :=
sorry

end isosceles_triangle_base_l242_24273


namespace sum_of_roots_quadratic_sum_of_roots_specific_equation_l242_24243

theorem sum_of_roots_quadratic (a b c : ℝ) (h : a ≠ 0) :
  let f : ℝ → ℝ := λ x => a * x^2 + b * x + c
  (∃ x y : ℝ, x ≠ y ∧ f x = 0 ∧ f y = 0) →
  x + y = -b / a :=
sorry

theorem sum_of_roots_specific_equation :
  let f : ℝ → ℝ := λ x => x^2 - 2003 * x - 2004
  (∃ x y : ℝ, x ≠ y ∧ f x = 0 ∧ f y = 0) →
  x + y = 2003 :=
sorry

end sum_of_roots_quadratic_sum_of_roots_specific_equation_l242_24243


namespace subtraction_with_division_l242_24298

theorem subtraction_with_division : 3034 - (1002 / 200.4) = 3029 := by
  sorry

end subtraction_with_division_l242_24298


namespace floor_sqrt_33_squared_l242_24277

theorem floor_sqrt_33_squared : ⌊Real.sqrt 33⌋^2 = 25 := by
  sorry

end floor_sqrt_33_squared_l242_24277


namespace matrix_not_invertible_iff_y_eq_two_fifths_l242_24226

def matrix (y : ℝ) : Matrix (Fin 2) (Fin 2) ℝ := !![2 + y, 6; 4 - y, 9]

theorem matrix_not_invertible_iff_y_eq_two_fifths :
  ∀ y : ℝ, ¬(Matrix.det (matrix y) ≠ 0) ↔ y = 2/5 := by sorry

end matrix_not_invertible_iff_y_eq_two_fifths_l242_24226


namespace gross_revenue_increase_l242_24218

theorem gross_revenue_increase
  (original_price : ℝ)
  (original_quantity : ℝ)
  (price_decrease_percentage : ℝ)
  (quantity_increase_percentage : ℝ)
  (h1 : price_decrease_percentage = 20)
  (h2 : quantity_increase_percentage = 70)
  : let new_price := original_price * (1 - price_decrease_percentage / 100)
    let new_quantity := original_quantity * (1 + quantity_increase_percentage / 100)
    let original_revenue := original_price * original_quantity
    let new_revenue := new_price * new_quantity
    (new_revenue - original_revenue) / original_revenue * 100 = 36 := by
  sorry

end gross_revenue_increase_l242_24218


namespace place_value_ratio_l242_24222

def number : ℚ := 86743.2951

def place_value_6 : ℚ := 10000
def place_value_5 : ℚ := 0.1

theorem place_value_ratio :
  place_value_6 / place_value_5 = 100000 := by
  sorry

#check place_value_ratio

end place_value_ratio_l242_24222


namespace math_problem_distribution_l242_24260

theorem math_problem_distribution :
  let num_problems : ℕ := 7
  let num_friends : ℕ := 12
  (num_friends ^ num_problems : ℕ) = 35831808 :=
by sorry

end math_problem_distribution_l242_24260


namespace range_of_inequality_l242_24247

/-- An even function that is monotonically decreasing on (-∞,0] -/
class EvenDecreasingFunction (f : ℝ → ℝ) : Prop where
  even : ∀ x, f x = f (-x)
  decreasing : ∀ {x y}, x ≤ y → y ≤ 0 → f y ≤ f x

/-- The theorem statement -/
theorem range_of_inequality (f : ℝ → ℝ) [EvenDecreasingFunction f] :
  {x : ℝ | f (2*x + 1) < f 3} = Set.Ioo (-2) 1 := by sorry

end range_of_inequality_l242_24247


namespace theater_rows_count_l242_24266

/-- Represents a theater seating arrangement -/
structure Theater where
  total_seats : ℕ
  num_rows : ℕ
  first_row_seats : ℕ

/-- Checks if the theater satisfies the given conditions -/
def is_valid_theater (t : Theater) : Prop :=
  t.num_rows > 16 ∧
  t.total_seats = (t.first_row_seats + (t.first_row_seats + t.num_rows - 1)) * t.num_rows / 2

/-- The main theorem stating that a theater with 1000 seats satisfying the conditions has 25 rows -/
theorem theater_rows_count : 
  ∀ t : Theater, t.total_seats = 1000 → is_valid_theater t → t.num_rows = 25 :=
by sorry

end theater_rows_count_l242_24266


namespace percentage_difference_l242_24285

theorem percentage_difference (x y z n : ℝ) : 
  x = 8 * y ∧ 
  y = 2 * |z - n| ∧ 
  z = 1.1 * n → 
  (x - y) / x * 100 = 87.5 := by
sorry

end percentage_difference_l242_24285


namespace square_sum_ge_third_square_sum_l242_24238

theorem square_sum_ge_third_square_sum (a b c : ℝ) :
  a^2 + b^2 + c^2 ≥ (1/3) * (a + b + c)^2 := by
  sorry

end square_sum_ge_third_square_sum_l242_24238


namespace solution_set_linear_inequalities_l242_24229

theorem solution_set_linear_inequalities :
  ∀ x : ℝ, (x - 2 > 1 ∧ x < 4) ↔ (3 < x ∧ x < 4) := by sorry

end solution_set_linear_inequalities_l242_24229


namespace log2_derivative_l242_24246

theorem log2_derivative (x : ℝ) (h : x > 0) : 
  deriv (λ x => Real.log x / Real.log 2) x = 1 / (x * Real.log 2) := by
  sorry

end log2_derivative_l242_24246


namespace construction_rearrangements_l242_24294

def word : String := "CONSTRUCTION"

def is_vowel (c : Char) : Bool :=
  c ∈ ['A', 'E', 'I', 'O', 'U']

def vowels : List Char :=
  word.toList.filter is_vowel

def consonants : List Char :=
  word.toList.filter (fun c => !is_vowel c)

def vowel_arrangements : ℕ :=
  vowels.length.factorial

def consonant_arrangements : ℕ :=
  consonants.length.factorial / ((consonants.countP (· = 'C')).factorial *
                                 (consonants.countP (· = 'T')).factorial *
                                 (consonants.countP (· = 'N')).factorial)

theorem construction_rearrangements :
  vowel_arrangements * consonant_arrangements = 30240 := by
  sorry

end construction_rearrangements_l242_24294


namespace candy_distribution_l242_24200

theorem candy_distribution (num_students : ℕ) (pieces_per_student : ℕ) 
  (h1 : num_students = 43) 
  (h2 : pieces_per_student = 8) : 
  num_students * pieces_per_student = 344 := by
  sorry

end candy_distribution_l242_24200


namespace expression_simplification_l242_24292

theorem expression_simplification : 
  ∃ (a b c : ℕ+), 
    (2 * Real.sqrt 3 + 2 / Real.sqrt 3 + 3 * Real.sqrt 2 + 3 / Real.sqrt 2 = (a * Real.sqrt 3 + b * Real.sqrt 2) / c) ∧
    (∀ (a' b' c' : ℕ+), 
      (2 * Real.sqrt 3 + 2 / Real.sqrt 3 + 3 * Real.sqrt 2 + 3 / Real.sqrt 2 = (a' * Real.sqrt 3 + b' * Real.sqrt 2) / c') →
      c ≤ c') ∧
    (a + b + c = 45) := by
  sorry

end expression_simplification_l242_24292


namespace simplify_fraction_1_simplify_fraction_2_l242_24289

-- Problem 1
theorem simplify_fraction_1 (a : ℝ) (h : a ≠ 1) :
  (a^2 / (a - 1)) - (a / (a - 1)) = a :=
sorry

-- Problem 2
theorem simplify_fraction_2 (x : ℝ) (h : x ≠ -1) :
  (x^2 / (x + 1)) - x + 1 = 1 / (x + 1) :=
sorry

end simplify_fraction_1_simplify_fraction_2_l242_24289


namespace perfect_squares_condition_l242_24252

theorem perfect_squares_condition (n : ℤ) : 
  (∃ a : ℤ, 4 * n + 1 = a ^ 2) ∧ (∃ b : ℤ, 9 * n + 1 = b ^ 2) → n = 0 :=
by sorry

end perfect_squares_condition_l242_24252


namespace range_of_a_l242_24256

def sequence_a (a : ℝ) (n : ℕ) : ℝ :=
  if n ≤ 5 then (5 - a) * n - 11 else a^(n - 4)

theorem range_of_a (a : ℝ) :
  (∀ n m : ℕ, n < m → sequence_a a n < sequence_a a m) →
  2 < a ∧ a < 5 := by
  sorry

end range_of_a_l242_24256


namespace rectangle_similarity_l242_24279

theorem rectangle_similarity (x y : ℝ) (hxy : 0 < x ∧ x < y) :
  let r := (y, x)
  let r' := (y - x, x)
  let r'' := if y - x < x then ((y - x), (2 * x - y)) else (x, (y - 2 * x))
  ¬ (r'.1 / r'.2 = r.1 / r.2) →
  (r''.1 / r''.2 = r.1 / r.2) →
  y / x = 1 + Real.sqrt 2 ∨ y / x = (1 + Real.sqrt 5) / 2 := by
sorry

end rectangle_similarity_l242_24279


namespace four_digit_sum_l242_24257

theorem four_digit_sum (a b c d : ℕ) : 
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d →
  a * b * c * d = 810 →
  1000 ≤ a * 1000 + b * 100 + c * 10 + d →
  a * 1000 + b * 100 + c * 10 + d < 10000 →
  a + b + c + d = 23 := by
sorry

end four_digit_sum_l242_24257


namespace mary_initial_weight_l242_24295

/-- Mary's weight changes and final weight --/
structure WeightChanges where
  initial_loss : ℕ
  final_weight : ℕ

/-- Calculate Mary's initial weight given her weight changes --/
def calculate_initial_weight (changes : WeightChanges) : ℕ :=
  changes.final_weight         -- Start with final weight
  + changes.initial_loss * 3   -- Add back the triple loss
  - changes.initial_loss * 2   -- Subtract the double gain
  - 6                          -- Subtract the final gain
  + changes.initial_loss       -- Add back the initial loss

/-- Theorem stating that Mary's initial weight was 99 pounds --/
theorem mary_initial_weight :
  let changes : WeightChanges := { initial_loss := 12, final_weight := 81 }
  calculate_initial_weight changes = 99 := by
  sorry


end mary_initial_weight_l242_24295


namespace chickens_and_rabbits_equation_l242_24249

/-- Represents the number of chickens in the cage -/
def chickens : ℕ := sorry

/-- Represents the number of rabbits in the cage -/
def rabbits : ℕ := sorry

/-- The total number of heads in the cage -/
def total_heads : ℕ := 16

/-- The total number of feet in the cage -/
def total_feet : ℕ := 44

/-- Theorem stating that the system of equations correctly represents the problem -/
theorem chickens_and_rabbits_equation :
  (chickens + rabbits = total_heads) ∧
  (2 * chickens + 4 * rabbits = total_feet) :=
sorry

end chickens_and_rabbits_equation_l242_24249


namespace negative_one_greater_than_negative_sqrt_three_l242_24297

theorem negative_one_greater_than_negative_sqrt_three : -1 > -Real.sqrt 3 := by
  sorry

end negative_one_greater_than_negative_sqrt_three_l242_24297


namespace delta_value_l242_24274

theorem delta_value : ∀ Δ : ℤ, 5 * (-3) = Δ - 3 → Δ = -12 := by
  sorry

end delta_value_l242_24274
