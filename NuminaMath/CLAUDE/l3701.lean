import Mathlib

namespace NUMINAMATH_CALUDE_sum_x_coordinates_on_parabola_l3701_370118

/-- The parabola equation y = x² - 2x + 1 -/
def parabola (x : ℝ) : ℝ := x^2 - 2*x + 1

/-- Theorem: For any two points P(x₁, 1) and Q(x₂, 1) on the parabola y = x² - 2x + 1,
    the sum of their x-coordinates (x₁ + x₂) is equal to 2. -/
theorem sum_x_coordinates_on_parabola (x₁ x₂ : ℝ) 
    (h₁ : parabola x₁ = 1) 
    (h₂ : parabola x₂ = 1) : 
  x₁ + x₂ = 2 := by
  sorry

end NUMINAMATH_CALUDE_sum_x_coordinates_on_parabola_l3701_370118


namespace NUMINAMATH_CALUDE_common_root_quadratic_l3701_370126

theorem common_root_quadratic (a b : ℝ) : 
  (∃! t : ℝ, t^2 + a*t + b = 0 ∧ t^2 + b*t + a = 0) → 
  (a + b + 1 = 0 ∧ a ≠ b) :=
by sorry

end NUMINAMATH_CALUDE_common_root_quadratic_l3701_370126


namespace NUMINAMATH_CALUDE_part_one_part_two_l3701_370143

-- Define the function f
def f (x k m : ℝ) : ℝ := |x^2 - k*x - m|

-- Theorem for part (1)
theorem part_one (k m : ℝ) :
  m = 2 * k^2 →
  (∀ x y, 1 < x ∧ x < y → f x k m < f y k m) →
  -1 ≤ k ∧ k ≤ 1/2 := by sorry

-- Theorem for part (2)
theorem part_two (k m a b : ℝ) :
  (∀ x, x ∈ Set.Icc a b → f x k m ≤ 1) →
  b - a ≤ 2 * Real.sqrt 2 := by sorry

end NUMINAMATH_CALUDE_part_one_part_two_l3701_370143


namespace NUMINAMATH_CALUDE_root_sum_reciprocals_l3701_370121

theorem root_sum_reciprocals (p q r s : ℂ) : 
  (p^4 - 6*p^3 + 23*p^2 - 72*p + 8 = 0) →
  (q^4 - 6*q^3 + 23*q^2 - 72*q + 8 = 0) →
  (r^4 - 6*r^3 + 23*r^2 - 72*r + 8 = 0) →
  (s^4 - 6*s^3 + 23*s^2 - 72*s + 8 = 0) →
  1/(p*q) + 1/(p*r) + 1/(p*s) + 1/(q*r) + 1/(q*s) + 1/(r*s) = -9 :=
by sorry

end NUMINAMATH_CALUDE_root_sum_reciprocals_l3701_370121


namespace NUMINAMATH_CALUDE_total_shelves_calculation_l3701_370109

/-- Calculate the total number of shelves needed for coloring books and puzzle books --/
theorem total_shelves_calculation (initial_coloring : ℕ) (initial_puzzle : ℕ)
                                  (sold_coloring : ℕ) (sold_puzzle : ℕ)
                                  (coloring_per_shelf : ℕ) (puzzle_per_shelf : ℕ)
                                  (h1 : initial_coloring = 435)
                                  (h2 : initial_puzzle = 523)
                                  (h3 : sold_coloring = 218)
                                  (h4 : sold_puzzle = 304)
                                  (h5 : coloring_per_shelf = 17)
                                  (h6 : puzzle_per_shelf = 22) :
  (((initial_coloring - sold_coloring) + coloring_per_shelf - 1) / coloring_per_shelf +
   ((initial_puzzle - sold_puzzle) + puzzle_per_shelf - 1) / puzzle_per_shelf) = 23 := by
  sorry

#eval ((435 - 218) + 17 - 1) / 17 + ((523 - 304) + 22 - 1) / 22

end NUMINAMATH_CALUDE_total_shelves_calculation_l3701_370109


namespace NUMINAMATH_CALUDE_other_root_of_quadratic_l3701_370181

theorem other_root_of_quadratic (m : ℝ) : 
  (∃ x, 3 * x^2 + m * x = 5) → 
  (3 * 5^2 + m * 5 = 5) → 
  (3 * (-1/3)^2 + m * (-1/3) = 5) :=
by
  sorry

end NUMINAMATH_CALUDE_other_root_of_quadratic_l3701_370181


namespace NUMINAMATH_CALUDE_student_count_l3701_370179

/-- The number of storybooks distributed to the class -/
def total_books : ℕ := 60

/-- The number of students in the class -/
def num_students : ℕ := 20

theorem student_count :
  (num_students < total_books) ∧ 
  (total_books - num_students) % 2 = 0 ∧
  (total_books - num_students) / 2 = num_students :=
by sorry

end NUMINAMATH_CALUDE_student_count_l3701_370179


namespace NUMINAMATH_CALUDE_total_passengers_is_120_l3701_370184

/-- The total number of passengers on the flight -/
def total_passengers : ℕ := 120

/-- The proportion of female passengers -/
def female_proportion : ℚ := 55 / 100

/-- The proportion of passengers in first class -/
def first_class_proportion : ℚ := 10 / 100

/-- The proportion of male passengers in first class -/
def male_first_class_proportion : ℚ := 1 / 3

/-- The number of females in coach class -/
def females_in_coach : ℕ := 58

/-- Theorem stating that the total number of passengers is 120 -/
theorem total_passengers_is_120 : 
  total_passengers = 120 ∧
  female_proportion * total_passengers = 
    (females_in_coach : ℚ) + 
    (1 - male_first_class_proportion) * first_class_proportion * total_passengers :=
by sorry

end NUMINAMATH_CALUDE_total_passengers_is_120_l3701_370184


namespace NUMINAMATH_CALUDE_quadratic_equation_identity_l3701_370195

theorem quadratic_equation_identity 
  (a₀ a₁ a₂ r s x : ℝ) 
  (h₁ : a₂ ≠ 0) 
  (h₂ : a₀ ≠ 0) 
  (h₃ : a₀ + a₁ * r + a₂ * r^2 = 0) 
  (h₄ : a₀ + a₁ * s + a₂ * s^2 = 0) :
  a₀ + a₁ * x + a₂ * x^2 = a₀ * (1 - x / r) * (1 - x / s) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_identity_l3701_370195


namespace NUMINAMATH_CALUDE_binomial_probability_one_third_l3701_370105

/-- A random variable following a binomial distribution -/
structure BinomialVariable where
  n : ℕ
  p : ℝ
  h_p_range : 0 ≤ p ∧ p ≤ 1

/-- The expectation of a binomial variable -/
def expectation (X : BinomialVariable) : ℝ := X.n * X.p

/-- The variance of a binomial variable -/
def variance (X : BinomialVariable) : ℝ := X.n * X.p * (1 - X.p)

theorem binomial_probability_one_third 
  (X : BinomialVariable) 
  (h_expectation : expectation X = 30)
  (h_variance : variance X = 20) : 
  X.p = 1/3 := by
  sorry

end NUMINAMATH_CALUDE_binomial_probability_one_third_l3701_370105


namespace NUMINAMATH_CALUDE_fraction_calculation_l3701_370182

theorem fraction_calculation (x y : ℚ) 
  (hx : x = 7/8) 
  (hy : y = 5/6) 
  (hx_nonzero : x ≠ 0) 
  (hy_nonzero : y ≠ 0) : 
  (4*x - 6*y) / (60*x*y) = -6/175 := by
  sorry

end NUMINAMATH_CALUDE_fraction_calculation_l3701_370182


namespace NUMINAMATH_CALUDE_sin_pi_minus_alpha_l3701_370185

theorem sin_pi_minus_alpha (α : Real) : 
  (∃ (x y : Real), x = Real.sqrt 3 ∧ y = 1 ∧ x = Real.tan α * y) →
  Real.sin (Real.pi - α) = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_sin_pi_minus_alpha_l3701_370185


namespace NUMINAMATH_CALUDE_insufficient_shots_l3701_370107

-- Define the number of points on the circle
def n : ℕ := 29

-- Define the number of shots
def shots : ℕ := 134

-- Function to calculate binomial coefficient
def binomial_coefficient (n k : ℕ) : ℕ := 
  Nat.choose n k

-- Total number of possible triangles
def total_triangles : ℕ := 
  binomial_coefficient n 3

-- Number of triangles that can be hit by one shot
def triangles_per_shot : ℕ := n - 2

-- Maximum number of triangles that can be hit by all shots
def max_hit_triangles : ℕ := 
  shots * triangles_per_shot

-- Theorem stating that 134 shots are insufficient
theorem insufficient_shots : max_hit_triangles < total_triangles := by
  sorry


end NUMINAMATH_CALUDE_insufficient_shots_l3701_370107


namespace NUMINAMATH_CALUDE_no_nonzero_integer_solution_l3701_370139

theorem no_nonzero_integer_solution :
  ∀ (x y z : ℤ), 2 * x^4 + 2 * x^2 * y^2 + y^4 = z^2 → x = 0 := by
  sorry

end NUMINAMATH_CALUDE_no_nonzero_integer_solution_l3701_370139


namespace NUMINAMATH_CALUDE_overall_loss_percentage_l3701_370141

/-- Calculate the overall loss percentage for three items given their cost and selling prices -/
theorem overall_loss_percentage
  (cp_radio : ℚ) (cp_speaker : ℚ) (cp_headphones : ℚ)
  (sp_radio : ℚ) (sp_speaker : ℚ) (sp_headphones : ℚ)
  (h1 : cp_radio = 1500)
  (h2 : cp_speaker = 2500)
  (h3 : cp_headphones = 800)
  (h4 : sp_radio = 1275)
  (h5 : sp_speaker = 2300)
  (h6 : sp_headphones = 700) :
  let total_cp := cp_radio + cp_speaker + cp_headphones
  let total_sp := sp_radio + sp_speaker + sp_headphones
  let loss := total_cp - total_sp
  let loss_percentage := (loss / total_cp) * 100
  abs (loss_percentage - 10.94) < 0.01 := by
  sorry

end NUMINAMATH_CALUDE_overall_loss_percentage_l3701_370141


namespace NUMINAMATH_CALUDE_min_side_length_l3701_370116

theorem min_side_length (PQ PR SR SQ : ℝ) (h1 : PQ = 7.5) (h2 : PR = 14.5) (h3 : SR = 9.5) (h4 : SQ = 23.5) :
  ∃ (QR : ℕ), (QR : ℝ) > PR - PQ ∧ (QR : ℝ) > SQ - SR ∧ ∀ (n : ℕ), (n : ℝ) > PR - PQ ∧ (n : ℝ) > SQ - SR → n ≥ QR :=
by
  sorry

#check min_side_length

end NUMINAMATH_CALUDE_min_side_length_l3701_370116


namespace NUMINAMATH_CALUDE_fence_cost_l3701_370174

/-- The cost of building a fence around a square plot -/
theorem fence_cost (area : ℝ) (price_per_foot : ℝ) (h1 : area = 289) (h2 : price_per_foot = 60) :
  4 * Real.sqrt area * price_per_foot = 4080 := by
  sorry

#check fence_cost

end NUMINAMATH_CALUDE_fence_cost_l3701_370174


namespace NUMINAMATH_CALUDE_imaginary_part_of_reciprocal_plus_one_l3701_370180

theorem imaginary_part_of_reciprocal_plus_one (z : ℂ) (x y : ℝ) 
  (h1 : z = x + y * I) 
  (h2 : z ≠ x) -- z is nonreal
  (h3 : Complex.abs z = 1) : 
  Complex.im (1 / (1 + z)) = -y / (2 * (1 + x)) :=
by sorry

end NUMINAMATH_CALUDE_imaginary_part_of_reciprocal_plus_one_l3701_370180


namespace NUMINAMATH_CALUDE_intersection_M_N_l3701_370198

def M : Set ℝ := {x | x^2 - 3*x - 28 ≤ 0}
def N : Set ℝ := {x | x^2 - x - 6 > 0}

theorem intersection_M_N : M ∩ N = {x : ℝ | -4 ≤ x ∧ x ≤ -2 ∨ 3 < x ∧ x ≤ 7} := by
  sorry

end NUMINAMATH_CALUDE_intersection_M_N_l3701_370198


namespace NUMINAMATH_CALUDE_equilateral_triangle_hexagon_area_l3701_370123

-- Define the triangle PQR
def Triangle (P Q R : ℝ × ℝ) : Prop :=
  ∃ (s : ℝ), s > 0 ∧ 
  (dist P Q = s) ∧ (dist Q R = s) ∧ (dist R P = s)

-- Define the perimeter of the triangle
def Perimeter (P Q R : ℝ × ℝ) : ℝ :=
  dist P Q + dist Q R + dist R P

-- Define the circumcircle and perpendicular bisectors
def CircumcirclePerpBisectors (P Q R P' Q' R' : ℝ × ℝ) : Prop :=
  ∃ (O : ℝ × ℝ), 
    dist O P = dist O Q ∧ dist O Q = dist O R ∧
    dist O P' = dist O Q' ∧ dist O Q' = dist O R' ∧
    (P'.1 + Q.1) / 2 = P'.1 ∧ (P'.2 + Q.2) / 2 = P'.2 ∧
    (Q'.1 + R.1) / 2 = Q'.1 ∧ (Q'.2 + R.2) / 2 = Q'.2 ∧
    (R'.1 + P.1) / 2 = R'.1 ∧ (R'.2 + P.2) / 2 = R'.2

-- Define the area of a hexagon
def HexagonArea (P Q' R Q' P R' : ℝ × ℝ) : ℝ :=
  sorry  -- Actual calculation of area would go here

-- The main theorem
theorem equilateral_triangle_hexagon_area 
  (P Q R P' Q' R' : ℝ × ℝ) :
  Triangle P Q R →
  Perimeter P Q R = 42 →
  CircumcirclePerpBisectors P Q R P' Q' R' →
  HexagonArea P Q' R Q' P R' = 49 * Real.sqrt 3 :=
sorry

end NUMINAMATH_CALUDE_equilateral_triangle_hexagon_area_l3701_370123


namespace NUMINAMATH_CALUDE_sarah_vacation_reading_l3701_370166

/-- Given Sarah's reading speed, book characteristics, and available reading time, prove she can read 6 books. -/
theorem sarah_vacation_reading 
  (reading_speed : ℕ) 
  (words_per_page : ℕ) 
  (pages_per_book : ℕ) 
  (reading_hours : ℕ) 
  (h1 : reading_speed = 40)
  (h2 : words_per_page = 100)
  (h3 : pages_per_book = 80)
  (h4 : reading_hours = 20) : 
  (reading_hours * 60) / ((words_per_page * pages_per_book) / reading_speed) = 6 := by
  sorry

#check sarah_vacation_reading

end NUMINAMATH_CALUDE_sarah_vacation_reading_l3701_370166


namespace NUMINAMATH_CALUDE_vector_parallel_perpendicular_l3701_370197

/-- Two vectors are parallel if their corresponding components are proportional -/
def IsParallel (a b : ℝ × ℝ) : Prop :=
  ∃ k : ℝ, a.1 = k * b.1 ∧ a.2 = k * b.2

/-- Two vectors are perpendicular if their dot product is zero -/
def IsPerpendicular (a b : ℝ × ℝ) : Prop :=
  a.1 * b.1 + a.2 * b.2 = 0

/-- Given vectors a and b, prove the values of m for parallel and perpendicular cases -/
theorem vector_parallel_perpendicular (m : ℝ) :
  let a : ℝ × ℝ := (m, 1)
  let b : ℝ × ℝ := (1, 2)
  (IsParallel a b → m = 1/2) ∧
  (IsPerpendicular a b → m = -2) := by
  sorry

end NUMINAMATH_CALUDE_vector_parallel_perpendicular_l3701_370197


namespace NUMINAMATH_CALUDE_intersection_distance_l3701_370122

-- Define the ellipse E
def ellipse (x y : ℝ) : Prop :=
  x^2 / 16 + y^2 / 12 = 1

-- Define the parabola C
def parabola (x y : ℝ) : Prop :=
  y^2 = 8 * x

-- Define the directrix of C
def directrix (x : ℝ) : Prop :=
  x = -2

-- Define the intersection points A and B
def A : ℝ × ℝ := (-2, 3)
def B : ℝ × ℝ := (-2, -3)

-- State the theorem
theorem intersection_distance :
  ellipse A.1 A.2 ∧
  ellipse B.1 B.2 ∧
  directrix A.1 ∧
  directrix B.1 ∧
  (∃ (x : ℝ), x > 0 ∧ ellipse x 0 ∧ parabola x 0) →
  Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) = 6 :=
sorry

end NUMINAMATH_CALUDE_intersection_distance_l3701_370122


namespace NUMINAMATH_CALUDE_complex_equation_solution_l3701_370111

theorem complex_equation_solution (x y : ℝ) :
  (x : ℂ) - 3 * I = (8 * x - y : ℂ) * I → x = 0 ∧ y = 3 := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l3701_370111


namespace NUMINAMATH_CALUDE_final_price_is_135_l3701_370178

/-- The original price of the dress -/
def original_price : ℝ := 250

/-- The first discount rate -/
def first_discount_rate : ℝ := 0.4

/-- The additional holiday discount rate -/
def holiday_discount_rate : ℝ := 0.1

/-- The price after the first discount -/
def price_after_first_discount : ℝ := original_price * (1 - first_discount_rate)

/-- The final price after both discounts -/
def final_price : ℝ := price_after_first_discount * (1 - holiday_discount_rate)

/-- Theorem stating that the final price is $135 -/
theorem final_price_is_135 : final_price = 135 := by sorry

end NUMINAMATH_CALUDE_final_price_is_135_l3701_370178


namespace NUMINAMATH_CALUDE_max_b_value_l3701_370157

theorem max_b_value (y : ℤ) (b : ℕ+) (h : y^2 + b * y = -21) :
  b ≤ 22 ∧ ∃ y : ℤ, y^2 + 22 * y = -21 :=
sorry

end NUMINAMATH_CALUDE_max_b_value_l3701_370157


namespace NUMINAMATH_CALUDE_sector_radius_l3701_370193

/-- Given a circular sector with central angle 5π/7 and perimeter 5π + 14, its radius is 7. -/
theorem sector_radius (r : ℝ) : 
  (5 / 7 : ℝ) * π * r + 2 * r = 5 * π + 14 → r = 7 := by
  sorry

end NUMINAMATH_CALUDE_sector_radius_l3701_370193


namespace NUMINAMATH_CALUDE_isosceles_triangle_perimeter_l3701_370134

/-- An isosceles triangle with side lengths 4 and 8 has a perimeter of 20. -/
theorem isosceles_triangle_perimeter : ∀ (a b c : ℝ),
  a = 8 → b = 8 → c = 4 →
  (a = b ∨ a = c ∨ b = c) →  -- isosceles condition
  a + b + c = 20 := by sorry

end NUMINAMATH_CALUDE_isosceles_triangle_perimeter_l3701_370134


namespace NUMINAMATH_CALUDE_league_games_l3701_370155

theorem league_games (n : ℕ) (h : n = 14) : (n * (n - 1)) / 2 = 91 := by
  sorry

end NUMINAMATH_CALUDE_league_games_l3701_370155


namespace NUMINAMATH_CALUDE_canoe_downstream_speed_l3701_370176

/-- Given a canoe that rows upstream at 6 km/hr and a stream with a speed of 2 km/hr,
    this theorem proves that the speed of the canoe when rowing downstream is 10 km/hr. -/
theorem canoe_downstream_speed
  (upstream_speed : ℝ)
  (stream_speed : ℝ)
  (h1 : upstream_speed = 6)
  (h2 : stream_speed = 2) :
  upstream_speed + 2 * stream_speed = 10 :=
by sorry

end NUMINAMATH_CALUDE_canoe_downstream_speed_l3701_370176


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_condition_l3701_370100

theorem sufficient_not_necessary_condition (a : ℝ) :
  (∀ a, a > 1 → a^2 > 1) ∧
  (∃ a, a^2 > 1 ∧ ¬(a > 1)) :=
by sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_condition_l3701_370100


namespace NUMINAMATH_CALUDE_flower_perimeter_l3701_370190

/-- The perimeter of a flower-like figure created by a regular hexagon inscribed in a circle -/
theorem flower_perimeter (c : ℝ) (h : c = 16) : 
  let hexagon_arc := c / 6
  let petal_arc := 2 * hexagon_arc
  let num_petals := 6
  num_petals * petal_arc = 32 := by
  sorry

end NUMINAMATH_CALUDE_flower_perimeter_l3701_370190


namespace NUMINAMATH_CALUDE_isosceles_triangle_vertex_angle_l3701_370101

-- Define an isosceles triangle
structure IsoscelesTriangle where
  -- We only need to define two angles, as the third is determined by these two
  angle1 : ℝ
  angle2 : ℝ
  -- Condition: sum of angles is 180°
  sum_angles : angle1 + angle2 + (180 - angle1 - angle2) = 180
  -- Condition: at least two angles are equal (isosceles property)
  isosceles : angle1 = angle2 ∨ angle1 = (180 - angle1 - angle2) ∨ angle2 = (180 - angle1 - angle2)

-- Theorem statement
theorem isosceles_triangle_vertex_angle (t : IsoscelesTriangle) (h : t.angle1 = 70 ∨ t.angle2 = 70 ∨ (180 - t.angle1 - t.angle2) = 70) :
  t.angle1 = 70 ∨ t.angle2 = 70 ∨ (180 - t.angle1 - t.angle2) = 70 ∨
  t.angle1 = 40 ∨ t.angle2 = 40 ∨ (180 - t.angle1 - t.angle2) = 40 :=
by sorry

end NUMINAMATH_CALUDE_isosceles_triangle_vertex_angle_l3701_370101


namespace NUMINAMATH_CALUDE_train_platform_crossing_time_l3701_370150

/-- Given a train and platform with known dimensions, calculate the time to cross the platform. -/
theorem train_platform_crossing_time 
  (train_length : ℝ) 
  (platform_length : ℝ) 
  (time_to_cross_pole : ℝ) 
  (train_length_positive : 0 < train_length)
  (platform_length_positive : 0 < platform_length)
  (time_to_cross_pole_positive : 0 < time_to_cross_pole) :
  (train_length + platform_length) / (train_length / time_to_cross_pole) = 
    (train_length + platform_length) * time_to_cross_pole / train_length := by
  sorry

end NUMINAMATH_CALUDE_train_platform_crossing_time_l3701_370150


namespace NUMINAMATH_CALUDE_lion_king_cost_l3701_370113

theorem lion_king_cost (
  lion_king_earnings : ℝ)
  (star_wars_cost : ℝ)
  (star_wars_earnings : ℝ)
  (h1 : lion_king_earnings = 200)
  (h2 : star_wars_cost = 25)
  (h3 : star_wars_earnings = 405)
  (h4 : lion_king_earnings - (lion_king_earnings - (star_wars_earnings - star_wars_cost) / 2) = 10) :
  ∃ (lion_king_cost : ℝ), lion_king_cost = 10 := by
  sorry

end NUMINAMATH_CALUDE_lion_king_cost_l3701_370113


namespace NUMINAMATH_CALUDE_simple_interest_problem_l3701_370133

/-- Given a principal P at simple interest for 6 years, if increasing the
    interest rate by 4% results in $144 more interest, then P = $600. -/
theorem simple_interest_problem (P : ℝ) (R : ℝ) : 
  (P * (R + 4) * 6) / 100 - (P * R * 6) / 100 = 144 → P = 600 := by
sorry

end NUMINAMATH_CALUDE_simple_interest_problem_l3701_370133


namespace NUMINAMATH_CALUDE_rectangular_field_area_l3701_370137

/-- The area of a rectangular field with one side 16 m and a diagonal of 17 m is 16 * √33 square meters. -/
theorem rectangular_field_area (a b : ℝ) (h1 : a = 16) (h2 : a^2 + b^2 = 17^2) :
  a * b = 16 * Real.sqrt 33 := by
  sorry

end NUMINAMATH_CALUDE_rectangular_field_area_l3701_370137


namespace NUMINAMATH_CALUDE_three_digit_numbers_satisfying_condition_l3701_370119

def is_valid_number (n : ℕ) : Prop :=
  100 ≤ n ∧ n < 1000

def satisfies_condition (n : ℕ) : Prop :=
  let a := n / 100
  let b := (n / 10) % 10
  let c := n % 10
  n = 2 * ((10 * a + b) + (10 * b + c) + (10 * a + c))

def solution_set : Set ℕ := {134, 144, 150, 288, 294}

theorem three_digit_numbers_satisfying_condition :
  ∀ n : ℕ, is_valid_number n ∧ satisfies_condition n ↔ n ∈ solution_set :=
sorry

end NUMINAMATH_CALUDE_three_digit_numbers_satisfying_condition_l3701_370119


namespace NUMINAMATH_CALUDE_equation_solution_l3701_370165

theorem equation_solution : 
  ∃! x : ℝ, (1 / (x - 3) = 3 / (x + 1)) ∧ x = 5 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l3701_370165


namespace NUMINAMATH_CALUDE_loisa_savings_l3701_370164

/-- The amount Loisa saves by buying the tablet in cash instead of on installment -/
theorem loisa_savings (cash_price : ℕ) (down_payment : ℕ) 
  (first_four_months : ℕ) (next_four_months : ℕ) (last_four_months : ℕ) : 
  cash_price = 450 →
  down_payment = 100 →
  first_four_months = 40 →
  next_four_months = 35 →
  last_four_months = 30 →
  down_payment + 4 * first_four_months + 4 * next_four_months + 4 * last_four_months - cash_price = 70 := by
  sorry

#check loisa_savings

end NUMINAMATH_CALUDE_loisa_savings_l3701_370164


namespace NUMINAMATH_CALUDE_chord_length_squared_l3701_370168

/-- Two circles with radii 10 and 7, centers 15 units apart, intersecting at P.
    A line through P creates equal chords QP and PR. -/
structure IntersectingCircles where
  r₁ : ℝ
  r₂ : ℝ
  center_distance : ℝ
  chord_length : ℝ
  h₁ : r₁ = 10
  h₂ : r₂ = 7
  h₃ : center_distance = 15
  h₄ : chord_length > 0

/-- The square of the length of chord QP in the given configuration is 289. -/
theorem chord_length_squared (c : IntersectingCircles) : c.chord_length ^ 2 = 289 := by
  sorry

end NUMINAMATH_CALUDE_chord_length_squared_l3701_370168


namespace NUMINAMATH_CALUDE_henry_finishes_before_zoe_l3701_370154

/-- Represents the race parameters and results -/
structure RaceData where
  distance : ℕ  -- race distance in miles
  zoe_pace : ℕ  -- Zoe's pace in minutes per mile
  henry_pace : ℕ  -- Henry's pace in minutes per mile

/-- Calculates the time difference between Zoe and Henry finishing the race -/
def timeDifference (race : RaceData) : Int :=
  race.zoe_pace * race.distance - race.henry_pace * race.distance

/-- Theorem stating that Henry finishes 24 minutes before Zoe in the given race conditions -/
theorem henry_finishes_before_zoe (race : RaceData) 
  (h1 : race.distance = 12)
  (h2 : race.zoe_pace = 9)
  (h3 : race.henry_pace = 7) : 
  timeDifference race = 24 := by
  sorry

end NUMINAMATH_CALUDE_henry_finishes_before_zoe_l3701_370154


namespace NUMINAMATH_CALUDE_derivative_implies_limit_l3701_370115

theorem derivative_implies_limit (f : ℝ → ℝ) (x₀ a : ℝ) (h : HasDerivAt f a x₀) :
  ∀ ε > 0, ∃ δ > 0, ∀ Δx, 0 < |Δx| → |Δx| < δ →
    |(f (x₀ + Δx) - f (x₀ - Δx)) / Δx - 2*a| < ε :=
by sorry

end NUMINAMATH_CALUDE_derivative_implies_limit_l3701_370115


namespace NUMINAMATH_CALUDE_orange_caterpillar_length_l3701_370106

theorem orange_caterpillar_length 
  (green_length : ℝ) 
  (length_difference : ℝ) 
  (h1 : green_length = 3) 
  (h2 : length_difference = 1.83) 
  (h3 : green_length = length_difference + orange_length) : 
  orange_length = 1.17 := by
  sorry

end NUMINAMATH_CALUDE_orange_caterpillar_length_l3701_370106


namespace NUMINAMATH_CALUDE_haj_daily_cost_l3701_370151

/-- The daily operation cost for Mr. Haj's grocery store -/
def daily_cost : ℝ → Prop := λ T => 
  -- 2/5 of total cost is for salary
  let salary := (2/5) * T
  -- Remaining after salary
  let remaining_after_salary := T - salary
  -- 1/4 of remaining after salary is for delivery
  let delivery := (1/4) * remaining_after_salary
  -- Amount for orders
  let orders := 1800
  -- Total cost equals sum of salary, delivery, and orders
  T = salary + delivery + orders

/-- Theorem stating the daily operation cost for Mr. Haj's grocery store -/
theorem haj_daily_cost : ∃ T : ℝ, daily_cost T ∧ T = 8000 := by
  sorry

end NUMINAMATH_CALUDE_haj_daily_cost_l3701_370151


namespace NUMINAMATH_CALUDE_remainder_233_divided_by_d_l3701_370183

theorem remainder_233_divided_by_d (a b c d : ℕ) : 
  1 < a → a < b → b < c → a + c = 13 → d = a * b * c → 
  233 % d = 53 := by sorry

end NUMINAMATH_CALUDE_remainder_233_divided_by_d_l3701_370183


namespace NUMINAMATH_CALUDE_sum_of_reciprocal_squares_l3701_370158

theorem sum_of_reciprocal_squares (a b c : ℝ) : 
  a^3 - 12*a^2 + 14*a + 3 = 0 →
  b^3 - 12*b^2 + 14*b + 3 = 0 →
  c^3 - 12*c^2 + 14*c + 3 = 0 →
  a ≠ b → b ≠ c → a ≠ c →
  1/a^2 + 1/b^2 + 1/c^2 = 268/9 := by
sorry

end NUMINAMATH_CALUDE_sum_of_reciprocal_squares_l3701_370158


namespace NUMINAMATH_CALUDE_complex_sum_powers_of_i_l3701_370130

theorem complex_sum_powers_of_i : ∃ (i : ℂ), i^2 = -1 ∧ i + i^2 + i^3 + i^4 = 0 := by
  sorry

end NUMINAMATH_CALUDE_complex_sum_powers_of_i_l3701_370130


namespace NUMINAMATH_CALUDE_zero_in_interval_l3701_370192

def f (x : ℝ) : ℝ := x^5 + 8*x^3 - 1

theorem zero_in_interval :
  (f 0 < 0) → (f 0.5 > 0) → ∃ x, x ∈ Set.Ioo 0 0.5 ∧ f x = 0 :=
by
  sorry

end NUMINAMATH_CALUDE_zero_in_interval_l3701_370192


namespace NUMINAMATH_CALUDE_number_remainder_l3701_370129

theorem number_remainder (A : ℤ) (h : 9 * A + 1 = 10 * A - 100) : A % 7 = 3 := by
  sorry

end NUMINAMATH_CALUDE_number_remainder_l3701_370129


namespace NUMINAMATH_CALUDE_sum_of_digits_of_10_pow_97_minus_97_l3701_370186

/-- The sum of digits of a natural number -/
def sum_of_digits (n : ℕ) : ℕ := sorry

/-- The main theorem stating that the sum of digits of 10^97 - 97 is 858 -/
theorem sum_of_digits_of_10_pow_97_minus_97 : sum_of_digits (10^97 - 97) = 858 := by sorry

end NUMINAMATH_CALUDE_sum_of_digits_of_10_pow_97_minus_97_l3701_370186


namespace NUMINAMATH_CALUDE_expression_value_l3701_370187

theorem expression_value (x y z : ℤ) (hx : x = 3) (hy : y = 2) (hz : z = 4) :
  3 * x - 4 * y + 5 * z = 21 := by
  sorry

end NUMINAMATH_CALUDE_expression_value_l3701_370187


namespace NUMINAMATH_CALUDE_max_min_sum_of_f_l3701_370112

def f (g : ℝ → ℝ) (x : ℝ) : ℝ := g x + 2

theorem max_min_sum_of_f (g : ℝ → ℝ) (h : ∀ x, g (-x) = -g x) :
  let f := f g
  let M := ⨆ (x : ℝ) (hx : x ∈ Set.Icc (-3) 3), f x
  let N := ⨅ (x : ℝ) (hx : x ∈ Set.Icc (-3) 3), f x
  M + N = 4 := by
  sorry

end NUMINAMATH_CALUDE_max_min_sum_of_f_l3701_370112


namespace NUMINAMATH_CALUDE_blacksmith_iron_calculation_l3701_370140

/-- The amount of iron needed for one horseshoe in kilograms -/
def iron_per_horseshoe : ℕ := 2

/-- The number of horseshoes needed for one horse -/
def horseshoes_per_horse : ℕ := 4

/-- The number of farms -/
def num_farms : ℕ := 2

/-- The number of horses in each farm -/
def horses_per_farm : ℕ := 2

/-- The number of stables -/
def num_stables : ℕ := 2

/-- The number of horses in each stable -/
def horses_per_stable : ℕ := 5

/-- The number of horses at the riding school -/
def riding_school_horses : ℕ := 36

/-- The total amount of iron the blacksmith had initially in kilograms -/
def initial_iron : ℕ := 400

theorem blacksmith_iron_calculation : 
  initial_iron = 
    (num_farms * horses_per_farm + num_stables * horses_per_stable + riding_school_horses) * 
    horseshoes_per_horse * iron_per_horseshoe :=
by sorry

end NUMINAMATH_CALUDE_blacksmith_iron_calculation_l3701_370140


namespace NUMINAMATH_CALUDE_unique_function_solution_l3701_370156

/-- A function from non-negative reals to non-negative reals -/
def NonnegFunction := {f : ℝ → ℝ // ∀ x, 0 ≤ x → 0 ≤ f x}

theorem unique_function_solution (f : NonnegFunction) 
  (h : ∀ x : ℝ, 0 ≤ x → f.val (f.val x) + f.val x = 12 * x) :
  ∀ x : ℝ, 0 ≤ x → f.val x = 3 * x := by
  sorry

end NUMINAMATH_CALUDE_unique_function_solution_l3701_370156


namespace NUMINAMATH_CALUDE_cube_decomposition_91_l3701_370102

/-- Decomposition of a cube into consecutive odd numbers -/
def cube_decomposition (n : ℕ+) : List ℕ :=
  sorry

/-- The smallest number in the decomposition of m³ -/
def smallest_in_decomposition (m : ℕ+) : ℕ :=
  sorry

/-- Theorem: If the smallest number in the decomposition of m³ is 91, then m = 10 -/
theorem cube_decomposition_91 (m : ℕ+) :
  smallest_in_decomposition m = 91 → m = 10 := by
  sorry

end NUMINAMATH_CALUDE_cube_decomposition_91_l3701_370102


namespace NUMINAMATH_CALUDE_median_intersection_property_l3701_370171

-- Define the triangle ABC
structure Triangle :=
  (A B C : ℝ × ℝ)

-- Define a point
def Point := ℝ × ℝ

-- Define the median intersection point
def medianIntersection (t : Triangle) : Point :=
  sorry

-- Define points M, N, P on the sides of the triangle
def dividePoint (A B : Point) (p q : ℝ) : Point :=
  sorry

-- Theorem statement
theorem median_intersection_property (ABC : Triangle) (p q : ℝ) :
  let O := medianIntersection ABC
  let M := dividePoint ABC.A ABC.B p q
  let N := dividePoint ABC.B ABC.C p q
  let P := dividePoint ABC.C ABC.A p q
  let MNP : Triangle := ⟨M, N, P⟩
  let ANBPCMTriangle : Triangle := 
    ⟨ABC.A, ABC.B, ABC.C⟩  -- This is a placeholder, as we don't have a way to define the intersection points
  (O = medianIntersection MNP) ∧ 
  (O = medianIntersection ANBPCMTriangle) :=
sorry

end NUMINAMATH_CALUDE_median_intersection_property_l3701_370171


namespace NUMINAMATH_CALUDE_billboard_dimensions_l3701_370131

/-- Given a rectangular photograph and a billboard, prove the billboard's dimensions -/
theorem billboard_dimensions 
  (photo_width : ℝ) 
  (photo_length : ℝ) 
  (billboard_area : ℝ) 
  (h1 : photo_width = 30) 
  (h2 : photo_length = 40) 
  (h3 : billboard_area = 48) : 
  ∃ (billboard_width billboard_length : ℝ), 
    billboard_width = 6 ∧ 
    billboard_length = 8 ∧ 
    billboard_width * billboard_length = billboard_area :=
by sorry

end NUMINAMATH_CALUDE_billboard_dimensions_l3701_370131


namespace NUMINAMATH_CALUDE_negation_equivalence_l3701_370163

theorem negation_equivalence :
  (¬ ∃ x : ℝ, x < 1 ∧ x^2 + 2*x + 1 ≤ 0) ↔ (∀ x : ℝ, x < 1 → x^2 + 2*x + 1 > 0) := by
  sorry

end NUMINAMATH_CALUDE_negation_equivalence_l3701_370163


namespace NUMINAMATH_CALUDE_tangerine_problem_l3701_370153

/-- The number of tangerines initially in the basket -/
def initial_tangerines : ℕ := 24

/-- The number of tangerines Eunji ate -/
def eaten_tangerines : ℕ := 9

/-- The number of tangerines Eunji's mother added -/
def added_tangerines : ℕ := 5

/-- The final number of tangerines in the basket -/
def final_tangerines : ℕ := 20

theorem tangerine_problem :
  initial_tangerines - eaten_tangerines + added_tangerines = final_tangerines :=
by sorry

end NUMINAMATH_CALUDE_tangerine_problem_l3701_370153


namespace NUMINAMATH_CALUDE_polygon_area_bound_l3701_370199

/-- A polygon in 2D space --/
structure Polygon where
  -- We don't need to define the exact structure of the polygon,
  -- just its projections and area
  proj_ox : ℝ
  proj_bisector13 : ℝ
  proj_oy : ℝ
  proj_bisector24 : ℝ
  area : ℝ

/-- Theorem stating that the area of a polygon with given projections is bounded --/
theorem polygon_area_bound (p : Polygon)
  (h1 : p.proj_ox = 4)
  (h2 : p.proj_bisector13 = 3 * Real.sqrt 2)
  (h3 : p.proj_oy = 5)
  (h4 : p.proj_bisector24 = 4 * Real.sqrt 2)
  : p.area ≤ 17.5 := by
  sorry

end NUMINAMATH_CALUDE_polygon_area_bound_l3701_370199


namespace NUMINAMATH_CALUDE_intersection_equality_l3701_370138

-- Define the sets M and N
def M : Set ℝ := {x | x > 1}
def N : Set ℝ := {x | x < 5}

-- State the theorem
theorem intersection_equality : M ∩ N = Set.Ioo 1 5 := by sorry

end NUMINAMATH_CALUDE_intersection_equality_l3701_370138


namespace NUMINAMATH_CALUDE_rotated_square_height_l3701_370188

theorem rotated_square_height :
  let square_side : ℝ := 1
  let rotation_angle : ℝ := 30 * π / 180
  let initial_center_height : ℝ := square_side / 2
  let diagonal : ℝ := square_side * Real.sqrt 2
  let rotated_height : ℝ := diagonal * Real.sin rotation_angle
  initial_center_height + rotated_height = (1 + Real.sqrt 2) / 2 := by
sorry

end NUMINAMATH_CALUDE_rotated_square_height_l3701_370188


namespace NUMINAMATH_CALUDE_student_distribution_theorem_l3701_370177

/-- Represents the number of classes available --/
def num_classes : ℕ := 3

/-- Represents the number of students requesting to change classes --/
def num_students : ℕ := 4

/-- Represents the maximum number of additional students a class can accept --/
def max_per_class : ℕ := 2

/-- Calculates the number of ways to distribute students among classes --/
def distribution_ways : ℕ := 54

/-- Theorem stating the number of ways to distribute students --/
theorem student_distribution_theorem :
  (num_classes = 3) →
  (num_students = 4) →
  (max_per_class = 2) →
  distribution_ways = 54 :=
by
  sorry

#check student_distribution_theorem

end NUMINAMATH_CALUDE_student_distribution_theorem_l3701_370177


namespace NUMINAMATH_CALUDE_exists_special_function_l3701_370162

/-- A function satisfying specific properties --/
def special_function (f : ℝ → ℝ) : Prop :=
  (∀ x, f x = f (-x)) ∧ 
  (f 0 = 1) ∧
  (∀ x, f (x + 3) = -f (-(x + 3))) ∧
  (f (-9) = 0) ∧
  (f 18 = -1) ∧
  (f 24 = 1)

/-- Theorem stating the existence of a function with the specified properties --/
theorem exists_special_function : ∃ f : ℝ → ℝ, special_function f := by
  sorry

end NUMINAMATH_CALUDE_exists_special_function_l3701_370162


namespace NUMINAMATH_CALUDE_cubic_polynomials_with_rational_roots_l3701_370152

/-- A cubic polynomial with rational coefficients -/
structure CubicPolynomial where
  a : ℚ
  b : ℚ
  c : ℚ

/-- The set of cubic polynomials with all rational roots -/
def CubicPolynomialsWithRationalRoots : Set CubicPolynomial :=
  {p : CubicPolynomial | ∃ (r₁ r₂ r₃ : ℚ),
    ∀ x, x^3 + p.a * x^2 + p.b * x + p.c = (x - r₁) * (x - r₂) * (x - r₃)}

/-- The two specific polynomials from the solution -/
def f₁ : CubicPolynomial := ⟨1, -2, 0⟩
def f₂ : CubicPolynomial := ⟨1, -1, -1⟩

/-- The main theorem stating that f₁ and f₂ are the only cubic polynomials with all rational roots -/
theorem cubic_polynomials_with_rational_roots :
  CubicPolynomialsWithRationalRoots = {f₁, f₂} :=
by sorry

end NUMINAMATH_CALUDE_cubic_polynomials_with_rational_roots_l3701_370152


namespace NUMINAMATH_CALUDE_positive_number_relationship_l3701_370191

theorem positive_number_relationship (n : ℕ) (a b : ℝ) 
  (h_n : n ≥ 2) 
  (h_a_pos : a > 0) 
  (h_b_pos : b > 0) 
  (h_a_eq : a^n = a + 1) 
  (h_b_eq : b^(2*n) = b + 3*a) : 
  a > b ∧ b > 1 := by
sorry

end NUMINAMATH_CALUDE_positive_number_relationship_l3701_370191


namespace NUMINAMATH_CALUDE_sector_central_angle_l3701_370146

/-- Given a sector with arc length 2m and radius 2cm, prove that its central angle is 100 radians. -/
theorem sector_central_angle (arc_length : ℝ) (radius : ℝ) (h1 : arc_length = 2) (h2 : radius = 0.02) :
  arc_length = radius * 100 := by
sorry

end NUMINAMATH_CALUDE_sector_central_angle_l3701_370146


namespace NUMINAMATH_CALUDE_sin_300_degrees_l3701_370103

theorem sin_300_degrees : Real.sin (300 * π / 180) = -Real.sqrt 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_sin_300_degrees_l3701_370103


namespace NUMINAMATH_CALUDE_cube_volume_doubled_edges_l3701_370108

theorem cube_volume_doubled_edges (a : ℝ) (h : a > 0) :
  (2 * a)^3 = 8 * a^3 := by sorry

end NUMINAMATH_CALUDE_cube_volume_doubled_edges_l3701_370108


namespace NUMINAMATH_CALUDE_tangent_and_chord_l3701_370160

noncomputable section

-- Define the circle M
def circle_M (x y : ℝ) : Prop := x^2 + (y - 4)^2 = 1

-- Define the line l
def line_l (x y : ℝ) : Prop := 2*x - y = 0

-- Define the point P
structure Point_P where
  x : ℝ
  y : ℝ
  on_line : line_l x y

-- Define the tangent property
def is_tangent (x y : ℝ) : Prop := ∃ (t : ℝ), circle_M (x + t) (y + 2*t)

-- Main theorem
theorem tangent_and_chord :
  ∃ (P : Point_P),
    (∃ (A B : ℝ × ℝ),
      is_tangent (A.1 - P.x) (A.2 - P.y) ∧
      is_tangent (B.1 - P.x) (B.2 - P.y) ∧
      (A.1 - P.x) * (B.1 - P.x) + (A.2 - P.y) * (B.2 - P.y) = 
        ((A.1 - P.x)^2 + (A.2 - P.y)^2)^(1/2) * ((B.1 - P.x)^2 + (B.2 - P.y)^2)^(1/2) / 2) ∧
    ((P.x = 2 ∧ P.y = 4) ∨ (P.x = 6/5 ∧ P.y = 12/5)) ∧
    (∃ (C : ℝ × ℝ),
      (C.1 - P.x)^2 + (C.2 - P.y)^2 = (0 - P.x)^2 + (4 - P.y)^2 ∧
      ∃ (D : ℝ × ℝ),
        circle_M D.1 D.2 ∧
        (D.1 - C.1) * (1/2 - C.1) + (D.2 - C.2) * (15/4 - C.2) = 0) :=
sorry

end NUMINAMATH_CALUDE_tangent_and_chord_l3701_370160


namespace NUMINAMATH_CALUDE_min_sum_mn_l3701_370127

theorem min_sum_mn (m n : ℕ+) (h : m.val * n.val - 2 * m.val - 3 * n.val - 20 = 0) :
  ∃ (p q : ℕ+), p.val * q.val - 2 * p.val - 3 * q.val - 20 = 0 ∧ 
  p.val + q.val = 20 ∧ 
  ∀ (x y : ℕ+), x.val * y.val - 2 * x.val - 3 * y.val - 20 = 0 → x.val + y.val ≥ 20 :=
by sorry

end NUMINAMATH_CALUDE_min_sum_mn_l3701_370127


namespace NUMINAMATH_CALUDE_union_A_complement_B_l3701_370125

open Set

-- Define the universal set U
def U : Set ℕ := {1, 2, 3, 4, 5}

-- Define set A
def A : Set ℕ := {1, 3}

-- Define set B
def B : Set ℕ := {2, 3, 4}

-- Theorem to prove
theorem union_A_complement_B : A ∪ (U \ B) = {1, 3, 5} := by
  sorry

end NUMINAMATH_CALUDE_union_A_complement_B_l3701_370125


namespace NUMINAMATH_CALUDE_simplify_expression_l3701_370142

theorem simplify_expression :
  (3/2) * Real.sqrt 5 - (1/3) * Real.sqrt 6 + (1/2) * (-Real.sqrt 5 + 2 * Real.sqrt 6) =
  Real.sqrt 5 + (2/3) * Real.sqrt 6 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l3701_370142


namespace NUMINAMATH_CALUDE_ellipse_eccentricity_l3701_370175

theorem ellipse_eccentricity (m : ℝ) : 
  (∀ x y : ℝ, x^2/m + y^2/4 = 1) →  -- Equation of the ellipse
  (∃ a b c : ℝ, a > b ∧ b > 0 ∧ c^2 = a^2 - b^2 ∧ 2*c = 2) →  -- Eccentricity is 2
  (m = 3 ∨ m = 5) :=
by sorry

end NUMINAMATH_CALUDE_ellipse_eccentricity_l3701_370175


namespace NUMINAMATH_CALUDE_julia_played_with_17_kids_on_monday_l3701_370159

/-- The number of kids Julia played with on Monday -/
def monday_kids : ℕ := sorry

/-- The number of kids Julia played with on Tuesday -/
def tuesday_kids : ℕ := 15

/-- The number of kids Julia played with on Wednesday -/
def wednesday_kids : ℕ := 2

/-- The total number of kids Julia played with -/
def total_kids : ℕ := 34

/-- Theorem stating that Julia played with 17 kids on Monday -/
theorem julia_played_with_17_kids_on_monday :
  monday_kids = 17 :=
by
  sorry

end NUMINAMATH_CALUDE_julia_played_with_17_kids_on_monday_l3701_370159


namespace NUMINAMATH_CALUDE_a2_value_l3701_370148

def arithmetic_sequence (a : ℕ → ℝ) (d : ℝ) :=
  ∀ n, a (n + 1) = a n + d

def geometric_sequence (a b c : ℝ) :=
  b^2 = a * c

theorem a2_value (a : ℕ → ℝ) :
  arithmetic_sequence a 2 →
  geometric_sequence (a 1) (a 3) (a 4) →
  a 2 = -6 := by
  sorry

end NUMINAMATH_CALUDE_a2_value_l3701_370148


namespace NUMINAMATH_CALUDE_additional_wolves_in_pack_l3701_370124

/-- The number of additional wolves in a pack, given hunting conditions -/
def additional_wolves (hunting_wolves : ℕ) (meat_per_day : ℕ) (hunting_period : ℕ) 
                      (meat_per_deer : ℕ) (deer_per_hunter : ℕ) : ℕ :=
  let total_meat := hunting_wolves * deer_per_hunter * meat_per_deer
  let wolves_fed := total_meat / (meat_per_day * hunting_period)
  wolves_fed - hunting_wolves

theorem additional_wolves_in_pack : 
  additional_wolves 4 8 5 200 1 = 16 := by
  sorry

end NUMINAMATH_CALUDE_additional_wolves_in_pack_l3701_370124


namespace NUMINAMATH_CALUDE_lower_price_option2_l3701_370147

def initial_value : ℝ := 12000

def apply_discount (price : ℝ) (discount : ℝ) : ℝ :=
  price * (1 - discount)

def option1_final_price : ℝ :=
  apply_discount (apply_discount (apply_discount initial_value 0.3) 0.1) 0.05

def option2_final_price : ℝ :=
  apply_discount (apply_discount (apply_discount initial_value 0.3) 0.05) 0.15

theorem lower_price_option2 :
  option2_final_price < option1_final_price ∧ option2_final_price = 6783 :=
by sorry

end NUMINAMATH_CALUDE_lower_price_option2_l3701_370147


namespace NUMINAMATH_CALUDE_denominator_conversion_l3701_370149

theorem denominator_conversion (x : ℝ) : 
  (x + 1) / 0.4 - (0.2 * x - 1) / 0.7 = 1 ↔ (10 * x + 10) / 4 - (2 * x - 10) / 7 = 1 := by
  sorry

end NUMINAMATH_CALUDE_denominator_conversion_l3701_370149


namespace NUMINAMATH_CALUDE_project_selection_probability_l3701_370170

theorem project_selection_probability :
  let n_employees : ℕ := 4
  let n_projects : ℕ := 4
  let total_outcomes : ℕ := n_projects ^ n_employees
  let favorable_outcomes : ℕ := Nat.choose n_projects 2 * (Nat.factorial n_employees / Nat.factorial 1)
  (favorable_outcomes : ℚ) / total_outcomes = 9 / 16 :=
by sorry

end NUMINAMATH_CALUDE_project_selection_probability_l3701_370170


namespace NUMINAMATH_CALUDE_certain_number_of_seconds_l3701_370144

/-- Converts minutes to seconds -/
def minutes_to_seconds (m : ℕ) : ℕ := m * 60

/-- The proportion given in the problem -/
def proportion (x : ℕ) : Prop :=
  15 / x = 30 / minutes_to_seconds 10

theorem certain_number_of_seconds : ∃ x : ℕ, proportion x ∧ x = 300 :=
  sorry

end NUMINAMATH_CALUDE_certain_number_of_seconds_l3701_370144


namespace NUMINAMATH_CALUDE_fraction_to_decimal_l3701_370135

theorem fraction_to_decimal (n d : ℕ) (h : d ≠ 0) :
  (n : ℚ) / d = 175 / 1000 :=
by
  sorry

end NUMINAMATH_CALUDE_fraction_to_decimal_l3701_370135


namespace NUMINAMATH_CALUDE_inner_perimeter_le_outer_perimeter_l3701_370189

/-- A convex polygon in a 2D plane -/
structure ConvexPolygon where
  vertices : List (ℝ × ℝ)
  is_convex : sorry -- Axiom stating that the polygon is convex

/-- Defines when one polygon is inside another -/
def is_inside (inner outer : ConvexPolygon) : Prop := sorry

/-- Calculates the perimeter of a convex polygon -/
def perimeter (p : ConvexPolygon) : ℝ := sorry

/-- Theorem: If one convex polygon is inside another, then the perimeter of the inner polygon
    does not exceed the perimeter of the outer polygon -/
theorem inner_perimeter_le_outer_perimeter (inner outer : ConvexPolygon) 
  (h : is_inside inner outer) : perimeter inner ≤ perimeter outer := by
  sorry

end NUMINAMATH_CALUDE_inner_perimeter_le_outer_perimeter_l3701_370189


namespace NUMINAMATH_CALUDE_staircase_has_31_steps_l3701_370145

/-- Represents a staircase with a middle step and specific movement rules -/
structure Staircase where
  total_steps : ℕ
  middle_step : ℕ
  (middle_property : middle_step * 2 - 1 = total_steps)
  (movement_property : middle_step + 7 - 15 = 8)

/-- Theorem stating that a staircase satisfying the given conditions has 31 steps -/
theorem staircase_has_31_steps (s : Staircase) : s.total_steps = 31 := by
  sorry

#check staircase_has_31_steps

end NUMINAMATH_CALUDE_staircase_has_31_steps_l3701_370145


namespace NUMINAMATH_CALUDE_sum_of_solutions_is_zero_l3701_370169

theorem sum_of_solutions_is_zero : 
  ∃ (x₁ x₂ : ℝ), (6 * x₁) / 30 = 8 / x₁ ∧ 
                 (6 * x₂) / 30 = 8 / x₂ ∧ 
                 x₁ + x₂ = 0 ∧
                 ∀ (y : ℝ), (6 * y) / 30 = 8 / y → y = x₁ ∨ y = x₂ := by
  sorry

end NUMINAMATH_CALUDE_sum_of_solutions_is_zero_l3701_370169


namespace NUMINAMATH_CALUDE_rhombus_diagonal_l3701_370136

theorem rhombus_diagonal (A : ℝ) (d : ℝ) : 
  d > 0 →  -- shorter diagonal is positive
  3 * d > 0 →  -- longer diagonal is positive
  A = (1/2) * d * (3*d) →  -- area formula
  40 = 4 * (((d/2)^2 + ((3*d)/2)^2)^(1/2)) →  -- perimeter formula
  d = (1/3) * (10 * A)^(1/2) := by sorry

end NUMINAMATH_CALUDE_rhombus_diagonal_l3701_370136


namespace NUMINAMATH_CALUDE_travel_time_difference_l3701_370196

/-- Proves that the difference in travel time between two cars is 2 hours -/
theorem travel_time_difference (distance : ℝ) (speed_r speed_p : ℝ) : 
  distance = 600 ∧ 
  speed_r = 50 ∧ 
  speed_p = speed_r + 10 →
  distance / speed_r - distance / speed_p = 2 := by
sorry


end NUMINAMATH_CALUDE_travel_time_difference_l3701_370196


namespace NUMINAMATH_CALUDE_trees_planted_by_two_classes_l3701_370114

theorem trees_planted_by_two_classes 
  (trees_A : ℕ) 
  (trees_B : ℕ) 
  (h1 : trees_A = 8) 
  (h2 : trees_B = 7) : 
  trees_A + trees_B = 15 := by
sorry

end NUMINAMATH_CALUDE_trees_planted_by_two_classes_l3701_370114


namespace NUMINAMATH_CALUDE_four_inequalities_l3701_370167

theorem four_inequalities :
  (∃ (x : ℝ), x = Real.sqrt (2 * Real.sqrt (2 * Real.sqrt (2 * Real.sqrt 2))) ∧ x < 2) ∧
  (∃ (y : ℝ), y = Real.sqrt (2 + Real.sqrt (2 + Real.sqrt (2 + Real.sqrt 2))) ∧ y < 2) ∧
  (∃ (z : ℝ), z = Real.sqrt (3 * Real.sqrt (3 * Real.sqrt (3 * Real.sqrt 3))) ∧ z < 3) ∧
  (∃ (w : ℝ), w = Real.sqrt (3 + Real.sqrt (3 + Real.sqrt (3 + Real.sqrt 3))) ∧ w < 3) :=
by sorry

end NUMINAMATH_CALUDE_four_inequalities_l3701_370167


namespace NUMINAMATH_CALUDE_right_triangle_matchsticks_l3701_370117

theorem right_triangle_matchsticks (a b c : ℕ) : 
  a = 6 ∧ b = 8 ∧ c^2 = a^2 + b^2 → a + b + c = 24 :=
by
  sorry

end NUMINAMATH_CALUDE_right_triangle_matchsticks_l3701_370117


namespace NUMINAMATH_CALUDE_candy_box_capacity_l3701_370161

theorem candy_box_capacity (dan_capacity : ℕ) (dan_height dan_width dan_length : ℝ) 
  (ella_height ella_width ella_length : ℝ) :
  dan_capacity = 150 →
  ella_height = 3 * dan_height →
  ella_width = 3 * dan_width →
  ella_length = 3 * dan_length →
  ⌊(ella_height * ella_width * ella_length) / (dan_height * dan_width * dan_length) * dan_capacity⌋ = 4050 :=
by sorry

end NUMINAMATH_CALUDE_candy_box_capacity_l3701_370161


namespace NUMINAMATH_CALUDE_room_width_calculation_l3701_370104

/-- Given a rectangular room with length 5.5 m, and a total paving cost of 12375 at a rate of 600 per sq. meter, the width of the room is 3.75 meters. -/
theorem room_width_calculation (length : ℝ) (total_cost : ℝ) (cost_per_sqm : ℝ) (width : ℝ) : 
  length = 5.5 → 
  total_cost = 12375 → 
  cost_per_sqm = 600 → 
  width = total_cost / (length * cost_per_sqm) → 
  width = 3.75 := by
sorry

#eval (12375 : Float) / (5.5 * 600) -- Evaluates to 3.75

end NUMINAMATH_CALUDE_room_width_calculation_l3701_370104


namespace NUMINAMATH_CALUDE_blue_marble_probability_l3701_370173

theorem blue_marble_probability (total : ℕ) (yellow : ℕ) :
  total = 120 →
  yellow = 30 →
  let green := yellow / 3
  let red := 2 * yellow
  let blue := total - (yellow + green + red)
  (blue : ℚ) / total = 1 / 6 := by sorry

end NUMINAMATH_CALUDE_blue_marble_probability_l3701_370173


namespace NUMINAMATH_CALUDE_plane_speed_theorem_l3701_370132

/-- Given a plane's speed against a tailwind and the tailwind speed, 
    calculate the plane's speed with the tailwind. -/
def plane_speed_with_tailwind (speed_against_tailwind : ℝ) (tailwind_speed : ℝ) : ℝ :=
  speed_against_tailwind + 2 * tailwind_speed

/-- Theorem: The plane's speed with the tailwind is 460 mph given the conditions. -/
theorem plane_speed_theorem (speed_against_tailwind : ℝ) (tailwind_speed : ℝ) 
  (h1 : speed_against_tailwind = 310)
  (h2 : tailwind_speed = 75) :
  plane_speed_with_tailwind speed_against_tailwind tailwind_speed = 460 := by
  sorry

#eval plane_speed_with_tailwind 310 75

end NUMINAMATH_CALUDE_plane_speed_theorem_l3701_370132


namespace NUMINAMATH_CALUDE_magic_square_sum_l3701_370128

/-- Represents a 3x3 magic square with five unknown values -/
structure MagicSquare where
  a : ℕ
  b : ℕ
  c : ℕ
  d : ℕ
  e : ℕ

/-- The magic sum (sum of each row, column, and diagonal) -/
def magicSum (sq : MagicSquare) : ℕ := 15 + sq.b + 27

/-- Conditions for the magic square -/
def isMagicSquare (sq : MagicSquare) : Prop :=
  magicSum sq = 15 + sq.b + 27
  ∧ magicSum sq = 24 + sq.a + sq.d
  ∧ magicSum sq = sq.e + 18 + sq.c
  ∧ magicSum sq = 15 + sq.a + sq.c
  ∧ magicSum sq = sq.b + sq.a + 18
  ∧ magicSum sq = 27 + sq.d + sq.c
  ∧ magicSum sq = 15 + sq.a + sq.c
  ∧ magicSum sq = 27 + sq.a + sq.e

theorem magic_square_sum (sq : MagicSquare) (h : isMagicSquare sq) : sq.d + sq.e = 47 := by
  sorry


end NUMINAMATH_CALUDE_magic_square_sum_l3701_370128


namespace NUMINAMATH_CALUDE_j_range_l3701_370194

def h (x : ℝ) : ℝ := 2 * x + 1

def j (x : ℝ) : ℝ := h (h (h (h (h x))))

theorem j_range :
  ∀ y ∈ Set.range j,
  -1 ≤ y ∧ y ≤ 127 ∧
  ∃ x : ℝ, -1 ≤ x ∧ x ≤ 3 ∧ j x = y :=
by sorry

end NUMINAMATH_CALUDE_j_range_l3701_370194


namespace NUMINAMATH_CALUDE_arithmetic_sequence_problem_l3701_370110

/-- An arithmetic sequence -/
def ArithmeticSequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

/-- The theorem statement -/
theorem arithmetic_sequence_problem (a : ℕ → ℝ) 
    (h_arith : ArithmeticSequence a)
    (h_sum : a 6 + a 8 = 10)
    (h_a3 : a 3 = 1) :
    a 11 = 9 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_problem_l3701_370110


namespace NUMINAMATH_CALUDE_rectangle_dimensions_l3701_370120

theorem rectangle_dimensions (x : ℝ) : 
  (x - 3 > 0) →
  (3 * x + 4 > 0) →
  ((x - 3) * (3 * x + 4) = 12 * x - 9) →
  (x = (17 + 5 * Real.sqrt 13) / 6) :=
by sorry

end NUMINAMATH_CALUDE_rectangle_dimensions_l3701_370120


namespace NUMINAMATH_CALUDE_xiao_dong_language_understanding_l3701_370172

-- Define propositions
variable (P : Prop) -- Xiao Dong understands English
variable (Q : Prop) -- Xiao Dong understands French

-- Theorem statement
theorem xiao_dong_language_understanding : 
  ¬(P ∧ Q) → (P → ¬Q) :=
by
  sorry

end NUMINAMATH_CALUDE_xiao_dong_language_understanding_l3701_370172
