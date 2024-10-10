import Mathlib

namespace hexagon_area_is_32_l1942_194274

/-- A hexagon surrounded by triangles forming a rectangle -/
structure HexagonWithTriangles where
  num_triangles : ℕ
  triangle_area : ℝ
  rectangle_area : ℝ

/-- The area of the hexagon -/
def hexagon_area (h : HexagonWithTriangles) : ℝ :=
  h.rectangle_area - h.num_triangles * h.triangle_area

/-- Theorem: The area of the hexagon is 32 square units -/
theorem hexagon_area_is_32 (h : HexagonWithTriangles) 
    (h_num_triangles : h.num_triangles = 4)
    (h_triangle_area : h.triangle_area = 2)
    (h_rectangle_area : h.rectangle_area = 40) : 
  hexagon_area h = 32 := by
  sorry

end hexagon_area_is_32_l1942_194274


namespace solution_implies_m_value_l1942_194200

theorem solution_implies_m_value (x m : ℝ) : 
  x = 2 → 4 * x + 2 * m - 14 = 0 → m = 3 := by
  sorry

end solution_implies_m_value_l1942_194200


namespace outfit_combinations_l1942_194232

theorem outfit_combinations (shirts : ℕ) (pants : ℕ) (restricted_combinations : ℕ) :
  shirts = 5 →
  pants = 4 →
  restricted_combinations = 1 →
  shirts * pants - restricted_combinations = 19 :=
by sorry

end outfit_combinations_l1942_194232


namespace fourth_power_nested_square_roots_l1942_194282

theorem fourth_power_nested_square_roots :
  (Real.sqrt (1 + Real.sqrt (2 + Real.sqrt (3 + Real.sqrt 4))))^4 = 3 + Real.sqrt 5 + 2 * Real.sqrt (2 + Real.sqrt 5) :=
by sorry

end fourth_power_nested_square_roots_l1942_194282


namespace onion_weight_problem_l1942_194219

theorem onion_weight_problem (total_weight : Real) (avg_weight_35 : Real) :
  total_weight = 7.68 ∧ avg_weight_35 = 0.190 →
  (total_weight * 1000 - 35 * avg_weight_35 * 1000) / 5 = 206 := by
  sorry

end onion_weight_problem_l1942_194219


namespace triangle_inequality_expression_l1942_194281

theorem triangle_inequality_expression (a b c : ℝ) : 
  (a > 0) → (b > 0) → (c > 0) → 
  (a + b > c) → (b + c > a) → (c + a > b) →
  (a^2 + b^2 - c^2 - 2*a*b < 0) :=
by
  sorry

end triangle_inequality_expression_l1942_194281


namespace outfit_choices_l1942_194206

/-- The number of shirts -/
def num_shirts : ℕ := 5

/-- The number of pants -/
def num_pants : ℕ := 5

/-- The number of hats -/
def num_hats : ℕ := 7

/-- The number of colors shared by shirts and pants -/
def num_shared_colors : ℕ := 5

/-- The total number of possible outfit combinations -/
def total_combinations : ℕ := num_shirts * num_pants * num_hats

/-- The number of undesired outfit combinations (shirt and pants same color) -/
def undesired_combinations : ℕ := num_shared_colors * num_hats

/-- The number of valid outfit choices -/
def valid_outfit_choices : ℕ := total_combinations - undesired_combinations

theorem outfit_choices :
  valid_outfit_choices = 140 := by sorry

end outfit_choices_l1942_194206


namespace total_ribbons_used_l1942_194268

def dresses_per_day_first_week : ℕ := 2
def days_first_week : ℕ := 7
def dresses_per_day_second_week : ℕ := 3
def days_second_week : ℕ := 2
def ribbons_per_dress : ℕ := 2

theorem total_ribbons_used :
  (dresses_per_day_first_week * days_first_week + 
   dresses_per_day_second_week * days_second_week) * 
  ribbons_per_dress = 40 := by
  sorry

end total_ribbons_used_l1942_194268


namespace initial_snowflakes_count_l1942_194258

/-- Calculates the initial number of snowflakes given the rate of snowfall and total snowflakes after one hour -/
def initial_snowflakes (rate : ℕ) (interval : ℕ) (total_after_hour : ℕ) : ℕ :=
  total_after_hour - (60 / interval) * rate

/-- Theorem: The initial number of snowflakes is 10 -/
theorem initial_snowflakes_count : initial_snowflakes 4 5 58 = 10 := by
  sorry

end initial_snowflakes_count_l1942_194258


namespace power_equality_l1942_194255

theorem power_equality (m : ℝ) : (16 : ℝ) ^ (3/4) = 2^m → m = 3 := by
  sorry

end power_equality_l1942_194255


namespace sachin_gain_is_487_50_l1942_194228

/-- Calculates Sachin's gain in one year based on given borrowing and lending conditions. -/
def sachinsGain (X R1 R2 R3 : ℚ) : ℚ :=
  let interestFromRahul := X * R2 / 100
  let interestFromRavi := X * R3 / 100
  let interestPaid := X * R1 / 100
  interestFromRahul + interestFromRavi - interestPaid

/-- Theorem stating that Sachin's gain in one year is 487.50 rupees. -/
theorem sachin_gain_is_487_50 :
  sachinsGain 5000 4 (25/4) (15/2) = 487.5 := by
  sorry

#eval sachinsGain 5000 4 (25/4) (15/2)

end sachin_gain_is_487_50_l1942_194228


namespace cubic_roots_sum_l1942_194220

theorem cubic_roots_sum (p q r : ℝ) : 
  (p^3 - 5*p^2 + 9*p - 7 = 0) →
  (q^3 - 5*q^2 + 9*q - 7 = 0) →
  (r^3 - 5*r^2 + 9*r - 7 = 0) →
  ∃ (u v : ℝ), ((p+q)^3 + u*(p+q)^2 + v*(p+q) + (-13) = 0) ∧
               ((q+r)^3 + u*(q+r)^2 + v*(q+r) + (-13) = 0) ∧
               ((r+p)^3 + u*(r+p)^2 + v*(r+p) + (-13) = 0) :=
by sorry

end cubic_roots_sum_l1942_194220


namespace polynomial_divisibility_l1942_194284

theorem polynomial_divisibility : ∃ q : Polynomial ℝ, 
  (X^3 * 6 + X^2 * 1 + -1) = (X * 2 + -1) * q :=
by
  sorry

end polynomial_divisibility_l1942_194284


namespace product_of_polynomials_l1942_194254

theorem product_of_polynomials (p q : ℝ) : 
  (∀ k : ℝ, (5 * k^2 - 2 * k + p) * (4 * k^2 + q * k - 6) = 20 * k^4 - 18 * k^3 - 31 * k^2 + 12 * k + 18) →
  p + q = -3 := by
sorry

end product_of_polynomials_l1942_194254


namespace min_sum_of_bases_l1942_194265

theorem min_sum_of_bases (a b : ℕ+) : 
  (3 * a + 5 = 5 * b + 3) → 
  (∀ (x y : ℕ+), (3 * x + 5 = 5 * y + 3) → (x + y ≥ a + b)) →
  a + b = 10 := by
sorry

end min_sum_of_bases_l1942_194265


namespace arithmetic_sequence_sum_l1942_194267

/-- An arithmetic sequence of 5 terms starting with 3 and ending with 15 -/
def ArithmeticSequence (a b c : ℝ) : Prop :=
  ∃ d : ℝ, (a = 3 + d) ∧ (b = 3 + 2*d) ∧ (c = 3 + 3*d) ∧ (15 = 3 + 4*d)

/-- The sum of the middle three terms of the arithmetic sequence is 27 -/
theorem arithmetic_sequence_sum (a b c : ℝ) 
  (h : ArithmeticSequence a b c) : a + b + c = 27 := by
  sorry

end arithmetic_sequence_sum_l1942_194267


namespace decimal_to_fraction_l1942_194297

theorem decimal_to_fraction : 
  (2.75 : ℚ) = 11 / 4 := by sorry

end decimal_to_fraction_l1942_194297


namespace chocolate_difference_l1942_194212

theorem chocolate_difference (robert_chocolates nickel_chocolates : ℕ) 
  (h1 : robert_chocolates = 7)
  (h2 : nickel_chocolates = 3) : 
  robert_chocolates - nickel_chocolates = 4 := by
sorry

end chocolate_difference_l1942_194212


namespace geometric_sequence_fifth_term_l1942_194209

/-- A geometric sequence is a sequence where each term after the first is found by multiplying the previous term by a fixed, non-zero number called the common ratio. -/
def GeometricSequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, r ≠ 0 ∧ ∀ n : ℕ, a (n + 1) = a n * r

/-- The theorem states that for a geometric sequence with positive terms, if the third term is 8 and the seventh term is 18, then the fifth term is 12. -/
theorem geometric_sequence_fifth_term
  (a : ℕ → ℝ)
  (h_pos : ∀ n, a n > 0)
  (h_geom : GeometricSequence a)
  (h_third : a 3 = 8)
  (h_seventh : a 7 = 18) :
  a 5 = 12 :=
sorry

end geometric_sequence_fifth_term_l1942_194209


namespace equation_solution_l1942_194271

theorem equation_solution :
  ∃! y : ℝ, 7 * (4 * y + 3) - 5 = -3 * (2 - 8 * y) + 5 * y ∧ y = 22 := by
  sorry

end equation_solution_l1942_194271


namespace functional_inequality_implies_zero_function_l1942_194235

theorem functional_inequality_implies_zero_function 
  (f : ℝ → ℝ) (h : ∀ x y : ℝ, f (x * y) ≤ y * f x + f y) : 
  ∀ x : ℝ, f x = 0 := by
  sorry

end functional_inequality_implies_zero_function_l1942_194235


namespace pyramid_volume_l1942_194224

theorem pyramid_volume (total_area : ℝ) (base_area : ℝ) (triangular_face_area : ℝ) :
  total_area = 648 ∧
  triangular_face_area = (1/3) * base_area ∧
  total_area = base_area + 4 * triangular_face_area →
  ∃ (s h : ℝ),
    s > 0 ∧
    h > 0 ∧
    base_area = s^2 ∧
    (1/3) * s^2 * h = 486 * Real.sqrt 15 :=
by sorry

end pyramid_volume_l1942_194224


namespace quadratic_one_root_l1942_194226

/-- Given a quadratic equation x^2 + (6+4m)x + (9-m) = 0 where m is a real number,
    prove that it has exactly one real root if and only if m = 0 and m ≥ 0 -/
theorem quadratic_one_root (m : ℝ) : 
  (∃! x : ℝ, x^2 + (6+4*m)*x + (9-m) = 0) ↔ (m = 0 ∧ m ≥ 0) := by
  sorry

#check quadratic_one_root

end quadratic_one_root_l1942_194226


namespace kaleb_allowance_l1942_194272

theorem kaleb_allowance (savings : ℕ) (toy_cost : ℕ) (num_toys : ℕ) (allowance : ℕ) : 
  savings = 21 → 
  toy_cost = 6 → 
  num_toys = 6 → 
  allowance = num_toys * toy_cost - savings → 
  allowance = 15 := by
sorry

end kaleb_allowance_l1942_194272


namespace inseparable_triangles_exist_l1942_194273

-- Define a Point in 3D space
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

-- Define a Triangle in 3D space
structure Triangle3D where
  a : Point3D
  b : Point3D
  c : Point3D

-- Define a function to check if two triangles can be separated by a plane
def canBeSeparated (t1 t2 : Triangle3D) : Prop :=
  ∃ (a b c d : ℝ), ∀ (p : Point3D),
    (p = t1.a ∨ p = t1.b ∨ p = t1.c) →
      a * p.x + b * p.y + c * p.z + d > 0 ∧
    (p = t2.a ∨ p = t2.b ∨ p = t2.c) →
      a * p.x + b * p.y + c * p.z + d < 0

-- Theorem statement
theorem inseparable_triangles_exist (points : Fin 6 → Point3D) :
  ∃ (t1 t2 : Triangle3D),
    (∀ i j k : Fin 6, i ≠ j ∧ j ≠ k ∧ i ≠ k →
      (t1 = Triangle3D.mk (points i) (points j) (points k) ∨
       t2 = Triangle3D.mk (points i) (points j) (points k))) ∧
    ¬(canBeSeparated t1 t2) :=
sorry

end inseparable_triangles_exist_l1942_194273


namespace number_of_c_animals_l1942_194270

/-- Given the number of (A) and (B) animals, and the relationship between (A), (B), and (C) animals,
    prove that the number of (C) animals is 5. -/
theorem number_of_c_animals (a b : ℕ) (h1 : a = 45) (h2 : b = 32) 
    (h3 : b + c = a - 8) : c = 5 :=
by sorry

end number_of_c_animals_l1942_194270


namespace sheep_herds_l1942_194295

theorem sheep_herds (total_sheep : ℕ) (sheep_per_herd : ℕ) (h1 : total_sheep = 60) (h2 : sheep_per_herd = 20) :
  total_sheep / sheep_per_herd = 3 := by
sorry

end sheep_herds_l1942_194295


namespace no_infinite_sequence_with_greater_than_neighbors_average_l1942_194266

theorem no_infinite_sequence_with_greater_than_neighbors_average :
  ¬ ∃ (a : ℕ → ℕ+), ∀ n : ℕ, n ≥ 1 → (a n : ℚ) > ((a (n - 1) : ℚ) + (a (n + 1) : ℚ)) / 2 :=
sorry

end no_infinite_sequence_with_greater_than_neighbors_average_l1942_194266


namespace largest_y_value_l1942_194203

theorem largest_y_value (x y : ℝ) 
  (eq1 : x^2 + 3*x*y - y^2 = 27)
  (eq2 : 3*x^2 - x*y + y^2 = 27) :
  ∃ (y_max : ℝ), y_max = 3 ∧ 
  (∀ (y' : ℝ), (∃ (x' : ℝ), x'^2 + 3*x'*y' - y'^2 = 27 ∧ 
                             3*x'^2 - x'*y' + y'^2 = 27) → 
                y' ≤ y_max) :=
sorry

end largest_y_value_l1942_194203


namespace factorization_2m_squared_minus_18_l1942_194288

theorem factorization_2m_squared_minus_18 (m : ℝ) : 2 * m^2 - 18 = 2 * (m + 3) * (m - 3) := by
  sorry

end factorization_2m_squared_minus_18_l1942_194288


namespace minimum_rental_fee_is_3520_l1942_194257

/-- Represents a bus type with its seat capacity and rental fee. -/
structure BusType where
  seats : ℕ
  fee : ℕ

/-- Calculates the minimum rental fee for transporting a given number of people
    using two types of buses. -/
def minimumRentalFee (people : ℕ) (busA : BusType) (busB : BusType) : ℕ :=
  let totalBuses := 8
  let x := 4  -- number of Type A buses
  x * busA.fee + (totalBuses - x) * busB.fee

/-- Theorem stating that the minimum rental fee for 360 people using the given bus types is 3520 yuan. -/
theorem minimum_rental_fee_is_3520 :
  let people := 360
  let busA := BusType.mk 40 400
  let busB := BusType.mk 50 480
  minimumRentalFee people busA busB = 3520 := by
  sorry

#eval minimumRentalFee 360 (BusType.mk 40 400) (BusType.mk 50 480)

end minimum_rental_fee_is_3520_l1942_194257


namespace probability_all_even_simplified_l1942_194211

def total_slips : ℕ := 49
def even_slips : ℕ := 9
def draws : ℕ := 8

def probability_all_even : ℚ :=
  (9 * 8 * 7 * 6 * 5 * 4 * 3 * 2) / (49 * 48 * 47 * 46 * 45 * 44 * 43 * 42)

theorem probability_all_even_simplified (h : probability_all_even = 5 / (6 * 47 * 23 * 45 * 11 * 43 * 7)) :
  probability_all_even = 5 / (6 * 47 * 23 * 45 * 11 * 43 * 7) := by
  sorry

end probability_all_even_simplified_l1942_194211


namespace arithmetic_sequence_sum_l1942_194239

def arithmetic_sequence (a : ℕ → ℤ) : Prop :=
  ∃ d : ℤ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_sum
  (a : ℕ → ℤ)
  (h_arithmetic : arithmetic_sequence a)
  (h_a1 : a 1 = 1)
  (h_a3 : a 3 = -5) :
  a 1 - a 2 - a 3 - a 4 = 16 := by
sorry

end arithmetic_sequence_sum_l1942_194239


namespace tangent_lines_condition_l1942_194253

/-- The function f(x) = 4x + ax² has two tangent lines passing through (1,1) iff a ∈ (-∞, -3) ∪ (0, +∞) -/
theorem tangent_lines_condition (a : ℝ) :
  (∃ (x₁ x₂ : ℝ), x₁ ≠ x₂ ∧
    (4 * x₁ + a * x₁^2 - (4 + 2*a*x₁) * x₁ + (4 + 2*a*x₁) = 1) ∧
    (4 * x₂ + a * x₂^2 - (4 + 2*a*x₂) * x₂ + (4 + 2*a*x₂) = 1)) ↔
  (a < -3 ∨ a > 0) :=
by sorry


end tangent_lines_condition_l1942_194253


namespace largest_number_problem_l1942_194217

theorem largest_number_problem (a b c : ℝ) : 
  a < b → b < c → 
  a + b + c = 67 → 
  c - b = 7 → 
  b - a = 3 → 
  c = 28 := by
sorry

end largest_number_problem_l1942_194217


namespace right_triangle_hypotenuse_l1942_194248

theorem right_triangle_hypotenuse (a b c : ℝ) : 
  a = 60 → b = 80 → c^2 = a^2 + b^2 → c = 100 := by sorry

end right_triangle_hypotenuse_l1942_194248


namespace gcf_of_180_240_45_l1942_194221

theorem gcf_of_180_240_45 : Nat.gcd 180 (Nat.gcd 240 45) = 15 := by
  sorry

end gcf_of_180_240_45_l1942_194221


namespace finite_solutions_of_exponential_equation_l1942_194269

theorem finite_solutions_of_exponential_equation :
  ∃ (S : Finset (ℕ × ℕ × ℕ × ℕ)),
    ∀ (x y z n : ℕ), (2^x : ℕ) + 5^y - 31^z = n.factorial → (x, y, z, n) ∈ S := by
  sorry

end finite_solutions_of_exponential_equation_l1942_194269


namespace one_ball_selection_l1942_194287

/-- The number of red balls in the bag -/
def num_red_balls : ℕ := 2

/-- The number of blue balls in the bag -/
def num_blue_balls : ℕ := 4

/-- Each ball has a different number -/
axiom balls_are_distinct : True

/-- The number of ways to select one ball from the bag -/
def ways_to_select_one_ball : ℕ := num_red_balls + num_blue_balls

theorem one_ball_selection :
  ways_to_select_one_ball = 6 :=
by sorry

end one_ball_selection_l1942_194287


namespace circle_properties_l1942_194252

/-- 
Given an equation x^2 + y^2 - 2x + 4y + m = 0 representing a circle,
prove that the center coordinates are (1, -2) and the range of m is (-∞, 5)
-/
theorem circle_properties (x y m : ℝ) :
  (x^2 + y^2 - 2*x + 4*y + m = 0) →
  (∃ r : ℝ, r > 0 ∧ (x - 1)^2 + (y + 2)^2 = r^2) →
  ((1, -2) = (1, -2) ∧ m < 5) := by
sorry

end circle_properties_l1942_194252


namespace ryan_age_problem_l1942_194202

/-- Ryan's age problem -/
theorem ryan_age_problem : ∃ x : ℕ, 
  (∃ n : ℕ, x - 2 = n^3) ∧ 
  (∃ m : ℕ, x + 3 = m^2) ∧ 
  x = 2195 :=
sorry

end ryan_age_problem_l1942_194202


namespace park_circle_diameter_l1942_194283

/-- Represents the circular arrangement in the park -/
structure ParkCircle where
  fountain_diameter : ℝ
  garden_width : ℝ
  inner_path_width : ℝ
  outer_path_width : ℝ

/-- Calculates the diameter of the outermost boundary of the park circle -/
def outer_diameter (park : ParkCircle) : ℝ :=
  park.fountain_diameter + 2 * (park.garden_width + park.inner_path_width + park.outer_path_width)

/-- Theorem stating that for the given dimensions, the outer diameter is 50 feet -/
theorem park_circle_diameter :
  let park : ParkCircle := {
    fountain_diameter := 12,
    garden_width := 9,
    inner_path_width := 3,
    outer_path_width := 7
  }
  outer_diameter park = 50 := by
  sorry

end park_circle_diameter_l1942_194283


namespace parabola_intersection_l1942_194251

/-- 
Given a parabola y = 2x² translated right by p units and down by q units,
prove that it intersects y = x - 4 at exactly one point when p = q = 31/8.
-/
theorem parabola_intersection (p q : ℝ) : 
  (∃! x, 2*(x - p)^2 - q = x - 4) ↔ (p = 31/8 ∧ q = 31/8) := by
  sorry

#check parabola_intersection

end parabola_intersection_l1942_194251


namespace line_segment_endpoint_l1942_194201

theorem line_segment_endpoint (x : ℝ) :
  x > 0 →
  (((x - 2)^2 + (6 - 2)^2).sqrt = 7) →
  x = 2 + Real.sqrt 33 := by
sorry

end line_segment_endpoint_l1942_194201


namespace line_intercepts_sum_l1942_194292

theorem line_intercepts_sum (x y : ℝ) : 
  x / 3 - y / 4 = 1 → x + y = -1 := by
  sorry

end line_intercepts_sum_l1942_194292


namespace parabola_line_intersections_l1942_194290

theorem parabola_line_intersections (a b c : ℝ) (ha : a ≠ 0) :
  (∀ (x y : ℝ), (y = a * x^2 + b * x + c ∧ y = a * x + b) → 
    (∃! p : ℝ × ℝ, p.1 = x ∧ p.2 = y)) ∧
  (∀ (x y : ℝ), (y = a * x^2 + b * x + c ∧ y = b * x + c) → 
    (∃! p : ℝ × ℝ, p.1 = x ∧ p.2 = y)) ∧
  (∀ (x y : ℝ), (y = a * x^2 + b * x + c ∧ y = c * x + a) → 
    (∃! p : ℝ × ℝ, p.1 = x ∧ p.2 = y)) ∧
  (∀ (x y : ℝ), (y = a * x^2 + b * x + c ∧ y = b * x + a) → 
    (∃! p : ℝ × ℝ, p.1 = x ∧ p.2 = y)) ∧
  (∀ (x y : ℝ), (y = a * x^2 + b * x + c ∧ y = c * x + b) → 
    (∃! p : ℝ × ℝ, p.1 = x ∧ p.2 = y)) ∧
  (∀ (x y : ℝ), (y = a * x^2 + b * x + c ∧ y = a * x + c) → 
    (∃! p : ℝ × ℝ, p.1 = x ∧ p.2 = y)) →
  1 ≤ c / a ∧ c / a ≤ 5 := by
sorry

end parabola_line_intersections_l1942_194290


namespace jeremy_pill_count_l1942_194222

/-- Calculates the total number of pills taken over a period of time given dosage information --/
def total_pills (dose_mg : ℕ) (dose_interval_hours : ℕ) (pill_mg : ℕ) (duration_weeks : ℕ) : ℕ :=
  let pills_per_dose := dose_mg / pill_mg
  let doses_per_day := 24 / dose_interval_hours
  let pills_per_day := pills_per_dose * doses_per_day
  let days := duration_weeks * 7
  pills_per_day * days

/-- Proves that Jeremy takes 112 pills in total during his 2-week treatment --/
theorem jeremy_pill_count : total_pills 1000 6 500 2 = 112 := by
  sorry

end jeremy_pill_count_l1942_194222


namespace raul_initial_money_l1942_194240

def initial_money (comics_bought : ℕ) (comic_price : ℕ) (money_left : ℕ) : ℕ :=
  comics_bought * comic_price + money_left

theorem raul_initial_money :
  initial_money 8 4 55 = 87 := by
  sorry

end raul_initial_money_l1942_194240


namespace fraction_ceiling_evaluation_l1942_194299

theorem fraction_ceiling_evaluation : 
  (⌈(23 : ℚ) / 11 - ⌈(31 : ℚ) / 19⌉⌉) / (⌈(35 : ℚ) / 9 + ⌈(9 * 19 : ℚ) / 35⌉⌉) = 1 / 9 := by
  sorry

end fraction_ceiling_evaluation_l1942_194299


namespace product_of_binary_and_ternary_l1942_194238

-- Define a function to convert from binary to decimal
def binary_to_decimal (binary : List Bool) : ℕ :=
  binary.enum.foldl (fun acc (i, b) => acc + if b then 2^i else 0) 0

-- Define a function to convert from ternary to decimal
def ternary_to_decimal (ternary : List ℕ) : ℕ :=
  ternary.enum.foldl (fun acc (i, d) => acc + d * 3^i) 0

-- Define the binary and ternary numbers
def binary_num : List Bool := [true, true, false, true]
def ternary_num : List ℕ := [2, 2, 1]

-- State the theorem
theorem product_of_binary_and_ternary :
  (binary_to_decimal binary_num) * (ternary_to_decimal ternary_num) = 187 := by
  sorry

end product_of_binary_and_ternary_l1942_194238


namespace common_roots_product_l1942_194294

-- Define the cubic equations
def cubic1 (x C : ℝ) : ℝ := x^3 + 3*x^2 + C*x + 15
def cubic2 (x D : ℝ) : ℝ := x^3 + D*x^2 + 70

-- Define the condition of having two common roots
def has_two_common_roots (C D : ℝ) : Prop :=
  ∃ (p q : ℝ), p ≠ q ∧ cubic1 p C = 0 ∧ cubic1 q C = 0 ∧ cubic2 p D = 0 ∧ cubic2 q D = 0

-- The main theorem
theorem common_roots_product (C D : ℝ) : 
  has_two_common_roots C D → 
  ∃ (p q : ℝ), p * q = 10 * (7/2)^(1/3) :=
sorry

end common_roots_product_l1942_194294


namespace group_commutativity_l1942_194256

theorem group_commutativity (G : Type*) [Group G] (m n : ℕ) 
  (coprime_mn : Nat.Coprime m n)
  (surj_m : Function.Surjective (fun x : G => x^(m+1)))
  (surj_n : Function.Surjective (fun x : G => x^(n+1)))
  (endo_m : ∀ (x y : G), (x*y)^(m+1) = x^(m+1) * y^(m+1))
  (endo_n : ∀ (x y : G), (x*y)^(n+1) = x^(n+1) * y^(n+1)) :
  ∀ (a b : G), a * b = b * a := by
  sorry

end group_commutativity_l1942_194256


namespace bicycle_trip_average_speed_l1942_194241

/-- Calculates the average speed of a bicycle trip with varying conditions -/
theorem bicycle_trip_average_speed :
  let total_distance : ℝ := 500
  let flat_road_distance : ℝ := 100
  let flat_road_speed : ℝ := 20
  let uphill_distance : ℝ := 50
  let uphill_speed : ℝ := 10
  let flat_terrain_distance : ℝ := 200
  let flat_terrain_speed : ℝ := 15
  let headwind_distance : ℝ := 150
  let headwind_speed : ℝ := 12
  let rest_time_1 : ℝ := 0.5  -- 30 minutes in hours
  let rest_time_2 : ℝ := 1/3  -- 20 minutes in hours
  let rest_time_3 : ℝ := 2/3  -- 40 minutes in hours
  
  let total_time : ℝ := 
    flat_road_distance / flat_road_speed +
    uphill_distance / uphill_speed +
    flat_terrain_distance / flat_terrain_speed +
    headwind_distance / headwind_speed +
    rest_time_1 + rest_time_2 + rest_time_3
  
  let average_speed : ℝ := total_distance / total_time
  
  ∃ ε > 0, |average_speed - 13.4| < ε :=
by sorry

end bicycle_trip_average_speed_l1942_194241


namespace teacher_worksheets_proof_l1942_194261

def calculate_total_worksheets (initial : ℕ) (graded : ℕ) (additional : ℕ) : ℕ :=
  let remaining := initial - graded
  let after_additional := remaining + additional
  after_additional + 2 * after_additional

theorem teacher_worksheets_proof : 
  let initial := 6
  let graded := 4
  let additional := 18
  calculate_total_worksheets initial graded additional = 60 := by
  sorry

end teacher_worksheets_proof_l1942_194261


namespace absolute_value_inequality_solution_set_l1942_194298

theorem absolute_value_inequality_solution_set :
  {x : ℝ | x ≠ 0 ∧ |((x - 2) / x)| > (x - 2) / x} = Set.Ioo 0 2 := by
  sorry

end absolute_value_inequality_solution_set_l1942_194298


namespace gcd_of_72_120_168_l1942_194218

theorem gcd_of_72_120_168 : Nat.gcd 72 (Nat.gcd 120 168) = 24 := by
  sorry

end gcd_of_72_120_168_l1942_194218


namespace minimum_additional_amount_l1942_194242

def current_order : ℝ := 49.90
def discount_rate : ℝ := 0.10
def target_amount : ℝ := 50.00

theorem minimum_additional_amount :
  ∃ (x : ℝ), x ≥ 0 ∧
  (current_order + x) * (1 - discount_rate) = target_amount ∧
  ∀ (y : ℝ), y ≥ 0 → (current_order + y) * (1 - discount_rate) ≥ target_amount → y ≥ x :=
by sorry

end minimum_additional_amount_l1942_194242


namespace prob_odd_total_is_221_441_l1942_194243

/-- Represents a standard die with one dot removed randomly -/
structure ModifiedDie :=
  (remaining_dots : Fin 21)

/-- The probability of a modified die showing an odd number of dots on top -/
def prob_odd_top (d : ModifiedDie) : ℚ := 11 / 21

/-- The probability of a modified die showing an even number of dots on top -/
def prob_even_top (d : ModifiedDie) : ℚ := 10 / 21

/-- The probability of two modified dice showing an odd total number of dots on top when rolled simultaneously -/
def prob_odd_total (d1 d2 : ModifiedDie) : ℚ :=
  (prob_odd_top d1 * prob_odd_top d2) + (prob_even_top d1 * prob_even_top d2)

theorem prob_odd_total_is_221_441 (d1 d2 : ModifiedDie) :
  prob_odd_total d1 d2 = 221 / 441 := by
  sorry

end prob_odd_total_is_221_441_l1942_194243


namespace tan_theta_value_l1942_194280

theorem tan_theta_value (θ : ℝ) (z₁ z₂ : ℂ) 
  (h1 : z₁ = Complex.mk (Real.sin θ) (-4/5))
  (h2 : z₂ = Complex.mk (3/5) (-Real.cos θ))
  (h3 : (z₁ - z₂).re = 0) : 
  Real.tan θ = -3/4 := by
  sorry

end tan_theta_value_l1942_194280


namespace prove_earnings_l1942_194247

/-- Gondor's earnings from repairing devices -/
def earnings_problem : Prop :=
  let phone_repair_fee : ℕ := 10
  let laptop_repair_fee : ℕ := 20
  let monday_phones : ℕ := 3
  let tuesday_phones : ℕ := 5
  let wednesday_laptops : ℕ := 2
  let thursday_laptops : ℕ := 4
  let total_earnings : ℕ := 
    phone_repair_fee * (monday_phones + tuesday_phones) +
    laptop_repair_fee * (wednesday_laptops + thursday_laptops)
  total_earnings = 200

theorem prove_earnings : earnings_problem := by
  sorry

end prove_earnings_l1942_194247


namespace isosceles_trapezoid_rotation_l1942_194296

/-- Represents an isosceles trapezoid -/
structure IsoscelesTrapezoid where
  -- Add necessary fields for an isosceles trapezoid
  longerBase : ℝ
  shorterBase : ℝ
  height : ℝ
  -- Add necessary conditions for an isosceles trapezoid
  longerBase_gt_shorterBase : longerBase > shorterBase
  bases_positive : longerBase > 0 ∧ shorterBase > 0
  height_positive : height > 0

/-- Represents a solid of revolution -/
inductive SolidOfRevolution
  | Cylinder
  | Cone
  | FrustumOfCone

/-- The result of rotating an isosceles trapezoid around its longer base -/
def rotateIsoscelesTrapezoid (t : IsoscelesTrapezoid) : List SolidOfRevolution :=
  [SolidOfRevolution.Cylinder, SolidOfRevolution.Cone, SolidOfRevolution.Cone]

theorem isosceles_trapezoid_rotation (t : IsoscelesTrapezoid) :
  rotateIsoscelesTrapezoid t = [SolidOfRevolution.Cylinder, SolidOfRevolution.Cone, SolidOfRevolution.Cone] := by
  sorry

end isosceles_trapezoid_rotation_l1942_194296


namespace total_shaded_area_l1942_194275

/-- Given a square carpet with the following properties:
  * Total side length of 16 feet
  * Contains one large shaded square and twelve smaller congruent shaded squares
  * Ratio of carpet side to large shaded square side (S) is 4:1
  * Ratio of large shaded square side (S) to smaller shaded square side (T) is 2:1
  The total shaded area is 64 square feet. -/
theorem total_shaded_area (carpet_side : ℝ) (S T : ℝ) : 
  carpet_side = 16 ∧ 
  carpet_side / S = 4 ∧ 
  S / T = 2 → 
  S^2 + 12 * T^2 = 64 := by
  sorry

end total_shaded_area_l1942_194275


namespace coincide_points_l1942_194225

/-- A point on the coordinate plane with integer coordinates -/
structure IntPoint where
  x : Int
  y : Int

/-- A vector between two integer points -/
def vector (a b : IntPoint) : IntPoint :=
  ⟨b.x - a.x, b.y - a.y⟩

/-- Move a point by a vector -/
def movePoint (p v : IntPoint) : IntPoint :=
  ⟨p.x + v.x, p.y + v.y⟩

/-- The main theorem stating that any two points can be made to coincide -/
theorem coincide_points (a b c d : IntPoint) :
  ∃ (moves : List (IntPoint → IntPoint)),
    ∃ (p q : IntPoint),
      (p ∈ [a, b, c, d]) ∧
      (q ∈ [a, b, c, d]) ∧
      (p ≠ q) ∧
      (moves.foldl (λ acc f => f acc) p = moves.foldl (λ acc f => f acc) q) :=
by sorry

end coincide_points_l1942_194225


namespace x_squared_less_than_one_iff_l1942_194264

theorem x_squared_less_than_one_iff (x : ℝ) : -1 < x ∧ x < 1 ↔ x^2 < 1 := by sorry

end x_squared_less_than_one_iff_l1942_194264


namespace percentage_of_quarters_l1942_194263

theorem percentage_of_quarters (dimes quarters half_dollars : ℕ) : 
  dimes = 75 → quarters = 35 → half_dollars = 15 →
  (quarters * 25 : ℚ) / (dimes * 10 + quarters * 25 + half_dollars * 50) = 368 / 1000 := by
  sorry

end percentage_of_quarters_l1942_194263


namespace music_movements_duration_l1942_194259

theorem music_movements_duration 
  (a b c : ℝ) 
  (h_order : a ≤ b ∧ b ≤ c) 
  (h_total : a + b + c = 60) 
  (h_max : c ≤ a + b) 
  (h_diff : b - a ≥ 3 ∧ c - b ≥ 3) 
  (h_positive : a > 0 ∧ b > 0 ∧ c > 0) : 
  3 ≤ a ∧ a ≤ 17 := by
sorry

end music_movements_duration_l1942_194259


namespace min_value_of_expression_l1942_194289

theorem min_value_of_expression (m n : ℝ) (hm : m > 0) (hn : n > 0) : 
  let a : ℝ × ℝ := (m, 1)
  let b : ℝ × ℝ := (4 - n, 2)
  (∃ (k : ℝ), a = k • b) → 
  (∀ (x y : ℝ), x > 0 → y > 0 → (1 / x + 8 / y ≥ 9 / 2) ∧ 
  (∃ (x₀ y₀ : ℝ), x₀ > 0 ∧ y₀ > 0 ∧ 1 / x₀ + 8 / y₀ = 9 / 2)) :=
by sorry

end min_value_of_expression_l1942_194289


namespace denise_removed_five_bananas_l1942_194208

/-- The number of bananas Denise removed from the jar -/
def bananas_removed (original : ℕ) (remaining : ℕ) : ℕ := original - remaining

/-- Theorem stating that Denise removed 5 bananas -/
theorem denise_removed_five_bananas :
  bananas_removed 46 41 = 5 := by
  sorry

end denise_removed_five_bananas_l1942_194208


namespace west_60_meters_representation_l1942_194291

/-- Represents the direction of movement --/
inductive Direction
  | East
  | West

/-- Represents a movement with direction and distance --/
structure Movement where
  direction : Direction
  distance : ℝ

/-- Converts a movement to its numerical representation --/
def Movement.toNumber (m : Movement) : ℝ :=
  match m.direction with
  | Direction.East => -m.distance
  | Direction.West => m.distance

theorem west_60_meters_representation :
  ∀ (m : Movement),
    m.direction = Direction.West ∧
    m.distance = 60 →
    m.toNumber = 60 :=
by sorry

end west_60_meters_representation_l1942_194291


namespace aqua_opposite_red_l1942_194230

-- Define the set of colors
inductive Color : Type
  | Red | White | Green | Brown | Aqua | Purple

-- Define a cube as a function from face positions to colors
def Cube := Fin 6 → Color

-- Define face positions
def top : Fin 6 := 0
def bottom : Fin 6 := 1
def front : Fin 6 := 2
def back : Fin 6 := 3
def right : Fin 6 := 4
def left : Fin 6 := 5

-- Define the conditions of the problem
def cube_conditions (c : Cube) : Prop :=
  (c top = Color.Brown) ∧
  (c right = Color.Green) ∧
  (c front = Color.Red ∨ c front = Color.White ∨ c front = Color.Purple) ∧
  (c back = Color.Aqua)

-- State the theorem
theorem aqua_opposite_red (c : Cube) :
  cube_conditions c → c front = Color.Red :=
by sorry

end aqua_opposite_red_l1942_194230


namespace infinitely_many_primes_divide_fib_l1942_194246

-- Define the Fibonacci sequence
def fib : ℕ → ℕ
  | 0 => 0
  | 1 => 1
  | (n + 2) => fib (n + 1) + fib n

-- State the theorem
theorem infinitely_many_primes_divide_fib : 
  ∃ (S : Set ℕ), (Set.Infinite S) ∧ (∀ p ∈ S, Prime p ∧ (fib (p - 1) % p = 0)) :=
sorry

end infinitely_many_primes_divide_fib_l1942_194246


namespace cos_2x_value_l1942_194237

theorem cos_2x_value (x : Real) (h : 2 * Real.sin (Real.pi - x) + 1 = 0) : 
  Real.cos (2 * x) = 1 / 2 := by
  sorry

end cos_2x_value_l1942_194237


namespace odd_integer_pairs_theorem_l1942_194286

def phi : ℕ → ℕ := sorry  -- Euler's totient function

theorem odd_integer_pairs_theorem (a b : ℕ) (ha : Odd a) (hb : Odd b) (ha_gt_1 : a > 1) (hb_gt_1 : b > 1) :
  7 * (phi a)^2 - phi (a * b) + 11 * (phi b)^2 = 2 * (a^2 + b^2) →
  ∃ x : ℕ, a = 15 * 3^x ∧ b = 3 * 3^x :=
sorry

end odd_integer_pairs_theorem_l1942_194286


namespace trigonometric_equation_solution_l1942_194249

theorem trigonometric_equation_solution (n : ℤ) :
  let f (x : ℝ) := (Real.sin x) ^ (Real.arctan (Real.sin x + Real.cos x))
  let g (x : ℝ) := (1 / Real.sin x) ^ (Real.arctan (Real.sin (2 * x)) + π / 4)
  let x₁ := 2 * n * π + π / 2
  let x₂ := 2 * n * π + 3 * π / 4
  ∀ x ∈ Set.Ioo (2 * n * π) ((2 * n + 1) * π),
    f x = g x ↔ (x = x₁ ∨ x = x₂) := by sorry

end trigonometric_equation_solution_l1942_194249


namespace jeong_hyeok_is_nine_l1942_194229

/-- Jeong-hyeok's age -/
def jeong_hyeok_age : ℕ := sorry

/-- Jeong-hyeok's uncle's age -/
def uncle_age : ℕ := sorry

/-- Condition 1: Jeong-hyeok's age is 1 year less than 1/4 of his uncle's age -/
axiom condition1 : jeong_hyeok_age = uncle_age / 4 - 1

/-- Condition 2: His uncle's age is 5 years less than 5 times Jeong-hyeok's age -/
axiom condition2 : uncle_age = 5 * jeong_hyeok_age - 5

/-- Theorem: Jeong-hyeok is 9 years old -/
theorem jeong_hyeok_is_nine : jeong_hyeok_age = 9 := by sorry

end jeong_hyeok_is_nine_l1942_194229


namespace hyperbola_intersection_l1942_194262

/-- Hyperbola with given properties -/
structure Hyperbola where
  a : ℝ
  b : ℝ
  h_positive : 0 < a ∧ 0 < b
  h_real_axis : a = 1
  h_focus : a^2 + b^2 = 5

/-- Line intersecting the hyperbola -/
def intersecting_line (x : ℝ) : ℝ := x + 2

/-- Theorem about the hyperbola and its intersecting line -/
theorem hyperbola_intersection (C : Hyperbola) :
  (∀ x y, x^2 / C.a^2 - y^2 / C.b^2 = 1 ↔ x^2 - y^2 / 4 = 1) ∧
  (∃ A B : ℝ × ℝ,
    A ≠ B ∧
    (A.1^2 - A.2^2 / 4 = 1) ∧
    (B.1^2 - B.2^2 / 4 = 1) ∧
    (A.2 = intersecting_line A.1) ∧
    (B.2 = intersecting_line B.1) ∧
    ((A.1 - B.1)^2 + (A.2 - B.2)^2 = 32 / 3)) := by
  sorry

end hyperbola_intersection_l1942_194262


namespace sqrt_62_plus_24_sqrt_11_l1942_194213

theorem sqrt_62_plus_24_sqrt_11 :
  ∃ (a b c : ℤ), 
    (∀ (n : ℕ), n > 1 → ¬(∃ (k : ℕ), c = n^2 * k)) →
    Real.sqrt (62 + 24 * Real.sqrt 11) = a + b * Real.sqrt c ∧
    a = 6 ∧ b = 2 ∧ c = 11 := by
  sorry

end sqrt_62_plus_24_sqrt_11_l1942_194213


namespace pig_teeth_count_l1942_194234

/-- The number of teeth a dog has -/
def dog_teeth : ℕ := 42

/-- The number of teeth a cat has -/
def cat_teeth : ℕ := 30

/-- The number of dogs Vann will clean -/
def num_dogs : ℕ := 5

/-- The number of cats Vann will clean -/
def num_cats : ℕ := 10

/-- The number of pigs Vann will clean -/
def num_pigs : ℕ := 7

/-- The total number of teeth Vann will clean -/
def total_teeth : ℕ := 706

/-- Theorem stating that pigs have 28 teeth each -/
theorem pig_teeth_count : 
  (total_teeth - (num_dogs * dog_teeth + num_cats * cat_teeth)) / num_pigs = 28 := by
  sorry

end pig_teeth_count_l1942_194234


namespace vincent_songs_l1942_194233

/-- The number of songs Vincent knows after summer camp -/
def total_songs (initial_songs : ℕ) (new_songs : ℕ) : ℕ :=
  initial_songs + new_songs

/-- Theorem stating that Vincent knows 74 songs after summer camp -/
theorem vincent_songs : total_songs 56 18 = 74 := by
  sorry

end vincent_songs_l1942_194233


namespace ellipse_m_range_l1942_194276

/-- An ellipse equation with parameter m -/
def ellipse_equation (x y m : ℝ) : Prop :=
  x^2 / (m + 2) - y^2 / (m + 1) = 1

/-- Condition for foci on y-axis -/
def foci_on_y_axis (m : ℝ) : Prop :=
  -(m + 1) > m + 2 ∧ m + 2 > 0

/-- Theorem stating the range of m for the given ellipse -/
theorem ellipse_m_range :
  ∀ m : ℝ, (∃ x y : ℝ, ellipse_equation x y m) ∧ foci_on_y_axis m ↔ -2 < m ∧ m < -3/2 :=
sorry

end ellipse_m_range_l1942_194276


namespace seagulls_remaining_l1942_194277

theorem seagulls_remaining (initial : ℕ) (scared_fraction : ℚ) (flew_fraction : ℚ) : 
  initial = 36 → scared_fraction = 1/4 → flew_fraction = 1/3 → 
  (initial - initial * scared_fraction - (initial - initial * scared_fraction) * flew_fraction : ℚ) = 18 := by
  sorry

end seagulls_remaining_l1942_194277


namespace a_range_l1942_194231

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := 3 * x^2 - 2 * a * x - 8

-- State the theorem
theorem a_range (a : ℝ) :
  (∀ x ∈ Set.Ioo 1 2, ¬Monotone (f a)) →
  a ∈ Set.Ioo 3 6 :=
by
  sorry

end a_range_l1942_194231


namespace parabola_equation_l1942_194285

/-- A parabola in the Cartesian coordinate system with focus at (-2, 0) has the standard equation y^2 = -8x -/
theorem parabola_equation (x y : ℝ) : 
  (∃ (p : ℝ), p > 0 ∧ y^2 = -2*p*x ∧ p/2 = 2) → 
  y^2 = -8*x := by
sorry

end parabola_equation_l1942_194285


namespace prize_guesses_count_l1942_194216

def digit_partitions : List (Nat × Nat × Nat) :=
  [(1,1,6), (1,2,5), (1,3,4), (1,4,3), (1,5,2), (1,6,1),
   (2,1,5), (2,2,4), (2,3,3), (2,4,2), (2,5,1),
   (3,1,4), (3,2,3), (3,3,2), (3,4,1),
   (4,1,3), (4,2,2), (4,3,1)]

def digit_arrangements : Nat := 70

theorem prize_guesses_count : 
  (List.length digit_partitions) * digit_arrangements = 1260 := by
  sorry

end prize_guesses_count_l1942_194216


namespace string_average_length_l1942_194279

theorem string_average_length : 
  let string1 : ℝ := 2
  let string2 : ℝ := 6
  let string3 : ℝ := 9
  let num_strings : ℕ := 3
  (string1 + string2 + string3) / num_strings = 17 / 3 := by
  sorry

end string_average_length_l1942_194279


namespace smallest_weight_set_has_11_weights_l1942_194244

/-- A set of weights that can be divided into equal piles -/
structure WeightSet where
  weights : List ℕ
  divisible_by_4 : ∃ (n : ℕ), 4 * n = weights.sum
  divisible_by_5 : ∃ (n : ℕ), 5 * n = weights.sum
  divisible_by_6 : ∃ (n : ℕ), 6 * n = weights.sum

/-- The property of being the smallest set of weights divisible by 4, 5, and 6 -/
def is_smallest_weight_set (ws : WeightSet) : Prop :=
  ∀ (other : WeightSet), other.weights.length ≥ ws.weights.length

/-- The theorem stating that 11 is the smallest number of weights divisible by 4, 5, and 6 -/
theorem smallest_weight_set_has_11_weights :
  ∃ (ws : WeightSet), ws.weights.length = 11 ∧ is_smallest_weight_set ws :=
sorry

end smallest_weight_set_has_11_weights_l1942_194244


namespace product_of_powers_plus_one_l1942_194260

theorem product_of_powers_plus_one : (3 + 1) * (3^2 + 1^2) * (3^4 + 1^4) * (3^8 + 1^8) = 21527360 := by
  sorry

end product_of_powers_plus_one_l1942_194260


namespace car_distance_l1942_194205

theorem car_distance (train_speed : ℝ) (car_speed_ratio : ℝ) (time : ℝ) :
  train_speed = 100 →
  car_speed_ratio = 5 / 8 →
  time = 45 / 60 →
  car_speed_ratio * train_speed * time = 46.875 := by
  sorry

end car_distance_l1942_194205


namespace road_repair_workers_l1942_194207

/-- The number of persons in the first group -/
def first_group : ℕ := 63

/-- The number of days the first group works -/
def first_days : ℕ := 12

/-- The number of hours per day the first group works -/
def first_hours : ℕ := 5

/-- The number of days the second group works -/
def second_days : ℕ := 21

/-- The number of hours per day the second group works -/
def second_hours : ℕ := 6

/-- The total man-hours required to complete the work -/
def total_man_hours : ℕ := first_group * first_days * first_hours

/-- The number of persons in the second group -/
def second_group : ℕ := total_man_hours / (second_days * second_hours)

theorem road_repair_workers :
  second_group = 30 :=
by sorry

end road_repair_workers_l1942_194207


namespace segment_sum_is_132_div_7_l1942_194250

/-- Represents an acute triangle with two altitudes dividing its sides. -/
structure AcuteTriangleWithAltitudes where
  /-- Length of the first known segment -/
  a : ℝ
  /-- Length of the second known segment -/
  b : ℝ
  /-- Length of the third known segment -/
  c : ℝ
  /-- Length of the unknown segment -/
  y : ℝ
  /-- Condition that all segment lengths are positive -/
  ha : a > 0
  hb : b > 0
  hc : c > 0
  hy : y > 0
  /-- Condition that the triangle is acute -/
  acute : True  -- We don't have enough information to express this condition precisely

/-- The sum of all segments on the sides of the triangle cut by the altitudes -/
def segmentSum (t : AcuteTriangleWithAltitudes) : ℝ :=
  t.a + t.b + t.c + t.y

/-- Theorem stating that for a triangle with segments 7, 4, 5, and y, the sum is 132/7 -/
theorem segment_sum_is_132_div_7 (t : AcuteTriangleWithAltitudes)
  (h1 : t.a = 7) (h2 : t.b = 4) (h3 : t.c = 5) :
  segmentSum t = 132 / 7 := by
  sorry

end segment_sum_is_132_div_7_l1942_194250


namespace parabola_properties_l1942_194214

/-- A parabola with the given properties -/
def Parabola : Set (ℝ × ℝ) :=
  {(x, y) | y^2 = 8*x}

theorem parabola_properties :
  -- The parabola is symmetric about the x-axis
  (∀ x y, (x, y) ∈ Parabola ↔ (x, -y) ∈ Parabola) ∧
  -- The vertex of the parabola is at the origin
  (0, 0) ∈ Parabola ∧
  -- The parabola passes through point (2, 4)
  (2, 4) ∈ Parabola :=
by sorry

end parabola_properties_l1942_194214


namespace blue_fish_ratio_l1942_194210

theorem blue_fish_ratio (total_fish : ℕ) (blue_spotted_fish : ℕ) : 
  total_fish = 30 →
  blue_spotted_fish = 5 →
  (blue_spotted_fish : ℚ) / (total_fish : ℚ) = 1/6 →
  (2 * blue_spotted_fish : ℚ) / (total_fish : ℚ) = 1/3 := by
  sorry

end blue_fish_ratio_l1942_194210


namespace min_diff_same_last_two_digits_l1942_194236

/-- Given positive integers m and n where m > n, if the last two digits of 9^m and 9^n are the same, 
    then the minimum value of m - n is 10. -/
theorem min_diff_same_last_two_digits (m n : ℕ) : 
  m > n → 
  (∃ k : ℕ, 9^m ≡ k [ZMOD 100] ∧ 9^n ≡ k [ZMOD 100]) → 
  (∀ p q : ℕ, p > q → (∃ j : ℕ, 9^p ≡ j [ZMOD 100] ∧ 9^q ≡ j [ZMOD 100]) → m - n ≤ p - q) → 
  m - n = 10 := by
sorry

end min_diff_same_last_two_digits_l1942_194236


namespace sport_formulation_water_amount_l1942_194278

/-- Represents the ratio of ingredients in a drink formulation -/
structure DrinkRatio :=
  (flavoring : ℚ)
  (corn_syrup : ℚ)
  (water : ℚ)

/-- The standard formulation ratio -/
def standard_ratio : DrinkRatio :=
  ⟨1, 12, 30⟩

/-- The sport formulation ratio -/
def sport_ratio : DrinkRatio :=
  ⟨1, 4, 60⟩

/-- Theorem stating the relationship between corn syrup and water in the sport formulation -/
theorem sport_formulation_water_amount
  (corn_syrup_amount : ℚ)
  (h1 : corn_syrup_amount = 5)
  (h2 : sport_ratio.flavoring * sport_ratio.water = 
        2 * (standard_ratio.flavoring * standard_ratio.water))
  (h3 : sport_ratio.flavoring * sport_ratio.corn_syrup = 
        3 * (standard_ratio.flavoring * standard_ratio.corn_syrup)) :
  corn_syrup_amount * (sport_ratio.water / sport_ratio.corn_syrup) = 75 :=
sorry

end sport_formulation_water_amount_l1942_194278


namespace line_slope_is_two_l1942_194204

-- Define the polar equation of the line
def polar_equation (ρ θ : ℝ) : Prop := ρ * Real.sin θ - 2 * ρ * Real.cos θ + 3 = 0

-- Theorem: The slope of the line defined by the polar equation is 2
theorem line_slope_is_two :
  ∃ (m : ℝ), m = 2 ∧
  ∀ (x y : ℝ), (∃ (ρ θ : ℝ), polar_equation ρ θ ∧ x = ρ * Real.cos θ ∧ y = ρ * Real.sin θ) →
  y = m * x - 3 :=
sorry

end line_slope_is_two_l1942_194204


namespace partner_investment_time_l1942_194227

/-- Given two partners p and q with investment ratio 7:5 and profit ratio 7:10,
    where q invested for 40 months, prove that p invested for 28 months. -/
theorem partner_investment_time
  (investment_ratio : ℚ) -- Ratio of p's investment to q's investment
  (profit_ratio : ℚ) -- Ratio of p's profit to q's profit
  (q_time : ℕ) -- Time q invested in months
  (h1 : investment_ratio = 7 / 5)
  (h2 : profit_ratio = 7 / 10)
  (h3 : q_time = 40) :
  ∃ (p_time : ℕ), p_time = 28 := by
  sorry

end partner_investment_time_l1942_194227


namespace interest_rate_calculation_l1942_194293

/-- Calculates the interest rate for a purchase with a payment plan. -/
theorem interest_rate_calculation (purchase_price down_payment monthly_payment num_months : ℚ)
  (h_purchase : purchase_price = 112)
  (h_down : down_payment = 12)
  (h_monthly : monthly_payment = 10)
  (h_months : num_months = 12) :
  ∃ (interest_rate : ℚ), 
    (abs (interest_rate - 17.9) < 0.05) ∧ 
    (interest_rate = (((down_payment + monthly_payment * num_months) - purchase_price) / purchase_price) * 100) :=
by sorry

end interest_rate_calculation_l1942_194293


namespace tan_alpha_minus_pi_sixth_l1942_194245

theorem tan_alpha_minus_pi_sixth (α : Real) : 
  (∃ (x y : Real), x = -Real.sqrt 3 ∧ y = 2 ∧ 
   Real.tan α = y / x) →
  Real.tan (α - π/6) = -3 * Real.sqrt 3 := by
sorry

end tan_alpha_minus_pi_sixth_l1942_194245


namespace solution_set_f_greater_than_2_range_of_t_l1942_194223

-- Define the function f
def f (x : ℝ) : ℝ := |2 * x + 2| - |x - 2|

-- Theorem for the solution set of f(x) > 2
theorem solution_set_f_greater_than_2 :
  {x : ℝ | f x > 2} = {x : ℝ | x < -6 ∨ x > 2/3} := by sorry

-- Theorem for the range of t
theorem range_of_t :
  {t : ℝ | ∀ x, f x ≥ t^2 - (7/2)*t} = {t : ℝ | 3/2 ≤ t ∧ t ≤ 2} := by sorry

end solution_set_f_greater_than_2_range_of_t_l1942_194223


namespace tan_difference_l1942_194215

theorem tan_difference (α β : Real) (h1 : Real.tan α = 3) (h2 : Real.tan β = 4/3) :
  Real.tan (α - β) = 1/3 := by
  sorry

end tan_difference_l1942_194215
