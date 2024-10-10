import Mathlib

namespace workshop_workers_count_l532_53272

/-- Proves that the total number of workers in a workshop is 21, given specific salary conditions. -/
theorem workshop_workers_count :
  let total_average_salary : ℕ := 8000
  let technician_count : ℕ := 7
  let technician_average_salary : ℕ := 12000
  let non_technician_average_salary : ℕ := 6000
  ∃ (total_workers : ℕ),
    total_workers * total_average_salary = 
      technician_count * technician_average_salary + 
      (total_workers - technician_count) * non_technician_average_salary ∧
    total_workers = 21 := by
sorry

end workshop_workers_count_l532_53272


namespace combinatorial_identity_l532_53243

theorem combinatorial_identity (n k : ℕ) (h : k ≤ n) :
  Nat.choose (n + 1) k = Nat.choose n k + Nat.choose n (k - 1) := by
  sorry

end combinatorial_identity_l532_53243


namespace first_solution_percentage_l532_53213

-- Define the volumes and percentages
def volume_first : ℝ := 40
def volume_second : ℝ := 60
def percent_second : ℝ := 0.7
def percent_final : ℝ := 0.5
def total_volume : ℝ := 100

-- Define the theorem
theorem first_solution_percentage :
  ∃ (percent_first : ℝ),
    volume_first * percent_first + volume_second * percent_second = total_volume * percent_final ∧
    percent_first = 0.2 := by
  sorry

end first_solution_percentage_l532_53213


namespace intersection_nonempty_iff_a_greater_than_one_l532_53238

theorem intersection_nonempty_iff_a_greater_than_one (a : ℝ) :
  ({x : ℝ | x > 1} ∩ {x : ℝ | x ≤ a}).Nonempty ↔ a > 1 := by
  sorry

end intersection_nonempty_iff_a_greater_than_one_l532_53238


namespace odd_painted_faces_5x5x1_l532_53290

/-- Represents a 3D grid of unit cubes -/
structure CubeGrid :=
  (length : Nat)
  (width : Nat)
  (height : Nat)

/-- Counts the number of cubes with an odd number of painted faces in a given grid -/
def countOddPaintedFaces (grid : CubeGrid) : Nat :=
  sorry

/-- The main theorem stating that a 5x5x1 grid has 9 cubes with an odd number of painted faces -/
theorem odd_painted_faces_5x5x1 :
  let grid := CubeGrid.mk 5 5 1
  countOddPaintedFaces grid = 9 := by
  sorry

end odd_painted_faces_5x5x1_l532_53290


namespace divisibility_by_power_of_two_l532_53276

theorem divisibility_by_power_of_two (n : ℕ) (h : n > 0) :
  ∃ x : ℤ, (2^n : ℤ) ∣ (x^2 - 17) := by sorry

end divisibility_by_power_of_two_l532_53276


namespace company_j_payroll_company_j_payroll_correct_l532_53248

/-- Calculates the total monthly payroll for factory workers given the conditions of Company J. -/
theorem company_j_payroll (factory_workers : ℕ) (office_workers : ℕ) 
  (office_payroll : ℕ) (salary_difference : ℕ) : ℕ :=
  let factory_workers := 15
  let office_workers := 30
  let office_payroll := 75000
  let salary_difference := 500
  30000

theorem company_j_payroll_correct : 
  company_j_payroll 15 30 75000 500 = 30000 := by sorry

end company_j_payroll_company_j_payroll_correct_l532_53248


namespace inverse_of_A_l532_53256

def A : Matrix (Fin 2) (Fin 2) ℚ := ![![5, -3], ![2, 1]]

def A_inv : Matrix (Fin 2) (Fin 2) ℚ := ![![1/11, 3/11], ![-2/11, 5/11]]

theorem inverse_of_A : A * A_inv = 1 ∧ A_inv * A = 1 := by sorry

end inverse_of_A_l532_53256


namespace actual_height_is_191_l532_53240

/-- Represents the height correction problem for a class of students. -/
structure HeightCorrectionProblem where
  num_students : ℕ
  initial_average : ℝ
  incorrect_height : ℝ
  actual_average : ℝ

/-- Calculates the actual height of the student with the incorrect measurement. -/
def calculate_actual_height (problem : HeightCorrectionProblem) : ℝ :=
  problem.num_students * (problem.initial_average - problem.actual_average) + problem.incorrect_height

/-- Theorem stating that the actual height of the student with the incorrect measurement is 191 cm. -/
theorem actual_height_is_191 (problem : HeightCorrectionProblem)
  (h1 : problem.num_students = 20)
  (h2 : problem.initial_average = 175)
  (h3 : problem.incorrect_height = 151)
  (h4 : problem.actual_average = 173) :
  calculate_actual_height problem = 191 := by
  sorry

end actual_height_is_191_l532_53240


namespace multiply_polynomial_equality_l532_53239

theorem multiply_polynomial_equality (x : ℝ) :
  (x^6 + 27*x^3 + 729) * (x^3 - 27) = x^12 + 27*x^9 - 19683*x^3 - 531441 := by
  sorry

end multiply_polynomial_equality_l532_53239


namespace not_right_triangle_6_7_8_l532_53266

/-- A function that checks if three numbers can form a right triangle --/
def is_right_triangle (a b c : ℝ) : Prop :=
  a > 0 ∧ b > 0 ∧ c > 0 ∧ (a^2 + b^2 = c^2 ∨ a^2 + c^2 = b^2 ∨ b^2 + c^2 = a^2)

/-- Theorem stating that 6, 7, and 8 cannot form a right triangle --/
theorem not_right_triangle_6_7_8 : ¬ is_right_triangle 6 7 8 := by
  sorry

end not_right_triangle_6_7_8_l532_53266


namespace sum_interior_angles_is_3240_l532_53293

/-- A regular polygon Q where each interior angle is 9 times its corresponding exterior angle -/
structure RegularPolygon where
  n : ℕ  -- number of sides
  interior_angle : ℝ  -- measure of each interior angle
  exterior_angle : ℝ  -- measure of each exterior angle
  is_regular : interior_angle = 9 * exterior_angle
  sum_exterior : n * exterior_angle = 360

/-- The sum of interior angles of a RegularPolygon -/
def sum_interior_angles (Q : RegularPolygon) : ℝ :=
  Q.n * Q.interior_angle

/-- Theorem: The sum of interior angles of a RegularPolygon is 3240° -/
theorem sum_interior_angles_is_3240 (Q : RegularPolygon) :
  sum_interior_angles Q = 3240 := by
  sorry

end sum_interior_angles_is_3240_l532_53293


namespace paper_sheets_count_l532_53254

/-- Represents the dimensions of a rectangle in centimeters -/
structure Dimensions where
  width : ℝ
  height : ℝ

/-- Calculates the area of a rectangle given its dimensions -/
def area (d : Dimensions) : ℝ := d.width * d.height

/-- Converts meters to centimeters -/
def meters_to_cm (m : ℝ) : ℝ := m * 100

theorem paper_sheets_count :
  let plank : Dimensions := ⟨meters_to_cm 6, meters_to_cm 4⟩
  let paper : Dimensions := ⟨60, 20⟩
  (area plank) / (area paper) = 200 := by sorry

end paper_sheets_count_l532_53254


namespace find_n_l532_53233

theorem find_n (e n : ℕ+) (h1 : Nat.lcm e n = 690) 
  (h2 : ¬ 3 ∣ n) (h3 : ¬ 2 ∣ e) : n = 230 := by
  sorry

end find_n_l532_53233


namespace pencil_box_cost_is_280_l532_53206

/-- Represents the school's purchase of pencils and markers -/
structure SchoolPurchase where
  pencil_cartons : ℕ
  boxes_per_pencil_carton : ℕ
  marker_cartons : ℕ
  boxes_per_marker_carton : ℕ
  marker_carton_cost : ℚ
  total_spent : ℚ

/-- Calculates the cost of each box of pencils -/
def pencil_box_cost (purchase : SchoolPurchase) : ℚ :=
  (purchase.total_spent - purchase.marker_cartons * purchase.marker_carton_cost) /
  (purchase.pencil_cartons * purchase.boxes_per_pencil_carton)

/-- Theorem stating that for the given purchase, each box of pencils costs $2.80 -/
theorem pencil_box_cost_is_280 (purchase : SchoolPurchase) 
  (h1 : purchase.pencil_cartons = 20)
  (h2 : purchase.boxes_per_pencil_carton = 10)
  (h3 : purchase.marker_cartons = 10)
  (h4 : purchase.boxes_per_marker_carton = 5)
  (h5 : purchase.marker_carton_cost = 4)
  (h6 : purchase.total_spent = 600) :
  pencil_box_cost purchase = 280 / 100 := by
  sorry

end pencil_box_cost_is_280_l532_53206


namespace shortest_distance_parabola_to_line_l532_53289

/-- The shortest distance between a point on the parabola y = -x^2 + 5x + 7 
    and a point on the line y = 2x - 3 is 31√5/20 -/
theorem shortest_distance_parabola_to_line :
  let parabola := fun x : ℝ => -x^2 + 5*x + 7
  let line := fun x : ℝ => 2*x - 3
  ∃ (d : ℝ), d = (31 * Real.sqrt 5) / 20 ∧
    ∀ (x₁ x₂ : ℝ), 
      d ≤ Real.sqrt ((x₁ - x₂)^2 + (parabola x₁ - line x₂)^2) :=
by
  sorry


end shortest_distance_parabola_to_line_l532_53289


namespace polynomial_simplification_l532_53242

theorem polynomial_simplification (x : ℝ) :
  (2 * x^6 + x^5 + 3 * x^4 + x^3 + 5) - (x^6 + 2 * x^5 + x^4 - x^3 + 7) =
  x^6 - x^5 + 2 * x^4 + 2 * x^3 - 2 := by
  sorry

end polynomial_simplification_l532_53242


namespace gift_wrapping_combinations_l532_53267

/-- The number of wrapping paper varieties -/
def wrapping_paper_varieties : ℕ := 10

/-- The number of ribbon colors -/
def ribbon_colors : ℕ := 5

/-- The number of gift card types -/
def gift_card_types : ℕ := 4

/-- The number of decorative bow types -/
def bow_types : ℕ := 2

/-- The total number of distinct gift-wrapping combinations -/
def total_combinations : ℕ := wrapping_paper_varieties * ribbon_colors * gift_card_types * bow_types

theorem gift_wrapping_combinations :
  total_combinations = 400 :=
by sorry

end gift_wrapping_combinations_l532_53267


namespace sum_of_roots_quadratic_l532_53287

/-- The sum of roots of two quadratic equations given specific conditions -/
theorem sum_of_roots_quadratic (a b c d p q : ℝ) : a ≠ 0 →
  (∃ x y : ℝ, x ≠ y ∧ x^2 + 2020*a*x + c = 0 ∧ y^2 + 2020*a*y + c = 0) →
  (∃ x y : ℝ, x ≠ y ∧ a*x^2 + b*x + d = 0 ∧ a*y^2 + b*y + d = 0) →
  (∃ x y : ℝ, x ≠ y ∧ a*x^2 + p*x + q = 0 ∧ a*y^2 + p*y + q = 0) →
  (∃ w x y z : ℝ, a*w^2 + b*w + d = 0 ∧ a*x^2 + b*x + d = 0 ∧
                  a*y^2 + p*y + q = 0 ∧ a*z^2 + p*z + q = 0 ∧
                  w + x + y + z = 2020) :=
by sorry

end sum_of_roots_quadratic_l532_53287


namespace no_solution_implies_a_range_l532_53286

theorem no_solution_implies_a_range (a : ℝ) : 
  (∀ x : ℝ, ¬(x - 2*a > 0 ∧ 3 - 2*x > x - 6)) → a ≥ 3/2 := by
  sorry

end no_solution_implies_a_range_l532_53286


namespace log_sum_equals_three_l532_53280

theorem log_sum_equals_three : Real.log 4 / Real.log 10 + Real.log 25 / Real.log 10 + (-1/8)^0 = 3 := by
  sorry

end log_sum_equals_three_l532_53280


namespace special_hexagon_perimeter_l532_53220

/-- An equilateral hexagon with three nonadjacent 120° angles -/
structure SpecialHexagon where
  -- Side length
  s : ℝ
  -- Condition that s is positive
  s_pos : s > 0
  -- Area of the hexagon
  area : ℝ
  -- Condition that area is 12 square units
  area_eq : area = 12

/-- The perimeter of a SpecialHexagon is 24 units -/
theorem special_hexagon_perimeter (h : SpecialHexagon) : 
  6 * h.s = 24 := by sorry

end special_hexagon_perimeter_l532_53220


namespace figure_100_cubes_l532_53283

-- Define the sequence of unit cubes for the first four figures
def cube_sequence : Fin 4 → ℕ
  | 0 => 1
  | 1 => 8
  | 2 => 27
  | 3 => 64

-- Define the general formula for the number of cubes in figure n
def num_cubes (n : ℕ) : ℕ := n^3

-- Theorem statement
theorem figure_100_cubes :
  (∀ k : Fin 4, cube_sequence k = num_cubes k) →
  num_cubes 100 = 1000000 := by
  sorry

end figure_100_cubes_l532_53283


namespace ball_drawing_game_l532_53241

theorem ball_drawing_game (x : ℕ) : 
  (2 : ℕ) > 0 ∧ x > 0 →
  (4 * x : ℚ) / ((x + 2) * (x + 1)) ≥ 1/5 ∧
  (4 * x : ℚ) / ((x + 2) * (x + 1)) ≤ 33/100 →
  9 ≤ x ∧ x ≤ 16 :=
by sorry

end ball_drawing_game_l532_53241


namespace fraction_inequality_l532_53234

theorem fraction_inequality (a b : ℕ) (ha : a ≠ 0) (hb : b ≠ 0) 
  (h : (a : ℚ) / b < (a + 1 : ℚ) / (b + 1)) : 
  (2012 * a : ℚ) / b > 1 := by
  sorry

end fraction_inequality_l532_53234


namespace imperial_examination_middle_volume_l532_53265

/-- The number of candidates admitted in the Middle volume given a total number of candidates and a proportion -/
def middle_volume_candidates (total : ℕ) (south north middle : ℕ) : ℕ :=
  total * middle /(south + north + middle)

/-- Theorem stating that given 100 total candidates and a proportion of 11:7:2,
    the number of candidates in the Middle volume is 10 -/
theorem imperial_examination_middle_volume :
  middle_volume_candidates 100 11 7 2 = 10 := by
  sorry

end imperial_examination_middle_volume_l532_53265


namespace b_net_share_is_1450_l532_53229

/-- Represents the salary distribution ratio for employees A, B, C, and D. -/
def salary_ratio : Fin 4 → ℕ
  | 0 => 2  -- A
  | 1 => 3  -- B
  | 2 => 4  -- C
  | 3 => 6  -- D

/-- Represents the salary difference between D and C. -/
def salary_difference : ℕ := 700

/-- Represents the minimum wage requirement. -/
def minimum_wage : ℕ := 1000

/-- Represents the tax rates for different salary brackets. -/
def tax_rate (salary : ℕ) : ℚ :=
  if salary ≤ 1000 then 0
  else if salary ≤ 2000 then 1/10
  else if salary ≤ 3000 then 1/5
  else 3/10

/-- Represents the salary caps for each employee. -/
def salary_cap : Fin 4 → ℕ
  | 0 => 4000  -- A
  | 1 => 3500  -- B
  | 2 => 4500  -- C
  | 3 => 6000  -- D

/-- Calculates B's net share after tax deductions. -/
def b_net_share : ℕ := sorry

/-- Theorem stating that B's net share after tax deductions is $1450. -/
theorem b_net_share_is_1450 : b_net_share = 1450 := by sorry

end b_net_share_is_1450_l532_53229


namespace triangle_exists_and_satisfies_inequality_l532_53232

/-- Theorem: Existence of a triangle with sides 9, 15, and 21 satisfying the triangle inequality. -/
theorem triangle_exists_and_satisfies_inequality : ∃ (a b c : ℝ),
  a = 9 ∧ b = 15 ∧ c = 21 ∧
  (a + b > c) ∧ (b + c > a) ∧ (c + a > b) ∧
  (∃ (x : ℝ), a = 9 ∧ b = x + 6 ∧ c = 2*x + 3 ∧ a + b + c = 45) :=
by sorry

end triangle_exists_and_satisfies_inequality_l532_53232


namespace distance_to_sons_house_l532_53263

/-- The distance to Jennie's son's house -/
def distance : ℝ := 200

/-- The travel time during heavy traffic (in hours) -/
def heavy_traffic_time : ℝ := 5

/-- The travel time with no traffic (in hours) -/
def no_traffic_time : ℝ := 4

/-- The difference in average speed between no traffic and heavy traffic conditions (in mph) -/
def speed_difference : ℝ := 10

/-- Theorem stating that the distance to Jennie's son's house is 200 miles -/
theorem distance_to_sons_house :
  distance = heavy_traffic_time * (distance / heavy_traffic_time) ∧
  distance = no_traffic_time * (distance / no_traffic_time) ∧
  distance / no_traffic_time = distance / heavy_traffic_time + speed_difference :=
by sorry

end distance_to_sons_house_l532_53263


namespace roots_of_equation_l532_53244

theorem roots_of_equation : ∀ x : ℝ, (x - 3)^2 = 25 ↔ x = 8 ∨ x = -2 := by sorry

end roots_of_equation_l532_53244


namespace power_division_equality_l532_53270

theorem power_division_equality : 6^12 / 36^5 = 36 := by sorry

end power_division_equality_l532_53270


namespace lori_marble_sharing_l532_53218

theorem lori_marble_sharing :
  ∀ (total_marbles : ℕ) (share_percent : ℚ) (num_friends : ℕ),
    total_marbles = 60 →
    share_percent = 75 / 100 →
    num_friends = 5 →
    (total_marbles : ℚ) * share_percent / num_friends = 9 := by
  sorry

end lori_marble_sharing_l532_53218


namespace equal_cake_division_l532_53217

theorem equal_cake_division (total_cakes : ℕ) (num_children : ℕ) (cakes_per_child : ℕ) :
  total_cakes = 18 →
  num_children = 3 →
  total_cakes = num_children * cakes_per_child →
  cakes_per_child = 6 := by
sorry

end equal_cake_division_l532_53217


namespace log_properties_l532_53261

-- Define the logarithm function
noncomputable def log (b : ℝ) (x : ℝ) : ℝ := Real.log x / Real.log b

-- Main theorem
theorem log_properties (b : ℝ) (x : ℝ) (y : ℝ) 
    (h1 : b > 1) 
    (h2 : y = log b (x^2)) :
  (x = 1 → y = 0) ∧ 
  (x = -b → y = 2) ∧ 
  (-1 < x ∧ x < 1 → y < 0) := by
  sorry

end log_properties_l532_53261


namespace train_length_l532_53245

/-- The length of a train given its speed and time to cross a pole -/
theorem train_length (speed : ℝ) (time : ℝ) : speed = 52 → time = 18 → ∃ length : ℝ, abs (length - 259.92) < 0.01 := by
  sorry

end train_length_l532_53245


namespace present_ages_sum_l532_53252

theorem present_ages_sum (A B S : ℕ) : 
  A + B = S →
  A = 2 * B →
  (A + 3) + (B + 3) = 66 →
  S = 60 := by
sorry

end present_ages_sum_l532_53252


namespace angle_a_is_30_angle_b_is_150_polygon_sides_is_12_march_day_is_24_l532_53294

-- Define the angle a from the geometric figure
def a : ℝ := 30

-- Define the angle b
def b : ℝ := 150

-- Define the number of sides n in the regular polygon
def n : ℕ := 12

-- Define k as the day of March
def k : ℕ := 24

-- Theorem 1: The angle a in the given geometric figure is 30°
theorem angle_a_is_30 : a = 30 := by sorry

-- Theorem 2: If sin(30° + 210°) = cos b° and 90° < b < 180°, then b = 150°
theorem angle_b_is_150 (h1 : Real.sin (30 + 210) = Real.cos b) (h2 : 90 < b ∧ b < 180) : b = 150 := by sorry

-- Theorem 3: If each interior angle of an n-sided regular polygon is 150°, then n = 12
theorem polygon_sides_is_12 (h : (n - 2) * 180 / n = 150) : n = 12 := by sorry

-- Theorem 4: If the nth day of March is Friday, the kth day is Wednesday, and 20 < k < 25, then k = 24
theorem march_day_is_24 (h1 : k % 7 = (n + 3) % 7) (h2 : 20 < k ∧ k < 25) : k = 24 := by sorry

end angle_a_is_30_angle_b_is_150_polygon_sides_is_12_march_day_is_24_l532_53294


namespace sqrt_five_squared_minus_four_squared_l532_53215

theorem sqrt_five_squared_minus_four_squared : 
  Real.sqrt (5^2 - 4^2) = 3 := by sorry

end sqrt_five_squared_minus_four_squared_l532_53215


namespace equal_area_rectangles_l532_53274

/-- Given two rectangles with equal areas, where one rectangle has dimensions 5 by 24 inches
    and the other has a length of 3 inches, prove that the width of the second rectangle is 40 inches. -/
theorem equal_area_rectangles (carol_length carol_width jordan_length : ℝ)
    (carol_area jordan_area : ℝ) (h1 : carol_length = 5)
    (h2 : carol_width = 24) (h3 : jordan_length = 3)
    (h4 : carol_area = carol_length * carol_width)
    (h5 : jordan_area = jordan_length * (jordan_area / jordan_length))
    (h6 : carol_area = jordan_area) :
  jordan_area / jordan_length = 40 := by
  sorry

end equal_area_rectangles_l532_53274


namespace smallest_sum_for_equation_l532_53251

theorem smallest_sum_for_equation : ∃ (a b : ℕ+), 
  (2^10 * 7^4 : ℕ) = a^(b:ℕ) ∧ 
  (∀ (c d : ℕ+), (2^10 * 7^4 : ℕ) = c^(d:ℕ) → a + b ≤ c + d) ∧
  a + b = 1570 := by
  sorry

end smallest_sum_for_equation_l532_53251


namespace fraction_leading_zeros_l532_53214

-- Define the fraction
def fraction : ℚ := 7 / 5000

-- Define a function to count leading zeros in a decimal representation
def countLeadingZeros (q : ℚ) : ℕ := sorry

-- Theorem statement
theorem fraction_leading_zeros :
  countLeadingZeros fraction = 2 := by sorry

end fraction_leading_zeros_l532_53214


namespace polynomial_identity_l532_53298

theorem polynomial_identity (P : ℝ → ℝ) 
  (h1 : ∀ x, P (x^3) = (P x)^3) 
  (h2 : P 2 = 2) :
  ∀ x, P x = x := by
  sorry

end polynomial_identity_l532_53298


namespace no_triangle_cosine_sum_one_l532_53219

theorem no_triangle_cosine_sum_one :
  ¬ ∃ (A B C : ℝ), 
    (0 < A ∧ A < π) ∧ 
    (0 < B ∧ B < π) ∧ 
    (0 < C ∧ C < π) ∧ 
    (A + B + C = π) ∧
    (Real.cos A + Real.cos B + Real.cos C = 1) :=
by sorry

end no_triangle_cosine_sum_one_l532_53219


namespace three_lines_intersection_l532_53292

/-- A line in the plane represented by ax + by = c -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- The number of intersection points between three lines -/
def num_intersections (l1 l2 l3 : Line) : ℕ :=
  sorry

theorem three_lines_intersection :
  let l1 : Line := { a := -4, b := 6, c := 2 }
  let l2 : Line := { a := 1, b := 2, c := 2 }
  let l3 : Line := { a := -4, b := 6, c := 3 }
  num_intersections l1 l2 l3 = 2 :=
by sorry

end three_lines_intersection_l532_53292


namespace initial_cash_calculation_l532_53295

-- Define the initial cash as a real number
variable (X : ℝ)

-- Define the constants from the problem
def raw_materials : ℝ := 500
def machinery : ℝ := 400
def sales_tax : ℝ := 0.05
def exchange_rate : ℝ := 1.2
def labor_cost_rate : ℝ := 0.1
def inflation_rate : ℝ := 0.02
def years : ℕ := 2
def remaining_amount : ℝ := 900

-- State the theorem
theorem initial_cash_calculation :
  remaining_amount = (X - ((1 + sales_tax) * (raw_materials + machinery) * exchange_rate + labor_cost_rate * X)) / (1 + inflation_rate) ^ years :=
by sorry

end initial_cash_calculation_l532_53295


namespace circle_tangent_and_passes_through_l532_53236

/-- The line to which the circle is tangent -/
def tangent_line (x y : ℝ) : Prop := 4 * x - 3 * y + 6 = 0

/-- The point of tangency -/
def point_A : ℝ × ℝ := (3, 6)

/-- The point through which the circle passes -/
def point_B : ℝ × ℝ := (5, 2)

/-- The equation of the circle -/
def circle_equation (x y : ℝ) : Prop := (x - 5)^2 + (y - 9/2)^2 = 25/4

/-- Theorem stating that the given circle equation represents the circle
    that is tangent to the line at point A and passes through point B -/
theorem circle_tangent_and_passes_through :
  (∀ x y, tangent_line x y → circle_equation x y → (x, y) = point_A) ∧
  circle_equation point_B.1 point_B.2 :=
sorry

end circle_tangent_and_passes_through_l532_53236


namespace simplify_and_ratio_l532_53282

theorem simplify_and_ratio : ∃ (a b : ℤ), 
  (∀ k, (6 * k + 12) / 6 = a * k + b) ∧ 
  (a : ℚ) / b = 1 / 2 := by
  sorry

end simplify_and_ratio_l532_53282


namespace sphere_volume_equals_surface_area_l532_53257

theorem sphere_volume_equals_surface_area (r : ℝ) : 
  (4 / 3 : ℝ) * Real.pi * r^3 = 4 * Real.pi * r^2 → r = 3 := by
  sorry

end sphere_volume_equals_surface_area_l532_53257


namespace quadruple_primes_l532_53247

theorem quadruple_primes (p q r : Nat) (n : Nat) : 
  Nat.Prime p ∧ Nat.Prime q ∧ Nat.Prime r ∧ n > 0 ∧ p^2 = q^2 + r^n →
  ((p = 3 ∧ q = 2 ∧ r = 5 ∧ n = 1) ∨ (p = 5 ∧ q = 3 ∧ r = 2 ∧ n = 4)) :=
by sorry

end quadruple_primes_l532_53247


namespace custom_mul_five_three_l532_53201

-- Define the custom multiplication operation
def custom_mul (a b : ℤ) : ℤ := a^2 + a*b - b^2

-- Theorem statement
theorem custom_mul_five_three : custom_mul 5 3 = 31 := by
  sorry

end custom_mul_five_three_l532_53201


namespace expression_simplification_l532_53230

theorem expression_simplification :
  (3 * Real.sqrt 12) / (Real.sqrt 3 + Real.sqrt 4 + Real.sqrt 6) = Real.sqrt 3 + 2 * Real.sqrt 2 - Real.sqrt 6 := by
  sorry

end expression_simplification_l532_53230


namespace tims_age_l532_53299

theorem tims_age (tim rommel jenny : ℕ) 
  (h1 : rommel = 3 * tim)
  (h2 : jenny = rommel + 2)
  (h3 : tim + 12 = jenny) :
  tim = 5 := by
sorry

end tims_age_l532_53299


namespace min_boat_speed_l532_53202

/-- The minimum speed required for a boat to complete a round trip on a river with a given flow speed, distance, and time constraint. -/
theorem min_boat_speed (S v : ℝ) (h_S : S > 0) (h_v : v ≥ 0) :
  let min_speed := (3 * S + Real.sqrt (9 * S^2 + 4 * v^2)) / 2
  ∀ x : ℝ, x ≥ min_speed →
    S / (x - v) + S / (x + v) + 1/12 ≤ 3/4 :=
by sorry

end min_boat_speed_l532_53202


namespace smallest_area_right_triangle_l532_53281

theorem smallest_area_right_triangle (a b : ℝ) (ha : a = 6) (hb : b = 8) :
  let area1 := a * b / 2
  let c := Real.sqrt (a^2 + b^2)
  let area2 := a * Real.sqrt (b^2 - a^2) / 2
  area2 < area1 := by sorry

end smallest_area_right_triangle_l532_53281


namespace sin_2x_derivative_l532_53231

open Real

theorem sin_2x_derivative (x : ℝ) : 
  deriv (λ x => sin (2 * x)) x = 2 * cos (2 * x) := by sorry

end sin_2x_derivative_l532_53231


namespace mean_median_difference_l532_53212

/-- Represents the frequency distribution of days missed --/
def frequency_distribution : List (Nat × Nat) :=
  [(0, 4), (1, 2), (2, 5), (3, 2), (4, 3), (5, 4)]

/-- Total number of students --/
def total_students : Nat := 20

/-- Calculates the median of the dataset --/
def median (data : List (Nat × Nat)) (total : Nat) : Rat :=
  sorry

/-- Calculates the mean of the dataset --/
def mean (data : List (Nat × Nat)) (total : Nat) : Rat :=
  sorry

/-- The main theorem to prove --/
theorem mean_median_difference :
  (mean frequency_distribution total_students) - 
  (median frequency_distribution total_students) = 7 / 10 := by
  sorry

end mean_median_difference_l532_53212


namespace smallest_common_multiple_9_6_l532_53221

theorem smallest_common_multiple_9_6 : ∀ n : ℕ, n > 0 ∧ 9 ∣ n ∧ 6 ∣ n → n ≥ 18 := by
  sorry

end smallest_common_multiple_9_6_l532_53221


namespace students_taking_paper_c_l532_53222

/-- Represents the systematic sampling setup for the school test -/
structure SchoolSampling where
  total_students : ℕ
  sample_size : ℕ
  first_selected : ℕ
  sampling_interval : ℕ

/-- Calculates the nth term in the arithmetic sequence of selected student numbers -/
def nth_selected (s : SchoolSampling) (n : ℕ) : ℕ :=
  s.first_selected + s.sampling_interval * (n - 1)

/-- Theorem stating the number of students taking test paper C -/
theorem students_taking_paper_c (s : SchoolSampling) 
  (h1 : s.total_students = 800)
  (h2 : s.sample_size = 40)
  (h3 : s.first_selected = 18)
  (h4 : s.sampling_interval = 20) :
  (Finset.filter (fun n => 561 ≤ nth_selected s n ∧ nth_selected s n ≤ 800) 
    (Finset.range s.sample_size)).card = 12 := by
  sorry

end students_taking_paper_c_l532_53222


namespace symmetry_condition_l532_53296

/-- Given a curve y = (ax + b) / (cx - d) where a, b, c, and d are nonzero real numbers,
    if y = x and y = -x are axes of symmetry, then d + b = 0 -/
theorem symmetry_condition (a b c d : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) (hd : d ≠ 0) :
  (∀ x : ℝ, x = (a * x + b) / (c * x - d)) →
  (∀ x : ℝ, x = (a * (-x) + b) / (c * (-x) - d)) →
  d + b = 0 := by
  sorry


end symmetry_condition_l532_53296


namespace smallest_fraction_above_five_sevenths_l532_53209

theorem smallest_fraction_above_five_sevenths :
  ∀ a b : ℕ,
  10 ≤ a ∧ a ≤ 99 →  -- a is a two-digit number
  10 ≤ b ∧ b ≤ 99 →  -- b is a two-digit number
  (5 : ℚ) / 7 < (a : ℚ) / b →  -- fraction is greater than 5/7
  (68 : ℚ) / 95 ≤ (a : ℚ) / b :=
by sorry

end smallest_fraction_above_five_sevenths_l532_53209


namespace roll_three_probability_l532_53284

/-- A fair six-sided die -/
structure FairDie :=
  (sides : Finset Nat)
  (fair : sides = {1, 2, 3, 4, 5, 6})

/-- The event of rolling a 3 -/
def rollThree (d : FairDie) : Finset Nat :=
  {3}

/-- The probability of an event for a fair die -/
def probability (d : FairDie) (event : Finset Nat) : Rat :=
  (event ∩ d.sides).card / d.sides.card

theorem roll_three_probability (d : FairDie) :
  probability d (rollThree d) = 1 / 6 := by
  sorry

end roll_three_probability_l532_53284


namespace minimum_cents_to_win_l532_53227

/-- Represents the state of the game -/
structure GameState where
  beans : ℕ
  cents : ℕ

/-- Applies the penny rule: multiply beans by 5 and add 1 cent -/
def applyPenny (state : GameState) : GameState :=
  { beans := state.beans * 5, cents := state.cents + 1 }

/-- Applies the nickel rule: add 1 bean and 5 cents -/
def applyNickel (state : GameState) : GameState :=
  { beans := state.beans + 1, cents := state.cents + 5 }

/-- Checks if the game is won -/
def isWinningState (state : GameState) : Prop :=
  state.beans > 2008 ∧ state.beans % 100 = 42

/-- Represents a sequence of moves in the game -/
inductive GameMove
  | penny
  | nickel

def applyMove (state : GameState) (move : GameMove) : GameState :=
  match move with
  | GameMove.penny => applyPenny state
  | GameMove.nickel => applyNickel state

def applyMoves (state : GameState) (moves : List GameMove) : GameState :=
  moves.foldl applyMove state

theorem minimum_cents_to_win :
  ∃ (moves : List GameMove),
    let finalState := applyMoves { beans := 0, cents := 0 } moves
    isWinningState finalState ∧
    finalState.cents = 35 ∧
    (∀ (otherMoves : List GameMove),
      let otherFinalState := applyMoves { beans := 0, cents := 0 } otherMoves
      isWinningState otherFinalState → otherFinalState.cents ≥ 35) :=
by sorry

end minimum_cents_to_win_l532_53227


namespace equilateral_triangle_ratio_l532_53208

/-- Given two equilateral triangles with side lengths A and a, and altitudes h_A and h_a respectively,
    if h_A = 2h_a, then the ratio of their perimeters is equal to the ratio of their altitudes. -/
theorem equilateral_triangle_ratio (A a h_A h_a : ℝ) 
  (h_positive : h_a > 0)
  (h_eq : h_A = 2 * h_a) :
  3 * A / (3 * a) = h_A / h_a := by
  sorry

end equilateral_triangle_ratio_l532_53208


namespace quadratic_two_roots_k_range_l532_53203

theorem quadratic_two_roots_k_range (k : ℝ) : 
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ x₁^2 + 2*x₁ - k = 0 ∧ x₂^2 + 2*x₂ - k = 0) → k > -1 := by
  sorry

end quadratic_two_roots_k_range_l532_53203


namespace least_addition_for_divisibility_l532_53235

theorem least_addition_for_divisibility (n : ℕ) : 
  let x := 278
  (∀ y : ℕ, y < x → ¬((1056 + y) % 23 = 0 ∧ (1056 + y) % 29 = 0)) ∧
  ((1056 + x) % 23 = 0 ∧ (1056 + x) % 29 = 0) := by
  sorry

end least_addition_for_divisibility_l532_53235


namespace binders_for_1600_books_20_days_l532_53224

/-- The number of binders required to bind a certain number of books in a given number of days -/
def binders_required (books : ℕ) (days : ℕ) : ℚ :=
  books / (days * (1400 / (30 * 21)))

theorem binders_for_1600_books_20_days :
  binders_required 1600 20 = 36 :=
sorry

end binders_for_1600_books_20_days_l532_53224


namespace smallest_non_prime_without_small_factors_l532_53200

theorem smallest_non_prime_without_small_factors :
  ∃ n : ℕ,
    n > 1 ∧
    ¬ (Nat.Prime n) ∧
    (∀ p : ℕ, Nat.Prime p → p < 10 → ¬ (p ∣ n)) ∧
    (∀ m : ℕ, m > 1 → ¬ (Nat.Prime m) → (∀ q : ℕ, Nat.Prime q → q < 10 → ¬ (q ∣ m)) → m ≥ n) ∧
    120 < n ∧
    n ≤ 130 :=
by sorry

end smallest_non_prime_without_small_factors_l532_53200


namespace tetrahedron_existence_l532_53259

-- Define a tetrahedron type
structure Tetrahedron :=
  (edges : Fin 6 → ℝ)

-- Define the conditions for configuration (a)
def config_a (t : Tetrahedron) : Prop :=
  (∃ i j : Fin 6, i ≠ j ∧ t.edges i < 0.01 ∧ t.edges j < 0.01) ∧
  (∀ k : Fin 6, (t.edges k ≤ 0.01) ∨ (t.edges k > 1000))

-- Define the conditions for configuration (b)
def config_b (t : Tetrahedron) : Prop :=
  (∃ i j k l : Fin 6, i ≠ j ∧ i ≠ k ∧ i ≠ l ∧ j ≠ k ∧ j ≠ l ∧ k ≠ l ∧
    t.edges i < 0.01 ∧ t.edges j < 0.01 ∧ t.edges k < 0.01 ∧ t.edges l < 0.01) ∧
  (∀ m : Fin 6, (t.edges m < 0.01) ∨ (t.edges m > 1000))

-- Theorem statements
theorem tetrahedron_existence :
  (∃ t : Tetrahedron, config_a t) ∧ (¬ ∃ t : Tetrahedron, config_b t) :=
sorry

end tetrahedron_existence_l532_53259


namespace necessary_not_sufficient_condition_l532_53226

theorem necessary_not_sufficient_condition :
  (∀ x : ℝ, x^2 - x < 0 → -1 < x ∧ x < 1) ∧
  (∃ x : ℝ, -1 < x ∧ x < 1 ∧ ¬(x^2 - x < 0)) :=
sorry

end necessary_not_sufficient_condition_l532_53226


namespace expand_expression_l532_53288

theorem expand_expression (x : ℝ) : (7 * x - 3) * (3 * x^2) = 21 * x^3 - 9 * x^2 := by
  sorry

end expand_expression_l532_53288


namespace ship_journey_l532_53258

theorem ship_journey (D : ℝ) (speed : ℝ) (h1 : D > 0) (h2 : speed = 30) :
  D / 2 - 200 = D / 3 →
  D = 1200 ∧ (D / 2) / speed = 20 :=
by sorry

end ship_journey_l532_53258


namespace max_value_quadratic_l532_53279

theorem max_value_quadratic (x y : ℝ) (hx : x > 0) (hy : y > 0) 
  (h : x^2 - 2*x*y + 3*y^2 = 12) : 
  x^2 + 2*x*y + 3*y^2 ≤ 18 + 12*Real.sqrt 3 := by
  sorry

end max_value_quadratic_l532_53279


namespace orange_preference_percentage_l532_53223

/-- The color preferences survey results -/
def color_frequencies : List (String × ℕ) :=
  [("Red", 75), ("Blue", 80), ("Green", 50), ("Yellow", 45), ("Purple", 60), ("Orange", 55)]

/-- The total number of responses in the survey -/
def total_responses : ℕ := (color_frequencies.map (·.2)).sum

/-- Calculate the percentage of respondents who preferred a given color -/
def color_percentage (color : String) : ℚ :=
  match color_frequencies.find? (·.1 = color) with
  | some (_, freq) => (freq : ℚ) / (total_responses : ℚ) * 100
  | none => 0

/-- The theorem stating that the percentage who preferred orange is 15% -/
theorem orange_preference_percentage :
  ⌊color_percentage "Orange"⌋ = 15 := by sorry

end orange_preference_percentage_l532_53223


namespace convex_polygon_four_equal_areas_l532_53211

/-- A convex polygon in 2D space -/
structure ConvexPolygon where
  -- We don't need to define the internal structure of the polygon
  -- for this theorem statement

/-- Represents a line in 2D space -/
structure Line where
  -- We don't need to define the internal structure of the line
  -- for this theorem statement

/-- Represents an area measurement -/
def Area : Type := ℝ

/-- Function to calculate the area of a region of a polygon -/
def areaOfRegion (p : ConvexPolygon) (region : Set (ℝ × ℝ)) : Area :=
  sorry -- Implementation not needed for the theorem statement

/-- Two lines are perpendicular -/
def arePerpendicular (l1 l2 : Line) : Prop :=
  sorry -- Definition not needed for the theorem statement

/-- A line divides a polygon into two regions -/
def dividePolygon (p : ConvexPolygon) (l : Line) : (Set (ℝ × ℝ)) × (Set (ℝ × ℝ)) :=
  sorry -- Implementation not needed for the theorem statement

/-- Theorem: Any convex polygon can be divided into four equal areas by two perpendicular lines -/
theorem convex_polygon_four_equal_areas (p : ConvexPolygon) :
  ∃ (l1 l2 : Line),
    arePerpendicular l1 l2 ∧
    let (r1, r2) := dividePolygon p l1
    let (r11, r12) := dividePolygon p l2
    let a1 := areaOfRegion p (r1 ∩ r11)
    let a2 := areaOfRegion p (r1 ∩ r12)
    let a3 := areaOfRegion p (r2 ∩ r11)
    let a4 := areaOfRegion p (r2 ∩ r12)
    a1 = a2 ∧ a2 = a3 ∧ a3 = a4 :=
  sorry

end convex_polygon_four_equal_areas_l532_53211


namespace unique_four_digit_square_l532_53255

/-- A function that checks if a number is a four-digit number -/
def isFourDigit (n : ℕ) : Prop := 1000 ≤ n ∧ n ≤ 9999

/-- A function that checks if a number has its first two digits equal -/
def firstTwoDigitsEqual (n : ℕ) : Prop :=
  (n / 1000 = (n / 100) % 10)

/-- A function that checks if a number has its last two digits equal -/
def lastTwoDigitsEqual (n : ℕ) : Prop :=
  ((n / 10) % 10 = n % 10)

/-- The main theorem stating that 7744 is the only four-digit perfect square
    with equal first two digits and equal last two digits -/
theorem unique_four_digit_square :
  ∀ n : ℕ, isFourDigit n ∧ ∃ k : ℕ, n = k^2 ∧ firstTwoDigitsEqual n ∧ lastTwoDigitsEqual n
  ↔ n = 7744 := by
  sorry

end unique_four_digit_square_l532_53255


namespace employed_males_percentage_proof_l532_53210

/-- The percentage of the population that is employed -/
def employed_percentage : ℝ := 72

/-- The percentage of employed people who are female -/
def female_employed_percentage : ℝ := 50

/-- The percentage of the population who are employed males -/
def employed_males_percentage : ℝ := 36

theorem employed_males_percentage_proof :
  employed_males_percentage = employed_percentage * (100 - female_employed_percentage) / 100 :=
by sorry

end employed_males_percentage_proof_l532_53210


namespace treewidth_bound_for_grid_free_graphs_l532_53207

/-- A k-grid of order h in a graph -/
def kGridOfOrderH (G : Graph) (k h : ℕ) : Prop := sorry

/-- The treewidth of a graph -/
def treewidth (G : Graph) : ℕ := sorry

/-- Theorem: If a graph G does not contain a k-grid of order h, then its treewidth is less than h + k - 1 -/
theorem treewidth_bound_for_grid_free_graphs
  (G : Graph) (h k : ℕ) (h_ge_k : h ≥ k) (k_ge_1 : k ≥ 1)
  (no_grid : ¬ kGridOfOrderH G k h) :
  treewidth G < h + k - 1 := by
  sorry

end treewidth_bound_for_grid_free_graphs_l532_53207


namespace deaf_students_count_l532_53216

/-- Represents a school for deaf and blind students. -/
structure DeafBlindSchool where
  total_students : ℕ
  deaf_students : ℕ
  blind_students : ℕ
  deaf_triple_blind : deaf_students = 3 * blind_students
  total_sum : total_students = deaf_students + blind_students

/-- Theorem: In a school with 240 total students, where the number of deaf students
    is three times the number of blind students, the number of deaf students is 180. -/
theorem deaf_students_count (school : DeafBlindSchool) 
  (h_total : school.total_students = 240) : school.deaf_students = 180 := by
  sorry

end deaf_students_count_l532_53216


namespace sum_of_ten_numbers_l532_53246

theorem sum_of_ten_numbers (numbers : Finset ℕ) (group_of_ten : Finset ℕ) (group_of_207 : Finset ℕ) :
  numbers = Finset.range 217 →
  numbers = group_of_ten ∪ group_of_207 →
  group_of_ten.card = 10 →
  group_of_207.card = 207 →
  group_of_ten ∩ group_of_207 = ∅ →
  (Finset.sum group_of_ten id) / 10 = (Finset.sum group_of_207 id) / 207 →
  Finset.sum group_of_ten id = 1090 :=
by sorry


end sum_of_ten_numbers_l532_53246


namespace max_candies_eaten_l532_53277

theorem max_candies_eaten (n : ℕ) (h : n = 25) : 
  (n.choose 2) = 300 := by
  sorry

#check max_candies_eaten

end max_candies_eaten_l532_53277


namespace min_value_fraction_l532_53271

theorem min_value_fraction (a b : ℕ) (h1 : a > b) (h2 : b > 0) :
  (2 * a + b : ℚ) / (a - 2 * b) + (a - 2 * b : ℚ) / (2 * a + b) ≥ 50 / 7 :=
sorry

end min_value_fraction_l532_53271


namespace problem_statement_l532_53269

theorem problem_statement (a b c d m : ℝ) 
  (h1 : a + b = 0)
  (h2 : c * d = 1)
  (h3 : |m| = 2) :
  m + c * d + (a + b) / m = 3 ∨ m + c * d + (a + b) / m = -1 := by
sorry

end problem_statement_l532_53269


namespace f_properties_l532_53291

noncomputable def f (x : ℝ) : ℝ := (x^2 - 3) * Real.exp x

theorem f_properties :
  (∃ (x₀ : ℝ), IsLocalMax f x₀) ∧
  (¬ ∃ (M : ℝ), ∀ (x : ℝ), f x ≤ M) ∧
  (∀ (b : ℝ), (∃! (x₁ x₂ x₃ : ℝ), x₁ ≠ x₂ ∧ x₁ ≠ x₃ ∧ x₂ ≠ x₃ ∧ f x₁ = b ∧ f x₂ = b ∧ f x₃ = b) →
    (0 < b ∧ b < 6 * Real.exp (-3))) :=
by sorry

end f_properties_l532_53291


namespace system_solution_unique_l532_53249

theorem system_solution_unique (x y : ℝ) : 
  (4 * x + 3 * y = 11 ∧ 4 * x - 3 * y = 5) ↔ (x = 2 ∧ y = 1) := by
  sorry

end system_solution_unique_l532_53249


namespace watch_correction_theorem_l532_53260

/-- Represents the time loss of a watch in minutes per day -/
def timeLossPerDay : ℚ := 13/4

/-- Represents the number of hours between 4 PM on March 21 and 12 PM on March 28 -/
def totalHours : ℕ := 7 * 24 + 20

/-- Calculates the positive correction in minutes needed for the watch -/
def positiveCorrection : ℚ :=
  (timeLossPerDay * (totalHours : ℚ)) / 24

theorem watch_correction_theorem :
  positiveCorrection = 25 + 17/96 := by sorry

end watch_correction_theorem_l532_53260


namespace incorrect_locus_definition_l532_53264

-- Define the type for points in our space
variable {X : Type*}

-- Define the locus as a set of points
variable (locus : Set X)

-- Define the condition as a predicate on points
variable (condition : X → Prop)

-- Statement to be proven incorrect
theorem incorrect_locus_definition :
  ¬(∀ x : X, condition x → x ∈ locus) ∧
  (∃ x : X, x ∈ locus ∧ condition x) →
  ¬(∀ x : X, x ∈ locus ↔ condition x) :=
by sorry

end incorrect_locus_definition_l532_53264


namespace tomatoes_picked_l532_53228

/-- Calculates the number of tomatoes picked by a farmer -/
theorem tomatoes_picked (initial_tomatoes : ℕ) (initial_potatoes : ℕ) (final_total : ℕ) : 
  initial_tomatoes - (final_total - initial_potatoes) = 53 :=
by
  sorry

end tomatoes_picked_l532_53228


namespace intersection_complement_equals_three_l532_53268

def U : Set Int := {-2, -1, 0, 1, 2, 3}
def M : Set Int := {0, 1, 2}
def N : Set Int := {0, 1, 2, 3}

theorem intersection_complement_equals_three : (U \ M) ∩ N = {3} := by
  sorry

end intersection_complement_equals_three_l532_53268


namespace final_expression_l532_53297

theorem final_expression (b : ℚ) : 
  (3 * b + 6 - 5 * b) / 3 = -2/3 * b + 2 := by sorry

end final_expression_l532_53297


namespace expected_other_marbles_is_two_l532_53273

/-- Represents the distribution of marble colors in Percius's collection -/
structure MarbleCollection where
  clear_percent : ℚ
  black_percent : ℚ
  other_percent : ℚ
  sum_to_one : clear_percent + black_percent + other_percent = 1

/-- Calculates the expected number of marbles of a certain color when taking a sample -/
def expected_marbles (collection : MarbleCollection) (sample_size : ℕ) (color_percent : ℚ) : ℚ :=
  color_percent * sample_size

/-- Theorem: The expected number of other-colored marbles in a sample of 5 is 2 -/
theorem expected_other_marbles_is_two (collection : MarbleCollection) 
    (h1 : collection.clear_percent = 2/5)
    (h2 : collection.black_percent = 1/5) :
    expected_marbles collection 5 collection.other_percent = 2 := by
  sorry

#eval expected_marbles ⟨2/5, 1/5, 2/5, by norm_num⟩ 5 (2/5)

end expected_other_marbles_is_two_l532_53273


namespace marble_distribution_l532_53285

theorem marble_distribution (total_marbles : ℕ) (initial_group : ℕ) (joining_group : ℕ) : 
  total_marbles = 312 →
  initial_group = 24 →
  (total_marbles / initial_group : ℕ) = ((total_marbles / (initial_group + joining_group)) + 1 : ℕ) →
  joining_group = 2 :=
by sorry

end marble_distribution_l532_53285


namespace quadratic_equation_two_distinct_roots_l532_53237

theorem quadratic_equation_two_distinct_roots :
  let a : ℝ := 1
  let b : ℝ := 0
  let c : ℝ := -2
  let Δ : ℝ := b^2 - 4*a*c
  (∀ (a b c : ℝ), (b^2 - 4*a*c > 0) ↔ (∃ (x y : ℝ), x ≠ y ∧ a*x^2 + b*x + c = 0 ∧ a*y^2 + b*y + c = 0)) →
  ∃ (x y : ℝ), x ≠ y ∧ x^2 - 2 = 0 ∧ y^2 - 2 = 0 :=
by sorry

end quadratic_equation_two_distinct_roots_l532_53237


namespace tangent_ellipse_major_axis_length_l532_53205

/-- An ellipse with foci at (3, -4 + 2√3) and (3, -4 - 2√3), tangent to both x and y axes -/
structure TangentEllipse where
  /-- The ellipse is tangent to the x-axis -/
  tangent_x : Bool
  /-- The ellipse is tangent to the y-axis -/
  tangent_y : Bool
  /-- The first focus of the ellipse -/
  focus1 : ℝ × ℝ
  /-- The second focus of the ellipse -/
  focus2 : ℝ × ℝ
  /-- Ensure the foci are at the specified points -/
  foci_constraint : focus1 = (3, -4 + 2 * Real.sqrt 3) ∧ focus2 = (3, -4 - 2 * Real.sqrt 3)
  /-- Ensure the ellipse is tangent to both axes -/
  tangent_constraint : tangent_x ∧ tangent_y

/-- The length of the major axis of the ellipse -/
def majorAxisLength (e : TangentEllipse) : ℝ := 8

/-- Theorem stating that the major axis length of the specified ellipse is 8 -/
theorem tangent_ellipse_major_axis_length (e : TangentEllipse) : 
  majorAxisLength e = 8 := by sorry

end tangent_ellipse_major_axis_length_l532_53205


namespace proposition_count_l532_53262

theorem proposition_count : 
  (∃ (correct : Finset (Fin 6)) (h : correct.card = 5),
    (∀ i : Fin 6, i ∈ correct ↔
      (i = 0 ∧ ∀ a b : ℝ, b < a ∧ a < 0 → |a| > |b|) ∨
      (i = 1 ∧ ∀ a b : ℝ, b < a ∧ a < 0 → a + b < a * b) ∨
      (i = 2 ∧ ∀ a b : ℝ, b < a ∧ a < 0 → b / a + a / b > 2) ∨
      (i = 3 ∧ ∀ a b : ℝ, b < a ∧ a < 0 → a^2 / b < 2 * a - b) ∨
      (i = 4 ∧ ∀ a b : ℝ, b < a ∧ a < 0 → (2 * a + b) / (a + 2 * b) > a / b) ∨
      (i = 5 ∧ ∀ a b : ℝ, a + b = 1 → a^2 + b^2 ≥ 1 / 2))) :=
by sorry

end proposition_count_l532_53262


namespace parabola_equation_l532_53250

/-- The equation of a parabola with focus (2, 0) and directrix x + 2 = 0 -/
theorem parabola_equation :
  ∀ (x y : ℝ),
    (∃ (P : ℝ × ℝ), P.1 = x ∧ P.2 = y) →
    (∀ (P : ℝ × ℝ), P.1 = x ∧ P.2 = y →
      (P.1 - 2)^2 + P.2^2 = (P.1 + 2)^2) ↔
    y^2 = 8*x :=
by sorry

end parabola_equation_l532_53250


namespace total_sales_revenue_marie_sales_revenue_l532_53204

/-- Calculates the total sales revenue from selling magazines and newspapers -/
theorem total_sales_revenue 
  (magazines_sold : ℕ) 
  (newspapers_sold : ℕ) 
  (magazine_price : ℚ) 
  (newspaper_price : ℚ) : ℚ :=
  magazines_sold * magazine_price + newspapers_sold * newspaper_price

/-- Proves that the total sales revenue for the given quantities and prices is correct -/
theorem marie_sales_revenue : 
  total_sales_revenue 425 275 (35/10) (5/4) = 1831.25 := by
  sorry

end total_sales_revenue_marie_sales_revenue_l532_53204


namespace right_triangle_dot_product_l532_53278

/-- Given a right triangle ABC with ∠ABC = 90°, AB = 4, and BC = 3, 
    prove that the dot product of AC and BC is 9. -/
theorem right_triangle_dot_product (A B C : ℝ × ℝ) : 
  (B.1 - A.1)^2 + (B.2 - A.2)^2 = 4^2 →  -- AB = 4
  (C.1 - B.1)^2 + (C.2 - B.2)^2 = 3^2 →  -- BC = 3
  (B.1 - A.1) * (C.1 - B.1) + (B.2 - A.2) * (C.2 - B.2) = 0 →  -- ∠ABC = 90°
  ((C.1 - A.1) * (C.1 - B.1) + (C.2 - A.2) * (C.2 - B.2)) = 9 := by
sorry

end right_triangle_dot_product_l532_53278


namespace factorial_difference_l532_53275

theorem factorial_difference : Nat.factorial 10 - Nat.factorial 9 = 3265920 := by
  sorry

end factorial_difference_l532_53275


namespace min_distance_MN_min_distance_is_two_l532_53253

noncomputable section

def f (x : ℝ) : ℝ := Real.exp x - (1/2) * x^2
def g (x : ℝ) : ℝ := x - 1

def M (x₁ : ℝ) : ℝ × ℝ := (x₁, f x₁)
def N (x₂ : ℝ) : ℝ × ℝ := (x₂, g x₂)

theorem min_distance_MN (x₁ x₂ : ℝ) (h₁ : x₁ ≥ 0) (h₂ : x₂ > 0) 
  (h₃ : f x₁ = g x₂) : 
  ∀ y₁ y₂ : ℝ, y₁ ≥ 0 → y₂ > 0 → f y₁ = g y₂ → 
  |x₂ - x₁| ≤ |y₂ - y₁| := by sorry

theorem min_distance_is_two (x₁ x₂ : ℝ) (h₁ : x₁ ≥ 0) (h₂ : x₂ > 0) 
  (h₃ : f x₁ = g x₂) : 
  |x₂ - x₁| = 2 := by sorry

end min_distance_MN_min_distance_is_two_l532_53253


namespace min_value_expression_l532_53225

theorem min_value_expression (x y z : ℝ) (hpos : x > 0 ∧ y > 0 ∧ z > 0) 
  (hsum : x^2 + y^2 + z^2 = 1) :
  (x*y/z + y*z/x + z*x/y) ≥ Real.sqrt 3 ∧ 
  ∃ (a b c : ℝ), a > 0 ∧ b > 0 ∧ c > 0 ∧ a^2 + b^2 + c^2 = 1 ∧ 
    a*b/c + b*c/a + c*a/b = Real.sqrt 3 := by
  sorry

end min_value_expression_l532_53225
