import Mathlib

namespace NUMINAMATH_CALUDE_markers_problem_l701_70163

theorem markers_problem (initial_markers : ℕ) (markers_per_box : ℕ) (final_markers : ℕ) :
  initial_markers = 32 →
  markers_per_box = 9 →
  final_markers = 86 →
  (final_markers - initial_markers) / markers_per_box = 6 :=
by
  sorry

end NUMINAMATH_CALUDE_markers_problem_l701_70163


namespace NUMINAMATH_CALUDE_rachel_lunch_spending_l701_70141

theorem rachel_lunch_spending (initial_amount : ℝ) 
  (h1 : initial_amount = 200)
  (h2 : ∃ dvd_amount : ℝ, dvd_amount = initial_amount / 2)
  (h3 : ∃ amount_left : ℝ, amount_left = 50) :
  ∃ lunch_fraction : ℝ, lunch_fraction = 1 / 4 := by
  sorry

end NUMINAMATH_CALUDE_rachel_lunch_spending_l701_70141


namespace NUMINAMATH_CALUDE_last_two_digits_1976_power_100_l701_70182

theorem last_two_digits_1976_power_100 : 
  1976^100 % 100 = 76 := by sorry

end NUMINAMATH_CALUDE_last_two_digits_1976_power_100_l701_70182


namespace NUMINAMATH_CALUDE_right_triangle_with_consecutive_legs_l701_70188

theorem right_triangle_with_consecutive_legs (a b c : ℕ) : 
  a^2 + b^2 = c^2 →  -- Pythagorean theorem
  b = a + 1 →        -- Consecutive whole number legs
  c = 41 →           -- Hypotenuse is 41 units
  a + b = 57 :=      -- Sum of leg lengths is 57
by sorry

end NUMINAMATH_CALUDE_right_triangle_with_consecutive_legs_l701_70188


namespace NUMINAMATH_CALUDE_original_equals_scientific_l701_70146

/-- Scientific notation representation of a number -/
structure ScientificNotation where
  mantissa : ℝ
  exponent : ℤ
  mantissa_bounds : 1 ≤ mantissa ∧ mantissa < 10

/-- The number to be expressed in scientific notation -/
def original_number : ℕ := 274000000

/-- The scientific notation representation of the original number -/
def scientific_form : ScientificNotation :=
  { mantissa := 2.74
  , exponent := 8
  , mantissa_bounds := by sorry }

/-- Theorem stating that the original number is equal to its scientific notation representation -/
theorem original_equals_scientific :
  (original_number : ℝ) = scientific_form.mantissa * (10 : ℝ) ^ scientific_form.exponent := by
  sorry

end NUMINAMATH_CALUDE_original_equals_scientific_l701_70146


namespace NUMINAMATH_CALUDE_trajectory_of_point_on_moving_segment_l701_70105

/-- The trajectory of a point M on a moving line segment AB -/
theorem trajectory_of_point_on_moving_segment (A B M : ℝ × ℝ) 
  (h_length : (A.1 - B.1)^2 + (A.2 - B.2)^2 = 4)
  (h_A_on_x : A.2 = 0)
  (h_B_on_y : B.1 = 0)
  (h_M_on_AB : ∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ M = (1 - t) • A + t • B)
  (h_ratio : ∃ k : ℝ, k > 0 ∧ 
    (M.1 - A.1)^2 + (M.2 - A.2)^2 = k^2 * ((B.1 - M.1)^2 + (B.2 - M.2)^2) ∧
    k = 1/2) :
  9 * M.1^2 + 36 * M.2^2 = 16 := by
  sorry

end NUMINAMATH_CALUDE_trajectory_of_point_on_moving_segment_l701_70105


namespace NUMINAMATH_CALUDE_remainder_6n_mod_4_l701_70195

theorem remainder_6n_mod_4 (n : ℤ) (h : n % 4 = 3) : (6 * n) % 4 = 2 := by
  sorry

end NUMINAMATH_CALUDE_remainder_6n_mod_4_l701_70195


namespace NUMINAMATH_CALUDE_jelly_bracelet_cost_l701_70161

def friends : List String := ["Jessica", "Tori", "Lily", "Patrice"]

def total_spent : ℕ := 44

def total_bracelets : ℕ := (friends.map String.length).sum

theorem jelly_bracelet_cost :
  total_spent / total_bracelets = 2 := by sorry

end NUMINAMATH_CALUDE_jelly_bracelet_cost_l701_70161


namespace NUMINAMATH_CALUDE_quadratic_integer_root_existence_l701_70197

/-- A quadratic polynomial with integer coefficients -/
structure QuadraticPolynomial where
  a : ℕ
  b : ℕ
  c : ℕ

/-- Check if a quadratic polynomial has an integer root -/
def has_integer_root (p : QuadraticPolynomial) : Prop :=
  ∃ x : ℤ, p.a * x * x + p.b * x + p.c = 0

/-- Calculate the cost of changing from one polynomial to another -/
def change_cost (p q : QuadraticPolynomial) : ℕ :=
  (Int.natAbs (p.a - q.a)) + (Int.natAbs (p.b - q.b)) + (Int.natAbs (p.c - q.c))

/-- The main theorem -/
theorem quadratic_integer_root_existence (p : QuadraticPolynomial) 
    (h : p.a + p.b + p.c = 2000) :
    ∃ q : QuadraticPolynomial, has_integer_root q ∧ change_cost p q ≤ 1022 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_integer_root_existence_l701_70197


namespace NUMINAMATH_CALUDE_quotient_of_composites_l701_70100

def first_five_even_composites : List Nat := [4, 6, 8, 10, 12]
def next_five_odd_composites : List Nat := [9, 15, 21, 25, 27]

def product_even : Nat := first_five_even_composites.prod
def product_odd : Nat := next_five_odd_composites.prod

theorem quotient_of_composites :
  (product_even : ℚ) / (product_odd : ℚ) = 512 / 28525 := by
  sorry

end NUMINAMATH_CALUDE_quotient_of_composites_l701_70100


namespace NUMINAMATH_CALUDE_corner_circle_radius_l701_70160

/-- The radius of a circle placed tangentially to four corner circles in a specific rectangle configuration -/
theorem corner_circle_radius (rectangle_width : ℝ) (rectangle_length : ℝ) 
  (h_width : rectangle_width = 3)
  (h_length : rectangle_length = 4)
  (large_circle_radius : ℝ)
  (h_large_radius : large_circle_radius = 2/3)
  (small_circle_radius : ℝ) :
  small_circle_radius = 1 :=
sorry

end NUMINAMATH_CALUDE_corner_circle_radius_l701_70160


namespace NUMINAMATH_CALUDE_slices_left_over_l701_70175

/-- The number of initial pizza slices -/
def initial_slices : ℕ := 34

/-- The number of slices eaten by Dean -/
def dean_slices : ℕ := 7

/-- The number of slices eaten by Frank -/
def frank_slices : ℕ := 3

/-- The number of slices eaten by Sammy -/
def sammy_slices : ℕ := 4

/-- The number of slices eaten by Nancy -/
def nancy_slices : ℕ := 3

/-- The number of slices eaten by Olivia -/
def olivia_slices : ℕ := 3

/-- The total number of slices eaten -/
def total_eaten : ℕ := dean_slices + frank_slices + sammy_slices + nancy_slices + olivia_slices

/-- Theorem: The number of pizza slices left over is 14 -/
theorem slices_left_over : initial_slices - total_eaten = 14 := by
  sorry

end NUMINAMATH_CALUDE_slices_left_over_l701_70175


namespace NUMINAMATH_CALUDE_triple_sharp_72_l701_70170

-- Define the # operation
def sharp (N : ℝ) : ℝ := 0.4 * N + 3

-- Theorem statement
theorem triple_sharp_72 : sharp (sharp (sharp 72)) = 9.288 := by
  sorry

end NUMINAMATH_CALUDE_triple_sharp_72_l701_70170


namespace NUMINAMATH_CALUDE_t_shape_area_is_12_l701_70102

def square_area (side : ℝ) : ℝ := side * side

def t_shape_area (outer_side : ℝ) (inner_side1 : ℝ) (inner_side2 : ℝ) : ℝ :=
  square_area outer_side - (2 * square_area inner_side1 + square_area inner_side2)

theorem t_shape_area_is_12 :
  t_shape_area 6 2 4 = 12 := by sorry

end NUMINAMATH_CALUDE_t_shape_area_is_12_l701_70102


namespace NUMINAMATH_CALUDE_sqrt_product_equality_l701_70183

theorem sqrt_product_equality : Real.sqrt 2 * Real.sqrt 3 = Real.sqrt 6 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_product_equality_l701_70183


namespace NUMINAMATH_CALUDE_pet_store_cages_l701_70157

theorem pet_store_cages (initial_puppies : ℕ) (sold_puppies : ℕ) (puppies_per_cage : ℕ) 
  (h1 : initial_puppies = 18)
  (h2 : sold_puppies = 3)
  (h3 : puppies_per_cage = 5) :
  (initial_puppies - sold_puppies) / puppies_per_cage = 3 :=
by
  sorry

end NUMINAMATH_CALUDE_pet_store_cages_l701_70157


namespace NUMINAMATH_CALUDE_square_perimeter_problem_l701_70198

theorem square_perimeter_problem (M N : Real) (h1 : M = 100) (h2 : N = 4 * M) :
  4 * Real.sqrt N = 80 := by
  sorry

end NUMINAMATH_CALUDE_square_perimeter_problem_l701_70198


namespace NUMINAMATH_CALUDE_a_6_value_l701_70178

def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = a n * r

theorem a_6_value
  (a : ℕ → ℝ)
  (h_geo : geometric_sequence a)
  (h_roots : a 4 * a 8 = 9 ∧ a 4 + a 8 = -11) :
  a 6 = -3 := by
sorry

end NUMINAMATH_CALUDE_a_6_value_l701_70178


namespace NUMINAMATH_CALUDE_two_true_propositions_l701_70130

-- Define the types for planes and lines
def Plane : Type := Unit
def Line : Type := Unit

-- Define the parallel and perpendicular relations
def parallel : Plane → Plane → Prop := sorry
def perpendicular : Plane → Plane → Prop := sorry
def parallel_line_plane : Line → Plane → Prop := sorry
def perpendicular_line_plane : Line → Plane → Prop := sorry
def parallel_lines : Line → Line → Prop := sorry
def perpendicular_lines : Line → Line → Prop := sorry

-- Define the original proposition for planes
def original_proposition (α β γ : Plane) : Prop :=
  parallel α β ∧ perpendicular α γ → perpendicular β γ

-- Define the propositions with two planes replaced by lines
def prop_αβ_lines (a b : Line) (γ : Plane) : Prop :=
  parallel_lines a b ∧ perpendicular_line_plane a γ → perpendicular_line_plane b γ

def prop_αγ_lines (a : Line) (β : Plane) (b : Line) : Prop :=
  parallel_line_plane a β ∧ perpendicular_lines a b → perpendicular_line_plane b β

def prop_βγ_lines (α : Plane) (a b : Line) : Prop :=
  parallel_line_plane a α ∧ perpendicular_line_plane α b → perpendicular_lines a b

-- The main theorem
theorem two_true_propositions :
  ∃ (α β γ : Plane),
    original_proposition α β γ = true ∧
    (∀ (a b : Line),
      (prop_αβ_lines a b γ = true ∧ prop_αγ_lines a β b = false ∧ prop_βγ_lines α a b = true) ∨
      (prop_αβ_lines a b γ = true ∧ prop_αγ_lines a β b = true ∧ prop_βγ_lines α a b = false) ∨
      (prop_αβ_lines a b γ = false ∧ prop_αγ_lines a β b = true ∧ prop_βγ_lines α a b = true)) :=
sorry

end NUMINAMATH_CALUDE_two_true_propositions_l701_70130


namespace NUMINAMATH_CALUDE_hyperbola_point_distance_l701_70126

/-- A point on a hyperbola with a specific distance to a line has a specific distance to another line --/
theorem hyperbola_point_distance (m n : ℝ) : 
  m^2 - n^2 = 9 →                        -- P(m, n) is on the hyperbola x^2 - y^2 = 9
  (|m + n| / Real.sqrt 2) = 2016 →       -- Distance from P to y = -x is 2016
  (|m - n| / Real.sqrt 2) = 448 :=       -- Distance from P to y = x is 448
by sorry

end NUMINAMATH_CALUDE_hyperbola_point_distance_l701_70126


namespace NUMINAMATH_CALUDE_prime_product_sum_squared_sum_l701_70139

theorem prime_product_sum_squared_sum (a b c : ℕ) : 
  Prime a ∧ Prime b ∧ Prime c ∧ 
  a ≠ b ∧ b ≠ c ∧ a ≠ c ∧
  a * b * c = 5 * (a + b + c) →
  a^2 + b^2 + c^2 = 78 :=
by sorry

end NUMINAMATH_CALUDE_prime_product_sum_squared_sum_l701_70139


namespace NUMINAMATH_CALUDE_used_car_seller_problem_l701_70103

theorem used_car_seller_problem (num_clients : ℕ) (cars_per_client : ℕ) (selections_per_car : ℕ) 
  (h1 : num_clients = 24)
  (h2 : cars_per_client = 2)
  (h3 : selections_per_car = 3) :
  (num_clients * cars_per_client) / selections_per_car = 16 := by
  sorry

end NUMINAMATH_CALUDE_used_car_seller_problem_l701_70103


namespace NUMINAMATH_CALUDE_day_of_week_N_minus_1_l701_70166

/-- Represents days of the week -/
inductive DayOfWeek
  | Sunday
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday
  | Saturday

/-- Represents a specific day in a year -/
structure YearDay where
  year : Int
  dayNumber : Nat

/-- Function to determine the day of the week for a given YearDay -/
def dayOfWeek (yd : YearDay) : DayOfWeek :=
  sorry

theorem day_of_week_N_minus_1 
  (N : Int)
  (h1 : dayOfWeek ⟨N, 250⟩ = DayOfWeek.Friday)
  (h2 : dayOfWeek ⟨N+1, 150⟩ = DayOfWeek.Friday) :
  dayOfWeek ⟨N-1, 250⟩ = DayOfWeek.Saturday :=
sorry

end NUMINAMATH_CALUDE_day_of_week_N_minus_1_l701_70166


namespace NUMINAMATH_CALUDE_similar_triangles_leg_sum_l701_70106

theorem similar_triangles_leg_sum (area_small area_large hyp_small : ℝ) 
  (h1 : area_small = 10)
  (h2 : area_large = 250)
  (h3 : hyp_small = 13)
  (h4 : area_small > 0)
  (h5 : area_large > 0)
  (h6 : hyp_small > 0) :
  ∃ (leg1_small leg2_small leg1_large leg2_large : ℝ),
    leg1_small^2 + leg2_small^2 = hyp_small^2 ∧
    leg1_small * leg2_small / 2 = area_small ∧
    leg1_large^2 + leg2_large^2 = (hyp_small * (area_large / area_small).sqrt)^2 ∧
    leg1_large * leg2_large / 2 = area_large ∧
    leg1_large + leg2_large = 35 := by
sorry


end NUMINAMATH_CALUDE_similar_triangles_leg_sum_l701_70106


namespace NUMINAMATH_CALUDE_complex_conjugate_sum_l701_70152

theorem complex_conjugate_sum (α β : ℝ) :
  2 * Complex.exp (Complex.I * α) + 2 * Complex.exp (Complex.I * β) = -1/2 + 4/5 * Complex.I →
  2 * Complex.exp (-Complex.I * α) + 2 * Complex.exp (-Complex.I * β) = -1/2 - 4/5 * Complex.I :=
by sorry

end NUMINAMATH_CALUDE_complex_conjugate_sum_l701_70152


namespace NUMINAMATH_CALUDE_min_sum_of_digits_prime_l701_70133

def f (n : ℕ) : ℕ := n^2 - 69*n + 2250

def sum_of_digits (n : ℕ) : ℕ :=
  if n < 10 then n else (n % 10) + sum_of_digits (n / 10)

def is_prime (p : ℕ) : Prop :=
  p > 1 ∧ ∀ m : ℕ, m > 1 → m < p → ¬(p % m = 0)

theorem min_sum_of_digits_prime :
  ∃ (p : ℕ), is_prime p ∧
    ∀ (q : ℕ), is_prime q →
      sum_of_digits (f (p^2 + 32)) ≤ sum_of_digits (f (q^2 + 32)) ∧
      p = 3 :=
sorry

end NUMINAMATH_CALUDE_min_sum_of_digits_prime_l701_70133


namespace NUMINAMATH_CALUDE_spherical_distance_for_pi_over_six_l701_70167

/-- The spherical distance between two points on a sphere's surface -/
def spherical_distance (R : ℝ) (angle : ℝ) : ℝ := R * angle

/-- Theorem: The spherical distance between two points A and B on a sphere with radius R,
    where the angle AOB is π/6, is equal to (π/6)R -/
theorem spherical_distance_for_pi_over_six (R : ℝ) (h : R > 0) :
  spherical_distance R (π/6) = (π/6) * R := by sorry

end NUMINAMATH_CALUDE_spherical_distance_for_pi_over_six_l701_70167


namespace NUMINAMATH_CALUDE_negative_fractions_comparison_l701_70180

theorem negative_fractions_comparison : -2/3 < -1/2 := by sorry

end NUMINAMATH_CALUDE_negative_fractions_comparison_l701_70180


namespace NUMINAMATH_CALUDE_downstream_distance_l701_70174

/-- Calculates the distance traveled downstream by a boat -/
theorem downstream_distance 
  (boat_speed : ℝ) 
  (stream_speed : ℝ) 
  (time : ℝ) 
  (h1 : boat_speed = 30) 
  (h2 : stream_speed = 5) 
  (h3 : time = 2) : 
  boat_speed + stream_speed * time = 70 := by
  sorry

#check downstream_distance

end NUMINAMATH_CALUDE_downstream_distance_l701_70174


namespace NUMINAMATH_CALUDE_minimize_f_minimum_l701_70179

/-- The function f(x) defined in the problem -/
def f (a : ℝ) (x : ℝ) : ℝ :=
  |7*x - 3*a + 8| + |5*x + 4*a - 6| + |x - a - 8| - 24

/-- Theorem stating that 82/43 is the value of a that minimizes the minimum value of f(x) -/
theorem minimize_f_minimum (a : ℝ) :
  (∀ x, f (82/43) x ≤ f a x) ∧ (∃ x, f (82/43) x < f a x) ∨ a = 82/43 := by
  sorry

#check minimize_f_minimum

end NUMINAMATH_CALUDE_minimize_f_minimum_l701_70179


namespace NUMINAMATH_CALUDE_course_selection_count_l701_70120

/-- The number of different course selection schemes for 3 students choosing from 3 elective courses -/
def course_selection_schemes : ℕ := 18

/-- The number of elective courses -/
def num_courses : ℕ := 3

/-- The number of students -/
def num_students : ℕ := 3

/-- Proposition that each student chooses only one course -/
axiom one_course_per_student : True

/-- Proposition that exactly one course has no students -/
axiom one_empty_course : True

/-- Theorem stating that the number of different course selection schemes is 18 -/
theorem course_selection_count : course_selection_schemes = 18 := by
  sorry

end NUMINAMATH_CALUDE_course_selection_count_l701_70120


namespace NUMINAMATH_CALUDE_ship_travel_ratio_l701_70194

/-- Proves that the ratio of the distance traveled on day 2 to day 1 is 3:1 given the ship's travel conditions --/
theorem ship_travel_ratio : 
  ∀ (day1_distance day2_distance day3_distance : ℝ),
  day1_distance = 100 →
  day3_distance = day2_distance + 110 →
  day1_distance + day2_distance + day3_distance = 810 →
  day2_distance / day1_distance = 3 := by
sorry


end NUMINAMATH_CALUDE_ship_travel_ratio_l701_70194


namespace NUMINAMATH_CALUDE_tan_4050_undefined_l701_70122

theorem tan_4050_undefined : ¬∃ (x : ℝ), Real.tan (4050 * Real.pi / 180) = x := by
  sorry

end NUMINAMATH_CALUDE_tan_4050_undefined_l701_70122


namespace NUMINAMATH_CALUDE_intersection_and_union_when_a_is_neg_one_intersection_equals_B_iff_l701_70168

-- Define sets A and B
def A : Set ℝ := {x | x^2 - 4*x - 5 ≥ 0}
def B (a : ℝ) : Set ℝ := {x | 2*a ≤ x ∧ x ≤ a + 2}

-- Theorem for part 1
theorem intersection_and_union_when_a_is_neg_one :
  (A ∩ B (-1)) = {x | -2 ≤ x ∧ x ≤ -1} ∧
  (A ∪ B (-1)) = {x | x ≤ 1 ∨ x ≥ 5} := by sorry

-- Theorem for part 2
theorem intersection_equals_B_iff :
  ∀ a : ℝ, (A ∩ B a = B a) ↔ (a > 2 ∨ a ≤ -3) := by sorry

end NUMINAMATH_CALUDE_intersection_and_union_when_a_is_neg_one_intersection_equals_B_iff_l701_70168


namespace NUMINAMATH_CALUDE_credit_card_balance_l701_70199

theorem credit_card_balance (B : ℝ) : 
  (1.44 * B + 24 = 96) → B = 50 := by
sorry

end NUMINAMATH_CALUDE_credit_card_balance_l701_70199


namespace NUMINAMATH_CALUDE_no_solution_factorial_equation_l701_70107

theorem no_solution_factorial_equation :
  ∀ (m n : ℕ), m.factorial + 48 ≠ 48 * (m + 1) * n := by
  sorry

end NUMINAMATH_CALUDE_no_solution_factorial_equation_l701_70107


namespace NUMINAMATH_CALUDE_passengers_off_north_carolina_l701_70124

/-- Represents the number of passengers at different stages of the flight --/
structure FlightPassengers where
  initial : ℕ
  offTexas : ℕ
  onTexas : ℕ
  onNorthCarolina : ℕ
  crew : ℕ
  landedVirginia : ℕ

/-- Calculates the number of passengers who got off in North Carolina --/
def passengersOffNorthCarolina (fp : FlightPassengers) : ℕ :=
  fp.initial - fp.offTexas + fp.onTexas - (fp.landedVirginia - fp.crew - fp.onNorthCarolina)

/-- Theorem stating that 47 passengers got off in North Carolina --/
theorem passengers_off_north_carolina :
  let fp : FlightPassengers := {
    initial := 124,
    offTexas := 58,
    onTexas := 24,
    onNorthCarolina := 14,
    crew := 10,
    landedVirginia := 67
  }
  passengersOffNorthCarolina fp = 47 := by
  sorry


end NUMINAMATH_CALUDE_passengers_off_north_carolina_l701_70124


namespace NUMINAMATH_CALUDE_circle_center_coordinates_l701_70132

/-- The equation of a circle in the form (x - h)^2 + (y - k)^2 = r^2 --/
def CircleEquation (h k r : ℝ) : ℝ × ℝ → Prop :=
  λ p => (p.1 - h)^2 + (p.2 - k)^2 = r^2

/-- The center of a circle given by its equation --/
def CircleCenter (eq : ℝ × ℝ → Prop) : ℝ × ℝ := sorry

theorem circle_center_coordinates :
  CircleCenter (CircleEquation 1 2 1) = (1, 2) := by sorry

end NUMINAMATH_CALUDE_circle_center_coordinates_l701_70132


namespace NUMINAMATH_CALUDE_corn_ratio_proof_l701_70148

theorem corn_ratio_proof (marcel_corn : ℕ) (marcel_potatoes : ℕ) (dale_potatoes : ℕ) (total_vegetables : ℕ) :
  marcel_corn = 10 →
  marcel_potatoes = 4 →
  dale_potatoes = 8 →
  total_vegetables = 27 →
  ∃ (dale_corn : ℕ), 
    marcel_corn + marcel_potatoes + dale_corn + dale_potatoes = total_vegetables ∧
    dale_corn * 2 = marcel_corn :=
by sorry

end NUMINAMATH_CALUDE_corn_ratio_proof_l701_70148


namespace NUMINAMATH_CALUDE_negation_of_forall_geq_zero_l701_70138

theorem negation_of_forall_geq_zero :
  (¬ ∀ x : ℝ, 2 * x + 4 ≥ 0) ↔ (∃ x : ℝ, 2 * x + 4 < 0) := by
  sorry

end NUMINAMATH_CALUDE_negation_of_forall_geq_zero_l701_70138


namespace NUMINAMATH_CALUDE_quadratic_equations_solutions_l701_70129

theorem quadratic_equations_solutions :
  (∃ x : ℝ, x^2 - 4*x + 3 = 0) ∧ 
  (∃ x : ℝ, x*(x-2) = 2*(2-x)) ∧
  (∀ x : ℝ, x^2 - 4*x + 3 = 0 ↔ (x = 3 ∨ x = 1)) ∧
  (∀ x : ℝ, x*(x-2) = 2*(2-x) ↔ (x = 2 ∨ x = -2)) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equations_solutions_l701_70129


namespace NUMINAMATH_CALUDE_expression_equals_forty_l701_70125

theorem expression_equals_forty : (20 - (2010 - 201)) + (2010 - (201 - 20)) = 40 := by
  sorry

end NUMINAMATH_CALUDE_expression_equals_forty_l701_70125


namespace NUMINAMATH_CALUDE_fuel_price_per_gallon_l701_70112

/-- Given the following conditions:
    1. The total amount of fuel is 100 gallons
    2. The fuel consumption rate is $0.40 worth of fuel per hour
    3. It takes 175 hours to consume all the fuel
    Prove that the price per gallon of fuel is $0.70 -/
theorem fuel_price_per_gallon 
  (total_fuel : ℝ) 
  (consumption_rate : ℝ) 
  (total_hours : ℝ) 
  (h1 : total_fuel = 100) 
  (h2 : consumption_rate = 0.40) 
  (h3 : total_hours = 175) : 
  (consumption_rate * total_hours) / total_fuel = 0.70 := by
  sorry

#check fuel_price_per_gallon

end NUMINAMATH_CALUDE_fuel_price_per_gallon_l701_70112


namespace NUMINAMATH_CALUDE_largest_interesting_is_max_l701_70101

/-- A natural number is interesting if all its digits, except for the first and last,
    are less than the arithmetic mean of their two neighboring digits. -/
def is_interesting (n : ℕ) : Prop :=
  let digits := n.digits 10
  ∀ i, 1 < i ∧ i < digits.length - 1 →
    digits[i]! < (digits[i-1]! + digits[i+1]!) / 2

/-- The largest interesting number -/
def largest_interesting : ℕ := 96433469

theorem largest_interesting_is_max :
  is_interesting largest_interesting ∧
  ∀ n : ℕ, is_interesting n → n ≤ largest_interesting :=
sorry

end NUMINAMATH_CALUDE_largest_interesting_is_max_l701_70101


namespace NUMINAMATH_CALUDE_gcd_problems_l701_70165

theorem gcd_problems :
  (Nat.gcd 840 1764 = 84) ∧ (Nat.gcd 459 357 = 51) := by
  sorry

end NUMINAMATH_CALUDE_gcd_problems_l701_70165


namespace NUMINAMATH_CALUDE_min_value_expression_min_value_is_17_2_l701_70184

theorem min_value_expression (a b : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : a + 2 * b = 1) :
  ∀ x y : ℝ, x > 0 → y > 0 → x + 2 * y = 1 → 
    a^2 + 4 * b^2 + 1 / (a * b) ≤ x^2 + 4 * y^2 + 1 / (x * y) :=
by sorry

theorem min_value_is_17_2 (a b : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : a + 2 * b = 1) :
  a^2 + 4 * b^2 + 1 / (a * b) = 17 / 2 :=
by sorry

end NUMINAMATH_CALUDE_min_value_expression_min_value_is_17_2_l701_70184


namespace NUMINAMATH_CALUDE_system_equation_solution_range_l701_70115

theorem system_equation_solution_range (x y m : ℝ) : 
  (3 * x + y = m - 1) → 
  (x - 3 * y = 2 * m) → 
  (x + 2 * y ≥ 0) → 
  (m ≤ -1) := by
sorry

end NUMINAMATH_CALUDE_system_equation_solution_range_l701_70115


namespace NUMINAMATH_CALUDE_distinct_collections_biology_l701_70111

def Word : Type := List Char

def isVowel (c : Char) : Bool :=
  c ∈ ['A', 'E', 'I', 'O', 'U', 'Y']

def isConsonant (c : Char) : Bool :=
  c ∉ ['A', 'E', 'I', 'O', 'U', 'Y']

def biology : Word := ['B', 'I', 'O', 'L', 'O', 'G', 'Y']

def indistinguishable (c : Char) : Bool :=
  c ∈ ['B', 'I', 'G']

def Collection := List Char

def isValidCollection (c : Collection) : Bool :=
  c.length = 6 ∧ 
  (c.filter isVowel).length = 3 ∧
  (c.filter isConsonant).length = 3

def distinctCollections (w : Word) : Finset Collection :=
  sorry

theorem distinct_collections_biology :
  (distinctCollections biology).card = 2 :=
sorry

end NUMINAMATH_CALUDE_distinct_collections_biology_l701_70111


namespace NUMINAMATH_CALUDE_train_meeting_point_train_A_distance_l701_70159

theorem train_meeting_point (total_distance : ℝ) (time_A time_B : ℝ) (h1 : total_distance = 75) 
  (h2 : time_A = 3) (h3 : time_B = 2) : ℝ :=
  let speed_A := total_distance / time_A
  let speed_B := total_distance / time_B
  let relative_speed := speed_A + speed_B
  let meeting_time := total_distance / relative_speed
  speed_A * meeting_time

theorem train_A_distance (total_distance : ℝ) (time_A time_B : ℝ) (h1 : total_distance = 75) 
  (h2 : time_A = 3) (h3 : time_B = 2) : 
  train_meeting_point total_distance time_A time_B h1 h2 h3 = 30 := by
  sorry

end NUMINAMATH_CALUDE_train_meeting_point_train_A_distance_l701_70159


namespace NUMINAMATH_CALUDE_cost_difference_l701_70186

theorem cost_difference (alice_paid bob_paid charlie_paid dana_paid : ℝ)
  (h1 : alice_paid = 120)
  (h2 : bob_paid = 150)
  (h3 : charlie_paid = 180)
  (h4 : dana_paid = 200)
  (h5 : ∀ person_share, person_share = (alice_paid + bob_paid + charlie_paid + dana_paid) / 4) :
  person_share - alice_paid - (person_share - bob_paid) = 30 := by
  sorry

end NUMINAMATH_CALUDE_cost_difference_l701_70186


namespace NUMINAMATH_CALUDE_power_function_through_point_l701_70127

/-- A power function passing through a specific point -/
def isPowerFunctionThroughPoint (m : ℝ) : Prop :=
  ∃ (y : ℝ → ℝ), (∀ x, y x = (m^2 - 3*m + 3) * x^m) ∧ y 2 = 4

/-- The value of m for which the power function passes through (2, 4) -/
theorem power_function_through_point :
  ∃ (m : ℝ), isPowerFunctionThroughPoint m ∧ m = 2 := by
  sorry

end NUMINAMATH_CALUDE_power_function_through_point_l701_70127


namespace NUMINAMATH_CALUDE_expand_expression_l701_70149

theorem expand_expression (x : ℝ) : 12 * (3 * x - 4) = 36 * x - 48 := by
  sorry

end NUMINAMATH_CALUDE_expand_expression_l701_70149


namespace NUMINAMATH_CALUDE_quadratic_inequality_all_reals_l701_70123

theorem quadratic_inequality_all_reals (a b c : ℝ) :
  (∀ x : ℝ, -a/3 * x^2 + 2*b*x - c < 0) ↔ (a > 0 ∧ 4*b^2 - 4/3*a*c < 0) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_all_reals_l701_70123


namespace NUMINAMATH_CALUDE_range_of_m_when_g_has_three_zeros_l701_70150

/-- The quadratic function f(x) -/
def f (x : ℝ) : ℝ := x^2 + x - 2

/-- The function g(x) -/
def g (m : ℝ) (x : ℝ) : ℝ := |f x| - f x - 2*m*x - 2*m^2

/-- The theorem stating the range of m when g(x) has three distinct zeros -/
theorem range_of_m_when_g_has_three_zeros :
  ∀ m : ℝ, (∃ x y z : ℝ, x ≠ y ∧ y ≠ z ∧ x ≠ z ∧ g m x = 0 ∧ g m y = 0 ∧ g m z = 0) →
  m ∈ Set.Ioo ((1 - 2*Real.sqrt 7)/3) (-1) ∪ Set.Ioo 2 ((1 + 2*Real.sqrt 7)/3) :=
sorry

end NUMINAMATH_CALUDE_range_of_m_when_g_has_three_zeros_l701_70150


namespace NUMINAMATH_CALUDE_tens_digit_of_8_pow_2023_l701_70108

-- Define a function to get the last two digits of 8^n
def lastTwoDigits (n : ℕ) : ℕ := 8^n % 100

-- Define the cycle of last two digits
def lastTwoDigitsCycle : List ℕ := [8, 64, 12, 96, 68, 44, 52, 16, 28, 24]

-- Theorem statement
theorem tens_digit_of_8_pow_2023 :
  (lastTwoDigits 2023 / 10) % 10 = 1 :=
sorry

end NUMINAMATH_CALUDE_tens_digit_of_8_pow_2023_l701_70108


namespace NUMINAMATH_CALUDE_tissue_cost_theorem_l701_70172

/-- Calculates the total cost of tissue boxes given the number of boxes, packs per box, tissues per pack, and cost per tissue. -/
def total_cost (boxes : ℕ) (packs_per_box : ℕ) (tissues_per_pack : ℕ) (cost_per_tissue : ℚ) : ℚ :=
  (boxes * packs_per_box * tissues_per_pack : ℚ) * cost_per_tissue

/-- Proves that the total cost of 10 boxes of tissues is $1000 given the specified conditions. -/
theorem tissue_cost_theorem :
  let boxes : ℕ := 10
  let packs_per_box : ℕ := 20
  let tissues_per_pack : ℕ := 100
  let cost_per_tissue : ℚ := 5 / 100
  total_cost boxes packs_per_box tissues_per_pack cost_per_tissue = 1000 := by
  sorry

#eval total_cost 10 20 100 (5 / 100)

end NUMINAMATH_CALUDE_tissue_cost_theorem_l701_70172


namespace NUMINAMATH_CALUDE_lattice_points_on_segment_l701_70121

/-- The number of lattice points on a line segment -/
def latticePointCount (x1 y1 x2 y2 : ℤ) : ℕ :=
  sorry

/-- Theorem stating the number of lattice points on the given line segment -/
theorem lattice_points_on_segment : latticePointCount 5 23 60 353 = 56 := by
  sorry

end NUMINAMATH_CALUDE_lattice_points_on_segment_l701_70121


namespace NUMINAMATH_CALUDE_birth_death_rate_interval_l701_70171

/-- Prove that the time interval for birth and death rates is 2 seconds given the conditions --/
theorem birth_death_rate_interval (birth_rate death_rate net_increase_per_day : ℕ) 
  (h1 : birth_rate = 6)
  (h2 : death_rate = 2)
  (h3 : net_increase_per_day = 172800) :
  (24 * 60 * 60) / ((birth_rate - death_rate) * net_increase_per_day) = 2 := by
  sorry


end NUMINAMATH_CALUDE_birth_death_rate_interval_l701_70171


namespace NUMINAMATH_CALUDE_problem_statement_l701_70104

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := ((x + 1)^2 + a * Real.sin x) / (x^2 + 1) + 3

theorem problem_statement (a : ℝ) :
  f a (Real.log (Real.log 5 / Real.log 2)) = 5 →
  f a (Real.log (Real.log 2 / Real.log 5)) = 3 := by
sorry

end NUMINAMATH_CALUDE_problem_statement_l701_70104


namespace NUMINAMATH_CALUDE_min_balls_for_single_color_l701_70143

theorem min_balls_for_single_color (red green yellow blue white black : ℕ) 
  (h_red : red = 35)
  (h_green : green = 22)
  (h_yellow : yellow = 18)
  (h_blue : blue = 15)
  (h_white : white = 12)
  (h_black : black = 8) :
  let total := red + green + yellow + blue + white + black
  ∀ n : ℕ, n ≥ 87 → 
    ∃ color : ℕ, color ≥ 18 ∧ 
      (color ≤ red ∨ color ≤ green ∨ color ≤ yellow ∨ 
       color ≤ blue ∨ color ≤ white ∨ color ≤ black) ∧
    ∀ m : ℕ, m < 87 → 
      ¬(∃ color : ℕ, color ≥ 18 ∧ 
        (color ≤ red ∨ color ≤ green ∨ color ≤ yellow ∨ 
         color ≤ blue ∨ color ≤ white ∨ color ≤ black)) :=
by sorry

end NUMINAMATH_CALUDE_min_balls_for_single_color_l701_70143


namespace NUMINAMATH_CALUDE_mistaken_multiplication_l701_70151

theorem mistaken_multiplication (x : ℤ) : 139 * 43 - 139 * x = 1251 → x = 34 := by
  sorry

end NUMINAMATH_CALUDE_mistaken_multiplication_l701_70151


namespace NUMINAMATH_CALUDE_no_identical_snakes_swallowing_l701_70110

/-- A snake is represented as a set of points in a topological space. -/
def Snake (α : Type*) [TopologicalSpace α] := Set α

/-- Two snakes are identical if they are equal as sets. -/
def IdenticalSnakes {α : Type*} [TopologicalSpace α] (s1 s2 : Snake α) : Prop :=
  s1 = s2

/-- The process of one snake swallowing another from the tail. -/
def Swallow {α : Type*} [TopologicalSpace α] (s1 s2 : Snake α) : Prop :=
  ∃ t : Set α, t ⊆ s1 ∧ t = s2

/-- The theorem stating that it's impossible for two identical snakes to swallow each other from the tail. -/
theorem no_identical_snakes_swallowing {α : Type*} [TopologicalSpace α] (s1 s2 : Snake α) :
  IdenticalSnakes s1 s2 → ¬(Swallow s1 s2 ∧ Swallow s2 s1) :=
by
  sorry

end NUMINAMATH_CALUDE_no_identical_snakes_swallowing_l701_70110


namespace NUMINAMATH_CALUDE_alcohol_concentration_proof_l701_70173

/-- Proves that adding 7.5 litres of pure alcohol to a 10 litre solution
    that is 30% alcohol results in a 60% alcohol solution -/
theorem alcohol_concentration_proof :
  let initial_volume : ℝ := 10
  let initial_concentration : ℝ := 0.30
  let added_alcohol : ℝ := 7.5
  let final_concentration : ℝ := 0.60
  let final_volume : ℝ := initial_volume + added_alcohol
  let final_alcohol : ℝ := initial_volume * initial_concentration + added_alcohol
  final_alcohol / final_volume = final_concentration :=
by sorry


end NUMINAMATH_CALUDE_alcohol_concentration_proof_l701_70173


namespace NUMINAMATH_CALUDE_modulus_of_complex_cube_l701_70147

theorem modulus_of_complex_cube (z : ℂ) : Complex.abs ((((1 + Complex.I) / (1 - Complex.I)) ^ 3) : ℂ) = 1 := by
  sorry

end NUMINAMATH_CALUDE_modulus_of_complex_cube_l701_70147


namespace NUMINAMATH_CALUDE_ellipse_eccentricity_l701_70169

/-- Given an ellipse with semi-major axis a and semi-minor axis b,
    if the triangle formed by its left vertex, right vertex, and top vertex
    is an isosceles triangle with base angle 30°, then its eccentricity is √6/3 -/
theorem ellipse_eccentricity (a b : ℝ) (h1 : a > b) (h2 : b > 0) : 
  (b / a = Real.sqrt 3 / 3) → 
  Real.sqrt (1 - (b / a)^2) = Real.sqrt 6 / 3 := by
  sorry

end NUMINAMATH_CALUDE_ellipse_eccentricity_l701_70169


namespace NUMINAMATH_CALUDE_sculpture_and_base_height_l701_70164

/-- Converts feet to inches -/
def feet_to_inches (feet : ℕ) : ℕ := feet * 12

/-- Calculates the total height of a sculpture and its base in inches -/
def total_height (sculpture_feet : ℕ) (sculpture_inches : ℕ) (base_inches : ℕ) : ℕ :=
  feet_to_inches sculpture_feet + sculpture_inches + base_inches

/-- Theorem stating that a sculpture of 2 feet 10 inches on an 8-inch base has a total height of 42 inches -/
theorem sculpture_and_base_height :
  total_height 2 10 8 = 42 := by
  sorry

end NUMINAMATH_CALUDE_sculpture_and_base_height_l701_70164


namespace NUMINAMATH_CALUDE_min_value_of_2a_plus_b_l701_70114

theorem min_value_of_2a_plus_b (a b : ℝ) (ha : 0 < a) (hb : 0 < b) (h : a - 2*a*b + b = 0) :
  (∀ x y : ℝ, 0 < x → 0 < y → x - 2*x*y + y = 0 → 2*x + y ≥ 2*a + b) →
  2*a + b = 3/2 + Real.sqrt 2 :=
sorry

end NUMINAMATH_CALUDE_min_value_of_2a_plus_b_l701_70114


namespace NUMINAMATH_CALUDE_car_a_speed_car_a_speed_is_58_l701_70131

/-- Proves that the speed of Car A is 58 miles per hour given the initial conditions -/
theorem car_a_speed (initial_distance : ℝ) (time : ℝ) (speed_b : ℝ) : ℝ :=
  let distance_b := speed_b * time
  let total_distance := distance_b + initial_distance + 8
  total_distance / time

#check car_a_speed 24 4 50 = 58

/-- Theorem stating that the speed of Car A is indeed 58 miles per hour -/
theorem car_a_speed_is_58 :
  car_a_speed 24 4 50 = 58 := by sorry

end NUMINAMATH_CALUDE_car_a_speed_car_a_speed_is_58_l701_70131


namespace NUMINAMATH_CALUDE_problem_1_l701_70145

theorem problem_1 : 
  Real.sqrt 48 / Real.sqrt 3 - 4 * Real.sqrt (1/5) * Real.sqrt 30 + (2 * Real.sqrt 2 + Real.sqrt 3)^2 = 15 := by
  sorry

end NUMINAMATH_CALUDE_problem_1_l701_70145


namespace NUMINAMATH_CALUDE_geometric_sequence_fourth_term_l701_70109

/-- A geometric sequence is a sequence where each term after the first is found by multiplying the previous term by a fixed, non-zero number called the common ratio. -/
def IsGeometricSequence (a : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, q ≠ 0 ∧ ∀ n : ℕ, a (n + 1) = a n * q

theorem geometric_sequence_fourth_term
  (a : ℕ → ℝ)
  (h_geom : IsGeometricSequence a)
  (h1 : a 1 + a 2 = -1)
  (h2 : a 1 - a 3 = -3) :
  a 4 = -8 :=
sorry

end NUMINAMATH_CALUDE_geometric_sequence_fourth_term_l701_70109


namespace NUMINAMATH_CALUDE_f_properties_l701_70135

noncomputable def f (x : ℝ) : ℝ := Real.sin x ^ 2 + Real.sqrt 3 * Real.sin x * Real.cos x

theorem f_properties :
  (f (π / 12) = 1 / 2) ∧
  (Set.Icc 0 (3 / 2) = Set.image f (Set.Icc 0 (π / 2))) ∧
  (∃ (zeros : Finset ℝ), zeros.card = 5 ∧
    (∀ x ∈ zeros, x ∈ Set.Icc 0 (2 * π) ∧ f x = 0) ∧
    (∀ x ∈ Set.Icc 0 (2 * π), f x = 0 → x ∈ zeros)) ∧
  (∃ (zeros : Finset ℝ), zeros.card = 5 ∧
    (∀ x ∈ zeros, x ∈ Set.Icc 0 (2 * π) ∧ f x = 0) ∧
    (∀ x ∈ Set.Icc 0 (2 * π), f x = 0 → x ∈ zeros) ∧
    (zeros.sum id = 16 * π / 3)) :=
by sorry

end NUMINAMATH_CALUDE_f_properties_l701_70135


namespace NUMINAMATH_CALUDE_max_t_for_exponential_sequence_range_a_for_quadratic_sequence_l701_70189

/-- Definition of property P(t) for a sequence -/
def has_property_P (a : ℕ → ℝ) (t : ℝ) : Prop :=
  ∀ m n : ℕ, m ≠ n → (a m - a n) / (m - n : ℝ) ≥ t

/-- Theorem for part (i) -/
theorem max_t_for_exponential_sequence :
  ∃ t_max : ℝ, (∀ t : ℝ, has_property_P (λ n => (2 : ℝ) ^ n) t → t ≤ t_max) ∧
            has_property_P (λ n => (2 : ℝ) ^ n) t_max :=
sorry

/-- Theorem for part (ii) -/
theorem range_a_for_quadratic_sequence :
  ∃ a_min : ℝ, (∀ a : ℝ, has_property_P (λ n => n^2 - a / n) 10 → a ≥ a_min) ∧
            has_property_P (λ n => n^2 - a_min / n) 10 :=
sorry

end NUMINAMATH_CALUDE_max_t_for_exponential_sequence_range_a_for_quadratic_sequence_l701_70189


namespace NUMINAMATH_CALUDE_village_population_l701_70162

/-- If 80% of a village's population is 64,000, then the total population is 80,000. -/
theorem village_population (population : ℕ) (h : (80 : ℕ) * population = 100 * 64000) :
  population = 80000 := by
  sorry

end NUMINAMATH_CALUDE_village_population_l701_70162


namespace NUMINAMATH_CALUDE_floor_plus_self_unique_solution_l701_70193

theorem floor_plus_self_unique_solution :
  ∃! s : ℝ, ⌊s⌋ + s = 20.75 := by
  sorry

end NUMINAMATH_CALUDE_floor_plus_self_unique_solution_l701_70193


namespace NUMINAMATH_CALUDE_polynomial_factorization_l701_70158

theorem polynomial_factorization (m n a b : ℝ) : 
  (|m - 4| + (n^2 - 8*n + 16) = 0) → 
  (a^2 + 4*b^2 - m*a*b - n = (a - 2*b + 2) * (a - 2*b - 2)) := by
sorry

end NUMINAMATH_CALUDE_polynomial_factorization_l701_70158


namespace NUMINAMATH_CALUDE_sales_tax_percentage_l701_70155

/-- Given a purchase with a total cost, tax rate, and cost of tax-free items,
    calculate the percentage of the total cost that went on sales tax. -/
theorem sales_tax_percentage
  (total_cost : ℝ)
  (tax_rate : ℝ)
  (tax_free_cost : ℝ)
  (h1 : total_cost = 20)
  (h2 : tax_rate = 0.06)
  (h3 : tax_free_cost = 14.7) :
  (tax_rate * (total_cost - tax_free_cost)) / total_cost * 100 = 1.59 := by
sorry

end NUMINAMATH_CALUDE_sales_tax_percentage_l701_70155


namespace NUMINAMATH_CALUDE_line_equivalence_l701_70181

/-- Given a line in the form (3, 4) · ((x, y) - (2, 8)) = 0, 
    prove that it's equivalent to y = -3/4 * x + 9.5 -/
theorem line_equivalence :
  ∀ (x y : ℝ), 3 * (x - 2) + 4 * (y - 8) = 0 ↔ y = -3/4 * x + 9.5 := by
sorry

end NUMINAMATH_CALUDE_line_equivalence_l701_70181


namespace NUMINAMATH_CALUDE_james_puzzles_l701_70176

/-- Calculates the number of puzzles James bought given the puzzle size, completion rate, and total time --/
theorem james_puzzles (puzzle_size : ℕ) (pieces_per_interval : ℕ) (interval_minutes : ℕ) (total_minutes : ℕ) :
  puzzle_size = 2000 →
  pieces_per_interval = 100 →
  interval_minutes = 10 →
  total_minutes = 400 →
  (total_minutes / interval_minutes) * pieces_per_interval / puzzle_size = 2 :=
by
  sorry

end NUMINAMATH_CALUDE_james_puzzles_l701_70176


namespace NUMINAMATH_CALUDE_f_monotonicity_and_max_root_difference_l701_70177

noncomputable def f (x : ℝ) : ℝ := 4 * x - x^4

theorem f_monotonicity_and_max_root_difference :
  (∀ x y, x < y ∧ y < 1 → f x < f y) ∧
  (∀ x y, 1 < x ∧ x < y → f x > f y) ∧
  (∀ a x₁ x₂, f x₁ = a ∧ f x₂ = a ∧ x₁ ≠ x₂ ∧ 1 < x₂ → x₂ - 1 ≤ 0) :=
sorry

end NUMINAMATH_CALUDE_f_monotonicity_and_max_root_difference_l701_70177


namespace NUMINAMATH_CALUDE_tori_trash_count_l701_70144

/-- The number of pieces of trash Tori picked up in the classrooms -/
def classroom_trash : ℕ := 344

/-- The number of pieces of trash Tori picked up outside the classrooms -/
def outside_trash : ℕ := 1232

/-- The total number of pieces of trash Tori picked up -/
def total_trash : ℕ := classroom_trash + outside_trash

theorem tori_trash_count : total_trash = 1576 := by
  sorry

end NUMINAMATH_CALUDE_tori_trash_count_l701_70144


namespace NUMINAMATH_CALUDE_mode_is_tallest_rectangle_midpoint_l701_70137

/-- Represents a frequency distribution histogram --/
structure FrequencyHistogram where
  -- Add necessary fields here
  -- This is a simplified representation

/-- The mode of a frequency distribution --/
def mode (h : FrequencyHistogram) : ℝ :=
  sorry

/-- The abscissa of the midpoint of the base of the tallest rectangle --/
def tallestRectangleMidpoint (h : FrequencyHistogram) : ℝ :=
  sorry

/-- Theorem stating that the abscissa of the midpoint of the base of the tallest rectangle
    represents the mode in a frequency distribution histogram --/
theorem mode_is_tallest_rectangle_midpoint (h : FrequencyHistogram) :
  mode h = tallestRectangleMidpoint h :=
sorry

end NUMINAMATH_CALUDE_mode_is_tallest_rectangle_midpoint_l701_70137


namespace NUMINAMATH_CALUDE_simplify_trig_fraction_l701_70191

theorem simplify_trig_fraction (x : ℝ) :
  (2 + 2 * Real.sin x - 2 * Real.cos x) / (2 + 2 * Real.sin x + 2 * Real.cos x) = Real.tan (x / 2) := by
  sorry

end NUMINAMATH_CALUDE_simplify_trig_fraction_l701_70191


namespace NUMINAMATH_CALUDE_number_with_specific_remainders_l701_70153

theorem number_with_specific_remainders : ∃ n : ℕ, 
  n > 0 ∧ 
  n % 13 = 11 ∧ 
  n % 17 = 9 ∧ 
  ∀ m : ℕ, m > 0 ∧ m % 13 = 11 ∧ m % 17 = 9 → m ≥ n :=
by sorry

end NUMINAMATH_CALUDE_number_with_specific_remainders_l701_70153


namespace NUMINAMATH_CALUDE_C_power_50_l701_70118

def C : Matrix (Fin 2) (Fin 2) ℤ :=
  !![3, 1;
    -4, -1]

theorem C_power_50 :
  C^50 = !![101, 50;
            -200, -99] := by
  sorry

end NUMINAMATH_CALUDE_C_power_50_l701_70118


namespace NUMINAMATH_CALUDE_log_identity_l701_70134

theorem log_identity (a : ℝ) (h : a * Real.log 4 / Real.log 3 = 1) :
  2^a + 2^(-a) = 4 * Real.sqrt 3 / 3 := by
  sorry

end NUMINAMATH_CALUDE_log_identity_l701_70134


namespace NUMINAMATH_CALUDE_combine_equations_l701_70154

theorem combine_equations :
  (15 / 5 = 3) →
  (24 - 3 = 21) →
  (24 - 15 / 3 : ℚ) = 19 := by
  sorry

end NUMINAMATH_CALUDE_combine_equations_l701_70154


namespace NUMINAMATH_CALUDE_greatest_integer_less_than_negative_seventeen_thirds_l701_70142

theorem greatest_integer_less_than_negative_seventeen_thirds :
  ⌊-17/3⌋ = -6 := by sorry

end NUMINAMATH_CALUDE_greatest_integer_less_than_negative_seventeen_thirds_l701_70142


namespace NUMINAMATH_CALUDE_number_of_students_l701_70156

/-- The number of storybooks available for distribution -/
def total_books : ℕ := 60

/-- Predicate to check if there are books left over after initial distribution -/
def has_leftover (n : ℕ) : Prop := n < total_books

/-- Predicate to check if remaining books can be evenly distributed with 2 students sharing 1 book -/
def can_evenly_distribute_remainder (n : ℕ) : Prop :=
  ∃ k : ℕ, total_books - n = 2 * k

/-- The theorem stating the number of students in the class -/
theorem number_of_students :
  ∃ n : ℕ, n = 40 ∧ 
    has_leftover n ∧ 
    can_evenly_distribute_remainder n :=
sorry

end NUMINAMATH_CALUDE_number_of_students_l701_70156


namespace NUMINAMATH_CALUDE_class_grade_point_average_l701_70128

/-- Calculate the grade point average of a class given the distribution of grades --/
theorem class_grade_point_average 
  (total_students : ℕ) 
  (gpa_60_percent : ℚ) 
  (gpa_65_percent : ℚ) 
  (gpa_70_percent : ℚ) 
  (gpa_80_percent : ℚ) 
  (h1 : total_students = 120)
  (h2 : gpa_60_percent = 25 / 100)
  (h3 : gpa_65_percent = 35 / 100)
  (h4 : gpa_70_percent = 15 / 100)
  (h5 : gpa_80_percent = 1 - (gpa_60_percent + gpa_65_percent + gpa_70_percent))
  (h6 : gpa_60_percent + gpa_65_percent + gpa_70_percent + gpa_80_percent = 1) :
  let weighted_average := 
    (gpa_60_percent * 60 + gpa_65_percent * 65 + gpa_70_percent * 70 + gpa_80_percent * 80)
  weighted_average = 68.25 := by
  sorry

end NUMINAMATH_CALUDE_class_grade_point_average_l701_70128


namespace NUMINAMATH_CALUDE_carries_cucumber_harvest_l701_70119

/-- Represents the dimensions of a rectangular garden -/
structure GardenDimensions where
  length : ℝ
  width : ℝ

/-- Calculates the expected cucumber harvest given garden dimensions and planting parameters -/
def expected_harvest (garden : GardenDimensions) (plants_per_sqft : ℝ) (cucumbers_per_plant : ℝ) : ℝ :=
  garden.length * garden.width * plants_per_sqft * cucumbers_per_plant

/-- Theorem stating that Carrie's garden will yield 9000 cucumbers -/
theorem carries_cucumber_harvest :
  let garden := GardenDimensions.mk 10 12
  let plants_per_sqft := 5
  let cucumbers_per_plant := 15
  expected_harvest garden plants_per_sqft cucumbers_per_plant = 9000 := by
  sorry


end NUMINAMATH_CALUDE_carries_cucumber_harvest_l701_70119


namespace NUMINAMATH_CALUDE_beavers_still_working_l701_70196

def initial_beavers : ℕ := 7
def swimming_beavers : ℕ := 2
def stick_collecting_beaver : ℕ := 1
def food_searching_beaver : ℕ := 1

theorem beavers_still_working : ℕ := by
  sorry

end NUMINAMATH_CALUDE_beavers_still_working_l701_70196


namespace NUMINAMATH_CALUDE_ring_weight_sum_l701_70113

/-- The weight of the orange ring in ounces -/
def orange_ring : ℚ := 0.08

/-- The weight of the purple ring in ounces -/
def purple_ring : ℚ := 0.33

/-- The weight of the white ring in ounces -/
def white_ring : ℚ := 0.42

/-- The total weight of all rings in ounces -/
def total_weight : ℚ := orange_ring + purple_ring + white_ring

theorem ring_weight_sum :
  total_weight = 0.83 := by sorry

end NUMINAMATH_CALUDE_ring_weight_sum_l701_70113


namespace NUMINAMATH_CALUDE_sum_congruence_l701_70192

theorem sum_congruence : ∃ k : ℤ, (85 + 86 + 87 + 88 + 89 + 90 + 91 + 92) = 16 * k + 2 := by
  sorry

end NUMINAMATH_CALUDE_sum_congruence_l701_70192


namespace NUMINAMATH_CALUDE_distance_after_walk_on_hexagon_l701_70136

/-- The distance from the starting point after walking along a regular hexagon's perimeter -/
theorem distance_after_walk_on_hexagon (side_length : ℝ) (walk_distance : ℝ) 
  (h1 : side_length = 3)
  (h2 : walk_distance = 10) :
  ∃ (end_point : ℝ × ℝ),
    (end_point.1^2 + end_point.2^2) = (3 * Real.sqrt 3)^2 := by
  sorry

end NUMINAMATH_CALUDE_distance_after_walk_on_hexagon_l701_70136


namespace NUMINAMATH_CALUDE_inequality_not_always_true_l701_70116

theorem inequality_not_always_true (x y : ℝ) (h : x > 1 ∧ 1 > y) :
  ¬ (∀ x y : ℝ, x > 1 ∧ 1 > y → x - 1 > 1 - y) :=
by sorry

end NUMINAMATH_CALUDE_inequality_not_always_true_l701_70116


namespace NUMINAMATH_CALUDE_scarf_to_tie_belt_ratio_l701_70140

-- Define the quantities given in the problem
def ties : ℕ := 34
def belts : ℕ := 40
def black_shirts : ℕ := 63
def white_shirts : ℕ := 42

-- Define the number of jeans based on the given condition
def jeans : ℕ := (2 * (black_shirts + white_shirts)) / 3

-- Define the number of scarves based on the given condition
def scarves : ℕ := jeans - 33

-- Theorem to prove
theorem scarf_to_tie_belt_ratio :
  scarves * 2 = ties + belts := by
  sorry


end NUMINAMATH_CALUDE_scarf_to_tie_belt_ratio_l701_70140


namespace NUMINAMATH_CALUDE_expression_evaluation_l701_70187

theorem expression_evaluation :
  let x : ℝ := Real.sqrt 3 - 1
  (2 / (x + 1) + 1 / (x - 2)) / ((x - 1) / (x - 2)) = Real.sqrt 3 :=
by sorry

end NUMINAMATH_CALUDE_expression_evaluation_l701_70187


namespace NUMINAMATH_CALUDE_bert_profit_l701_70190

def selling_price : ℝ := 90
def markup : ℝ := 10
def tax_rate : ℝ := 0.1

theorem bert_profit : 
  let cost_price := selling_price - markup
  let tax := selling_price * tax_rate
  selling_price - cost_price - tax = 1 := by
  sorry

end NUMINAMATH_CALUDE_bert_profit_l701_70190


namespace NUMINAMATH_CALUDE_quadratic_identities_max_bound_l701_70185

/-- Given 0 ≤ p, r ≤ 1 and two identities, prove that max(a, b, c) and max(α, β, γ) are ≥ 4/9 -/
theorem quadratic_identities_max_bound {p r : ℝ} (hp : 0 ≤ p ∧ p ≤ 1) (hr : 0 ≤ r ∧ r ≤ 1)
  (h1 : ∀ x y, (p * x + (1 - p) * y)^2 = a * x^2 + b * x * y + c * y^2)
  (h2 : ∀ x y, (p * x + (1 - p) * y) * (r * x + (1 - r) * y) = α * x^2 + β * x * y + γ * y^2) :
  max a (max b c) ≥ 4/9 ∧ max α (max β γ) ≥ 4/9 := by
sorry

end NUMINAMATH_CALUDE_quadratic_identities_max_bound_l701_70185


namespace NUMINAMATH_CALUDE_footprint_calculation_l701_70117

/-- Calculates the total number of footprints left by three creatures on their respective planets -/
theorem footprint_calculation (pogo_rate : ℕ) (grimzi_rate : ℕ) (zeb_rate : ℕ)
  (pogo_distance : ℕ) (grimzi_distance : ℕ) (zeb_distance : ℕ)
  (total_distance : ℕ) :
  pogo_rate = 4 ∧ 
  grimzi_rate = 3 ∧ 
  zeb_rate = 5 ∧
  pogo_distance = 1 ∧ 
  grimzi_distance = 6 ∧ 
  zeb_distance = 8 ∧
  total_distance = 6000 →
  pogo_rate * total_distance + 
  (total_distance / grimzi_distance) * grimzi_rate + 
  (total_distance / zeb_distance) * zeb_rate = 30750 := by
sorry


end NUMINAMATH_CALUDE_footprint_calculation_l701_70117
