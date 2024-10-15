import Mathlib

namespace NUMINAMATH_CALUDE_cylinder_in_hemisphere_height_l793_79365

theorem cylinder_in_hemisphere_height (r c h : ℝ) : 
  r > 0 → c > 0 → h > 0 →
  r = 7 → c = 3 →
  h^2 = r^2 - c^2 →
  h = Real.sqrt 40 := by
sorry

end NUMINAMATH_CALUDE_cylinder_in_hemisphere_height_l793_79365


namespace NUMINAMATH_CALUDE_decagon_diagonals_l793_79368

/-- The number of internal diagonals in a polygon with n sides -/
def num_diagonals (n : ℕ) : ℕ := n * (n - 3) / 2

/-- A decagon is a polygon with 10 sides -/
def decagon_sides : ℕ := 10

theorem decagon_diagonals : num_diagonals decagon_sides = 35 := by
  sorry

end NUMINAMATH_CALUDE_decagon_diagonals_l793_79368


namespace NUMINAMATH_CALUDE_movie_theater_popcorn_ratio_l793_79370

/-- Movie theater revenue calculation and customer ratio --/
theorem movie_theater_popcorn_ratio 
  (matinee_price evening_price opening_price popcorn_price : ℕ)
  (matinee_customers evening_customers opening_customers : ℕ)
  (total_revenue : ℕ)
  (h_matinee : matinee_price = 5)
  (h_evening : evening_price = 7)
  (h_opening : opening_price = 10)
  (h_popcorn : popcorn_price = 10)
  (h_matinee_cust : matinee_customers = 32)
  (h_evening_cust : evening_customers = 40)
  (h_opening_cust : opening_customers = 58)
  (h_total_rev : total_revenue = 1670) :
  (total_revenue - (matinee_price * matinee_customers + 
                    evening_price * evening_customers + 
                    opening_price * opening_customers)) / popcorn_price = 
  (matinee_customers + evening_customers + opening_customers) / 2 :=
by sorry

end NUMINAMATH_CALUDE_movie_theater_popcorn_ratio_l793_79370


namespace NUMINAMATH_CALUDE_q_div_p_eq_90_l793_79319

/-- The number of cards in the box -/
def total_cards : ℕ := 50

/-- The number of distinct numbers on the cards -/
def distinct_numbers : ℕ := 10

/-- The number of cards for each number -/
def cards_per_number : ℕ := 5

/-- The number of cards drawn -/
def cards_drawn : ℕ := 4

/-- The probability of drawing all cards with the same number -/
def p : ℚ := (distinct_numbers * Nat.choose cards_per_number cards_drawn) / Nat.choose total_cards cards_drawn

/-- The probability of drawing three cards with one number and one card with a different number -/
def q : ℚ := (distinct_numbers * Nat.choose cards_per_number 3 * (distinct_numbers - 1) * Nat.choose cards_per_number 1) / Nat.choose total_cards cards_drawn

/-- The main theorem stating that q/p = 90 -/
theorem q_div_p_eq_90 : q / p = 90 := by
  sorry

end NUMINAMATH_CALUDE_q_div_p_eq_90_l793_79319


namespace NUMINAMATH_CALUDE_new_students_count_l793_79304

/-- Represents the problem of calculating the number of new students joining a school --/
theorem new_students_count (initial_avg_age initial_count new_students_avg_age final_avg_age final_count : ℕ) : 
  initial_avg_age = 48 →
  new_students_avg_age = 32 →
  final_avg_age = 44 →
  final_count = 160 →
  ∃ new_students : ℕ,
    new_students = 40 ∧
    final_count = initial_count + new_students ∧
    final_avg_age * final_count = initial_avg_age * initial_count + new_students_avg_age * new_students :=
by sorry

end NUMINAMATH_CALUDE_new_students_count_l793_79304


namespace NUMINAMATH_CALUDE_sum_plus_count_theorem_l793_79300

def sum_of_integers (a b : ℕ) : ℕ := ((b - a + 1) * (a + b)) / 2

def count_even_integers (a b : ℕ) : ℕ := ((b - a) / 2) + 1

theorem sum_plus_count_theorem : 
  sum_of_integers 50 70 + count_even_integers 50 70 = 1271 := by
  sorry

end NUMINAMATH_CALUDE_sum_plus_count_theorem_l793_79300


namespace NUMINAMATH_CALUDE_increasing_function_inequality_l793_79361

theorem increasing_function_inequality (f : ℝ → ℝ) (h_increasing : Monotone f) :
  (∀ x : ℝ, f 4 < f (2^x)) → {x : ℝ | x > 2}.Nonempty := by
  sorry

end NUMINAMATH_CALUDE_increasing_function_inequality_l793_79361


namespace NUMINAMATH_CALUDE_simplify_and_evaluate_l793_79364

theorem simplify_and_evaluate (a b : ℝ) (ha : a = Real.sqrt 3 - Real.sqrt 11) (hb : b = Real.sqrt 3 + Real.sqrt 11) :
  (a^2 - b^2) / (a^2 * b - a * b^2) / (1 + (a^2 + b^2) / (2 * a * b)) = Real.sqrt 3 / 3 := by
  sorry

end NUMINAMATH_CALUDE_simplify_and_evaluate_l793_79364


namespace NUMINAMATH_CALUDE_equilateral_triangle_cd_product_l793_79342

/-- An equilateral triangle with vertices at (0,0), (c, 20), and (d, 51) -/
structure EquilateralTriangle where
  c : ℝ
  d : ℝ
  is_equilateral : (c^2 + 20^2 = d^2 + 51^2) ∧ 
                   (c^2 + 20^2 = c^2 + d^2 + 51^2 - 2*c*d - 2*20*51) ∧
                   (d^2 + 51^2 = c^2 + d^2 + 51^2 - 2*c*d - 2*20*51)

/-- The product of c and d in the equilateral triangle equals -5822/3 -/
theorem equilateral_triangle_cd_product (t : EquilateralTriangle) : t.c * t.d = -5822/3 := by
  sorry

end NUMINAMATH_CALUDE_equilateral_triangle_cd_product_l793_79342


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_condition_l793_79308

theorem sufficient_not_necessary_condition (x : ℝ) :
  (x < -1 → 2*x^2 + x - 1 > 0) ∧ 
  ¬(2*x^2 + x - 1 > 0 → x < -1) := by
  sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_condition_l793_79308


namespace NUMINAMATH_CALUDE_inequality_solution_l793_79399

theorem inequality_solution (x : ℝ) : 
  (2 * x + 2) / (3 * x + 1) < (x - 3) / (x + 4) ↔ 
  (x > -Real.sqrt 11 ∧ x < -1/3) ∨ (x > Real.sqrt 11) :=
by sorry

end NUMINAMATH_CALUDE_inequality_solution_l793_79399


namespace NUMINAMATH_CALUDE_slope_angle_range_l793_79392

/-- Given two lines L1 and L2, where L1 has slope k and y-intercept -b,
    and their intersection point M is in the first quadrant,
    prove that the slope angle α of L1 is between arctan(-2/3) and π/2. -/
theorem slope_angle_range (k b : ℝ) :
  let L1 := λ x y : ℝ => y = k * x - b
  let L2 := λ x y : ℝ => 2 * x + 3 * y - 6 = 0
  let M := (((3 * b + 6) / (2 + 3 * k)), ((6 * k + 2 * b) / (2 + 3 * k)))
  let α := Real.arctan k
  (M.1 > 0 ∧ M.2 > 0) → (α > Real.arctan (-2/3) ∧ α < π/2) := by
  sorry

end NUMINAMATH_CALUDE_slope_angle_range_l793_79392


namespace NUMINAMATH_CALUDE_reimbursement_is_correct_l793_79333

/-- Represents the type of client --/
inductive ClientType
| Industrial
| Commercial

/-- Represents the day of the week --/
inductive DayType
| Weekday
| Weekend

/-- Calculates the reimbursement rate based on client type and day type --/
def reimbursementRate (client : ClientType) (day : DayType) : ℚ :=
  match client, day with
  | ClientType.Industrial, DayType.Weekday => 36/100
  | ClientType.Commercial, DayType.Weekday => 42/100
  | _, DayType.Weekend => 45/100

/-- Represents a day's travel --/
structure DayTravel where
  miles : ℕ
  client : ClientType
  day : DayType

/-- Calculates the reimbursement for a single day --/
def dailyReimbursement (travel : DayTravel) : ℚ :=
  (travel.miles : ℚ) * reimbursementRate travel.client travel.day

/-- The week's travel schedule --/
def weekSchedule : List DayTravel := [
  ⟨18, ClientType.Industrial, DayType.Weekday⟩,
  ⟨26, ClientType.Commercial, DayType.Weekday⟩,
  ⟨20, ClientType.Industrial, DayType.Weekday⟩,
  ⟨20, ClientType.Commercial, DayType.Weekday⟩,
  ⟨16, ClientType.Industrial, DayType.Weekday⟩,
  ⟨12, ClientType.Commercial, DayType.Weekend⟩
]

/-- Calculates the total reimbursement for the week --/
def totalReimbursement (schedule : List DayTravel) : ℚ :=
  schedule.map dailyReimbursement |>.sum

/-- Theorem: The total reimbursement for the given week schedule is $44.16 --/
theorem reimbursement_is_correct : totalReimbursement weekSchedule = 4416/100 := by
  sorry

end NUMINAMATH_CALUDE_reimbursement_is_correct_l793_79333


namespace NUMINAMATH_CALUDE_ellipse_minor_axis_length_l793_79376

/-- The length of the minor axis of an ellipse with semi-focal distance 2 and eccentricity 1/2 is 2√3. -/
theorem ellipse_minor_axis_length : 
  ∀ (c a b : ℝ), 
  c = 2 → -- semi-focal distance
  a / c = 2 → -- derived from eccentricity e = 1/2
  b ^ 2 = a ^ 2 - c ^ 2 → -- relationship between a, b, and c in an ellipse
  b = 2 * Real.sqrt 3 := by
sorry

end NUMINAMATH_CALUDE_ellipse_minor_axis_length_l793_79376


namespace NUMINAMATH_CALUDE_smallest_prime_after_seven_nonprimes_l793_79320

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d > 1 → d < n → ¬(n % d = 0)

def consecutive_nonprimes (start : ℕ) (count : ℕ) : Prop :=
  ∀ k : ℕ, k ≥ start → k < start + count → ¬(is_prime k)

theorem smallest_prime_after_seven_nonprimes :
  (∃ start : ℕ, consecutive_nonprimes start 7) ∧
  (∀ p : ℕ, p < 97 → ¬(is_prime p ∧ ∃ start : ℕ, consecutive_nonprimes start 7 ∧ start + 7 ≤ p)) ∧
  is_prime 97 :=
sorry

end NUMINAMATH_CALUDE_smallest_prime_after_seven_nonprimes_l793_79320


namespace NUMINAMATH_CALUDE_perimeter_relations_l793_79335

variable (n : ℕ+) (r : ℝ) (hr : r > 0)

/-- Perimeter of regular n-gon circumscribed around a circle with radius r -/
noncomputable def K (n : ℕ+) (r : ℝ) : ℝ := sorry

/-- Perimeter of regular n-gon inscribed in a circle with radius r -/
noncomputable def k (n : ℕ+) (r : ℝ) : ℝ := sorry

theorem perimeter_relations (n : ℕ+) (r : ℝ) (hr : r > 0) :
  (K (2 * n) r = (2 * K n r * k n r) / (K n r + k n r)) ∧
  (k (2 * n) r = Real.sqrt ((k n r) * (K (2 * n) r))) := by sorry

end NUMINAMATH_CALUDE_perimeter_relations_l793_79335


namespace NUMINAMATH_CALUDE_sequence_periodicity_l793_79313

def sequence_rule (x : ℤ → ℝ) : Prop :=
  ∀ n : ℤ, n > 1 → x (n + 1) = |x n| - x (n - 1)

theorem sequence_periodicity (x : ℤ → ℝ) (h : sequence_rule x) :
  ∀ k : ℤ, x (k + 9) = x k ∧ x (k + 8) = x (k - 1) :=
sorry

end NUMINAMATH_CALUDE_sequence_periodicity_l793_79313


namespace NUMINAMATH_CALUDE_prism_128_cubes_ratio_l793_79357

/-- Represents the dimensions of a rectangular prism -/
structure PrismDimensions where
  width : ℕ
  length : ℕ
  height : ℕ

/-- Checks if the given dimensions form a valid prism of 128 cubes -/
def is_valid_prism (d : PrismDimensions) : Prop :=
  d.width * d.length * d.height = 128

/-- Checks if the given dimensions have the ratio 1:1:2 -/
def has_ratio_1_1_2 (d : PrismDimensions) : Prop :=
  d.width = d.length ∧ d.height = 2 * d.width

/-- Theorem stating that a valid prism of 128 cubes has dimensions with ratio 1:1:2 -/
theorem prism_128_cubes_ratio :
  ∀ d : PrismDimensions, is_valid_prism d → has_ratio_1_1_2 d :=
by sorry

end NUMINAMATH_CALUDE_prism_128_cubes_ratio_l793_79357


namespace NUMINAMATH_CALUDE_line_perpendicular_to_plane_and_parallel_line_l793_79331

structure Plane where

structure Line where

def perpendicular (l : Line) (p : Plane) : Prop := sorry

def parallel (l : Line) (p : Plane) : Prop := sorry

def perpendicular_lines (l1 l2 : Line) : Prop := sorry

theorem line_perpendicular_to_plane_and_parallel_line 
  (α : Plane) (m n : Line) 
  (h1 : perpendicular m α) 
  (h2 : parallel n α) : 
  perpendicular_lines m n := by sorry

end NUMINAMATH_CALUDE_line_perpendicular_to_plane_and_parallel_line_l793_79331


namespace NUMINAMATH_CALUDE_ninth_term_of_specific_arithmetic_sequence_l793_79322

def arithmetic_sequence (a₁ : ℚ) (d : ℚ) : ℕ → ℚ :=
  λ n => a₁ + (n - 1 : ℚ) * d

theorem ninth_term_of_specific_arithmetic_sequence :
  ∃ (d : ℚ), 
    let seq := arithmetic_sequence (3/4) d
    seq 1 = 3/4 ∧ seq 17 = 1/2 ∧ seq 9 = 5/8 :=
by
  sorry

end NUMINAMATH_CALUDE_ninth_term_of_specific_arithmetic_sequence_l793_79322


namespace NUMINAMATH_CALUDE_tetrahedron_surface_area_l793_79366

/-- Given a regular tetrahedron with a square cross-section of area m², 
    its surface area is 4m²√3. -/
theorem tetrahedron_surface_area (m : ℝ) (h : m > 0) : 
  let square_area : ℝ := m^2
  let tetrahedron_surface_area : ℝ := 4 * m^2 * Real.sqrt 3
  square_area = m^2 → tetrahedron_surface_area = 4 * m^2 * Real.sqrt 3 :=
by sorry

end NUMINAMATH_CALUDE_tetrahedron_surface_area_l793_79366


namespace NUMINAMATH_CALUDE_jessica_seashells_l793_79386

theorem jessica_seashells (initial_shells : ℝ) (given_away : ℝ) (remaining : ℝ) : 
  initial_shells = 8.5 → 
  given_away = 6.25 → 
  remaining = initial_shells - given_away → 
  remaining = 2.25 := by
sorry

end NUMINAMATH_CALUDE_jessica_seashells_l793_79386


namespace NUMINAMATH_CALUDE_range_of_x_when_not_p_range_of_m_for_not_q_sufficient_not_necessary_l793_79398

-- Define the propositions p and q
def p (x : ℝ) : Prop := x^2 - x - 2 ≤ 0
def q (x m : ℝ) : Prop := x^2 - x - m^2 - m ≤ 0

-- Theorem for the range of x when ¬p is true
theorem range_of_x_when_not_p (x : ℝ) :
  ¬(p x) ↔ (x > 2 ∨ x < -1) :=
sorry

-- Theorem for the range of m when ¬q is a sufficient but not necessary condition for ¬p
theorem range_of_m_for_not_q_sufficient_not_necessary (m : ℝ) :
  (∀ x, ¬(q x m) → ¬(p x)) ∧ ¬(∀ x, q x m → p x) ↔ (m > 1 ∨ m < -2) :=
sorry

end NUMINAMATH_CALUDE_range_of_x_when_not_p_range_of_m_for_not_q_sufficient_not_necessary_l793_79398


namespace NUMINAMATH_CALUDE_field_trip_passengers_l793_79301

/-- The number of passengers a single bus can transport -/
def passengers_per_bus : ℕ := 48

/-- The number of buses needed for the field trip -/
def buses_needed : ℕ := 26

/-- The total number of passengers (students and teachers) going on the field trip -/
def total_passengers : ℕ := passengers_per_bus * buses_needed

theorem field_trip_passengers :
  total_passengers = 1248 :=
sorry

end NUMINAMATH_CALUDE_field_trip_passengers_l793_79301


namespace NUMINAMATH_CALUDE_author_writing_speed_l793_79390

/-- Calculates the words written per hour, given the total words, total hours, and break hours -/
def wordsPerHour (totalWords : ℕ) (totalHours : ℕ) (breakHours : ℕ) : ℕ :=
  totalWords / (totalHours - breakHours)

/-- Theorem stating that under the given conditions, the author wrote at least 705 words per hour -/
theorem author_writing_speed :
  let totalWords : ℕ := 60000
  let totalHours : ℕ := 100
  let breakHours : ℕ := 15
  wordsPerHour totalWords totalHours breakHours ≥ 705 := by
  sorry

#eval wordsPerHour 60000 100 15

end NUMINAMATH_CALUDE_author_writing_speed_l793_79390


namespace NUMINAMATH_CALUDE_smallest_shift_l793_79307

-- Define the function g with the given property
def g_periodic (g : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, g (x - 15) = g x

-- Define the property for the shifted function
def shifted_function_equal (g : ℝ → ℝ) (b : ℝ) : Prop :=
  ∀ x : ℝ, g ((x - b) / 3) = g (x / 3)

-- Theorem statement
theorem smallest_shift (g : ℝ → ℝ) :
  g_periodic g →
  (∃ b : ℝ, b > 0 ∧ shifted_function_equal g b ∧
    ∀ b' : ℝ, b' > 0 ∧ shifted_function_equal g b' → b ≤ b') →
  (∃ b : ℝ, b = 45 ∧ shifted_function_equal g b) :=
sorry

end NUMINAMATH_CALUDE_smallest_shift_l793_79307


namespace NUMINAMATH_CALUDE_rectangular_garden_area_l793_79337

theorem rectangular_garden_area :
  ∀ (length width area : ℝ),
    length = 175 →
    width = 12 →
    area = length * width →
    area = 2100 := by
  sorry

end NUMINAMATH_CALUDE_rectangular_garden_area_l793_79337


namespace NUMINAMATH_CALUDE_xy_positive_l793_79381

theorem xy_positive (x y : ℝ) (h1 : x * y > 1) (h2 : x + y ≥ 0) : x > 0 ∧ y > 0 := by
  sorry

end NUMINAMATH_CALUDE_xy_positive_l793_79381


namespace NUMINAMATH_CALUDE_solve_equation_l793_79371

theorem solve_equation (x : ℝ) (h : 3 * x = (26 - x) + 26) : x = 13 := by
  sorry

end NUMINAMATH_CALUDE_solve_equation_l793_79371


namespace NUMINAMATH_CALUDE_cost_of_paving_floor_l793_79311

/-- The cost of paving a rectangular floor given its dimensions and rate per square meter. -/
theorem cost_of_paving_floor (length width rate : ℝ) : 
  length = 5.5 → width = 4 → rate = 800 → length * width * rate = 17600 := by
  sorry

end NUMINAMATH_CALUDE_cost_of_paving_floor_l793_79311


namespace NUMINAMATH_CALUDE_part_time_employees_count_l793_79347

def total_employees : ℕ := 65134
def full_time_employees : ℕ := 63093

theorem part_time_employees_count : total_employees - full_time_employees = 2041 := by
  sorry

end NUMINAMATH_CALUDE_part_time_employees_count_l793_79347


namespace NUMINAMATH_CALUDE_parabola_point_x_coordinate_l793_79323

/-- The x-coordinate of a point on the parabola y^2 = 4x that is 4 units away from the focus -/
theorem parabola_point_x_coordinate (x y : ℝ) : 
  y^2 = 4*x →                            -- Point (x, y) is on the parabola y^2 = 4x
  (x - 1)^2 + y^2 = 4^2 →                -- Distance from (x, y) to focus (1, 0) is 4
  x = 3 := by
sorry

end NUMINAMATH_CALUDE_parabola_point_x_coordinate_l793_79323


namespace NUMINAMATH_CALUDE_triangle_inequality_l793_79384

theorem triangle_inequality (a b c : ℝ) (x y z : ℝ) 
  (h1 : a ≥ b) (h2 : b ≥ c) (h3 : c > 0) 
  (h4 : 0 ≤ x ∧ x ≤ π) (h5 : 0 ≤ y ∧ y ≤ π) (h6 : 0 ≤ z ∧ z ≤ π) 
  (h7 : x + y + z = π) : 
  b * c + c * a - a * b < b * c * Real.cos x + c * a * Real.cos y + a * b * Real.cos z ∧
  b * c * Real.cos x + c * a * Real.cos y + a * b * Real.cos z ≤ (a^2 + b^2 + c^2) / 2 :=
by sorry

end NUMINAMATH_CALUDE_triangle_inequality_l793_79384


namespace NUMINAMATH_CALUDE_max_area_inscribed_rectangle_l793_79343

-- Define a circle with radius R
variable (R : ℝ) (hR : R > 0)

-- Define an inscribed rectangle with side lengths x and y
def inscribed_rectangle (x y : ℝ) : Prop :=
  x > 0 ∧ y > 0 ∧ x^2 + y^2 = (2*R)^2

-- Define the area of a rectangle
def rectangle_area (x y : ℝ) : ℝ := x * y

-- Theorem: The area of any inscribed rectangle is less than or equal to 2R^2
theorem max_area_inscribed_rectangle (x y : ℝ) 
  (h : inscribed_rectangle R x y) : rectangle_area x y ≤ 2 * R^2 := by
  sorry

-- Note: The actual proof is omitted and replaced with 'sorry'

end NUMINAMATH_CALUDE_max_area_inscribed_rectangle_l793_79343


namespace NUMINAMATH_CALUDE_trig_expression_equals_three_halves_l793_79316

theorem trig_expression_equals_three_halves :
  (Real.sin (30 * π / 180) - 1) ^ 0 - Real.sqrt 2 * Real.sin (45 * π / 180) +
  Real.tan (60 * π / 180) * Real.cos (30 * π / 180) = 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_trig_expression_equals_three_halves_l793_79316


namespace NUMINAMATH_CALUDE_pineapple_shipping_cost_l793_79377

/-- The shipping cost for a dozen pineapples, given the initial cost and total cost per pineapple. -/
theorem pineapple_shipping_cost 
  (initial_cost : ℚ)  -- Cost of each pineapple before shipping
  (total_cost : ℚ)    -- Total cost of each pineapple including shipping
  (h1 : initial_cost = 1.25)  -- Each pineapple costs $1.25
  (h2 : total_cost = 3)       -- Each pineapple ends up costing $3
  : (12 : ℚ) * (total_cost - initial_cost) = 21 := by
  sorry

#check pineapple_shipping_cost

end NUMINAMATH_CALUDE_pineapple_shipping_cost_l793_79377


namespace NUMINAMATH_CALUDE_remainder_problem_l793_79372

theorem remainder_problem (n : ℕ) : 
  n % 101 = 0 ∧ n / 101 = 347 → n % 89 = 70 := by
  sorry

end NUMINAMATH_CALUDE_remainder_problem_l793_79372


namespace NUMINAMATH_CALUDE_students_liking_both_desserts_l793_79354

/-- Proves the number of students liking both apple pie and chocolate cake in a class --/
theorem students_liking_both_desserts 
  (total_students : ℕ) 
  (apple_pie_fans : ℕ) 
  (chocolate_cake_fans : ℕ) 
  (neither_fans : ℕ) 
  (only_cookies_fans : ℕ) 
  (h1 : total_students = 50)
  (h2 : apple_pie_fans = 22)
  (h3 : chocolate_cake_fans = 20)
  (h4 : neither_fans = 10)
  (h5 : only_cookies_fans = 5)
  : ∃ (both_fans : ℕ), both_fans = 7 := by
  sorry


end NUMINAMATH_CALUDE_students_liking_both_desserts_l793_79354


namespace NUMINAMATH_CALUDE_tangent_roots_expression_value_l793_79352

theorem tangent_roots_expression_value (α β : Real) : 
  (∃ x y : Real, x^2 - 4*x - 2 = 0 ∧ y^2 - 4*y - 2 = 0 ∧ x ≠ y ∧ Real.tan α = x ∧ Real.tan β = y) →
  (Real.cos (α + β))^2 + 2*(Real.sin (α + β))*(Real.cos (α + β)) - 2*(Real.sin (α + β))^2 = 1/25 := by
  sorry

end NUMINAMATH_CALUDE_tangent_roots_expression_value_l793_79352


namespace NUMINAMATH_CALUDE_twenty_fifth_term_of_sequence_l793_79382

def arithmetic_sequence (a₁ : ℝ) (d : ℝ) (n : ℕ) : ℝ := a₁ + (n - 1 : ℝ) * d

theorem twenty_fifth_term_of_sequence : 
  let a₁ := 2
  let a₂ := 5
  let d := a₂ - a₁
  arithmetic_sequence a₁ d 25 = 74 := by
sorry

end NUMINAMATH_CALUDE_twenty_fifth_term_of_sequence_l793_79382


namespace NUMINAMATH_CALUDE_f_property_l793_79324

theorem f_property (f : ℝ → ℝ) 
  (h : ∀ x y z : ℝ, f (x^2 + y * f z + z) = x * f x + z * f y + f z) :
  (∃ a b : ℝ, (∀ x : ℝ, f x = a ∨ f x = b) ∧ f 5 = a ∨ f 5 = b) ∧
  (∃ s : ℝ, s = (f 5 + f 5) ∧ s = 5) :=
sorry

end NUMINAMATH_CALUDE_f_property_l793_79324


namespace NUMINAMATH_CALUDE_video_game_enemies_l793_79326

/-- The number of points earned per defeated enemy -/
def points_per_enemy : ℕ := 3

/-- The number of enemies left undefeated -/
def undefeated_enemies : ℕ := 2

/-- The total points earned -/
def total_points : ℕ := 12

/-- The total number of enemies in the level -/
def total_enemies : ℕ := 6

theorem video_game_enemies :
  total_enemies = (total_points / points_per_enemy) + undefeated_enemies :=
sorry

end NUMINAMATH_CALUDE_video_game_enemies_l793_79326


namespace NUMINAMATH_CALUDE_area_of_three_presentable_set_l793_79388

/-- A complex number is three-presentable if there exists a complex number w
    with |w| = 3 such that z = w - 1/w -/
def ThreePresentable (z : ℂ) : Prop :=
  ∃ w : ℂ, Complex.abs w = 3 ∧ z = w - 1 / w

/-- T is the set of all three-presentable complex numbers -/
def T : Set ℂ :=
  {z : ℂ | ThreePresentable z}

/-- The area of a set in the complex plane -/
noncomputable def AreaInside (S : Set ℂ) : ℝ := sorry

theorem area_of_three_presentable_set :
  AreaInside T = (80 / 9) * Real.pi := by sorry

end NUMINAMATH_CALUDE_area_of_three_presentable_set_l793_79388


namespace NUMINAMATH_CALUDE_distance_to_focus_l793_79305

def parabola (x y : ℝ) : Prop := y^2 = 4*x

def focus (x y : ℝ) : Prop := x = 1 ∧ y = 0

theorem distance_to_focus (x y : ℝ) (h1 : parabola x y) (h2 : x = 3) : 
  ∃ (fx fy : ℝ), focus fx fy ∧ Real.sqrt ((x - fx)^2 + (y - fy)^2) = 4 :=
sorry

end NUMINAMATH_CALUDE_distance_to_focus_l793_79305


namespace NUMINAMATH_CALUDE_circle_covering_theorem_l793_79341

/-- A point in the plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- A set of n points in the plane -/
def PointSet (n : ℕ) := Fin n → Point

/-- A circle in the plane -/
structure Circle where
  center : Point
  radius : ℝ

/-- Predicate to check if a point is covered by a circle -/
def covered (p : Point) (c : Circle) : Prop :=
  (p.x - c.center.x)^2 + (p.y - c.center.y)^2 ≤ c.radius^2

/-- Predicate to check if a set of points can be covered by a circle -/
def canBeCovered (S : Set Point) (c : Circle) : Prop :=
  ∀ p ∈ S, covered p c

theorem circle_covering_theorem (n : ℕ) (points : PointSet n) :
  (∀ (i j k : Fin n), ∃ (c : Circle), c.radius = 1 ∧ 
    canBeCovered {points i, points j, points k} c) →
  ∃ (c : Circle), c.radius = 1 ∧ canBeCovered (Set.range points) c :=
sorry

end NUMINAMATH_CALUDE_circle_covering_theorem_l793_79341


namespace NUMINAMATH_CALUDE_expression_value_l793_79362

theorem expression_value (x y z w : ℝ) 
  (h1 : 4 * x * z + y * w = 3) 
  (h2 : x * w + y * z = 6) : 
  (2 * x + y) * (2 * z + w) = 15 := by
  sorry

end NUMINAMATH_CALUDE_expression_value_l793_79362


namespace NUMINAMATH_CALUDE_sequence_a_correct_l793_79380

def sequence_a (n : ℕ) : ℚ :=
  (2 * 3^n) / (3^n - 1)

theorem sequence_a_correct (n : ℕ) : 
  n ≥ 1 → 
  sequence_a (n + 1) = (3^(n + 1) * sequence_a n) / (sequence_a n + 3^(n + 1)) ∧
  sequence_a 1 = 3 := by
sorry

end NUMINAMATH_CALUDE_sequence_a_correct_l793_79380


namespace NUMINAMATH_CALUDE_exp_five_factorial_30_l793_79302

/-- The exponent of 5 in the prime factorization of n! -/
def exp_five_factorial (n : ℕ) : ℕ :=
  (n / 5) + (n / 25)

/-- Theorem: The exponent of 5 in the prime factorization of 30! is 7 -/
theorem exp_five_factorial_30 : exp_five_factorial 30 = 7 := by
  sorry

end NUMINAMATH_CALUDE_exp_five_factorial_30_l793_79302


namespace NUMINAMATH_CALUDE_special_arithmetic_sequence_general_term_l793_79373

/-- An arithmetic sequence with a1 = 4 and a1, a5, a13 forming a geometric sequence -/
structure SpecialArithmeticSequence where
  a : ℕ → ℝ
  is_arithmetic : ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d
  a1_eq_4 : a 1 = 4
  geometric_subsequence : ∃ r : ℝ, a 5 = a 1 * r ∧ a 13 = a 5 * r

/-- The general term formula for the special arithmetic sequence -/
def general_term (seq : SpecialArithmeticSequence) (n : ℕ) : ℝ :=
  n + 3

theorem special_arithmetic_sequence_general_term (seq : SpecialArithmeticSequence) :
  ∀ n : ℕ, seq.a n = general_term seq n ∨ seq.a n = 4 := by
  sorry

end NUMINAMATH_CALUDE_special_arithmetic_sequence_general_term_l793_79373


namespace NUMINAMATH_CALUDE_safe_game_probabilities_l793_79393

/-- The probability of opening all safes given the number of safes and initially opened safes. -/
def P (m n : ℕ) : ℚ :=
  sorry

theorem safe_game_probabilities (n : ℕ) (h : n ≥ 2) :
  P 2 3 = 2/3 ∧
  (∀ k, P 1 k = 1/k) ∧
  (∀ k ≥ 2, P 2 k = (2/k) * P 1 (k-1) + ((k-2)/k) * P 2 (k-1)) ∧
  (∀ k ≥ 2, P 2 k = 2/k) :=
sorry

end NUMINAMATH_CALUDE_safe_game_probabilities_l793_79393


namespace NUMINAMATH_CALUDE_runner_time_difference_l793_79315

theorem runner_time_difference 
  (x y : ℝ) 
  (h1 : y - x / 2 = 12) 
  (h2 : x - y / 2 = 36) : 
  2 * y - 2 * x = -16 := by
sorry

end NUMINAMATH_CALUDE_runner_time_difference_l793_79315


namespace NUMINAMATH_CALUDE_cos_double_angle_special_case_l793_79329

/-- Given a vector a = (cos α, 1/2) with magnitude √2/2, prove that cos 2α = -1/2 -/
theorem cos_double_angle_special_case (α : ℝ) (a : ℝ × ℝ) :
  a = (Real.cos α, (1 : ℝ) / 2) →
  Real.sqrt ((a.1 ^ 2) + (a.2 ^ 2)) = Real.sqrt 2 / 2 →
  Real.cos (2 * α) = -(1 : ℝ) / 2 := by
  sorry

end NUMINAMATH_CALUDE_cos_double_angle_special_case_l793_79329


namespace NUMINAMATH_CALUDE_egor_can_always_achieve_two_roots_egor_cannot_always_achieve_more_than_two_roots_l793_79385

/-- Represents a player in the polynomial coefficient game -/
inductive Player
| Igor
| Egor

/-- Represents the state of the game after each move -/
structure GameState where
  coefficients : Vector ℤ 100
  currentPlayer : Player

/-- A strategy for a player in the game -/
def Strategy := GameState → ℕ → ℤ

/-- The game play function -/
def play (igorStrategy : Strategy) (egorStrategy : Strategy) : Vector ℤ 100 := sorry

/-- Counts the number of distinct integer roots of a polynomial -/
def countDistinctIntegerRoots (coeffs : Vector ℤ 100) : ℕ := sorry

/-- The main theorem stating Egor can always achieve 2 distinct integer roots -/
theorem egor_can_always_achieve_two_roots :
  ∃ (egorStrategy : Strategy),
    ∀ (igorStrategy : Strategy),
      countDistinctIntegerRoots (play igorStrategy egorStrategy) ≥ 2 := sorry

/-- The main theorem stating Egor cannot always achieve more than 2 distinct integer roots -/
theorem egor_cannot_always_achieve_more_than_two_roots :
  ∀ (egorStrategy : Strategy),
    ∃ (igorStrategy : Strategy),
      countDistinctIntegerRoots (play igorStrategy egorStrategy) ≤ 2 := sorry

end NUMINAMATH_CALUDE_egor_can_always_achieve_two_roots_egor_cannot_always_achieve_more_than_two_roots_l793_79385


namespace NUMINAMATH_CALUDE_lemon_heads_in_package_l793_79318

/-- The number of Lemon Heads Louis ate -/
def total_lemon_heads : ℕ := 54

/-- The number of whole boxes Louis ate -/
def whole_boxes : ℕ := 9

/-- The number of Lemon Heads in one package -/
def lemon_heads_per_package : ℕ := total_lemon_heads / whole_boxes

theorem lemon_heads_in_package : lemon_heads_per_package = 6 := by
  sorry

end NUMINAMATH_CALUDE_lemon_heads_in_package_l793_79318


namespace NUMINAMATH_CALUDE_least_number_with_conditions_l793_79344

theorem least_number_with_conditions : ∃ n : ℕ, 
  (n = 1262) ∧ 
  (∀ k : ℕ, 2 ≤ k ∧ k ≤ 7 → n % k = 2) ∧
  (n % 13 = 0) ∧
  (∀ m : ℕ, m < n → ¬(∀ k : ℕ, 2 ≤ k ∧ k ≤ 7 → m % k = 2) ∨ m % 13 ≠ 0) :=
by sorry

end NUMINAMATH_CALUDE_least_number_with_conditions_l793_79344


namespace NUMINAMATH_CALUDE_no_condition_satisfies_equations_l793_79306

theorem no_condition_satisfies_equations (a b c : ℤ) : 
  a + b + c = 3 →
  (∀ (condition : Prop), 
    (condition = (a = b ∧ b = c ∧ c = 1) ∨
     condition = (a = b - 1 ∧ b = c - 1) ∨
     condition = (a = b ∧ b = c) ∨
     condition = (a > c ∧ c = b - 1)) →
    ¬(condition → a*(a-b)^3 + b*(b-c)^3 + c*(c-a)^3 = 3)) :=
by sorry

end NUMINAMATH_CALUDE_no_condition_satisfies_equations_l793_79306


namespace NUMINAMATH_CALUDE_expression_evaluation_l793_79309

theorem expression_evaluation :
  let a : ℚ := -3/2
  let expr := 1 + (1 - a) / a / ((a^2 - 1) / (a^2 + 2*a))
  expr = 2 := by sorry

end NUMINAMATH_CALUDE_expression_evaluation_l793_79309


namespace NUMINAMATH_CALUDE_triangle_construction_feasibility_l793_79340

/-- Given a triangle with sides a and c, and angle condition α = 2β, 
    the triangle construction is feasible if and only if a > (2/3)c -/
theorem triangle_construction_feasibility (a c : ℝ) (α β : ℝ) 
  (h_positive_a : a > 0) (h_positive_c : c > 0) (h_angle : α = 2 * β) :
  (∃ b : ℝ, b > 0 ∧ a + b > c ∧ b + c > a ∧ c + a > b) ↔ a > (2/3) * c := by
sorry

end NUMINAMATH_CALUDE_triangle_construction_feasibility_l793_79340


namespace NUMINAMATH_CALUDE_opposite_rational_division_l793_79332

theorem opposite_rational_division (a : ℚ) : 
  (a ≠ 0 → a / (-a) = -1) ∧ (a = 0 → a / (-a) = 0/0) :=
sorry

end NUMINAMATH_CALUDE_opposite_rational_division_l793_79332


namespace NUMINAMATH_CALUDE_set_equality_l793_79397

universe u

def U : Set ℕ := {1, 2, 3, 4, 5, 6, 7}
def M : Set ℕ := {3, 4, 5}
def N : Set ℕ := {1, 3, 6}

theorem set_equality : (Uᶜ ∪ M) ∩ (Uᶜ ∪ N) = {2, 7} := by sorry

end NUMINAMATH_CALUDE_set_equality_l793_79397


namespace NUMINAMATH_CALUDE_test_score_ranges_l793_79383

theorem test_score_ranges (range1 range2 range3 : ℕ) : 
  range1 ≤ range2 ∧ range2 ≤ range3 →  -- Assuming ranges are ordered
  range1 ≥ 30 →                        -- Minimum range is 30
  range3 = 32 →                        -- One range is 32
  range2 = 18 :=                       -- Prove second range is 18
by sorry

end NUMINAMATH_CALUDE_test_score_ranges_l793_79383


namespace NUMINAMATH_CALUDE_modified_rectangle_areas_l793_79317

/-- Represents a rectangle with integer dimensions -/
structure Rectangle where
  length : ℕ
  width : ℕ

/-- Calculates the area of a rectangle -/
def area (r : Rectangle) : ℕ := r.length * r.width

/-- The original rectangle -/
def original : Rectangle := { length := 7, width := 5 }

/-- Theorem stating the relationship between the two modified rectangles -/
theorem modified_rectangle_areas :
  ∃ (r1 r2 : Rectangle),
    (r1.length = original.length ∧ r1.width + 2 = original.width ∧ area r1 = 21) →
    (r2.width = original.width ∧ r2.length + 2 = original.length) →
    area r2 = 25 := by
  sorry

end NUMINAMATH_CALUDE_modified_rectangle_areas_l793_79317


namespace NUMINAMATH_CALUDE_material_used_calculation_l793_79314

-- Define the materials and their quantities
def first_material_bought : ℚ := 4/9
def second_material_bought : ℚ := 2/3
def third_material_bought : ℚ := 5/6

def first_material_left : ℚ := 8/18
def second_material_left : ℚ := 3/9
def third_material_left : ℚ := 2/12

-- Define conversion factors
def sq_meter_to_sq_yard : ℚ := 1196/1000

-- Define the theorem
theorem material_used_calculation :
  let first_used := first_material_bought - first_material_left
  let second_used := second_material_bought - second_material_left
  let third_used := (third_material_bought - third_material_left) * sq_meter_to_sq_yard
  let total_used := first_used + second_used + third_used
  ∃ (ε : ℚ), ε > 0 ∧ ε < 1/1000 ∧ abs (total_used - 1130666/1000000) < ε :=
by sorry

end NUMINAMATH_CALUDE_material_used_calculation_l793_79314


namespace NUMINAMATH_CALUDE_smallest_n_for_grape_contest_l793_79367

theorem smallest_n_for_grape_contest : ∃ (c : ℕ+), 
  (c : ℕ) * (89 - c + 1) = 2009 ∧ 
  89 ≥ 2 * (c - 1) ∧
  ∀ (n : ℕ), n < 89 → ¬(∃ (d : ℕ+), (d : ℕ) * (n - d + 1) = 2009 ∧ n ≥ 2 * (d - 1)) :=
by sorry

end NUMINAMATH_CALUDE_smallest_n_for_grape_contest_l793_79367


namespace NUMINAMATH_CALUDE_subset_implies_a_range_l793_79360

-- Define the sets M and N
def M : Set ℝ := {x | x^2 - 2*x < 0}
def N (a : ℝ) : Set ℝ := {x | x < a}

-- State the theorem
theorem subset_implies_a_range (a : ℝ) : M ⊆ N a → a ∈ Set.Ici 2 := by
  sorry

end NUMINAMATH_CALUDE_subset_implies_a_range_l793_79360


namespace NUMINAMATH_CALUDE_polynomial_expansion_coefficient_l793_79334

theorem polynomial_expansion_coefficient (x : ℝ) : 
  ∃ (a a₁ a₂ a₃ a₄ a₅ a₆ a₇ a₈ a₉ a₁₀ : ℝ), 
    (x^10 = a + a₁*(x-1) + a₂*(x-1)^2 + a₃*(x-1)^3 + a₄*(x-1)^4 + 
            a₅*(x-1)^5 + a₆*(x-1)^6 + a₇*(x-1)^7 + a₈*(x-1)^8 + 
            a₉*(x-1)^9 + a₁₀*(x-1)^10) ∧ 
    a₇ = 120 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_expansion_coefficient_l793_79334


namespace NUMINAMATH_CALUDE_prob_standard_bulb_l793_79375

/-- Probability of selecting a light bulb from the first factory -/
def p_factory1 : ℝ := 0.2

/-- Probability of selecting a light bulb from the second factory -/
def p_factory2 : ℝ := 0.3

/-- Probability of selecting a light bulb from the third factory -/
def p_factory3 : ℝ := 0.5

/-- Probability of producing a defective light bulb in the first factory -/
def q1 : ℝ := 0.01

/-- Probability of producing a defective light bulb in the second factory -/
def q2 : ℝ := 0.005

/-- Probability of producing a defective light bulb in the third factory -/
def q3 : ℝ := 0.006

/-- Theorem: The probability of randomly selecting a standard (non-defective) light bulb -/
theorem prob_standard_bulb : 
  p_factory1 * (1 - q1) + p_factory2 * (1 - q2) + p_factory3 * (1 - q3) = 0.9935 := by
  sorry

end NUMINAMATH_CALUDE_prob_standard_bulb_l793_79375


namespace NUMINAMATH_CALUDE_max_d_value_l793_79338

/-- Represents a 6-digit number of the form 7d7,33e -/
def SixDigitNumber (d e : ℕ) : ℕ := 700000 + d * 10000 + 7000 + 330 + e

/-- Checks if a natural number is a digit (0-9) -/
def isDigit (n : ℕ) : Prop := n ≥ 0 ∧ n ≤ 9

/-- The main theorem stating the maximum value of d -/
theorem max_d_value : 
  ∃ (d e : ℕ), isDigit d ∧ isDigit e ∧ 
  (SixDigitNumber d e) % 33 = 0 ∧
  ∀ (d' e' : ℕ), isDigit d' ∧ isDigit e' ∧ (SixDigitNumber d' e') % 33 = 0 → d' ≤ d :=
sorry

end NUMINAMATH_CALUDE_max_d_value_l793_79338


namespace NUMINAMATH_CALUDE_complex_number_problem_l793_79395

theorem complex_number_problem (a b : ℝ) (i : ℂ) : 
  (a - 2*i) * i = b - i → a^2 + b^2 = 5 := by sorry

end NUMINAMATH_CALUDE_complex_number_problem_l793_79395


namespace NUMINAMATH_CALUDE_rectangle_area_difference_l793_79369

/-- Given a large rectangle of dimensions A × B and a small rectangle of dimensions a × b,
    where the small rectangle is entirely contained within the large rectangle,
    this theorem proves that the absolute difference between the total area of the parts
    of the small rectangle outside the large rectangle and the area of the large rectangle
    not covered by the small rectangle is equal to 572, given specific dimensions. -/
theorem rectangle_area_difference (A B a b : ℝ) 
    (h1 : A = 20) (h2 : B = 30) (h3 : a = 4) (h4 : b = 7)
    (h5 : a ≤ A ∧ b ≤ B) : 
    |0 - (A * B - a * b)| = 572 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_area_difference_l793_79369


namespace NUMINAMATH_CALUDE_sqrt_a_squared_plus_one_is_quadratic_radical_l793_79349

/-- A function is a quadratic radical if it involves a square root and its radicand is non-negative for all real inputs. -/
def is_quadratic_radical (f : ℝ → ℝ) : Prop :=
  ∃ g : ℝ → ℝ, (∀ x, g x ≥ 0) ∧ (∀ x, f x = Real.sqrt (g x))

/-- The function f(a) = √(a² + 1) is a quadratic radical. -/
theorem sqrt_a_squared_plus_one_is_quadratic_radical :
  is_quadratic_radical (fun a : ℝ ↦ Real.sqrt (a^2 + 1)) :=
sorry

end NUMINAMATH_CALUDE_sqrt_a_squared_plus_one_is_quadratic_radical_l793_79349


namespace NUMINAMATH_CALUDE_rainfall_difference_l793_79339

/-- The number of Mondays -/
def numMondays : ℕ := 7

/-- The number of Tuesdays -/
def numTuesdays : ℕ := 9

/-- The amount of rain on each Monday in centimeters -/
def rainPerMonday : ℝ := 1.5

/-- The amount of rain on each Tuesday in centimeters -/
def rainPerTuesday : ℝ := 2.5

/-- The difference in total rainfall between Tuesdays and Mondays -/
theorem rainfall_difference : 
  (numTuesdays : ℝ) * rainPerTuesday - (numMondays : ℝ) * rainPerMonday = 12 := by
  sorry

end NUMINAMATH_CALUDE_rainfall_difference_l793_79339


namespace NUMINAMATH_CALUDE_line_slope_intercept_sum_l793_79358

/-- A line with slope 3 passing through (-2, 4) has m + b = 13 when written as y = mx + b -/
theorem line_slope_intercept_sum (m b : ℝ) : 
  m = 3 → 
  4 = m * (-2) + b → 
  m + b = 13 := by sorry

end NUMINAMATH_CALUDE_line_slope_intercept_sum_l793_79358


namespace NUMINAMATH_CALUDE_min_value_trig_expression_l793_79396

theorem min_value_trig_expression (γ δ : ℝ) :
  ∃ (min : ℝ), min = 36 ∧
  ∀ (γ' δ' : ℝ), (3 * Real.cos γ' + 4 * Real.sin δ' - 7)^2 + 
    (3 * Real.sin γ' + 4 * Real.cos δ' - 12)^2 ≥ min ∧
  ∃ (γ₀ δ₀ : ℝ), (3 * Real.cos γ₀ + 4 * Real.sin δ₀ - 7)^2 + 
    (3 * Real.sin γ₀ + 4 * Real.cos δ₀ - 12)^2 = min :=
by sorry

end NUMINAMATH_CALUDE_min_value_trig_expression_l793_79396


namespace NUMINAMATH_CALUDE_largest_rational_root_quadratic_l793_79310

theorem largest_rational_root_quadratic (a b c : ℕ+) 
  (ha : a ≤ 100) (hb : b ≤ 100) (hc : c ≤ 100) :
  let roots := {x : ℚ | a * x^2 + b * x + c = 0}
  ∃ (max_root : ℚ), max_root ∈ roots ∧ 
    ∀ (r : ℚ), r ∈ roots → r ≤ max_root ∧
    max_root = -1 / 99 := by
  sorry

end NUMINAMATH_CALUDE_largest_rational_root_quadratic_l793_79310


namespace NUMINAMATH_CALUDE_percentage_difference_l793_79336

theorem percentage_difference (p q : ℝ) (h : p = 1.25 * q) : 
  (p - q) / p = 0.2 := by
sorry

end NUMINAMATH_CALUDE_percentage_difference_l793_79336


namespace NUMINAMATH_CALUDE_fixed_point_of_function_l793_79389

theorem fixed_point_of_function (a : ℝ) :
  let f : ℝ → ℝ := λ x ↦ 4 + a^(x - 1)
  f 1 = 5 := by sorry

end NUMINAMATH_CALUDE_fixed_point_of_function_l793_79389


namespace NUMINAMATH_CALUDE_distance_to_point_l793_79355

-- Define the point
def point : ℝ × ℝ := (-12, 5)

-- Define the origin
def origin : ℝ × ℝ := (0, 0)

-- Theorem statement
theorem distance_to_point : Real.sqrt ((point.1 - origin.1)^2 + (point.2 - origin.2)^2) = 13 := by
  sorry

end NUMINAMATH_CALUDE_distance_to_point_l793_79355


namespace NUMINAMATH_CALUDE_length_of_ac_l793_79378

/-- Given 5 consecutive points on a straight line, prove that the length of ac is 11 -/
theorem length_of_ac (a b c d e : Real) : 
  (b - a) = 5 →
  (c - b) = 3 * (d - c) →
  (e - d) = 7 →
  (e - a) = 20 →
  (c - a) = 11 :=
by sorry

end NUMINAMATH_CALUDE_length_of_ac_l793_79378


namespace NUMINAMATH_CALUDE_power_of_power_l793_79351

theorem power_of_power (a : ℝ) : (a ^ 2) ^ 3 = a ^ 6 := by
  sorry

end NUMINAMATH_CALUDE_power_of_power_l793_79351


namespace NUMINAMATH_CALUDE_initial_average_equals_correct_average_l793_79374

/-- The number of values in the set -/
def n : ℕ := 10

/-- The correct average of the numbers -/
def correct_average : ℚ := 401/10

/-- The difference between the first incorrectly copied number and its actual value -/
def first_error : ℤ := 17

/-- The difference between the second incorrectly copied number and its actual value -/
def second_error : ℤ := 13 - 31

/-- The sum of all errors in the incorrectly copied numbers -/
def total_error : ℤ := first_error + second_error

theorem initial_average_equals_correct_average :
  let S := n * correct_average
  let initial_average := (S + total_error) / n
  initial_average = correct_average := by sorry

end NUMINAMATH_CALUDE_initial_average_equals_correct_average_l793_79374


namespace NUMINAMATH_CALUDE_perpendicular_lines_b_value_l793_79356

-- Define the slopes of the two lines
def slope1 : ℚ := -1/5
def slope2 (b : ℚ) : ℚ := -b/4

-- Define the perpendicularity condition
def perpendicular (b : ℚ) : Prop := slope1 * slope2 b = -1

-- Theorem statement
theorem perpendicular_lines_b_value : 
  ∀ b : ℚ, perpendicular b → b = -20 := by
  sorry

end NUMINAMATH_CALUDE_perpendicular_lines_b_value_l793_79356


namespace NUMINAMATH_CALUDE_circle_and_line_intersection_l793_79394

-- Define the circle C
def circle_C (x y : ℝ) : Prop :=
  (x - 2)^2 + (y - 4)^2 = 25

-- Define the moving line l
def line_l (m x y : ℝ) : Prop :=
  (m + 2) * x + (2 * m + 1) * y - 7 * m - 8 = 0

-- Theorem statement
theorem circle_and_line_intersection :
  -- Given conditions
  (circle_C (-2) 1) ∧  -- Circle C passes through A(-2, 1)
  (circle_C 5 0) ∧     -- Circle C passes through B(5, 0)
  (∃ x y : ℝ, circle_C x y ∧ y = 2 * x) →  -- Center of C is on y = 2x
  -- Conclusions
  ((∀ x y : ℝ, circle_C x y ↔ (x - 2)^2 + (y - 4)^2 = 25) ∧
   (∃ min_PQ : ℝ, 
     (min_PQ = 4 * Real.sqrt 5) ∧
     (∀ m x1 y1 x2 y2 : ℝ,
       (circle_C x1 y1 ∧ circle_C x2 y2 ∧ 
        line_l m x1 y1 ∧ line_l m x2 y2) →
       ((x1 - x2)^2 + (y1 - y2)^2 ≥ min_PQ^2))))
  := by sorry

end NUMINAMATH_CALUDE_circle_and_line_intersection_l793_79394


namespace NUMINAMATH_CALUDE_equation_solution_existence_l793_79350

theorem equation_solution_existence (n : ℤ) : 
  (∃ x y z : ℤ, x^2 + y^2 + z^2 - x*y - y*z - z*x = n) → 
  (∃ a b : ℤ, a^2 + b^2 - a*b = n) := by
sorry

end NUMINAMATH_CALUDE_equation_solution_existence_l793_79350


namespace NUMINAMATH_CALUDE_quadratic_roots_condition_l793_79327

theorem quadratic_roots_condition (a : ℝ) :
  (∃ x y : ℝ, x > 0 ∧ y < 0 ∧ x^2 + 2*x + a = 0 ∧ y^2 + 2*y + a = 0) → a < 1 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_roots_condition_l793_79327


namespace NUMINAMATH_CALUDE_equal_area_rectangles_l793_79379

/-- Given two rectangles with equal area, where one rectangle measures 12 inches by 15 inches
    and the other has a width of 30 inches, the length of the second rectangle is 6 inches. -/
theorem equal_area_rectangles (carol_length carol_width jordan_width : ℝ) 
  (h1 : carol_length = 12)
  (h2 : carol_width = 15)
  (h3 : jordan_width = 30)
  (h4 : carol_length * carol_width = jordan_width * (carol_length * carol_width / jordan_width)) :
  carol_length * carol_width / jordan_width = 6 :=
by sorry

end NUMINAMATH_CALUDE_equal_area_rectangles_l793_79379


namespace NUMINAMATH_CALUDE_min_value_of_x_l793_79387

theorem min_value_of_x (x : ℝ) (h1 : x > 0) (h2 : Real.log x / Real.log 3 ≥ 1 + (1/3) * (Real.log x / Real.log 3)) :
  x ≥ 3 * Real.sqrt 3 ∧ ∀ y : ℝ, y > 0 → Real.log y / Real.log 3 ≥ 1 + (1/3) * (Real.log y / Real.log 3) → y ≥ x :=
sorry

end NUMINAMATH_CALUDE_min_value_of_x_l793_79387


namespace NUMINAMATH_CALUDE_number_equation_solution_l793_79321

theorem number_equation_solution :
  ∃ x : ℝ, (3 * x = 2 * x - 7) ∧ (x = -7) := by sorry

end NUMINAMATH_CALUDE_number_equation_solution_l793_79321


namespace NUMINAMATH_CALUDE_man_speed_on_bridge_l793_79359

/-- Calculates the speed of a man crossing a bridge -/
theorem man_speed_on_bridge (bridge_length : ℝ) (crossing_time : ℝ) : 
  bridge_length = 2500 →  -- bridge length in meters
  crossing_time = 15 →    -- crossing time in minutes
  bridge_length / (crossing_time / 60) / 1000 = 10 := by
  sorry

end NUMINAMATH_CALUDE_man_speed_on_bridge_l793_79359


namespace NUMINAMATH_CALUDE_april_flower_sale_earnings_l793_79330

/-- April's flower sale earnings calculation --/
theorem april_flower_sale_earnings : 
  ∀ (initial_roses final_roses price_per_rose : ℕ),
  initial_roses = 9 →
  final_roses = 4 →
  price_per_rose = 7 →
  (initial_roses - final_roses) * price_per_rose = 35 :=
by
  sorry

end NUMINAMATH_CALUDE_april_flower_sale_earnings_l793_79330


namespace NUMINAMATH_CALUDE_farm_has_eleven_goats_l793_79353

/-- Represents the number of animals on a farm -/
structure Farm where
  goats : ℕ
  cows : ℕ
  pigs : ℕ

/-- Defines the properties of the farm in the problem -/
def ProblemFarm (f : Farm) : Prop :=
  (f.pigs = 2 * f.cows) ∧ 
  (f.cows = f.goats + 4) ∧ 
  (f.goats + f.cows + f.pigs = 56)

/-- Theorem stating that a farm satisfying the problem conditions has 11 goats -/
theorem farm_has_eleven_goats (f : Farm) (h : ProblemFarm f) : f.goats = 11 := by
  sorry


end NUMINAMATH_CALUDE_farm_has_eleven_goats_l793_79353


namespace NUMINAMATH_CALUDE_one_seventh_comparison_l793_79348

theorem one_seventh_comparison : (1 : ℚ) / 7 - 142857142857 / 1000000000000 = 1 / (7 * 1000000000000) := by
  sorry

end NUMINAMATH_CALUDE_one_seventh_comparison_l793_79348


namespace NUMINAMATH_CALUDE_area_EFGH_extended_l793_79391

/-- Represents a quadrilateral with extended sides -/
structure ExtendedQuadrilateral where
  ef : ℝ
  fg : ℝ
  gh : ℝ
  he : ℝ
  area : ℝ

/-- Calculates the area of the extended quadrilateral -/
def area_extended_quadrilateral (q : ExtendedQuadrilateral) : ℝ :=
  q.area + 2 * q.area

/-- Theorem stating the area of the extended quadrilateral E'F'G'H' -/
theorem area_EFGH_extended (q : ExtendedQuadrilateral)
  (h_ef : q.ef = 5)
  (h_fg : q.fg = 6)
  (h_gh : q.gh = 7)
  (h_he : q.he = 8)
  (h_area : q.area = 20) :
  area_extended_quadrilateral q = 60 := by
  sorry

end NUMINAMATH_CALUDE_area_EFGH_extended_l793_79391


namespace NUMINAMATH_CALUDE_max_rect_box_length_l793_79363

-- Define the dimensions of the wooden box in centimeters
def wooden_box_length : ℝ := 800
def wooden_box_width : ℝ := 700
def wooden_box_height : ℝ := 600

-- Define the dimensions of the rectangular box in centimeters
def rect_box_width : ℝ := 7
def rect_box_height : ℝ := 6

-- Define the maximum number of rectangular boxes
def max_boxes : ℕ := 2000000

-- Theorem statement
theorem max_rect_box_length :
  ∀ x : ℝ,
  x > 0 →
  (x * rect_box_width * rect_box_height * max_boxes : ℝ) ≤ wooden_box_length * wooden_box_width * wooden_box_height →
  x ≤ 4 := by
sorry


end NUMINAMATH_CALUDE_max_rect_box_length_l793_79363


namespace NUMINAMATH_CALUDE_stratified_sample_female_count_l793_79345

/-- Represents the number of female athletes in a stratified sample -/
def female_athletes_in_sample (total_athletes : ℕ) (female_athletes : ℕ) (sample_size : ℕ) : ℕ :=
  (female_athletes * sample_size) / total_athletes

theorem stratified_sample_female_count :
  female_athletes_in_sample 98 42 28 = 12 := by
  sorry

#eval female_athletes_in_sample 98 42 28

end NUMINAMATH_CALUDE_stratified_sample_female_count_l793_79345


namespace NUMINAMATH_CALUDE_eunjis_rank_l793_79328

/-- Given that Minyoung arrived 33rd in a race and Eunji arrived 11 places after Minyoung,
    prove that Eunji's rank is 44th. -/
theorem eunjis_rank (minyoungs_rank : ℕ) (places_after : ℕ) 
  (h1 : minyoungs_rank = 33) 
  (h2 : places_after = 11) : 
  minyoungs_rank + places_after = 44 := by
  sorry

end NUMINAMATH_CALUDE_eunjis_rank_l793_79328


namespace NUMINAMATH_CALUDE_det_sin_matrix_zero_l793_79303

theorem det_sin_matrix_zero : 
  let A : Matrix (Fin 3) (Fin 3) ℝ := ![![Real.sin 1, Real.sin 2, Real.sin 3],
                                        ![Real.sin 2, Real.sin 3, Real.sin 4],
                                        ![Real.sin 3, Real.sin 4, Real.sin 5]]
  Matrix.det A = 0 := by
  sorry

end NUMINAMATH_CALUDE_det_sin_matrix_zero_l793_79303


namespace NUMINAMATH_CALUDE_shortest_side_right_triangle_l793_79346

theorem shortest_side_right_triangle (a b c : ℝ) (ha : a = 9) (hb : b = 12) 
  (hright : a^2 + c^2 = b^2) : 
  c = Real.sqrt (b^2 - a^2) :=
sorry

end NUMINAMATH_CALUDE_shortest_side_right_triangle_l793_79346


namespace NUMINAMATH_CALUDE_binary_ternary_equality_l793_79312

theorem binary_ternary_equality (a b : ℕ) (h1 : a > 0) (h2 : b > 0) 
  (h3 : b ≤ 1) (h4 : a ≤ 2) (h5 : 9 + 2*b = 9*a + 2) : 2*a + b = 3 := by
  sorry

end NUMINAMATH_CALUDE_binary_ternary_equality_l793_79312


namespace NUMINAMATH_CALUDE_unique_number_with_pairable_divisors_l793_79325

def is_own_divisor (d n : ℕ) : Prop :=
  d ∣ n ∧ d ≠ 1 ∧ d ≠ n

def has_pairable_own_divisors (n : ℕ) : Prop :=
  ∃ (f : ℕ → ℕ), 
    (∀ d, is_own_divisor d n → is_own_divisor (f d) n) ∧
    (∀ d, is_own_divisor d n → f (f d) = d) ∧
    (∀ d, is_own_divisor d n → (f d = d + 545 ∨ f d = d - 545))

theorem unique_number_with_pairable_divisors :
  ∃! n : ℕ, has_pairable_own_divisors n ∧ n = 1094 :=
sorry

end NUMINAMATH_CALUDE_unique_number_with_pairable_divisors_l793_79325
