import Mathlib

namespace NUMINAMATH_CALUDE_power_equality_l3016_301656

theorem power_equality (q : ℕ) : 16^10 = 4^q → q = 20 := by
  sorry

end NUMINAMATH_CALUDE_power_equality_l3016_301656


namespace NUMINAMATH_CALUDE_no_savings_on_joint_purchase_l3016_301639

/-- Represents the cost of a window in dollars -/
def window_cost : ℕ := 100

/-- Represents the number of windows purchased to get free windows -/
def windows_for_offer : ℕ := 9

/-- Represents the number of free windows given in the offer -/
def free_windows : ℕ := 2

/-- Represents the number of windows Dave needs -/
def dave_windows : ℕ := 10

/-- Represents the number of windows Doug needs -/
def doug_windows : ℕ := 9

/-- Calculates the cost of purchasing windows with the special offer -/
def calculate_cost (num_windows : ℕ) : ℕ :=
  let paid_windows := num_windows - (num_windows / windows_for_offer) * free_windows
  paid_windows * window_cost

/-- Theorem stating that there are no savings when Dave and Doug purchase windows together -/
theorem no_savings_on_joint_purchase :
  calculate_cost dave_windows + calculate_cost doug_windows =
  calculate_cost (dave_windows + doug_windows) :=
sorry

end NUMINAMATH_CALUDE_no_savings_on_joint_purchase_l3016_301639


namespace NUMINAMATH_CALUDE_min_points_per_player_l3016_301600

theorem min_points_per_player 
  (num_players : ℕ) 
  (total_points : ℕ) 
  (max_individual_points : ℕ) 
  (h1 : num_players = 12)
  (h2 : total_points = 100)
  (h3 : max_individual_points = 23) :
  ∃ (min_points : ℕ), 
    min_points = 7 ∧ 
    (∃ (scores : List ℕ), 
      scores.length = num_players ∧ 
      scores.sum = total_points ∧
      (∀ s ∈ scores, s ≥ min_points) ∧
      (∃ s ∈ scores, s = max_individual_points) ∧
      (∀ s ∈ scores, s ≤ max_individual_points)) :=
by sorry

end NUMINAMATH_CALUDE_min_points_per_player_l3016_301600


namespace NUMINAMATH_CALUDE_max_coins_distribution_l3016_301612

theorem max_coins_distribution (k : ℕ) : 
  (∀ n : ℕ, n < 100 ∧ ∃ k : ℕ, n = 13 * k + 3) → 
  (∀ m : ℕ, m < 100 ∧ ∃ k : ℕ, m = 13 * k + 3 → m ≤ 91) ∧
  (∃ k : ℕ, 91 = 13 * k + 3) ∧ 
  91 < 100 :=
by sorry

end NUMINAMATH_CALUDE_max_coins_distribution_l3016_301612


namespace NUMINAMATH_CALUDE_min_value_theorem_l3016_301685

theorem min_value_theorem (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : 2 * x + y - 3 = 0) :
  (4 * y - x + 6) / (x * y) ≥ 9 :=
sorry

end NUMINAMATH_CALUDE_min_value_theorem_l3016_301685


namespace NUMINAMATH_CALUDE_rectangles_form_square_l3016_301674

/-- Represents a rectangle with integer dimensions -/
structure Rectangle where
  width : ℕ
  height : ℕ

/-- The set of given rectangles -/
def rectangles : List Rectangle := [
  ⟨1, 2⟩, ⟨7, 10⟩, ⟨6, 5⟩, ⟨8, 12⟩, ⟨9, 3⟩
]

/-- Calculates the area of a rectangle -/
def area (r : Rectangle) : ℕ := r.width * r.height

/-- Theorem: The given rectangles can form a square -/
theorem rectangles_form_square : ∃ (s : ℕ), s > 0 ∧ s * s = (rectangles.map area).sum := by
  sorry

end NUMINAMATH_CALUDE_rectangles_form_square_l3016_301674


namespace NUMINAMATH_CALUDE_line_parabola_intersection_l3016_301604

/-- The line y = kx + 1 and the parabola y^2 = 4x have exactly one point in common if and only if k = 0 or k = 1 -/
theorem line_parabola_intersection (k : ℝ) : 
  (∃! p : ℝ × ℝ, p.2 = k * p.1 + 1 ∧ p.2^2 = 4 * p.1) ↔ k = 0 ∨ k = 1 := by
  sorry

end NUMINAMATH_CALUDE_line_parabola_intersection_l3016_301604


namespace NUMINAMATH_CALUDE_special_triangle_sum_l3016_301690

/-- A right triangle with a special inscribed circle -/
structure SpecialTriangle where
  /-- The perimeter of the triangle -/
  perimeter : ℝ
  /-- The radius of the inscribed circle -/
  radius : ℝ
  /-- The distance from the center of the circle to one vertex -/
  center_to_vertex : ℝ
  /-- The numerator of the fraction representing center_to_vertex -/
  p : ℕ
  /-- The denominator of the fraction representing center_to_vertex -/
  q : ℕ
  /-- The perimeter is 180 -/
  perimeter_eq : perimeter = 180
  /-- The radius is 25 -/
  radius_eq : radius = 25
  /-- center_to_vertex is equal to p/q -/
  center_to_vertex_eq : center_to_vertex = p / q
  /-- p and q are coprime -/
  coprime : Nat.Coprime p q

/-- The main theorem -/
theorem special_triangle_sum (t : SpecialTriangle) : t.p + t.q = 145 := by
  sorry

end NUMINAMATH_CALUDE_special_triangle_sum_l3016_301690


namespace NUMINAMATH_CALUDE_coral_reading_pages_l3016_301603

/-- The number of pages Coral read on the first night -/
def night1 : ℕ := 30

/-- The number of pages Coral read on the second night -/
def night2 : ℕ := 2 * night1 - 2

/-- The number of pages Coral read on the third night -/
def night3 : ℕ := night1 + night2 + 3

/-- The total number of pages Coral read over three nights -/
def totalPages : ℕ := night1 + night2 + night3

/-- Theorem stating that the total number of pages read is 179 -/
theorem coral_reading_pages : totalPages = 179 := by
  sorry

end NUMINAMATH_CALUDE_coral_reading_pages_l3016_301603


namespace NUMINAMATH_CALUDE_hyperbola_eccentricity_l3016_301684

/-- For a hyperbola with equation x²/a² - y²/b² = 1, if the distance between
    its vertices (2a) is one-third of its focal length (2c), then its
    eccentricity (e) is equal to 3. -/
theorem hyperbola_eccentricity (a b c : ℝ) (h : a > 0) (h' : b > 0) (h'' : c > 0) :
  (2 * a = (1/3) * (2 * c)) → (c / a = 3) :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_eccentricity_l3016_301684


namespace NUMINAMATH_CALUDE_max_abs_z_cubed_minus_3z_minus_2_l3016_301632

/-- Given a complex number z with |z| = 1, the maximum value of |z^3 - 3z - 2| is 3√3. -/
theorem max_abs_z_cubed_minus_3z_minus_2 (z : ℂ) (h : Complex.abs z = 1) :
  ∃ (max_val : ℝ), max_val = 3 * Real.sqrt 3 ∧
  ∀ (w : ℂ), Complex.abs w = 1 → Complex.abs (w^3 - 3*w - 2) ≤ max_val :=
sorry

end NUMINAMATH_CALUDE_max_abs_z_cubed_minus_3z_minus_2_l3016_301632


namespace NUMINAMATH_CALUDE_complex_equation_sum_l3016_301610

theorem complex_equation_sum (a b : ℝ) : 
  (a - 2 * Complex.I) * Complex.I = b - Complex.I → a + b = 1 := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_sum_l3016_301610


namespace NUMINAMATH_CALUDE_tangent_line_determines_n_l3016_301661

/-- A curve defined by a cubic function -/
structure CubicCurve where
  m : ℝ
  n : ℝ

/-- A line defined by a linear function -/
structure Line where
  k : ℝ

/-- Checks if a line is tangent to a cubic curve at a given point -/
def is_tangent_at (c : CubicCurve) (l : Line) (x₀ y₀ : ℝ) : Prop :=
  y₀ = x₀^3 + c.m * x₀ + c.n ∧
  y₀ = l.k * x₀ + 2 ∧
  3 * x₀^2 + c.m = l.k

theorem tangent_line_determines_n (c : CubicCurve) (l : Line) :
  is_tangent_at c l 1 4 → c.n = 4 := by sorry

end NUMINAMATH_CALUDE_tangent_line_determines_n_l3016_301661


namespace NUMINAMATH_CALUDE_A_inter_complement_B_l3016_301664

def U : Set Int := Set.univ

def A : Set Int := {-2, -1, 0, 1, 2}

def B : Set Int := {x | x^2 + 2*x = 0}

theorem A_inter_complement_B : A ∩ (U \ B) = {-1, 1, 2} := by sorry

end NUMINAMATH_CALUDE_A_inter_complement_B_l3016_301664


namespace NUMINAMATH_CALUDE_divisibility_problem_l3016_301675

theorem divisibility_problem (d r : ℤ) : 
  d > 1 ∧ 
  (∃ k₁ k₂ k₃ k₄ : ℤ, 
    29 * 11 = k₁ * d + r ∧
    1059 = k₂ * d + r ∧
    1417 = k₃ * d + r ∧
    2312 = k₄ * d + r) →
  d - r = 15 := by
sorry

end NUMINAMATH_CALUDE_divisibility_problem_l3016_301675


namespace NUMINAMATH_CALUDE_flagpole_breaking_point_l3016_301655

theorem flagpole_breaking_point (h : ℝ) (b : ℝ) (t : ℝ) :
  h = 12 ∧ t = 2 ∧ b > 0 →
  b^2 + (h - t)^2 = h^2 →
  b = 2 * Real.sqrt 11 :=
by sorry

end NUMINAMATH_CALUDE_flagpole_breaking_point_l3016_301655


namespace NUMINAMATH_CALUDE_original_egg_count_l3016_301634

/-- Given a jar of eggs, prove that the original number of eggs is 27 
    when 7 eggs are removed and 20 eggs remain. -/
theorem original_egg_count (removed : ℕ) (remaining : ℕ) : removed = 7 → remaining = 20 → removed + remaining = 27 := by
  sorry

end NUMINAMATH_CALUDE_original_egg_count_l3016_301634


namespace NUMINAMATH_CALUDE_average_equation_solution_l3016_301616

theorem average_equation_solution (x : ℚ) : 
  (1 / 3 : ℚ) * ((3 * x + 8) + (7 * x - 3) + (4 * x + 5)) = 5 * x - 6 → x = -28 := by
  sorry

end NUMINAMATH_CALUDE_average_equation_solution_l3016_301616


namespace NUMINAMATH_CALUDE_complex_equality_implies_modulus_l3016_301606

theorem complex_equality_implies_modulus (x y : ℝ) :
  (Complex.I + 1) * Complex.mk x y = 2 →
  Complex.abs (Complex.mk (2*x) y) = Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_complex_equality_implies_modulus_l3016_301606


namespace NUMINAMATH_CALUDE_triangle_area_l3016_301646

/-- Triangle ABC with given properties -/
structure Triangle :=
  (BD : ℝ)
  (DC : ℝ)
  (height : ℝ)
  (hBD : BD = 3)
  (hDC : DC = 2 * BD)
  (hHeight : height = 4)

/-- The area of triangle ABC is 18 square units -/
theorem triangle_area (t : Triangle) : (1/2 : ℝ) * (t.BD + t.DC) * t.height = 18 := by
  sorry

end NUMINAMATH_CALUDE_triangle_area_l3016_301646


namespace NUMINAMATH_CALUDE_smallest_bookmark_count_l3016_301691

theorem smallest_bookmark_count (b : ℕ) : 
  (b > 0) →
  (b % 5 = 4) →
  (b % 6 = 3) →
  (b % 8 = 7) →
  (∀ x : ℕ, x > 0 ∧ x % 5 = 4 ∧ x % 6 = 3 ∧ x % 8 = 7 → x ≥ b) →
  b = 39 := by
sorry

end NUMINAMATH_CALUDE_smallest_bookmark_count_l3016_301691


namespace NUMINAMATH_CALUDE_vasyas_birthday_l3016_301636

-- Define the days of the week
inductive DayOfWeek
  | Sunday
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday
  | Saturday

def next_day (d : DayOfWeek) : DayOfWeek :=
  match d with
  | DayOfWeek.Sunday => DayOfWeek.Monday
  | DayOfWeek.Monday => DayOfWeek.Tuesday
  | DayOfWeek.Tuesday => DayOfWeek.Wednesday
  | DayOfWeek.Wednesday => DayOfWeek.Thursday
  | DayOfWeek.Thursday => DayOfWeek.Friday
  | DayOfWeek.Friday => DayOfWeek.Saturday
  | DayOfWeek.Saturday => DayOfWeek.Sunday

def two_days_after (d : DayOfWeek) : DayOfWeek :=
  next_day (next_day d)

theorem vasyas_birthday (birthday : DayOfWeek) 
  (h1 : next_day birthday ≠ DayOfWeek.Sunday)
  (h2 : two_days_after (next_day birthday) = DayOfWeek.Sunday) :
  birthday = DayOfWeek.Thursday := by
  sorry

end NUMINAMATH_CALUDE_vasyas_birthday_l3016_301636


namespace NUMINAMATH_CALUDE_sqrt_difference_equality_l3016_301673

theorem sqrt_difference_equality : Real.sqrt (49 + 81) - Real.sqrt (36 - 25) = Real.sqrt 130 - Real.sqrt 11 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_difference_equality_l3016_301673


namespace NUMINAMATH_CALUDE_sufficient_but_not_necessary_l3016_301622

theorem sufficient_but_not_necessary (x : ℝ) :
  (∀ x, x > 1 → x^2 + 2*x > 0) ∧
  (∃ x, x^2 + 2*x > 0 ∧ x ≤ 1) :=
by sorry

end NUMINAMATH_CALUDE_sufficient_but_not_necessary_l3016_301622


namespace NUMINAMATH_CALUDE_trapezoid_diagonal_length_l3016_301650

/-- Represents a trapezoid ABCD with diagonal AC -/
structure Trapezoid where
  -- Points
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ
  D : ℝ × ℝ
  -- Lengths
  AB : ℝ
  DC : ℝ
  AD : ℝ
  -- Properties
  is_trapezoid : (B.2 = C.2) ∧ (A.2 = D.2) -- BC parallel to AD
  AB_length : dist A B = AB
  DC_length : dist D C = DC
  AD_length : dist A D = AD

/-- The length of AC in the trapezoid is approximately 30.1 -/
theorem trapezoid_diagonal_length (t : Trapezoid) (h1 : t.AB = 15) (h2 : t.DC = 24) (h3 : t.AD = 7) :
  ∃ ε > 0, abs (dist t.A t.C - 30.1) < ε :=
sorry

end NUMINAMATH_CALUDE_trapezoid_diagonal_length_l3016_301650


namespace NUMINAMATH_CALUDE_x_value_l3016_301602

theorem x_value (x : ℝ) : x = 40 * (1 + 0.2) → x = 48 := by
  sorry

end NUMINAMATH_CALUDE_x_value_l3016_301602


namespace NUMINAMATH_CALUDE_remainder_property_l3016_301683

theorem remainder_property (n : ℕ) (h : n % 25 = 4) : (n + 15) % 5 = 4 := by
  sorry

end NUMINAMATH_CALUDE_remainder_property_l3016_301683


namespace NUMINAMATH_CALUDE_arithmetic_calculation_l3016_301686

theorem arithmetic_calculation : 18 * 36 + 54 * 18 + 18 * 9 = 1782 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_calculation_l3016_301686


namespace NUMINAMATH_CALUDE_vehicles_meeting_time_l3016_301649

-- Define the vehicles
structure Vehicle where
  id : Nat
  speed : ℝ

-- Define the meeting points
structure MeetingPoint where
  vehicle1 : Vehicle
  vehicle2 : Vehicle
  time : ℝ

-- Define the problem
theorem vehicles_meeting_time
  (v1 v2 v3 v4 : Vehicle)
  (m12 m13 m14 m24 m34 : MeetingPoint)
  (h1 : m12.vehicle1 = v1 ∧ m12.vehicle2 = v2 ∧ m12.time = 0)
  (h2 : m13.vehicle1 = v1 ∧ m13.vehicle2 = v3 ∧ m13.time = 220)
  (h3 : m14.vehicle1 = v1 ∧ m14.vehicle2 = v4 ∧ m14.time = 280)
  (h4 : m24.vehicle1 = v2 ∧ m24.vehicle2 = v4 ∧ m24.time = 240)
  (h5 : m34.vehicle1 = v3 ∧ m34.vehicle2 = v4 ∧ m34.time = 130)
  (h_constant_speed : ∀ v : Vehicle, v.speed > 0)
  : ∃ m23 : MeetingPoint, m23.vehicle1 = v2 ∧ m23.vehicle2 = v3 ∧ m23.time = 200 :=
sorry

end NUMINAMATH_CALUDE_vehicles_meeting_time_l3016_301649


namespace NUMINAMATH_CALUDE_eighteen_percent_of_42_equals_27_percent_of_x_l3016_301687

theorem eighteen_percent_of_42_equals_27_percent_of_x (x : ℝ) : 
  (18 / 100) * 42 = (27 / 100) * x → x = 28 := by
sorry

end NUMINAMATH_CALUDE_eighteen_percent_of_42_equals_27_percent_of_x_l3016_301687


namespace NUMINAMATH_CALUDE_pq_length_is_1098_over_165_l3016_301643

/-- The line y = (5/3)x --/
def line1 (x y : ℝ) : Prop := y = (5/3) * x

/-- The line y = (5/12)x --/
def line2 (x y : ℝ) : Prop := y = (5/12) * x

/-- The midpoint of two points --/
def is_midpoint (mx my px py qx qy : ℝ) : Prop :=
  mx = (px + qx) / 2 ∧ my = (py + qy) / 2

/-- The squared distance between two points --/
def distance_squared (x1 y1 x2 y2 : ℝ) : ℝ :=
  (x2 - x1)^2 + (y2 - y1)^2

theorem pq_length_is_1098_over_165 :
  ∀ (px py qx qy : ℝ),
    line1 px py →
    line2 qx qy →
    is_midpoint 10 8 px py qx qy →
    distance_squared px py qx qy = (1098/165)^2 := by
  sorry

end NUMINAMATH_CALUDE_pq_length_is_1098_over_165_l3016_301643


namespace NUMINAMATH_CALUDE_triangle_inequality_l3016_301688

theorem triangle_inequality (a b c : ℝ) (A B C : ℝ) : 
  a > 0 → b > 0 → c > 0 → 
  A > 0 → B > 0 → C > 0 → 
  A + B + C = π →
  (a + b) * Real.sin (C / 2) + (b + c) * Real.sin (A / 2) + (c + a) * Real.sin (B / 2) ≤ a + b + c := by
  sorry

end NUMINAMATH_CALUDE_triangle_inequality_l3016_301688


namespace NUMINAMATH_CALUDE_choose_4_from_10_l3016_301653

theorem choose_4_from_10 : Nat.choose 10 4 = 210 := by
  sorry

end NUMINAMATH_CALUDE_choose_4_from_10_l3016_301653


namespace NUMINAMATH_CALUDE_final_balance_is_214_12_l3016_301658

/-- Calculates the credit card balance after five months given the initial balance and monthly transactions. -/
def creditCardBalance (initialBalance : ℚ) 
  (month1Interest : ℚ)
  (month2Spent month2Payment month2Interest : ℚ)
  (month3Spent month3Payment month3Interest : ℚ)
  (month4Spent month4Payment month4Interest : ℚ)
  (month5Spent month5Payment month5Interest : ℚ) : ℚ :=
  let balance1 := initialBalance * (1 + month1Interest)
  let balance2 := (balance1 + month2Spent - month2Payment) * (1 + month2Interest)
  let balance3 := (balance2 + month3Spent - month3Payment) * (1 + month3Interest)
  let balance4 := (balance3 + month4Spent - month4Payment) * (1 + month4Interest)
  let balance5 := (balance4 + month5Spent - month5Payment) * (1 + month5Interest)
  balance5

/-- Theorem stating that the credit card balance after five months is $214.12 given the specific transactions. -/
theorem final_balance_is_214_12 : 
  creditCardBalance 50 0.2 20 15 0.18 30 5 0.22 25 20 0.15 40 10 0.2 = 214.12 := by
  sorry

end NUMINAMATH_CALUDE_final_balance_is_214_12_l3016_301658


namespace NUMINAMATH_CALUDE_quadratic_factorization_l3016_301644

theorem quadratic_factorization :
  ∀ x : ℝ, 12 * x^2 - 40 * x + 25 = (2 * Real.sqrt 3 * x - 5)^2 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_factorization_l3016_301644


namespace NUMINAMATH_CALUDE_shina_probability_l3016_301630

def word : Finset Char := {'М', 'А', 'Ш', 'И', 'Н', 'А'}

def draw_probability (word : Finset Char) (target : List Char) : ℚ :=
  let n := word.card
  let prob := target.foldl (λ acc c =>
    acc * (word.filter (λ x => x = c)).card / n) 1
  prob * (n - 1) * (n - 2) * (n - 3) / n

theorem shina_probability :
  draw_probability word ['Ш', 'И', 'Н', 'А'] = 1 / 180 := by
  sorry

end NUMINAMATH_CALUDE_shina_probability_l3016_301630


namespace NUMINAMATH_CALUDE_original_group_size_l3016_301668

theorem original_group_size (initial_days : ℕ) (absent_men : ℕ) (final_days : ℕ) : ℕ :=
  let original_size := 42
  let work_amount := original_size * initial_days
  have h1 : work_amount = (original_size - absent_men) * final_days := by sorry
  have h2 : initial_days = 12 := by sorry
  have h3 : absent_men = 6 := by sorry
  have h4 : final_days = 14 := by sorry
  original_size

#check original_group_size

end NUMINAMATH_CALUDE_original_group_size_l3016_301668


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_set_l3016_301657

theorem quadratic_inequality_solution_set :
  {x : ℝ | x^2 - 2*x - 3 < 0} = {x : ℝ | -1 < x ∧ x < 3} := by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_set_l3016_301657


namespace NUMINAMATH_CALUDE_sum_of_fractions_l3016_301628

theorem sum_of_fractions (A B : ℕ) (h : (A : ℚ) / 11 + (B : ℚ) / 3 = 17 / 33) : A + B = 3 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_fractions_l3016_301628


namespace NUMINAMATH_CALUDE_rectangles_in_grid_l3016_301637

/-- The number of different rectangles in a 3x5 grid -/
def num_rectangles : ℕ := 30

/-- The number of rows in the grid -/
def num_rows : ℕ := 3

/-- The number of columns in the grid -/
def num_cols : ℕ := 5

/-- Theorem stating that the number of rectangles in a 3x5 grid is 30 -/
theorem rectangles_in_grid :
  num_rectangles = (num_rows.choose 2) * (num_cols.choose 2) := by
  sorry

#eval num_rectangles -- This should output 30

end NUMINAMATH_CALUDE_rectangles_in_grid_l3016_301637


namespace NUMINAMATH_CALUDE_geometric_sequence_condition_l3016_301669

def is_geometric_sequence_with_ratio_2 (a : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, n ≥ 1 → a (n + 1) = 2 * a n

def satisfies_condition (a : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, n ≥ 2 → a n = 2 * a (n - 1)

theorem geometric_sequence_condition (a : ℕ → ℝ) :
  (is_geometric_sequence_with_ratio_2 a → satisfies_condition a) ∧
  ¬(satisfies_condition a → is_geometric_sequence_with_ratio_2 a) :=
sorry

end NUMINAMATH_CALUDE_geometric_sequence_condition_l3016_301669


namespace NUMINAMATH_CALUDE_cloud_ratio_l3016_301648

theorem cloud_ratio : 
  let carson_clouds : ℕ := 6
  let total_clouds : ℕ := 24
  let brother_clouds : ℕ := total_clouds - carson_clouds
  (brother_clouds : ℚ) / carson_clouds = 3 := by
  sorry

end NUMINAMATH_CALUDE_cloud_ratio_l3016_301648


namespace NUMINAMATH_CALUDE_all_2k_trips_use_both_modes_all_2k_plus_1_trips_use_both_modes_l3016_301652

/-- A type representing cities in a country. -/
structure City where
  id : Nat

/-- A type representing transportation modes. -/
inductive TransportMode
  | Bus
  | Flight

/-- A function that determines if two cities are directly connected by a given transport mode. -/
def directConnection (c1 c2 : City) (mode : TransportMode) : Prop :=
  sorry

/-- A proposition stating that any two cities are connected by either a direct flight or a direct bus route. -/
axiom connected_cities (c1 c2 : City) :
  directConnection c1 c2 TransportMode.Bus ∨ directConnection c1 c2 TransportMode.Flight

/-- A type representing a round trip as a list of cities. -/
def RoundTrip := List City

/-- A function that checks if a round trip uses both bus and flight. -/
def usesBothModes (trip : RoundTrip) : Prop :=
  sorry

/-- A theorem stating that all round trips touching 2k cities (k > 3) must use both bus and flight. -/
theorem all_2k_trips_use_both_modes (k : Nat) (h : k > 3) :
  ∀ (trip : RoundTrip), trip.length = 2 * k → usesBothModes trip :=
  sorry

/-- The main theorem to prove: if all round trips touching 2k cities (k > 3) must use both bus and flight,
    then all round trips touching 2k+1 cities must also use both bus and flight. -/
theorem all_2k_plus_1_trips_use_both_modes (k : Nat) (h : k > 3) :
  (∀ (trip : RoundTrip), trip.length = 2 * k → usesBothModes trip) →
  (∀ (trip : RoundTrip), trip.length = 2 * k + 1 → usesBothModes trip) :=
  sorry

end NUMINAMATH_CALUDE_all_2k_trips_use_both_modes_all_2k_plus_1_trips_use_both_modes_l3016_301652


namespace NUMINAMATH_CALUDE_c_finishes_in_60_days_l3016_301667

/-- The number of days it takes for worker c to finish the job alone, given:
  * Workers a and b together finish the job in 15 days
  * Workers a, b, and c together finish the job in 12 days
-/
def days_for_c_alone : ℚ :=
  let rate_ab : ℚ := 1 / 15  -- Combined rate of a and b
  let rate_abc : ℚ := 1 / 12 -- Combined rate of a, b, and c
  let rate_c : ℚ := rate_abc - rate_ab -- Rate of c alone
  1 / rate_c -- Days for c to finish the job

/-- Theorem stating that worker c alone can finish the job in 60 days -/
theorem c_finishes_in_60_days : days_for_c_alone = 60 := by
  sorry


end NUMINAMATH_CALUDE_c_finishes_in_60_days_l3016_301667


namespace NUMINAMATH_CALUDE_polynomial_expansion_l3016_301672

theorem polynomial_expansion (z : R) [CommRing R] :
  (3 * z^2 + 4 * z - 7) * (4 * z^3 - 3 * z + 2) =
  12 * z^5 + 16 * z^4 - 37 * z^3 - 6 * z^2 + 29 * z - 14 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_expansion_l3016_301672


namespace NUMINAMATH_CALUDE_sunset_time_calculation_l3016_301645

/-- Represents time in hours and minutes -/
structure Time where
  hours : Nat
  minutes : Nat

/-- Represents a duration in hours and minutes -/
structure Duration where
  hours : Nat
  minutes : Nat

/-- Adds a duration to a time -/
def addDuration (t : Time) (d : Duration) : Time :=
  let totalMinutes := t.minutes + t.hours * 60 + d.minutes + d.hours * 60
  { hours := totalMinutes / 60, minutes := totalMinutes % 60 }

/-- Converts 24-hour time to 12-hour time -/
def to12Hour (t : Time) : Time :=
  if t.hours ≥ 12 then
    { hours := if t.hours = 12 then 12 else t.hours - 12, minutes := t.minutes }
  else
    { hours := if t.hours = 0 then 12 else t.hours, minutes := t.minutes }

theorem sunset_time_calculation (sunrise : Time) (daylight : Duration) : 
  to12Hour (addDuration sunrise daylight) = { hours := 5, minutes := 31 } :=
  by sorry

end NUMINAMATH_CALUDE_sunset_time_calculation_l3016_301645


namespace NUMINAMATH_CALUDE_apartment_count_l3016_301693

theorem apartment_count
  (num_entrances : ℕ)
  (initial_number : ℕ)
  (new_number : ℕ)
  (h1 : num_entrances = 5)
  (h2 : initial_number = 636)
  (h3 : new_number = 242)
  (h4 : initial_number > new_number) :
  (initial_number - new_number) / 2 * num_entrances = 985 :=
by sorry

end NUMINAMATH_CALUDE_apartment_count_l3016_301693


namespace NUMINAMATH_CALUDE_min_sum_at_five_l3016_301654

/-- An arithmetic sequence -/
def arithmetic_sequence : ℕ → ℝ := sorry

/-- Sum of the first n terms of the arithmetic sequence -/
def S (n : ℕ) : ℝ := sorry

/-- The conditions given in the problem -/
axiom condition1 : S 10 = 0
axiom condition2 : S 15 = 25

/-- The theorem to prove -/
theorem min_sum_at_five :
  ∃ (n : ℕ), n = 5 ∧ ∀ (m : ℕ), S m ≥ S n :=
sorry

end NUMINAMATH_CALUDE_min_sum_at_five_l3016_301654


namespace NUMINAMATH_CALUDE_cary_calorie_deficit_l3016_301631

/-- Calculates the net calorie deficit for Cary's grocery store trip -/
theorem cary_calorie_deficit :
  let miles_walked : ℕ := 3
  let calories_per_mile : ℕ := 150
  let candy_bar_calories : ℕ := 200
  let total_calories_burned := miles_walked * calories_per_mile
  let net_deficit := total_calories_burned - candy_bar_calories
  net_deficit = 250 := by sorry

end NUMINAMATH_CALUDE_cary_calorie_deficit_l3016_301631


namespace NUMINAMATH_CALUDE_tan_sum_eq_two_l3016_301620

theorem tan_sum_eq_two (x y : Real) 
  (h1 : (Real.sin x / Real.cos y) + (Real.sin y / Real.cos x) = 2)
  (h2 : (Real.cos x / Real.sin y) + (Real.cos y / Real.sin x) = 4) :
  (Real.tan x / Real.tan y) + (Real.tan y / Real.tan x) = 2 := by
  sorry

end NUMINAMATH_CALUDE_tan_sum_eq_two_l3016_301620


namespace NUMINAMATH_CALUDE_actual_distance_traveled_l3016_301679

/-- Proves that the actual distance traveled is 60 km given the conditions of the problem -/
theorem actual_distance_traveled (speed_slow speed_fast distance_difference : ℝ) 
  (h1 : speed_slow = 15)
  (h2 : speed_fast = 30)
  (h3 : distance_difference = 60)
  (h4 : speed_slow > 0)
  (h5 : speed_fast > speed_slow) :
  ∃ (actual_distance : ℝ), 
    actual_distance / speed_slow = (actual_distance + distance_difference) / speed_fast ∧ 
    actual_distance = 60 := by
  sorry

end NUMINAMATH_CALUDE_actual_distance_traveled_l3016_301679


namespace NUMINAMATH_CALUDE_distance_between_points_l3016_301663

theorem distance_between_points : Real.sqrt ((7 - (-5))^2 + (3 - (-2))^2) = 13 := by
  sorry

end NUMINAMATH_CALUDE_distance_between_points_l3016_301663


namespace NUMINAMATH_CALUDE_product_xy_equals_one_l3016_301626

theorem product_xy_equals_one (x y : ℝ) (h_distinct : x ≠ y) 
    (h_eq : (1 / (1 + x^2)) + (1 / (1 + y^2)) = 2 / (1 + x*y)) : x * y = 1 := by
  sorry

end NUMINAMATH_CALUDE_product_xy_equals_one_l3016_301626


namespace NUMINAMATH_CALUDE_store_loss_percentage_l3016_301678

/-- Calculate the store's loss percentage on a radio sale -/
theorem store_loss_percentage
  (cost_price : ℝ)
  (discount_rate : ℝ)
  (tax_rate : ℝ)
  (actual_selling_price : ℝ)
  (h1 : cost_price = 25000)
  (h2 : discount_rate = 0.15)
  (h3 : tax_rate = 0.05)
  (h4 : actual_selling_price = 22000) :
  let discounted_price := cost_price * (1 - discount_rate)
  let final_selling_price := discounted_price * (1 + tax_rate)
  let loss := final_selling_price - actual_selling_price
  let loss_percentage := (loss / cost_price) * 100
  loss_percentage = 1.25 := by
sorry


end NUMINAMATH_CALUDE_store_loss_percentage_l3016_301678


namespace NUMINAMATH_CALUDE_expression_value_l3016_301625

theorem expression_value (x y : ℝ) (h : x + y = 3) : 2*x + 2*y - 1 = 5 := by
  sorry

end NUMINAMATH_CALUDE_expression_value_l3016_301625


namespace NUMINAMATH_CALUDE_quadratic_monotonic_condition_l3016_301660

-- Define the quadratic function
def f (a : ℝ) (x : ℝ) : ℝ := x^2 - a*x + 1

-- Define the property of monotonic interval starting at 1
def monotonic_from_one (a : ℝ) : Prop :=
  ∀ x y, 1 ≤ x → x < y → f a x < f a y

-- Theorem statement
theorem quadratic_monotonic_condition (a : ℝ) :
  monotonic_from_one a → a ≤ 2 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_monotonic_condition_l3016_301660


namespace NUMINAMATH_CALUDE_min_sum_squares_l3016_301638

def S : Finset Int := {-9, -4, -3, 0, 1, 5, 8, 10}

theorem min_sum_squares (p q r s t u v w : Int) 
  (h_distinct : p ≠ q ∧ p ≠ r ∧ p ≠ s ∧ p ≠ t ∧ p ≠ u ∧ p ≠ v ∧ p ≠ w ∧
                q ≠ r ∧ q ≠ s ∧ q ≠ t ∧ q ≠ u ∧ q ≠ v ∧ q ≠ w ∧
                r ≠ s ∧ r ≠ t ∧ r ≠ u ∧ r ≠ v ∧ r ≠ w ∧
                s ≠ t ∧ s ≠ u ∧ s ≠ v ∧ s ≠ w ∧
                t ≠ u ∧ t ≠ v ∧ t ≠ w ∧
                u ≠ v ∧ u ≠ w ∧
                v ≠ w)
  (h_in_S : p ∈ S ∧ q ∈ S ∧ r ∈ S ∧ s ∈ S ∧ t ∈ S ∧ u ∈ S ∧ v ∈ S ∧ w ∈ S) :
  32 ≤ (p + q + r + s)^2 + (t + u + v + w)^2 :=
by
  sorry

#check min_sum_squares

end NUMINAMATH_CALUDE_min_sum_squares_l3016_301638


namespace NUMINAMATH_CALUDE_perpendicular_parallel_transitive_l3016_301624

-- Define the types for planes and lines
variable (Plane Line : Type*)

-- Define the relations
variable (perpendicular : Line → Plane → Prop)
variable (parallel : Plane → Plane → Prop)

-- State the theorem
theorem perpendicular_parallel_transitive
  (α β : Plane) (l : Line)
  (h1 : perpendicular l α)
  (h2 : parallel α β) :
  perpendicular l β :=
sorry

end NUMINAMATH_CALUDE_perpendicular_parallel_transitive_l3016_301624


namespace NUMINAMATH_CALUDE_shaded_fraction_of_rectangle_l3016_301623

theorem shaded_fraction_of_rectangle : 
  let rectangle_length : ℝ := 10
  let rectangle_width : ℝ := 20
  let total_area : ℝ := rectangle_length * rectangle_width
  let quarter_area : ℝ := total_area / 4
  let shaded_area : ℝ := quarter_area / 2
  shaded_area / total_area = 1 / 8 := by sorry

end NUMINAMATH_CALUDE_shaded_fraction_of_rectangle_l3016_301623


namespace NUMINAMATH_CALUDE_complement_intersection_theorem_l3016_301666

def U : Finset Int := {-1, 0, 1, 2, 3}
def A : Finset Int := {-1, 0, 2}
def B : Finset Int := {0, 1}

theorem complement_intersection_theorem :
  (U \ A) ∩ (U \ B) = {3} := by sorry

end NUMINAMATH_CALUDE_complement_intersection_theorem_l3016_301666


namespace NUMINAMATH_CALUDE_prob_three_faces_is_8_27_l3016_301680

/-- Represents a small cube sawed from a larger painted cube -/
structure SmallCube :=
  (painted_faces : Fin 4)

/-- The set of all small cubes obtained from sawing a painted cube -/
def all_cubes : Finset SmallCube := sorry

/-- The set of small cubes with exactly three painted faces -/
def three_face_cubes : Finset SmallCube := sorry

/-- The probability of selecting a small cube with three painted faces -/
def prob_three_faces : ℚ := (three_face_cubes.card : ℚ) / (all_cubes.card : ℚ)

theorem prob_three_faces_is_8_27 : prob_three_faces = 8 / 27 := by sorry

end NUMINAMATH_CALUDE_prob_three_faces_is_8_27_l3016_301680


namespace NUMINAMATH_CALUDE_principal_amount_l3016_301607

/-- Proves that given the conditions of the problem, the principal amount is 3000 --/
theorem principal_amount (P : ℝ) : 
  (P * 4 * 5) / 100 = P - 2400 → P = 3000 := by
  sorry

end NUMINAMATH_CALUDE_principal_amount_l3016_301607


namespace NUMINAMATH_CALUDE_fraction_simplification_l3016_301651

theorem fraction_simplification :
  (3 / 7 + 4 / 5) / (5 / 12 + 2 / 3) = 516 / 455 := by
  sorry

end NUMINAMATH_CALUDE_fraction_simplification_l3016_301651


namespace NUMINAMATH_CALUDE_infinite_composite_values_l3016_301608

theorem infinite_composite_values (m n k : ℕ) :
  (∃ f : ℕ → ℕ, ∀ k ≥ 2, f k = 4 * k^4) ∧
  (∀ k ≥ 2, ∀ m, ∃ a b : ℕ, a > 1 ∧ b > 1 ∧ m^4 + 4 * k^4 = a * b) :=
sorry

end NUMINAMATH_CALUDE_infinite_composite_values_l3016_301608


namespace NUMINAMATH_CALUDE_gus_egg_consumption_l3016_301697

/-- The number of eggs Gus ate for breakfast -/
def breakfast_eggs : ℕ := 2

/-- The number of eggs Gus ate for lunch -/
def lunch_eggs : ℕ := 3

/-- The number of eggs Gus ate for dinner -/
def dinner_eggs : ℕ := 1

/-- The total number of eggs Gus ate throughout the day -/
def total_eggs : ℕ := breakfast_eggs + lunch_eggs + dinner_eggs

theorem gus_egg_consumption : total_eggs = 6 := by
  sorry

end NUMINAMATH_CALUDE_gus_egg_consumption_l3016_301697


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_set_l3016_301642

theorem quadratic_inequality_solution_set :
  {x : ℝ | 2 * x^2 - x - 1 > 0} = {x : ℝ | x < -1/2 ∨ x > 1} :=
by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_set_l3016_301642


namespace NUMINAMATH_CALUDE_system1_solution_system2_solution_l3016_301615

-- System 1
theorem system1_solution :
  ∃ (x y : ℝ), x - 2*y = 1 ∧ 4*x + 3*y = 26 ∧ x = 5 ∧ y = 2 := by
  sorry

-- System 2
theorem system2_solution :
  ∃ (x y : ℝ), 2*x + 3*y = 3 ∧ 5*x - 3*y = 18 ∧ x = 3 ∧ y = -1 := by
  sorry

end NUMINAMATH_CALUDE_system1_solution_system2_solution_l3016_301615


namespace NUMINAMATH_CALUDE_geometric_sequence_first_term_l3016_301670

def is_geometric_sequence (a : ℕ → ℚ) : Prop :=
  ∃ r : ℚ, ∀ n : ℕ, a (n + 1) = a n * r

theorem geometric_sequence_first_term
  (a : ℕ → ℚ)
  (h_geometric : is_geometric_sequence a)
  (h_a2 : a 2 = 2)
  (h_a5 : a 5 = -54) :
  a 1 = -2/3 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_first_term_l3016_301670


namespace NUMINAMATH_CALUDE_marbles_given_to_juan_l3016_301647

theorem marbles_given_to_juan (initial_marbles : ℕ) (remaining_marbles : ℕ) 
  (h1 : initial_marbles = 73)
  (h2 : remaining_marbles = 3) :
  initial_marbles - remaining_marbles = 70 := by
  sorry

end NUMINAMATH_CALUDE_marbles_given_to_juan_l3016_301647


namespace NUMINAMATH_CALUDE_score_difference_l3016_301689

theorem score_difference (hajar_score : ℕ) (total_score : ℕ) : 
  hajar_score = 24 →
  total_score = 69 →
  ∃ (farah_score : ℕ),
    farah_score > hajar_score ∧
    farah_score + hajar_score = total_score ∧
    farah_score - hajar_score = 21 :=
by
  sorry

end NUMINAMATH_CALUDE_score_difference_l3016_301689


namespace NUMINAMATH_CALUDE_square_difference_305_301_l3016_301659

theorem square_difference_305_301 : 305^2 - 301^2 = 2424 := by sorry

end NUMINAMATH_CALUDE_square_difference_305_301_l3016_301659


namespace NUMINAMATH_CALUDE_special_house_profit_calculation_l3016_301613

def special_house_profit (extra_cost : ℝ) (price_multiplier : ℝ) (standard_house_price : ℝ) : ℝ :=
  price_multiplier * standard_house_price - standard_house_price - extra_cost

theorem special_house_profit_calculation :
  special_house_profit 100000 1.5 320000 = 60000 := by
  sorry

end NUMINAMATH_CALUDE_special_house_profit_calculation_l3016_301613


namespace NUMINAMATH_CALUDE_convex_hull_perimeter_bounds_l3016_301692

/-- A regular polygon inscribed in a unit circle -/
structure RegularPolygon where
  n : ℕ
  n_ge_3 : n ≥ 3

/-- The convex hull formed by the vertices of two regular polygons inscribed in a unit circle -/
structure ConvexHull where
  p1 : RegularPolygon
  p2 : RegularPolygon

/-- The perimeter of the convex hull -/
noncomputable def perimeter (ch : ConvexHull) : ℝ :=
  sorry

theorem convex_hull_perimeter_bounds (ch : ConvexHull) 
  (h1 : ch.p1.n = 6) 
  (h2 : ch.p2.n = 7) : 
  6.1610929 ≤ perimeter ch ∧ perimeter ch ≤ 6.1647971 :=
sorry

end NUMINAMATH_CALUDE_convex_hull_perimeter_bounds_l3016_301692


namespace NUMINAMATH_CALUDE_largest_product_of_three_l3016_301694

def S : Finset Int := {-5, -4, -1, 6, 7}

theorem largest_product_of_three (a b c : Int) 
  (ha : a ∈ S) (hb : b ∈ S) (hc : c ∈ S) 
  (hab : a ≠ b) (hbc : b ≠ c) (hac : a ≠ c) :
  (∀ x y z : Int, x ∈ S → y ∈ S → z ∈ S → 
   x ≠ y → y ≠ z → x ≠ z → 
   x * y * z ≤ a * b * c) →
  a * b * c = 140 :=
sorry

end NUMINAMATH_CALUDE_largest_product_of_three_l3016_301694


namespace NUMINAMATH_CALUDE_difference_greater_than_one_l3016_301609

theorem difference_greater_than_one : 19^91 - (999991:ℕ)^19 ≥ 1 := by
  sorry

end NUMINAMATH_CALUDE_difference_greater_than_one_l3016_301609


namespace NUMINAMATH_CALUDE_consecutive_integers_sqrt_3_l3016_301641

theorem consecutive_integers_sqrt_3 (a b : ℤ) : 
  (b = a + 1) →  -- a and b are consecutive integers
  (↑a < Real.sqrt 3) →  -- a < sqrt(3)
  (Real.sqrt 3 < ↑b) →  -- sqrt(3) < b
  a + b = 3 := by sorry

end NUMINAMATH_CALUDE_consecutive_integers_sqrt_3_l3016_301641


namespace NUMINAMATH_CALUDE_diagonal_intersection_probability_l3016_301611

/-- The probability that two randomly chosen diagonals intersect in a convex polygon with 2n+1 vertices -/
theorem diagonal_intersection_probability (n : ℕ) (h : n > 0) :
  let vertices := 2 * n + 1
  let total_diagonals := vertices.choose 2 - vertices
  let intersecting_pairs := vertices.choose 4
  let probability := intersecting_pairs / total_diagonals.choose 2
  probability = n * (2 * n - 1) / (3 * (2 * n^2 - n - 2)) :=
by sorry

end NUMINAMATH_CALUDE_diagonal_intersection_probability_l3016_301611


namespace NUMINAMATH_CALUDE_disjunction_and_negation_implication_l3016_301633

theorem disjunction_and_negation_implication (p q : Prop) :
  (p ∨ q) → ¬p → (¬p ∧ q) := by
  sorry

end NUMINAMATH_CALUDE_disjunction_and_negation_implication_l3016_301633


namespace NUMINAMATH_CALUDE_distinct_prime_factors_of_90_l3016_301605

theorem distinct_prime_factors_of_90 : Finset.card (Nat.factors 90).toFinset = 3 := by
  sorry

end NUMINAMATH_CALUDE_distinct_prime_factors_of_90_l3016_301605


namespace NUMINAMATH_CALUDE_garden_separation_possible_l3016_301618

/-- Represents the content of a garden plot -/
inductive PlotContent
  | Empty
  | Cabbage
  | Goat

/-- Represents a position in the garden -/
structure Position where
  x : Fin 4
  y : Fin 4

/-- Represents a fence in the garden -/
inductive Fence
  | Vertical (x : Fin 3) -- A vertical fence after column x
  | Horizontal (y : Fin 3) -- A horizontal fence after row y

/-- Represents the garden layout -/
def Garden := Position → PlotContent

/-- Checks if a fence separates two positions -/
def separates (f : Fence) (p1 p2 : Position) : Prop :=
  match f with
  | Fence.Vertical x => p1.x ≤ x ∧ x < p2.x
  | Fence.Horizontal y => p1.y ≤ y ∧ y < p2.y

/-- The theorem to be proved -/
theorem garden_separation_possible (g : Garden) :
  ∃ (f1 f2 f3 : Fence),
    (∀ p1 p2 : Position,
      g p1 = PlotContent.Goat →
      g p2 = PlotContent.Cabbage →
      separates f1 p1 p2 ∨ separates f2 p1 p2 ∨ separates f3 p1 p2) ∧
    (∀ f : Fence, f ∈ [f1, f2, f3] →
      ∀ p : Position,
        g p ≠ PlotContent.Empty →
        ¬(∃ p' : Position, g p' ≠ PlotContent.Empty ∧ separates f p p')) :=
by sorry

end NUMINAMATH_CALUDE_garden_separation_possible_l3016_301618


namespace NUMINAMATH_CALUDE_points_on_opposite_sides_l3016_301671

def plane_equation (x y z : ℝ) : ℝ := x + 2*y + 3*z

def point1 : ℝ × ℝ × ℝ := (1, 2, -2)
def point2 : ℝ × ℝ × ℝ := (2, 1, -1)

theorem points_on_opposite_sides :
  (plane_equation point1.1 point1.2.1 point1.2.2) * (plane_equation point2.1 point2.2.1 point2.2.2) < 0 := by
  sorry

end NUMINAMATH_CALUDE_points_on_opposite_sides_l3016_301671


namespace NUMINAMATH_CALUDE_tuna_distribution_l3016_301621

theorem tuna_distribution (total_customers : ℕ) (tuna_count : ℕ) (tuna_weight : ℕ) (customers_without_fish : ℕ) :
  total_customers = 100 →
  tuna_count = 10 →
  tuna_weight = 200 →
  customers_without_fish = 20 →
  (tuna_count * tuna_weight) / (total_customers - customers_without_fish) = 25 := by
  sorry

end NUMINAMATH_CALUDE_tuna_distribution_l3016_301621


namespace NUMINAMATH_CALUDE_vector_parallel_sum_l3016_301699

/-- Given vectors a and b, if a is parallel to (a + b), then the second component of b is 3. -/
theorem vector_parallel_sum (a b : ℝ × ℝ) (h : ∃ (k : ℝ), a = k • (a + b)) :
  a.1 = 1 ∧ a.2 = 1 ∧ b.1 = 3 → b.2 = 3 := by
  sorry

end NUMINAMATH_CALUDE_vector_parallel_sum_l3016_301699


namespace NUMINAMATH_CALUDE_two_x_value_l3016_301601

theorem two_x_value (x : ℚ) (h : 4 * x + 14 = 8 * x - 48) : 2 * x = 31 := by
  sorry

end NUMINAMATH_CALUDE_two_x_value_l3016_301601


namespace NUMINAMATH_CALUDE_no_real_solution_l3016_301640

theorem no_real_solution :
  ¬∃ (x y z : ℝ), x^2 + 4*y*z + 2*z = 0 ∧ x + 2*x*y + 2*z^2 = 0 ∧ 2*x*z + y^2 + y + 1 = 0 := by
  sorry

end NUMINAMATH_CALUDE_no_real_solution_l3016_301640


namespace NUMINAMATH_CALUDE_residue_mod_17_l3016_301614

theorem residue_mod_17 : (207 * 13 - 22 * 8 + 5) % 17 = 3 := by
  sorry

end NUMINAMATH_CALUDE_residue_mod_17_l3016_301614


namespace NUMINAMATH_CALUDE_box_volume_increase_l3016_301619

/-- Given a rectangular box with length l, width w, and height h satisfying certain conditions,
    prove that increasing each dimension by 2 results in a volume of 7208 cubic inches. -/
theorem box_volume_increase (l w h : ℝ) 
  (volume : l * w * h = 5400)
  (surface_area : 2 * (l * w + w * h + h * l) = 1560)
  (edge_sum : 4 * (l + w + h) = 240) :
  (l + 2) * (w + 2) * (h + 2) = 7208 := by
  sorry

end NUMINAMATH_CALUDE_box_volume_increase_l3016_301619


namespace NUMINAMATH_CALUDE_range_of_absolute_value_sum_l3016_301627

theorem range_of_absolute_value_sum (a b : ℝ) (ha : a > 0) (hb : b < 0) :
  ∀ x : ℝ, |x - a| + |x - b| = a - b ↔ b ≤ x ∧ x ≤ a :=
by sorry

end NUMINAMATH_CALUDE_range_of_absolute_value_sum_l3016_301627


namespace NUMINAMATH_CALUDE_multiply_82519_9999_l3016_301635

theorem multiply_82519_9999 : 82519 * 9999 = 825107481 := by
  sorry

end NUMINAMATH_CALUDE_multiply_82519_9999_l3016_301635


namespace NUMINAMATH_CALUDE_initial_amount_calculation_l3016_301681

/-- Represents the simple interest scenario --/
structure SimpleInterest where
  initialAmount : ℝ
  rate : ℝ
  time : ℝ
  finalAmount : ℝ

/-- The simple interest calculation is correct --/
def isValidSimpleInterest (si : SimpleInterest) : Prop :=
  si.finalAmount = si.initialAmount * (1 + si.rate * si.time / 100)

/-- Theorem stating the initial amount given the conditions --/
theorem initial_amount_calculation (si : SimpleInterest) 
  (h1 : si.finalAmount = 1050)
  (h2 : si.rate = 8)
  (h3 : si.time = 5)
  (h4 : isValidSimpleInterest si) : 
  si.initialAmount = 750 := by
  sorry

#check initial_amount_calculation

end NUMINAMATH_CALUDE_initial_amount_calculation_l3016_301681


namespace NUMINAMATH_CALUDE_arccos_sum_solution_l3016_301662

theorem arccos_sum_solution (x : ℝ) : 
  Real.arccos (2 * x) + Real.arccos (3 * x) = π / 2 → x = 1 / Real.sqrt 13 ∨ x = -1 / Real.sqrt 13 :=
by sorry

end NUMINAMATH_CALUDE_arccos_sum_solution_l3016_301662


namespace NUMINAMATH_CALUDE_quadratic_equations_solutions_l3016_301677

theorem quadratic_equations_solutions :
  (∃ x1 x2 : ℝ, x1^2 + 4*x1 - 2 = 0 ∧ x2^2 + 4*x2 - 2 = 0 ∧ x1 = -2 + Real.sqrt 6 ∧ x2 = -2 - Real.sqrt 6) ∧
  (∃ y1 y2 : ℝ, 2*y1^2 - 3*y1 + 1 = 0 ∧ 2*y2^2 - 3*y2 + 1 = 0 ∧ y1 = 1/2 ∧ y2 = 1) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equations_solutions_l3016_301677


namespace NUMINAMATH_CALUDE_sum_of_seventh_powers_l3016_301695

/-- Given a sequence of sums of powers of a and b, prove that a^7 + b^7 = 29 -/
theorem sum_of_seventh_powers (a b : ℝ) 
  (h1 : a + b = 1)
  (h2 : a^2 + b^2 = 3)
  (h3 : a^3 + b^3 = 4)
  (h4 : a^4 + b^4 = 7)
  (h5 : a^5 + b^5 = 11) :
  a^7 + b^7 = 29 := by sorry

end NUMINAMATH_CALUDE_sum_of_seventh_powers_l3016_301695


namespace NUMINAMATH_CALUDE_total_distinct_students_l3016_301617

/-- Represents the number of distinct students in the mathematics competition --/
def distinct_students (germain newton young germain_newton_overlap germain_young_overlap : ℕ) : ℕ :=
  germain + newton + young - germain_newton_overlap - germain_young_overlap

/-- Theorem stating that the total number of distinct students is 32 --/
theorem total_distinct_students :
  distinct_students 13 10 12 2 1 = 32 := by
  sorry

#eval distinct_students 13 10 12 2 1

end NUMINAMATH_CALUDE_total_distinct_students_l3016_301617


namespace NUMINAMATH_CALUDE_no_special_subset_exists_l3016_301682

theorem no_special_subset_exists : ¬∃ M : Set ℕ,
  (∀ n : ℕ, n > 1 → ∃ a b : ℕ, a ∈ M ∧ b ∈ M ∧ n = a + b) ∧
  (∀ a b c d : ℕ, a ∈ M → b ∈ M → c ∈ M → d ∈ M →
    a > 10 → b > 10 → c > 10 → d > 10 →
    ((a + b = c + d) ↔ (a = c ∨ a = d))) :=
by sorry

end NUMINAMATH_CALUDE_no_special_subset_exists_l3016_301682


namespace NUMINAMATH_CALUDE_class_size_proof_l3016_301676

theorem class_size_proof (n : ℕ) : 
  (n / 6 : ℚ) = (n / 18 : ℚ) + 4 →  -- One-sixth wear glasses, split into girls and boys
  n = 36 :=
by
  sorry

#check class_size_proof

end NUMINAMATH_CALUDE_class_size_proof_l3016_301676


namespace NUMINAMATH_CALUDE_average_age_combined_l3016_301629

theorem average_age_combined (n_students : ℕ) (n_parents : ℕ) 
  (avg_age_students : ℚ) (avg_age_parents : ℚ) :
  n_students = 33 →
  n_parents = 55 →
  avg_age_students = 11 →
  avg_age_parents = 33 →
  (n_students * avg_age_students + n_parents * avg_age_parents) / (n_students + n_parents : ℚ) = 24.75 := by
  sorry

end NUMINAMATH_CALUDE_average_age_combined_l3016_301629


namespace NUMINAMATH_CALUDE_chris_previous_savings_l3016_301698

/-- Represents the amount of money Chris received as birthday gifts in different currencies -/
structure BirthdayGifts where
  usd : ℝ
  eur : ℝ
  cad : ℝ
  gbp : ℝ

/-- Represents the conversion rates from different currencies to USD -/
structure ConversionRates where
  eur_to_usd : ℝ
  cad_to_usd : ℝ
  gbp_to_usd : ℝ

/-- Calculates Chris's savings before his birthday -/
def calculate_previous_savings (gifts : BirthdayGifts) (rates : ConversionRates) (total_after : ℝ) : ℝ :=
  total_after - (gifts.usd + 
                 gifts.eur * rates.eur_to_usd + 
                 gifts.cad * rates.cad_to_usd + 
                 gifts.gbp * rates.gbp_to_usd)

/-- Theorem stating that Chris's savings before his birthday were 128.80 USD -/
theorem chris_previous_savings 
  (gifts : BirthdayGifts) 
  (rates : ConversionRates) 
  (total_after : ℝ) : 
  gifts.usd = 25 ∧ 
  gifts.eur = 20 ∧ 
  gifts.cad = 75 ∧ 
  gifts.gbp = 30 ∧
  rates.eur_to_usd = 1 / 0.85 ∧
  rates.cad_to_usd = 1 / 1.25 ∧
  rates.gbp_to_usd = 1 / 0.72 ∧
  total_after = 279 →
  calculate_previous_savings gifts rates total_after = 128.80 := by
    sorry

end NUMINAMATH_CALUDE_chris_previous_savings_l3016_301698


namespace NUMINAMATH_CALUDE_optimal_investment_plan_l3016_301696

/-- Represents an investment project --/
structure Project where
  maxProfitRate : ℝ
  maxLossRate : ℝ

/-- Represents an investment plan --/
structure InvestmentPlan where
  projectA : ℝ
  projectB : ℝ

def totalInvestment (plan : InvestmentPlan) : ℝ :=
  plan.projectA + plan.projectB

def potentialProfit (plan : InvestmentPlan) (projectA projectB : Project) : ℝ :=
  plan.projectA * projectA.maxProfitRate + plan.projectB * projectB.maxProfitRate

def potentialLoss (plan : InvestmentPlan) (projectA projectB : Project) : ℝ :=
  plan.projectA * projectA.maxLossRate + plan.projectB * projectB.maxLossRate

theorem optimal_investment_plan 
  (projectA : Project)
  (projectB : Project)
  (h_profitA : projectA.maxProfitRate = 1)
  (h_profitB : projectB.maxProfitRate = 0.5)
  (h_lossA : projectA.maxLossRate = 0.3)
  (h_lossB : projectB.maxLossRate = 0.1)
  (optimalPlan : InvestmentPlan)
  (h_optimalA : optimalPlan.projectA = 40000)
  (h_optimalB : optimalPlan.projectB = 60000) :
  (∀ plan : InvestmentPlan, 
    totalInvestment plan ≤ 100000 ∧ 
    potentialLoss plan projectA projectB ≤ 18000 →
    potentialProfit plan projectA projectB ≤ potentialProfit optimalPlan projectA projectB) ∧
  totalInvestment optimalPlan ≤ 100000 ∧
  potentialLoss optimalPlan projectA projectB ≤ 18000 :=
sorry

end NUMINAMATH_CALUDE_optimal_investment_plan_l3016_301696


namespace NUMINAMATH_CALUDE_decimal_addition_subtraction_l3016_301665

theorem decimal_addition_subtraction :
  (0.513 : ℝ) + (0.0067 : ℝ) - (0.048 : ℝ) = (0.4717 : ℝ) := by
  sorry

end NUMINAMATH_CALUDE_decimal_addition_subtraction_l3016_301665
