import Mathlib

namespace NUMINAMATH_CALUDE_fraction_simplification_l1212_121209

/-- The number of quarters Sarah has -/
def total_quarters : ℕ := 30

/-- The number of states that joined the union from 1790 to 1799 -/
def states_1790_1799 : ℕ := 8

/-- The fraction of Sarah's quarters representing states that joined from 1790 to 1799 -/
def fraction_1790_1799 : ℚ := states_1790_1799 / total_quarters

theorem fraction_simplification :
  fraction_1790_1799 = 4 / 15 := by sorry

end NUMINAMATH_CALUDE_fraction_simplification_l1212_121209


namespace NUMINAMATH_CALUDE_parallel_lines_m_value_l1212_121266

/-- Given two lines AB and CD, where:
    - AB passes through points A(-2,m) and B(m,4)
    - CD passes through points C(m+1,1) and D(m,3)
    - AB is parallel to CD
    Prove that m = -8 -/
theorem parallel_lines_m_value :
  ∀ m : ℝ,
  let A : ℝ × ℝ := (-2, m)
  let B : ℝ × ℝ := (m, 4)
  let C : ℝ × ℝ := (m + 1, 1)
  let D : ℝ × ℝ := (m, 3)
  let slope_AB := (B.2 - A.2) / (B.1 - A.1)
  let slope_CD := (D.2 - C.2) / (D.1 - C.1)
  slope_AB = slope_CD →
  m = -8 := by
sorry

end NUMINAMATH_CALUDE_parallel_lines_m_value_l1212_121266


namespace NUMINAMATH_CALUDE_math_competition_correct_answers_l1212_121236

theorem math_competition_correct_answers 
  (total_questions : Nat) 
  (correct_score : Nat) 
  (incorrect_penalty : Nat) 
  (xiao_ming_score : Nat) 
  (xiao_hong_score : Nat) 
  (xiao_hua_score : Nat) :
  total_questions = 10 →
  correct_score = 10 →
  incorrect_penalty = 3 →
  xiao_ming_score = 87 →
  xiao_hong_score = 74 →
  xiao_hua_score = 9 →
  (total_questions - (total_questions * correct_score - xiao_ming_score) / (correct_score + incorrect_penalty)) +
  (total_questions - (total_questions * correct_score - xiao_hong_score) / (correct_score + incorrect_penalty)) +
  (total_questions - (total_questions * correct_score - xiao_hua_score) / (correct_score + incorrect_penalty)) = 20 :=
by sorry

end NUMINAMATH_CALUDE_math_competition_correct_answers_l1212_121236


namespace NUMINAMATH_CALUDE_betty_height_in_feet_l1212_121279

/-- Given the heights of Carter, his dog, and Betty, prove Betty's height in feet. -/
theorem betty_height_in_feet :
  ∀ (carter_height dog_height betty_height : ℕ),
    carter_height = 2 * dog_height →
    dog_height = 24 →
    betty_height = carter_height - 12 →
    betty_height / 12 = 3 := by
  sorry

end NUMINAMATH_CALUDE_betty_height_in_feet_l1212_121279


namespace NUMINAMATH_CALUDE_tax_threshold_value_l1212_121218

def tax_calculation (X : ℝ) (I : ℝ) : ℝ := 0.12 * X + 0.20 * (I - X)

theorem tax_threshold_value :
  ∃ (X : ℝ), 
    X = 40000 ∧
    tax_calculation X 56000 = 8000 := by
  sorry

end NUMINAMATH_CALUDE_tax_threshold_value_l1212_121218


namespace NUMINAMATH_CALUDE_min_n_for_60n_divisible_by_4_and_8_l1212_121212

theorem min_n_for_60n_divisible_by_4_and_8 :
  ∃ (n : ℕ), n > 0 ∧ 4 ∣ 60 * n ∧ 8 ∣ 60 * n ∧
  ∀ (m : ℕ), m > 0 → 4 ∣ 60 * m → 8 ∣ 60 * m → n ≤ m :=
by sorry

end NUMINAMATH_CALUDE_min_n_for_60n_divisible_by_4_and_8_l1212_121212


namespace NUMINAMATH_CALUDE_cube_volume_in_pyramid_l1212_121293

/-- A regular pyramid with a rectangular base and isosceles triangular lateral faces -/
structure RegularPyramid where
  base_length : ℝ
  base_width : ℝ
  lateral_faces_isosceles : Bool

/-- A cube placed inside the pyramid -/
structure InsideCube where
  side_length : ℝ

/-- The theorem stating the volume of the cube inside the pyramid -/
theorem cube_volume_in_pyramid (pyramid : RegularPyramid) (cube : InsideCube) : 
  pyramid.base_length = 2 →
  pyramid.base_width = 3 →
  pyramid.lateral_faces_isosceles = true →
  (cube.side_length * Real.sqrt 3 = Real.sqrt 13) →
  cube.side_length^3 = (39 * Real.sqrt 39) / 27 := by
  sorry

end NUMINAMATH_CALUDE_cube_volume_in_pyramid_l1212_121293


namespace NUMINAMATH_CALUDE_jonathan_saved_eight_l1212_121288

/-- Calculates the amount of money saved given the costs of three books and the additional amount needed. -/
def money_saved (book1_cost book2_cost book3_cost additional_needed : ℕ) : ℕ :=
  (book1_cost + book2_cost + book3_cost) - additional_needed

/-- Proves that given the specific costs and additional amount needed, the money saved is 8. -/
theorem jonathan_saved_eight :
  money_saved 11 19 7 29 = 8 := by
  sorry

end NUMINAMATH_CALUDE_jonathan_saved_eight_l1212_121288


namespace NUMINAMATH_CALUDE_solve_for_x_l1212_121208

theorem solve_for_x : ∃ x : ℤ, x + 1315 + 9211 - 1569 = 11901 ∧ x = 2944 := by
  sorry

end NUMINAMATH_CALUDE_solve_for_x_l1212_121208


namespace NUMINAMATH_CALUDE_line_length_difference_l1212_121264

theorem line_length_difference (white_line blue_line : ℝ) 
  (h1 : white_line = 7.67) 
  (h2 : blue_line = 3.33) : 
  white_line - blue_line = 4.34 := by
sorry

end NUMINAMATH_CALUDE_line_length_difference_l1212_121264


namespace NUMINAMATH_CALUDE_autograph_distribution_theorem_l1212_121247

/-- Represents a set of autographs from 11 players -/
def Autographs := Fin 11 → Bool

/-- The set of all residents -/
def Residents := Fin 1111

/-- Distribution of autographs to residents -/
def AutographDistribution := Residents → Autographs

theorem autograph_distribution_theorem (d : AutographDistribution) 
  (h : ∀ (i j : Residents), i ≠ j → d i ≠ d j) :
  ∃ (i j : Residents), i ≠ j ∧ 
    (∀ (k : Fin 11), (d i k = true ∧ d j k = false) ∨ (d i k = false ∧ d j k = true)) :=
sorry

end NUMINAMATH_CALUDE_autograph_distribution_theorem_l1212_121247


namespace NUMINAMATH_CALUDE_bc_is_one_sixth_of_ad_l1212_121281

/-- Given a line segment AD with points E and B on it, prove that BC is 1/6 of AD -/
theorem bc_is_one_sixth_of_ad (A B C D E : ℝ) : 
  A < E ∧ E < D ∧   -- E is on AD
  A < B ∧ B < D ∧   -- B is on AD
  E - A = 3 * (D - E) ∧   -- AE is 3 times ED
  B - A = 5 * (D - B) ∧   -- AB is 5 times BD
  C = (B + E) / 2   -- C is midpoint of BE
  → 
  (C - B) / (D - A) = 1 / 6 :=
by sorry

end NUMINAMATH_CALUDE_bc_is_one_sixth_of_ad_l1212_121281


namespace NUMINAMATH_CALUDE_negative_one_third_squared_l1212_121272

theorem negative_one_third_squared : (-1/3 : ℚ)^2 = 1/9 := by
  sorry

end NUMINAMATH_CALUDE_negative_one_third_squared_l1212_121272


namespace NUMINAMATH_CALUDE_cos_equality_proof_l1212_121206

theorem cos_equality_proof (n : ℤ) : 
  0 ≤ n ∧ n ≤ 360 → (Real.cos (n * π / 180) = Real.cos (123 * π / 180) ↔ n = 123 ∨ n = 237) := by
  sorry

end NUMINAMATH_CALUDE_cos_equality_proof_l1212_121206


namespace NUMINAMATH_CALUDE_product_plus_number_equals_result_l1212_121278

theorem product_plus_number_equals_result : ∃ x : ℝ,
  12.05 * 5.4 + x = 108.45000000000003 ∧ x = 43.38000000000003 := by
  sorry

end NUMINAMATH_CALUDE_product_plus_number_equals_result_l1212_121278


namespace NUMINAMATH_CALUDE_cylinder_min_surface_area_l1212_121275

/-- For a cylindrical tank with volume V, the surface area (without a lid) is minimized when the radius and height are both equal to ∛(V/π) -/
theorem cylinder_min_surface_area (V : ℝ) (h : V > 0) :
  let surface_area (r h : ℝ) := π * r^2 + 2 * π * r * h
  let volume (r h : ℝ) := π * r^2 * h
  ∃ (r : ℝ), r > 0 ∧ 
    (∀ (r' h' : ℝ), r' > 0 → h' > 0 → volume r' h' = V → 
      surface_area r' h' ≥ surface_area r r) ∧
    r = (V / π)^(1/3) :=
by sorry

end NUMINAMATH_CALUDE_cylinder_min_surface_area_l1212_121275


namespace NUMINAMATH_CALUDE_inequality_theorem_l1212_121280

theorem inequality_theorem (a b : ℝ) (h : a > b) :
  ∀ x : ℝ, (a / (2^x + 1)) > (b / (2^x + 1)) := by
sorry

end NUMINAMATH_CALUDE_inequality_theorem_l1212_121280


namespace NUMINAMATH_CALUDE_marys_maximum_earnings_l1212_121295

/-- Mary's maximum weekly earnings problem -/
theorem marys_maximum_earnings :
  let max_hours : ℕ := 60
  let regular_rate : ℚ := 12
  let regular_hours : ℕ := 30
  let overtime_rate : ℚ := regular_rate * (3/2)
  let overtime_hours : ℕ := max_hours - regular_hours
  let regular_earnings : ℚ := regular_rate * regular_hours
  let overtime_earnings : ℚ := overtime_rate * overtime_hours
  regular_earnings + overtime_earnings = 900 := by
  sorry

end NUMINAMATH_CALUDE_marys_maximum_earnings_l1212_121295


namespace NUMINAMATH_CALUDE_subcommittee_formation_ways_l1212_121269

def number_of_ways_to_form_subcommittee (total_republicans : ℕ) (total_democrats : ℕ) 
  (subcommittee_republicans : ℕ) (subcommittee_democrats : ℕ) : ℕ :=
  Nat.choose total_republicans subcommittee_republicans * 
  Nat.choose total_democrats subcommittee_democrats

theorem subcommittee_formation_ways : 
  number_of_ways_to_form_subcommittee 10 8 4 3 = 11760 := by
  sorry

end NUMINAMATH_CALUDE_subcommittee_formation_ways_l1212_121269


namespace NUMINAMATH_CALUDE_f_properties_l1212_121257

def f_property (f : ℝ → ℝ) : Prop :=
  (∃ x, f x ≠ 0) ∧
  (∀ x y, f (x * y) = x * f y + y * f x) ∧
  (∀ x, x > 1 → f x < 0)

theorem f_properties (f : ℝ → ℝ) (h : f_property f) :
  (f 1 = 0 ∧ f (-1) = 0) ∧
  (∀ x, f (-x) = -f x) ∧
  (∀ x₁ x₂, x₁ > x₂ ∧ x₂ > 1 → f x₁ < f x₂) := by
  sorry

end NUMINAMATH_CALUDE_f_properties_l1212_121257


namespace NUMINAMATH_CALUDE_union_eq_univ_complement_inter_B_a_range_l1212_121287

-- Define the sets A, B, and C
def A : Set ℝ := {x | x^2 - 9*x + 18 ≥ 0}
def B : Set ℝ := {x | -2 < x ∧ x < 9}
def C (a : ℝ) : Set ℝ := {x | a < x ∧ x < a + 1}

-- State the theorems to be proved
theorem union_eq_univ : A ∪ B = Set.univ := by sorry

theorem complement_inter_B : (Aᶜ) ∩ B = {x : ℝ | 3 < x ∧ x < 6} := by sorry

theorem a_range (a : ℝ) : C a ⊆ B → -2 ≤ a ∧ a ≤ 8 := by sorry

end NUMINAMATH_CALUDE_union_eq_univ_complement_inter_B_a_range_l1212_121287


namespace NUMINAMATH_CALUDE_parabola_vertex_coordinates_l1212_121270

/-- The vertex coordinates of the parabola y = -2x^2 + 8x - 3 are (2, 5) -/
theorem parabola_vertex_coordinates :
  let f (x : ℝ) := -2 * x^2 + 8 * x - 3
  ∃ (x y : ℝ), (x, y) = (2, 5) ∧ 
    (∀ t : ℝ, f t ≤ f x) ∧
    y = f x :=
by sorry

end NUMINAMATH_CALUDE_parabola_vertex_coordinates_l1212_121270


namespace NUMINAMATH_CALUDE_swim_meet_car_occupancy_l1212_121286

theorem swim_meet_car_occupancy :
  let num_cars : ℕ := 2
  let num_vans : ℕ := 3
  let people_per_van : ℕ := 3
  let max_car_capacity : ℕ := 6
  let max_van_capacity : ℕ := 8
  let additional_capacity : ℕ := 17
  
  let total_van_occupancy : ℕ := num_vans * people_per_van
  let total_max_capacity : ℕ := num_cars * max_car_capacity + num_vans * max_van_capacity
  let actual_total_occupancy : ℕ := total_max_capacity - additional_capacity
  let car_occupancy : ℕ := actual_total_occupancy - total_van_occupancy
  
  car_occupancy / num_cars = 5 :=
by sorry

end NUMINAMATH_CALUDE_swim_meet_car_occupancy_l1212_121286


namespace NUMINAMATH_CALUDE_triangle_side_length_l1212_121220

theorem triangle_side_length (a b c : ℝ) (A B C : ℝ) :
  a > 0 ∧ b > 0 ∧ c > 0 →
  a^2 - c^2 = 2*b →
  Real.sin B = 4 * Real.cos A * Real.sin C →
  b = 4 := by
  sorry

end NUMINAMATH_CALUDE_triangle_side_length_l1212_121220


namespace NUMINAMATH_CALUDE_square_of_sum_equals_81_l1212_121222

theorem square_of_sum_equals_81 (x : ℝ) (h : Real.sqrt (x + 3) = 3) : 
  (x + 3)^2 = 81 := by
  sorry

end NUMINAMATH_CALUDE_square_of_sum_equals_81_l1212_121222


namespace NUMINAMATH_CALUDE_ten_to_ninety_mod_seven_day_after_ten_to_ninety_friday_after_ten_to_ninety_is_saturday_l1212_121250

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

def day_after_n_days (start : DayOfWeek) (n : Nat) : DayOfWeek :=
  match n with
  | 0 => start
  | n + 1 => next_day (day_after_n_days start n)

theorem ten_to_ninety_mod_seven : 10^90 % 7 = 1 := by sorry

theorem day_after_ten_to_ninety (start : DayOfWeek) :
  day_after_n_days start (10^90) = next_day start := by sorry

theorem friday_after_ten_to_ninety_is_saturday :
  day_after_n_days DayOfWeek.Friday (10^90) = DayOfWeek.Saturday := by sorry

end NUMINAMATH_CALUDE_ten_to_ninety_mod_seven_day_after_ten_to_ninety_friday_after_ten_to_ninety_is_saturday_l1212_121250


namespace NUMINAMATH_CALUDE_inequality_system_solution_exists_l1212_121229

theorem inequality_system_solution_exists : ∃ (x y z t : ℝ), 
  (x ≠ 0 ∨ y ≠ 0 ∨ z ≠ 0 ∨ t ≠ 0) ∧
  (abs x < abs (y - z + t)) ∧
  (abs y < abs (x - z + t)) ∧
  (abs z < abs (x - y + t)) ∧
  (abs t < abs (x - y + z)) := by
  sorry

end NUMINAMATH_CALUDE_inequality_system_solution_exists_l1212_121229


namespace NUMINAMATH_CALUDE_sqrt_square_fourteen_l1212_121271

theorem sqrt_square_fourteen : Real.sqrt (14^2) = 14 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_square_fourteen_l1212_121271


namespace NUMINAMATH_CALUDE_circle_tangent_to_directrix_l1212_121234

-- Define the parabola
def parabola (x y : ℝ) : Prop := y = (1/4) * x^2

-- Define the point A
def point_A : ℝ × ℝ := (0, 1)

-- Define the circle
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

-- Define the property that the circle passes through point A
def passes_through_A (c : Circle) : Prop :=
  let (x, y) := c.center
  (x - point_A.1)^2 + (y - point_A.2)^2 = c.radius^2

-- Define the property that the circle's center lies on the parabola
def center_on_parabola (c : Circle) : Prop :=
  let (x, y) := c.center
  parabola x y

-- Define the property that the circle is tangent to line l
def tangent_to_l (c : Circle) (l : ℝ → ℝ) : Prop :=
  let (x, y) := c.center
  (y - l x)^2 = c.radius^2

-- State the theorem
theorem circle_tangent_to_directrix :
  ∀ c : Circle,
  passes_through_A c →
  center_on_parabola c →
  ∃ l : ℝ → ℝ, (∀ x, l x = -1) ∧ tangent_to_l c l :=
sorry

end NUMINAMATH_CALUDE_circle_tangent_to_directrix_l1212_121234


namespace NUMINAMATH_CALUDE_fraction_simplification_l1212_121242

theorem fraction_simplification :
  (1 * 2 + 2 * 4 - 3 * 8 + 4 * 16 + 5 * 32 - 6 * 64) /
  (2 * 4 + 4 * 8 - 6 * 16 + 8 * 32 + 10 * 64 - 12 * 128) = 1 / 4 := by
  sorry

end NUMINAMATH_CALUDE_fraction_simplification_l1212_121242


namespace NUMINAMATH_CALUDE_cordelia_hair_dyeing_l1212_121207

/-- Cordelia's hair dyeing problem -/
theorem cordelia_hair_dyeing (total_time bleach_time dye_time : ℝ) : 
  total_time = 9 ∧ dye_time = 2 * bleach_time ∧ total_time = bleach_time + dye_time → 
  bleach_time = 3 := by
  sorry

end NUMINAMATH_CALUDE_cordelia_hair_dyeing_l1212_121207


namespace NUMINAMATH_CALUDE_geometric_sequence_property_l1212_121219

/-- A geometric sequence -/
def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

/-- Theorem: In a geometric sequence, if a₅ = 2, then a₁ * a₉ = 4 -/
theorem geometric_sequence_property (a : ℕ → ℝ) 
  (h_geom : geometric_sequence a) (h_a5 : a 5 = 2) : a 1 * a 9 = 4 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_property_l1212_121219


namespace NUMINAMATH_CALUDE_possible_values_of_a_l1212_121241

theorem possible_values_of_a (a b x : ℝ) 
  (h1 : a^3 - b^3 = 27*x^3) 
  (h2 : a - b = 2*x) : 
  a = x + 5*x/Real.sqrt 6 ∨ a = x - 5*x/Real.sqrt 6 :=
sorry

end NUMINAMATH_CALUDE_possible_values_of_a_l1212_121241


namespace NUMINAMATH_CALUDE_probability_red_or_blue_l1212_121262

theorem probability_red_or_blue 
  (prob_red : ℝ) 
  (prob_red_or_yellow : ℝ) 
  (h1 : prob_red = 0.45) 
  (h2 : prob_red_or_yellow = 0.65) 
  : prob_red + (1 - prob_red_or_yellow) = 0.80 := by
  sorry

end NUMINAMATH_CALUDE_probability_red_or_blue_l1212_121262


namespace NUMINAMATH_CALUDE_square_area_ratio_l1212_121258

/-- Given three squares with the specified relationships, prove that the ratio of the areas of the first and second squares is 1/2. -/
theorem square_area_ratio (s₃ : ℝ) (h₃ : s₃ > 0) : 
  let s₁ := s₃ * Real.sqrt 2
  let s₂ := s₁ * Real.sqrt 2
  (s₁^2) / (s₂^2) = 1/2 := by sorry

end NUMINAMATH_CALUDE_square_area_ratio_l1212_121258


namespace NUMINAMATH_CALUDE_smallest_multiplier_perfect_square_l1212_121277

theorem smallest_multiplier_perfect_square (x : ℕ+) :
  (∃ y : ℕ+, y = 2 ∧ 
    (∃ z : ℕ+, x * y = z^2) ∧
    (∀ w : ℕ+, w < y → ¬∃ v : ℕ+, x * w = v^2)) →
  x = 2 := by
sorry

end NUMINAMATH_CALUDE_smallest_multiplier_perfect_square_l1212_121277


namespace NUMINAMATH_CALUDE_school_costume_problem_l1212_121297

/-- Represents the price of a costume set based on the quantity purchased -/
def price (n : ℕ) : ℕ :=
  if n ≤ 45 then 60
  else if n ≤ 90 then 50
  else 40

/-- The problem statement -/
theorem school_costume_problem :
  ∃ (a b : ℕ),
    a + b = 92 ∧
    a > b ∧
    a < 90 ∧
    a * price a + b * price b = 5020 ∧
    a = 50 ∧
    b = 42 ∧
    92 * 40 = a * price a + b * price b - 480 :=
by
  sorry


end NUMINAMATH_CALUDE_school_costume_problem_l1212_121297


namespace NUMINAMATH_CALUDE_f_of_4_equals_82_l1212_121227

-- Define a monotonic function f
def monotonic_function (f : ℝ → ℝ) : Prop :=
  ∀ x y, x ≤ y → f x ≤ f y ∨ f y ≤ f x

-- State the theorem
theorem f_of_4_equals_82
  (f : ℝ → ℝ)
  (h_monotonic : monotonic_function f)
  (h_condition : ∀ x : ℝ, f (f x - 3^x) = 4) :
  f 4 = 82 := by
  sorry

end NUMINAMATH_CALUDE_f_of_4_equals_82_l1212_121227


namespace NUMINAMATH_CALUDE_expression_evaluation_l1212_121238

theorem expression_evaluation (x : ℝ) : 
  x * (x * (x * (3 - x) - 5) + 12) + 2 = -x^4 + 3*x^3 - 5*x^2 + 12*x + 2 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l1212_121238


namespace NUMINAMATH_CALUDE_parabola_vertex_specific_parabola_vertex_l1212_121290

/-- The vertex of a parabola defined by y = a(x-h)^2 + k has coordinates (h, k) -/
theorem parabola_vertex (a h k : ℝ) :
  let f : ℝ → ℝ := λ x => a * (x - h)^2 + k
  (∀ x, f x ≥ f h) ∧ f h = k :=
by sorry

/-- The vertex of the parabola y = 3(x-5)^2 + 4 has coordinates (5, 4) -/
theorem specific_parabola_vertex :
  let f : ℝ → ℝ := λ x => 3 * (x - 5)^2 + 4
  (∀ x, f x ≥ f 5) ∧ f 5 = 4 :=
by sorry

end NUMINAMATH_CALUDE_parabola_vertex_specific_parabola_vertex_l1212_121290


namespace NUMINAMATH_CALUDE_square_area_ratio_l1212_121215

theorem square_area_ratio (side_C side_D : ℝ) (h1 : side_C = 48) (h2 : side_D = 60) :
  (side_C ^ 2) / (side_D ^ 2) = 16 / 25 := by
  sorry

end NUMINAMATH_CALUDE_square_area_ratio_l1212_121215


namespace NUMINAMATH_CALUDE_acute_triangle_special_angles_l1212_121296

theorem acute_triangle_special_angles :
  ∃ (α β γ : ℕ),
    α + β + γ = 180 ∧
    0 < γ ∧ γ < β ∧ β < α ∧ α < 90 ∧
    α = 5 * γ ∧
    (α = 85 ∧ β = 78 ∧ γ = 17) := by
  sorry

end NUMINAMATH_CALUDE_acute_triangle_special_angles_l1212_121296


namespace NUMINAMATH_CALUDE_not_parabola_l1212_121228

/-- The equation x^2 + ky^2 = 1 cannot represent a parabola for any real k -/
theorem not_parabola (k : ℝ) : 
  ¬ ∃ (a b c d e : ℝ), ∀ (x y : ℝ), 
    x^2 + k*y^2 = 1 ↔ a*x^2 + b*x*y + c*y^2 + d*x + e*y = 0 ∧ b^2 = 4*a*c := by
  sorry

end NUMINAMATH_CALUDE_not_parabola_l1212_121228


namespace NUMINAMATH_CALUDE_bobs_weight_l1212_121224

/-- Given two people, Jim and Bob, prove Bob's weight under specific conditions. -/
theorem bobs_weight (jim_weight bob_weight : ℝ) : 
  (jim_weight + bob_weight = 200) →
  (bob_weight + jim_weight = bob_weight / 3) →
  bob_weight = 120 := by
  sorry

end NUMINAMATH_CALUDE_bobs_weight_l1212_121224


namespace NUMINAMATH_CALUDE_equation_solution_l1212_121261

theorem equation_solution :
  ∀ x y : ℝ, 
    y ≠ 0 →
    (4 * y^2 + 1) * (x^4 + 2 * x^2 + 2) = 8 * |y| * (x^2 + 1) →
    ((x = 0 ∧ y = 1/2) ∨ (x = 0 ∧ y = -1/2)) :=
by sorry

end NUMINAMATH_CALUDE_equation_solution_l1212_121261


namespace NUMINAMATH_CALUDE_total_dividend_is_825_l1212_121230

/-- Represents the investment scenario with two types of shares --/
structure Investment where
  total_amount : ℕ
  type_a_face_value : ℕ
  type_b_face_value : ℕ
  type_a_premium : ℚ
  type_b_discount : ℚ
  type_a_dividend_rate : ℚ
  type_b_dividend_rate : ℚ

/-- Calculates the total dividend received from the investment --/
def calculate_total_dividend (inv : Investment) : ℚ :=
  sorry

/-- Theorem stating that the total dividend received is 825 --/
theorem total_dividend_is_825 :
  let inv : Investment := {
    total_amount := 14400,
    type_a_face_value := 100,
    type_b_face_value := 100,
    type_a_premium := 1/5,
    type_b_discount := 1/10,
    type_a_dividend_rate := 7/100,
    type_b_dividend_rate := 1/20
  }
  calculate_total_dividend inv = 825 := by sorry

end NUMINAMATH_CALUDE_total_dividend_is_825_l1212_121230


namespace NUMINAMATH_CALUDE_function_always_positive_l1212_121252

theorem function_always_positive (a : ℝ) (h : a ∈ Set.Icc (-1) 1) :
  (∀ x, x^2 + (a - 4) * x + 4 - 2 * a > 0) ↔ (∀ x, x < 1 ∨ x > 3) :=
sorry

end NUMINAMATH_CALUDE_function_always_positive_l1212_121252


namespace NUMINAMATH_CALUDE_ratio_of_shares_l1212_121235

theorem ratio_of_shares (total amount_c : ℕ) (h1 : total = 2000) (h2 : amount_c = 1600) :
  (total - amount_c) / amount_c = 1 / 4 := by
  sorry

end NUMINAMATH_CALUDE_ratio_of_shares_l1212_121235


namespace NUMINAMATH_CALUDE_average_age_of_first_and_fifth_dog_l1212_121285

def dog_ages (age1 age2 age3 age4 age5 : ℕ) : Prop :=
  age1 = 10 ∧
  age2 = age1 - 2 ∧
  age3 = age2 + 4 ∧
  age4 * 2 = age3 ∧
  age5 = age4 + 20

theorem average_age_of_first_and_fifth_dog (age1 age2 age3 age4 age5 : ℕ) :
  dog_ages age1 age2 age3 age4 age5 →
  (age1 + age5) / 2 = 18 :=
by sorry

end NUMINAMATH_CALUDE_average_age_of_first_and_fifth_dog_l1212_121285


namespace NUMINAMATH_CALUDE_book_selling_price_l1212_121298

theorem book_selling_price (cost_price selling_price : ℝ) : 
  cost_price = 200 →
  selling_price - cost_price = (340 - cost_price) + 0.05 * cost_price →
  selling_price = 350 := by
sorry

end NUMINAMATH_CALUDE_book_selling_price_l1212_121298


namespace NUMINAMATH_CALUDE_shaded_area_theorem_l1212_121213

/-- The fraction of shaded area in each subdivision -/
def shaded_fraction : ℚ := 7 / 16

/-- The ratio of area of each subdivision to the whole square -/
def subdivision_ratio : ℚ := 1 / 16

/-- The total shaded fraction of the square -/
def total_shaded_fraction : ℚ := 7 / 15

theorem shaded_area_theorem :
  (shaded_fraction * (1 - subdivision_ratio)⁻¹ : ℚ) = total_shaded_fraction := by
  sorry

end NUMINAMATH_CALUDE_shaded_area_theorem_l1212_121213


namespace NUMINAMATH_CALUDE_two_corners_are_diagonal_endpoints_l1212_121231

/-- A structure representing a checkered rectangle divided into dominoes with diagonals -/
structure CheckeredRectangle where
  rows : ℕ
  cols : ℕ
  dominoes : List (Nat × Nat × Nat × Nat)
  diagonals : List (Nat × Nat × Nat × Nat)

/-- Predicate to check if a point is a corner of the rectangle -/
def is_corner (r : CheckeredRectangle) (x y : ℕ) : Prop :=
  (x = 0 ∨ x = r.cols - 1) ∧ (y = 0 ∨ y = r.rows - 1)

/-- Predicate to check if a point is an endpoint of any diagonal -/
def is_diagonal_endpoint (r : CheckeredRectangle) (x y : ℕ) : Prop :=
  ∃ (x1 y1 x2 y2 : ℕ), (x1, y1, x2, y2) ∈ r.diagonals ∧ ((x = x1 ∧ y = y1) ∨ (x = x2 ∧ y = y2))

/-- The main theorem stating that exactly two corners are diagonal endpoints -/
theorem two_corners_are_diagonal_endpoints (r : CheckeredRectangle) 
  (h1 : ∀ (x1 y1 x2 y2 : ℕ), (x1, y1, x2, y2) ∈ r.dominoes → 
    ((x2 = x1 + 1 ∧ y2 = y1) ∨ (x2 = x1 ∧ y2 = y1 + 1)))
  (h2 : ∀ (x1 y1 x2 y2 : ℕ), (x1, y1, x2, y2) ∈ r.diagonals → 
    ∃ (x3 y3 x4 y4 : ℕ), (x3, y3, x4, y4) ∈ r.dominoes ∧ 
    ((x1 = x3 ∧ y1 = y3 ∧ x2 = x4 ∧ y2 = y4) ∨ (x1 = x4 ∧ y1 = y4 ∧ x2 = x3 ∧ y2 = y3)))
  (h3 : ∀ (x1 y1 x2 y2 x3 y3 x4 y4 : ℕ), 
    (x1, y1, x2, y2) ∈ r.diagonals → (x3, y3, x4, y4) ∈ r.diagonals → 
    (x1 ≠ x3 ∨ y1 ≠ y3) ∧ (x1 ≠ x4 ∨ y1 ≠ y4) ∧ (x2 ≠ x3 ∨ y2 ≠ y3) ∧ (x2 ≠ x4 ∨ y2 ≠ y4)) :
  ∃! (c1 c2 : ℕ × ℕ), 
    c1 ≠ c2 ∧ 
    is_corner r c1.1 c1.2 ∧ 
    is_corner r c2.1 c2.2 ∧ 
    is_diagonal_endpoint r c1.1 c1.2 ∧ 
    is_diagonal_endpoint r c2.1 c2.2 ∧ 
    (∀ (x y : ℕ), is_corner r x y → (x, y) ≠ c1 → (x, y) ≠ c2 → ¬is_diagonal_endpoint r x y) :=
sorry

end NUMINAMATH_CALUDE_two_corners_are_diagonal_endpoints_l1212_121231


namespace NUMINAMATH_CALUDE_fraction_decomposition_l1212_121237

theorem fraction_decomposition : 
  ∃ (A B : ℚ), A = -12/11 ∧ B = 113/11 ∧
  ∀ (x : ℚ), x ≠ 1 ∧ x ≠ -8/3 →
  (7*x - 19) / (3*x^2 + 5*x - 8) = A / (x - 1) + B / (3*x + 8) := by
  sorry

end NUMINAMATH_CALUDE_fraction_decomposition_l1212_121237


namespace NUMINAMATH_CALUDE_graph_horizontal_shift_l1212_121201

-- Define a generic function g
variable (g : ℝ → ℝ)

-- Define a point (x, y) on the graph of y = g(x)
variable (x y : ℝ)

-- Define the horizontal shift
def h : ℝ := 3

-- Theorem statement
theorem graph_horizontal_shift :
  y = g x ↔ y = g (x - h) :=
sorry

end NUMINAMATH_CALUDE_graph_horizontal_shift_l1212_121201


namespace NUMINAMATH_CALUDE_no_three_primes_arithmetic_progression_no_k_primes_arithmetic_progression_l1212_121294

theorem no_three_primes_arithmetic_progression (p₁ p₂ p₃ : ℕ) (d : ℕ) : 
  p₁ > 3 → p₂ > 3 → p₃ > 3 → 
  Nat.Prime p₁ → Nat.Prime p₂ → Nat.Prime p₃ → 
  d < 5 → 
  ¬(p₂ = p₁ + d ∧ p₃ = p₁ + 2*d) :=
sorry

theorem no_k_primes_arithmetic_progression (k : ℕ) (p : ℕ → ℕ) (d : ℕ) :
  k > 3 → 
  (∀ i, i ≤ k → p i > k) →
  (∀ i, i ≤ k → Nat.Prime (p i)) →
  d ≤ k + 1 →
  ¬(∀ i, i ≤ k → p i = p 1 + (i - 1) * d) :=
sorry

end NUMINAMATH_CALUDE_no_three_primes_arithmetic_progression_no_k_primes_arithmetic_progression_l1212_121294


namespace NUMINAMATH_CALUDE_smallest_integer_square_triple_plus_75_l1212_121251

theorem smallest_integer_square_triple_plus_75 :
  ∃ x : ℤ, (∀ y : ℤ, y^2 = 3*y + 75 → x ≤ y) ∧ x^2 = 3*x + 75 :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_smallest_integer_square_triple_plus_75_l1212_121251


namespace NUMINAMATH_CALUDE_log_function_unique_parameters_l1212_121253

-- Define the logarithm function
noncomputable def log_base (a : ℝ) (x : ℝ) : ℝ := Real.log x / Real.log a

-- Define the function f(x) = log_a(x+b)
noncomputable def f (a b : ℝ) (x : ℝ) : ℝ := log_base a (x + b)

-- State the theorem
theorem log_function_unique_parameters :
  ∀ a b : ℝ, a > 0 → a ≠ 1 →
  (f a b (-1) = 0 ∧ f a b 0 = 1) →
  (a = 2 ∧ b = 2) :=
by sorry

end NUMINAMATH_CALUDE_log_function_unique_parameters_l1212_121253


namespace NUMINAMATH_CALUDE_value_of_x_l1212_121232

theorem value_of_x : ∀ (x a b c : ℤ),
  x = a + 7 →
  a = b + 12 →
  b = c + 25 →
  c = 95 →
  x = 139 := by
  sorry

end NUMINAMATH_CALUDE_value_of_x_l1212_121232


namespace NUMINAMATH_CALUDE_quadratic_coefficient_theorem_l1212_121283

theorem quadratic_coefficient_theorem (b c : ℝ) : 
  (∀ x : ℝ, x^2 + b*x + c = 0 ↔ x = 2 ∨ x = -8) → 
  b = 6 ∧ c = -16 := by
sorry

end NUMINAMATH_CALUDE_quadratic_coefficient_theorem_l1212_121283


namespace NUMINAMATH_CALUDE_cube_difference_l1212_121274

theorem cube_difference (a b : ℝ) (h1 : a - b = 7) (h2 : a^2 + b^2 = 53) : 
  a^3 - b^3 = 385 := by
sorry

end NUMINAMATH_CALUDE_cube_difference_l1212_121274


namespace NUMINAMATH_CALUDE_sum_of_first_5n_integers_l1212_121221

theorem sum_of_first_5n_integers (n : ℕ) : 
  (3 * n * (3 * n + 1)) / 2 = (n * (n + 1)) / 2 + 210 → 
  (5 * n * (5 * n + 1)) / 2 = 630 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_first_5n_integers_l1212_121221


namespace NUMINAMATH_CALUDE_fraction_addition_l1212_121226

theorem fraction_addition (a : ℝ) (ha : a ≠ 0) : 3 / a + 2 / a = 5 / a := by
  sorry

end NUMINAMATH_CALUDE_fraction_addition_l1212_121226


namespace NUMINAMATH_CALUDE_chord_length_theorem_l1212_121263

theorem chord_length_theorem (R AB BC : ℝ) (h_R : R = 12) (h_AB : AB = 6) (h_BC : BC = 4) :
  ∃ (AC : ℝ), (AC = Real.sqrt 35 + Real.sqrt 15) ∨ (AC = Real.sqrt 35 - Real.sqrt 15) := by
  sorry

end NUMINAMATH_CALUDE_chord_length_theorem_l1212_121263


namespace NUMINAMATH_CALUDE_intersection_complement_equality_l1212_121265

def U : Set ℕ := {1, 2, 3, 4, 5}
def A : Set ℕ := {1, 2, 3}
def B : Set ℕ := {2, 5}

theorem intersection_complement_equality : A ∩ (U \ B) = {1, 3} := by
  sorry

end NUMINAMATH_CALUDE_intersection_complement_equality_l1212_121265


namespace NUMINAMATH_CALUDE_steves_gold_bars_l1212_121243

theorem steves_gold_bars (friends : ℕ) (lost_bars : ℕ) (bars_per_friend : ℕ) : 
  friends = 4 → lost_bars = 20 → bars_per_friend = 20 →
  friends * bars_per_friend + lost_bars = 100 := by
  sorry

end NUMINAMATH_CALUDE_steves_gold_bars_l1212_121243


namespace NUMINAMATH_CALUDE_average_weight_increase_l1212_121204

theorem average_weight_increase (initial_count : ℕ) (old_weight new_weight : ℝ) :
  initial_count = 8 →
  old_weight = 65 →
  new_weight = 98.6 →
  (new_weight - old_weight) / initial_count = 4.2 :=
by sorry

end NUMINAMATH_CALUDE_average_weight_increase_l1212_121204


namespace NUMINAMATH_CALUDE_max_piles_660_l1212_121284

/-- The maximum number of piles that can be created from a given number of stones,
    where any two pile sizes differ by strictly less than 2 times. -/
def maxPiles (totalStones : ℕ) : ℕ :=
  30 -- The actual implementation is not provided, just the result

/-- The condition that any two pile sizes differ by strictly less than 2 times -/
def validPileSizes (piles : List ℕ) : Prop :=
  ∀ (a b : ℕ), a ∈ piles → b ∈ piles → (a : ℝ) < 2 * b ∧ (b : ℝ) < 2 * a

theorem max_piles_660 :
  let n := maxPiles 660
  ∃ (piles : List ℕ), 
    piles.length = n ∧ 
    validPileSizes piles ∧ 
    piles.sum = 660 ∧
    ∀ (m : ℕ), m > n → 
      ¬∃ (largerPiles : List ℕ), 
        largerPiles.length = m ∧ 
        validPileSizes largerPiles ∧ 
        largerPiles.sum = 660 :=
by
  sorry

#eval maxPiles 660

end NUMINAMATH_CALUDE_max_piles_660_l1212_121284


namespace NUMINAMATH_CALUDE_original_bales_count_l1212_121245

theorem original_bales_count (bales_stacked bales_now : ℕ) 
  (h1 : bales_stacked = 26)
  (h2 : bales_now = 54) :
  bales_now - bales_stacked = 28 := by
  sorry

end NUMINAMATH_CALUDE_original_bales_count_l1212_121245


namespace NUMINAMATH_CALUDE_absolute_value_equation_product_l1212_121255

theorem absolute_value_equation_product (x : ℝ) : 
  (∃ x₁ x₂ : ℝ, (|5 * x₁| + 3 = 47 ∧ |5 * x₂| + 3 = 47) ∧ x₁ ≠ x₂ ∧ x₁ * x₂ = -1936/25) := by
  sorry

end NUMINAMATH_CALUDE_absolute_value_equation_product_l1212_121255


namespace NUMINAMATH_CALUDE_total_distance_is_963_l1212_121211

/-- The total combined distance of objects thrown by Bill, Ted, and Alice -/
def total_distance (ted_sticks ted_rocks : ℕ) 
  (bill_stick_dist bill_rock_dist : ℝ) : ℝ :=
  let bill_sticks := ted_sticks - 6
  let alice_sticks := ted_sticks / 2
  let bill_rocks := ted_rocks / 2
  let alice_rocks := bill_rocks * 3
  let ted_stick_dist := bill_stick_dist * 1.5
  let alice_stick_dist := bill_stick_dist * 2
  let ted_rock_dist := bill_rock_dist * 1.25
  let alice_rock_dist := bill_rock_dist * 3
  (bill_sticks : ℝ) * bill_stick_dist +
  (ted_sticks : ℝ) * ted_stick_dist +
  (alice_sticks : ℝ) * alice_stick_dist +
  (bill_rocks : ℝ) * bill_rock_dist +
  (ted_rocks : ℝ) * ted_rock_dist +
  (alice_rocks : ℝ) * alice_rock_dist

/-- Theorem stating the total distance is 963 meters given the problem conditions -/
theorem total_distance_is_963 :
  total_distance 12 18 8 6 = 963 := by
  sorry

end NUMINAMATH_CALUDE_total_distance_is_963_l1212_121211


namespace NUMINAMATH_CALUDE_circle_symmetry_l1212_121256

-- Define the given circle
def given_circle (x y : ℝ) : Prop := (x - 1)^2 + y^2 = 1

-- Define the symmetry line
def symmetry_line (x y : ℝ) : Prop := y = -x

-- Define the symmetry transformation
def symmetry (x y x' y' : ℝ) : Prop :=
  symmetry_line ((x + x') / 2) ((y + y') / 2) ∧ 
  (x - x')^2 + (y - y')^2 = (x' - x)^2 + (y' - y)^2

-- State the theorem
theorem circle_symmetry :
  ∀ (x y : ℝ),
    (∃ (x' y' : ℝ), symmetry x y x' y' ∧ given_circle x' y') ↔
    x^2 + (y + 1)^2 = 1 :=
by sorry

end NUMINAMATH_CALUDE_circle_symmetry_l1212_121256


namespace NUMINAMATH_CALUDE_projection_periodicity_l1212_121205

/-- Regular n-gon with vertices A₁, A₂, ..., Aₙ -/
structure RegularNGon (n : ℕ) where
  vertices : Fin n → ℝ × ℝ

/-- Point on a side of the n-gon -/
structure PointOnSide (n : ℕ) where
  ngon : RegularNGon n
  side : Fin n
  point : ℝ × ℝ

/-- Projection function that maps Mᵢ to Mᵢ₊₁ -/
def project (n : ℕ) (m : PointOnSide n) : PointOnSide n :=
  sorry

/-- The k-th projection of a point -/
def kthProjection (n k : ℕ) (m : PointOnSide n) : PointOnSide n :=
  sorry

theorem projection_periodicity (n : ℕ) (m : PointOnSide n) :
  (n = 4 → kthProjection n 13 m = m) ∧
  (n = 6 → kthProjection n 13 m = m) ∧
  (n = 10 → kthProjection n 11 m = m) :=
sorry

end NUMINAMATH_CALUDE_projection_periodicity_l1212_121205


namespace NUMINAMATH_CALUDE_highest_score_l1212_121246

theorem highest_score (a b c d : ℝ) 
  (sum_eq : a + b = c + d)
  (sum_ineq : b + d > a + c)
  (a_gt_bc : a > b + c) :
  d > a ∧ d > b ∧ d > c := by
  sorry

end NUMINAMATH_CALUDE_highest_score_l1212_121246


namespace NUMINAMATH_CALUDE_late_train_speed_l1212_121225

/-- Proves that given a journey of 15 km, if a train traveling at 100 kmph reaches the destination on time,
    and a train traveling at speed v kmph reaches the destination 15 minutes late, then v = 37.5 kmph. -/
theorem late_train_speed (journey_length : ℝ) (on_time_speed : ℝ) (late_time_diff : ℝ) (v : ℝ) :
  journey_length = 15 →
  on_time_speed = 100 →
  late_time_diff = 0.25 →
  journey_length / on_time_speed + late_time_diff = journey_length / v →
  v = 37.5 := by
  sorry

#check late_train_speed

end NUMINAMATH_CALUDE_late_train_speed_l1212_121225


namespace NUMINAMATH_CALUDE_light_bulb_resistance_l1212_121203

theorem light_bulb_resistance (U I R : ℝ) (hU : U = 220) (hI : I ≤ 0.11) (hOhm : I = U / R) : R ≥ 2000 := by
  sorry

end NUMINAMATH_CALUDE_light_bulb_resistance_l1212_121203


namespace NUMINAMATH_CALUDE_pharmacist_weights_exist_l1212_121217

theorem pharmacist_weights_exist : ∃ (a b c : ℝ), 
  0 < a ∧ a < 90 ∧
  0 < b ∧ b < 90 ∧
  0 < c ∧ c < 90 ∧
  a + b = 100 ∧
  a + c = 101 ∧
  b + c = 102 := by
sorry

end NUMINAMATH_CALUDE_pharmacist_weights_exist_l1212_121217


namespace NUMINAMATH_CALUDE_symmetry_implies_values_l1212_121223

/-- Two points are symmetric with respect to the y-axis if their x-coordinates are opposite and y-coordinates are equal -/
def symmetric_wrt_y_axis (p1 p2 : ℝ × ℝ) : Prop :=
  p1.1 = -p2.1 ∧ p1.2 = p2.2

theorem symmetry_implies_values :
  ∀ (a b : ℝ), symmetric_wrt_y_axis (a, 1) (5, b) → a = -5 ∧ b = 1 := by
  sorry

end NUMINAMATH_CALUDE_symmetry_implies_values_l1212_121223


namespace NUMINAMATH_CALUDE_orcs_per_squad_l1212_121249

theorem orcs_per_squad (total_weight : ℕ) (num_squads : ℕ) (weight_per_orc : ℕ) :
  total_weight = 1200 →
  num_squads = 10 →
  weight_per_orc = 15 →
  (total_weight / weight_per_orc) / num_squads = 8 :=
by
  sorry

end NUMINAMATH_CALUDE_orcs_per_squad_l1212_121249


namespace NUMINAMATH_CALUDE_average_of_numbers_l1212_121244

theorem average_of_numbers (x : ℝ) : 
  ((x + 5) + 14 + x + 5) / 4 = 9 → x = 6 := by
sorry

end NUMINAMATH_CALUDE_average_of_numbers_l1212_121244


namespace NUMINAMATH_CALUDE_train_length_l1212_121292

/-- The length of a train given its speed and time to cross a bridge -/
theorem train_length (train_speed : ℝ) (crossing_time : ℝ) (bridge_length : ℝ) : 
  train_speed = 45 * (5 / 18) → 
  crossing_time = 30 →
  bridge_length = 240 →
  (train_speed * crossing_time) - bridge_length = 135 := by
  sorry

#check train_length

end NUMINAMATH_CALUDE_train_length_l1212_121292


namespace NUMINAMATH_CALUDE_triangle_options_l1212_121268

/-- Represents a triangle with side lengths a, b, c and angles A, B, C -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- Checks if a triangle is right-angled -/
def isRightAngled (t : Triangle) : Prop :=
  t.a^2 + t.b^2 = t.c^2 ∨ t.a^2 + t.c^2 = t.b^2 ∨ t.b^2 + t.c^2 = t.a^2 ∨
  t.A = 90 ∨ t.B = 90 ∨ t.C = 90

theorem triangle_options (t : Triangle) :
  (t.b^2 = t.a^2 - t.c^2 → isRightAngled t) ∧
  (t.a / t.b = 3 / 4 ∧ t.a / t.c = 3 / 5 ∧ t.b / t.c = 4 / 5 → isRightAngled t) ∧
  (t.C = t.A - t.B → isRightAngled t) ∧
  (t.A / t.B = 3 / 4 ∧ t.A / t.C = 3 / 5 ∧ t.B / t.C = 4 / 5 → ¬isRightAngled t) :=
by sorry


end NUMINAMATH_CALUDE_triangle_options_l1212_121268


namespace NUMINAMATH_CALUDE_max_value_of_J_l1212_121276

def consecutive_numbers (n : ℕ) : List ℕ :=
  List.range n |>.map (· + 1)

def sum_equals_21 (a b c d : ℕ) : Prop :=
  a + b + c + d = 21

theorem max_value_of_J (nums : List ℕ) (A B C D E F G H I J K : ℕ) :
  nums = consecutive_numbers 11 →
  D ∈ nums → G ∈ nums → I ∈ nums → F ∈ nums → A ∈ nums →
  B ∈ nums → C ∈ nums → E ∈ nums → H ∈ nums → J ∈ nums → K ∈ nums →
  D > G → G > I → I > F → F > A →
  sum_equals_21 A B C D →
  sum_equals_21 D E F G →
  sum_equals_21 G H F I →
  sum_equals_21 I J K A →
  J ≤ 9 :=
sorry

end NUMINAMATH_CALUDE_max_value_of_J_l1212_121276


namespace NUMINAMATH_CALUDE_units_digit_problem_l1212_121210

theorem units_digit_problem : ∃ x : ℕ, 
  (x > 0) ∧ 
  (x % 10 = 3) ∧ 
  ((35^87 + x^53) % 10 = 8) := by
  sorry

end NUMINAMATH_CALUDE_units_digit_problem_l1212_121210


namespace NUMINAMATH_CALUDE_find_number_of_elements_number_of_elements_is_ten_l1212_121259

/-- Given an incorrect average and a correction, find the number of elements -/
theorem find_number_of_elements (incorrect_avg correct_avg : ℚ) 
  (incorrect_value correct_value : ℚ) : ℚ :=
  let n := (correct_value - incorrect_value) / (correct_avg - incorrect_avg)
  n

/-- Proof that the number of elements is 10 given the specific conditions -/
theorem number_of_elements_is_ten : 
  find_number_of_elements 20 26 26 86 = 10 := by
  sorry

end NUMINAMATH_CALUDE_find_number_of_elements_number_of_elements_is_ten_l1212_121259


namespace NUMINAMATH_CALUDE_tiling_colors_l1212_121248

/-- Represents the type of tiling: squares or hexagons -/
inductive TilingType
  | Squares
  | Hexagons

/-- Calculates the number of colors needed for a specific tiling type and grid parameters -/
def number_of_colors (t : TilingType) (k l : ℕ) : ℕ :=
  match t with
  | TilingType.Squares => k^2 + l^2
  | TilingType.Hexagons => k^2 + k*l + l^2

/-- Theorem stating the number of colors needed for a valid tiling -/
theorem tiling_colors (t : TilingType) (k l : ℕ) (h : k ≠ 0 ∨ l ≠ 0) :
  ∃ (n : ℕ), n = number_of_colors t k l ∧ n > 0 :=
by sorry

end NUMINAMATH_CALUDE_tiling_colors_l1212_121248


namespace NUMINAMATH_CALUDE_initial_bacteria_count_l1212_121233

def bacteria_growth (initial_count : ℕ) (time : ℕ) : ℕ :=
  initial_count * 4^(time / 30)

theorem initial_bacteria_count :
  ∃ (initial_count : ℕ),
    bacteria_growth initial_count 360 = 262144 ∧
    initial_count = 1 :=
by sorry

end NUMINAMATH_CALUDE_initial_bacteria_count_l1212_121233


namespace NUMINAMATH_CALUDE_child_tickets_sold_l1212_121282

theorem child_tickets_sold (adult_price child_price total_tickets total_receipts : ℕ) 
  (h1 : adult_price = 12)
  (h2 : child_price = 4)
  (h3 : total_tickets = 130)
  (h4 : total_receipts = 840) :
  ∃ (adult_tickets child_tickets : ℕ),
    adult_tickets + child_tickets = total_tickets ∧
    adult_price * adult_tickets + child_price * child_tickets = total_receipts ∧
    child_tickets = 90 := by
  sorry

end NUMINAMATH_CALUDE_child_tickets_sold_l1212_121282


namespace NUMINAMATH_CALUDE_choose_three_from_eight_l1212_121289

theorem choose_three_from_eight (n : ℕ) (k : ℕ) : n = 8 → k = 3 → Nat.choose n k = 56 := by sorry

end NUMINAMATH_CALUDE_choose_three_from_eight_l1212_121289


namespace NUMINAMATH_CALUDE_esperanza_savings_l1212_121214

theorem esperanza_savings :
  let rent : ℕ := 600
  let food_cost : ℕ := (3 * rent) / 5
  let mortgage : ℕ := 3 * food_cost
  let gross_salary : ℕ := 4840
  let expenses : ℕ := rent + food_cost + mortgage
  let pre_tax_savings : ℕ := gross_salary - expenses
  let taxes : ℕ := (2 * pre_tax_savings) / 5
  let savings : ℕ := pre_tax_savings - taxes
  savings = 1680 := by sorry

end NUMINAMATH_CALUDE_esperanza_savings_l1212_121214


namespace NUMINAMATH_CALUDE_investment_roi_difference_l1212_121267

def emma_investment : ℝ := 300
def briana_investment : ℝ := 500
def emma_yield_rate : ℝ := 0.15
def briana_yield_rate : ℝ := 0.10
def time_period : ℕ := 2

theorem investment_roi_difference :
  briana_investment * briana_yield_rate * time_period - 
  emma_investment * emma_yield_rate * time_period = 10 := by
  sorry

end NUMINAMATH_CALUDE_investment_roi_difference_l1212_121267


namespace NUMINAMATH_CALUDE_rancher_animals_count_l1212_121239

/-- Proves that a rancher with 5 times as many cows as horses and 140 cows has 168 animals in total -/
theorem rancher_animals_count (horses : ℕ) (cows : ℕ) : 
  cows = 5 * horses → cows = 140 → horses + cows = 168 := by
  sorry

end NUMINAMATH_CALUDE_rancher_animals_count_l1212_121239


namespace NUMINAMATH_CALUDE_solution_set_f_geq_4_range_of_a_l1212_121273

-- Define the function f
def f (x : ℝ) := |1 - 2*x| - |1 + x|

-- Theorem for the solution set of f(x) ≥ 4
theorem solution_set_f_geq_4 : 
  {x : ℝ | f x ≥ 4} = {x : ℝ | x ≤ -2 ∨ x ≥ 6} := by sorry

-- Theorem for the range of a
theorem range_of_a : 
  {a : ℝ | ∃ x, a^2 + 2*a + |1 + x| < f x} = {a : ℝ | -3 < a ∧ a < 1} := by sorry

end NUMINAMATH_CALUDE_solution_set_f_geq_4_range_of_a_l1212_121273


namespace NUMINAMATH_CALUDE_max_marks_calculation_l1212_121216

/-- The maximum marks in an exam where:
  - The passing mark is 35% of the maximum marks
  - A student got 185 marks
  - The student failed by 25 marks
-/
theorem max_marks_calculation : ∃ (M : ℝ), 
  (0.35 * M = 185 + 25) ∧ 
  (M = 600) := by
  sorry

end NUMINAMATH_CALUDE_max_marks_calculation_l1212_121216


namespace NUMINAMATH_CALUDE_least_denominator_for_0711_l1212_121202

theorem least_denominator_for_0711 : 
  ∃ (m : ℕ+), (711 : ℚ)/1000 ≤ m/45 ∧ m/45 < (712 : ℚ)/1000 ∧ 
  ∀ (n : ℕ+) (k : ℕ+), n < 45 → ¬((711 : ℚ)/1000 ≤ k/n ∧ k/n < (712 : ℚ)/1000) :=
by sorry

end NUMINAMATH_CALUDE_least_denominator_for_0711_l1212_121202


namespace NUMINAMATH_CALUDE_right_triangle_area_l1212_121299

/-- Given a right triangle with hypotenuse 13 meters and one side 5 meters, its area is 30 square meters. -/
theorem right_triangle_area (a b c : ℝ) : 
  a = 13 → -- hypotenuse is 13 meters
  b = 5 → -- one side is 5 meters
  c^2 + b^2 = a^2 → -- Pythagorean theorem (right triangle condition)
  (1/2 : ℝ) * b * c = 30 := by -- area formula
sorry

end NUMINAMATH_CALUDE_right_triangle_area_l1212_121299


namespace NUMINAMATH_CALUDE_inequality_solution_set_l1212_121240

theorem inequality_solution_set (x : ℝ) :
  (3 * x^2 - 1 > 13 - 5 * x) ↔ (x < -7 ∨ x > 2) :=
by sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l1212_121240


namespace NUMINAMATH_CALUDE_candy_cookie_packs_l1212_121260

-- Define the problem parameters
def num_trays : ℕ := 4
def cookies_per_tray : ℕ := 24
def cookies_per_pack : ℕ := 12

-- Define the theorem
theorem candy_cookie_packs : 
  (num_trays * cookies_per_tray) / cookies_per_pack = 8 := by
  sorry

end NUMINAMATH_CALUDE_candy_cookie_packs_l1212_121260


namespace NUMINAMATH_CALUDE_combination_equality_implies_five_l1212_121200

theorem combination_equality_implies_five (n : ℕ+) : 
  Nat.choose n 2 = Nat.choose (n - 1) 2 + Nat.choose (n - 1) 3 → n = 5 := by
  sorry

end NUMINAMATH_CALUDE_combination_equality_implies_five_l1212_121200


namespace NUMINAMATH_CALUDE_ball_drawing_problem_l1212_121254

-- Define the sample space
def Ω : Type := Fin 4

-- Define the probability measure
def P : Set Ω → ℝ := sorry

-- Define the events
def A : Set Ω := sorry -- Both balls are the same color
def B : Set Ω := sorry -- Both balls are different colors
def C : Set Ω := sorry -- The first ball drawn is red
def D : Set Ω := sorry -- The second ball drawn is red

-- State the theorem
theorem ball_drawing_problem :
  (P (A ∩ B) = 0) ∧
  (P (A ∩ C) = P A * P C) ∧
  (P (B ∩ C) = P B * P C) := by
  sorry

end NUMINAMATH_CALUDE_ball_drawing_problem_l1212_121254


namespace NUMINAMATH_CALUDE_xiao_bing_winning_probability_l1212_121291

-- Define the game parameters
def dice_outcomes : ℕ := 6 * 6
def same_number_outcomes : ℕ := 6
def xiao_cong_score : ℕ := 10
def xiao_bing_score : ℕ := 2

-- Define the probabilities
def prob_same_numbers : ℚ := same_number_outcomes / dice_outcomes
def prob_different_numbers : ℚ := 1 - prob_same_numbers

-- Define the expected scores
def xiao_cong_expected_score : ℚ := prob_same_numbers * xiao_cong_score
def xiao_bing_expected_score : ℚ := prob_different_numbers * xiao_bing_score

-- Theorem: The probability of Xiao Bing winning is 1/2
theorem xiao_bing_winning_probability : 
  xiao_cong_expected_score = xiao_bing_expected_score → 
  (1 : ℚ) / 2 = prob_different_numbers := by
  sorry

end NUMINAMATH_CALUDE_xiao_bing_winning_probability_l1212_121291
