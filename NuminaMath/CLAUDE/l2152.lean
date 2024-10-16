import Mathlib

namespace NUMINAMATH_CALUDE_total_birds_is_148_l2152_215209

/-- The number of birds seen on Monday -/
def monday_birds : ℕ := 70

/-- The number of birds seen on Tuesday -/
def tuesday_birds : ℕ := monday_birds / 2

/-- The number of birds seen on Wednesday -/
def wednesday_birds : ℕ := tuesday_birds + 8

/-- The total number of birds seen from Monday to Wednesday -/
def total_birds : ℕ := monday_birds + tuesday_birds + wednesday_birds

/-- Theorem stating that the total number of birds seen is 148 -/
theorem total_birds_is_148 : total_birds = 148 := by
  sorry

end NUMINAMATH_CALUDE_total_birds_is_148_l2152_215209


namespace NUMINAMATH_CALUDE_function_expression_l2152_215252

-- Define the function f
noncomputable def f (a b : ℝ) : ℝ → ℝ := λ x => (2 * b * x) / (a * x - 1)

-- State the theorem
theorem function_expression (a b : ℝ) 
  (h1 : a ≠ 0)
  (h2 : f a b 1 = 1)
  (h3 : ∃! x : ℝ, f a b x = 2 * x) :
  ∃ g : ℝ → ℝ, (∀ x, f a b x = g x) ∧ (∀ x, g x = (2 * x) / (x + 1)) :=
sorry

end NUMINAMATH_CALUDE_function_expression_l2152_215252


namespace NUMINAMATH_CALUDE_initial_peaches_l2152_215240

/-- Given a basket of peaches, prove that the initial number of peaches is 20 
    when 25 more are added to make a total of 45. -/
theorem initial_peaches (initial : ℕ) : initial + 25 = 45 → initial = 20 := by
  sorry

end NUMINAMATH_CALUDE_initial_peaches_l2152_215240


namespace NUMINAMATH_CALUDE_stock_price_decrease_l2152_215280

theorem stock_price_decrease (F : ℝ) (h1 : F > 0) : 
  let J := 0.9 * F
  let M := 0.8 * J
  (F - M) / F * 100 = 28 := by
sorry

end NUMINAMATH_CALUDE_stock_price_decrease_l2152_215280


namespace NUMINAMATH_CALUDE_min_value_theorem_l2152_215246

theorem min_value_theorem (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) (h : x * y * z = 27) :
  3 * x + 2 * y + z ≥ 18 * Real.rpow 2 (1/3) ∧
  ∃ (x₀ y₀ z₀ : ℝ), x₀ > 0 ∧ y₀ > 0 ∧ z₀ > 0 ∧ x₀ * y₀ * z₀ = 27 ∧
    3 * x₀ + 2 * y₀ + z₀ = 18 * Real.rpow 2 (1/3) :=
by sorry

end NUMINAMATH_CALUDE_min_value_theorem_l2152_215246


namespace NUMINAMATH_CALUDE_range_of_a_inequality_proof_l2152_215200

-- Question 1
theorem range_of_a (a : ℝ) :
  (∀ x ∈ Set.Icc (1/2 : ℝ) 1, |x - a| + |2*x - 1| ≤ |2*x + 1|) →
  a ∈ Set.Icc (-1 : ℝ) (5/2) :=
sorry

-- Question 2
theorem inequality_proof (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  a^2 * b^2 + a^2 + b^2 ≥ a * b * (a + b + 1) :=
sorry

end NUMINAMATH_CALUDE_range_of_a_inequality_proof_l2152_215200


namespace NUMINAMATH_CALUDE_isosceles_triangle_exists_unique_l2152_215292

/-- An isosceles triangle with given heights -/
structure IsoscelesTriangle where
  /-- Height corresponding to the base -/
  m_a : ℝ
  /-- Height corresponding to one of the equal sides -/
  m_b : ℝ
  /-- Condition for existence -/
  h : 2 * m_a > m_b

/-- Theorem stating the existence and uniqueness of an isosceles triangle with given heights -/
theorem isosceles_triangle_exists_unique (m_a m_b : ℝ) :
  Nonempty (Unique (IsoscelesTriangle)) ↔ 2 * m_a > m_b :=
sorry

end NUMINAMATH_CALUDE_isosceles_triangle_exists_unique_l2152_215292


namespace NUMINAMATH_CALUDE_cone_lateral_surface_area_l2152_215218

theorem cone_lateral_surface_area (r h : ℝ) (hr : r = 4) (hh : h = 5) :
  π * r * h = 20 * π := by
  sorry

end NUMINAMATH_CALUDE_cone_lateral_surface_area_l2152_215218


namespace NUMINAMATH_CALUDE_allan_initial_balloons_l2152_215281

/-- The number of balloons Allan initially brought to the park -/
def initial_balloons : ℕ := sorry

/-- The number of balloons Allan bought at the park -/
def bought_balloons : ℕ := 3

/-- The total number of balloons Allan had after buying more -/
def total_balloons : ℕ := 8

/-- Theorem stating that Allan initially brought 5 balloons to the park -/
theorem allan_initial_balloons : 
  initial_balloons = total_balloons - bought_balloons := by sorry

end NUMINAMATH_CALUDE_allan_initial_balloons_l2152_215281


namespace NUMINAMATH_CALUDE_vector_decomposition_l2152_215261

theorem vector_decomposition (e₁ e₂ a : ℝ × ℝ) :
  e₁ = (1, 2) →
  e₂ = (-2, 3) →
  a = (-1, 2) →
  a = (1/7 : ℝ) • e₁ + (4/7 : ℝ) • e₂ := by sorry

end NUMINAMATH_CALUDE_vector_decomposition_l2152_215261


namespace NUMINAMATH_CALUDE_eight_spotlights_illuminate_space_l2152_215293

/-- Represents a point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Represents a spotlight that can illuminate an octant -/
structure Spotlight where
  position : Point3D
  direction : Point3D  -- Normalized vector representing the direction

/-- Represents the space to be illuminated -/
def Space : Type := Unit

/-- Checks if a spotlight illuminates a given point in space -/
def illuminates (s : Spotlight) (p : Point3D) : Prop := sorry

/-- Checks if a set of spotlights illuminates the entire space -/
def illuminatesEntireSpace (spotlights : Finset Spotlight) : Prop := 
  ∀ p : Point3D, ∃ s ∈ spotlights, illuminates s p

/-- The main theorem stating that 8 spotlights can illuminate the entire space -/
theorem eight_spotlights_illuminate_space 
  (points : Finset Point3D) 
  (h : points.card = 8) : 
  ∃ spotlights : Finset Spotlight, 
    spotlights.card = 8 ∧ 
    (∀ s ∈ spotlights, ∃ p ∈ points, s.position = p) ∧
    illuminatesEntireSpace spotlights := by
  sorry

end NUMINAMATH_CALUDE_eight_spotlights_illuminate_space_l2152_215293


namespace NUMINAMATH_CALUDE_donation_start_age_l2152_215298

def annual_donation : ℕ := 8000
def current_age : ℕ := 71
def total_donation : ℕ := 440000

theorem donation_start_age :
  ∃ (start_age : ℕ),
    start_age = current_age - (total_donation / annual_donation) ∧
    start_age = 16 := by
  sorry

end NUMINAMATH_CALUDE_donation_start_age_l2152_215298


namespace NUMINAMATH_CALUDE_inscribed_trapezoid_area_l2152_215241

/-- Represents a trapezoid with an inscribed circle -/
structure InscribedTrapezoid where
  /-- The radius of the inscribed circle -/
  radius : ℝ
  /-- The length of the shorter base of the trapezoid -/
  shorterBase : ℝ
  /-- Assumption that the center of the circle lies on the longer base -/
  centerOnBase : Bool

/-- 
Given a trapezoid ABCD with an inscribed circle of radius 6,
where the center of the circle lies on the base AD,
and BC = 4, prove that the area of the trapezoid is 24√2.
-/
theorem inscribed_trapezoid_area 
  (t : InscribedTrapezoid) 
  (h1 : t.radius = 6) 
  (h2 : t.shorterBase = 4) 
  (h3 : t.centerOnBase = true) : 
  ∃ (area : ℝ), area = 24 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_inscribed_trapezoid_area_l2152_215241


namespace NUMINAMATH_CALUDE_perfect_square_condition_l2152_215259

theorem perfect_square_condition (m : ℝ) : 
  (∀ x : ℝ, ∃ y : ℝ, x^2 + m*x + 1 = y^2) → (m = 2 ∨ m = -2) := by
  sorry

end NUMINAMATH_CALUDE_perfect_square_condition_l2152_215259


namespace NUMINAMATH_CALUDE_dance_troupe_size_l2152_215227

/-- Represents the number of performers in a dance troupe with various skills -/
structure DanceTroupe where
  singers : ℕ
  dancers : ℕ
  instrumentalists : ℕ
  singer_dancers : ℕ
  singer_instrumentalists : ℕ
  dancer_instrumentalists : ℕ
  all_skilled : ℕ

/-- The conditions of the dance troupe problem -/
def dance_troupe_conditions (dt : DanceTroupe) : Prop :=
  dt.singers = 2 ∧
  dt.dancers = 26 ∧
  dt.instrumentalists = 22 ∧
  dt.singer_dancers = 8 ∧
  dt.singer_instrumentalists = 10 ∧
  dt.dancer_instrumentalists = 11 ∧
  (dt.singers + dt.dancers + dt.instrumentalists
    - dt.singer_dancers - dt.singer_instrumentalists - dt.dancer_instrumentalists + dt.all_skilled
    - (dt.singer_dancers + dt.singer_instrumentalists + dt.dancer_instrumentalists - 2 * dt.all_skilled)) =
  (dt.singer_dancers + dt.singer_instrumentalists + dt.dancer_instrumentalists - 2 * dt.all_skilled)

/-- The total number of performers in the dance troupe -/
def total_performers (dt : DanceTroupe) : ℕ :=
  dt.singers + dt.dancers + dt.instrumentalists
  - dt.singer_dancers - dt.singer_instrumentalists - dt.dancer_instrumentalists
  + dt.all_skilled

/-- Theorem stating that the total number of performers is 46 -/
theorem dance_troupe_size (dt : DanceTroupe) :
  dance_troupe_conditions dt → total_performers dt = 46 := by
  sorry


end NUMINAMATH_CALUDE_dance_troupe_size_l2152_215227


namespace NUMINAMATH_CALUDE_mike_working_time_l2152_215255

/-- Calculates the total working time in hours for Mike's car service tasks. -/
def calculate_working_time (wash_time min_per_car : ℕ) (oil_change_time min_per_car : ℕ)
  (tire_change_time min_per_set : ℕ) (cars_washed : ℕ) (oil_changes : ℕ)
  (tire_sets_changed : ℕ) : ℚ :=
  let total_minutes := wash_time * cars_washed +
                       oil_change_time * oil_changes +
                       tire_change_time * tire_sets_changed
  total_minutes / 60

theorem mike_working_time :
  calculate_working_time 10 15 30 9 6 2 = 4 := by
  sorry

end NUMINAMATH_CALUDE_mike_working_time_l2152_215255


namespace NUMINAMATH_CALUDE_tank_capacity_is_120_gallons_l2152_215220

/-- Represents the capacity of a water tank in gallons -/
def tank_capacity : ℝ := 120

/-- Represents the difference in gallons between 70% and 40% full -/
def difference : ℝ := 36

/-- Theorem stating that the tank capacity is 120 gallons -/
theorem tank_capacity_is_120_gallons : 
  (0.7 * tank_capacity - 0.4 * tank_capacity = difference) → 
  tank_capacity = 120 := by
  sorry

end NUMINAMATH_CALUDE_tank_capacity_is_120_gallons_l2152_215220


namespace NUMINAMATH_CALUDE_vacuum_savings_theorem_l2152_215243

/-- Calculate the number of weeks needed to save for a vacuum cleaner. -/
def weeks_to_save (initial_amount : ℕ) (weekly_savings : ℕ) (vacuum_cost : ℕ) : ℕ :=
  ((vacuum_cost - initial_amount) + weekly_savings - 1) / weekly_savings

/-- Theorem: It takes 10 weeks to save for the vacuum cleaner. -/
theorem vacuum_savings_theorem :
  weeks_to_save 20 10 120 = 10 := by
  sorry

#eval weeks_to_save 20 10 120

end NUMINAMATH_CALUDE_vacuum_savings_theorem_l2152_215243


namespace NUMINAMATH_CALUDE_pocket_balls_theorem_l2152_215201

/-- Represents the number of balls in each pocket -/
def pocket_balls : List Nat := [2, 4, 5]

/-- The total number of ways to take a ball from any pocket -/
def total_ways_one_ball : Nat := pocket_balls.sum

/-- The total number of ways to take one ball from each pocket -/
def total_ways_three_balls : Nat := pocket_balls.prod

theorem pocket_balls_theorem :
  total_ways_one_ball = 11 ∧ total_ways_three_balls = 40 := by
  sorry

end NUMINAMATH_CALUDE_pocket_balls_theorem_l2152_215201


namespace NUMINAMATH_CALUDE_arithmetic_equality_l2152_215211

theorem arithmetic_equality : 3 * (7 - 5) - 5 = 1 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_equality_l2152_215211


namespace NUMINAMATH_CALUDE_no_universal_rational_compact_cover_l2152_215235

theorem no_universal_rational_compact_cover :
  ¬ (∃ (A : ℕ → Set ℚ), 
    (∀ n, IsCompact (A n)) ∧ 
    (∀ K : Set ℚ, IsCompact K → ∃ n, K ⊆ A n)) := by
  sorry

end NUMINAMATH_CALUDE_no_universal_rational_compact_cover_l2152_215235


namespace NUMINAMATH_CALUDE_sunzi_car_problem_l2152_215284

theorem sunzi_car_problem (x : ℕ) : 
  (3 * (x - 2) = 2 * x + 9) → x = 15 := by
  sorry

end NUMINAMATH_CALUDE_sunzi_car_problem_l2152_215284


namespace NUMINAMATH_CALUDE_roots_imply_k_range_l2152_215273

/-- The quadratic function f(x) = 2x^2 - kx + k - 3 -/
def f (k : ℝ) (x : ℝ) : ℝ := 2 * x^2 - k * x + k - 3

theorem roots_imply_k_range :
  ∀ k : ℝ,
  (∃ x₁ x₂ : ℝ, 
    f k x₁ = 0 ∧ 0 < x₁ ∧ x₁ < 1 ∧
    f k x₂ = 0 ∧ 1 < x₂ ∧ x₂ < 2) →
  3 < k ∧ k < 5 :=
by sorry

end NUMINAMATH_CALUDE_roots_imply_k_range_l2152_215273


namespace NUMINAMATH_CALUDE_inequality_proof_l2152_215213

theorem inequality_proof (a b c : ℝ) (ha : a ≥ 0) (hb : b ≥ 0) (hc : c ≥ 0) :
  a^4 + b^4 + c^4 - 2*(a^2*b^2 + a^2*c^2 + b^2*c^2) + a^2*b*c + b^2*a*c + c^2*a*b ≥ 0 :=
by sorry

end NUMINAMATH_CALUDE_inequality_proof_l2152_215213


namespace NUMINAMATH_CALUDE_rakesh_salary_rakesh_salary_proof_l2152_215294

theorem rakesh_salary : ℝ → Prop :=
  fun salary =>
    let fixed_deposit := 0.15 * salary
    let remaining_after_deposit := salary - fixed_deposit
    let groceries := 0.30 * remaining_after_deposit
    let cash_in_hand := remaining_after_deposit - groceries
    cash_in_hand = 2380 → salary = 4000

-- Proof
theorem rakesh_salary_proof : rakesh_salary 4000 := by
  sorry

end NUMINAMATH_CALUDE_rakesh_salary_rakesh_salary_proof_l2152_215294


namespace NUMINAMATH_CALUDE_hammond_statues_weight_l2152_215289

/-- The weight of Hammond's marble statues problem -/
theorem hammond_statues_weight (original_weight : ℕ) (first_statue : ℕ) (third_statue : ℕ) (fourth_statue : ℕ) (discarded : ℕ) :
  original_weight = 80 ∧ 
  first_statue = 10 ∧ 
  third_statue = 15 ∧ 
  fourth_statue = 15 ∧ 
  discarded = 22 →
  ∃ (second_statue : ℕ), 
    second_statue = 18 ∧ 
    original_weight = first_statue + second_statue + third_statue + fourth_statue + discarded :=
by sorry

end NUMINAMATH_CALUDE_hammond_statues_weight_l2152_215289


namespace NUMINAMATH_CALUDE_candy_bar_price_is_correct_l2152_215224

/-- The selling price of a candy bar that results in a $25 profit when selling 5 boxes of 10 candy bars, each bought for $1. -/
def candy_bar_price : ℚ :=
  let boxes : ℕ := 5
  let bars_per_box : ℕ := 10
  let cost_price : ℚ := 1
  let total_profit : ℚ := 25
  let total_bars : ℕ := boxes * bars_per_box
  (total_profit / total_bars + cost_price)

theorem candy_bar_price_is_correct : candy_bar_price = 3/2 := by
  sorry

end NUMINAMATH_CALUDE_candy_bar_price_is_correct_l2152_215224


namespace NUMINAMATH_CALUDE_tom_apple_slices_l2152_215275

theorem tom_apple_slices (total_apples : ℕ) (slices_per_apple : ℕ) (slices_left : ℕ) :
  total_apples = 2 →
  slices_per_apple = 8 →
  slices_left = 5 →
  (∃ (slices_given : ℕ),
    slices_given + 2 * slices_left = total_apples * slices_per_apple ∧
    slices_given = (3 : ℚ) / 8 * (total_apples * slices_per_apple : ℚ)) :=
by sorry

end NUMINAMATH_CALUDE_tom_apple_slices_l2152_215275


namespace NUMINAMATH_CALUDE_sweet_potato_harvest_l2152_215205

theorem sweet_potato_harvest (sold_to_adams : ℕ) (sold_to_lenon : ℕ) (not_sold : ℕ) :
  sold_to_adams = 20 →
  sold_to_lenon = 15 →
  not_sold = 45 →
  sold_to_adams + sold_to_lenon + not_sold = 80 :=
by sorry

end NUMINAMATH_CALUDE_sweet_potato_harvest_l2152_215205


namespace NUMINAMATH_CALUDE_total_books_l2152_215272

theorem total_books (stu_books : ℝ) (albert_ratio : ℝ) : 
  stu_books = 9 → 
  albert_ratio = 4.5 → 
  stu_books + albert_ratio * stu_books = 49.5 := by
sorry

end NUMINAMATH_CALUDE_total_books_l2152_215272


namespace NUMINAMATH_CALUDE_complex_equation_solution_l2152_215250

theorem complex_equation_solution (z : ℂ) : 
  Complex.I * (z - 1) = 1 + Complex.I → z = 2 - Complex.I := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l2152_215250


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_condition_l2152_215239

-- Define the line and hyperbola
def line (a x y : ℝ) : Prop := 2 * a * x - y + 2 * a^2 = 0
def hyperbola (a x y : ℝ) : Prop := x^2 / a^2 - y^2 / 4 = 1

-- Define the condition for no focus
def no_focus (a : ℝ) : Prop := ∀ x y : ℝ, line a x y → hyperbola a x y → False

-- State the theorem
theorem sufficient_not_necessary_condition (a : ℝ) (h : a > 0) :
  (a ≥ 2 → no_focus a) ∧ ¬(no_focus a → a ≥ 2) :=
by sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_condition_l2152_215239


namespace NUMINAMATH_CALUDE_point_B_coordinates_l2152_215223

-- Define the coordinate system
def Point := ℝ × ℝ

-- Define point A
def A : Point := (2, 1)

-- Define the length of AB
def AB_length : ℝ := 4

-- Define a function to check if a point is on the line parallel to x-axis passing through A
def on_parallel_line (p : Point) : Prop :=
  p.2 = A.2

-- Define a function to check if the distance between two points is correct
def correct_distance (p : Point) : Prop :=
  (p.1 - A.1)^2 + (p.2 - A.2)^2 = AB_length^2

-- Theorem statement
theorem point_B_coordinates :
  ∃ (B : Point), on_parallel_line B ∧ correct_distance B ∧
  (B = (6, 1) ∨ B = (-2, 1)) :=
sorry

end NUMINAMATH_CALUDE_point_B_coordinates_l2152_215223


namespace NUMINAMATH_CALUDE_factorization_of_cubic_l2152_215215

theorem factorization_of_cubic (x : ℝ) : 6 * x^3 - 24 = 6 * (x - 2) * (x^2 + 2*x + 4) := by
  sorry

end NUMINAMATH_CALUDE_factorization_of_cubic_l2152_215215


namespace NUMINAMATH_CALUDE_function_inequality_proof_l2152_215263

def f (x : ℝ) : ℝ := |2*x + 1| + |2*x - 3|

def solution_set_A : Set ℝ := {x | x < -1 ∨ x > 2}

def solution_set_B (a : ℝ) : Set ℝ := {x | f x > |a - 1|}

theorem function_inequality_proof :
  (∀ x, f x > 6 ↔ x ∈ solution_set_A) ∧
  (∀ a, solution_set_B a ⊆ solution_set_A ↔ a ≤ -5 ∨ a ≥ 7) :=
sorry

end NUMINAMATH_CALUDE_function_inequality_proof_l2152_215263


namespace NUMINAMATH_CALUDE_square_difference_l2152_215210

theorem square_difference (x y : ℚ) 
  (h1 : (x + y)^2 = 49/144) 
  (h2 : (x - y)^2 = 1/144) : 
  x^2 - y^2 = 7/144 := by
sorry

end NUMINAMATH_CALUDE_square_difference_l2152_215210


namespace NUMINAMATH_CALUDE_savings_proof_l2152_215208

def original_savings (furniture_fraction : ℚ) (tv_cost : ℕ) : ℕ :=
  4 * tv_cost

theorem savings_proof (furniture_fraction : ℚ) (tv_cost : ℕ) 
  (h1 : furniture_fraction = 3/4) 
  (h2 : tv_cost = 210) : 
  original_savings furniture_fraction tv_cost = 840 := by
  sorry

end NUMINAMATH_CALUDE_savings_proof_l2152_215208


namespace NUMINAMATH_CALUDE_yoongis_answer_l2152_215216

theorem yoongis_answer : ∃ x : ℝ, 5 * x = 100 ∧ x / 10 = 2 := by
  sorry

end NUMINAMATH_CALUDE_yoongis_answer_l2152_215216


namespace NUMINAMATH_CALUDE_prime_square_minus_cube_eq_one_l2152_215245

theorem prime_square_minus_cube_eq_one (p q : ℕ) : 
  Nat.Prime p ∧ Nat.Prime q ∧ p > 0 ∧ q > 0 → (p^2 - q^3 = 1 ↔ p = 3 ∧ q = 2) := by
  sorry

end NUMINAMATH_CALUDE_prime_square_minus_cube_eq_one_l2152_215245


namespace NUMINAMATH_CALUDE_right_triangle_area_l2152_215291

/-- The area of a right triangle with hypotenuse 13 and shortest side 5 is 30 -/
theorem right_triangle_area (a b c : ℝ) (h1 : a^2 + b^2 = c^2) (h2 : c = 13) (h3 : a = 5) (h4 : a ≤ b) : (1/2) * a * b = 30 := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_area_l2152_215291


namespace NUMINAMATH_CALUDE_average_divisible_by_seven_l2152_215285

theorem average_divisible_by_seven : ∃ (S : Finset ℕ), 
  (∀ n ∈ S, 12 < n ∧ n < 150 ∧ n % 7 = 0) ∧ 
  (∀ n, 12 < n → n < 150 → n % 7 = 0 → n ∈ S) ∧
  (S.sum id / S.card : ℚ) = 161/2 := by
  sorry

end NUMINAMATH_CALUDE_average_divisible_by_seven_l2152_215285


namespace NUMINAMATH_CALUDE_area_ratio_hexagons_l2152_215222

/-- A point in 2D space -/
structure Point :=
  (x : ℝ) (y : ℝ)

/-- A regular hexagon -/
structure RegularHexagon :=
  (center : Point)
  (sideLength : ℝ)

/-- An equilateral triangle -/
structure EquilateralTriangle :=
  (center : Point)
  (sideLength : ℝ)

/-- The hexagon ABCD -/
def hexagonABCD : RegularHexagon := sorry

/-- The equilateral triangles constructed on the sides of ABCD -/
def trianglesOnABCD : List EquilateralTriangle := sorry

/-- The hexagon EFGHIJ formed by the centers of the equilateral triangles -/
def hexagonEFGHIJ : RegularHexagon := sorry

/-- The area of a regular hexagon -/
def areaRegularHexagon (h : RegularHexagon) : ℝ := sorry

/-- Theorem: The ratio of the area of hexagon EFGHIJ to the area of hexagon ABCD is 4/3 -/
theorem area_ratio_hexagons :
  (areaRegularHexagon hexagonEFGHIJ) / (areaRegularHexagon hexagonABCD) = 4/3 := by
  sorry

end NUMINAMATH_CALUDE_area_ratio_hexagons_l2152_215222


namespace NUMINAMATH_CALUDE_exists_non_prime_product_l2152_215271

/-- The k-th prime number -/
def nthPrime (k : ℕ) : ℕ := sorry

/-- The product of the first n prime numbers plus 1 -/
def primeProduct (n : ℕ) : ℕ :=
  (List.range n).foldl (fun acc i => acc * nthPrime (i + 1)) 1 + 1

/-- Theorem stating that there exists a number n such that primeProduct n is not prime -/
theorem exists_non_prime_product : ∃ n : ℕ, ¬ Nat.Prime (primeProduct n) := by sorry

end NUMINAMATH_CALUDE_exists_non_prime_product_l2152_215271


namespace NUMINAMATH_CALUDE_dice_product_probability_dice_product_probability_proof_l2152_215230

/-- The probability of obtaining a product of 2 when tossing four standard dice -/
theorem dice_product_probability : ℝ :=
  let n_dice : ℕ := 4
  let dice_sides : ℕ := 6
  let target_product : ℕ := 2
  1 / 324

/-- Proof of the dice product probability theorem -/
theorem dice_product_probability_proof :
  dice_product_probability = 1 / 324 := by
  sorry

end NUMINAMATH_CALUDE_dice_product_probability_dice_product_probability_proof_l2152_215230


namespace NUMINAMATH_CALUDE_pythagorean_from_law_of_cosines_l2152_215207

/-- The law of cosines for a triangle with sides a, b, c and angle γ opposite side c -/
def lawOfCosines (a b c : ℝ) (γ : Real) : Prop :=
  c^2 = a^2 + b^2 - 2*a*b*Real.cos γ

/-- The Pythagorean theorem for a right triangle with sides a, b, c where c is the hypotenuse -/
def pythagoreanTheorem (a b c : ℝ) : Prop :=
  c^2 = a^2 + b^2

/-- Theorem stating that the Pythagorean theorem is a special case of the law of cosines -/
theorem pythagorean_from_law_of_cosines (a b c : ℝ) :
  lawOfCosines a b c (π/2) → pythagoreanTheorem a b c :=
by sorry

end NUMINAMATH_CALUDE_pythagorean_from_law_of_cosines_l2152_215207


namespace NUMINAMATH_CALUDE_fraction_simplification_l2152_215236

theorem fraction_simplification (a : ℝ) (h : a ≠ 1) :
  (a + 1) / (1 - a) * (a^2 + a) / (a^2 + 2*a + 1) - 1 / (1 - a) = -1 := by
  sorry

end NUMINAMATH_CALUDE_fraction_simplification_l2152_215236


namespace NUMINAMATH_CALUDE_similar_triangle_shortest_side_l2152_215249

theorem similar_triangle_shortest_side 
  (a b c : ℝ) 
  (h1 : a = 24) 
  (h2 : c = 25) 
  (h3 : a^2 + b^2 = c^2) 
  (h4 : 0 < a ∧ 0 < b ∧ 0 < c) 
  (h5 : a ≤ b) 
  (h_hypotenuse : ℝ) 
  (h6 : h_hypotenuse = 100) :
  ∃ (x : ℝ), x = 28 ∧ x = (a * h_hypotenuse) / c :=
by sorry

end NUMINAMATH_CALUDE_similar_triangle_shortest_side_l2152_215249


namespace NUMINAMATH_CALUDE_hyperbola_equation_l2152_215295

/-- A hyperbola with one focus at (0, 2) and asymptotic lines y = ±√3x has the equation y²/3 - x² = 1 -/
theorem hyperbola_equation (f : ℝ × ℝ) (asym : ℝ → ℝ) :
  f = (0, 2) →
  (∀ x, asym x = Real.sqrt 3 * x ∨ asym x = -Real.sqrt 3 * x) →
  ∃ hyperbola : ℝ × ℝ → Prop,
    (∀ x y, hyperbola (x, y) ↔ y^2 / 3 - x^2 = 1) :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_equation_l2152_215295


namespace NUMINAMATH_CALUDE_expression_equals_power_of_seven_l2152_215270

theorem expression_equals_power_of_seven : 
  6 * (7 + 1) * (7^2 + 1) * (7^4 + 1) * (7^8 + 1) + 1 = 7^16 := by sorry

end NUMINAMATH_CALUDE_expression_equals_power_of_seven_l2152_215270


namespace NUMINAMATH_CALUDE_no_triple_perfect_squares_l2152_215266

theorem no_triple_perfect_squares : 
  ¬ ∃ (a b c : ℕ+), 
    (∃ (x y z : ℕ), (a^2 * b * c + 2 : ℕ) = x^2 ∧ 
                    (b^2 * c * a + 2 : ℕ) = y^2 ∧ 
                    (c^2 * a * b + 2 : ℕ) = z^2) :=
by sorry

end NUMINAMATH_CALUDE_no_triple_perfect_squares_l2152_215266


namespace NUMINAMATH_CALUDE_moving_circle_properties_l2152_215237

/-- The trajectory of the center of a moving circle M that is externally tangent to O₁ and internally tangent to O₂ -/
def trajectory_equation (x y : ℝ) : Prop :=
  x^2 / 36 + y^2 / 27 = 1

/-- The product of slopes of lines connecting M(x,y) with fixed points -/
def slope_product (x y : ℝ) : Prop :=
  y ≠ 0 → (y / (x + 6)) * (y / (x - 6)) = -3/4

/-- Circle O₁ equation -/
def circle_O₁ (x y : ℝ) : Prop :=
  x^2 + y^2 + 6*x + 5 = 0

/-- Circle O₂ equation -/
def circle_O₂ (x y : ℝ) : Prop :=
  x^2 + y^2 - 6*x - 91 = 0

theorem moving_circle_properties
  (x y : ℝ)
  (h₁ : ∃ (r : ℝ), r > 0 ∧ ∀ (x' y' : ℝ), (x' - x)^2 + (y' - y)^2 = r^2 →
    (circle_O₁ x' y' ∨ circle_O₂ x' y') ∧ ¬(circle_O₁ x' y' ∧ circle_O₂ x' y')) :
  trajectory_equation x y ∧ slope_product x y :=
sorry

end NUMINAMATH_CALUDE_moving_circle_properties_l2152_215237


namespace NUMINAMATH_CALUDE_friends_chicken_pieces_l2152_215286

/-- Given a total number of chicken pieces, the number eaten by Lyndee, and the number of friends,
    calculate the number of pieces each friend ate. -/
def chicken_per_friend (total : ℕ) (lyndee_ate : ℕ) (num_friends : ℕ) : ℕ :=
  (total - lyndee_ate) / num_friends

theorem friends_chicken_pieces :
  chicken_per_friend 11 1 5 = 2 := by
sorry

end NUMINAMATH_CALUDE_friends_chicken_pieces_l2152_215286


namespace NUMINAMATH_CALUDE_cos_beta_equals_cos_alpha_l2152_215297

-- Define the angles α and β
variable (α β : Real)

-- Define the conditions
axiom vertices_at_origin : True  -- This condition is implicit in the angle definitions
axiom initial_sides_on_x_axis : True  -- This condition is implicit in the angle definitions
axiom terminal_sides_symmetric : β = 2 * Real.pi - α
axiom cos_alpha : Real.cos α = 2/3

-- Theorem to prove
theorem cos_beta_equals_cos_alpha : Real.cos β = 2/3 := by
  sorry

end NUMINAMATH_CALUDE_cos_beta_equals_cos_alpha_l2152_215297


namespace NUMINAMATH_CALUDE_exists_common_tiling_l2152_215221

/-- Represents a domino type with integer dimensions -/
structure Domino where
  length : ℤ
  width : ℤ

/-- Checks if a rectangle can be tiled by a given domino type -/
def canTile (d : Domino) (rectLength rectWidth : ℤ) : Prop :=
  rectLength ≥ max 1 (2 * d.length) ∧ rectWidth % (2 * d.width) = 0

/-- Proves the existence of a rectangle that can be tiled by either of two domino types -/
theorem exists_common_tiling (d1 d2 : Domino) : 
  ∃ (rectLength rectWidth : ℤ), 
    canTile d1 rectLength rectWidth ∧ canTile d2 rectLength rectWidth :=
by
  sorry

end NUMINAMATH_CALUDE_exists_common_tiling_l2152_215221


namespace NUMINAMATH_CALUDE_expression_equality_l2152_215277

theorem expression_equality (a x : ℝ) (h : a^(2*x) = Real.sqrt 2 - 1) :
  (a^(3*x) + a^(-3*x)) / (a^x + a^(-x)) = 2 * Real.sqrt 2 - 1 := by
  sorry

end NUMINAMATH_CALUDE_expression_equality_l2152_215277


namespace NUMINAMATH_CALUDE_bake_sale_money_raised_l2152_215274

/-- Represents the number of items in a dozen -/
def dozenSize : Nat := 12

/-- Calculates the total number of items given the number of dozens -/
def totalItems (dozens : Nat) : Nat := dozens * dozenSize

/-- Represents the price of a cookie in cents -/
def cookiePrice : Nat := 100

/-- Represents the price of a brownie or blondie in cents -/
def browniePrice : Nat := 200

/-- Calculates the total money raised from the bake sale -/
def totalMoneyRaised : Nat :=
  let bettyChocolateChip := totalItems 4
  let bettyOatmealRaisin := totalItems 6
  let bettyBrownies := totalItems 2
  let paigeSugar := totalItems 6
  let paigeBlondies := totalItems 3
  let paigeCreamCheese := totalItems 5

  let totalCookies := bettyChocolateChip + bettyOatmealRaisin + paigeSugar
  let totalBrowniesBlondies := bettyBrownies + paigeBlondies + paigeCreamCheese

  totalCookies * cookiePrice + totalBrowniesBlondies * browniePrice

theorem bake_sale_money_raised :
  totalMoneyRaised = 43200 := by sorry

end NUMINAMATH_CALUDE_bake_sale_money_raised_l2152_215274


namespace NUMINAMATH_CALUDE_arithmetic_mean_of_normal_distribution_l2152_215269

-- Define the arithmetic mean and standard deviation
variable (μ : ℝ) -- arithmetic mean
def σ : ℝ := 1.5 -- standard deviation

-- Define the relationship between the mean, standard deviation, and the given value
def value_two_std_below_mean : ℝ := μ - 2 * σ

-- State the theorem
theorem arithmetic_mean_of_normal_distribution :
  value_two_std_below_mean = 12 → μ = 15 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_mean_of_normal_distribution_l2152_215269


namespace NUMINAMATH_CALUDE_product_selection_probabilities_l2152_215225

def totalProducts : ℕ := 5
def authenticProducts : ℕ := 3
def defectiveProducts : ℕ := 2

theorem product_selection_probabilities :
  let totalSelections := totalProducts.choose 2
  let bothAuthenticSelections := authenticProducts.choose 2
  let mixedSelections := authenticProducts * defectiveProducts
  (bothAuthenticSelections : ℚ) / totalSelections = 3 / 10 ∧
  (mixedSelections : ℚ) / totalSelections = 3 / 5 ∧
  1 - (bothAuthenticSelections : ℚ) / totalSelections = 7 / 10 := by
  sorry

end NUMINAMATH_CALUDE_product_selection_probabilities_l2152_215225


namespace NUMINAMATH_CALUDE_math_club_payment_l2152_215287

theorem math_club_payment (B : ℕ) : 
  B < 10 →  -- Ensure B is a single digit
  (100 + 10 * B + 8) % 9 = 0 →  -- The number 1B8 is divisible by 9
  B = 0 := by
sorry

end NUMINAMATH_CALUDE_math_club_payment_l2152_215287


namespace NUMINAMATH_CALUDE_meaningful_set_equiv_range_expression_meaningful_iff_in_set_l2152_215257

-- Define the set of real numbers for which the expression is meaningful
def MeaningfulSet : Set ℝ :=
  {x : ℝ | x ≥ -2/3 ∧ x ≠ 0}

-- Theorem stating that the MeaningfulSet is equivalent to the given range
theorem meaningful_set_equiv_range :
  MeaningfulSet = Set.Icc (-2/3) 0 ∪ Set.Ioi 0 :=
sorry

-- Theorem proving that the expression is meaningful if and only if x is in MeaningfulSet
theorem expression_meaningful_iff_in_set (x : ℝ) :
  (3 * x + 2 ≥ 0 ∧ x ≠ 0) ↔ x ∈ MeaningfulSet :=
sorry

end NUMINAMATH_CALUDE_meaningful_set_equiv_range_expression_meaningful_iff_in_set_l2152_215257


namespace NUMINAMATH_CALUDE_cylinder_height_ratio_l2152_215228

/-- 
Given two cylinders where:
- The first cylinder has height h and is 7/8 full of water
- The second cylinder has a radius 25% larger than the first
- All water from the first cylinder fills 3/5 of the second cylinder
Prove that the height of the second cylinder is 14/15 of h
-/
theorem cylinder_height_ratio (h : ℝ) (h' : ℝ) : 
  (7/8 : ℝ) * π * r^2 * h = (3/5 : ℝ) * π * (1.25 * r)^2 * h' → 
  h' = (14/15 : ℝ) * h :=
by sorry

end NUMINAMATH_CALUDE_cylinder_height_ratio_l2152_215228


namespace NUMINAMATH_CALUDE_sin_squared_sum_l2152_215212

theorem sin_squared_sum (α β : ℝ) 
  (h : Real.arcsin (Real.sin α + Real.sin β) + Real.arcsin (Real.sin α - Real.sin β) = π / 2) : 
  Real.sin α ^ 2 + Real.sin β ^ 2 = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_sin_squared_sum_l2152_215212


namespace NUMINAMATH_CALUDE_profit_percentage_l2152_215276

theorem profit_percentage (cost_price selling_price : ℚ) : 
  cost_price = 32 → 
  selling_price = 56 → 
  (selling_price - cost_price) / cost_price * 100 = 75 := by
  sorry

end NUMINAMATH_CALUDE_profit_percentage_l2152_215276


namespace NUMINAMATH_CALUDE_equation_solutions_l2152_215251

/-- The equation x^2 + xy + y^2 + x + y - 5 = 0 has exactly three integer solutions. -/
theorem equation_solutions :
  ∃! (S : Set (ℤ × ℤ)), S = {(1, 1), (1, -3), (-3, 1)} ∧
  ∀ (x y : ℤ), (x, y) ∈ S ↔ x^2 + x*y + y^2 + x + y - 5 = 0 := by
  sorry

end NUMINAMATH_CALUDE_equation_solutions_l2152_215251


namespace NUMINAMATH_CALUDE_tomatoes_left_l2152_215247

theorem tomatoes_left (initial : ℕ) (picked_day1 : ℕ) (picked_day2 : ℕ) 
  (h1 : initial = 171) 
  (h2 : picked_day1 = 134) 
  (h3 : picked_day2 = 30) : 
  initial - picked_day1 - picked_day2 = 7 := by
  sorry

end NUMINAMATH_CALUDE_tomatoes_left_l2152_215247


namespace NUMINAMATH_CALUDE_quadratic_inequality_solutions_l2152_215253

theorem quadratic_inequality_solutions (b : ℤ) : 
  (∃! (s : Finset ℤ), s.card = 3 ∧ ∀ x ∈ s, x^2 + b*x + 10 ≤ 0) ↔ 
  b = 7 ∨ b = -7 :=
sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solutions_l2152_215253


namespace NUMINAMATH_CALUDE_kevins_initial_cards_l2152_215268

theorem kevins_initial_cards (found_cards end_cards : ℕ) 
  (h1 : found_cards = 47) 
  (h2 : end_cards = 54) : 
  end_cards - found_cards = 7 := by
  sorry

end NUMINAMATH_CALUDE_kevins_initial_cards_l2152_215268


namespace NUMINAMATH_CALUDE_carriage_hourly_rate_l2152_215299

/-- Calculates the hourly rate for a carriage given the journey details and costs. -/
theorem carriage_hourly_rate
  (distance : ℝ)
  (speed : ℝ)
  (flat_fee : ℝ)
  (total_cost : ℝ)
  (h1 : distance = 20)
  (h2 : speed = 10)
  (h3 : flat_fee = 20)
  (h4 : total_cost = 80) :
  (total_cost - flat_fee) / (distance / speed) = 30 := by
  sorry

#check carriage_hourly_rate

end NUMINAMATH_CALUDE_carriage_hourly_rate_l2152_215299


namespace NUMINAMATH_CALUDE_probability_sum_25_is_7_200_l2152_215204

-- Define the structure of a die
structure Die :=
  (faces : Finset ℕ)
  (blank_face : Bool)
  (fair : Bool)

-- Define the two dice
def die1 : Die :=
  { faces := Finset.range 20 \ {20},
    blank_face := true,
    fair := true }

def die2 : Die :=
  { faces := (Finset.range 21 \ {0, 8}),
    blank_face := true,
    fair := true }

-- Define the function to calculate the probability
def probability_sum_25 (d1 d2 : Die) : ℚ :=
  let total_outcomes := 20 * 20
  let valid_combinations := 14
  valid_combinations / total_outcomes

-- State the theorem
theorem probability_sum_25_is_7_200 :
  probability_sum_25 die1 die2 = 7 / 200 := by
  sorry

end NUMINAMATH_CALUDE_probability_sum_25_is_7_200_l2152_215204


namespace NUMINAMATH_CALUDE_no_integer_roots_l2152_215202

theorem no_integer_roots : ∀ (x : ℤ), x^3 - 3*x^2 - 10*x + 20 ≠ 0 := by
  sorry

end NUMINAMATH_CALUDE_no_integer_roots_l2152_215202


namespace NUMINAMATH_CALUDE_reflect_y_of_neg_five_two_l2152_215234

/-- Reflects a point across the y-axis -/
def reflect_y (p : ℝ × ℝ) : ℝ × ℝ :=
  (-p.1, p.2)

theorem reflect_y_of_neg_five_two :
  reflect_y (-5, 2) = (5, 2) := by sorry

end NUMINAMATH_CALUDE_reflect_y_of_neg_five_two_l2152_215234


namespace NUMINAMATH_CALUDE_correct_percentage_calculation_l2152_215296

theorem correct_percentage_calculation (x : ℝ) (h : x > 0) :
  let total_problems := 7 * x
  let missed_problems := 2 * x
  let correct_problems := total_problems - missed_problems
  (correct_problems / total_problems) * 100 = (5 / 7) * 100 := by
sorry

end NUMINAMATH_CALUDE_correct_percentage_calculation_l2152_215296


namespace NUMINAMATH_CALUDE_f_composition_equals_negative_262144_l2152_215226

-- Define the complex function f
noncomputable def f (z : ℂ) : ℂ :=
  if ¬(z.re = 0 ∧ z.im = 0) then z^2
  else if 0 < z.re then -z^2
  else z^3

-- State the theorem
theorem f_composition_equals_negative_262144 :
  f (f (f (f (1 + I)))) = -262144 := by
  sorry

end NUMINAMATH_CALUDE_f_composition_equals_negative_262144_l2152_215226


namespace NUMINAMATH_CALUDE_intersection_complement_theorem_l2152_215214

open Set

def U : Set ℝ := univ

def M : Set ℝ := {x | -1 ≤ x ∧ x ≤ 1}

def N : Set ℝ := {x | x ≥ 0}

theorem intersection_complement_theorem :
  M ∩ (U \ N) = {x : ℝ | -1 ≤ x ∧ x < 0} := by sorry

end NUMINAMATH_CALUDE_intersection_complement_theorem_l2152_215214


namespace NUMINAMATH_CALUDE_equal_roots_quadratic_l2152_215233

theorem equal_roots_quadratic (m : ℝ) : 
  (∃ x : ℝ, x^2 - 4*x + m = 0 ∧ 
   ∀ y : ℝ, y^2 - 4*y + m = 0 → y = x) → 
  m = 4 := by
  sorry

end NUMINAMATH_CALUDE_equal_roots_quadratic_l2152_215233


namespace NUMINAMATH_CALUDE_suraya_vs_mia_l2152_215254

/-- The number of apples picked by each person -/
structure ApplePickers where
  kayla : ℕ
  caleb : ℕ
  suraya : ℕ
  mia : ℕ

/-- The conditions of the apple picking scenario -/
def apple_picking_scenario (a : ApplePickers) : Prop :=
  a.kayla = 20 ∧
  a.caleb = a.kayla / 2 - 5 ∧
  a.suraya = 3 * a.caleb ∧
  a.mia = 2 * a.caleb

/-- The theorem stating that Suraya picked 5 more apples than Mia -/
theorem suraya_vs_mia (a : ApplePickers) 
  (h : apple_picking_scenario a) : a.suraya - a.mia = 5 := by
  sorry

end NUMINAMATH_CALUDE_suraya_vs_mia_l2152_215254


namespace NUMINAMATH_CALUDE_arithmetic_sequence_general_term_l2152_215229

/-- Given an arithmetic sequence {a_n} where n ∈ ℕ+, if a_n + a_{n+2} = 4n + 6,
    then a_n = 2n + 1 for all n ∈ ℕ+ -/
theorem arithmetic_sequence_general_term
  (a : ℕ+ → ℝ)  -- a is a function from positive naturals to reals
  (h : ∀ n : ℕ+, a n + a (n + 2) = 4 * n + 6) :  -- given condition
  ∀ n : ℕ+, a n = 2 * n + 1 :=  -- conclusion to prove
by sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_general_term_l2152_215229


namespace NUMINAMATH_CALUDE_difference_d_minus_b_l2152_215290

theorem difference_d_minus_b (a b c d : ℕ+) 
  (h1 : a^5 = b^4) 
  (h2 : c^3 = d^2) 
  (h3 : c - a = 19) : 
  d - b = 757 := by
  sorry

end NUMINAMATH_CALUDE_difference_d_minus_b_l2152_215290


namespace NUMINAMATH_CALUDE_damages_cost_l2152_215256

def tire_cost_1 : ℕ := 230
def tire_cost_2 : ℕ := 250
def tire_cost_3 : ℕ := 280
def window_cost_1 : ℕ := 700
def window_cost_2 : ℕ := 800
def window_cost_3 : ℕ := 900

def total_damages : ℕ := 
  2 * tire_cost_1 + 2 * tire_cost_2 + 2 * tire_cost_3 +
  window_cost_1 + window_cost_2 + window_cost_3

theorem damages_cost : total_damages = 3920 := by
  sorry

end NUMINAMATH_CALUDE_damages_cost_l2152_215256


namespace NUMINAMATH_CALUDE_chess_tournament_games_l2152_215264

/-- The number of games in a chess tournament --/
def num_games (n : ℕ) : ℕ :=
  3 * (n * (n - 1) / 2)

/-- Theorem: In a chess tournament with 35 players, where each player plays
    three times with every other player, the total number of games is 1785 --/
theorem chess_tournament_games :
  num_games 35 = 1785 := by
  sorry


end NUMINAMATH_CALUDE_chess_tournament_games_l2152_215264


namespace NUMINAMATH_CALUDE_three_digit_perfect_cube_divisible_by_16_l2152_215206

theorem three_digit_perfect_cube_divisible_by_16 : 
  ∃! n : ℕ, 100 ≤ n ∧ n ≤ 999 ∧ ∃ m : ℕ, n = m^3 ∧ 16 ∣ n :=
by sorry

end NUMINAMATH_CALUDE_three_digit_perfect_cube_divisible_by_16_l2152_215206


namespace NUMINAMATH_CALUDE_bill_score_l2152_215288

theorem bill_score (john sue ella bill : ℕ) 
  (h1 : bill = john + 20)
  (h2 : bill * 2 = sue)
  (h3 : ella = bill + john - 10)
  (h4 : bill + john + sue + ella = 250) : 
  bill = 50 := by
sorry

end NUMINAMATH_CALUDE_bill_score_l2152_215288


namespace NUMINAMATH_CALUDE_sum_of_divisors_143_l2152_215238

/-- The sum of all positive integer divisors of 143 is 168 -/
theorem sum_of_divisors_143 : (Finset.filter (· ∣ 143) (Finset.range 144)).sum id = 168 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_divisors_143_l2152_215238


namespace NUMINAMATH_CALUDE_problem_proof_l2152_215242

theorem problem_proof : (-1)^2023 - (-1/4)^0 + 2 * Real.cos (π/3) = -1 := by
  sorry

end NUMINAMATH_CALUDE_problem_proof_l2152_215242


namespace NUMINAMATH_CALUDE_sequence_sum_formula_l2152_215265

theorem sequence_sum_formula (a : ℕ → ℕ) (S : ℕ → ℕ) :
  (∀ n : ℕ, S n = 2 * a n - 2) →
  ∀ n : ℕ, a n = 2^n :=
by sorry

end NUMINAMATH_CALUDE_sequence_sum_formula_l2152_215265


namespace NUMINAMATH_CALUDE_max_diagonals_same_length_l2152_215262

/-- The number of sides in the regular polygon -/
def n : ℕ := 1000

/-- The number of different diagonal lengths in a regular n-gon -/
def num_diagonal_lengths (n : ℕ) : ℕ := n / 2

/-- The total number of diagonals in a regular n-gon -/
def total_diagonals (n : ℕ) : ℕ := n * (n - 3) / 2

/-- The number of diagonals of each length in a regular n-gon -/
def diagonals_per_length (n : ℕ) : ℕ := n

/-- Theorem: The maximum number of diagonals that can be selected in a regular 1000-gon 
    such that among any three of the chosen diagonals, at least two have the same length is 2000 -/
theorem max_diagonals_same_length : 
  ∃ (k : ℕ), k = 2000 ∧ 
  k ≤ total_diagonals n ∧
  k = 2 * diagonals_per_length n ∧
  ∀ (m : ℕ), m > k → ¬(∀ (a b c : ℕ), a < m ∧ b < m ∧ c < m → a = b ∨ b = c ∨ a = c) :=
sorry

end NUMINAMATH_CALUDE_max_diagonals_same_length_l2152_215262


namespace NUMINAMATH_CALUDE_inequality_system_solution_l2152_215219

theorem inequality_system_solution (x : ℝ) :
  (x > -6 - 2*x ∧ x ≤ (3 + x) / 4) ↔ (-2 < x ∧ x ≤ 1) := by sorry

end NUMINAMATH_CALUDE_inequality_system_solution_l2152_215219


namespace NUMINAMATH_CALUDE_only_zero_point_eight_greater_than_zero_point_seven_l2152_215232

theorem only_zero_point_eight_greater_than_zero_point_seven :
  let numbers : List ℝ := [0.07, -0.41, 0.8, 0.35, -0.9]
  ∀ x ∈ numbers, x > 0.7 ↔ x = 0.8 := by
  sorry

end NUMINAMATH_CALUDE_only_zero_point_eight_greater_than_zero_point_seven_l2152_215232


namespace NUMINAMATH_CALUDE_bridge_length_l2152_215217

/-- The length of a bridge given train specifications and crossing time -/
theorem bridge_length (train_length : Real) (train_speed_kmh : Real) (crossing_time : Real) :
  train_length = 170 →
  train_speed_kmh = 45 →
  crossing_time = 30 →
  (train_speed_kmh * 1000 / 3600 * crossing_time) - train_length = 205 := by
  sorry

end NUMINAMATH_CALUDE_bridge_length_l2152_215217


namespace NUMINAMATH_CALUDE_multiples_count_l2152_215248

def count_multiples (n : ℕ) : ℕ := 
  (n.div 3 + n.div 4 - n.div 12) - (n.div 15 + n.div 20 - n.div 60)

theorem multiples_count : count_multiples 2010 = 804 := by sorry

end NUMINAMATH_CALUDE_multiples_count_l2152_215248


namespace NUMINAMATH_CALUDE_gcd_lcm_product_90_150_l2152_215279

theorem gcd_lcm_product_90_150 : Nat.gcd 90 150 * Nat.lcm 90 150 = 13500 := by
  sorry

end NUMINAMATH_CALUDE_gcd_lcm_product_90_150_l2152_215279


namespace NUMINAMATH_CALUDE_cone_generatrix_length_l2152_215278

/-- Given a cone with base radius √2 and lateral surface that unfolds into a semicircle,
    the length of the generatrix is 2√2. -/
theorem cone_generatrix_length (r : ℝ) (l : ℝ) : 
  r = Real.sqrt 2 →  -- Base radius is √2
  2 * Real.pi * r = Real.pi * l →  -- Lateral surface unfolds into a semicircle
  l = 2 * Real.sqrt 2 :=  -- Length of generatrix is 2√2
by sorry

end NUMINAMATH_CALUDE_cone_generatrix_length_l2152_215278


namespace NUMINAMATH_CALUDE_smallest_prime_average_l2152_215244

-- Define a function to check if a number is prime
def isPrime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m > 1 → m < n → ¬(n % m = 0)

-- Define a function to check if a list contains five different prime numbers
def isFiveDifferentPrimes (list : List ℕ) : Prop :=
  list.length = 5 ∧ list.Nodup ∧ ∀ n ∈ list, isPrime n

-- Define a function to calculate the average of a list of numbers
def average (list : List ℕ) : ℚ :=
  (list.sum : ℚ) / list.length

theorem smallest_prime_average :
  ∀ list : List ℕ, isFiveDifferentPrimes list → (average list).isInt → average list ≥ 6 :=
sorry

end NUMINAMATH_CALUDE_smallest_prime_average_l2152_215244


namespace NUMINAMATH_CALUDE_average_change_after_removal_l2152_215231

def average_after_removal (n : ℕ) (initial_avg : ℚ) (removed1 removed2 : ℚ) : ℚ :=
  ((n : ℚ) * initial_avg - removed1 - removed2) / ((n - 2) : ℚ)

theorem average_change_after_removal :
  average_after_removal 50 38 45 55 = 37.5 := by
  sorry

end NUMINAMATH_CALUDE_average_change_after_removal_l2152_215231


namespace NUMINAMATH_CALUDE_damage_cost_is_1450_l2152_215203

/-- Calculates the total cost of damages caused by Jack --/
def total_damage_cost (num_tires : ℕ) (cost_per_tire : ℕ) (window_cost : ℕ) : ℕ :=
  num_tires * cost_per_tire + window_cost

/-- Proves that the total cost of damages is $1450 --/
theorem damage_cost_is_1450 :
  total_damage_cost 3 250 700 = 1450 :=
by sorry

end NUMINAMATH_CALUDE_damage_cost_is_1450_l2152_215203


namespace NUMINAMATH_CALUDE_orchestra_size_l2152_215260

def percussion_count : ℕ := 3
def brass_count : ℕ := 5 + 4 + 2 + 2
def strings_count : ℕ := 7 + 5 + 4 + 2
def woodwinds_count : ℕ := 3 + 4 + 2 + 1
def keyboards_harp_count : ℕ := 1 + 1
def conductor_count : ℕ := 1

theorem orchestra_size :
  percussion_count + brass_count + strings_count + woodwinds_count + keyboards_harp_count + conductor_count = 47 := by
  sorry

end NUMINAMATH_CALUDE_orchestra_size_l2152_215260


namespace NUMINAMATH_CALUDE_binomial_coeff_same_binomial_coeff_600_l2152_215283

theorem binomial_coeff_same (n : ℕ) : Nat.choose n n = 1 := by sorry

theorem binomial_coeff_600 : Nat.choose 600 600 = 1 := by sorry

end NUMINAMATH_CALUDE_binomial_coeff_same_binomial_coeff_600_l2152_215283


namespace NUMINAMATH_CALUDE_symmetric_point_proof_l2152_215282

/-- Given a point (0, 2) and a line x + y - 1 = 0, prove that (-1, 1) is the symmetric point --/
theorem symmetric_point_proof (P : ℝ × ℝ) (P' : ℝ × ℝ) (l : ℝ → ℝ → Prop) :
  P = (0, 2) →
  (∀ x y, l x y ↔ x + y - 1 = 0) →
  P' = (-1, 1) →
  (∀ x y, l ((P.1 + x) / 2) ((P.2 + y) / 2) ↔ l x y) →
  (P'.1 - P.1) * (P'.1 - P.1) + (P'.2 - P.2) * (P'.2 - P.2) =
    ((0 : ℝ) - P.1) * ((0 : ℝ) - P.1) + ((0 : ℝ) - P.2) * ((0 : ℝ) - P.2) :=
by sorry


end NUMINAMATH_CALUDE_symmetric_point_proof_l2152_215282


namespace NUMINAMATH_CALUDE_probability_of_rerolling_two_is_one_over_144_l2152_215258

/-- Represents the outcome of rolling a single die -/
inductive DieOutcome
| One | Two | Three | Four | Five | Six

/-- Represents the state of a die (original or rerolled) -/
inductive DieState
| Original (outcome : DieOutcome)
| Rerolled (original : DieOutcome) (new : DieOutcome)

/-- Represents the game state after Jason's decision -/
structure GameState :=
(dice : Fin 3 → DieState)
(rerolledCount : Nat)

/-- Determines if a game state is winning -/
def isWinningState (state : GameState) : Bool :=
  sorry

/-- Calculates the probability of a given game state -/
def probabilityOfState (state : GameState) : ℚ :=
  sorry

/-- Calculates the probability of Jason choosing to reroll exactly two dice -/
def probabilityOfRerollingTwo : ℚ :=
  sorry

theorem probability_of_rerolling_two_is_one_over_144 :
  probabilityOfRerollingTwo = 1 / 144 :=
sorry

end NUMINAMATH_CALUDE_probability_of_rerolling_two_is_one_over_144_l2152_215258


namespace NUMINAMATH_CALUDE_P_greater_than_Q_l2152_215267

theorem P_greater_than_Q (a : ℝ) (h : a > -38) :
  Real.sqrt (a + 40) - Real.sqrt (a + 41) > Real.sqrt (a + 38) - Real.sqrt (a + 39) := by
sorry

end NUMINAMATH_CALUDE_P_greater_than_Q_l2152_215267
