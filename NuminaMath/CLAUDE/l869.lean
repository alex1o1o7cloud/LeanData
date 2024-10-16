import Mathlib

namespace NUMINAMATH_CALUDE_tickets_per_candy_l869_86948

def whack_a_mole_tickets : ℕ := 2
def skee_ball_tickets : ℕ := 13
def candies_bought : ℕ := 5

def total_tickets : ℕ := whack_a_mole_tickets + skee_ball_tickets

theorem tickets_per_candy : total_tickets / candies_bought = 3 := by
  sorry

end NUMINAMATH_CALUDE_tickets_per_candy_l869_86948


namespace NUMINAMATH_CALUDE_fraction_to_decimal_l869_86928

theorem fraction_to_decimal : (7 : ℚ) / 16 = 0.4375 := by sorry

end NUMINAMATH_CALUDE_fraction_to_decimal_l869_86928


namespace NUMINAMATH_CALUDE_four_percent_of_y_is_sixteen_l869_86962

theorem four_percent_of_y_is_sixteen (y : ℝ) (h1 : y > 0) (h2 : 0.04 * y = 16) : y = 400 := by
  sorry

end NUMINAMATH_CALUDE_four_percent_of_y_is_sixteen_l869_86962


namespace NUMINAMATH_CALUDE_angle_sum_proof_l869_86905

theorem angle_sum_proof (x : ℝ) : 
  (6*x + 3*x + 4*x + 2*x = 360) → x = 24 := by
  sorry

end NUMINAMATH_CALUDE_angle_sum_proof_l869_86905


namespace NUMINAMATH_CALUDE_sqrt_inequality_l869_86972

theorem sqrt_inequality : Real.sqrt 2 + Real.sqrt 7 < Real.sqrt 3 + Real.sqrt 6 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_inequality_l869_86972


namespace NUMINAMATH_CALUDE_C₁_intersects_C₂_max_value_on_C₂_l869_86926

-- Define the curves C₁ and C₂
def C₁ (x y : ℝ) : Prop := x + y - 1 = 0
def C₂ (x y : ℝ) : Prop := (x - Real.sqrt 2 / 2)^2 + (y + Real.sqrt 2 / 2)^2 = 1

-- Define a point M on C₂
structure PointOnC₂ where
  x : ℝ
  y : ℝ
  on_C₂ : C₂ x y

-- Theorem 1: C₁ intersects C₂
theorem C₁_intersects_C₂ : ∃ (x y : ℝ), C₁ x y ∧ C₂ x y :=
sorry

-- Theorem 2: Maximum value of 2x + y for points on C₂
theorem max_value_on_C₂ :
  ∃ (max : ℝ), max = Real.sqrt 2 / 2 + Real.sqrt 5 ∧
  ∀ (M : PointOnC₂), 2 * M.x + M.y ≤ max :=
sorry

end NUMINAMATH_CALUDE_C₁_intersects_C₂_max_value_on_C₂_l869_86926


namespace NUMINAMATH_CALUDE_final_student_count_l869_86940

/-- Represents the arrangement of students in the photo. -/
structure StudentArrangement where
  rows : ℕ
  columns : ℕ

/-- The initial arrangement of students before any changes. -/
def initial_arrangement : StudentArrangement := { rows := 0, columns := 0 }

/-- The arrangement after moving one student from each row. -/
def first_adjustment (a : StudentArrangement) : StudentArrangement :=
  { rows := a.rows + 1, columns := a.columns - 1 }

/-- The arrangement after moving a second student from each row. -/
def second_adjustment (a : StudentArrangement) : StudentArrangement :=
  { rows := a.rows + 1, columns := a.columns - 1 }

/-- Calculates the total number of students in the arrangement. -/
def total_students (a : StudentArrangement) : ℕ := a.rows * a.columns

/-- The theorem stating the final number of students in the photo. -/
theorem final_student_count :
  ∃ (a : StudentArrangement),
    (first_adjustment a).columns = (first_adjustment a).rows + 4 ∧
    (second_adjustment (first_adjustment a)).columns = (second_adjustment (first_adjustment a)).rows ∧
    total_students (second_adjustment (first_adjustment a)) = 24 :=
  sorry

end NUMINAMATH_CALUDE_final_student_count_l869_86940


namespace NUMINAMATH_CALUDE_fourth_person_height_l869_86979

theorem fourth_person_height :
  ∀ (h₁ h₂ h₃ h₄ : ℝ),
  h₁ < h₂ ∧ h₂ < h₃ ∧ h₃ < h₄ →
  h₂ - h₁ = 2 →
  h₃ - h₂ = 2 →
  h₄ - h₃ = 6 →
  (h₁ + h₂ + h₃ + h₄) / 4 = 79 →
  h₄ = 85 :=
by
  sorry

end NUMINAMATH_CALUDE_fourth_person_height_l869_86979


namespace NUMINAMATH_CALUDE_ann_oatmeal_raisin_cookies_l869_86980

/-- The number of dozens of oatmeal raisin cookies Ann baked -/
def oatmeal_raisin_dozens : ℝ := sorry

/-- The number of dozens of sugar cookies Ann baked -/
def sugar_dozens : ℝ := 2

/-- The number of dozens of chocolate chip cookies Ann baked -/
def chocolate_chip_dozens : ℝ := 4

/-- The number of dozens of oatmeal raisin cookies Ann gave away -/
def oatmeal_raisin_given : ℝ := 2

/-- The number of dozens of sugar cookies Ann gave away -/
def sugar_given : ℝ := 1.5

/-- The number of dozens of chocolate chip cookies Ann gave away -/
def chocolate_chip_given : ℝ := 2.5

/-- The number of dozens of cookies Ann kept -/
def kept_dozens : ℝ := 3

theorem ann_oatmeal_raisin_cookies :
  oatmeal_raisin_dozens = 3 :=
by sorry

end NUMINAMATH_CALUDE_ann_oatmeal_raisin_cookies_l869_86980


namespace NUMINAMATH_CALUDE_log_sum_property_l869_86938

theorem log_sum_property (a b : ℝ) (ha : a > 1) (hb : b > 1) 
  (h : Real.log (a + b) = Real.log a + Real.log b) : 
  Real.log (a - 1) + Real.log (b - 1) = 0 := by
  sorry

end NUMINAMATH_CALUDE_log_sum_property_l869_86938


namespace NUMINAMATH_CALUDE_ones_digit_of_largest_power_of_2_dividing_32_factorial_l869_86902

/-- The largest power of 2 that divides n! -/
def largestPowerOf2DividingFactorial (n : ℕ) : ℕ :=
  (n / 2) + (n / 4) + (n / 8) + (n / 16) + (n / 32)

/-- The ones digit of a natural number -/
def onesDigit (n : ℕ) : ℕ :=
  n % 10

theorem ones_digit_of_largest_power_of_2_dividing_32_factorial :
  onesDigit (2^(largestPowerOf2DividingFactorial 32)) = 8 :=
sorry

end NUMINAMATH_CALUDE_ones_digit_of_largest_power_of_2_dividing_32_factorial_l869_86902


namespace NUMINAMATH_CALUDE_train_passing_jogger_time_l869_86908

/-- Time for a train to pass a jogger given their speeds and initial positions -/
theorem train_passing_jogger_time
  (jogger_speed : ℝ)
  (train_speed : ℝ)
  (train_length : ℝ)
  (initial_distance : ℝ)
  (h1 : jogger_speed = 9)
  (h2 : train_speed = 45)
  (h3 : train_length = 120)
  (h4 : initial_distance = 240)
  : (initial_distance + train_length) / (train_speed - jogger_speed) * (3600 / 1000) = 36 :=
by
  sorry

#check train_passing_jogger_time

end NUMINAMATH_CALUDE_train_passing_jogger_time_l869_86908


namespace NUMINAMATH_CALUDE_solution_equation1_solution_equation2_l869_86943

-- Define the quadratic equations
def equation1 (x : ℝ) : Prop := x^2 + 2*x - 3 = 0
def equation2 (x : ℝ) : Prop := 2*x^2 + 4*x - 3 = 0

-- Theorem for the first equation
theorem solution_equation1 : 
  (∃ x₁ x₂ : ℝ, x₁ = 1 ∧ x₂ = -3 ∧ equation1 x₁ ∧ equation1 x₂) ∧
  (∀ x : ℝ, equation1 x → x = 1 ∨ x = -3) :=
sorry

-- Theorem for the second equation
theorem solution_equation2 : 
  (∃ x₁ x₂ : ℝ, x₁ = (-2 + Real.sqrt 10) / 2 ∧ x₂ = (-2 - Real.sqrt 10) / 2 ∧ equation2 x₁ ∧ equation2 x₂) ∧
  (∀ x : ℝ, equation2 x → x = (-2 + Real.sqrt 10) / 2 ∨ x = (-2 - Real.sqrt 10) / 2) :=
sorry

end NUMINAMATH_CALUDE_solution_equation1_solution_equation2_l869_86943


namespace NUMINAMATH_CALUDE_right_triangle_segment_ratio_l869_86945

theorem right_triangle_segment_ratio :
  ∀ (a b c r s : ℝ),
  a > 0 → b > 0 → c > 0 → r > 0 → s > 0 →
  a^2 + b^2 = c^2 →  -- Right triangle condition
  a / b = 2 / 5 →    -- Given ratio of sides
  r + s = c →        -- Perpendicular divides hypotenuse
  r * c = a^2 →      -- Geometric mean theorem for r
  s * c = b^2 →      -- Geometric mean theorem for s
  r / s = 4 / 25 := by
sorry

end NUMINAMATH_CALUDE_right_triangle_segment_ratio_l869_86945


namespace NUMINAMATH_CALUDE_absolute_value_inequality_solution_set_l869_86996

theorem absolute_value_inequality_solution_set :
  {x : ℝ | |2*x - 1| ≤ 3} = {x : ℝ | -1 ≤ x ∧ x ≤ 2} :=
by sorry

end NUMINAMATH_CALUDE_absolute_value_inequality_solution_set_l869_86996


namespace NUMINAMATH_CALUDE_inverse_of_A_l869_86931

def A : Matrix (Fin 2) (Fin 2) ℚ := !![5, -3; 3, -2]

theorem inverse_of_A :
  let A_inv : Matrix (Fin 2) (Fin 2) ℚ := !![2, -3; 3, -5]
  A * A_inv = 1 ∧ A_inv * A = 1 := by sorry

end NUMINAMATH_CALUDE_inverse_of_A_l869_86931


namespace NUMINAMATH_CALUDE_infinite_geometric_series_first_term_l869_86933

theorem infinite_geometric_series_first_term 
  (r : ℚ) (S : ℚ) (h1 : r = -1/3) (h2 : S = 9) :
  let a := S * (1 - r)
  a = 12 := by sorry

end NUMINAMATH_CALUDE_infinite_geometric_series_first_term_l869_86933


namespace NUMINAMATH_CALUDE_count_400000_to_500000_by_50_l869_86967

def count_sequence (start : ℕ) (increment : ℕ) (end_value : ℕ) : ℕ :=
  (end_value - start) / increment + 1

theorem count_400000_to_500000_by_50 :
  count_sequence 400000 50 500000 = 2000 := by
  sorry

end NUMINAMATH_CALUDE_count_400000_to_500000_by_50_l869_86967


namespace NUMINAMATH_CALUDE_range_of_a_l869_86920

theorem range_of_a (a : ℝ) : 
  (∀ x : ℝ, (2*x - 1) / (x - 1) ≤ 0 → x^2 - (2*a + 1)*x + a*(a + 1) < 0) ∧ 
  (∃ x : ℝ, x^2 - (2*a + 1)*x + a*(a + 1) < 0 ∧ ¬((2*x - 1) / (x - 1) ≤ 0)) →
  0 ≤ a ∧ a < 1/2 :=
sorry

end NUMINAMATH_CALUDE_range_of_a_l869_86920


namespace NUMINAMATH_CALUDE_garden_width_l869_86918

theorem garden_width (playground_side : ℕ) (garden_length : ℕ) (total_fencing : ℕ) :
  playground_side = 27 →
  garden_length = 12 →
  total_fencing = 150 →
  4 * playground_side + 2 * garden_length + 2 * (total_fencing - 4 * playground_side - 2 * garden_length) / 2 = 150 →
  (total_fencing - 4 * playground_side - 2 * garden_length) / 2 = 9 := by
  sorry

#check garden_width

end NUMINAMATH_CALUDE_garden_width_l869_86918


namespace NUMINAMATH_CALUDE_jane_morning_reading_l869_86917

/-- The number of pages Jane reads in the morning -/
def morning_pages : ℕ := 5

/-- The number of pages Jane reads in the evening -/
def evening_pages : ℕ := 10

/-- The number of days in a week -/
def days_in_week : ℕ := 7

/-- The total number of pages Jane reads in a week -/
def total_pages : ℕ := 105

/-- Theorem stating that Jane reads 5 pages in the morning -/
theorem jane_morning_reading :
  morning_pages = 5 ∧
  evening_pages = 10 ∧
  days_in_week = 7 ∧
  total_pages = 105 ∧
  days_in_week * (morning_pages + evening_pages) = total_pages :=
by sorry

end NUMINAMATH_CALUDE_jane_morning_reading_l869_86917


namespace NUMINAMATH_CALUDE_negation_of_implication_l869_86989

theorem negation_of_implication (a b : ℝ) :
  ¬(ab = 0 → a = 0 ∨ b = 0) ↔ (ab = 0 → a ≠ 0 ∧ b ≠ 0) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_implication_l869_86989


namespace NUMINAMATH_CALUDE_total_football_games_l869_86973

/-- The total number of football games in a year, given the number of games
    Keith attended and missed. -/
theorem total_football_games (attended : ℕ) (missed : ℕ) 
  (h1 : attended = 4) (h2 : missed = 4) : 
  attended + missed = 8 := by
  sorry

end NUMINAMATH_CALUDE_total_football_games_l869_86973


namespace NUMINAMATH_CALUDE_least_triangle_area_l869_86983

/-- The solutions of the equation (z+4)^10 = 32 form a convex regular decagon in the complex plane. -/
def is_solution (z : ℂ) : Prop := (z + 4) ^ 10 = 32

/-- The set of all solutions forms a convex regular decagon. -/
def solution_set : Set ℂ := {z | is_solution z}

/-- A point is a vertex of the decagon if it's a solution. -/
def is_vertex (z : ℂ) : Prop := z ∈ solution_set

/-- The area of a triangle formed by three vertices of the decagon. -/
def triangle_area (v1 v2 v3 : ℂ) : ℝ :=
  sorry -- Definition of the area calculation

/-- The theorem stating the least possible area of a triangle formed by three vertices. -/
theorem least_triangle_area :
  ∃ (v1 v2 v3 : ℂ), is_vertex v1 ∧ is_vertex v2 ∧ is_vertex v3 ∧
    (∀ (w1 w2 w3 : ℂ), is_vertex w1 → is_vertex w2 → is_vertex w3 →
      triangle_area v1 v2 v3 ≤ triangle_area w1 w2 w3) ∧
    triangle_area v1 v2 v3 = (2^(2/5) * (Real.sqrt 5 - 1)) / 8 :=
sorry

end NUMINAMATH_CALUDE_least_triangle_area_l869_86983


namespace NUMINAMATH_CALUDE_integer_pair_divisibility_l869_86950

theorem integer_pair_divisibility (m n : ℕ+) : 
  (∃ k : ℤ, (m : ℤ) + (n : ℤ)^2 = k * ((m : ℤ)^2 - (n : ℤ))) ∧ 
  (∃ l : ℤ, (n : ℤ) + (m : ℤ)^2 = l * ((n : ℤ)^2 - (m : ℤ))) ↔ 
  ((m = 2 ∧ n = 1) ∨ (m = 3 ∧ n = 2)) := by
sorry

end NUMINAMATH_CALUDE_integer_pair_divisibility_l869_86950


namespace NUMINAMATH_CALUDE_parabola_point_coordinates_l869_86975

theorem parabola_point_coordinates :
  ∀ (x y : ℝ),
  y^2 = 4*x →                             -- Point P(x, y) lies on the parabola y^2 = 4x
  (x - 1)^2 + y^2 = 100 →                 -- Distance from P to focus F(1, 0) is 10
  x = 9 ∧ (y = 6 ∨ y = -6) :=             -- Conclusion: x = 9 and y = ±6
by
  sorry


end NUMINAMATH_CALUDE_parabola_point_coordinates_l869_86975


namespace NUMINAMATH_CALUDE_polygon_with_135_degree_angles_is_octagon_l869_86946

theorem polygon_with_135_degree_angles_is_octagon :
  ∀ (n : ℕ) (interior_angle : ℝ),
    n ≥ 3 →
    interior_angle = 135 →
    (n - 2) * 180 / n = interior_angle →
    n = 8 :=
by
  sorry

end NUMINAMATH_CALUDE_polygon_with_135_degree_angles_is_octagon_l869_86946


namespace NUMINAMATH_CALUDE_bookstore_new_releases_fraction_l869_86998

/-- Represents a bookstore inventory --/
structure Bookstore where
  total : ℕ
  historicalFiction : ℕ
  historicalFictionNewReleases : ℕ
  otherNewReleases : ℕ

/-- Calculates the fraction of new releases that are historical fiction --/
def newReleasesFraction (store : Bookstore) : ℚ :=
  store.historicalFictionNewReleases / (store.historicalFictionNewReleases + store.otherNewReleases)

theorem bookstore_new_releases_fraction :
  ∀ (store : Bookstore),
    store.total > 0 →
    store.historicalFiction = (2 * store.total) / 5 →
    store.historicalFictionNewReleases = (2 * store.historicalFiction) / 5 →
    store.otherNewReleases = (7 * (store.total - store.historicalFiction)) / 10 →
    newReleasesFraction store = 8 / 29 := by
  sorry

end NUMINAMATH_CALUDE_bookstore_new_releases_fraction_l869_86998


namespace NUMINAMATH_CALUDE_mickey_horses_per_week_l869_86976

/-- The number of days in a week -/
def days_in_week : ℕ := 7

/-- The number of horses Minnie mounts per day -/
def minnie_horses_per_day : ℕ := days_in_week + 3

/-- The number of horses Mickey mounts per day -/
def mickey_horses_per_day : ℕ := 2 * minnie_horses_per_day - 6

/-- Theorem: Mickey mounts 98 horses per week -/
theorem mickey_horses_per_week : mickey_horses_per_day * days_in_week = 98 := by
  sorry

end NUMINAMATH_CALUDE_mickey_horses_per_week_l869_86976


namespace NUMINAMATH_CALUDE_craft_store_optimal_solution_l869_86957

/-- Represents the craft store problem -/
structure CraftStore where
  profit_per_item : ℝ
  cost_50_items : ℝ
  revenue_40_items : ℝ
  initial_daily_sales : ℕ
  sales_increase_per_yuan : ℕ

/-- Theorem stating the optimal solution for the craft store problem -/
theorem craft_store_optimal_solution (store : CraftStore) 
  (h1 : store.profit_per_item = 45)
  (h2 : store.cost_50_items = store.revenue_40_items)
  (h3 : store.initial_daily_sales = 100)
  (h4 : store.sales_increase_per_yuan = 4) :
  ∃ (cost_price marked_price optimal_reduction max_profit : ℝ),
    cost_price = 180 ∧
    marked_price = 225 ∧
    optimal_reduction = 10 ∧
    max_profit = 4900 := by
  sorry

end NUMINAMATH_CALUDE_craft_store_optimal_solution_l869_86957


namespace NUMINAMATH_CALUDE_quadratic_equation_root_and_sum_l869_86909

theorem quadratic_equation_root_and_sum : 
  ∃ (a b c : ℚ), 
    (a = 1 ∧ b = 6 ∧ c = -4) ∧ 
    (a * (Real.sqrt 5 - 3)^2 + b * (Real.sqrt 5 - 3) + c = 0) ∧
    (∀ x y : ℝ, x^2 + 6*x - 4 = 0 ∧ y^2 + 6*y - 4 = 0 → x + y = -6) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equation_root_and_sum_l869_86909


namespace NUMINAMATH_CALUDE_sqrt_difference_abs_plus_two_sqrt_two_l869_86993

theorem sqrt_difference_abs_plus_two_sqrt_two :
  |Real.sqrt 2 - Real.sqrt 3| + 2 * Real.sqrt 2 = Real.sqrt 3 + Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_difference_abs_plus_two_sqrt_two_l869_86993


namespace NUMINAMATH_CALUDE_pythagorean_squares_area_l869_86922

theorem pythagorean_squares_area (a b c : ℝ) (h_right_angle : a^2 + b^2 = c^2)
  (h_sum_areas : a^2 + b^2 + 2*c^2 = 500) : c^2 = 500/3 :=
by sorry

end NUMINAMATH_CALUDE_pythagorean_squares_area_l869_86922


namespace NUMINAMATH_CALUDE_valid_table_exists_l869_86900

/-- Represents a geometric property of a shape -/
inductive Property
| HasAcuteAngle
| HasEqualSides
| Property3
| Property4

/-- Represents a geometric shape -/
inductive Shape
| Triangle1
| Triangle2
| Quadrilateral1
| Quadrilateral2

/-- A function that determines if a shape has a property -/
def hasProperty (s : Shape) (p : Property) : Bool :=
  match s, p with
  | Shape.Triangle1, Property.HasAcuteAngle => true
  | Shape.Triangle1, Property.HasEqualSides => false
  | Shape.Triangle2, Property.HasAcuteAngle => true
  | Shape.Triangle2, Property.HasEqualSides => true
  | Shape.Quadrilateral1, Property.HasAcuteAngle => false
  | Shape.Quadrilateral1, Property.HasEqualSides => false
  | Shape.Quadrilateral2, Property.HasAcuteAngle => true
  | Shape.Quadrilateral2, Property.HasEqualSides => false
  | _, _ => false  -- Default case for other combinations

/-- The main theorem stating the existence of a valid table -/
theorem valid_table_exists : ∃ (p3 p4 : Property),
  p3 ≠ Property.HasAcuteAngle ∧ p3 ≠ Property.HasEqualSides ∧
  p4 ≠ Property.HasAcuteAngle ∧ p4 ≠ Property.HasEqualSides ∧ p3 ≠ p4 ∧
  (∀ s : Shape, (hasProperty s Property.HasAcuteAngle).toNat +
                (hasProperty s Property.HasEqualSides).toNat +
                (hasProperty s p3).toNat +
                (hasProperty s p4).toNat = 3) ∧
  (∀ p : Property, (p = Property.HasAcuteAngle ∨ p = Property.HasEqualSides ∨ p = p3 ∨ p = p4) →
    (hasProperty Shape.Triangle1 p).toNat +
    (hasProperty Shape.Triangle2 p).toNat +
    (hasProperty Shape.Quadrilateral1 p).toNat +
    (hasProperty Shape.Quadrilateral2 p).toNat = 3) :=
by sorry

end NUMINAMATH_CALUDE_valid_table_exists_l869_86900


namespace NUMINAMATH_CALUDE_brother_difference_is_two_l869_86949

/-- The number of Aaron's brothers -/
def aaron_brothers : ℕ := 4

/-- The number of Bennett's brothers -/
def bennett_brothers : ℕ := 6

/-- The difference between twice the number of Aaron's brothers and the number of Bennett's brothers -/
def brother_difference : ℕ := 2 * aaron_brothers - bennett_brothers

/-- Theorem stating that the difference between twice the number of Aaron's brothers
    and the number of Bennett's brothers is 2 -/
theorem brother_difference_is_two : brother_difference = 2 := by
  sorry

end NUMINAMATH_CALUDE_brother_difference_is_two_l869_86949


namespace NUMINAMATH_CALUDE_divisibility_of_f_l869_86944

def f (x : ℕ) : ℕ := x^3 + 17

theorem divisibility_of_f :
  ∀ n : ℕ, n ≥ 2 →
  ∃ x : ℕ, (3^n ∣ f x) ∧ ¬(3^(n+1) ∣ f x) := by
sorry

end NUMINAMATH_CALUDE_divisibility_of_f_l869_86944


namespace NUMINAMATH_CALUDE_stone_counting_135_l869_86974

/-- Represents the stone counting pattern described in the problem -/
def stoneCounting (n : ℕ) : ℕ := 
  let cycle := n % 24
  if cycle ≤ 12 
  then (cycle + 1) / 2 
  else (25 - cycle) / 2

/-- The problem statement -/
theorem stone_counting_135 : stoneCounting 135 = 3 := by
  sorry

end NUMINAMATH_CALUDE_stone_counting_135_l869_86974


namespace NUMINAMATH_CALUDE_decreasing_function_inequality_l869_86906

/-- A function f is decreasing on an open interval (a,b) if for all x, y in (a,b), x < y implies f(x) > f(y) -/
def DecreasingOn (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x y, a < x → x < y → y < b → f x > f y

theorem decreasing_function_inequality (f : ℝ → ℝ) (a : ℝ) 
  (h1 : DecreasingOn f (-1) 1)
  (h2 : f (1 - a) < f (3 * a - 1)) :
  0 < a ∧ a < 1/2 := by
  sorry

end NUMINAMATH_CALUDE_decreasing_function_inequality_l869_86906


namespace NUMINAMATH_CALUDE_inequality_proof_l869_86914

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (habc : a * b * c = 1) :
  Real.sqrt (a^3 / (1 + b * c)) + Real.sqrt (b^3 / (1 + a * c)) + Real.sqrt (c^3 / (1 + a * b)) > 2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l869_86914


namespace NUMINAMATH_CALUDE_triangle_problem_l869_86912

/-- Triangle ABC with sides a, b, c opposite to angles A, B, C respectively -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

theorem triangle_problem (t : Triangle) 
  (h1 : (t.a + t.b) * (Real.sin t.A - Real.sin t.B) = (t.c - t.b) * Real.sin t.C)
  (h2 : 2 * t.c = 3 * t.b)
  (h3 : 1/2 * t.b * t.c * Real.sin t.A = 6 * Real.sqrt 3) :
  t.A = π/3 ∧ t.a = 2 * Real.sqrt (21/3) := by
  sorry

end NUMINAMATH_CALUDE_triangle_problem_l869_86912


namespace NUMINAMATH_CALUDE_sqrt_sum_equals_seventeen_sixths_l869_86951

theorem sqrt_sum_equals_seventeen_sixths : 
  Real.sqrt (9 / 4) + Real.sqrt (16 / 9) = 17 / 6 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_sum_equals_seventeen_sixths_l869_86951


namespace NUMINAMATH_CALUDE_right_triangle_circles_radius_l869_86929

/-- Represents a right triangle with a circle tangent to one side and another circle --/
structure RightTriangleWithCircles where
  -- The length of side AC
  ac : ℝ
  -- The length of side AB
  ab : ℝ
  -- The radius of circle C
  rc : ℝ
  -- The radius of circle A
  ra : ℝ
  -- Circle C is tangent to AB
  c_tangent_ab : True
  -- Circle A and circle C are tangent
  a_tangent_c : True
  -- Angle C is 90 degrees
  angle_c_90 : True

/-- The main theorem --/
theorem right_triangle_circles_radius 
  (t : RightTriangleWithCircles) 
  (h1 : t.ac = 6) 
  (h2 : t.ab = 10) : 
  t.ra = 1.2 ∨ t.ra = 10.8 := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_circles_radius_l869_86929


namespace NUMINAMATH_CALUDE_rope_cutting_l869_86964

theorem rope_cutting (total_length : ℕ) (long_piece_length : ℕ) (num_short_pieces : ℕ) 
  (h1 : total_length = 27)
  (h2 : long_piece_length = 4)
  (h3 : num_short_pieces = 3) :
  ∃ (num_long_pieces : ℕ) (short_piece_length : ℕ),
    num_long_pieces * long_piece_length + num_short_pieces * short_piece_length = total_length ∧
    num_long_pieces = total_length / long_piece_length ∧
    short_piece_length = 1 :=
by
  sorry

end NUMINAMATH_CALUDE_rope_cutting_l869_86964


namespace NUMINAMATH_CALUDE_shelves_needed_l869_86963

theorem shelves_needed (total_books : ℕ) (books_per_shelf : ℕ) (h1 : total_books = 12) (h2 : books_per_shelf = 4) :
  total_books / books_per_shelf = 3 := by
  sorry

end NUMINAMATH_CALUDE_shelves_needed_l869_86963


namespace NUMINAMATH_CALUDE_parabola_axis_of_symmetry_l869_86913

/-- The axis of symmetry of the parabola y = x^2 + 4x - 5 is the line x = -2 -/
theorem parabola_axis_of_symmetry :
  let f : ℝ → ℝ := fun x ↦ x^2 + 4*x - 5
  ∃ (a : ℝ), a = -2 ∧ ∀ (x y : ℝ), f (a + x) = f (a - x) := by
  sorry

end NUMINAMATH_CALUDE_parabola_axis_of_symmetry_l869_86913


namespace NUMINAMATH_CALUDE_vector_operations_l869_86954

def a : ℝ × ℝ := (2, 1)
def b : ℝ × ℝ := (-3, -4)

def is_unit_vector (v : ℝ × ℝ) : Prop := v.1^2 + v.2^2 = 1

def is_perpendicular (v w : ℝ × ℝ) : Prop := v.1 * w.1 + v.2 * w.2 = 0

theorem vector_operations (c : ℝ × ℝ) 
  (h1 : is_unit_vector c) 
  (h2 : is_perpendicular c (a.1 - b.1, a.2 - b.2)) : 
  (2 * a.1 + 3 * b.1, 2 * a.2 + 3 * b.2) = (-5, -10) ∧
  (a.1 - 2 * b.1)^2 + (a.2 - 2 * b.2)^2 = 145 ∧
  (c = (Real.sqrt 2 / 2, -Real.sqrt 2 / 2) ∨ c = (-Real.sqrt 2 / 2, Real.sqrt 2 / 2)) := by
  sorry

end NUMINAMATH_CALUDE_vector_operations_l869_86954


namespace NUMINAMATH_CALUDE_third_divisor_l869_86907

theorem third_divisor (n : ℕ) (h1 : n = 200) 
  (h2 : ∃ k₁ k₂ k₃ k₄ : ℕ, n + 20 = 15 * k₁ ∧ n + 20 = 30 * k₂ ∧ n + 20 = 60 * k₄) 
  (h3 : ∃ x : ℕ, x ≠ 15 ∧ x ≠ 30 ∧ x ≠ 60 ∧ ∃ k : ℕ, n + 20 = x * k) : 
  ∃ x : ℕ, x = 11 ∧ x ≠ 15 ∧ x ≠ 30 ∧ x ≠ 60 ∧ ∃ k : ℕ, n + 20 = x * k :=
sorry

end NUMINAMATH_CALUDE_third_divisor_l869_86907


namespace NUMINAMATH_CALUDE_ninth_term_of_arithmetic_sequence_l869_86927

def arithmetic_sequence (a : ℕ → ℚ) : Prop :=
  ∀ n m : ℕ, a (n + m) - a n = m * (a (n + 1) - a n)

theorem ninth_term_of_arithmetic_sequence
  (a : ℕ → ℚ)
  (h_arithmetic : arithmetic_sequence a)
  (h_first : a 1 = 3/8)
  (h_seventeenth : a 17 = 2/3) :
  a 9 = 25/48 := by
sorry

end NUMINAMATH_CALUDE_ninth_term_of_arithmetic_sequence_l869_86927


namespace NUMINAMATH_CALUDE_anna_baking_trays_l869_86959

/-- The number of cupcakes per tray -/
def cupcakes_per_tray : ℕ := 20

/-- The price of each cupcake in dollars -/
def cupcake_price : ℚ := 2

/-- The fraction of cupcakes sold -/
def fraction_sold : ℚ := 3/5

/-- The total earnings in dollars -/
def total_earnings : ℚ := 96

/-- The number of baking trays Anna used -/
def num_trays : ℕ := 4

theorem anna_baking_trays :
  (cupcakes_per_tray : ℚ) * num_trays * fraction_sold * cupcake_price = total_earnings := by
  sorry

end NUMINAMATH_CALUDE_anna_baking_trays_l869_86959


namespace NUMINAMATH_CALUDE_probability_same_group_l869_86984

/-- The probability that two specific cards (5 and 14) are in the same group
    when drawing 4 cards from 20, where groups are determined by card value. -/
theorem probability_same_group : 
  let total_cards : ℕ := 20
  let cards_drawn : ℕ := 4
  let remaining_cards : ℕ := total_cards - cards_drawn + 2  -- +2 because 5 and 14 are known
  let favorable_outcomes : ℕ := (remaining_cards - 14 + 1) * (remaining_cards - 14) + 
                                (5 - 1) * (5 - 2)
  let total_outcomes : ℕ := remaining_cards * (remaining_cards - 1)
  (favorable_outcomes : ℚ) / total_outcomes = 7 / 51 := by
  sorry

end NUMINAMATH_CALUDE_probability_same_group_l869_86984


namespace NUMINAMATH_CALUDE_find_m_l869_86971

-- Define the inequality
def inequality (x m : ℝ) : Prop := -1/2 * x^2 + 2*x > -m*x

-- Define the solution set
def solution_set (m : ℝ) : Set ℝ := {x : ℝ | 0 < x ∧ x < 2}

-- Theorem statement
theorem find_m : 
  ∀ m : ℝ, (∀ x : ℝ, inequality x m ↔ x ∈ solution_set m) → m = -1 :=
sorry

end NUMINAMATH_CALUDE_find_m_l869_86971


namespace NUMINAMATH_CALUDE_negation_equivalence_l869_86923

theorem negation_equivalence :
  (¬ ∀ x : ℝ, ∃ n : ℕ+, (n : ℝ) ≥ x^2) ↔ (∃ x : ℝ, ∀ n : ℕ+, (n : ℝ) < x^2) := by
  sorry

end NUMINAMATH_CALUDE_negation_equivalence_l869_86923


namespace NUMINAMATH_CALUDE_min_value_implies_a_eq_one_exp_log_sin_positive_l869_86919

noncomputable section

variable (x : ℝ)
variable (a : ℝ)

def f (x : ℝ) : ℝ := Real.log x + a / x - 1

theorem min_value_implies_a_eq_one (h : ∀ x > 0, f x ≥ 0) (h' : ∃ x > 0, f x = 0) : a = 1 :=
sorry

theorem exp_log_sin_positive : ∀ x > 0, Real.exp x + (Real.log x - 1) * Real.sin x > 0 :=
sorry

end NUMINAMATH_CALUDE_min_value_implies_a_eq_one_exp_log_sin_positive_l869_86919


namespace NUMINAMATH_CALUDE_triangle_angle_measure_l869_86981

theorem triangle_angle_measure (A B C : ℝ) (h1 : A + C = 2 * B) (h2 : C - A = 80) 
  (h3 : A + B + C = 180) : C = 100 := by
  sorry

end NUMINAMATH_CALUDE_triangle_angle_measure_l869_86981


namespace NUMINAMATH_CALUDE_quadratic_has_two_real_roots_roots_difference_three_l869_86958

/-- The quadratic equation x^2 - (m-1)x + (m-2) = 0 -/
def quadratic (m : ℝ) (x : ℝ) : ℝ := x^2 - (m-1)*x + (m-2)

/-- The discriminant of the quadratic equation -/
def discriminant (m : ℝ) : ℝ := (m-1)^2 - 4*(m-2)

theorem quadratic_has_two_real_roots (m : ℝ) : 
  ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ quadratic m x₁ = 0 ∧ quadratic m x₂ = 0 :=
sorry

theorem roots_difference_three (m : ℝ) :
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ quadratic m x₁ = 0 ∧ quadratic m x₂ = 0 ∧ |x₁ - x₂| = 3) →
  m = 0 ∨ m = 6 :=
sorry

end NUMINAMATH_CALUDE_quadratic_has_two_real_roots_roots_difference_three_l869_86958


namespace NUMINAMATH_CALUDE_age_ratio_12_years_ago_l869_86904

-- Define Neha's current age and her mother's current age
def neha_age : ℕ := sorry
def mother_age : ℕ := 60

-- Define the relationship between their ages 12 years ago
axiom past_relation : ∃ x : ℚ, mother_age - 12 = x * (neha_age - 12)

-- Define the relationship between their ages 12 years from now
axiom future_relation : mother_age + 12 = 2 * (neha_age + 12)

-- Theorem to prove
theorem age_ratio_12_years_ago : 
  (mother_age - 12) / (neha_age - 12) = 4 := by sorry

end NUMINAMATH_CALUDE_age_ratio_12_years_ago_l869_86904


namespace NUMINAMATH_CALUDE_binomial_coefficient_n_minus_two_l869_86994

theorem binomial_coefficient_n_minus_two (n : ℕ) (h : n > 3) :
  Nat.choose n (n - 2) = n * (n - 1) / 2 := by
  sorry

end NUMINAMATH_CALUDE_binomial_coefficient_n_minus_two_l869_86994


namespace NUMINAMATH_CALUDE_inequality_proof_l869_86941

theorem inequality_proof (a b c d : ℝ) 
  (sum_condition : a + b + c + d = 6)
  (sum_squares_condition : a^2 + b^2 + c^2 + d^2 = 12) :
  36 ≤ 4 * (a^3 + b^3 + c^3 + d^3) - (a^4 + b^4 + c^4 + d^4) ∧ 
  4 * (a^3 + b^3 + c^3 + d^3) - (a^4 + b^4 + c^4 + d^4) ≤ 48 :=
by sorry

end NUMINAMATH_CALUDE_inequality_proof_l869_86941


namespace NUMINAMATH_CALUDE_class_fund_total_l869_86997

theorem class_fund_total (ten_bills : ℕ) (twenty_bills : ℕ) : 
  ten_bills = 2 * twenty_bills →
  twenty_bills = 3 →
  ten_bills * 10 + twenty_bills * 20 = 120 :=
by
  sorry

end NUMINAMATH_CALUDE_class_fund_total_l869_86997


namespace NUMINAMATH_CALUDE_polynomial_division_remainder_l869_86987

theorem polynomial_division_remainder :
  ∃ q : Polynomial ℝ,
  x^5 + 3*x^3 + 1 = (x - 3)^2 * q + (324*x - 488) :=
sorry

end NUMINAMATH_CALUDE_polynomial_division_remainder_l869_86987


namespace NUMINAMATH_CALUDE_percentage_neither_is_twenty_percent_l869_86988

/-- Represents the health survey data for teachers -/
structure HealthSurvey where
  total : ℕ
  high_bp : ℕ
  heart_trouble : ℕ
  both : ℕ

/-- Calculates the percentage of teachers with neither high blood pressure nor heart trouble -/
def percentage_neither (survey : HealthSurvey) : ℚ :=
  let neither := survey.total - (survey.high_bp + survey.heart_trouble - survey.both)
  (neither : ℚ) / survey.total * 100

/-- Theorem stating that the percentage of teachers with neither condition is 20% -/
theorem percentage_neither_is_twenty_percent (survey : HealthSurvey)
  (h_total : survey.total = 150)
  (h_high_bp : survey.high_bp = 90)
  (h_heart_trouble : survey.heart_trouble = 60)
  (h_both : survey.both = 30) :
  percentage_neither survey = 20 := by
  sorry

#eval percentage_neither { total := 150, high_bp := 90, heart_trouble := 60, both := 30 }

end NUMINAMATH_CALUDE_percentage_neither_is_twenty_percent_l869_86988


namespace NUMINAMATH_CALUDE_triangle_area_l869_86937

/-- The area of a triangle with base 12 and height 5 is 30 -/
theorem triangle_area : 
  let base : ℝ := 12
  let height : ℝ := 5
  (1/2 : ℝ) * base * height = 30 := by sorry

end NUMINAMATH_CALUDE_triangle_area_l869_86937


namespace NUMINAMATH_CALUDE_range_of_a_l869_86936

def f (a : ℝ) (x : ℝ) : ℝ := a * x^2 + a * x - 1

theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, f a x < 0) ↔ -4 < a ∧ a ≤ 0 :=
by sorry

end NUMINAMATH_CALUDE_range_of_a_l869_86936


namespace NUMINAMATH_CALUDE_triangle_coordinate_difference_l869_86932

/-- Triangle ABC with vertices A(0,10), B(4,0), C(10,0), and a vertical line
    intersecting AC at R and BC at S. If the area of triangle RSC is 20,
    then the positive difference between the x and y coordinates of R is 4√10 - 10. -/
theorem triangle_coordinate_difference (R : ℝ × ℝ) :
  let A : ℝ × ℝ := (0, 10)
  let B : ℝ × ℝ := (4, 0)
  let C : ℝ × ℝ := (10, 0)
  let S : ℝ × ℝ := (R.1, 0)  -- S has same x-coordinate as R and y-coordinate 0
  -- R is on line AC
  (10 - R.1) / (0 - R.2) = 1 →
  -- RS is vertical (same x-coordinate)
  R.1 = S.1 →
  -- Area of triangle RSC is 20
  abs ((R.1 - 10) * R.2) / 2 = 20 →
  -- The positive difference between x and y coordinates of R
  abs (R.2 - R.1) = 4 * Real.sqrt 10 - 10 :=
by sorry

end NUMINAMATH_CALUDE_triangle_coordinate_difference_l869_86932


namespace NUMINAMATH_CALUDE_isosceles_triangle_sides_l869_86901

/-- Represents the sides of an isosceles triangle -/
structure IsoscelesTriangle where
  base : ℝ
  leg : ℝ

/-- Checks if the given sides form a valid isosceles triangle -/
def is_valid_isosceles (t : IsoscelesTriangle) : Prop :=
  t.base > 0 ∧ t.leg > 0 ∧ t.leg + t.leg > t.base

/-- Calculates the perimeter of an isosceles triangle -/
def perimeter (t : IsoscelesTriangle) : ℝ :=
  t.base + 2 * t.leg

theorem isosceles_triangle_sides (p : ℝ) (s : ℝ) :
  p = 26 ∧ s = 11 →
  ∃ (t1 t2 : IsoscelesTriangle),
    (perimeter t1 = p ∧ (t1.base = s ∨ t1.leg = s) ∧ is_valid_isosceles t1) ∧
    (perimeter t2 = p ∧ (t2.base = s ∨ t2.leg = s) ∧ is_valid_isosceles t2) ∧
    ((t1.base = 11 ∧ t1.leg = 7.5) ∨ (t1.leg = 11 ∧ t1.base = 4)) ∧
    ((t2.base = 11 ∧ t2.leg = 7.5) ∨ (t2.leg = 11 ∧ t2.base = 4)) ∧
    t1 ≠ t2 :=
by sorry

end NUMINAMATH_CALUDE_isosceles_triangle_sides_l869_86901


namespace NUMINAMATH_CALUDE_irrational_sum_equivalence_l869_86992

theorem irrational_sum_equivalence 
  (a b c d : ℝ) 
  (ha : Irrational a) 
  (hb : Irrational b) 
  (hc : Irrational c) 
  (hd : Irrational d) 
  (hab : a + b = 1) :
  (c + d = 1) ↔ 
  (∀ n : ℕ+, ⌊n * a⌋ + ⌊n * b⌋ = ⌊n * c⌋ + ⌊n * d⌋) :=
by sorry

end NUMINAMATH_CALUDE_irrational_sum_equivalence_l869_86992


namespace NUMINAMATH_CALUDE_compute_expression_l869_86978

theorem compute_expression : 20 * (256 / 4 + 64 / 16 + 16 / 64 + 2) = 1405 := by
  sorry

end NUMINAMATH_CALUDE_compute_expression_l869_86978


namespace NUMINAMATH_CALUDE_set_equality_l869_86916

open Set

def U : Set ℝ := univ
def E : Set ℝ := {x | x ≤ -3 ∨ x ≥ 2}
def F : Set ℝ := {x | -1 < x ∧ x < 5}

theorem set_equality : {x : ℝ | -1 < x ∧ x < 2} = (U \ E) ∩ F := by sorry

end NUMINAMATH_CALUDE_set_equality_l869_86916


namespace NUMINAMATH_CALUDE_expression_evaluation_l869_86903

theorem expression_evaluation (x y : ℕ) (h1 : x = 3) (h2 : y = 4) :
  5 * x^y + 6 * y^x = 789 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l869_86903


namespace NUMINAMATH_CALUDE_ralph_tv_watching_hours_l869_86955

/-- The number of hours Ralph watches TV on a weekday -/
def weekday_hours : ℕ := 4

/-- The number of hours Ralph watches TV on a weekend day -/
def weekend_hours : ℕ := 6

/-- The number of weekdays in a week -/
def weekdays : ℕ := 5

/-- The number of weekend days in a week -/
def weekend_days : ℕ := 2

/-- The total number of hours Ralph watches TV in a week -/
def total_hours : ℕ := weekday_hours * weekdays + weekend_hours * weekend_days

theorem ralph_tv_watching_hours :
  total_hours = 32 := by sorry

end NUMINAMATH_CALUDE_ralph_tv_watching_hours_l869_86955


namespace NUMINAMATH_CALUDE_sine_of_angle_between_vectors_l869_86960

/-- Given vectors a and b with an angle θ between them, 
    if a = (2, 1) and 3b + a = (5, 4), then sin θ = √10/10 -/
theorem sine_of_angle_between_vectors (a b : ℝ × ℝ) (θ : ℝ) :
  a = (2, 1) →
  3 • b + a = (5, 4) →
  Real.sin θ = (Real.sqrt 10) / 10 := by
  sorry

end NUMINAMATH_CALUDE_sine_of_angle_between_vectors_l869_86960


namespace NUMINAMATH_CALUDE_problem_1_l869_86991

theorem problem_1 : (-2)^0 + 1 / Real.sqrt 2 - Real.sqrt 9 = Real.sqrt 2 / 2 - 2 := by sorry

end NUMINAMATH_CALUDE_problem_1_l869_86991


namespace NUMINAMATH_CALUDE_square_triangle_perimeter_equality_l869_86999

theorem square_triangle_perimeter_equality (x : ℝ) :
  x = 4 →
  4 * (x + 2) = 3 * (2 * x) := by
  sorry

end NUMINAMATH_CALUDE_square_triangle_perimeter_equality_l869_86999


namespace NUMINAMATH_CALUDE_andy_weight_change_l869_86961

/-- Calculates Andy's weight change over the year -/
theorem andy_weight_change (initial_weight : ℝ) (weight_gain : ℝ) (months : ℕ) : 
  initial_weight = 156 →
  weight_gain = 36 →
  months = 3 →
  initial_weight - (initial_weight + weight_gain) * (1 - 1/8)^months = 36 := by
  sorry

#check andy_weight_change

end NUMINAMATH_CALUDE_andy_weight_change_l869_86961


namespace NUMINAMATH_CALUDE_line_not_in_second_quadrant_l869_86977

/-- A line with equation y = 3x - 2 does not pass through the second quadrant. -/
theorem line_not_in_second_quadrant :
  ∀ x y : ℝ, y = 3 * x - 2 → ¬(x > 0 ∧ y > 0) :=
by
  sorry

end NUMINAMATH_CALUDE_line_not_in_second_quadrant_l869_86977


namespace NUMINAMATH_CALUDE_rowing_speed_contradiction_l869_86925

theorem rowing_speed_contradiction (man_rate : ℝ) (with_stream : ℝ) (against_stream : ℝ) :
  man_rate = 6 →
  with_stream = 20 →
  with_stream = man_rate + (with_stream - man_rate) →
  against_stream = man_rate - (with_stream - man_rate) →
  against_stream < 0 :=
by sorry

#check rowing_speed_contradiction

end NUMINAMATH_CALUDE_rowing_speed_contradiction_l869_86925


namespace NUMINAMATH_CALUDE_divide_books_into_portions_l869_86910

theorem divide_books_into_portions (n : ℕ) (k : ℕ) : n = 6 → k = 3 → 
  (Nat.choose n 2 * Nat.choose (n - 2) 2) / Nat.factorial k = 15 := by
  sorry

end NUMINAMATH_CALUDE_divide_books_into_portions_l869_86910


namespace NUMINAMATH_CALUDE_adam_quarters_l869_86953

/-- The number of quarters Adam spent at the arcade -/
def quarters_spent : ℕ := 9

/-- The number of quarters Adam had left over -/
def quarters_left : ℕ := 79

/-- The initial number of quarters Adam had -/
def initial_quarters : ℕ := quarters_spent + quarters_left

theorem adam_quarters : initial_quarters = 88 := by
  sorry

end NUMINAMATH_CALUDE_adam_quarters_l869_86953


namespace NUMINAMATH_CALUDE_tracy_has_two_dogs_l869_86956

/-- The number of dogs Tracy has -/
def num_dogs : ℕ :=
  let cups_per_meal : ℚ := 3/2  -- 1.5 cups per meal
  let meals_per_day : ℕ := 3
  let total_pounds : ℕ := 4
  let cups_per_pound : ℚ := 9/4  -- 2.25 cups per pound

  let total_cups : ℚ := total_pounds * cups_per_pound
  let cups_per_dog_per_day : ℚ := cups_per_meal * meals_per_day

  (total_cups / cups_per_dog_per_day).num.toNat

/-- Theorem stating that Tracy has 2 dogs -/
theorem tracy_has_two_dogs : num_dogs = 2 := by
  sorry

end NUMINAMATH_CALUDE_tracy_has_two_dogs_l869_86956


namespace NUMINAMATH_CALUDE_min_sum_abs_roots_irrational_quadratic_l869_86934

theorem min_sum_abs_roots_irrational_quadratic (p q : ℤ) 
  (h_irrational : ∀ (α : ℝ), α^2 + p*α + q = 0 → ¬ IsAlgebraic ℚ α) :
  ∃ (α₁ α₂ : ℝ), 
    α₁^2 + p*α₁ + q = 0 ∧ 
    α₂^2 + p*α₂ + q = 0 ∧ 
    |α₁| + |α₂| ≥ Real.sqrt 5 ∧
    (∃ (p' q' : ℤ) (β₁ β₂ : ℝ), 
      β₁^2 + p'*β₁ + q' = 0 ∧ 
      β₂^2 + p'*β₂ + q' = 0 ∧ 
      |β₁| + |β₂| = Real.sqrt 5) := by
  sorry

end NUMINAMATH_CALUDE_min_sum_abs_roots_irrational_quadratic_l869_86934


namespace NUMINAMATH_CALUDE_polynomial_roots_and_factorization_l869_86930

theorem polynomial_roots_and_factorization (m : ℤ) :
  (∀ x : ℤ, 2 * x^4 + m * x^2 + 8 = 0 → (∃ y : ℤ, x = y)) →
  (m = -10 ∧
   ∀ x : ℝ, 2 * x^4 + m * x^2 + 8 = 2 * (x + 1) * (x - 1) * (x + 2) * (x - 2)) :=
by sorry

end NUMINAMATH_CALUDE_polynomial_roots_and_factorization_l869_86930


namespace NUMINAMATH_CALUDE_school_size_calculation_l869_86966

/-- Represents a school with stratified sampling -/
structure School where
  total_students : ℕ
  grade11_students : ℕ
  sample_size : ℕ
  grade10_sample : ℕ
  grade12_sample : ℕ

/-- The theorem stating the conditions and the conclusion to be proved -/
theorem school_size_calculation (s : School)
  (h1 : s.grade11_students = 600)
  (h2 : s.sample_size = 50)
  (h3 : s.grade10_sample = 15)
  (h4 : s.grade12_sample = 20) :
  s.total_students = 2000 := by
  sorry


end NUMINAMATH_CALUDE_school_size_calculation_l869_86966


namespace NUMINAMATH_CALUDE_solution_is_816_div_5_l869_86995

/-- The function g(y) = ∛(30y + ∛(30y + 17)) is increasing --/
axiom g_increasing (y : ℝ) : 
  Monotone (fun y => Real.rpow (30 * y + Real.rpow (30 * y + 17) (1/3 : ℝ)) (1/3 : ℝ))

/-- The equation ∛(30y + ∛(30y + 17)) = 17 has a unique solution --/
axiom unique_solution : ∃! y : ℝ, Real.rpow (30 * y + Real.rpow (30 * y + 17) (1/3 : ℝ)) (1/3 : ℝ) = 17

theorem solution_is_816_div_5 : 
  ∃! y : ℝ, Real.rpow (30 * y + Real.rpow (30 * y + 17) (1/3 : ℝ)) (1/3 : ℝ) = 17 ∧ y = 816 / 5 := by
  sorry

end NUMINAMATH_CALUDE_solution_is_816_div_5_l869_86995


namespace NUMINAMATH_CALUDE_solution_to_equation_l869_86969

theorem solution_to_equation : 
  ∃ (x₁ x₂ : ℝ), 
    (x₁ = 1.5 + Real.sqrt 1.5 ∧ x₂ = 1.5 - Real.sqrt 1.5) ∧ 
    (∀ x : ℝ, x^4 + (3 - x)^4 = 130 ↔ (x = x₁ ∨ x = x₂)) := by
  sorry

end NUMINAMATH_CALUDE_solution_to_equation_l869_86969


namespace NUMINAMATH_CALUDE_company_workforce_after_hiring_l869_86965

theorem company_workforce_after_hiring 
  (initial_female_percentage : Real) 
  (additional_male_hires : Nat) 
  (final_female_percentage : Real) :
  initial_female_percentage = 0.60 →
  additional_male_hires = 26 →
  final_female_percentage = 0.55 →
  ∃ (initial_total : Nat),
    (initial_total + additional_male_hires : Real) *
      (1 - final_female_percentage) =
    (initial_total : Real) * (1 - initial_female_percentage) +
      additional_male_hires ∧
    initial_total + additional_male_hires = 312 :=
by
  sorry

end NUMINAMATH_CALUDE_company_workforce_after_hiring_l869_86965


namespace NUMINAMATH_CALUDE_fraction_increase_l869_86911

theorem fraction_increase (x y : ℝ) (square : ℝ) :
  (2 * x * y) / (x + square) = (1 / 5) * (2 * (5 * x) * (5 * y)) / (5 * x + 5 * square) →
  square = 3 * y :=
by sorry

end NUMINAMATH_CALUDE_fraction_increase_l869_86911


namespace NUMINAMATH_CALUDE_base_prime_182_l869_86985

/-- Represents a number in base prime notation --/
def BasePrime : Type := List Nat

/-- Converts a natural number to its base prime representation --/
def toBasePrime (n : Nat) : BasePrime :=
  sorry

/-- Theorem: The base prime representation of 182 is [1, 0, 0, 1, 0, 1] --/
theorem base_prime_182 : toBasePrime 182 = [1, 0, 0, 1, 0, 1] := by
  sorry

end NUMINAMATH_CALUDE_base_prime_182_l869_86985


namespace NUMINAMATH_CALUDE_max_complex_norm_squared_l869_86921

theorem max_complex_norm_squared (θ : ℝ) : 
  let z : ℂ := 2 * Complex.cos θ + Complex.I * Complex.sin θ
  ∃ (M : ℝ), M = 4 ∧ ∀ θ' : ℝ, Complex.normSq z ≤ M :=
sorry

end NUMINAMATH_CALUDE_max_complex_norm_squared_l869_86921


namespace NUMINAMATH_CALUDE_restaurant_seating_capacity_l869_86935

theorem restaurant_seating_capacity :
  ∀ (new_tables original_tables : ℕ),
    new_tables + original_tables = 40 →
    new_tables = original_tables + 12 →
    6 * new_tables + 4 * original_tables = 212 :=
by
  sorry

end NUMINAMATH_CALUDE_restaurant_seating_capacity_l869_86935


namespace NUMINAMATH_CALUDE_floor_area_less_than_ten_l869_86970

/-- Represents a rectangular room -/
structure Room where
  length : ℝ
  width : ℝ
  height : ℝ

/-- The condition that the room's height is 3 meters -/
def height_is_three (r : Room) : Prop :=
  r.height = 3

/-- The condition that each wall's area is greater than the floor area -/
def walls_larger_than_floor (r : Room) : Prop :=
  r.length * r.height > r.length * r.width ∧ 
  r.width * r.height > r.length * r.width

/-- The theorem stating that under the given conditions, 
    the floor area must be less than 10 square meters -/
theorem floor_area_less_than_ten (r : Room) 
  (h1 : height_is_three r) 
  (h2 : walls_larger_than_floor r) : 
  r.length * r.width < 10 := by
  sorry


end NUMINAMATH_CALUDE_floor_area_less_than_ten_l869_86970


namespace NUMINAMATH_CALUDE_ellipse_parabola_intersection_l869_86924

-- Define the ellipse
def ellipse (x y : ℝ) : Prop := x^2/4 + y^2 = 1

-- Define the parabola
def parabola (x y : ℝ) : Prop := x^2 = 4*y

-- Define the line
def line (x y : ℝ) : Prop := y = x + 1

theorem ellipse_parabola_intersection :
  ∃ (x₁ x₂ y₁ y₂ : ℝ),
    ellipse 0 1 ∧  -- Vertex of ellipse at (0, 1)
    parabola 0 1 ∧  -- Focus of parabola at (0, 1)
    line x₁ y₁ ∧
    line x₂ y₂ ∧
    parabola x₁ y₁ ∧
    parabola x₂ y₂ ∧
    x₁ * x₂ = -4 := by
  sorry

end NUMINAMATH_CALUDE_ellipse_parabola_intersection_l869_86924


namespace NUMINAMATH_CALUDE_inscribed_squares_inequality_l869_86952

/-- Given a triangle ABC with semiperimeter s and area F, and squares with side lengths x, y, and z
    inscribed such that:
    - Square with side x has two vertices on BC
    - Square with side y has two vertices on AC
    - Square with side z has two vertices on AB
    The sum of the reciprocals of their side lengths is less than or equal to s(2+√3)/(2F) -/
theorem inscribed_squares_inequality (s F x y z : ℝ) (h_pos : s > 0 ∧ F > 0 ∧ x > 0 ∧ y > 0 ∧ z > 0) :
  1/x + 1/y + 1/z ≤ s * (2 + Real.sqrt 3) / (2 * F) := by
  sorry


end NUMINAMATH_CALUDE_inscribed_squares_inequality_l869_86952


namespace NUMINAMATH_CALUDE_ellipse_parabola_properties_l869_86982

/-- Represents an ellipse with semi-major axis a and semi-minor axis b -/
structure Ellipse where
  a : ℝ
  b : ℝ
  h_pos : 0 < b ∧ b < a

/-- Represents a parabola with focal length c -/
structure Parabola where
  c : ℝ
  h_pos : 0 < c

/-- The configuration of the ellipse and parabola as described in the problem -/
structure Configuration where
  C₁ : Ellipse
  C₂ : Parabola
  h_focus : C₂.c = C₁.a - C₁.b  -- Right focus of C₁ coincides with focus of C₂
  h_center : C₁.a = 2 * C₂.c    -- Center of C₁ coincides with vertex of C₂
  h_chord_ratio : 3 * C₂.c = 2 * C₁.b^2 / C₁.a  -- |CD| = 4/3 |AB|
  h_vertex_sum : C₁.a + C₂.c = 6  -- Sum of distances from vertices to directrix is 12

/-- The main theorem stating the properties to be proved -/
theorem ellipse_parabola_properties (config : Configuration) :
  config.C₁.a = 4 ∧ 
  config.C₁.b^2 = 12 ∧ 
  config.C₂.c = 2 ∧
  (config.C₁.a - config.C₁.b) / config.C₁.a = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_ellipse_parabola_properties_l869_86982


namespace NUMINAMATH_CALUDE_four_digit_divisible_by_fourteen_l869_86986

theorem four_digit_divisible_by_fourteen (n : Nat) : 
  n < 10 ∧ 945 * n < 10000 ∧ 945 * n ≥ 1000 ∧ (945 * n) % 14 = 0 → n = 8 :=
by sorry

end NUMINAMATH_CALUDE_four_digit_divisible_by_fourteen_l869_86986


namespace NUMINAMATH_CALUDE_equation_solver_l869_86942

theorem equation_solver (m n : ℕ) : 
  ((1^(m+1))/(5^(m+1))) * ((1^n)/(4^n)) = 1/(2*(10^35)) ∧ m = 34 → n = 18 := by
  sorry

end NUMINAMATH_CALUDE_equation_solver_l869_86942


namespace NUMINAMATH_CALUDE_exponent_division_l869_86968

theorem exponent_division (x : ℝ) : x^10 / x^2 = x^8 := by
  sorry

end NUMINAMATH_CALUDE_exponent_division_l869_86968


namespace NUMINAMATH_CALUDE_binary_1010_is_10_l869_86939

/-- Converts a binary digit (0 or 1) to its decimal value -/
def binaryToDecimal (digit : Nat) : Nat :=
  if digit = 0 ∨ digit = 1 then digit else 0

/-- Converts a list of binary digits to its decimal representation -/
def binaryListToDecimal (bits : List Nat) : Nat :=
  bits.enum.foldl (fun acc (i, b) => acc + binaryToDecimal b * 2^i) 0

theorem binary_1010_is_10 :
  binaryListToDecimal [0, 1, 0, 1] = 10 := by
  sorry

end NUMINAMATH_CALUDE_binary_1010_is_10_l869_86939


namespace NUMINAMATH_CALUDE_sqrt_defined_iff_l869_86915

theorem sqrt_defined_iff (x : ℝ) : Real.sqrt (5 - 3 * x) ≥ 0 ↔ x ≤ 5 / 3 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_defined_iff_l869_86915


namespace NUMINAMATH_CALUDE_expression_evaluation_l869_86990

theorem expression_evaluation :
  let a : ℚ := 1/2
  let b : ℚ := 2
  4 * (2 * a^2 * b - a * b^2) - (3 * a * b^2 + 2 * a^2 * b) = -11 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l869_86990


namespace NUMINAMATH_CALUDE_coltons_remaining_stickers_l869_86947

/-- The number of stickers Colton has left after giving some away to friends. -/
def stickers_left (initial_stickers : ℕ) (stickers_per_friend : ℕ) (num_friends : ℕ) 
  (extra_to_mandy : ℕ) (less_to_justin : ℕ) : ℕ :=
  let stickers_to_friends := stickers_per_friend * num_friends
  let stickers_to_mandy := stickers_to_friends + extra_to_mandy
  let stickers_to_justin := stickers_to_mandy - less_to_justin
  let total_given_away := stickers_to_friends + stickers_to_mandy + stickers_to_justin
  initial_stickers - total_given_away

/-- Theorem stating that Colton has 42 stickers left given the problem conditions. -/
theorem coltons_remaining_stickers : 
  stickers_left 72 4 3 2 10 = 42 := by
  sorry

end NUMINAMATH_CALUDE_coltons_remaining_stickers_l869_86947
