import Mathlib

namespace NUMINAMATH_CALUDE_percentage_increase_l2733_273329

theorem percentage_increase (original_earnings new_earnings : ℝ) 
  (h1 : original_earnings = 55)
  (h2 : new_earnings = 60) :
  ((new_earnings - original_earnings) / original_earnings) * 100 = 9.09 := by
  sorry

end NUMINAMATH_CALUDE_percentage_increase_l2733_273329


namespace NUMINAMATH_CALUDE_fraction_c_simplest_form_l2733_273365

/-- A fraction is in its simplest form if its numerator and denominator have no common factors other than 1 and -1. -/
def IsSimplestForm (num den : ℤ → ℤ → ℤ) : Prop :=
  ∀ a b : ℤ, (∀ k : ℤ, k ≠ 1 ∧ k ≠ -1 → (k ∣ num a b ↔ k ∣ den a b) → False)

/-- The fraction (3a + b) / (a + b) is in its simplest form. -/
theorem fraction_c_simplest_form :
  IsSimplestForm (fun a b => 3*a + b) (fun a b => a + b) := by
  sorry

#check fraction_c_simplest_form

end NUMINAMATH_CALUDE_fraction_c_simplest_form_l2733_273365


namespace NUMINAMATH_CALUDE_perfect_square_trinomial_l2733_273388

theorem perfect_square_trinomial (a : ℝ) : 
  (∃ b : ℝ, ∀ x : ℝ, x^2 - a*x + 9 = (x + b)^2) → a = 6 ∨ a = -6 := by
  sorry

end NUMINAMATH_CALUDE_perfect_square_trinomial_l2733_273388


namespace NUMINAMATH_CALUDE_cos_alpha_value_l2733_273383

theorem cos_alpha_value (α β : Real) 
  (h1 : -π/2 < α ∧ α < π/2)
  (h2 : 2 * Real.tan β = Real.tan (2 * α))
  (h3 : Real.tan (β - α) = -2 * Real.sqrt 2) :
  Real.cos α = Real.sqrt 3 / 3 := by
sorry

end NUMINAMATH_CALUDE_cos_alpha_value_l2733_273383


namespace NUMINAMATH_CALUDE_school_seminar_cost_l2733_273389

/-- Calculates the total amount spent by a school for a seminar with discounts and food allowance -/
theorem school_seminar_cost
  (regular_fee : ℝ)
  (discount_percent : ℝ)
  (num_teachers : ℕ)
  (food_allowance : ℝ)
  (h1 : regular_fee = 150)
  (h2 : discount_percent = 5)
  (h3 : num_teachers = 10)
  (h4 : food_allowance = 10)
  : (1 - discount_percent / 100) * regular_fee * num_teachers + food_allowance * num_teachers = 1525 := by
  sorry

#check school_seminar_cost

end NUMINAMATH_CALUDE_school_seminar_cost_l2733_273389


namespace NUMINAMATH_CALUDE_triangle_count_theorem_l2733_273355

/-- Represents a rectangle divided into columns and rows with diagonal lines -/
structure DividedRectangle where
  columns : Nat
  rows : Nat

/-- Counts the number of triangles in a divided rectangle -/
def count_triangles (rect : DividedRectangle) : Nat :=
  let smallest_triangles := rect.columns * rect.rows * 2
  let small_isosceles := rect.columns + rect.rows * 2
  let medium_right := (rect.columns / 2) * rect.rows * 2
  let large_isosceles := rect.columns / 2
  smallest_triangles + small_isosceles + medium_right + large_isosceles

/-- The main theorem stating the number of triangles in the specific rectangle -/
theorem triangle_count_theorem (rect : DividedRectangle) 
    (h_columns : rect.columns = 8) 
    (h_rows : rect.rows = 2) : 
  count_triangles rect = 76 := by
  sorry

#eval count_triangles ⟨8, 2⟩

end NUMINAMATH_CALUDE_triangle_count_theorem_l2733_273355


namespace NUMINAMATH_CALUDE_min_value_of_a_l2733_273356

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := 2 * Real.log x + a / x^2

theorem min_value_of_a (a : ℝ) (h1 : a > 0) :
  (∀ x > 0, f a x ≥ 2) → a ≥ Real.exp 1 := by
  sorry

end NUMINAMATH_CALUDE_min_value_of_a_l2733_273356


namespace NUMINAMATH_CALUDE_cos_alpha_plus_pi_sixth_l2733_273374

theorem cos_alpha_plus_pi_sixth (α : Real) :
  (∃ (x y : Real), x = 1 ∧ y = 2 ∧ x = Real.cos α * Real.sqrt (x^2 + y^2) ∧ y = Real.sin α * Real.sqrt (x^2 + y^2)) →
  Real.cos (α + π/6) = (Real.sqrt 15 - 2 * Real.sqrt 5) / 10 := by
sorry

end NUMINAMATH_CALUDE_cos_alpha_plus_pi_sixth_l2733_273374


namespace NUMINAMATH_CALUDE_max_servings_is_56_l2733_273311

/-- Represents the recipe requirements for one serving of salad -/
structure Recipe where
  cucumbers : ℕ
  tomatoes : ℕ
  brynza : ℕ  -- in grams
  peppers : ℕ

/-- Represents the available ingredients in the warehouse -/
structure Warehouse where
  cucumbers : ℕ
  tomatoes : ℕ
  brynza : ℕ  -- in grams
  peppers : ℕ

/-- Calculates the maximum number of servings that can be made -/
def max_servings (recipe : Recipe) (warehouse : Warehouse) : ℕ :=
  min
    (warehouse.cucumbers / recipe.cucumbers)
    (min
      (warehouse.tomatoes / recipe.tomatoes)
      (min
        (warehouse.brynza / recipe.brynza)
        (warehouse.peppers / recipe.peppers)))

/-- Theorem: The maximum number of servings that can be made is 56 -/
theorem max_servings_is_56 :
  let recipe := Recipe.mk 2 2 75 1
  let warehouse := Warehouse.mk 117 116 4200 60
  max_servings recipe warehouse = 56 := by
  sorry

#eval max_servings (Recipe.mk 2 2 75 1) (Warehouse.mk 117 116 4200 60)

end NUMINAMATH_CALUDE_max_servings_is_56_l2733_273311


namespace NUMINAMATH_CALUDE_volleyball_team_math_players_l2733_273334

/-- The number of players taking mathematics in a volleyball team -/
def players_taking_mathematics (total_players : ℕ) (physics_players : ℕ) (both_subjects : ℕ) : ℕ :=
  total_players - (physics_players - both_subjects)

/-- Theorem stating the number of players taking mathematics -/
theorem volleyball_team_math_players :
  let total_players : ℕ := 30
  let physics_players : ℕ := 15
  let both_subjects : ℕ := 6
  players_taking_mathematics total_players physics_players both_subjects = 21 := by
  sorry

#check volleyball_team_math_players

end NUMINAMATH_CALUDE_volleyball_team_math_players_l2733_273334


namespace NUMINAMATH_CALUDE_quadratic_single_intersection_l2733_273399

/-- 
A quadratic function f(x) = ax^2 - ax + 3x + 1 intersects the x-axis at only one point 
if and only if a = 1 or a = 9.
-/
theorem quadratic_single_intersection (a : ℝ) : 
  (∃! x, a * x^2 - a * x + 3 * x + 1 = 0) ↔ (a = 1 ∨ a = 9) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_single_intersection_l2733_273399


namespace NUMINAMATH_CALUDE_largest_valid_number_l2733_273335

def is_valid_number (n : ℕ) : Prop :=
  1000 ≤ n ∧ n < 10000 ∧
  ∃ (a b : ℕ), a ≠ b ∧ a < 10 ∧ b < 10 ∧
    n = 7000 + 100 * a + 20 + b ∧
    n % 30 = 0

theorem largest_valid_number :
  ∀ n : ℕ, is_valid_number n → n ≤ 7920 :=
sorry

end NUMINAMATH_CALUDE_largest_valid_number_l2733_273335


namespace NUMINAMATH_CALUDE_hexagon_arrangement_count_l2733_273339

/-- Represents a valid arrangement of digits on a regular hexagon with center -/
structure HexagonArrangement where
  vertices : Fin 6 → Fin 7
  center : Fin 7
  all_different : ∀ i j : Fin 6, i ≠ j → vertices i ≠ vertices j
  center_different : ∀ i : Fin 6, center ≠ vertices i
  sum_equal : ∀ i : Fin 3, 
    (vertices i).val + center.val + (vertices (i + 3)).val = 
    (vertices (i + 1)).val + center.val + (vertices (i + 4)).val

/-- The number of valid hexagon arrangements -/
def count_arrangements : ℕ := sorry

/-- Theorem stating the correct number of arrangements -/
theorem hexagon_arrangement_count : count_arrangements = 144 := by sorry

end NUMINAMATH_CALUDE_hexagon_arrangement_count_l2733_273339


namespace NUMINAMATH_CALUDE_geometric_sum_remainder_l2733_273396

theorem geometric_sum_remainder (n : ℕ) (a r : ℤ) (m : ℕ) (h : m > 0) :
  (a * (r^(n+1) - 1)) / (r - 1) % m = 91 :=
by
  sorry

#check geometric_sum_remainder 2002 1 9 500

end NUMINAMATH_CALUDE_geometric_sum_remainder_l2733_273396


namespace NUMINAMATH_CALUDE_population_trend_decreasing_l2733_273310

theorem population_trend_decreasing 
  (k : ℝ) 
  (h1 : -1 < k) 
  (h2 : k < 0) 
  (P : ℝ) 
  (hP : P > 0) :
  ∀ n : ℕ, ∀ m : ℕ, n < m → P * (1 + k)^n > P * (1 + k)^m :=
sorry

end NUMINAMATH_CALUDE_population_trend_decreasing_l2733_273310


namespace NUMINAMATH_CALUDE_meaningful_iff_greater_than_one_l2733_273384

-- Define the condition for the expression to be meaningful
def is_meaningful (x : ℝ) : Prop := x > 1

-- Theorem stating that the expression is meaningful if and only if x > 1
theorem meaningful_iff_greater_than_one (x : ℝ) :
  is_meaningful x ↔ x > 1 :=
by sorry

end NUMINAMATH_CALUDE_meaningful_iff_greater_than_one_l2733_273384


namespace NUMINAMATH_CALUDE_absent_students_percentage_l2733_273359

theorem absent_students_percentage
  (total_students : ℕ)
  (boys : ℕ)
  (girls : ℕ)
  (boys_absent_fraction : ℚ)
  (girls_absent_fraction : ℚ)
  (h1 : total_students = 240)
  (h2 : boys = 150)
  (h3 : girls = 90)
  (h4 : boys_absent_fraction = 1 / 5)
  (h5 : girls_absent_fraction = 1 / 2)
  (h6 : total_students = boys + girls) :
  (boys_absent_fraction * boys + girls_absent_fraction * girls) / total_students = 5 / 16 :=
by sorry

end NUMINAMATH_CALUDE_absent_students_percentage_l2733_273359


namespace NUMINAMATH_CALUDE_pizza_toppings_l2733_273370

theorem pizza_toppings (total_slices : ℕ) (pepperoni_slices : ℕ) (mushroom_slices : ℕ) 
  (h1 : total_slices = 24)
  (h2 : pepperoni_slices = 15)
  (h3 : mushroom_slices = 16)
  (h4 : ∀ (slice : ℕ), slice < total_slices → (slice < pepperoni_slices ∨ slice < mushroom_slices)) :
  pepperoni_slices + mushroom_slices - total_slices = 7 :=
by sorry

end NUMINAMATH_CALUDE_pizza_toppings_l2733_273370


namespace NUMINAMATH_CALUDE_train_journey_distance_l2733_273347

/-- Represents the train journey problem -/
def TrainJourney (x v : ℝ) : Prop :=
  -- Train stops after 1 hour and remains halted for 0.5 hours
  let initial_stop_time : ℝ := 1.5
  -- Train continues at 3/4 of original speed
  let reduced_speed : ℝ := 3/4 * v
  -- Total delay equation
  let delay_equation : Prop := (x/v + initial_stop_time + (x-v)/reduced_speed - x/v = 3.5)
  -- Equation for incident 90 miles further
  let further_incident_equation : Prop := 
    ((x-90)/v + initial_stop_time + (x-90)/reduced_speed - x/v + 90/v = 3)
  
  -- All conditions must be satisfied
  delay_equation ∧ further_incident_equation

/-- The theorem to be proved -/
theorem train_journey_distance : 
  ∃ (v : ℝ), TrainJourney 600 v := by sorry

end NUMINAMATH_CALUDE_train_journey_distance_l2733_273347


namespace NUMINAMATH_CALUDE_face_value_in_product_l2733_273363

/-- Given a number with specific local values for its digits and their product, 
    prove the face value of a digit with a given local value in the product. -/
theorem face_value_in_product (n : ℕ) (product : ℕ) 
  (local_value_6 : ℕ) (local_value_8 : ℕ) :
  n = 7098060 →
  local_value_6 = 6000 →
  local_value_8 = 80 →
  product = local_value_6 * local_value_8 →
  (product / 1000) % 10 = 6 →
  (product / 1000) % 10 = 6 := by
  sorry

end NUMINAMATH_CALUDE_face_value_in_product_l2733_273363


namespace NUMINAMATH_CALUDE_circles_relationship_l2733_273391

/-- The positional relationship between two circles -/
theorem circles_relationship (C1 C2 : ℝ × ℝ) (r1 r2 : ℝ) : 
  (C1.1 + 1)^2 + (C1.2 + 1)^2 = 4 →
  (C2.1 - 2)^2 + (C2.2 - 1)^2 = 4 →
  r1 = 2 →
  r2 = 2 →
  C1 = (-1, -1) →
  C2 = (2, 1) →
  (r1 - r2)^2 < (C1.1 - C2.1)^2 + (C1.2 - C2.2)^2 ∧ 
  (C1.1 - C2.1)^2 + (C1.2 - C2.2)^2 < (r1 + r2)^2 :=
by sorry


end NUMINAMATH_CALUDE_circles_relationship_l2733_273391


namespace NUMINAMATH_CALUDE_sum_of_digits_of_power_l2733_273308

def tens_digit (n : ℕ) : ℕ := (n / 10) % 10

def ones_digit (n : ℕ) : ℕ := n % 10

theorem sum_of_digits_of_power : 
  tens_digit ((4 + 3) ^ 12) + ones_digit ((4 + 3) ^ 12) = 1 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_digits_of_power_l2733_273308


namespace NUMINAMATH_CALUDE_least_possible_y_l2733_273301

/-- Given that x is an even integer, y and z are odd integers,
    y - x > 5, and the least possible value of z - x is 9,
    prove that the least possible value of y is 7. -/
theorem least_possible_y (x y z : ℤ) 
  (h_x_even : Even x)
  (h_y_odd : Odd y)
  (h_z_odd : Odd z)
  (h_y_minus_x : y - x > 5)
  (h_z_minus_x_min : ∀ w, z - x ≤ w - x → w - x ≥ 9) :
  y ≥ 7 ∧ ∀ w, (Odd w ∧ w - x > 5) → y ≤ w := by
  sorry

end NUMINAMATH_CALUDE_least_possible_y_l2733_273301


namespace NUMINAMATH_CALUDE_toothpick_grids_count_l2733_273344

/-- Calculates the number of toothpicks needed for a grid -/
def toothpicks_for_grid (length : ℕ) (width : ℕ) : ℕ :=
  (length + 1) * width + (width + 1) * length

/-- The total number of toothpicks for two separate grids -/
def total_toothpicks (outer_length outer_width inner_length inner_width : ℕ) : ℕ :=
  toothpicks_for_grid outer_length outer_width + toothpicks_for_grid inner_length inner_width

theorem toothpick_grids_count :
  total_toothpicks 80 40 30 20 = 7770 := by
  sorry

end NUMINAMATH_CALUDE_toothpick_grids_count_l2733_273344


namespace NUMINAMATH_CALUDE_quadratic_inequality_condition_l2733_273304

theorem quadratic_inequality_condition (x : ℝ) : 
  (((x < 1) ∨ (x > 4)) → (x^2 - 3*x + 2 > 0)) ∧ 
  (∃ x, (x^2 - 3*x + 2 > 0) ∧ ¬((x < 1) ∨ (x > 4))) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_condition_l2733_273304


namespace NUMINAMATH_CALUDE_triangle_angle_measure_l2733_273390

theorem triangle_angle_measure (D E F : ℝ) : 
  0 < D ∧ 0 < E ∧ 0 < F →  -- Angles are positive
  D + E + F = 180 →        -- Sum of angles in a triangle
  E = 3 * F →              -- Angle E is three times angle F
  F = 18 →                 -- Angle F is 18 degrees
  D = 108 :=               -- Conclusion: Angle D is 108 degrees
by sorry

end NUMINAMATH_CALUDE_triangle_angle_measure_l2733_273390


namespace NUMINAMATH_CALUDE_no_dual_integer_root_quadratics_l2733_273397

theorem no_dual_integer_root_quadratics : 
  ¬ ∃ (a b c : ℤ), 
    (∃ (x₁ x₂ : ℤ), x₁ ≠ x₂ ∧ a * x₁^2 + b * x₁ + c = 0 ∧ a * x₂^2 + b * x₂ + c = 0) ∧
    (∃ (y₁ y₂ : ℤ), y₁ ≠ y₂ ∧ (a + 1) * y₁^2 + (b + 1) * y₁ + (c + 1) = 0 ∧ (a + 1) * y₂^2 + (b + 1) * y₂ + (c + 1) = 0) :=
by sorry

end NUMINAMATH_CALUDE_no_dual_integer_root_quadratics_l2733_273397


namespace NUMINAMATH_CALUDE_largest_eight_digit_even_digits_proof_l2733_273376

def even_digits : List Nat := [0, 2, 4, 6, 8]

def is_eight_digit (n : Nat) : Prop :=
  10000000 ≤ n ∧ n ≤ 99999999

def contains_all_even_digits (n : Nat) : Prop :=
  ∀ d ∈ even_digits, ∃ k : Nat, n / (10^k) % 10 = d

def largest_eight_digit_with_even_digits : Nat := 99986420

theorem largest_eight_digit_even_digits_proof :
  is_eight_digit largest_eight_digit_with_even_digits ∧
  contains_all_even_digits largest_eight_digit_with_even_digits ∧
  ∀ m : Nat, is_eight_digit m ∧ contains_all_even_digits m →
    m ≤ largest_eight_digit_with_even_digits :=
by sorry

end NUMINAMATH_CALUDE_largest_eight_digit_even_digits_proof_l2733_273376


namespace NUMINAMATH_CALUDE_sqrt_difference_equals_five_sixths_l2733_273361

theorem sqrt_difference_equals_five_sixths : 
  Real.sqrt (9 / 4) - Real.sqrt (4 / 9) = 5 / 6 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_difference_equals_five_sixths_l2733_273361


namespace NUMINAMATH_CALUDE_p_sufficient_not_necessary_l2733_273382

-- Define the propositions p and q
def p (x : ℝ) : Prop := x = 2
def q (x : ℝ) : Prop := 0 < x ∧ x < 3

-- Theorem stating that p is a sufficient but not necessary condition for q
theorem p_sufficient_not_necessary :
  (∀ x, p x → q x) ∧ 
  (∃ x, q x ∧ ¬(p x)) :=
sorry

end NUMINAMATH_CALUDE_p_sufficient_not_necessary_l2733_273382


namespace NUMINAMATH_CALUDE_path_area_is_78_l2733_273362

/-- Represents the dimensions of a garden with flower beds and paths. -/
structure GardenDimensions where
  rows : Nat
  columns : Nat
  bedLength : Nat
  bedWidth : Nat
  pathWidth : Nat

/-- Calculates the total area of paths in a garden given its dimensions. -/
def pathArea (g : GardenDimensions) : Nat :=
  let totalWidth := g.pathWidth + g.columns * g.bedLength + (g.columns - 1) * g.pathWidth + g.pathWidth
  let totalHeight := g.pathWidth + g.rows * g.bedWidth + (g.rows - 1) * g.pathWidth + g.pathWidth
  let totalArea := totalWidth * totalHeight
  let bedArea := g.rows * g.columns * g.bedLength * g.bedWidth
  totalArea - bedArea

/-- Theorem stating that the path area for the given garden dimensions is 78 square feet. -/
theorem path_area_is_78 (g : GardenDimensions) 
    (h1 : g.rows = 3) 
    (h2 : g.columns = 2) 
    (h3 : g.bedLength = 6) 
    (h4 : g.bedWidth = 2) 
    (h5 : g.pathWidth = 1) : 
  pathArea g = 78 := by
  sorry

end NUMINAMATH_CALUDE_path_area_is_78_l2733_273362


namespace NUMINAMATH_CALUDE_peak_speed_scientific_notation_l2733_273320

/-- The peak computing speed of a certain server in operations per second. -/
def peak_speed : ℕ := 403200000000

/-- The scientific notation representation of the peak speed. -/
def scientific_notation : ℝ := 4.032 * (10 ^ 11)

/-- Theorem stating that the peak speed is equal to its scientific notation representation. -/
theorem peak_speed_scientific_notation : (peak_speed : ℝ) = scientific_notation := by
  sorry

end NUMINAMATH_CALUDE_peak_speed_scientific_notation_l2733_273320


namespace NUMINAMATH_CALUDE_line_relationships_l2733_273309

/-- Two lines in the 2D plane -/
structure TwoLines where
  l1 : ℝ → ℝ → ℝ → ℝ  -- (2a+1)x+(a+2)y+3=0
  l2 : ℝ → ℝ → ℝ → ℝ  -- (a-1)x-2y+2=0

/-- Definition of parallel lines -/
def parallel (lines : TwoLines) (a : ℝ) : Prop :=
  ∀ x y, lines.l1 a x y = 0 ↔ lines.l2 a x y = 0

/-- Definition of perpendicular lines -/
def perpendicular (lines : TwoLines) (a : ℝ) : Prop :=
  ∀ x1 y1 x2 y2, lines.l1 a x1 y1 = 0 ∧ lines.l2 a x2 y2 = 0 →
    (x2 - x1) * ((2 * a + 1) * (x2 - x1) + (a + 2) * (y2 - y1)) +
    (y2 - y1) * ((a - 1) * (x2 - x1) - 2 * (y2 - y1)) = 0

/-- The main theorem -/
theorem line_relationships (lines : TwoLines) :
  (∀ a, parallel lines a ↔ a = 0) ∧
  (∀ a, perpendicular lines a ↔ a = -1 ∨ a = 5/2) := by
  sorry

#check line_relationships

end NUMINAMATH_CALUDE_line_relationships_l2733_273309


namespace NUMINAMATH_CALUDE_smallest_candy_count_l2733_273398

theorem smallest_candy_count : ∃ (n : ℕ), 
  (n ≥ 100 ∧ n < 1000) ∧ 
  (n + 5) % 8 = 0 ∧ 
  (n - 8) % 5 = 0 ∧
  (∀ m : ℕ, m ≥ 100 ∧ m < 1000 ∧ (m + 5) % 8 = 0 ∧ (m - 8) % 5 = 0 → n ≤ m) ∧
  n = 123 :=
by sorry

end NUMINAMATH_CALUDE_smallest_candy_count_l2733_273398


namespace NUMINAMATH_CALUDE_power_function_through_point_l2733_273343

theorem power_function_through_point (n : ℝ) : 
  (∀ x y : ℝ, y = x^n → (x = 2 ∧ y = 8) → n = 3) :=
by sorry

end NUMINAMATH_CALUDE_power_function_through_point_l2733_273343


namespace NUMINAMATH_CALUDE_max_ab_value_l2733_273358

theorem max_ab_value (a b : ℝ) : 
  (∀ x : ℤ, (20 * x + a > 0 ∧ 15 * x - b ≤ 0) ↔ (x = 2 ∨ x = 3 ∨ x = 4)) →
  (∃ (a' b' : ℝ), a' * b' = -1200 ∧ ∀ (a'' b'' : ℝ), 
    (∀ x : ℤ, (20 * x + a'' > 0 ∧ 15 * x - b'' ≤ 0) ↔ (x = 2 ∨ x = 3 ∨ x = 4)) →
    a'' * b'' ≤ a' * b') :=
by sorry

end NUMINAMATH_CALUDE_max_ab_value_l2733_273358


namespace NUMINAMATH_CALUDE_intersection_line_of_planes_l2733_273368

/-- Represents a plane with its first trace and angle of inclination -/
structure Plane where
  firstTrace : Line2D
  inclinationAngle : ℝ

/-- Represents a point in 2D space -/
structure Point2D where
  x : ℝ
  y : ℝ

/-- Represents a line in 2D space -/
structure Line2D where
  point1 : Point2D
  point2 : Point2D

/-- Finds the intersection point of two 2D lines -/
def intersectionPoint (l1 l2 : Line2D) : Point2D :=
  sorry

/-- Constructs a point using the angles of inclination -/
def constructPoint (p1 p2 : Plane) : Point2D :=
  sorry

/-- Theorem stating that the intersection line of two planes can be determined
    by connecting two specific points -/
theorem intersection_line_of_planes (p1 p2 : Plane) :
  ∃ (l : Line2D),
    l.point1 = intersectionPoint p1.firstTrace p2.firstTrace ∧
    l.point2 = constructPoint p1 p2 :=
  sorry

end NUMINAMATH_CALUDE_intersection_line_of_planes_l2733_273368


namespace NUMINAMATH_CALUDE_variety_show_arrangements_l2733_273327

def dance_song_count : ℕ := 3
def comedy_skit_count : ℕ := 2
def cross_talk_count : ℕ := 1

def non_adjacent_arrangements (ds c ct : ℕ) : ℕ :=
  ds.factorial * (2 * ds.factorial + c.choose 1 * (ds - 1).factorial * (ds - 1).factorial)

theorem variety_show_arrangements :
  non_adjacent_arrangements dance_song_count comedy_skit_count cross_talk_count = 120 := by
  sorry

end NUMINAMATH_CALUDE_variety_show_arrangements_l2733_273327


namespace NUMINAMATH_CALUDE_business_trip_bus_distance_l2733_273394

theorem business_trip_bus_distance (total_distance : ℝ) 
  (h_total : total_distance = 1800) 
  (h_plane : total_distance / 4 = 450) 
  (h_train : total_distance / 6 = 300) 
  (h_taxi : total_distance / 8 = 225) 
  (h_bus_rental : ∃ (bus rental : ℝ), 
    bus + rental = total_distance - (450 + 300 + 225) ∧ 
    bus = rental / 2) : 
  ∃ (bus : ℝ), bus = 275 := by
  sorry

end NUMINAMATH_CALUDE_business_trip_bus_distance_l2733_273394


namespace NUMINAMATH_CALUDE_host_horse_speed_calculation_l2733_273372

/-- The daily travel distance of the guest's horse in li -/
def guest_horse_speed : ℚ := 300

/-- The fraction of the day that passes before the host realizes the guest left without clothes -/
def realization_time : ℚ := 1/3

/-- The fraction of the day that has passed when the host returns home -/
def return_time : ℚ := 3/4

/-- The daily travel distance of the host's horse in li -/
def host_horse_speed : ℚ := 780

theorem host_horse_speed_calculation :
  let catch_up_time : ℚ := return_time - realization_time
  let guest_travel_time : ℚ := realization_time + catch_up_time
  2 * guest_horse_speed * guest_travel_time = host_horse_speed * catch_up_time :=
by sorry

end NUMINAMATH_CALUDE_host_horse_speed_calculation_l2733_273372


namespace NUMINAMATH_CALUDE_tourist_meeting_time_l2733_273331

/-- Represents a tourist -/
structure Tourist where
  name : String

/-- Represents a meeting between two tourists -/
structure Meeting where
  tourist1 : Tourist
  tourist2 : Tourist
  time : ℕ  -- Time in hours after noon

/-- The problem setup -/
def tourist_problem (vitya pasha katya masha : Tourist) : Prop :=
  ∃ (vitya_masha vitya_katya pasha_masha pasha_katya : Meeting),
    -- Meetings
    vitya_masha.tourist1 = vitya ∧ vitya_masha.tourist2 = masha ∧ vitya_masha.time = 0 ∧
    vitya_katya.tourist1 = vitya ∧ vitya_katya.tourist2 = katya ∧ vitya_katya.time = 2 ∧
    pasha_masha.tourist1 = pasha ∧ pasha_masha.tourist2 = masha ∧ pasha_masha.time = 3 ∧
    -- Vitya and Pasha travel at the same speed from A to B
    (vitya_masha.time - vitya_katya.time = pasha_masha.time - pasha_katya.time) ∧
    -- Katya and Masha travel at the same speed from B to A
    (vitya_masha.time - pasha_masha.time = vitya_katya.time - pasha_katya.time) →
    pasha_katya.time = 5

theorem tourist_meeting_time (vitya pasha katya masha : Tourist) :
  tourist_problem vitya pasha katya masha := by
  sorry

#check tourist_meeting_time

end NUMINAMATH_CALUDE_tourist_meeting_time_l2733_273331


namespace NUMINAMATH_CALUDE_tiling_pattern_ratio_l2733_273345

/-- The ratio of the area covered by triangles to the total area in a specific tiling pattern -/
theorem tiling_pattern_ratio : ∀ s : ℝ,
  s > 0 →
  let hexagon_area := (3 * Real.sqrt 3 / 2) * s^2
  let triangle_area := (Real.sqrt 3 / 16) * s^2
  let total_area := hexagon_area + 2 * triangle_area
  triangle_area / total_area = 1 / 13 :=
by sorry

end NUMINAMATH_CALUDE_tiling_pattern_ratio_l2733_273345


namespace NUMINAMATH_CALUDE_largest_coeff_sixth_term_l2733_273342

-- Define the binomial coefficient
def binomial_coeff (n k : ℕ) : ℚ := (Nat.choose n k : ℚ)

-- Define the coefficient of the r-th term in the expansion
def coeff (r : ℕ) : ℚ := (1/2)^r * binomial_coeff 15 r

-- Theorem statement
theorem largest_coeff_sixth_term :
  ∀ k : ℕ, k ≠ 5 → coeff 5 ≥ coeff k :=
sorry

end NUMINAMATH_CALUDE_largest_coeff_sixth_term_l2733_273342


namespace NUMINAMATH_CALUDE_fraction_equivalence_l2733_273395

theorem fraction_equivalence :
  (14 / 12 : ℚ) = 7 / 6 ∧
  (1 + 1 / 6 : ℚ) = 7 / 6 ∧
  (1 + 5 / 30 : ℚ) = 7 / 6 ∧
  (1 + 2 / 6 : ℚ) ≠ 7 / 6 ∧
  (1 + 14 / 42 : ℚ) = 7 / 6 :=
by sorry

end NUMINAMATH_CALUDE_fraction_equivalence_l2733_273395


namespace NUMINAMATH_CALUDE_rug_area_theorem_l2733_273328

/-- Given three overlapping rugs, calculates their combined area -/
def combined_rug_area (total_floor_area two_layer_area three_layer_area : ℝ) : ℝ :=
  let one_layer_area := total_floor_area - two_layer_area - three_layer_area
  one_layer_area + 2 * two_layer_area + 3 * three_layer_area

/-- Theorem stating that the combined area of three rugs is 200 square meters
    given the specified overlapping conditions -/
theorem rug_area_theorem :
  combined_rug_area 138 24 19 = 200 := by
  sorry


end NUMINAMATH_CALUDE_rug_area_theorem_l2733_273328


namespace NUMINAMATH_CALUDE_solve_cafeteria_problem_l2733_273351

/-- Represents the amount paid by each friend in kopecks -/
structure Payment where
  misha : ℕ
  sasha : ℕ
  grisha : ℕ

/-- Represents the number of dishes each friend paid for -/
structure Dishes where
  misha : ℕ
  sasha : ℕ
  total : ℕ

def cafeteria_problem (p : Payment) (d : Dishes) : Prop :=
  -- All dishes cost the same
  ∃ (dish_cost : ℕ),
  -- Misha paid for 3 dishes
  p.misha = d.misha * dish_cost ∧
  -- Sasha paid for 2 dishes
  p.sasha = d.sasha * dish_cost ∧
  -- Together they ate 5 dishes
  d.total = d.misha + d.sasha ∧
  -- Grisha should pay his friends a total of 50 kopecks
  p.grisha = 50 ∧
  -- Each friend should receive an equal payment
  p.misha + p.sasha + p.grisha = d.total * dish_cost ∧
  -- Prove that Grisha should pay 40 kopecks to Misha and 10 kopecks to Sasha
  p.misha - (p.misha + p.sasha + p.grisha) / 3 = 40 ∧
  p.sasha - (p.misha + p.sasha + p.grisha) / 3 = 10

theorem solve_cafeteria_problem :
  ∃ (p : Payment) (d : Dishes), cafeteria_problem p d :=
sorry

end NUMINAMATH_CALUDE_solve_cafeteria_problem_l2733_273351


namespace NUMINAMATH_CALUDE_fraction_value_l2733_273317

theorem fraction_value (x y : ℚ) (hx : x = 3) (hy : y = 4) :
  (1 / x + 1 / y) / (1 / x - 1 / y) = 7 := by
  sorry

end NUMINAMATH_CALUDE_fraction_value_l2733_273317


namespace NUMINAMATH_CALUDE_expression_value_l2733_273323

theorem expression_value (a b c : ℝ) (h : a^2 + b = b^2 + c ∧ b^2 + c = c^2 + a) :
  a * (a^2 - b^2) + b * (b^2 - c^2) + c * (c^2 - a^2) = 0 := by
  sorry

end NUMINAMATH_CALUDE_expression_value_l2733_273323


namespace NUMINAMATH_CALUDE_function_upper_bound_l2733_273360

/-- A function satisfying the given conditions -/
def satisfies_conditions (f : ℝ → ℝ) : Prop :=
  (∀ x, x ≥ 0 → ∃ (fx : ℝ), f x = fx) ∧  -- f is defined for x ≥ 0
  (∀ x y, x ≥ 0 → y ≥ 0 → f x * f y ≤ y^2 * f (x/2) + x^2 * f (y/2)) ∧
  (∃ M : ℝ, M > 0 ∧ ∀ x, 0 ≤ x → x ≤ 1 → |f x| ≤ M)

/-- The main theorem -/
theorem function_upper_bound (f : ℝ → ℝ) (h : satisfies_conditions f) :
  ∀ x, x ≥ 0 → f x ≤ x^2 := by
  sorry

end NUMINAMATH_CALUDE_function_upper_bound_l2733_273360


namespace NUMINAMATH_CALUDE_f_sum_positive_l2733_273306

def f (x : ℝ) : ℝ := x^2015

theorem f_sum_positive (a b : ℝ) (h1 : a + b > 0) (h2 : a * b < 0) : 
  f a + f b > 0 := by
  sorry

end NUMINAMATH_CALUDE_f_sum_positive_l2733_273306


namespace NUMINAMATH_CALUDE_min_value_of_z_l2733_273315

theorem min_value_of_z (x y : ℝ) : 
  3 * x^2 + y^2 + 12 * x - 6 * y + 40 ≥ 19 ∧ 
  ∃ x y : ℝ, 3 * x^2 + y^2 + 12 * x - 6 * y + 40 = 19 := by
sorry

end NUMINAMATH_CALUDE_min_value_of_z_l2733_273315


namespace NUMINAMATH_CALUDE_sum_of_solutions_quadratic_l2733_273366

theorem sum_of_solutions_quadratic (x : ℝ) : 
  (∃ p q : ℝ, p + q = -27 ∧ 81 - 27*x - x^2 = 0 → x = p ∨ x = q) :=
by sorry

end NUMINAMATH_CALUDE_sum_of_solutions_quadratic_l2733_273366


namespace NUMINAMATH_CALUDE_boys_neither_happy_nor_sad_l2733_273307

/-- Given a group of children with various emotional states and genders, 
    prove the number of boys who are neither happy nor sad. -/
theorem boys_neither_happy_nor_sad 
  (total_children : ℕ) 
  (happy_children : ℕ) 
  (sad_children : ℕ) 
  (neither_children : ℕ) 
  (total_boys : ℕ) 
  (total_girls : ℕ) 
  (happy_boys : ℕ) 
  (sad_girls : ℕ) 
  (h1 : total_children = 60)
  (h2 : happy_children = 30)
  (h3 : sad_children = 10)
  (h4 : neither_children = 20)
  (h5 : total_boys = 17)
  (h6 : total_girls = 43)
  (h7 : happy_boys = 6)
  (h8 : sad_girls = 4)
  (h9 : total_children = happy_children + sad_children + neither_children)
  (h10 : total_children = total_boys + total_girls) :
  total_boys - (happy_boys + (sad_children - sad_girls)) = 5 := by
  sorry

end NUMINAMATH_CALUDE_boys_neither_happy_nor_sad_l2733_273307


namespace NUMINAMATH_CALUDE_max_elevation_l2733_273300

/-- The elevation function of a particle projected vertically upward -/
def s (t : ℝ) : ℝ := 200 * t - 20 * t^2

/-- The maximum elevation reached by the particle -/
theorem max_elevation : ∃ t : ℝ, ∀ u : ℝ, s u ≤ s t ∧ s t = 500 := by
  sorry

end NUMINAMATH_CALUDE_max_elevation_l2733_273300


namespace NUMINAMATH_CALUDE_image_square_characterization_l2733_273337

-- Define the transformation
def transform (x y : ℝ) : ℝ × ℝ := (x^2 - y^2, x*y)

-- Define the unit square
def unit_square : Set (ℝ × ℝ) := {p | 0 ≤ p.1 ∧ p.1 ≤ 1 ∧ 0 ≤ p.2 ∧ p.2 ≤ 1}

-- Define the image of the unit square
def image_square : Set (ℝ × ℝ) := {p | ∃ q ∈ unit_square, transform q.1 q.2 = p}

-- Define the boundary curves
def curve_OC : Set (ℝ × ℝ) := {p | ∃ y ∈ Set.Icc 0 1, p = (-y^2, 0)}
def curve_OA : Set (ℝ × ℝ) := {p | ∃ x ∈ Set.Icc 0 1, p = (x^2, 0)}
def curve_AB : Set (ℝ × ℝ) := {p | ∃ y ∈ Set.Icc 0 1, p = (1 - y^2, y)}
def curve_BC : Set (ℝ × ℝ) := {p | ∃ x ∈ Set.Icc 0 1, p = (x^2 - 1, x)}

-- Define the boundary of the image
def image_boundary : Set (ℝ × ℝ) := curve_OC ∪ curve_OA ∪ curve_AB ∪ curve_BC

-- Theorem statement
theorem image_square_characterization :
  image_square = {p | p ∈ image_boundary ∨ (∃ q ∈ image_boundary, p.1 < q.1 ∧ p.2 < q.2)} := by
  sorry

end NUMINAMATH_CALUDE_image_square_characterization_l2733_273337


namespace NUMINAMATH_CALUDE_article_pricing_l2733_273350

theorem article_pricing (P : ℝ) (P_pos : P > 0) : 
  (2/3 * P = 0.9 * ((2/3 * P) / 0.9)) → 
  ((P - ((2/3 * P) / 0.9)) / ((2/3 * P) / 0.9)) * 100 = 35 := by
  sorry

end NUMINAMATH_CALUDE_article_pricing_l2733_273350


namespace NUMINAMATH_CALUDE_stock_price_increase_l2733_273393

/-- Calculate the percent increase in stock price -/
theorem stock_price_increase (opening_price closing_price : ℝ) 
  (h1 : opening_price = 25)
  (h2 : closing_price = 28) :
  (closing_price - opening_price) / opening_price * 100 = 12 := by
  sorry

end NUMINAMATH_CALUDE_stock_price_increase_l2733_273393


namespace NUMINAMATH_CALUDE_larry_channels_l2733_273340

/-- Calculates the final number of channels Larry has after a series of changes. -/
def final_channels (initial : ℕ) (removed1 : ℕ) (added1 : ℕ) (removed2 : ℕ) (added2 : ℕ) (added3 : ℕ) : ℕ :=
  initial - removed1 + added1 - removed2 + added2 + added3

/-- Theorem stating that given the specific changes to Larry's channel package, he ends up with 147 channels. -/
theorem larry_channels : final_channels 150 20 12 10 8 7 = 147 := by
  sorry

end NUMINAMATH_CALUDE_larry_channels_l2733_273340


namespace NUMINAMATH_CALUDE_reflected_ray_equation_l2733_273387

-- Define the points and lines
def P : ℝ × ℝ := (2, 3)
def A : ℝ × ℝ := (1, 1)
def incident_line (x y : ℝ) : Prop := x + y + 1 = 0

-- Define the reflected ray
def reflected_ray (x y : ℝ) : Prop := 4*x - 5*y + 1 = 0

-- Theorem statement
theorem reflected_ray_equation :
  ∃ (x₀ y₀ : ℝ), 
    incident_line x₀ y₀ ∧  -- The incident ray strikes the line x + y + 1 = 0
    (∃ (t : ℝ), (1 - t) • P.1 + t • x₀ = P.1 ∧ (1 - t) • P.2 + t • y₀ = P.2) ∧  -- The incident ray passes through P
    reflected_ray A.1 A.2  -- The reflected ray passes through A
  → ∀ (x y : ℝ), reflected_ray x y ↔ 4*x - 5*y + 1 = 0 :=
by sorry

end NUMINAMATH_CALUDE_reflected_ray_equation_l2733_273387


namespace NUMINAMATH_CALUDE_binomial_coefficient_divisibility_equivalence_l2733_273392

theorem binomial_coefficient_divisibility_equivalence 
  (n : ℕ) (p : ℕ) (h_prime : Prime p) : 
  (∀ k : ℕ, k ≤ n → ¬(p ∣ Nat.choose n k)) ↔ 
  (∃ s m : ℕ, s > 0 ∧ m < p ∧ n = p^s * m - 1) :=
sorry

end NUMINAMATH_CALUDE_binomial_coefficient_divisibility_equivalence_l2733_273392


namespace NUMINAMATH_CALUDE_equation_solutions_l2733_273354

theorem equation_solutions (x : ℝ) :
  x ≠ 2 → x ≠ 4 →
  ((x - 1) * (x - 2) * (x - 3) * (x - 4) * (x - 3) * (x - 2) * (x - 1) = 
   (x - 2) * (x - 4) * (x - 2)) ↔ 
  (x = 2 + Real.sqrt 2 ∨ x = 2 - Real.sqrt 2) :=
by sorry

end NUMINAMATH_CALUDE_equation_solutions_l2733_273354


namespace NUMINAMATH_CALUDE_greatest_integer_problem_l2733_273375

theorem greatest_integer_problem : 
  ∃ (m : ℕ), m < 150 ∧ 
  (∃ (k : ℕ), m = 9 * k - 2) ∧ 
  (∃ (j : ℕ), m = 11 * j - 4) ∧
  (∀ (n : ℕ), n < 150 → 
    (∃ (k' : ℕ), n = 9 * k' - 2) → 
    (∃ (j' : ℕ), n = 11 * j' - 4) → 
    n ≤ m) ∧
  m = 142 := by
sorry

end NUMINAMATH_CALUDE_greatest_integer_problem_l2733_273375


namespace NUMINAMATH_CALUDE_polygon_sides_count_l2733_273349

theorem polygon_sides_count : ∃ n : ℕ, 
  n > 2 ∧ 
  (n * (n - 3)) / 2 = 2 * n ∧ 
  ∀ m : ℕ, m > 2 → (m * (m - 3)) / 2 = 2 * m → m = n :=
by sorry

end NUMINAMATH_CALUDE_polygon_sides_count_l2733_273349


namespace NUMINAMATH_CALUDE_arithmetic_sequence_example_l2733_273325

def isArithmeticSequence (a : ℕ → ℕ) : Prop :=
  ∃ d : ℕ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_example :
  ∀ a : ℕ → ℕ,
  isArithmeticSequence a →
  a 1 = 7 →
  a 3 = 11 →
  a 2 = 9 ∧ a 4 = 13 ∧ a 5 = 15 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_example_l2733_273325


namespace NUMINAMATH_CALUDE_g_one_equals_three_l2733_273357

-- Define f as an odd function
def f : ℝ → ℝ := sorry

-- Define g as an even function
def g : ℝ → ℝ := sorry

-- Axiom for odd function
axiom f_odd : ∀ x : ℝ, f (-x) = -f x

-- Axiom for even function
axiom g_even : ∀ x : ℝ, g (-x) = g x

-- Given conditions
axiom condition1 : f (-1) + g 1 = 2
axiom condition2 : f 1 + g (-1) = 4

-- Theorem to prove
theorem g_one_equals_three : g 1 = 3 := by sorry

end NUMINAMATH_CALUDE_g_one_equals_three_l2733_273357


namespace NUMINAMATH_CALUDE_fraction_sum_equals_one_l2733_273341

theorem fraction_sum_equals_one (x : ℝ) : x / (x + 1) + 1 / (x + 1) = 1 := by
  sorry

end NUMINAMATH_CALUDE_fraction_sum_equals_one_l2733_273341


namespace NUMINAMATH_CALUDE_repeating_decimal_sum_l2733_273369

-- Define repeating decimals
def repeating_decimal_6 : ℚ := 2/3
def repeating_decimal_4 : ℚ := 4/9
def repeating_decimal_8 : ℚ := 8/9

-- Theorem statement
theorem repeating_decimal_sum :
  repeating_decimal_6 - repeating_decimal_4 + repeating_decimal_8 = 10/9 := by
  sorry

end NUMINAMATH_CALUDE_repeating_decimal_sum_l2733_273369


namespace NUMINAMATH_CALUDE_balloon_distribution_l2733_273318

theorem balloon_distribution (red white green chartreuse : ℕ) (friends : ℕ) : 
  red = 25 → white = 40 → green = 55 → chartreuse = 80 → friends = 10 →
  (red + white + green + chartreuse) % friends = 0 :=
by
  sorry

end NUMINAMATH_CALUDE_balloon_distribution_l2733_273318


namespace NUMINAMATH_CALUDE_uncle_bob_parking_probability_l2733_273313

/-- The number of parking spaces -/
def total_spaces : ℕ := 20

/-- The number of cars that have already parked -/
def parked_cars : ℕ := 15

/-- The number of adjacent spaces needed for Uncle Bob's truck -/
def needed_spaces : ℕ := 2

/-- The probability of having at least two adjacent empty spaces -/
def probability_adjacent_spaces : ℚ := 232 / 323

theorem uncle_bob_parking_probability :
  let total_combinations := Nat.choose total_spaces parked_cars
  let unfavorable_combinations := Nat.choose (parked_cars + needed_spaces + 1) (needed_spaces + 1)
  (1 : ℚ) - (unfavorable_combinations : ℚ) / (total_combinations : ℚ) = probability_adjacent_spaces :=
sorry

end NUMINAMATH_CALUDE_uncle_bob_parking_probability_l2733_273313


namespace NUMINAMATH_CALUDE_product_divisible_by_eight_l2733_273319

theorem product_divisible_by_eight (n : ℤ) (h : 1 ≤ n ∧ n ≤ 96) : 
  8 ∣ (n * (n + 1) * (n + 2)) := by
  sorry

end NUMINAMATH_CALUDE_product_divisible_by_eight_l2733_273319


namespace NUMINAMATH_CALUDE_total_fruits_grown_special_technique_watermelons_special_pineapples_l2733_273332

/-- Represents the fruit growing data for a person -/
structure FruitData where
  watermelons : ℕ
  pineapples : ℕ
  mangoes : ℕ
  organic_watermelons : ℕ
  hydroponic_watermelons : ℕ
  dry_season_pineapples : ℕ
  vertical_pineapples : ℕ

/-- The fruit growing data for Jason -/
def jason : FruitData := {
  watermelons := 37,
  pineapples := 56,
  mangoes := 0,
  organic_watermelons := 15,
  hydroponic_watermelons := 0,
  dry_season_pineapples := 23,
  vertical_pineapples := 0
}

/-- The fruit growing data for Mark -/
def mark : FruitData := {
  watermelons := 68,
  pineapples := 27,
  mangoes := 0,
  organic_watermelons := 0,
  hydroponic_watermelons := 21,
  dry_season_pineapples := 0,
  vertical_pineapples := 17
}

/-- The fruit growing data for Sandy -/
def sandy : FruitData := {
  watermelons := 11,
  pineapples := 14,
  mangoes := 42,
  organic_watermelons := 0,
  hydroponic_watermelons := 0,
  dry_season_pineapples := 0,
  vertical_pineapples := 0
}

/-- Calculate the total fruits for a person -/
def totalFruits (data : FruitData) : ℕ :=
  data.watermelons + data.pineapples + data.mangoes

/-- Theorem stating the total number of fruits grown by all three people -/
theorem total_fruits_grown :
  totalFruits jason + totalFruits mark + totalFruits sandy = 255 := by
  sorry

/-- Theorem stating the number of watermelons grown using special techniques -/
theorem special_technique_watermelons :
  jason.organic_watermelons + mark.hydroponic_watermelons = 36 := by
  sorry

/-- Theorem stating the number of pineapples grown in dry season or vertically -/
theorem special_pineapples :
  jason.dry_season_pineapples + mark.vertical_pineapples = 40 := by
  sorry

end NUMINAMATH_CALUDE_total_fruits_grown_special_technique_watermelons_special_pineapples_l2733_273332


namespace NUMINAMATH_CALUDE_div_2880_by_smallest_is_square_smaller_divisors_not_square_l2733_273348

/-- The smallest positive integer that divides 2880 and results in a perfect square -/
def smallest_divisor_to_square : ℕ := 10

/-- 2880 divided by the smallest divisor is a perfect square -/
theorem div_2880_by_smallest_is_square :
  ∃ m : ℕ, 2880 / smallest_divisor_to_square = m ^ 2 :=
sorry

/-- For any positive integer smaller than the smallest divisor, 
    dividing 2880 by it does not result in a perfect square -/
theorem smaller_divisors_not_square :
  ∀ k : ℕ, 0 < k → k < smallest_divisor_to_square →
  ¬∃ m : ℕ, 2880 / k = m ^ 2 :=
sorry

end NUMINAMATH_CALUDE_div_2880_by_smallest_is_square_smaller_divisors_not_square_l2733_273348


namespace NUMINAMATH_CALUDE_actual_tissue_diameter_l2733_273338

/-- The actual diameter of a circular tissue given its magnification and magnified image diameter -/
theorem actual_tissue_diameter 
  (magnification : ℝ) 
  (magnified_diameter : ℝ) 
  (h_magnification : magnification = 1000) 
  (h_magnified_diameter : magnified_diameter = 1) : 
  magnified_diameter / magnification = 0.001 := by
  sorry

end NUMINAMATH_CALUDE_actual_tissue_diameter_l2733_273338


namespace NUMINAMATH_CALUDE_jake_has_seven_peaches_l2733_273353

/-- The number of peaches Steven has -/
def steven_peaches : ℕ := 19

/-- The number of peaches Jake has fewer than Steven -/
def jake_fewer_than_steven : ℕ := 12

/-- The number of peaches Jake has more than Jill -/
def jake_more_than_jill : ℕ := 72

/-- The number of peaches Jake has -/
def jake_peaches : ℕ := steven_peaches - jake_fewer_than_steven

theorem jake_has_seven_peaches : jake_peaches = 7 := by
  sorry

end NUMINAMATH_CALUDE_jake_has_seven_peaches_l2733_273353


namespace NUMINAMATH_CALUDE_line_intersects_circle_l2733_273303

-- Define the circle C
def circle_C (x y : ℝ) : Prop := x^2 + y^2 - 4*x = 0

-- Define the point P
def point_P : ℝ × ℝ := (3, 0)

-- Define a line passing through point P
def line_through_P (m : ℝ) (x y : ℝ) : Prop := y = m * (x - point_P.1)

-- Theorem statement
theorem line_intersects_circle :
  ∃ (m : ℝ) (x y : ℝ), line_through_P m x y ∧ circle_C x y :=
sorry

end NUMINAMATH_CALUDE_line_intersects_circle_l2733_273303


namespace NUMINAMATH_CALUDE_change_calculation_l2733_273377

/-- Given an apple cost of $0.75 and a payment of $5, the change returned is $4.25. -/
theorem change_calculation (apple_cost payment : ℚ) (h1 : apple_cost = 0.75) (h2 : payment = 5) :
  payment - apple_cost = 4.25 := by
  sorry

end NUMINAMATH_CALUDE_change_calculation_l2733_273377


namespace NUMINAMATH_CALUDE_penny_draw_probability_l2733_273336

/-- The number of shiny pennies in the box -/
def shiny_pennies : ℕ := 5

/-- The number of dull pennies in the box -/
def dull_pennies : ℕ := 3

/-- The total number of pennies in the box -/
def total_pennies : ℕ := shiny_pennies + dull_pennies

/-- The probability of needing more than five draws to get the fourth shiny penny -/
def probability : ℚ := 31 / 56

theorem penny_draw_probability :
  probability = (Nat.choose 5 3 * Nat.choose 3 1 + Nat.choose 5 0 * Nat.choose 3 3) / Nat.choose total_pennies shiny_pennies ∧
  probability.num + probability.den = 87 := by sorry

end NUMINAMATH_CALUDE_penny_draw_probability_l2733_273336


namespace NUMINAMATH_CALUDE_simplify_sqrt_difference_l2733_273333

theorem simplify_sqrt_difference : (Real.sqrt 300 / Real.sqrt 75) - (Real.sqrt 128 / Real.sqrt 32) = 0 := by
  sorry

end NUMINAMATH_CALUDE_simplify_sqrt_difference_l2733_273333


namespace NUMINAMATH_CALUDE_candy_distribution_proof_l2733_273330

def distribute_candies (total_candies : ℕ) (num_boxes : ℕ) : ℕ :=
  sorry

theorem candy_distribution_proof :
  distribute_candies 10 5 = 7 :=
by sorry

end NUMINAMATH_CALUDE_candy_distribution_proof_l2733_273330


namespace NUMINAMATH_CALUDE_absolute_value_equation_l2733_273373

theorem absolute_value_equation (x : ℝ) :
  |x - 25| + |x - 15| = |2*x - 40| ↔ x ≤ 15 ∨ x ≥ 25 := by
sorry

end NUMINAMATH_CALUDE_absolute_value_equation_l2733_273373


namespace NUMINAMATH_CALUDE_smaller_two_digit_factor_of_2210_l2733_273321

theorem smaller_two_digit_factor_of_2210 (a b : ℕ) : 
  10 ≤ a ∧ a < 100 ∧ 
  10 ≤ b ∧ b < 100 ∧ 
  a * b = 2210 →
  min a b = 26 := by
sorry

end NUMINAMATH_CALUDE_smaller_two_digit_factor_of_2210_l2733_273321


namespace NUMINAMATH_CALUDE_arithmetic_evaluation_l2733_273312

theorem arithmetic_evaluation : 3 * 4^2 - (8 / 2) = 44 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_evaluation_l2733_273312


namespace NUMINAMATH_CALUDE_inscribed_quadrilateral_perimeter_l2733_273324

/-- A rectangle with length twice its width and diagonal 10 units -/
structure SpecialRectangle where
  width : ℝ
  length : ℝ
  diagonal : ℝ
  length_eq : length = 2 * width
  diagonal_eq : diagonal = 10
  pythagoras : diagonal^2 = length^2 + width^2

/-- A quadrilateral inscribed in the special rectangle -/
structure InscribedQuadrilateral (r : SpecialRectangle) where
  perimeter : ℝ
  perimeter_eq : perimeter = 2 * r.length + 2 * r.width

/-- Theorem stating that the perimeter of the inscribed quadrilateral is 12√5 -/
theorem inscribed_quadrilateral_perimeter 
  (r : SpecialRectangle) (q : InscribedQuadrilateral r) : 
  q.perimeter = 12 * Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_inscribed_quadrilateral_perimeter_l2733_273324


namespace NUMINAMATH_CALUDE_zoo_ticket_cost_l2733_273386

theorem zoo_ticket_cost 
  (total_spent : ℚ)
  (family_size : ℕ)
  (adult_ticket_cost : ℚ)
  (adult_tickets : ℕ)
  (h1 : total_spent = 119)
  (h2 : family_size = 7)
  (h3 : adult_ticket_cost = 21)
  (h4 : adult_tickets = 4) :
  let children_tickets := family_size - adult_tickets
  let children_total_cost := total_spent - (adult_ticket_cost * adult_tickets)
  children_total_cost / children_tickets = 35 / 3 :=
sorry

end NUMINAMATH_CALUDE_zoo_ticket_cost_l2733_273386


namespace NUMINAMATH_CALUDE_water_removal_proof_l2733_273367

/-- Represents the fraction of water remaining after n steps -/
def remainingWater (n : ℕ) : ℚ :=
  2 / (n + 2 : ℚ)

/-- The number of steps after which one eighth of the water remains -/
def stepsToOneEighth : ℕ := 14

theorem water_removal_proof :
  remainingWater stepsToOneEighth = 1/8 :=
sorry

end NUMINAMATH_CALUDE_water_removal_proof_l2733_273367


namespace NUMINAMATH_CALUDE_intersection_of_specific_sets_l2733_273371

theorem intersection_of_specific_sets :
  let A : Set ℤ := {1, 2, -3}
  let B : Set ℤ := {1, -4, 5}
  A ∩ B = {1} := by
sorry

end NUMINAMATH_CALUDE_intersection_of_specific_sets_l2733_273371


namespace NUMINAMATH_CALUDE_min_value_reciprocal_sum_l2733_273305

theorem min_value_reciprocal_sum (x y : ℝ) (hx : x > 0) (hy : y > 0) (hsum : x + y = 2) :
  (∀ a b : ℝ, a > 0 → b > 0 → a + b = 2 → 1/a + 4/b ≥ 9/2) ∧
  (∃ a b : ℝ, a > 0 ∧ b > 0 ∧ a + b = 2 ∧ 1/a + 4/b = 9/2) :=
by sorry

end NUMINAMATH_CALUDE_min_value_reciprocal_sum_l2733_273305


namespace NUMINAMATH_CALUDE_simplify_expression_l2733_273364

theorem simplify_expression (s r : ℝ) : 
  (2 * s^2 + 4 * r - 5) - (s^2 + 6 * r - 8) = s^2 - 2 * r + 3 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l2733_273364


namespace NUMINAMATH_CALUDE_area_le_sqrt_product_area_eq_sqrt_product_iff_rectangle_l2733_273352

/-- A quadrilateral circumscribed about a circle -/
structure CircumscribedQuadrilateral where
  a : ℝ
  b : ℝ
  c : ℝ
  d : ℝ
  area : ℝ
  a_pos : 0 < a
  b_pos : 0 < b
  c_pos : 0 < c
  d_pos : 0 < d
  area_pos : 0 < area

/-- The theorem stating that the area of a circumscribed quadrilateral is at most the square root of the product of its side lengths -/
theorem area_le_sqrt_product (q : CircumscribedQuadrilateral) : 
  q.area ≤ Real.sqrt (q.a * q.b * q.c * q.d) := by
  sorry

/-- The condition for equality in the above inequality -/
def is_rectangle (q : CircumscribedQuadrilateral) : Prop :=
  (q.a = q.c ∧ q.b = q.d) ∨ (q.a = q.b ∧ q.c = q.d)

/-- The theorem stating that equality holds if and only if the quadrilateral is a rectangle -/
theorem area_eq_sqrt_product_iff_rectangle (q : CircumscribedQuadrilateral) :
  q.area = Real.sqrt (q.a * q.b * q.c * q.d) ↔ is_rectangle q := by
  sorry

end NUMINAMATH_CALUDE_area_le_sqrt_product_area_eq_sqrt_product_iff_rectangle_l2733_273352


namespace NUMINAMATH_CALUDE_log_relation_l2733_273302

theorem log_relation (a b : ℝ) : 
  a = Real.log 343 / Real.log 6 → 
  b = Real.log 18 / Real.log 7 → 
  a = 6 / (b + 2 * Real.log 2 / Real.log 7) := by
sorry

end NUMINAMATH_CALUDE_log_relation_l2733_273302


namespace NUMINAMATH_CALUDE_least_subtraction_for_divisibility_l2733_273378

theorem least_subtraction_for_divisibility (n m : ℕ) (hn : n > 0) (hm : m > 0) : 
  ∃ k, k ≥ 0 ∧ k < m ∧ (n ^ 1000 - k) % m = 0 ∧ 
  ∀ j, 0 ≤ j ∧ j < k → (n ^ 1000 - j) % m ≠ 0 :=
by sorry

#check least_subtraction_for_divisibility 10 97

end NUMINAMATH_CALUDE_least_subtraction_for_divisibility_l2733_273378


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_l2733_273316

theorem quadratic_inequality_solution (m : ℝ) : 
  (∀ x : ℝ, mx^2 + 8*m*x + 60 < 0 ↔ -5 < x ∧ x < -3) → m = 4 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_l2733_273316


namespace NUMINAMATH_CALUDE_monotonic_f_iff_a_range_l2733_273379

-- Define the function f(x)
def f (a : ℝ) (x : ℝ) : ℝ := x^3 + x^2 - a*x + 1

-- Define monotonically increasing
def monotonically_increasing (f : ℝ → ℝ) : Prop :=
  ∀ x y, x < y → f x < f y

-- Theorem statement
theorem monotonic_f_iff_a_range :
  ∀ a : ℝ, (monotonically_increasing (f a)) ↔ a ≤ -1/3 :=
sorry

end NUMINAMATH_CALUDE_monotonic_f_iff_a_range_l2733_273379


namespace NUMINAMATH_CALUDE_smallest_multiple_of_45_and_75_not_20_l2733_273322

theorem smallest_multiple_of_45_and_75_not_20 :
  ∃ n : ℕ, n > 0 ∧ 45 ∣ n ∧ 75 ∣ n ∧ ¬(20 ∣ n) ∧
  ∀ m : ℕ, m > 0 ∧ 45 ∣ m ∧ 75 ∣ m ∧ ¬(20 ∣ m) → n ≤ m :=
by sorry

end NUMINAMATH_CALUDE_smallest_multiple_of_45_and_75_not_20_l2733_273322


namespace NUMINAMATH_CALUDE_equation_solution_l2733_273314

theorem equation_solution : 
  ∀ x : ℝ, (3*x - 1)*(2*x + 4) = 1 ↔ x = (-5 + Real.sqrt 55) / 6 ∨ x = (-5 - Real.sqrt 55) / 6 := by
sorry

end NUMINAMATH_CALUDE_equation_solution_l2733_273314


namespace NUMINAMATH_CALUDE_games_missed_l2733_273385

theorem games_missed (total_games attended_games : ℕ) (h1 : total_games = 31) (h2 : attended_games = 13) :
  total_games - attended_games = 18 := by
  sorry

end NUMINAMATH_CALUDE_games_missed_l2733_273385


namespace NUMINAMATH_CALUDE_find_x_l2733_273326

theorem find_x (x y : ℝ) (h1 : x - y = 10) (h2 : x + y = 14) (h3 : x > y) : x = 12 := by
  sorry

end NUMINAMATH_CALUDE_find_x_l2733_273326


namespace NUMINAMATH_CALUDE_min_distance_to_line_l2733_273380

/-- The minimum distance from the origin to a point on the line x + y - 4 = 0 is 2√2 -/
theorem min_distance_to_line : 
  let line := {p : ℝ × ℝ | p.1 + p.2 = 4}
  ∃ d : ℝ, d = 2 * Real.sqrt 2 ∧ 
    ∀ p ∈ line, Real.sqrt (p.1^2 + p.2^2) ≥ d ∧
    ∃ q ∈ line, Real.sqrt (q.1^2 + q.2^2) = d :=
by sorry

end NUMINAMATH_CALUDE_min_distance_to_line_l2733_273380


namespace NUMINAMATH_CALUDE_marble_count_l2733_273346

theorem marble_count (r : ℝ) (b g y : ℝ) : 
  r > 0 →
  b = r / 1.3 →
  g = 1.5 * r →
  y = 1.2 * g →
  r + b + g + y = 5.069 * r := by
sorry

end NUMINAMATH_CALUDE_marble_count_l2733_273346


namespace NUMINAMATH_CALUDE_unique_kids_count_l2733_273381

/-- The number of unique kids Julia played with across the week -/
def total_unique_kids (monday tuesday wednesday thursday friday : ℕ) 
  (wednesday_from_monday thursday_from_tuesday friday_from_monday friday_from_wednesday : ℕ) : ℕ :=
  monday + tuesday + (wednesday - wednesday_from_monday) + 
  (thursday - thursday_from_tuesday) + 
  (friday - friday_from_monday - (friday_from_wednesday - wednesday_from_monday))

theorem unique_kids_count :
  let monday := 12
  let tuesday := 7
  let wednesday := 15
  let thursday := 10
  let friday := 18
  let wednesday_from_monday := 5
  let thursday_from_tuesday := 7
  let friday_from_monday := 9
  let friday_from_wednesday := 5
  total_unique_kids monday tuesday wednesday thursday friday
    wednesday_from_monday thursday_from_tuesday friday_from_monday friday_from_wednesday = 36 := by
  sorry

end NUMINAMATH_CALUDE_unique_kids_count_l2733_273381
