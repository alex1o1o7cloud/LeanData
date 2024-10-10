import Mathlib

namespace special_rectangle_area_l2057_205770

/-- Represents a rectangle divided into 6 squares with specific properties -/
structure SpecialRectangle where
  /-- The side length of the smallest square -/
  smallest_side : ℝ
  /-- The side length of the D square -/
  d_side : ℝ
  /-- Condition: The smallest square has an area of 4 square centimeters -/
  smallest_area : smallest_side ^ 2 = 4
  /-- Condition: The side lengths increase incrementally by 2 centimeters -/
  incremental_increase : d_side = smallest_side + 6

/-- The theorem stating the area of the special rectangle -/
theorem special_rectangle_area (r : SpecialRectangle) : 
  (2 * r.d_side + (r.d_side + 2)) * (r.d_side + 2 + (r.d_side + 4)) = 572 := by
  sorry

end special_rectangle_area_l2057_205770


namespace find_m_l2057_205700

theorem find_m (a : ℝ) (n m : ℕ) (h1 : a^n = 2) (h2 : a^(m*n) = 16) : m = 4 := by
  sorry

end find_m_l2057_205700


namespace contractor_absence_solution_l2057_205765

/-- Represents the problem of calculating a contractor's absence days --/
def ContractorAbsenceProblem (total_days : ℕ) (daily_pay : ℚ) (daily_fine : ℚ) (total_received : ℚ) : Prop :=
  ∃ (absent_days : ℕ),
    (absent_days ≤ total_days) ∧
    (daily_pay * (total_days - absent_days : ℚ) - daily_fine * (absent_days : ℚ) = total_received)

/-- Theorem stating the solution to the contractor absence problem --/
theorem contractor_absence_solution :
  ContractorAbsenceProblem 30 25 7.5 425 → ∃ (absent_days : ℕ), absent_days = 10 := by
  sorry

#check contractor_absence_solution

end contractor_absence_solution_l2057_205765


namespace car_purchase_cost_difference_l2057_205775

/-- Calculates the difference in individual cost for buying a car when the group size changes --/
theorem car_purchase_cost_difference 
  (base_cost : ℕ) 
  (discount_per_person : ℕ) 
  (car_wash_earnings : ℕ) 
  (original_group_size : ℕ) 
  (new_group_size : ℕ) : 
  base_cost = 1700 →
  discount_per_person = 50 →
  car_wash_earnings = 500 →
  original_group_size = 6 →
  new_group_size = 5 →
  (base_cost - new_group_size * discount_per_person - car_wash_earnings) / new_group_size -
  (base_cost - original_group_size * discount_per_person - car_wash_earnings) / original_group_size = 40 := by
  sorry


end car_purchase_cost_difference_l2057_205775


namespace sum_of_odd_naturals_900_l2057_205740

theorem sum_of_odd_naturals_900 :
  ∃ n : ℕ, n^2 = 900 ∧ (∀ k : ℕ, k ≤ n → (2*k - 1) ≤ n^2) :=
by sorry

end sum_of_odd_naturals_900_l2057_205740


namespace negation_of_universal_proposition_l2057_205779

theorem negation_of_universal_proposition :
  (¬ (∀ a : ℝ, a > 0 → Real.exp a ≥ 1)) ↔ (∃ a : ℝ, a > 0 ∧ Real.exp a < 1) :=
by sorry

end negation_of_universal_proposition_l2057_205779


namespace least_k_for_inequality_l2057_205747

theorem least_k_for_inequality (k : ℤ) : 
  (∀ n : ℤ, n < k → (0.0010101 : ℝ) * (10 : ℝ) ^ (n : ℝ) ≤ 100) ∧ 
  (0.0010101 : ℝ) * (10 : ℝ) ^ (k : ℝ) > 100 → 
  k = 6 :=
sorry

end least_k_for_inequality_l2057_205747


namespace at_least_four_2x2_squares_sum_greater_than_100_l2057_205774

/-- Represents a square on the 8x8 board -/
structure Square where
  row : Fin 8
  col : Fin 8

/-- Represents the board configuration -/
def Board := Square → Fin 64

/-- Checks if a given 2x2 square has a sum greater than 100 -/
def is_sum_greater_than_100 (board : Board) (top_left : Square) : Prop :=
  let sum := (board top_left).val + 
              (board ⟨top_left.row, top_left.col.succ⟩).val + 
              (board ⟨top_left.row.succ, top_left.col⟩).val + 
              (board ⟨top_left.row.succ, top_left.col.succ⟩).val
  sum > 100

/-- The main theorem to be proved -/
theorem at_least_four_2x2_squares_sum_greater_than_100 (board : Board) 
  (h_unique : ∀ (s1 s2 : Square), board s1 = board s2 → s1 = s2) :
  ∃ (s1 s2 s3 s4 : Square), 
    s1 ≠ s2 ∧ s1 ≠ s3 ∧ s1 ≠ s4 ∧ s2 ≠ s3 ∧ s2 ≠ s4 ∧ s3 ≠ s4 ∧
    is_sum_greater_than_100 board s1 ∧
    is_sum_greater_than_100 board s2 ∧
    is_sum_greater_than_100 board s3 ∧
    is_sum_greater_than_100 board s4 :=
  sorry

end at_least_four_2x2_squares_sum_greater_than_100_l2057_205774


namespace binary_addition_theorem_l2057_205742

/-- Converts a binary number represented as a list of bits to its decimal equivalent -/
def binary_to_decimal (bits : List Bool) : Nat :=
  bits.foldl (fun acc b => 2 * acc + if b then 1 else 0) 0

/-- Converts a decimal number to its binary representation as a list of bits -/
def decimal_to_binary (n : Nat) : List Bool :=
  if n = 0 then [false] else
  let rec to_binary_aux (m : Nat) : List Bool :=
    if m = 0 then [] else (m % 2 = 1) :: to_binary_aux (m / 2)
  to_binary_aux n

theorem binary_addition_theorem :
  let a := [true, false, true]  -- 101₂
  let b := [true, true]         -- 11₂
  let c := [false, false, true, true]  -- 1100₂
  let d := [true, false, true, true, true]  -- 11101₂
  let result := [true, false, false, false, false, true, true]  -- 110001₂
  binary_to_decimal a + binary_to_decimal b + binary_to_decimal c + binary_to_decimal d =
  binary_to_decimal result := by
  sorry

end binary_addition_theorem_l2057_205742


namespace function_properties_l2057_205702

def is_even (f : ℝ → ℝ) : Prop := ∀ x, f x = f (-x)

def periodic (f : ℝ → ℝ) (p : ℝ) : Prop := ∀ x, f (x + p) = f x

def symmetric_about (f : ℝ → ℝ) (a : ℝ) : Prop := ∀ x, f (a + x) = f (a - x)

def increasing_on (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x y, a ≤ x ∧ x < y ∧ y ≤ b → f x < f y

theorem function_properties (f : ℝ → ℝ) 
  (h1 : is_even f)
  (h2 : ∀ x, f (x + 1) = -f x)
  (h3 : increasing_on f (-1) 0) :
  (periodic f 2) ∧ 
  (symmetric_about f 1) ∧ 
  (f 2 = f 0) := by
  sorry

end function_properties_l2057_205702


namespace max_value_of_fraction_difference_l2057_205716

theorem max_value_of_fraction_difference (a b : ℝ) 
  (ha : a > 0) (hb : b > 0) (h : 4 * a - b ≥ 2) :
  (∀ x y : ℝ, x > 0 → y > 0 → 4 * x - y ≥ 2 → 1 / x - 1 / y ≤ 1 / 2) ∧
  (∃ x y : ℝ, x > 0 ∧ y > 0 ∧ 4 * x - y ≥ 2 ∧ 1 / x - 1 / y = 1 / 2) :=
by sorry

end max_value_of_fraction_difference_l2057_205716


namespace intersection_complement_eq_l2057_205712

open Set

def U : Set ℝ := univ
def A : Set ℝ := {-2, -1, 0, 1, 2}
def B : Set ℝ := {x | x ≥ 2}

theorem intersection_complement_eq : A ∩ (U \ B) = {-2, -1, 0, 1} := by sorry

end intersection_complement_eq_l2057_205712


namespace distinct_values_count_l2057_205763

def parenthesization1 : ℕ := 3^(3^(3^3))
def parenthesization2 : ℕ := 3^((3^3)^3)
def parenthesization3 : ℕ := ((3^3)^3)^3
def parenthesization4 : ℕ := 3^((3^3)^(3^2))

def distinctValues : Finset ℕ := {parenthesization1, parenthesization2, parenthesization3, parenthesization4}

theorem distinct_values_count :
  Finset.card distinctValues = 3 := by sorry

end distinct_values_count_l2057_205763


namespace sum_product_bound_l2057_205710

theorem sum_product_bound (a b c : ℝ) (h : a^2 + b^2 + c^2 = 1) :
  -1/2 ≤ a*b + b*c + c*a ∧ a*b + b*c + c*a ≤ 1 := by
  sorry

end sum_product_bound_l2057_205710


namespace two_pi_irrational_l2057_205777

theorem two_pi_irrational : Irrational (2 * Real.pi) := by
  sorry

end two_pi_irrational_l2057_205777


namespace unique_operation_equals_one_l2057_205709

theorem unique_operation_equals_one :
  ((-3 + (-3) = 1) = False) ∧
  ((-3 - (-3) = 1) = False) ∧
  ((-3 / (-3) = 1) = True) ∧
  ((-3 * (-3) = 1) = False) := by
  sorry

end unique_operation_equals_one_l2057_205709


namespace frank_defeated_six_enemies_l2057_205724

/-- The number of enemies Frank defeated in the game --/
def enemies_defeated : ℕ := sorry

/-- The points earned per enemy defeated --/
def points_per_enemy : ℕ := 9

/-- The bonus points for completing the level --/
def bonus_points : ℕ := 8

/-- The total points Frank earned --/
def total_points : ℕ := 62

/-- Theorem stating that Frank defeated 6 enemies --/
theorem frank_defeated_six_enemies :
  enemies_defeated = 6 ∧
  enemies_defeated * points_per_enemy + bonus_points = total_points :=
sorry

end frank_defeated_six_enemies_l2057_205724


namespace tennis_ball_storage_l2057_205726

theorem tennis_ball_storage (n : ℕ) : n = 105 ↔ 
  (n % 25 = 5 ∧ n % 20 = 5 ∧ ∀ m : ℕ, m < n → (m % 25 ≠ 5 ∨ m % 20 ≠ 5)) :=
by sorry

end tennis_ball_storage_l2057_205726


namespace jacob_and_nathan_letters_l2057_205772

/-- The number of letters Nathan can write in one hour -/
def nathan_letters_per_hour : ℕ := 25

/-- Jacob's writing speed relative to Nathan's -/
def jacob_speed_multiplier : ℕ := 2

/-- The number of hours Jacob and Nathan work together -/
def total_hours : ℕ := 10

/-- Theorem: Jacob and Nathan can write 750 letters in 10 hours together -/
theorem jacob_and_nathan_letters : 
  (nathan_letters_per_hour + jacob_speed_multiplier * nathan_letters_per_hour) * total_hours = 750 := by
  sorry

end jacob_and_nathan_letters_l2057_205772


namespace acute_triangle_contains_grid_point_l2057_205799

/-- Represents a point on a 2D grid -/
structure GridPoint where
  x : ℤ
  y : ℤ

/-- Represents a triangle on a 2D grid -/
structure GridTriangle where
  A : GridPoint
  B : GridPoint
  C : GridPoint

/-- Checks if a triangle is acute -/
def isAcute (t : GridTriangle) : Prop := sorry

/-- Checks if a point is inside or on the sides of a triangle -/
def isInsideOrOnSides (p : GridPoint) (t : GridTriangle) : Prop := sorry

/-- Main theorem: If a triangle on a grid is acute, there exists a grid point 
    (other than its vertices) inside or on its sides -/
theorem acute_triangle_contains_grid_point (t : GridTriangle) :
  isAcute t → ∃ p : GridPoint, p ≠ t.A ∧ p ≠ t.B ∧ p ≠ t.C ∧ isInsideOrOnSides p t := by
  sorry

end acute_triangle_contains_grid_point_l2057_205799


namespace passion_fruit_crates_l2057_205798

theorem passion_fruit_crates (total grapes mangoes : ℕ) 
  (h1 : total = 50)
  (h2 : grapes = 13)
  (h3 : mangoes = 20) :
  total - (grapes + mangoes) = 17 := by
  sorry

end passion_fruit_crates_l2057_205798


namespace contrapositive_equivalence_l2057_205773

theorem contrapositive_equivalence (a b m : ℝ) :
  (¬(a > b → a * (m^2 + 1) > b * (m^2 + 1))) ↔ (a * (m^2 + 1) ≤ b * (m^2 + 1) → a ≤ b) :=
by sorry

end contrapositive_equivalence_l2057_205773


namespace furniture_shop_cost_price_l2057_205750

theorem furniture_shop_cost_price (selling_price : ℕ) (markup_percentage : ℕ) : 
  selling_price = 1000 → markup_percentage = 100 → 
  ∃ (cost_price : ℕ), cost_price * (100 + markup_percentage) / 100 = selling_price ∧ cost_price = 500 := by
sorry

end furniture_shop_cost_price_l2057_205750


namespace square_side_lengths_average_l2057_205737

theorem square_side_lengths_average (a₁ a₂ a₃ : ℝ) (h₁ : a₁ = 25) (h₂ : a₂ = 36) (h₃ : a₃ = 64) :
  (Real.sqrt a₁ + Real.sqrt a₂ + Real.sqrt a₃) / 3 = 19 / 3 := by
  sorry

end square_side_lengths_average_l2057_205737


namespace largest_integer_in_sequence_l2057_205749

theorem largest_integer_in_sequence (n : ℕ) (start : ℤ) (h1 : n = 40) (h2 : start = -11) :
  (start + (n - 1) : ℤ) = 28 := by
  sorry

end largest_integer_in_sequence_l2057_205749


namespace num_men_is_seven_l2057_205753

/-- Represents the amount of work a person can do per hour -/
structure WorkRate where
  amount : ℝ

/-- The number of men working with 2 boys -/
def numMen : ℕ := sorry

/-- The work rate of a man -/
def manWorkRate : WorkRate := sorry

/-- The work rate of a boy -/
def boyWorkRate : WorkRate := sorry

/-- The ratio of work done by a man to a boy is 4:1 -/
axiom work_ratio : manWorkRate.amount = 4 * boyWorkRate.amount

/-- The group (numMen men and 2 boys) can do 6 times as much work per hour as a man and a boy together -/
axiom group_work_rate : 
  numMen * manWorkRate.amount + 2 * boyWorkRate.amount = 
  6 * (manWorkRate.amount + boyWorkRate.amount)

theorem num_men_is_seven : numMen = 7 := by sorry

end num_men_is_seven_l2057_205753


namespace lines_intersection_l2057_205780

def line1 (t : ℝ) : ℝ × ℝ := (1 + 2*t, 2 - 3*t)
def line2 (u : ℝ) : ℝ × ℝ := (-1 + 3*u, 4 + u)

theorem lines_intersection :
  ∃! p : ℝ × ℝ, (∃ t : ℝ, line1 t = p) ∧ (∃ u : ℝ, line2 u = p) :=
  by
    use (-5/11, 46/11)
    sorry

#check lines_intersection

end lines_intersection_l2057_205780


namespace glass_count_l2057_205796

/-- Given glasses with a capacity of 6 ounces that are 4/5 full, 
    prove that if 12 ounces of water are needed to fill all glasses, 
    there are 10 glasses. -/
theorem glass_count (glass_capacity : ℚ) (initial_fill : ℚ) (total_water_needed : ℚ) :
  glass_capacity = 6 →
  initial_fill = 4 / 5 →
  total_water_needed = 12 →
  (total_water_needed / (glass_capacity * (1 - initial_fill))) = 10 := by
  sorry


end glass_count_l2057_205796


namespace expression_simplification_l2057_205705

theorem expression_simplification (x y : ℚ) 
  (hx : x = -1/2) (hy : y = -3) : 
  3 * (x^2 - 2*x*y) - (3*x^2 - 2*y + 2*(x*y + y)) = -12 := by
  sorry

end expression_simplification_l2057_205705


namespace isosceles_triangle_exists_l2057_205762

-- Define a circle
def Circle : Type := Unit

-- Define a color
inductive Color
| Red
| Blue

-- Define a point on the circle
structure Point (c : Circle) where
  color : Color

-- Define a coloring of the circle
def Coloring (c : Circle) := Point c → Color

-- Define an isosceles triangle
structure IsoscelesTriangle (c : Circle) where
  a : Point c
  b : Point c
  c : Point c
  isIsosceles : True  -- Placeholder for the isosceles property

-- Theorem statement
theorem isosceles_triangle_exists (c : Circle) (coloring : Coloring c) :
  ∃ (t : IsoscelesTriangle c), t.a.color = t.b.color ∧ t.b.color = t.c.color :=
sorry

end isosceles_triangle_exists_l2057_205762


namespace tysons_ocean_speed_l2057_205746

/-- Tyson's swimming speed problem -/
theorem tysons_ocean_speed (lake_speed : ℝ) (total_races : ℕ) (race_distance : ℝ) (total_time : ℝ) :
  lake_speed = 3 →
  total_races = 10 →
  race_distance = 3 →
  total_time = 11 →
  ∃ (ocean_speed : ℝ),
    ocean_speed = 2.5 ∧
    (lake_speed * (total_races / 2 * race_distance) + ocean_speed * (total_races / 2 * race_distance)) / total_races = race_distance / (total_time / total_races) :=
by sorry

end tysons_ocean_speed_l2057_205746


namespace consecutive_odd_integers_sum_l2057_205776

theorem consecutive_odd_integers_sum (a : ℤ) : 
  (a % 2 = 1) →                 -- a is odd
  (a + (a + 4) = 150) →         -- sum of first and third is 150
  (a + (a + 2) + (a + 4) = 225) -- sum of all three is 225
  := by sorry

end consecutive_odd_integers_sum_l2057_205776


namespace system_unique_solution_l2057_205794

/-- The system of equations has a unique solution -/
theorem system_unique_solution :
  ∃! (x₁ x₂ x₃ : ℝ),
    3 * x₁ + 4 * x₂ + 3 * x₃ = 0 ∧
    x₁ - x₂ + x₃ = 0 ∧
    x₁ + 3 * x₂ - x₃ = -2 ∧
    x₁ + 2 * x₂ + 3 * x₃ = 2 ∧
    x₁ = 1 ∧ x₂ = 0 ∧ x₃ = 1 := by
  sorry


end system_unique_solution_l2057_205794


namespace inequality_proof_l2057_205725

theorem inequality_proof (x y z : ℝ) (hx : x ∈ Set.Icc 0 1) (hy : y ∈ Set.Icc 0 1) (hz : z ∈ Set.Icc 0 1) :
  (x / (y + z + 1)) + (y / (z + x + 1)) + (z / (x + y + 1)) ≤ 1 - (1 - x) * (1 - y) * (1 - z) := by
sorry

end inequality_proof_l2057_205725


namespace constant_function_from_parallel_tangent_l2057_205745

-- Define a function f: ℝ → ℝ
variable (f : ℝ → ℝ)

-- Define the property that the tangent line is parallel to the x-axis at every point
def tangent_parallel_to_x_axis (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, deriv f x = 0

-- Theorem statement
theorem constant_function_from_parallel_tangent :
  tangent_parallel_to_x_axis f → ∃ c : ℝ, ∀ x : ℝ, f x = c :=
by
  sorry

end constant_function_from_parallel_tangent_l2057_205745


namespace late_condition_early_condition_60_early_condition_50_exact_time_speed_correct_l2057_205767

/-- Represents the speed Mr. Bird needs to drive to arrive exactly on time -/
def exact_time_speed : ℚ :=
  160 / 3

/-- Represents the distance to Mr. Bird's workplace in miles -/
def distance_to_work : ℚ :=
  40 / 3

/-- Represents the ideal time to reach work on time in hours -/
def ideal_time : ℚ :=
  1 / 4

/-- Given that driving at 40 mph makes Mr. Bird 5 minutes late -/
theorem late_condition (speed : ℚ) (time : ℚ) :
  speed = 40 → time = ideal_time + 5 / 60 → speed * time = distance_to_work :=
sorry

/-- Given that driving at 60 mph makes Mr. Bird 2 minutes early -/
theorem early_condition_60 (speed : ℚ) (time : ℚ) :
  speed = 60 → time = ideal_time - 2 / 60 → speed * time = distance_to_work :=
sorry

/-- Given that driving at 50 mph makes Mr. Bird 1 minute early -/
theorem early_condition_50 (speed : ℚ) (time : ℚ) :
  speed = 50 → time = ideal_time - 1 / 60 → speed * time = distance_to_work :=
sorry

/-- Theorem stating that the exact_time_speed is the speed required to arrive exactly on time -/
theorem exact_time_speed_correct :
  exact_time_speed * ideal_time = distance_to_work :=
sorry

end late_condition_early_condition_60_early_condition_50_exact_time_speed_correct_l2057_205767


namespace shortest_distance_parabola_to_line_l2057_205761

/-- The shortest distance from a point on the parabola y = x^2 to the line 2x - y = 4 -/
theorem shortest_distance_parabola_to_line :
  let parabola := {p : ℝ × ℝ | p.2 = p.1^2}
  let line := {p : ℝ × ℝ | 2 * p.1 - p.2 = 4}
  let distance (p : ℝ × ℝ) := |2 * p.1 - p.2 - 4| / Real.sqrt 5
  (∀ p ∈ parabola, distance p ≥ 3 * Real.sqrt 5 / 5) ∧
  (∃ p ∈ parabola, distance p = 3 * Real.sqrt 5 / 5) :=
by sorry


end shortest_distance_parabola_to_line_l2057_205761


namespace triangle_parallelogram_relation_l2057_205739

theorem triangle_parallelogram_relation (triangle_area : ℝ) (parallelogram_height : ℝ) : 
  triangle_area = 15 → parallelogram_height = 5 → 
  ∃ (parallelogram_area parallelogram_base : ℝ),
    parallelogram_area = 2 * triangle_area ∧
    parallelogram_area = parallelogram_height * parallelogram_base ∧
    parallelogram_area = 30 ∧
    parallelogram_base = 6 := by
  sorry

end triangle_parallelogram_relation_l2057_205739


namespace P_in_fourth_quadrant_l2057_205758

/-- A point in the Cartesian coordinate system -/
structure CartesianPoint where
  x : ℝ
  y : ℝ

/-- Definition of the fourth quadrant -/
def is_in_fourth_quadrant (p : CartesianPoint) : Prop :=
  p.x > 0 ∧ p.y < 0

/-- The given point P -/
def P : CartesianPoint :=
  { x := 1, y := -4 }

/-- Theorem: P lies in the fourth quadrant -/
theorem P_in_fourth_quadrant : is_in_fourth_quadrant P := by
  sorry

end P_in_fourth_quadrant_l2057_205758


namespace last_triangle_perimeter_l2057_205701

/-- Represents a triangle with side lengths a, b, and c -/
structure Triangle where
  a : ℚ
  b : ℚ
  c : ℚ

/-- Checks if the given side lengths form a valid triangle -/
def Triangle.isValid (t : Triangle) : Prop :=
  t.a + t.b > t.c ∧ t.b + t.c > t.a ∧ t.c + t.a > t.b

/-- Generates the next triangle in the sequence -/
def nextTriangle (t : Triangle) : Triangle :=
  { a := (t.b + t.c - t.a) / 2,
    b := (t.c + t.a - t.b) / 2,
    c := (t.a + t.b - t.c) / 2 }

/-- The initial triangle T₁ -/
def T₁ : Triangle := { a := 401, b := 403, c := 405 }

/-- The sequence of triangles -/
def triangleSequence : ℕ → Triangle
  | 0 => T₁
  | n + 1 => nextTriangle (triangleSequence n)

/-- The perimeter of a triangle -/
def Triangle.perimeter (t : Triangle) : ℚ := t.a + t.b + t.c

theorem last_triangle_perimeter :
  ∃ n : ℕ, 
    (Triangle.isValid (triangleSequence n)) ∧ 
    ¬(Triangle.isValid (triangleSequence (n + 1))) ∧
    (Triangle.perimeter (triangleSequence n) = 1209 / 512) := by
  sorry

#check last_triangle_perimeter

end last_triangle_perimeter_l2057_205701


namespace fraction_inequality_l2057_205733

theorem fraction_inequality (a b c d : ℝ) 
  (h1 : a > b) (h2 : b > 0) (h3 : c > d) (h4 : d > 0) : 
  a / d > b / c := by
  sorry

end fraction_inequality_l2057_205733


namespace smallest_multiple_of_3_5_7_9_l2057_205769

theorem smallest_multiple_of_3_5_7_9 (n : ℕ) :
  (∀ m : ℕ, m > 0 ∧ 3 ∣ m ∧ 5 ∣ m ∧ 7 ∣ m ∧ 9 ∣ m → n ≤ m) ↔ n = 315 :=
by sorry

end smallest_multiple_of_3_5_7_9_l2057_205769


namespace difference_of_squares_102_99_l2057_205727

theorem difference_of_squares_102_99 : 102^2 - 99^2 = 603 := by
  sorry

end difference_of_squares_102_99_l2057_205727


namespace increasing_quadratic_condition_l2057_205722

/-- A function f is increasing on an interval [a, +∞) if for any x₁, x₂ in the interval with x₁ < x₂, we have f(x₁) < f(x₂) -/
def IncreasingOn (f : ℝ → ℝ) (a : ℝ) :=
  ∀ x₁ x₂, a ≤ x₁ ∧ x₁ < x₂ → f x₁ < f x₂

theorem increasing_quadratic_condition (a : ℝ) :
  (IncreasingOn (fun x => x^2 + 2*(a-1)*x + 2) 4) → a ≥ -3 :=
by sorry

end increasing_quadratic_condition_l2057_205722


namespace M_equals_singleton_l2057_205720

def M : Set (ℝ × ℝ) := {p | 2 * p.1 + p.2 = 2 ∧ p.1 - p.2 = 1}

theorem M_equals_singleton : M = {(1, 0)} := by sorry

end M_equals_singleton_l2057_205720


namespace tower_count_mod_1000_l2057_205721

/-- Represents the number of towers that can be built with cubes of sizes 1 to n -/
def T : ℕ → ℕ
  | 0 => 0
  | 1 => 1
  | 2 => 2
  | n + 1 => 4 * T n

/-- The main theorem stating that the number of towers with 9 cubes is congruent to 768 mod 1000 -/
theorem tower_count_mod_1000 : T 9 ≡ 768 [MOD 1000] := by sorry

end tower_count_mod_1000_l2057_205721


namespace probability_inner_circle_l2057_205719

theorem probability_inner_circle (R : ℝ) (r : ℝ) (h1 : R = 6) (h2 : r = 2) :
  (π * r^2) / (π * R^2) = 1 / 9 :=
sorry

end probability_inner_circle_l2057_205719


namespace semicircle_perimeter_l2057_205771

/-- The perimeter of a semicircle with radius 6.3 cm is equal to π * 6.3 + 2 * 6.3 cm -/
theorem semicircle_perimeter :
  let r : ℝ := 6.3
  (π * r + 2 * r) = (π * 6.3 + 2 * 6.3) :=
by sorry

end semicircle_perimeter_l2057_205771


namespace oil_tank_capacity_l2057_205757

theorem oil_tank_capacity (t : ℝ) (h1 : t > 0) : 
  (1/4 : ℝ) * t + 6 = (1/3 : ℝ) * t → t = 72 := by
  sorry

end oil_tank_capacity_l2057_205757


namespace fifteenth_term_of_sequence_l2057_205760

def arithmetic_sequence (a₁ : ℝ) (d : ℝ) (n : ℕ) : ℝ := a₁ + (n - 1 : ℝ) * d

theorem fifteenth_term_of_sequence (a₁ a₂ : ℝ) (h : a₂ = a₁ + 5) :
  arithmetic_sequence a₁ (a₂ - a₁) 15 = 74 :=
by sorry

end fifteenth_term_of_sequence_l2057_205760


namespace permutation_product_difference_divisible_l2057_205759

def is_permutation (s : Fin 2016 → ℕ) : Prop :=
  ∀ n : ℕ, n ≥ 1 ∧ n ≤ 2016 → ∃ i : Fin 2016, s i = n

theorem permutation_product_difference_divisible
  (a b : Fin 2016 → ℕ)
  (ha : is_permutation a)
  (hb : is_permutation b) :
  ∃ i j : Fin 2016, i ≠ j ∧ (2017 ∣ a i * b i - a j * b j) :=
sorry

end permutation_product_difference_divisible_l2057_205759


namespace hvac_cost_per_vent_l2057_205756

/-- Calculates the cost per vent of an HVAC system -/
def cost_per_vent (total_cost : ℕ) (num_zones : ℕ) (vents_per_zone : ℕ) : ℕ :=
  total_cost / (num_zones * vents_per_zone)

/-- Proves that the cost per vent of the given HVAC system is $2,000 -/
theorem hvac_cost_per_vent :
  cost_per_vent 20000 2 5 = 2000 := by
  sorry

#eval cost_per_vent 20000 2 5

end hvac_cost_per_vent_l2057_205756


namespace journey_distance_l2057_205764

theorem journey_distance (speed1 speed2 time1 total_time : ℝ) 
  (h1 : speed1 = 20)
  (h2 : speed2 = 70)
  (h3 : time1 = 3.2)
  (h4 : total_time = 8) :
  speed1 * time1 + speed2 * (total_time - time1) = 400 := by
  sorry

end journey_distance_l2057_205764


namespace defective_units_shipped_percentage_l2057_205792

theorem defective_units_shipped_percentage
  (total_units : ℝ)
  (defective_percentage : ℝ)
  (defective_shipped_percentage : ℝ)
  (h1 : defective_percentage = 5)
  (h2 : defective_shipped_percentage = 0.2)
  : (defective_shipped_percentage * total_units) / (defective_percentage * total_units) * 100 = 4 := by
  sorry

end defective_units_shipped_percentage_l2057_205792


namespace initial_persimmons_l2057_205728

/-- The number of persimmons eaten -/
def eaten : ℕ := 5

/-- The number of persimmons left -/
def left : ℕ := 12

/-- The initial number of persimmons -/
def initial : ℕ := eaten + left

theorem initial_persimmons : initial = 17 := by
  sorry

end initial_persimmons_l2057_205728


namespace right_triangle_area_l2057_205781

/-- The area of a right triangle with a leg of 28 inches and a hypotenuse of 30 inches is 28√29 square inches. -/
theorem right_triangle_area (a b c : ℝ) (h1 : a = 28) (h2 : c = 30) (h3 : a^2 + b^2 = c^2) :
  (1/2) * a * b = 28 * Real.sqrt 29 := by
  sorry

end right_triangle_area_l2057_205781


namespace regular_quad_pyramid_theorem_l2057_205778

/-- A regular quadrilateral pyramid with a plane drawn through the diagonal of the base and the height -/
structure RegularQuadPyramid where
  /-- The ratio of the area of the cross-section to the lateral surface -/
  k : ℝ
  /-- The ratio k is positive -/
  k_pos : k > 0

/-- The cosine of the angle between slant heights of opposite lateral faces -/
def slant_height_angle_cos (p : RegularQuadPyramid) : ℝ := 16 * p.k^2 - 1

/-- The theorem stating the cosine of the angle between slant heights and the permissible values of k -/
theorem regular_quad_pyramid_theorem (p : RegularQuadPyramid) :
  slant_height_angle_cos p = 16 * p.k^2 - 1 ∧ p.k < 0.25 * Real.sqrt 2 :=
sorry

end regular_quad_pyramid_theorem_l2057_205778


namespace base_2_representation_of_123_l2057_205736

theorem base_2_representation_of_123 :
  ∃ (a b c d e f g : ℕ),
    (a = 1 ∧ b = 1 ∧ c = 1 ∧ d = 1 ∧ e = 0 ∧ f = 1 ∧ g = 1) ∧
    123 = a * 2^6 + b * 2^5 + c * 2^4 + d * 2^3 + e * 2^2 + f * 2^1 + g * 2^0 :=
by sorry

end base_2_representation_of_123_l2057_205736


namespace dice_probability_l2057_205730

/-- The number of faces on each die -/
def num_faces : ℕ := 6

/-- The number of dice rolled -/
def num_dice : ℕ := 6

/-- The total number of possible outcomes when rolling the dice -/
def total_outcomes : ℕ := num_faces ^ num_dice

/-- The number of favorable outcomes (at least one pair but not a three-of-a-kind) -/
def favorable_outcomes : ℕ := 27000

/-- The probability of rolling at least one pair but not a three-of-a-kind -/
def probability : ℚ := favorable_outcomes / total_outcomes

theorem dice_probability : probability = 625 / 1089 := by
  sorry

end dice_probability_l2057_205730


namespace existence_of_non_dividing_sum_l2057_205768

theorem existence_of_non_dividing_sum (n : ℕ) (a : Fin n → ℕ+) (h_n : n ≥ 3) 
  (h_distinct : ∀ i j, i ≠ j → a i ≠ a j) : 
  ∃ i j, i ≠ j ∧ ∀ k, ¬((a i + a j : ℕ) ∣ (3 * (a k : ℕ))) := by
  sorry

end existence_of_non_dividing_sum_l2057_205768


namespace investment_percentage_l2057_205706

/-- Given two investments with a total of $2000, where $600 is invested at 8%,
    and the annual income from the first investment exceeds the second by $92,
    prove that the percentage of the first investment is 10%. -/
theorem investment_percentage : 
  ∀ (total_investment first_investment_amount first_investment_rate : ℝ),
  total_investment = 2000 →
  first_investment_amount = 1400 →
  first_investment_rate * first_investment_amount - 0.08 * 600 = 92 →
  first_investment_rate = 0.1 := by
sorry

end investment_percentage_l2057_205706


namespace max_area_APBQ_l2057_205795

noncomputable section

-- Define the Cartesian coordinate system
def O : ℝ × ℝ := (0, 0)
def F : ℝ × ℝ := (-1, 0)

-- Define the distance ratio condition
def distance_ratio (P : ℝ × ℝ) : ℝ :=
  Real.sqrt ((P.1 + 1)^2 + P.2^2) / |P.1 + 2|

-- Define the trajectory C
def C : Set (ℝ × ℝ) :=
  {P | distance_ratio P = Real.sqrt 2 / 2}

-- Define the circle C₁
def C₁ : Set (ℝ × ℝ) :=
  {P | (P.1 - 4)^2 + P.2^2 = 32}

-- Define a chord AB of C passing through F
def chord_AB (m : ℝ) : Set (ℝ × ℝ) :=
  {P | P ∈ C ∧ P.1 = m * P.2 - 1}

-- Define the midpoint M of AB
def M (m : ℝ) : ℝ × ℝ :=
  (-2 / (m^2 + 2), m / (m^2 + 2))

-- Define the line OM
def line_OM (m : ℝ) : Set (ℝ × ℝ) :=
  {P | P.2 = (m / (m^2 + 2)) * P.1}

-- Define the intersection points P and Q
def P_Q (m : ℝ) : Set (ℝ × ℝ) :=
  {P | P ∈ C₁ ∧ P ∈ line_OM m}

-- Define the area of quadrilateral APBQ
def area_APBQ (m : ℝ) : ℝ :=
  8 * Real.sqrt 2 * Real.sqrt ((m^2 + 8) * (m^2 + 1) / (m^2 + 4)^2)

-- Theorem statement
theorem max_area_APBQ :
  ∃ m : ℝ, ∀ n : ℝ, area_APBQ m ≥ area_APBQ n ∧ area_APBQ m = 14 * Real.sqrt 6 / 3 :=
sorry

end max_area_APBQ_l2057_205795


namespace households_with_bike_only_l2057_205786

theorem households_with_bike_only 
  (total : ℕ) 
  (neither : ℕ) 
  (both : ℕ) 
  (with_car : ℕ) 
  (h1 : total = 90)
  (h2 : neither = 11)
  (h3 : both = 20)
  (h4 : with_car = 44) :
  total - neither - with_car + both = 35 := by
  sorry

end households_with_bike_only_l2057_205786


namespace jerry_has_36_stickers_l2057_205751

/-- Given the number of stickers for Fred, calculate the number of stickers for Jerry. -/
def jerrys_stickers (freds_stickers : ℕ) : ℕ :=
  let georges_stickers := freds_stickers - 6
  3 * georges_stickers

/-- Prove that Jerry has 36 stickers given the conditions in the problem. -/
theorem jerry_has_36_stickers :
  jerrys_stickers 18 = 36 := by
  sorry

end jerry_has_36_stickers_l2057_205751


namespace people_in_house_l2057_205708

theorem people_in_house : 
  ∀ (initial_bedroom : ℕ) (entering_bedroom : ℕ) (living_room : ℕ),
    initial_bedroom = 2 →
    entering_bedroom = 5 →
    living_room = 8 →
    initial_bedroom + entering_bedroom + living_room = 15 :=
by
  sorry

end people_in_house_l2057_205708


namespace abc_inequality_l2057_205791

theorem abc_inequality (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0)
  (h : a^2 + b^2 + c^2 + a*b*c = 4) : a + b + c ≤ 3 := by
  sorry

end abc_inequality_l2057_205791


namespace ninas_homework_is_40_l2057_205793

/-- The amount of Nina's total homework given Ruby's homework and the ratios --/
def ninas_total_homework (rubys_math_homework : ℕ) (rubys_reading_homework : ℕ) 
  (math_ratio : ℕ) (reading_ratio : ℕ) : ℕ :=
  math_ratio * rubys_math_homework + reading_ratio * rubys_reading_homework

/-- Theorem stating that Nina's total homework is 40 given the problem conditions --/
theorem ninas_homework_is_40 :
  ninas_total_homework 6 2 4 8 = 40 := by
  sorry

end ninas_homework_is_40_l2057_205793


namespace correct_pricing_l2057_205729

/-- A hotel's pricing structure -/
structure HotelPricing where
  flat_fee : ℝ
  additional_fee : ℝ
  discount : ℝ := 10

/-- Calculate the cost of a stay given the pricing and number of nights -/
def stay_cost (p : HotelPricing) (nights : ℕ) : ℝ :=
  if nights ≤ 4 then
    p.flat_fee + p.additional_fee * (nights - 1 : ℝ)
  else
    p.flat_fee + p.additional_fee * 3 + (p.additional_fee - p.discount) * ((nights - 4) : ℝ)

/-- The theorem stating the correct pricing structure -/
theorem correct_pricing :
  ∃ (p : HotelPricing),
    stay_cost p 4 = 180 ∧
    stay_cost p 7 = 302 ∧
    p.flat_fee = 28 ∧
    p.additional_fee = 50.67 := by
  sorry

end correct_pricing_l2057_205729


namespace peggy_left_knee_bandages_l2057_205717

/-- The number of bandages Peggy used on her left knee -/
def bandages_on_left_knee (initial_bandages : ℕ) (remaining_bandages : ℕ) (right_knee_bandages : ℕ) : ℕ :=
  initial_bandages - remaining_bandages - right_knee_bandages

/-- Proof that Peggy used 2 bandages on her left knee -/
theorem peggy_left_knee_bandages : 
  let initial_bandages := 24 - 8
  let remaining_bandages := 11
  let right_knee_bandages := 3
  bandages_on_left_knee initial_bandages remaining_bandages right_knee_bandages = 2 := by
sorry

#eval bandages_on_left_knee (24 - 8) 11 3

end peggy_left_knee_bandages_l2057_205717


namespace cubic_function_c_value_l2057_205704

theorem cubic_function_c_value (a b c d y₁ y₂ : ℝ) :
  y₁ = a + b + c + d →
  y₂ = 8*a + 4*b + 2*c + d →
  y₁ - y₂ = -17 →
  c = -17 + 7*a + 3*b :=
by sorry

end cubic_function_c_value_l2057_205704


namespace complex_modulus_problem_l2057_205743

theorem complex_modulus_problem (θ : ℝ) (z : ℂ) : 
  z = (Complex.I * (Real.sin θ - Complex.I)) / Complex.I →
  Real.cos θ = 1/3 →
  Complex.abs z = Real.sqrt 17 / 3 := by
sorry

end complex_modulus_problem_l2057_205743


namespace fourth_number_12th_row_l2057_205738

/-- Given a number pattern where each row has 8 numbers, and the last number of each row is 8 times the row number, this function calculates the nth number in the mth row. -/
def patternNumber (m n : ℕ) : ℕ :=
  8 * (m - 1) + n

/-- Theorem stating that the fourth number in the 12th row of the described pattern is 92. -/
theorem fourth_number_12th_row : patternNumber 12 4 = 92 := by
  sorry

end fourth_number_12th_row_l2057_205738


namespace no_primes_satisfying_congruence_l2057_205783

theorem no_primes_satisfying_congruence : 
  ¬ ∃ (p : ℕ) (hp : Nat.Prime p) (r s : ℤ),
    (∀ (x : ℤ), (x^3 - x + 2) % p = ((x - r)^2 * (x - s)) % p) ∧
    (∀ (r' s' : ℤ), (∀ (x : ℤ), (x^3 - x + 2) % p = ((x - r')^2 * (x - s')) % p) → r' = r ∧ s' = s) :=
by sorry


end no_primes_satisfying_congruence_l2057_205783


namespace middle_number_proof_l2057_205782

theorem middle_number_proof (x y z : ℝ) 
  (h_distinct : x < y ∧ y < z)
  (h_sum1 : x + y = 15)
  (h_sum2 : x + z = 18)
  (h_sum3 : y + z = 21) :
  y = 9 := by
sorry

end middle_number_proof_l2057_205782


namespace inequality_solution_l2057_205744

theorem inequality_solution (a : ℝ) : 
  (∀ x > 0, (a * x - 9) * Real.log (2 * a / x) ≤ 0) ↔ a = 3 * Real.sqrt 2 / 2 :=
by sorry

end inequality_solution_l2057_205744


namespace quadratic_roots_property_l2057_205711

theorem quadratic_roots_property (a b : ℝ) : 
  (2 * a^2 + 6 * a - 14 = 0) → 
  (2 * b^2 + 6 * b - 14 = 0) → 
  (2 * a - 3) * (4 * b - 6) = -2 := by
sorry

end quadratic_roots_property_l2057_205711


namespace sam_total_money_l2057_205735

/-- The value of a penny in dollars -/
def penny_value : ℚ := 0.01

/-- The value of a nickel in dollars -/
def nickel_value : ℚ := 0.05

/-- The value of a dime in dollars -/
def dime_value : ℚ := 0.10

/-- The value of a quarter in dollars -/
def quarter_value : ℚ := 0.25

/-- The number of pennies Sam earned -/
def num_pennies : ℕ := 15

/-- The number of nickels Sam earned -/
def num_nickels : ℕ := 11

/-- The number of dimes Sam earned -/
def num_dimes : ℕ := 21

/-- The number of quarters Sam earned -/
def num_quarters : ℕ := 29

/-- The total value of Sam's coins in dollars -/
def total_value : ℚ := 
  num_pennies * penny_value + 
  num_nickels * nickel_value + 
  num_dimes * dime_value + 
  num_quarters * quarter_value

theorem sam_total_money : total_value = 10.05 := by
  sorry

end sam_total_money_l2057_205735


namespace equal_chord_circle_exists_l2057_205785

/-- A triangle in a 2D plane --/
structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ

/-- The incenter of a triangle --/
def incenter (t : Triangle) : ℝ × ℝ := sorry

/-- A circle in a 2D plane --/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- The length of a chord formed by the intersection of a circle and a line segment --/
def chordLength (c : Circle) (p1 p2 : ℝ × ℝ) : ℝ := sorry

/-- Theorem: For any triangle, there exists a circle that cuts chords of equal length from its sides --/
theorem equal_chord_circle_exists (t : Triangle) : 
  ∃ (c : Circle), 
    chordLength c t.A t.B = chordLength c t.B t.C ∧ 
    chordLength c t.B t.C = chordLength c t.C t.A := by
  sorry

end equal_chord_circle_exists_l2057_205785


namespace wolves_heads_count_l2057_205788

/-- Represents the count of normal wolves -/
def normal_wolves : ℕ := sorry

/-- Represents the count of mutant wolves -/
def mutant_wolves : ℕ := sorry

/-- The total number of heads for all creatures -/
def total_heads : ℕ := 21

/-- The total number of legs for all creatures -/
def total_legs : ℕ := 57

/-- The number of heads a person has -/
def person_heads : ℕ := 1

/-- The number of legs a person has -/
def person_legs : ℕ := 2

/-- The number of heads a normal wolf has -/
def normal_wolf_heads : ℕ := 1

/-- The number of legs a normal wolf has -/
def normal_wolf_legs : ℕ := 4

/-- The number of heads a mutant wolf has -/
def mutant_wolf_heads : ℕ := 2

/-- The number of legs a mutant wolf has -/
def mutant_wolf_legs : ℕ := 3

theorem wolves_heads_count :
  normal_wolves * normal_wolf_heads + mutant_wolves * mutant_wolf_heads = total_heads - person_heads ∧
  normal_wolves * normal_wolf_legs + mutant_wolves * mutant_wolf_legs = total_legs - person_legs := by
  sorry

end wolves_heads_count_l2057_205788


namespace matrix_transpose_inverse_sum_squares_l2057_205703

theorem matrix_transpose_inverse_sum_squares (p q r s : ℝ) :
  let B : Matrix (Fin 2) (Fin 2) ℝ := !![p, q; r, s]
  B.transpose = B⁻¹ →
  p^2 + q^2 + r^2 + s^2 = 2 := by
sorry

end matrix_transpose_inverse_sum_squares_l2057_205703


namespace quadratic_function_unique_l2057_205789

/-- A quadratic function passing through the origin with a given derivative -/
def quadratic_function (f : ℝ → ℝ) : Prop :=
  ∃ a b : ℝ,
    (∀ x, f x = a * x^2 + b * x) ∧
    (f 0 = 0) ∧
    (∀ x, deriv f x = 3 * x - 1/2)

/-- The theorem stating the unique form of the quadratic function -/
theorem quadratic_function_unique (f : ℝ → ℝ) (hf : quadratic_function f) :
  ∀ x, f x = 3/2 * x^2 - 1/2 * x :=
sorry

end quadratic_function_unique_l2057_205789


namespace find_x_l2057_205741

theorem find_x : ∃ x : ℝ, (85 + x / 113) * 113 = 9637 ∧ x = 9552 := by
  sorry

end find_x_l2057_205741


namespace initial_strawberry_weight_l2057_205766

/-- The initial total weight of strawberries collected by Marco and his dad -/
def initial_total (marco_weight dad_weight lost_weight : ℕ) : ℕ :=
  marco_weight + dad_weight + lost_weight

/-- Proof that the initial total weight of strawberries is 36 pounds -/
theorem initial_strawberry_weight :
  ∀ (marco_weight dad_weight lost_weight : ℕ),
    marco_weight = 12 →
    dad_weight = 16 →
    lost_weight = 8 →
    initial_total marco_weight dad_weight lost_weight = 36 :=
by
  sorry

end initial_strawberry_weight_l2057_205766


namespace linear_system_solution_l2057_205713

theorem linear_system_solution (a₁ a₂ b₁ b₂ c₁ c₂ : ℝ) 
  (h₁ : a₁ * ((c₁ * b₂ - c₂ * b₁) / (a₁ * b₂ - a₂ * b₁)) + b₁ * y = c₁)
  (h₂ : a₂ * ((c₁ * b₂ - c₂ * b₁) / (a₁ * b₂ - a₂ * b₁)) + b₂ * y = c₂)
  (h₃ : a₁ * b₂ ≠ a₂ * b₁) :
  y = (c₁ * a₂ - c₂ * a₁) / (b₁ * a₂ - b₂ * a₁) :=
by sorry

end linear_system_solution_l2057_205713


namespace no_distinct_natural_power_sum_equality_l2057_205731

theorem no_distinct_natural_power_sum_equality :
  ∀ (x y z t : ℕ),
    x ≠ y ∧ x ≠ z ∧ x ≠ t ∧ y ≠ z ∧ y ≠ t ∧ z ≠ t →
    x^x + y^y ≠ z^z + t^t :=
by
  sorry

end no_distinct_natural_power_sum_equality_l2057_205731


namespace larger_cylinder_height_l2057_205784

/-- The height of a larger cylinder given specific conditions -/
theorem larger_cylinder_height : 
  ∀ (d_large h_small r_small : ℝ) (n : ℕ),
  d_large = 6 →
  h_small = 5 →
  r_small = 2 →
  n = 3 →
  ∃ (h_large : ℝ),
    h_large = 20 / 3 ∧
    π * (d_large / 2)^2 * h_large = n * π * r_small^2 * h_small :=
by sorry

end larger_cylinder_height_l2057_205784


namespace frosting_calculation_l2057_205797

/-- Calculates the total number of frosting cans needed for a bakery order -/
theorem frosting_calculation (layer_cake_frosting : ℝ) (single_item_frosting : ℝ) 
  (tiered_cake_frosting : ℝ) (mini_cupcake_pair_frosting : ℝ) 
  (layer_cakes : ℕ) (tiered_cakes : ℕ) (cupcake_dozens : ℕ) 
  (mini_cupcakes : ℕ) (single_cakes : ℕ) (brownie_pans : ℕ) :
  layer_cake_frosting = 1 →
  single_item_frosting = 0.5 →
  tiered_cake_frosting = 1.5 →
  mini_cupcake_pair_frosting = 0.25 →
  layer_cakes = 4 →
  tiered_cakes = 8 →
  cupcake_dozens = 10 →
  mini_cupcakes = 30 →
  single_cakes = 15 →
  brownie_pans = 24 →
  layer_cakes * layer_cake_frosting +
  tiered_cakes * tiered_cake_frosting +
  cupcake_dozens * single_item_frosting +
  (mini_cupcakes / 2) * mini_cupcake_pair_frosting +
  single_cakes * single_item_frosting +
  brownie_pans * single_item_frosting = 44.25 := by
sorry

end frosting_calculation_l2057_205797


namespace log_inequality_l2057_205790

theorem log_inequality (a b : ℝ) : Real.log a > Real.log b → a > b := by
  sorry

end log_inequality_l2057_205790


namespace intersection_of_A_and_B_l2057_205787

def A : Set ℤ := {1, 2, 3}

def B : Set ℤ := {x | (x + 1) * (x - 2) < 0}

theorem intersection_of_A_and_B : A ∩ B = {1} := by
  sorry

end intersection_of_A_and_B_l2057_205787


namespace minimum_workers_needed_l2057_205734

/-- The number of units completed per worker per day in the first process -/
def process1_rate : ℕ := 48

/-- The number of units completed per worker per day in the second process -/
def process2_rate : ℕ := 32

/-- The number of units completed per worker per day in the third process -/
def process3_rate : ℕ := 28

/-- The minimum number of workers needed for the first process -/
def workers1 : ℕ := 14

/-- The minimum number of workers needed for the second process -/
def workers2 : ℕ := 21

/-- The minimum number of workers needed for the third process -/
def workers3 : ℕ := 24

/-- The theorem stating the minimum number of workers needed for each process -/
theorem minimum_workers_needed :
  (∃ n : ℕ, n > 0 ∧ 
    n = process1_rate * workers1 ∧ 
    n = process2_rate * workers2 ∧ 
    n = process3_rate * workers3) ∧
  (∀ w1 w2 w3 : ℕ, 
    (∃ m : ℕ, m > 0 ∧ 
      m = process1_rate * w1 ∧ 
      m = process2_rate * w2 ∧ 
      m = process3_rate * w3) →
    w1 ≥ workers1 ∧ w2 ≥ workers2 ∧ w3 ≥ workers3) :=
by sorry

end minimum_workers_needed_l2057_205734


namespace range_of_f_on_interval_l2057_205718

-- Define the function f
def f (m : ℝ) (x : ℝ) : ℝ := 4 * x^2 - m * x + 1

-- State the theorem
theorem range_of_f_on_interval (m : ℝ) :
  (∀ x₁ x₂, x₁ < x₂ ∧ x₂ ≤ -2 → f m x₁ > f m x₂) ∧ 
  (∀ x₁ x₂, -2 ≤ x₁ ∧ x₁ < x₂ → f m x₁ < f m x₂) →
  Set.Icc (f m 1) (f m 2) = Set.Icc (-11) 33 :=
sorry

end range_of_f_on_interval_l2057_205718


namespace spherical_to_rectangular_conversion_l2057_205715

theorem spherical_to_rectangular_conversion :
  let ρ : ℝ := 10
  let θ : ℝ := 4 * π / 3
  let φ : ℝ := π / 3
  let x : ℝ := ρ * Real.sin φ * Real.cos θ
  let y : ℝ := ρ * Real.sin φ * Real.sin θ
  let z : ℝ := ρ * Real.cos φ
  (x = -5 * Real.sqrt 3) ∧ (y = -15 / 2) ∧ (z = 5) :=
by sorry

end spherical_to_rectangular_conversion_l2057_205715


namespace t_shaped_figure_perimeter_l2057_205707

/-- A geometric figure composed of four identical squares in a T shape -/
structure TShapedFigure where
  /-- The side length of each square in the figure -/
  side_length : ℝ
  /-- The total area of the figure is 144 cm² -/
  area_eq : 4 * side_length ^ 2 = 144

/-- The perimeter of a T-shaped figure -/
def perimeter (f : TShapedFigure) : ℝ :=
  5 * f.side_length

theorem t_shaped_figure_perimeter (f : TShapedFigure) : perimeter f = 30 :=
sorry

end t_shaped_figure_perimeter_l2057_205707


namespace quadrilaterals_form_polygons_l2057_205723

/-- A point in 2D space --/
structure Point :=
  (x : ℤ)
  (y : ℤ)

/-- A polygon defined by its vertices --/
structure Polygon :=
  (vertices : List Point)

/-- Definition of a square --/
def is_square (p : Polygon) : Prop :=
  p.vertices.length = 4 ∧
  ∃ (x y : ℤ), p.vertices = [Point.mk x y, Point.mk (x+2) y, Point.mk (x+2) (y+2), Point.mk x (y+2)]

/-- Definition of a triangle --/
def is_triangle (p : Polygon) : Prop :=
  p.vertices.length = 3

/-- Definition of a pentagon --/
def is_pentagon (p : Polygon) : Prop :=
  p.vertices.length = 5

/-- The two squares from the problem --/
def square1 : Polygon :=
  Polygon.mk [Point.mk 0 0, Point.mk 2 0, Point.mk 2 2, Point.mk 0 2]

def square2 : Polygon :=
  Polygon.mk [Point.mk 2 2, Point.mk 4 2, Point.mk 4 4, Point.mk 2 4]

/-- Main theorem --/
theorem quadrilaterals_form_polygons :
  (is_square square1 ∧ is_square square2) →
  (∃ (t : Polygon) (p : Polygon), is_triangle t ∧ is_pentagon p ∧
    (∀ v : Point, v ∈ t.vertices → v ∈ square1.vertices ∨ v ∈ square2.vertices) ∧
    (∀ v : Point, v ∈ p.vertices → v ∈ square1.vertices ∨ v ∈ square2.vertices)) ∧
  (∃ (t : Polygon) (q : Polygon) (p : Polygon), 
    is_triangle t ∧ p.vertices.length = 4 ∧ is_pentagon p ∧
    (∀ v : Point, v ∈ t.vertices → v ∈ square1.vertices ∨ v ∈ square2.vertices) ∧
    (∀ v : Point, v ∈ q.vertices → v ∈ square1.vertices ∨ v ∈ square2.vertices) ∧
    (∀ v : Point, v ∈ p.vertices → v ∈ square1.vertices ∨ v ∈ square2.vertices)) :=
by sorry

end quadrilaterals_form_polygons_l2057_205723


namespace can_determine_contents_l2057_205714

/-- Represents the possible contents of a box -/
inductive BoxContent
  | Red
  | White
  | Mixed

/-- Represents a box with a label and actual content -/
structure Box where
  label : BoxContent
  content : BoxContent

/-- The result of opening a box and drawing a ball -/
inductive DrawResult
  | Red
  | White

/-- Represents the state of the puzzle -/
structure PuzzleState where
  boxes : Fin 3 → Box
  all_labels_incorrect : ∀ i, (boxes i).label ≠ (boxes i).content
  contents_distinct : ∀ i j, i ≠ j → (boxes i).content ≠ (boxes j).content

/-- Function to determine the contents of all boxes based on the draw result -/
def determineContents (state : PuzzleState) (draw : DrawResult) : Fin 3 → BoxContent :=
  sorry

theorem can_determine_contents (state : PuzzleState) :
  ∃ (i : Fin 3) (draw : DrawResult),
    determineContents state draw = λ j => (state.boxes j).content :=
  sorry

end can_determine_contents_l2057_205714


namespace point_translation_point_translation_proof_l2057_205752

/-- Given a point B with coordinates (-5, 1), moving it 4 units right and 2 units up
    results in a point B' with coordinates (-1, 3). -/
theorem point_translation : ℝ × ℝ → ℝ × ℝ → Prop :=
  fun B B' => B = (-5, 1) → B' = (B.1 + 4, B.2 + 2) → B' = (-1, 3)

/-- The proof of the theorem. -/
theorem point_translation_proof : point_translation (-5, 1) (-1, 3) := by
  sorry

end point_translation_point_translation_proof_l2057_205752


namespace count_pairs_equals_210_l2057_205754

def count_pairs : ℕ := 
  (Finset.range 20).sum (fun a => 21 - a)

theorem count_pairs_equals_210 : count_pairs = 210 := by sorry

end count_pairs_equals_210_l2057_205754


namespace interest_rate_approximately_three_percent_l2057_205732

/-- Calculates the interest rate for the first part of a loan given the total sum,
    the amount of the second part, and the interest rate for the second part. -/
def calculate_interest_rate (total_sum second_part second_rate : ℚ) : ℚ :=
  let first_part := total_sum - second_part
  let second_interest := second_part * second_rate * 3
  second_interest / (first_part * 8)

/-- Theorem stating that under the given conditions, the interest rate
    for the first part is approximately 3%. -/
theorem interest_rate_approximately_three_percent :
  let total_sum : ℚ := 2678
  let second_part : ℚ := 1648
  let second_rate : ℚ := 5 / 100
  let calculated_rate := calculate_interest_rate total_sum second_part second_rate
  abs (calculated_rate - 3 / 100) < 1 / 1000 :=
by
  sorry

#eval calculate_interest_rate 2678 1648 (5/100)

end interest_rate_approximately_three_percent_l2057_205732


namespace monotonic_decreasing_range_l2057_205748

-- Define the function f
def f (k : ℝ) (x : ℝ) : ℝ := (2 * k - 1) * x + 1

-- State the theorem
theorem monotonic_decreasing_range (k : ℝ) :
  (∀ x y : ℝ, x < y → f k x > f k y) →
  k < (1 / 2 : ℝ) :=
by sorry

end monotonic_decreasing_range_l2057_205748


namespace expression_simplification_l2057_205755

theorem expression_simplification (y : ℝ) :
  3 * y + 12 * y^2 + 18 - (6 - 3 * y - 12 * y^2) + 5 * y^3 = 5 * y^3 + 24 * y^2 + 6 * y + 12 := by
  sorry

end expression_simplification_l2057_205755
