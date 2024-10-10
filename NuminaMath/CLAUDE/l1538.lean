import Mathlib

namespace no_real_roots_l1538_153856

theorem no_real_roots : ¬∃ x : ℝ, x + Real.sqrt (2 * x - 3) = 5 := by
  sorry

end no_real_roots_l1538_153856


namespace staff_members_count_correct_staff_count_l1538_153851

theorem staff_members_count (allowance_days : ℕ) (allowance_rate : ℕ) 
  (accountant_amount : ℕ) (petty_cash : ℕ) : ℕ :=
  let allowance_per_staff := allowance_days * allowance_rate
  let total_amount := accountant_amount + petty_cash
  total_amount / allowance_per_staff

theorem correct_staff_count : 
  staff_members_count 30 100 65000 1000 = 22 := by sorry

end staff_members_count_correct_staff_count_l1538_153851


namespace cube_properties_l1538_153879

-- Define the surface area of the cube
def surface_area : ℝ := 150

-- Define the relationship between surface area and edge length
def edge_length (s : ℝ) : Prop := 6 * s^2 = surface_area

-- Define the volume of a cube given its edge length
def volume (s : ℝ) : ℝ := s^3

-- Theorem statement
theorem cube_properties :
  ∃ (s : ℝ), edge_length s ∧ s = 5 ∧ volume s = 125 :=
sorry

end cube_properties_l1538_153879


namespace convex_polygon_sides_l1538_153893

theorem convex_polygon_sides (n : ℕ) (sum_except_one : ℝ) : 
  sum_except_one = 2190 → 
  (∃ (missing_angle : ℝ), 
    missing_angle > 0 ∧ 
    missing_angle < 180 ∧ 
    sum_except_one + missing_angle = 180 * (n - 2)) → 
  n = 15 := by
sorry

end convex_polygon_sides_l1538_153893


namespace last_three_average_l1538_153887

theorem last_three_average (a b c d : ℝ) : 
  (a + b + c) / 3 = 6 →
  a + d = 11 →
  d = 4 →
  (b + c + d) / 3 = 5 := by
sorry

end last_three_average_l1538_153887


namespace polygon_area_l1538_153870

structure Polygon :=
  (sides : ℕ)
  (side_length : ℝ)
  (perimeter : ℝ)
  (is_rectangular_with_removed_corners : Prop)

def area_of_polygon (p : Polygon) : ℝ :=
  20 * p.side_length^2

theorem polygon_area (p : Polygon) 
  (h1 : p.sides = 20)
  (h2 : p.perimeter = 60)
  (h3 : p.is_rectangular_with_removed_corners)
  (h4 : p.side_length = p.perimeter / p.sides) :
  area_of_polygon p = 180 := by
  sorry

end polygon_area_l1538_153870


namespace sum_greater_than_one_l1538_153826

theorem sum_greater_than_one : 
  (let a := [1/4, 2/8, 3/4]
   let b := [3, -1.5, -0.5]
   let c := [0.25, 0.75, 0.05]
   let d := [3/2, -3/4, 1/4]
   let e := [1.5, 1.5, -2]
   (a.sum > 1 ∧ c.sum > 1) ∧
   (b.sum ≤ 1 ∧ d.sum ≤ 1 ∧ e.sum ≤ 1)) := by
  sorry

end sum_greater_than_one_l1538_153826


namespace hyperbola_eccentricity_l1538_153876

theorem hyperbola_eccentricity (m : ℝ) : 
  (∀ x y : ℝ, x^2 / m + y^2 / 2 = 1) →
  (∃ a b c : ℝ, a^2 = 2 ∧ b^2 = -m ∧ c^2 = a^2 + b^2 ∧ c^2 / a^2 = 4) →
  m = -6 := by
sorry

end hyperbola_eccentricity_l1538_153876


namespace c_rent_share_l1538_153888

/-- Represents a person's pasture usage -/
structure PastureUsage where
  oxen : ℕ
  months : ℕ

/-- Calculates the total oxen-months for a given pasture usage -/
def oxenMonths (usage : PastureUsage) : ℕ :=
  usage.oxen * usage.months

/-- Calculates the share of rent for a given usage and total usage -/
def rentShare (usage : PastureUsage) (totalUsage : ℕ) (totalRent : ℕ) : ℚ :=
  (oxenMonths usage : ℚ) / (totalUsage : ℚ) * (totalRent : ℚ)

theorem c_rent_share :
  let a := PastureUsage.mk 10 7
  let b := PastureUsage.mk 12 5
  let c := PastureUsage.mk 15 3
  let totalRent := 175
  let totalUsage := oxenMonths a + oxenMonths b + oxenMonths c
  rentShare c totalUsage totalRent = 45 := by
  sorry

end c_rent_share_l1538_153888


namespace water_jars_problem_l1538_153827

theorem water_jars_problem (S L : ℝ) (h1 : S > 0) (h2 : L > 0) (h3 : S < L) :
  let water_amount := S * (1/3)
  (water_amount = L * (1/2)) →
  (L * (1/2) + water_amount) / L = 1 := by
sorry

end water_jars_problem_l1538_153827


namespace smallest_number_divisible_l1538_153802

theorem smallest_number_divisible (n : ℕ) : 
  (∀ m : ℕ, m < 32127 → ¬(510 ∣ (m + 3) ∧ 4590 ∣ (m + 3) ∧ 105 ∣ (m + 3))) ∧
  (510 ∣ (32127 + 3) ∧ 4590 ∣ (32127 + 3) ∧ 105 ∣ (32127 + 3)) :=
sorry

end smallest_number_divisible_l1538_153802


namespace larger_integer_problem_l1538_153866

theorem larger_integer_problem (a b : ℕ+) : 
  a * b = 168 → 
  (a : ℤ) - (b : ℤ) = 4 ∨ (b : ℤ) - (a : ℤ) = 4 → 
  max a b = 14 := by
sorry

end larger_integer_problem_l1538_153866


namespace total_rainfall_is_23_l1538_153818

/-- Rainfall data for three days --/
structure RainfallData :=
  (monday_hours : ℕ)
  (monday_rate : ℕ)
  (tuesday_hours : ℕ)
  (tuesday_rate : ℕ)
  (wednesday_hours : ℕ)

/-- Calculate total rainfall for three days --/
def total_rainfall (data : RainfallData) : ℕ :=
  data.monday_hours * data.monday_rate +
  data.tuesday_hours * data.tuesday_rate +
  data.wednesday_hours * (2 * data.tuesday_rate)

/-- Theorem: The total rainfall for the given conditions is 23 inches --/
theorem total_rainfall_is_23 (data : RainfallData)
  (h1 : data.monday_hours = 7)
  (h2 : data.monday_rate = 1)
  (h3 : data.tuesday_hours = 4)
  (h4 : data.tuesday_rate = 2)
  (h5 : data.wednesday_hours = 2) :
  total_rainfall data = 23 := by
  sorry

end total_rainfall_is_23_l1538_153818


namespace spider_eats_all_flies_l1538_153834

/-- Represents a position on the grid -/
structure Position where
  x : Nat
  y : Nat

/-- Represents the spider's movement strategy -/
structure SpiderStrategy where
  initialPosition : Position
  moveSequence : List Position

/-- Represents the web with flies -/
structure Web where
  size : Nat
  flyPositions : List Position

/-- Theorem stating that the spider can eat all flies in at most 1980 moves -/
theorem spider_eats_all_flies (web : Web) (strategy : SpiderStrategy) : 
  web.size = 100 → 
  web.flyPositions.length = 100 → 
  strategy.initialPosition.x = 0 ∨ strategy.initialPosition.x = 99 → 
  strategy.initialPosition.y = 0 ∨ strategy.initialPosition.y = 99 → 
  ∃ (moves : List Position), 
    moves.length ≤ 1980 ∧ 
    (∀ fly ∈ web.flyPositions, fly ∈ moves) := by
  sorry

end spider_eats_all_flies_l1538_153834


namespace subtraction_problem_solution_l1538_153850

theorem subtraction_problem_solution :
  ∀ h t u : ℕ,
  h > u →
  h < 10 ∧ t < 10 ∧ u < 10 →
  (100 * h + 10 * t + u) - (100 * t + 10 * h + u) = 553 →
  h = 9 ∧ t = 4 ∧ u = 3 := by
sorry

end subtraction_problem_solution_l1538_153850


namespace triangle_inequality_tangent_l1538_153874

theorem triangle_inequality_tangent (a b c α β : ℝ) 
  (h : a + b < 3 * c) : 
  Real.tan (α / 2) * Real.tan (β / 2) < 1 / 2 :=
by sorry

end triangle_inequality_tangent_l1538_153874


namespace min_fencing_cost_problem_l1538_153855

/-- Represents the cost of fencing materials in rupees per meter -/
structure FencingMaterial where
  cost : ℚ

/-- Represents a rectangular field -/
structure RectangularField where
  length : ℚ
  width : ℚ
  area : ℚ

/-- Calculates the minimum fencing cost for a rectangular field -/
def minFencingCost (field : RectangularField) (materials : List FencingMaterial) : ℚ :=
  sorry

/-- Theorem stating the minimum fencing cost for the given problem -/
theorem min_fencing_cost_problem :
  let field : RectangularField := {
    length := 108,
    width := 81,
    area := 8748
  }
  let materials : List FencingMaterial := [
    { cost := 0.25 },
    { cost := 0.35 },
    { cost := 0.40 }
  ]
  minFencingCost field materials = 87.75 := by sorry

end min_fencing_cost_problem_l1538_153855


namespace consecutive_integers_product_not_square_l1538_153814

theorem consecutive_integers_product_not_square (a : ℕ) : 
  let A := Finset.range 20
  let sum := A.sum (λ i => a + i)
  let prod := A.prod (λ i => a + i)
  (sum % 23 ≠ 0) → (prod % 23 ≠ 0) → ¬ ∃ (n : ℕ), prod = n^2 := by
sorry

end consecutive_integers_product_not_square_l1538_153814


namespace nested_sqrt_equality_l1538_153852

theorem nested_sqrt_equality (x : ℝ) (h : x ≥ 0) :
  Real.sqrt (x * Real.sqrt (x * Real.sqrt (x * Real.sqrt x))) = (x ^ 11) ^ (1/8) := by
  sorry

end nested_sqrt_equality_l1538_153852


namespace book_page_ratio_l1538_153813

theorem book_page_ratio (total_pages : ℕ) (intro_pages : ℕ) (text_pages : ℕ) 
  (h1 : total_pages = 98)
  (h2 : intro_pages = 11)
  (h3 : text_pages = 19)
  (h4 : text_pages = (total_pages - intro_pages - text_pages * 2) / 2) :
  (total_pages - intro_pages - text_pages * 2) / total_pages = 1 / 2 := by
  sorry

end book_page_ratio_l1538_153813


namespace circle_center_from_intersection_l1538_153877

/-- Given a parabola y = k x^2 and a circle x^2 - 2px + y^2 - 2qy = 0,
    if the abscissas of their intersection points are the roots of x^3 + ax + b = 0,
    then the center of the circle is (-b/2, (1-a)/2). -/
theorem circle_center_from_intersection (k a b : ℝ) :
  ∃ (p q : ℝ),
    (∀ x y : ℝ, y = k * x^2 ∧ x^2 - 2*p*x + y^2 - 2*q*y = 0 →
      x^3 + a*x + b = 0) →
    (p = b/2 ∧ q = (a-1)/2) :=
sorry

end circle_center_from_intersection_l1538_153877


namespace shaded_area_between_circles_l1538_153816

theorem shaded_area_between_circles (r₁ r₂ : ℝ) (h₁ : r₁ = 3) (h₂ : r₂ = 4) : 
  let R := r₁ + r₂
  let A_large := π * R^2
  let A_small₁ := π * r₁^2
  let A_small₂ := π * r₂^2
  let A_shaded := A_large - A_small₁ - A_small₂
  A_shaded = 24 * π :=
by sorry

end shaded_area_between_circles_l1538_153816


namespace inequality_proof_l1538_153847

theorem inequality_proof (x y z : ℝ) 
  (sum_zero : x + y + z = 0) 
  (abs_sum_le_one : |x| + |y| + |z| ≤ 1) : 
  x + y/2 + z/3 ≤ 1/3 := by
sorry

end inequality_proof_l1538_153847


namespace determinant_of_trigonometric_matrix_l1538_153821

theorem determinant_of_trigonometric_matrix (α β γ : ℝ) : 
  let M : Matrix (Fin 3) (Fin 3) ℝ := ![
    ![Real.cos α * Real.cos β, Real.cos α * Real.sin β, -Real.sin (α + γ)],
    ![-Real.sin β, Real.cos β * Real.cos γ, Real.sin β * Real.sin γ],
    ![Real.sin (α + γ) * Real.cos β, Real.sin (α + γ) * Real.sin β, Real.cos (α + γ)]
  ]
  Matrix.det M = 1 := by
sorry

end determinant_of_trigonometric_matrix_l1538_153821


namespace max_cubes_fit_l1538_153845

def small_cube_edge : ℝ := 10.7
def large_cube_edge : ℝ := 100

theorem max_cubes_fit (small_cube_edge : ℝ) (large_cube_edge : ℝ) :
  small_cube_edge = 10.7 →
  large_cube_edge = 100 →
  ⌊(large_cube_edge ^ 3) / (small_cube_edge ^ 3)⌋ = 816 := by
  sorry

end max_cubes_fit_l1538_153845


namespace factors_of_given_number_l1538_153811

/-- The number of distinct natural-number factors of 4^5 · 5^3 · 7^2 -/
def num_factors : ℕ := 132

/-- The given number -/
def given_number : ℕ := 4^5 * 5^3 * 7^2

/-- A function that counts the number of distinct natural-number factors of a given natural number -/
def count_factors (n : ℕ) : ℕ := sorry

theorem factors_of_given_number :
  count_factors given_number = num_factors := by sorry

end factors_of_given_number_l1538_153811


namespace unit_digit_sum_factorials_l1538_153864

def factorial (n : ℕ) : ℕ := Nat.factorial n

def sum_factorials (n : ℕ) : ℕ := (List.range n).map factorial |>.sum

def unit_digit (n : ℕ) : ℕ := n % 10

theorem unit_digit_sum_factorials :
  unit_digit (sum_factorials 2012) = unit_digit (sum_factorials 4) :=
sorry

end unit_digit_sum_factorials_l1538_153864


namespace macys_weekly_goal_l1538_153840

/-- Macy's weekly running goal in miles -/
def weekly_goal : ℕ := 24

/-- Miles Macy runs per day -/
def miles_per_day : ℕ := 3

/-- Number of days Macy has run -/
def days_run : ℕ := 6

/-- Miles left to run after 6 days -/
def miles_left : ℕ := 6

/-- Theorem stating Macy's weekly running goal -/
theorem macys_weekly_goal : 
  weekly_goal = miles_per_day * days_run + miles_left := by sorry

end macys_weekly_goal_l1538_153840


namespace inequality_problem_l1538_153828

theorem inequality_problem (a b c d : ℝ) 
  (h1 : b < 0) (h2 : 0 < a) (h3 : d < c) (h4 : c < 0) : 
  a + c > b + d := by
  sorry

end inequality_problem_l1538_153828


namespace triangle_area_l1538_153896

theorem triangle_area (a b c : ℝ) (h₁ : a = 15) (h₂ : b = 36) (h₃ : c = 39) :
  (1/2) * a * b = 270 :=
by sorry

end triangle_area_l1538_153896


namespace sets_problem_l1538_153884

-- Define the sets A, B, and C
def A : Set ℝ := {x | x^2 - 9*x + 18 ≥ 0}
def B : Set ℝ := {x | -2 < x ∧ x < 9}
def C (a : ℝ) : Set ℝ := {x | a < x ∧ x < a + 1}

-- Theorem statement
theorem sets_problem :
  (A ∪ B = Set.univ) ∧
  ((Set.univ \ A) ∩ B = {x | 3 < x ∧ x < 6}) ∧
  (∀ a : ℝ, C a ⊆ B → -2 ≤ a ∧ a ≤ 8) := by
  sorry


end sets_problem_l1538_153884


namespace cube_operations_impossibility_l1538_153817

structure Cube :=
  (vertices : Fin 8 → ℕ)

def initial_state : Cube :=
  { vertices := λ i => if i = 0 then 1 else 0 }

def operation (c : Cube) (e : Fin 8 × Fin 8) : Cube :=
  { vertices := λ i => if i = e.1 ∨ i = e.2 then c.vertices i + 1 else c.vertices i }

def all_equal (c : Cube) : Prop :=
  ∀ i j, c.vertices i = c.vertices j

def all_divisible_by_three (c : Cube) : Prop :=
  ∀ i, c.vertices i % 3 = 0

theorem cube_operations_impossibility :
  (¬ ∃ (ops : List (Fin 8 × Fin 8)), all_equal (ops.foldl operation initial_state)) ∧
  (¬ ∃ (ops : List (Fin 8 × Fin 8)), all_divisible_by_three (ops.foldl operation initial_state)) :=
sorry

end cube_operations_impossibility_l1538_153817


namespace trigonometric_identities_l1538_153853

theorem trigonometric_identities :
  ∀ α : ℝ,
  (((Real.sqrt 3 * Real.sin (-1200 * π / 180)) / Real.tan (11 * π / 3)) - 
   (Real.cos (585 * π / 180) * Real.tan (-37 * π / 4)) = 
   Real.sqrt 3 / 2 - Real.sqrt 2 / 2) ∧
  ((Real.cos (α - π / 2) / Real.sin (5 * π / 2 + α)) * 
   Real.sin (α - 2 * π) * Real.cos (2 * π - α) = 
   Real.sin α ^ 2) := by
  sorry

end trigonometric_identities_l1538_153853


namespace quadratic_solution_sum_l1538_153880

theorem quadratic_solution_sum (a b : ℝ) : 
  (∀ x : ℂ, (6 * x^2 + 7 = 5 * x - 11) ↔ (x = a + b * I ∨ x = a - b * I)) →
  a + b^2 = 467/144 := by
  sorry

end quadratic_solution_sum_l1538_153880


namespace parallel_tangents_and_zero_points_l1538_153892

-- Define the function f(x)
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a / x + Real.log x - 1

-- Define the derivative of f(x)
noncomputable def f_deriv (a : ℝ) (x : ℝ) : ℝ := (x - a) / (x^2)

theorem parallel_tangents_and_zero_points (a : ℝ) (h : a > 0) :
  -- Part 1: Parallel tangents imply a = 3.5
  (f_deriv a 3 = f_deriv a (3/2) → a = 3.5) ∧
  -- Part 2: Zero points imply 0 < a ≤ 1
  (∃ x, f a x = 0 → 0 < a ∧ a ≤ 1) :=
sorry

end parallel_tangents_and_zero_points_l1538_153892


namespace initial_milk_collected_l1538_153801

/-- Proves that the initial amount of milk collected equals 30,000 gallons -/
theorem initial_milk_collected (
  pumping_hours : ℕ)
  (pumping_rate : ℕ)
  (adding_hours : ℕ)
  (adding_rate : ℕ)
  (milk_left : ℕ)
  (h1 : pumping_hours = 4)
  (h2 : pumping_rate = 2880)
  (h3 : adding_hours = 7)
  (h4 : adding_rate = 1500)
  (h5 : milk_left = 28980)
  (h6 : ∃ initial_milk : ℕ, 
    initial_milk + adding_hours * adding_rate - pumping_hours * pumping_rate = milk_left) :
  ∃ initial_milk : ℕ, initial_milk = 30000 := by
sorry


end initial_milk_collected_l1538_153801


namespace whale_ratio_theorem_l1538_153894

/-- The ratio of male whales on the third trip to the first trip -/
def whale_ratio : ℚ := 1 / 2

/-- The number of male whales on the first trip -/
def first_trip_males : ℕ := 28

/-- The number of female whales on the first trip -/
def first_trip_females : ℕ := 2 * first_trip_males

/-- The number of baby whales on the second trip -/
def second_trip_babies : ℕ := 8

/-- The total number of whales observed -/
def total_whales : ℕ := 178

/-- The number of male whales on the third trip -/
def third_trip_males : ℕ := total_whales - (first_trip_males + first_trip_females + second_trip_babies + 2 * second_trip_babies + first_trip_females)

theorem whale_ratio_theorem : 
  (third_trip_males : ℚ) / first_trip_males = whale_ratio := by
  sorry

end whale_ratio_theorem_l1538_153894


namespace three_digit_puzzle_l1538_153807

theorem three_digit_puzzle :
  ∀ (A B C : ℕ),
  (A ≥ 1 ∧ A ≤ 9) →
  (B ≥ 0 ∧ B ≤ 9) →
  (C ≥ 0 ∧ C ≤ 9) →
  (100 * A + 10 * B + B ≥ 100 ∧ 100 * A + 10 * B + B ≤ 999) →
  (A * B * B ≥ 10 ∧ A * B * B ≤ 99) →
  A * B * B = 10 * A + C →
  A * C = C →
  100 * A + 10 * B + B = 144 :=
by sorry

end three_digit_puzzle_l1538_153807


namespace ordered_triples_satisfying_equation_l1538_153872

theorem ordered_triples_satisfying_equation :
  ∀ m n p : ℕ,
    m > 0 ∧ n > 0 ∧ Nat.Prime p ∧ p^n + 144 = m^2 →
    ((m = 13 ∧ n = 2 ∧ p = 5) ∨
     (m = 20 ∧ n = 8 ∧ p = 2) ∨
     (m = 15 ∧ n = 4 ∧ p = 3)) :=
by sorry

end ordered_triples_satisfying_equation_l1538_153872


namespace drive_duration_proof_l1538_153833

def podcast1_duration : ℕ := 45
def podcast2_duration : ℕ := 2 * podcast1_duration
def podcast3_duration : ℕ := 105
def podcast4_duration : ℕ := 60
def podcast5_duration : ℕ := 60

def total_duration : ℕ := podcast1_duration + podcast2_duration + podcast3_duration + podcast4_duration + podcast5_duration

theorem drive_duration_proof :
  total_duration / 60 = 6 := by sorry

end drive_duration_proof_l1538_153833


namespace product_expansion_sum_l1538_153815

theorem product_expansion_sum (a b c d : ℝ) :
  (∀ x, (4 * x^2 - 6 * x + 5) * (9 - 3 * x) = a * x^3 + b * x^2 + c * x + d) →
  9 * a + 3 * b + 2 * c + d = -39 := by
sorry

end product_expansion_sum_l1538_153815


namespace smartphone_savings_theorem_l1538_153800

/-- The amount saved per month in yuan -/
def monthly_savings : ℕ := 530

/-- The cost of the smartphone in yuan -/
def smartphone_cost : ℕ := 2000

/-- The number of months required to save for the smartphone -/
def months_required : ℕ := 4

theorem smartphone_savings_theorem : 
  monthly_savings * months_required ≥ smartphone_cost :=
sorry

end smartphone_savings_theorem_l1538_153800


namespace reflection_result_l1538_153830

/-- Reflects a point across the x-axis -/
def reflect_x_axis (p : ℝ × ℝ) : ℝ × ℝ :=
  (p.1, -p.2)

/-- Reflects a point across the line y = -x + 2 -/
def reflect_line (p : ℝ × ℝ) : ℝ × ℝ :=
  let p' := (p.1, p.2 - 2)  -- Translate down by 2
  let p'' := (-p'.2, -p'.1) -- Reflect across y = -x
  (p''.1, p''.2 + 2)        -- Translate back up by 2

/-- The final position of point R after two reflections -/
def R_final : ℝ × ℝ :=
  reflect_line (reflect_x_axis (6, 1))

theorem reflection_result :
  R_final = (-3, -4) :=
by sorry

end reflection_result_l1538_153830


namespace flowerbed_count_l1538_153841

theorem flowerbed_count (total_seeds : ℕ) (seeds_per_flowerbed : ℕ) (h1 : total_seeds = 45) (h2 : seeds_per_flowerbed = 5) :
  total_seeds / seeds_per_flowerbed = 9 := by
  sorry

end flowerbed_count_l1538_153841


namespace locus_of_centers_l1538_153882

/-- The locus of centers of circles externally tangent to C1 and internally tangent to C2 -/
theorem locus_of_centers (a b : ℝ) : 
  (∃ r : ℝ, 
    (a^2 + b^2 = (r + 1)^2) ∧ 
    ((a - 3)^2 + b^2 = (5 - r)^2)) ↔ 
  (4*a^2 + 4*b^2 - 6*a - 25 = 0) := by sorry

end locus_of_centers_l1538_153882


namespace pet_food_price_l1538_153891

theorem pet_food_price (regular_discount_min : ℝ) (regular_discount_max : ℝ) 
  (additional_discount : ℝ) (lowest_price : ℝ) :
  regular_discount_min = 0.1 →
  regular_discount_max = 0.3 →
  additional_discount = 0.2 →
  lowest_price = 16.8 →
  ∃ (original_price : ℝ), 
    original_price * (1 - regular_discount_max) * (1 - additional_discount) = lowest_price ∧
    original_price = 30 :=
by sorry

end pet_food_price_l1538_153891


namespace equation_solution_l1538_153819

theorem equation_solution (x : ℝ) : 
  x ≠ 3 → (-x^2 = (3*x - 3) / (x - 3)) → x = 1 := by
  sorry

end equation_solution_l1538_153819


namespace average_of_a_and_b_l1538_153832

theorem average_of_a_and_b (a b : ℝ) : 
  (5 + a + b) / 3 = 33 → (a + b) / 2 = 47 := by
sorry

end average_of_a_and_b_l1538_153832


namespace deceased_member_income_l1538_153831

/-- Given a family with 4 earning members and an average monthly income,
    calculate the income of a deceased member when the average income changes. -/
theorem deceased_member_income
  (initial_members : ℕ)
  (initial_average_income : ℚ)
  (final_members : ℕ)
  (final_average_income : ℚ)
  (h1 : initial_members = 4)
  (h2 : final_members = initial_members - 1)
  (h3 : initial_average_income = 840)
  (h4 : final_average_income = 650) :
  (initial_members : ℚ) * initial_average_income - (final_members : ℚ) * final_average_income = 1410 :=
by sorry

end deceased_member_income_l1538_153831


namespace parallel_vectors_m_value_l1538_153846

/-- Two vectors are parallel if their components are proportional -/
def are_parallel (v1 v2 : ℝ × ℝ) : Prop :=
  v1.1 * v2.2 = v1.2 * v2.1

theorem parallel_vectors_m_value :
  ∀ m : ℝ, are_parallel (1, m) (m, 4) → m = 2 := by
  sorry

end parallel_vectors_m_value_l1538_153846


namespace sqrt_pattern_l1538_153867

theorem sqrt_pattern (n : ℕ+) : 
  Real.sqrt (1 + 1 / n^2 + 1 / (n + 1)^2) = 1 + 1 / (n * (n + 1)) := by
  sorry

end sqrt_pattern_l1538_153867


namespace sum_of_possible_radii_l1538_153889

/-- A circle with center C(r, r) is tangent to the positive x-axis and y-axis,
    and externally tangent to another circle centered at (3,3) with radius 2.
    This theorem states that the sum of all possible radii r is 16. -/
theorem sum_of_possible_radii : ∃ r₁ r₂ : ℝ,
  (r₁ - 3)^2 + (r₁ - 3)^2 = (r₁ + 2)^2 ∧
  (r₂ - 3)^2 + (r₂ - 3)^2 = (r₂ + 2)^2 ∧
  r₁ + r₂ = 16 :=
by sorry

end sum_of_possible_radii_l1538_153889


namespace fair_remaining_money_l1538_153871

/-- Calculates the remaining money after purchases at a fair --/
theorem fair_remaining_money 
  (initial_amount : ℝ) 
  (toy_cost : ℝ) 
  (hot_dog_cost : ℝ) 
  (candy_apple_cost : ℝ) 
  (discount_percentage : ℝ) 
  (h1 : initial_amount = 15)
  (h2 : toy_cost = 2)
  (h3 : hot_dog_cost = 3.5)
  (h4 : candy_apple_cost = 1.5)
  (h5 : discount_percentage = 0.5)
  (h6 : hot_dog_cost ≥ toy_cost ∧ hot_dog_cost ≥ candy_apple_cost) :
  initial_amount - (toy_cost + hot_dog_cost * (1 - discount_percentage) + candy_apple_cost) = 9.75 := by
  sorry


end fair_remaining_money_l1538_153871


namespace equation_represents_hyperbola_l1538_153838

/-- The equation (x+y)^2 = x^2 + y^2 + 2 represents a hyperbola -/
theorem equation_represents_hyperbola :
  ∃ (x y : ℝ), (x + y)^2 = x^2 + y^2 + 2 ↔ x * y = 1 :=
sorry

end equation_represents_hyperbola_l1538_153838


namespace remi_water_consumption_l1538_153898

/-- The amount of water Remi drinks in a week, given his bottle capacity, refill frequency, and spills. -/
def water_consumed (bottle_capacity : ℕ) (refills_per_day : ℕ) (days : ℕ) (spill1 : ℕ) (spill2 : ℕ) : ℕ :=
  bottle_capacity * refills_per_day * days - (spill1 + spill2)

/-- Theorem stating that Remi drinks 407 ounces of water in 7 days under the given conditions. -/
theorem remi_water_consumption :
  water_consumed 20 3 7 5 8 = 407 := by
  sorry

#eval water_consumed 20 3 7 5 8

end remi_water_consumption_l1538_153898


namespace max_correct_answers_l1538_153810

theorem max_correct_answers (total_questions : ℕ) (score : ℤ) : 
  total_questions = 25 → score = 65 → 
  ∃ (correct incorrect unanswered : ℕ),
    correct + incorrect + unanswered = total_questions ∧
    4 * correct - incorrect = score ∧
    correct ≤ 18 ∧
    ∀ c i u : ℕ, 
      c + i + u = total_questions → 
      4 * c - i = score → 
      c ≤ correct :=
by sorry

end max_correct_answers_l1538_153810


namespace grunters_win_probability_l1538_153837

theorem grunters_win_probability (n : ℕ) (p : ℚ) (h : p = 3/5) :
  p^n = 243/3125 → n = 5 :=
by
  sorry

end grunters_win_probability_l1538_153837


namespace mika_initial_stickers_l1538_153809

/-- The number of stickers Mika had initially -/
def initial_stickers : ℕ := 26

/-- The number of stickers Mika bought -/
def bought_stickers : ℕ := 26

/-- The number of stickers Mika got for her birthday -/
def birthday_stickers : ℕ := 20

/-- The number of stickers Mika gave to her sister -/
def given_stickers : ℕ := 6

/-- The number of stickers Mika used for the greeting card -/
def used_stickers : ℕ := 58

/-- The number of stickers Mika is left with -/
def remaining_stickers : ℕ := 2

theorem mika_initial_stickers :
  initial_stickers + bought_stickers + birthday_stickers - given_stickers - used_stickers = remaining_stickers :=
by
  sorry

#check mika_initial_stickers

end mika_initial_stickers_l1538_153809


namespace line_perp_plane_necessity_not_sufficiency_l1538_153885

-- Define the types for lines and planes
variable (L : Type*) [NormedAddCommGroup L] [InnerProductSpace ℝ L]
variable (P : Type*) [NormedAddCommGroup P] [InnerProductSpace ℝ P]

-- Define the perpendicular relation between lines and between a line and a plane
variable (perpendicular_lines : L → L → Prop)
variable (perpendicular_line_plane : L → P → Prop)

-- Define the containment relation between a line and a plane
variable (contained_in : L → P → Prop)

-- State the theorem
theorem line_perp_plane_necessity_not_sufficiency
  (m n : L) (α : P) (h_contained : contained_in n α) :
  (perpendicular_line_plane m α → perpendicular_lines m n) ∧
  ∃ (m' n' : L) (α' : P),
    contained_in n' α' ∧
    perpendicular_lines m' n' ∧
    ¬perpendicular_line_plane m' α' :=
sorry

end line_perp_plane_necessity_not_sufficiency_l1538_153885


namespace journey_distance_l1538_153869

/-- Proves that the total distance of a journey is 35 miles given specific conditions -/
theorem journey_distance (speed : ℝ) (time : ℝ) (total_portions : ℕ) (covered_portions : ℕ) :
  speed = 40 →
  time = 0.7 →
  total_portions = 5 →
  covered_portions = 4 →
  (speed * time) / covered_portions * total_portions = 35 :=
by sorry

end journey_distance_l1538_153869


namespace cubic_roots_difference_squared_l1538_153865

theorem cubic_roots_difference_squared (r s : ℝ) : 
  (∃ c : ℝ, r^3 - 2*r + c = 0 ∧ s^3 - 2*s + c = 0 ∧ 1^3 - 2*1 + c = 0) →
  (r - s)^2 = 5 := by
sorry

end cubic_roots_difference_squared_l1538_153865


namespace vet_spay_ratio_l1538_153862

theorem vet_spay_ratio (total_animals : ℕ) (cats : ℕ) (dogs : ℕ) :
  total_animals = 21 →
  cats = 7 →
  dogs = total_animals - cats →
  (dogs : ℚ) / (cats : ℚ) = 2 / 1 :=
by sorry

end vet_spay_ratio_l1538_153862


namespace sqrt_sum_expression_l1538_153895

theorem sqrt_sum_expression (a : ℝ) (h : a ≥ 1) :
  Real.sqrt (a + 2 * Real.sqrt (a - 1)) + Real.sqrt (a - 2 * Real.sqrt (a - 1)) =
    if 1 ≤ a ∧ a ≤ 2 then 2 else 2 * Real.sqrt (a - 1) :=
by sorry

end sqrt_sum_expression_l1538_153895


namespace juice_problem_l1538_153890

/-- Given the number of oranges per glass and the total number of oranges,
    calculate the number of glasses of juice. -/
def glasses_of_juice (oranges_per_glass : ℕ) (total_oranges : ℕ) : ℕ :=
  total_oranges / oranges_per_glass

theorem juice_problem :
  glasses_of_juice 2 12 = 6 := by
  sorry

end juice_problem_l1538_153890


namespace point_coordinates_l1538_153808

theorem point_coordinates (x y : ℝ) (h1 : x > 0) (h2 : y > 0) (h3 : y = 2) (h4 : x = 4) :
  (x, y) = (4, 2) := by
  sorry

end point_coordinates_l1538_153808


namespace quadratic_equation_roots_l1538_153824

theorem quadratic_equation_roots (k : ℝ) : 
  (∃ x : ℝ, x^2 - (k + 1) * x - 6 = 0 ∧ x = 2) →
  (∃ y : ℝ, y^2 - (k + 1) * y - 6 = 0 ∧ y = -3 ∧ k = -2) :=
by sorry

end quadratic_equation_roots_l1538_153824


namespace root_value_theorem_l1538_153881

theorem root_value_theorem (a : ℝ) (h : a^2 + 2*a - 1 = 0) : 2*a^2 + 4*a - 2024 = -2022 := by
  sorry

end root_value_theorem_l1538_153881


namespace linear_function_k_range_l1538_153860

theorem linear_function_k_range (k b : ℝ) :
  k ≠ 0 →
  (2 * k + b = -3) →
  (0 < b ∧ b < 1) →
  (-2 < k ∧ k < -3/2) :=
by sorry

end linear_function_k_range_l1538_153860


namespace smallest_five_digit_divisible_by_first_five_primes_l1538_153804

theorem smallest_five_digit_divisible_by_first_five_primes : 
  ∃ n : ℕ, 
    n ≥ 10000 ∧ 
    n < 100000 ∧ 
    2 ∣ n ∧ 
    3 ∣ n ∧ 
    5 ∣ n ∧ 
    7 ∣ n ∧ 
    11 ∣ n ∧ 
    ∀ m : ℕ, 
      m ≥ 10000 ∧ 
      m < 100000 ∧ 
      2 ∣ m ∧ 
      3 ∣ m ∧ 
      5 ∣ m ∧ 
      7 ∣ m ∧ 
      11 ∣ m → 
      n ≤ m :=
by
  -- The proof goes here
  sorry

end smallest_five_digit_divisible_by_first_five_primes_l1538_153804


namespace divisibility_condition_iff_n_le_3_l1538_153858

/-- A complete graph with n vertices -/
structure CompleteGraph (n : ℕ) where
  vertices : Fin n
  edges : Fin (n.choose 2)

/-- A labeling of edges with consecutive natural numbers -/
def EdgeLabeling (n : ℕ) := Fin (n.choose 2) → ℕ

/-- Condition for divisibility in a path of length 3 -/
def DivisibilityCondition (g : CompleteGraph n) (l : EdgeLabeling n) : Prop :=
  ∀ (a b c : Fin (n.choose 2)),
    (l b) ∣ (Nat.gcd (l a) (l c))

/-- Main theorem: The divisibility condition can be satisfied if and only if n ≤ 3 -/
theorem divisibility_condition_iff_n_le_3 (n : ℕ) :
  (∃ (g : CompleteGraph n) (l : EdgeLabeling n),
    DivisibilityCondition g l ∧
    (∀ i : Fin (n.choose 2), l i = i.val + 1)) ↔
  n ≤ 3 :=
sorry

end divisibility_condition_iff_n_le_3_l1538_153858


namespace inequality_of_powers_l1538_153857

theorem inequality_of_powers (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  a^(2*a) * b^(2*b) * c^(2*c) ≥ a^(b+c) * b^(a+c) * c^(a+b) := by
  sorry

end inequality_of_powers_l1538_153857


namespace decimal_multiplication_l1538_153822

theorem decimal_multiplication : (0.8 : ℝ) * 0.12 = 0.096 := by
  sorry

end decimal_multiplication_l1538_153822


namespace weight_of_four_cakes_l1538_153854

/-- The weight of a cake in grams -/
def cake_weight : ℕ := sorry

/-- The weight of a piece of bread in grams -/
def bread_weight : ℕ := sorry

/-- Theorem stating the weight of 4 cakes -/
theorem weight_of_four_cakes : 4 * cake_weight = 800 :=
by
  have h1 : 3 * cake_weight + 5 * bread_weight = 1100 := sorry
  have h2 : cake_weight = bread_weight + 100 := sorry
  sorry

#check weight_of_four_cakes

end weight_of_four_cakes_l1538_153854


namespace range_of_3a_minus_b_l1538_153878

theorem range_of_3a_minus_b (a b : ℝ) 
  (h1 : 1 ≤ a + b) (h2 : a + b ≤ 4) 
  (h3 : -1 ≤ a - b) (h4 : a - b ≤ 2) : 
  (∃ (x y : ℝ), (1 ≤ x + y ∧ x + y ≤ 4 ∧ -1 ≤ x - y ∧ x - y ≤ 2 ∧ 3*x - y = -1)) ∧
  (∃ (x y : ℝ), (1 ≤ x + y ∧ x + y ≤ 4 ∧ -1 ≤ x - y ∧ x - y ≤ 2 ∧ 3*x - y = 8)) ∧
  (∀ (x y : ℝ), 1 ≤ x + y → x + y ≤ 4 → -1 ≤ x - y → x - y ≤ 2 → -1 ≤ 3*x - y ∧ 3*x - y ≤ 8) :=
by sorry

end range_of_3a_minus_b_l1538_153878


namespace hyperbola_equation_proof_l1538_153849

/-- The focal length of the hyperbola -/
def focal_length : ℝ := 4

/-- The equation of circle C₂ -/
def circle_equation (x y : ℝ) : Prop := (x + 2)^2 + y^2 = 1

/-- The equation of hyperbola C₁ -/
def hyperbola_equation (a b x y : ℝ) : Prop := x^2 / a^2 - y^2 / b^2 = 1

/-- The asymptotes of the hyperbola -/
def asymptote_equation (a b x y : ℝ) : Prop := b * x = a * y ∨ b * x = -a * y

/-- The asymptotes are tangent to the circle -/
def asymptotes_tangent_to_circle (a b : ℝ) : Prop :=
  ∀ x y, asymptote_equation a b x y → (abs (-2 * b) / Real.sqrt (a^2 + b^2) = 1)

theorem hyperbola_equation_proof (a b : ℝ) (ha : a > 0) (hb : b > 0) 
  (h_focal : a^2 - b^2 = focal_length^2 / 4)
  (h_tangent : asymptotes_tangent_to_circle a b) :
  ∀ x y, hyperbola_equation a b x y ↔ x^2 / 3 - y^2 = 1 :=
sorry

end hyperbola_equation_proof_l1538_153849


namespace expression_evaluation_l1538_153873

theorem expression_evaluation : 
  let a : ℤ := -2
  (a - 1)^2 - a*(a + 3) + 2*(a + 2)*(a - 2) = 11 := by
  sorry

end expression_evaluation_l1538_153873


namespace opposite_pairs_l1538_153839

theorem opposite_pairs :
  (- (-2) = - (- (-2))) ∧
  ((-1)^2 = - ((-1)^2)) ∧
  ((-2)^3 ≠ -6) ∧
  ((-2)^7 = -2^7) := by
  sorry

end opposite_pairs_l1538_153839


namespace B_power_106_l1538_153836

def B : Matrix (Fin 3) (Fin 3) ℤ := ![![0, 1, 0], ![0, 0, -1], ![0, 1, 0]]

theorem B_power_106 : B^106 = ![![0, 0, -1], ![0, -1, 0], ![0, 0, -1]] := by sorry

end B_power_106_l1538_153836


namespace two_approve_probability_l1538_153842

/-- The probability of a voter approving the mayor's work -/
def approval_rate : ℝ := 0.6

/-- The number of voters randomly selected -/
def sample_size : ℕ := 4

/-- The number of approving voters we're interested in -/
def target_approvals : ℕ := 2

/-- The probability of exactly two out of four randomly selected voters approving the mayor's work -/
def prob_two_approve : ℝ := Nat.choose sample_size target_approvals * approval_rate ^ target_approvals * (1 - approval_rate) ^ (sample_size - target_approvals)

theorem two_approve_probability :
  prob_two_approve = 0.864 := by sorry

end two_approve_probability_l1538_153842


namespace no_solution_exists_l1538_153820

theorem no_solution_exists (a c : ℝ) : ¬∃ x : ℝ, 
  ((a + x) / 2 = 110) ∧ 
  ((x + c) / 2 = 170) ∧ 
  (a - c = 120) := by
sorry

end no_solution_exists_l1538_153820


namespace semicircle_chord_length_l1538_153806

theorem semicircle_chord_length (R a b : ℝ) (h1 : R > 0) (h2 : a > 0) (h3 : b > 0) 
  (h4 : a + b = R) (h5 : (π/2) * (R^2 - a^2 - b^2) = 10*π) : 
  ∃ (chord_length : ℝ), chord_length = 2 * Real.sqrt 10 := by
  sorry

end semicircle_chord_length_l1538_153806


namespace ellipse_line_intersection_tangent_line_l1538_153897

/-- The ellipse E -/
def E (x y : ℝ) : Prop := y^2 / 8 + x^2 / 4 = 1

/-- The line l -/
def l (x y : ℝ) : Prop := x + y - 3 = 0

/-- The function f -/
def f (x : ℝ) : ℝ := x^2 - 3*x + 4

theorem ellipse_line_intersection (A B : ℝ × ℝ) :
  E A.1 A.2 ∧ E B.1 B.2 ∧ l A.1 A.2 ∧ l B.1 B.2 ∧ A ≠ B ∧
  (A.1 + B.1) / 2 = 1 ∧ (A.2 + B.2) / 2 = 2 →
  ∀ x y, l x y ↔ x + y - 3 = 0 :=
sorry

theorem tangent_line (P : ℝ × ℝ) :
  P = (1, 2) ∧ (∀ x, f x = x^2 - 3*x + 4) ∧
  (∀ x y, l x y ↔ x + y - 3 = 0) →
  ∃ a b, f P.1 = P.2 ∧ (deriv f) P.1 = -1 ∧
  ∀ x, f x = x^2 - a*x + b :=
sorry

end ellipse_line_intersection_tangent_line_l1538_153897


namespace circle_parabola_intersection_l1538_153823

/-- The circle and parabola intersect at exactly one point if and only if b = 1/4 -/
theorem circle_parabola_intersection (b : ℝ) : 
  (∃! p : ℝ × ℝ, p.1^2 + p.2^2 = 4*b^2 ∧ p.2 = p.1^2 - 2*b) ↔ b = 1/4 :=
by sorry

end circle_parabola_intersection_l1538_153823


namespace smallest_angle_measure_l1538_153835

/-- A trapezoid with angles in arithmetic progression -/
structure ArithmeticTrapezoid where
  a : ℝ  -- smallest angle
  d : ℝ  -- common difference
  angle_sum : a + (a + d) + (a + 2*d) + (a + 3*d) = 360
  largest_angle : a + 3*d = 150

theorem smallest_angle_measure (t : ArithmeticTrapezoid) : t.a = 30 := by
  sorry

end smallest_angle_measure_l1538_153835


namespace opposite_of_fraction_l1538_153875

theorem opposite_of_fraction (n : ℕ) (h : n ≠ 0) : 
  -(1 : ℚ) / n = -(1 / n) := by sorry

end opposite_of_fraction_l1538_153875


namespace greatest_divisor_with_remainders_l1538_153859

theorem greatest_divisor_with_remainders : 
  let a := 6215 - 23
  let b := 7373 - 29
  let c := 8927 - 35
  Nat.gcd a (Nat.gcd b c) = 36 := by
  sorry

end greatest_divisor_with_remainders_l1538_153859


namespace max_sin_sum_60_degrees_l1538_153861

open Real

theorem max_sin_sum_60_degrees (x y : ℝ) : 
  0 < x → x < π/2 →
  0 < y → y < π/2 →
  x + y = π/3 →
  (∀ a b : ℝ, 0 < a → a < π/2 → 0 < b → b < π/2 → a + b = π/3 → sin a + sin b ≤ sin x + sin y) →
  sin x + sin y = 1 := by
sorry


end max_sin_sum_60_degrees_l1538_153861


namespace multiply_specific_numbers_l1538_153863

theorem multiply_specific_numbers : 469138 * 9999 = 4690692862 := by
  sorry

end multiply_specific_numbers_l1538_153863


namespace james_lifting_weight_l1538_153883

/-- Calculates the weight James can lift with straps for 10 meters given initial conditions -/
def weight_with_straps (initial_weight : ℝ) (distance_increase : ℝ) (short_distance_factor : ℝ) (strap_factor : ℝ) : ℝ :=
  let base_weight := initial_weight + distance_increase
  let short_distance_weight := base_weight * (1 + short_distance_factor)
  short_distance_weight * (1 + strap_factor)

/-- Theorem stating the final weight James can lift with straps for 10 meters -/
theorem james_lifting_weight :
  weight_with_straps 300 50 0.3 0.2 = 546 := by
  sorry

#eval weight_with_straps 300 50 0.3 0.2

end james_lifting_weight_l1538_153883


namespace second_discount_percentage_l1538_153829

theorem second_discount_percentage (original_price : ℝ) (first_discount : ℝ) (final_price : ℝ)
  (h1 : original_price = 400)
  (h2 : first_discount = 10)
  (h3 : final_price = 342) :
  let price_after_first_discount := original_price * (1 - first_discount / 100)
  let second_discount := (price_after_first_discount - final_price) / price_after_first_discount * 100
  second_discount = 5 := by
sorry

end second_discount_percentage_l1538_153829


namespace max_product_of_sum_constrained_naturals_l1538_153886

theorem max_product_of_sum_constrained_naturals
  (n k : ℕ) (h : k > 0) :
  let t : ℕ := n / k
  let r : ℕ := n % k
  ∃ (l : List ℕ),
    (l.length = k) ∧
    (l.sum = n) ∧
    (∀ (m : List ℕ), m.length = k → m.sum = n → l.prod ≥ m.prod) ∧
    (l.prod = (t + 1)^r * t^(k - r)) := by
  sorry

end max_product_of_sum_constrained_naturals_l1538_153886


namespace rahul_savings_l1538_153843

/-- Proves that given the conditions on Rahul's savings, the total amount saved is 180,000 Rs. -/
theorem rahul_savings (nsc ppf : ℕ) : 
  (1 / 3 : ℚ) * nsc = (1 / 2 : ℚ) * ppf →
  ppf = 72000 →
  nsc + ppf = 180000 := by
sorry

end rahul_savings_l1538_153843


namespace equation_solution_l1538_153803

theorem equation_solution : 
  ∃ x : ℚ, (5 * x - 2) / (6 * x - 6) = 3 / 4 ∧ x = -5 := by sorry

end equation_solution_l1538_153803


namespace patrick_pencil_purchase_l1538_153844

/-- The number of pencils Patrick purchased -/
def num_pencils : ℕ := 60

/-- The ratio of cost price to selling price -/
def cost_to_sell_ratio : ℚ := 1.3333333333333333

/-- The number of pencils whose selling price equals the total loss -/
def loss_in_pencils : ℕ := 20

theorem patrick_pencil_purchase :
  num_pencils = 60 ∧
  (cost_to_sell_ratio - 1) * num_pencils = loss_in_pencils :=
sorry

end patrick_pencil_purchase_l1538_153844


namespace zain_total_coins_l1538_153848

/-- Represents the number of coins of each type --/
structure CoinCount where
  quarters : ℕ
  dimes : ℕ
  nickels : ℕ
  pennies : ℕ
  halfDollars : ℕ

/-- Calculates the total number of coins --/
def totalCoins (coins : CoinCount) : ℕ :=
  coins.quarters + coins.dimes + coins.nickels + coins.pennies + coins.halfDollars

/-- Represents Emerie's coin count --/
def emerieCoins : CoinCount :=
  { quarters := 6
  , dimes := 7
  , nickels := 5
  , pennies := 10
  , halfDollars := 2 }

/-- Calculates Zain's coin count based on Emerie's --/
def zainCoins (emerie : CoinCount) : CoinCount :=
  { quarters := emerie.quarters + 10
  , dimes := emerie.dimes + 10
  , nickels := emerie.nickels + 10
  , pennies := emerie.pennies + 10
  , halfDollars := emerie.halfDollars + 10 }

/-- Theorem: Zain has 80 coins in total --/
theorem zain_total_coins : totalCoins (zainCoins emerieCoins) = 80 := by
  sorry

end zain_total_coins_l1538_153848


namespace largest_angle_obtuse_isosceles_triangle_l1538_153899

/-- An obtuse isosceles triangle with one angle measuring 20 degrees has its largest angle measuring 140 degrees. -/
theorem largest_angle_obtuse_isosceles_triangle (A B C : ℝ) :
  A = 20 → -- Angle A measures 20 degrees
  A + B + C = 180 → -- Sum of angles in a triangle is 180 degrees
  (A = C ∨ A = B) → -- Isosceles triangle condition
  A < 90 ∧ B < 90 ∧ C < 90 → -- Obtuse triangle condition (no right angle)
  A ≤ B ∧ A ≤ C → -- A is not the largest angle
  max B C = 140 := by -- The largest angle (either B or C) is 140 degrees
sorry

end largest_angle_obtuse_isosceles_triangle_l1538_153899


namespace f_ratio_calc_l1538_153805

axiom f : ℝ → ℝ

axiom f_property : ∀ (a b : ℝ), b^2 * f a = a^2 * f b

axiom f2_nonzero : f 2 ≠ 0

theorem f_ratio_calc : (f 6 - f 3) / f 2 = 27 / 4 := by
  sorry

end f_ratio_calc_l1538_153805


namespace temperature_conversion_l1538_153868

theorem temperature_conversion (t k r : ℝ) 
  (eq1 : t = 5/9 * (k - 32))
  (eq2 : r = 3*t)
  (eq3 : r = 150) : 
  k = 122 := by
sorry

end temperature_conversion_l1538_153868


namespace workout_calculation_l1538_153825

-- Define the exercise parameters
def bicep_curls_weight : ℕ := 20
def bicep_curls_dumbbells : ℕ := 2
def bicep_curls_reps : ℕ := 10
def bicep_curls_sets : ℕ := 3

def shoulder_press_weight1 : ℕ := 30
def shoulder_press_weight2 : ℕ := 40
def shoulder_press_reps : ℕ := 8
def shoulder_press_sets : ℕ := 2

def lunges_weight : ℕ := 30
def lunges_dumbbells : ℕ := 2
def lunges_reps : ℕ := 12
def lunges_sets : ℕ := 4

def bench_press_weight : ℕ := 40
def bench_press_dumbbells : ℕ := 2
def bench_press_reps : ℕ := 6
def bench_press_sets : ℕ := 3

-- Define the theorem
theorem workout_calculation :
  -- Total weight calculation
  (bicep_curls_weight * bicep_curls_dumbbells * bicep_curls_reps * bicep_curls_sets) +
  ((shoulder_press_weight1 + shoulder_press_weight2) * shoulder_press_reps * shoulder_press_sets) +
  (lunges_weight * lunges_dumbbells * lunges_reps * lunges_sets) +
  (bench_press_weight * bench_press_dumbbells * bench_press_reps * bench_press_sets) = 6640 ∧
  -- Average weight per rep for each exercise
  bicep_curls_weight * bicep_curls_dumbbells = 40 ∧
  shoulder_press_weight1 + shoulder_press_weight2 = 70 ∧
  lunges_weight * lunges_dumbbells = 60 ∧
  bench_press_weight * bench_press_dumbbells = 80 := by
  sorry

end workout_calculation_l1538_153825


namespace det_specific_matrix_l1538_153812

theorem det_specific_matrix :
  let A : Matrix (Fin 3) (Fin 3) ℝ := !![2, -4, 2; 0, 6, -1; 5, -3, 1]
  Matrix.det A = -34 := by
    sorry

end det_specific_matrix_l1538_153812
