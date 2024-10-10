import Mathlib

namespace mixed_number_calculation_l2286_228662

theorem mixed_number_calculation : 7 * (12 + 2/5) - 3 = 83.8 := by
  sorry

end mixed_number_calculation_l2286_228662


namespace smallest_square_area_for_rectangles_l2286_228661

/-- Represents a rectangle with width and height -/
structure Rectangle where
  width : ℕ
  height : ℕ

/-- Calculates the area of a square -/
def square_area (side : ℕ) : ℕ := side * side

/-- Checks if a square can contain two rectangles -/
def can_contain_rectangles (side : ℕ) (rect1 rect2 : Rectangle) : Prop :=
  (max rect1.width rect2.width ≤ side) ∧ (rect1.height + rect2.height ≤ side)

theorem smallest_square_area_for_rectangles :
  ∃ (side : ℕ),
    let rect1 : Rectangle := ⟨3, 4⟩
    let rect2 : Rectangle := ⟨4, 5⟩
    can_contain_rectangles side rect1 rect2 ∧
    square_area side = 49 ∧
    ∀ (smaller_side : ℕ), smaller_side < side →
      ¬ can_contain_rectangles smaller_side rect1 rect2 := by
  sorry

end smallest_square_area_for_rectangles_l2286_228661


namespace general_quadratic_is_quadratic_specific_quadratic_is_quadratic_l2286_228678

/-- Definition of a quadratic equation -/
def is_quadratic_equation (f : ℝ → ℝ) : Prop :=
  ∃ (a b c : ℝ), a ≠ 0 ∧ ∀ x, f x = a * x^2 + b * x + c

/-- The equation ax² + bx + c = 0 is quadratic -/
theorem general_quadratic_is_quadratic (a b c : ℝ) (ha : a ≠ 0) :
  is_quadratic_equation (λ x => a * x^2 + b * x + c) :=
sorry

/-- The equation x² - 4 = 0 is quadratic -/
theorem specific_quadratic_is_quadratic :
  is_quadratic_equation (λ x => x^2 - 4) :=
sorry

end general_quadratic_is_quadratic_specific_quadratic_is_quadratic_l2286_228678


namespace total_eyes_l2286_228652

/-- The total number of eyes given the number of boys, girls, cats, and spiders -/
theorem total_eyes (boys girls cats spiders : ℕ) : 
  boys = 23 → 
  girls = 18 → 
  cats = 10 → 
  spiders = 5 → 
  boys * 2 + girls * 2 + cats * 2 + spiders * 8 = 142 := by
  sorry


end total_eyes_l2286_228652


namespace largest_stamps_per_page_l2286_228664

theorem largest_stamps_per_page (a b c : ℕ) 
  (ha : a = 924) (hb : b = 1386) (hc : c = 1848) : 
  Nat.gcd a (Nat.gcd b c) = 462 := by
  sorry

end largest_stamps_per_page_l2286_228664


namespace max_c_value_l2286_228672

-- Define the function f
def f (a b c d x : ℝ) : ℝ := a * x^3 + 2 * b * x^2 + 3 * c * x + 4 * d

-- Define the conditions
def is_valid_function (a b c d : ℝ) : Prop :=
  a < 0 ∧ c > 0 ∧
  (∀ x, f a b c d x = -f a b c d (-x)) ∧
  (∀ x ∈ Set.Icc 0 1, f a b c d x ∈ Set.Icc 0 1)

-- Theorem statement
theorem max_c_value (a b c d : ℝ) (h : is_valid_function a b c d) :
  c ≤ Real.sqrt 3 / 2 ∧ ∃ a₀ b₀ d₀, is_valid_function a₀ b₀ (Real.sqrt 3 / 2) d₀ :=
sorry

end max_c_value_l2286_228672


namespace triangle_angle_and_perimeter_l2286_228619

open Real

/-- Triangle ABC with sides a, b, c opposite to angles A, B, C respectively -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

theorem triangle_angle_and_perimeter (t : Triangle) (h : t.b^2 + t.c^2 - t.a^2 = t.b * t.c) :
  t.A = π / 3 ∧ 
  (t.a = Real.sqrt 3 → ∃ y : ℝ, y > 2 * Real.sqrt 3 ∧ y ≤ 3 * Real.sqrt 3 ∧ y = t.a + t.b + t.c) :=
by sorry

end triangle_angle_and_perimeter_l2286_228619


namespace lollipops_left_l2286_228657

theorem lollipops_left (initial : ℕ) (eaten : ℕ) (h1 : initial = 12) (h2 : eaten = 5) :
  initial - eaten = 7 := by
  sorry

end lollipops_left_l2286_228657


namespace g_of_three_l2286_228641

/-- Given a function g : ℝ → ℝ satisfying g(x) - 3 * g(1/x) = 3^x + 1 for all x ≠ 0,
    prove that g(3) = -17/4 -/
theorem g_of_three (g : ℝ → ℝ) 
    (h : ∀ x ≠ 0, g x - 3 * g (1/x) = 3^x + 1) : 
    g 3 = -17/4 := by
  sorry

end g_of_three_l2286_228641


namespace baseball_average_runs_l2286_228674

theorem baseball_average_runs (games : ℕ) (runs_once : ℕ) (runs_twice : ℕ) (runs_thrice : ℕ)
  (h_games : games = 6)
  (h_once : runs_once = 1)
  (h_twice : runs_twice = 4)
  (h_thrice : runs_thrice = 5)
  (h_pattern : 1 * runs_once + 2 * runs_twice + 3 * runs_thrice = games * 4) :
  (1 * runs_once + 2 * runs_twice + 3 * runs_thrice) / games = 4 := by
sorry

end baseball_average_runs_l2286_228674


namespace intersection_of_A_and_B_l2286_228616

def A : Set ℕ := {1, 2, 3, 4}

def B : Set ℕ := {x : ℕ | ∃ n ∈ A, x = 2 * n}

theorem intersection_of_A_and_B : A ∩ B = {2, 4} := by sorry

end intersection_of_A_and_B_l2286_228616


namespace remi_tomato_seedlings_l2286_228683

theorem remi_tomato_seedlings (day1 : ℕ) (total : ℕ) : 
  day1 = 200 →
  total = 5000 →
  (day1 + 2 * day1 + 3 * (2 * day1) + 4 * (2 * day1) = total) →
  3 * (2 * day1) + 4 * (2 * day1) = 4400 :=
by
  sorry

end remi_tomato_seedlings_l2286_228683


namespace right_triangle_leg_construction_l2286_228645

theorem right_triangle_leg_construction (c m : ℝ) (h_positive : c > 0) :
  ∃ (a b p : ℝ),
    a > 0 ∧ b > 0 ∧ p > 0 ∧
    a^2 + b^2 = c^2 ∧
    a^2 - b^2 = 4 * m^2 ∧
    p = (c * (1 + Real.sqrt 5)) / 4 :=
by sorry

end right_triangle_leg_construction_l2286_228645


namespace max_product_sum_22_l2286_228640

/-- A list of distinct natural numbers -/
def DistinctNatList := List Nat

/-- Check if a list contains distinct elements -/
def isDistinct (l : List Nat) : Prop :=
  l.Nodup

/-- Sum of elements in a list -/
def listSum (l : List Nat) : Nat :=
  l.sum

/-- Product of elements in a list -/
def listProduct (l : List Nat) : Nat :=
  l.prod

/-- The maximum product of distinct natural numbers that sum to 22 -/
def maxProductSum22 : Nat :=
  1008

theorem max_product_sum_22 :
  ∀ (l : DistinctNatList), 
    isDistinct l → 
    listSum l = 22 → 
    listProduct l ≤ maxProductSum22 :=
by
  sorry

end max_product_sum_22_l2286_228640


namespace longest_segment_in_cylinder_l2286_228690

/-- The longest segment in a cylinder --/
theorem longest_segment_in_cylinder (r h : ℝ) (hr : r = 4) (hh : h = 10) :
  Real.sqrt ((2 * r) ^ 2 + h ^ 2) = 2 * Real.sqrt 41 := by
  sorry

end longest_segment_in_cylinder_l2286_228690


namespace quadratic_negative_root_l2286_228631

/-- The quadratic equation ax^2 + 2x + 1 = 0 has at least one negative root
    if and only if a < 0 or 0 < a ≤ 1 -/
theorem quadratic_negative_root (a : ℝ) (ha : a ≠ 0) :
  (∃ x : ℝ, x < 0 ∧ a * x^2 + 2 * x + 1 = 0) ↔ (a < 0 ∨ (0 < a ∧ a ≤ 1)) := by
  sorry

end quadratic_negative_root_l2286_228631


namespace lilibeth_baskets_l2286_228663

/-- The number of strawberries each basket holds -/
def strawberries_per_basket : ℕ := 50

/-- The total number of strawberries picked by Lilibeth and her friends -/
def total_strawberries : ℕ := 1200

/-- The number of people picking strawberries (Lilibeth and her three friends) -/
def number_of_pickers : ℕ := 4

/-- The number of baskets Lilibeth filled -/
def baskets_filled : ℕ := 6

theorem lilibeth_baskets :
  strawberries_per_basket * number_of_pickers * baskets_filled = total_strawberries :=
by sorry

end lilibeth_baskets_l2286_228663


namespace original_denominator_proof_l2286_228691

theorem original_denominator_proof (d : ℤ) : 
  (3 : ℚ) / d ≠ 0 →
  (3 + 7 : ℚ) / (d + 7) = 2 / 5 →
  d = 18 := by
sorry

end original_denominator_proof_l2286_228691


namespace trigonometric_identities_l2286_228689

theorem trigonometric_identities (α : Real) (h : Real.tan (π + α) = -1/2) :
  (2 * Real.cos (π - α) - 3 * Real.sin (π + α)) / (4 * Real.cos (α - 2*π) + Real.sin (4*π - α)) = -7/9 ∧
  Real.sin (α - 7*π) * Real.cos (α + 5*π) = -2/5 := by sorry

end trigonometric_identities_l2286_228689


namespace car_selection_proof_l2286_228694

theorem car_selection_proof (num_cars : ℕ) (num_clients : ℕ) (selections_per_client : ℕ)
  (h1 : num_cars = 10)
  (h2 : num_clients = 15)
  (h3 : selections_per_client = 2)
  (h4 : ∀ i, 1 ≤ i ∧ i ≤ num_cars → ∃ (x : ℕ), x > 0) :
  ∀ i, 1 ≤ i ∧ i ≤ num_cars → ∃ (x : ℕ), x = 3 :=
by sorry

end car_selection_proof_l2286_228694


namespace polynomial_expansion_l2286_228667

theorem polynomial_expansion (x : ℝ) : 
  (x - 3) * (x + 5) * (x^2 + 9) = x^4 + 2*x^3 - 6*x^2 + 18*x - 135 := by
  sorry

end polynomial_expansion_l2286_228667


namespace point_location_l2286_228698

theorem point_location (m : ℝ) :
  (m < 0 ∧ 1 > 0) →  -- P (m, 1) is in the second quadrant
  (-m > 0 ∧ 0 = 0)   -- Q (-m, 0) is on the positive half of the x-axis
  := by sorry

end point_location_l2286_228698


namespace product_of_sums_powers_specific_product_evaluation_l2286_228623

theorem product_of_sums_powers (a b : ℕ) : 
  (a + b) * (a^2 + b^2) * (a^4 + b^4) * (a^8 + b^8) = 
  (1/2 : ℚ) * ((a^16 : ℚ) - (b^16 : ℚ)) :=
by sorry

theorem specific_product_evaluation : 
  (3 + 1) * (3^2 + 1^2) * (3^4 + 1^4) * (3^8 + 1^8) = 21523360 :=
by sorry

end product_of_sums_powers_specific_product_evaluation_l2286_228623


namespace gcd_of_A_and_B_l2286_228639

def A : ℕ := 2 * 3 * 5
def B : ℕ := 2 * 2 * 5 * 7

theorem gcd_of_A_and_B : Nat.gcd A B = 10 := by
  sorry

end gcd_of_A_and_B_l2286_228639


namespace cube_construction_condition_l2286_228635

/-- A brick is composed of twelve unit cubes arranged in a three-step staircase of width 2. -/
def Brick : Type := Unit

/-- Predicate indicating whether it's possible to build a cube of side length n using Bricks. -/
def CanBuildCube (n : ℕ) : Prop := sorry

theorem cube_construction_condition (n : ℕ) : 
  CanBuildCube n ↔ 12 ∣ n :=
sorry

end cube_construction_condition_l2286_228635


namespace min_value_theorem_l2286_228693

theorem min_value_theorem (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : 2*x + 3*y = 5*x*y) :
  3*x + 2*y ≥ 5 ∧ ∃ (x₀ y₀ : ℝ), x₀ > 0 ∧ y₀ > 0 ∧ 2*x₀ + 3*y₀ = 5*x₀*y₀ ∧ 3*x₀ + 2*y₀ = 5 := by
  sorry

end min_value_theorem_l2286_228693


namespace function_proof_l2286_228677

noncomputable section

-- Define the function f
def f (a b : ℝ) (x : ℝ) := a * x^3 + b * x^2 - 3 * x

-- Define the derivative of f
def f' (a b : ℝ) (x : ℝ) := 3 * a * x^2 + 2 * b * x - 3

-- Define the function g
def g (a b : ℝ) (x : ℝ) := (1/3) * f a b x - 6 * Real.log x

-- Define the curve y = xf(x)
def curve (a b : ℝ) (x : ℝ) := x * (f a b x)

theorem function_proof (a b : ℝ) :
  (∀ x, f' a b x = f' a b (-x)) →  -- f' is even
  f' a b 1 = 0 →                   -- f'(1) = 0
  (∃ c, ∀ x₁ x₂, x₁ ∈ Set.Icc 1 2 → x₂ ∈ Set.Icc 1 2 → 
    |g a b x₁ - g a b x₂| ≤ c) →   -- |g(x₁) - g(x₂)| ≤ c for x₁, x₂ ∈ [1, 2]
  (∀ x, f a b x = x^3 - 3*x) ∧     -- f(x) = x³ - 3x
  (∃ c_min, c_min = -4/3 + 6 * Real.log 2 ∧ 
    ∀ c', (∀ x₁ x₂, x₁ ∈ Set.Icc 1 2 → x₂ ∈ Set.Icc 1 2 → 
      |g a b x₁ - g a b x₂| ≤ c') → c' ≥ c_min) ∧  -- Minimum value of c
  (∃ s : Set ℝ, s = {4, 3/4 - 4 * Real.sqrt 2} ∧ 
    ∀ m, m ∈ s ↔ (∃ x₁ x₂ x₃, x₁ ≠ x₂ ∧ x₁ ≠ x₃ ∧ x₂ ≠ x₃ ∧
      curve a b x₁ = m * x₁ - 2 * m ∧
      curve a b x₂ = m * x₂ - 2 * m ∧
      curve a b x₃ = m * x₃ - 2 * m)) -- Set of m values for three tangent lines
  := by sorry

end function_proof_l2286_228677


namespace x_equals_six_l2286_228643

theorem x_equals_six (a b x : ℝ) (h1 : 2^a = x) (h2 : 3^b = x) (h3 : 1/a + 1/b = 1) : x = 6 := by
  sorry

end x_equals_six_l2286_228643


namespace hyperbola_asymptote_slope_l2286_228682

/-- The slope of the asymptotes of a hyperbola with specific properties -/
theorem hyperbola_asymptote_slope (a b c : ℝ) (ha : a > 0) (hb : b > 0) :
  let A₁ : ℝ × ℝ := (-a, 0)
  let A₂ : ℝ × ℝ := (a, 0)
  let B : ℝ × ℝ := (c, b^2/a)
  let C : ℝ × ℝ := (c, -b^2/a)
  (∀ x y, x^2/a^2 - y^2/b^2 = 1 → (x = c ∧ (y = b^2/a ∨ y = -b^2/a))) →
  ((B.2 - A₁.2) * (C.2 - A₂.2) = -(B.1 - A₁.1) * (C.1 - A₂.1)) →
  (∀ x, (x : ℝ) = x ∨ (x : ℝ) = -x) :=
by sorry

end hyperbola_asymptote_slope_l2286_228682


namespace positive_real_solutions_range_l2286_228684

theorem positive_real_solutions_range (a : ℝ) : 
  (∀ x : ℝ, x > 0 → Real.pi ^ x = (a + 1) / (2 - a)) ↔ 1/2 < a ∧ a < 2 := by
  sorry

end positive_real_solutions_range_l2286_228684


namespace rectangle_ratio_is_two_l2286_228647

/-- Represents the configuration of rectangles around a square -/
structure RectangleSquareArrangement where
  inner_square_side : ℝ
  rectangle_short_side : ℝ
  rectangle_long_side : ℝ

/-- The conditions of the arrangement -/
def valid_arrangement (a : RectangleSquareArrangement) : Prop :=
  -- The outer square's side length is 3 times the inner square's side length
  a.inner_square_side + 2 * a.rectangle_short_side = 3 * a.inner_square_side ∧
  -- The outer square's side length is also the sum of the long side and short side
  a.rectangle_long_side + a.rectangle_short_side = 3 * a.inner_square_side

/-- The theorem to be proved -/
theorem rectangle_ratio_is_two (a : RectangleSquareArrangement) 
    (h : valid_arrangement a) : 
    a.rectangle_long_side / a.rectangle_short_side = 2 := by
  sorry

end rectangle_ratio_is_two_l2286_228647


namespace shipping_cost_calculation_l2286_228679

def fish_weight : ℕ := 540
def crate_capacity : ℕ := 30
def crate_cost : ℚ := 3/2

theorem shipping_cost_calculation :
  (fish_weight / crate_capacity) * crate_cost = 27 := by
  sorry

end shipping_cost_calculation_l2286_228679


namespace inverse_of_matrix_A_l2286_228648

def A : Matrix (Fin 2) (Fin 2) ℝ := !![5, 15; 2, 6]

theorem inverse_of_matrix_A (h : Matrix.det A = 0) :
  A⁻¹ = !![0, 0; 0, 0] := by
  sorry

end inverse_of_matrix_A_l2286_228648


namespace no_integer_root_2016_l2286_228692

/-- A cubic polynomial with integer coefficients -/
def cubic_poly (a b c d : ℤ) : ℤ → ℤ := fun x ↦ a * x^3 + b * x^2 + c * x + d

theorem no_integer_root_2016 (a b c d : ℤ) :
  cubic_poly a b c d 1 = 2015 →
  cubic_poly a b c d 2 = 2017 →
  ∀ k : ℤ, cubic_poly a b c d k ≠ 2016 := by
sorry

end no_integer_root_2016_l2286_228692


namespace calculator_probability_l2286_228680

/-- The probability of a specific number M appearing on the display
    when starting from a number N, where M < N -/
def prob_appear (N M : ℕ) : ℚ :=
  if M < N then 1 / (M + 1 : ℚ) else 0

/-- The probability of all numbers in a list appearing on the display
    when starting from a given number -/
def prob_all_appear (start : ℕ) (numbers : List ℕ) : ℚ :=
  numbers.foldl (fun acc n => acc * prob_appear start n) 1

theorem calculator_probability :
  prob_all_appear 2003 [1000, 100, 10, 1] = 1 / 2224222 := by
  sorry

end calculator_probability_l2286_228680


namespace volunteer_quota_allocation_l2286_228625

theorem volunteer_quota_allocation :
  let n : ℕ := 24  -- Total number of quotas
  let k : ℕ := 3   -- Number of venues
  let total_partitions : ℕ := Nat.choose (n - 1) (k - 1)
  let invalid_partitions : ℕ := (k - 1) * Nat.choose k 2 + 1
  total_partitions - invalid_partitions = 222 := by
  sorry

end volunteer_quota_allocation_l2286_228625


namespace prob_male_monday_female_tuesday_is_one_third_l2286_228610

/-- Represents the number of male volunteers -/
def num_men : ℕ := 2

/-- Represents the number of female volunteers -/
def num_women : ℕ := 2

/-- Represents the total number of volunteers -/
def total_volunteers : ℕ := num_men + num_women

/-- Represents the number of days for which volunteers are selected -/
def num_days : ℕ := 2

/-- Calculates the probability of selecting a male volunteer for Monday
    and a female volunteer for Tuesday -/
def prob_male_monday_female_tuesday : ℚ :=
  (num_men * num_women) / (total_volunteers * (total_volunteers - 1))

/-- Proves that the probability of selecting a male volunteer for Monday
    and a female volunteer for Tuesday is 1/3 -/
theorem prob_male_monday_female_tuesday_is_one_third :
  prob_male_monday_female_tuesday = 1 / 3 := by
  sorry

end prob_male_monday_female_tuesday_is_one_third_l2286_228610


namespace certain_number_problem_l2286_228686

theorem certain_number_problem (x : ℕ) (certain_number : ℕ) 
  (h1 : 9873 + x = certain_number) (h2 : x = 3327) : 
  certain_number = 13200 := by
  sorry

end certain_number_problem_l2286_228686


namespace math_club_election_l2286_228655

theorem math_club_election (total_candidates : ℕ) (past_officers : ℕ) (positions : ℕ) :
  total_candidates = 20 →
  past_officers = 9 →
  positions = 6 →
  (Nat.choose total_candidates positions - Nat.choose (total_candidates - past_officers) positions) = 38298 :=
by sorry

end math_club_election_l2286_228655


namespace aunt_age_proof_l2286_228650

def cori_age_today : ℕ := 3
def years_until_comparison : ℕ := 5

def aunt_age_today : ℕ := 19

theorem aunt_age_proof :
  (cori_age_today + years_until_comparison) * 3 = aunt_age_today + years_until_comparison :=
by sorry

end aunt_age_proof_l2286_228650


namespace original_salary_calculation_l2286_228620

/-- Proves that if a salary S is increased by 2% to result in €10,200, then S equals €10,000. -/
theorem original_salary_calculation (S : ℝ) : S * 1.02 = 10200 → S = 10000 := by
  sorry

end original_salary_calculation_l2286_228620


namespace negation_of_no_vegetarian_students_eat_at_cafeteria_l2286_228699

-- Define the universe of discourse
variable (Student : Type)

-- Define predicates
variable (isVegetarian : Student → Prop)
variable (eatsAtCafeteria : Student → Prop)

-- State the theorem
theorem negation_of_no_vegetarian_students_eat_at_cafeteria :
  ¬(∀ s : Student, isVegetarian s → ¬(eatsAtCafeteria s)) ↔
  ∃ s : Student, isVegetarian s ∧ eatsAtCafeteria s :=
by sorry


end negation_of_no_vegetarian_students_eat_at_cafeteria_l2286_228699


namespace kilmer_park_tree_height_l2286_228681

/-- Calculates the height of a tree in inches after a given number of years -/
def tree_height_in_inches (initial_height : ℕ) (annual_growth : ℕ) (years : ℕ) : ℕ :=
  (initial_height + annual_growth * years) * 12

/-- Theorem: The height of the tree in Kilmer Park after 8 years is 1104 inches -/
theorem kilmer_park_tree_height : tree_height_in_inches 52 5 8 = 1104 := by
  sorry

end kilmer_park_tree_height_l2286_228681


namespace parabola_circle_intersection_l2286_228668

/-- Theorem: For a parabola y = x^2 - ax - 3 (a ∈ ℝ) intersecting the x-axis at points A and B,
    and passing through point C(0, -3), if a circle passing through A, B, and C intersects
    the y-axis at point D(0, b), then b = 1. -/
theorem parabola_circle_intersection (a : ℝ) (A B : ℝ × ℝ) (b : ℝ) :
  let f : ℝ → ℝ := λ x ↦ x^2 - a*x - 3
  (f A.1 = 0 ∧ f B.1 = 0) →  -- A and B are on the x-axis
  (∃ D E F : ℝ, (D^2 + E^2 - 4*F > 0) ∧  -- Circle equation coefficients
    (∀ x y : ℝ, x^2 + y^2 + D*x + E*y + F = 0 ↔
      ((x = A.1 ∧ y = A.2) ∨ (x = B.1 ∧ y = B.2) ∨ (x = 0 ∧ y = -3) ∨ (x = 0 ∧ y = b)))) →
  b = 1 :=
by sorry

end parabola_circle_intersection_l2286_228668


namespace multiplication_proof_l2286_228611

theorem multiplication_proof (m : ℕ) : m = 32505 → m * 9999 = 325027405 := by
  sorry

end multiplication_proof_l2286_228611


namespace original_fraction_is_two_thirds_l2286_228688

theorem original_fraction_is_two_thirds 
  (x y : ℚ) 
  (h1 : x / (y + 1) = 1/2) 
  (h2 : (x + 1) / y = 1) : 
  x / y = 2/3 := by
  sorry

end original_fraction_is_two_thirds_l2286_228688


namespace average_age_calculation_l2286_228600

/-- The average age of a group of fifth-graders, their parents, and teachers -/
theorem average_age_calculation (n_students : ℕ) (n_parents : ℕ) (n_teachers : ℕ)
  (avg_age_students : ℚ) (avg_age_parents : ℚ) (avg_age_teachers : ℚ)
  (h_students : n_students = 30)
  (h_parents : n_parents = 50)
  (h_teachers : n_teachers = 10)
  (h_avg_students : avg_age_students = 10)
  (h_avg_parents : avg_age_parents = 40)
  (h_avg_teachers : avg_age_teachers = 35) :
  (n_students * avg_age_students + n_parents * avg_age_parents + n_teachers * avg_age_teachers) /
  (n_students + n_parents + n_teachers : ℚ) = 530 / 18 :=
by sorry

end average_age_calculation_l2286_228600


namespace sum_of_repeating_decimals_l2286_228627

/-- Represents a repeating decimal with a single digit repeating -/
def SingleDigitRepeatingDecimal (n : ℕ) : ℚ := n / 9

/-- Represents a repeating decimal with two digits repeating -/
def TwoDigitRepeatingDecimal (n : ℕ) : ℚ := n / 99

/-- The sum of 0.5̅ and 0.07̅ is equal to 62/99 -/
theorem sum_of_repeating_decimals : 
  SingleDigitRepeatingDecimal 5 + TwoDigitRepeatingDecimal 7 = 62 / 99 := by
  sorry

end sum_of_repeating_decimals_l2286_228627


namespace steve_final_marbles_l2286_228618

theorem steve_final_marbles (sam_initial steve_initial sally_initial sam_final : ℕ) :
  sam_initial = 2 * steve_initial →
  sally_initial = sam_initial - 5 →
  sam_final = sam_initial - 6 →
  sam_final = 8 →
  steve_initial + 3 = 10 :=
by
  sorry

end steve_final_marbles_l2286_228618


namespace product_of_place_values_l2286_228665

/-- The place value of a digit in a decimal number -/
def placeValue (digit : ℕ) (position : ℤ) : ℚ :=
  (digit : ℚ) * (10 : ℚ) ^ position

/-- The numeral under consideration -/
def numeral : ℚ := 7804830.88

/-- The product of place values of the three 8's in the numeral -/
def productOfPlaceValues : ℚ :=
  placeValue 8 5 * placeValue 8 1 * placeValue 8 (-2)

theorem product_of_place_values :
  productOfPlaceValues = 5120000 := by sorry

end product_of_place_values_l2286_228665


namespace traveler_distance_l2286_228687

theorem traveler_distance (north south west east : ℝ) : 
  north = 25 → 
  south = 10 → 
  west = 15 → 
  east = 7 → 
  let net_north := north - south
  let net_west := west - east
  Real.sqrt (net_north ^ 2 + net_west ^ 2) = 17 :=
by sorry

end traveler_distance_l2286_228687


namespace choir_members_l2286_228671

theorem choir_members (n : ℕ) : 
  50 ≤ n ∧ n ≤ 150 ∧ 
  n % 6 = 4 ∧ 
  n % 10 = 4 → 
  n = 64 ∨ n = 94 ∨ n = 124 := by
sorry

end choir_members_l2286_228671


namespace reservoir_water_amount_l2286_228642

theorem reservoir_water_amount 
  (total_capacity : ℝ) 
  (end_amount : ℝ) 
  (normal_level : ℝ) 
  (h1 : end_amount = 2 * normal_level)
  (h2 : end_amount = 0.75 * total_capacity)
  (h3 : normal_level = total_capacity - 20) :
  end_amount = 24 := by
  sorry

end reservoir_water_amount_l2286_228642


namespace marble_distribution_correct_group_size_l2286_228628

/-- The number of marbles in the jar -/
def total_marbles : ℕ := 500

/-- The number of additional people that would join the group -/
def additional_people : ℕ := 5

/-- The number of marbles each person would receive less if additional people joined -/
def marbles_less : ℕ := 2

/-- The number of people in the group today -/
def group_size : ℕ := 33

theorem marble_distribution :
  (total_marbles = group_size * (total_marbles / group_size)) ∧
  (total_marbles = (group_size + additional_people) * (total_marbles / group_size - marbles_less)) :=
by sorry

theorem correct_group_size : group_size = 33 :=
by sorry

end marble_distribution_correct_group_size_l2286_228628


namespace digit_55_is_2_l2286_228622

/-- The decimal representation of 1/17 as a list of digits -/
def decimal_rep_1_17 : List Nat := [0, 5, 8, 8, 2, 3, 5, 2, 9, 4, 1, 1, 7, 6, 4, 7]

/-- The length of the repeating sequence in the decimal representation of 1/17 -/
def repeat_length : Nat := 16

/-- The 55th digit after the decimal point in the decimal representation of 1/17 -/
def digit_55 : Nat := decimal_rep_1_17[(55 - 1) % repeat_length]

theorem digit_55_is_2 : digit_55 = 2 := by
  sorry

end digit_55_is_2_l2286_228622


namespace sphere_surface_area_l2286_228604

-- Define the sphere and its properties
def sphere_radius : ℝ := 13
def water_cross_section_radius : ℝ := 12
def submerged_depth : ℝ := 8

-- Theorem statement
theorem sphere_surface_area :
  (sphere_radius ^ 2 = water_cross_section_radius ^ 2 + (sphere_radius - submerged_depth) ^ 2) →
  (4 * π * sphere_radius ^ 2 = 676 * π) :=
by
  sorry

end sphere_surface_area_l2286_228604


namespace minimum_value_of_function_l2286_228644

theorem minimum_value_of_function (x : ℝ) (h : x > 0) : 
  (∀ y : ℝ, y > 0 → 4 / y + y ≥ 4) ∧ (∃ z : ℝ, z > 0 ∧ 4 / z + z = 4) := by
  sorry

end minimum_value_of_function_l2286_228644


namespace difference_of_squares_special_case_l2286_228612

theorem difference_of_squares_special_case : (723 : ℤ) * 723 - 722 * 724 = 1 := by
  sorry

end difference_of_squares_special_case_l2286_228612


namespace num_aplus_needed_is_two_l2286_228606

/-- Represents the grading system and reward calculation for Paul's courses. -/
structure GradingSystem where
  numCourses : Nat
  bPlusReward : Nat
  aReward : Nat
  aPlusReward : Nat
  maxReward : Nat

/-- Calculates the number of A+ grades needed to double the previous rewards. -/
def numAPlusNeeded (gs : GradingSystem) : Nat :=
  sorry

/-- Theorem stating that the number of A+ grades needed is 2. -/
theorem num_aplus_needed_is_two (gs : GradingSystem) 
  (h1 : gs.numCourses = 10)
  (h2 : gs.bPlusReward = 5)
  (h3 : gs.aReward = 10)
  (h4 : gs.aPlusReward = 15)
  (h5 : gs.maxReward = 190) :
  numAPlusNeeded gs = 2 := by
  sorry

end num_aplus_needed_is_two_l2286_228606


namespace smallest_angle_satisfies_equation_l2286_228634

/-- The smallest positive angle x in degrees that satisfies the given equation -/
def smallest_angle : ℝ := 11.25

theorem smallest_angle_satisfies_equation :
  let x := smallest_angle * Real.pi / 180
  Real.tan (6 * x) = (Real.cos (2 * x) - Real.sin (2 * x)) / (Real.cos (2 * x) + Real.sin (2 * x)) ∧
  ∀ y : ℝ, 0 < y ∧ y < smallest_angle →
    Real.tan (6 * y * Real.pi / 180) ≠ (Real.cos (2 * y * Real.pi / 180) - Real.sin (2 * y * Real.pi / 180)) /
                                       (Real.cos (2 * y * Real.pi / 180) + Real.sin (2 * y * Real.pi / 180)) :=
by sorry

end smallest_angle_satisfies_equation_l2286_228634


namespace largest_two_digit_divisor_l2286_228654

def a : ℕ := 2^5 * 3^3 * 5^2 * 7

theorem largest_two_digit_divisor :
  (∀ d : ℕ, d > 96 → d < 100 → ¬(d ∣ a)) ∧ (96 ∣ a) := by sorry

end largest_two_digit_divisor_l2286_228654


namespace rosencrans_wins_iff_odd_l2286_228653

/-- Represents a chord-drawing game on a circle with n points. -/
structure ChordGame where
  n : ℕ
  h : n ≥ 5

/-- Represents the outcome of the game. -/
inductive Outcome
  | RosencransWins
  | GildensternWins

/-- Determines the winner of the chord game based on the number of points. -/
def ChordGame.winner (game : ChordGame) : Outcome :=
  if game.n % 2 = 1 then Outcome.RosencransWins else Outcome.GildensternWins

/-- Theorem stating that Rosencrans wins if and only if n is odd. -/
theorem rosencrans_wins_iff_odd (game : ChordGame) :
  game.winner = Outcome.RosencransWins ↔ Odd game.n :=
sorry

end rosencrans_wins_iff_odd_l2286_228653


namespace car_down_payment_calculation_l2286_228624

/-- Calculates the down payment for a car purchase given the specified conditions. -/
theorem car_down_payment_calculation
  (car_cost : ℚ)
  (loan_term : ℕ)
  (monthly_payment : ℚ)
  (interest_rate : ℚ)
  (h_car_cost : car_cost = 32000)
  (h_loan_term : loan_term = 48)
  (h_monthly_payment : monthly_payment = 525)
  (h_interest_rate : interest_rate = 5 / 100)
  : ∃ (down_payment : ℚ),
    down_payment = car_cost - (loan_term * monthly_payment + loan_term * (interest_rate * monthly_payment)) ∧
    down_payment = 5540 :=
by sorry

end car_down_payment_calculation_l2286_228624


namespace largest_multiple_of_15_under_500_l2286_228613

theorem largest_multiple_of_15_under_500 : 
  ∀ n : ℕ, n > 0 ∧ 15 ∣ n ∧ n < 500 → n ≤ 495 := by
  sorry

end largest_multiple_of_15_under_500_l2286_228613


namespace circle1_properties_circle2_properties_l2286_228608

-- Define the equations of the circles
def circle1_equation (x y : ℝ) : Prop := (x + 1)^2 + y^2 = 20
def circle2_equation (x y : ℝ) : Prop := (x - 1)^2 + (y + 2)^2 = 2

-- Define the lines
def line_y0 (x y : ℝ) : Prop := y = 0
def line_2x_y0 (x y : ℝ) : Prop := 2*x + y = 0
def line_tangent (x y : ℝ) : Prop := x + y - 1 = 0

-- Define the points
def point_A : ℝ × ℝ := (1, 4)
def point_B : ℝ × ℝ := (3, 2)
def point_M : ℝ × ℝ := (2, -1)

-- Theorem for the first circle
theorem circle1_properties :
  ∃ (center_x : ℝ),
    (∀ (y : ℝ), circle1_equation center_x y → line_y0 center_x y) ∧
    circle1_equation point_A.1 point_A.2 ∧
    circle1_equation point_B.1 point_B.2 :=
sorry

-- Theorem for the second circle
theorem circle2_properties :
  ∃ (center_x center_y : ℝ),
    line_2x_y0 center_x center_y ∧
    (∀ (x y : ℝ), circle2_equation x y → 
      (x = point_M.1 ∧ y = point_M.2) → line_tangent x y) :=
sorry

end circle1_properties_circle2_properties_l2286_228608


namespace product_of_sums_of_squares_l2286_228633

theorem product_of_sums_of_squares (a b c d : ℤ) :
  ∃ x y : ℤ, (a^2 + b^2) * (c^2 + d^2) = x^2 + y^2 := by
  sorry

end product_of_sums_of_squares_l2286_228633


namespace no_real_roots_l2286_228649

theorem no_real_roots : ¬∃ (x : ℝ), x + Real.sqrt (2*x - 6) = 5 := by
  sorry

end no_real_roots_l2286_228649


namespace max_polygons_bound_l2286_228638

/-- The number of points marked on the circle. -/
def num_points : ℕ := 12

/-- The minimum allowed internal angle at the circle's center (in degrees). -/
def min_angle : ℝ := 30

/-- A function that calculates the maximum number of distinct convex polygons
    that can be formed under the given conditions. -/
def max_polygons (n : ℕ) (θ : ℝ) : ℕ :=
  2^n - (n.choose 0 + n.choose 1 + n.choose 2)

/-- Theorem stating that the maximum number of distinct convex polygons
    satisfying the conditions is less than or equal to 4017. -/
theorem max_polygons_bound :
  max_polygons num_points min_angle ≤ 4017 :=
sorry

end max_polygons_bound_l2286_228638


namespace chairs_bought_l2286_228607

theorem chairs_bought (chair_cost : ℕ) (total_spent : ℕ) (num_chairs : ℕ) : 
  chair_cost = 15 → total_spent = 180 → num_chairs * chair_cost = total_spent → num_chairs = 12 := by
  sorry

end chairs_bought_l2286_228607


namespace arithmetic_sum_l2286_228609

theorem arithmetic_sum : 4 * 7 + 5 * 12 + 6 * 4 + 7 * 5 = 147 := by
  sorry

end arithmetic_sum_l2286_228609


namespace max_value_of_f_l2286_228670

-- Define the function
def f (x : ℝ) : ℝ := -4 * x^2 + 12 * x + 1

-- State the theorem
theorem max_value_of_f :
  ∃ (m : ℝ), (∀ (x : ℝ), f x ≤ m) ∧ (∃ (x : ℝ), f x = m) ∧ m = 10 := by
  sorry

end max_value_of_f_l2286_228670


namespace f_properties_l2286_228632

noncomputable section

def f (a : ℝ) (x : ℝ) : ℝ := x^2 - a*x - a^2 * Real.log x

theorem f_properties (a : ℝ) (h : a ≠ 0) :
  (∀ x > 0, f a x = x^2 - a*x - a^2 * Real.log x) ∧
  (a = 1 → ∃ x > 0, ∀ y > 0, f a x ≤ f a y) ∧
  (a = 1 → ∃ x > 0, f a x = 0) ∧
  ((-2 ≤ a ∧ a ≤ 1) → ∀ x y, 1 < x ∧ x < y → f a x < f a y) ∧
  (a > 1 → ∀ x y, 1 < x ∧ x < y ∧ y < a → f a x > f a y) ∧
  (a > 1 → ∀ x y, a < x ∧ x < y → f a x < f a y) ∧
  (a < -2 → ∀ x y, 1 < x ∧ x < y ∧ y < -a/2 → f a x > f a y) ∧
  (a < -2 → ∀ x y, -a/2 < x ∧ x < y → f a x < f a y) :=
by sorry

end

end f_properties_l2286_228632


namespace find_x_l2286_228695

theorem find_x : ∃ X : ℤ, X - (5 - (6 + 2 * (7 - 8 - 5))) = 89 ∧ X = 100 := by
  sorry

end find_x_l2286_228695


namespace winning_sequence_exists_l2286_228651

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m > 1 → m < n → ¬(m ∣ n)

def first_digit (n : ℕ) : ℕ :=
  if n < 10 then n else first_digit (n / 10)

def last_digit (n : ℕ) : ℕ := n % 10

def valid_sequence (seq : List ℕ) : Prop :=
  seq.length > 0 ∧
  ∀ n ∈ seq, is_prime n ∧ n ≤ 100 ∧
  ∀ i < seq.length - 1, last_digit (seq.get ⟨i, by sorry⟩) = first_digit (seq.get ⟨i+1, by sorry⟩) ∧
  ∀ i j, i ≠ j → seq.get ⟨i, by sorry⟩ ≠ seq.get ⟨j, by sorry⟩

theorem winning_sequence_exists :
  ∃ seq : List ℕ, valid_sequence seq ∧ seq.length = 3 ∧
  ∀ p : ℕ, is_prime p → p ≤ 100 → p ∉ seq →
    (seq.length > 0 → first_digit p ≠ last_digit (seq.getLast (by sorry))) :=
sorry

end winning_sequence_exists_l2286_228651


namespace pencil_buyers_difference_l2286_228626

theorem pencil_buyers_difference : ∀ (pencil_cost : ℕ) 
  (eighth_graders fifth_graders : ℕ),
  pencil_cost > 0 ∧
  pencil_cost * eighth_graders = 234 ∧
  pencil_cost * fifth_graders = 285 ∧
  fifth_graders ≤ 25 →
  fifth_graders - eighth_graders = 17 := by
sorry

end pencil_buyers_difference_l2286_228626


namespace tigger_climbing_speed_ratio_l2286_228696

theorem tigger_climbing_speed_ratio :
  ∀ (T t : ℝ),
  T > 0 ∧ t > 0 →
  2 * T = t / 3 →
  T + t = 2 * T + t / 3 →
  T / t = 3 / 2 :=
by sorry

end tigger_climbing_speed_ratio_l2286_228696


namespace flooring_cost_is_14375_l2286_228675

/-- Represents the dimensions and cost of a rectangular room -/
structure RectRoom where
  length : Float
  width : Float
  cost_per_sqm : Float

/-- Represents the dimensions and cost of an L-shaped room -/
structure LShapeRoom where
  rect1_length : Float
  rect1_width : Float
  rect2_length : Float
  rect2_width : Float
  cost_per_sqm : Float

/-- Represents the dimensions and cost of a triangular room -/
structure TriRoom where
  base : Float
  height : Float
  cost_per_sqm : Float

/-- Calculates the total cost of flooring for all rooms -/
def total_flooring_cost (room1 : RectRoom) (room2 : LShapeRoom) (room3 : TriRoom) : Float :=
  (room1.length * room1.width * room1.cost_per_sqm) +
  ((room2.rect1_length * room2.rect1_width + room2.rect2_length * room2.rect2_width) * room2.cost_per_sqm) +
  (0.5 * room3.base * room3.height * room3.cost_per_sqm)

/-- Theorem stating that the total flooring cost for the given rooms is $14,375 -/
theorem flooring_cost_is_14375 
  (room1 : RectRoom)
  (room2 : LShapeRoom)
  (room3 : TriRoom)
  (h1 : room1 = { length := 5.5, width := 3.75, cost_per_sqm := 400 })
  (h2 : room2 = { rect1_length := 4, rect1_width := 2.5, rect2_length := 2, rect2_width := 1.5, cost_per_sqm := 350 })
  (h3 : room3 = { base := 3.5, height := 2, cost_per_sqm := 450 }) :
  total_flooring_cost room1 room2 room3 = 14375 := by
  sorry

end flooring_cost_is_14375_l2286_228675


namespace grade2_sample_count_l2286_228629

/-- Represents the number of students in a grade -/
def GradeCount := ℕ

/-- Represents a ratio of students across three grades -/
structure GradeRatio :=
  (grade1 : ℕ)
  (grade2 : ℕ)
  (grade3 : ℕ)

/-- Calculates the number of students in a stratified sample for a specific grade -/
def stratifiedSampleCount (totalSample : ℕ) (ratio : GradeRatio) (gradeRatio : ℕ) : ℕ :=
  (totalSample * gradeRatio) / (ratio.grade1 + ratio.grade2 + ratio.grade3)

/-- Theorem stating the number of Grade 2 students in the stratified sample -/
theorem grade2_sample_count 
  (totalSample : ℕ) 
  (ratio : GradeRatio) 
  (h1 : totalSample = 240) 
  (h2 : ratio = GradeRatio.mk 5 4 3) : 
  stratifiedSampleCount totalSample ratio ratio.grade2 = 80 := by
  sorry

end grade2_sample_count_l2286_228629


namespace circular_seating_theorem_l2286_228673

/-- The number of people seated at a circular table. -/
def n : ℕ := sorry

/-- The distance between two positions in a circular arrangement. -/
def circularDistance (a b : ℕ) : ℕ :=
  min ((a - b + n) % n) ((b - a + n) % n)

/-- The theorem stating that if the distance from 31 to 7 equals the distance from 31 to 14
    in a circular arrangement of n people, then n must be 41. -/
theorem circular_seating_theorem :
  circularDistance 31 7 = circularDistance 31 14 → n = 41 := by
  sorry

end circular_seating_theorem_l2286_228673


namespace finite_good_not_divisible_by_k_l2286_228666

/-- The number of divisors of an integer n -/
def τ (n : ℕ) : ℕ := sorry

/-- An integer n is "good" if for all m < n, we have τ(m) < τ(n) -/
def is_good (n : ℕ) : Prop :=
  ∀ m < n, τ m < τ n

/-- The set of good integers not divisible by k is finite -/
theorem finite_good_not_divisible_by_k (k : ℕ) (h : k ≥ 1) :
  {n : ℕ | is_good n ∧ ¬k ∣ n}.Finite :=
sorry

end finite_good_not_divisible_by_k_l2286_228666


namespace container_fullness_l2286_228658

def container_capacity : ℝ := 120
def initial_fullness : ℝ := 0.35
def added_water : ℝ := 48

theorem container_fullness :
  let initial_water := initial_fullness * container_capacity
  let total_water := initial_water + added_water
  let final_fullness := total_water / container_capacity
  final_fullness = 0.75 := by sorry

end container_fullness_l2286_228658


namespace randy_initial_money_l2286_228636

/-- Randy's initial amount of money -/
def initial_money : ℕ := sorry

/-- Amount Smith gave to Randy -/
def smith_gave : ℕ := 200

/-- Amount Randy gave to Sally -/
def sally_received : ℕ := 1200

/-- Amount Randy kept after giving money to Sally -/
def randy_kept : ℕ := 2000

theorem randy_initial_money :
  initial_money + smith_gave - sally_received = randy_kept ∧
  initial_money = 3000 := by sorry

end randy_initial_money_l2286_228636


namespace inequality_solution_l2286_228646

-- Define the function f
variable (f : ℝ → ℝ)

-- Define the conditions
variable (h1 : ∀ x y, x > 2 → y > 2 → x < y → f x > f y)
variable (h2 : ∀ x, f (x + 2) = f (-x + 2))

-- Define the solution set
def solution_set (x : ℝ) := 4/3 < x ∧ x < 2

-- State the theorem
theorem inequality_solution :
  (∀ x, solution_set x ↔ f (2*x - 1) - f (x + 1) > 0) :=
sorry

end inequality_solution_l2286_228646


namespace quadratic_equation_coefficients_l2286_228659

theorem quadratic_equation_coefficients :
  let original_eq : ℝ → Prop := λ x => 3 * x^2 - 1 = 6 * x
  let general_form : ℝ → ℝ → ℝ → ℝ → Prop := λ a b c x => a * x^2 + b * x + c = 0
  ∃ (a b c : ℝ), (∀ x, original_eq x ↔ general_form a b c x) ∧ a = 3 ∧ b = -6 :=
by sorry

end quadratic_equation_coefficients_l2286_228659


namespace problem_solution_l2286_228656

noncomputable section

def f (x : ℝ) : ℝ := Real.log x + 1 / x
def g (x : ℝ) : ℝ := x - Real.log x

theorem problem_solution :
  (∃ (a : ℝ), ∀ (x : ℝ), x > 0 → f x ≥ a ∧ 
    ∀ (b : ℝ), (∀ (y : ℝ), y > 0 → f y ≥ b) → b ≤ a) ∧
  (∀ (x : ℝ), x > 1 → f x < g x) ∧
  (∀ (x₁ x₂ : ℝ), x₁ > x₂ ∧ x₂ > 0 ∧ g x₁ = g x₂ → x₁ * x₂ < 1) :=
by sorry

end

end problem_solution_l2286_228656


namespace gcd_of_B_is_two_l2286_228676

def B : Set ℕ := {x | ∃ n : ℕ, x = 4*n + 6 ∧ n > 0}

theorem gcd_of_B_is_two : 
  ∃ d : ℕ, d > 1 ∧ (∀ x ∈ B, d ∣ x) ∧ 
  (∀ k : ℕ, k > 1 → (∀ x ∈ B, k ∣ x) → k ≤ d) ∧ d = 2 := by
  sorry

end gcd_of_B_is_two_l2286_228676


namespace range_of_a_l2286_228602

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := |2*x - 5*a| + |2*x + 1|
def g (x : ℝ) : ℝ := |x - 1| + 3

theorem range_of_a (a : ℝ) :
  (∀ x₁ : ℝ, ∃ x₂ : ℝ, f a x₁ = g x₂) →
  a ≥ 0.4 ∨ a ≤ -0.8 :=
by sorry

end range_of_a_l2286_228602


namespace scout_troop_profit_is_480_l2286_228621

/-- Calculates the profit of a scout troop selling candy bars -/
def scout_troop_profit (total_bars : ℕ) (cost_per_six : ℚ) (discount_rate : ℚ)
  (price_first_tier : ℚ) (price_second_tier : ℚ) (first_tier_limit : ℕ) : ℚ :=
  let cost_per_bar := cost_per_six / 6
  let total_cost := total_bars * cost_per_bar
  let discounted_cost := total_cost * (1 - discount_rate)
  let revenue_first_tier := min first_tier_limit total_bars * price_first_tier
  let revenue_second_tier := max 0 (total_bars - first_tier_limit) * price_second_tier
  let total_revenue := revenue_first_tier + revenue_second_tier
  total_revenue - discounted_cost

theorem scout_troop_profit_is_480 :
  scout_troop_profit 1200 3 (5/100) 1 (3/4) 600 = 480 := by
  sorry

end scout_troop_profit_is_480_l2286_228621


namespace smallest_satisfying_number_l2286_228660

/-- Returns the number of digits in a positive integer -/
def num_digits (n : ℕ) : ℕ :=
  if n < 10 then 1 else 1 + num_digits (n / 10)

/-- Returns the leftmost digit of a positive integer -/
def leftmost_digit (n : ℕ) : ℕ :=
  n / (10 ^ (num_digits n - 1))

/-- Returns the number after removing the leftmost digit -/
def remove_leftmost_digit (n : ℕ) : ℕ :=
  n % (10 ^ (num_digits n - 1))

/-- Checks if a number satisfies the condition -/
def satisfies_condition (n : ℕ) : Prop :=
  remove_leftmost_digit n = n / 19

theorem smallest_satisfying_number :
  ∀ n : ℕ, n > 0 → n < 1350 → ¬(satisfies_condition n) ∧ satisfies_condition 1350 :=
sorry

end smallest_satisfying_number_l2286_228660


namespace xy_value_l2286_228615

theorem xy_value (x y : ℝ) 
  (h1 : (16 : ℝ)^x / (4 : ℝ)^(x + y) = 16)
  (h2 : (25 : ℝ)^(x + y) / (5 : ℝ)^(6 * y) = 625) :
  x * y = 0 := by
  sorry

end xy_value_l2286_228615


namespace sin_150_degrees_l2286_228614

theorem sin_150_degrees : Real.sin (150 * π / 180) = 1 / 2 := by
  sorry

end sin_150_degrees_l2286_228614


namespace fraction_equality_l2286_228617

theorem fraction_equality (a b : ℝ) (h : a / b = 2 / 3) : a / (a + b) = 2 / 5 := by
  sorry

end fraction_equality_l2286_228617


namespace v_2002_equals_4_l2286_228603

-- Define the function g
def g : ℕ → ℕ
| 1 => 2
| 2 => 3
| 3 => 5
| 4 => 1
| 5 => 4
| _ => 0  -- For completeness, though not used in the problem

-- Define the sequence v
def v : ℕ → ℕ
| 0 => 3
| (n + 1) => g (v n)

-- Theorem to prove
theorem v_2002_equals_4 : v 2002 = 4 := by
  sorry

end v_2002_equals_4_l2286_228603


namespace intersection_A_complement_B_E_subset_B_implies_a_geq_neg_one_l2286_228637

-- Define the sets A and B
def A : Set ℝ := {x | (x + 3) * (x - 6) ≥ 0}
def B : Set ℝ := {x | (x + 2) / (x - 14) < 0}

-- Define the set E
def E (a : ℝ) : Set ℝ := {x | 2 * a < x ∧ x < a + 1}

-- Statement for the first part of the problem
theorem intersection_A_complement_B :
  A ∩ (Set.univ \ B) = {x | x ≤ -3 ∨ x ≥ 14} := by sorry

-- Statement for the second part of the problem
theorem E_subset_B_implies_a_geq_neg_one (a : ℝ) :
  E a ⊆ B → a ≥ -1 := by sorry

end intersection_A_complement_B_E_subset_B_implies_a_geq_neg_one_l2286_228637


namespace square_sum_equals_sixteen_l2286_228601

theorem square_sum_equals_sixteen (a b : ℝ) (h : a + b = 4) : a^2 + 2*a*b + b^2 = 16 := by
  sorry

end square_sum_equals_sixteen_l2286_228601


namespace expected_worth_is_one_third_l2286_228697

/-- Represents the outcome of a coin flip -/
inductive CoinFlip
| Heads
| Tails

/-- Represents the probability of a coin flip outcome -/
def probability : CoinFlip → ℚ
| CoinFlip.Heads => 2/3
| CoinFlip.Tails => 1/3

/-- Represents the monetary outcome of a coin flip -/
def monetaryOutcome : CoinFlip → ℤ
| CoinFlip.Heads => 5
| CoinFlip.Tails => -9

/-- The expected worth of a coin flip -/
def expectedWorth : ℚ :=
  (probability CoinFlip.Heads * monetaryOutcome CoinFlip.Heads) +
  (probability CoinFlip.Tails * monetaryOutcome CoinFlip.Tails)

theorem expected_worth_is_one_third :
  expectedWorth = 1/3 := by
  sorry

end expected_worth_is_one_third_l2286_228697


namespace jeans_business_weekly_hours_l2286_228605

/-- Represents the operating hours of a business for a single day -/
structure DailyHours where
  open_time : Nat
  close_time : Nat

/-- Calculates the number of hours a business is open in a day -/
def hours_open (dh : DailyHours) : Nat :=
  dh.close_time - dh.open_time

/-- Represents the operating hours of a business for a week -/
structure WeeklyHours where
  weekday : DailyHours
  weekend : DailyHours

/-- Calculates the total hours a business is open in a week -/
def total_weekly_hours (wh : WeeklyHours) : Nat :=
  (hours_open wh.weekday * 5) + (hours_open wh.weekend * 2)

/-- Jean's business hours -/
def jeans_business : WeeklyHours :=
  { weekday := { open_time := 16, close_time := 22 }
    weekend := { open_time := 18, close_time := 22 } }

theorem jeans_business_weekly_hours :
  total_weekly_hours jeans_business = 38 := by
  sorry

end jeans_business_weekly_hours_l2286_228605


namespace problem_solution_l2286_228685

theorem problem_solution : ∃ x : ℝ, 70 + 5 * 12 / (x / 3) = 71 ∧ x = 180 := by
  sorry

end problem_solution_l2286_228685


namespace min_sum_reciprocals_roots_l2286_228669

theorem min_sum_reciprocals_roots (k : ℝ) (x₁ x₂ : ℝ) : 
  x₁ > 0 ∧ x₂ > 0 ∧ 
  x₁^2 - k*x₁ + k + 3 = 0 ∧ 
  x₂^2 - k*x₂ + k + 3 = 0 ∧ 
  x₁ ≠ x₂ →
  (∃ (s : ℝ), s = 1/x₁ + 1/x₂ ∧ s ≥ 2/3 ∧ ∀ (t : ℝ), t = 1/x₁ + 1/x₂ → t ≥ 2/3) :=
sorry

end min_sum_reciprocals_roots_l2286_228669


namespace largest_angle_convex_pentagon_l2286_228630

theorem largest_angle_convex_pentagon (x : ℝ) : 
  (x + 2) + (2*x + 3) + (3*x - 5) + (4*x + 1) + (5*x - 1) = 540 →
  max (x + 2) (max (2*x + 3) (max (3*x - 5) (max (4*x + 1) (5*x - 1)))) = 179 := by
  sorry

end largest_angle_convex_pentagon_l2286_228630
