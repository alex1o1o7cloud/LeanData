import Mathlib

namespace gym_income_is_10800_l1013_101366

/-- A gym charges its members twice a month and has a fixed number of members. -/
structure Gym where
  charge_per_half_month : ℕ
  charges_per_month : ℕ
  num_members : ℕ

/-- Calculate the monthly income of the gym -/
def monthly_income (g : Gym) : ℕ :=
  g.charge_per_half_month * g.charges_per_month * g.num_members

/-- Theorem stating that the gym's monthly income is $10,800 -/
theorem gym_income_is_10800 (g : Gym) 
  (h1 : g.charge_per_half_month = 18)
  (h2 : g.charges_per_month = 2)
  (h3 : g.num_members = 300) : 
  monthly_income g = 10800 := by
  sorry

end gym_income_is_10800_l1013_101366


namespace intersection_of_A_and_B_l1013_101303

def A : Set ℝ := {x | -2 ≤ x ∧ x ≤ 3}
def B : Set ℝ := {y | ∃ x, y = x^2 + 2}

theorem intersection_of_A_and_B : A ∩ B = {x | 2 ≤ x ∧ x ≤ 3} := by sorry

end intersection_of_A_and_B_l1013_101303


namespace simplify_expression_l1013_101363

theorem simplify_expression (x : ℝ) : (4*x)^4 + (5*x)*(x^3) = 261*x^4 := by
  sorry

end simplify_expression_l1013_101363


namespace triangle_side_inequality_l1013_101332

theorem triangle_side_inequality (a b c : ℝ) (h : 0 < a ∧ 0 < b ∧ 0 < c ∧ a + b > c ∧ b + c > a ∧ c + a > b) :
  (a^2 + c^2) / b^2 ≥ (1 : ℝ) / 2 ∧ ∃ a' b' c' : ℝ, 0 < a' ∧ 0 < b' ∧ 0 < c' ∧ a' + b' > c' ∧ b' + c' > a' ∧ c' + a' > b' ∧ (a'^2 + c'^2) / b'^2 = (1 : ℝ) / 2 :=
sorry

end triangle_side_inequality_l1013_101332


namespace neglart_students_count_l1013_101317

/-- Represents the number of toes on a Hoopit's hand -/
def hoopit_toes_per_hand : ℕ := 3

/-- Represents the number of hands a Hoopit has -/
def hoopit_hands : ℕ := 4

/-- Represents the number of toes on a Neglart's hand -/
def neglart_toes_per_hand : ℕ := 2

/-- Represents the number of hands a Neglart has -/
def neglart_hands : ℕ := 5

/-- Represents the number of Hoopit students on the bus -/
def hoopit_students : ℕ := 7

/-- Represents the total number of toes on the bus -/
def total_toes : ℕ := 164

/-- Theorem stating that the number of Neglart students on the bus is 8 -/
theorem neglart_students_count : ∃ (n : ℕ), 
  n * (neglart_toes_per_hand * neglart_hands) + 
  hoopit_students * (hoopit_toes_per_hand * hoopit_hands) = total_toes ∧ 
  n = 8 := by
  sorry

end neglart_students_count_l1013_101317


namespace line_x_intercept_l1013_101377

/-- A line passing through two points (-3, 3) and (2, 10) has x-intercept -36/7 -/
theorem line_x_intercept : 
  let p₁ : ℝ × ℝ := (-3, 3)
  let p₂ : ℝ × ℝ := (2, 10)
  let m : ℝ := (p₂.2 - p₁.2) / (p₂.1 - p₁.1)
  let b : ℝ := p₁.2 - m * p₁.1
  (0 - b) / m = -36/7 :=
by sorry

end line_x_intercept_l1013_101377


namespace base6_45_equals_decimal_29_l1013_101395

-- Define a function to convert a base-6 number to decimal
def base6ToDecimal (digits : List Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * (6 ^ i)) 0

-- Theorem statement
theorem base6_45_equals_decimal_29 :
  base6ToDecimal [5, 4] = 29 := by
  sorry

end base6_45_equals_decimal_29_l1013_101395


namespace max_value_AMC_l1013_101374

theorem max_value_AMC (A M C : ℕ) (h : A + M + C = 12) :
  A * M * C + A * M + M * C + C * A ≤ 112 :=
by sorry

end max_value_AMC_l1013_101374


namespace florist_roses_l1013_101320

theorem florist_roses (initial : ℕ) : 
  (initial - 3 + 34 = 36) → initial = 5 := by
  sorry

end florist_roses_l1013_101320


namespace parallel_vectors_m_value_l1013_101385

/-- Given two vectors a and b in ℝ³, where a = (-2, 1, 5) and b = (6, m, -15),
    if a and b are parallel, then m = -3. -/
theorem parallel_vectors_m_value (m : ℝ) :
  let a : Fin 3 → ℝ := ![(-2 : ℝ), 1, 5]
  let b : Fin 3 → ℝ := ![6, m, -15]
  (∃ (t : ℝ), b = fun i => t * a i) → m = -3 := by
  sorry

end parallel_vectors_m_value_l1013_101385


namespace basic_astrophysics_degrees_l1013_101387

theorem basic_astrophysics_degrees (total_degrees : ℝ) 
  (microphotonics home_electronics food_additives gm_microorganisms industrial_lubricants : ℝ) :
  total_degrees = 360 →
  microphotonics = 14 →
  home_electronics = 24 →
  food_additives = 10 →
  gm_microorganisms = 29 →
  industrial_lubricants = 8 →
  let other_sectors := microphotonics + home_electronics + food_additives + gm_microorganisms + industrial_lubricants
  let basic_astrophysics_percent := 100 - other_sectors
  let basic_astrophysics_degrees := (basic_astrophysics_percent / 100) * total_degrees
  basic_astrophysics_degrees = 54 :=
by sorry

end basic_astrophysics_degrees_l1013_101387


namespace smaller_root_comparison_l1013_101325

theorem smaller_root_comparison (a a' b b' : ℝ) (ha : a ≠ 0) (ha' : a' ≠ 0) :
  (-b / a < -b' / a') ↔ (b / a > b' / a') :=
sorry

end smaller_root_comparison_l1013_101325


namespace geometric_sequence_sum_l1013_101324

/-- A geometric sequence with positive terms and common ratio 2 -/
def GeometricSequence (a : ℕ → ℝ) : Prop :=
  (∀ n, a n > 0) ∧ (∀ n, a (n + 1) = 2 * a n)

theorem geometric_sequence_sum (a : ℕ → ℝ) :
  GeometricSequence a →
  a 1 + a 2 + a 3 = 21 →
  a 3 + a 4 + a 5 = 84 := by
  sorry

end geometric_sequence_sum_l1013_101324


namespace positive_A_value_l1013_101309

-- Define the # relation
def hash (A B : ℝ) : ℝ := A^2 - B^2

-- Theorem statement
theorem positive_A_value (A : ℝ) (h1 : hash A 7 = 72) (h2 : A > 0) : A = 11 := by
  sorry

end positive_A_value_l1013_101309


namespace tylers_meal_combinations_l1013_101354

theorem tylers_meal_combinations (meat_types : ℕ) (vegetable_types : ℕ) (dessert_types : ℕ) 
  (h1 : meat_types = 4)
  (h2 : vegetable_types = 5)
  (h3 : dessert_types = 5) :
  meat_types * (vegetable_types.choose 3) * (dessert_types.choose 2) = 400 := by
  sorry

end tylers_meal_combinations_l1013_101354


namespace cos_sum_seventh_roots_l1013_101352

theorem cos_sum_seventh_roots : 
  Real.cos (2 * π / 7) + Real.cos (4 * π / 7) + Real.cos (6 * π / 7) = -1/2 := by
  sorry

end cos_sum_seventh_roots_l1013_101352


namespace percentage_difference_l1013_101312

theorem percentage_difference (x y : ℝ) 
  (hx : 3 = 0.15 * x) 
  (hy : 3 = 0.25 * y) : 
  x - y = 8 := by
  sorry

end percentage_difference_l1013_101312


namespace old_bridge_traffic_l1013_101392

/-- Represents the number of vehicles passing through the old bridge every month -/
def old_bridge_monthly_traffic : ℕ := sorry

/-- Represents the number of vehicles passing through the new bridge every month -/
def new_bridge_monthly_traffic : ℕ := sorry

/-- The new bridge has twice the capacity of the old one -/
axiom new_bridge_capacity : new_bridge_monthly_traffic = 2 * old_bridge_monthly_traffic

/-- The number of vehicles passing through the new bridge increased by 60% compared to the old bridge -/
axiom traffic_increase : new_bridge_monthly_traffic = old_bridge_monthly_traffic + (60 * old_bridge_monthly_traffic) / 100

/-- The total number of vehicles passing through both bridges in a year is 62,400 -/
axiom total_yearly_traffic : 12 * (old_bridge_monthly_traffic + new_bridge_monthly_traffic) = 62400

theorem old_bridge_traffic : old_bridge_monthly_traffic = 2000 :=
sorry

end old_bridge_traffic_l1013_101392


namespace smallest_right_triangle_area_l1013_101310

theorem smallest_right_triangle_area :
  let side1 : ℝ := 6
  let side2 : ℝ := 8
  let area1 : ℝ := (1/2) * side1 * side2
  let area2 : ℝ := (1/2) * side1 * Real.sqrt (side2^2 - side1^2)
  min area1 area2 = 6 * Real.sqrt 7 := by sorry

end smallest_right_triangle_area_l1013_101310


namespace smallest_tree_height_l1013_101313

/-- Given three trees with specific height relationships, prove the height of the smallest tree -/
theorem smallest_tree_height (tallest middle smallest : ℝ) : 
  tallest = 108 →
  middle = tallest / 2 - 6 →
  smallest = middle / 4 →
  smallest = 12 := by sorry

end smallest_tree_height_l1013_101313


namespace math_homework_pages_l1013_101378

theorem math_homework_pages (total_pages reading_pages : ℕ) 
  (h1 : total_pages = 7)
  (h2 : reading_pages = 2) :
  total_pages - reading_pages = 5 := by
  sorry

end math_homework_pages_l1013_101378


namespace parallel_segment_ratio_sum_l1013_101361

/-- Represents a triangle with side lengths a, b, c -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  ha : a > 0
  hb : b > 0
  hc : c > 0

/-- Represents the parallel line segments drawn from a point P inside the triangle -/
structure ParallelSegments where
  a' : ℝ
  b' : ℝ
  c' : ℝ
  ha' : a' > 0
  hb' : b' > 0
  hc' : c' > 0

/-- Theorem: For any triangle and any point P inside it, the sum of ratios of 
    parallel segments to corresponding sides is always 1 -/
theorem parallel_segment_ratio_sum (t : Triangle) (p : ParallelSegments) :
  p.a' / t.a + p.b' / t.b + p.c' / t.c = 1 := by sorry

end parallel_segment_ratio_sum_l1013_101361


namespace value_preserving_interval_iff_m_in_M_l1013_101368

/-- A function f has a value-preserving interval [a,b] if it is monotonic on [a,b]
    and its range on [a,b] is [a,b] -/
def has_value_preserving_interval (f : ℝ → ℝ) : Prop :=
  ∃ a b : ℝ, a < b ∧
    (∀ x y, a ≤ x ∧ x < y ∧ y ≤ b → (f x < f y ∨ f y < f x)) ∧
    (∀ y, a ≤ y ∧ y ≤ b → ∃ x, a ≤ x ∧ x ≤ b ∧ f x = y)

/-- The set of m values for which f(x) = x^2 - (1/2)x + m has a value-preserving interval -/
def M : Set ℝ :=
  {m | m ∈ Set.Icc (5/16) (9/16) ∪ Set.Icc (-11/16) (-7/16)}

/-- The main theorem stating the equivalence between the existence of a value-preserving interval
    and m being in the set M -/
theorem value_preserving_interval_iff_m_in_M :
  ∀ m : ℝ, has_value_preserving_interval (fun x => x^2 - (1/2)*x + m) ↔ m ∈ M :=
sorry

end value_preserving_interval_iff_m_in_M_l1013_101368


namespace sum_of_squares_minus_fourth_power_l1013_101384

theorem sum_of_squares_minus_fourth_power (a b : ℕ+) : 
  a^2 - b^4 = 2009 → a + b = 47 := by
  sorry

end sum_of_squares_minus_fourth_power_l1013_101384


namespace min_value_in_region_l1013_101349

-- Define the region
def enclosed_region (x y : ℝ) : Prop :=
  abs x ≤ y ∧ y ≤ 2

-- Define the function to minimize
def f (x y : ℝ) : ℝ := 2 * x - y

-- Theorem statement
theorem min_value_in_region :
  ∃ (min : ℝ), min = -6 ∧
  ∀ (x y : ℝ), enclosed_region x y → f x y ≥ min :=
sorry

end min_value_in_region_l1013_101349


namespace specific_arrangement_eq_3456_l1013_101335

/-- The number of ways to arrange players from different teams in a row -/
def arrange_players (num_teams : ℕ) (team_sizes : List ℕ) : ℕ :=
  (Nat.factorial num_teams) * (team_sizes.map Nat.factorial).prod

/-- The specific arrangement for the given problem -/
def specific_arrangement : ℕ :=
  arrange_players 4 [3, 2, 3, 2]

/-- Theorem stating that the specific arrangement equals 3456 -/
theorem specific_arrangement_eq_3456 : specific_arrangement = 3456 := by
  sorry

end specific_arrangement_eq_3456_l1013_101335


namespace square_difference_of_product_and_sum_l1013_101338

theorem square_difference_of_product_and_sum (p q : ℝ) 
  (h1 : p * q = 16) (h2 : p + q = 10) : (p - q)^2 = 36 := by
  sorry

end square_difference_of_product_and_sum_l1013_101338


namespace wedding_catering_ratio_l1013_101355

/-- Represents the catering problem for Jenny's wedding --/
def CateringProblem (total_guests : ℕ) (steak_cost chicken_cost : ℚ) (total_budget : ℚ) : Prop :=
  ∃ (steak_guests chicken_guests : ℕ),
    steak_guests + chicken_guests = total_guests ∧
    steak_cost * steak_guests + chicken_cost * chicken_guests = total_budget ∧
    steak_guests = 3 * chicken_guests

/-- Theorem stating that the given conditions result in a 3:1 ratio of steak to chicken guests --/
theorem wedding_catering_ratio :
  CateringProblem 80 25 18 1860 :=
by
  sorry

end wedding_catering_ratio_l1013_101355


namespace stuffed_animal_sales_difference_stuffed_animal_sales_difference_proof_l1013_101357

theorem stuffed_animal_sales_difference : ℕ → ℕ → ℕ → Prop :=
  fun thor jake quincy =>
    (jake = thor + 10) →
    (quincy = thor * 10) →
    (quincy = 200) →
    (quincy - jake = 170)

-- The proof would go here, but we're skipping it as requested
theorem stuffed_animal_sales_difference_proof :
  ∃ (thor jake quincy : ℕ), stuffed_animal_sales_difference thor jake quincy :=
sorry

end stuffed_animal_sales_difference_stuffed_animal_sales_difference_proof_l1013_101357


namespace dragon_unicorn_equivalence_l1013_101393

theorem dragon_unicorn_equivalence (R U : Prop) :
  (R → U) ↔ ((¬U → ¬R) ∧ (¬R ∨ U)) :=
sorry

end dragon_unicorn_equivalence_l1013_101393


namespace smallest_number_of_eggs_l1013_101308

/-- The number of eggs in a full container -/
def full_container : ℕ := 15

/-- The number of containers with one missing egg -/
def partial_containers : ℕ := 3

/-- The minimum number of eggs specified in the problem -/
def min_eggs : ℕ := 150

/-- The number of eggs in the solution -/
def solution_eggs : ℕ := 162

/-- Theorem stating that the smallest number of eggs satisfying the conditions is 162 -/
theorem smallest_number_of_eggs :
  ∀ n : ℕ,
  (∃ c : ℕ, n = full_container * c - partial_containers) →
  n > min_eggs →
  n ≥ solution_eggs ∧
  (∀ m : ℕ, m < solution_eggs → 
    (∀ d : ℕ, m ≠ full_container * d - partial_containers) ∨ m ≤ min_eggs) :=
by sorry

end smallest_number_of_eggs_l1013_101308


namespace old_man_coins_l1013_101337

theorem old_man_coins (x y : ℕ) (h1 : x ≠ y) (h2 : x^2 - y^2 = 49 * (x - y)) : x + y = 49 := by
  sorry

end old_man_coins_l1013_101337


namespace quadruplet_babies_l1013_101358

theorem quadruplet_babies (total_babies : ℕ) 
  (h_total : total_babies = 1500)
  (h_triplets : ∃ (b c : ℕ), b = 3 * c)
  (h_twins : ∃ (a b : ℕ), a = 5 * b)
  (h_sum : ∃ (a b c : ℕ), 2 * a + 3 * b + 4 * c = total_babies) :
  ∃ (c : ℕ), 4 * c = 136 ∧ c * 4 ≤ total_babies := by
sorry

#eval 136

end quadruplet_babies_l1013_101358


namespace max_sum_of_counts_l1013_101388

/-- Represents a sign in the table -/
inductive Sign
| Plus
| Minus

/-- Represents the table -/
def Table := Fin 30 → Fin 30 → Option Sign

/-- Count of pluses in the table -/
def count_pluses (t : Table) : ℕ := sorry

/-- Count of minuses in the table -/
def count_minuses (t : Table) : ℕ := sorry

/-- Check if a row has at most 17 signs -/
def row_valid (t : Table) (row : Fin 30) : Prop := sorry

/-- Check if a column has at most 17 signs -/
def col_valid (t : Table) (col : Fin 30) : Prop := sorry

/-- Calculate the sum of counts -/
def sum_of_counts (t : Table) : ℕ := sorry

/-- Main theorem -/
theorem max_sum_of_counts :
  ∀ (t : Table),
    count_pluses t = 162 →
    count_minuses t = 144 →
    (∀ row, row_valid t row) →
    (∀ col, col_valid t col) →
    sum_of_counts t ≤ 2592 :=
sorry

end max_sum_of_counts_l1013_101388


namespace rectangular_field_width_l1013_101359

theorem rectangular_field_width (length width : ℝ) : 
  length = 24 ∧ length = 2 * width - 3 → width = 13.5 :=
by sorry

end rectangular_field_width_l1013_101359


namespace twelfth_number_with_digit_sum_12_l1013_101346

/-- A function that returns the sum of the digits of a positive integer -/
def digit_sum (n : ℕ+) : ℕ := sorry

/-- A function that returns the nth positive integer whose digits sum to 12 -/
def nth_number_with_digit_sum_12 (n : ℕ+) : ℕ+ := sorry

/-- Theorem stating that the 12th number with digit sum 12 is 165 -/
theorem twelfth_number_with_digit_sum_12 : 
  nth_number_with_digit_sum_12 12 = 165 := by sorry

end twelfth_number_with_digit_sum_12_l1013_101346


namespace statement_true_for_lines_statement_true_for_planes_statement_true_cases_l1013_101315

-- Define a type for geometric objects (lines or planes)
inductive GeometricObject
| Line
| Plane

-- Define a parallel relation
def parallel (a b : GeometricObject) : Prop := sorry

-- Define the statement we want to prove
def statement (x y z : GeometricObject) : Prop :=
  (parallel x z ∧ parallel y z) ∧ ¬(parallel x y)

-- Theorem for the case when all objects are lines
theorem statement_true_for_lines :
  ∃ (x y z : GeometricObject), 
    x = GeometricObject.Line ∧ 
    y = GeometricObject.Line ∧ 
    z = GeometricObject.Line ∧ 
    statement x y z := by sorry

-- Theorem for the case when all objects are planes
theorem statement_true_for_planes :
  ∃ (x y z : GeometricObject), 
    x = GeometricObject.Plane ∧ 
    y = GeometricObject.Plane ∧ 
    z = GeometricObject.Plane ∧ 
    statement x y z := by sorry

-- Main theorem combining both cases
theorem statement_true_cases :
  (∃ (x y z : GeometricObject), 
    x = GeometricObject.Line ∧ 
    y = GeometricObject.Line ∧ 
    z = GeometricObject.Line ∧ 
    statement x y z) ∧
  (∃ (x y z : GeometricObject), 
    x = GeometricObject.Plane ∧ 
    y = GeometricObject.Plane ∧ 
    z = GeometricObject.Plane ∧ 
    statement x y z) := by sorry

end statement_true_for_lines_statement_true_for_planes_statement_true_cases_l1013_101315


namespace sales_tax_percentage_l1013_101360

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

end sales_tax_percentage_l1013_101360


namespace product_maximized_at_11_l1013_101330

/-- Represents a geometric sequence with first term a₁ and common ratio q -/
structure GeometricSequence where
  a₁ : ℝ
  q : ℝ

/-- Calculates the nth term of a geometric sequence -/
def nthTerm (gs : GeometricSequence) (n : ℕ) : ℝ :=
  gs.a₁ * gs.q ^ (n - 1)

/-- Calculates the product of the first n terms of a geometric sequence -/
def productFirstNTerms (gs : GeometricSequence) (n : ℕ) : ℝ :=
  (gs.a₁ ^ n) * (gs.q ^ (n * (n - 1) / 2))

/-- Theorem: The product of the first n terms is maximized when n = 11 for the given sequence -/
theorem product_maximized_at_11 (gs : GeometricSequence) 
    (h1 : gs.a₁ = 1536) (h2 : gs.q = -1/2) :
    ∀ k : ℕ, k ≠ 11 → productFirstNTerms gs 11 ≥ productFirstNTerms gs k := by
  sorry

end product_maximized_at_11_l1013_101330


namespace valid_set_iff_ge_four_l1013_101336

/-- A set of positive integers satisfying the given conditions -/
def ValidSet (n : ℕ) (S : Finset ℕ) : Prop :=
  (S.card = n) ∧
  (∀ x ∈ S, x > 0 ∧ x < 2^(n-1)) ∧
  (∀ A B : Finset ℕ, A ⊆ S → B ⊆ S → A ≠ ∅ → B ≠ ∅ → A ≠ B →
    (A.sum id ≠ B.sum id))

/-- The main theorem stating the existence of a valid set if and only if n ≥ 4 -/
theorem valid_set_iff_ge_four (n : ℕ) :
  (∃ S : Finset ℕ, ValidSet n S) ↔ n ≥ 4 := by
  sorry

end valid_set_iff_ge_four_l1013_101336


namespace unique_lattice_point_l1013_101304

theorem unique_lattice_point : 
  ∃! (x y : ℤ), x^2 - y^2 = 75 ∧ x - y = 5 := by sorry

end unique_lattice_point_l1013_101304


namespace max_value_of_trig_function_l1013_101364

theorem max_value_of_trig_function :
  let f : ℝ → ℝ := λ x ↦ Real.sin (x / 2) + Real.sqrt 3 * Real.cos (x / 2)
  ∀ x : ℝ, f x ≤ 2 ∧ ∃ x₀ : ℝ, f x₀ = 2 := by sorry

end max_value_of_trig_function_l1013_101364


namespace f_is_even_and_increasing_l1013_101383

def f (x : ℝ) : ℝ := |x| + 1

theorem f_is_even_and_increasing : 
  (∀ x : ℝ, f x = f (-x)) ∧ 
  (∀ x y : ℝ, 0 < x → x < y → f x < f y) :=
by sorry

end f_is_even_and_increasing_l1013_101383


namespace no_positive_integer_solution_l1013_101307

theorem no_positive_integer_solution :
  ¬∃ (x y : ℕ+), x^5 = y^2 + 4 := by sorry

end no_positive_integer_solution_l1013_101307


namespace f_one_lower_bound_l1013_101375

/-- Given a quadratic function f(x) = 4x^2 - mx + 5 that is increasing on [-2, +∞),
    prove that f(1) ≥ 25. -/
theorem f_one_lower_bound
  (f : ℝ → ℝ)
  (m : ℝ)
  (h1 : ∀ x, f x = 4 * x^2 - m * x + 5)
  (h2 : ∀ x y, x ≥ -2 → y ≥ -2 → x < y → f x < f y) :
  f 1 ≥ 25 := by
sorry

end f_one_lower_bound_l1013_101375


namespace eight_round_game_probability_l1013_101339

/-- Represents the probability of a specific outcome in an 8-round game -/
def game_probability (p1 p2 p3 : ℝ) (n1 n2 n3 : ℕ) : ℝ :=
  (p1^n1 * p2^n2 * p3^n3) * (Nat.choose 8 n1 * Nat.choose (8 - n1) n2)

theorem eight_round_game_probability :
  let p1 := (1 : ℝ) / 2
  let p2 := (1 : ℝ) / 3
  let p3 := (1 : ℝ) / 6
  game_probability p1 p2 p3 4 3 1 = 35 / 324 := by
  sorry

end eight_round_game_probability_l1013_101339


namespace perpendicular_vectors_k_value_l1013_101302

theorem perpendicular_vectors_k_value (a b : ℝ × ℝ) (k : ℝ) :
  a = (2, 1) →
  b = (-2, k) →
  (a.1 * (2 * a.1 - b.1) + a.2 * (2 * a.2 - b.2) = 0) →
  k = 14 := by sorry

end perpendicular_vectors_k_value_l1013_101302


namespace second_pipe_fill_time_l1013_101350

/-- The time it takes for the first pipe to fill the cistern -/
def t1 : ℝ := 10

/-- The time it takes for the third pipe to empty the cistern -/
def t3 : ℝ := 25

/-- The time it takes to fill the cistern when all pipes are opened simultaneously -/
def t_all : ℝ := 6.976744186046512

/-- The time it takes for the second pipe to fill the cistern -/
def t2 : ℝ := 11.994

theorem second_pipe_fill_time :
  ∃ (t2 : ℝ), t2 > 0 ∧ (1 / t1 + 1 / t2 - 1 / t3 = 1 / t_all) :=
sorry

end second_pipe_fill_time_l1013_101350


namespace angle_trig_values_l1013_101394

def l₁ (x y : ℝ) : Prop := x - y = 0
def l₂ (x y : ℝ) : Prop := 2 * x + y - 3 = 0

def intersection_point (P : ℝ × ℝ) : Prop :=
  l₁ P.1 P.2 ∧ l₂ P.1 P.2

theorem angle_trig_values (α : ℝ) (P : ℝ × ℝ) :
  intersection_point P →
  Real.sin α = Real.sqrt 2 / 2 ∧
  Real.cos α = Real.sqrt 2 / 2 ∧
  Real.tan α = 1 :=
by sorry

end angle_trig_values_l1013_101394


namespace parabola_vertex_l1013_101365

/-- The equation of a parabola in the xy-plane. -/
def ParabolaEquation (x y : ℝ) : Prop :=
  y^2 - 4*y + x + 7 = 0

/-- The vertex of a parabola. -/
def Vertex : ℝ × ℝ := (-3, 2)

/-- Theorem stating that the vertex of the parabola defined by y^2 - 4y + x + 7 = 0 is (-3, 2). -/
theorem parabola_vertex :
  ∀ x y : ℝ, ParabolaEquation x y → (x, y) = Vertex :=
sorry

end parabola_vertex_l1013_101365


namespace trail_mix_nuts_l1013_101386

theorem trail_mix_nuts (walnuts almonds : ℚ) 
  (h1 : walnuts = 0.25)
  (h2 : almonds = 0.25) : 
  walnuts + almonds = 0.50 := by
sorry

end trail_mix_nuts_l1013_101386


namespace object_is_cylinder_l1013_101345

-- Define the possible shapes
inductive Shape
  | Rectangle
  | Cylinder
  | Cuboid
  | Cone

-- Define the types of views
inductive View
  | Rectangular
  | Circular

-- Define the object's properties
structure Object where
  frontView : View
  topView : View
  sideView : View

-- Theorem statement
theorem object_is_cylinder (obj : Object)
  (h1 : obj.frontView = View.Rectangular)
  (h2 : obj.sideView = View.Rectangular)
  (h3 : obj.topView = View.Circular) :
  Shape.Cylinder = 
    match obj.frontView, obj.topView, obj.sideView with
    | View.Rectangular, View.Circular, View.Rectangular => Shape.Cylinder
    | _, _, _ => Shape.Rectangle  -- default case, won't be reached
  := by sorry

end object_is_cylinder_l1013_101345


namespace described_relationship_is_correlation_l1013_101399

/-- Represents a variable in a statistical relationship -/
structure Variable where
  name : String
  is_independent : Bool

/-- Represents a relationship between two variables -/
structure Relationship where
  x : Variable
  y : Variable
  is_uncertain : Bool
  y_has_randomness : Bool

/-- Defines what a correlation is -/
def is_correlation (r : Relationship) : Prop :=
  r.x.is_independent ∧ 
  ¬r.y.is_independent ∧ 
  r.is_uncertain ∧ 
  r.y_has_randomness

/-- Theorem stating that the described relationship is a correlation -/
theorem described_relationship_is_correlation (x y : Variable) (r : Relationship) 
  (h1 : x.is_independent)
  (h2 : ¬y.is_independent)
  (h3 : r.x = x)
  (h4 : r.y = y)
  (h5 : r.is_uncertain)
  (h6 : r.y_has_randomness) :
  is_correlation r := by
  sorry


end described_relationship_is_correlation_l1013_101399


namespace books_checked_out_wednesday_l1013_101340

theorem books_checked_out_wednesday (initial_books : ℕ) (thursday_returned : ℕ) 
  (thursday_checked_out : ℕ) (friday_returned : ℕ) (final_books : ℕ) :
  initial_books = 98 →
  thursday_returned = 23 →
  thursday_checked_out = 5 →
  friday_returned = 7 →
  final_books = 80 →
  ∃ (wednesday_checked_out : ℕ),
    wednesday_checked_out = 43 ∧
    final_books = initial_books - wednesday_checked_out + thursday_returned - 
      thursday_checked_out + friday_returned :=
by
  sorry

end books_checked_out_wednesday_l1013_101340


namespace least_positive_integer_congruence_l1013_101353

theorem least_positive_integer_congruence :
  ∃ (x : ℕ), x > 0 ∧ (x + 7071 : ℤ) ≡ 3540 [ZMOD 15] ∧
  ∀ (y : ℕ), y > 0 → (y + 7071 : ℤ) ≡ 3540 [ZMOD 15] → x ≤ y ∧
  x = 9 :=
by sorry

end least_positive_integer_congruence_l1013_101353


namespace quiz_score_theorem_l1013_101391

def quiz_scores : List ℕ := [91, 94, 88, 90, 101]
def target_mean : ℕ := 95
def num_quizzes : ℕ := 6
def required_score : ℕ := 106

theorem quiz_score_theorem :
  (List.sum quiz_scores + required_score) / num_quizzes = target_mean := by
  sorry

end quiz_score_theorem_l1013_101391


namespace factorization_equalities_l1013_101373

theorem factorization_equalities (x y : ℝ) : 
  (2 * x^3 - 8 * x = 2 * x * (x + 2) * (x - 2)) ∧ 
  (x^3 - 5 * x^2 + 6 * x = x * (x - 3) * (x - 2)) ∧ 
  (4 * x^4 * y^2 - 5 * x^2 * y^2 - 9 * y^2 = y^2 * (2 * x + 3) * (2 * x - 3) * (x^2 + 1)) ∧ 
  (3 * x^2 - 10 * x * y + 3 * y^2 = (3 * x - y) * (x - 3 * y)) := by
  sorry

end factorization_equalities_l1013_101373


namespace max_distance_Z₁Z₂_l1013_101370

-- Define complex numbers z₁ and z₂
variable (z₁ z₂ : ℂ)

-- Define the conditions
def condition_z₁ : Prop := Complex.abs z₁ ≤ 2
def condition_z₂ : Prop := z₂ = Complex.mk 3 (-4)

-- Define the vector from Z₁ to Z₂
def vector_Z₁Z₂ : ℂ := z₂ - z₁

-- Theorem statement
theorem max_distance_Z₁Z₂ (hz₁ : condition_z₁ z₁) (hz₂ : condition_z₂ z₂) :
  ∃ (max_dist : ℝ), max_dist = 7 ∧ ∀ (z₁' : ℂ), condition_z₁ z₁' → Complex.abs (vector_Z₁Z₂ z₁' z₂) ≤ max_dist :=
sorry

end max_distance_Z₁Z₂_l1013_101370


namespace equation_solutions_l1013_101348

theorem equation_solutions :
  (∀ x : ℝ, (x - 2)^2 = 25 ↔ x = 7 ∨ x = -3) ∧
  (∀ x : ℝ, (x - 5)^2 = 2*(5 - x) ↔ x = 5 ∨ x = 3) := by
  sorry

end equation_solutions_l1013_101348


namespace bottle_capacity_proof_l1013_101376

theorem bottle_capacity_proof (num_boxes : ℕ) (bottles_per_box : ℕ) (fill_ratio : ℚ) (total_volume : ℚ) :
  num_boxes = 10 →
  bottles_per_box = 50 →
  fill_ratio = 3/4 →
  total_volume = 4500 →
  (num_boxes * bottles_per_box * fill_ratio * (12 : ℚ) = total_volume) := by
  sorry

end bottle_capacity_proof_l1013_101376


namespace miley_purchase_cost_l1013_101301

/-- Calculates the total cost of Miley's purchase including discounts and sales tax -/
def total_cost (cellphone_price earbuds_price case_price : ℝ)
               (cellphone_discount earbuds_discount case_discount sales_tax : ℝ) : ℝ :=
  let cellphone_total := 2 * cellphone_price * (1 - cellphone_discount)
  let earbuds_total := 2 * earbuds_price * (1 - earbuds_discount)
  let case_total := 2 * case_price * (1 - case_discount)
  let subtotal := cellphone_total + earbuds_total + case_total
  subtotal * (1 + sales_tax)

/-- Theorem stating that the total cost of Miley's purchase is $2006.64 -/
theorem miley_purchase_cost :
  total_cost 800 150 40 0.05 0.10 0.15 0.08 = 2006.64 := by
  sorry

end miley_purchase_cost_l1013_101301


namespace prob_no_consecutive_heads_10_l1013_101326

/-- The number of coin tosses -/
def n : ℕ := 10

/-- The probability of no two heads appearing consecutively in n coin tosses -/
def prob_no_consecutive_heads (n : ℕ) : ℚ :=
  if n ≤ 1 then 1 else sorry

theorem prob_no_consecutive_heads_10 : 
  prob_no_consecutive_heads n = 9/64 := by sorry

end prob_no_consecutive_heads_10_l1013_101326


namespace inequality_range_l1013_101367

theorem inequality_range (a : ℝ) : 
  (∀ x : ℝ, (a - 2) * x^2 - 2 * (a - 2) * x - 4 < 0) ↔ -2 < a ∧ a ≤ 2 :=
sorry

end inequality_range_l1013_101367


namespace base_eight_digit_product_l1013_101306

/-- Represents a number in base 8 as a list of digits --/
def BaseEight := List Nat

/-- Converts a natural number to its base 8 representation --/
def toBaseEight (n : Nat) : BaseEight :=
  sorry

/-- Decrements each digit in a BaseEight number by 1, removing 0s --/
def decrementDigits (b : BaseEight) : BaseEight :=
  sorry

/-- Computes the product of a list of natural numbers --/
def product (l : List Nat) : Nat :=
  sorry

theorem base_eight_digit_product (n : Nat) :
  n = 7654 →
  product (decrementDigits (toBaseEight n)) = 10 :=
sorry

end base_eight_digit_product_l1013_101306


namespace quadratic_discriminant_l1013_101369

-- Define a quadratic polynomial P(x) = ax^2 + bx + c
def P (a b c : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x + c

-- Define the discriminant of a quadratic polynomial
def discriminant (a b c : ℝ) : ℝ := b^2 - 4*a*c

-- State the theorem
theorem quadratic_discriminant (a b c : ℝ) (h₁ : a ≠ 0) :
  (∃! x, P a b c x = x - 2) →
  (∃! x, P a b c x = 1 - x/2) →
  discriminant a b c = -1/2 := by sorry

end quadratic_discriminant_l1013_101369


namespace complex_equation_solution_l1013_101372

theorem complex_equation_solution (z : ℂ) (h : Complex.I * z = 2 - 4 * Complex.I) : 
  z = -4 - 2 * Complex.I := by
  sorry

end complex_equation_solution_l1013_101372


namespace areas_sum_equal_largest_l1013_101396

/-- Represents a triangle inscribed in a circle -/
structure InscribedTriangle where
  -- The sides of the triangle
  side1 : ℝ
  side2 : ℝ
  side3 : ℝ
  -- The areas of the non-triangular regions
  area_D : ℝ
  area_E : ℝ
  area_F : ℝ
  -- Conditions
  isosceles : side1 = side2
  sides : side1 = 12 ∧ side2 = 12 ∧ side3 = 20
  largest_F : area_F ≥ area_D ∧ area_F ≥ area_E

/-- Theorem stating that D + E = F for the given inscribed triangle -/
theorem areas_sum_equal_largest (t : InscribedTriangle) : t.area_D + t.area_E = t.area_F := by
  sorry

end areas_sum_equal_largest_l1013_101396


namespace planes_parallel_condition_l1013_101334

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the subset relation for lines and planes
variable (subset : Line → Plane → Prop)

-- Define the parallel relation for lines and planes
variable (parallel : Line → Line → Prop)
variable (planeParallel : Plane → Plane → Prop)

-- Define the intersection operation for lines
variable (intersect : Line → Line → Set Point)

-- Define the specific lines and planes
variable (m n l₁ l₂ : Line) (α β : Plane) (M : Point)

-- State the theorem
theorem planes_parallel_condition 
  (h1 : subset m α)
  (h2 : subset n α)
  (h3 : subset l₁ β)
  (h4 : subset l₂ β)
  (h5 : intersect l₁ l₂ = {M})
  (h6 : parallel m l₁)
  (h7 : parallel n l₂) :
  planeParallel α β :=
sorry

end planes_parallel_condition_l1013_101334


namespace adult_ticket_price_is_five_l1013_101329

/-- Represents the ticket sales for a baseball game at a community center -/
structure TicketSales where
  total_tickets : ℕ
  adult_tickets : ℕ
  child_ticket_price : ℕ
  total_revenue : ℕ

/-- The price of an adult ticket given the ticket sales information -/
def adult_ticket_price (sales : TicketSales) : ℕ :=
  (sales.total_revenue - (sales.total_tickets - sales.adult_tickets) * sales.child_ticket_price) / sales.adult_tickets

/-- Theorem stating that the adult ticket price is $5 given the specific sales information -/
theorem adult_ticket_price_is_five :
  let sales : TicketSales := {
    total_tickets := 85,
    adult_tickets := 35,
    child_ticket_price := 2,
    total_revenue := 275
  }
  adult_ticket_price sales = 5 := by
  sorry

end adult_ticket_price_is_five_l1013_101329


namespace pop_survey_result_l1013_101321

theorem pop_survey_result (total_surveyed : ℕ) (pop_angle : ℕ) (people_chose_pop : ℕ) : 
  total_surveyed = 540 → pop_angle = 270 → people_chose_pop = total_surveyed * pop_angle / 360 →
  people_chose_pop = 405 := by
sorry

end pop_survey_result_l1013_101321


namespace sqrt_seven_squared_minus_four_l1013_101351

theorem sqrt_seven_squared_minus_four : (Real.sqrt 7 + 2) * (Real.sqrt 7 - 2) = 3 := by
  sorry

end sqrt_seven_squared_minus_four_l1013_101351


namespace egg_pack_size_l1013_101344

/-- The number of rotten eggs in the pack -/
def rotten_eggs : ℕ := 3

/-- The probability of choosing 2 rotten eggs -/
def prob_two_rotten : ℚ := 47619047619047615 / 10000000000000000

/-- The total number of eggs in the pack -/
def total_eggs : ℕ := 36

/-- Theorem stating that given the number of rotten eggs and the probability of choosing 2 rotten eggs, 
    the total number of eggs in the pack is 36 -/
theorem egg_pack_size :
  (rotten_eggs : ℚ) / total_eggs * (rotten_eggs - 1 : ℚ) / (total_eggs - 1) = prob_two_rotten :=
sorry

end egg_pack_size_l1013_101344


namespace cos_225_degrees_l1013_101311

theorem cos_225_degrees : Real.cos (225 * π / 180) = -Real.sqrt 2 / 2 := by
  sorry

end cos_225_degrees_l1013_101311


namespace magnitude_of_5_minus_12i_l1013_101342

theorem magnitude_of_5_minus_12i : Complex.abs (5 - 12 * Complex.I) = 13 := by
  sorry

end magnitude_of_5_minus_12i_l1013_101342


namespace work_completion_time_l1013_101300

theorem work_completion_time (man_time son_time : ℝ) (h1 : man_time = 6) (h2 : son_time = 6) :
  1 / (1 / man_time + 1 / son_time) = 3 := by
  sorry

end work_completion_time_l1013_101300


namespace floor_equality_l1013_101347

theorem floor_equality (m : ℝ) (h : m ≥ 3) :
  ⌊m * (m + 1) / (2 * (2 * m - 1))⌋ = ⌊(m + 1) / 4⌋ := by
  sorry

end floor_equality_l1013_101347


namespace pages_left_after_eleven_days_l1013_101398

/-- Represents the number of pages left unread after reading for a given number of days -/
def pages_left (total_pages : ℕ) (pages_per_day : ℕ) (days : ℕ) : ℕ :=
  total_pages - pages_per_day * days

/-- Theorem stating that reading 15 pages a day for 11 days from a 250-page book leaves 85 pages unread -/
theorem pages_left_after_eleven_days :
  pages_left 250 15 11 = 85 := by
  sorry

end pages_left_after_eleven_days_l1013_101398


namespace original_number_not_800_l1013_101341

theorem original_number_not_800 : ¬(∃ x : ℝ, x * 10 = x + 720 ∧ x = 800) := by
  sorry

end original_number_not_800_l1013_101341


namespace angle_problem_l1013_101323

theorem angle_problem (angle1 angle2 angle3 angle4 : ℝ) : 
  angle1 + angle2 = 180 →
  angle3 = angle4 →
  angle1 + 50 + 60 = 180 →
  angle4 = 35 := by
sorry

end angle_problem_l1013_101323


namespace pauls_and_sarahs_ages_l1013_101333

theorem pauls_and_sarahs_ages (p s : ℕ) : 
  p = s + 8 →                   -- Paul is eight years older than Sarah
  p + 6 = 3 * (s - 2) →         -- In six years, Paul will be three times as old as Sarah was two years ago
  p + s = 28                    -- The sum of their current ages is 28
  := by sorry

end pauls_and_sarahs_ages_l1013_101333


namespace alcohol_concentration_reduction_specific_alcohol_reduction_l1013_101328

/-- Calculates the percentage reduction in alcohol concentration when water is added to a solution. -/
theorem alcohol_concentration_reduction 
  (initial_volume : ℝ) 
  (initial_concentration : ℝ) 
  (water_added : ℝ) : ℝ :=
  let initial_alcohol := initial_volume * initial_concentration
  let final_volume := initial_volume + water_added
  let final_concentration := initial_alcohol / final_volume
  let reduction_percentage := (initial_concentration - final_concentration) / initial_concentration * 100
  by
    -- Proof goes here
    sorry

/-- The specific problem statement -/
theorem specific_alcohol_reduction : 
  alcohol_concentration_reduction 15 0.20 25 = 62.5 := by
  -- Proof goes here
  sorry

end alcohol_concentration_reduction_specific_alcohol_reduction_l1013_101328


namespace tangent_line_to_exponential_curve_l1013_101397

/-- A line is tangent to a curve if it intersects the curve at exactly one point and has the same slope as the curve at that point. -/
def is_tangent (f g : ℝ → ℝ) : Prop :=
  ∃ x₀ : ℝ, f x₀ = g x₀ ∧ (deriv f) x₀ = (deriv g) x₀

/-- The problem statement -/
theorem tangent_line_to_exponential_curve (a : ℝ) :
  is_tangent (fun x => x - 3) (fun x => Real.exp (x + a)) → a = -4 := by
  sorry

end tangent_line_to_exponential_curve_l1013_101397


namespace minimum_cost_theorem_l1013_101319

/-- Represents the cost and capacity of buses -/
structure BusType where
  cost : ℕ
  capacity : ℕ

/-- Represents the problem setup -/
structure BusRentalProblem where
  busA : BusType
  busB : BusType
  totalPeople : ℕ
  totalBuses : ℕ
  costOneEach : ℕ
  costTwoAThreeB : ℕ

/-- Calculates the total cost for a given number of each bus type -/
def totalCost (problem : BusRentalProblem) (numA : ℕ) : ℕ :=
  numA * problem.busA.cost + (problem.totalBuses - numA) * problem.busB.cost

/-- Calculates the total capacity for a given number of each bus type -/
def totalCapacity (problem : BusRentalProblem) (numA : ℕ) : ℕ :=
  numA * problem.busA.capacity + (problem.totalBuses - numA) * problem.busB.capacity

/-- The main theorem to prove -/
theorem minimum_cost_theorem (problem : BusRentalProblem) 
  (h1 : problem.busA.cost + problem.busB.cost = problem.costOneEach)
  (h2 : 2 * problem.busA.cost + 3 * problem.busB.cost = problem.costTwoAThreeB)
  (h3 : problem.busA.capacity = 15)
  (h4 : problem.busB.capacity = 25)
  (h5 : problem.totalPeople = 170)
  (h6 : problem.totalBuses = 8)
  (h7 : problem.costOneEach = 500)
  (h8 : problem.costTwoAThreeB = 1300) :
  ∃ (numA : ℕ), 
    numA ≤ problem.totalBuses ∧ 
    totalCapacity problem numA ≥ problem.totalPeople ∧
    totalCost problem numA = 2100 ∧
    ∀ (k : ℕ), k ≤ problem.totalBuses → 
      totalCapacity problem k ≥ problem.totalPeople → 
      totalCost problem k ≥ 2100 := by
  sorry


end minimum_cost_theorem_l1013_101319


namespace solution_set_equivalence_l1013_101356

theorem solution_set_equivalence (x : ℝ) :
  (3*x - 1) / (2 - x) ≥ 1 ↔ 3/4 ≤ x ∧ x < 2 := by
sorry

end solution_set_equivalence_l1013_101356


namespace brand_a_millet_percentage_l1013_101382

/-- Represents the composition of a birdseed brand -/
structure BirdseedBrand where
  millet : ℝ
  sunflower : ℝ
  composition_sum : millet + sunflower = 100

/-- Represents a mix of two birdseed brands -/
structure BirdseedMix where
  brand_a : BirdseedBrand
  brand_b : BirdseedBrand
  proportion_a : ℝ
  proportion_b : ℝ
  proportions_sum : proportion_a + proportion_b = 100
  sunflower_percent : ℝ
  sunflower_balance : proportion_a / 100 * brand_a.sunflower + proportion_b / 100 * brand_b.sunflower = sunflower_percent

/-- Theorem stating that Brand A has 40% millet given the problem conditions -/
theorem brand_a_millet_percentage 
  (brand_a : BirdseedBrand)
  (brand_b : BirdseedBrand)
  (mix : BirdseedMix)
  (ha : brand_a.sunflower = 60)
  (hb1 : brand_b.millet = 65)
  (hb2 : brand_b.sunflower = 35)
  (hm1 : mix.sunflower_percent = 50)
  (hm2 : mix.proportion_a = 60)
  (hm3 : mix.brand_a = brand_a)
  (hm4 : mix.brand_b = brand_b) :
  brand_a.millet = 40 :=
sorry

end brand_a_millet_percentage_l1013_101382


namespace cameron_paper_count_l1013_101343

theorem cameron_paper_count (initial_papers : ℕ) : 
  (initial_papers : ℚ) * (60 : ℚ) / 100 = 240 → initial_papers = 400 := by
  sorry

end cameron_paper_count_l1013_101343


namespace intersection_and_union_when_m_is_3_union_equals_A_iff_m_in_range_l1013_101316

-- Define sets A and B
def A : Set ℝ := {x | -2 ≤ x ∧ x ≤ 5}
def B (m : ℝ) : Set ℝ := {x | m - 1 ≤ x ∧ x ≤ 2 * m + 1}

-- Theorem for part 1
theorem intersection_and_union_when_m_is_3 :
  (A ∩ B 3 = {x | 2 ≤ x ∧ x ≤ 5}) ∧
  ((Aᶜ ∪ B 3) = {x | x < -2 ∨ 2 ≤ x}) :=
sorry

-- Theorem for part 2
theorem union_equals_A_iff_m_in_range :
  ∀ m : ℝ, (A ∪ B m = A) ↔ (m < -2 ∨ (-1 ≤ m ∧ m ≤ 2)) :=
sorry

end intersection_and_union_when_m_is_3_union_equals_A_iff_m_in_range_l1013_101316


namespace shampoo_duration_l1013_101305

-- Define the amount of rose shampoo Janet has
def rose_shampoo : ℚ := 1/3

-- Define the amount of jasmine shampoo Janet has
def jasmine_shampoo : ℚ := 1/4

-- Define the amount of shampoo Janet uses per day
def daily_usage : ℚ := 1/12

-- Theorem statement
theorem shampoo_duration :
  (rose_shampoo + jasmine_shampoo) / daily_usage = 7 := by
  sorry

end shampoo_duration_l1013_101305


namespace inverse_89_mod_91_l1013_101371

theorem inverse_89_mod_91 : ∃ x : ℕ, x < 91 ∧ (89 * x) % 91 = 1 ∧ x = 45 := by
  sorry

end inverse_89_mod_91_l1013_101371


namespace product_of_roots_eq_one_l1013_101380

theorem product_of_roots_eq_one : 
  ∃ (r₁ r₂ : ℝ), r₁ * r₂ = 1 ∧ r₁^(2*Real.log r₁) = ℯ ∧ r₂^(2*Real.log r₂) = ℯ ∧
  ∀ (x : ℝ), x^(2*Real.log x) = ℯ → x = r₁ ∨ x = r₂ :=
by sorry

end product_of_roots_eq_one_l1013_101380


namespace circle_equation_l1013_101314

theorem circle_equation (x y : ℝ) : 
  let center : ℝ × ℝ := (2, -1)
  let point : ℝ × ℝ := (-1, 3)
  let radius : ℝ := Real.sqrt ((center.1 - point.1)^2 + (center.2 - point.2)^2)
  (x - center.1)^2 + (y - center.2)^2 = radius^2 :=
by sorry

end circle_equation_l1013_101314


namespace greatest_number_with_odd_factors_under_500_l1013_101389

def is_perfect_square (n : ℕ) : Prop := ∃ m : ℕ, n = m * m

def has_odd_number_of_factors (n : ℕ) : Prop := is_perfect_square n

theorem greatest_number_with_odd_factors_under_500 :
  ∃ n : ℕ, n < 500 ∧ has_odd_number_of_factors n ∧
  ∀ m : ℕ, m < 500 ∧ has_odd_number_of_factors m → m ≤ n :=
by sorry

end greatest_number_with_odd_factors_under_500_l1013_101389


namespace rodney_commission_l1013_101318

/-- Rodney's commission for selling home security systems --/
def commission_per_sale : ℕ := 25

/-- Number of streets in the neighborhood --/
def num_streets : ℕ := 4

/-- Number of houses on each street --/
def houses_per_street : ℕ := 8

/-- Sales on the second street --/
def sales_second_street : ℕ := 4

/-- Sales on the first street (half of second street) --/
def sales_first_street : ℕ := sales_second_street / 2

/-- Sales on the third street (no sales) --/
def sales_third_street : ℕ := 0

/-- Sales on the fourth street --/
def sales_fourth_street : ℕ := 1

/-- Total sales across all streets --/
def total_sales : ℕ := sales_first_street + sales_second_street + sales_third_street + sales_fourth_street

/-- Rodney's total commission --/
def total_commission : ℕ := total_sales * commission_per_sale

theorem rodney_commission : total_commission = 175 := by
  sorry

end rodney_commission_l1013_101318


namespace project_completion_time_l1013_101379

/-- The number of days it takes for two workers to complete a job -/
structure WorkerPair :=
  (worker1 : ℕ)
  (worker2 : ℕ)
  (days : ℕ)

/-- The rate at which a worker completes the job per day -/
def workerRate (days : ℕ) : ℚ :=
  1 / days

theorem project_completion_time 
  (ab : WorkerPair) 
  (bc : WorkerPair) 
  (c_alone : ℕ) 
  (a_days : ℕ) 
  (b_days : ℕ) :
  ab.days = 10 →
  bc.days = 18 →
  c_alone = 45 →
  a_days = 5 →
  b_days = 10 →
  ∃ (c_days : ℕ), c_days = 15 ∧ 
    (workerRate ab.days * a_days + 
     workerRate ab.days * b_days + 
     workerRate c_alone * c_days = 1) :=
by sorry

end project_completion_time_l1013_101379


namespace rational_solutions_quadratic_l1013_101327

theorem rational_solutions_quadratic (k : ℕ+) : 
  (∃ x : ℚ, k * x^2 + 26 * x + k = 0) ↔ k = 5 := by
sorry

end rational_solutions_quadratic_l1013_101327


namespace sales_volume_estimate_l1013_101362

-- Define the regression equation
def regression_equation (x : ℝ) : ℝ := -10 * x + 200

-- State the theorem
theorem sales_volume_estimate :
  ∃ ε > 0, |regression_equation 10 - 100| < ε :=
sorry

end sales_volume_estimate_l1013_101362


namespace shells_added_l1013_101390

theorem shells_added (initial_amount final_amount : ℕ) 
  (h1 : initial_amount = 5)
  (h2 : final_amount = 17) :
  final_amount - initial_amount = 12 := by
  sorry

end shells_added_l1013_101390


namespace quadratic_complex_conjugate_roots_l1013_101331

theorem quadratic_complex_conjugate_roots (a b : ℝ) : 
  (∃ x y : ℝ, (Complex.I * x + y) ^ 2 + (6 + Complex.I * a) * (Complex.I * x + y) + (15 + Complex.I * b) = 0 ∧
               (Complex.I * (-x) + y) ^ 2 + (6 + Complex.I * a) * (Complex.I * (-x) + y) + (15 + Complex.I * b) = 0) →
  a = 0 ∧ b = 0 := by
sorry

end quadratic_complex_conjugate_roots_l1013_101331


namespace quadratic_coefficient_sum_l1013_101322

theorem quadratic_coefficient_sum (a k n : ℤ) : 
  (∀ x : ℤ, (3*x + 2)*(2*x - 7) = a*x^2 + k*x + n) → 
  a - n + k = 3 := by
  sorry

end quadratic_coefficient_sum_l1013_101322


namespace x_pow_zero_eq_one_f_eq_S_l1013_101381

-- Define the functions
def f (x : ℝ) := x^2
def S (t : ℝ) := t^2

-- Theorem statements
theorem x_pow_zero_eq_one (x : ℝ) (h : x ≠ 0) : x^0 = 1 := by sorry

theorem f_eq_S : ∀ x : ℝ, f x = S x := by sorry

end x_pow_zero_eq_one_f_eq_S_l1013_101381
