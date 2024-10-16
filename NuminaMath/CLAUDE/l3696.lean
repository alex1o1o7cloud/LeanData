import Mathlib

namespace NUMINAMATH_CALUDE_nine_to_fourth_equals_three_to_eighth_l3696_369601

theorem nine_to_fourth_equals_three_to_eighth : (9 : ℕ) ^ 4 = 3 ^ 8 := by
  sorry

end NUMINAMATH_CALUDE_nine_to_fourth_equals_three_to_eighth_l3696_369601


namespace NUMINAMATH_CALUDE_square_properties_l3696_369696

structure Square where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ
  D : ℝ × ℝ

def square : Square := {
  A := (0, 0),
  B := (-5, -3),
  C := (-4, -8),
  D := (1, -5)
}

theorem square_properties (s : Square) (h : s = square) :
  let side_length := Real.sqrt ((s.B.1 - s.A.1)^2 + (s.B.2 - s.A.2)^2)
  (side_length^2 = 34) ∧ (4 * side_length = 4 * Real.sqrt 34) := by
  sorry

#check square_properties

end NUMINAMATH_CALUDE_square_properties_l3696_369696


namespace NUMINAMATH_CALUDE_sqrt_product_sqrt_three_times_sqrt_two_l3696_369686

theorem sqrt_product (a b : ℝ) (ha : 0 ≤ a) (hb : 0 ≤ b) :
  Real.sqrt (a * b) = Real.sqrt a * Real.sqrt b := by sorry

theorem sqrt_three_times_sqrt_two :
  Real.sqrt 3 * Real.sqrt 2 = Real.sqrt 6 := by sorry

end NUMINAMATH_CALUDE_sqrt_product_sqrt_three_times_sqrt_two_l3696_369686


namespace NUMINAMATH_CALUDE_poly_has_four_nonzero_terms_l3696_369641

/-- The polynomial expression -/
def poly (x : ℝ) : ℝ := (2*x + 5)*(3*x^2 - x + 4) + 4*(x^3 + x^2 - 6*x)

/-- The expansion of the polynomial -/
def expanded_poly (x : ℝ) : ℝ := 10*x^3 + 17*x^2 - 21*x + 20

/-- Theorem stating that the polynomial has exactly 4 nonzero terms -/
theorem poly_has_four_nonzero_terms :
  ∃ (a b c d : ℝ) (n : ℕ), 
    a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0 ∧ d ≠ 0 ∧
    (∀ x, poly x = expanded_poly x) ∧
    (∀ x, expanded_poly x = a*x^3 + b*x^2 + c*x + d) ∧
    n = 4 := by sorry

end NUMINAMATH_CALUDE_poly_has_four_nonzero_terms_l3696_369641


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_sets_l3696_369646

theorem quadratic_inequality_solution_sets (m x : ℝ) :
  let f := fun x => m * x^2 - (m + 1) * x + 1
  (m = 2 → (f x < 0 ↔ 1/2 < x ∧ x < 1)) ∧
  (m > 0 →
    ((0 < m ∧ m < 1) → (f x < 0 ↔ 1 < x ∧ x < 1/m)) ∧
    (m = 1 → ¬∃ x, f x < 0) ∧
    (m > 1 → (f x < 0 ↔ 1/m < x ∧ x < 1))) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_sets_l3696_369646


namespace NUMINAMATH_CALUDE_total_lines_through_centers_l3696_369615

/-- The size of the cube --/
def cube_size : Nat := 2008

/-- The number of lines parallel to the edges of the cube --/
def parallel_lines : Nat := cube_size * cube_size * 3

/-- The number of diagonal lines within the planes --/
def diagonal_lines : Nat := cube_size * 2 * 3

/-- The number of space diagonals of the cube --/
def space_diagonals : Nat := 4

/-- Theorem stating the total number of lines passing through the centers of exactly 2008 unit cubes in a 2008 x 2008 x 2008 cube --/
theorem total_lines_through_centers (cube_size : Nat) (h : cube_size = 2008) :
  parallel_lines + diagonal_lines + space_diagonals = 12115300 := by
  sorry

#eval parallel_lines + diagonal_lines + space_diagonals

end NUMINAMATH_CALUDE_total_lines_through_centers_l3696_369615


namespace NUMINAMATH_CALUDE_tree_height_l3696_369650

/-- Given Jane's height, Jane's shadow length, and the tree's shadow length,
    prove that the tree's height is 30 meters. -/
theorem tree_height (jane_height jane_shadow tree_shadow : ℝ)
  (h1 : jane_height = 1.5)
  (h2 : jane_shadow = 0.5)
  (h3 : tree_shadow = 10)
  (h4 : ∀ (obj1 obj2 : ℝ), obj1 / jane_shadow = jane_height / jane_shadow → obj1 / obj2 = jane_height / jane_shadow) :
  jane_height / jane_shadow * tree_shadow = 30 := by
  sorry

end NUMINAMATH_CALUDE_tree_height_l3696_369650


namespace NUMINAMATH_CALUDE_proposition_p_false_implies_a_range_l3696_369655

theorem proposition_p_false_implies_a_range (a : ℝ) : 
  (¬ ∀ x : ℝ, a * x^2 + a * x + 1 ≥ 0) → 
  (a < 0 ∨ a > 4) := by
sorry

end NUMINAMATH_CALUDE_proposition_p_false_implies_a_range_l3696_369655


namespace NUMINAMATH_CALUDE_sector_area_l3696_369666

theorem sector_area (r : ℝ) (θ : ℝ) (h : θ = 72 * π / 180) :
  let A := (θ / (2 * π)) * π * r^2
  r = 20 → A = 80 * π := by
  sorry

end NUMINAMATH_CALUDE_sector_area_l3696_369666


namespace NUMINAMATH_CALUDE_irrationality_of_sqrt_two_and_rationality_of_others_l3696_369607

-- Define rationality
def IsRational (x : ℝ) : Prop := ∃ (p q : ℤ), q ≠ 0 ∧ x = p / q

-- Define irrationality
def IsIrrational (x : ℝ) : Prop := ¬(IsRational x)

-- Theorem statement
theorem irrationality_of_sqrt_two_and_rationality_of_others :
  IsIrrational (Real.sqrt 2) ∧ 
  IsRational 3.14 ∧ 
  IsRational (22 / 7) ∧ 
  IsRational 0 :=
sorry

end NUMINAMATH_CALUDE_irrationality_of_sqrt_two_and_rationality_of_others_l3696_369607


namespace NUMINAMATH_CALUDE_smallest_positive_d_smallest_d_is_zero_l3696_369670

theorem smallest_positive_d (d : ℝ) (hd : d > 0) :
  ∀ (x y : ℝ), x ≥ 0 → y ≥ 0 → Real.sqrt (x * y) + d * (x - y)^2 ≥ (x + y) / 2 := by
  sorry

/-- The smallest positive real number d that satisfies the inequality for all nonnegative x and y is 0 -/
theorem smallest_d_is_zero :
  ∀ ε > 0, ∃ (x y : ℝ), x ≥ 0 ∧ y ≥ 0 ∧ Real.sqrt (x * y) + ε * (x - y)^2 < (x + y) / 2 := by
  sorry

end NUMINAMATH_CALUDE_smallest_positive_d_smallest_d_is_zero_l3696_369670


namespace NUMINAMATH_CALUDE_parallel_lines_k_values_l3696_369662

/-- Two lines in the plane -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Check if two lines are parallel -/
def parallel (l1 l2 : Line) : Prop :=
  l1.a * l2.b = l1.b * l2.a

/-- The first line -/
def l1 (k : ℝ) : Line :=
  { a := k - 3
    b := 4 - k
    c := 1 }

/-- The second line -/
def l2 (k : ℝ) : Line :=
  { a := 2 * (k - 3)
    b := -2
    c := 3 }

/-- Theorem: If l1 and l2 are parallel, then k is either 3 or 5 -/
theorem parallel_lines_k_values :
  ∀ k : ℝ, parallel (l1 k) (l2 k) → k = 3 ∨ k = 5 := by
  sorry

end NUMINAMATH_CALUDE_parallel_lines_k_values_l3696_369662


namespace NUMINAMATH_CALUDE_addition_is_unique_solution_l3696_369648

-- Define the possible operations
inductive Operation
| Add
| Subtract
| Multiply
| Divide

-- Define a function to apply the operation
def applyOperation (op : Operation) (a b : Int) : Int :=
  match op with
  | Operation.Add => a + b
  | Operation.Subtract => a - b
  | Operation.Multiply => a * b
  | Operation.Divide => a / b

-- Theorem statement
theorem addition_is_unique_solution :
  ∃! op : Operation, applyOperation op 5 (-5) = 0 :=
sorry

end NUMINAMATH_CALUDE_addition_is_unique_solution_l3696_369648


namespace NUMINAMATH_CALUDE_spider_plant_theorem_l3696_369626

def spider_plant_problem (baby_plants_per_time : ℕ) (times_per_year : ℕ) (total_baby_plants : ℕ) : Prop :=
  let baby_plants_per_year := baby_plants_per_time * times_per_year
  let years_passed := total_baby_plants / baby_plants_per_year
  years_passed = 4

theorem spider_plant_theorem :
  spider_plant_problem 2 2 16 := by
  sorry

end NUMINAMATH_CALUDE_spider_plant_theorem_l3696_369626


namespace NUMINAMATH_CALUDE_floor_plus_x_eq_seventeen_fourths_l3696_369661

theorem floor_plus_x_eq_seventeen_fourths :
  ∃ (x : ℚ), (⌊x⌋ : ℚ) + x = 17 / 4 ∧ x = 9 / 4 := by
  sorry

end NUMINAMATH_CALUDE_floor_plus_x_eq_seventeen_fourths_l3696_369661


namespace NUMINAMATH_CALUDE_number_of_divisors_2310_l3696_369642

theorem number_of_divisors_2310 : Nat.card (Nat.divisors 2310) = 32 := by
  sorry

end NUMINAMATH_CALUDE_number_of_divisors_2310_l3696_369642


namespace NUMINAMATH_CALUDE_cookie_production_cost_l3696_369699

/-- The cost to produce one cookie -/
def production_cost : ℝ := sorry

/-- The selling price of one cookie -/
def selling_price : ℝ := 1.2 * production_cost

/-- The number of cookies sold -/
def cookies_sold : ℕ := 50

/-- The total revenue from selling the cookies -/
def total_revenue : ℝ := 60

theorem cookie_production_cost :
  production_cost = 1 :=
by sorry

end NUMINAMATH_CALUDE_cookie_production_cost_l3696_369699


namespace NUMINAMATH_CALUDE_gmat_test_problem_l3696_369635

theorem gmat_test_problem (second_correct : ℝ) (neither_correct : ℝ) (both_correct : ℝ)
  (h1 : second_correct = 65)
  (h2 : neither_correct = 5)
  (h3 : both_correct = 55) :
  100 - neither_correct - (second_correct - both_correct) = 85 :=
by
  sorry

end NUMINAMATH_CALUDE_gmat_test_problem_l3696_369635


namespace NUMINAMATH_CALUDE_square_area_with_four_circles_l3696_369651

theorem square_area_with_four_circles (r : ℝ) (h : r = 8) : 
  (2 * (2 * r))^2 = 1024 := by
  sorry

end NUMINAMATH_CALUDE_square_area_with_four_circles_l3696_369651


namespace NUMINAMATH_CALUDE_no_solution_exists_l3696_369698

theorem no_solution_exists : ¬ ∃ (n m r : ℕ), 
  n ≥ 1 ∧ m ≥ 1 ∧ r ≥ 1 ∧ n^5 + 49^m = 1221^r := by
  sorry

end NUMINAMATH_CALUDE_no_solution_exists_l3696_369698


namespace NUMINAMATH_CALUDE_stadium_empty_seats_l3696_369690

/-- The number of empty seats in a stadium -/
def empty_seats (total_seats people_present : ℕ) : ℕ :=
  total_seats - people_present

/-- Theorem: Given a stadium with 92 seats and 47 people present, there are 45 empty seats -/
theorem stadium_empty_seats : empty_seats 92 47 = 45 := by
  sorry

end NUMINAMATH_CALUDE_stadium_empty_seats_l3696_369690


namespace NUMINAMATH_CALUDE_daisies_bought_l3696_369633

theorem daisies_bought (flower_price : ℕ) (roses_bought : ℕ) (total_spent : ℕ) : ℕ :=
  let daisies : ℕ := (total_spent - roses_bought * flower_price) / flower_price
  by
    -- Proof goes here
    sorry

#check daisies_bought 6 7 60 = 3

end NUMINAMATH_CALUDE_daisies_bought_l3696_369633


namespace NUMINAMATH_CALUDE_waiter_new_customers_l3696_369652

theorem waiter_new_customers 
  (initial_customers : ℕ) 
  (customers_left : ℕ) 
  (remaining_customers : ℕ) 
  (final_total_customers : ℕ) : 
  initial_customers = 8 →
  customers_left = 3 →
  remaining_customers = 5 →
  final_total_customers = 104 →
  final_total_customers - remaining_customers = 99 :=
by sorry

end NUMINAMATH_CALUDE_waiter_new_customers_l3696_369652


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l3696_369628

-- Define set A
def A : Set ℝ := {y | ∃ x, y = 2^x - 1}

-- Define set B
def B : Set ℝ := {x | |2*x - 3| ≤ 3}

-- Theorem statement
theorem intersection_of_A_and_B :
  A ∩ B = {x | 0 < x ∧ x ≤ 3} := by
  sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l3696_369628


namespace NUMINAMATH_CALUDE_triangle_perimeter_property_l3696_369644

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a regular hexagon with squares on alternate sides -/
structure HexagonWithSquares where
  /-- Side length of the hexagon -/
  hexagonSide : ℝ
  /-- Side length of the squares -/
  squareSide : ℝ
  /-- Centers of the squares -/
  squareCenters : Fin 3 → Point

/-- The perimeter of the triangle formed by the centers of the squares -/
def trianglePerimeter (h : HexagonWithSquares) : ℝ :=
  sorry

/-- Theorem stating the property of the triangle perimeter -/
theorem triangle_perimeter_property (h : HexagonWithSquares) 
  (h1 : h.squareSide = 4) : 
  ∃ (a b : ℤ), trianglePerimeter h = a + b * Real.sqrt 3 ∧ a + b = 20 := by
  sorry

end NUMINAMATH_CALUDE_triangle_perimeter_property_l3696_369644


namespace NUMINAMATH_CALUDE_largest_prime_divisor_l3696_369639

/-- Converts a base 4 number to decimal --/
def base4ToDecimal (n : List Nat) : Nat :=
  n.enum.foldl (fun acc (i, d) => acc + d * (4 ^ i)) 0

/-- The base 4 representation of the number --/
def number : List Nat := [1, 2, 0, 1, 0, 0, 2, 0, 1]

/-- The decimal representation of the number --/
def decimalNumber : Nat := base4ToDecimal number

theorem largest_prime_divisor :
  ∃ (p : Nat), Nat.Prime p ∧ p ∣ decimalNumber ∧ ∀ (q : Nat), Nat.Prime q → q ∣ decimalNumber → q ≤ p ∧ p = 181 := by
  sorry

end NUMINAMATH_CALUDE_largest_prime_divisor_l3696_369639


namespace NUMINAMATH_CALUDE_abs_lt_one_sufficient_not_necessary_for_lt_one_l3696_369618

theorem abs_lt_one_sufficient_not_necessary_for_lt_one :
  (∃ x : ℝ, (|x| < 1 → x < 1) ∧ ¬(x < 1 → |x| < 1)) := by
  sorry

end NUMINAMATH_CALUDE_abs_lt_one_sufficient_not_necessary_for_lt_one_l3696_369618


namespace NUMINAMATH_CALUDE_youngest_brother_age_l3696_369647

/-- Represents the ages of Rick and his brothers -/
structure FamilyAges where
  rick : ℕ
  oldest : ℕ
  middle : ℕ
  smallest : ℕ
  youngest : ℕ

/-- Defines the relationships between the ages in the family -/
def validFamilyAges (ages : FamilyAges) : Prop :=
  ages.rick = 15 ∧
  ages.oldest = 2 * ages.rick ∧
  ages.middle = ages.oldest / 3 ∧
  ages.smallest = ages.middle / 2 ∧
  ages.youngest = ages.smallest - 2

/-- Theorem stating that given the family age relationships, the youngest brother is 3 years old -/
theorem youngest_brother_age (ages : FamilyAges) (h : validFamilyAges ages) : ages.youngest = 3 := by
  sorry

end NUMINAMATH_CALUDE_youngest_brother_age_l3696_369647


namespace NUMINAMATH_CALUDE_four_digit_cubes_divisible_by_16_l3696_369691

theorem four_digit_cubes_divisible_by_16 :
  (∃! (s : Finset ℕ), s = {n : ℕ | 1000 ≤ (2*n)^3 ∧ (2*n)^3 ≤ 9999} ∧ Finset.card s = 3) := by
  sorry

end NUMINAMATH_CALUDE_four_digit_cubes_divisible_by_16_l3696_369691


namespace NUMINAMATH_CALUDE_inequality_solution_l3696_369610

theorem inequality_solution (a : ℝ) (f : ℝ → ℝ) (h : ∀ x, f x = a * x * (x + 1) + 1) :
  {x : ℝ | f x < 0} = {x : ℝ | x < 1/a ∨ x > 1} ∩ {x : ℝ | a ≠ 0} :=
by sorry

end NUMINAMATH_CALUDE_inequality_solution_l3696_369610


namespace NUMINAMATH_CALUDE_sector_radius_l3696_369649

theorem sector_radius (l a : ℝ) (hl : l = 10 * Real.pi) (ha : a = 60 * Real.pi) :
  ∃ r : ℝ, r = 12 ∧ a = (1 / 2) * l * r := by
  sorry

end NUMINAMATH_CALUDE_sector_radius_l3696_369649


namespace NUMINAMATH_CALUDE_monotonic_increasing_cubic_l3696_369675

/-- A cubic function with parameters m and n -/
def f (m n : ℝ) (x : ℝ) : ℝ := 4 * x^3 + m * x^2 + (m - 3) * x + n

/-- The derivative of f with respect to x -/
def f' (m : ℝ) (x : ℝ) : ℝ := 12 * x^2 + 2 * m * x + (m - 3)

theorem monotonic_increasing_cubic (m n : ℝ) :
  (∀ x : ℝ, Monotone (f m n)) → m = 6 := by
  sorry

end NUMINAMATH_CALUDE_monotonic_increasing_cubic_l3696_369675


namespace NUMINAMATH_CALUDE_morgan_paid_twenty_l3696_369671

/-- Represents the cost of Morgan's lunch items and the change received --/
structure LunchTransaction where
  hamburger_cost : ℕ
  onion_rings_cost : ℕ
  smoothie_cost : ℕ
  change_received : ℕ

/-- Calculates the total amount paid by Morgan --/
def amount_paid (t : LunchTransaction) : ℕ :=
  t.hamburger_cost + t.onion_rings_cost + t.smoothie_cost + t.change_received

/-- Theorem stating that Morgan paid $20 --/
theorem morgan_paid_twenty :
  ∀ (t : LunchTransaction),
    t.hamburger_cost = 4 →
    t.onion_rings_cost = 2 →
    t.smoothie_cost = 3 →
    t.change_received = 11 →
    amount_paid t = 20 :=
by sorry

end NUMINAMATH_CALUDE_morgan_paid_twenty_l3696_369671


namespace NUMINAMATH_CALUDE_inequality_proof_l3696_369669

theorem inequality_proof (x y : ℝ) (hx : x > -1) (hy : y > -1) (hsum : x + y = 1) :
  x / (y + 1) + y / (x + 1) ≥ 2 / 3 ∧
  (x / (y + 1) + y / (x + 1) = 2 / 3 ↔ x = 1 / 2 ∧ y = 1 / 2) :=
by sorry

end NUMINAMATH_CALUDE_inequality_proof_l3696_369669


namespace NUMINAMATH_CALUDE_tangent_line_and_max_value_l3696_369602

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (x^2 - 2*x) * Real.log x + a*x^2 + 2

noncomputable def g (a : ℝ) (x : ℝ) : ℝ := f a x - x - 2

theorem tangent_line_and_max_value :
  (∀ x : ℝ, x > 0 → (3*x + f (-1) x - 4 = 0 ↔ x = 1)) ∧
  (∀ a : ℝ, a > 0 →
    (∃! x : ℝ, g a x = 0) →
    (∀ x : ℝ, Real.exp (-2) < x → x < Real.exp 1 → g a x ≤ 2 * Real.exp 2 - 3 * Real.exp 1) ∧
    (∃ x : ℝ, Real.exp (-2) < x ∧ x < Real.exp 1 ∧ g a x = 2 * Real.exp 2 - 3 * Real.exp 1)) :=
by sorry

end NUMINAMATH_CALUDE_tangent_line_and_max_value_l3696_369602


namespace NUMINAMATH_CALUDE_right_triangle_leg_length_l3696_369600

theorem right_triangle_leg_length (c a b : ℝ) : 
  c = 10 →  -- hypotenuse length
  a = 6 →   -- length of one leg
  c^2 = a^2 + b^2 →  -- Pythagorean theorem (right-angled triangle condition)
  b = 8 := by  -- length of the other leg
sorry

end NUMINAMATH_CALUDE_right_triangle_leg_length_l3696_369600


namespace NUMINAMATH_CALUDE_row_5_seat_4_denotation_l3696_369688

/-- Represents a seat in a theater -/
structure Seat where
  row : ℕ
  number : ℕ

/-- Converts a seat to its denotation as an ordered pair -/
def seat_denotation (s : Seat) : ℕ × ℕ := (s.row, s.number)

/-- Given condition: "Row 4, Seat 5" is denoted as (4, 5) -/
axiom example_seat : seat_denotation ⟨4, 5⟩ = (4, 5)

/-- Theorem: The denotation of "Row 5, Seat 4" is (5, 4) -/
theorem row_5_seat_4_denotation : seat_denotation ⟨5, 4⟩ = (5, 4) := by
  sorry

end NUMINAMATH_CALUDE_row_5_seat_4_denotation_l3696_369688


namespace NUMINAMATH_CALUDE_litter_patrol_problem_l3696_369658

/-- The Litter Patrol Problem -/
theorem litter_patrol_problem (total_litter : ℕ) (aluminum_cans : ℕ) (glass_bottles : ℕ) :
  total_litter = 18 →
  aluminum_cans = 8 →
  total_litter = aluminum_cans + glass_bottles →
  glass_bottles = 10 := by
sorry

end NUMINAMATH_CALUDE_litter_patrol_problem_l3696_369658


namespace NUMINAMATH_CALUDE_flag_arrangement_theorem_l3696_369689

/-- The number of distinguishable flagpoles -/
def num_poles : ℕ := 2

/-- The total number of flags -/
def total_flags : ℕ := 25

/-- The number of blue flags -/
def blue_flags : ℕ := 15

/-- The number of green flags -/
def green_flags : ℕ := 10

/-- Function to calculate the number of distinguishable arrangements -/
def calculate_arrangements (np gf bf : ℕ) : ℕ := sorry

/-- Theorem stating that the number of distinguishable arrangements,
    when divided by 1000, yields a remainder of 122 -/
theorem flag_arrangement_theorem :
  calculate_arrangements num_poles green_flags blue_flags % 1000 = 122 := by sorry

end NUMINAMATH_CALUDE_flag_arrangement_theorem_l3696_369689


namespace NUMINAMATH_CALUDE_roots_sum_and_product_l3696_369603

theorem roots_sum_and_product (c d : ℝ) : 
  c^2 - 6*c + 8 = 0 → d^2 - 6*d + 8 = 0 → c^3 + c^4*d^2 + c^2*d^4 + d^3 = 1352 := by
  sorry

end NUMINAMATH_CALUDE_roots_sum_and_product_l3696_369603


namespace NUMINAMATH_CALUDE_marshmallow_roasting_l3696_369616

/-- The number of marshmallows Joe's dad has -/
def dads_marshmallows : ℕ := 21

/-- The number of marshmallows Joe has -/
def joes_marshmallows : ℕ := 4 * dads_marshmallows

/-- The number of marshmallows Joe's dad roasts -/
def dads_roasted : ℕ := dads_marshmallows / 3

/-- The number of marshmallows Joe roasts -/
def joes_roasted : ℕ := joes_marshmallows / 2

/-- The total number of marshmallows roasted by Joe and his dad -/
def total_roasted : ℕ := dads_roasted + joes_roasted

theorem marshmallow_roasting :
  total_roasted = 49 := by
  sorry

end NUMINAMATH_CALUDE_marshmallow_roasting_l3696_369616


namespace NUMINAMATH_CALUDE_problems_per_page_l3696_369677

theorem problems_per_page 
  (total_problems : ℕ) 
  (finished_problems : ℕ) 
  (remaining_pages : ℕ) 
  (h1 : total_problems = 110) 
  (h2 : finished_problems = 47) 
  (h3 : remaining_pages = 7) 
  (h4 : finished_problems < total_problems) : 
  (total_problems - finished_problems) / remaining_pages = 9 := by
  sorry

end NUMINAMATH_CALUDE_problems_per_page_l3696_369677


namespace NUMINAMATH_CALUDE_max_candy_leftover_l3696_369678

theorem max_candy_leftover (x : ℕ) : ∃ (q r : ℕ), x = 7 * q + r ∧ r < 7 ∧ r ≤ 6 :=
by sorry

end NUMINAMATH_CALUDE_max_candy_leftover_l3696_369678


namespace NUMINAMATH_CALUDE_max_a_is_eight_l3696_369645

/-- The quadratic polynomial f(x) = ax^2 - ax + 1 -/
def f (a : ℝ) (x : ℝ) : ℝ := a * x^2 - a * x + 1

/-- The condition that |f(x)| ≤ 1 for all x in [0, 1] -/
def condition (a : ℝ) : Prop :=
  ∀ x : ℝ, x ∈ Set.Icc 0 1 → |f a x| ≤ 1

/-- The maximum value of a is 8 -/
theorem max_a_is_eight :
  (∃ a : ℝ, condition a) →
  (∀ a : ℝ, condition a → a ≤ 8) ∧
  condition 8 :=
sorry

end NUMINAMATH_CALUDE_max_a_is_eight_l3696_369645


namespace NUMINAMATH_CALUDE_imaginary_part_of_complex_fraction_l3696_369614

theorem imaginary_part_of_complex_fraction : 
  let z : ℂ := (15 * Complex.I) / (3 + 4 * Complex.I)
  Complex.im z = 9 / 5 := by sorry

end NUMINAMATH_CALUDE_imaginary_part_of_complex_fraction_l3696_369614


namespace NUMINAMATH_CALUDE_room_freezer_temp_difference_l3696_369653

/-- The temperature difference between room and freezer --/
def temperature_difference (room_temp freezer_temp : ℤ) : ℤ :=
  room_temp - freezer_temp

/-- Theorem stating the temperature difference between room and freezer --/
theorem room_freezer_temp_difference :
  temperature_difference 10 (-6) = 16 := by
  sorry

end NUMINAMATH_CALUDE_room_freezer_temp_difference_l3696_369653


namespace NUMINAMATH_CALUDE_car_fuel_efficiency_l3696_369611

def distance : ℝ := 120
def gasoline : ℝ := 6

theorem car_fuel_efficiency : distance / gasoline = 20 := by
  sorry

end NUMINAMATH_CALUDE_car_fuel_efficiency_l3696_369611


namespace NUMINAMATH_CALUDE_eliminate_first_power_term_l3696_369692

theorem eliminate_first_power_term (a m : ℝ) : 
  (∀ k, (a + m) * (a + 1/2) = k * a^2 + c) ↔ m = -1/2 := by sorry

end NUMINAMATH_CALUDE_eliminate_first_power_term_l3696_369692


namespace NUMINAMATH_CALUDE_solve_equation_l3696_369673

theorem solve_equation : ∃ x : ℝ, 3 * x = (62 - x) + 26 ∧ x = 22 := by
  sorry

end NUMINAMATH_CALUDE_solve_equation_l3696_369673


namespace NUMINAMATH_CALUDE_isosceles_obtuse_triangle_smallest_angle_l3696_369695

/-- 
Given an isosceles, obtuse triangle where the largest angle is 20% larger than 60 degrees,
prove that the measure of each of the two smallest angles is 54 degrees.
-/
theorem isosceles_obtuse_triangle_smallest_angle 
  (α β γ : ℝ) 
  (isosceles : α = β)
  (obtuse : γ > 90)
  (largest_angle : γ = 60 * (1 + 0.2))
  (angle_sum : α + β + γ = 180) :
  α = 54 := by
sorry

end NUMINAMATH_CALUDE_isosceles_obtuse_triangle_smallest_angle_l3696_369695


namespace NUMINAMATH_CALUDE_election_votes_count_l3696_369613

theorem election_votes_count :
  ∀ (total_votes : ℕ) (losing_candidate_votes winning_candidate_votes : ℕ),
    losing_candidate_votes = (35 * total_votes) / 100 →
    winning_candidate_votes = losing_candidate_votes + 2370 →
    losing_candidate_votes + winning_candidate_votes = total_votes →
    total_votes = 7900 :=
by
  sorry

end NUMINAMATH_CALUDE_election_votes_count_l3696_369613


namespace NUMINAMATH_CALUDE_winnie_balloons_l3696_369687

/-- The number of balloons Winnie keeps for herself -/
def balloons_kept (total_balloons : ℕ) (num_friends : ℕ) : ℕ :=
  total_balloons % num_friends

theorem winnie_balloons :
  let red_balloons : ℕ := 15
  let blue_balloons : ℕ := 42
  let yellow_balloons : ℕ := 54
  let purple_balloons : ℕ := 92
  let total_balloons : ℕ := red_balloons + blue_balloons + yellow_balloons + purple_balloons
  let num_friends : ℕ := 11
  balloons_kept total_balloons num_friends = 5 :=
by sorry

end NUMINAMATH_CALUDE_winnie_balloons_l3696_369687


namespace NUMINAMATH_CALUDE_sawyer_octopus_count_l3696_369680

-- Define the number of legs Sawyer saw
def total_legs : ℕ := 40

-- Define the number of legs each octopus has
def legs_per_octopus : ℕ := 8

-- Theorem statement
theorem sawyer_octopus_count :
  total_legs / legs_per_octopus = 5 := by
  sorry

end NUMINAMATH_CALUDE_sawyer_octopus_count_l3696_369680


namespace NUMINAMATH_CALUDE_min_side_length_of_A_l3696_369676

-- Define the squares
structure Square where
  sideLength : ℕ

-- Define the configuration
structure SquareConfiguration where
  A : Square
  B : Square
  C : Square
  D : Square
  vertexCondition : A.sideLength = B.sideLength + C.sideLength + D.sideLength
  areaCondition : A.sideLength^2 / 2 = B.sideLength^2 + C.sideLength^2 + D.sideLength^2

-- Theorem statement
theorem min_side_length_of_A (config : SquareConfiguration) :
  config.A.sideLength ≥ 3 :=
sorry

end NUMINAMATH_CALUDE_min_side_length_of_A_l3696_369676


namespace NUMINAMATH_CALUDE_prob_end_after_two_draw_prob_exactly_two_white_l3696_369604

/-- Represents the color of a ping-pong ball -/
inductive BallColor
  | Red
  | White
  | Blue

/-- Represents the box of ping-pong balls -/
structure Box where
  total : Nat
  red : Nat
  white : Nat
  blue : Nat

/-- The probability of drawing a specific color ball from the box -/
def drawProbability (box : Box) (color : BallColor) : Rat :=
  match color with
  | BallColor.Red => box.red / box.total
  | BallColor.White => box.white / box.total
  | BallColor.Blue => box.blue / box.total

/-- The box configuration as per the problem -/
def problemBox : Box := {
  total := 10
  red := 5
  white := 3
  blue := 2
}

/-- The probability of the process ending after two draws -/
def probEndAfterTwoDraw (box : Box) : Rat :=
  (1 - drawProbability box BallColor.Blue) * drawProbability box BallColor.Blue

/-- The probability of exactly drawing 2 white balls -/
def probExactlyTwoWhite (box : Box) : Rat :=
  3 * drawProbability box BallColor.Red * (drawProbability box BallColor.White)^2 +
  drawProbability box BallColor.White * drawProbability box BallColor.White * drawProbability box BallColor.Blue

theorem prob_end_after_two_draw :
  probEndAfterTwoDraw problemBox = 4 / 25 := by sorry

theorem prob_exactly_two_white :
  probExactlyTwoWhite problemBox = 153 / 1000 := by sorry

end NUMINAMATH_CALUDE_prob_end_after_two_draw_prob_exactly_two_white_l3696_369604


namespace NUMINAMATH_CALUDE_arithmetic_sequence_property_l3696_369643

/-- An arithmetic sequence with the given property -/
def ArithmeticSequence (a : ℕ → ℝ) : Prop :=
  ∃ (a₁ d : ℝ), ∀ n, a n = a₁ + (n - 1) * d

theorem arithmetic_sequence_property (a : ℕ → ℝ) 
    (h : ArithmeticSequence a) 
    (h_sum : a 3 + a 9 + a 27 = 12) : 
    a 13 = 4 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_property_l3696_369643


namespace NUMINAMATH_CALUDE_annie_candy_cost_l3696_369622

/-- Represents the cost of candies for Annie's class --/
def total_cost (candy_a_cost candy_b_cost candy_c_cost : ℚ) 
               (candy_a_count candy_b_count candy_c_count : ℕ) 
               (class_size : ℕ) : ℚ :=
  class_size * (candy_a_cost * candy_a_count + 
                candy_b_cost * candy_b_count + 
                candy_c_cost * candy_c_count)

/-- Theorem stating that Annie's total candy cost is $28 --/
theorem annie_candy_cost : 
  total_cost 0.1 0.15 0.2 3 2 1 35 = 28 := by
  sorry

end NUMINAMATH_CALUDE_annie_candy_cost_l3696_369622


namespace NUMINAMATH_CALUDE_archie_antibiotics_duration_l3696_369681

/-- Calculates the number of days Archie can take antibiotics given the cost, 
    daily frequency, and available money. -/
def daysOfAntibiotics (costPerDose : ℚ) (dosesPerDay : ℕ) (availableMoney : ℚ) : ℚ :=
  availableMoney / (costPerDose * dosesPerDay)

/-- Proves that Archie can take antibiotics for 7 days given the specified conditions. -/
theorem archie_antibiotics_duration :
  daysOfAntibiotics 3 3 63 = 7 := by
sorry

end NUMINAMATH_CALUDE_archie_antibiotics_duration_l3696_369681


namespace NUMINAMATH_CALUDE_platform_length_l3696_369621

/-- Given a train of length l traveling at constant velocity, if it passes a pole in t seconds
    and a platform in 6t seconds, then the length of the platform is 5l. -/
theorem platform_length (l t : ℝ) (h1 : l > 0) (h2 : t > 0) : 
  (∃ v : ℝ, v > 0 ∧ v = l / t ∧ v = (l + 5 * l) / (6 * t)) := by
  sorry

end NUMINAMATH_CALUDE_platform_length_l3696_369621


namespace NUMINAMATH_CALUDE_simplify_fraction_difference_quotient_l3696_369637

theorem simplify_fraction_difference_quotient (a : ℝ) (h1 : a ≠ 2) (h2 : a ≠ -2) :
  (1 / (a + 2) - 1 / (a - 2)) / (1 / (a - 2)) = -4 / (a + 2) := by
  sorry

end NUMINAMATH_CALUDE_simplify_fraction_difference_quotient_l3696_369637


namespace NUMINAMATH_CALUDE_fraction_equivalence_l3696_369625

theorem fraction_equivalence (a c : ℝ) (ha : a ≠ 0) (hc : c ≠ 0) :
  (∀ x y : ℝ, (x + a) / (y + c) = a / c) ↔ (∀ x y : ℝ, x = (a / c) * y) :=
sorry

end NUMINAMATH_CALUDE_fraction_equivalence_l3696_369625


namespace NUMINAMATH_CALUDE_mn_positive_necessary_not_sufficient_l3696_369624

/-- Predicate defining when a curve is an ellipse -/
def is_ellipse (m n : ℝ) : Prop :=
  m > 0 ∧ n > 0 ∧ m ≠ n

/-- The equation of the curve -/
def curve_equation (m n : ℝ) (x y : ℝ) : Prop :=
  m * x^2 + n * y^2 = 1

theorem mn_positive_necessary_not_sufficient (m n : ℝ) :
  (is_ellipse m n → m * n > 0) ∧
  ¬(m * n > 0 → is_ellipse m n) :=
sorry

end NUMINAMATH_CALUDE_mn_positive_necessary_not_sufficient_l3696_369624


namespace NUMINAMATH_CALUDE_banana_theorem_l3696_369693

def banana_problem (initial_bananas final_bananas : ℕ) : Prop :=
  final_bananas - initial_bananas = 7

theorem banana_theorem : banana_problem 2 9 := by
  sorry

end NUMINAMATH_CALUDE_banana_theorem_l3696_369693


namespace NUMINAMATH_CALUDE_total_cost_calculation_l3696_369627

def vegetable_price : ℝ := 2
def beef_price_multiplier : ℝ := 3
def vegetable_weight : ℝ := 6
def beef_weight : ℝ := 4

theorem total_cost_calculation : 
  (vegetable_price * vegetable_weight) + (vegetable_price * beef_price_multiplier * beef_weight) = 36 := by
  sorry

end NUMINAMATH_CALUDE_total_cost_calculation_l3696_369627


namespace NUMINAMATH_CALUDE_intersection_with_complement_l3696_369660

def U : Finset Nat := {1, 2, 3, 4}
def A : Finset Nat := {1, 4}
def B : Finset Nat := {2, 4}

theorem intersection_with_complement : A ∩ (U \ B) = {1} := by sorry

end NUMINAMATH_CALUDE_intersection_with_complement_l3696_369660


namespace NUMINAMATH_CALUDE_square_perimeter_equals_circle_area_l3696_369667

theorem square_perimeter_equals_circle_area (r : ℝ) : r = 8 / Real.pi :=
  -- Define the perimeter of the square
  let square_perimeter := 8 * r
  -- Define the area of the circle
  let circle_area := Real.pi * r^2
  -- State that the perimeter of the square equals the area of the circle
  have h : square_perimeter = circle_area := by sorry
  -- Prove that r = 8 / π
  sorry

end NUMINAMATH_CALUDE_square_perimeter_equals_circle_area_l3696_369667


namespace NUMINAMATH_CALUDE_cut_scene_length_is_8_minutes_l3696_369634

/-- Calculates the length of a cut scene given the original and final movie lengths -/
def cut_scene_length (original_length final_length : ℕ) : ℕ :=
  original_length - final_length

theorem cut_scene_length_is_8_minutes :
  cut_scene_length 60 52 = 8 := by
  sorry

end NUMINAMATH_CALUDE_cut_scene_length_is_8_minutes_l3696_369634


namespace NUMINAMATH_CALUDE_exists_counterexample_for_option_c_l3696_369672

theorem exists_counterexample_for_option_c (h : ∃ a b : ℝ, a > b ∧ b > 0) :
  ∃ a b : ℝ, a > b ∧ b > 0 ∧ ¬(a > Real.sqrt b) :=
sorry

end NUMINAMATH_CALUDE_exists_counterexample_for_option_c_l3696_369672


namespace NUMINAMATH_CALUDE_min_draw_count_correct_l3696_369679

/-- Represents the number of balls of each color in the box -/
structure BallCounts where
  red : Nat
  green : Nat
  yellow : Nat
  blue : Nat
  white : Nat
  black : Nat

/-- The minimum number of balls to draw to guarantee the condition -/
def minDrawCount : Nat := 82

/-- The theorem stating the minimum number of balls to draw -/
theorem min_draw_count_correct (box : BallCounts) 
  (h1 : box.red = 30)
  (h2 : box.green = 22)
  (h3 : box.yellow = 20)
  (h4 : box.blue = 15)
  (h5 : box.white = 12)
  (h6 : box.black = 10) :
  minDrawCount = 82 ∧ 
  (∀ n : Nat, n < 82 → 
    ∃ draw : BallCounts, 
      draw.red + draw.green + draw.yellow + draw.blue + draw.white + draw.black = n ∧
      draw.red ≤ box.red ∧ draw.green ≤ box.green ∧ draw.yellow ≤ box.yellow ∧ 
      draw.blue ≤ box.blue ∧ draw.white ≤ box.white ∧ draw.black ≤ box.black ∧
      draw.white ≤ 12 ∧
      draw.red < 16 ∧ draw.green < 16 ∧ draw.yellow < 16 ∧ draw.blue < 16 ∧ draw.black < 16) ∧
  (∃ draw : BallCounts,
    draw.red + draw.green + draw.yellow + draw.blue + draw.white + draw.black = 82 ∧
    draw.red ≤ box.red ∧ draw.green ≤ box.green ∧ draw.yellow ≤ box.yellow ∧ 
    draw.blue ≤ box.blue ∧ draw.white ≤ box.white ∧ draw.black ≤ box.black ∧
    draw.white ≤ 12 ∧
    (draw.red ≥ 16 ∨ draw.green ≥ 16 ∨ draw.yellow ≥ 16 ∨ draw.blue ≥ 16 ∨ draw.black ≥ 16)) :=
by sorry

end NUMINAMATH_CALUDE_min_draw_count_correct_l3696_369679


namespace NUMINAMATH_CALUDE_error_permutations_l3696_369684

/-- The number of incorrect permutations of the letters in "error" -/
def incorrect_permutations : ℕ :=
  Nat.factorial 5 / Nat.factorial 3 - 1

/-- The word "error" has 5 letters -/
def word_length : ℕ := 5

/-- The letter 'r' is repeated three times -/
def r_count : ℕ := 3

/-- The letters 'e' and 'o' appear once each -/
def unique_letters : ℕ := 2

theorem error_permutations :
  incorrect_permutations = (Nat.factorial word_length / Nat.factorial r_count) - 1 :=
by sorry

end NUMINAMATH_CALUDE_error_permutations_l3696_369684


namespace NUMINAMATH_CALUDE_weight_replacement_l3696_369605

theorem weight_replacement (initial_count : ℕ) (avg_increase : ℝ) (replaced_weight : ℝ) :
  initial_count = 8 →
  avg_increase = 1.5 →
  replaced_weight = 65 →
  (initial_count : ℝ) * avg_increase + replaced_weight = 77 := by
  sorry

end NUMINAMATH_CALUDE_weight_replacement_l3696_369605


namespace NUMINAMATH_CALUDE_unique_prime_satisfying_conditions_l3696_369664

theorem unique_prime_satisfying_conditions : 
  ∃! p : ℕ, 
    Nat.Prime p ∧ 
    ∃ x y : ℕ, 
      x > 0 ∧ 
      y > 0 ∧ 
      p - 1 = 2 * x^2 ∧ 
      p^2 - 1 = 2 * y^2 := by
  sorry

end NUMINAMATH_CALUDE_unique_prime_satisfying_conditions_l3696_369664


namespace NUMINAMATH_CALUDE_set_operations_l3696_369674

def A : Set ℝ := {x | 3 ≤ x ∧ x < 7}
def B : Set ℝ := {x | 5 < x ∧ x < 10}

theorem set_operations :
  (Set.compl (A ∪ B) = {x | x < 3 ∨ x ≥ 10}) ∧
  (Set.compl (A ∩ B) = {x | x ≤ 5 ∨ x ≥ 7}) ∧
  ((Set.compl A) ∩ B = {x | 7 ≤ x ∧ x < 10}) ∧
  (A ∪ (Set.compl B) = {x | x < 7 ∨ x ≥ 10}) :=
by sorry

end NUMINAMATH_CALUDE_set_operations_l3696_369674


namespace NUMINAMATH_CALUDE_sam_puppies_count_l3696_369606

/-- The number of puppies Sam originally had with spots -/
def original_puppies : ℕ := 8

/-- The number of puppies Sam gave to his friends -/
def given_away_puppies : ℕ := 2

/-- The number of puppies Sam has now -/
def remaining_puppies : ℕ := original_puppies - given_away_puppies

theorem sam_puppies_count : remaining_puppies = 6 := by
  sorry

end NUMINAMATH_CALUDE_sam_puppies_count_l3696_369606


namespace NUMINAMATH_CALUDE_smallest_solution_l3696_369668

-- Define the equation
def equation (x : ℝ) : Prop := x * (abs x) + 3 * x = 5 * x + 2

-- Define the solution set
def solution_set : Set ℝ := {x | equation x}

-- State the theorem
theorem smallest_solution :
  ∃ (x : ℝ), x ∈ solution_set ∧ ∀ (y : ℝ), y ∈ solution_set → x ≤ y ∧ x = -1 - Real.sqrt 3 :=
sorry

end NUMINAMATH_CALUDE_smallest_solution_l3696_369668


namespace NUMINAMATH_CALUDE_vector_not_parallel_implies_m_l3696_369640

/-- Two vectors are parallel if and only if their cross product is zero -/
def are_parallel (a b : ℝ × ℝ) : Prop :=
  a.1 * b.2 = a.2 * b.1

theorem vector_not_parallel_implies_m (m : ℝ) :
  let a : ℝ × ℝ := (m, 4)
  let b : ℝ × ℝ := (3, -2)
  ¬(are_parallel a b) → m = -6 := by
  sorry

end NUMINAMATH_CALUDE_vector_not_parallel_implies_m_l3696_369640


namespace NUMINAMATH_CALUDE_ceiling_floor_difference_l3696_369656

theorem ceiling_floor_difference : 
  ⌈(15 : ℝ) / 8 * (-34 : ℝ) / 4⌉ - ⌊(15 : ℝ) / 8 * ⌊(-34 : ℝ) / 4⌋⌋ = 2 :=
by sorry

end NUMINAMATH_CALUDE_ceiling_floor_difference_l3696_369656


namespace NUMINAMATH_CALUDE_fourth_square_area_l3696_369617

-- Define the triangles and their properties
def triangle_ABC (AB BC AC : ℝ) : Prop :=
  AB^2 + BC^2 = AC^2 ∧ AB^2 = 25 ∧ BC^2 = 49

def triangle_ACD (AC CD AD : ℝ) : Prop :=
  AC^2 + CD^2 = AD^2 ∧ CD^2 = 64

-- Theorem statement
theorem fourth_square_area 
  (AB BC AC CD AD : ℝ) 
  (h1 : triangle_ABC AB BC AC) 
  (h2 : triangle_ACD AC CD AD) :
  AD^2 = 138 := by sorry

end NUMINAMATH_CALUDE_fourth_square_area_l3696_369617


namespace NUMINAMATH_CALUDE_sum_of_two_smallest_prime_factors_of_280_l3696_369619

theorem sum_of_two_smallest_prime_factors_of_280 : 
  ∃ (p q : Nat), Nat.Prime p ∧ Nat.Prime q ∧ p < q ∧ 
  p ∣ 280 ∧ q ∣ 280 ∧
  (∀ (r : Nat), Nat.Prime r → r ∣ 280 → r = p ∨ r ≥ q) ∧
  p + q = 7 :=
by sorry

end NUMINAMATH_CALUDE_sum_of_two_smallest_prime_factors_of_280_l3696_369619


namespace NUMINAMATH_CALUDE_second_number_calculation_l3696_369697

theorem second_number_calculation (A B : ℝ) : 
  A = 6400 → 
  0.05 * A = 0.20 * B + 190 → 
  B = 650 := by
sorry

end NUMINAMATH_CALUDE_second_number_calculation_l3696_369697


namespace NUMINAMATH_CALUDE_complex_modulus_problem_l3696_369631

theorem complex_modulus_problem (z : ℂ) : z = 3 + (3 + 4*I) / (4 - 3*I) → Complex.abs z = Real.sqrt 10 := by
  sorry

end NUMINAMATH_CALUDE_complex_modulus_problem_l3696_369631


namespace NUMINAMATH_CALUDE_certain_number_proof_l3696_369663

theorem certain_number_proof (w : ℝ) (x : ℝ) 
  (h1 : x = 13 * w / (1 - w)) 
  (h2 : w^2 = 1) : 
  x = -13/2 := by
  sorry

end NUMINAMATH_CALUDE_certain_number_proof_l3696_369663


namespace NUMINAMATH_CALUDE_dara_half_jane_age_l3696_369638

/-- The problem statement about Dara and Jane's ages -/
theorem dara_half_jane_age :
  let min_age : ℕ := 25  -- Minimum age for employment
  let jane_age : ℕ := 28  -- Jane's current age
  let years_to_min : ℕ := 14  -- Years until Dara reaches minimum age
  let dara_age : ℕ := min_age - years_to_min  -- Dara's current age
  let x : ℕ := 6  -- Years until Dara is half Jane's age
  dara_age + x = (jane_age + x) / 2 := by sorry

end NUMINAMATH_CALUDE_dara_half_jane_age_l3696_369638


namespace NUMINAMATH_CALUDE_hiking_equipment_cost_l3696_369632

def hoodie_cost : ℝ := 80
def flashlight_cost : ℝ := 0.2 * hoodie_cost
def boots_original_cost : ℝ := 110
def boots_discount : ℝ := 0.1
def water_filter_cost : ℝ := 65
def water_filter_discount : ℝ := 0.25
def camping_mat_cost : ℝ := 45
def camping_mat_discount : ℝ := 0.15
def backpack_cost : ℝ := 105

def clothing_tax_rate : ℝ := 0.05
def electronics_tax_rate : ℝ := 0.1
def other_equipment_tax_rate : ℝ := 0.08

def total_cost : ℝ :=
  (hoodie_cost * (1 + clothing_tax_rate)) +
  (flashlight_cost * (1 + electronics_tax_rate)) +
  (boots_original_cost * (1 - boots_discount) * (1 + clothing_tax_rate)) +
  (water_filter_cost * (1 - water_filter_discount) * (1 + other_equipment_tax_rate)) +
  (camping_mat_cost * (1 - camping_mat_discount) * (1 + other_equipment_tax_rate)) +
  (backpack_cost * (1 + other_equipment_tax_rate))

theorem hiking_equipment_cost : total_cost = 413.91 := by
  sorry

end NUMINAMATH_CALUDE_hiking_equipment_cost_l3696_369632


namespace NUMINAMATH_CALUDE_quadratic_real_root_condition_l3696_369665

/-- A quadratic equation x^2 + bx + 25 = 0 has at least one real root if and only if b ∈ (-∞, -10] ∪ [10, ∞) -/
theorem quadratic_real_root_condition (b : ℝ) :
  (∃ x : ℝ, x^2 + b*x + 25 = 0) ↔ b ≤ -10 ∨ b ≥ 10 := by sorry

end NUMINAMATH_CALUDE_quadratic_real_root_condition_l3696_369665


namespace NUMINAMATH_CALUDE_parallel_equal_sides_is_parallelogram_l3696_369657

/-- A quadrilateral in 2D space --/
structure Quadrilateral where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ
  D : ℝ × ℝ

/-- Definition of parallel sides in a quadrilateral --/
def has_parallel_sides (q : Quadrilateral) : Prop :=
  (q.A.1 - q.B.1) / (q.A.2 - q.B.2) = (q.D.1 - q.C.1) / (q.D.2 - q.C.2) ∧
  (q.A.1 - q.D.1) / (q.A.2 - q.D.2) = (q.B.1 - q.C.1) / (q.B.2 - q.C.2)

/-- Definition of equal sides in a quadrilateral --/
def has_equal_sides (q : Quadrilateral) : Prop :=
  (q.A.1 - q.B.1)^2 + (q.A.2 - q.B.2)^2 = (q.B.1 - q.C.1)^2 + (q.B.2 - q.C.2)^2 ∧
  (q.B.1 - q.C.1)^2 + (q.B.2 - q.C.2)^2 = (q.C.1 - q.D.1)^2 + (q.C.2 - q.D.2)^2 ∧
  (q.C.1 - q.D.1)^2 + (q.C.2 - q.D.2)^2 = (q.D.1 - q.A.1)^2 + (q.D.2 - q.A.2)^2

/-- Definition of a parallelogram --/
def is_parallelogram (q : Quadrilateral) : Prop :=
  has_parallel_sides q

/-- Theorem: A quadrilateral with parallel and equal sides is a parallelogram --/
theorem parallel_equal_sides_is_parallelogram (q : Quadrilateral) :
  has_parallel_sides q → has_equal_sides q → is_parallelogram q :=
by
  sorry

end NUMINAMATH_CALUDE_parallel_equal_sides_is_parallelogram_l3696_369657


namespace NUMINAMATH_CALUDE_janinas_daily_rent_l3696_369609

/-- Janina's pancake stand financial model -/
def pancake_stand_model (daily_supply_cost : ℝ) (pancake_price : ℝ) (breakeven_pancakes : ℕ) : ℝ :=
  pancake_price * (breakeven_pancakes : ℝ) - daily_supply_cost

/-- Theorem: Janina's daily rent is $30 -/
theorem janinas_daily_rent :
  pancake_stand_model 12 2 21 = 30 := by
  sorry

end NUMINAMATH_CALUDE_janinas_daily_rent_l3696_369609


namespace NUMINAMATH_CALUDE_line_circle_intersection_range_l3696_369685

/-- Given a line x - 2y + a = 0 and a circle (x-2)^2 + y^2 = 1 with common points,
    the range of values for the real number a is [-2-√5, -2+√5]. -/
theorem line_circle_intersection_range (a : ℝ) : 
  (∃ x y : ℝ, x - 2*y + a = 0 ∧ (x-2)^2 + y^2 = 1) →
  a ∈ Set.Icc (-2 - Real.sqrt 5) (-2 + Real.sqrt 5) :=
sorry

end NUMINAMATH_CALUDE_line_circle_intersection_range_l3696_369685


namespace NUMINAMATH_CALUDE_calculation_proof_l3696_369659

theorem calculation_proof :
  ((-20) - (-18) + 5 + (-9) = -6) ∧
  ((-3) * ((-1)^2003) - ((-4)^2) / (-2) = 11) :=
by sorry

end NUMINAMATH_CALUDE_calculation_proof_l3696_369659


namespace NUMINAMATH_CALUDE_geometric_sequence_first_term_l3696_369620

/-- Given a geometric sequence where the third term is 12 and the fourth term is 16,
    prove that the first term is 27/4. -/
theorem geometric_sequence_first_term
  (a : ℚ) -- First term of the sequence
  (r : ℚ) -- Common ratio of the sequence
  (h1 : a * r^2 = 12) -- Third term is 12
  (h2 : a * r^3 = 16) -- Fourth term is 16
  : a = 27 / 4 :=
by sorry

end NUMINAMATH_CALUDE_geometric_sequence_first_term_l3696_369620


namespace NUMINAMATH_CALUDE_blue_pill_cost_correct_l3696_369612

/-- The cost of a blue pill in dollars -/
def blue_pill_cost : ℝ := 23.50

/-- The cost of a red pill in dollars -/
def red_pill_cost : ℝ := blue_pill_cost - 2

/-- The number of days of medication -/
def days : ℕ := 21

/-- The total cost of medication for the entire period -/
def total_cost : ℝ := 945

theorem blue_pill_cost_correct :
  blue_pill_cost * days + red_pill_cost * days = total_cost :=
by sorry

end NUMINAMATH_CALUDE_blue_pill_cost_correct_l3696_369612


namespace NUMINAMATH_CALUDE_min_value_expression_l3696_369636

theorem min_value_expression (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (∃ (x y : ℝ) (hx : x > 0) (hy : y > 0), 
    (|6*x - 4*y| + |3*(x + y*Real.sqrt 3) + 2*(x*Real.sqrt 3 - y)|) / Real.sqrt (x^2 + y^2) 
    ≤ (|6*a - 4*b| + |3*(a + b*Real.sqrt 3) + 2*(a*Real.sqrt 3 - b)|) / Real.sqrt (a^2 + b^2))
  ∧ 
  (∀ (x y : ℝ) (hx : x > 0) (hy : y > 0), 
    (|6*x - 4*y| + |3*(x + y*Real.sqrt 3) + 2*(x*Real.sqrt 3 - y)|) / Real.sqrt (x^2 + y^2) 
    ≥ Real.sqrt 39) := by
  sorry

end NUMINAMATH_CALUDE_min_value_expression_l3696_369636


namespace NUMINAMATH_CALUDE_acid_dilution_l3696_369629

/-- Given n ounces of n% acid solution, to obtain a (n-20)% solution by adding y ounces of water, 
    where n > 30, y must equal 20n / (n-20). -/
theorem acid_dilution (n : ℝ) (y : ℝ) (h : n > 30) :
  (n * n / 100 = (n - 20) * (n + y) / 100) → y = 20 * n / (n - 20) := by
  sorry

end NUMINAMATH_CALUDE_acid_dilution_l3696_369629


namespace NUMINAMATH_CALUDE_shirt_cost_l3696_369654

theorem shirt_cost (J S X : ℝ) 
  (eq1 : 3 * J + 2 * S = X)
  (eq2 : 2 * J + 3 * S = 66)
  (eq3 : 3 * J + 2 * S = 2 * J + 3 * S) : 
  S = 13.20 := by
  sorry

end NUMINAMATH_CALUDE_shirt_cost_l3696_369654


namespace NUMINAMATH_CALUDE_max_red_squares_is_twelve_l3696_369683

/-- A configuration of colored squares on a 5x5 grid -/
def ColorConfiguration := Fin 5 → Fin 5 → Bool

/-- Checks if four points form an axis-parallel rectangle -/
def isAxisParallelRectangle (p1 p2 p3 p4 : Fin 5 × Fin 5) : Bool :=
  sorry

/-- Checks if a configuration contains an axis-parallel rectangle formed by red squares -/
def containsAxisParallelRectangle (config : ColorConfiguration) : Bool :=
  sorry

/-- Counts the number of red squares in a configuration -/
def countRedSquares (config : ColorConfiguration) : Nat :=
  sorry

/-- The maximum number of red squares possible without forming an axis-parallel rectangle -/
def maxRedSquares : Nat :=
  sorry

theorem max_red_squares_is_twelve :
  maxRedSquares = 12 :=
sorry

end NUMINAMATH_CALUDE_max_red_squares_is_twelve_l3696_369683


namespace NUMINAMATH_CALUDE_calculate_new_interest_rate_l3696_369630

/-- Given a principal amount and interest rates, proves the new interest rate -/
theorem calculate_new_interest_rate
  (P : ℝ)
  (h1 : P * 0.045 = 405)
  (h2 : P * 0.05 = 450) :
  0.05 = (405 + 45) / P :=
by sorry

end NUMINAMATH_CALUDE_calculate_new_interest_rate_l3696_369630


namespace NUMINAMATH_CALUDE_xy_value_l3696_369623

theorem xy_value (x y : ℝ) (h : x * (x + 3 * y) = x^2 + 24) : 3 * x * y = 24 := by
  sorry

end NUMINAMATH_CALUDE_xy_value_l3696_369623


namespace NUMINAMATH_CALUDE_sin_cos_product_trig_expression_value_l3696_369694

-- Part I
theorem sin_cos_product (α : ℝ) 
  (h : (4 * Real.sin α - 2 * Real.cos α) / (5 * Real.cos α + 3 * Real.sin α) = 5 / 7) :
  Real.sin α * Real.cos α = 3 / 10 := by
  sorry

-- Part II
theorem trig_expression_value :
  (Real.sqrt (1 - 2 * Real.sin (10 * π / 180) * Real.cos (10 * π / 180))) / 
  (Real.cos (10 * π / 180) - Real.sqrt (1 - Real.cos (170 * π / 180)^2)) = 1 := by
  sorry

end NUMINAMATH_CALUDE_sin_cos_product_trig_expression_value_l3696_369694


namespace NUMINAMATH_CALUDE_square_area_increase_l3696_369608

theorem square_area_increase (s : ℝ) (h : s > 0) :
  let original_area := s^2
  let new_side := 1.15 * s
  let new_area := new_side^2
  (new_area - original_area) / original_area * 100 = 32.25 := by
  sorry

end NUMINAMATH_CALUDE_square_area_increase_l3696_369608


namespace NUMINAMATH_CALUDE_max_sunny_days_thursday_l3696_369682

/-- Represents the days of the week -/
inductive Day : Type
  | monday
  | tuesday
  | wednesday
  | thursday
  | friday
  | saturday
  | sunday

/-- Represents the weather conditions -/
inductive Weather : Type
  | sunny
  | rainy
  | foggy

/-- The weather pattern for each day of the week -/
def weatherPattern (d : Day) : Weather :=
  match d with
  | Day.monday => Weather.rainy
  | Day.friday => Weather.rainy
  | Day.saturday => Weather.foggy
  | _ => Weather.sunny

/-- Calculates the number of sunny days in a 30-day period starting from a given day -/
def sunnyDaysCount (startDay : Day) : Nat :=
  sorry

/-- Theorem: Starting on Thursday maximizes the number of sunny days in a 30-day period -/
theorem max_sunny_days_thursday :
  ∀ d : Day, sunnyDaysCount Day.thursday ≥ sunnyDaysCount d :=
  sorry

end NUMINAMATH_CALUDE_max_sunny_days_thursday_l3696_369682
