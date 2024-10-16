import Mathlib

namespace NUMINAMATH_CALUDE_sum_of_photo_areas_l1029_102937

-- Define the side lengths of the three square photos
def photo1_side : ℝ := 2
def photo2_side : ℝ := 3
def photo3_side : ℝ := 1

-- Define the function to calculate the area of a square
def square_area (side : ℝ) : ℝ := side * side

-- Theorem: The sum of the areas of the three square photos is 14 square inches
theorem sum_of_photo_areas :
  square_area photo1_side + square_area photo2_side + square_area photo3_side = 14 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_photo_areas_l1029_102937


namespace NUMINAMATH_CALUDE_greater_number_problem_l1029_102925

theorem greater_number_problem (x y : ℝ) : 
  x + y = 40 → x - y = 10 → x > y → x = 25 := by
sorry

end NUMINAMATH_CALUDE_greater_number_problem_l1029_102925


namespace NUMINAMATH_CALUDE_parallel_planes_imply_parallel_lines_l1029_102908

-- Define the types for planes and lines
variable (Plane Line : Type)

-- Define the intersection operation
variable (intersect : Plane → Plane → Line)

-- Define the parallel relation for planes and lines
variable (parallel_planes : Plane → Plane → Prop)
variable (parallel_lines : Line → Line → Prop)

-- State the theorem
theorem parallel_planes_imply_parallel_lines 
  (α β γ : Plane) (m n : Line) :
  α ≠ β → α ≠ γ → β ≠ γ →  -- Three different planes
  intersect α γ = m →      -- α ∩ γ = m
  intersect β γ = n →      -- β ∩ γ = n
  parallel_planes α β →    -- If α ∥ β
  parallel_lines m n :=    -- Then m ∥ n
by sorry

end NUMINAMATH_CALUDE_parallel_planes_imply_parallel_lines_l1029_102908


namespace NUMINAMATH_CALUDE_cos_150_degrees_l1029_102978

theorem cos_150_degrees : Real.cos (150 * Real.pi / 180) = -Real.sqrt 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_cos_150_degrees_l1029_102978


namespace NUMINAMATH_CALUDE_g_of_six_l1029_102988

/-- A function satisfying the given properties -/
def FunctionG (g : ℝ → ℝ) : Prop :=
  (∀ x y : ℝ, g (x + y) = g x + g y) ∧ g 5 = 6

/-- The main theorem -/
theorem g_of_six (g : ℝ → ℝ) (h : FunctionG g) : g 6 = 36/5 := by
  sorry

end NUMINAMATH_CALUDE_g_of_six_l1029_102988


namespace NUMINAMATH_CALUDE_adjacent_fractions_property_l1029_102955

-- Define the type for our rational numbers
def IrreducibleRational := {q : ℚ // q > 0 ∧ Irreducible q ∧ q.num * q.den < 1988}

-- Define the property of being adjacent in the sequence
def Adjacent (q1 q2 : IrreducibleRational) : Prop :=
  q1.val < q2.val ∧ ∀ q : IrreducibleRational, q.val ≤ q1.val ∨ q2.val ≤ q.val

-- State the theorem
theorem adjacent_fractions_property (q1 q2 : IrreducibleRational) 
  (h : Adjacent q1 q2) : 
  q1.val.den * q2.val.num - q1.val.num * q2.val.den = 1 :=
sorry

end NUMINAMATH_CALUDE_adjacent_fractions_property_l1029_102955


namespace NUMINAMATH_CALUDE_quadratic_factorization_l1029_102949

/-- Represents factorization from left to right -/
def is_factorization_left_to_right (f : ℝ → ℝ) (g : ℝ → ℝ → ℝ) : Prop :=
  ∀ x, f x = g (x + 2) (x - 2)

/-- The equation m^2 - 4 = (m + 2)(m - 2) represents factorization from left to right -/
theorem quadratic_factorization :
  is_factorization_left_to_right (λ m => m^2 - 4) (λ a b => a * b) :=
sorry

end NUMINAMATH_CALUDE_quadratic_factorization_l1029_102949


namespace NUMINAMATH_CALUDE_seed_placement_count_l1029_102944

/-- The number of ways to select k items from n distinct items -/
def choose (n k : ℕ) : ℕ := sorry

/-- The number of ways to arrange k items from n distinct items -/
def arrange (n k : ℕ) : ℕ := sorry

/-- The total number of seed placement methods -/
def totalPlacements : ℕ := sorry

theorem seed_placement_count :
  let totalSeeds : ℕ := 10
  let bottleCount : ℕ := 6
  let seedsNotForBottleOne : ℕ := 2
  totalPlacements = choose (totalSeeds - seedsNotForBottleOne) 1 * arrange (totalSeeds - 1) (bottleCount - 1) := by
  sorry

end NUMINAMATH_CALUDE_seed_placement_count_l1029_102944


namespace NUMINAMATH_CALUDE_painted_cube_theorem_l1029_102907

theorem painted_cube_theorem (n : ℕ) (h1 : n > 4) :
  (2 * (n - 2) = n^2 - 2*n + 1) → n = 5 := by
  sorry

end NUMINAMATH_CALUDE_painted_cube_theorem_l1029_102907


namespace NUMINAMATH_CALUDE_teachers_made_28_materials_l1029_102965

/-- Given the number of recycled materials made by a group and the total number of recycled products to be sold, 
    calculate the number of recycled materials made by teachers. -/
def teachers_recycled_materials (group_materials : ℕ) (total_products : ℕ) : ℕ :=
  total_products - group_materials

/-- Theorem: Given that the group made 65 recycled materials and the total number of recycled products
    to be sold is 93, prove that the teachers made 28 recycled materials. -/
theorem teachers_made_28_materials : teachers_recycled_materials 65 93 = 28 := by
  sorry

end NUMINAMATH_CALUDE_teachers_made_28_materials_l1029_102965


namespace NUMINAMATH_CALUDE_polynomial_has_negative_root_l1029_102987

-- Define the polynomial
def P (x : ℝ) : ℝ := x^7 - 2*x^6 - 7*x^4 - x^2 + 10

-- Theorem statement
theorem polynomial_has_negative_root : ∃ x : ℝ, x < 0 ∧ P x = 0 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_has_negative_root_l1029_102987


namespace NUMINAMATH_CALUDE_nickels_count_l1029_102963

/-- Represents the number of coins of each type found by Harriett --/
structure CoinCounts where
  quarters : Nat
  dimes : Nat
  nickels : Nat
  pennies : Nat

/-- Calculates the total value in cents for a given set of coin counts --/
def totalValue (coins : CoinCounts) : Nat :=
  coins.quarters * 25 + coins.dimes * 10 + coins.nickels * 5 + coins.pennies

/-- Theorem stating that given the number of quarters, dimes, pennies, and the total value,
    the number of nickels must be 3 to make the total $3.00 --/
theorem nickels_count (coins : CoinCounts) :
  coins.quarters = 10 ∧ coins.dimes = 3 ∧ coins.pennies = 5 ∧ totalValue coins = 300 →
  coins.nickels = 3 := by
  sorry


end NUMINAMATH_CALUDE_nickels_count_l1029_102963


namespace NUMINAMATH_CALUDE_triangle_cosine_theorem_l1029_102971

theorem triangle_cosine_theorem (a b c : ℝ) (A B C : ℝ) :
  (a > 0) → (b > 0) → (c > 0) →
  (3 * a * Real.cos A = c * Real.cos B + b * Real.cos C) →
  (Real.cos A = 1 / 3) ∧
  (a = 2 * Real.sqrt 3 ∧ Real.cos B + Real.cos C = 2 * Real.sqrt 3 / 3 → c = 3) :=
by sorry

end NUMINAMATH_CALUDE_triangle_cosine_theorem_l1029_102971


namespace NUMINAMATH_CALUDE_unique_intersection_implies_a_equals_three_l1029_102951

noncomputable def f (x : ℝ) : ℝ := x * Real.log x

noncomputable def g (a : ℝ) (x : ℝ) : ℝ := -x^2 + a*x - 2

def intersection_count (f g : ℝ → ℝ) : ℕ := sorry

theorem unique_intersection_implies_a_equals_three :
  ∀ a : ℝ, intersection_count f (g a) = 1 → a = 3 := by sorry

end NUMINAMATH_CALUDE_unique_intersection_implies_a_equals_three_l1029_102951


namespace NUMINAMATH_CALUDE_recurrence_implies_general_formula_l1029_102917

/-- A sequence satisfying the given recurrence relation -/
def RecurrenceSequence (a : ℕ → ℝ) : Prop :=
  a 1 = 1 ∧ ∀ n : ℕ, n > 0 → (n * a n - 2 * a (n + 1)) / a (n + 1) = n

/-- The general formula for the sequence -/
def GeneralFormula (a : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, n > 0 → a n = 2 / (n * (n + 1))

theorem recurrence_implies_general_formula (a : ℕ → ℝ) :
  RecurrenceSequence a → GeneralFormula a := by
  sorry

end NUMINAMATH_CALUDE_recurrence_implies_general_formula_l1029_102917


namespace NUMINAMATH_CALUDE_simplify_expression_l1029_102923

theorem simplify_expression (a b : ℝ) : 
  3*b*(3*b^2 + 2*b) - b^2 + 2*a*(2*a^2 - 3*a) - 4*a*b = 
  9*b^3 + 5*b^2 + 4*a^3 - 6*a^2 - 4*a*b := by sorry

end NUMINAMATH_CALUDE_simplify_expression_l1029_102923


namespace NUMINAMATH_CALUDE_rectangular_prism_diagonal_l1029_102929

theorem rectangular_prism_diagonal : 
  let a : ℝ := 12
  let b : ℝ := 24
  let c : ℝ := 15
  let diagonal := Real.sqrt (a^2 + b^2 + c^2)
  diagonal = 3 * Real.sqrt 105 := by
  sorry

end NUMINAMATH_CALUDE_rectangular_prism_diagonal_l1029_102929


namespace NUMINAMATH_CALUDE_average_marks_combined_classes_l1029_102959

theorem average_marks_combined_classes (n1 n2 : ℕ) (avg1 avg2 : ℚ) :
  n1 = 22 →
  n2 = 28 →
  avg1 = 40 →
  avg2 = 60 →
  (n1 * avg1 + n2 * avg2) / (n1 + n2 : ℚ) = 51.2 := by
  sorry

end NUMINAMATH_CALUDE_average_marks_combined_classes_l1029_102959


namespace NUMINAMATH_CALUDE_box_bottles_count_l1029_102900

-- Define the number of items in a dozen
def dozen : ℕ := 12

-- Define the number of water bottles
def water_bottles : ℕ := 2 * dozen

-- Define the number of additional apple bottles
def additional_apple_bottles : ℕ := dozen / 2

-- Define the total number of apple bottles
def apple_bottles : ℕ := water_bottles + additional_apple_bottles

-- Define the total number of bottles
def total_bottles : ℕ := water_bottles + apple_bottles

-- Theorem statement
theorem box_bottles_count : total_bottles = 54 := by
  sorry

end NUMINAMATH_CALUDE_box_bottles_count_l1029_102900


namespace NUMINAMATH_CALUDE_triangle_properties_l1029_102936

/-- Triangle ABC with sides a, b, c opposite to angles A, B, C respectively -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- The main theorem about the triangle -/
theorem triangle_properties (t : Triangle) 
  (h1 : (t.b^2 + t.c^2 - t.a^2) / Real.cos t.A = 2)
  (h2 : (t.a * Real.cos t.B - t.b * Real.cos t.A) / (t.a * Real.cos t.B + t.b * Real.cos t.A) - t.b / t.c = 1) :
  t.b * t.c = 1 ∧ 
  (1/2 : ℝ) * t.b * t.c * Real.sin t.A = Real.sqrt 3 / 4 := by
  sorry

end NUMINAMATH_CALUDE_triangle_properties_l1029_102936


namespace NUMINAMATH_CALUDE_value_of_expression_l1029_102948

-- Define the polynomial
def p (x h k : ℝ) : ℝ := 5 * x^4 - h * x^2 + k

-- State the theorem
theorem value_of_expression (h k : ℝ) :
  (p 3 h k = 0) → (p (-1) h k = 0) → (p 2 h k = 0) → |5 * h - 4 * k| = 70 := by
  sorry

end NUMINAMATH_CALUDE_value_of_expression_l1029_102948


namespace NUMINAMATH_CALUDE_dogs_not_liking_any_food_l1029_102911

theorem dogs_not_liking_any_food (total : ℕ) (watermelon salmon chicken : ℕ) 
  (watermelon_salmon watermelon_chicken salmon_chicken : ℕ) (all_three : ℕ)
  (h_total : total = 100)
  (h_watermelon : watermelon = 20)
  (h_salmon : salmon = 70)
  (h_chicken : chicken = 10)
  (h_watermelon_salmon : watermelon_salmon = 10)
  (h_salmon_chicken : salmon_chicken = 5)
  (h_watermelon_chicken : watermelon_chicken = 3)
  (h_all_three : all_three = 2) :
  total - (watermelon + salmon + chicken - watermelon_salmon - watermelon_chicken - salmon_chicken + all_three) = 28 := by
  sorry

end NUMINAMATH_CALUDE_dogs_not_liking_any_food_l1029_102911


namespace NUMINAMATH_CALUDE_bracelet_capacity_is_fifteen_l1029_102972

/-- Represents the jewelry store inventory and pricing --/
structure JewelryStore where
  necklace_capacity : ℕ
  current_necklaces : ℕ
  ring_capacity : ℕ
  current_rings : ℕ
  current_bracelets : ℕ
  necklace_price : ℕ
  ring_price : ℕ
  bracelet_price : ℕ
  total_fill_cost : ℕ

/-- Calculates the bracelet display capacity given the store's inventory and pricing --/
def bracelet_display_capacity (store : JewelryStore) : ℕ :=
  store.current_bracelets +
    ((store.total_fill_cost -
      (store.necklace_price * (store.necklace_capacity - store.current_necklaces) +
       store.ring_price * (store.ring_capacity - store.current_rings)))
     / store.bracelet_price)

/-- Theorem stating that for the given store configuration, the bracelet display capacity is 15 --/
theorem bracelet_capacity_is_fifteen :
  let store : JewelryStore := {
    necklace_capacity := 12,
    current_necklaces := 5,
    ring_capacity := 30,
    current_rings := 18,
    current_bracelets := 8,
    necklace_price := 4,
    ring_price := 10,
    bracelet_price := 5,
    total_fill_cost := 183
  }
  bracelet_display_capacity store = 15 := by
  sorry


end NUMINAMATH_CALUDE_bracelet_capacity_is_fifteen_l1029_102972


namespace NUMINAMATH_CALUDE_max_side_length_of_triangle_l1029_102939

theorem max_side_length_of_triangle (a b c : ℕ) : 
  a ≠ b ∧ b ≠ c ∧ a ≠ c →  -- three different side lengths
  a + b + c = 30 →        -- perimeter is 30
  a < 15 ∧ b < 15 ∧ c < 15 →  -- each side is less than 15
  a ≤ 14 ∧ b ≤ 14 ∧ c ≤ 14 →  -- each side is at most 14
  ∃ (x y z : ℕ), x + y + z = 30 ∧ x = 14 ∧ y < x ∧ z < x ∧ y ≠ z →  -- there exists a triangle with max side 14
  (∀ (s : ℕ), s ≤ a ∨ s ≤ b ∨ s ≤ c) →  -- s is not greater than all sides
  14 = max a (max b c)  -- 14 is the maximum side length
  := by sorry

end NUMINAMATH_CALUDE_max_side_length_of_triangle_l1029_102939


namespace NUMINAMATH_CALUDE_library_theorem_l1029_102916

def library_problem (total_books : ℕ) (books_per_student : ℕ) 
  (day1_students : ℕ) (day2_students : ℕ) (day3_students : ℕ) : ℕ :=
  let books_remaining := total_books - 
    (day1_students + day2_students + day3_students) * books_per_student
  books_remaining / books_per_student

theorem library_theorem : 
  library_problem 120 5 4 5 6 = 9 := by
  sorry

end NUMINAMATH_CALUDE_library_theorem_l1029_102916


namespace NUMINAMATH_CALUDE_original_cost_after_discount_l1029_102990

theorem original_cost_after_discount (decreased_cost : ℝ) (discount_rate : ℝ) :
  decreased_cost = 100 ∧ discount_rate = 0.5 → 
  ∃ original_cost : ℝ, original_cost = 200 ∧ decreased_cost = original_cost * (1 - discount_rate) :=
by
  sorry

end NUMINAMATH_CALUDE_original_cost_after_discount_l1029_102990


namespace NUMINAMATH_CALUDE_completing_square_quadratic_l1029_102903

theorem completing_square_quadratic (x : ℝ) :
  (x^2 - 6*x + 8 = 0) ↔ ((x - 3)^2 = 1) := by
  sorry

end NUMINAMATH_CALUDE_completing_square_quadratic_l1029_102903


namespace NUMINAMATH_CALUDE_intersection_M_N_l1029_102926

-- Define the sets M and N
def M : Set ℝ := {y | ∃ x : ℝ, y = x^2 - 4*x + 3}
def N : Set ℝ := {y | ∃ x : ℝ, y = x - 1}

-- Theorem statement
theorem intersection_M_N : M ∩ N = {y | y ≥ -1} := by sorry

end NUMINAMATH_CALUDE_intersection_M_N_l1029_102926


namespace NUMINAMATH_CALUDE_a_subset_M_l1029_102984

noncomputable def a : ℝ := Real.sqrt 3

def M : Set ℝ := {x | x ≤ 3}

theorem a_subset_M : {a} ⊆ M := by sorry

end NUMINAMATH_CALUDE_a_subset_M_l1029_102984


namespace NUMINAMATH_CALUDE_derivative_f_at_zero_l1029_102932

-- Define the function f
noncomputable def f (x : ℝ) : ℝ :=
  if x ≠ 0 then x + Real.arcsin (x^2 * Real.sin (6 / x))
  else 0

-- State the theorem
theorem derivative_f_at_zero :
  deriv f 0 = 1 := by sorry

end NUMINAMATH_CALUDE_derivative_f_at_zero_l1029_102932


namespace NUMINAMATH_CALUDE_convex_polygon_three_obtuse_sides_l1029_102995

/-- A convex polygon with n sides and exactly 3 obtuse angles -/
structure ConvexPolygon (n : ℕ) :=
  (sides : ℕ)
  (is_convex : Bool)
  (obtuse_angles : ℕ)
  (sides_eq : sides = n)
  (convex : is_convex = true)
  (obtuse : obtuse_angles = 3)

/-- The theorem stating that a convex polygon with exactly 3 obtuse angles can only have 5 or 6 sides -/
theorem convex_polygon_three_obtuse_sides (n : ℕ) (p : ConvexPolygon n) : n = 5 ∨ n = 6 :=
sorry

end NUMINAMATH_CALUDE_convex_polygon_three_obtuse_sides_l1029_102995


namespace NUMINAMATH_CALUDE_sqrt_4_minus_abs_sqrt_3_minus_2_plus_neg_1_pow_2023_l1029_102974

theorem sqrt_4_minus_abs_sqrt_3_minus_2_plus_neg_1_pow_2023 :
  Real.sqrt 4 - |Real.sqrt 3 - 2| + (-1)^2023 = Real.sqrt 3 - 1 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_4_minus_abs_sqrt_3_minus_2_plus_neg_1_pow_2023_l1029_102974


namespace NUMINAMATH_CALUDE_least_integer_satisfying_inequality_l1029_102996

theorem least_integer_satisfying_inequality :
  ∀ y : ℤ, (∀ x : ℤ, 3 * |x| - 4 < 20 → y ≤ x) → y = -7 :=
by sorry

end NUMINAMATH_CALUDE_least_integer_satisfying_inequality_l1029_102996


namespace NUMINAMATH_CALUDE_divisibility_of_n_squared_n_squared_minus_one_l1029_102998

theorem divisibility_of_n_squared_n_squared_minus_one (n : ℤ) : 
  12 ∣ n^2 * (n^2 - 1) := by sorry

end NUMINAMATH_CALUDE_divisibility_of_n_squared_n_squared_minus_one_l1029_102998


namespace NUMINAMATH_CALUDE_no_difference_of_primes_in_S_l1029_102958

/-- The set of numbers we're considering -/
def S : Set ℕ := {n : ℕ | ∃ k : ℕ, n = 10 * k + 7}

/-- A function that checks if a natural number is prime -/
def is_prime (n : ℕ) : Prop := Nat.Prime n

/-- A function that checks if a number can be expressed as the difference of two primes -/
def is_difference_of_primes (n : ℕ) : Prop :=
  ∃ p q : ℕ, is_prime p ∧ is_prime q ∧ p - q = n

/-- The main theorem: no number in S can be expressed as the difference of two primes -/
theorem no_difference_of_primes_in_S : ∀ n ∈ S, ¬(is_difference_of_primes n) := by
  sorry

end NUMINAMATH_CALUDE_no_difference_of_primes_in_S_l1029_102958


namespace NUMINAMATH_CALUDE_rhombus_perimeter_l1029_102910

/-- The perimeter of a rhombus with diagonals 24 and 10 is 52 -/
theorem rhombus_perimeter (d1 d2 : ℝ) : 
  d1 = 24 → d2 = 10 → 4 * Real.sqrt ((d1/2)^2 + (d2/2)^2) = 52 := by
  sorry

end NUMINAMATH_CALUDE_rhombus_perimeter_l1029_102910


namespace NUMINAMATH_CALUDE_sector_angle_l1029_102977

/-- Given a sector with radius 1 and area 3π/8, its central angle is 3π/4 -/
theorem sector_angle (r : ℝ) (A : ℝ) (α : ℝ) : 
  r = 1 → A = (3 * π) / 8 → A = (1 / 2) * α * r^2 → α = (3 * π) / 4 := by
  sorry

end NUMINAMATH_CALUDE_sector_angle_l1029_102977


namespace NUMINAMATH_CALUDE_square_difference_equals_720_l1029_102922

theorem square_difference_equals_720 : (30 + 12)^2 - (12^2 + 30^2) = 720 := by
  sorry

end NUMINAMATH_CALUDE_square_difference_equals_720_l1029_102922


namespace NUMINAMATH_CALUDE_book_reading_end_day_l1029_102989

-- Define the days of the week
inductive DayOfWeek
  | Sunday
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday
  | Saturday

def nextDay (d : DayOfWeek) : DayOfWeek :=
  match d with
  | DayOfWeek.Sunday => DayOfWeek.Monday
  | DayOfWeek.Monday => DayOfWeek.Tuesday
  | DayOfWeek.Tuesday => DayOfWeek.Wednesday
  | DayOfWeek.Wednesday => DayOfWeek.Thursday
  | DayOfWeek.Thursday => DayOfWeek.Friday
  | DayOfWeek.Friday => DayOfWeek.Saturday
  | DayOfWeek.Saturday => DayOfWeek.Sunday

def advanceDays (startDay : DayOfWeek) (n : Nat) : DayOfWeek :=
  match n with
  | 0 => startDay
  | n + 1 => nextDay (advanceDays startDay n)

theorem book_reading_end_day :
  let numBooks : Nat := 20
  let startDay := DayOfWeek.Wednesday
  let totalDays := (numBooks * (numBooks + 1)) / 2
  advanceDays startDay totalDays = startDay := by
  sorry


end NUMINAMATH_CALUDE_book_reading_end_day_l1029_102989


namespace NUMINAMATH_CALUDE_coins_value_is_78_percent_of_dollar_l1029_102973

-- Define the value of each coin in cents
def penny_value : ℕ := 1
def nickel_value : ℕ := 5
def dime_value : ℕ := 10
def quarter_value : ℕ := 25

-- Define the number of each coin
def num_pennies : ℕ := 3
def num_nickels : ℕ := 2
def num_dimes : ℕ := 4
def num_quarters : ℕ := 1

-- Define the total value in cents
def total_cents : ℕ := 
  num_pennies * penny_value + 
  num_nickels * nickel_value + 
  num_dimes * dime_value + 
  num_quarters * quarter_value

-- Define one dollar in cents
def dollar_in_cents : ℕ := 100

-- Theorem to prove
theorem coins_value_is_78_percent_of_dollar : 
  (total_cents : ℚ) / (dollar_in_cents : ℚ) = 78 / 100 := by
  sorry

end NUMINAMATH_CALUDE_coins_value_is_78_percent_of_dollar_l1029_102973


namespace NUMINAMATH_CALUDE_union_of_M_and_Q_l1029_102940

def M : Set ℕ := {0, 2, 4, 6}
def Q : Set ℕ := {0, 1, 3, 5}

theorem union_of_M_and_Q : M ∪ Q = {0, 1, 2, 3, 4, 5, 6} := by
  sorry

end NUMINAMATH_CALUDE_union_of_M_and_Q_l1029_102940


namespace NUMINAMATH_CALUDE_arithmetic_sequence_product_l1029_102941

def is_arithmetic_sequence (b : ℕ → ℤ) (d : ℤ) : Prop :=
  ∀ n : ℕ, b (n + 1) = b n + d

theorem arithmetic_sequence_product (b : ℕ → ℤ) (d : ℤ) :
  is_arithmetic_sequence b d →
  d > 0 →
  b 5 * b 6 = 21 →
  b 4 * b 7 = -779 ∨ b 4 * b 7 = -11 :=
by sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_product_l1029_102941


namespace NUMINAMATH_CALUDE_first_customer_payment_l1029_102980

/-- The cost of one MP3 player -/
def mp3_cost : ℕ := sorry

/-- The cost of one set of headphones -/
def headphone_cost : ℕ := 30

/-- The total cost for the second customer -/
def second_customer_total : ℕ := 480

/-- The number of MP3 players bought by the first customer -/
def first_customer_mp3 : ℕ := 5

/-- The number of headphones bought by the first customer -/
def first_customer_headphones : ℕ := 8

/-- The number of MP3 players bought by the second customer -/
def second_customer_mp3 : ℕ := 3

/-- The number of headphones bought by the second customer -/
def second_customer_headphones : ℕ := 4

theorem first_customer_payment :
  second_customer_mp3 * mp3_cost + second_customer_headphones * headphone_cost = second_customer_total →
  first_customer_mp3 * mp3_cost + first_customer_headphones * headphone_cost = 840 :=
by sorry

end NUMINAMATH_CALUDE_first_customer_payment_l1029_102980


namespace NUMINAMATH_CALUDE_triangle_side_length_l1029_102943

theorem triangle_side_length (A B C : ℝ) (a b c : ℝ) :
  -- Conditions
  (b * Real.sin A = 3 * c * Real.sin B) →
  (a = 3) →
  (Real.cos B = 2/3) →
  -- Triangle inequality (to ensure it's a valid triangle)
  (a + b > c) → (b + c > a) → (c + a > b) →
  -- Positive side lengths
  (a > 0) → (b > 0) → (c > 0) →
  -- Conclusion
  b = Real.sqrt 6 := by
sorry

end NUMINAMATH_CALUDE_triangle_side_length_l1029_102943


namespace NUMINAMATH_CALUDE_smallest_possible_FG_l1029_102975

/-- Given two triangles EFG and HFG sharing side FG, prove that the smallest possible
    integer length for FG is 15, given the following side lengths:
    EF = 6, EG = 15, HG = 10, HF = 25 -/
theorem smallest_possible_FG (EF EG HG HF : ℝ) (hEF : EF = 6) (hEG : EG = 15) 
    (hHG : HG = 10) (hHF : HF = 25) : 
  (∀ FG : ℕ, FG > EG - EF ∧ FG > HF - HG) → ∀ FG : ℕ, FG ≥ 15 := by
  sorry

end NUMINAMATH_CALUDE_smallest_possible_FG_l1029_102975


namespace NUMINAMATH_CALUDE_egg_packing_problem_l1029_102909

theorem egg_packing_problem (initial_eggs : Nat) (eggs_per_carton : Nat) (broken_eggs : Nat) :
  initial_eggs = 1000 →
  eggs_per_carton = 12 →
  broken_eggs < 12 →
  ∃ (filled_cartons : Nat), (initial_eggs - broken_eggs) = filled_cartons * eggs_per_carton →
  broken_eggs = 4 := by
  sorry

end NUMINAMATH_CALUDE_egg_packing_problem_l1029_102909


namespace NUMINAMATH_CALUDE_large_coin_equivalent_mass_l1029_102935

theorem large_coin_equivalent_mass (large_coin_mass : ℝ) (pound_coin_mass : ℝ) :
  large_coin_mass = 100000 →
  pound_coin_mass = 10 →
  (large_coin_mass / pound_coin_mass : ℝ) = 10000 := by
  sorry

end NUMINAMATH_CALUDE_large_coin_equivalent_mass_l1029_102935


namespace NUMINAMATH_CALUDE_distance_to_plane_value_l1029_102985

-- Define the sphere and points
def Sphere : Type := ℝ × ℝ × ℝ
def Point : Type := ℝ × ℝ × ℝ

-- Define the center and radius of the sphere
def S : Sphere := sorry
def radius : ℝ := 25

-- Define the points on the sphere
def P : Point := sorry
def Q : Point := sorry
def R : Point := sorry

-- Define the distances between points
def PQ : ℝ := 20
def QR : ℝ := 21
def RP : ℝ := 29

-- Define the distance from S to the plane of triangle PQR
def distance_to_plane : ℝ := sorry

-- State the theorem
theorem distance_to_plane_value : distance_to_plane = (266 : ℝ) * Real.sqrt 154 / 14 := by sorry

end NUMINAMATH_CALUDE_distance_to_plane_value_l1029_102985


namespace NUMINAMATH_CALUDE_stating_chess_tournament_players_l1029_102920

/-- The number of players in a chess tournament -/
def num_players : ℕ := 19

/-- The total number of games played in the tournament -/
def total_games : ℕ := 342

/-- 
Theorem stating that the number of players in the chess tournament is correct,
given the conditions of the problem.
-/
theorem chess_tournament_players :
  2 * num_players * (num_players - 1) = total_games := by
  sorry

end NUMINAMATH_CALUDE_stating_chess_tournament_players_l1029_102920


namespace NUMINAMATH_CALUDE_mike_total_hours_l1029_102953

/-- Calculate the total hours worked given hours per day and number of days -/
def total_hours (hours_per_day : ℕ) (days : ℕ) : ℕ := hours_per_day * days

/-- Proof that Mike worked 15 hours in total -/
theorem mike_total_hours : total_hours 3 5 = 15 := by
  sorry

end NUMINAMATH_CALUDE_mike_total_hours_l1029_102953


namespace NUMINAMATH_CALUDE_only_solutions_all_negative_one_or_all_one_l1029_102933

/-- A sequence of 2016 real numbers satisfying the given equation -/
def SequenceSatisfyingEquation (x : Fin 2016 → ℝ) : Prop :=
  ∀ i : Fin 2016, x i ^ 2 + x i - 1 = x (i.succ)

/-- The theorem stating the only solutions are all -1 or all 1 -/
theorem only_solutions_all_negative_one_or_all_one
  (x : Fin 2016 → ℝ) (h : SequenceSatisfyingEquation x) :
  (∀ i, x i = -1) ∨ (∀ i, x i = 1) := by
  sorry

#check only_solutions_all_negative_one_or_all_one

end NUMINAMATH_CALUDE_only_solutions_all_negative_one_or_all_one_l1029_102933


namespace NUMINAMATH_CALUDE_spadesuit_calculation_l1029_102902

-- Define the spadesuit operation
def spadesuit (a b : ℝ) : ℝ := |a - b|

-- State the theorem
theorem spadesuit_calculation : spadesuit 3 (spadesuit 5 (spadesuit 7 10)) = 1 := by
  sorry

end NUMINAMATH_CALUDE_spadesuit_calculation_l1029_102902


namespace NUMINAMATH_CALUDE_race_catchup_time_l1029_102905

theorem race_catchup_time 
  (cristina_speed : ℝ) 
  (nicky_speed : ℝ) 
  (head_start : ℝ) 
  (h1 : cristina_speed = 6)
  (h2 : nicky_speed = 3)
  (h3 : head_start = 36) :
  ∃ t : ℝ, t = 12 ∧ cristina_speed * t = head_start + nicky_speed * t := by
sorry

end NUMINAMATH_CALUDE_race_catchup_time_l1029_102905


namespace NUMINAMATH_CALUDE_soda_price_l1029_102901

/-- The cost of a burger in cents -/
def burger_cost : ℚ := sorry

/-- The cost of a soda in cents -/
def soda_cost : ℚ := sorry

/-- Uri's purchase: 3 burgers and 1 soda for 360 cents -/
axiom uri_purchase : 3 * burger_cost + soda_cost = 360

/-- Gen's purchase: 1 burger and 3 sodas for 330 cents -/
axiom gen_purchase : burger_cost + 3 * soda_cost = 330

theorem soda_price : soda_cost = 78.75 := by sorry

end NUMINAMATH_CALUDE_soda_price_l1029_102901


namespace NUMINAMATH_CALUDE_own_square_and_cube_root_l1029_102942

theorem own_square_and_cube_root : 
  ∀ x : ℝ, (x^2 = x ∧ x^3 = x) ↔ x = 0 :=
by sorry

end NUMINAMATH_CALUDE_own_square_and_cube_root_l1029_102942


namespace NUMINAMATH_CALUDE_sum_and_equality_problem_l1029_102921

theorem sum_and_equality_problem (a b c : ℚ) :
  a + b + c = 120 ∧ (a + 8 = b - 3) ∧ (b - 3 = 3 * c) →
  b = 56 + 4/7 := by
sorry

end NUMINAMATH_CALUDE_sum_and_equality_problem_l1029_102921


namespace NUMINAMATH_CALUDE_ball_return_to_start_l1029_102994

theorem ball_return_to_start (n : ℕ) (k : ℕ) (h1 : n = 15) (h2 : k = 5) : 
  ∃ m : ℕ, m > 0 ∧ (m * k) % n = 0 ∧ m = 3 :=
sorry

end NUMINAMATH_CALUDE_ball_return_to_start_l1029_102994


namespace NUMINAMATH_CALUDE_limit_log_div_power_l1029_102914

open Real

-- Define the function f(x) = ln(x) / x^α
noncomputable def f (α : ℝ) (x : ℝ) : ℝ := (log x) / (x ^ α)

-- State the theorem
theorem limit_log_div_power (α : ℝ) (h₁ : α > 0) :
  ∀ ε > 0, ∃ N, ∀ x ≥ N, x > 0 → |f α x - 0| < ε :=
sorry

end NUMINAMATH_CALUDE_limit_log_div_power_l1029_102914


namespace NUMINAMATH_CALUDE_equation_simplification_l1029_102997

theorem equation_simplification (x y z : ℕ) (hx : x > 1) (hy : y > 1) (hz : z > 1) :
  9 * x - (10 / (2 * y) / 3 + 7 * z) * Real.pi = 9 * x - (5 * Real.pi / (3 * y)) - (7 * Real.pi * z) :=
by sorry

end NUMINAMATH_CALUDE_equation_simplification_l1029_102997


namespace NUMINAMATH_CALUDE_probability_purple_ten_sided_die_l1029_102981

/-- A die with a specific number of sides and purple faces -/
structure Die :=
  (sides : ℕ)
  (purple_faces : ℕ)

/-- The probability of rolling a purple face on a given die -/
def probability_purple (d : Die) : ℚ :=
  d.purple_faces / d.sides

/-- Theorem: The probability of rolling a purple face on a 10-sided die with 3 purple faces is 3/10 -/
theorem probability_purple_ten_sided_die :
  let d : Die := ⟨10, 3⟩
  probability_purple d = 3 / 10 := by
  sorry

end NUMINAMATH_CALUDE_probability_purple_ten_sided_die_l1029_102981


namespace NUMINAMATH_CALUDE_complex_number_quadrant_l1029_102915

theorem complex_number_quadrant : 
  let z : ℂ := (1/2 : ℂ) + (Complex.I * (Real.sqrt 3 / 2))
  let w : ℂ := z^2
  (w.re < 0) ∧ (w.im > 0) :=
by sorry

end NUMINAMATH_CALUDE_complex_number_quadrant_l1029_102915


namespace NUMINAMATH_CALUDE_quadratic_inequality_l1029_102960

def f (a : ℝ) (x : ℝ) : ℝ := x^2 + 2*a*x + 2

theorem quadratic_inequality (a : ℝ) :
  (∀ x ∈ Set.Icc 1 5, f a x > 3*a*x) ↔ a < 2*Real.sqrt 2 ∧
  (∀ x : ℝ, (a + 1)*x^2 + x > f a x ↔
    (a = 0 ∧ x > 2) ∨
    (a > 0 ∧ (x < -1/a ∨ x > 2)) ∨
    (-1/2 < a ∧ a < 0 ∧ 2 < x ∧ x < -1/a) ∨
    (a < -1/2 ∧ -1/a < x ∧ x < 2)) :=
sorry

end NUMINAMATH_CALUDE_quadratic_inequality_l1029_102960


namespace NUMINAMATH_CALUDE_no_odd_tens_digit_squares_l1029_102966

/-- The set of numbers from 1 to 50 -/
def S : Set Nat := {n | 1 ≤ n ∧ n ≤ 50}

/-- A number ends in 3 or 7 -/
def ends_in_3_or_7 (n : Nat) : Prop := n % 10 = 3 ∨ n % 10 = 7

/-- The tens digit of a number -/
def tens_digit (n : Nat) : Nat := (n / 10) % 10

/-- A number is even -/
def is_even (n : Nat) : Prop := n % 2 = 0

theorem no_odd_tens_digit_squares :
  ∀ n ∈ S, ends_in_3_or_7 n → is_even (tens_digit (n^2)) := by sorry

end NUMINAMATH_CALUDE_no_odd_tens_digit_squares_l1029_102966


namespace NUMINAMATH_CALUDE_basketball_team_squads_l1029_102957

/-- The number of ways to choose k items from n items without replacement and where order doesn't matter. -/
def binomial (n k : ℕ) : ℕ := sorry

/-- The number of different team squads that can be formed from a given number of players,
    selecting a captain and a specified number of additional players. -/
def teamSquads (totalPlayers captains additionalPlayers : ℕ) : ℕ :=
  totalPlayers * binomial (totalPlayers - 1) additionalPlayers

theorem basketball_team_squads :
  teamSquads 12 1 5 = 5544 := by sorry

end NUMINAMATH_CALUDE_basketball_team_squads_l1029_102957


namespace NUMINAMATH_CALUDE_expression_equality_l1029_102934

theorem expression_equality : 
  500 * 987 * 0.0987 * 50 = 2.5 * 987^2 := by sorry

end NUMINAMATH_CALUDE_expression_equality_l1029_102934


namespace NUMINAMATH_CALUDE_min_value_of_f_l1029_102967

-- Define the function f(x)
def f (x : ℝ) : ℝ := x^2 - 4*x + 4

-- Theorem stating that the minimum value of f(x) is 0
theorem min_value_of_f :
  ∃ (x₀ : ℝ), ∀ (x : ℝ), f x ≥ f x₀ ∧ f x₀ = 0 :=
by sorry

end NUMINAMATH_CALUDE_min_value_of_f_l1029_102967


namespace NUMINAMATH_CALUDE_horizontal_axis_independent_l1029_102982

/-- Represents the different types of variables in a graph --/
inductive AxisVariable
  | Dependent
  | Constant
  | Independent
  | Function

/-- Represents a standard graph showing relationships between variables --/
structure StandardGraph where
  horizontalAxis : AxisVariable
  verticalAxis : AxisVariable

/-- Theorem stating that the horizontal axis in a standard graph usually represents the independent variable --/
theorem horizontal_axis_independent (g : StandardGraph) : g.horizontalAxis = AxisVariable.Independent := by
  sorry

end NUMINAMATH_CALUDE_horizontal_axis_independent_l1029_102982


namespace NUMINAMATH_CALUDE_max_value_fraction_l1029_102927

theorem max_value_fraction (x y : ℝ) (hx : -6 ≤ x ∧ x ≤ -3) (hy : 1 ≤ y ∧ y ≤ 5) :
  (∀ a b, -6 ≤ a ∧ a ≤ -3 ∧ 1 ≤ b ∧ b ≤ 5 → (a + b) / a ≤ (x + y) / x) →
  (x + y) / x = 1/6 :=
by sorry

end NUMINAMATH_CALUDE_max_value_fraction_l1029_102927


namespace NUMINAMATH_CALUDE_student_performance_l1029_102969

structure Student :=
  (name : String)
  (scores : Fin 6 → ℝ)

def class_avg : Fin 6 → ℝ
| 0 => 128.2
| 1 => 118.3
| 2 => 125.4
| 3 => 120.3
| 4 => 115.7
| 5 => 122.1

def student_A : Student :=
  ⟨"A", λ i => [138, 127, 131, 132, 128, 135].get i⟩

def student_B : Student :=
  ⟨"B", λ i => [130, 116, 128, 115, 126, 120].get i⟩

def student_C : Student :=
  ⟨"C", λ i => [108, 105, 113, 112, 115, 123].get i⟩

theorem student_performance :
  (∀ i : Fin 6, student_A.scores i > class_avg i) ∧
  (∃ i j : Fin 6, student_B.scores i > class_avg i ∧ student_B.scores j < class_avg j) ∧
  (∃ k : Fin 6, ∀ i j : Fin 6, i < j → j ≥ k →
    (student_C.scores j - class_avg j) > (student_C.scores i - class_avg i)) :=
by sorry

end NUMINAMATH_CALUDE_student_performance_l1029_102969


namespace NUMINAMATH_CALUDE_max_value_trig_expression_l1029_102919

theorem max_value_trig_expression (a b c : ℝ) :
  (∀ θ : ℝ, a * Real.cos θ + b * Real.sin θ + c * Real.cos (2 * θ) ≤ Real.sqrt (a^2 + b^2 + 2 * c^2)) ∧
  (∃ θ : ℝ, a * Real.cos θ + b * Real.sin θ + c * Real.cos (2 * θ) = Real.sqrt (a^2 + b^2 + 2 * c^2)) :=
by sorry

end NUMINAMATH_CALUDE_max_value_trig_expression_l1029_102919


namespace NUMINAMATH_CALUDE_darius_age_is_8_l1029_102947

-- Define the ages of Jenna and Darius
def jenna_age : ℕ := 13
def darius_age : ℕ := 21 - jenna_age

-- Theorem statement
theorem darius_age_is_8 :
  (jenna_age > darius_age) ∧ 
  (jenna_age + darius_age = 21) ∧
  (jenna_age = 13) →
  darius_age = 8 := by
sorry

end NUMINAMATH_CALUDE_darius_age_is_8_l1029_102947


namespace NUMINAMATH_CALUDE_sibling_product_specific_household_l1029_102964

/-- In a household with girls and boys, one boy counts all other children as siblings. -/
structure Household where
  girls : ℕ
  boys : ℕ
  counter : ℕ
  counter_is_boy : counter < boys

/-- The number of sisters the counter sees -/
def sisters (h : Household) : ℕ := h.girls

/-- The number of brothers the counter sees -/
def brothers (h : Household) : ℕ := h.boys - 1

/-- The product of sisters and brothers the counter sees -/
def sibling_product (h : Household) : ℕ := sisters h * brothers h

theorem sibling_product_specific_household :
  ∀ h : Household, h.girls = 5 → h.boys = 7 → sibling_product h = 24 := by
  sorry

end NUMINAMATH_CALUDE_sibling_product_specific_household_l1029_102964


namespace NUMINAMATH_CALUDE_sequence_formula_correct_l1029_102924

def sequence_term (n : ℕ) : ℚ := (-1)^n * (n^2 : ℚ) / (2*n - 1)

theorem sequence_formula_correct : 
  (sequence_term 1 = -1) ∧ 
  (sequence_term 2 = 4/3) ∧ 
  (sequence_term 3 = -9/5) ∧ 
  (sequence_term 4 = 16/7) := by
  sorry

end NUMINAMATH_CALUDE_sequence_formula_correct_l1029_102924


namespace NUMINAMATH_CALUDE_duck_eggs_sum_l1029_102931

theorem duck_eggs_sum (yesterday_eggs : ℕ) (fewer_today : ℕ) : 
  yesterday_eggs = 1925 →
  fewer_today = 138 →
  yesterday_eggs + (yesterday_eggs - fewer_today) = 3712 := by
  sorry

end NUMINAMATH_CALUDE_duck_eggs_sum_l1029_102931


namespace NUMINAMATH_CALUDE_part_I_part_II_l1029_102904

-- Define set A
def A : Set ℝ := {x | 6 / (x + 1) ≥ 1}

-- Define set B (parameterized by m)
def B (m : ℝ) : Set ℝ := {x | x^2 - 2*x - m < 0}

-- Theorem for part (I)
theorem part_I : A = {x | -1 < x ∧ x ≤ 5} := by sorry

-- Theorem for part (II)
theorem part_II : A ∩ (Set.univ \ B 3) = {x | 3 ≤ x ∧ x ≤ 5} := by sorry

end NUMINAMATH_CALUDE_part_I_part_II_l1029_102904


namespace NUMINAMATH_CALUDE_max_third_altitude_is_seven_l1029_102952

/-- A scalene triangle with two known altitudes -/
structure ScaleneTriangle where
  /-- The length of the first known altitude -/
  altitude1 : ℝ
  /-- The length of the second known altitude -/
  altitude2 : ℝ
  /-- The triangle is scalene -/
  scalene : True
  /-- The known altitudes have lengths 5 and 15 -/
  altitudes_given : altitude1 = 5 ∧ altitude2 = 15

/-- The maximum possible integer length of the third altitude -/
def max_third_altitude (triangle : ScaleneTriangle) : ℕ :=
  7

/-- Theorem stating that the maximum possible integer length of the third altitude is 7 -/
theorem max_third_altitude_is_seven (triangle : ScaleneTriangle) :
  max_third_altitude triangle = 7 := by
  sorry

end NUMINAMATH_CALUDE_max_third_altitude_is_seven_l1029_102952


namespace NUMINAMATH_CALUDE_c_share_is_64_l1029_102999

/-- Given a total sum of money divided among three parties a, b, and c,
    where b's share is 65% of a's and c's share is 40% of a's,
    prove that c's share is 64 when the total sum is 328. -/
theorem c_share_is_64 (total : ℝ) (a b c : ℝ) :
  total = 328 →
  b = 0.65 * a →
  c = 0.40 * a →
  total = a + b + c →
  c = 64 := by
  sorry

end NUMINAMATH_CALUDE_c_share_is_64_l1029_102999


namespace NUMINAMATH_CALUDE_inequality_proof_l1029_102992

theorem inequality_proof (a : ℝ) (n : ℕ) (h1 : a > -1) (h2 : a ≠ 0) (h3 : n ≥ 2) :
  (1 + a)^n > 1 + n * a := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l1029_102992


namespace NUMINAMATH_CALUDE_odd_function_period_4_symmetric_exists_a_inequality_f_is_odd_not_unique_a_for_odd_g_l1029_102930

-- Define an odd function with period 4
def OddFunctionPeriod4 (f : ℝ → ℝ) : Prop :=
  (∀ x, f (-x) = -f x) ∧ (∀ x, f (x + 4) = f x)

-- Define symmetry about (2,0)
def SymmetricAbout2_0 (f : ℝ → ℝ) : Prop :=
  ∀ x, f (4 - x) = f x

-- Statement 1
theorem odd_function_period_4_symmetric :
  ∀ f : ℝ → ℝ, OddFunctionPeriod4 f → SymmetricAbout2_0 f :=
sorry

-- Statement 2
theorem exists_a_inequality :
  ∃ a : ℝ, 0 < a ∧ a < 1 ∧ a^(1 + a) ≥ a^(1 + 1/a) :=
sorry

-- Define the logarithmic function
noncomputable def f (x : ℝ) : ℝ := Real.log ((1 + x) / (1 - x))

-- Statement 3
theorem f_is_odd :
  ∀ x : ℝ, -1 < x → x < 1 → f (-x) = -f x :=
sorry

-- Define the function g
noncomputable def g (a : ℝ) (x : ℝ) : ℝ := Real.log (a * x + Real.sqrt (2 * x^2 + 1))

-- Statement 4
theorem not_unique_a_for_odd_g :
  ¬ ∃! a : ℝ, ∀ x : ℝ, g a (-x) = -g a x :=
sorry

end NUMINAMATH_CALUDE_odd_function_period_4_symmetric_exists_a_inequality_f_is_odd_not_unique_a_for_odd_g_l1029_102930


namespace NUMINAMATH_CALUDE_line_through_midpoint_l1029_102991

-- Define the points and lines
def P : ℝ × ℝ := (1, 2)
def L1 : ℝ → ℝ → Prop := λ x y => 3 * x - y + 2 = 0
def L2 : ℝ → ℝ → Prop := λ x y => x - 2 * y + 1 = 0

-- Define the property of A and B being on L1 and L2 respectively
def A_on_L1 (A : ℝ × ℝ) : Prop := L1 A.1 A.2
def B_on_L2 (B : ℝ × ℝ) : Prop := L2 B.1 B.2

-- Define the midpoint property
def is_midpoint (M A B : ℝ × ℝ) : Prop :=
  M.1 = (A.1 + B.1) / 2 ∧ M.2 = (A.2 + B.2) / 2

-- Define the line equation
def line_equation (x y : ℝ) : Prop := 3 * x + 4 * y - 11 = 0

-- State the theorem
theorem line_through_midpoint :
  ∀ (A B : ℝ × ℝ),
    A_on_L1 A →
    B_on_L2 B →
    is_midpoint P A B →
    ∀ (x y : ℝ),
      (∃ (t : ℝ), x = P.1 + t * (A.1 - P.1) ∧ y = P.2 + t * (A.2 - P.2)) →
      line_equation x y :=
sorry

end NUMINAMATH_CALUDE_line_through_midpoint_l1029_102991


namespace NUMINAMATH_CALUDE_percent_relation_l1029_102954

theorem percent_relation (a b c : ℝ) (x : ℝ) 
  (h1 : c = 0.20 * a) 
  (h2 : b = 2.00 * a) 
  (h3 : c = (x / 100) * b) : 
  x = 10 := by
sorry

end NUMINAMATH_CALUDE_percent_relation_l1029_102954


namespace NUMINAMATH_CALUDE_sin_value_from_tan_l1029_102986

theorem sin_value_from_tan (α : Real) : 
  α > 0 ∧ α < Real.pi / 2 →  -- α is in the first quadrant
  Real.tan α = 3 / 4 →       -- tan α = 3/4
  Real.sin α = 3 / 5 :=      -- sin α = 3/5
by
  sorry

end NUMINAMATH_CALUDE_sin_value_from_tan_l1029_102986


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l1029_102906

/-- An arithmetic sequence with its partial sums -/
structure ArithmeticSequence where
  S : ℕ → ℝ  -- S_n is the sum of the first n terms

/-- Given an arithmetic sequence with S_3 = 3 and S_6 = 7, prove S_9 = 12 -/
theorem arithmetic_sequence_sum (a : ArithmeticSequence) 
  (h3 : a.S 3 = 3) 
  (h6 : a.S 6 = 7) : 
  a.S 9 = 12 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l1029_102906


namespace NUMINAMATH_CALUDE_susan_money_left_l1029_102912

/-- The amount of money Susan has left after spending at the fair -/
def money_left (initial_amount food_cost ride_cost game_cost : ℕ) : ℕ :=
  initial_amount - (food_cost + ride_cost + game_cost)

/-- Theorem stating that Susan has 10 dollars left to spend -/
theorem susan_money_left :
  let initial_amount := 80
  let food_cost := 15
  let ride_cost := 3 * food_cost
  let game_cost := 10
  money_left initial_amount food_cost ride_cost game_cost = 10 := by
  sorry

end NUMINAMATH_CALUDE_susan_money_left_l1029_102912


namespace NUMINAMATH_CALUDE_volume_ratio_octahedron_cube_l1029_102938

/-- A regular octahedron -/
structure RegularOctahedron where
  edge_length : ℝ
  volume : ℝ

/-- A cube whose vertices are the centers of the faces of a regular octahedron -/
structure RelatedCube where
  diagonal : ℝ
  volume : ℝ

/-- The relationship between a regular octahedron and its related cube -/
def octahedron_cube_relation (o : RegularOctahedron) (c : RelatedCube) : Prop :=
  c.diagonal = 2 * o.edge_length

theorem volume_ratio_octahedron_cube (o : RegularOctahedron) (c : RelatedCube) 
  (h : octahedron_cube_relation o c) : 
  o.volume / c.volume = 3 / 8 := by
  sorry

end NUMINAMATH_CALUDE_volume_ratio_octahedron_cube_l1029_102938


namespace NUMINAMATH_CALUDE_sum_S_six_cards_l1029_102956

/-- The number of strictly increasing subsequences of length 2 or more in a sequence -/
def S (π : List ℕ) : ℕ := sorry

/-- The sum of S(π) over all permutations of n distinct elements -/
def sum_S (n : ℕ) : ℕ := sorry

/-- Theorem: The sum of S(π) over all permutations of 6 distinct elements is 8287 -/
theorem sum_S_six_cards : sum_S 6 = 8287 := by sorry

end NUMINAMATH_CALUDE_sum_S_six_cards_l1029_102956


namespace NUMINAMATH_CALUDE_largest_c_for_two_in_range_l1029_102993

theorem largest_c_for_two_in_range : 
  let f (x c : ℝ) := 3 * x^2 - 6 * x + c
  ∃ (c_max : ℝ), c_max = 5 ∧ 
    (∀ c : ℝ, (∃ x : ℝ, f x c = 2) → c ≤ c_max) ∧
    (∃ x : ℝ, f x c_max = 2) :=
by sorry

end NUMINAMATH_CALUDE_largest_c_for_two_in_range_l1029_102993


namespace NUMINAMATH_CALUDE_system_solutions_l1029_102918

theorem system_solutions (x y z : ℚ) : 
  ((x + 1) * (3 - 4 * y) = (6 * x + 1) * (3 - 2 * y) ∧
   (4 * x - 1) * (z + 1) = (x + 1) * (z - 1) ∧
   (3 - y) * (z - 2) = (1 - 3 * y) * (z - 6)) ↔
  ((x = 0 ∧ y = 0 ∧ z = 0) ∨
   (x = 10/19 ∧ y = 25/7 ∧ z = 25/4)) :=
by sorry


end NUMINAMATH_CALUDE_system_solutions_l1029_102918


namespace NUMINAMATH_CALUDE_intersection_of_P_and_Q_l1029_102950

def P : Set (ℝ × ℝ) := {p | p.1 + p.2 = 3}
def Q : Set (ℝ × ℝ) := {q | q.1 - q.2 = 5}

theorem intersection_of_P_and_Q :
  P ∩ Q = {(4, -1)} := by sorry

end NUMINAMATH_CALUDE_intersection_of_P_and_Q_l1029_102950


namespace NUMINAMATH_CALUDE_right_triangle_area_l1029_102913

theorem right_triangle_area (hypotenuse : ℝ) (angle : ℝ) :
  hypotenuse = 8 * Real.sqrt 2 →
  angle = 45 * (π / 180) →
  let area := (hypotenuse^2 / 4)
  area = 32 := by sorry

end NUMINAMATH_CALUDE_right_triangle_area_l1029_102913


namespace NUMINAMATH_CALUDE_area_of_efgh_is_72_l1029_102983

/-- Represents a circle with a center point and radius -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- Represents a rectangle with two opposite corners -/
structure Rectangle where
  corner1 : ℝ × ℝ
  corner2 : ℝ × ℝ

/-- The configuration of circles and rectangle in the problem -/
structure CircleConfiguration where
  efgh : Rectangle
  circleA : Circle
  circleB : Circle
  circleC : Circle
  circleD : Circle

/-- Checks if two circles are congruent -/
def areCongruentCircles (c1 c2 : Circle) : Prop :=
  c1.radius = c2.radius

/-- Checks if a circle is tangent to two adjacent sides of a rectangle -/
def isTangentToAdjacentSides (c : Circle) (r : Rectangle) : Prop :=
  sorry -- Definition omitted for brevity

/-- Checks if the centers of four circles form a rectangle -/
def centersFormRectangle (c1 c2 c3 c4 : Circle) : Prop :=
  sorry -- Definition omitted for brevity

/-- Calculates the area of a rectangle -/
def rectangleArea (r : Rectangle) : ℝ :=
  sorry -- Definition omitted for brevity

theorem area_of_efgh_is_72 (config : CircleConfiguration) :
  (areCongruentCircles config.circleA config.circleB) →
  (areCongruentCircles config.circleA config.circleC) →
  (areCongruentCircles config.circleA config.circleD) →
  (config.circleB.radius = 3) →
  (isTangentToAdjacentSides config.circleB config.efgh) →
  (centersFormRectangle config.circleA config.circleB config.circleC config.circleD) →
  rectangleArea config.efgh = 72 :=
by
  sorry

end NUMINAMATH_CALUDE_area_of_efgh_is_72_l1029_102983


namespace NUMINAMATH_CALUDE_number_thought_of_l1029_102968

theorem number_thought_of (x : ℝ) : x / 5 + 23 = 42 → x = 95 := by
  sorry

end NUMINAMATH_CALUDE_number_thought_of_l1029_102968


namespace NUMINAMATH_CALUDE_log_equality_l1029_102945

theorem log_equality (a b : ℝ) (h1 : a = Real.log 900 / Real.log 4) (h2 : b = Real.log 30 / Real.log 2) : a = b := by
  sorry

end NUMINAMATH_CALUDE_log_equality_l1029_102945


namespace NUMINAMATH_CALUDE_students_not_in_biology_l1029_102979

theorem students_not_in_biology (total_students : ℕ) (biology_percentage : ℚ) 
  (h1 : total_students = 840)
  (h2 : biology_percentage = 35 / 100) :
  total_students - (total_students * biology_percentage).floor = 546 := by
  sorry

end NUMINAMATH_CALUDE_students_not_in_biology_l1029_102979


namespace NUMINAMATH_CALUDE_x_intercept_distance_l1029_102928

/-- Two lines intersecting at (4, 6) with slopes 2 and 6 respectively have x-intercepts 2 units apart -/
theorem x_intercept_distance (line1 line2 : ℝ → ℝ) : 
  (∃ (y : ℝ), line1 4 = 6 ∧ line2 4 = 6) →  -- Lines intersect at (4, 6)
  (∀ (x y : ℝ), line1 y - line1 x = 2 * (y - x)) →  -- Slope of line1 is 2
  (∀ (x y : ℝ), line2 y - line2 x = 6 * (y - x)) →  -- Slope of line2 is 6
  ∃ (x1 x2 : ℝ), 
    line1 x1 = 0 ∧  -- x1 is x-intercept of line1
    line2 x2 = 0 ∧  -- x2 is x-intercept of line2
    |x2 - x1| = 2   -- Distance between x-intercepts is 2
:= by sorry

end NUMINAMATH_CALUDE_x_intercept_distance_l1029_102928


namespace NUMINAMATH_CALUDE_smaller_number_proof_l1029_102961

theorem smaller_number_proof (x y : ℝ) 
  (sum_eq : x + y = 18)
  (diff_eq : x - y = 4)
  (prod_eq : x * y = 77) :
  y = 7 := by
  sorry

end NUMINAMATH_CALUDE_smaller_number_proof_l1029_102961


namespace NUMINAMATH_CALUDE_problem_statement_l1029_102962

theorem problem_statement (a b c : ℝ) 
  (h_pos : a > 0 ∧ b > 0 ∧ c > 0) 
  (h_not_all_equal : ¬(a = b ∧ b = c)) : 
  ((a - b)^2 + (b - c)^2 + (c - a)^2 ≠ 0) ∧ 
  (a > b ∨ a < b ∨ a = b) ∧
  (∃ (x y z : ℝ), x ≠ z ∧ y ≠ z ∧ x ≠ y) :=
by sorry

end NUMINAMATH_CALUDE_problem_statement_l1029_102962


namespace NUMINAMATH_CALUDE_initial_soldiers_count_l1029_102946

theorem initial_soldiers_count (provisions : ℝ) : ∃ (initial_soldiers : ℕ),
  provisions = initial_soldiers * 3 * 30 ∧
  provisions = (initial_soldiers + 528) * 2.5 * 25 ∧
  initial_soldiers = 1200 := by
sorry

end NUMINAMATH_CALUDE_initial_soldiers_count_l1029_102946


namespace NUMINAMATH_CALUDE_carnival_activities_popularity_order_l1029_102976

def dodgeball_popularity : ℚ := 13/40
def karaoke_popularity : ℚ := 9/30
def magic_show_popularity : ℚ := 17/60
def quiz_bowl_popularity : ℚ := 23/120

theorem carnival_activities_popularity_order :
  dodgeball_popularity > karaoke_popularity ∧
  karaoke_popularity > magic_show_popularity ∧
  magic_show_popularity > quiz_bowl_popularity :=
by sorry

end NUMINAMATH_CALUDE_carnival_activities_popularity_order_l1029_102976


namespace NUMINAMATH_CALUDE_quadratic_two_distinct_roots_l1029_102970

theorem quadratic_two_distinct_roots (m : ℝ) :
  (∃ x y : ℝ, x ≠ y ∧ 
    x^2 - 2*x + m - 1 = 0 ∧ 
    y^2 - 2*y + m - 1 = 0) →
  m < 2 := by
sorry

end NUMINAMATH_CALUDE_quadratic_two_distinct_roots_l1029_102970
