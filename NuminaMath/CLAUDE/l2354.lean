import Mathlib

namespace NUMINAMATH_CALUDE_total_like_count_l2354_235473

/-- Represents the number of employees with a "dislike" attitude -/
def dislike_count : ℕ := sorry

/-- Represents the number of employees with a "neutral" attitude -/
def neutral_count : ℕ := dislike_count + 12

/-- Represents the number of employees with a "like" attitude -/
def like_count : ℕ := 6 * dislike_count

/-- Represents the ratio of employees with each attitude in the stratified sample -/
def sample_ratio : ℕ × ℕ × ℕ := (6, 1, 3)

theorem total_like_count : like_count = 36 := by sorry

end NUMINAMATH_CALUDE_total_like_count_l2354_235473


namespace NUMINAMATH_CALUDE_water_pricing_l2354_235432

/-- Water pricing problem -/
theorem water_pricing
  (a : ℝ) -- Previous year's water usage
  (k : ℝ) -- Proportionality coefficient
  (h_a : a > 0) -- Assumption: water usage is positive
  (h_k : k > 0) -- Assumption: coefficient is positive
  :
  -- 1. Revenue function
  let revenue (x : ℝ) := (a + k / (x - 2)) * (x - 1.8)
  -- 2. Minimum water price for 20% increase when k = 0.4a
  ∃ (x : ℝ), x = 2.4 ∧ 
    (∀ y ∈ Set.Icc 2.3 2.6, 
      revenue y ≥ 1.2 * (2.8 * a - 1.8 * a) → y ≥ x) ∧
    k = 0.4 * a →
    revenue x ≥ 1.2 * (2.8 * a - 1.8 * a)
  -- 3. Water price for minimum revenue and minimum revenue when k = 0.8a
  ∧ ∃ (x : ℝ), x = 2.4 ∧
    (∀ y ∈ Set.Icc 2.3 2.6, revenue x ≤ revenue y) ∧
    k = 0.8 * a →
    revenue x = 1.8 * a :=
by
  sorry

end NUMINAMATH_CALUDE_water_pricing_l2354_235432


namespace NUMINAMATH_CALUDE_line_canonical_equations_l2354_235403

/-- The canonical equations of a line given by the intersection of two planes -/
theorem line_canonical_equations (x y z : ℝ) : 
  (x + 5*y - z + 11 = 0 ∧ x - y + 2*z - 1 = 0) → 
  ((x + 1)/9 = (y + 2)/(-3) ∧ (y + 2)/(-3) = z/(-6)) :=
by sorry

end NUMINAMATH_CALUDE_line_canonical_equations_l2354_235403


namespace NUMINAMATH_CALUDE_thabos_books_l2354_235482

/-- Represents the number of books Thabo owns in each category -/
structure BookCollection where
  paperbackFiction : ℕ
  paperbackNonfiction : ℕ
  hardcoverNonfiction : ℕ

/-- Thabo's book collection satisfies the given conditions -/
def validCollection (books : BookCollection) : Prop :=
  books.paperbackFiction + books.paperbackNonfiction + books.hardcoverNonfiction = 160 ∧
  books.paperbackNonfiction > books.hardcoverNonfiction ∧
  books.paperbackFiction = 2 * books.paperbackNonfiction ∧
  books.hardcoverNonfiction = 25

theorem thabos_books (books : BookCollection) (h : validCollection books) :
  books.paperbackNonfiction - books.hardcoverNonfiction = 20 := by
  sorry

end NUMINAMATH_CALUDE_thabos_books_l2354_235482


namespace NUMINAMATH_CALUDE_twenty_five_percent_problem_l2354_235444

theorem twenty_five_percent_problem : ∃ x : ℝ, (0.75 * 80 = 1.25 * x) ∧ (x = 48) := by
  sorry

end NUMINAMATH_CALUDE_twenty_five_percent_problem_l2354_235444


namespace NUMINAMATH_CALUDE_logarithm_sum_property_l2354_235463

theorem logarithm_sum_property (a b : ℝ) (ha : a > 1) (hb : b > 1) 
  (h : Real.log (a + b) = Real.log a + Real.log b) : 
  Real.log (a - 1) + Real.log (b - 1) = 0 := by sorry

end NUMINAMATH_CALUDE_logarithm_sum_property_l2354_235463


namespace NUMINAMATH_CALUDE_cube_root_equation_solution_l2354_235479

theorem cube_root_equation_solution :
  ∃! x : ℝ, (3 - x / 3) ^ (1/3 : ℝ) = -2 :=
by sorry

end NUMINAMATH_CALUDE_cube_root_equation_solution_l2354_235479


namespace NUMINAMATH_CALUDE_farmer_profit_percentage_l2354_235476

/-- Calculate the profit percentage for a farmer's corn harvest --/
theorem farmer_profit_percentage
  (corn_seeds_cost : ℝ)
  (fertilizers_pesticides_cost : ℝ)
  (labor_cost : ℝ)
  (num_corn_bags : ℕ)
  (price_per_bag : ℝ)
  (h1 : corn_seeds_cost = 50)
  (h2 : fertilizers_pesticides_cost = 35)
  (h3 : labor_cost = 15)
  (h4 : num_corn_bags = 10)
  (h5 : price_per_bag = 11) :
  let total_cost := corn_seeds_cost + fertilizers_pesticides_cost + labor_cost
  let total_revenue := (num_corn_bags : ℝ) * price_per_bag
  let profit := total_revenue - total_cost
  let profit_percentage := (profit / total_cost) * 100
  profit_percentage = 10 := by
sorry

end NUMINAMATH_CALUDE_farmer_profit_percentage_l2354_235476


namespace NUMINAMATH_CALUDE_smallest_mustang_length_l2354_235450

/-- Proves that the smallest model Mustang is 12 inches long given the specified conditions -/
theorem smallest_mustang_length :
  let full_size : ℝ := 240
  let mid_size_ratio : ℝ := 1 / 10
  let smallest_ratio : ℝ := 1 / 2
  let mid_size : ℝ := full_size * mid_size_ratio
  let smallest_size : ℝ := mid_size * smallest_ratio
  smallest_size = 12 := by sorry

end NUMINAMATH_CALUDE_smallest_mustang_length_l2354_235450


namespace NUMINAMATH_CALUDE_parallelogram_side_length_l2354_235454

theorem parallelogram_side_length 
  (s : ℝ) 
  (angle : ℝ) 
  (area : ℝ) 
  (h₁ : angle = π / 3) -- 60 degrees in radians
  (h₂ : area = 27 * Real.sqrt 3)
  (h₃ : area = 3 * s * s * Real.sin angle) :
  s = 3 * Real.sqrt 2 := by
sorry

end NUMINAMATH_CALUDE_parallelogram_side_length_l2354_235454


namespace NUMINAMATH_CALUDE_opposite_of_seven_l2354_235483

theorem opposite_of_seven : 
  (-(7 : ℝ) = -7) := by sorry

end NUMINAMATH_CALUDE_opposite_of_seven_l2354_235483


namespace NUMINAMATH_CALUDE_inequality_range_l2354_235408

theorem inequality_range (m : ℝ) : 
  (∀ x : ℝ, (m^2 - 2*m - 3)*x^2 - (m - 3)*x - 1 < 0) ↔ 
  m > -1/5 ∧ m ≤ 3 := by sorry

end NUMINAMATH_CALUDE_inequality_range_l2354_235408


namespace NUMINAMATH_CALUDE_andrey_gleb_distance_l2354_235413

/-- Represents the position of a home on a straight street -/
structure Home where
  position : ℝ

/-- The street with four homes -/
structure Street where
  andrey : Home
  borya : Home
  vova : Home
  gleb : Home

/-- The distance between two homes -/
def distance (h1 h2 : Home) : ℝ := |h1.position - h2.position|

/-- The conditions of the problem -/
def valid_street (s : Street) : Prop :=
  distance s.andrey s.borya = 600 ∧
  distance s.vova s.gleb = 600 ∧
  distance s.andrey s.gleb = 3 * distance s.borya s.vova

/-- The theorem to be proved -/
theorem andrey_gleb_distance (s : Street) :
  valid_street s →
  distance s.andrey s.gleb = 1500 ∨ distance s.andrey s.gleb = 1800 :=
sorry

end NUMINAMATH_CALUDE_andrey_gleb_distance_l2354_235413


namespace NUMINAMATH_CALUDE_triangle_abc_properties_l2354_235436

theorem triangle_abc_properties (a b c : ℝ) (A B C : ℝ) :
  -- Given conditions
  (2 * c = Real.sqrt 3 * a + 2 * b * Real.cos A) →
  (c = 1) →
  (1 / 2 * a * c * Real.sin B = Real.sqrt 3 / 2) →
  -- Conclusions
  (B = π / 6) ∧ (b = Real.sqrt 7) :=
by sorry

end NUMINAMATH_CALUDE_triangle_abc_properties_l2354_235436


namespace NUMINAMATH_CALUDE_rectangle_area_increase_l2354_235451

theorem rectangle_area_increase (L W : ℝ) (hL : L > 0) (hW : W > 0) :
  let original_area := L * W
  let new_length := 1.2 * L
  let new_width := 1.2 * W
  let new_area := new_length * new_width
  (new_area - original_area) / original_area = 0.44 := by
sorry

end NUMINAMATH_CALUDE_rectangle_area_increase_l2354_235451


namespace NUMINAMATH_CALUDE_total_cost_is_21_93_l2354_235445

/-- The amount Alyssa spent on grapes -/
def grapes_cost : ℚ := 12.08

/-- The amount Alyssa spent on cherries -/
def cherries_cost : ℚ := 9.85

/-- The total amount Alyssa spent on fruits -/
def total_cost : ℚ := grapes_cost + cherries_cost

/-- Theorem stating that the total cost is equal to $21.93 -/
theorem total_cost_is_21_93 : total_cost = 21.93 := by
  sorry

end NUMINAMATH_CALUDE_total_cost_is_21_93_l2354_235445


namespace NUMINAMATH_CALUDE_base4_addition_theorem_l2354_235448

/-- Addition of numbers in base 4 -/
def base4_add (a b c d : ℕ) : ℕ := sorry

/-- Conversion from base 4 to decimal -/
def base4_to_decimal (n : ℕ) : ℕ := sorry

theorem base4_addition_theorem :
  base4_add (base4_to_decimal 2) (base4_to_decimal 23) (base4_to_decimal 132) (base4_to_decimal 1320) = base4_to_decimal 20200 := by
  sorry

end NUMINAMATH_CALUDE_base4_addition_theorem_l2354_235448


namespace NUMINAMATH_CALUDE_zoo_elephant_count_l2354_235469

/-- Represents the number of animals of each type in the zoo -/
structure ZooPopulation where
  giraffes : ℕ
  penguins : ℕ
  elephants : ℕ
  total : ℕ

/-- The conditions of the zoo population -/
def zoo_conditions (pop : ZooPopulation) : Prop :=
  pop.giraffes = 5 ∧
  pop.penguins = 2 * pop.giraffes ∧
  pop.penguins = (20 : ℕ) * pop.total / 100 ∧
  pop.elephants = (4 : ℕ) * pop.total / 100 ∧
  pop.total = pop.giraffes + pop.penguins + pop.elephants

theorem zoo_elephant_count :
  ∀ pop : ZooPopulation, zoo_conditions pop → pop.elephants = 2 :=
by sorry

end NUMINAMATH_CALUDE_zoo_elephant_count_l2354_235469


namespace NUMINAMATH_CALUDE_sum_of_abc_equals_45_l2354_235494

-- Define a triangle with side lengths 3, 7, and x
structure Triangle where
  x : ℝ
  side1 : ℝ := 3
  side2 : ℝ := 7
  side3 : ℝ := x

-- Define the property of angles in arithmetic progression
def anglesInArithmeticProgression (t : Triangle) : Prop := sorry

-- Define the sum of possible values of x
def sumOfPossibleX (t : Triangle) : ℝ := sorry

-- Define a, b, and c as positive integers
def a : ℕ+ := sorry
def b : ℕ+ := sorry
def c : ℕ+ := sorry

-- Theorem statement
theorem sum_of_abc_equals_45 (t : Triangle) 
  (h1 : anglesInArithmeticProgression t) 
  (h2 : sumOfPossibleX t = a + Real.sqrt b + Real.sqrt c) : 
  a + b + c = 45 := by sorry

end NUMINAMATH_CALUDE_sum_of_abc_equals_45_l2354_235494


namespace NUMINAMATH_CALUDE_quadratic_equation_roots_l2354_235412

/-- Given a quadratic equation x^2 - (3+m)x + 3m = 0 with real roots x1 and x2
    satisfying 2x1 - x1x2 + 2x2 = 12, prove that x1 = -6 and x2 = 3 -/
theorem quadratic_equation_roots (m : ℝ) (x1 x2 : ℝ) :
  x1^2 - (3+m)*x1 + 3*m = 0 →
  x2^2 - (3+m)*x2 + 3*m = 0 →
  2*x1 - x1*x2 + 2*x2 = 12 →
  (x1 = -6 ∧ x2 = 3) ∨ (x1 = 3 ∧ x2 = -6) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equation_roots_l2354_235412


namespace NUMINAMATH_CALUDE_force_balance_l2354_235427

/-- A force in 2D space represented by its x and y components -/
structure Force where
  x : ℝ
  y : ℝ

/-- The sum of two forces -/
def Force.add (f g : Force) : Force :=
  ⟨f.x + g.x, f.y + g.y⟩

/-- The negation of a force -/
def Force.neg (f : Force) : Force :=
  ⟨-f.x, -f.y⟩

/-- Given two forces F₁ and F₂, prove that F₃ balances the system -/
theorem force_balance (F₁ F₂ F₃ : Force) 
    (h₁ : F₁ = ⟨1, 1⟩) 
    (h₂ : F₂ = ⟨2, 3⟩) 
    (h₃ : F₃ = ⟨-3, -4⟩) : 
  F₃.add (F₁.add F₂) = ⟨0, 0⟩ := by
  sorry


end NUMINAMATH_CALUDE_force_balance_l2354_235427


namespace NUMINAMATH_CALUDE_special_function_inequality_l2354_235442

/-- A function satisfying the given conditions -/
structure SpecialFunction where
  f : ℝ → ℝ
  has_derivative : Differentiable ℝ f
  symmetric : ∀ x, f x = 6 * x^2 - f (-x)
  derivative_bound : ∀ x, x < 0 → 2 * deriv f x + 1 < 12 * x

/-- The main theorem -/
theorem special_function_inequality (sf : SpecialFunction) :
  ∀ m : ℝ, sf.f (m + 2) ≤ sf.f (-2 * m) + 12 * m + 12 - 9 * m^2 ↔ m ≥ - 2/3 :=
by sorry

end NUMINAMATH_CALUDE_special_function_inequality_l2354_235442


namespace NUMINAMATH_CALUDE_chairs_to_remove_l2354_235400

/-- Given a conference hall setup with the following conditions:
  - Each row has 15 chairs
  - Initially, there are 195 chairs
  - 120 attendees are expected
  - All rows must be complete
  - The number of remaining chairs must be the smallest multiple of 15 that is greater than or equal to 120
  
  This theorem proves that the number of chairs to be removed is 60. -/
theorem chairs_to_remove (chairs_per_row : ℕ) (initial_chairs : ℕ) (expected_attendees : ℕ)
  (h1 : chairs_per_row = 15)
  (h2 : initial_chairs = 195)
  (h3 : expected_attendees = 120)
  (h4 : ∃ (n : ℕ), n * chairs_per_row ≥ expected_attendees ∧
        ∀ (m : ℕ), m * chairs_per_row ≥ expected_attendees → n ≤ m) :
  initial_chairs - (chairs_per_row * (initial_chairs / chairs_per_row)) = 60 :=
sorry

end NUMINAMATH_CALUDE_chairs_to_remove_l2354_235400


namespace NUMINAMATH_CALUDE_triangle_angle_cosine_l2354_235433

theorem triangle_angle_cosine (A B C : ℝ) : 
  0 < A ∧ 0 < B ∧ 0 < C ∧ -- Angles are positive
  A + B + C = π ∧ -- Sum of angles in a triangle
  A + C = 2 * B ∧ -- Given condition
  1 / Real.cos A + 1 / Real.cos C = -Real.sqrt 2 / Real.cos B -- Given condition
  → Real.cos ((A - C) / 2) = Real.sqrt 2 / 2 := by
    sorry

end NUMINAMATH_CALUDE_triangle_angle_cosine_l2354_235433


namespace NUMINAMATH_CALUDE_vector_sum_proof_l2354_235447

/-- Given points A, B, and C in ℝ², prove that AC + (1/3)BA = (2, -3) -/
theorem vector_sum_proof (A B C : ℝ × ℝ) 
  (hA : A = (2, 4)) 
  (hB : B = (-1, -5)) 
  (hC : C = (3, -2)) : 
  (C.1 - A.1, C.2 - A.2) + (1/3 * (A.1 - B.1), 1/3 * (A.2 - B.2)) = (2, -3) := by
  sorry

end NUMINAMATH_CALUDE_vector_sum_proof_l2354_235447


namespace NUMINAMATH_CALUDE_quadratic_transformation_l2354_235478

def f (x : ℝ) : ℝ := x^2

def shift_right (f : ℝ → ℝ) (a : ℝ) : ℝ → ℝ := λ x => f (x - a)

def shift_up (f : ℝ → ℝ) (b : ℝ) : ℝ → ℝ := λ x => f x + b

def g (x : ℝ) : ℝ := (x - 3)^2 + 4

theorem quadratic_transformation :
  shift_up (shift_right f 3) 4 = g := by sorry

end NUMINAMATH_CALUDE_quadratic_transformation_l2354_235478


namespace NUMINAMATH_CALUDE_equation_solution_l2354_235406

theorem equation_solution (x : ℝ) :
  x ≥ 0 →
  (2021 * (x^2020)^(1/202) - 1 = 2020 * x) ↔
  x = 1 :=
by sorry

end NUMINAMATH_CALUDE_equation_solution_l2354_235406


namespace NUMINAMATH_CALUDE_lamps_remaining_lit_l2354_235431

/-- The number of lamps initially lit -/
def total_lamps : ℕ := 1997

/-- Function to count lamps that are multiples of a given number -/
def count_multiples (n : ℕ) : ℕ :=
  (total_lamps - (total_lamps % n)) / n

/-- Function to count lamps that are multiples of two given numbers -/
def count_common_multiples (a b : ℕ) : ℕ :=
  (total_lamps - (total_lamps % (a * b))) / (a * b)

/-- Function to count lamps that are multiples of three given numbers -/
def count_triple_multiples (a b c : ℕ) : ℕ :=
  (total_lamps - (total_lamps % (a * b * c))) / (a * b * c)

/-- The main theorem stating the number of lamps that remain lit -/
theorem lamps_remaining_lit : 
  total_lamps - 
  (count_multiples 2 - count_common_multiples 2 3 - count_common_multiples 2 5 + count_triple_multiples 2 3 5) -
  (count_multiples 3 - count_common_multiples 2 3 - count_common_multiples 3 5 + count_triple_multiples 2 3 5) -
  (count_multiples 5 - count_common_multiples 2 5 - count_common_multiples 3 5 + count_triple_multiples 2 3 5) = 999 := by
  sorry

end NUMINAMATH_CALUDE_lamps_remaining_lit_l2354_235431


namespace NUMINAMATH_CALUDE_intersection_is_empty_l2354_235490

def A : Set ℝ := {α | ∃ k : ℤ, α = (5 * k * Real.pi) / 3}
def B : Set ℝ := {β | ∃ k : ℤ, β = (3 * k * Real.pi) / 2}

theorem intersection_is_empty : A ∩ B = ∅ := by
  sorry

end NUMINAMATH_CALUDE_intersection_is_empty_l2354_235490


namespace NUMINAMATH_CALUDE_unit_vector_parallel_l2354_235466

/-- Given two vectors a and b in ℝ², prove that the unit vector parallel to 2a - 3b
    is either (√5/5, 2√5/5) or (-√5/5, -2√5/5) -/
theorem unit_vector_parallel (a b : ℝ × ℝ) (ha : a = (5, 4)) (hb : b = (3, 2)) :
  let v := (2 • a.1 - 3 • b.1, 2 • a.2 - 3 • b.2)
  (v.1 / Real.sqrt (v.1^2 + v.2^2), v.2 / Real.sqrt (v.1^2 + v.2^2)) = (Real.sqrt 5 / 5, 2 * Real.sqrt 5 / 5) ∨
  (v.1 / Real.sqrt (v.1^2 + v.2^2), v.2 / Real.sqrt (v.1^2 + v.2^2)) = (-Real.sqrt 5 / 5, -2 * Real.sqrt 5 / 5) :=
by sorry

end NUMINAMATH_CALUDE_unit_vector_parallel_l2354_235466


namespace NUMINAMATH_CALUDE_circle_center_correct_l2354_235440

/-- The center of a circle given by the equation x^2 + y^2 - 2x + 4y = 0 --/
def circle_center : ℝ × ℝ := sorry

/-- The equation of the circle --/
def circle_equation (x y : ℝ) : Prop :=
  x^2 + y^2 - 2*x + 4*y = 0

theorem circle_center_correct :
  let (h, k) := circle_center
  ∀ x y, circle_equation x y ↔ (x - h)^2 + (y - k)^2 = (h^2 + k^2 - 1) ∧ 
  h = 1 ∧ k = -2 := by sorry

end NUMINAMATH_CALUDE_circle_center_correct_l2354_235440


namespace NUMINAMATH_CALUDE_min_socks_for_15_pairs_l2354_235487

/-- Represents the number of socks of each color in the drawer -/
def Drawer := List Nat

/-- The total number of socks in the drawer -/
def total_socks (d : Drawer) : Nat :=
  d.sum

/-- The number of different colors of socks in the drawer -/
def num_colors (d : Drawer) : Nat :=
  d.length

/-- The minimum number of socks needed to guarantee a certain number of pairs -/
def min_socks_for_pairs (num_pairs : Nat) (num_colors : Nat) : Nat :=
  num_colors + 2 * (num_pairs - 1)

theorem min_socks_for_15_pairs (d : Drawer) :
  num_colors d = 5 →
  total_socks d ≥ 400 →
  min_socks_for_pairs 15 (num_colors d) = 33 :=
by sorry

end NUMINAMATH_CALUDE_min_socks_for_15_pairs_l2354_235487


namespace NUMINAMATH_CALUDE_geometric_sum_problem_l2354_235422

/-- Sum of a geometric sequence -/
def geometric_sum (a : ℚ) (r : ℚ) (n : ℕ) : ℚ :=
  a * (1 - r^n) / (1 - r)

/-- The problem statement -/
theorem geometric_sum_problem : 
  let a : ℚ := 1/3
  let r : ℚ := 1/3
  let n : ℕ := 8
  geometric_sum a r n = 3280/6561 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sum_problem_l2354_235422


namespace NUMINAMATH_CALUDE_total_profit_is_45000_l2354_235404

/-- Represents the total profit earned by Tom and Jose given their investments and Jose's share of profit. -/
def total_profit (tom_investment : ℕ) (tom_months : ℕ) (jose_investment : ℕ) (jose_months : ℕ) (jose_profit : ℕ) : ℕ :=
  let tom_ratio : ℕ := tom_investment * tom_months
  let jose_ratio : ℕ := jose_investment * jose_months
  let total_ratio : ℕ := tom_ratio + jose_ratio
  (jose_profit * total_ratio) / jose_ratio

/-- Theorem stating that the total profit is 45000 given the specified conditions. -/
theorem total_profit_is_45000 :
  total_profit 30000 12 45000 10 25000 = 45000 := by
  sorry

end NUMINAMATH_CALUDE_total_profit_is_45000_l2354_235404


namespace NUMINAMATH_CALUDE_circumcircle_radius_of_specific_triangle_l2354_235471

/-- The radius of the circumcircle of a triangle with side lengths 8, 15, and 17 is 8.5. -/
theorem circumcircle_radius_of_specific_triangle : 
  ∀ (a b c : ℝ) (r : ℝ),
    a = 8 → b = 15 → c = 17 →
    (a^2 + b^2 = c^2) →  -- right triangle condition
    r = c / 2 →          -- radius is half the hypotenuse
    r = 8.5 := by
  sorry

end NUMINAMATH_CALUDE_circumcircle_radius_of_specific_triangle_l2354_235471


namespace NUMINAMATH_CALUDE_distinct_prime_factors_of_252_l2354_235409

theorem distinct_prime_factors_of_252 : Nat.card (Nat.factors 252).toFinset = 3 := by
  sorry

end NUMINAMATH_CALUDE_distinct_prime_factors_of_252_l2354_235409


namespace NUMINAMATH_CALUDE_repeating_base_k_representation_l2354_235455

/-- Given positive integers m and k, if the repeating base-k representation of 3/28 is 0.121212...₍ₖ₎, then k = 10 -/
theorem repeating_base_k_representation (m k : ℕ+) :
  (∃ (a : ℕ → ℕ), (∀ n, a n < k) ∧
    (∀ n, a (2*n) = 1 ∧ a (2*n+1) = 2) ∧
    (3 : ℚ) / 28 = ∑' n, (a n : ℚ) / k^(n+1)) →
  k = 10 := by sorry

end NUMINAMATH_CALUDE_repeating_base_k_representation_l2354_235455


namespace NUMINAMATH_CALUDE_arielle_age_l2354_235443

theorem arielle_age (elvie_age : ℕ) (total : ℕ) : 
  elvie_age = 10 →
  (∃ (arielle_age : ℕ), 
    elvie_age + arielle_age + elvie_age * arielle_age = total ∧
    total = 131) →
  ∃ (arielle_age : ℕ), arielle_age = 11 :=
by sorry

end NUMINAMATH_CALUDE_arielle_age_l2354_235443


namespace NUMINAMATH_CALUDE_alyssa_soccer_games_l2354_235481

theorem alyssa_soccer_games (this_year last_year next_year total : ℕ) 
  (h1 : this_year = 11)
  (h2 : last_year = 13)
  (h3 : next_year = 15)
  (h4 : total = 39)
  (h5 : this_year + last_year + next_year = total) : 
  this_year - (total - (last_year + next_year)) = 0 :=
by sorry

end NUMINAMATH_CALUDE_alyssa_soccer_games_l2354_235481


namespace NUMINAMATH_CALUDE_log_relation_l2354_235485

theorem log_relation (p q : ℝ) : 
  p = Real.log 192 / Real.log 5 → 
  q = Real.log 12 / Real.log 3 → 
  p = (q * (Real.log 12 / Real.log 3 + 8/3)) / (Real.log 5 / Real.log 3) := by
sorry

end NUMINAMATH_CALUDE_log_relation_l2354_235485


namespace NUMINAMATH_CALUDE_linear_equation_not_proportional_l2354_235425

/-- A linear equation in two variables -/
structure LinearEquation where
  a : ℝ
  b : ℝ
  c : ℝ
  a_nonzero : a ≠ 0
  b_nonzero : b ≠ 0

/-- Direct proportionality between x and y -/
def DirectlyProportional (x y : ℝ → ℝ) : Prop :=
  ∃ k : ℝ, ∀ t, y t = k * x t

/-- Inverse proportionality between x and y -/
def InverselyProportional (x y : ℝ → ℝ) : Prop :=
  ∃ k : ℝ, ∀ t, x t * y t = k

/-- 
For a linear equation ax + by = c, where a ≠ 0 and b ≠ 0,
y is neither directly nor inversely proportional to x
-/
theorem linear_equation_not_proportional (eq : LinearEquation) :
  let x : ℝ → ℝ := λ t => t
  let y : ℝ → ℝ := λ t => (eq.c - eq.a * t) / eq.b
  ¬(DirectlyProportional x y ∨ InverselyProportional x y) := by
  sorry


end NUMINAMATH_CALUDE_linear_equation_not_proportional_l2354_235425


namespace NUMINAMATH_CALUDE_intersection_A_complement_B_range_of_a_l2354_235467

-- Define the sets A, B, and M
def A : Set ℝ := {x | x^2 - x - 2 < 0}
def B : Set ℝ := {x | x * (3 - x) > 0}
def M (a : ℝ) : Set ℝ := {x | 2 * x - a < 0}

-- Theorem for part 1
theorem intersection_A_complement_B : A ∩ (Set.univ \ B) = {x : ℝ | -1 < x ∧ x ≤ 0} := by sorry

-- Theorem for part 2
theorem range_of_a (a : ℝ) : (A ∪ B) ⊆ M a → a ≥ 6 := by sorry

end NUMINAMATH_CALUDE_intersection_A_complement_B_range_of_a_l2354_235467


namespace NUMINAMATH_CALUDE_square_minus_a_nonpositive_l2354_235411

theorem square_minus_a_nonpositive (a : ℝ) (h : a > 4) :
  ∀ x : ℝ, x ∈ Set.Icc (-1) 2 → x^2 - a ≤ 0 := by sorry

end NUMINAMATH_CALUDE_square_minus_a_nonpositive_l2354_235411


namespace NUMINAMATH_CALUDE_johns_pens_l2354_235464

/-- The number of pens John has -/
def total_pens (blue black red : ℕ) : ℕ := blue + black + red

theorem johns_pens :
  ∀ (blue black red : ℕ),
  blue = 18 →
  blue = 2 * black →
  black = red + 5 →
  total_pens blue black red = 31 := by
sorry

end NUMINAMATH_CALUDE_johns_pens_l2354_235464


namespace NUMINAMATH_CALUDE_consecutive_integers_sum_of_squares_l2354_235429

theorem consecutive_integers_sum_of_squares : 
  ∃ (b : ℕ), 
    (b > 0) ∧ 
    ((b - 1) * b * (b + 1) = 12 * (3 * b)) → 
    ((b - 1)^2 + b^2 + (b + 1)^2 = 110) :=
by sorry

end NUMINAMATH_CALUDE_consecutive_integers_sum_of_squares_l2354_235429


namespace NUMINAMATH_CALUDE_circus_ticket_cost_l2354_235472

theorem circus_ticket_cost (ticket_price : ℝ) (num_tickets : ℕ) (total_cost : ℝ) : 
  ticket_price = 44 ∧ num_tickets = 7 ∧ total_cost = ticket_price * num_tickets → total_cost = 308 :=
by sorry

end NUMINAMATH_CALUDE_circus_ticket_cost_l2354_235472


namespace NUMINAMATH_CALUDE_no_four_primes_product_11_times_sum_l2354_235497

theorem no_four_primes_product_11_times_sum : 
  ¬ ∃ (a b c d : ℕ), 
    Prime a ∧ Prime b ∧ Prime c ∧ Prime d ∧
    (a * b * c * d = 11 * (a + b + c + d)) ∧
    ((a + b + c + d = 46) ∨ (a + b + c + d = 47) ∨ (a + b + c + d = 48)) :=
sorry

end NUMINAMATH_CALUDE_no_four_primes_product_11_times_sum_l2354_235497


namespace NUMINAMATH_CALUDE_probability_of_sum_seven_l2354_235468

def standard_die := Finset.range 6
def special_die := Finset.range 7

def sum_of_dice (a : ℕ) (b : ℕ) : ℕ :=
  a + if b = 6 then 0 else b + 1

def favorable_outcomes : Finset (ℕ × ℕ) :=
  (standard_die.product special_die).filter (λ p => sum_of_dice p.1 p.2 = 7)

theorem probability_of_sum_seven :
  (favorable_outcomes.card : ℚ) / ((standard_die.card * special_die.card) : ℚ) = 1 / 7 := by
  sorry

end NUMINAMATH_CALUDE_probability_of_sum_seven_l2354_235468


namespace NUMINAMATH_CALUDE_prob_multiple_of_3_twice_in_four_rolls_l2354_235489

/-- The probability of rolling a multiple of 3 on a fair six-sided die -/
def prob_multiple_of_3 : ℚ := 1 / 3

/-- The number of times the die is rolled -/
def num_rolls : ℕ := 4

/-- The number of times we want to see a multiple of 3 -/
def target_occurrences : ℕ := 2

/-- The probability of rolling a multiple of 3 exactly twice in four rolls of a fair die -/
theorem prob_multiple_of_3_twice_in_four_rolls :
  Nat.choose num_rolls target_occurrences * prob_multiple_of_3 ^ target_occurrences * (1 - prob_multiple_of_3) ^ (num_rolls - target_occurrences) = 8 / 27 := by
  sorry

end NUMINAMATH_CALUDE_prob_multiple_of_3_twice_in_four_rolls_l2354_235489


namespace NUMINAMATH_CALUDE_abs_inequality_solution_set_l2354_235415

def solution_set (x : ℝ) := -1 < x ∧ x < 0

theorem abs_inequality_solution_set :
  {x : ℝ | |2*x + 1| < 1} = {x : ℝ | solution_set x} := by sorry

end NUMINAMATH_CALUDE_abs_inequality_solution_set_l2354_235415


namespace NUMINAMATH_CALUDE_integer_solutions_l2354_235418

def is_integer (x : ℚ) : Prop := ∃ (n : ℤ), x = n

def expression_a (n : ℤ) : ℚ := (n^4 + 3) / (n^2 + n + 1)
def expression_b (n : ℤ) : ℚ := (n^3 + n + 1) / (n^2 - n + 1)

theorem integer_solutions :
  (∀ n : ℤ, is_integer (expression_a n) ↔ n = -3 ∨ n = -1 ∨ n = 0) ∧
  (∀ n : ℤ, is_integer (expression_b n) ↔ n = 0 ∨ n = 1) :=
sorry

end NUMINAMATH_CALUDE_integer_solutions_l2354_235418


namespace NUMINAMATH_CALUDE_girl_scout_cookie_sales_l2354_235434

theorem girl_scout_cookie_sales
  (total_boxes : ℕ)
  (total_value : ℚ)
  (choc_chip_price : ℚ)
  (plain_price : ℚ)
  (h1 : total_boxes = 1585)
  (h2 : total_value = 1586.75)
  (h3 : choc_chip_price = 1.25)
  (h4 : plain_price = 0.75) :
  ∃ (plain_boxes : ℕ) (choc_chip_boxes : ℕ),
    plain_boxes + choc_chip_boxes = total_boxes ∧
    plain_price * plain_boxes + choc_chip_price * choc_chip_boxes = total_value ∧
    plain_boxes = 789 :=
by sorry

end NUMINAMATH_CALUDE_girl_scout_cookie_sales_l2354_235434


namespace NUMINAMATH_CALUDE_function_inequality_l2354_235417

open Real

theorem function_inequality (f : ℝ → ℝ) (f' : ℝ → ℝ) (h : ∀ x ∈ (Set.Ioo 0 (π/2)), HasDerivAt f (f' x) x) :
  (∀ x ∈ (Set.Ioo 0 (π/2)), f' x * sin x - cos x * f x > 0) →
  Real.sqrt 3 * f (π/6) < f (π/3) := by
  sorry

end NUMINAMATH_CALUDE_function_inequality_l2354_235417


namespace NUMINAMATH_CALUDE_smartphone_price_difference_l2354_235486

def store_a_full_price : ℚ := 125
def store_a_discount : ℚ := 8 / 100
def store_b_full_price : ℚ := 130
def store_b_discount : ℚ := 10 / 100

theorem smartphone_price_difference :
  store_b_full_price * (1 - store_b_discount) - store_a_full_price * (1 - store_a_discount) = 2 := by
  sorry

end NUMINAMATH_CALUDE_smartphone_price_difference_l2354_235486


namespace NUMINAMATH_CALUDE_total_spears_l2354_235498

/-- The number of spears that can be made from a sapling -/
def spears_per_sapling : ℕ := 3

/-- The number of spears that can be made from a log -/
def spears_per_log : ℕ := 9

/-- The number of spears that can be made from a bundle of branches -/
def spears_per_bundle : ℕ := 7

/-- The number of spears that can be made from a large tree trunk -/
def spears_per_trunk : ℕ := 15

/-- The number of saplings Marcy has -/
def num_saplings : ℕ := 6

/-- The number of logs Marcy has -/
def num_logs : ℕ := 1

/-- The number of bundles of branches Marcy has -/
def num_bundles : ℕ := 3

/-- The number of large tree trunks Marcy has -/
def num_trunks : ℕ := 2

/-- Theorem stating the total number of spears Marcy can make -/
theorem total_spears : 
  num_saplings * spears_per_sapling + 
  num_logs * spears_per_log + 
  num_bundles * spears_per_bundle + 
  num_trunks * spears_per_trunk = 78 := by
  sorry


end NUMINAMATH_CALUDE_total_spears_l2354_235498


namespace NUMINAMATH_CALUDE_complement_A_intersect_integers_l2354_235414

def A : Set ℝ := {x | x ≤ -2 ∨ x ≥ 3}

theorem complement_A_intersect_integers :
  (Set.univ \ A) ∩ Set.range (Int.cast : ℤ → ℝ) = {-1, 0, 1, 2} := by sorry

end NUMINAMATH_CALUDE_complement_A_intersect_integers_l2354_235414


namespace NUMINAMATH_CALUDE_line_intersects_circle_l2354_235491

/-- The line y = k(x-2) + 4 intersects the curve y = √(4-x²) if and only if k ∈ [3/4, +∞) -/
theorem line_intersects_circle (k : ℝ) : 
  (∃ x y : ℝ, y = k * (x - 2) + 4 ∧ y = Real.sqrt (4 - x^2)) ↔ k ≥ 3/4 := by
  sorry

end NUMINAMATH_CALUDE_line_intersects_circle_l2354_235491


namespace NUMINAMATH_CALUDE_jamie_remaining_capacity_l2354_235407

/-- The maximum amount of liquid Jamie can consume before needing the bathroom -/
def max_liquid : ℕ := 32

/-- The amount of liquid Jamie has already consumed -/
def consumed_liquid : ℕ := 24

/-- The amount of additional liquid Jamie can consume -/
def remaining_capacity : ℕ := max_liquid - consumed_liquid

theorem jamie_remaining_capacity :
  remaining_capacity = 8 := by
  sorry

end NUMINAMATH_CALUDE_jamie_remaining_capacity_l2354_235407


namespace NUMINAMATH_CALUDE_angle_sum_in_square_configuration_l2354_235488

/-- Given a configuration of 13 identical squares with marked points, this theorem proves
    that the sum of specific angles equals 405 degrees. -/
theorem angle_sum_in_square_configuration :
  ∀ (FPB FPD APC APE AQG QCF RQF CQD : ℝ),
  RQF + CQD = 45 →
  FPB + FPD + APE = 180 →
  AQG + QCF + APC = 180 →
  (FPB + FPD + APC + APE) + (AQG + QCF + RQF + CQD) = 405 := by
  sorry

end NUMINAMATH_CALUDE_angle_sum_in_square_configuration_l2354_235488


namespace NUMINAMATH_CALUDE_license_plate_difference_l2354_235439

/-- The number of possible letters in a license plate position -/
def num_letters : ℕ := 26

/-- The number of possible digits in a license plate position -/
def num_digits : ℕ := 10

/-- The number of possible license plates for State A -/
def state_a_plates : ℕ := num_letters^5 * num_digits

/-- The number of possible license plates for State B -/
def state_b_plates : ℕ := num_letters^3 * num_digits^3

theorem license_plate_difference :
  state_a_plates - state_b_plates = 10123776 := by
  sorry

end NUMINAMATH_CALUDE_license_plate_difference_l2354_235439


namespace NUMINAMATH_CALUDE_factor_of_polynomial_l2354_235456

theorem factor_of_polynomial (x : ℝ) : 
  ∃ (q : ℝ → ℝ), (x^4 + 4 : ℝ) = (x^2 - 2*x + 2) * q x :=
sorry

end NUMINAMATH_CALUDE_factor_of_polynomial_l2354_235456


namespace NUMINAMATH_CALUDE_games_for_512_players_l2354_235452

/-- A single-elimination tournament with a given number of initial players. -/
structure SingleEliminationTournament where
  initial_players : ℕ
  initial_players_pos : initial_players > 0

/-- The number of games played in a single-elimination tournament. -/
def games_played (t : SingleEliminationTournament) : ℕ :=
  t.initial_players - 1

/-- Theorem stating that a single-elimination tournament with 512 initial players
    requires 511 games to determine the champion. -/
theorem games_for_512_players :
  ∀ (t : SingleEliminationTournament), t.initial_players = 512 → games_played t = 511 := by
  sorry

end NUMINAMATH_CALUDE_games_for_512_players_l2354_235452


namespace NUMINAMATH_CALUDE_randy_initial_amount_l2354_235405

/-- Represents Randy's piggy bank finances over a year -/
structure PiggyBank where
  initial_amount : ℕ
  monthly_deposit : ℕ
  store_visits : ℕ
  min_cost_per_visit : ℕ
  max_cost_per_visit : ℕ
  final_balance : ℕ

/-- Theorem stating that Randy's initial amount was $104 -/
theorem randy_initial_amount (pb : PiggyBank) 
  (h1 : pb.monthly_deposit = 50)
  (h2 : pb.store_visits = 200)
  (h3 : pb.min_cost_per_visit = 2)
  (h4 : pb.max_cost_per_visit = 3)
  (h5 : pb.final_balance = 104) :
  pb.initial_amount = 104 := by
  sorry

#check randy_initial_amount

end NUMINAMATH_CALUDE_randy_initial_amount_l2354_235405


namespace NUMINAMATH_CALUDE_power_equality_l2354_235462

theorem power_equality (k m : ℕ) 
  (h1 : 3 ^ (k - 1) = 9) 
  (h2 : 4 ^ (m + 2) = 64) : 
  2 ^ (3 * k + 2 * m) = 2 ^ 11 := by
  sorry

end NUMINAMATH_CALUDE_power_equality_l2354_235462


namespace NUMINAMATH_CALUDE_z_pure_imaginary_iff_m_eq_2013_l2354_235401

/-- A complex number z is pure imaginary if and only if its real part is zero and its imaginary part is non-zero. -/
def is_pure_imaginary (z : ℂ) : Prop :=
  z.re = 0 ∧ z.im ≠ 0

/-- The complex number z defined in terms of real number m. -/
def z (m : ℝ) : ℂ :=
  Complex.mk (m - 2013) (m - 1)

/-- Theorem stating that z is pure imaginary if and only if m = 2013. -/
theorem z_pure_imaginary_iff_m_eq_2013 :
    ∀ m : ℝ, is_pure_imaginary (z m) ↔ m = 2013 := by
  sorry

end NUMINAMATH_CALUDE_z_pure_imaginary_iff_m_eq_2013_l2354_235401


namespace NUMINAMATH_CALUDE_calculator_profit_l2354_235453

theorem calculator_profit : 
  let selling_price : ℝ := 64
  let profit_percentage : ℝ := 0.6
  let loss_percentage : ℝ := 0.2
  let cost_price1 : ℝ := selling_price / (1 + profit_percentage)
  let cost_price2 : ℝ := selling_price / (1 - loss_percentage)
  let total_cost : ℝ := cost_price1 + cost_price2
  let total_revenue : ℝ := 2 * selling_price
  total_revenue - total_cost = 8 := by
  sorry

end NUMINAMATH_CALUDE_calculator_profit_l2354_235453


namespace NUMINAMATH_CALUDE_profit_decrease_l2354_235426

theorem profit_decrease (march_profit : ℝ) (h1 : march_profit > 0) : 
  let april_profit := march_profit * 1.4
  let june_profit := march_profit * 1.68
  ∃ (may_profit : ℝ), 
    may_profit = april_profit * 0.8 ∧ 
    june_profit = may_profit * 1.5 := by
  sorry

end NUMINAMATH_CALUDE_profit_decrease_l2354_235426


namespace NUMINAMATH_CALUDE_max_product_of_three_integers_l2354_235495

theorem max_product_of_three_integers (a b c : ℕ+) : 
  (a * b * c = 8 * (a + b + c)) → (c = a + b) → (a * b * c ≤ 272) := by
  sorry

end NUMINAMATH_CALUDE_max_product_of_three_integers_l2354_235495


namespace NUMINAMATH_CALUDE_ellipse_properties_l2354_235477

/-- Given an ellipse defined by the equation 25x^2 + 9y^2 = 225, 
    this theorem proves its major axis length, minor axis length, and eccentricity. -/
theorem ellipse_properties : ∃ (a b c : ℝ),
  (∀ (x y : ℝ), 25 * x^2 + 9 * y^2 = 225 → x^2 / a^2 + y^2 / b^2 = 1) ∧
  2 * a = 10 ∧
  2 * b = 6 ∧
  c^2 = a^2 - b^2 ∧
  c / a = 0.8 := by
sorry

end NUMINAMATH_CALUDE_ellipse_properties_l2354_235477


namespace NUMINAMATH_CALUDE_hillarys_descending_rate_l2354_235410

/-- Proof of Hillary's descending rate on Mt. Everest --/
theorem hillarys_descending_rate 
  (total_distance : ℝ) 
  (hillary_climbing_rate : ℝ) 
  (eddy_climbing_rate : ℝ) 
  (hillary_stop_short : ℝ) 
  (total_time : ℝ) :
  total_distance = 4700 →
  hillary_climbing_rate = 800 →
  eddy_climbing_rate = 500 →
  hillary_stop_short = 700 →
  total_time = 6 →
  ∃ (hillary_descending_rate : ℝ),
    hillary_descending_rate = 1000 ∧
    hillary_descending_rate * (total_time - (total_distance - hillary_stop_short) / hillary_climbing_rate) = 
    (total_distance - hillary_stop_short) - (eddy_climbing_rate * total_time) :=
by sorry

end NUMINAMATH_CALUDE_hillarys_descending_rate_l2354_235410


namespace NUMINAMATH_CALUDE_percentage_same_grade_l2354_235499

def total_students : ℕ := 40
def students_with_all_As : ℕ := 3
def students_with_all_Bs : ℕ := 2
def students_with_all_Cs : ℕ := 6
def students_with_all_Ds : ℕ := 1

def students_with_same_grade : ℕ := 
  students_with_all_As + students_with_all_Bs + students_with_all_Cs + students_with_all_Ds

theorem percentage_same_grade : 
  (students_with_same_grade : ℚ) / total_students * 100 = 30 := by
  sorry

end NUMINAMATH_CALUDE_percentage_same_grade_l2354_235499


namespace NUMINAMATH_CALUDE_total_crayons_l2354_235492

/-- Given that each child has 8 crayons and there are 7 children, prove that the total number of crayons is 56. -/
theorem total_crayons (crayons_per_child : ℕ) (num_children : ℕ) 
  (h1 : crayons_per_child = 8) (h2 : num_children = 7) : 
  crayons_per_child * num_children = 56 := by
  sorry


end NUMINAMATH_CALUDE_total_crayons_l2354_235492


namespace NUMINAMATH_CALUDE_plant_growth_probability_l2354_235402

theorem plant_growth_probability (p_1m : ℝ) (p_2m : ℝ) 
  (h1 : p_1m = 0.8) 
  (h2 : p_2m = 0.4) : 
  p_2m / p_1m = 0.5 := by
  sorry

end NUMINAMATH_CALUDE_plant_growth_probability_l2354_235402


namespace NUMINAMATH_CALUDE_pencil_packs_l2354_235461

theorem pencil_packs (pencils_per_pack : ℕ) (pencils_per_row : ℕ) (num_rows : ℕ) : 
  pencils_per_pack = 24 →
  pencils_per_row = 16 →
  num_rows = 42 →
  (num_rows * pencils_per_row) / pencils_per_pack = 28 := by
sorry

end NUMINAMATH_CALUDE_pencil_packs_l2354_235461


namespace NUMINAMATH_CALUDE_set_inclusion_implies_a_bound_l2354_235435

-- Define set A
def A : Set ℝ := {x : ℝ | (4 : ℝ) / (x + 1) > 1}

-- Define set B (parameterized by a)
def B (a : ℝ) : Set ℝ := {x : ℝ | x^2 + 2*x - a^2 + 2*a < 0}

-- Theorem statement
theorem set_inclusion_implies_a_bound (a : ℝ) (h1 : a < 1) :
  (∀ x : ℝ, x ∈ A → x ∈ B a) → a ≤ -3 := by
  sorry

end NUMINAMATH_CALUDE_set_inclusion_implies_a_bound_l2354_235435


namespace NUMINAMATH_CALUDE_closest_point_l2354_235465

def v (t : ℝ) : Fin 3 → ℝ := fun i => 
  match i with
  | 0 => 2 + 7*t
  | 1 => -3 + 5*t
  | 2 => -3 - t

def a : Fin 3 → ℝ := fun i =>
  match i with
  | 0 => 4
  | 1 => 4
  | 2 => 5

def direction : Fin 3 → ℝ := fun i =>
  match i with
  | 0 => 7
  | 1 => 5
  | 2 => -1

theorem closest_point (t : ℝ) : 
  (∀ s : ℝ, ‖v t - a‖ ≤ ‖v s - a‖) ↔ t = 41/75 :=
sorry

end NUMINAMATH_CALUDE_closest_point_l2354_235465


namespace NUMINAMATH_CALUDE_final_temperature_is_correct_l2354_235441

/-- Calculates the final temperature after a series of adjustments --/
def finalTemperature (initial : ℝ) : ℝ :=
  let temp1 := initial * 2
  let temp2 := temp1 - 30
  let temp3 := temp2 * 0.7
  let temp4 := temp3 + 24
  let temp5 := temp4 * 0.9
  let temp6 := temp5 + 8
  let temp7 := temp6 * 1.2
  temp7 - 15

/-- Theorem stating that the final temperature is 58.32 degrees --/
theorem final_temperature_is_correct : 
  abs (finalTemperature 40 - 58.32) < 0.01 := by
  sorry

end NUMINAMATH_CALUDE_final_temperature_is_correct_l2354_235441


namespace NUMINAMATH_CALUDE_complex_product_pure_imaginary_l2354_235437

/-- A complex number z is pure imaginary if its real part is zero and its imaginary part is non-zero -/
def IsPureImaginary (z : ℂ) : Prop := (z.re = 0) ∧ (z.im ≠ 0)

/-- Given that b is a real number and (1+bi)(2+i) is a pure imaginary number, b equals 2 -/
theorem complex_product_pure_imaginary (b : ℝ) 
  (h : IsPureImaginary ((1 + b * Complex.I) * (2 + Complex.I))) : b = 2 := by
  sorry

end NUMINAMATH_CALUDE_complex_product_pure_imaginary_l2354_235437


namespace NUMINAMATH_CALUDE_matthew_friends_count_l2354_235496

def total_crackers : ℝ := 36
def crackers_per_friend : ℝ := 6.5

theorem matthew_friends_count :
  ⌊total_crackers / crackers_per_friend⌋ = 5 :=
by sorry

end NUMINAMATH_CALUDE_matthew_friends_count_l2354_235496


namespace NUMINAMATH_CALUDE_simplify_expression_l2354_235475

theorem simplify_expression (x : ℝ) : (2*x + 20) + (150*x + 20) = 152*x + 40 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l2354_235475


namespace NUMINAMATH_CALUDE_kim_initial_classes_l2354_235424

def initial_classes (class_duration : ℕ) (dropped_classes : ℕ) (total_hours : ℕ) : ℕ :=
  (total_hours / class_duration) + dropped_classes

theorem kim_initial_classes :
  initial_classes 2 1 6 = 4 := by
  sorry

end NUMINAMATH_CALUDE_kim_initial_classes_l2354_235424


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l2354_235423

def set_A : Set ℝ := {x | -1 ≤ x ∧ x ≤ 2}

def set_B : Set ℤ := {x | x^2 - 3*x - 4 < 0}

theorem intersection_of_A_and_B : set_A ∩ (set_B.image (coe : ℤ → ℝ)) = {0, 1, 2} := by
  sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l2354_235423


namespace NUMINAMATH_CALUDE_cubic_equation_root_l2354_235474

theorem cubic_equation_root (a b : ℚ) : 
  (3 - 5 * Real.sqrt 2)^3 + a * (3 - 5 * Real.sqrt 2)^2 + b * (3 - 5 * Real.sqrt 2) - 47 = 0 → 
  a = -199/41 := by
sorry

end NUMINAMATH_CALUDE_cubic_equation_root_l2354_235474


namespace NUMINAMATH_CALUDE_luke_gave_five_stickers_l2354_235416

/-- Calculates the number of stickers Luke gave to his sister -/
def stickers_given_to_sister (initial : ℕ) (bought : ℕ) (birthday : ℕ) (used : ℕ) (left : ℕ) : ℕ :=
  initial + bought + birthday - used - left

/-- Proves that Luke gave 5 stickers to his sister -/
theorem luke_gave_five_stickers :
  stickers_given_to_sister 20 12 20 8 39 = 5 := by
  sorry

#eval stickers_given_to_sister 20 12 20 8 39

end NUMINAMATH_CALUDE_luke_gave_five_stickers_l2354_235416


namespace NUMINAMATH_CALUDE_pavan_travel_distance_l2354_235430

theorem pavan_travel_distance :
  ∀ (total_distance : ℝ),
  (total_distance / 2 / 30 + total_distance / 2 / 25 = 11) →
  total_distance = 150 := by
sorry

end NUMINAMATH_CALUDE_pavan_travel_distance_l2354_235430


namespace NUMINAMATH_CALUDE_triangle_base_length_l2354_235480

/-- Theorem: The base of a triangle with specific side lengths -/
theorem triangle_base_length (left_side right_side base : ℝ) : 
  left_side = 12 →
  right_side = left_side + 2 →
  left_side + right_side + base = 50 →
  base = 24 := by
sorry

end NUMINAMATH_CALUDE_triangle_base_length_l2354_235480


namespace NUMINAMATH_CALUDE_no_infinite_prime_sequence_l2354_235428

theorem no_infinite_prime_sequence :
  ¬ ∃ (p : ℕ → ℕ), (∀ k, p (k + 1) = 5 * p k + 4) ∧ (∀ n, Nat.Prime (p n)) :=
by sorry

end NUMINAMATH_CALUDE_no_infinite_prime_sequence_l2354_235428


namespace NUMINAMATH_CALUDE_similar_triangles_side_ratio_l2354_235460

theorem similar_triangles_side_ratio 
  (a b ka kb : ℝ) 
  (C : Real) 
  (k : ℝ) 
  (h1 : ka = k * a) 
  (h2 : kb = k * b) 
  (h3 : C > 0 ∧ C < 180) : 
  ∃ (c kc : ℝ), c > 0 ∧ kc > 0 ∧ kc = k * c :=
sorry

end NUMINAMATH_CALUDE_similar_triangles_side_ratio_l2354_235460


namespace NUMINAMATH_CALUDE_inequality_proof_l2354_235484

theorem inequality_proof (x y : ℝ) (h : x^4 + y^4 ≤ 1) : x^6 - y^6 + 2*y^3 < π/2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l2354_235484


namespace NUMINAMATH_CALUDE_loss_percentage_proof_l2354_235470

def cost_price : ℝ := 1250
def price_increase : ℝ := 500
def gain_percentage : ℝ := 0.15

theorem loss_percentage_proof (selling_price : ℝ) 
  (h1 : selling_price + price_increase = cost_price * (1 + gain_percentage)) :
  (cost_price - selling_price) / cost_price = 0.25 := by
  sorry

end NUMINAMATH_CALUDE_loss_percentage_proof_l2354_235470


namespace NUMINAMATH_CALUDE_intersection_distance_l2354_235420

-- Define the ellipse
def ellipse (x y : ℝ) : Prop := x^2 / 36 + y^2 / 16 = 1

-- Define the parabola (using the derived equation from the solution)
def parabola (x y : ℝ) : Prop := x = y^2 / (4 * Real.sqrt 5) + Real.sqrt 5

-- Define the intersection points
def intersection_points : Set (ℝ × ℝ) :=
  {p | ellipse p.1 p.2 ∧ parabola p.1 p.2}

-- Theorem statement
theorem intersection_distance :
  ∃ p1 p2 : ℝ × ℝ, p1 ∈ intersection_points ∧ p2 ∈ intersection_points ∧
  p1 ≠ p2 ∧ Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2) = 2 * Real.sqrt 10 :=
sorry

end NUMINAMATH_CALUDE_intersection_distance_l2354_235420


namespace NUMINAMATH_CALUDE_trigonometric_calculations_l2354_235419

theorem trigonometric_calculations :
  (((Real.pi - 2) ^ 0 - |1 - Real.tan (60 * Real.pi / 180)| - (1/2)⁻¹ + 6 / Real.sqrt 3) = Real.sqrt 3) ∧
  ((Real.sin (45 * Real.pi / 180) - Real.cos (30 * Real.pi / 180) * Real.tan (60 * Real.pi / 180)) = (Real.sqrt 2 - 3) / 2) :=
by sorry

end NUMINAMATH_CALUDE_trigonometric_calculations_l2354_235419


namespace NUMINAMATH_CALUDE_quadratic_roots_difference_l2354_235446

theorem quadratic_roots_difference (a b c : ℝ) (h : a ≠ 0) :
  let r₁ := (-b + Real.sqrt (b^2 - 4*a*c)) / (2*a)
  let r₂ := (-b - Real.sqrt (b^2 - 4*a*c)) / (2*a)
  a * X^2 + b * X + c = 0 → |r₁ - r₂| = 3 :=
by
  sorry

#check quadratic_roots_difference 1 (-7) 10

end NUMINAMATH_CALUDE_quadratic_roots_difference_l2354_235446


namespace NUMINAMATH_CALUDE_harvest_difference_l2354_235493

theorem harvest_difference (apples peaches pears : ℕ) : 
  apples = 60 →
  peaches = 3 * apples →
  pears = apples / 2 →
  (apples + peaches) - pears = 210 := by
  sorry

end NUMINAMATH_CALUDE_harvest_difference_l2354_235493


namespace NUMINAMATH_CALUDE_cubic_sum_theorem_l2354_235459

theorem cubic_sum_theorem (a b c : ℝ) 
  (sum_cond : a + b + c = 2)
  (prod_sum_cond : a * b + a * c + b * c = -3)
  (prod_cond : a * b * c = -3) :
  a^3 + b^3 + c^3 = 6 := by
  sorry

end NUMINAMATH_CALUDE_cubic_sum_theorem_l2354_235459


namespace NUMINAMATH_CALUDE_average_of_quadratic_roots_l2354_235421

theorem average_of_quadratic_roots (c : ℝ) 
  (h : ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ 3 * x₁^2 - 6 * x₁ + c = 0 ∧ 3 * x₂^2 - 6 * x₂ + c = 0) :
  ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ 
    3 * x₁^2 - 6 * x₁ + c = 0 ∧ 
    3 * x₂^2 - 6 * x₂ + c = 0 ∧
    (x₁ + x₂) / 2 = 1 :=
by sorry

end NUMINAMATH_CALUDE_average_of_quadratic_roots_l2354_235421


namespace NUMINAMATH_CALUDE_inequality_proof_l2354_235457

theorem inequality_proof (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a + b = 1) :
  (2 * a^2 + 3 * b^2 ≥ 6/5) ∧ ((a + 1/a) * (b + 1/b) ≥ 25/4) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l2354_235457


namespace NUMINAMATH_CALUDE_smallest_integer_multiple_conditions_l2354_235458

theorem smallest_integer_multiple_conditions :
  ∃ n : ℕ, n > 0 ∧
  (∃ k : ℤ, n = 5 * k + 3) ∧
  (∃ m : ℤ, n = 12 * m) ∧
  (∀ x : ℕ, x > 0 →
    (∃ k' : ℤ, x = 5 * k' + 3) →
    (∃ m' : ℤ, x = 12 * m') →
    n ≤ x) ∧
  n = 48 :=
by sorry

end NUMINAMATH_CALUDE_smallest_integer_multiple_conditions_l2354_235458


namespace NUMINAMATH_CALUDE_system_sum_l2354_235438

theorem system_sum (x y z : ℝ) 
  (eq1 : x + y = 4)
  (eq2 : y + z = 6)
  (eq3 : z + x = 8) :
  x + y + z = 9 := by
  sorry

end NUMINAMATH_CALUDE_system_sum_l2354_235438


namespace NUMINAMATH_CALUDE_solve_system_l2354_235449

theorem solve_system (w u y z x : ℤ) 
  (hw : w = 100)
  (hz : z = w + 25)
  (hy : y = z + 12)
  (hu : u = y + 5)
  (hx : x = u + 7) : x = 149 := by
  sorry

end NUMINAMATH_CALUDE_solve_system_l2354_235449
