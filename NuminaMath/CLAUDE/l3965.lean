import Mathlib

namespace NUMINAMATH_CALUDE_complement_A_in_U_l3965_396560

-- Define the universal set U
def U : Set ℝ := {x | x > 1}

-- Define set A
def A : Set ℝ := {x | x ≥ 2}

-- State the theorem
theorem complement_A_in_U : 
  (U \ A) = {x : ℝ | 1 < x ∧ x < 2} := by sorry

end NUMINAMATH_CALUDE_complement_A_in_U_l3965_396560


namespace NUMINAMATH_CALUDE_factorization_problem1_l3965_396520

theorem factorization_problem1 (x y : ℚ) : x^2 * y - 4 * x * y = x * y * (x - 4) := by sorry

end NUMINAMATH_CALUDE_factorization_problem1_l3965_396520


namespace NUMINAMATH_CALUDE_robert_nickel_difference_l3965_396525

/-- Represents the number of chocolates eaten by each person -/
structure Chocolates where
  sarah : ℕ
  nickel : ℕ
  robert : ℕ

/-- The chocolate eating scenario -/
def chocolate_scenario : Chocolates :=
  { sarah := 15,
    nickel := 15 - 5,
    robert := 2 * (15 - 5) }

/-- Theorem stating the difference between Robert's and Nickel's chocolates -/
theorem robert_nickel_difference :
  chocolate_scenario.robert - chocolate_scenario.nickel = 10 := by
  sorry

end NUMINAMATH_CALUDE_robert_nickel_difference_l3965_396525


namespace NUMINAMATH_CALUDE_cylinder_from_constant_rho_l3965_396541

/-- Cylindrical coordinates -/
structure CylindricalCoord where
  ρ : ℝ
  φ : ℝ
  z : ℝ

/-- A set of points in cylindrical coordinates -/
def CylindricalSet (c : ℝ) : Set CylindricalCoord :=
  {p : CylindricalCoord | p.ρ = c}

/-- Definition of a cylinder in cylindrical coordinates -/
def IsCylinder (S : Set CylindricalCoord) : Prop :=
  ∃ c > 0, S = CylindricalSet c

/-- Theorem: The set of points satisfying ρ = c forms a cylinder -/
theorem cylinder_from_constant_rho (c : ℝ) (hc : c > 0) :
  IsCylinder (CylindricalSet c) := by
  sorry


end NUMINAMATH_CALUDE_cylinder_from_constant_rho_l3965_396541


namespace NUMINAMATH_CALUDE_first_divisor_problem_l3965_396593

theorem first_divisor_problem (y : ℝ) (h : (320 / y) / 3 = 53.33) : y = 2 := by
  sorry

end NUMINAMATH_CALUDE_first_divisor_problem_l3965_396593


namespace NUMINAMATH_CALUDE_truck_distance_truck_distance_proof_l3965_396563

/-- The distance traveled by a truck given specific conditions -/
theorem truck_distance (truck_time car_time : ℝ) 
  (speed_difference distance_difference : ℝ) : ℝ :=
  let truck_speed := (car_time * speed_difference + distance_difference) / (car_time - truck_time)
  truck_speed * truck_time

/-- Prove that the truck travels 296 km under the given conditions -/
theorem truck_distance_proof : 
  truck_distance 8 5.5 18 6.5 = 296 := by
  sorry

end NUMINAMATH_CALUDE_truck_distance_truck_distance_proof_l3965_396563


namespace NUMINAMATH_CALUDE_A_minus_2B_A_minus_2B_special_case_A_minus_2B_independent_of_x_l3965_396575

-- Define the expressions A and B
def A (x y : ℝ) : ℝ := 2 * x^2 + 3 * x * y + 2 * y
def B (x y : ℝ) : ℝ := x^2 - x * y + x

-- Theorem 1: A - 2B = 5xy - 2x + 2y
theorem A_minus_2B (x y : ℝ) : A x y - 2 * B x y = 5 * x * y - 2 * x + 2 * y := by sorry

-- Theorem 2: When x² = 9 and |y| = 2, A - 2B ∈ {28, -40, -20, 32}
theorem A_minus_2B_special_case (x y : ℝ) (h1 : x^2 = 9) (h2 : |y| = 2) :
  A x y - 2 * B x y ∈ ({28, -40, -20, 32} : Set ℝ) := by sorry

-- Theorem 3: If A - 2B is independent of x, then y = 2/5
theorem A_minus_2B_independent_of_x (y : ℝ) :
  (∀ x : ℝ, A x y - 2 * B x y = A 0 y - 2 * B 0 y) → y = 2/5 := by sorry

end NUMINAMATH_CALUDE_A_minus_2B_A_minus_2B_special_case_A_minus_2B_independent_of_x_l3965_396575


namespace NUMINAMATH_CALUDE_stone_slab_floor_area_l3965_396535

/-- Calculates the total floor area covered by square stone slabs -/
theorem stone_slab_floor_area 
  (num_slabs : ℕ) 
  (slab_length : ℝ) 
  (h_num_slabs : num_slabs = 30)
  (h_slab_length : slab_length = 140) : 
  (num_slabs * slab_length^2) / 10000 = 58.8 := by
  sorry

end NUMINAMATH_CALUDE_stone_slab_floor_area_l3965_396535


namespace NUMINAMATH_CALUDE_nuts_cost_correct_l3965_396584

/-- The cost of nuts per kilogram -/
def cost_of_nuts : ℝ := 12

/-- The cost of dried fruits per kilogram -/
def cost_of_dried_fruits : ℝ := 8

/-- The amount of nuts bought in kilograms -/
def amount_of_nuts : ℝ := 3

/-- The amount of dried fruits bought in kilograms -/
def amount_of_dried_fruits : ℝ := 2.5

/-- The total cost of the purchase -/
def total_cost : ℝ := 56

theorem nuts_cost_correct : 
  cost_of_nuts * amount_of_nuts + cost_of_dried_fruits * amount_of_dried_fruits = total_cost :=
by sorry

end NUMINAMATH_CALUDE_nuts_cost_correct_l3965_396584


namespace NUMINAMATH_CALUDE_hyperbola_parabola_intersection_l3965_396511

-- Define the hyperbola and parabola
def hyperbola (a b x y : ℝ) : Prop := x^2 / a^2 - y^2 / b^2 = 1

def parabola (x y : ℝ) : Prop := y^2 = 4 * x

-- Define the latus rectum of the parabola
def latus_rectum (x : ℝ) : Prop := x = -1

-- Define the length of the line segment
def line_segment_length (b y : ℝ) : Prop := 2 * y = b

-- Main theorem
theorem hyperbola_parabola_intersection (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (∀ x y : ℝ, hyperbola a b x y ∧ parabola x y ∧ latus_rectum x ∧ line_segment_length b y) →
  a = 2 * Real.sqrt 5 / 5 :=
sorry

end NUMINAMATH_CALUDE_hyperbola_parabola_intersection_l3965_396511


namespace NUMINAMATH_CALUDE_reinforcement_size_correct_l3965_396513

/-- Calculates the size of reinforcement given initial garrison size, initial provision duration,
    days before reinforcement, and remaining provision duration after reinforcement. -/
def reinforcement_size (initial_garrison : ℕ) (initial_duration : ℕ) (days_before_reinforcement : ℕ) (remaining_duration : ℕ) : ℕ :=
  let remaining_provisions := initial_garrison * (initial_duration - days_before_reinforcement)
  let total_men_days := remaining_provisions
  (total_men_days / remaining_duration) - initial_garrison

theorem reinforcement_size_correct (initial_garrison : ℕ) (initial_duration : ℕ) (days_before_reinforcement : ℕ) (remaining_duration : ℕ) 
    (h1 : initial_garrison = 2000)
    (h2 : initial_duration = 54)
    (h3 : days_before_reinforcement = 15)
    (h4 : remaining_duration = 20) :
  reinforcement_size initial_garrison initial_duration days_before_reinforcement remaining_duration = 1900 := by
  sorry

#eval reinforcement_size 2000 54 15 20

end NUMINAMATH_CALUDE_reinforcement_size_correct_l3965_396513


namespace NUMINAMATH_CALUDE_eggs_per_basket_l3965_396523

theorem eggs_per_basket (blue_eggs : Nat) (yellow_eggs : Nat) (min_eggs : Nat)
  (h1 : blue_eggs = 30)
  (h2 : yellow_eggs = 42)
  (h3 : min_eggs = 6) :
  ∃ (x : Nat), x ≥ min_eggs ∧ x ∣ blue_eggs ∧ x ∣ yellow_eggs ∧
    ∀ (y : Nat), y > x → ¬(y ∣ blue_eggs ∧ y ∣ yellow_eggs) :=
by
  sorry

end NUMINAMATH_CALUDE_eggs_per_basket_l3965_396523


namespace NUMINAMATH_CALUDE_problem_statement_l3965_396599

theorem problem_statement (x y : ℝ) 
  (h1 : (x - y) / (x + y) = 9)
  (h2 : x * y / (x + y) = -60) :
  (x + y) + (x - y) + x * y = -150 := by sorry

end NUMINAMATH_CALUDE_problem_statement_l3965_396599


namespace NUMINAMATH_CALUDE_max_sum_of_distances_l3965_396587

/-- Triangle ABC is a right triangle with ∠ABC = 90°, AC = 10, AB = 8, BC = 6 -/
structure RightTriangle where
  AC : ℝ
  AB : ℝ
  BC : ℝ
  right_angle : AC^2 = AB^2 + BC^2
  AC_eq : AC = 10
  AB_eq : AB = 8
  BC_eq : BC = 6

/-- Point on triangle A'B'C'' -/
structure PointOnTriangleABC' where
  x : ℝ
  y : ℝ
  on_triangle : 0 ≤ x ∧ 0 ≤ y ∧ x + y ≤ 1

/-- Sum of distances from a point to the sides of triangle ABC -/
def sum_of_distances (t : RightTriangle) (p : PointOnTriangleABC') : ℝ :=
  p.x * t.AB + p.y * t.BC + 1

/-- Maximum sum of distances theorem -/
theorem max_sum_of_distances (t : RightTriangle) :
  ∃ (max : ℝ), max = 7 ∧ ∀ (p : PointOnTriangleABC'), sum_of_distances t p ≤ max :=
sorry

end NUMINAMATH_CALUDE_max_sum_of_distances_l3965_396587


namespace NUMINAMATH_CALUDE_palindrome_product_sum_theorem_l3965_396548

/-- A positive three-digit palindrome is a number between 100 and 999 inclusive,
    where the first and third digits are the same. -/
def IsPositiveThreeDigitPalindrome (n : ℕ) : Prop :=
  100 ≤ n ∧ n ≤ 999 ∧ (n / 100 = n % 10)

/-- There exist two positive three-digit palindromes whose product is 589185 and whose sum is 1534. -/
theorem palindrome_product_sum_theorem :
  ∃ (a b : ℕ), IsPositiveThreeDigitPalindrome a ∧
                IsPositiveThreeDigitPalindrome b ∧
                a * b = 589185 ∧
                a + b = 1534 := by
  sorry

end NUMINAMATH_CALUDE_palindrome_product_sum_theorem_l3965_396548


namespace NUMINAMATH_CALUDE_clips_sold_and_average_earning_l3965_396592

/-- Calculates the total number of clips sold and average earning per clip -/
theorem clips_sold_and_average_earning 
  (x : ℝ) -- number of clips sold in April
  (y : ℝ) -- number of clips sold in May
  (z : ℝ) -- number of clips sold in June
  (W : ℝ) -- total earnings
  (h1 : y = x / 2) -- May sales condition
  (h2 : z = y + 0.25 * y) -- June sales condition
  : (x + y + z = 2.125 * x) ∧ (W / (x + y + z) = W / (2.125 * x)) := by
  sorry

end NUMINAMATH_CALUDE_clips_sold_and_average_earning_l3965_396592


namespace NUMINAMATH_CALUDE_equation_solution_l3965_396568

theorem equation_solution : ∃ y : ℚ, y + 5/8 = 2/9 + 1/2 ∧ y = 7/72 := by sorry

end NUMINAMATH_CALUDE_equation_solution_l3965_396568


namespace NUMINAMATH_CALUDE_largest_n_satisfying_inequality_l3965_396564

theorem largest_n_satisfying_inequality :
  ∀ n : ℤ, (1 : ℚ) / 3 + (n : ℚ) / 7 < 1 ↔ n ≤ 4 :=
by sorry

end NUMINAMATH_CALUDE_largest_n_satisfying_inequality_l3965_396564


namespace NUMINAMATH_CALUDE_mikes_age_l3965_396590

theorem mikes_age (mike anna : ℝ) 
  (h1 : mike = 3 * anna - 20)
  (h2 : mike + anna = 70) : 
  mike = 47.5 := by
sorry

end NUMINAMATH_CALUDE_mikes_age_l3965_396590


namespace NUMINAMATH_CALUDE_cube_root_54880000_l3965_396528

theorem cube_root_54880000 : 
  (Real.rpow 54880000 (1/3 : ℝ)) = 140 * Real.rpow 2 (1/3 : ℝ) := by sorry

end NUMINAMATH_CALUDE_cube_root_54880000_l3965_396528


namespace NUMINAMATH_CALUDE_complex_roots_theorem_l3965_396591

theorem complex_roots_theorem (a b : ℝ) : 
  (Complex.I : ℂ) ^ 2 = -1 →
  (a + 4 * Complex.I) * (a + 4 * Complex.I) - (10 + 9 * Complex.I) * (a + 4 * Complex.I) + (4 + 46 * Complex.I) = 0 →
  (b + 5 * Complex.I) * (b + 5 * Complex.I) - (10 + 9 * Complex.I) * (b + 5 * Complex.I) + (4 + 46 * Complex.I) = 0 →
  a = 6 ∧ b = 4 := by
sorry

end NUMINAMATH_CALUDE_complex_roots_theorem_l3965_396591


namespace NUMINAMATH_CALUDE_simplified_expression_equals_zero_l3965_396543

theorem simplified_expression_equals_zero (x y : ℝ) (hx : x > 0) (hy : y > 0) 
  (h : x^3 + y^3 = x + y) : x/y + y/x - 2/(x*y) = 0 := by
  sorry

end NUMINAMATH_CALUDE_simplified_expression_equals_zero_l3965_396543


namespace NUMINAMATH_CALUDE_shortest_reflected_light_path_l3965_396559

/-- The shortest path length for a reflected light ray -/
theorem shortest_reflected_light_path :
  let A : ℝ × ℝ := (-3, 9)
  let C : Set (ℝ × ℝ) := {p | (p.1 - 2)^2 + (p.2 - 3)^2 = 1}
  let reflect_point (p : ℝ × ℝ) : ℝ × ℝ := (p.1, -p.2)
  let dist (p q : ℝ × ℝ) : ℝ := Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)
  ∃ (p : ℝ × ℝ),
    p.2 = 0 ∧
    p ∉ C ∧
    (∀ (q : ℝ × ℝ), q.2 = 0 ∧ q ∉ C →
      dist A p + dist p (reflect_point (2, 3)) ≤ dist A q + dist q (reflect_point (2, 3))) ∧
    dist A p + dist p (reflect_point (2, 3)) = 12 :=
by sorry

end NUMINAMATH_CALUDE_shortest_reflected_light_path_l3965_396559


namespace NUMINAMATH_CALUDE_complex_square_simplification_l3965_396572

theorem complex_square_simplification :
  let i : ℂ := Complex.I
  (4 - 3 * i)^2 = 7 - 24 * i :=
by sorry

end NUMINAMATH_CALUDE_complex_square_simplification_l3965_396572


namespace NUMINAMATH_CALUDE_problem_proof_l3965_396565

theorem problem_proof : ((12^12 / 12^11)^2 * 4^2) / 2^4 = 144 := by
  sorry

end NUMINAMATH_CALUDE_problem_proof_l3965_396565


namespace NUMINAMATH_CALUDE_three_digit_number_divisible_by_6_l3965_396527

def is_divisible_by_6 (n : ℕ) : Prop := n % 6 = 0

theorem three_digit_number_divisible_by_6 (n : ℕ) (h1 : n ≥ 500 ∧ n < 600) 
  (h2 : n % 10 = 2) (h3 : is_divisible_by_6 n) : 
  n ≥ 100 ∧ n < 1000 :=
sorry

end NUMINAMATH_CALUDE_three_digit_number_divisible_by_6_l3965_396527


namespace NUMINAMATH_CALUDE_water_removal_for_concentration_l3965_396504

theorem water_removal_for_concentration (initial_volume : ℝ) (final_concentration : ℝ) 
  (water_removed : ℝ) : 
  initial_volume = 21 ∧ 
  final_concentration = 60 ∧ 
  water_removed = 7 → 
  water_removed = initial_volume - (initial_volume * (initial_volume * final_concentration) / 
    (100 * (initial_volume - water_removed))) / 100 :=
by sorry

end NUMINAMATH_CALUDE_water_removal_for_concentration_l3965_396504


namespace NUMINAMATH_CALUDE_quadratic_inequality_l3965_396532

def f (x : ℝ) : ℝ := x^2 - 4*x + 4

theorem quadratic_inequality (x₁ x₂ y₁ y₂ : ℝ) 
  (h₁ : 1 < x₁) (h₂ : x₁ < 2) (h₃ : 3 < x₂) (h₄ : x₂ < 4)
  (hy₁ : y₁ = f x₁) (hy₂ : y₂ = f x₂) : y₁ < y₂ := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_l3965_396532


namespace NUMINAMATH_CALUDE_lara_miles_walked_l3965_396556

/-- Represents a pedometer with a maximum count before resetting --/
structure Pedometer where
  max_count : ℕ
  reset_count : ℕ
  final_reading : ℕ
  steps_per_mile : ℕ

/-- Calculates the total steps walked based on pedometer data --/
def total_steps (p : Pedometer) : ℕ :=
  p.reset_count * (p.max_count + 1) + p.final_reading

/-- Calculates the approximate miles walked based on total steps --/
def miles_walked (p : Pedometer) : ℕ :=
  (total_steps p + p.steps_per_mile - 1) / p.steps_per_mile

/-- Theorem stating the approximate miles walked --/
theorem lara_miles_walked (p : Pedometer) 
  (h1 : p.max_count = 99999)
  (h2 : p.reset_count = 52)
  (h3 : p.final_reading = 38200)
  (h4 : p.steps_per_mile = 2000) :
  miles_walked p = 2619 := by
  sorry

end NUMINAMATH_CALUDE_lara_miles_walked_l3965_396556


namespace NUMINAMATH_CALUDE_c_can_be_any_real_l3965_396574

theorem c_can_be_any_real (a b c d : ℝ) (h1 : b ≠ 0) (h2 : d ≠ 0) (h3 : a / b + c / d < 0) :
  ∃ (a' b' d' : ℝ) (h1' : b' ≠ 0) (h2' : d' ≠ 0),
    (∀ c' : ℝ, ∃ (h3' : a' / b' + c' / d' < 0), True) :=
sorry

end NUMINAMATH_CALUDE_c_can_be_any_real_l3965_396574


namespace NUMINAMATH_CALUDE_condition_necessary_not_sufficient_l3965_396586

/-- A sequence of real numbers -/
def Sequence := ℕ → ℝ

/-- The condition that a_n^2 = a_(n-1) * a_(n+1) for n ≥ 2 -/
def Condition (a : Sequence) : Prop :=
  ∀ n : ℕ, n ≥ 2 → a n ^ 2 = a (n - 1) * a (n + 1)

/-- Definition of a geometric sequence -/
def IsGeometric (a : Sequence) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

/-- The main theorem stating that the condition is necessary but not sufficient -/
theorem condition_necessary_not_sufficient :
  (∀ a : Sequence, IsGeometric a → Condition a) ∧
  ¬(∀ a : Sequence, Condition a → IsGeometric a) :=
by sorry

end NUMINAMATH_CALUDE_condition_necessary_not_sufficient_l3965_396586


namespace NUMINAMATH_CALUDE_soda_price_proof_l3965_396507

/-- Given a regular price per can of soda, prove that it equals $0.55 under the given conditions -/
theorem soda_price_proof (P : ℝ) : 
  (∃ (discounted_price : ℝ), 
    discounted_price = 0.75 * P ∧ 
    70 * discounted_price = 28.875) → 
  P = 0.55 := by
sorry

end NUMINAMATH_CALUDE_soda_price_proof_l3965_396507


namespace NUMINAMATH_CALUDE_average_weight_increase_l3965_396538

theorem average_weight_increase (initial_count : ℕ) (old_weight new_weight : ℝ) :
  initial_count = 8 →
  old_weight = 62 →
  new_weight = 90 →
  (new_weight - old_weight) / initial_count = 3.5 :=
by
  sorry

end NUMINAMATH_CALUDE_average_weight_increase_l3965_396538


namespace NUMINAMATH_CALUDE_equation_solution_l3965_396537

theorem equation_solution :
  let f : ℝ → ℝ := λ x => (4*x+1)*(3*x+1)*(2*x+1)*(x+1) - 3*x^4
  ∀ x : ℝ, f x = 0 ↔ x = (-5 + Real.sqrt 13) / 6 ∨ x = (-5 - Real.sqrt 13) / 6 := by
sorry

end NUMINAMATH_CALUDE_equation_solution_l3965_396537


namespace NUMINAMATH_CALUDE_circles_internally_tangent_l3965_396583

/-- Two circles are internally tangent if the distance between their centers
    is equal to the absolute difference of their radii -/
def internally_tangent (c1 c2 : ℝ × ℝ) (r1 r2 : ℝ) : Prop :=
  (c1.1 - c2.1)^2 + (c1.2 - c2.2)^2 = (r1 - r2)^2

/-- The equation of the first circle: x^2 + y^2 - 2x = 0 -/
def circle1 (x y : ℝ) : Prop := x^2 + y^2 - 2*x = 0

/-- The equation of the second circle: x^2 + y^2 - 2x - 6y - 6 = 0 -/
def circle2 (x y : ℝ) : Prop := x^2 + y^2 - 2*x - 6*y - 6 = 0

theorem circles_internally_tangent :
  ∃ (c1 c2 : ℝ × ℝ) (r1 r2 : ℝ),
    (∀ x y, circle1 x y ↔ (x - c1.1)^2 + (y - c1.2)^2 = r1^2) ∧
    (∀ x y, circle2 x y ↔ (x - c2.1)^2 + (y - c2.2)^2 = r2^2) ∧
    internally_tangent c1 c2 r1 r2 :=
  sorry

end NUMINAMATH_CALUDE_circles_internally_tangent_l3965_396583


namespace NUMINAMATH_CALUDE_units_digit_of_product_l3965_396544

def units_digit (n : ℕ) : ℕ := n % 10

theorem units_digit_of_product :
  units_digit (17 * 28) = 6 := by
  sorry

end NUMINAMATH_CALUDE_units_digit_of_product_l3965_396544


namespace NUMINAMATH_CALUDE_parallelogram_height_l3965_396547

theorem parallelogram_height (area base height : ℝ) : 
  area = 364 ∧ base = 26 ∧ area = base * height → height = 14 := by
  sorry

end NUMINAMATH_CALUDE_parallelogram_height_l3965_396547


namespace NUMINAMATH_CALUDE_calculate_expression_l3965_396582

theorem calculate_expression : (3.6 * 0.5) / 0.2 = 9 := by
  sorry

end NUMINAMATH_CALUDE_calculate_expression_l3965_396582


namespace NUMINAMATH_CALUDE_car_wash_group_composition_l3965_396508

theorem car_wash_group_composition (total : ℕ) (initial_girls : ℕ) : 
  (initial_girls : ℚ) / total = 2 / 5 →
  ((initial_girls : ℚ) - 2) / total = 3 / 10 →
  initial_girls = 8 := by
sorry

end NUMINAMATH_CALUDE_car_wash_group_composition_l3965_396508


namespace NUMINAMATH_CALUDE_chocolate_bars_l3965_396579

theorem chocolate_bars (cost_per_bar : ℕ) (remaining_bars : ℕ) (revenue : ℕ) :
  cost_per_bar = 4 →
  remaining_bars = 3 →
  revenue = 20 →
  ∃ total_bars : ℕ, total_bars = 8 ∧ cost_per_bar * (total_bars - remaining_bars) = revenue :=
by
  sorry

end NUMINAMATH_CALUDE_chocolate_bars_l3965_396579


namespace NUMINAMATH_CALUDE_sqrt_720_simplification_l3965_396501

theorem sqrt_720_simplification : Real.sqrt 720 = 12 * Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_720_simplification_l3965_396501


namespace NUMINAMATH_CALUDE_even_and_divisible_by_six_l3965_396517

theorem even_and_divisible_by_six (n : ℕ) : 
  (2 ∣ n * (n + 1)) ∧ (6 ∣ n * (n + 1) * (2 * n + 1)) := by
  sorry

end NUMINAMATH_CALUDE_even_and_divisible_by_six_l3965_396517


namespace NUMINAMATH_CALUDE_smallest_whole_number_above_sum_l3965_396561

theorem smallest_whole_number_above_sum : ∃ n : ℕ, 
  (n : ℚ) > (3 + 1/3 : ℚ) + (4 + 1/6 : ℚ) + (5 + 1/12 : ℚ) + (6 + 1/8 : ℚ) ∧
  ∀ m : ℕ, (m : ℚ) > (3 + 1/3 : ℚ) + (4 + 1/6 : ℚ) + (5 + 1/12 : ℚ) + (6 + 1/8 : ℚ) → n ≤ m :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_CALUDE_smallest_whole_number_above_sum_l3965_396561


namespace NUMINAMATH_CALUDE_limsup_subset_l3965_396554

open Set

theorem limsup_subset {α : Type*} (A B : ℕ → Set α) (h : ∀ n, A n ⊆ B n) :
  (⋂ k, ⋃ n ≥ k, A n) ⊆ (⋂ k, ⋃ n ≥ k, B n) := by
  sorry

end NUMINAMATH_CALUDE_limsup_subset_l3965_396554


namespace NUMINAMATH_CALUDE_fun_math_book_price_l3965_396570

/-- The price of the "Fun Math" book in yuan -/
def book_price : ℝ := 4

/-- The amount Xiaohong is short in yuan -/
def xiaohong_short : ℝ := 2.2

/-- The amount Xiaoming is short in yuan -/
def xiaoming_short : ℝ := 1.8

/-- Theorem stating that the book price is 4 yuan given the conditions -/
theorem fun_math_book_price :
  (book_price - xiaohong_short) + (book_price - xiaoming_short) = book_price :=
by sorry

end NUMINAMATH_CALUDE_fun_math_book_price_l3965_396570


namespace NUMINAMATH_CALUDE_pure_imaginary_condition_l3965_396569

/-- A complex number is pure imaginary if its real part is zero -/
def IsPureImaginary (z : ℂ) : Prop := z.re = 0

/-- The theorem states that if (a - i)² * i³ is a pure imaginary number,
    then the real number a must be equal to 0 -/
theorem pure_imaginary_condition (a : ℝ) :
  IsPureImaginary ((a - Complex.I)^2 * Complex.I^3) → a = 0 := by
  sorry

end NUMINAMATH_CALUDE_pure_imaginary_condition_l3965_396569


namespace NUMINAMATH_CALUDE_find_number_l3965_396558

theorem find_number : ∃ x : ℝ, 1.2 * x = 2 * (0.8 * (x - 20)) ∧ x = 80 := by
  sorry

end NUMINAMATH_CALUDE_find_number_l3965_396558


namespace NUMINAMATH_CALUDE_one_third_of_eleven_y_plus_three_l3965_396550

theorem one_third_of_eleven_y_plus_three (y : ℝ) : (1 / 3) * (11 * y + 3) = (11 * y / 3) + 1 := by
  sorry

end NUMINAMATH_CALUDE_one_third_of_eleven_y_plus_three_l3965_396550


namespace NUMINAMATH_CALUDE_hexagon_side_count_l3965_396571

/-- A convex hexagon with exactly two distinct side lengths -/
structure ConvexHexagon where
  side_length1 : ℝ
  side_length2 : ℝ
  num_sides1 : ℕ
  num_sides2 : ℕ
  distinct_lengths : side_length1 ≠ side_length2
  total_sides : num_sides1 + num_sides2 = 6

/-- The perimeter of a convex hexagon -/
def perimeter (h : ConvexHexagon) : ℝ :=
  h.side_length1 * h.num_sides1 + h.side_length2 * h.num_sides2

theorem hexagon_side_count (h : ConvexHexagon)
  (side1_length : h.side_length1 = 8)
  (side2_length : h.side_length2 = 10)
  (total_perimeter : perimeter h = 56) :
  h.num_sides2 = 4 :=
sorry

end NUMINAMATH_CALUDE_hexagon_side_count_l3965_396571


namespace NUMINAMATH_CALUDE_remainder_of_sum_of_fourth_powers_l3965_396515

theorem remainder_of_sum_of_fourth_powers (x y : ℕ+) (P Q : ℕ) :
  x^4 + y^4 = (x + y) * (P + 13) + Q ∧ Q < x + y →
  Q = 8 :=
by sorry

end NUMINAMATH_CALUDE_remainder_of_sum_of_fourth_powers_l3965_396515


namespace NUMINAMATH_CALUDE_total_profit_calculation_l3965_396533

def total_subscription : ℕ := 50000
def a_extra_over_b : ℕ := 4000
def b_extra_over_c : ℕ := 5000
def a_profit : ℕ := 15120

theorem total_profit_calculation :
  ∃ (c_subscription : ℕ) (total_profit : ℕ),
    let b_subscription := c_subscription + b_extra_over_c
    let a_subscription := b_subscription + a_extra_over_b
    a_subscription + b_subscription + c_subscription = total_subscription ∧
    a_subscription * total_profit = a_profit * total_subscription ∧
    total_profit = 36000 := by
  sorry

end NUMINAMATH_CALUDE_total_profit_calculation_l3965_396533


namespace NUMINAMATH_CALUDE_ned_shirts_problem_l3965_396516

theorem ned_shirts_problem (total_shirts : ℕ) (short_sleeve : ℕ) (washed_shirts : ℕ)
  (h1 : total_shirts = 30)
  (h2 : short_sleeve = 9)
  (h3 : washed_shirts = 29) :
  total_shirts - short_sleeve - (total_shirts - washed_shirts) = 19 := by
  sorry

end NUMINAMATH_CALUDE_ned_shirts_problem_l3965_396516


namespace NUMINAMATH_CALUDE_gcd_of_720_120_168_l3965_396542

theorem gcd_of_720_120_168 : Nat.gcd 720 (Nat.gcd 120 168) = 24 := by
  sorry

end NUMINAMATH_CALUDE_gcd_of_720_120_168_l3965_396542


namespace NUMINAMATH_CALUDE_restaurant_gratuities_l3965_396503

/-- Calculates the gratuities charged by a restaurant --/
theorem restaurant_gratuities
  (total_bill : ℝ)
  (sales_tax_rate : ℝ)
  (striploin_cost : ℝ)
  (wine_cost : ℝ)
  (h_total_bill : total_bill = 140)
  (h_sales_tax_rate : sales_tax_rate = 0.1)
  (h_striploin_cost : striploin_cost = 80)
  (h_wine_cost : wine_cost = 10) :
  total_bill - (striploin_cost + wine_cost) * (1 + sales_tax_rate) = 41 := by
  sorry

end NUMINAMATH_CALUDE_restaurant_gratuities_l3965_396503


namespace NUMINAMATH_CALUDE_range_of_a_l3965_396581

def A : Set ℝ := {x | x^2 ≤ 5*x - 4}

def M (a : ℝ) : Set ℝ := {x | x^2 - (a+2)*x + 2*a ≤ 0}

theorem range_of_a (a : ℝ) : (M a ⊆ A) ↔ (1 ≤ a ∧ a ≤ 4) := by sorry

end NUMINAMATH_CALUDE_range_of_a_l3965_396581


namespace NUMINAMATH_CALUDE_peter_walk_time_l3965_396585

/-- Calculates the remaining time to walk given total distance, walking speed, and distance already walked -/
def remaining_walk_time (total_distance : ℝ) (walking_speed : ℝ) (distance_walked : ℝ) : ℝ :=
  (total_distance - distance_walked) * walking_speed

theorem peter_walk_time :
  let total_distance : ℝ := 2.5
  let walking_speed : ℝ := 20
  let distance_walked : ℝ := 1
  remaining_walk_time total_distance walking_speed distance_walked = 30 := by
sorry

end NUMINAMATH_CALUDE_peter_walk_time_l3965_396585


namespace NUMINAMATH_CALUDE_eldest_age_is_fifteen_l3965_396567

/-- The ages of three grandchildren satisfying specific conditions -/
structure GrandchildrenAges where
  youngest : ℕ
  middle : ℕ
  eldest : ℕ
  age_difference : middle - youngest = 3
  eldest_triple_youngest : eldest = 3 * youngest
  eldest_sum_plus_two : eldest = youngest + middle + 2

/-- The age of the eldest grandchild is 15 -/
theorem eldest_age_is_fifteen (ages : GrandchildrenAges) : ages.eldest = 15 := by
  sorry

end NUMINAMATH_CALUDE_eldest_age_is_fifteen_l3965_396567


namespace NUMINAMATH_CALUDE_fourth_term_of_progression_l3965_396526

-- Define the geometric progression
def geometric_progression (a : ℝ) (r : ℝ) (n : ℕ) : ℝ := a * r^(n - 1)

-- Define our specific progression
def our_progression (n : ℕ) : ℝ := 5^(1 / (5 * 2^(n - 1)))

-- Theorem statement
theorem fourth_term_of_progression :
  our_progression 4 = 5^(1/10) := by
  sorry

end NUMINAMATH_CALUDE_fourth_term_of_progression_l3965_396526


namespace NUMINAMATH_CALUDE_student_mistake_fraction_l3965_396573

theorem student_mistake_fraction (x y : ℚ) : 
  (x / y) * 576 = 480 → x / y = 5 / 6 :=
by sorry

end NUMINAMATH_CALUDE_student_mistake_fraction_l3965_396573


namespace NUMINAMATH_CALUDE_x_intercept_of_line_l3965_396555

/-- The x-intercept of the line 4x + 7y = 28 is (7, 0) -/
theorem x_intercept_of_line (x y : ℝ) : 
  4 * x + 7 * y = 28 → y = 0 → x = 7 := by
  sorry

end NUMINAMATH_CALUDE_x_intercept_of_line_l3965_396555


namespace NUMINAMATH_CALUDE_solve_equation_l3965_396509

theorem solve_equation : 
  ∃ y : ℝ, ((10 - y)^2 = 4 * y^2) ∧ (y = 10/3 ∨ y = -10) :=
by
  sorry

end NUMINAMATH_CALUDE_solve_equation_l3965_396509


namespace NUMINAMATH_CALUDE_parabola_triangle_problem_l3965_396540

/-- Given three distinct points A, B, C on the parabola y = x^2, where AB is parallel to the x-axis
    and ABC forms a right triangle with area 2016, prove that the y-coordinate of C is 4064255 -/
theorem parabola_triangle_problem (A B C : ℝ × ℝ) : 
  (∃ m n : ℝ, A = (m, m^2) ∧ B = (n, n^2) ∧ C = ((m+n)/2, ((m+n)/2)^2)) →  -- Points on y = x^2
  (A.2 = B.2) →  -- AB parallel to x-axis
  (C.1 = (A.1 + B.1) / 2) →  -- C is above midpoint of AB (right angle)
  (abs (B.1 - A.1) * abs (C.2 - A.2) / 2 = 2016) →  -- Area of triangle ABC
  C.2 = 4064255 := by
  sorry

end NUMINAMATH_CALUDE_parabola_triangle_problem_l3965_396540


namespace NUMINAMATH_CALUDE_max_third_altitude_exists_max_altitude_l3965_396545

/-- An isosceles triangle with specific altitude properties -/
structure IsoscelesTriangle where
  -- The lengths of the sides
  AB : ℝ
  BC : ℝ
  -- The altitudes
  h_AB : ℝ
  h_AC : ℝ
  h_BC : ℕ
  -- Isosceles property
  isIsosceles : AB = BC
  -- Given altitude lengths
  alt_AB : h_AB = 6
  alt_AC : h_AC = 18
  -- Triangle inequality
  triangle_inequality : AB + BC > AC ∧ AB + AC > BC ∧ BC + AC > AB

/-- The theorem stating the maximum possible integer length of the third altitude -/
theorem max_third_altitude (t : IsoscelesTriangle) : t.h_BC ≤ 6 := by
  sorry

/-- The existence of such a triangle with the maximum third altitude -/
theorem exists_max_altitude : ∃ t : IsoscelesTriangle, t.h_BC = 6 := by
  sorry

end NUMINAMATH_CALUDE_max_third_altitude_exists_max_altitude_l3965_396545


namespace NUMINAMATH_CALUDE_modulus_of_complex_fraction_l3965_396529

theorem modulus_of_complex_fraction (z : ℂ) : z = (1 + Complex.I) / Complex.I → Complex.abs z = Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_modulus_of_complex_fraction_l3965_396529


namespace NUMINAMATH_CALUDE_remaining_macaroons_count_l3965_396562

/-- The number of remaining macaroons after eating some -/
def remaining_macaroons (initial_red initial_green eaten_green : ℕ) : ℕ :=
  let eaten_red := 2 * eaten_green
  let remaining_red := initial_red - eaten_red
  let remaining_green := initial_green - eaten_green
  remaining_red + remaining_green

/-- Theorem stating that the number of remaining macaroons is 45 -/
theorem remaining_macaroons_count :
  remaining_macaroons 50 40 15 = 45 := by
  sorry

end NUMINAMATH_CALUDE_remaining_macaroons_count_l3965_396562


namespace NUMINAMATH_CALUDE_company_survey_problem_l3965_396534

/-- The number of employees who do not use social networks -/
def non_social_users : ℕ := 40

/-- The proportion of social network users who use VKontakte -/
def vk_users_ratio : ℚ := 3/4

/-- The proportion of social network users who use both VKontakte and Odnoklassniki -/
def both_users_ratio : ℚ := 13/20

/-- The proportion of total employees who use Odnoklassniki -/
def ok_users_ratio : ℚ := 5/6

/-- The total number of employees in the company -/
def total_employees : ℕ := 540

theorem company_survey_problem :
  ∃ (N : ℕ),
    N = total_employees ∧
    (N - non_social_users : ℚ) * (vk_users_ratio + (1 - vk_users_ratio)) = N * ok_users_ratio :=
sorry

end NUMINAMATH_CALUDE_company_survey_problem_l3965_396534


namespace NUMINAMATH_CALUDE_books_count_l3965_396530

/-- The number of books Darryl has -/
def darryl_books : ℕ := 20

/-- The number of books Lamont has -/
def lamont_books : ℕ := 2 * darryl_books

/-- The number of books Loris has -/
def loris_books : ℕ := lamont_books - 3

/-- The total number of books all three have -/
def total_books : ℕ := darryl_books + lamont_books + loris_books

theorem books_count : total_books = 97 := by
  sorry

end NUMINAMATH_CALUDE_books_count_l3965_396530


namespace NUMINAMATH_CALUDE_tangent_intersection_y_coordinate_l3965_396506

noncomputable def curve (x : ℝ) : ℝ := x^3

theorem tangent_intersection_y_coordinate 
  (a b : ℝ) 
  (hA : ∃ y, y = curve a) 
  (hB : ∃ y, y = curve b) 
  (h_perp : (3 * a^2) * (3 * b^2) = -1) :
  ∃ x y, y = -1/3 ∧ 
    y = 3 * a^2 * (x - a) + a^3 ∧ 
    y = 3 * b^2 * (x - b) + b^3 :=
sorry

end NUMINAMATH_CALUDE_tangent_intersection_y_coordinate_l3965_396506


namespace NUMINAMATH_CALUDE_dot_product_of_perpendicular_vectors_l3965_396502

/-- Given two planar vectors a and b, where a = (1, √3) and a is perpendicular to (a - b),
    prove that the dot product of a and b is 4. -/
theorem dot_product_of_perpendicular_vectors (a b : ℝ × ℝ) :
  a = (1, Real.sqrt 3) →
  a.1 * (a.1 - b.1) + a.2 * (a.2 - b.2) = 0 →
  a.1 * b.1 + a.2 * b.2 = 4 := by
  sorry

end NUMINAMATH_CALUDE_dot_product_of_perpendicular_vectors_l3965_396502


namespace NUMINAMATH_CALUDE_circle_area_20cm_diameter_l3965_396519

/-- The area of a circle with diameter 20 cm is 314 square cm, given π = 3.14 -/
theorem circle_area_20cm_diameter (π : ℝ) (h : π = 3.14) :
  let d : ℝ := 20
  let r : ℝ := d / 2
  π * r^2 = 314 := by sorry

end NUMINAMATH_CALUDE_circle_area_20cm_diameter_l3965_396519


namespace NUMINAMATH_CALUDE_mary_fruit_expenses_l3965_396551

theorem mary_fruit_expenses :
  let berries_cost : ℚ := 1108 / 100
  let apples_cost : ℚ := 1433 / 100
  let peaches_cost : ℚ := 931 / 100
  berries_cost + apples_cost + peaches_cost = 3472 / 100 := by
  sorry

end NUMINAMATH_CALUDE_mary_fruit_expenses_l3965_396551


namespace NUMINAMATH_CALUDE_square_minus_five_equals_two_l3965_396577

theorem square_minus_five_equals_two : ∃ x : ℤ, x - 5 = 2 := by
  use 7
  sorry

end NUMINAMATH_CALUDE_square_minus_five_equals_two_l3965_396577


namespace NUMINAMATH_CALUDE_number_difference_l3965_396597

theorem number_difference (L S : ℕ) (h1 : L = 1620) (h2 : L = 6 * S + 15) : L - S = 1353 := by
  sorry

end NUMINAMATH_CALUDE_number_difference_l3965_396597


namespace NUMINAMATH_CALUDE_arithmetic_expression_equality_l3965_396595

theorem arithmetic_expression_equality : (3^2 * 5) + (7 * 4) - (42 / 3) = 59 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_expression_equality_l3965_396595


namespace NUMINAMATH_CALUDE_f_max_value_l3965_396505

/-- The quadratic function f(x) = -2x^2 - 5 -/
def f (x : ℝ) : ℝ := -2 * x^2 - 5

/-- The maximum value of f(x) is -5 -/
theorem f_max_value : ∃ (M : ℝ), M = -5 ∧ ∀ x, f x ≤ M := by sorry

end NUMINAMATH_CALUDE_f_max_value_l3965_396505


namespace NUMINAMATH_CALUDE_total_rent_is_6500_l3965_396552

/-- Represents the grazing data for a milkman -/
structure GrazingData where
  cows : ℕ
  months : ℕ

/-- Calculates the total rent of a pasture given grazing data and one milkman's rent -/
def totalRent (a b c d : GrazingData) (aRent : ℕ) : ℕ :=
  let totalCowMonths := a.cows * a.months + b.cows * b.months + c.cows * c.months + d.cows * d.months
  let rentPerCowMonth := aRent / (a.cows * a.months)
  rentPerCowMonth * totalCowMonths

/-- Theorem stating that the total rent is 6500 given the problem conditions -/
theorem total_rent_is_6500 :
  let a : GrazingData := ⟨24, 3⟩
  let b : GrazingData := ⟨10, 5⟩
  let c : GrazingData := ⟨35, 4⟩
  let d : GrazingData := ⟨21, 3⟩
  let aRent : ℕ := 1440
  totalRent a b c d aRent = 6500 := by
  sorry

end NUMINAMATH_CALUDE_total_rent_is_6500_l3965_396552


namespace NUMINAMATH_CALUDE_square_area_from_diagonal_l3965_396512

theorem square_area_from_diagonal (d : ℝ) (h : d = 8 * Real.sqrt 2) :
  (d / Real.sqrt 2) ^ 2 = 64 := by
  sorry

end NUMINAMATH_CALUDE_square_area_from_diagonal_l3965_396512


namespace NUMINAMATH_CALUDE_function_bounds_l3965_396536

theorem function_bounds (x : ℝ) : 
  (1 : ℝ) / 2 ≤ (x^2 + x + 1) / (x^2 + 1) ∧ (x^2 + x + 1) / (x^2 + 1) ≤ (3 : ℝ) / 2 := by
  sorry

end NUMINAMATH_CALUDE_function_bounds_l3965_396536


namespace NUMINAMATH_CALUDE_complex_equation_solution_l3965_396566

theorem complex_equation_solution (i : ℂ) (h : i * i = -1) :
  ∃ z : ℂ, (2 + i) * z = 5 ∧ z = 2 - i :=
by sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l3965_396566


namespace NUMINAMATH_CALUDE_total_shirts_made_l3965_396524

/-- The number of shirts a machine can make per minute -/
def shirts_per_minute : ℕ := 6

/-- The number of minutes the machine worked yesterday -/
def minutes_yesterday : ℕ := 12

/-- The number of minutes the machine worked today -/
def minutes_today : ℕ := 14

/-- Theorem: The total number of shirts made by the machine is 156 -/
theorem total_shirts_made : 
  shirts_per_minute * minutes_yesterday + shirts_per_minute * minutes_today = 156 := by
  sorry

end NUMINAMATH_CALUDE_total_shirts_made_l3965_396524


namespace NUMINAMATH_CALUDE_sum_of_binary_digits_157_l3965_396576

/-- Converts a natural number to its binary representation as a list of bits -/
def toBinary (n : ℕ) : List ℕ := sorry

/-- Sums the digits of a natural number in base 10 -/
def sumDigits (n : ℕ) : ℕ := sorry

/-- Sums the elements of a list of natural numbers -/
def sumList (l : List ℕ) : ℕ := sorry

theorem sum_of_binary_digits_157 : 
  let binary157 := toBinary 157
  let sumBinary157 := sumList binary157
  let sumDigits157 := sumDigits 157
  let binarySumDigits157 := toBinary sumDigits157
  let sumBinarySumDigits157 := sumList binarySumDigits157
  sumBinary157 + sumBinarySumDigits157 = 8 := by sorry

end NUMINAMATH_CALUDE_sum_of_binary_digits_157_l3965_396576


namespace NUMINAMATH_CALUDE_probability_theorem_l3965_396580

def num_male_students : ℕ := 5
def num_female_students : ℕ := 2
def num_representatives : ℕ := 3

def probability_B_or_C_given_A (total_students : ℕ) (remaining_selections : ℕ) : ℚ :=
  let favorable_outcomes := (remaining_selections * (total_students - 3)) + 1
  let total_outcomes := Nat.choose (total_students - 1) remaining_selections
  favorable_outcomes / total_outcomes

theorem probability_theorem :
  probability_B_or_C_given_A (num_male_students + num_female_students) (num_representatives - 1) = 3/5 :=
by sorry

end NUMINAMATH_CALUDE_probability_theorem_l3965_396580


namespace NUMINAMATH_CALUDE_ball_bounce_theorem_l3965_396500

theorem ball_bounce_theorem (h : Real) (r : Real) (target : Real) :
  h = 700 ∧ r = 1/3 ∧ target = 2 →
  (∀ k : ℕ, h * r^k < target ↔ k ≥ 6) :=
by sorry

end NUMINAMATH_CALUDE_ball_bounce_theorem_l3965_396500


namespace NUMINAMATH_CALUDE_f_is_smallest_f_is_minimal_l3965_396589

/-- 
For a given integer n ≥ 4, f(n) is the smallest integer such that 
any f(n)-element subset of {m, m+1, ..., m+n-1} contains at least 
3 pairwise coprime elements, where m is any positive integer.
-/
def f (n : ℕ) : ℕ :=
  (n + 1) / 2 + (n + 1) / 3 - (n + 1) / 6 + 1

/-- 
Theorem: For integers n ≥ 4, f(n) is the smallest integer such that 
any f(n)-element subset of {m, m+1, ..., m+n-1} contains at least 
3 pairwise coprime elements, where m is any positive integer.
-/
theorem f_is_smallest (n : ℕ) (h : n ≥ 4) : 
  ∀ (m : ℕ+), ∀ (S : Finset ℕ), 
    S.card = f n → 
    (∀ x ∈ S, ∃ k, x = m + k ∧ k < n) → 
    ∃ (a b c : ℕ), a ∈ S ∧ b ∈ S ∧ c ∈ S ∧ 
      Nat.Coprime a b ∧ Nat.Coprime b c ∧ Nat.Coprime a c :=
by
  sorry

/-- 
Corollary: There is no smaller integer than f(n) that satisfies 
the conditions for all n ≥ 4.
-/
theorem f_is_minimal (n : ℕ) (h : n ≥ 4) :
  ∀ g : ℕ → ℕ, (∀ k ≥ 4, g k < f k) → 
    ∃ (m : ℕ+) (S : Finset ℕ), 
      S.card = g n ∧
      (∀ x ∈ S, ∃ k, x = m + k ∧ k < n) ∧
      ∀ (a b c : ℕ), a ∈ S → b ∈ S → c ∈ S → 
        ¬(Nat.Coprime a b ∧ Nat.Coprime b c ∧ Nat.Coprime a c) :=
by
  sorry

end NUMINAMATH_CALUDE_f_is_smallest_f_is_minimal_l3965_396589


namespace NUMINAMATH_CALUDE_masks_donated_to_museum_l3965_396546

/-- Given that Alicia initially had 90 sets of masks and was left with 39 sets after donating to a museum,
    prove that she gave 51 sets to the museum. -/
theorem masks_donated_to_museum (initial_sets : ℕ) (remaining_sets : ℕ) 
    (h1 : initial_sets = 90) 
    (h2 : remaining_sets = 39) : 
  initial_sets - remaining_sets = 51 := by
  sorry

end NUMINAMATH_CALUDE_masks_donated_to_museum_l3965_396546


namespace NUMINAMATH_CALUDE_eight_digit_divisible_by_11_l3965_396594

/-- An eight-digit number in the form 8524m637 -/
def eight_digit_number (m : ℕ) : ℕ := 85240000 + m * 1000 + 637

/-- Sum of digits in odd positions -/
def sum_odd_digits (m : ℕ) : ℕ := 8 + 2 + m + 3

/-- Sum of digits in even positions -/
def sum_even_digits : ℕ := 5 + 4 + 6 + 7

/-- A number is divisible by 11 if the difference between the sum of digits in odd and even positions is a multiple of 11 -/
def divisible_by_11 (n : ℕ) : Prop :=
  ∃ k : ℤ, (sum_odd_digits n - sum_even_digits : ℤ) = 11 * k

theorem eight_digit_divisible_by_11 :
  ∃ m : ℕ, m < 10 ∧ divisible_by_11 (eight_digit_number m) ↔ m = 9 :=
sorry

end NUMINAMATH_CALUDE_eight_digit_divisible_by_11_l3965_396594


namespace NUMINAMATH_CALUDE_even_function_implies_a_equals_one_l3965_396557

/-- A function f is even if f(x) = f(-x) for all x in its domain. -/
def IsEven (f : ℝ → ℝ) : Prop :=
  ∀ x, f x = f (-x)

/-- The function f(x) = (a - 2)x^2 + (a - 1)x + 3 -/
def f (a : ℝ) (x : ℝ) : ℝ :=
  (a - 2) * x^2 + (a - 1) * x + 3

/-- If f(x) = (a - 2)x^2 + (a - 1)x + 3 is an even function, then a = 1 -/
theorem even_function_implies_a_equals_one :
  ∀ a : ℝ, IsEven (f a) → a = 1 := by
  sorry

end NUMINAMATH_CALUDE_even_function_implies_a_equals_one_l3965_396557


namespace NUMINAMATH_CALUDE_total_revenue_calculation_l3965_396598

def ticket_revenue (student_price adult_price child_price senior_price : ℚ)
                   (group_discount : ℚ)
                   (separate_students separate_adults separate_children separate_seniors : ℕ)
                   (group_students group_adults group_children group_seniors : ℕ) : ℚ :=
  let separate_revenue := student_price * separate_students +
                          adult_price * separate_adults +
                          child_price * separate_children +
                          senior_price * separate_seniors
  let group_subtotal := student_price * group_students +
                        adult_price * group_adults +
                        child_price * group_children +
                        senior_price * group_seniors
  let group_size := group_students + group_adults + group_children + group_seniors
  let group_revenue := if group_size > 10 then group_subtotal * (1 - group_discount) else group_subtotal
  separate_revenue + group_revenue

theorem total_revenue_calculation :
  ticket_revenue 6 8 4 7 (1/10)
                 20 12 15 10
                 5 8 10 9 = 523.3 := by
  sorry

end NUMINAMATH_CALUDE_total_revenue_calculation_l3965_396598


namespace NUMINAMATH_CALUDE_remaining_problems_l3965_396588

theorem remaining_problems (total : ℕ) (first_20min : ℕ) (second_20min : ℕ) 
  (h1 : total = 75)
  (h2 : first_20min = 10)
  (h3 : second_20min = 2 * first_20min) :
  total - (first_20min + second_20min) = 45 := by
  sorry

end NUMINAMATH_CALUDE_remaining_problems_l3965_396588


namespace NUMINAMATH_CALUDE_watch_cost_price_l3965_396596

def watch_problem (cost_price : ℝ) : Prop :=
  let loss_percentage : ℝ := 10
  let gain_percentage : ℝ := 4
  let additional_amount : ℝ := 210
  let selling_price_1 : ℝ := cost_price * (1 - loss_percentage / 100)
  let selling_price_2 : ℝ := cost_price * (1 + gain_percentage / 100)
  selling_price_2 = selling_price_1 + additional_amount

theorem watch_cost_price : 
  ∃ (cost_price : ℝ), watch_problem cost_price ∧ cost_price = 1500 :=
sorry

end NUMINAMATH_CALUDE_watch_cost_price_l3965_396596


namespace NUMINAMATH_CALUDE_inequality_counterexample_l3965_396514

theorem inequality_counterexample (a b : ℝ) (h : a < b) :
  ∃ c : ℝ, ¬(a * c < b * c) := by
  sorry

end NUMINAMATH_CALUDE_inequality_counterexample_l3965_396514


namespace NUMINAMATH_CALUDE_door_diagonal_equation_l3965_396510

theorem door_diagonal_equation (x : ℝ) : x ^ 2 - (x - 2) ^ 2 = (x - 4) ^ 2 := by
  sorry

end NUMINAMATH_CALUDE_door_diagonal_equation_l3965_396510


namespace NUMINAMATH_CALUDE_sum_of_derived_geometric_progression_l3965_396518

/-- An arithmetic progression with the given properties -/
structure ArithmeticProgression where
  a : ℕ  -- First term
  d : ℕ  -- Common difference
  sum_first_three : a + (a + d) + (a + 2 * d) = 21
  increasing : d > 0

/-- A geometric progression derived from the arithmetic progression -/
def geometric_from_arithmetic (ap : ArithmeticProgression) : Fin 3 → ℕ
  | 0 => ap.a - 1
  | 1 => ap.a + ap.d - 1
  | 2 => ap.a + 2 * ap.d + 2

/-- The theorem to be proved -/
theorem sum_of_derived_geometric_progression (ap : ArithmeticProgression) :
  let gp := geometric_from_arithmetic ap
  let q := gp 1 / gp 0  -- Common ratio of the geometric progression
  gp 0 * (q^8 - 1) / (q - 1) = 765 := by
  sorry


end NUMINAMATH_CALUDE_sum_of_derived_geometric_progression_l3965_396518


namespace NUMINAMATH_CALUDE_abs_neg_three_equals_three_l3965_396553

theorem abs_neg_three_equals_three : abs (-3 : ℝ) = 3 := by
  sorry

end NUMINAMATH_CALUDE_abs_neg_three_equals_three_l3965_396553


namespace NUMINAMATH_CALUDE_job_completion_time_l3965_396521

/-- Given the work rates of machines A, B, and C, prove that 15 type A machines and 7 type B machines complete the job in 4 hours. -/
theorem job_completion_time 
  (A B C : ℝ) -- Work rates of machines A, B, and C in jobs per hour
  (h1 : 15 * A + 7 * B = 1 / 4) -- 15 type A and 7 type B machines complete the job in x hours
  (h2 : 8 * B + 15 * C = 1 / 11) -- 8 type B and 15 type C machines complete the job in 11 hours
  (h3 : A + B + C = 1 / 44) -- 1 of each machine type completes the job in 44 hours
  : 15 * A + 7 * B = 1 / 4 := by
  sorry

end NUMINAMATH_CALUDE_job_completion_time_l3965_396521


namespace NUMINAMATH_CALUDE_valid_pairs_count_l3965_396531

/-- Represents a person's age --/
structure Age :=
  (value : ℕ)

/-- Represents the current ages of Jane and Dick --/
structure CurrentAges :=
  (jane : Age)
  (dick : Age)

/-- Represents the ages of Jane and Dick after n years --/
structure FutureAges :=
  (jane : Age)
  (dick : Age)

/-- Checks if an age is a two-digit number --/
def is_two_digit (age : Age) : Prop :=
  10 ≤ age.value ∧ age.value ≤ 99

/-- Checks if two ages have interchanged digits --/
def has_interchanged_digits (age1 age2 : Age) : Prop :=
  ∃ (a b : ℕ), a ≠ b ∧ age1.value = 10 * a + b ∧ age2.value = 10 * b + a

/-- Calculates the number of valid (d, n) pairs --/
def count_valid_pairs (current : CurrentAges) : ℕ :=
  sorry

/-- The main theorem to be proved --/
theorem valid_pairs_count (current : CurrentAges) :
  current.jane.value = 30 ∧
  current.dick.value > current.jane.value →
  (∀ n : ℕ, n > 0 →
    let future : FutureAges := ⟨⟨current.jane.value + n⟩, ⟨current.dick.value + n⟩⟩
    is_two_digit future.jane ∧
    is_two_digit future.dick ∧
    has_interchanged_digits future.jane future.dick) →
  count_valid_pairs current = 26 :=
by sorry

end NUMINAMATH_CALUDE_valid_pairs_count_l3965_396531


namespace NUMINAMATH_CALUDE_budi_can_win_l3965_396578

/-- The set of numbers from which players choose -/
def S : Finset ℕ := Finset.range 30

/-- The total number of balls in the game -/
def totalBalls : ℕ := 2015

/-- Astri's chosen numbers -/
structure AstriChoice where
  a : ℕ
  b : ℕ
  a_in_S : a ∈ S
  b_in_S : b ∈ S
  a_ne_b : a ≠ b

/-- Budi's chosen numbers -/
structure BudiChoice (ac : AstriChoice) where
  c : ℕ
  d : ℕ
  c_in_S : c ∈ S
  d_in_S : d ∈ S
  c_ne_d : c ≠ d
  c_ne_a : c ≠ ac.a
  c_ne_b : c ≠ ac.b
  d_ne_a : d ≠ ac.a
  d_ne_b : d ≠ ac.b

/-- The game state -/
structure GameState where
  ballsLeft : ℕ
  astriTurn : Bool

/-- A winning strategy for Budi -/
def isWinningStrategy (ac : AstriChoice) (bc : BudiChoice ac) (strategy : GameState → ℕ) : Prop :=
  ∀ (gs : GameState), 
    (gs.astriTurn ∧ gs.ballsLeft < ac.a ∧ gs.ballsLeft < ac.b) ∨
    (¬gs.astriTurn ∧ 
      ((strategy gs = bc.c ∧ gs.ballsLeft ≥ bc.c) ∨ 
       (strategy gs = bc.d ∧ gs.ballsLeft ≥ bc.d)))

/-- The main theorem -/
theorem budi_can_win : 
  ∀ (ac : AstriChoice), ∃ (bc : BudiChoice ac) (strategy : GameState → ℕ), 
    isWinningStrategy ac bc strategy :=
sorry

end NUMINAMATH_CALUDE_budi_can_win_l3965_396578


namespace NUMINAMATH_CALUDE_distance_between_trees_l3965_396549

theorem distance_between_trees (yard_length : ℕ) (num_trees : ℕ) : 
  yard_length = 414 → num_trees = 24 → 
  (yard_length : ℚ) / (num_trees - 1 : ℚ) = 18 := by
  sorry

end NUMINAMATH_CALUDE_distance_between_trees_l3965_396549


namespace NUMINAMATH_CALUDE_cylinder_dimensions_from_sphere_l3965_396539

/-- Given a sphere and a right circular cylinder with equal surface areas,
    prove that the height and diameter of the cylinder are both 14 cm
    when the radius of the sphere is 7 cm. -/
theorem cylinder_dimensions_from_sphere (r : ℝ) (h d : ℝ) : 
  r = 7 →  -- radius of the sphere is 7 cm
  h = d →  -- height and diameter of cylinder are equal
  4 * Real.pi * r^2 = 2 * Real.pi * (d/2) * h →  -- surface areas are equal
  h = 14 ∧ d = 14 := by
  sorry

end NUMINAMATH_CALUDE_cylinder_dimensions_from_sphere_l3965_396539


namespace NUMINAMATH_CALUDE_parabola_symmetry_l3965_396522

/-- A parabola with vertex form y = a(x-h)^2 + k -/
structure Parabola where
  a : ℝ
  h : ℝ
  k : ℝ

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Predicate to check if a point is on a parabola -/
def IsOnParabola (p : Point) (para : Parabola) : Prop :=
  p.y = para.a * (p.x - para.h)^2 + para.k

theorem parabola_symmetry (para : Parabola) (m : ℝ) :
  let A : Point := { x := -1, y := 4 }
  let B : Point := { x := m, y := 4 }
  (IsOnParabola A para ∧ IsOnParabola B para) → m = 7 := by
  sorry

end NUMINAMATH_CALUDE_parabola_symmetry_l3965_396522
