import Mathlib

namespace NUMINAMATH_CALUDE_lego_sale_quadruple_pieces_l3298_329874

/-- Represents the number of Lego pieces sold for each type -/
structure LegoSale where
  single : ℕ
  double : ℕ
  triple : ℕ
  quadruple : ℕ

/-- Calculates the total number of circles from a LegoSale -/
def totalCircles (sale : LegoSale) : ℕ :=
  sale.single + 2 * sale.double + 3 * sale.triple + 4 * sale.quadruple

/-- The main theorem to prove -/
theorem lego_sale_quadruple_pieces (sale : LegoSale) :
  sale.single = 100 →
  sale.double = 45 →
  sale.triple = 50 →
  totalCircles sale = 1000 →
  sale.quadruple = 165 := by
  sorry

#check lego_sale_quadruple_pieces

end NUMINAMATH_CALUDE_lego_sale_quadruple_pieces_l3298_329874


namespace NUMINAMATH_CALUDE_fish_tank_ratio_l3298_329833

/-- Given 3 fish tanks with a total of 100 fish, where one tank has 20 fish
    and the other two have an equal number of fish, prove that the ratio of fish
    in each of the other two tanks to the first tank is 2:1 -/
theorem fish_tank_ratio :
  ∀ (fish_in_other_tanks : ℕ),
  3 * 20 + 2 * fish_in_other_tanks = 100 →
  fish_in_other_tanks = 2 * 20 :=
by sorry

end NUMINAMATH_CALUDE_fish_tank_ratio_l3298_329833


namespace NUMINAMATH_CALUDE_journey_distance_l3298_329832

theorem journey_distance (total_time : ℝ) (speed1 : ℝ) (speed2 : ℝ) 
  (h1 : total_time = 10)
  (h2 : speed1 = 21)
  (h3 : speed2 = 24) :
  ∃ (distance : ℝ), 
    distance / (2 * speed1) + distance / (2 * speed2) = total_time ∧ 
    distance = 224 := by
  sorry

end NUMINAMATH_CALUDE_journey_distance_l3298_329832


namespace NUMINAMATH_CALUDE_gym_attendance_l3298_329802

theorem gym_attendance (initial_lifters : ℕ) : 
  initial_lifters + 5 - 2 = 19 → initial_lifters = 16 := by
  sorry

end NUMINAMATH_CALUDE_gym_attendance_l3298_329802


namespace NUMINAMATH_CALUDE_two_correct_implications_l3298_329811

-- Define the types for planes and lines
def Plane : Type := Unit
def Line : Type := Unit

-- Define the relations between lines and planes
def perpendicular_to_plane (l : Line) (p : Plane) : Prop := sorry
def parallel_to_plane (l : Line) (p : Plane) : Prop := sorry
def perpendicular_planes (p1 p2 : Plane) : Prop := sorry
def not_in_plane (l : Line) (p : Plane) : Prop := sorry

theorem two_correct_implications 
  (α β : Plane) 
  (l : Line) 
  (h_diff : α ≠ β)
  (h_not_in_α : not_in_plane l α)
  (h_not_in_β : not_in_plane l β)
  (h1 : perpendicular_to_plane l α)
  (h2 : parallel_to_plane l β)
  (h3 : perpendicular_planes α β) :
  ∃ (P Q R : Prop),
    (P ∧ Q → R) ∧
    (P ∧ R → Q) ∧
    ¬(Q ∧ R → P) ∧
    P ≠ Q ∧ Q ≠ R ∧ R ≠ P ∧
    (P = perpendicular_to_plane l α ∨ 
     P = parallel_to_plane l β ∨ 
     P = perpendicular_planes α β) ∧
    (Q = perpendicular_to_plane l α ∨ 
     Q = parallel_to_plane l β ∨ 
     Q = perpendicular_planes α β) ∧
    (R = perpendicular_to_plane l α ∨ 
     R = parallel_to_plane l β ∨ 
     R = perpendicular_planes α β) :=
by sorry

end NUMINAMATH_CALUDE_two_correct_implications_l3298_329811


namespace NUMINAMATH_CALUDE_floor_a_equals_1994_minus_n_l3298_329887

def a : ℕ → ℚ
  | 0 => 1994
  | n + 1 => (a n)^2 / (a n + 1)

theorem floor_a_equals_1994_minus_n (n : ℕ) (h : n ≤ 998) :
  ⌊a n⌋ = 1994 - n :=
by sorry

end NUMINAMATH_CALUDE_floor_a_equals_1994_minus_n_l3298_329887


namespace NUMINAMATH_CALUDE_farm_legs_l3298_329806

/-- The total number of animal legs on a farm with ducks, dogs, and spiders -/
def total_legs (num_ducks : ℕ) (num_dogs : ℕ) (num_spiders : ℕ) (num_three_legged_dogs : ℕ) : ℕ :=
  2 * num_ducks + 4 * (num_dogs - num_three_legged_dogs) + 3 * num_three_legged_dogs + 8 * num_spiders

/-- Theorem stating that the total number of animal legs on the farm is 55 -/
theorem farm_legs : total_legs 6 5 3 1 = 55 := by
  sorry

end NUMINAMATH_CALUDE_farm_legs_l3298_329806


namespace NUMINAMATH_CALUDE_max_y_value_l3298_329830

theorem max_y_value (x y : ℝ) (h : x^2 + y^2 = 20*x + 54*y) :
  y ≤ 27 + Real.sqrt 829 ∧ ∃ (x₀ y₀ : ℝ), x₀^2 + y₀^2 = 20*x₀ + 54*y₀ ∧ y₀ = 27 + Real.sqrt 829 := by
  sorry

end NUMINAMATH_CALUDE_max_y_value_l3298_329830


namespace NUMINAMATH_CALUDE_complex_fraction_simplification_l3298_329870

def f (x : ℕ) : ℚ := (x^4 + 625 : ℚ)

theorem complex_fraction_simplification :
  (f 20 * f 40 * f 60 * f 80) / (f 10 * f 30 * f 50 * f 70) = 7 := by
  sorry

end NUMINAMATH_CALUDE_complex_fraction_simplification_l3298_329870


namespace NUMINAMATH_CALUDE_fraction_sum_equals_decimal_l3298_329837

theorem fraction_sum_equals_decimal : 
  (3 : ℚ) / 10 + (3 : ℚ) / 100 + (3 : ℚ) / 1000 = (333 : ℚ) / 1000 := by
  sorry

end NUMINAMATH_CALUDE_fraction_sum_equals_decimal_l3298_329837


namespace NUMINAMATH_CALUDE_max_notebooks_nine_notebooks_possible_l3298_329847

/-- Represents the cost and quantity of an item --/
structure Item where
  cost : ℕ
  quantity : ℕ

/-- Represents a purchase of pens, pencils, and notebooks --/
structure Purchase where
  pens : Item
  pencils : Item
  notebooks : Item

/-- The total cost of a purchase --/
def totalCost (p : Purchase) : ℕ :=
  p.pens.cost * p.pens.quantity +
  p.pencils.cost * p.pencils.quantity +
  p.notebooks.cost * p.notebooks.quantity

/-- A purchase is valid if it meets the given conditions --/
def isValidPurchase (p : Purchase) : Prop :=
  p.pens.cost = 3 ∧
  p.pencils.cost = 4 ∧
  p.notebooks.cost = 10 ∧
  p.pens.quantity ≥ 1 ∧
  p.pencils.quantity ≥ 1 ∧
  p.notebooks.quantity ≥ 1 ∧
  totalCost p ≤ 100

theorem max_notebooks (p : Purchase) (h : isValidPurchase p) :
  p.notebooks.quantity ≤ 9 := by
  sorry

theorem nine_notebooks_possible :
  ∃ p : Purchase, isValidPurchase p ∧ p.notebooks.quantity = 9 := by
  sorry

end NUMINAMATH_CALUDE_max_notebooks_nine_notebooks_possible_l3298_329847


namespace NUMINAMATH_CALUDE_range_of_m_l3298_329888

/-- Proposition p: m + 2 < 0 -/
def p (m : ℝ) : Prop := m + 2 < 0

/-- Proposition q: the equation x^2 + mx + 1 = 0 has no real roots -/
def q (m : ℝ) : Prop := ∀ x : ℝ, x^2 + m*x + 1 ≠ 0

/-- The range of real numbers for m given the conditions -/
theorem range_of_m (m : ℝ) (h1 : ¬¬p m) (h2 : ¬(p m ∧ q m)) : m < -2 := by
  sorry

end NUMINAMATH_CALUDE_range_of_m_l3298_329888


namespace NUMINAMATH_CALUDE_greatest_two_digit_multiple_of_17_l3298_329841

theorem greatest_two_digit_multiple_of_17 : ∃ n : ℕ, n = 85 ∧ 
  (∀ m : ℕ, m ≤ 99 ∧ m ≥ 10 ∧ 17 ∣ m → m ≤ n) ∧ 
  17 ∣ n ∧ n ≤ 99 ∧ n ≥ 10 := by
  sorry

end NUMINAMATH_CALUDE_greatest_two_digit_multiple_of_17_l3298_329841


namespace NUMINAMATH_CALUDE_small_triangle_perimeter_l3298_329869

/-- Represents a triangle divided into smaller triangles -/
structure DividedTriangle where
  large_perimeter : ℝ
  num_small_triangles : ℕ
  small_perimeter : ℝ

/-- The property that the sum of 6 small triangle perimeters minus 3 small triangle perimeters
    equals the large triangle perimeter -/
def perimeter_property (dt : DividedTriangle) : Prop :=
  6 * dt.small_perimeter - 3 * dt.small_perimeter = dt.large_perimeter

/-- Theorem stating that for a triangle with perimeter 120 divided into 9 equal smaller triangles,
    each small triangle has a perimeter of 40 -/
theorem small_triangle_perimeter
  (dt : DividedTriangle)
  (h1 : dt.large_perimeter = 120)
  (h2 : dt.num_small_triangles = 9)
  (h3 : perimeter_property dt) :
  dt.small_perimeter = 40 := by
  sorry

end NUMINAMATH_CALUDE_small_triangle_perimeter_l3298_329869


namespace NUMINAMATH_CALUDE_quadratic_root_and_m_l3298_329886

/-- Given a quadratic equation x^2 + 2x + m = 0 where 2 is a root,
    prove that the other root is -4 and m = -8 -/
theorem quadratic_root_and_m (m : ℝ) : 
  (2 : ℝ)^2 + 2*2 + m = 0 → 
  (∃ (other_root : ℝ), other_root = -4 ∧ 
   other_root^2 + 2*other_root + m = 0 ∧ 
   m = -8) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_root_and_m_l3298_329886


namespace NUMINAMATH_CALUDE_midpoint_implies_xy_24_l3298_329892

-- Define the points
def A : ℝ × ℝ := (2, 10)
def C : ℝ × ℝ := (4, 7)

-- Define B as a function of x and y
def B (x y : ℝ) : ℝ × ℝ := (x, y)

-- Define the midpoint condition
def is_midpoint (m a b : ℝ × ℝ) : Prop :=
  m.1 = (a.1 + b.1) / 2 ∧ m.2 = (a.2 + b.2) / 2

-- Theorem statement
theorem midpoint_implies_xy_24 (x y : ℝ) :
  is_midpoint C A (B x y) → x * y = 24 := by
  sorry

end NUMINAMATH_CALUDE_midpoint_implies_xy_24_l3298_329892


namespace NUMINAMATH_CALUDE_original_gross_profit_percentage_l3298_329813

theorem original_gross_profit_percentage
  (old_price new_price : ℝ)
  (new_profit_percentage : ℝ)
  (cost : ℝ)
  (h1 : old_price = 88)
  (h2 : new_price = 92)
  (h3 : new_profit_percentage = 0.15)
  (h4 : new_price = cost * (1 + new_profit_percentage)) :
  (old_price - cost) / cost = 0.1 := by
  sorry

end NUMINAMATH_CALUDE_original_gross_profit_percentage_l3298_329813


namespace NUMINAMATH_CALUDE_quadratic_inequality_minimum_l3298_329863

theorem quadratic_inequality_minimum (a b c : ℝ) 
  (h1 : ∀ x, 3 < x ∧ x < 4 → a * x^2 + b * x + c > 0)
  (h2 : ∀ x, x ≤ 3 ∨ x ≥ 4 → a * x^2 + b * x + c ≤ 0) :
  ∃ m, m = (c^2 + 5) / (a + b) ∧ 
    (∀ k, k = (c^2 + 5) / (a + b) → m ≤ k) ∧
    m = 4 * Real.sqrt 5 :=
sorry

end NUMINAMATH_CALUDE_quadratic_inequality_minimum_l3298_329863


namespace NUMINAMATH_CALUDE_triangle_bisector_angle_tangent_l3298_329878

/-- Represents a triangle with given side lengths -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Represents a line that bisects both perimeter and area of a triangle -/
structure BisectingLine where
  startPoint : ℝ × ℝ
  endPoint : ℝ × ℝ

/-- The acute angle between two bisecting lines -/
def angleBetweenBisectors (l1 l2 : BisectingLine) : ℝ := sorry

/-- Checks if a line bisects both perimeter and area of a triangle -/
def isBisectingLine (t : Triangle) (l : BisectingLine) : Prop := sorry

theorem triangle_bisector_angle_tangent (t : Triangle) 
  (h1 : t.a = 13 ∧ t.b = 14 ∧ t.c = 15) : 
  ∃ (l1 l2 : BisectingLine) (θ : ℝ),
    isBisectingLine t l1 ∧ 
    isBisectingLine t l2 ∧ 
    θ = angleBetweenBisectors l1 l2 ∧ 
    0 < θ ∧ θ < π/2 ∧
    Real.tan θ = sorry -- This should be replaced with the actual value or expression
    := by sorry

end NUMINAMATH_CALUDE_triangle_bisector_angle_tangent_l3298_329878


namespace NUMINAMATH_CALUDE_tom_speed_proof_l3298_329817

/-- Represents the speed from B to C in miles per hour -/
def speed_B_to_C : ℝ := 64.8

/-- Represents the distance between B and C in miles -/
def distance_B_to_C : ℝ := 1  -- We use 1 as a variable to represent this distance

theorem tom_speed_proof :
  let distance_W_to_B : ℝ := 2 * distance_B_to_C
  let speed_W_to_B : ℝ := 60
  let average_speed : ℝ := 36
  let total_distance : ℝ := distance_W_to_B + distance_B_to_C
  let time_W_to_B : ℝ := distance_W_to_B / speed_W_to_B
  let time_B_to_C : ℝ := distance_B_to_C / speed_B_to_C
  let total_time : ℝ := time_W_to_B + time_B_to_C
  average_speed = total_distance / total_time →
  speed_B_to_C = 64.8 := by
  sorry

#check tom_speed_proof

end NUMINAMATH_CALUDE_tom_speed_proof_l3298_329817


namespace NUMINAMATH_CALUDE_student_divisor_error_l3298_329848

theorem student_divisor_error (D : ℚ) (x : ℚ) : 
  D / 36 = 48 → D / x = 24 → x = 72 := by
  sorry

end NUMINAMATH_CALUDE_student_divisor_error_l3298_329848


namespace NUMINAMATH_CALUDE_monotone_increasing_constraint_l3298_329819

def f (a : ℝ) (x : ℝ) : ℝ := a * x^2 + 2 * x - 3

theorem monotone_increasing_constraint (a : ℝ) :
  (∀ x y, x < y ∧ y < 4 → f a x < f a y) →
  -1/4 ≤ a ∧ a ≤ 0 :=
by sorry

end NUMINAMATH_CALUDE_monotone_increasing_constraint_l3298_329819


namespace NUMINAMATH_CALUDE_mean_problem_l3298_329884

theorem mean_problem (x : ℝ) : 
  (12 + x + 42 + 78 + 104) / 5 = 62 → 
  (128 + 255 + 511 + 1023 + x) / 5 = 398.2 := by
sorry

end NUMINAMATH_CALUDE_mean_problem_l3298_329884


namespace NUMINAMATH_CALUDE_fixed_point_on_line_l3298_329843

/-- For any real number m, the line mx-y+1-3m=0 passes through the point (3, 1) -/
theorem fixed_point_on_line (m : ℝ) : m * 3 - 1 + 1 - 3 * m = 0 := by
  sorry

end NUMINAMATH_CALUDE_fixed_point_on_line_l3298_329843


namespace NUMINAMATH_CALUDE_volunteer_selection_l3298_329861

/-- The number of ways to select 5 people out of 9 (5 male and 4 female), 
    ensuring both genders are included. -/
theorem volunteer_selection (n m f : ℕ) 
  (h1 : n = 5) -- Total number to be selected
  (h2 : m = 5) -- Number of male students
  (h3 : f = 4) -- Number of female students
  : Nat.choose (m + f) n - Nat.choose m n = 125 := by
  sorry

end NUMINAMATH_CALUDE_volunteer_selection_l3298_329861


namespace NUMINAMATH_CALUDE_intersection_M_N_l3298_329893

def M : Set ℝ := {x | -1 ≤ x ∧ x < 3}
def N : Set ℝ := {-1, 0, 1, 2, 3}

theorem intersection_M_N : M ∩ N = {-1, 0, 1, 2} := by sorry

end NUMINAMATH_CALUDE_intersection_M_N_l3298_329893


namespace NUMINAMATH_CALUDE_work_completion_time_l3298_329842

/-- The number of days it takes A to complete the work alone -/
def a_days : ℕ := 30

/-- The total payment for the work in Rupees -/
def total_payment : ℕ := 1000

/-- B's share of the payment in Rupees -/
def b_share : ℕ := 600

/-- The number of days it takes B to complete the work alone -/
def b_days : ℕ := 20

theorem work_completion_time :
  a_days = 30 ∧ 
  total_payment = 1000 ∧ 
  b_share = 600 →
  b_days = 20 :=
by sorry

end NUMINAMATH_CALUDE_work_completion_time_l3298_329842


namespace NUMINAMATH_CALUDE_slope_angle_of_sqrt_three_line_l3298_329851

theorem slope_angle_of_sqrt_three_line :
  let line : ℝ → ℝ := λ x ↦ Real.sqrt 3 * x
  let slope : ℝ := Real.sqrt 3
  let angle : ℝ := 60 * Real.pi / 180
  (∀ x, line x = slope * x) ∧
  slope = Real.tan angle :=
by sorry

end NUMINAMATH_CALUDE_slope_angle_of_sqrt_three_line_l3298_329851


namespace NUMINAMATH_CALUDE_monotonically_increasing_iff_a_geq_one_third_l3298_329827

def f (a : ℝ) (x : ℝ) : ℝ := a * x^3 - x^2 + x - 5

theorem monotonically_increasing_iff_a_geq_one_third :
  ∀ a : ℝ, (∀ x y : ℝ, x < y → f a x < f a y) ↔ a ≥ 1/3 := by
  sorry

end NUMINAMATH_CALUDE_monotonically_increasing_iff_a_geq_one_third_l3298_329827


namespace NUMINAMATH_CALUDE_joan_initial_oranges_l3298_329815

/-- Proves that Joan initially picked 37 oranges given the conditions -/
theorem joan_initial_oranges (initial : ℕ) (sold : ℕ) (remaining : ℕ)
  (h1 : sold = 10)
  (h2 : remaining = 27)
  (h3 : initial = remaining + sold) :
  initial = 37 := by
  sorry

end NUMINAMATH_CALUDE_joan_initial_oranges_l3298_329815


namespace NUMINAMATH_CALUDE_purchase_problem_l3298_329895

/-- Represents the prices and quantities of small light bulbs and electric motors --/
structure PurchaseInfo where
  bulb_price : ℝ
  motor_price : ℝ
  bulb_quantity : ℕ
  motor_quantity : ℕ

/-- Calculates the total cost of a purchase --/
def total_cost (info : PurchaseInfo) : ℝ :=
  info.bulb_price * info.bulb_quantity + info.motor_price * info.motor_quantity

/-- Theorem stating the properties of the purchase problem --/
theorem purchase_problem :
  ∃ (info : PurchaseInfo),
    -- Conditions
    info.bulb_price + info.motor_price = 12 ∧
    info.bulb_price * info.bulb_quantity = 30 ∧
    info.motor_price * info.motor_quantity = 45 ∧
    info.bulb_quantity = 2 * info.motor_quantity ∧
    -- Results
    info.bulb_price = 3 ∧
    info.motor_price = 9 ∧
    -- Optimal purchase
    (∀ (alt_info : PurchaseInfo),
      alt_info.bulb_quantity + alt_info.motor_quantity = 90 ∧
      alt_info.bulb_quantity ≤ alt_info.motor_quantity / 2 →
      total_cost info ≤ total_cost alt_info) ∧
    info.bulb_quantity = 30 ∧
    info.motor_quantity = 60 ∧
    total_cost info = 630 :=
  sorry


end NUMINAMATH_CALUDE_purchase_problem_l3298_329895


namespace NUMINAMATH_CALUDE_street_paths_l3298_329896

theorem street_paths (P Q : ℕ) (h1 : P = 130) (h2 : Q = 65) : P - 2*Q + 2014 = 2014 := by
  sorry

end NUMINAMATH_CALUDE_street_paths_l3298_329896


namespace NUMINAMATH_CALUDE_relatively_prime_squares_l3298_329867

theorem relatively_prime_squares (a b c : ℤ) 
  (h_coprime : ∀ d : ℤ, d ∣ a ∧ d ∣ b ∧ d ∣ c → d = 1 ∨ d = -1)
  (h_eq : 1 / a + 1 / b = 1 / c) :
  ∃ (p q r : ℤ), (a + b = p^2) ∧ (a - c = q^2) ∧ (b - c = r^2) := by
  sorry

end NUMINAMATH_CALUDE_relatively_prime_squares_l3298_329867


namespace NUMINAMATH_CALUDE_cube_root_sum_equals_sixty_l3298_329810

theorem cube_root_sum_equals_sixty : 
  (30^3 + 40^3 + 50^3 : ℝ)^(1/3) = 60 := by sorry

end NUMINAMATH_CALUDE_cube_root_sum_equals_sixty_l3298_329810


namespace NUMINAMATH_CALUDE_tangent_segments_area_l3298_329875

/-- The area of the region formed by all line segments of length 6 that are tangent to a circle with radius 4 at their midpoints -/
theorem tangent_segments_area (r : ℝ) (l : ℝ) (h_r : r = 4) (h_l : l = 6) :
  let outer_radius := Real.sqrt (r^2 + (l/2)^2)
  (π * outer_radius^2 - π * r^2) = 9 * π :=
sorry

end NUMINAMATH_CALUDE_tangent_segments_area_l3298_329875


namespace NUMINAMATH_CALUDE_f_max_value_l3298_329880

/-- The function f(z) = -6z^2 + 24z - 12 -/
def f (z : ℝ) : ℝ := -6 * z^2 + 24 * z - 12

theorem f_max_value :
  (∀ z : ℝ, f z ≤ 12) ∧ (∃ z : ℝ, f z = 12) := by sorry

end NUMINAMATH_CALUDE_f_max_value_l3298_329880


namespace NUMINAMATH_CALUDE_intersection_when_a_is_one_union_equals_reals_iff_l3298_329882

-- Define sets A and B
def A : Set ℝ := {x | x^2 - 2*x - 3 > 0}
def B (a : ℝ) : Set ℝ := {x | x - 4*a ≤ 0}

-- Part I
theorem intersection_when_a_is_one :
  A ∩ B 1 = {x : ℝ | x < -1 ∨ (3 < x ∧ x ≤ 4)} := by sorry

-- Part II
theorem union_equals_reals_iff (a : ℝ) :
  A ∪ B a = Set.univ ↔ a ≥ 3/4 := by sorry

end NUMINAMATH_CALUDE_intersection_when_a_is_one_union_equals_reals_iff_l3298_329882


namespace NUMINAMATH_CALUDE_pet_ownership_percentage_l3298_329807

/-- Represents the school with students and their pet ownership. -/
structure School where
  total_students : ℕ
  cat_owners : ℕ
  dog_owners : ℕ
  rabbit_owners : ℕ
  h_no_multiple_pets : cat_owners + dog_owners + rabbit_owners ≤ total_students

/-- Calculates the percentage of students owning at least one pet. -/
def percentage_pet_owners (s : School) : ℚ :=
  (s.cat_owners + s.dog_owners + s.rabbit_owners : ℚ) / s.total_students * 100

/-- Theorem stating that in the given school, 48% of students own at least one pet. -/
theorem pet_ownership_percentage (s : School) 
    (h_total : s.total_students = 500)
    (h_cats : s.cat_owners = 80)
    (h_dogs : s.dog_owners = 120)
    (h_rabbits : s.rabbit_owners = 40) : 
    percentage_pet_owners s = 48 := by sorry

end NUMINAMATH_CALUDE_pet_ownership_percentage_l3298_329807


namespace NUMINAMATH_CALUDE_min_P_over_Q_l3298_329855

theorem min_P_over_Q (x P Q : ℝ) (hx : x > 0) (hP : P > 0) (hQ : Q > 0)
  (hP_def : x^2 + 1/x^2 = P) (hQ_def : x^3 - 1/x^3 = Q) :
  ∀ y : ℝ, y > 0 → y^2 + 1/y^2 = P → y^3 - 1/y^3 = Q → P / Q ≥ 1 / Real.sqrt 3 :=
by sorry

end NUMINAMATH_CALUDE_min_P_over_Q_l3298_329855


namespace NUMINAMATH_CALUDE_paula_meal_combinations_l3298_329846

/-- The number of meat options available --/
def meat_options : ℕ := 3

/-- The number of vegetable options available --/
def vegetable_options : ℕ := 5

/-- The number of dessert options available --/
def dessert_options : ℕ := 5

/-- The number of vegetables Paula must choose --/
def vegetables_to_choose : ℕ := 3

/-- Calculates the number of ways to choose k items from n items --/
def choose (n k : ℕ) : ℕ :=
  Nat.choose n k

/-- The total number of meal combinations Paula can construct --/
def total_meals : ℕ :=
  meat_options * choose vegetable_options vegetables_to_choose * dessert_options

theorem paula_meal_combinations :
  total_meals = 150 :=
sorry

end NUMINAMATH_CALUDE_paula_meal_combinations_l3298_329846


namespace NUMINAMATH_CALUDE_mixed_number_division_equality_l3298_329860

theorem mixed_number_division_equality :
  (4 + 2/3 + 5 + 1/4) / (3 + 1/2 - (2 + 3/5)) = 11 + 1/54 := by sorry

end NUMINAMATH_CALUDE_mixed_number_division_equality_l3298_329860


namespace NUMINAMATH_CALUDE_marbles_per_bag_is_ten_l3298_329850

/-- The number of marbles in each bag of blue marbles --/
def marbles_per_bag : ℕ := sorry

/-- The initial number of green marbles --/
def initial_green : ℕ := 26

/-- The number of bags of blue marbles bought --/
def blue_bags : ℕ := 6

/-- The number of green marbles given away --/
def green_gift : ℕ := 6

/-- The number of blue marbles given away --/
def blue_gift : ℕ := 8

/-- The total number of marbles Janelle has after giving away the gift --/
def final_total : ℕ := 72

theorem marbles_per_bag_is_ten :
  (initial_green - green_gift) + (blue_bags * marbles_per_bag - blue_gift) = final_total →
  marbles_per_bag = 10 := by
  sorry

end NUMINAMATH_CALUDE_marbles_per_bag_is_ten_l3298_329850


namespace NUMINAMATH_CALUDE_eight_divided_by_point_three_repeating_l3298_329889

theorem eight_divided_by_point_three_repeating (x : ℚ) : x = 1/3 → 8 / x = 24 := by
  sorry

end NUMINAMATH_CALUDE_eight_divided_by_point_three_repeating_l3298_329889


namespace NUMINAMATH_CALUDE_election_ratio_l3298_329840

theorem election_ratio (R D : ℝ) : 
  R > 0 ∧ D > 0 →  -- Positive number of Republicans and Democrats
  (0.9 * R + 0.15 * D) / (R + D) = 0.7 →  -- Candidate X's vote share
  (0.1 * R + 0.85 * D) / (R + D) = 0.3 →  -- Candidate Y's vote share
  R / D = 2.75 := by
sorry

end NUMINAMATH_CALUDE_election_ratio_l3298_329840


namespace NUMINAMATH_CALUDE_cos_squared_minus_sin_squared_15_deg_l3298_329831

theorem cos_squared_minus_sin_squared_15_deg : 
  Real.cos (15 * π / 180) ^ 2 - Real.sin (15 * π / 180) ^ 2 = Real.sqrt 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_cos_squared_minus_sin_squared_15_deg_l3298_329831


namespace NUMINAMATH_CALUDE_dandelion_puff_distribution_l3298_329876

theorem dandelion_puff_distribution (total : ℕ) (given_away : ℕ) (friends : ℕ) 
  (h1 : total = 100) 
  (h2 : given_away = 42) 
  (h3 : friends = 7) :
  (total - given_away) / friends = 8 ∧ 
  (8 : ℚ) / (total - given_away) = 4 / 29 := by
  sorry

end NUMINAMATH_CALUDE_dandelion_puff_distribution_l3298_329876


namespace NUMINAMATH_CALUDE_quadratic_root_c_value_l3298_329818

theorem quadratic_root_c_value (c : ℝ) : 
  (∀ x : ℝ, 2 * x^2 + 13 * x + c = 0 ↔ x = (-13 + Real.sqrt 19) / 4 ∨ x = (-13 - Real.sqrt 19) / 4) →
  c = 18.75 := by
sorry

end NUMINAMATH_CALUDE_quadratic_root_c_value_l3298_329818


namespace NUMINAMATH_CALUDE_historian_writing_speed_l3298_329856

/-- Given a historian who wrote 60,000 words in 150 hours,
    prove that the average number of words written per hour is 400. -/
theorem historian_writing_speed :
  let total_words : ℕ := 60000
  let total_hours : ℕ := 150
  let average_words_per_hour : ℚ := total_words / total_hours
  average_words_per_hour = 400 := by
  sorry

end NUMINAMATH_CALUDE_historian_writing_speed_l3298_329856


namespace NUMINAMATH_CALUDE_arithmetic_mean_reciprocals_first_five_primes_l3298_329852

def first_five_primes : List Nat := [2, 3, 5, 7, 11]

theorem arithmetic_mean_reciprocals_first_five_primes :
  let reciprocals := first_five_primes.map (λ x => (1 : ℚ) / x)
  (reciprocals.sum / reciprocals.length) = 2927 / 11550 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_mean_reciprocals_first_five_primes_l3298_329852


namespace NUMINAMATH_CALUDE_min_value_of_function_l3298_329890

theorem min_value_of_function (x : ℝ) (h : x > 2) : 
  (4 / (x - 2) + x) ≥ 6 := by
sorry

end NUMINAMATH_CALUDE_min_value_of_function_l3298_329890


namespace NUMINAMATH_CALUDE_complex_ratio_theorem_l3298_329844

theorem complex_ratio_theorem (z₁ z₂ : ℂ) 
  (h₁ : Complex.abs z₁ = 1)
  (h₂ : Complex.abs z₂ = (5/2 : ℝ))
  (h₃ : Complex.abs (3 * z₁ - 2 * z₂) = 7) :
  z₁ / z₂ = -1/5 + Complex.I * Real.sqrt 3 / 5 ∨
  z₁ / z₂ = -1/5 - Complex.I * Real.sqrt 3 / 5 := by
sorry

end NUMINAMATH_CALUDE_complex_ratio_theorem_l3298_329844


namespace NUMINAMATH_CALUDE_smallest_gcd_bc_l3298_329864

theorem smallest_gcd_bc (a b c : ℕ+) (h1 : Nat.gcd a b = 255) (h2 : Nat.gcd a c = 855) :
  ∃ (b' c' : ℕ+), Nat.gcd a b' = 255 ∧ Nat.gcd a c' = 855 ∧ 
    Nat.gcd b' c' = 15 ∧ 
    ∀ (b'' c'' : ℕ+), Nat.gcd a b'' = 255 → Nat.gcd a c'' = 855 → 
      Nat.gcd b'' c'' ≥ 15 :=
by sorry

end NUMINAMATH_CALUDE_smallest_gcd_bc_l3298_329864


namespace NUMINAMATH_CALUDE_class_presentation_periods_l3298_329868

/-- The number of periods required for all student presentations in a class --/
def periods_required (total_students : ℕ) (period_length : ℕ) (individual_presentation_length : ℕ) 
  (group_presentation_length : ℕ) (group_presentations : ℕ) : ℕ :=
  let individual_students := total_students - group_presentations
  let total_minutes := individual_students * individual_presentation_length + 
                       group_presentations * group_presentation_length
  (total_minutes + period_length - 1) / period_length

theorem class_presentation_periods :
  periods_required 32 40 8 12 4 = 7 := by
  sorry

end NUMINAMATH_CALUDE_class_presentation_periods_l3298_329868


namespace NUMINAMATH_CALUDE_school_teachers_count_l3298_329834

/-- The number of departments in the school -/
def num_departments : ℕ := 7

/-- The number of teachers in each department -/
def teachers_per_department : ℕ := 20

/-- The total number of teachers in the school -/
def total_teachers : ℕ := num_departments * teachers_per_department

theorem school_teachers_count : total_teachers = 140 := by
  sorry

end NUMINAMATH_CALUDE_school_teachers_count_l3298_329834


namespace NUMINAMATH_CALUDE_f_minimum_at_negative_one_l3298_329859

-- Define the function f(x) = xe^x
noncomputable def f (x : ℝ) : ℝ := x * Real.exp x

-- State the theorem
theorem f_minimum_at_negative_one :
  IsLocalMin f (-1 : ℝ) := by sorry

end NUMINAMATH_CALUDE_f_minimum_at_negative_one_l3298_329859


namespace NUMINAMATH_CALUDE_amount_paid_l3298_329822

def lemonade_cups : ℕ := 2
def lemonade_price : ℚ := 2
def sandwich_count : ℕ := 2
def sandwich_price : ℚ := 2.5
def change_received : ℚ := 11

def total_cost : ℚ := lemonade_cups * lemonade_price + sandwich_count * sandwich_price

theorem amount_paid (paid : ℚ) : paid = 20 ↔ paid = total_cost + change_received := by
  sorry

end NUMINAMATH_CALUDE_amount_paid_l3298_329822


namespace NUMINAMATH_CALUDE_sum_90_is_neg_180_l3298_329828

/-- Represents an arithmetic progression -/
structure ArithmeticProgression where
  a : ℚ  -- First term
  d : ℚ  -- Common difference

/-- Sum of the first n terms of an arithmetic progression -/
def sum_n (ap : ArithmeticProgression) (n : ℕ) : ℚ :=
  (n : ℚ) / 2 * (2 * ap.a + (n - 1 : ℚ) * ap.d)

/-- Theorem: For the given arithmetic progression, the sum of the first 90 terms is -180 -/
theorem sum_90_is_neg_180 (ap : ArithmeticProgression) 
  (h15 : sum_n ap 15 = 150)
  (h75 : sum_n ap 75 = 30) : 
  sum_n ap 90 = -180 := by
  sorry

end NUMINAMATH_CALUDE_sum_90_is_neg_180_l3298_329828


namespace NUMINAMATH_CALUDE_monotone_decreasing_implies_k_nonpositive_l3298_329839

/-- A function f(x) = kx² + (3k-2)x - 5 is monotonically decreasing on [1, +∞) -/
def is_monotone_decreasing (f : ℝ → ℝ) (k : ℝ) : Prop :=
  ∀ x y, 1 ≤ x ∧ x < y → f y < f x

/-- The main theorem stating that if f(x) = kx² + (3k-2)x - 5 is monotonically
    decreasing on [1, +∞), then k ∈ (-∞, 0] -/
theorem monotone_decreasing_implies_k_nonpositive (k : ℝ) :
  is_monotone_decreasing (fun x => k*x^2 + (3*k-2)*x - 5) k →
  k ∈ Set.Iic 0 :=
by sorry

end NUMINAMATH_CALUDE_monotone_decreasing_implies_k_nonpositive_l3298_329839


namespace NUMINAMATH_CALUDE_expo_arrangements_l3298_329897

/-- The number of ways to arrange n distinct objects. -/
def arrangements (n : ℕ) : ℕ := Nat.factorial n

/-- The number of ways to choose k objects from n distinct objects. -/
def choose (n k : ℕ) : ℕ := Nat.choose n k

/-- The number of volunteers. -/
def num_volunteers : ℕ := 5

/-- The number of foreign friends. -/
def num_foreign_friends : ℕ := 2

/-- The total number of people. -/
def total_people : ℕ := num_volunteers + num_foreign_friends

/-- The number of positions where the foreign friends can be placed. -/
def foreign_friend_positions : ℕ := total_people - num_foreign_friends - 1

theorem expo_arrangements : 
  choose foreign_friend_positions 1 * arrangements num_volunteers * arrangements num_foreign_friends = 960 := by
  sorry

end NUMINAMATH_CALUDE_expo_arrangements_l3298_329897


namespace NUMINAMATH_CALUDE_gate_width_scientific_notation_l3298_329881

theorem gate_width_scientific_notation :
  ∃ (a : ℝ) (n : ℤ), 0.000000014 = a * 10^n ∧ 1 ≤ a ∧ a < 10 ∧ a = 1.4 ∧ n = -8 := by
  sorry

end NUMINAMATH_CALUDE_gate_width_scientific_notation_l3298_329881


namespace NUMINAMATH_CALUDE_norris_money_left_l3298_329804

def savings_september : ℕ := 29
def savings_october : ℕ := 25
def savings_november : ℕ := 31
def spending_game : ℕ := 75

theorem norris_money_left : 
  savings_september + savings_october + savings_november - spending_game = 10 := by
  sorry

end NUMINAMATH_CALUDE_norris_money_left_l3298_329804


namespace NUMINAMATH_CALUDE_emily_took_55_apples_l3298_329877

/-- The number of apples Ruby initially had -/
def initial_apples : ℕ := 63

/-- The number of apples Ruby has left -/
def remaining_apples : ℕ := 8

/-- The number of apples Emily took -/
def emily_took : ℕ := initial_apples - remaining_apples

/-- Theorem stating that Emily took 55 apples -/
theorem emily_took_55_apples : emily_took = 55 := by
  sorry

end NUMINAMATH_CALUDE_emily_took_55_apples_l3298_329877


namespace NUMINAMATH_CALUDE_union_equal_iff_a_geq_one_l3298_329857

/-- The set A defined as {x | 2 ≤ x ≤ 6} -/
def A : Set ℝ := {x : ℝ | 2 ≤ x ∧ x ≤ 6}

/-- The set B defined as {x | 2a ≤ x ≤ a+3} -/
def B (a : ℝ) : Set ℝ := {x : ℝ | 2*a ≤ x ∧ x ≤ a+3}

/-- Theorem stating that A ∪ B = A if and only if a ≥ 1 -/
theorem union_equal_iff_a_geq_one (a : ℝ) : A ∪ B a = A ↔ a ≥ 1 := by sorry

end NUMINAMATH_CALUDE_union_equal_iff_a_geq_one_l3298_329857


namespace NUMINAMATH_CALUDE_perspective_drawing_preserves_parallel_equal_l3298_329816

/-- A plane figure -/
structure PlaneFigure where
  -- Add necessary fields

/-- A perspective drawing of a plane figure -/
structure PerspectiveDrawing where
  -- Add necessary fields

/-- A line segment in a plane figure or perspective drawing -/
structure LineSegment where
  -- Add necessary fields

/-- Predicate to check if two line segments are parallel -/
def are_parallel (s1 s2 : LineSegment) : Prop := sorry

/-- Predicate to check if two line segments are equal in length -/
def are_equal (s1 s2 : LineSegment) : Prop := sorry

/-- Function to get the corresponding line segments in a perspective drawing -/
def perspective_line_segments (pf : PlaneFigure) (pd : PerspectiveDrawing) (s1 s2 : LineSegment) : 
  (LineSegment × LineSegment) := sorry

theorem perspective_drawing_preserves_parallel_equal 
  (pf : PlaneFigure) (pd : PerspectiveDrawing) (s1 s2 : LineSegment) :
  are_parallel s1 s2 → are_equal s1 s2 → 
  let (p1, p2) := perspective_line_segments pf pd s1 s2
  are_parallel p1 p2 ∧ are_equal p1 p2 :=
by sorry

end NUMINAMATH_CALUDE_perspective_drawing_preserves_parallel_equal_l3298_329816


namespace NUMINAMATH_CALUDE_find_a_and_b_l3298_329894

-- Define the system of inequalities
def inequality_system (a b x : ℝ) : Prop :=
  (3 * x - 2 < a + 1) ∧ (6 - 2 * x < b + 2)

-- Define the solution set
def solution_set (x : ℝ) : Prop :=
  -1 < x ∧ x < 2

-- Theorem statement
theorem find_a_and_b :
  ∀ a b : ℝ,
  (∀ x : ℝ, inequality_system a b x ↔ solution_set x) →
  a = 3 ∧ b = 6 := by
  sorry

end NUMINAMATH_CALUDE_find_a_and_b_l3298_329894


namespace NUMINAMATH_CALUDE_battery_price_is_56_l3298_329845

/-- The price of a battery given the total cost of four tires and one battery, and the cost of each tire. -/
def battery_price (total_cost : ℕ) (tire_price : ℕ) : ℕ :=
  total_cost - 4 * tire_price

/-- Theorem stating that the battery price is $56 given the conditions. -/
theorem battery_price_is_56 :
  battery_price 224 42 = 56 := by
  sorry

end NUMINAMATH_CALUDE_battery_price_is_56_l3298_329845


namespace NUMINAMATH_CALUDE_tangerines_oranges_percentage_l3298_329873

/-- Represents the quantities of fruits in Tina's bag -/
structure FruitBag where
  apples : ℕ
  oranges : ℕ
  tangerines : ℕ
  grapes : ℕ
  kiwis : ℕ

/-- Calculates the total number of fruits in the bag -/
def totalFruits (bag : FruitBag) : ℕ :=
  bag.apples + bag.oranges + bag.tangerines + bag.grapes + bag.kiwis

/-- Calculates the number of tangerines and oranges in the bag -/
def tangerinesAndOranges (bag : FruitBag) : ℕ :=
  bag.tangerines + bag.oranges

/-- Theorem stating that the percentage of tangerines and oranges in the remaining fruits is 47.5% -/
theorem tangerines_oranges_percentage (initialBag : FruitBag)
    (h1 : initialBag.apples = 9)
    (h2 : initialBag.oranges = 5)
    (h3 : initialBag.tangerines = 17)
    (h4 : initialBag.grapes = 12)
    (h5 : initialBag.kiwis = 7) :
    let finalBag : FruitBag := {
      apples := initialBag.apples,
      oranges := initialBag.oranges - 2 + 3,
      tangerines := initialBag.tangerines - 10 + 6,
      grapes := initialBag.grapes - 4,
      kiwis := initialBag.kiwis - 3
    }
    (tangerinesAndOranges finalBag : ℚ) / (totalFruits finalBag : ℚ) * 100 = 47.5 := by
  sorry

end NUMINAMATH_CALUDE_tangerines_oranges_percentage_l3298_329873


namespace NUMINAMATH_CALUDE_min_value_expression_l3298_329865

theorem min_value_expression (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  (a + 3*c) / (a + 2*b + c) + 4*b / (a + b + 2*c) - 8*c / (a + b + 3*c) ≥ -17 + 12 * Real.sqrt 2 :=
sorry

end NUMINAMATH_CALUDE_min_value_expression_l3298_329865


namespace NUMINAMATH_CALUDE_cube_volume_ratio_l3298_329803

theorem cube_volume_ratio (e : ℝ) (h : e > 0) :
  let small_cube_volume := e^3
  let large_cube_volume := (4*e)^3
  large_cube_volume / small_cube_volume = 64 := by
  sorry

end NUMINAMATH_CALUDE_cube_volume_ratio_l3298_329803


namespace NUMINAMATH_CALUDE_closed_grid_path_even_length_l3298_329814

/-- A closed path on a grid -/
structure GridPath where
  up : ℕ
  down : ℕ
  right : ℕ
  left : ℕ
  closed : up = down ∧ right = left

/-- The length of a grid path -/
def GridPath.length (p : GridPath) : ℕ :=
  p.up + p.down + p.right + p.left

/-- Theorem: The length of any closed grid path is even -/
theorem closed_grid_path_even_length (p : GridPath) : 
  Even p.length := by
sorry

end NUMINAMATH_CALUDE_closed_grid_path_even_length_l3298_329814


namespace NUMINAMATH_CALUDE_exists_valid_nail_sequence_l3298_329826

/-- Represents a nail operation -/
inductive NailOp
| Blue1 | Blue2 | Blue3
| Red1 | Red2 | Red3

/-- Represents a sequence of nail operations -/
def NailSeq := List NailOp

/-- Checks if a nail sequence becomes trivial when a specific operation is removed -/
def becomes_trivial_without (seq : NailSeq) (op : NailOp) : Prop := sorry

/-- Checks if a nail sequence becomes trivial when two specific operations are removed -/
def becomes_trivial_without_two (seq : NailSeq) (op1 op2 : NailOp) : Prop := sorry

/-- The main theorem stating the existence of a valid nail sequence -/
theorem exists_valid_nail_sequence :
  ∃ (W : NailSeq),
    (∀ blue : NailOp, blue ∈ [NailOp.Blue1, NailOp.Blue2, NailOp.Blue3] →
      becomes_trivial_without W blue) ∧
    (∀ red1 red2 : NailOp, red1 ≠ red2 →
      red1 ∈ [NailOp.Red1, NailOp.Red2, NailOp.Red3] →
      red2 ∈ [NailOp.Red1, NailOp.Red2, NailOp.Red3] →
      becomes_trivial_without_two W red1 red2) ∧
    (∀ red : NailOp, red ∈ [NailOp.Red1, NailOp.Red2, NailOp.Red3] →
      ¬becomes_trivial_without W red) :=
sorry

end NUMINAMATH_CALUDE_exists_valid_nail_sequence_l3298_329826


namespace NUMINAMATH_CALUDE_range_of_m_l3298_329891

def p (x : ℝ) : Prop := 12 / (x + 2) ≥ 1

def q (x m : ℝ) : Prop := x^2 - 2*x + 1 - m^2 ≤ 0

theorem range_of_m (m : ℝ) :
  (m > 0) →
  (∀ x : ℝ, ¬(p x) → ¬(q x m)) →
  (∃ x : ℝ, ¬(p x) ∧ (q x m)) →
  (0 < m ∧ m < 3) :=
sorry

end NUMINAMATH_CALUDE_range_of_m_l3298_329891


namespace NUMINAMATH_CALUDE_fraction_zero_implies_x_negative_one_l3298_329872

theorem fraction_zero_implies_x_negative_one (x : ℝ) :
  (abs x - 1) / (x - 1) = 0 → x = -1 := by
  sorry

end NUMINAMATH_CALUDE_fraction_zero_implies_x_negative_one_l3298_329872


namespace NUMINAMATH_CALUDE_stratified_sampling_correct_l3298_329879

/-- Represents the number of employees in each age group -/
structure EmployeeGroups where
  over50 : ℕ
  between35and49 : ℕ
  under35 : ℕ

/-- Represents the sampling results for each age group -/
structure SamplingResult where
  over50 : ℕ
  between35and49 : ℕ
  under35 : ℕ

/-- Calculates the correct stratified sampling for given employee groups and sample size -/
def stratifiedSampling (groups : EmployeeGroups) (sampleSize : ℕ) : SamplingResult :=
  sorry

/-- The theorem statement for the stratified sampling problem -/
theorem stratified_sampling_correct 
  (groups : EmployeeGroups)
  (h1 : groups.over50 = 15)
  (h2 : groups.between35and49 = 45)
  (h3 : groups.under35 = 90)
  (h4 : groups.over50 + groups.between35and49 + groups.under35 = 150)
  (sampleSize : ℕ)
  (h5 : sampleSize = 30) :
  stratifiedSampling groups sampleSize = SamplingResult.mk 3 9 18 :=
by sorry

end NUMINAMATH_CALUDE_stratified_sampling_correct_l3298_329879


namespace NUMINAMATH_CALUDE_herd_division_l3298_329866

theorem herd_division (total : ℕ) (fourth_son : ℕ) : 
  (total : ℚ) / 3 + total / 5 + total / 6 + fourth_son = total ∧ 
  fourth_son = 19 → 
  total = 63 := by
sorry

end NUMINAMATH_CALUDE_herd_division_l3298_329866


namespace NUMINAMATH_CALUDE_chord_equation_l3298_329829

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a parabola defined by y² = 6x -/
def Parabola := {p : Point | p.y^2 = 6 * p.x}

/-- Represents a line in the form ax + by + c = 0 -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Checks if a point lies on a line -/
def Point.onLine (p : Point) (l : Line) : Prop :=
  l.a * p.x + l.b * p.y + l.c = 0

/-- Checks if a point bisects a chord of the parabola -/
def bisectsChord (p : Point) (l : Line) : Prop :=
  p ∈ Parabola ∧ 
  ∃ a b : Point, a ≠ b ∧ 
    a ∈ Parabola ∧ b ∈ Parabola ∧
    a.onLine l ∧ b.onLine l ∧
    p.x = (a.x + b.x) / 2 ∧ p.y = (a.y + b.y) / 2

/-- The main theorem to be proved -/
theorem chord_equation : 
  let p := Point.mk 4 1
  let l := Line.mk 3 (-1) (-11)
  p ∈ Parabola ∧ bisectsChord p l := by sorry

end NUMINAMATH_CALUDE_chord_equation_l3298_329829


namespace NUMINAMATH_CALUDE_function_form_proof_l3298_329823

theorem function_form_proof (f : ℝ → ℝ) 
  (h1 : ∀ x, |f x + Real.cos x ^ 2| ≤ 3/4)
  (h2 : ∀ x, |f x - Real.sin x ^ 2| ≤ 1/4) :
  ∀ x, f x = 3/4 - Real.cos x ^ 2 := by
  sorry

end NUMINAMATH_CALUDE_function_form_proof_l3298_329823


namespace NUMINAMATH_CALUDE_ps_length_approx_l3298_329898

/-- A quadrilateral with given side and diagonal segment lengths -/
structure Quadrilateral :=
  (QT TS PT TR PQ : ℝ)

/-- The length of PS in the quadrilateral -/
noncomputable def lengthPS (q : Quadrilateral) : ℝ :=
  Real.sqrt (q.PT^2 + q.TS^2 - 2 * q.PT * q.TS * (-((q.PQ^2 - q.PT^2 - q.QT^2) / (2 * q.PT * q.QT))))

/-- Theorem stating that for a quadrilateral with given measurements, PS ≈ 19.9 -/
theorem ps_length_approx (q : Quadrilateral) 
  (h1 : q.QT = 5) (h2 : q.TS = 7) (h3 : q.PT = 9) (h4 : q.TR = 4) (h5 : q.PQ = 7) : 
  ∃ (ε : ℝ), ε > 0 ∧ ε < 0.1 ∧ |lengthPS q - 19.9| < ε :=
sorry

end NUMINAMATH_CALUDE_ps_length_approx_l3298_329898


namespace NUMINAMATH_CALUDE_f_composition_negative_three_l3298_329854

noncomputable def f (x : ℝ) : ℝ :=
  if x ≤ 0 then 1 / (5 - x) else Real.log x / Real.log 4

theorem f_composition_negative_three : f (f (-3)) = -3/2 := by
  sorry

end NUMINAMATH_CALUDE_f_composition_negative_three_l3298_329854


namespace NUMINAMATH_CALUDE_three_digit_number_proof_l3298_329820

/-- Given a three-digit number satisfying specific conditions, prove it equals 824 -/
theorem three_digit_number_proof (x y z : ℕ) : 
  z^2 = x * y →
  y = (x + z) / 6 →
  100 * x + 10 * y + z - 396 = 100 * z + 10 * y + x →
  100 * x + 10 * y + z = 824 := by
  sorry

end NUMINAMATH_CALUDE_three_digit_number_proof_l3298_329820


namespace NUMINAMATH_CALUDE_vector_equality_iff_magnitude_and_parallel_l3298_329801

/-- Two plane vectors are equal if and only if their magnitudes are equal and they are parallel. -/
theorem vector_equality_iff_magnitude_and_parallel {a b : ℝ × ℝ} :
  a = b ↔ (‖a‖ = ‖b‖ ∧ ∃ (k : ℝ), a = k • b) :=
by sorry

end NUMINAMATH_CALUDE_vector_equality_iff_magnitude_and_parallel_l3298_329801


namespace NUMINAMATH_CALUDE_round_robin_tournament_matches_l3298_329825

theorem round_robin_tournament_matches (n : ℕ) (h : n = 6) : 
  (n * (n - 1)) / 2 = Nat.choose n 2 := by
  sorry

end NUMINAMATH_CALUDE_round_robin_tournament_matches_l3298_329825


namespace NUMINAMATH_CALUDE_hyperbola_parameter_l3298_329849

/-- Given a parabola y^2 = 16x and a hyperbola (x^2/a^2) - (y^2/b^2) = 1 where:
    1. The right focus of the hyperbola coincides with the focus of the parabola (4, 0)
    2. The left directrix of the hyperbola is x = -3
    Then a^2 = 12 -/
theorem hyperbola_parameter (a b : ℝ) : 
  (∃ (x y : ℝ), y^2 = 16*x) → -- Parabola exists
  (∃ (x y : ℝ), (x^2/a^2) - (y^2/b^2) = 1) → -- Hyperbola exists
  (4 : ℝ) = a^2/(2*a) → -- Right focus of hyperbola is (4, 0)
  (-3 : ℝ) = -a^2/(2*a) → -- Left directrix of hyperbola is x = -3
  a^2 = 12 := by sorry

end NUMINAMATH_CALUDE_hyperbola_parameter_l3298_329849


namespace NUMINAMATH_CALUDE_negation_of_proposition_l3298_329824

theorem negation_of_proposition (p : (x : ℝ) → x > 1 → x^3 + 1 > 8*x) :
  (¬ ∀ (x : ℝ), x > 1 → x^3 + 1 > 8*x) ↔ 
  (∃ (x : ℝ), x > 1 ∧ x^3 + 1 ≤ 8*x) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_proposition_l3298_329824


namespace NUMINAMATH_CALUDE_parabola_focus_directrix_relation_l3298_329800

/-- Represents a parabola with equation y^2 = 8px where p > 0 -/
structure Parabola where
  p : ℝ
  h_p_pos : p > 0

/-- The focus of a parabola -/
def focus (para : Parabola) : ℝ × ℝ := (2 * para.p, 0)

/-- The x-coordinate of the directrix of a parabola -/
def directrix_x (para : Parabola) : ℝ := -2 * para.p

/-- The distance from the focus to the directrix -/
def focus_directrix_distance (para : Parabola) : ℝ :=
  (focus para).1 - directrix_x para

theorem parabola_focus_directrix_relation (para : Parabola) :
  para.p = (1/4) * focus_directrix_distance para := by sorry

end NUMINAMATH_CALUDE_parabola_focus_directrix_relation_l3298_329800


namespace NUMINAMATH_CALUDE_decimal_to_fraction_l3298_329899

theorem decimal_to_fraction (x : ℚ) (h : x = 368/100) : x = 92/25 := by
  sorry

end NUMINAMATH_CALUDE_decimal_to_fraction_l3298_329899


namespace NUMINAMATH_CALUDE_largest_integer_less_than_100_with_remainder_5_mod_8_l3298_329835

theorem largest_integer_less_than_100_with_remainder_5_mod_8 :
  ∀ n : ℕ, n < 100 → n % 8 = 5 → n ≤ 93 :=
by
  sorry

end NUMINAMATH_CALUDE_largest_integer_less_than_100_with_remainder_5_mod_8_l3298_329835


namespace NUMINAMATH_CALUDE_complex_square_sum_zero_l3298_329809

theorem complex_square_sum_zero (i : ℂ) (h : i^2 = -1) : 
  (1 + i)^2 + (1 - i)^2 = 0 := by
  sorry

end NUMINAMATH_CALUDE_complex_square_sum_zero_l3298_329809


namespace NUMINAMATH_CALUDE_prob_three_tails_correct_l3298_329862

/-- Represents a coin with a given probability of heads -/
structure Coin where
  prob_heads : ℚ
  prob_heads_nonneg : 0 ≤ prob_heads
  prob_heads_le_one : prob_heads ≤ 1

/-- A fair coin with probability of heads = 1/2 -/
def fair_coin : Coin where
  prob_heads := 1/2
  prob_heads_nonneg := by norm_num
  prob_heads_le_one := by norm_num

/-- A biased coin with probability of heads = 2/3 -/
def biased_coin : Coin where
  prob_heads := 2/3
  prob_heads_nonneg := by norm_num
  prob_heads_le_one := by norm_num

/-- Sequence of coins: two fair coins, one biased coin, two fair coins -/
def coin_sequence : List Coin :=
  [fair_coin, fair_coin, biased_coin, fair_coin, fair_coin]

/-- Calculates the probability of getting at least 3 tails in a row -/
def prob_three_tails_in_row (coins : List Coin) : ℚ :=
  sorry

theorem prob_three_tails_correct :
  prob_three_tails_in_row coin_sequence = 13/48 := by
  sorry

end NUMINAMATH_CALUDE_prob_three_tails_correct_l3298_329862


namespace NUMINAMATH_CALUDE_dance_team_recruitment_l3298_329883

theorem dance_team_recruitment :
  ∀ (track_team choir dance_team : ℕ),
  track_team + choir + dance_team = 100 →
  choir = 2 * track_team →
  dance_team = choir + 10 →
  dance_team = 46 := by
sorry

end NUMINAMATH_CALUDE_dance_team_recruitment_l3298_329883


namespace NUMINAMATH_CALUDE_gcd_of_sequence_l3298_329821

theorem gcd_of_sequence (n : ℕ) : 
  ∃ d : ℕ, d > 0 ∧ 
  (∀ m : ℕ, d ∣ (7^(m+2) + 8^(2*m+1))) ∧
  (∀ k : ℕ, k > 0 → (∀ m : ℕ, k ∣ (7^(m+2) + 8^(2*m+1))) → k ≤ d) ∧
  d = 57 := by
sorry

end NUMINAMATH_CALUDE_gcd_of_sequence_l3298_329821


namespace NUMINAMATH_CALUDE_complex_number_simplification_l3298_329836

theorem complex_number_simplification :
  let i : ℂ := Complex.I
  let Z : ℂ := (2 + 4 * i) / (1 + i)
  Z = 3 + i := by sorry

end NUMINAMATH_CALUDE_complex_number_simplification_l3298_329836


namespace NUMINAMATH_CALUDE_solution_set_implies_a_b_values_solution_on_interval_implies_a_range_three_integer_solutions_implies_a_range_l3298_329853

-- Define the function f
def f (x a b : ℝ) : ℝ := x^2 + (3-a)*x + 2 + 2*a + b

-- Theorem 1
theorem solution_set_implies_a_b_values (a b : ℝ) :
  (∀ x, f x a b > 0 ↔ x < -4 ∨ x > 2) →
  a = 1 ∧ b = -12 := by sorry

-- Theorem 2
theorem solution_on_interval_implies_a_range (a b : ℝ) :
  (∃ x ∈ Set.Icc 1 3, f x a b ≤ b) →
  a ≤ -6 ∨ a ≥ 20 := by sorry

-- Theorem 3
theorem three_integer_solutions_implies_a_range (a b : ℝ) :
  (∃! (s : Finset ℤ), s.card = 3 ∧ ∀ x ∈ s, f x a b < 12 + b) →
  (3 ≤ a ∧ a < 4) ∨ (10 < a ∧ a ≤ 11) := by sorry

end NUMINAMATH_CALUDE_solution_set_implies_a_b_values_solution_on_interval_implies_a_range_three_integer_solutions_implies_a_range_l3298_329853


namespace NUMINAMATH_CALUDE_point_coordinate_product_l3298_329805

theorem point_coordinate_product : 
  ∀ y₁ y₂ : ℝ,
  (((4 - (-2))^2 + (y₁ - 5)^2 = 13^2) ∧
   ((4 - (-2))^2 + (y₂ - 5)^2 = 13^2) ∧
   (∀ y : ℝ, ((4 - (-2))^2 + (y - 5)^2 = 13^2) → (y = y₁ ∨ y = y₂))) →
  y₁ * y₂ = -108 := by
sorry

end NUMINAMATH_CALUDE_point_coordinate_product_l3298_329805


namespace NUMINAMATH_CALUDE_decimal_point_problem_l3298_329838

theorem decimal_point_problem (x : ℝ) (h1 : x > 0) (h2 : 1000 * x = 3 * (1 / x)) : 
  x = Real.sqrt 30 / 100 := by
  sorry

end NUMINAMATH_CALUDE_decimal_point_problem_l3298_329838


namespace NUMINAMATH_CALUDE_rose_additional_money_needed_l3298_329812

/-- The amount of additional money Rose needs to buy her art supplies -/
theorem rose_additional_money_needed 
  (paintbrush_cost : ℚ)
  (paints_cost : ℚ)
  (easel_cost : ℚ)
  (rose_current_money : ℚ)
  (h1 : paintbrush_cost = 2.40)
  (h2 : paints_cost = 9.20)
  (h3 : easel_cost = 6.50)
  (h4 : rose_current_money = 7.10) :
  paintbrush_cost + paints_cost + easel_cost - rose_current_money = 11 :=
by sorry

end NUMINAMATH_CALUDE_rose_additional_money_needed_l3298_329812


namespace NUMINAMATH_CALUDE_least_subtraction_for_divisibility_l3298_329871

def original_number : ℕ := 42398
def divisor : ℕ := 15
def number_to_subtract : ℕ := 8

theorem least_subtraction_for_divisibility :
  (∀ k : ℕ, k < number_to_subtract → ¬(divisor ∣ (original_number - k))) ∧
  (divisor ∣ (original_number - number_to_subtract)) := by
  sorry

end NUMINAMATH_CALUDE_least_subtraction_for_divisibility_l3298_329871


namespace NUMINAMATH_CALUDE_least_integer_with_1323_divisors_l3298_329858

/-- The number of distinct positive divisors of n -/
def num_divisors (n : ℕ) : ℕ := sorry

/-- Checks if n can be expressed as m * 30^k where 30 is not a divisor of m -/
def is_valid_form (n m k : ℕ) : Prop :=
  n = m * (30 ^ k) ∧ ¬(30 ∣ m)

theorem least_integer_with_1323_divisors :
  ∃ (n m k : ℕ),
    (∀ i < n, num_divisors i ≠ 1323) ∧
    num_divisors n = 1323 ∧
    is_valid_form n m k ∧
    m + k = 83 :=
sorry

end NUMINAMATH_CALUDE_least_integer_with_1323_divisors_l3298_329858


namespace NUMINAMATH_CALUDE_scientific_notation_equiv_l3298_329885

theorem scientific_notation_equiv : 
  0.0000006 = 6 * 10^(-7) := by sorry

end NUMINAMATH_CALUDE_scientific_notation_equiv_l3298_329885


namespace NUMINAMATH_CALUDE_train_speed_clicks_l3298_329808

theorem train_speed_clicks (x : ℝ) : x > 0 →
  let t := (2400 : ℝ) / 5280
  t ≠ 0.25 ∧ t ≠ 1 ∧ t ≠ 2 ∧ t ≠ 3 := by
  sorry

end NUMINAMATH_CALUDE_train_speed_clicks_l3298_329808
