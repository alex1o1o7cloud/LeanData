import Mathlib

namespace NUMINAMATH_CALUDE_jade_transactions_l2485_248584

theorem jade_transactions (mabel anthony cal jade : ℕ) : 
  mabel = 90 →
  anthony = mabel + mabel / 10 →
  cal = anthony * 2 / 3 →
  jade = cal + 19 →
  jade = 85 := by
  sorry

end NUMINAMATH_CALUDE_jade_transactions_l2485_248584


namespace NUMINAMATH_CALUDE_rhombus_perimeter_l2485_248552

/-- A rhombus with given diagonal lengths has a specific perimeter. -/
theorem rhombus_perimeter (d1 d2 : ℝ) (h1 : d1 = 12) (h2 : d2 = 16) :
  let side := Real.sqrt ((d1/2)^2 + (d2/2)^2)
  4 * side = 40 := by
  sorry

end NUMINAMATH_CALUDE_rhombus_perimeter_l2485_248552


namespace NUMINAMATH_CALUDE_two_roots_implies_c_values_l2485_248592

-- Define the function f(x) = x³ - 3x + c
def f (c : ℝ) (x : ℝ) : ℝ := x^3 - 3*x + c

-- State the theorem
theorem two_roots_implies_c_values (c : ℝ) :
  (∃ (x₁ x₂ : ℝ), x₁ ≠ x₂ ∧ f c x₁ = 0 ∧ f c x₂ = 0 ∧
    (∀ x : ℝ, f c x = 0 → x = x₁ ∨ x = x₂)) →
  c = -2 ∨ c = 2 := by
sorry

end NUMINAMATH_CALUDE_two_roots_implies_c_values_l2485_248592


namespace NUMINAMATH_CALUDE_trig_identity_proof_l2485_248572

theorem trig_identity_proof (θ : ℝ) : 
  Real.sin (θ + Real.pi / 180 * 75) + Real.cos (θ + Real.pi / 180 * 45) - Real.sqrt 3 * Real.cos (θ + Real.pi / 180 * 15) = 0 := by
  sorry

end NUMINAMATH_CALUDE_trig_identity_proof_l2485_248572


namespace NUMINAMATH_CALUDE_quinn_caught_four_frogs_l2485_248538

-- Define the number of frogs caught by each person
def alster_frogs : ℕ := 2
def bret_frogs : ℕ := 12

-- Define Quinn's frogs in terms of Alster's
def quinn_frogs : ℕ := alster_frogs

-- Define the relationship between Bret's and Quinn's frogs
axiom bret_quinn_relation : bret_frogs = 3 * quinn_frogs

theorem quinn_caught_four_frogs : quinn_frogs = 4 := by
  sorry

end NUMINAMATH_CALUDE_quinn_caught_four_frogs_l2485_248538


namespace NUMINAMATH_CALUDE_value_of_m_l2485_248574

/-- A function f(x) is a direct proportion function with respect to x if f(x) = kx for some constant k ≠ 0 -/
def IsDirectProportionFunction (f : ℝ → ℝ) : Prop :=
  ∃ k : ℝ, k ≠ 0 ∧ ∀ x, f x = k * x

/-- A function f(x) passes through the second and fourth quadrants if f(x) < 0 for x > 0 and f(x) > 0 for x < 0 -/
def PassesThroughSecondAndFourthQuadrants (f : ℝ → ℝ) : Prop :=
  (∀ x > 0, f x < 0) ∧ (∀ x < 0, f x > 0)

/-- The main theorem -/
theorem value_of_m (m : ℝ) :
  IsDirectProportionFunction (fun x ↦ (m - 2) * x^(m^2 - 8)) ∧
  PassesThroughSecondAndFourthQuadrants (fun x ↦ (m - 2) * x^(m^2 - 8)) →
  m = -3 := by
  sorry

end NUMINAMATH_CALUDE_value_of_m_l2485_248574


namespace NUMINAMATH_CALUDE_initial_pears_eq_sum_l2485_248513

/-- The number of pears Sara picked initially -/
def initial_pears : ℕ := sorry

/-- The number of apples Sara picked -/
def apples : ℕ := 27

/-- The number of pears Sara gave to Dan -/
def pears_given : ℕ := 28

/-- The number of pears Sara has left -/
def pears_left : ℕ := 7

/-- Theorem stating that the initial number of pears is equal to the sum of pears given and pears left -/
theorem initial_pears_eq_sum : initial_pears = pears_given + pears_left := by sorry

end NUMINAMATH_CALUDE_initial_pears_eq_sum_l2485_248513


namespace NUMINAMATH_CALUDE_f_of_f_3_l2485_248586

def f (x : ℝ) : ℝ := 3 * x^2 + 3 * x - 2

theorem f_of_f_3 : f (f 3) = 3568 := by
  sorry

end NUMINAMATH_CALUDE_f_of_f_3_l2485_248586


namespace NUMINAMATH_CALUDE_cafeteria_apples_l2485_248560

/-- The number of apples handed out to students -/
def apples_handed_out : ℕ := 8

/-- The number of apples needed for each pie -/
def apples_per_pie : ℕ := 9

/-- The number of pies that could be made with the remaining apples -/
def pies_made : ℕ := 6

/-- The initial number of apples in the cafeteria -/
def initial_apples : ℕ := 62

theorem cafeteria_apples :
  initial_apples = apples_handed_out + apples_per_pie * pies_made :=
by sorry

end NUMINAMATH_CALUDE_cafeteria_apples_l2485_248560


namespace NUMINAMATH_CALUDE_largest_prime_factors_difference_l2485_248588

-- Define the number we're working with
def n : ℕ := 175616

-- State the theorem
theorem largest_prime_factors_difference (p q : ℕ) : 
  (Nat.Prime p ∧ Nat.Prime q ∧ p ∣ n ∧ q ∣ n ∧ 
   ∀ r, Nat.Prime r → r ∣ n → r ≤ p ∧ r ≤ q) → 
  p - q = 5 ∨ q - p = 5 :=
sorry

end NUMINAMATH_CALUDE_largest_prime_factors_difference_l2485_248588


namespace NUMINAMATH_CALUDE_work_completion_time_l2485_248532

/-- The time needed to complete the work -/
def complete_work (p q : ℝ) (t : ℝ) : Prop :=
  let work_p := t / p
  let work_q := (t - 16) / q
  work_p + work_q = 1

theorem work_completion_time :
  ∀ p q : ℝ,
  p > 0 → q > 0 →
  complete_work p q 40 →
  complete_work q q 24 →
  ∃ t : ℝ, t > 0 ∧ complete_work p q t ∧ t = 25 := by
sorry

end NUMINAMATH_CALUDE_work_completion_time_l2485_248532


namespace NUMINAMATH_CALUDE_inscribed_angle_theorem_l2485_248580

/-- Represents a circle -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- Represents an arc on a circle -/
structure Arc (c : Circle) where
  start_point : ℝ × ℝ
  end_point : ℝ × ℝ

/-- The angle subtended by an arc at the center of the circle -/
def central_angle (c : Circle) (a : Arc c) : ℝ :=
  sorry

/-- The angle subtended by an arc at a point on the circumference of the circle -/
def inscribed_angle (c : Circle) (a : Arc c) : ℝ :=
  sorry

/-- The Inscribed Angle Theorem -/
theorem inscribed_angle_theorem (c : Circle) (a : Arc c) :
  inscribed_angle c a = (1 / 2) * central_angle c a :=
sorry

end NUMINAMATH_CALUDE_inscribed_angle_theorem_l2485_248580


namespace NUMINAMATH_CALUDE_function_inequality_l2485_248526

open Set
open Real

theorem function_inequality (f : ℝ → ℝ) (f' : ℝ → ℝ) (h1 : ∀ x, HasDerivAt f (f' x) x) 
  (h2 : ∀ x, f x + f' x > 2) (h3 : f 0 = 2021) :
  ∀ x, f x > 2 + 2019 / exp x ↔ x > 0 := by
sorry

end NUMINAMATH_CALUDE_function_inequality_l2485_248526


namespace NUMINAMATH_CALUDE_solve_for_a_l2485_248500

theorem solve_for_a (a : ℝ) (h : 2 * 2^2 * a = 2^6) : a = 8 := by
  sorry

end NUMINAMATH_CALUDE_solve_for_a_l2485_248500


namespace NUMINAMATH_CALUDE_only_two_consecutive_primes_l2485_248531

theorem only_two_consecutive_primes : ∀ p : ℕ, 
  (Nat.Prime p ∧ Nat.Prime (p + 1)) → p = 2 := by
  sorry

end NUMINAMATH_CALUDE_only_two_consecutive_primes_l2485_248531


namespace NUMINAMATH_CALUDE_water_evaporation_period_l2485_248598

theorem water_evaporation_period (initial_water : ℝ) (daily_evaporation : ℝ) (evaporation_percentage : ℝ) :
  initial_water = 10 →
  daily_evaporation = 0.012 →
  evaporation_percentage = 0.06 →
  (initial_water * evaporation_percentage) / daily_evaporation = 50 :=
by sorry

end NUMINAMATH_CALUDE_water_evaporation_period_l2485_248598


namespace NUMINAMATH_CALUDE_four_students_three_communities_l2485_248523

/-- The number of ways to distribute students among communities -/
def distribute_students (num_students : ℕ) (num_communities : ℕ) : ℕ :=
  sorry

/-- Theorem stating the correct number of arrangements for 4 students and 3 communities -/
theorem four_students_three_communities :
  distribute_students 4 3 = 36 := by sorry

end NUMINAMATH_CALUDE_four_students_three_communities_l2485_248523


namespace NUMINAMATH_CALUDE_lateral_surface_area_square_pyramid_l2485_248522

/-- Lateral surface area of a regular square pyramid -/
theorem lateral_surface_area_square_pyramid 
  (base_edge : ℝ) 
  (height : ℝ) 
  (h : base_edge = 2 * Real.sqrt 3) 
  (h' : height = 1) : 
  4 * (1/2 * base_edge * Real.sqrt (base_edge^2/4 + height^2)) = 8 * Real.sqrt 3 :=
by sorry

end NUMINAMATH_CALUDE_lateral_surface_area_square_pyramid_l2485_248522


namespace NUMINAMATH_CALUDE_smallest_positive_integer_e_l2485_248517

theorem smallest_positive_integer_e (a b c d e : ℤ) : 
  (∀ x : ℚ, a * x^4 + b * x^3 + c * x^2 + d * x + e = 0 ↔ 
    x = -4 ∨ x = 6 ∨ x = 10 ∨ x = -1/2) →
  e > 0 →
  (∀ e' : ℤ, e' > 0 → 
    (∀ x : ℚ, a * x^4 + b * x^3 + c * x^2 + d * x + e' = 0 ↔ 
      x = -4 ∨ x = 6 ∨ x = 10 ∨ x = -1/2) → 
    e ≤ e') →
  e = 200 := by
sorry

end NUMINAMATH_CALUDE_smallest_positive_integer_e_l2485_248517


namespace NUMINAMATH_CALUDE_necessary_but_not_sufficient_condition_l2485_248583

theorem necessary_but_not_sufficient_condition (a b : ℝ) :
  (∀ x y : ℝ, x > y → x > y - 2) ∧
  ¬(∀ x y : ℝ, x > y - 2 → x > y) :=
by sorry

end NUMINAMATH_CALUDE_necessary_but_not_sufficient_condition_l2485_248583


namespace NUMINAMATH_CALUDE_sum_of_roots_l2485_248597

noncomputable def f (x : ℝ) : ℝ := x^2 - 2*|x + 4| - 27

theorem sum_of_roots : ∃ (r₁ r₂ : ℝ), f r₁ = 0 ∧ f r₂ = 0 ∧ r₁ + r₂ = 6 - Real.sqrt 20 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_roots_l2485_248597


namespace NUMINAMATH_CALUDE_midpoint_triangle_area_l2485_248528

/-- The area of the nth triangle formed by repeatedly connecting midpoints -/
def triangleArea (n : ℕ) : ℚ :=
  (1 / 4 : ℚ) ^ (n - 1) * (3 / 2 : ℚ)

/-- The original right triangle ABC with sides 3, 4, and 5 -/
structure OriginalTriangle where
  sideA : ℕ := 3
  sideB : ℕ := 4
  sideC : ℕ := 5

theorem midpoint_triangle_area (t : OriginalTriangle) (n : ℕ) (h : n ≥ 1) :
  triangleArea n = (1 / 4 : ℚ) ^ (n - 1) * (3 / 2 : ℚ) :=
by sorry

end NUMINAMATH_CALUDE_midpoint_triangle_area_l2485_248528


namespace NUMINAMATH_CALUDE_intersection_sum_l2485_248567

/-- Two lines intersect at a point -/
def intersect_at (m b : ℝ) (x y : ℝ) : Prop :=
  y = m * x + 6 ∧ y = 4 * x + b

/-- The theorem statement -/
theorem intersection_sum (m b : ℝ) :
  intersect_at m b 8 14 → b + m = -17 := by
  sorry

end NUMINAMATH_CALUDE_intersection_sum_l2485_248567


namespace NUMINAMATH_CALUDE_max_visible_cubes_eq_400_l2485_248529

/-- The size of the cube's edge -/
def cube_size : ℕ := 12

/-- The number of unit cubes visible on a single face -/
def face_count : ℕ := cube_size ^ 2

/-- The number of unit cubes overcounted on each edge -/
def edge_overcount : ℕ := cube_size - 1

/-- The maximum number of visible unit cubes from a single point -/
def max_visible_cubes : ℕ := 3 * face_count - 3 * edge_overcount + 1

theorem max_visible_cubes_eq_400 : max_visible_cubes = 400 := by
  sorry

end NUMINAMATH_CALUDE_max_visible_cubes_eq_400_l2485_248529


namespace NUMINAMATH_CALUDE_hyperbola_m_range_l2485_248551

-- Define the hyperbola equation
def hyperbola_equation (x y m : ℝ) : Prop :=
  x^2 / (m^2 - 4) - y^2 / (m + 1) = 1

-- Define the condition that foci are on y-axis
def foci_on_y_axis (m : ℝ) : Prop :=
  -(m + 1) > 0 ∧ 4 - m^2 > 0

-- Theorem statement
theorem hyperbola_m_range :
  ∀ m : ℝ, (∃ x y : ℝ, hyperbola_equation x y m) ∧ foci_on_y_axis m → 
  m > -2 ∧ m < -1 :=
sorry

end NUMINAMATH_CALUDE_hyperbola_m_range_l2485_248551


namespace NUMINAMATH_CALUDE_houses_in_block_l2485_248533

theorem houses_in_block (junk_mails_per_house : ℕ) (total_junk_mails_per_block : ℕ) 
  (h1 : junk_mails_per_house = 2) 
  (h2 : total_junk_mails_per_block = 14) : 
  total_junk_mails_per_block / junk_mails_per_house = 7 := by
  sorry

end NUMINAMATH_CALUDE_houses_in_block_l2485_248533


namespace NUMINAMATH_CALUDE_village_population_equality_l2485_248515

-- Define the initial populations and known rate of decrease
def population_X : ℕ := 70000
def population_Y : ℕ := 42000
def decrease_rate_X : ℕ := 1200
def years : ℕ := 14

-- Define the unknown rate of increase for Village Y
def increase_rate_Y : ℕ := sorry

-- Theorem statement
theorem village_population_equality :
  population_X - years * decrease_rate_X = population_Y + years * increase_rate_Y ∧
  increase_rate_Y = 800 := by sorry

end NUMINAMATH_CALUDE_village_population_equality_l2485_248515


namespace NUMINAMATH_CALUDE_bobby_toy_cars_increase_l2485_248544

/-- The annual percentage increase in Bobby's toy cars -/
def annual_increase : ℝ := 0.5

theorem bobby_toy_cars_increase :
  let initial_cars : ℝ := 16
  let years : ℕ := 3
  let final_cars : ℝ := 54
  initial_cars * (1 + annual_increase) ^ years = final_cars :=
by sorry

end NUMINAMATH_CALUDE_bobby_toy_cars_increase_l2485_248544


namespace NUMINAMATH_CALUDE_apple_cost_price_l2485_248540

/-- The cost price of an apple, given its selling price and loss ratio. -/
def cost_price (selling_price : ℚ) (loss_ratio : ℚ) : ℚ :=
  selling_price / (1 - loss_ratio)

/-- Theorem: The cost price of an apple is 20.4 when sold for 17 with a 1/6 loss. -/
theorem apple_cost_price :
  cost_price 17 (1/6) = 20.4 := by
sorry

end NUMINAMATH_CALUDE_apple_cost_price_l2485_248540


namespace NUMINAMATH_CALUDE_sum_divisible_by_addends_l2485_248501

theorem sum_divisible_by_addends : 
  ∃ (a b c : ℕ), 
    a ≠ b ∧ b ≠ c ∧ a ≠ c ∧ 
    (a + b + c) % a = 0 ∧ 
    (a + b + c) % b = 0 ∧ 
    (a + b + c) % c = 0 :=
by sorry

end NUMINAMATH_CALUDE_sum_divisible_by_addends_l2485_248501


namespace NUMINAMATH_CALUDE_rectangular_window_area_l2485_248585

/-- The area of a rectangular window with length 47.3 cm and width 24 cm is 1135.2 cm². -/
theorem rectangular_window_area : 
  let length : ℝ := 47.3
  let width : ℝ := 24
  let area : ℝ := length * width
  area = 1135.2 := by sorry

end NUMINAMATH_CALUDE_rectangular_window_area_l2485_248585


namespace NUMINAMATH_CALUDE_agent_007_encryption_possible_l2485_248520

theorem agent_007_encryption_possible : ∃ (m n : ℕ), (1 : ℚ) / m + (1 : ℚ) / n = (7 : ℚ) / 100 := by
  sorry

end NUMINAMATH_CALUDE_agent_007_encryption_possible_l2485_248520


namespace NUMINAMATH_CALUDE_three_numbers_problem_l2485_248518

theorem three_numbers_problem :
  ∃ (a b c : ℕ),
    (Nat.gcd a b = 8) ∧
    (Nat.gcd b c = 2) ∧
    (Nat.gcd a c = 6) ∧
    (Nat.lcm (Nat.lcm a b) c = 1680) ∧
    (max a (max b c) > 100) ∧
    (max a (max b c) ≤ 200) ∧
    ((∃ n : ℕ, a = n^4) ∨ (∃ n : ℕ, b = n^4) ∨ (∃ n : ℕ, c = n^4)) ∧
    ((a = 120 ∧ b = 16 ∧ c = 42) ∨ (a = 168 ∧ b = 16 ∧ c = 30)) :=
by
  sorry

#check three_numbers_problem

end NUMINAMATH_CALUDE_three_numbers_problem_l2485_248518


namespace NUMINAMATH_CALUDE_minimum_bailing_rate_l2485_248566

/-- Represents the problem of determining the minimum bailing rate for a leaking boat --/
theorem minimum_bailing_rate 
  (distance_to_shore : ℝ) 
  (water_intake_rate : ℝ) 
  (max_water_capacity : ℝ) 
  (rowing_speed : ℝ) 
  (h1 : distance_to_shore = 2) 
  (h2 : water_intake_rate = 15) 
  (h3 : max_water_capacity = 50) 
  (h4 : rowing_speed = 3) : 
  ∃ (min_bailing_rate : ℝ), 
    13 < min_bailing_rate ∧ 
    min_bailing_rate ≤ 14 ∧ 
    (distance_to_shore / rowing_speed) * water_intake_rate - 
      (distance_to_shore / rowing_speed) * min_bailing_rate ≤ max_water_capacity :=
by sorry

end NUMINAMATH_CALUDE_minimum_bailing_rate_l2485_248566


namespace NUMINAMATH_CALUDE_total_cost_is_18_l2485_248553

-- Define the cost of a single soda
def soda_cost : ℝ := 1

-- Define the cost of a single soup
def soup_cost : ℝ := 3 * soda_cost

-- Define the cost of a sandwich
def sandwich_cost : ℝ := 3 * soup_cost

-- Define the total cost
def total_cost : ℝ := 3 * soda_cost + 2 * soup_cost + sandwich_cost

-- Theorem statement
theorem total_cost_is_18 : total_cost = 18 := by
  sorry

end NUMINAMATH_CALUDE_total_cost_is_18_l2485_248553


namespace NUMINAMATH_CALUDE_sequence_sum_l2485_248579

theorem sequence_sum (n : ℕ) (x : ℕ → ℝ) (h1 : x 1 = 3) 
  (h2 : ∀ k ∈ Finset.range (n - 1), x (k + 1) = x k + k) : 
  Finset.sum (Finset.range n) (λ k => x (k + 1)) = 3*n + (n*(n+1)*(2*n-1))/12 := by
  sorry

end NUMINAMATH_CALUDE_sequence_sum_l2485_248579


namespace NUMINAMATH_CALUDE_triangle_perimeter_l2485_248535

-- Define the lines
def line_through_origin (m : ℝ) := {(x, y) : ℝ × ℝ | y = m * x}
def vertical_line (a : ℝ) := {(x, y) : ℝ × ℝ | x = a}
def sloped_line (m : ℝ) (b : ℝ) := {(x, y) : ℝ × ℝ | y = m * x + b}

-- Define the triangle
def right_triangle (m : ℝ) := 
  (0, 0) ∈ line_through_origin m ∧
  (1, -m) ∈ line_through_origin m ∧
  (1, -m) ∈ vertical_line 1 ∧
  (1, 1.5) ∈ sloped_line (1/2) 1 ∧
  (1, 1.5) ∈ vertical_line 1

-- Theorem statement
theorem triangle_perimeter :
  ∀ m : ℝ, right_triangle m → 
  (Real.sqrt ((1:ℝ)^2 + m^2) + Real.sqrt ((1:ℝ)^2 + (1.5 + m)^2) + 0.5) = 3 + Real.sqrt 5 :=
sorry

end NUMINAMATH_CALUDE_triangle_perimeter_l2485_248535


namespace NUMINAMATH_CALUDE_vegetable_baskets_l2485_248545

/-- Calculates the number of baskets needed to store vegetables --/
theorem vegetable_baskets
  (keith_turnips : ℕ)
  (alyssa_turnips : ℕ)
  (sean_carrots : ℕ)
  (turnips_per_basket : ℕ)
  (carrots_per_basket : ℕ)
  (h1 : keith_turnips = 6)
  (h2 : alyssa_turnips = 9)
  (h3 : sean_carrots = 5)
  (h4 : turnips_per_basket = 5)
  (h5 : carrots_per_basket = 4) :
  (((keith_turnips + alyssa_turnips) + turnips_per_basket - 1) / turnips_per_basket) +
  ((sean_carrots + carrots_per_basket - 1) / carrots_per_basket) = 5 :=
by sorry

end NUMINAMATH_CALUDE_vegetable_baskets_l2485_248545


namespace NUMINAMATH_CALUDE_complex_number_in_second_quadrant_l2485_248550

def i : ℂ := Complex.I

theorem complex_number_in_second_quadrant :
  let z : ℂ := (1 + i) * i
  (z.re < 0) ∧ (z.im > 0) :=
by sorry

end NUMINAMATH_CALUDE_complex_number_in_second_quadrant_l2485_248550


namespace NUMINAMATH_CALUDE_quadratic_integer_root_existence_l2485_248536

theorem quadratic_integer_root_existence (a b c : ℕ) (h : a + b + c = 2000) :
  ∃ (a' b' c' : ℤ), 
    (∃ (x : ℤ), a' * x^2 + b' * x + c' = 0) ∧ 
    (|a - a'| + |b - b'| + |c - c'| : ℤ) ≤ 1050 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_integer_root_existence_l2485_248536


namespace NUMINAMATH_CALUDE_F_difference_l2485_248599

/-- Represents the infinite repeating decimal 0.726726726... -/
def F : ℚ := 726 / 999

/-- The fraction representation of F in lowest terms -/
def F_reduced : ℚ := 242 / 333

theorem F_difference : (F_reduced.den : ℤ) - (F_reduced.num : ℤ) = 91 := by sorry

end NUMINAMATH_CALUDE_F_difference_l2485_248599


namespace NUMINAMATH_CALUDE_a_range_l2485_248539

theorem a_range (a : ℝ) : 
  (∀ x : ℝ, |x - a| - |x| < 2 - a^2) → 
  a > -1 ∧ a < 1 := by
sorry

end NUMINAMATH_CALUDE_a_range_l2485_248539


namespace NUMINAMATH_CALUDE_sin_cos_sum_equals_negative_one_l2485_248575

theorem sin_cos_sum_equals_negative_one : 
  Real.sin ((11 / 6) * Real.pi) + Real.cos ((10 / 3) * Real.pi) = -1 := by
  sorry

end NUMINAMATH_CALUDE_sin_cos_sum_equals_negative_one_l2485_248575


namespace NUMINAMATH_CALUDE_expected_black_pairs_standard_deck_l2485_248525

/-- A standard deck of cards -/
structure Deck :=
  (total_cards : ℕ)
  (black_cards : ℕ)
  (red_cards : ℕ)
  (h_total : total_cards = black_cards + red_cards)

/-- A standard 104-card deck -/
def standard_deck : Deck :=
  { total_cards := 104,
    black_cards := 52,
    red_cards := 52,
    h_total := rfl }

/-- The expected number of pairs of adjacent black cards when dealt in a line -/
def expected_black_pairs (d : Deck) : ℚ :=
  (d.black_cards - 1 : ℚ) * (d.black_cards - 1) / (d.total_cards - 1)

theorem expected_black_pairs_standard_deck :
  expected_black_pairs standard_deck = 2601 / 103 :=
sorry

end NUMINAMATH_CALUDE_expected_black_pairs_standard_deck_l2485_248525


namespace NUMINAMATH_CALUDE_max_dot_product_l2485_248543

-- Define the ellipse
def is_on_ellipse (x y : ℝ) : Prop := x^2 / 4 + y^2 = 1

-- Define the center and left focus
def O : ℝ × ℝ := (0, 0)
def F : ℝ × ℝ := (-1, 0)

-- Define the dot product of OF and OP
def dot_product (x y : ℝ) : ℝ := (x + 1) * x + y * y

-- Theorem statement
theorem max_dot_product :
  ∀ x y : ℝ, is_on_ellipse x y →
  ∀ x' y' : ℝ, is_on_ellipse x' y' →
  dot_product x y ≤ 6 :=
sorry

end NUMINAMATH_CALUDE_max_dot_product_l2485_248543


namespace NUMINAMATH_CALUDE_sock_pairs_count_l2485_248562

def white_socks : ℕ := 5
def brown_socks : ℕ := 4
def blue_socks : ℕ := 3

def different_color_pairs_with_blue : ℕ := (blue_socks * white_socks) + (blue_socks * brown_socks)

theorem sock_pairs_count : different_color_pairs_with_blue = 27 := by
  sorry

end NUMINAMATH_CALUDE_sock_pairs_count_l2485_248562


namespace NUMINAMATH_CALUDE_clock_angle_at_13_20_clock_angle_at_13_20_is_80_l2485_248573

/-- The angle between the hour and minute hands of a clock at 13:20 (1:20 PM) --/
theorem clock_angle_at_13_20 : ℝ :=
  let hour := 1
  let minute := 20
  let degrees_per_hour := 360 / 12
  let degrees_per_minute := 360 / 60
  let hour_hand_angle := hour * degrees_per_hour + (minute / 60) * degrees_per_hour
  let minute_hand_angle := minute * degrees_per_minute
  |minute_hand_angle - hour_hand_angle|

/-- The angle between the hour and minute hands of a clock at 13:20 (1:20 PM) is 80 degrees --/
theorem clock_angle_at_13_20_is_80 : clock_angle_at_13_20 = 80 := by
  sorry

end NUMINAMATH_CALUDE_clock_angle_at_13_20_clock_angle_at_13_20_is_80_l2485_248573


namespace NUMINAMATH_CALUDE_expression_simplification_l2485_248547

theorem expression_simplification :
  (2^2 - 1) * (3^2 - 1) * (4^2 - 1) * (5^2 - 1) / 
  ((2 * 3) * (3 * 4) * (4 * 5) * (5 * 6)) = 1/5 := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l2485_248547


namespace NUMINAMATH_CALUDE_square_area_from_rectangle_l2485_248565

theorem square_area_from_rectangle (rectangle_area : ℝ) (rectangle_breadth : ℝ) : 
  rectangle_area = 100 →
  rectangle_breadth = 10 →
  ∃ (circle_radius : ℝ),
    (2 / 5 : ℝ) * circle_radius * rectangle_breadth = rectangle_area →
    circle_radius ^ 2 = 625 := by
  sorry

end NUMINAMATH_CALUDE_square_area_from_rectangle_l2485_248565


namespace NUMINAMATH_CALUDE_hulk_seventh_jump_exceeds_1km_l2485_248568

def hulk_jump (n : ℕ) : ℝ := 3 * (3 ^ (n - 1))

theorem hulk_seventh_jump_exceeds_1km :
  (∀ k < 7, hulk_jump k ≤ 1000) ∧ hulk_jump 7 > 1000 := by
  sorry

end NUMINAMATH_CALUDE_hulk_seventh_jump_exceeds_1km_l2485_248568


namespace NUMINAMATH_CALUDE_line_equation_through_point_parallel_to_vector_l2485_248578

/-- The equation of a line passing through point P(-1, 2) and parallel to vector {8, -4} --/
theorem line_equation_through_point_parallel_to_vector :
  let P : ℝ × ℝ := (-1, 2)
  let a : ℝ × ℝ := (8, -4)
  let line_eq (x y : ℝ) := y = -1/2 * x + 3/2
  (∀ x y : ℝ, line_eq x y ↔ 
    (∃ t : ℝ, x = P.1 + t * a.1 ∧ y = P.2 + t * a.2)) :=
by sorry

end NUMINAMATH_CALUDE_line_equation_through_point_parallel_to_vector_l2485_248578


namespace NUMINAMATH_CALUDE_smallest_divisor_after_391_l2485_248556

/-- Given an even 4-digit number m where 391 is a divisor,
    the smallest possible divisor of m greater than 391 is 441 -/
theorem smallest_divisor_after_391 (m : ℕ) (h1 : 1000 ≤ m) (h2 : m < 10000) 
    (h3 : Even m) (h4 : m % 391 = 0) : 
  ∃ (d : ℕ), d ∣ m ∧ d > 391 ∧ d ≥ 441 ∧ ∀ (x : ℕ), x ∣ m → x > 391 → x ≥ d :=
by sorry

end NUMINAMATH_CALUDE_smallest_divisor_after_391_l2485_248556


namespace NUMINAMATH_CALUDE_car_overtake_distance_l2485_248596

/-- Proves that the initial distance between two cars is equal to the product of their relative speed and the overtaking time. -/
theorem car_overtake_distance (v_red v_black : ℝ) (t : ℝ) (h1 : 0 < v_red) (h2 : v_red < v_black) (h3 : 0 < t) :
  (v_black - v_red) * t = (v_black - v_red) * t :=
by sorry

/-- Calculates the initial distance between two cars given their speeds and overtaking time. -/
def initial_distance (v_red v_black t : ℝ) : ℝ :=
  (v_black - v_red) * t

#check car_overtake_distance
#check initial_distance

end NUMINAMATH_CALUDE_car_overtake_distance_l2485_248596


namespace NUMINAMATH_CALUDE_albert_horses_l2485_248503

/-- Proves that Albert bought 4 horses given the conditions of the problem -/
theorem albert_horses : 
  ∀ (n : ℕ) (cow_price : ℕ),
  2000 * n + 9 * cow_price = 13400 →
  200 * n + 18 / 10 * cow_price = 1880 →
  n = 4 :=
by
  sorry

end NUMINAMATH_CALUDE_albert_horses_l2485_248503


namespace NUMINAMATH_CALUDE_cubic_function_property_l2485_248577

/-- Given a cubic function f(x) = ax³ + bx - 2 where f(2014) = 3, prove that f(-2014) = -7 -/
theorem cubic_function_property (a b : ℝ) : 
  let f : ℝ → ℝ := λ x => a * x^3 + b * x - 2
  (f 2014 = 3) → (f (-2014) = -7) := by
sorry

end NUMINAMATH_CALUDE_cubic_function_property_l2485_248577


namespace NUMINAMATH_CALUDE_stating_first_alloy_amount_l2485_248564

/-- Represents an alloy with a specific ratio of lead to tin -/
structure Alloy where
  lead : ℝ
  tin : ℝ

/-- The first available alloy -/
def alloy1 : Alloy := { lead := 1, tin := 2 }

/-- The second available alloy -/
def alloy2 : Alloy := { lead := 2, tin := 3 }

/-- The desired new alloy -/
def newAlloy : Alloy := { lead := 4, tin := 7 }

/-- The total mass of the new alloy -/
def totalMass : ℝ := 22

/-- 
Theorem stating that 12 grams of the first alloy is needed to create the new alloy
with the desired properties
-/
theorem first_alloy_amount : 
  ∃ (x y : ℝ),
    x * (alloy1.lead + alloy1.tin) + y * (alloy2.lead + alloy2.tin) = totalMass ∧
    (x * alloy1.lead + y * alloy2.lead) / (x * alloy1.tin + y * alloy2.tin) = newAlloy.lead / newAlloy.tin ∧
    x * (alloy1.lead + alloy1.tin) = 12 := by
  sorry


end NUMINAMATH_CALUDE_stating_first_alloy_amount_l2485_248564


namespace NUMINAMATH_CALUDE_negation_of_existence_statement_l2485_248510

theorem negation_of_existence_statement :
  ¬(∃ x : ℝ, x ≤ 0) ↔ (∀ x : ℝ, x > 0) := by sorry

end NUMINAMATH_CALUDE_negation_of_existence_statement_l2485_248510


namespace NUMINAMATH_CALUDE_basketball_competition_equation_l2485_248571

/-- Represents the number of matches in a basketball competition where each pair of classes plays once --/
def number_of_matches (x : ℕ) : ℕ := x * (x - 1) / 2

/-- Theorem stating that for 10 total matches, the equation x(x-1)/2 = 10 correctly represents the situation --/
theorem basketball_competition_equation (x : ℕ) (h : number_of_matches x = 10) : 
  x * (x - 1) / 2 = 10 := by
  sorry

end NUMINAMATH_CALUDE_basketball_competition_equation_l2485_248571


namespace NUMINAMATH_CALUDE_parallel_vectors_imply_k_value_l2485_248548

/-- Given vectors a, b, and c in R², prove that if (a - c) is parallel to b, then k = 5 -/
theorem parallel_vectors_imply_k_value (a b c : ℝ × ℝ) (k : ℝ) :
  a = (3, 1) →
  b = (1, 3) →
  c = (k, 7) →
  (∃ (t : ℝ), a - c = t • b) →
  k = 5 := by
  sorry

end NUMINAMATH_CALUDE_parallel_vectors_imply_k_value_l2485_248548


namespace NUMINAMATH_CALUDE_ace_king_queen_probability_l2485_248581

-- Define the total number of cards in a standard deck
def totalCards : ℕ := 52

-- Define the number of each face card (Ace, King, Queen)
def faceCards : ℕ := 4

-- Define the probability of drawing the sequence (Ace, King, Queen)
def probAceKingQueen : ℚ := (faceCards : ℚ) / totalCards *
                            (faceCards : ℚ) / (totalCards - 1) *
                            (faceCards : ℚ) / (totalCards - 2)

-- Theorem statement
theorem ace_king_queen_probability :
  probAceKingQueen = 8 / 16575 := by sorry

end NUMINAMATH_CALUDE_ace_king_queen_probability_l2485_248581


namespace NUMINAMATH_CALUDE_coins_fit_in_new_box_l2485_248557

/-- Represents a rectangular box -/
structure Box where
  width : ℝ
  height : ℝ

/-- Represents a collection of coins -/
structure CoinCollection where
  maxDiameter : ℝ

/-- Check if a coin collection can fit in a box -/
def canFitIn (coins : CoinCollection) (box : Box) : Prop :=
  box.width * box.height ≥ 0 -- This is a simplification, as we don't know the exact arrangement

/-- Theorem: If coins fit in the original box, they can fit in the new box -/
theorem coins_fit_in_new_box 
  (coins : CoinCollection)
  (originalBox : Box)
  (newBox : Box)
  (h1 : coins.maxDiameter ≤ 10)
  (h2 : originalBox.width = 30 ∧ originalBox.height = 70)
  (h3 : newBox.width = 40 ∧ newBox.height = 60)
  (h4 : canFitIn coins originalBox) :
  canFitIn coins newBox :=
by
  sorry

#check coins_fit_in_new_box

end NUMINAMATH_CALUDE_coins_fit_in_new_box_l2485_248557


namespace NUMINAMATH_CALUDE_f_passes_through_point_two_zero_l2485_248549

-- Define the function f
def f (x : ℝ) : ℝ := x - 2

-- Theorem statement
theorem f_passes_through_point_two_zero : f 2 = 0 := by
  sorry

end NUMINAMATH_CALUDE_f_passes_through_point_two_zero_l2485_248549


namespace NUMINAMATH_CALUDE_subset_implies_m_equals_three_l2485_248563

theorem subset_implies_m_equals_three (A B : Set ℕ) (m : ℕ) :
  A = {1, 3} →
  B = {1, 2, m} →
  A ⊆ B →
  m = 3 := by
sorry

end NUMINAMATH_CALUDE_subset_implies_m_equals_three_l2485_248563


namespace NUMINAMATH_CALUDE_factorization_of_2m_squared_minus_8_l2485_248558

theorem factorization_of_2m_squared_minus_8 (m : ℝ) : 2 * m^2 - 8 = 2 * (m + 2) * (m - 2) := by
  sorry

end NUMINAMATH_CALUDE_factorization_of_2m_squared_minus_8_l2485_248558


namespace NUMINAMATH_CALUDE_smallest_sum_of_perfect_squares_l2485_248582

theorem smallest_sum_of_perfect_squares (x y : ℕ) : 
  x^2 - y^2 = 145 → (∀ a b : ℕ, a^2 - b^2 = 145 → x^2 + y^2 ≤ a^2 + b^2) → x^2 + y^2 = 433 := by
  sorry

end NUMINAMATH_CALUDE_smallest_sum_of_perfect_squares_l2485_248582


namespace NUMINAMATH_CALUDE_function_not_satisfying_differential_equation_l2485_248519

open Real

theorem function_not_satisfying_differential_equation :
  ¬∃ y : ℝ → ℝ, ∀ x : ℝ,
    (y x = (x + 1) * (Real.exp (x^2))) ∧
    (deriv y x - 2 * x * y x = 2 * x * (Real.exp (x^2))) :=
sorry

end NUMINAMATH_CALUDE_function_not_satisfying_differential_equation_l2485_248519


namespace NUMINAMATH_CALUDE_units_digit_product_l2485_248593

theorem units_digit_product : (5^2 + 1) * (5^3 + 1) * (5^23 + 1) % 10 = 6 := by
  sorry

end NUMINAMATH_CALUDE_units_digit_product_l2485_248593


namespace NUMINAMATH_CALUDE_circle_center_sum_l2485_248576

/-- The equation of a circle in the xy-plane -/
def CircleEquation (x y : ℝ) : Prop :=
  x^2 + y^2 = 10*x + 4*y + 9

/-- The center of a circle given its equation -/
def CircleCenter (eq : (ℝ → ℝ → Prop)) : ℝ × ℝ :=
  sorry

/-- The sum of coordinates of a point -/
def SumOfCoordinates (p : ℝ × ℝ) : ℝ :=
  p.1 + p.2

theorem circle_center_sum :
  SumOfCoordinates (CircleCenter CircleEquation) = 7 := by
  sorry

end NUMINAMATH_CALUDE_circle_center_sum_l2485_248576


namespace NUMINAMATH_CALUDE_fraction_value_l2485_248595

theorem fraction_value : (3100 - 3037)^2 / 81 = 49 := by
  sorry

end NUMINAMATH_CALUDE_fraction_value_l2485_248595


namespace NUMINAMATH_CALUDE_intersection_distance_sum_l2485_248530

-- Define the curve C
def curve_C (x y : ℝ) : Prop := y^2 = 4*x

-- Define the line
def line (x y : ℝ) : Prop := x - y - 2 = 0

-- Define point P
def point_P : ℝ × ℝ := (-2, -4)

-- Define the theorem
theorem intersection_distance_sum :
  ∃ (M N : ℝ × ℝ),
    curve_C M.1 M.2 ∧
    curve_C N.1 N.2 ∧
    line M.1 M.2 ∧
    line N.1 N.2 ∧
    M ≠ N ∧
    Real.sqrt ((M.1 - point_P.1)^2 + (M.2 - point_P.2)^2) +
    Real.sqrt ((N.1 - point_P.1)^2 + (N.2 - point_P.2)^2) =
    12 * Real.sqrt 2 :=
sorry

end NUMINAMATH_CALUDE_intersection_distance_sum_l2485_248530


namespace NUMINAMATH_CALUDE_josanna_minimum_score_l2485_248511

def current_scores : List ℕ := [75, 85, 65, 95, 70]
def increase_amount : ℕ := 10

def minimum_next_score (scores : List ℕ) (increase : ℕ) : ℕ :=
  let current_sum := scores.sum
  let current_count := scores.length
  let current_avg := current_sum / current_count
  let target_avg := current_avg + increase
  let total_count := current_count + 1
  target_avg * total_count - current_sum

theorem josanna_minimum_score :
  minimum_next_score current_scores increase_amount = 138 := by
  sorry

end NUMINAMATH_CALUDE_josanna_minimum_score_l2485_248511


namespace NUMINAMATH_CALUDE_jons_payment_per_visit_l2485_248559

/-- Represents the payment structure for Jon's website -/
structure WebsitePayment where
  visits_per_hour : ℕ
  hours_per_day : ℕ
  days_per_month : ℕ
  monthly_revenue : ℚ

/-- Calculates the payment per visit given the website payment structure -/
def payment_per_visit (wp : WebsitePayment) : ℚ :=
  wp.monthly_revenue / (wp.visits_per_hour * wp.hours_per_day * wp.days_per_month)

/-- Theorem stating that Jon's payment per visit is $0.10 -/
theorem jons_payment_per_visit :
  let wp : WebsitePayment := {
    visits_per_hour := 50,
    hours_per_day := 24,
    days_per_month := 30,
    monthly_revenue := 3600
  }
  payment_per_visit wp = 1/10 := by
  sorry

end NUMINAMATH_CALUDE_jons_payment_per_visit_l2485_248559


namespace NUMINAMATH_CALUDE_max_piece_length_l2485_248594

def rope_lengths : List Nat := [48, 72, 120, 144]

def min_pieces : Nat := 5

def is_valid_piece_length (len : Nat) : Bool :=
  rope_lengths.all (fun rope => rope % len = 0 ∧ rope / len ≥ min_pieces)

theorem max_piece_length :
  ∃ (max_len : Nat), max_len = 8 ∧
    is_valid_piece_length max_len ∧
    ∀ (len : Nat), len > max_len → ¬is_valid_piece_length len :=
by sorry

end NUMINAMATH_CALUDE_max_piece_length_l2485_248594


namespace NUMINAMATH_CALUDE_difference_le_two_l2485_248507

/-- Represents a right-angled triangle with integer sides -/
structure RightTriangle where
  a : ℕ
  b : ℕ
  c : ℕ
  h_right_angle : a ^ 2 + b ^ 2 = c ^ 2
  h_ordered : a < b ∧ b < c
  h_coprime : Nat.Coprime a b ∧ Nat.Coprime b c ∧ Nat.Coprime a c

/-- The difference between the hypotenuse and the middle side -/
def difference (t : RightTriangle) : ℕ := t.c - t.b

/-- Theorem: For a right-angled triangle with integer sides a, b, c where
    a < b < c, a, b, c are pairwise co-prime, and (c - b) divides a,
    then (c - b) ≤ 2 -/
theorem difference_le_two (t : RightTriangle) (h_divides : t.a % (difference t) = 0) :
  difference t ≤ 2 := by
  sorry

end NUMINAMATH_CALUDE_difference_le_two_l2485_248507


namespace NUMINAMATH_CALUDE_sum_of_digits_problem_l2485_248542

def S (n : ℕ) : ℕ := sorry  -- Definition of S(n) as sum of digits

theorem sum_of_digits_problem (n : ℕ) (h : n + S n = 2009) : n = 1990 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_digits_problem_l2485_248542


namespace NUMINAMATH_CALUDE_eddy_spider_plant_production_l2485_248509

/-- A spider plant that produces baby plants -/
structure SpiderPlant where
  /-- Number of baby plants produced each time -/
  baby_per_time : ℕ
  /-- Total number of baby plants produced -/
  total_babies : ℕ
  /-- Number of years -/
  years : ℕ

/-- The number of times per year a spider plant produces baby plants -/
def times_per_year (plant : SpiderPlant) : ℚ :=
  (plant.total_babies : ℚ) / (plant.years * plant.baby_per_time : ℚ)

/-- Theorem stating that Eddy's spider plant produces baby plants 2 times per year -/
theorem eddy_spider_plant_production :
  ∃ (plant : SpiderPlant),
    plant.baby_per_time = 2 ∧
    plant.total_babies = 16 ∧
    plant.years = 4 ∧
    times_per_year plant = 2 := by
  sorry

end NUMINAMATH_CALUDE_eddy_spider_plant_production_l2485_248509


namespace NUMINAMATH_CALUDE_g_of_8_l2485_248506

theorem g_of_8 (g : ℝ → ℝ) (h : ∀ x, x ≠ 2 → g x = (7 * x + 3) / (x - 2)) :
  g 8 = 59 / 6 := by
  sorry

end NUMINAMATH_CALUDE_g_of_8_l2485_248506


namespace NUMINAMATH_CALUDE_imaginary_part_of_z_l2485_248504

theorem imaginary_part_of_z (z : ℂ) : z = (2 * Complex.I) / (1 + Complex.I) → Complex.im z = 1 := by
  sorry

end NUMINAMATH_CALUDE_imaginary_part_of_z_l2485_248504


namespace NUMINAMATH_CALUDE_coin_value_difference_max_value_achievable_min_value_achievable_l2485_248546

/-- Represents the types of coins --/
inductive CoinType
  | Penny
  | Nickel
  | Dime

/-- The value of a coin in cents --/
def coinValue : CoinType → Nat
  | CoinType.Penny => 1
  | CoinType.Nickel => 5
  | CoinType.Dime => 10

/-- A distribution of coins --/
structure CoinDistribution where
  pennies : Nat
  nickels : Nat
  dimes : Nat
  total_coins : pennies + nickels + dimes = 3030
  at_least_one : pennies ≥ 1 ∧ nickels ≥ 1 ∧ dimes ≥ 1

/-- The total value of a coin distribution in cents --/
def totalValue (d : CoinDistribution) : Nat :=
  d.pennies * coinValue CoinType.Penny +
  d.nickels * coinValue CoinType.Nickel +
  d.dimes * coinValue CoinType.Dime

/-- The maximum possible value for any valid coin distribution --/
def maxValue : Nat := 30286

/-- The minimum possible value for any valid coin distribution --/
def minValue : Nat := 3043

theorem coin_value_difference :
  maxValue - minValue = 27243 :=
by
  sorry

theorem max_value_achievable (d : CoinDistribution) :
  totalValue d ≤ maxValue :=
by
  sorry

theorem min_value_achievable (d : CoinDistribution) :
  totalValue d ≥ minValue :=
by
  sorry

end NUMINAMATH_CALUDE_coin_value_difference_max_value_achievable_min_value_achievable_l2485_248546


namespace NUMINAMATH_CALUDE_base4_calculation_l2485_248591

/-- Converts a number from base 4 to base 10 -/
def base4ToBase10 (n : ℕ) : ℕ := sorry

/-- Converts a number from base 10 to base 4 -/
def base10ToBase4 (n : ℕ) : ℕ := sorry

/-- Performs division in base 4 -/
def divBase4 (a b : ℕ) : ℕ := sorry

/-- Performs multiplication in base 4 -/
def mulBase4 (a b : ℕ) : ℕ := sorry

theorem base4_calculation :
  mulBase4 (divBase4 130 3) 14 = 1200 := by sorry

end NUMINAMATH_CALUDE_base4_calculation_l2485_248591


namespace NUMINAMATH_CALUDE_triangle_area_formulas_l2485_248555

/-- Given a triangle with area t, semiperimeter s, angles α, β, γ, and sides a, b, c,
    prove two formulas for the area. -/
theorem triangle_area_formulas (t s a b c α β γ : ℝ) 
  (h_area : t > 0)
  (h_semiperimeter : s > 0)
  (h_sides : a > 0 ∧ b > 0 ∧ c > 0)
  (h_angles : α > 0 ∧ β > 0 ∧ γ > 0)
  (h_sum_angles : α + β + γ = π)
  (h_semiperimeter_def : s = (a + b + c) / 2) :
  (t = s^2 * Real.tan (α/2) * Real.tan (β/2) * Real.tan (γ/2)) ∧
  (t = (a*b*c/s) * Real.cos (α/2) * Real.cos (β/2) * Real.cos (γ/2)) := by
  sorry


end NUMINAMATH_CALUDE_triangle_area_formulas_l2485_248555


namespace NUMINAMATH_CALUDE_min_gumballs_for_four_same_color_gumball_problem_solution_l2485_248505

/-- Represents the number of gumballs of each color -/
structure GumballCounts where
  green : Nat
  red : Nat
  white : Nat
  blue : Nat

/-- The minimum number of gumballs needed to ensure getting four of the same color -/
def minGumballsForFourSameColor (counts : GumballCounts) : Nat :=
  13

/-- Theorem stating that for the given gumball counts, 
    the minimum number of gumballs needed to ensure 
    getting four of the same color is 13 -/
theorem min_gumballs_for_four_same_color 
  (counts : GumballCounts) 
  (h1 : counts.green = 12) 
  (h2 : counts.red = 10) 
  (h3 : counts.white = 9) 
  (h4 : counts.blue = 11) : 
  minGumballsForFourSameColor counts = 13 := by
  sorry

/-- Main theorem that proves the result for the specific problem instance -/
theorem gumball_problem_solution : 
  ∃ (counts : GumballCounts), 
    counts.green = 12 ∧ 
    counts.red = 10 ∧ 
    counts.white = 9 ∧ 
    counts.blue = 11 ∧ 
    minGumballsForFourSameColor counts = 13 := by
  sorry

end NUMINAMATH_CALUDE_min_gumballs_for_four_same_color_gumball_problem_solution_l2485_248505


namespace NUMINAMATH_CALUDE_vet_donation_portion_is_about_third_l2485_248541

/-- Represents the adoption event scenario at an animal shelter --/
structure AdoptionEvent where
  dog_fee : ℕ
  cat_fee : ℕ
  dog_adoptions : ℕ
  cat_adoptions : ℕ
  vet_donation : ℕ

/-- Calculates the portion of fees donated by the vet --/
def donation_portion (event : AdoptionEvent) : ℚ :=
  let total_fees := event.dog_fee * event.dog_adoptions + event.cat_fee * event.cat_adoptions
  (event.vet_donation : ℚ) / total_fees

/-- Theorem stating that the portion of fees donated is approximately 33.33% --/
theorem vet_donation_portion_is_about_third (event : AdoptionEvent) 
    (h1 : event.dog_fee = 15)
    (h2 : event.cat_fee = 13)
    (h3 : event.dog_adoptions = 8)
    (h4 : event.cat_adoptions = 3)
    (h5 : event.vet_donation = 53) :
    ∃ (ε : ℚ), ε > 0 ∧ ε < 1/100 ∧ |donation_portion event - 1/3| < ε := by
  sorry

#eval donation_portion { dog_fee := 15, cat_fee := 13, dog_adoptions := 8, cat_adoptions := 3, vet_donation := 53 }

end NUMINAMATH_CALUDE_vet_donation_portion_is_about_third_l2485_248541


namespace NUMINAMATH_CALUDE_tangent_circle_equation_l2485_248527

/-- A circle passing through two points and tangent to a line -/
structure TangentCircle where
  center : ℝ × ℝ
  radius : ℝ
  passes_origin : center.1^2 + center.2^2 = radius^2
  passes_point : (center.1 - 4)^2 + center.2^2 = radius^2
  tangent_to_line : |center.2 - 1| = radius

/-- The equation of the circle -/
def circle_equation (c : TangentCircle) (x y : ℝ) : Prop :=
  (x - c.center.1)^2 + (y - c.center.2)^2 = c.radius^2

theorem tangent_circle_equation :
  ∀ (c : TangentCircle),
    ∀ (x y : ℝ),
      circle_equation c x y ↔ (x - 2)^2 + (y + 3/2)^2 = 25/4 :=
by sorry

end NUMINAMATH_CALUDE_tangent_circle_equation_l2485_248527


namespace NUMINAMATH_CALUDE_multiplication_solution_l2485_248514

def possible_digits : Set Nat := {2, 4, 5, 6, 7, 8, 9}

def valid_multiplication (A B C D E : Nat) : Prop :=
  A ∈ possible_digits ∧ B ∈ possible_digits ∧ C ∈ possible_digits ∧ 
  D ∈ possible_digits ∧ E ∈ possible_digits ∧
  A ≠ B ∧ A ≠ C ∧ A ≠ D ∧ A ≠ E ∧ B ≠ C ∧ B ≠ D ∧ B ≠ E ∧ C ≠ D ∧ C ≠ E ∧ D ≠ E ∧
  E = 7 ∧
  (3 * (100000 * A + 10000 * B + 1000 * C + 100 * D + 10 * E + 1) = 
   100000 * A + 10000 * B + 1000 * C + 100 * D + 10 * E + 1)

theorem multiplication_solution : 
  ∃ (A B C D E : Nat), valid_multiplication A B C D E ∧ A = 4 ∧ B = 2 ∧ C = 8 ∧ D = 5 := by
  sorry

end NUMINAMATH_CALUDE_multiplication_solution_l2485_248514


namespace NUMINAMATH_CALUDE_distance_to_directrix_l2485_248524

/-- A parabola C is defined by the equation y² = 2px. -/
structure Parabola where
  p : ℝ

/-- A point on a 2D plane. -/
structure Point where
  x : ℝ
  y : ℝ

/-- The theorem states that for a parabola C where point A(1, √5) lies on it,
    the distance from A to the directrix of C is 9/4. -/
theorem distance_to_directrix (C : Parabola) (A : Point) :
  A.x = 1 →
  A.y = Real.sqrt 5 →
  A.y ^ 2 = 2 * C.p * A.x →
  (A.x + C.p / 2) = 9 / 4 := by
  sorry

end NUMINAMATH_CALUDE_distance_to_directrix_l2485_248524


namespace NUMINAMATH_CALUDE_min_value_of_f_l2485_248587

/-- Given positive real numbers a, b, c, x, y, z satisfying certain conditions,
    the function f(x, y, z) has a minimum value of 1/2. -/
theorem min_value_of_f (a b c x y z : ℝ) 
    (pos_a : 0 < a) (pos_b : 0 < b) (pos_c : 0 < c)
    (pos_x : 0 < x) (pos_y : 0 < y) (pos_z : 0 < z)
    (eq1 : c * y + b * z = a)
    (eq2 : a * z + c * x = b)
    (eq3 : b * x + a * y = c) :
    let f := fun (x y z : ℝ) => x^2 / (1 + x) + y^2 / (1 + y) + z^2 / (1 + z)
    ∀ x' y' z' : ℝ, 0 < x' → 0 < y' → 0 < z' → f x' y' z' ≥ 1/2 ∧ 
    ∃ x₀ y₀ z₀ : ℝ, 0 < x₀ ∧ 0 < y₀ ∧ 0 < z₀ ∧ f x₀ y₀ z₀ = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_min_value_of_f_l2485_248587


namespace NUMINAMATH_CALUDE_hyperbola_eccentricity_k_range_l2485_248502

theorem hyperbola_eccentricity_k_range :
  ∀ (k : ℝ) (e : ℝ),
    (∀ (x y : ℝ), x^2 / 4 + y^2 / k = 1) →
    (1 < e ∧ e < 3) →
    (e = Real.sqrt (1 - k / 4)) →
    (-32 < k ∧ k < 0) :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_eccentricity_k_range_l2485_248502


namespace NUMINAMATH_CALUDE_ceiling_negative_three_point_eight_l2485_248569

theorem ceiling_negative_three_point_eight :
  ⌈(-3.8 : ℝ)⌉ = -3 := by sorry

end NUMINAMATH_CALUDE_ceiling_negative_three_point_eight_l2485_248569


namespace NUMINAMATH_CALUDE_retest_probability_l2485_248534

theorem retest_probability (total : ℕ) (p_physics : ℚ) (p_chemistry : ℚ) (p_biology : ℚ) :
  total = 50 →
  p_physics = 9 / 50 →
  p_chemistry = 1 / 5 →
  p_biology = 11 / 50 →
  let p_one_subject := p_physics + p_chemistry + p_biology
  let p_more_than_one := 1 - p_one_subject
  p_more_than_one = 2 / 5 := by
  sorry

#eval (2 : ℚ) / 5 -- This should output 0.4

end NUMINAMATH_CALUDE_retest_probability_l2485_248534


namespace NUMINAMATH_CALUDE_parabola_coefficients_from_vertex_and_point_l2485_248516

/-- Represents a parabola of the form y = ax^2 + bx + c -/
structure Parabola where
  a : ℝ
  b : ℝ
  c : ℝ

/-- The y-coordinate for a given x on the parabola -/
def Parabola.y (p : Parabola) (x : ℝ) : ℝ :=
  p.a * x^2 + p.b * x + p.c

theorem parabola_coefficients_from_vertex_and_point
  (p : Parabola)
  (vertex_x vertex_y : ℝ)
  (point_x point_y : ℝ)
  (h_vertex : p.y vertex_x = vertex_y)
  (h_point : p.y point_x = point_y)
  (h_vertex_x : vertex_x = 4)
  (h_vertex_y : vertex_y = 3)
  (h_point_x : point_x = 2)
  (h_point_y : point_y = 1) :
  p.a = -1/2 ∧ p.b = 4 ∧ p.c = -5 := by
  sorry

end NUMINAMATH_CALUDE_parabola_coefficients_from_vertex_and_point_l2485_248516


namespace NUMINAMATH_CALUDE_minimum_teacher_time_l2485_248554

def student_time (explanation_time : ℕ) (completion_time : ℕ) : ℕ :=
  explanation_time + completion_time

theorem minimum_teacher_time 
  (student_A : ℕ) 
  (student_B : ℕ) 
  (student_C : ℕ) 
  (explanation_time : ℕ) 
  (h1 : student_A = student_time explanation_time 13)
  (h2 : student_B = student_time explanation_time 10)
  (h3 : student_C = student_time explanation_time 16)
  (h4 : explanation_time = 3) :
  3 * explanation_time + 2 * student_B + student_A + student_C = 90 :=
sorry

end NUMINAMATH_CALUDE_minimum_teacher_time_l2485_248554


namespace NUMINAMATH_CALUDE_hundredth_term_is_14_l2485_248590

def S (n : ℕ) : ℕ := n * (n + 1) / 2

def is_nth_term (x n : ℕ) : Prop :=
  ∃ k, k ≤ n ∧ S (x - 1) < k ∧ k ≤ S x

theorem hundredth_term_is_14 : is_nth_term 14 100 := by sorry

end NUMINAMATH_CALUDE_hundredth_term_is_14_l2485_248590


namespace NUMINAMATH_CALUDE_wrapping_paper_fraction_l2485_248570

theorem wrapping_paper_fraction (total_fraction : ℚ) (num_small : ℕ) (num_large : ℕ) :
  total_fraction = 3/8 →
  num_small = 4 →
  num_large = 2 →
  (∃ small_fraction : ℚ, 
    total_fraction = num_small * small_fraction + num_large * (2 * small_fraction) ∧
    small_fraction = 3/64) :=
by sorry

end NUMINAMATH_CALUDE_wrapping_paper_fraction_l2485_248570


namespace NUMINAMATH_CALUDE_quadratic_polynomial_inequality_l2485_248589

def quadratic_polynomial (a b c : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x + c

theorem quadratic_polynomial_inequality 
  (a b c : ℝ) 
  (ha : a ≥ 0) (hb : b ≥ 0) (hc : c ≥ 0) 
  (x y : ℝ) : 
  (quadratic_polynomial a b c (x * y))^2 ≤ 
  (quadratic_polynomial a b c (x^2)) * (quadratic_polynomial a b c (y^2)) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_polynomial_inequality_l2485_248589


namespace NUMINAMATH_CALUDE_diamond_three_four_l2485_248537

-- Define the diamond operation
def diamond (x y : ℝ) : ℝ := 4 * x + 6 * y

-- Theorem statement
theorem diamond_three_four : diamond 3 4 = 36 := by
  sorry

end NUMINAMATH_CALUDE_diamond_three_four_l2485_248537


namespace NUMINAMATH_CALUDE_rectangular_garden_perimeter_l2485_248561

theorem rectangular_garden_perimeter 
  (x y : ℝ) 
  (diagonal_squared : x^2 + y^2 = 900)
  (area : x * y = 240) : 
  2 * (x + y) = 4 * Real.sqrt 345 := by
sorry

end NUMINAMATH_CALUDE_rectangular_garden_perimeter_l2485_248561


namespace NUMINAMATH_CALUDE_segment_length_l2485_248521

/-- Given two points P and Q on a line segment AB, where:
    - P and Q are on the same side of the midpoint of AB
    - P divides AB in the ratio 3:5
    - Q divides AB in the ratio 4:5
    - PQ = 3
    Prove that the length of AB is 43.2 -/
theorem segment_length (A B P Q : Real) (h1 : P ∈ Set.Icc A B) (h2 : Q ∈ Set.Icc A B)
    (h3 : (P - A) / (B - A) = 3 / 8) (h4 : (Q - A) / (B - A) = 4 / 9) (h5 : Q - P = 3) :
    B - A = 43.2 := by
  sorry

end NUMINAMATH_CALUDE_segment_length_l2485_248521


namespace NUMINAMATH_CALUDE_annual_increase_rate_proof_l2485_248508

/-- Proves that given an initial value of 32000 and a value of 40500 after two years,
    the annual increase rate is 0.125. -/
theorem annual_increase_rate_proof (initial_value final_value : ℝ) 
  (h1 : initial_value = 32000)
  (h2 : final_value = 40500)
  (h3 : final_value = initial_value * (1 + 0.125)^2) : 
  ∃ (r : ℝ), r = 0.125 ∧ final_value = initial_value * (1 + r)^2 := by
  sorry

end NUMINAMATH_CALUDE_annual_increase_rate_proof_l2485_248508


namespace NUMINAMATH_CALUDE_dice_probability_l2485_248512

/-- The number of sides on each die -/
def numSides : ℕ := 8

/-- The number of dice rolled -/
def numDice : ℕ := 8

/-- The number of sides showing numbers less than or equal to 4 -/
def favorableOutcomes : ℕ := 4

/-- The number of dice required to show numbers less than or equal to 4 -/
def requiredSuccesses : ℕ := 4

/-- The probability of rolling a number less than or equal to 4 on a single die -/
def singleDieProbability : ℚ := favorableOutcomes / numSides

theorem dice_probability :
  Nat.choose numDice requiredSuccesses *
  singleDieProbability ^ requiredSuccesses *
  (1 - singleDieProbability) ^ (numDice - requiredSuccesses) =
  35 / 128 :=
sorry

end NUMINAMATH_CALUDE_dice_probability_l2485_248512
