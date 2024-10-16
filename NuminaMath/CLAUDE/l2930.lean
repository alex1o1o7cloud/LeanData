import Mathlib

namespace NUMINAMATH_CALUDE_even_sum_probability_l2930_293083

/-- Represents a 4x4 grid filled with numbers from 1 to 16 -/
def Grid := Fin 4 → Fin 4 → Fin 16

/-- Checks if a list of numbers has an even sum -/
def hasEvenSum (l : List (Fin 16)) : Prop :=
  (l.map (fun x => x.val + 1)).sum % 2 = 0

/-- Checks if all rows and columns in a grid have even sums -/
def allRowsAndColumnsEven (g : Grid) : Prop :=
  (∀ i : Fin 4, hasEvenSum [g i 0, g i 1, g i 2, g i 3]) ∧
  (∀ j : Fin 4, hasEvenSum [g 0 j, g 1 j, g 2 j, g 3 j])

/-- The total number of ways to arrange 16 numbers in a 4x4 grid -/
def totalArrangements : ℕ := 20922789888000

/-- The number of valid arrangements with even sums in all rows and columns -/
def validArrangements : ℕ := 36

theorem even_sum_probability :
  (validArrangements : ℚ) / totalArrangements =
  (36 : ℚ) / 20922789888000 :=
sorry

end NUMINAMATH_CALUDE_even_sum_probability_l2930_293083


namespace NUMINAMATH_CALUDE_x_power_ten_equals_one_l2930_293022

theorem x_power_ten_equals_one (x : ℂ) (h : x + 1/x = Real.sqrt 5) : x^10 = 1 := by
  sorry

end NUMINAMATH_CALUDE_x_power_ten_equals_one_l2930_293022


namespace NUMINAMATH_CALUDE_square_root_of_one_sixty_fourth_l2930_293047

theorem square_root_of_one_sixty_fourth : Real.sqrt (1 / 64) = 1 / 8 := by
  sorry

end NUMINAMATH_CALUDE_square_root_of_one_sixty_fourth_l2930_293047


namespace NUMINAMATH_CALUDE_count_integers_satisfying_inequality_l2930_293086

theorem count_integers_satisfying_inequality : 
  ∃! (S : Finset ℤ), (∀ n ∈ S, (Real.sqrt (n + 1) ≤ Real.sqrt (3 * n + 2) ∧ 
    Real.sqrt (3 * n + 2) < Real.sqrt (2 * n + 7))) ∧ 
    S.card = 5 :=
by sorry

end NUMINAMATH_CALUDE_count_integers_satisfying_inequality_l2930_293086


namespace NUMINAMATH_CALUDE_fair_apple_distribution_l2930_293014

/-- Represents the work done by each girl -/
structure WorkDistribution :=
  (anna : ℕ)
  (varya : ℕ)
  (sveta : ℕ)

/-- Represents the fair distribution of apples -/
structure AppleDistribution :=
  (anna : ℚ)
  (varya : ℚ)
  (sveta : ℚ)

/-- Calculates the fair distribution of apples based on work done -/
def calculateAppleDistribution (work : WorkDistribution) (totalApples : ℕ) : AppleDistribution :=
  let totalWork := work.anna + work.varya + work.sveta
  { anna := (work.anna : ℚ) / totalWork * totalApples,
    varya := (work.varya : ℚ) / totalWork * totalApples,
    sveta := (work.sveta : ℚ) / totalWork * totalApples }

/-- The main theorem to prove -/
theorem fair_apple_distribution 
  (work : WorkDistribution) 
  (h_work : work = ⟨20, 35, 45⟩) 
  (totalApples : ℕ) 
  (h_apples : totalApples = 10) :
  calculateAppleDistribution work totalApples = ⟨2, (7:ℚ)/2, (9:ℚ)/2⟩ := by
  sorry

#check fair_apple_distribution

end NUMINAMATH_CALUDE_fair_apple_distribution_l2930_293014


namespace NUMINAMATH_CALUDE_potato_bundle_price_l2930_293030

/-- Calculates the price of potato bundles given the harvest and sales information --/
theorem potato_bundle_price
  (potato_count : ℕ)
  (potato_bundle_size : ℕ)
  (carrot_count : ℕ)
  (carrot_bundle_size : ℕ)
  (carrot_bundle_price : ℚ)
  (total_revenue : ℚ)
  (h1 : potato_count = 250)
  (h2 : potato_bundle_size = 25)
  (h3 : carrot_count = 320)
  (h4 : carrot_bundle_size = 20)
  (h5 : carrot_bundle_price = 2)
  (h6 : total_revenue = 51) :
  (total_revenue - (carrot_count / carrot_bundle_size * carrot_bundle_price)) / (potato_count / potato_bundle_size) = 1.9 := by
sorry

end NUMINAMATH_CALUDE_potato_bundle_price_l2930_293030


namespace NUMINAMATH_CALUDE_a_less_than_11_necessary_not_sufficient_l2930_293071

theorem a_less_than_11_necessary_not_sufficient :
  (∀ a : ℝ, (∃ x : ℝ, x^2 - 2*x + a < 0) → a < 11) ∧
  (∃ a : ℝ, a < 11 ∧ ¬(∃ x : ℝ, x^2 - 2*x + a < 0)) :=
sorry

end NUMINAMATH_CALUDE_a_less_than_11_necessary_not_sufficient_l2930_293071


namespace NUMINAMATH_CALUDE_min_value_theorem_l2930_293031

theorem min_value_theorem (a : ℝ) (h : a > 3) :
  a + 4 / (a - 3) ≥ 7 ∧ (a + 4 / (a - 3) = 7 ↔ a = 5) := by sorry

end NUMINAMATH_CALUDE_min_value_theorem_l2930_293031


namespace NUMINAMATH_CALUDE_smallest_number_with_all_factors_l2930_293045

def alice_number : ℕ := 36

def has_all_prime_factors (n m : ℕ) : Prop :=
  ∀ p : ℕ, p.Prime → (p ∣ n → p ∣ m)

theorem smallest_number_with_all_factors :
  ∃ k : ℕ, k > 0 ∧ has_all_prime_factors alice_number k ∧
  ∀ m : ℕ, m > 0 → has_all_prime_factors alice_number m → k ≤ m :=
by sorry

end NUMINAMATH_CALUDE_smallest_number_with_all_factors_l2930_293045


namespace NUMINAMATH_CALUDE_cube_displacement_l2930_293064

/-- The volume of water displaced by a cube in a cylindrical barrel -/
theorem cube_displacement (cube_side : ℝ) (barrel_radius barrel_height : ℝ) 
  (h_cube_side : cube_side = 6)
  (h_barrel_radius : barrel_radius = 5)
  (h_barrel_height : barrel_height = 12)
  (h_fully_submerged : cube_side ≤ barrel_height) :
  cube_side ^ 3 = 216 := by
  sorry

end NUMINAMATH_CALUDE_cube_displacement_l2930_293064


namespace NUMINAMATH_CALUDE_function_inequality_and_zero_relation_l2930_293005

noncomputable section

variables (a : ℝ) (x x₀ x₁ x₂ : ℝ)

def f (x : ℝ) : ℝ := a * x^2 + Real.log x

def g (x : ℝ) : ℝ := 2 * x + (a / 2) * Real.log x

theorem function_inequality_and_zero_relation 
  (h₁ : ∀ x > 0, f x ≥ g x)
  (h₂ : f x₁ = 0)
  (h₃ : f x₂ = 0)
  (h₄ : x₁ < x₂)
  (h₅ : x₀ = -a/4) :
  a ≥ (4 + 4 * Real.log 2) / (1 + 2 * Real.log 2) ∧ 
  x₁ / x₂ > 4 * Real.exp x₀ :=
sorry

end NUMINAMATH_CALUDE_function_inequality_and_zero_relation_l2930_293005


namespace NUMINAMATH_CALUDE_sqrt_six_range_l2930_293078

theorem sqrt_six_range : 2 < Real.sqrt 6 ∧ Real.sqrt 6 < 3 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_six_range_l2930_293078


namespace NUMINAMATH_CALUDE_remaining_digits_count_l2930_293040

theorem remaining_digits_count (total_count : ℕ) (total_avg : ℚ) (subset_count : ℕ) (subset_avg : ℚ) (remaining_avg : ℚ) :
  total_count = 8 →
  total_avg = 20 →
  subset_count = 5 →
  subset_avg = 12 →
  remaining_avg = 33333333333333336 / 1000000000000000 →
  total_count - subset_count = 3 :=
by sorry

end NUMINAMATH_CALUDE_remaining_digits_count_l2930_293040


namespace NUMINAMATH_CALUDE_right_triangle_area_l2930_293033

theorem right_triangle_area (a b c : ℝ) (h1 : a = 18) (h2 : c = 30) (h3 : a^2 + b^2 = c^2) :
  (1/2) * a * b = 216 := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_area_l2930_293033


namespace NUMINAMATH_CALUDE_gcd_factorial_problem_l2930_293018

theorem gcd_factorial_problem : Nat.gcd (Nat.factorial 8) ((Nat.factorial 6)^2 * 2^3) = 5760 := by
  sorry

end NUMINAMATH_CALUDE_gcd_factorial_problem_l2930_293018


namespace NUMINAMATH_CALUDE_min_value_theorem_l2930_293056

theorem min_value_theorem (x y : ℝ) (hx : x > 0) (hy : y > 0)
  (h : (x + 3)⁻¹ ^ (1/3 : ℝ) + (y + 3)⁻¹ ^ (1/3 : ℝ) = 1/2) :
  x + 3*y ≥ 4*(1 + 3^(1/3 : ℝ))^2 - 12 := by
  sorry

end NUMINAMATH_CALUDE_min_value_theorem_l2930_293056


namespace NUMINAMATH_CALUDE_point_not_in_fourth_quadrant_l2930_293007

theorem point_not_in_fourth_quadrant (m : ℝ) :
  ¬(m > 0 ∧ m + 1 < 0) :=
by sorry

end NUMINAMATH_CALUDE_point_not_in_fourth_quadrant_l2930_293007


namespace NUMINAMATH_CALUDE_solve_equation_l2930_293061

theorem solve_equation : 45 / (7 - 3/4) = 36/5 := by sorry

end NUMINAMATH_CALUDE_solve_equation_l2930_293061


namespace NUMINAMATH_CALUDE_malcom_cards_left_l2930_293099

theorem malcom_cards_left (brandon_cards : ℕ) (malcom_extra_cards : ℕ) : 
  brandon_cards = 20 →
  malcom_extra_cards = 8 →
  (brandon_cards + malcom_extra_cards) / 2 = 14 := by
sorry

end NUMINAMATH_CALUDE_malcom_cards_left_l2930_293099


namespace NUMINAMATH_CALUDE_board_tileable_iff_divisibility_l2930_293070

/-- A board is tileable if it can be covered completely with 3×1 tiles -/
def is_tileable (m n : ℕ) : Prop :=
  ∃ (tiling : Set (ℕ × ℕ × Bool)), 
    (∀ (tile : ℕ × ℕ × Bool), tile ∈ tiling → 
      (let (x, y, horizontal) := tile
       (x ≥ m ∨ y ≥ m) ∧ x < n ∧ y < n ∧
       (if horizontal then x + 2 < n else y + 2 < n))) ∧
    (∀ (i j : ℕ), m ≤ i ∧ i < n ∧ m ≤ j ∧ j < n → 
      ∃! (tile : ℕ × ℕ × Bool), tile ∈ tiling ∧
        (let (x, y, horizontal) := tile
         i ∈ Set.range (fun k => x + k) ∧ 
         j ∈ Set.range (fun k => y + k) ∧
         (if horizontal then x + 2 = i else y + 2 = j)))

/-- The main theorem -/
theorem board_tileable_iff_divisibility {m n : ℕ} (h_pos : 0 < m) (h_lt : m < n) :
  is_tileable m n ↔ (n - m) * (n + m) % 3 = 0 :=
sorry

end NUMINAMATH_CALUDE_board_tileable_iff_divisibility_l2930_293070


namespace NUMINAMATH_CALUDE_task_completion_time_l2930_293000

/-- 
Given:
- Person A can complete a task in time a
- Person A and Person B together can complete the task in time c
- The rate of work is the reciprocal of the time taken

Prove:
- Person B can complete the task alone in time b, where 1/a + 1/b = 1/c
-/
theorem task_completion_time (a c : ℝ) (ha : a > 0) (hc : c > 0) (hac : c < a) :
  ∃ b : ℝ, b > 0 ∧ 1/a + 1/b = 1/c := by sorry

end NUMINAMATH_CALUDE_task_completion_time_l2930_293000


namespace NUMINAMATH_CALUDE_investment_sum_l2930_293094

/-- Given a sum invested at different interest rates, prove the sum's value --/
theorem investment_sum (P : ℝ) : 
  (P * 0.15 * 2 - P * 0.12 * 2 = 720) → P = 12000 := by
  sorry

end NUMINAMATH_CALUDE_investment_sum_l2930_293094


namespace NUMINAMATH_CALUDE_polynomial_change_l2930_293029

/-- Given a polynomial f(x) = 2x^2 - 5 and a positive real number b,
    the change in the polynomial's value when x changes by ±b is 4bx ± 2b^2 -/
theorem polynomial_change (x b : ℝ) (h : b > 0) :
  let f : ℝ → ℝ := λ t ↦ 2 * t^2 - 5
  (f (x + b) - f x) = 4 * b * x + 2 * b^2 ∧
  (f (x - b) - f x) = -4 * b * x + 2 * b^2 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_change_l2930_293029


namespace NUMINAMATH_CALUDE_graph_transformation_l2930_293006

/-- Given a function f(x) = sin(x - π/3), prove that after stretching its x-coordinates
    to twice their original length and shifting the resulting graph to the right by π/3 units,
    the equation of the resulting graph is y = sin(x/2 - π/2). -/
theorem graph_transformation (x : ℝ) :
  let f : ℝ → ℝ := fun x ↦ Real.sin (x - π/3)
  let g : ℝ → ℝ := fun x ↦ f (x/2)
  let h : ℝ → ℝ := fun x ↦ g (x - π/3)
  h x = Real.sin (x/2 - π/2) := by
  sorry

end NUMINAMATH_CALUDE_graph_transformation_l2930_293006


namespace NUMINAMATH_CALUDE_sum_of_repeating_decimals_l2930_293074

/-- Represents a repeating decimal with a single digit repeating infinitely. -/
def RepeatingDecimal (n : Nat) : ℚ := n / 9

theorem sum_of_repeating_decimals :
  RepeatingDecimal 6 + RepeatingDecimal 4 = 10 / 9 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_repeating_decimals_l2930_293074


namespace NUMINAMATH_CALUDE_roots_of_equation_l2930_293010

theorem roots_of_equation : 
  let f : ℝ → ℝ := fun x => x * (x - 1) + 3 * (x - 1)
  (f (-3) = 0 ∧ f 1 = 0) ∧ ∀ x : ℝ, f x = 0 → (x = -3 ∨ x = 1) :=
by sorry

end NUMINAMATH_CALUDE_roots_of_equation_l2930_293010


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_condition_l2930_293098

theorem sufficient_not_necessary_condition : 
  (∃ x : ℝ, x ≠ 5 ∧ x^2 - 4*x - 5 = 0) ∧
  (∀ x : ℝ, x = 5 → x^2 - 4*x - 5 = 0) :=
by sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_condition_l2930_293098


namespace NUMINAMATH_CALUDE_smallest_cut_length_l2930_293016

theorem smallest_cut_length : 
  ∃ (x : ℕ), x > 0 ∧ x ≤ 8 ∧ 
  (∀ (y : ℕ), y > 0 → y < x → (8 - y) + (15 - y) > 17 - y) ∧
  (8 - x) + (15 - x) ≤ 17 - x ∧
  x = 6 := by
  sorry

end NUMINAMATH_CALUDE_smallest_cut_length_l2930_293016


namespace NUMINAMATH_CALUDE_inequality_solution_l2930_293032

theorem inequality_solution (x : ℝ) : (x^2 - 9) / (x^2 - 4) > 0 ↔ x < -3 ∨ x > 3 := by
  sorry

end NUMINAMATH_CALUDE_inequality_solution_l2930_293032


namespace NUMINAMATH_CALUDE_three_points_in_circle_l2930_293003

/-- A point in a 2D plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- A square in a 2D plane -/
structure Square where
  side : ℝ

/-- A circle in a 2D plane -/
structure Circle where
  center : Point
  radius : ℝ

/-- Check if a point is inside a square -/
def Point.inSquare (p : Point) (s : Square) : Prop :=
  0 ≤ p.x ∧ p.x ≤ s.side ∧ 0 ≤ p.y ∧ p.y ≤ s.side

/-- Check if a point is inside a circle -/
def Point.inCircle (p : Point) (c : Circle) : Prop :=
  (p.x - c.center.x)^2 + (p.y - c.center.y)^2 ≤ c.radius^2

/-- The main theorem -/
theorem three_points_in_circle (points : Finset Point) (s : Square) :
  s.side = 1 →
  points.card = 51 →
  ∀ p ∈ points, p.inSquare s →
  ∃ (c : Circle) (p1 p2 p3 : Point),
    c.radius = 1/7 ∧
    p1 ∈ points ∧ p2 ∈ points ∧ p3 ∈ points ∧
    p1 ≠ p2 ∧ p1 ≠ p3 ∧ p2 ≠ p3 ∧
    p1.inCircle c ∧ p2.inCircle c ∧ p3.inCircle c :=
sorry


end NUMINAMATH_CALUDE_three_points_in_circle_l2930_293003


namespace NUMINAMATH_CALUDE_fans_with_all_items_l2930_293037

def stadium_capacity : ℕ := 3600
def hat_interval : ℕ := 60
def tshirt_interval : ℕ := 40
def gloves_interval : ℕ := 90

theorem fans_with_all_items :
  (stadium_capacity / (Nat.lcm hat_interval (Nat.lcm tshirt_interval gloves_interval))) = 10 := by
  sorry

end NUMINAMATH_CALUDE_fans_with_all_items_l2930_293037


namespace NUMINAMATH_CALUDE_triangle_problem_l2930_293009

/-- 
Given a triangle ABC with sides a, b, c opposite to angles A, B, C respectively,
prove that if 2b*sin(B) = (2a+c)*sin(A) + (2c+a)*sin(C), b = √3, and A = π/4,
then B = 2π/3 and the area of the triangle is (3 - √3)/4.
-/
theorem triangle_problem (a b c A B C : ℝ) : 
  2 * b * Real.sin B = (2 * a + c) * Real.sin A + (2 * c + a) * Real.sin C →
  b = Real.sqrt 3 →
  A = π / 4 →
  B = 2 * π / 3 ∧ 
  (1 / 2 : ℝ) * b * c * Real.sin A = (3 - Real.sqrt 3) / 4 := by
  sorry


end NUMINAMATH_CALUDE_triangle_problem_l2930_293009


namespace NUMINAMATH_CALUDE_kenny_basketball_hours_l2930_293038

/-- Represents the number of hours Kenny spent on different activities -/
structure KennyActivities where
  basketball : ℕ
  running : ℕ
  trumpet : ℕ

/-- Defines the relationships between Kenny's activities -/
def valid_activities (k : KennyActivities) : Prop :=
  k.running = 2 * k.basketball ∧
  k.trumpet = 2 * k.running ∧
  k.trumpet = 40

/-- Theorem: Given the conditions, Kenny played basketball for 10 hours -/
theorem kenny_basketball_hours (k : KennyActivities) 
  (h : valid_activities k) : k.basketball = 10 := by
  sorry


end NUMINAMATH_CALUDE_kenny_basketball_hours_l2930_293038


namespace NUMINAMATH_CALUDE_intersection_nonempty_implies_a_less_than_one_l2930_293059

theorem intersection_nonempty_implies_a_less_than_one (a : ℝ) : 
  let M := {x : ℝ | x ≤ 1}
  let P := {x : ℝ | x > a}
  (M ∩ P).Nonempty → a < 1 := by
  sorry

end NUMINAMATH_CALUDE_intersection_nonempty_implies_a_less_than_one_l2930_293059


namespace NUMINAMATH_CALUDE_quadratic_right_triangle_specific_quadratic_right_triangle_l2930_293090

-- Define the quadratic function
def quadratic (a b c x : ℝ) : ℝ := a * x^2 + b * x + c

-- Define the specific quadratic function
def specific_quadratic (m x : ℝ) : ℝ := x^2 - (2*m + 2)*x + m^2 + 5*m + 3

-- Define the linear function
def linear (x : ℝ) : ℝ := 3*x - 1

-- Theorem for the first part
theorem quadratic_right_triangle (a b c : ℝ) (ha : a ≠ 0) :
  (∃ A B C : ℝ × ℝ, 
    quadratic a b c A.1 = 0 ∧ 
    quadratic a b c B.1 = 0 ∧
    (C.1 = -b / (2*a) ∧ C.2 = quadratic a b c C.1) ∧
    (B.1 - A.1)^2 + (C.2 - A.2)^2 = (C.1 - A.1)^2) →
  b^2 - 4*a*c = 4 :=
sorry

-- Theorem for the second part
theorem specific_quadratic_right_triangle (m : ℝ) :
  (∃ E F G : ℝ × ℝ,
    specific_quadratic m E.1 = 0 ∧
    specific_quadratic m F.1 = 0 ∧
    specific_quadratic m G.1 = linear G.1 ∧
    (∀ x, specific_quadratic m x = linear x → G.2 ≤ specific_quadratic m x) ∧
    (F.1 - E.1)^2 + (G.2 - E.2)^2 = (G.1 - E.1)^2) →
  m = -1 :=
sorry

end NUMINAMATH_CALUDE_quadratic_right_triangle_specific_quadratic_right_triangle_l2930_293090


namespace NUMINAMATH_CALUDE_problem_statement_l2930_293036

/-- Given positive real numbers a, b, c, and a function f with minimum value 1, 
    prove that a + b + c = 1 and a² + b² + c² ≥ 1/3 -/
theorem problem_statement (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) 
    (hf : ∀ x, |x - a| + |x + b| + c ≥ 1) : 
    (a + b + c = 1) ∧ (a^2 + b^2 + c^2 ≥ 1/3) := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l2930_293036


namespace NUMINAMATH_CALUDE_solve_exponential_equation_l2930_293049

theorem solve_exponential_equation :
  ∃ x : ℝ, 5^(3*x) = Real.sqrt 625 ∧ x = 2/3 := by
  sorry

end NUMINAMATH_CALUDE_solve_exponential_equation_l2930_293049


namespace NUMINAMATH_CALUDE_cube_volume_from_surface_area_l2930_293085

/-- Given a cube with surface area 24 square centimeters, its volume is 8 cubic centimeters. -/
theorem cube_volume_from_surface_area :
  ∀ (side_length : ℝ),
  (6 * side_length^2 = 24) →
  side_length^3 = 8 := by
sorry

end NUMINAMATH_CALUDE_cube_volume_from_surface_area_l2930_293085


namespace NUMINAMATH_CALUDE_special_polynomial_form_l2930_293008

/-- A polynomial satisfying the given functional equation. -/
structure SpecialPolynomial where
  P : ℝ → ℝ
  equation_holds : ∀ (a b c : ℝ),
    P (a + b - 2*c) + P (b + c - 2*a) + P (c + a - 2*b) =
    3 * P (a - b) + 3 * P (b - c) + 3 * P (c - a)

/-- The theorem stating the form of polynomials satisfying the functional equation. -/
theorem special_polynomial_form (p : SpecialPolynomial) :
  ∃ (a b : ℝ), (∀ x, p.P x = a * x + b) ∨ (∀ x, p.P x = a * x^2 + b * x) := by
  sorry

end NUMINAMATH_CALUDE_special_polynomial_form_l2930_293008


namespace NUMINAMATH_CALUDE_prime_average_count_l2930_293087

theorem prime_average_count : 
  ∃ (p₁ p₂ p₃ : ℕ), 
    Prime p₁ ∧ Prime p₂ ∧ Prime p₃ ∧
    p₁ > 20 ∧ p₂ > 20 ∧ p₃ > 20 ∧
    (p₁ + p₂ + p₃) / 3 = 83 / 3 ∧
    ∀ (q₁ q₂ q₃ q₄ : ℕ), 
      Prime q₁ ∧ Prime q₂ ∧ Prime q₃ ∧ Prime q₄ ∧
      q₁ > 20 ∧ q₂ > 20 ∧ q₃ > 20 ∧ q₄ > 20 →
      (q₁ + q₂ + q₃ + q₄) / 4 ≠ 83 / 3 :=
by sorry

end NUMINAMATH_CALUDE_prime_average_count_l2930_293087


namespace NUMINAMATH_CALUDE_sum_of_k_values_l2930_293013

theorem sum_of_k_values : ∃ (S : Finset ℕ), 
  (∀ k ∈ S, ∃ j : ℕ, j > 0 ∧ k > 0 ∧ (1 : ℚ) / j + 1 / k = (1 : ℚ) / 4) ∧
  (∀ k : ℕ, k > 0 → (∃ j : ℕ, j > 0 ∧ (1 : ℚ) / j + 1 / k = (1 : ℚ) / 4) → k ∈ S) ∧
  Finset.sum S id = 51 :=
sorry

end NUMINAMATH_CALUDE_sum_of_k_values_l2930_293013


namespace NUMINAMATH_CALUDE_min_sum_of_indices_l2930_293081

def is_geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, r > 0 ∧ ∀ n : ℕ, a (n + 1) = a n * r

theorem min_sum_of_indices (a : ℕ → ℝ) :
  is_geometric_sequence a →
  (∀ n : ℕ, a n > 0) →
  a 7 = a 6 + 2 * a 5 →
  (∃ m n : ℕ, a m + a n = 4 * a 1) →
  ∃ m n : ℕ, a m + a n = 4 * a 1 ∧ m + n = 4 ∧ ∀ k l : ℕ, a k + a l = 4 * a 1 → k + l ≥ 4 := by
  sorry

end NUMINAMATH_CALUDE_min_sum_of_indices_l2930_293081


namespace NUMINAMATH_CALUDE_opposite_of_negative_two_l2930_293058

theorem opposite_of_negative_two : -((-2 : ℤ)) = 2 := by sorry

end NUMINAMATH_CALUDE_opposite_of_negative_two_l2930_293058


namespace NUMINAMATH_CALUDE_largest_multiple_of_8_with_negation_greater_than_neg_200_l2930_293020

theorem largest_multiple_of_8_with_negation_greater_than_neg_200 :
  ∃ (n : ℤ), n = 192 ∧ 
  (∀ (m : ℤ), m % 8 = 0 ∧ -m > -200 → m ≤ n) ∧
  192 % 8 = 0 ∧
  -192 > -200 :=
by sorry

end NUMINAMATH_CALUDE_largest_multiple_of_8_with_negation_greater_than_neg_200_l2930_293020


namespace NUMINAMATH_CALUDE_count_solutions_l2930_293091

/-- Sum of digits function -/
def S (n : ℕ) : ℕ := sorry

/-- The main theorem -/
theorem count_solutions : 
  (Finset.filter (fun n => n + S n + S (S n) = 2023) (Finset.range 2024)).card = 4 := by sorry

end NUMINAMATH_CALUDE_count_solutions_l2930_293091


namespace NUMINAMATH_CALUDE_largest_triangle_perimeter_l2930_293073

theorem largest_triangle_perimeter : ∀ x : ℤ,
  (8 : ℝ) + 11 > (x : ℝ) ∧ 
  (8 : ℝ) + (x : ℝ) > 11 ∧ 
  (11 : ℝ) + (x : ℝ) > 8 →
  (8 : ℝ) + 11 + (x : ℝ) ≤ 37 :=
by
  sorry

end NUMINAMATH_CALUDE_largest_triangle_perimeter_l2930_293073


namespace NUMINAMATH_CALUDE_line_passes_through_fixed_point_l2930_293023

/-- The line equation passes through a fixed point for all values of k -/
theorem line_passes_through_fixed_point :
  ∀ (k : ℝ), (k + 2) * 3 + (1 - k) * (-1) - 4 * k - 5 = 0 := by
  sorry

#check line_passes_through_fixed_point

end NUMINAMATH_CALUDE_line_passes_through_fixed_point_l2930_293023


namespace NUMINAMATH_CALUDE_phi_eq_c_is_cone_l2930_293025

/-- Spherical coordinates -/
structure SphericalCoord where
  ρ : ℝ
  θ : ℝ
  φ : ℝ

/-- A constant angle from the z-axis -/
def ConstantAngle (c : ℝ) (p : SphericalCoord) : Prop :=
  p.φ = c

/-- Definition of a cone in spherical coordinates -/
def IsCone (s : Set SphericalCoord) : Prop :=
  ∃ c : ℝ, ∀ p ∈ s, ConstantAngle c p

/-- Theorem: The equation φ = c describes a cone in spherical coordinates -/
theorem phi_eq_c_is_cone (c : ℝ) :
  IsCone {p : SphericalCoord | p.φ = c} :=
sorry

end NUMINAMATH_CALUDE_phi_eq_c_is_cone_l2930_293025


namespace NUMINAMATH_CALUDE_least_three_digit_multiple_of_eight_l2930_293021

theorem least_three_digit_multiple_of_eight : 
  (∀ n : ℕ, n ≥ 100 ∧ n < 104 → n % 8 ≠ 0) ∧ 
  104 % 8 = 0 ∧ 
  104 ≥ 100 ∧ 
  104 < 1000 :=
sorry

end NUMINAMATH_CALUDE_least_three_digit_multiple_of_eight_l2930_293021


namespace NUMINAMATH_CALUDE_banana_cantaloupe_cost_l2930_293060

def cost_problem (apple banana cantaloupe date : ℝ) : Prop :=
  apple + banana + cantaloupe + date = 40 ∧
  date = 3 * apple ∧
  banana = cantaloupe - 2

theorem banana_cantaloupe_cost 
  (apple banana cantaloupe date : ℝ)
  (h : cost_problem apple banana cantaloupe date) :
  banana + cantaloupe = 20 := by
sorry

end NUMINAMATH_CALUDE_banana_cantaloupe_cost_l2930_293060


namespace NUMINAMATH_CALUDE_wonderful_quadratic_range_l2930_293052

/-- A function is wonderful on a domain if it's monotonic and there exists an interval [a,b] in the domain
    such that the range of f on [a,b] is exactly [a,b] --/
def IsWonderful (f : ℝ → ℝ) (D : Set ℝ) : Prop :=
  Monotone f ∧ ∃ a b, a < b ∧ Set.Icc a b ⊆ D ∧
    (∀ x ∈ Set.Icc a b, f x ∈ Set.Icc a b) ∧
    (∀ y ∈ Set.Icc a b, ∃ x ∈ Set.Icc a b, f x = y)

theorem wonderful_quadratic_range (m : ℝ) :
  IsWonderful (fun x => x^2 + m) (Set.Iic 0) →
  m ∈ Set.Ioo (-1) (-3/4) :=
sorry

end NUMINAMATH_CALUDE_wonderful_quadratic_range_l2930_293052


namespace NUMINAMATH_CALUDE_circle_placement_existence_l2930_293039

/-- Represents a rectangle --/
structure Rectangle where
  width : ℝ
  height : ℝ

/-- Represents a square --/
structure Square where
  side : ℝ

/-- Represents a circle --/
structure Circle where
  diameter : ℝ

/-- Checks if a circle intersects a square --/
def circleIntersectsSquare (c : Circle) (s : Square) : Prop :=
  sorry

/-- Theorem: In a 20 by 25 rectangle with 120 unit squares, 
    there exists a point for a circle with diameter 1 that doesn't intersect any square --/
theorem circle_placement_existence 
  (r : Rectangle) 
  (squares : Finset Square) 
  (c : Circle) : 
  r.width = 20 ∧ 
  r.height = 25 ∧ 
  squares.card = 120 ∧ 
  (∀ s ∈ squares, s.side = 1) ∧ 
  c.diameter = 1 →
  ∃ (x y : ℝ), ∀ s ∈ squares, ¬circleIntersectsSquare { diameter := 1 } s :=
sorry

end NUMINAMATH_CALUDE_circle_placement_existence_l2930_293039


namespace NUMINAMATH_CALUDE_two_fifths_percent_of_450_l2930_293062

theorem two_fifths_percent_of_450 : (2 / 5) / 100 * 450 = 1.8 := by
  sorry

end NUMINAMATH_CALUDE_two_fifths_percent_of_450_l2930_293062


namespace NUMINAMATH_CALUDE_arithmetic_sequence_problem_l2930_293028

/-- An arithmetic sequence is a sequence where the difference between
    any two consecutive terms is constant. -/
def ArithmeticSequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

/-- Given an arithmetic sequence where the 4th term is 23 and the 6th term is 47,
    the 8th term is 71. -/
theorem arithmetic_sequence_problem (a : ℕ → ℝ) 
    (h_arith : ArithmeticSequence a) 
    (h_4th : a 4 = 23) 
    (h_6th : a 6 = 47) : 
  a 8 = 71 := by
  sorry


end NUMINAMATH_CALUDE_arithmetic_sequence_problem_l2930_293028


namespace NUMINAMATH_CALUDE_min_area_quadrilateral_l2930_293093

/-- Given a rectangle ABCD with points A₁, B₁, C₁, D₁ on the rays AB, BC, CD, DA respectively,
    such that AA₁/AB = BB₁/BC = CC₁/CD = DD₁/DA = k > 0,
    prove that the area of quadrilateral A₁B₁C₁D₁ is minimized when k = 1/2 -/
theorem min_area_quadrilateral (a b : ℝ) (k : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : k > 0) :
  let area := a * b * (1 - k + k^2)
  (∀ k' > 0, area ≤ a * b * (1 - k' + k'^2)) ↔ k = 1/2 := by
  sorry


end NUMINAMATH_CALUDE_min_area_quadrilateral_l2930_293093


namespace NUMINAMATH_CALUDE_vasyas_numbers_l2930_293095

theorem vasyas_numbers (x y : ℝ) (h1 : x + y = x * y) (h2 : x * y = x / y) : 
  x = (1 : ℝ) / 2 ∧ y = -(1 : ℝ) := by
sorry

end NUMINAMATH_CALUDE_vasyas_numbers_l2930_293095


namespace NUMINAMATH_CALUDE_linda_coin_ratio_l2930_293092

/-- Represents the coin types in Linda's bag -/
inductive Coin
  | Dime
  | Quarter
  | Nickel

/-- Represents Linda's initial coin counts -/
structure InitialCoins where
  dimes : Nat
  quarters : Nat
  nickels : Nat

/-- Represents the additional coins given by Linda's mother -/
structure AdditionalCoins where
  dimes : Nat
  quarters : Nat

def total_coins : Nat := 35

theorem linda_coin_ratio 
  (initial : InitialCoins)
  (additional : AdditionalCoins)
  (h_initial_dimes : initial.dimes = 2)
  (h_initial_quarters : initial.quarters = 6)
  (h_initial_nickels : initial.nickels = 5)
  (h_additional_dimes : additional.dimes = 2)
  (h_additional_quarters : additional.quarters = 10)
  (h_total_coins : total_coins = 35) :
  (total_coins - (initial.dimes + additional.dimes + initial.quarters + additional.quarters) - initial.nickels) / initial.nickels = 2 := by
  sorry


end NUMINAMATH_CALUDE_linda_coin_ratio_l2930_293092


namespace NUMINAMATH_CALUDE_difference_of_cubes_factorization_l2930_293066

theorem difference_of_cubes_factorization (a b c d e : ℚ) :
  (∀ x, 512 * x^3 - 27 = (a*x + b) * (c*x^2 + d*x + e)) →
  a + b + c + d + e = 102 := by
sorry

end NUMINAMATH_CALUDE_difference_of_cubes_factorization_l2930_293066


namespace NUMINAMATH_CALUDE_sqrt_sum_square_condition_l2930_293068

theorem sqrt_sum_square_condition (a b : ℝ) :
  Real.sqrt (a^2 + b^2 + 2*a*b) = a + b ↔ a + b ≥ 0 := by sorry

end NUMINAMATH_CALUDE_sqrt_sum_square_condition_l2930_293068


namespace NUMINAMATH_CALUDE_range_of_x_l2930_293097

noncomputable def f (x : ℝ) : ℝ := Real.log x + 2^x + x - 1

theorem range_of_x (x : ℝ) : f (x^2 - 4) < 2 → x ∈ Set.Ioo (-Real.sqrt 5) (-2) ∪ Set.Ioo 2 (Real.sqrt 5) := by
  sorry

end NUMINAMATH_CALUDE_range_of_x_l2930_293097


namespace NUMINAMATH_CALUDE_inequality_holds_iff_p_greater_than_two_point_five_l2930_293080

theorem inequality_holds_iff_p_greater_than_two_point_five (p q : ℝ) (hq : q > 0) :
  (5 * (p * q^2 + p^2 * q + 4 * q^2 + 4 * p * q)) / (p + q) > 3 * p^2 * q ↔ p > 2.5 := by
  sorry

end NUMINAMATH_CALUDE_inequality_holds_iff_p_greater_than_two_point_five_l2930_293080


namespace NUMINAMATH_CALUDE_correct_algebraic_operation_l2930_293069

theorem correct_algebraic_operation (a b c : ℝ) : 2 * a^2 * b * c - a^2 * b * c = a^2 * b * c := by
  sorry

end NUMINAMATH_CALUDE_correct_algebraic_operation_l2930_293069


namespace NUMINAMATH_CALUDE_pet_store_cats_l2930_293084

/-- The number of house cats initially in the pet store -/
def initial_house_cats : ℝ := 5.0

/-- The number of Siamese cats initially in the pet store -/
def initial_siamese_cats : ℝ := 13.0

/-- The number of cats added during the purchase -/
def added_cats : ℝ := 10.0

/-- The total number of cats after the addition -/
def total_cats_after : ℝ := 28.0

theorem pet_store_cats :
  initial_house_cats + initial_siamese_cats + added_cats = total_cats_after :=
by sorry

end NUMINAMATH_CALUDE_pet_store_cats_l2930_293084


namespace NUMINAMATH_CALUDE_sufficiency_not_necessity_a_squared_greater_b_squared_l2930_293072

theorem sufficiency_not_necessity_a_squared_greater_b_squared (a b : ℝ) :
  (∀ a b : ℝ, a > b ∧ b > 0 → a^2 > b^2) ∧
  (∃ a b : ℝ, a^2 > b^2 ∧ ¬(a > b ∧ b > 0)) :=
sorry

end NUMINAMATH_CALUDE_sufficiency_not_necessity_a_squared_greater_b_squared_l2930_293072


namespace NUMINAMATH_CALUDE_tangent_point_coordinates_l2930_293011

theorem tangent_point_coordinates (x y : ℝ) :
  y = x^2 →  -- curve equation
  (2 * x = -3) →  -- slope condition
  (x = -3/2 ∧ y = 9/4)  -- coordinates of point P
  := by sorry

end NUMINAMATH_CALUDE_tangent_point_coordinates_l2930_293011


namespace NUMINAMATH_CALUDE_parabola_chord_theorem_l2930_293046

/-- Represents a point on a 2D plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a parabola y = x^2 -/
def parabola (p : Point) : Prop := p.y = p.x^2

/-- Checks if three points are collinear -/
def collinear (p1 p2 p3 : Point) : Prop :=
  (p2.y - p1.y) * (p3.x - p1.x) = (p3.y - p1.y) * (p2.x - p1.x)

/-- Checks if a point divides a line segment in a given ratio -/
def divides_in_ratio (p1 p2 p : Point) (m n : ℝ) : Prop :=
  n * (p.x - p1.x) = m * (p2.x - p.x) ∧ n * (p.y - p1.y) = m * (p2.y - p.y)

theorem parabola_chord_theorem (A B C : Point) :
  parabola A ∧ parabola B ∧  -- A and B lie on the parabola
  C.x = 0 ∧ C.y = 15 ∧  -- C is on y-axis with y-coordinate 15
  collinear A B C ∧  -- A, B, and C are collinear
  divides_in_ratio A B C 5 3 →  -- C divides AB in ratio 5:3
  ((A.x = -5 ∧ B.x = 3) ∨ (A.x = 5 ∧ B.x = -3)) := by sorry

end NUMINAMATH_CALUDE_parabola_chord_theorem_l2930_293046


namespace NUMINAMATH_CALUDE_team_combinations_eq_18018_l2930_293044

/-- The number of ways to choose k elements from n elements --/
def choose (n k : ℕ) : ℕ := Nat.choose n k

/-- The total number of players in the basketball team --/
def total_players : ℕ := 18

/-- The number of quadruplets in the team --/
def num_quadruplets : ℕ := 4

/-- The size of the team to be formed --/
def team_size : ℕ := 8

/-- The number of quadruplets that must be in the team --/
def required_quadruplets : ℕ := 2

/-- The number of ways to choose 8 players from a team of 18 players, 
    including exactly 2 out of 4 quadruplets --/
def team_combinations : ℕ :=
  choose num_quadruplets required_quadruplets * 
  choose (total_players - num_quadruplets) (team_size - required_quadruplets)

theorem team_combinations_eq_18018 : team_combinations = 18018 := by
  sorry

end NUMINAMATH_CALUDE_team_combinations_eq_18018_l2930_293044


namespace NUMINAMATH_CALUDE_fourth_group_number_l2930_293063

/-- Systematic sampling function -/
def systematic_sample (total : ℕ) (start : ℕ) (interval : ℕ) (group : ℕ) : ℕ :=
  start + (group - 1) * interval

/-- Theorem: In a systematic sampling of 90 students, with adjacent group numbers 14 and 23,
    the student number from the fourth group is 32. -/
theorem fourth_group_number :
  let total := 90
  let start := 14
  let interval := 23 - 14
  let group := 4
  systematic_sample total start interval group = 32 := by
  sorry

end NUMINAMATH_CALUDE_fourth_group_number_l2930_293063


namespace NUMINAMATH_CALUDE_complex_magnitude_l2930_293024

theorem complex_magnitude (z : ℂ) (h : z * (1 + Complex.I) = 3 + Complex.I) : Complex.abs z = Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_complex_magnitude_l2930_293024


namespace NUMINAMATH_CALUDE_correct_bouquet_flowers_l2930_293055

def flowers_for_bouquets (tulips roses extra : ℕ) : ℕ :=
  tulips + roses - extra

theorem correct_bouquet_flowers :
  flowers_for_bouquets 39 49 7 = 81 := by
  sorry

end NUMINAMATH_CALUDE_correct_bouquet_flowers_l2930_293055


namespace NUMINAMATH_CALUDE_certain_amount_problem_l2930_293088

theorem certain_amount_problem (first_number : ℕ) (certain_amount : ℕ) : 
  first_number = 5 →
  first_number + (11 + certain_amount) = 19 →
  certain_amount = 3 := by
sorry

end NUMINAMATH_CALUDE_certain_amount_problem_l2930_293088


namespace NUMINAMATH_CALUDE_polynomial_factorization_l2930_293027

theorem polynomial_factorization (m x y : ℝ) : 4*m*x^2 - m*y^2 = m*(2*x+y)*(2*x-y) := by
  sorry

end NUMINAMATH_CALUDE_polynomial_factorization_l2930_293027


namespace NUMINAMATH_CALUDE_sarah_desserts_l2930_293089

theorem sarah_desserts (michael_cookies : ℕ) (sarah_cupcakes : ℕ) :
  michael_cookies = 5 →
  sarah_cupcakes = 9 →
  sarah_cupcakes / 3 = sarah_cupcakes - (sarah_cupcakes / 3) →
  michael_cookies + (sarah_cupcakes - (sarah_cupcakes / 3)) = 11 :=
by sorry

end NUMINAMATH_CALUDE_sarah_desserts_l2930_293089


namespace NUMINAMATH_CALUDE_sequence_constant_iff_perfect_square_l2930_293019

/-- S(n) is defined as n minus the largest perfect square less than or equal to n -/
def S (n : ℕ) : ℕ := n - (Nat.sqrt n) ^ 2

/-- The sequence a_n is defined recursively -/
def a (A : ℕ) : ℕ → ℕ
  | 0 => A
  | n + 1 => a A n + S (a A n)

/-- A non-negative integer is a perfect square if it's equal to some integer squared -/
def is_perfect_square (n : ℕ) : Prop := ∃ k : ℕ, n = k ^ 2

/-- The main theorem: the sequence becomes constant iff A is a perfect square -/
theorem sequence_constant_iff_perfect_square (A : ℕ) :
  (∃ N : ℕ, ∀ n ≥ N, a A n = a A N) ↔ is_perfect_square A := by
  sorry

end NUMINAMATH_CALUDE_sequence_constant_iff_perfect_square_l2930_293019


namespace NUMINAMATH_CALUDE_notebook_cost_l2930_293002

/-- Proves that the cost of each notebook before discount is $1.48 -/
theorem notebook_cost 
  (total_spent : ℚ)
  (num_backpacks : ℕ)
  (num_pen_packs : ℕ)
  (num_pencil_packs : ℕ)
  (num_notebooks : ℕ)
  (num_calculators : ℕ)
  (discount_rate : ℚ)
  (backpack_price : ℚ)
  (pen_pack_price : ℚ)
  (pencil_pack_price : ℚ)
  (calculator_price : ℚ)
  (h1 : total_spent = 56)
  (h2 : num_backpacks = 1)
  (h3 : num_pen_packs = 3)
  (h4 : num_pencil_packs = 2)
  (h5 : num_notebooks = 5)
  (h6 : num_calculators = 1)
  (h7 : discount_rate = 1/10)
  (h8 : backpack_price = 30)
  (h9 : pen_pack_price = 2)
  (h10 : pencil_pack_price = 3/2)
  (h11 : calculator_price = 15) :
  let other_items_cost := backpack_price * num_backpacks + 
                          pen_pack_price * num_pen_packs + 
                          pencil_pack_price * num_pencil_packs + 
                          calculator_price * num_calculators
  let discounted_other_items_cost := other_items_cost * (1 - discount_rate)
  let notebooks_cost := total_spent - discounted_other_items_cost
  notebooks_cost / num_notebooks = 37/25 := by
  sorry

end NUMINAMATH_CALUDE_notebook_cost_l2930_293002


namespace NUMINAMATH_CALUDE_sum_of_logarithms_l2930_293017

-- Define the base-10 logarithm
noncomputable def log10 (x : ℝ) : ℝ := Real.log x / Real.log 10

theorem sum_of_logarithms (a b : ℝ) (ha : (10 : ℝ) ^ a = 2) (hb : b = log10 5) :
  a + b = 1 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_logarithms_l2930_293017


namespace NUMINAMATH_CALUDE_change_received_l2930_293057

def skirt_price : ℕ := 13
def skirt_count : ℕ := 2
def blouse_price : ℕ := 6
def blouse_count : ℕ := 3
def amount_paid : ℕ := 100

def total_cost : ℕ := skirt_price * skirt_count + blouse_price * blouse_count

theorem change_received : amount_paid - total_cost = 56 := by
  sorry

end NUMINAMATH_CALUDE_change_received_l2930_293057


namespace NUMINAMATH_CALUDE_friendly_pair_solution_l2930_293076

/-- Definition of a friendly number pair -/
def is_friendly_pair (m n : ℚ) : Prop :=
  m / 2 + n / 4 = (m + n) / (2 + 4)

/-- Theorem: If (a, 3) is a friendly number pair, then a = -3/4 -/
theorem friendly_pair_solution (a : ℚ) :
  is_friendly_pair a 3 → a = -3/4 := by
  sorry

end NUMINAMATH_CALUDE_friendly_pair_solution_l2930_293076


namespace NUMINAMATH_CALUDE_square_value_l2930_293035

theorem square_value (square q : ℤ) 
  (eq1 : square + q = 74)
  (eq2 : square + 2 * q ^ 2 = 180) : 
  square = 66 := by sorry

end NUMINAMATH_CALUDE_square_value_l2930_293035


namespace NUMINAMATH_CALUDE_fifteen_tomorrow_l2930_293048

/-- Represents the fishing schedule in a coastal village -/
structure FishingSchedule where
  daily : ℕ           -- Number of people fishing daily
  everyOther : ℕ      -- Number of people fishing every other day
  everyThree : ℕ      -- Number of people fishing every three days
  yesterday : ℕ       -- Number of people who fished yesterday
  today : ℕ           -- Number of people fishing today

/-- Calculates the number of people who will fish tomorrow given a fishing schedule -/
def tomorrowsFishers (schedule : FishingSchedule) : ℕ :=
  sorry

/-- Theorem stating that given the specific fishing schedule, 15 people will fish tomorrow -/
theorem fifteen_tomorrow (schedule : FishingSchedule) 
  (h1 : schedule.daily = 7)
  (h2 : schedule.everyOther = 8)
  (h3 : schedule.everyThree = 3)
  (h4 : schedule.yesterday = 12)
  (h5 : schedule.today = 10) :
  tomorrowsFishers schedule = 15 := by
  sorry

end NUMINAMATH_CALUDE_fifteen_tomorrow_l2930_293048


namespace NUMINAMATH_CALUDE_cos_shift_equals_sin_l2930_293051

theorem cos_shift_equals_sin (x : ℝ) : 
  Real.cos (2 * x - π / 4) = Real.sin (2 * (x + π / 8)) := by
  sorry

end NUMINAMATH_CALUDE_cos_shift_equals_sin_l2930_293051


namespace NUMINAMATH_CALUDE_min_tangent_length_l2930_293054

/-- The minimum length of a tangent from a point on y = x + 1 to (x-3)^2 + y^2 = 1 is √7 -/
theorem min_tangent_length :
  let line := {p : ℝ × ℝ | p.2 = p.1 + 1}
  let circle := {p : ℝ × ℝ | (p.1 - 3)^2 + p.2^2 = 1}
  ∃ (min_length : ℝ),
    min_length = Real.sqrt 7 ∧
    ∀ (p : ℝ × ℝ) (t : ℝ × ℝ),
      p ∈ line → t ∈ circle →
      dist p t ≥ min_length :=
by sorry


end NUMINAMATH_CALUDE_min_tangent_length_l2930_293054


namespace NUMINAMATH_CALUDE_complex_equation_sum_l2930_293042

theorem complex_equation_sum (a b : ℝ) : 
  (a - Complex.I = 2 + b * Complex.I) → (a + b = 1) := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_sum_l2930_293042


namespace NUMINAMATH_CALUDE_octagon_area_theorem_l2930_293065

/-- The area of a regular octagon inscribed in a square with perimeter 144 cm,
    where each side of the square is trisected by the vertices of the octagon. -/
def inscribedOctagonArea : ℝ := 1008

/-- The perimeter of the square. -/
def squarePerimeter : ℝ := 144

/-- A side of the square is trisected by the vertices of the octagon. -/
def isTrisected (s : ℝ) : Prop := ∃ p : ℝ, s = 3 * p

theorem octagon_area_theorem (s : ℝ) (h1 : s * 4 = squarePerimeter) (h2 : isTrisected s) :
  inscribedOctagonArea = s^2 - 4 * (s/3)^2 :=
sorry

end NUMINAMATH_CALUDE_octagon_area_theorem_l2930_293065


namespace NUMINAMATH_CALUDE_two_digit_numbers_count_l2930_293096

def digits_a : Finset Nat := {1, 2, 3, 4, 5, 6}
def digits_b : Finset Nat := {0, 1, 2, 3, 4, 5, 6}

def is_two_digit (n : Nat) : Prop := n ≥ 10 ∧ n ≤ 99

def count_two_digit_numbers (digits : Finset Nat) : Nat :=
  (digits.filter (λ d => d > 0)).card * digits.card

theorem two_digit_numbers_count :
  (count_two_digit_numbers digits_a = 36) ∧
  (count_two_digit_numbers digits_b = 42) := by sorry

end NUMINAMATH_CALUDE_two_digit_numbers_count_l2930_293096


namespace NUMINAMATH_CALUDE_min_value_of_quartic_plus_constant_l2930_293082

theorem min_value_of_quartic_plus_constant :
  ∃ (min : ℝ), min = 2023 ∧ ∀ (x : ℝ), (x + 2) * (x + 3) * (x + 4) * (x + 5) + 2024 ≥ min := by
  sorry

end NUMINAMATH_CALUDE_min_value_of_quartic_plus_constant_l2930_293082


namespace NUMINAMATH_CALUDE_algebraic_expression_value_l2930_293034

theorem algebraic_expression_value (a b : ℝ) (h1 : a ≠ b) 
  (h2 : a^2 - 5*a + 2 = 0) (h3 : b^2 - 5*b + 2 = 0) : 
  (b - 1) / (a - 1) + (a - 1) / (b - 1) = -13/2 := by
  sorry

end NUMINAMATH_CALUDE_algebraic_expression_value_l2930_293034


namespace NUMINAMATH_CALUDE_f_properties_l2930_293050

open Real

noncomputable def f (x : ℝ) : ℝ := (log x) / x

theorem f_properties :
  (∃ (x_max : ℝ), x_max = ℯ ∧ ∀ (x : ℝ), x > 0 → f x ≤ f x_max) ∧
  (f 4 < f π ∧ f π < f 3) ∧
  (π^4 < 4^π) := by
  sorry

end NUMINAMATH_CALUDE_f_properties_l2930_293050


namespace NUMINAMATH_CALUDE_select_two_from_five_assign_prizes_l2930_293075

/-- The number of ways to select 2 people from n employees and assign them distinct prizes -/
def select_and_assign (n : ℕ) : ℕ := n * (n - 1)

/-- Theorem: For 5 employees, there are 20 ways to select 2 and assign distinct prizes -/
theorem select_two_from_five_assign_prizes :
  select_and_assign 5 = 20 := by
  sorry

end NUMINAMATH_CALUDE_select_two_from_five_assign_prizes_l2930_293075


namespace NUMINAMATH_CALUDE_f_odd_iff_a_b_zero_l2930_293079

/-- The function f defined with parameters a and b -/
def f (a b : ℝ) : ℝ → ℝ := λ x ↦ x * |x + a| + b

/-- f is an odd function if and only if a^2 + b^2 = 0 -/
theorem f_odd_iff_a_b_zero (a b : ℝ) :
  (∀ x, f a b (-x) = -(f a b x)) ↔ a^2 + b^2 = 0 := by
  sorry

end NUMINAMATH_CALUDE_f_odd_iff_a_b_zero_l2930_293079


namespace NUMINAMATH_CALUDE_jack_initial_marbles_l2930_293026

/-- The number of marbles Jack shared with Rebecca -/
def shared_marbles : ℕ := 33

/-- The number of marbles Jack had after sharing -/
def remaining_marbles : ℕ := 29

/-- The initial number of marbles Jack had -/
def initial_marbles : ℕ := shared_marbles + remaining_marbles

theorem jack_initial_marbles : initial_marbles = 62 := by
  sorry

end NUMINAMATH_CALUDE_jack_initial_marbles_l2930_293026


namespace NUMINAMATH_CALUDE_leg_head_difference_l2930_293001

/-- Represents the number of ducks in the group -/
def num_ducks : ℕ := sorry

/-- Represents the number of cows in the group -/
def num_cows : ℕ := 13

/-- Calculates the total number of legs in the group -/
def total_legs : ℕ := 2 * num_ducks + 4 * num_cows

/-- Calculates the total number of heads in the group -/
def total_heads : ℕ := num_ducks + num_cows

/-- States that the difference between total legs and thrice the total heads is 13 -/
theorem leg_head_difference : total_legs - 3 * total_heads = 13 := by sorry

end NUMINAMATH_CALUDE_leg_head_difference_l2930_293001


namespace NUMINAMATH_CALUDE_license_plate_count_l2930_293043

/-- The number of letters in the alphabet --/
def alphabet_size : ℕ := 26

/-- The number of letters in a license plate --/
def letters_count : ℕ := 4

/-- The number of digits in a license plate --/
def digits_count : ℕ := 3

/-- The number of available digits (0-9) --/
def available_digits : ℕ := 10

/-- Calculates the number of license plate combinations --/
def license_plate_combinations : ℕ :=
  alphabet_size *
  (Nat.choose (alphabet_size - 1) 2) *
  (Nat.choose letters_count 2) *
  2 *
  available_digits *
  (available_digits - 1) *
  (available_digits - 2)

theorem license_plate_count :
  license_plate_combinations = 67392000 := by
  sorry

end NUMINAMATH_CALUDE_license_plate_count_l2930_293043


namespace NUMINAMATH_CALUDE_blue_fish_with_spots_l2930_293012

theorem blue_fish_with_spots (total_fish : ℕ) (blue_fish : ℕ) (spotted_blue_fish : ℕ) 
  (h1 : total_fish = 60)
  (h2 : blue_fish = total_fish / 3)
  (h3 : spotted_blue_fish = blue_fish / 2) :
  spotted_blue_fish = 10 := by
  sorry

end NUMINAMATH_CALUDE_blue_fish_with_spots_l2930_293012


namespace NUMINAMATH_CALUDE_expression_evaluation_l2930_293004

theorem expression_evaluation : 
  Real.sqrt (16 - 8 * Real.sqrt 3) + Real.sqrt (16 + 8 * Real.sqrt 3) + Real.sqrt 12 = 6 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l2930_293004


namespace NUMINAMATH_CALUDE_count_divisible_integers_l2930_293015

theorem count_divisible_integers : 
  ∃! (S : Finset ℕ), 
    (∀ m ∈ S, m > 0 ∧ (1806 : ℤ) ∣ (m^2 - 2)) ∧ 
    (∀ m : ℕ, m > 0 ∧ (1806 : ℤ) ∣ (m^2 - 2) → m ∈ S) ∧
    Finset.card S = 2 :=
by sorry

end NUMINAMATH_CALUDE_count_divisible_integers_l2930_293015


namespace NUMINAMATH_CALUDE_equation_root_sum_l2930_293041

def equation (x : ℝ) : Prop :=
  1/x + 1/(x + 3) - 1/(x + 6) - 1/(x + 9) - 1/(x + 12) - 1/(x + 15) + 1/(x + 18) + 1/(x + 21) = 0

def is_root (x a b c d : ℝ) : Prop :=
  (x = -a + Real.sqrt (b + c * Real.sqrt d) ∨ x = -a - Real.sqrt (b + c * Real.sqrt d) ∨
   x = -a + Real.sqrt (b - c * Real.sqrt d) ∨ x = -a - Real.sqrt (b - c * Real.sqrt d)) ∧
  ¬ ∃ (p : ℕ), Prime p ∧ (p^2 : ℝ) ∣ d

theorem equation_root_sum (a b c d : ℝ) :
  (∃ x : ℝ, equation x ∧ is_root x a b c d) →
  a + b + c + d = 57.5 := by
  sorry

end NUMINAMATH_CALUDE_equation_root_sum_l2930_293041


namespace NUMINAMATH_CALUDE_no_sum_of_squares_representation_l2930_293053

theorem no_sum_of_squares_representation : ¬∃ (n : ℕ), ∃ (x y : ℕ+), 
  2 * n * (n + 1) * (n + 2) * (n + 3) + 12 = x^2 + y^2 := by
  sorry

end NUMINAMATH_CALUDE_no_sum_of_squares_representation_l2930_293053


namespace NUMINAMATH_CALUDE_magnitude_of_complex_product_l2930_293077

theorem magnitude_of_complex_product : 
  Complex.abs ((7 - 4*I) * (3 + 10*I)) = Real.sqrt 7085 := by
  sorry

end NUMINAMATH_CALUDE_magnitude_of_complex_product_l2930_293077


namespace NUMINAMATH_CALUDE_johns_journey_length_l2930_293067

theorem johns_journey_length :
  ∀ (total_length : ℝ),
  (total_length / 4 : ℝ) + 30 + (1/3 : ℝ) * (total_length - total_length / 4 - 30) = total_length →
  total_length = 160 := by
sorry

end NUMINAMATH_CALUDE_johns_journey_length_l2930_293067
