import Mathlib

namespace NUMINAMATH_CALUDE_distance_between_points_l1469_146953

/-- The distance between points (1, 3) and (-5, 7) is 2√13. -/
theorem distance_between_points : 
  let p1 : ℝ × ℝ := (1, 3)
  let p2 : ℝ × ℝ := (-5, 7)
  Real.sqrt ((p2.1 - p1.1)^2 + (p2.2 - p1.2)^2) = 2 * Real.sqrt 13 := by
  sorry

end NUMINAMATH_CALUDE_distance_between_points_l1469_146953


namespace NUMINAMATH_CALUDE_modulo_equivalence_l1469_146980

theorem modulo_equivalence : ∃ n : ℕ, 173 * 927 ≡ n [ZMOD 50] ∧ n < 50 ∧ n = 21 := by
  sorry

end NUMINAMATH_CALUDE_modulo_equivalence_l1469_146980


namespace NUMINAMATH_CALUDE_negation_equivalence_l1469_146909

theorem negation_equivalence : 
  (¬ ∃ x₀ : ℝ, x₀ > 0 ∧ Real.log x₀ = x₀ - 1) ↔ 
  (∀ x : ℝ, x > 0 → Real.log x ≠ x - 1) :=
by sorry

end NUMINAMATH_CALUDE_negation_equivalence_l1469_146909


namespace NUMINAMATH_CALUDE_not_power_of_two_l1469_146939

def lower_bound : Nat := 11111
def upper_bound : Nat := 99999

theorem not_power_of_two : ∃ (n : Nat), 
  (n = (upper_bound - lower_bound + 1) * upper_bound) ∧ 
  (n % 9 = 0) ∧
  (∀ (m : Nat), (2^m ≠ n)) := by
  sorry

end NUMINAMATH_CALUDE_not_power_of_two_l1469_146939


namespace NUMINAMATH_CALUDE_student_representatives_distribution_l1469_146947

theorem student_representatives_distribution (n m : ℕ) : 
  n = 6 ∧ m = 4 → (Nat.choose (n + m - 2) (m - 1) = Nat.choose 5 3) := by
  sorry

end NUMINAMATH_CALUDE_student_representatives_distribution_l1469_146947


namespace NUMINAMATH_CALUDE_min_value_theorem_l1469_146938

theorem min_value_theorem (a b c d : ℝ) 
  (h1 : 0 ≤ a ∧ a < 2^(1/4))
  (h2 : 0 ≤ b ∧ b < 2^(1/4))
  (h3 : 0 ≤ c ∧ c < 2^(1/4))
  (h4 : 0 ≤ d ∧ d < 2^(1/4))
  (h5 : a^3 + b^3 + c^3 + d^3 = 2) :
  (a / Real.sqrt (2 - a^4)) + (b / Real.sqrt (2 - b^4)) + 
  (c / Real.sqrt (2 - c^4)) + (d / Real.sqrt (2 - d^4)) ≥ 2 := by
  sorry

end NUMINAMATH_CALUDE_min_value_theorem_l1469_146938


namespace NUMINAMATH_CALUDE_parallelogram_area_l1469_146902

/-- The area of a parallelogram with base 12 feet and height 5 feet is 60 square feet. -/
theorem parallelogram_area (base height : ℝ) (h1 : base = 12) (h2 : height = 5) :
  base * height = 60 := by
  sorry

end NUMINAMATH_CALUDE_parallelogram_area_l1469_146902


namespace NUMINAMATH_CALUDE_foci_coordinates_l1469_146973

/-- The curve equation -/
def curve (a x y : ℝ) : Prop :=
  x^2 / (a - 4) + y^2 / (a + 5) = 1

/-- The foci are fixed points -/
def fixed_foci (a : ℝ) : Prop :=
  ∃ x y : ℝ, ∀ b : ℝ, curve b x y → (x, y) = (0, 3) ∨ (x, y) = (0, -3)

/-- Theorem: If the foci of the curve are fixed points, then their coordinates are (0, ±3) -/
theorem foci_coordinates (a : ℝ) :
  fixed_foci a → ∃ x y : ℝ, curve a x y ∧ ((x, y) = (0, 3) ∨ (x, y) = (0, -3)) :=
sorry

end NUMINAMATH_CALUDE_foci_coordinates_l1469_146973


namespace NUMINAMATH_CALUDE_lock_combinations_l1469_146956

def digits : ℕ := 10
def dials : ℕ := 4
def even_digits : ℕ := 5

theorem lock_combinations : 
  (even_digits) * (digits - 1) * (digits - 2) * (digits - 3) = 2520 :=
by sorry

end NUMINAMATH_CALUDE_lock_combinations_l1469_146956


namespace NUMINAMATH_CALUDE_distance_QR_l1469_146908

-- Define the triangle
def Triangle (D E F : ℝ × ℝ) : Prop :=
  let de := Real.sqrt ((E.1 - D.1)^2 + (E.2 - D.2)^2)
  let ef := Real.sqrt ((F.1 - E.1)^2 + (F.2 - E.2)^2)
  let df := Real.sqrt ((F.1 - D.1)^2 + (F.2 - D.2)^2)
  de = 5 ∧ ef = 12 ∧ df = 13

-- Define the circles
def Circle (Q : ℝ × ℝ) (D E F : ℝ × ℝ) : Prop :=
  let qe := Real.sqrt ((E.1 - Q.1)^2 + (E.2 - Q.2)^2)
  let qd := Real.sqrt ((D.1 - Q.1)^2 + (D.2 - Q.2)^2)
  qe = qd ∧ (E.1 - Q.1) * (F.1 - E.1) + (E.2 - Q.2) * (F.2 - E.2) = 0

def Circle' (R : ℝ × ℝ) (D E F : ℝ × ℝ) : Prop :=
  let rd := Real.sqrt ((D.1 - R.1)^2 + (D.2 - R.2)^2)
  let re := Real.sqrt ((E.1 - R.1)^2 + (E.2 - R.2)^2)
  rd = re ∧ (D.1 - R.1) * (F.1 - D.1) + (D.2 - R.2) * (F.2 - D.2) = 0

-- Theorem statement
theorem distance_QR (D E F Q R : ℝ × ℝ) :
  Triangle D E F →
  Circle Q D E F →
  Circle' R D E F →
  Real.sqrt ((R.1 - Q.1)^2 + (R.2 - Q.2)^2) = 25/12 := by
  sorry

end NUMINAMATH_CALUDE_distance_QR_l1469_146908


namespace NUMINAMATH_CALUDE_sams_initial_money_l1469_146983

/-- Calculates the initial amount of money given the number of books bought, 
    cost per book, and money left after purchase. -/
def initial_money (num_books : ℕ) (cost_per_book : ℕ) (money_left : ℕ) : ℕ :=
  num_books * cost_per_book + money_left

/-- Theorem stating that given the specific conditions of Sam's purchase,
    his initial amount of money was 79 dollars. -/
theorem sams_initial_money : 
  initial_money 9 7 16 = 79 := by
  sorry

#eval initial_money 9 7 16

end NUMINAMATH_CALUDE_sams_initial_money_l1469_146983


namespace NUMINAMATH_CALUDE_consecutive_odd_product_l1469_146900

theorem consecutive_odd_product (n : ℕ) : (2*n - 1) * (2*n + 1) = (2*n)^2 - 1 := by
  sorry

end NUMINAMATH_CALUDE_consecutive_odd_product_l1469_146900


namespace NUMINAMATH_CALUDE_smallest_value_of_root_products_l1469_146919

def g (x : ℝ) : ℝ := x^4 + 16*x^3 + 69*x^2 + 112*x + 64

theorem smallest_value_of_root_products (w₁ w₂ w₃ w₄ : ℝ) 
  (h₁ : g w₁ = 0) (h₂ : g w₂ = 0) (h₃ : g w₃ = 0) (h₄ : g w₄ = 0) :
  ∃ (min : ℝ), min = 8 ∧ ∀ (p : ℝ), p = |w₁*w₂ + w₃*w₄| → p ≥ min :=
by sorry

end NUMINAMATH_CALUDE_smallest_value_of_root_products_l1469_146919


namespace NUMINAMATH_CALUDE_triangle_inequality_l1469_146989

theorem triangle_inequality (a b c : ℝ) 
  (pos_a : a > 0) (pos_b : b > 0) (pos_c : c > 0)
  (h : 5 * a * b * c > a^3 + b^3 + c^3) :
  a + b > c ∧ a + c > b ∧ b + c > a := by
  sorry

end NUMINAMATH_CALUDE_triangle_inequality_l1469_146989


namespace NUMINAMATH_CALUDE_sqrt_450_simplification_l1469_146998

theorem sqrt_450_simplification : Real.sqrt 450 = 15 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_450_simplification_l1469_146998


namespace NUMINAMATH_CALUDE_fencing_championship_medals_l1469_146951

/-- The number of ways to select first and second place winners from n fencers -/
def awardMedals (n : ℕ) : ℕ := n * (n - 1)

/-- Theorem: There are 72 ways to award first and second place medals among 9 fencers -/
theorem fencing_championship_medals :
  awardMedals 9 = 72 := by
  sorry

end NUMINAMATH_CALUDE_fencing_championship_medals_l1469_146951


namespace NUMINAMATH_CALUDE_solve_age_problem_l1469_146999

def age_problem (a b : ℕ) : Prop :=
  (a + 10 = 2 * (b - 10)) ∧ (a = b + 12)

theorem solve_age_problem :
  ∀ a b : ℕ, age_problem a b → b = 42 :=
by
  sorry

end NUMINAMATH_CALUDE_solve_age_problem_l1469_146999


namespace NUMINAMATH_CALUDE_complex_root_problem_l1469_146946

theorem complex_root_problem (a b c : ℂ) (h_real : b.im = 0) 
  (h_sum : a + b + c = 4)
  (h_prod_sum : a * b + b * c + c * a = 5)
  (h_prod : a * b * c = 6) :
  b = 1 := by sorry

end NUMINAMATH_CALUDE_complex_root_problem_l1469_146946


namespace NUMINAMATH_CALUDE_initial_ducks_l1469_146949

theorem initial_ducks (initial additional total : ℕ) 
  (h1 : additional = 20)
  (h2 : total = 33)
  (h3 : initial + additional = total) : 
  initial = 13 := by
sorry

end NUMINAMATH_CALUDE_initial_ducks_l1469_146949


namespace NUMINAMATH_CALUDE_toy_cost_price_l1469_146916

/-- The cost price of a toy -/
def cost_price : ℕ := sorry

/-- The number of toys sold -/
def toys_sold : ℕ := 18

/-- The total selling price of all toys -/
def total_selling_price : ℕ := 16800

/-- The number of toys whose cost price equals the gain -/
def toys_equal_to_gain : ℕ := 3

theorem toy_cost_price : 
  cost_price * (toys_sold + toys_equal_to_gain) = total_selling_price ∧ 
  cost_price = 800 := by sorry

end NUMINAMATH_CALUDE_toy_cost_price_l1469_146916


namespace NUMINAMATH_CALUDE_total_profit_calculation_l1469_146975

-- Define the investments and c's profit share
def investment_a : ℕ := 5000
def investment_b : ℕ := 15000
def investment_c : ℕ := 30000
def c_profit_share : ℕ := 3000

-- Theorem statement
theorem total_profit_calculation :
  let total_investment := investment_a + investment_b + investment_c
  let profit_ratio_c := investment_c / total_investment
  let total_profit := c_profit_share / profit_ratio_c
  total_profit = 5000 := by sorry

end NUMINAMATH_CALUDE_total_profit_calculation_l1469_146975


namespace NUMINAMATH_CALUDE_line_tangent_to_circle_l1469_146981

/-- The line equation √3x - y + m = 0 is tangent to the circle x² + y² - 2x - 2 = 0 
    if and only if m = √3 or m = -3√3 -/
theorem line_tangent_to_circle (m : ℝ) : 
  (∀ x y : ℝ, (Real.sqrt 3 * x - y + m = 0) → 
   (x^2 + y^2 - 2*x - 2 = 0) → 
   (∀ ε > 0, ∃ x' y' : ℝ, 
     x' ≠ x ∧ y' ≠ y ∧ 
     (Real.sqrt 3 * x' - y' + m = 0) ∧ 
     (x'^2 + y'^2 - 2*x' - 2 ≠ 0) ∧
     ((x' - x)^2 + (y' - y)^2 < ε^2))) ↔ 
  (m = Real.sqrt 3 ∨ m = -3 * Real.sqrt 3) :=
sorry

end NUMINAMATH_CALUDE_line_tangent_to_circle_l1469_146981


namespace NUMINAMATH_CALUDE_school_boys_count_l1469_146907

theorem school_boys_count :
  ∀ (boys girls : ℕ),
  (boys : ℚ) / (girls : ℚ) = 5 / 13 →
  girls = boys + 64 →
  boys = 40 := by
sorry

end NUMINAMATH_CALUDE_school_boys_count_l1469_146907


namespace NUMINAMATH_CALUDE_inequality_equivalence_l1469_146971

theorem inequality_equivalence (x : ℝ) : (x - 2) / 3 ≤ x ↔ x ≥ -1 := by sorry

end NUMINAMATH_CALUDE_inequality_equivalence_l1469_146971


namespace NUMINAMATH_CALUDE_ben_pea_picking_l1469_146967

/-- Ben's pea-picking problem -/
theorem ben_pea_picking (P : ℕ) : ∃ (T : ℚ), T = P / 8 :=
  by
  -- Define Ben's picking rates
  have rate1 : (56 : ℚ) / 7 = 8 := by sorry
  have rate2 : (72 : ℚ) / 9 = 8 := by sorry

  -- Prove the theorem
  sorry

end NUMINAMATH_CALUDE_ben_pea_picking_l1469_146967


namespace NUMINAMATH_CALUDE_discount_calculation_l1469_146992

theorem discount_calculation (list_price : ℝ) (final_price : ℝ) (first_discount : ℝ) :
  list_price = 65 →
  final_price = 57.33 →
  first_discount = 10 →
  ∃ second_discount : ℝ,
    second_discount = 2 ∧
    final_price = list_price * (1 - first_discount / 100) * (1 - second_discount / 100) :=
by sorry

end NUMINAMATH_CALUDE_discount_calculation_l1469_146992


namespace NUMINAMATH_CALUDE_black_card_fraction_l1469_146941

theorem black_card_fraction (total : ℕ) (red_fraction : ℚ) (green : ℕ) : 
  total = 120 → 
  red_fraction = 2 / 5 → 
  green = 32 → 
  (5 : ℚ) / 9 = (total - (red_fraction * total) - green) / (total - (red_fraction * total)) := by
  sorry

end NUMINAMATH_CALUDE_black_card_fraction_l1469_146941


namespace NUMINAMATH_CALUDE_train_length_l1469_146944

/-- Given a train with speed 72 km/hr crossing a 250 m long platform in 26 seconds,
    prove that the length of the train is 270 meters. -/
theorem train_length (train_speed : ℝ) (platform_length : ℝ) (crossing_time : ℝ) :
  train_speed = 72 * (1000 / 3600) → 
  platform_length = 250 →
  crossing_time = 26 →
  (train_speed * crossing_time) - platform_length = 270 := by
  sorry

#eval (72 * (1000 / 3600) * 26) - 250  -- Should output 270

end NUMINAMATH_CALUDE_train_length_l1469_146944


namespace NUMINAMATH_CALUDE_parity_of_p_and_q_l1469_146961

theorem parity_of_p_and_q (m n p q : ℤ) :
  Odd m →
  Even n →
  p - 1998 * q = n →
  1999 * p + 3 * q = m →
  (Even p ∧ Odd q) :=
by sorry

end NUMINAMATH_CALUDE_parity_of_p_and_q_l1469_146961


namespace NUMINAMATH_CALUDE_total_cost_of_two_items_l1469_146950

/-- The total cost of two items is the sum of their individual costs -/
theorem total_cost_of_two_items (yoyo_cost whistle_cost : ℕ) :
  yoyo_cost = 24 → whistle_cost = 14 →
  yoyo_cost + whistle_cost = 38 := by
  sorry

end NUMINAMATH_CALUDE_total_cost_of_two_items_l1469_146950


namespace NUMINAMATH_CALUDE_parabola_vertex_in_second_quadrant_l1469_146982

/-- Represents a parabola of the form y = 2(x-m-1)^2 + 2m + 4 -/
def Parabola (m : ℝ) := λ x : ℝ => 2 * (x - m - 1)^2 + 2 * m + 4

/-- The x-coordinate of the vertex of the parabola -/
def vertex_x (m : ℝ) : ℝ := m + 1

/-- The y-coordinate of the vertex of the parabola -/
def vertex_y (m : ℝ) : ℝ := 2 * m + 4

/-- Predicate for a point being in the second quadrant -/
def in_second_quadrant (x y : ℝ) : Prop := x < 0 ∧ y > 0

theorem parabola_vertex_in_second_quadrant (m : ℝ) :
  in_second_quadrant (vertex_x m) (vertex_y m) ↔ -2 < m ∧ m < -1 :=
sorry

end NUMINAMATH_CALUDE_parabola_vertex_in_second_quadrant_l1469_146982


namespace NUMINAMATH_CALUDE_pages_revised_once_is_35_l1469_146991

/-- Represents the manuscript typing problem -/
structure ManuscriptTyping where
  total_pages : ℕ
  pages_revised_twice : ℕ
  first_typing_cost : ℕ
  revision_cost : ℕ
  total_cost : ℕ

/-- Calculates the number of pages revised once -/
def pages_revised_once (m : ManuscriptTyping) : ℕ :=
  ((m.total_cost - m.first_typing_cost * m.total_pages - 
    m.revision_cost * m.pages_revised_twice * 2) / m.revision_cost)

/-- Theorem stating that the number of pages revised once is 35 -/
theorem pages_revised_once_is_35 (m : ManuscriptTyping) 
  (h1 : m.total_pages = 100)
  (h2 : m.pages_revised_twice = 15)
  (h3 : m.first_typing_cost = 6)
  (h4 : m.revision_cost = 4)
  (h5 : m.total_cost = 860) :
  pages_revised_once m = 35 := by
  sorry

#eval pages_revised_once ⟨100, 15, 6, 4, 860⟩

end NUMINAMATH_CALUDE_pages_revised_once_is_35_l1469_146991


namespace NUMINAMATH_CALUDE_triangle_inequality_l1469_146915

theorem triangle_inequality (a b c : ℝ) (h : 0 < a ∧ 0 < b ∧ 0 < c ∧ a + b > c ∧ b + c > a ∧ c + a > b) :
  (a + b - c) * (a - b + c) * (-a + b + c) ≤ a * b * c := by
  sorry

end NUMINAMATH_CALUDE_triangle_inequality_l1469_146915


namespace NUMINAMATH_CALUDE_solve_m_l1469_146964

def g (n : Int) : Int :=
  if n % 2 = 1 then n + 5 else n / 2

theorem solve_m (m : Int) (h1 : m % 2 = 1) (h2 : g (g (g m)) = 15) : m = 55 := by
  sorry

end NUMINAMATH_CALUDE_solve_m_l1469_146964


namespace NUMINAMATH_CALUDE_quadratic_factor_problem_l1469_146911

theorem quadratic_factor_problem (d e : ℤ) :
  let q : ℝ → ℝ := fun x ↦ x^2 + d*x + e
  (∃ r : ℝ → ℝ, (fun x ↦ x^4 + x^3 + 8*x^2 + 7*x + 18) = q * r) ∧
  (∃ s : ℝ → ℝ, (fun x ↦ 2*x^4 + 3*x^3 + 9*x^2 + 8*x + 20) = q * s) →
  q 1 = -6 := by
sorry

end NUMINAMATH_CALUDE_quadratic_factor_problem_l1469_146911


namespace NUMINAMATH_CALUDE_sum_of_parts_for_specific_complex_l1469_146914

theorem sum_of_parts_for_specific_complex (z : ℂ) (h : z = 1 - Complex.I) : 
  z.re + z.im = 0 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_parts_for_specific_complex_l1469_146914


namespace NUMINAMATH_CALUDE_production_problem_l1469_146924

/-- Calculates the production for today given the number of past days, 
    past average production, and new average including today's production. -/
def todaysProduction (n : ℕ) (pastAvg newAvg : ℚ) : ℚ :=
  (n + 1) * newAvg - n * pastAvg

/-- Proves that given the conditions, today's production is 105 units. -/
theorem production_problem (n : ℕ) (pastAvg newAvg : ℚ) 
  (h1 : n = 10)
  (h2 : pastAvg = 50)
  (h3 : newAvg = 55) :
  todaysProduction n pastAvg newAvg = 105 := by
  sorry

#eval todaysProduction 10 50 55

end NUMINAMATH_CALUDE_production_problem_l1469_146924


namespace NUMINAMATH_CALUDE_functional_equation_solution_l1469_146955

/-- A function satisfying the given functional equation -/
def SatisfiesFunctionalEq (f : ℝ → ℝ) : Prop :=
  ∀ (x y : ℝ), f (f x + 9 * y) = f y + 9 * x + 24 * y

/-- The main theorem stating that any function satisfying the functional equation must be f(x) = 3x -/
theorem functional_equation_solution :
  ∀ (f : ℝ → ℝ), SatisfiesFunctionalEq f → (∀ x, f x = 3 * x) :=
by sorry

end NUMINAMATH_CALUDE_functional_equation_solution_l1469_146955


namespace NUMINAMATH_CALUDE_largest_n_for_rational_sum_of_roots_l1469_146920

theorem largest_n_for_rational_sum_of_roots : 
  ∀ n : ℕ, n > 2501 → ¬(∃ (q : ℚ), q = Real.sqrt (n - 100) + Real.sqrt (n + 100)) := by
  sorry

end NUMINAMATH_CALUDE_largest_n_for_rational_sum_of_roots_l1469_146920


namespace NUMINAMATH_CALUDE_line_intersects_plane_l1469_146976

theorem line_intersects_plane (α : Subspace ℝ (Fin 3 → ℝ)) 
  (a b u : Fin 3 → ℝ) 
  (ha : a ∈ α) (hb : b ∈ α)
  (ha_def : a = ![1, 1/2, 3])
  (hb_def : b = ![1/2, 1, 1])
  (hu_def : u = ![1/2, 0, 1]) :
  ∃ (t : ℝ), (t • u) ∈ α ∧ t • u ≠ 0 := by
  sorry

end NUMINAMATH_CALUDE_line_intersects_plane_l1469_146976


namespace NUMINAMATH_CALUDE_angle_D_is_100_l1469_146937

-- Define a cyclic quadrilateral
structure CyclicQuadrilateral where
  A : ℝ
  B : ℝ
  C : ℝ
  D : ℝ
  sum_360 : A + B + C + D = 360
  ratio_abc : ∃ (x : ℝ), A = 3*x ∧ B = 4*x ∧ C = 6*x

-- Theorem statement
theorem angle_D_is_100 (q : CyclicQuadrilateral) : q.D = 100 := by
  sorry

end NUMINAMATH_CALUDE_angle_D_is_100_l1469_146937


namespace NUMINAMATH_CALUDE_halley_21st_century_appearance_l1469_146906

/-- Represents the year of Halley's Comet's appearance -/
def halley_appearance (n : ℕ) : ℕ := 1682 + 76 * n

/-- Predicate to check if a year is in the 21st century -/
def is_21st_century (year : ℕ) : Prop := 2001 ≤ year ∧ year ≤ 2100

theorem halley_21st_century_appearance :
  ∃ n : ℕ, is_21st_century (halley_appearance n) ∧ halley_appearance n = 2062 :=
sorry

end NUMINAMATH_CALUDE_halley_21st_century_appearance_l1469_146906


namespace NUMINAMATH_CALUDE_dividend_calculation_l1469_146993

theorem dividend_calculation (divisor quotient remainder : ℕ) 
  (h1 : divisor = 14)
  (h2 : quotient = 9)
  (h3 : remainder = 5) :
  divisor * quotient + remainder = 131 := by
  sorry

end NUMINAMATH_CALUDE_dividend_calculation_l1469_146993


namespace NUMINAMATH_CALUDE_fibonacci_gcd_2002_1998_l1469_146925

def fibonacci : ℕ → ℕ
  | 0 => 0
  | 1 => 1
  | 2 => 1
  | (n + 3) => fibonacci (n + 2) + fibonacci (n + 1)

theorem fibonacci_gcd_2002_1998 : Nat.gcd (fibonacci 2002) (fibonacci 1998) = 1 := by
  sorry

end NUMINAMATH_CALUDE_fibonacci_gcd_2002_1998_l1469_146925


namespace NUMINAMATH_CALUDE_quadratic_inequality_range_l1469_146928

theorem quadratic_inequality_range (m : ℝ) : 
  (∀ x : ℝ, x > 1 ∧ x < 2 → x^2 + m*x + 4 < 0) ↔ m ≤ -5 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_range_l1469_146928


namespace NUMINAMATH_CALUDE_bells_toll_together_l1469_146963

theorem bells_toll_together (a b c d : ℕ) 
  (ha : a = 9) (hb : b = 10) (hc : c = 14) (hd : d = 18) : 
  Nat.lcm a (Nat.lcm b (Nat.lcm c d)) = 630 := by
  sorry

end NUMINAMATH_CALUDE_bells_toll_together_l1469_146963


namespace NUMINAMATH_CALUDE_function_symmetry_l1469_146933

/-- Given a function f and a real number a, 
    if f(x) = x³cos(x) + 1 and f(a) = 11, then f(-a) = -9 -/
theorem function_symmetry (f : ℝ → ℝ) (a : ℝ) 
    (h1 : ∀ x, f x = x^3 * Real.cos x + 1) 
    (h2 : f a = 11) : 
  f (-a) = -9 := by
sorry

end NUMINAMATH_CALUDE_function_symmetry_l1469_146933


namespace NUMINAMATH_CALUDE_probability_not_adjacent_l1469_146930

def total_chairs : ℕ := 10
def broken_chair : ℕ := 5
def available_chairs : ℕ := total_chairs - 1

theorem probability_not_adjacent : 
  let total_ways := available_chairs.choose 2
  let adjacent_pairs := 6
  (1 - (adjacent_pairs : ℚ) / total_ways) = 5/6 := by sorry

end NUMINAMATH_CALUDE_probability_not_adjacent_l1469_146930


namespace NUMINAMATH_CALUDE_salary_calculation_l1469_146979

theorem salary_calculation (salary : ℝ) : 
  salary * (1/5 + 1/10 + 3/5) + 16000 = salary → salary = 160000 := by
  sorry

end NUMINAMATH_CALUDE_salary_calculation_l1469_146979


namespace NUMINAMATH_CALUDE_matt_twice_james_age_l1469_146932

/-- 
Given:
- James turned 27 three years ago
- Matt is now 65 years old

Prove that in 5 years, Matt will be twice James' age.
-/
theorem matt_twice_james_age (james_age_three_years_ago : ℕ) (matt_current_age : ℕ) :
  james_age_three_years_ago = 27 →
  matt_current_age = 65 →
  ∃ (years_from_now : ℕ), 
    years_from_now = 5 ∧
    matt_current_age + years_from_now = 2 * (james_age_three_years_ago + 3 + years_from_now) :=
by sorry

end NUMINAMATH_CALUDE_matt_twice_james_age_l1469_146932


namespace NUMINAMATH_CALUDE_second_chapter_pages_l1469_146945

/-- A book with three chapters -/
structure Book where
  chapter1 : ℕ
  chapter2 : ℕ
  chapter3 : ℕ

/-- The book satisfies the given conditions -/
def satisfiesConditions (b : Book) : Prop :=
  b.chapter1 = 35 ∧ b.chapter3 = 3 ∧ b.chapter2 = b.chapter3 + 15

theorem second_chapter_pages (b : Book) (h : satisfiesConditions b) : b.chapter2 = 18 := by
  sorry

end NUMINAMATH_CALUDE_second_chapter_pages_l1469_146945


namespace NUMINAMATH_CALUDE_train_passing_jogger_time_l1469_146988

/-- Time taken for a train to pass a jogger given their speeds and initial positions -/
theorem train_passing_jogger_time (jogger_speed : ℝ) (train_speed : ℝ) (initial_distance : ℝ) (train_length : ℝ) : 
  jogger_speed = 9 * (1000 / 3600) →
  train_speed = 45 * (1000 / 3600) →
  initial_distance = 190 →
  train_length = 120 →
  (initial_distance + train_length) / (train_speed - jogger_speed) = 31 := by
  sorry

#check train_passing_jogger_time

end NUMINAMATH_CALUDE_train_passing_jogger_time_l1469_146988


namespace NUMINAMATH_CALUDE_average_price_is_52_cents_l1469_146942

/-- Represents the fruit selection problem --/
structure FruitSelection where
  apple_price : ℚ
  orange_price : ℚ
  total_fruits : ℕ
  initial_avg_price : ℚ
  oranges_returned : ℕ

/-- Calculates the average price of fruits kept --/
def average_price_kept (fs : FruitSelection) : ℚ :=
  let apples := fs.total_fruits - (fs.initial_avg_price * fs.total_fruits - fs.apple_price * fs.total_fruits) / (fs.orange_price - fs.apple_price)
  let oranges := fs.total_fruits - apples
  let kept_oranges := oranges - fs.oranges_returned
  let total_kept := apples + kept_oranges
  (fs.apple_price * apples + fs.orange_price * kept_oranges) / total_kept

/-- Theorem stating that the average price of fruits kept is 52 cents --/
theorem average_price_is_52_cents (fs : FruitSelection) 
    (h1 : fs.apple_price = 40/100)
    (h2 : fs.orange_price = 60/100)
    (h3 : fs.total_fruits = 30)
    (h4 : fs.initial_avg_price = 56/100)
    (h5 : fs.oranges_returned = 15) :
  average_price_kept fs = 52/100 := by
  sorry

#eval average_price_kept {
  apple_price := 40/100,
  orange_price := 60/100,
  total_fruits := 30,
  initial_avg_price := 56/100,
  oranges_returned := 15
}

end NUMINAMATH_CALUDE_average_price_is_52_cents_l1469_146942


namespace NUMINAMATH_CALUDE_second_part_multiplier_l1469_146960

theorem second_part_multiplier (total : ℕ) (first_part : ℕ) (k : ℕ) : 
  total = 36 →
  first_part = 19 →
  8 * first_part + k * (total - first_part) = 203 →
  k = 3 := by sorry

end NUMINAMATH_CALUDE_second_part_multiplier_l1469_146960


namespace NUMINAMATH_CALUDE_cookies_divisible_by_bags_l1469_146966

/-- Represents the number of snack bags Destiny can make -/
def num_bags : ℕ := 6

/-- Represents the total number of chocolate candy bars -/
def total_candy_bars : ℕ := 18

/-- Represents the number of cookies Destiny received -/
def num_cookies : ℕ := sorry

/-- Theorem stating that the number of cookies is divisible by the number of bags -/
theorem cookies_divisible_by_bags : num_bags ∣ num_cookies := by sorry

end NUMINAMATH_CALUDE_cookies_divisible_by_bags_l1469_146966


namespace NUMINAMATH_CALUDE_circle_center_and_radius_l1469_146904

/-- The polar equation of a circle -/
def polar_equation (ρ θ : ℝ) : Prop := ρ = 2 * Real.cos θ

/-- The Cartesian equation of a circle with center (h, k) and radius r -/
def cartesian_equation (x y h k r : ℝ) : Prop :=
  (x - h)^2 + (y - k)^2 = r^2

/-- Theorem stating that the circle ρ = 2cosθ has center (1, 0) and radius 1 -/
theorem circle_center_and_radius :
  ∀ x y ρ θ : ℝ,
  polar_equation ρ θ →
  x = ρ * Real.cos θ →
  y = ρ * Real.sin θ →
  cartesian_equation x y 1 0 1 :=
by sorry

end NUMINAMATH_CALUDE_circle_center_and_radius_l1469_146904


namespace NUMINAMATH_CALUDE_mike_book_count_l1469_146917

/-- The number of books Tim has -/
def tim_books : ℕ := 22

/-- The total number of books Tim and Mike have together -/
def total_books : ℕ := 42

/-- The number of books Mike has -/
def mike_books : ℕ := total_books - tim_books

theorem mike_book_count : mike_books = 20 := by
  sorry

end NUMINAMATH_CALUDE_mike_book_count_l1469_146917


namespace NUMINAMATH_CALUDE_complex_square_value_l1469_146954

theorem complex_square_value (m n : ℝ) (h : m * (1 + Complex.I) = 1 + n * Complex.I) :
  ((m + n * Complex.I) / (m - n * Complex.I)) ^ 2 = -1 := by
  sorry

end NUMINAMATH_CALUDE_complex_square_value_l1469_146954


namespace NUMINAMATH_CALUDE_total_frisbee_distance_l1469_146962

/-- The distance Bess can throw the Frisbee -/
def bess_throw_distance : ℕ := 20

/-- The number of times Bess throws the Frisbee -/
def bess_throw_count : ℕ := 4

/-- The distance Holly can throw the Frisbee -/
def holly_throw_distance : ℕ := 8

/-- The number of times Holly throws the Frisbee -/
def holly_throw_count : ℕ := 5

/-- Theorem stating the total distance traveled by both Frisbees -/
theorem total_frisbee_distance : 
  2 * bess_throw_distance * bess_throw_count + holly_throw_distance * holly_throw_count = 200 := by
  sorry

end NUMINAMATH_CALUDE_total_frisbee_distance_l1469_146962


namespace NUMINAMATH_CALUDE_solve_baseball_card_problem_l1469_146965

def baseball_card_problem (patricia_money : ℕ) (card_price : ℕ) : Prop :=
  let lisa_money := 5 * patricia_money
  let charlotte_money := lisa_money / 2
  let james_money := 10 + charlotte_money + lisa_money
  let total_money := patricia_money + lisa_money + charlotte_money + james_money
  card_price - total_money = 144

theorem solve_baseball_card_problem :
  baseball_card_problem 6 250 := by
  sorry

end NUMINAMATH_CALUDE_solve_baseball_card_problem_l1469_146965


namespace NUMINAMATH_CALUDE_repeating_decimal_equals_fraction_l1469_146926

/-- The repeating decimal 0.3̄45 as a real number -/
def repeating_decimal : ℚ := 3/10 + 45/990

/-- The fraction 83/110 -/
def target_fraction : ℚ := 83/110

/-- Theorem stating that the repeating decimal 0.3̄45 is equal to the fraction 83/110 -/
theorem repeating_decimal_equals_fraction : repeating_decimal = target_fraction := by
  sorry

end NUMINAMATH_CALUDE_repeating_decimal_equals_fraction_l1469_146926


namespace NUMINAMATH_CALUDE_calculator_display_after_50_presses_l1469_146913

def calculator_function (x : ℚ) : ℚ := 1 / (1 - x)

def iterate_function (f : ℚ → ℚ) (x : ℚ) (n : ℕ) : ℚ :=
  match n with
  | 0 => x
  | n + 1 => f (iterate_function f x n)

theorem calculator_display_after_50_presses :
  iterate_function calculator_function (1/2) 50 = -1 := by
  sorry

end NUMINAMATH_CALUDE_calculator_display_after_50_presses_l1469_146913


namespace NUMINAMATH_CALUDE_matrix_A_properties_l1469_146934

/-- The line l: 2x - y = 3 -/
def line_l (x y : ℝ) : Prop := 2 * x - y = 3

/-- The transformation matrix A -/
def matrix_A : Matrix (Fin 2) (Fin 2) ℝ :=
  ![![-1, 1],
    ![-4, 3]]

/-- The inverse of matrix A -/
def matrix_A_inv : Matrix (Fin 2) (Fin 2) ℝ :=
  ![![3, -1],
    ![4, -1]]

/-- The transformation σ maps the line l onto itself -/
def transformation_preserves_line (A : Matrix (Fin 2) (Fin 2) ℝ) : Prop :=
  ∀ x y : ℝ, line_l x y → line_l (A 0 0 * x + A 0 1 * y) (A 1 0 * x + A 1 1 * y)

theorem matrix_A_properties :
  transformation_preserves_line matrix_A ∧
  matrix_A * matrix_A_inv = 1 ∧
  matrix_A_inv * matrix_A = 1 := by
  sorry

end NUMINAMATH_CALUDE_matrix_A_properties_l1469_146934


namespace NUMINAMATH_CALUDE_number_of_americans_l1469_146969

theorem number_of_americans (total : ℕ) (chinese : ℕ) (australians : ℕ) 
  (h1 : total = 49)
  (h2 : chinese = 22)
  (h3 : australians = 11) :
  total - chinese - australians = 16 := by
  sorry

end NUMINAMATH_CALUDE_number_of_americans_l1469_146969


namespace NUMINAMATH_CALUDE_consecutive_integers_sum_l1469_146952

theorem consecutive_integers_sum (n : ℤ) : n * (n + 1) = 20412 → n + (n + 1) = 287 := by
  sorry

end NUMINAMATH_CALUDE_consecutive_integers_sum_l1469_146952


namespace NUMINAMATH_CALUDE_classics_section_books_l1469_146986

/-- The number of classic authors in Jack's collection -/
def num_authors : ℕ := 6

/-- The number of books per author -/
def books_per_author : ℕ := 33

/-- The total number of books in the classics section -/
def total_books : ℕ := num_authors * books_per_author

theorem classics_section_books :
  total_books = 198 := by sorry

end NUMINAMATH_CALUDE_classics_section_books_l1469_146986


namespace NUMINAMATH_CALUDE_solution_product_l1469_146948

theorem solution_product (p q : ℝ) : 
  (p - 6) * (3 * p + 10) = p^2 - 19 * p + 50 →
  (q - 6) * (3 * q + 10) = q^2 - 19 * q + 50 →
  p ≠ q →
  (p + 2) * (q + 2) = 108 := by
sorry

end NUMINAMATH_CALUDE_solution_product_l1469_146948


namespace NUMINAMATH_CALUDE_range_of_t_l1469_146958

-- Define the function f
def f : ℝ → ℝ := sorry

-- Define the properties of f
axiom f_odd : ∀ x, f (-x) = -f x
axiom f_increasing : ∀ x y, x < y → x ∈ [-1, 1] → y ∈ [-1, 1] → f x < f y
axiom f_inequality : ∀ t, f (3*t) + f ((1/3) - t) > 0

-- Define the set of t that satisfies the conditions
def T : Set ℝ := {t | -1/6 < t ∧ t ≤ 1/3}

-- Theorem to prove
theorem range_of_t : ∀ t, (f (3*t) + f ((1/3) - t) > 0) ↔ t ∈ T := by sorry

end NUMINAMATH_CALUDE_range_of_t_l1469_146958


namespace NUMINAMATH_CALUDE_triple_345_is_right_triangle_l1469_146970

/-- A triple of natural numbers representing the sides of a triangle -/
structure TripleNat where
  a : ℕ
  b : ℕ
  c : ℕ

/-- Checks if a triple of natural numbers satisfies the Pythagorean theorem -/
def is_right_triangle (t : TripleNat) : Prop :=
  t.a ^ 2 + t.b ^ 2 = t.c ^ 2

/-- The specific triple (3, 4, 5) -/
def triple_345 : TripleNat :=
  { a := 3, b := 4, c := 5 }

/-- Theorem stating that (3, 4, 5) forms a right triangle -/
theorem triple_345_is_right_triangle : is_right_triangle triple_345 := by
  sorry

end NUMINAMATH_CALUDE_triple_345_is_right_triangle_l1469_146970


namespace NUMINAMATH_CALUDE_f_two_equals_two_thirds_l1469_146985

-- Define the function f
noncomputable def f : ℝ → ℝ := fun x ↦ x / (x + 1)

-- State the theorem
theorem f_two_equals_two_thirds :
  (∀ x : ℝ, x ≠ 0 → f (1 / x) = 1 / (x + 1)) →
  f 2 = 2 / 3 := by
  sorry

end NUMINAMATH_CALUDE_f_two_equals_two_thirds_l1469_146985


namespace NUMINAMATH_CALUDE_train_length_problem_l1469_146968

/-- Given two trains running in opposite directions, calculate the length of the second train. -/
theorem train_length_problem (length_A : ℝ) (speed_A speed_B : ℝ) (crossing_time : ℝ) :
  length_A = 230 →
  speed_A = 120 * 1000 / 3600 →
  speed_B = 80 * 1000 / 3600 →
  crossing_time = 9 →
  ∃ length_B : ℝ, abs (length_B - 269.95) < 0.01 ∧
    length_A + length_B = (speed_A + speed_B) * crossing_time :=
by sorry

end NUMINAMATH_CALUDE_train_length_problem_l1469_146968


namespace NUMINAMATH_CALUDE_not_all_exp_increasing_l1469_146959

-- Define the exponential function
noncomputable def exp (a : ℝ) (x : ℝ) : ℝ := a^x

-- State the theorem
theorem not_all_exp_increasing :
  ¬ (∀ (a : ℝ), a > 0 → (∀ (x y : ℝ), x < y → exp a x < exp a y)) :=
by sorry

end NUMINAMATH_CALUDE_not_all_exp_increasing_l1469_146959


namespace NUMINAMATH_CALUDE_line_passes_through_point_l1469_146997

/-- The value of b for which the line 2bx + (3b - 2)y = 5b + 6 passes through the point (6, -10) -/
theorem line_passes_through_point : 
  ∃ b : ℚ, b = 14/23 ∧ 2*b*6 + (3*b - 2)*(-10) = 5*b + 6 := by
  sorry

end NUMINAMATH_CALUDE_line_passes_through_point_l1469_146997


namespace NUMINAMATH_CALUDE_converse_and_inverse_false_l1469_146940

-- Define the universe of polygons
variable (Polygon : Type)

-- Define properties of polygons
variable (is_rhombus : Polygon → Prop)
variable (is_parallelogram : Polygon → Prop)

-- Original statement
axiom original_statement : ∀ p : Polygon, is_rhombus p → is_parallelogram p

-- Theorem to prove
theorem converse_and_inverse_false :
  (¬ ∀ p : Polygon, is_parallelogram p → is_rhombus p) ∧
  (¬ ∀ p : Polygon, ¬is_rhombus p → ¬is_parallelogram p) :=
by sorry

end NUMINAMATH_CALUDE_converse_and_inverse_false_l1469_146940


namespace NUMINAMATH_CALUDE_z_in_second_quadrant_l1469_146943

noncomputable def z : ℂ := Complex.exp (-4 * Complex.I)

theorem z_in_second_quadrant : 
  z.re < 0 ∧ z.im > 0 :=
sorry

end NUMINAMATH_CALUDE_z_in_second_quadrant_l1469_146943


namespace NUMINAMATH_CALUDE_white_area_is_69_l1469_146929

/-- Represents the dimensions of a rectangular sign -/
structure SignDimensions where
  width : ℕ
  height : ℕ

/-- Represents the area covered by a letter -/
structure LetterArea where
  area : ℕ

/-- Calculates the total area of the sign -/
def totalSignArea (dim : SignDimensions) : ℕ :=
  dim.width * dim.height

/-- Calculates the area covered by the letter M -/
def mArea : LetterArea :=
  { area := 2 * (6 * 1) + 2 * 2 }

/-- Calculates the area covered by the letter A -/
def aArea : LetterArea :=
  { area := 2 * 4 + 1 * 2 }

/-- Calculates the area covered by the letter T -/
def tArea : LetterArea :=
  { area := 1 * 4 + 6 * 1 }

/-- Calculates the area covered by the letter H -/
def hArea : LetterArea :=
  { area := 2 * (6 * 1) + 1 * 3 }

/-- Calculates the total area covered by all letters -/
def totalLettersArea : ℕ :=
  mArea.area + aArea.area + tArea.area + hArea.area

/-- The main theorem: proves that the white area of the sign is 69 square units -/
theorem white_area_is_69 (sign : SignDimensions) 
    (h1 : sign.width = 20) 
    (h2 : sign.height = 6) : 
    totalSignArea sign - totalLettersArea = 69 := by
  sorry


end NUMINAMATH_CALUDE_white_area_is_69_l1469_146929


namespace NUMINAMATH_CALUDE_remaining_land_to_clean_l1469_146977

theorem remaining_land_to_clean 
  (total_land : ℕ) 
  (lizzie_group : ℕ) 
  (other_group : ℕ) 
  (h1 : total_land = 900) 
  (h2 : lizzie_group = 250) 
  (h3 : other_group = 265) : 
  total_land - (lizzie_group + other_group) = 385 := by
sorry

end NUMINAMATH_CALUDE_remaining_land_to_clean_l1469_146977


namespace NUMINAMATH_CALUDE_books_loaned_out_l1469_146931

theorem books_loaned_out (initial_books : ℕ) (return_rate : ℚ) (final_books : ℕ) :
  initial_books = 75 →
  return_rate = 65 / 100 →
  final_books = 61 →
  (initial_books - final_books : ℚ) / (1 - return_rate) = 40 := by
sorry

end NUMINAMATH_CALUDE_books_loaned_out_l1469_146931


namespace NUMINAMATH_CALUDE_relationship_a_b_l1469_146957

theorem relationship_a_b (a b : ℝ) (ha : a^(1/5) > 1) (hb : 1 > b^(1/5)) : a > 1 ∧ 1 > b := by
  sorry

end NUMINAMATH_CALUDE_relationship_a_b_l1469_146957


namespace NUMINAMATH_CALUDE_fraction_simplification_l1469_146936

theorem fraction_simplification (a b c : ℝ) 
  (h1 : a + 2*b + 3*c ≠ 0)
  (h2 : a^2 + 9*c^2 - 4*b^2 + 6*a*c ≠ 0) :
  (a^2 + 4*b^2 - 9*c^2 + 4*a*b) / (a^2 + 9*c^2 - 4*b^2 + 6*a*c) = (a + 2*b - 3*c) / (a - 2*b + 3*c) :=
by sorry

end NUMINAMATH_CALUDE_fraction_simplification_l1469_146936


namespace NUMINAMATH_CALUDE_jersey_profit_calculation_l1469_146995

-- Define the given conditions
def tshirt_profit : ℝ := 25
def tshirts_sold : ℕ := 113
def jerseys_sold : ℕ := 78
def jersey_price_difference : ℝ := 90

-- Define the theorem to be proved
theorem jersey_profit_calculation :
  let jersey_profit := tshirt_profit + jersey_price_difference
  jersey_profit = 115 := by sorry

end NUMINAMATH_CALUDE_jersey_profit_calculation_l1469_146995


namespace NUMINAMATH_CALUDE_smallest_ellipse_area_l1469_146990

theorem smallest_ellipse_area (a b : ℝ) (h_ellipse : ∀ x y : ℝ, x^2 / a^2 + y^2 / b^2 = 1 → 
  ((x - 1/2)^2 + y^2 ≥ 1/4 ∧ (x + 1/2)^2 + y^2 ≥ 1/4)) :
  ∃ k : ℝ, k = 4 ∧ π * a * b ≥ k * π := by
  sorry

end NUMINAMATH_CALUDE_smallest_ellipse_area_l1469_146990


namespace NUMINAMATH_CALUDE_equation_solutions_l1469_146912

theorem equation_solutions :
  (∀ x : ℝ, x^2 - 3*x = 0 ↔ x = 0 ∨ x = 3) ∧
  (∀ y : ℝ, 2*y^2 + 4*y = y + 2 ↔ y = -2 ∨ y = 1/2) ∧
  (∀ y : ℝ, (2*y + 1)^2 - 25 = 0 ↔ y = -3 ∨ y = 2) :=
by sorry

end NUMINAMATH_CALUDE_equation_solutions_l1469_146912


namespace NUMINAMATH_CALUDE_log_division_simplification_l1469_146987

theorem log_division_simplification :
  Real.log 64 / Real.log (1/64) = -1 := by
  sorry

end NUMINAMATH_CALUDE_log_division_simplification_l1469_146987


namespace NUMINAMATH_CALUDE_divide_friends_among_teams_l1469_146978

theorem divide_friends_among_teams (n : ℕ) (k : ℕ) : 
  n = 8 ∧ k = 3 →
  (k^n : ℕ) - k * ((k-1)^n : ℕ) + (k * (k-1) * (k-2) * 1^n) / 2 = 5796 :=
by sorry

end NUMINAMATH_CALUDE_divide_friends_among_teams_l1469_146978


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l1469_146923

-- Define sets A and B
def A : Set ℝ := {x | 2 + x ≥ 4}
def B : Set ℝ := {x | -1 ≤ x ∧ x ≤ 5}

-- Theorem statement
theorem intersection_of_A_and_B : A ∩ B = {x : ℝ | 2 ≤ x ∧ x ≤ 5} := by
  sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l1469_146923


namespace NUMINAMATH_CALUDE_evaluate_expression_l1469_146927

theorem evaluate_expression (x y z : ℚ) 
  (hx : x = 1/4) (hy : y = 4/5) (hz : z = -2) : 
  x^3 * y^2 * z^2 = 1/25 := by sorry

end NUMINAMATH_CALUDE_evaluate_expression_l1469_146927


namespace NUMINAMATH_CALUDE_more_students_than_pets_l1469_146905

theorem more_students_than_pets :
  let num_classrooms : ℕ := 5
  let students_per_classroom : ℕ := 25
  let rabbits_per_classroom : ℕ := 3
  let guinea_pigs_per_classroom : ℕ := 3
  let total_students : ℕ := num_classrooms * students_per_classroom
  let total_rabbits : ℕ := num_classrooms * rabbits_per_classroom
  let total_guinea_pigs : ℕ := num_classrooms * guinea_pigs_per_classroom
  let total_pets : ℕ := total_rabbits + total_guinea_pigs
  total_students - total_pets = 95 := by
  sorry

end NUMINAMATH_CALUDE_more_students_than_pets_l1469_146905


namespace NUMINAMATH_CALUDE_rectangular_solid_surface_area_l1469_146903

theorem rectangular_solid_surface_area
  (a b c : ℝ)
  (sum_edges : a + b + c = 14)
  (diagonal : a^2 + b^2 + c^2 = 11^2) :
  2 * (a * b + b * c + a * c) = 75 := by
  sorry

end NUMINAMATH_CALUDE_rectangular_solid_surface_area_l1469_146903


namespace NUMINAMATH_CALUDE_roots_transformation_l1469_146910

theorem roots_transformation (r₁ r₂ r₃ : ℂ) : 
  (r₁^3 - 5*r₁^2 + 12 = 0) ∧ 
  (r₂^3 - 5*r₂^2 + 12 = 0) ∧ 
  (r₃^3 - 5*r₃^2 + 12 = 0) → 
  ((3*r₁)^3 - 15*(3*r₁)^2 + 324 = 0) ∧ 
  ((3*r₂)^3 - 15*(3*r₂)^2 + 324 = 0) ∧ 
  ((3*r₃)^3 - 15*(3*r₃)^2 + 324 = 0) := by
sorry

end NUMINAMATH_CALUDE_roots_transformation_l1469_146910


namespace NUMINAMATH_CALUDE_all_children_receive_candy_candy_distribution_works_l1469_146922

/-- Represents the candy distribution function -/
def candyDistribution (n : ℕ+) (k : ℕ) : ℕ :=
  (k * (k + 1) / 2) % n

/-- Theorem stating that all children receive candy iff n is a power of 2 -/
theorem all_children_receive_candy (n : ℕ+) :
  (∀ i : ℕ, i < n → ∃ k : ℕ, candyDistribution n k = i) ↔ ∃ m : ℕ, n = 2^m := by
  sorry

/-- Corollary: The number of children for which the candy distribution works -/
theorem candy_distribution_works (n : ℕ+) :
  (∀ i : ℕ, i < n → ∃ k : ℕ, candyDistribution n k = i) → ∃ m : ℕ, n = 2^m := by
  sorry

end NUMINAMATH_CALUDE_all_children_receive_candy_candy_distribution_works_l1469_146922


namespace NUMINAMATH_CALUDE_line_equation_through_point_with_inclination_l1469_146974

/-- Proves that the equation of a line passing through point (2, -3) with an inclination angle of 45° is x - y - 5 = 0 -/
theorem line_equation_through_point_with_inclination 
  (M : ℝ × ℝ) 
  (h_M : M = (2, -3)) 
  (α : ℝ) 
  (h_α : α = π / 4) : 
  ∀ x y : ℝ, (x - M.1) = (y - M.2) → x - y - 5 = 0 := by
  sorry

end NUMINAMATH_CALUDE_line_equation_through_point_with_inclination_l1469_146974


namespace NUMINAMATH_CALUDE_calculation_proof_l1469_146972

theorem calculation_proof : (((15^15 / 15^14)^3 * 8^3) / 2^9) = 3375 := by sorry

end NUMINAMATH_CALUDE_calculation_proof_l1469_146972


namespace NUMINAMATH_CALUDE_element_in_set_l1469_146996

def U : Set Nat := {1, 2, 3, 4, 5}

theorem element_in_set (M : Set Nat) (h : Set.compl M = {1, 3}) : 2 ∈ M := by
  sorry

end NUMINAMATH_CALUDE_element_in_set_l1469_146996


namespace NUMINAMATH_CALUDE_three_heads_in_four_tosses_l1469_146901

def coin_toss_probability (n : ℕ) (k : ℕ) : ℚ :=
  (Nat.choose n k : ℚ) * (1 / 2) ^ k * (1 / 2) ^ (n - k)

theorem three_heads_in_four_tosses :
  coin_toss_probability 4 3 = 1 / 4 := by
  sorry

end NUMINAMATH_CALUDE_three_heads_in_four_tosses_l1469_146901


namespace NUMINAMATH_CALUDE_binomial_coefficient_10_3_l1469_146994

theorem binomial_coefficient_10_3 : Nat.choose 10 3 = 120 := by sorry

end NUMINAMATH_CALUDE_binomial_coefficient_10_3_l1469_146994


namespace NUMINAMATH_CALUDE_intersection_complement_l1469_146984

def U : Set ℕ := {1, 2, 3, 4}

theorem intersection_complement (A B : Set ℕ) 
  (h1 : (A ∪ B)ᶜ = {4})
  (h2 : B = {1, 2}) : 
  A ∩ Bᶜ = {3} := by
  sorry

end NUMINAMATH_CALUDE_intersection_complement_l1469_146984


namespace NUMINAMATH_CALUDE_even_sum_condition_l1469_146921

theorem even_sum_condition (m n : ℤ) : 
  (∃ (k l : ℤ), m = 2 * k ∧ n = 2 * l → ∃ (p : ℤ), m + n = 2 * p) ∧ 
  (∃ (m n : ℤ), ∃ (q : ℤ), m + n = 2 * q ∧ ¬(∃ (r s : ℤ), m = 2 * r ∧ n = 2 * s)) :=
by sorry

end NUMINAMATH_CALUDE_even_sum_condition_l1469_146921


namespace NUMINAMATH_CALUDE_max_value_expression_l1469_146935

theorem max_value_expression (a b c d : ℝ) 
  (h_pos : a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0) 
  (h_sum : a + b + c + d ≤ 4) :
  (a * (b + 2 * c)) ^ (1/4) + 
  (b * (c + 2 * d)) ^ (1/4) + 
  (c * (d + 2 * a)) ^ (1/4) + 
  (d * (a + 2 * b)) ^ (1/4) ≤ 4 * 3 ^ (1/4) := by
sorry

end NUMINAMATH_CALUDE_max_value_expression_l1469_146935


namespace NUMINAMATH_CALUDE_solve_equation_one_solve_equation_two_l1469_146918

-- First equation: 3x = 2x + 12
theorem solve_equation_one : ∃ x : ℝ, 3 * x = 2 * x + 12 ∧ x = 12 := by sorry

-- Second equation: x/2 - 3 = 5
theorem solve_equation_two : ∃ x : ℝ, x / 2 - 3 = 5 ∧ x = 16 := by sorry

end NUMINAMATH_CALUDE_solve_equation_one_solve_equation_two_l1469_146918
