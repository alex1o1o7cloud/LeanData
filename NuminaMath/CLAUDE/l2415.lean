import Mathlib

namespace NUMINAMATH_CALUDE_horse_oats_consumption_l2415_241580

theorem horse_oats_consumption 
  (num_horses : ℕ) 
  (meals_per_day : ℕ) 
  (days : ℕ) 
  (total_oats : ℕ) 
  (h1 : num_horses = 4) 
  (h2 : meals_per_day = 2) 
  (h3 : days = 3) 
  (h4 : total_oats = 96) : 
  total_oats / (num_horses * meals_per_day * days) = 4 := by
sorry

end NUMINAMATH_CALUDE_horse_oats_consumption_l2415_241580


namespace NUMINAMATH_CALUDE_simplify_fraction_l2415_241511

theorem simplify_fraction : (222 : ℚ) / 8888 * 44 = 111 / 101 := by
  sorry

end NUMINAMATH_CALUDE_simplify_fraction_l2415_241511


namespace NUMINAMATH_CALUDE_least_period_is_30_l2415_241555

/-- A function satisfying the given condition -/
def SatisfyingFunction (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (x + 5) + f (x - 5) = f x

/-- The period of a function -/
def IsPeriod (f : ℝ → ℝ) (p : ℝ) : Prop :=
  ∀ x : ℝ, f (x + p) = f x

/-- The least positive period of a function -/
def IsLeastPeriod (f : ℝ → ℝ) (p : ℝ) : Prop :=
  p > 0 ∧ IsPeriod f p ∧ ∀ q, 0 < q ∧ q < p → ¬IsPeriod f q

/-- The main theorem -/
theorem least_period_is_30 :
  ∀ f : ℝ → ℝ, SatisfyingFunction f → IsLeastPeriod f 30 :=
sorry

end NUMINAMATH_CALUDE_least_period_is_30_l2415_241555


namespace NUMINAMATH_CALUDE_sum_of_integers_l2415_241523

theorem sum_of_integers (x y : ℤ) 
  (h1 : x^2 + y^2 = 290) 
  (h2 : x * y = 96) : 
  x + y = 22 := by
sorry

end NUMINAMATH_CALUDE_sum_of_integers_l2415_241523


namespace NUMINAMATH_CALUDE_derivative_f_at_negative_one_l2415_241514

-- Define the function f
def f (x : ℝ) : ℝ := x^3 + 2*x^2 - 1

-- State the theorem
theorem derivative_f_at_negative_one :
  deriv f (-1) = -1 := by sorry

end NUMINAMATH_CALUDE_derivative_f_at_negative_one_l2415_241514


namespace NUMINAMATH_CALUDE_pen_sales_problem_l2415_241565

theorem pen_sales_problem (d : ℕ) : 
  (96 + 44 * d) / (d + 1) = 48 → d = 12 := by
  sorry

end NUMINAMATH_CALUDE_pen_sales_problem_l2415_241565


namespace NUMINAMATH_CALUDE_min_phase_shift_cosine_l2415_241529

/-- Given a cosine function with a specific symmetry point, prove the minimum absolute value of the phase shift -/
theorem min_phase_shift_cosine (φ : ℝ) : 
  (∀ x : ℝ, 3 * Real.cos (2 * x + φ) = 3 * Real.cos (2 * (8 * Real.pi / 3 - x) + φ)) → 
  (∃ k : ℤ, φ = k * Real.pi - 13 * Real.pi / 6) →
  ∃ ψ : ℝ, (∀ θ : ℝ, |θ| ≥ |ψ|) ∧ |ψ| = Real.pi / 6 :=
by sorry

end NUMINAMATH_CALUDE_min_phase_shift_cosine_l2415_241529


namespace NUMINAMATH_CALUDE_polynomial_composition_difference_l2415_241517

theorem polynomial_composition_difference (f : Polynomial ℝ) :
  ∃ (g h : Polynomial ℝ), f = g.comp h - h.comp g := by
  sorry

end NUMINAMATH_CALUDE_polynomial_composition_difference_l2415_241517


namespace NUMINAMATH_CALUDE_ball_fall_time_l2415_241586

/-- The time for a ball to fall from 60 meters to 30 meters under gravity -/
theorem ball_fall_time (g : Real) (h₀ h₁ : Real) (t : Real) :
  g = 9.8 →
  h₀ = 60 →
  h₁ = 30 →
  h₁ = h₀ - (1/2) * g * t^2 →
  t = Real.sqrt (2 * (h₀ - h₁) / g) :=
by sorry

end NUMINAMATH_CALUDE_ball_fall_time_l2415_241586


namespace NUMINAMATH_CALUDE_widget_carton_height_l2415_241521

/-- Represents the dimensions of a box -/
structure BoxDimensions where
  width : ℝ
  length : ℝ
  height : ℝ

/-- Represents the packing configuration -/
structure PackingConfig where
  widgetsPerCarton : ℕ
  widgetsPerShippingBox : ℕ
  cartonDimensions : BoxDimensions
  shippingBoxDimensions : BoxDimensions

/-- The packing configuration for the Widget Factory -/
def widgetFactoryConfig : PackingConfig :=
  { widgetsPerCarton := 3
  , widgetsPerShippingBox := 300
  , cartonDimensions := 
    { width := 4
    , length := 4
    , height := 0  -- Unknown, to be determined
    }
  , shippingBoxDimensions := 
    { width := 20
    , length := 20
    , height := 20
    }
  }

/-- Theorem: The height of each carton in the Widget Factory configuration is 5 inches -/
theorem widget_carton_height (config : PackingConfig := widgetFactoryConfig) : 
  config.cartonDimensions.height = 5 := by
  sorry


end NUMINAMATH_CALUDE_widget_carton_height_l2415_241521


namespace NUMINAMATH_CALUDE_triangle_inequality_fraction_l2415_241556

theorem triangle_inequality_fraction (a b c : ℝ) (h : a > 0 ∧ b > 0 ∧ c > 0) 
  (triangle_ineq : a + b > c ∧ b + c > a ∧ c + a > b) : (a + b) / (a + b + c) > 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_triangle_inequality_fraction_l2415_241556


namespace NUMINAMATH_CALUDE_inequality_system_solution_l2415_241581

theorem inequality_system_solution :
  let S : Set ℝ := {x | 5 * x - 2 < 3 * (x + 1) ∧ (3 * x - 2) / 3 ≥ x + (x - 2) / 2}
  S = {x | x ≤ 2/3} := by
sorry

end NUMINAMATH_CALUDE_inequality_system_solution_l2415_241581


namespace NUMINAMATH_CALUDE_patsy_appetizers_needed_l2415_241542

def appetizers_per_guest : ℕ := 6
def number_of_guests : ℕ := 30
def deviled_eggs_dozens : ℕ := 3
def pigs_in_blanket_dozens : ℕ := 2
def kebabs_dozens : ℕ := 2
def items_per_dozen : ℕ := 12

theorem patsy_appetizers_needed : 
  (appetizers_per_guest * number_of_guests - 
   (deviled_eggs_dozens + pigs_in_blanket_dozens + kebabs_dozens) * items_per_dozen) / items_per_dozen = 8 := by
  sorry

end NUMINAMATH_CALUDE_patsy_appetizers_needed_l2415_241542


namespace NUMINAMATH_CALUDE_librarian_shelves_l2415_241564

/-- The number of books on the top shelf -/
def first_term : ℕ := 3

/-- The difference in the number of books between each consecutive shelf -/
def common_difference : ℕ := 3

/-- The total number of books on all shelves -/
def total_books : ℕ := 225

/-- The number of shelves used by the librarian -/
def num_shelves : ℕ := 15

theorem librarian_shelves :
  ∃ (n : ℕ), n = num_shelves ∧
  n * (2 * first_term + (n - 1) * common_difference) = 2 * total_books :=
sorry

end NUMINAMATH_CALUDE_librarian_shelves_l2415_241564


namespace NUMINAMATH_CALUDE_article_price_calculation_l2415_241533

/-- The original price of an article before discounts and tax -/
def original_price : ℝ := 259.20

/-- The final price of the article after discounts and tax -/
def final_price : ℝ := 144

/-- The first discount rate -/
def discount1 : ℝ := 0.12

/-- The second discount rate -/
def discount2 : ℝ := 0.22

/-- The third discount rate -/
def discount3 : ℝ := 0.15

/-- The sales tax rate -/
def tax_rate : ℝ := 0.06

theorem article_price_calculation (ε : ℝ) (hε : ε > 0) :
  ∃ (price : ℝ), 
    abs (price - original_price) < ε ∧ 
    price * (1 - discount1) * (1 - discount2) * (1 - discount3) * (1 + tax_rate) = final_price :=
sorry

end NUMINAMATH_CALUDE_article_price_calculation_l2415_241533


namespace NUMINAMATH_CALUDE_quadratic_trinomial_zero_discriminant_sum_l2415_241501

/-- A quadratic trinomial can be represented as the sum of two quadratic trinomials with zero discriminants -/
theorem quadratic_trinomial_zero_discriminant_sum (a b c : ℝ) (ha : a ≠ 0) :
  ∃ (f g : ℝ → ℝ),
    (∀ x, a * x^2 + b * x + c = f x + g x) ∧
    (∃ (a₁ b₁ c₁ : ℝ), ∀ x, f x = a₁ * x^2 + b₁ * x + c₁) ∧
    (∃ (a₂ b₂ c₂ : ℝ), ∀ x, g x = a₂ * x^2 + b₂ * x + c₂) ∧
    (b₁^2 - 4 * a₁ * c₁ = 0) ∧
    (b₂^2 - 4 * a₂ * c₂ = 0) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_trinomial_zero_discriminant_sum_l2415_241501


namespace NUMINAMATH_CALUDE_function_inequality_l2415_241598

def is_odd (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

def is_decreasing_on (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x y, a ≤ x ∧ x < y ∧ y ≤ b → f y < f x

theorem function_inequality (f : ℝ → ℝ) 
  (h_odd : is_odd f)
  (h_periodic : ∀ x, f (x + 2) = -f x)
  (h_decreasing : is_decreasing_on f 0 1) :
  f (3/2) < f (1/4) ∧ f (1/4) < f (-1/4) := by
  sorry

end NUMINAMATH_CALUDE_function_inequality_l2415_241598


namespace NUMINAMATH_CALUDE_quadratic_roots_problem_l2415_241562

theorem quadratic_roots_problem (x₁ x₂ b : ℝ) : 
  (∀ x, x^2 + b*x + 4 = 0 ↔ x = x₁ ∨ x = x₂) →
  x₁ - x₁*x₂ + x₂ = 2 →
  b = -6 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_roots_problem_l2415_241562


namespace NUMINAMATH_CALUDE_candidate_failing_marks_l2415_241594

def max_marks : ℕ := 500
def passing_percentage : ℚ := 45 / 100
def candidate_marks : ℕ := 180

theorem candidate_failing_marks :
  (max_marks * passing_percentage).floor - candidate_marks = 45 := by
  sorry

end NUMINAMATH_CALUDE_candidate_failing_marks_l2415_241594


namespace NUMINAMATH_CALUDE_line_perp_plane_implies_planes_perp_l2415_241592

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relations
variable (subset : Line → Plane → Prop)
variable (perp : Line → Plane → Prop)
variable (perpPlanes : Plane → Plane → Prop)

-- State the theorem
theorem line_perp_plane_implies_planes_perp
  (m n : Line) (α β : Plane)
  (h1 : m ≠ n)
  (h2 : α ≠ β)
  (h3 : subset m α)
  (h4 : subset n β)
  (h5 : perp m β) :
  perpPlanes α β :=
sorry

end NUMINAMATH_CALUDE_line_perp_plane_implies_planes_perp_l2415_241592


namespace NUMINAMATH_CALUDE_arithmetic_sequence_a15_l2415_241567

/-- An arithmetic sequence is a sequence where the difference between
    any two consecutive terms is constant. -/
def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_a15
  (a : ℕ → ℝ)
  (h_arithmetic : is_arithmetic_sequence a)
  (h_sum : a 3 + a 13 = 20)
  (h_a2 : a 2 = -2) :
  a 15 = 24 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_a15_l2415_241567


namespace NUMINAMATH_CALUDE_cheryl_prob_correct_l2415_241582

/-- Represents the number of marbles of each color in the box -/
def marbles_per_color : ℕ := 3

/-- Represents the number of colors of marbles in the box -/
def num_colors : ℕ := 4

/-- Represents the total number of marbles in the box -/
def total_marbles : ℕ := marbles_per_color * num_colors

/-- Represents the number of marbles each person draws -/
def marbles_drawn : ℕ := 3

/-- Represents the probability of Cheryl getting 3 marbles of the same color,
    given that Claudia did not draw 3 marbles of the same color -/
def cheryl_same_color_prob : ℚ := 55 / 1540

theorem cheryl_prob_correct :
  cheryl_same_color_prob =
    (num_colors - 1) * (Nat.choose total_marbles marbles_drawn) /
    (Nat.choose total_marbles marbles_drawn *
     (Nat.choose (total_marbles - marbles_drawn) marbles_drawn -
      num_colors * 1) * 1) :=
by sorry

end NUMINAMATH_CALUDE_cheryl_prob_correct_l2415_241582


namespace NUMINAMATH_CALUDE_total_tickets_proof_l2415_241588

/-- The number of tickets Tom spent at the 'dunk a clown' booth -/
def tickets_spent_at_booth : ℕ := 28

/-- The number of rides Tom went on -/
def number_of_rides : ℕ := 3

/-- The cost of each ride in tickets -/
def cost_per_ride : ℕ := 4

/-- The total number of tickets Tom bought at the state fair -/
def total_tickets : ℕ := tickets_spent_at_booth + number_of_rides * cost_per_ride

theorem total_tickets_proof : total_tickets = 40 := by
  sorry

end NUMINAMATH_CALUDE_total_tickets_proof_l2415_241588


namespace NUMINAMATH_CALUDE_f_500_equals_39_l2415_241508

/-- A function satisfying the given properties -/
def special_function (f : ℕ+ → ℕ) : Prop :=
  (∀ x y : ℕ+, f (x * y) = f x + f y) ∧ 
  (f 10 = 14) ∧ 
  (f 40 = 20)

/-- Theorem stating the result for f(500) -/
theorem f_500_equals_39 (f : ℕ+ → ℕ) (h : special_function f) : f 500 = 39 := by
  sorry

end NUMINAMATH_CALUDE_f_500_equals_39_l2415_241508


namespace NUMINAMATH_CALUDE_min_value_quadratic_l2415_241593

theorem min_value_quadratic (x : ℝ) : 
  ∃ (m : ℝ), m = 217 ∧ ∀ x, 3 * x^2 - 18 * x + 244 ≥ m := by
  sorry

end NUMINAMATH_CALUDE_min_value_quadratic_l2415_241593


namespace NUMINAMATH_CALUDE_yellow_pill_cost_l2415_241585

theorem yellow_pill_cost (weeks : ℕ) (daily_yellow : ℕ) (daily_blue : ℕ) 
  (yellow_blue_diff : ℚ) (total_cost : ℚ) :
  weeks = 3 →
  daily_yellow = 1 →
  daily_blue = 1 →
  yellow_blue_diff = 2 →
  total_cost = 903 →
  ∃ (yellow_cost : ℚ), 
    yellow_cost = 22.5 ∧ 
    (weeks * 7 * (yellow_cost + (yellow_cost - yellow_blue_diff)) = total_cost) :=
by sorry

end NUMINAMATH_CALUDE_yellow_pill_cost_l2415_241585


namespace NUMINAMATH_CALUDE_average_cost_approx_1_50_l2415_241541

/-- Calculates the average cost per piece of fruit given specific quantities and prices. -/
def average_cost_per_fruit (apple_price banana_price orange_price grape_price kiwi_price : ℚ)
  (apple_qty banana_qty orange_qty grape_qty kiwi_qty : ℕ) : ℚ :=
  let apple_cost := if apple_qty ≥ 10 then (apple_qty - 2) * apple_price else apple_qty * apple_price
  let orange_cost := if orange_qty ≥ 3 then (orange_qty - (orange_qty / 3)) * orange_price else orange_qty * orange_price
  let grape_cost := if grape_qty * grape_price > 10 then grape_qty * grape_price * (1 - 0.2) else grape_qty * grape_price
  let kiwi_cost := if kiwi_qty ≥ 10 then kiwi_qty * kiwi_price * (1 - 0.15) else kiwi_qty * kiwi_price
  let banana_cost := banana_qty * banana_price
  let total_cost := apple_cost + orange_cost + grape_cost + kiwi_cost + banana_cost
  let total_pieces := apple_qty + orange_qty + grape_qty + kiwi_qty + banana_qty
  total_cost / total_pieces

/-- The average cost per piece of fruit is approximately $1.50 given the specific conditions. -/
theorem average_cost_approx_1_50 :
  ∃ ε > 0, |average_cost_per_fruit 2 1 3 (3/2) (7/4) 12 4 4 10 10 - (3/2)| < ε :=
by sorry

end NUMINAMATH_CALUDE_average_cost_approx_1_50_l2415_241541


namespace NUMINAMATH_CALUDE_maries_speed_l2415_241503

/-- Given that Marie can bike 372 miles in 31 hours, prove that her speed is 12 miles per hour. -/
theorem maries_speed (distance : ℝ) (time : ℝ) (h1 : distance = 372) (h2 : time = 31) :
  distance / time = 12 := by
  sorry

end NUMINAMATH_CALUDE_maries_speed_l2415_241503


namespace NUMINAMATH_CALUDE_second_order_eq_circle_iff_l2415_241553

/-- A general second-order equation in two variables -/
structure SecondOrderEquation where
  a11 : ℝ
  a12 : ℝ
  a22 : ℝ
  a13 : ℝ
  a23 : ℝ
  a33 : ℝ

/-- Predicate to check if a second-order equation represents a circle -/
def IsCircle (eq : SecondOrderEquation) : Prop :=
  eq.a11 = eq.a22 ∧ eq.a12 = 0

/-- Theorem stating the conditions for a second-order equation to represent a circle -/
theorem second_order_eq_circle_iff (eq : SecondOrderEquation) :
  IsCircle eq ↔ ∃ (h k : ℝ) (r : ℝ), r > 0 ∧
    ∀ (x y : ℝ), eq.a11 * x^2 + 2*eq.a12 * x*y + eq.a22 * y^2 + 2*eq.a13 * x + 2*eq.a23 * y + eq.a33 = 0 ↔
    (x - h)^2 + (y - k)^2 = r^2 :=
  sorry


end NUMINAMATH_CALUDE_second_order_eq_circle_iff_l2415_241553


namespace NUMINAMATH_CALUDE_negation_of_existence_negation_of_proposition_l2415_241536

theorem negation_of_existence (p : ℝ → Prop) : 
  (¬ ∃ x, p x) ↔ (∀ x, ¬ p x) :=
by sorry

theorem negation_of_proposition : 
  (¬ ∃ x : ℝ, x^2 - x + 1 ≤ 0) ↔ (∀ x : ℝ, x^2 - x + 1 > 0) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_existence_negation_of_proposition_l2415_241536


namespace NUMINAMATH_CALUDE_binomial_12_choose_3_l2415_241535

theorem binomial_12_choose_3 : Nat.choose 12 3 = 220 := by
  sorry

end NUMINAMATH_CALUDE_binomial_12_choose_3_l2415_241535


namespace NUMINAMATH_CALUDE_disjoint_triangles_exist_l2415_241544

/-- A point in a 2D plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- Checks if three points are collinear -/
def collinear (p q r : Point) : Prop :=
  (q.x - p.x) * (r.y - p.y) = (r.x - p.x) * (q.y - p.y)

/-- A triangle formed by three points -/
structure Triangle where
  a : Point
  b : Point
  c : Point

/-- Checks if two triangles are disjoint -/
def disjoint (t1 t2 : Triangle) : Prop :=
  t1.a ≠ t2.a ∧ t1.a ≠ t2.b ∧ t1.a ≠ t2.c ∧
  t1.b ≠ t2.a ∧ t1.b ≠ t2.b ∧ t1.b ≠ t2.c ∧
  t1.c ≠ t2.a ∧ t1.c ≠ t2.b ∧ t1.c ≠ t2.c

/-- The main theorem -/
theorem disjoint_triangles_exist (n : ℕ) (points : Fin (3 * n) → Point) 
  (h : ∀ i j k, i ≠ j → j ≠ k → i ≠ k → ¬collinear (points i) (points j) (points k)) :
  ∃ triangles : Fin n → Triangle, 
    (∀ i, ∃ j k l, triangles i = ⟨points j, points k, points l⟩) ∧ 
    (∀ i j, i ≠ j → disjoint (triangles i) (triangles j)) :=
  sorry


end NUMINAMATH_CALUDE_disjoint_triangles_exist_l2415_241544


namespace NUMINAMATH_CALUDE_square_area_ratio_l2415_241579

theorem square_area_ratio (y : ℝ) (h : y > 0) : 
  (y^2) / ((5*y)^2) = 1 / 25 := by sorry

end NUMINAMATH_CALUDE_square_area_ratio_l2415_241579


namespace NUMINAMATH_CALUDE_perpendicular_implies_y_parallel_implies_x_l2415_241558

/-- Given vectors in R² -/
def a : Fin 2 → ℚ := ![3, -2]
def b (y : ℚ) : Fin 2 → ℚ := ![-1, y]
def c (x : ℚ) : Fin 2 → ℚ := ![x, 5]

/-- Dot product of two vectors in R² -/
def dot_product (v w : Fin 2 → ℚ) : ℚ := (v 0) * (w 0) + (v 1) * (w 1)

/-- Two vectors are perpendicular if their dot product is zero -/
def is_perpendicular (v w : Fin 2 → ℚ) : Prop := dot_product v w = 0

/-- Two vectors are parallel if one is a scalar multiple of the other -/
def is_parallel (v w : Fin 2 → ℚ) : Prop := ∃ (k : ℚ), v 0 = k * (w 0) ∧ v 1 = k * (w 1)

theorem perpendicular_implies_y (y : ℚ) : 
  is_perpendicular a (b y) → y = -3/2 := by sorry

theorem parallel_implies_x (x : ℚ) :
  is_parallel a (c x) → x = -15/2 := by sorry

end NUMINAMATH_CALUDE_perpendicular_implies_y_parallel_implies_x_l2415_241558


namespace NUMINAMATH_CALUDE_closest_to_zero_minus_one_closest_l2415_241504

def integers : List ℤ := [-1, 2, -3, 4, -5]

theorem closest_to_zero (n : ℤ) (h : n ∈ integers) : 
  ∀ m ∈ integers, |n| ≤ |m| :=
by
  sorry

theorem minus_one_closest : 
  ∃ n ∈ integers, ∀ m ∈ integers, |n| ≤ |m| ∧ n = -1 :=
by
  sorry

end NUMINAMATH_CALUDE_closest_to_zero_minus_one_closest_l2415_241504


namespace NUMINAMATH_CALUDE_factorable_implies_even_b_l2415_241530

/-- A quadratic expression of the form 15x^2 + bx + 15 -/
def quadratic_expr (b : ℤ) (x : ℝ) : ℝ := 15 * x^2 + b * x + 15

/-- Represents a linear binomial factor with integer coefficients -/
structure LinearFactor where
  c : ℤ
  d : ℤ

/-- Checks if a quadratic expression can be factored into two linear binomial factors -/
def is_factorable (b : ℤ) : Prop :=
  ∃ (f1 f2 : LinearFactor), ∀ x, 
    quadratic_expr b x = (f1.c * x + f1.d) * (f2.c * x + f2.d)

theorem factorable_implies_even_b :
  ∀ b : ℤ, is_factorable b → Even b :=
sorry

end NUMINAMATH_CALUDE_factorable_implies_even_b_l2415_241530


namespace NUMINAMATH_CALUDE_line_passes_through_fixed_point_l2415_241584

/-- The line equation passes through a fixed point for all values of m -/
theorem line_passes_through_fixed_point :
  ∀ (m : ℝ), 2 * (-1/2) - m * (-3) + 1 - 3*m = 0 := by
  sorry

end NUMINAMATH_CALUDE_line_passes_through_fixed_point_l2415_241584


namespace NUMINAMATH_CALUDE_elizabeth_haircut_l2415_241506

theorem elizabeth_haircut (first_day : ℝ) (second_day : ℝ) 
  (h1 : first_day = 0.38)
  (h2 : second_day = 0.5) :
  first_day + second_day = 0.88 := by
sorry

end NUMINAMATH_CALUDE_elizabeth_haircut_l2415_241506


namespace NUMINAMATH_CALUDE_sin_cos_identity_l2415_241568

theorem sin_cos_identity (x : ℝ) (h : Real.sin (x + π / 3) = 1 / 3) :
  Real.sin (5 * π / 3 - x) - Real.cos (2 * x - π / 3) = 4 / 9 := by
  sorry

end NUMINAMATH_CALUDE_sin_cos_identity_l2415_241568


namespace NUMINAMATH_CALUDE_ellipse_circle_tangent_l2415_241576

/-- Given an ellipse M and a circle N with specific properties, prove their equations and the equation of their common tangent line. -/
theorem ellipse_circle_tangent (a b c : ℝ) (k m : ℝ) :
  a > 0 ∧ b > 0 ∧ a > b ∧  -- conditions on a, b
  c / a = 1 / 2 ∧  -- eccentricity
  a^2 / c - c = 3 ∧  -- distance condition
  c > 0 →  -- c is positive (implied by being a distance)
  -- Prove:
  ((∀ x y : ℝ, x^2 / a^2 + y^2 / b^2 = 1 ↔ x^2 / 4 + y^2 / 3 = 1) ∧  -- equation of M
   (∀ x y : ℝ, (x - c)^2 + y^2 = a^2 + c^2 ↔ (x - 1)^2 + y^2 = 5) ∧  -- equation of N
   ((k = 1/2 ∧ m = 2) ∨ (k = -1/2 ∧ m = -2)) ∧  -- equation of tangent line l
   (∀ x : ℝ, (x^2 / 4 + (k * x + m)^2 / 3 = 1 →  -- l is tangent to M
              ∃! y : ℝ, y = k * x + m ∧ x^2 / 4 + y^2 / 3 = 1) ∧
    ((k * 1 + m)^2 + 1^2 = 5)))  -- l is tangent to N
  := by sorry

end NUMINAMATH_CALUDE_ellipse_circle_tangent_l2415_241576


namespace NUMINAMATH_CALUDE_ellipse_equation_hyperbola_equation_l2415_241537

-- Ellipse problem
theorem ellipse_equation (f c a b : ℝ) (h1 : f = 8) (h2 : c = 4) (h3 : a = 5) (h4 : b = 3) (h5 : c / a = 0.8) :
  (∀ x y : ℝ, x^2 / 25 + y^2 / 9 = 1) ∨ (∀ x y : ℝ, y^2 / 25 + x^2 / 9 = 1) :=
sorry

-- Hyperbola problem
theorem hyperbola_equation (a b m : ℝ) 
  (h1 : ∀ x y : ℝ, y^2 / 4 - x^2 / 3 = 1 → y^2 / (4*m) - x^2 / (3*m) = 1) 
  (h2 : 3^2 / (6*m) - 2^2 / (8*m) = 1) :
  (∀ x y : ℝ, x^2 / 6 - y^2 / 8 = 1) :=
sorry

end NUMINAMATH_CALUDE_ellipse_equation_hyperbola_equation_l2415_241537


namespace NUMINAMATH_CALUDE_escalator_steps_count_l2415_241528

/-- Represents the number of steps a person climbs on the escalator -/
structure ClimbingSteps where
  steps : ℕ

/-- Represents the speed at which a person climbs the escalator -/
structure ClimbingSpeed where
  speed : ℕ

/-- Represents a person climbing the escalator -/
structure Person where
  climbingSteps : ClimbingSteps
  climbingSpeed : ClimbingSpeed

/-- Calculates the total number of steps in the escalator -/
def escalatorSteps (personA personB : Person) : ℕ :=
  sorry

theorem escalator_steps_count
  (personA personB : Person)
  (hA : personA.climbingSteps.steps = 55)
  (hB : personB.climbingSteps.steps = 60)
  (hSpeed : personB.climbingSpeed.speed = 2 * personA.climbingSpeed.speed) :
  escalatorSteps personA personB = 66 :=
sorry

end NUMINAMATH_CALUDE_escalator_steps_count_l2415_241528


namespace NUMINAMATH_CALUDE_linear_function_passes_through_point_l2415_241591

/-- A linear function y = kx - k (k ≠ 0) that passes through the point (-1, 4) also passes through the point (1, 0). -/
theorem linear_function_passes_through_point (k : ℝ) (hk : k ≠ 0) :
  (∃ y : ℝ, y = k * (-1) - k ∧ y = 4) →
  (∃ y : ℝ, y = k * 1 - k ∧ y = 0) :=
by sorry

end NUMINAMATH_CALUDE_linear_function_passes_through_point_l2415_241591


namespace NUMINAMATH_CALUDE_equation_solution_l2415_241573

theorem equation_solution : ∃ N : ℝ,
  (∃ e₁ e₂ : ℝ, 2 * |2 - e₁| = N ∧ 2 * |2 - e₂| = N ∧ e₁ + e₂ = 4) →
  N = 0 :=
by sorry

end NUMINAMATH_CALUDE_equation_solution_l2415_241573


namespace NUMINAMATH_CALUDE_certain_number_proof_l2415_241575

theorem certain_number_proof :
  let first_number : ℝ := 15
  let certain_number : ℝ := (0.4 * first_number) - (0.8 * 5)
  certain_number = 2 := by
sorry

end NUMINAMATH_CALUDE_certain_number_proof_l2415_241575


namespace NUMINAMATH_CALUDE_similar_triangles_leg_length_l2415_241510

/-- Two similar right triangles with legs 12 and 9 in the first, and x and 6 in the second, have x = 8 -/
theorem similar_triangles_leg_length : ∀ x : ℝ,
  (12 : ℝ) / x = 9 / 6 → x = 8 := by
  sorry

end NUMINAMATH_CALUDE_similar_triangles_leg_length_l2415_241510


namespace NUMINAMATH_CALUDE_sandy_clothes_cost_l2415_241563

-- Define the costs of individual items
def shorts_cost : ℚ := 13.99
def shirt_cost : ℚ := 12.14
def jacket_cost : ℚ := 7.43

-- Define the total cost
def total_cost : ℚ := shorts_cost + shirt_cost + jacket_cost

-- Theorem statement
theorem sandy_clothes_cost : total_cost = 33.56 := by
  sorry

end NUMINAMATH_CALUDE_sandy_clothes_cost_l2415_241563


namespace NUMINAMATH_CALUDE_smallest_a_value_l2415_241549

/-- Given two quadratic equations with integer coefficients and integer roots less than -1,
    this theorem states that the smallest possible value for the constant term 'a' is 15. -/
theorem smallest_a_value (a b c : ℤ) : 
  (∃ x y : ℤ, x < -1 ∧ y < -1 ∧ x^2 + b*x + a = 0 ∧ y^2 + b*y + a = 0) →
  (∃ z w : ℤ, z < -1 ∧ w < -1 ∧ z^2 + c*z + a = 1 ∧ w^2 + c*w + a = 1) →
  (∀ a' : ℤ, (∃ b' c' : ℤ, 
    (∃ x y : ℤ, x < -1 ∧ y < -1 ∧ x^2 + b'*x + a' = 0 ∧ y^2 + b'*y + a' = 0) ∧
    (∃ z w : ℤ, z < -1 ∧ w < -1 ∧ z^2 + c'*z + a' = 1 ∧ w^2 + c'*w + a' = 1)) →
    a' ≥ 15) →
  a = 15 :=
by sorry

end NUMINAMATH_CALUDE_smallest_a_value_l2415_241549


namespace NUMINAMATH_CALUDE_value_of_y_l2415_241583

theorem value_of_y (x y z : ℝ) 
  (h1 : 3 * x = 0.75 * y)
  (h2 : x + z = 24)
  (h3 : z = 8) :
  y = 64 := by
sorry

end NUMINAMATH_CALUDE_value_of_y_l2415_241583


namespace NUMINAMATH_CALUDE_typists_count_l2415_241557

/-- The number of typists in the initial group -/
def initial_typists : ℕ := 20

/-- The number of letters typed by the initial group in 20 minutes -/
def letters_20min : ℕ := 46

/-- The number of typists in the second group -/
def second_group_typists : ℕ := 30

/-- The number of letters typed by the second group in 1 hour -/
def letters_1hour : ℕ := 207

/-- The duration of the first typing session in minutes -/
def first_duration : ℕ := 20

/-- The duration of the second typing session in minutes -/
def second_duration : ℕ := 60

theorem typists_count : 
  initial_typists * second_group_typists * letters_1hour * first_duration = 
  letters_20min * second_group_typists * second_duration := by
  sorry

end NUMINAMATH_CALUDE_typists_count_l2415_241557


namespace NUMINAMATH_CALUDE_juice_cans_for_two_dollars_l2415_241599

def anniversary_sale (original_price : ℕ) (discount : ℕ) (total_cost : ℕ) (ice_cream_count : ℕ) (juice_cans : ℕ) : Prop :=
  let sale_price := original_price - discount
  let ice_cream_total := sale_price * ice_cream_count
  let juice_cost := total_cost - ice_cream_total
  ∃ (cans_per_two_dollars : ℕ), 
    cans_per_two_dollars * (juice_cost / 2) = juice_cans ∧
    cans_per_two_dollars = 5

theorem juice_cans_for_two_dollars :
  anniversary_sale 12 2 24 2 10 → ∃ (x : ℕ), x = 5 :=
by
  sorry

end NUMINAMATH_CALUDE_juice_cans_for_two_dollars_l2415_241599


namespace NUMINAMATH_CALUDE_equation_solution_l2415_241547

theorem equation_solution : ∃ x : ℚ, (5*x + 2*x = 450 - 10*(x - 5) + 4) ∧ (x = 504/17) := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l2415_241547


namespace NUMINAMATH_CALUDE_percentage_of_sikh_boys_l2415_241570

/-- Given a school with the following demographics:
  - Total number of boys: 850
  - 44% are Muslims
  - 28% are Hindus
  - 153 boys belong to other communities
  Prove that 10% of the boys are Sikhs -/
theorem percentage_of_sikh_boys (total : ℕ) (muslim_percent : ℚ) (hindu_percent : ℚ) (other : ℕ) :
  total = 850 →
  muslim_percent = 44 / 100 →
  hindu_percent = 28 / 100 →
  other = 153 →
  (total - (muslim_percent * total + hindu_percent * total + other : ℚ)) / total = 1 / 10 := by
  sorry

end NUMINAMATH_CALUDE_percentage_of_sikh_boys_l2415_241570


namespace NUMINAMATH_CALUDE_randy_trip_length_l2415_241559

theorem randy_trip_length :
  ∀ (total_length : ℚ),
    (1 / 4 : ℚ) * total_length +  -- gravel road
    30 +                          -- pavement
    (1 / 8 : ℚ) * total_length +  -- scenic route
    (1 / 6 : ℚ) * total_length    -- dirt road
    = total_length →
    total_length = 720 / 11 := by
  sorry

end NUMINAMATH_CALUDE_randy_trip_length_l2415_241559


namespace NUMINAMATH_CALUDE_interest_rate_20_percent_l2415_241502

-- Define the compound interest function
def compound_interest (P : ℝ) (r : ℝ) (t : ℕ) : ℝ := P * (1 + r) ^ t

-- State the theorem
theorem interest_rate_20_percent 
  (P : ℝ) 
  (r : ℝ) 
  (h1 : compound_interest P r 3 = 3000) 
  (h2 : compound_interest P r 4 = 3600) :
  r = 0.2 :=
sorry

end NUMINAMATH_CALUDE_interest_rate_20_percent_l2415_241502


namespace NUMINAMATH_CALUDE_daisy_percentage_is_62_l2415_241539

/-- Represents the composition of flowers in a garden -/
structure Garden where
  total : ℝ
  yellow_ratio : ℝ
  yellow_tulip_ratio : ℝ
  red_daisy_ratio : ℝ

/-- The percentage of daisies in the garden -/
def daisy_percentage (g : Garden) : ℝ :=
  ((g.yellow_ratio - g.yellow_ratio * g.yellow_tulip_ratio) + 
   ((1 - g.yellow_ratio) * g.red_daisy_ratio)) * 100

/-- Theorem stating that the percentage of daisies in the garden is 62% -/
theorem daisy_percentage_is_62 (g : Garden) 
  (h1 : g.yellow_tulip_ratio = 1/5)
  (h2 : g.red_daisy_ratio = 1/2)
  (h3 : g.yellow_ratio = 4/10) :
  daisy_percentage g = 62 := by
  sorry

#eval daisy_percentage { total := 100, yellow_ratio := 0.4, yellow_tulip_ratio := 0.2, red_daisy_ratio := 0.5 }

end NUMINAMATH_CALUDE_daisy_percentage_is_62_l2415_241539


namespace NUMINAMATH_CALUDE_profit_sharing_l2415_241513

/-- The profit sharing problem -/
theorem profit_sharing
  (invest_a invest_b invest_c : ℝ)
  (total_profit : ℝ)
  (h1 : invest_a = 3 * invest_b)
  (h2 : invest_a = 2 / 3 * invest_c)
  (h3 : total_profit = 12375) :
  (invest_c / (invest_a + invest_b + invest_c)) * total_profit = (9 / 17) * 12375 := by
sorry

#eval (9 / 17 : ℚ) * 12375

end NUMINAMATH_CALUDE_profit_sharing_l2415_241513


namespace NUMINAMATH_CALUDE_writing_ways_equals_notebooks_l2415_241507

/-- The number of ways to start writing given a ratio of pens to notebooks and their quantities -/
def ways_to_start_writing (pen_ratio : ℕ) (notebook_ratio : ℕ) (num_pens : ℕ) (num_notebooks : ℕ) : ℕ :=
  min num_pens num_notebooks

/-- Theorem: Given the ratio of pens to notebooks is 5:4, with 50 pens and 40 notebooks,
    the number of ways to start writing is equal to the number of notebooks -/
theorem writing_ways_equals_notebooks :
  ways_to_start_writing 5 4 50 40 = 40 := by
  sorry

end NUMINAMATH_CALUDE_writing_ways_equals_notebooks_l2415_241507


namespace NUMINAMATH_CALUDE_tan_one_condition_l2415_241519

theorem tan_one_condition (x : Real) : 
  (∃ k : Int, x = (k * Real.pi) / 4) ∧ 
  (∃ x : Real, (∃ k : Int, x = (k * Real.pi) / 4) ∧ Real.tan x ≠ 1) ∧
  (∀ x : Real, Real.tan x = 1 → ∃ k : Int, x = ((4 * k + 1) * Real.pi) / 4) :=
by sorry

end NUMINAMATH_CALUDE_tan_one_condition_l2415_241519


namespace NUMINAMATH_CALUDE_rectangle_golden_ratio_l2415_241561

/-- A rectangle with sides x and y, where x > y, can be cut in half parallel to the longer side
    to produce scaled-down versions of the original if and only if x/y = √2 -/
theorem rectangle_golden_ratio (x y : ℝ) (h : x > y) (h' : x > 0) (h'' : y > 0) :
  (x / 2 : ℝ) / y = x / y ↔ x / y = Real.sqrt 2 := by
sorry

end NUMINAMATH_CALUDE_rectangle_golden_ratio_l2415_241561


namespace NUMINAMATH_CALUDE_cos_squared_alpha_plus_pi_fourth_l2415_241589

theorem cos_squared_alpha_plus_pi_fourth (α : ℝ) (h : Real.sin (2 * α) = 2/3) :
  Real.cos (α + Real.pi/4)^2 = 1/6 := by
  sorry

end NUMINAMATH_CALUDE_cos_squared_alpha_plus_pi_fourth_l2415_241589


namespace NUMINAMATH_CALUDE_games_in_own_division_l2415_241560

/-- Represents a baseball league with specific game scheduling rules -/
structure BaseballLeague where
  N : ℕ  -- Number of games against each team in own division
  M : ℕ  -- Number of games against each team in other division
  h1 : N > 2 * M
  h2 : M > 4
  h3 : 4 * N + 5 * M = 82

/-- The number of games a team plays within its own division is 52 -/
theorem games_in_own_division (league : BaseballLeague) : 4 * league.N = 52 := by
  sorry

end NUMINAMATH_CALUDE_games_in_own_division_l2415_241560


namespace NUMINAMATH_CALUDE_x_squared_eq_one_is_quadratic_l2415_241518

/-- Definition of a quadratic equation in one variable x -/
def is_quadratic_equation (f : ℝ → ℝ) : Prop :=
  ∃ (a b c : ℝ), a ≠ 0 ∧ (∀ x, f x = a * x^2 + b * x + c)

/-- The equation x² = 1 -/
def f (x : ℝ) : ℝ := x^2 - 1

/-- Theorem: x² = 1 is a quadratic equation -/
theorem x_squared_eq_one_is_quadratic : is_quadratic_equation f := by
  sorry


end NUMINAMATH_CALUDE_x_squared_eq_one_is_quadratic_l2415_241518


namespace NUMINAMATH_CALUDE_custom_op_one_neg_three_l2415_241520

-- Define the custom operation ※
def custom_op (a b : ℤ) : ℤ := 2 * a * b - b^2

-- Theorem statement
theorem custom_op_one_neg_three : custom_op 1 (-3) = -15 := by
  sorry

end NUMINAMATH_CALUDE_custom_op_one_neg_three_l2415_241520


namespace NUMINAMATH_CALUDE_apple_redistribution_theorem_l2415_241578

/-- Represents the state of apples in baskets -/
structure AppleBaskets where
  total_apples : ℕ
  baskets : List ℕ
  deriving Repr

/-- Checks if all non-empty baskets have the same number of apples -/
def all_equal (ab : AppleBaskets) : Prop :=
  let non_empty := ab.baskets.filter (· > 0)
  non_empty.all (· = non_empty.head!)

/-- Checks if the total number of apples is at least 100 -/
def at_least_100 (ab : AppleBaskets) : Prop :=
  ab.total_apples ≥ 100

/-- Represents a valid redistribution of apples -/
def is_valid_redistribution (initial final : AppleBaskets) : Prop :=
  final.total_apples ≤ initial.total_apples ∧
  final.baskets.length ≤ initial.baskets.length

/-- The main theorem to prove -/
theorem apple_redistribution_theorem (initial : AppleBaskets) :
  initial.total_apples = 2000 →
  ∃ (final : AppleBaskets), 
    is_valid_redistribution initial final ∧
    all_equal final ∧
    at_least_100 final := by
  sorry

end NUMINAMATH_CALUDE_apple_redistribution_theorem_l2415_241578


namespace NUMINAMATH_CALUDE_trajectory_and_constant_product_l2415_241534

-- Define the points and circles
def G : ℝ × ℝ := (5, 4)
def A : ℝ × ℝ := (1, 0)

def C1 (x y : ℝ) : Prop := (x - 1)^2 + (y - 4)^2 = 25

-- Define the lines
def l1 (k x y : ℝ) : Prop := k * x - y - k = 0
def l2 (x y : ℝ) : Prop := x + 2 * y + 2 = 0

-- Define the trajectory C2
def C2 (x y : ℝ) : Prop := (x - 3)^2 + (y - 4)^2 = 4

-- Define the theorem
theorem trajectory_and_constant_product :
  ∃ (M N : ℝ × ℝ) (k : ℝ),
    (∀ x y, C2 x y ↔ (∃ E F : ℝ × ℝ, C1 E.1 E.2 ∧ C1 F.1 F.2 ∧ 
      (x, y) = ((E.1 + F.1) / 2, (E.2 + F.2) / 2))) ∧
    l1 k M.1 M.2 ∧ 
    l1 k N.1 N.2 ∧ 
    l2 N.1 N.2 ∧
    C2 M.1 M.2 ∧
    (M.1 - A.1)^2 + (M.2 - A.2)^2 * ((N.1 - A.1)^2 + (N.2 - A.2)^2) = 36 :=
by sorry


end NUMINAMATH_CALUDE_trajectory_and_constant_product_l2415_241534


namespace NUMINAMATH_CALUDE_chip_sales_ratio_l2415_241531

/-- Represents the sales data for a convenience store's chip sales over a month. -/
structure ChipSales where
  total : ℕ
  first_week : ℕ
  third_week : ℕ
  fourth_week : ℕ

/-- Calculates the ratio of second week sales to first week sales. -/
def sales_ratio (sales : ChipSales) : ℚ :=
  let second_week := sales.total - sales.first_week - sales.third_week - sales.fourth_week
  (second_week : ℚ) / sales.first_week

/-- Theorem stating that given the specific sales conditions, the ratio of second week to first week sales is 3:1. -/
theorem chip_sales_ratio :
  ∀ (sales : ChipSales),
    sales.total = 100 ∧
    sales.first_week = 15 ∧
    sales.third_week = 20 ∧
    sales.fourth_week = 20 →
    sales_ratio sales = 3 := by
  sorry

end NUMINAMATH_CALUDE_chip_sales_ratio_l2415_241531


namespace NUMINAMATH_CALUDE_min_value_of_expression_equality_condition_exists_l2415_241595

theorem min_value_of_expression (x : ℝ) (h : x > 1) : x + 1 / (x - 1) ≥ 3 := by
  sorry

theorem equality_condition_exists : ∃ x > 1, x + 1 / (x - 1) = 3 := by
  sorry

end NUMINAMATH_CALUDE_min_value_of_expression_equality_condition_exists_l2415_241595


namespace NUMINAMATH_CALUDE_assignment_count_l2415_241550

theorem assignment_count : 
  (∀ n : ℕ, n = 8 → ∀ k : ℕ, k = 4 → (k : ℕ) ^ n = 65536) := by sorry

end NUMINAMATH_CALUDE_assignment_count_l2415_241550


namespace NUMINAMATH_CALUDE_line_equation_correct_l2415_241571

/-- A line in the 2D plane represented by its slope and a point it passes through. -/
structure Line where
  slope : ℝ
  point : ℝ × ℝ

/-- The equation of a line in slope-intercept form (y = mx + b). -/
def lineEquation (l : Line) : ℝ → ℝ := fun x => l.slope * x + (l.point.2 - l.slope * l.point.1)

theorem line_equation_correct (l : Line) : 
  l.slope = 3 ∧ l.point = (-2, 0) → lineEquation l = fun x => 3 * x + 6 := by
  sorry

end NUMINAMATH_CALUDE_line_equation_correct_l2415_241571


namespace NUMINAMATH_CALUDE_homework_problems_per_page_l2415_241509

/-- Given a student with homework pages and total problems, prove the number of problems per page. -/
theorem homework_problems_per_page (math_pages reading_pages total_pages total_problems : ℕ)
  (h1 : math_pages + reading_pages = total_pages)
  (h2 : total_pages * (total_problems / total_pages) = total_problems)
  (h3 : total_pages > 0)
  : total_problems / total_pages = 5 :=
by sorry

end NUMINAMATH_CALUDE_homework_problems_per_page_l2415_241509


namespace NUMINAMATH_CALUDE_cube_volume_from_surface_area_l2415_241587

theorem cube_volume_from_surface_area :
  ∀ (surface_area : ℝ) (volume : ℝ),
    surface_area = 384 →
    volume = (surface_area / 6) ^ (3/2) →
    volume = 512 := by
  sorry

end NUMINAMATH_CALUDE_cube_volume_from_surface_area_l2415_241587


namespace NUMINAMATH_CALUDE_fire_safety_competition_l2415_241552

theorem fire_safety_competition (p_a1 p_a2 p_b1 p_b2 : ℝ) 
  (h_p_a1 : p_a1 = 4/5)
  (h_p_a2 : p_a2 = 2/3)
  (h_p_b1 : p_b1 = 3/5)
  (h_p_b2 : p_b2 = 3/4)
  (h_independence : True) : -- We assume independence, but don't need to express it formally here
  (p_a1 * (1 - p_a2) + (1 - p_a1) * p_a2 = 2/5) ∧ 
  (1 - (1 - p_a1 * p_a2) * (1 - p_b1 * p_b2) = 223/300) := by
  sorry

end NUMINAMATH_CALUDE_fire_safety_competition_l2415_241552


namespace NUMINAMATH_CALUDE_three_numbers_theorem_l2415_241524

theorem three_numbers_theorem (x y z : ℝ) 
  (h1 : (x + y + z)^2 = x^2 + y^2 + z^2)
  (h2 : x * y = z^2) :
  (x = 0 ∧ z = 0) ∨ (y = 0 ∧ z = 0) :=
by sorry

end NUMINAMATH_CALUDE_three_numbers_theorem_l2415_241524


namespace NUMINAMATH_CALUDE_average_speeding_percentage_l2415_241543

def zone_a_speeding_percentage : ℝ := 30
def zone_b_speeding_percentage : ℝ := 20
def zone_c_speeding_percentage : ℝ := 25

def number_of_zones : ℕ := 3

theorem average_speeding_percentage :
  (zone_a_speeding_percentage + zone_b_speeding_percentage + zone_c_speeding_percentage) / number_of_zones = 25 := by
  sorry

end NUMINAMATH_CALUDE_average_speeding_percentage_l2415_241543


namespace NUMINAMATH_CALUDE_arithmetic_sequence_property_l2415_241569

def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) - a n = d

theorem arithmetic_sequence_property 
  (a : ℕ → ℝ) 
  (h_arith : arithmetic_sequence a) 
  (h_a5 : a 5 = 2) : 
  a 4 - a 5 + a 6 = 2 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_property_l2415_241569


namespace NUMINAMATH_CALUDE_ratio_equality_l2415_241526

theorem ratio_equality : (2024^2 - 2017^2) / (2031^2 - 2010^2) = 1/3 := by sorry

end NUMINAMATH_CALUDE_ratio_equality_l2415_241526


namespace NUMINAMATH_CALUDE_line_through_points_with_45_degree_angle_l2415_241515

/-- Given a line passing through points (-1, 3) and (2, a) with an inclination angle of 45°, prove that a = 6 -/
theorem line_through_points_with_45_degree_angle (a : ℝ) : 
  (∃ (line : ℝ → ℝ), 
    line (-1) = 3 ∧ 
    line 2 = a ∧ 
    (∀ x y : ℝ, y = line x → (y - 3) / (x - (-1)) = 1)) → 
  a = 6 := by
sorry

end NUMINAMATH_CALUDE_line_through_points_with_45_degree_angle_l2415_241515


namespace NUMINAMATH_CALUDE_ralph_tv_hours_l2415_241566

/-- The number of hours Ralph watches TV on weekdays (Monday to Friday) -/
def weekday_hours : ℕ := 4

/-- The number of hours Ralph watches TV on weekend days (Saturday and Sunday) -/
def weekend_hours : ℕ := 6

/-- The number of weekdays in a week -/
def weekdays : ℕ := 5

/-- The number of weekend days in a week -/
def weekend_days : ℕ := 2

/-- The total number of hours Ralph watches TV in one week -/
def total_hours : ℕ := weekday_hours * weekdays + weekend_hours * weekend_days

theorem ralph_tv_hours : total_hours = 32 := by
  sorry

end NUMINAMATH_CALUDE_ralph_tv_hours_l2415_241566


namespace NUMINAMATH_CALUDE_image_of_negative_one_two_l2415_241548

-- Define the set of real pairs
def RealPair := ℝ × ℝ

-- Define the mapping f
def f (p : RealPair) : RealPair :=
  (p.1 - p.2, p.1 + p.2)

-- Theorem statement
theorem image_of_negative_one_two :
  f (-1, 2) = (-3, 1) := by
  sorry

end NUMINAMATH_CALUDE_image_of_negative_one_two_l2415_241548


namespace NUMINAMATH_CALUDE_derek_age_is_42_l2415_241574

-- Define the ages as natural numbers
def anne_age : ℕ := 36
def brianna_age : ℕ := (2 * anne_age) / 3
def caitlin_age : ℕ := brianna_age - 3
def derek_age : ℕ := 2 * caitlin_age

-- Theorem to prove Derek's age is 42
theorem derek_age_is_42 : derek_age = 42 := by
  sorry

end NUMINAMATH_CALUDE_derek_age_is_42_l2415_241574


namespace NUMINAMATH_CALUDE_rahul_deepak_age_ratio_l2415_241505

/-- Given that Rahul will be 26 years old in 10 years and Deepak is currently 8 years old,
    prove that the ratio of Rahul's age to Deepak's age is 2:1. -/
theorem rahul_deepak_age_ratio :
  ∀ (rahul_age deepak_age : ℕ),
    rahul_age + 10 = 26 →
    deepak_age = 8 →
    rahul_age / deepak_age = 2 := by
  sorry

end NUMINAMATH_CALUDE_rahul_deepak_age_ratio_l2415_241505


namespace NUMINAMATH_CALUDE_boat_speed_in_still_water_l2415_241512

/-- Proves that a boat traveling 54 km downstream in 3 hours and 54 km upstream in 9 hours has a speed of 12 km/hr in still water. -/
theorem boat_speed_in_still_water : 
  ∀ (v_b v_r : ℝ), 
    v_b > 0 → 
    v_r > 0 → 
    v_b + v_r = 54 / 3 → 
    v_b - v_r = 54 / 9 → 
    v_b = 12 := by
  sorry

#check boat_speed_in_still_water

end NUMINAMATH_CALUDE_boat_speed_in_still_water_l2415_241512


namespace NUMINAMATH_CALUDE_number_relationship_l2415_241554

theorem number_relationship (x m : ℚ) : 
  x = 25 / 3 → 
  (3 * x + 15 = m * x - 10) → 
  m = 6 := by
  sorry

end NUMINAMATH_CALUDE_number_relationship_l2415_241554


namespace NUMINAMATH_CALUDE_product_sum_fractions_l2415_241551

theorem product_sum_fractions : (3 * 4 * 5) * (1/3 + 1/4 + 1/5) = 47 := by
  sorry

end NUMINAMATH_CALUDE_product_sum_fractions_l2415_241551


namespace NUMINAMATH_CALUDE_cylinder_volume_from_unfolded_surface_l2415_241532

/-- 
Given a cylinder whose lateral surface unfolds into a square with side length 1,
the volume of the cylinder is 1/(4π).
-/
theorem cylinder_volume_from_unfolded_surface (r h : ℝ) : 
  (2 * π * r = 1) → (h = 1) → (π * r^2 * h = 1 / (4 * π)) := by
  sorry

end NUMINAMATH_CALUDE_cylinder_volume_from_unfolded_surface_l2415_241532


namespace NUMINAMATH_CALUDE_arrangement_equality_l2415_241516

theorem arrangement_equality (n : ℕ) (r₁ r₂ c₁ c₂ : ℕ) 
  (h₁ : n = r₁ * c₁)
  (h₂ : n = r₂ * c₂)
  (h₃ : n = 48)
  (h₄ : r₁ = 6)
  (h₅ : c₁ = 8)
  (h₆ : r₂ = 2)
  (h₇ : c₂ = 24) :
  Nat.factorial n = Nat.factorial n :=
by sorry

end NUMINAMATH_CALUDE_arrangement_equality_l2415_241516


namespace NUMINAMATH_CALUDE_jerry_original_butterflies_l2415_241590

/-- The number of butterflies Jerry let go -/
def butterflies_released : ℕ := 11

/-- The number of butterflies Jerry still has -/
def butterflies_remaining : ℕ := 82

/-- The original number of butterflies Jerry had -/
def original_butterflies : ℕ := butterflies_released + butterflies_remaining

theorem jerry_original_butterflies : original_butterflies = 93 := by
  sorry

end NUMINAMATH_CALUDE_jerry_original_butterflies_l2415_241590


namespace NUMINAMATH_CALUDE_quadratic_roots_sum_of_squares_l2415_241525

theorem quadratic_roots_sum_of_squares (α β : ℝ) : 
  (∀ x, x^2 - 7*x + 3 = 0 ↔ x = α ∨ x = β) →
  α^2 + β^2 = 43 := by
sorry

end NUMINAMATH_CALUDE_quadratic_roots_sum_of_squares_l2415_241525


namespace NUMINAMATH_CALUDE_rectangular_prism_volume_l2415_241597

/-- Conversion factor from yards to feet -/
def yards_to_feet : ℚ := 3

/-- Length of the rectangular prism in yards -/
def length_yd : ℚ := 1

/-- Width of the rectangular prism in yards -/
def width_yd : ℚ := 2

/-- Height of the rectangular prism in yards -/
def height_yd : ℚ := 3

/-- Volume of a rectangular prism in cubic feet -/
def volume_cubic_feet (l w h : ℚ) : ℚ := l * w * h * (yards_to_feet ^ 3)

/-- Theorem stating that the volume of the given rectangular prism is 162 cubic feet -/
theorem rectangular_prism_volume :
  volume_cubic_feet length_yd width_yd height_yd = 162 := by
  sorry

end NUMINAMATH_CALUDE_rectangular_prism_volume_l2415_241597


namespace NUMINAMATH_CALUDE_thirtiethDigitOf_1_11_plus_1_13_l2415_241572

/-- The decimal representation of a rational number -/
def decimalRepresentation (q : ℚ) : ℕ → ℕ := sorry

/-- The nth digit after the decimal point in the sum of the decimal representations of two rational numbers -/
def nthDigitAfterDecimal (q₁ q₂ : ℚ) (n : ℕ) : ℕ := sorry

/-- Theorem: The 30th digit after the decimal point in the sum of 1/11 and 1/13 is 2 -/
theorem thirtiethDigitOf_1_11_plus_1_13 : 
  nthDigitAfterDecimal (1/11) (1/13) 30 = 2 := by sorry

end NUMINAMATH_CALUDE_thirtiethDigitOf_1_11_plus_1_13_l2415_241572


namespace NUMINAMATH_CALUDE_no_decreasing_nat_function_exists_decreasing_int_function_l2415_241546

-- Define φ as a function from ℕ to ℕ
variable (φ : ℕ → ℕ)

-- Theorem 1: No such function f : ℕ → ℕ exists
theorem no_decreasing_nat_function : 
  ¬ ∃ f : ℕ → ℕ, ∀ x : ℕ, f x > f (φ x) := by sorry

-- Theorem 2: Such a function f : ℕ → ℤ exists
theorem exists_decreasing_int_function : 
  ∃ f : ℕ → ℤ, ∀ x : ℕ, f x > f (φ x) := by sorry

end NUMINAMATH_CALUDE_no_decreasing_nat_function_exists_decreasing_int_function_l2415_241546


namespace NUMINAMATH_CALUDE_hairdresser_initial_amount_l2415_241540

def hairdresser_savings (initial_amount : ℕ) : Prop :=
  let first_year_spent := initial_amount / 2
  let second_year_spent := initial_amount / 3
  let third_year_spent := 200
  let remaining := initial_amount - first_year_spent - second_year_spent - third_year_spent
  (remaining = 50) ∧ 
  (first_year_spent = initial_amount / 2) ∧
  (second_year_spent = initial_amount / 3) ∧
  (third_year_spent = 200)

theorem hairdresser_initial_amount : 
  ∃ (initial_amount : ℕ), hairdresser_savings initial_amount ∧ initial_amount = 1500 :=
by
  sorry

end NUMINAMATH_CALUDE_hairdresser_initial_amount_l2415_241540


namespace NUMINAMATH_CALUDE_units_digit_sum_factorials_15_l2415_241527

def factorial (n : ℕ) : ℕ := (List.range n).foldl (· * ·) 1

def sum_factorials (n : ℕ) : ℕ := (List.range n).map factorial |>.sum

def units_digit (n : ℕ) : ℕ := n % 10

theorem units_digit_sum_factorials_15 :
  units_digit (sum_factorials 15) = 3 := by
  sorry

end NUMINAMATH_CALUDE_units_digit_sum_factorials_15_l2415_241527


namespace NUMINAMATH_CALUDE_price_adjustment_percentage_l2415_241538

theorem price_adjustment_percentage (P : ℝ) (x : ℝ) (h : P > 0) :
  P * (1 + x / 100) * (1 - x / 100) = 0.75 * P →
  x = 50 := by
sorry

end NUMINAMATH_CALUDE_price_adjustment_percentage_l2415_241538


namespace NUMINAMATH_CALUDE_parabolas_intersection_l2415_241596

-- Define the two parabolas
def parabola1 (x : ℝ) : ℝ := 3 * x^2 - 12 * x - 15
def parabola2 (x : ℝ) : ℝ := x^2 - 6 * x + 11

-- Define the intersection points
def intersection_points : Set ℝ := {x | parabola1 x = parabola2 x}

-- Theorem statement
theorem parabolas_intersection :
  ∃ (x1 x2 : ℝ), x1 ∈ intersection_points ∧ x2 ∈ intersection_points ∧
  x1 = (3 + Real.sqrt 61) / 2 ∧ x2 = (3 - Real.sqrt 61) / 2 :=
sorry

end NUMINAMATH_CALUDE_parabolas_intersection_l2415_241596


namespace NUMINAMATH_CALUDE_susans_gift_is_eight_l2415_241577

/-- The number of apples Sean had initially -/
def initial_apples : ℕ := 9

/-- The total number of apples Sean had after Susan's gift -/
def total_apples : ℕ := 17

/-- The number of apples Susan gave to Sean -/
def susans_gift : ℕ := total_apples - initial_apples

theorem susans_gift_is_eight : susans_gift = 8 := by
  sorry

end NUMINAMATH_CALUDE_susans_gift_is_eight_l2415_241577


namespace NUMINAMATH_CALUDE_carbonated_water_percentage_l2415_241545

/-- Represents a solution with percentages of lemonade and carbonated water -/
structure Solution where
  lemonade : ℝ
  carbonated : ℝ
  sum_to_one : lemonade + carbonated = 1

/-- Represents a mixture of two solutions -/
structure Mixture where
  solution1 : Solution
  solution2 : Solution
  proportion1 : ℝ
  proportion2 : ℝ
  sum_to_one : proportion1 + proportion2 = 1

theorem carbonated_water_percentage
  (sol1 : Solution)
  (sol2 : Solution)
  (mix : Mixture)
  (h1 : sol1.carbonated = 0.8)
  (h2 : sol2.lemonade = 0.45)
  (h3 : mix.solution1 = sol1)
  (h4 : mix.solution2 = sol2)
  (h5 : mix.proportion1 = 0.5)
  (h6 : mix.proportion2 = 0.5)
  (h7 : mix.proportion1 * sol1.carbonated + mix.proportion2 * sol2.carbonated = 0.675) :
  sol2.carbonated = 0.55 := by
  sorry


end NUMINAMATH_CALUDE_carbonated_water_percentage_l2415_241545


namespace NUMINAMATH_CALUDE_factorization_sum_l2415_241522

theorem factorization_sum (a b c : ℤ) : 
  (∀ x : ℝ, x^2 + 12*x + 35 = (x + a)*(x + b)) → 
  (∀ x : ℝ, x^2 - 15*x + 56 = (x - b)*(x - c)) → 
  a + b + c = 20 := by
sorry

end NUMINAMATH_CALUDE_factorization_sum_l2415_241522


namespace NUMINAMATH_CALUDE_train_length_l2415_241500

/-- The length of a train given its speed and time to cross a pole -/
theorem train_length (speed : ℝ) (time : ℝ) : speed = 45 → time = 16 → speed * time * (1000 / 3600) = 200 := by
  sorry

end NUMINAMATH_CALUDE_train_length_l2415_241500
