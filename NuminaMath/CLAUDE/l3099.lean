import Mathlib

namespace NUMINAMATH_CALUDE_merchant_discount_percentage_l3099_309919

/-- Calculates the discount percentage for a merchant's pricing strategy -/
theorem merchant_discount_percentage
  (markup_percentage : ℝ)
  (profit_percentage : ℝ)
  (h_markup : markup_percentage = 50)
  (h_profit : profit_percentage = 35)
  : ∃ (discount_percentage : ℝ),
    discount_percentage = 10 ∧
    (1 + markup_percentage / 100) * (1 - discount_percentage / 100) = 1 + profit_percentage / 100 :=
by sorry

end NUMINAMATH_CALUDE_merchant_discount_percentage_l3099_309919


namespace NUMINAMATH_CALUDE_floor_painting_cost_l3099_309998

theorem floor_painting_cost 
  (length : ℝ) 
  (paint_rate : ℝ) 
  (length_ratio : ℝ) : 
  length = 12.24744871391589 →
  paint_rate = 2 →
  length_ratio = 3 →
  (length * (length / length_ratio)) * paint_rate = 100 := by
  sorry

end NUMINAMATH_CALUDE_floor_painting_cost_l3099_309998


namespace NUMINAMATH_CALUDE_parabola_directrix_l3099_309981

/-- A parabola with equation y = 1/4 * x^2 has a directrix with equation y = -1 -/
theorem parabola_directrix (x y : ℝ) :
  y = (1/4) * x^2 → ∃ (k : ℝ), k = -1 ∧ (∀ (x₀ y₀ : ℝ), y₀ = k → 
    (x₀ - x)^2 + (y₀ - y)^2 = (y₀ - (y + 1/4))^2) := by
  sorry

end NUMINAMATH_CALUDE_parabola_directrix_l3099_309981


namespace NUMINAMATH_CALUDE_line_circle_intersection_and_dot_product_l3099_309938

-- Define the line l passing through A(0, 1) with slope k
def line_l (k : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.2 = k * p.1 + 1}

-- Define the circle C: (x-2)^2+(y-3)^2=1
def circle_C : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | (p.1 - 2)^2 + (p.2 - 3)^2 = 1}

-- Define the point A
def point_A : ℝ × ℝ := (0, 1)

-- Define the origin O
def origin : ℝ × ℝ := (0, 0)

-- Define the dot product of two vectors
def dot_product (v w : ℝ × ℝ) : ℝ :=
  v.1 * w.1 + v.2 * w.2

theorem line_circle_intersection_and_dot_product
  (k : ℝ) (M N : ℝ × ℝ) :
  (M ∈ line_l k ∧ M ∈ circle_C ∧ N ∈ line_l k ∧ N ∈ circle_C) →
  ((4 - Real.sqrt 7) / 3 < k ∧ k < (4 + Real.sqrt 7) / 3) ∧
  (dot_product (M.1 - point_A.1, M.2 - point_A.2) (N.1 - point_A.1, N.2 - point_A.2) = 7) ∧
  (dot_product (M.1 - origin.1, M.2 - origin.2) (N.1 - origin.1, N.2 - origin.2) = 12 →
    k = 1 ∧ line_l k = {p : ℝ × ℝ | p.1 - p.2 + 1 = 0}) := by
  sorry

end NUMINAMATH_CALUDE_line_circle_intersection_and_dot_product_l3099_309938


namespace NUMINAMATH_CALUDE_perpendicular_line_equation_l3099_309969

/-- A line passing through (1,1) and perpendicular to x+2y-3=0 has the equation y=2x-1 -/
theorem perpendicular_line_equation :
  ∀ (l : Set (ℝ × ℝ)),
  (∀ p : ℝ × ℝ, p ∈ l ↔ p.1 + 2 * p.2 - 3 = 0) →  -- Definition of line l'
  ((1, 1) ∈ l) →  -- l passes through (1,1)
  (∀ p q : ℝ × ℝ, p ∈ l → q ∈ l → p ≠ q → (p.1 - q.1) * (p.1 + 2 * p.2 - 3 - (q.1 + 2 * q.2 - 3)) = 0) →  -- l is perpendicular to l'
  (∀ p : ℝ × ℝ, p ∈ l ↔ p.2 = 2 * p.1 - 1) :=  -- l has equation y=2x-1
by sorry

end NUMINAMATH_CALUDE_perpendicular_line_equation_l3099_309969


namespace NUMINAMATH_CALUDE_office_network_connections_l3099_309942

/-- A network of switches where each switch connects to exactly four others. -/
structure SwitchNetwork where
  num_switches : ℕ
  connections_per_switch : ℕ
  connection_count : ℕ

/-- The theorem stating the correct number of connections in the given network. -/
theorem office_network_connections (network : SwitchNetwork)
  (h1 : network.num_switches = 30)
  (h2 : network.connections_per_switch = 4) :
  network.connection_count = 60 := by
  sorry

end NUMINAMATH_CALUDE_office_network_connections_l3099_309942


namespace NUMINAMATH_CALUDE_distance_to_focus_l3099_309977

/-- A point on a parabola -/
structure PointOnParabola where
  x : ℝ
  y : ℝ
  on_parabola : y^2 = 4*x

/-- The distance between a point and a vertical line -/
def distance_to_vertical_line (p : ℝ × ℝ) (line_x : ℝ) : ℝ :=
  |p.1 - line_x|

/-- The focus of the parabola y^2 = 4x -/
def parabola_focus : ℝ × ℝ := (1, 0)

theorem distance_to_focus (P : PointOnParabola) 
  (h : distance_to_vertical_line (P.x, P.y) (-2) = 6) : 
  distance_to_vertical_line (P.x, P.y) (parabola_focus.1) = 5 := by
  sorry

end NUMINAMATH_CALUDE_distance_to_focus_l3099_309977


namespace NUMINAMATH_CALUDE_employee_salary_l3099_309993

theorem employee_salary (total_salary : ℝ) (m_salary_percentage : ℝ) 
  (h1 : total_salary = 616)
  (h2 : m_salary_percentage = 120 / 100) : 
  ∃ (n_salary : ℝ), 
    n_salary + m_salary_percentage * n_salary = total_salary ∧ 
    n_salary = 280 := by
  sorry

end NUMINAMATH_CALUDE_employee_salary_l3099_309993


namespace NUMINAMATH_CALUDE_same_color_probability_l3099_309902

def total_balls : ℕ := 3
def red_balls : ℕ := 2
def white_balls : ℕ := 1

def prob_red : ℚ := red_balls / total_balls
def prob_white : ℚ := white_balls / total_balls

theorem same_color_probability :
  let prob_same_color := prob_red^2 + prob_white^2
  prob_same_color = 5/9 := by sorry

end NUMINAMATH_CALUDE_same_color_probability_l3099_309902


namespace NUMINAMATH_CALUDE_total_weekly_pay_l3099_309930

/-- The weekly pay of employee Y in Rupees -/
def y_pay : ℝ := 150

/-- The weekly pay of employee X as a percentage of Y's pay -/
def x_pay_percentage : ℝ := 120

/-- Theorem stating the total weekly pay of both employees -/
theorem total_weekly_pay : 
  y_pay + (x_pay_percentage / 100 * y_pay) = 330 := by sorry

end NUMINAMATH_CALUDE_total_weekly_pay_l3099_309930


namespace NUMINAMATH_CALUDE_pig_purchase_equation_l3099_309991

/-- Represents a group purchase of pigs -/
structure PigPurchase where
  numPeople : ℕ
  excessAmount : ℕ
  exactAmount : ℕ

/-- The equation for the pig purchase problem is correct -/
theorem pig_purchase_equation (p : PigPurchase) 
  (h1 : p.numPeople * p.excessAmount - p.numPeople * p.exactAmount = p.excessAmount) 
  (h2 : p.excessAmount = 100) 
  (h3 : p.exactAmount = 90) : 
  100 * p.numPeople - 90 * p.numPeople = 100 := by
  sorry

end NUMINAMATH_CALUDE_pig_purchase_equation_l3099_309991


namespace NUMINAMATH_CALUDE_temperature_function_and_max_l3099_309924

-- Define the temperature function
def T (a b c d : ℝ) (t : ℝ) : ℝ := a * t^3 + b * t^2 + c * t + d

-- Define the derivative of the temperature function
def T_prime (a b c : ℝ) (t : ℝ) : ℝ := 3 * a * t^2 + 2 * b * t + c

-- State the theorem
theorem temperature_function_and_max (a b c d : ℝ) 
  (ha : a ≠ 0)
  (h1 : T a b c d (-4) = 8)
  (h2 : T a b c d 0 = 60)
  (h3 : T a b c d 1 = 58)
  (h4 : T_prime a b c (-4) = T_prime a b c 4) :
  (∃ (t : ℝ), t ≥ -2 ∧ t ≤ 2 ∧ 
    (∀ (s : ℝ), s ≥ -2 ∧ s ≤ 2 → T 1 0 (-3) 60 t ≥ T 1 0 (-3) 60 s) ∧
    T 1 0 (-3) 60 t = 62) ∧
  (∀ (t : ℝ), T 1 0 (-3) 60 t = t^3 - 3*t + 60) := by
  sorry


end NUMINAMATH_CALUDE_temperature_function_and_max_l3099_309924


namespace NUMINAMATH_CALUDE_intersection_length_l3099_309939

/-- The length of the line segment formed by the intersection of y = x + 1 and x²/4 + y²/3 = 1 is 24/7 -/
theorem intersection_length :
  let l : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.2 = p.1 + 1}
  let C : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.1^2/4 + p.2^2/3 = 1}
  let A : Set (ℝ × ℝ) := l ∩ C
  ∃ p q : ℝ × ℝ, p ∈ A ∧ q ∈ A ∧ p ≠ q ∧ ‖p - q‖ = 24/7 := by
  sorry

end NUMINAMATH_CALUDE_intersection_length_l3099_309939


namespace NUMINAMATH_CALUDE_sum_of_coefficients_is_negative_three_l3099_309965

/-- The polynomial x^3 - 7x^2 + 12x - 18 -/
def p (x : ℝ) : ℝ := x^3 - 7*x^2 + 12*x - 18

/-- The sum of the kth powers of the roots of p -/
def s (k : ℕ) : ℝ := sorry

/-- The recursive relationship for s_k -/
def recursive_relation (a b c : ℝ) : Prop :=
  ∀ k : ℕ, k ≥ 2 → s (k + 1) = a * s k + b * s (k - 1) + c * s (k - 2)

theorem sum_of_coefficients_is_negative_three :
  ∀ a b c : ℝ,
  (∃ α β γ : ℝ, p α = 0 ∧ p β = 0 ∧ p γ = 0) →
  s 0 = 3 →
  s 1 = 7 →
  s 2 = 13 →
  recursive_relation a b c →
  a + b + c = -3 := by sorry

end NUMINAMATH_CALUDE_sum_of_coefficients_is_negative_three_l3099_309965


namespace NUMINAMATH_CALUDE_det_A_formula_l3099_309974

theorem det_A_formula (n : ℕ) (h : n > 2) :
  let φ : ℝ := 2 * Real.pi / n
  let A : Matrix (Fin n) (Fin n) ℝ := λ i j =>
    if i = j then 1 + Real.cos (2 * φ * j) else Real.cos (φ * (i + j))
  Matrix.det A = -n^2 / 4 + 1 := by
  sorry

end NUMINAMATH_CALUDE_det_A_formula_l3099_309974


namespace NUMINAMATH_CALUDE_tied_rope_length_l3099_309940

/-- Calculates the length of a rope made by tying multiple shorter ropes together. -/
def ropeLength (n : ℕ) (ropeLength : ℕ) (knotReduction : ℕ) : ℕ :=
  n * ropeLength - (n - 1) * knotReduction

/-- Proves that tying 64 ropes of 25 cm each, with 3 cm reduction per knot, results in a 1411 cm rope. -/
theorem tied_rope_length :
  ropeLength 64 25 3 = 1411 := by
  sorry

end NUMINAMATH_CALUDE_tied_rope_length_l3099_309940


namespace NUMINAMATH_CALUDE_complement_P_intersect_Q_l3099_309912

-- Define the sets P and Q
def P : Set ℝ := {x : ℝ | x^2 - 2*x ≥ 0}
def Q : Set ℝ := {x : ℝ | 1 < x ∧ x ≤ 2}

-- State the theorem
theorem complement_P_intersect_Q : (P.compl ∩ Q) = Set.Ioo 1 2 := by
  sorry

end NUMINAMATH_CALUDE_complement_P_intersect_Q_l3099_309912


namespace NUMINAMATH_CALUDE_cone_volume_l3099_309943

/-- Given a cone whose lateral surface unfolds into a sector with radius 3 and central angle 2π/3,
    the volume of the cone is 2√2π/3 -/
theorem cone_volume (r l : ℝ) (h : ℝ) : 
  r > 0 → l > 0 → h > 0 →
  l = 3 →
  2 * π * r = 2 * π / 3 * l →
  h^2 + r^2 = l^2 →
  (1/3) * π * r^2 * h = (2 * Real.sqrt 2 / 3) * π := by
  sorry

end NUMINAMATH_CALUDE_cone_volume_l3099_309943


namespace NUMINAMATH_CALUDE_hyperbola_real_axis_length_l3099_309971

/-- A hyperbola with center at the origin and foci on the x-axis -/
structure Hyperbola where
  center : ℝ × ℝ
  foci_on_x_axis : Bool

/-- The length of a line segment -/
def length (a b : ℝ × ℝ) : ℝ := sorry

/-- The real axis of a hyperbola -/
def real_axis (h : Hyperbola) : ℝ := sorry

theorem hyperbola_real_axis_length
  (C : Hyperbola)
  (h_center : C.center = (0, 0))
  (h_foci : C.foci_on_x_axis = true)
  (A B : ℝ × ℝ)
  (h_intersect : A.1 = -4 ∧ B.1 = -4)
  (h_distance : length A B = 4) :
  real_axis C = 4 := by sorry

end NUMINAMATH_CALUDE_hyperbola_real_axis_length_l3099_309971


namespace NUMINAMATH_CALUDE_color_p_gon_l3099_309931

theorem color_p_gon (p a : ℕ) (hp : Nat.Prime p) :
  let total_colorings := a^p
  let monochromatic_colorings := a
  let distinct_non_monochromatic := (total_colorings - monochromatic_colorings) / p
  distinct_non_monochromatic + monochromatic_colorings = (a^p - a) / p + a := by
  sorry

end NUMINAMATH_CALUDE_color_p_gon_l3099_309931


namespace NUMINAMATH_CALUDE_hyperbola_and_related_ellipse_l3099_309948

/-- Given a hyperbola with equation x²/a² - y²/b² = 1, prove its asymptotes and related ellipse equations -/
theorem hyperbola_and_related_ellipse 
  (a b : ℝ) 
  (ha : a > 0) 
  (hb : b > 0) 
  (h_imag_axis : b = 1) 
  (h_focal_dist : 2 * Real.sqrt 3 = 2 * Real.sqrt (a^2 + b^2)) :
  (∃ (f : ℝ → ℝ), (∀ x, f x = Real.sqrt 2 / 2 * x) ∧ 
                   (∀ x, f (-x) = -f x) ∧ 
                   (∀ x y, y = f x ∨ y = -f x → x^2/a^2 - y^2/b^2 = 1)) ∧
  (∀ x y, x^2/3 + y^2 = 1 → 
    ∃ (t : ℝ), x = Real.sqrt 3 * Real.cos t ∧ y = Real.sin t) := by
  sorry

end NUMINAMATH_CALUDE_hyperbola_and_related_ellipse_l3099_309948


namespace NUMINAMATH_CALUDE_balloon_count_l3099_309911

-- Define the number of balloons for each person
def alyssa_balloons : ℕ := 37
def sandy_balloons : ℕ := 28
def sally_balloons : ℕ := 39

-- Define the total number of balloons
def total_balloons : ℕ := alyssa_balloons + sandy_balloons + sally_balloons

-- Theorem to prove
theorem balloon_count : total_balloons = 104 := by
  sorry

end NUMINAMATH_CALUDE_balloon_count_l3099_309911


namespace NUMINAMATH_CALUDE_prob_at_least_one_unqualified_is_correct_l3099_309937

/-- The total number of products -/
def total_products : ℕ := 6

/-- The number of qualified products -/
def qualified_products : ℕ := 4

/-- The number of unqualified products -/
def unqualified_products : ℕ := 2

/-- The number of products randomly selected -/
def selected_products : ℕ := 2

/-- The probability of selecting at least one unqualified product -/
def prob_at_least_one_unqualified : ℚ := 3/5

theorem prob_at_least_one_unqualified_is_correct :
  (1 : ℚ) - (Nat.choose qualified_products selected_products : ℚ) / (Nat.choose total_products selected_products : ℚ) = prob_at_least_one_unqualified :=
sorry

end NUMINAMATH_CALUDE_prob_at_least_one_unqualified_is_correct_l3099_309937


namespace NUMINAMATH_CALUDE_factor_tree_X_value_l3099_309933

theorem factor_tree_X_value :
  ∀ (X Y Z F G : ℕ),
    Y = 4 * F →
    F = 2 * F →
    Z = 7 * G →
    G = 7 * G →
    X = Y * Z →
    X = 392 := by
  sorry

end NUMINAMATH_CALUDE_factor_tree_X_value_l3099_309933


namespace NUMINAMATH_CALUDE_meal_cost_calculation_l3099_309982

theorem meal_cost_calculation (adults children : ℕ) (total_bill : ℚ) :
  adults = 2 →
  children = 5 →
  total_bill = 21 →
  total_bill / (adults + children : ℚ) = 3 := by
  sorry

end NUMINAMATH_CALUDE_meal_cost_calculation_l3099_309982


namespace NUMINAMATH_CALUDE_simplify_fraction_l3099_309905

theorem simplify_fraction : (36 : ℚ) / 4536 = 1 / 126 := by
  sorry

end NUMINAMATH_CALUDE_simplify_fraction_l3099_309905


namespace NUMINAMATH_CALUDE_smallest_x_absolute_value_equation_l3099_309972

theorem smallest_x_absolute_value_equation :
  ∃ x : ℝ, (∀ y : ℝ, |y + 3| = 15 → x ≤ y) ∧ |x + 3| = 15 :=
by sorry

end NUMINAMATH_CALUDE_smallest_x_absolute_value_equation_l3099_309972


namespace NUMINAMATH_CALUDE_geometric_arithmetic_sequence_ratio_l3099_309986

theorem geometric_arithmetic_sequence_ratio (x y z : ℝ) 
  (h1 : (4 * y) / (3 * x) = (5 * z) / (4 * y))  -- geometric sequence condition
  (h2 : 1 / y - 1 / x = 1 / z - 1 / y)         -- arithmetic sequence condition
  (h3 : x ≠ 0)
  (h4 : y ≠ 0)
  (h5 : z ≠ 0) :
  x / z + z / x = 34 / 15 := by
sorry

end NUMINAMATH_CALUDE_geometric_arithmetic_sequence_ratio_l3099_309986


namespace NUMINAMATH_CALUDE_fraction_equality_l3099_309984

theorem fraction_equality (p q : ℚ) (h : p / q = 4 / 5) : 
  18 / 7 + (2 * q - p) / ((14/5) * q) = 3 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_l3099_309984


namespace NUMINAMATH_CALUDE_log_equation_equivalence_l3099_309989

-- Define the logarithm function with base 10
noncomputable def lg (x : ℝ) : ℝ := Real.log x / Real.log 10

-- State the theorem
theorem log_equation_equivalence (x : ℝ) (h : x > 0) :
  lg x ^ 2 + lg (x ^ 2) = 0 ↔ lg x ^ 2 + 2 * lg x = 0 :=
by sorry

end NUMINAMATH_CALUDE_log_equation_equivalence_l3099_309989


namespace NUMINAMATH_CALUDE_sqrt_square_sum_zero_implies_diff_l3099_309927

theorem sqrt_square_sum_zero_implies_diff (x y : ℝ) : 
  Real.sqrt (8 - x) + (y + 4)^2 = 0 → x - y = 12 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_square_sum_zero_implies_diff_l3099_309927


namespace NUMINAMATH_CALUDE_largest_house_number_l3099_309958

def phone_number : List Nat := [3, 4, 6, 2, 8, 9, 0]

def sum_digits (num : List Nat) : Nat :=
  num.foldl (· + ·) 0

def is_distinct (num : List Nat) : Prop :=
  num.length = num.toFinset.card

def is_valid_house_number (num : List Nat) : Prop :=
  num.length = 4 ∧ is_distinct num ∧ sum_digits num = sum_digits phone_number

theorem largest_house_number :
  ∀ (house_num : List Nat),
    is_valid_house_number house_num →
    house_num.foldl (fun acc d => acc * 10 + d) 0 ≤ 9876 :=
by sorry

end NUMINAMATH_CALUDE_largest_house_number_l3099_309958


namespace NUMINAMATH_CALUDE_marble_transfer_result_l3099_309915

/-- Represents the marble transfer game between A and B -/
def marbleTransfer (a b n : ℕ) : Prop :=
  -- Initial conditions
  b < a ∧
  -- After 2n transfers, A has b marbles
  -- The ratio of initial marbles (a) to final marbles (b) is given by the formula
  (a : ℚ) / b = (2 * (4^n + 1)) / (1 - 4^n)

/-- Theorem stating the result of the marble transfer game -/
theorem marble_transfer_result {a b n : ℕ} (h : marbleTransfer a b n) :
  (a : ℚ) / b = (2 * (4^n + 1)) / (1 - 4^n) :=
by
  sorry

#check marble_transfer_result

end NUMINAMATH_CALUDE_marble_transfer_result_l3099_309915


namespace NUMINAMATH_CALUDE_total_unread_books_is_17_l3099_309925

/-- Represents a book series with total books and read books -/
structure BookSeries where
  total : ℕ
  read : ℕ

/-- Calculates the number of unread books in a series -/
def unread_books (series : BookSeries) : ℕ :=
  series.total - series.read

/-- The three book series -/
def series1 : BookSeries := ⟨14, 8⟩
def series2 : BookSeries := ⟨10, 5⟩
def series3 : BookSeries := ⟨18, 12⟩

/-- Theorem stating that the total number of unread books is 17 -/
theorem total_unread_books_is_17 :
  unread_books series1 + unread_books series2 + unread_books series3 = 17 := by
  sorry

end NUMINAMATH_CALUDE_total_unread_books_is_17_l3099_309925


namespace NUMINAMATH_CALUDE_hemisphere_surface_area_l3099_309923

theorem hemisphere_surface_area :
  ∀ (r : ℝ), 
    r > 0 →
    π * r^2 = 3 →
    let sphere_area := 4 * π * r^2
    let hemisphere_area := sphere_area / 2 + π * r^2
    hemisphere_area = 9 := by
  sorry

end NUMINAMATH_CALUDE_hemisphere_surface_area_l3099_309923


namespace NUMINAMATH_CALUDE_equation_root_of_increase_implies_m_equals_two_l3099_309966

-- Define the equation
def equation (x m : ℝ) : Prop := (x - 1) / (x - 3) = m / (x - 3)

-- Define the root of increase
def root_of_increase (x : ℝ) : Prop := x = 3

-- Theorem statement
theorem equation_root_of_increase_implies_m_equals_two :
  ∀ x m : ℝ, equation x m → root_of_increase x → m = 2 := by
  sorry

end NUMINAMATH_CALUDE_equation_root_of_increase_implies_m_equals_two_l3099_309966


namespace NUMINAMATH_CALUDE_power_log_fourth_root_l3099_309929

theorem power_log_fourth_root (x : ℝ) (h : x > 0) :
  ((625 ^ (Real.log x / Real.log 5)) ^ (1/4) : ℝ) = x :=
by sorry

end NUMINAMATH_CALUDE_power_log_fourth_root_l3099_309929


namespace NUMINAMATH_CALUDE_liouvilles_theorem_l3099_309956

theorem liouvilles_theorem (p m : ℕ) (h_prime : Nat.Prime p) (h_p_gt_5 : p > 5) (h_m_pos : m > 0) :
  (Nat.factorial (p - 1) + 1) ≠ p ^ m := by
  sorry

end NUMINAMATH_CALUDE_liouvilles_theorem_l3099_309956


namespace NUMINAMATH_CALUDE_range_of_a_l3099_309953

theorem range_of_a (a : ℝ) : (∀ x : ℝ, |x - 2| + |x - a| ≥ a) → a ≤ 1 := by
  sorry

end NUMINAMATH_CALUDE_range_of_a_l3099_309953


namespace NUMINAMATH_CALUDE_trigonometric_values_for_point_l3099_309970

theorem trigonometric_values_for_point (α : Real) :
  (∃ (r : Real), r > 0 ∧ r * Real.cos α = -4 ∧ r * Real.sin α = 3) →
  Real.sin α = 3/5 ∧ Real.cos α = -4/5 ∧ Real.tan α = -3/4 :=
by sorry

end NUMINAMATH_CALUDE_trigonometric_values_for_point_l3099_309970


namespace NUMINAMATH_CALUDE_muffin_banana_cost_ratio_l3099_309945

/-- The cost ratio of a muffin to a banana given Susie's and Nathan's purchases -/
theorem muffin_banana_cost_ratio :
  ∀ (muffin_cost banana_cost : ℝ),
  muffin_cost > 0 →
  banana_cost > 0 →
  5 * muffin_cost + 4 * banana_cost > 0 →
  4 * (5 * muffin_cost + 4 * banana_cost) = 4 * muffin_cost + 12 * banana_cost →
  muffin_cost / banana_cost = 4 :=
by
  sorry

end NUMINAMATH_CALUDE_muffin_banana_cost_ratio_l3099_309945


namespace NUMINAMATH_CALUDE_union_A_B_complement_A_l3099_309947

open Set

-- Define the universe set U as ℝ
def U : Set ℝ := univ

-- Define set A
def A : Set ℝ := {x : ℝ | -1 ≤ x ∧ x ≤ 3}

-- Define set B
def B : Set ℝ := {x : ℝ | x^2 < 4}

-- Theorem for the union of A and B
theorem union_A_B : A ∪ B = {x : ℝ | -2 < x ∧ x ≤ 3} := by sorry

-- Theorem for the complement of A with respect to U
theorem complement_A : (U \ A) = {x : ℝ | x < -1 ∨ x > 3} := by sorry

end NUMINAMATH_CALUDE_union_A_B_complement_A_l3099_309947


namespace NUMINAMATH_CALUDE_day_relationship_l3099_309968

/-- Represents days of the week -/
inductive DayOfWeek
  | Sunday
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday
  | Saturday

/-- Represents a specific day in a year -/
structure YearDay where
  year : Int
  day : Nat

/-- Function to determine the day of the week for a given YearDay -/
def dayOfWeek : YearDay → DayOfWeek := sorry

/-- Theorem stating the relationship between the days in different years -/
theorem day_relationship (N : Int) :
  dayOfWeek { year := N, day := 275 } = DayOfWeek.Thursday →
  dayOfWeek { year := N + 1, day := 215 } = DayOfWeek.Thursday →
  dayOfWeek { year := N - 1, day := 150 } = DayOfWeek.Saturday :=
by sorry

end NUMINAMATH_CALUDE_day_relationship_l3099_309968


namespace NUMINAMATH_CALUDE_cylinder_volume_l3099_309941

/-- The volume of a cylinder with base diameter and height both equal to 3 is (27/4)π. -/
theorem cylinder_volume (d h : ℝ) (hd : d = 3) (hh : h = 3) :
  let r := d / 2
  π * r^2 * h = (27 / 4) * π :=
by sorry

end NUMINAMATH_CALUDE_cylinder_volume_l3099_309941


namespace NUMINAMATH_CALUDE_intersection_implies_m_equals_one_l3099_309961

def A (m : ℝ) : Set ℝ := {3, 4, m^2 - 3*m - 1}
def B (m : ℝ) : Set ℝ := {2*m, -3}

theorem intersection_implies_m_equals_one :
  ∀ m : ℝ, A m ∩ B m = {-3} → m = 1 := by
  sorry

end NUMINAMATH_CALUDE_intersection_implies_m_equals_one_l3099_309961


namespace NUMINAMATH_CALUDE_prob_three_two_digit_out_of_five_l3099_309901

/-- A 12-sided die with numbers from 1 to 12 -/
def TwelveSidedDie : Type := Fin 12

/-- The probability of rolling a two-digit number on a single 12-sided die -/
def prob_two_digit : ℚ := 1 / 4

/-- The probability of rolling a one-digit number on a single 12-sided die -/
def prob_one_digit : ℚ := 3 / 4

/-- The number of 12-sided dice rolled -/
def num_dice : ℕ := 5

/-- The number of dice required to show a two-digit number -/
def required_two_digit : ℕ := 3

/-- Theorem stating the probability of exactly 3 out of 5 12-sided dice showing a two-digit number -/
theorem prob_three_two_digit_out_of_five :
  (Nat.choose num_dice required_two_digit : ℚ) *
  (prob_two_digit ^ required_two_digit) *
  (prob_one_digit ^ (num_dice - required_two_digit)) = 45 / 512 := by
  sorry

end NUMINAMATH_CALUDE_prob_three_two_digit_out_of_five_l3099_309901


namespace NUMINAMATH_CALUDE_even_function_condition_l3099_309973

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := (x - 1) * (x - a)

-- State the theorem
theorem even_function_condition (a : ℝ) : 
  (∀ x, f a x = f a (-x)) → a = -1 := by
  sorry

end NUMINAMATH_CALUDE_even_function_condition_l3099_309973


namespace NUMINAMATH_CALUDE_pants_price_decrease_percentage_l3099_309959

theorem pants_price_decrease_percentage (purchase_price : ℝ) (markup_percentage : ℝ) (gross_profit : ℝ) : 
  purchase_price = 81 →
  markup_percentage = 0.25 →
  gross_profit = 5.40 →
  let original_price := purchase_price / (1 - markup_percentage)
  let decreased_price := original_price - gross_profit
  let decrease_amount := original_price - decreased_price
  (decrease_amount / original_price) * 100 = 5 := by
  sorry

end NUMINAMATH_CALUDE_pants_price_decrease_percentage_l3099_309959


namespace NUMINAMATH_CALUDE_specific_pyramid_base_edge_length_l3099_309922

/-- A square pyramid with a sphere inside --/
structure PyramidWithSphere where
  pyramid_height : ℝ
  sphere_radius : ℝ
  sphere_tangent_to_faces : Bool
  sphere_contacts_base : Bool

/-- Calculates the edge length of the pyramid's base --/
def base_edge_length (p : PyramidWithSphere) : ℝ :=
  sorry

/-- Theorem stating the base edge length of the specific pyramid --/
theorem specific_pyramid_base_edge_length :
  let p : PyramidWithSphere := {
    pyramid_height := 9,
    sphere_radius := 3,
    sphere_tangent_to_faces := true,
    sphere_contacts_base := true
  }
  base_edge_length p = 4.5 := by
  sorry

end NUMINAMATH_CALUDE_specific_pyramid_base_edge_length_l3099_309922


namespace NUMINAMATH_CALUDE_magnitude_of_complex_number_l3099_309960

theorem magnitude_of_complex_number (z : ℂ) : z = (5 * Complex.I) / (2 + Complex.I) → Complex.abs z = Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_magnitude_of_complex_number_l3099_309960


namespace NUMINAMATH_CALUDE_diamond_zero_not_always_double_l3099_309928

-- Define the diamond operation
def diamond (x y : ℝ) : ℝ := 2 * |x - y|

-- Statement to prove
theorem diamond_zero_not_always_double : ¬ (∀ x : ℝ, diamond x 0 = 2 * x) := by
  sorry

end NUMINAMATH_CALUDE_diamond_zero_not_always_double_l3099_309928


namespace NUMINAMATH_CALUDE_specific_tank_insulation_cost_l3099_309964

/-- The cost to insulate a rectangular tank -/
def insulation_cost (length width height cost_per_sqft : ℝ) : ℝ :=
  (2 * (length * width + length * height + width * height)) * cost_per_sqft

/-- Theorem: The cost to insulate a specific rectangular tank -/
theorem specific_tank_insulation_cost :
  insulation_cost 4 5 3 20 = 1880 := by
  sorry

end NUMINAMATH_CALUDE_specific_tank_insulation_cost_l3099_309964


namespace NUMINAMATH_CALUDE_coin_flip_probability_difference_l3099_309909

-- Define a fair coin
def fair_coin_prob : ℚ := 1 / 2

-- Define the number of flips
def total_flips : ℕ := 5

-- Define the number of heads for the first probability
def heads_count_1 : ℕ := 3

-- Define the number of heads for the second probability
def heads_count_2 : ℕ := 5

-- Function to calculate binomial coefficient
def binomial (n k : ℕ) : ℕ := sorry

-- Function to calculate probability of exactly k heads in n flips
def prob_k_heads (n k : ℕ) (p : ℚ) : ℚ := 
  (binomial n k : ℚ) * p^k * (1 - p)^(n - k)

-- Theorem statement
theorem coin_flip_probability_difference : 
  (prob_k_heads total_flips heads_count_1 fair_coin_prob - 
   prob_k_heads total_flips heads_count_2 fair_coin_prob) = 9 / 32 := by sorry

end NUMINAMATH_CALUDE_coin_flip_probability_difference_l3099_309909


namespace NUMINAMATH_CALUDE_unique_value_in_set_l3099_309910

theorem unique_value_in_set (a : ℝ) : 1 ∈ ({a, a + 1, a^2} : Set ℝ) → a = -1 := by
  sorry

end NUMINAMATH_CALUDE_unique_value_in_set_l3099_309910


namespace NUMINAMATH_CALUDE_gcf_48_72_l3099_309955

theorem gcf_48_72 : Nat.gcd 48 72 = 24 := by
  sorry

end NUMINAMATH_CALUDE_gcf_48_72_l3099_309955


namespace NUMINAMATH_CALUDE_calculate_expression_solve_system_of_equations_l3099_309967

-- Problem 1
theorem calculate_expression : 
  Real.sqrt ((-4)^2) + 2 * (Real.sqrt 2 - 3) - |-(2 * Real.sqrt 2)| = -2 := by sorry

-- Problem 2
theorem solve_system_of_equations :
  ∃ (x y : ℝ), x / 2 + y / 3 = 4 ∧ x + 2 * y = 16 ∧ x = 4 ∧ y = 6 := by sorry

end NUMINAMATH_CALUDE_calculate_expression_solve_system_of_equations_l3099_309967


namespace NUMINAMATH_CALUDE_min_value_reciprocal_sum_l3099_309916

theorem min_value_reciprocal_sum (a b : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : a + 3 * b = 1) :
  (∀ x y : ℝ, x > 0 → y > 0 → x + 3 * y = 1 → 1 / a + 1 / b ≤ 1 / x + 1 / y) ∧
  (1 / a + 1 / b = 4 + 2 * Real.sqrt 3) :=
sorry

end NUMINAMATH_CALUDE_min_value_reciprocal_sum_l3099_309916


namespace NUMINAMATH_CALUDE_dress_price_calculation_l3099_309963

def calculate_final_price (original_price : ℝ) (initial_discount : ℝ) (additional_discount : ℝ) (store_credit : ℝ) (sales_tax : ℝ) : ℝ :=
  let price_after_initial_discount := original_price * (1 - initial_discount)
  let price_after_additional_discount := price_after_initial_discount * (1 - additional_discount)
  let price_after_credit := price_after_additional_discount - store_credit
  let final_price := price_after_credit * (1 + sales_tax)
  final_price

theorem dress_price_calculation :
  calculate_final_price 50 0.3 0.2 10 0.075 = 19.35 := by
  sorry

end NUMINAMATH_CALUDE_dress_price_calculation_l3099_309963


namespace NUMINAMATH_CALUDE_poster_area_l3099_309908

theorem poster_area (x y : ℕ) (h1 : x > 0) (h2 : y > 0) 
  (h3 : (3 * x + 4) * (y + 3) = 63) : x * y = 15 := by
  sorry

end NUMINAMATH_CALUDE_poster_area_l3099_309908


namespace NUMINAMATH_CALUDE_linear_function_problem_l3099_309962

/-- A linear function satisfying certain conditions -/
def f (x : ℝ) : ℝ := sorry

/-- The inverse of f -/
def f_inv (x : ℝ) : ℝ := sorry

theorem linear_function_problem :
  (∀ x y : ℝ, ∃ a b : ℝ, f x = a * x + b) →  -- f is linear
  (∀ x : ℝ, f x = 4 * f_inv x + 8) →         -- f(x) = 4f^(-1)(x) + 8
  (f 1 = 5) →                                -- f(1) = 5
  (f 2 = 20 / 3) :=                          -- f(2) = 20/3
by sorry

end NUMINAMATH_CALUDE_linear_function_problem_l3099_309962


namespace NUMINAMATH_CALUDE_ninth_minus_eighth_rectangle_tiles_l3099_309900

/-- The number of tiles in the nth rectangle of the sequence -/
def tiles (n : ℕ) : ℕ := 2 * n * n

/-- The difference in tiles between the 9th and 8th rectangles -/
def tile_difference : ℕ := tiles 9 - tiles 8

theorem ninth_minus_eighth_rectangle_tiles : tile_difference = 34 := by
  sorry

end NUMINAMATH_CALUDE_ninth_minus_eighth_rectangle_tiles_l3099_309900


namespace NUMINAMATH_CALUDE_binomial_expansion_coefficient_ratio_l3099_309921

theorem binomial_expansion_coefficient_ratio (n : ℕ) : 
  4 * (n.choose 2) = 7 * (2 * n) → n = 8 := by
  sorry

end NUMINAMATH_CALUDE_binomial_expansion_coefficient_ratio_l3099_309921


namespace NUMINAMATH_CALUDE_mouse_seeds_count_l3099_309951

/-- Represents the number of seeds per burrow for each animal -/
structure SeedsPerBurrow where
  mouse : ℕ
  rabbit : ℕ

/-- Represents the number of burrows for each animal -/
structure Burrows where
  mouse : ℕ
  rabbit : ℕ

/-- The total number of seeds hidden by each animal -/
def totalSeeds (spb : SeedsPerBurrow) (b : Burrows) : ℕ × ℕ :=
  (spb.mouse * b.mouse, spb.rabbit * b.rabbit)

theorem mouse_seeds_count (spb : SeedsPerBurrow) (b : Burrows) :
  spb.mouse = 4 →
  spb.rabbit = 6 →
  b.mouse = b.rabbit + 2 →
  (totalSeeds spb b).1 = (totalSeeds spb b).2 →
  (totalSeeds spb b).1 = 24 :=
by
  sorry

#check mouse_seeds_count

end NUMINAMATH_CALUDE_mouse_seeds_count_l3099_309951


namespace NUMINAMATH_CALUDE_distribute_five_into_four_l3099_309936

def distribute_objects (n : ℕ) (k : ℕ) : ℕ :=
  if n < k then 0
  else (n - k) + 1

theorem distribute_five_into_four :
  distribute_objects 5 4 = 4 := by
  sorry

end NUMINAMATH_CALUDE_distribute_five_into_four_l3099_309936


namespace NUMINAMATH_CALUDE_arctan_equation_solution_l3099_309903

theorem arctan_equation_solution :
  ∀ x : ℝ, 2 * Real.arctan (1/5) + Real.arctan (1/15) + Real.arctan (1/x) = π/3 → x = -49 := by
  sorry

end NUMINAMATH_CALUDE_arctan_equation_solution_l3099_309903


namespace NUMINAMATH_CALUDE_ellipse_circle_relation_l3099_309995

/-- An ellipse in the Cartesian coordinate system -/
structure Ellipse where
  a : ℝ
  b : ℝ
  h_pos : 0 < b ∧ b < a

/-- A circle in the Cartesian coordinate system -/
structure Circle where
  r : ℝ
  h_pos : r > 0

/-- A line in the Cartesian coordinate system -/
structure Line where
  k : ℝ
  m : ℝ

/-- A point in the Cartesian coordinate system -/
structure Point where
  x : ℝ
  y : ℝ

/-- Checks if a point is on an ellipse -/
def on_ellipse (p : Point) (e : Ellipse) : Prop :=
  p.x^2 / e.a^2 + p.y^2 / e.b^2 = 1

/-- Checks if a point is on a circle -/
def on_circle (p : Point) (c : Circle) : Prop :=
  p.x^2 + p.y^2 = c.r^2

/-- Checks if a line is tangent to a circle -/
def is_tangent (l : Line) (c : Circle) : Prop :=
  l.m^2 = c.r^2 * (1 + l.k^2)

/-- Checks if three points form a right angle -/
def is_right_angle (p1 p2 p3 : Point) : Prop :=
  (p2.x - p1.x) * (p3.x - p1.x) + (p2.y - p1.y) * (p3.y - p1.y) = 0

theorem ellipse_circle_relation 
  (e : Ellipse) (c : Circle) (l : Line) (A B : Point) :
  c.r < e.b →
  is_tangent l c →
  on_ellipse A e ∧ on_ellipse B e →
  on_circle A c ∧ on_circle B c →
  is_right_angle A B (Point.mk 0 0) →
  1 / e.a^2 + 1 / e.b^2 = 1 / c.r^2 :=
sorry

end NUMINAMATH_CALUDE_ellipse_circle_relation_l3099_309995


namespace NUMINAMATH_CALUDE_geometric_sequence_ratio_l3099_309934

/-- Given a geometric sequence {a_n} where each term is positive and 2a_1 + a_2 = a_3,
    prove that (a_4 + a_5) / (a_3 + a_4) = 2 -/
theorem geometric_sequence_ratio (a : ℕ → ℝ) (h_pos : ∀ n, a n > 0)
  (h_geom : ∃ q : ℝ, q > 0 ∧ ∀ n, a (n + 1) = q * a n)
  (h_cond : 2 * a 1 + a 2 = a 3) :
  (a 4 + a 5) / (a 3 + a 4) = 2 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_ratio_l3099_309934


namespace NUMINAMATH_CALUDE_square_root_3adic_l3099_309950

/-- Checks if 201 is the square root of 112101 in 3-adic arithmetic up to 3 digits of precision -/
theorem square_root_3adic (n : Nat) : n = 201 → n * n ≡ 112101 [ZMOD 27] := by
  sorry

end NUMINAMATH_CALUDE_square_root_3adic_l3099_309950


namespace NUMINAMATH_CALUDE_highlighter_difference_l3099_309920

/-- Proves that the difference between blue and pink highlighters is 5 --/
theorem highlighter_difference (yellow pink blue : ℕ) : 
  yellow = 7 →
  pink = yellow + 7 →
  yellow + pink + blue = 40 →
  blue - pink = 5 := by
sorry


end NUMINAMATH_CALUDE_highlighter_difference_l3099_309920


namespace NUMINAMATH_CALUDE_range_of_b_l3099_309917

theorem range_of_b (a b : ℝ) (h : a * b^2 > a ∧ a > a * b) : b < -1 := by
  sorry

end NUMINAMATH_CALUDE_range_of_b_l3099_309917


namespace NUMINAMATH_CALUDE_jose_and_jane_time_l3099_309980

-- Define the time taken by Jose to complete the task alone
def jose_time : ℝ := 15

-- Define the total time when Jose does half and Jane does half
def half_half_time : ℝ := 15

-- Define the time taken by Jose and Jane together
def combined_time : ℝ := 7.5

-- Theorem statement
theorem jose_and_jane_time : 
  (jose_time : ℝ) = 15 ∧ 
  (half_half_time : ℝ) = 15 → 
  (combined_time : ℝ) = 7.5 := by
  sorry

end NUMINAMATH_CALUDE_jose_and_jane_time_l3099_309980


namespace NUMINAMATH_CALUDE_tangent_intercept_implies_a_value_l3099_309987

/-- A function f(x) = ax³ + 4x + 5 -/
def f (a : ℝ) (x : ℝ) : ℝ := a * x^3 + 4 * x + 5

/-- The derivative of f(x) -/
def f' (a : ℝ) (x : ℝ) : ℝ := 3 * a * x^2 + 4

theorem tangent_intercept_implies_a_value (a : ℝ) :
  (f' a 1 * (-3/7 - 1) + f a 1 = 0) → a = 1 := by
  sorry

end NUMINAMATH_CALUDE_tangent_intercept_implies_a_value_l3099_309987


namespace NUMINAMATH_CALUDE_power_of_three_mod_eleven_l3099_309994

theorem power_of_three_mod_eleven : 3^2023 % 11 = 5 := by
  sorry

end NUMINAMATH_CALUDE_power_of_three_mod_eleven_l3099_309994


namespace NUMINAMATH_CALUDE_marble_probability_l3099_309997

theorem marble_probability (blue green white : ℝ) 
  (prob_sum : blue + green + white = 1)
  (prob_blue : blue = 0.25)
  (prob_green : green = 0.4) :
  white = 0.35 := by
  sorry

end NUMINAMATH_CALUDE_marble_probability_l3099_309997


namespace NUMINAMATH_CALUDE_distance_ratio_forms_circle_l3099_309983

/-- Given points A(0,0) and B(1,0) on a plane, the set of all points M(x,y) such that 
    the distance from M to A is three times the distance from M to B forms a circle 
    with center (-1/8, 0) and radius 3/8. -/
theorem distance_ratio_forms_circle :
  ∀ (x y : ℝ),
    (Real.sqrt (x^2 + y^2) = 3 * Real.sqrt ((x-1)^2 + y^2)) →
    ((x + 1/8)^2 + y^2 = (3/8)^2) := by
  sorry

end NUMINAMATH_CALUDE_distance_ratio_forms_circle_l3099_309983


namespace NUMINAMATH_CALUDE_set_operations_and_inclusion_l3099_309988

-- Define the sets A, B, and C
def A : Set ℝ := {x | -1 ≤ x ∧ x < 3}
def B : Set ℝ := {x | 2*x - 4 ≥ x - 2}
def C (a : ℝ) : Set ℝ := {x | 2*x + a ≥ 0}

-- Theorem statement
theorem set_operations_and_inclusion :
  (A ∩ B = {x | 2 ≤ x ∧ x < 3}) ∧
  (A ∪ B = {x | x ≥ -1}) ∧
  (∀ a : ℝ, B ⊆ C a → a > -4) :=
by sorry

end NUMINAMATH_CALUDE_set_operations_and_inclusion_l3099_309988


namespace NUMINAMATH_CALUDE_group_size_l3099_309999

theorem group_size (iceland : ℕ) (norway : ℕ) (both : ℕ) (neither : ℕ) 
  (h1 : iceland = 35)
  (h2 : norway = 23)
  (h3 : both = 31)
  (h4 : neither = 33) :
  iceland + norway - both + neither = 60 := by
  sorry

end NUMINAMATH_CALUDE_group_size_l3099_309999


namespace NUMINAMATH_CALUDE_repeating_decimal_36_equals_4_11_l3099_309904

/-- Represents a repeating decimal with a repeating part of two digits -/
def RepeatingDecimal (a b : ℕ) : ℚ :=
  (a * 10 + b) / 99

/-- The theorem states that 0.¯36 is equal to 4/11 -/
theorem repeating_decimal_36_equals_4_11 :
  RepeatingDecimal 3 6 = 4 / 11 := by
  sorry

end NUMINAMATH_CALUDE_repeating_decimal_36_equals_4_11_l3099_309904


namespace NUMINAMATH_CALUDE_square_of_sum_l3099_309932

theorem square_of_sum (a b : ℝ) : (a + b)^2 = a^2 + 2*a*b + b^2 := by
  sorry

end NUMINAMATH_CALUDE_square_of_sum_l3099_309932


namespace NUMINAMATH_CALUDE_dry_grapes_weight_l3099_309949

/-- Calculates the weight of dry grapes obtained from fresh grapes -/
theorem dry_grapes_weight
  (fresh_water_content : Real)
  (dry_water_content : Real)
  (fresh_weight : Real)
  (h1 : fresh_water_content = 0.90)
  (h2 : dry_water_content = 0.20)
  (h3 : fresh_weight = 20)
  : Real :=
by
  -- The weight of dry grapes obtained from fresh_weight of fresh grapes
  -- is equal to 2.5
  sorry

#check dry_grapes_weight

end NUMINAMATH_CALUDE_dry_grapes_weight_l3099_309949


namespace NUMINAMATH_CALUDE_arithmetic_sequence_common_difference_l3099_309946

/-- 
Given an arithmetic sequence with first term 5, last term 50, and sum of all terms 330,
prove that the common difference is 45/11.
-/
theorem arithmetic_sequence_common_difference 
  (a₁ : ℚ) (aₙ : ℚ) (S : ℚ) (n : ℕ) (d : ℚ) :
  a₁ = 5 →
  aₙ = 50 →
  S = 330 →
  S = n / 2 * (a₁ + aₙ) →
  aₙ = a₁ + (n - 1) * d →
  d = 45 / 11 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_common_difference_l3099_309946


namespace NUMINAMATH_CALUDE_unique_positive_integer_solution_l3099_309992

theorem unique_positive_integer_solution :
  ∃! (m n : ℕ+), 15 * m * n = 75 - 5 * m - 3 * n :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_unique_positive_integer_solution_l3099_309992


namespace NUMINAMATH_CALUDE_prime_sequence_divisibility_l3099_309913

theorem prime_sequence_divisibility (p d : ℕ+) 
  (h1 : Nat.Prime p)
  (h2 : Nat.Prime (p + d))
  (h3 : Nat.Prime (p + 2*d))
  (h4 : Nat.Prime (p + 3*d))
  (h5 : Nat.Prime (p + 4*d))
  (h6 : Nat.Prime (p + 5*d)) :
  2 ∣ d ∧ 3 ∣ d ∧ 5 ∣ d := by
  sorry

end NUMINAMATH_CALUDE_prime_sequence_divisibility_l3099_309913


namespace NUMINAMATH_CALUDE_total_amount_is_15_l3099_309975

-- Define the shares of w, x, and y
def w_share : ℝ := 10
def x_share : ℝ := w_share * 0.3
def y_share : ℝ := w_share * 0.2

-- Define the total amount
def total_amount : ℝ := w_share + x_share + y_share

-- Theorem statement
theorem total_amount_is_15 : total_amount = 15 := by
  sorry

end NUMINAMATH_CALUDE_total_amount_is_15_l3099_309975


namespace NUMINAMATH_CALUDE_jackson_investment_ratio_l3099_309979

-- Define the initial investment amount
def initial_investment : ℝ := 500

-- Define Brandon's final investment as a percentage of the initial
def brandon_final_percentage : ℝ := 0.2

-- Define the difference between Jackson's and Brandon's final investments
def investment_difference : ℝ := 1900

-- Theorem to prove
theorem jackson_investment_ratio :
  let brandon_final := initial_investment * brandon_final_percentage
  let jackson_final := brandon_final + investment_difference
  jackson_final / initial_investment = 4 := by
  sorry

end NUMINAMATH_CALUDE_jackson_investment_ratio_l3099_309979


namespace NUMINAMATH_CALUDE_multiplicative_inverse_exists_l3099_309954

theorem multiplicative_inverse_exists : ∃ N : ℕ, 
  N > 0 ∧ 
  N < 1000000 ∧ 
  (123456 * 654321 * N) % 1234567 = 1 := by
sorry

end NUMINAMATH_CALUDE_multiplicative_inverse_exists_l3099_309954


namespace NUMINAMATH_CALUDE_computer_profit_percentage_l3099_309907

theorem computer_profit_percentage (cost : ℝ) 
  (h1 : 2240 = cost * 1.4) 
  (h2 : 2400 > cost) : 
  (2400 - cost) / cost = 0.5 := by
sorry

end NUMINAMATH_CALUDE_computer_profit_percentage_l3099_309907


namespace NUMINAMATH_CALUDE_tank_capacity_proof_l3099_309918

def tank_capacity (oil_bought : ℕ) (oil_in_tank : ℕ) : ℕ :=
  oil_bought + oil_in_tank

theorem tank_capacity_proof (oil_bought : ℕ) (oil_in_tank : ℕ) 
  (h1 : oil_bought = 728) (h2 : oil_in_tank = 24) : 
  tank_capacity oil_bought oil_in_tank = 752 := by
  sorry

#check tank_capacity_proof

end NUMINAMATH_CALUDE_tank_capacity_proof_l3099_309918


namespace NUMINAMATH_CALUDE_max_value_of_expression_l3099_309976

theorem max_value_of_expression :
  ∃ (a b c d : ℕ),
    ({a, b, c, d} : Finset ℕ) = {1, 2, 4, 5} →
    ∀ (w x y z : ℕ),
      ({w, x, y, z} : Finset ℕ) = {1, 2, 4, 5} →
      c * a^b - d ≤ 79 ∧
      (c * a^b - d = 79 → (a = 2 ∧ b = 4 ∧ c = 5 ∧ d = 1) ∨ (a = 4 ∧ b = 2 ∧ c = 5 ∧ d = 1)) :=
by sorry

#check max_value_of_expression

end NUMINAMATH_CALUDE_max_value_of_expression_l3099_309976


namespace NUMINAMATH_CALUDE_min_p_value_l3099_309952

theorem min_p_value (p q : ℕ+) 
  (h1 : (2008 : ℚ) / 2009 < p / q)
  (h2 : p / q < (2009 : ℚ) / 2010) : 
  4017 ≤ p.val := by
  sorry

end NUMINAMATH_CALUDE_min_p_value_l3099_309952


namespace NUMINAMATH_CALUDE_cos_330_degrees_l3099_309944

theorem cos_330_degrees : Real.cos (330 * π / 180) = Real.sqrt 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_cos_330_degrees_l3099_309944


namespace NUMINAMATH_CALUDE_smallest_n_satisfying_conditions_l3099_309906

theorem smallest_n_satisfying_conditions : ∃ (n : ℕ), 
  n > 20 ∧ 
  n % 6 = 4 ∧ 
  n % 7 = 3 ∧ 
  n % 8 = 5 ∧ 
  (∀ m : ℕ, m > 20 ∧ m % 6 = 4 ∧ m % 7 = 3 ∧ m % 8 = 5 → m ≥ n) ∧
  n = 220 :=
by sorry

end NUMINAMATH_CALUDE_smallest_n_satisfying_conditions_l3099_309906


namespace NUMINAMATH_CALUDE_max_value_x_plus_7y_l3099_309935

theorem max_value_x_plus_7y :
  ∀ x y : ℝ,
  0 ≤ x ∧ x ≤ 1 →
  0 ≤ y ∧ y ≤ 1 →
  Real.sqrt (x * y) + Real.sqrt ((1 - x) * (1 - y)) = Real.sqrt (7 * x * (1 - y)) + Real.sqrt (y * (1 - x)) / Real.sqrt 7 →
  (∀ z w : ℝ, 0 ≤ z ∧ z ≤ 1 → 0 ≤ w ∧ w ≤ 1 → z + 7 * w ≤ 57 / 8) ∧
  ∃ a b : ℝ, 0 ≤ a ∧ a ≤ 1 ∧ 0 ≤ b ∧ b ≤ 1 ∧ a + 7 * b = 57 / 8 :=
by sorry

end NUMINAMATH_CALUDE_max_value_x_plus_7y_l3099_309935


namespace NUMINAMATH_CALUDE_smallest_factor_of_4896_l3099_309990

theorem smallest_factor_of_4896 : 
  ∃ (a b : ℕ), 
    10 ≤ a ∧ a < 100 ∧ 
    10 ≤ b ∧ b < 100 ∧ 
    a * b = 4896 ∧ 
    (∀ (x y : ℕ), 10 ≤ x ∧ x < 100 ∧ 10 ≤ y ∧ y < 100 ∧ x * y = 4896 → min x y ≥ 32) :=
by sorry

end NUMINAMATH_CALUDE_smallest_factor_of_4896_l3099_309990


namespace NUMINAMATH_CALUDE_amount_calculation_l3099_309957

/-- If 5% of 25% of an amount is $20, then that amount is $1600. -/
theorem amount_calculation (x : ℝ) : (0.05 * (0.25 * x) = 20) → x = 1600 := by
  sorry

end NUMINAMATH_CALUDE_amount_calculation_l3099_309957


namespace NUMINAMATH_CALUDE_harrys_morning_routine_time_l3099_309996

def morning_routine (coffee_bagel_time : ℕ) (reading_eating_factor : ℕ) : ℕ :=
  coffee_bagel_time + reading_eating_factor * coffee_bagel_time

theorem harrys_morning_routine_time :
  morning_routine 15 2 = 45 :=
by sorry

end NUMINAMATH_CALUDE_harrys_morning_routine_time_l3099_309996


namespace NUMINAMATH_CALUDE_ellipse_intersection_properties_l3099_309926

-- Define the ellipse
def ellipse (x y : ℝ) : Prop := x^2/2 + y^2 = 1

-- Define the line
def line (x y : ℝ) : Prop := y = x - 1

-- Define the left focus
def left_focus : ℝ × ℝ := (-1, 0)

-- Define the intersection points
def intersection_points : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | ellipse p.1 p.2 ∧ line p.1 p.2}

-- Theorem statement
theorem ellipse_intersection_properties :
  let A := (0, -1)
  let B := (4/3, 1/3)
  (A ∈ intersection_points) ∧
  (B ∈ intersection_points) ∧
  (∃ (AB : ℝ), AB = (4 * Real.sqrt 2) / 3 ∧
    AB = Real.sqrt ((B.1 - A.1)^2 + (B.2 - A.2)^2)) ∧
  (∃ (S : ℝ), S = 4/3 ∧
    S = (1/2) * ((4 * Real.sqrt 2) / 3) * Real.sqrt 2) :=
by sorry

end NUMINAMATH_CALUDE_ellipse_intersection_properties_l3099_309926


namespace NUMINAMATH_CALUDE_coefficient_equals_21_l3099_309985

-- Define the coefficient of x^2 in the expansion of (ax+1)^5(x+1)^2
def coefficient (a : ℝ) : ℝ := 10 * a^2 + 10 * a + 1

-- Theorem statement
theorem coefficient_equals_21 (a : ℝ) : 
  coefficient a = 21 ↔ a = 1 ∨ a = -2 := by
  sorry


end NUMINAMATH_CALUDE_coefficient_equals_21_l3099_309985


namespace NUMINAMATH_CALUDE_day_division_count_l3099_309978

-- Define the number of seconds in a day
def seconds_in_day : ℕ := 72000

-- Define a function to count the number of ways to divide the day
def count_divisions (total_seconds : ℕ) : ℕ :=
  -- The actual implementation is not provided, as per instructions
  sorry

-- Theorem statement
theorem day_division_count :
  count_divisions seconds_in_day = 60 := by sorry

end NUMINAMATH_CALUDE_day_division_count_l3099_309978


namespace NUMINAMATH_CALUDE_evaluate_expression_l3099_309914

theorem evaluate_expression : (3^10 + 3^7) / (3^10 - 3^7) = 14/13 := by
  sorry

end NUMINAMATH_CALUDE_evaluate_expression_l3099_309914
