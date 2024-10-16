import Mathlib

namespace NUMINAMATH_CALUDE_total_population_l1795_179563

/-- Represents the number of boys, girls, and teachers in a school -/
structure School where
  b : ℕ  -- number of boys
  g : ℕ  -- number of girls
  t : ℕ  -- number of teachers

/-- The conditions of the school population -/
def school_conditions (s : School) : Prop :=
  s.b = 4 * s.g ∧ s.g = 5 * s.t

/-- The theorem stating that the total population is 26 times the number of teachers -/
theorem total_population (s : School) (h : school_conditions s) : 
  s.b + s.g + s.t = 26 * s.t := by
  sorry

end NUMINAMATH_CALUDE_total_population_l1795_179563


namespace NUMINAMATH_CALUDE_brian_cards_left_l1795_179504

/-- Given that Brian has 76 cards initially and Wayne takes 59 cards away,
    prove that Brian will have 17 cards left. -/
theorem brian_cards_left (initial_cards : ℕ) (cards_taken : ℕ) (cards_left : ℕ) : 
  initial_cards = 76 → cards_taken = 59 → cards_left = initial_cards - cards_taken → cards_left = 17 := by
  sorry

end NUMINAMATH_CALUDE_brian_cards_left_l1795_179504


namespace NUMINAMATH_CALUDE_bus_total_capacity_l1795_179520

/-- Represents the seating capacity of a bus with specific seating arrangements -/
def bus_capacity (left_seats : ℕ) (right_seat_diff : ℕ) (people_per_seat : ℕ) (back_seat_capacity : ℕ) : ℕ :=
  let right_seats := left_seats - right_seat_diff
  let total_regular_seats := left_seats + right_seats
  let regular_capacity := total_regular_seats * people_per_seat
  regular_capacity + back_seat_capacity

/-- Theorem stating the total seating capacity of the bus -/
theorem bus_total_capacity :
  bus_capacity 15 3 3 8 = 89 := by
  sorry

#eval bus_capacity 15 3 3 8

end NUMINAMATH_CALUDE_bus_total_capacity_l1795_179520


namespace NUMINAMATH_CALUDE_rectangular_stadium_length_l1795_179564

theorem rectangular_stadium_length 
  (perimeter : ℝ) 
  (breadth : ℝ) 
  (h1 : perimeter = 800) 
  (h2 : breadth = 300) : 
  2 * (breadth + 100) = perimeter :=
by sorry

end NUMINAMATH_CALUDE_rectangular_stadium_length_l1795_179564


namespace NUMINAMATH_CALUDE_equation_solutions_l1795_179585

theorem equation_solutions : 
  (∃ (x₁ x₂ : ℝ), (x₁ = 3/5 ∧ x₂ = -3) ∧ 
    (2*x₁ - 3)^2 = 9*x₁^2 ∧ (2*x₂ - 3)^2 = 9*x₂^2) ∧
  (∃ (y₁ y₂ : ℝ), (y₁ = 2 ∧ y₂ = -1/2) ∧ 
    2*y₁*(y₁-2) + y₁ = 2 ∧ 2*y₂*(y₂-2) + y₂ = 2) :=
by sorry

end NUMINAMATH_CALUDE_equation_solutions_l1795_179585


namespace NUMINAMATH_CALUDE_special_function_inequality_l1795_179586

open Set

/-- A function satisfying the given conditions -/
structure SpecialFunction where
  f : ℝ → ℝ
  f_diff : Differentiable ℝ f
  f_domain : ∀ x, x < 0 → f x ≠ 0
  f_ineq : ∀ x, x < 0 → 2 * (f x) + x * (deriv f x) > x^2

/-- The main theorem -/
theorem special_function_inequality (sf : SpecialFunction) :
  {x : ℝ | (x + 2016)^2 * sf.f (x + 2016) - 4 * sf.f (-2) > 0} = Iio (-2018) := by
  sorry

end NUMINAMATH_CALUDE_special_function_inequality_l1795_179586


namespace NUMINAMATH_CALUDE_expand_polynomial_l1795_179508

theorem expand_polynomial (x : ℝ) : (x + 3) * (4 * x^2 - 5 * x + 7) = 4 * x^3 + 7 * x^2 - 8 * x + 21 := by
  sorry

end NUMINAMATH_CALUDE_expand_polynomial_l1795_179508


namespace NUMINAMATH_CALUDE_geometric_sequence_property_l1795_179554

/-- A geometric sequence with the given properties -/
def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

theorem geometric_sequence_property 
  (a : ℕ → ℝ) 
  (h_geometric : geometric_sequence a)
  (h_a5 : a 5 = -9)
  (h_a8 : a 8 = 6) :
  a 11 = -4 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_property_l1795_179554


namespace NUMINAMATH_CALUDE_geometric_sequence_minimum_value_l1795_179596

/-- A positive geometric sequence -/
def GeometricSequence (a : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, q > 0 ∧ ∀ n : ℕ, a (n + 1) = q * a n

theorem geometric_sequence_minimum_value (a : ℕ → ℝ) (m n : ℕ) :
  GeometricSequence a →
  a 2016 = a 2015 + 2 * a 2014 →
  a m * a n = 16 * (a 1)^2 →
  (4 : ℝ) / m + 1 / n ≥ 3 / 2 :=
by sorry

end NUMINAMATH_CALUDE_geometric_sequence_minimum_value_l1795_179596


namespace NUMINAMATH_CALUDE_f_increasing_f_odd_range_l1795_179558

noncomputable section

-- Define the function f(x)
def f (a : ℝ) (x : ℝ) : ℝ := a - 1 / (2^x + 1)

-- Theorem 1: f(x) is an increasing function on ℝ
theorem f_increasing (a : ℝ) : 
  ∀ x₁ x₂ : ℝ, x₁ < x₂ → f a x₁ < f a x₂ :=
sorry

-- Theorem 2: When f(x) is an odd function, its range on [-1, 2] is [-1/6, 3/10]
theorem f_odd_range (a : ℝ) 
  (h_odd : ∀ x : ℝ, f a (-x) = -(f a x)) : 
  Set.range (fun x => f a x) ∩ Set.Icc (-1 : ℝ) 2 = Set.Icc (-1/6 : ℝ) (3/10) :=
sorry

end

end NUMINAMATH_CALUDE_f_increasing_f_odd_range_l1795_179558


namespace NUMINAMATH_CALUDE_reversed_segment_appears_in_powers_of_two_l1795_179575

/-- The sequence of first digits of powers of 5 -/
def firstDigitsPowersOf5 : ℕ → ℕ :=
  λ n => (5^n : ℕ) % 10

/-- The sequence of first digits of powers of 2 -/
def firstDigitsPowersOf2 : ℕ → ℕ :=
  λ n => (2^n : ℕ) % 10

/-- Check if a list is a subsequence of another list -/
def isSubsequence {α : Type} [DecidableEq α] : List α → List α → Bool :=
  λ subseq seq => sorry

/-- Theorem: Any reversed segment of firstDigitsPowersOf5 appears in firstDigitsPowersOf2 -/
theorem reversed_segment_appears_in_powers_of_two :
  ∀ (start finish : ℕ),
    start ≤ finish →
    ∃ (n m : ℕ),
      isSubsequence
        ((List.range (finish - start + 1)).map (λ i => firstDigitsPowersOf5 (start + i))).reverse
        ((List.range (m - n + 1)).map (λ i => firstDigitsPowersOf2 (n + i))) = true :=
by
  sorry

end NUMINAMATH_CALUDE_reversed_segment_appears_in_powers_of_two_l1795_179575


namespace NUMINAMATH_CALUDE_line_y_axis_intersection_l1795_179509

/-- The equation of the line -/
def line_equation (x y : ℝ) : Prop := 5 * y + 3 * x = 15

/-- A point is on the y-axis if its x-coordinate is 0 -/
def on_y_axis (x y : ℝ) : Prop := x = 0

/-- The intersection point of the line with the y-axis -/
def intersection_point : ℝ × ℝ := (0, 3)

theorem line_y_axis_intersection :
  let (x, y) := intersection_point
  line_equation x y ∧ on_y_axis x y :=
sorry

end NUMINAMATH_CALUDE_line_y_axis_intersection_l1795_179509


namespace NUMINAMATH_CALUDE_sum_of_squared_coefficients_l1795_179551

def original_expression (x : ℝ) : ℝ := 3 * (x^3 - 4*x^2 + x) - 5 * (x^3 + 2*x^2 - 5*x + 3)

def simplified_expression (x : ℝ) : ℝ := -2*x^3 - 22*x^2 + 28*x - 15

def coefficients : List ℤ := [-2, -22, 28, -15]

theorem sum_of_squared_coefficients :
  (coefficients.map (λ c => c^2)).sum = 1497 :=
sorry

end NUMINAMATH_CALUDE_sum_of_squared_coefficients_l1795_179551


namespace NUMINAMATH_CALUDE_alisha_todd_ratio_l1795_179502

/-- Represents the number of gumballs given to each person and the total purchased -/
structure GumballDistribution where
  total : ℕ
  todd : ℕ
  alisha : ℕ
  bobby : ℕ
  remaining : ℕ

/-- Defines the conditions of the gumball distribution problem -/
def gumball_problem (g : GumballDistribution) : Prop :=
  g.total = 45 ∧
  g.todd = 4 ∧
  g.bobby = 4 * g.alisha - 5 ∧
  g.remaining = 6 ∧
  g.total = g.todd + g.alisha + g.bobby + g.remaining

/-- Theorem stating the ratio of gumballs given to Alisha vs Todd -/
theorem alisha_todd_ratio (g : GumballDistribution) 
  (h : gumball_problem g) : g.alisha = 2 * g.todd := by
  sorry


end NUMINAMATH_CALUDE_alisha_todd_ratio_l1795_179502


namespace NUMINAMATH_CALUDE_geometric_sequence_property_l1795_179516

/-- A geometric sequence with positive terms -/
def GeometricSequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, r > 0 ∧ ∀ n : ℕ, a (n + 1) = a n * r

theorem geometric_sequence_property (a : ℕ → ℝ) 
  (h_geom : GeometricSequence a)
  (h_pos : ∀ n : ℕ, a n > 0)
  (h_eq : 1 / (a 2 * a 4) + 2 / (a 4 * a 4) + 1 / (a 4 * a 6) = 81) :
  1 / a 3 + 1 / a 5 = 9 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_property_l1795_179516


namespace NUMINAMATH_CALUDE_video_game_lives_l1795_179517

theorem video_game_lives (initial_players : Nat) (quitting_players : Nat) (lives_per_player : Nat) : 
  initial_players = 20 → quitting_players = 10 → lives_per_player = 7 → 
  (initial_players - quitting_players) * lives_per_player = 70 := by
  sorry

end NUMINAMATH_CALUDE_video_game_lives_l1795_179517


namespace NUMINAMATH_CALUDE_afternoon_bike_sales_l1795_179506

theorem afternoon_bike_sales (morning_sales : ℕ) (total_clamps : ℕ) (clamps_per_bike : ℕ) :
  morning_sales = 19 →
  total_clamps = 92 →
  clamps_per_bike = 2 →
  ∃ (afternoon_sales : ℕ), 
    afternoon_sales = 27 ∧
    total_clamps = clamps_per_bike * (morning_sales + afternoon_sales) :=
by sorry

end NUMINAMATH_CALUDE_afternoon_bike_sales_l1795_179506


namespace NUMINAMATH_CALUDE_proposition_3_proposition_4_l1795_179527

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relationships between lines and planes
variable (belongs_to : Line → Plane → Prop)
variable (intersect : Plane → Plane → Line → Prop)
variable (parallel : Line → Line → Prop)
variable (plane_parallel : Plane → Plane → Prop)
variable (perpendicular : Line → Line → Prop)
variable (plane_perpendicular : Plane → Plane → Prop)
variable (line_perpendicular_plane : Line → Plane → Prop)

-- Proposition 3
theorem proposition_3 
  (α β : Plane) (a b : Line) :
  plane_perpendicular α β →
  intersect α β a →
  belongs_to b β →
  perpendicular a b →
  line_perpendicular_plane b α :=
sorry

-- Proposition 4
theorem proposition_4
  (α : Plane) (a b l : Line) :
  belongs_to a α →
  belongs_to b α →
  perpendicular l a →
  perpendicular l b →
  line_perpendicular_plane l α :=
sorry

end NUMINAMATH_CALUDE_proposition_3_proposition_4_l1795_179527


namespace NUMINAMATH_CALUDE_z_in_first_quadrant_l1795_179510

def z : ℂ := Complex.I * (2 - Complex.I)

theorem z_in_first_quadrant : 
  Real.sign (z.re) = 1 ∧ Real.sign (z.im) = 1 := by sorry

end NUMINAMATH_CALUDE_z_in_first_quadrant_l1795_179510


namespace NUMINAMATH_CALUDE_ratio_problem_l1795_179591

theorem ratio_problem (x y : ℤ) : 
  (y = 4 * x) →  -- The two integers are in the ratio of 1 to 4
  (x + 12 = y) → -- Adding 12 to the smaller number makes the ratio 1 to 1
  y = 16 :=      -- The larger integer is 16
by
  sorry

end NUMINAMATH_CALUDE_ratio_problem_l1795_179591


namespace NUMINAMATH_CALUDE_amaya_total_marks_l1795_179513

/-- Represents the marks scored in different subjects -/
structure Marks where
  arts : ℕ
  maths : ℕ
  music : ℕ
  social_studies : ℕ

/-- Calculates the total marks across all subjects -/
def total_marks (m : Marks) : ℕ :=
  m.arts + m.maths + m.music + m.social_studies

/-- Theorem stating the total marks Amaya scored given the conditions -/
theorem amaya_total_marks :
  ∀ (m : Marks),
    m.arts - m.maths = 20 →
    m.social_studies > m.music →
    m.music = 70 →
    m.maths = (9 * m.arts) / 10 →
    m.social_studies - m.music = 10 →
    total_marks m = 530 := by
  sorry

#check amaya_total_marks

end NUMINAMATH_CALUDE_amaya_total_marks_l1795_179513


namespace NUMINAMATH_CALUDE_school_prizes_l1795_179560

theorem school_prizes (total_money : ℝ) (pen_cost notebook_cost : ℝ) 
  (h1 : total_money = 60 * (pen_cost + 2 * notebook_cost))
  (h2 : total_money = 50 * (pen_cost + 3 * notebook_cost)) :
  (total_money / pen_cost : ℝ) = 100 := by
  sorry

end NUMINAMATH_CALUDE_school_prizes_l1795_179560


namespace NUMINAMATH_CALUDE_crayon_difference_l1795_179557

/-- Given an initial number of crayons, the number given away, and the number lost,
    prove that the difference between lost and given away is their subtraction. -/
theorem crayon_difference (initial given lost : ℕ) : lost - given = lost - given := by
  sorry

end NUMINAMATH_CALUDE_crayon_difference_l1795_179557


namespace NUMINAMATH_CALUDE_birds_on_fence_l1795_179532

theorem birds_on_fence (initial_birds : ℝ) (birds_flew_away : ℝ) (remaining_birds : ℝ) : 
  initial_birds = 12.0 → birds_flew_away = 8.0 → remaining_birds = initial_birds - birds_flew_away → remaining_birds = 4.0 := by
  sorry

end NUMINAMATH_CALUDE_birds_on_fence_l1795_179532


namespace NUMINAMATH_CALUDE_extreme_values_depend_on_a_consistent_monotonicity_implies_b_bound_max_ab_difference_l1795_179570

-- Define the functions f and g
def f (a : ℝ) (x : ℝ) : ℝ := x^3 + a*x
def g (b : ℝ) (x : ℝ) : ℝ := x^2 + b*x

-- Define the derivatives of f and g
def f' (a : ℝ) (x : ℝ) : ℝ := 3*x^2 + a
def g' (b : ℝ) (x : ℝ) : ℝ := 2*x + b

-- Define consistent monotonicity
def consistent_monotonicity (a b : ℝ) (l : Set ℝ) : Prop :=
  ∀ x ∈ l, f' a x * g' b x ≥ 0

theorem extreme_values_depend_on_a (a : ℝ) : 
  (a ≥ 0 → ∀ x : ℝ, f' a x ≥ 0) ∧ 
  (a < 0 → ∃ x₁ x₂ : ℝ, x₁ < x₂ ∧ f' a x₁ < 0 ∧ f' a x₂ > 0) :=
sorry

theorem consistent_monotonicity_implies_b_bound (a b : ℝ) :
  a > 0 → consistent_monotonicity a b { x | x ≥ -2 } → b ≥ 4 :=
sorry

theorem max_ab_difference (a b : ℝ) :
  a < 0 → a ≠ b → consistent_monotonicity a b (Set.Ioo a b) → |a - b| ≤ 1/3 :=
sorry

end NUMINAMATH_CALUDE_extreme_values_depend_on_a_consistent_monotonicity_implies_b_bound_max_ab_difference_l1795_179570


namespace NUMINAMATH_CALUDE_second_number_is_40_l1795_179599

theorem second_number_is_40 (a b c : ℝ) 
  (sum_eq : a + b + c = 120)
  (ratio_ab : a / b = 3 / 4)
  (ratio_bc : b / c = 4 / 5)
  (pos_a : a > 0)
  (pos_b : b > 0)
  (pos_c : c > 0) :
  b = 40 := by
sorry

end NUMINAMATH_CALUDE_second_number_is_40_l1795_179599


namespace NUMINAMATH_CALUDE_razorback_shop_tshirt_profit_l1795_179540

/-- The amount of money made per t-shirt, given the number of t-shirts sold and the total revenue from t-shirt sales. -/
def amount_per_tshirt (num_tshirts : ℕ) (total_revenue : ℕ) : ℚ :=
  total_revenue / num_tshirts

/-- Theorem stating that the amount made per t-shirt is $215, given the conditions. -/
theorem razorback_shop_tshirt_profit :
  amount_per_tshirt 20 4300 = 215 := by
  sorry

end NUMINAMATH_CALUDE_razorback_shop_tshirt_profit_l1795_179540


namespace NUMINAMATH_CALUDE_class_size_difference_l1795_179562

theorem class_size_difference (students : ℕ) (teachers : ℕ) (enrollments : List ℕ) : 
  students = 120 →
  teachers = 6 →
  enrollments = [60, 30, 15, 5, 5, 5] →
  (enrollments.sum = students) →
  (enrollments.length = teachers) →
  let t : ℚ := (enrollments.sum : ℚ) / teachers
  let s : ℚ := (enrollments.map (λ x => x * x)).sum / students
  t - s = -20 := by
  sorry

#check class_size_difference

end NUMINAMATH_CALUDE_class_size_difference_l1795_179562


namespace NUMINAMATH_CALUDE_line_y_intercept_l1795_179561

/-- A line with slope 3 and x-intercept (7,0) has y-intercept (0, -21) -/
theorem line_y_intercept (m : ℝ) (x₀ : ℝ) (y : ℝ → ℝ) :
  m = 3 →
  x₀ = 7 →
  y 0 = 0 →
  (∀ x, y x = m * (x - x₀)) →
  y 0 = -21 :=
by sorry

end NUMINAMATH_CALUDE_line_y_intercept_l1795_179561


namespace NUMINAMATH_CALUDE_unique_intersection_l1795_179543

/-- The value of m for which the vertical line x = m intersects the parabola x = -4y^2 + 2y + 3 at exactly one point -/
def m : ℚ := 13/4

/-- The equation of the parabola -/
def parabola (y : ℝ) : ℝ := -4 * y^2 + 2 * y + 3

/-- Theorem stating that the vertical line x = m intersects the parabola at exactly one point -/
theorem unique_intersection :
  ∃! y : ℝ, parabola y = m :=
sorry

end NUMINAMATH_CALUDE_unique_intersection_l1795_179543


namespace NUMINAMATH_CALUDE_land_area_scientific_notation_l1795_179593

/-- Scientific notation representation of a real number -/
structure ScientificNotation where
  coefficient : ℝ
  exponent : ℤ
  norm_coeff : 1 ≤ |coefficient| ∧ |coefficient| < 10

/-- The land area in square kilometers -/
def land_area : ℝ := 9600000

/-- The scientific notation of the land area -/
def land_area_scientific : ScientificNotation :=
  { coefficient := 9.6
  , exponent := 6
  , norm_coeff := by sorry }

theorem land_area_scientific_notation :
  land_area = land_area_scientific.coefficient * (10 : ℝ) ^ land_area_scientific.exponent :=
by sorry

end NUMINAMATH_CALUDE_land_area_scientific_notation_l1795_179593


namespace NUMINAMATH_CALUDE_smallest_value_of_floor_sum_l1795_179541

theorem smallest_value_of_floor_sum (a b c : ℕ+) 
  (hab : (a : ℚ) / b = 2)
  (hbc : (b : ℚ) / c = 2)
  (hca : (c : ℚ) / a = 1 / 4) :
  ⌊(a + b : ℚ) / c⌋ + ⌊(b + c : ℚ) / a⌋ + ⌊(c + a : ℚ) / b⌋ = 8 :=
by sorry

end NUMINAMATH_CALUDE_smallest_value_of_floor_sum_l1795_179541


namespace NUMINAMATH_CALUDE_no_defective_products_exactly_two_defective_products_at_least_two_defective_products_l1795_179556

-- Define the total number of items
def total_items : ℕ := 100

-- Define the number of defective items
def defective_items : ℕ := 3

-- Define the number of items to be selected
def selected_items : ℕ := 5

-- Theorem for scenario (I): No defective product
theorem no_defective_products : 
  Nat.choose (total_items - defective_items) selected_items = 64446024 := by sorry

-- Theorem for scenario (II): Exactly two defective products
theorem exactly_two_defective_products :
  Nat.choose defective_items 2 * Nat.choose (total_items - defective_items) (selected_items - 2) = 442320 := by sorry

-- Theorem for scenario (III): At least two defective products
theorem at_least_two_defective_products :
  Nat.choose defective_items 2 * Nat.choose (total_items - defective_items) (selected_items - 2) +
  Nat.choose defective_items 3 * Nat.choose (total_items - defective_items) (selected_items - 3) = 446886 := by sorry

end NUMINAMATH_CALUDE_no_defective_products_exactly_two_defective_products_at_least_two_defective_products_l1795_179556


namespace NUMINAMATH_CALUDE_comparison_of_expressions_l1795_179525

theorem comparison_of_expressions (x : ℝ) (h : x ≠ 1) :
  (x > 1 → 1 + x > 1 / (1 - x)) ∧ (x < 1 → 1 + x < 1 / (1 - x)) := by
  sorry

end NUMINAMATH_CALUDE_comparison_of_expressions_l1795_179525


namespace NUMINAMATH_CALUDE_equation_solutions_l1795_179537

def solution_set : Set (ℤ × ℤ) :=
  {(6, 3), (6, -9), (1, 1), (1, -2), (2, -1)}

def satisfies_equation (p : ℤ × ℤ) : Prop :=
  let (x, y) := p
  y * (x + y) = x^3 - 7*x^2 + 11*x - 3

theorem equation_solutions :
  ∀ p : ℤ × ℤ, satisfies_equation p ↔ p ∈ solution_set :=
sorry

end NUMINAMATH_CALUDE_equation_solutions_l1795_179537


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_condition_l1795_179544

theorem sufficient_not_necessary_condition (a b : ℝ) :
  (∀ a b : ℝ, |a + b| > 1 → |a| + |b| > 1) ∧
  (∃ a b : ℝ, |a| + |b| > 1 ∧ |a + b| ≤ 1) :=
by sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_condition_l1795_179544


namespace NUMINAMATH_CALUDE_least_factorial_divisible_by_7350_l1795_179534

theorem least_factorial_divisible_by_7350 : ∃ (n : ℕ), n > 0 ∧ 7350 ∣ n.factorial ∧ ∀ (m : ℕ), m > 0 → 7350 ∣ m.factorial → n ≤ m :=
  sorry

end NUMINAMATH_CALUDE_least_factorial_divisible_by_7350_l1795_179534


namespace NUMINAMATH_CALUDE_school_chairs_problem_l1795_179550

theorem school_chairs_problem (initial_chairs : ℕ) : 
  initial_chairs < 35 →
  ∃ (k : ℕ), initial_chairs + 27 = 35 * k →
  initial_chairs = 8 := by
sorry

end NUMINAMATH_CALUDE_school_chairs_problem_l1795_179550


namespace NUMINAMATH_CALUDE_gcd_of_powers_of_79_l1795_179568

theorem gcd_of_powers_of_79 :
  Nat.Prime 79 →
  Nat.gcd (79^7 + 1) (79^7 + 79^3 + 1) = 1 := by
sorry

end NUMINAMATH_CALUDE_gcd_of_powers_of_79_l1795_179568


namespace NUMINAMATH_CALUDE_A_expression_l1795_179523

theorem A_expression (a : ℝ) (A : ℝ) 
  (h : 2.353 * A = (3 * a + Real.sqrt (6 * a - 1))^(1/2) + (3 * a - Real.sqrt (6 * a - 1))^(1/2)) :
  ((1/6 ≤ a ∧ a < 1/3) → A = Real.sqrt 2 / (1 - 3 * a)) ∧
  (a > 1/3 → A = Real.sqrt (12 * a - 2) / (3 * a - 1)) := by
  sorry

end NUMINAMATH_CALUDE_A_expression_l1795_179523


namespace NUMINAMATH_CALUDE_unique_solution_exponential_equation_l1795_179505

theorem unique_solution_exponential_equation :
  ∃! x : ℝ, (2 : ℝ)^(4*x + 2) * (4 : ℝ)^(2*x + 4) = (8 : ℝ)^(3*x + 4) ∧ x = -2 := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_exponential_equation_l1795_179505


namespace NUMINAMATH_CALUDE_divisible_polynomial_sum_l1795_179528

-- Define the polynomial
def p (A B : ℝ) (x : ℂ) := x^101 + A*x + B

-- Define the condition of divisibility
def is_divisible (A B : ℝ) : Prop :=
  ∀ x : ℂ, x^2 + x + 1 = 0 → p A B x = 0

-- Theorem statement
theorem divisible_polynomial_sum (A B : ℝ) (h : is_divisible A B) : A + B = 2 := by
  sorry

end NUMINAMATH_CALUDE_divisible_polynomial_sum_l1795_179528


namespace NUMINAMATH_CALUDE_sum_of_products_l1795_179545

theorem sum_of_products (x₁ x₂ x₃ x₄ x₅ x₆ x₇ x₈ : ℝ) 
  (eq1 : x₁ + 5*x₂ + 10*x₃ + 17*x₄ + 26*x₅ + 37*x₆ + 50*x₇ + 65*x₈ = 2)
  (eq2 : 5*x₁ + 10*x₂ + 17*x₃ + 26*x₄ + 37*x₅ + 50*x₆ + 65*x₇ + 82*x₈ = 14)
  (eq3 : 10*x₁ + 17*x₂ + 26*x₃ + 37*x₄ + 50*x₅ + 65*x₆ + 82*x₇ + 101*x₈ = 140) :
  17*x₁ + 26*x₂ + 37*x₃ + 50*x₄ + 65*x₅ + 82*x₆ + 101*x₇ + 122*x₈ = 608 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_products_l1795_179545


namespace NUMINAMATH_CALUDE_sum_of_roots_l1795_179552

theorem sum_of_roots (x : ℝ) : (x + 2) * (x - 3) = 16 → ∃ y : ℝ, (y + 2) * (y - 3) = 16 ∧ x + y = 1 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_roots_l1795_179552


namespace NUMINAMATH_CALUDE_initial_hamburgers_count_l1795_179539

/-- Proves that the number of hamburgers made initially equals 9 -/
theorem initial_hamburgers_count (initial : ℕ) (additional : ℕ) (total : ℕ)
  (h1 : additional = 3)
  (h2 : total = 12)
  (h3 : initial + additional = total) :
  initial = 9 := by
  sorry

end NUMINAMATH_CALUDE_initial_hamburgers_count_l1795_179539


namespace NUMINAMATH_CALUDE_probability_of_three_correct_l1795_179501

/-- Two fair dice are thrown once each -/
def dice : ℕ := 2

/-- Each die has 6 faces -/
def faces_per_die : ℕ := 6

/-- The numbers facing up are different -/
def different_numbers : Prop := true

/-- The probability that one of the dice shows a 3 -/
def probability_of_three : ℚ := 1 / 3

/-- Theorem stating that the probability of getting a 3 on one die when two fair dice are thrown with different numbers is 1/3 -/
theorem probability_of_three_correct (h : different_numbers) : probability_of_three = 1 / 3 := by
  sorry

end NUMINAMATH_CALUDE_probability_of_three_correct_l1795_179501


namespace NUMINAMATH_CALUDE_volume_ratio_volume_112oz_l1795_179521

/-- A substance with volume directly proportional to weight -/
structure Substance where
  /-- Constant of proportionality between volume and weight -/
  k : ℝ
  /-- Assumption that k is positive -/
  k_pos : k > 0

/-- Volume of the substance given its weight -/
def volume (s : Substance) (weight : ℝ) : ℝ :=
  s.k * weight

/-- Theorem stating the relationship between volumes of different weights -/
theorem volume_ratio (s : Substance) (w1 w2 v1 : ℝ) (hw1 : w1 > 0) (hw2 : w2 > 0) (hv1 : v1 > 0)
    (h : volume s w1 = v1) :
    volume s w2 = v1 * (w2 / w1) := by
  sorry

/-- Main theorem proving the volume for 112 ounces given the volume for 84 ounces -/
theorem volume_112oz (s : Substance) (h : volume s 84 = 36) :
    volume s 112 = 48 := by
  sorry

end NUMINAMATH_CALUDE_volume_ratio_volume_112oz_l1795_179521


namespace NUMINAMATH_CALUDE_constant_b_value_l1795_179533

theorem constant_b_value (a b : ℝ) : 
  (∀ x : ℝ, (x - 3) * (x - a) = x^2 - b*x - 10) → b = -1/3 := by
  sorry

end NUMINAMATH_CALUDE_constant_b_value_l1795_179533


namespace NUMINAMATH_CALUDE_system_solution_l1795_179529

theorem system_solution (k : ℝ) : 
  (∃ x y : ℝ, 4*x - 3*y = k ∧ 2*x + 3*y = 5 ∧ x = y) → k = 1 := by
  sorry

end NUMINAMATH_CALUDE_system_solution_l1795_179529


namespace NUMINAMATH_CALUDE_least_integer_with_12_factors_l1795_179584

-- Define a function to count the number of positive factors of a natural number
def countFactors (n : ℕ) : ℕ := sorry

-- Define a function to check if a number has exactly 12 factors
def has12Factors (n : ℕ) : Prop := countFactors n = 12

-- Theorem statement
theorem least_integer_with_12_factors :
  ∃ (k : ℕ), k > 0 ∧ has12Factors k ∧ ∀ (m : ℕ), m > 0 → has12Factors m → k ≤ m :=
sorry

end NUMINAMATH_CALUDE_least_integer_with_12_factors_l1795_179584


namespace NUMINAMATH_CALUDE_other_donation_is_100_l1795_179547

/-- Represents the fundraiser for basketball equipment -/
structure Fundraiser where
  goal : ℕ
  bronze_donation : ℕ
  silver_donation : ℕ
  bronze_count : ℕ
  silver_count : ℕ
  other_count : ℕ
  final_day_goal : ℕ

/-- Calculates the amount donated by the family with another status -/
def other_donation (f : Fundraiser) : ℕ :=
  f.goal - (f.bronze_donation * f.bronze_count + f.silver_donation * f.silver_count + f.final_day_goal)

/-- Theorem stating that the family with another status donated $100 -/
theorem other_donation_is_100 (f : Fundraiser)
  (h1 : f.goal = 750)
  (h2 : f.bronze_donation = 25)
  (h3 : f.silver_donation = 50)
  (h4 : f.bronze_count = 10)
  (h5 : f.silver_count = 7)
  (h6 : f.other_count = 1)
  (h7 : f.final_day_goal = 50) :
  other_donation f = 100 := by
  sorry

end NUMINAMATH_CALUDE_other_donation_is_100_l1795_179547


namespace NUMINAMATH_CALUDE_factorial_solutions_l1795_179536

def factorial : ℕ → ℕ
  | 0 => 1
  | n + 1 => (n + 1) * factorial n

theorem factorial_solutions :
  ∀ x y z : ℕ, factorial x + 2^y = factorial z →
    ((x = 0 ∧ y = 0) ∨ (x = 1 ∧ y = 0) ∨ (x = 2 ∧ y = 2)) :=
by sorry

end NUMINAMATH_CALUDE_factorial_solutions_l1795_179536


namespace NUMINAMATH_CALUDE_sqrt_cube_root_power_six_l1795_179579

theorem sqrt_cube_root_power_six : (Real.sqrt ((Real.sqrt 3) ^ 4)) ^ 6 = 729 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_cube_root_power_six_l1795_179579


namespace NUMINAMATH_CALUDE_fourth_rectangle_area_l1795_179531

/-- Represents a rectangle with length and width -/
structure Rectangle where
  length : ℝ
  width : ℝ

/-- Calculates the area of a rectangle -/
def area (r : Rectangle) : ℝ := r.length * r.width

theorem fourth_rectangle_area 
  (large : Rectangle) 
  (small1 small2 small3 small4 : Rectangle) 
  (h1 : small1.length + small3.length = large.length)
  (h2 : small1.width + small2.width = large.width)
  (h3 : small1.length = small2.length)
  (h4 : small1.width = small3.width)
  (h5 : area large = area small1 + area small2 + area small3 + area small4) :
  area small4 = small2.width * small3.length := by
  sorry

#check fourth_rectangle_area

end NUMINAMATH_CALUDE_fourth_rectangle_area_l1795_179531


namespace NUMINAMATH_CALUDE_line_slope_intercept_product_l1795_179546

/-- Given a line passing through points (0, -2) and (1, 1), 
    prove that the product of its slope and y-intercept equals -6 -/
theorem line_slope_intercept_product : 
  ∀ (m b : ℝ), 
    (∀ x : ℝ, b = -2 ∧ m * 0 + b = -2) →  -- line passes through (0, -2)
    (∀ x : ℝ, m * 1 + b = 1) →            -- line passes through (1, 1)
    m * b = -6 := by
  sorry

end NUMINAMATH_CALUDE_line_slope_intercept_product_l1795_179546


namespace NUMINAMATH_CALUDE_largest_812_double_l1795_179512

/-- Converts a base-10 number to its base-8 representation as a list of digits -/
def toBase8 (n : ℕ) : List ℕ :=
  sorry

/-- Interprets a list of digits as a base-12 number -/
def fromBase12 (digits : List ℕ) : ℕ :=
  sorry

/-- Checks if a number is an 8-12 double -/
def is812Double (n : ℕ) : Prop :=
  fromBase12 (toBase8 n) = 2 * n

theorem largest_812_double :
  (∀ m : ℕ, is812Double m → m ≤ 4032) ∧ is812Double 4032 :=
sorry

end NUMINAMATH_CALUDE_largest_812_double_l1795_179512


namespace NUMINAMATH_CALUDE_power_function_through_point_l1795_179555

/-- A power function that passes through the point (2, √2) has exponent 1/2 -/
theorem power_function_through_point (f : ℝ → ℝ) (a : ℝ) :
  (∀ x, f x = x^a) →  -- f is a power function with exponent a
  f 2 = Real.sqrt 2 →  -- f passes through the point (2, √2)
  a = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_power_function_through_point_l1795_179555


namespace NUMINAMATH_CALUDE_cookie_flour_weight_l1795_179588

/-- Given the conditions of Matt's cookie baking, prove that each bag of flour weighs 5 pounds -/
theorem cookie_flour_weight 
  (cookies_per_batch : ℕ) 
  (flour_per_batch : ℕ) 
  (num_bags : ℕ) 
  (cookies_eaten : ℕ) 
  (cookies_left : ℕ) 
  (h1 : cookies_per_batch = 12)
  (h2 : flour_per_batch = 2)
  (h3 : num_bags = 4)
  (h4 : cookies_eaten = 15)
  (h5 : cookies_left = 105) :
  (cookies_eaten + cookies_left) * flour_per_batch / (cookies_per_batch * num_bags) = 5 := by
  sorry

#check cookie_flour_weight

end NUMINAMATH_CALUDE_cookie_flour_weight_l1795_179588


namespace NUMINAMATH_CALUDE_complex_division_l1795_179567

/-- Given a complex number z = 1 + ai where a is a positive real number and |z| = √10,
    prove that z / (1 - 2i) = -1 + i -/
theorem complex_division (a : ℝ) (z : ℂ) (h1 : a > 0) (h2 : z = 1 + a * Complex.I) 
    (h3 : Complex.abs z = Real.sqrt 10) : 
  z / (1 - 2 * Complex.I) = -1 + Complex.I := by
  sorry

end NUMINAMATH_CALUDE_complex_division_l1795_179567


namespace NUMINAMATH_CALUDE_equivalent_ratios_l1795_179569

theorem equivalent_ratios (x : ℚ) : (3 : ℚ) / 12 = x / 16 → x = 4 := by
  sorry

end NUMINAMATH_CALUDE_equivalent_ratios_l1795_179569


namespace NUMINAMATH_CALUDE_heart_properties_l1795_179542

def heart (x y : ℝ) : ℝ := |x - y|

theorem heart_properties :
  ∀ x y : ℝ,
  (heart x y = heart y x) ∧
  (3 * heart x y = heart (3 * x) (3 * y)) ∧
  (heart (x + 1) (y + 1) = heart x y) ∧
  (heart x x = 0) ∧
  (heart x y ≥ 0) ∧
  (heart x y > 0 ↔ x ≠ y) := by
  sorry

end NUMINAMATH_CALUDE_heart_properties_l1795_179542


namespace NUMINAMATH_CALUDE_hare_hunt_probability_l1795_179594

theorem hare_hunt_probability (p1 p2 p3 : ℝ) 
  (h1 : p1 = 3/5) 
  (h2 : p2 = 3/10) 
  (h3 : p3 = 1/10) : 
  1 - (1 - p1) * (1 - p2) * (1 - p3) = 0.748 := by
  sorry

end NUMINAMATH_CALUDE_hare_hunt_probability_l1795_179594


namespace NUMINAMATH_CALUDE_inequality_proof_l1795_179574

theorem inequality_proof (a b c d : ℝ) 
  (pos_a : 0 < a) (pos_b : 0 < b) (pos_c : 0 < c) (pos_d : 0 < d)
  (sum_eq_four : a + b + c + d = 4) :
  a^2 * b * c + b^2 * d * a + c^2 * d * a + d^2 * b * c ≤ 4 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l1795_179574


namespace NUMINAMATH_CALUDE_journey_speed_calculation_l1795_179514

theorem journey_speed_calculation (total_distance : ℝ) (total_time : ℝ) (first_half_speed : ℝ) :
  total_distance = 400 ∧ 
  total_time = 30 ∧ 
  first_half_speed = 20 →
  (total_distance / 2) / ((total_time - (total_distance / 2) / first_half_speed)) = 10 :=
by sorry

end NUMINAMATH_CALUDE_journey_speed_calculation_l1795_179514


namespace NUMINAMATH_CALUDE_certain_number_exists_l1795_179580

theorem certain_number_exists : ∃ x : ℝ, ((x + 10) * 2 / 2)^3 - 2 = 120 / 3 := by
  sorry

end NUMINAMATH_CALUDE_certain_number_exists_l1795_179580


namespace NUMINAMATH_CALUDE_bouncy_ball_difference_l1795_179595

/-- Proves that the difference between red and yellow bouncy balls is 18 -/
theorem bouncy_ball_difference :
  ∀ (red_packs yellow_packs balls_per_pack : ℕ),
  red_packs = 5 →
  yellow_packs = 4 →
  balls_per_pack = 18 →
  red_packs * balls_per_pack - yellow_packs * balls_per_pack = 18 :=
by
  sorry

#check bouncy_ball_difference

end NUMINAMATH_CALUDE_bouncy_ball_difference_l1795_179595


namespace NUMINAMATH_CALUDE_binomial_expansion_sum_l1795_179582

theorem binomial_expansion_sum (a b c d e f : ℤ) : 
  (∀ x : ℤ, (x - 2)^5 = a*x^5 + b*x^4 + c*x^3 + d*x^2 + e*x + f) →
  16*(a + b) + 4*(c + d) + (e + f) = -256 := by
sorry

end NUMINAMATH_CALUDE_binomial_expansion_sum_l1795_179582


namespace NUMINAMATH_CALUDE_function_equals_square_l1795_179572

-- Define the property that f has the same number of intersections as x^2 with any line
def SameIntersections (f : ℝ → ℝ) : Prop :=
  ∀ (m c : ℝ), (∃ (x : ℝ), f x = m * x + c) ↔ (∃ (x : ℝ), x^2 = m * x + c)

-- State the theorem
theorem function_equals_square (f : ℝ → ℝ) (h : SameIntersections f) : 
  ∀ x : ℝ, f x = x^2 := by
sorry

end NUMINAMATH_CALUDE_function_equals_square_l1795_179572


namespace NUMINAMATH_CALUDE_max_profit_at_12_ships_l1795_179576

-- Define the output function
def R (x : ℕ) : ℤ := 3700 * x + 45 * x^2 - 10 * x^3

-- Define the cost function
def C (x : ℕ) : ℤ := 460 * x + 5000

-- Define the profit function
def P (x : ℕ) : ℤ := R x - C x

-- Define the marginal profit function
def MP (x : ℕ) : ℤ := P (x + 1) - P x

-- Theorem statement
theorem max_profit_at_12_ships :
  ∀ x : ℕ, 1 ≤ x → x ≤ 20 → P x ≤ P 12 :=
sorry

end NUMINAMATH_CALUDE_max_profit_at_12_ships_l1795_179576


namespace NUMINAMATH_CALUDE_ball_selection_count_l1795_179511

/-- Represents the set of colors available for the balls -/
inductive Color
| Red
| Yellow
| Blue

/-- Represents the set of letters used to mark the balls -/
inductive Letter
| A
| B
| C
| D
| E

/-- The total number of balls for each color -/
def ballsPerColor : Nat := 5

/-- The total number of colors -/
def numColors : Nat := 3

/-- The number of balls to be selected -/
def ballsToSelect : Nat := 5

/-- Calculates the number of ways to select the balls -/
def selectBalls : Nat := numColors ^ ballsToSelect

theorem ball_selection_count :
  selectBalls = 243 :=
sorry

end NUMINAMATH_CALUDE_ball_selection_count_l1795_179511


namespace NUMINAMATH_CALUDE_total_matting_cost_l1795_179548

/-- Represents the dimensions of a room -/
structure RoomDimensions where
  length : ℝ
  width : ℝ
  height : ℝ

/-- Represents a room with its dimensions and matting cost -/
structure Room where
  dimensions : RoomDimensions
  mattingCostPerSquareMeter : ℝ

/-- Calculates the floor area of a room -/
def floorArea (room : Room) : ℝ :=
  room.dimensions.length * room.dimensions.width

/-- Calculates the matting cost for a room -/
def mattingCost (room : Room) : ℝ :=
  floorArea room * room.mattingCostPerSquareMeter

/-- The three rooms in the house -/
def hall : Room :=
  { dimensions := { length := 20, width := 15, height := 5 },
    mattingCostPerSquareMeter := 40 }

def bedroom : Room :=
  { dimensions := { length := 10, width := 5, height := 4 },
    mattingCostPerSquareMeter := 35 }

def study : Room :=
  { dimensions := { length := 8, width := 6, height := 3 },
    mattingCostPerSquareMeter := 45 }

/-- Theorem: The total cost of matting for all three rooms is 15910 -/
theorem total_matting_cost :
  mattingCost hall + mattingCost bedroom + mattingCost study = 15910 := by
  sorry

end NUMINAMATH_CALUDE_total_matting_cost_l1795_179548


namespace NUMINAMATH_CALUDE_bicycle_speed_correct_l1795_179518

/-- The speed of bicycles that satisfies the given conditions -/
def bicycle_speed : ℝ := 15

theorem bicycle_speed_correct :
  let distance : ℝ := 10
  let car_speed : ℝ → ℝ := λ x => 2 * x
  let bicycle_time : ℝ → ℝ := λ x => distance / x
  let car_time : ℝ → ℝ := λ x => distance / (car_speed x)
  let time_difference : ℝ := 1 / 3
  bicycle_time bicycle_speed = car_time bicycle_speed + time_difference :=
by sorry

end NUMINAMATH_CALUDE_bicycle_speed_correct_l1795_179518


namespace NUMINAMATH_CALUDE_rhombus_diagonal_sum_max_l1795_179598

theorem rhombus_diagonal_sum_max (s x y : ℝ) : 
  s = 5 → 
  x^2 + y^2 = 4 * s^2 →
  x ≥ 6 →
  y ≤ 6 →
  x + y ≤ 14 :=
by sorry

end NUMINAMATH_CALUDE_rhombus_diagonal_sum_max_l1795_179598


namespace NUMINAMATH_CALUDE_freshman_percentage_l1795_179590

theorem freshman_percentage (total_students : ℝ) (freshmen : ℝ) 
  (h1 : freshmen > 0) (h2 : total_students > 0) :
  let liberal_arts_fraction : ℝ := 0.6
  let psychology_fraction : ℝ := 0.5
  let freshmen_psych_liberal_fraction : ℝ := 0.24
  (liberal_arts_fraction * psychology_fraction * (freshmen / total_students) = 
    freshmen_psych_liberal_fraction) →
  freshmen / total_students = 0.8 := by
sorry

end NUMINAMATH_CALUDE_freshman_percentage_l1795_179590


namespace NUMINAMATH_CALUDE_intersection_A_B_l1795_179549

def A : Set ℝ := {x | |x| < 2}
def B : Set ℝ := {0, 1, 2}

theorem intersection_A_B : A ∩ B = {0, 1} := by
  sorry

end NUMINAMATH_CALUDE_intersection_A_B_l1795_179549


namespace NUMINAMATH_CALUDE_acoustic_guitar_price_l1795_179553

theorem acoustic_guitar_price (total_guitars : ℕ) (total_revenue : ℕ) 
  (electric_price : ℕ) (electric_count : ℕ) :
  total_guitars = 9 →
  total_revenue = 3611 →
  electric_price = 479 →
  electric_count = 4 →
  (total_revenue - electric_price * electric_count) / (total_guitars - electric_count) = 339 := by
sorry

end NUMINAMATH_CALUDE_acoustic_guitar_price_l1795_179553


namespace NUMINAMATH_CALUDE_smallest_fraction_greater_than_three_fourths_l1795_179587

def is_two_digit (n : ℕ) : Prop := 10 ≤ n ∧ n ≤ 99

theorem smallest_fraction_greater_than_three_fourths :
  ∃ (a b : ℕ), 
    is_two_digit a ∧ 
    is_two_digit b ∧ 
    (a : ℚ) / b > 3 / 4 ∧
    (∀ (c d : ℕ), is_two_digit c → is_two_digit d → (c : ℚ) / d > 3 / 4 → a ≤ c) ∧
    a = 73 :=
by sorry

end NUMINAMATH_CALUDE_smallest_fraction_greater_than_three_fourths_l1795_179587


namespace NUMINAMATH_CALUDE_p_sufficient_t_l1795_179500

-- Define the propositions
variable (p q r s t : Prop)

-- Define the conditions
axiom p_r_sufficient_q : (p → q) ∧ (r → q)
axiom s_necessary_sufficient_q : (q ↔ s)
axiom t_necessary_s : (s → t)
axiom t_sufficient_r : (t → r)

-- Theorem to prove
theorem p_sufficient_t : p → t := by sorry

end NUMINAMATH_CALUDE_p_sufficient_t_l1795_179500


namespace NUMINAMATH_CALUDE_determinant_transformation_l1795_179503

theorem determinant_transformation (p q r s : ℝ) 
  (h : Matrix.det !![p, q; r, s] = 6) : 
  Matrix.det !![p, 9*p + 4*q; r, 9*r + 4*s] = 24 := by
  sorry

end NUMINAMATH_CALUDE_determinant_transformation_l1795_179503


namespace NUMINAMATH_CALUDE_sum_of_numbers_l1795_179589

theorem sum_of_numbers (a b c : ℝ) (ha : a = 0.8) (hb : b = 1/2) (hc : c = 0.9) :
  (if a ≥ 0.3 then a else 0) + (if b ≥ 0.3 then b else 0) + (if c ≥ 0.3 then c else 0) = 2.2 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_numbers_l1795_179589


namespace NUMINAMATH_CALUDE_problem_solution_l1795_179559

-- Define the function f(x) = ax^3 + bx^2
def f (a b x : ℝ) : ℝ := a * x^3 + b * x^2

-- Define the derivative of f
def f_deriv (a b x : ℝ) : ℝ := 3 * a * x^2 + 2 * b * x

theorem problem_solution :
  ∀ (a b : ℝ),
    (f a b 1 = 4 ∧ f_deriv a b 1 = 9) →
    (a = 1 ∧ b = 3) ∧
    ∀ (m : ℝ),
      (∀ x ∈ Set.Icc m (m + 1), f_deriv 1 3 x ≥ 0) →
      (m ≥ 0 ∨ m ≤ -3) :=
by sorry

end NUMINAMATH_CALUDE_problem_solution_l1795_179559


namespace NUMINAMATH_CALUDE_perpendicular_vectors_k_value_l1795_179581

/-- Given vectors a and b, if they are perpendicular, then k = 3 -/
theorem perpendicular_vectors_k_value (k : ℝ) :
  let a : ℝ × ℝ := (2*k - 3, -6)
  let b : ℝ × ℝ := (2, 1)
  (a.1 * b.1 + a.2 * b.2 = 0) → k = 3 := by
  sorry

end NUMINAMATH_CALUDE_perpendicular_vectors_k_value_l1795_179581


namespace NUMINAMATH_CALUDE_isosceles_triangle_perimeter_isosceles_triangle_perimeter_proof_l1795_179522

/-- An isosceles triangle with side lengths 5 and 11 has a perimeter of 27. -/
theorem isosceles_triangle_perimeter : ℝ → ℝ → ℝ → Prop :=
  fun a b c =>
    (a = 5 ∧ b = 11 ∧ c = 11) ∨ (a = 11 ∧ b = 5 ∧ c = 11) →
    IsoscelesTriangle a b c →
    a + b + c = 27
  where
    IsoscelesTriangle a b c := (a = b ∧ a ≠ c) ∨ (b = c ∧ b ≠ a) ∨ (a = c ∧ a ≠ b)

theorem isosceles_triangle_perimeter_proof : isosceles_triangle_perimeter 5 11 11 := by
  sorry

end NUMINAMATH_CALUDE_isosceles_triangle_perimeter_isosceles_triangle_perimeter_proof_l1795_179522


namespace NUMINAMATH_CALUDE_average_age_of_nine_students_l1795_179530

theorem average_age_of_nine_students 
  (total_students : ℕ) 
  (average_age_all : ℝ) 
  (students_group1 : ℕ) 
  (average_age_group1 : ℝ) 
  (age_last_student : ℕ) 
  (h1 : total_students = 20)
  (h2 : average_age_all = 20)
  (h3 : students_group1 = 10)
  (h4 : average_age_group1 = 24)
  (h5 : age_last_student = 61) :
  let students_group2 := total_students - students_group1 - 1
  let total_age_all := average_age_all * total_students
  let total_age_group1 := average_age_group1 * students_group1
  let total_age_group2 := total_age_all - total_age_group1 - age_last_student
  (total_age_group2 / students_group2 : ℝ) = 11 := by
  sorry

end NUMINAMATH_CALUDE_average_age_of_nine_students_l1795_179530


namespace NUMINAMATH_CALUDE_expression_evaluation_l1795_179597

theorem expression_evaluation :
  (((3^0 : ℝ) - 1 + 4^2 - 3)^(-1 : ℝ)) * 4 = 4/13 := by sorry

end NUMINAMATH_CALUDE_expression_evaluation_l1795_179597


namespace NUMINAMATH_CALUDE_tripling_radius_and_negative_quantity_l1795_179538

theorem tripling_radius_and_negative_quantity : ∀ (r : ℝ) (x : ℝ), 
  r > 0 → x < 0 → 
  (π * (3 * r)^2 ≠ 3 * (π * r^2)) ∧ (3 * x ≤ x) := by sorry

end NUMINAMATH_CALUDE_tripling_radius_and_negative_quantity_l1795_179538


namespace NUMINAMATH_CALUDE_inequality_solution_l1795_179592

theorem inequality_solution :
  {x : ℝ | (x^2 - 9) / (x^2 - 16) > 0} = {x : ℝ | x < -4 ∨ x > 4} := by
  sorry

end NUMINAMATH_CALUDE_inequality_solution_l1795_179592


namespace NUMINAMATH_CALUDE_min_bills_for_payment_l1795_179526

/-- Represents the available denominations of bills and coins --/
structure Denominations :=
  (ten_dollar : ℕ)
  (five_dollar : ℕ)
  (two_dollar : ℕ)
  (one_dollar : ℕ)
  (fifty_cent : ℕ)

/-- Calculates the minimum number of bills and coins needed to pay a given amount --/
def min_bills_and_coins (d : Denominations) (amount : ℚ) : ℕ :=
  sorry

/-- Tim's available bills and coins --/
def tims_denominations : Denominations :=
  { ten_dollar := 15
  , five_dollar := 7
  , two_dollar := 12
  , one_dollar := 20
  , fifty_cent := 10 }

/-- The theorem stating that Tim needs 17 bills and coins to pay $152.50 --/
theorem min_bills_for_payment :
  min_bills_and_coins tims_denominations (152.5 : ℚ) = 17 :=
sorry

end NUMINAMATH_CALUDE_min_bills_for_payment_l1795_179526


namespace NUMINAMATH_CALUDE_jelly_bean_probability_l1795_179535

theorem jelly_bean_probability (p_red p_orange p_green p_yellow : ℝ) :
  p_red = 0.15 →
  p_orange = 0.35 →
  p_green = 0.25 →
  p_red + p_orange + p_green + p_yellow = 1 →
  p_yellow = 0.25 := by
sorry

end NUMINAMATH_CALUDE_jelly_bean_probability_l1795_179535


namespace NUMINAMATH_CALUDE_divisible_by_two_l1795_179571

theorem divisible_by_two (m n : ℕ) : 
  2 ∣ (5*m + n + 1) * (3*m - n + 4) := by
  sorry

end NUMINAMATH_CALUDE_divisible_by_two_l1795_179571


namespace NUMINAMATH_CALUDE_book_pages_count_l1795_179577

theorem book_pages_count (total_notebooks : ℕ) (sum_of_four_pages : ℕ) : 
  total_notebooks = 12 ∧ sum_of_four_pages = 338 → 
  ∃ (total_pages : ℕ), 
    total_pages = 288 ∧
    (total_pages / total_notebooks : ℚ) + 1 + 
    (total_pages / total_notebooks : ℚ) + 2 + 
    (total_pages / 3 : ℚ) - 1 + 
    (total_pages / 3 : ℚ) = sum_of_four_pages := by
  sorry

#check book_pages_count

end NUMINAMATH_CALUDE_book_pages_count_l1795_179577


namespace NUMINAMATH_CALUDE_inequality_proof_l1795_179565

theorem inequality_proof (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) (h_xyz : x * y * z = 1) :
  (1 + x) * (1 + y) * (1 + z) ≥ 2 * (1 + (y / x)^(1/3) + (z / y)^(1/3) + (x / z)^(1/3)) :=
by sorry

end NUMINAMATH_CALUDE_inequality_proof_l1795_179565


namespace NUMINAMATH_CALUDE_bobby_candy_problem_l1795_179583

/-- Proves that Bobby ate 6 pieces of candy initially -/
theorem bobby_candy_problem :
  ∀ (initial_candy : ℕ) (eaten_initially : ℕ) (eaten_later : ℕ) (remaining_candy : ℕ),
    initial_candy = 22 →
    eaten_later = 5 →
    remaining_candy = 8 →
    initial_candy - (eaten_initially + eaten_initially / 2 + eaten_later) = remaining_candy →
    eaten_initially = 6 := by
  sorry

end NUMINAMATH_CALUDE_bobby_candy_problem_l1795_179583


namespace NUMINAMATH_CALUDE_binomial_expansion_properties_l1795_179507

/-- The binomial expansion of (√x + 2/x²)¹⁰ -/
def binomial_expansion (x : ℝ) : ℕ → ℝ :=
  λ r => (Nat.choose 10 r) * (2^r) * (x^((10 - 5*r)/2))

/-- A term in the expansion is rational if its exponent is an integer -/
def is_rational_term (r : ℕ) : Prop :=
  (10 - 5*r) % 2 = 0

theorem binomial_expansion_properties :
  (∃ (S : Finset ℕ), S.card = 6 ∧ ∀ r, r ∈ S ↔ is_rational_term r) ∧
  (∃ r : ℕ, r = 7 ∧ ∀ k : ℕ, k ≠ r → |binomial_expansion 1 r| ≥ |binomial_expansion 1 k|) ∧
  binomial_expansion 1 7 = 15360 := by sorry

end NUMINAMATH_CALUDE_binomial_expansion_properties_l1795_179507


namespace NUMINAMATH_CALUDE_fourth_root_equivalence_l1795_179578

theorem fourth_root_equivalence (x : ℝ) (hx : x > 0) : (x^2 * x^(1/2))^(1/4) = x^(5/8) := by
  sorry

end NUMINAMATH_CALUDE_fourth_root_equivalence_l1795_179578


namespace NUMINAMATH_CALUDE_problem_solution_l1795_179515

def has_at_least_four_divisors (n : ℕ) : Prop :=
  (Nat.divisors n).card ≥ 4

def divisor_differences_divide (n : ℕ) : Prop :=
  ∀ a b : ℕ, a ∣ n → b ∣ n → 1 < a → a < b → b < n → (b - a) ∣ n

def satisfies_conditions (n : ℕ) : Prop :=
  has_at_least_four_divisors n ∧ divisor_differences_divide n

theorem problem_solution : 
  {n : ℕ | satisfies_conditions n} = {6, 8, 12} := by sorry

end NUMINAMATH_CALUDE_problem_solution_l1795_179515


namespace NUMINAMATH_CALUDE_min_value_of_z_l1795_179519

theorem min_value_of_z (x y : ℝ) :
  x^2 + 3*y^2 + 8*x - 6*y + 30 ≥ 11 ∧
  ∃ (x y : ℝ), x^2 + 3*y^2 + 8*x - 6*y + 30 = 11 := by
sorry

end NUMINAMATH_CALUDE_min_value_of_z_l1795_179519


namespace NUMINAMATH_CALUDE_ramesh_profit_share_l1795_179566

/-- Calculates the share of profit for a partner in a business partnership --/
def calculate_profit_share (investment1 : ℕ) (investment2 : ℕ) (total_profit : ℕ) : ℕ :=
  (investment2 * total_profit) / (investment1 + investment2)

/-- Theorem stating that Ramesh's share of the profit is 11,875 --/
theorem ramesh_profit_share :
  calculate_profit_share 24000 40000 19000 = 11875 := by
  sorry

end NUMINAMATH_CALUDE_ramesh_profit_share_l1795_179566


namespace NUMINAMATH_CALUDE_rectangular_sheet_area_l1795_179573

theorem rectangular_sheet_area :
  ∀ (area_small area_large total_area : ℝ),
  area_large = 4 * area_small →
  area_large - area_small = 2208 →
  total_area = area_small + area_large →
  total_area = 3680 :=
by
  sorry

end NUMINAMATH_CALUDE_rectangular_sheet_area_l1795_179573


namespace NUMINAMATH_CALUDE_max_imaginary_part_of_roots_l1795_179524

open Complex

theorem max_imaginary_part_of_roots (z : ℂ) (k : ℤ) :
  z^12 - z^9 + z^6 - z^3 + 1 = 0 →
  z = exp (I * Real.pi * (1/15 + 2/15 * k)) →
  ∃ θ : ℝ, -Real.pi/2 ≤ θ ∧ θ ≤ Real.pi/2 ∧
    (∀ w : ℂ, w^12 - w^9 + w^6 - w^3 + 1 = 0 →
      Complex.abs (Complex.im w) ≤ Real.sin (7*Real.pi/30)) :=
by sorry

end NUMINAMATH_CALUDE_max_imaginary_part_of_roots_l1795_179524
