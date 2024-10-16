import Mathlib

namespace NUMINAMATH_CALUDE_joan_toy_cars_cost_l3637_363779

theorem joan_toy_cars_cost (total_toys cost_skateboard cost_trucks : ℚ)
  (h1 : total_toys = 25.62)
  (h2 : cost_skateboard = 4.88)
  (h3 : cost_trucks = 5.86) :
  total_toys - cost_skateboard - cost_trucks = 14.88 := by
  sorry

end NUMINAMATH_CALUDE_joan_toy_cars_cost_l3637_363779


namespace NUMINAMATH_CALUDE_square_roots_problem_l3637_363753

theorem square_roots_problem (x : ℝ) :
  (x + 1 > 0) ∧ (4 - 2*x > 0) ∧ (x + 1)^2 = (4 - 2*x)^2 →
  (x + 1)^2 = 36 :=
by sorry

end NUMINAMATH_CALUDE_square_roots_problem_l3637_363753


namespace NUMINAMATH_CALUDE_jake_has_nine_peaches_l3637_363761

/-- Jake has 7 fewer peaches than Steven and 9 more peaches than Jill. Steven has 16 peaches. -/
def peach_problem (jake steven jill : ℕ) : Prop :=
  jake + 7 = steven ∧ jake = jill + 9 ∧ steven = 16

/-- Prove that Jake has 9 peaches. -/
theorem jake_has_nine_peaches :
  ∀ jake steven jill : ℕ, peach_problem jake steven jill → jake = 9 := by
  sorry

end NUMINAMATH_CALUDE_jake_has_nine_peaches_l3637_363761


namespace NUMINAMATH_CALUDE_vip_ticket_price_l3637_363798

/-- Represents the price of concert tickets and savings --/
structure ConcertTickets where
  savings : ℕ
  vipTickets : ℕ
  regularTickets : ℕ
  regularPrice : ℕ
  remainingMoney : ℕ

/-- Theorem: The price of each VIP ticket is $100 --/
theorem vip_ticket_price (ct : ConcertTickets)
  (h1 : ct.savings = 500)
  (h2 : ct.vipTickets = 2)
  (h3 : ct.regularTickets = 3)
  (h4 : ct.regularPrice = 50)
  (h5 : ct.remainingMoney = 150) :
  (ct.savings - ct.remainingMoney - ct.regularTickets * ct.regularPrice) / ct.vipTickets = 100 := by
  sorry


end NUMINAMATH_CALUDE_vip_ticket_price_l3637_363798


namespace NUMINAMATH_CALUDE_complex_product_l3637_363777

theorem complex_product (A B C : ℂ) : 
  A = 7 + 3*I ∧ B = I ∧ C = 7 - 3*I → A * B * C = 58 * I :=
by sorry

end NUMINAMATH_CALUDE_complex_product_l3637_363777


namespace NUMINAMATH_CALUDE_parallel_lines_condition_l3637_363792

/-- Represents a line in the form ax + by + c = 0 --/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Checks if two lines are parallel --/
def parallel (l1 l2 : Line) : Prop :=
  l1.a * l2.b = l1.b * l2.a ∧ l1.a ≠ 0 ∧ l2.a ≠ 0

/-- The main theorem to be proved --/
theorem parallel_lines_condition (a : ℝ) :
  (parallel ⟨a, 1, -1⟩ ⟨1, a, 2⟩ → a = 1) ∧
  ¬(a = 1 → parallel ⟨a, 1, -1⟩ ⟨1, a, 2⟩) := by
  sorry

end NUMINAMATH_CALUDE_parallel_lines_condition_l3637_363792


namespace NUMINAMATH_CALUDE_quadratic_trinomial_minimum_l3637_363764

theorem quadratic_trinomial_minimum (a b : ℝ) (h1 : a > b)
  (h2 : ∀ x : ℝ, a * x^2 + 2*x + b ≥ 0)
  (h3 : ∃ x₀ : ℝ, a * x₀^2 + 2*x₀ + b = 0) :
  (∀ x : ℝ, (a^2 + b^2) / (a - b) ≥ 2 * Real.sqrt 2) ∧
  (∃ x : ℝ, (a^2 + b^2) / (a - b) = 2 * Real.sqrt 2) :=
sorry

end NUMINAMATH_CALUDE_quadratic_trinomial_minimum_l3637_363764


namespace NUMINAMATH_CALUDE_cash_realized_proof_l3637_363770

/-- Given an amount before brokerage and a brokerage rate, calculates the cash realized after brokerage. -/
def cash_realized (amount_before_brokerage : ℚ) (brokerage_rate : ℚ) : ℚ :=
  amount_before_brokerage - (amount_before_brokerage * brokerage_rate)

/-- Theorem stating that for the given conditions, the cash realized is 104.7375 -/
theorem cash_realized_proof :
  let amount_before_brokerage : ℚ := 105
  let brokerage_rate : ℚ := 1 / 400
  cash_realized amount_before_brokerage brokerage_rate = 104.7375 := by
  sorry

#eval cash_realized 105 (1/400)

end NUMINAMATH_CALUDE_cash_realized_proof_l3637_363770


namespace NUMINAMATH_CALUDE_greatest_whole_number_inequality_zero_satisfies_inequality_no_positive_integer_satisfies_inequality_l3637_363767

theorem greatest_whole_number_inequality (x : ℤ) : 
  (6 * x - 4 < 5 - 3 * x) → x ≤ 0 :=
by sorry

theorem zero_satisfies_inequality : 
  6 * 0 - 4 < 5 - 3 * 0 :=
by sorry

theorem no_positive_integer_satisfies_inequality (x : ℤ) :
  x > 0 → ¬(6 * x - 4 < 5 - 3 * x) :=
by sorry

end NUMINAMATH_CALUDE_greatest_whole_number_inequality_zero_satisfies_inequality_no_positive_integer_satisfies_inequality_l3637_363767


namespace NUMINAMATH_CALUDE_point_on_line_l3637_363776

/-- The value of m for which the point (m + 1, 3) lies on the line x + y + 1 = 0 -/
theorem point_on_line (m : ℝ) : (m + 1) + 3 + 1 = 0 ↔ m = -5 := by sorry

end NUMINAMATH_CALUDE_point_on_line_l3637_363776


namespace NUMINAMATH_CALUDE_absolute_value_equality_implies_product_zero_l3637_363704

theorem absolute_value_equality_implies_product_zero (x y : ℝ) :
  |x - Real.log y| = x + Real.log y → x * (y - 1) = 0 := by
  sorry

end NUMINAMATH_CALUDE_absolute_value_equality_implies_product_zero_l3637_363704


namespace NUMINAMATH_CALUDE_sin_2alpha_in_terms_of_y_l3637_363722

theorem sin_2alpha_in_terms_of_y (α y : ℝ) (h1 : 0 < α ∧ α < π/2) (h2 : y > 0) 
  (h3 : Real.cos (α/2) = Real.sqrt ((y+1)/(2*y))) : 
  Real.sin (2*α) = (2 * Real.sqrt (y^2 - 1)) / y := by
sorry

end NUMINAMATH_CALUDE_sin_2alpha_in_terms_of_y_l3637_363722


namespace NUMINAMATH_CALUDE_intersection_range_l3637_363759

theorem intersection_range (k : ℝ) : 
  (∃ x₁ x₂ y₁ y₂ : ℝ, 
    x₁ ≠ x₂ ∧ 
    x₁ > 0 ∧ x₂ > 0 ∧
    y₁ = k * x₁ + 2 ∧
    y₂ = k * x₂ + 2 ∧
    x₁^2 - y₁^2 = 6 ∧
    x₂^2 - y₂^2 = 6) →
  -Real.sqrt 15 / 3 < k ∧ k < -1 :=
by sorry

end NUMINAMATH_CALUDE_intersection_range_l3637_363759


namespace NUMINAMATH_CALUDE_adjacent_lateral_faces_angle_l3637_363717

/-- A regular quadrilateral pyramid is a pyramid with a square base and four congruent triangular faces. -/
structure RegularQuadrilateralPyramid where
  /-- The side length of the square base -/
  base_side : ℝ
  /-- The angle between a lateral face and the base plane -/
  lateral_base_angle : ℝ

/-- The theorem states that if the lateral face of a regular quadrilateral pyramid
    forms a 45° angle with the base plane, then the angle between adjacent lateral faces is 120°. -/
theorem adjacent_lateral_faces_angle
  (pyramid : RegularQuadrilateralPyramid)
  (h : pyramid.lateral_base_angle = Real.pi / 4) :
  let adjacent_angle := Real.arccos (-1/3)
  adjacent_angle = 2 * Real.pi / 3 := by
  sorry

end NUMINAMATH_CALUDE_adjacent_lateral_faces_angle_l3637_363717


namespace NUMINAMATH_CALUDE_third_class_duration_l3637_363714

/-- Calculates the duration of the third class in a course --/
theorem third_class_duration 
  (weeks : ℕ) 
  (fixed_class_hours : ℕ) 
  (fixed_classes_per_week : ℕ) 
  (homework_hours : ℕ) 
  (total_hours : ℕ) 
  (h1 : weeks = 24)
  (h2 : fixed_class_hours = 3)
  (h3 : fixed_classes_per_week = 2)
  (h4 : homework_hours = 4)
  (h5 : total_hours = 336) :
  ∃ (third_class_hours : ℕ), 
    (fixed_classes_per_week * fixed_class_hours + third_class_hours + homework_hours) * weeks = total_hours ∧
    third_class_hours = 4 :=
by sorry

end NUMINAMATH_CALUDE_third_class_duration_l3637_363714


namespace NUMINAMATH_CALUDE_decimal_50_to_ternary_l3637_363720

/-- Converts a natural number to its ternary (base-3) representation -/
def to_ternary (n : ℕ) : List ℕ :=
  if n = 0 then [0]
  else
    let rec aux (m : ℕ) (acc : List ℕ) : List ℕ :=
      if m = 0 then acc
      else aux (m / 3) ((m % 3) :: acc)
    aux n []

/-- Checks if a list of digits is a valid ternary number -/
def is_valid_ternary (l : List ℕ) : Prop :=
  l.all (λ d => d < 3)

theorem decimal_50_to_ternary :
  let ternary := to_ternary 50
  is_valid_ternary ternary ∧ ternary = [1, 2, 1, 2] := by sorry

end NUMINAMATH_CALUDE_decimal_50_to_ternary_l3637_363720


namespace NUMINAMATH_CALUDE_order_of_roots_l3637_363703

theorem order_of_roots (a b c : ℝ) (ha : a = 2^(4/3)) (hb : b = 3^(2/3)) (hc : c = 25^(1/3)) :
  b < a ∧ a < c := by
  sorry

end NUMINAMATH_CALUDE_order_of_roots_l3637_363703


namespace NUMINAMATH_CALUDE_omega_circle_l3637_363756

open Complex

/-- Given a complex number z satisfying |z - i| = 1, z ≠ 0, z ≠ 2i, and a complex number ω
    such that (ω / (ω - 2i)) * ((z - 2i) / z) is real, prove that ω lies on the circle
    centered at (0, 1) with radius 1, excluding the points (0, 0) and (0, 2). -/
theorem omega_circle (z ω : ℂ) 
  (h1 : abs (z - I) = 1)
  (h2 : z ≠ 0)
  (h3 : z ≠ 2 * I)
  (h4 : ∃ (r : ℝ), ω / (ω - 2 * I) * ((z - 2 * I) / z) = r) :
  abs (ω - I) = 1 ∧ ω ≠ 0 ∧ ω ≠ 2 * I :=
sorry

end NUMINAMATH_CALUDE_omega_circle_l3637_363756


namespace NUMINAMATH_CALUDE_reassembled_prism_surface_area_l3637_363737

/-- Represents the dimensions of a rectangular prism -/
structure PrismDimensions where
  width : ℝ
  height : ℝ
  length : ℝ

/-- Represents the cuts made to the prism -/
structure PrismCuts where
  first_cut : ℝ
  second_cut : ℝ
  third_cut : ℝ

/-- Calculates the surface area of the reassembled prism -/
def surface_area_reassembled (dim : PrismDimensions) (cuts : PrismCuts) : ℝ :=
  sorry

/-- Theorem stating that the surface area of the reassembled prism is 16 square feet -/
theorem reassembled_prism_surface_area 
  (dim : PrismDimensions) 
  (cuts : PrismCuts) 
  (h1 : dim.width = 1) 
  (h2 : dim.height = 1) 
  (h3 : dim.length = 2) 
  (h4 : cuts.first_cut = 1/4) 
  (h5 : cuts.second_cut = 1/5) 
  (h6 : cuts.third_cut = 1/6) : 
  surface_area_reassembled dim cuts = 16 := by
  sorry

end NUMINAMATH_CALUDE_reassembled_prism_surface_area_l3637_363737


namespace NUMINAMATH_CALUDE_jesse_pencils_l3637_363774

/-- Given that Jesse starts with 78 pencils and gives away 44 pencils,
    prove that he ends up with 34 pencils. -/
theorem jesse_pencils :
  let initial_pencils : ℕ := 78
  let pencils_given_away : ℕ := 44
  initial_pencils - pencils_given_away = 34 :=
by sorry

end NUMINAMATH_CALUDE_jesse_pencils_l3637_363774


namespace NUMINAMATH_CALUDE_lcm_gcd_product_15_75_l3637_363788

theorem lcm_gcd_product_15_75 : Nat.lcm 15 75 * Nat.gcd 15 75 = 1125 := by
  sorry

end NUMINAMATH_CALUDE_lcm_gcd_product_15_75_l3637_363788


namespace NUMINAMATH_CALUDE_correct_proposition_l3637_363766

-- Define proposition p
def p : Prop := ∀ x : ℝ, x > 1 → x > 2

-- Define proposition q
def q : Prop := ∀ x y : ℝ, x + y ≠ 2 → x ≠ 1 ∨ y ≠ 1

-- Theorem to prove
theorem correct_proposition : (¬p) ∧ q := by
  sorry

end NUMINAMATH_CALUDE_correct_proposition_l3637_363766


namespace NUMINAMATH_CALUDE_mean_home_runs_l3637_363787

def number_of_players : ℕ := 9

def home_run_distribution : List (ℕ × ℕ) :=
  [(5, 2), (6, 3), (8, 2), (10, 1), (12, 1)]

def total_home_runs : ℕ :=
  (home_run_distribution.map (λ (hr, count) => hr * count)).sum

theorem mean_home_runs :
  (total_home_runs : ℚ) / number_of_players = 66 / 9 := by sorry

end NUMINAMATH_CALUDE_mean_home_runs_l3637_363787


namespace NUMINAMATH_CALUDE_problem_solution_l3637_363728

theorem problem_solution (k : ℝ) (h1 : k ≠ 0) :
  (∀ x : ℝ, (x^2 - k) * (x + k) = x^3 + k * (x^2 - x - 8)) → k = 8 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l3637_363728


namespace NUMINAMATH_CALUDE_library_shelf_theorem_l3637_363778

/-- Represents the thickness of a biology book -/
def biology_thickness : ℝ := 1

/-- Represents the thickness of a history book -/
def history_thickness : ℝ := 2 * biology_thickness

/-- Represents the length of the shelf -/
def shelf_length : ℝ := 1

theorem library_shelf_theorem 
  (B G P Q F : ℕ) 
  (h_distinct : B ≠ G ∧ B ≠ P ∧ B ≠ Q ∧ B ≠ F ∧ 
                G ≠ P ∧ G ≠ Q ∧ G ≠ F ∧ 
                P ≠ Q ∧ P ≠ F ∧ 
                Q ≠ F)
  (h_positive : B > 0 ∧ G > 0 ∧ P > 0 ∧ Q > 0 ∧ F > 0)
  (h_fill1 : B * biology_thickness + G * history_thickness = shelf_length)
  (h_fill2 : P * biology_thickness + Q * history_thickness = shelf_length)
  (h_fill3 : F * biology_thickness = shelf_length) :
  F = B + 2*G ∧ F = P + 2*Q :=
sorry

end NUMINAMATH_CALUDE_library_shelf_theorem_l3637_363778


namespace NUMINAMATH_CALUDE_rectangular_garden_width_l3637_363762

theorem rectangular_garden_width (width length area : ℝ) : 
  length = 3 * width →
  area = length * width →
  area = 675 →
  width = 15 := by
sorry

end NUMINAMATH_CALUDE_rectangular_garden_width_l3637_363762


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l3637_363709

/-- An arithmetic sequence -/
def ArithmeticSequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_sum (a : ℕ → ℝ) :
  ArithmeticSequence a →
  a 1 + a 2 = 3 →
  a 7 + a 8 = 27 →
  a 9 + a 10 = 35 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l3637_363709


namespace NUMINAMATH_CALUDE_total_space_after_compaction_l3637_363749

/-- Represents the types of cans -/
inductive CanType
  | Small
  | Large

/-- Represents the properties of a can type -/
structure CanProperties where
  originalSize : ℕ
  compactionRate : ℚ

/-- Calculates the space taken by a type of can after compaction -/
def spaceAfterCompaction (props : CanProperties) (quantity : ℕ) : ℚ :=
  ↑(props.originalSize * quantity) * props.compactionRate

theorem total_space_after_compaction :
  let smallCanProps : CanProperties := ⟨20, 3/10⟩
  let largeCanProps : CanProperties := ⟨40, 4/10⟩
  let smallCanQuantity : ℕ := 50
  let largeCanQuantity : ℕ := 50
  let totalSpaceAfterCompaction :=
    spaceAfterCompaction smallCanProps smallCanQuantity +
    spaceAfterCompaction largeCanProps largeCanQuantity
  totalSpaceAfterCompaction = 1100 := by
  sorry


end NUMINAMATH_CALUDE_total_space_after_compaction_l3637_363749


namespace NUMINAMATH_CALUDE_pigeon_percentage_among_non_swans_l3637_363780

def bird_distribution (total : ℝ) : ℝ × ℝ × ℝ × ℝ × ℝ := 
  (0.20 * total, 0.30 * total, 0.15 * total, 0.25 * total, 0.10 * total)

theorem pigeon_percentage_among_non_swans (total : ℝ) (h : total > 0) :
  let (geese, swans, herons, ducks, pigeons) := bird_distribution total
  let non_swans := total - swans
  (pigeons / non_swans) * 100 = 14 := by
  sorry

end NUMINAMATH_CALUDE_pigeon_percentage_among_non_swans_l3637_363780


namespace NUMINAMATH_CALUDE_unique_solution_condition_l3637_363731

/-- Given real numbers m, n, p, q, and functions f and g,
    prove that f(g(x)) = g(f(x)) has a unique solution
    if and only if mq = p and q = n -/
theorem unique_solution_condition (m n p q : ℝ)
  (f : ℝ → ℝ) (g : ℝ → ℝ)
  (hf : ∀ x, f x = m * x^2 + n)
  (hg : ∀ x, g x = p * x + q) :
  (∃! x, f (g x) = g (f x)) ↔ (m * q = p ∧ q = n) :=
sorry

end NUMINAMATH_CALUDE_unique_solution_condition_l3637_363731


namespace NUMINAMATH_CALUDE_opposite_of_nine_l3637_363769

/-- The opposite number of 9 is -9 -/
theorem opposite_of_nine : -(9 : ℤ) = -9 := by sorry

end NUMINAMATH_CALUDE_opposite_of_nine_l3637_363769


namespace NUMINAMATH_CALUDE_smallest_unrepresentable_odd_number_l3637_363758

theorem smallest_unrepresentable_odd_number :
  ∀ n : ℕ, n > 0 → n % 2 = 1 →
    (∃ x y : ℕ, x > 0 ∧ y > 0 ∧ n = 7^x - 3 * 2^y) ∨ n ≥ 3 :=
by sorry

end NUMINAMATH_CALUDE_smallest_unrepresentable_odd_number_l3637_363758


namespace NUMINAMATH_CALUDE_parallelogram_angle_measure_l3637_363754

theorem parallelogram_angle_measure (a b : ℝ) : 
  a = 70 → b = a + 40 → b = 110 := by sorry

end NUMINAMATH_CALUDE_parallelogram_angle_measure_l3637_363754


namespace NUMINAMATH_CALUDE_unique_positive_solution_l3637_363791

theorem unique_positive_solution : ∃! (y : ℝ), y > 0 ∧ (y / 100) * y = 9 := by
  sorry

end NUMINAMATH_CALUDE_unique_positive_solution_l3637_363791


namespace NUMINAMATH_CALUDE_gumball_machine_total_l3637_363785

/-- Represents the number of gumballs of each color in a gumball machine. -/
structure GumballMachine where
  red : ℕ
  green : ℕ
  blue : ℕ
  yellow : ℕ
  orange : ℕ

/-- Represents the conditions of the gumball machine problem. -/
def gumball_machine_conditions (m : GumballMachine) : Prop :=
  m.blue = m.red / 2 ∧
  m.green = 4 * m.blue ∧
  m.yellow = (7 * m.blue) / 2 ∧
  m.orange = (2 * (m.red + m.blue)) / 3 ∧
  m.red = (3 * m.yellow) / 2 ∧
  m.yellow = 24

/-- The theorem stating that a gumball machine satisfying the given conditions has 186 gumballs. -/
theorem gumball_machine_total (m : GumballMachine) 
  (h : gumball_machine_conditions m) : 
  m.red + m.green + m.blue + m.yellow + m.orange = 186 := by
  sorry


end NUMINAMATH_CALUDE_gumball_machine_total_l3637_363785


namespace NUMINAMATH_CALUDE_some_employees_not_managers_l3637_363739

-- Define the universe
variable (U : Type)

-- Define the predicates
variable (Employee : U → Prop)
variable (Manager : U → Prop)
variable (Punctual : U → Prop)
variable (Shareholder : U → Prop)

-- State the theorem
theorem some_employees_not_managers
  (h1 : ∃ x, Employee x ∧ ¬Punctual x)
  (h2 : ∀ x, Manager x → Punctual x)
  (h3 : ∃ x, Manager x ∧ Shareholder x) :
  ∃ x, Employee x ∧ ¬Manager x :=
sorry

end NUMINAMATH_CALUDE_some_employees_not_managers_l3637_363739


namespace NUMINAMATH_CALUDE_platform_length_l3637_363752

/-- Given a train of length 450 meters that takes 39 seconds to cross a platform
    and 18 seconds to cross a signal pole, the length of the platform is 525 meters. -/
theorem platform_length (train_length : ℝ) (time_platform : ℝ) (time_pole : ℝ) :
  train_length = 450 →
  time_platform = 39 →
  time_pole = 18 →
  (train_length / time_pole) * time_platform - train_length = 525 :=
by sorry

end NUMINAMATH_CALUDE_platform_length_l3637_363752


namespace NUMINAMATH_CALUDE_geometric_sequence_constant_l3637_363738

/-- A geometric sequence with positive terms -/
def GeometricSequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, r > 0 ∧ ∀ n : ℕ, a (n + 1) = r * a n

theorem geometric_sequence_constant
  (a : ℕ → ℝ)
  (h_geo : GeometricSequence a)
  (h_pos : ∀ n : ℕ, a n > 0)
  (h_eq : (a 1 + a 3) * (a 5 + a 7) = 4 * (a 4)^2) :
  ∃ c : ℝ, ∀ n : ℕ, a n = c :=
sorry

end NUMINAMATH_CALUDE_geometric_sequence_constant_l3637_363738


namespace NUMINAMATH_CALUDE_unique_solution_l3637_363784

/-- A discrete random variable with three possible values -/
structure DiscreteRV where
  p₁ : ℝ
  p₂ : ℝ
  p₃ : ℝ
  sum_to_one : p₁ + p₂ + p₃ = 1
  nonnegative : 0 ≤ p₁ ∧ 0 ≤ p₂ ∧ 0 ≤ p₃

/-- The expected value of X -/
def expectation (X : DiscreteRV) : ℝ := -X.p₁ + X.p₃

/-- The expected value of X² -/
def expectation_squared (X : DiscreteRV) : ℝ := X.p₁ + X.p₃

/-- Theorem stating the unique solution for the given conditions -/
theorem unique_solution (X : DiscreteRV) 
  (h₁ : expectation X = 0.1) 
  (h₂ : expectation_squared X = 0.9) : 
  X.p₁ = 0.4 ∧ X.p₂ = 0.1 ∧ X.p₃ = 0.5 := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_l3637_363784


namespace NUMINAMATH_CALUDE_kenneth_rowing_speed_l3637_363744

/-- Calculates the rowing speed of Kenneth given the race conditions -/
theorem kenneth_rowing_speed 
  (race_distance : ℝ) 
  (biff_speed : ℝ) 
  (kenneth_extra_distance : ℝ) 
  (h1 : race_distance = 500) 
  (h2 : biff_speed = 50) 
  (h3 : kenneth_extra_distance = 10) : 
  (race_distance + kenneth_extra_distance) / (race_distance / biff_speed) = 51 := by
  sorry

end NUMINAMATH_CALUDE_kenneth_rowing_speed_l3637_363744


namespace NUMINAMATH_CALUDE_domain_of_f_l3637_363721

noncomputable def f (x : ℝ) : ℝ := Real.log (Real.tan x - 1) + Real.sqrt (9 - x^2)

theorem domain_of_f :
  {x : ℝ | ∃ y, f x = y} = {x | -3*π/4 < x ∧ x < -π/2} ∪ {x | π/4 < x ∧ x < π/2} :=
by sorry

end NUMINAMATH_CALUDE_domain_of_f_l3637_363721


namespace NUMINAMATH_CALUDE_binary_10011_equals_19_l3637_363794

def binary_to_decimal (b : List Bool) : ℕ :=
  b.enum.foldr (fun (i, bit) acc => acc + if bit then 2^i else 0) 0

theorem binary_10011_equals_19 : 
  binary_to_decimal [true, false, false, true, true] = 19 := by
  sorry

end NUMINAMATH_CALUDE_binary_10011_equals_19_l3637_363794


namespace NUMINAMATH_CALUDE_number_of_planes_l3637_363701

/-- The number of wings on a commercial plane -/
def wings_per_plane : ℕ := 2

/-- The total number of wings counted -/
def total_wings : ℕ := 50

/-- Theorem: The number of commercial planes is 25 -/
theorem number_of_planes : 
  (total_wings / wings_per_plane : ℕ) = 25 := by sorry

end NUMINAMATH_CALUDE_number_of_planes_l3637_363701


namespace NUMINAMATH_CALUDE_ellipse_area_l3637_363773

/-- The area of an ellipse defined by the equation 9x^2 + 16y^2 = 144 is 12π. -/
theorem ellipse_area (x y : ℝ) : 
  (9 * x^2 + 16 * y^2 = 144) → (π * 4 * 3 : ℝ) = 12 * π := by
  sorry

end NUMINAMATH_CALUDE_ellipse_area_l3637_363773


namespace NUMINAMATH_CALUDE_train_problem_l3637_363763

/-- The length of the longer train given the conditions of the problem -/
def longer_train_length : ℝ := 319.96

theorem train_problem (train1_length train1_speed train2_speed clearing_time : ℝ) 
  (h1 : train1_length = 160)
  (h2 : train1_speed = 42)
  (h3 : train2_speed = 30)
  (h4 : clearing_time = 23.998) : 
  longer_train_length = 319.96 := by
  sorry

#check train_problem

end NUMINAMATH_CALUDE_train_problem_l3637_363763


namespace NUMINAMATH_CALUDE_zero_product_probability_l3637_363734

def S : Finset ℤ := {-3, -2, -1, 0, 0, 2, 4, 5}

def different_pairs (s : Finset ℤ) : Finset (ℤ × ℤ) :=
  (s.product s).filter (λ (a, b) => a ≠ b)

def zero_product_pairs (s : Finset ℤ) : Finset (ℤ × ℤ) :=
  (different_pairs s).filter (λ (a, b) => a * b = 0)

theorem zero_product_probability :
  (zero_product_pairs S).card / (different_pairs S).card = 3 / 14 := by
  sorry

end NUMINAMATH_CALUDE_zero_product_probability_l3637_363734


namespace NUMINAMATH_CALUDE_f_72_value_l3637_363795

/-- A function satisfying f(ab) = f(a) + f(b) for all a and b -/
def MultiplicativeToAdditive (f : ℕ → ℝ) : Prop :=
  ∀ a b : ℕ, f (a * b) = f a + f b

/-- The main theorem -/
theorem f_72_value (f : ℕ → ℝ) (p q : ℝ) 
    (h1 : MultiplicativeToAdditive f) 
    (h2 : f 2 = p) 
    (h3 : f 3 = q) : 
  f 72 = 3 * p + 2 * q := by
  sorry

end NUMINAMATH_CALUDE_f_72_value_l3637_363795


namespace NUMINAMATH_CALUDE_max_correct_answers_jesse_l3637_363783

/-- Represents a math contest with given parameters -/
structure MathContest where
  total_questions : ℕ
  correct_points : ℤ
  unanswered_points : ℤ
  incorrect_points : ℤ

/-- Represents a contestant's performance in the math contest -/
structure ContestPerformance where
  contest : MathContest
  total_score : ℤ

/-- Calculates the maximum number of correctly answered questions for a given contest performance -/
def max_correct_answers (performance : ContestPerformance) : ℕ :=
  sorry

/-- The specific contest Jesse participated in -/
def jesses_contest : MathContest := {
  total_questions := 60,
  correct_points := 4,
  unanswered_points := 0,
  incorrect_points := -1
}

/-- Jesse's performance in the contest -/
def jesses_performance : ContestPerformance := {
  contest := jesses_contest,
  total_score := 112
}

theorem max_correct_answers_jesse :
  max_correct_answers jesses_performance = 34 := by
  sorry

end NUMINAMATH_CALUDE_max_correct_answers_jesse_l3637_363783


namespace NUMINAMATH_CALUDE_sibling_age_difference_l3637_363707

/-- Given three siblings whose ages are in the ratio 3:2:1 and whose total combined age is 90 years,
    the difference between the eldest sibling's age and the youngest sibling's age is 30 years. -/
theorem sibling_age_difference (x : ℝ) (h1 : 3*x + 2*x + x = 90) : 3*x - x = 30 := by
  sorry

end NUMINAMATH_CALUDE_sibling_age_difference_l3637_363707


namespace NUMINAMATH_CALUDE_abc_inequality_l3637_363743

theorem abc_inequality (a b c : ℝ) 
  (distinct : a ≠ b ∧ b ≠ c ∧ a ≠ c) 
  (sum_eq : a + b + c = 6) 
  (prod_sum_eq : a * b + b * c + c * a = 9) : 
  0 < a * b * c ∧ a * b * c < 4 := by
sorry

end NUMINAMATH_CALUDE_abc_inequality_l3637_363743


namespace NUMINAMATH_CALUDE_student_ticket_price_l3637_363713

theorem student_ticket_price 
  (total_sales : ℝ)
  (student_ticket_surplus : ℕ)
  (nonstudent_tickets : ℕ)
  (nonstudent_price : ℝ)
  (h1 : total_sales = 10500)
  (h2 : student_ticket_surplus = 250)
  (h3 : nonstudent_tickets = 850)
  (h4 : nonstudent_price = 9) :
  ∃ (student_price : ℝ), 
    student_price = 2.59 ∧ 
    (nonstudent_tickets : ℝ) * nonstudent_price + 
    ((nonstudent_tickets : ℝ) + (student_ticket_surplus : ℝ)) * student_price = total_sales :=
by sorry

end NUMINAMATH_CALUDE_student_ticket_price_l3637_363713


namespace NUMINAMATH_CALUDE_bouncy_balls_per_package_l3637_363705

theorem bouncy_balls_per_package (red_packs green_packs yellow_packs total_balls : ℕ) 
  (h1 : red_packs = 4)
  (h2 : yellow_packs = 8)
  (h3 : green_packs = 4)
  (h4 : total_balls = 160) :
  ∃ (balls_per_pack : ℕ), 
    balls_per_pack * (red_packs + yellow_packs + green_packs) = total_balls ∧ 
    balls_per_pack = 10 := by
  sorry

end NUMINAMATH_CALUDE_bouncy_balls_per_package_l3637_363705


namespace NUMINAMATH_CALUDE_max_percentage_both_amenities_l3637_363727

/-- Represents the percentage of companies with type A planes -/
def percentage_type_A : ℝ := 0.4

/-- Represents the percentage of companies with type B planes -/
def percentage_type_B : ℝ := 0.6

/-- Represents the percentage of type A planes with wireless internet -/
def wireless_A : ℝ := 0.8

/-- Represents the percentage of type B planes with wireless internet -/
def wireless_B : ℝ := 0.1

/-- Represents the percentage of type A planes offering free snacks -/
def snacks_A : ℝ := 0.9

/-- Represents the percentage of type B planes offering free snacks -/
def snacks_B : ℝ := 0.5

/-- Theorem stating the maximum percentage of companies offering both amenities -/
theorem max_percentage_both_amenities :
  let max_both_A := min wireless_A snacks_A
  let max_both_B := min wireless_B snacks_B
  let max_percentage := percentage_type_A * max_both_A + percentage_type_B * max_both_B
  max_percentage = 0.38 := by sorry

end NUMINAMATH_CALUDE_max_percentage_both_amenities_l3637_363727


namespace NUMINAMATH_CALUDE_sum_of_solutions_quadratic_l3637_363796

theorem sum_of_solutions_quadratic (x : ℝ) : 
  let equation := -16 * x^2 + 72 * x - 108
  let sum_of_roots := -72 / (-16)
  equation = 0 → sum_of_roots = 9/2 := by
sorry

end NUMINAMATH_CALUDE_sum_of_solutions_quadratic_l3637_363796


namespace NUMINAMATH_CALUDE_simplify_A_plus_2B_value_A_plus_2B_at_1_neg1_l3637_363732

-- Define polynomials A and B
def A (a b : ℝ) : ℝ := 3*a^2 - 6*a*b + b^2
def B (a b : ℝ) : ℝ := -2*a^2 + 3*a*b - 5*b^2

-- Theorem for the simplified form of A + 2B
theorem simplify_A_plus_2B (a b : ℝ) : A a b + 2 * B a b = -a^2 - 9*b^2 := by sorry

-- Theorem for the value of A + 2B when a = 1 and b = -1
theorem value_A_plus_2B_at_1_neg1 : A 1 (-1) + 2 * B 1 (-1) = -10 := by sorry

end NUMINAMATH_CALUDE_simplify_A_plus_2B_value_A_plus_2B_at_1_neg1_l3637_363732


namespace NUMINAMATH_CALUDE_largest_angle_in_special_triangle_l3637_363747

theorem largest_angle_in_special_triangle : ∀ (a b c : ℝ),
  -- Two angles sum to 4/3 of a right angle
  a + b = (4/3) * 90 →
  -- One angle is 20° larger than the other
  b = a + 20 →
  -- All angles are non-negative
  a ≥ 0 ∧ b ≥ 0 ∧ c ≥ 0 →
  -- Sum of all angles in a triangle is 180°
  a + b + c = 180 →
  -- The largest angle is 70°
  max a (max b c) = 70 := by
sorry


end NUMINAMATH_CALUDE_largest_angle_in_special_triangle_l3637_363747


namespace NUMINAMATH_CALUDE_sixteen_team_tournament_games_l3637_363724

/-- Calculates the number of games in a single-elimination tournament. -/
def num_games_in_tournament (num_teams : ℕ) : ℕ :=
  num_teams - 1

/-- Theorem: In a single-elimination tournament with 16 teams, 15 games are played to determine the winner. -/
theorem sixteen_team_tournament_games :
  num_games_in_tournament 16 = 15 := by
  sorry

#eval num_games_in_tournament 16  -- Should output 15

end NUMINAMATH_CALUDE_sixteen_team_tournament_games_l3637_363724


namespace NUMINAMATH_CALUDE_article_price_l3637_363750

theorem article_price (P : ℝ) : 
  P * (1 - 0.1) * (1 - 0.2) = 72 → P = 100 := by
  sorry

end NUMINAMATH_CALUDE_article_price_l3637_363750


namespace NUMINAMATH_CALUDE_expression_simplification_l3637_363726

theorem expression_simplification (b : ℝ) :
  ((3 * b - 3) - 5 * b) / 3 = -2/3 * b - 1 := by sorry

end NUMINAMATH_CALUDE_expression_simplification_l3637_363726


namespace NUMINAMATH_CALUDE_cattle_train_departure_time_l3637_363746

/-- Proves that the cattle train left 6 hours before the diesel train --/
theorem cattle_train_departure_time (cattle_speed diesel_speed : ℝ) 
  (time_difference total_time : ℝ) (total_distance : ℝ) : 
  cattle_speed = 56 →
  diesel_speed = cattle_speed - 33 →
  total_time = 12 →
  total_distance = 1284 →
  total_distance = diesel_speed * total_time + cattle_speed * total_time + cattle_speed * time_difference →
  time_difference = 6 := by
  sorry

end NUMINAMATH_CALUDE_cattle_train_departure_time_l3637_363746


namespace NUMINAMATH_CALUDE_simplify_fraction_l3637_363730

theorem simplify_fraction : 
  ((2^1010)^2 - (2^1008)^2) / ((2^1009)^2 - (2^1007)^2) = 4 := by
  sorry

end NUMINAMATH_CALUDE_simplify_fraction_l3637_363730


namespace NUMINAMATH_CALUDE_area_of_region_S_l3637_363723

/-- A rhombus with side length 3 and angle B = 150° --/
structure Rhombus :=
  (side_length : ℝ)
  (angle_B : ℝ)
  (h_side : side_length = 3)
  (h_angle : angle_B = 150)

/-- The region S inside the rhombus closer to vertex B than to A, C, or D --/
def region_S (r : Rhombus) : Set (ℝ × ℝ) := sorry

/-- The area of a set in ℝ² --/
noncomputable def area (s : Set (ℝ × ℝ)) : ℝ := sorry

/-- The theorem stating that the area of region S is approximately 1.1 --/
theorem area_of_region_S (r : Rhombus) : 
  ∃ (ε : ℝ), ε > 0 ∧ ε < 0.01 ∧ |area (region_S r) - 1.1| < ε :=
sorry

end NUMINAMATH_CALUDE_area_of_region_S_l3637_363723


namespace NUMINAMATH_CALUDE_cheese_purchase_l3637_363789

theorem cheese_purchase (initial_amount : ℕ) (cheese_cost beef_cost : ℕ) (remaining_amount : ℕ) 
  (h1 : initial_amount = 87)
  (h2 : cheese_cost = 7)
  (h3 : beef_cost = 5)
  (h4 : remaining_amount = 61) :
  (initial_amount - remaining_amount - beef_cost) / cheese_cost = 3 :=
by sorry

end NUMINAMATH_CALUDE_cheese_purchase_l3637_363789


namespace NUMINAMATH_CALUDE_slope_one_points_l3637_363729

theorem slope_one_points (a : ℝ) : 
  let A : ℝ × ℝ := (-a, 3)
  let B : ℝ × ℝ := (5, -a)
  let slope := (B.2 - A.2) / (B.1 - A.1)
  slope = 1 → a = -4 := by
sorry

end NUMINAMATH_CALUDE_slope_one_points_l3637_363729


namespace NUMINAMATH_CALUDE_scientific_notation_of_340000_l3637_363706

theorem scientific_notation_of_340000 :
  340000 = 3.4 * (10 : ℝ) ^ 5 := by sorry

end NUMINAMATH_CALUDE_scientific_notation_of_340000_l3637_363706


namespace NUMINAMATH_CALUDE_strictly_increasing_function_inequality_l3637_363797

theorem strictly_increasing_function_inequality (k : ℕ) (f : ℕ → ℕ)
  (h_increasing : ∀ m n, m < n → f m < f n)
  (h_composite : ∀ n, f (f n) = k * n) :
  ∀ n : ℕ, n ≠ 0 → (2 * k * n) / (k + 1) ≤ f n ∧ f n ≤ (k + 1) * n / 2 :=
by sorry

end NUMINAMATH_CALUDE_strictly_increasing_function_inequality_l3637_363797


namespace NUMINAMATH_CALUDE_geometric_sequence_product_l3637_363702

theorem geometric_sequence_product (a b c : ℝ) : 
  (8/3 < a) ∧ (a < b) ∧ (b < c) ∧ (c < 27/2) ∧ 
  (∃ q : ℝ, q ≠ 0 ∧ a = 8/3 * q ∧ b = 8/3 * q^2 ∧ c = 8/3 * q^3 ∧ 27/2 = 8/3 * q^4) →
  a * b * c = 216 := by
sorry


end NUMINAMATH_CALUDE_geometric_sequence_product_l3637_363702


namespace NUMINAMATH_CALUDE_tangent_line_theorem_l3637_363733

/-- The function f(x) -/
def f (a b x : ℝ) : ℝ := x^3 + 2*a*x^2 + b*x + a

/-- The function g(x) -/
def g (x : ℝ) : ℝ := x^2 - 3*x + 2

/-- The derivative of f(x) -/
def f_deriv (a b x : ℝ) : ℝ := 3*x^2 + 4*a*x + b

/-- The derivative of g(x) -/
def g_deriv (x : ℝ) : ℝ := 2*x - 3

theorem tangent_line_theorem (a b : ℝ) :
  f a b 2 = 0 ∧ g 2 = 0 ∧ f_deriv a b 2 = g_deriv 2 →
  a = -3 ∧ b = 1 ∧ ∀ x y, y = x - 2 ↔ f a b x = y ∧ g x = y :=
sorry

end NUMINAMATH_CALUDE_tangent_line_theorem_l3637_363733


namespace NUMINAMATH_CALUDE_pen_pencil_length_difference_l3637_363760

theorem pen_pencil_length_difference :
  ∀ (rubber pen pencil : ℝ),
  pen = rubber + 3 →
  pencil = 12 →
  rubber + pen + pencil = 29 →
  pencil - pen = 2 :=
by
  sorry

end NUMINAMATH_CALUDE_pen_pencil_length_difference_l3637_363760


namespace NUMINAMATH_CALUDE_new_person_weight_is_143_l3637_363782

/-- Calculates the weight of a new person given the following conditions:
  * There are 15 people initially
  * The average weight increases by 5 kg when the new person replaces one person
  * The replaced person weighs 68 kg
-/
def new_person_weight (initial_count : ℕ) (avg_increase : ℝ) (replaced_weight : ℝ) : ℝ :=
  initial_count * avg_increase + replaced_weight

/-- Proves that under the given conditions, the weight of the new person is 143 kg -/
theorem new_person_weight_is_143 :
  new_person_weight 15 5 68 = 143 := by
  sorry

#eval new_person_weight 15 5 68

end NUMINAMATH_CALUDE_new_person_weight_is_143_l3637_363782


namespace NUMINAMATH_CALUDE_tangent_circle_equation_l3637_363710

/-- A circle tangent to both coordinate axes with its center on the line 5x - 3y = 8 -/
structure TangentCircle where
  center : ℝ × ℝ
  radius : ℝ
  tangent_to_axes : center.1 = radius ∧ center.2 = radius
  center_on_line : 5 * center.1 - 3 * center.2 = 8

/-- The equation of the circle is either (x-4)² + (y-4)² = 16 or (x-1)² + (y+1)² = 1 -/
theorem tangent_circle_equation (c : TangentCircle) :
  (∀ x y : ℝ, (x - 4)^2 + (y - 4)^2 = 16) ∨
  (∀ x y : ℝ, (x - 1)^2 + (y + 1)^2 = 1) :=
sorry

end NUMINAMATH_CALUDE_tangent_circle_equation_l3637_363710


namespace NUMINAMATH_CALUDE_extreme_points_cubic_l3637_363712

/-- A function f(x) = x^3 + ax has exactly two extreme points on R if and only if a < 0 -/
theorem extreme_points_cubic (a : ℝ) :
  (∃! (p q : ℝ), p ≠ q ∧ 
    (∀ x : ℝ, (3 * x^2 + a = 0) ↔ (x = p ∨ x = q))) ↔ 
  a < 0 := by
  sorry

end NUMINAMATH_CALUDE_extreme_points_cubic_l3637_363712


namespace NUMINAMATH_CALUDE_compound_simple_interest_principal_l3637_363748

theorem compound_simple_interest_principal (P r : ℝ) : 
  P * (1 + r)^2 - P = 11730 → P * r * 2 = 10200 → P = 17000 := by
  sorry

end NUMINAMATH_CALUDE_compound_simple_interest_principal_l3637_363748


namespace NUMINAMATH_CALUDE_watson_class_composition_l3637_363772

/-- The number of kindergartners in Ms. Watson's class -/
def num_kindergartners : ℕ := 42 - (24 + 4)

theorem watson_class_composition :
  num_kindergartners = 14 :=
by sorry

end NUMINAMATH_CALUDE_watson_class_composition_l3637_363772


namespace NUMINAMATH_CALUDE_train_speed_conversion_l3637_363786

/-- Conversion factor from meters per second to kilometers per hour -/
def mps_to_kmph : ℝ := 3.6

/-- Train's speed in meters per second -/
def train_speed_mps : ℝ := 45.0036

/-- Train's speed in kilometers per hour -/
def train_speed_kmph : ℝ := train_speed_mps * mps_to_kmph

theorem train_speed_conversion :
  train_speed_kmph = 162.013 := by sorry

end NUMINAMATH_CALUDE_train_speed_conversion_l3637_363786


namespace NUMINAMATH_CALUDE_equation_solution_l3637_363757

theorem equation_solution : ∃ x : ℚ, 5 * (x - 8) + 6 = 3 * (3 - 3 * x) + 15 ∧ x = 29 / 7 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l3637_363757


namespace NUMINAMATH_CALUDE_right_triangle_from_special_case_l3637_363755

/-- 
Given a triangle with sides a, 2a, and c, where the angle between sides a and 2a is 60°,
prove that the angle opposite side 2a is 90°.
-/
theorem right_triangle_from_special_case (a : ℝ) (h : a > 0) :
  let c := a * Real.sqrt 3
  let cos_alpha := (a^2 + c^2 - (2*a)^2) / (2 * a * c)
  cos_alpha = 0 := by sorry

end NUMINAMATH_CALUDE_right_triangle_from_special_case_l3637_363755


namespace NUMINAMATH_CALUDE_profit_is_152_l3637_363711

/-- The profit made from selling jerseys -/
def profit_from_jerseys (profit_per_jersey : ℕ) (jerseys_sold : ℕ) : ℕ :=
  profit_per_jersey * jerseys_sold

/-- Theorem: The profit from selling jerseys is $152 -/
theorem profit_is_152 :
  profit_from_jerseys 76 2 = 152 := by
  sorry

end NUMINAMATH_CALUDE_profit_is_152_l3637_363711


namespace NUMINAMATH_CALUDE_negation_of_universal_proposition_l3637_363793

theorem negation_of_universal_proposition :
  (¬ ∀ x : ℝ, Real.exp x > x^2) ↔ (∃ x₀ : ℝ, Real.exp x₀ ≤ x₀^2) := by
  sorry

end NUMINAMATH_CALUDE_negation_of_universal_proposition_l3637_363793


namespace NUMINAMATH_CALUDE_average_weight_increase_l3637_363715

theorem average_weight_increase (initial_count : ℕ) (old_weight new_weight : ℝ) :
  initial_count = 8 →
  old_weight = 65 →
  new_weight = 97 →
  (new_weight - old_weight) / initial_count = 4 :=
by sorry

end NUMINAMATH_CALUDE_average_weight_increase_l3637_363715


namespace NUMINAMATH_CALUDE_triangle_tangent_circles_l3637_363751

/-- Given a triangle with side lengths a, b, and c, there exist radii r₁, r₂, and r₃ for circles
    centered at the triangle's vertices that satisfy both external and internal tangency conditions. -/
theorem triangle_tangent_circles
  (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c)
  (triangle_ineq : a < b + c ∧ b < a + c ∧ c < a + b) :
  ∃ (r₁ r₂ r₃ : ℝ),
    (r₁ > 0 ∧ r₂ > 0 ∧ r₃ > 0) ∧
    (r₁ + r₂ = c ∧ r₂ + r₃ = a ∧ r₃ + r₁ = b) ∧
    ∃ (r₁' r₂' r₃' : ℝ),
      (r₁' > 0 ∧ r₂' > 0 ∧ r₃' > 0) ∧
      (r₃' - r₂' = a ∧ r₃' - r₁' = b ∧ r₁' + r₂' = c) :=
by sorry

end NUMINAMATH_CALUDE_triangle_tangent_circles_l3637_363751


namespace NUMINAMATH_CALUDE_sequence_realignment_l3637_363736

def letter_cycle_length : ℕ := 6
def digit_cycle_length : ℕ := 4

theorem sequence_realignment :
  ∃ n : ℕ, n > 0 ∧ n % letter_cycle_length = 0 ∧ n % digit_cycle_length = 0 ∧
  ∀ m : ℕ, (m > 0 ∧ m % letter_cycle_length = 0 ∧ m % digit_cycle_length = 0) → m ≥ n :=
by
  sorry

end NUMINAMATH_CALUDE_sequence_realignment_l3637_363736


namespace NUMINAMATH_CALUDE_unique_solution_quadratic_l3637_363735

theorem unique_solution_quadratic (k : ℝ) :
  (∃! x : ℝ, k * x^2 + 4 * x + 4 = 0) ↔ (k = 0 ∨ k = 1) := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_quadratic_l3637_363735


namespace NUMINAMATH_CALUDE_encounters_for_2015_trips_relative_speeds_main_encounters_theorem_l3637_363771

/-- The number of encounters between two people traveling between two points -/
def encounters (a_trips b_trips : ℕ) : ℕ :=
  let full_cycles := a_trips / 2
  let remainder := a_trips % 2
  3 * full_cycles + if remainder = 0 then 0 else 2

/-- The theorem stating the number of encounters when A reaches point B 2015 times -/
theorem encounters_for_2015_trips : encounters 2015 2015 = 3023 := by
  sorry

/-- The theorem stating the relative speeds of A and B -/
theorem relative_speeds : ∀ (va vb : ℝ), 
  5 * va = 9 * vb → vb = (18/5) * va := by
  sorry

/-- The main theorem proving the number of encounters -/
theorem main_encounters_theorem : 
  ∃ (va vb : ℝ), va > 0 ∧ vb > 0 ∧ 5 * va = 9 * vb ∧ encounters 2015 2015 = 3023 := by
  sorry

end NUMINAMATH_CALUDE_encounters_for_2015_trips_relative_speeds_main_encounters_theorem_l3637_363771


namespace NUMINAMATH_CALUDE_expansion_equality_l3637_363742

theorem expansion_equality (x : ℝ) : 
  (x - 2)^5 + 5*(x - 2)^4 + 10*(x - 2)^3 + 10*(x - 2)^2 + 5*(x - 2) + 1 = (x - 1)^5 := by
sorry

end NUMINAMATH_CALUDE_expansion_equality_l3637_363742


namespace NUMINAMATH_CALUDE_card_cost_calculation_l3637_363790

theorem card_cost_calculation (christmas_cards : ℕ) (birthday_cards : ℕ) (total_spent : ℕ) : 
  christmas_cards = 20 →
  birthday_cards = 15 →
  total_spent = 70 →
  (total_spent : ℚ) / (christmas_cards + birthday_cards : ℚ) = 2 := by
  sorry

end NUMINAMATH_CALUDE_card_cost_calculation_l3637_363790


namespace NUMINAMATH_CALUDE_contractor_daily_wage_l3637_363708

/-- Represents the contractor's payment scenario --/
structure ContractorPayment where
  totalDays : ℕ
  absentDays : ℕ
  finePerDay : ℚ
  totalPayment : ℚ

/-- Calculates the daily wage given the contractor's payment scenario --/
def calculateDailyWage (c : ContractorPayment) : ℚ :=
  (c.totalPayment + c.finePerDay * c.absentDays) / (c.totalDays - c.absentDays)

/-- Theorem stating that the daily wage is 25 rupees given the problem conditions --/
theorem contractor_daily_wage :
  let c : ContractorPayment := {
    totalDays := 30,
    absentDays := 10,
    finePerDay := 15/2,
    totalPayment := 425
  }
  calculateDailyWage c = 25 := by sorry

end NUMINAMATH_CALUDE_contractor_daily_wage_l3637_363708


namespace NUMINAMATH_CALUDE_complex_division_equality_l3637_363719

theorem complex_division_equality : (3 + Complex.I) / (1 + Complex.I) = 2 - Complex.I := by
  sorry

end NUMINAMATH_CALUDE_complex_division_equality_l3637_363719


namespace NUMINAMATH_CALUDE_vacation_cost_division_l3637_363745

theorem vacation_cost_division (total_cost : ℕ) (cost_difference : ℕ) : 
  (total_cost = 360) →
  (total_cost / 4 + cost_difference = total_cost / 3) →
  (cost_difference = 30) →
  3 = total_cost / (total_cost / 3) :=
by
  sorry

end NUMINAMATH_CALUDE_vacation_cost_division_l3637_363745


namespace NUMINAMATH_CALUDE_incorrect_height_calculation_l3637_363799

theorem incorrect_height_calculation (n : ℕ) (initial_avg real_avg actual_height : ℝ) 
  (h1 : n = 20)
  (h2 : initial_avg = 175)
  (h3 : real_avg = 174.25)
  (h4 : actual_height = 136) :
  ∃ (incorrect_height : ℝ),
    incorrect_height = n * initial_avg - (n * real_avg - actual_height) ∧
    incorrect_height = 151 := by
  sorry

end NUMINAMATH_CALUDE_incorrect_height_calculation_l3637_363799


namespace NUMINAMATH_CALUDE_trees_survival_difference_l3637_363768

theorem trees_survival_difference (initial_trees dead_trees : ℕ) 
  (h1 : initial_trees = 13)
  (h2 : dead_trees = 6) :
  initial_trees - dead_trees - dead_trees = 1 :=
by sorry

end NUMINAMATH_CALUDE_trees_survival_difference_l3637_363768


namespace NUMINAMATH_CALUDE_polynomial_factor_coefficients_l3637_363741

theorem polynomial_factor_coefficients :
  ∀ (a b : ℤ),
  (∃ (c d : ℤ),
    (∀ x : ℝ, a * x^4 + b * x^3 + 40 * x^2 - 20 * x + 8 = (3 * x^2 - 2 * x + 2) * (c * x^2 + d * x + 4))) →
  a = -51 ∧ b = 25 := by
sorry

end NUMINAMATH_CALUDE_polynomial_factor_coefficients_l3637_363741


namespace NUMINAMATH_CALUDE_sequence_term_l3637_363716

/-- The sum of the first n terms of a sequence -/
def S (n : ℕ) : ℤ := n^2 - 3*n

/-- The nth term of the sequence -/
def a (n : ℕ) : ℤ := 2*n - 4

theorem sequence_term (n : ℕ) (h : n ≥ 1) : 
  a n = S n - S (n-1) :=
sorry

end NUMINAMATH_CALUDE_sequence_term_l3637_363716


namespace NUMINAMATH_CALUDE_cone_radius_l3637_363725

/-- Given a cone with angle π/3 between generatrix and base, and volume 3π, its base radius is √3 -/
theorem cone_radius (angle : Real) (volume : Real) (radius : Real) : 
  angle = π / 3 → volume = 3 * π → 
  (1 / 3) * π * radius^2 * (radius * Real.sqrt 3) = volume → 
  radius = Real.sqrt 3 := by
sorry

end NUMINAMATH_CALUDE_cone_radius_l3637_363725


namespace NUMINAMATH_CALUDE_k_minus_one_not_square_k_plus_one_not_square_l3637_363781

/-- k is the product of several of the first prime numbers -/
def k : ℕ := sorry

/-- k is the product of at least two prime numbers -/
axiom k_def : ∃ (p q : ℕ), Prime p ∧ Prime q ∧ p < q ∧ k = p * q

/-- k-1 is not a perfect square -/
theorem k_minus_one_not_square : ¬∃ (n : ℕ), n^2 = k - 1 := by sorry

/-- k+1 is not a perfect square -/
theorem k_plus_one_not_square : ¬∃ (n : ℕ), n^2 = k + 1 := by sorry

end NUMINAMATH_CALUDE_k_minus_one_not_square_k_plus_one_not_square_l3637_363781


namespace NUMINAMATH_CALUDE_five_digit_divisible_by_72_l3637_363718

theorem five_digit_divisible_by_72 (a b : Nat) : 
  (a < 10 ∧ b < 10) →
  (a * 10000 + 6 * 1000 + 7 * 100 + 9 * 10 + b) % 72 = 0 ↔ 
  (a = 3 ∧ b = 2) := by
  sorry

end NUMINAMATH_CALUDE_five_digit_divisible_by_72_l3637_363718


namespace NUMINAMATH_CALUDE_g_of_2_l3637_363700

/-- Given a function g(x) = px^8 + qx^4 + rx + 7 where g(-2) = -5,
    prove that g(2) = 2p(256) + 2q(16) + 19 -/
theorem g_of_2 (p q r : ℝ) (g : ℝ → ℝ) 
    (h1 : ∀ x, g x = p * x^8 + q * x^4 + r * x + 7)
    (h2 : g (-2) = -5) :
  g 2 = 2 * p * 256 + 2 * q * 16 + 19 := by
  sorry

end NUMINAMATH_CALUDE_g_of_2_l3637_363700


namespace NUMINAMATH_CALUDE_cost_price_correct_l3637_363740

/-- The cost price of an eye-protection lamp -/
def cost_price : ℝ := 150

/-- The original selling price of the lamp -/
def original_price : ℝ := 200

/-- The discount rate during the special period -/
def discount_rate : ℝ := 0.1

/-- The profit rate after the discount -/
def profit_rate : ℝ := 0.2

/-- Theorem stating that the cost price is correct given the conditions -/
theorem cost_price_correct : 
  original_price * (1 - discount_rate) = cost_price * (1 + profit_rate) :=
sorry

end NUMINAMATH_CALUDE_cost_price_correct_l3637_363740


namespace NUMINAMATH_CALUDE_system_of_equations_solutions_l3637_363775

theorem system_of_equations_solutions :
  -- First system of equations
  (∃ x y : ℝ, 2*x - y = 3 ∧ x + y = 3 ∧ x = 2 ∧ y = 1) ∧
  -- Second system of equations
  (∃ x y : ℝ, x/4 + y/3 = 3 ∧ 3*x - 2*(y-1) = 11 ∧ x = 6 ∧ y = 9/2) :=
by sorry

end NUMINAMATH_CALUDE_system_of_equations_solutions_l3637_363775


namespace NUMINAMATH_CALUDE_totient_product_inequality_l3637_363765

theorem totient_product_inequality (m n : ℕ) (h : m ≠ n) : 
  n * (Nat.totient n) ≠ m * (Nat.totient m) := by
  sorry

end NUMINAMATH_CALUDE_totient_product_inequality_l3637_363765
