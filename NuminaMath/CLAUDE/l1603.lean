import Mathlib

namespace NUMINAMATH_CALUDE_sin_thirteen_pi_fourths_l1603_160364

theorem sin_thirteen_pi_fourths : Real.sin (13 * π / 4) = -Real.sqrt 2 / 2 := by
  sorry

end NUMINAMATH_CALUDE_sin_thirteen_pi_fourths_l1603_160364


namespace NUMINAMATH_CALUDE_extra_invitations_needed_carol_extra_invitations_l1603_160304

theorem extra_invitations_needed 
  (packs_bought : ℕ) 
  (invitations_per_pack : ℕ) 
  (friends_to_invite : ℕ) : ℕ :=
  friends_to_invite - (packs_bought * invitations_per_pack)

theorem carol_extra_invitations : 
  extra_invitations_needed 2 3 9 = 3 := by
  sorry

end NUMINAMATH_CALUDE_extra_invitations_needed_carol_extra_invitations_l1603_160304


namespace NUMINAMATH_CALUDE_min_value_of_sum_l1603_160353

theorem min_value_of_sum (x y : ℝ) (hx : x > 0) (hy : y > 0) 
  (h : 1 / (x + 3) + 1 / (y + 3) = 1 / 4) : 
  x + 3 * y ≥ 4 + 8 * Real.sqrt 3 ∧ 
  ∃ (x₀ y₀ : ℝ), x₀ > 0 ∧ y₀ > 0 ∧ 
    1 / (x₀ + 3) + 1 / (y₀ + 3) = 1 / 4 ∧ 
    x₀ + 3 * y₀ = 4 + 8 * Real.sqrt 3 :=
sorry

end NUMINAMATH_CALUDE_min_value_of_sum_l1603_160353


namespace NUMINAMATH_CALUDE_nobel_laureates_count_l1603_160337

/-- Represents the number of scientists at a workshop with various prize combinations -/
structure WorkshopScientists where
  total : Nat
  wolf : Nat
  wolfAndNobel : Nat
  nonWolfNobel : Nat
  nonWolfNonNobel : Nat

/-- The conditions of the workshop -/
def workshop : WorkshopScientists where
  total := 50
  wolf := 31
  wolfAndNobel := 12
  nonWolfNobel := (50 - 31 + 3) / 2
  nonWolfNonNobel := (50 - 31 - 3) / 2

/-- Theorem stating the total number of Nobel prize laureates -/
theorem nobel_laureates_count (w : WorkshopScientists) (h1 : w = workshop) :
  w.wolfAndNobel + w.nonWolfNobel = 23 := by
  sorry

#check nobel_laureates_count

end NUMINAMATH_CALUDE_nobel_laureates_count_l1603_160337


namespace NUMINAMATH_CALUDE_cubic_fraction_equals_fifteen_l1603_160331

theorem cubic_fraction_equals_fifteen :
  let a : ℤ := 8
  let b : ℤ := a - 1
  (a^3 + b^3) / (a^2 - a*b + b^2) = 15 := by
  sorry

end NUMINAMATH_CALUDE_cubic_fraction_equals_fifteen_l1603_160331


namespace NUMINAMATH_CALUDE_quadratic_distinct_roots_l1603_160362

theorem quadratic_distinct_roots (m : ℝ) :
  (∃ x y : ℝ, x ≠ y ∧ x^2 - 4*x + m = 0 ∧ y^2 - 4*y + m = 0) → m < 4 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_distinct_roots_l1603_160362


namespace NUMINAMATH_CALUDE_tangent_slope_angle_l1603_160396

noncomputable def f (x : ℝ) : ℝ := (1/3) * x^3 - x^2 + 5

theorem tangent_slope_angle (x : ℝ) : 
  x = 1 → 
  ∃ θ : ℝ, θ = 3 * Real.pi / 4 ∧ 
    θ = Real.pi + Real.arctan ((deriv f) x) :=
by sorry

end NUMINAMATH_CALUDE_tangent_slope_angle_l1603_160396


namespace NUMINAMATH_CALUDE_decimal_sum_as_fraction_l1603_160310

/-- The sum of 0.01, 0.002, 0.0003, 0.00004, and 0.000005 is equal to 2469/200000 -/
theorem decimal_sum_as_fraction : 
  (0.01 : ℚ) + 0.002 + 0.0003 + 0.00004 + 0.000005 = 2469 / 200000 := by
  sorry

end NUMINAMATH_CALUDE_decimal_sum_as_fraction_l1603_160310


namespace NUMINAMATH_CALUDE_d_value_l1603_160393

theorem d_value (d : ℚ) (h : 10 * d + 8 = 528) : 2 * d = 104 := by
  sorry

end NUMINAMATH_CALUDE_d_value_l1603_160393


namespace NUMINAMATH_CALUDE_circle_area_ratio_l1603_160325

-- Define the circle structure
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

-- Define the point structure
structure Point where
  x : ℝ
  y : ℝ

-- Define the midpoint property
def isMidpoint (a b m : Point) : Prop :=
  m.x = (a.x + b.x) / 2 ∧ m.y = (a.y + b.y) / 2

-- Define the theorem
theorem circle_area_ratio
  (c1 c2 : Circle)
  (o p x : Point)
  (h1 : c1.center = c2.center)
  (h2 : c1.center = (o.x, o.y))
  (h3 : c1.radius = (p.x - o.x))
  (h4 : c2.radius = (x.x - o.x))
  (h5 : isMidpoint o p x) :
  (π * c2.radius^2) / (π * c1.radius^2) = 1/4 :=
by
  sorry


end NUMINAMATH_CALUDE_circle_area_ratio_l1603_160325


namespace NUMINAMATH_CALUDE_factors_of_6000_l1603_160314

/-- The number of positive integer factors of a number -/
def num_factors (n : ℕ) : ℕ := sorry

/-- The number of positive integer factors of a number that are perfect squares -/
def num_square_factors (n : ℕ) : ℕ := sorry

theorem factors_of_6000 :
  let n : ℕ := 6000
  let factorization : List (ℕ × ℕ) := [(2, 4), (3, 1), (5, 3)]
  (num_factors n = 40) ∧
  (num_factors n - num_square_factors n = 34) := by sorry

end NUMINAMATH_CALUDE_factors_of_6000_l1603_160314


namespace NUMINAMATH_CALUDE_distance_difference_l1603_160382

/-- The distance Aleena biked in 5 hours -/
def aleena_distance : ℕ := 75

/-- The distance Bob biked in 5 hours -/
def bob_distance : ℕ := 60

/-- Theorem stating the difference between Aleena's and Bob's distances after 5 hours -/
theorem distance_difference : aleena_distance - bob_distance = 15 := by
  sorry

end NUMINAMATH_CALUDE_distance_difference_l1603_160382


namespace NUMINAMATH_CALUDE_min_marking_for_range_l1603_160383

def covers (marked : Finset ℕ) (n : ℕ) : Prop :=
  n ∈ marked ∨ (∃ m ∈ marked, n ∣ m ∨ m ∣ n)

def covers_range (marked : Finset ℕ) (start finish : ℕ) : Prop :=
  ∀ n, start ≤ n → n ≤ finish → covers marked n

theorem min_marking_for_range :
  ∃ (marked : Finset ℕ), covers_range marked 2 30 ∧ marked.card = 5 ∧
    ∀ (other : Finset ℕ), covers_range other 2 30 → other.card ≥ 5 :=
by sorry

end NUMINAMATH_CALUDE_min_marking_for_range_l1603_160383


namespace NUMINAMATH_CALUDE_no_sequence_exists_l1603_160377

theorem no_sequence_exists : ¬ ∃ (a : Fin 7 → ℝ), 
  (∀ i, 0 ≤ a i) ∧ 
  (a 0 = 0) ∧ 
  (a 6 = 0) ∧ 
  (∀ i ∈ Finset.range 5, a (i + 2) + a i > Real.sqrt 3 * a (i + 1)) := by
sorry

end NUMINAMATH_CALUDE_no_sequence_exists_l1603_160377


namespace NUMINAMATH_CALUDE_hyperbola_equation_proof_l1603_160343

/-- The hyperbola equation -/
def hyperbola_equation (a b x y : ℝ) : Prop :=
  x^2 / a^2 - y^2 / b^2 = 1

/-- Point on asymptote condition -/
def point_on_asymptote (a b : ℝ) : Prop :=
  4 / 3 = b / a

/-- Perpendicular foci condition -/
def perpendicular_foci (c : ℝ) : Prop :=
  4 / (3 + c) * (4 / (3 - c)) = -1

/-- Relationship between a, b, and c -/
def foci_distance (a b c : ℝ) : Prop :=
  c^2 = a^2 + b^2

/-- Main theorem -/
theorem hyperbola_equation_proof (a b : ℝ) (ha : a > 0) (hb : b > 0) 
  (h_asymptote : point_on_asymptote a b)
  (h_foci : ∃ c, perpendicular_foci c ∧ foci_distance a b c) :
  hyperbola_equation 3 4 x y :=
sorry

end NUMINAMATH_CALUDE_hyperbola_equation_proof_l1603_160343


namespace NUMINAMATH_CALUDE_approx_root_e_2019_l1603_160388

/-- Approximation of the 2019th root of e using tangent line method -/
theorem approx_root_e_2019 (e : ℝ) (h : e = Real.exp 1) :
  ∃ (ε : ℝ), ε > 0 ∧ ε < 0.00001 ∧ |e^(1/2019) - (1 + 1/2019)| < ε :=
sorry

end NUMINAMATH_CALUDE_approx_root_e_2019_l1603_160388


namespace NUMINAMATH_CALUDE_smallest_solution_of_quadratic_l1603_160333

theorem smallest_solution_of_quadratic (y : ℝ) : 
  (3 * y^2 + 15 * y - 90 = y * (y + 20)) → y ≥ -6 :=
by
  sorry

end NUMINAMATH_CALUDE_smallest_solution_of_quadratic_l1603_160333


namespace NUMINAMATH_CALUDE_fraction_decomposition_l1603_160390

theorem fraction_decomposition :
  ∀ (x : ℝ) (C D : ℚ),
    (C / (x - 2) + D / (3 * x + 7) = (3 * x^2 + 7 * x - 20) / (3 * x^2 - x - 14)) →
    (C = -14/13 ∧ D = 81/13) := by
  sorry

end NUMINAMATH_CALUDE_fraction_decomposition_l1603_160390


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_set_l1603_160301

theorem quadratic_inequality_solution_set :
  {x : ℝ | (x + 2) * (x - 3) < 0} = {x : ℝ | -2 < x ∧ x < 3} := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_set_l1603_160301


namespace NUMINAMATH_CALUDE_parabola_zeros_difference_l1603_160350

/-- Represents a quadratic function ax^2 + bx + c -/
structure QuadraticFunction where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Calculates the y-value for a given x in a quadratic function -/
def QuadraticFunction.eval (f : QuadraticFunction) (x : ℝ) : ℝ :=
  f.a * x^2 + f.b * x + f.c

/-- Finds the zeros of a quadratic function -/
def QuadraticFunction.zeros (f : QuadraticFunction) : Set ℝ :=
  {x : ℝ | f.eval x = 0}

theorem parabola_zeros_difference (f : QuadraticFunction) :
  f.eval 3 = -9 →  -- vertex at (3, -9)
  f.eval 5 = 7 →   -- passes through (5, 7)
  ∃ m n, m ∈ f.zeros ∧ n ∈ f.zeros ∧ m > n ∧ m - n = 3 := by
  sorry

end NUMINAMATH_CALUDE_parabola_zeros_difference_l1603_160350


namespace NUMINAMATH_CALUDE_centrally_symmetric_implies_congruent_l1603_160376

-- Define a shape
def Shape : Type := sorry

-- Define central symmetry
def centrally_symmetric (s1 s2 : Shape) : Prop := 
  ∃ p : ℝ × ℝ, ∃ rotation : Shape → Shape, 
    rotation s1 = s2 ∧ 
    (∀ x : Shape, rotation (rotation x) = x)

-- Define congruence
def congruent (s1 s2 : Shape) : Prop := sorry

-- Theorem statement
theorem centrally_symmetric_implies_congruent (s1 s2 : Shape) :
  centrally_symmetric s1 s2 → congruent s1 s2 := by sorry

end NUMINAMATH_CALUDE_centrally_symmetric_implies_congruent_l1603_160376


namespace NUMINAMATH_CALUDE_tangent_trapezoid_ratio_l1603_160316

/-- Represents a trapezoid with a circle tangent to two sides -/
structure TangentTrapezoid where
  /-- Length of side EF -/
  ef : ℝ
  /-- Length of side FG -/
  fg : ℝ
  /-- Length of side GH -/
  gh : ℝ
  /-- Length of side HE -/
  he : ℝ
  /-- EF is parallel to GH -/
  parallel : ef ≠ gh
  /-- Circle with center Q on EF is tangent to FG and HE -/
  tangent : True

/-- The ratio of EQ to QF in the trapezoid -/
def ratio (t : TangentTrapezoid) : ℚ :=
  12 / 37

theorem tangent_trapezoid_ratio (t : TangentTrapezoid) 
  (h1 : t.ef = 40)
  (h2 : t.fg = 25)
  (h3 : t.gh = 12)
  (h4 : t.he = 35) :
  ratio t = 12 / 37 := by
  sorry

end NUMINAMATH_CALUDE_tangent_trapezoid_ratio_l1603_160316


namespace NUMINAMATH_CALUDE_smallest_sum_of_reciprocals_l1603_160346

theorem smallest_sum_of_reciprocals (x y : ℕ+) : 
  x ≠ y → 
  (1 : ℚ) / x + (1 : ℚ) / y = (1 : ℚ) / 15 → 
  ∀ a b : ℕ+, a ≠ b → (1 : ℚ) / a + (1 : ℚ) / b = (1 : ℚ) / 15 → 
  (x : ℤ) + y ≤ (a : ℤ) + b → 
  (x : ℤ) + y = 64 :=
by sorry

end NUMINAMATH_CALUDE_smallest_sum_of_reciprocals_l1603_160346


namespace NUMINAMATH_CALUDE_min_sum_of_product_100_l1603_160341

theorem min_sum_of_product_100 (a b : ℤ) (h : a * b = 100) :
  ∀ (x y : ℤ), x * y = 100 → a + b ≤ x + y ∧ ∃ (a₀ b₀ : ℤ), a₀ * b₀ = 100 ∧ a₀ + b₀ = -101 :=
by sorry

end NUMINAMATH_CALUDE_min_sum_of_product_100_l1603_160341


namespace NUMINAMATH_CALUDE_bobs_mile_time_l1603_160367

/-- Bob's mile run time problem -/
theorem bobs_mile_time (sister_time : ℝ) (improvement_percent : ℝ) (bob_time : ℝ) : 
  sister_time = 9 * 60 + 42 →
  improvement_percent = 9.062499999999996 →
  bob_time = sister_time * (1 + improvement_percent / 100) →
  bob_time = 634.5 := by
  sorry

end NUMINAMATH_CALUDE_bobs_mile_time_l1603_160367


namespace NUMINAMATH_CALUDE_simplify_expression_l1603_160354

theorem simplify_expression (x : ℝ) : 2*x - 3*(2+x) + 4*(2-x) - 5*(2+3*x) = -20*x - 8 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l1603_160354


namespace NUMINAMATH_CALUDE_hamburger_cost_satisfies_conditions_l1603_160327

/-- The cost of a pack of hamburger meat that satisfies the given conditions -/
def hamburger_cost : ℝ :=
  let crackers : ℝ := 3.50
  let vegetables : ℝ := 4 * 2.00
  let cheese : ℝ := 3.50
  let discount_rate : ℝ := 0.10
  let total_after_discount : ℝ := 18.00
  5.00

/-- Theorem stating that the hamburger cost satisfies the given conditions -/
theorem hamburger_cost_satisfies_conditions :
  let crackers : ℝ := 3.50
  let vegetables : ℝ := 4 * 2.00
  let cheese : ℝ := 3.50
  let discount_rate : ℝ := 0.10
  let total_after_discount : ℝ := 18.00
  total_after_discount = (hamburger_cost + crackers + vegetables + cheese) * (1 - discount_rate) := by
  sorry

#eval hamburger_cost

end NUMINAMATH_CALUDE_hamburger_cost_satisfies_conditions_l1603_160327


namespace NUMINAMATH_CALUDE_system_one_solutions_system_two_solutions_l1603_160319

-- System 1
theorem system_one_solutions (x y : ℝ) :
  (x^2 - 2*x = 0 ∧ x^3 + y = 6) ↔ ((x = 0 ∧ y = 6) ∨ (x = 2 ∧ y = -2)) :=
sorry

-- System 2
theorem system_two_solutions (x y : ℝ) :
  (y^2 - 4*y + 3 = 0 ∧ 2*x + y = 9) ↔ ((x = 4 ∧ y = 1) ∨ (x = 3 ∧ y = 3)) :=
sorry

end NUMINAMATH_CALUDE_system_one_solutions_system_two_solutions_l1603_160319


namespace NUMINAMATH_CALUDE_seventh_term_is_84_l1603_160359

/-- A sequence where the differences between consecutive terms form a quadratic sequence -/
def CookieSequence (a : ℕ → ℕ) : Prop :=
  ∃ p q r : ℕ,
    (∀ n, a (n + 1) - a n = p * n * n + q * n + r) ∧
    a 1 = 5 ∧ a 2 = 9 ∧ a 3 = 14 ∧ a 4 = 22 ∧ a 5 = 35

theorem seventh_term_is_84 (a : ℕ → ℕ) (h : CookieSequence a) : a 7 = 84 := by
  sorry

end NUMINAMATH_CALUDE_seventh_term_is_84_l1603_160359


namespace NUMINAMATH_CALUDE_functional_equation_solution_l1603_160389

theorem functional_equation_solution (f : ℝ → ℝ) :
  (∀ x y : ℝ, f (x * y + 1) = x * f y + 2) →
  (∀ x : ℝ, f x = 2 * x) :=
by sorry

end NUMINAMATH_CALUDE_functional_equation_solution_l1603_160389


namespace NUMINAMATH_CALUDE_geometric_sequence_sum_l1603_160345

def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, r > 0 ∧ ∀ n : ℕ, a (n + 1) = a n * r

theorem geometric_sequence_sum (a : ℕ → ℝ) :
  geometric_sequence a →
  (∀ n : ℕ, a n > 0) →
  a 6 * a 10 + a 3 * a 5 = 26 →
  a 5 * a 7 = 5 →
  a 4 + a 8 = 6 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_sum_l1603_160345


namespace NUMINAMATH_CALUDE_pentagon_perimeter_l1603_160336

theorem pentagon_perimeter (A B C D E : ℝ × ℝ) : 
  let AB := 2
  let BC := Real.sqrt 5
  let CD := Real.sqrt 3
  let DE := 1
  let AC := Real.sqrt ((AB ^ 2) + (BC ^ 2))
  let AD := Real.sqrt ((AC ^ 2) + (CD ^ 2))
  let AE := Real.sqrt ((AD ^ 2) + (DE ^ 2))
  AB + BC + CD + DE + AE = 3 + Real.sqrt 5 + Real.sqrt 3 + 1 + Real.sqrt 13 :=
by sorry

end NUMINAMATH_CALUDE_pentagon_perimeter_l1603_160336


namespace NUMINAMATH_CALUDE_average_hours_worked_l1603_160399

/-- Represents the number of hours worked on a given day type in a month -/
structure MonthlyHours where
  weekday : ℕ
  saturday : ℕ
  sunday : ℕ

/-- Represents the work schedule for a month -/
structure MonthSchedule where
  days : ℕ
  weekdays : ℕ
  saturdays : ℕ
  sundays : ℕ
  hours : MonthlyHours
  vacation_days : ℕ

def april : MonthSchedule :=
  { days := 30
    weekdays := 22
    saturdays := 4
    sundays := 4
    hours := { weekday := 6, saturday := 4, sunday := 0 }
    vacation_days := 5 }

def june : MonthSchedule :=
  { days := 30
    weekdays := 30
    saturdays := 0
    sundays := 0
    hours := { weekday := 5, saturday := 5, sunday := 5 }
    vacation_days := 4 }

def september : MonthSchedule :=
  { days := 30
    weekdays := 22
    saturdays := 4
    sundays := 4
    hours := { weekday := 8, saturday := 0, sunday := 0 }
    vacation_days := 0 }

def calculate_hours (m : MonthSchedule) : ℕ :=
  (m.weekdays - m.vacation_days) * m.hours.weekday +
  m.saturdays * m.hours.saturday +
  m.sundays * m.hours.sunday

theorem average_hours_worked :
  (calculate_hours april + calculate_hours june + calculate_hours september) / 3 = 141 :=
sorry

end NUMINAMATH_CALUDE_average_hours_worked_l1603_160399


namespace NUMINAMATH_CALUDE_max_y_coordinate_l1603_160348

theorem max_y_coordinate (x y : ℝ) : 
  (x^2 / 49) + ((y - 3)^2 / 25) = 0 → y ≤ 3 :=
by sorry

end NUMINAMATH_CALUDE_max_y_coordinate_l1603_160348


namespace NUMINAMATH_CALUDE_rebecca_earring_ratio_l1603_160379

/-- Proves the ratio of gemstones to buttons for Rebecca's earrings --/
theorem rebecca_earring_ratio 
  (magnets_per_earring : ℕ)
  (buttons_to_magnets_ratio : ℚ)
  (sets_of_earrings : ℕ)
  (total_gemstones : ℕ)
  (h1 : magnets_per_earring = 2)
  (h2 : buttons_to_magnets_ratio = 1/2)
  (h3 : sets_of_earrings = 4)
  (h4 : total_gemstones = 24) :
  (total_gemstones : ℚ) / ((sets_of_earrings * 2 * magnets_per_earring * buttons_to_magnets_ratio) : ℚ) = 3 := by
  sorry

#check rebecca_earring_ratio

end NUMINAMATH_CALUDE_rebecca_earring_ratio_l1603_160379


namespace NUMINAMATH_CALUDE_max_sum_l1603_160329

/-- An arithmetic sequence {an} with sum Sn -/
structure ArithmeticSequence where
  a : ℕ → ℝ
  d : ℝ
  sum : ℕ → ℝ
  is_arithmetic : ∀ n, a (n + 1) = a n + d
  sum_formula : ∀ n, sum n = n * a 1 + n * (n - 1) / 2 * d

/-- The conditions of the problem -/
def problem_conditions (seq : ArithmeticSequence) : Prop :=
  seq.a 1 + seq.a 2 + seq.a 3 = 156 ∧
  seq.a 2 + seq.a 3 + seq.a 4 = 147

/-- The theorem to prove -/
theorem max_sum (seq : ArithmeticSequence) 
  (h : problem_conditions seq) : 
  ∃ (n : ℕ), n = 19 ∧ 
  ∀ (m : ℕ), m > 0 → seq.sum n ≥ seq.sum m :=
sorry

end NUMINAMATH_CALUDE_max_sum_l1603_160329


namespace NUMINAMATH_CALUDE_plane_representations_l1603_160335

/-- Given a plane with equation 2x - 2y + z - 20 = 0, prove its representations in intercept and normal forms -/
theorem plane_representations (x y z : ℝ) :
  (2*x - 2*y + z - 20 = 0) →
  (x/10 + y/(-10) + z/20 = 1) ∧
  (-2/3*x + 2/3*y - 1/3*z + 20/3 = 0) :=
by sorry

end NUMINAMATH_CALUDE_plane_representations_l1603_160335


namespace NUMINAMATH_CALUDE_employee_transfer_solution_l1603_160344

/-- Represents the company's employee transfer problem -/
def EmployeeTransfer (a : ℝ) (x : ℕ) : Prop :=
  let total_employees : ℕ := 100
  let manufacturing_before : ℝ := a * total_employees
  let manufacturing_after : ℝ := a * 1.2 * (total_employees - x)
  let service_output : ℝ := 3.5 * a * x
  (manufacturing_after ≥ manufacturing_before) ∧ 
  (service_output ≥ 0.5 * manufacturing_before) ∧
  (x ≤ total_employees)

/-- Theorem stating the solution to the employee transfer problem -/
theorem employee_transfer_solution (a : ℝ) (h : a > 0) :
  ∃ x : ℕ, EmployeeTransfer a x ∧ x ≥ 15 ∧ x ≤ 16 :=
sorry

end NUMINAMATH_CALUDE_employee_transfer_solution_l1603_160344


namespace NUMINAMATH_CALUDE_part_one_part_two_l1603_160395

-- Define the function f
def f (a x : ℝ) : ℝ := a * x^2 - (2*a + 1) * x - 1

-- Part (1)
theorem part_one (a : ℝ) :
  (∀ x : ℝ, f a x ≤ -3/4) ↔ a ∈ Set.Icc (-1) (-1/4) :=
sorry

-- Part (2)
theorem part_two (a : ℝ) :
  a ≤ 0 → ((∀ x : ℝ, x > 0 → x * f a x ≤ 1) ↔ a ∈ Set.Icc (-3) 0) :=
sorry

end NUMINAMATH_CALUDE_part_one_part_two_l1603_160395


namespace NUMINAMATH_CALUDE_fountain_area_l1603_160309

theorem fountain_area (AB DC : ℝ) (h1 : AB = 24) (h2 : DC = 14) : 
  let AD : ℝ := AB / 3
  let R : ℝ := Real.sqrt (AD^2 + DC^2)
  π * R^2 = 260 * π := by sorry

end NUMINAMATH_CALUDE_fountain_area_l1603_160309


namespace NUMINAMATH_CALUDE_ada_original_seat_l1603_160371

/-- Represents the seats in the theater --/
inductive Seat
| one
| two
| three
| four
| five
| six

/-- Represents the friends --/
inductive Friend
| ada
| bea
| ceci
| dee
| edie
| fred

/-- Represents the movement of a friend --/
structure Movement where
  friend : Friend
  displacement : Int

/-- The seating arrangement before Ada left --/
def initial_arrangement : Friend → Seat := sorry

/-- The seating arrangement after all movements --/
def final_arrangement : Friend → Seat := sorry

/-- The list of all movements --/
def movements : List Movement := sorry

/-- Calculates the net displacement of all movements --/
def net_displacement (mvs : List Movement) : Int := sorry

/-- Checks if a seat is an end seat --/
def is_end_seat (s : Seat) : Prop := s = Seat.one ∨ s = Seat.six

theorem ada_original_seat (h1 : net_displacement movements = 0)
                          (h2 : is_end_seat (final_arrangement Friend.ada)) :
  is_end_seat (initial_arrangement Friend.ada) := by sorry

end NUMINAMATH_CALUDE_ada_original_seat_l1603_160371


namespace NUMINAMATH_CALUDE_condition_necessary_not_sufficient_l1603_160322

theorem condition_necessary_not_sufficient :
  (∀ x : ℝ, x > 0 → x ≠ 0) ∧
  (∃ x : ℝ, x ≠ 0 ∧ ¬(x > 0)) := by
  sorry

end NUMINAMATH_CALUDE_condition_necessary_not_sufficient_l1603_160322


namespace NUMINAMATH_CALUDE_exactly_one_success_probability_l1603_160381

/-- The probability of success in a single trial -/
def p : ℚ := 1/3

/-- The number of trials -/
def n : ℕ := 3

/-- The number of successes we're interested in -/
def k : ℕ := 1

/-- The binomial coefficient function -/
def binomial_coef (n k : ℕ) : ℚ := (Nat.choose n k : ℚ)

/-- The probability of exactly k successes in n trials with probability p -/
def binomial_probability (n k : ℕ) (p : ℚ) : ℚ :=
  binomial_coef n k * p^k * (1 - p)^(n - k)

theorem exactly_one_success_probability :
  binomial_probability n k p = 4/9 := by sorry

end NUMINAMATH_CALUDE_exactly_one_success_probability_l1603_160381


namespace NUMINAMATH_CALUDE_simplify_expression_l1603_160317

theorem simplify_expression (x : ℝ) (hx : x ≠ 0) :
  x⁻¹ - 3*x + 2 = -(3*x^2 - 2*x - 1) / x := by sorry

end NUMINAMATH_CALUDE_simplify_expression_l1603_160317


namespace NUMINAMATH_CALUDE_dentist_age_problem_l1603_160375

/-- Given a dentist's current age and the relationship between his past and future ages,
    calculate how many years ago his age was being considered. -/
theorem dentist_age_problem (current_age : ℕ) (h : current_age = 32) : 
  ∃ (x : ℕ), (1 / 6 : ℚ) * (current_age - x) = (1 / 10 : ℚ) * (current_age + 8) ∧ x = 8 :=
by sorry

end NUMINAMATH_CALUDE_dentist_age_problem_l1603_160375


namespace NUMINAMATH_CALUDE_max_edges_no_cycle4_l1603_160361

/-- A graph with no cycle of length 4 -/
structure NoCycle4Graph where
  vertexCount : ℕ
  edgeCount : ℕ
  noCycle4 : Bool

/-- The maximum number of edges in a graph with 8 vertices and no 4-cycle -/
def maxEdgesNoCycle4 (g : NoCycle4Graph) : Prop :=
  g.vertexCount = 8 ∧ g.noCycle4 = true → g.edgeCount ≤ 25

/-- Theorem stating the maximum number of edges in a graph with 8 vertices and no 4-cycle -/
theorem max_edges_no_cycle4 (g : NoCycle4Graph) : maxEdgesNoCycle4 g := by
  sorry

#check max_edges_no_cycle4

end NUMINAMATH_CALUDE_max_edges_no_cycle4_l1603_160361


namespace NUMINAMATH_CALUDE_merchant_profit_l1603_160342

/-- Calculates the profit percentage given the ratio of cost price to selling price -/
def profit_percentage (cost_price : ℚ) (selling_price : ℚ) : ℚ :=
  (selling_price - cost_price) / cost_price * 100

/-- Proves that if the cost price of 19 articles is equal to the selling price of 16 articles,
    then the merchant makes a profit of 18.75% -/
theorem merchant_profit :
  ∀ (cost_price selling_price : ℚ),
  19 * cost_price = 16 * selling_price →
  profit_percentage cost_price selling_price = 18.75 := by
sorry

#eval profit_percentage 16 19 -- Should evaluate to 18.75

end NUMINAMATH_CALUDE_merchant_profit_l1603_160342


namespace NUMINAMATH_CALUDE_walking_speed_ratio_l1603_160360

/-- The ratio of a slower walking speed to a usual walking speed, given the times taken for the same distance. -/
theorem walking_speed_ratio (usual_time slower_time : ℝ) 
  (h1 : usual_time = 32)
  (h2 : slower_time = 40) :
  (usual_time / slower_time) = 4 / 5 := by
  sorry

end NUMINAMATH_CALUDE_walking_speed_ratio_l1603_160360


namespace NUMINAMATH_CALUDE_rectangle_perimeter_l1603_160351

/-- Given a triangle with sides 9, 12, and 15 units, and a rectangle with width 6 units
    and area equal to the triangle's area, the perimeter of the rectangle is 30 units. -/
theorem rectangle_perimeter (a b c w : ℝ) (h1 : a = 9) (h2 : b = 12) (h3 : c = 15) (h4 : w = 6)
    (h5 : w * (a * b / 2 / w) = a * b / 2) : 2 * (w + a * b / 2 / w) = 30 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_perimeter_l1603_160351


namespace NUMINAMATH_CALUDE_prime_pair_divisibility_l1603_160398

theorem prime_pair_divisibility (p q : ℕ) (hp : Prime p) (hq : Prime q) :
  (∃ k₁ k₂ : ℤ, ((2 * p^2 - 1)^q + 1 : ℤ) = k₁ * (p + q) ∧ 
                ((2 * q^2 - 1)^p + 1 : ℤ) = k₂ * (p + q)) ↔ 
  p = q := by sorry

end NUMINAMATH_CALUDE_prime_pair_divisibility_l1603_160398


namespace NUMINAMATH_CALUDE_not_divisible_by_5_and_9_l1603_160302

def count_not_divisible (n : ℕ) (a b : ℕ) : ℕ :=
  n - (n / a + n / b - n / (a * b))

theorem not_divisible_by_5_and_9 :
  count_not_divisible 1199 5 9 = 853 := by
  sorry

end NUMINAMATH_CALUDE_not_divisible_by_5_and_9_l1603_160302


namespace NUMINAMATH_CALUDE_min_value_theorem_l1603_160338

/-- The line equation ax - by + 3 = 0 --/
def line_equation (a b x y : ℝ) : Prop := a * x - b * y + 3 = 0

/-- The circle equation x^2 + y^2 + 2x - 4y + 1 = 0 --/
def circle_equation (x y : ℝ) : Prop := x^2 + y^2 + 2*x - 4*y + 1 = 0

/-- The line divides the area of the circle in half --/
def line_bisects_circle (a b : ℝ) : Prop := 
  ∃ x y : ℝ, line_equation a b x y ∧ circle_equation x y

/-- The main theorem --/
theorem min_value_theorem (a b : ℝ) (ha : a > 0) (hb : b > 1) 
  (h_bisect : line_bisects_circle a b) : 
  (∀ a' b' : ℝ, a' > 0 → b' > 1 → line_bisects_circle a' b' → 
    2/a + 1/(b-1) ≤ 2/a' + 1/(b'-1)) → 
  2/a + 1/(b-1) = 8 :=
sorry

end NUMINAMATH_CALUDE_min_value_theorem_l1603_160338


namespace NUMINAMATH_CALUDE_circle_center_sum_l1603_160306

/-- Given a circle with equation x^2 + y^2 = 4x - 6y + 9, 
    the sum of the x and y coordinates of its center is -1. -/
theorem circle_center_sum (x y : ℝ) : 
  x^2 + y^2 = 4*x - 6*y + 9 → 
  ∃ (h k : ℝ), (∀ (a b : ℝ), (a - h)^2 + (b - k)^2 = (x - h)^2 + (y - k)^2) ∧ h + k = -1 := by
  sorry

end NUMINAMATH_CALUDE_circle_center_sum_l1603_160306


namespace NUMINAMATH_CALUDE_johns_piano_expenses_l1603_160320

/-- The total cost of John's piano learning expenses --/
def total_cost (piano_cost lesson_count lesson_price discount sheet_music maintenance : ℚ) : ℚ :=
  piano_cost + 
  (lesson_count * lesson_price * (1 - discount)) + 
  sheet_music + 
  maintenance

/-- Theorem stating that John's total piano learning expenses are $1275 --/
theorem johns_piano_expenses : 
  total_cost 500 20 40 (25/100) 75 100 = 1275 := by
  sorry

end NUMINAMATH_CALUDE_johns_piano_expenses_l1603_160320


namespace NUMINAMATH_CALUDE_dance_group_equality_l1603_160358

def dance_group_total (initial_boys initial_girls weekly_boys_increase weekly_girls_increase : ℕ) : ℕ :=
  let weeks := (initial_boys - initial_girls) / (weekly_girls_increase - weekly_boys_increase)
  let final_boys := initial_boys + weeks * weekly_boys_increase
  let final_girls := initial_girls + weeks * weekly_girls_increase
  final_boys + final_girls

theorem dance_group_equality (initial_boys initial_girls weekly_boys_increase weekly_girls_increase : ℕ) 
  (h1 : initial_boys = 39)
  (h2 : initial_girls = 23)
  (h3 : weekly_boys_increase = 6)
  (h4 : weekly_girls_increase = 8) :
  dance_group_total initial_boys initial_girls weekly_boys_increase weekly_girls_increase = 174 := by
  sorry

end NUMINAMATH_CALUDE_dance_group_equality_l1603_160358


namespace NUMINAMATH_CALUDE_committee_count_l1603_160391

/-- Represents a department with male and female professors -/
structure Department where
  male_profs : Nat
  female_profs : Nat

/-- Represents the configuration of the science division -/
structure ScienceDivision where
  departments : Fin 3 → Department

/-- Represents a committee formation -/
structure Committee where
  members : Fin 6 → Nat
  department_count : Fin 3 → Nat
  male_count : Nat
  female_count : Nat

def is_valid_committee (sd : ScienceDivision) (c : Committee) : Prop :=
  c.male_count = 3 ∧ 
  c.female_count = 3 ∧ 
  (∀ d : Fin 3, c.department_count d = 2)

def count_valid_committees (sd : ScienceDivision) : Nat :=
  sorry

theorem committee_count (sd : ScienceDivision) : 
  (∀ d : Fin 3, sd.departments d = ⟨3, 3⟩) → 
  count_valid_committees sd = 1215 := by
  sorry

end NUMINAMATH_CALUDE_committee_count_l1603_160391


namespace NUMINAMATH_CALUDE_john_personal_payment_l1603_160347

def hearing_aid_cost : ℝ := 2500
def insurance_coverage_percentage : ℝ := 80
def number_of_hearing_aids : ℕ := 2

theorem john_personal_payment (total_cost : ℝ) (insurance_payment : ℝ) (john_payment : ℝ) :
  total_cost = number_of_hearing_aids * hearing_aid_cost →
  insurance_payment = (insurance_coverage_percentage / 100) * total_cost →
  john_payment = total_cost - insurance_payment →
  john_payment = 1000 := by
sorry

end NUMINAMATH_CALUDE_john_personal_payment_l1603_160347


namespace NUMINAMATH_CALUDE_jiwon_distance_to_school_l1603_160384

/-- The distance from Taehong's house to school in kilometers -/
def taehong_distance : ℝ := 1.05

/-- The difference between Taehong's and Jiwon's distances in kilometers -/
def distance_difference : ℝ := 0.46

/-- The distance from Jiwon's house to school in kilometers -/
def jiwon_distance : ℝ := taehong_distance - distance_difference

theorem jiwon_distance_to_school :
  jiwon_distance = 0.59 := by sorry

end NUMINAMATH_CALUDE_jiwon_distance_to_school_l1603_160384


namespace NUMINAMATH_CALUDE_special_line_equation_l1603_160303

/-- A line passing through (-3, 4) with intercepts summing to 12 -/
def special_line (a b : ℝ) : Prop :=
  a + b = 12 ∧ -3 / a + 4 / b = 1

/-- The equation of the special line -/
def line_equation (x y : ℝ) : Prop :=
  x + 3 * y - 9 = 0 ∨ 4 * x - y + 16 = 0

/-- Theorem stating that the special line satisfies one of the two equations -/
theorem special_line_equation :
  ∀ a b : ℝ, special_line a b → ∃ x y : ℝ, line_equation x y :=
by
  sorry

end NUMINAMATH_CALUDE_special_line_equation_l1603_160303


namespace NUMINAMATH_CALUDE_square_difference_fifty_fortynine_l1603_160397

theorem square_difference_fifty_fortynine : (50 : ℕ)^2 - (49 : ℕ)^2 = 99 := by
  sorry

end NUMINAMATH_CALUDE_square_difference_fifty_fortynine_l1603_160397


namespace NUMINAMATH_CALUDE_largest_a_for_fibonacci_sum_l1603_160324

/-- Fibonacci sequence -/
def fib : ℕ → ℕ
  | 0 => 0
  | 1 => 1
  | n + 2 => fib (n + 1) + fib n

/-- Property: Fₐ, Fᵦ, Fᶜ form an increasing arithmetic sequence -/
def is_arithmetic_seq (a b c : ℕ) : Prop :=
  fib b - fib a = fib c - fib b ∧ fib a < fib b ∧ fib b < fib c

/-- Main theorem -/
theorem largest_a_for_fibonacci_sum (a b c : ℕ) :
  is_arithmetic_seq a b c →
  a + b + c ≤ 3000 →
  a ≤ 998 :=
by sorry

end NUMINAMATH_CALUDE_largest_a_for_fibonacci_sum_l1603_160324


namespace NUMINAMATH_CALUDE_sqrt_sum_eq_8_implies_product_l1603_160349

theorem sqrt_sum_eq_8_implies_product (x : ℝ) :
  Real.sqrt (8 + x) + Real.sqrt (25 - x) = 8 →
  (8 + x) * (25 - x) = 961 / 4 := by
sorry

end NUMINAMATH_CALUDE_sqrt_sum_eq_8_implies_product_l1603_160349


namespace NUMINAMATH_CALUDE_part1_part2_l1603_160318

-- Define the function f
def f (a b x : ℝ) : ℝ := x^2 - a*x + b

-- Define the solution set of f(x) < 0
def solution_set (a b : ℝ) : Set ℝ := {x | 2 < x ∧ x < 3}

-- Define the condition for part 2
def condition_part2 (a : ℝ) : Prop := ∀ x ∈ Set.Icc (-1) 0, f a (3 - a) x ≥ 0

-- Statement for part 1
theorem part1 (a b : ℝ) : 
  (∀ x, f a b x < 0 ↔ x ∈ solution_set a b) → 
  (∀ x, b*x^2 - a*x + 1 > 0 ↔ x < 1/3 ∨ x > 1/2) :=
sorry

-- Statement for part 2
theorem part2 (a : ℝ) :
  condition_part2 a → a ≤ 3 :=
sorry

end NUMINAMATH_CALUDE_part1_part2_l1603_160318


namespace NUMINAMATH_CALUDE_karl_savings_l1603_160308

def folder_cost : ℝ := 3.50
def num_folders : ℕ := 10
def base_discount : ℝ := 0.15
def bulk_discount : ℝ := 0.05
def total_discount : ℝ := base_discount + bulk_discount

theorem karl_savings : 
  num_folders * folder_cost - num_folders * folder_cost * (1 - total_discount) = 7 :=
by sorry

end NUMINAMATH_CALUDE_karl_savings_l1603_160308


namespace NUMINAMATH_CALUDE_planted_field_fraction_l1603_160340

theorem planted_field_fraction (a b x : ℝ) (h₁ : a = 5) (h₂ : b = 12) (h₃ : x = 3) :
  let total_area := (a * b) / 2
  let square_area := x^2
  let planted_area := total_area - square_area
  planted_area / total_area = 7 / 10 := by
sorry

end NUMINAMATH_CALUDE_planted_field_fraction_l1603_160340


namespace NUMINAMATH_CALUDE_largest_prime_factor_of_expression_l1603_160380

theorem largest_prime_factor_of_expression : 
  ∃ (p : ℕ), Nat.Prime p ∧ 
  p ∣ (18^4 + 2 * 18^2 + 1 - 17^4) ∧ 
  ∀ (q : ℕ), Nat.Prime q → q ∣ (18^4 + 2 * 18^2 + 1 - 17^4) → q ≤ p ∧
  p = 307 := by
  sorry

end NUMINAMATH_CALUDE_largest_prime_factor_of_expression_l1603_160380


namespace NUMINAMATH_CALUDE_pyramid_volume_l1603_160368

/-- The volume of a pyramid with a rectangular base and a slant edge perpendicular to two adjacent sides of the base. -/
theorem pyramid_volume (base_length base_width slant_edge : ℝ) 
  (hl : base_length = 10) 
  (hw : base_width = 6) 
  (hs : slant_edge = 20) : 
  (1 / 3 : ℝ) * base_length * base_width * Real.sqrt (slant_edge^2 - base_length^2) = 200 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_pyramid_volume_l1603_160368


namespace NUMINAMATH_CALUDE_probability_is_one_over_sixtythree_l1603_160300

/-- Represents the color of a bead -/
inductive BeadColor
  | Red
  | White
  | Blue

/-- Represents a line of beads -/
def BeadLine := List BeadColor

/-- The total number of beads -/
def totalBeads : Nat := 9

/-- The number of red beads -/
def redBeads : Nat := 4

/-- The number of white beads -/
def whiteBeads : Nat := 3

/-- The number of blue beads -/
def blueBeads : Nat := 2

/-- Checks if no two neighboring beads in a line have the same color -/
def noAdjacentSameColor (line : BeadLine) : Bool :=
  sorry

/-- Generates all possible bead lines -/
def allBeadLines : List BeadLine :=
  sorry

/-- Counts the number of valid bead lines where no two neighboring beads have the same color -/
def countValidLines : Nat :=
  sorry

/-- The probability of no two neighboring beads being the same color -/
def probability : Rat :=
  sorry

theorem probability_is_one_over_sixtythree : 
  probability = 1 / 63 := by sorry

end NUMINAMATH_CALUDE_probability_is_one_over_sixtythree_l1603_160300


namespace NUMINAMATH_CALUDE_mistaken_division_l1603_160311

theorem mistaken_division (n : ℕ) : 
  (n % 32 = 0 ∧ n / 32 = 3) → n / 4 = 24 := by
  sorry

end NUMINAMATH_CALUDE_mistaken_division_l1603_160311


namespace NUMINAMATH_CALUDE_complement_of_M_l1603_160328

def U : Set ℝ := Set.univ
def M : Set ℝ := {x | x^2 - 4 ≤ 0}

theorem complement_of_M (x : ℝ) : x ∈ (Set.compl M) ↔ x < -2 ∨ x > 2 := by
  sorry

end NUMINAMATH_CALUDE_complement_of_M_l1603_160328


namespace NUMINAMATH_CALUDE_ten_thousand_eight_hundred_seventy_scientific_notation_l1603_160307

def scientific_notation (n : ℝ) (a : ℝ) (b : ℤ) : Prop :=
  n = a * (10 : ℝ) ^ b ∧ 1 ≤ a ∧ a < 10

theorem ten_thousand_eight_hundred_seventy_scientific_notation :
  scientific_notation 10870 1.087 4 := by
  sorry

end NUMINAMATH_CALUDE_ten_thousand_eight_hundred_seventy_scientific_notation_l1603_160307


namespace NUMINAMATH_CALUDE_prob_three_games_correct_constant_term_is_three_f_one_half_l1603_160370

/-- Represents the probability of player A winning a single game -/
def p : ℝ := sorry

/-- Assumption that p is between 0 and 1 -/
axiom p_range : 0 ≤ p ∧ p ≤ 1

/-- The probability of the match ending in three games -/
def prob_three_games : ℝ := p^3 + (1-p)^3

/-- The expected number of games played in the match -/
def f (p : ℝ) : ℝ := 6*p^4 - 12*p^3 + 3*p^2 + 3*p + 3

/-- Theorem: The probability of the match ending in three games is p³ + (1-p)³ -/
theorem prob_three_games_correct : prob_three_games = p^3 + (1-p)^3 := by sorry

/-- Theorem: The constant term of f(p) is 3 -/
theorem constant_term_is_three : f 0 = 3 := by sorry

/-- Theorem: f(1/2) = 33/8 -/
theorem f_one_half : f (1/2) = 33/8 := by sorry

end NUMINAMATH_CALUDE_prob_three_games_correct_constant_term_is_three_f_one_half_l1603_160370


namespace NUMINAMATH_CALUDE_school_attendance_problem_l1603_160356

theorem school_attendance_problem (boys : ℕ) (girls : ℕ) :
  boys = 2000 →
  (boys + girls : ℝ) = 1.4 * boys →
  girls = 800 := by
sorry

end NUMINAMATH_CALUDE_school_attendance_problem_l1603_160356


namespace NUMINAMATH_CALUDE_matilda_earnings_l1603_160321

/-- Calculates the total earnings for a newspaper delivery job -/
def calculate_earnings (hourly_wage : ℚ) (per_newspaper : ℚ) (newspapers_per_hour : ℕ) (shift_duration : ℕ) : ℚ :=
  let wage_earnings := hourly_wage * shift_duration
  let newspaper_earnings := per_newspaper * newspapers_per_hour * shift_duration
  wage_earnings + newspaper_earnings

/-- Proves that Matilda's earnings for a 3-hour shift equal $40.50 -/
theorem matilda_earnings : 
  calculate_earnings 6 (1/4) 30 3 = 81/2 := by
  sorry

#eval calculate_earnings 6 (1/4) 30 3

end NUMINAMATH_CALUDE_matilda_earnings_l1603_160321


namespace NUMINAMATH_CALUDE_certain_number_for_prime_squared_l1603_160326

theorem certain_number_for_prime_squared (p : ℕ) (h_prime : Nat.Prime p) (h_gt_3 : p > 3) :
  ∃! n : ℕ, (p^2 + n) % 12 = 2 ∧ n = 1 := by
  sorry

end NUMINAMATH_CALUDE_certain_number_for_prime_squared_l1603_160326


namespace NUMINAMATH_CALUDE_quadratic_rational_root_even_coefficient_l1603_160357

theorem quadratic_rational_root_even_coefficient 
  (a b c : ℤ) (h_a_nonzero : a ≠ 0) 
  (h_rational_root : ∃ (p q : ℤ), q ≠ 0 ∧ a * (p / q)^2 + b * (p / q) + c = 0) :
  Even a ∨ Even b ∨ Even c :=
sorry

end NUMINAMATH_CALUDE_quadratic_rational_root_even_coefficient_l1603_160357


namespace NUMINAMATH_CALUDE_shoe_cost_l1603_160373

/-- The cost of shoes given an initial budget and remaining amount --/
theorem shoe_cost (initial_budget remaining : ℚ) (h1 : initial_budget = 999) (h2 : remaining = 834) :
  initial_budget - remaining = 165 := by
  sorry

end NUMINAMATH_CALUDE_shoe_cost_l1603_160373


namespace NUMINAMATH_CALUDE_solution_for_given_condition_l1603_160312

noncomputable def f (a x : ℝ) : ℝ := (a * x - 1) / (x^2 - 1)

theorem solution_for_given_condition (a : ℝ) :
  (∀ x, f a x > 0 ↔ a > 1/3) → ∃ x, x = 3 ∧ f (1/3) x = 0 :=
by sorry

end NUMINAMATH_CALUDE_solution_for_given_condition_l1603_160312


namespace NUMINAMATH_CALUDE_gcd_polynomial_and_y_l1603_160330

theorem gcd_polynomial_and_y (y : ℤ) (h : ∃ k : ℤ, y = 46896 * k) :
  let g := fun (y : ℤ) => (3*y+5)*(8*y+3)*(16*y+9)*(y+16)
  Nat.gcd (Int.natAbs (g y)) (Int.natAbs y) = 2160 := by
  sorry

end NUMINAMATH_CALUDE_gcd_polynomial_and_y_l1603_160330


namespace NUMINAMATH_CALUDE_initial_amount_proof_l1603_160363

/-- The initial amount given on interest -/
def P : ℝ := 1250

/-- The interest rate per annum (in decimal form) -/
def r : ℝ := 0.04

/-- The number of years -/
def n : ℕ := 2

/-- The difference between compound and simple interest -/
def interest_difference : ℝ := 2.0000000000002274

theorem initial_amount_proof :
  P * ((1 + r)^n - (1 + r * n)) = interest_difference := by
  sorry

end NUMINAMATH_CALUDE_initial_amount_proof_l1603_160363


namespace NUMINAMATH_CALUDE_questions_for_first_project_l1603_160394

/-- Given a mathematician who completes a fixed number of questions per day for a week and needs to write a specific number of questions for one project, this theorem calculates the number of questions for the other project. -/
theorem questions_for_first_project 
  (questions_per_day : ℕ) 
  (days_in_week : ℕ) 
  (questions_for_second_project : ℕ) 
  (h1 : questions_per_day = 142) 
  (h2 : days_in_week = 7) 
  (h3 : questions_for_second_project = 476) : 
  questions_per_day * days_in_week - questions_for_second_project = 518 := by
  sorry

#eval 142 * 7 - 476  -- Should output 518

end NUMINAMATH_CALUDE_questions_for_first_project_l1603_160394


namespace NUMINAMATH_CALUDE_factory_problem_l1603_160385

/-- Represents the production rates and working days of two factories -/
structure FactoryProduction where
  initial_rate_B : ℝ
  initial_rate_A : ℝ
  total_days : ℕ
  adjustment_days : ℕ

/-- The solution to the factory production problem -/
def factory_problem_solution (fp : FactoryProduction) : ℝ :=
  3

/-- Theorem stating the solution to the factory production problem -/
theorem factory_problem (fp : FactoryProduction) :
  fp.initial_rate_A = (4/3) * fp.initial_rate_B →
  fp.total_days = 6 →
  fp.adjustment_days = 1 →
  let days_before := fp.total_days - fp.adjustment_days - (factory_problem_solution fp)
  let production_A := fp.initial_rate_A * fp.total_days
  let production_B := fp.initial_rate_B * days_before + 2 * fp.initial_rate_B * (factory_problem_solution fp)
  production_A = production_B :=
by sorry

end NUMINAMATH_CALUDE_factory_problem_l1603_160385


namespace NUMINAMATH_CALUDE_gcd_problem_l1603_160352

theorem gcd_problem (a : ℤ) (h : ∃ k : ℤ, a = (2*k + 1) * 1183) :
  Int.gcd (2*a^2 + 29*a + 65) (a + 13) = 26 := by
  sorry

end NUMINAMATH_CALUDE_gcd_problem_l1603_160352


namespace NUMINAMATH_CALUDE_tv_production_last_five_days_l1603_160355

theorem tv_production_last_five_days 
  (total_days : Nat) 
  (first_period : Nat) 
  (avg_first_period : Nat) 
  (avg_total : Nat) 
  (h1 : total_days = 30)
  (h2 : first_period = 25)
  (h3 : avg_first_period = 50)
  (h4 : avg_total = 48) :
  (total_days * avg_total - first_period * avg_first_period) / (total_days - first_period) = 38 := by
  sorry

#check tv_production_last_five_days

end NUMINAMATH_CALUDE_tv_production_last_five_days_l1603_160355


namespace NUMINAMATH_CALUDE_square_division_l1603_160334

/-- A square can be divided into n smaller squares for any natural number n ≥ 6 -/
theorem square_division (n : ℕ) (h : n ≥ 6) : 
  ∃ (partition : List (ℕ × ℕ)), 
    (partition.length = n) ∧ 
    (∀ (x y : ℕ × ℕ), x ∈ partition → y ∈ partition → x ≠ y → 
      (x.1 < y.1 ∨ x.2 < y.2 ∨ y.1 < x.1 ∨ y.2 < x.2)) ∧
    (∃ (side : ℕ), ∀ (square : ℕ × ℕ), square ∈ partition → 
      square.1 ≤ side ∧ square.2 ≤ side) := by
  sorry

end NUMINAMATH_CALUDE_square_division_l1603_160334


namespace NUMINAMATH_CALUDE_d₁_d₂_not_divisible_by_3_l1603_160323

-- Define d₁ and d₂ as functions of a
def d₁ (a : ℕ) : ℕ := a^3 + 3^a + a * 3^((a+1)/2)
def d₂ (a : ℕ) : ℕ := a^3 + 3^a - a * 3^((a+1)/2)

-- Define the main theorem
theorem d₁_d₂_not_divisible_by_3 :
  ∀ a : ℕ, 1 ≤ a → a ≤ 251 → ¬(3 ∣ (d₁ a * d₂ a)) :=
by sorry

end NUMINAMATH_CALUDE_d₁_d₂_not_divisible_by_3_l1603_160323


namespace NUMINAMATH_CALUDE_sum_radii_greater_incircle_radius_l1603_160387

-- Define the triangle and circles
variable (A B C : EuclideanPlane) (S S₁ S₂ : Circle EuclideanPlane)

-- Define the radii
variable (r r₁ r₂ : ℝ)

-- Assumptions
variable (h_triangle : Triangle A B C)
variable (h_incircle : S.IsIncircle h_triangle)
variable (h_S₁_tangent : S₁.IsTangentTo (SegmentND A B) ∧ S₁.IsTangentTo (SegmentND A C))
variable (h_S₂_tangent : S₂.IsTangentTo (SegmentND A B) ∧ S₂.IsTangentTo (SegmentND B C))
variable (h_S₁S₂_tangent : S₁.IsExternallyTangentTo S₂)
variable (h_r : S.radius = r)
variable (h_r₁ : S₁.radius = r₁)
variable (h_r₂ : S₂.radius = r₂)

-- Theorem statement
theorem sum_radii_greater_incircle_radius : r₁ + r₂ > r := by
  sorry

end NUMINAMATH_CALUDE_sum_radii_greater_incircle_radius_l1603_160387


namespace NUMINAMATH_CALUDE_tan_sin_equation_l1603_160374

theorem tan_sin_equation (m : ℝ) : 
  Real.tan (20 * π / 180) + m * Real.sin (20 * π / 180) = Real.sqrt 3 → m = 4 := by
  sorry

end NUMINAMATH_CALUDE_tan_sin_equation_l1603_160374


namespace NUMINAMATH_CALUDE_ast_equation_solutions_l1603_160392

-- Define the operation ※
def ast (a b : ℝ) : ℝ := a + b^2

-- Theorem statement
theorem ast_equation_solutions :
  ∃! (s : Set ℝ), s = {x : ℝ | ast x (x + 1) = 5} ∧ s = {1, -4} :=
by sorry

end NUMINAMATH_CALUDE_ast_equation_solutions_l1603_160392


namespace NUMINAMATH_CALUDE_max_sundays_in_45_days_l1603_160369

/-- Represents the day of the week -/
inductive DayOfWeek
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday
  | Saturday
  | Sunday

/-- Represents a date within the first 45 days of a year -/
structure Date :=
  (day : Nat)
  (dayOfWeek : DayOfWeek)

/-- Returns the number of Sundays in the first 45 days of a year -/
def countSundays (startDay : DayOfWeek) : Nat :=
  sorry

/-- The maximum number of Sundays in the first 45 days of a year -/
def maxSundays : Nat :=
  sorry

theorem max_sundays_in_45_days :
  maxSundays = 7 :=
sorry

end NUMINAMATH_CALUDE_max_sundays_in_45_days_l1603_160369


namespace NUMINAMATH_CALUDE_lanas_final_page_count_l1603_160372

theorem lanas_final_page_count (lana_initial : ℕ) (duane_total : ℕ) : 
  lana_initial = 8 → duane_total = 42 → lana_initial + duane_total / 2 = 29 := by
  sorry

end NUMINAMATH_CALUDE_lanas_final_page_count_l1603_160372


namespace NUMINAMATH_CALUDE_roots_of_equation_l1603_160339

def f (x : ℝ) := x^2 - |x - 1| - 1

theorem roots_of_equation :
  ∃ (x₁ x₂ : ℝ), x₁ > x₂ ∧ f x₁ = 0 ∧ f x₂ = 0 ∧ x₁ = 1 ∧ x₂ = -2 :=
by
  sorry

end NUMINAMATH_CALUDE_roots_of_equation_l1603_160339


namespace NUMINAMATH_CALUDE_decagon_triangles_l1603_160386

def regularDecagonVertices : ℕ := 10

def trianglesFromDecagon : ℕ :=
  Nat.choose regularDecagonVertices 3

theorem decagon_triangles :
  trianglesFromDecagon = 120 := by sorry

end NUMINAMATH_CALUDE_decagon_triangles_l1603_160386


namespace NUMINAMATH_CALUDE_sum_of_complex_roots_of_unity_l1603_160305

theorem sum_of_complex_roots_of_unity (a b c : ℂ) :
  (Complex.abs a = 1) →
  (Complex.abs b = 1) →
  (Complex.abs c = 1) →
  (a^2 / (b*c) + b^2 / (a*c) + c^2 / (a*b) = -1) →
  (Complex.abs (a + b + c) = 1 ∨ Complex.abs (a + b + c) = 2) :=
by sorry

end NUMINAMATH_CALUDE_sum_of_complex_roots_of_unity_l1603_160305


namespace NUMINAMATH_CALUDE_triangle_reciprocal_sum_l1603_160313

/-- Given a triangle ABC with angle ratio A:B:C = 4:2:1, 
    prove that 1/a + 1/b = 1/c, where a, b, and c are the 
    sides opposite to angles A, B, and C respectively. -/
theorem triangle_reciprocal_sum (A B C a b c : ℝ) 
  (h_triangle : A + B + C = π) 
  (h_ratio : A = 4 * (π / 7) ∧ B = 2 * (π / 7) ∧ C = π / 7)
  (h_sides : a = 2 * Real.sin A ∧ b = 2 * Real.sin B ∧ c = 2 * Real.sin C) :
  1 / a + 1 / b = 1 / c :=
sorry

end NUMINAMATH_CALUDE_triangle_reciprocal_sum_l1603_160313


namespace NUMINAMATH_CALUDE_problem_statement_l1603_160378

theorem problem_statement (m n k : ℝ) 
  (h1 : 3^m = k) 
  (h2 : 5^n = k) 
  (h3 : 1/m + 1/n = 2) 
  (h4 : k > 0) : k = Real.sqrt 15 := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l1603_160378


namespace NUMINAMATH_CALUDE_abs_m_minus_n_equals_2_sqrt_3_l1603_160365

theorem abs_m_minus_n_equals_2_sqrt_3 (m n p : ℝ) 
  (h1 : m * n = 6)
  (h2 : m + n + p = 7)
  (h3 : p = 1) :
  |m - n| = 2 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_abs_m_minus_n_equals_2_sqrt_3_l1603_160365


namespace NUMINAMATH_CALUDE_hyosung_mimi_distance_l1603_160315

/-- Calculates the remaining distance between two people walking towards each other. -/
def remaining_distance (initial_distance : ℝ) (speed1 : ℝ) (speed2 : ℝ) (time : ℝ) : ℝ :=
  initial_distance - (speed1 + speed2) * time

/-- Theorem stating the remaining distance between Hyosung and Mimi after 15 minutes. -/
theorem hyosung_mimi_distance :
  let initial_distance : ℝ := 2.5
  let mimi_speed : ℝ := 2.4
  let hyosung_speed : ℝ := 0.08 * 60
  let time : ℝ := 15 / 60
  remaining_distance initial_distance mimi_speed hyosung_speed time = 0.7 := by
  sorry

end NUMINAMATH_CALUDE_hyosung_mimi_distance_l1603_160315


namespace NUMINAMATH_CALUDE_assembled_figure_surface_area_l1603_160366

/-- The surface area of a figure assembled from four identical bars -/
def figureSurfaceArea (barSurfaceArea : ℝ) (lostAreaPerJunction : ℝ) : ℝ :=
  4 * (barSurfaceArea - lostAreaPerJunction)

/-- Theorem: The surface area of the assembled figure is 64 cm² -/
theorem assembled_figure_surface_area :
  figureSurfaceArea 18 2 = 64 := by
  sorry

end NUMINAMATH_CALUDE_assembled_figure_surface_area_l1603_160366


namespace NUMINAMATH_CALUDE_min_value_of_expression_l1603_160332

/-- Given a line ax - 2by = 2 (where a > 0 and b > 0) passing through the center of the circle 
    x² + y² - 4x + 2y + 1 = 0, the minimum value of 4/(a+2) + 1/(b+1) is 9/4. -/
theorem min_value_of_expression (a b : ℝ) : a > 0 → b > 0 → 
  (∃ x y : ℝ, x^2 + y^2 - 4*x + 2*y + 1 = 0 ∧ a*x - 2*b*y = 2) → 
  (∀ a' b' : ℝ, a' > 0 → b' > 0 → 
    (∃ x' y' : ℝ, x'^2 + y'^2 - 4*x' + 2*y' + 1 = 0 ∧ a'*x' - 2*b'*y' = 2) → 
    4/(a+2) + 1/(b+1) ≤ 4/(a'+2) + 1/(b'+1)) → 
  4/(a+2) + 1/(b+1) = 9/4 :=
sorry

end NUMINAMATH_CALUDE_min_value_of_expression_l1603_160332
