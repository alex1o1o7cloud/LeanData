import Mathlib

namespace NUMINAMATH_CALUDE_percentage_students_like_blue_l3670_367018

/-- Proves that 30% of students like blue given the problem conditions --/
theorem percentage_students_like_blue :
  ∀ (total_students : ℕ) (blue_yellow_count : ℕ) (red_ratio : ℚ),
    total_students = 200 →
    blue_yellow_count = 144 →
    red_ratio = 2/5 →
    ∃ (blue_ratio : ℚ),
      blue_ratio = 3/10 ∧
      blue_ratio * total_students + 
      (1 - blue_ratio) * (1 - red_ratio) * total_students = blue_yellow_count :=
by sorry

end NUMINAMATH_CALUDE_percentage_students_like_blue_l3670_367018


namespace NUMINAMATH_CALUDE_quadratic_minimum_l3670_367072

theorem quadratic_minimum (x : ℝ) : x^2 + 6*x + 1 ≥ -8 ∧ ∃ y : ℝ, y^2 + 6*y + 1 = -8 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_minimum_l3670_367072


namespace NUMINAMATH_CALUDE_quadratic_roots_range_l3670_367010

theorem quadratic_roots_range (a : ℝ) : 
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ x₁ > 1 ∧ x₂ > 1 ∧ x₁^2 - 4*x₁ + a = 0 ∧ x₂^2 - 4*x₂ + a = 0) 
  ↔ 
  (3 < a ∧ a ≤ 4) :=
sorry

end NUMINAMATH_CALUDE_quadratic_roots_range_l3670_367010


namespace NUMINAMATH_CALUDE_specific_plot_fencing_cost_l3670_367050

/-- A rectangular plot with specific dimensions and fencing cost. -/
structure RectangularPlot where
  width : ℝ
  length : ℝ
  fencing_cost_per_meter : ℝ
  perimeter : ℝ

/-- The total cost of fencing for a rectangular plot. -/
def total_fencing_cost (plot : RectangularPlot) : ℝ :=
  plot.perimeter * plot.fencing_cost_per_meter

/-- Theorem stating the total fencing cost for a specific rectangular plot. -/
theorem specific_plot_fencing_cost :
  ∃ (plot : RectangularPlot),
    plot.length = plot.width + 10 ∧
    plot.perimeter = 220 ∧
    plot.fencing_cost_per_meter = 6.5 ∧
    total_fencing_cost plot = 1430 := by
  sorry

end NUMINAMATH_CALUDE_specific_plot_fencing_cost_l3670_367050


namespace NUMINAMATH_CALUDE_sqrt_sin_cos_identity_l3670_367075

theorem sqrt_sin_cos_identity : 
  Real.sqrt (1 - 2 * Real.sin (π + 2) * Real.cos (π - 2)) = Real.sin 2 - Real.cos 2 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_sin_cos_identity_l3670_367075


namespace NUMINAMATH_CALUDE_stairs_climbed_total_l3670_367002

theorem stairs_climbed_total (jonny_stairs : ℕ) (julia_stairs : ℕ) : 
  jonny_stairs = 1269 →
  julia_stairs = jonny_stairs / 3 - 7 →
  jonny_stairs + julia_stairs = 1685 :=
by sorry

end NUMINAMATH_CALUDE_stairs_climbed_total_l3670_367002


namespace NUMINAMATH_CALUDE_quadratic_root_problem_l3670_367012

theorem quadratic_root_problem (m : ℝ) : 
  (3^2 - m * 3 + 3 = 0) → 
  (∃ (x : ℝ), x ≠ 3 ∧ x^2 - m * x + 3 = 0 ∧ x = 1) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_root_problem_l3670_367012


namespace NUMINAMATH_CALUDE_both_are_dwarves_l3670_367059

-- Define the types of inhabitants
inductive Inhabitant : Type
| Elf : Inhabitant
| Dwarf : Inhabitant

-- Define the types of statements
inductive Statement : Type
| GoldStatement : Statement
| AboutDwarf : Statement
| Other : Statement

-- Define a function to determine if a statement is true based on the speaker and the type of statement
def isTruthful (speaker : Inhabitant) (stmnt : Statement) : Prop :=
  match speaker, stmnt with
  | Inhabitant.Dwarf, Statement.GoldStatement => False
  | Inhabitant.Elf, Statement.AboutDwarf => False
  | _, _ => True

-- A's statement
def a_statement : Statement := Statement.GoldStatement

-- B's statement about A
def b_statement (a_type : Inhabitant) : Statement :=
  match a_type with
  | Inhabitant.Dwarf => Statement.Other
  | Inhabitant.Elf => Statement.AboutDwarf

-- Theorem to prove
theorem both_are_dwarves :
  ∃ (a_type b_type : Inhabitant),
    a_type = Inhabitant.Dwarf ∧
    b_type = Inhabitant.Dwarf ∧
    isTruthful a_type a_statement = False ∧
    isTruthful b_type (b_statement a_type) = True :=
sorry

end NUMINAMATH_CALUDE_both_are_dwarves_l3670_367059


namespace NUMINAMATH_CALUDE_ticket_sales_solution_l3670_367065

/-- Represents the number of tickets sold for each type -/
structure TicketSales where
  a : ℕ
  b : ℕ
  c : ℕ

/-- Defines the conditions of the ticket sales problem -/
def validTicketSales (s : TicketSales) : Prop :=
  s.a + s.b + s.c = 400 ∧
  50 * s.a + 40 * s.b + 30 * s.c = 15500 ∧
  s.b = s.c

/-- Theorem stating the solution to the ticket sales problem -/
theorem ticket_sales_solution :
  ∃ (s : TicketSales), validTicketSales s ∧ s.a = 100 ∧ s.b = 150 ∧ s.c = 150 := by
  sorry


end NUMINAMATH_CALUDE_ticket_sales_solution_l3670_367065


namespace NUMINAMATH_CALUDE_wendy_distance_difference_l3670_367061

/-- The distance Wendy ran in miles -/
def distance_run : ℝ := 19.83

/-- The distance Wendy walked in miles -/
def distance_walked : ℝ := 9.17

/-- The difference between the distance Wendy ran and walked -/
def distance_difference : ℝ := distance_run - distance_walked

theorem wendy_distance_difference :
  distance_difference = 10.66 := by sorry

end NUMINAMATH_CALUDE_wendy_distance_difference_l3670_367061


namespace NUMINAMATH_CALUDE_complex_fraction_equals_two_l3670_367001

theorem complex_fraction_equals_two (c d : ℂ) (hc : c ≠ 0) (hd : d ≠ 0) 
  (h : c^2 - c*d + d^2 = 0) : 
  (c^6 + d^6) / (c - d)^6 = 2 := by
  sorry

end NUMINAMATH_CALUDE_complex_fraction_equals_two_l3670_367001


namespace NUMINAMATH_CALUDE_pitcher_juice_distribution_l3670_367092

theorem pitcher_juice_distribution (pitcher_capacity : ℝ) (num_cups : ℕ) :
  pitcher_capacity > 0 →
  num_cups = 8 →
  let juice_amount := pitcher_capacity / 2
  let juice_per_cup := juice_amount / num_cups
  juice_per_cup / pitcher_capacity = 1 / 16 := by
  sorry

end NUMINAMATH_CALUDE_pitcher_juice_distribution_l3670_367092


namespace NUMINAMATH_CALUDE_students_taking_one_subject_l3670_367088

theorem students_taking_one_subject (both : ℕ) (algebra : ℕ) (geometry_only : ℕ)
  (h1 : both = 16)
  (h2 : algebra = 36)
  (h3 : geometry_only = 15) :
  algebra - both + geometry_only = 35 := by
  sorry

end NUMINAMATH_CALUDE_students_taking_one_subject_l3670_367088


namespace NUMINAMATH_CALUDE_min_value_theorem_l3670_367049

theorem min_value_theorem (a b : ℝ) (ha : a > 0) (hb : b > 0) (h_sum : 2*a + b = 6) :
  (1/a + 2/b) ≥ 4/3 ∧ ∃ (a₀ b₀ : ℝ), a₀ > 0 ∧ b₀ > 0 ∧ 2*a₀ + b₀ = 6 ∧ 1/a₀ + 2/b₀ = 4/3 := by
sorry

end NUMINAMATH_CALUDE_min_value_theorem_l3670_367049


namespace NUMINAMATH_CALUDE_students_per_bus_l3670_367071

theorem students_per_bus (total_students : ℕ) (num_buses : ℕ) (students_in_cars : ℕ) :
  total_students = 375 →
  num_buses = 7 →
  students_in_cars = 4 →
  (total_students - students_in_cars) / num_buses = 53 := by
  sorry

end NUMINAMATH_CALUDE_students_per_bus_l3670_367071


namespace NUMINAMATH_CALUDE_smallest_n_for_inequality_l3670_367081

theorem smallest_n_for_inequality : ∃ (n : ℕ), n > 0 ∧ (1 - 1 / (2^n : ℚ) > 315 / 412) ∧ ∀ (m : ℕ), m > 0 ∧ m < n → 1 - 1 / (2^m : ℚ) ≤ 315 / 412 :=
by sorry

end NUMINAMATH_CALUDE_smallest_n_for_inequality_l3670_367081


namespace NUMINAMATH_CALUDE_smallest_factor_l3670_367043

theorem smallest_factor (w : ℕ) (other : ℕ) : 
  w = 144 →
  (∃ k : ℕ, w * other = k * 2^5) →
  (∃ k : ℕ, w * other = k * 3^3) →
  (∃ k : ℕ, w * other = k * 12^2) →
  (∀ x : ℕ, x < other → 
    (∃ k : ℕ, w * x = k * 2^5) ∧ 
    (∃ k : ℕ, w * x = k * 3^3) ∧ 
    (∃ k : ℕ, w * x = k * 12^2) → false) →
  other = 6 := by
sorry

end NUMINAMATH_CALUDE_smallest_factor_l3670_367043


namespace NUMINAMATH_CALUDE_black_squares_eaten_l3670_367062

/-- Represents a square on the chessboard -/
structure Square where
  row : Nat
  col : Nat

/-- Defines whether a square is black -/
def isBlack (s : Square) : Bool :=
  (s.row + s.col) % 2 = 0

/-- The list of squares eaten by termites -/
def eatenSquares : List Square := [
  ⟨3, 1⟩, ⟨4, 6⟩, ⟨3, 7⟩,
  ⟨4, 1⟩, ⟨2, 3⟩, ⟨2, 4⟩, ⟨4, 3⟩,
  ⟨3, 5⟩, ⟨3, 2⟩, ⟨4, 7⟩,
  ⟨3, 6⟩, ⟨2, 6⟩
]

/-- Counts the number of black squares in a list of squares -/
def countBlackSquares (squares : List Square) : Nat :=
  squares.filter isBlack |>.length

/-- Theorem stating that the number of black squares eaten is 12 -/
theorem black_squares_eaten :
  countBlackSquares eatenSquares = 12 := by
  sorry


end NUMINAMATH_CALUDE_black_squares_eaten_l3670_367062


namespace NUMINAMATH_CALUDE_initial_deposit_proof_l3670_367047

def bank_account (initial_deposit : ℝ) : ℝ := 
  ((initial_deposit * 1.1 + 10) * 1.1 + 10)

theorem initial_deposit_proof (initial_deposit : ℝ) : 
  bank_account initial_deposit = 142 → initial_deposit = 100 := by
sorry

end NUMINAMATH_CALUDE_initial_deposit_proof_l3670_367047


namespace NUMINAMATH_CALUDE_circle_trajectory_and_max_area_l3670_367084

-- Define the circles F₁ and F₂
def F₁ (x y : ℝ) : Prop := (x + 2)^2 + y^2 = 49
def F₂ (x y : ℝ) : Prop := (x - 2)^2 + y^2 = 1

-- Define the curve C
def C (x y : ℝ) : Prop := x^2 / 9 + y^2 / 5 = 1

-- Define the property of point Q
def Q_property (x y : ℝ) : Prop := C x y ∧ y ≠ 0

-- Define the line MN parallel to OQ and passing through F₂
def MN_parallel_OQ (m : ℝ) (x y : ℝ) : Prop := x = m * y + 2

-- Define the distinct intersection points M and N
def distinct_intersections (m : ℝ) : Prop :=
  ∃ x₁ y₁ x₂ y₂, x₁ ≠ x₂ ∧ y₁ ≠ y₂ ∧
  C x₁ y₁ ∧ C x₂ y₂ ∧
  MN_parallel_OQ m x₁ y₁ ∧ MN_parallel_OQ m x₂ y₂

-- Theorem statement
theorem circle_trajectory_and_max_area :
  (∀ x y, C x y → (∃ R, (∀ x' y', F₁ x' y' → (x - x')^2 + (y - y')^2 = (7 - R)^2) ∧
                      (∀ x' y', F₂ x' y' → (x - x')^2 + (y - y')^2 = (R - 1)^2))) ∧
  (∀ m, distinct_intersections m →
    ∃ x₃ y₃, Q_property x₃ y₃ ∧
    (∀ A, (∃ x₁ y₁ x₂ y₂, C x₁ y₁ ∧ C x₂ y₂ ∧ MN_parallel_OQ m x₁ y₁ ∧ MN_parallel_OQ m x₂ y₂ ∧
           A = (1/2) * abs ((x₂ - x₁) * (y₃ - y₁) - (y₂ - y₁) * (x₃ - x₁))) →
    A ≤ 10/3)) :=
sorry

end NUMINAMATH_CALUDE_circle_trajectory_and_max_area_l3670_367084


namespace NUMINAMATH_CALUDE_star_equation_roots_l3670_367034

-- Define the operation ※
def star (a b : ℝ) : ℝ := a^2 + a*b

-- Theorem statement
theorem star_equation_roots (m : ℝ) : 
  (∃ x y : ℝ, x ≠ y ∧ star x 3 = -m) → m = 2 :=
by
  sorry

end NUMINAMATH_CALUDE_star_equation_roots_l3670_367034


namespace NUMINAMATH_CALUDE_arithmetic_mean_squares_first_four_odd_numbers_l3670_367095

theorem arithmetic_mean_squares_first_four_odd_numbers : 
  let odd_numbers := [1, 3, 5, 7]
  let squares := List.map (λ x => x^2) odd_numbers
  (List.sum squares) / (List.length squares) = 21 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_mean_squares_first_four_odd_numbers_l3670_367095


namespace NUMINAMATH_CALUDE_power_multiplication_l3670_367027

theorem power_multiplication (t : ℝ) : t^5 * t^2 = t^7 := by
  sorry

end NUMINAMATH_CALUDE_power_multiplication_l3670_367027


namespace NUMINAMATH_CALUDE_radish_carrot_ratio_l3670_367040

theorem radish_carrot_ratio :
  let cucumbers : ℕ := 15
  let radishes : ℕ := 3 * cucumbers
  let carrots : ℕ := 9
  (radishes : ℚ) / carrots = 5 := by
  sorry

end NUMINAMATH_CALUDE_radish_carrot_ratio_l3670_367040


namespace NUMINAMATH_CALUDE_ron_ticket_sales_l3670_367004

/-- Proves that Ron sold 12 student tickets given the problem conditions -/
theorem ron_ticket_sales
  (student_price : ℝ)
  (adult_price : ℝ)
  (total_tickets : ℕ)
  (total_income : ℝ)
  (h1 : student_price = 2)
  (h2 : adult_price = 4.5)
  (h3 : total_tickets = 20)
  (h4 : total_income = 60)
  : ∃ (student_tickets : ℕ) (adult_tickets : ℕ),
    student_tickets + adult_tickets = total_tickets ∧
    student_price * student_tickets + adult_price * adult_tickets = total_income ∧
    student_tickets = 12 :=
by sorry

end NUMINAMATH_CALUDE_ron_ticket_sales_l3670_367004


namespace NUMINAMATH_CALUDE_worker_wage_problem_l3670_367016

theorem worker_wage_problem (ordinary_rate : ℝ) (overtime_rate : ℝ) (total_hours : ℕ) 
  (overtime_hours : ℕ) (total_earnings : ℝ) :
  overtime_rate = 0.90 →
  total_hours = 50 →
  overtime_hours = 8 →
  total_earnings = 32.40 →
  ordinary_rate * (total_hours - overtime_hours : ℝ) + overtime_rate * overtime_hours = total_earnings →
  ordinary_rate = 0.60 := by
sorry

end NUMINAMATH_CALUDE_worker_wage_problem_l3670_367016


namespace NUMINAMATH_CALUDE_patty_score_proof_l3670_367029

def june_score : ℝ := 97
def josh_score : ℝ := 100
def henry_score : ℝ := 94
def average_score : ℝ := 94

theorem patty_score_proof (patty_score : ℝ) : 
  (june_score + josh_score + henry_score + patty_score) / 4 = average_score →
  patty_score = 85 := by
sorry

end NUMINAMATH_CALUDE_patty_score_proof_l3670_367029


namespace NUMINAMATH_CALUDE_quadratic_minimum_l3670_367086

theorem quadratic_minimum (x : ℝ) : x^2 - 6*x + 13 ≥ 4 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_minimum_l3670_367086


namespace NUMINAMATH_CALUDE_chocolate_problem_l3670_367070

theorem chocolate_problem (C S : ℝ) (N : ℕ) :
  (N * C = 77 * S) →  -- The total cost price equals the selling price of 77 chocolates
  ((S - C) / C = 4 / 7) →  -- The gain percent is 4/7
  N = 121 := by  -- Prove that N (number of chocolates bought at cost price) is 121
sorry

end NUMINAMATH_CALUDE_chocolate_problem_l3670_367070


namespace NUMINAMATH_CALUDE_rental_miles_driven_l3670_367083

-- Define the rental parameters
def daily_rate : ℚ := 29
def mile_rate : ℚ := 0.08
def total_paid : ℚ := 46.12

-- Define the function to calculate miles driven
def miles_driven (daily_rate mile_rate total_paid : ℚ) : ℚ :=
  (total_paid - daily_rate) / mile_rate

-- Theorem statement
theorem rental_miles_driven :
  miles_driven daily_rate mile_rate total_paid = 214 := by
  sorry

end NUMINAMATH_CALUDE_rental_miles_driven_l3670_367083


namespace NUMINAMATH_CALUDE_last_digit_to_appear_is_zero_l3670_367033

def modifiedFibonacci : ℕ → ℕ
  | 0 => 2
  | 1 => 1
  | n + 2 => modifiedFibonacci (n + 1) + modifiedFibonacci n

def unitsDigit (n : ℕ) : ℕ := n % 10

def allDigitsAppeared (n : ℕ) : Prop :=
  ∀ d : ℕ, d < 10 → ∃ k : ℕ, k ≤ n ∧ unitsDigit (modifiedFibonacci k) = d

theorem last_digit_to_appear_is_zero :
  ∃ N : ℕ, allDigitsAppeared N ∧
    ¬(allDigitsAppeared (N - 1)) ∧
    unitsDigit (modifiedFibonacci N) = 0 :=
  sorry

end NUMINAMATH_CALUDE_last_digit_to_appear_is_zero_l3670_367033


namespace NUMINAMATH_CALUDE_P_positive_P_surjective_l3670_367035

/-- A polynomial in two real variables that takes only positive values and achieves all positive values -/
def P (x y : ℝ) : ℝ := x^2 + (x*y + 1)^2

/-- The polynomial P is always positive for any real x and y -/
theorem P_positive (x y : ℝ) : P x y > 0 := by sorry

/-- For any positive real t, there exist real x and y such that P(x,y) = t -/
theorem P_surjective (t : ℝ) (ht : t > 0) : ∃ x y : ℝ, P x y = t := by sorry

end NUMINAMATH_CALUDE_P_positive_P_surjective_l3670_367035


namespace NUMINAMATH_CALUDE_swimmers_pass_count_l3670_367038

/-- Represents a swimmer in the pool --/
structure Swimmer where
  speed : ℝ
  startPosition : ℝ

/-- Represents the swimming problem setup --/
structure SwimmingProblem where
  poolLength : ℝ
  swimmer1 : Swimmer
  swimmer2 : Swimmer
  totalTime : ℝ

/-- Calculates the number of times the swimmers pass each other --/
def countPasses (problem : SwimmingProblem) : ℕ :=
  sorry

/-- The main theorem to prove --/
theorem swimmers_pass_count (problem : SwimmingProblem) 
  (h1 : problem.poolLength = 120)
  (h2 : problem.swimmer1.speed = 4)
  (h3 : problem.swimmer2.speed = 3)
  (h4 : problem.swimmer1.startPosition = 0)
  (h5 : problem.swimmer2.startPosition = 120)
  (h6 : problem.totalTime = 15 * 60) : 
  countPasses problem = 29 := by
  sorry

end NUMINAMATH_CALUDE_swimmers_pass_count_l3670_367038


namespace NUMINAMATH_CALUDE_three_over_x_plus_one_is_fraction_l3670_367030

/-- A fraction is an expression where the denominator includes a variable. -/
def is_fraction (n d : ℝ → ℝ) : Prop :=
  ∃ x, d x ≠ d 0

/-- The expression 3/(x+1) is a fraction. -/
theorem three_over_x_plus_one_is_fraction :
  is_fraction (λ _ ↦ 3) (λ x ↦ x + 1) := by
sorry

end NUMINAMATH_CALUDE_three_over_x_plus_one_is_fraction_l3670_367030


namespace NUMINAMATH_CALUDE_cube_sum_divided_by_quadratic_minus_product_plus_square_l3670_367008

theorem cube_sum_divided_by_quadratic_minus_product_plus_square (a b : ℝ) :
  a = 6 ∧ b = 3 → (a^3 + b^3) / (a^2 - a*b + b^2) = 9 := by
  sorry

end NUMINAMATH_CALUDE_cube_sum_divided_by_quadratic_minus_product_plus_square_l3670_367008


namespace NUMINAMATH_CALUDE_geometry_propositions_l3670_367013

-- Define the types for planes and lines
def Plane : Type := Unit
def Line : Type := Unit

-- Define the operations
def perpendicular (l : Line) (p : Plane) : Prop := sorry
def parallel_planes (p1 p2 : Plane) : Prop := sorry
def parallel_lines (l1 l2 : Line) : Prop := sorry
def parallel_line_plane (l : Line) (p : Plane) : Prop := sorry
def intersect (p1 p2 : Plane) : Line := sorry
def contained_in (l : Line) (p : Plane) : Prop := sorry

-- The theorem
theorem geometry_propositions (α β : Plane) (m n : Line) :
  (∀ α β m, perpendicular m α → perpendicular m β → parallel_planes α β) ∧
  ¬(∀ α β m n, parallel_line_plane m α → intersect α β = n → parallel_lines m n) ∧
  (∀ α m n, parallel_lines m n → perpendicular m α → perpendicular n α) ∧
  (∀ α β m n, perpendicular m α → parallel_lines m n → contained_in n β → perpendicular α β) :=
by sorry

end NUMINAMATH_CALUDE_geometry_propositions_l3670_367013


namespace NUMINAMATH_CALUDE_centroid_dot_product_l3670_367015

/-- Triangle ABC with centroid G -/
structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ
  G : ℝ × ℝ

/-- Vector from point P to point Q -/
def vector (P Q : ℝ × ℝ) : ℝ × ℝ := (Q.1 - P.1, Q.2 - P.2)

/-- Dot product of two 2D vectors -/
def dot_product (v w : ℝ × ℝ) : ℝ := v.1 * w.1 + v.2 * w.2

/-- Squared distance between two points -/
def distance_squared (P Q : ℝ × ℝ) : ℝ := (Q.1 - P.1)^2 + (Q.2 - P.2)^2

theorem centroid_dot_product (t : Triangle) : 
  (distance_squared t.A t.B = 1) →
  (distance_squared t.B t.C = 2) →
  (distance_squared t.A t.C = 3) →
  (t.G = ((t.A.1 + t.B.1 + t.C.1) / 3, (t.A.2 + t.B.2 + t.C.2) / 3)) →
  (dot_product (vector t.A t.G) (vector t.A t.C) = 4/3) := by
  sorry

end NUMINAMATH_CALUDE_centroid_dot_product_l3670_367015


namespace NUMINAMATH_CALUDE_sqrt_ratio_simplification_l3670_367076

theorem sqrt_ratio_simplification :
  (Real.sqrt (8^2 + 15^2)) / (Real.sqrt (49 + 64)) = 17 / Real.sqrt 113 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_ratio_simplification_l3670_367076


namespace NUMINAMATH_CALUDE_floor_equation_natural_numbers_l3670_367068

theorem floor_equation_natural_numbers (a b : ℕ) :
  (a ≠ 0 ∧ b ≠ 0) →
  (Int.floor (a^2 / b : ℚ) + Int.floor (b^2 / a : ℚ) = 
   Int.floor ((a^2 + b^2) / (a * b) : ℚ) + a * b) ↔ 
  (b = a^2 + 1 ∨ a = b^2 + 1) :=
sorry

end NUMINAMATH_CALUDE_floor_equation_natural_numbers_l3670_367068


namespace NUMINAMATH_CALUDE_circle_distance_extrema_l3670_367067

-- Define the circle C
def C (x y : ℝ) : Prop := (x - 3)^2 + (y - 4)^2 = 1

-- Define points A and B
def A : ℝ × ℝ := (-1, 0)
def B : ℝ × ℝ := (1, 0)

-- Define the distance function d
def d (P : ℝ × ℝ) : ℝ :=
  let (x, y) := P
  (x + 1)^2 + y^2 + (x - 1)^2 + y^2

-- Theorem statement
theorem circle_distance_extrema :
  (∃ P : ℝ × ℝ, C P.1 P.2 ∧ ∀ Q : ℝ × ℝ, C Q.1 Q.2 → d P ≥ d Q) ∧
  (∃ P : ℝ × ℝ, C P.1 P.2 ∧ ∀ Q : ℝ × ℝ, C Q.1 Q.2 → d P ≤ d Q) ∧
  (∀ P : ℝ × ℝ, C P.1 P.2 → d P ≤ 74) ∧
  (∀ P : ℝ × ℝ, C P.1 P.2 → d P ≥ 34) :=
sorry

end NUMINAMATH_CALUDE_circle_distance_extrema_l3670_367067


namespace NUMINAMATH_CALUDE_supplement_of_angle_with_complement_50_l3670_367099

def angle_with_complement_50 (θ : ℝ) : Prop :=
  90 - θ = 50

theorem supplement_of_angle_with_complement_50 (θ : ℝ) 
  (h : angle_with_complement_50 θ) : 180 - θ = 140 := by
  sorry

end NUMINAMATH_CALUDE_supplement_of_angle_with_complement_50_l3670_367099


namespace NUMINAMATH_CALUDE_remaining_clothing_problem_l3670_367000

/-- The number of remaining pieces of clothing to fold -/
def remaining_clothing (total_shirts : ℕ) (total_shorts : ℕ) (folded_shirts : ℕ) (folded_shorts : ℕ) : ℕ :=
  (total_shirts - folded_shirts) + (total_shorts - folded_shorts)

/-- Theorem stating that given 20 shirts and 8 pairs of shorts, if 12 shirts and 5 shorts are folded,
    the remaining number of pieces of clothing to fold is 11. -/
theorem remaining_clothing_problem :
  remaining_clothing 20 8 12 5 = 11 := by
  sorry

end NUMINAMATH_CALUDE_remaining_clothing_problem_l3670_367000


namespace NUMINAMATH_CALUDE_a_gt_1_sufficient_not_necessary_for_a_sq_gt_a_l3670_367060

theorem a_gt_1_sufficient_not_necessary_for_a_sq_gt_a :
  (∀ a : ℝ, a > 1 → a^2 > a) ∧
  (∃ a : ℝ, a^2 > a ∧ a ≤ 1) := by
  sorry

end NUMINAMATH_CALUDE_a_gt_1_sufficient_not_necessary_for_a_sq_gt_a_l3670_367060


namespace NUMINAMATH_CALUDE_angle_inequality_l3670_367036

theorem angle_inequality (θ : Real) (h1 : 0 ≤ θ) (h2 : θ ≤ 2 * Real.pi) :
  (∀ x : Real, 0 ≤ x ∧ x ≤ 1 →
    x^2 * Real.cos θ - x * (1 - x) * Real.tan θ + (1 - x)^2 * Real.sin θ > 0) ↔
  (0 < θ ∧ θ < Real.pi / 2) :=
by sorry

end NUMINAMATH_CALUDE_angle_inequality_l3670_367036


namespace NUMINAMATH_CALUDE_simplify_expression_l3670_367058

theorem simplify_expression (a b c : ℝ) : a - (b - c) = a - b + c := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l3670_367058


namespace NUMINAMATH_CALUDE_sqrt_sum_equality_l3670_367089

theorem sqrt_sum_equality : 
  Real.sqrt 2 + Real.sqrt (2 + 4) + Real.sqrt (2 + 4 + 6) + Real.sqrt (2 + 4 + 6 + 8) = 
  Real.sqrt 2 + Real.sqrt 6 + 2 * Real.sqrt 3 + 2 * Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_sum_equality_l3670_367089


namespace NUMINAMATH_CALUDE_obtuse_triangle_proof_l3670_367005

theorem obtuse_triangle_proof (α : Real) (h1 : 0 < α ∧ α < π) 
  (h2 : Real.sin α + Real.cos α = 2/3) : π/2 < α := by
  sorry

end NUMINAMATH_CALUDE_obtuse_triangle_proof_l3670_367005


namespace NUMINAMATH_CALUDE_polynomial_equality_l3670_367048

theorem polynomial_equality (x : ℝ) : 
  let g : ℝ → ℝ := λ x => -2*x^5 + 4*x^4 - 12*x^3 + 2*x^2 + 4*x + 4
  2*x^5 + 3*x^3 - 4*x + 1 + g x = 4*x^4 - 9*x^3 + 2*x^2 + 5 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_equality_l3670_367048


namespace NUMINAMATH_CALUDE_fraction_simplification_l3670_367078

theorem fraction_simplification (x : ℝ) : (x + 3) / 4 - (5 - 2*x) / 3 = (11*x - 11) / 12 := by
  sorry

end NUMINAMATH_CALUDE_fraction_simplification_l3670_367078


namespace NUMINAMATH_CALUDE_product_mod_23_l3670_367024

theorem product_mod_23 : (2011 * 2012 * 2013 * 2014 * 2015) % 23 = 5 := by
  sorry

end NUMINAMATH_CALUDE_product_mod_23_l3670_367024


namespace NUMINAMATH_CALUDE_tan_45_degrees_l3670_367080

theorem tan_45_degrees : Real.tan (π / 4) = 1 := by
  sorry

end NUMINAMATH_CALUDE_tan_45_degrees_l3670_367080


namespace NUMINAMATH_CALUDE_probability_two_in_same_group_l3670_367025

/-- The probability of two specific individuals being in the same group when dividing 4 individuals into two equal groups -/
def probability_same_group : ℚ := 1 / 3

/-- The number of ways to divide 4 individuals into two equal groups -/
def total_ways : ℕ := 3

/-- The number of ways to have two specific individuals in the same group when dividing 4 individuals into two equal groups -/
def favorable_ways : ℕ := 1

theorem probability_two_in_same_group :
  probability_same_group = favorable_ways / total_ways := by
  sorry

#eval probability_same_group

end NUMINAMATH_CALUDE_probability_two_in_same_group_l3670_367025


namespace NUMINAMATH_CALUDE_log_calculation_l3670_367093

theorem log_calculation : Real.log 25 / Real.log 10 + 
  (Real.log 2 / Real.log 10) * (Real.log 50 / Real.log 10) + 
  (Real.log 2 / Real.log 10)^2 = 2 := by
  sorry

end NUMINAMATH_CALUDE_log_calculation_l3670_367093


namespace NUMINAMATH_CALUDE_parabola_line_intersection_l3670_367011

/-- Parabola structure -/
structure Parabola where
  p : ℝ
  h : p > 0

/-- Line structure -/
structure Line where
  m : ℝ
  b : ℝ

/-- Circle structure -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- Theorem statement -/
theorem parabola_line_intersection (C : Parabola) (l : Line) (M N : ℝ × ℝ) :
  l.m = -Real.sqrt 3 ∧ l.b = Real.sqrt 3 →  -- Line equation: y = -√3(x-1)
  (C.p / 2, 0) ∈ {(x, y) | y = l.m * x + l.b} →  -- Line passes through focus
  M ∈ {(x, y) | y^2 = 2 * C.p * x} ∧ N ∈ {(x, y) | y^2 = 2 * C.p * x} →  -- M and N on parabola
  M ∈ {(x, y) | y = l.m * x + l.b} ∧ N ∈ {(x, y) | y = l.m * x + l.b} →  -- M and N on line
  ∃ (circ : Circle), circ.center = ((M.1 + N.1) / 2, (M.2 + N.2) / 2) ∧
                     circ.radius = Real.sqrt ((M.1 - N.1)^2 + (M.2 - N.2)^2) / 2 →
  C.p = 2 ∧  -- First conclusion
  circ.center.1 - circ.radius = -C.p / 2  -- Second conclusion: circle tangent to directrix
  := by sorry

end NUMINAMATH_CALUDE_parabola_line_intersection_l3670_367011


namespace NUMINAMATH_CALUDE_car_payment_remainder_l3670_367069

theorem car_payment_remainder (part_payment : ℝ) (percentage : ℝ) (total_cost : ℝ) (remainder : ℝ) : 
  part_payment = 300 →
  percentage = 5 →
  part_payment = percentage / 100 * total_cost →
  remainder = total_cost - part_payment →
  remainder = 5700 := by
sorry

end NUMINAMATH_CALUDE_car_payment_remainder_l3670_367069


namespace NUMINAMATH_CALUDE_matchstick_ratio_is_half_l3670_367028

/-- The ratio of matchsticks used to matchsticks originally had -/
def matchstick_ratio (houses : ℕ) (sticks_per_house : ℕ) (original_sticks : ℕ) : ℚ :=
  (houses * sticks_per_house : ℚ) / original_sticks

/-- Proof that the ratio of matchsticks used to matchsticks originally had is 1/2 -/
theorem matchstick_ratio_is_half :
  matchstick_ratio 30 10 600 = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_matchstick_ratio_is_half_l3670_367028


namespace NUMINAMATH_CALUDE_total_subjects_l3670_367079

theorem total_subjects (average_all : ℝ) (average_five : ℝ) (last_subject : ℝ) 
  (h1 : average_all = 79)
  (h2 : average_five = 74)
  (h3 : last_subject = 104) : 
  ∃ n : ℕ, n = 6 ∧ n * average_all = 5 * average_five + last_subject :=
by sorry

end NUMINAMATH_CALUDE_total_subjects_l3670_367079


namespace NUMINAMATH_CALUDE_art_gallery_sculpture_fraction_l3670_367094

theorem art_gallery_sculpture_fraction 
  (total : ℕ) 
  (displayed : ℕ) 
  (sculptures_not_displayed : ℕ) 
  (h1 : displayed = total / 3)
  (h2 : sculptures_not_displayed = 800)
  (h3 : total = 1800)
  (h4 : (total - displayed) / 3 = total - displayed - sculptures_not_displayed) :
  3 * (sculptures_not_displayed + displayed - total + sculptures_not_displayed) = 2 * displayed := by
  sorry

end NUMINAMATH_CALUDE_art_gallery_sculpture_fraction_l3670_367094


namespace NUMINAMATH_CALUDE_field_dimensions_l3670_367041

/-- Proves that for a rectangular field with given dimensions, if the area is 92, then m = 4 -/
theorem field_dimensions (m : ℝ) : 
  (3*m + 6) * (m - 3) = 92 → m = 4 := by
sorry

end NUMINAMATH_CALUDE_field_dimensions_l3670_367041


namespace NUMINAMATH_CALUDE_triangle_inequality_l3670_367026

theorem triangle_inequality (A B C : ℝ) (h_triangle : A + B + C = π) : 
  (Real.sqrt (Real.sin A * Real.sin B) / Real.sin (C / 2)) + 
  (Real.sqrt (Real.sin B * Real.sin C) / Real.sin (A / 2)) + 
  (Real.sqrt (Real.sin C * Real.sin A) / Real.sin (B / 2)) ≥ 3 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_triangle_inequality_l3670_367026


namespace NUMINAMATH_CALUDE_first_quarter_2016_has_91_days_l3670_367007

/-- The number of days in the first quarter of 2016 -/
def first_quarter_days_2016 : ℕ :=
  let year := 2016
  let is_leap_year := year % 4 = 0
  let february_days := if is_leap_year then 29 else 28
  let january_days := 31
  let march_days := 31
  january_days + february_days + march_days

/-- Theorem stating that the first quarter of 2016 has 91 days -/
theorem first_quarter_2016_has_91_days :
  first_quarter_days_2016 = 91 := by
  sorry

end NUMINAMATH_CALUDE_first_quarter_2016_has_91_days_l3670_367007


namespace NUMINAMATH_CALUDE_polynomial_coefficient_sum_l3670_367006

theorem polynomial_coefficient_sum (b₁ b₂ b₃ c₁ c₂ : ℝ) :
  (∀ x : ℝ, x^7 - x^6 + x^5 - x^4 + x^3 - x^2 + x - 1 = 
    (x^2 + b₁*x + c₁) * (x^2 + b₂*x + c₂) * (x^2 + b₃*x + 1)) →
  b₁*c₁ + b₂*c₂ + b₃ = -1 := by
sorry

end NUMINAMATH_CALUDE_polynomial_coefficient_sum_l3670_367006


namespace NUMINAMATH_CALUDE_rectangular_plot_poles_l3670_367082

/-- Calculate the number of poles needed for a rectangular fence --/
def poles_needed (length width long_spacing short_spacing : ℕ) : ℕ :=
  let long_poles := (length / long_spacing + 1) * 2
  let short_poles := (width / short_spacing + 1) * 2
  long_poles + short_poles - 4

/-- Theorem: The number of poles needed for the given rectangular plot is 70 --/
theorem rectangular_plot_poles :
  poles_needed 90 70 4 5 = 70 := by
  sorry

end NUMINAMATH_CALUDE_rectangular_plot_poles_l3670_367082


namespace NUMINAMATH_CALUDE_union_of_sets_l3670_367077

/-- Given sets A and B with specific properties, prove their union -/
theorem union_of_sets (a : ℝ) : 
  let A : Set ℝ := {|a + 1|, 3, 5}
  let B : Set ℝ := {2*a + 1, a^(2*a + 2), a^2 + 2*a - 1}
  (A ∩ B = {2, 3}) → (A ∪ B = {1, 2, 3, 5}) := by
  sorry


end NUMINAMATH_CALUDE_union_of_sets_l3670_367077


namespace NUMINAMATH_CALUDE_sqrt_four_fourth_powers_l3670_367053

theorem sqrt_four_fourth_powers : Real.sqrt (4^4 + 4^4 + 4^4 + 4^4) = 32 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_four_fourth_powers_l3670_367053


namespace NUMINAMATH_CALUDE_translation_result_l3670_367097

/-- Represents a point in 2D Cartesian coordinate system -/
structure Point where
  x : ℝ
  y : ℝ

/-- Translates a point horizontally by a given distance -/
def translate_x (p : Point) (dx : ℝ) : Point :=
  { x := p.x + dx, y := p.y }

/-- The initial point P -/
def P : Point :=
  { x := -2, y := 4 }

/-- The translation distance to the right -/
def translation_distance : ℝ := 1

theorem translation_result :
  translate_x P translation_distance = { x := -1, y := 4 } := by
  sorry

end NUMINAMATH_CALUDE_translation_result_l3670_367097


namespace NUMINAMATH_CALUDE_probability_is_one_eighth_l3670_367054

/-- A standard die with 8 sides -/
def StandardDie : Finset ℕ := Finset.range 8

/-- The set of all possible outcomes when rolling the die twice -/
def AllOutcomes : Finset (ℕ × ℕ) := StandardDie.product StandardDie

/-- The set of favorable outcomes (pairs that differ by 3) -/
def FavorableOutcomes : Finset (ℕ × ℕ) :=
  AllOutcomes.filter (fun p => (p.1 + 3 = p.2) ∨ (p.2 + 3 = p.1))

/-- The probability of rolling two integers that differ by 3 -/
def probability : ℚ := (FavorableOutcomes.card : ℚ) / (AllOutcomes.card : ℚ)

theorem probability_is_one_eighth :
  probability = 1 / 8 := by sorry

end NUMINAMATH_CALUDE_probability_is_one_eighth_l3670_367054


namespace NUMINAMATH_CALUDE_systematic_sample_smallest_number_l3670_367019

/-- Systematic sampling function -/
def systematicSample (n : ℕ) (k : ℕ) (i : ℕ) : ℕ := i * k

/-- Proposition: In a systematic sample of size 5 from 80 products, if 42 is in the sample, 
    then the smallest number in the sample is 10 -/
theorem systematic_sample_smallest_number :
  ∀ (i : ℕ), i < 5 →
  systematicSample 80 5 i = 42 →
  (∀ (j : ℕ), j < 5 → systematicSample 80 5 j ≥ 10) ∧
  (∃ (j : ℕ), j < 5 ∧ systematicSample 80 5 j = 10) :=
by sorry

end NUMINAMATH_CALUDE_systematic_sample_smallest_number_l3670_367019


namespace NUMINAMATH_CALUDE_max_distance_from_origin_l3670_367056

theorem max_distance_from_origin (x y : ℝ) :
  x^2 + y^2 - 4*x - 4*y + 6 = 0 →
  ∃ (max_val : ℝ), max_val = 3 * Real.sqrt 2 ∧
    ∀ (x' y' : ℝ), x'^2 + y'^2 - 4*x' - 4*y' + 6 = 0 →
      Real.sqrt (x'^2 + y'^2) ≤ max_val :=
by sorry

end NUMINAMATH_CALUDE_max_distance_from_origin_l3670_367056


namespace NUMINAMATH_CALUDE_functional_equation_solution_l3670_367046

theorem functional_equation_solution (f : ℚ → ℚ) :
  (∀ x y : ℚ, f x + f y = 2 * f ((x + y) / 2)) →
  ∃ c : ℚ, ∀ x : ℚ, f x = c * x :=
by sorry

end NUMINAMATH_CALUDE_functional_equation_solution_l3670_367046


namespace NUMINAMATH_CALUDE_arc_length_for_specific_circle_l3670_367090

/-- Given a circle with radius π and a central angle of 120°, the arc length is (2π²)/3 -/
theorem arc_length_for_specific_circle :
  let r : ℝ := Real.pi
  let θ : ℝ := 120
  let l : ℝ := (θ / 180) * Real.pi * r
  l = (2 * Real.pi^2) / 3 := by sorry

end NUMINAMATH_CALUDE_arc_length_for_specific_circle_l3670_367090


namespace NUMINAMATH_CALUDE_trivia_team_groups_l3670_367045

/-- Given a total number of students, number of students not picked, and number of students per group,
    calculate the number of groups formed. -/
def calculate_groups (total : ℕ) (not_picked : ℕ) (per_group : ℕ) : ℕ :=
  (total - not_picked) / per_group

/-- Theorem stating that with 65 total students, 17 not picked, and 6 per group, 8 groups are formed. -/
theorem trivia_team_groups : calculate_groups 65 17 6 = 8 := by
  sorry

end NUMINAMATH_CALUDE_trivia_team_groups_l3670_367045


namespace NUMINAMATH_CALUDE_inequality_theorem_l3670_367055

theorem inequality_theorem (x y : ℝ) : 
  x^2 + y^2 + 1 ≥ 2*(x*y - x + y) ∧ 
  (x^2 + y^2 + 1 = 2*(x*y - x + y) ↔ x = y - 1) := by
  sorry

end NUMINAMATH_CALUDE_inequality_theorem_l3670_367055


namespace NUMINAMATH_CALUDE_partition_positive_integers_l3670_367051

theorem partition_positive_integers : ∃ (A B : Set ℕ), 
  (∀ n : ℕ, n > 0 → (n ∈ A ∨ n ∈ B)) ∧ 
  (A ∩ B = ∅) ∧
  (∀ a b c : ℕ, a ∈ A → b ∈ A → c ∈ A → a < b → b < c → b - a ≠ c - b) ∧
  (∀ f : ℕ → ℕ, (∀ n : ℕ, f n ∈ B) → 
    ∃ i j k : ℕ, i < j ∧ j < k ∧ f j - f i ≠ f k - f j) :=
sorry

end NUMINAMATH_CALUDE_partition_positive_integers_l3670_367051


namespace NUMINAMATH_CALUDE_abs_neg_two_neg_two_pow_zero_l3670_367064

-- Prove that the absolute value of -2 is equal to 2
theorem abs_neg_two : |(-2 : ℤ)| = 2 := by sorry

-- Prove that -2 raised to the power of 0 is equal to 1
theorem neg_two_pow_zero : (-2 : ℤ) ^ (0 : ℕ) = 1 := by sorry

end NUMINAMATH_CALUDE_abs_neg_two_neg_two_pow_zero_l3670_367064


namespace NUMINAMATH_CALUDE_third_candidate_votes_l3670_367096

theorem third_candidate_votes :
  ∀ (total_votes : ℕ) (winning_votes second_votes third_votes : ℕ),
    winning_votes = 11628 →
    second_votes = 7636 →
    winning_votes = (49.69230769230769 / 100 : ℚ) * total_votes →
    total_votes = winning_votes + second_votes + third_votes →
    third_votes = 4136 := by
  sorry

end NUMINAMATH_CALUDE_third_candidate_votes_l3670_367096


namespace NUMINAMATH_CALUDE_min_value_of_function_l3670_367052

theorem min_value_of_function (x : ℝ) (h1 : 0 < x) (h2 : x < 1) :
  (∀ y : ℝ, y > 0 ∧ y < 1 → (4 / x + 1 / (1 - x)) ≤ (4 / y + 1 / (1 - y))) ∧
  (∃ z : ℝ, z > 0 ∧ z < 1 ∧ 4 / z + 1 / (1 - z) = 9) :=
sorry

end NUMINAMATH_CALUDE_min_value_of_function_l3670_367052


namespace NUMINAMATH_CALUDE_rubber_band_area_l3670_367022

/-- Represents a nail on the board -/
structure Nail where
  x : ℝ
  y : ℝ

/-- Represents the quadrilateral formed by the rubber band -/
structure Quadrilateral where
  nails : Fin 4 → Nail

/-- The area of a quadrilateral formed by a rubber band looped around four nails arranged in a 2x2 grid with 1 unit spacing -/
def quadrilateralArea (q : Quadrilateral) : ℝ :=
  sorry

/-- The theorem stating that the area of the quadrilateral is 6 square units -/
theorem rubber_band_area (q : Quadrilateral) 
  (h1 : q.nails 0 = ⟨0, 0⟩)
  (h2 : q.nails 1 = ⟨1, 0⟩)
  (h3 : q.nails 2 = ⟨0, 1⟩)
  (h4 : q.nails 3 = ⟨1, 1⟩) :
  quadrilateralArea q = 6 :=
sorry

end NUMINAMATH_CALUDE_rubber_band_area_l3670_367022


namespace NUMINAMATH_CALUDE_final_price_after_discounts_l3670_367032

def original_price : Float := 49.99
def first_discount_rate : Float := 0.10
def second_discount_rate : Float := 0.20

theorem final_price_after_discounts :
  let price_after_first_discount := original_price * (1 - first_discount_rate)
  let final_price := price_after_first_discount * (1 - second_discount_rate)
  final_price = 36.00 := by
  sorry

end NUMINAMATH_CALUDE_final_price_after_discounts_l3670_367032


namespace NUMINAMATH_CALUDE_convex_broken_line_in_triangle_l3670_367063

/-- A convex broken line in 2D space -/
structure ConvexBrokenLine where
  points : List (Real × Real)
  is_convex : sorry
  length : Real

/-- An equilateral triangle in 2D space -/
structure EquilateralTriangle where
  center : Real × Real
  side_length : Real

/-- A function to check if a broken line is enclosed within a triangle -/
def is_enclosed (line : ConvexBrokenLine) (triangle : EquilateralTriangle) : Prop :=
  sorry

theorem convex_broken_line_in_triangle 
  (line : ConvexBrokenLine) 
  (triangle : EquilateralTriangle) : 
  line.length = 1 → 
  triangle.side_length = 1 → 
  is_enclosed line triangle :=
sorry

end NUMINAMATH_CALUDE_convex_broken_line_in_triangle_l3670_367063


namespace NUMINAMATH_CALUDE_bottle_cap_collection_l3670_367057

/-- Given that 7 bottle caps weigh one ounce and a collection of bottle caps weighs 18 pounds,
    prove that the number of bottle caps in the collection is 2016. -/
theorem bottle_cap_collection (caps_per_ounce : ℕ) (collection_weight_pounds : ℕ) 
  (h1 : caps_per_ounce = 7)
  (h2 : collection_weight_pounds = 18) :
  caps_per_ounce * (collection_weight_pounds * 16) = 2016 := by
  sorry

#check bottle_cap_collection

end NUMINAMATH_CALUDE_bottle_cap_collection_l3670_367057


namespace NUMINAMATH_CALUDE_arithmetic_mean_problem_l3670_367091

theorem arithmetic_mean_problem (x : ℝ) : 
  ((x + 10) + 20 + 3*x + 16 + (3*x + 6)) / 5 = 30 → x = 14 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_mean_problem_l3670_367091


namespace NUMINAMATH_CALUDE_donna_card_shop_days_l3670_367042

/-- Represents Donna's work schedule and earnings --/
structure DonnaWork where
  dog_walking_hours : ℕ
  dog_walking_rate : ℚ
  card_shop_hours : ℕ
  card_shop_rate : ℚ
  babysitting_hours : ℕ
  babysitting_rate : ℚ
  total_earnings : ℚ
  total_days : ℕ

/-- Calculates the number of days Donna worked at the card shop --/
def card_shop_days (work : DonnaWork) : ℚ :=
  let dog_walking_earnings := ↑work.dog_walking_hours * work.dog_walking_rate * ↑work.total_days
  let babysitting_earnings := ↑work.babysitting_hours * work.babysitting_rate
  let card_shop_earnings := work.total_earnings - dog_walking_earnings - babysitting_earnings
  card_shop_earnings / (↑work.card_shop_hours * work.card_shop_rate)

/-- Theorem stating that Donna worked 5 days at the card shop --/
theorem donna_card_shop_days :
  ∀ (work : DonnaWork),
  work.dog_walking_hours = 2 ∧
  work.dog_walking_rate = 10 ∧
  work.card_shop_hours = 2 ∧
  work.card_shop_rate = 25/2 ∧
  work.babysitting_hours = 4 ∧
  work.babysitting_rate = 10 ∧
  work.total_earnings = 305 ∧
  work.total_days = 7 →
  card_shop_days work = 5 := by
  sorry


end NUMINAMATH_CALUDE_donna_card_shop_days_l3670_367042


namespace NUMINAMATH_CALUDE_fraction_deviation_from_sqrt_l3670_367044

theorem fraction_deviation_from_sqrt (x : ℝ) (h : 1 ≤ x ∧ x ≤ 9) : 
  |Real.sqrt x - (6 * x + 6) / (x + 11)| < 0.05 := by
  sorry

end NUMINAMATH_CALUDE_fraction_deviation_from_sqrt_l3670_367044


namespace NUMINAMATH_CALUDE_australians_in_group_l3670_367023

theorem australians_in_group (total : Nat) (chinese : Nat) (americans : Nat) 
  (h1 : total = 49)
  (h2 : chinese = 22)
  (h3 : americans = 16) :
  total - (chinese + americans) = 11 := by
  sorry

end NUMINAMATH_CALUDE_australians_in_group_l3670_367023


namespace NUMINAMATH_CALUDE_odd_function_a_value_l3670_367039

-- Define the function f
noncomputable def f (a : ℝ) : ℝ → ℝ := λ x =>
  if x > 0 then 1 + a^x else -1 - a^(-x)

-- State the theorem
theorem odd_function_a_value :
  ∀ a : ℝ,
  a > 0 →
  a ≠ 1 →
  (∀ x : ℝ, f a (-x) = -(f a x)) →
  f a (-1) = -3/2 →
  a = 1/2 :=
by
  sorry

end NUMINAMATH_CALUDE_odd_function_a_value_l3670_367039


namespace NUMINAMATH_CALUDE_effective_distance_is_seven_l3670_367020

/-- Calculates the effective distance walked given a constant walking rate, wind resistance reduction, and walking duration. -/
def effective_distance_walked (rate : ℝ) (wind_resistance : ℝ) (duration : ℝ) : ℝ :=
  (rate - wind_resistance) * duration

/-- Proves that given the specified conditions, the effective distance walked is 7 miles. -/
theorem effective_distance_is_seven :
  let rate : ℝ := 4
  let wind_resistance : ℝ := 0.5
  let duration : ℝ := 2
  effective_distance_walked rate wind_resistance duration = 7 := by
sorry

end NUMINAMATH_CALUDE_effective_distance_is_seven_l3670_367020


namespace NUMINAMATH_CALUDE_one_minus_repeating_thirds_l3670_367017

/-- The decimal 0.333... (repeating 3) -/
def repeating_thirds : ℚ :=
  1 / 3

theorem one_minus_repeating_thirds :
  1 - repeating_thirds = 2 / 3 := by
  sorry

end NUMINAMATH_CALUDE_one_minus_repeating_thirds_l3670_367017


namespace NUMINAMATH_CALUDE_arithmetic_mean_of_first_four_primes_reciprocals_l3670_367098

theorem arithmetic_mean_of_first_four_primes_reciprocals :
  let first_four_primes := [2, 3, 5, 7]
  let reciprocals := first_four_primes.map (λ x => 1 / x)
  (reciprocals.sum / reciprocals.length : ℚ) = 247 / 840 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_mean_of_first_four_primes_reciprocals_l3670_367098


namespace NUMINAMATH_CALUDE_trigonometric_identities_l3670_367085

theorem trigonometric_identities (θ : Real) (h : Real.tan θ = 2) :
  (Real.sin (π - θ) + Real.cos (θ - π)) / (Real.sin (θ + π) + Real.cos (θ + π)) = -1/3 ∧
  Real.sin (2 * θ) = 4/5 := by
  sorry

end NUMINAMATH_CALUDE_trigonometric_identities_l3670_367085


namespace NUMINAMATH_CALUDE_divisor_problem_l3670_367066

theorem divisor_problem (n d : ℕ) : 
  (n % d = 3) → (n^2 % d = 3) → d = 6 := by sorry

end NUMINAMATH_CALUDE_divisor_problem_l3670_367066


namespace NUMINAMATH_CALUDE_meaningful_expression_l3670_367087

theorem meaningful_expression (x : ℝ) : 
  (∃ y : ℝ, y = x / Real.sqrt (4 - x)) ↔ x < 4 :=
by sorry

end NUMINAMATH_CALUDE_meaningful_expression_l3670_367087


namespace NUMINAMATH_CALUDE_place_balls_count_l3670_367003

/-- The number of ways to place six numbered balls into six numbered boxes --/
def place_balls : ℕ :=
  let n : ℕ := 6  -- number of balls and boxes
  let k : ℕ := 2  -- number of balls placed in boxes with the same number
  let choose_two : ℕ := n.choose k
  let derangement_four : ℕ := 8  -- number of valid derangements for remaining 4 balls
  choose_two * derangement_four

/-- Theorem stating that the number of ways to place the balls is 120 --/
theorem place_balls_count : place_balls = 120 := by
  sorry

end NUMINAMATH_CALUDE_place_balls_count_l3670_367003


namespace NUMINAMATH_CALUDE_vector_simplification_l3670_367021

-- Define the Euclidean space
variable {E : Type*} [NormedAddCommGroup E] [InnerProductSpace ℝ E]

-- Define points in the Euclidean space
variable (A B C D : E)

-- Define vectors as differences between points
def vector (P Q : E) : E := Q - P

-- State the theorem
theorem vector_simplification (A B C D : E) :
  vector A B + vector B C - vector A D = vector D C := by sorry

end NUMINAMATH_CALUDE_vector_simplification_l3670_367021


namespace NUMINAMATH_CALUDE_inequality_range_l3670_367009

theorem inequality_range (m : ℝ) : 
  (∀ x : ℝ, 3 * x^2 + 1 ≥ m * x * (x - 1)) → 
  -6 ≤ m ∧ m ≤ 2 := by
sorry

end NUMINAMATH_CALUDE_inequality_range_l3670_367009


namespace NUMINAMATH_CALUDE_apple_cost_l3670_367073

/-- Proves that the cost of each apple is 4 dollars given the conditions -/
theorem apple_cost (total_money : ℕ) (kids : ℕ) (apples_per_kid : ℕ) :
  total_money = 360 →
  kids = 18 →
  apples_per_kid = 5 →
  total_money / (kids * apples_per_kid) = 4 := by
sorry

end NUMINAMATH_CALUDE_apple_cost_l3670_367073


namespace NUMINAMATH_CALUDE_solve_flower_problem_l3670_367031

def flower_problem (initial_flowers : ℕ) (flowers_per_bouquet : ℕ) (bouquets_after_wilting : ℕ) : Prop :=
  let remaining_flowers := bouquets_after_wilting * flowers_per_bouquet
  let wilted_flowers := initial_flowers - remaining_flowers
  wilted_flowers = 35

theorem solve_flower_problem :
  flower_problem 45 5 2 :=
by
  sorry

end NUMINAMATH_CALUDE_solve_flower_problem_l3670_367031


namespace NUMINAMATH_CALUDE_unique_p_type_prime_l3670_367014

/-- A prime number q is a P-type prime if q + 1 is a perfect square. -/
def is_p_type_prime (q : ℕ) : Prop :=
  Nat.Prime q ∧ ∃ m : ℕ, q + 1 = m^2

/-- There exists exactly one P-type prime number. -/
theorem unique_p_type_prime : ∃! q : ℕ, is_p_type_prime q :=
sorry

end NUMINAMATH_CALUDE_unique_p_type_prime_l3670_367014


namespace NUMINAMATH_CALUDE_shelter_dogs_count_l3670_367074

theorem shelter_dogs_count (dogs cats : ℕ) : 
  (dogs : ℚ) / cats = 15 / 7 →
  dogs / (cats + 8) = 15 / 11 →
  dogs = 30 := by
sorry

end NUMINAMATH_CALUDE_shelter_dogs_count_l3670_367074


namespace NUMINAMATH_CALUDE_sheila_hourly_wage_l3670_367037

/-- Represents Sheila's work schedule and earnings --/
structure WorkSchedule where
  hours_mon_wed_fri : ℕ
  hours_tue_thu : ℕ
  weekly_earnings : ℕ

/-- Calculates the total hours worked in a week --/
def total_hours (schedule : WorkSchedule) : ℕ :=
  3 * schedule.hours_mon_wed_fri + 2 * schedule.hours_tue_thu

/-- Calculates the hourly wage --/
def hourly_wage (schedule : WorkSchedule) : ℚ :=
  schedule.weekly_earnings / (total_hours schedule)

/-- Sheila's work schedule --/
def sheila_schedule : WorkSchedule :=
  { hours_mon_wed_fri := 8
  , hours_tue_thu := 6
  , weekly_earnings := 252 }

/-- Theorem stating that Sheila's hourly wage is $7 --/
theorem sheila_hourly_wage : hourly_wage sheila_schedule = 7 := by
  sorry


end NUMINAMATH_CALUDE_sheila_hourly_wage_l3670_367037
