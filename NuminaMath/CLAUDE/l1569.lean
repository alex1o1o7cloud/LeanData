import Mathlib

namespace NUMINAMATH_CALUDE_mrs_hilt_pies_l1569_156970

def pecan_pies : ℝ := 16.0
def apple_pies : ℝ := 14.0
def increase_factor : ℝ := 5.0

theorem mrs_hilt_pies : 
  (pecan_pies + apple_pies) * increase_factor = 150.0 := by
  sorry

end NUMINAMATH_CALUDE_mrs_hilt_pies_l1569_156970


namespace NUMINAMATH_CALUDE_condition_a_geq_4_l1569_156969

theorem condition_a_geq_4 (a : ℝ) :
  (a ≥ 4 → ∃ x : ℝ, x ∈ Set.Icc (-1) 2 ∧ x^2 - 2*x + 4 - a ≤ 0) ∧
  ¬(∃ x : ℝ, x ∈ Set.Icc (-1) 2 ∧ x^2 - 2*x + 4 - a ≤ 0 → a ≥ 4) :=
by sorry

end NUMINAMATH_CALUDE_condition_a_geq_4_l1569_156969


namespace NUMINAMATH_CALUDE_max_value_ratio_l1569_156942

theorem max_value_ratio (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) :
  (x + y + z)^2 / (x^2 + y^2 + z^2) ≤ 3 ∧
  ∃ (a b c : ℝ), a > 0 ∧ b > 0 ∧ c > 0 ∧ (a + b + c)^2 / (a^2 + b^2 + c^2) = 3 :=
by sorry

end NUMINAMATH_CALUDE_max_value_ratio_l1569_156942


namespace NUMINAMATH_CALUDE_dans_marbles_l1569_156952

/-- The number of violet marbles Dan has -/
def violet_marbles : ℕ := 64

/-- The number of red marbles Mary gave to Dan -/
def red_marbles : ℕ := 14

/-- The total number of marbles Dan has now -/
def total_marbles : ℕ := violet_marbles + red_marbles

theorem dans_marbles : total_marbles = 78 := by
  sorry

end NUMINAMATH_CALUDE_dans_marbles_l1569_156952


namespace NUMINAMATH_CALUDE_factor_implies_coefficient_value_l1569_156908

/-- Given a polynomial Q(x) = x^4 + 3x^3 + ax^2 + 17x + 27, 
    if (x-3) is a factor of Q(x), then a = -80/3 -/
theorem factor_implies_coefficient_value (a : ℚ) : 
  let Q := fun (x : ℚ) => x^4 + 3*x^3 + a*x^2 + 17*x + 27
  (∃ (P : ℚ → ℚ), Q = fun x => P x * (x - 3)) → a = -80/3 := by
sorry

end NUMINAMATH_CALUDE_factor_implies_coefficient_value_l1569_156908


namespace NUMINAMATH_CALUDE_fraction_inequality_l1569_156987

def numerator (x : ℝ) : ℝ := 7 * x - 3

def denominator (x : ℝ) : ℝ := x^2 - x - 12

def valid_x (x : ℝ) : Prop := denominator x ≠ 0

def inequality_holds (x : ℝ) : Prop := numerator x ≥ denominator x

def solution_set : Set ℝ := {x | x ∈ Set.Icc (-1) 3 ∪ Set.Ioo 3 4 ∪ Set.Ico 4 9}

theorem fraction_inequality :
  {x : ℝ | inequality_holds x ∧ valid_x x} = solution_set := by sorry

end NUMINAMATH_CALUDE_fraction_inequality_l1569_156987


namespace NUMINAMATH_CALUDE_six_digit_permutations_eq_90_l1569_156920

/-- The number of different positive six-digit integers formed using 1, 1, 3, 3, 7, and 7 -/
def six_digit_permutations : ℕ :=
  Nat.factorial 6 / (Nat.factorial 2 * Nat.factorial 2 * Nat.factorial 2)

theorem six_digit_permutations_eq_90 : six_digit_permutations = 90 := by
  sorry

end NUMINAMATH_CALUDE_six_digit_permutations_eq_90_l1569_156920


namespace NUMINAMATH_CALUDE_events_mutually_exclusive_not_complementary_l1569_156989

-- Define the set of people
inductive Person : Type
| A : Person
| B : Person
| C : Person
| D : Person

-- Define the set of cards
inductive Card : Type
| Red : Card
| Black : Card
| White : Card
| Blue : Card

-- Define a distribution as a function from Person to Card
def Distribution := Person → Card

-- Define the event "Person A gets the red card"
def event_A_red (d : Distribution) : Prop := d Person.A = Card.Red

-- Define the event "Person B gets the red card"
def event_B_red (d : Distribution) : Prop := d Person.B = Card.Red

-- Define mutual exclusivity
def mutually_exclusive (e1 e2 : Distribution → Prop) : Prop :=
  ∀ d : Distribution, ¬(e1 d ∧ e2 d)

-- Define complementary events
def complementary (e1 e2 : Distribution → Prop) : Prop :=
  ∀ d : Distribution, e1 d ↔ ¬(e2 d)

-- Theorem statement
theorem events_mutually_exclusive_not_complementary :
  mutually_exclusive event_A_red event_B_red ∧
  ¬(complementary event_A_red event_B_red) :=
sorry

end NUMINAMATH_CALUDE_events_mutually_exclusive_not_complementary_l1569_156989


namespace NUMINAMATH_CALUDE_polynomial_division_quotient_l1569_156932

theorem polynomial_division_quotient :
  let dividend : Polynomial ℚ := 3 * X^4 - 5 * X^3 + 6 * X^2 - 8 * X + 3
  let divisor : Polynomial ℚ := X^2 + X + 1
  let quotient : Polynomial ℚ := 3 * X^2 - 8 * X
  (dividend / divisor) = quotient := by sorry

end NUMINAMATH_CALUDE_polynomial_division_quotient_l1569_156932


namespace NUMINAMATH_CALUDE_equation_solution_l1569_156919

theorem equation_solution (x : ℚ) : (40 : ℚ) / 60 = Real.sqrt (x / 60) → x = 80 / 3 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l1569_156919


namespace NUMINAMATH_CALUDE_no_member_divisible_by_four_l1569_156906

-- Define the set T
def T : Set ℤ := {s | ∃ n : ℤ, s = (n - 1)^2 + n^2 + (n + 1)^2 + (n + 2)^2}

-- Theorem statement
theorem no_member_divisible_by_four : ∀ s ∈ T, ¬(4 ∣ s) := by
  sorry

end NUMINAMATH_CALUDE_no_member_divisible_by_four_l1569_156906


namespace NUMINAMATH_CALUDE_family_hard_shell_tacos_l1569_156975

/-- The number of hard shell tacos bought by a family -/
def hard_shell_tacos : ℕ := sorry

/-- The price of a soft taco in dollars -/
def soft_taco_price : ℕ := 2

/-- The price of a hard shell taco in dollars -/
def hard_shell_taco_price : ℕ := 5

/-- The number of soft tacos bought by the family -/
def family_soft_tacos : ℕ := 3

/-- The number of additional customers -/
def additional_customers : ℕ := 10

/-- The number of soft tacos bought by each additional customer -/
def soft_tacos_per_customer : ℕ := 2

/-- The total revenue in dollars -/
def total_revenue : ℕ := 66

theorem family_hard_shell_tacos :
  hard_shell_tacos = 4 :=
by sorry

end NUMINAMATH_CALUDE_family_hard_shell_tacos_l1569_156975


namespace NUMINAMATH_CALUDE_min_sum_of_a_and_b_l1569_156902

/-- Given a line x/a + y/b = 1 where a > 0 and b > 0, and the line passes through (2, 2),
    the minimum value of a + b is 8. -/
theorem min_sum_of_a_and_b (a b : ℝ) (ha : a > 0) (hb : b > 0) 
    (h_line : 2/a + 2/b = 1) : 
  ∀ (x y : ℝ), x/a + y/b = 1 → a + b ≥ 8 ∧ ∃ (a₀ b₀ : ℝ), a₀ + b₀ = 8 := by
  sorry

end NUMINAMATH_CALUDE_min_sum_of_a_and_b_l1569_156902


namespace NUMINAMATH_CALUDE_number_puzzle_l1569_156949

theorem number_puzzle : ∃ x : ℝ, x / 3 = x - 36 ∧ x = 54 := by
  sorry

end NUMINAMATH_CALUDE_number_puzzle_l1569_156949


namespace NUMINAMATH_CALUDE_range_of_p_range_of_p_xor_q_l1569_156964

-- Define proposition p
def p (a : ℝ) : Prop := ∀ x : ℝ, a * x^2 + a * x + 1 > 0

-- Define proposition q
def q (a : ℝ) : Prop := |2 * a - 1| < 3

-- Define the set of a for which p is true
def S₁ : Set ℝ := {a : ℝ | p a}

-- Define the set of a for which p∨q is true and p∧q is false
def S₂ : Set ℝ := {a : ℝ | (p a ∨ q a) ∧ ¬(p a ∧ q a)}

-- Theorem 1
theorem range_of_p : S₁ = Set.Ici 0 ∩ Set.Iio 4 := by sorry

-- Theorem 2
theorem range_of_p_xor_q : S₂ = (Set.Ioi (-1) ∩ Set.Iio 0) ∪ (Set.Ici 2 ∩ Set.Iio 4) := by sorry

end NUMINAMATH_CALUDE_range_of_p_range_of_p_xor_q_l1569_156964


namespace NUMINAMATH_CALUDE_inverse_composition_l1569_156980

-- Define the functions f and g
variable (f g : ℝ → ℝ)

-- Define the condition
variable (h : ∀ x, f⁻¹ (g x) = 7 * x - 4)

-- State the theorem
theorem inverse_composition :
  g⁻¹ (f (-9)) = -5/7 := by
  sorry

end NUMINAMATH_CALUDE_inverse_composition_l1569_156980


namespace NUMINAMATH_CALUDE_perpendicular_to_parallel_line_l1569_156936

-- Define the types for planes and lines
variable (Plane Line : Type)

-- Define the relations
variable (perpendicular : Line → Plane → Prop)
variable (parallel : Line → Plane → Prop)
variable (perpendicular_lines : Line → Line → Prop)

-- State the theorem
theorem perpendicular_to_parallel_line 
  (α β : Plane) (m n : Line) 
  (h1 : α ≠ β) 
  (h2 : m ≠ n) 
  (h3 : perpendicular m α) 
  (h4 : parallel n α) : 
  perpendicular_lines m n :=
sorry

end NUMINAMATH_CALUDE_perpendicular_to_parallel_line_l1569_156936


namespace NUMINAMATH_CALUDE_factorization_sum_l1569_156983

def P (y : ℤ) : ℤ := y^6 - y^3 - 2*y - 2

def is_irreducible_factor (q : ℤ → ℤ) : Prop :=
  (∀ y, q y ∣ P y) ∧ 
  (∀ f g : ℤ → ℤ, (∀ y, q y = f y * g y) → (∀ y, f y = 1 ∨ g y = 1))

theorem factorization_sum (q₁ q₂ q₃ q₄ : ℤ → ℤ) :
  is_irreducible_factor q₁ ∧
  is_irreducible_factor q₂ ∧
  is_irreducible_factor q₃ ∧
  is_irreducible_factor q₄ ∧
  (∀ y, P y = q₁ y * q₂ y * q₃ y * q₄ y) →
  q₁ 3 + q₂ 3 + q₃ 3 + q₄ 3 = 30 := by
  sorry

end NUMINAMATH_CALUDE_factorization_sum_l1569_156983


namespace NUMINAMATH_CALUDE_fraction_zero_implies_x_negative_three_l1569_156911

theorem fraction_zero_implies_x_negative_three (x : ℝ) : 
  (x + 3) / (x - 4) = 0 → x = -3 := by
  sorry

end NUMINAMATH_CALUDE_fraction_zero_implies_x_negative_three_l1569_156911


namespace NUMINAMATH_CALUDE_triangle_existence_l1569_156935

-- Define the function f
def f (m : ℝ) (x : ℝ) : ℝ := x^3 - 3*x + m

-- Define the theorem
theorem triangle_existence (m : ℝ) : 
  (∀ a b c : ℝ, 0 ≤ a ∧ a ≤ 2 ∧ 0 ≤ b ∧ b ≤ 2 ∧ 0 ≤ c ∧ c ≤ 2 ∧ 
   a ≠ b ∧ b ≠ c ∧ a ≠ c →
   f m a + f m b > f m c ∧
   f m a + f m c > f m b ∧
   f m b + f m c > f m a) ↔
  m > 6 := by sorry

end NUMINAMATH_CALUDE_triangle_existence_l1569_156935


namespace NUMINAMATH_CALUDE_ball_color_probability_l1569_156978

theorem ball_color_probability : 
  let n : ℕ := 8
  let p : ℝ := 1 / 2
  let k : ℕ := 4
  Nat.choose n k * p^n = 35 / 128 := by sorry

end NUMINAMATH_CALUDE_ball_color_probability_l1569_156978


namespace NUMINAMATH_CALUDE_combined_tax_rate_l1569_156993

/-- Given two individuals with different tax rates and income levels, 
    calculate their combined tax rate -/
theorem combined_tax_rate 
  (mork_rate : ℚ) 
  (mindy_rate : ℚ) 
  (income_ratio : ℚ) : 
  mork_rate = 45/100 → 
  mindy_rate = 25/100 → 
  income_ratio = 4 → 
  (mork_rate + income_ratio * mindy_rate) / (1 + income_ratio) = 29/100 :=
by sorry

end NUMINAMATH_CALUDE_combined_tax_rate_l1569_156993


namespace NUMINAMATH_CALUDE_money_left_after_tickets_l1569_156999

/-- The amount of money Olivia and Nigel have left after buying tickets -/
def money_left (olivia_money : ℕ) (nigel_money : ℕ) (num_tickets : ℕ) (ticket_price : ℕ) : ℕ :=
  (olivia_money + nigel_money) - (num_tickets * ticket_price)

/-- Theorem stating the amount of money left after buying tickets -/
theorem money_left_after_tickets :
  money_left 112 139 6 28 = 83 := by
  sorry

end NUMINAMATH_CALUDE_money_left_after_tickets_l1569_156999


namespace NUMINAMATH_CALUDE_number_of_dimes_l1569_156928

-- Define the coin values in cents
def quarter_value : ℕ := 25
def nickel_value : ℕ := 5
def penny_value : ℕ := 1
def dime_value : ℕ := 10

-- Define the number of coins Tom found
def num_quarters : ℕ := 10
def num_nickels : ℕ := 4
def num_pennies : ℕ := 200

-- Define the total amount Tom found in cents
def total_amount : ℕ := 500

-- Theorem to prove
theorem number_of_dimes :
  ∃ (num_dimes : ℕ),
    num_dimes * dime_value =
      total_amount -
      (num_quarters * quarter_value +
       num_nickels * nickel_value +
       num_pennies * penny_value) ∧
    num_dimes = 3 :=
by sorry

end NUMINAMATH_CALUDE_number_of_dimes_l1569_156928


namespace NUMINAMATH_CALUDE_intersection_implies_value_l1569_156937

theorem intersection_implies_value (a : ℝ) : 
  let A : Set ℝ := {2, a - 1}
  let B : Set ℝ := {a^2 - 7, -1}
  A ∩ B = {2} → a = -3 :=
by
  sorry

end NUMINAMATH_CALUDE_intersection_implies_value_l1569_156937


namespace NUMINAMATH_CALUDE_max_value_of_d_l1569_156943

theorem max_value_of_d (a b c d : ℝ) 
  (sum_eq : a + b + c + d = 10)
  (sum_products_eq : a*b + a*c + a*d + b*c + b*d + c*d = 20) :
  d ≤ (5 + Real.sqrt 105) / 2 :=
sorry

end NUMINAMATH_CALUDE_max_value_of_d_l1569_156943


namespace NUMINAMATH_CALUDE_whitney_money_left_l1569_156988

/-- The amount of money Whitney has left after her purchase at the school book fair --/
def money_left_over : ℕ :=
  let initial_money : ℕ := 2 * 20
  let poster_cost : ℕ := 5
  let notebook_cost : ℕ := 4
  let bookmark_cost : ℕ := 2
  let poster_quantity : ℕ := 2
  let notebook_quantity : ℕ := 3
  let bookmark_quantity : ℕ := 2
  let total_cost : ℕ := poster_cost * poster_quantity + 
                        notebook_cost * notebook_quantity + 
                        bookmark_cost * bookmark_quantity
  initial_money - total_cost

theorem whitney_money_left : money_left_over = 14 := by
  sorry

end NUMINAMATH_CALUDE_whitney_money_left_l1569_156988


namespace NUMINAMATH_CALUDE_quadratic_roots_theorem_l1569_156921

theorem quadratic_roots_theorem (c : ℝ) : 
  (∀ x : ℝ, x^2 + 3*x + c = 0 ↔ x = (-3 + Real.sqrt 7) / 2 ∨ x = (-3 - Real.sqrt 7) / 2) →
  c = 1/2 := by
sorry

end NUMINAMATH_CALUDE_quadratic_roots_theorem_l1569_156921


namespace NUMINAMATH_CALUDE_sequence_ratio_l1569_156940

def is_arithmetic_sequence (a b c d : ℝ) : Prop :=
  b - a = c - b ∧ c - b = d - c

def is_geometric_sequence (a b c d e : ℝ) : Prop :=
  b / a = c / b ∧ c / b = d / c ∧ d / c = e / d

theorem sequence_ratio (a₁ a₂ b₁ b₂ b₃ : ℝ) :
  is_arithmetic_sequence 1 a₁ a₂ 9 →
  is_geometric_sequence 1 b₁ b₂ b₃ 9 →
  b₂ / (a₁ + a₂) = 3 / 10 :=
by sorry

end NUMINAMATH_CALUDE_sequence_ratio_l1569_156940


namespace NUMINAMATH_CALUDE_rectangular_prism_sum_l1569_156962

/-- A rectangular prism with dimensions 3, 4, and 5 units -/
structure RectangularPrism where
  length : ℕ := 3
  width : ℕ := 4
  height : ℕ := 5

/-- The number of edges in a rectangular prism -/
def num_edges (p : RectangularPrism) : ℕ := 12

/-- The number of vertices in a rectangular prism -/
def num_vertices (p : RectangularPrism) : ℕ := 8

/-- The number of faces in a rectangular prism -/
def num_faces (p : RectangularPrism) : ℕ := 6

theorem rectangular_prism_sum (p : RectangularPrism) :
  num_edges p + num_vertices p + num_faces p = 26 := by
  sorry

end NUMINAMATH_CALUDE_rectangular_prism_sum_l1569_156962


namespace NUMINAMATH_CALUDE_nathans_score_l1569_156982

theorem nathans_score (total_students : ℕ) (students_without_nathan : ℕ) 
  (avg_without_nathan : ℚ) (avg_with_nathan : ℚ) :
  total_students = 18 →
  students_without_nathan = 17 →
  avg_without_nathan = 84 →
  avg_with_nathan = 87 →
  (total_students * avg_with_nathan - students_without_nathan * avg_without_nathan : ℚ) = 138 :=
by sorry

end NUMINAMATH_CALUDE_nathans_score_l1569_156982


namespace NUMINAMATH_CALUDE_circle_symmetry_l1569_156992

def circle_equation (center : ℝ × ℝ) (radius : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | (p.1 - center.1)^2 + (p.2 - center.2)^2 = radius^2}

def symmetric_point (p : ℝ × ℝ) : ℝ × ℝ :=
  (-p.1, -p.2)

theorem circle_symmetry (h : Set (ℝ × ℝ)) :
  h = circle_equation (-2, 1) 1 →
  (λ (x, y) => circle_equation (2, -1) 1 (x, y)) =
  (λ (x, y) => (x - 2)^2 + (y + 1)^2 = 1) := by
  sorry

end NUMINAMATH_CALUDE_circle_symmetry_l1569_156992


namespace NUMINAMATH_CALUDE_square_area_from_adjacent_points_l1569_156953

/-- Given two adjacent points (2,1) and (2,7) on a square in a Cartesian coordinate plane,
    the area of the square is 36. -/
theorem square_area_from_adjacent_points :
  let p1 : ℝ × ℝ := (2, 1)
  let p2 : ℝ × ℝ := (2, 7)
  let square_side := Real.sqrt ((p2.1 - p1.1)^2 + (p2.2 - p1.2)^2)
  square_side^2 = 36 :=
by sorry

end NUMINAMATH_CALUDE_square_area_from_adjacent_points_l1569_156953


namespace NUMINAMATH_CALUDE_solve_airport_distance_l1569_156905

/-- Represents the problem of calculating the distance to the airport --/
def airport_distance_problem (initial_speed : ℝ) (speed_increase : ℝ) (initial_time : ℝ) 
  (late_time : ℝ) (early_time : ℝ) : Prop :=
  ∃ (distance : ℝ) (total_time : ℝ),
    -- Initial part of the journey
    initial_speed * initial_time = initial_speed
    -- Total distance equation
    ∧ distance = initial_speed * (total_time + late_time)
    -- Remaining distance equation with increased speed
    ∧ distance - initial_speed * initial_time = (initial_speed + speed_increase) * (total_time - initial_time - early_time)
    -- The solution
    ∧ distance = 264

/-- The theorem stating the solution to the airport distance problem --/
theorem solve_airport_distance : 
  airport_distance_problem 45 20 1 0.75 0.75 := by
  sorry

end NUMINAMATH_CALUDE_solve_airport_distance_l1569_156905


namespace NUMINAMATH_CALUDE_third_number_proof_l1569_156944

def digit_sum (n : ℕ) : ℕ := sorry

def has_same_remainder (a b c n : ℕ) : Prop :=
  ∃ r, a % n = r ∧ b % n = r ∧ c % n = r

theorem third_number_proof :
  ∃! x : ℕ,
    ∃ n : ℕ,
      has_same_remainder 1305 4665 x n ∧
      (∀ m : ℕ, has_same_remainder 1305 4665 x m → m ≤ n) ∧
      digit_sum n = 4 ∧
      x = 4705 :=
sorry

end NUMINAMATH_CALUDE_third_number_proof_l1569_156944


namespace NUMINAMATH_CALUDE_geometric_sequence_second_term_l1569_156939

/-- A geometric sequence with first term 5 and third term 20 has second term 10 -/
theorem geometric_sequence_second_term :
  ∀ (a : ℝ) (r : ℝ),
    a = 5 →
    a * r^2 = 20 →
    a * r = 10 :=
by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_second_term_l1569_156939


namespace NUMINAMATH_CALUDE_eliminate_uvw_l1569_156966

theorem eliminate_uvw (a b c d u v w : ℝ) 
  (eq1 : a = Real.cos u + Real.cos v + Real.cos w)
  (eq2 : b = Real.sin u + Real.sin v + Real.sin w)
  (eq3 : c = Real.cos (2*u) + Real.cos (2*v) + Real.cos (2*w))
  (eq4 : d = Real.sin (2*u) + Real.sin (2*v) + Real.sin (2*w)) :
  (a^2 - b^2 - c)^2 + (2*a*b - d)^2 = 4*(a^2 + b^2) := by
sorry

end NUMINAMATH_CALUDE_eliminate_uvw_l1569_156966


namespace NUMINAMATH_CALUDE_garden_length_l1569_156967

theorem garden_length (width : ℝ) (length : ℝ) : 
  width > 0 →
  length = 2 * width →
  2 * length + 2 * width = 240 →
  length = 80 :=
by
  sorry

end NUMINAMATH_CALUDE_garden_length_l1569_156967


namespace NUMINAMATH_CALUDE_ellipse_sum_property_l1569_156900

/-- Properties of an ellipse -/
structure Ellipse where
  h : ℝ  -- x-coordinate of the center
  k : ℝ  -- y-coordinate of the center
  a : ℝ  -- length of semi-major axis
  b : ℝ  -- length of semi-minor axis

/-- Theorem about the sum of center coordinates and axis lengths for a specific ellipse -/
theorem ellipse_sum_property (E : Ellipse) 
  (center_x : E.h = 3) 
  (center_y : E.k = -5) 
  (major_axis : E.a = 6) 
  (minor_axis : E.b = 2) : 
  E.h + E.k + E.a + E.b = 6 := by
  sorry

#check ellipse_sum_property

end NUMINAMATH_CALUDE_ellipse_sum_property_l1569_156900


namespace NUMINAMATH_CALUDE_cubic_sum_in_terms_of_products_l1569_156991

theorem cubic_sum_in_terms_of_products (x y z p q r : ℝ) 
  (h_xy : x * y = p)
  (h_xz : x * z = q)
  (h_yz : y * z = r)
  (h_x_nonzero : x ≠ 0)
  (h_y_nonzero : y ≠ 0)
  (h_z_nonzero : z ≠ 0) :
  x^3 + y^3 + z^3 = (p^2 * q^2 + p^2 * r^2 + q^2 * r^2) / (p * q * r) :=
by sorry

end NUMINAMATH_CALUDE_cubic_sum_in_terms_of_products_l1569_156991


namespace NUMINAMATH_CALUDE_juvy_garden_rosemary_rows_l1569_156923

/-- Represents a garden with rows of plants -/
structure Garden where
  total_rows : ℕ
  plants_per_row : ℕ
  parsley_rows : ℕ
  chive_plants : ℕ

/-- Calculates the number of rows planted with rosemary -/
def rosemary_rows (g : Garden) : ℕ :=
  g.total_rows - g.parsley_rows - (g.chive_plants / g.plants_per_row)

/-- Theorem stating that Juvy's garden has 2 rows of rosemary -/
theorem juvy_garden_rosemary_rows :
  let g : Garden := {
    total_rows := 20,
    plants_per_row := 10,
    parsley_rows := 3,
    chive_plants := 150
  }
  rosemary_rows g = 2 := by sorry

end NUMINAMATH_CALUDE_juvy_garden_rosemary_rows_l1569_156923


namespace NUMINAMATH_CALUDE_max_schools_donation_l1569_156927

/-- Represents the donation problem for The Khan Corporation --/
structure DonationProblem where
  total_computers : ℕ
  total_printers : ℕ
  min_computers_per_school : ℕ
  max_computers_per_school : ℕ
  min_printers_per_school : ℕ
  max_printers_per_school : ℕ

/-- Checks if a number of schools satisfies the donation criteria --/
def satisfies_criteria (problem : DonationProblem) (num_schools : ℕ) : Prop :=
  num_schools > 0 ∧
  (problem.total_computers % num_schools = 0) ∧
  (problem.total_printers % num_schools = 0) ∧
  (problem.total_computers / num_schools ≥ problem.min_computers_per_school) ∧
  (problem.total_computers / num_schools ≤ problem.max_computers_per_school) ∧
  (problem.total_printers / num_schools ≥ problem.min_printers_per_school) ∧
  (problem.total_printers / num_schools ≤ problem.max_printers_per_school)

/-- The Khan Corporation donation problem --/
def khan_problem : DonationProblem :=
  { total_computers := 48,
    total_printers := 32,
    min_computers_per_school := 4,
    max_computers_per_school := 8,
    min_printers_per_school := 2,
    max_printers_per_school := 4 }

/-- Theorem stating that 12 is the maximum number of schools that can receive donations --/
theorem max_schools_donation :
  ∀ n : ℕ, satisfies_criteria khan_problem n → n ≤ 12 :=
by sorry

end NUMINAMATH_CALUDE_max_schools_donation_l1569_156927


namespace NUMINAMATH_CALUDE_cube_surface_area_equal_volume_l1569_156926

/-- The surface area of a cube with volume equal to a 9x3x27 inch rectangular prism is 486 square inches. -/
theorem cube_surface_area_equal_volume (l w h : ℝ) (cube_edge : ℝ) : 
  l = 9 ∧ w = 3 ∧ h = 27 →
  cube_edge ^ 3 = l * w * h →
  6 * cube_edge ^ 2 = 486 := by
  sorry

end NUMINAMATH_CALUDE_cube_surface_area_equal_volume_l1569_156926


namespace NUMINAMATH_CALUDE_simplify_and_evaluate_l1569_156946

theorem simplify_and_evaluate (a : ℝ) (h : a = 2) : 
  (1 - 1 / (a + 1)) / (a / (a^2 - 1)) = 1 := by
  sorry

end NUMINAMATH_CALUDE_simplify_and_evaluate_l1569_156946


namespace NUMINAMATH_CALUDE_probability_of_specific_arrangement_l1569_156979

def total_tiles : ℕ := 5
def x_tiles : ℕ := 3
def o_tiles : ℕ := 2

def specific_arrangement : List Char := ['X', 'O', 'X', 'O', 'X']

def probability_of_arrangement : ℚ :=
  1 / (total_tiles.factorial / (x_tiles.factorial * o_tiles.factorial))

theorem probability_of_specific_arrangement :
  probability_of_arrangement = 1 / 10 := by
  sorry

end NUMINAMATH_CALUDE_probability_of_specific_arrangement_l1569_156979


namespace NUMINAMATH_CALUDE_circus_dogs_count_l1569_156977

theorem circus_dogs_count :
  ∀ (total_dogs : ℕ) (paws_on_ground : ℕ),
    paws_on_ground = 36 →
    (total_dogs / 2 : ℕ) * 2 + (total_dogs / 2 : ℕ) * 4 = paws_on_ground →
    total_dogs = 12 :=
by sorry

end NUMINAMATH_CALUDE_circus_dogs_count_l1569_156977


namespace NUMINAMATH_CALUDE_five_digit_diff_last_two_count_l1569_156910

/-- The number of five-digit numbers -/
def total_five_digit_numbers : ℕ := 90000

/-- The number of five-digit numbers where the last two digits are the same -/
def five_digit_same_last_two : ℕ := 9000

/-- The number of five-digit numbers where at least the last two digits are different -/
def five_digit_diff_last_two : ℕ := total_five_digit_numbers - five_digit_same_last_two

theorem five_digit_diff_last_two_count : five_digit_diff_last_two = 81000 := by
  sorry

end NUMINAMATH_CALUDE_five_digit_diff_last_two_count_l1569_156910


namespace NUMINAMATH_CALUDE_option_B_not_mapping_l1569_156933

-- Define the sets and mappings
def CartesianPlane : Type := ℝ × ℝ
def CircleOnPlane : Type := Unit -- Placeholder type for circles
def TriangleOnPlane : Type := Unit -- Placeholder type for triangles

-- Option A
def mappingA : CartesianPlane → CartesianPlane := id

-- Option B (not a mapping)
noncomputable def correspondenceB : CircleOnPlane → Set TriangleOnPlane := sorry

-- Option C
def mappingC : ℕ → Fin 2 := fun n => n % 2

-- Option D
def mappingD : Fin 3 → Fin 3 := fun n => n^2

-- Theorem stating that B is not a mapping while others are
theorem option_B_not_mapping :
  (∀ x : CartesianPlane, ∃! y : CartesianPlane, mappingA x = y) ∧
  (∃ c : CircleOnPlane, ¬∃! t : TriangleOnPlane, t ∈ correspondenceB c) ∧
  (∀ n : ℕ, ∃! m : Fin 2, mappingC n = m) ∧
  (∀ x : Fin 3, ∃! y : Fin 3, mappingD x = y) := by
  sorry

end NUMINAMATH_CALUDE_option_B_not_mapping_l1569_156933


namespace NUMINAMATH_CALUDE_sum_remainder_l1569_156956

theorem sum_remainder (x y z : ℕ+) 
  (hx : x ≡ 36 [ZMOD 53])
  (hy : y ≡ 15 [ZMOD 53])
  (hz : z ≡ 7 [ZMOD 53]) :
  (x + y + z : ℤ) ≡ 5 [ZMOD 53] := by
  sorry

end NUMINAMATH_CALUDE_sum_remainder_l1569_156956


namespace NUMINAMATH_CALUDE_jugglers_balls_l1569_156998

theorem jugglers_balls (num_jugglers : ℕ) (total_balls : ℕ) 
  (h1 : num_jugglers = 378) 
  (h2 : total_balls = 2268) : 
  total_balls / num_jugglers = 6 := by
  sorry

end NUMINAMATH_CALUDE_jugglers_balls_l1569_156998


namespace NUMINAMATH_CALUDE_book_prices_l1569_156986

def total_cost : ℕ := 104

def is_valid_price (price : ℕ) : Prop :=
  ∃ (n : ℕ), 10 < n ∧ n < 60 ∧ n * price = total_cost

theorem book_prices :
  {p : ℕ | is_valid_price p} = {2, 4, 8} :=
by sorry

end NUMINAMATH_CALUDE_book_prices_l1569_156986


namespace NUMINAMATH_CALUDE_points_4_units_from_neg5_are_neg9_and_neg1_l1569_156965

-- Define the distance between two points on a number line
def distance (x y : ℝ) : ℝ := |x - y|

-- Define the set of points that are 4 units away from -5
def points_4_units_from_neg5 : Set ℝ := {x : ℝ | distance x (-5) = 4}

-- Theorem statement
theorem points_4_units_from_neg5_are_neg9_and_neg1 :
  points_4_units_from_neg5 = {-9, -1} := by sorry

end NUMINAMATH_CALUDE_points_4_units_from_neg5_are_neg9_and_neg1_l1569_156965


namespace NUMINAMATH_CALUDE_fraction_A_proof_l1569_156963

/-- The fraction that A gets compared to what B and C together get -/
def fraction_A (total amount_A amount_B amount_C : ℚ) : ℚ :=
  amount_A / (amount_B + amount_C)

theorem fraction_A_proof 
  (total : ℚ) 
  (amount_A amount_B amount_C : ℚ) 
  (h1 : total = 1260)
  (h2 : ∃ x : ℚ, amount_A = x * (amount_B + amount_C))
  (h3 : amount_B = 2/7 * (amount_A + amount_C))
  (h4 : amount_A = amount_B + 35)
  (h5 : total = amount_A + amount_B + amount_C) :
  fraction_A total amount_A amount_B amount_C = 63/119 := by
  sorry

#eval fraction_A 1260 315 280 665

end NUMINAMATH_CALUDE_fraction_A_proof_l1569_156963


namespace NUMINAMATH_CALUDE_complex_sum_theorem_l1569_156951

theorem complex_sum_theorem (a b c d : ℝ) (ω : ℂ) 
  (ha : a ≠ -1) (hb : b ≠ -1) (hc : c ≠ -1) (hd : d ≠ -1)
  (hω1 : ω^3 = 1) (hω2 : ω ≠ 1)
  (h : 1 / (a + ω) + 1 / (b + ω) + 1 / (c + ω) + 1 / (d + ω) = 2 / ω) :
  1 / (a + 1) + 1 / (b + 1) + 1 / (c + 1) + 1 / (d + 1) = 2 := by
  sorry

end NUMINAMATH_CALUDE_complex_sum_theorem_l1569_156951


namespace NUMINAMATH_CALUDE_min_value_of_f_l1569_156934

def f (x a : ℝ) : ℝ := |x - 4| + |x - a|

theorem min_value_of_f (a : ℝ) : (∀ x, f x a ≥ 3) ∧ (∃ x, f x a = 3) ↔ a = 1 ∨ a = 7 := by
  sorry

end NUMINAMATH_CALUDE_min_value_of_f_l1569_156934


namespace NUMINAMATH_CALUDE_triangle_side_value_l1569_156916

theorem triangle_side_value (A B C : ℝ) (a b c : ℝ) :
  0 < A ∧ 0 < B ∧ 0 < C ∧ 
  A + B + C = π ∧
  a > 0 ∧ b > 0 ∧ c > 0 ∧
  a + b + 10 * c = 2 * (Real.sin A + Real.sin B + 10 * Real.sin C) ∧
  A = π / 3 →
  a = Real.sqrt 3 := by
sorry

end NUMINAMATH_CALUDE_triangle_side_value_l1569_156916


namespace NUMINAMATH_CALUDE_total_questions_on_math_test_l1569_156958

/-- The number of questions on a math test -/
def math_test_questions (word_problems subtraction_problems answered_questions blank_questions : ℕ) : Prop :=
  word_problems + subtraction_problems = answered_questions + blank_questions

/-- Theorem: There are 45 questions on the math test -/
theorem total_questions_on_math_test :
  ∃ (word_problems subtraction_problems answered_questions blank_questions : ℕ),
    word_problems = 17 ∧
    subtraction_problems = 28 ∧
    answered_questions = 38 ∧
    blank_questions = 7 ∧
    math_test_questions word_problems subtraction_problems answered_questions blank_questions ∧
    answered_questions + blank_questions = 45 :=
by
  sorry

end NUMINAMATH_CALUDE_total_questions_on_math_test_l1569_156958


namespace NUMINAMATH_CALUDE_coin_toss_problem_l1569_156954

theorem coin_toss_problem (n : ℕ) 
  (total_outcomes : ℕ) 
  (equally_likely : total_outcomes = 8)
  (die_roll_prob : ℚ) 
  (die_roll_prob_value : die_roll_prob = 1/3) :
  (2^n = total_outcomes) → n = 3 := by
  sorry

end NUMINAMATH_CALUDE_coin_toss_problem_l1569_156954


namespace NUMINAMATH_CALUDE_intersection_and_union_of_sets_l1569_156997

def A (a : ℝ) : Set ℝ := {a^2, a+1, -3}
def B (a : ℝ) : Set ℝ := {-3+a, 2*a-1, a^2+1}

theorem intersection_and_union_of_sets :
  ∃ (a : ℝ), (A a ∩ B a = {-3}) ∧ (a = -1) ∧ (A a ∪ B a = {-4, -3, 0, 1, 2}) := by
  sorry

end NUMINAMATH_CALUDE_intersection_and_union_of_sets_l1569_156997


namespace NUMINAMATH_CALUDE_least_five_digit_square_cube_l1569_156996

theorem least_five_digit_square_cube : 
  (∀ n : ℕ, n < 15625 → ¬(∃ a b : ℕ, n = a^2 ∧ n = b^3 ∧ n ≥ 10000)) ∧ 
  (∃ a b : ℕ, 15625 = a^2 ∧ 15625 = b^3) ∧ 
  15625 ≥ 10000 :=
sorry

end NUMINAMATH_CALUDE_least_five_digit_square_cube_l1569_156996


namespace NUMINAMATH_CALUDE_range_of_b_monotonicity_condition_comparison_inequality_l1569_156922

noncomputable section

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := a * x^2 + x - x * Real.log x

-- Statement 1
theorem range_of_b (a : ℝ) (b : ℝ) (h1 : a > 0) (h2 : f a 1 = 2) 
  (h3 : ∀ x > 0, f a x ≥ b * x^2 + 2 * x) : b ≤ 0 := sorry

-- Statement 2
theorem monotonicity_condition (a : ℝ) (h : a > 0) : 
  (∀ x > 0, Monotone (f a)) ↔ a ≥ 1 / (2 * Real.exp 1) := sorry

-- Statement 3
theorem comparison_inequality (x y : ℝ) (h1 : 1 / Real.exp 1 < x) (h2 : x < y) (h3 : y < 1) :
  y / x < (1 + Real.log y) / (1 + Real.log x) := sorry

end NUMINAMATH_CALUDE_range_of_b_monotonicity_condition_comparison_inequality_l1569_156922


namespace NUMINAMATH_CALUDE_trigonometric_identity_l1569_156968

open Real

theorem trigonometric_identity (x : ℝ) : 
  sin x * cos x + sin x^3 * cos x + sin x^5 * (1 / cos x) = tan x := by
  sorry

end NUMINAMATH_CALUDE_trigonometric_identity_l1569_156968


namespace NUMINAMATH_CALUDE_angle_sum_eq_pi_fourth_l1569_156990

theorem angle_sum_eq_pi_fourth (α β : Real) 
  (h1 : α ∈ Set.Ioo 0 (π / 2))
  (h2 : β ∈ Set.Ioo 0 (π / 2))
  (h3 : Real.tan α = 1 / 7)
  (h4 : Real.tan β = 1 / 3) :
  α + 2 * β = π / 4 := by
  sorry

end NUMINAMATH_CALUDE_angle_sum_eq_pi_fourth_l1569_156990


namespace NUMINAMATH_CALUDE_average_shift_l1569_156924

theorem average_shift (x₁ x₂ x₃ : ℝ) (h : (x₁ + x₂ + x₃) / 3 = 40) :
  ((x₁ + 40) + (x₂ + 40) + (x₃ + 40)) / 3 = 80 := by
  sorry

end NUMINAMATH_CALUDE_average_shift_l1569_156924


namespace NUMINAMATH_CALUDE_polynomial_factorization_l1569_156959

theorem polynomial_factorization (x : ℝ) :
  (x^2 + 4*x + 3) * (x^2 + 6*x + 5) + (x^2 + 3*x - 18) = (x^2 + 10*x + 64) * (x^2 + 10*x - 1) := by
  sorry

end NUMINAMATH_CALUDE_polynomial_factorization_l1569_156959


namespace NUMINAMATH_CALUDE_continuous_stripe_probability_l1569_156957

/-- Represents a cube with diagonal stripes on each face --/
structure StripedCube where
  faces : Fin 6 → Bool  -- True for one diagonal orientation, False for the other

/-- The probability of a continuous stripe loop on a cube --/
def probability_continuous_loop : ℚ :=
  2 / 64

theorem continuous_stripe_probability :
  probability_continuous_loop = 1 / 32 := by
  sorry

#check continuous_stripe_probability

end NUMINAMATH_CALUDE_continuous_stripe_probability_l1569_156957


namespace NUMINAMATH_CALUDE_parabola_equation_l1569_156930

/-- A parabola with focus at (-2, 0) has the standard equation y^2 = -8x -/
theorem parabola_equation (F : ℝ × ℝ) (h : F = (-2, 0)) : 
  ∃ (x y : ℝ), y^2 = -8*x := by sorry

end NUMINAMATH_CALUDE_parabola_equation_l1569_156930


namespace NUMINAMATH_CALUDE_reciprocal_and_fraction_operations_l1569_156971

theorem reciprocal_and_fraction_operations :
  (∀ a b c : ℚ, (a + b) / c = -2 → c / (a + b) = -1/2) ∧
  (5/12 - 1/9 + 2/3) / (1/36) = 35 ∧
  (-1/36) / (5/12 - 1/9 + 2/3) = -1/35 := by
  sorry

end NUMINAMATH_CALUDE_reciprocal_and_fraction_operations_l1569_156971


namespace NUMINAMATH_CALUDE_division_value_problem_l1569_156915

theorem division_value_problem (x : ℝ) : 
  (740 / x) - 175 = 10 → x = 4 := by
  sorry

end NUMINAMATH_CALUDE_division_value_problem_l1569_156915


namespace NUMINAMATH_CALUDE_intersection_complement_equality_l1569_156974

theorem intersection_complement_equality (U A B : Set Nat) : 
  U = {1, 2, 3, 4, 5} → 
  A = {1, 3} → 
  B = {2, 5} → 
  A ∩ (U \ B) = {1, 3} := by
sorry

end NUMINAMATH_CALUDE_intersection_complement_equality_l1569_156974


namespace NUMINAMATH_CALUDE_vegetable_planting_methods_l1569_156950

theorem vegetable_planting_methods (n m : ℕ) (hn : n = 5) (hm : m = 4) :
  (n.choose m) * (m.factorial) = 120 := by
  sorry

end NUMINAMATH_CALUDE_vegetable_planting_methods_l1569_156950


namespace NUMINAMATH_CALUDE_smallest_rectangles_covering_square_l1569_156994

theorem smallest_rectangles_covering_square :
  ∃ (n : ℕ), n > 0 ∧ 
  (∀ m : ℕ, m > 0 → 
    (∃ (s : ℕ), s > 0 ∧ 
      s * s = m * 3 * 4 ∧ 
      s % 3 = 0 ∧ 
      s % 4 = 0) → 
    m ≥ n) ∧
  (∃ (s : ℕ), s > 0 ∧ 
    s * s = n * 3 * 4 ∧ 
    s % 3 = 0 ∧ 
    s % 4 = 0) ∧
  n = 12 :=
by sorry

end NUMINAMATH_CALUDE_smallest_rectangles_covering_square_l1569_156994


namespace NUMINAMATH_CALUDE_percentage_of_juniors_l1569_156945

def total_students : ℕ := 800
def seniors : ℕ := 160

theorem percentage_of_juniors : 
  ∀ (freshmen sophomores juniors : ℕ),
  freshmen + sophomores + juniors + seniors = total_students →
  sophomores = total_students / 4 →
  freshmen = sophomores + 32 →
  (juniors : ℚ) / total_students * 100 = 26 :=
by sorry

end NUMINAMATH_CALUDE_percentage_of_juniors_l1569_156945


namespace NUMINAMATH_CALUDE_gcd_of_three_numbers_l1569_156947

theorem gcd_of_three_numbers : Nat.gcd 10711 (Nat.gcd 15809 28041) = 1 := by
  sorry

end NUMINAMATH_CALUDE_gcd_of_three_numbers_l1569_156947


namespace NUMINAMATH_CALUDE_sqrt_five_squared_l1569_156938

theorem sqrt_five_squared : ∃ y : ℝ, y^2 = 5 ∧ y > 0 :=
sorry

end NUMINAMATH_CALUDE_sqrt_five_squared_l1569_156938


namespace NUMINAMATH_CALUDE_interest_rate_difference_l1569_156995

/-- Proves that the difference between two interest rates is 1/3% when one rate produces $81 more interest than the other over 3 years for a $900 principal using simple interest. -/
theorem interest_rate_difference (principal : ℝ) (time : ℝ) (rate1 : ℝ) (rate2 : ℝ) : 
  principal = 900 → 
  time = 3 → 
  principal * rate2 * time - principal * rate1 * time = 81 → 
  rate2 - rate1 = 1/3 * (1/100) := by sorry

end NUMINAMATH_CALUDE_interest_rate_difference_l1569_156995


namespace NUMINAMATH_CALUDE_arithmetic_sequence_common_difference_l1569_156973

/-- An arithmetic sequence with first term 2 and the sum of the 3rd and 5th terms equal to 10 has a common difference of 1. -/
theorem arithmetic_sequence_common_difference : 
  ∀ (a : ℕ → ℝ), 
  (∀ n, a (n + 1) - a n = a 2 - a 1) →  -- arithmetic sequence condition
  a 1 = 2 →                            -- first term is 2
  a 3 + a 5 = 10 →                     -- sum of 3rd and 5th terms is 10
  a 2 - a 1 = 1 :=                     -- common difference is 1
by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_common_difference_l1569_156973


namespace NUMINAMATH_CALUDE_laura_rental_cost_l1569_156907

/-- Calculates the total cost of a car rental given the daily rate, mileage rate, number of days, and miles driven. -/
def rentalCost (dailyRate : ℝ) (mileageRate : ℝ) (days : ℕ) (miles : ℝ) : ℝ :=
  dailyRate * (days : ℝ) + mileageRate * miles

/-- Theorem stating that the total cost of Laura's car rental is $165. -/
theorem laura_rental_cost :
  let dailyRate : ℝ := 30
  let mileageRate : ℝ := 0.25
  let days : ℕ := 3
  let miles : ℝ := 300
  rentalCost dailyRate mileageRate days miles = 165 := by
  sorry

end NUMINAMATH_CALUDE_laura_rental_cost_l1569_156907


namespace NUMINAMATH_CALUDE_base_9_4527_equals_3346_l1569_156929

def base_9_to_10 (digits : List Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * (9 ^ i)) 0

theorem base_9_4527_equals_3346 :
  base_9_to_10 [7, 2, 5, 4] = 3346 := by
  sorry

end NUMINAMATH_CALUDE_base_9_4527_equals_3346_l1569_156929


namespace NUMINAMATH_CALUDE_tony_drives_five_days_a_week_l1569_156961

/-- Represents the problem of determining Tony's work commute frequency --/
def TonysDriving (car_efficiency : ℝ) (round_trip : ℝ) (tank_capacity : ℝ) (gas_price : ℝ) (total_spent : ℝ) (weeks : ℕ) : Prop :=
  let gallons_per_day := round_trip / car_efficiency
  let total_gallons := total_spent / gas_price
  let gallons_per_week := total_gallons / weeks
  gallons_per_week / gallons_per_day = 5

/-- Theorem stating that given the problem conditions, Tony drives to work 5 days a week --/
theorem tony_drives_five_days_a_week :
  TonysDriving 25 50 10 2 80 4 := by
  sorry

end NUMINAMATH_CALUDE_tony_drives_five_days_a_week_l1569_156961


namespace NUMINAMATH_CALUDE_fraction_subtraction_l1569_156917

theorem fraction_subtraction : 
  (2 + 4 + 6 + 8) / (1 + 3 + 5 + 7) - (1 + 3 + 5 + 7) / (2 + 4 + 6 + 8) = 9 / 20 := by
  sorry

end NUMINAMATH_CALUDE_fraction_subtraction_l1569_156917


namespace NUMINAMATH_CALUDE_initial_candies_count_l1569_156931

/-- The number of candies sold on Day 1 -/
def day1_sales : ℕ := 1249

/-- The additional number of candies sold on Day 2 compared to Day 1 -/
def day2_additional : ℕ := 328

/-- The additional number of candies sold on Day 3 compared to Day 2 -/
def day3_additional : ℕ := 275

/-- The number of candies remaining after three days of sales -/
def remaining_candies : ℕ := 367

/-- The total number of candies at the beginning -/
def initial_candies : ℕ := day1_sales + (day1_sales + day2_additional) + (day1_sales + day2_additional + day3_additional) + remaining_candies

theorem initial_candies_count : initial_candies = 5045 := by
  sorry

end NUMINAMATH_CALUDE_initial_candies_count_l1569_156931


namespace NUMINAMATH_CALUDE_largest_special_number_l1569_156904

/-- A function that returns true if all digits in a natural number are distinct -/
def has_distinct_digits (n : ℕ) : Prop := sorry

/-- A function that returns true if a natural number is divisible by all of its digits -/
def divisible_by_all_digits (n : ℕ) : Prop := sorry

/-- A function that returns true if a natural number contains the digit 5 -/
def contains_digit_five (n : ℕ) : Prop := sorry

theorem largest_special_number : 
  ∀ n : ℕ, 
    has_distinct_digits n ∧ 
    divisible_by_all_digits n ∧ 
    contains_digit_five n →
    n ≤ 9315 :=
sorry

end NUMINAMATH_CALUDE_largest_special_number_l1569_156904


namespace NUMINAMATH_CALUDE_sphere_sum_l1569_156985

theorem sphere_sum (x y z : ℝ) : 
  x^2 + y^2 + z^2 - 2*x + 4*y - 6*z + 14 = 0 → x + y + z = 2 := by
  sorry

end NUMINAMATH_CALUDE_sphere_sum_l1569_156985


namespace NUMINAMATH_CALUDE_sin_1440_degrees_l1569_156914

theorem sin_1440_degrees : Real.sin (1440 * π / 180) = 0 := by
  sorry

end NUMINAMATH_CALUDE_sin_1440_degrees_l1569_156914


namespace NUMINAMATH_CALUDE_exponential_inequality_l1569_156913

theorem exponential_inequality (a b c : ℝ) :
  0 < 0.8 ∧ 0.8 < 1 ∧ 5.2 > 1 →
  0.8^5.5 < 0.8^5.2 ∧ 0.8^5.2 < 5.2^0.1 :=
by sorry

end NUMINAMATH_CALUDE_exponential_inequality_l1569_156913


namespace NUMINAMATH_CALUDE_smallest_sum_divisible_by_2016_l1569_156955

theorem smallest_sum_divisible_by_2016 :
  ∃ (n₁ n₂ n₃ n₄ n₅ n₆ n₇ : ℕ),
    0 < n₁ ∧ n₁ < n₂ ∧ n₂ < n₃ ∧ n₃ < n₄ ∧ n₄ < n₅ ∧ n₅ < n₆ ∧ n₆ < n₇ ∧
    (n₁ * n₂ * n₃ * n₄ * n₅ * n₆ * n₇) % 2016 = 0 ∧
    n₁ + n₂ + n₃ + n₄ + n₅ + n₆ + n₇ = 31 ∧
    ∀ (m₁ m₂ m₃ m₄ m₅ m₆ m₇ : ℕ),
      0 < m₁ ∧ m₁ < m₂ ∧ m₂ < m₃ ∧ m₃ < m₄ ∧ m₄ < m₅ ∧ m₅ < m₆ ∧ m₆ < m₇ →
      (m₁ * m₂ * m₃ * m₄ * m₅ * m₆ * m₇) % 2016 = 0 →
      m₁ + m₂ + m₃ + m₄ + m₅ + m₆ + m₇ ≥ 31 :=
by sorry

end NUMINAMATH_CALUDE_smallest_sum_divisible_by_2016_l1569_156955


namespace NUMINAMATH_CALUDE_expansion_coefficient_x_fifth_l1569_156941

theorem expansion_coefficient_x_fifth (x : ℝ) :
  ∃ (aₙ a₁ a₂ a₃ a₄ a₅ : ℝ),
    x^5 = aₙ + a₁*(x-1) + a₂*(x-1)^2 + a₃*(x-1)^3 + a₄*(x-1)^4 + a₅*(x-1)^5 ∧
    a₄ = 5 := by
  sorry

end NUMINAMATH_CALUDE_expansion_coefficient_x_fifth_l1569_156941


namespace NUMINAMATH_CALUDE_fatimas_number_probability_l1569_156981

def first_three_options : List Nat := [296, 299, 295]
def last_five_digits : List Nat := [0, 1, 6, 7, 8]

def total_possibilities : Nat :=
  (first_three_options.length) * (Nat.factorial last_five_digits.length)

theorem fatimas_number_probability :
  (1 : ℚ) / total_possibilities = (1 : ℚ) / 360 := by
  sorry

end NUMINAMATH_CALUDE_fatimas_number_probability_l1569_156981


namespace NUMINAMATH_CALUDE_right_triangle_segment_ratio_l1569_156918

/-- Given a right triangle with sides a and b, hypotenuse c, and a perpendicular from
    the right angle vertex dividing c into segments r and s, prove that if a : b = 2 : 3,
    then r : s = 4 : 9. -/
theorem right_triangle_segment_ratio (a b c r s : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : c > 0)
    (h4 : r > 0) (h5 : s > 0) (h6 : a^2 + b^2 = c^2) (h7 : r + s = c) (h8 : r * c = a^2)
    (h9 : s * c = b^2) (h10 : a / b = 2 / 3) : r / s = 4 / 9 := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_segment_ratio_l1569_156918


namespace NUMINAMATH_CALUDE_A_inverse_l1569_156925

-- Define the matrix A
def A : Matrix (Fin 2) (Fin 2) ℚ :=
  !![4, 5;
    -2, 9]

-- Define the claimed inverse matrix A_inv
def A_inv : Matrix (Fin 2) (Fin 2) ℚ :=
  !![9/46, -5/46;
    1/23, 2/23]

-- Theorem stating that A_inv is the inverse of A
theorem A_inverse : A⁻¹ = A_inv := by
  sorry

end NUMINAMATH_CALUDE_A_inverse_l1569_156925


namespace NUMINAMATH_CALUDE_last_remaining_is_c_implies_start_is_f_l1569_156972

/-- Represents the children in the circle -/
inductive Child : Type
| a | b | c | d | e | f

/-- The number of children in the circle -/
def numChildren : Nat := 6

/-- The number of words in the song -/
def songWords : Nat := 9

/-- Function to determine the last remaining child given a starting position -/
def lastRemaining (start : Child) : Child :=
  sorry

/-- Theorem stating that if c is the last remaining child, the starting position must be f -/
theorem last_remaining_is_c_implies_start_is_f :
  lastRemaining Child.f = Child.c :=
sorry

end NUMINAMATH_CALUDE_last_remaining_is_c_implies_start_is_f_l1569_156972


namespace NUMINAMATH_CALUDE_price_decrease_units_sold_ratio_l1569_156903

theorem price_decrease_units_sold_ratio (P U : ℝ) (h : P > 0) (k : U > 0) :
  let new_price := 0.25 * P
  let new_units := U / 0.25
  let revenue_unchanged := P * U = new_price * new_units
  let percent_decrease_price := 75
  let percent_increase_units := (new_units - U) / U * 100
  revenue_unchanged →
  percent_increase_units / percent_decrease_price = 4 := by
sorry

end NUMINAMATH_CALUDE_price_decrease_units_sold_ratio_l1569_156903


namespace NUMINAMATH_CALUDE_jamie_max_correct_answers_l1569_156948

theorem jamie_max_correct_answers
  (total_questions : ℕ)
  (correct_points : ℤ)
  (blank_points : ℤ)
  (incorrect_points : ℤ)
  (total_score : ℤ)
  (h1 : total_questions = 60)
  (h2 : correct_points = 5)
  (h3 : blank_points = 0)
  (h4 : incorrect_points = -2)
  (h5 : total_score = 150) :
  ∃ (x : ℕ), x ≤ 38 ∧
    ∀ (y : ℕ), y > 38 →
      ¬∃ (blank incorrect : ℕ),
        y + blank + incorrect = total_questions ∧
        y * correct_points + blank * blank_points + incorrect * incorrect_points = total_score :=
by sorry

end NUMINAMATH_CALUDE_jamie_max_correct_answers_l1569_156948


namespace NUMINAMATH_CALUDE_stratified_sampling_participation_l1569_156960

/-- Given a school with 1000 students, 300 of which are in the third year,
    prove that when 20 students are selected using stratified sampling,
    14 first and second-year students participate in the activity. -/
theorem stratified_sampling_participation
  (total_students : ℕ) (third_year_students : ℕ) (selected_students : ℕ)
  (h_total : total_students = 1000)
  (h_third_year : third_year_students = 300)
  (h_selected : selected_students = 20) :
  (selected_students : ℚ) * (total_students - third_year_students : ℚ) / total_students = 14 :=
by sorry

end NUMINAMATH_CALUDE_stratified_sampling_participation_l1569_156960


namespace NUMINAMATH_CALUDE_relations_correctness_l1569_156901

-- Define the relations
def relation1 (a b c : ℝ) : Prop := (a > b) ↔ (a * c^2 > b * c^2)
def relation2 (a b : ℝ) : Prop := (a > b) → (1/a < 1/b)
def relation3 (a b c d : ℝ) : Prop := (a > b ∧ b > 0 ∧ c > d) → (a/d > b/c)
def relation4 (a b c : ℝ) : Prop := (a > b ∧ b > 1 ∧ c < 0) → (a^c < b^c)

-- State the theorem
theorem relations_correctness :
  (∃ a b c : ℝ, ¬(relation1 a b c)) ∧
  (∃ a b : ℝ, ¬(relation2 a b)) ∧
  (∃ a b c d : ℝ, ¬(relation3 a b c d)) ∧
  (∀ a b c : ℝ, relation4 a b c) :=
sorry

end NUMINAMATH_CALUDE_relations_correctness_l1569_156901


namespace NUMINAMATH_CALUDE_dense_S_l1569_156909

-- Define the set S
def S : Set ℝ := {x : ℝ | ∃ (m n : ℕ+), x = Real.sqrt m - Real.sqrt n}

-- State the theorem
theorem dense_S : ∀ (a b : ℝ), a < b → Set.Infinite (S ∩ Set.Ioo a b) := by sorry

end NUMINAMATH_CALUDE_dense_S_l1569_156909


namespace NUMINAMATH_CALUDE_borrowing_interest_rate_l1569_156912

/-- Proves that the interest rate at which a person borrowed money is 4% per annum,
    given the specified conditions. -/
theorem borrowing_interest_rate
  (loan_amount : ℝ)
  (loan_duration : ℕ)
  (lending_rate : ℝ)
  (yearly_gain : ℝ)
  (h1 : loan_amount = 7000)
  (h2 : loan_duration = 2)
  (h3 : lending_rate = 0.06)
  (h4 : yearly_gain = 140)
  : ∃ (borrowing_rate : ℝ), borrowing_rate = 0.04 := by
  sorry

end NUMINAMATH_CALUDE_borrowing_interest_rate_l1569_156912


namespace NUMINAMATH_CALUDE_line_moved_down_l1569_156976

/-- Given a line y = -x + 1 moved down 3 units, prove that the resulting line is y = -x - 2 -/
theorem line_moved_down (x y : ℝ) :
  (y = -x + 1) → (y - 3 = -x - 2) := by
  sorry

end NUMINAMATH_CALUDE_line_moved_down_l1569_156976


namespace NUMINAMATH_CALUDE_ratio_of_a_over_3_to_b_over_2_l1569_156984

theorem ratio_of_a_over_3_to_b_over_2 (a b c : ℝ) 
  (h1 : 2 * a = 3 * b) 
  (h2 : c ≠ 0) 
  (h3 : 3 * a + 2 * b = c) : 
  (a / 3) / (b / 2) = 1 := by
sorry

end NUMINAMATH_CALUDE_ratio_of_a_over_3_to_b_over_2_l1569_156984
