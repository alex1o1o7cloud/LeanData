import Mathlib

namespace NUMINAMATH_CALUDE_triangle_internal_point_theorem_l2600_260023

/-- Triangle with sides a, b, c and internal point P --/
structure TriangleWithInternalPoint where
  a : ℝ
  b : ℝ
  c : ℝ
  P : ℝ × ℝ

/-- Parallel segments through P have equal length d --/
def parallelSegmentsEqual (T : TriangleWithInternalPoint) (d : ℝ) : Prop :=
  ∃ (x y z : ℝ), x + y + z = T.a ∧ x + y + z = T.b ∧ x + y + z = T.c ∧ x = y ∧ y = z ∧ z = d

theorem triangle_internal_point_theorem (T : TriangleWithInternalPoint) 
    (h1 : T.a = 550) (h2 : T.b = 580) (h3 : T.c = 620) :
    ∃ (d : ℝ), parallelSegmentsEqual T d ∧ d = 342 := by
  sorry

end NUMINAMATH_CALUDE_triangle_internal_point_theorem_l2600_260023


namespace NUMINAMATH_CALUDE_ginas_expenses_theorem_l2600_260097

/-- Calculates Gina's total college expenses for the year --/
def ginasCollegeExpenses : ℕ :=
  let totalCredits : ℕ := 18
  let regularCredits : ℕ := 12
  let labCredits : ℕ := 6
  let regularCreditCost : ℕ := 450
  let labCreditCost : ℕ := 550
  let textbookCount : ℕ := 3
  let textbookCost : ℕ := 150
  let onlineResourceCount : ℕ := 4
  let onlineResourceCost : ℕ := 95
  let facilitiesFee : ℕ := 200
  let labFeePerCredit : ℕ := 75

  regularCredits * regularCreditCost +
  labCredits * labCreditCost +
  textbookCount * textbookCost +
  onlineResourceCount * onlineResourceCost +
  facilitiesFee +
  labCredits * labFeePerCredit

theorem ginas_expenses_theorem : ginasCollegeExpenses = 10180 := by
  sorry

end NUMINAMATH_CALUDE_ginas_expenses_theorem_l2600_260097


namespace NUMINAMATH_CALUDE_spatial_relationships_l2600_260046

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relationships between lines and planes
variable (perpendicular : Line → Plane → Prop)
variable (parallel_line_plane : Line → Plane → Prop)
variable (parallel_plane : Plane → Plane → Prop)

-- Define the perpendicular relationship between lines
variable (perpendicular_line : Line → Line → Prop)

theorem spatial_relationships 
  (m n : Line) (α β γ : Plane)
  (h_diff_lines : m ≠ n)
  (h_diff_planes : α ≠ β ∧ β ≠ γ ∧ α ≠ γ) :
  (perpendicular m α ∧ parallel_line_plane n α → perpendicular_line m n) ∧
  (parallel_plane α β ∧ parallel_plane β γ ∧ perpendicular m α → perpendicular m γ) :=
sorry

end NUMINAMATH_CALUDE_spatial_relationships_l2600_260046


namespace NUMINAMATH_CALUDE_golf_ball_distribution_returns_to_initial_state_l2600_260052

/-- Represents the distribution of golf balls in boxes and the next starting box -/
structure State :=
  (balls : Fin 10 → ℕ)
  (next_box : Fin 10)

/-- The set of all possible states -/
def S : Set State := {s | ∀ i, s.balls i > 0}

/-- Represents one move in the game -/
def move (s : State) : State :=
  sorry

theorem golf_ball_distribution_returns_to_initial_state :
  ∀ (initial : State),
  initial ∈ S →
  ∃ (n : ℕ+),
  (move^[n] initial) = initial :=
sorry

end NUMINAMATH_CALUDE_golf_ball_distribution_returns_to_initial_state_l2600_260052


namespace NUMINAMATH_CALUDE_no_solution_iff_n_eq_neg_two_l2600_260082

theorem no_solution_iff_n_eq_neg_two (n : ℝ) :
  (∀ x y z : ℝ, (n * x + y + z = 2 ∧ x + n * y + z = 2 ∧ x + y + n * z = 2) → False) ↔ n = -2 := by
  sorry

end NUMINAMATH_CALUDE_no_solution_iff_n_eq_neg_two_l2600_260082


namespace NUMINAMATH_CALUDE_unique_root_quadratic_l2600_260035

theorem unique_root_quadratic (c : ℝ) : 
  (∃ b : ℝ, b = c^2 + 1 ∧ 
   (∃! x : ℝ, x^2 + b*x + c = 0)) → 
  c = 1 := by
sorry

end NUMINAMATH_CALUDE_unique_root_quadratic_l2600_260035


namespace NUMINAMATH_CALUDE_sum_of_fractions_equals_one_l2600_260086

theorem sum_of_fractions_equals_one
  (a b c x y z : ℝ)
  (eq1 : 17 * x + b * y + c * z = 0)
  (eq2 : a * x + 29 * y + c * z = 0)
  (eq3 : a * x + b * y + 50 * z = 0)
  (ha : a ≠ 17)
  (hx : x ≠ 0) :
  a / (a - 17) + b / (b - 29) + c / (c - 50) = 1 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_fractions_equals_one_l2600_260086


namespace NUMINAMATH_CALUDE_problem_solution_l2600_260031

theorem problem_solution (a b c x : ℝ) 
  (h1 : a + x^2 = 2015)
  (h2 : b + x^2 = 2016)
  (h3 : c + x^2 = 2017)
  (h4 : a * b * c = 24) :
  a / (b * c) + b / (a * c) + c / (a * b) - 1 / a - 1 / b - 1 / c = 1 / 8 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l2600_260031


namespace NUMINAMATH_CALUDE_bug_position_after_2010_jumps_l2600_260027

/-- Represents the six points on the circle -/
inductive Point
| one
| two
| three
| four
| five
| six

/-- Determines if a point is odd-numbered -/
def is_odd (p : Point) : Bool :=
  match p with
  | Point.one => true
  | Point.three => true
  | Point.five => true
  | _ => false

/-- Performs one jump based on the current point -/
def jump (p : Point) : Point :=
  if is_odd p then
    match p with
    | Point.one => Point.two
    | Point.three => Point.four
    | Point.five => Point.six
    | _ => p  -- This case should never occur
  else
    match p with
    | Point.two => Point.five
    | Point.four => Point.one
    | Point.six => Point.three
    | _ => p  -- This case should never occur

/-- Performs n jumps starting from a given point -/
def multi_jump (start : Point) (n : Nat) : Point :=
  match n with
  | 0 => start
  | Nat.succ m => jump (multi_jump start m)

theorem bug_position_after_2010_jumps : 
  multi_jump Point.six 2010 = Point.two :=
by sorry

end NUMINAMATH_CALUDE_bug_position_after_2010_jumps_l2600_260027


namespace NUMINAMATH_CALUDE_mikes_pens_l2600_260009

theorem mikes_pens (initial_pens : ℕ) (final_pens : ℕ) : 
  initial_pens = 5 → final_pens = 40 → ∃ M : ℕ, 
    2 * (initial_pens + M) - 10 = final_pens ∧ M = 20 := by
  sorry

end NUMINAMATH_CALUDE_mikes_pens_l2600_260009


namespace NUMINAMATH_CALUDE_opposite_of_fraction_l2600_260090

theorem opposite_of_fraction (n : ℕ) (h : n ≠ 0) : 
  (-(1 : ℚ) / n) = -((1 : ℚ) / n) := by sorry

end NUMINAMATH_CALUDE_opposite_of_fraction_l2600_260090


namespace NUMINAMATH_CALUDE_hyperbola_eccentricity_l2600_260067

/-- A hyperbola with foci on the x-axis and asymptotic lines y = ±√3x has eccentricity 2 -/
theorem hyperbola_eccentricity (a b : ℝ) (h1 : a > 0) (h2 : b > 0) :
  (b / a = Real.sqrt 3) → 
  let c := Real.sqrt (a^2 + b^2)
  c / a = 2 := by sorry

end NUMINAMATH_CALUDE_hyperbola_eccentricity_l2600_260067


namespace NUMINAMATH_CALUDE_investment_ratio_l2600_260016

/-- Given two partners p and q, their profit ratio, and investment times, 
    prove the ratio of their investments. -/
theorem investment_ratio 
  (profit_ratio_p profit_ratio_q : ℚ) 
  (investment_time_p investment_time_q : ℚ) 
  (profit_ratio_constraint : profit_ratio_p / profit_ratio_q = 7 / 10)
  (time_constraint_p : investment_time_p = 8)
  (time_constraint_q : investment_time_q = 16) :
  ∃ (investment_p investment_q : ℚ),
    investment_p / investment_q = 7 / 5 ∧
    profit_ratio_p / profit_ratio_q = 
      (investment_p * investment_time_p) / (investment_q * investment_time_q) :=
by sorry

end NUMINAMATH_CALUDE_investment_ratio_l2600_260016


namespace NUMINAMATH_CALUDE_empty_seats_l2600_260088

theorem empty_seats (children : ℕ) (adults : ℕ) (total_seats : ℕ) : 
  children = 52 → adults = 29 → total_seats = 95 → 
  total_seats - (children + adults) = 14 := by
  sorry

end NUMINAMATH_CALUDE_empty_seats_l2600_260088


namespace NUMINAMATH_CALUDE_factorization_y_squared_minus_one_l2600_260064

/-- A factorization is valid if the expanded form equals the factored form -/
def IsValidFactorization (expanded factored : ℝ → ℝ) : Prop :=
  ∀ x, expanded x = factored x

/-- A factorization is from left to right if it's in the form of factors multiplied together -/
def IsFactorizationLeftToRight (f : ℝ → ℝ) : Prop :=
  ∃ (g h : ℝ → ℝ), ∀ x, f x = g x * h x

theorem factorization_y_squared_minus_one :
  IsValidFactorization (fun y => y^2 - 1) (fun y => (y + 1) * (y - 1)) ∧
  IsFactorizationLeftToRight (fun y => (y + 1) * (y - 1)) ∧
  ¬IsValidFactorization (fun x => x * (a - b)) (fun x => a*x - b*x) ∧
  ¬IsValidFactorization (fun x => x^2 - 2*x) (fun x => x * (x - 2/x)) ∧
  ¬IsFactorizationLeftToRight (fun x => x * (a + b) + c) :=
by sorry

end NUMINAMATH_CALUDE_factorization_y_squared_minus_one_l2600_260064


namespace NUMINAMATH_CALUDE_bus_ticket_cost_l2600_260093

theorem bus_ticket_cost 
  (total_tickets : ℕ)
  (senior_ticket_cost : ℕ)
  (total_sales : ℕ)
  (senior_tickets_sold : ℕ)
  (h1 : total_tickets = 65)
  (h2 : senior_ticket_cost = 10)
  (h3 : total_sales = 855)
  (h4 : senior_tickets_sold = 24) :
  (total_sales - senior_tickets_sold * senior_ticket_cost) / (total_tickets - senior_tickets_sold) = 15 :=
by sorry

end NUMINAMATH_CALUDE_bus_ticket_cost_l2600_260093


namespace NUMINAMATH_CALUDE_class_average_weight_l2600_260002

theorem class_average_weight (n₁ : ℕ) (n₂ : ℕ) (w₁ : ℝ) (w_total : ℝ) :
  n₁ = 16 →
  n₂ = 8 →
  w₁ = 50.25 →
  w_total = 48.55 →
  ((n₁ * w₁ + n₂ * ((n₁ + n₂) * w_total - n₁ * w₁) / n₂) / (n₁ + n₂) = w_total) →
  ((n₁ + n₂) * w_total - n₁ * w₁) / n₂ = 45.15 :=
by sorry

end NUMINAMATH_CALUDE_class_average_weight_l2600_260002


namespace NUMINAMATH_CALUDE_book_arrangement_and_distribution_l2600_260019

/-- The number of ways to arrange 5 books, including 2 mathematics books, in a row such that
    the mathematics books are not adjacent and not placed at both ends simultaneously. -/
def arrange_books : ℕ := 60

/-- The number of ways to distribute 5 books, including 2 mathematics books, to 3 students,
    with each student receiving at least 1 book. -/
def distribute_books : ℕ := 150

/-- Theorem stating the correct number of arrangements and distributions -/
theorem book_arrangement_and_distribution :
  arrange_books = 60 ∧ distribute_books = 150 := by
  sorry

end NUMINAMATH_CALUDE_book_arrangement_and_distribution_l2600_260019


namespace NUMINAMATH_CALUDE_point_on_line_l2600_260013

/-- Given five points O, A, B, C, D on a straight line and a point P between B and C,
    prove that OP = 1 + 4√3 under the given conditions. -/
theorem point_on_line (O A B C D P : ℝ) : 
  O < A ∧ A < B ∧ B < C ∧ C < D ∧  -- Points are in order on the line
  A - O = 1 ∧                      -- OA = 1
  B - O = 3 ∧                      -- OB = 3
  C - O = 5 ∧                      -- OC = 5
  D - O = 7 ∧                      -- OD = 7
  B < P ∧ P < C ∧                  -- P is between B and C
  (P - A) / (D - P) = 2 * (P - B) / (C - P)  -- AP : PD = 2(BP : PC)
  → P - O = 1 + 4 * Real.sqrt 3 := by
sorry

end NUMINAMATH_CALUDE_point_on_line_l2600_260013


namespace NUMINAMATH_CALUDE_polygon_diagonals_integer_l2600_260050

theorem polygon_diagonals_integer (n : ℕ) (h : n > 0) : ∃ k : ℤ, (n * (n - 3) : ℤ) / 2 = k := by
  sorry

end NUMINAMATH_CALUDE_polygon_diagonals_integer_l2600_260050


namespace NUMINAMATH_CALUDE_circle_distance_range_l2600_260005

theorem circle_distance_range (x y : ℝ) (h : x^2 + y^2 = 1) :
  3 - 2 * Real.sqrt 2 ≤ x^2 - 2*x + y^2 + 2*y + 2 ∧ 
  x^2 - 2*x + y^2 + 2*y + 2 ≤ 3 + 2 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_circle_distance_range_l2600_260005


namespace NUMINAMATH_CALUDE_triangle_inequality_ratio_three_fourths_is_optimal_l2600_260049

-- Define a triangle type
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  pos_a : 0 < a
  pos_b : 0 < b
  pos_c : 0 < c
  triangle_ineq_ab : c < a + b
  triangle_ineq_bc : a < b + c
  triangle_ineq_ca : b < c + a

-- Theorem statement
theorem triangle_inequality_ratio (t : Triangle) :
  (t.a^2 + t.b^2 + t.a * t.b) / t.c^2 ≥ (3/4 : ℝ) :=
sorry

-- Theorem for the optimality of the bound
theorem three_fourths_is_optimal :
  ∀ ε > 0, ∃ t : Triangle, (t.a^2 + t.b^2 + t.a * t.b) / t.c^2 < 3/4 + ε :=
sorry

end NUMINAMATH_CALUDE_triangle_inequality_ratio_three_fourths_is_optimal_l2600_260049


namespace NUMINAMATH_CALUDE_sqrt_two_times_two_minus_sqrt_two_sqrt_six_div_sqrt_three_times_sqrt_twentyfour_sum_of_square_roots_squared_difference_minus_product_l2600_260001

-- Problem 1
theorem sqrt_two_times_two_minus_sqrt_two :
  Real.sqrt 2 * (2 - Real.sqrt 2) = 2 * Real.sqrt 2 - 2 := by sorry

-- Problem 2
theorem sqrt_six_div_sqrt_three_times_sqrt_twentyfour :
  Real.sqrt 6 / Real.sqrt 3 * Real.sqrt 24 = 4 * Real.sqrt 3 := by sorry

-- Problem 3
theorem sum_of_square_roots :
  Real.sqrt 54 + Real.sqrt 24 - Real.sqrt 18 + 2 * Real.sqrt (1/2) = 5 * Real.sqrt 6 - 2 * Real.sqrt 2 := by sorry

-- Problem 4
theorem squared_difference_minus_product :
  (Real.sqrt 2 - 1)^2 - (Real.sqrt 5 + 2) * (Real.sqrt 5 - 2) = 2 - 2 * Real.sqrt 2 := by sorry

end NUMINAMATH_CALUDE_sqrt_two_times_two_minus_sqrt_two_sqrt_six_div_sqrt_three_times_sqrt_twentyfour_sum_of_square_roots_squared_difference_minus_product_l2600_260001


namespace NUMINAMATH_CALUDE_complex_number_coordinate_l2600_260000

theorem complex_number_coordinate : 
  let z : ℂ := 1 + (1 / Complex.I)
  (z.re = 1 ∧ z.im = -1) := by sorry

end NUMINAMATH_CALUDE_complex_number_coordinate_l2600_260000


namespace NUMINAMATH_CALUDE_parabola_focus_and_directrix_l2600_260047

-- Define the parabola equation
def parabola_equation (x y : ℝ) : Prop := x - 2 = (y - 3)^2

-- Define the focus of the parabola
def focus : ℝ × ℝ := (2.25, 3)

-- Define the directrix of the parabola
def directrix : ℝ → Prop := λ x => x = 1.75

theorem parabola_focus_and_directrix :
  ∀ x y : ℝ, parabola_equation x y →
  (∃ p : ℝ × ℝ, p = focus ∧ 
   ∀ q : ℝ × ℝ, parabola_equation q.1 q.2 → 
   (q.1 - p.1)^2 + (q.2 - p.2)^2 = (q.1 - 1.75)^2) ∧
  (∀ q : ℝ × ℝ, parabola_equation q.1 q.2 → 
   ∃ r : ℝ, directrix r ∧ 
   (q.1 - focus.1)^2 + (q.2 - focus.2)^2 = (q.1 - r)^2) :=
by sorry


end NUMINAMATH_CALUDE_parabola_focus_and_directrix_l2600_260047


namespace NUMINAMATH_CALUDE_systematic_sampling_probability_l2600_260011

/-- Represents a batch of parts with different classes -/
structure Batch :=
  (total : ℕ)
  (first_class : ℕ)
  (second_class : ℕ)
  (third_class : ℕ)

/-- Represents a sampling process -/
structure Sampling :=
  (batch : Batch)
  (sample_size : ℕ)

/-- The probability of selecting an individual part in systematic sampling -/
def selection_probability (s : Sampling) : ℚ :=
  s.sample_size / s.batch.total

/-- Theorem stating the probability of selecting each part in the given scenario -/
theorem systematic_sampling_probability (b : Batch) (s : Sampling) :
  b.total = 120 →
  b.first_class = 24 →
  b.second_class = 36 →
  b.third_class = 60 →
  s.batch = b →
  s.sample_size = 20 →
  selection_probability s = 1 / 6 := by
  sorry


end NUMINAMATH_CALUDE_systematic_sampling_probability_l2600_260011


namespace NUMINAMATH_CALUDE_carries_money_from_mom_l2600_260006

def sweater_cost : ℕ := 24
def tshirt_cost : ℕ := 6
def shoes_cost : ℕ := 11
def money_left : ℕ := 50

theorem carries_money_from_mom : 
  sweater_cost + tshirt_cost + shoes_cost + money_left = 91 := by
  sorry

end NUMINAMATH_CALUDE_carries_money_from_mom_l2600_260006


namespace NUMINAMATH_CALUDE_shape_perimeter_l2600_260057

theorem shape_perimeter (total_area : ℝ) (num_squares : ℕ) (h1 : total_area = 196) (h2 : num_squares = 4) :
  let side_length := Real.sqrt (total_area / num_squares)
  let perimeter := (num_squares + 1) * side_length + 2 * num_squares * side_length
  perimeter = 91 := by
sorry

end NUMINAMATH_CALUDE_shape_perimeter_l2600_260057


namespace NUMINAMATH_CALUDE_field_area_in_square_yards_l2600_260041

/-- Conversion rate from feet to yards -/
def feet_to_yard : ℝ := 3

/-- Length of the field in feet -/
def field_length_feet : ℝ := 12

/-- Width of the field in feet -/
def field_width_feet : ℝ := 9

/-- Theorem stating that the area of the field in square yards is 12 -/
theorem field_area_in_square_yards :
  (field_length_feet / feet_to_yard) * (field_width_feet / feet_to_yard) = 12 :=
by sorry

end NUMINAMATH_CALUDE_field_area_in_square_yards_l2600_260041


namespace NUMINAMATH_CALUDE_sport_participation_l2600_260032

theorem sport_participation (total : ℕ) (football : ℕ) (basketball : ℕ) (baseball : ℕ) (all_three : ℕ)
  (h1 : total = 427)
  (h2 : football = 128)
  (h3 : basketball = 291)
  (h4 : baseball = 318)
  (h5 : all_three = 36)
  (h6 : total = football + basketball + baseball - (football_basketball + football_baseball + basketball_baseball) + all_three)
  : football_basketball + football_baseball + basketball_baseball - 3 * all_three = 274 :=
by sorry

end NUMINAMATH_CALUDE_sport_participation_l2600_260032


namespace NUMINAMATH_CALUDE_math_problems_l2600_260084

theorem math_problems :
  (∀ a b : ℝ, a < b ∧ b < 0 → a^2 > a*b ∧ a*b > b^2) ∧
  (∀ a b c d : ℝ, c > d ∧ a > b → a - d > b - c) ∧
  (∀ a b c : ℝ, b < a ∧ a < 0 ∧ c < 0 → c/a > c/b) ∧
  (∀ a b c : ℝ, a > 0 ∧ b > c ∧ c > 0 → (c+a)/(b+a) > c/b) :=
by sorry

end NUMINAMATH_CALUDE_math_problems_l2600_260084


namespace NUMINAMATH_CALUDE_sum_of_roots_equals_one_l2600_260089

theorem sum_of_roots_equals_one :
  let f : ℝ → ℝ := λ x ↦ (x + 3) * (x - 4) - 20
  ∃ x₁ x₂ : ℝ, f x₁ = 0 ∧ f x₂ = 0 ∧ x₁ + x₂ = 1 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_roots_equals_one_l2600_260089


namespace NUMINAMATH_CALUDE_product_divisible_by_5184_l2600_260078

theorem product_divisible_by_5184 (k m : ℕ) : 
  5184 ∣ ((k^3 - 1) * k^3 * (k^3 + 1) * (m^3 - 1) * m^3 * (m^3 + 1)) := by
sorry

end NUMINAMATH_CALUDE_product_divisible_by_5184_l2600_260078


namespace NUMINAMATH_CALUDE_muscovy_duck_count_muscovy_duck_count_proof_l2600_260030

theorem muscovy_duck_count : ℕ → ℕ → ℕ → Prop :=
  fun muscovy cayuga khaki =>
    muscovy = cayuga + 4 ∧
    muscovy = 2 * cayuga + khaki + 3 ∧
    muscovy + cayuga + khaki = 90 →
    muscovy = 89

-- The proof is omitted
theorem muscovy_duck_count_proof : muscovy_duck_count 89 85 6 :=
  sorry

end NUMINAMATH_CALUDE_muscovy_duck_count_muscovy_duck_count_proof_l2600_260030


namespace NUMINAMATH_CALUDE_females_attending_correct_l2600_260042

/-- The number of females attending the meeting -/
def females_attending : ℕ := 50

/-- The total population of Nantucket -/
def total_population : ℕ := 300

/-- The number of people attending the meeting -/
def meeting_attendance : ℕ := total_population / 2

/-- The number of males attending the meeting -/
def males_attending : ℕ := 2 * females_attending

theorem females_attending_correct :
  females_attending = 50 ∧
  meeting_attendance = total_population / 2 ∧
  total_population = 300 ∧
  males_attending = 2 * females_attending ∧
  meeting_attendance = females_attending + males_attending :=
by sorry

end NUMINAMATH_CALUDE_females_attending_correct_l2600_260042


namespace NUMINAMATH_CALUDE_reduced_rates_fraction_l2600_260051

-- Define the number of days in a week
def days_in_week : ℕ := 7

-- Define the number of hours in a day
def hours_in_day : ℕ := 24

-- Define the number of weekdays (Monday to Friday)
def weekdays : ℕ := 5

-- Define the number of weekend days (Saturday and Sunday)
def weekend_days : ℕ := 2

-- Define the number of hours with reduced rates on weekdays (8 p.m. to 8 a.m.)
def reduced_hours_weekday : ℕ := 12

-- Define the number of hours with reduced rates on weekend days (24 hours)
def reduced_hours_weekend : ℕ := 24

-- Theorem stating that the fraction of a week with reduced rates is 9/14
theorem reduced_rates_fraction :
  (weekdays * reduced_hours_weekday + weekend_days * reduced_hours_weekend) / 
  (days_in_week * hours_in_day) = 9 / 14 := by
  sorry

end NUMINAMATH_CALUDE_reduced_rates_fraction_l2600_260051


namespace NUMINAMATH_CALUDE_expected_digits_icosahedral_die_l2600_260056

def icosahedralDie : Finset ℕ := Finset.range 20

theorem expected_digits_icosahedral_die :
  let E := (icosahedralDie.filter (λ n => n < 10)).card / 20 +
           2 * (icosahedralDie.filter (λ n => n ≥ 10)).card / 20
  E = 31 / 20 := by sorry

end NUMINAMATH_CALUDE_expected_digits_icosahedral_die_l2600_260056


namespace NUMINAMATH_CALUDE_toy_store_shelves_l2600_260072

/-- Calculates the number of shelves needed to store bears in a toy store. -/
def shelves_needed (initial_stock : ℕ) (new_shipment : ℕ) (bears_per_shelf : ℕ) : ℕ :=
  (initial_stock + new_shipment) / bears_per_shelf

/-- Proves that given the specific conditions, the number of shelves needed is 5. -/
theorem toy_store_shelves : shelves_needed 15 45 12 = 5 := by
  sorry

end NUMINAMATH_CALUDE_toy_store_shelves_l2600_260072


namespace NUMINAMATH_CALUDE_school_c_variance_l2600_260060

/-- Represents the data for a school's strong math foundation group -/
structure SchoolData where
  students : ℕ
  average : ℝ
  variance : ℝ

/-- Represents the overall data for all schools -/
structure OverallData where
  total_students : ℕ
  average : ℝ
  variance : ℝ

/-- Theorem stating that given the conditions, the variance of school C is 12 -/
theorem school_c_variance
  (ratio : Fin 3 → ℕ)
  (h_ratio : ratio = ![3, 2, 1])
  (overall : OverallData)
  (h_overall : overall = { total_students := 48, average := 117, variance := 21.5 })
  (school_a : SchoolData)
  (h_school_a : school_a = { students := 24, average := 118, variance := 15 })
  (school_b : SchoolData)
  (h_school_b : school_b = { students := 16, average := 114, variance := 21 })
  (school_c : SchoolData)
  (h_school_c_students : school_c.students = 8) :
  school_c.variance = 12 := by
  sorry

end NUMINAMATH_CALUDE_school_c_variance_l2600_260060


namespace NUMINAMATH_CALUDE_unique_two_digit_integer_l2600_260014

theorem unique_two_digit_integer (s : ℕ) : s ≥ 10 ∧ s < 100 ∧ (13 * s) % 100 = 42 ↔ s = 34 := by
  sorry

end NUMINAMATH_CALUDE_unique_two_digit_integer_l2600_260014


namespace NUMINAMATH_CALUDE_function_identity_l2600_260073

theorem function_identity (f : ℕ → ℕ) (h : ∀ n, f (n + 1) > f (f n)) : ∀ n, f n = n := by
  sorry

end NUMINAMATH_CALUDE_function_identity_l2600_260073


namespace NUMINAMATH_CALUDE_polynomial_simplification_l2600_260058

theorem polynomial_simplification (x : ℝ) :
  3 - 5*x - 7*x^2 + 9 + 11*x - 13*x^2 - 15 + 17*x + 19*x^2 + 2*x^3 =
  2*x^3 - x^2 + 23*x - 3 := by
sorry

end NUMINAMATH_CALUDE_polynomial_simplification_l2600_260058


namespace NUMINAMATH_CALUDE_point_in_second_quadrant_l2600_260096

theorem point_in_second_quadrant (A B C : ℝ) : 
  0 < A ∧ A < π/2 →  -- A is acute
  0 < B ∧ B < π/2 →  -- B is acute
  0 < C ∧ C < π/2 →  -- C is acute
  A + B + C = π →    -- A, B, C are angles of a triangle
  Real.cos B - Real.sin A < 0 ∧ Real.sin B - Real.cos A > 0 := by
sorry

end NUMINAMATH_CALUDE_point_in_second_quadrant_l2600_260096


namespace NUMINAMATH_CALUDE_candy_division_l2600_260044

theorem candy_division (total_candy : ℕ) (num_bags : ℕ) (candy_per_bag : ℕ) :
  total_candy = 42 →
  num_bags = 2 →
  candy_per_bag * num_bags = total_candy →
  candy_per_bag = 21 := by
  sorry

end NUMINAMATH_CALUDE_candy_division_l2600_260044


namespace NUMINAMATH_CALUDE_difference_of_sum_and_product_l2600_260076

theorem difference_of_sum_and_product (x y : ℝ) 
  (sum_eq : x + y = 30) 
  (prod_eq : x * y = 221) : 
  |x - y| = 4 := by sorry

end NUMINAMATH_CALUDE_difference_of_sum_and_product_l2600_260076


namespace NUMINAMATH_CALUDE_symmetric_points_line_intercept_l2600_260069

/-- Given two points A and B symmetric with respect to a line y = kx + b,
    prove that the x-intercept of the line is 5/6. -/
theorem symmetric_points_line_intercept 
  (A B : ℝ × ℝ) 
  (h_A : A = (1, 3)) 
  (h_B : B = (-2, 1)) 
  (k b : ℝ) 
  (h_symmetric : B = (2 * ((k * A.1 + b) / (1 + k^2) - k * A.2 / (1 + k^2)) - A.1,
                      2 * (k * (k * A.1 + b) / (1 + k^2) + A.2 / (1 + k^2)) - A.2)) :
  (- b / k : ℝ) = 5/6 := by
  sorry

end NUMINAMATH_CALUDE_symmetric_points_line_intercept_l2600_260069


namespace NUMINAMATH_CALUDE_remainder_of_645_l2600_260066

-- Define the set s
def s : Set ℕ := {n : ℕ | n > 0 ∧ ∃ k, n = 8 * k + 5}

-- Define the 81st element of s
def element_81 : ℕ := 645

-- Theorem statement
theorem remainder_of_645 : 
  element_81 ∈ s ∧ (∃ k : ℕ, element_81 = 8 * k + 5) :=
by sorry

end NUMINAMATH_CALUDE_remainder_of_645_l2600_260066


namespace NUMINAMATH_CALUDE_flu_transmission_rate_l2600_260036

theorem flu_transmission_rate : 
  ∃ x : ℝ, 
    x > 0 ∧ 
    (1 + x) + x * (1 + x) = 100 ∧ 
    x = 9 := by
  sorry

end NUMINAMATH_CALUDE_flu_transmission_rate_l2600_260036


namespace NUMINAMATH_CALUDE_similar_squares_side_length_l2600_260077

/-- Given two similar squares with an area ratio of 1:9 and the smaller square's side length of 5 cm,
    prove that the larger square's side length is 15 cm. -/
theorem similar_squares_side_length (small_side : ℝ) (large_side : ℝ) : 
  small_side = 5 →  -- The side length of the smaller square is 5 cm
  (large_side / small_side)^2 = 9 →  -- The ratio of their areas is 1:9
  large_side = 15 :=  -- The side length of the larger square is 15 cm
by
  sorry

end NUMINAMATH_CALUDE_similar_squares_side_length_l2600_260077


namespace NUMINAMATH_CALUDE_roots_polynomial_sum_l2600_260022

theorem roots_polynomial_sum (a b c : ℝ) (s : ℝ) : 
  (∀ x, x^3 - 9*x^2 + 11*x - 1 = 0 ↔ (x = a ∨ x = b ∨ x = c)) →
  s = Real.sqrt a + Real.sqrt b + Real.sqrt c →
  s^4 - 18*s^2 - 8*s = -37 := by
sorry

end NUMINAMATH_CALUDE_roots_polynomial_sum_l2600_260022


namespace NUMINAMATH_CALUDE_cookies_per_bag_l2600_260024

theorem cookies_per_bag (total_cookies : ℕ) (num_bags : ℕ) (cookies_per_bag : ℕ) 
  (h1 : total_cookies = 75)
  (h2 : num_bags = 25)
  (h3 : total_cookies = num_bags * cookies_per_bag) :
  cookies_per_bag = 3 := by
  sorry

end NUMINAMATH_CALUDE_cookies_per_bag_l2600_260024


namespace NUMINAMATH_CALUDE_sum_of_squares_l2600_260034

theorem sum_of_squares (a b : ℝ) : (a^2 + b^2) * (a^2 + b^2 - 4) - 5 = 0 → a^2 + b^2 = 5 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_squares_l2600_260034


namespace NUMINAMATH_CALUDE_min_pet_owners_l2600_260040

/-- Represents the number of people who own only dogs -/
def only_dogs : Nat := 15

/-- Represents the number of people who own only cats -/
def only_cats : Nat := 10

/-- Represents the number of people who own only cats and dogs -/
def cats_and_dogs : Nat := 5

/-- Represents the number of people who own cats, dogs, and snakes -/
def cats_dogs_snakes : Nat := 3

/-- Represents the total number of snakes -/
def total_snakes : Nat := 59

/-- Theorem stating that the minimum number of pet owners is 33 -/
theorem min_pet_owners : 
  only_dogs + only_cats + cats_and_dogs + cats_dogs_snakes = 33 := by
  sorry

#check min_pet_owners

end NUMINAMATH_CALUDE_min_pet_owners_l2600_260040


namespace NUMINAMATH_CALUDE_parabola_square_min_area_l2600_260037

/-- A square in a Cartesian plane with vertices on two parabolas -/
structure ParabolaSquare where
  /-- x-coordinate of a vertex on y = x^2 -/
  a : ℝ
  /-- The square's side length -/
  s : ℝ
  /-- Two opposite vertices lie on y = x^2 -/
  h1 : (a, a^2) ∈ {p : ℝ × ℝ | p.2 = p.1^2}
  h2 : (-a, a^2) ∈ {p : ℝ × ℝ | p.2 = p.1^2}
  /-- The other two opposite vertices lie on y = -x^2 + 4 -/
  h3 : (a, -a^2 + 4) ∈ {p : ℝ × ℝ | p.2 = -p.1^2 + 4}
  h4 : (-a, -a^2 + 4) ∈ {p : ℝ × ℝ | p.2 = -p.1^2 + 4}
  /-- The side length is the distance between vertices -/
  h5 : s^2 = (2*a)^2 + (2*a^2 - 4)^2

/-- The smallest possible area of the ParabolaSquare is 4 -/
theorem parabola_square_min_area :
  ∀ (ps : ParabolaSquare), ∃ (min_ps : ParabolaSquare), min_ps.s^2 = 4 ∧ ∀ (ps' : ParabolaSquare), ps'.s^2 ≥ 4 :=
sorry

end NUMINAMATH_CALUDE_parabola_square_min_area_l2600_260037


namespace NUMINAMATH_CALUDE_consecutive_integer_average_l2600_260029

theorem consecutive_integer_average (c d : ℤ) : 
  (∀ i : ℕ, i < 7 → c + i > 0) →
  d = (c + (c + 1) + (c + 2) + (c + 3) + (c + 4) + (c + 5) + (c + 6)) / 7 →
  ((d + (d + 1) + (d + 2) + (d + 3) + (d + 4) + (d + 5) + (d + 6)) / 7 : ℚ) = c + 6 := by
  sorry

end NUMINAMATH_CALUDE_consecutive_integer_average_l2600_260029


namespace NUMINAMATH_CALUDE_journey_distance_is_420_l2600_260085

/-- Represents the journey details -/
structure Journey where
  urban_speed : ℝ
  highway_speed : ℝ
  urban_time : ℝ
  highway_time : ℝ

/-- Calculates the total distance of the journey -/
def total_distance (j : Journey) : ℝ :=
  j.urban_speed * j.urban_time + j.highway_speed * j.highway_time

/-- Theorem stating that the journey distance is 420 km -/
theorem journey_distance_is_420 (j : Journey) 
  (h1 : j.urban_speed = 55)
  (h2 : j.highway_speed = 85)
  (h3 : j.urban_time = 3)
  (h4 : j.highway_time = 3) :
  total_distance j = 420 := by
  sorry

#eval total_distance { urban_speed := 55, highway_speed := 85, urban_time := 3, highway_time := 3 }

end NUMINAMATH_CALUDE_journey_distance_is_420_l2600_260085


namespace NUMINAMATH_CALUDE_bruces_shopping_money_l2600_260055

theorem bruces_shopping_money (initial_amount : ℕ) (shirt_cost : ℕ) (num_shirts : ℕ) (pants_cost : ℕ) (remaining_amount : ℕ) : 
  initial_amount = 71 →
  shirt_cost = 5 →
  num_shirts = 5 →
  pants_cost = 26 →
  remaining_amount = initial_amount - (num_shirts * shirt_cost + pants_cost) →
  remaining_amount = 20 := by
sorry

end NUMINAMATH_CALUDE_bruces_shopping_money_l2600_260055


namespace NUMINAMATH_CALUDE_first_student_stickers_l2600_260026

/-- Given a sequence of gold sticker counts for students 2 to 6, 
    prove that the first student received 29 stickers. -/
theorem first_student_stickers 
  (second : ℕ) 
  (third : ℕ) 
  (fourth : ℕ) 
  (fifth : ℕ) 
  (sixth : ℕ) 
  (h1 : second = 35) 
  (h2 : third = 41) 
  (h3 : fourth = 47) 
  (h4 : fifth = 53) 
  (h5 : sixth = 59) : 
  second - 6 = 29 := by
  sorry

end NUMINAMATH_CALUDE_first_student_stickers_l2600_260026


namespace NUMINAMATH_CALUDE_alice_purse_value_l2600_260020

-- Define the values of coins in cents
def penny : ℕ := 1
def dime : ℕ := 10
def quarter : ℕ := 25
def half_dollar : ℕ := 50

-- Define the total value of coins in Alice's purse
def purse_value : ℕ := penny + dime + quarter + half_dollar

-- Define one dollar in cents
def one_dollar : ℕ := 100

-- Theorem statement
theorem alice_purse_value :
  (purse_value : ℚ) / one_dollar = 86 / 100 := by sorry

end NUMINAMATH_CALUDE_alice_purse_value_l2600_260020


namespace NUMINAMATH_CALUDE_group_collection_l2600_260071

/-- Calculates the total collection in rupees for a group contribution -/
def total_collection (num_members : ℕ) : ℚ :=
  (num_members * num_members : ℚ) / 100

/-- Theorem stating that for a group of 93 members, the total collection is 86.49 rupees -/
theorem group_collection :
  total_collection 93 = 86.49 := by
  sorry

end NUMINAMATH_CALUDE_group_collection_l2600_260071


namespace NUMINAMATH_CALUDE_sin_690_degrees_l2600_260017

theorem sin_690_degrees : Real.sin (690 * π / 180) = -1/2 := by sorry

end NUMINAMATH_CALUDE_sin_690_degrees_l2600_260017


namespace NUMINAMATH_CALUDE_intersection_M_N_l2600_260053

def M : Set (ℝ × ℝ) := {p | (p.1^2 / 9) + (p.2^2 / 4) = 1}

def N : Set (ℝ × ℝ) := {p | (p.1 / 3) + (p.2 / 2) = 1}

theorem intersection_M_N : M ∩ N = {(3, 0), (0, 2)} := by sorry

end NUMINAMATH_CALUDE_intersection_M_N_l2600_260053


namespace NUMINAMATH_CALUDE_product_remainder_l2600_260033

theorem product_remainder (x y : ℤ) 
  (hx : x % 315 = 53) 
  (hy : y % 385 = 41) : 
  (x * y) % 21 = 10 := by
  sorry

end NUMINAMATH_CALUDE_product_remainder_l2600_260033


namespace NUMINAMATH_CALUDE_geometric_sequence_first_term_l2600_260068

/-- Given a geometric sequence where the fourth term is 32 and the fifth term is 64, prove that the first term is 4. -/
theorem geometric_sequence_first_term (a b c : ℝ) : 
  (∃ r : ℝ, r ≠ 0 ∧ b = a * r ∧ c = b * r ∧ 32 = c * r ∧ 64 = 32 * r) → a = 4 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_first_term_l2600_260068


namespace NUMINAMATH_CALUDE_inverse_f_at_negative_31_96_l2600_260094

noncomputable def f (x : ℝ) : ℝ := (x^5 - 1) / 3

theorem inverse_f_at_negative_31_96 : f⁻¹ (-31/96) = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_inverse_f_at_negative_31_96_l2600_260094


namespace NUMINAMATH_CALUDE_smallest_number_of_students_l2600_260048

theorem smallest_number_of_students (grade12 grade11 grade10 : ℕ) : 
  grade12 > 0 ∧ grade11 > 0 ∧ grade10 > 0 →
  grade12 * 3 = grade10 * 4 →
  grade12 * 5 = grade11 * 7 →
  grade11 * 9 = grade10 * 10 →
  grade12 + grade11 + grade10 ≥ 66 :=
by
  sorry

end NUMINAMATH_CALUDE_smallest_number_of_students_l2600_260048


namespace NUMINAMATH_CALUDE_no_prime_sum_for_10003_l2600_260081

/-- A function that returns the number of ways to write a natural number as the sum of two primes -/
def count_prime_sum_representations (n : ℕ) : ℕ :=
  sorry

/-- The theorem stating that 10003 cannot be written as the sum of two primes -/
theorem no_prime_sum_for_10003 : count_prime_sum_representations 10003 = 0 := by
  sorry

end NUMINAMATH_CALUDE_no_prime_sum_for_10003_l2600_260081


namespace NUMINAMATH_CALUDE_gemstones_for_four_sets_l2600_260045

/-- The number of gemstones needed for a given number of earring sets -/
def gemstones_needed (num_sets : ℕ) : ℕ :=
  let magnets_per_earring : ℕ := 2
  let buttons_per_earring : ℕ := magnets_per_earring / 2
  let gemstones_per_earring : ℕ := buttons_per_earring * 3
  let earrings_per_set : ℕ := 2
  num_sets * earrings_per_set * gemstones_per_earring

/-- Theorem: 4 sets of earrings require 24 gemstones -/
theorem gemstones_for_four_sets : gemstones_needed 4 = 24 := by
  sorry

end NUMINAMATH_CALUDE_gemstones_for_four_sets_l2600_260045


namespace NUMINAMATH_CALUDE_simplify_expression_l2600_260061

theorem simplify_expression (a b c : ℝ) 
  (ha : a > 0) (hb : b < 0) (hc : c < 0) 
  (hab : abs a > abs b) (hca : abs c > abs a) : 
  abs (a + c) - abs (b + c) - abs (a + b) = -2 * a := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l2600_260061


namespace NUMINAMATH_CALUDE_floor_sqrt_120_l2600_260070

theorem floor_sqrt_120 : ⌊Real.sqrt 120⌋ = 10 := by
  sorry

end NUMINAMATH_CALUDE_floor_sqrt_120_l2600_260070


namespace NUMINAMATH_CALUDE_complex_symmetry_product_l2600_260095

theorem complex_symmetry_product (z₁ z₂ : ℂ) : 
  (z₁.im = -z₂.im) → (z₁.re = z₂.re) → (z₁ ≠ 1 + I) → z₁ * z₂ = 2 := by
  sorry

end NUMINAMATH_CALUDE_complex_symmetry_product_l2600_260095


namespace NUMINAMATH_CALUDE_park_area_l2600_260075

/-- Represents a rectangular park with sides in ratio 3:2 -/
structure Park where
  x : ℝ
  length : ℝ := 3 * x
  width : ℝ := 2 * x

/-- The perimeter of the park -/
def Park.perimeter (p : Park) : ℝ := 2 * (p.length + p.width)

/-- The area of the park -/
def Park.area (p : Park) : ℝ := p.length * p.width

/-- The cost of fencing per meter in rupees -/
def fencing_cost_per_meter : ℝ := 0.50

/-- The total cost of fencing the park in rupees -/
def total_fencing_cost : ℝ := 175

theorem park_area (p : Park) : 
  p.perimeter * fencing_cost_per_meter = total_fencing_cost → 
  p.area = 7350 := by
  sorry

#check park_area

end NUMINAMATH_CALUDE_park_area_l2600_260075


namespace NUMINAMATH_CALUDE_angle_sum_quadrilateral_l2600_260028

theorem angle_sum_quadrilateral (a b : ℝ) : 
  36 + b + 44 + 52 = 180 → b = 48 := by
  sorry

end NUMINAMATH_CALUDE_angle_sum_quadrilateral_l2600_260028


namespace NUMINAMATH_CALUDE_songs_leftover_l2600_260038

theorem songs_leftover (total_songs : ℕ) (num_playlists : ℕ) (h1 : total_songs = 372) (h2 : num_playlists = 9) :
  total_songs % num_playlists = 3 := by
  sorry

end NUMINAMATH_CALUDE_songs_leftover_l2600_260038


namespace NUMINAMATH_CALUDE_min_m_intersection_nonempty_l2600_260063

def set_B (m : ℝ) : Set (ℝ × ℝ) := {p | 3 * p.1 + 2 * p.2 - m = 0}

theorem min_m_intersection_nonempty (A : Set (ℝ × ℝ)) (h : ∃ m : ℝ, (A ∩ set_B m).Nonempty) :
  ∃ m_min : ℝ, m_min = 0 ∧ (A ∩ set_B m_min).Nonempty ∧ ∀ m : ℝ, (A ∩ set_B m).Nonempty → m ≥ m_min :=
by
  sorry

end NUMINAMATH_CALUDE_min_m_intersection_nonempty_l2600_260063


namespace NUMINAMATH_CALUDE_linear_function_shift_l2600_260004

/-- Given a linear function y = mx - 1, prove that if the graph is shifted down by 2 units
    and passes through the point (-2, 1), then m = -2. -/
theorem linear_function_shift (m : ℝ) : 
  (∀ x y : ℝ, y = m * x - 3 → (x = -2 ∧ y = 1)) → m = -2 := by
  sorry

end NUMINAMATH_CALUDE_linear_function_shift_l2600_260004


namespace NUMINAMATH_CALUDE_zero_is_rational_l2600_260039

/-- A number is rational if it can be expressed as the quotient of two integers with a non-zero denominator -/
def IsRational (x : ℚ) : Prop := ∃ (p q : ℤ), q ≠ 0 ∧ x = p / q

/-- Theorem: Zero is a rational number -/
theorem zero_is_rational : IsRational 0 := by
  sorry

end NUMINAMATH_CALUDE_zero_is_rational_l2600_260039


namespace NUMINAMATH_CALUDE_sum_of_coefficients_l2600_260054

noncomputable def g (x : ℝ) (A B C : ℤ) : ℝ := x^2 / (A * x^2 + B * x + C)

theorem sum_of_coefficients 
  (A B C : ℤ) 
  (h1 : ∀ x > 5, g x A B C > 0.5)
  (h2 : (A : ℝ) * (-3)^2 + B * (-3) + C = 0)
  (h3 : (A : ℝ) * 4^2 + B * 4 + C = 0) :
  A + B + C = -24 := by sorry

end NUMINAMATH_CALUDE_sum_of_coefficients_l2600_260054


namespace NUMINAMATH_CALUDE_scooter_gain_percent_l2600_260003

/-- Calculate the gain percent for a scooter sale -/
theorem scooter_gain_percent (purchase_price repair_costs selling_price : ℝ) 
  (h1 : purchase_price = 4700)
  (h2 : repair_costs = 800)
  (h3 : selling_price = 6000) : 
  (selling_price - (purchase_price + repair_costs)) / (purchase_price + repair_costs) * 100 = 9.09 :=
by sorry

end NUMINAMATH_CALUDE_scooter_gain_percent_l2600_260003


namespace NUMINAMATH_CALUDE_inscribed_square_in_acute_triangle_l2600_260091

/-- A triangle is acute-angled if all its angles are less than 90 degrees -/
def IsAcuteAngledTriangle (A B C : Point) : Prop := sorry

/-- A square is inscribed in a triangle if all its vertices lie on the sides of the triangle -/
def IsInscribedSquare (K L M N : Point) (A B C : Point) : Prop := sorry

/-- Two points lie on the same side of a triangle -/
def LieOnSameSide (P Q : Point) (A B C : Point) : Prop := sorry

theorem inscribed_square_in_acute_triangle 
  (A B C : Point) (h : IsAcuteAngledTriangle A B C) :
  ∃ (K L M N : Point), 
    IsInscribedSquare K L M N A B C ∧ 
    LieOnSameSide L M A B C ∧
    ((LieOnSameSide K N A B C ∧ ¬LieOnSameSide K N B C A) ∨
     (LieOnSameSide K N B C A ∧ ¬LieOnSameSide K N A B C)) :=
sorry

end NUMINAMATH_CALUDE_inscribed_square_in_acute_triangle_l2600_260091


namespace NUMINAMATH_CALUDE_function_inequality_and_sum_inequality_l2600_260015

-- Define the function f
def f (x : ℝ) : ℝ := |x - 3| + |x + 2|

-- Define the theorem
theorem function_inequality_and_sum_inequality :
  (∀ x m : ℝ, f x ≥ |m + 1|) →
  (∃ M : ℝ, M = 4 ∧
    (∀ m : ℝ, (∀ x : ℝ, f x ≥ |m + 1|) → m ≤ M) ∧
    (∀ a b c : ℝ, a > 0 → b > 0 → c > 0 → a + 2*b + c = M →
      1 / (a + b) + 1 / (b + c) ≥ 1)) :=
by sorry

end NUMINAMATH_CALUDE_function_inequality_and_sum_inequality_l2600_260015


namespace NUMINAMATH_CALUDE_silverware_probability_l2600_260008

/-- The probability of selecting one fork, one spoon, and one knife when
    randomly removing three pieces of silverware from a drawer. -/
theorem silverware_probability (forks spoons knives : ℕ) 
  (h1 : forks = 6)
  (h2 : spoons = 8)
  (h3 : knives = 6) :
  (forks * spoons * knives : ℚ) / (Nat.choose (forks + spoons + knives) 3) = 24 / 95 :=
by sorry

end NUMINAMATH_CALUDE_silverware_probability_l2600_260008


namespace NUMINAMATH_CALUDE_sequential_structure_essential_l2600_260062

/-- Represents the different types of algorithm structures -/
inductive AlgorithmStructure
  | Logical
  | Selection
  | Loop
  | Sequential

/-- Represents an algorithm -/
structure Algorithm where
  structures : List AlgorithmStructure

/-- Defines what it means for a structure to be essential for all algorithms -/
def isEssentialStructure (s : AlgorithmStructure) : Prop :=
  ∀ (a : Algorithm), s ∈ a.structures

/-- States that an algorithm can exist without Logical, Selection, or Loop structures -/
axiom non_essential_structures :
  ∃ (a : Algorithm),
    AlgorithmStructure.Logical ∉ a.structures ∧
    AlgorithmStructure.Selection ∉ a.structures ∧
    AlgorithmStructure.Loop ∉ a.structures

/-- The main theorem: Sequential structure is the only essential structure -/
theorem sequential_structure_essential :
  isEssentialStructure AlgorithmStructure.Sequential ∧
  (∀ s : AlgorithmStructure, s ≠ AlgorithmStructure.Sequential → ¬isEssentialStructure s) :=
sorry

end NUMINAMATH_CALUDE_sequential_structure_essential_l2600_260062


namespace NUMINAMATH_CALUDE_simplify_fraction_l2600_260080

theorem simplify_fraction (b : ℚ) (h : b = 2) : 15 * b^4 / (45 * b^3) = 2 / 3 := by
  sorry

end NUMINAMATH_CALUDE_simplify_fraction_l2600_260080


namespace NUMINAMATH_CALUDE_token_game_result_l2600_260059

def iterate_operation (n : ℕ) (f : ℤ → ℤ) (initial : ℤ) : ℤ :=
  match n with
  | 0 => initial
  | m + 1 => f (iterate_operation m f initial)

theorem token_game_result :
  let square (x : ℤ) := x * x
  let cube (x : ℤ) := x * x * x
  let iterations := 50
  let token1 := iterate_operation iterations square 2
  let token2 := iterate_operation iterations cube (-2)
  let token3 := iterate_operation iterations square 0
  token1 + token2 + token3 = -496 := by
  sorry

end NUMINAMATH_CALUDE_token_game_result_l2600_260059


namespace NUMINAMATH_CALUDE_lawrence_county_kids_at_home_l2600_260092

/-- The number of kids from Lawrence county who stay home during summer break -/
def kids_stay_home (total_kids : ℕ) (kids_at_camp : ℕ) : ℕ :=
  total_kids - kids_at_camp

/-- Proof that 590796 kids from Lawrence county stay home during summer break -/
theorem lawrence_county_kids_at_home : 
  kids_stay_home 1201565 610769 = 590796 := by
  sorry

end NUMINAMATH_CALUDE_lawrence_county_kids_at_home_l2600_260092


namespace NUMINAMATH_CALUDE_min_value_of_f_l2600_260018

/-- The quadratic function f(x) = x^2 + 16x + 20 -/
def f (x : ℝ) : ℝ := x^2 + 16*x + 20

/-- The minimum value of f(x) is -44 -/
theorem min_value_of_f :
  ∃ (m : ℝ), (∀ x, f x ≥ m) ∧ (∃ x, f x = m) ∧ (m = -44) := by
  sorry

end NUMINAMATH_CALUDE_min_value_of_f_l2600_260018


namespace NUMINAMATH_CALUDE_midpoint_distance_theorem_l2600_260074

theorem midpoint_distance_theorem (t : ℝ) : 
  let P : ℝ × ℝ := (t - 5, -2)
  let Q : ℝ × ℝ := (-3, t + 4)
  let midpoint : ℝ × ℝ := ((P.1 + Q.1) / 2, (P.2 + Q.2) / 2)
  let midpoint_to_endpoint_sq := ((midpoint.1 - P.1)^2 + (midpoint.2 - P.2)^2)
  midpoint_to_endpoint_sq = t^2 / 3 →
  t = -12 - 2 * Real.sqrt 21 ∨ t = -12 + 2 * Real.sqrt 21 :=
by sorry

end NUMINAMATH_CALUDE_midpoint_distance_theorem_l2600_260074


namespace NUMINAMATH_CALUDE_inequality_solution_l2600_260007

theorem inequality_solution (x : ℝ) : 
  2 / (x + 2) + 4 / (x + 8) ≥ 1 / 2 ↔ x ∈ Set.Ioc (-8) (-4) ∪ Set.Icc (-2) 2 :=
sorry

end NUMINAMATH_CALUDE_inequality_solution_l2600_260007


namespace NUMINAMATH_CALUDE_jesses_friends_l2600_260012

theorem jesses_friends (bananas_per_friend : ℝ) (total_bananas : ℕ) 
  (h1 : bananas_per_friend = 21.0) 
  (h2 : total_bananas = 63) : 
  (total_bananas : ℝ) / bananas_per_friend = 3 := by
  sorry

end NUMINAMATH_CALUDE_jesses_friends_l2600_260012


namespace NUMINAMATH_CALUDE_physics_marks_calculation_l2600_260079

def english_marks : ℕ := 91
def math_marks : ℕ := 65
def chemistry_marks : ℕ := 67
def biology_marks : ℕ := 85
def total_subjects : ℕ := 5
def average_marks : ℕ := 78

theorem physics_marks_calculation :
  let known_marks := english_marks + math_marks + chemistry_marks + biology_marks
  let total_marks := average_marks * total_subjects
  total_marks - known_marks = 82 := by
  sorry

end NUMINAMATH_CALUDE_physics_marks_calculation_l2600_260079


namespace NUMINAMATH_CALUDE_tangent_line_circle_range_l2600_260099

theorem tangent_line_circle_range (m n : ℝ) : 
  (∃ (x y : ℝ), (m + 1) * x + (n + 1) * y - 2 = 0 ∧ 
   (x - 1)^2 + (y - 1)^2 = 1 ∧ 
   ∀ (x' y' : ℝ), (m + 1) * x' + (n + 1) * y' - 2 = 0 → (x' - 1)^2 + (y' - 1)^2 ≥ 1) →
  m + n ≤ 2 - 2 * Real.sqrt 2 ∨ m + n ≥ 2 + 2 * Real.sqrt 2 :=
by sorry

end NUMINAMATH_CALUDE_tangent_line_circle_range_l2600_260099


namespace NUMINAMATH_CALUDE_intersection_M_N_l2600_260021

def M : Set ℝ := {x | 4 < x ∧ x < 8}
def N : Set ℝ := {x | x^2 - 6*x < 0}

theorem intersection_M_N : M ∩ N = {x | 4 < x ∧ x < 6} := by sorry

end NUMINAMATH_CALUDE_intersection_M_N_l2600_260021


namespace NUMINAMATH_CALUDE_triangle_existence_l2600_260087

/-- A set of points in space -/
structure PointSet where
  n : ℕ
  points : Finset (Fin (2 * n))
  segments : Finset (Fin (2 * n) × Fin (2 * n))
  n_gt_one : n > 1
  segment_count : segments.card ≥ n^2 + 1

/-- A triangle in a point set -/
def Triangle (ps : PointSet) : Prop :=
  ∃ a b c, a ∈ ps.points ∧ b ∈ ps.points ∧ c ∈ ps.points ∧
    (a, b) ∈ ps.segments ∧ (b, c) ∈ ps.segments ∧ (c, a) ∈ ps.segments

/-- Theorem: If a point set satisfies the conditions, then it contains a triangle -/
theorem triangle_existence (ps : PointSet) : Triangle ps := by
  sorry

end NUMINAMATH_CALUDE_triangle_existence_l2600_260087


namespace NUMINAMATH_CALUDE_rectangle_composition_l2600_260098

/-- The side length of the middle square in a specific rectangular arrangement -/
def square_side_length : ℝ := by sorry

theorem rectangle_composition (total_width total_height : ℝ) 
  (h_width : total_width = 3500)
  (h_height : total_height = 2100)
  (h_composition : ∃ (r : ℝ), 2 * r + square_side_length = total_height ∧ 
                               (square_side_length + 100) + square_side_length + (square_side_length + 200) = total_width) :
  square_side_length = 1066.67 := by sorry

end NUMINAMATH_CALUDE_rectangle_composition_l2600_260098


namespace NUMINAMATH_CALUDE_sin_alpha_plus_pi_12_l2600_260043

theorem sin_alpha_plus_pi_12 (α : ℝ) 
  (h1 : α ∈ Set.Ioo (-π/3) 0)
  (h2 : Real.cos (α + π/6) - Real.sin α = 4*Real.sqrt 3/5) :
  Real.sin (α + π/12) = -Real.sqrt 2/10 := by sorry

end NUMINAMATH_CALUDE_sin_alpha_plus_pi_12_l2600_260043


namespace NUMINAMATH_CALUDE_berry_theorem_l2600_260010

def berry_problem (total_needed : ℕ) (strawberries : ℕ) (blueberries : ℕ) : ℕ :=
  total_needed - (strawberries + blueberries)

theorem berry_theorem : berry_problem 26 10 9 = 7 := by
  sorry

end NUMINAMATH_CALUDE_berry_theorem_l2600_260010


namespace NUMINAMATH_CALUDE_parallel_angles_theorem_l2600_260025

/-- Given two angles A and B where the sides of A are parallel to the sides of B, 
    prove that if B = 3A - 60°, then B is either 30° or 120° -/
theorem parallel_angles_theorem (A B : ℝ) : 
  (B = 3 * A - 60) → (B = 30 ∨ B = 120) := by
  sorry

end NUMINAMATH_CALUDE_parallel_angles_theorem_l2600_260025


namespace NUMINAMATH_CALUDE_permutation_equation_solution_l2600_260065

def A (n : ℕ) : ℕ := n * (n - 1)

theorem permutation_equation_solution :
  ∃ (x : ℕ), 3 * (A (x + 1))^3 = 2 * (A (x + 2))^2 + 6 * (A (x + 1))^2 ∧ x = 4 := by
  sorry

end NUMINAMATH_CALUDE_permutation_equation_solution_l2600_260065


namespace NUMINAMATH_CALUDE_inequality_solution_set_l2600_260083

theorem inequality_solution_set (x : ℝ) : (2 * x - 1) / (3 * x + 1) > 1 ↔ -2 < x ∧ x < 1/3 := by
  sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l2600_260083
