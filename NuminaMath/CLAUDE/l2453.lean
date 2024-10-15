import Mathlib

namespace NUMINAMATH_CALUDE_two_moments_theorem_l2453_245320

/-- Represents a reader visiting the library -/
structure Reader where
  id : Nat

/-- Represents a moment in time -/
structure Moment where
  time : Nat

/-- Represents the state of a reader being in the library at a given moment -/
def ReaderPresent (r : Reader) (m : Moment) : Prop := sorry

/-- The library visit condition: each reader visits only once -/
axiom visit_once (r : Reader) :
  ∃! m : Moment, ReaderPresent r m

/-- The meeting condition: among any three readers, two meet each other -/
axiom meet_condition (r1 r2 r3 : Reader) :
  ∃ m : Moment, (ReaderPresent r1 m ∧ ReaderPresent r2 m) ∨
                (ReaderPresent r1 m ∧ ReaderPresent r3 m) ∨
                (ReaderPresent r2 m ∧ ReaderPresent r3 m)

/-- The main theorem: there exist two moments such that every reader is present at least at one of them -/
theorem two_moments_theorem (readers : Set Reader) :
  ∃ m1 m2 : Moment, ∀ r ∈ readers, ReaderPresent r m1 ∨ ReaderPresent r m2 :=
sorry

end NUMINAMATH_CALUDE_two_moments_theorem_l2453_245320


namespace NUMINAMATH_CALUDE_study_time_difference_l2453_245388

/-- Converts hours to minutes -/
def hoursToMinutes (hours : ℚ) : ℚ := hours * 60

/-- Converts days to minutes -/
def daysToMinutes (days : ℚ) : ℚ := days * 24 * 60

theorem study_time_difference :
  let kwame := hoursToMinutes 2.5
  let connor := hoursToMinutes 1.5
  let lexia := 97
  let michael := hoursToMinutes 3 + 45
  let cassandra := 165
  let aria := daysToMinutes 0.5
  (lexia + aria) - (kwame + connor + michael + cassandra) = 187 := by sorry

end NUMINAMATH_CALUDE_study_time_difference_l2453_245388


namespace NUMINAMATH_CALUDE_simplify_fraction_l2453_245313

theorem simplify_fraction (x : ℝ) (h : x ≠ 1) :
  (x^2 + 1) / (x - 1) - 2*x / (x - 1) = x - 1 := by
  sorry

end NUMINAMATH_CALUDE_simplify_fraction_l2453_245313


namespace NUMINAMATH_CALUDE_james_meditation_sessions_l2453_245373

/-- Calculates the number of meditation sessions per day given the session duration and weekly meditation time. -/
def meditation_sessions_per_day (session_duration : ℕ) (weekly_meditation_hours : ℕ) : ℕ :=
  let minutes_per_week : ℕ := weekly_meditation_hours * 60
  let minutes_per_day : ℕ := minutes_per_week / 7
  minutes_per_day / session_duration

/-- Theorem stating that given the specified conditions, the number of meditation sessions per day is 2. -/
theorem james_meditation_sessions :
  meditation_sessions_per_day 30 7 = 2 :=
by sorry

end NUMINAMATH_CALUDE_james_meditation_sessions_l2453_245373


namespace NUMINAMATH_CALUDE_fraction_addition_simplest_form_l2453_245323

theorem fraction_addition : 8 / 15 + 7 / 10 = 37 / 30 := by sorry

theorem simplest_form : ∀ n m : ℕ, n ≠ 0 → m ≠ 0 → Nat.gcd n m = 1 → (n : ℚ) / m = 37 / 30 → n = 37 ∧ m = 30 := by sorry

end NUMINAMATH_CALUDE_fraction_addition_simplest_form_l2453_245323


namespace NUMINAMATH_CALUDE_triangle_side_b_l2453_245347

theorem triangle_side_b (a c S : ℝ) (h1 : a = 5) (h2 : c = 2) (h3 : S = 4) :
  ∃ b : ℝ, (b = Real.sqrt 17 ∨ b = Real.sqrt 41) ∧
    S = (1/2) * a * c * Real.sqrt (1 - ((a^2 + c^2 - b^2) / (2*a*c))^2) :=
sorry

end NUMINAMATH_CALUDE_triangle_side_b_l2453_245347


namespace NUMINAMATH_CALUDE_division_problem_l2453_245330

theorem division_problem (x y z : ℚ) 
  (h1 : x / y = 3)
  (h2 : y / z = 5 / 2) :
  z / x = 2 / 15 := by
  sorry

end NUMINAMATH_CALUDE_division_problem_l2453_245330


namespace NUMINAMATH_CALUDE_probability_theorem_l2453_245389

/-- The probability of opening all safes given the number of keys and safes -/
def probability_open_all_safes (k n : ℕ) : ℚ :=
  if n > k then k / n else 1

/-- The theorem stating the probability of opening all safes -/
theorem probability_theorem (k n : ℕ) (h : n > k) :
  probability_open_all_safes k n = k / n := by
  sorry

#eval probability_open_all_safes 2 94

end NUMINAMATH_CALUDE_probability_theorem_l2453_245389


namespace NUMINAMATH_CALUDE_alice_least_money_l2453_245394

-- Define the set of people
inductive Person : Type
  | Alice : Person
  | Bob : Person
  | Charlie : Person
  | Dana : Person
  | Eve : Person

-- Define the money function
variable (money : Person → ℝ)

-- Define the conditions
axiom different_amounts : ∀ (p q : Person), p ≠ q → money p ≠ money q

axiom charlie_most : ∀ (p : Person), p ≠ Person.Charlie → money p < money Person.Charlie

axiom bob_more_than_alice : money Person.Alice < money Person.Bob

axiom dana_more_than_alice : money Person.Alice < money Person.Dana

axiom eve_between_alice_and_bob : 
  money Person.Alice < money Person.Eve ∧ money Person.Eve < money Person.Bob

-- State the theorem
theorem alice_least_money : 
  ∀ (p : Person), p ≠ Person.Alice → money Person.Alice < money p := by
  sorry

end NUMINAMATH_CALUDE_alice_least_money_l2453_245394


namespace NUMINAMATH_CALUDE_largest_class_size_l2453_245379

theorem largest_class_size (num_classes : ℕ) (student_diff : ℕ) (total_students : ℕ) :
  num_classes = 5 →
  student_diff = 2 →
  total_students = 120 →
  ∃ (x : ℕ), x = 28 ∧ 
    (x + (x - student_diff) + (x - 2*student_diff) + (x - 3*student_diff) + (x - 4*student_diff) = total_students) :=
by sorry

end NUMINAMATH_CALUDE_largest_class_size_l2453_245379


namespace NUMINAMATH_CALUDE_gravel_path_cost_l2453_245327

/-- Represents the dimensions of a rectangular plot -/
structure PlotDimensions where
  length : ℝ
  width : ℝ

/-- Calculates the area of a rectangular path inside a plot -/
def pathArea (plot : PlotDimensions) (pathWidth : ℝ) : ℝ :=
  plot.length * plot.width - (plot.length - 2 * pathWidth) * (plot.width - 2 * pathWidth)

/-- Calculates the cost of gravelling a path -/
def gravellingCost (area : ℝ) (costPerSqMetre : ℝ) : ℝ :=
  area * costPerSqMetre

/-- Theorem: The cost of gravelling the path is 680 Rupees -/
theorem gravel_path_cost :
  let plot := PlotDimensions.mk 110 65
  let pathWidth := 2.5
  let costPerSqMetre := 0.8 -- 80 paise = 0.8 Rupees
  gravellingCost (pathArea plot pathWidth) costPerSqMetre = 680 := by
  sorry

end NUMINAMATH_CALUDE_gravel_path_cost_l2453_245327


namespace NUMINAMATH_CALUDE_perpendicular_line_through_point_l2453_245342

/-- Given a point P and a line L, this theorem proves that the line
    perpendicular to L passing through P has the correct equation. -/
theorem perpendicular_line_through_point
  (P : ℝ × ℝ)  -- Point P
  (L : ℝ → ℝ → Prop)  -- Line L
  (h_L : L = fun x y ↦ x - 2 * y + 3 = 0)  -- Equation of line L
  (h_P : P = (-1, 3))  -- Coordinates of point P
  : (fun x y ↦ 2 * x + y - 1 = 0) P.1 P.2 ∧  -- The line passes through P
    (∀ x₁ y₁ x₂ y₂, L x₁ y₁ → L x₂ y₂ →
      (x₂ - x₁) * 2 + (y₂ - y₁) * 1 = 0)  -- The lines are perpendicular
  := by sorry

end NUMINAMATH_CALUDE_perpendicular_line_through_point_l2453_245342


namespace NUMINAMATH_CALUDE_part1_part2_l2453_245396

-- Define the quadratic inequality
def quadratic_inequality (a : ℝ) (x : ℝ) : Prop :=
  a * x^2 + (a - 1) * x - 1 ≥ 0

-- Define the solution set for part 1
def solution_set_part1 : Set ℝ := {x | -1 ≤ x ∧ x ≤ -1/2}

-- Define the solution set for part 2
def solution_set_part2 (a : ℝ) : Set ℝ :=
  if a = -1 then {-1}
  else if a < -1 then {x | -1 ≤ x ∧ x ≤ 1/a}
  else {x | 1/a ≤ x ∧ x ≤ -1}

-- Theorem for part 1
theorem part1 :
  ∀ x ∈ solution_set_part1, quadratic_inequality (-2) x ∧
  ∀ a ≠ -2, ∃ x ∈ solution_set_part1, ¬(quadratic_inequality a x) :=
sorry

-- Theorem for part 2
theorem part2 (a : ℝ) (h : a < 0) :
  ∀ x, quadratic_inequality a x ↔ x ∈ solution_set_part2 a :=
sorry

end NUMINAMATH_CALUDE_part1_part2_l2453_245396


namespace NUMINAMATH_CALUDE_prob_one_girl_l2453_245392

theorem prob_one_girl (p_two_boys p_two_girls : ℚ) 
  (h1 : p_two_boys = 1/3) 
  (h2 : p_two_girls = 2/15) : 
  1 - (p_two_boys + p_two_girls) = 8/15 := by
  sorry

end NUMINAMATH_CALUDE_prob_one_girl_l2453_245392


namespace NUMINAMATH_CALUDE_fraction_value_l2453_245349

theorem fraction_value (a b c : ℚ) (h1 : a/b = 3) (h2 : b/c = 2) : (a-b)/(c-b) = -4 := by
  sorry

end NUMINAMATH_CALUDE_fraction_value_l2453_245349


namespace NUMINAMATH_CALUDE_roys_height_l2453_245328

/-- Given the heights of Sara, Joe, and Roy, prove Roy's height -/
theorem roys_height
  (sara_height : ℕ)
  (sara_joe_diff : ℕ)
  (joe_roy_diff : ℕ)
  (h_sara_height : sara_height = 45)
  (h_sara_joe : sara_height = sara_joe_diff + joe_height)
  (h_joe_roy : joe_height = joe_roy_diff + roy_height)
  : roy_height = 36 := by
  sorry


end NUMINAMATH_CALUDE_roys_height_l2453_245328


namespace NUMINAMATH_CALUDE_complex_magnitude_equality_l2453_245391

theorem complex_magnitude_equality (s : ℝ) (hs : s > 0) :
  Complex.abs (-3 + s * Complex.I) = 2 * Real.sqrt 10 → s = Real.sqrt 31 := by
  sorry

end NUMINAMATH_CALUDE_complex_magnitude_equality_l2453_245391


namespace NUMINAMATH_CALUDE_samantha_last_name_has_seven_letters_l2453_245301

/-- The number of letters in Jamie's last name -/
def jamie_last_name_length : ℕ := 4

/-- The number of letters in Bobbie's last name -/
def bobbie_last_name_length : ℕ := jamie_last_name_length * 2 + 2

/-- The number of letters in Samantha's last name -/
def samantha_last_name_length : ℕ := bobbie_last_name_length - 3

theorem samantha_last_name_has_seven_letters :
  samantha_last_name_length = 7 := by
  sorry

end NUMINAMATH_CALUDE_samantha_last_name_has_seven_letters_l2453_245301


namespace NUMINAMATH_CALUDE_unique_four_digit_power_sum_l2453_245308

theorem unique_four_digit_power_sum : ∃! (peru : ℕ), 
  1000 ≤ peru ∧ peru < 10000 ∧
  ∃ (p e r u : ℕ), 
    p > 0 ∧ p < 10 ∧ e < 10 ∧ r < 10 ∧ u < 10 ∧
    peru = 1000 * p + 100 * e + 10 * r + u ∧
    peru = (p + e + r + u) ^ u ∧
    peru = 4913 := by
  sorry

end NUMINAMATH_CALUDE_unique_four_digit_power_sum_l2453_245308


namespace NUMINAMATH_CALUDE_x_not_equal_y_l2453_245329

-- Define the sequences x_n and y_n
def x : ℕ → ℤ
  | 0 => 1
  | 1 => 1
  | (n + 2) => x (n + 1) + 2 * x n

def y : ℕ → ℤ
  | 0 => 1
  | 1 => 7
  | (n + 2) => 2 * y (n + 1) + 3 * y n

-- State the theorem
theorem x_not_equal_y (m n : ℕ) (hm : m > 0) (hn : n > 0) : x m ≠ y n := by
  sorry

end NUMINAMATH_CALUDE_x_not_equal_y_l2453_245329


namespace NUMINAMATH_CALUDE_helicopter_rental_cost_per_hour_l2453_245331

/-- Calculates the cost per hour for renting a helicopter --/
theorem helicopter_rental_cost_per_hour
  (hours_per_day : ℕ)
  (num_days : ℕ)
  (total_cost : ℕ)
  (h1 : hours_per_day = 2)
  (h2 : num_days = 3)
  (h3 : total_cost = 450) :
  total_cost / (hours_per_day * num_days) = 75 := by
sorry

end NUMINAMATH_CALUDE_helicopter_rental_cost_per_hour_l2453_245331


namespace NUMINAMATH_CALUDE_thirty_percent_less_than_80_l2453_245334

theorem thirty_percent_less_than_80 (x : ℝ) : x + (1/4) * x = 80 - 0.3 * 80 → x = 44.8 := by
  sorry

end NUMINAMATH_CALUDE_thirty_percent_less_than_80_l2453_245334


namespace NUMINAMATH_CALUDE_absolute_value_of_z_l2453_245343

theorem absolute_value_of_z (r : ℝ) (z : ℂ) (h1 : |r| < 3) (h2 : z + 1/z = r) : Complex.abs z = 1 := by
  sorry

end NUMINAMATH_CALUDE_absolute_value_of_z_l2453_245343


namespace NUMINAMATH_CALUDE_solve_age_ratio_l2453_245305

def age_ratio_problem (p q x : ℕ) : Prop :=
  -- P's current age is 15
  p = 15 ∧
  -- Some years ago, the ratio of P's age to Q's age was 4:3
  (p - x) * 3 = (q - x) * 4 ∧
  -- 6 years from now, the ratio of their ages will be 7:6
  (p + 6) * 6 = (q + 6) * 7

theorem solve_age_ratio : ∃ q x, age_ratio_problem 15 q x ∧ x = 3 := by
  sorry

end NUMINAMATH_CALUDE_solve_age_ratio_l2453_245305


namespace NUMINAMATH_CALUDE_simplify_trigonometric_expression_l2453_245314

theorem simplify_trigonometric_expression (θ : Real) 
  (h : θ ∈ Set.Icc (5 * Real.pi / 4) (3 * Real.pi / 2)) : 
  Real.sqrt (1 - Real.sin (2 * θ)) - Real.sqrt (1 + Real.sin (2 * θ)) = -2 * Real.cos θ := by
  sorry

end NUMINAMATH_CALUDE_simplify_trigonometric_expression_l2453_245314


namespace NUMINAMATH_CALUDE_single_stuffed_animal_cost_l2453_245399

/-- Represents the cost of items at a garage sale. -/
structure GarageSaleCost where
  magnet : ℝ
  sticker : ℝ
  stuffed_animals : ℝ
  toy_car : ℝ
  discount_rate : ℝ
  max_budget : ℝ

/-- Conditions for the garage sale problem. -/
def garage_sale_conditions (cost : GarageSaleCost) : Prop :=
  cost.magnet = 6 ∧
  cost.magnet = 3 * cost.sticker ∧
  cost.magnet = cost.stuffed_animals / 4 ∧
  cost.toy_car = cost.stuffed_animals / 4 ∧
  cost.toy_car = 2 * cost.sticker ∧
  cost.discount_rate = 0.1 ∧
  cost.max_budget = 30

/-- The theorem to be proved. -/
theorem single_stuffed_animal_cost 
  (cost : GarageSaleCost) 
  (h : garage_sale_conditions cost) : 
  cost.stuffed_animals / 2 = 12 := by
  sorry

end NUMINAMATH_CALUDE_single_stuffed_animal_cost_l2453_245399


namespace NUMINAMATH_CALUDE_min_diagonal_rectangle_l2453_245387

/-- The minimum diagonal length of a rectangle with perimeter 30 -/
theorem min_diagonal_rectangle (l w : ℝ) (h_perimeter : l + w = 15) :
  ∃ (min_diag : ℝ), min_diag = Real.sqrt 112.5 ∧
  ∀ (diag : ℝ), diag = Real.sqrt (l^2 + w^2) → diag ≥ min_diag :=
by sorry

end NUMINAMATH_CALUDE_min_diagonal_rectangle_l2453_245387


namespace NUMINAMATH_CALUDE_range_of_n_l2453_245386

-- Define the quadratic function
def f (x : ℝ) : ℝ := x^2 + 2*x + 2

-- Define the point P
structure Point where
  m : ℝ
  n : ℝ

-- Define the condition that P lies on the graph of f
def on_graph (P : Point) : Prop := P.n = f P.m

-- Define the condition that the circle intersects the y-axis
def circle_intersects_y_axis (P : Point) : Prop := abs P.m ≤ 2

-- Theorem statement
theorem range_of_n (P : Point) 
  (h1 : on_graph P) 
  (h2 : circle_intersects_y_axis P) : 
  1 ≤ P.n ∧ P.n < 10 := by sorry

end NUMINAMATH_CALUDE_range_of_n_l2453_245386


namespace NUMINAMATH_CALUDE_investment_problem_l2453_245304

/-- Investment problem -/
theorem investment_problem (a b total_profit a_profit c : ℚ) 
  (ha : a = 6300)
  (hb : b = 4200)
  (htotal : total_profit = 13600)
  (ha_profit : a_profit = 4080)
  (h_ratio : a / (a + b + c) = a_profit / total_profit) :
  c = 10500 := by
  sorry


end NUMINAMATH_CALUDE_investment_problem_l2453_245304


namespace NUMINAMATH_CALUDE_unique_number_l2453_245363

/-- A structure representing the statements made by a boy -/
structure BoyStatements where
  statement1 : Nat → Prop
  statement2 : Nat → Prop

/-- The set of statements made by each boy -/
def boyStatements : Fin 3 → BoyStatements
  | 0 => ⟨λ n => n % 10 = 6, λ n => n % 7 = 0⟩  -- Andrey
  | 1 => ⟨λ n => n > 26, λ n => n % 10 = 8⟩     -- Borya
  | 2 => ⟨λ n => n % 13 = 0, λ n => n < 27⟩     -- Sasha
  | _ => ⟨λ _ => False, λ _ => False⟩           -- Unreachable case

/-- The theorem stating that 91 is the only two-digit number satisfying all conditions -/
theorem unique_number : ∃! n : Nat, 10 ≤ n ∧ n < 100 ∧
  (∀ i : Fin 3, (boyStatements i).statement1 n ≠ (boyStatements i).statement2 n) ∧
  (∀ i : Fin 3, (boyStatements i).statement1 n ∨ (boyStatements i).statement2 n) :=
  sorry

end NUMINAMATH_CALUDE_unique_number_l2453_245363


namespace NUMINAMATH_CALUDE_root_shift_polynomial_l2453_245335

theorem root_shift_polynomial (s₁ s₂ s₃ : ℂ) : 
  (s₁^3 - 4*s₁^2 + 5*s₁ - 7 = 0) →
  (s₂^3 - 4*s₂^2 + 5*s₂ - 7 = 0) →
  (s₃^3 - 4*s₃^2 + 5*s₃ - 7 = 0) →
  ((s₁ + 3)^3 - 13*(s₁ + 3)^2 + 56*(s₁ + 3) - 85 = 0) ∧
  ((s₂ + 3)^3 - 13*(s₂ + 3)^2 + 56*(s₂ + 3) - 85 = 0) ∧
  ((s₃ + 3)^3 - 13*(s₃ + 3)^2 + 56*(s₃ + 3) - 85 = 0) :=
by sorry

end NUMINAMATH_CALUDE_root_shift_polynomial_l2453_245335


namespace NUMINAMATH_CALUDE_expression_bounds_bounds_are_tight_l2453_245357

theorem expression_bounds (a b c d e : ℝ) 
  (ha : 0 ≤ a ∧ a ≤ 1) (hb : 0 ≤ b ∧ b ≤ 1) (hc : 0 ≤ c ∧ c ≤ 1) 
  (hd : 0 ≤ d ∧ d ≤ 1) (he : 0 ≤ e ∧ e ≤ 1) : 
  5 / Real.sqrt 2 ≤ Real.sqrt (a^2 + (1-b)^2) + Real.sqrt (b^2 + (1-c)^2) + 
    Real.sqrt (c^2 + (1-d)^2) + Real.sqrt (d^2 + (1-e)^2) + Real.sqrt (e^2 + (1-a)^2) ∧
  Real.sqrt (a^2 + (1-b)^2) + Real.sqrt (b^2 + (1-c)^2) + Real.sqrt (c^2 + (1-d)^2) + 
    Real.sqrt (d^2 + (1-e)^2) + Real.sqrt (e^2 + (1-a)^2) ≤ 5 :=
by sorry

theorem bounds_are_tight : 
  ∃ (a b c d e : ℝ), (0 ≤ a ∧ a ≤ 1) ∧ (0 ≤ b ∧ b ≤ 1) ∧ (0 ≤ c ∧ c ≤ 1) ∧ 
    (0 ≤ d ∧ d ≤ 1) ∧ (0 ≤ e ∧ e ≤ 1) ∧
    Real.sqrt (a^2 + (1-b)^2) + Real.sqrt (b^2 + (1-c)^2) + Real.sqrt (c^2 + (1-d)^2) + 
    Real.sqrt (d^2 + (1-e)^2) + Real.sqrt (e^2 + (1-a)^2) = 5 / Real.sqrt 2 ∧
  ∃ (a' b' c' d' e' : ℝ), (0 ≤ a' ∧ a' ≤ 1) ∧ (0 ≤ b' ∧ b' ≤ 1) ∧ (0 ≤ c' ∧ c' ≤ 1) ∧ 
    (0 ≤ d' ∧ d' ≤ 1) ∧ (0 ≤ e' ∧ e' ≤ 1) ∧
    Real.sqrt (a'^2 + (1-b')^2) + Real.sqrt (b'^2 + (1-c')^2) + Real.sqrt (c'^2 + (1-d')^2) + 
    Real.sqrt (d'^2 + (1-e')^2) + Real.sqrt (e'^2 + (1-a')^2) = 5 :=
by sorry

end NUMINAMATH_CALUDE_expression_bounds_bounds_are_tight_l2453_245357


namespace NUMINAMATH_CALUDE_triangle_area_l2453_245398

/-- Given a triangle ABC with sides a, b, c opposite to angles A, B, C respectively,
    prove that its area is 4 under the given conditions. -/
theorem triangle_area (a b c : ℝ) (A B C : ℝ) : 
  a = 2 → c = 5 → Real.cos B = 3/5 → 
  (1/2) * a * c * Real.sin B = 4 := by sorry

end NUMINAMATH_CALUDE_triangle_area_l2453_245398


namespace NUMINAMATH_CALUDE_find_MN_length_l2453_245346

-- Define the triangles
structure Triangle :=
  (a b c : ℝ)

-- Define similarity relation
def similar (t1 t2 : Triangle) : Prop := sorry

-- Define the triangles
def PQR : Triangle := ⟨4, 8, sorry⟩
def XYZ : Triangle := ⟨sorry, 24, sorry⟩
def MNO : Triangle := ⟨sorry, sorry, 32⟩

-- State the theorem
theorem find_MN_length :
  similar PQR XYZ →
  similar XYZ MNO →
  MNO.a = 16 := by sorry

end NUMINAMATH_CALUDE_find_MN_length_l2453_245346


namespace NUMINAMATH_CALUDE_problem_solution_l2453_245317

theorem problem_solution :
  (∀ x y : ℝ, 3 * (x - y)^2 - 6 * (x - y)^2 + 2 * (x - y)^2 = -(x - y)^2) ∧
  (∀ a b : ℝ, a^2 - 2*b = 2 → 4*a^2 - 8*b - 9 = -1) ∧
  (∀ a b c d : ℝ, a - 2*b = 4 → b - c = -5 → 3*c + d = 10 → (a + 3*c) - (2*b + c) + (b + d) = 9) :=
by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l2453_245317


namespace NUMINAMATH_CALUDE_odd_symmetric_points_range_l2453_245319

theorem odd_symmetric_points_range (a : ℝ) :
  (∃ x₀ : ℝ, x₀ ≠ 0 ∧ Real.exp x₀ - a = -(Real.exp (-x₀) - a)) ↔ a > 1 := by
  sorry

end NUMINAMATH_CALUDE_odd_symmetric_points_range_l2453_245319


namespace NUMINAMATH_CALUDE_sqrt_difference_equality_l2453_245374

theorem sqrt_difference_equality : Real.sqrt (49 + 49) - Real.sqrt (36 - 25) = 7 * Real.sqrt 2 - Real.sqrt 11 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_difference_equality_l2453_245374


namespace NUMINAMATH_CALUDE_club_leadership_combinations_l2453_245340

/-- Represents the total number of students in the club -/
def total_students : ℕ := 30

/-- Represents the number of boys in the club -/
def num_boys : ℕ := 18

/-- Represents the number of girls in the club -/
def num_girls : ℕ := 12

/-- Represents the number of boy seniors (equal to boy juniors) -/
def num_boy_seniors : ℕ := num_boys / 2

/-- Represents the number of girl seniors (equal to girl juniors) -/
def num_girl_seniors : ℕ := num_girls / 2

/-- Represents the number of genders (boys and girls) -/
def num_genders : ℕ := 2

/-- Represents the number of class years (senior and junior) -/
def num_class_years : ℕ := 2

theorem club_leadership_combinations : 
  (num_genders * num_class_years * num_boy_seniors * num_boy_seniors) + 
  (num_genders * num_class_years * num_girl_seniors * num_girl_seniors) = 324 := by
  sorry

end NUMINAMATH_CALUDE_club_leadership_combinations_l2453_245340


namespace NUMINAMATH_CALUDE_intersection_empty_iff_not_p_sufficient_not_necessary_l2453_245332

-- Define the sets A and B
def A : Set ℝ := {x | -1 < x ∧ x < 6}
def B (a : ℝ) : Set ℝ := {x | x^2 - 2*x + 1 - a^2 ≥ 0}

-- Define the propositions p and q
def p (x : ℝ) : Prop := x ∈ A
def q (a : ℝ) (x : ℝ) : Prop := x ∈ B a

-- Theorem 1: A ∩ B = ∅ if and only if a ≥ 5
theorem intersection_empty_iff (a : ℝ) : A ∩ B a = ∅ ↔ a ≥ 5 := by sorry

-- Theorem 2: ¬p is a sufficient but not necessary condition for q if and only if 0 < a ≤ 2
theorem not_p_sufficient_not_necessary (a : ℝ) : 
  (∀ x, ¬p x → q a x) ∧ (∃ x, q a x ∧ p x) ↔ 0 < a ∧ a ≤ 2 := by sorry

end NUMINAMATH_CALUDE_intersection_empty_iff_not_p_sufficient_not_necessary_l2453_245332


namespace NUMINAMATH_CALUDE_seconds_in_five_and_half_minutes_l2453_245322

-- Define the number of seconds in a minute
def seconds_per_minute : ℕ := 60

-- Define the number of minutes we're converting
def minutes : ℚ := 5 + 1/2

-- Theorem statement
theorem seconds_in_five_and_half_minutes : 
  (minutes * seconds_per_minute : ℚ) = 330 := by
  sorry

end NUMINAMATH_CALUDE_seconds_in_five_and_half_minutes_l2453_245322


namespace NUMINAMATH_CALUDE_construction_delay_l2453_245310

/-- Represents the efficiency of a group of workers -/
structure WorkerGroup where
  count : ℕ
  efficiency : ℕ
  startDay : ℕ

/-- Calculates the total work units completed by a worker group -/
def totalWorkUnits (wg : WorkerGroup) (totalDays : ℕ) : ℕ :=
  wg.count * wg.efficiency * (totalDays - wg.startDay)

/-- Theorem: The construction would be 244 days behind schedule without additional workers -/
theorem construction_delay :
  let totalDays : ℕ := 150
  let initialWorkers : WorkerGroup := ⟨100, 1, 0⟩
  let additionalWorkers : List WorkerGroup := [
    ⟨50, 2, 30⟩,
    ⟨25, 3, 45⟩,
    ⟨15, 4, 75⟩
  ]
  let additionalWorkUnits : ℕ := (additionalWorkers.map (totalWorkUnits · totalDays)).sum
  let daysWithoutAdditional : ℕ := totalDays + (additionalWorkUnits / initialWorkers.count / initialWorkers.efficiency)
  daysWithoutAdditional - totalDays = 244 := by
    sorry

end NUMINAMATH_CALUDE_construction_delay_l2453_245310


namespace NUMINAMATH_CALUDE_systematic_sampling_fourth_number_l2453_245369

/-- Represents a systematic sampling scheme -/
structure SystematicSampling where
  total : Nat
  sample_size : Nat
  interval : Nat

/-- Checks if a number is part of the systematic sample -/
def SystematicSampling.isSampled (s : SystematicSampling) (n : Nat) : Prop :=
  ∃ k : Nat, n = s.interval * k + 1 ∧ k < s.sample_size

theorem systematic_sampling_fourth_number 
  (s : SystematicSampling)
  (h_total : s.total = 52)
  (h_sample_size : s.sample_size = 4)
  (h_5 : s.isSampled 5)
  (h_31 : s.isSampled 31)
  (h_44 : s.isSampled 44) :
  s.isSampled 18 :=
sorry

end NUMINAMATH_CALUDE_systematic_sampling_fourth_number_l2453_245369


namespace NUMINAMATH_CALUDE_divisors_multiple_of_five_l2453_245360

def n : ℕ := 7560

-- Define a function that counts positive divisors of n that are multiples of 5
def count_divisors_multiple_of_five (n : ℕ) : ℕ :=
  (Finset.filter (λ d => d ∣ n ∧ 5 ∣ d) (Finset.range (n + 1))).card

-- State the theorem
theorem divisors_multiple_of_five :
  count_divisors_multiple_of_five n = 32 := by
  sorry

end NUMINAMATH_CALUDE_divisors_multiple_of_five_l2453_245360


namespace NUMINAMATH_CALUDE_correct_growth_rate_equation_l2453_245325

/-- Represents the monthly average growth rate of sales volume for a product -/
def monthly_growth_rate (march_sales may_sales : ℝ) (x : ℝ) : Prop :=
  x > 0 ∧ 10 * (1 + x)^2 = 11.5 ∧ may_sales = march_sales * (1 + x)^2

/-- Theorem stating that the given equation correctly represents the monthly average growth rate -/
theorem correct_growth_rate_equation :
  ∃ x : ℝ, monthly_growth_rate 100000 115000 x :=
sorry

end NUMINAMATH_CALUDE_correct_growth_rate_equation_l2453_245325


namespace NUMINAMATH_CALUDE_five_fourths_of_eight_thirds_l2453_245393

theorem five_fourths_of_eight_thirds (x : ℚ) : 
  x = 8 / 3 → (5 / 4) * x = 10 / 3 := by
  sorry

end NUMINAMATH_CALUDE_five_fourths_of_eight_thirds_l2453_245393


namespace NUMINAMATH_CALUDE_ellipse_hyperbola_product_l2453_245326

/-- Given an ellipse and a hyperbola with specified foci, prove that the product of their semi-axes lengths is √868.5 -/
theorem ellipse_hyperbola_product (a b : ℝ) : 
  (∀ x y : ℝ, x^2 / a^2 + y^2 / b^2 = 1 → (x = 0 ∧ y = 5) ∨ (x = 0 ∧ y = -5)) →
  (∀ x y : ℝ, x^2 / a^2 - y^2 / b^2 = 1 → (x = 8 ∧ y = 0) ∨ (x = -8 ∧ y = 0)) →
  |a * b| = Real.sqrt 868.5 := by
  sorry

end NUMINAMATH_CALUDE_ellipse_hyperbola_product_l2453_245326


namespace NUMINAMATH_CALUDE_spellbook_cost_l2453_245355

/-- Proves that each spellbook costs 5 gold given the conditions of Harry's purchase --/
theorem spellbook_cost (num_spellbooks : ℕ) (num_potion_kits : ℕ) (owl_cost_gold : ℕ) 
  (potion_kit_cost_silver : ℕ) (silver_per_gold : ℕ) (total_cost_silver : ℕ) :
  num_spellbooks = 5 →
  num_potion_kits = 3 →
  owl_cost_gold = 28 →
  potion_kit_cost_silver = 20 →
  silver_per_gold = 9 →
  total_cost_silver = 537 →
  (total_cost_silver - (owl_cost_gold * silver_per_gold + num_potion_kits * potion_kit_cost_silver)) / num_spellbooks / silver_per_gold = 5 := by
  sorry

#check spellbook_cost

end NUMINAMATH_CALUDE_spellbook_cost_l2453_245355


namespace NUMINAMATH_CALUDE_truck_capacity_l2453_245336

theorem truck_capacity (total_boxes : ℕ) (num_trips : ℕ) (h1 : total_boxes = 871) (h2 : num_trips = 218) :
  total_boxes / num_trips = 4 := by
sorry

end NUMINAMATH_CALUDE_truck_capacity_l2453_245336


namespace NUMINAMATH_CALUDE_jqk_count_l2453_245380

/-- Given a pack of 52 cards, if the probability of drawing a jack, queen, or king
    is 0.23076923076923078, then the number of jacks, queens, and kings in the pack is 12. -/
theorem jqk_count (total_cards : ℕ) (prob : ℝ) (jqk_count : ℕ) : 
  total_cards = 52 →
  prob = 0.23076923076923078 →
  prob = (jqk_count : ℝ) / total_cards →
  jqk_count = 12 :=
by sorry

end NUMINAMATH_CALUDE_jqk_count_l2453_245380


namespace NUMINAMATH_CALUDE_parallelogram_area_l2453_245381

/-- Represents a point in 2D space -/
structure Point :=
  (x : ℝ) (y : ℝ)

/-- Represents a parallelogram -/
structure Parallelogram :=
  (A B C D : Point)

/-- Calculates the area of a parallelogram -/
def area (p : Parallelogram) : ℝ := sorry

/-- Checks if a line is perpendicular to another line -/
def isPerpendicular (p1 p2 p3 p4 : Point) : Prop := sorry

/-- Checks if a point is the midpoint of a line segment -/
def isMidpoint (m p1 p2 : Point) : Prop := sorry

/-- Main theorem: Area of parallelogram ABCD is √35 -/
theorem parallelogram_area (ABCD : Parallelogram) (E : Point) :
  area ABCD = Real.sqrt 35 :=
  sorry

end NUMINAMATH_CALUDE_parallelogram_area_l2453_245381


namespace NUMINAMATH_CALUDE_toy_value_proof_l2453_245333

theorem toy_value_proof (total_toys : ℕ) (total_worth : ℕ) (special_toy_value : ℕ) :
  total_toys = 9 →
  total_worth = 52 →
  special_toy_value = 12 →
  ∃ (other_toy_value : ℕ),
    (total_toys - 1) * other_toy_value + special_toy_value = total_worth ∧
    other_toy_value = 5 :=
by sorry

end NUMINAMATH_CALUDE_toy_value_proof_l2453_245333


namespace NUMINAMATH_CALUDE_percentage_problem_l2453_245359

theorem percentage_problem (x : ℝ) : 0.25 * x = 0.15 * 1500 - 20 → x = 820 := by
  sorry

end NUMINAMATH_CALUDE_percentage_problem_l2453_245359


namespace NUMINAMATH_CALUDE_log_square_problem_l2453_245324

theorem log_square_problem (x y : ℝ) 
  (hx_pos : x > 0) (hy_pos : y > 0)
  (hx_neq_one : x ≠ 1) (hy_neq_one : y ≠ 1)
  (h_log : Real.log x / Real.log 2 = Real.log 8 / Real.log y)
  (h_prod : x * y = 128) :
  (Real.log (x / y) / Real.log 2) ^ 2 = 37 := by
sorry

end NUMINAMATH_CALUDE_log_square_problem_l2453_245324


namespace NUMINAMATH_CALUDE_identical_second_differences_imply_arithmetic_progression_l2453_245312

/-- Second difference of a function -/
def secondDifference (g : ℕ → ℝ) (n : ℕ) : ℝ :=
  g (n + 2) - 2 * g (n + 1) + g n

/-- A sequence is an arithmetic progression if its second difference is zero -/
def isArithmeticProgression (g : ℕ → ℝ) : Prop :=
  ∀ n, secondDifference g n = 0

theorem identical_second_differences_imply_arithmetic_progression
  (f φ : ℕ → ℝ)
  (h : ∀ n, secondDifference f n = secondDifference φ n) :
  isArithmeticProgression (fun n ↦ f n - φ n) :=
by sorry

end NUMINAMATH_CALUDE_identical_second_differences_imply_arithmetic_progression_l2453_245312


namespace NUMINAMATH_CALUDE_max_stickers_for_one_player_l2453_245383

theorem max_stickers_for_one_player (n : ℕ) (avg : ℕ) (min_stickers : ℕ) 
  (h1 : n = 22)
  (h2 : avg = 4)
  (h3 : min_stickers = 1) :
  ∃ (max_stickers : ℕ), max_stickers = n * avg - (n - 1) * min_stickers ∧ max_stickers = 67 := by
  sorry

end NUMINAMATH_CALUDE_max_stickers_for_one_player_l2453_245383


namespace NUMINAMATH_CALUDE_sum_and_product_to_a_plus_b_l2453_245364

theorem sum_and_product_to_a_plus_b (a b : ℝ) 
  (sum_eq : (a + Real.sqrt b) + (a - Real.sqrt b) = -8)
  (product_eq : (a + Real.sqrt b) * (a - Real.sqrt b) = 4) :
  a + b = 8 := by
  sorry

end NUMINAMATH_CALUDE_sum_and_product_to_a_plus_b_l2453_245364


namespace NUMINAMATH_CALUDE_school_trip_photos_l2453_245352

theorem school_trip_photos (claire lisa robert : ℕ) : 
  lisa = 3 * claire →
  robert = claire + 24 →
  lisa = robert →
  claire = 12 := by
sorry

end NUMINAMATH_CALUDE_school_trip_photos_l2453_245352


namespace NUMINAMATH_CALUDE_billion_to_scientific_notation_l2453_245300

/-- Represents a number in scientific notation -/
structure ScientificNotation where
  coefficient : ℝ
  exponent : ℤ
  is_valid : 1 ≤ coefficient ∧ coefficient < 10

/-- Converts a real number to scientific notation -/
def toScientificNotation (x : ℝ) : ScientificNotation :=
  sorry

theorem billion_to_scientific_notation :
  let billion : ℝ := 1000000000
  let gdp : ℝ := 53100 * billion
  toScientificNotation gdp = ScientificNotation.mk 5.31 12 (by norm_num) :=
sorry

end NUMINAMATH_CALUDE_billion_to_scientific_notation_l2453_245300


namespace NUMINAMATH_CALUDE_triangle_properties_l2453_245367

theorem triangle_properties (A B C : ℝ) (a b c : ℝ) :
  -- Triangle ABC exists
  0 < A ∧ 0 < B ∧ 0 < C ∧ A + B + C = π →
  -- Sides a, b, c are positive
  0 < a ∧ 0 < b ∧ 0 < c →
  -- a, b, c form an arithmetic progression
  2 * b = a + c →
  -- 7 * sin(A) = 3 * sin(C)
  7 * Real.sin A = 3 * Real.sin C →
  -- Area of triangle is 15√3/4
  (1/2) * a * c * Real.sin B = (15 * Real.sqrt 3) / 4 →
  -- Prove: cos(B) = 11/14 and b = 5
  Real.cos B = 11/14 ∧ b = 5 := by
  sorry

end NUMINAMATH_CALUDE_triangle_properties_l2453_245367


namespace NUMINAMATH_CALUDE_cubic_root_ceiling_divisibility_l2453_245378

theorem cubic_root_ceiling_divisibility (x₁ x₂ x₃ : ℝ) (n : ℕ+) :
  x₁ < x₂ ∧ x₂ < x₃ →
  x₁^3 - 3*x₁^2 + 1 = 0 →
  x₂^3 - 3*x₂^2 + 1 = 0 →
  x₃^3 - 3*x₃^2 + 1 = 0 →
  ∃ k : ℤ, ⌈x₃^(n : ℝ)⌉ = 3 * k :=
by sorry

end NUMINAMATH_CALUDE_cubic_root_ceiling_divisibility_l2453_245378


namespace NUMINAMATH_CALUDE_ellipse_area_irrational_l2453_245372

/-- The area of an ellipse with rational semi-axes is irrational -/
theorem ellipse_area_irrational (a b : ℚ) (h1 : a > 0) (h2 : b > 0) : 
  Irrational (Real.pi * (a * b)) :=
by sorry

end NUMINAMATH_CALUDE_ellipse_area_irrational_l2453_245372


namespace NUMINAMATH_CALUDE_farmer_pumpkin_seeds_l2453_245344

/-- Represents the farmer's vegetable planting scenario -/
structure FarmerPlanting where
  bean_seedlings : ℕ
  bean_per_row : ℕ
  radishes : ℕ
  radishes_per_row : ℕ
  pumpkin_per_row : ℕ
  plant_beds : ℕ
  rows_per_bed : ℕ

/-- Calculates the number of pumpkin seeds in the given planting scenario -/
def calculate_pumpkin_seeds (f : FarmerPlanting) : ℕ :=
  let total_rows := f.plant_beds * f.rows_per_bed
  let bean_rows := f.bean_seedlings / f.bean_per_row
  let radish_rows := f.radishes / f.radishes_per_row
  let pumpkin_rows := total_rows - bean_rows - radish_rows
  pumpkin_rows * f.pumpkin_per_row

/-- Theorem stating that the farmer had 84 pumpkin seeds -/
theorem farmer_pumpkin_seeds :
  let f : FarmerPlanting := {
    bean_seedlings := 64,
    bean_per_row := 8,
    radishes := 48,
    radishes_per_row := 6,
    pumpkin_per_row := 7,
    plant_beds := 14,
    rows_per_bed := 2
  }
  calculate_pumpkin_seeds f = 84 := by
  sorry

end NUMINAMATH_CALUDE_farmer_pumpkin_seeds_l2453_245344


namespace NUMINAMATH_CALUDE_statue_cost_calculation_l2453_245353

/-- Given a statue sold for $750 with a 35% profit, prove that the original cost was $555.56 (rounded to two decimal places). -/
theorem statue_cost_calculation (selling_price : ℝ) (profit_percentage : ℝ) 
  (h1 : selling_price = 750)
  (h2 : profit_percentage = 35) : 
  ∃ (original_cost : ℝ), 
    selling_price = original_cost * (1 + profit_percentage / 100) ∧ 
    (round (original_cost * 100) / 100 : ℝ) = 555.56 := by
  sorry

end NUMINAMATH_CALUDE_statue_cost_calculation_l2453_245353


namespace NUMINAMATH_CALUDE_three_multiples_of_three_l2453_245309

theorem three_multiples_of_three (x y z : ℕ) : 
  x > 0 ∧ y > 0 ∧ z > 0 →
  x % 3 = 0 ∧ y % 3 = 0 ∧ z % 3 = 0 →
  x + y + z = 36 →
  (∃ m : ℕ, x = 3 * m ∧ y = 3 * (m + 1) ∧ z = 3 * (m + 2)) ∧
  (∃ n : ℕ, x = 6 * n ∧ y = 6 * (n + 1) ∧ z = 6 * (n + 2)) :=
by sorry

#check three_multiples_of_three

end NUMINAMATH_CALUDE_three_multiples_of_three_l2453_245309


namespace NUMINAMATH_CALUDE_trigonometric_values_and_difference_l2453_245351

def angle_α : ℝ := sorry

def point_P : ℝ × ℝ := (3, -4)

theorem trigonometric_values_and_difference :
  (point_P.1 = 3 ∧ point_P.2 = -4) →  -- Point P(3, -4) lies on the terminal side of angle α
  (Real.sin α * Real.cos α = 1/8) →   -- sinα*cosα = 1/8
  (π < α ∧ α < 5*π/4) →               -- π < α < 5π/4
  (Real.sin α = -4/5 ∧ 
   Real.cos α = 3/5 ∧ 
   Real.tan α = -4/3 ∧
   Real.cos α - Real.sin α = -Real.sqrt 3 / 12) := by
sorry

end NUMINAMATH_CALUDE_trigonometric_values_and_difference_l2453_245351


namespace NUMINAMATH_CALUDE_absolute_value_inequality_solution_set_l2453_245370

/-- The set of all real numbers x satisfying |x-5|+|x+1|<8 is equal to the open interval (-2, 6). -/
theorem absolute_value_inequality_solution_set :
  {x : ℝ | |x - 5| + |x + 1| < 8} = Set.Ioo (-2 : ℝ) 6 := by sorry

end NUMINAMATH_CALUDE_absolute_value_inequality_solution_set_l2453_245370


namespace NUMINAMATH_CALUDE_quadratic_form_sum_l2453_245302

/-- Given a quadratic x^2 - 40x + 121 that can be written as (x+b)^2 + c,
    prove that b + c = -299 -/
theorem quadratic_form_sum (b c : ℝ) : 
  (∀ x, x^2 - 40*x + 121 = (x + b)^2 + c) → b + c = -299 := by
sorry

end NUMINAMATH_CALUDE_quadratic_form_sum_l2453_245302


namespace NUMINAMATH_CALUDE_locus_characterization_locus_is_ray_l2453_245345

/-- The locus of points P satisfying |PM| - |PN| = 4, where M(-2,0) and N(2,0) -/
def locus : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.2 = 0 ∧ p.1 ≥ 2}

theorem locus_characterization (P : ℝ × ℝ) :
  P ∈ locus ↔ Real.sqrt ((P.1 + 2)^2 + P.2^2) - Real.sqrt ((P.1 - 2)^2 + P.2^2) = 4 :=
sorry

/-- The points M and N -/
def M : ℝ × ℝ := (-2, 0)
def N : ℝ × ℝ := (2, 0)

theorem locus_is_ray :
  locus = {p : ℝ × ℝ | p.2 = 0 ∧ p.1 ≥ 2} :=
sorry

end NUMINAMATH_CALUDE_locus_characterization_locus_is_ray_l2453_245345


namespace NUMINAMATH_CALUDE_monic_quartic_polynomial_value_l2453_245365

/-- A monic quartic polynomial is a polynomial of degree 4 with leading coefficient 1 -/
def MonicQuarticPolynomial (f : ℝ → ℝ) : Prop :=
  ∃ a b c d : ℝ, ∀ x, f x = x^4 + a*x^3 + b*x^2 + c*x + d

theorem monic_quartic_polynomial_value (f : ℝ → ℝ) :
  MonicQuarticPolynomial f →
  f (-2) = -4 →
  f 1 = -1 →
  f 3 = -9 →
  f (-4) = -16 →
  f 2 = -28 := by
    sorry

end NUMINAMATH_CALUDE_monic_quartic_polynomial_value_l2453_245365


namespace NUMINAMATH_CALUDE_expression_evaluation_l2453_245307

theorem expression_evaluation :
  let x : ℚ := 2
  let y : ℚ := -1/2
  (x * (x - 4*y) + (2*x + y) * (2*x - y) - (2*x - y)^2) = 7/2 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l2453_245307


namespace NUMINAMATH_CALUDE_necessary_not_sufficient_condition_l2453_245366

theorem necessary_not_sufficient_condition (a b c d : ℝ) (h : c > d) :
  (∀ a b, (a - c > b - d) → (a > b)) ∧
  (∃ a b, (a > b) ∧ ¬(a - c > b - d)) :=
sorry

end NUMINAMATH_CALUDE_necessary_not_sufficient_condition_l2453_245366


namespace NUMINAMATH_CALUDE_area_of_combined_rectangle_l2453_245375

/-- The area of a rectangle formed by three identical rectangles --/
theorem area_of_combined_rectangle (short_side : ℝ) (h : short_side = 7) : 
  let long_side : ℝ := 2 * short_side
  let width : ℝ := 2 * short_side
  let length : ℝ := long_side
  width * length = 196 := by sorry

end NUMINAMATH_CALUDE_area_of_combined_rectangle_l2453_245375


namespace NUMINAMATH_CALUDE_mayas_books_pages_l2453_245390

/-- Proves that if Maya read 5 books last week, read twice as much this week, 
    and read a total of 4500 pages this week, then each book had 450 pages. -/
theorem mayas_books_pages (books_last_week : ℕ) (pages_this_week : ℕ) 
  (h1 : books_last_week = 5)
  (h2 : pages_this_week = 4500) :
  (pages_this_week / (2 * books_last_week) : ℚ) = 450 := by
  sorry

#check mayas_books_pages

end NUMINAMATH_CALUDE_mayas_books_pages_l2453_245390


namespace NUMINAMATH_CALUDE_unique_number_in_range_l2453_245356

theorem unique_number_in_range : ∃! x : ℝ, 3 < x ∧ x < 8 ∧ 6 < x ∧ x < 10 ∧ x = 7 := by
  sorry

end NUMINAMATH_CALUDE_unique_number_in_range_l2453_245356


namespace NUMINAMATH_CALUDE_city_distance_l2453_245385

def is_valid_distance (S : ℕ+) : Prop :=
  ∀ x : ℕ, x < S → (Nat.gcd x (S - x) = 1 ∨ Nat.gcd x (S - x) = 3 ∨ Nat.gcd x (S - x) = 13)

theorem city_distance : ∃ S : ℕ+, is_valid_distance S ∧ ∀ T : ℕ+, T < S → ¬is_valid_distance T :=
  sorry

end NUMINAMATH_CALUDE_city_distance_l2453_245385


namespace NUMINAMATH_CALUDE_power_of_128_fourths_sevenths_l2453_245338

theorem power_of_128_fourths_sevenths : (128 : ℝ) ^ (4/7) = 16 := by
  sorry

end NUMINAMATH_CALUDE_power_of_128_fourths_sevenths_l2453_245338


namespace NUMINAMATH_CALUDE_reciprocal_sum_fractions_l2453_245395

theorem reciprocal_sum_fractions : 
  (1 / (1/4 + 1/6) : ℚ) = 12/5 := by sorry

end NUMINAMATH_CALUDE_reciprocal_sum_fractions_l2453_245395


namespace NUMINAMATH_CALUDE_counting_functions_l2453_245315

/-- The number of strictly increasing functions from {1,2,...,m} to {1,2,...,n} -/
def strictlyIncreasingFunctions (m n : ℕ) : ℕ :=
  Nat.choose n m

/-- The number of increasing functions from {1,2,...,m} to {1,2,...,n} -/
def increasingFunctions (m n : ℕ) : ℕ :=
  Nat.choose (m + n) m

theorem counting_functions (m n : ℕ) (h : m ≠ 0 ∧ n ≠ 0) :
  (m ≤ n → strictlyIncreasingFunctions m n = Nat.choose n m) ∧
  increasingFunctions m n = Nat.choose (m + n) m := by
  sorry

end NUMINAMATH_CALUDE_counting_functions_l2453_245315


namespace NUMINAMATH_CALUDE_teacher_age_teacher_age_problem_l2453_245358

/-- Given a class of students and their teacher, calculate the teacher's age based on how it affects the class average. -/
theorem teacher_age (num_students : ℕ) (student_avg_age : ℚ) (new_avg_age : ℚ) : ℚ :=
  let total_student_age := num_students * student_avg_age
  let total_new_age := (num_students + 1) * new_avg_age
  total_new_age - total_student_age

/-- Prove that for a class of 25 students with an average age of 26 years, 
    if including the teacher's age increases the average by 1 year, 
    then the teacher's age is 52 years. -/
theorem teacher_age_problem : teacher_age 25 26 27 = 52 := by
  sorry

end NUMINAMATH_CALUDE_teacher_age_teacher_age_problem_l2453_245358


namespace NUMINAMATH_CALUDE_added_amount_proof_l2453_245337

theorem added_amount_proof (x y : ℝ) : x = 16 ∧ 3 * (2 * x + y) = 111 → y = 5 := by
  sorry

end NUMINAMATH_CALUDE_added_amount_proof_l2453_245337


namespace NUMINAMATH_CALUDE_basketball_team_selection_l2453_245316

def total_players : ℕ := 15
def quadruplets : ℕ := 4
def team_size : ℕ := 7
def remaining_players : ℕ := total_players - quadruplets
def players_to_choose : ℕ := team_size - quadruplets

theorem basketball_team_selection :
  Nat.choose remaining_players players_to_choose = 165 := by
  sorry

end NUMINAMATH_CALUDE_basketball_team_selection_l2453_245316


namespace NUMINAMATH_CALUDE_shifted_graph_symmetry_l2453_245382

noncomputable def f (x : ℝ) : ℝ := Real.sqrt 3 * Real.cos (x / 2) + Real.sin (x / 2)

theorem shifted_graph_symmetry (m : ℝ) (h : m > 0) :
  (∀ x : ℝ, f (x + m) = f (-x + m)) ↔ m ≥ 4 * Real.pi / 3 :=
sorry

end NUMINAMATH_CALUDE_shifted_graph_symmetry_l2453_245382


namespace NUMINAMATH_CALUDE_perpendicular_parallel_transitive_l2453_245361

/-- A structure representing a line in 3D space -/
structure Line3D where
  -- Add necessary fields

/-- A structure representing a plane in 3D space -/
structure Plane3D where
  -- Add necessary fields

/-- Perpendicularity relation between a line and a plane -/
def perpendicular (l : Line3D) (p : Plane3D) : Prop :=
  sorry

/-- Parallelism relation between two lines -/
def parallel (l1 l2 : Line3D) : Prop :=
  sorry

/-- Main theorem: If a line is perpendicular to a plane and parallel to another line,
    then the other line is also perpendicular to the plane -/
theorem perpendicular_parallel_transitive
  (m n : Line3D) (α : Plane3D)
  (h1 : perpendicular m α)
  (h2 : parallel m n) :
  perpendicular n α :=
sorry

end NUMINAMATH_CALUDE_perpendicular_parallel_transitive_l2453_245361


namespace NUMINAMATH_CALUDE_amusement_park_spending_l2453_245348

theorem amusement_park_spending (total : ℕ) (admission : ℕ) (food : ℕ) 
  (h1 : total = 77)
  (h2 : admission = 45)
  (h3 : total = admission + food)
  (h4 : food < admission) :
  admission - food = 13 := by
  sorry

end NUMINAMATH_CALUDE_amusement_park_spending_l2453_245348


namespace NUMINAMATH_CALUDE_midpoint_distance_squared_l2453_245339

/-- A rectangle ABCD with given dimensions and midpoints -/
structure Rectangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ
  D : ℝ × ℝ
  X : ℝ × ℝ
  Y : ℝ × ℝ
  h_rectangle : A.1 = D.1 ∧ B.1 = C.1 ∧ A.2 = B.2 ∧ C.2 = D.2
  h_AB : (B.1 - A.1)^2 + (B.2 - A.2)^2 = 15^2
  h_BC : (C.1 - B.1)^2 + (C.2 - B.2)^2 = 8^2
  h_right_angle : (B.1 - A.1) * (C.1 - B.1) + (B.2 - A.2) * (C.2 - B.2) = 0
  h_X_midpoint : X = ((B.1 + C.1) / 2, (B.2 + C.2) / 2)
  h_Y_midpoint : Y = ((D.1 + A.1) / 2, (D.2 + A.2) / 2)

/-- The square of the distance between midpoints X and Y is 64 -/
theorem midpoint_distance_squared (r : Rectangle) : 
  (r.X.1 - r.Y.1)^2 + (r.X.2 - r.Y.2)^2 = 64 := by
  sorry

end NUMINAMATH_CALUDE_midpoint_distance_squared_l2453_245339


namespace NUMINAMATH_CALUDE_chess_tournament_rounds_l2453_245306

theorem chess_tournament_rounds (total_games : ℕ) (h : total_games = 224) :
  ∃ (participants rounds : ℕ),
    participants > 1 ∧
    rounds > 0 ∧
    participants * (participants - 1) * rounds = 2 * total_games ∧
    rounds = 8 := by
sorry

end NUMINAMATH_CALUDE_chess_tournament_rounds_l2453_245306


namespace NUMINAMATH_CALUDE_intersection_A_B_l2453_245377

def A : Set ℝ := {-3, -1, 0, 1}

def B : Set ℝ := {x | (x + 2) * (x - 1) < 0}

theorem intersection_A_B : A ∩ B = {-1, 0} := by sorry

end NUMINAMATH_CALUDE_intersection_A_B_l2453_245377


namespace NUMINAMATH_CALUDE_skew_diagonal_cube_volume_l2453_245318

/-- Represents a cube with skew diagonals on its surface. -/
structure SkewDiagonalCube where
  side_length : ℝ
  has_skew_diagonals : Bool
  skew_diagonal_distance : ℝ

/-- Theorem stating that for a cube with skew diagonals where the distance between two skew lines is 1,
    the volume of the cube is either 1 or 3√3. -/
theorem skew_diagonal_cube_volume 
  (cube : SkewDiagonalCube) 
  (h1 : cube.has_skew_diagonals = true) 
  (h2 : cube.skew_diagonal_distance = 1) : 
  cube.side_length ^ 3 = 1 ∨ cube.side_length ^ 3 = 3 * Real.sqrt 3 := by
  sorry

#check skew_diagonal_cube_volume

end NUMINAMATH_CALUDE_skew_diagonal_cube_volume_l2453_245318


namespace NUMINAMATH_CALUDE_min_area_k_sum_l2453_245376

/-- A point in 2D space -/
structure Point :=
  (x : ℝ)
  (y : ℝ)

/-- Calculate the area of a triangle given three points -/
def triangleArea (p1 p2 p3 : Point) : ℝ :=
  0.5 * abs ((p2.x - p1.x) * (p3.y - p1.y) - (p3.x - p1.x) * (p2.y - p1.y))

/-- The theorem stating the sum of k values that minimize the triangle area -/
theorem min_area_k_sum :
  let p1 : Point := ⟨2, 5⟩
  let p2 : Point := ⟨10, 20⟩
  let p3 (k : ℤ) : Point := ⟨7, k⟩
  let minArea := fun (k : ℤ) ↦ triangleArea p1 p2 (p3 k)
  ∃ (k1 k2 : ℤ),
    (∀ (k : ℤ), minArea k ≥ minArea k1 ∧ minArea k ≥ minArea k2) ∧
    k1 + k2 = 29 :=
sorry


end NUMINAMATH_CALUDE_min_area_k_sum_l2453_245376


namespace NUMINAMATH_CALUDE_sum_of_squares_of_roots_l2453_245362

theorem sum_of_squares_of_roots : 
  ∃ (r s t : ℝ), 
    (∀ x : ℝ, x ≥ 0 → (x * Real.sqrt x - 8 * x + 9 * Real.sqrt x - 2 = 0 ↔ x = r ∨ x = s ∨ x = t)) →
    r ≥ 0 ∧ s ≥ 0 ∧ t ≥ 0 →
    r^2 + s^2 + t^2 = 46 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_squares_of_roots_l2453_245362


namespace NUMINAMATH_CALUDE_union_and_intersection_of_rational_and_irrational_l2453_245311

-- Define A as the set of rational numbers
def A : Set ℝ := {x : ℝ | ∃ (p q : ℤ), q ≠ 0 ∧ x = p / q}

-- Define B as the set of irrational numbers
def B : Set ℝ := {x : ℝ | x ∉ A}

theorem union_and_intersection_of_rational_and_irrational :
  (A ∪ B = Set.univ) ∧ (A ∩ B = ∅) := by
  sorry

end NUMINAMATH_CALUDE_union_and_intersection_of_rational_and_irrational_l2453_245311


namespace NUMINAMATH_CALUDE_correct_calculation_l2453_245384

theorem correct_calculation (x y : ℝ) : -2*x*y + 3*y*x = x*y := by
  sorry

end NUMINAMATH_CALUDE_correct_calculation_l2453_245384


namespace NUMINAMATH_CALUDE_vertical_shift_equivalence_l2453_245321

/-- A function that represents a vertical shift of another function -/
def verticalShift (f : ℝ → ℝ) (c : ℝ) : ℝ → ℝ := λ x ↦ f x + c

/-- Theorem stating that a vertical shift of a function is equivalent to adding a constant to its output -/
theorem vertical_shift_equivalence (f : ℝ → ℝ) (c : ℝ) :
  ∀ x : ℝ, verticalShift f c x = f x + c := by sorry

end NUMINAMATH_CALUDE_vertical_shift_equivalence_l2453_245321


namespace NUMINAMATH_CALUDE_smallest_valid_strategy_l2453_245397

/-- Represents a 9x9 game board -/
def GameBoard := Fin 9 → Fin 9 → Bool

/-- Represents an L-shaped tromino -/
structure Tromino :=
  (x : Fin 9) (y : Fin 9) (orientation : Fin 4)

/-- Checks if a tromino covers a given cell -/
def covers (t : Tromino) (x y : Fin 9) : Bool :=
  sorry

/-- Checks if a marking strategy allows unique determination of tromino placement -/
def is_valid_strategy (board : GameBoard) : Prop :=
  ∀ t1 t2 : Tromino, t1 ≠ t2 →
    ∃ x y : Fin 9, board x y ∧ (covers t1 x y ≠ covers t2 x y)

/-- Counts the number of marked cells on the board -/
def count_marked (board : GameBoard) : Nat :=
  sorry

theorem smallest_valid_strategy :
  ∃ (board : GameBoard),
    is_valid_strategy board ∧
    count_marked board = 68 ∧
    ∀ (other_board : GameBoard),
      is_valid_strategy other_board →
      count_marked other_board ≥ 68 :=
sorry

end NUMINAMATH_CALUDE_smallest_valid_strategy_l2453_245397


namespace NUMINAMATH_CALUDE_smallest_d_value_l2453_245303

theorem smallest_d_value (c d : ℕ+) (h1 : c.val - d.val = 8) 
  (h2 : Nat.gcd ((c.val^3 + d.val^3) / (c.val + d.val)) (c.val * d.val) = 16) : 
  d.val ≥ 4 ∧ ∃ (c' d' : ℕ+), d'.val = 4 ∧ c'.val - d'.val = 8 ∧ 
    Nat.gcd ((c'.val^3 + d'.val^3) / (c'.val + d'.val)) (c'.val * d'.val) = 16 :=
by sorry

end NUMINAMATH_CALUDE_smallest_d_value_l2453_245303


namespace NUMINAMATH_CALUDE_gcd_3060_561_l2453_245341

theorem gcd_3060_561 : Nat.gcd 3060 561 = 51 := by
  sorry

end NUMINAMATH_CALUDE_gcd_3060_561_l2453_245341


namespace NUMINAMATH_CALUDE_ratio_odd_even_divisors_M_l2453_245350

/-- The number M as defined in the problem -/
def M : ℕ := 36 * 36 * 95 * 400

/-- Sum of odd divisors of a natural number -/
def sum_odd_divisors (n : ℕ) : ℕ := sorry

/-- Sum of even divisors of a natural number -/
def sum_even_divisors (n : ℕ) : ℕ := sorry

/-- Theorem stating the ratio of sum of odd divisors to sum of even divisors of M -/
theorem ratio_odd_even_divisors_M :
  (sum_odd_divisors M : ℚ) / (sum_even_divisors M : ℚ) = 1 / 510 := by sorry

end NUMINAMATH_CALUDE_ratio_odd_even_divisors_M_l2453_245350


namespace NUMINAMATH_CALUDE_added_number_after_doubling_l2453_245368

theorem added_number_after_doubling (x y : ℝ) : 
  x = 4 → 3 * (2 * x + y) = 51 → y = 9 := by
  sorry

end NUMINAMATH_CALUDE_added_number_after_doubling_l2453_245368


namespace NUMINAMATH_CALUDE_min_value_of_y_l2453_245371

theorem min_value_of_y (a b : ℝ) (ha : a > 0) (hb : b > 0) (hrel : b = (1 - a) / 3) :
  ∃ (y_min : ℝ), y_min = 2 * Real.sqrt 3 ∧ ∀ (y : ℝ), y = 3^a + 27^b → y ≥ y_min :=
sorry

end NUMINAMATH_CALUDE_min_value_of_y_l2453_245371


namespace NUMINAMATH_CALUDE_equation_B_positive_correlation_l2453_245354

/-- Represents an empirical regression equation of the form ŷ = ax + b -/
structure RegressionEquation where
  a : ℝ  -- Coefficient of x
  b : ℝ  -- y-intercept

/-- Defines what it means for a regression equation to show positive correlation -/
def shows_positive_correlation (eq : RegressionEquation) : Prop :=
  eq.a > 0

/-- The specific regression equation we're interested in -/
def equation_B : RegressionEquation :=
  { a := 1.2, b := 1.5 }

/-- Theorem stating that equation B shows a positive correlation -/
theorem equation_B_positive_correlation :
  shows_positive_correlation equation_B := by
  sorry


end NUMINAMATH_CALUDE_equation_B_positive_correlation_l2453_245354
