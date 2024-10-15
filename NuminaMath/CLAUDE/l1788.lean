import Mathlib

namespace NUMINAMATH_CALUDE_equal_intercept_line_equation_l1788_178836

/-- A line passing through (1,1) with equal horizontal and vertical intercepts -/
structure EqualInterceptLine where
  /-- The slope of the line -/
  slope : ℝ
  /-- The y-intercept of the line -/
  y_intercept : ℝ
  /-- The line passes through (1,1) -/
  passes_through_point : slope + y_intercept = 1
  /-- The line has equal horizontal and vertical intercepts -/
  equal_intercepts : y_intercept = slope * y_intercept

/-- The equation of an EqualInterceptLine is either y = x or x + y - 2 = 0 -/
theorem equal_intercept_line_equation (l : EqualInterceptLine) :
  (l.slope = 1 ∧ l.y_intercept = 0) ∨
  (l.slope = -1 ∧ l.y_intercept = 2) :=
sorry

end NUMINAMATH_CALUDE_equal_intercept_line_equation_l1788_178836


namespace NUMINAMATH_CALUDE_max_value_expression_l1788_178823

theorem max_value_expression (w x y z t : ℝ) 
  (h_nonneg : w ≥ 0 ∧ x ≥ 0 ∧ y ≥ 0 ∧ z ≥ 0 ∧ t ≥ 0) 
  (h_sum : w + x + y + z + t = 120) : 
  w * x + x * y + y * z + z * t ≤ 3600 :=
by sorry

end NUMINAMATH_CALUDE_max_value_expression_l1788_178823


namespace NUMINAMATH_CALUDE_matthews_crackers_l1788_178801

/-- The number of crackers Matthew gave to each friend -/
def crackers_per_friend : ℕ := 6

/-- The number of friends Matthew gave crackers to -/
def number_of_friends : ℕ := 6

/-- The total number of crackers Matthew had -/
def total_crackers : ℕ := crackers_per_friend * number_of_friends

theorem matthews_crackers : total_crackers = 36 := by
  sorry

end NUMINAMATH_CALUDE_matthews_crackers_l1788_178801


namespace NUMINAMATH_CALUDE_hyperbola_standard_equation_l1788_178880

def ellipse_equation (x y : ℝ) : Prop := x^2 / 27 + y^2 / 36 = 1

def hyperbola_equation (a b x y : ℝ) : Prop := y^2 / a^2 - x^2 / b^2 = 1

theorem hyperbola_standard_equation :
  ∃ a b : ℝ,
    (∀ x y : ℝ, ellipse_equation x y → hyperbola_equation a b x y) ∧
    hyperbola_equation a b (Real.sqrt 15) 4 ∧
    a^2 = 4 ∧ b^2 = 5 :=
sorry

end NUMINAMATH_CALUDE_hyperbola_standard_equation_l1788_178880


namespace NUMINAMATH_CALUDE_bee_flight_time_l1788_178870

/-- Flight time of a honey bee between flowers -/
theorem bee_flight_time (time_daisy_to_rose : ℝ) (speed_daisy_to_rose : ℝ) (speed_difference : ℝ) (distance_difference : ℝ) :
  time_daisy_to_rose = 10 →
  speed_daisy_to_rose = 2.6 →
  speed_difference = 3 →
  distance_difference = 8 →
  ∃ (time_rose_to_poppy : ℝ),
    time_rose_to_poppy > 0 ∧
    time_rose_to_poppy < 4 ∧
    (speed_daisy_to_rose * time_daisy_to_rose - distance_difference) / (speed_daisy_to_rose + speed_difference) = time_rose_to_poppy :=
by sorry

end NUMINAMATH_CALUDE_bee_flight_time_l1788_178870


namespace NUMINAMATH_CALUDE_max_value_product_l1788_178886

theorem max_value_product (x y z : ℝ) (h1 : 0 ≤ x) (h2 : 0 ≤ y) (h3 : 0 ≤ z) (h4 : x + y + z = 3) :
  (x^2 + x*y + y^2) * (x^2 + x*z + z^2) * (y^2 + y*z + z^2) ≤ 27 :=
by sorry

end NUMINAMATH_CALUDE_max_value_product_l1788_178886


namespace NUMINAMATH_CALUDE_toy_cars_count_l1788_178846

/-- The number of toy cars given to boys in a charity event -/
def toy_cars_to_boys (total_toys : ℕ) (dolls_to_girls : ℕ) : ℕ :=
  total_toys - dolls_to_girls

/-- Theorem stating that the number of toy cars given to boys is 134 -/
theorem toy_cars_count :
  toy_cars_to_boys 403 269 = 134 := by
  sorry

end NUMINAMATH_CALUDE_toy_cars_count_l1788_178846


namespace NUMINAMATH_CALUDE_power_ends_in_12890625_l1788_178840

theorem power_ends_in_12890625 (a n : ℕ) (h : a % 10^8 = 12890625) :
  (a^n) % 10^8 = 12890625 := by
  sorry

end NUMINAMATH_CALUDE_power_ends_in_12890625_l1788_178840


namespace NUMINAMATH_CALUDE_range_m_for_f_negative_solution_inequality_range_m_for_f_geq_quadratic_l1788_178890

/-- The function f(x) defined in the problem -/
def f (m : ℝ) (x : ℝ) : ℝ := (m + 1) * x^2 - (m - 1) * x + (m - 1)

/-- The range of m for which f(x) < 0 has solution set ℝ -/
theorem range_m_for_f_negative (m : ℝ) : 
  (∀ x, f m x < 0) ↔ m < -5/3 := by sorry

/-- The solution to f(x) ≥ 3x + m - 2 when m < 0 -/
theorem solution_inequality (m : ℝ) (hm : m < 0) :
  (∀ x, f m x ≥ 3*x + m - 2) ↔ 
    ((-1 < m ∧ (∀ x, x ≤ 1 ∨ x ≥ 1/(m+1))) ∨
     (m = -1 ∧ (∀ x, x ≤ 1)) ∨
     (m < -1 ∧ (∀ x, 1/(m+1) ≤ x ∧ x ≤ 1))) := by sorry

/-- The range of m for which f(x) ≥ x^2 + 2x holds for all x ∈ [0,2] -/
theorem range_m_for_f_geq_quadratic (m : ℝ) :
  (∀ x ∈ Set.Icc 0 2, f m x ≥ x^2 + 2*x) ↔ m ≥ 2*Real.sqrt 3/3 + 1 := by sorry

end NUMINAMATH_CALUDE_range_m_for_f_negative_solution_inequality_range_m_for_f_geq_quadratic_l1788_178890


namespace NUMINAMATH_CALUDE_prob_both_type_a_prob_different_types_l1788_178819

/-- Represents the total number of questions -/
def total_questions : ℕ := 6

/-- Represents the number of type A questions -/
def type_a_questions : ℕ := 4

/-- Represents the number of type B questions -/
def type_b_questions : ℕ := 2

/-- Represents the number of questions to be selected -/
def selected_questions : ℕ := 2

/-- The probability of selecting 2 questions of type A -/
theorem prob_both_type_a : 
  (Nat.choose type_a_questions selected_questions : ℚ) / 
  (Nat.choose total_questions selected_questions : ℚ) = 2/5 := by sorry

/-- The probability of selecting 2 questions of different types -/
theorem prob_different_types :
  ((type_a_questions * type_b_questions : ℚ) / 
  (Nat.choose total_questions selected_questions : ℚ)) = 8/15 := by sorry

end NUMINAMATH_CALUDE_prob_both_type_a_prob_different_types_l1788_178819


namespace NUMINAMATH_CALUDE_coffee_shop_spending_l1788_178888

theorem coffee_shop_spending (A B : ℝ) : 
  B = 0.5 * A → A = B + 15 → A + B = 45 := by sorry

end NUMINAMATH_CALUDE_coffee_shop_spending_l1788_178888


namespace NUMINAMATH_CALUDE_pencil_sale_problem_l1788_178824

theorem pencil_sale_problem (total_students : Nat) (first_group : Nat) (second_group : Nat) (third_group : Nat)
  (first_group_pencils : Nat) (third_group_pencils : Nat) (total_pencils : Nat) :
  total_students = first_group + second_group + third_group →
  first_group = 2 →
  third_group = 2 →
  first_group_pencils = 2 →
  third_group_pencils = 1 →
  total_pencils = 24 →
  ∃ (second_group_pencils : Nat),
    second_group_pencils * second_group + first_group_pencils * first_group + third_group_pencils * third_group = total_pencils ∧
    second_group_pencils = 3 :=
by sorry


end NUMINAMATH_CALUDE_pencil_sale_problem_l1788_178824


namespace NUMINAMATH_CALUDE_soda_cost_l1788_178865

/-- The cost of items in cents -/
structure Cost where
  burger : ℕ
  soda : ℕ
  fry : ℕ

/-- The problem statement -/
theorem soda_cost (c : Cost) : 
  (3 * c.burger + 2 * c.soda + 2 * c.fry = 590) ∧ 
  (2 * c.burger + 3 * c.soda + c.fry = 610) → 
  c.soda = 140 := by
  sorry

end NUMINAMATH_CALUDE_soda_cost_l1788_178865


namespace NUMINAMATH_CALUDE_total_books_count_l1788_178859

theorem total_books_count (sam_books joan_books tom_books alice_books : ℕ)
  (h1 : sam_books = 110)
  (h2 : joan_books = 102)
  (h3 : tom_books = 125)
  (h4 : alice_books = 97) :
  sam_books + joan_books + tom_books + alice_books = 434 := by
  sorry

end NUMINAMATH_CALUDE_total_books_count_l1788_178859


namespace NUMINAMATH_CALUDE_sum_units_digits_734_99_347_83_l1788_178876

-- Define a function to get the units digit of a number
def unitsDigit (n : ℕ) : ℕ := n % 10

-- Define a function to get the units digit of a number raised to a power
def unitsDigitPower (base : ℕ) (exp : ℕ) : ℕ :=
  unitsDigit (unitsDigit base ^ exp)

theorem sum_units_digits_734_99_347_83 : 
  (unitsDigitPower 734 99 + unitsDigitPower 347 83) = 7 := by
  sorry

end NUMINAMATH_CALUDE_sum_units_digits_734_99_347_83_l1788_178876


namespace NUMINAMATH_CALUDE_average_monthly_income_l1788_178828

def monthly_expense_first_3 : ℕ := 1700
def monthly_expense_next_4 : ℕ := 1550
def monthly_expense_last_5 : ℕ := 1800
def annual_savings : ℕ := 5200

def total_expenses : ℕ := 
  monthly_expense_first_3 * 3 + 
  monthly_expense_next_4 * 4 + 
  monthly_expense_last_5 * 5

def total_income : ℕ := total_expenses + annual_savings

theorem average_monthly_income : 
  total_income / 12 = 2125 := by sorry

end NUMINAMATH_CALUDE_average_monthly_income_l1788_178828


namespace NUMINAMATH_CALUDE_total_pages_read_three_weeks_l1788_178889

/-- Represents the reading statistics for a week --/
structure WeeklyReading where
  books : Nat
  pages_per_book : Nat
  magazines : Nat
  pages_per_magazine : Nat
  newspapers : Nat
  pages_per_newspaper : Nat

/-- Calculates the total pages read in a week --/
def total_pages_read (w : WeeklyReading) : Nat :=
  w.books * w.pages_per_book +
  w.magazines * w.pages_per_magazine +
  w.newspapers * w.pages_per_newspaper

/-- The reading statistics for the first week --/
def week1 : WeeklyReading :=
  { books := 5
    pages_per_book := 300
    magazines := 3
    pages_per_magazine := 120
    newspapers := 2
    pages_per_newspaper := 50 }

/-- The reading statistics for the second week --/
def week2 : WeeklyReading :=
  { books := 2 * week1.books
    pages_per_book := 350
    magazines := 4
    pages_per_magazine := 150
    newspapers := 1
    pages_per_newspaper := 60 }

/-- The reading statistics for the third week --/
def week3 : WeeklyReading :=
  { books := 3 * week1.books
    pages_per_book := 400
    magazines := 5
    pages_per_magazine := 125
    newspapers := 1
    pages_per_newspaper := 70 }

/-- Theorem: The total number of pages read over three weeks is 12815 --/
theorem total_pages_read_three_weeks :
  total_pages_read week1 + total_pages_read week2 + total_pages_read week3 = 12815 := by
  sorry


end NUMINAMATH_CALUDE_total_pages_read_three_weeks_l1788_178889


namespace NUMINAMATH_CALUDE_runner_catch_up_count_l1788_178894

def num_flags : ℕ := 2015
def laps_A : ℕ := 23
def laps_B : ℕ := 13

theorem runner_catch_up_count :
  let relative_speed := laps_A - laps_B
  let catch_up_count := (relative_speed * num_flags) / (2 * num_flags)
  catch_up_count = 5 := by
  sorry

end NUMINAMATH_CALUDE_runner_catch_up_count_l1788_178894


namespace NUMINAMATH_CALUDE_d₂_equals_six_l1788_178817

/-- E(n) is the number of quintuples (b₁, b₂, b₃, b₄, b₅) of distinct integers
    with 1 ≤ bᵢ ≤ n for all i such that n divides b₁+b₂+b₃+b₄+b₅ -/
def E (n : ℕ) : ℕ := sorry

/-- p(x) is a polynomial of degree 4 that satisfies E(n) = p(n)
    for all odd integers n ≥ 7 divisible by 3 -/
noncomputable def p : ℝ → ℝ := 
  fun x => d₄ * x^4 + d₃ * x^3 + d₂ * x^2 + d₁ * x + d₀
  where
    d₄ : ℝ := sorry
    d₃ : ℝ := sorry
    d₂ : ℝ := sorry
    d₁ : ℝ := sorry
    d₀ : ℝ := sorry

theorem d₂_equals_six :
  ∀ n : ℕ, n ≥ 7 → Odd n → 3 ∣ n → E n = p n → d₂ = 6 := by sorry

end NUMINAMATH_CALUDE_d₂_equals_six_l1788_178817


namespace NUMINAMATH_CALUDE_least_upper_bound_quadratic_form_l1788_178831

theorem least_upper_bound_quadratic_form (x₁ x₂ x₃ x₄ : ℝ) (h : x₁ ≠ 0 ∨ x₂ ≠ 0 ∨ x₃ ≠ 0 ∨ x₄ ≠ 0) :
  (x₁ * x₂ + 2 * x₂ * x₃ + x₃ * x₄) / (x₁^2 + x₂^2 + x₃^2 + x₄^2) ≤ (Real.sqrt 2 + 1) / 2 ∧
  ∀ ε > 0, ∃ y₁ y₂ y₃ y₄ : ℝ, (y₁ ≠ 0 ∨ y₂ ≠ 0 ∨ y₃ ≠ 0 ∨ y₄ ≠ 0) ∧
    (y₁ * y₂ + 2 * y₂ * y₃ + y₃ * y₄) / (y₁^2 + y₂^2 + y₃^2 + y₄^2) > (Real.sqrt 2 + 1) / 2 - ε :=
by sorry

end NUMINAMATH_CALUDE_least_upper_bound_quadratic_form_l1788_178831


namespace NUMINAMATH_CALUDE_f_properties_l1788_178816

noncomputable def f (x φ : ℝ) : ℝ :=
  (1/2) * Real.sin (2*x) * Real.sin φ + Real.cos x ^ 2 * Real.cos φ + (1/2) * Real.sin (3*Real.pi/2 - φ)

theorem f_properties (φ : ℝ) (h1 : 0 < φ) (h2 : φ < Real.pi) (h3 : f (Real.pi/6) φ = 1/2) :
  (∀ x ∈ Set.Icc (Real.pi/6) ((2*Real.pi)/3), StrictMonoOn f (Set.Icc (Real.pi/6) ((2*Real.pi)/3))) ∧
  (∀ x₀ : ℝ, x₀ ∈ Set.Ioo (Real.pi/2) Real.pi → Real.sin x₀ = 3/5 → f x₀ φ = (7 - 24*Real.sqrt 3) / 100) :=
by sorry

end NUMINAMATH_CALUDE_f_properties_l1788_178816


namespace NUMINAMATH_CALUDE_imaginary_unit_fraction_l1788_178867

theorem imaginary_unit_fraction : 
  ∃ (i : ℂ), i * i = -1 ∧ (i^2019) / (1 + i) = -1/2 - 1/2 * i :=
by sorry

end NUMINAMATH_CALUDE_imaginary_unit_fraction_l1788_178867


namespace NUMINAMATH_CALUDE_only_30_40_50_is_pythagorean_triple_l1788_178874

def is_pythagorean_triple (a b c : ℕ) : Prop :=
  a * a + b * b = c * c

theorem only_30_40_50_is_pythagorean_triple :
  (is_pythagorean_triple 30 40 50) ∧
  ¬(is_pythagorean_triple 1 1 2) ∧
  ¬(is_pythagorean_triple 1 2 2) ∧
  ¬(is_pythagorean_triple 7 14 15) :=
by sorry

end NUMINAMATH_CALUDE_only_30_40_50_is_pythagorean_triple_l1788_178874


namespace NUMINAMATH_CALUDE_first_player_wins_l1788_178892

/-- Represents the state of the candy game -/
structure GameState :=
  (box1 : Nat) (box2 : Nat)

/-- Checks if a move is valid according to the game rules -/
def isValidMove (s : GameState) (newBox1 : Nat) (newBox2 : Nat) : Prop :=
  (newBox1 < s.box1 ∨ newBox2 < s.box2) ∧
  (newBox1 ≠ 0 ∧ newBox2 ≠ 0) ∧
  ¬(newBox1 % newBox2 = 0 ∨ newBox2 % newBox1 = 0)

/-- Defines a winning strategy for the first player -/
def hasWinningStrategy (initialState : GameState) : Prop :=
  ∃ (strategy : GameState → GameState),
    (∀ s : GameState, isValidMove s (strategy s).box1 (strategy s).box2) ∧
    (∀ s : GameState, ∃ n : Nat, (strategy s).box1 = 2*n ∧ (strategy s).box2 = 2*n + 1) ∧
    (∀ s : GameState, ∀ move : GameState, 
      isValidMove s move.box1 move.box2 → 
      isValidMove (strategy move) (strategy (strategy move)).box1 (strategy (strategy move)).box2)

/-- The main theorem stating that the first player has a winning strategy -/
theorem first_player_wins :
  hasWinningStrategy ⟨2017, 2018⟩ :=
sorry

end NUMINAMATH_CALUDE_first_player_wins_l1788_178892


namespace NUMINAMATH_CALUDE_four_numbers_problem_l1788_178825

theorem four_numbers_problem (A B C D : ℤ) : 
  A + B + C + D = 43 ∧ 
  2 * A + 8 = 3 * B ∧ 
  3 * B = 4 * C ∧ 
  4 * C = 5 * D - 4 →
  A = 14 ∧ B = 12 ∧ C = 9 ∧ D = 8 := by
sorry

end NUMINAMATH_CALUDE_four_numbers_problem_l1788_178825


namespace NUMINAMATH_CALUDE_problem_solution_l1788_178827

noncomputable section

def e : ℝ := Real.exp 1

def f (x : ℝ) : ℝ := x * Real.log x

def g (x : ℝ) : ℝ := f x + x^2 - 2*(e+1)*x + 6

theorem problem_solution :
  (∃ x₀ ∈ Set.Icc 1 e, ∀ m : ℝ, m * (f x₀ - 1) > x₀^2 + 1 → m < -2 ∨ m > (e^2 + 1) / (e - 1)) ∧
  (∀ a : ℝ, (∃ x y : ℝ, x ≠ y ∧ g x = a ∧ g y = a) → a ∈ Set.Ioo (6 - e^2 - e) 6) ∧
  (HasDerivAt g 0 e) := by sorry

end

end NUMINAMATH_CALUDE_problem_solution_l1788_178827


namespace NUMINAMATH_CALUDE_triangle_inequality_l1788_178866

theorem triangle_inequality (a b c Δ : ℝ) (h_positive : a > 0 ∧ b > 0 ∧ c > 0) 
  (h_triangle : a + b > c ∧ b + c > a ∧ c + a > b) (h_area : Δ > 0) 
  (h_heron : Δ = Real.sqrt (s * (s - a) * (s - b) * (s - c))) 
  (h_semiperimeter : s = (a + b + c) / 2) : 
  a^2 + b^2 + c^2 ≥ 4 * Real.sqrt 3 * Δ + (a - b)^2 + (b - c)^2 + (c - a)^2 := by
sorry


end NUMINAMATH_CALUDE_triangle_inequality_l1788_178866


namespace NUMINAMATH_CALUDE_smallest_prime_factor_of_1471_l1788_178830

theorem smallest_prime_factor_of_1471 :
  (Nat.minFac 1471 = 13) := by
  sorry

end NUMINAMATH_CALUDE_smallest_prime_factor_of_1471_l1788_178830


namespace NUMINAMATH_CALUDE_probability_both_genders_selected_l1788_178804

/-- The probability of selecting both boys and girls when randomly choosing 3 students from 2 boys and 3 girls -/
theorem probability_both_genders_selected (total_students : ℕ) (boys : ℕ) (girls : ℕ) (selected : ℕ) : 
  total_students = boys + girls →
  boys = 2 →
  girls = 3 →
  selected = 3 →
  (1 - (Nat.choose girls selected : ℚ) / (Nat.choose total_students selected : ℚ)) = 9/10 :=
by sorry

end NUMINAMATH_CALUDE_probability_both_genders_selected_l1788_178804


namespace NUMINAMATH_CALUDE_shaded_area_ratio_l1788_178879

theorem shaded_area_ratio (square_side : ℝ) (h : square_side = 8) :
  let r := square_side / 2
  let semicircle_area := π * r^2 / 2
  let quarter_circle_area := π * r^2 / 4
  let shaded_area := 2 * semicircle_area - quarter_circle_area
  let full_circle_area := π * r^2
  shaded_area / full_circle_area = 3 / 4 := by
sorry

end NUMINAMATH_CALUDE_shaded_area_ratio_l1788_178879


namespace NUMINAMATH_CALUDE_events_mutually_exclusive_not_complementary_l1788_178803

-- Define the set of people
inductive Person : Type
  | A | B | C | D

-- Define the set of balls
inductive Ball : Type
  | one | two | three | four

-- Define a distribution as a function from Person to Ball
def Distribution := Person → Ball

-- Define the event "Person A gets ball number 1"
def event_A (d : Distribution) : Prop := d Person.A = Ball.one

-- Define the event "Person B gets ball number 1"
def event_B (d : Distribution) : Prop := d Person.B = Ball.one

-- Define mutually exclusive events
def mutually_exclusive (e1 e2 : Distribution → Prop) : Prop :=
  ∀ d : Distribution, ¬(e1 d ∧ e2 d)

-- Define complementary events
def complementary (e1 e2 : Distribution → Prop) : Prop :=
  ∀ d : Distribution, e1 d ↔ ¬(e2 d)

-- Theorem statement
theorem events_mutually_exclusive_not_complementary :
  mutually_exclusive event_A event_B ∧ ¬(complementary event_A event_B) := by
  sorry

end NUMINAMATH_CALUDE_events_mutually_exclusive_not_complementary_l1788_178803


namespace NUMINAMATH_CALUDE_correct_stratified_sample_l1788_178808

/-- Represents a camp in the summer program -/
structure Camp where
  size : ℕ
  deriving Repr

/-- Represents the summer program -/
structure SummerProgram where
  totalStudents : ℕ
  camps : List Camp
  sampleSize : ℕ
  deriving Repr

/-- Calculates the number of students to be sampled from each camp -/
def stratifiedSample (program : SummerProgram) : List ℕ :=
  program.camps.map (fun camp => 
    (camp.size * program.sampleSize) / program.totalStudents)

/-- Theorem stating the correct stratified sampling for the given summer program -/
theorem correct_stratified_sample :
  let program : SummerProgram := {
    totalStudents := 500,
    camps := [{ size := 200 }, { size := 150 }, { size := 150 }],
    sampleSize := 50
  }
  stratifiedSample program = [20, 15, 15] := by sorry

end NUMINAMATH_CALUDE_correct_stratified_sample_l1788_178808


namespace NUMINAMATH_CALUDE_euclidean_division_remainder_congruence_l1788_178822

theorem euclidean_division_remainder_congruence 
  (a b : ℤ) (d : ℕ) (h : d ≠ 0) 
  (a' b' : ℕ) 
  (ha : a' = a % d)
  (hb : b' = b % d) : 
  (a ≡ b [ZMOD d]) ↔ (a' = b') :=
sorry

end NUMINAMATH_CALUDE_euclidean_division_remainder_congruence_l1788_178822


namespace NUMINAMATH_CALUDE_complex_imaginary_part_l1788_178883

theorem complex_imaginary_part (Z : ℂ) (h1 : Z.re = 1) (h2 : Complex.abs Z = 2) :
  Z.im = Real.sqrt 3 ∨ Z.im = -Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_complex_imaginary_part_l1788_178883


namespace NUMINAMATH_CALUDE_triangular_array_coin_sum_l1788_178868

/-- The number of rows in the triangular array -/
def N : ℕ := 77

/-- The total number of coins in the triangular array -/
def total_coins : ℕ := 3003

/-- Function to calculate the sum of digits of a natural number -/
def sum_of_digits (n : ℕ) : ℕ :=
  if n < 10 then n else (n % 10) + sum_of_digits (n / 10)

/-- Theorem stating the properties of the triangular array and the sum of digits of N -/
theorem triangular_array_coin_sum :
  (N * (N + 1)) / 2 = total_coins ∧ sum_of_digits N = 14 := by
  sorry

#eval sum_of_digits N

end NUMINAMATH_CALUDE_triangular_array_coin_sum_l1788_178868


namespace NUMINAMATH_CALUDE_east_northwest_angle_is_144_degrees_l1788_178815

/-- Represents a circular garden with equally spaced radial paths -/
structure CircularGarden where
  numPaths : ℕ
  northPathIndex : ℕ
  eastPathIndex : ℕ
  northwestPathIndex : ℕ

/-- Calculates the angle between two paths in a circular garden -/
def angleBetweenPaths (garden : CircularGarden) (path1 : ℕ) (path2 : ℕ) : ℝ :=
  let angleBetweenConsecutivePaths := 360 / garden.numPaths
  let pathDifference := (path2 - path1 + garden.numPaths) % garden.numPaths
  pathDifference * angleBetweenConsecutivePaths

/-- Theorem stating that the smaller angle between East and Northwest paths is 144 degrees -/
theorem east_northwest_angle_is_144_degrees (garden : CircularGarden) :
  garden.numPaths = 10 →
  garden.northPathIndex = 0 →
  garden.eastPathIndex = 3 →
  garden.northwestPathIndex = 8 →
  min (angleBetweenPaths garden garden.eastPathIndex garden.northwestPathIndex)
      (angleBetweenPaths garden garden.northwestPathIndex garden.eastPathIndex) = 144 :=
by
  sorry

end NUMINAMATH_CALUDE_east_northwest_angle_is_144_degrees_l1788_178815


namespace NUMINAMATH_CALUDE_function_equivalence_l1788_178839

-- Define the function f
def f : ℝ → ℝ := fun x => x^2 + 2*x

-- State the theorem
theorem function_equivalence : ∀ x : ℝ, f (x - 1) = x^2 - 1 := by
  sorry

end NUMINAMATH_CALUDE_function_equivalence_l1788_178839


namespace NUMINAMATH_CALUDE_triangle_shape_l1788_178873

theorem triangle_shape (a b : ℝ) (A B : ℝ) :
  0 < A → A < π →
  0 < B → B < π →
  a^2 * Real.tan B = b^2 * Real.tan A →
  (A = B ∨ A + B = π / 2) :=
sorry

end NUMINAMATH_CALUDE_triangle_shape_l1788_178873


namespace NUMINAMATH_CALUDE_expression_evaluation_l1788_178834

theorem expression_evaluation :
  let x : ℝ := Real.sqrt 2 - 1
  let expr := ((x / (x - 1)) - (x / (x^2 - 1))) / ((x^2 - x) / (x^2 - 2*x + 1))
  expr = 1 - Real.sqrt 2 / 2 := by
sorry

end NUMINAMATH_CALUDE_expression_evaluation_l1788_178834


namespace NUMINAMATH_CALUDE_power_product_equality_l1788_178858

theorem power_product_equality (x : ℝ) : 2 * (x^3 * x^2) = 2 * x^5 := by
  sorry

end NUMINAMATH_CALUDE_power_product_equality_l1788_178858


namespace NUMINAMATH_CALUDE_range_of_a_l1788_178805

theorem range_of_a (a : ℝ) : 
  (∀ x ∈ Set.Icc 0 1, a ≥ Real.exp x) ∧ 
  (∃ x : ℝ, x^2 + 4*x + a = 0) → 
  a ∈ Set.Icc (Real.exp 1) 4 := by
sorry

end NUMINAMATH_CALUDE_range_of_a_l1788_178805


namespace NUMINAMATH_CALUDE_vector_sum_equals_c_l1788_178881

def a : ℝ × ℝ := (1, -1)
def b : ℝ × ℝ := (-1, 1)
def c : ℝ × ℝ := (5, 1)

theorem vector_sum_equals_c : c + a + b = c := by sorry

end NUMINAMATH_CALUDE_vector_sum_equals_c_l1788_178881


namespace NUMINAMATH_CALUDE_arithmetic_computation_l1788_178835

theorem arithmetic_computation : 2 + 8 * 3 - 4 + 7 * 6 / 3 = 36 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_computation_l1788_178835


namespace NUMINAMATH_CALUDE_cuts_for_331_pieces_l1788_178842

/-- The number of cuts needed to transform initial sheets into a given number of pieces -/
def number_of_cuts (initial_sheets : ℕ) (final_pieces : ℕ) : ℕ :=
  (final_pieces - initial_sheets) / 6

/-- Theorem stating that 54 cuts are needed to transform 7 sheets into 331 pieces -/
theorem cuts_for_331_pieces : number_of_cuts 7 331 = 54 := by
  sorry

end NUMINAMATH_CALUDE_cuts_for_331_pieces_l1788_178842


namespace NUMINAMATH_CALUDE_tangent_equality_implies_x_130_l1788_178814

theorem tangent_equality_implies_x_130 (x : ℝ) :
  0 < x → x < 180 →
  Real.tan ((150 - x) * π / 180) = 
    (Real.sin (150 * π / 180) - Real.sin (x * π / 180)) / 
    (Real.cos (150 * π / 180) - Real.cos (x * π / 180)) →
  x = 130 := by sorry

end NUMINAMATH_CALUDE_tangent_equality_implies_x_130_l1788_178814


namespace NUMINAMATH_CALUDE_mean_interior_angles_quadrilateral_l1788_178820

-- Define a quadrilateral
def Quadrilateral : Type := Unit

-- Define the function that gives the sum of interior angles of a quadrilateral
def sum_interior_angles (q : Quadrilateral) : ℝ := 360

-- Define the number of interior angles in a quadrilateral
def num_interior_angles (q : Quadrilateral) : ℕ := 4

-- Theorem: The mean value of the measures of the four interior angles of any quadrilateral is 90°
theorem mean_interior_angles_quadrilateral (q : Quadrilateral) :
  (sum_interior_angles q) / (num_interior_angles q : ℝ) = 90 := by sorry

end NUMINAMATH_CALUDE_mean_interior_angles_quadrilateral_l1788_178820


namespace NUMINAMATH_CALUDE_probability_at_least_one_black_l1788_178812

def total_balls : ℕ := 10
def red_balls : ℕ := 6
def black_balls : ℕ := 4
def selected_balls : ℕ := 4

theorem probability_at_least_one_black :
  let total_ways := Nat.choose total_balls selected_balls
  let all_red_ways := Nat.choose red_balls selected_balls
  let at_least_one_black_ways := total_ways - all_red_ways
  (at_least_one_black_ways : ℚ) / total_ways = 13 / 14 := by
  sorry

end NUMINAMATH_CALUDE_probability_at_least_one_black_l1788_178812


namespace NUMINAMATH_CALUDE_least_prime_factor_of_eight_cubed_minus_eight_squared_l1788_178853

theorem least_prime_factor_of_eight_cubed_minus_eight_squared :
  Nat.minFac (8^3 - 8^2) = 2 := by
  sorry

end NUMINAMATH_CALUDE_least_prime_factor_of_eight_cubed_minus_eight_squared_l1788_178853


namespace NUMINAMATH_CALUDE_total_tickets_sold_l1788_178869

/-- Represents the price of an adult ticket in dollars -/
def adult_price : ℝ := 4

/-- Represents the price of a student ticket in dollars -/
def student_price : ℝ := 2.5

/-- Represents the total revenue from ticket sales in dollars -/
def total_revenue : ℝ := 222.5

/-- Represents the number of student tickets sold -/
def student_tickets : ℕ := 9

/-- Theorem stating the total number of tickets sold -/
theorem total_tickets_sold : ℕ := by
  sorry

end NUMINAMATH_CALUDE_total_tickets_sold_l1788_178869


namespace NUMINAMATH_CALUDE_expected_value_of_event_A_l1788_178837

theorem expected_value_of_event_A (p : ℝ) (h1 : (1 - p) ^ 4 = 16 / 81) :
  4 * p = 4 / 3 := by
  sorry

end NUMINAMATH_CALUDE_expected_value_of_event_A_l1788_178837


namespace NUMINAMATH_CALUDE_min_trees_for_three_types_l1788_178864

/-- Represents the four types of trees in the grove -/
inductive TreeType
  | Birch
  | Spruce
  | Pine
  | Aspen

/-- Represents the grove of trees -/
structure Grove :=
  (trees : Finset TreeType)
  (total_count : ℕ)
  (type_count : TreeType → ℕ)
  (total_is_100 : total_count = 100)
  (sum_of_types : (type_count TreeType.Birch) + (type_count TreeType.Spruce) + (type_count TreeType.Pine) + (type_count TreeType.Aspen) = total_count)
  (all_types_in_85 : ∀ (subset : Finset TreeType), subset.card = 85 → (∀ t : TreeType, t ∈ subset))

/-- The main theorem to be proved -/
theorem min_trees_for_three_types (g : Grove) :
  ∃ (n : ℕ), n = 69 ∧ 
  (∀ (subset : Finset TreeType), subset.card ≥ n → (∃ (t1 t2 t3 : TreeType), t1 ≠ t2 ∧ t2 ≠ t3 ∧ t1 ≠ t3 ∧ t1 ∈ subset ∧ t2 ∈ subset ∧ t3 ∈ subset)) ∧
  (∃ (subset : Finset TreeType), subset.card = n - 1 ∧ (∀ (t1 t2 t3 : TreeType), t1 ≠ t2 → t2 ≠ t3 → t1 ≠ t3 → ¬(t1 ∈ subset ∧ t2 ∈ subset ∧ t3 ∈ subset))) :=
sorry

end NUMINAMATH_CALUDE_min_trees_for_three_types_l1788_178864


namespace NUMINAMATH_CALUDE_polynomial_simplification_l1788_178863

theorem polynomial_simplification (x : ℝ) :
  (2 * x^2 + 5 * x - 3) - (2 * x^2 + 9 * x - 7) = -4 * x + 4 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_simplification_l1788_178863


namespace NUMINAMATH_CALUDE_hexagon_area_decrease_l1788_178871

/-- The area decrease of a regular hexagon when its sides are shortened -/
theorem hexagon_area_decrease (initial_area : ℝ) (side_decrease : ℝ) : 
  initial_area = 150 * Real.sqrt 3 →
  side_decrease = 3 →
  let original_side := Real.sqrt (200 / 3)
  let new_side := original_side - side_decrease
  let new_area := 3 * Real.sqrt 3 / 2 * new_side ^ 2
  initial_area - new_area = 76.5 * Real.sqrt 3 := by
sorry


end NUMINAMATH_CALUDE_hexagon_area_decrease_l1788_178871


namespace NUMINAMATH_CALUDE_base_subtraction_l1788_178878

/-- Convert a number from base b to base 10 -/
def to_base_10 (digits : List Nat) (b : Nat) : Nat :=
  digits.enum.foldr (fun (i, d) acc => acc + d * b ^ i) 0

/-- The result of subtracting the base 10 representation of 243 in base 6
    from the base 10 representation of 325 in base 9 is 167 -/
theorem base_subtraction :
  to_base_10 [5, 2, 3] 9 - to_base_10 [3, 4, 2] 6 = 167 := by
  sorry

end NUMINAMATH_CALUDE_base_subtraction_l1788_178878


namespace NUMINAMATH_CALUDE_max_consecutive_irreducible_l1788_178852

def is_five_digit (n : ℕ) : Prop := 10000 ≤ n ∧ n ≤ 99999

def is_three_digit (n : ℕ) : Prop := 100 ≤ n ∧ n ≤ 999

def is_irreducible (n : ℕ) : Prop :=
  is_five_digit n ∧ ∀ a b : ℕ, is_three_digit a → is_three_digit b → n ≠ a * b

def consecutive_irreducible (start : ℕ) (count : ℕ) : Prop :=
  ∀ i : ℕ, i < count → is_irreducible (start + i)

theorem max_consecutive_irreducible :
  ∃ start : ℕ, consecutive_irreducible start 99 ∧
  ∀ start' count' : ℕ, count' > 99 → ¬(consecutive_irreducible start' count') :=
sorry

end NUMINAMATH_CALUDE_max_consecutive_irreducible_l1788_178852


namespace NUMINAMATH_CALUDE_first_five_terms_of_series_l1788_178862

def a (n : ℕ+) : ℚ := 1 / (n * (n + 1))

theorem first_five_terms_of_series :
  (List.range 5).map (fun i => a ⟨i + 1, Nat.succ_pos i⟩) = [1/2, 1/6, 1/12, 1/20, 1/30] := by
  sorry

end NUMINAMATH_CALUDE_first_five_terms_of_series_l1788_178862


namespace NUMINAMATH_CALUDE_f_monotonicity_and_m_range_l1788_178806

noncomputable def f (a b x : ℝ) : ℝ := a * Real.log x - b * x

theorem f_monotonicity_and_m_range :
  ∀ (a b : ℝ),
  (∀ (x : ℝ), x > 0 → f a b x = a * Real.log x - b * x) →
  (b = 1 →
    ((a ≤ 0 → ∀ (x y : ℝ), 0 < x → x < y → f a b y < f a b x) ∧
     (a > 0 → (∀ (x y : ℝ), 0 < x → x < y → y < a → f a b x < f a b y) ∧
              (∀ (x y : ℝ), a < x → x < y → f a b y < f a b x)))) ∧
  (a = 1 →
    (∀ (x : ℝ), x > 0 → f a b x ≤ -1) →
    (∀ (x : ℝ), x > 0 → f a b x ≤ x * Real.exp x - (b + 1) * x - 1) →
    b ≤ 1) :=
by sorry

end NUMINAMATH_CALUDE_f_monotonicity_and_m_range_l1788_178806


namespace NUMINAMATH_CALUDE_ten_integer_segments_l1788_178896

/-- Represents a right triangle ABC with integer side lengths -/
structure RightTriangle where
  ab : ℕ
  bc : ℕ

/-- Counts the number of distinct integer lengths of line segments
    from vertex B to the hypotenuse AC in a right triangle -/
def count_integer_segments (t : RightTriangle) : ℕ :=
  sorry

/-- The main theorem stating that for a right triangle with sides 18 and 24,
    there are exactly 10 distinct integer lengths of segments from B to AC -/
theorem ten_integer_segments :
  ∃ (t : RightTriangle), t.ab = 18 ∧ t.bc = 24 ∧ count_integer_segments t = 10 :=
sorry

end NUMINAMATH_CALUDE_ten_integer_segments_l1788_178896


namespace NUMINAMATH_CALUDE_solution_set_part_i_range_of_a_part_ii_l1788_178838

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := |x - 4| + |x - a|

-- Part I
theorem solution_set_part_i :
  {x : ℝ | f 2 x > 10} = {x : ℝ | x > 8 ∨ x < -2} := by sorry

-- Part II
theorem range_of_a_part_ii :
  (∀ x : ℝ, f a x ≥ 1) → (a ≥ 5 ∨ a ≤ 3) := by sorry

end NUMINAMATH_CALUDE_solution_set_part_i_range_of_a_part_ii_l1788_178838


namespace NUMINAMATH_CALUDE_seating_arrangements_equals_60_l1788_178855

/-- The number of ways to arrange 3 people in a row of 9 seats,
    with empty seats on both sides of each person. -/
def seating_arrangements (total_seats : ℕ) (people : ℕ) : ℕ :=
  let gaps := total_seats - people - people
  let combinations := Nat.choose gaps people
  combinations * Nat.factorial people

/-- Theorem stating that the number of seating arrangements
    for 3 people in 9 seats with required spacing is 60. -/
theorem seating_arrangements_equals_60 :
  seating_arrangements 9 3 = 60 := by
  sorry

end NUMINAMATH_CALUDE_seating_arrangements_equals_60_l1788_178855


namespace NUMINAMATH_CALUDE_cubic_two_roots_l1788_178810

/-- The cubic function we're analyzing -/
def f (d : ℝ) (x : ℝ) : ℝ := x^3 - 3*x + d

/-- A function has exactly two roots -/
def has_exactly_two_roots (g : ℝ → ℝ) : Prop :=
  ∃ (a b : ℝ), a ≠ b ∧ g a = 0 ∧ g b = 0 ∧ ∀ x, g x = 0 → x = a ∨ x = b

/-- If f(x) = x^3 - 3x + d has exactly two roots, then d = 2 or d = -2 -/
theorem cubic_two_roots (d : ℝ) : has_exactly_two_roots (f d) → d = 2 ∨ d = -2 := by
  sorry

end NUMINAMATH_CALUDE_cubic_two_roots_l1788_178810


namespace NUMINAMATH_CALUDE_complex_solutions_count_l1788_178895

open Complex

theorem complex_solutions_count : 
  let f : ℂ → ℂ := λ z => (z^4 - 1) / (z^3 + z^2 - 3*z - 3)
  ∃ (S : Finset ℂ), (∀ z ∈ S, f z = 0) ∧ (∀ z ∉ S, f z ≠ 0) ∧ Finset.card S = 3 :=
by sorry

end NUMINAMATH_CALUDE_complex_solutions_count_l1788_178895


namespace NUMINAMATH_CALUDE_events_mutually_exclusive_not_contradictory_l1788_178893

/-- A bag containing red and white balls -/
structure Bag where
  red : ℕ
  white : ℕ

/-- The number of balls drawn from the bag -/
def drawn : ℕ := 3

/-- The initial bag configuration -/
def initial_bag : Bag := ⟨5, 3⟩

/-- Event: Exactly one red ball is drawn -/
def exactly_one_red (b : Bag) : Prop := sorry

/-- Event: Exactly two red balls are drawn -/
def exactly_two_red (b : Bag) : Prop := sorry

/-- Two events are mutually exclusive -/
def mutually_exclusive (e1 e2 : Bag → Prop) : Prop := sorry

/-- Two events are contradictory -/
def contradictory (e1 e2 : Bag → Prop) : Prop := sorry

theorem events_mutually_exclusive_not_contradictory :
  mutually_exclusive exactly_one_red exactly_two_red ∧
  ¬contradictory exactly_one_red exactly_two_red :=
sorry

end NUMINAMATH_CALUDE_events_mutually_exclusive_not_contradictory_l1788_178893


namespace NUMINAMATH_CALUDE_beaver_carrot_count_l1788_178844

/-- Represents the number of carrots stored per burrow by the beaver -/
def beaver_carrots_per_burrow : ℕ := 4

/-- Represents the number of carrots stored per burrow by the rabbit -/
def rabbit_carrots_per_burrow : ℕ := 5

/-- Represents the difference in the number of burrows between the beaver and the rabbit -/
def burrow_difference : ℕ := 3

theorem beaver_carrot_count (beaver_burrows rabbit_burrows total_carrots : ℕ) :
  beaver_burrows = rabbit_burrows + burrow_difference →
  beaver_carrots_per_burrow * beaver_burrows = total_carrots →
  rabbit_carrots_per_burrow * rabbit_burrows = total_carrots →
  total_carrots = 60 := by
  sorry

end NUMINAMATH_CALUDE_beaver_carrot_count_l1788_178844


namespace NUMINAMATH_CALUDE_total_students_from_stratified_sample_l1788_178872

/-- Given a stratified sample from a high school population, prove the total number of students. -/
theorem total_students_from_stratified_sample
  (total_sample : ℕ)
  (first_year_sample : ℕ)
  (third_year_sample : ℕ)
  (total_second_year : ℕ)
  (h1 : total_sample = 45)
  (h2 : first_year_sample = 20)
  (h3 : third_year_sample = 10)
  (h4 : total_second_year = 300) :
  ∃ (total_students : ℕ), total_students = 900 := by
  sorry

end NUMINAMATH_CALUDE_total_students_from_stratified_sample_l1788_178872


namespace NUMINAMATH_CALUDE_ganzhi_2019_l1788_178851

-- Define the Heavenly Stems
inductive HeavenlyStem
| Jia | Yi | Bing | Ding | Wu | Ji | Geng | Xin | Ren | Gui

-- Define the Earthly Branches
inductive EarthlyBranch
| Zi | Chou | Yin | Mao | Chen | Si | Wu | Wei | Shen | You | Xu | Hai

-- Define the Ganzhi combination
structure Ganzhi where
  stem : HeavenlyStem
  branch : EarthlyBranch

-- Define a function to get the next Heavenly Stem
def nextStem (s : HeavenlyStem) : HeavenlyStem :=
  match s with
  | HeavenlyStem.Jia => HeavenlyStem.Yi
  | HeavenlyStem.Yi => HeavenlyStem.Bing
  | HeavenlyStem.Bing => HeavenlyStem.Ding
  | HeavenlyStem.Ding => HeavenlyStem.Wu
  | HeavenlyStem.Wu => HeavenlyStem.Ji
  | HeavenlyStem.Ji => HeavenlyStem.Geng
  | HeavenlyStem.Geng => HeavenlyStem.Xin
  | HeavenlyStem.Xin => HeavenlyStem.Ren
  | HeavenlyStem.Ren => HeavenlyStem.Gui
  | HeavenlyStem.Gui => HeavenlyStem.Jia

-- Define a function to get the next Earthly Branch
def nextBranch (b : EarthlyBranch) : EarthlyBranch :=
  match b with
  | EarthlyBranch.Zi => EarthlyBranch.Chou
  | EarthlyBranch.Chou => EarthlyBranch.Yin
  | EarthlyBranch.Yin => EarthlyBranch.Mao
  | EarthlyBranch.Mao => EarthlyBranch.Chen
  | EarthlyBranch.Chen => EarthlyBranch.Si
  | EarthlyBranch.Si => EarthlyBranch.Wu
  | EarthlyBranch.Wu => EarthlyBranch.Wei
  | EarthlyBranch.Wei => EarthlyBranch.Shen
  | EarthlyBranch.Shen => EarthlyBranch.You
  | EarthlyBranch.You => EarthlyBranch.Xu
  | EarthlyBranch.Xu => EarthlyBranch.Hai
  | EarthlyBranch.Hai => EarthlyBranch.Zi

-- Define a function to get the next Ganzhi combination
def nextGanzhi (g : Ganzhi) : Ganzhi :=
  { stem := nextStem g.stem, branch := nextBranch g.branch }

-- Define a function to advance Ganzhi by n years
def advanceGanzhi (g : Ganzhi) (n : Nat) : Ganzhi :=
  match n with
  | 0 => g
  | n + 1 => advanceGanzhi (nextGanzhi g) n

-- Theorem statement
theorem ganzhi_2019 (ganzhi_2010 : Ganzhi)
  (h2010 : ganzhi_2010 = { stem := HeavenlyStem.Geng, branch := EarthlyBranch.Yin }) :
  advanceGanzhi ganzhi_2010 9 = { stem := HeavenlyStem.Ji, branch := EarthlyBranch.You } :=
by sorry


end NUMINAMATH_CALUDE_ganzhi_2019_l1788_178851


namespace NUMINAMATH_CALUDE_denise_expenditure_l1788_178813

/-- Represents the menu items --/
inductive MenuItem
| SimpleDish
| MeatDish
| FishDish
| MilkSmoothie
| FruitSmoothie
| SpecialSmoothie

/-- Price of a menu item in reais --/
def price (item : MenuItem) : ℕ :=
  match item with
  | MenuItem.SimpleDish => 7
  | MenuItem.MeatDish => 11
  | MenuItem.FishDish => 14
  | MenuItem.MilkSmoothie => 6
  | MenuItem.FruitSmoothie => 7
  | MenuItem.SpecialSmoothie => 9

/-- Total cost of a meal (one dish and one smoothie) --/
def mealCost (dish : MenuItem) (smoothie : MenuItem) : ℕ :=
  price dish + price smoothie

/-- Denise's possible expenditures --/
def deniseExpenditure : Set ℕ :=
  {14, 17}

/-- Theorem stating Denise's possible expenditures --/
theorem denise_expenditure :
  ∀ (deniseDish deniseSmoothie julioDish julioSmoothie : MenuItem),
    mealCost julioDish julioSmoothie = mealCost deniseDish deniseSmoothie + 6 →
    mealCost deniseDish deniseSmoothie ∈ deniseExpenditure :=
by sorry

end NUMINAMATH_CALUDE_denise_expenditure_l1788_178813


namespace NUMINAMATH_CALUDE_divisibility_after_subtraction_l1788_178832

theorem divisibility_after_subtraction :
  ∃ (n : ℕ), n = 15 ∧ (427398 - 3) % n = 0 :=
by
  sorry

end NUMINAMATH_CALUDE_divisibility_after_subtraction_l1788_178832


namespace NUMINAMATH_CALUDE_g_sum_symmetric_l1788_178821

def g (x : ℝ) : ℝ := 2 * x^6 + 3 * x^4 - x^2 + 7

theorem g_sum_symmetric (h : g 5 = 29) : g 5 + g (-5) = 58 := by
  sorry

end NUMINAMATH_CALUDE_g_sum_symmetric_l1788_178821


namespace NUMINAMATH_CALUDE_smallest_valid_number_l1788_178841

def is_valid_number (n : Nat) : Prop :=
  n ≥ 100000 ∧ n < 1000000 ∧
  ∀ i : Nat, i ∈ [0, 1, 2, 3] →
    let three_digit := (n / 10^i) % 1000
    three_digit % 6 = 0 ∨ three_digit % 7 = 0

theorem smallest_valid_number :
  is_valid_number 112642 ∧
  ∀ m : Nat, m < 112642 → ¬(is_valid_number m) :=
sorry

end NUMINAMATH_CALUDE_smallest_valid_number_l1788_178841


namespace NUMINAMATH_CALUDE_reflection_set_bounded_l1788_178860

/-- A type representing a point in a plane -/
structure Point :=
  (x : ℝ) (y : ℝ)

/-- The set of points generated by the reflection process -/
def ReflectionSet (A B C : Point) : Set Point :=
  sorry

/-- A line in the plane -/
structure Line :=
  (a b c : ℝ)

/-- Predicate to check if a point is on one side of a line -/
def onOneSideOfLine (p : Point) (l : Line) : Prop :=
  sorry

theorem reflection_set_bounded (A B C : Point) (hDistinct : A ≠ B ∧ B ≠ C ∧ C ≠ A) :
  ∃ (l : Line), ∀ (p : Point), p ∈ ReflectionSet A B C → onOneSideOfLine p l :=
sorry

end NUMINAMATH_CALUDE_reflection_set_bounded_l1788_178860


namespace NUMINAMATH_CALUDE_no_persistent_numbers_l1788_178887

/-- A number is persistent if, when multiplied by any positive integer, 
    the result always contains all ten digits 0,1,...,9. -/
def IsPersistent (n : ℕ) : Prop :=
  ∀ k : ℕ+, ∀ d : Fin 10, ∃ m : ℕ, (n * k : ℕ) / 10^m % 10 = d

/-- There are no persistent numbers. -/
theorem no_persistent_numbers : ¬∃ n : ℕ, IsPersistent n := by
  sorry


end NUMINAMATH_CALUDE_no_persistent_numbers_l1788_178887


namespace NUMINAMATH_CALUDE_neds_weekly_sales_l1788_178856

def normal_mouse_price : ℝ := 120
def normal_keyboard_price : ℝ := 80
def normal_scissors_price : ℝ := 30

def left_handed_mouse_price : ℝ := normal_mouse_price * 1.3
def left_handed_keyboard_price : ℝ := normal_keyboard_price * 1.2
def left_handed_scissors_price : ℝ := normal_scissors_price * 1.5

def daily_mouse_sales : ℝ := 25
def daily_keyboard_sales : ℝ := 10
def daily_scissors_sales : ℝ := 15
def daily_bundle_sales : ℝ := 5

def bundle_price : ℝ := (left_handed_mouse_price + left_handed_keyboard_price + left_handed_scissors_price) * 0.9

def regular_open_days : ℕ := 3
def extended_open_days : ℕ := 1
def extended_day_multiplier : ℝ := 1.5

def total_weekly_sales : ℝ :=
  (daily_mouse_sales * left_handed_mouse_price +
   daily_keyboard_sales * left_handed_keyboard_price +
   daily_scissors_sales * left_handed_scissors_price +
   daily_bundle_sales * bundle_price) *
  (regular_open_days + extended_open_days * extended_day_multiplier)

theorem neds_weekly_sales :
  total_weekly_sales = 29922.25 := by sorry

end NUMINAMATH_CALUDE_neds_weekly_sales_l1788_178856


namespace NUMINAMATH_CALUDE_theater_sales_result_l1788_178854

/-- Calculates the total amount collected from ticket sales for a theater performance -/
def theater_sales (adult_price child_price total_attendees children_attendees : ℕ) : ℕ :=
  let adults := total_attendees - children_attendees
  adult_price * adults + child_price * children_attendees

/-- Theorem stating that the theater collected $258 from ticket sales -/
theorem theater_sales_result : theater_sales 16 9 24 18 = 258 := by
  sorry

end NUMINAMATH_CALUDE_theater_sales_result_l1788_178854


namespace NUMINAMATH_CALUDE_converse_of_square_angles_is_false_l1788_178800

-- Define a quadrilateral
structure Quadrilateral where
  angles : Fin 4 → ℝ

-- Define a property for right angles
def has_right_angles (q : Quadrilateral) : Prop :=
  ∀ i : Fin 4, q.angles i = 90

-- Define a property for equal sides
def has_equal_sides (q : Quadrilateral) : Prop :=
  -- This is a placeholder definition, as we don't have side lengths in our structure
  True

-- Define a square
def is_square (q : Quadrilateral) : Prop :=
  has_right_angles q ∧ has_equal_sides q

-- The theorem to prove
theorem converse_of_square_angles_is_false : 
  ¬(∀ q : Quadrilateral, has_right_angles q → is_square q) := by
  sorry

end NUMINAMATH_CALUDE_converse_of_square_angles_is_false_l1788_178800


namespace NUMINAMATH_CALUDE_tuesday_kids_l1788_178891

/-- The number of kids Julia played with on Monday -/
def monday_kids : ℕ := 11

/-- The additional number of kids Julia played with on Tuesday compared to Monday -/
def additional_kids : ℕ := 1

/-- Theorem stating the number of kids Julia played with on Tuesday -/
theorem tuesday_kids : monday_kids + additional_kids = 12 := by
  sorry

end NUMINAMATH_CALUDE_tuesday_kids_l1788_178891


namespace NUMINAMATH_CALUDE_square_difference_given_sum_and_weighted_sum_l1788_178826

theorem square_difference_given_sum_and_weighted_sum (x y : ℝ) 
  (h1 : x + y = 15) 
  (h2 : 3 * x + y = 20) : 
  x^2 - y^2 = -150 := by
  sorry

end NUMINAMATH_CALUDE_square_difference_given_sum_and_weighted_sum_l1788_178826


namespace NUMINAMATH_CALUDE_polynomial_product_bound_l1788_178843

/-- A polynomial with real coefficients -/
def RealPolynomial := ℝ → ℝ

/-- The property that |P(x) * P(1/x)| ≤ 1 for all positive real x -/
def HasBoundedProduct (P : RealPolynomial) : Prop :=
  ∀ x : ℝ, x > 0 → |P x * P (1/x)| ≤ 1

/-- The form c * x^n where |c| ≤ 1 and n is a non-negative integer -/
def IsMonomial (P : RealPolynomial) : Prop :=
  ∃ (c : ℝ) (n : ℕ), (|c| ≤ 1) ∧ (∀ x : ℝ, P x = c * x^n)

/-- The main theorem -/
theorem polynomial_product_bound (P : RealPolynomial) :
  HasBoundedProduct P → IsMonomial P := by
  sorry

end NUMINAMATH_CALUDE_polynomial_product_bound_l1788_178843


namespace NUMINAMATH_CALUDE_intersection_complement_equal_l1788_178899

open Set

def U : Finset Nat := {1,2,3,4,5}
def M : Finset Nat := {1,4}
def N : Finset Nat := {1,3,5}

theorem intersection_complement_equal : N ∩ (U \ M) = {3,5} := by
  sorry

end NUMINAMATH_CALUDE_intersection_complement_equal_l1788_178899


namespace NUMINAMATH_CALUDE_not_universal_quantifier_negation_equivalence_not_necessary_not_sufficient_sufficient_not_necessary_l1788_178845

-- Define the proposition
def P : Prop := ∃ x : ℝ, x^2 + x + 1 = 0

-- Statement 1
theorem not_universal_quantifier : ¬(∀ x : ℝ, x^2 + x + 1 = 0) := by sorry

-- Statement 2
theorem negation_equivalence : 
  (¬∃ x : ℝ, x + 1 ≤ 2) ↔ (∀ x : ℝ, x + 1 > 2) := by sorry

-- Statement 3
theorem not_necessary_not_sufficient (A B : Set ℝ) :
  ¬(∀ x : ℝ, x ∈ A → x ∈ A ∩ B) ∧ ∀ x : ℝ, x ∈ A ∩ B → x ∈ A := by sorry

-- Statement 4
theorem sufficient_not_necessary :
  (∀ x : ℝ, x > 3 → x^2 > 9) ∧ ¬(∀ x : ℝ, x^2 > 9 → x > 3) := by sorry

end NUMINAMATH_CALUDE_not_universal_quantifier_negation_equivalence_not_necessary_not_sufficient_sufficient_not_necessary_l1788_178845


namespace NUMINAMATH_CALUDE_rectangle_side_length_l1788_178848

theorem rectangle_side_length (area : ℚ) (side1 : ℚ) (side2 : ℚ) : 
  area = 9/16 → side1 = 3/4 → side1 * side2 = area → side2 = 3/4 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_side_length_l1788_178848


namespace NUMINAMATH_CALUDE_wind_pressure_theorem_l1788_178861

/-- Represents the joint variation of pressure with area and velocity squared -/
noncomputable def pressure (k : ℝ) (A : ℝ) (V : ℝ) : ℝ := k * A * V^2

/-- Theorem stating the relationship between pressure, area, and velocity -/
theorem wind_pressure_theorem (k : ℝ) :
  (pressure k 2 20 = 4) →
  (pressure k 4 (40 * Real.sqrt 2) = 64) :=
by sorry

end NUMINAMATH_CALUDE_wind_pressure_theorem_l1788_178861


namespace NUMINAMATH_CALUDE_initial_mean_calculation_l1788_178897

/-- Given 50 observations with an initial mean, if one observation
    of 60 was wrongly recorded as 23, and the corrected mean is 36.5,
    then the initial mean is 35.76. -/
theorem initial_mean_calculation (n : ℕ) (M : ℝ) (wrong_value correct_value new_mean : ℝ) :
  n = 50 →
  wrong_value = 23 →
  correct_value = 60 →
  new_mean = 36.5 →
  ((n : ℝ) * M + (correct_value - wrong_value)) / n = new_mean →
  M = 35.76 := by
  sorry

end NUMINAMATH_CALUDE_initial_mean_calculation_l1788_178897


namespace NUMINAMATH_CALUDE_cone_height_l1788_178811

/-- Given a cone with slant height 13 cm and lateral area 65π cm², prove its height is 12 cm -/
theorem cone_height (s : ℝ) (l : ℝ) (h : ℝ) : 
  s = 13 →  -- slant height
  l = 65 * Real.pi →  -- lateral area
  l = Real.pi * s * (s^2 - h^2).sqrt →  -- formula for lateral area
  h = 12 :=
by sorry

end NUMINAMATH_CALUDE_cone_height_l1788_178811


namespace NUMINAMATH_CALUDE_eight_digit_even_integers_count_l1788_178850

/-- The set of even digits -/
def EvenDigits : Finset Nat := {0, 2, 4, 6, 8}

/-- The set of non-zero even digits -/
def NonZeroEvenDigits : Finset Nat := {2, 4, 6, 8}

/-- The number of 8-digit positive integers with all even digits -/
def EightDigitEvenIntegers : Nat :=
  Finset.card NonZeroEvenDigits * (Finset.card EvenDigits ^ 7)

theorem eight_digit_even_integers_count :
  EightDigitEvenIntegers = 312500 := by
sorry

end NUMINAMATH_CALUDE_eight_digit_even_integers_count_l1788_178850


namespace NUMINAMATH_CALUDE_entertainment_committee_combinations_l1788_178857

theorem entertainment_committee_combinations (n : ℕ) : 
  (n.choose 3 = 20) → (n.choose 2 = 15) := by
  sorry

end NUMINAMATH_CALUDE_entertainment_committee_combinations_l1788_178857


namespace NUMINAMATH_CALUDE_ellipse_focal_length_l1788_178882

/-- For an ellipse with equation x²/4 + y²/9 = 1, the focal length is 2√5 -/
theorem ellipse_focal_length : 
  ∀ (x y : ℝ), x^2/4 + y^2/9 = 1 → 
  ∃ (f : ℝ), f = 2 * Real.sqrt 5 ∧ 
  (∃ (c : ℝ), c^2 = 5 ∧ f = 2*c) := by
  sorry

end NUMINAMATH_CALUDE_ellipse_focal_length_l1788_178882


namespace NUMINAMATH_CALUDE_equation_solutions_l1788_178877

theorem equation_solutions :
  (∀ x : ℝ, (2*x + 3)^2 = 16 ↔ x = 1/2 ∨ x = -7/2) ∧
  (∀ x : ℝ, x^2 - 4*x - 3 = 0 ↔ x = 2 + Real.sqrt 7 ∨ x = 2 - Real.sqrt 7) := by
  sorry

end NUMINAMATH_CALUDE_equation_solutions_l1788_178877


namespace NUMINAMATH_CALUDE_marble_ratio_l1788_178818

theorem marble_ratio (b y r : ℚ) 
  (h1 : b / y = 1.2) 
  (h2 : y / r = 5 / 6) 
  (h3 : b > 0) 
  (h4 : y > 0) 
  (h5 : r > 0) : 
  b / r = 1 := by
sorry

end NUMINAMATH_CALUDE_marble_ratio_l1788_178818


namespace NUMINAMATH_CALUDE_equation_solution_l1788_178885

theorem equation_solution :
  ∃ x : ℚ, ((15 - 2 + (4/x))/2 * 8 = 77) ∧ (x = 16/25) := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l1788_178885


namespace NUMINAMATH_CALUDE_min_value_of_expression_l1788_178833

theorem min_value_of_expression (x : ℝ) :
  ∃ (m : ℝ), m = -784 ∧ ∀ (y : ℝ), (15 - y) * (13 - y) * (15 + y) * (13 + y) ≥ m :=
by sorry

end NUMINAMATH_CALUDE_min_value_of_expression_l1788_178833


namespace NUMINAMATH_CALUDE_notebook_purchase_solution_l1788_178849

/-- Represents the number of notebooks bought at each price point -/
structure NotebookPurchase where
  two_dollar : ℕ
  five_dollar : ℕ
  six_dollar : ℕ

/-- Checks if the purchase satisfies the given conditions -/
def is_valid_purchase (p : NotebookPurchase) : Prop :=
  p.two_dollar ≥ 1 ∧ 
  p.five_dollar ≥ 1 ∧ 
  p.six_dollar ≥ 1 ∧
  p.two_dollar + p.five_dollar + p.six_dollar = 20 ∧
  2 * p.two_dollar + 5 * p.five_dollar + 6 * p.six_dollar = 62

theorem notebook_purchase_solution :
  ∃ (p : NotebookPurchase), is_valid_purchase p ∧ p.two_dollar = 14 :=
by sorry

end NUMINAMATH_CALUDE_notebook_purchase_solution_l1788_178849


namespace NUMINAMATH_CALUDE_division_problem_l1788_178829

theorem division_problem (total : ℚ) (a b c d : ℚ) : 
  total = 2880 →
  a = (1/3) * b →
  b = (2/5) * c →
  c = (3/4) * d →
  a + b + c + d = total →
  b = 403.2 := by
  sorry

end NUMINAMATH_CALUDE_division_problem_l1788_178829


namespace NUMINAMATH_CALUDE_triangle_side_length_l1788_178884

theorem triangle_side_length (A B C : ℝ × ℝ) : 
  let angle_BAC := Real.pi / 3  -- 60 degrees in radians
  let AB := 2
  let AC := 4
  let BC := ‖B - C‖  -- Euclidean distance between B and C
  (angle_BAC = Real.arccos ((B.1 - A.1) * (C.1 - A.1) + (B.2 - A.2) * (C.2 - A.2)) / (AB * AC)) →  -- angle condition
  (AB = ‖B - A‖) →  -- AB length condition
  (AC = ‖C - A‖) →  -- AC length condition
  BC = 2 * Real.sqrt 3 :=
by sorry

end NUMINAMATH_CALUDE_triangle_side_length_l1788_178884


namespace NUMINAMATH_CALUDE_arcsin_sqrt3_over_2_l1788_178898

theorem arcsin_sqrt3_over_2 : Real.arcsin (Real.sqrt 3 / 2) = π / 3 := by
  sorry

end NUMINAMATH_CALUDE_arcsin_sqrt3_over_2_l1788_178898


namespace NUMINAMATH_CALUDE_philip_initial_paintings_l1788_178802

/-- Represents the number of paintings Philip makes per day -/
def paintings_per_day : ℕ := 2

/-- Represents the number of days Philip will paint -/
def days : ℕ := 30

/-- Represents the total number of paintings Philip will have after 30 days -/
def total_paintings : ℕ := 80

/-- Calculates the initial number of paintings Philip had -/
def initial_paintings : ℕ := total_paintings - (paintings_per_day * days)

theorem philip_initial_paintings : initial_paintings = 20 := by
  sorry

end NUMINAMATH_CALUDE_philip_initial_paintings_l1788_178802


namespace NUMINAMATH_CALUDE_beef_weight_loss_percentage_l1788_178875

theorem beef_weight_loss_percentage (initial_weight : Real) (processed_weight : Real) 
  (h1 : initial_weight = 861.54)
  (h2 : processed_weight = 560) :
  let weight_loss := initial_weight - processed_weight
  let percentage_loss := (weight_loss / initial_weight) * 100
  ∃ ε > 0, abs (percentage_loss - 34.99) < ε :=
by sorry

end NUMINAMATH_CALUDE_beef_weight_loss_percentage_l1788_178875


namespace NUMINAMATH_CALUDE_total_airflow_theorem_l1788_178847

/-- Represents a fan with its airflow rate, operation time, and days of operation -/
structure Fan where
  airflow_rate : ℝ  -- Liters per second
  operation_time : ℝ  -- Minutes per day
  days_of_operation : ℕ

/-- Calculates the total airflow for a fan in one week -/
def fan_airflow (f : Fan) : ℝ :=
  f.airflow_rate * f.operation_time * 60 * f.days_of_operation

/-- The five fans in the room -/
def fan_A : Fan := ⟨10, 10, 7⟩
def fan_B : Fan := ⟨15, 20, 5⟩
def fan_C : Fan := ⟨25, 30, 5⟩
def fan_D : Fan := ⟨20, 15, 2⟩
def fan_E : Fan := ⟨30, 60, 6⟩

/-- Theorem: The total airflow generated by all five fans in one week is 1,041,000 liters -/
theorem total_airflow_theorem :
  fan_airflow fan_A + fan_airflow fan_B + fan_airflow fan_C +
  fan_airflow fan_D + fan_airflow fan_E = 1041000 := by
  sorry

end NUMINAMATH_CALUDE_total_airflow_theorem_l1788_178847


namespace NUMINAMATH_CALUDE_yoongi_calculation_l1788_178809

theorem yoongi_calculation (x : ℝ) : 5 * x = 30 → x - 7 = -1 := by
  sorry

end NUMINAMATH_CALUDE_yoongi_calculation_l1788_178809


namespace NUMINAMATH_CALUDE_integer_root_prime_coefficients_l1788_178807

/-- A polynomial of degree 4 with prime coefficients p and q that has an integer root -/
def has_integer_root (p q : ℕ) : Prop :=
  ∃ x : ℤ, x^4 - (p : ℤ) * x^3 + (q : ℤ) = 0

/-- The main theorem stating that if x^4 - px^3 + q = 0 has an integer root,
    and p and q are prime numbers, then p = 3 and q = 2 -/
theorem integer_root_prime_coefficients :
  ∀ p q : ℕ, Prime p → Prime q → has_integer_root p q → p = 3 ∧ q = 2 := by
  sorry

end NUMINAMATH_CALUDE_integer_root_prime_coefficients_l1788_178807
