import Mathlib

namespace NUMINAMATH_CALUDE_karen_picked_up_three_cases_l4014_401405

/-- The number of boxes of Tagalongs Karen sold -/
def boxes_sold : ℕ := 36

/-- The number of boxes in each case -/
def boxes_per_case : ℕ := 12

/-- The number of cases Karen picked up -/
def cases_picked_up : ℕ := boxes_sold / boxes_per_case

theorem karen_picked_up_three_cases : cases_picked_up = 3 := by
  sorry

end NUMINAMATH_CALUDE_karen_picked_up_three_cases_l4014_401405


namespace NUMINAMATH_CALUDE_student_council_committees_l4014_401479

theorem student_council_committees (x : ℕ) : 
  (x.choose 3 = 20) → (x.choose 4 = 15) := by sorry

end NUMINAMATH_CALUDE_student_council_committees_l4014_401479


namespace NUMINAMATH_CALUDE_X_is_element_of_Y_l4014_401428

def X : Set Nat := {0, 1}

def Y : Set (Set Nat) := {s | s ⊆ X}

theorem X_is_element_of_Y : X ∈ Y := by sorry

end NUMINAMATH_CALUDE_X_is_element_of_Y_l4014_401428


namespace NUMINAMATH_CALUDE_parabola_shift_theorem_l4014_401432

/-- Represents a parabola in the form y = ax² + bx + c -/
structure Parabola where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Shifts a parabola horizontally and vertically -/
def shift_parabola (p : Parabola) (h : ℝ) (k : ℝ) : Parabola :=
  { a := p.a,
    b := -2 * p.a * h + p.b,
    c := p.a * h^2 - p.b * h + p.c + k }

theorem parabola_shift_theorem (x y : ℝ) :
  let original := Parabola.mk 2 0 0
  let shifted := shift_parabola original 4 1
  y = 2*x^2 → y = shifted.a * (x + 4)^2 + 1 :=
by sorry

end NUMINAMATH_CALUDE_parabola_shift_theorem_l4014_401432


namespace NUMINAMATH_CALUDE_arcsin_arccos_equation_solution_l4014_401419

theorem arcsin_arccos_equation_solution :
  ∃ x : ℝ, x = -1 / Real.sqrt 7 ∧ Real.arcsin (3 * x) - Real.arccos (2 * x) = π / 6 :=
by sorry

end NUMINAMATH_CALUDE_arcsin_arccos_equation_solution_l4014_401419


namespace NUMINAMATH_CALUDE_books_read_per_year_l4014_401429

/-- The number of books read by the entire student body in one year -/
def total_books_read (c s : ℕ) : ℕ :=
  84 * c * s

/-- Theorem stating the total number of books read by the student body in one year -/
theorem books_read_per_year (c s : ℕ) :
  total_books_read c s = 84 * c * s :=
by
  sorry

#check books_read_per_year

end NUMINAMATH_CALUDE_books_read_per_year_l4014_401429


namespace NUMINAMATH_CALUDE_polynomial_factorization_l4014_401490

theorem polynomial_factorization (x : ℝ) : x^3 + 2*x^2 - 3*x = x*(x+3)*(x-1) := by
  sorry

end NUMINAMATH_CALUDE_polynomial_factorization_l4014_401490


namespace NUMINAMATH_CALUDE_right_triangle_ratio_l4014_401484

theorem right_triangle_ratio (a b c x y : ℝ) : 
  a > 0 → b > 0 → c > 0 → x > 0 → y > 0 →
  c^2 = a^2 + b^2 →  -- Pythagorean theorem
  x * y = a^2 →     -- Geometric mean theorem for x
  x * y = b^2 →     -- Geometric mean theorem for y
  x + y = c →       -- x and y form the hypotenuse
  a / b = 2 / 5 →   -- Given ratio
  x / y = 4 / 25 := by sorry

end NUMINAMATH_CALUDE_right_triangle_ratio_l4014_401484


namespace NUMINAMATH_CALUDE_first_terrific_tuesday_after_school_starts_l4014_401440

/-- Represents a date with a month and day. -/
structure Date where
  month : Nat
  day : Nat

/-- Represents a day of the week. -/
inductive DayOfWeek
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday
  | Saturday
  | Sunday

/-- Returns the number of days in a given month. -/
def daysInMonth (month : Nat) : Nat :=
  match month with
  | 10 => 31  -- October
  | 11 => 30  -- November
  | 12 => 31  -- December
  | _ => 30   -- Default for other months (not used in this problem)

/-- Returns the day of the week for a given date. -/
def dayOfWeek (date : Date) : DayOfWeek :=
  sorry  -- Implementation not needed for the statement

/-- Returns true if the given date is a Terrific Tuesday. -/
def isTerrificTuesday (date : Date) : Prop :=
  dayOfWeek date = DayOfWeek.Tuesday ∧
  (∀ d : Nat, d < date.day → dayOfWeek { month := date.month, day := d } = DayOfWeek.Tuesday →
    (∃ d' : Nat, d' < d ∧ dayOfWeek { month := date.month, day := d' } = DayOfWeek.Tuesday))

/-- The main theorem stating that December 31 is the first Terrific Tuesday after October 3. -/
theorem first_terrific_tuesday_after_school_starts :
  let schoolStart : Date := { month := 10, day := 3 }
  let firstTerrificTuesday : Date := { month := 12, day := 31 }
  dayOfWeek schoolStart = DayOfWeek.Tuesday →
  isTerrificTuesday firstTerrificTuesday ∧
  (∀ date : Date, schoolStart.month ≤ date.month ∧ date.month ≤ firstTerrificTuesday.month →
    (if date.month = schoolStart.month then schoolStart.day ≤ date.day else True) →
    (if date.month = firstTerrificTuesday.month then date.day ≤ firstTerrificTuesday.day else True) →
    date.day ≤ daysInMonth date.month →
    isTerrificTuesday date → date = firstTerrificTuesday) :=
by sorry

end NUMINAMATH_CALUDE_first_terrific_tuesday_after_school_starts_l4014_401440


namespace NUMINAMATH_CALUDE_andy_cake_profit_l4014_401413

/-- Calculates the profit per cake given the total ingredient cost for two cakes,
    the packaging cost per cake, and the selling price per cake. -/
def profit_per_cake (ingredient_cost_for_two : ℚ) (packaging_cost : ℚ) (selling_price : ℚ) : ℚ :=
  selling_price - (ingredient_cost_for_two / 2 + packaging_cost)

/-- Theorem stating that for Andy's cake business, given the specific costs and selling price,
    the profit per cake is $8. -/
theorem andy_cake_profit :
  profit_per_cake 12 1 15 = 8 := by
  sorry

end NUMINAMATH_CALUDE_andy_cake_profit_l4014_401413


namespace NUMINAMATH_CALUDE_tangent_circle_rectangle_area_l4014_401437

/-- A rectangle with a circle tangent to three sides and passing through the diagonal midpoint -/
structure TangentCircleRectangle where
  /-- The length of the rectangle -/
  w : ℝ
  /-- The height of the rectangle -/
  h : ℝ
  /-- The radius of the circle -/
  r : ℝ
  /-- The circle is tangent to three sides -/
  tangent_to_sides : h = r
  /-- The circle passes through the midpoint of the diagonal -/
  passes_through_midpoint : r^2 = (w/2)^2 + (h/2)^2

/-- The area of the rectangle is √3 * r^2 -/
theorem tangent_circle_rectangle_area (rect : TangentCircleRectangle) : 
  rect.w * rect.h = Real.sqrt 3 * rect.r^2 := by
  sorry

end NUMINAMATH_CALUDE_tangent_circle_rectangle_area_l4014_401437


namespace NUMINAMATH_CALUDE_sum_of_three_numbers_l4014_401463

theorem sum_of_three_numbers (a b c : ℝ) 
  (h1 : a^2 + b^2 + c^2 = 252) 
  (h2 : a*b + b*c + c*a = 116) : 
  a + b + c = 22 := by
sorry

end NUMINAMATH_CALUDE_sum_of_three_numbers_l4014_401463


namespace NUMINAMATH_CALUDE_deepak_age_l4014_401499

/-- Given that the ratio of Rahul's age to Deepak's age is 5:2,
    and Rahul will be 26 years old after 6 years,
    prove that Deepak's current age is 8 years. -/
theorem deepak_age (rahul_age deepak_age : ℕ) 
  (h_ratio : rahul_age * 2 = deepak_age * 5)
  (h_future : rahul_age + 6 = 26) : 
  deepak_age = 8 := by
  sorry

end NUMINAMATH_CALUDE_deepak_age_l4014_401499


namespace NUMINAMATH_CALUDE_unique_r_value_l4014_401438

/-- The polynomial function f(x) -/
def f (r : ℝ) (x : ℝ) : ℝ := 3 * x^4 + 2 * x^3 - x^2 - 5 * x + r

/-- Theorem stating that r = -5 is the unique value that satisfies f(-1) = 0 -/
theorem unique_r_value : ∃! r : ℝ, f r (-1) = 0 ∧ r = -5 := by sorry

end NUMINAMATH_CALUDE_unique_r_value_l4014_401438


namespace NUMINAMATH_CALUDE_segment_ratio_l4014_401485

/-- Represents a point on a line segment --/
structure Point :=
  (x : ℝ)

/-- Represents a line segment between two points --/
structure Segment (A B : Point) :=
  (length : ℝ)

/-- The main theorem --/
theorem segment_ratio 
  (A B C D : Point)
  (h1 : Segment A D)
  (h2 : Segment A B)
  (h3 : Segment B D)
  (h4 : Segment A C)
  (h5 : Segment C D)
  (h6 : Segment B C)
  (cond1 : B.x < D.x ∧ D.x < C.x)
  (cond2 : h2.length = 3 * h3.length)
  (cond3 : h4.length = 4 * h5.length)
  (cond4 : h1.length = h2.length + h3.length + h5.length) :
  h6.length / h1.length = 5 / 6 :=
sorry

end NUMINAMATH_CALUDE_segment_ratio_l4014_401485


namespace NUMINAMATH_CALUDE_min_value_theorem_l4014_401407

theorem min_value_theorem (a b : ℝ) (ha : 0 < a) (hb : 0 < b) (hab : a + b = 1) :
  (1 / a + 2 / b) ≥ 3 + 2 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_min_value_theorem_l4014_401407


namespace NUMINAMATH_CALUDE_two_cars_meeting_on_highway_l4014_401424

/-- Theorem: Two cars meeting on a highway --/
theorem two_cars_meeting_on_highway 
  (highway_length : ℝ) 
  (time : ℝ) 
  (speed_car2 : ℝ) 
  (h1 : highway_length = 105) 
  (h2 : time = 3) 
  (h3 : speed_car2 = 20) : 
  ∃ (speed_car1 : ℝ), 
    speed_car1 * time + speed_car2 * time = highway_length ∧ 
    speed_car1 = 15 := by
  sorry

end NUMINAMATH_CALUDE_two_cars_meeting_on_highway_l4014_401424


namespace NUMINAMATH_CALUDE_problem_solution_l4014_401436

theorem problem_solution : ∃ n : ℕ, 
  n = (2123 + 1787) * (6 * (2123 - 1787)) + 384 ∧ n = 7884144 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l4014_401436


namespace NUMINAMATH_CALUDE_triangle_formation_range_l4014_401449

theorem triangle_formation_range (x : ℝ) : 
  let circle := {p : ℝ × ℝ | p.1^2 + p.2^2 = 1}
  let A : ℝ × ℝ := (-1, 0)
  let B : ℝ × ℝ := (1, 0)
  let D : ℝ × ℝ := (x, 0)
  let C : ℝ × ℝ := (x, Real.sqrt (1 - x^2))
  let AD := Real.sqrt ((D.1 - A.1)^2 + (D.2 - A.2)^2)
  let BD := Real.sqrt ((D.1 - B.1)^2 + (D.2 - B.2)^2)
  let CD := Real.sqrt ((C.1 - D.1)^2 + (C.2 - D.2)^2)
  (AD + BD > CD ∧ AD + CD > BD ∧ BD + CD > AD) ↔ 
  (x > 2 - Real.sqrt 5 ∧ x < Real.sqrt 5 - 2) :=
by sorry

end NUMINAMATH_CALUDE_triangle_formation_range_l4014_401449


namespace NUMINAMATH_CALUDE_product_xyz_l4014_401433

theorem product_xyz (x y z : ℝ) 
  (hx : x > 0) (hy : y > 0) (hz : z > 0)
  (eq1 : x * (y + z) = 198)
  (eq2 : y * (z + x) = 216)
  (eq3 : z * (x + y) = 234) :
  x * y * z = 1080 := by
sorry

end NUMINAMATH_CALUDE_product_xyz_l4014_401433


namespace NUMINAMATH_CALUDE_no_integer_solutions_l4014_401492

theorem no_integer_solutions : ¬ ∃ (x : ℤ), x^2 - 9*x + 20 < 0 := by
  sorry

end NUMINAMATH_CALUDE_no_integer_solutions_l4014_401492


namespace NUMINAMATH_CALUDE_sin_seven_pi_sixths_l4014_401458

theorem sin_seven_pi_sixths : Real.sin (7 * π / 6) = -1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_sin_seven_pi_sixths_l4014_401458


namespace NUMINAMATH_CALUDE_solve_equation_l4014_401493

theorem solve_equation (x : ℝ) (h : Real.sqrt ((2 / x) + 2) = 4 / 3) : x = -9 := by
  sorry

end NUMINAMATH_CALUDE_solve_equation_l4014_401493


namespace NUMINAMATH_CALUDE_fraction_meaningful_l4014_401453

theorem fraction_meaningful (x : ℝ) : (1 : ℝ) / (x - 4) ≠ 0 ↔ x ≠ 4 := by
  sorry

end NUMINAMATH_CALUDE_fraction_meaningful_l4014_401453


namespace NUMINAMATH_CALUDE_brave_2022_first_appearance_l4014_401462

/-- The cycle length of the letters "BRAVE" -/
def letter_cycle_length : ℕ := 5

/-- The cycle length of the digits "2022" -/
def digit_cycle_length : ℕ := 4

/-- The line number where "BRAVE 2022" first appears -/
def first_appearance : ℕ := 20

theorem brave_2022_first_appearance :
  Nat.lcm letter_cycle_length digit_cycle_length = first_appearance :=
by sorry

end NUMINAMATH_CALUDE_brave_2022_first_appearance_l4014_401462


namespace NUMINAMATH_CALUDE_problem_solution_l4014_401457

theorem problem_solution : 
  3.2 * 2.25 - (5 * 0.85) / 2.5 = 5.5 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l4014_401457


namespace NUMINAMATH_CALUDE_first_term_of_specific_series_l4014_401466

/-- Given an infinite geometric series with sum S and sum of squares T,
    this function returns the first term of the series. -/
def first_term_of_geometric_series (S : ℝ) (T : ℝ) : ℝ := 
  sorry

/-- Theorem stating that for an infinite geometric series with sum 27 and 
    sum of squares 108, the first term is 216/31. -/
theorem first_term_of_specific_series : 
  first_term_of_geometric_series 27 108 = 216 / 31 := by
  sorry

end NUMINAMATH_CALUDE_first_term_of_specific_series_l4014_401466


namespace NUMINAMATH_CALUDE_hour_division_theorem_l4014_401474

/-- The number of seconds in an hour -/
def seconds_in_hour : ℕ := 3600

/-- The number of ways to divide an hour into periods -/
def num_divisions : ℕ := 44

/-- Theorem: The number of ordered pairs of positive integers (n, m) 
    such that n * m = 3600 is equal to 44 -/
theorem hour_division_theorem : 
  (Finset.filter (fun p : ℕ × ℕ => p.1 * p.2 = seconds_in_hour) 
    (Finset.product (Finset.range (seconds_in_hour + 1)) (Finset.range (seconds_in_hour + 1)))).card 
  = num_divisions := by
  sorry


end NUMINAMATH_CALUDE_hour_division_theorem_l4014_401474


namespace NUMINAMATH_CALUDE_perfect_square_factors_of_3456_l4014_401430

/-- Given that 3456 = 2^7 * 3^3, this function counts the number of its positive integer factors that are perfect squares. -/
def count_perfect_square_factors : ℕ :=
  let prime_factorization : List (ℕ × ℕ) := [(2, 7), (3, 3)]
  sorry

/-- The number of positive integer factors of 3456 that are perfect squares is 8. -/
theorem perfect_square_factors_of_3456 : count_perfect_square_factors = 8 := by
  sorry

end NUMINAMATH_CALUDE_perfect_square_factors_of_3456_l4014_401430


namespace NUMINAMATH_CALUDE_fibSeriesSum_l4014_401498

/-- Fibonacci sequence -/
def fib : ℕ → ℕ
  | 0 => 0
  | 1 => 1
  | (n + 2) => fib (n + 1) + fib n

/-- Sum of the infinite series of Fibonacci numbers divided by powers of 2 -/
noncomputable def fibSeries : ℝ := ∑' n, (fib n : ℝ) / 2^n

/-- The sum of the infinite series of Fibonacci numbers divided by powers of 2 equals 2 -/
theorem fibSeriesSum : fibSeries = 2 := by sorry

end NUMINAMATH_CALUDE_fibSeriesSum_l4014_401498


namespace NUMINAMATH_CALUDE_initial_plums_count_l4014_401455

/-- The number of plums Melanie picked initially -/
def initial_plums : ℕ := sorry

/-- The number of plums Melanie gave to Sam -/
def plums_given : ℕ := 3

/-- The number of plums Melanie has left -/
def plums_left : ℕ := 4

/-- Theorem stating that the initial number of plums equals 7 -/
theorem initial_plums_count : initial_plums = 7 := by sorry

end NUMINAMATH_CALUDE_initial_plums_count_l4014_401455


namespace NUMINAMATH_CALUDE_function_property_l4014_401447

-- Define the functions
def f₁ (x : ℝ) : ℝ := |2 * x|
def f₂ (x : ℝ) : ℝ := x
noncomputable def f₃ (x : ℝ) : ℝ := Real.sqrt x
def f₄ (x : ℝ) : ℝ := x - |x|

-- State the theorem
theorem function_property :
  (∀ x, f₁ (2 * x) = 2 * f₁ x) ∧
  (∀ x, f₂ (2 * x) = 2 * f₂ x) ∧
  (∃ x, f₃ (2 * x) ≠ 2 * f₃ x) ∧
  (∀ x, f₄ (2 * x) = 2 * f₄ x) :=
by
  sorry

end NUMINAMATH_CALUDE_function_property_l4014_401447


namespace NUMINAMATH_CALUDE_sam_puppies_l4014_401442

theorem sam_puppies (initial : ℝ) (given_away : ℝ) (h1 : initial = 6.0) (h2 : given_away = 2.0) :
  initial - given_away = 4.0 := by sorry

end NUMINAMATH_CALUDE_sam_puppies_l4014_401442


namespace NUMINAMATH_CALUDE_smallest_prime_perimeter_triangle_l4014_401480

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m > 1 → m < n → ¬(n % m = 0)

def triangle_inequality (a b c : ℕ) : Prop := a + b > c ∧ b + c > a ∧ c + a > b

def is_scalene (a b c : ℕ) : Prop := a ≠ b ∧ b ≠ c ∧ c ≠ a

theorem smallest_prime_perimeter_triangle :
  ∀ a b c : ℕ,
    a = 5 →
    is_prime a →
    is_prime b →
    is_prime c →
    is_prime (a + b + c) →
    triangle_inequality a b c →
    is_scalene a b c →
    ∀ p q r : ℕ,
      p = 5 →
      is_prime p →
      is_prime q →
      is_prime r →
      is_prime (p + q + r) →
      triangle_inequality p q r →
      is_scalene p q r →
      a + b + c ≤ p + q + r →
    a + b + c = 23 :=
sorry

end NUMINAMATH_CALUDE_smallest_prime_perimeter_triangle_l4014_401480


namespace NUMINAMATH_CALUDE_blueberries_count_l4014_401403

/-- The number of strawberries in each red box -/
def strawberries_per_red_box : ℕ := 100

/-- The difference between strawberries in a red box and blueberries in a blue box -/
def berry_difference : ℕ := 30

/-- The number of blueberries in each blue box -/
def blueberries_per_blue_box : ℕ := strawberries_per_red_box - berry_difference

theorem blueberries_count : blueberries_per_blue_box = 70 := by
  sorry

end NUMINAMATH_CALUDE_blueberries_count_l4014_401403


namespace NUMINAMATH_CALUDE_collatz_7_11_collatz_10_probability_l4014_401421

-- Define the Collatz operation
def collatz (n : ℕ) : ℕ :=
  if n % 2 = 0 then n / 2 else 3 * n + 1

-- Define the Collatz sequence
def collatz_seq (a₀ : ℕ) : ℕ → ℕ
  | 0 => a₀
  | n + 1 => collatz (collatz_seq a₀ n)

-- Statement 1: When a₀ = 7, a₁₁ = 5
theorem collatz_7_11 : collatz_seq 7 11 = 5 := by sorry

-- Helper function to check if a number is odd
def is_odd (n : ℕ) : Bool := n % 2 ≠ 0

-- Statement 2: When a₀ = 10, the probability of randomly selecting two numbers
-- from aᵢ (i = 1,2,3,4,5,6), at least one of which is odd, is 3/5
theorem collatz_10_probability :
  let seq := List.range 6 |> List.map (collatz_seq 10)
  let total_pairs := seq.length.choose 2
  let odd_pairs := (seq.filterMap (fun n => if is_odd n then some n else none)).length
  (total_pairs - (seq.length - odd_pairs).choose 2) / total_pairs = 3 / 5 := by sorry

end NUMINAMATH_CALUDE_collatz_7_11_collatz_10_probability_l4014_401421


namespace NUMINAMATH_CALUDE_problem_solution_l4014_401456

theorem problem_solution (x y : ℝ) (hx : x = 2 - Real.sqrt 3) (hy : y = 2 + Real.sqrt 3) :
  (x^2 - y^2 = -8 * Real.sqrt 3) ∧ (x^2 + x*y + y^2 = 15) := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l4014_401456


namespace NUMINAMATH_CALUDE_tangent_circles_existence_l4014_401408

-- Define the necessary geometric objects
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

structure Line where
  point1 : ℝ × ℝ
  point2 : ℝ × ℝ

-- Define the tangency relations
def isTangentToCircle (c1 c2 : Circle) : Prop :=
  sorry

def isTangentToLine (c : Circle) (l : Line) : Prop :=
  sorry

def isOnLine (p : ℝ × ℝ) (l : Line) : Prop :=
  sorry

-- Theorem statement
theorem tangent_circles_existence
  (C : Circle) (l : Line) (M : ℝ × ℝ) 
  (h : isOnLine M l) :
  ∃ (C' C'' : Circle),
    (isTangentToCircle C' C ∧ isTangentToLine C' l ∧ isOnLine M l) ∧
    (isTangentToCircle C'' C ∧ isTangentToLine C'' l ∧ isOnLine M l) ∧
    (C' ≠ C'') :=
by sorry

end NUMINAMATH_CALUDE_tangent_circles_existence_l4014_401408


namespace NUMINAMATH_CALUDE_marble_bag_count_l4014_401448

theorem marble_bag_count :
  ∀ (total white : ℕ),
  (6 : ℝ) + 9 + white = total →
  (9 + white : ℝ) / total = 0.7 →
  total = 20 :=
by
  sorry

end NUMINAMATH_CALUDE_marble_bag_count_l4014_401448


namespace NUMINAMATH_CALUDE_smallest_M_inequality_l4014_401471

theorem smallest_M_inequality (a b c : ℝ) :
  let M : ℝ := (9 * Real.sqrt 2) / 32
  ∀ N : ℝ, (∀ x y z : ℝ, |x*y*(x^2 - y^2) + y*z*(y^2 - z^2) + z*x*(z^2 - x^2)| ≤ N*(x^2 + y^2 + z^2)^2) →
    M ≤ N ∧ |a*b*(a^2 - b^2) + b*c*(b^2 - c^2) + c*a*(c^2 - a^2)| ≤ M*(a^2 + b^2 + c^2)^2 :=
by sorry

end NUMINAMATH_CALUDE_smallest_M_inequality_l4014_401471


namespace NUMINAMATH_CALUDE_max_playtime_is_180_minutes_l4014_401420

/-- Represents an arcade bundle with tokens, playtime in hours, and cost --/
structure Bundle where
  tokens : ℕ
  playtime : ℕ
  cost : ℕ

/-- Mike's weekly pay in dollars --/
def weekly_pay : ℕ := 100

/-- Mike's arcade budget in dollars (half of weekly pay) --/
def arcade_budget : ℕ := weekly_pay / 2

/-- Cost of snacks in dollars --/
def snack_cost : ℕ := 5

/-- Available bundles at the arcade --/
def bundles : List Bundle := [
  ⟨50, 1, 25⟩,   -- Bundle A
  ⟨120, 3, 45⟩,  -- Bundle B
  ⟨200, 5, 60⟩   -- Bundle C
]

/-- Remaining budget after buying snacks --/
def remaining_budget : ℕ := arcade_budget - snack_cost

/-- Function to calculate total playtime in minutes for a given bundle and quantity --/
def total_playtime (bundle : Bundle) (quantity : ℕ) : ℕ :=
  bundle.playtime * quantity * 60

/-- Theorem: The maximum playtime Mike can achieve is 180 minutes --/
theorem max_playtime_is_180_minutes :
  ∃ (bundle : Bundle) (quantity : ℕ),
    bundle ∈ bundles ∧
    bundle.cost * quantity ≤ remaining_budget ∧
    total_playtime bundle quantity = 180 ∧
    ∀ (other_bundle : Bundle) (other_quantity : ℕ),
      other_bundle ∈ bundles →
      other_bundle.cost * other_quantity ≤ remaining_budget →
      total_playtime other_bundle other_quantity ≤ 180 :=
sorry

end NUMINAMATH_CALUDE_max_playtime_is_180_minutes_l4014_401420


namespace NUMINAMATH_CALUDE_finite_intersection_l4014_401465

def sequence_a : ℕ → ℕ → ℕ
  | a₁, 0 => a₁
  | a₁, n + 1 => n * sequence_a a₁ n + 1

def sequence_b : ℕ → ℕ → ℕ
  | b₁, 0 => b₁
  | b₁, n + 1 => n * sequence_b b₁ n - 1

theorem finite_intersection (a₁ b₁ : ℕ) :
  ∃ (N : ℕ), ∀ (n : ℕ), n ≥ N → sequence_a a₁ n ≠ sequence_b b₁ n :=
sorry

end NUMINAMATH_CALUDE_finite_intersection_l4014_401465


namespace NUMINAMATH_CALUDE_divide_multiply_problem_l4014_401491

theorem divide_multiply_problem : (2.25 / 3) * 12 = 9 := by
  sorry

end NUMINAMATH_CALUDE_divide_multiply_problem_l4014_401491


namespace NUMINAMATH_CALUDE_increasing_cubic_function_parameter_range_l4014_401473

theorem increasing_cubic_function_parameter_range 
  (f : ℝ → ℝ) (a : ℝ) (h : ∀ x ∈ Set.Ioo (-1) 1, StrictMono f) : a ≥ 3 := by
  sorry

end NUMINAMATH_CALUDE_increasing_cubic_function_parameter_range_l4014_401473


namespace NUMINAMATH_CALUDE_petya_running_time_l4014_401470

theorem petya_running_time (V D : ℝ) (hV : V > 0) (hD : D > 0) : 
  let T := D / V
  let V1 := 1.25 * V
  let V2 := 0.8 * V
  let T1 := D / (2 * V1)
  let T2 := D / (2 * V2)
  let Tactual := T1 + T2
  Tactual > T :=
by sorry

end NUMINAMATH_CALUDE_petya_running_time_l4014_401470


namespace NUMINAMATH_CALUDE_real_part_of_z_l4014_401434

theorem real_part_of_z (z : ℂ) (h : Complex.I * (z + 1) = -3 + 2 * Complex.I) : 
  z.re = 1 := by
  sorry

end NUMINAMATH_CALUDE_real_part_of_z_l4014_401434


namespace NUMINAMATH_CALUDE_largest_even_digit_multiple_of_9_under_1000_l4014_401427

def has_only_even_digits (n : ℕ) : Prop :=
  ∀ d, d ∈ n.digits 10 → d % 2 = 0

def sum_of_digits (n : ℕ) : ℕ :=
  (n.digits 10).sum

theorem largest_even_digit_multiple_of_9_under_1000 :
  ∃ (n : ℕ), n = 360 ∧
    n < 1000 ∧
    has_only_even_digits n ∧
    n % 9 = 0 ∧
    ∀ (m : ℕ), m < 1000 → has_only_even_digits m → m % 9 = 0 → m ≤ n :=
by sorry

end NUMINAMATH_CALUDE_largest_even_digit_multiple_of_9_under_1000_l4014_401427


namespace NUMINAMATH_CALUDE_min_value_expression_l4014_401494

theorem min_value_expression (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) :
  (x + y) / z + (x + z) / y + (y + z) / x + 3 ≥ 9 ∧
  ((x + y) / z + (x + z) / y + (y + z) / x + 3 = 9 ↔ x = y ∧ y = z) :=
by sorry

end NUMINAMATH_CALUDE_min_value_expression_l4014_401494


namespace NUMINAMATH_CALUDE_complex_real_condition_l4014_401496

theorem complex_real_condition (m : ℝ) :
  let z : ℂ := m^2 * (1 + Complex.I) - m * (m + Complex.I)
  (z.im = 0) ↔ (m = 0 ∨ m = 1) := by
  sorry

end NUMINAMATH_CALUDE_complex_real_condition_l4014_401496


namespace NUMINAMATH_CALUDE_rhombus_diagonal_length_l4014_401451

/-- Proves that in a rhombus with one diagonal of 62 meters and an area of 2480 square meters,
    the length of the other diagonal is 80 meters. -/
theorem rhombus_diagonal_length (d1 : ℝ) (area : ℝ) (d2 : ℝ) 
    (h1 : d1 = 62) 
    (h2 : area = 2480) 
    (h3 : area = (d1 * d2) / 2) : d2 = 80 := by
  sorry

end NUMINAMATH_CALUDE_rhombus_diagonal_length_l4014_401451


namespace NUMINAMATH_CALUDE_remainder_of_m_l4014_401402

theorem remainder_of_m (m : ℕ) (h1 : m^3 % 7 = 6) (h2 : m^4 % 7 = 4) : m % 7 = 3 := by
  sorry

end NUMINAMATH_CALUDE_remainder_of_m_l4014_401402


namespace NUMINAMATH_CALUDE_exists_x_squared_minus_one_nonnegative_l4014_401488

theorem exists_x_squared_minus_one_nonnegative :
  ∃ x : ℝ, x ∈ Set.Icc 0 1 ∧ x^2 - 1 ≥ 0 := by
  sorry

end NUMINAMATH_CALUDE_exists_x_squared_minus_one_nonnegative_l4014_401488


namespace NUMINAMATH_CALUDE_ball_max_height_l4014_401431

/-- The height function of the ball's trajectory -/
def f (t : ℝ) : ℝ := -10 * t^2 + 50 * t - 24

/-- The maximum height reached by the ball -/
def max_height : ℝ := 38.5

theorem ball_max_height :
  IsGreatest { y | ∃ t, f t = y } max_height := by
  sorry

end NUMINAMATH_CALUDE_ball_max_height_l4014_401431


namespace NUMINAMATH_CALUDE_smallest_value_complex_expression_l4014_401472

theorem smallest_value_complex_expression (a b c : ℤ) (ω : ℂ) 
  (h_distinct : a ≠ b ∧ b ≠ c ∧ a ≠ c)
  (h_omega_fourth : ω^4 = 1)
  (h_omega_not_one : ω ≠ 1) :
  ∃ (min_val : ℝ), 
    (∀ (x y z : ℤ) (h_xyz_distinct : x ≠ y ∧ y ≠ z ∧ x ≠ z), 
      Complex.abs (x + y * ω + z * ω^3) ≥ min_val) ∧
    (∃ (p q r : ℤ) (h_pqr_distinct : p ≠ q ∧ q ≠ r ∧ p ≠ r), 
      Complex.abs (p + q * ω + r * ω^3) = min_val) ∧
    min_val = Real.sqrt 3 :=
by sorry

end NUMINAMATH_CALUDE_smallest_value_complex_expression_l4014_401472


namespace NUMINAMATH_CALUDE_calculate_expression_l4014_401467

theorem calculate_expression : (3752 / (39 * 2) + 5030 / (39 * 10) : ℚ) = 61 := by
  sorry

end NUMINAMATH_CALUDE_calculate_expression_l4014_401467


namespace NUMINAMATH_CALUDE_consecutive_integers_product_120_l4014_401441

theorem consecutive_integers_product_120 :
  ∃ (a b c d e : ℤ),
    b = a + 1 ∧
    d = c + 1 ∧
    e = c + 2 ∧
    a * b = 120 ∧
    c * d * e = 120 ∧
    a + b + c + d + e = 37 :=
by sorry

end NUMINAMATH_CALUDE_consecutive_integers_product_120_l4014_401441


namespace NUMINAMATH_CALUDE_strawberry_picking_total_weight_l4014_401422

theorem strawberry_picking_total_weight 
  (marco_weight : ℕ) 
  (dad_weight : ℕ) 
  (h1 : marco_weight = 8) 
  (h2 : dad_weight = 32) : 
  marco_weight + dad_weight = 40 := by
sorry

end NUMINAMATH_CALUDE_strawberry_picking_total_weight_l4014_401422


namespace NUMINAMATH_CALUDE_original_number_proof_l4014_401461

theorem original_number_proof : ∃ N : ℕ, N > 0 ∧ N - 28 ≡ 0 [MOD 87] ∧ ∀ M : ℕ, M > 0 ∧ M - 28 ≡ 0 [MOD 87] → M ≥ N :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_original_number_proof_l4014_401461


namespace NUMINAMATH_CALUDE_complement_intersection_theorem_l4014_401477

-- Define the universal set U
def U : Set ℝ := {x | x ≤ 5}

-- Define set A
def A : Set ℝ := {x | -3 < x ∧ x < 4}

-- Define set B
def B : Set ℝ := {x | -5 ≤ x ∧ x ≤ 3}

-- Define the result set
def result : Set ℝ := {x | -5 ≤ x ∧ x ≤ -3}

-- Theorem statement
theorem complement_intersection_theorem :
  (Set.compl A ∩ B) = result := by sorry

end NUMINAMATH_CALUDE_complement_intersection_theorem_l4014_401477


namespace NUMINAMATH_CALUDE_completing_square_transformation_l4014_401486

theorem completing_square_transformation (x : ℝ) :
  (x^2 - 2*x - 5 = 0) ↔ ((x - 1)^2 = 6) :=
sorry

end NUMINAMATH_CALUDE_completing_square_transformation_l4014_401486


namespace NUMINAMATH_CALUDE_janet_tulips_l4014_401417

/-- The number of tulips Janet picked -/
def T : ℕ := sorry

/-- The total number of flowers Janet picked -/
def total_flowers : ℕ := T + 11

/-- The number of flowers Janet used -/
def used_flowers : ℕ := 11

/-- The number of extra flowers Janet had -/
def extra_flowers : ℕ := 4

theorem janet_tulips : T = 4 := by sorry

end NUMINAMATH_CALUDE_janet_tulips_l4014_401417


namespace NUMINAMATH_CALUDE_range_of_a_in_acute_triangle_l4014_401414

/-- Given an acute triangle ABC with sides a, b, c opposite to angles A, B, C respectively,
    if b^2 - a^2 = ac and c = 2, then 2/3 < a < 2 -/
theorem range_of_a_in_acute_triangle (a b c : ℝ) (A B C : ℝ) :
  0 < A ∧ A < π / 2 →
  0 < B ∧ B < π / 2 →
  0 < C ∧ C < π / 2 →
  a > 0 ∧ b > 0 ∧ c > 0 →
  b^2 - a^2 = a * c →
  c = 2 →
  2/3 < a ∧ a < 2 := by sorry

end NUMINAMATH_CALUDE_range_of_a_in_acute_triangle_l4014_401414


namespace NUMINAMATH_CALUDE_right_triangle_shortest_leg_l4014_401410

theorem right_triangle_shortest_leg : ∃ (a b : ℕ),
  a < b ∧ a^2 + b^2 = 65^2 ∧ ∀ (x y : ℕ), x < y ∧ x^2 + y^2 = 65^2 → a ≤ x :=
by sorry

end NUMINAMATH_CALUDE_right_triangle_shortest_leg_l4014_401410


namespace NUMINAMATH_CALUDE_barking_ratio_is_one_fourth_l4014_401401

/-- Represents the state of dogs in a park -/
structure DogPark where
  total : ℕ
  running : ℕ
  playing : ℕ
  idle : ℕ

/-- The ratio of barking dogs to total dogs -/
def barkingRatio (park : DogPark) : Rat :=
  let barking := park.total - (park.running + park.playing + park.idle)
  barking / park.total

/-- Theorem stating the barking ratio in the given scenario -/
theorem barking_ratio_is_one_fourth :
  ∃ (park : DogPark),
    park.total = 88 ∧
    park.running = 12 ∧
    park.playing = 44 ∧
    park.idle = 10 ∧
    barkingRatio park = 1 / 4 := by
  sorry


end NUMINAMATH_CALUDE_barking_ratio_is_one_fourth_l4014_401401


namespace NUMINAMATH_CALUDE_injective_function_equality_l4014_401459

theorem injective_function_equality (f : ℕ → ℕ) (h_inj : Function.Injective f) 
  (h_cond : ∀ n : ℕ, f (f n) ≤ (f n + n) / 2) : 
  ∀ n : ℕ, f n = n := by
  sorry

end NUMINAMATH_CALUDE_injective_function_equality_l4014_401459


namespace NUMINAMATH_CALUDE_result_calculation_l4014_401487

theorem result_calculation (h1 : 7125 / 1.25 = 5700) (h2 : x = 3) : 
  (712.5 / 12.5) ^ x = 185193 := by
sorry

end NUMINAMATH_CALUDE_result_calculation_l4014_401487


namespace NUMINAMATH_CALUDE_triangle_with_angle_ratio_2_3_5_is_right_triangle_l4014_401483

theorem triangle_with_angle_ratio_2_3_5_is_right_triangle (a b c : ℝ) 
  (h_triangle : a + b + c = 180)
  (h_ratio : ∃ (x : ℝ), a = 2*x ∧ b = 3*x ∧ c = 5*x) :
  c = 90 := by
sorry

end NUMINAMATH_CALUDE_triangle_with_angle_ratio_2_3_5_is_right_triangle_l4014_401483


namespace NUMINAMATH_CALUDE_percent_exceeding_speed_limit_l4014_401452

theorem percent_exceeding_speed_limit 
  (total_motorists : ℕ) 
  (h_total_positive : total_motorists > 0)
  (percent_ticketed : ℝ) 
  (h_percent_ticketed : percent_ticketed = 10)
  (percent_unticketed_speeders : ℝ) 
  (h_percent_unticketed : percent_unticketed_speeders = 50) : 
  (percent_ticketed * total_motorists / 100 + 
   percent_ticketed * total_motorists / 100) / total_motorists * 100 = 20 := by
  sorry

#check percent_exceeding_speed_limit

end NUMINAMATH_CALUDE_percent_exceeding_speed_limit_l4014_401452


namespace NUMINAMATH_CALUDE_play_admission_receipts_l4014_401415

/-- Calculates the total admission receipts for a play -/
def totalAdmissionReceipts (totalPeople : ℕ) (adultPrice childPrice : ℕ) (children : ℕ) : ℕ :=
  let adults := totalPeople - children
  adults * adultPrice + children * childPrice

/-- Theorem: The total admission receipts for the play is $960 -/
theorem play_admission_receipts :
  totalAdmissionReceipts 610 2 1 260 = 960 := by
  sorry

end NUMINAMATH_CALUDE_play_admission_receipts_l4014_401415


namespace NUMINAMATH_CALUDE_frood_game_theorem_l4014_401411

/-- Score for dropping n froods -/
def drop_score (n : ℕ) : ℕ := n * (n + 1)

/-- Score for eating n froods -/
def eat_score (n : ℕ) : ℕ := 8 * n

/-- The least number of froods for which dropping them earns more points than eating them -/
def least_frood_number : ℕ := 8

theorem frood_game_theorem :
  least_frood_number = 8 ∧
  ∀ n : ℕ, n < least_frood_number → drop_score n ≤ eat_score n ∧
  drop_score least_frood_number > eat_score least_frood_number :=
by sorry

end NUMINAMATH_CALUDE_frood_game_theorem_l4014_401411


namespace NUMINAMATH_CALUDE_complex_number_magnitude_squared_l4014_401409

theorem complex_number_magnitude_squared (z : ℂ) (h : z^2 + Complex.abs z ^ 2 = 8 - 2*I) : 
  Complex.abs z ^ 2 = 17/4 := by sorry

end NUMINAMATH_CALUDE_complex_number_magnitude_squared_l4014_401409


namespace NUMINAMATH_CALUDE_partial_fraction_A_value_l4014_401439

-- Define the polynomial
def p (x : ℝ) : ℝ := x^4 - 20*x^3 + 147*x^2 - 490*x + 588

-- Define the partial fraction decomposition
def partial_fraction (A B C D x : ℝ) : Prop :=
  1 / p x = A / (x + 3) + B / (x - 4) + C / (x - 4)^2 + D / (x - 7)

-- Theorem statement
theorem partial_fraction_A_value :
  ∀ A B C D : ℝ, (∀ x : ℝ, partial_fraction A B C D x) → A = -1/490 :=
by
  sorry

end NUMINAMATH_CALUDE_partial_fraction_A_value_l4014_401439


namespace NUMINAMATH_CALUDE_dartboard_angle_l4014_401495

theorem dartboard_angle (p : ℝ) (θ : ℝ) : 
  0 < p → p < 1 → 0 ≤ θ → θ ≤ 360 →
  (p = θ / 360) → (p = 1 / 8) → θ = 45 := by
  sorry

end NUMINAMATH_CALUDE_dartboard_angle_l4014_401495


namespace NUMINAMATH_CALUDE_six_students_three_colleges_l4014_401482

/-- The number of ways to distribute n indistinguishable objects into k distinguishable bins,
    where each bin must contain at least one object. -/
def distributeWithMinimum (n k : ℕ) : ℕ :=
  k^n - k * (k-1)^n + (k.choose 2) * (k-2)^n

/-- The specific case for 6 students and 3 colleges -/
theorem six_students_three_colleges :
  distributeWithMinimum 6 3 = 540 := by
  sorry

end NUMINAMATH_CALUDE_six_students_three_colleges_l4014_401482


namespace NUMINAMATH_CALUDE_add_9999_seconds_to_8am_l4014_401416

/-- Represents time in hours, minutes, and seconds -/
structure Time where
  hours : Nat
  minutes : Nat
  seconds : Nat
  deriving Repr

/-- Adds seconds to a given time -/
def addSeconds (t : Time) (s : Nat) : Time :=
  sorry

/-- Converts a natural number to a Time structure -/
def natToTime (n : Nat) : Time :=
  sorry

theorem add_9999_seconds_to_8am (startTime endTime : Time) :
  startTime = { hours := 8, minutes := 0, seconds := 0 } →
  endTime = addSeconds startTime 9999 →
  endTime = { hours := 10, minutes := 46, seconds := 39 } :=
sorry

end NUMINAMATH_CALUDE_add_9999_seconds_to_8am_l4014_401416


namespace NUMINAMATH_CALUDE_system_solution_l4014_401412

theorem system_solution :
  ∃ (x y z : ℚ),
    (4 * x - 6 * y + 2 * z = -14) ∧
    (8 * x + 3 * y - z = -15) ∧
    (3 * x + z = 7) ∧
    (x = 100 / 33) ∧
    (y = 146 / 33) ∧
    (z = 29 / 11) :=
by sorry

end NUMINAMATH_CALUDE_system_solution_l4014_401412


namespace NUMINAMATH_CALUDE_complex_fourth_root_of_negative_sixteen_l4014_401454

theorem complex_fourth_root_of_negative_sixteen :
  let solutions := {z : ℂ | z^4 = -16 ∧ z.im ≥ 0}
  solutions = {Complex.mk (Real.sqrt 2) (Real.sqrt 2), Complex.mk (-Real.sqrt 2) (-Real.sqrt 2)} := by
  sorry

end NUMINAMATH_CALUDE_complex_fourth_root_of_negative_sixteen_l4014_401454


namespace NUMINAMATH_CALUDE_tangent_line_equation_l4014_401425

/-- Represents an ellipse with semi-major axis a and semi-minor axis b -/
structure Ellipse where
  a : ℝ
  b : ℝ
  h_pos : 0 < b ∧ b < a

/-- A point on an ellipse -/
structure PointOnEllipse (e : Ellipse) where
  x : ℝ
  y : ℝ
  h_on_ellipse : x^2 / e.a^2 + y^2 / e.b^2 = 1

/-- The equation of a line -/
structure Line where
  A : ℝ
  B : ℝ
  C : ℝ
  h_not_zero : A ≠ 0 ∨ B ≠ 0

/-- Theorem: The equation of the tangent line at a point on an ellipse -/
theorem tangent_line_equation (e : Ellipse) (p : PointOnEllipse e) :
  ∃ (l : Line), l.A = p.x / e.a^2 ∧ l.B = p.y / e.b^2 ∧ l.C = -1 ∧
  (∀ (x y : ℝ), x^2 / e.a^2 + y^2 / e.b^2 = 1 → l.A * x + l.B * y + l.C = 0 → x = p.x ∧ y = p.y) :=
sorry

end NUMINAMATH_CALUDE_tangent_line_equation_l4014_401425


namespace NUMINAMATH_CALUDE_intersection_line_slope_l4014_401423

/-- The slope of the line passing through the intersection points of two circles -/
theorem intersection_line_slope (x y : ℝ) : 
  (x^2 + y^2 + 6*x - 8*y - 40 = 0) ∧ 
  (x^2 + y^2 + 22*x - 2*y + 20 = 0) →
  (∃ m : ℝ, m = 8/3 ∧ 
    ∀ (x₁ y₁ x₂ y₂ : ℝ), 
      (x₁^2 + y₁^2 + 6*x₁ - 8*y₁ - 40 = 0) ∧ 
      (x₁^2 + y₁^2 + 22*x₁ - 2*y₁ + 20 = 0) ∧
      (x₂^2 + y₂^2 + 6*x₂ - 8*y₂ - 40 = 0) ∧ 
      (x₂^2 + y₂^2 + 22*x₂ - 2*y₂ + 20 = 0) ∧
      (x₁ ≠ x₂) →
      m = (y₂ - y₁) / (x₂ - x₁)) :=
by sorry

end NUMINAMATH_CALUDE_intersection_line_slope_l4014_401423


namespace NUMINAMATH_CALUDE_prob_different_fruits_l4014_401481

/-- The number of fruit types available --/
def num_fruits : ℕ := 5

/-- The number of meals over two days --/
def num_meals : ℕ := 6

/-- The probability of choosing a specific fruit for all meals --/
def prob_same_fruit : ℚ := (1 / num_fruits) ^ num_meals

/-- The probability of eating at least two different kinds of fruit over two days --/
theorem prob_different_fruits : 
  1 - num_fruits * prob_same_fruit = 15620 / 15625 := by
  sorry

end NUMINAMATH_CALUDE_prob_different_fruits_l4014_401481


namespace NUMINAMATH_CALUDE_inequality_proof_l4014_401475

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  2 * a^2 / (b + c) + 2 * b^2 / (c + a) + 2 * c^2 / (a + b) ≥ 
  a + b + c + (2 * a - b - c)^2 / (a + b + c) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l4014_401475


namespace NUMINAMATH_CALUDE_sneakers_final_price_l4014_401489

/-- Calculates the final price of sneakers after applying discounts and sales tax -/
def finalPrice (originalPrice couponDiscount promoDiscountRate membershipDiscountRate salesTaxRate : ℚ) : ℚ :=
  let priceAfterCoupon := originalPrice - couponDiscount
  let priceAfterPromo := priceAfterCoupon * (1 - promoDiscountRate)
  let priceAfterMembership := priceAfterPromo * (1 - membershipDiscountRate)
  let finalPriceBeforeTax := priceAfterMembership
  finalPriceBeforeTax * (1 + salesTaxRate)

/-- Theorem stating that the final price of the sneakers is $100.63 -/
theorem sneakers_final_price :
  finalPrice 120 10 (5/100) (10/100) (7/100) = 10063/100 := by
  sorry

end NUMINAMATH_CALUDE_sneakers_final_price_l4014_401489


namespace NUMINAMATH_CALUDE_ellipse_foci_distance_l4014_401478

/-- Given an ellipse with equation 9x^2 + 16y^2 = 144, the distance between its foci is 2√7 -/
theorem ellipse_foci_distance (x y : ℝ) :
  (9 * x^2 + 16 * y^2 = 144) → (∃ f₁ f₂ : ℝ × ℝ, 
    f₁ ≠ f₂ ∧ 
    (∀ p : ℝ × ℝ, 9 * p.1^2 + 16 * p.2^2 = 144 → 
      Real.sqrt ((p.1 - f₁.1)^2 + (p.2 - f₁.2)^2) + 
      Real.sqrt ((p.1 - f₂.1)^2 + (p.2 - f₂.2)^2) = 8) ∧
    Real.sqrt ((f₁.1 - f₂.1)^2 + (f₁.2 - f₂.2)^2) = 2 * Real.sqrt 7) :=
by sorry

end NUMINAMATH_CALUDE_ellipse_foci_distance_l4014_401478


namespace NUMINAMATH_CALUDE_ratio_fraction_equality_l4014_401460

theorem ratio_fraction_equality (A B C : ℚ) (h : A / B = 3 / 2 ∧ B / C = 2 / 5) :
  (4 * A + 3 * B) / (5 * C - 2 * A) = 18 / 19 := by
  sorry

end NUMINAMATH_CALUDE_ratio_fraction_equality_l4014_401460


namespace NUMINAMATH_CALUDE_segment_ratios_l4014_401443

/-- Given line segments AC, AB, and BC, where AB consists of 3 parts and BC consists of 4 parts,
    prove the ratios of AB:AC and BC:AC. -/
theorem segment_ratios (AC AB BC : ℝ) (h1 : AB = 3) (h2 : BC = 4) (h3 : AC = AB + BC) :
  (AB / AC = 3 / 7) ∧ (BC / AC = 4 / 7) := by
  sorry

end NUMINAMATH_CALUDE_segment_ratios_l4014_401443


namespace NUMINAMATH_CALUDE_tom_hockey_games_l4014_401450

/-- The number of hockey games Tom attended this year -/
def games_this_year : ℕ := 4

/-- The number of hockey games Tom attended last year -/
def games_last_year : ℕ := 9

/-- The total number of hockey games Tom attended -/
def total_games : ℕ := games_this_year + games_last_year

theorem tom_hockey_games :
  total_games = 13 := by sorry

end NUMINAMATH_CALUDE_tom_hockey_games_l4014_401450


namespace NUMINAMATH_CALUDE_complex_square_roots_l4014_401400

theorem complex_square_roots (z : ℂ) : 
  z^2 = -45 - 28*I ↔ z = 2 - 7*I ∨ z = -2 + 7*I :=
sorry

end NUMINAMATH_CALUDE_complex_square_roots_l4014_401400


namespace NUMINAMATH_CALUDE_drug_price_reduction_l4014_401446

theorem drug_price_reduction (initial_price final_price : ℝ) (x : ℝ) 
  (h_initial : initial_price = 56)
  (h_final : final_price = 31.5)
  (h_positive : 0 < x ∧ x < 1) :
  initial_price * (1 - x)^2 = final_price := by
  sorry

end NUMINAMATH_CALUDE_drug_price_reduction_l4014_401446


namespace NUMINAMATH_CALUDE_seojun_apple_fraction_l4014_401468

theorem seojun_apple_fraction :
  let total_apples : ℕ := 100
  let seojun_apples : ℕ := 11
  (seojun_apples : ℚ) / total_apples = 0.11 := by
  sorry

end NUMINAMATH_CALUDE_seojun_apple_fraction_l4014_401468


namespace NUMINAMATH_CALUDE_disjunction_true_l4014_401435

def p : Prop := ∃ k : ℕ, 2 = 2 * k

def q : Prop := ∃ k : ℕ, 3 = 2 * k

theorem disjunction_true : p ∨ q := by
  sorry

end NUMINAMATH_CALUDE_disjunction_true_l4014_401435


namespace NUMINAMATH_CALUDE_sin_3x_periodic_l4014_401404

/-- The function f(x) = sin(3x) is periodic with period 2π/3 -/
theorem sin_3x_periodic (x : ℝ) : Real.sin (3 * (x + 2 * Real.pi / 3)) = Real.sin (3 * x) := by
  sorry

end NUMINAMATH_CALUDE_sin_3x_periodic_l4014_401404


namespace NUMINAMATH_CALUDE_susan_remaining_money_l4014_401497

def susan_spending (initial_amount games_multiplier snacks_cost souvenir_cost : ℕ) : ℕ :=
  initial_amount - (snacks_cost + games_multiplier * snacks_cost + souvenir_cost)

theorem susan_remaining_money :
  susan_spending 80 3 15 10 = 10 := by
  sorry

end NUMINAMATH_CALUDE_susan_remaining_money_l4014_401497


namespace NUMINAMATH_CALUDE_paper_thickness_after_two_folds_l4014_401476

/-- The thickness of a paper after folding it in half a given number of times. -/
def thickness (initial : ℝ) (folds : ℕ) : ℝ :=
  initial * (2 ^ folds)

/-- Theorem: The thickness of a paper with initial thickness 0.1 mm after 2 folds is 0.4 mm. -/
theorem paper_thickness_after_two_folds :
  thickness 0.1 2 = 0.4 := by
  sorry

end NUMINAMATH_CALUDE_paper_thickness_after_two_folds_l4014_401476


namespace NUMINAMATH_CALUDE_august_math_problems_l4014_401444

theorem august_math_problems (a1 a2 a3 : ℕ) : 
  a1 = 600 →
  a3 = a1 + a2 - 400 →
  a1 + a2 + a3 = 3200 →
  a2 / a1 = 2 := by
sorry

end NUMINAMATH_CALUDE_august_math_problems_l4014_401444


namespace NUMINAMATH_CALUDE_pool_draining_and_filling_time_l4014_401426

/-- The time it takes for a pool to reach a certain water level when being simultaneously drained and filled -/
theorem pool_draining_and_filling_time 
  (pool_capacity : ℝ) 
  (drain_rate : ℝ) 
  (fill_rate : ℝ) 
  (final_volume : ℝ) 
  (h1 : pool_capacity = 120)
  (h2 : drain_rate = 1 / 4)
  (h3 : fill_rate = 1 / 6)
  (h4 : final_volume = 90) :
  ∃ t : ℝ, t = 3 ∧ 
  pool_capacity - (drain_rate * pool_capacity - fill_rate * pool_capacity) * t = final_volume :=
sorry

end NUMINAMATH_CALUDE_pool_draining_and_filling_time_l4014_401426


namespace NUMINAMATH_CALUDE_train_passenger_ratio_l4014_401469

theorem train_passenger_ratio :
  let initial_passengers : ℕ := 288
  let first_drop : ℕ := initial_passengers / 3
  let first_take : ℕ := 280
  let second_take : ℕ := 12
  let third_station_passengers : ℕ := 248
  
  let after_first_station : ℕ := initial_passengers - first_drop + first_take
  let dropped_second_station : ℕ := after_first_station - (third_station_passengers - second_take)
  let ratio : ℚ := dropped_second_station / after_first_station
  
  ratio = 1 / 2 := by sorry

end NUMINAMATH_CALUDE_train_passenger_ratio_l4014_401469


namespace NUMINAMATH_CALUDE_storm_rainfall_l4014_401418

theorem storm_rainfall (first_hour : ℝ) (second_hour : ℝ) : 
  second_hour = 2 * first_hour + 7 →
  first_hour + second_hour = 22 →
  first_hour = 5 := by
sorry

end NUMINAMATH_CALUDE_storm_rainfall_l4014_401418


namespace NUMINAMATH_CALUDE_sum_of_ages_matt_age_relation_l4014_401406

/-- Given Matt's age and John's age, prove the sum of their ages -/
theorem sum_of_ages (matt_age john_age : ℕ) (h1 : matt_age = 41) (h2 : john_age = 11) :
  matt_age + john_age = 52 := by
  sorry

/-- Matt's age in relation to John's -/
theorem matt_age_relation (matt_age john_age : ℕ) (h1 : matt_age = 41) (h2 : john_age = 11) :
  matt_age = 4 * john_age - 3 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_ages_matt_age_relation_l4014_401406


namespace NUMINAMATH_CALUDE_smallest_gcd_qr_l4014_401445

theorem smallest_gcd_qr (p q r : ℕ+) (h1 : Nat.gcd p q = 240) (h2 : Nat.gcd p r = 540) :
  ∃ (q' r' : ℕ+), Nat.gcd q'.val r'.val = 60 ∧
    ∀ (q'' r'' : ℕ+), Nat.gcd q''.val r''.val ≥ 60 :=
sorry

end NUMINAMATH_CALUDE_smallest_gcd_qr_l4014_401445


namespace NUMINAMATH_CALUDE_red_minus_white_equals_three_l4014_401464

-- Define the flower counts for each category
def total_flowers : ℕ := 100
def yellow_white : ℕ := 13
def red_yellow : ℕ := 17
def red_white : ℕ := 14
def blue_yellow : ℕ := 16
def blue_white : ℕ := 9
def red_blue_yellow : ℕ := 8
def red_white_blue : ℕ := 6

-- Define the number of flowers containing red
def red_flowers : ℕ := red_yellow + red_white + red_blue_yellow + red_white_blue

-- Define the number of flowers containing white
def white_flowers : ℕ := yellow_white + red_white + blue_white + red_white_blue

-- Theorem statement
theorem red_minus_white_equals_three :
  red_flowers - white_flowers = 3 :=
by sorry

end NUMINAMATH_CALUDE_red_minus_white_equals_three_l4014_401464
