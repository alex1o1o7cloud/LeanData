import Mathlib

namespace NUMINAMATH_CALUDE_greatest_n_with_perfect_square_property_l885_88586

def sum_of_squares (n : ℕ) : ℕ := n * (n + 1) * (2 * n + 1) / 6

def is_perfect_square (m : ℕ) : Prop := ∃ k : ℕ, m = k^2

theorem greatest_n_with_perfect_square_property :
  ∃ (n : ℕ), n = 1921 ∧ n ≤ 2008 ∧
  (∀ m : ℕ, m ≤ 2008 → m > n →
    ¬ is_perfect_square ((sum_of_squares n) * (sum_of_squares (2 * n) - sum_of_squares n))) ∧
  is_perfect_square ((sum_of_squares n) * (sum_of_squares (2 * n) - sum_of_squares n)) :=
sorry

end NUMINAMATH_CALUDE_greatest_n_with_perfect_square_property_l885_88586


namespace NUMINAMATH_CALUDE_sum_of_roots_zero_l885_88585

/-- Given a quadratic equation x^2 + px + q = 0 where p and q are its roots and p = -q,
    prove that the sum of the roots is 0 -/
theorem sum_of_roots_zero (p q : ℝ) (h1 : p = -q) (h2 : p * p + p * q + q = 0) :
  p + q = 0 := by sorry

end NUMINAMATH_CALUDE_sum_of_roots_zero_l885_88585


namespace NUMINAMATH_CALUDE_fraction_sum_equals_half_l885_88531

theorem fraction_sum_equals_half : (2 / 12 : ℚ) + (4 / 24 : ℚ) + (6 / 36 : ℚ) = (1 / 2 : ℚ) := by
  sorry

end NUMINAMATH_CALUDE_fraction_sum_equals_half_l885_88531


namespace NUMINAMATH_CALUDE_total_tickets_sold_l885_88537

theorem total_tickets_sold (child_cost adult_cost total_revenue child_count : ℕ) 
  (h1 : child_cost = 6)
  (h2 : adult_cost = 9)
  (h3 : total_revenue = 1875)
  (h4 : child_count = 50) :
  child_count + (total_revenue - child_cost * child_count) / adult_cost = 225 :=
by sorry

end NUMINAMATH_CALUDE_total_tickets_sold_l885_88537


namespace NUMINAMATH_CALUDE_donna_weekly_episodes_l885_88573

/-- The number of episodes Donna can watch on a weekday -/
def weekday_episodes : ℕ := 8

/-- The number of weekdays in a week -/
def weekdays : ℕ := 5

/-- The number of weekend days in a week -/
def weekend_days : ℕ := 2

/-- The factor by which weekend watching increases compared to weekdays -/
def weekend_factor : ℕ := 3

/-- The total number of episodes Donna can watch in a week -/
def total_episodes : ℕ := weekday_episodes * weekdays + weekend_factor * weekday_episodes * weekend_days

theorem donna_weekly_episodes : total_episodes = 88 := by
  sorry

end NUMINAMATH_CALUDE_donna_weekly_episodes_l885_88573


namespace NUMINAMATH_CALUDE_coloring_existence_l885_88519

theorem coloring_existence : ∃ (f : ℕ → Bool), 
  ∀ (a : ℕ → ℕ) (d : ℕ),
    (∀ i : Fin 18, a i < a (i + 1)) →
    (∀ i j : Fin 18, a j - a i = d * (j - i)) →
    1 ≤ a 0 → a 17 ≤ 1986 →
    ∃ i j : Fin 18, f (a i) ≠ f (a j) := by
  sorry

end NUMINAMATH_CALUDE_coloring_existence_l885_88519


namespace NUMINAMATH_CALUDE_area_of_specific_trapezoid_l885_88575

/-- A right trapezoid with an inscribed circle -/
structure RightTrapezoidWithInscribedCircle where
  /-- The length of the smaller segment of the larger lateral side -/
  smaller_segment : ℝ
  /-- The length of the larger segment of the larger lateral side -/
  larger_segment : ℝ
  /-- The smaller segment is positive -/
  smaller_segment_pos : 0 < smaller_segment
  /-- The larger segment is positive -/
  larger_segment_pos : 0 < larger_segment

/-- The area of a right trapezoid with an inscribed circle -/
def area (t : RightTrapezoidWithInscribedCircle) : ℝ :=
  18 -- Definition without proof

/-- Theorem stating that the area of the specific right trapezoid is 18 -/
theorem area_of_specific_trapezoid :
  ∀ t : RightTrapezoidWithInscribedCircle,
  t.smaller_segment = 1 ∧ t.larger_segment = 4 →
  area t = 18 := by
  sorry

end NUMINAMATH_CALUDE_area_of_specific_trapezoid_l885_88575


namespace NUMINAMATH_CALUDE_area_of_triangle_perimeter_of_triangle_l885_88599

-- Define the triangle
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

-- Define the conditions
def triangle_conditions (t : Triangle) : Prop :=
  t.a = 3 ∧ t.b = 2 * t.c

-- Part 1
theorem area_of_triangle (t : Triangle) (h : triangle_conditions t) (h_A : t.A = 2 * Real.pi / 3) :
  (1/2) * t.b * t.c * Real.sin t.A = 9 * Real.sqrt 3 / 14 := by sorry

-- Part 2
theorem perimeter_of_triangle (t : Triangle) (h : triangle_conditions t) (h_BC : 2 * Real.sin t.B - Real.sin t.C = 1) :
  t.a + t.b + t.c = 4 * Real.sqrt 2 - Real.sqrt 5 + 3 ∨ 
  t.a + t.b + t.c = 4 * Real.sqrt 2 + Real.sqrt 5 + 3 := by sorry

end NUMINAMATH_CALUDE_area_of_triangle_perimeter_of_triangle_l885_88599


namespace NUMINAMATH_CALUDE_complex_magnitude_square_counterexample_l885_88582

theorem complex_magnitude_square_counterexample : 
  ∃ (z₁ z₂ : ℂ), Complex.abs z₁ = Complex.abs z₂ ∧ z₁^2 ≠ z₂^2 := by
  sorry

end NUMINAMATH_CALUDE_complex_magnitude_square_counterexample_l885_88582


namespace NUMINAMATH_CALUDE_max_product_with_constraint_l885_88521

theorem max_product_with_constraint (a b : ℝ) : 
  a > 0 → b > 0 → 9 * a^2 + 16 * b^2 = 25 → a * b ≤ 25 / 24 := by
  sorry

end NUMINAMATH_CALUDE_max_product_with_constraint_l885_88521


namespace NUMINAMATH_CALUDE_jake_balloons_count_l885_88580

/-- The number of balloons Allan brought to the park -/
def allan_balloons : ℕ := 5

/-- The additional number of balloons Jake brought compared to Allan -/
def jake_extra_balloons : ℕ := 6

/-- The number of balloons Jake brought to the park -/
def jake_balloons : ℕ := allan_balloons + jake_extra_balloons

theorem jake_balloons_count : jake_balloons = 11 := by sorry

end NUMINAMATH_CALUDE_jake_balloons_count_l885_88580


namespace NUMINAMATH_CALUDE_equation_represents_line_l885_88501

-- Define the equation
def equation (x y : ℝ) : Prop :=
  ((x^2 + y^2 - 2*x) * Real.sqrt (x + y - 3) = 0)

-- Define what it means for the equation to represent a line
def represents_line (f : ℝ → ℝ → Prop) : Prop :=
  ∃ a b c : ℝ, a ≠ 0 ∨ b ≠ 0 ∧ ∀ x y : ℝ, f x y ↔ a*x + b*y + c = 0

-- Theorem statement
theorem equation_represents_line :
  represents_line equation :=
sorry

end NUMINAMATH_CALUDE_equation_represents_line_l885_88501


namespace NUMINAMATH_CALUDE_race_head_start_l885_88514

theorem race_head_start (L : ℝ) (va vb : ℝ) (h : va = (15 / 13) * vb) :
  let H := (L - (13 * L / 15) + (1 / 4 * L))
  H = (23 / 60) * L := by sorry

end NUMINAMATH_CALUDE_race_head_start_l885_88514


namespace NUMINAMATH_CALUDE_smallest_sum_of_coefficients_l885_88503

theorem smallest_sum_of_coefficients (a b : ℤ) : 
  (∀ x : ℝ, (x^2 + a*x + 20)*(x^2 + 17*x + b) = 0 → (∃ k : ℤ, x = ↑k ∧ k < 0)) →
  (∀ c d : ℤ, (∀ y : ℝ, (y^2 + c*y + 20)*(y^2 + 17*y + d) = 0 → (∃ m : ℤ, y = ↑m ∧ m < 0)) → 
    a + b ≤ c + d) →
  a + b = -5 := by
sorry

end NUMINAMATH_CALUDE_smallest_sum_of_coefficients_l885_88503


namespace NUMINAMATH_CALUDE_exists_triangle_cut_into_2005_l885_88584

/-- There exists a right-angled triangle with integer side lengths that can be cut into 2005 congruent triangles. -/
theorem exists_triangle_cut_into_2005 : ∃ (a b c : ℕ), 
  a^2 + b^2 = c^2 ∧ a * b = 2005 := by
  sorry

end NUMINAMATH_CALUDE_exists_triangle_cut_into_2005_l885_88584


namespace NUMINAMATH_CALUDE_triangle_abc_properties_l885_88507

theorem triangle_abc_properties (A B C : ℝ) (a b c : ℝ) :
  -- Triangle ABC exists
  0 < A ∧ 0 < B ∧ 0 < C ∧ A + B + C = π →
  -- Side lengths are positive
  0 < a ∧ 0 < b ∧ 0 < c →
  -- Law of sines
  a / (Real.sin A) = b / (Real.sin B) →
  a / (Real.sin A) = c / (Real.sin C) →
  -- Given condition
  a * Real.cos B + b * Real.cos A = 2 * c * Real.cos B →
  -- Conclusions
  B = π / 3 ∧
  (∀ x, x ∈ Set.Ioo (-3/2) (1/2) ↔
    ∃ A', 0 < A' ∧ A' < 2*π/3 ∧
    x = Real.sin A' * (Real.sqrt 3 * Real.cos A' - Real.sin A')) :=
by sorry

end NUMINAMATH_CALUDE_triangle_abc_properties_l885_88507


namespace NUMINAMATH_CALUDE_circle_radius_l885_88554

theorem circle_radius (x y : ℝ) : 
  (x^2 - 10*x + y^2 - 8*y + 29 = 0) → 
  (∃ (h k r : ℝ), (x - h)^2 + (y - k)^2 = r^2 ∧ r = 2*Real.sqrt 3) :=
by sorry

end NUMINAMATH_CALUDE_circle_radius_l885_88554


namespace NUMINAMATH_CALUDE_solve_for_y_l885_88551

theorem solve_for_y (x y : ℝ) (h1 : x^2 - 3*x + 6 = y + 2) (h2 : x = -5) : y = 44 := by
  sorry

end NUMINAMATH_CALUDE_solve_for_y_l885_88551


namespace NUMINAMATH_CALUDE_vector_parallel_value_l885_88558

/-- Two 2D vectors are parallel if their cross product is zero -/
def parallel (v w : ℝ × ℝ) : Prop :=
  v.1 * w.2 = v.2 * w.1

theorem vector_parallel_value : 
  let a : ℝ × ℝ := (3, 2)
  let b : ℝ → ℝ × ℝ := λ m ↦ (m, -4)
  ∀ m : ℝ, parallel a (b m) → m = 6 := by
sorry

end NUMINAMATH_CALUDE_vector_parallel_value_l885_88558


namespace NUMINAMATH_CALUDE_vectors_form_basis_l885_88590

def e₁ : ℝ × ℝ := (-1, 2)
def e₂ : ℝ × ℝ := (5, 7)

theorem vectors_form_basis : 
  LinearIndependent ℝ (![e₁, e₂] : Fin 2 → ℝ × ℝ) :=
sorry

end NUMINAMATH_CALUDE_vectors_form_basis_l885_88590


namespace NUMINAMATH_CALUDE_forum_posts_theorem_l885_88561

/-- Calculates the total number of questions and answers posted on a forum in a day. -/
def forum_posts (members : ℕ) (questions_per_hour : ℕ) (answer_ratio : ℕ) : ℕ :=
  let questions_per_day := questions_per_hour * 24
  let answers_per_day := questions_per_day * answer_ratio
  members * (questions_per_day + answers_per_day)

/-- Theorem: Given the forum conditions, the total posts in a day is 1,008,000. -/
theorem forum_posts_theorem :
  forum_posts 1000 7 5 = 1008000 := by
  sorry

end NUMINAMATH_CALUDE_forum_posts_theorem_l885_88561


namespace NUMINAMATH_CALUDE_paving_cost_l885_88533

/-- The cost of paving a rectangular floor -/
theorem paving_cost (length width rate : ℝ) (h1 : length = 6.5) (h2 : width = 2.75) (h3 : rate = 600) :
  length * width * rate = 10725 := by
  sorry

end NUMINAMATH_CALUDE_paving_cost_l885_88533


namespace NUMINAMATH_CALUDE_train_speed_problem_l885_88596

theorem train_speed_problem (distance : ℝ) (speed_ab : ℝ) (time_difference : ℝ) :
  distance = 480 →
  speed_ab = 160 →
  time_difference = 1 →
  let time_ab := distance / speed_ab
  let time_ba := time_ab + time_difference
  let speed_ba := distance / time_ba
  speed_ba = 120 := by sorry

end NUMINAMATH_CALUDE_train_speed_problem_l885_88596


namespace NUMINAMATH_CALUDE_rhombus_diagonal_length_l885_88523

/-- A rhombus with given area and diagonal ratio has a specific longer diagonal length -/
theorem rhombus_diagonal_length 
  (area : ℝ) 
  (diagonal_ratio : ℚ) 
  (h_area : area = 150) 
  (h_ratio : diagonal_ratio = 4 / 3) : 
  ∃ (d1 d2 : ℝ), d1 > d2 ∧ d1 / d2 = diagonal_ratio ∧ area = (d1 * d2) / 2 ∧ d1 = 20 := by
  sorry

end NUMINAMATH_CALUDE_rhombus_diagonal_length_l885_88523


namespace NUMINAMATH_CALUDE_chord_equation_through_midpoint_l885_88553

/-- The equation of a line containing a chord of an ellipse, where the chord passes through a given point and has that point as its midpoint. -/
theorem chord_equation_through_midpoint (x y : ℝ) :
  (4 * x^2 + 9 * y^2 = 144) →  -- Ellipse equation
  (3 : ℝ)^2 * 4 + 2^2 * 9 < 144 →  -- Point (3, 2) is inside the ellipse
  ∃ (x₁ y₁ x₂ y₂ : ℝ),
    (4 * x₁^2 + 9 * y₁^2 = 144) ∧  -- Point (x₁, y₁) is on the ellipse
    (4 * x₂^2 + 9 * y₂^2 = 144) ∧  -- Point (x₂, y₂) is on the ellipse
    (x₁ + x₂) / 2 = 3 ∧  -- (3, 2) is the midpoint of (x₁, y₁) and (x₂, y₂)
    (y₁ + y₂) / 2 = 2 ∧
    2 * x + 3 * y - 12 = 0  -- Equation of the line containing the chord
  := by sorry

end NUMINAMATH_CALUDE_chord_equation_through_midpoint_l885_88553


namespace NUMINAMATH_CALUDE_nested_root_equality_l885_88574

theorem nested_root_equality (a : ℝ) (ha : a > 0) : 
  Real.sqrt (a * Real.sqrt (a * Real.sqrt a)) = a ^ (7/8) :=
by sorry

end NUMINAMATH_CALUDE_nested_root_equality_l885_88574


namespace NUMINAMATH_CALUDE_prime_cube_plus_five_prime_l885_88542

theorem prime_cube_plus_five_prime (p : ℕ) 
  (hp : Nat.Prime p) 
  (hp_cube : Nat.Prime (p^3 + 5)) : 
  p^5 - 7 = 25 := by
sorry

end NUMINAMATH_CALUDE_prime_cube_plus_five_prime_l885_88542


namespace NUMINAMATH_CALUDE_unit_digit_of_product_l885_88525

theorem unit_digit_of_product : (5 + 1) * (5^3 + 1) * (5^6 + 1) * (5^12 + 1) % 10 = 6 := by
  sorry

end NUMINAMATH_CALUDE_unit_digit_of_product_l885_88525


namespace NUMINAMATH_CALUDE_newspaper_photos_newspaper_photos_proof_l885_88565

theorem newspaper_photos : ℕ → ℕ → ℕ → ℕ → ℕ → Prop :=
  fun (pages_with_two_photos : ℕ) 
      (photos_per_page_first : ℕ) 
      (pages_with_three_photos : ℕ) 
      (photos_per_page_second : ℕ) 
      (total_photos : ℕ) =>
    pages_with_two_photos = 12 ∧ 
    photos_per_page_first = 2 ∧
    pages_with_three_photos = 9 ∧ 
    photos_per_page_second = 3 →
    total_photos = pages_with_two_photos * photos_per_page_first + 
                   pages_with_three_photos * photos_per_page_second ∧
    total_photos = 51

theorem newspaper_photos_proof : newspaper_photos 12 2 9 3 51 := by
  sorry

end NUMINAMATH_CALUDE_newspaper_photos_newspaper_photos_proof_l885_88565


namespace NUMINAMATH_CALUDE_smallest_satisfying_number_l885_88516

/-- Represents the decimal expansion of a rational number -/
def DecimalExpansion (q : ℚ) : List ℕ := sorry

/-- Checks if a list contains three consecutive identical elements -/
def hasThreeConsecutiveIdentical (l : List ℕ) : Prop := sorry

/-- Checks if a list is entirely composed of identical elements -/
def isEntirelyIdentical (l : List ℕ) : Prop := sorry

/-- Checks if a natural number satisfies the given conditions -/
def satisfiesConditions (n : ℕ) : Prop :=
  let expansion := DecimalExpansion (1 / n)
  hasThreeConsecutiveIdentical expansion ∧ ¬isEntirelyIdentical expansion

theorem smallest_satisfying_number :
  satisfiesConditions 157 ∧ ∀ m < 157, ¬satisfiesConditions m := by sorry

end NUMINAMATH_CALUDE_smallest_satisfying_number_l885_88516


namespace NUMINAMATH_CALUDE_absolute_value_inequality_l885_88595

theorem absolute_value_inequality (m : ℝ) :
  (∀ x : ℝ, |x - 3| + |x + 4| ≥ |2*m - 1|) ↔ -3 ≤ m ∧ m ≤ 4 := by
  sorry

end NUMINAMATH_CALUDE_absolute_value_inequality_l885_88595


namespace NUMINAMATH_CALUDE_unique_quadratic_solution_l885_88510

theorem unique_quadratic_solution (a b : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) 
  (h_unique : ∃! x, (5*a + 2*b)*x^2 + a*x + b = 0) : 
  ∃ x, (5*a + 2*b)*x^2 + a*x + b = 0 ∧ x = 5/2 := by
  sorry

end NUMINAMATH_CALUDE_unique_quadratic_solution_l885_88510


namespace NUMINAMATH_CALUDE_distribute_books_correct_l885_88517

/-- The number of ways to distribute 6 different books among three people. -/
def distribute_books : Nat × Nat × Nat × Nat :=
  let n : Nat := 6  -- Total number of books
  let k : Nat := 3  -- Number of people

  -- Scenario 1: One person gets 1 book, another gets 2 books, the last gets 3 books
  let scenario1 : Nat := k.factorial * n.choose 1 * (n - 1).choose 2 * (n - 3).choose 3

  -- Scenario 2: Books are evenly distributed, each person getting 2 books
  let scenario2 : Nat := (n.choose 2 * (n - 2).choose 2 * (n - 4).choose 2) / k.factorial

  -- Scenario 3: One part gets 4 books, other two parts get 1 book each
  let scenario3 : Nat := n.choose 4

  -- Scenario 4: A gets 1 book, B gets 1 book, C gets 4 books
  let scenario4 : Nat := n.choose 4 * (n - 4).choose 1

  (scenario1, scenario2, scenario3, scenario4)

theorem distribute_books_correct :
  distribute_books = (360, 90, 15, 30) := by
  sorry

end NUMINAMATH_CALUDE_distribute_books_correct_l885_88517


namespace NUMINAMATH_CALUDE_study_group_lawyers_l885_88579

theorem study_group_lawyers (total_members : ℝ) (h1 : total_members > 0) : 
  let women_ratio : ℝ := 0.4
  let women_lawyer_prob : ℝ := 0.08
  let women_lawyer_ratio : ℝ := women_lawyer_prob / women_ratio
  women_lawyer_ratio = 0.2 := by sorry

end NUMINAMATH_CALUDE_study_group_lawyers_l885_88579


namespace NUMINAMATH_CALUDE_factorial_ratio_simplification_l885_88572

theorem factorial_ratio_simplification (N : ℕ) :
  (Nat.factorial N * (N + 2)) / Nat.factorial (N + 3) = 1 / ((N + 3) * (N + 1)) :=
by sorry

end NUMINAMATH_CALUDE_factorial_ratio_simplification_l885_88572


namespace NUMINAMATH_CALUDE_sum_of_specific_S_l885_88515

def S (n : ℕ) : ℤ :=
  if n % 2 = 0 then
    -n / 2
  else
    (n + 1) / 2

theorem sum_of_specific_S : S 17 + S 33 + S 50 = 1 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_specific_S_l885_88515


namespace NUMINAMATH_CALUDE_eulers_formula_two_power_inequality_l885_88509

-- Define the complex number i
noncomputable def i : ℂ := Complex.I

-- Statement 1: Euler's formula
theorem eulers_formula (x : ℝ) : Complex.exp (i * x) = Complex.cos x + i * Complex.sin x := by sorry

-- Statement 2: Inequality for 2^x
theorem two_power_inequality (x : ℝ) (h : x ≥ 0) : 
  (2 : ℝ) ^ x ≥ 1 + x * Real.log 2 + (x * Real.log 2)^2 / 2 := by sorry

end NUMINAMATH_CALUDE_eulers_formula_two_power_inequality_l885_88509


namespace NUMINAMATH_CALUDE_francine_daily_drive_distance_l885_88546

/-- The number of days Francine doesn't go to work each week -/
def days_off_per_week : ℕ := 3

/-- The total distance Francine drives to work in 4 weeks (in km) -/
def total_distance_4_weeks : ℕ := 2240

/-- The number of weeks in the given period -/
def num_weeks : ℕ := 4

/-- The number of working days in a week -/
def work_days_per_week : ℕ := 7 - days_off_per_week

/-- The total number of working days in 4 weeks -/
def total_work_days : ℕ := work_days_per_week * num_weeks

/-- The distance Francine drives to work each day (in km) -/
def daily_distance : ℕ := total_distance_4_weeks / total_work_days

theorem francine_daily_drive_distance :
  daily_distance = 280 := by sorry

end NUMINAMATH_CALUDE_francine_daily_drive_distance_l885_88546


namespace NUMINAMATH_CALUDE_p_necessary_not_sufficient_for_q_l885_88545

theorem p_necessary_not_sufficient_for_q :
  (∀ x : ℝ, |x - 1| < 2 → (x + 2) * (x - 3) < 0) ∧
  (∃ x : ℝ, (x + 2) * (x - 3) < 0 ∧ |x - 1| ≥ 2) := by
  sorry

end NUMINAMATH_CALUDE_p_necessary_not_sufficient_for_q_l885_88545


namespace NUMINAMATH_CALUDE_f_strictly_increasing_when_a_eq_one_f_increasing_intervals_l885_88576

noncomputable section

-- Define the function f(x)
def f (a : ℝ) (x : ℝ) : ℝ := x - (a + 1) * Real.log x - a / x

-- Theorem for part (I)
theorem f_strictly_increasing_when_a_eq_one :
  ∀ x₁ x₂ : ℝ, 0 < x₁ → 0 < x₂ → x₁ < x₂ → f 1 x₁ < f 1 x₂ := by sorry

-- Theorem for part (II)
theorem f_increasing_intervals (a : ℝ) :
  (a ≤ 0 → ∀ x₁ x₂ : ℝ, 1 < x₁ → x₁ < x₂ → f a x₁ < f a x₂) ∧
  (0 < a → a < 1 → 
    (∀ x₁ x₂ : ℝ, 0 < x₁ → x₁ < x₂ → x₂ < a → f a x₁ < f a x₂) ∧
    (∀ x₁ x₂ : ℝ, 1 < x₁ → x₁ < x₂ → f a x₁ < f a x₂)) ∧
  (a = 1 → ∀ x₁ x₂ : ℝ, 0 < x₁ → x₁ < x₂ → f a x₁ < f a x₂) ∧
  (1 < a → 
    (∀ x₁ x₂ : ℝ, 0 < x₁ → x₁ < x₂ → x₂ < 1 → f a x₁ < f a x₂) ∧
    (∀ x₁ x₂ : ℝ, a < x₁ → x₁ < x₂ → f a x₁ < f a x₂)) := by sorry

end

end NUMINAMATH_CALUDE_f_strictly_increasing_when_a_eq_one_f_increasing_intervals_l885_88576


namespace NUMINAMATH_CALUDE_red_ball_probability_l885_88568

theorem red_ball_probability (w r : ℕ+) 
  (h1 : r > w)
  (h2 : r < 2 * w)
  (h3 : 2 * w + 3 * r = 60) :
  (r : ℚ) / (w + r) = 4 / 7 := by
  sorry

end NUMINAMATH_CALUDE_red_ball_probability_l885_88568


namespace NUMINAMATH_CALUDE_dance_attendance_l885_88502

/-- The number of boys attending the dance -/
def num_boys : ℕ := 14

/-- The number of girls attending the dance -/
def num_girls : ℕ := num_boys / 2

theorem dance_attendance :
  (num_boys = 2 * num_girls) ∧
  (num_boys = (num_girls - 1) + 8) →
  num_boys = 14 :=
by sorry

end NUMINAMATH_CALUDE_dance_attendance_l885_88502


namespace NUMINAMATH_CALUDE_intersection_distance_l885_88559

/-- The distance between intersection points of a line and a circle --/
theorem intersection_distance (t : ℝ) : 
  let x : ℝ → ℝ := λ t => -1 + (Real.sqrt 3 / 2) * t
  let y : ℝ → ℝ := λ t => (1 / 2) * t
  let l : ℝ → ℝ × ℝ := λ t => (x t, y t)
  let C : ℝ → ℝ := λ θ => 4 * Real.cos θ
  ∃ P Q : ℝ × ℝ, P ≠ Q ∧ 
    (P.1 - 2)^2 + P.2^2 = 4 ∧
    (Q.1 - 2)^2 + Q.2^2 = 4 ∧
    P.1 - Real.sqrt 3 * P.2 + 1 = 0 ∧
    Q.1 - Real.sqrt 3 * Q.2 + 1 = 0 ∧
    (P.1 - Q.1)^2 + (P.2 - Q.2)^2 = 7 :=
by
  sorry

end NUMINAMATH_CALUDE_intersection_distance_l885_88559


namespace NUMINAMATH_CALUDE_min_games_for_five_guys_five_girls_l885_88520

/-- Represents a bridge game configuration -/
structure BridgeConfig where
  num_guys : ℕ
  num_girls : ℕ
  guys_per_team : ℕ
  girls_per_team : ℕ

/-- Calculates the minimum number of games required for a given bridge configuration -/
def min_games (config : BridgeConfig) : ℕ :=
  (config.num_guys * config.num_girls * config.girls_per_team) / (2 * config.guys_per_team)

/-- Theorem stating the minimum number of games for the specific configuration -/
theorem min_games_for_five_guys_five_girls :
  let config := BridgeConfig.mk 5 5 2 2
  min_games config = 25 := by sorry

end NUMINAMATH_CALUDE_min_games_for_five_guys_five_girls_l885_88520


namespace NUMINAMATH_CALUDE_specific_value_problem_l885_88529

theorem specific_value_problem (x : ℕ) (specific_value : ℕ) 
  (h1 : 25 * x = specific_value) 
  (h2 : x = 27) : 
  specific_value = 675 := by
sorry

end NUMINAMATH_CALUDE_specific_value_problem_l885_88529


namespace NUMINAMATH_CALUDE_inscribed_circle_radius_l885_88518

/-- Configuration of semicircles and inscribed circle -/
structure SemicircleConfig where
  R : ℝ  -- Radius of larger semicircle
  r : ℝ  -- Radius of smaller semicircle
  x : ℝ  -- Radius of inscribed circle

/-- The inscribed circle touches both semicircles and the diameter -/
def touches_all (c : SemicircleConfig) : Prop :=
  ∃ (O O₁ O₂ : ℝ × ℝ) (P : ℝ × ℝ),
    let (xₒ, yₒ) := O
    let (x₁, y₁) := O₁
    let (x₂, y₂) := O₂
    let (xₚ, yₚ) := P
    (xₒ - x₂)^2 + (yₒ - y₂)^2 = (c.R - c.x)^2 ∧  -- Larger semicircle touches inscribed circle
    (x₁ - x₂)^2 + (y₁ - y₂)^2 = (c.r + c.x)^2 ∧  -- Smaller semicircle touches inscribed circle
    (x₂ - xₚ)^2 + (y₂ - yₚ)^2 = c.x^2           -- Inscribed circle touches diameter

/-- Main theorem: The radius of the inscribed circle is 8 cm -/
theorem inscribed_circle_radius
  (c : SemicircleConfig)
  (h₁ : c.R = 18)
  (h₂ : c.r = 9)
  (h₃ : touches_all c) :
  c.x = 8 :=
sorry

end NUMINAMATH_CALUDE_inscribed_circle_radius_l885_88518


namespace NUMINAMATH_CALUDE_smaller_square_area_percentage_l885_88524

/-- Represents a square inscribed in a circle with another smaller square -/
structure InscribedSquares where
  -- Radius of the circle
  r : ℝ
  -- Side length of the larger square
  s : ℝ
  -- Side length of the smaller square
  x : ℝ
  -- The larger square is inscribed in the circle
  h1 : r = s * Real.sqrt 2 / 2
  -- The smaller square has one side coinciding with the larger square
  h2 : x ≤ s
  -- Two vertices of the smaller square are on the circle
  h3 : (s/2 + x)^2 + x^2 = r^2

/-- The theorem stating that the area of the smaller square is 4% of the larger square -/
theorem smaller_square_area_percentage (sq : InscribedSquares) (h : sq.s = 4) :
  (sq.x^2) / (sq.s^2) = 0.04 := by
  sorry

end NUMINAMATH_CALUDE_smaller_square_area_percentage_l885_88524


namespace NUMINAMATH_CALUDE_special_polygon_properties_l885_88598

/-- A polygon where each interior angle is 4 times the exterior angle at the same vertex -/
structure SpecialPolygon where
  vertices : ℕ
  interior_angle : Fin vertices → ℝ
  exterior_angle : Fin vertices → ℝ
  angle_relation : ∀ i, interior_angle i = 4 * exterior_angle i
  sum_exterior_angles : (Finset.univ.sum exterior_angle) = 360

theorem special_polygon_properties (Q : SpecialPolygon) :
  (Finset.univ.sum Q.interior_angle = 1440) ∧
  (∀ i j, Q.interior_angle i = Q.interior_angle j) := by
  sorry

#check special_polygon_properties

end NUMINAMATH_CALUDE_special_polygon_properties_l885_88598


namespace NUMINAMATH_CALUDE_product_of_imaginary_parts_l885_88564

-- Define the complex number i
def i : ℂ := Complex.I

-- Define the equation
def equation (z : ℂ) : Prop := z^2 + 3*z = 3 - 4*i

-- Define a function to get the imaginary part of a complex number
def imag (z : ℂ) : ℝ := z.im

-- Theorem statement
theorem product_of_imaginary_parts : 
  ∃ (z₁ z₂ : ℂ), equation z₁ ∧ equation z₂ ∧ z₁ ≠ z₂ ∧ (imag z₁ * imag z₂ = 16/25) :=
sorry

end NUMINAMATH_CALUDE_product_of_imaginary_parts_l885_88564


namespace NUMINAMATH_CALUDE_simplify_complex_fraction_l885_88547

theorem simplify_complex_fraction :
  (1 / ((1 / (Real.sqrt 5 + 2)) - (2 / (Real.sqrt 7 - 3)))) =
  ((Real.sqrt 5 + Real.sqrt 7 - 1) / (11 + 2 * Real.sqrt 35)) := by
  sorry

end NUMINAMATH_CALUDE_simplify_complex_fraction_l885_88547


namespace NUMINAMATH_CALUDE_eight_divides_Q_largest_divisor_eight_largest_divisor_l885_88538

/-- The product of three consecutive positive even integers -/
def Q (n : ℕ) : ℕ := (2*n) * (2*n + 2) * (2*n + 4)

/-- 8 divides Q for all positive n -/
theorem eight_divides_Q (n : ℕ) : (8 : ℕ) ∣ Q n := by sorry

/-- For any d > 8, there exists an n such that d does not divide Q n -/
theorem largest_divisor (d : ℕ) (h : d > 8) : ∃ n : ℕ, ¬(d ∣ Q n) := by sorry

/-- 8 is the largest integer that divides Q for all positive n -/
theorem eight_largest_divisor : ∀ d : ℕ, (∀ n : ℕ, d ∣ Q n) → d ≤ 8 := by sorry

end NUMINAMATH_CALUDE_eight_divides_Q_largest_divisor_eight_largest_divisor_l885_88538


namespace NUMINAMATH_CALUDE_solution_to_system_l885_88541

/-- Prove that (4, 2, 3) is the solution to the given system of equations --/
theorem solution_to_system : ∃ (x y z : ℝ), x > 0 ∧ y > 0 ∧ z > 0 ∧
  x^2 + y^2 + Real.sqrt 3 * x * y = 20 + 8 * Real.sqrt 3 ∧
  y^2 + z^2 = 13 ∧
  z^2 + x^2 + x * z = 37 ∧
  x = 4 ∧ y = 2 ∧ z = 3 := by
  sorry

end NUMINAMATH_CALUDE_solution_to_system_l885_88541


namespace NUMINAMATH_CALUDE_hyperbola_foci_distance_l885_88543

-- Define the asymptotes
def asymptote1 (x : ℝ) : ℝ := 2 * x + 3
def asymptote2 (x : ℝ) : ℝ := -2 * x + 1

-- Define the point the hyperbola passes through
def point : ℝ × ℝ := (5, 7)

-- Define the hyperbola (implicitly)
def is_on_hyperbola (x y : ℝ) : Prop :=
  ∃ (a b : ℝ), a > 0 ∧ b > 0 ∧
    ((y - 2)^2 / a^2) - ((x + 1/2)^2 / b^2) = 1

-- Theorem statement
theorem hyperbola_foci_distance :
  ∃ (f1 f2 : ℝ × ℝ),
    is_on_hyperbola point.1 point.2 ∧
    ‖f1 - f2‖ = 15 := by
  sorry

end NUMINAMATH_CALUDE_hyperbola_foci_distance_l885_88543


namespace NUMINAMATH_CALUDE_yeongju_shortest_wire_l885_88552

-- Define the wire lengths in centimeters
def suzy_length : ℝ := 9.8
def yeongju_length : ℝ := 8.9
def youngho_length : ℝ := 9.3

-- Define the conversion factor from cm to mm
def cm_to_mm : ℝ := 10

-- Theorem to prove Yeongju has the shortest wire
theorem yeongju_shortest_wire :
  let suzy_mm := suzy_length * cm_to_mm
  let yeongju_mm := yeongju_length * cm_to_mm
  let youngho_mm := youngho_length * cm_to_mm
  yeongju_mm < suzy_mm ∧ yeongju_mm < youngho_mm :=
by sorry

end NUMINAMATH_CALUDE_yeongju_shortest_wire_l885_88552


namespace NUMINAMATH_CALUDE_geometric_sequence_a6_l885_88540

def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = a n * r

theorem geometric_sequence_a6 (a : ℕ → ℝ) :
  geometric_sequence a → a 2 = 4 → a 4 = 2 → a 6 = 1 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_a6_l885_88540


namespace NUMINAMATH_CALUDE_regular_polygon_exterior_angle_l885_88548

theorem regular_polygon_exterior_angle (n : ℕ) (n_pos : 0 < n) :
  (360 : ℝ) / n = 36 → n = 10 := by
  sorry

end NUMINAMATH_CALUDE_regular_polygon_exterior_angle_l885_88548


namespace NUMINAMATH_CALUDE_geometric_sequence_product_l885_88512

/-- A geometric sequence -/
def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, ∀ n : ℕ, a (n + 1) = a n * q

theorem geometric_sequence_product (a : ℕ → ℝ) :
  geometric_sequence a →
  a 1 * a 3 * a 11 = 8 →
  a 2 * a 8 = 4 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_product_l885_88512


namespace NUMINAMATH_CALUDE_dot_movement_l885_88592

-- Define a square
structure Square where
  side : ℝ
  center : ℝ × ℝ

-- Define a point on the square
structure Point where
  x : ℝ
  y : ℝ

-- Define the operations
def fold_diagonal (s : Square) (p : Point) : Point :=
  sorry

def rotate_90_clockwise (s : Square) (p : Point) : Point :=
  sorry

def unfold (s : Square) (p : Point) : Point :=
  sorry

-- Define the initial and final positions
def top_right (s : Square) : Point :=
  sorry

def top_center (s : Square) : Point :=
  sorry

-- Theorem statement
theorem dot_movement (s : Square) :
  let initial_pos := top_right s
  let folded_pos := fold_diagonal s initial_pos
  let rotated_pos := rotate_90_clockwise s folded_pos
  let final_pos := unfold s rotated_pos
  final_pos = top_center s :=
sorry

end NUMINAMATH_CALUDE_dot_movement_l885_88592


namespace NUMINAMATH_CALUDE_fred_has_five_balloons_l885_88570

/-- The number of yellow balloons Fred has -/
def fred_balloons (total sam mary : ℕ) : ℕ := total - (sam + mary)

/-- Theorem: Fred has 5 yellow balloons -/
theorem fred_has_five_balloons (total sam mary : ℕ) 
  (h_total : total = 18) 
  (h_sam : sam = 6) 
  (h_mary : mary = 7) : 
  fred_balloons total sam mary = 5 := by
  sorry

end NUMINAMATH_CALUDE_fred_has_five_balloons_l885_88570


namespace NUMINAMATH_CALUDE_repeating_decimal_difference_l885_88504

/-- Proves that the difference between the repeating decimals 0.353535... and 0.777777... is equal to -14/33 -/
theorem repeating_decimal_difference : 
  (35 : ℚ) / 99 - (7 : ℚ) / 9 = -14 / 33 := by sorry

end NUMINAMATH_CALUDE_repeating_decimal_difference_l885_88504


namespace NUMINAMATH_CALUDE_solve_system_l885_88506

theorem solve_system (x y : ℝ) 
  (eq1 : 3 * x - y = 7) 
  (eq2 : x + 3 * y = 6) : 
  x = 2.7 := by
  sorry

end NUMINAMATH_CALUDE_solve_system_l885_88506


namespace NUMINAMATH_CALUDE_min_reciprocal_sum_min_value_is_two_l885_88544

theorem min_reciprocal_sum (a b : ℝ) (h1 : b > 0) (h2 : a + b = 2) :
  ∀ x y : ℝ, x > 0 ∧ y > 0 ∧ x + y = 2 → 1/a + 1/b ≤ 1/x + 1/y :=
by sorry

theorem min_value_is_two (a b : ℝ) (h1 : b > 0) (h2 : a + b = 2) :
  1/a + 1/b = 2 :=
by sorry

end NUMINAMATH_CALUDE_min_reciprocal_sum_min_value_is_two_l885_88544


namespace NUMINAMATH_CALUDE_total_profit_is_89_10_l885_88563

def base_price : ℚ := 12
def day1_sales : ℕ := 3
def day2_sales : ℕ := 4
def day3_sales : ℕ := 5
def day1_cost : ℚ := 4
def day2_cost : ℚ := 5
def day3_cost : ℚ := 2
def extra_money : ℚ := 7
def day3_discount : ℚ := 2
def sales_tax_rate : ℚ := 1/10

def day1_profit : ℚ := (day1_sales * base_price + extra_money - day1_sales * day1_cost) * (1 - sales_tax_rate)
def day2_profit : ℚ := (day2_sales * base_price - day2_sales * day2_cost) * (1 - sales_tax_rate)
def day3_profit : ℚ := (day3_sales * (base_price - day3_discount) - day3_sales * day3_cost) * (1 - sales_tax_rate)

theorem total_profit_is_89_10 : 
  day1_profit + day2_profit + day3_profit = 89.1 := by
  sorry

end NUMINAMATH_CALUDE_total_profit_is_89_10_l885_88563


namespace NUMINAMATH_CALUDE_husband_catches_up_l885_88569

/-- Yolanda's bike speed in miles per hour -/
def yolanda_speed : ℝ := 20

/-- Yolanda's husband's car speed in miles per hour -/
def husband_speed : ℝ := 40

/-- Time difference between Yolanda and her husband's departure in minutes -/
def time_difference : ℝ := 15

/-- The time it takes for Yolanda's husband to catch up to her in minutes -/
def catch_up_time : ℝ := 15

theorem husband_catches_up :
  yolanda_speed * (catch_up_time + time_difference) / 60 = husband_speed * catch_up_time / 60 :=
sorry

end NUMINAMATH_CALUDE_husband_catches_up_l885_88569


namespace NUMINAMATH_CALUDE_modified_cube_surface_area_l885_88578

/-- Represents a cube with given dimensions -/
structure Cube where
  size : ℕ
  deriving Repr

/-- Represents the modified cube structure after tunneling -/
structure ModifiedCube where
  original : Cube
  smallCubeSize : ℕ
  removedCenters : ℕ
  deriving Repr

/-- Calculates the surface area of the modified cube structure -/
def surfaceArea (mc : ModifiedCube) : ℕ :=
  sorry

/-- Theorem stating the surface area of the specific modified cube structure -/
theorem modified_cube_surface_area :
  let originalCube : Cube := { size := 12 }
  let modifiedCube : ModifiedCube := {
    original := originalCube,
    smallCubeSize := 2,
    removedCenters := 6
  }
  surfaceArea modifiedCube = 1824 := by
  sorry

end NUMINAMATH_CALUDE_modified_cube_surface_area_l885_88578


namespace NUMINAMATH_CALUDE_smallest_k_for_inequality_l885_88522

theorem smallest_k_for_inequality : 
  ∃ (k : ℕ), k > 0 ∧ 
  (∀ (x : ℝ) (n : ℕ), x ∈ Set.Icc 0 1 → n > 0 → x^k * (1-x)^n < 1 / (1+n:ℝ)^3) ∧
  (∀ (k' : ℕ), k' > 0 → k' < k → 
    ∃ (x : ℝ) (n : ℕ), x ∈ Set.Icc 0 1 ∧ n > 0 ∧ x^k' * (1-x)^n ≥ 1 / (1+n:ℝ)^3) ∧
  k = 4 := by
sorry

end NUMINAMATH_CALUDE_smallest_k_for_inequality_l885_88522


namespace NUMINAMATH_CALUDE_factor_sum_l885_88566

/-- If x^2 + 3x + 4 is a factor of x^4 + Px^2 + Q, then P + Q = 15 -/
theorem factor_sum (P Q : ℝ) : 
  (∃ b c : ℝ, (X^2 + 3*X + 4) * (X^2 + b*X + c) = X^4 + P*X^2 + Q) → 
  P + Q = 15 := by
sorry

end NUMINAMATH_CALUDE_factor_sum_l885_88566


namespace NUMINAMATH_CALUDE_min_b_value_l885_88532

/-- Given a parabola and a circle with specific intersection properties, 
    the minimum value of b is 2 -/
theorem min_b_value (k a b r : ℝ) : 
  k > 0 → 
  (∀ x y, y = k * x^2 → (x - a)^2 + (y - b)^2 = r^2 → 
    (x = 0 ∧ y = 0) ∨ y = k * x + b) →
  a^2 + b^2 = r^2 →
  (∃ x₁ x₂ x₃, x₁ ≠ x₂ ∧ x₁ ≠ x₃ ∧ x₂ ≠ x₃ ∧ 
    k * x₁^2 = (x₁ - a)^2 + (k * x₁^2 - b)^2 - r^2 ∧
    k * x₂^2 = (x₂ - a)^2 + (k * x₂^2 - b)^2 - r^2 ∧
    k * x₃^2 = (x₃ - a)^2 + (k * x₃^2 - b)^2 - r^2) →
  b ≥ 2 :=
by sorry

end NUMINAMATH_CALUDE_min_b_value_l885_88532


namespace NUMINAMATH_CALUDE_unique_solution_for_exponential_equation_l885_88534

theorem unique_solution_for_exponential_equation :
  ∀ a b p : ℕ+,
    p.val.Prime →
    2^(a.val) + p^(b.val) = 19^(a.val) →
    a = 1 ∧ b = 1 ∧ p = 17 :=
by sorry

end NUMINAMATH_CALUDE_unique_solution_for_exponential_equation_l885_88534


namespace NUMINAMATH_CALUDE_infinite_division_sum_equal_l885_88536

/-- Represents a shape with an area -/
class HasArea (α : Type*) where
  area : α → ℝ

/-- Represents a shape that can be divided -/
class Divisible (α : Type*) where
  divide : α → ℝ → α

variable (T : Type*) [HasArea T] [Divisible T]
variable (Q : Type*) [HasArea Q] [Divisible Q]

/-- The sum of areas after infinite divisions -/
noncomputable def infiniteDivisionSum (shape : T) (ratio : ℝ) : ℝ := sorry

/-- Theorem stating the equality of infinite division sums -/
theorem infinite_division_sum_equal
  (triangle : T)
  (quad : Q)
  (ratio : ℝ)
  (h : HasArea.area triangle = 1.5 * HasArea.area quad) :
  infiniteDivisionSum T triangle ratio = infiniteDivisionSum Q quad ratio := by
  sorry

end NUMINAMATH_CALUDE_infinite_division_sum_equal_l885_88536


namespace NUMINAMATH_CALUDE_equation_solution_l885_88505

theorem equation_solution (x y z w : ℝ) (h : x ≠ 0 ∧ y ≠ 0 ∧ z ≠ 0) 
  (eq : 1/x + 1/y = 1/z + w) : 
  z = x*y / (x + y - w*x*y) := by
sorry

end NUMINAMATH_CALUDE_equation_solution_l885_88505


namespace NUMINAMATH_CALUDE_smallest_days_to_triple_l885_88577

def borrowed_amount : ℝ := 20
def interest_rate : ℝ := 0.12

def amount_owed (days : ℕ) : ℝ :=
  borrowed_amount + borrowed_amount * interest_rate * days

def is_at_least_triple (days : ℕ) : Prop :=
  amount_owed days ≥ 3 * borrowed_amount

theorem smallest_days_to_triple : 
  (∀ d : ℕ, d < 17 → ¬(is_at_least_triple d)) ∧ 
  (is_at_least_triple 17) :=
sorry

end NUMINAMATH_CALUDE_smallest_days_to_triple_l885_88577


namespace NUMINAMATH_CALUDE_turtle_fraction_l885_88500

theorem turtle_fraction (trey kris kristen : ℕ) : 
  trey = 7 * kris →
  trey = kristen + 9 →
  kristen = 12 →
  kris / kristen = 1 / 4 := by
sorry

end NUMINAMATH_CALUDE_turtle_fraction_l885_88500


namespace NUMINAMATH_CALUDE_cotangent_half_angle_identity_l885_88530

theorem cotangent_half_angle_identity (α : Real) (m : Real) :
  (Real.tan (α / 2))⁻¹ = m → (1 - Real.sin α) / Real.cos α = (m - 1) / (m + 1) := by
  sorry

end NUMINAMATH_CALUDE_cotangent_half_angle_identity_l885_88530


namespace NUMINAMATH_CALUDE_stacy_paper_pages_per_day_l885_88560

/-- Given a paper with a certain number of pages due in a certain number of days,
    calculate the number of pages that need to be written per day to finish on time. -/
def pages_per_day (total_pages : ℕ) (days : ℕ) : ℚ :=
  total_pages / days

/-- Theorem stating that for a 100-page paper due in 5 days,
    the number of pages to be written per day is 20. -/
theorem stacy_paper_pages_per_day :
  pages_per_day 100 5 = 20 := by
  sorry

end NUMINAMATH_CALUDE_stacy_paper_pages_per_day_l885_88560


namespace NUMINAMATH_CALUDE_marble_selection_l885_88539

theorem marble_selection (n m k b : ℕ) (h1 : n = 10) (h2 : m = 2) (h3 : k = 4) (h4 : b = 2) :
  (Nat.choose n k) - (Nat.choose (n - m) k) = 140 := by
  sorry

end NUMINAMATH_CALUDE_marble_selection_l885_88539


namespace NUMINAMATH_CALUDE_bug_crawl_distance_l885_88597

/-- The minimum distance a bug must crawl on the surface of a right circular cone --/
theorem bug_crawl_distance (r h a b θ : ℝ) (hr : r = 500) (hh : h = 300) 
  (ha : a = 100) (hb : b = 400) (hθ : θ = π / 2) : 
  let d := Real.sqrt ((b * Real.cos θ - a)^2 + (b * Real.sin θ)^2)
  d = Real.sqrt 170000 := by
sorry

end NUMINAMATH_CALUDE_bug_crawl_distance_l885_88597


namespace NUMINAMATH_CALUDE_beads_taken_out_l885_88527

theorem beads_taken_out (green brown red left : ℕ) : 
  green = 1 → brown = 2 → red = 3 → left = 4 → 
  (green + brown + red) - left = 2 := by
  sorry

end NUMINAMATH_CALUDE_beads_taken_out_l885_88527


namespace NUMINAMATH_CALUDE_basketball_team_sales_l885_88562

/-- The number of cupcakes sold -/
def cupcakes : ℕ := 50

/-- The price of each cupcake in dollars -/
def cupcake_price : ℚ := 2

/-- The price of each cookie in dollars -/
def cookie_price : ℚ := 1/2

/-- The number of basketballs bought -/
def basketballs : ℕ := 2

/-- The price of each basketball in dollars -/
def basketball_price : ℚ := 40

/-- The number of energy drinks bought -/
def energy_drinks : ℕ := 20

/-- The price of each energy drink in dollars -/
def energy_drink_price : ℚ := 2

/-- The number of cookies sold -/
def cookies_sold : ℕ := 40

theorem basketball_team_sales :
  cookies_sold * cookie_price = 
    basketballs * basketball_price + 
    energy_drinks * energy_drink_price - 
    cupcakes * cupcake_price :=
by sorry

end NUMINAMATH_CALUDE_basketball_team_sales_l885_88562


namespace NUMINAMATH_CALUDE_edge_sum_is_112_l885_88591

/-- A rectangular solid with specific properties -/
structure RectangularSolid where
  x : ℝ
  d : ℝ
  volume_eq : x^3 * (d + 1)^3 = 512
  surface_area_eq : 2 * (x^2 * (d + 1) + x^2 * (d + 1)^2 + x^2 * (d + 1)^3) = 448

/-- The sum of the lengths of all edges of the rectangular solid -/
def edge_sum (solid : RectangularSolid) : ℝ :=
  4 * (solid.x + solid.x * (solid.d + 1) + solid.x * (solid.d + 1)^2)

/-- Theorem stating that the sum of the lengths of all edges is 112 -/
theorem edge_sum_is_112 (solid : RectangularSolid) : edge_sum solid = 112 := by
  sorry

#check edge_sum_is_112

end NUMINAMATH_CALUDE_edge_sum_is_112_l885_88591


namespace NUMINAMATH_CALUDE_game_playing_time_l885_88528

theorem game_playing_time (num_children : ℕ) (game_duration : ℕ) (players_at_once : ℕ) :
  num_children = 8 →
  game_duration = 120 →  -- 2 hours in minutes
  players_at_once = 2 →
  (game_duration * players_at_once) % num_children = 0 →
  (game_duration * players_at_once) / num_children = 30 :=
by sorry

end NUMINAMATH_CALUDE_game_playing_time_l885_88528


namespace NUMINAMATH_CALUDE_parabola_point_distance_to_origin_l885_88589

theorem parabola_point_distance_to_origin :
  ∀ (x y : ℝ),
  y^2 = 2*x →  -- Point A is on the parabola y^2 = 2x
  (x + 1/2) / |y| = 5/4 →  -- Ratio condition
  ((x - 1/2)^2 + y^2)^(1/2) > 2 →  -- |AF| > 2
  (x^2 + y^2)^(1/2) = 2 * (2^(1/2)) :=
by
  sorry

end NUMINAMATH_CALUDE_parabola_point_distance_to_origin_l885_88589


namespace NUMINAMATH_CALUDE_fence_cost_l885_88526

theorem fence_cost (area : ℝ) (price_per_foot : ℝ) (h1 : area = 289) (h2 : price_per_foot = 59) :
  4 * Real.sqrt area * price_per_foot = 4012 := by
  sorry

end NUMINAMATH_CALUDE_fence_cost_l885_88526


namespace NUMINAMATH_CALUDE_exists_number_with_specific_digit_sum_l885_88555

/-- Sum of digits of a natural number -/
def sumOfDigits (n : ℕ) : ℕ := sorry

/-- Theorem stating the existence of a number with specific digit sum properties -/
theorem exists_number_with_specific_digit_sum : 
  ∃ m : ℕ, sumOfDigits m = 1990 ∧ sumOfDigits (m^2) = 1990^2 := by sorry

end NUMINAMATH_CALUDE_exists_number_with_specific_digit_sum_l885_88555


namespace NUMINAMATH_CALUDE_additional_oil_amount_l885_88511

-- Define the original price, reduced price, and additional amount
def original_price : ℝ := 42.75
def reduced_price : ℝ := 34.2
def additional_amount : ℝ := 684

-- Define the price reduction percentage
def price_reduction : ℝ := 0.2

-- Theorem statement
theorem additional_oil_amount :
  reduced_price = original_price * (1 - price_reduction) →
  additional_amount / reduced_price = 20 := by
sorry

end NUMINAMATH_CALUDE_additional_oil_amount_l885_88511


namespace NUMINAMATH_CALUDE_upper_limit_of_b_l885_88556

theorem upper_limit_of_b (a b : ℤ) (h1 : 6 < a) (h2 : a < 17) (h3 : 3 < b) 
  (h4 : (a : ℚ) / b ≤ 3.75) (h5 : 3.75 ≤ (a : ℚ) / b) : b ≤ 4 := by
  sorry

end NUMINAMATH_CALUDE_upper_limit_of_b_l885_88556


namespace NUMINAMATH_CALUDE_cookie_cost_difference_l885_88571

theorem cookie_cost_difference (cookie_cost diane_money : ℕ) 
  (h1 : cookie_cost = 65)
  (h2 : diane_money = 27) :
  cookie_cost - diane_money = 38 := by
  sorry

end NUMINAMATH_CALUDE_cookie_cost_difference_l885_88571


namespace NUMINAMATH_CALUDE_probability_multiple_6_or_8_l885_88587

def is_multiple_of_6_or_8 (n : ℕ) : Bool :=
  n % 6 = 0 || n % 8 = 0

def count_multiples (n : ℕ) : ℕ :=
  (List.range n).filter is_multiple_of_6_or_8 |>.length

theorem probability_multiple_6_or_8 :
  count_multiples 100 / 100 = 6 / 25 := by
  sorry

end NUMINAMATH_CALUDE_probability_multiple_6_or_8_l885_88587


namespace NUMINAMATH_CALUDE_triangle_angle_sum_l885_88594

theorem triangle_angle_sum (first_angle second_angle third_angle : ℝ) : 
  second_angle = 2 * first_angle →
  third_angle = 15 →
  first_angle = third_angle + 40 →
  first_angle + second_angle = 165 := by
sorry

end NUMINAMATH_CALUDE_triangle_angle_sum_l885_88594


namespace NUMINAMATH_CALUDE_inequality_proof_l885_88593

theorem inequality_proof (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (h_abc : a * b * c = 1) :
  (a - 1 + 1/b) * (b - 1 + 1/c) * (c - 1 + 1/a) ≤ 1 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l885_88593


namespace NUMINAMATH_CALUDE_hiking_team_gloves_l885_88508

/-- The minimum number of gloves needed for a hiking team -/
theorem hiking_team_gloves (participants : ℕ) (gloves_per_pair : ℕ) : 
  participants = 43 → gloves_per_pair = 2 → participants * gloves_per_pair = 86 := by
  sorry

end NUMINAMATH_CALUDE_hiking_team_gloves_l885_88508


namespace NUMINAMATH_CALUDE_last_two_digits_of_nine_to_h_l885_88550

def a : ℕ := 1
def b : ℕ := 2^a
def c : ℕ := 3^b
def d : ℕ := 4^c
def e : ℕ := 5^d
def f : ℕ := 6^e
def g : ℕ := 7^f
def h : ℕ := 8^g

theorem last_two_digits_of_nine_to_h (a b c d e f g h : ℕ) 
  (ha : a = 1)
  (hb : b = 2^a)
  (hc : c = 3^b)
  (hd : d = 4^c)
  (he : e = 5^d)
  (hf : f = 6^e)
  (hg : g = 7^f)
  (hh : h = 8^g) :
  9^h % 100 = 21 := by
  sorry

end NUMINAMATH_CALUDE_last_two_digits_of_nine_to_h_l885_88550


namespace NUMINAMATH_CALUDE_restaurant_outdoor_section_area_l885_88588

/-- The area of a rectangular section with width 7 feet and length 5 feet is 35 square feet. -/
theorem restaurant_outdoor_section_area :
  let width : ℝ := 7
  let length : ℝ := 5
  width * length = 35 := by sorry

end NUMINAMATH_CALUDE_restaurant_outdoor_section_area_l885_88588


namespace NUMINAMATH_CALUDE_f_even_iff_a_zero_l885_88535

/-- A function f: ℝ → ℝ is even if f(-x) = f(x) for all x ∈ ℝ -/
def IsEven (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = f x

/-- The quadratic function f(x) = x^2 + ax + 1 -/
def f (a : ℝ) : ℝ → ℝ := fun x ↦ x^2 + a*x + 1

/-- Theorem: f is an even function if and only if a = 0 -/
theorem f_even_iff_a_zero (a : ℝ) :
  IsEven (f a) ↔ a = 0 := by
  sorry

end NUMINAMATH_CALUDE_f_even_iff_a_zero_l885_88535


namespace NUMINAMATH_CALUDE_moles_of_CH3COOH_l885_88557

-- Define the chemical reaction
structure Reaction where
  reactant1 : ℝ  -- moles of CH3COOH
  reactant2 : ℝ  -- moles of NaOH
  product1  : ℝ  -- moles of NaCH3COO
  product2  : ℝ  -- moles of H2O

-- Define the balanced equation
def balanced_equation (r : Reaction) : Prop :=
  r.reactant1 = r.reactant2 ∧ r.reactant1 = r.product1 ∧ r.reactant1 = r.product2

-- Theorem statement
theorem moles_of_CH3COOH (r : Reaction) 
  (h1 : r.reactant2 = 1)  -- 1 mole of NaOH is used
  (h2 : r.product1 = 1)   -- 1 mole of NaCH3COO is formed
  (h3 : balanced_equation r)  -- The reaction follows the balanced equation
  : r.reactant1 = 1 :=  -- The number of moles of CH3COOH combined is 1
by sorry

end NUMINAMATH_CALUDE_moles_of_CH3COOH_l885_88557


namespace NUMINAMATH_CALUDE_locus_of_point_P_l885_88567

/-- The locus of points P(x, y) such that the product of slopes of AP and BP is -1/4,
    where A(-2, 0) and B(2, 0) are fixed points. -/
theorem locus_of_point_P (x y : ℝ) (h : x ≠ 2 ∧ x ≠ -2) : 
  (y / (x + 2)) * (y / (x - 2)) = -1/4 ↔ x^2 / 4 + y^2 = 1 :=
sorry

end NUMINAMATH_CALUDE_locus_of_point_P_l885_88567


namespace NUMINAMATH_CALUDE_smallest_x_for_fifth_power_l885_88513

theorem smallest_x_for_fifth_power (x : ℕ) (K : ℤ) : 
  (x = 135000 ∧ 
   180 * x = K^5 ∧ 
   ∀ y : ℕ, y < x → ¬∃ L : ℤ, 180 * y = L^5) :=
sorry

end NUMINAMATH_CALUDE_smallest_x_for_fifth_power_l885_88513


namespace NUMINAMATH_CALUDE_g_8_l885_88583

/-- A function g: ℝ → ℝ satisfying the given functional equation for all real x and y -/
def g : ℝ → ℝ :=
  fun x => sorry

/-- The functional equation that g satisfies for all real x and y -/
axiom g_equation (x y : ℝ) : g x + 2 * g (x + 2 * y) + 3 * x * y = g (4 * x - y) + 3 * x^2 + 2

/-- Theorem stating the value of g(8) -/
theorem g_8 : g 8 = -3/2 * 64 + 2 := by
  sorry

end NUMINAMATH_CALUDE_g_8_l885_88583


namespace NUMINAMATH_CALUDE_triangle_abc_properties_l885_88581

/-- Given a triangle ABC with sides a, b, c opposite to angles A, B, C respectively,
    prove that if a = 2c cos A and √5 sin A = 1, then sin C = 1/4 and b/c = (2√5 + 5√3) / 5 -/
theorem triangle_abc_properties (a b c A B C : ℝ) 
    (h1 : a = 2 * c * Real.cos A)
    (h2 : Real.sqrt 5 * Real.sin A = 1) :
    Real.sin C = 1/4 ∧ b/c = (2 * Real.sqrt 5 + 5 * Real.sqrt 3) / 5 := by
  sorry

end NUMINAMATH_CALUDE_triangle_abc_properties_l885_88581


namespace NUMINAMATH_CALUDE_regular_polygon_with_40_degree_exterior_angle_has_9_sides_l885_88549

/-- A regular polygon with an exterior angle of 40° has 9 sides. -/
theorem regular_polygon_with_40_degree_exterior_angle_has_9_sides :
  ∀ (n : ℕ), n > 0 →
  (360 : ℝ) / n = 40 →
  n = 9 := by
  sorry

end NUMINAMATH_CALUDE_regular_polygon_with_40_degree_exterior_angle_has_9_sides_l885_88549
