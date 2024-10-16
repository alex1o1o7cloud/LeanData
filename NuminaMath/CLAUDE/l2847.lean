import Mathlib

namespace NUMINAMATH_CALUDE_no_solution_in_range_l2847_284739

theorem no_solution_in_range (x y : ℕ+) (h : 3 * x^2 + x = 4 * y^2 + y) :
  x - y ≠ 2013 ∧ x - y ≠ 2014 ∧ x - y ≠ 2015 ∧ x - y ≠ 2016 := by
  sorry

end NUMINAMATH_CALUDE_no_solution_in_range_l2847_284739


namespace NUMINAMATH_CALUDE_median_and_mode_are_23_l2847_284746

/-- Represents the shoe size distribution of a class --/
structure ShoeSizeDistribution where
  sizes : List Nat
  frequencies : List Nat
  total_students : Nat

/-- Calculates the median of a shoe size distribution --/
def median (dist : ShoeSizeDistribution) : Nat :=
  sorry

/-- Calculates the mode of a shoe size distribution --/
def mode (dist : ShoeSizeDistribution) : Nat :=
  sorry

/-- The shoe size distribution for the given class --/
def class_distribution : ShoeSizeDistribution :=
  { sizes := [20, 21, 22, 23, 24],
    frequencies := [2, 8, 9, 19, 2],
    total_students := 40 }

theorem median_and_mode_are_23 :
  median class_distribution = 23 ∧ mode class_distribution = 23 := by
  sorry

end NUMINAMATH_CALUDE_median_and_mode_are_23_l2847_284746


namespace NUMINAMATH_CALUDE_rectangle_area_difference_l2847_284733

theorem rectangle_area_difference (A B a b : ℕ) 
  (h1 : A = 20) (h2 : B = 30) (h3 : a = 4) (h4 : b = 7) : 
  (A * B - a * b) - ((A - a) * B + A * (B - b)) = 20 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_area_difference_l2847_284733


namespace NUMINAMATH_CALUDE_specific_semicircle_chord_product_l2847_284781

/-- A structure representing a semicircle with equally spaced points -/
structure SemicircleWithPoints where
  radius : ℝ
  num_points : ℕ

/-- The product of chord lengths in a semicircle with equally spaced points -/
def chord_product (s : SemicircleWithPoints) : ℝ :=
  sorry

/-- Theorem stating the product of chord lengths for a specific semicircle configuration -/
theorem specific_semicircle_chord_product :
  let s : SemicircleWithPoints := { radius := 4, num_points := 8 }
  chord_product s = 4718592 := by
  sorry

end NUMINAMATH_CALUDE_specific_semicircle_chord_product_l2847_284781


namespace NUMINAMATH_CALUDE_parallelogram_sum_l2847_284757

/-- A parallelogram with side lengths 12, 3x+6, 10y-2, and 15 units consecutively -/
structure Parallelogram (x y : ℝ) :=
  (side1 : ℝ := 12)
  (side2 : ℝ := 3*x + 6)
  (side3 : ℝ := 10*y - 2)
  (side4 : ℝ := 15)
  (opposite_sides_equal1 : side1 = side3)
  (opposite_sides_equal2 : side2 = side4)

/-- The sum of x and y in the parallelogram is 4.4 -/
theorem parallelogram_sum (x y : ℝ) (p : Parallelogram x y) : x + y = 4.4 := by
  sorry

end NUMINAMATH_CALUDE_parallelogram_sum_l2847_284757


namespace NUMINAMATH_CALUDE_expression_evaluation_l2847_284758

theorem expression_evaluation :
  (12 - 11 + 10 - 9 + 8 - 7 + 6 - 5 + 4 - 3 + 2 - 1) / 
  (2 - 3 + 4 - 5 + 6 - 7 + 8 - 9 + 10 - 11 + 12) = 6 / 7 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l2847_284758


namespace NUMINAMATH_CALUDE_ratio_problem_l2847_284732

theorem ratio_problem (x y : ℝ) (h : (3*x - 2*y) / (2*x + y) = 5/4) : y / x = 2/13 := by
  sorry

end NUMINAMATH_CALUDE_ratio_problem_l2847_284732


namespace NUMINAMATH_CALUDE_system_solution_l2847_284718

/-- The system of linear equations -/
def system (x y : ℝ) : Prop :=
  x + y = 6 ∧ x = 2*y

/-- The solution set of the system -/
def solution_set : Set (ℝ × ℝ) :=
  {(4, 2)}

/-- Theorem stating that the solution set is correct -/
theorem system_solution : 
  {(x, y) | system x y} = solution_set :=
sorry

end NUMINAMATH_CALUDE_system_solution_l2847_284718


namespace NUMINAMATH_CALUDE_trapezoid_perimeter_is_46_l2847_284722

/-- A trapezoid within a rectangle -/
structure TrapezoidInRectangle where
  a : ℝ  -- Length of longer parallel side of trapezoid
  b : ℝ  -- Length of shorter parallel side of trapezoid
  h : ℝ  -- Height of trapezoid (equal to non-parallel sides)
  rect_perimeter : ℝ  -- Perimeter of the rectangle

/-- The perimeter of the trapezoid -/
def trapezoid_perimeter (t : TrapezoidInRectangle) : ℝ :=
  t.a + t.b + 2 * t.h

/-- Theorem stating the perimeter of the trapezoid is 46 meters -/
theorem trapezoid_perimeter_is_46 (t : TrapezoidInRectangle)
  (h1 : t.a = 15)
  (h2 : t.b = 9)
  (h3 : t.rect_perimeter = 52)
  (h4 : t.h = (t.rect_perimeter - 2 * t.a) / 2) :
  trapezoid_perimeter t = 46 := by
  sorry

end NUMINAMATH_CALUDE_trapezoid_perimeter_is_46_l2847_284722


namespace NUMINAMATH_CALUDE_provisions_duration_l2847_284725

/-- Given provisions for a certain number of boys and days, calculate how long the provisions will last with additional boys. -/
theorem provisions_duration (initial_boys : ℕ) (initial_days : ℕ) (additional_boys : ℕ) :
  let total_boys := initial_boys + additional_boys
  let new_days := (initial_boys * initial_days) / total_boys
  initial_boys = 1500 → initial_days = 25 → additional_boys = 350 →
  ⌊(new_days : ℚ)⌋ = 20 := by
  sorry

end NUMINAMATH_CALUDE_provisions_duration_l2847_284725


namespace NUMINAMATH_CALUDE_no_prime_satisfies_condition_l2847_284755

theorem no_prime_satisfies_condition : ¬ ∃ p : ℕ, Nat.Prime p ∧ (10 : ℝ) * p = p + 5.4 := by
  sorry

end NUMINAMATH_CALUDE_no_prime_satisfies_condition_l2847_284755


namespace NUMINAMATH_CALUDE_complex_product_real_condition_l2847_284734

theorem complex_product_real_condition (a b c d : ℝ) :
  (Complex.I * b + a) * (Complex.I * d + c) ∈ Set.range Complex.ofReal ↔ a * d + b * c = 0 := by
  sorry

end NUMINAMATH_CALUDE_complex_product_real_condition_l2847_284734


namespace NUMINAMATH_CALUDE_equation_equivalent_to_lines_l2847_284741

-- Define the original equation
def original_equation (x y : ℝ) : Prop :=
  2 * x^2 + y^2 + 3 * x * y + 3 * x + y = 2

-- Define the two lines
def line1 (x y : ℝ) : Prop := y = -x - 2
def line2 (x y : ℝ) : Prop := y = -2 * x + 1

-- Theorem statement
theorem equation_equivalent_to_lines :
  ∀ x y : ℝ, original_equation x y ↔ (line1 x y ∨ line2 x y) :=
sorry

end NUMINAMATH_CALUDE_equation_equivalent_to_lines_l2847_284741


namespace NUMINAMATH_CALUDE_geometric_sequence_property_l2847_284736

/-- Theorem: In a geometric sequence where a₅ = 4 and a₇ = 6, a₉ = 9 -/
theorem geometric_sequence_property (a : ℕ → ℝ) :
  (∀ n m : ℕ, a (n + m) = a n * a m) →  -- geometric sequence property
  a 5 = 4 →
  a 7 = 6 →
  a 9 = 9 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_property_l2847_284736


namespace NUMINAMATH_CALUDE_cosine_equality_l2847_284794

theorem cosine_equality (n : ℤ) : 
  100 ≤ n ∧ n ≤ 280 ∧ Real.cos (n * π / 180) = Real.cos (317 * π / 180) → n = 317 :=
by sorry

end NUMINAMATH_CALUDE_cosine_equality_l2847_284794


namespace NUMINAMATH_CALUDE_shaded_area_percentage_l2847_284750

theorem shaded_area_percentage (total_squares : Nat) (shaded_squares : Nat) :
  total_squares = 16 →
  shaded_squares = 8 →
  (shaded_squares : ℚ) / (total_squares : ℚ) * 100 = 50 := by
  sorry

end NUMINAMATH_CALUDE_shaded_area_percentage_l2847_284750


namespace NUMINAMATH_CALUDE_equilateral_triangle_x_value_l2847_284738

/-- An equilateral triangle with side lengths expressed in terms of x -/
structure EquilateralTriangle where
  x : ℝ
  side_length : ℝ
  eq_sides : side_length = 4 * x ∧ side_length = x + 12

theorem equilateral_triangle_x_value (t : EquilateralTriangle) : t.x = 4 := by
  sorry

end NUMINAMATH_CALUDE_equilateral_triangle_x_value_l2847_284738


namespace NUMINAMATH_CALUDE_product_expansion_l2847_284780

theorem product_expansion (x : ℝ) : (x^2 - 3*x + 3) * (x^2 + 3*x + 3) = x^4 - 3*x^2 + 9 := by
  sorry

end NUMINAMATH_CALUDE_product_expansion_l2847_284780


namespace NUMINAMATH_CALUDE_same_grade_probability_l2847_284767

theorem same_grade_probability (total : ℕ) (first : ℕ) (second : ℕ) (third : ℕ) 
  (h_total : total = 10)
  (h_first : first = 4)
  (h_second : second = 3)
  (h_third : third = 3)
  (h_sum : first + second + third = total) :
  (Nat.choose first 2 + Nat.choose second 2 + Nat.choose third 2) / Nat.choose total 2 = 4 / 15 := by
sorry

end NUMINAMATH_CALUDE_same_grade_probability_l2847_284767


namespace NUMINAMATH_CALUDE_projectile_meeting_time_l2847_284761

theorem projectile_meeting_time (initial_distance : ℝ) (speed1 speed2 : ℝ) :
  initial_distance = 1182 →
  speed1 = 460 →
  speed2 = 525 →
  (initial_distance / (speed1 + speed2)) * 60 = 72 := by
  sorry

end NUMINAMATH_CALUDE_projectile_meeting_time_l2847_284761


namespace NUMINAMATH_CALUDE_perpendicular_lines_planes_l2847_284784

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the perpendicular relation for lines and planes
variable (perp_line_plane : Line → Plane → Prop)
variable (perp_line_line : Line → Line → Prop)
variable (perp_plane_plane : Plane → Plane → Prop)

-- State the theorem
theorem perpendicular_lines_planes 
  (a b : Line) (α β : Plane) 
  (h_non_coincident : a ≠ b) 
  (h_a_perp_α : perp_line_plane a α) 
  (h_b_perp_β : perp_line_plane b β) : 
  (perp_line_line a b ↔ perp_plane_plane α β) :=
sorry

end NUMINAMATH_CALUDE_perpendicular_lines_planes_l2847_284784


namespace NUMINAMATH_CALUDE_largest_divisor_of_five_consecutive_integers_l2847_284792

theorem largest_divisor_of_five_consecutive_integers :
  ∀ n : ℤ, ∃ k : ℤ, k * 120 = (n * (n + 1) * (n + 2) * (n + 3) * (n + 4)) ∧
  ∀ m : ℤ, m > 120 → ∃ l : ℤ, l * m ≠ (l * (l + 1) * (l + 2) * (l + 3) * (l + 4)) :=
by sorry

end NUMINAMATH_CALUDE_largest_divisor_of_five_consecutive_integers_l2847_284792


namespace NUMINAMATH_CALUDE_quadratic_to_linear_solutions_l2847_284797

theorem quadratic_to_linear_solutions (x : ℝ) :
  x^2 - 2*x - 1 = 0 ∧ (x - 1 = Real.sqrt 2 ∨ x - 1 = -Real.sqrt 2) →
  (x - 1 = Real.sqrt 2 → x - 1 = -Real.sqrt 2) ∧
  (x - 1 = -Real.sqrt 2 → x - 1 = Real.sqrt 2) :=
sorry

end NUMINAMATH_CALUDE_quadratic_to_linear_solutions_l2847_284797


namespace NUMINAMATH_CALUDE_sunRiseOnlyCertainEvent_l2847_284717

-- Define the type for events
inductive Event
  | SunRise
  | OpenBook
  | Thumbtack
  | Student

-- Define a function to check if an event is certain
def isCertain (e : Event) : Prop :=
  match e with
  | Event.SunRise => true
  | _ => false

-- Theorem stating that SunRise is the only certain event
theorem sunRiseOnlyCertainEvent : 
  ∀ (e : Event), isCertain e ↔ e = Event.SunRise :=
by
  sorry


end NUMINAMATH_CALUDE_sunRiseOnlyCertainEvent_l2847_284717


namespace NUMINAMATH_CALUDE_domain_log_range_exp_intersection_empty_l2847_284771

-- Define the sets A and B
def A : Set ℝ := {x : ℝ | x < 0}
def B : Set ℝ := {y : ℝ | y > 0}

-- State the theorem
theorem domain_log_range_exp_intersection_empty : A ∩ B = ∅ := by
  sorry

end NUMINAMATH_CALUDE_domain_log_range_exp_intersection_empty_l2847_284771


namespace NUMINAMATH_CALUDE_smallest_k_for_power_inequality_l2847_284766

theorem smallest_k_for_power_inequality : ∃ k : ℕ, k = 14 ∧ 
  (∀ n : ℕ, n < k → (7 : ℝ)^n ≤ 4^19) ∧ (7 : ℝ)^k > 4^19 := by
  sorry

end NUMINAMATH_CALUDE_smallest_k_for_power_inequality_l2847_284766


namespace NUMINAMATH_CALUDE_problem_solution_l2847_284752

/-- Given constants a, b, and c satisfying the specified conditions, prove that a + 2b + 3c = 74 -/
theorem problem_solution (a b c : ℝ) 
  (h1 : ∀ x, (x - a) * (x - b) / (x - c) ≤ 0 ↔ (x < -6 ∨ |x - 30| ≤ 2))
  (h2 : a < b) : 
  a + 2 * b + 3 * c = 74 := by
sorry

end NUMINAMATH_CALUDE_problem_solution_l2847_284752


namespace NUMINAMATH_CALUDE_tenth_term_is_39_l2847_284773

/-- An arithmetic sequence with given properties -/
structure ArithmeticSequence where
  a : ℝ  -- First term
  d : ℝ  -- Common difference
  second_term : a + d = 7
  fifth_term : a + 4 * d = 19

/-- The tenth term of the arithmetic sequence is 39 -/
theorem tenth_term_is_39 (seq : ArithmeticSequence) : seq.a + 9 * seq.d = 39 := by
  sorry

end NUMINAMATH_CALUDE_tenth_term_is_39_l2847_284773


namespace NUMINAMATH_CALUDE_seating_arrangements_eq_48_num_seating_arrangements_l2847_284709

/- Define the number of teams -/
def num_teams : ℕ := 3

/- Define the number of athletes per team -/
def athletes_per_team : ℕ := 2

/- Define the total number of athletes -/
def total_athletes : ℕ := num_teams * athletes_per_team

/- Function to calculate the number of seating arrangements -/
def seating_arrangements : ℕ :=
  (Nat.factorial num_teams) * (Nat.factorial athletes_per_team)^num_teams

/- Theorem stating that the number of seating arrangements is 48 -/
theorem seating_arrangements_eq_48 :
  seating_arrangements = 48 := by
  sorry

/- Main theorem to prove -/
theorem num_seating_arrangements :
  ∀ (n m : ℕ), n = num_teams → m = athletes_per_team →
  (Nat.factorial n) * (Nat.factorial m)^n = 48 := by
  sorry

end NUMINAMATH_CALUDE_seating_arrangements_eq_48_num_seating_arrangements_l2847_284709


namespace NUMINAMATH_CALUDE_hyperbola_equation_l2847_284753

/-- Given a hyperbola with the equation x²/a² - y²/b² = 1 where a > 0 and b > 0,
    left and right foci F₁ and F₂, and a point P(3,4) on its asymptote,
    prove that if |PF₁ + PF₂| = |F₁F₂|, then the equation of the hyperbola is x²/9 - y²/16 = 1 -/
theorem hyperbola_equation (a b : ℝ) (ha : a > 0) (hb : b > 0)
  (F₁ F₂ : ℝ × ℝ) (hF : F₁.1 < F₂.1)
  (P : ℝ × ℝ) (hP : P = (3, 4))
  (h_asymptote : ∃ (k : ℝ), P.2 = k * P.1 ∧ k^2 * a^2 = b^2)
  (h_foci : |P - F₁ + (P - F₂)| = |F₂ - F₁|) :
  ∀ (x y : ℝ), x^2 / 9 - y^2 / 16 = 1 ↔ x^2 / a^2 - y^2 / b^2 = 1 :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_equation_l2847_284753


namespace NUMINAMATH_CALUDE_zero_neither_positive_nor_negative_l2847_284745

theorem zero_neither_positive_nor_negative :
  ¬(0 > 0) ∧ ¬(0 < 0) :=
by
  sorry

#check zero_neither_positive_nor_negative

end NUMINAMATH_CALUDE_zero_neither_positive_nor_negative_l2847_284745


namespace NUMINAMATH_CALUDE_cleaning_time_proof_l2847_284798

theorem cleaning_time_proof (total_time : ℝ) (lilly_fraction : ℝ) : 
  total_time = 8 → lilly_fraction = 1/4 → 
  (total_time - lilly_fraction * total_time) * 60 = 360 := by
  sorry

end NUMINAMATH_CALUDE_cleaning_time_proof_l2847_284798


namespace NUMINAMATH_CALUDE_angle_A_in_special_triangle_l2847_284744

theorem angle_A_in_special_triangle (A B C : ℝ) : 
  0 < A ∧ 0 < B ∧ 0 < C →  -- Ensuring positive angles
  B = A + 10 →             -- Given condition
  C = B + 10 →             -- Given condition
  A + B + C = 180 →        -- Sum of angles in a triangle
  A = 50 := by sorry

end NUMINAMATH_CALUDE_angle_A_in_special_triangle_l2847_284744


namespace NUMINAMATH_CALUDE_spongebob_daily_earnings_l2847_284774

/-- Calculates Spongebob's earnings for the day based on burger and fries sales -/
def spongebob_earnings (num_burgers : ℕ) (burger_price : ℚ) (num_fries : ℕ) (fries_price : ℚ) : ℚ :=
  num_burgers * burger_price + num_fries * fries_price

/-- Theorem stating Spongebob's earnings for the day -/
theorem spongebob_daily_earnings :
  spongebob_earnings 30 2 12 (3/2) = 78 := by
  sorry


end NUMINAMATH_CALUDE_spongebob_daily_earnings_l2847_284774


namespace NUMINAMATH_CALUDE_total_tickets_after_sharing_l2847_284728

def tate_initial_tickets : ℕ := 32
def tate_bought_tickets : ℕ := 2

def tate_final_tickets : ℕ := tate_initial_tickets + tate_bought_tickets

def peyton_initial_tickets : ℕ := tate_final_tickets / 2

def peyton_given_away_tickets : ℕ := peyton_initial_tickets / 3

def peyton_final_tickets : ℕ := peyton_initial_tickets - peyton_given_away_tickets

theorem total_tickets_after_sharing :
  tate_final_tickets + peyton_final_tickets = 46 := by
  sorry

end NUMINAMATH_CALUDE_total_tickets_after_sharing_l2847_284728


namespace NUMINAMATH_CALUDE_line_segment_ratio_l2847_284789

/-- Given points E, F, G, and H on a line in that order, prove that EG:FH = 10:17 -/
theorem line_segment_ratio (E F G H : ℝ) : 
  (F - E = 3) → (G - F = 7) → (H - E = 20) → (G - E) / (H - F) = 10 / 17 := by
sorry

end NUMINAMATH_CALUDE_line_segment_ratio_l2847_284789


namespace NUMINAMATH_CALUDE_sum_of_integers_l2847_284749

theorem sum_of_integers (x y : ℕ+) 
  (h1 : x.val^2 + y.val^2 = 130) 
  (h2 : x.val * y.val = 36) : 
  x.val + y.val = Real.sqrt 202 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_integers_l2847_284749


namespace NUMINAMATH_CALUDE_smallest_integer_gcf_24_is_4_l2847_284768

theorem smallest_integer_gcf_24_is_4 : 
  ∀ n : ℕ, n > 100 → Nat.gcd n 24 = 4 → n ≥ 104 :=
by sorry

end NUMINAMATH_CALUDE_smallest_integer_gcf_24_is_4_l2847_284768


namespace NUMINAMATH_CALUDE_mixture_ratio_l2847_284778

theorem mixture_ratio (p q : ℝ) : 
  p + q = 20 →
  p / (q + 1) = 4 / 3 →
  p / q = 3 / 2 := by
sorry

end NUMINAMATH_CALUDE_mixture_ratio_l2847_284778


namespace NUMINAMATH_CALUDE_suresh_completion_time_l2847_284715

theorem suresh_completion_time (ashutosh_time : ℝ) (suresh_partial_time : ℝ) (ashutosh_partial_time : ℝ) 
  (h1 : ashutosh_time = 30)
  (h2 : suresh_partial_time = 9)
  (h3 : ashutosh_partial_time = 12)
  : ∃ (suresh_time : ℝ), 
    suresh_partial_time / suresh_time + ashutosh_partial_time / ashutosh_time = 1 ∧ 
    suresh_time = 15 := by
  sorry

end NUMINAMATH_CALUDE_suresh_completion_time_l2847_284715


namespace NUMINAMATH_CALUDE_lemonade_stand_boys_l2847_284740

theorem lemonade_stand_boys (initial_group : ℕ) : 
  let initial_boys : ℕ := (6 * initial_group) / 10
  let final_group : ℕ := initial_group
  let final_boys : ℕ := initial_boys - 3
  (6 * initial_group = 10 * initial_boys) ∧ 
  (2 * final_boys = final_group) →
  initial_boys = 18 := by
sorry

end NUMINAMATH_CALUDE_lemonade_stand_boys_l2847_284740


namespace NUMINAMATH_CALUDE_greatest_two_digit_multiple_of_3_and_5_l2847_284704

theorem greatest_two_digit_multiple_of_3_and_5 : 
  ∀ n : ℕ, n ≤ 99 → n ≥ 10 → n % 3 = 0 → n % 5 = 0 → n ≤ 90 :=
by
  sorry

end NUMINAMATH_CALUDE_greatest_two_digit_multiple_of_3_and_5_l2847_284704


namespace NUMINAMATH_CALUDE_iesha_school_books_l2847_284702

/-- The number of books Iesha has about school -/
def books_about_school (total_books sports_books : ℕ) : ℕ :=
  total_books - sports_books

/-- Theorem stating that Iesha has 19 books about school -/
theorem iesha_school_books : 
  books_about_school 58 39 = 19 := by
  sorry

end NUMINAMATH_CALUDE_iesha_school_books_l2847_284702


namespace NUMINAMATH_CALUDE_successive_numbers_product_l2847_284776

theorem successive_numbers_product (n : ℤ) : 
  n * (n + 1) = 2652 → n = 51 := by
  sorry

end NUMINAMATH_CALUDE_successive_numbers_product_l2847_284776


namespace NUMINAMATH_CALUDE_triangle_perimeter_l2847_284786

/-- The perimeter of a right triangle formed by specific lines --/
theorem triangle_perimeter : 
  ∀ (l₁ l₂ l₃ : Set (ℝ × ℝ)),
  (∃ (m : ℝ), l₁ = {(x, y) | y = m * x}) →  -- l₁ passes through origin
  (l₂ = {(x, y) | x = 2}) →                 -- l₂ is x = 2
  (l₃ = {(x, y) | y = 2 - (Real.sqrt 5 / 5) * x}) →  -- l₃ is y = 2 - (√5/5)x
  (∃ (p₁ p₂ p₃ : ℝ × ℝ), p₁ ∈ l₁ ∩ l₂ ∧ p₂ ∈ l₁ ∩ l₃ ∧ p₃ ∈ l₂ ∩ l₃) →  -- intersection points exist
  (∃ (v₁ v₂ : ℝ × ℝ), v₁ ∈ l₁ ∧ v₂ ∈ l₃ ∧ (v₁.1 - v₂.1) * (v₁.2 - v₂.2) = 0) →  -- right angle condition
  let perimeter := 2 + (12 * Real.sqrt 5 - 10) / 5 + 2 * Real.sqrt 6
  ∃ (p₁ p₂ p₃ : ℝ × ℝ), 
    p₁ ∈ l₁ ∩ l₂ ∧ p₂ ∈ l₁ ∩ l₃ ∧ p₃ ∈ l₂ ∩ l₃ ∧
    Real.sqrt ((p₁.1 - p₂.1)^2 + (p₁.2 - p₂.2)^2) + 
    Real.sqrt ((p₂.1 - p₃.1)^2 + (p₂.2 - p₃.2)^2) + 
    Real.sqrt ((p₃.1 - p₁.1)^2 + (p₃.2 - p₁.2)^2) = perimeter :=
by
  sorry

end NUMINAMATH_CALUDE_triangle_perimeter_l2847_284786


namespace NUMINAMATH_CALUDE_sum_of_a_and_b_is_twelve_l2847_284791

/-- Given a function y of x, prove that a + b = 12 -/
theorem sum_of_a_and_b_is_twelve 
  (y : ℝ → ℝ) 
  (a b : ℝ) 
  (h1 : ∀ x, y x = a + b / (x + 1))
  (h2 : y (-2) = 2)
  (h3 : y (-6) = 6) :
  a + b = 12 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_a_and_b_is_twelve_l2847_284791


namespace NUMINAMATH_CALUDE_least_common_multiple_first_ten_l2847_284719

theorem least_common_multiple_first_ten : ∃ n : ℕ, 
  (∀ k : ℕ, k ≤ 10 → k > 0 → n % k = 0) ∧ 
  (∀ m : ℕ, m < n → ∃ k : ℕ, k ≤ 10 ∧ k > 0 ∧ m % k ≠ 0) ∧
  n = 2520 := by
  sorry

end NUMINAMATH_CALUDE_least_common_multiple_first_ten_l2847_284719


namespace NUMINAMATH_CALUDE_sin_240_degrees_l2847_284764

theorem sin_240_degrees : Real.sin (240 * π / 180) = -Real.sqrt 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_sin_240_degrees_l2847_284764


namespace NUMINAMATH_CALUDE_diophantine_equation_solutions_l2847_284714

theorem diophantine_equation_solutions :
  ∀ x y : ℤ, x^2 = 1 + 4*y^3*(y + 2) ↔ 
    (x = 1 ∧ y = 0) ∨ (x = 1 ∧ y = -2) ∨ (x = -1 ∧ y = 0) ∨ (x = -1 ∧ y = -2) := by
  sorry

end NUMINAMATH_CALUDE_diophantine_equation_solutions_l2847_284714


namespace NUMINAMATH_CALUDE_no_prime_sum_10001_l2847_284700

/-- A function that returns the number of ways to write n as the sum of two primes -/
def countPrimePairs (n : ℕ) : ℕ :=
  (Finset.filter (fun p => Nat.Prime p ∧ Nat.Prime (n - p)) (Finset.range n)).card

/-- Theorem stating that 10001 cannot be written as the sum of two primes -/
theorem no_prime_sum_10001 : countPrimePairs 10001 = 0 := by sorry

end NUMINAMATH_CALUDE_no_prime_sum_10001_l2847_284700


namespace NUMINAMATH_CALUDE_extremum_implies_a_equals_e_l2847_284723

/-- If f(x) = e^x - ax has an extremum at x = 1, then a = e -/
theorem extremum_implies_a_equals_e (a : ℝ) : 
  (∃ (f : ℝ → ℝ), (∀ x, f x = Real.exp x - a * x) ∧ 
   (∃ ε > 0, ∀ h ≠ 0, |h| < ε → f (1 + h) ≤ f 1)) → 
  a = Real.exp 1 := by
sorry

end NUMINAMATH_CALUDE_extremum_implies_a_equals_e_l2847_284723


namespace NUMINAMATH_CALUDE_cookies_per_box_l2847_284796

/-- The number of cookies in each box, given the collection amounts of Abigail, Grayson, and Olivia, and the total number of cookies. -/
theorem cookies_per_box (abigail_boxes : ℚ) (grayson_boxes : ℚ) (olivia_boxes : ℚ) (total_cookies : ℕ) :
  abigail_boxes = 2 →
  grayson_boxes = 3 / 4 →
  olivia_boxes = 3 →
  total_cookies = 276 →
  total_cookies / (abigail_boxes + grayson_boxes + olivia_boxes) = 48 := by
sorry

end NUMINAMATH_CALUDE_cookies_per_box_l2847_284796


namespace NUMINAMATH_CALUDE_equality_from_inequalities_l2847_284703

theorem equality_from_inequalities (x₁ x₂ x₃ x₄ x₅ : ℝ) 
  (h_pos : x₁ > 0 ∧ x₂ > 0 ∧ x₃ > 0 ∧ x₄ > 0 ∧ x₅ > 0)
  (h1 : (x₁^2 - x₃*x₅)*(x₂^2 - x₃*x₅) ≤ 0)
  (h2 : (x₂^2 - x₄*x₁)*(x₃^2 - x₄*x₁) ≤ 0)
  (h3 : (x₃^2 - x₅*x₂)*(x₄^2 - x₅*x₂) ≤ 0)
  (h4 : (x₄^2 - x₁*x₃)*(x₅^2 - x₁*x₃) ≤ 0)
  (h5 : (x₅^2 - x₂*x₄)*(x₁^2 - x₂*x₄) ≤ 0) :
  x₁ = x₂ ∧ x₂ = x₃ ∧ x₃ = x₄ ∧ x₄ = x₅ :=
by sorry

end NUMINAMATH_CALUDE_equality_from_inequalities_l2847_284703


namespace NUMINAMATH_CALUDE_divisibility_problem_l2847_284726

theorem divisibility_problem (n : ℕ) (h : n = (List.range 2001).foldl (· * ·) 1) :
  ∃ k : ℤ, n + (4003 * n - 4002) = 4003 * k := by
  sorry

end NUMINAMATH_CALUDE_divisibility_problem_l2847_284726


namespace NUMINAMATH_CALUDE_seating_theorem_l2847_284730

def factorial (n : ℕ) : ℕ := (List.range n).foldl (·*·) 1

def seating_arrangements (n : ℕ) (k : ℕ) : ℕ :=
  factorial n - factorial (n - k + 1) * factorial k

theorem seating_theorem :
  seating_arrangements 8 3 = 36000 :=
sorry

end NUMINAMATH_CALUDE_seating_theorem_l2847_284730


namespace NUMINAMATH_CALUDE_union_A_B_disjoint_A_B_l2847_284742

-- Define set A
def A : Set ℝ := {x | 1 < x ∧ x ≤ 2}

-- Define set B (parameterized by a)
def B (a : ℝ) : Set ℝ := {y | a < y ∧ y ≤ a + 1}

-- Theorem 1: Union of A and B when a = 3/2
theorem union_A_B : A ∪ B (3/2) = {x | 1 < x ∧ x ≤ 5/2} := by sorry

-- Theorem 2: Condition for A and B to be disjoint
theorem disjoint_A_B : ∀ a : ℝ, A ∩ B a = ∅ ↔ a ≥ 2 ∨ a ≤ 0 := by sorry

end NUMINAMATH_CALUDE_union_A_B_disjoint_A_B_l2847_284742


namespace NUMINAMATH_CALUDE_unique_solution_linear_system_l2847_284795

theorem unique_solution_linear_system :
  ∃! (x y z : ℝ), 
    2*x - 3*y + z = -4 ∧
    5*x - 2*y - 3*z = 7 ∧
    x + y - 4*z = -6 := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_linear_system_l2847_284795


namespace NUMINAMATH_CALUDE_fraction_calculation_l2847_284707

theorem fraction_calculation : 
  (2 / 5 + 3 / 7) / ((4 / 9) * (1 / 8)) = 522 / 35 := by
  sorry

end NUMINAMATH_CALUDE_fraction_calculation_l2847_284707


namespace NUMINAMATH_CALUDE_no_quadratic_term_implies_m_value_l2847_284790

theorem no_quadratic_term_implies_m_value (m : ℝ) : 
  (∀ x : ℝ, ∃ a b c : ℝ, (x + m) * (x^2 + 2*x - 1) = a*x^3 + b*x + c) → m = -2 := by
  sorry

end NUMINAMATH_CALUDE_no_quadratic_term_implies_m_value_l2847_284790


namespace NUMINAMATH_CALUDE_negation_equivalence_l2847_284762

/-- An exponential function -/
def ExponentialFunction (f : ℝ → ℝ) : Prop := sorry

/-- A monotonic function -/
def MonotonicFunction (f : ℝ → ℝ) : Prop := sorry

/-- The statement "All exponential functions are monotonic functions" -/
def AllExponentialAreMonotonic : Prop :=
  ∀ f : ℝ → ℝ, ExponentialFunction f → MonotonicFunction f

/-- The negation of "All exponential functions are monotonic functions" -/
def NegationAllExponentialAreMonotonic : Prop :=
  ∃ f : ℝ → ℝ, ExponentialFunction f ∧ ¬MonotonicFunction f

/-- Theorem: The negation of "All exponential functions are monotonic functions"
    is equivalent to "There exists at least one exponential function that is not a monotonic function" -/
theorem negation_equivalence :
  ¬AllExponentialAreMonotonic ↔ NegationAllExponentialAreMonotonic :=
sorry

end NUMINAMATH_CALUDE_negation_equivalence_l2847_284762


namespace NUMINAMATH_CALUDE_partial_fraction_decomposition_l2847_284754

theorem partial_fraction_decomposition :
  ∃ (A B C : ℝ), ∀ (x : ℝ), x ≠ 0 → x^2 + 1 ≠ 0 →
    (-x^2 + 3*x - 4) / (x^3 + x) = A / x + (B*x + C) / (x^2 + 1) ∧
    A = -4 ∧ B = 3 ∧ C = 3 := by
  sorry

end NUMINAMATH_CALUDE_partial_fraction_decomposition_l2847_284754


namespace NUMINAMATH_CALUDE_two_white_balls_probability_l2847_284708

/-- The probability of drawing two white balls successively without replacement
    from a box containing 8 white balls and 9 black balls is 7/34. -/
theorem two_white_balls_probability :
  let total_balls : ℕ := 8 + 9
  let white_balls : ℕ := 8
  let black_balls : ℕ := 9
  let prob_first_white : ℚ := white_balls / total_balls
  let prob_second_white : ℚ := (white_balls - 1) / (total_balls - 1)
  prob_first_white * prob_second_white = 7 / 34 := by
  sorry

end NUMINAMATH_CALUDE_two_white_balls_probability_l2847_284708


namespace NUMINAMATH_CALUDE_trail_mix_portions_l2847_284729

theorem trail_mix_portions (nuts dried_fruit chocolate coconut : ℕ) 
  (h1 : nuts = 16) (h2 : dried_fruit = 6) (h3 : chocolate = 8) (h4 : coconut = 4) :
  Nat.gcd nuts (Nat.gcd dried_fruit (Nat.gcd chocolate coconut)) = 2 := by
  sorry

end NUMINAMATH_CALUDE_trail_mix_portions_l2847_284729


namespace NUMINAMATH_CALUDE_intersection_tangent_negative_x_l2847_284785

theorem intersection_tangent_negative_x (x₀ y₀ : ℝ) : 
  x₀ > 0 → y₀ = Real.tan x₀ → y₀ = -x₀ → 
  (x₀^2 + 1) * (Real.cos (2 * x₀) + 1) = 2 := by sorry

end NUMINAMATH_CALUDE_intersection_tangent_negative_x_l2847_284785


namespace NUMINAMATH_CALUDE_system_solution_l2847_284763

theorem system_solution : 
  ∃ (x y z : ℝ), 
    (x = 1/2 ∧ y = 0 ∧ z = 0) ∧
    (2*x + 3*y + z = 1) ∧
    (4*x - y + 2*z = 2) ∧
    (8*x + 5*y + 3*z = 4) := by
  sorry

end NUMINAMATH_CALUDE_system_solution_l2847_284763


namespace NUMINAMATH_CALUDE_ferris_wheel_capacity_l2847_284701

/-- The number of small seats on the Ferris wheel -/
def num_small_seats : ℕ := 2

/-- The total number of people that can ride on small seats -/
def total_people_small_seats : ℕ := 28

/-- The number of people each small seat can hold -/
def people_per_small_seat : ℕ := total_people_small_seats / num_small_seats

theorem ferris_wheel_capacity : people_per_small_seat = 14 := by
  sorry

end NUMINAMATH_CALUDE_ferris_wheel_capacity_l2847_284701


namespace NUMINAMATH_CALUDE_train_speed_l2847_284706

/-- The speed of a train given its length, the speed of a man running in the opposite direction, and the time it takes for the train to pass the man. -/
theorem train_speed (train_length : ℝ) (man_speed : ℝ) (passing_time : ℝ) :
  train_length = 110 →
  man_speed = 4 →
  passing_time = 9 / 3600 →
  (train_length / 1000) / passing_time - man_speed = 40 :=
by sorry

end NUMINAMATH_CALUDE_train_speed_l2847_284706


namespace NUMINAMATH_CALUDE_reach_probability_is_5_128_l2847_284759

/-- Represents a point in the 2D coordinate plane -/
structure Point where
  x : ℤ
  y : ℤ

/-- Represents a step direction -/
inductive Direction
  | Left
  | Right
  | Up
  | Down

/-- The probability of taking a specific step -/
def stepProbability : ℚ := 1 / 4

/-- The starting point -/
def start : Point := ⟨0, 0⟩

/-- The target point -/
def target : Point := ⟨3, 1⟩

/-- The maximum number of steps allowed -/
def maxSteps : ℕ := 5

/-- Calculates the probability of reaching the target point from the start point
    in at most maxSteps steps -/
def reachProbability (start target : Point) (maxSteps : ℕ) : ℚ :=
  sorry

theorem reach_probability_is_5_128 :
  reachProbability start target maxSteps = 5 / 128 :=
sorry

end NUMINAMATH_CALUDE_reach_probability_is_5_128_l2847_284759


namespace NUMINAMATH_CALUDE_negative_fractions_comparison_l2847_284799

theorem negative_fractions_comparison : -1/2 < -1/3 := by
  sorry

end NUMINAMATH_CALUDE_negative_fractions_comparison_l2847_284799


namespace NUMINAMATH_CALUDE_part1_part2_l2847_284782

-- Part 1
theorem part1 (f : ℝ → ℝ) :
  (∀ x ≥ 0, f (Real.sqrt x + 1) = x + 2 * Real.sqrt x) →
  (∀ x ≥ 1, f x = x^2 - 2*x) :=
sorry

-- Part 2
theorem part2 (f : ℝ → ℝ) :
  (∃ k b : ℝ, ∀ x, f x = k * x + b) →
  (∀ x, 3 * f (x + 1) - 2 * f (x - 1) = 2 * x + 17) →
  (∀ x, f x = 2 * x + 7) :=
sorry

end NUMINAMATH_CALUDE_part1_part2_l2847_284782


namespace NUMINAMATH_CALUDE_largest_harmonious_n_is_correct_l2847_284783

/-- A coloring of a regular polygon's sides and diagonals. -/
def Coloring (n : ℕ) := Fin n → Fin n → Fin 2018

/-- A harmonious coloring has no two-colored triangles. -/
def Harmonious (n : ℕ) (c : Coloring n) : Prop :=
  ∀ i j k : Fin n, (c i j = c i k ∧ c i j ≠ c j k) → c i k = c j k

/-- The largest N for which a harmonious coloring of a regular N-gon exists. -/
def LargestHarmoniousN : ℕ := 2017^2

theorem largest_harmonious_n_is_correct :
  (∃ (c : Coloring LargestHarmoniousN), Harmonious LargestHarmoniousN c) ∧
  (∀ n > LargestHarmoniousN, ¬∃ (c : Coloring n), Harmonious n c) :=
sorry

end NUMINAMATH_CALUDE_largest_harmonious_n_is_correct_l2847_284783


namespace NUMINAMATH_CALUDE_complex_equation_solution_l2847_284770

theorem complex_equation_solution (x y : ℝ) : 
  (Complex.mk (2 * x - 1) 1 = Complex.mk y (y - 2)) → x = 2 ∧ y = 3 := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l2847_284770


namespace NUMINAMATH_CALUDE_yoki_cans_count_l2847_284735

def total_cans : ℕ := 85
def ladonna_cans : ℕ := 25
def prikya_cans : ℕ := 2 * ladonna_cans
def avi_initial_cans : ℕ := 8
def avi_remaining_cans : ℕ := avi_initial_cans / 2

theorem yoki_cans_count : 
  total_cans - (ladonna_cans + prikya_cans + avi_remaining_cans) = 6 := by
  sorry

end NUMINAMATH_CALUDE_yoki_cans_count_l2847_284735


namespace NUMINAMATH_CALUDE_negation_equivalence_l2847_284769

theorem negation_equivalence : 
  (¬ ∃ x : ℝ, x^2 - 2*x > 2) ↔ (∀ x : ℝ, x^2 - 2*x ≤ 2) := by
  sorry

end NUMINAMATH_CALUDE_negation_equivalence_l2847_284769


namespace NUMINAMATH_CALUDE_math_competition_prizes_l2847_284751

theorem math_competition_prizes (x y s : ℝ) 
  (h1 : 100 * (x + 3 * y) = s)
  (h2 : 80 * (x + 5 * y) = s) :
  x = 5 * y ∧ s = 160 * x ∧ s = 800 * y := by
  sorry

end NUMINAMATH_CALUDE_math_competition_prizes_l2847_284751


namespace NUMINAMATH_CALUDE_annika_hiking_time_l2847_284748

/-- Annika's hiking problem -/
theorem annika_hiking_time (rate : ℝ) (initial_distance : ℝ) (total_distance : ℝ) : 
  rate = 12 →
  initial_distance = 2.75 →
  total_distance = 3.5 →
  (total_distance - initial_distance) * rate + total_distance * rate = 51 := by
sorry

end NUMINAMATH_CALUDE_annika_hiking_time_l2847_284748


namespace NUMINAMATH_CALUDE_max_value_2x_3y_l2847_284747

theorem max_value_2x_3y (x y : ℝ) (h : 3 * x^2 + y^2 ≤ 3) :
  ∃ (M : ℝ), M = Real.sqrt 31 ∧ 2*x + 3*y ≤ M ∧ ∀ (N : ℝ), (∀ (a b : ℝ), 3 * a^2 + b^2 ≤ 3 → 2*a + 3*b ≤ N) → M ≤ N :=
sorry

end NUMINAMATH_CALUDE_max_value_2x_3y_l2847_284747


namespace NUMINAMATH_CALUDE_sum_of_a_and_b_l2847_284731

theorem sum_of_a_and_b (a b : ℝ) (ha : |a| = 5) (hb : |b| = 2) (ha_neg : a < 0) (hb_pos : b > 0) :
  a + b = -3 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_a_and_b_l2847_284731


namespace NUMINAMATH_CALUDE_probability_green_is_81_160_l2847_284775

structure Container where
  red : ℕ
  green : ℕ

def containerA : Container := ⟨3, 5⟩
def containerB : Container := ⟨5, 5⟩
def containerC : Container := ⟨7, 3⟩
def containerD : Container := ⟨4, 6⟩

def containers : List Container := [containerA, containerB, containerC, containerD]

def probabilityGreenFromContainer (c : Container) : ℚ :=
  c.green / (c.red + c.green)

def probabilityGreen : ℚ :=
  (1 / containers.length) * (containers.map probabilityGreenFromContainer).sum

theorem probability_green_is_81_160 : probabilityGreen = 81 / 160 := by
  sorry

end NUMINAMATH_CALUDE_probability_green_is_81_160_l2847_284775


namespace NUMINAMATH_CALUDE_fraction_sum_equality_l2847_284716

theorem fraction_sum_equality : 
  (2 : ℚ) / 100 + 5 / 1000 + 5 / 10000 + 3 * (4 / 1000) = 375 / 10000 := by
  sorry

end NUMINAMATH_CALUDE_fraction_sum_equality_l2847_284716


namespace NUMINAMATH_CALUDE_problem_solution_l2847_284777

theorem problem_solution (p_xavier p_yvonne p_zelda p_wendell : ℚ)
  (h_xavier : p_xavier = 1/4)
  (h_yvonne : p_yvonne = 1/3)
  (h_zelda : p_zelda = 5/8)
  (h_wendell : p_wendell = 1/2) :
  p_xavier * p_yvonne * (1 - p_zelda) * (1 - p_wendell) = 1/64 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l2847_284777


namespace NUMINAMATH_CALUDE_no_all_ones_quadratic_l2847_284720

/-- A natural number whose decimal representation consists only of ones -/
def all_ones (n : ℕ) : Prop :=
  ∃ k : ℕ, n = (10^k - 1) / 9

/-- The property that a natural number's decimal representation consists only of ones -/
def has_all_ones_representation (n : ℕ) : Prop :=
  all_ones n

/-- A quadratic polynomial with integer coefficients -/
def is_quadratic_polynomial (P : ℕ → ℕ) : Prop :=
  ∃ a b c : ℤ, ∀ x : ℕ, P x = a * x^2 + b * x + c

theorem no_all_ones_quadratic :
  ∀ P : ℕ → ℕ, is_quadratic_polynomial P →
    ∃ n : ℕ, has_all_ones_representation n ∧ ¬(has_all_ones_representation (P n)) :=
sorry

end NUMINAMATH_CALUDE_no_all_ones_quadratic_l2847_284720


namespace NUMINAMATH_CALUDE_circumscribed_isosceles_trapezoid_radius_l2847_284705

/-- An isosceles trapezoid circumscribed around a circle -/
structure CircumscribedIsoscelesTrapezoid where
  /-- The angle at the base of the trapezoid -/
  baseAngle : ℝ
  /-- The length of the midline of the trapezoid -/
  midline : ℝ
  /-- The radius of the inscribed circle -/
  radius : ℝ

/-- The theorem stating the relationship between the trapezoid's properties and the inscribed circle's radius -/
theorem circumscribed_isosceles_trapezoid_radius 
  (t : CircumscribedIsoscelesTrapezoid) 
  (h1 : t.baseAngle = 30 * π / 180)  -- 30 degrees in radians
  (h2 : t.midline = 10) : 
  t.radius = 2.5 := by
  sorry


end NUMINAMATH_CALUDE_circumscribed_isosceles_trapezoid_radius_l2847_284705


namespace NUMINAMATH_CALUDE_point_d_is_multiple_of_fifteen_l2847_284788

/-- Represents a point on the number line -/
structure Point where
  value : ℤ

/-- Represents the number line with four special points -/
structure NumberLine where
  w : Point
  x : Point
  y : Point
  z : Point
  consecutive : w.value < x.value ∧ x.value < y.value ∧ y.value < z.value
  multiples_of_three : (w.value % 3 = 0 ∧ y.value % 3 = 0) ∨ (x.value % 3 = 0 ∧ z.value % 3 = 0)
  multiples_of_five : (w.value % 5 = 0 ∧ z.value % 5 = 0) ∨ (x.value % 5 = 0 ∧ y.value % 5 = 0)

/-- The point D, which is 5 units away from one of the multiples of 5 -/
def point_d (nl : NumberLine) : Point :=
  if nl.w.value % 5 = 0 then { value := nl.w.value + 5 }
  else if nl.x.value % 5 = 0 then { value := nl.x.value + 5 }
  else if nl.y.value % 5 = 0 then { value := nl.y.value + 5 }
  else { value := nl.z.value + 5 }

theorem point_d_is_multiple_of_fifteen (nl : NumberLine) :
  (point_d nl).value % 15 = 0 := by
  sorry

end NUMINAMATH_CALUDE_point_d_is_multiple_of_fifteen_l2847_284788


namespace NUMINAMATH_CALUDE_smallest_positive_sum_l2847_284712

theorem smallest_positive_sum (x y : ℝ) : 
  (Real.sin x + Real.cos y) * (Real.cos x - Real.sin y) = 1 + Real.sin (x - y) * Real.cos (x + y) →
  ∃ (k : ℤ), x + y = 2 * π * (k : ℝ) ∧ 
  (∀ (m : ℤ), x + y = 2 * π * (m : ℝ) → k ≤ m) ∧
  0 < 2 * π * (k : ℝ) :=
sorry

end NUMINAMATH_CALUDE_smallest_positive_sum_l2847_284712


namespace NUMINAMATH_CALUDE_paving_rate_per_square_metre_l2847_284793

/-- Proves that the rate per square metre for paving a room is Rs. 950 given the specified conditions. -/
theorem paving_rate_per_square_metre
  (length : ℝ)
  (width : ℝ)
  (total_cost : ℝ)
  (h1 : length = 5.5)
  (h2 : width = 4)
  (h3 : total_cost = 20900) :
  total_cost / (length * width) = 950 := by
  sorry

#check paving_rate_per_square_metre

end NUMINAMATH_CALUDE_paving_rate_per_square_metre_l2847_284793


namespace NUMINAMATH_CALUDE_y_derivative_l2847_284727

noncomputable def y (x : ℝ) : ℝ := 
  (3 / (8 * Real.sqrt 2)) * Real.log ((Real.sqrt 2 + Real.tanh x) / (Real.sqrt 2 - Real.tanh x)) - 
  (Real.tanh x) / (4 * (2 - Real.tanh x ^ 2))

theorem y_derivative (x : ℝ) : 
  deriv y x = 1 / (2 + Real.cosh x ^ 2) ^ 2 :=
sorry

end NUMINAMATH_CALUDE_y_derivative_l2847_284727


namespace NUMINAMATH_CALUDE_opposite_absolute_value_and_square_l2847_284737

theorem opposite_absolute_value_and_square (x y : ℝ) :
  |x + y - 2| + (2*x - 3*y + 5)^2 = 0 → x = 1/5 ∧ y = 9/5 := by
  sorry

end NUMINAMATH_CALUDE_opposite_absolute_value_and_square_l2847_284737


namespace NUMINAMATH_CALUDE_harris_flour_amount_l2847_284760

theorem harris_flour_amount (flour_per_cake : ℕ) (total_cakes : ℕ) (traci_flour : ℕ) :
  flour_per_cake = 100 →
  total_cakes = 9 →
  traci_flour = 500 →
  flour_per_cake * total_cakes - traci_flour = 400 := by
sorry

end NUMINAMATH_CALUDE_harris_flour_amount_l2847_284760


namespace NUMINAMATH_CALUDE_pyramid_hemisphere_tangency_l2847_284743

theorem pyramid_hemisphere_tangency (h : ℝ) (r : ℝ) (edge_length : ℝ) : 
  h = 8 → r = 3 → 
  (edge_length * edge_length = 2 * ((h * h - r * r) / h * r)^2) →
  edge_length = 24 * Real.sqrt 110 / 55 :=
by sorry

end NUMINAMATH_CALUDE_pyramid_hemisphere_tangency_l2847_284743


namespace NUMINAMATH_CALUDE_four_different_results_l2847_284724

/-- Represents a parenthesized expression of 3^3^3^3 -/
inductive ParenthesizedExpr
| Single : ParenthesizedExpr
| Left : ParenthesizedExpr → ParenthesizedExpr
| Right : ParenthesizedExpr → ParenthesizedExpr
| Both : ParenthesizedExpr → ParenthesizedExpr → ParenthesizedExpr

/-- Evaluates a parenthesized expression to a natural number -/
def evaluate : ParenthesizedExpr → ℕ
| ParenthesizedExpr.Single => 3^3^3^3
| ParenthesizedExpr.Left e => 3^(evaluate e)
| ParenthesizedExpr.Right e => (evaluate e)^3
| ParenthesizedExpr.Both e1 e2 => (evaluate e1)^(evaluate e2)

/-- All possible parenthesized expressions of 3^3^3^3 -/
def allExpressions : List ParenthesizedExpr := [
  ParenthesizedExpr.Single,
  ParenthesizedExpr.Left (ParenthesizedExpr.Left (ParenthesizedExpr.Single)),
  ParenthesizedExpr.Left (ParenthesizedExpr.Right ParenthesizedExpr.Single),
  ParenthesizedExpr.Right (ParenthesizedExpr.Left ParenthesizedExpr.Single),
  ParenthesizedExpr.Right (ParenthesizedExpr.Right ParenthesizedExpr.Single),
  ParenthesizedExpr.Both ParenthesizedExpr.Single ParenthesizedExpr.Single
]

/-- The theorem stating that there are exactly 4 different results -/
theorem four_different_results :
  (allExpressions.map evaluate).toFinset.card = 4 := by sorry

end NUMINAMATH_CALUDE_four_different_results_l2847_284724


namespace NUMINAMATH_CALUDE_rectangle_max_m_l2847_284756

/-- Given a rectangle with area S and perimeter p, 
    M = (16 - p) / (p^2 + 2p) is maximized when the rectangle is a square -/
theorem rectangle_max_m (S : ℝ) (p : ℝ) (h_S : S > 0) (h_p : p > 0) :
  let M := (16 - p) / (p^2 + 2*p)
  M ≤ (4 - Real.sqrt S) / (4*S + 2*Real.sqrt S) :=
by sorry

end NUMINAMATH_CALUDE_rectangle_max_m_l2847_284756


namespace NUMINAMATH_CALUDE_smallest_d_value_l2847_284772

theorem smallest_d_value (c d : ℕ+) (h1 : c - d = 8) 
  (h2 : Nat.gcd ((c^3 + d^3) / (c + d)) (c * d) = 16) : 
  d ≥ 4 ∧ ∃ (c' d' : ℕ+), c' - d' = 8 ∧ 
    Nat.gcd ((c'^3 + d'^3) / (c' + d')) (c' * d') = 16 ∧ d' = 4 :=
sorry

end NUMINAMATH_CALUDE_smallest_d_value_l2847_284772


namespace NUMINAMATH_CALUDE_bus_speed_with_stoppages_l2847_284779

/-- Given a bus that travels at 90 km/hr excluding stoppages and stops for 4 minutes per hour,
    its speed including stoppages is 84 km/hr. -/
theorem bus_speed_with_stoppages 
  (speed_without_stoppages : ℝ) 
  (stoppage_time : ℝ) 
  (total_time : ℝ) :
  speed_without_stoppages = 90 →
  stoppage_time = 4 →
  total_time = 60 →
  (speed_without_stoppages * (total_time - stoppage_time)) / total_time = 84 := by
  sorry

#check bus_speed_with_stoppages

end NUMINAMATH_CALUDE_bus_speed_with_stoppages_l2847_284779


namespace NUMINAMATH_CALUDE_track_team_initial_girls_l2847_284713

theorem track_team_initial_girls (initial_boys : ℕ) (girls_joined : ℕ) (boys_quit : ℕ) (final_total : ℕ) :
  initial_boys = 15 →
  girls_joined = 7 →
  boys_quit = 4 →
  final_total = 36 →
  ∃ initial_girls : ℕ, initial_girls + initial_boys + girls_joined - boys_quit = final_total ∧ initial_girls = 18 :=
by sorry

end NUMINAMATH_CALUDE_track_team_initial_girls_l2847_284713


namespace NUMINAMATH_CALUDE_quadratic_decreasing_before_vertex_l2847_284710

def f (x : ℝ) : ℝ := 2 * (x - 3)^2 + 1

theorem quadratic_decreasing_before_vertex :
  ∀ (x1 x2 : ℝ), x1 < x2 → x2 < 3 → f x1 > f x2 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_decreasing_before_vertex_l2847_284710


namespace NUMINAMATH_CALUDE_washer_dryer_cost_l2847_284721

theorem washer_dryer_cost (total_cost : ℝ) (price_difference : ℝ) (dryer_cost : ℝ) : 
  total_cost = 1200 →
  price_difference = 220 →
  total_cost = dryer_cost + (dryer_cost + price_difference) →
  dryer_cost = 490 := by
sorry

end NUMINAMATH_CALUDE_washer_dryer_cost_l2847_284721


namespace NUMINAMATH_CALUDE_possible_values_of_a_l2847_284765

theorem possible_values_of_a (x y a : ℝ) 
  (h1 : x + y = a) 
  (h2 : x^3 + y^3 = a) 
  (h3 : x^5 + y^5 = a) : 
  a ∈ ({-2, -1, 0, 1, 2} : Set ℝ) := by
  sorry

end NUMINAMATH_CALUDE_possible_values_of_a_l2847_284765


namespace NUMINAMATH_CALUDE_soccer_match_goals_l2847_284711

/-- Calculates the total number of goals scored in a soccer match -/
def total_goals (kickers_first : ℕ) : ℕ := 
  let kickers_second : ℕ := 2 * kickers_first
  let spiders_first : ℕ := kickers_first / 2
  let spiders_second : ℕ := 2 * kickers_second
  kickers_first + kickers_second + spiders_first + spiders_second

/-- Theorem stating that given the conditions of the soccer match, the total goals scored is 15 -/
theorem soccer_match_goals : total_goals 2 = 15 := by
  sorry

end NUMINAMATH_CALUDE_soccer_match_goals_l2847_284711


namespace NUMINAMATH_CALUDE_factor_x_squared_minus_196_l2847_284787

theorem factor_x_squared_minus_196 (x : ℝ) : x^2 - 196 = (x - 14) * (x + 14) := by
  sorry

end NUMINAMATH_CALUDE_factor_x_squared_minus_196_l2847_284787
