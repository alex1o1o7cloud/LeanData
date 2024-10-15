import Mathlib

namespace NUMINAMATH_CALUDE_unique_digit_arrangement_l1957_195731

/-- A type representing digits from 1 to 9 -/
inductive Digit : Type
  | one : Digit
  | two : Digit
  | three : Digit
  | four : Digit
  | five : Digit
  | six : Digit
  | seven : Digit
  | eight : Digit
  | nine : Digit

/-- Convert a Digit to a natural number -/
def digit_to_nat (d : Digit) : ℕ :=
  match d with
  | Digit.one => 1
  | Digit.two => 2
  | Digit.three => 3
  | Digit.four => 4
  | Digit.five => 5
  | Digit.six => 6
  | Digit.seven => 7
  | Digit.eight => 8
  | Digit.nine => 9

/-- Convert three Digits to a natural number -/
def three_digit_to_nat (e f g : Digit) : ℕ :=
  100 * (digit_to_nat e) + 10 * (digit_to_nat f) + (digit_to_nat g)

/-- The main theorem -/
theorem unique_digit_arrangement :
  ∃! (a b c d e f g h : Digit),
    (a ≠ b) ∧ (a ≠ c) ∧ (a ≠ d) ∧ (a ≠ e) ∧ (a ≠ f) ∧ (a ≠ g) ∧ (a ≠ h) ∧
    (b ≠ c) ∧ (b ≠ d) ∧ (b ≠ e) ∧ (b ≠ f) ∧ (b ≠ g) ∧ (b ≠ h) ∧
    (c ≠ d) ∧ (c ≠ e) ∧ (c ≠ f) ∧ (c ≠ g) ∧ (c ≠ h) ∧
    (d ≠ e) ∧ (d ≠ f) ∧ (d ≠ g) ∧ (d ≠ h) ∧
    (e ≠ f) ∧ (e ≠ g) ∧ (e ≠ h) ∧
    (f ≠ g) ∧ (f ≠ h) ∧
    (g ≠ h) ∧
    ((digit_to_nat a) / (digit_to_nat b) = (digit_to_nat c) / (digit_to_nat d)) ∧
    ((digit_to_nat c) / (digit_to_nat d) = (three_digit_to_nat e f g) / (10 * (digit_to_nat h) + 9)) :=
by sorry


end NUMINAMATH_CALUDE_unique_digit_arrangement_l1957_195731


namespace NUMINAMATH_CALUDE_cody_book_series_l1957_195769

/-- The number of books in Cody's favorite book series -/
def books_in_series (first_week : ℕ) (second_week : ℕ) (subsequent_weeks : ℕ) (total_weeks : ℕ) : ℕ :=
  first_week + second_week + (subsequent_weeks * (total_weeks - 2))

/-- Theorem stating the number of books in Cody's series -/
theorem cody_book_series : books_in_series 6 3 9 7 = 54 := by
  sorry

end NUMINAMATH_CALUDE_cody_book_series_l1957_195769


namespace NUMINAMATH_CALUDE_maria_papers_left_l1957_195728

/-- The number of papers Maria has left after giving away some papers -/
def papers_left (desk : ℕ) (backpack : ℕ) (given_away : ℕ) : ℕ :=
  desk + backpack - given_away

/-- Theorem stating that Maria has 91 - x papers left after giving away x papers -/
theorem maria_papers_left (x : ℕ) :
  papers_left 50 41 x = 91 - x :=
by sorry

end NUMINAMATH_CALUDE_maria_papers_left_l1957_195728


namespace NUMINAMATH_CALUDE_exists_perpendicular_k_line_intersects_circle_chord_length_when_k_neg_one_l1957_195751

-- Define the line l
def line_l (k : ℝ) (x y : ℝ) : Prop := k * x - y + 2 * k = 0

-- Define the circle O
def circle_O (x y : ℝ) : Prop := x^2 + y^2 = 9

-- Define the line l₀
def line_l0 (x y : ℝ) : Prop := x - 2 * y + 2 = 0

-- Statement 1: Perpendicularity condition
theorem exists_perpendicular_k : ∃ k : ℝ, ∀ x y : ℝ, 
  line_l k x y → line_l0 x y → k * (1/2) = -1 :=
sorry

-- Statement 2: Intersection of line l and circle O
theorem line_intersects_circle : ∀ k : ℝ, ∃ x y : ℝ, 
  line_l k x y ∧ circle_O x y :=
sorry

-- Statement 3: Chord length when k = -1
theorem chord_length_when_k_neg_one : 
  ∃ x₁ y₁ x₂ y₂ : ℝ, 
    line_l (-1) x₁ y₁ ∧ line_l (-1) x₂ y₂ ∧ 
    circle_O x₁ y₁ ∧ circle_O x₂ y₂ ∧
    (x₂ - x₁)^2 + (y₂ - y₁)^2 = 28 :=
sorry

end NUMINAMATH_CALUDE_exists_perpendicular_k_line_intersects_circle_chord_length_when_k_neg_one_l1957_195751


namespace NUMINAMATH_CALUDE_bush_spacing_l1957_195763

theorem bush_spacing (yard_side_length : ℕ) (num_sides : ℕ) (num_bushes : ℕ) :
  yard_side_length = 16 →
  num_sides = 3 →
  num_bushes = 12 →
  (yard_side_length * num_sides) / num_bushes = 4 := by
  sorry

end NUMINAMATH_CALUDE_bush_spacing_l1957_195763


namespace NUMINAMATH_CALUDE_symmetrical_lines_and_ellipse_intersection_l1957_195701

/-- Given two lines symmetrical about y = x + 1, prove their slopes multiply to 1 and their intersection points with an ellipse form a line passing through a fixed point. -/
theorem symmetrical_lines_and_ellipse_intersection
  (k : ℝ) (h_k_pos : k > 0) (h_k_neq_one : k ≠ 1)
  (l : Set (ℝ × ℝ)) (l_eq : l = {(x, y) | y = k * x + 1})
  (l₁ : Set (ℝ × ℝ)) (k₁ : ℝ) (l₁_eq : l₁ = {(x, y) | y = k₁ * x + 1})
  (h_symmetry : ∀ (x y : ℝ), (x, y) ∈ l ↔ (y - 1, x + 1) ∈ l₁)
  (E : Set (ℝ × ℝ)) (E_eq : E = {(x, y) | x^2 / 4 + y^2 = 1})
  (A M : ℝ × ℝ) (h_AM : A ∈ E ∧ M ∈ E ∧ A ∈ l ∧ M ∈ l ∧ A ≠ M)
  (N : ℝ × ℝ) (h_AN : A ∈ E ∧ N ∈ E ∧ A ∈ l₁ ∧ N ∈ l₁ ∧ A ≠ N) :
  (k * k₁ = 1) ∧
  (∃ (m b : ℝ), ∀ (x : ℝ), M.2 - N.2 = m * (M.1 - N.1) ∧ N.2 = m * N.1 + b ∧ b = -5/3) :=
by sorry

end NUMINAMATH_CALUDE_symmetrical_lines_and_ellipse_intersection_l1957_195701


namespace NUMINAMATH_CALUDE_ellipse_intersection_area_and_ratio_l1957_195715

/-- The ellipse C: (x^2/4) + (y^2/b^2) = 1 with 0 < b < 2 -/
def ellipse (b : ℝ) (x y : ℝ) : Prop :=
  (x^2 / 4) + (y^2 / b^2) = 1 ∧ 0 < b ∧ b < 2

/-- The point M(1,1) is on the ellipse -/
def M_on_ellipse (b : ℝ) : Prop := ellipse b 1 1

/-- The line l: x + ty - 1 = 0 -/
def line (t : ℝ) (x y : ℝ) : Prop := x + t * y - 1 = 0

/-- Points A and B are the intersections of the line and the ellipse -/
def intersections (b t : ℝ) (A B : ℝ × ℝ) : Prop :=
  ellipse b A.1 A.2 ∧ ellipse b B.1 B.2 ∧ 
  line t A.1 A.2 ∧ line t B.1 B.2

/-- The area of triangle ABM -/
noncomputable def area_ABM (A B : ℝ × ℝ) : ℝ := sorry

/-- Points P and Q are intersections of AM and BM with x = 4 -/
def P_Q_intersections (A B : ℝ × ℝ) (P Q : ℝ × ℝ) : Prop :=
  P.1 = 4 ∧ Q.1 = 4 ∧ 
  (∃ k₁ k₂ : ℝ, P = k₁ • A + (1 - k₁) • (1, 1)) ∧
  (∃ k₃ k₄ : ℝ, Q = k₃ • B + (1 - k₃) • (1, 1))

/-- The area of triangle PQM -/
noncomputable def area_PQM (P Q : ℝ × ℝ) : ℝ := sorry

theorem ellipse_intersection_area_and_ratio 
  (b : ℝ) (h_M : M_on_ellipse b) :
  (∃ A B : ℝ × ℝ, intersections b 1 A B ∧ area_ABM A B = Real.sqrt 13 / 4) ∧
  (∃ t : ℝ, ∃ A B P Q : ℝ × ℝ, 
    intersections b t A B ∧ 
    P_Q_intersections A B P Q ∧
    area_PQM P Q = 5 * area_ABM A B ∧
    t = 3 * Real.sqrt 2 / 2 ∨ t = -3 * Real.sqrt 2 / 2) :=
sorry

end NUMINAMATH_CALUDE_ellipse_intersection_area_and_ratio_l1957_195715


namespace NUMINAMATH_CALUDE_binomial_expansion_theorem_l1957_195722

def binomial_coefficient (n k : ℕ) : ℕ := sorry

def binomial_expansion_coefficient (a : ℝ) : ℝ :=
  (-a) * binomial_coefficient 9 1

theorem binomial_expansion_theorem (a : ℝ) :
  binomial_expansion_coefficient a = 36 → a = -4 := by
  sorry

end NUMINAMATH_CALUDE_binomial_expansion_theorem_l1957_195722


namespace NUMINAMATH_CALUDE_circle_relationship_l1957_195768

-- Define the circles and point
def circle_O1 (x y : ℝ) : Prop := x^2 + y^2 = 9
def circle_O2 (a b x y : ℝ) : Prop := (x - a)^2 + (y - b)^2 = 1
def point_on_O1 (x y : ℝ) : Prop := circle_O1 x y

-- Define the condition for P and Q
def condition_PQ (x₁ y₁ a b : ℝ) : Prop := (a - x₁)^2 + (b - y₁)^2 = 1

-- Define the possible relationships
inductive CircleRelationship
  | ExternallyTangent
  | Intersecting
  | InternallyTangent

-- Theorem statement
theorem circle_relationship 
  (x₁ y₁ a b : ℝ) 
  (h1 : point_on_O1 x₁ y₁) 
  (h2 : condition_PQ x₁ y₁ a b) : 
  ∃ r : CircleRelationship, r = CircleRelationship.ExternallyTangent ∨ 
                            r = CircleRelationship.Intersecting ∨ 
                            r = CircleRelationship.InternallyTangent :=
sorry

end NUMINAMATH_CALUDE_circle_relationship_l1957_195768


namespace NUMINAMATH_CALUDE_probability_x_plus_2y_leq_6_l1957_195733

-- Define the region
def region : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | 0 ≤ p.1 ∧ p.1 ≤ 4 ∧ 0 ≤ p.2 ∧ p.2 ≤ 5}

-- Define the condition x + 2y ≤ 6
def condition (p : ℝ × ℝ) : Prop :=
  p.1 + 2 * p.2 ≤ 6

-- Define the probability measure on the region
noncomputable def prob : MeasureTheory.Measure (ℝ × ℝ) :=
  sorry

-- State the theorem
theorem probability_x_plus_2y_leq_6 :
  prob {p ∈ region | condition p} / prob region = 3 / 10 := by
  sorry

end NUMINAMATH_CALUDE_probability_x_plus_2y_leq_6_l1957_195733


namespace NUMINAMATH_CALUDE_xy_negative_implies_abs_sum_less_abs_diff_l1957_195779

theorem xy_negative_implies_abs_sum_less_abs_diff (x y : ℝ) 
  (h1 : x * y < 0) : 
  |x + y| < |x - y| := by sorry

end NUMINAMATH_CALUDE_xy_negative_implies_abs_sum_less_abs_diff_l1957_195779


namespace NUMINAMATH_CALUDE_remaining_distance_calculation_l1957_195738

/-- The remaining distance to travel after four people have traveled part of the way. -/
def remaining_distance (total_distance : ℝ) 
  (amoli_speed1 amoli_time1 amoli_speed2 amoli_time2 : ℝ)
  (anayet_speed1 anayet_time1 anayet_speed2 anayet_time2 : ℝ)
  (bimal_speed1 bimal_time1 bimal_speed2 bimal_time2 : ℝ)
  (chandni_distance : ℝ) : ℝ :=
  total_distance - 
  (amoli_speed1 * amoli_time1 + amoli_speed2 * amoli_time2 +
   anayet_speed1 * anayet_time1 + anayet_speed2 * anayet_time2 +
   bimal_speed1 * bimal_time1 + bimal_speed2 * bimal_time2 +
   chandni_distance)

/-- Theorem stating the remaining distance to travel. -/
theorem remaining_distance_calculation : 
  remaining_distance 1475 42 3.5 38 2 61 2.5 75 1.5 55 4 30 2 35 = 672 := by
  sorry

end NUMINAMATH_CALUDE_remaining_distance_calculation_l1957_195738


namespace NUMINAMATH_CALUDE_quadratic_inequality_properties_l1957_195764

/-- Given a quadratic inequality ax^2 + bx + c < 0 with solution set (1/t, t) where t > 0,
    prove certain properties about a, b, c, and related equations. -/
theorem quadratic_inequality_properties
  (a b c t : ℝ)
  (h_solution_set : ∀ x : ℝ, a * x^2 + b * x + c < 0 ↔ 1/t < x ∧ x < t)
  (h_t_pos : t > 0) :
  abc < 0 ∧
  2*a + b < 0 ∧
  (∀ x₁ x₂ : ℝ, (a * x₁ + b * Real.sqrt x₁ + c = 0 ∧
                 a * x₂ + b * Real.sqrt x₂ + c = 0) →
                x₁ + x₂ > t + 1/t) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_properties_l1957_195764


namespace NUMINAMATH_CALUDE_siblings_total_weight_l1957_195766

def total_weight (weight1 weight2 backpack1 backpack2 : ℕ) : ℕ :=
  weight1 + weight2 + backpack1 + backpack2

theorem siblings_total_weight :
  ∀ (antonio_weight antonio_sister_weight : ℕ),
    antonio_weight = 50 →
    antonio_sister_weight = antonio_weight - 12 →
    total_weight antonio_weight antonio_sister_weight 5 3 = 96 := by
  sorry

end NUMINAMATH_CALUDE_siblings_total_weight_l1957_195766


namespace NUMINAMATH_CALUDE_function_value_2023_l1957_195735

/-- A function f is even if f(-x) = f(x) for all x -/
def IsEven (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = f x

/-- The main theorem -/
theorem function_value_2023 (f : ℝ → ℝ) 
    (h_even : IsEven f)
    (h_not_zero : ∃ x, f x ≠ 0)
    (h_equation : ∀ x, x * f (x + 2) = (x + 2) * f x + 2) :
  f 2023 = -1 := by
sorry

end NUMINAMATH_CALUDE_function_value_2023_l1957_195735


namespace NUMINAMATH_CALUDE_toucan_count_l1957_195716

theorem toucan_count (initial_toucans : ℕ) (joining_toucans : ℕ) : 
  initial_toucans = 2 → joining_toucans = 1 → initial_toucans + joining_toucans = 3 := by
  sorry

end NUMINAMATH_CALUDE_toucan_count_l1957_195716


namespace NUMINAMATH_CALUDE_arithmetic_sequence_problem_l1957_195711

/-- An arithmetic sequence -/
def arithmeticSequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

/-- Theorem: In an arithmetic sequence where a₂ = 2 and a₆ = 10, a₁₀ = 18 -/
theorem arithmetic_sequence_problem (a : ℕ → ℝ) 
    (h_arith : arithmeticSequence a) 
    (h_a2 : a 2 = 2) 
    (h_a6 : a 6 = 10) : 
  a 10 = 18 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_problem_l1957_195711


namespace NUMINAMATH_CALUDE_investment_scientific_notation_l1957_195705

-- Define the total investment in yuan
def total_investment : ℝ := 7.7e9

-- Define the scientific notation representation
def scientific_notation : ℝ := 7.7 * (10 ^ 9)

-- Theorem statement
theorem investment_scientific_notation : total_investment = scientific_notation := by
  sorry

end NUMINAMATH_CALUDE_investment_scientific_notation_l1957_195705


namespace NUMINAMATH_CALUDE_min_boys_is_two_l1957_195702

/-- Represents the number of apples collected by a boy -/
inductive AppleCount
  | fixed : AppleCount  -- Represents 20 apples
  | percentage : AppleCount  -- Represents 20% of the total

/-- Represents a group of boys collecting apples -/
structure AppleCollection where
  boys : ℕ  -- Number of boys
  fixed_count : ℕ  -- Number of boys collecting fixed amount
  percentage_count : ℕ  -- Number of boys collecting percentage
  total_apples : ℕ  -- Total number of apples collected

/-- Checks if an AppleCollection is valid according to the problem conditions -/
def is_valid_collection (c : AppleCollection) : Prop :=
  c.boys = c.fixed_count + c.percentage_count ∧
  c.fixed_count > 0 ∧
  c.percentage_count > 0 ∧
  c.total_apples = 20 * c.fixed_count + (c.total_apples / 5) * c.percentage_count

/-- The main theorem stating that 2 is the minimum number of boys -/
theorem min_boys_is_two :
  ∀ c : AppleCollection, is_valid_collection c → c.boys ≥ 2 :=
sorry

end NUMINAMATH_CALUDE_min_boys_is_two_l1957_195702


namespace NUMINAMATH_CALUDE_garden_perimeter_l1957_195727

theorem garden_perimeter (garden_width playground_length : ℝ) 
  (h1 : garden_width = 4)
  (h2 : playground_length = 16)
  (h3 : ∃ (garden_length playground_width : ℝ), 
    garden_width * garden_length = playground_length * playground_width)
  (h4 : ∃ (garden_length : ℝ), 2 * (garden_width + garden_length) = 104) :
  ∃ (garden_length : ℝ), 2 * (garden_width + garden_length) = 104 :=
by
  sorry

#check garden_perimeter

end NUMINAMATH_CALUDE_garden_perimeter_l1957_195727


namespace NUMINAMATH_CALUDE_triangle_proof_l1957_195710

-- Define the triangle ABC
structure Triangle where
  A : Real
  B : Real
  C : Real
  a : Real
  b : Real
  c : Real

-- Define the properties of the triangle
def scalene_triangle (t : Triangle) : Prop :=
  t.a ≠ t.b ∧ t.b ≠ t.c ∧ t.c ≠ t.a

-- Define the given conditions
def triangle_conditions (t : Triangle) : Prop :=
  t.a = 3 ∧ t.c = 4 ∧ t.C = 2 * t.A

-- Theorem statement
theorem triangle_proof (t : Triangle) 
  (h1 : scalene_triangle t) 
  (h2 : triangle_conditions t) : 
  Real.cos t.A = 2/3 ∧ 
  t.b = 7/3 ∧ 
  Real.cos (2 * t.A + Real.pi/6) = -(Real.sqrt 3 + 4 * Real.sqrt 5)/18 := by
  sorry

end NUMINAMATH_CALUDE_triangle_proof_l1957_195710


namespace NUMINAMATH_CALUDE_power_subtraction_division_l1957_195744

theorem power_subtraction_division (n : ℕ) : 1^567 - 3^8 / 3^5 = -26 := by
  sorry

end NUMINAMATH_CALUDE_power_subtraction_division_l1957_195744


namespace NUMINAMATH_CALUDE_f_5_equals_102_l1957_195748

def f (x y : ℝ) : ℝ := 2 * x^2 + y

theorem f_5_equals_102 (y : ℝ) (some_value : ℝ) :
  f some_value y = 60 →
  f 5 y = 102 →
  f 5 y = 102 := by
  sorry

end NUMINAMATH_CALUDE_f_5_equals_102_l1957_195748


namespace NUMINAMATH_CALUDE_geometric_series_first_term_l1957_195729

theorem geometric_series_first_term 
  (a r : ℝ) 
  (h1 : 0 ≤ r ∧ r < 1) -- Condition for convergence of geometric series
  (h2 : a / (1 - r) = 20) -- Sum of terms is 20
  (h3 : a^2 / (1 - r^2) = 80) -- Sum of squares of terms is 80
  : a = 20 / 3 := by
sorry

end NUMINAMATH_CALUDE_geometric_series_first_term_l1957_195729


namespace NUMINAMATH_CALUDE_international_call_rate_l1957_195785

/-- Represents the cost and duration of phone calls -/
structure PhoneCall where
  localRate : ℚ
  localDuration : ℚ
  internationalDuration : ℚ
  totalCost : ℚ

/-- Calculates the cost per minute of an international call -/
def internationalRate (call : PhoneCall) : ℚ :=
  (call.totalCost - call.localRate * call.localDuration) / call.internationalDuration

/-- Theorem: Given the specified conditions, the international call rate is 25 cents per minute -/
theorem international_call_rate (call : PhoneCall) 
  (h1 : call.localRate = 5/100)
  (h2 : call.localDuration = 45)
  (h3 : call.internationalDuration = 31)
  (h4 : call.totalCost = 10) :
  internationalRate call = 25/100 := by
  sorry


end NUMINAMATH_CALUDE_international_call_rate_l1957_195785


namespace NUMINAMATH_CALUDE_project_completion_time_l1957_195794

/-- Represents the project completion time given the conditions -/
theorem project_completion_time 
  (total_man_days : ℝ) 
  (initial_workers : ℕ) 
  (workers_left : ℕ) 
  (h1 : total_man_days = 200)
  (h2 : initial_workers = 10)
  (h3 : workers_left = 4)
  : ∃ (x : ℝ), x > 0 ∧ x + (total_man_days - initial_workers * x) / (initial_workers - workers_left) = 40 := by
  sorry

#check project_completion_time

end NUMINAMATH_CALUDE_project_completion_time_l1957_195794


namespace NUMINAMATH_CALUDE_function_continuity_l1957_195754

-- Define the function f
variable (f : ℝ → ℝ)

-- Define the condition that f(x) + f(ax) is continuous for any a > 1
def condition (f : ℝ → ℝ) : Prop :=
  ∀ a : ℝ, a > 1 → Continuous (fun x ↦ f x + f (a * x))

-- State the theorem
theorem function_continuity (hf : condition f) : Continuous f := by
  sorry

end NUMINAMATH_CALUDE_function_continuity_l1957_195754


namespace NUMINAMATH_CALUDE_no_integer_solutions_binomial_power_l1957_195745

theorem no_integer_solutions_binomial_power (n k m t : ℕ) (l : ℕ) (h1 : l ≥ 2) (h2 : 4 ≤ k) (h3 : k ≤ n - 4) :
  Nat.choose n k ≠ m ^ t := by
  sorry

end NUMINAMATH_CALUDE_no_integer_solutions_binomial_power_l1957_195745


namespace NUMINAMATH_CALUDE_point_transformation_difference_l1957_195707

-- Define the rotation and reflection transformations
def rotate90CounterClockwise (center x : ℝ × ℝ) : ℝ × ℝ :=
  let (cx, cy) := center
  let (px, py) := x
  (cx - (py - cy), cy + (px - cx))

def reflectAboutYEqualNegX (p : ℝ × ℝ) : ℝ × ℝ :=
  let (x, y) := p
  (-y, -x)

-- State the theorem
theorem point_transformation_difference (a b : ℝ) :
  let p : ℝ × ℝ := (a, b)
  let rotated := rotate90CounterClockwise (2, 6) p
  let final := reflectAboutYEqualNegX rotated
  final = (-5, 2) → b - a = 15 := by
  sorry


end NUMINAMATH_CALUDE_point_transformation_difference_l1957_195707


namespace NUMINAMATH_CALUDE_complement_to_set_l1957_195721

def U : Finset Nat := {1,2,3,4,5,6,7,8}

theorem complement_to_set (B : Finset Nat) :
  (U \ B = {1,3}) → (B = {2,4,5,6,7,8}) := by
  sorry

end NUMINAMATH_CALUDE_complement_to_set_l1957_195721


namespace NUMINAMATH_CALUDE_leftover_bolts_count_l1957_195717

theorem leftover_bolts_count :
  let bolt_boxes : Nat := 7
  let bolts_per_box : Nat := 11
  let nut_boxes : Nat := 3
  let nuts_per_box : Nat := 15
  let used_bolts_and_nuts : Nat := 113
  let leftover_nuts : Nat := 6
  
  let total_bolts : Nat := bolt_boxes * bolts_per_box
  let total_nuts : Nat := nut_boxes * nuts_per_box
  let total_bolts_and_nuts : Nat := total_bolts + total_nuts
  let leftover_bolts_and_nuts : Nat := total_bolts_and_nuts - used_bolts_and_nuts
  let leftover_bolts : Nat := leftover_bolts_and_nuts - leftover_nuts

  leftover_bolts = 3 := by sorry

end NUMINAMATH_CALUDE_leftover_bolts_count_l1957_195717


namespace NUMINAMATH_CALUDE_sqrt_four_squared_times_five_to_sixth_l1957_195772

theorem sqrt_four_squared_times_five_to_sixth : Real.sqrt (4^2 * 5^6) = 500 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_four_squared_times_five_to_sixth_l1957_195772


namespace NUMINAMATH_CALUDE_plants_given_to_friend_l1957_195703

def initial_plants : ℕ := 3
def months : ℕ := 3
def remaining_plants : ℕ := 20

def plants_after_doubling (initial : ℕ) (months : ℕ) : ℕ :=
  initial * (2 ^ months)

theorem plants_given_to_friend :
  plants_after_doubling initial_plants months - remaining_plants = 4 := by
  sorry

end NUMINAMATH_CALUDE_plants_given_to_friend_l1957_195703


namespace NUMINAMATH_CALUDE_u_plus_v_value_l1957_195755

theorem u_plus_v_value (u v : ℚ) 
  (eq1 : 3 * u + 7 * v = 17)
  (eq2 : 5 * u - 3 * v = 9) :
  u + v = 43 / 11 := by
sorry

end NUMINAMATH_CALUDE_u_plus_v_value_l1957_195755


namespace NUMINAMATH_CALUDE_total_skips_five_throws_l1957_195787

-- Define the skip function
def S (n : ℕ) : ℕ := n^2 + n

-- Define the sum of skips from 1 to n
def sum_skips (n : ℕ) : ℕ :=
  (List.range n).map S |> List.sum

-- Theorem statement
theorem total_skips_five_throws :
  sum_skips 5 = 70 := by sorry

end NUMINAMATH_CALUDE_total_skips_five_throws_l1957_195787


namespace NUMINAMATH_CALUDE_range_of_m_l1957_195706

-- Define p and q as predicates on real numbers
def p (x : ℝ) : Prop := |x - 3| ≤ 2
def q (x m : ℝ) : Prop := (x - m + 1) * (x - m - 1) ≤ 0

-- Define the theorem
theorem range_of_m :
  (∀ x m : ℝ, (¬(p x) → ¬(q x m)) ∧ ¬(¬(q x m) → ¬(p x))) →
  ∃ a b : ℝ, a = 2 ∧ b = 4 ∧ ∀ m : ℝ, a ≤ m ∧ m ≤ b :=
by sorry

end NUMINAMATH_CALUDE_range_of_m_l1957_195706


namespace NUMINAMATH_CALUDE_min_overlap_social_media_l1957_195724

/-- The minimum percentage of adults using both Facebook and Instagram -/
theorem min_overlap_social_media (facebook_users instagram_users : ℝ) 
  (h1 : facebook_users = 85)
  (h2 : instagram_users = 75) :
  (facebook_users + instagram_users) - 100 = 60 := by
  sorry

end NUMINAMATH_CALUDE_min_overlap_social_media_l1957_195724


namespace NUMINAMATH_CALUDE_relationship_abc_l1957_195795

theorem relationship_abc : 
  let a : ℝ := 1.1 * Real.log 1.1
  let b : ℝ := 0.1 * Real.exp 0.1
  let c : ℝ := 1 / 9
  a < b ∧ b < c := by sorry

end NUMINAMATH_CALUDE_relationship_abc_l1957_195795


namespace NUMINAMATH_CALUDE_pentagon_area_l1957_195752

/-- Given points on a coordinate plane, prove the area of the pentagon formed by these points and their intersection. -/
theorem pentagon_area (A B D E C : ℝ × ℝ) : 
  A = (9, 1) →
  B = (2, 0) →
  D = (1, 5) →
  E = (9, 7) →
  (C.1 - A.1) / (D.1 - A.1) = (C.2 - A.2) / (D.2 - A.2) →
  (C.1 - B.1) / (E.1 - B.1) = (C.2 - B.2) / (E.2 - B.2) →
  abs ((A.1 * B.2 + B.1 * C.2 + C.1 * D.2 + D.1 * E.2 + E.1 * A.2) -
       (A.2 * B.1 + B.2 * C.1 + C.2 * D.1 + D.2 * E.1 + E.2 * A.1)) / 2 = 33 :=
by sorry

end NUMINAMATH_CALUDE_pentagon_area_l1957_195752


namespace NUMINAMATH_CALUDE_three_heads_one_tail_probability_three_heads_one_tail_probability_proof_l1957_195776

/-- The probability of getting exactly three heads and one tail when four fair coins are tossed simultaneously -/
theorem three_heads_one_tail_probability : ℝ :=
  1 / 4

/-- Proof that the probability of getting exactly three heads and one tail when four fair coins are tossed simultaneously is 1/4 -/
theorem three_heads_one_tail_probability_proof :
  three_heads_one_tail_probability = 1 / 4 := by
  sorry

end NUMINAMATH_CALUDE_three_heads_one_tail_probability_three_heads_one_tail_probability_proof_l1957_195776


namespace NUMINAMATH_CALUDE_truck_toll_calculation_l1957_195742

/-- Calculates the total toll for a truck crossing a bridge --/
def calculate_total_toll (B A1 A2 X1 X2 w : ℚ) (is_peak_hour : Bool) : ℚ :=
  let T := B + A1 * (X1 - 2) + A2 * X2
  let F := if w > 10000 then 0.1 * (w - 10000) else 0
  let total_without_surcharge := T + F
  let S := if is_peak_hour then 0.02 * total_without_surcharge else 0
  total_without_surcharge + S

theorem truck_toll_calculation :
  let B : ℚ := 0.50
  let A1 : ℚ := 0.75
  let A2 : ℚ := 0.50
  let X1 : ℚ := 1  -- One axle with 2 wheels
  let X2 : ℚ := 4  -- Four axles with 4 wheels each
  let w : ℚ := 12000
  let is_peak_hour : Bool := true  -- 9 AM is during peak hours
  calculate_total_toll B A1 A2 X1 X2 w is_peak_hour = 205.79 := by
  sorry


end NUMINAMATH_CALUDE_truck_toll_calculation_l1957_195742


namespace NUMINAMATH_CALUDE_f_derivative_l1957_195786

noncomputable def f (x : ℝ) : ℝ := Real.exp (2 * x) * Real.cos x

theorem f_derivative :
  deriv f = fun x => Real.exp (2 * x) * (2 * Real.cos x - Real.sin x) := by
  sorry

end NUMINAMATH_CALUDE_f_derivative_l1957_195786


namespace NUMINAMATH_CALUDE_factorization_equality_l1957_195770

theorem factorization_equality (m n : ℝ) : 2*m*n^2 - 12*m*n + 18*m = 2*m*(n-3)^2 := by
  sorry

end NUMINAMATH_CALUDE_factorization_equality_l1957_195770


namespace NUMINAMATH_CALUDE_sum_of_roots_bound_l1957_195791

/-- Given a quadratic equation x^2 - 2(1-k)x + k^2 = 0 with real roots α and β, 
    the sum of these roots α + β is greater than or equal to 1. -/
theorem sum_of_roots_bound (k : ℝ) (α β : ℝ) : 
  (∀ x, x^2 - 2*(1-k)*x + k^2 = 0 ↔ x = α ∨ x = β) →
  α + β ≥ 1 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_roots_bound_l1957_195791


namespace NUMINAMATH_CALUDE_solution_set_of_inequality_l1957_195746

theorem solution_set_of_inequality (x : ℝ) :
  (1 ≤ |x + 2| ∧ |x + 2| ≤ 5) ↔ ((-7 ≤ x ∧ x ≤ -3) ∨ (-1 ≤ x ∧ x ≤ 3)) :=
by sorry

end NUMINAMATH_CALUDE_solution_set_of_inequality_l1957_195746


namespace NUMINAMATH_CALUDE_product_of_specific_numbers_l1957_195782

theorem product_of_specific_numbers : 469157 * 9999 = 4690872843 := by
  sorry

end NUMINAMATH_CALUDE_product_of_specific_numbers_l1957_195782


namespace NUMINAMATH_CALUDE_exists_q_no_zeros_in_decimal_l1957_195740

theorem exists_q_no_zeros_in_decimal : ∃ q : ℚ, ∃ a : ℕ, (
  (q * 2^1000 = a) ∧
  (∀ d : ℕ, d < 10 → (a.digits 10).contains d → (d = 1 ∨ d = 2))
) := by sorry

end NUMINAMATH_CALUDE_exists_q_no_zeros_in_decimal_l1957_195740


namespace NUMINAMATH_CALUDE_winning_strategy_works_l1957_195765

/-- Represents a player in the coin game -/
inductive Player : Type
| One : Player
| Two : Player

/-- The game state -/
structure GameState :=
  (coins : ℕ)
  (currentPlayer : Player)

/-- Valid moves for each player -/
def validMove (player : Player) (n : ℕ) : Prop :=
  match player with
  | Player.One => n % 2 = 1 ∧ 1 ≤ n ∧ n ≤ 99
  | Player.Two => n % 2 = 0 ∧ 2 ≤ n ∧ n ≤ 100

/-- The winning strategy function -/
def winningStrategy (state : GameState) : Option ℕ :=
  match state.currentPlayer with
  | Player.One => 
    if state.coins > 95 then some 95
    else if state.coins % 101 ≠ 0 then some (state.coins % 101)
    else none
  | Player.Two => none

/-- The main theorem -/
theorem winning_strategy_works : 
  ∀ (state : GameState), 
    state.coins = 2015 → 
    state.currentPlayer = Player.One → 
    ∃ (move : ℕ), 
      validMove Player.One move ∧ 
      move = 95 ∧
      ∀ (opponentMove : ℕ), 
        validMove Player.Two opponentMove → 
        ∃ (nextMove : ℕ), 
          validMove Player.One nextMove ∧ 
          state.coins - move - opponentMove - nextMove ≡ 0 [MOD 101] :=
sorry

#check winning_strategy_works

end NUMINAMATH_CALUDE_winning_strategy_works_l1957_195765


namespace NUMINAMATH_CALUDE_olympiad_participants_impossibility_l1957_195726

theorem olympiad_participants_impossibility : ¬ ∃ (x : ℕ), x + (x + 43) = 1000 := by
  sorry

end NUMINAMATH_CALUDE_olympiad_participants_impossibility_l1957_195726


namespace NUMINAMATH_CALUDE_race_time_calculation_l1957_195741

/-- Given that Prejean's speed is three-quarters of Rickey's speed and Rickey took 40 minutes to finish a race, 
    prove that the total time taken by both runners is 40 + 40 * (4/3) minutes. -/
theorem race_time_calculation (rickey_speed rickey_time prejean_speed : ℝ) 
    (h1 : rickey_time = 40)
    (h2 : prejean_speed = 3/4 * rickey_speed) : 
  rickey_time + (rickey_time / (prejean_speed / rickey_speed)) = 40 + 40 * (4/3) := by
  sorry

end NUMINAMATH_CALUDE_race_time_calculation_l1957_195741


namespace NUMINAMATH_CALUDE_p_necessary_not_sufficient_for_q_l1957_195778

-- Define the conditions
def p (x : ℝ) : Prop := x < -1 ∨ x > 1
def q (x : ℝ) : Prop := x < -2

-- Theorem statement
theorem p_necessary_not_sufficient_for_q :
  (∀ x, q x → p x) ∧ (∃ x, p x ∧ ¬q x) := by
  sorry

end NUMINAMATH_CALUDE_p_necessary_not_sufficient_for_q_l1957_195778


namespace NUMINAMATH_CALUDE_larger_number_l1957_195789

theorem larger_number (x y : ℝ) (h1 : y = 2 * x - 3) (h2 : x + y = 51) : y = 33 := by
  sorry

end NUMINAMATH_CALUDE_larger_number_l1957_195789


namespace NUMINAMATH_CALUDE_find_A_value_l1957_195777

theorem find_A_value : ∃! A : ℕ, ∃ B : ℕ, 
  (A < 10 ∧ B < 10) ∧ 
  (500 + 10 * A + 8) - (100 * B + 14) = 364 :=
by
  -- The proof would go here
  sorry

end NUMINAMATH_CALUDE_find_A_value_l1957_195777


namespace NUMINAMATH_CALUDE_unique_congruence_solution_l1957_195774

theorem unique_congruence_solution :
  ∃! n : ℤ, 0 ≤ n ∧ n < 19 ∧ -200 ≡ n [ZMOD 19] ∧ n = 9 := by
  sorry

end NUMINAMATH_CALUDE_unique_congruence_solution_l1957_195774


namespace NUMINAMATH_CALUDE_family_event_handshakes_l1957_195709

/-- The number of sets of twins at the family event -/
def twin_sets : ℕ := 12

/-- The number of sets of triplets at the family event -/
def triplet_sets : ℕ := 4

/-- The total number of twins at the family event -/
def total_twins : ℕ := twin_sets * 2

/-- The total number of triplets at the family event -/
def total_triplets : ℕ := triplet_sets * 3

/-- The fraction of triplets each twin shakes hands with -/
def twin_triplet_fraction : ℚ := 1 / 3

/-- The fraction of twins each triplet shakes hands with -/
def triplet_twin_fraction : ℚ := 2 / 3

/-- The total number of unique handshakes at the family event -/
def total_handshakes : ℕ := 462

theorem family_event_handshakes :
  (total_twins * (total_twins - 2) / 2) +
  (total_triplets * (total_triplets - 3) / 2) +
  ((total_twins * (total_triplets * twin_triplet_fraction).floor +
    total_triplets * (total_twins * triplet_twin_fraction).floor) / 2) =
  total_handshakes := by sorry

end NUMINAMATH_CALUDE_family_event_handshakes_l1957_195709


namespace NUMINAMATH_CALUDE_sphere_cap_cone_volume_equality_l1957_195790

theorem sphere_cap_cone_volume_equality (R : ℝ) (R_pos : R > 0) :
  ∃ x : ℝ, x > 0 ∧ x < R ∧
  (2 / 3 * R^2 * π * (R - x) = 1 / 3 * (R^2 - x^2) * π * x) ∧
  x = R * (Real.sqrt 5 - 1) / 2 :=
by sorry

end NUMINAMATH_CALUDE_sphere_cap_cone_volume_equality_l1957_195790


namespace NUMINAMATH_CALUDE_inequality_preservation_l1957_195753

theorem inequality_preservation (a b : ℝ) (h : a > b) : a + 1 > b + 1 := by
  sorry

end NUMINAMATH_CALUDE_inequality_preservation_l1957_195753


namespace NUMINAMATH_CALUDE_rectangle_perimeter_l1957_195704

theorem rectangle_perimeter (a b : ℕ) : 
  b = 3 * a →                   -- One side is three times as long as the other
  a * b = 2 * (a + b) + 12 →    -- Area equals perimeter plus 12
  2 * (a + b) = 32              -- Perimeter is 32 units
  := by sorry

end NUMINAMATH_CALUDE_rectangle_perimeter_l1957_195704


namespace NUMINAMATH_CALUDE_x_plus_y_value_l1957_195797

theorem x_plus_y_value (x y : ℝ) (hx : |x| = 3) (hy : |y| = 5) (hxy : x * y < 0) :
  x + y = 2 ∨ x + y = -2 := by
sorry

end NUMINAMATH_CALUDE_x_plus_y_value_l1957_195797


namespace NUMINAMATH_CALUDE_cylinder_unique_non_identical_views_l1957_195784

-- Define the types of solid objects
inductive SolidObject
  | Sphere
  | TriangularPyramid
  | Cube
  | Cylinder

-- Define a function that checks if all views are identical
def hasIdenticalViews (obj : SolidObject) : Prop :=
  match obj with
  | SolidObject.Sphere => True
  | SolidObject.TriangularPyramid => False
  | SolidObject.Cube => True
  | SolidObject.Cylinder => False

-- Theorem statement
theorem cylinder_unique_non_identical_views :
  ∀ (obj : SolidObject), ¬(hasIdenticalViews obj) ↔ obj = SolidObject.Cylinder :=
by sorry

end NUMINAMATH_CALUDE_cylinder_unique_non_identical_views_l1957_195784


namespace NUMINAMATH_CALUDE_complex_magnitude_example_l1957_195783

theorem complex_magnitude_example : Complex.abs (12 - 5*Complex.I) = 13 := by
  sorry

end NUMINAMATH_CALUDE_complex_magnitude_example_l1957_195783


namespace NUMINAMATH_CALUDE_value_of_expression_l1957_195780

theorem value_of_expression (x y z : ℝ) 
  (h1 : 3 * x - 4 * y - 2 * z = 0)
  (h2 : x - 2 * y - 8 * z = 0)
  (h3 : z ≠ 0) :
  (x^2 + 3*x*y) / (y^2 + z^2) = 329/61 := by
  sorry

end NUMINAMATH_CALUDE_value_of_expression_l1957_195780


namespace NUMINAMATH_CALUDE_six_digit_palindrome_divisible_by_11_l1957_195775

theorem six_digit_palindrome_divisible_by_11 (a b c : ℕ) (h1 : a < 10) (h2 : b < 10) (h3 : c < 10) :
  let W := 100000 * a + 10000 * b + 1000 * c + 100 * c + 10 * b + a
  11 ∣ W :=
by sorry

end NUMINAMATH_CALUDE_six_digit_palindrome_divisible_by_11_l1957_195775


namespace NUMINAMATH_CALUDE_data_analysis_l1957_195796

def data : List ℝ := [11, 10, 11, 13, 11, 13, 15]

def mode (l : List ℝ) : ℝ := sorry

def mean (l : List ℝ) : ℝ := sorry

def variance (l : List ℝ) : ℝ := sorry

def median (l : List ℝ) : ℝ := sorry

theorem data_analysis :
  mode data = 11 ∧
  mean data = 12 ∧
  variance data = 18/7 ∧
  median data = 11 := by sorry

end NUMINAMATH_CALUDE_data_analysis_l1957_195796


namespace NUMINAMATH_CALUDE_cone_sphere_volume_ratio_l1957_195736

/-- Given a sphere and a right circular cone where:
    1. The cone's height is twice the sphere's radius
    2. The cone's base radius is equal to the sphere's radius
    Prove that the ratio of the cone's volume to the sphere's volume is 1/2 -/
theorem cone_sphere_volume_ratio (r : ℝ) (h : ℝ) (h_pos : r > 0) (h_eq : h = 2 * r) :
  (1 / 3 * π * r^2 * h) / (4 / 3 * π * r^3) = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_cone_sphere_volume_ratio_l1957_195736


namespace NUMINAMATH_CALUDE_max_b_value_l1957_195799

theorem max_b_value (a b : ℤ) : 
  (a + b)^2 + a*(a + b) + b = 0 →
  b ≤ 9 ∧ ∃ (a₀ b₀ : ℤ), (a₀ + b₀)^2 + a₀*(a₀ + b₀) + b₀ = 0 ∧ b₀ = 9 :=
by sorry

end NUMINAMATH_CALUDE_max_b_value_l1957_195799


namespace NUMINAMATH_CALUDE_algebraic_expression_value_l1957_195723

theorem algebraic_expression_value
  (m n p q x : ℝ)
  (h1 : m = -n)
  (h2 : p * q = 1)
  (h3 : |x| = 2) :
  (m + n) / 2022 + 2023 * p * q + x^2 = 2027 :=
by sorry

end NUMINAMATH_CALUDE_algebraic_expression_value_l1957_195723


namespace NUMINAMATH_CALUDE_inequality_proof_l1957_195725

theorem inequality_proof (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) 
  (h : x * y * z ≥ x * y + y * z + z * x) : 
  Real.sqrt (x * y * z) ≥ Real.sqrt x + Real.sqrt y + Real.sqrt z := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l1957_195725


namespace NUMINAMATH_CALUDE_trigonometric_factorization_l1957_195714

theorem trigonometric_factorization (x : Real) :
  1 - Real.sin x ^ 5 - Real.cos x ^ 5 =
  (1 - Real.sin x) * (1 - Real.cos x) *
  (3 + 2 * (Real.sin x + Real.cos x) + 2 * Real.sin x * Real.cos x +
   Real.sin x * Real.cos x * (Real.sin x + Real.cos x)) := by
  sorry

end NUMINAMATH_CALUDE_trigonometric_factorization_l1957_195714


namespace NUMINAMATH_CALUDE_cone_base_diameter_l1957_195700

theorem cone_base_diameter (r : ℝ) (h1 : r > 0) : 
  (π * r^2 + π * r * (2 * r) = 3 * π) → 
  (2 * r = 2) := by
sorry

end NUMINAMATH_CALUDE_cone_base_diameter_l1957_195700


namespace NUMINAMATH_CALUDE_simplify_expression_l1957_195767

theorem simplify_expression (y : ℝ) : 7 * y - 3 * y + 9 + 15 = 4 * y + 24 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l1957_195767


namespace NUMINAMATH_CALUDE_quadratic_always_has_real_roots_min_a_for_positive_integer_roots_min_a_is_zero_l1957_195708

/-- The quadratic equation x^2 - (a+2)x + (a+1) = 0 -/
def quadratic (a : ℝ) (x : ℝ) : ℝ := x^2 - (a+2)*x + (a+1)

/-- The discriminant of the quadratic equation -/
def discriminant (a : ℝ) : ℝ := (a+2)^2 - 4*(a+1)

theorem quadratic_always_has_real_roots (a : ℝ) :
  discriminant a ≥ 0 := by sorry

theorem min_a_for_positive_integer_roots :
  ∀ a : ℕ, (∃ x y : ℕ, x ≠ y ∧ quadratic a x = 0 ∧ quadratic a y = 0) →
  a ≥ 0 := by sorry

theorem min_a_is_zero :
  ∃ a : ℕ, a = 0 ∧
  (∃ x y : ℕ, x ≠ y ∧ quadratic a x = 0 ∧ quadratic a y = 0) ∧
  ∀ b : ℕ, b < a →
  ¬(∃ x y : ℕ, x ≠ y ∧ quadratic b x = 0 ∧ quadratic b y = 0) := by sorry

end NUMINAMATH_CALUDE_quadratic_always_has_real_roots_min_a_for_positive_integer_roots_min_a_is_zero_l1957_195708


namespace NUMINAMATH_CALUDE_garrison_size_l1957_195761

/-- Represents the initial number of men in the garrison -/
def initial_men : ℕ := 150

/-- Represents the number of days the provisions were initially meant to last -/
def initial_days : ℕ := 31

/-- Represents the number of days that passed before reinforcements arrived -/
def days_before_reinforcement : ℕ := 16

/-- Represents the number of reinforcement men that arrived -/
def reinforcement_men : ℕ := 300

/-- Represents the number of days the provisions lasted after reinforcements arrived -/
def remaining_days : ℕ := 5

theorem garrison_size :
  initial_men * initial_days = 
  initial_men * (initial_days - days_before_reinforcement) ∧
  initial_men * (initial_days - days_before_reinforcement) = 
  (initial_men + reinforcement_men) * remaining_days :=
by sorry

end NUMINAMATH_CALUDE_garrison_size_l1957_195761


namespace NUMINAMATH_CALUDE_min_games_for_20_teams_l1957_195756

/-- Represents a football tournament --/
structure Tournament where
  num_teams : ℕ
  num_games : ℕ

/-- Checks if a tournament satisfies the condition that among any three teams, 
    two have played against each other --/
def satisfies_condition (t : Tournament) : Prop :=
  ∀ (a b c : Fin t.num_teams), a ≠ b ∧ b ≠ c ∧ a ≠ c → 
    (∃ (x y : Fin t.num_teams), x ≠ y ∧ ((x = a ∧ y = b) ∨ (x = b ∧ y = c) ∨ (x = a ∧ y = c)))

/-- The main theorem stating the minimum number of games required --/
theorem min_games_for_20_teams : 
  ∃ (t : Tournament), t.num_teams = 20 ∧ t.num_games = 90 ∧ 
    satisfies_condition t ∧ 
    (∀ (t' : Tournament), t'.num_teams = 20 ∧ satisfies_condition t' → t'.num_games ≥ 90) :=
sorry

end NUMINAMATH_CALUDE_min_games_for_20_teams_l1957_195756


namespace NUMINAMATH_CALUDE_arithmetic_sequence_30th_term_l1957_195743

/-- An arithmetic sequence is a sequence where the difference between
    any two consecutive terms is constant. -/
def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

/-- The 30th term of the given arithmetic sequence is 351. -/
theorem arithmetic_sequence_30th_term
  (a : ℕ → ℝ)
  (h_arith : is_arithmetic_sequence a)
  (h_a1 : a 1 = 3)
  (h_a2 : a 2 = 15)
  (h_a3 : a 3 = 27) :
  a 30 = 351 :=
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_30th_term_l1957_195743


namespace NUMINAMATH_CALUDE_sum_of_powers_of_i_l1957_195739

theorem sum_of_powers_of_i : 
  let i : ℂ := Complex.I
  (i + 2 * i^2 + 3 * i^3 + 4 * i^4 + 5 * i^5 + 6 * i^6 + 7 * i^7 + 8 * i^8) = (4 : ℂ) - 4 * i :=
by sorry

end NUMINAMATH_CALUDE_sum_of_powers_of_i_l1957_195739


namespace NUMINAMATH_CALUDE_total_students_proof_l1957_195792

/-- Represents the total number of senior students -/
def total_students : ℕ := 300

/-- Represents the number of students who didn't receive scholarships -/
def no_scholarship_students : ℕ := 255

/-- Percentage of students who received full merit scholarships -/
def full_scholarship_percent : ℚ := 5 / 100

/-- Percentage of students who received half merit scholarships -/
def half_scholarship_percent : ℚ := 10 / 100

/-- Theorem stating that the total number of students is 300 given the scholarship distribution -/
theorem total_students_proof :
  (1 - full_scholarship_percent - half_scholarship_percent) * total_students = no_scholarship_students :=
sorry

end NUMINAMATH_CALUDE_total_students_proof_l1957_195792


namespace NUMINAMATH_CALUDE_digital_earth_properties_l1957_195771

-- Define the properties of Digital Earth
structure DigitalEarth where
  is_digitized : Bool
  meets_current_needs : Bool
  main_feature_virtual_reality : Bool
  uses_centralized_storage : Bool

-- Define the correct properties of Digital Earth
def correct_digital_earth : DigitalEarth := {
  is_digitized := true,
  meets_current_needs := false,
  main_feature_virtual_reality := true,
  uses_centralized_storage := false
}

-- Theorem stating the correct properties of Digital Earth
theorem digital_earth_properties :
  correct_digital_earth.is_digitized ∧
  correct_digital_earth.main_feature_virtual_reality ∧
  ¬correct_digital_earth.meets_current_needs ∧
  ¬correct_digital_earth.uses_centralized_storage :=
by sorry


end NUMINAMATH_CALUDE_digital_earth_properties_l1957_195771


namespace NUMINAMATH_CALUDE_mississippi_permutations_count_l1957_195712

/-- The number of unique permutations of MISSISSIPPI -/
def mississippi_permutations : ℕ :=
  Nat.factorial 11 / (Nat.factorial 1 * Nat.factorial 4 * Nat.factorial 4 * Nat.factorial 2)

/-- Theorem stating that the number of unique permutations of MISSISSIPPI is 34650 -/
theorem mississippi_permutations_count : mississippi_permutations = 34650 := by
  sorry

end NUMINAMATH_CALUDE_mississippi_permutations_count_l1957_195712


namespace NUMINAMATH_CALUDE_total_eggs_collected_l1957_195773

/-- The number of dozen eggs collected by Benjamin, Carla, Trisha, and David -/
def total_eggs (benjamin carla trisha david : ℕ) : ℕ :=
  benjamin + carla + trisha + david

/-- Theorem stating the total number of dozen eggs collected -/
theorem total_eggs_collected :
  ∃ (benjamin carla trisha david : ℕ),
    benjamin = 6 ∧
    carla = 3 * benjamin ∧
    trisha = benjamin - 4 ∧
    david = 2 * trisha ∧
    total_eggs benjamin carla trisha david = 30 := by
  sorry

end NUMINAMATH_CALUDE_total_eggs_collected_l1957_195773


namespace NUMINAMATH_CALUDE_contrapositive_equivalence_l1957_195781

theorem contrapositive_equivalence (m : ℕ+) : 
  (¬(∃ x : ℝ, x^2 + x - m.val = 0) → m.val ≤ 0) ↔ 
  (m.val > 0 → ∃ x : ℝ, x^2 + x - m.val = 0) :=
by sorry

end NUMINAMATH_CALUDE_contrapositive_equivalence_l1957_195781


namespace NUMINAMATH_CALUDE_triangle_area_l1957_195718

theorem triangle_area (a b c : ℝ) (h_right_angle : a^2 + b^2 = c^2) 
  (h_angle : a / c = 1 / 2) (h_hypotenuse : c = 40) : 
  (a * b) / 2 = 200 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_triangle_area_l1957_195718


namespace NUMINAMATH_CALUDE_area_of_rectangle_with_squares_l1957_195793

/-- A rectangle divided into four identical squares with a given perimeter -/
structure RectangleWithSquares where
  side_length : ℝ
  perimeter : ℝ
  perimeter_eq : perimeter = 8 * side_length

/-- The area of a rectangle divided into four identical squares -/
def area (r : RectangleWithSquares) : ℝ :=
  4 * r.side_length^2

/-- Theorem: A rectangle divided into four identical squares with a perimeter of 160 has an area of 1600 -/
theorem area_of_rectangle_with_squares (r : RectangleWithSquares) (h : r.perimeter = 160) :
  area r = 1600 := by
  sorry

#check area_of_rectangle_with_squares

end NUMINAMATH_CALUDE_area_of_rectangle_with_squares_l1957_195793


namespace NUMINAMATH_CALUDE_combinatorial_identity_l1957_195747

theorem combinatorial_identity (n : ℕ) : 
  (n.choose 2) * 2 = 42 → n.choose 3 = 35 := by
  sorry

end NUMINAMATH_CALUDE_combinatorial_identity_l1957_195747


namespace NUMINAMATH_CALUDE_triangle_properties_l1957_195798

/-- An acute triangle with side lengths a, b, c opposite to angles A, B, C respectively -/
structure AcuteTriangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ
  acute : A > 0 ∧ B > 0 ∧ C > 0 ∧ A + B + C = π
  law_of_sines : a / Real.sin A = b / Real.sin B ∧ b / Real.sin B = c / Real.sin C

/-- The main theorem about the specific triangle -/
theorem triangle_properties (t : AcuteTriangle)
    (h1 : t.a = 2)
    (h2 : 2 * Real.sin t.A = Real.sin t.C)
    (h3 : Real.cos t.C = 1/4) :
    t.c = 4 ∧ (1/2 * t.a * t.b * Real.sin t.C) = Real.sqrt 15 := by
  sorry

end NUMINAMATH_CALUDE_triangle_properties_l1957_195798


namespace NUMINAMATH_CALUDE_perpendicular_transitivity_perpendicular_parallel_planes_l1957_195757

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relations
variable (parallel : Line → Line → Prop)
variable (perpendicular : Line → Plane → Prop)
variable (plane_parallel : Plane → Plane → Prop)
variable (line_in_plane : Line → Plane → Prop)

-- Theorem 1
theorem perpendicular_transitivity 
  (m n : Line) (α : Plane) 
  (h1 : parallel m n) (h2 : perpendicular m α) : 
  perpendicular n α :=
sorry

-- Theorem 2
theorem perpendicular_parallel_planes 
  (m n : Line) (α β : Plane)
  (h1 : plane_parallel α β) (h2 : parallel m n) (h3 : perpendicular m α) :
  perpendicular n β :=
sorry

end NUMINAMATH_CALUDE_perpendicular_transitivity_perpendicular_parallel_planes_l1957_195757


namespace NUMINAMATH_CALUDE_intersection_points_form_hyperbola_l1957_195719

theorem intersection_points_form_hyperbola :
  ∀ (t x y : ℝ), 
    (2 * t * x - 3 * y - 4 * t = 0) → 
    (x - 3 * t * y + 4 = 0) → 
    (x^2 / 16 - y^2 = 1) :=
by sorry

end NUMINAMATH_CALUDE_intersection_points_form_hyperbola_l1957_195719


namespace NUMINAMATH_CALUDE_flower_production_percentage_l1957_195732

theorem flower_production_percentage
  (daisy_seeds sunflower_seeds : ℕ)
  (daisy_germination_rate sunflower_germination_rate : ℚ)
  (flowering_plants : ℕ)
  (h1 : daisy_seeds = 25)
  (h2 : sunflower_seeds = 25)
  (h3 : daisy_germination_rate = 0.6)
  (h4 : sunflower_germination_rate = 0.8)
  (h5 : flowering_plants = 28)
  : (flowering_plants : ℚ) / ((daisy_germination_rate * daisy_seeds + sunflower_germination_rate * sunflower_seeds) : ℚ) = 0.8 := by
  sorry

end NUMINAMATH_CALUDE_flower_production_percentage_l1957_195732


namespace NUMINAMATH_CALUDE_sum_of_digits_B_is_seven_l1957_195750

/-- Sum of digits function -/
def sumOfDigits (n : ℕ) : ℕ := sorry

/-- A is the sum of digits of 4444^4444 -/
def A : ℕ := sumOfDigits (4444^4444)

/-- B is the sum of digits of A -/
def B : ℕ := sumOfDigits A

/-- Theorem: The sum of digits of B is 7 -/
theorem sum_of_digits_B_is_seven : sumOfDigits B = 7 := by sorry

end NUMINAMATH_CALUDE_sum_of_digits_B_is_seven_l1957_195750


namespace NUMINAMATH_CALUDE_special_collection_books_l1957_195713

/-- The number of books in a special collection at the beginning of a month,
    given the number of books loaned, returned, and remaining at the end of the month. -/
theorem special_collection_books
  (loaned : ℕ)
  (return_rate : ℚ)
  (end_count : ℕ)
  (h1 : loaned = 30)
  (h2 : return_rate = 7/10)
  (h3 : end_count = 66)
  : ℕ := by
  sorry

end NUMINAMATH_CALUDE_special_collection_books_l1957_195713


namespace NUMINAMATH_CALUDE_z_plus_one_is_pure_imaginary_l1957_195762

theorem z_plus_one_is_pure_imaginary : 
  let z : ℂ := (-2 * Complex.I) / (1 + Complex.I)
  ∃ (y : ℝ), z + 1 = y * Complex.I :=
by
  sorry

end NUMINAMATH_CALUDE_z_plus_one_is_pure_imaginary_l1957_195762


namespace NUMINAMATH_CALUDE_book_and_painting_participants_l1957_195760

/-- Represents the number of participants in various activities and their intersections --/
structure ActivityParticipants where
  total : ℕ
  book_club : ℕ
  fun_sports : ℕ
  env_painting : ℕ
  book_and_sports : ℕ
  sports_and_painting : ℕ

/-- Theorem stating the number of participants in both Book Club and Environmental Theme Painting --/
theorem book_and_painting_participants (ap : ActivityParticipants)
  (h_total : ap.total = 120)
  (h_book : ap.book_club = 80)
  (h_sports : ap.fun_sports = 50)
  (h_painting : ap.env_painting = 40)
  (h_book_sports : ap.book_and_sports = 20)
  (h_sports_painting : ap.sports_and_painting = 10)
  (h_max_two : ∀ p, p ≤ 2) :
  ap.book_club + ap.fun_sports + ap.env_painting - ap.total - ap.book_and_sports - ap.sports_and_painting = 20 :=
by sorry

end NUMINAMATH_CALUDE_book_and_painting_participants_l1957_195760


namespace NUMINAMATH_CALUDE_paint_coverage_l1957_195759

/-- Given a cube with 10-foot edges, if it costs $16 to paint the entire outside surface
    of the cube and paint costs $3.20 per quart, then one quart of paint covers 120 square feet. -/
theorem paint_coverage (cube_edge : Real) (total_cost : Real) (paint_cost_per_quart : Real) :
  cube_edge = 10 ∧ total_cost = 16 ∧ paint_cost_per_quart = 3.2 →
  (6 * cube_edge^2) / (total_cost / paint_cost_per_quart) = 120 := by
  sorry

end NUMINAMATH_CALUDE_paint_coverage_l1957_195759


namespace NUMINAMATH_CALUDE_weekly_rate_is_190_l1957_195749

/-- Represents the car rental problem --/
structure CarRental where
  dailyRate : ℕ
  totalDays : ℕ
  totalCost : ℕ
  weeklyRate : ℕ

/-- The car rental agency's pricing policy --/
def rentalPolicy (r : CarRental) : Prop :=
  r.dailyRate = 30 ∧
  r.totalDays = 11 ∧
  r.totalCost = 310 ∧
  r.weeklyRate = r.totalCost - (r.totalDays - 7) * r.dailyRate

/-- Theorem stating that the weekly rate is $190 --/
theorem weekly_rate_is_190 (r : CarRental) :
  rentalPolicy r → r.weeklyRate = 190 := by
  sorry

#check weekly_rate_is_190

end NUMINAMATH_CALUDE_weekly_rate_is_190_l1957_195749


namespace NUMINAMATH_CALUDE_parabola_smallest_a_l1957_195720

/-- Given a parabola with vertex (1/2, -5/4), equation y = ax^2 + bx + c,
    a > 0, and directrix y = -2, prove that the smallest possible value of a is 2/3 -/
theorem parabola_smallest_a (a b c : ℝ) :
  (∀ x y : ℝ, y = a * x^2 + b * x + c) →  -- Equation of parabola
  (a > 0) →                               -- a is positive
  (∀ x : ℝ, a * (x - 1/2)^2 - 5/4 = a * x^2 + b * x + c) →  -- Vertex form
  (∀ x : ℝ, -2 = a * x^2 + b * x + c - 3/4 * (1/a)) →       -- Directrix equation
  a = 2/3 :=
by sorry

end NUMINAMATH_CALUDE_parabola_smallest_a_l1957_195720


namespace NUMINAMATH_CALUDE_cistern_filling_time_l1957_195788

theorem cistern_filling_time (x : ℝ) 
  (h1 : x > 0)
  (h2 : 1/x + 1/12 - 1/15 = 7/60) : x = 10 := by
  sorry

end NUMINAMATH_CALUDE_cistern_filling_time_l1957_195788


namespace NUMINAMATH_CALUDE_no_solution_for_system_l1957_195734

theorem no_solution_for_system :
  ¬ ∃ (x y z : ℝ), x > 0 ∧ y > 0 ∧ z > 0 ∧
    x^(1/3) - y^(1/3) - z^(1/3) = 64 ∧
    x^(1/4) - y^(1/4) - z^(1/4) = 32 ∧
    x^(1/6) - y^(1/6) - z^(1/6) = 8 :=
by sorry

end NUMINAMATH_CALUDE_no_solution_for_system_l1957_195734


namespace NUMINAMATH_CALUDE_large_rectangle_perimeter_l1957_195737

/-- Represents a rectangle with length and width -/
structure Rectangle where
  length : ℝ
  width : ℝ

/-- Represents a square -/
structure Square where
  side : ℝ

/-- Calculates the perimeter of a rectangle -/
def Rectangle.perimeter (r : Rectangle) : ℝ := 2 * (r.length + r.width)

/-- Calculates the perimeter of a square -/
def Square.perimeter (s : Square) : ℝ := 4 * s.side

/-- Theorem: The perimeter of a large rectangle composed of three identical squares
    and three identical small rectangles is 52, given the conditions. -/
theorem large_rectangle_perimeter : 
  ∀ (s : Square) (r : Rectangle),
    s.perimeter = 24 →
    r.perimeter = 16 →
    (3 * s.side = r.length) →
    (s.side + r.width = 3 * r.length + 3 * r.width) →
    Rectangle.perimeter { length := 3 * s.side, width := s.side + r.width } = 52 := by
  sorry

end NUMINAMATH_CALUDE_large_rectangle_perimeter_l1957_195737


namespace NUMINAMATH_CALUDE_library_visitors_l1957_195730

theorem library_visitors (non_sunday_avg : ℕ) (monthly_avg : ℕ) (month_days : ℕ) (sundays : ℕ) :
  non_sunday_avg = 240 →
  monthly_avg = 285 →
  month_days = 30 →
  sundays = 5 →
  (sundays * (monthly_avg * month_days - non_sunday_avg * (month_days - sundays))) / sundays = 510 := by
  sorry

end NUMINAMATH_CALUDE_library_visitors_l1957_195730


namespace NUMINAMATH_CALUDE_solution_to_linear_equation_l1957_195758

theorem solution_to_linear_equation :
  ∃ x : ℝ, 3 * x - 6 = 0 ∧ x = 2 := by
  sorry

end NUMINAMATH_CALUDE_solution_to_linear_equation_l1957_195758
