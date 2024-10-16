import Mathlib

namespace NUMINAMATH_CALUDE_combined_time_calculation_l3708_370879

/-- The time taken by the car to reach station B -/
def car_time : ℝ := 4.5

/-- The additional time taken by the train compared to the car -/
def train_additional_time : ℝ := 2

/-- The time taken by the train to reach station B -/
def train_time : ℝ := car_time + train_additional_time

/-- The combined time taken by both the car and the train to reach station B -/
def combined_time : ℝ := car_time + train_time

theorem combined_time_calculation : combined_time = 11 := by sorry

end NUMINAMATH_CALUDE_combined_time_calculation_l3708_370879


namespace NUMINAMATH_CALUDE_female_students_count_l3708_370867

theorem female_students_count (total_students sample_size male_in_sample : ℕ) 
  (h1 : total_students = 1600)
  (h2 : sample_size = 200)
  (h3 : male_in_sample = 110) : 
  total_students - (total_students * male_in_sample / sample_size) = 720 := by
  sorry

end NUMINAMATH_CALUDE_female_students_count_l3708_370867


namespace NUMINAMATH_CALUDE_latus_rectum_of_parabola_l3708_370848

/-- Given a parabola with equation x = 4y², prove that its latus rectum has the equation x = -1/16 -/
theorem latus_rectum_of_parabola (y : ℝ) :
  let x := 4 * y^2
  (∃ p : ℝ, p = 1/8 ∧ x = -p) → x = -1/16 := by
  sorry

end NUMINAMATH_CALUDE_latus_rectum_of_parabola_l3708_370848


namespace NUMINAMATH_CALUDE_initial_amount_proof_l3708_370862

/-- 
Proves that if an amount increases by 1/8th of itself each year for two years 
and becomes 72900, then the initial amount was 57600.
-/
theorem initial_amount_proof (P : ℝ) : 
  (((P + P / 8) + (P + P / 8) / 8) = 72900) → P = 57600 :=
by sorry

end NUMINAMATH_CALUDE_initial_amount_proof_l3708_370862


namespace NUMINAMATH_CALUDE_complex_division_result_l3708_370864

theorem complex_division_result : (5 - I) / (1 - I) = 3 + 2*I := by
  sorry

end NUMINAMATH_CALUDE_complex_division_result_l3708_370864


namespace NUMINAMATH_CALUDE_quadratic_root_value_l3708_370865

theorem quadratic_root_value (x : ℝ) : x = -4 → Real.sqrt (1 - 2*x) = 3 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_root_value_l3708_370865


namespace NUMINAMATH_CALUDE_mathematical_run_disqualified_team_size_l3708_370824

theorem mathematical_run_disqualified_team_size 
  (initial_teams : ℕ) 
  (initial_average : ℕ) 
  (final_teams : ℕ) 
  (final_average : ℕ) 
  (h1 : initial_teams = 9)
  (h2 : initial_average = 7)
  (h3 : final_teams = initial_teams - 1)
  (h4 : final_average = 6) :
  initial_teams * initial_average - final_teams * final_average = 15 :=
by sorry

end NUMINAMATH_CALUDE_mathematical_run_disqualified_team_size_l3708_370824


namespace NUMINAMATH_CALUDE_max_product_constrained_max_product_value_max_product_achieved_l3708_370838

theorem max_product_constrained (m n : ℝ) : 
  m = 8 - n → m > 0 → n > 0 → ∀ x y : ℝ, x = 8 - y → x > 0 → y > 0 → x * y ≤ m * n := by
  sorry

theorem max_product_value (m n : ℝ) :
  m = 8 - n → m > 0 → n > 0 → m * n ≤ 16 := by
  sorry

theorem max_product_achieved (m n : ℝ) :
  m = 8 - n → m > 0 → n > 0 → ∃ x y : ℝ, x = 8 - y ∧ x > 0 ∧ y > 0 ∧ x * y = 16 := by
  sorry

end NUMINAMATH_CALUDE_max_product_constrained_max_product_value_max_product_achieved_l3708_370838


namespace NUMINAMATH_CALUDE_binomial_coefficient_equality_l3708_370853

theorem binomial_coefficient_equality (n : ℕ) : 
  (Nat.choose n 2 = Nat.choose n 5) → n = 7 := by sorry

end NUMINAMATH_CALUDE_binomial_coefficient_equality_l3708_370853


namespace NUMINAMATH_CALUDE_wall_length_is_800_l3708_370834

/-- Represents the dimensions of a brick in centimeters -/
structure BrickDimensions where
  length : ℝ
  width : ℝ
  height : ℝ

/-- Represents the dimensions of a wall in centimeters -/
structure WallDimensions where
  length : ℝ
  width : ℝ
  height : ℝ

/-- Calculates the volume of a brick given its dimensions -/
def brickVolume (b : BrickDimensions) : ℝ :=
  b.length * b.width * b.height

/-- Calculates the volume of a wall given its dimensions -/
def wallVolume (w : WallDimensions) : ℝ :=
  w.length * w.width * w.height

theorem wall_length_is_800 (brick : BrickDimensions) (wall : WallDimensions) 
    (h1 : brick.length = 25)
    (h2 : brick.width = 11.25)
    (h3 : brick.height = 6)
    (h4 : wall.width = 600)
    (h5 : wall.height = 22.5)
    (h6 : brickVolume brick * 6400 = wallVolume wall) :
    wall.length = 800 := by
  sorry

#check wall_length_is_800

end NUMINAMATH_CALUDE_wall_length_is_800_l3708_370834


namespace NUMINAMATH_CALUDE_intersection_equals_open_interval_l3708_370891

-- Define the sets M and N
def M : Set ℝ := {x : ℝ | -1 < x ∧ x < 3}
def N : Set ℝ := {x : ℝ | -2 < x ∧ x < 1}

-- Define the open interval (-1, 1)
def openInterval : Set ℝ := {x : ℝ | -1 < x ∧ x < 1}

-- Theorem statement
theorem intersection_equals_open_interval : M ∩ N = openInterval := by
  sorry

end NUMINAMATH_CALUDE_intersection_equals_open_interval_l3708_370891


namespace NUMINAMATH_CALUDE_sum_of_parts_for_specific_complex_l3708_370830

theorem sum_of_parts_for_specific_complex (z : ℂ) (h : z = 1 - Complex.I) : 
  z.re + z.im = 0 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_parts_for_specific_complex_l3708_370830


namespace NUMINAMATH_CALUDE_end_behavior_of_g_l3708_370887

noncomputable def g (x : ℝ) : ℝ := -3 * x^4 + 5 * x^3 - 4

theorem end_behavior_of_g :
  (∀ M : ℝ, ∃ N : ℝ, ∀ x : ℝ, x > N → g x < M) ∧
  (∀ M : ℝ, ∃ N : ℝ, ∀ x : ℝ, x < -N → g x < M) :=
sorry

end NUMINAMATH_CALUDE_end_behavior_of_g_l3708_370887


namespace NUMINAMATH_CALUDE_ellipse_properties_l3708_370827

-- Define the ellipse
def ellipse (x y : ℝ) : Prop := x^2 / 4 + y^2 / 3 = 1

-- Define the foci
def F1 : ℝ × ℝ := sorry
def F2 : ℝ × ℝ := sorry

-- Define a point on the ellipse
def P : ℝ × ℝ := sorry
axiom P_on_ellipse : ellipse P.1 P.2

-- Define the eccentricity
def eccentricity : ℝ := sorry

-- Define the distance between two points
def distance (p1 p2 : ℝ × ℝ) : ℝ := sorry

-- Define the angle between three points
def angle (p1 p2 p3 : ℝ × ℝ) : ℝ := sorry

-- Theorem statement
theorem ellipse_properties :
  eccentricity = 1/2 ∧
  ∃ (Q : ℝ × ℝ), ellipse Q.1 Q.2 ∧ distance Q F1 = 3 ∧ ∀ (R : ℝ × ℝ), ellipse R.1 R.2 → distance R F1 ≤ 3 ∧
  0 ≤ angle F1 P F2 ∧ angle F1 P F2 ≤ π/3 :=
sorry

end NUMINAMATH_CALUDE_ellipse_properties_l3708_370827


namespace NUMINAMATH_CALUDE_exists_valid_selection_l3708_370822

/-- A vertex of a polygon with two distinct numbers -/
structure Vertex :=
  (num1 : ℕ)
  (num2 : ℕ)
  (distinct : num1 ≠ num2)

/-- A convex 100-gon with two numbers at each vertex -/
def Polygon := Fin 100 → Vertex

/-- A selection of numbers from the vertices -/
def Selection := Fin 100 → ℕ

/-- Predicate to check if a selection is valid (no adjacent vertices have the same number) -/
def ValidSelection (p : Polygon) (s : Selection) : Prop :=
  ∀ i : Fin 100, s i ≠ s (i + 1)

/-- Theorem stating that for any 100-gon with two distinct numbers at each vertex,
    there exists a valid selection of numbers -/
theorem exists_valid_selection (p : Polygon) :
  ∃ s : Selection, ValidSelection p s ∧ ∀ i : Fin 100, s i = (p i).num1 ∨ s i = (p i).num2 :=
sorry

end NUMINAMATH_CALUDE_exists_valid_selection_l3708_370822


namespace NUMINAMATH_CALUDE_total_sum_is_correct_l3708_370868

/-- Represents the share ratios and total sum for a money division problem -/
structure MoneyDivision where
  a_ratio : ℝ
  b_ratio : ℝ
  c_ratio : ℝ
  c_share : ℝ
  total_sum : ℝ

/-- The money division problem with given ratios and c's share -/
def problem : MoneyDivision :=
  { a_ratio := 1
    b_ratio := 0.65
    c_ratio := 0.40
    c_share := 48
    total_sum := 246 }

/-- Theorem stating that the total sum is correct given the problem conditions -/
theorem total_sum_is_correct (p : MoneyDivision) :
  p.a_ratio = 1 ∧
  p.b_ratio = 0.65 ∧
  p.c_ratio = 0.40 ∧
  p.c_share = 48 →
  p.total_sum = 246 := by
  sorry

#check total_sum_is_correct problem

end NUMINAMATH_CALUDE_total_sum_is_correct_l3708_370868


namespace NUMINAMATH_CALUDE_parabola_intersection_distance_l3708_370832

theorem parabola_intersection_distance : 
  let f (x : ℝ) := x^2 - 2*x - 3
  let roots := {x : ℝ | f x = 0}
  ∃ (a b : ℝ), a ∈ roots ∧ b ∈ roots ∧ a ≠ b ∧ |a - b| = 4 :=
by
  sorry

end NUMINAMATH_CALUDE_parabola_intersection_distance_l3708_370832


namespace NUMINAMATH_CALUDE_problem_statement_l3708_370897

open Set Real

def M (a : ℝ) : Set ℝ := {x | (x + a) * (x - 1) ≤ 0}
def N : Set ℝ := {x | 4 * x^2 - 4 * x - 3 < 0}

theorem problem_statement (a : ℝ) (h : a > 0) :
  (M a ∪ N = Icc (-2) (3/2) → a = 2) ∧
  (N ∪ (univ \ M a) = univ → 0 < a ∧ a ≤ 1/2) :=
sorry

end NUMINAMATH_CALUDE_problem_statement_l3708_370897


namespace NUMINAMATH_CALUDE_triangle_inequality_l3708_370831

theorem triangle_inequality (a b c : ℝ) (h : 0 < a ∧ 0 < b ∧ 0 < c ∧ a + b > c ∧ b + c > a ∧ c + a > b) :
  (a + b - c) * (a - b + c) * (-a + b + c) ≤ a * b * c := by
  sorry

end NUMINAMATH_CALUDE_triangle_inequality_l3708_370831


namespace NUMINAMATH_CALUDE_circle_intersection_theorem_l3708_370812

-- Define the circle
def Circle (r : ℝ) := {p : ℝ × ℝ | p.1^2 + p.2^2 = r^2}

-- Define the line
def Line := {p : ℝ × ℝ | p.2 = -p.1 + 2}

-- Define the origin
def O : ℝ × ℝ := (0, 0)

-- Theorem statement
theorem circle_intersection_theorem (r : ℝ) 
  (A B C : ℝ × ℝ) 
  (hA : A ∈ Circle r ∩ Line) 
  (hB : B ∈ Circle r ∩ Line) 
  (hC : C ∈ Circle r) 
  (hOC : C.1 * C.1 + C.2 * C.2 = (5/4 * A.1 + 3/4 * B.1)^2 + (5/4 * A.2 + 3/4 * B.2)^2) :
  r = Real.sqrt 10 := by
  sorry


end NUMINAMATH_CALUDE_circle_intersection_theorem_l3708_370812


namespace NUMINAMATH_CALUDE_workshop_salary_calculation_l3708_370833

/-- Given a workshop with workers and technicians, calculate the average salary of non-technician workers. -/
theorem workshop_salary_calculation
  (total_workers : ℕ)
  (avg_salary_all : ℝ)
  (num_technicians : ℕ)
  (avg_salary_technicians : ℝ)
  (h_total_workers : total_workers = 21)
  (h_avg_salary_all : avg_salary_all = 8000)
  (h_num_technicians : num_technicians = 7)
  (h_avg_salary_technicians : avg_salary_technicians = 12000) :
  let num_rest := total_workers - num_technicians
  let total_salary := avg_salary_all * total_workers
  let total_salary_technicians := avg_salary_technicians * num_technicians
  let total_salary_rest := total_salary - total_salary_technicians
  total_salary_rest / num_rest = 6000 := by
  sorry

end NUMINAMATH_CALUDE_workshop_salary_calculation_l3708_370833


namespace NUMINAMATH_CALUDE_linear_equation_integer_solution_l3708_370819

theorem linear_equation_integer_solution : ∃ (x y : ℤ), 2 * x + y - 1 = 0 := by
  sorry

end NUMINAMATH_CALUDE_linear_equation_integer_solution_l3708_370819


namespace NUMINAMATH_CALUDE_bead_arrangement_probability_l3708_370881

/-- The number of red beads -/
def num_red : ℕ := 4

/-- The number of white beads -/
def num_white : ℕ := 2

/-- The number of green beads -/
def num_green : ℕ := 1

/-- The total number of beads -/
def total_beads : ℕ := num_red + num_white + num_green

/-- A function that calculates the probability of arranging the beads
    such that no two neighboring beads are the same color -/
def prob_no_adjacent_same_color : ℚ :=
  2 / 15

/-- Theorem stating that the probability of arranging the beads
    such that no two neighboring beads are the same color is 2/15 -/
theorem bead_arrangement_probability :
  prob_no_adjacent_same_color = 2 / 15 := by
  sorry

end NUMINAMATH_CALUDE_bead_arrangement_probability_l3708_370881


namespace NUMINAMATH_CALUDE_distribute_three_letters_four_mailboxes_l3708_370826

/-- The number of ways to distribute n distinct objects into k distinct containers -/
def distribute (n k : ℕ) : ℕ := k^n

/-- Theorem: Distributing 3 letters into 4 mailboxes results in 4^3 ways -/
theorem distribute_three_letters_four_mailboxes : 
  distribute 3 4 = 4^3 := by sorry

end NUMINAMATH_CALUDE_distribute_three_letters_four_mailboxes_l3708_370826


namespace NUMINAMATH_CALUDE_min_correct_responses_l3708_370880

def score (correct : ℕ) : ℤ :=
  8 * (correct : ℤ) - 20

theorem min_correct_responses : ∃ n : ℕ, 
  (∀ m : ℕ, m < n → score m + 10 < 120) ∧ 
  (score n + 10 ≥ 120) ∧
  n = 17 :=
sorry

end NUMINAMATH_CALUDE_min_correct_responses_l3708_370880


namespace NUMINAMATH_CALUDE_book_price_comparison_l3708_370878

theorem book_price_comparison (price_second : ℝ) (price_first : ℝ) :
  price_first = price_second * 1.5 →
  (price_first - price_second) / price_first * 100 = 100 / 3 := by
  sorry

end NUMINAMATH_CALUDE_book_price_comparison_l3708_370878


namespace NUMINAMATH_CALUDE_circular_view_not_rectangular_prism_l3708_370847

/-- A geometric body in three-dimensional space. -/
class GeometricBody :=
(has_circular_view : Bool)

/-- A Rectangular Prism is a type of GeometricBody. -/
def RectangularPrism : GeometricBody :=
{ has_circular_view := false }

/-- Theorem: If a geometric body has a circular view from some direction, it cannot be a Rectangular Prism. -/
theorem circular_view_not_rectangular_prism (body : GeometricBody) :
  body.has_circular_view → body ≠ RectangularPrism :=
sorry

end NUMINAMATH_CALUDE_circular_view_not_rectangular_prism_l3708_370847


namespace NUMINAMATH_CALUDE_lcm_of_five_numbers_l3708_370894

theorem lcm_of_five_numbers : Nat.lcm 53 (Nat.lcm 71 (Nat.lcm 89 (Nat.lcm 103 200))) = 788045800 := by
  sorry

end NUMINAMATH_CALUDE_lcm_of_five_numbers_l3708_370894


namespace NUMINAMATH_CALUDE_repeating_decimal_85_l3708_370804

/-- Represents a repeating decimal with a two-digit repetend -/
def RepeatingDecimal (a b : ℕ) : ℚ :=
  (10 * a + b : ℚ) / 99

theorem repeating_decimal_85 :
  RepeatingDecimal 8 5 = 85 / 99 := by
  sorry

end NUMINAMATH_CALUDE_repeating_decimal_85_l3708_370804


namespace NUMINAMATH_CALUDE_equation_solution_range_l3708_370863

theorem equation_solution_range (a : ℝ) : 
  (∃ x : ℝ, (Real.exp (2 * x) + a * Real.exp x + 1 = 0)) ↔ a ≤ -2 :=
by sorry

end NUMINAMATH_CALUDE_equation_solution_range_l3708_370863


namespace NUMINAMATH_CALUDE_parallelogram_area_l3708_370883

/-- The area of a parallelogram with base 12 feet and height 5 feet is 60 square feet. -/
theorem parallelogram_area (base height : ℝ) (h1 : base = 12) (h2 : height = 5) :
  base * height = 60 := by
  sorry

end NUMINAMATH_CALUDE_parallelogram_area_l3708_370883


namespace NUMINAMATH_CALUDE_decimal_representation_contradiction_l3708_370803

theorem decimal_representation_contradiction (m n : ℕ) (h_n : n ≤ 100) :
  ∃ (k : ℕ) (B : ℕ), (1000 * B : ℚ) / n = 167 + (k : ℚ) / 1000 → False :=
by sorry

end NUMINAMATH_CALUDE_decimal_representation_contradiction_l3708_370803


namespace NUMINAMATH_CALUDE_tan_neg_390_degrees_l3708_370855

theorem tan_neg_390_degrees : Real.tan ((-390 : ℝ) * π / 180) = -Real.sqrt 3 / 3 := by
  sorry

end NUMINAMATH_CALUDE_tan_neg_390_degrees_l3708_370855


namespace NUMINAMATH_CALUDE_eccentricity_relation_l3708_370811

/-- Given an ellipse with eccentricity e₁ and a hyperbola with eccentricity e₂,
    both sharing common foci F₁ and F₂, and a common point P such that
    the vectors PF₁ and PF₂ are perpendicular, prove that
    (e₁² + e₂²) / (e₁e₂)² = 2 -/
theorem eccentricity_relation (e₁ e₂ : ℝ) (F₁ F₂ P : ℝ × ℝ) :
  e₁ > 0 ∧ e₁ < 1 →  -- Condition for ellipse eccentricity
  e₂ > 1 →  -- Condition for hyperbola eccentricity
  (P.1 - F₁.1) * (P.1 - F₂.1) + (P.2 - F₁.2) * (P.2 - F₂.2) = 0 →  -- Perpendicularity condition
  (e₁^2 + e₂^2) / (e₁ * e₂)^2 = 2 := by
  sorry

end NUMINAMATH_CALUDE_eccentricity_relation_l3708_370811


namespace NUMINAMATH_CALUDE_investment_time_period_l3708_370810

/-- Proves that given a principal of 2000 invested at simple interest rates of 18% p.a. and 12% p.a.,
    if the difference in interest received is 240, then the time period of investment is 20 years. -/
theorem investment_time_period (principal : ℝ) (rate_high : ℝ) (rate_low : ℝ) (interest_diff : ℝ) :
  principal = 2000 →
  rate_high = 18 →
  rate_low = 12 →
  interest_diff = 240 →
  ∃ time : ℝ,
    principal * (rate_high / 100) * time - principal * (rate_low / 100) * time = interest_diff ∧
    time = 20 := by
  sorry

end NUMINAMATH_CALUDE_investment_time_period_l3708_370810


namespace NUMINAMATH_CALUDE_die_roll_probability_l3708_370825

theorem die_roll_probability : 
  let p : ℚ := 1/3  -- probability of rolling a number divisible by 3
  let n : ℕ := 8    -- number of rolls
  1 - (1 - p)^n = 6305/6561 := by
sorry

end NUMINAMATH_CALUDE_die_roll_probability_l3708_370825


namespace NUMINAMATH_CALUDE_january_first_day_l3708_370845

/-- Represents days of the week -/
inductive DayOfWeek
  | Sunday
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday
  | Saturday

/-- Represents a month with specific properties -/
structure Month where
  days : Nat
  tuesdays : Nat
  saturdays : Nat

/-- Returns the day of the week for the first day of the month -/
def firstDayOfMonth (m : Month) : DayOfWeek :=
  sorry

theorem january_first_day (m : Month) 
  (h1 : m.days = 31)
  (h2 : m.tuesdays = 4)
  (h3 : m.saturdays = 4) :
  firstDayOfMonth m = DayOfWeek.Wednesday :=
sorry

end NUMINAMATH_CALUDE_january_first_day_l3708_370845


namespace NUMINAMATH_CALUDE_triangle_y_coordinate_l3708_370875

/-- Given a triangle with vertices (-1, 0), (7, y), and (7, -4), if its area is 32, then y = 4 -/
theorem triangle_y_coordinate (y : ℝ) : 
  let vertices := [(-1, 0), (7, y), (7, -4)]
  let area := (1/2 : ℝ) * |(-1 * (y - (-4)) + 7 * ((-4) - 0) + 7 * (0 - y))|
  area = 32 → y = 4 := by sorry

end NUMINAMATH_CALUDE_triangle_y_coordinate_l3708_370875


namespace NUMINAMATH_CALUDE_james_brothers_cut_sixty_percent_fewer_trees_l3708_370809

/-- The percentage fewer trees James' brothers cut per day compared to James -/
def brothers_percentage_fewer (james_trees_per_day : ℕ) (total_days : ℕ) (james_solo_days : ℕ) (total_trees : ℕ) : ℚ :=
  let total_with_brothers := total_trees - james_trees_per_day * james_solo_days
  let daily_with_brothers := total_with_brothers / (total_days - james_solo_days)
  let brothers_trees_per_day := daily_with_brothers - james_trees_per_day
  (brothers_trees_per_day : ℚ) / james_trees_per_day * 100

theorem james_brothers_cut_sixty_percent_fewer_trees
  (h1 : brothers_percentage_fewer 20 5 2 196 = 60) :
  ∃ (james_trees_per_day : ℕ) (total_days : ℕ) (james_solo_days : ℕ) (total_trees : ℕ),
    james_trees_per_day = 20 ∧
    total_days = 5 ∧
    james_solo_days = 2 ∧
    total_trees = 196 ∧
    brothers_percentage_fewer james_trees_per_day total_days james_solo_days total_trees = 60 :=
by sorry

end NUMINAMATH_CALUDE_james_brothers_cut_sixty_percent_fewer_trees_l3708_370809


namespace NUMINAMATH_CALUDE_range_a_all_real_range_a_interval_l3708_370842

/-- The function f(x) = x^2 + ax + 3 -/
def f (a : ℝ) (x : ℝ) : ℝ := x^2 + a*x + 3

/-- Theorem for the range of 'a' when f(x) ≥ a for all real x -/
theorem range_a_all_real (a : ℝ) :
  (∀ x : ℝ, f a x ≥ a) ↔ a ≤ 3 :=
sorry

/-- Theorem for the range of 'a' when f(x) ≥ a for x in [-2, 2] -/
theorem range_a_interval (a : ℝ) :
  (∀ x : ℝ, x ∈ Set.Icc (-2) 2 → f a x ≥ a) ↔ a ∈ Set.Icc (-6) 2 :=
sorry

end NUMINAMATH_CALUDE_range_a_all_real_range_a_interval_l3708_370842


namespace NUMINAMATH_CALUDE_midpoint_of_number_line_l3708_370876

theorem midpoint_of_number_line (a b : ℝ) (ha : a = -1) (hb : b = 3) :
  (a + b) / 2 = 1 := by sorry

end NUMINAMATH_CALUDE_midpoint_of_number_line_l3708_370876


namespace NUMINAMATH_CALUDE_cubic_poly_max_value_l3708_370806

/-- A cubic monic polynomial with roots a, b, and c -/
def cubic_monic_poly (a b c : ℝ) : ℝ → ℝ :=
  fun x => x^3 + (-(a + b + c)) * x^2 + (a*b + b*c + c*a) * x - a*b*c

/-- The theorem statement -/
theorem cubic_poly_max_value (a b c : ℝ) :
  let P := cubic_monic_poly a b c
  P 1 = 91 ∧ P (-1) = -121 →
  (∀ x y z : ℝ, (x*y + y*z + z*x) / (x*y*z + x + y + z) ≤ 7) ∧
  (∃ x y z : ℝ, (x*y + y*z + z*x) / (x*y*z + x + y + z) = 7) :=
by sorry

end NUMINAMATH_CALUDE_cubic_poly_max_value_l3708_370806


namespace NUMINAMATH_CALUDE_geometric_series_sum_6_terms_l3708_370861

def geometricSeriesSum (a : ℚ) (r : ℚ) (n : ℕ) : ℚ :=
  a * (1 - r^n) / (1 - r)

theorem geometric_series_sum_6_terms :
  let a : ℚ := 2
  let r : ℚ := 1/3
  let n : ℕ := 6
  geometricSeriesSum a r n = 2184/729 := by
sorry

end NUMINAMATH_CALUDE_geometric_series_sum_6_terms_l3708_370861


namespace NUMINAMATH_CALUDE_parallel_iff_abs_x_eq_two_l3708_370884

def vector_a (x : ℝ) : ℝ × ℝ := (1, x)
def vector_b (x : ℝ) : ℝ × ℝ := (x^2, 4*x)

def parallel (v w : ℝ × ℝ) : Prop :=
  ∃ (k : ℝ), v.1 = k * w.1 ∧ v.2 = k * w.2

theorem parallel_iff_abs_x_eq_two (x : ℝ) :
  (vector_a x ≠ (0, 0)) → (vector_b x ≠ (0, 0)) →
  (parallel (vector_a x) (vector_b x) ↔ |x| = 2) :=
sorry

end NUMINAMATH_CALUDE_parallel_iff_abs_x_eq_two_l3708_370884


namespace NUMINAMATH_CALUDE_sum_of_multiples_l3708_370814

theorem sum_of_multiples (m n : ℝ) : 2 * m + 3 * n = 2*m + 3*n := by sorry

end NUMINAMATH_CALUDE_sum_of_multiples_l3708_370814


namespace NUMINAMATH_CALUDE_triangle_side_length_l3708_370886

theorem triangle_side_length (a b : ℝ) (A B : Real) :
  a = 10 →
  B = Real.pi / 3 →
  A = Real.pi / 4 →
  b = 10 * (Real.sin (Real.pi / 3) / Real.sin (Real.pi / 4)) :=
by sorry

end NUMINAMATH_CALUDE_triangle_side_length_l3708_370886


namespace NUMINAMATH_CALUDE_michaels_brother_initial_money_l3708_370856

/-- Proof that Michael's brother initially had $17 -/
theorem michaels_brother_initial_money :
  ∀ (michael_money : ℕ) (brother_money_after : ℕ) (candy_cost : ℕ),
    michael_money = 42 →
    brother_money_after = 35 →
    candy_cost = 3 →
    ∃ (brother_initial_money : ℕ),
      brother_initial_money = 17 ∧
      brother_money_after = brother_initial_money + michael_money / 2 - candy_cost :=
by
  sorry

#check michaels_brother_initial_money

end NUMINAMATH_CALUDE_michaels_brother_initial_money_l3708_370856


namespace NUMINAMATH_CALUDE_simplify_fraction_l3708_370835

theorem simplify_fraction : (1 / (2 + Real.sqrt 3)) * (1 / (2 - Real.sqrt 3)) = 1 := by sorry

end NUMINAMATH_CALUDE_simplify_fraction_l3708_370835


namespace NUMINAMATH_CALUDE_inequality_proof_l3708_370895

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (sum_eq_3 : a + b + c = 3) :
  a^2 / (a + b) + b^2 / (b + c) + c^2 / (c + a) ≥ 3/2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l3708_370895


namespace NUMINAMATH_CALUDE_points_same_side_of_line_l3708_370874

theorem points_same_side_of_line (a : ℝ) : 
  (∃ (s : ℝ), s * (3 * 3 - 2 * 1 + a) > 0 ∧ s * (3 * (-4) - 2 * 6 + a) > 0) ↔ 
  (a < -7 ∨ a > 24) :=
sorry

end NUMINAMATH_CALUDE_points_same_side_of_line_l3708_370874


namespace NUMINAMATH_CALUDE_three_heads_in_four_tosses_l3708_370882

def coin_toss_probability (n : ℕ) (k : ℕ) : ℚ :=
  (Nat.choose n k : ℚ) * (1 / 2) ^ k * (1 / 2) ^ (n - k)

theorem three_heads_in_four_tosses :
  coin_toss_probability 4 3 = 1 / 4 := by
  sorry

end NUMINAMATH_CALUDE_three_heads_in_four_tosses_l3708_370882


namespace NUMINAMATH_CALUDE_parabola_shift_l3708_370823

/-- The original parabola function -/
def original_parabola (x : ℝ) : ℝ := x^2 + 1

/-- The shifted parabola function -/
def shifted_parabola (x : ℝ) : ℝ := (x + 2)^2 - 2

/-- Theorem stating that the shifted parabola is equivalent to 
    shifting the original parabola 2 units left and 3 units down -/
theorem parabola_shift : 
  ∀ x : ℝ, shifted_parabola x = original_parabola (x + 2) - 3 := by
  sorry

end NUMINAMATH_CALUDE_parabola_shift_l3708_370823


namespace NUMINAMATH_CALUDE_square_tile_count_l3708_370877

theorem square_tile_count (n : ℕ) (h : n^2 = 81) : 
  n^2 * n^2 - (2*n - 1) = 6544 := by
  sorry

end NUMINAMATH_CALUDE_square_tile_count_l3708_370877


namespace NUMINAMATH_CALUDE_fifteen_sided_polygon_diagonals_l3708_370893

/-- The number of diagonals in a convex polygon with n sides -/
def num_diagonals (n : ℕ) : ℕ := n * (n - 3) / 2

/-- Theorem: A convex polygon with 15 sides has 90 diagonals -/
theorem fifteen_sided_polygon_diagonals :
  num_diagonals 15 = 90 := by
  sorry

end NUMINAMATH_CALUDE_fifteen_sided_polygon_diagonals_l3708_370893


namespace NUMINAMATH_CALUDE_factorial_not_prime_l3708_370873

theorem factorial_not_prime (n : ℕ) (h : n > 1) : ¬ Nat.Prime (n!) := by
  sorry

end NUMINAMATH_CALUDE_factorial_not_prime_l3708_370873


namespace NUMINAMATH_CALUDE_system_solution_l3708_370899

theorem system_solution (x y : ℝ) : 
  (x^2 + x*y + y^2) / (x^2 - x*y + y^2) = 3 →
  x^3 + y^3 = 2 →
  x = 1 ∧ y = 1 := by
sorry

end NUMINAMATH_CALUDE_system_solution_l3708_370899


namespace NUMINAMATH_CALUDE_chord_diagonal_intersections_collinear_l3708_370858

namespace CircleChords

-- Define a circle
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

-- Define a point on the plane
structure Point where
  x : ℝ
  y : ℝ

-- Define a chord as a pair of points
structure Chord where
  p1 : Point
  p2 : Point

-- Define the problem setup
structure ChordConfiguration where
  circle : Circle
  chordAB : Chord
  chordCD : Chord
  chordEF : Chord
  -- Ensure chords are non-intersecting
  non_intersecting : 
    chordAB.p1 ≠ chordCD.p1 ∧ chordAB.p1 ≠ chordCD.p2 ∧
    chordAB.p2 ≠ chordCD.p1 ∧ chordAB.p2 ≠ chordCD.p2 ∧
    chordAB.p1 ≠ chordEF.p1 ∧ chordAB.p1 ≠ chordEF.p2 ∧
    chordAB.p2 ≠ chordEF.p1 ∧ chordAB.p2 ≠ chordEF.p2 ∧
    chordCD.p1 ≠ chordEF.p1 ∧ chordCD.p1 ≠ chordEF.p2 ∧
    chordCD.p2 ≠ chordEF.p1 ∧ chordCD.p2 ≠ chordEF.p2

-- Define the intersection of diagonals
def diagonalIntersection (q1 q2 q3 q4 : Point) : Point :=
  sorry -- Actual implementation would calculate the intersection

-- Define collinearity
def collinear (p1 p2 p3 : Point) : Prop :=
  sorry -- Actual implementation would define collinearity

-- Theorem statement
theorem chord_diagonal_intersections_collinear (config : ChordConfiguration) :
  let M := diagonalIntersection config.chordAB.p1 config.chordAB.p2 config.chordEF.p1 config.chordEF.p2
  let N := diagonalIntersection config.chordCD.p1 config.chordCD.p2 config.chordEF.p1 config.chordEF.p2
  let P := diagonalIntersection config.chordAB.p1 config.chordAB.p2 config.chordCD.p1 config.chordCD.p2
  collinear M N P :=
by
  sorry

end CircleChords

end NUMINAMATH_CALUDE_chord_diagonal_intersections_collinear_l3708_370858


namespace NUMINAMATH_CALUDE_geometry_propositions_l3708_370871

-- Define the basic types
variable (α : Type*) [NormedAddCommGroup α] [InnerProductSpace ℝ α]
variable (Point : Type*) [AddCommGroup Point] [Module ℝ Point]
variable (Line : Type*) (Plane : Type*)

-- Define the relations
variable (perpendicular : Line → Plane → Prop)
variable (parallel : Line → Plane → Prop)
variable (parallel_lines : Line → Line → Prop)
variable (perpendicular_lines : Line → Line → Prop)

-- State the theorem
theorem geometry_propositions
  (m n : Line) (π : Plane) (h_diff : m ≠ n) :
  (∀ m n π, perpendicular m π → perpendicular n π → parallel_lines m n) ∧
  ¬(∀ m n π, parallel m π → parallel n π → parallel_lines m n) ∧
  (∀ m n π, parallel m π → perpendicular n π → perpendicular_lines m n) :=
sorry

end NUMINAMATH_CALUDE_geometry_propositions_l3708_370871


namespace NUMINAMATH_CALUDE_min_value_inequality_l3708_370802

theorem min_value_inequality (x y : ℝ) (hx : x > 0) (hy : y > 0) :
  (Real.sqrt ((2 * x^2 + y^2) * (4 * x^2 + y^2))) / (x * y) ≥ 2 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_min_value_inequality_l3708_370802


namespace NUMINAMATH_CALUDE_sum_geq_abs_sum_div_3_l3708_370888

theorem sum_geq_abs_sum_div_3 (a b c : ℝ) 
  (hab : a + b ≥ 0) (hbc : b + c ≥ 0) (hca : c + a ≥ 0) : 
  a + b + c ≥ (|a| + |b| + |c|) / 3 := by
  sorry

end NUMINAMATH_CALUDE_sum_geq_abs_sum_div_3_l3708_370888


namespace NUMINAMATH_CALUDE_min_sum_of_distinct_integers_with_odd_square_sums_l3708_370815

theorem min_sum_of_distinct_integers_with_odd_square_sums (a b c d : ℕ+) : 
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d →
  ∃ (m n p : ℕ), 
    (a + b = 2 * m + 1) ∧ (a + c = 2 * n + 1) ∧ (a + d = 2 * p + 1) ∧
    (∃ (x y z : ℕ), (2 * m + 1 = x^2) ∧ (2 * n + 1 = y^2) ∧ (2 * p + 1 = z^2)) →
  10 * (a + b + c + d) ≥ 670 :=
by sorry

#check min_sum_of_distinct_integers_with_odd_square_sums

end NUMINAMATH_CALUDE_min_sum_of_distinct_integers_with_odd_square_sums_l3708_370815


namespace NUMINAMATH_CALUDE_folded_square_area_ratio_l3708_370840

/-- The ratio of the area of a square paper folded along a line connecting points
    at 1/3 and 2/3 of one side to the area of the original square is 5/6. -/
theorem folded_square_area_ratio (s : ℝ) (h : s > 0) : 
  let A := s^2
  let B := s^2 - (1/2 * (s/3) * s)
  B / A = 5/6 := by sorry

end NUMINAMATH_CALUDE_folded_square_area_ratio_l3708_370840


namespace NUMINAMATH_CALUDE_inscribed_circle_hypotenuse_length_l3708_370866

/-- A circle inscribed on the hypotenuse of a right triangle -/
structure InscribedCircle where
  /-- The right triangle -/
  triangle : Set (ℝ × ℝ)
  /-- The inscribed circle -/
  circle : Set (ℝ × ℝ)
  /-- Point A of the triangle -/
  A : ℝ × ℝ
  /-- Point B of the triangle -/
  B : ℝ × ℝ
  /-- Point C of the triangle -/
  C : ℝ × ℝ
  /-- Point M, tangency point of the circle with AB -/
  M : ℝ × ℝ
  /-- Point N, intersection of the circle with AC -/
  N : ℝ × ℝ
  /-- Center of the circle -/
  O : ℝ × ℝ
  /-- The triangle is right-angled at B -/
  right_angle : (B.1 - A.1) * (C.1 - A.1) + (B.2 - A.2) * (C.2 - A.2) = 0
  /-- The circle touches AB at M -/
  touches_AB : M ∈ circle ∩ {p | (p.1 - A.1) * (B.2 - A.2) = (p.2 - A.2) * (B.1 - A.1)}
  /-- The circle touches BC -/
  touches_BC : ∃ p ∈ circle, (p.1 - B.1) * (C.2 - B.2) = (p.2 - B.2) * (C.1 - B.1)
  /-- The circle lies on AC -/
  on_AC : O ∈ {p | (p.1 - A.1) * (C.2 - A.2) = (p.2 - A.2) * (C.1 - A.1)}
  /-- AM = 20/9 -/
  AM_length : Real.sqrt ((M.1 - A.1)^2 + (M.2 - A.2)^2) = 20/9
  /-- AN:MN = 6:1 -/
  AN_MN_ratio : Real.sqrt ((N.1 - A.1)^2 + (N.2 - A.2)^2) / Real.sqrt ((N.1 - M.1)^2 + (N.2 - M.2)^2) = 6

/-- The main theorem -/
theorem inscribed_circle_hypotenuse_length (ic : InscribedCircle) :
  Real.sqrt ((ic.C.1 - ic.A.1)^2 + (ic.C.2 - ic.A.2)^2) = Real.sqrt 5 + 1/4 := by
  sorry


end NUMINAMATH_CALUDE_inscribed_circle_hypotenuse_length_l3708_370866


namespace NUMINAMATH_CALUDE_yahs_to_bahs_1500_l3708_370885

/-- Conversion rates between bahs, rahs, and yahs -/
structure ConversionRates where
  bahs_to_rahs : ℚ
  rahs_to_yahs : ℚ

/-- Calculate the number of bahs equivalent to a given number of yahs -/
def yahs_to_bahs (rates : ConversionRates) (yahs : ℚ) : ℚ :=
  yahs * rates.rahs_to_yahs * rates.bahs_to_rahs

/-- Theorem stating that 1500 yahs are equivalent to 600 bahs given the conversion rates -/
theorem yahs_to_bahs_1500 (rates : ConversionRates) 
    (h1 : rates.bahs_to_rahs = 30 / 20)
    (h2 : rates.rahs_to_yahs = 12 / 20) : 
  yahs_to_bahs rates 1500 = 600 := by
  sorry

end NUMINAMATH_CALUDE_yahs_to_bahs_1500_l3708_370885


namespace NUMINAMATH_CALUDE_exactly_one_integer_n_for_n_plus_i_sixth_power_integer_l3708_370837

theorem exactly_one_integer_n_for_n_plus_i_sixth_power_integer :
  ∃! (n : ℤ), ∃ (m : ℤ), (n + Complex.I) ^ 6 = m := by sorry

end NUMINAMATH_CALUDE_exactly_one_integer_n_for_n_plus_i_sixth_power_integer_l3708_370837


namespace NUMINAMATH_CALUDE_robins_hair_length_l3708_370800

/-- Calculates the final hair length after growth and cut -/
def final_hair_length (initial : ℕ) (growth : ℕ) (cut : ℕ) : ℕ :=
  if initial + growth ≥ cut then
    initial + growth - cut
  else
    0

/-- Theorem stating that Robin's final hair length is 2 inches -/
theorem robins_hair_length :
  final_hair_length 14 8 20 = 2 := by
  sorry

end NUMINAMATH_CALUDE_robins_hair_length_l3708_370800


namespace NUMINAMATH_CALUDE_inequality_iff_quadratic_nonpositive_l3708_370820

/-- A function satisfying the given inequality condition -/
def SatisfiesInequality (f : ℝ → ℝ) : Prop :=
  ∀ (x y z : ℝ), x < y → y < z →
    f y - ((z - y) / (z - x) * f x + (y - x) / (z - x) * f z) ≤
    f ((x + z) / 2) - (f x + f z) / 2

/-- The set of quadratic functions with non-positive leading coefficient -/
def QuadraticNonPositive (f : ℝ → ℝ) : Prop :=
  ∃ (a b c : ℝ), a ≤ 0 ∧ ∀ (x : ℝ), f x = a * x^2 + b * x + c

/-- Main theorem: A function satisfies the inequality if and only if it's quadratic with non-positive leading coefficient -/
theorem inequality_iff_quadratic_nonpositive (f : ℝ → ℝ) :
  SatisfiesInequality f ↔ QuadraticNonPositive f :=
sorry

end NUMINAMATH_CALUDE_inequality_iff_quadratic_nonpositive_l3708_370820


namespace NUMINAMATH_CALUDE_billion_to_scientific_notation_l3708_370816

def billion : ℝ := 10^9

theorem billion_to_scientific_notation :
  let value : ℝ := 27.58 * billion
  value = 2.758 * 10^10 := by sorry

end NUMINAMATH_CALUDE_billion_to_scientific_notation_l3708_370816


namespace NUMINAMATH_CALUDE_max_square_sum_l3708_370859

def triangle_numbers : Finset ℕ := {5, 6, 7, 8, 9}

def circle_product (a b c : ℕ) : ℕ := a * b * c

def square_sum (f g h : ℕ) : ℕ := f + g + h

theorem max_square_sum :
  ∃ (a b c d e : ℕ),
    a ∈ triangle_numbers ∧
    b ∈ triangle_numbers ∧
    c ∈ triangle_numbers ∧
    d ∈ triangle_numbers ∧
    e ∈ triangle_numbers ∧
    a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧
    b ≠ c ∧ b ≠ d ∧ b ≠ e ∧
    c ≠ d ∧ c ≠ e ∧
    d ≠ e ∧
    square_sum (circle_product a b c) (circle_product b c d) (circle_product c d e) = 1251 ∧
    ∀ (x y z w v : ℕ),
      x ∈ triangle_numbers →
      y ∈ triangle_numbers →
      z ∈ triangle_numbers →
      w ∈ triangle_numbers →
      v ∈ triangle_numbers →
      x ≠ y ∧ x ≠ z ∧ x ≠ w ∧ x ≠ v ∧
      y ≠ z ∧ y ≠ w ∧ y ≠ v ∧
      z ≠ w ∧ z ≠ v ∧
      w ≠ v →
      square_sum (circle_product x y z) (circle_product y z w) (circle_product z w v) ≤ 1251 :=
sorry

end NUMINAMATH_CALUDE_max_square_sum_l3708_370859


namespace NUMINAMATH_CALUDE_additional_red_flowers_needed_l3708_370857

def white_flowers : ℕ := 555
def red_flowers : ℕ := 347

theorem additional_red_flowers_needed : white_flowers - red_flowers = 208 := by
  sorry

end NUMINAMATH_CALUDE_additional_red_flowers_needed_l3708_370857


namespace NUMINAMATH_CALUDE_min_value_sum_squares_l3708_370805

theorem min_value_sum_squares (x y z : ℝ) (h : 2*x + y + 2*z = 6) :
  ∃ (m : ℝ), m = 4 ∧ ∀ (a b c : ℝ), 2*a + b + 2*c = 6 → x^2 + y^2 + z^2 ≥ m ∧ a^2 + b^2 + c^2 ≥ m :=
sorry

end NUMINAMATH_CALUDE_min_value_sum_squares_l3708_370805


namespace NUMINAMATH_CALUDE_x_value_l3708_370828

theorem x_value : ∃ x : ℝ, (0.25 * x = 0.12 * 1500 - 15) ∧ (x = 660) := by
  sorry

end NUMINAMATH_CALUDE_x_value_l3708_370828


namespace NUMINAMATH_CALUDE_solution_set_when_a_eq_2_range_of_a_l3708_370818

-- Define the functions f and g
def f (x : ℝ) : ℝ := |x - 1|
def g (a x : ℝ) : ℝ := 2 * |x - a|

-- Question 1
theorem solution_set_when_a_eq_2 :
  {x : ℝ | f x - g 2 x ≤ x - 3} = {x : ℝ | x ≤ 1 ∨ x ≥ 3} := by sorry

-- Question 2
theorem range_of_a :
  {a : ℝ | ∀ m > 1, ∃ x₀ : ℝ, f x₀ + g a x₀ ≤ (m^2 + m + 4) / (m - 1)} =
  {a : ℝ | -2 - 2 * Real.sqrt 6 ≤ a ∧ a ≤ 2 * Real.sqrt 6 + 4} := by sorry

end NUMINAMATH_CALUDE_solution_set_when_a_eq_2_range_of_a_l3708_370818


namespace NUMINAMATH_CALUDE_sorting_problem_l3708_370870

/-- The number of parcels sorted by a machine per hour relative to a person -/
def machine_efficiency : ℕ := 20

/-- The number of machines used in the comparison -/
def num_machines : ℕ := 5

/-- The number of people used in the comparison -/
def num_people : ℕ := 20

/-- The number of parcels sorted in the comparison -/
def parcels_sorted : ℕ := 6000

/-- The time difference in hours between machines and people sorting -/
def time_difference : ℕ := 4

/-- The number of hours machines work per day -/
def machine_work_hours : ℕ := 16

/-- The number of parcels that need to be sorted per day -/
def daily_parcels : ℕ := 100000

/-- The number of parcels sorted manually by one person per hour -/
def parcels_per_person (x : ℕ) : Prop :=
  (parcels_sorted / (num_people * x)) - (parcels_sorted / (num_machines * machine_efficiency * x)) = time_difference

/-- The minimum number of machines needed to sort the daily parcels -/
def machines_needed (y : ℕ) : Prop :=
  y = (daily_parcels + machine_work_hours * machine_efficiency * 60 - 1) / (machine_work_hours * machine_efficiency * 60)

theorem sorting_problem :
  ∃ (x y : ℕ), parcels_per_person x ∧ machines_needed y ∧ x = 60 ∧ y = 6 :=
sorry

end NUMINAMATH_CALUDE_sorting_problem_l3708_370870


namespace NUMINAMATH_CALUDE_quadratic_equation_1_quadratic_equation_2_l3708_370839

-- Equation 1
theorem quadratic_equation_1 :
  ∃ x₁ x₂ : ℝ, x₁ = 1/3 ∧ x₂ = -1 ∧ 
  (3 * x₁^2 + 2 * x₁ - 1 = 0) ∧ 
  (3 * x₂^2 + 2 * x₂ - 1 = 0) :=
sorry

-- Equation 2
theorem quadratic_equation_2 :
  ∃ x : ℝ, x = 3 ∧ 
  (x + 2) * (x - 3) = 5 * x - 15 :=
sorry

end NUMINAMATH_CALUDE_quadratic_equation_1_quadratic_equation_2_l3708_370839


namespace NUMINAMATH_CALUDE_tara_book_sales_l3708_370829

/-- Calculates the total number of books Tara needs to sell to buy a new clarinet and an accessory, given initial savings, clarinet cost, book price, and additional accessory cost. -/
def total_books_sold (initial_savings : ℕ) (clarinet_cost : ℕ) (book_price : ℕ) (accessory_cost : ℕ) : ℕ :=
  let initial_goal := clarinet_cost - initial_savings
  let halfway_books := (initial_goal / 2) / book_price
  let final_goal := initial_goal + accessory_cost
  let final_books := final_goal / book_price
  halfway_books + final_books

/-- Theorem stating that Tara needs to sell 28 books in total to reach her goal. -/
theorem tara_book_sales : total_books_sold 10 90 5 20 = 28 := by
  sorry

end NUMINAMATH_CALUDE_tara_book_sales_l3708_370829


namespace NUMINAMATH_CALUDE_circumcircles_intersect_at_common_point_l3708_370801

-- Define the basic structures
structure Point : Type := (x y : ℝ)

structure Triangle : Type :=
  (A B C : Point)

structure Circle : Type :=
  (center : Point) (radius : ℝ)

-- Define the properties and conditions
def is_acute_triangle (t : Triangle) : Prop := sorry

def are_not_equal (p q : Point) : Prop := sorry

def is_midpoint (m : Point) (p q : Point) : Prop := sorry

def is_midpoint_of_minor_arc (m : Point) (a b c : Point) : Prop := sorry

def is_midpoint_of_major_arc (n : Point) (a b c : Point) : Prop := sorry

def is_incenter (w : Point) (t : Triangle) : Prop := sorry

def is_excenter (x : Point) (t : Triangle) (v : Point) : Prop := sorry

def circumcircle (t : Triangle) : Circle := sorry

def circles_intersect_at_point (c₁ c₂ c₃ : Circle) (p : Point) : Prop := sorry

-- State the theorem
theorem circumcircles_intersect_at_common_point
  (A B C D E F M N W X Y Z : Point) :
  is_acute_triangle (Triangle.mk A B C) →
  are_not_equal A B →
  are_not_equal A C →
  is_midpoint D B C →
  is_midpoint E C A →
  is_midpoint F A B →
  is_midpoint_of_minor_arc M B C A →
  is_midpoint_of_major_arc N B A C →
  is_incenter W (Triangle.mk D E F) →
  is_excenter X (Triangle.mk D E F) D →
  is_excenter Y (Triangle.mk D E F) E →
  is_excenter Z (Triangle.mk D E F) F →
  ∃ (P : Point),
    circles_intersect_at_point
      (circumcircle (Triangle.mk A B C))
      (circumcircle (Triangle.mk W N X))
      (circumcircle (Triangle.mk Y M Z))
      P :=
by
  sorry

end NUMINAMATH_CALUDE_circumcircles_intersect_at_common_point_l3708_370801


namespace NUMINAMATH_CALUDE_problem_statement_l3708_370889

noncomputable section

-- Define the functions f and g
def f (a : ℝ) (x : ℝ) : ℝ := a / x + x * Real.log x
def g (x : ℝ) : ℝ := x^3 - x^2 - 3

-- Define the intervals
def I : Set ℝ := Set.Icc 0 2
def J : Set ℝ := Set.Icc (1/2) 2

-- State the theorem
theorem problem_statement :
  (∃ M : ℤ, (M = 4 ∧ ∀ N : ℤ, N > M → ¬∃ x₁ x₂ : ℝ, x₁ ∈ I ∧ x₂ ∈ I ∧ g x₁ - g x₂ ≥ N)) ∧
  (∃ a : ℝ, (a = 1 ∧ ∀ s t : ℝ, s ∈ J → t ∈ J → f a s ≥ g t) ∧
            ∀ b : ℝ, b < a → ∃ s t : ℝ, s ∈ J ∧ t ∈ J ∧ f b s < g t) :=
by sorry

end

end NUMINAMATH_CALUDE_problem_statement_l3708_370889


namespace NUMINAMATH_CALUDE_kg_to_lb_conversion_rate_l3708_370852

/-- Conversion rate from kilograms to pounds -/
def kg_to_lb_rate : ℝ := 2.2

/-- Initial weight in kilograms -/
def initial_weight_kg : ℝ := 80

/-- Weight loss in pounds per hour of exercise -/
def weight_loss_per_hour : ℝ := 1.5

/-- Hours of exercise per day -/
def exercise_hours_per_day : ℝ := 2

/-- Number of days of exercise -/
def exercise_days : ℝ := 14

/-- Final weight in pounds after exercise period -/
def final_weight_lb : ℝ := 134

theorem kg_to_lb_conversion_rate :
  kg_to_lb_rate * initial_weight_kg =
    final_weight_lb + weight_loss_per_hour * exercise_hours_per_day * exercise_days :=
by sorry

end NUMINAMATH_CALUDE_kg_to_lb_conversion_rate_l3708_370852


namespace NUMINAMATH_CALUDE_sum_x_y_is_85_l3708_370807

/-- An arithmetic sequence with known terms 10, x, 30, y, 65 -/
structure ArithmeticSequence where
  x : ℝ
  y : ℝ
  isArithmetic : ∃ d : ℝ, x = 10 + d ∧ 30 = x + d ∧ y = 30 + 2*d ∧ 65 = y + d

/-- The sum of x and y in the arithmetic sequence is 85 -/
theorem sum_x_y_is_85 (seq : ArithmeticSequence) : seq.x + seq.y = 85 := by
  sorry

end NUMINAMATH_CALUDE_sum_x_y_is_85_l3708_370807


namespace NUMINAMATH_CALUDE_fixed_point_sum_l3708_370892

theorem fixed_point_sum (a : ℝ) (m n : ℝ) (h1 : a > 0) (h2 : a ≠ 1) :
  (fun x => a^(x - 1) + 2 : ℝ → ℝ) m = n → m + n = 4 := by
  sorry

end NUMINAMATH_CALUDE_fixed_point_sum_l3708_370892


namespace NUMINAMATH_CALUDE_rectangle_area_l3708_370851

/-- The area of a rectangle with length 2x and width 2x-1 is 4x^2 - 2x -/
theorem rectangle_area (x : ℝ) : 
  let length : ℝ := 2 * x
  let width : ℝ := 2 * x - 1
  length * width = 4 * x^2 - 2 * x := by
sorry

end NUMINAMATH_CALUDE_rectangle_area_l3708_370851


namespace NUMINAMATH_CALUDE_line_y_coordinate_l3708_370850

/-- A line passing through points (-12, y1) and (x2, 3) with x-intercept at (4, 0) has y1 = 0 -/
theorem line_y_coordinate (y1 x2 : ℝ) : 
  (∃ (m : ℝ), (3 - y1) = m * (x2 - (-12)) ∧ 
               0 - y1 = m * (4 - (-12)) ∧
               3 = m * (x2 - 4)) →
  y1 = 0 := by
sorry

end NUMINAMATH_CALUDE_line_y_coordinate_l3708_370850


namespace NUMINAMATH_CALUDE_total_books_l3708_370841

/-- The number of books on a mystery shelf -/
def mystery_books_per_shelf : ℕ := 7

/-- The number of books on a picture book shelf -/
def picture_books_per_shelf : ℕ := 5

/-- The number of books on a science fiction shelf -/
def scifi_books_per_shelf : ℕ := 8

/-- The number of books on a biography shelf -/
def biography_books_per_shelf : ℕ := 6

/-- The number of mystery shelves -/
def mystery_shelves : ℕ := 8

/-- The number of picture book shelves -/
def picture_shelves : ℕ := 2

/-- The number of science fiction shelves -/
def scifi_shelves : ℕ := 3

/-- The number of biography shelves -/
def biography_shelves : ℕ := 4

/-- The total number of books on Megan's shelves -/
theorem total_books : 
  mystery_books_per_shelf * mystery_shelves + 
  picture_books_per_shelf * picture_shelves + 
  scifi_books_per_shelf * scifi_shelves + 
  biography_books_per_shelf * biography_shelves = 114 := by
  sorry

end NUMINAMATH_CALUDE_total_books_l3708_370841


namespace NUMINAMATH_CALUDE_power_of_product_equals_product_of_powers_l3708_370872

theorem power_of_product_equals_product_of_powers (a : ℝ) :
  (3 * a^3)^2 = 9 * a^6 := by
  sorry

end NUMINAMATH_CALUDE_power_of_product_equals_product_of_powers_l3708_370872


namespace NUMINAMATH_CALUDE_sampled_classes_proportional_prob_at_least_one_grade12_prob_both_classes_selected_l3708_370808

/-- Represents the number of classes in each grade -/
structure GradeClasses where
  grade10 : Nat
  grade11 : Nat
  grade12 : Nat

/-- Represents the number of classes sampled from each grade -/
structure SampledClasses where
  grade10 : Nat
  grade11 : Nat
  grade12 : Nat

/-- The total number of classes across all grades -/
def totalClasses (gc : GradeClasses) : Nat :=
  gc.grade10 + gc.grade11 + gc.grade12

/-- The number of classes to be sampled -/
def totalSampled : Nat := 9

/-- The school's grade distribution -/
def schoolClasses : GradeClasses :=
  { grade10 := 16, grade11 := 12, grade12 := 8 }

/-- Theorem stating that the sampled classes are proportional to the total classes in each grade -/
theorem sampled_classes_proportional (sc : SampledClasses) :
    sc.grade10 * totalClasses schoolClasses = schoolClasses.grade10 * totalSampled ∧
    sc.grade11 * totalClasses schoolClasses = schoolClasses.grade11 * totalSampled ∧
    sc.grade12 * totalClasses schoolClasses = schoolClasses.grade12 * totalSampled :=
  sorry

/-- The probability of selecting at least one class from grade 12 -/
def probAtLeastOneGrade12 : Rat := 7 / 10

/-- Theorem stating the probability of selecting at least one class from grade 12 -/
theorem prob_at_least_one_grade12 (sc : SampledClasses) :
    probAtLeastOneGrade12 = 7 / 10 :=
  sorry

/-- The probability of selecting both class A from grade 11 and class B from grade 12 -/
def probBothClassesSelected : Rat := 1 / 6

/-- Theorem stating the probability of selecting both class A from grade 11 and class B from grade 12 -/
theorem prob_both_classes_selected (sc : SampledClasses) :
    probBothClassesSelected = 1 / 6 :=
  sorry

end NUMINAMATH_CALUDE_sampled_classes_proportional_prob_at_least_one_grade12_prob_both_classes_selected_l3708_370808


namespace NUMINAMATH_CALUDE_pirate_treasure_probability_l3708_370817

theorem pirate_treasure_probability : 
  let n : ℕ := 8  -- Total number of islands
  let k : ℕ := 4  -- Number of islands with treasure
  let p_treasure : ℚ := 1/3  -- Probability of treasure and no traps
  let p_neither : ℚ := 1/2  -- Probability of neither treasure nor traps
  Nat.choose n k * p_treasure^k * p_neither^(n-k) = 35/648 := by
sorry

end NUMINAMATH_CALUDE_pirate_treasure_probability_l3708_370817


namespace NUMINAMATH_CALUDE_helicopter_rental_cost_l3708_370854

/-- Calculates the total cost of helicopter rental given the specified conditions -/
theorem helicopter_rental_cost : 
  let hours_per_day : ℕ := 2
  let num_days : ℕ := 3
  let rate_day1 : ℚ := 85
  let rate_day2 : ℚ := 75
  let rate_day3 : ℚ := 65
  let discount_rate : ℚ := 0.05
  let cost_before_discount : ℚ := hours_per_day * (rate_day1 + rate_day2 + rate_day3)
  let discount : ℚ := discount_rate * cost_before_discount
  let total_cost : ℚ := cost_before_discount - discount
  total_cost = 427.5 := by sorry

end NUMINAMATH_CALUDE_helicopter_rental_cost_l3708_370854


namespace NUMINAMATH_CALUDE_line_tangent_to_circle_l3708_370890

/-- A circle with a diameter of 10 units -/
def Circle := {p : ℝ × ℝ | (p.1 ^ 2 + p.2 ^ 2) ≤ 25}

/-- A line at distance d from the origin -/
def Line (d : ℝ) := {p : ℝ × ℝ | p.2 = d}

/-- The line is tangent to the circle if and only if the distance is 5 -/
theorem line_tangent_to_circle (d : ℝ) : 
  (∃ (p : ℝ × ℝ), p ∈ Circle ∩ Line d ∧ 
    ∀ (q : ℝ × ℝ), q ∈ Circle ∩ Line d → q = p) ↔ 
  d = 5 :=
sorry

end NUMINAMATH_CALUDE_line_tangent_to_circle_l3708_370890


namespace NUMINAMATH_CALUDE_E_and_G_complementary_l3708_370896

/-- The sample space of selecting 3 products from 100 products. -/
def Ω : Type := Unit

/-- The probability measure on the sample space. -/
def P : Ω → ℝ := sorry

/-- The event that all 3 selected products are non-defective. -/
def E : Set Ω := sorry

/-- The event that all 3 selected products are defective. -/
def F : Set Ω := sorry

/-- The event that at least one of the 3 selected products is defective. -/
def G : Set Ω := sorry

/-- The total number of products. -/
def total_products : ℕ := 100

/-- The number of defective products. -/
def defective_products : ℕ := 5

/-- The number of products selected. -/
def selected_products : ℕ := 3

theorem E_and_G_complementary :
  E ∪ G = Set.univ ∧ E ∩ G = ∅ :=
sorry

end NUMINAMATH_CALUDE_E_and_G_complementary_l3708_370896


namespace NUMINAMATH_CALUDE_shooting_scores_l3708_370849

def scores_A : List ℝ := [4, 5, 5, 6, 6, 7, 7, 8, 8, 9]
def scores_B : List ℝ := [2, 5, 6, 6, 7, 7, 7, 8, 9, 10]

theorem shooting_scores :
  let avg_A := (scores_A.sum) / (scores_A.length : ℝ)
  let avg_B := (scores_B.sum) / (scores_B.length : ℝ)
  let avg_total := ((scores_A ++ scores_B).sum) / ((scores_A ++ scores_B).length : ℝ)
  avg_A < avg_B ∧ avg_total = 6.6 := by sorry

end NUMINAMATH_CALUDE_shooting_scores_l3708_370849


namespace NUMINAMATH_CALUDE_complex_equation_solution_l3708_370846

theorem complex_equation_solution (a : ℝ) : 
  (2 + a * Complex.I) / (1 + Complex.I) = (3 : ℂ) + Complex.I → a = 4 := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l3708_370846


namespace NUMINAMATH_CALUDE_trig_expression_simplification_l3708_370860

theorem trig_expression_simplification (x : Real) :
  x = π / 4 →
  (1 + Real.sin (x + π / 4) - Real.cos (x + π / 4)) / 
  (1 + Real.sin (x + π / 4) + Real.cos (x + π / 4)) = 1 := by
  sorry

end NUMINAMATH_CALUDE_trig_expression_simplification_l3708_370860


namespace NUMINAMATH_CALUDE_hyperbola_eccentricity_l3708_370844

/-- Given a hyperbola with equation x²/a² - y²/b² = 1 (a > 0, b > 0) 
    and asymptote equations y = ±x, its eccentricity is √2 -/
theorem hyperbola_eccentricity (a b : ℝ) (ha : a > 0) (hb : b > 0) 
  (h_asymptote : ∀ (x y : ℝ), (y = x ∨ y = -x) → (x^2 / a^2 - y^2 / b^2 = 1)) :
  let e := Real.sqrt ((a^2 + b^2) / a^2)
  e = Real.sqrt 2 := by sorry

end NUMINAMATH_CALUDE_hyperbola_eccentricity_l3708_370844


namespace NUMINAMATH_CALUDE_find_m_find_k_l3708_370813

-- Define the vectors
def a : ℝ × ℝ := (1, -3)
def b (m : ℝ) : ℝ × ℝ := (-2, m)

-- Define dot product
def dot_product (v w : ℝ × ℝ) : ℝ := v.1 * w.1 + v.2 * w.2

-- Define vector subtraction
def vec_sub (v w : ℝ × ℝ) : ℝ × ℝ := (v.1 - w.1, v.2 - w.2)

-- Define vector addition
def vec_add (v w : ℝ × ℝ) : ℝ × ℝ := (v.1 + w.1, v.2 + w.2)

-- Define scalar multiplication
def scalar_mul (k : ℝ) (v : ℝ × ℝ) : ℝ × ℝ := (k * v.1, k * v.2)

-- Define parallel vectors
def parallel (v w : ℝ × ℝ) : Prop := ∃ (k : ℝ), v = scalar_mul k w

-- Theorem 1: Find the value of m
theorem find_m :
  ∃ (m : ℝ), dot_product a (vec_sub a (b m)) = 0 ∧ m = -4 :=
sorry

-- Theorem 2: Find the value of k
theorem find_k :
  ∃ (k : ℝ), parallel (vec_add (scalar_mul k a) (b (-4))) (vec_sub a (b (-4))) ∧ k = -1 :=
sorry

end NUMINAMATH_CALUDE_find_m_find_k_l3708_370813


namespace NUMINAMATH_CALUDE_square_of_difference_l3708_370836

theorem square_of_difference (y : ℝ) (h : y^2 ≥ 49) :
  (7 - Real.sqrt (y^2 - 49))^2 = y^2 - 14 * Real.sqrt (y^2 - 49) := by
  sorry

end NUMINAMATH_CALUDE_square_of_difference_l3708_370836


namespace NUMINAMATH_CALUDE_g_neg_three_l3708_370898

def g (x : ℝ) : ℝ := x^2 - x + 2*x^3

theorem g_neg_three : g (-3) = -42 := by sorry

end NUMINAMATH_CALUDE_g_neg_three_l3708_370898


namespace NUMINAMATH_CALUDE_angle_in_third_quadrant_l3708_370869

theorem angle_in_third_quadrant (α : Real) : 
  (Real.sin α * Real.tan α < 0) → 
  (Real.cos α / Real.tan α < 0) → 
  (α > Real.pi ∧ α < 3 * Real.pi / 2) := by
sorry

end NUMINAMATH_CALUDE_angle_in_third_quadrant_l3708_370869


namespace NUMINAMATH_CALUDE_t_shaped_area_concrete_t_shaped_area_l3708_370843

/-- The area of a T-shaped region formed by subtracting three smaller rectangles from a larger rectangle -/
theorem t_shaped_area (a b c d e f : ℕ) : 
  a * b - (c * d + e * f + c * (b - f)) = 24 :=
by
  sorry

/-- Concrete instance of the T-shaped area theorem -/
theorem concrete_t_shaped_area : 
  8 * 6 - (2 * 2 + 4 * 2 + 2 * 6) = 24 :=
by
  sorry

end NUMINAMATH_CALUDE_t_shaped_area_concrete_t_shaped_area_l3708_370843


namespace NUMINAMATH_CALUDE_selection_theorem_l3708_370821

/-- The number of ways to select 3 people from 4 male and 3 female students, ensuring both genders are represented. -/
def selection_ways (male_count female_count : ℕ) (total_selected : ℕ) : ℕ :=
  Nat.choose (male_count + female_count) total_selected -
  Nat.choose male_count total_selected -
  Nat.choose female_count total_selected

/-- Theorem stating that the number of ways to select 3 people from 4 male and 3 female students, ensuring both genders are represented, is 30. -/
theorem selection_theorem :
  selection_ways 4 3 3 = 30 := by
  sorry

end NUMINAMATH_CALUDE_selection_theorem_l3708_370821
