import Mathlib

namespace NUMINAMATH_CALUDE_quadratic_roots_relation_l3864_386486

theorem quadratic_roots_relation (a b : ℝ) (r₁ r₂ : ℂ) : 
  (∀ x : ℂ, x^2 + a*x + b = 0 ↔ x = r₁ ∨ x = r₂) →
  (∀ x : ℂ, x^2 + b*x + a = 0 ↔ x = 3*r₁ ∨ x = 3*r₂) →
  a/b = -3 := by
sorry

end NUMINAMATH_CALUDE_quadratic_roots_relation_l3864_386486


namespace NUMINAMATH_CALUDE_lunch_break_is_60_minutes_l3864_386489

/-- Represents the painting rates and work done on each day -/
structure PaintingData where
  paula_rate : ℝ
  helpers_rate : ℝ
  day1_hours : ℝ
  day1_work : ℝ
  day2_hours : ℝ
  day2_work : ℝ
  day3_hours : ℝ
  day3_work : ℝ

/-- The lunch break duration in hours -/
def lunch_break : ℝ := 1

/-- Theorem stating that the lunch break is 60 minutes given the painting data -/
theorem lunch_break_is_60_minutes (data : PaintingData) : 
  (data.day1_hours - lunch_break) * (data.paula_rate + data.helpers_rate) = data.day1_work ∧
  (data.day2_hours - lunch_break) * data.helpers_rate = data.day2_work ∧
  (data.day3_hours - lunch_break) * data.paula_rate = data.day3_work →
  lunch_break * 60 = 60 := by
  sorry

#eval lunch_break * 60  -- Should output 60

end NUMINAMATH_CALUDE_lunch_break_is_60_minutes_l3864_386489


namespace NUMINAMATH_CALUDE_bbq_cooking_time_l3864_386447

/-- Calculates the time required to cook burgers for a BBQ --/
theorem bbq_cooking_time 
  (cooking_time_per_side : ℕ) 
  (grill_capacity : ℕ) 
  (total_guests : ℕ) 
  (guests_wanting_two : ℕ) 
  (guests_wanting_one : ℕ) 
  (h1 : cooking_time_per_side = 4)
  (h2 : grill_capacity = 5)
  (h3 : total_guests = 30)
  (h4 : guests_wanting_two = total_guests / 2)
  (h5 : guests_wanting_one = total_guests / 2)
  : (((guests_wanting_two * 2 + guests_wanting_one) / grill_capacity) * 
     (cooking_time_per_side * 2)) = 72 := by
  sorry

end NUMINAMATH_CALUDE_bbq_cooking_time_l3864_386447


namespace NUMINAMATH_CALUDE_problem_solution_l3864_386427

theorem problem_solution (x y : ℝ) 
  (h1 : x^2 + y^2 - x*y = 2) 
  (h2 : x^4 + y^4 + x^2*y^2 = 8) : 
  x^8 + y^8 + x^2014*y^2014 = 48 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l3864_386427


namespace NUMINAMATH_CALUDE_greatest_fraction_l3864_386495

theorem greatest_fraction : 
  let f1 := (3 : ℚ) / 10
  let f2 := (4 : ℚ) / 7
  let f3 := (5 : ℚ) / 23
  let f4 := (2 : ℚ) / 3
  let f5 := (1 : ℚ) / 2
  f4 > f1 ∧ f4 > f2 ∧ f4 > f3 ∧ f4 > f5 := by sorry

end NUMINAMATH_CALUDE_greatest_fraction_l3864_386495


namespace NUMINAMATH_CALUDE_cassini_oval_properties_l3864_386442

-- Define the curve Γ
def Γ (m : ℝ) (x y : ℝ) : Prop :=
  Real.sqrt ((x + 1)^2 + y^2) * Real.sqrt ((x - 1)^2 + y^2) = m ∧ m > 0

-- Define a single-track curve
def SingleTrackCurve (C : ℝ → ℝ → Prop) : Prop :=
  ∃ (f : ℝ → ℝ), ∀ x, C x (f x)

-- Define a double-track curve
def DoubleTrackCurve (C : ℝ → ℝ → Prop) : Prop :=
  ∃ (f g : ℝ → ℝ), (∀ x, C x (f x) ∨ C x (g x)) ∧ 
  (∃ x, f x ≠ g x)

-- The main theorem
theorem cassini_oval_properties :
  (∃ m : ℝ, m > 1 ∧ SingleTrackCurve (Γ m)) ∧
  (∃ m : ℝ, 0 < m ∧ m < 1 ∧ DoubleTrackCurve (Γ m)) := by
  sorry

end NUMINAMATH_CALUDE_cassini_oval_properties_l3864_386442


namespace NUMINAMATH_CALUDE_baker_cakes_left_l3864_386491

theorem baker_cakes_left (total_cakes sold_cakes : ℕ) 
  (h1 : total_cakes = 54)
  (h2 : sold_cakes = 41) :
  total_cakes - sold_cakes = 13 := by
  sorry

end NUMINAMATH_CALUDE_baker_cakes_left_l3864_386491


namespace NUMINAMATH_CALUDE_unique_quadratic_root_l3864_386465

theorem unique_quadratic_root (k : ℝ) : 
  (∃! x : ℝ, x^2 - 4*x + k = 0) → k = 4 := by
  sorry

end NUMINAMATH_CALUDE_unique_quadratic_root_l3864_386465


namespace NUMINAMATH_CALUDE_tan_alpha_value_l3864_386426

theorem tan_alpha_value (α : Real) (h : Real.tan α = 3) :
  (1 + Real.cos α ^ 2) / (Real.sin α * Real.cos α + Real.sin α ^ 2) = 11/12 := by
  sorry

end NUMINAMATH_CALUDE_tan_alpha_value_l3864_386426


namespace NUMINAMATH_CALUDE_triangle_side_length_l3864_386468

theorem triangle_side_length (a c : ℝ) (B : ℝ) (h1 : a = 2) (h2 : c = 4) (h3 : B = π / 3) :
  let b := Real.sqrt (a^2 + c^2 - 2*a*c*(Real.cos B))
  b = 2 * Real.sqrt 3 := by
sorry

end NUMINAMATH_CALUDE_triangle_side_length_l3864_386468


namespace NUMINAMATH_CALUDE_babysitting_earnings_l3864_386431

/-- Represents the babysitting rates based on child's age --/
def BabysittingRate : ℕ → ℕ
  | age => if age < 2 then 5 else if age ≤ 5 then 7 else 8

/-- Calculates the total earnings from babysitting --/
def TotalEarnings (childrenAges : List ℕ) (hours : List ℕ) : ℕ :=
  List.sum (List.zipWith (λ age hour => BabysittingRate age * hour) childrenAges hours)

theorem babysitting_earnings :
  let janeStartAge : ℕ := 18
  let childA : ℕ := janeStartAge / 2
  let childB : ℕ := childA - 2
  let childC : ℕ := childB + 3
  let childD : ℕ := childC
  let childrenAges : List ℕ := [childA, childB, childC, childD]
  let hours : List ℕ := [50, 90, 130, 70]
  TotalEarnings childrenAges hours = 2720 := by
  sorry


end NUMINAMATH_CALUDE_babysitting_earnings_l3864_386431


namespace NUMINAMATH_CALUDE_rectangular_prism_surface_area_l3864_386443

/-- The surface area of a rectangular prism -/
def surface_area (l w h : ℝ) : ℝ := 2 * (l * w + l * h + w * h)

/-- Theorem: The surface area of a rectangular prism with dimensions 10, 6, and 5 is 280 -/
theorem rectangular_prism_surface_area :
  surface_area 10 6 5 = 280 := by
  sorry

end NUMINAMATH_CALUDE_rectangular_prism_surface_area_l3864_386443


namespace NUMINAMATH_CALUDE_cows_ran_away_theorem_l3864_386458

/-- Represents the number of cows that ran away from a farm --/
def cows_that_ran_away (initial_cows : ℕ) (initial_days : ℕ) (days_passed : ℕ) (remaining_cows : ℕ) : ℕ :=
  initial_cows - remaining_cows

/-- Theorem stating the number of cows that ran away under given conditions --/
theorem cows_ran_away_theorem (initial_cows : ℕ) (initial_days : ℕ) (days_passed : ℕ) :
  initial_cows = 1000 →
  initial_days = 50 →
  days_passed = 10 →
  (initial_cows * initial_days - initial_cows * days_passed) = 
    (initial_cows - cows_that_ran_away initial_cows initial_days days_passed (initial_cows - 200)) * initial_days →
  cows_that_ran_away initial_cows initial_days days_passed (initial_cows - 200) = 200 :=
by
  sorry

#eval cows_that_ran_away 1000 50 10 800

end NUMINAMATH_CALUDE_cows_ran_away_theorem_l3864_386458


namespace NUMINAMATH_CALUDE_complex_square_equality_l3864_386401

theorem complex_square_equality (a b : ℝ) (i : ℂ) 
  (h1 : i^2 = -1) 
  (h2 : a + i = 2 - b*i) : 
  (a + b*i)^2 = 3 - 4*i := by
sorry

end NUMINAMATH_CALUDE_complex_square_equality_l3864_386401


namespace NUMINAMATH_CALUDE_cubic_root_reciprocal_squares_sum_l3864_386480

theorem cubic_root_reciprocal_squares_sum (p q : ℂ) (z₁ z₂ z₃ : ℂ) : 
  z₁^3 + p*z₁ + q = 0 → 
  z₂^3 + p*z₂ + q = 0 → 
  z₃^3 + p*z₃ + q = 0 → 
  z₁ ≠ z₂ → z₂ ≠ z₃ → z₃ ≠ z₁ →
  q ≠ 0 →
  1/z₁^2 + 1/z₂^2 + 1/z₃^2 = p^2 / q^2 := by
sorry

end NUMINAMATH_CALUDE_cubic_root_reciprocal_squares_sum_l3864_386480


namespace NUMINAMATH_CALUDE_dice_probability_l3864_386400

theorem dice_probability (p_neither : ℚ) (h : p_neither = 4/9) : 
  1 - p_neither = 5/9 := by
  sorry

end NUMINAMATH_CALUDE_dice_probability_l3864_386400


namespace NUMINAMATH_CALUDE_family_brownie_consumption_percentage_l3864_386449

theorem family_brownie_consumption_percentage
  (total_brownies : ℕ)
  (children_consumption_percentage : ℚ)
  (lorraine_extra_consumption : ℕ)
  (leftover_brownies : ℕ)
  (h1 : total_brownies = 16)
  (h2 : children_consumption_percentage = 1/4)
  (h3 : lorraine_extra_consumption = 1)
  (h4 : leftover_brownies = 5) :
  let remaining_after_children := total_brownies - (children_consumption_percentage * total_brownies).num
  let family_consumption := remaining_after_children - leftover_brownies - lorraine_extra_consumption
  (family_consumption : ℚ) / remaining_after_children = 1/2 :=
sorry

end NUMINAMATH_CALUDE_family_brownie_consumption_percentage_l3864_386449


namespace NUMINAMATH_CALUDE_equation_condition_l3864_386461

theorem equation_condition (a b c : ℕ) 
  (ha : 0 < a ∧ a < 10) 
  (hb : 0 < b ∧ b < 10) 
  (hc : 0 < c ∧ c < 10) : 
  (11 * a + b) * (11 * a + c) = 121 * a * (a + 1) + 11 * b * c ↔ b + c = 11 :=
sorry

end NUMINAMATH_CALUDE_equation_condition_l3864_386461


namespace NUMINAMATH_CALUDE_no_real_roots_equation_implies_value_l3864_386437

theorem no_real_roots_equation_implies_value (a b : ℝ) :
  (∀ x : ℝ, x ≠ 0 ∧ x ≠ 1 → (x / (x - 1) + (x - 1) / x ≠ (a + b * x) / (x^2 - x))) →
  8 * a + 4 * b - 5 = 11 := by
sorry

end NUMINAMATH_CALUDE_no_real_roots_equation_implies_value_l3864_386437


namespace NUMINAMATH_CALUDE_proportional_survey_distribution_l3864_386436

/-- Represents the number of surveys to be drawn from a group -/
def surveyCount (totalSurveys : ℕ) (groupSize : ℕ) (totalPopulation : ℕ) : ℕ :=
  (totalSurveys * groupSize) / totalPopulation

theorem proportional_survey_distribution 
  (totalSurveys : ℕ) 
  (facultyStaffSize juniorHighSize seniorHighSize : ℕ) 
  (h1 : totalSurveys = 120)
  (h2 : facultyStaffSize = 500)
  (h3 : juniorHighSize = 3000)
  (h4 : seniorHighSize = 4000) :
  let totalPopulation := facultyStaffSize + juniorHighSize + seniorHighSize
  (surveyCount totalSurveys facultyStaffSize totalPopulation = 8) ∧ 
  (surveyCount totalSurveys juniorHighSize totalPopulation = 48) ∧
  (surveyCount totalSurveys seniorHighSize totalPopulation = 64) :=
by sorry

#check proportional_survey_distribution

end NUMINAMATH_CALUDE_proportional_survey_distribution_l3864_386436


namespace NUMINAMATH_CALUDE_complex_on_line_l3864_386441

/-- Given a complex number z = (m-1) + (m+2)i that corresponds to a point on the line 2x-y=0,
    prove that m = 4. -/
theorem complex_on_line (m : ℝ) : 
  let z : ℂ := Complex.mk (m - 1) (m + 2)
  2 * z.re - z.im = 0 → m = 4 := by
  sorry

end NUMINAMATH_CALUDE_complex_on_line_l3864_386441


namespace NUMINAMATH_CALUDE_student_multiplication_problem_l3864_386450

theorem student_multiplication_problem (x : ℝ) (y : ℝ) 
  (h1 : x = 129)
  (h2 : x * y - 148 = 110) : 
  y = 2 := by
  sorry

end NUMINAMATH_CALUDE_student_multiplication_problem_l3864_386450


namespace NUMINAMATH_CALUDE_equal_one_and_two_digit_prob_l3864_386452

def num_sides : ℕ := 15
def num_dice : ℕ := 5

def prob_one_digit : ℚ := 3 / 5
def prob_two_digit : ℚ := 2 / 5

def choose (n k : ℕ) : ℕ := (Nat.factorial n) / ((Nat.factorial k) * (Nat.factorial (n - k)))

theorem equal_one_and_two_digit_prob :
  (choose num_dice (num_dice / 2)) * (prob_two_digit ^ (num_dice / 2)) * (prob_one_digit ^ (num_dice / 2 + 1)) = 108 / 625 :=
sorry

end NUMINAMATH_CALUDE_equal_one_and_two_digit_prob_l3864_386452


namespace NUMINAMATH_CALUDE_plate_cutting_theorem_l3864_386408

def can_measure (weights : List ℕ) (target : ℕ) : Prop :=
  ∃ (pos neg : List ℕ), pos.sum - neg.sum = target ∧ pos.toFinset ∪ neg.toFinset ⊆ weights.toFinset

theorem plate_cutting_theorem :
  let weights := [1, 3, 7]
  ∀ n : ℕ, 1 ≤ n ∧ n ≤ 11 → can_measure weights n :=
by sorry

end NUMINAMATH_CALUDE_plate_cutting_theorem_l3864_386408


namespace NUMINAMATH_CALUDE_triangle_value_l3864_386417

theorem triangle_value (p : ℤ) (h1 : ∃ triangle : ℤ, triangle + p = 67) 
  (h2 : ∃ triangle : ℤ, 3 * (triangle + p) - p = 185) : 
  ∃ triangle : ℤ, triangle = 51 := by
sorry

end NUMINAMATH_CALUDE_triangle_value_l3864_386417


namespace NUMINAMATH_CALUDE_grid_domino_coverage_l3864_386460

/-- Represents a 5x5 grid with a square removed at (i, j) -/
structure Grid :=
  (i : Nat) (j : Nat)

/-- Predicate to check if a number is odd -/
def is_odd (n : Nat) : Prop := n % 2 = 1

/-- Predicate to check if the grid can be covered by dominoes -/
def can_cover_with_dominoes (g : Grid) : Prop :=
  is_odd g.i ∧ is_odd g.j

theorem grid_domino_coverage (g : Grid) :
  (g.i ≤ 5 ∧ g.j ≤ 5) →
  (can_cover_with_dominoes g ↔ (is_odd g.i ∧ is_odd g.j)) :=
sorry

end NUMINAMATH_CALUDE_grid_domino_coverage_l3864_386460


namespace NUMINAMATH_CALUDE_sqrt_one_minus_two_sin_two_cos_two_l3864_386423

theorem sqrt_one_minus_two_sin_two_cos_two (h : π / 2 < 2 ∧ 2 < π) :
  Real.sqrt (1 - 2 * Real.sin 2 * Real.cos 2) = Real.sin 2 - Real.cos 2 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_one_minus_two_sin_two_cos_two_l3864_386423


namespace NUMINAMATH_CALUDE_circle_triangle_area_ratio_l3864_386492

/-- For a right triangle circumscribed about a circle -/
theorem circle_triangle_area_ratio
  (h a b R : ℝ)
  (h_positive : h > 0)
  (R_positive : R > 0)
  (right_triangle : a^2 + b^2 = h^2)
  (circumradius : R = h / 2) :
  π * R^2 / (a * b / 2) = π * h / (4 * R) :=
by sorry

end NUMINAMATH_CALUDE_circle_triangle_area_ratio_l3864_386492


namespace NUMINAMATH_CALUDE_single_layer_cake_cost_l3864_386410

/-- The cost of a single layer cake slice -/
def single_layer_cost : ℝ := 4

/-- The cost of a double layer cake slice -/
def double_layer_cost : ℝ := 7

/-- The number of single layer cake slices bought -/
def single_layer_count : ℕ := 7

/-- The number of double layer cake slices bought -/
def double_layer_count : ℕ := 5

/-- The total amount spent -/
def total_spent : ℝ := 63

theorem single_layer_cake_cost :
  single_layer_cost * single_layer_count + double_layer_cost * double_layer_count = total_spent :=
by sorry

end NUMINAMATH_CALUDE_single_layer_cake_cost_l3864_386410


namespace NUMINAMATH_CALUDE_vector_sum_length_one_l3864_386440

theorem vector_sum_length_one (x : Real) :
  let a := (Real.cos (3 * x / 2), Real.sin (3 * x / 2))
  let b := (Real.cos (x / 2), -Real.sin (x / 2))
  (0 ≤ x) ∧ (x ≤ Real.pi) →
  (a.1 + b.1)^2 + (a.2 + b.2)^2 = 1 →
  x = Real.pi / 3 ∨ x = 2 * Real.pi / 3 := by
sorry

end NUMINAMATH_CALUDE_vector_sum_length_one_l3864_386440


namespace NUMINAMATH_CALUDE_harper_mineral_water_cost_l3864_386477

/-- The amount Harper spends on mineral water for 240 days -/
def mineral_water_cost (daily_consumption : ℚ) (bottles_per_case : ℕ) (case_cost : ℚ) (total_days : ℕ) : ℚ :=
  let cases_needed := (total_days : ℚ) * daily_consumption / bottles_per_case
  cases_needed.ceil * case_cost

/-- Theorem stating the cost of mineral water for Harper -/
theorem harper_mineral_water_cost :
  mineral_water_cost (1/2) 24 12 240 = 60 := by
  sorry

end NUMINAMATH_CALUDE_harper_mineral_water_cost_l3864_386477


namespace NUMINAMATH_CALUDE_angle_D_value_l3864_386412

-- Define the angles as real numbers (in degrees)
variable (A B C D : ℝ)

-- State the theorem
theorem angle_D_value 
  (h1 : A + B = 180)
  (h2 : C = 2 * D)
  (h3 : A = 100)
  (h4 : B + C + D = 180) :
  D = 100 / 3 := by
  sorry

end NUMINAMATH_CALUDE_angle_D_value_l3864_386412


namespace NUMINAMATH_CALUDE_least_positive_integer_divisible_by_four_primes_l3864_386497

theorem least_positive_integer_divisible_by_four_primes : 
  ∃ (n : ℕ), n > 0 ∧ 
  (∃ (p₁ p₂ p₃ p₄ : ℕ), Prime p₁ ∧ Prime p₂ ∧ Prime p₃ ∧ Prime p₄ ∧ 
    p₁ ≠ p₂ ∧ p₁ ≠ p₃ ∧ p₁ ≠ p₄ ∧ p₂ ≠ p₃ ∧ p₂ ≠ p₄ ∧ p₃ ≠ p₄ ∧
    p₁ ∣ n ∧ p₂ ∣ n ∧ p₃ ∣ n ∧ p₄ ∣ n) ∧
  (∀ m : ℕ, m > 0 ∧ m < n → 
    ¬(∃ (q₁ q₂ q₃ q₄ : ℕ), Prime q₁ ∧ Prime q₂ ∧ Prime q₃ ∧ Prime q₄ ∧ 
      q₁ ≠ q₂ ∧ q₁ ≠ q₃ ∧ q₁ ≠ q₄ ∧ q₂ ≠ q₃ ∧ q₂ ≠ q₄ ∧ q₃ ≠ q₄ ∧
      q₁ ∣ m ∧ q₂ ∣ m ∧ q₃ ∣ m ∧ q₄ ∣ m)) ∧
  n = 210 :=
by
  sorry

end NUMINAMATH_CALUDE_least_positive_integer_divisible_by_four_primes_l3864_386497


namespace NUMINAMATH_CALUDE_bethany_riding_time_l3864_386494

/-- Represents the number of hours in a day -/
def hours_in_day : ℕ := 24

/-- Represents the number of minutes in an hour -/
def minutes_in_hour : ℕ := 60

/-- Represents the number of days in two weeks -/
def days_in_two_weeks : ℕ := 14

/-- Represents Bethany's riding schedule -/
structure RidingSchedule where
  monday : ℕ     -- minutes ridden on Monday
  wednesday : ℕ  -- minutes ridden on Wednesday
  friday : ℕ     -- minutes ridden on Friday
  tuesday : ℕ    -- minutes ridden on Tuesday
  thursday : ℕ   -- minutes ridden on Thursday
  saturday : ℕ   -- minutes ridden on Saturday

/-- Calculates the total minutes ridden in two weeks -/
def total_minutes (schedule : RidingSchedule) : ℕ :=
  2 * (schedule.monday + schedule.wednesday + schedule.friday + 
       schedule.tuesday + schedule.thursday + schedule.saturday)

/-- Theorem stating Bethany's riding time on Monday, Wednesday, and Friday -/
theorem bethany_riding_time (schedule : RidingSchedule) 
  (h1 : schedule.tuesday = 30)
  (h2 : schedule.thursday = 30)
  (h3 : schedule.saturday = 2 * minutes_in_hour)
  (h4 : total_minutes schedule = 12 * minutes_in_hour) :
  2 * (schedule.monday + schedule.wednesday + schedule.friday) = 6 * minutes_in_hour := by
  sorry

#check bethany_riding_time

end NUMINAMATH_CALUDE_bethany_riding_time_l3864_386494


namespace NUMINAMATH_CALUDE_worker_d_rate_l3864_386462

-- Define work rates for workers a, b, c, and d
variable (A B C D : ℚ)

-- Define the conditions
def condition1 : Prop := A + B = 1 / 15
def condition2 : Prop := A + B + C = 1 / 12
def condition3 : Prop := C + D = 1 / 20

-- Theorem statement
theorem worker_d_rate 
  (h1 : condition1 A B) 
  (h2 : condition2 A B C) 
  (h3 : condition3 C D) : 
  D = 1 / 30 := by sorry

end NUMINAMATH_CALUDE_worker_d_rate_l3864_386462


namespace NUMINAMATH_CALUDE_function_minimum_at_three_l3864_386424

/-- The function f(x) = x(x - c)^2 has a minimum value at x = 3 if and only if c = 3 -/
theorem function_minimum_at_three (c : ℝ) : 
  (∀ x, x * (x - c)^2 ≥ 3 * (3 - c)^2) ↔ c = 3 := by sorry

end NUMINAMATH_CALUDE_function_minimum_at_three_l3864_386424


namespace NUMINAMATH_CALUDE_vector_product_l3864_386463

/-- Given vectors a and b, if |a| = 2 and a ⊥ b, then mn = -6 -/
theorem vector_product (m n : ℝ) : 
  let a : ℝ × ℝ := (m, 1)
  let b : ℝ × ℝ := (2, n)
  (a.1^2 + a.2^2 = 4) → -- |a| = 2
  (a.1 * b.1 + a.2 * b.2 = 0) → -- a ⊥ b
  m * n = -6 := by
sorry

end NUMINAMATH_CALUDE_vector_product_l3864_386463


namespace NUMINAMATH_CALUDE_complex_modulus_l3864_386479

theorem complex_modulus (a b : ℝ) (z : ℂ) :
  (a + Complex.I)^2 = b * Complex.I →
  z = a + b * Complex.I →
  Complex.abs z = Real.sqrt 5 := by
sorry

end NUMINAMATH_CALUDE_complex_modulus_l3864_386479


namespace NUMINAMATH_CALUDE_largest_prime_factor_of_sum_of_divisors_180_l3864_386483

def sum_of_divisors (n : ℕ) : ℕ := sorry

theorem largest_prime_factor_of_sum_of_divisors_180 :
  ∃ (p : ℕ), Nat.Prime p ∧ p ∣ sum_of_divisors 180 ∧
  ∀ (q : ℕ), Nat.Prime q → q ∣ sum_of_divisors 180 → q ≤ p ∧ p = 13 :=
sorry

end NUMINAMATH_CALUDE_largest_prime_factor_of_sum_of_divisors_180_l3864_386483


namespace NUMINAMATH_CALUDE_f_is_even_and_increasing_l3864_386476

def f (x : ℝ) : ℝ := |x| + 1

theorem f_is_even_and_increasing :
  (∀ x : ℝ, f x = f (-x)) ∧
  (∀ x y : ℝ, 0 < x → x < y → f x < f y) :=
by sorry

end NUMINAMATH_CALUDE_f_is_even_and_increasing_l3864_386476


namespace NUMINAMATH_CALUDE_square_sum_equals_three_times_product_l3864_386419

theorem square_sum_equals_three_times_product
  (x y : ℝ) 
  (h1 : 1/x + 1/y = 5) 
  (h2 : x + y = 5*x*y) : 
  x^2 + y^2 = 3*x*y :=
by
  sorry

end NUMINAMATH_CALUDE_square_sum_equals_three_times_product_l3864_386419


namespace NUMINAMATH_CALUDE_quadratic_inequality_equivalent_to_interval_l3864_386482

theorem quadratic_inequality_equivalent_to_interval (x : ℝ) :
  x^2 - 5*x + 6 < 0 ↔ 2 < x ∧ x < 3 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_equivalent_to_interval_l3864_386482


namespace NUMINAMATH_CALUDE_star_property_l3864_386499

def star (m n : ℝ) : ℝ := (3 * m - 2 * n)^2

theorem star_property (x y : ℝ) : star ((3 * x - 2 * y)^2) ((2 * y - 3 * x)^2) = (3 * x - 2 * y)^4 := by
  sorry

end NUMINAMATH_CALUDE_star_property_l3864_386499


namespace NUMINAMATH_CALUDE_constant_value_l3864_386403

-- Define the function f
def f (x : ℝ) : ℝ := x + 4

-- Define the equation
def equation (x : ℝ) (c : ℝ) : Prop :=
  (3 * f (x - 2)) / f 0 + c = f (2 * x + 1)

-- Theorem statement
theorem constant_value :
  ∃ (c : ℝ), equation 0.4 c ∧ ∀ (x : ℝ), equation x c → x = 0.4 :=
by sorry

end NUMINAMATH_CALUDE_constant_value_l3864_386403


namespace NUMINAMATH_CALUDE_intersection_point_y_coordinate_l3864_386487

theorem intersection_point_y_coordinate : ∃ (x : ℝ), 
  0 < x ∧ x < π / 2 ∧ 
  2 + 3 * Real.cos (2 * x) = 3 * Real.sqrt 3 * Real.sin x ∧
  2 + 3 * Real.cos (2 * x) = 3 := by sorry

end NUMINAMATH_CALUDE_intersection_point_y_coordinate_l3864_386487


namespace NUMINAMATH_CALUDE_product_inequality_l3864_386406

theorem product_inequality (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (h_abc : a * b * c = 1) :
  (a - 1 + 1 / b) * (b - 1 + 1 / c) * (c - 1 + 1 / a) ≤ 1 := by
  sorry

end NUMINAMATH_CALUDE_product_inequality_l3864_386406


namespace NUMINAMATH_CALUDE_odd_perfect_square_l3864_386402

theorem odd_perfect_square (n : ℕ+) 
  (h : (Finset.sum (Nat.divisors n.val) id) = 2 * n.val + 1) : 
  ∃ (k : ℕ), n.val = 2 * k + 1 ∧ ∃ (m : ℕ), n.val = m ^ 2 :=
sorry

end NUMINAMATH_CALUDE_odd_perfect_square_l3864_386402


namespace NUMINAMATH_CALUDE_absolute_value_inequality_l3864_386414

theorem absolute_value_inequality (x : ℝ) : |2*x + 1| - 2*|x - 1| > 0 ↔ x > (1/4 : ℝ) := by
  sorry

end NUMINAMATH_CALUDE_absolute_value_inequality_l3864_386414


namespace NUMINAMATH_CALUDE_order_of_numbers_l3864_386493

def w : ℕ := 2^129 * 3^81 * 5^128
def x : ℕ := 2^127 * 3^81 * 5^128
def y : ℕ := 2^126 * 3^82 * 5^128
def z : ℕ := 2^125 * 3^82 * 5^129

theorem order_of_numbers : x < y ∧ y < z ∧ z < w := by sorry

end NUMINAMATH_CALUDE_order_of_numbers_l3864_386493


namespace NUMINAMATH_CALUDE_philip_paintings_l3864_386498

/-- Calculates the total number of paintings Philip will have after a given number of days -/
def total_paintings (paintings_per_day : ℕ) (initial_paintings : ℕ) (days : ℕ) : ℕ :=
  initial_paintings + paintings_per_day * days

/-- Theorem: Philip will have 80 paintings after 30 days -/
theorem philip_paintings :
  total_paintings 2 20 30 = 80 := by
  sorry

end NUMINAMATH_CALUDE_philip_paintings_l3864_386498


namespace NUMINAMATH_CALUDE_quadratic_roots_condition_l3864_386454

theorem quadratic_roots_condition (c : ℝ) : 
  (∀ x : ℝ, x^2 - 3*x + c = 0 ↔ x = (3 + Real.sqrt c) / 2 ∨ x = (3 - Real.sqrt c) / 2) → 
  c = 9/5 := by
sorry

end NUMINAMATH_CALUDE_quadratic_roots_condition_l3864_386454


namespace NUMINAMATH_CALUDE_tickets_left_l3864_386404

def tickets_bought : ℕ := 13
def ticket_cost : ℕ := 9
def spent_on_ferris_wheel : ℕ := 81

theorem tickets_left : tickets_bought - (spent_on_ferris_wheel / ticket_cost) = 4 := by
  sorry

end NUMINAMATH_CALUDE_tickets_left_l3864_386404


namespace NUMINAMATH_CALUDE_cubic_roots_sum_l3864_386439

theorem cubic_roots_sum (r s t : ℝ) : 
  r^3 - 15*r^2 + 13*r - 8 = 0 →
  s^3 - 15*s^2 + 13*s - 8 = 0 →
  t^3 - 15*t^2 + 13*t - 8 = 0 →
  (r / (1/r + s*t)) + (s / (1/s + t*r)) + (t / (1/t + r*s)) = 199/9 := by
sorry

end NUMINAMATH_CALUDE_cubic_roots_sum_l3864_386439


namespace NUMINAMATH_CALUDE_circle_intersection_distance_l3864_386409

-- Define the circle M
def CircleM (x y : ℝ) : Prop :=
  x^2 + y^2 - 8*x + 6*y = 0

-- Define the points on the circle
def PointO : ℝ × ℝ := (0, 0)
def PointA : ℝ × ℝ := (1, 1)
def PointB : ℝ × ℝ := (4, 2)

-- Define the intersection points
def PointS : ℝ × ℝ := (8, 0)
def PointT : ℝ × ℝ := (0, -6)

-- Theorem statement
theorem circle_intersection_distance :
  CircleM PointO.1 PointO.2 ∧
  CircleM PointA.1 PointA.2 ∧
  CircleM PointB.1 PointB.2 ∧
  CircleM PointS.1 PointS.2 ∧
  CircleM PointT.1 PointT.2 ∧
  PointS.1 > 0 ∧
  PointT.2 < 0 →
  Real.sqrt ((PointS.1 - PointT.1)^2 + (PointS.2 - PointT.2)^2) = 10 :=
by sorry

end NUMINAMATH_CALUDE_circle_intersection_distance_l3864_386409


namespace NUMINAMATH_CALUDE_tate_high_school_duration_l3864_386453

theorem tate_high_school_duration (normal_hs_duration : ℕ) (total_time : ℕ) (x : ℕ) : 
  normal_hs_duration = 4 →
  total_time = 12 →
  (normal_hs_duration - x) + 3 * (normal_hs_duration - x) = total_time →
  x = 4 := by
sorry

end NUMINAMATH_CALUDE_tate_high_school_duration_l3864_386453


namespace NUMINAMATH_CALUDE_inequality_proof_l3864_386438

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) 
  (hnz : a ≠ 0 ∨ b ≠ 0 ∨ c ≠ 0) : 
  Real.sqrt ((b + c) / (2 * a + b + c)) + 
  Real.sqrt ((c + a) / (2 * b + c + a)) + 
  Real.sqrt ((a + b) / (2 * c + a + b)) ≤ 1 + 2 / Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l3864_386438


namespace NUMINAMATH_CALUDE_total_books_is_182_l3864_386456

/-- The number of books each person has -/
def joan_books : ℕ := 10
def tom_books : ℕ := 38
def lisa_books : ℕ := 27
def steve_books : ℕ := 45
def kim_books : ℕ := 14
def alex_books : ℕ := 48

/-- The total number of books -/
def total_books : ℕ := joan_books + tom_books + lisa_books + steve_books + kim_books + alex_books

/-- Theorem stating that the total number of books is 182 -/
theorem total_books_is_182 : total_books = 182 := by
  sorry

end NUMINAMATH_CALUDE_total_books_is_182_l3864_386456


namespace NUMINAMATH_CALUDE_intersection_of_sets_l3864_386485

open Set

theorem intersection_of_sets :
  let A : Set ℝ := {x | x > 2}
  let B : Set ℝ := {x | (x - 1) * (x - 3) < 0}
  A ∩ B = {x | 2 < x ∧ x < 3} := by
  sorry

end NUMINAMATH_CALUDE_intersection_of_sets_l3864_386485


namespace NUMINAMATH_CALUDE_monster_feeding_interval_l3864_386469

/-- Represents the monster's feeding pattern over 300 years -/
structure MonsterFeedingPattern where
  interval : ℕ  -- The interval at which the monster rises
  total_consumed : ℕ  -- Total number of people consumed over 300 years
  first_ship : ℕ  -- Number of people on the first ship

/-- Theorem stating the conditions and the conclusion about the monster's feeding interval -/
theorem monster_feeding_interval (m : MonsterFeedingPattern) : 
  m.total_consumed = 847 ∧ 
  m.first_ship = 121 ∧ 
  m.total_consumed = m.first_ship + 2 * m.first_ship + 4 * m.first_ship → 
  m.interval = 100 := by
  sorry

end NUMINAMATH_CALUDE_monster_feeding_interval_l3864_386469


namespace NUMINAMATH_CALUDE_paintbrush_cost_l3864_386420

/-- The cost of each paintbrush given Marc's purchases -/
theorem paintbrush_cost (model_cars : ℕ) (car_cost : ℕ) (paint_bottles : ℕ) (paint_cost : ℕ) 
  (paintbrushes : ℕ) (total_spent : ℕ) : 
  model_cars = 5 → 
  car_cost = 20 → 
  paint_bottles = 5 → 
  paint_cost = 10 → 
  paintbrushes = 5 → 
  total_spent = 160 → 
  (total_spent - (model_cars * car_cost + paint_bottles * paint_cost)) / paintbrushes = 2 := by
  sorry

#check paintbrush_cost

end NUMINAMATH_CALUDE_paintbrush_cost_l3864_386420


namespace NUMINAMATH_CALUDE_range_of_a_l3864_386430

theorem range_of_a (a : ℝ) : Real.sqrt ((1 - 2*a)^2) = 2*a - 1 → a ≥ 1/2 := by
  sorry

end NUMINAMATH_CALUDE_range_of_a_l3864_386430


namespace NUMINAMATH_CALUDE_mean_temperature_l3864_386457

def temperatures : List ℝ := [82, 84, 86, 88, 90, 92, 84, 85]

theorem mean_temperature : 
  (temperatures.sum / temperatures.length : ℝ) = 86.375 := by
  sorry

end NUMINAMATH_CALUDE_mean_temperature_l3864_386457


namespace NUMINAMATH_CALUDE_arithmetic_sequence_a8_l3864_386481

def arithmetic_sequence (a : ℕ → ℤ) : Prop :=
  ∃ d : ℤ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_a8 (a : ℕ → ℤ) 
  (h_arith : arithmetic_sequence a) 
  (h_a2 : a 2 = 4) 
  (h_a4 : a 4 = 2) : 
  a 8 = -2 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_a8_l3864_386481


namespace NUMINAMATH_CALUDE_diagonal_FH_range_l3864_386444

/-- Represents a quadrilateral with integer side lengths and diagonals -/
structure Quadrilateral where
  EF : ℕ
  FG : ℕ
  GH : ℕ
  HE : ℕ
  EH : ℕ
  FH : ℕ

/-- The specific quadrilateral from the problem -/
def specificQuad : Quadrilateral where
  EF := 7
  FG := 13
  GH := 7
  HE := 20
  EH := 0  -- We don't know the exact value, but it's an integer
  FH := 0  -- This is what we're trying to prove

theorem diagonal_FH_range (q : Quadrilateral) (h : q = specificQuad) : 
  14 ≤ q.FH ∧ q.FH ≤ 19 := by
  sorry

#check diagonal_FH_range

end NUMINAMATH_CALUDE_diagonal_FH_range_l3864_386444


namespace NUMINAMATH_CALUDE_bicycle_cost_price_l3864_386433

/-- Proves that given a bicycle sold twice with profits of 20% and 25% respectively,
    and a final selling price of 225, the original cost price was 150. -/
theorem bicycle_cost_price 
  (profit_A : Real) 
  (profit_B : Real)
  (final_price : Real)
  (h1 : profit_A = 0.20)
  (h2 : profit_B = 0.25)
  (h3 : final_price = 225) :
  ∃ (initial_price : Real),
    initial_price * (1 + profit_A) * (1 + profit_B) = final_price ∧ 
    initial_price = 150 := by
  sorry

end NUMINAMATH_CALUDE_bicycle_cost_price_l3864_386433


namespace NUMINAMATH_CALUDE_right_triangles_bc_length_l3864_386455

/-- Given two right triangles ABC and ABD where B is vertically above A,
    and C and D lie on the horizontal axis, prove that if AC = 20, AD = 45,
    and BD = 13, then BC = 47. -/
theorem right_triangles_bc_length (A B C D : ℝ × ℝ) : 
  (∃ k : ℝ, B = (A.1, A.2 + k)) →  -- B is vertically above A
  (C.2 = A.2 ∧ D.2 = A.2) →        -- C and D lie on the horizontal axis
  (B.1 - A.1)^2 + (B.2 - A.2)^2 = (C.1 - A.1)^2 + (C.2 - A.2)^2 →  -- ABC is a right triangle
  (B.1 - A.1)^2 + (B.2 - A.2)^2 = (D.1 - A.1)^2 + (D.2 - A.2)^2 →  -- ABD is a right triangle
  Real.sqrt ((C.1 - A.1)^2 + (C.2 - A.2)^2) = 20 →  -- AC = 20
  Real.sqrt ((D.1 - A.1)^2 + (D.2 - A.2)^2) = 45 →  -- AD = 45
  Real.sqrt ((D.1 - B.1)^2 + (D.2 - B.2)^2) = 13 →  -- BD = 13
  Real.sqrt ((C.1 - B.1)^2 + (C.2 - B.2)^2) = 47    -- BC = 47
  := by sorry


end NUMINAMATH_CALUDE_right_triangles_bc_length_l3864_386455


namespace NUMINAMATH_CALUDE_first_nonzero_digit_after_decimal_of_1_157_l3864_386459

theorem first_nonzero_digit_after_decimal_of_1_157 : ∃ (n : ℕ) (d : ℕ), 
  0 < d ∧ d < 10 ∧ 
  (1000 : ℚ) / 157 = 6 + (d : ℚ) / 10 + (n : ℚ) / 100 ∧ 
  d = 3 :=
sorry

end NUMINAMATH_CALUDE_first_nonzero_digit_after_decimal_of_1_157_l3864_386459


namespace NUMINAMATH_CALUDE_senior_score_is_140_8_l3864_386413

/-- Represents the AHSME exam results at Century High School -/
structure AHSMEResults where
  total_students : ℕ
  average_score : ℝ
  senior_ratio : ℝ
  senior_score_ratio : ℝ

/-- Calculates the mean score of seniors given AHSME results -/
def senior_mean_score (results : AHSMEResults) : ℝ :=
  sorry

/-- Theorem stating that the mean score of seniors is 140.8 -/
theorem senior_score_is_140_8 (results : AHSMEResults)
  (h1 : results.total_students = 120)
  (h2 : results.average_score = 110)
  (h3 : results.senior_ratio = 1 / 1.4)
  (h4 : results.senior_score_ratio = 1.6) :
  senior_mean_score results = 140.8 := by
  sorry

end NUMINAMATH_CALUDE_senior_score_is_140_8_l3864_386413


namespace NUMINAMATH_CALUDE_chess_club_boys_count_l3864_386473

theorem chess_club_boys_count :
  ∀ (G B : ℕ),
  G + B = 30 →
  (2 * G) / 3 + (3 * B) / 4 = 18 →
  B = 24 :=
by
  sorry

end NUMINAMATH_CALUDE_chess_club_boys_count_l3864_386473


namespace NUMINAMATH_CALUDE_prob_one_boy_one_girl_prob_one_boy_one_girl_given_boy_prob_one_boy_one_girl_given_monday_boy_l3864_386471

/-- Represents the gender of a child -/
inductive Gender
| Boy
| Girl

/-- Represents the day of the week -/
inductive Day
| Monday
| OtherDay

/-- Represents a child with their gender and birth day -/
structure Child :=
  (gender : Gender)
  (birthDay : Day)

/-- Represents a family with two children -/
structure Family :=
  (child1 : Child)
  (child2 : Child)

/-- The probability of having a boy or a girl is equal -/
axiom equal_gender_probability : ℝ

/-- The probability of being born on a Monday -/
axiom monday_probability : ℝ

/-- Theorem for the probability of having one boy and one girl in a family with two children -/
theorem prob_one_boy_one_girl : ℝ := by sorry

/-- Theorem for the probability of having one boy and one girl, given that one child is a boy -/
theorem prob_one_boy_one_girl_given_boy : ℝ := by sorry

/-- Theorem for the probability of having one boy and one girl, given that one child is a boy born on a Monday -/
theorem prob_one_boy_one_girl_given_monday_boy : ℝ := by sorry

end NUMINAMATH_CALUDE_prob_one_boy_one_girl_prob_one_boy_one_girl_given_boy_prob_one_boy_one_girl_given_monday_boy_l3864_386471


namespace NUMINAMATH_CALUDE_area_of_median_triangle_l3864_386448

/-- Given a triangle ABC with area S, the area of a triangle whose sides are equal to the medians of ABC is 3/4 * S -/
theorem area_of_median_triangle (A B C : ℝ × ℝ) (S : ℝ) : 
  let triangle_area := S
  let median_triangle_area := (3/4 : ℝ) * S
  triangle_area = S → median_triangle_area = (3/4 : ℝ) * triangle_area := by
sorry

end NUMINAMATH_CALUDE_area_of_median_triangle_l3864_386448


namespace NUMINAMATH_CALUDE_tetrahedron_formation_condition_l3864_386416

/-- Represents a point in 2D space -/
structure Point2D where
  x : ℝ
  y : ℝ

/-- Represents a square with side length 2 -/
structure Square where
  A : Point2D
  B : Point2D
  C : Point2D
  D : Point2D

/-- Condition for forming a tetrahedron by folding triangles in a square -/
def canFormTetrahedron (s : Square) (x : ℝ) : Prop :=
  let E : Point2D := { x := (s.A.x + s.B.x) / 2, y := (s.A.y + s.B.y) / 2 }
  let F : Point2D := { x := s.B.x + x, y := s.B.y }
  let EA' := 1
  let EF := Real.sqrt (1 + x^2)
  let FA' := 2 - x
  EA' + EF > FA' ∧ EF + FA' > EA' ∧ FA' + EA' > EF

theorem tetrahedron_formation_condition (s : Square) :
  (∀ x, canFormTetrahedron s x ↔ 0 < x ∧ x < 4/3) :=
by sorry

end NUMINAMATH_CALUDE_tetrahedron_formation_condition_l3864_386416


namespace NUMINAMATH_CALUDE_expression_value_l3864_386445

theorem expression_value : (2^2 * 5) / (8 * 10) * (3 * 4 * 8) / (2 * 5 * 3) = 0.8 := by
  sorry

end NUMINAMATH_CALUDE_expression_value_l3864_386445


namespace NUMINAMATH_CALUDE_coat_price_calculation_shopper_pays_112_75_l3864_386415

/-- Calculate the final price of a coat after discounts and tax -/
theorem coat_price_calculation (original_price : ℝ) (initial_discount_percent : ℝ) 
  (additional_discount : ℝ) (sales_tax_percent : ℝ) : ℝ :=
  let price_after_initial_discount := original_price * (1 - initial_discount_percent / 100)
  let price_after_additional_discount := price_after_initial_discount - additional_discount
  let final_price := price_after_additional_discount * (1 + sales_tax_percent / 100)
  final_price

/-- Proof that the shopper pays $112.75 for the coat -/
theorem shopper_pays_112_75 :
  coat_price_calculation 150 25 10 10 = 112.75 := by
  sorry

end NUMINAMATH_CALUDE_coat_price_calculation_shopper_pays_112_75_l3864_386415


namespace NUMINAMATH_CALUDE_quadratic_equation_roots_l3864_386466

theorem quadratic_equation_roots : ∃ x : ℝ, (∀ y : ℝ, -y^2 + 2*y - 1 = 0 ↔ y = x) :=
sorry

end NUMINAMATH_CALUDE_quadratic_equation_roots_l3864_386466


namespace NUMINAMATH_CALUDE_product_of_powers_l3864_386405

theorem product_of_powers (x y : ℝ) : -x^2 * y^3 * (2 * x * y^2) = -2 * x^3 * y^5 := by
  sorry

end NUMINAMATH_CALUDE_product_of_powers_l3864_386405


namespace NUMINAMATH_CALUDE_min_value_reciprocal_sum_l3864_386467

/-- Given a line 2ax + by - 2 = 0 where a > 0 and b > 0, and the line passes through the point (1, 2),
    the minimum value of 1/a + 1/b is 4 -/
theorem min_value_reciprocal_sum (a b : ℝ) : 
  a > 0 → b > 0 → 2*a + b = 2 → (∀ x y : ℝ, x > 0 ∧ y > 0 ∧ 2*x + y = 2 → 1/a + 1/b ≤ 1/x + 1/y) → 
  1/a + 1/b = 4 := by sorry

end NUMINAMATH_CALUDE_min_value_reciprocal_sum_l3864_386467


namespace NUMINAMATH_CALUDE_average_weight_b_c_l3864_386421

/-- Given the weights of three people a, b, and c, prove that the average weight of b and c is 42 kg -/
theorem average_weight_b_c (a b c : ℝ) : 
  (a + b + c) / 3 = 43 →  -- The average weight of a, b, and c is 43 kg
  (a + b) / 2 = 48 →      -- The average weight of a and b is 48 kg
  b = 51 →                -- The weight of b is 51 kg
  (b + c) / 2 = 42 :=     -- The average weight of b and c is 42 kg
by
  sorry

end NUMINAMATH_CALUDE_average_weight_b_c_l3864_386421


namespace NUMINAMATH_CALUDE_rectangular_room_tiles_l3864_386478

/-- Calculates the number of tiles touching the walls in a rectangular room -/
def tiles_touching_walls (length width : ℕ) : ℕ :=
  2 * length + 2 * width - 4

theorem rectangular_room_tiles (length width : ℕ) 
  (h_length : length = 10) (h_width : width = 5) : 
  tiles_touching_walls length width = 26 := by
  sorry

#eval tiles_touching_walls 10 5

end NUMINAMATH_CALUDE_rectangular_room_tiles_l3864_386478


namespace NUMINAMATH_CALUDE_initial_investment_rate_l3864_386484

/-- Proves that the initial investment rate is 5% given the problem conditions --/
theorem initial_investment_rate
  (initial_investment : ℝ)
  (additional_investment : ℝ)
  (additional_rate : ℝ)
  (total_rate : ℝ)
  (h1 : initial_investment = 8000)
  (h2 : additional_investment = 4000)
  (h3 : additional_rate = 8)
  (h4 : total_rate = 6)
  (h5 : initial_investment + additional_investment = 12000) :
  ∃ R : ℝ, R = 5 ∧
    (initial_investment * R / 100 + additional_investment * additional_rate / 100 =
     (initial_investment + additional_investment) * total_rate / 100) :=
by
  sorry

end NUMINAMATH_CALUDE_initial_investment_rate_l3864_386484


namespace NUMINAMATH_CALUDE_sequence_a_general_term_l3864_386451

/-- Sequence a_n with sum S_n satisfying the given conditions -/
def sequence_a (n : ℕ) : ℚ := sorry

/-- Sum of first n terms of sequence a_n -/
def S (n : ℕ) : ℚ := sorry

/-- The main theorem to prove -/
theorem sequence_a_general_term :
  ∀ n : ℕ, n > 0 →
  (2 * S n - n * sequence_a n = n) ∧
  (sequence_a 2 = 3) →
  sequence_a n = 2 * n - 1 := by sorry

end NUMINAMATH_CALUDE_sequence_a_general_term_l3864_386451


namespace NUMINAMATH_CALUDE_vp_factorial_and_binomial_l3864_386464

/-- The p-adic valuation of a natural number -/
noncomputable def v_p (p : ℕ) (n : ℕ) : ℕ := sorry

/-- The sum of floor of N divided by increasing powers of p -/
def sum_floor (N : ℕ) (p : ℕ) : ℕ := sorry

theorem vp_factorial_and_binomial 
  (N k : ℕ) (p : ℕ) (h_prime : Nat.Prime p) (h_pow : ∃ n, N = p ^ n) (h_ge : N ≥ k) :
  (v_p p (N.factorial) = sum_floor N p) ∧ 
  (v_p p (Nat.choose N k) = v_p p N - v_p p k) := by
  sorry

end NUMINAMATH_CALUDE_vp_factorial_and_binomial_l3864_386464


namespace NUMINAMATH_CALUDE_expression_equality_l3864_386472

theorem expression_equality (x : ℝ) (h1 : x^3 + 1 ≠ 0) (h2 : x^3 - 1 ≠ 0) : 
  ((x + 1)^3 * (x^2 - x + 1)^3 / (x^3 + 1)^3)^2 * 
  ((x - 1)^3 * (x^2 + x + 1)^3 / (x^3 - 1)^3)^2 = 1 := by
  sorry

end NUMINAMATH_CALUDE_expression_equality_l3864_386472


namespace NUMINAMATH_CALUDE_class_average_theorem_l3864_386407

theorem class_average_theorem (total_students : ℕ) 
                               (excluded_students : ℕ) 
                               (excluded_average : ℝ) 
                               (remaining_average : ℝ) : 
  total_students = 56 →
  excluded_students = 8 →
  excluded_average = 20 →
  remaining_average = 90 →
  (total_students * (total_students * remaining_average - excluded_students * remaining_average + excluded_students * excluded_average)) / 
  (total_students * total_students) = 80 := by
sorry

end NUMINAMATH_CALUDE_class_average_theorem_l3864_386407


namespace NUMINAMATH_CALUDE_non_equilateral_combinations_count_l3864_386418

/-- The number of dots evenly spaced on the circumference of a circle -/
def num_dots : ℕ := 6

/-- A function that calculates the number of combinations that do not form an equilateral triangle -/
def non_equilateral_combinations (n : ℕ) : ℕ :=
  if n = 1 then num_dots
  else if n = 2 then num_dots.choose 2
  else if n = 3 then num_dots.choose 3 - 2
  else 0

/-- The total number of combinations that do not form an equilateral triangle -/
def total_combinations : ℕ :=
  (non_equilateral_combinations 1) + (non_equilateral_combinations 2) + (non_equilateral_combinations 3)

theorem non_equilateral_combinations_count :
  total_combinations = 18 :=
by sorry

end NUMINAMATH_CALUDE_non_equilateral_combinations_count_l3864_386418


namespace NUMINAMATH_CALUDE_negation_equivalence_l3864_386422

theorem negation_equivalence :
  (¬ ∃ x₀ : ℝ, x₀ > 1 ∧ x₀^2 - x₀ + 2016 > 0) ↔
  (¬ ∃ x : ℝ, x > 1 ∧ x^2 - x + 2016 ≤ 0) := by
sorry

end NUMINAMATH_CALUDE_negation_equivalence_l3864_386422


namespace NUMINAMATH_CALUDE_tshirts_per_package_l3864_386434

theorem tshirts_per_package (total_packages : ℕ) (total_tshirts : ℕ) 
  (h1 : total_packages = 71) 
  (h2 : total_tshirts = 426) : 
  total_tshirts / total_packages = 6 := by
  sorry

end NUMINAMATH_CALUDE_tshirts_per_package_l3864_386434


namespace NUMINAMATH_CALUDE_roots_inequality_l3864_386425

theorem roots_inequality (m : ℝ) (x₁ x₂ : ℝ) (hm : m < -2) 
  (hx : x₁ < x₂) (hf₁ : Real.log x₁ - x₁ = m) (hf₂ : Real.log x₂ - x₂ = m) :
  x₁ * x₂^2 < 2 := by
  sorry

end NUMINAMATH_CALUDE_roots_inequality_l3864_386425


namespace NUMINAMATH_CALUDE_function_f_negative_two_l3864_386429

/-- A function satisfying the given properties -/
def FunctionF (f : ℝ → ℝ) : Prop :=
  (∀ a b : ℝ, f (a + b) = f a * f b) ∧
  (∀ x : ℝ, f x > 0) ∧
  (f 1 = 1/3)

/-- The main theorem -/
theorem function_f_negative_two (f : ℝ → ℝ) (h : FunctionF f) : f (-2) = 9 := by
  sorry

end NUMINAMATH_CALUDE_function_f_negative_two_l3864_386429


namespace NUMINAMATH_CALUDE_root_product_negative_l3864_386435

-- Define a monotonic function
def Monotonic (f : ℝ → ℝ) : Prop :=
  ∀ x y, x ≤ y → f x ≤ f y

-- State the theorem
theorem root_product_negative
  (f : ℝ → ℝ) (a x₁ x₂ : ℝ)
  (h_monotonic : Monotonic f)
  (h_root : f a = 0)
  (h_order : x₁ < a ∧ a < x₂) :
  f x₁ * f x₂ < 0 :=
sorry

end NUMINAMATH_CALUDE_root_product_negative_l3864_386435


namespace NUMINAMATH_CALUDE_square_tiles_count_l3864_386490

/-- Represents the number of edges for each type of tile -/
def edges_per_tile : Fin 3 → ℕ
| 0 => 3  -- triangular
| 1 => 4  -- square
| 2 => 5  -- pentagonal
| _ => 0  -- unreachable

/-- Proves that given 30 tiles with 108 edges in total, there are 6 square tiles -/
theorem square_tiles_count 
  (total_tiles : ℕ) 
  (total_edges : ℕ) 
  (h_total_tiles : total_tiles = 30) 
  (h_total_edges : total_edges = 108) :
  ∃ (t s p : ℕ), 
    t + s + p = total_tiles ∧ 
    3 * t + 4 * s + 5 * p = total_edges ∧ 
    s = 6 :=
by
  sorry

#check square_tiles_count

end NUMINAMATH_CALUDE_square_tiles_count_l3864_386490


namespace NUMINAMATH_CALUDE_right_triangle_area_l3864_386411

/-- The area of a right triangle with hypotenuse 14 inches and one 45-degree angle is 49 square inches. -/
theorem right_triangle_area (hypotenuse : ℝ) (angle : ℝ) :
  hypotenuse = 14 →
  angle = 45 * (π / 180) →
  let leg := hypotenuse / Real.sqrt 2
  let area := (1 / 2) * leg * leg
  area = 49 := by
sorry

end NUMINAMATH_CALUDE_right_triangle_area_l3864_386411


namespace NUMINAMATH_CALUDE_gcf_of_lcms_equals_15_l3864_386446

theorem gcf_of_lcms_equals_15 : Nat.gcd (Nat.lcm 9 15) (Nat.lcm 10 21) = 15 := by
  sorry

end NUMINAMATH_CALUDE_gcf_of_lcms_equals_15_l3864_386446


namespace NUMINAMATH_CALUDE_project_completion_time_l3864_386475

theorem project_completion_time (a b c : ℝ) 
  (h1 : a + b = 1/2)   -- A and B together complete in 2 days
  (h2 : b + c = 1/4)   -- B and C together complete in 4 days
  (h3 : c + a = 1/2.4) -- C and A together complete in 2.4 days
  : 1/a = 3 :=         -- A alone completes in 3 days
by
  sorry

end NUMINAMATH_CALUDE_project_completion_time_l3864_386475


namespace NUMINAMATH_CALUDE_cone_properties_l3864_386496

/-- Properties of a cone with specific dimensions -/
theorem cone_properties (r h l : ℝ) : 
  r = 2 → -- base radius is 2
  π * l = 2 * π * r → -- lateral surface unfolds to a semicircle
  l^2 = r^2 + h^2 → -- Pythagorean theorem
  (l = 4 ∧ (1/3) * π * r^2 * h = (8 * Real.sqrt 3 / 3) * π) := by sorry

end NUMINAMATH_CALUDE_cone_properties_l3864_386496


namespace NUMINAMATH_CALUDE_negative_integer_solution_l3864_386474

theorem negative_integer_solution : ∃! (N : ℤ), N < 0 ∧ N + 2 * N^2 = 12 ∧ N = -3 := by
  sorry

end NUMINAMATH_CALUDE_negative_integer_solution_l3864_386474


namespace NUMINAMATH_CALUDE_angle_between_given_lines_l3864_386432

def line1 (x y : ℝ) : Prop := 2 * x - y + 1 = 0
def line2 (x y : ℝ) : Prop := x - y - 2 = 0

def angle_between_lines (l1 l2 : (ℝ → ℝ → Prop)) : ℝ := sorry

theorem angle_between_given_lines :
  angle_between_lines line1 line2 = Real.arctan (1/3) := by sorry

end NUMINAMATH_CALUDE_angle_between_given_lines_l3864_386432


namespace NUMINAMATH_CALUDE_table_tennis_players_l3864_386428

theorem table_tennis_players (singles_tables doubles_tables : ℕ) : 
  singles_tables + doubles_tables = 13 → 
  4 * doubles_tables = 2 * singles_tables + 4 → 
  4 * doubles_tables = 20 := by
  sorry

end NUMINAMATH_CALUDE_table_tennis_players_l3864_386428


namespace NUMINAMATH_CALUDE_collinear_vectors_x_equals_three_l3864_386470

/-- Two vectors are collinear if one is a scalar multiple of the other -/
def collinear (a b : ℝ × ℝ) : Prop :=
  ∃ k : ℝ, a = (k * b.1, k * b.2)

/-- Given vectors a and b, prove that if they are collinear, then x = 3 -/
theorem collinear_vectors_x_equals_three (x : ℝ) :
  let a : ℝ × ℝ := (2, 4)
  let b : ℝ × ℝ := (x, 6)
  collinear a b → x = 3 := by
  sorry


end NUMINAMATH_CALUDE_collinear_vectors_x_equals_three_l3864_386470


namespace NUMINAMATH_CALUDE_minimum_square_side_l3864_386488

theorem minimum_square_side (area_min : ℝ) (side : ℝ) : 
  area_min = 625 → side^2 ≥ area_min → side ≥ 0 → side ≥ 25 :=
by sorry

end NUMINAMATH_CALUDE_minimum_square_side_l3864_386488
