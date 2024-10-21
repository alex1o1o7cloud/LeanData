import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_nine_smallest_even_multiples_of_7_l503_50365

/-- The sum of the nine smallest distinct even positive integer multiples of 7 -/
def sumOfNineSmallestEvenMultiplesOf7 : ℕ :=
  let multiples := List.range 9 |>.map (fun i => 14 * (i + 1))
  multiples.sum

/-- Theorem stating that the sum of the nine smallest distinct even positive integer multiples of 7 is 630 -/
theorem sum_of_nine_smallest_even_multiples_of_7 :
  sumOfNineSmallestEvenMultiplesOf7 = 630 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_nine_smallest_even_multiples_of_7_l503_50365


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_triangle_area_max_area_line_equation_min_quadrilateral_area_l503_50334

-- Define the ellipse
def ellipse (x y : ℝ) : Prop := x^2/2 + y^2 = 1

-- Define a line
structure Line where
  slope : ℝ
  intercept : ℝ

-- Define the set of lines that give maximum area
def max_area_lines : Set Line := 
  {l : Line | ∃ k, l.intercept = Real.sqrt (k^2 + 1/2) ∨ l.intercept = -Real.sqrt (k^2 + 1/2) ∧ l.slope = k}

-- Define the area of triangle AOB given a line
noncomputable def triangle_area (l : Line) : ℝ := 
  Real.sqrt 2 * Real.sqrt ((1 + 2*l.slope^2 - l.intercept^2) * l.intercept^2) / (1 + 2*l.slope^2)

-- Theorem for maximum area of triangle AOB
theorem max_triangle_area :
  ∀ l : Line, triangle_area l ≤ Real.sqrt 2 / 2 := by sorry

-- Theorem for equation of line achieving maximum area
theorem max_area_line_equation (l : Line) :
  l ∈ max_area_lines ↔ ∃ k, (l.intercept = Real.sqrt (k^2 + 1/2) ∨ l.intercept = -Real.sqrt (k^2 + 1/2)) ∧ l.slope = k := by sorry

-- Theorem for minimum area of quadrilateral
theorem min_quadrilateral_area (l₁ l₂ l₃ l₄ : Line) 
  (h₁ : l₁ ∈ max_area_lines) (h₂ : l₂ ∈ max_area_lines) 
  (h₃ : l₃ ∈ max_area_lines) (h₄ : l₄ ∈ max_area_lines)
  (h_parallel₁ : l₁.slope = l₂.slope) (h_parallel₂ : l₃.slope = l₄.slope)
  (h_sum : l₁.slope + l₂.slope + l₃.slope + l₄.slope = 0) :
  2 * Real.sqrt 2 ≤ 2 * (l₁.slope + 1 / (2 * l₁.slope)) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_triangle_area_max_area_line_equation_min_quadrilateral_area_l503_50334


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_not_always_possible_to_swap_l503_50319

/-- Represents a city in the kingdom -/
structure City where
  name : String

/-- Represents the railroad network in the kingdom -/
structure Kingdom where
  cities : Set City
  connected : City → City → Prop

/-- An automorphism of the kingdom's railroad network -/
structure KingdomAutomorphism (k : Kingdom) where
  f : City → City
  bijective : Function.Bijective f
  preserves_connections : ∀ c₁ c₂, k.connected c₁ c₂ ↔ k.connected (f c₁) (f c₂)

/-- For any two cities, there exists an automorphism mapping one to the other -/
axiom exists_automorphism_mapping (k : Kingdom) (c₁ c₂ : City) : 
  c₁ ∈ k.cities → c₂ ∈ k.cities → ∃ φ : KingdomAutomorphism k, φ.f c₁ = c₂

/-- The main theorem: It's not always possible to swap any two cities -/
theorem not_always_possible_to_swap (k : Kingdom) : 
  ¬(∀ c₁ c₂ : City, c₁ ∈ k.cities → c₂ ∈ k.cities → 
    ∃ φ : KingdomAutomorphism k, φ.f c₁ = c₂ ∧ φ.f c₂ = c₁) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_not_always_possible_to_swap_l503_50319


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_log_sum_upper_bound_l503_50378

theorem log_sum_upper_bound (x y : ℝ) (h1 : x ≥ y) (h2 : y > 2) :
  (Real.log x / Real.log y + Real.log y / Real.log x) ≤ 0 ∧
  ∃ x y : ℝ, x ≥ y ∧ y > 2 ∧ Real.log x / Real.log y + Real.log y / Real.log x = 0 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_log_sum_upper_bound_l503_50378


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_decimal_fraction_equality_l503_50385

theorem decimal_fraction_equality : 
  (14 : ℚ) / 99 + (218 : ℚ) / 999 = (1 : ℚ) / 3 * ((11 : ℚ) / 111) := by
  sorry

#eval (14 : ℚ) / 99 + (218 : ℚ) / 999
#eval (1 : ℚ) / 3 * ((11 : ℚ) / 111)

end NUMINAMATH_CALUDE_ERRORFEEDBACK_decimal_fraction_equality_l503_50385


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_net_profit_is_63000_l503_50381

/-- Represents the agricultural scenario with eggplants and tomatoes -/
structure FarmScenario where
  total_land : ℚ
  total_expenditure : ℚ
  eggplant_cost : ℚ
  eggplant_profit : ℚ
  tomato_cost : ℚ
  tomato_profit : ℚ

/-- Calculates the total net profit for a given farm scenario -/
noncomputable def calculate_net_profit (scenario : FarmScenario) : ℚ :=
  let eggplant_acres := (scenario.total_expenditure - scenario.tomato_cost * scenario.total_land) / (scenario.eggplant_cost - scenario.tomato_cost)
  let tomato_acres := scenario.total_land - eggplant_acres
  eggplant_acres * scenario.eggplant_profit + tomato_acres * scenario.tomato_profit

/-- Theorem stating that the net profit for the given scenario is 63000 -/
theorem net_profit_is_63000 (scenario : FarmScenario) 
  (h1 : scenario.total_land = 25)
  (h2 : scenario.total_expenditure = 44000)
  (h3 : scenario.eggplant_cost = 1700)
  (h4 : scenario.eggplant_profit = 2400)
  (h5 : scenario.tomato_cost = 1800)
  (h6 : scenario.tomato_profit = 2600) :
  calculate_net_profit scenario = 63000 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_net_profit_is_63000_l503_50381


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l503_50354

noncomputable def f (x : ℝ) := Real.sin x + Real.cos x

theorem f_properties :
  (∀ x : ℝ, f (π/4 + x) = f (π/4 - x)) ∧
  (∃ k : ℝ, ∀ x : ℝ, f x = Real.sqrt 2 * Real.cos (x - k) ∧ k = π/4) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l503_50354


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_no_magic_square_with_given_condition_l503_50345

/-- Represents a 4x4 grid --/
def Grid := Fin 4 → Fin 4 → ℕ

/-- Checks if a number is in the grid --/
def contains (g : Grid) (n : ℕ) : Prop :=
  ∃ i j, g i j = n

/-- Checks if all numbers from 1 to 16 are in the grid --/
def contains_all (g : Grid) : Prop :=
  ∀ n, 1 ≤ n ∧ n ≤ 16 → contains g n

/-- Calculates the sum of a row --/
def row_sum (g : Grid) (i : Fin 4) : ℕ :=
  (Finset.sum (Finset.range 4) fun j => g i j)

/-- Calculates the sum of a column --/
def col_sum (g : Grid) (j : Fin 4) : ℕ :=
  (Finset.sum (Finset.range 4) fun i => g i j)

/-- Calculates the sum of the main diagonal --/
def main_diag_sum (g : Grid) : ℕ :=
  (Finset.sum (Finset.range 4) fun i => g i i)

/-- Calculates the sum of the anti-diagonal --/
def anti_diag_sum (g : Grid) : ℕ :=
  (Finset.sum (Finset.range 4) fun i => g i (3 - i))

/-- Checks if the grid is a magic square --/
def is_magic_square (g : Grid) : Prop :=
  contains_all g ∧
  (∃ s, (∀ i, row_sum g i = s) ∧
        (∀ j, col_sum g j = s) ∧
        main_diag_sum g = s ∧
        anti_diag_sum g = s)

/-- Checks if two cells are adjacent --/
def are_adjacent (i1 j1 i2 j2 : Fin 4) : Prop :=
  (i1 = i2 ∧ (j1.val + 1 = j2.val ∨ j2.val + 1 = j1.val)) ∨
  (j1 = j2 ∧ (i1.val + 1 = i2.val ∨ i2.val + 1 = i1.val))

/-- The main theorem --/
theorem no_magic_square_with_given_condition :
  ¬∃ (g : Grid) (i1 j1 i2 j2 i3 j3 : Fin 4),
    is_magic_square g ∧
    g i1 j1 = 1 ∧
    g i2 j2 = 2 ∧
    g i3 j3 = 3 ∧
    are_adjacent i1 j1 i2 j2 ∧
    are_adjacent i1 j1 i3 j3 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_no_magic_square_with_given_condition_l503_50345


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_between_symmetry_axes_l503_50301

-- Define the function f(x)
noncomputable def f (x : ℝ) : ℝ := Real.cos (3 * x) ^ 2 - 1/2

-- Define the distance between adjacent symmetry axes
noncomputable def symmetry_axis_distance : ℝ := Real.pi / 6

-- Theorem statement
theorem distance_between_symmetry_axes :
  ∀ x : ℝ, f (x + symmetry_axis_distance) = f x :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_between_symmetry_axes_l503_50301


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sixth_root_18_gt_golden_ratio_l503_50361

/-- The golden ratio -/
noncomputable def φ : ℝ := (1 + Real.sqrt 5) / 2

/-- Theorem stating that the sixth root of 18 is greater than the golden ratio -/
theorem sixth_root_18_gt_golden_ratio : (18 : ℝ) ^ (1/6) > φ := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sixth_root_18_gt_golden_ratio_l503_50361


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_finite_nondividing_primes_l503_50304

/-- A set S of integers satisfying the given condition -/
def ClosedSet (S : Set ℤ) : Prop :=
  S.Nonempty ∧ ∀ a b, a ∈ S → b ∈ S → (a * b + 1) ∈ S

/-- The set of primes that do not divide any element of S -/
def NondividingPrimes (S : Set ℤ) : Set ℕ :=
  {p : ℕ | Nat.Prime p ∧ ∀ s ∈ S, ¬(p : ℤ) ∣ s}

/-- Main theorem: The set of primes not dividing any element of S is finite -/
theorem finite_nondividing_primes (S : Set ℤ) (hS : ClosedSet S) :
  (NondividingPrimes S).Finite := by
  sorry

#check finite_nondividing_primes

end NUMINAMATH_CALUDE_ERRORFEEDBACK_finite_nondividing_primes_l503_50304


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_river_speed_calculation_l503_50357

/-- The speed of a man rowing in still water (in km/h) -/
noncomputable def still_water_speed : ℝ := 7.5

/-- The time taken to row to a place and back (in hours) -/
noncomputable def total_time : ℝ := 50 / 60

/-- The total distance rowed (in km) -/
noncomputable def total_distance : ℝ := 6

/-- The speed of the river current (in km/h) -/
noncomputable def river_speed : ℝ := 1.5

/-- Theorem stating that the calculated river speed satisfies the given conditions -/
theorem river_speed_calculation : 
  total_distance / (still_water_speed - river_speed) + 
  total_distance / (still_water_speed + river_speed) = total_time := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_river_speed_calculation_l503_50357


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_fraction_value_l503_50341

/-- The original function g(x) -/
noncomputable def g (x : ℝ) : ℝ := (3*x - 2) / (x - 4)

/-- The inverse function g⁻¹(x) -/
noncomputable def g_inv (x : ℝ) : ℝ := (4*x - 2) / (x - 3)

/-- Theorem stating that a/c = 4 for the inverse function of g(x) -/
theorem inverse_fraction_value (a b c d : ℝ) :
  (∀ x, x ≠ 4 → g_inv (g x) = x) →
  (∀ x, x ≠ 3 → g (g_inv x) = x) →
  (∀ x, x ≠ 3 → g_inv x = (a*x + b) / (c*x + d)) →
  a / c = 4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_fraction_value_l503_50341


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_first_day_of_month_seventeenth_saturday_implies_first_thursday_l503_50330

/-- Represents the days of the week -/
inductive DayOfWeek
  | Sunday
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday
  | Saturday

/-- Returns the day of the week that is n days after the given day -/
def dayAfter (d : DayOfWeek) (n : Nat) : DayOfWeek :=
  match n with
  | 0 => d
  | n + 1 => dayAfter (match d with
    | DayOfWeek.Sunday => DayOfWeek.Monday
    | DayOfWeek.Monday => DayOfWeek.Tuesday
    | DayOfWeek.Tuesday => DayOfWeek.Wednesday
    | DayOfWeek.Wednesday => DayOfWeek.Thursday
    | DayOfWeek.Thursday => DayOfWeek.Friday
    | DayOfWeek.Friday => DayOfWeek.Saturday
    | DayOfWeek.Saturday => DayOfWeek.Sunday) n

/-- Returns the day of the week that is n days before the given day -/
def dayBefore (d : DayOfWeek) (n : Nat) : DayOfWeek :=
  dayAfter d (7 - (n % 7))

theorem first_day_of_month (d : DayOfWeek) :
  dayBefore d 16 = DayOfWeek.Thursday → d = DayOfWeek.Saturday := by
  sorry

/-- If the 17th day of a month is Saturday, then the first day of that month is Thursday -/
theorem seventeenth_saturday_implies_first_thursday :
  dayBefore DayOfWeek.Saturday 16 = DayOfWeek.Thursday := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_first_day_of_month_seventeenth_saturday_implies_first_thursday_l503_50330


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trajectory_equation_l503_50386

/-- The distance between two points (x₁, y₁) and (x₂, y₂) in R² -/
noncomputable def distance_point_point (x₁ y₁ x₂ y₂ : ℝ) : ℝ :=
  Real.sqrt ((x₁ - x₂)^2 + (y₁ - y₂)^2)

/-- The distance between a point (x, y) and a line Ax + By + C = 0 -/
noncomputable def distance_point_line (x y A B C : ℝ) : ℝ :=
  abs (A * x + B * y + C) / Real.sqrt (A^2 + B^2)

/-- The theorem stating the equation of the trajectory of point P -/
theorem trajectory_equation (x y : ℝ) :
  distance_point_point x y 1 1 = distance_point_line x y 3 1 (-4) →
  x - 3 * y + 2 = 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trajectory_equation_l503_50386


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_grade_is_four_l503_50370

/-- Represents the grading system for an economics course -/
structure EconomicsGrade where
  /-- Time spent studying microeconomics -/
  micro_time : ℝ
  /-- Time spent studying macroeconomics -/
  macro_time : ℝ
  /-- Total available study time -/
  total_time : ℝ
  /-- Points earned per unit time studying microeconomics -/
  micro_points_per_unit : ℝ
  /-- Points earned per unit time studying macroeconomics -/
  macro_points_per_unit : ℝ
  /-- Constraint: total study time is the sum of micro and macro time -/
  time_constraint : micro_time + macro_time = total_time

/-- Calculates the final grade based on the given grading system -/
noncomputable def finalGrade (g : EconomicsGrade) : ℝ :=
  let O_mic := g.micro_time * g.micro_points_per_unit
  let O_mac := g.macro_time * g.macro_points_per_unit
  min (0.25 * O_mic + 0.75 * O_mac) (0.75 * O_mic + 0.25 * O_mac)

/-- Theorem stating that the maximum achievable grade is 4 -/
theorem max_grade_is_four (g : EconomicsGrade) 
    (h1 : g.total_time = 4.6)
    (h2 : g.micro_points_per_unit = 2.5)
    (h3 : g.macro_points_per_unit = 1.5) : 
  ∃ (g' : EconomicsGrade), finalGrade g' ≤ 4 ∧ 
    ∀ (g'' : EconomicsGrade), finalGrade g'' ≤ finalGrade g' := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_grade_is_four_l503_50370


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_speed_clicks_time_not_standard_intervals_l503_50377

/-- The length of a rail in feet -/
noncomputable def rail_length : ℝ := 40

/-- Converts miles per hour to feet per minute -/
noncomputable def mph_to_fpm (x : ℝ) : ℝ := x * 5280 / 60

/-- Calculates the number of clicks per minute for a given speed in miles per hour -/
noncomputable def clicks_per_minute (speed : ℝ) : ℝ := mph_to_fpm speed / rail_length

/-- Calculates the time in minutes for the number of clicks to equal the speed -/
noncomputable def time_for_clicks_equal_speed (speed : ℝ) : ℝ := speed / (clicks_per_minute speed)

/-- Converts minutes to seconds -/
noncomputable def minutes_to_seconds (t : ℝ) : ℝ := t * 60

theorem train_speed_clicks_time_not_standard_intervals :
  ∀ (speed : ℝ), speed > 0 →
    (let t := minutes_to_seconds (time_for_clicks_equal_speed speed)
     t ≠ 15 ∧ t ≠ 60 ∧ t ≠ 120 ∧ t ≠ 180) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_speed_clicks_time_not_standard_intervals_l503_50377


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sledding_time_difference_l503_50392

/-- Calculates the time difference between Ann's and Mary's sledding trips -/
theorem sledding_time_difference 
  (mary_hill_length : ℝ) 
  (mary_speed : ℝ) 
  (ann_hill_length : ℝ) 
  (ann_speed : ℝ) 
  (h1 : mary_hill_length = 630)
  (h2 : mary_speed = 90)
  (h3 : ann_hill_length = 800)
  (h4 : ann_speed = 40) :
  ann_hill_length / ann_speed - mary_hill_length / mary_speed = 13 := by
  sorry

#check sledding_time_difference

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sledding_time_difference_l503_50392


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_total_amount_after_two_years_theorem_l503_50375

/-- Calculates the total amount after compound interest for 2 years -/
noncomputable def totalAmountAfterTwoYears (initialDeposit : ℝ) (interestRate : ℝ) : ℝ :=
  initialDeposit * (1 + interestRate / 100) ^ 2

/-- Theorem: The total amount after 2 years for a 5000 yuan deposit at x% annual interest rate is 5000(1+x%)^2 -/
theorem total_amount_after_two_years_theorem (x : ℝ) :
  totalAmountAfterTwoYears 5000 x = 5000 * (1 + x / 100) ^ 2 := by
  sorry

#check total_amount_after_two_years_theorem

end NUMINAMATH_CALUDE_ERRORFEEDBACK_total_amount_after_two_years_theorem_l503_50375


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_remainder_one_mod_three_set_l503_50335

theorem remainder_one_mod_three_set : 
  {x : ℕ | x > 0 ∧ ∃ k : ℕ, x = 3 * k + 1} = {x : ℕ | x > 0 ∧ x % 3 = 1} := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_remainder_one_mod_three_set_l503_50335


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trig_identity_l503_50363

theorem trig_identity (a : ℝ) (h1 : Real.sin a + Real.cos a = 1/2) (h2 : 0 < a ∧ a < Real.pi) :
  (1 - Real.tan a) / (1 + Real.tan a) = -Real.sqrt 7 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trig_identity_l503_50363


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_f_l503_50389

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := (2 * x + 3) / (x - 1)

-- Define the domain
def domain : Set ℝ := { x | x ≥ 2 }

-- State the theorem
theorem range_of_f :
  { y | ∃ x ∈ domain, f x = y } = Set.Ioo 2 7 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_f_l503_50389


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_final_retail_price_correct_l503_50324

/-- Given a cost price, initial markup percentage, and adjustment percentage,
    calculate the final retail price of a shirt. -/
noncomputable def final_retail_price (m : ℝ) (a : ℝ) (b : ℝ) : ℝ :=
  m * (1 + a / 100) * (b / 100)

/-- Theorem stating that the final retail price formula is correct -/
theorem final_retail_price_correct (m : ℝ) (a : ℝ) (b : ℝ) :
  final_retail_price m a b = m * (1 + a / 100) * (b / 100) := by
  rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_final_retail_price_correct_l503_50324


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_distinct_lines_l503_50347

-- Define the slopes of the two lines
noncomputable def slope1 : ℝ := 1/2
noncomputable def slope2 (k : ℝ) : ℝ := 18/k^2

-- Define the y-intercepts of the two lines
noncomputable def intercept1 : ℝ := -3/2
noncomputable def intercept2 (k : ℝ) : ℝ := -9/k

-- Theorem statement
theorem parallel_distinct_lines :
  ∃! k : ℝ, k ≠ 0 ∧ slope1 = slope2 k ∧ intercept1 ≠ intercept2 k ∧ k = -6 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_distinct_lines_l503_50347


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exactly_two_conclusions_true_l503_50393

structure Space where
  Line : Type
  Plane : Type
  parallel_line_plane : Line → Plane → Prop
  parallel_plane_plane : Plane → Plane → Prop
  line_in_plane : Line → Plane → Prop
  parallel_line_line : Line → Line → Prop

variable (S : Space)

def conclusion1 (l : S.Line) (α : S.Plane) : Prop :=
  S.parallel_line_plane l α → ∃ (lines : Set S.Line), Set.Infinite lines ∧ 
    ∀ l' ∈ lines, S.line_in_plane l' α ∧ S.parallel_line_line l' l

def conclusion2 (l : S.Line) (α : S.Plane) : Prop :=
  S.parallel_line_plane l α → ∀ l', S.line_in_plane l' α → S.parallel_line_line l' l

def conclusion3 (α β : S.Plane) : Prop :=
  S.parallel_plane_plane α β → ∀ l, S.line_in_plane l α → S.parallel_line_plane l β

def conclusion4 (α β : S.Plane) : Prop :=
  S.parallel_plane_plane α β → ∀ l, S.line_in_plane l α → 
    ∃! l', S.line_in_plane l' β ∧ S.parallel_line_line l l'

theorem exactly_two_conclusions_true :
  ∃! (correct : Finset (Fin 4)), correct.card = 2 ∧
    (∀ i ∈ correct, match i with
      | 0 => ∀ l α, conclusion1 S l α
      | 1 => ∀ l α, conclusion2 S l α
      | 2 => ∀ α β, conclusion3 S α β
      | 3 => ∀ α β, conclusion4 S α β) ∧
    (∀ i ∉ correct, match i with
      | 0 => ∃ l α, ¬conclusion1 S l α
      | 1 => ∃ l α, ¬conclusion2 S l α
      | 2 => ∃ α β, ¬conclusion3 S α β
      | 3 => ∃ α β, ¬conclusion4 S α β) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_exactly_two_conclusions_true_l503_50393


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_line_polar_equation_l503_50328

/-- PolarCoord represents a point in polar coordinates (ρ, θ) -/
structure PolarCoord where
  ρ : ℝ
  θ : ℝ

/-- line_through_perpendicular_to_polar_axis P represents the set of points
    on the line passing through P and perpendicular to the polar axis -/
def line_through_perpendicular_to_polar_axis (P : PolarCoord) : Set PolarCoord :=
  sorry

/-- Given a point P in polar coordinates (1, π/3), the polar equation of the line
    passing through P and perpendicular to the polar axis is ρ = 1/(2cos θ). -/
theorem perpendicular_line_polar_equation (P : PolarCoord) (h : P = PolarCoord.mk 1 (π/3)) :
  ∃ (f : ℝ → ℝ), (∀ θ, f θ = 1 / (2 * Real.cos θ)) ∧
    (∀ ρ θ, PolarCoord.mk ρ θ ∈ line_through_perpendicular_to_polar_axis P ↔ ρ = f θ) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_line_polar_equation_l503_50328


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_gretchen_walking_distance_l503_50332

/-- Calculates the total distance walked during a workday based on sitting time and walking rules -/
noncomputable def total_distance_walked (total_hours : ℕ) (meeting_hours : ℕ) (desk_hours : ℕ) 
  (lunch_break_hours : ℕ) (lunch_break_walk_minutes : ℕ) (walking_speed : ℝ) 
  (sitting_interval : ℕ) (walking_interval : ℕ) : ℝ :=
  let total_sitting_minutes : ℕ := desk_hours * 60 + (lunch_break_hours * 60 - lunch_break_walk_minutes)
  let walking_intervals : ℕ := total_sitting_minutes / sitting_interval
  let rule_based_walking_minutes : ℕ := walking_intervals * walking_interval
  let total_walking_minutes : ℕ := rule_based_walking_minutes + lunch_break_walk_minutes
  let total_walking_hours : ℝ := (total_walking_minutes : ℝ) / 60
  walking_speed * total_walking_hours

/-- Theorem stating that given the specific conditions, Gretchen walks 4.5 miles during her workday -/
theorem gretchen_walking_distance : 
  total_distance_walked 8 2 4 2 30 3 75 15 = 4.5 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_gretchen_walking_distance_l503_50332


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_factor_for_perfect_cube_l503_50391

open Nat BigOperators

def is_cube (n : ℕ) : Prop := ∃ m : ℕ, n = m^3

theorem smallest_factor_for_perfect_cube : ∃ (n : ℕ), n > 0 ∧ 
  (∀ (m : ℕ), m > 0 → is_cube (45216 * m) → n ≤ m) ∧ 
  is_cube (45216 * n) ∧ n = 7 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_factor_for_perfect_cube_l503_50391


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l503_50398

-- Define the function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (1 - x) / (a * x) + Real.log x

-- Define the function g
noncomputable def g (x : ℝ) : ℝ := Real.log (1 + x) - x

theorem problem_solution 
  (a b : ℝ) 
  (h1 : a > 0)
  (h2 : b > 0)
  (h3 : a > 1)
  (h4 : ∀ x > 1, Monotone (fun x => f a x)) :
  (a ≥ 1) ∧ 
  (∀ x ≥ 0, g x ≤ 0) ∧
  (1 / (a + b) ≤ Real.log ((a + b) / b) ∧ Real.log ((a + b) / b) < a / b) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l503_50398


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_trig_expression_l503_50344

/-- Given a triangle PQR with side lengths PQ = 7, PR = 6, and QR = 8,
    the expression (cos((P-Q)/2) / sin(R/2)) - (sin((P-Q)/2) / cos(R/2)) equals 12/7 -/
theorem triangle_trig_expression (P Q R : ℝ) 
  (h1 : 7 = ‖Q - P‖) (h2 : 6 = ‖R - P‖) (h3 : 8 = ‖R - Q‖) :
  (Real.cos ((P - Q) / 2) / Real.sin (R / 2)) - 
  (Real.sin ((P - Q) / 2) / Real.cos (R / 2)) = 12 / 7 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_trig_expression_l503_50344


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_chord_length_MN_l503_50358

-- Define the curves in polar coordinates
noncomputable def C₁ (θ : ℝ) : ℝ := 2 * Real.sin θ
noncomputable def C₂ : ℝ := Real.pi / 3

-- Define the intersection points M and N (existence assumed)
def M : ℝ × ℝ := sorry
def N : ℝ × ℝ := sorry

-- State the theorem
theorem chord_length_MN :
  let d := Real.sqrt 3
  (M.1 - N.1)^2 + (M.2 - N.2)^2 = d^2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_chord_length_MN_l503_50358


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_consecutive_product_divisibility_l503_50317

theorem consecutive_product_divisibility (n : ℕ) :
  (∃ k : ℤ, n = Int.natAbs (k * (k + 1) * (k + 2) * (k + 3))) →
  11 ∣ n →
  ¬ (∀ m : ℕ, (∃ k : ℤ, m = Int.natAbs (k * (k + 1) * (k + 2) * (k + 3))) → 132 ∣ m) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_consecutive_product_divisibility_l503_50317


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_count_l503_50338

open Real

-- Define the line and curve equations
noncomputable def line (t : ℝ) : ℝ × ℝ := (t * cos (75 * π / 180), t * sin (75 * π / 180))

noncomputable def curve (θ : ℝ) : ℝ × ℝ := (3 * sin θ, 2 * cos θ)

-- Define the number of intersection points
def num_intersections : ℕ := 2

-- Theorem statement
theorem intersection_count :
  ∃ (t₁ t₂ θ₁ θ₂ : ℝ), t₁ ≠ t₂ ∧ θ₁ ≠ θ₂ ∧ 
  line t₁ = curve θ₁ ∧ line t₂ = curve θ₂ ∧
  (∀ (t θ : ℝ), line t = curve θ → (t = t₁ ∧ θ = θ₁) ∨ (t = t₂ ∧ θ = θ₂)) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_count_l503_50338


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_partition_into_benelux_sets_l503_50326

/-- A finite set of integers is called bad if its elements add up to 2010. -/
def IsBad (s : Finset ℤ) : Prop :=
  s.sum id = 2010

/-- A finite set of integers is a Benelux-set if none of its subsets is bad. -/
def IsBeneluxSet (s : Finset ℤ) : Prop :=
  ∀ t : Finset ℤ, t ⊆ s → ¬IsBad t

/-- The set S = {502, 503, 504, ..., 2009} -/
def S : Finset ℤ :=
  Finset.image (fun i : ℕ ↦ (i : ℤ) + 502) (Finset.range 1508)

/-- A partition of a set S into n subsets is a collection of n pairwise disjoint subsets of S, the union of which equals S. -/
def IsPartition (S : Finset ℤ) (P : Finset (Finset ℤ)) : Prop :=
  (∀ s, s ∈ P → s ⊆ S) ∧
  (∀ s t, s ∈ P → t ∈ P → s ≠ t → Disjoint s t) ∧
  (P.biUnion id = S)

/-- The theorem to be proved -/
theorem smallest_partition_into_benelux_sets :
  ∃ P : Finset (Finset ℤ),
    IsPartition S P ∧
    (∀ s, s ∈ P → IsBeneluxSet s) ∧
    P.card = 2 ∧
    (∀ Q : Finset (Finset ℤ), IsPartition S Q → (∀ s, s ∈ Q → IsBeneluxSet s) → Q.card ≥ 2) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_partition_into_benelux_sets_l503_50326


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_3x_period_l503_50388

/-- The period of the tangent function with a coefficient -/
noncomputable def tanPeriod (b : ℝ) : ℝ := Real.pi / b

/-- The function y = tan(3x) -/
noncomputable def f (x : ℝ) : ℝ := Real.tan (3 * x)

/-- Theorem: The period of y = tan(3x) is π/3 -/
theorem tan_3x_period : ∀ x : ℝ, f (x + tanPeriod 3) = f x := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_3x_period_l503_50388


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_root_interval_l503_50340

-- Define the function f
noncomputable def f (a b x : ℝ) : ℝ := a^x + x - b

-- State the theorem
theorem root_interval (a b : ℝ) (n : ℤ) :
  (2:ℝ)^a = 3 →
  (3:ℝ)^b = 2 →
  (∃ x₀ : ℝ, x₀ ∈ Set.Ioo (n:ℝ) ((n:ℝ) + 1) ∧ f a b x₀ = 0) →
  n = -1 :=
by
  sorry

#check root_interval

end NUMINAMATH_CALUDE_ERRORFEEDBACK_root_interval_l503_50340


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_problem_l503_50337

/-- Two-dimensional vector type -/
def Vector2D := ℝ × ℝ

/-- Dot product of two 2D vectors -/
def dot_product (v w : Vector2D) : ℝ := v.1 * w.1 + v.2 * w.2

/-- Magnitude (length) of a 2D vector -/
noncomputable def magnitude (v : Vector2D) : ℝ := Real.sqrt (v.1^2 + v.2^2)

/-- Angle between two 2D vectors -/
noncomputable def angle (v w : Vector2D) : ℝ := 
  Real.arccos (dot_product v w / (magnitude v * magnitude w))

theorem vector_problem :
  let a : Vector2D := (1, -1)
  ∀ b : Vector2D,
  magnitude b = 1 →
  (angle a b = π / 3 → dot_product a b = Real.sqrt 2 / 2) ∧
  (dot_product (a.1 - b.1, a.2 - b.2) b = 0 → angle a b = π / 4) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_problem_l503_50337


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_distance_circle_to_line_l503_50300

/-- Line l with parametric equation x = t, y = 4 - t -/
def line_l (t : ℝ) : ℝ × ℝ := (t, 4 - t)

/-- Circle C with equation x^2 + y^2 = 4 -/
def circle_C : Set (ℝ × ℝ) := {p | p.1^2 + p.2^2 = 4}

/-- Distance from a point to line l -/
noncomputable def distance_to_line (p : ℝ × ℝ) : ℝ :=
  |p.2 + p.1 - 4| / Real.sqrt 2

/-- Theorem: The maximum distance from any point on circle C to line l is 3√2 -/
theorem max_distance_circle_to_line :
  ∃ (max_dist : ℝ), max_dist = 3 * Real.sqrt 2 ∧
  ∀ (p : ℝ × ℝ), p ∈ circle_C →
  distance_to_line p ≤ max_dist ∧
  ∃ (q : ℝ × ℝ), q ∈ circle_C ∧ distance_to_line q = max_dist := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_distance_circle_to_line_l503_50300


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inequalities_not_equivalent_inequality_1_not_implies_2_inequality_2_not_implies_1_l503_50394

noncomputable section

def inequality_1 (a b c : ℝ) : Prop :=
  a^2 + b^2 + c^2 ≤ 2*(a*b + b*c + a*c)

def inequality_2 (a b c : ℝ) : Prop :=
  a^4 + b^4 + c^4 ≤ 2*(a^2*b^2 + b^2*c^2 + a^2*c^2)

theorem inequalities_not_equivalent (a b c : ℝ) (h : a ≥ b ∧ b ≥ c ∧ c ≥ 0) :
  ¬(∀ x y z : ℝ, x ≥ y ∧ y ≥ z ∧ z ≥ 0 → (inequality_1 x y z ↔ inequality_2 x y z)) := by
  sorry

theorem inequality_1_not_implies_2 (a b c : ℝ) (h : a ≥ b ∧ b ≥ c ∧ c ≥ 0) :
  ¬(∀ x y z : ℝ, x ≥ y ∧ y ≥ z ∧ z ≥ 0 → (inequality_1 x y z → inequality_2 x y z)) := by
  sorry

theorem inequality_2_not_implies_1 (a b c : ℝ) (h : a ≥ b ∧ b ≥ c ∧ c ≥ 0) :
  ¬(∀ x y z : ℝ, x ≥ y ∧ y ≥ z ∧ z ≥ 0 → (inequality_2 x y z → inequality_1 x y z)) := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inequalities_not_equivalent_inequality_1_not_implies_2_inequality_2_not_implies_1_l503_50394


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_i_squared_complex_fraction_simplification_l503_50318

-- Define the complex number i
noncomputable def i : ℂ := Complex.I

-- Define the property i² = -1
theorem i_squared : i ^ 2 = -1 := Complex.I_sq

-- State the theorem
theorem complex_fraction_simplification :
  (2 - 2 * i) / (3 + 4 * i) = -2 / 25 - 14 / 25 * i :=
by
  -- Proof steps would go here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_i_squared_complex_fraction_simplification_l503_50318


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_nFacPow_divisibility_l503_50360

def nFacPow (n : ℕ) : ℕ → ℕ
  | 0 => n
  | 1 => Nat.factorial n
  | k+1 => Nat.factorial (nFacPow n k)

def productTerm (n : ℕ) (k : ℕ) : ℕ :=
  Nat.factorial n * Nat.factorial (n - 1) * Nat.factorial (Nat.factorial n - 1) *
  (Nat.factorial (nFacPow n 2 - 1) *
   (Finset.range (k - 2)).prod (fun i => Nat.factorial (nFacPow n (i + 2) - 1)))

theorem nFacPow_divisibility (n : ℕ) (k : ℕ) (h : k ≥ 2) :
  (productTerm n k) ∣ (nFacPow n k) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_nFacPow_divisibility_l503_50360


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_athletes_running_problem_l503_50349

/-- The athletes' running problem -/
theorem athletes_running_problem 
  (field_length : ℝ) 
  (mary_fraction edna_fraction lucy_fraction john_fraction susan_fraction : ℚ)
  (h1 : field_length = 24)
  (h2 : mary_fraction = 3/8)
  (h3 : edna_fraction = 2/3)
  (h4 : lucy_fraction = 5/6)
  (h5 : john_fraction = 13/16)
  (h6 : susan_fraction = 8/15) :
  let mary_distance := mary_fraction * field_length
  let edna_distance := edna_fraction * mary_distance
  let lucy_distance := lucy_fraction * edna_distance
  let john_distance := john_fraction * lucy_distance
  let susan_distance := susan_fraction * john_distance
  (mary_distance - lucy_distance = 4) ∧ 
  (edna_distance - john_distance = 1.75) ∧
  (edna_distance - susan_distance = 3.75) := by
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_athletes_running_problem_l503_50349


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_integer_pairs_satisfying_equation_l503_50316

theorem integer_pairs_satisfying_equation :
  ∃! (s : Finset (ℤ × ℤ)), 
    (∀ (x y : ℤ), (x, y) ∈ s ↔ x^4 + x^2 + y^2 = 2*y + 1) ∧ 
    s.card = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_integer_pairs_satisfying_equation_l503_50316


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_properties_l503_50374

/-- Represents an ellipse with given properties -/
structure Ellipse where
  major_axis : ℝ
  focus1 : ℝ × ℝ
  focus2 : ℝ × ℝ

/-- The standard equation of an ellipse -/
def standard_equation (x y : ℝ) : Prop :=
  x^2 / 4 + y^2 / 3 = 1

/-- The eccentricity of an ellipse -/
noncomputable def eccentricity : ℝ :=
  1 / 2

theorem ellipse_properties (e : Ellipse) 
    (h1 : e.major_axis = 4)
    (h2 : e.focus1 = (-1, 0))
    (h3 : e.focus2 = (1, 0)) :
  (∀ x y, standard_equation x y) ∧ eccentricity = 1 / 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_properties_l503_50374


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_race_medal_awards_l503_50351

/-- The number of ways to award medals in a race with the given constraints -/
def medal_award_ways (total_sprinters : ℕ) (american_sprinters : ℕ) (medals : ℕ) : ℕ :=
  let non_american_sprinters := total_sprinters - american_sprinters
  let no_american_medals := (non_american_sprinters.choose medals) * medals.factorial
  let one_american_medal := american_sprinters * medals * ((non_american_sprinters.choose (medals - 1)) * (medals - 1).factorial)
  no_american_medals + one_american_medal

/-- Theorem stating the number of ways to award medals in the specific race scenario -/
theorem race_medal_awards :
  medal_award_ways 10 4 3 = 480 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_race_medal_awards_l503_50351


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_discount_is_five_percent_l503_50307

/-- Represents the cost of an iPhone X in dollars -/
noncomputable def iphone_cost : ℝ := 600

/-- Represents the total savings when buying three iPhones together -/
noncomputable def total_savings : ℝ := 90

/-- Represents the number of iPhones purchased together -/
def num_iphones : ℕ := 3

/-- Calculates the discount percentage offered by the smartphone seller -/
noncomputable def discount_percentage : ℝ :=
  (total_savings / (iphone_cost * (num_iphones : ℝ))) * 100

/-- Theorem stating that the discount percentage is 5% -/
theorem discount_is_five_percent :
  discount_percentage = 5 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_discount_is_five_percent_l503_50307


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_points_theorem_l503_50313

noncomputable def A : ℝ × ℝ := (1, 0)
noncomputable def B : ℝ × ℝ := (4, 0)

def is_on_ellipse (P : ℝ × ℝ) (a : ℝ) : Prop :=
  (P.1^2 / a^2) + (P.2^2 / (a^2 - 1)) = 1

noncomputable def distance (P Q : ℝ × ℝ) : ℝ :=
  Real.sqrt ((P.1 - Q.1)^2 + (P.2 - Q.2)^2)

def satisfies_ratio (P : ℝ × ℝ) : Prop :=
  distance P A / distance P B = 1/2

theorem ellipse_points_theorem (a : ℝ) :
  (∃ P : ℝ × ℝ, is_on_ellipse P a ∧ satisfies_ratio P) ↔ 
  (a = 2 ∨ a = -2 ∨ a = Real.sqrt 5 ∨ a = -Real.sqrt 5) := by
  sorry

#check ellipse_points_theorem

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_points_theorem_l503_50313


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_center_of_symmetry_monotonicity_in_interval_l503_50362

open Real

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := sin x ^ 2 - cos (x + π / 3) ^ 2

-- Theorem for the center of symmetry
theorem center_of_symmetry :
  ∃ (k : ℤ), ∀ (x : ℝ), f (k * π / 2 + π / 12 + x) = f (k * π / 2 + π / 12 - x) :=
sorry

-- Theorem for monotonicity in the given interval
theorem monotonicity_in_interval :
  (∀ (x y : ℝ), -π/6 ≤ x ∧ x < y ∧ y ≤ π/4 → f x < f y) ∧
  (∀ (x y : ℝ), -π/3 ≤ x ∧ x < y ∧ y ≤ -π/6 → f x > f y) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_center_of_symmetry_monotonicity_in_interval_l503_50362


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_volume_ratio_sum_m_n_l503_50367

/-- Represents a point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- The original cube with vertices at (±1, ±1, ±1) -/
def originalCube : Set Point3D :=
  {p | (p.x = 1 ∨ p.x = -1) ∧ (p.y = 1 ∨ p.y = -1) ∧ (p.z = 1 ∨ p.z = -1)}

/-- The volume of the original cube -/
def originalVolume : ℝ := 8

/-- A smaller internal cube formed by joining midpoints -/
def smallerCube : Set Point3D :=
  {p | (p.x = 1 ∧ p.y = 0 ∧ p.z = 0) ∨
       (p.x = -1 ∧ p.y = 0 ∧ p.z = 0) ∨
       (p.x = 0 ∧ p.y = 1 ∧ p.z = 0) ∨
       (p.x = 0 ∧ p.y = -1 ∧ p.z = 0) ∨
       (p.x = 0 ∧ p.y = 0 ∧ p.z = 1) ∨
       (p.x = 0 ∧ p.y = 0 ∧ p.z = -1)}

/-- The volume of a smaller internal cube -/
noncomputable def smallerVolume : ℝ := (Real.sqrt 2 / 2) ^ 3

/-- The theorem stating the ratio of volumes -/
theorem volume_ratio :
  smallerVolume / originalVolume = 1 / 32 := by sorry

/-- The sum of m and n, where m/n is the ratio of volumes -/
def m_plus_n : ℕ := 33

/-- The theorem stating that m + n = 33 -/
theorem sum_m_n : m_plus_n = 33 := by rfl

end NUMINAMATH_CALUDE_ERRORFEEDBACK_volume_ratio_sum_m_n_l503_50367


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pizza_volume_theorem_l503_50368

/-- The volume of a piece of pizza -/
noncomputable def pizza_piece_volume (thickness : ℝ) (diameter : ℝ) (num_pieces : ℕ) : ℝ :=
  (thickness * Real.pi * (diameter / 2)^2) / num_pieces

/-- Theorem: The volume of one piece of a round pizza is 2π cubic inches -/
theorem pizza_volume_theorem :
  pizza_piece_volume (1/4) 16 8 = 2 * Real.pi := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_pizza_volume_theorem_l503_50368


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_is_52_l503_50311

/-- Represents a type of rock with its weight and value -/
structure Rock where
  weight : Nat
  value : Nat

/-- The knapsack problem setup -/
structure KnapsackProblem where
  rocks : List Rock
  weightCapacity : Nat
  itemCapacity : Nat

/-- Calculates the total weight of a list of rocks -/
def totalWeight (rocks : List Rock) : Nat :=
  rocks.foldl (fun acc r => acc + r.weight) 0

/-- Calculates the total value of a list of rocks -/
def totalValue (rocks : List Rock) : Nat :=
  rocks.foldl (fun acc r => acc + r.value) 0

/-- Checks if a list of rocks is valid for the given problem -/
def isValidSelection (problem : KnapsackProblem) (selection : List Rock) : Prop :=
  totalWeight selection ≤ problem.weightCapacity ∧
  selection.length ≤ problem.itemCapacity

/-- The main theorem to prove -/
theorem max_value_is_52 (problem : KnapsackProblem) :
  problem.rocks = [⟨6, 16⟩, ⟨3, 9⟩, ⟨2, 3⟩] ∧
  problem.weightCapacity = 20 ∧
  problem.itemCapacity = 5 →
  (∃ (selection : List Rock),
    isValidSelection problem selection ∧
    totalValue selection = 52 ∧
    (∀ (otherSelection : List Rock),
      isValidSelection problem otherSelection →
      totalValue otherSelection ≤ 52)) := by
  sorry

#check max_value_is_52

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_is_52_l503_50311


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vertex_on_y₂_when_m_zero_locus_of_vertices_y₂_less_than_y₁_implies_m_minus_n_leq_one_l503_50333

-- Define the functions y₁ and y₂
noncomputable def y₁ (m x : ℝ) : ℝ := -x^2 + (m+2)*x - 2*m + 1
noncomputable def y₂ (n x : ℝ) : ℝ := (n+2)*x - 2*n - 3

-- Define the vertex of a quadratic function
noncomputable def vertex (m : ℝ) : ℝ × ℝ := ((m+2)/2, y₁ m ((m+2)/2))

-- Part 1
theorem vertex_on_y₂_when_m_zero (n : ℝ) (h : n ≠ -2) :
  y₂ n (vertex 0).1 = (vertex 0).2 → n = -3 := by sorry

-- Part 2
theorem locus_of_vertices (m x y : ℝ) :
  (∃ m, vertex m = (x, y)) ↔ y = x^2 - 4*x + 5 := by sorry

-- Part 3
theorem y₂_less_than_y₁_implies_m_minus_n_leq_one (m n : ℝ) :
  (∀ x, -1 < x → x < 2 → y₂ n x < y₁ m x) → m - n ≤ 1 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_vertex_on_y₂_when_m_zero_locus_of_vertices_y₂_less_than_y₁_implies_m_minus_n_leq_one_l503_50333


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_cos_shift_l503_50339

theorem sin_cos_shift (x : ℝ) :
  Real.sin (2 * x) - Real.cos (2 * x) = Real.sqrt 2 * Real.sin (2 * (x - Real.pi / 8)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_cos_shift_l503_50339


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_root_floor_is_two_l503_50315

-- Define the function f(x) = ln x - 2/x
noncomputable def f (x : ℝ) : ℝ := Real.log x - 2 / x

-- Define the floor function
noncomputable def floor (x : ℝ) : ℤ := ⌊x⌋

-- Theorem statement
theorem root_floor_is_two (x₀ : ℝ) (h : f x₀ = 0) : floor x₀ = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_root_floor_is_two_l503_50315


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_constant_distance_l503_50355

/-- An ellipse with semi-major axis a and semi-minor axis b -/
structure Ellipse (a b : ℝ) : Type where
  h : a > b ∧ b > 0

/-- A point on the ellipse -/
def PointOnEllipse (a b : ℝ) (x y : ℝ) : Prop :=
  x^2 / a^2 + y^2 / b^2 = 1

/-- The distance between two points -/
noncomputable def distance (x₁ y₁ x₂ y₂ : ℝ) : ℝ :=
  Real.sqrt ((x₂ - x₁)^2 + (y₂ - y₁)^2)

/-- The intersection points M and N -/
noncomputable def MNPoints (a b : ℝ) (P : ℝ × ℝ) : (ℝ × ℝ) × (ℝ × ℝ) :=
  sorry

/-- The theorem to be proved -/
theorem ellipse_constant_distance (a b : ℝ) (e : Ellipse a b) :
  (∀ P : ℝ × ℝ, PointOnEllipse a b P.1 P.2 →
    let (M, N) := MNPoints a b P
    ∃ c : ℝ, ∀ Q : ℝ × ℝ, PointOnEllipse a b Q.1 Q.2 →
      let (M', N') := MNPoints a b Q
      distance M.1 M.2 N.1 N.2 = c) →
  Real.sqrt (a / b) = 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_constant_distance_l503_50355


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_x0_implies_a_lower_bound_l503_50396

noncomputable def f (a x : ℝ) : ℝ := (x - a - 1) * Real.exp x - 1/2 * x^2 + a * x

theorem exists_x0_implies_a_lower_bound (a : ℝ) :
  (∃ x0 : ℝ, x0 ∈ Set.Icc 1 2 ∧ f a x0 < 0) →
  a > 1 / (2 * (1 - Real.exp 1)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_x0_implies_a_lower_bound_l503_50396


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_expected_trials_bound_l503_50359

/-- The success probability of obtaining a premium fruit -/
def p : ℝ := 0.2

/-- The maximum number of samples -/
def n : ℕ := 4

/-- The expected value of the number of samples -/
def E (n : ℕ) : ℝ := 5 * (1 - (1 - p)^n)

/-- Theorem stating that the expected number of trials does not exceed 3 
    if and only if n ≤ 4 -/
theorem expected_trials_bound : 
  E n ≤ 3 ↔ n ≤ 4 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_expected_trials_bound_l503_50359


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_f_nonnegative_l503_50323

-- Define the function f(x) = log₂x
noncomputable def f (x : ℝ) : ℝ := Real.log x / Real.log 2

-- Define the interval [1/2, 2]
def interval : Set ℝ := {x | 1/2 ≤ x ∧ x ≤ 2}

-- Define a random variable x₀ in the interval [1/2, 2]
noncomputable def x₀ : ℝ := sorry

-- State that x₀ is in the interval [1/2, 2]
axiom x₀_in_interval : x₀ ∈ interval

-- Define the probability measure on the interval [1/2, 2]
noncomputable def prob : Set ℝ → ℝ := sorry

-- State that the probability measure is uniform on the interval
axiom prob_uniform : ∀ (a b : ℝ), a ∈ interval → b ∈ interval → 
  prob {x | a ≤ x ∧ x ≤ b} = (b - a) / (2 - 1/2)

-- Theorem: The probability that f(x₀) ≥ 0 is 2/3
theorem prob_f_nonnegative : prob {x ∈ interval | f x ≥ 0} = 2/3 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_f_nonnegative_l503_50323


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_elephant_weight_in_pounds_l503_50308

/-- Conversion factor from kilograms to pounds -/
noncomputable def kg_to_pound : ℝ := 1 / 0.4536

/-- Weight of the elephant in kilograms -/
def elephant_weight_kg : ℝ := 1200

/-- Converts kilograms to pounds and rounds to the nearest integer -/
noncomputable def kg_to_nearest_pound (kg : ℝ) : ℤ :=
  Int.floor (kg * kg_to_pound + 0.5)

theorem elephant_weight_in_pounds :
  kg_to_nearest_pound elephant_weight_kg = 2646 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_elephant_weight_in_pounds_l503_50308


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_remainder_l503_50395

theorem polynomial_remainder :
  ∃! (P S : Polynomial ℂ),
    (X : Polynomial ℂ)^2023 + 2 = (X^3 + X^2 + 1) * P + S ∧ 
    Polynomial.degree S < 3 ∧
    S = X + 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_remainder_l503_50395


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangle_area_proof_l503_50379

-- Define the rectangles
structure Rectangle where
  width : ℝ
  height : ℝ

-- Define similarity between rectangles
def similar (r1 r2 : Rectangle) : Prop :=
  r1.width / r1.height = r2.width / r2.height

-- Define diagonal of a rectangle
noncomputable def diagonal (r : Rectangle) : ℝ :=
  Real.sqrt (r.width ^ 2 + r.height ^ 2)

-- Define area of a rectangle
def area (r : Rectangle) : ℝ :=
  r.width * r.height

-- Theorem statement
theorem rectangle_area_proof (r1 r2 : Rectangle) : 
  r1.width = 3 ∧ r1.height = 9 ∧ 
  similar r1 r2 ∧ 
  diagonal r2 = 18 → 
  area r2 = 97.2 := by
  sorry

#check rectangle_area_proof

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangle_area_proof_l503_50379


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l503_50352

noncomputable def f (x : ℝ) := Real.sin x ^ 2 + Real.sqrt 3 * Real.sin x * Real.cos x + 1 / 2

theorem f_properties :
  let period := Real.pi
  let max_value := 2
  let max_points (k : ℤ) := (k : ℝ) * Real.pi + Real.pi / 3
  let decreasing_interval (k : ℤ) := Set.Icc ((k : ℝ) * Real.pi + Real.pi / 3) ((k : ℝ) * Real.pi + 5 * Real.pi / 6)
  ∀ (x : ℝ),
    (∀ (y : ℝ), f (x + period) = f x) ∧
    (f x ≤ max_value) ∧
    (∀ (k : ℤ), f (max_points k) = max_value) ∧
    (∀ (k : ℤ) (y z : ℝ), y ∈ decreasing_interval k → z ∈ decreasing_interval k → y ≤ z → f z ≤ f y) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l503_50352


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_factorial_as_consecutive_product_one_factorial_as_consecutive_product_l503_50353

theorem largest_factorial_as_consecutive_product : 
  ∀ n : ℕ, n > 1 → ¬(∃ a : ℕ, a ≥ 4 ∧ n * Nat.factorial (n - 1) = (Nat.factorial (n - 4 + a)) / (Nat.factorial a)) :=
by sorry

theorem one_factorial_as_consecutive_product : 
  ∃ a : ℕ, a ≥ 4 ∧ Nat.factorial 1 = (Nat.factorial (1 - 4 + a)) / (Nat.factorial a) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_factorial_as_consecutive_product_one_factorial_as_consecutive_product_l503_50353


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_chord_angle_l503_50348

/-- Given a parabola y² = 2px (p > 0) and a chord AB passing through its focus F,
    making an angle θ (0 < θ ≤ π/2) with the x-axis, prove that the angle ∠AOB
    is equal to π - arctan(4 / (3 * sin θ)). -/
theorem parabola_chord_angle (p θ : ℝ) (hp : p > 0) (hθ : 0 < θ ∧ θ ≤ π / 2) :
  let y : ℝ → ℝ := fun x ↦ Real.sqrt (2 * p * x)
  let F : ℝ × ℝ := (p / 2, 0)
  let A : ℝ × ℝ := (p * (1 + Real.cos θ)^2 / (2 * Real.sin θ^2),
                    p * (1 + Real.cos θ) / Real.sin θ)
  let B : ℝ × ℝ := (p * (1 - Real.cos θ)^2 / (2 * Real.sin θ^2),
                    p * (Real.cos θ - 1) / Real.sin θ)
  let O : ℝ × ℝ := (0, 0)
  let angle (P Q R : ℝ × ℝ) : ℝ := Real.arccos (
    ((P.1 - Q.1) * (R.1 - Q.1) + (P.2 - Q.2) * (R.2 - Q.2)) /
    (Real.sqrt ((P.1 - Q.1)^2 + (P.2 - Q.2)^2) * Real.sqrt ((R.1 - Q.1)^2 + (R.2 - Q.2)^2))
  )
  angle A O B = π - Real.arctan (4 / (3 * Real.sin θ)) := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_chord_angle_l503_50348


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_v_coefficient_sum_l503_50321

/-- Sequence v_n defined recursively -/
def v : ℕ → ℚ
  | 0 => 7
  | n + 1 => v n + (2 + 5 * n)

/-- v_n as a polynomial function -/
def v_poly (n : ℚ) : ℚ := 5/2 * n^2 - 11/2 * n + 10

theorem v_coefficient_sum :
  (∃ (a b c : ℚ), ∀ n : ℕ, v n = a * (n : ℚ)^2 + b * (n : ℚ) + c) →
  (∃ (a b c : ℚ), (∀ n : ℕ, v n = a * (n : ℚ)^2 + b * (n : ℚ) + c) ∧ a + b + c = 2) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_v_coefficient_sum_l503_50321


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_det_B_squared_minus_3B_l503_50383

def B : Matrix (Fin 2) (Fin 2) ℚ := !![2, 4; 3, 2]

theorem det_B_squared_minus_3B : Matrix.det (B ^ 2 - 3 • B) = 88 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_det_B_squared_minus_3B_l503_50383


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l503_50356

theorem hyperbola_eccentricity :
  let hyperbola := (fun (x y : ℝ) => x^2 / 2 - y^2 / 3 = 1)
  ∃ (e : ℝ), e = Real.sqrt 10 / 2 ∧
    ∀ (x y : ℝ), hyperbola x y →
      e = Real.sqrt (1 + (3 / 2)) :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l503_50356


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_and_p_range_l503_50322

noncomputable def f (p : ℝ) (x : ℝ) : ℝ := p * x - p / x - 2 * Real.log x

noncomputable def g (x : ℝ) : ℝ := 2 * Real.exp 1 / x

theorem tangent_line_and_p_range :
  (∀ x : ℝ, x > 0 →
    (∀ y : ℝ, y = 2 * x - 2 ↔ y = f 2 1 + (deriv (f 2)) 1 * (x - 1))) ∧
  (∃ p : ℝ, p > 4 * Real.exp 1 / (Real.exp 2 - 1) ∧
    (∀ x : ℝ, x > 0 → (deriv (f p)) x ≥ 0) ∧
    (∃ x₀ : ℝ, x₀ ∈ Set.Icc 1 (Real.exp 1) ∧ f p x₀ > g x₀)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_and_p_range_l503_50322


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_five_digit_odd_number_problem_l503_50364

/-- A function that replaces 2s with 5s and 5s with 2s in a number -/
def replace_digits (n : ℕ) : ℕ := sorry

theorem five_digit_odd_number_problem :
  ∀ x : ℕ,
  (10000 ≤ x ∧ x < 100000) →  -- x is a five-digit number
  (x % 2 = 1) →               -- x is odd
  (let y := replace_digits x
   y = 2 * (x + 1)) →         -- y = 2(x + 1)
  x = 29995 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_five_digit_odd_number_problem_l503_50364


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_red_dragons_l503_50314

/-- Represents the color of a dragon -/
inductive DragonColor
| Red
| Green
| Blue
deriving Repr, DecidableEq

/-- Represents a dragon with its color and the colors of adjacent dragons -/
structure Dragon where
  color : DragonColor
  leftColor : DragonColor
  rightColor : DragonColor
deriving Repr

/-- Represents the state of a head (truthful or lying) -/
inductive HeadState
| Truthful
| Lying
deriving Repr, DecidableEq

/-- Defines the constraints for a valid dragon based on the problem description -/
def isValidDragon (d : Dragon) (h1 h2 h3 : HeadState) : Prop :=
  (h1 = HeadState.Truthful ∨ h2 = HeadState.Truthful ∨ h3 = HeadState.Truthful) ∧
  (d.color = DragonColor.Red →
    ((h1 = HeadState.Truthful → d.leftColor = DragonColor.Green) ∧
     (h2 = HeadState.Truthful → d.rightColor = DragonColor.Blue) ∧
     (h3 = HeadState.Truthful → d.leftColor ≠ DragonColor.Red ∧ d.rightColor ≠ DragonColor.Red)))

/-- Theorem stating the maximum number of red dragons in a group of 530 dragons -/
theorem max_red_dragons (dragons : Fin 530 → Dragon) 
  (valid : ∀ i, ∃ h1 h2 h3, isValidDragon (dragons i) h1 h2 h3) :
  (Finset.filter (λ i => (dragons i).color = DragonColor.Red) Finset.univ).card ≤ 176 := by
  sorry

#check max_red_dragons

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_red_dragons_l503_50314


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_area_is_100_l503_50384

/-- Two circles touching externally with diameters AB and CD forming a trapezoid ABCD -/
structure ExternallyTouchingCircles where
  /-- Length of diameter AB -/
  ab : ℝ
  /-- Length of diameter CD -/
  cd : ℝ
  /-- AB is positive -/
  ab_pos : 0 < ab
  /-- CD is positive -/
  cd_pos : 0 < cd
  /-- The circles touch externally -/
  touching_externally : True

/-- The maximum area of the trapezoid ABCD formed by two externally touching circles -/
noncomputable def max_trapezoid_area (circles : ExternallyTouchingCircles) : ℝ :=
  (circles.ab + circles.cd) * (circles.ab / 2 + circles.cd / 2) / 2

/-- Theorem stating the maximum area of the trapezoid ABCD is 100 cm² -/
theorem max_area_is_100 (circles : ExternallyTouchingCircles)
    (h1 : circles.ab = 6)
    (h2 : circles.cd = 14) :
    max_trapezoid_area circles = 100 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_area_is_100_l503_50384


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_neither_red_nor_purple_l503_50329

def total_balls : ℕ := 100
def white_balls : ℕ := 50
def green_balls : ℕ := 30
def yellow_balls : ℕ := 10
def red_balls : ℕ := 7
def purple_balls : ℕ := 3

theorem probability_neither_red_nor_purple :
  (total_balls - (red_balls + purple_balls)) / total_balls = 9 / 10 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_neither_red_nor_purple_l503_50329


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_net_increase_in_one_day_l503_50369

/-- Represents the number of seconds in a day -/
def seconds_per_day : ℕ := 24 * 60 * 60

/-- Represents the birth rate in people per 2 seconds -/
def birth_rate : ℚ := 7

/-- Represents the death rate in people per 2 seconds -/
def death_rate : ℚ := 1

/-- Calculates the net population increase over one day -/
noncomputable def net_population_increase : ℕ := 
  ((birth_rate - death_rate) * (seconds_per_day / 2 : ℚ)).floor.toNat

/-- Theorem stating the net population increase over one day -/
theorem net_increase_in_one_day : 
  net_population_increase = 259200 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_net_increase_in_one_day_l503_50369


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_directrix_l503_50343

/-- The parabola equation -/
noncomputable def parabola (y : ℝ) : ℝ := -1/6 * y^2

/-- The directrix equation -/
noncomputable def directrix : ℝ := 3/2

/-- Theorem: The directrix of the parabola x = -1/6 * y^2 is x = 3/2 -/
theorem parabola_directrix :
  ∀ y : ℝ, ∃ f : ℝ,
    (parabola y - f)^2 + y^2 = (parabola y - directrix)^2 ∧
    f = -directrix :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_directrix_l503_50343


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_greatest_gcd_square_pyramidal_l503_50382

def square_pyramidal (n : ℕ+) : ℕ := (n.val * (n.val + 1) * (2 * n.val + 1)) / 6

theorem greatest_gcd_square_pyramidal : 
  ∃ (n : ℕ+), ∀ (m : ℕ+), Nat.gcd (6 * square_pyramidal m) (m.val - 2) ≤ 12 ∧ 
  Nat.gcd (6 * square_pyramidal n) (n.val - 2) = 12 := by
  sorry

#eval square_pyramidal 14
#eval Nat.gcd (6 * square_pyramidal 14) (14 - 2)

end NUMINAMATH_CALUDE_ERRORFEEDBACK_greatest_gcd_square_pyramidal_l503_50382


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_orange_gain_percentage_l503_50387

-- Define the selling rates
def rate_loss : ℚ := 36
def rate_gain : ℚ := 24

-- Define the loss percentage
def loss_percentage : ℚ := 4

-- Define the function to calculate selling price per orange
def selling_price (rate : ℚ) : ℚ := 1 / rate

-- Define the function to calculate cost price given selling price and loss percentage
def cost_price (sp : ℚ) (loss : ℚ) : ℚ := sp / (1 - loss / 100)

-- Define the function to calculate gain percentage
def gain_percentage (sp : ℚ) (cp : ℚ) : ℚ := (sp - cp) / cp * 100

-- Theorem statement
theorem orange_gain_percentage :
  let sp_loss := selling_price rate_loss
  let cp := cost_price sp_loss loss_percentage
  let sp_gain := selling_price rate_gain
  ∃ (ε : ℚ), abs (gain_percentage sp_gain cp - 44) < ε ∧ ε > 0 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_orange_gain_percentage_l503_50387


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sqrt_equality_l503_50306

theorem sqrt_equality (a b : ℝ) : 
  a = 4 ∧ b = -Real.sqrt 3 → 
  Real.sqrt (16 - 12 * Real.sin (π / 3)) = a + b * (1 / Real.sin (π / 3)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sqrt_equality_l503_50306


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_properties_l503_50325

def mySequence (a : ℕ → ℕ) : Prop :=
  a 1 = 1 ∧
  (∀ n : ℕ, a (2 * n) = a n) ∧
  (∀ n : ℕ, a (4 * n - 1) = 0) ∧
  (∀ n : ℕ, a (4 * n + 1) = 1)

theorem sequence_properties (a : ℕ → ℕ) (h : mySequence a) :
  a 4 = 1 ∧ a 7 = 0 ∧ ¬∃ T : ℕ, T > 0 ∧ ∀ n : ℕ, n > 0 → a (n + T) = a n :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_properties_l503_50325


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_lines_equal_slopes_parallel_lines_m_value_l503_50399

/-- Two lines are parallel if and only if their slopes are equal -/
theorem parallel_lines_equal_slopes {a b c d e f : ℝ} (h : a ≠ 0 ∧ c ≠ 0) : 
  (∃ k : ℝ, a * k = c ∧ b * k = d) ↔ (b / a = d / c) := by sorry

/-- The value of m for which the given lines are parallel -/
theorem parallel_lines_m_value : 
  ∃ m : ℝ, (2 : ℝ) / (m + 1) = m ∧ m = 1 := by
  use 1
  constructor
  · -- Prove (2 : ℝ) / (1 + 1) = 1
    norm_num
  · -- Prove 1 = 1
    rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_lines_equal_slopes_parallel_lines_m_value_l503_50399


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_nested_hexagon_area_ratio_l503_50305

/-- The area of a regular hexagon with side length s -/
noncomputable def regularHexagonArea (s : ℝ) : ℝ := (3 * Real.sqrt 3 / 2) * s^2

/-- The area of a nested hexagon formed by joining points at one-fourth and three-fourths along each side of a regular hexagon with side length s -/
noncomputable def nestedHexagonArea (s : ℝ) : ℝ := regularHexagonArea (s / 2)

/-- Theorem: The area of the nested hexagon is 1/4 of the area of the original hexagon -/
theorem nested_hexagon_area_ratio (s : ℝ) (h : s > 0) :
  nestedHexagonArea s / regularHexagonArea s = 1 / 4 := by
  -- Expand the definitions
  unfold nestedHexagonArea regularHexagonArea
  -- Simplify the expression
  simp [pow_two, mul_assoc, mul_comm, mul_left_comm]
  -- The proof steps would go here, but we'll use sorry for now
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_nested_hexagon_area_ratio_l503_50305


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_plot_perimeter_is_340_l503_50372

/-- Calculates the perimeter of a rectangular plot given the conditions --/
noncomputable def plot_perimeter (width_length_difference : ℝ) (fencing_cost_per_meter : ℝ) (total_fencing_cost : ℝ) : ℝ :=
  let width := (total_fencing_cost / fencing_cost_per_meter - 2 * width_length_difference) / 4
  let length := width + width_length_difference
  2 * (width + length)

/-- Theorem: The perimeter of the plot is 340 meters --/
theorem plot_perimeter_is_340 :
  plot_perimeter 10 6.5 2210 = 340 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_plot_perimeter_is_340_l503_50372


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_at_pi_over_4_m_range_for_g_root_l503_50366

noncomputable def a (x : ℝ) : ℝ × ℝ := (Real.sqrt 2 * Real.sin x, Real.sqrt 2 * Real.cos x)

noncomputable def b (x : ℝ) : ℝ × ℝ := (Real.sin x, 2 * Real.cos x - Real.sin x)

noncomputable def f (x : ℝ) : ℝ :=
  (a x).1 * (b x).1 + (a x).2 * (b x).2 - (3 * Real.sqrt 2) / 2

noncomputable def g (x m : ℝ) : ℝ :=
  Real.sin (4 * x) + Real.sqrt 2 * m * f x - 3

theorem f_at_pi_over_4 : f (Real.pi / 4) = -(Real.sqrt 2) / 2 := by sorry

theorem m_range_for_g_root :
  ∀ m : ℝ, (∃ x : ℝ, -Real.pi / 4 < x ∧ x < Real.pi / 8 ∧ g x m = 0) →
    2 * Real.sqrt 2 ≤ m := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_at_pi_over_4_m_range_for_g_root_l503_50366


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_prime_divisor_existence_l503_50331

theorem prime_divisor_existence (n : ℕ) (h_n : n > 1) :
  ∀ k ∈ Finset.range n,
  ∃ (p : ℕ) (l : ℕ),
  Nat.Prime p ∧
  (n * l + k.succ = p * ((n * l + k.succ) / p)) ∧
  ∀ j ∈ Finset.range n,
  j ≠ k → (n * l + j.succ) % p ≠ 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_prime_divisor_existence_l503_50331


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_distance_inequality_l503_50312

-- Define the circle and line
def circleSet (r : ℝ) : Set (ℝ × ℝ) := {p | p.1^2 + p.2^2 = r^2}
def lineSeg (m : ℝ) : Set (ℝ × ℝ) := {p | m * p.1 - p.2 + 1 = 0}

-- Define the intersection points
def intersectionPoints (r m : ℝ) : Set (ℝ × ℝ) := circleSet r ∩ lineSeg m

-- Define the origin
def origin : ℝ × ℝ := (0, 0)

-- Define the distance between two points
noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ := Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

-- State the theorem
theorem intersection_distance_inequality (r : ℝ) (hr : r > 0) :
  (∃ m : ℝ, ∀ A B, A ∈ intersectionPoints r m → B ∈ intersectionPoints r m → 
    distance origin A + distance origin B ≥ distance A B) ∧ 
  (1 < r ∧ r ≤ Real.sqrt 2) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_distance_inequality_l503_50312


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_sequence_equals_two_fifths_l503_50380

def b : ℕ → ℚ
  | 0 => 2  -- Add this case for 0
  | 1 => 2
  | 2 => 3
  | (n + 3) => b (n + 2) + b (n + 1)

theorem sum_of_sequence_equals_two_fifths :
  ∑' n, b n / 3^(n + 1) = 2 / 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_sequence_equals_two_fifths_l503_50380


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_pentagon_l503_50310

-- Define the rectangle
structure Rectangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ
  D : ℝ × ℝ

-- Define the points E and F
noncomputable def E (rect : Rectangle) : ℝ × ℝ := ((rect.A.1 + rect.B.1) / 2, rect.A.2)
noncomputable def F (rect : Rectangle) : ℝ × ℝ := (rect.C.1, (2 * rect.C.2 + rect.D.2) / 3)

-- Define the intersection point X
noncomputable def X (rect : Rectangle) : ℝ × ℝ := (2, 14/3)

-- Define the area function for the pentagon BCFXE
noncomputable def area_BCFXE (rect : Rectangle) : ℝ :=
  let AB := rect.B.1 - rect.A.1
  let BC := rect.C.2 - rect.B.2
  AB * BC - (AB / 2 * (X rect).2 + BC * (X rect).1 + BC * (F rect).2 / 2)

-- State the theorem
theorem area_pentagon (rect : Rectangle) 
  (h1 : rect.B.1 - rect.A.1 = 12) 
  (h2 : rect.C.2 - rect.B.2 = 7) 
  (h3 : rect.A = (0, 0)) 
  (h4 : rect.B = (12, 0)) 
  (h5 : rect.C = (12, 7)) 
  (h6 : rect.D = (0, 7)) :
  area_BCFXE rect = 329/6 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_pentagon_l503_50310


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_x_intercept_specific_line_l503_50371

/-- The x-intercept of a line passing through two given points -/
noncomputable def x_intercept (x1 y1 x2 y2 : ℝ) : ℝ :=
  (x1 * y2 - x2 * y1) / (y2 - y1)

theorem x_intercept_specific_line :
  x_intercept 3 9 (-1) 1 = -3/2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_x_intercept_specific_line_l503_50371


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tank_capacity_ratio_l503_50303

-- Define the properties of the tanks
def height_A : ℝ := 10
def circumference_A : ℝ := 8
def height_B : ℝ := 8
def circumference_B : ℝ := 10

-- Define the volume of a cylinder
noncomputable def volume (h : ℝ) (c : ℝ) : ℝ := (h * c^2) / (4 * Real.pi)

-- State the theorem
theorem tank_capacity_ratio :
  (volume height_A circumference_A) / (volume height_B circumference_B) = 0.8 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tank_capacity_ratio_l503_50303


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_not_decreasing_in_first_two_quadrants_max_value_of_k_sin_x_plus_1_period_multiple_is_period_sin_symmetry_axis_tan_not_increasing_on_entire_domain_l503_50320

-- (1)
theorem cos_not_decreasing_in_first_two_quadrants :
  ¬(∀ x y : ℝ, 0 ≤ x ∧ x < y ∧ y ≤ π → Real.cos x ≥ Real.cos y) :=
by sorry

-- (2)
theorem max_value_of_k_sin_x_plus_1 (k : ℝ) :
  (∀ x : ℝ, k * Real.sin x + 1 ≤ |k| + 1) ∧
  (∃ x : ℝ, k * Real.sin x + 1 = |k| + 1) :=
by sorry

-- (3)
theorem period_multiple_is_period (f : ℝ → ℝ) (T : ℝ) (k : ℤ) :
  (T ≠ 0 ∧ ∀ x : ℝ, f (x + T) = f x) →
  (k ≠ 0 → ∀ x : ℝ, f (x + ↑k * T) = f x) :=
by sorry

-- (4)
theorem sin_symmetry_axis :
  ∀ k : ℤ, (∀ x : ℝ, Real.sin (↑k * π + π/2 + x) = Real.sin (↑k * π + π/2 - x)) ∧
           (∀ n : ℤ, n ≠ k → ∃ x : ℝ, Real.sin (↑n * π + π/2 + x) ≠ Real.sin (↑n * π + π/2 - x)) :=
by sorry

-- (5)
theorem tan_not_increasing_on_entire_domain :
  ¬(∀ x y : ℝ, x < y → Real.tan x < Real.tan y) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_not_decreasing_in_first_two_quadrants_max_value_of_k_sin_x_plus_1_period_multiple_is_period_sin_symmetry_axis_tan_not_increasing_on_entire_domain_l503_50320


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_prime_factor_of_3113_l503_50327

theorem largest_prime_factor_of_3113 :
  (Nat.factors 3113).maximum? = some 23 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_prime_factor_of_3113_l503_50327


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_property_sum_l503_50350

theorem function_property_sum (f : ℝ → ℝ) 
  (h1 : ∀ x : ℝ, f (x^3) = (f x)^3)
  (h2 : ∀ x₁ x₂ : ℝ, x₁ ≠ x₂ → f x₁ ≠ f x₂) : 
  f 0 + f (-1) + f 1 = 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_property_sum_l503_50350


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_of_A_and_B_l503_50302

-- Define set A
def A : Set ℝ := {x | -x^2 + x + 2 > 0}

-- Define set B
def B : Set ℝ := {x | x^2 + 2*x - 3 < 0}

-- Theorem statement
theorem intersection_of_A_and_B : A ∩ B = {x : ℝ | -1 < x ∧ x < 1} := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_of_A_and_B_l503_50302


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_N_has_at_least_8_divisors_N_has_at_least_32_divisors_l503_50390

/-- The number composed of 1986 ones -/
def N : ℕ := (10^1986 - 1) / 9

/-- The set of divisors of N -/
def divisors_of_N : Finset ℕ := Finset.filter (· ∣ N) (Finset.range (N + 1))

/-- Theorem stating that N has at least 8 different divisors -/
theorem N_has_at_least_8_divisors : Finset.card divisors_of_N ≥ 8 := by
  sorry

/-- Theorem stating that N has at least 32 different divisors -/
theorem N_has_at_least_32_divisors : Finset.card divisors_of_N ≥ 32 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_N_has_at_least_8_divisors_N_has_at_least_32_divisors_l503_50390


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_u_minus_v_negative_l503_50376

/-- Given approximate values for U and V on a number line, prove that U - V is negative. -/
theorem u_minus_v_negative (U V : ℝ) (hU : U ≤ -2.4 ∧ U ≥ -2.6) (hV : V ≤ -0.7 ∧ V ≥ -0.9) : U - V < 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_u_minus_v_negative_l503_50376


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_m_for_nth_root_in_T_l503_50309

def T : Set ℂ :=
  {z | ∃ x y : ℝ, z = x + y * Complex.I ∧ Real.sqrt 3 / 2 ≤ x ∧ x ≤ 2 / Real.sqrt 3}

theorem smallest_m_for_nth_root_in_T : 
  ∃ m : ℕ+, (∀ n : ℕ+, n ≥ m → ∃ z ∈ T, z ^ (n : ℂ) = 1) ∧
  (∀ m' : ℕ+, m' < m → ∃ n : ℕ+, n ≥ m' ∧ ∀ z ∈ T, z ^ (n : ℂ) ≠ 1) ∧
  m = 12 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_m_for_nth_root_in_T_l503_50309


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tax_rate_calculation_l503_50346

/-- Proves that the tax rate on taxable purchases is approximately 226.42% --/
theorem tax_rate_calculation (total_spent : ℝ) (tax_amount_percentage : ℝ) (tax_free_cost : ℝ) :
  total_spent = 40 ∧ 
  tax_amount_percentage = 30 ∧ 
  tax_free_cost = 34.7 →
  ∃ tax_rate : ℝ, abs (tax_rate - 226.42) < 0.01 := by
  intro h
  -- Extract given conditions
  have h1 : total_spent = 40 := h.left
  have h2 : tax_amount_percentage = 30 := h.right.left
  have h3 : tax_free_cost = 34.7 := h.right.right

  -- Calculate tax amount
  let tax_amount := (tax_amount_percentage / 100) * total_spent

  -- Calculate cost of taxable items before tax
  let taxable_cost := total_spent - tax_free_cost

  -- Calculate tax rate
  let calculated_tax_rate := (tax_amount / taxable_cost) * 100

  -- Assert the existence of a tax rate close to 226.42%
  use calculated_tax_rate

  sorry  -- Proof details omitted for brevity


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tax_rate_calculation_l503_50346


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exam_fail_margin_l503_50373

/-- Represents an exam with a total number of marks and a passing mark. -/
structure Exam where
  total_marks : ℕ
  passing_mark : ℕ

/-- Represents a candidate taking the exam. -/
structure Candidate where
  percentage : ℚ
  marks : ℕ

theorem exam_fail_margin (e : Exam) (c1 c2 : Candidate) :
  e.passing_mark = 160 ∧
  c1.percentage = 1/5 ∧
  c2.percentage = 3/10 ∧
  c2.marks = e.passing_mark + 20 ∧
  c1.marks = Int.floor (c1.percentage * e.total_marks) ∧
  c2.marks = Int.floor (c2.percentage * e.total_marks) →
  e.passing_mark - c1.marks = 40 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_exam_fail_margin_l503_50373


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_condition_l503_50336

/-- Two lines in the plane -/
structure Line where
  slope : ℝ
  intercept : ℝ

/-- Definition of perpendicular lines -/
def perpendicular (l1 l2 : Line) : Prop :=
  l1.slope * l2.slope = -1

/-- The first line y = 3x - 4 -/
noncomputable def line1 : Line :=
  { slope := 3, intercept := -4 }

/-- The second line 4y + bx = 8 -/
noncomputable def line2 (b : ℝ) : Line :=
  { slope := -b/4, intercept := 2 }

/-- Theorem stating the condition for perpendicularity -/
theorem perpendicular_condition (b : ℝ) :
  perpendicular line1 (line2 b) ↔ b = 4/3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_condition_l503_50336


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_difference_square_rectangle_l503_50342

/-- The difference between the area of a square with side length 8 cm
    and a rectangle with length 10 cm and width 5 cm is 14 cm². -/
theorem area_difference_square_rectangle : 
  (8 : ℝ) ^ 2 - 10 * 5 = 14 := by
  norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_difference_square_rectangle_l503_50342


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l503_50397

noncomputable def f (x : ℝ) := 4 * Real.sin x ^ 2 - 2 * Real.cos (2 * x) - 1

theorem f_properties :
  (∀ x, f x ≤ 5) ∧ 
  (∀ x, f x ≥ 3) ∧ 
  (∃ x, f x = 5) ∧ 
  (∃ x, f x = 3) ∧
  (∀ m, (∀ x, |f x - m| < 2) ↔ (3 < m ∧ m < 5)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l503_50397
