import Mathlib

namespace NUMINAMATH_CALUDE_salary_percentage_increase_l2963_296373

theorem salary_percentage_increase 
  (original : ℝ) 
  (decrease_percent : ℝ) 
  (increase_percent : ℝ) 
  (overall_decrease_percent : ℝ) 
  (h1 : decrease_percent = 50) 
  (h2 : overall_decrease_percent = 35) 
  (h3 : original * (1 - decrease_percent / 100) * (1 + increase_percent / 100) = 
        original * (1 - overall_decrease_percent / 100)) : 
  increase_percent = 30 := by
sorry

end NUMINAMATH_CALUDE_salary_percentage_increase_l2963_296373


namespace NUMINAMATH_CALUDE_hammer_order_sequence_l2963_296345

theorem hammer_order_sequence (sequence : ℕ → ℕ) : 
  sequence 1 = 3 →  -- June (1st month)
  sequence 3 = 6 →  -- August (3rd month)
  sequence 4 = 9 →  -- September (4th month)
  sequence 5 = 13 → -- October (5th month)
  sequence 2 = 6    -- July (2nd month)
:= by sorry

end NUMINAMATH_CALUDE_hammer_order_sequence_l2963_296345


namespace NUMINAMATH_CALUDE_range_of_f_l2963_296306

def f (x : ℤ) : ℤ := x^2 - 2*x

def domain : Set ℤ := {x : ℤ | -2 ≤ x ∧ x ≤ 4}

theorem range_of_f :
  {y : ℤ | ∃ x ∈ domain, f x = y} = {-1, 0, 3, 8} := by
  sorry

end NUMINAMATH_CALUDE_range_of_f_l2963_296306


namespace NUMINAMATH_CALUDE_markup_rate_proof_l2963_296377

theorem markup_rate_proof (S : ℝ) (h_positive : S > 0) : 
  let profit_rate : ℝ := 0.20
  let expense_rate : ℝ := 0.10
  let C : ℝ := S * (1 - profit_rate - expense_rate)
  ((S - C) / C) * 100 = 42.857 := by
sorry

end NUMINAMATH_CALUDE_markup_rate_proof_l2963_296377


namespace NUMINAMATH_CALUDE_rogers_nickels_l2963_296325

theorem rogers_nickels :
  ∀ (N : ℕ),
  (42 + N + 15 : ℕ) - 66 = 27 →
  N = 36 :=
by
  sorry

end NUMINAMATH_CALUDE_rogers_nickels_l2963_296325


namespace NUMINAMATH_CALUDE_problem_solution_l2963_296315

noncomputable def f (x : ℝ) := Real.exp x - Real.exp (-x) - 2 * x

noncomputable def g (b : ℝ) (x : ℝ) := f (2 * x) - 4 * b * f x

theorem problem_solution :
  (∀ x : ℝ, (deriv f) x ≥ 0) ∧
  (∃ b_max : ℝ, b_max = 2 ∧ ∀ b : ℝ, (∀ x : ℝ, x > 0 → g b x > 0) → b ≤ b_max) ∧
  (0.693 < Real.log 2 ∧ Real.log 2 < 0.694) :=
by sorry

end NUMINAMATH_CALUDE_problem_solution_l2963_296315


namespace NUMINAMATH_CALUDE_max_x_on_circle_l2963_296337

/-- The maximum x-coordinate of a point on the circle (x-10)^2 + (y-30)^2 = 100 is 20. -/
theorem max_x_on_circle : 
  ∀ x y : ℝ, (x - 10)^2 + (y - 30)^2 = 100 → x ≤ 20 :=
by sorry

end NUMINAMATH_CALUDE_max_x_on_circle_l2963_296337


namespace NUMINAMATH_CALUDE_subtraction_multiplication_equality_l2963_296376

theorem subtraction_multiplication_equality : 
  ((2000000000000 - 1111111111111) * 2) = 1777777777778 := by
  sorry

end NUMINAMATH_CALUDE_subtraction_multiplication_equality_l2963_296376


namespace NUMINAMATH_CALUDE_rectangular_prism_sum_l2963_296393

/-- A rectangular prism is a three-dimensional shape with 6 rectangular faces. -/
structure RectangularPrism where
  /-- The number of faces of a rectangular prism -/
  faces : ℕ
  /-- The number of edges of a rectangular prism -/
  edges : ℕ
  /-- The number of vertices of a rectangular prism -/
  vertices : ℕ
  /-- A rectangular prism has 6 faces -/
  face_count : faces = 6
  /-- A rectangular prism has 12 edges -/
  edge_count : edges = 12
  /-- A rectangular prism has 8 vertices -/
  vertex_count : vertices = 8

/-- The sum of faces, edges, and vertices of a rectangular prism is 26 -/
theorem rectangular_prism_sum (rp : RectangularPrism) : 
  rp.faces + rp.edges + rp.vertices = 26 := by
  sorry

end NUMINAMATH_CALUDE_rectangular_prism_sum_l2963_296393


namespace NUMINAMATH_CALUDE_success_rate_paradox_l2963_296321

structure Player :=
  (name : String)
  (attempts_season1 : ℕ)
  (successes_season1 : ℕ)
  (attempts_season2 : ℕ)
  (successes_season2 : ℕ)

def success_rate (attempts : ℕ) (successes : ℕ) : ℚ :=
  if attempts = 0 then 0 else (successes : ℚ) / (attempts : ℚ)

def combined_success_rate (p : Player) : ℚ :=
  success_rate (p.attempts_season1 + p.attempts_season2) (p.successes_season1 + p.successes_season2)

theorem success_rate_paradox (p1 p2 : Player) :
  (success_rate p1.attempts_season1 p1.successes_season1 > success_rate p2.attempts_season1 p2.successes_season1) ∧
  (success_rate p1.attempts_season2 p1.successes_season2 > success_rate p2.attempts_season2 p2.successes_season2) ∧
  (combined_success_rate p1 < combined_success_rate p2) :=
sorry

end NUMINAMATH_CALUDE_success_rate_paradox_l2963_296321


namespace NUMINAMATH_CALUDE_fixed_points_subset_stable_points_exists_function_with_infinite_stable_points_stable_points_are_fixed_points_for_increasing_functions_l2963_296309

-- Define the concept of a fixed point
def IsFixedPoint (f : ℝ → ℝ) (x : ℝ) : Prop := f x = x

-- Define the concept of a stable point
def IsStablePoint (f : ℝ → ℝ) (x : ℝ) : Prop := f (f x) = x

-- Define the set of fixed points
def FixedPoints (f : ℝ → ℝ) : Set ℝ := {x | IsFixedPoint f x}

-- Define the set of stable points
def StablePoints (f : ℝ → ℝ) : Set ℝ := {x | IsStablePoint f x}

-- Statement 1: Fixed points are a subset of stable points
theorem fixed_points_subset_stable_points (f : ℝ → ℝ) :
  FixedPoints f ⊆ StablePoints f := by sorry

-- Statement 2: There exists a function with infinitely many stable points
theorem exists_function_with_infinite_stable_points :
  ∃ f : ℝ → ℝ, ¬(Finite (StablePoints f)) := by sorry

-- Statement 3: For monotonically increasing functions, stable points are fixed points
theorem stable_points_are_fixed_points_for_increasing_functions
  (f : ℝ → ℝ) (h : ∀ x y, x < y → f x < f y) :
  ∀ x, IsStablePoint f x → IsFixedPoint f x := by sorry

end NUMINAMATH_CALUDE_fixed_points_subset_stable_points_exists_function_with_infinite_stable_points_stable_points_are_fixed_points_for_increasing_functions_l2963_296309


namespace NUMINAMATH_CALUDE_sum_of_four_consecutive_integers_divisible_by_two_l2963_296364

theorem sum_of_four_consecutive_integers_divisible_by_two (n : ℤ) :
  ∃ k : ℤ, (n - 1) + n + (n + 1) + (n + 2) = 2 * k :=
sorry

end NUMINAMATH_CALUDE_sum_of_four_consecutive_integers_divisible_by_two_l2963_296364


namespace NUMINAMATH_CALUDE_power_set_intersection_nonempty_l2963_296327

theorem power_set_intersection_nonempty :
  ∃ (A B : Set α), (A ∩ B).Nonempty ∧ (𝒫 A ∩ 𝒫 B).Nonempty :=
sorry

end NUMINAMATH_CALUDE_power_set_intersection_nonempty_l2963_296327


namespace NUMINAMATH_CALUDE_glove_pair_probability_l2963_296381

def num_black_pairs : ℕ := 6
def num_beige_pairs : ℕ := 4

def total_gloves : ℕ := 2 * (num_black_pairs + num_beige_pairs)

def prob_black_pair : ℚ := (num_black_pairs * 2 / total_gloves) * ((num_black_pairs * 2 - 1) / (total_gloves - 1))
def prob_beige_pair : ℚ := (num_beige_pairs * 2 / total_gloves) * ((num_beige_pairs * 2 - 1) / (total_gloves - 1))

theorem glove_pair_probability :
  prob_black_pair + prob_beige_pair = 47 / 95 := by
  sorry

end NUMINAMATH_CALUDE_glove_pair_probability_l2963_296381


namespace NUMINAMATH_CALUDE_max_m_value_l2963_296310

def f (x : ℝ) : ℝ := x^2 + 2*x + 1

theorem max_m_value : 
  (∃ (m : ℝ), m > 0 ∧ 
    (∃ (t : ℝ), ∀ (x : ℝ), x ∈ Set.Icc 1 m → f (x + t) ≤ x) ∧ 
    (∀ (m' : ℝ), m' > m → 
      ¬(∃ (t : ℝ), ∀ (x : ℝ), x ∈ Set.Icc 1 m' → f (x + t) ≤ x))) ∧
  (∀ (m : ℝ), 
    (∃ (t : ℝ), ∀ (x : ℝ), x ∈ Set.Icc 1 m → f (x + t) ≤ x) → 
    m ≤ 4) :=
sorry

end NUMINAMATH_CALUDE_max_m_value_l2963_296310


namespace NUMINAMATH_CALUDE_dress_making_hours_l2963_296348

def total_fabric : ℕ := 56
def fabric_per_dress : ℕ := 4
def hours_per_dress : ℕ := 3

theorem dress_making_hours : 
  (total_fabric / fabric_per_dress) * hours_per_dress = 42 := by
  sorry

end NUMINAMATH_CALUDE_dress_making_hours_l2963_296348


namespace NUMINAMATH_CALUDE_shooting_competition_score_l2963_296335

theorem shooting_competition_score 
  (team_size : ℕ) 
  (best_score : ℕ) 
  (hypothetical_best_score : ℕ) 
  (hypothetical_average : ℕ) 
  (h1 : team_size = 8)
  (h2 : best_score = 85)
  (h3 : hypothetical_best_score = 92)
  (h4 : hypothetical_average = 84)
  (h5 : hypothetical_average * team_size = 
        (hypothetical_best_score - best_score) + total_score) :
  total_score = 665 :=
by
  sorry

#check shooting_competition_score

end NUMINAMATH_CALUDE_shooting_competition_score_l2963_296335


namespace NUMINAMATH_CALUDE_license_plate_count_l2963_296308

/-- The number of letters in the alphabet -/
def num_letters : ℕ := 26

/-- The number of digits available -/
def num_digits : ℕ := 10

/-- The total number of characters available (letters + digits) -/
def num_chars : ℕ := num_letters + num_digits

/-- The format of the license plate -/
inductive LicensePlateChar
| Letter
| Digit
| Any

/-- The structure of the license plate -/
def license_plate_format : List LicensePlateChar :=
  [LicensePlateChar.Letter, LicensePlateChar.Digit, LicensePlateChar.Any, LicensePlateChar.Digit]

/-- 
  The number of ways to create a 4-character license plate 
  where the format is a letter followed by a digit, then any character, and ending with a digit,
  ensuring that exactly two characters on the license plate are the same.
-/
theorem license_plate_count : 
  (num_letters * num_digits * num_chars) = 9360 := by
  sorry

end NUMINAMATH_CALUDE_license_plate_count_l2963_296308


namespace NUMINAMATH_CALUDE_intersection_of_M_and_N_l2963_296379

-- Define the sets M and N
def M : Set ℝ := {x | -2 ≤ x ∧ x ≤ 2}
def N : Set ℝ := {x | x < 1}

-- State the theorem
theorem intersection_of_M_and_N :
  M ∩ N = {x : ℝ | -2 ≤ x ∧ x < 1} := by sorry

end NUMINAMATH_CALUDE_intersection_of_M_and_N_l2963_296379


namespace NUMINAMATH_CALUDE_complex_equation_solution_l2963_296313

theorem complex_equation_solution (z : ℂ) : z * Complex.I = 2 / (1 + Complex.I) → z = -1 - Complex.I := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l2963_296313


namespace NUMINAMATH_CALUDE_even_function_shift_l2963_296340

/-- Given a function f and a real number a, proves that if f(x) = 3sin(2x - π/3) 
    and y = f(x + a) is an even function where 0 < a < π/2, then a = 5π/12 -/
theorem even_function_shift (f : ℝ → ℝ) (a : ℝ) : 
  (∀ x, f x = 3 * Real.sin (2 * x - π / 3)) →
  (∀ x, f (x + a) = f (-x + a)) →
  (0 < a) →
  (a < π / 2) →
  a = 5 * π / 12 := by
sorry

end NUMINAMATH_CALUDE_even_function_shift_l2963_296340


namespace NUMINAMATH_CALUDE_isosceles_triangle_base_length_l2963_296344

/-- An isosceles triangle with integer side lengths and perimeter 10 has a base length of 2 or 4 -/
theorem isosceles_triangle_base_length : 
  ∀ x y : ℕ, 
  x > 0 → y > 0 →
  x + x + y = 10 → 
  y = 2 ∨ y = 4 :=
by sorry

end NUMINAMATH_CALUDE_isosceles_triangle_base_length_l2963_296344


namespace NUMINAMATH_CALUDE_rain_both_days_l2963_296382

-- Define the probabilities
def prob_rain_monday : ℝ := 0.62
def prob_rain_tuesday : ℝ := 0.54
def prob_no_rain : ℝ := 0.28

-- Theorem statement
theorem rain_both_days :
  let prob_rain_both := prob_rain_monday + prob_rain_tuesday - (1 - prob_no_rain)
  prob_rain_both = 0.44 := by sorry

end NUMINAMATH_CALUDE_rain_both_days_l2963_296382


namespace NUMINAMATH_CALUDE_cube_edge_sum_length_l2963_296392

/-- The sum of the lengths of all edges of a cube with edge length 15 cm is 180 cm. -/
theorem cube_edge_sum_length (edge_length : ℝ) (num_edges : ℕ) : 
  edge_length = 15 → num_edges = 12 → edge_length * num_edges = 180 := by
  sorry

end NUMINAMATH_CALUDE_cube_edge_sum_length_l2963_296392


namespace NUMINAMATH_CALUDE_kitty_cleanup_time_l2963_296346

/-- Represents the weekly cleaning routine for a living room -/
structure CleaningRoutine where
  pickup_time : ℕ  -- Time spent picking up toys and straightening
  vacuum_time : ℕ  -- Time spent vacuuming
  window_time : ℕ  -- Time spent cleaning windows
  dusting_time : ℕ  -- Time spent dusting furniture

/-- Calculates the total cleaning time for a given number of weeks -/
def total_cleaning_time (routine : CleaningRoutine) (weeks : ℕ) : ℕ :=
  weeks * (routine.pickup_time + routine.vacuum_time + routine.window_time + routine.dusting_time)

theorem kitty_cleanup_time :
  ∃ (routine : CleaningRoutine),
    routine.vacuum_time = 20 ∧
    routine.window_time = 15 ∧
    routine.dusting_time = 10 ∧
    total_cleaning_time routine 4 = 200 ∧
    routine.pickup_time = 5 := by
  sorry

end NUMINAMATH_CALUDE_kitty_cleanup_time_l2963_296346


namespace NUMINAMATH_CALUDE_population_difference_l2963_296383

/-- Given that the sum of populations of City A and City B exceeds the sum of populations
    of City B and City C by 5000, prove that the population of City A exceeds
    the population of City C by 5000. -/
theorem population_difference (A B C : ℕ) 
  (h : A + B = B + C + 5000) : A - C = 5000 := by
  sorry

end NUMINAMATH_CALUDE_population_difference_l2963_296383


namespace NUMINAMATH_CALUDE_customs_duration_l2963_296334

theorem customs_duration (navigation_time transport_time total_time : ℕ) 
  (h1 : navigation_time = 21)
  (h2 : transport_time = 7)
  (h3 : total_time = 30) :
  total_time - navigation_time - transport_time = 2 := by
  sorry

end NUMINAMATH_CALUDE_customs_duration_l2963_296334


namespace NUMINAMATH_CALUDE_intersection_points_l2963_296368

-- Define the quadratic and linear functions
def quadratic (a b c x : ℝ) : ℝ := a * x^2 + b * x + c
def linear (s t x : ℝ) : ℝ := s * x + t

-- Define the discriminant
def discriminant (a b c s t : ℝ) : ℝ := (b - s)^2 - 4 * a * (c - t)

-- Theorem statement
theorem intersection_points (a b c s t : ℝ) (ha : a ≠ 0) (hs : s ≠ 0) :
  let Δ := discriminant a b c s t
  -- Two intersection points when Δ > 0
  (Δ > 0 → ∃ x₁ x₂, x₁ ≠ x₂ ∧ quadratic a b c x₁ = linear s t x₁ ∧ quadratic a b c x₂ = linear s t x₂) ∧
  -- One intersection point when Δ = 0
  (Δ = 0 → ∃! x, quadratic a b c x = linear s t x) ∧
  -- No intersection points when Δ < 0
  (Δ < 0 → ∀ x, quadratic a b c x ≠ linear s t x) :=
by sorry

end NUMINAMATH_CALUDE_intersection_points_l2963_296368


namespace NUMINAMATH_CALUDE_regular_polygon_sides_l2963_296391

theorem regular_polygon_sides (D : ℕ) : D = 12 → ∃ n : ℕ, n = 6 ∧ D = n * (n - 3) / 2 := by
  sorry

end NUMINAMATH_CALUDE_regular_polygon_sides_l2963_296391


namespace NUMINAMATH_CALUDE_min_value_theorem_l2963_296385

theorem min_value_theorem (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : 1/x + 1/y = 1) :
  1/(x-1) + 3/(y-1) ≥ 2 * Real.sqrt 3 :=
by sorry

end NUMINAMATH_CALUDE_min_value_theorem_l2963_296385


namespace NUMINAMATH_CALUDE_voucher_distribution_l2963_296331

-- Define the number of representatives and vouchers
def num_representatives : ℕ := 5
def num_vouchers : ℕ := 4

-- Define the distribution method
def distribution_method (n m : ℕ) : ℕ := Nat.choose n m

-- Theorem statement
theorem voucher_distribution :
  distribution_method num_representatives num_vouchers = 5 := by
  sorry

end NUMINAMATH_CALUDE_voucher_distribution_l2963_296331


namespace NUMINAMATH_CALUDE_hyperbola_foci_distance_l2963_296343

/-- The distance between the foci of a hyperbola with equation xy = 4 is 8 -/
theorem hyperbola_foci_distance :
  ∃ (t : ℝ), t > 0 ∧
  (∀ (x y : ℝ), x * y = 4 →
    ∃ (d : ℝ), d > 0 ∧
    ∀ (P : ℝ × ℝ), P.1 * P.2 = 4 →
      Real.sqrt ((P.1 + t)^2 + (P.2 + t)^2) - Real.sqrt ((P.1 - t)^2 + (P.2 - t)^2) = d) →
  Real.sqrt ((t + t)^2 + (t + t)^2) = 8 :=
by sorry


end NUMINAMATH_CALUDE_hyperbola_foci_distance_l2963_296343


namespace NUMINAMATH_CALUDE_quadratic_function_k_value_l2963_296303

theorem quadratic_function_k_value (a b c k : ℤ) :
  let g := λ (x : ℤ) => a * x^2 + b * x + c
  (g 2 = 0) →
  (110 < g 9) →
  (g 9 < 120) →
  (130 < g 10) →
  (g 10 < 140) →
  (6000 * k < g 100) →
  (g 100 < 6000 * (k + 1)) →
  k = 1 := by sorry

end NUMINAMATH_CALUDE_quadratic_function_k_value_l2963_296303


namespace NUMINAMATH_CALUDE_range_of_a_l2963_296352

def p (x a : ℝ) : Prop := |x - a| < 4
def q (x : ℝ) : Prop := (x - 2) * (3 - x) > 0

theorem range_of_a : 
  (∀ x a : ℝ, (¬(p x a) → ¬(q x)) ∧ ∃ x, q x ∧ p x a) → 
  ∃ a : ℝ, -1 ≤ a ∧ a ≤ 6 :=
sorry

end NUMINAMATH_CALUDE_range_of_a_l2963_296352


namespace NUMINAMATH_CALUDE_sheila_weekly_earnings_l2963_296378

/-- Represents Sheila's work schedule and earnings --/
structure WorkSchedule where
  hours_mon_wed_fri : ℕ
  hours_tue_thu : ℕ
  hourly_rate : ℕ

/-- Calculates the total weekly hours worked --/
def total_weekly_hours (schedule : WorkSchedule) : ℕ :=
  3 * schedule.hours_mon_wed_fri + 2 * schedule.hours_tue_thu

/-- Calculates the weekly earnings --/
def weekly_earnings (schedule : WorkSchedule) : ℕ :=
  (total_weekly_hours schedule) * schedule.hourly_rate

/-- Theorem stating Sheila's weekly earnings --/
theorem sheila_weekly_earnings :
  ∀ (schedule : WorkSchedule),
  schedule.hours_mon_wed_fri = 8 →
  schedule.hours_tue_thu = 6 →
  schedule.hourly_rate = 10 →
  weekly_earnings schedule = 360 := by
  sorry


end NUMINAMATH_CALUDE_sheila_weekly_earnings_l2963_296378


namespace NUMINAMATH_CALUDE_distance_point_to_line_polar_l2963_296359

/-- The distance between a point in polar coordinates and a line given by a polar equation -/
theorem distance_point_to_line_polar (ρ_A θ_A : ℝ) :
  let r := 2 * ρ_A * Real.sin (θ_A - π / 4) - Real.sqrt 2
  let x := ρ_A * Real.cos θ_A
  let y := ρ_A * Real.sin θ_A
  ρ_A = 2 * Real.sqrt 2 ∧ θ_A = 7 * π / 4 →
  (r^2 / (1 + 1)) = (5 * Real.sqrt 2 / 2)^2 := by
sorry

end NUMINAMATH_CALUDE_distance_point_to_line_polar_l2963_296359


namespace NUMINAMATH_CALUDE_sum_remainder_l2963_296319

theorem sum_remainder (a b c d e : ℕ) 
  (ha : a % 13 = 3)
  (hb : b % 13 = 5)
  (hc : c % 13 = 7)
  (hd : d % 13 = 9)
  (he : e % 13 = 11) :
  (a + b + c + d + e) % 13 = 9 := by
  sorry

end NUMINAMATH_CALUDE_sum_remainder_l2963_296319


namespace NUMINAMATH_CALUDE_sin_product_equals_one_eighth_l2963_296351

theorem sin_product_equals_one_eighth : 
  Real.sin (12 * Real.pi / 180) * Real.sin (36 * Real.pi / 180) * 
  Real.sin (54 * Real.pi / 180) * Real.sin (72 * Real.pi / 180) = 1/8 := by
  sorry

end NUMINAMATH_CALUDE_sin_product_equals_one_eighth_l2963_296351


namespace NUMINAMATH_CALUDE_divisibility_by_1001_l2963_296342

theorem divisibility_by_1001 (n : ℤ) : n ≡ 300^3000 [ZMOD 1001] → n ≡ 1 [ZMOD 1001] := by
  sorry

end NUMINAMATH_CALUDE_divisibility_by_1001_l2963_296342


namespace NUMINAMATH_CALUDE_probability_defective_from_A_l2963_296384

-- Define the probabilities
def prob_factory_A : ℝ := 0.45
def prob_factory_B : ℝ := 0.55
def defect_rate_A : ℝ := 0.06
def defect_rate_B : ℝ := 0.05

-- Theorem statement
theorem probability_defective_from_A : 
  let prob_defective := prob_factory_A * defect_rate_A + prob_factory_B * defect_rate_B
  prob_factory_A * defect_rate_A / prob_defective = 54 / 109 := by
sorry

end NUMINAMATH_CALUDE_probability_defective_from_A_l2963_296384


namespace NUMINAMATH_CALUDE_complex_equation_solution_l2963_296356

theorem complex_equation_solution (z : ℂ) : (Complex.I * z = 4 + 3 * Complex.I) → z = 3 - 4 * Complex.I := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l2963_296356


namespace NUMINAMATH_CALUDE_parabola_intersections_and_point_position_l2963_296316

/-- Represents a parabola of the form y = x^2 + px + q -/
structure Parabola where
  p : ℝ
  q : ℝ

/-- A point on the coordinate plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- Theorem about parabola intersections and point position -/
theorem parabola_intersections_and_point_position 
  (parabola : Parabola) 
  (M : Point) 
  (h_below_x_axis : M.y < 0) :
  ∃ (x₁ x₂ : ℝ), 
    (x₁^2 + parabola.p * x₁ + parabola.q = 0) ∧ 
    (x₂^2 + parabola.p * x₂ + parabola.q = 0) ∧ 
    (x₁ < x₂) ∧
    (x₁ < M.x) ∧ (M.x < x₂) := by
  sorry


end NUMINAMATH_CALUDE_parabola_intersections_and_point_position_l2963_296316


namespace NUMINAMATH_CALUDE_cubic_inequality_l2963_296367

theorem cubic_inequality (x : ℝ) : (x^3 - 125) / (x + 3) < 0 ↔ -3 < x ∧ x < 5 :=
by sorry

end NUMINAMATH_CALUDE_cubic_inequality_l2963_296367


namespace NUMINAMATH_CALUDE_sin_ratio_comparison_l2963_296396

theorem sin_ratio_comparison :
  (Real.sin (3 * Real.pi / 180)) / (Real.sin (4 * Real.pi / 180)) >
  (Real.sin (1 * Real.pi / 180)) / (Real.sin (2 * Real.pi / 180)) :=
by sorry

end NUMINAMATH_CALUDE_sin_ratio_comparison_l2963_296396


namespace NUMINAMATH_CALUDE_ellipse_k_range_l2963_296336

/-- The equation of an ellipse with parameter k -/
def ellipse_equation (x y k : ℝ) : Prop :=
  x^2 / (5 - k) + y^2 / (k - 3) = 1

/-- Conditions for the equation to represent an ellipse -/
def is_ellipse (k : ℝ) : Prop :=
  5 - k > 0 ∧ k - 3 > 0 ∧ 5 - k ≠ k - 3

/-- The range of k for which the equation represents an ellipse -/
theorem ellipse_k_range :
  ∀ k : ℝ, is_ellipse k ↔ (3 < k ∧ k < 5 ∧ k ≠ 4) :=
by sorry

end NUMINAMATH_CALUDE_ellipse_k_range_l2963_296336


namespace NUMINAMATH_CALUDE_square_area_difference_l2963_296333

theorem square_area_difference (x : ℝ) : 
  (x + 2)^2 - x^2 = 32 → x + 2 = 9 :=
by sorry

end NUMINAMATH_CALUDE_square_area_difference_l2963_296333


namespace NUMINAMATH_CALUDE_town_population_problem_l2963_296317

theorem town_population_problem (original_population : ℕ) : 
  (((original_population + 1500) * 85 / 100 : ℕ) = original_population - 45) → 
  original_population = 8800 := by
  sorry

end NUMINAMATH_CALUDE_town_population_problem_l2963_296317


namespace NUMINAMATH_CALUDE_alchemerion_age_ratio_l2963_296358

/-- Represents the ages of Alchemerion, his son, and his father -/
structure WizardFamily where
  alchemerion : ℕ
  son : ℕ
  father : ℕ

/-- Defines the properties of the Wizard family's ages -/
def is_valid_wizard_family (f : WizardFamily) : Prop :=
  f.alchemerion = 360 ∧
  f.father = 2 * f.alchemerion + 40 ∧
  f.alchemerion + f.son + f.father = 1240 ∧
  ∃ k : ℕ, f.alchemerion = k * f.son

/-- Theorem stating that Alchemerion is 3 times older than his son -/
theorem alchemerion_age_ratio (f : WizardFamily) 
  (h : is_valid_wizard_family f) : 
  f.alchemerion = 3 * f.son :=
sorry

end NUMINAMATH_CALUDE_alchemerion_age_ratio_l2963_296358


namespace NUMINAMATH_CALUDE_complex_ratio_l2963_296338

theorem complex_ratio (z₁ z₂ : ℂ) (h1 : Complex.abs z₁ = 1) (h2 : Complex.abs z₂ = 5/2) 
  (h3 : Complex.abs (3 * z₁ - 2 * z₂) = 7) :
  z₁ / z₂ = -1/5 * (1 - Complex.I * Real.sqrt 3) ∨ z₁ / z₂ = -1/5 * (1 + Complex.I * Real.sqrt 3) := by
sorry

end NUMINAMATH_CALUDE_complex_ratio_l2963_296338


namespace NUMINAMATH_CALUDE_expression_equality_l2963_296365

theorem expression_equality : 
  (4 + 5) * (4^2 + 5^2) * (4^4 + 5^4) * (4^8 + 5^8) * (4^16 + 5^16) * (4^32 + 5^32) * (4^64 + 5^64) = 5^128 - 4^128 := by
  sorry

end NUMINAMATH_CALUDE_expression_equality_l2963_296365


namespace NUMINAMATH_CALUDE_vertex_on_parabola_and_line_intersection_l2963_296347

/-- The quadratic function -/
def f (m x : ℝ) : ℝ := x^2 + 2*(m + 1)*x - m + 1

/-- The vertex of the quadratic function -/
def vertex (m : ℝ) : ℝ × ℝ := (-m - 1, -m^2 - 3*m)

/-- The parabola on which the vertex lies -/
def parabola (x : ℝ) : ℝ := -x^2 + x + 2

/-- The line that may pass through the vertex -/
def line (x : ℝ) : ℝ := x + 1

theorem vertex_on_parabola_and_line_intersection (m : ℝ) :
  (∀ m, parabola (vertex m).1 = (vertex m).2) ∧
  (line (vertex m).1 = (vertex m).2 ↔ m = -2 ∨ m = 0) :=
by sorry

end NUMINAMATH_CALUDE_vertex_on_parabola_and_line_intersection_l2963_296347


namespace NUMINAMATH_CALUDE_rest_area_location_l2963_296300

theorem rest_area_location (city_a city_b rest_area : ℝ) : 
  city_a = 50 →
  city_b = 230 →
  rest_area - city_a = (5/8) * (city_b - city_a) →
  rest_area = 162.5 := by
sorry

end NUMINAMATH_CALUDE_rest_area_location_l2963_296300


namespace NUMINAMATH_CALUDE_coin_division_problem_l2963_296370

theorem coin_division_problem :
  ∃ n : ℕ, 
    n > 0 ∧
    n % 8 = 5 ∧
    n % 7 = 4 ∧
    (∀ m : ℕ, m > 0 → m % 8 = 5 → m % 7 = 4 → n ≤ m) ∧
    n % 9 = 8 :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_CALUDE_coin_division_problem_l2963_296370


namespace NUMINAMATH_CALUDE_square_root_product_plus_one_l2963_296307

theorem square_root_product_plus_one (a : ℕ) (n : ℕ) : 
  a = 2020 ∧ n = 4086461 → a * (a + 1) * (a + 2) * (a + 3) + 1 = n^2 := by
  sorry

end NUMINAMATH_CALUDE_square_root_product_plus_one_l2963_296307


namespace NUMINAMATH_CALUDE_parallel_vectors_dot_product_l2963_296311

/-- Given two vectors a and b in ℝ², if a is parallel to b and a = (1,3) and b = (-3,x), 
    then their dot product is -30 -/
theorem parallel_vectors_dot_product (x : ℝ) : 
  let a : ℝ × ℝ := (1, 3)
  let b : ℝ × ℝ := (-3, x)
  (∃ (k : ℝ), b = k • a) → a.1 * b.1 + a.2 * b.2 = -30 := by
  sorry

end NUMINAMATH_CALUDE_parallel_vectors_dot_product_l2963_296311


namespace NUMINAMATH_CALUDE_diagonals_30_sided_polygon_l2963_296339

/-- The number of diagonals in a convex polygon with n sides -/
def num_diagonals (n : ℕ) : ℕ := n * (n - 3) / 2

/-- Theorem: A convex polygon with 30 sides has 405 diagonals -/
theorem diagonals_30_sided_polygon : num_diagonals 30 = 405 := by
  sorry

end NUMINAMATH_CALUDE_diagonals_30_sided_polygon_l2963_296339


namespace NUMINAMATH_CALUDE_system_nonzero_solution_iff_condition_l2963_296357

/-- The system of equations has a non-zero solution iff 2abc + ab + bc + ca - 1 = 0 -/
theorem system_nonzero_solution_iff_condition (a b c : ℝ) :
  (∃ x y z : ℝ, (x ≠ 0 ∨ y ≠ 0 ∨ z ≠ 0) ∧
    x = b * y + c * z ∧
    y = c * z + a * x ∧
    z = a * x + b * y) ↔
  2 * a * b * c + a * b + b * c + c * a - 1 = 0 := by
sorry

end NUMINAMATH_CALUDE_system_nonzero_solution_iff_condition_l2963_296357


namespace NUMINAMATH_CALUDE_tan_product_lower_bound_l2963_296371

theorem tan_product_lower_bound (α β γ : Real) 
  (h_acute_α : 0 < α ∧ α < Real.pi / 2)
  (h_acute_β : 0 < β ∧ β < Real.pi / 2)
  (h_acute_γ : 0 < γ ∧ γ < Real.pi / 2)
  (h_cos_sum : Real.cos α ^ 2 + Real.cos β ^ 2 + Real.cos γ ^ 2 = 1) :
  Real.tan α * Real.tan β * Real.tan γ ≥ 2 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_tan_product_lower_bound_l2963_296371


namespace NUMINAMATH_CALUDE_diary_ratio_proof_l2963_296386

def diary_problem (initial_diaries : ℕ) (current_diaries : ℕ) : Prop :=
  let bought_diaries := 2 * initial_diaries
  let total_after_buying := initial_diaries + bought_diaries
  let lost_diaries := total_after_buying - current_diaries
  (lost_diaries : ℚ) / total_after_buying = 1 / 4

theorem diary_ratio_proof :
  diary_problem 8 18 := by
  sorry

end NUMINAMATH_CALUDE_diary_ratio_proof_l2963_296386


namespace NUMINAMATH_CALUDE_intersection_A_notB_C_subset_A_implies_a_range_l2963_296390

-- Define the sets A, B, and C
def A : Set ℝ := {x | -3 < x ∧ x < 4}
def B : Set ℝ := {x | x^2 + 2*x - 8 > 0}
def C (a : ℝ) : Set ℝ := {x | x^2 - 4*a*x + 3*a^2 < 0}

-- Define the complement of B in ℝ
def notB : Set ℝ := {x | ¬(x ∈ B)}

-- Theorem for part (I)
theorem intersection_A_notB : A ∩ notB = {x : ℝ | -3 < x ∧ x ≤ 2} := by sorry

-- Theorem for part (II)
theorem C_subset_A_implies_a_range (a : ℝ) (h : a ≠ 0) :
  C a ⊆ A → (-1 ≤ a ∧ a < 0) ∨ (0 < a ∧ a ≤ 4/3) := by sorry

end NUMINAMATH_CALUDE_intersection_A_notB_C_subset_A_implies_a_range_l2963_296390


namespace NUMINAMATH_CALUDE_range_of_a_l2963_296323

-- Define the propositions p and q
def p (x : ℝ) : Prop := (4*x - 3)^2 ≤ 1
def q (x a : ℝ) : Prop := x^2 - (2*a + 1)*x + a*(a + 1) ≤ 0

-- Define the sets A and B
def A : Set ℝ := {x | p x}
def B (a : ℝ) : Set ℝ := {x | q x a}

-- State the theorem
theorem range_of_a :
  (∀ a : ℝ, (∀ x : ℝ, ¬(p x) → ¬(q x a)) ∧ 
  (∃ x : ℝ, ¬(p x) ∧ q x a)) →
  (∃ S : Set ℝ, S = {a : ℝ | 0 ≤ a ∧ a ≤ 1/2} ∧ 
  (∀ a : ℝ, a ∈ S ↔ 
    (∀ x : ℝ, x ∈ A → x ∈ B a) ∧ 
    (∃ x : ℝ, x ∉ A ∧ x ∈ B a))) :=
sorry

end NUMINAMATH_CALUDE_range_of_a_l2963_296323


namespace NUMINAMATH_CALUDE_dogs_adopted_twenty_dogs_adopted_l2963_296363

/-- The number of dogs adopted from a pet center --/
theorem dogs_adopted (initial_dogs : ℕ) (initial_cats : ℕ) (additional_cats : ℕ) (final_total : ℕ) : ℕ :=
  let remaining_dogs := initial_dogs - (initial_dogs + initial_cats + additional_cats - final_total)
  initial_dogs - remaining_dogs

/-- Proof that 20 dogs were adopted given the problem conditions --/
theorem twenty_dogs_adopted : dogs_adopted 36 29 12 57 = 20 := by
  sorry

end NUMINAMATH_CALUDE_dogs_adopted_twenty_dogs_adopted_l2963_296363


namespace NUMINAMATH_CALUDE_profit_percentage_is_fifty_percent_l2963_296329

/-- Calculates the profit percentage given the costs and selling price -/
def profit_percentage (purchase_price repair_cost transport_cost selling_price : ℚ) : ℚ :=
  let total_cost := purchase_price + repair_cost + transport_cost
  let profit := selling_price - total_cost
  (profit / total_cost) * 100

/-- Theorem: The profit percentage is 50% given the specific costs and selling price -/
theorem profit_percentage_is_fifty_percent :
  profit_percentage 10000 5000 1000 24000 = 50 := by
  sorry

end NUMINAMATH_CALUDE_profit_percentage_is_fifty_percent_l2963_296329


namespace NUMINAMATH_CALUDE_unique_congruence_l2963_296362

theorem unique_congruence (n : ℤ) : 3 ≤ n ∧ n ≤ 11 ∧ n ≡ 2023 [ZMOD 7] → n = 7 := by
  sorry

end NUMINAMATH_CALUDE_unique_congruence_l2963_296362


namespace NUMINAMATH_CALUDE_timothy_speed_l2963_296387

theorem timothy_speed (mother_speed : ℝ) (distance : ℝ) (head_start : ℝ) :
  mother_speed = 36 →
  distance = 1.8 →
  head_start = 0.25 →
  let mother_time : ℝ := distance / mother_speed
  let total_time : ℝ := mother_time + head_start
  let timothy_speed : ℝ := distance / total_time
  timothy_speed = 6 := by sorry

end NUMINAMATH_CALUDE_timothy_speed_l2963_296387


namespace NUMINAMATH_CALUDE_exists_number_divisible_by_5_1000_without_zeros_l2963_296312

theorem exists_number_divisible_by_5_1000_without_zeros : 
  ∃ n : ℕ, (5^1000 ∣ n) ∧ (∀ d : ℕ, d < 10 → d ≠ 0 → ∃ k : ℕ, n / 10^k % 10 = d) :=
sorry

end NUMINAMATH_CALUDE_exists_number_divisible_by_5_1000_without_zeros_l2963_296312


namespace NUMINAMATH_CALUDE_womens_tennis_handshakes_l2963_296360

theorem womens_tennis_handshakes (n : ℕ) (k : ℕ) (h1 : n = 4) (h2 : k = 2) : 
  (n * k * (n * k - k)) / 2 = 24 := by
  sorry

end NUMINAMATH_CALUDE_womens_tennis_handshakes_l2963_296360


namespace NUMINAMATH_CALUDE_ant_movement_probability_l2963_296305

structure Octahedron where
  middleVertices : Finset Nat
  topVertex : Nat
  bottomVertex : Nat

def moveToMiddle (o : Octahedron) (start : Nat) : Finset Nat :=
  o.middleVertices.filter (λ v => v ≠ start)

def moveFromMiddle (o : Octahedron) (middle : Nat) : Finset Nat :=
  insert o.bottomVertex (insert o.topVertex (o.middleVertices.filter (λ v => v ≠ middle)))

theorem ant_movement_probability (o : Octahedron) (start : Nat) :
  start ∈ o.middleVertices →
  (1 : ℚ) / 4 = (moveToMiddle o start).sum (λ a =>
    (1 : ℚ) / (moveToMiddle o start).card *
    (1 : ℚ) / (moveFromMiddle o a).card *
    if o.bottomVertex ∈ moveFromMiddle o a then 1 else 0) :=
sorry

end NUMINAMATH_CALUDE_ant_movement_probability_l2963_296305


namespace NUMINAMATH_CALUDE_cyclic_inequality_l2963_296349

theorem cyclic_inequality (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  (a^2 + a*b + b^2) * (b^2 + b*c + c^2) * (c^2 + c*a + a^2) ≥ (a*b + b*c + c*a)^2 := by
  sorry

end NUMINAMATH_CALUDE_cyclic_inequality_l2963_296349


namespace NUMINAMATH_CALUDE_hyperbola_equation_l2963_296324

/-- The standard equation of a hyperbola with given eccentricity and focus -/
theorem hyperbola_equation (e : ℝ) (f : ℝ × ℝ) :
  e = 5/3 →
  f = (0, 5) →
  ∃ (a b : ℝ), a > 0 ∧ b > 0 ∧
    (∀ (x y : ℝ), y^2/a^2 - x^2/b^2 = 1 ↔ y^2/9 - x^2/16 = 1) :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_equation_l2963_296324


namespace NUMINAMATH_CALUDE_problem_statement_l2963_296369

theorem problem_statement (a b : ℝ) (h : |a + 5| + (b - 2)^2 = 0) :
  (a + b)^2010 = 3^2010 := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l2963_296369


namespace NUMINAMATH_CALUDE_line_m_equation_l2963_296326

-- Define the xy-plane
def xy_plane : Set (ℝ × ℝ) := Set.univ

-- Define lines ℓ and m
def line_ℓ : Set (ℝ × ℝ) := {p : ℝ × ℝ | 3 * p.1 + 4 * p.2 = 0}
def line_m : Set (ℝ × ℝ) := {p : ℝ × ℝ | 7 * p.1 - p.2 = 0}

-- Define points
def Q : ℝ × ℝ := (-3, 2)
def Q'' : ℝ × ℝ := (-4, -3)

-- Define the reflection operation (as a placeholder, actual implementation not provided)
def reflect (point : ℝ × ℝ) (line : Set (ℝ × ℝ)) : ℝ × ℝ := sorry

theorem line_m_equation :
  line_ℓ ⊆ xy_plane ∧
  line_m ⊆ xy_plane ∧
  line_ℓ ≠ line_m ∧
  (0, 0) ∈ line_ℓ ∩ line_m ∧
  Q ∈ xy_plane ∧
  Q'' ∈ xy_plane ∧
  reflect (reflect Q line_ℓ) line_m = Q'' →
  line_m = {p : ℝ × ℝ | 7 * p.1 - p.2 = 0} :=
by sorry

end NUMINAMATH_CALUDE_line_m_equation_l2963_296326


namespace NUMINAMATH_CALUDE_matrix_not_invertible_sum_fractions_l2963_296372

theorem matrix_not_invertible_sum_fractions (a b c : ℝ) :
  let M := !![a, b, c; b, c, a; c, a, b]
  ¬(IsUnit (Matrix.det M)) →
  (a / (b + c) + b / (a + c) + c / (a + b) = -3) ∨
  (a / (b + c) + b / (a + c) + c / (a + b) = 3/2) :=
by sorry

end NUMINAMATH_CALUDE_matrix_not_invertible_sum_fractions_l2963_296372


namespace NUMINAMATH_CALUDE_polynomial_factorization_l2963_296399

theorem polynomial_factorization (x : ℝ) :
  (x^2 + 4*x + 3) * (x^2 + 8*x + 15) + (x^2 + 6*x - 8) = 
  (x^2 + 6*x + 12) * (x^2 + 6*x + 3) := by
  sorry

end NUMINAMATH_CALUDE_polynomial_factorization_l2963_296399


namespace NUMINAMATH_CALUDE_equation_solutions_l2963_296318

theorem equation_solutions :
  (∃ x : ℚ, 4 * (x + 3) = 25 ∧ x = 13 / 4) ∧
  (∃ x₁ x₂ : ℚ, 5 * x₁^2 - 3 * x₁ = x₁ + 1 ∧ x₁ = -1 / 5 ∧
               5 * x₂^2 - 3 * x₂ = x₂ + 1 ∧ x₂ = 1) ∧
  (∃ x₁ x₂ : ℚ, 2 * (x₁ - 2)^2 - (x₁ - 2) = 0 ∧ x₁ = 2 ∧
               2 * (x₂ - 2)^2 - (x₂ - 2) = 0 ∧ x₂ = 5 / 2) :=
by sorry

end NUMINAMATH_CALUDE_equation_solutions_l2963_296318


namespace NUMINAMATH_CALUDE_company_french_speakers_l2963_296361

theorem company_french_speakers 
  (total_employees : ℝ) 
  (total_employees_positive : 0 < total_employees) :
  let men_percentage : ℝ := 65 / 100
  let women_percentage : ℝ := 1 - men_percentage
  let men_french_speakers_percentage : ℝ := 60 / 100
  let women_non_french_speakers_percentage : ℝ := 97.14285714285714 / 100
  let men_count : ℝ := men_percentage * total_employees
  let women_count : ℝ := women_percentage * total_employees
  let men_french_speakers : ℝ := men_french_speakers_percentage * men_count
  let women_french_speakers : ℝ := (1 - women_non_french_speakers_percentage) * women_count
  let total_french_speakers : ℝ := men_french_speakers + women_french_speakers
  let french_speakers_percentage : ℝ := total_french_speakers / total_employees * 100
  french_speakers_percentage = 40 := by
sorry


end NUMINAMATH_CALUDE_company_french_speakers_l2963_296361


namespace NUMINAMATH_CALUDE_max_triangle_area_is_85_l2963_296398

/-- Point in 2D plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- Line in 2D plane -/
structure Line where
  slope : ℝ
  point : Point

/-- Triangle formed by three lines -/
structure Triangle where
  l1 : Line
  l2 : Line
  l3 : Line

/-- Rotation of a line around its point -/
def rotate (l : Line) (angle : ℝ) : Line :=
  sorry

/-- Area of a triangle formed by three lines -/
def triangleArea (t : Triangle) : ℝ :=
  sorry

/-- Maximum area of triangle formed by rotating lines -/
def maxTriangleArea (l1 l2 l3 : Line) : ℝ :=
  sorry

theorem max_triangle_area_is_85 :
  let a := Point.mk 0 0
  let b := Point.mk 11 0
  let c := Point.mk 18 0
  let la := Line.mk 1 a
  let lb := Line.mk 0 b  -- Vertical line represented with slope 0
  let lc := Line.mk (-1) c
  maxTriangleArea la lb lc = 85 := by
  sorry

end NUMINAMATH_CALUDE_max_triangle_area_is_85_l2963_296398


namespace NUMINAMATH_CALUDE_geometric_series_ratio_l2963_296304

theorem geometric_series_ratio (a r : ℝ) (hr : r ≠ 1) (ha : a ≠ 0) :
  (a / (1 - r) = 81 * (a * r^4) / (1 - r)) → r = 1/3 := by
sorry

end NUMINAMATH_CALUDE_geometric_series_ratio_l2963_296304


namespace NUMINAMATH_CALUDE_difference_of_squares_103_97_l2963_296394

theorem difference_of_squares_103_97 : 
  |((103 : ℚ) / 2)^2 - ((97 : ℚ) / 2)^2| = 300 := by
  sorry

end NUMINAMATH_CALUDE_difference_of_squares_103_97_l2963_296394


namespace NUMINAMATH_CALUDE_savings_in_cents_l2963_296353

/-- The in-store price of the appliance in dollars -/
def in_store_price : ℚ := 99.99

/-- The price of one payment in the TV commercial in dollars -/
def tv_payment : ℚ := 29.98

/-- The number of payments in the TV commercial -/
def num_payments : ℕ := 3

/-- The shipping and handling charge in dollars -/
def shipping_charge : ℚ := 9.98

/-- The total cost from the TV advertiser in dollars -/
def tv_total_cost : ℚ := tv_payment * num_payments + shipping_charge

/-- The savings in dollars -/
def savings : ℚ := in_store_price - tv_total_cost

/-- Convert dollars to cents -/
def dollars_to_cents (dollars : ℚ) : ℕ := (dollars * 100).ceil.toNat

theorem savings_in_cents : dollars_to_cents savings = 7 := by sorry

end NUMINAMATH_CALUDE_savings_in_cents_l2963_296353


namespace NUMINAMATH_CALUDE_simplify_expressions_l2963_296395

theorem simplify_expressions :
  (3 * Real.sqrt 20 - Real.sqrt 45 + Real.sqrt (1/5) = 16 * Real.sqrt 5 / 5) ∧
  ((Real.sqrt 6 - 2 * Real.sqrt 3)^2 - (2 * Real.sqrt 5 + Real.sqrt 2) * (2 * Real.sqrt 5 - Real.sqrt 2) = -12 * Real.sqrt 2) := by
  sorry

end NUMINAMATH_CALUDE_simplify_expressions_l2963_296395


namespace NUMINAMATH_CALUDE_consecutive_integers_product_l2963_296388

theorem consecutive_integers_product (a b c d e : ℕ) : 
  a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0 ∧ e > 0 →
  b = a + 1 →
  c = b + 1 →
  d = c + 1 →
  e = d + 1 →
  a * b * c * d * e = 15120 →
  e = 9 := by
sorry

end NUMINAMATH_CALUDE_consecutive_integers_product_l2963_296388


namespace NUMINAMATH_CALUDE_sum_of_roots_quadratic_l2963_296320

theorem sum_of_roots_quadratic (x₁ x₂ : ℝ) : 
  (x₁^2 - 3*x₁ - 4 = 0) → (x₂^2 - 3*x₂ - 4 = 0) → x₁ + x₂ = 3 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_roots_quadratic_l2963_296320


namespace NUMINAMATH_CALUDE_inner_circle_radius_l2963_296314

theorem inner_circle_radius (s : ℝ) (h : s = 4) :
  let quarter_circle_radius := s / 2
  let square_diagonal := s * Real.sqrt 2
  let center_to_corner := square_diagonal / 2
  let r := (center_to_corner ^ 2 - quarter_circle_radius ^ 2).sqrt + quarter_circle_radius - center_to_corner
  r = 1 + Real.sqrt 3 := by sorry

end NUMINAMATH_CALUDE_inner_circle_radius_l2963_296314


namespace NUMINAMATH_CALUDE_max_intersections_nested_polygons_l2963_296328

/-- Represents a convex polygon -/
structure ConvexPolygon where
  sides : ℕ
  convex : Bool

/-- Represents the configuration of two nested convex polygons -/
structure NestedPolygons where
  inner : ConvexPolygon
  outer : ConvexPolygon
  nested : Bool
  no_shared_segments : Bool

/-- Calculates the maximum number of intersection points between two nested convex polygons -/
def max_intersections (np : NestedPolygons) : ℕ :=
  np.inner.sides * np.outer.sides

/-- Theorem stating the maximum number of intersections for the given configuration -/
theorem max_intersections_nested_polygons :
  ∀ (np : NestedPolygons),
    np.inner.sides = 5 →
    np.outer.sides = 8 →
    np.inner.convex = true →
    np.outer.convex = true →
    np.nested = true →
    np.no_shared_segments = true →
    max_intersections np = 40 :=
by sorry

end NUMINAMATH_CALUDE_max_intersections_nested_polygons_l2963_296328


namespace NUMINAMATH_CALUDE_clock_angle_at_8_l2963_296301

/-- The number of hours on a clock face -/
def clock_hours : ℕ := 12

/-- The number of degrees per hour on a clock face -/
def degrees_per_hour : ℕ := 360 / clock_hours

/-- The position of the minute hand at 8:00 in degrees -/
def minute_hand_position : ℕ := 0

/-- The position of the hour hand at 8:00 in degrees -/
def hour_hand_position : ℕ := 8 * degrees_per_hour

/-- The smaller angle between the hour and minute hands at 8:00 -/
def smaller_angle : ℕ := min (hour_hand_position - minute_hand_position) (360 - (hour_hand_position - minute_hand_position))

theorem clock_angle_at_8 : smaller_angle = 120 := by
  sorry

end NUMINAMATH_CALUDE_clock_angle_at_8_l2963_296301


namespace NUMINAMATH_CALUDE_exists_square_composition_function_l2963_296389

theorem exists_square_composition_function :
  ∃ f : ℕ → ℕ, ∀ n : ℕ, f (f n) = n ^ 2 := by
  sorry

end NUMINAMATH_CALUDE_exists_square_composition_function_l2963_296389


namespace NUMINAMATH_CALUDE_chessboard_tiling_l2963_296341

/-- Represents a chessboard -/
structure Chessboard :=
  (size : ℕ)

/-- Represents a polyomino -/
structure Polyomino :=
  (width : ℕ)
  (height : ℕ)

/-- Represents an L-shaped polyomino -/
def LPolyomino : Polyomino :=
  ⟨2, 2⟩

/-- Checks if a chessboard can be tiled with a given polyomino -/
def can_tile (board : Chessboard) (tile : Polyomino) : Prop :=
  ∃ n : ℕ, board.size * board.size = n * (tile.width * tile.height)

theorem chessboard_tiling (board : Chessboard) :
  board.size = 9 →
  ¬(can_tile board ⟨2, 1⟩) ∧
  (can_tile board ⟨3, 1⟩) ∧
  (can_tile board LPolyomino) :=
sorry

end NUMINAMATH_CALUDE_chessboard_tiling_l2963_296341


namespace NUMINAMATH_CALUDE_expected_heads_is_60_l2963_296375

/-- The number of coins --/
def num_coins : ℕ := 64

/-- The maximum number of tosses for each coin --/
def max_tosses : ℕ := 4

/-- The probability of getting heads on a single toss --/
def p_heads : ℚ := 1/2

/-- The probability of getting heads after up to four tosses --/
def p_heads_four_tosses : ℚ := 
  p_heads + (1 - p_heads) * p_heads + (1 - p_heads)^2 * p_heads + (1 - p_heads)^3 * p_heads

/-- The expected number of coins showing heads after up to four tosses --/
def expected_heads : ℚ := num_coins * p_heads_four_tosses

theorem expected_heads_is_60 : expected_heads = 60 := by sorry

end NUMINAMATH_CALUDE_expected_heads_is_60_l2963_296375


namespace NUMINAMATH_CALUDE_simplify_expression_l2963_296354

theorem simplify_expression (y : ℝ) :
  4 * y + 8 * y^2 + 6 - (3 - 4 * y - 8 * y^2) = 16 * y^2 + 8 * y + 3 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l2963_296354


namespace NUMINAMATH_CALUDE_a_3_value_l2963_296302

def S (n : ℕ+) : ℕ := 5 * n.val ^ 2 + 10 * n.val

theorem a_3_value : ∃ (a : ℕ+ → ℕ), a 3 = 35 :=
  sorry

end NUMINAMATH_CALUDE_a_3_value_l2963_296302


namespace NUMINAMATH_CALUDE_sin_beta_value_l2963_296332

theorem sin_beta_value (α β : Real) 
  (h1 : 0 < α ∧ α < Real.pi / 2)
  (h2 : 0 < β ∧ β < Real.pi / 2)
  (h3 : Real.cos α = 1 / 7)
  (h4 : Real.cos (α + β) = -11 / 14) :
  Real.sin β = Real.sqrt 3 / 2 := by
sorry

end NUMINAMATH_CALUDE_sin_beta_value_l2963_296332


namespace NUMINAMATH_CALUDE_combined_output_fraction_l2963_296397

/-- Represents the production rate of a machine relative to a base rate -/
structure ProductionRate :=
  (rate : ℚ)

/-- Represents a machine with its production rate -/
structure Machine :=
  (name : String)
  (rate : ProductionRate)

/-- The problem setup with four machines and their relative production rates -/
def production_problem (t n o p : Machine) : Prop :=
  t.rate.rate = 4 / 3 * n.rate.rate ∧
  n.rate.rate = 3 / 2 * o.rate.rate ∧
  o.rate = p.rate

/-- The theorem stating that machines N and P produce 6/13 of the total output -/
theorem combined_output_fraction 
  (t n o p : Machine) 
  (h : production_problem t n o p) : 
  (n.rate.rate + p.rate.rate) / (t.rate.rate + n.rate.rate + o.rate.rate + p.rate.rate) = 6 / 13 :=
sorry

end NUMINAMATH_CALUDE_combined_output_fraction_l2963_296397


namespace NUMINAMATH_CALUDE_sum_interior_angles_180_l2963_296350

-- Define a triangle
structure Triangle where
  vertices : Fin 3 → ℝ × ℝ

-- Define interior angles of a triangle
def interior_angles (t : Triangle) : Fin 3 → ℝ := sorry

-- Theorem: The sum of interior angles of any triangle is 180°
theorem sum_interior_angles_180 (t : Triangle) : 
  (interior_angles t 0) + (interior_angles t 1) + (interior_angles t 2) = 180 := by
  sorry

end NUMINAMATH_CALUDE_sum_interior_angles_180_l2963_296350


namespace NUMINAMATH_CALUDE_bananas_to_kiwis_ratio_l2963_296366

/-- Represents the cost of a dozen apples in dollars -/
def dozen_apples_cost : ℚ := 14

/-- Represents the amount Brian spent on kiwis in dollars -/
def kiwis_cost : ℚ := 10

/-- Represents the maximum number of apples Brian can buy -/
def max_apples : ℕ := 24

/-- Represents the amount Brian left his house with in dollars -/
def initial_amount : ℚ := 50

/-- Represents the subway fare in dollars -/
def subway_fare : ℚ := 3.5

/-- Calculates the amount spent on bananas -/
def bananas_cost : ℚ := initial_amount - 2 * subway_fare - kiwis_cost - (max_apples / 12) * dozen_apples_cost

/-- Theorem stating that the ratio of bananas cost to kiwis cost is 1:2 -/
theorem bananas_to_kiwis_ratio : bananas_cost / kiwis_cost = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_bananas_to_kiwis_ratio_l2963_296366


namespace NUMINAMATH_CALUDE_rhombus_area_l2963_296374

/-- The area of a rhombus with side length √145 and diagonals differing by 10 units is 100 square units. -/
theorem rhombus_area (side_length : ℝ) (diagonal_difference : ℝ) (area : ℝ) :
  side_length = Real.sqrt 145 →
  diagonal_difference = 10 →
  area = 100 →
  ∃ (d1 d2 : ℝ), d1 > 0 ∧ d2 > 0 ∧ 
    d2 - d1 = diagonal_difference ∧
    d1 * d2 / 2 = area ∧
    d1^2 / 4 + d2^2 / 4 = side_length^2 :=
by sorry

end NUMINAMATH_CALUDE_rhombus_area_l2963_296374


namespace NUMINAMATH_CALUDE_antimatter_prescription_fulfillment_l2963_296322

theorem antimatter_prescription_fulfillment :
  ∃ (x y z : ℕ), x ≥ 1 ∧ y ≥ 1 ∧ z ≥ 1 ∧
  (11 : ℝ) * x + 1.1 * y + 0.11 * z = 20.13 := by
  sorry

end NUMINAMATH_CALUDE_antimatter_prescription_fulfillment_l2963_296322


namespace NUMINAMATH_CALUDE_grandma_crane_folding_l2963_296355

/-- Represents time in hours and minutes -/
structure Time where
  hours : Nat
  minutes : Nat

/-- Adds minutes to a given time -/
def addMinutes (t : Time) (m : Nat) : Time :=
  let totalMinutes := t.hours * 60 + t.minutes + m
  { hours := totalMinutes / 60, minutes := totalMinutes % 60 }

theorem grandma_crane_folding :
  let foldTime : Nat := 3  -- Time to fold one crane
  let restTime : Nat := 1  -- Rest time after folding each crane
  let startTime : Time := { hours := 14, minutes := 30 }  -- 2:30 PM
  let numCranes : Nat := 5
  
  let totalFoldTime := foldTime * numCranes
  let totalRestTime := restTime * (numCranes - 1)
  let totalTime := totalFoldTime + totalRestTime
  
  addMinutes startTime totalTime = { hours := 14, minutes := 49 }  -- 2:49 PM
  := by sorry

end NUMINAMATH_CALUDE_grandma_crane_folding_l2963_296355


namespace NUMINAMATH_CALUDE_equation_solutions_l2963_296380

theorem equation_solutions :
  (∀ x, x^2 - 8*x - 1 = 0 ↔ x = 4 + Real.sqrt 17 ∨ x = 4 - Real.sqrt 17) ∧
  (∀ x, x*(2*x - 5) = 4*x - 10 ↔ x = 5/2 ∨ x = 2) := by
  sorry

end NUMINAMATH_CALUDE_equation_solutions_l2963_296380


namespace NUMINAMATH_CALUDE_tan_beta_value_l2963_296330

open Real

theorem tan_beta_value (α β : ℝ) 
  (h1 : tan (α + β) = 3) 
  (h2 : tan (α + π/4) = 2) : 
  tan β = 2 := by
sorry

end NUMINAMATH_CALUDE_tan_beta_value_l2963_296330
