import Mathlib

namespace NUMINAMATH_CALUDE_fiftieth_term_is_247_l1663_166365

/-- The nth term of an arithmetic sequence -/
def arithmetic_sequence (a : ℤ) (d : ℤ) (n : ℕ) : ℤ := a + (n - 1 : ℤ) * d

theorem fiftieth_term_is_247 : 
  arithmetic_sequence 2 5 50 = 247 := by sorry

end NUMINAMATH_CALUDE_fiftieth_term_is_247_l1663_166365


namespace NUMINAMATH_CALUDE_seeds_in_fourth_pot_l1663_166344

/-- Given 10 total seeds, 4 pots, and 3 seeds per pot for the first 3 pots,
    prove that the number of seeds in the fourth pot is 1. -/
theorem seeds_in_fourth_pot
  (total_seeds : ℕ)
  (num_pots : ℕ)
  (seeds_per_pot : ℕ)
  (h1 : total_seeds = 10)
  (h2 : num_pots = 4)
  (h3 : seeds_per_pot = 3)
  : total_seeds - (seeds_per_pot * (num_pots - 1)) = 1 := by
  sorry

end NUMINAMATH_CALUDE_seeds_in_fourth_pot_l1663_166344


namespace NUMINAMATH_CALUDE_today_is_thursday_l1663_166332

-- Define the days of the week
inductive Day
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday
  | Saturday
  | Sunday

-- Define the vehicles
inductive Vehicle
  | A
  | B
  | C
  | D
  | E

def is_weekday (d : Day) : Prop :=
  d ≠ Day.Saturday ∧ d ≠ Day.Sunday

def next_day (d : Day) : Day :=
  match d with
  | Day.Monday => Day.Tuesday
  | Day.Tuesday => Day.Wednesday
  | Day.Wednesday => Day.Thursday
  | Day.Thursday => Day.Friday
  | Day.Friday => Day.Saturday
  | Day.Saturday => Day.Sunday
  | Day.Sunday => Day.Monday

def can_operate (v : Vehicle) (d : Day) : Prop := sorry

theorem today_is_thursday 
  (h1 : ∀ (d : Day), is_weekday d → ∃ (v : Vehicle), ¬can_operate v d)
  (h2 : ∀ (d : Day), is_weekday d → (∃ (v1 v2 v3 v4 : Vehicle), can_operate v1 d ∧ can_operate v2 d ∧ can_operate v3 d ∧ can_operate v4 d))
  (h3 : ¬can_operate Vehicle.E Day.Thursday)
  (h4 : ¬can_operate Vehicle.B (next_day today))
  (h5 : ∀ (d : Day), d = today ∨ d = next_day today ∨ d = next_day (next_day today) ∨ d = next_day (next_day (next_day today)) → can_operate Vehicle.A d ∧ can_operate Vehicle.C d)
  (h6 : can_operate Vehicle.E (next_day today))
  : today = Day.Thursday :=
sorry

end NUMINAMATH_CALUDE_today_is_thursday_l1663_166332


namespace NUMINAMATH_CALUDE_roots_opposite_signs_l1663_166383

theorem roots_opposite_signs (a b c : ℝ) :
  (∃ x y : ℝ, x ≠ y ∧ a * x^2 + 2 * b * x + c = 0 ∧ a * y^2 + 2 * b * y + c = 0) →
  (∀ z : ℝ, a^2 * z^2 + 2 * b^2 * z + c^2 ≠ 0) →
  a * c < 0 := by
sorry


end NUMINAMATH_CALUDE_roots_opposite_signs_l1663_166383


namespace NUMINAMATH_CALUDE_symmetric_points_coordinates_l1663_166370

/-- Two points are symmetric about the origin if the sum of their coordinates is zero -/
def symmetric_about_origin (x₁ y₁ x₂ y₂ : ℝ) : Prop :=
  x₁ + x₂ = 0 ∧ y₁ + y₂ = 0

/-- Given that point A (1, 2) and point A' (a, b) are symmetric about the origin,
    prove that a = -1 and b = -2 -/
theorem symmetric_points_coordinates :
  ∀ (a b : ℝ), symmetric_about_origin 1 2 a b → a = -1 ∧ b = -2 := by
  sorry

end NUMINAMATH_CALUDE_symmetric_points_coordinates_l1663_166370


namespace NUMINAMATH_CALUDE_flour_already_put_in_l1663_166378

/-- Given a recipe that requires a certain amount of flour and the additional amount needed,
    calculate the amount of flour already put in. -/
theorem flour_already_put_in
  (recipe_requirement : ℕ)  -- Total cups of flour required by the recipe
  (additional_needed : ℕ)   -- Additional cups of flour needed
  (h1 : recipe_requirement = 7)  -- The recipe requires 7 cups of flour
  (h2 : additional_needed = 5)   -- Mary needs to add 5 more cups
  : recipe_requirement - additional_needed = 2 :=
by sorry

end NUMINAMATH_CALUDE_flour_already_put_in_l1663_166378


namespace NUMINAMATH_CALUDE_equal_remainders_divisor_l1663_166317

theorem equal_remainders_divisor : ∃ (n : ℕ), n > 0 ∧ 
  n ∣ (2287 - 2028) ∧ 
  n ∣ (2028 - 1806) ∧ 
  n ∣ (2287 - 1806) ∧
  ∀ (m : ℕ), m > n → ¬(m ∣ (2287 - 2028) ∧ m ∣ (2028 - 1806) ∧ m ∣ (2287 - 1806)) :=
by sorry

end NUMINAMATH_CALUDE_equal_remainders_divisor_l1663_166317


namespace NUMINAMATH_CALUDE_average_of_xyz_l1663_166399

theorem average_of_xyz (x y z : ℝ) (h : (5 / 4) * (x + y + z) = 20) :
  (x + y + z) / 3 = 16 / 3 := by
  sorry

end NUMINAMATH_CALUDE_average_of_xyz_l1663_166399


namespace NUMINAMATH_CALUDE_problem_solution_l1663_166393

theorem problem_solution (a b : ℝ) (h1 : a + b = 4) (h2 : a^2 - 2*a*b + b^2 + 2*a + 2*b = 17) : 
  ((a + 1) * (b + 1) - a * b = 5) ∧ ((a - b)^2 = 9) := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l1663_166393


namespace NUMINAMATH_CALUDE_number_interval_l1663_166363

theorem number_interval (x : ℝ) (h : x = (1/x) * (-x) + 4) : 2 < x ∧ x ≤ 4 := by
  sorry

end NUMINAMATH_CALUDE_number_interval_l1663_166363


namespace NUMINAMATH_CALUDE_undefined_values_count_l1663_166322

theorem undefined_values_count : ∃! (s : Finset ℤ), 
  (∀ x ∈ s, (x^2 - x - 6) * (x - 4) = 0) ∧ 
  (∀ x ∉ s, (x^2 - x - 6) * (x - 4) ≠ 0) ∧ 
  s.card = 3 := by
  sorry

end NUMINAMATH_CALUDE_undefined_values_count_l1663_166322


namespace NUMINAMATH_CALUDE_morley_theorem_l1663_166343

/-- Represents a point in 2D space -/
structure Point :=
  (x : ℝ) (y : ℝ)

/-- Represents a triangle -/
structure Triangle :=
  (A B C : Point)

/-- Represents a ray (half-line) -/
structure Ray :=
  (origin : Point) (direction : Point)

/-- Defines a trisector of an angle -/
def is_trisector (r : Ray) (A B C : Point) : Prop := sorry

/-- Defines the intersection point of two rays -/
def intersection (r1 r2 : Ray) : Point := sorry

/-- Morley's theorem -/
theorem morley_theorem (T : Triangle) :
  let A := T.A
  let B := T.B
  let C := T.C
  let trisector_B1 := Ray.mk B (sorry : Point)
  let trisector_B2 := Ray.mk B (sorry : Point)
  let trisector_C1 := Ray.mk C (sorry : Point)
  let trisector_C2 := Ray.mk C (sorry : Point)
  let trisector_A1 := Ray.mk A (sorry : Point)
  let trisector_A2 := Ray.mk A (sorry : Point)
  let A1 := intersection trisector_B1 trisector_C1
  let B1 := intersection trisector_C2 trisector_A1
  let C1 := intersection trisector_A2 trisector_B2
  is_trisector trisector_B1 B A C ∧
  is_trisector trisector_B2 B A C ∧
  is_trisector trisector_C1 C B A ∧
  is_trisector trisector_C2 C B A ∧
  is_trisector trisector_A1 A C B ∧
  is_trisector trisector_A2 A C B →
  -- A1B1 = B1C1 = C1A1
  (A1.x - B1.x)^2 + (A1.y - B1.y)^2 =
  (B1.x - C1.x)^2 + (B1.y - C1.y)^2 ∧
  (B1.x - C1.x)^2 + (B1.y - C1.y)^2 =
  (C1.x - A1.x)^2 + (C1.y - A1.y)^2 :=
sorry

end NUMINAMATH_CALUDE_morley_theorem_l1663_166343


namespace NUMINAMATH_CALUDE_selection_probabilities_l1663_166333

/-- Represents the number of boys in the group -/
def num_boys : ℕ := 4

/-- Represents the number of girls in the group -/
def num_girls : ℕ := 3

/-- Represents the total number of people in the group -/
def total_people : ℕ := num_boys + num_girls

/-- Represents the number of people to be selected -/
def num_selected : ℕ := 2

/-- Calculates the probability of selecting two boys -/
def prob_two_boys : ℚ := (num_boys.choose 2) / (total_people.choose 2)

/-- Calculates the probability of selecting exactly one girl -/
def prob_one_girl : ℚ := (num_boys.choose 1 * num_girls.choose 1) / (total_people.choose 2)

/-- Calculates the probability of selecting at least one girl -/
def prob_at_least_one_girl : ℚ := 1 - prob_two_boys

theorem selection_probabilities :
  prob_two_boys = 2/7 ∧
  prob_one_girl = 4/7 ∧
  prob_at_least_one_girl = 5/7 := by
  sorry

end NUMINAMATH_CALUDE_selection_probabilities_l1663_166333


namespace NUMINAMATH_CALUDE_min_ratio_two_digit_integers_l1663_166316

theorem min_ratio_two_digit_integers (x y : ℕ) : 
  (10 ≤ x ∧ x ≤ 99) →  -- x is a two-digit positive integer
  (10 ≤ y ∧ y ≤ 99) →  -- y is a two-digit positive integer
  (x + y) / 2 = 75 →   -- mean of x and y is 75
  (∀ a b : ℕ, (10 ≤ a ∧ a ≤ 99) → (10 ≤ b ∧ b ≤ 99) → (a + b) / 2 = 75 → 
    x / (3 * y + 4 : ℚ) ≤ a / (3 * b + 4 : ℚ)) →
  x / (3 * y + 4 : ℚ) = 70 / 17 := by
sorry

end NUMINAMATH_CALUDE_min_ratio_two_digit_integers_l1663_166316


namespace NUMINAMATH_CALUDE_monotone_decreasing_range_a_l1663_166391

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x < 1 then x^2 - 4*a*x + 3 else (2 - 3*a)*x + 1

theorem monotone_decreasing_range_a :
  ∀ a : ℝ, (∀ x y : ℝ, x < y → f a x > f a y) → a ∈ Set.Icc (1/2) (2/3) := by
  sorry

end NUMINAMATH_CALUDE_monotone_decreasing_range_a_l1663_166391


namespace NUMINAMATH_CALUDE_cube_volume_from_side_area_l1663_166356

theorem cube_volume_from_side_area :
  ∀ (side_area : ℝ) (volume : ℝ),
    side_area = 64 →
    volume = (side_area.sqrt) ^ 3 →
    volume = 512 := by
  sorry

end NUMINAMATH_CALUDE_cube_volume_from_side_area_l1663_166356


namespace NUMINAMATH_CALUDE_root_difference_quadratic_l1663_166396

theorem root_difference_quadratic (p : ℝ) : 
  let r := (2*p + Real.sqrt (9 : ℝ))
  let s := (2*p - Real.sqrt (9 : ℝ))
  r - s = 6 := by
sorry

end NUMINAMATH_CALUDE_root_difference_quadratic_l1663_166396


namespace NUMINAMATH_CALUDE_arrangements_count_l1663_166329

/-- The number of different arrangements of 5 students (2 girls and 3 boys) 
    where the two girls are not next to each other -/
def num_arrangements : ℕ := 72

/-- The number of ways to arrange 3 boys -/
def boy_arrangements : ℕ := 6

/-- The number of ways to insert 2 girls into 4 possible spaces -/
def girl_insertions : ℕ := 12

theorem arrangements_count : 
  num_arrangements = boy_arrangements * girl_insertions :=
by sorry

end NUMINAMATH_CALUDE_arrangements_count_l1663_166329


namespace NUMINAMATH_CALUDE_invoice_problem_l1663_166326

theorem invoice_problem (x y : ℚ) : 
  (0.3 < x) ∧ (x < 0.4) ∧ 
  (7000 < y) ∧ (y < 8000) ∧ 
  (y * 100 - (y.floor * 100) = 65) ∧
  (237 * x = y) →
  (x = 0.31245 ∧ y = 7400.65) := by
sorry

end NUMINAMATH_CALUDE_invoice_problem_l1663_166326


namespace NUMINAMATH_CALUDE_fraction_sum_equality_l1663_166318

theorem fraction_sum_equality : 
  (1 / (2 * 3 : ℚ)) + (1 / (3 * 4 : ℚ)) + (1 / (4 * 5 : ℚ)) + 
  (1 / (5 * 6 : ℚ)) + (1 / (6 * 7 : ℚ)) = 5 / 14 := by
  sorry

end NUMINAMATH_CALUDE_fraction_sum_equality_l1663_166318


namespace NUMINAMATH_CALUDE_gain_percent_calculation_l1663_166310

theorem gain_percent_calculation (MP : ℝ) (MP_pos : MP > 0) : 
  let CP := 0.25 * MP
  let SP := 0.5 * MP
  let gain := SP - CP
  let gain_percent := (gain / CP) * 100
  gain_percent = 100 := by sorry

end NUMINAMATH_CALUDE_gain_percent_calculation_l1663_166310


namespace NUMINAMATH_CALUDE_lisa_max_below_a_l1663_166388

/-- Represents Lisa's quiz performance and goal --/
structure QuizPerformance where
  total_quizzes : ℕ
  goal_percentage : ℚ
  completed_quizzes : ℕ
  completed_as : ℕ

/-- Calculates the maximum number of remaining quizzes where Lisa can score below 'A' --/
def max_below_a (qp : QuizPerformance) : ℕ :=
  let total_as_needed : ℕ := (qp.goal_percentage * qp.total_quizzes).ceil.toNat
  let remaining_quizzes : ℕ := qp.total_quizzes - qp.completed_quizzes
  remaining_quizzes - (total_as_needed - qp.completed_as)

/-- Theorem stating that given Lisa's quiz performance, the maximum number of remaining quizzes where she can score below 'A' is 7 --/
theorem lisa_max_below_a :
  let qp : QuizPerformance := {
    total_quizzes := 60,
    goal_percentage := 3/4,
    completed_quizzes := 30,
    completed_as := 22
  }
  max_below_a qp = 7 := by sorry

end NUMINAMATH_CALUDE_lisa_max_below_a_l1663_166388


namespace NUMINAMATH_CALUDE_total_highlighters_count_l1663_166362

/-- The number of pink highlighters in the teacher's desk -/
def pink_highlighters : ℕ := 4

/-- The number of yellow highlighters in the teacher's desk -/
def yellow_highlighters : ℕ := 2

/-- The number of blue highlighters in the teacher's desk -/
def blue_highlighters : ℕ := 5

/-- The total number of highlighters in the teacher's desk -/
def total_highlighters : ℕ := pink_highlighters + yellow_highlighters + blue_highlighters

/-- Theorem stating that the total number of highlighters is 11 -/
theorem total_highlighters_count : total_highlighters = 11 := by
  sorry

end NUMINAMATH_CALUDE_total_highlighters_count_l1663_166362


namespace NUMINAMATH_CALUDE_complement_of_M_l1663_166390

def U : Set Nat := {1, 2, 3, 4, 5, 6}
def M : Set Nat := {1, 3, 5}

theorem complement_of_M : (U \ M) = {2, 4, 6} := by sorry

end NUMINAMATH_CALUDE_complement_of_M_l1663_166390


namespace NUMINAMATH_CALUDE_cyclic_fourth_root_sum_inequality_l1663_166352

theorem cyclic_fourth_root_sum_inequality (a₁ a₂ a₃ a₄ a₅ a₆ : ℝ) 
  (ha₁ : a₁ > 0) (ha₂ : a₂ > 0) (ha₃ : a₃ > 0) (ha₄ : a₄ > 0) (ha₅ : a₅ > 0) (ha₆ : a₆ > 0) : 
  (a₁ / (a₂ + a₃ + a₄)) ^ (1/4 : ℝ) + 
  (a₂ / (a₃ + a₄ + a₅)) ^ (1/4 : ℝ) + 
  (a₃ / (a₄ + a₅ + a₆)) ^ (1/4 : ℝ) + 
  (a₄ / (a₅ + a₆ + a₁)) ^ (1/4 : ℝ) + 
  (a₅ / (a₆ + a₁ + a₂)) ^ (1/4 : ℝ) + 
  (a₆ / (a₁ + a₂ + a₃)) ^ (1/4 : ℝ) ≥ 2 := by
  sorry

end NUMINAMATH_CALUDE_cyclic_fourth_root_sum_inequality_l1663_166352


namespace NUMINAMATH_CALUDE_even_iff_divisible_by_72_l1663_166337

theorem even_iff_divisible_by_72 (n : ℕ) : 
  Even n ↔ 72 ∣ (3^n + 63) := by sorry

end NUMINAMATH_CALUDE_even_iff_divisible_by_72_l1663_166337


namespace NUMINAMATH_CALUDE_factorial_division_l1663_166325

theorem factorial_division (h : Nat.factorial 10 = 3628800) :
  Nat.factorial 10 / Nat.factorial 4 = 151200 := by
  sorry

end NUMINAMATH_CALUDE_factorial_division_l1663_166325


namespace NUMINAMATH_CALUDE_min_value_of_fraction_l1663_166353

theorem min_value_of_fraction (a : ℝ) (h : a > 1) :
  (a^2 - a + 1) / (a - 1) ≥ 3 ∧
  ∃ b > 1, (b^2 - b + 1) / (b - 1) = 3 :=
sorry

end NUMINAMATH_CALUDE_min_value_of_fraction_l1663_166353


namespace NUMINAMATH_CALUDE_pool_filling_proof_l1663_166302

/-- The rate at which the first hose sprays water in gallons per hour -/
def first_hose_rate : ℝ := 50

/-- The rate at which the second hose sprays water in gallons per hour -/
def second_hose_rate : ℝ := 70

/-- The capacity of the pool in gallons -/
def pool_capacity : ℝ := 390

/-- The time the first hose was used alone in hours -/
def first_hose_time : ℝ := 3

/-- The time both hoses were used together in hours -/
def both_hoses_time : ℝ := 2

theorem pool_filling_proof : 
  first_hose_rate * first_hose_time + 
  (first_hose_rate + second_hose_rate) * both_hoses_time = 
  pool_capacity := by sorry

end NUMINAMATH_CALUDE_pool_filling_proof_l1663_166302


namespace NUMINAMATH_CALUDE_article_cost_l1663_166348

theorem article_cost (sell_price_1 sell_price_2 : ℝ) 
  (h1 : sell_price_1 = 380)
  (h2 : sell_price_2 = 420)
  (h3 : sell_price_2 - sell_price_1 = 0.05 * cost) : cost = 800 := by
  sorry

end NUMINAMATH_CALUDE_article_cost_l1663_166348


namespace NUMINAMATH_CALUDE_second_village_sales_l1663_166345

/-- Given the number of cookie packs sold in the first village and the total number of packs sold,
    calculate the number of packs sold in the second village. -/
def cookiesSoldInSecondVillage (firstVillage : ℕ) (total : ℕ) : ℕ :=
  total - firstVillage

/-- Theorem stating that the number of cookie packs sold in the second village
    is equal to the total number of packs sold minus the number sold in the first village. -/
theorem second_village_sales (firstVillage : ℕ) (total : ℕ) 
    (h : firstVillage ≤ total) :
  cookiesSoldInSecondVillage firstVillage total = total - firstVillage := by
  sorry

#eval cookiesSoldInSecondVillage 23 51  -- Expected output: 28

end NUMINAMATH_CALUDE_second_village_sales_l1663_166345


namespace NUMINAMATH_CALUDE_square_area_side_perimeter_l1663_166321

theorem square_area_side_perimeter :
  ∀ (s p : ℝ),
  s > 0 →
  s^2 = 450 →
  p = 4 * s →
  s = 15 * Real.sqrt 2 ∧ p = 60 * Real.sqrt 2 :=
by sorry

end NUMINAMATH_CALUDE_square_area_side_perimeter_l1663_166321


namespace NUMINAMATH_CALUDE_ball_probability_l1663_166367

theorem ball_probability (n : ℕ) : 
  (2 : ℝ) / ((n : ℝ) + 2) = 0.4 → n = 3 := by
  sorry

end NUMINAMATH_CALUDE_ball_probability_l1663_166367


namespace NUMINAMATH_CALUDE_cody_tickets_l1663_166305

/-- Calculates the final number of tickets Cody has after winning, spending, and winning again. -/
def final_tickets (initial : ℕ) (spent : ℕ) (won_later : ℕ) : ℕ :=
  initial - spent + won_later

/-- Theorem stating that Cody ends up with 30 tickets given the problem conditions. -/
theorem cody_tickets : final_tickets 49 25 6 = 30 := by
  sorry

end NUMINAMATH_CALUDE_cody_tickets_l1663_166305


namespace NUMINAMATH_CALUDE_music_library_avg_megabytes_per_hour_l1663_166382

/-- Represents a digital music library -/
structure MusicLibrary where
  days : ℕ
  megabytes : ℕ

/-- Calculates the average megabytes per hour of music in a library, rounded to the nearest integer -/
def avgMegabytesPerHour (library : MusicLibrary) : ℕ :=
  let totalHours : ℕ := library.days * 24
  let exactAvg : ℚ := library.megabytes / totalHours
  (exactAvg + 1/2).floor.toNat

/-- Theorem stating that for a library with 15 days of music and 20,000 megabytes,
    the average megabytes per hour rounded to the nearest integer is 56 -/
theorem music_library_avg_megabytes_per_hour :
  let library : MusicLibrary := ⟨15, 20000⟩
  avgMegabytesPerHour library = 56 := by
  sorry

end NUMINAMATH_CALUDE_music_library_avg_megabytes_per_hour_l1663_166382


namespace NUMINAMATH_CALUDE_count_arrangements_eq_21_l1663_166360

/-- A function that counts the number of valid arrangements of digits 1, 1, 2, 5, 0 -/
def countArrangements : ℕ :=
  let digits : List ℕ := [1, 1, 2, 5, 0]
  let isValidArrangement (arr : List ℕ) : Bool :=
    arr.length = 5 ∧ 
    arr.head? ≠ some 0 ∧ 
    (arr.getLast? = some 0 ∨ arr.getLast? = some 5)

  -- Count valid arrangements
  sorry

/-- The theorem stating that the number of valid arrangements is 21 -/
theorem count_arrangements_eq_21 : countArrangements = 21 := by
  sorry

end NUMINAMATH_CALUDE_count_arrangements_eq_21_l1663_166360


namespace NUMINAMATH_CALUDE_no_leg_longer_than_both_l1663_166324

-- Define two right triangles
structure RightTriangle where
  leg1 : ℝ
  leg2 : ℝ
  hypotenuse : ℝ
  right_angle : leg1^2 + leg2^2 = hypotenuse^2

-- Theorem statement
theorem no_leg_longer_than_both (t1 t2 : RightTriangle) 
  (h : t1.hypotenuse = t2.hypotenuse) : 
  ¬(t1.leg1 > t2.leg1 ∧ t1.leg1 > t2.leg2) ∨ 
  ¬(t1.leg2 > t2.leg1 ∧ t1.leg2 > t2.leg2) :=
sorry

end NUMINAMATH_CALUDE_no_leg_longer_than_both_l1663_166324


namespace NUMINAMATH_CALUDE_inequality_proof_l1663_166336

theorem inequality_proof (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) :
  (1 / (2 * x)) + (1 / (2 * y)) + (1 / (2 * z)) > (1 / (y + z)) + (1 / (z + x)) + (1 / (x + y)) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l1663_166336


namespace NUMINAMATH_CALUDE_gary_book_multiple_l1663_166392

/-- Proves that Gary's books are 5 times the combined number of Darla's and Katie's books -/
theorem gary_book_multiple (darla_books katie_books gary_books : ℕ) : 
  darla_books = 6 →
  katie_books = darla_books / 2 →
  gary_books = (darla_books + katie_books) * (gary_books / (darla_books + katie_books)) →
  darla_books + katie_books + gary_books = 54 →
  gary_books / (darla_books + katie_books) = 5 := by
sorry

end NUMINAMATH_CALUDE_gary_book_multiple_l1663_166392


namespace NUMINAMATH_CALUDE_triangle_angle_measure_l1663_166320

theorem triangle_angle_measure (D E F : ℝ) : 
  D = 3 * E →  -- Angle D is thrice as large as angle E
  E = 18 →     -- Angle E measures 18°
  D + E + F = 180 →  -- Sum of angles in a triangle is 180°
  F = 108 :=   -- Angle F measures 108°
by
  sorry

end NUMINAMATH_CALUDE_triangle_angle_measure_l1663_166320


namespace NUMINAMATH_CALUDE_f_properties_l1663_166372

def f (x : ℝ) : ℝ := x^3 - 3*x^2

theorem f_properties : 
  (∀ x y, x < y ∧ ((x ≤ 0 ∧ y ≤ 0) ∨ (x ≥ 2 ∧ y ≥ 2)) → f x < f y) ∧ 
  (∀ x y, 0 ≤ x ∧ x < y ∧ y ≤ 2 → f x > f y) ∧
  (∃ δ > 0, ∀ x, 0 < |x| ∧ |x| < δ → f x < f 0) ∧
  (∃ δ > 0, ∀ x, 0 < |x - 2| ∧ |x - 2| < δ → f x > f 2) :=
by sorry

end NUMINAMATH_CALUDE_f_properties_l1663_166372


namespace NUMINAMATH_CALUDE_complex_fraction_simplification_l1663_166328

theorem complex_fraction_simplification :
  ((-1 : ℂ) + 3*Complex.I) / (1 + Complex.I) = 1 + 2*Complex.I :=
by sorry

end NUMINAMATH_CALUDE_complex_fraction_simplification_l1663_166328


namespace NUMINAMATH_CALUDE_distance_one_fourth_way_l1663_166338

/-- Represents an elliptical orbit -/
structure EllipticalOrbit where
  perigee : ℝ
  apogee : ℝ

/-- Calculates the distance from the focus to a point on the orbit -/
def distanceFromFocus (orbit : EllipticalOrbit) (fraction : ℝ) : ℝ :=
  orbit.perigee + fraction * (orbit.apogee - orbit.perigee)

/-- Theorem: For the given elliptical orbit, the distance from the focus to a point
    one-fourth way from perigee to apogee is 6.75 AU -/
theorem distance_one_fourth_way (orbit : EllipticalOrbit)
    (h1 : orbit.perigee = 3)
    (h2 : orbit.apogee = 15) :
    distanceFromFocus orbit (1/4) = 6.75 := by
  sorry

#check distance_one_fourth_way

end NUMINAMATH_CALUDE_distance_one_fourth_way_l1663_166338


namespace NUMINAMATH_CALUDE_real_part_of_complex_product_l1663_166386

theorem real_part_of_complex_product : ∃ (z : ℂ), z = (2 + Complex.I) * (1 - Complex.I) ∧ z.re = 3 := by
  sorry

end NUMINAMATH_CALUDE_real_part_of_complex_product_l1663_166386


namespace NUMINAMATH_CALUDE_repeating_decimal_sum_l1663_166311

/-- Represents a repeating decimal with a single digit repeating -/
def RepeatingDecimal (n : ℕ) : ℚ := n / 9

theorem repeating_decimal_sum :
  RepeatingDecimal 6 + RepeatingDecimal 2 - RepeatingDecimal 4 = 4 / 9 := by
  sorry

end NUMINAMATH_CALUDE_repeating_decimal_sum_l1663_166311


namespace NUMINAMATH_CALUDE_greater_number_problem_l1663_166368

theorem greater_number_problem (A B : ℕ+) : 
  (Nat.gcd A B = 11) → 
  (A * B = 363) → 
  (max A B = 33) := by
sorry

end NUMINAMATH_CALUDE_greater_number_problem_l1663_166368


namespace NUMINAMATH_CALUDE_cricketer_average_score_l1663_166359

theorem cricketer_average_score 
  (total_matches : ℕ) 
  (matches_with_known_average : ℕ) 
  (known_average : ℝ) 
  (total_average : ℝ) 
  (h1 : total_matches = 25)
  (h2 : matches_with_known_average = 15)
  (h3 : known_average = 70)
  (h4 : total_average = 66) :
  let remaining_matches := total_matches - matches_with_known_average
  (total_matches * total_average - matches_with_known_average * known_average) / remaining_matches = 60 := by
  sorry

end NUMINAMATH_CALUDE_cricketer_average_score_l1663_166359


namespace NUMINAMATH_CALUDE_arithmetic_square_root_of_negative_five_squared_l1663_166319

theorem arithmetic_square_root_of_negative_five_squared (x : ℝ) : 
  x = 5 ∧ x * x = (-5)^2 ∧ x ≥ 0 :=
sorry

end NUMINAMATH_CALUDE_arithmetic_square_root_of_negative_five_squared_l1663_166319


namespace NUMINAMATH_CALUDE_expression_evaluation_l1663_166327

/-- Given x = y + z and y > z > 0, prove that ((x+y)^z + (x+z)^y) / (y^z + z^y) = 2^y + 2^z -/
theorem expression_evaluation (x y z : ℝ) 
  (h1 : x = y + z) 
  (h2 : y > z) 
  (h3 : z > 0) : 
  ((x + y)^z + (x + z)^y) / (y^z + z^y) = 2^y + 2^z :=
sorry

end NUMINAMATH_CALUDE_expression_evaluation_l1663_166327


namespace NUMINAMATH_CALUDE_mary_cut_ten_roses_l1663_166341

/-- The number of roses Mary cut from her garden -/
def roses_cut : ℕ := 16 - 6

/-- Theorem stating that Mary cut 10 roses -/
theorem mary_cut_ten_roses : roses_cut = 10 := by
  sorry

end NUMINAMATH_CALUDE_mary_cut_ten_roses_l1663_166341


namespace NUMINAMATH_CALUDE_inequality_of_five_variables_l1663_166340

theorem inequality_of_five_variables (a b c d e : ℝ) 
  (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0) (he : e > 0) : 
  Real.sqrt (a / (b + c + d + e)) + 
  Real.sqrt (b / (a + c + d + e)) + 
  Real.sqrt (c / (a + b + d + e)) + 
  Real.sqrt (d / (a + b + c + e)) + 
  Real.sqrt (e / (a + b + c + d)) ≥ 2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_of_five_variables_l1663_166340


namespace NUMINAMATH_CALUDE_andy_total_distance_l1663_166355

/-- The total distance Andy walks given his trips to school and market -/
def total_distance (house_to_school market_to_house : ℕ) : ℕ :=
  2 * house_to_school + market_to_house

/-- Theorem stating the total distance Andy walks -/
theorem andy_total_distance :
  let house_to_school := 50
  let house_to_market := 40
  total_distance house_to_school house_to_market = 140 := by
  sorry

end NUMINAMATH_CALUDE_andy_total_distance_l1663_166355


namespace NUMINAMATH_CALUDE_particular_number_calculation_l1663_166373

theorem particular_number_calculation (x : ℝ) (h : 2.5 * x - 2.49 = 22.01) :
  (x / 2.5) + 2.49 + 22.01 = 28.42 := by
sorry

end NUMINAMATH_CALUDE_particular_number_calculation_l1663_166373


namespace NUMINAMATH_CALUDE_highest_power_equals_carries_l1663_166354

/-- The number of carries when adding two natural numbers in a given base. -/
def num_carries (m n p : ℕ) : ℕ := sorry

/-- The highest power of p that divides the binomial coefficient (n+m choose m). -/
def highest_power_dividing_binom (n m p : ℕ) : ℕ := sorry

/-- Theorem stating the relationship between the highest power of p dividing
    (n+m choose m) and the number of carries when adding m and n in base p. -/
theorem highest_power_equals_carries (p m n : ℕ) (hp : Nat.Prime p) :
  highest_power_dividing_binom n m p = num_carries m n p :=
sorry

end NUMINAMATH_CALUDE_highest_power_equals_carries_l1663_166354


namespace NUMINAMATH_CALUDE_parabola_vertex_l1663_166323

-- Define the parabola
def f (x : ℝ) : ℝ := -3 * (x + 1)^2 + 1

-- State the theorem
theorem parabola_vertex : 
  ∃ (x y : ℝ), (∀ t : ℝ, f t ≤ f x) ∧ y = f x ∧ x = -1 ∧ y = 1 :=
sorry

end NUMINAMATH_CALUDE_parabola_vertex_l1663_166323


namespace NUMINAMATH_CALUDE_f_neg_nine_eq_neg_one_l1663_166357

-- Define the logarithm base 10 function
noncomputable def lg (x : ℝ) : ℝ := Real.log x / Real.log 10

-- Define the properties of the function f
def is_valid_f (f : ℝ → ℝ) : Prop :=
  ∃ b : ℝ, 
    (∀ x : ℝ, f (-x) = -f x) ∧ 
    (∀ x : ℝ, x ≥ 0 → f x = lg (x + 1) - b)

-- State the theorem
theorem f_neg_nine_eq_neg_one (f : ℝ → ℝ) (h : is_valid_f f) : f (-9) = -1 := by
  sorry

end NUMINAMATH_CALUDE_f_neg_nine_eq_neg_one_l1663_166357


namespace NUMINAMATH_CALUDE_floor_negative_two_point_eight_l1663_166377

theorem floor_negative_two_point_eight :
  ⌊(-2.8 : ℝ)⌋ = -3 := by sorry

end NUMINAMATH_CALUDE_floor_negative_two_point_eight_l1663_166377


namespace NUMINAMATH_CALUDE_diophantine_equation_solution_l1663_166366

theorem diophantine_equation_solution :
  ∀ (x y : ℤ), 3 * x + 5 * y = 7 ↔ ∃ k : ℤ, x = 4 + 5 * k ∧ y = -1 - 3 * k :=
by sorry

end NUMINAMATH_CALUDE_diophantine_equation_solution_l1663_166366


namespace NUMINAMATH_CALUDE_arithmetic_sequence_property_l1663_166381

-- Define an arithmetic sequence
def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

-- State the theorem
theorem arithmetic_sequence_property (a : ℕ → ℝ) :
  is_arithmetic_sequence a → a 2 + a 8 = 12 → a 5 = 6 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_property_l1663_166381


namespace NUMINAMATH_CALUDE_average_after_exclusion_l1663_166342

theorem average_after_exclusion (numbers : Finset ℕ) (sum : ℕ) (excluded : ℕ) :
  numbers.card = 5 →
  sum / numbers.card = 27 →
  excluded ∈ numbers →
  excluded = 35 →
  (sum - excluded) / (numbers.card - 1) = 25 := by
  sorry

end NUMINAMATH_CALUDE_average_after_exclusion_l1663_166342


namespace NUMINAMATH_CALUDE_intersecting_squares_area_difference_l1663_166361

theorem intersecting_squares_area_difference :
  let A : ℝ := 12^2
  let B : ℝ := 9^2
  let C : ℝ := 7^2
  let D : ℝ := 3^2
  ∀ (E F G : ℝ),
  (A + E - (B + F)) - (C + G - (B + D + F)) = 103 :=
by sorry

end NUMINAMATH_CALUDE_intersecting_squares_area_difference_l1663_166361


namespace NUMINAMATH_CALUDE_absolute_value_equation_implies_zero_product_l1663_166335

theorem absolute_value_equation_implies_zero_product (x y : ℝ) (hy : y > 0) :
  |x - Real.log (y^2)| = x + Real.log (y^2) → x * (y - 1)^2 = 0 := by
  sorry

end NUMINAMATH_CALUDE_absolute_value_equation_implies_zero_product_l1663_166335


namespace NUMINAMATH_CALUDE_arithmetic_calculation_l1663_166398

theorem arithmetic_calculation : ((55 * 45 - 37 * 43) - (3 * 221 + 1)) / 22 = 10 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_calculation_l1663_166398


namespace NUMINAMATH_CALUDE_carbon_neutral_olympics_emissions_l1663_166301

theorem carbon_neutral_olympics_emissions (emissions : ℝ) : 
  emissions = 320000 → emissions = 3.2 * (10 ^ 5) := by
  sorry

end NUMINAMATH_CALUDE_carbon_neutral_olympics_emissions_l1663_166301


namespace NUMINAMATH_CALUDE_sum_number_and_square_l1663_166374

/-- If a number is 16, then the sum of this number and its square is 272. -/
theorem sum_number_and_square (x : ℕ) : x = 16 → x + x^2 = 272 := by
  sorry

end NUMINAMATH_CALUDE_sum_number_and_square_l1663_166374


namespace NUMINAMATH_CALUDE_cube_root_simplification_l1663_166306

theorem cube_root_simplification : Real.rpow (2^9 * 5^3 * 7^3) (1/3) = 280 := by
  sorry

end NUMINAMATH_CALUDE_cube_root_simplification_l1663_166306


namespace NUMINAMATH_CALUDE_range_of_product_l1663_166371

theorem range_of_product (x y z : ℝ) 
  (hx : -3 < x) (hxy : x < y) (hy : y < 1) 
  (hz1 : -4 < z) (hz2 : z < 0) : 
  0 < (x - y) * z ∧ (x - y) * z < 16 := by
  sorry

end NUMINAMATH_CALUDE_range_of_product_l1663_166371


namespace NUMINAMATH_CALUDE_initial_acidic_percentage_l1663_166346

/-- Proves that the initial percentage of acidic liquid is 40% given the conditions -/
theorem initial_acidic_percentage (initial_volume : ℝ) (final_concentration : ℝ) (water_removed : ℝ) : 
  initial_volume = 18 →
  final_concentration = 60 →
  water_removed = 6 →
  (initial_volume * (40 / 100) = (initial_volume - water_removed) * (final_concentration / 100)) :=
by sorry

end NUMINAMATH_CALUDE_initial_acidic_percentage_l1663_166346


namespace NUMINAMATH_CALUDE_ratio_odd_even_divisors_l1663_166330

def N : ℕ := 34 * 34 * 63 * 270

def sum_odd_divisors (n : ℕ) : ℕ := sorry

def sum_even_divisors (n : ℕ) : ℕ := sorry

theorem ratio_odd_even_divisors :
  (sum_odd_divisors N : ℚ) / (sum_even_divisors N : ℚ) = 1 / 14 := by sorry

end NUMINAMATH_CALUDE_ratio_odd_even_divisors_l1663_166330


namespace NUMINAMATH_CALUDE_min_reciprocal_sum_min_reciprocal_sum_achievable_l1663_166394

theorem min_reciprocal_sum (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : 2 * x + y = 1) :
  (1 / x + 1 / y) ≥ 3 + 2 * Real.sqrt 2 := by
  sorry

theorem min_reciprocal_sum_achievable :
  ∃ x y : ℝ, x > 0 ∧ y > 0 ∧ 2 * x + y = 1 ∧ 1 / x + 1 / y = 3 + 2 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_min_reciprocal_sum_min_reciprocal_sum_achievable_l1663_166394


namespace NUMINAMATH_CALUDE_frog_jump_probability_l1663_166376

/-- Represents a point in the 2D plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents the square garden -/
def Garden :=
  {p : Point | 0 ≤ p.x ∧ p.x ≤ 5 ∧ 0 ≤ p.y ∧ p.y ≤ 5}

/-- Possible jump directions -/
inductive Direction
  | Up
  | Down
  | Left
  | Right

/-- Represents a frog jump -/
def jump (p : Point) (d : Direction) : Point :=
  match d with
  | Direction.Up => ⟨p.x, p.y + 1⟩
  | Direction.Down => ⟨p.x, p.y - 1⟩
  | Direction.Left => ⟨p.x - 1, p.y⟩
  | Direction.Right => ⟨p.x + 1, p.y⟩

/-- Checks if a point is on the vertical sides of the garden -/
def isOnVerticalSide (p : Point) : Prop :=
  (p.x = 0 ∨ p.x = 5) ∧ 0 ≤ p.y ∧ p.y ≤ 5

/-- The probability of ending on a vertical side from a given point -/
noncomputable def probabilityVerticalSide (p : Point) : ℝ := sorry

/-- The theorem to be proved -/
theorem frog_jump_probability :
  probabilityVerticalSide ⟨2, 1⟩ = 13 / 20 := by sorry

end NUMINAMATH_CALUDE_frog_jump_probability_l1663_166376


namespace NUMINAMATH_CALUDE_output_is_six_l1663_166334

def program_output (a : ℕ) : ℕ :=
  if a < 10 then 2 * a else a * a

theorem output_is_six : program_output 3 = 6 := by
  sorry

end NUMINAMATH_CALUDE_output_is_six_l1663_166334


namespace NUMINAMATH_CALUDE_part_one_part_two_l1663_166395

-- Define the sets A and B
def A (a : ℝ) : Set ℝ := {x | (x - 2) / (x - (3 * a + 1)) < 0}
def B (a : ℝ) : Set ℝ := {x | (x - a^2 - 2) / (x - a) < 0}

-- Part 1
theorem part_one : 
  (Set.univ \ B (1/2)) ∩ A (1/2) = {x : ℝ | 9/4 ≤ x ∧ x < 5/2} := by sorry

-- Part 2
theorem part_two : 
  ∀ a : ℝ, (∀ x : ℝ, x ∈ A a → x ∈ B a) ↔ a ∈ Set.Icc (-1/2) ((3 - Real.sqrt 5) / 2) := by sorry

end NUMINAMATH_CALUDE_part_one_part_two_l1663_166395


namespace NUMINAMATH_CALUDE_book_price_problem_l1663_166347

theorem book_price_problem (n : ℕ) (d : ℝ) (middle_price : ℝ) : 
  n = 40 → d = 3 → middle_price = 75 → 
  ∃ (first_price : ℝ), 
    (∀ i : ℕ, i ≤ n → 
      (first_price + d * (i - 1) = middle_price) ↔ i = n / 2) ∧
    first_price = 18 :=
by sorry

end NUMINAMATH_CALUDE_book_price_problem_l1663_166347


namespace NUMINAMATH_CALUDE_rectangle_diagonal_l1663_166313

theorem rectangle_diagonal (a b : ℝ) (h_perimeter : 2 * (a + b) = 178) (h_area : a * b = 1848) :
  Real.sqrt (a^2 + b^2) = 65 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_diagonal_l1663_166313


namespace NUMINAMATH_CALUDE_inequality_range_l1663_166351

theorem inequality_range (m : ℝ) : 
  (∀ x y : ℝ, 3 * x^2 + y^2 ≥ m * x * (x + y)) → 
  m ∈ Set.Icc (-6) 2 := by
sorry

end NUMINAMATH_CALUDE_inequality_range_l1663_166351


namespace NUMINAMATH_CALUDE_b_share_is_correct_l1663_166380

/-- Represents the rental information for a person -/
structure RentalInfo where
  horses : ℕ
  months : ℕ

/-- Calculates the total horse-months for a given rental information -/
def horseMonths (info : RentalInfo) : ℕ :=
  info.horses * info.months

/-- Represents the pasture rental problem -/
structure PastureRental where
  totalRent : ℚ
  a : RentalInfo
  b : RentalInfo
  c : RentalInfo

/-- Calculates the total horse-months for all renters -/
def totalHorseMonths (rental : PastureRental) : ℕ :=
  horseMonths rental.a + horseMonths rental.b + horseMonths rental.c

/-- Calculates the rent per horse-month -/
def rentPerHorseMonth (rental : PastureRental) : ℚ :=
  rental.totalRent / totalHorseMonths rental

/-- Calculates the rent for a specific renter -/
def renterShare (rental : PastureRental) (renter : RentalInfo) : ℚ :=
  (rentPerHorseMonth rental) * (horseMonths renter)

/-- The main theorem stating b's share of the rent -/
theorem b_share_is_correct (rental : PastureRental) 
  (h1 : rental.totalRent = 841)
  (h2 : rental.a = ⟨12, 8⟩)
  (h3 : rental.b = ⟨16, 9⟩)
  (h4 : rental.c = ⟨18, 6⟩) :
  renterShare rental rental.b = 348.48 := by
  sorry

end NUMINAMATH_CALUDE_b_share_is_correct_l1663_166380


namespace NUMINAMATH_CALUDE_tan_x_eq_2_implies_expression_l1663_166387

theorem tan_x_eq_2_implies_expression (x : ℝ) (h : Real.tan x = 2) :
  (6 * Real.sin (2 * x) + 2 * Real.cos (2 * x)) / (Real.cos (2 * x) - 3 * Real.sin (2 * x)) = -2/5 := by
  sorry

end NUMINAMATH_CALUDE_tan_x_eq_2_implies_expression_l1663_166387


namespace NUMINAMATH_CALUDE_complex_magnitude_problem_l1663_166389

theorem complex_magnitude_problem (z : ℂ) (h : Complex.I * z = 2 + 4 * Complex.I) :
  Complex.abs z = 2 * Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_complex_magnitude_problem_l1663_166389


namespace NUMINAMATH_CALUDE_platform_length_calculation_l1663_166307

/-- Calculates the length of a platform given train parameters --/
theorem platform_length_calculation (train_length : ℝ) (train_speed_kmph : ℝ) (crossing_time : ℝ) : 
  train_length = 180 ∧ 
  train_speed_kmph = 72 ∧ 
  crossing_time = 20 →
  (train_speed_kmph * 1000 / 3600) * crossing_time - train_length = 220 := by
  sorry

#check platform_length_calculation

end NUMINAMATH_CALUDE_platform_length_calculation_l1663_166307


namespace NUMINAMATH_CALUDE_intersection_sum_l1663_166339

/-- Given two lines y = mx + 3 and y = 4x + b intersecting at (8, 14),
    where m and b are constants, prove that b + m = -133/8 -/
theorem intersection_sum (m b : ℚ) : 
  (∀ x y : ℚ, y = m * x + 3 ↔ y = 4 * x + b) → 
  (14 : ℚ) = m * 8 + 3 → 
  (14 : ℚ) = 4 * 8 + b → 
  b + m = -133/8 := by sorry

end NUMINAMATH_CALUDE_intersection_sum_l1663_166339


namespace NUMINAMATH_CALUDE_negation_of_all_not_divisible_by_two_are_odd_l1663_166303

theorem negation_of_all_not_divisible_by_two_are_odd :
  (¬ ∀ n : ℤ, ¬(2 ∣ n) → Odd n) ↔ (∃ n : ℤ, ¬(2 ∣ n) ∧ ¬(Odd n)) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_all_not_divisible_by_two_are_odd_l1663_166303


namespace NUMINAMATH_CALUDE_robin_gum_packages_l1663_166375

/-- Represents the number of pieces of gum in each package -/
def pieces_per_package : ℕ := 23

/-- Represents the number of extra pieces of gum Robin has -/
def extra_pieces : ℕ := 8

/-- Represents the total number of pieces of gum Robin has -/
def total_pieces : ℕ := 997

/-- Represents the number of packages Robin has -/
def num_packages : ℕ := (total_pieces - extra_pieces) / pieces_per_package

theorem robin_gum_packages : num_packages = 43 := by
  sorry

end NUMINAMATH_CALUDE_robin_gum_packages_l1663_166375


namespace NUMINAMATH_CALUDE_arithmetic_progression_sum_dependency_l1663_166358

/-- Given an arithmetic progression with first term a and common difference d,
    s₁, s₂, and s₄ are the sums of n, 2n, and 4n terms respectively.
    R is defined as s₄ - s₂ - s₁. -/
theorem arithmetic_progression_sum_dependency
  (n : ℕ) (a d : ℝ) 
  (s₁ : ℝ := n * (2 * a + (n - 1) * d) / 2)
  (s₂ : ℝ := 2 * n * (2 * a + (2 * n - 1) * d) / 2)
  (s₄ : ℝ := 4 * n * (2 * a + (4 * n - 1) * d) / 2)
  (R : ℝ := s₄ - s₂ - s₁) :
  R = 6 * d * n^2 := by sorry

end NUMINAMATH_CALUDE_arithmetic_progression_sum_dependency_l1663_166358


namespace NUMINAMATH_CALUDE_first_group_size_l1663_166350

/-- The number of persons in the first group -/
def P : ℕ := 42

/-- The number of days the first group works -/
def days_first : ℕ := 12

/-- The number of hours per day the first group works -/
def hours_first : ℕ := 5

/-- The number of persons in the second group -/
def persons_second : ℕ := 30

/-- The number of days the second group works -/
def days_second : ℕ := 14

/-- The number of hours per day the second group works -/
def hours_second : ℕ := 6

/-- Theorem stating that P is the correct number of persons in the first group -/
theorem first_group_size :
  P * days_first * hours_first = persons_second * days_second * hours_second :=
by sorry

end NUMINAMATH_CALUDE_first_group_size_l1663_166350


namespace NUMINAMATH_CALUDE_encyclopedia_sorting_l1663_166384

/-- Represents the number of volumes in the encyclopedia --/
def n : ℕ := 30

/-- Represents an operation of swapping two adjacent volumes --/
def swap : ℕ → ℕ → List ℕ → List ℕ := sorry

/-- Checks if a list of volumes is in the correct order --/
def is_sorted : List ℕ → Prop := sorry

/-- The maximum number of disorders in any arrangement of n volumes --/
def max_disorders (n : ℕ) : ℕ := n * (n - 1) / 2

/-- The minimum number of operations required to sort n volumes --/
def min_operations (n : ℕ) : ℕ := max_disorders n

theorem encyclopedia_sorting (arrangement : List ℕ) 
  (h : arrangement.length = n) :
  ∃ (sequence : List (ℕ × ℕ)), 
    sequence.length ≤ min_operations n ∧ 
    is_sorted (sequence.foldl (λ acc (i, j) => swap i j acc) arrangement) := by
  sorry

#eval min_operations n  -- Should evaluate to 435

end NUMINAMATH_CALUDE_encyclopedia_sorting_l1663_166384


namespace NUMINAMATH_CALUDE_existence_of_three_quadratic_polynomials_l1663_166379

theorem existence_of_three_quadratic_polynomials :
  ∃ (p₁ p₂ p₃ : ℝ → ℝ),
    (∃ x₁, p₁ x₁ = 0) ∧
    (∃ x₂, p₂ x₂ = 0) ∧
    (∃ x₃, p₃ x₃ = 0) ∧
    (∀ x, p₁ x + p₂ x ≠ 0) ∧
    (∀ x, p₁ x + p₃ x ≠ 0) ∧
    (∀ x, p₂ x + p₃ x ≠ 0) ∧
    (∀ x, p₁ x = (x^2 : ℝ)) ∧
    (∀ x, p₂ x = ((x - 1)^2 : ℝ)) ∧
    (∀ x, p₃ x = ((x - 2)^2 : ℝ)) :=
by sorry

end NUMINAMATH_CALUDE_existence_of_three_quadratic_polynomials_l1663_166379


namespace NUMINAMATH_CALUDE_square_comparison_l1663_166369

theorem square_comparison (a b : ℝ) (h1 : a + b < 0) (h2 : b > 0) : a^2 > b^2 := by
  sorry

end NUMINAMATH_CALUDE_square_comparison_l1663_166369


namespace NUMINAMATH_CALUDE_y_derivative_l1663_166308

open Real

noncomputable def y (x : ℝ) : ℝ :=
  (6^x * (sin (4*x) * log 6 - 4 * cos (4*x))) / (16 + (log 6)^2)

theorem y_derivative (x : ℝ) : 
  deriv y x = 6^x * sin (4*x) :=
sorry

end NUMINAMATH_CALUDE_y_derivative_l1663_166308


namespace NUMINAMATH_CALUDE_caesars_meal_cost_proof_l1663_166314

/-- The cost per meal at Caesar's banquet hall -/
def caesars_meal_cost : ℝ := 30

/-- The number of guests attending the prom -/
def num_guests : ℕ := 60

/-- Caesar's room rental fee -/
def caesars_room_fee : ℝ := 800

/-- Venus Hall's room rental fee -/
def venus_room_fee : ℝ := 500

/-- Venus Hall's cost per meal -/
def venus_meal_cost : ℝ := 35

theorem caesars_meal_cost_proof :
  caesars_room_fee + num_guests * caesars_meal_cost =
  venus_room_fee + num_guests * venus_meal_cost :=
by sorry

end NUMINAMATH_CALUDE_caesars_meal_cost_proof_l1663_166314


namespace NUMINAMATH_CALUDE_ben_win_probability_l1663_166315

theorem ben_win_probability (lose_prob : ℚ) (h1 : lose_prob = 5/8) (h2 : ¬ ∃ tie_prob : ℚ, tie_prob ≠ 0) :
  1 - lose_prob = 3/8 :=
by sorry

end NUMINAMATH_CALUDE_ben_win_probability_l1663_166315


namespace NUMINAMATH_CALUDE_certain_number_proof_l1663_166300

theorem certain_number_proof : ∃ x : ℝ, (x / 3 = 400 * 1.005) ∧ (x = 1206) := by
  sorry

end NUMINAMATH_CALUDE_certain_number_proof_l1663_166300


namespace NUMINAMATH_CALUDE_number_equation_proof_l1663_166364

theorem number_equation_proof (n : ℤ) : n - 8 = 5 * 7 + 12 ↔ n = 55 := by
  sorry

end NUMINAMATH_CALUDE_number_equation_proof_l1663_166364


namespace NUMINAMATH_CALUDE_min_abs_z_plus_i_l1663_166331

theorem min_abs_z_plus_i (z : ℂ) (h : Complex.abs (z^2 + 9) = Complex.abs (z * (z + 3*I))) :
  ∃ (w : ℂ), Complex.abs (w + I) = 2 ∧ ∀ (z : ℂ), Complex.abs (z^2 + 9) = Complex.abs (z * (z + 3*I)) → Complex.abs (z + I) ≥ 2 :=
sorry

end NUMINAMATH_CALUDE_min_abs_z_plus_i_l1663_166331


namespace NUMINAMATH_CALUDE_expression_value_l1663_166304

theorem expression_value : 
  2 * Real.tan (60 * π / 180) - (1/3)⁻¹ + (-2)^2 * (2017 - Real.sin (45 * π / 180))^0 - |-(12: ℝ).sqrt| = 1 := by
  sorry

end NUMINAMATH_CALUDE_expression_value_l1663_166304


namespace NUMINAMATH_CALUDE_old_supervisor_salary_l1663_166312

/-- Proves that the old supervisor's salary was $870 given the problem conditions -/
theorem old_supervisor_salary
  (num_workers : ℕ)
  (initial_average : ℚ)
  (new_average : ℚ)
  (new_supervisor_salary : ℚ)
  (h_num_workers : num_workers = 8)
  (h_initial_average : initial_average = 430)
  (h_new_average : new_average = 390)
  (h_new_supervisor_salary : new_supervisor_salary = 510)
  : ∃ (old_supervisor_salary : ℚ),
    (num_workers + 1) * initial_average = num_workers * new_average + old_supervisor_salary
    ∧ old_supervisor_salary = 870 :=
by sorry

end NUMINAMATH_CALUDE_old_supervisor_salary_l1663_166312


namespace NUMINAMATH_CALUDE_stratified_sampling_most_appropriate_school_staff_sampling_l1663_166309

/-- Represents a sampling method -/
inductive SamplingMethod
  | SimpleRandom
  | Systematic
  | Stratified
  | Other

/-- Represents a population with subgroups -/
structure Population where
  total : ℕ
  subgroups : List ℕ
  h_sum : total = subgroups.sum

/-- Represents a sample -/
structure Sample where
  size : ℕ
  method : SamplingMethod

/-- Determines if a sample is representative of a population -/
def is_representative (pop : Population) (samp : Sample) : Prop :=
  samp.method = SamplingMethod.Stratified ∧ pop.subgroups.length > 1

/-- The main theorem stating that stratified sampling is most appropriate for a population with subgroups -/
theorem stratified_sampling_most_appropriate (pop : Population) (samp : Sample) 
    (h_subgroups : pop.subgroups.length > 1) : 
    is_representative pop samp ↔ samp.method = SamplingMethod.Stratified :=
  sorry

/-- The specific instance from the problem -/
def school_staff : Population :=
  { total := 160
  , subgroups := [120, 16, 24]
  , h_sum := by simp }

def staff_sample : Sample :=
  { size := 20
  , method := SamplingMethod.Stratified }

/-- The theorem applied to the specific instance -/
theorem school_staff_sampling : 
    is_representative school_staff staff_sample :=
  sorry

end NUMINAMATH_CALUDE_stratified_sampling_most_appropriate_school_staff_sampling_l1663_166309


namespace NUMINAMATH_CALUDE_ellipse_properties_and_max_area_l1663_166397

/-- An ellipse with given properties -/
structure Ellipse where
  a : ℝ
  b : ℝ
  ecc : ℝ
  h_a_pos : a > 0
  h_b_pos : b > 0
  h_a_gt_b : a > b
  h_ecc : ecc = 2 * Real.sqrt 2 / 3
  h_vertex : b = 1

/-- A line intersecting the ellipse -/
structure IntersectingLine (E : Ellipse) where
  k : ℝ
  m : ℝ

/-- Triangle formed by intersection points and vertex -/
def triangle_area (E : Ellipse) (l : IntersectingLine E) : ℝ := sorry

/-- Theorem stating the properties of the ellipse and maximum triangle area -/
theorem ellipse_properties_and_max_area (E : Ellipse) :
  E.a = 3 ∧
  (∃ (l : IntersectingLine E), ∀ (l' : IntersectingLine E),
    triangle_area E l ≥ triangle_area E l') ∧
  (∃ (l : IntersectingLine E), triangle_area E l = 27/8) :=
sorry

end NUMINAMATH_CALUDE_ellipse_properties_and_max_area_l1663_166397


namespace NUMINAMATH_CALUDE_triangle_radii_relation_l1663_166385

/-- Given a triangle ABC with sides a, b, c, inradius r, circumradius R, and exradii rA, rB, rC,
    prove the relation between these quantities. -/
theorem triangle_radii_relation (a b c r R rA rB rC : ℝ) : 
  a^2 * (2/rA - r/(rB*rC)) + b^2 * (2/rB - r/(rA*rC)) + c^2 * (2/rC - r/(rA*rB)) = 4*(R + 3*r) :=
by sorry

end NUMINAMATH_CALUDE_triangle_radii_relation_l1663_166385


namespace NUMINAMATH_CALUDE_jack_marathon_time_l1663_166349

/-- Proves that Jack's marathon time is 5 hours given the specified conditions -/
theorem jack_marathon_time
  (marathon_distance : ℝ)
  (jill_time : ℝ)
  (speed_ratio : ℝ)
  (h1 : marathon_distance = 42)
  (h2 : jill_time = 4.2)
  (h3 : speed_ratio = 0.8400000000000001)
  : ℝ :=
by
  sorry

#check jack_marathon_time

end NUMINAMATH_CALUDE_jack_marathon_time_l1663_166349
