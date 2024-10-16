import Mathlib

namespace NUMINAMATH_CALUDE_ab_plus_cd_equals_twelve_l2193_219331

theorem ab_plus_cd_equals_twelve 
  (a b c d : ℝ) 
  (h1 : a + b + c = 3)
  (h2 : a + b + d = -1)
  (h3 : a + c + d = 8)
  (h4 : b + c + d = 5) :
  a * b + c * d = 12 := by
  sorry

end NUMINAMATH_CALUDE_ab_plus_cd_equals_twelve_l2193_219331


namespace NUMINAMATH_CALUDE_sin_30_degrees_l2193_219357

theorem sin_30_degrees : Real.sin (30 * π / 180) = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_sin_30_degrees_l2193_219357


namespace NUMINAMATH_CALUDE_triangle_area_range_l2193_219372

/-- Given an obtuse-angled triangle ABC with side c = 2 and angle B = π/3,
    the area S of the triangle satisfies: S ∈ (0, √3/2) ∪ (2√3, +∞) -/
theorem triangle_area_range (A B C : ℝ) (a b c : ℝ) (S : ℝ) :
  0 < A ∧ 0 < B ∧ 0 < C ∧  -- Angles are positive
  A + B + C = π ∧  -- Sum of angles in a triangle
  c = 2 ∧  -- Given condition
  B = π / 3 ∧  -- Given condition
  (A > π / 2 ∨ B > π / 2 ∨ C > π / 2) ∧  -- Obtuse-angled triangle condition
  S = (1 / 2) * a * c * Real.sin B →  -- Area formula
  S ∈ Set.Ioo 0 (Real.sqrt 3 / 2) ∪ Set.Ioi (2 * Real.sqrt 3) :=
by
  sorry

end NUMINAMATH_CALUDE_triangle_area_range_l2193_219372


namespace NUMINAMATH_CALUDE_a_b_reciprocals_l2193_219392

theorem a_b_reciprocals (a b : ℝ) : 
  a = 1 / (2 - Real.sqrt 3) → 
  b = 1 / (2 + Real.sqrt 3) → 
  a * b = 1 := by
sorry

end NUMINAMATH_CALUDE_a_b_reciprocals_l2193_219392


namespace NUMINAMATH_CALUDE_area_max_cyclic_l2193_219373

/-- A quadrilateral with sides a, b, c, d and diagonals e, f -/
structure Quadrilateral (α : Type*) [LinearOrderedField α] :=
  (a b c d e f : α)

/-- The area of a quadrilateral -/
def area {α : Type*} [LinearOrderedField α] (q : Quadrilateral α) : α :=
  ((q.b + q.d - q.a + q.c) * (q.b + q.d + q.a - q.c) * 
   (q.a + q.c - q.b + q.d) * (q.a + q.b + q.c - q.d) - 
   4 * (q.a * q.c + q.b * q.d - q.e * q.f) * (q.a * q.c + q.b * q.d + q.e * q.f)) / 16

/-- The theorem stating that the area is maximized when ef = ac + bd -/
theorem area_max_cyclic {α : Type*} [LinearOrderedField α] (q : Quadrilateral α) :
  area q ≤ area { q with e := (q.a * q.c + q.b * q.d) / q.f, f := q.f } :=
sorry

end NUMINAMATH_CALUDE_area_max_cyclic_l2193_219373


namespace NUMINAMATH_CALUDE_sufficient_condition_quadratic_l2193_219385

theorem sufficient_condition_quadratic (a : ℝ) :
  a ≥ 0 → ∃ x : ℝ, a * x^2 + x + 1 ≥ 0 := by
  sorry

end NUMINAMATH_CALUDE_sufficient_condition_quadratic_l2193_219385


namespace NUMINAMATH_CALUDE_sunday_max_available_l2193_219319

-- Define the days of the week
inductive Day
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday
  | Saturday
  | Sunday

-- Define the team members
inductive Member
  | Alice
  | Bob
  | Cara
  | Dave
  | Ella

-- Define a function to represent the availability of each member on each day
def isAvailable (m : Member) (d : Day) : Bool :=
  match m, d with
  | Member.Alice, Day.Monday => false
  | Member.Alice, Day.Thursday => false
  | Member.Alice, Day.Saturday => false
  | Member.Bob, Day.Tuesday => false
  | Member.Bob, Day.Wednesday => false
  | Member.Bob, Day.Friday => false
  | Member.Cara, Day.Monday => false
  | Member.Cara, Day.Tuesday => false
  | Member.Cara, Day.Thursday => false
  | Member.Cara, Day.Saturday => false
  | Member.Cara, Day.Sunday => false
  | Member.Dave, Day.Wednesday => false
  | Member.Dave, Day.Saturday => false
  | Member.Ella, Day.Monday => false
  | Member.Ella, Day.Friday => false
  | Member.Ella, Day.Saturday => false
  | _, _ => true

-- Define a function to count the number of available members on a given day
def countAvailable (d : Day) : Nat :=
  (List.filter (fun m => isAvailable m d) [Member.Alice, Member.Bob, Member.Cara, Member.Dave, Member.Ella]).length

-- Theorem: Sunday has the maximum number of available team members
theorem sunday_max_available :
  ∀ d : Day, countAvailable Day.Sunday ≥ countAvailable d := by
  sorry


end NUMINAMATH_CALUDE_sunday_max_available_l2193_219319


namespace NUMINAMATH_CALUDE_additional_amount_for_free_shipping_l2193_219368

-- Define the book prices and discount
def book1_price : ℝ := 13.00
def book2_price : ℝ := 15.00
def book3_price : ℝ := 10.00
def book4_price : ℝ := 10.00
def discount_rate : ℝ := 0.25
def free_shipping_threshold : ℝ := 50.00

-- Calculate the discounted prices for books 1 and 2
def discounted_book1_price : ℝ := book1_price * (1 - discount_rate)
def discounted_book2_price : ℝ := book2_price * (1 - discount_rate)

-- Calculate the total cost of all four books
def total_cost : ℝ := discounted_book1_price + discounted_book2_price + book3_price + book4_price

-- Define the additional amount needed for free shipping
def additional_amount : ℝ := free_shipping_threshold - total_cost

-- Theorem to prove
theorem additional_amount_for_free_shipping :
  additional_amount = 9.00 := by sorry

end NUMINAMATH_CALUDE_additional_amount_for_free_shipping_l2193_219368


namespace NUMINAMATH_CALUDE_decimal_to_fraction_l2193_219310

theorem decimal_to_fraction (x : ℚ) : x = 0.38 → x = 19/50 := by
  sorry

end NUMINAMATH_CALUDE_decimal_to_fraction_l2193_219310


namespace NUMINAMATH_CALUDE_power_inequality_l2193_219336

theorem power_inequality (a b : ℝ) (h1 : 0 < a) (h2 : a < b) (h3 : b < 1) :
  a^a < b^a := by sorry

end NUMINAMATH_CALUDE_power_inequality_l2193_219336


namespace NUMINAMATH_CALUDE_polynomial_factorization_l2193_219342

theorem polynomial_factorization (a b : ℝ) (x : ℝ) :
  a + (a+b)*x + (a+2*b)*x^2 + (a+3*b)*x^3 + 3*b*x^4 + 2*b*x^5 + b*x^6 = 
  (1 + x) * (1 + x^2) * (a + b*x + b*x^2 + b*x^3) := by
  sorry

end NUMINAMATH_CALUDE_polynomial_factorization_l2193_219342


namespace NUMINAMATH_CALUDE_grading_multiple_proof_l2193_219329

/-- Given a grading method that subtracts a multiple of incorrect responses
    from correct responses, prove that the multiple is 2 for a specific case. -/
theorem grading_multiple_proof (total_questions : ℕ) (correct_responses : ℕ) (score : ℕ) :
  total_questions = 100 →
  correct_responses = 87 →
  score = 61 →
  ∃ (m : ℚ), score = correct_responses - m * (total_questions - correct_responses) →
  m = 2 := by
  sorry

end NUMINAMATH_CALUDE_grading_multiple_proof_l2193_219329


namespace NUMINAMATH_CALUDE_chicken_nuggets_distribution_l2193_219399

theorem chicken_nuggets_distribution (total : ℕ) (alyssa : ℕ) : 
  total = 100 → alyssa + 2 * alyssa + 2 * alyssa = total → alyssa = 20 := by
  sorry

end NUMINAMATH_CALUDE_chicken_nuggets_distribution_l2193_219399


namespace NUMINAMATH_CALUDE_lisa_tommy_earnings_difference_l2193_219346

-- Define the earnings for each person
def sophia_earnings : ℕ := 10 + 15 + 25
def sarah_earnings : ℕ := 15 + 10 + 20 + 20
def lisa_earnings : ℕ := 20 + 30
def jack_earnings : ℕ := 10 + 10 + 10 + 15 + 15
def tommy_earnings : ℕ := 5 + 5 + 10 + 10

-- Define the total earnings
def total_earnings : ℕ := 180

-- Theorem statement
theorem lisa_tommy_earnings_difference :
  lisa_earnings - tommy_earnings = 20 :=
sorry

end NUMINAMATH_CALUDE_lisa_tommy_earnings_difference_l2193_219346


namespace NUMINAMATH_CALUDE_cube_surface_area_l2193_219364

/-- Given a cube with volume 1728 cubic centimeters, its surface area is 864 square centimeters. -/
theorem cube_surface_area (v : ℝ) (h : v = 1728) : 
  (6 * ((v ^ (1/3)) ^ 2)) = 864 :=
sorry

end NUMINAMATH_CALUDE_cube_surface_area_l2193_219364


namespace NUMINAMATH_CALUDE_days_to_pay_for_register_is_8_l2193_219345

/-- The number of days required to pay for a cash register given daily sales and costs -/
def days_to_pay_for_register (register_cost : ℕ) (bread_sold : ℕ) (bread_price : ℕ) 
  (cakes_sold : ℕ) (cake_price : ℕ) (rent : ℕ) (electricity : ℕ) : ℕ :=
  let daily_revenue := bread_sold * bread_price + cakes_sold * cake_price
  let daily_expenses := rent + electricity
  let daily_profit := daily_revenue - daily_expenses
  (register_cost + daily_profit - 1) / daily_profit

theorem days_to_pay_for_register_is_8 :
  days_to_pay_for_register 1040 40 2 6 12 20 2 = 8 := by sorry

end NUMINAMATH_CALUDE_days_to_pay_for_register_is_8_l2193_219345


namespace NUMINAMATH_CALUDE_certain_number_proof_l2193_219378

theorem certain_number_proof (k : ℝ) (certain_number : ℝ) 
  (h1 : 24 / k = certain_number) 
  (h2 : k = 6) : 
  certain_number = 4 := by
  sorry

end NUMINAMATH_CALUDE_certain_number_proof_l2193_219378


namespace NUMINAMATH_CALUDE_chromium_percentage_proof_l2193_219398

/-- The percentage of chromium in the second alloy -/
def chromium_percentage_second_alloy : ℝ := 8

theorem chromium_percentage_proof (
  first_alloy_chromium_percentage : ℝ)
  (first_alloy_weight : ℝ)
  (second_alloy_weight : ℝ)
  (new_alloy_chromium_percentage : ℝ)
  (h1 : first_alloy_chromium_percentage = 15)
  (h2 : first_alloy_weight = 15)
  (h3 : second_alloy_weight = 35)
  (h4 : new_alloy_chromium_percentage = 10.1)
  : chromium_percentage_second_alloy = 8 := by
  sorry

#check chromium_percentage_proof

end NUMINAMATH_CALUDE_chromium_percentage_proof_l2193_219398


namespace NUMINAMATH_CALUDE_largest_n_divisibility_equality_l2193_219353

/-- Count of integers less than or equal to n divisible by d -/
def count_divisible (n : ℕ) (d : ℕ) : ℕ := (n / d : ℕ)

/-- Count of integers less than or equal to n divisible by either a or b -/
def count_divisible_either (n : ℕ) (a b : ℕ) : ℕ :=
  count_divisible n a + count_divisible n b - count_divisible n (a * b)

theorem largest_n_divisibility_equality : ∀ m : ℕ, m > 65 →
  (count_divisible m 3 ≠ count_divisible_either m 5 7) ∧
  (count_divisible 65 3 = count_divisible_either 65 5 7) :=
by sorry

end NUMINAMATH_CALUDE_largest_n_divisibility_equality_l2193_219353


namespace NUMINAMATH_CALUDE_total_necklaces_is_1942_l2193_219371

/-- Represents the production of necklaces for a single machine on a given day -/
structure DailyProduction where
  machine : Nat
  day : Nat
  amount : Nat

/-- Calculates the total number of necklaces produced over three days -/
def totalNecklaces (productions : List DailyProduction) : Nat :=
  productions.map (·.amount) |>.sum

/-- The production data for all machines over three days -/
def necklaceProduction : List DailyProduction := [
  -- Sunday
  { machine := 1, day := 1, amount := 45 },
  { machine := 2, day := 1, amount := 108 },
  { machine := 3, day := 1, amount := 230 },
  { machine := 4, day := 1, amount := 184 },
  -- Monday
  { machine := 1, day := 2, amount := 59 },
  { machine := 2, day := 2, amount := 54 },
  { machine := 3, day := 2, amount := 230 },
  { machine := 4, day := 2, amount := 368 },
  -- Tuesday
  { machine := 1, day := 3, amount := 59 },
  { machine := 2, day := 3, amount := 108 },
  { machine := 3, day := 3, amount := 276 },
  { machine := 4, day := 3, amount := 221 }
]

/-- Theorem: The total number of necklaces produced over three days is 1942 -/
theorem total_necklaces_is_1942 : totalNecklaces necklaceProduction = 1942 := by
  sorry

end NUMINAMATH_CALUDE_total_necklaces_is_1942_l2193_219371


namespace NUMINAMATH_CALUDE_lottery_not_guaranteed_win_l2193_219303

/-- Represents a lottery with a total number of tickets and a winning rate. -/
structure Lottery where
  totalTickets : ℕ
  winningRate : ℝ
  winningRate_pos : winningRate > 0
  winningRate_le_one : winningRate ≤ 1

/-- The probability of not winning with a single ticket. -/
def Lottery.loseProb (l : Lottery) : ℝ := 1 - l.winningRate

/-- The probability of not winning with n tickets. -/
def Lottery.loseProbN (l : Lottery) (n : ℕ) : ℝ := (l.loseProb) ^ n

theorem lottery_not_guaranteed_win (l : Lottery) (h1 : l.totalTickets = 1000000) (h2 : l.winningRate = 0.001) :
  l.loseProbN 1000 > 0 := by sorry

end NUMINAMATH_CALUDE_lottery_not_guaranteed_win_l2193_219303


namespace NUMINAMATH_CALUDE_green_ducks_percentage_in_larger_pond_l2193_219340

/-- Represents the percentage of green ducks in the larger pond -/
def larger_pond_green_percentage : ℝ := 15

theorem green_ducks_percentage_in_larger_pond :
  let smaller_pond_ducks : ℕ := 20
  let larger_pond_ducks : ℕ := 80
  let smaller_pond_green_percentage : ℝ := 20
  let total_green_percentage : ℝ := 16
  larger_pond_green_percentage = 
    (total_green_percentage * (smaller_pond_ducks + larger_pond_ducks) - 
     smaller_pond_green_percentage * smaller_pond_ducks) / larger_pond_ducks := by
  sorry

end NUMINAMATH_CALUDE_green_ducks_percentage_in_larger_pond_l2193_219340


namespace NUMINAMATH_CALUDE_tan_10pi_minus_theta_l2193_219332

theorem tan_10pi_minus_theta (θ : Real) (h1 : π < θ) (h2 : θ < 2*π) 
  (h3 : Real.cos (θ - 9*π) = -3/5) : Real.tan (10*π - θ) = 4/3 := by
  sorry

end NUMINAMATH_CALUDE_tan_10pi_minus_theta_l2193_219332


namespace NUMINAMATH_CALUDE_circumscribed_sphere_surface_area_l2193_219305

/-- Given a rectangular tank with length 3√3, width 1, and height 2√2,
    the surface area of its circumscribed sphere is 36π. -/
theorem circumscribed_sphere_surface_area 
  (length : ℝ) (width : ℝ) (height : ℝ)
  (h_length : length = 3 * Real.sqrt 3)
  (h_width : width = 1)
  (h_height : height = 2 * Real.sqrt 2) :
  4 * Real.pi * ((length^2 + width^2 + height^2) / 4) = 36 * Real.pi := by
sorry

end NUMINAMATH_CALUDE_circumscribed_sphere_surface_area_l2193_219305


namespace NUMINAMATH_CALUDE_smaller_root_equation_l2193_219315

theorem smaller_root_equation (x : ℚ) : 
  let equation := (x - 3/4) * (x - 3/4) + (x - 3/4) * (x - 1/2) = 0
  let smaller_root := 5/8
  (equation ∧ x = smaller_root) ∨ 
  (equation ∧ x ≠ smaller_root ∧ x > smaller_root) :=
by sorry

end NUMINAMATH_CALUDE_smaller_root_equation_l2193_219315


namespace NUMINAMATH_CALUDE_orchard_problem_l2193_219301

/-- Represents the number of trees in the orchard -/
def T : ℕ := sorry

/-- The number of pure Fuji trees -/
def pure_fuji : ℕ := (3 * T) / 4

/-- The number of cross-pollinated trees -/
def cross_pollinated : ℕ := T / 10

/-- The number of pure Gala trees -/
def pure_gala : ℕ := T - pure_fuji - cross_pollinated

theorem orchard_problem :
  pure_fuji + cross_pollinated = 204 →
  pure_gala = 60 := by
  sorry

end NUMINAMATH_CALUDE_orchard_problem_l2193_219301


namespace NUMINAMATH_CALUDE_sunset_duration_l2193_219358

/-- Proves that a sunset with 12 color changes occurring every 10 minutes lasts 2 hours. -/
theorem sunset_duration (color_change_interval : ℕ) (total_changes : ℕ) (minutes_per_hour : ℕ) :
  color_change_interval = 10 →
  total_changes = 12 →
  minutes_per_hour = 60 →
  (color_change_interval * total_changes) / minutes_per_hour = 2 :=
by
  sorry

end NUMINAMATH_CALUDE_sunset_duration_l2193_219358


namespace NUMINAMATH_CALUDE_smaller_paintings_count_l2193_219314

/-- Represents a museum with paintings and artifacts --/
structure Museum where
  total_wings : ℕ
  painting_wings : ℕ
  artifacts_per_wing : ℕ
  artifact_painting_ratio : ℕ

/-- The number of smaller paintings in each of the two wings --/
def smaller_paintings_per_wing (m : Museum) : ℕ :=
  ((m.artifacts_per_wing * (m.total_wings - m.painting_wings)) / m.artifact_painting_ratio - 1) / 2

/-- Theorem stating the number of smaller paintings per wing --/
theorem smaller_paintings_count (m : Museum) 
  (h1 : m.total_wings = 8)
  (h2 : m.painting_wings = 3)
  (h3 : m.artifacts_per_wing = 20)
  (h4 : m.artifact_painting_ratio = 4) :
  smaller_paintings_per_wing m = 12 := by
  sorry

end NUMINAMATH_CALUDE_smaller_paintings_count_l2193_219314


namespace NUMINAMATH_CALUDE_imoCandidate1988_l2193_219335

theorem imoCandidate1988 (d r : ℤ) : 
  d > 1 ∧ 
  (∃ k m n : ℤ, 1059 = k * d + r ∧ 
               1417 = m * d + r ∧ 
               2312 = n * d + r) →
  d - r = 15 := by sorry

end NUMINAMATH_CALUDE_imoCandidate1988_l2193_219335


namespace NUMINAMATH_CALUDE_power_equation_solution_l2193_219389

theorem power_equation_solution (K : ℕ) : 32^4 * 4^6 = 2^K → K = 32 := by
  sorry

end NUMINAMATH_CALUDE_power_equation_solution_l2193_219389


namespace NUMINAMATH_CALUDE_modulus_z_l2193_219380

theorem modulus_z (w z : ℂ) (h1 : w * z = 15 - 20 * I) (h2 : Complex.abs w = Real.sqrt 13) :
  Complex.abs z = (25 * Real.sqrt 13) / 13 := by
  sorry

end NUMINAMATH_CALUDE_modulus_z_l2193_219380


namespace NUMINAMATH_CALUDE_quadratic_root_range_l2193_219382

theorem quadratic_root_range (a : ℝ) : 
  (∃ x y : ℝ, x ≠ y ∧ x^2 + (a^2 - 1)*x + a - 2 = 0 ∧ y^2 + (a^2 - 1)*y + a - 2 = 0 ∧ x > 1 ∧ y < 1) 
  → a > -2 ∧ a < 1 := by
sorry

end NUMINAMATH_CALUDE_quadratic_root_range_l2193_219382


namespace NUMINAMATH_CALUDE_probability_sum_seven_l2193_219326

def die_faces : Finset ℕ := {1, 2, 3, 4, 5, 6}

def sum_is_seven (roll : ℕ × ℕ) : Prop :=
  roll.1 + roll.2 = 7

def favorable_outcomes : Finset (ℕ × ℕ) :=
  {(1, 6), (6, 1), (2, 5), (5, 2), (3, 4), (4, 3)}

theorem probability_sum_seven :
  (Finset.card favorable_outcomes : ℚ) / (Finset.card (die_faces.product die_faces) : ℚ) = 1 / 6 := by
  sorry

end NUMINAMATH_CALUDE_probability_sum_seven_l2193_219326


namespace NUMINAMATH_CALUDE_sector_area_l2193_219366

/-- The area of a circular sector with radius 6 cm and central angle 120° is 12π cm². -/
theorem sector_area : 
  let r : ℝ := 6
  let θ : ℝ := 120
  let π : ℝ := Real.pi
  (θ / 360) * π * r^2 = 12 * π := by sorry

end NUMINAMATH_CALUDE_sector_area_l2193_219366


namespace NUMINAMATH_CALUDE_sqrt_16_equals_4_l2193_219308

theorem sqrt_16_equals_4 : Real.sqrt 16 = 4 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_16_equals_4_l2193_219308


namespace NUMINAMATH_CALUDE_parallelogram_division_max_parts_l2193_219367

/-- Given a parallelogram divided into a grid of M by N parts, with one additional line drawn,
    the maximum number of parts the parallelogram can be divided into is MN + M + N - 1. -/
theorem parallelogram_division_max_parts (M N : ℕ) :
  let initial_parts := M * N
  let additional_parts := M + N - 1
  initial_parts + additional_parts = M * N + M + N - 1 := by
  sorry

end NUMINAMATH_CALUDE_parallelogram_division_max_parts_l2193_219367


namespace NUMINAMATH_CALUDE_two_correct_conclusions_l2193_219390

theorem two_correct_conclusions : ∃ (S : Finset (Prop)), S.card = 2 ∧ S ⊆ 
  {∀ (k b x₁ x₂ y₁ y₂ : ℝ), k < 0 → y₁ = k * x₁ + b → y₂ = k * x₂ + b → x₁ > x₂ → y₁ > y₂,
   ∀ (k b : ℝ), (∃ (x y : ℝ), x > 0 ∧ y > 0 ∧ y = k * x + b) ∧ 
                (∃ (x y : ℝ), x < 0 ∧ y > 0 ∧ y = k * x + b) ∧ 
                (∃ (x y : ℝ), x < 0 ∧ y < 0 ∧ y = k * x + b) → 
                k > 0 ∧ b > 0,
   ∀ (m : ℝ), (m - 1) * 0 + m^2 + 2 = 3 → m = 1 ∨ m = -1} ∧ 
  (∀ p ∈ S, p) := by
sorry

end NUMINAMATH_CALUDE_two_correct_conclusions_l2193_219390


namespace NUMINAMATH_CALUDE_scott_distance_l2193_219323

/-- Given a 100-meter race where Scott runs 4 meters for every 5 meters that Chris runs,
    prove that Scott will have run 80 meters when Chris crosses the finish line. -/
theorem scott_distance (race_length : ℕ) (scott_ratio chris_ratio : ℕ) : 
  race_length = 100 →
  scott_ratio = 4 →
  chris_ratio = 5 →
  (scott_ratio * race_length) / chris_ratio = 80 := by
sorry

end NUMINAMATH_CALUDE_scott_distance_l2193_219323


namespace NUMINAMATH_CALUDE_inequality_problem_l2193_219341

theorem inequality_problem (a b c : ℝ) (h1 : a > b) (h2 : b > 0) (h3 : c ≠ 0) :
  (b / a < (b + c^2) / (a + c^2)) ∧ (a^2 - 1/a > b^2 - 1/b) := by
  sorry

end NUMINAMATH_CALUDE_inequality_problem_l2193_219341


namespace NUMINAMATH_CALUDE_quadratic_factorization_l2193_219355

/-- Given a quadratic equation x^2 + px + q = 0 with roots 2 and -3,
    prove that it can be factored as (x - 2)(x + 3) = 0 -/
theorem quadratic_factorization (p q : ℝ) :
  (∀ x, x^2 + p*x + q = 0 ↔ x = 2 ∨ x = -3) →
  ∀ x, x^2 + p*x + q = 0 ↔ (x - 2) * (x + 3) = 0 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_factorization_l2193_219355


namespace NUMINAMATH_CALUDE_parabola_intercepts_sum_l2193_219379

/-- Parabola equation: x = 3y^2 - 9y + 5 -/
def parabola_eq (x y : ℝ) : Prop := x = 3 * y^2 - 9 * y + 5

/-- X-intercept of the parabola -/
def x_intercept (a : ℝ) : Prop := parabola_eq a 0

/-- Y-intercepts of the parabola -/
def y_intercepts (b c : ℝ) : Prop := parabola_eq 0 b ∧ parabola_eq 0 c ∧ b ≠ c

theorem parabola_intercepts_sum :
  ∀ a b c : ℝ, x_intercept a → y_intercepts b c → a + b + c = 8 := by
  sorry

end NUMINAMATH_CALUDE_parabola_intercepts_sum_l2193_219379


namespace NUMINAMATH_CALUDE_matrix_identity_l2193_219377

open Matrix

theorem matrix_identity (B : Matrix (Fin n) (Fin n) ℝ) 
  (h_inv : Invertible B) 
  (h_eq : (B - 3 • 1) * (B - 5 • 1) = 0) : 
  B + 10 • B⁻¹ = 8 • 1 := by sorry

end NUMINAMATH_CALUDE_matrix_identity_l2193_219377


namespace NUMINAMATH_CALUDE_jerry_piercing_pricing_l2193_219391

theorem jerry_piercing_pricing (nose_price : ℝ) (total_revenue : ℝ) (num_noses : ℕ) (num_ears : ℕ) :
  nose_price = 20 →
  total_revenue = 390 →
  num_noses = 6 →
  num_ears = 9 →
  let ear_price := (total_revenue - nose_price * num_noses) / num_ears
  let percentage_increase := (ear_price - nose_price) / nose_price * 100
  percentage_increase = 50 := by
sorry


end NUMINAMATH_CALUDE_jerry_piercing_pricing_l2193_219391


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l2193_219316

def A : Set ℝ := {-1, 0, 1, 2}
def B : Set ℝ := {x : ℝ | -1 < x ∧ x ≤ 1}

theorem intersection_of_A_and_B : A ∩ B = {0, 1} := by sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l2193_219316


namespace NUMINAMATH_CALUDE_range_of_a_l2193_219322

-- Define the linear function
def f (a x : ℝ) : ℝ := (2 + a) * x + (5 - a)

-- Define the condition for the graph to pass through the first, second, and third quadrants
def passes_through_123_quadrants (a : ℝ) : Prop :=
  (2 + a > 0) ∧ (5 - a > 0)

-- Theorem statement
theorem range_of_a (a : ℝ) :
  passes_through_123_quadrants a → -2 < a ∧ a < 5 := by
  sorry

end NUMINAMATH_CALUDE_range_of_a_l2193_219322


namespace NUMINAMATH_CALUDE_triangle_properties_l2193_219321

theorem triangle_properties (A B C : ℝ) (a b c : ℝ) :
  0 < A ∧ 0 < B ∧ 0 < C ∧
  A + B + C = π ∧
  0 < a ∧ 0 < b ∧ 0 < c ∧
  a / (Real.sin A) = b / (Real.sin B) ∧
  b / (Real.sin B) = c / (Real.sin C) ∧
  4 * a = Real.sqrt 5 * c ∧
  Real.cos C = 3 / 5 →
  Real.sin A = Real.sqrt 5 / 5 ∧
  (b = 11 → a = 5) ∧
  (b = 11 → Real.cos (2 * A + C) = -7 / 25) := by
sorry


end NUMINAMATH_CALUDE_triangle_properties_l2193_219321


namespace NUMINAMATH_CALUDE_selling_price_with_loss_l2193_219302

theorem selling_price_with_loss (cost_price : ℝ) (loss_percent : ℝ) (selling_price : ℝ) :
  cost_price = 600 →
  loss_percent = 8.333333333333329 →
  selling_price = cost_price * (1 - loss_percent / 100) →
  selling_price = 550 := by
  sorry

end NUMINAMATH_CALUDE_selling_price_with_loss_l2193_219302


namespace NUMINAMATH_CALUDE_solutions_eq_divisors_l2193_219386

/-- The number of integer solutions to the equation xy + ax + by = c -/
def num_solutions (a b c : ℤ) : ℕ :=
  2 * (Nat.divisors (a * b + c).natAbs).card

/-- The number of divisors (positive and negative) of an integer n -/
def num_divisors (n : ℤ) : ℕ :=
  2 * (Nat.divisors n.natAbs).card

theorem solutions_eq_divisors (a b c : ℤ) :
  num_solutions a b c = num_divisors (a * b + c) :=
sorry

end NUMINAMATH_CALUDE_solutions_eq_divisors_l2193_219386


namespace NUMINAMATH_CALUDE_pyramid_rows_equal_ten_l2193_219307

/-- The number of spheres in a square-based pyramid with n rows -/
def square_pyramid (n : ℕ) : ℕ := n * (n + 1) * (2 * n + 1) / 6

/-- The number of spheres in a triangle-based pyramid with n rows -/
def triangle_pyramid (n : ℕ) : ℕ := n * (n + 1) * (n + 2) / 6

/-- The total number of spheres -/
def total_spheres : ℕ := 605

theorem pyramid_rows_equal_ten :
  ∃ (n : ℕ), n > 0 ∧ square_pyramid n + triangle_pyramid n = total_spheres := by
  sorry

end NUMINAMATH_CALUDE_pyramid_rows_equal_ten_l2193_219307


namespace NUMINAMATH_CALUDE_sinusoidal_function_properties_l2193_219313

/-- Given a sinusoidal function y = a * sin(b * x + c) with a > 0 and b > 0,
    if the maximum occurs at x = π/6 and the amplitude is 3,
    then a = 3 and c = (3 - b) * π/6 -/
theorem sinusoidal_function_properties (a b c : ℝ) (h1 : a > 0) (h2 : b > 0)
  (h3 : ∀ x, a * Real.sin (b * x + c) ≤ a * Real.sin (b * (π/6) + c))
  (h4 : a = 3) :
  a = 3 ∧ c = (3 - b) * π/6 := by
sorry

end NUMINAMATH_CALUDE_sinusoidal_function_properties_l2193_219313


namespace NUMINAMATH_CALUDE_fourth_rectangle_area_l2193_219360

/-- Represents a rectangle divided into four smaller rectangles -/
structure DividedRectangle where
  total_area : ℝ
  area1 : ℝ
  area2 : ℝ
  area3 : ℝ
  area4 : ℝ
  vertical_halves : area1 + area4 = area2 + area3
  sum_of_areas : total_area = area1 + area2 + area3 + area4

/-- Theorem stating that given the areas of three rectangles in a divided rectangle,
    we can determine the area of the fourth rectangle -/
theorem fourth_rectangle_area
  (rect : DividedRectangle)
  (h1 : rect.area1 = 12)
  (h2 : rect.area2 = 27)
  (h3 : rect.area3 = 18) :
  rect.area4 = 27 := by
  sorry

#check fourth_rectangle_area

end NUMINAMATH_CALUDE_fourth_rectangle_area_l2193_219360


namespace NUMINAMATH_CALUDE_olympic_arrangements_correct_l2193_219383

/-- The number of ways to arrange athletes in Olympic lanes -/
def olympicArrangements : ℕ := 2520

/-- The number of lanes -/
def numLanes : ℕ := 8

/-- The number of countries -/
def numCountries : ℕ := 4

/-- The number of athletes per country -/
def athletesPerCountry : ℕ := 2

/-- Theorem: The number of ways to arrange the athletes is correct -/
theorem olympic_arrangements_correct :
  olympicArrangements = (numLanes.choose athletesPerCountry) *
                        ((numLanes - athletesPerCountry).choose athletesPerCountry) *
                        ((numLanes - 2 * athletesPerCountry).choose athletesPerCountry) *
                        ((numLanes - 3 * athletesPerCountry).choose athletesPerCountry) :=
by sorry

end NUMINAMATH_CALUDE_olympic_arrangements_correct_l2193_219383


namespace NUMINAMATH_CALUDE_total_clothing_is_934_l2193_219311

/-- The number of shirts Mr. Anderson gave out -/
def shirts : ℕ := 589

/-- The number of trousers Mr. Anderson gave out -/
def trousers : ℕ := 345

/-- The total number of clothing pieces Mr. Anderson gave out -/
def total_clothing : ℕ := shirts + trousers

/-- Theorem stating that the total number of clothing pieces is 934 -/
theorem total_clothing_is_934 : total_clothing = 934 := by
  sorry

end NUMINAMATH_CALUDE_total_clothing_is_934_l2193_219311


namespace NUMINAMATH_CALUDE_absolute_value_inequality_solution_l2193_219370

theorem absolute_value_inequality_solution (x : ℝ) :
  |2*x - 7| < 3 ↔ 2 < x ∧ x < 5 := by sorry

end NUMINAMATH_CALUDE_absolute_value_inequality_solution_l2193_219370


namespace NUMINAMATH_CALUDE_only_cone_cannot_have_rectangular_cross_section_l2193_219348

-- Define the types of geometric solids
inductive GeometricSolid
  | Cylinder
  | Cone
  | PentagonalPrism
  | Cube

-- Define a function that determines if a solid can have a rectangular cross-section
def canHaveRectangularCrossSection (solid : GeometricSolid) : Prop :=
  match solid with
  | GeometricSolid.Cylinder => True
  | GeometricSolid.Cone => False
  | GeometricSolid.PentagonalPrism => True
  | GeometricSolid.Cube => True

-- Theorem stating that only the cone cannot have a rectangular cross-section
theorem only_cone_cannot_have_rectangular_cross_section :
  ∀ (solid : GeometricSolid),
    ¬(canHaveRectangularCrossSection solid) ↔ solid = GeometricSolid.Cone :=
by sorry

end NUMINAMATH_CALUDE_only_cone_cannot_have_rectangular_cross_section_l2193_219348


namespace NUMINAMATH_CALUDE_redbirds_count_l2193_219393

theorem redbirds_count (total : ℕ) (bluebird_fraction : ℚ) (h1 : total = 120) (h2 : bluebird_fraction = 5/6) :
  (1 - bluebird_fraction) * total = 20 := by
  sorry

end NUMINAMATH_CALUDE_redbirds_count_l2193_219393


namespace NUMINAMATH_CALUDE_sufficient_but_not_necessary_l2193_219312

theorem sufficient_but_not_necessary (x y : ℝ) : 
  (x = -y → x^2 - y^2 - x - y = 0) ∧ 
  ¬(x^2 - y^2 - x - y = 0 → x = -y) := by
  sorry

end NUMINAMATH_CALUDE_sufficient_but_not_necessary_l2193_219312


namespace NUMINAMATH_CALUDE_greatest_area_difference_l2193_219324

/-- Represents a rectangle with integer dimensions. -/
structure Rectangle where
  length : ℕ
  width : ℕ

/-- Calculates the perimeter of a rectangle. -/
def perimeter (r : Rectangle) : ℕ := 2 * (r.length + r.width)

/-- Calculates the area of a rectangle. -/
def area (r : Rectangle) : ℕ := r.length * r.width

/-- Represents the maximum length available for the rotated rectangle's diagonal. -/
def maxDiagonal : ℕ := 50

theorem greatest_area_difference : 
  ∃ (r1 r2 : Rectangle), 
    perimeter r1 = 100 ∧ 
    perimeter r2 = 100 ∧ 
    r2.length * r2.length + r2.width * r2.width ≤ maxDiagonal * maxDiagonal ∧
    ∀ (s1 s2 : Rectangle), 
      perimeter s1 = 100 → 
      perimeter s2 = 100 → 
      s2.length * s2.length + s2.width * s2.width ≤ maxDiagonal * maxDiagonal →
      (area r1 - area r2) ≥ (area s1 - area s2) ∧
      (area r1 - area r2) = 373 :=
by
  sorry

end NUMINAMATH_CALUDE_greatest_area_difference_l2193_219324


namespace NUMINAMATH_CALUDE_x_equals_four_l2193_219388

theorem x_equals_four : ∃! x : ℤ, 2^4 + x = 3^3 - 7 :=
by
  sorry

end NUMINAMATH_CALUDE_x_equals_four_l2193_219388


namespace NUMINAMATH_CALUDE_polynomial_expansion_and_sum_l2193_219374

theorem polynomial_expansion_and_sum (A B C D E : ℤ) : 
  (∀ x : ℝ, (x + 3) * (4 * x^3 - 2 * x^2 + 7 * x - 6) = A * x^4 + B * x^3 + C * x^2 + D * x + E) →
  A = 4 ∧ B = 10 ∧ C = 1 ∧ D = 15 ∧ E = -18 ∧ A + B + C + D + E = 12 := by
sorry

end NUMINAMATH_CALUDE_polynomial_expansion_and_sum_l2193_219374


namespace NUMINAMATH_CALUDE_sneakers_cost_l2193_219325

/-- The cost of sneakers given initial savings, action figure sales, and remaining money --/
theorem sneakers_cost
  (initial_savings : ℕ)
  (action_figures_sold : ℕ)
  (price_per_figure : ℕ)
  (money_left : ℕ)
  (h1 : initial_savings = 15)
  (h2 : action_figures_sold = 10)
  (h3 : price_per_figure = 10)
  (h4 : money_left = 25) :
  initial_savings + action_figures_sold * price_per_figure - money_left = 90 := by
  sorry

end NUMINAMATH_CALUDE_sneakers_cost_l2193_219325


namespace NUMINAMATH_CALUDE_final_bird_count_and_ratio_l2193_219338

/-- Represents the number of birds in the park -/
structure BirdCount where
  blackbirds : ℕ
  magpies : ℕ
  blueJays : ℕ
  robins : ℕ

/-- Calculates the initial bird count based on given conditions -/
def initialBirdCount : BirdCount :=
  { blackbirds := 3 * 7,
    magpies := 13,
    blueJays := 2 * 5,
    robins := 4 }

/-- Calculates the final bird count after changes -/
def finalBirdCount : BirdCount :=
  { blackbirds := initialBirdCount.blackbirds - 6,
    magpies := initialBirdCount.magpies + 8,
    blueJays := initialBirdCount.blueJays + 3,
    robins := initialBirdCount.robins }

/-- Calculates the total number of birds -/
def totalBirds (count : BirdCount) : ℕ :=
  count.blackbirds + count.magpies + count.blueJays + count.robins

/-- Theorem: The final number of birds is 53 and the ratio is 15:21:13:4 -/
theorem final_bird_count_and_ratio :
  totalBirds finalBirdCount = 53 ∧
  finalBirdCount.blackbirds = 15 ∧
  finalBirdCount.magpies = 21 ∧
  finalBirdCount.blueJays = 13 ∧
  finalBirdCount.robins = 4 := by
  sorry


end NUMINAMATH_CALUDE_final_bird_count_and_ratio_l2193_219338


namespace NUMINAMATH_CALUDE_sqrt_5184_div_18_eq_4_l2193_219306

theorem sqrt_5184_div_18_eq_4 : Real.sqrt 5184 / 18 = 4 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_5184_div_18_eq_4_l2193_219306


namespace NUMINAMATH_CALUDE_sqrt_inequality_l2193_219334

theorem sqrt_inequality (a : ℝ) (h : a ≥ 3) :
  Real.sqrt a - Real.sqrt (a - 2) < Real.sqrt (a - 1) - Real.sqrt (a - 3) := by
  sorry

end NUMINAMATH_CALUDE_sqrt_inequality_l2193_219334


namespace NUMINAMATH_CALUDE_proper_subsets_count_l2193_219347

def U : Finset Nat := {1,2,3,4,5}
def A : Finset Nat := {1,2}
def B : Finset Nat := {3,4}

theorem proper_subsets_count :
  (Finset.powerset (A ∩ (U \ B))).card - 1 = 3 := by sorry

end NUMINAMATH_CALUDE_proper_subsets_count_l2193_219347


namespace NUMINAMATH_CALUDE_complex_magnitude_range_l2193_219396

theorem complex_magnitude_range (z₁ z₂ : ℂ) 
  (h₁ : (z₁ - Complex.I) * (z₂ + Complex.I) = 1)
  (h₂ : Complex.abs z₁ = Real.sqrt 2) :
  ∃ (a b : ℝ), a = 2 - Real.sqrt 2 ∧ b = 2 + Real.sqrt 2 ∧ 
  a ≤ Complex.abs z₂ ∧ Complex.abs z₂ ≤ b :=
sorry

end NUMINAMATH_CALUDE_complex_magnitude_range_l2193_219396


namespace NUMINAMATH_CALUDE_arithmetic_sequence_ratio_l2193_219397

/-- Given an arithmetic sequence with non-zero common difference, 
    if a_2 + a_3 = a_6, then (a_1 + a_2) / (a_3 + a_4 + a_5) = 1/3 -/
theorem arithmetic_sequence_ratio (a : ℕ → ℚ) (d : ℚ) (h1 : d ≠ 0) 
  (h2 : ∀ n, a (n + 1) = a n + d) 
  (h3 : a 2 + a 3 = a 6) : 
  (a 1 + a 2) / (a 3 + a 4 + a 5) = 1 / 3 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_ratio_l2193_219397


namespace NUMINAMATH_CALUDE_fifteenth_term_of_sequence_l2193_219362

/-- The nth term of an arithmetic sequence -/
def arithmetic_sequence (a₁ : ℤ) (d : ℤ) (n : ℕ) : ℤ :=
  a₁ + (n - 1) * d

/-- The 15th term of the specific arithmetic sequence -/
theorem fifteenth_term_of_sequence : arithmetic_sequence (-3) 4 15 = 53 := by
  sorry

end NUMINAMATH_CALUDE_fifteenth_term_of_sequence_l2193_219362


namespace NUMINAMATH_CALUDE_book_pages_count_l2193_219304

/-- The total number of pages in Isabella's book -/
def total_pages : ℕ := 288

/-- The number of days Isabella took to read the book -/
def total_days : ℕ := 8

/-- The average number of pages Isabella read daily for the first four days -/
def first_four_days_avg : ℕ := 28

/-- The average number of pages Isabella read daily for the next three days -/
def next_three_days_avg : ℕ := 52

/-- The number of pages Isabella read on the final day -/
def final_day_pages : ℕ := 20

/-- Theorem stating that the total number of pages in the book is 288 -/
theorem book_pages_count : 
  (4 * first_four_days_avg + 3 * next_three_days_avg + final_day_pages = total_pages) ∧
  (total_days = 8) := by
  sorry

end NUMINAMATH_CALUDE_book_pages_count_l2193_219304


namespace NUMINAMATH_CALUDE_circle_area_doubling_l2193_219330

theorem circle_area_doubling (r n : ℝ) : 
  (r > 0) → (n > 0) → (π * (r + n)^2 = 2 * π * r^2) → (r = n * (Real.sqrt 2 + 1)) := by
  sorry

end NUMINAMATH_CALUDE_circle_area_doubling_l2193_219330


namespace NUMINAMATH_CALUDE_fifteen_team_league_games_l2193_219354

/-- The number of games played in a league where each team plays every other team once -/
def gamesPlayed (n : ℕ) : ℕ := n * (n - 1) / 2

/-- Theorem: In a league with 15 teams, where each team plays every other team once, 
    the total number of games played is 105 -/
theorem fifteen_team_league_games : gamesPlayed 15 = 105 := by
  sorry

end NUMINAMATH_CALUDE_fifteen_team_league_games_l2193_219354


namespace NUMINAMATH_CALUDE_cos_BAD_equals_sqrt_13_45_l2193_219375

-- Define the triangle ABC
def Triangle (A B C : ℝ × ℝ) : Prop := True

-- Define the lengths of the sides
def AB (A B : ℝ × ℝ) : ℝ := sorry
def AC (A C : ℝ × ℝ) : ℝ := sorry
def BC (B C : ℝ × ℝ) : ℝ := sorry

-- Define a point D on BC
def D_on_BC (B C D : ℝ × ℝ) : Prop := sorry

-- Define the angle bisector property
def is_angle_bisector (A B C D : ℝ × ℝ) : Prop := sorry

-- Define the cosine of an angle
def cos_angle (A B C : ℝ × ℝ) : ℝ := sorry

theorem cos_BAD_equals_sqrt_13_45 
  (A B C D : ℝ × ℝ) 
  (h1 : Triangle A B C)
  (h2 : AB A B = 5)
  (h3 : AC A C = 9)
  (h4 : BC B C = 12)
  (h5 : D_on_BC B C D)
  (h6 : is_angle_bisector A B C D) :
  cos_angle B A D = Real.sqrt (13 / 45) := by
  sorry

end NUMINAMATH_CALUDE_cos_BAD_equals_sqrt_13_45_l2193_219375


namespace NUMINAMATH_CALUDE_solution_of_linear_equation_l2193_219395

theorem solution_of_linear_equation :
  ∀ x : ℝ, x - 2 = 0 ↔ x = 2 := by sorry

end NUMINAMATH_CALUDE_solution_of_linear_equation_l2193_219395


namespace NUMINAMATH_CALUDE_sequence_properties_l2193_219328

/-- The sum of the first n terms of sequence a_n -/
def S (n : ℕ) : ℝ := sorry

/-- The nth term of sequence a_n -/
def a (n : ℕ) : ℝ := sorry

/-- The nth term of arithmetic sequence b_n -/
def b (n : ℕ) : ℝ := sorry

/-- The sum of the first n terms of sequence b_n -/
def T (n : ℕ) : ℝ := sorry

theorem sequence_properties :
  (∀ n, 2 * S n = 3 * a n - 3) ∧
  (b 1 = a 1) ∧
  (b 7 = b 1 * b 2) ∧
  (∀ n m, b (n + m) - b n = m * (b 2 - b 1)) →
  (∀ n, a n = 3^n) ∧
  (∀ n, T n = n^2 + 2*n) := by
sorry

end NUMINAMATH_CALUDE_sequence_properties_l2193_219328


namespace NUMINAMATH_CALUDE_magnitude_of_vector_sum_l2193_219394

/-- Given vectors e₁ and e₂ forming an angle of 2π/3, prove that |e₁ + 2e₂| = √3 -/
theorem magnitude_of_vector_sum (e₁ e₂ : ℝ × ℝ) : 
  e₁ • e₁ = 1 → 
  e₂ • e₂ = 1 → 
  e₁ • e₂ = -1/2 → 
  let a := e₁ + 2 • e₂ 
  ‖a‖ = Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_magnitude_of_vector_sum_l2193_219394


namespace NUMINAMATH_CALUDE_sphere_intersection_radius_l2193_219339

-- Define the sphere
def sphere_center : ℝ × ℝ × ℝ := (3, 5, -9)

-- Define the intersection circles
def xy_circle_center : ℝ × ℝ × ℝ := (3, 5, 0)
def xy_circle_radius : ℝ := 2

def xz_circle_center : ℝ × ℝ × ℝ := (0, 5, -9)

-- Theorem statement
theorem sphere_intersection_radius : 
  let s := Real.sqrt ((Real.sqrt 85)^2 - 3^2)
  s = Real.sqrt 76 :=
sorry

end NUMINAMATH_CALUDE_sphere_intersection_radius_l2193_219339


namespace NUMINAMATH_CALUDE_equation_solution_l2193_219356

theorem equation_solution (x : ℝ) : 
  (x = (-81 + Real.sqrt 5297) / 8 ∨ x = (-81 - Real.sqrt 5297) / 8) ↔ 
  (8 * x^2 + 89 * x + 3) / (3 * x + 41) = 4 * x + 2 := by sorry

end NUMINAMATH_CALUDE_equation_solution_l2193_219356


namespace NUMINAMATH_CALUDE_inequality_equivalence_l2193_219327

theorem inequality_equivalence (a : ℝ) :
  (∀ x : ℝ, |3*x + 2*a| + |2 - 3*x| - |a + 1| > 2) ↔ (a < -1/3 ∨ a > 5) := by
sorry

end NUMINAMATH_CALUDE_inequality_equivalence_l2193_219327


namespace NUMINAMATH_CALUDE_badge_exchange_l2193_219320

theorem badge_exchange (x : ℝ) : 
  (x + 5) - (24/100) * (x + 5) + (20/100) * x = x - (20/100) * x + (24/100) * (x + 5) - 1 → 
  x = 45 := by
sorry

end NUMINAMATH_CALUDE_badge_exchange_l2193_219320


namespace NUMINAMATH_CALUDE_correct_average_l2193_219352

theorem correct_average (n : ℕ) (incorrect_avg : ℚ) (incorrect_num : ℚ) (correct_num : ℚ) :
  n = 10 →
  incorrect_avg = 16 →
  incorrect_num = 25 →
  correct_num = 55 →
  (n : ℚ) * incorrect_avg + (correct_num - incorrect_num) = n * 19 := by
  sorry

#check correct_average

end NUMINAMATH_CALUDE_correct_average_l2193_219352


namespace NUMINAMATH_CALUDE_modulo_residue_problem_l2193_219300

theorem modulo_residue_problem :
  (250 * 15 - 337 * 5 + 22) % 13 = 7 := by
  sorry

end NUMINAMATH_CALUDE_modulo_residue_problem_l2193_219300


namespace NUMINAMATH_CALUDE_mod_seven_power_difference_l2193_219349

theorem mod_seven_power_difference : 47^2023 - 28^2023 ≡ 5 [ZMOD 7] := by
  sorry

end NUMINAMATH_CALUDE_mod_seven_power_difference_l2193_219349


namespace NUMINAMATH_CALUDE_ducks_in_marsh_l2193_219350

/-- The number of ducks in a marsh, given the total number of birds and the number of geese. -/
def number_of_ducks (total_birds geese : ℕ) : ℕ := total_birds - geese

/-- Theorem stating that there are 37 ducks in the marsh. -/
theorem ducks_in_marsh : number_of_ducks 95 58 = 37 := by
  sorry

end NUMINAMATH_CALUDE_ducks_in_marsh_l2193_219350


namespace NUMINAMATH_CALUDE_allocation_schemes_13_4_l2193_219344

/-- The number of ways to allocate outstanding member quotas to classes. -/
def allocationSchemes (totalMembers : ℕ) (numClasses : ℕ) : ℕ :=
  Nat.choose (totalMembers - numClasses + numClasses - 1) (numClasses - 1)

/-- Theorem stating the number of allocation schemes for 13 members to 4 classes. -/
theorem allocation_schemes_13_4 :
  allocationSchemes 13 4 = 220 := by
  sorry

#eval allocationSchemes 13 4

end NUMINAMATH_CALUDE_allocation_schemes_13_4_l2193_219344


namespace NUMINAMATH_CALUDE_equilateral_triangle_perimeter_l2193_219359

-- Define an equilateral triangle
structure EquilateralTriangle where
  side_length : ℝ

-- Define the given conditions
def given_triangle (x : ℝ) : Prop :=
  ∃ (t : EquilateralTriangle), t.side_length = 2*x ∧ t.side_length = x + 15

-- Define the perimeter of an equilateral triangle
def perimeter (t : EquilateralTriangle) : ℝ := 3 * t.side_length

-- Theorem statement
theorem equilateral_triangle_perimeter :
  ∀ x : ℝ, given_triangle x → ∃ (t : EquilateralTriangle), perimeter t = 90 :=
by sorry

end NUMINAMATH_CALUDE_equilateral_triangle_perimeter_l2193_219359


namespace NUMINAMATH_CALUDE_martha_lasagna_cost_l2193_219343

/-- The cost of ingredients for Martha's lasagna -/
def lasagna_cost (cheese_price meat_price pasta_price tomato_price : ℝ) : ℝ :=
  1.5 * cheese_price + 0.55 * meat_price + 0.28 * pasta_price + 2.2 * tomato_price

/-- Theorem stating the total cost of ingredients for Martha's lasagna -/
theorem martha_lasagna_cost :
  lasagna_cost 6.30 8.55 2.40 1.79 = 18.76 := by
  sorry


end NUMINAMATH_CALUDE_martha_lasagna_cost_l2193_219343


namespace NUMINAMATH_CALUDE_hotel_stay_cost_l2193_219317

/-- The total cost for a group staying at a hotel. -/
def total_cost (cost_per_night_per_person : ℕ) (num_people : ℕ) (num_nights : ℕ) : ℕ :=
  cost_per_night_per_person * num_people * num_nights

/-- Theorem: The total cost for 3 people staying 3 nights at $40 per night per person is $360. -/
theorem hotel_stay_cost : total_cost 40 3 3 = 360 := by
  sorry

end NUMINAMATH_CALUDE_hotel_stay_cost_l2193_219317


namespace NUMINAMATH_CALUDE_fraction_simplification_l2193_219369

theorem fraction_simplification (a b : ℝ) (h : a ≠ b) :
  a / (a - b) - b / (b - a) = (a + b) / (a - b) := by
  sorry

end NUMINAMATH_CALUDE_fraction_simplification_l2193_219369


namespace NUMINAMATH_CALUDE_average_price_of_cow_l2193_219309

/-- Given the total price for 2 cows and 8 goats, and the average price of a goat,
    prove that the average price of a cow is 460 rupees. -/
theorem average_price_of_cow (total_price : ℕ) (goat_price : ℕ) (cow_count : ℕ) (goat_count : ℕ) :
  total_price = 1400 →
  goat_price = 60 →
  cow_count = 2 →
  goat_count = 8 →
  (total_price - goat_count * goat_price) / cow_count = 460 := by
  sorry

end NUMINAMATH_CALUDE_average_price_of_cow_l2193_219309


namespace NUMINAMATH_CALUDE_ellipse_and_chord_problem_l2193_219384

-- Define the ellipse
def Ellipse (a b : ℝ) : Set (ℝ × ℝ) :=
  {p | (p.2^2 / a^2) + (p.1^2 / b^2) = 1}

-- Define the circle
def Circle (r : ℝ) : Set (ℝ × ℝ) :=
  {p | p.1^2 + p.2^2 = r^2}

theorem ellipse_and_chord_problem 
  (e : ℝ) (f : ℝ × ℝ) 
  (h_e : e = 2 * Real.sqrt 2 / 3)
  (h_f : f = (0, 2 * Real.sqrt 2))
  (h_foci : ∃ (f' : ℝ × ℝ), f'.1 = 0 ∧ f'.2 = -f.2) :
  -- Standard equation of the ellipse
  ∃ (a b : ℝ), a > 0 ∧ b > 0 ∧ Ellipse a b = Ellipse 3 1 ∧
  -- Maximum length of CP
  ∃ (C : ℝ × ℝ) (P : ℝ × ℝ), 
    C ∈ Ellipse 3 1 ∧ 
    P = (-1, 0) ∧
    ∀ (C' : ℝ × ℝ), C' ∈ Ellipse 3 1 → 
      Real.sqrt ((C.1 - P.1)^2 + (C.2 - P.2)^2) ≥ 
      Real.sqrt ((C'.1 - P.1)^2 + (C'.2 - P.2)^2) ∧
    Real.sqrt ((C.1 - P.1)^2 + (C.2 - P.2)^2) = 9 * Real.sqrt 2 / 4 ∧
  -- Length of AB when CP is maximum
  ∃ (A B : ℝ × ℝ),
    A ∈ Circle 2 ∧
    B ∈ Circle 2 ∧
    (A.2 - B.2) * (C.1 - P.1) + (B.1 - A.1) * (C.2 - P.2) = 0 ∧
    Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) = Real.sqrt 62 / 2 :=
sorry

end NUMINAMATH_CALUDE_ellipse_and_chord_problem_l2193_219384


namespace NUMINAMATH_CALUDE_inscribed_circle_segment_ratio_l2193_219337

-- Define the triangle and circle
def Triangle (a b c : ℝ) := a > 0 ∧ b > 0 ∧ c > 0 ∧ a + b > c ∧ b + c > a ∧ c + a > b

def InscribedCircle (t : Triangle a b c) := 
  ∃ (r : ℝ), r > 0 ∧ 
  ∃ (x y z : ℝ), x > 0 ∧ y > 0 ∧ z > 0 ∧
  x + y = a ∧ y + z = b ∧ z + x = c ∧
  x + y + z = (a + b + c) / 2

-- Define the theorem
theorem inscribed_circle_segment_ratio 
  (t : Triangle 10 15 19) 
  (c : InscribedCircle t) :
  ∃ (r s : ℝ), r > 0 ∧ s > 0 ∧ r < s ∧ r + s = 10 ∧ r / s = 3 / 7 :=
sorry

end NUMINAMATH_CALUDE_inscribed_circle_segment_ratio_l2193_219337


namespace NUMINAMATH_CALUDE_road_trip_distance_l2193_219361

theorem road_trip_distance (first_day : ℝ) (second_day : ℝ) (third_day : ℝ) : 
  first_day = 200 →
  second_day = 3/4 * first_day →
  third_day = 1/2 * (first_day + second_day) →
  first_day + second_day + third_day = 525 := by
sorry

end NUMINAMATH_CALUDE_road_trip_distance_l2193_219361


namespace NUMINAMATH_CALUDE_ap_contains_sixth_power_l2193_219387

/-- An arithmetic progression containing squares and cubes contains a sixth power -/
theorem ap_contains_sixth_power (a h : ℕ) (p q : ℕ) : 
  0 < a → 0 < h → p ≠ q → p > 0 → q > 0 →
  (∃ k : ℕ, a + k * h = p^2) → 
  (∃ m : ℕ, a + m * h = q^3) → 
  (∃ n x : ℕ, a + n * h = x^6) :=
sorry

end NUMINAMATH_CALUDE_ap_contains_sixth_power_l2193_219387


namespace NUMINAMATH_CALUDE_square_plus_one_geq_two_abs_l2193_219333

theorem square_plus_one_geq_two_abs (x : ℝ) : x^2 + 1 ≥ 2 * |x| := by
  sorry

end NUMINAMATH_CALUDE_square_plus_one_geq_two_abs_l2193_219333


namespace NUMINAMATH_CALUDE_factorial_expression_l2193_219365

theorem factorial_expression : (Nat.factorial (Nat.factorial 4)) / (Nat.factorial 4) = Nat.factorial 23 := by
  sorry

end NUMINAMATH_CALUDE_factorial_expression_l2193_219365


namespace NUMINAMATH_CALUDE_negation_of_universal_proposition_l2193_219318

theorem negation_of_universal_proposition :
  (¬ ∀ x : ℝ, x > 0 → x + 1/x ≥ 2) ↔ (∃ x : ℝ, x > 0 ∧ x + 1/x < 2) := by sorry

end NUMINAMATH_CALUDE_negation_of_universal_proposition_l2193_219318


namespace NUMINAMATH_CALUDE_sum_of_squares_of_roots_l2193_219376

theorem sum_of_squares_of_roots (p q r : ℝ) : 
  (3 * p^3 - 7 * p^2 + 2 * p - 4 = 0) →
  (3 * q^3 - 7 * q^2 + 2 * q - 4 = 0) →
  (3 * r^3 - 7 * r^2 + 2 * r - 4 = 0) →
  p^2 + q^2 + r^2 = 37/9 := by
sorry

end NUMINAMATH_CALUDE_sum_of_squares_of_roots_l2193_219376


namespace NUMINAMATH_CALUDE_hyperbola_properties_l2193_219381

/-- Represents a hyperbola with semi-major axis a and semi-minor axis b -/
structure Hyperbola where
  a : ℝ
  b : ℝ
  h_pos : a > 0 ∧ b > 0

/-- Represents a line with slope m and y-intercept c -/
structure Line where
  m : ℝ
  c : ℝ

/-- The focal length and eccentricity of a hyperbola -/
structure HyperbolaProperties where
  focal_length : ℝ
  eccentricity : ℝ

theorem hyperbola_properties (C : Hyperbola) (l : Line) :
  l.m = Real.sqrt 3 ∧ 
  l.c = -4 * Real.sqrt 3 ∧ 
  (∃ (x y : ℝ), x^2 / C.a^2 - y^2 / C.b^2 = 1 ∧ y = l.m * x + l.c) ∧ 
  (∀ (x₁ y₁ x₂ y₂ : ℝ), 
    x₁^2 / C.a^2 - y₁^2 / C.b^2 = 1 ∧ y₁ = l.m * x₁ + l.c ∧ 
    x₂^2 / C.a^2 - y₂^2 / C.b^2 = 1 ∧ y₂ = l.m * x₂ + l.c → 
    x₁ = x₂ ∧ y₁ = y₂) →
  ∃ (props : HyperbolaProperties), 
    props.focal_length = 8 ∧ 
    props.eccentricity = 8/3 := by
  sorry

end NUMINAMATH_CALUDE_hyperbola_properties_l2193_219381


namespace NUMINAMATH_CALUDE_reciprocal_and_inverse_sum_l2193_219363

theorem reciprocal_and_inverse_sum (a b : ℚ) (ha : a = 1 / a) (hb : b = -b) :
  a^2007 + b^2007 = 1 ∨ a^2007 + b^2007 = -1 := by
  sorry

end NUMINAMATH_CALUDE_reciprocal_and_inverse_sum_l2193_219363


namespace NUMINAMATH_CALUDE_balls_in_bins_probability_ratio_l2193_219351

def number_of_balls : ℕ := 20
def number_of_bins : ℕ := 5

def p' : ℚ := (number_of_bins * (number_of_bins - 1) * (Nat.factorial 11) / 
  (Nat.factorial 4 * Nat.factorial 4 * Nat.factorial 4 * Nat.factorial 3)) / 
  (Nat.choose (number_of_balls + number_of_bins - 1) (number_of_bins - 1))

def q : ℚ := (Nat.factorial number_of_balls) / 
  ((Nat.factorial 4)^number_of_bins * Nat.factorial number_of_bins) / 
  (Nat.choose (number_of_balls + number_of_bins - 1) (number_of_bins - 1))

theorem balls_in_bins_probability_ratio : 
  p' / q = 8 / 57 := by sorry

end NUMINAMATH_CALUDE_balls_in_bins_probability_ratio_l2193_219351
