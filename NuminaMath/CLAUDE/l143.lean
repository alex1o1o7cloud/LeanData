import Mathlib

namespace NUMINAMATH_CALUDE_inequality_proof_l143_14373

theorem inequality_proof (a b c d e : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0) (he : e > 0) :
  Real.sqrt (a / (b + c + d + e)) + Real.sqrt (b / (a + c + d + e)) + 
  Real.sqrt (c / (a + b + d + e)) + Real.sqrt (d / (a + b + c + e)) + 
  Real.sqrt (e / (a + b + c + d)) ≥ 2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l143_14373


namespace NUMINAMATH_CALUDE_johns_total_time_l143_14328

/-- Represents the time John spent on various activities related to his travels and book writing --/
structure TravelTime where
  southAmerica : ℕ  -- Time spent exploring South America (in years)
  africa : ℕ        -- Time spent exploring Africa (in years)
  manuscriptTime : ℕ -- Time spent compiling notes into a manuscript (in months)
  editingTime : ℕ   -- Time spent finalizing the book with an editor (in months)

/-- Calculates the total time John spent on his adventures, note writing, and book creation --/
def totalTime (t : TravelTime) : ℕ :=
  -- Convert exploration time to months and add note-writing time
  (t.southAmerica * 12 + t.southAmerica * 6) +
  -- Convert Africa exploration time to months and add note-writing time
  (t.africa * 12 + t.africa * 4) +
  -- Add manuscript compilation and editing time
  t.manuscriptTime + t.editingTime

/-- Theorem stating that John's total time spent is 100 months --/
theorem johns_total_time :
  ∀ t : TravelTime,
    t.southAmerica = 3 ∧
    t.africa = 2 ∧
    t.manuscriptTime = 8 ∧
    t.editingTime = 6 →
    totalTime t = 100 :=
by
  sorry


end NUMINAMATH_CALUDE_johns_total_time_l143_14328


namespace NUMINAMATH_CALUDE_diamond_fifteen_two_l143_14387

-- Define the diamond operation
def diamond (a b : ℤ) : ℚ := a + a / (b + 1)

-- State the theorem
theorem diamond_fifteen_two : diamond 15 2 = 20 := by sorry

end NUMINAMATH_CALUDE_diamond_fifteen_two_l143_14387


namespace NUMINAMATH_CALUDE_crop_allocation_theorem_l143_14385

/-- Represents the yield function for crop A -/
def yield_A (x : ℝ) : ℝ := (2 + x) * (1.2 - 0.1 * x)

/-- Represents the maximum yield for crop A -/
def max_yield_A : ℝ := 4.9

/-- Represents the yield for crop B -/
def yield_B : ℝ := 10 * 0.5

/-- The total land area in square meters -/
def total_area : ℝ := 100

/-- The minimum required total yield in kg -/
def min_total_yield : ℝ := 496

theorem crop_allocation_theorem :
  ∃ (a : ℝ), a ≤ 40 ∧ a ≥ 0 ∧
  ∀ (x : ℝ), x ≤ 40 ∧ x ≥ 0 →
    max_yield_A * a + yield_B * (total_area - a) ≥ min_total_yield ∧
    (x > a → max_yield_A * x + yield_B * (total_area - x) < min_total_yield) :=
by sorry

end NUMINAMATH_CALUDE_crop_allocation_theorem_l143_14385


namespace NUMINAMATH_CALUDE_cuboid_surface_area_4_8_6_l143_14370

/-- The surface area of a cuboid with given dimensions -/
def cuboid_surface_area (length width height : ℝ) : ℝ :=
  2 * (length * width + length * height + width * height)

/-- Theorem stating that the surface area of a cuboid with dimensions 4x8x6 is 208 -/
theorem cuboid_surface_area_4_8_6 :
  cuboid_surface_area 4 8 6 = 208 := by
  sorry

#eval cuboid_surface_area 4 8 6

end NUMINAMATH_CALUDE_cuboid_surface_area_4_8_6_l143_14370


namespace NUMINAMATH_CALUDE_right_triangle_hypotenuse_l143_14315

theorem right_triangle_hypotenuse : 
  ∀ (a b c : ℝ), 
    a = 90 → 
    b = 120 → 
    c^2 = a^2 + b^2 → 
    c = 150 := by
sorry

end NUMINAMATH_CALUDE_right_triangle_hypotenuse_l143_14315


namespace NUMINAMATH_CALUDE_parabola_c_value_l143_14348

-- Define the parabola
def parabola (a b c x : ℝ) : ℝ := a * x^2 + b * x + c

-- Define the vertex condition
def vertex_condition (a b c : ℝ) : Prop :=
  parabola a b c 3 = -5

-- Define the point condition
def point_condition (a b c : ℝ) : Prop :=
  parabola a b c 1 = -3

theorem parabola_c_value :
  ∀ a b c : ℝ,
  vertex_condition a b c →
  point_condition a b c →
  c = -0.5 := by sorry

end NUMINAMATH_CALUDE_parabola_c_value_l143_14348


namespace NUMINAMATH_CALUDE_seating_arrangement_l143_14323

structure Person where
  name : String
  is_sitting : Prop

def M : Person := ⟨"M", false⟩
def I : Person := ⟨"I", true⟩
def P : Person := ⟨"P", true⟩
def A : Person := ⟨"A", false⟩

theorem seating_arrangement :
  (¬M.is_sitting) →
  (¬M.is_sitting → I.is_sitting) →
  (I.is_sitting → P.is_sitting) →
  (¬A.is_sitting) →
  (I.is_sitting ∧ P.is_sitting) := by
  sorry

end NUMINAMATH_CALUDE_seating_arrangement_l143_14323


namespace NUMINAMATH_CALUDE_escalator_problem_l143_14319

/-- Represents the escalator system in the shopping mall -/
structure EscalatorSystem where
  boyStepRate : ℕ
  girlStepRate : ℕ
  boyStepsToTop : ℕ
  girlStepsToTop : ℕ
  escalatorSpeed : ℝ
  exposedSteps : ℕ

/-- The conditions of the problem -/
def problemConditions (sys : EscalatorSystem) : Prop :=
  sys.boyStepRate = 2 * sys.girlStepRate ∧
  sys.boyStepsToTop = 27 ∧
  sys.girlStepsToTop = 18 ∧
  sys.escalatorSpeed > 0

/-- The theorem to prove -/
theorem escalator_problem (sys : EscalatorSystem) 
  (h : problemConditions sys) : 
  sys.exposedSteps = 54 ∧ 
  ∃ (boySteps : ℕ), boySteps = 198 ∧ 
    (boySteps = 3 * sys.boyStepsToTop + 2 * sys.exposedSteps) :=
sorry

end NUMINAMATH_CALUDE_escalator_problem_l143_14319


namespace NUMINAMATH_CALUDE_f_of_3_equals_9_l143_14343

-- Define the function f
def f (x : ℝ) : ℝ := x^2

-- State the theorem
theorem f_of_3_equals_9 : f 3 = 9 := by
  sorry

end NUMINAMATH_CALUDE_f_of_3_equals_9_l143_14343


namespace NUMINAMATH_CALUDE_adjacent_knights_probability_l143_14309

/-- The number of knights at the round table -/
def total_knights : ℕ := 30

/-- The number of knights chosen for the quest -/
def chosen_knights : ℕ := 5

/-- The probability that at least two of the chosen knights are sitting next to each other -/
def P : ℚ := 141505 / 142506

/-- Theorem stating the probability of adjacent chosen knights -/
theorem adjacent_knights_probability :
  (1 : ℚ) - (Nat.choose (total_knights - chosen_knights - (chosen_knights - 1)) (chosen_knights - 1) : ℚ) / 
  (Nat.choose total_knights chosen_knights : ℚ) = P := by sorry

end NUMINAMATH_CALUDE_adjacent_knights_probability_l143_14309


namespace NUMINAMATH_CALUDE_josh_extracurricular_hours_l143_14369

/-- Represents the number of days Josh has soccer practice in a week -/
def soccer_days : ℕ := 3

/-- Represents the number of hours Josh spends on soccer practice each day -/
def soccer_hours_per_day : ℝ := 2

/-- Represents the number of days Josh has band practice in a week -/
def band_days : ℕ := 2

/-- Represents the number of hours Josh spends on band practice each day -/
def band_hours_per_day : ℝ := 1.5

/-- Calculates the total hours Josh spends on extracurricular activities in a week -/
def total_extracurricular_hours : ℝ :=
  (soccer_days : ℝ) * soccer_hours_per_day + (band_days : ℝ) * band_hours_per_day

/-- Theorem stating that Josh spends 9 hours on extracurricular activities in a week -/
theorem josh_extracurricular_hours :
  total_extracurricular_hours = 9 := by
  sorry

end NUMINAMATH_CALUDE_josh_extracurricular_hours_l143_14369


namespace NUMINAMATH_CALUDE_investment_sum_l143_14389

/-- Proves that if a sum P invested at 18% p.a. simple interest for two years yields Rs. 600 more interest than if invested at 12% p.a. simple interest for the same period, then P = 5000. -/
theorem investment_sum (P : ℝ) : 
  P * (18 / 100) * 2 - P * (12 / 100) * 2 = 600 → P = 5000 := by
  sorry

end NUMINAMATH_CALUDE_investment_sum_l143_14389


namespace NUMINAMATH_CALUDE_correct_calculation_l143_14332

theorem correct_calculation (y : ℝ) : 3 * y^2 - 2 * y^2 = y^2 := by
  sorry

end NUMINAMATH_CALUDE_correct_calculation_l143_14332


namespace NUMINAMATH_CALUDE_cycling_time_problem_l143_14366

theorem cycling_time_problem (total_distance : ℝ) (total_time : ℝ) (initial_speed : ℝ) (reduced_speed : ℝ)
  (h1 : total_distance = 140)
  (h2 : total_time = 7)
  (h3 : initial_speed = 25)
  (h4 : reduced_speed = 15) :
  ∃ (energetic_time : ℝ), 
    energetic_time * initial_speed + (total_time - energetic_time) * reduced_speed = total_distance ∧
    energetic_time = 7/2 := by
  sorry

end NUMINAMATH_CALUDE_cycling_time_problem_l143_14366


namespace NUMINAMATH_CALUDE_percentage_relation_l143_14324

theorem percentage_relation (x y z : ℝ) (hx : x = 0.06 * z) (hy : y = 0.18 * z) (hz : z > 0) :
  x / y * 100 = 100 / 3 :=
sorry

end NUMINAMATH_CALUDE_percentage_relation_l143_14324


namespace NUMINAMATH_CALUDE_committee_selection_count_l143_14344

theorem committee_selection_count : Nat.choose 12 5 = 792 := by
  sorry

end NUMINAMATH_CALUDE_committee_selection_count_l143_14344


namespace NUMINAMATH_CALUDE_range_of_a_l143_14374

-- Define the sets A and B
def A (a : ℝ) : Set ℝ := {x | (x - 1) * (a - x) > 0}
def B : Set ℝ := {x | |x + 1| + |x - 2| ≤ 3}

-- Define the complement of A
def C_R_A (a : ℝ) : Set ℝ := (A a)ᶜ

-- State the theorem
theorem range_of_a :
  (∀ a : ℝ, (C_R_A a ∪ B) = Set.univ) ↔ (∀ a : ℝ, a ∈ Set.Icc (-1) 2) :=
sorry

end NUMINAMATH_CALUDE_range_of_a_l143_14374


namespace NUMINAMATH_CALUDE_percentage_of_girls_in_class_l143_14316

theorem percentage_of_girls_in_class (B G : ℕ) :
  (G + B / 2 = (3 * B) / 4) →
  (G * 100) / (B + G) = 20 :=
by
  sorry

end NUMINAMATH_CALUDE_percentage_of_girls_in_class_l143_14316


namespace NUMINAMATH_CALUDE_inequality_proof_l143_14372

theorem inequality_proof (a m n p : ℝ) 
  (h1 : a * Real.log a = 1)
  (h2 : m = Real.exp (1/2 + a))
  (h3 : Real.exp n = 3^a)
  (h4 : a^p = 2^Real.exp 1) : 
  n < p ∧ p < m := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l143_14372


namespace NUMINAMATH_CALUDE_square_sum_greater_than_product_l143_14397

theorem square_sum_greater_than_product {a b : ℝ} (h : a > b) : a^2 + b^2 > a*b := by
  sorry

end NUMINAMATH_CALUDE_square_sum_greater_than_product_l143_14397


namespace NUMINAMATH_CALUDE_circle_properties_l143_14326

-- Define the set of points (x, y) satisfying the equation
def S : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.1^2 + p.2^2 - 4*p.2 + 1 = 0}

-- Theorem statement
theorem circle_properties (p : ℝ × ℝ) (h : p ∈ S) :
  (∃ (z : ℝ), ∀ (q : ℝ × ℝ), q ∈ S → q.1 + q.2 ≤ z ∧ z = 2 + Real.sqrt 6) ∧
  (∀ (q : ℝ × ℝ), q ∈ S → q.1 ≠ 0 → -Real.sqrt 2 ≤ (q.2 + 1) / q.1 ∧ (q.2 + 1) / q.1 ≤ Real.sqrt 2) ∧
  (∀ (q : ℝ × ℝ), q ∈ S → 8 - 2*Real.sqrt 15 ≤ q.1^2 - 2*q.1 + q.2^2 + 1 ∧ 
                         q.1^2 - 2*q.1 + q.2^2 + 1 ≤ 8 + 2*Real.sqrt 15) :=
by
  sorry


end NUMINAMATH_CALUDE_circle_properties_l143_14326


namespace NUMINAMATH_CALUDE_quadratic_real_roots_l143_14396

theorem quadratic_real_roots (k : ℝ) :
  (∃ x : ℝ, x^2 - 6*x + 9*k = 0) ↔ k ≤ 1 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_real_roots_l143_14396


namespace NUMINAMATH_CALUDE_probability_both_defective_six_two_two_l143_14351

/-- The probability of both selected products being defective, given that one is defective -/
def probability_both_defective (total : ℕ) (defective : ℕ) (selected : ℕ) : ℚ :=
  if total ≥ defective ∧ total ≥ selected ∧ selected > 0 then
    (defective.choose (selected - 1)) / (total.choose 1 * (total - 1).choose (selected - 1))
  else
    0

theorem probability_both_defective_six_two_two :
  probability_both_defective 6 2 2 = 1 / 15 := by
  sorry

end NUMINAMATH_CALUDE_probability_both_defective_six_two_two_l143_14351


namespace NUMINAMATH_CALUDE_factorial_ratio_l143_14335

theorem factorial_ratio : Nat.factorial 52 / Nat.factorial 50 = 2652 := by
  sorry

end NUMINAMATH_CALUDE_factorial_ratio_l143_14335


namespace NUMINAMATH_CALUDE_min_integer_solution_inequality_l143_14327

theorem min_integer_solution_inequality :
  ∀ x : ℤ, (4 * (x + 1) + 2 > x - 1) ↔ (x ≥ -2) :=
by sorry

end NUMINAMATH_CALUDE_min_integer_solution_inequality_l143_14327


namespace NUMINAMATH_CALUDE_max_integer_difference_l143_14307

theorem max_integer_difference (x y : ℤ) (hx : 4 < x ∧ x < 6) (hy : 6 < y ∧ y < 10) :
  (∃ (a b : ℤ), 4 < a ∧ a < 6 ∧ 6 < b ∧ b < 10 ∧ b - a ≤ y - x) ∧ y - x ≤ 4 :=
by sorry

end NUMINAMATH_CALUDE_max_integer_difference_l143_14307


namespace NUMINAMATH_CALUDE_ferris_wheel_small_seats_l143_14345

/-- The number of small seats on the Ferris wheel -/
def small_seats : ℕ := 2

/-- The number of people each small seat can hold -/
def people_per_small_seat : ℕ := 14

/-- The total number of people who can ride on small seats -/
def total_people_on_small_seats : ℕ := small_seats * people_per_small_seat

theorem ferris_wheel_small_seats :
  total_people_on_small_seats = 28 := by sorry

end NUMINAMATH_CALUDE_ferris_wheel_small_seats_l143_14345


namespace NUMINAMATH_CALUDE_xiaohong_journey_time_l143_14395

/-- Represents Xiaohong's journey to the meeting venue -/
structure Journey where
  initialSpeed : ℝ
  totalTime : ℝ

/-- The conditions of Xiaohong's journey -/
def journeyConditions (j : Journey) : Prop :=
  j.initialSpeed * 30 + (j.initialSpeed * 1.25) * (j.totalTime - 55) = j.initialSpeed * j.totalTime

/-- Theorem stating that the total time of Xiaohong's journey is 155 minutes -/
theorem xiaohong_journey_time :
  ∃ j : Journey, journeyConditions j ∧ j.totalTime = 155 := by
  sorry


end NUMINAMATH_CALUDE_xiaohong_journey_time_l143_14395


namespace NUMINAMATH_CALUDE_normal_distribution_probability_l143_14398

/-- A random variable following a normal distribution -/
structure NormalRV where
  μ : ℝ
  σ : ℝ
  σ_pos : σ > 0

/-- The probability that a normal random variable is less than a given value -/
noncomputable def prob_less (ξ : NormalRV) (x : ℝ) : ℝ := sorry

/-- The probability that a normal random variable is greater than a given value -/
noncomputable def prob_greater (ξ : NormalRV) (x : ℝ) : ℝ := sorry

/-- The probability that a normal random variable is between two given values -/
noncomputable def prob_between (ξ : NormalRV) (a b : ℝ) : ℝ := sorry

/-- Theorem: For a normally distributed random variable ξ with 
    P(ξ < -2) = P(ξ > 2) = 0.3, P(-2 < ξ < 0) = 0.2 -/
theorem normal_distribution_probability (ξ : NormalRV) 
    (h1 : prob_less ξ (-2) = 0.3)
    (h2 : prob_greater ξ 2 = 0.3) :
    prob_between ξ (-2) 0 = 0.2 := by sorry

end NUMINAMATH_CALUDE_normal_distribution_probability_l143_14398


namespace NUMINAMATH_CALUDE_three_numbers_sum_l143_14306

theorem three_numbers_sum (a b c : ℝ) : 
  a ≤ b → b ≤ c → 
  b = 10 → 
  (a + b + c) / 3 = a - 15 → 
  (a + b + c) / 3 = c + 10 → 
  a + b + c = 45 := by
sorry

end NUMINAMATH_CALUDE_three_numbers_sum_l143_14306


namespace NUMINAMATH_CALUDE_second_company_base_rate_l143_14350

/-- The base rate of United Telephone in dollars -/
def united_base_rate : ℝ := 8.00

/-- The per-minute rate of United Telephone in dollars -/
def united_per_minute : ℝ := 0.25

/-- The per-minute rate of the second company in dollars -/
def second_per_minute : ℝ := 0.20

/-- The number of minutes at which the bills are equal -/
def equal_minutes : ℝ := 80

/-- The base rate of the second company in dollars -/
def second_base_rate : ℝ := 12.00

/-- Theorem stating that the base rate of the second company is $12.00 -/
theorem second_company_base_rate :
  united_base_rate + united_per_minute * equal_minutes =
  second_base_rate + second_per_minute * equal_minutes :=
by sorry

end NUMINAMATH_CALUDE_second_company_base_rate_l143_14350


namespace NUMINAMATH_CALUDE_sqrt_2_irrational_sqrt_2_only_irrational_in_set_l143_14378

-- Define what it means for a real number to be rational
def IsRational (x : ℝ) : Prop := ∃ (p q : ℤ), q ≠ 0 ∧ x = p / q

-- Define what it means for a real number to be irrational
def IsIrrational (x : ℝ) : Prop := ¬(IsRational x)

-- State the theorem
theorem sqrt_2_irrational : IsIrrational (Real.sqrt 2) := by
  sorry

-- Define the set of numbers from the original problem
def problem_numbers : Set ℝ := {0, -1, Real.sqrt 2, 3.14}

-- State that √2 is the only irrational number in the set
theorem sqrt_2_only_irrational_in_set : 
  ∀ x ∈ problem_numbers, IsIrrational x ↔ x = Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_2_irrational_sqrt_2_only_irrational_in_set_l143_14378


namespace NUMINAMATH_CALUDE_arcsin_equation_solution_l143_14381

theorem arcsin_equation_solution (x : ℝ) : 
  Real.arcsin (3 * x) - Real.arcsin x = π / 6 → 
  x = 1 / Real.sqrt (40 - 12 * Real.sqrt 3) ∨ 
  x = -1 / Real.sqrt (40 - 12 * Real.sqrt 3) := by
sorry

end NUMINAMATH_CALUDE_arcsin_equation_solution_l143_14381


namespace NUMINAMATH_CALUDE_points_six_units_from_negative_one_l143_14310

theorem points_six_units_from_negative_one :
  let a : ℝ := -1
  let distance : ℝ := 6
  let point_left : ℝ := a - distance
  let point_right : ℝ := a + distance
  point_left = -7 ∧ point_right = 5 := by
sorry

end NUMINAMATH_CALUDE_points_six_units_from_negative_one_l143_14310


namespace NUMINAMATH_CALUDE_quadratic_decreasing_parameter_range_l143_14322

/-- Given a quadratic function f(x) = -x^2 - 2ax - 3 that is decreasing on the interval (-2, +∞),
    prove that the parameter a is in the range [2, +∞). -/
theorem quadratic_decreasing_parameter_range 
  (f : ℝ → ℝ) 
  (a : ℝ) 
  (h1 : ∀ x, f x = -x^2 - 2*a*x - 3) 
  (h2 : ∀ x y, x > -2 → y > x → f y < f x) : 
  a ∈ Set.Ici 2 := by
sorry

end NUMINAMATH_CALUDE_quadratic_decreasing_parameter_range_l143_14322


namespace NUMINAMATH_CALUDE_johns_number_proof_l143_14367

theorem johns_number_proof : 
  ∃! x : ℕ, 
    10 ≤ x ∧ x < 100 ∧ 
    (∃ a b : ℕ, 
      4 * x + 17 = 10 * a + b ∧
      10 * b + a ≥ 91 ∧ 
      10 * b + a ≤ 95) ∧
    x = 8 := by
  sorry

end NUMINAMATH_CALUDE_johns_number_proof_l143_14367


namespace NUMINAMATH_CALUDE_cyclic_difference_sum_lower_bound_l143_14349

theorem cyclic_difference_sum_lower_bound 
  (a₁ a₂ a₃ a₄ a₅ a₆ a₇ : ℕ) 
  (h_distinct : a₁ ≠ a₂ ∧ a₁ ≠ a₃ ∧ a₁ ≠ a₄ ∧ a₁ ≠ a₅ ∧ a₁ ≠ a₆ ∧ a₁ ≠ a₇ ∧
                a₂ ≠ a₃ ∧ a₂ ≠ a₄ ∧ a₂ ≠ a₅ ∧ a₂ ≠ a₆ ∧ a₂ ≠ a₇ ∧
                a₃ ≠ a₄ ∧ a₃ ≠ a₅ ∧ a₃ ≠ a₆ ∧ a₃ ≠ a₇ ∧
                a₄ ≠ a₅ ∧ a₄ ≠ a₆ ∧ a₄ ≠ a₇ ∧
                a₅ ≠ a₆ ∧ a₅ ≠ a₇ ∧
                a₆ ≠ a₇) :
  (a₁ - a₂)^4 + (a₂ - a₃)^4 + (a₃ - a₄)^4 + (a₄ - a₅)^4 + 
  (a₅ - a₆)^4 + (a₆ - a₇)^4 + (a₇ - a₁)^4 ≥ 82 := by
  sorry


end NUMINAMATH_CALUDE_cyclic_difference_sum_lower_bound_l143_14349


namespace NUMINAMATH_CALUDE_childrens_ticket_cost_l143_14347

/-- Calculates the cost of a children's ticket given the total cost and other ticket prices --/
theorem childrens_ticket_cost 
  (adult_price : ℕ) 
  (senior_price : ℕ) 
  (total_cost : ℕ) 
  (num_adults : ℕ) 
  (num_seniors : ℕ) 
  (num_children : ℕ) :
  adult_price = 11 →
  senior_price = 9 →
  num_adults = 2 →
  num_seniors = 2 →
  num_children = 3 →
  total_cost = 64 →
  (total_cost - (num_adults * adult_price + num_seniors * senior_price)) / num_children = 8 :=
by
  sorry

#check childrens_ticket_cost

end NUMINAMATH_CALUDE_childrens_ticket_cost_l143_14347


namespace NUMINAMATH_CALUDE_airport_distance_l143_14352

theorem airport_distance (initial_speed initial_time final_speed : ℝ)
  (late_time early_time : ℝ) :
  initial_speed = 40 →
  initial_time = 1 →
  final_speed = 60 →
  late_time = 1.5 →
  early_time = 1 →
  ∃ (total_time total_distance : ℝ),
    total_distance = initial_speed * initial_time +
      final_speed * (total_time - initial_time - early_time) ∧
    total_time = (total_distance / initial_speed) - late_time ∧
    total_distance = 420 :=
by sorry

end NUMINAMATH_CALUDE_airport_distance_l143_14352


namespace NUMINAMATH_CALUDE_tangent_line_properties_l143_14331

-- Define the curve
def f (x : ℝ) : ℝ := 4 * x^2 - 6 * x + 3

-- Define the derivative of the curve
def f' (x : ℝ) : ℝ := 8 * x - 6

theorem tangent_line_properties :
  -- Part a: Tangent line parallel to y = 2x at (1, 1)
  (f' 1 = 2 ∧ f 1 = 1) ∧
  -- Part b: Tangent line perpendicular to y = x/4 at (1/4, 7/4)
  (f' (1/4) = -4 ∧ f (1/4) = 7/4) :=
by sorry

end NUMINAMATH_CALUDE_tangent_line_properties_l143_14331


namespace NUMINAMATH_CALUDE_largest_angle_in_specific_hexagon_l143_14388

/-- Represents the ratio of angles in a hexagon -/
structure HexagonRatio :=
  (a b c d e f : ℕ)

/-- Calculates the largest angle in a hexagon given a ratio of angles -/
def largestAngleInHexagon (ratio : HexagonRatio) : ℚ :=
  let sum := ratio.a + ratio.b + ratio.c + ratio.d + ratio.e + ratio.f
  let angleUnit := 720 / sum
  angleUnit * (max ratio.a (max ratio.b (max ratio.c (max ratio.d (max ratio.e ratio.f)))))

theorem largest_angle_in_specific_hexagon :
  largestAngleInHexagon ⟨2, 3, 3, 4, 4, 5⟩ = 1200 / 7 := by
  sorry

#eval largestAngleInHexagon ⟨2, 3, 3, 4, 4, 5⟩

end NUMINAMATH_CALUDE_largest_angle_in_specific_hexagon_l143_14388


namespace NUMINAMATH_CALUDE_min_value_theorem_l143_14336

theorem min_value_theorem (α₁ α₂ : ℝ) 
  (h : (1 / (2 + Real.sin α₁)) + (1 / (2 + Real.sin (2 * α₂))) = 2) :
  ∃ (k₁ k₂ : ℤ), ∀ (m₁ m₂ : ℤ), 
    |10 * Real.pi - α₁ - α₂| ≥ |10 * Real.pi - ((-π/2 + 2*↑k₁*π) + (-π/4 + ↑k₂*π))| ∧
    |10 * Real.pi - ((-π/2 + 2*↑k₁*π) + (-π/4 + ↑k₂*π))| = π/4 :=
by sorry

end NUMINAMATH_CALUDE_min_value_theorem_l143_14336


namespace NUMINAMATH_CALUDE_power_fraction_equality_l143_14305

theorem power_fraction_equality : (40 ^ 56) / (10 ^ 28) = 160 ^ 28 := by
  sorry

end NUMINAMATH_CALUDE_power_fraction_equality_l143_14305


namespace NUMINAMATH_CALUDE_grade_c_boxes_l143_14301

theorem grade_c_boxes (total : ℕ) (m n t : ℕ) 
  (h1 : total = 420)
  (h2 : 2 * t = m + n) : 
  (total / 3 : ℕ) = 140 := by sorry

end NUMINAMATH_CALUDE_grade_c_boxes_l143_14301


namespace NUMINAMATH_CALUDE_symmetry_sum_l143_14360

/-- Two points are symmetric about the y-axis if their x-coordinates are negatives of each other
    and their y-coordinates are equal. -/
def symmetric_about_y_axis (p1 p2 : ℝ × ℝ) : Prop :=
  p1.1 = -p2.1 ∧ p1.2 = p2.2

theorem symmetry_sum (a b : ℝ) :
  symmetric_about_y_axis (a, 5) (2, b) → a + b = 3 := by
  sorry

end NUMINAMATH_CALUDE_symmetry_sum_l143_14360


namespace NUMINAMATH_CALUDE_profit_percentage_l143_14371

/-- Given that the cost price of 25 articles equals the selling price of 18 articles,
    prove that the profit percentage is 700/18. -/
theorem profit_percentage (C S : ℝ) (h : 25 * C = 18 * S) :
  (S - C) / C * 100 = 700 / 18 := by
  sorry

end NUMINAMATH_CALUDE_profit_percentage_l143_14371


namespace NUMINAMATH_CALUDE_quadratic_roots_difference_l143_14337

theorem quadratic_roots_difference (a b c : ℝ) (r₁ r₂ : ℝ) : 
  a * r₁^2 + b * r₁ + c = 0 →
  a * r₂^2 + b * r₂ + c = 0 →
  a = 1 →
  b = -8 →
  c = 15 →
  r₁ + r₂ = 8 →
  ∃ n : ℤ, (r₁ + r₂ : ℝ) = n^2 →
  r₁ - r₂ = 2 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_roots_difference_l143_14337


namespace NUMINAMATH_CALUDE_condition_sufficient_not_necessary_l143_14312

-- Define the condition for m and n
def condition (m n : ℝ) : Prop := m < 0 ∧ 0 < n

-- Define what it means for the equation to represent a hyperbola
def is_hyperbola (m n : ℝ) : Prop := 
  ∃ (x y : ℝ), n * x^2 + m * y^2 = 1 ∧ (m < 0 ∧ n > 0) ∨ (m > 0 ∧ n < 0)

-- State the theorem
theorem condition_sufficient_not_necessary (m n : ℝ) :
  (condition m n → is_hyperbola m n) ∧ 
  ¬(is_hyperbola m n → condition m n) :=
sorry

end NUMINAMATH_CALUDE_condition_sufficient_not_necessary_l143_14312


namespace NUMINAMATH_CALUDE_fruit_seller_apples_l143_14383

/-- Proves that if a fruit seller sells 50% of his apples and is left with 5000 apples, 
    then he originally had 10000 apples. -/
theorem fruit_seller_apples (original : ℕ) (sold_percentage : ℚ) (remaining : ℕ) 
    (h1 : sold_percentage = 1/2)
    (h2 : remaining = 5000)
    (h3 : (1 - sold_percentage) * original = remaining) : 
  original = 10000 := by
  sorry

end NUMINAMATH_CALUDE_fruit_seller_apples_l143_14383


namespace NUMINAMATH_CALUDE_factorization_equality_l143_14390

theorem factorization_equality (x y : ℝ) : x^2 - 1 + 2*x*y + y^2 = (x+y+1)*(x+y-1) := by
  sorry

end NUMINAMATH_CALUDE_factorization_equality_l143_14390


namespace NUMINAMATH_CALUDE_greatest_x_given_lcm_l143_14318

def is_lcm (a b c m : ℕ) : Prop := 
  (∀ n : ℕ, n % a = 0 ∧ n % b = 0 ∧ n % c = 0 → m ∣ n) ∧
  (m % a = 0 ∧ m % b = 0 ∧ m % c = 0)

theorem greatest_x_given_lcm : 
  ∀ x : ℕ, is_lcm x 15 21 105 → x ≤ 105 :=
by sorry

end NUMINAMATH_CALUDE_greatest_x_given_lcm_l143_14318


namespace NUMINAMATH_CALUDE_dispersion_measures_l143_14391

-- Define a sample as a list of real numbers
def Sample := List ℝ

-- Define the statistics
def standardDeviation (s : Sample) : ℝ := sorry
def range (s : Sample) : ℝ := sorry
def mean (s : Sample) : ℝ := sorry
def median (s : Sample) : ℝ := sorry

-- Define a predicate for whether a statistic measures dispersion
def measuresDispersion (f : Sample → ℝ) : Prop := sorry

-- Theorem stating which statistics measure dispersion
theorem dispersion_measures (s : Sample) :
  measuresDispersion (standardDeviation) ∧
  measuresDispersion (range) ∧
  ¬measuresDispersion (mean) ∧
  ¬measuresDispersion (median) :=
sorry

end NUMINAMATH_CALUDE_dispersion_measures_l143_14391


namespace NUMINAMATH_CALUDE_tangent_line_at_point_A_l143_14353

/-- The function f(x) = x³ - 3x -/
def f (x : ℝ) : ℝ := x^3 - 3*x

/-- The derivative of f(x) -/
def f' (x : ℝ) : ℝ := 3*x^2 - 3

theorem tangent_line_at_point_A :
  ∃ (m b : ℝ), 
    (f 0 = 16) ∧ 
    (∀ x : ℝ, m * x + b = f' 0 * x + f 0) ∧
    (m = 9 ∧ b = 22) :=
sorry

end NUMINAMATH_CALUDE_tangent_line_at_point_A_l143_14353


namespace NUMINAMATH_CALUDE_mike_bought_21_books_l143_14361

/-- The number of books Mike bought at a yard sale -/
def books_bought (initial_books final_books : ℕ) : ℕ :=
  final_books - initial_books

/-- Theorem stating that Mike bought 21 books at the yard sale -/
theorem mike_bought_21_books :
  books_bought 35 56 = 21 := by
  sorry

end NUMINAMATH_CALUDE_mike_bought_21_books_l143_14361


namespace NUMINAMATH_CALUDE_geometric_sequence_third_term_l143_14346

/-- Given a geometric sequence {a_n} where a₁ = -2 and a₅ = -4, prove that a₃ = -2√2 -/
theorem geometric_sequence_third_term 
  (a : ℕ → ℝ) 
  (h_geometric : ∀ n, a (n + 1) / a n = a (n + 2) / a (n + 1)) 
  (h_a1 : a 1 = -2) 
  (h_a5 : a 5 = -4) : 
  a 3 = -2 * Real.sqrt 2 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_third_term_l143_14346


namespace NUMINAMATH_CALUDE_forgotten_angle_measure_l143_14382

theorem forgotten_angle_measure (n : ℕ) (h : n > 2) :
  (n - 1) * 180 - 2017 = 143 :=
sorry

end NUMINAMATH_CALUDE_forgotten_angle_measure_l143_14382


namespace NUMINAMATH_CALUDE_airline_capacity_is_2482_l143_14303

/-- Calculates the number of passengers an airline can accommodate daily --/
def airline_capacity (small_planes medium_planes large_planes : ℕ)
  (small_rows small_seats small_flights small_occupancy : ℕ)
  (medium_rows medium_seats medium_flights medium_occupancy : ℕ)
  (large_rows large_seats large_flights large_occupancy : ℕ) : ℕ :=
  let small_capacity := small_planes * small_rows * small_seats * small_flights * small_occupancy / 100
  let medium_capacity := medium_planes * medium_rows * medium_seats * medium_flights * medium_occupancy / 100
  let large_capacity := large_planes * large_rows * large_seats * large_flights * large_occupancy / 100
  small_capacity + medium_capacity + large_capacity

/-- The airline's daily passenger capacity is 2482 --/
theorem airline_capacity_is_2482 :
  airline_capacity 2 2 1 15 6 3 80 25 8 2 90 35 10 4 95 = 2482 := by
  sorry

end NUMINAMATH_CALUDE_airline_capacity_is_2482_l143_14303


namespace NUMINAMATH_CALUDE_simple_interest_time_proof_l143_14302

/-- The simple interest rate per annum -/
def simple_interest_rate : ℚ := 8 / 100

/-- The principal amount for simple interest -/
def simple_principal : ℚ := 1750.000000000002

/-- The principal amount for compound interest -/
def compound_principal : ℚ := 4000

/-- The compound interest rate per annum -/
def compound_interest_rate : ℚ := 10 / 100

/-- The time period for compound interest in years -/
def compound_time : ℕ := 2

/-- Function to calculate compound interest -/
def compound_interest (p : ℚ) (r : ℚ) (t : ℕ) : ℚ :=
  p * ((1 + r) ^ t - 1)

/-- The time period for simple interest in years -/
def simple_time : ℕ := 3

theorem simple_interest_time_proof :
  simple_principal * simple_interest_rate * simple_time =
  (1 / 2) * compound_interest compound_principal compound_interest_rate compound_time :=
by sorry

end NUMINAMATH_CALUDE_simple_interest_time_proof_l143_14302


namespace NUMINAMATH_CALUDE_problem_statement_l143_14356

theorem problem_statement :
  (∀ a : ℝ, (2 * a + 1) / a ≤ 1 ↔ a ∈ Set.Icc (-1) 0) ∧
  (∀ a : ℝ, (∃ x : ℝ, x ∈ Set.Ico 0 2 ∧ -x^3 + 3*x + 2*a - 1 = 0) ↔ a ∈ Set.Ico (-1/2) (3/2)) ∧
  (∀ a : ℝ, ((2 * a + 1) / a ≤ 1 ∨ (∃ x : ℝ, x ∈ Set.Ico 0 2 ∧ -x^3 + 3*x + 2*a - 1 = 0)) ↔ a ∈ Set.Ico (-1) (3/2)) :=
by sorry

end NUMINAMATH_CALUDE_problem_statement_l143_14356


namespace NUMINAMATH_CALUDE_james_age_l143_14394

/-- Proves that James' current age is 11 years old, given the conditions from the problem. -/
theorem james_age (julio_age : ℕ) (years_later : ℕ) (james_age : ℕ) : 
  julio_age = 36 → 
  years_later = 14 → 
  julio_age + years_later = 2 * (james_age + years_later) → 
  james_age = 11 :=
by
  sorry

end NUMINAMATH_CALUDE_james_age_l143_14394


namespace NUMINAMATH_CALUDE_jay_used_zero_l143_14341

/-- Represents the amount of paint in a gallon -/
def gallon : ℚ := 1

/-- Represents the amount of paint Dexter used in gallons -/
def dexter_used : ℚ := 3/8

/-- Represents the amount of paint left in gallons -/
def paint_left : ℚ := 1

/-- Represents the amount of paint Jay used in gallons -/
def jay_used : ℚ := gallon - dexter_used - paint_left

theorem jay_used_zero : jay_used = 0 := by sorry

end NUMINAMATH_CALUDE_jay_used_zero_l143_14341


namespace NUMINAMATH_CALUDE_smallest_m_for_nth_root_in_T_l143_14321

def T : Set ℂ := {z : ℂ | 1/2 ≤ z.re ∧ z.re ≤ Real.sqrt 2 / 2}

theorem smallest_m_for_nth_root_in_T : 
  (∀ n : ℕ, n ≥ 12 → ∃ z ∈ T, z^n = 1) ∧ 
  (∀ m : ℕ, m < 12 → ∃ n : ℕ, n ≥ m ∧ ∀ z ∈ T, z^n ≠ 1) :=
sorry

end NUMINAMATH_CALUDE_smallest_m_for_nth_root_in_T_l143_14321


namespace NUMINAMATH_CALUDE_max_students_distribution_l143_14365

def max_students (pens pencils : ℕ) : ℕ :=
  Nat.gcd pens pencils

theorem max_students_distribution (pens pencils : ℕ) :
  pens = 100 → pencils = 50 → max_students pens pencils = 50 := by
  sorry

end NUMINAMATH_CALUDE_max_students_distribution_l143_14365


namespace NUMINAMATH_CALUDE_inequality_system_solution_implies_a_greater_than_negative_one_l143_14325

theorem inequality_system_solution_implies_a_greater_than_negative_one :
  (∃ x : ℝ, x + a ≥ 0 ∧ 1 - 2*x > x - 2) → a > -1 :=
by sorry

end NUMINAMATH_CALUDE_inequality_system_solution_implies_a_greater_than_negative_one_l143_14325


namespace NUMINAMATH_CALUDE_eleven_students_like_sports_l143_14330

/-- The number of students who like basketball or cricket or both -/
def students_basketball_or_cricket (basketball : ℕ) (cricket : ℕ) (both : ℕ) : ℕ :=
  basketball + cricket - both

/-- Theorem stating that given the conditions, 11 students like basketball or cricket or both -/
theorem eleven_students_like_sports : students_basketball_or_cricket 9 8 6 = 11 := by
  sorry

end NUMINAMATH_CALUDE_eleven_students_like_sports_l143_14330


namespace NUMINAMATH_CALUDE_child_tickets_sold_l143_14377

/-- Proves the number of child's tickets sold in a movie theater -/
theorem child_tickets_sold (adult_price child_price total_tickets total_revenue : ℕ) 
  (h1 : adult_price = 7)
  (h2 : child_price = 4)
  (h3 : total_tickets = 900)
  (h4 : total_revenue = 5100) :
  ∃ (adult_tickets child_tickets : ℕ),
    adult_tickets + child_tickets = total_tickets ∧
    adult_price * adult_tickets + child_price * child_tickets = total_revenue ∧
    child_tickets = 400 := by
  sorry

end NUMINAMATH_CALUDE_child_tickets_sold_l143_14377


namespace NUMINAMATH_CALUDE_intersecting_chords_area_theorem_l143_14379

/-- Represents a circle with two intersecting chords -/
structure IntersectingChordsCircle where
  radius : ℝ
  chord_length : ℝ
  intersection_distance : ℝ

/-- Represents the area of a region in the form m*π - n*√d -/
structure RegionArea where
  m : ℕ
  n : ℕ
  d : ℕ

/-- Checks if a number is square-free (not divisible by the square of any prime) -/
def is_square_free (n : ℕ) : Prop :=
  ∀ p : ℕ, Nat.Prime p → (p * p) ∣ n → p = 1

/-- The main theorem statement -/
theorem intersecting_chords_area_theorem (circle : IntersectingChordsCircle)
  (h1 : circle.radius = 50)
  (h2 : circle.chord_length = 90)
  (h3 : circle.intersection_distance = 24) :
  ∃ (area : RegionArea), 
    (area.m > 0 ∧ area.n > 0 ∧ area.d > 0) ∧
    is_square_free area.d ∧
    ∃ (region_area : ℝ), region_area = area.m * Real.pi - area.n * Real.sqrt area.d :=
by sorry

end NUMINAMATH_CALUDE_intersecting_chords_area_theorem_l143_14379


namespace NUMINAMATH_CALUDE_percentage_increase_l143_14314

theorem percentage_increase (x : ℝ) (h : x = 114.4) : 
  (x - 88) / 88 * 100 = 30 := by
  sorry

end NUMINAMATH_CALUDE_percentage_increase_l143_14314


namespace NUMINAMATH_CALUDE_x_squared_congruence_l143_14320

theorem x_squared_congruence (x : ℤ) : 
  (5 * x ≡ 15 [ZMOD 25]) → (4 * x ≡ 20 [ZMOD 25]) → (x^2 ≡ 0 [ZMOD 25]) := by
  sorry

end NUMINAMATH_CALUDE_x_squared_congruence_l143_14320


namespace NUMINAMATH_CALUDE_track_circumference_is_620_l143_14355

/-- Represents the circular track and the movement of A and B -/
structure Track :=
  (circumference : ℝ)
  (speed_A : ℝ)
  (speed_B : ℝ)

/-- The conditions of the problem -/
def problem_conditions (track : Track) : Prop :=
  ∃ (t1 t2 : ℝ),
    t1 > 0 ∧ t2 > t1 ∧
    track.speed_B * t1 = 120 ∧
    track.speed_A * t1 + track.speed_B * t1 = track.circumference / 2 ∧
    track.speed_A * t2 = track.circumference - 50 ∧
    track.speed_B * t2 = track.circumference / 2 + 50

/-- The theorem stating that the track circumference is 620 yards -/
theorem track_circumference_is_620 (track : Track) :
  problem_conditions track → track.circumference = 620 := by
  sorry

#check track_circumference_is_620

end NUMINAMATH_CALUDE_track_circumference_is_620_l143_14355


namespace NUMINAMATH_CALUDE_unique_solution_for_all_y_l143_14342

theorem unique_solution_for_all_y :
  ∃! x : ℝ, ∀ y : ℝ, 10 * x * y - 15 * y + 5 * x - 7.5 = 0 :=
by
  sorry

end NUMINAMATH_CALUDE_unique_solution_for_all_y_l143_14342


namespace NUMINAMATH_CALUDE_m_value_l143_14334

theorem m_value (m : ℝ) : 
  let A : Set ℝ := {0, m, m^2 - 3*m + 2}
  2 ∈ A → m = 3 := by
sorry

end NUMINAMATH_CALUDE_m_value_l143_14334


namespace NUMINAMATH_CALUDE_direct_proportion_shift_right_l143_14384

/-- A linear function represented by its slope and y-intercept -/
structure LinearFunction where
  slope : ℝ
  intercept : ℝ

/-- Shift a linear function horizontally -/
def shift_right (f : LinearFunction) (shift : ℝ) : LinearFunction :=
  { slope := f.slope, intercept := f.slope * shift + f.intercept }

theorem direct_proportion_shift_right :
  let f : LinearFunction := { slope := -2, intercept := 0 }
  let shifted_f := shift_right f 3
  shifted_f.slope = -2 ∧ shifted_f.intercept = 6 := by sorry

end NUMINAMATH_CALUDE_direct_proportion_shift_right_l143_14384


namespace NUMINAMATH_CALUDE_breaks_required_correct_l143_14333

/-- Represents a chocolate bar of dimensions m × n -/
structure ChocolateBar where
  m : ℕ+
  n : ℕ+

/-- The number of breaks required to separate all 1 × 1 squares in a chocolate bar -/
def breaks_required (bar : ChocolateBar) : ℕ :=
  bar.m.val * bar.n.val - 1

/-- Theorem stating that the number of breaks required is correct -/
theorem breaks_required_correct (bar : ChocolateBar) :
  breaks_required bar = bar.m.val * bar.n.val - 1 :=
by sorry

end NUMINAMATH_CALUDE_breaks_required_correct_l143_14333


namespace NUMINAMATH_CALUDE_complex_equation_solution_l143_14339

theorem complex_equation_solution (z : ℂ) : (z - 1) * I = 1 + I → z = 2 - I := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l143_14339


namespace NUMINAMATH_CALUDE_sum_product_equality_l143_14399

theorem sum_product_equality : 1.25 * 67.875 + 125 * 6.7875 + 1250 * 0.053375 = 1000 := by
  sorry

end NUMINAMATH_CALUDE_sum_product_equality_l143_14399


namespace NUMINAMATH_CALUDE_area_midpoint_triangle_is_sqrt3_l143_14380

/-- A regular hexagon with side length 2 -/
structure RegularHexagon :=
  (side_length : ℝ)
  (is_regular : side_length = 2)

/-- The triangle formed by connecting midpoints of three adjacent regular hexagons -/
structure MidpointTriangle :=
  (hexagon1 : RegularHexagon)
  (hexagon2 : RegularHexagon)
  (hexagon3 : RegularHexagon)
  (are_adjacent : hexagon1 ≠ hexagon2 ∧ hexagon2 ≠ hexagon3 ∧ hexagon3 ≠ hexagon1)

/-- The area of the triangle formed by connecting midpoints of three adjacent regular hexagons -/
def area_midpoint_triangle (t : MidpointTriangle) : ℝ :=
  sorry

/-- Theorem stating that the area of the midpoint triangle is √3 -/
theorem area_midpoint_triangle_is_sqrt3 (t : MidpointTriangle) : 
  area_midpoint_triangle t = Real.sqrt 3 :=
sorry

end NUMINAMATH_CALUDE_area_midpoint_triangle_is_sqrt3_l143_14380


namespace NUMINAMATH_CALUDE_battle_station_staffing_l143_14376

/-- Represents the number of ways to staff Captain Zarnin's battle station -/
def staff_battle_station (total_applicants : ℕ) (suitable_resumes : ℕ) 
  (assistant_engineer : ℕ) (weapons_maintenance1 : ℕ) (weapons_maintenance2 : ℕ)
  (field_technician : ℕ) (radio_specialist : ℕ) : ℕ :=
  assistant_engineer * weapons_maintenance1 * weapons_maintenance2 * field_technician * radio_specialist

/-- Theorem stating the number of ways to staff the battle station -/
theorem battle_station_staffing :
  staff_battle_station 30 15 3 4 4 5 5 = 960 := by
  sorry

end NUMINAMATH_CALUDE_battle_station_staffing_l143_14376


namespace NUMINAMATH_CALUDE_class_cans_collection_l143_14392

/-- Calculates the total number of cans collected by a class given specific conditions -/
def totalCansCollected (totalStudents : ℕ) (cansPerHalf : ℕ) (nonCollectingStudents : ℕ) 
  (remainingStudents : ℕ) (cansPerRemaining : ℕ) : ℕ :=
  let halfStudents := totalStudents / 2
  let cansFromHalf := halfStudents * cansPerHalf
  let cansFromRemaining := remainingStudents * cansPerRemaining
  cansFromHalf + cansFromRemaining

/-- Theorem stating that under given conditions, the class collects 232 cans in total -/
theorem class_cans_collection : 
  totalCansCollected 30 12 2 13 4 = 232 := by
  sorry

end NUMINAMATH_CALUDE_class_cans_collection_l143_14392


namespace NUMINAMATH_CALUDE_inverse_proportion_problem_l143_14311

/-- Given that α is inversely proportional to β, prove that α = -8/3 when β = -3,
    given that α = 4 when β = 2. -/
theorem inverse_proportion_problem (α β : ℝ) (h1 : ∃ k, ∀ x y, x * y = k → (α = x ↔ β = y))
    (h2 : α = 4 ∧ β = 2) : β = -3 → α = -8/3 := by
  sorry

end NUMINAMATH_CALUDE_inverse_proportion_problem_l143_14311


namespace NUMINAMATH_CALUDE_number_division_problem_l143_14354

theorem number_division_problem :
  ∃ (N p q : ℝ),
    N / p = 8 ∧
    N / q = 18 ∧
    p - q = 0.20833333333333334 ∧
    N = 3 := by
  sorry

end NUMINAMATH_CALUDE_number_division_problem_l143_14354


namespace NUMINAMATH_CALUDE_graduating_class_size_l143_14362

theorem graduating_class_size 
  (geometry : ℕ) 
  (biology : ℕ) 
  (overlap_diff : ℕ) 
  (h1 : geometry = 144) 
  (h2 : biology = 119) 
  (h3 : overlap_diff = 88) :
  geometry + biology - min geometry biology = 263 :=
by sorry

end NUMINAMATH_CALUDE_graduating_class_size_l143_14362


namespace NUMINAMATH_CALUDE_no_positive_integer_solution_l143_14340

theorem no_positive_integer_solution :
  ¬ ∃ (a b : ℕ), a > 0 ∧ b > 0 ∧ 3 * a^2 = b^2 + 1 := by
  sorry

end NUMINAMATH_CALUDE_no_positive_integer_solution_l143_14340


namespace NUMINAMATH_CALUDE_example_is_quadratic_l143_14300

/-- Definition of a quadratic equation -/
def is_quadratic_equation (f : ℝ → ℝ) : Prop :=
  ∃ (a b c : ℝ), a ≠ 0 ∧ ∀ x, f x = a * x^2 + b * x + c

/-- The equation x^2 - 3x + 1 = 0 is a quadratic equation -/
theorem example_is_quadratic : is_quadratic_equation (fun x ↦ x^2 - 3*x + 1) := by
  sorry

end NUMINAMATH_CALUDE_example_is_quadratic_l143_14300


namespace NUMINAMATH_CALUDE_intersection_when_a_is_one_subset_condition_l143_14359

-- Define set A as the solution set of -x^2 - 2x + 8 = 0
def A : Set ℝ := {x | -x^2 - 2*x + 8 = 0}

-- Define set B as the solution set of ax - 1 ≤ 0
def B (a : ℝ) : Set ℝ := {x | a*x - 1 ≤ 0}

-- Theorem 1: When a = 1, A ∩ B = {-4}
theorem intersection_when_a_is_one : A ∩ B 1 = {-4} := by sorry

-- Theorem 2: A ⊆ B if and only if -1/4 ≤ a ≤ 1/2
theorem subset_condition : 
  ∀ a : ℝ, A ⊆ B a ↔ -1/4 ≤ a ∧ a ≤ 1/2 := by sorry

end NUMINAMATH_CALUDE_intersection_when_a_is_one_subset_condition_l143_14359


namespace NUMINAMATH_CALUDE_point_movement_to_y_axis_l143_14329

/-- Given a point P that is moved 1 unit to the right to point M on the y-axis, 
    prove that M has coordinates (0, -2) -/
theorem point_movement_to_y_axis (m : ℝ) : 
  let P : ℝ × ℝ := (m + 2, 2 * m + 4)
  let M : ℝ × ℝ := (P.1 + 1, P.2)
  M.1 = 0 → M = (0, -2) := by
  sorry

end NUMINAMATH_CALUDE_point_movement_to_y_axis_l143_14329


namespace NUMINAMATH_CALUDE_f_minus_one_gt_f_one_l143_14317

theorem f_minus_one_gt_f_one :
  ∀ f : ℝ → ℝ, Differentiable ℝ f → (∀ x, f x = x^2 + 2*x * (deriv f 2)) → f (-1) > f 1 := by
  sorry

end NUMINAMATH_CALUDE_f_minus_one_gt_f_one_l143_14317


namespace NUMINAMATH_CALUDE_girls_in_blues_class_l143_14358

/-- Calculates the number of girls in a class given the total number of students and the ratio of girls to boys -/
def girlsInClass (totalStudents : ℕ) (girlRatio boyRatio : ℕ) : ℕ :=
  (totalStudents * girlRatio) / (girlRatio + boyRatio)

/-- Theorem: In a class of 56 students with a girl to boy ratio of 4:3, there are 32 girls -/
theorem girls_in_blues_class :
  girlsInClass 56 4 3 = 32 := by
  sorry


end NUMINAMATH_CALUDE_girls_in_blues_class_l143_14358


namespace NUMINAMATH_CALUDE_triangle_side_length_l143_14357

theorem triangle_side_length (A B C : ℝ) (angleA : ℝ) (sideBC sideAB : ℝ) :
  angleA = 2 * Real.pi / 3 →
  sideBC = Real.sqrt 19 →
  sideAB = 2 →
  ∃ (sideAC : ℝ), sideAC = 3 ∧
    sideBC ^ 2 = sideAC ^ 2 + sideAB ^ 2 - 2 * sideAC * sideAB * Real.cos angleA :=
by sorry

end NUMINAMATH_CALUDE_triangle_side_length_l143_14357


namespace NUMINAMATH_CALUDE_problem_statement_l143_14375

def is_even_function (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = f x

def has_period (f : ℝ → ℝ) (p : ℝ) : Prop := ∀ x, f (x + p) = f x

theorem problem_statement (f : ℝ → ℝ) 
  (h1 : is_even_function f)
  (h2 : has_period f 2)
  (h3 : ∀ x, 0 ≤ x ∧ x ≤ 1 → f x = x) :
  f (-1) + f (-2017) = 2 := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l143_14375


namespace NUMINAMATH_CALUDE_factor_condition_l143_14363

theorem factor_condition (t : ℚ) : 
  (∃ k : ℚ, (X - t) * k = 3 * X^2 + 10 * X - 8) ↔ (t = 2/3 ∨ t = -4) :=
by sorry

end NUMINAMATH_CALUDE_factor_condition_l143_14363


namespace NUMINAMATH_CALUDE_no_arithmetic_progression_l143_14386

theorem no_arithmetic_progression (m : ℕ+) :
  ∃ σ : Fin (2^m.val) ↪ Fin (2^m.val),
    ∀ (i j k : Fin (2^m.val)), i < j → j < k →
      σ j - σ i ≠ σ k - σ j :=
by sorry

end NUMINAMATH_CALUDE_no_arithmetic_progression_l143_14386


namespace NUMINAMATH_CALUDE_even_increasing_function_properties_l143_14393

/-- A function f: ℝ → ℝ that is even and increasing on (-∞, 0) -/
def EvenIncreasingFunction (f : ℝ → ℝ) : Prop :=
  (∀ x, f (-x) = f x) ∧ 
  (∀ x y, x < y ∧ y ≤ 0 → f x < f y)

/-- Theorem stating properties of an even increasing function -/
theorem even_increasing_function_properties (f : ℝ → ℝ) 
  (hf : EvenIncreasingFunction f) : 
  (∀ x, f (-x) - f x = 0) ∧ 
  (∀ x y, 0 < x ∧ x < y → f y < f x) :=
by sorry

end NUMINAMATH_CALUDE_even_increasing_function_properties_l143_14393


namespace NUMINAMATH_CALUDE_eulers_formula_eulers_identity_complex_exp_sum_bound_l143_14368

-- Define the complex exponential function
noncomputable def cexp (z : ℂ) : ℂ := sorry

-- Define the imaginary unit
def i : ℂ := sorry

-- Define pi
noncomputable def π : ℝ := sorry

-- Theorem 1: Euler's formula
theorem eulers_formula (x : ℝ) : cexp (i * x) = Complex.cos x + i * Complex.sin x := by sorry

-- Theorem 2: Euler's identity
theorem eulers_identity : cexp (i * π) + 1 = 0 := by sorry

-- Theorem 3: Bound on sum of complex exponentials
theorem complex_exp_sum_bound (x : ℝ) : Complex.abs (cexp (i * x) + cexp (-i * x)) ≤ 2 := by sorry

end NUMINAMATH_CALUDE_eulers_formula_eulers_identity_complex_exp_sum_bound_l143_14368


namespace NUMINAMATH_CALUDE_no_solution_triple_inequality_l143_14338

theorem no_solution_triple_inequality :
  ¬ ∃ (x y z : ℝ), (|x| < |y - z| ∧ |y| < |z - x| ∧ |z| < |x - y|) :=
by sorry

end NUMINAMATH_CALUDE_no_solution_triple_inequality_l143_14338


namespace NUMINAMATH_CALUDE_crease_length_of_folded_equilateral_triangle_l143_14364

-- Define an equilateral triangle
structure EquilateralTriangle :=
  (side_length : ℝ)

-- Define the folded triangle
structure FoldedTriangle extends EquilateralTriangle :=
  (crease_length : ℝ)

-- Theorem statement
theorem crease_length_of_folded_equilateral_triangle 
  (triangle : EquilateralTriangle) 
  (h : triangle.side_length = 6) : 
  ∃ (folded : FoldedTriangle), 
    folded.side_length = triangle.side_length ∧ 
    folded.crease_length = 3 * Real.sqrt 3 :=
sorry

end NUMINAMATH_CALUDE_crease_length_of_folded_equilateral_triangle_l143_14364


namespace NUMINAMATH_CALUDE_dawns_lemonade_price_l143_14308

/-- The price of Dawn's lemonade in cents -/
def dawns_price : ℕ := sorry

/-- The number of glasses Bea sold -/
def bea_glasses : ℕ := 10

/-- The number of glasses Dawn sold -/
def dawn_glasses : ℕ := 8

/-- The price of Bea's lemonade in cents -/
def bea_price : ℕ := 25

/-- The difference in earnings between Bea and Dawn in cents -/
def earnings_difference : ℕ := 26

theorem dawns_lemonade_price :
  dawns_price = 28 ∧
  bea_glasses * bea_price = dawn_glasses * dawns_price + earnings_difference :=
sorry

end NUMINAMATH_CALUDE_dawns_lemonade_price_l143_14308


namespace NUMINAMATH_CALUDE_area_of_intersection_region_l143_14313

noncomputable def f₀ (x : ℝ) : ℝ := |x|

noncomputable def f₁ (x : ℝ) : ℝ := |f₀ x - 1|

noncomputable def f₂ (x : ℝ) : ℝ := |f₁ x - 2|

theorem area_of_intersection_region (f₀ f₁ f₂ : ℝ → ℝ) :
  (f₀ = fun x ↦ |x|) →
  (f₁ = fun x ↦ |f₀ x - 1|) →
  (f₂ = fun x ↦ |f₁ x - 2|) →
  (∫ x in (-3)..(3), min (f₂ x) 0) = -7 :=
by sorry

end NUMINAMATH_CALUDE_area_of_intersection_region_l143_14313


namespace NUMINAMATH_CALUDE_complement_B_intersection_A_complement_B_l143_14304

-- Define the universal set U as the real numbers
def U : Set ℝ := Set.univ

-- Define set A
def A : Set ℝ := {x | x < 0}

-- Define set B
def B : Set ℝ := {x | x > 1}

-- Theorem for the complement of B with respect to U
theorem complement_B : Set.compl B = {x : ℝ | x ≤ 1} := by sorry

-- Theorem for the intersection of A and the complement of B
theorem intersection_A_complement_B : A ∩ Set.compl B = {x : ℝ | x < 0} := by sorry

end NUMINAMATH_CALUDE_complement_B_intersection_A_complement_B_l143_14304
