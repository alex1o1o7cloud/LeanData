import Mathlib

namespace NUMINAMATH_CALUDE_school_trip_combinations_l845_84552

/-- The number of different combinations of riding groups and ride choices -/
def ride_combinations (total_people : ℕ) (group_size : ℕ) (ride_choices : ℕ) : ℕ :=
  Nat.choose total_people group_size * ride_choices

/-- Theorem: Given 8 people, rides of 4, and 2 choices, there are 140 combinations -/
theorem school_trip_combinations :
  ride_combinations 8 4 2 = 140 := by
  sorry

end NUMINAMATH_CALUDE_school_trip_combinations_l845_84552


namespace NUMINAMATH_CALUDE_sock_order_ratio_l845_84501

/-- Represents the number of pairs of socks -/
structure SockOrder where
  black : ℕ
  blue : ℕ

/-- Represents the price of socks -/
structure SockPrice where
  blue : ℝ

/-- Calculates the total cost of a sock order given the prices -/
def totalCost (order : SockOrder) (price : SockPrice) : ℝ :=
  (order.black * 3 * price.blue) + (order.blue * price.blue)

/-- The theorem to be proved -/
theorem sock_order_ratio (original : SockOrder) (price : SockPrice) : 
  original.black = 5 →
  totalCost { black := original.blue, blue := original.black } price = 1.6 * totalCost original price →
  (original.black : ℝ) / original.blue = 5 / 14 := by
  sorry

#check sock_order_ratio

end NUMINAMATH_CALUDE_sock_order_ratio_l845_84501


namespace NUMINAMATH_CALUDE_second_car_speed_l845_84571

/-- 
Given two cars starting from opposite ends of a 105-mile highway, 
with one car traveling at 15 mph and both meeting after 3 hours, 
prove that the speed of the second car is 20 mph.
-/
theorem second_car_speed 
  (highway_length : ℝ) 
  (first_car_speed : ℝ) 
  (meeting_time : ℝ) 
  (h1 : highway_length = 105) 
  (h2 : first_car_speed = 15) 
  (h3 : meeting_time = 3) : 
  ∃ (second_car_speed : ℝ), 
    first_car_speed * meeting_time + second_car_speed * meeting_time = highway_length ∧ 
    second_car_speed = 20 := by
  sorry

end NUMINAMATH_CALUDE_second_car_speed_l845_84571


namespace NUMINAMATH_CALUDE_function_lower_bound_l845_84517

theorem function_lower_bound (a : ℝ) (ha : a > 0) :
  ∀ x : ℝ, a * (Real.exp x + a) - x > 2 * Real.log a + 3/2 := by
  sorry

end NUMINAMATH_CALUDE_function_lower_bound_l845_84517


namespace NUMINAMATH_CALUDE_square_sum_given_product_and_sum_l845_84500

/-- Given two real numbers p and q satisfying pq = 16 and p + q = 8, 
    prove that p^2 + q^2 = 32. -/
theorem square_sum_given_product_and_sum (p q : ℝ) 
  (h1 : p * q = 16) (h2 : p + q = 8) : p^2 + q^2 = 32 := by
  sorry

end NUMINAMATH_CALUDE_square_sum_given_product_and_sum_l845_84500


namespace NUMINAMATH_CALUDE_grade_study_sample_size_l845_84598

/-- Represents a statistical study of student grades -/
structure GradeStudy where
  total_students : ℕ
  selected_cards : ℕ

/-- Defines the sample size of a grade study -/
def sample_size (study : GradeStudy) : ℕ := study.selected_cards

/-- Theorem: The sample size of a study with 2000 total students and 200 selected cards is 200 -/
theorem grade_study_sample_size :
  ∀ (study : GradeStudy),
    study.total_students = 2000 →
    study.selected_cards = 200 →
    sample_size study = 200 := by
  sorry

end NUMINAMATH_CALUDE_grade_study_sample_size_l845_84598


namespace NUMINAMATH_CALUDE_parabola_c_value_l845_84569

/-- A parabola with equation x = ay^2 + by + c -/
structure Parabola where
  a : ℝ
  b : ℝ
  c : ℝ

/-- The x-coordinate of a point on the parabola given its y-coordinate -/
def Parabola.x_coord (p : Parabola) (y : ℝ) : ℝ :=
  p.a * y^2 + p.b * y + p.c

theorem parabola_c_value (p : Parabola) :
  p.x_coord (-1) = 3 →  -- vertex condition
  p.x_coord (-2) = 1 →  -- point condition
  p.c = 1 := by
sorry

end NUMINAMATH_CALUDE_parabola_c_value_l845_84569


namespace NUMINAMATH_CALUDE_temperature_peak_l845_84585

theorem temperature_peak (t : ℝ) : 
  (∀ s : ℝ, -s^2 + 10*s + 60 = 80 → s ≤ 5 + Real.sqrt 5) ∧ 
  (-((5 + Real.sqrt 5)^2) + 10*(5 + Real.sqrt 5) + 60 = 80) := by
sorry

end NUMINAMATH_CALUDE_temperature_peak_l845_84585


namespace NUMINAMATH_CALUDE_classroom_discussion_group_l845_84574

def group_sizes : List Nat := [2, 3, 5, 6, 7, 8, 11, 12, 13, 17, 20, 22, 24]

theorem classroom_discussion_group (
  total_groups : Nat) 
  (lecture_groups : Nat) 
  (chinese_lecture_ratio : Nat) 
  (h1 : total_groups = 13)
  (h2 : lecture_groups = 12)
  (h3 : chinese_lecture_ratio = 6)
  (h4 : group_sizes.length = total_groups)
  (h5 : group_sizes.sum = 150) :
  ∃ x : Nat, x ∈ group_sizes ∧ x % 7 = 4 := by
  sorry

end NUMINAMATH_CALUDE_classroom_discussion_group_l845_84574


namespace NUMINAMATH_CALUDE_domain_of_f_l845_84572

noncomputable def f (x : ℝ) : ℝ := (2 * x + 3) / (x + 5)

theorem domain_of_f : 
  {x : ℝ | ∃ y, f x = y} = {x : ℝ | x < -5 ∨ x > -5} := by sorry

end NUMINAMATH_CALUDE_domain_of_f_l845_84572


namespace NUMINAMATH_CALUDE_alice_paid_24_percent_l845_84512

-- Define the suggested retail price
def suggested_retail_price : ℝ := 100

-- Define the marked price as 60% of the suggested retail price
def marked_price : ℝ := 0.6 * suggested_retail_price

-- Define Alice's purchase price as 40% of the marked price
def alice_price : ℝ := 0.4 * marked_price

-- Theorem to prove
theorem alice_paid_24_percent :
  alice_price / suggested_retail_price = 0.24 := by
  sorry

end NUMINAMATH_CALUDE_alice_paid_24_percent_l845_84512


namespace NUMINAMATH_CALUDE_expected_value_of_game_l845_84564

-- Define the die
def die := Finset.range 10

-- Define prime numbers on the die
def primes : Finset ℕ := {2, 3, 5, 7}

-- Define composite numbers on the die
def composites : Finset ℕ := {4, 6, 8, 9, 10}

-- Define the winnings function
def winnings (n : ℕ) : ℚ :=
  if n ∈ primes then n
  else if n ∈ composites then 0
  else -5

-- Theorem statement
theorem expected_value_of_game : 
  (die.sum (fun i => winnings i) : ℚ) / 10 = 18 / 100 := by sorry

end NUMINAMATH_CALUDE_expected_value_of_game_l845_84564


namespace NUMINAMATH_CALUDE_expected_difference_zero_l845_84521

-- Define the die outcomes
inductive DieOutcome
  | prime : DieOutcome
  | composite : DieOutcome
  | nocereal : DieOutcome
  | reroll : DieOutcome

-- Define the probability of each outcome
def outcomeProb (outcome : DieOutcome) : ℚ :=
  match outcome with
  | DieOutcome.prime => 3/7
  | DieOutcome.composite => 3/7
  | DieOutcome.nocereal => 1/7
  | DieOutcome.reroll => 0

-- Define the number of days in a non-leap year
def daysInYear : ℕ := 365

-- Statement of the theorem
theorem expected_difference_zero :
  (outcomeProb DieOutcome.prime - outcomeProb DieOutcome.composite) * daysInYear = 0 := by
  sorry


end NUMINAMATH_CALUDE_expected_difference_zero_l845_84521


namespace NUMINAMATH_CALUDE_container_capacity_proof_l845_84502

/-- The capacity of a container in liters -/
def container_capacity : ℝ := 100

/-- The initial fill level of the container as a percentage -/
def initial_fill : ℝ := 30

/-- The final fill level of the container as a percentage -/
def final_fill : ℝ := 75

/-- The amount of water added to the container in liters -/
def water_added : ℝ := 45

theorem container_capacity_proof :
  (final_fill / 100 * container_capacity) - (initial_fill / 100 * container_capacity) = water_added :=
sorry

end NUMINAMATH_CALUDE_container_capacity_proof_l845_84502


namespace NUMINAMATH_CALUDE_lower_half_plane_inequality_l845_84562

/-- Given a line l passing through points A(2,1) and B(-1,3), 
    the inequality 2x + 3y - 7 ≤ 0 represents the lower half-plane including line l. -/
theorem lower_half_plane_inequality (x y : ℝ) : 
  let l : Set (ℝ × ℝ) := {p | ∃ t : ℝ, p = (2 - 3*t, 1 + 2*t)}
  (x, y) ∈ l ∨ (∃ p ∈ l, y < p.2) ↔ 2*x + 3*y - 7 ≤ 0 := by
  sorry

end NUMINAMATH_CALUDE_lower_half_plane_inequality_l845_84562


namespace NUMINAMATH_CALUDE_star_properties_l845_84529

def star (x y : ℝ) : ℝ := (x - 1) * (y - 1) - 1

theorem star_properties :
  (∀ x y : ℝ, star x y = star y x) ∧
  (∃ x y z : ℝ, star x (y + z) ≠ star x y + star x z) ∧
  (∀ x : ℝ, star (x + 1) (x - 1) = star x x - 1) ∧
  (∀ e : ℝ, ∃ x : ℝ, star x e ≠ x) ∧
  (∃ x y z : ℝ, star (star x y) z ≠ star x (star y z)) := by
  sorry

end NUMINAMATH_CALUDE_star_properties_l845_84529


namespace NUMINAMATH_CALUDE_rain_probability_l845_84514

theorem rain_probability (p : ℝ) (h : p = 1 / 2) : 
  1 - (1 - p)^4 = 15 / 16 := by
  sorry

end NUMINAMATH_CALUDE_rain_probability_l845_84514


namespace NUMINAMATH_CALUDE_logarithmic_function_fixed_point_l845_84536

theorem logarithmic_function_fixed_point (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) :
  let f : ℝ → ℝ := λ x ↦ Real.log x / Real.log a + 1
  f 1 = 1 := by sorry

end NUMINAMATH_CALUDE_logarithmic_function_fixed_point_l845_84536


namespace NUMINAMATH_CALUDE_function_interval_theorem_l845_84524

theorem function_interval_theorem (a b : Real) :
  let f := fun x => -1/2 * x^2 + 13/2
  (∀ x ∈ Set.Icc a b, f x ≥ 2*a) ∧
  (∀ x ∈ Set.Icc a b, f x ≤ 2*b) ∧
  (∃ x ∈ Set.Icc a b, f x = 2*a) ∧
  (∃ x ∈ Set.Icc a b, f x = 2*b) →
  ((a = 1 ∧ b = 3) ∨ (a = -2 - Real.sqrt 17 ∧ b = 13/4)) := by
  sorry

#check function_interval_theorem

end NUMINAMATH_CALUDE_function_interval_theorem_l845_84524


namespace NUMINAMATH_CALUDE_problem_statement_l845_84590

theorem problem_statement :
  (∃ x : ℝ, x^2 - x + 1 ≥ 0) ∧
  ¬(∀ a b : ℝ, a^2 < b^2 → a < b) :=
by sorry

end NUMINAMATH_CALUDE_problem_statement_l845_84590


namespace NUMINAMATH_CALUDE_gcd_equality_l845_84534

theorem gcd_equality (a b c : ℕ+) (h : Nat.gcd (a^2 - 1) (Nat.gcd (b^2 - 1) (c^2 - 1)) = 1) :
  Nat.gcd (a*b + c) (Nat.gcd (b*c + a) (c*a + b)) = Nat.gcd a (Nat.gcd b c) := by
  sorry

end NUMINAMATH_CALUDE_gcd_equality_l845_84534


namespace NUMINAMATH_CALUDE_expected_heads_value_l845_84568

/-- The number of coins -/
def num_coins : ℕ := 100

/-- The probability of a coin showing heads after a single flip -/
def prob_heads : ℚ := 1 / 2

/-- The maximum number of flips for each coin -/
def max_flips : ℕ := 4

/-- The probability of a coin showing heads after at most four flips -/
def prob_heads_after_four_flips : ℚ :=
  1 - (1 - prob_heads) ^ max_flips

/-- The expected number of coins showing heads after the series of flips -/
def expected_heads : ℚ := num_coins * prob_heads_after_four_flips

theorem expected_heads_value :
  expected_heads = 93.75 := by sorry

end NUMINAMATH_CALUDE_expected_heads_value_l845_84568


namespace NUMINAMATH_CALUDE_candy_remaining_l845_84583

theorem candy_remaining (initial_candy : ℕ) (people : ℕ) (eaten_per_person : ℕ) 
  (h1 : initial_candy = 68) 
  (h2 : people = 2) 
  (h3 : eaten_per_person = 4) : 
  initial_candy - (people * eaten_per_person) = 60 := by
  sorry

end NUMINAMATH_CALUDE_candy_remaining_l845_84583


namespace NUMINAMATH_CALUDE_function_cycle_existence_l845_84565

theorem function_cycle_existence :
  ∃ (f : ℝ → ℝ) (x₁ x₂ x₃ x₄ x₅ : ℝ),
    (∃ (a b c d : ℝ), ∀ x, f x = (a * x + b) / (c * x + d)) ∧
    x₁ ≠ x₂ ∧ x₁ ≠ x₃ ∧ x₁ ≠ x₄ ∧ x₁ ≠ x₅ ∧
    x₂ ≠ x₃ ∧ x₂ ≠ x₄ ∧ x₂ ≠ x₅ ∧
    x₃ ≠ x₄ ∧ x₃ ≠ x₅ ∧
    x₄ ≠ x₅ ∧
    f x₁ = x₂ ∧ f x₂ = x₃ ∧ f x₃ = x₄ ∧ f x₄ = x₅ ∧ f x₅ = x₁ := by
  sorry

end NUMINAMATH_CALUDE_function_cycle_existence_l845_84565


namespace NUMINAMATH_CALUDE_eleven_steps_seven_moves_l845_84592

/-- The number of ways to climb a staircase with a given number of steps in a fixed number of moves. -/
def climbStairs (totalSteps : ℕ) (requiredMoves : ℕ) : ℕ :=
  -- The actual implementation is not provided here
  sorry

/-- Theorem stating that there are 35 ways to climb 11 steps in 7 moves -/
theorem eleven_steps_seven_moves : climbStairs 11 7 = 35 := by
  sorry

end NUMINAMATH_CALUDE_eleven_steps_seven_moves_l845_84592


namespace NUMINAMATH_CALUDE_positive_root_iff_p_in_set_l845_84584

-- Define the polynomial equation
def f (p x : ℝ) : ℝ := x^4 + 4*p*x^3 + x^2 + 4*p*x + 4

-- Define the set of p values
def P : Set ℝ := {p | p < -Real.sqrt 2 / 2 ∨ p > Real.sqrt 2 / 2}

-- Theorem statement
theorem positive_root_iff_p_in_set (p : ℝ) :
  (∃ x : ℝ, x > 0 ∧ f p x = 0) ↔ p ∈ P :=
sorry

end NUMINAMATH_CALUDE_positive_root_iff_p_in_set_l845_84584


namespace NUMINAMATH_CALUDE_not_isosceles_l845_84547

/-- A set of three distinct real numbers that can form the sides of a triangle -/
structure TriangleSet where
  a : ℝ
  b : ℝ
  c : ℝ
  distinct : a ≠ b ∧ b ≠ c ∧ c ≠ a
  triangle_inequality : a + b > c ∧ b + c > a ∧ c + a > b

/-- The triangle formed by a TriangleSet cannot be isosceles -/
theorem not_isosceles (S : TriangleSet) : ¬(S.a = S.b ∨ S.b = S.c ∨ S.c = S.a) :=
sorry

end NUMINAMATH_CALUDE_not_isosceles_l845_84547


namespace NUMINAMATH_CALUDE_diamonds_in_G_15_l845_84578

-- Define the sequence G
def G : ℕ → ℕ
| 0 => 1  -- G_1 has 1 diamond
| n + 1 => G n + 4 * (n + 2)  -- G_{n+1} adds 4 sides with (n+2) more diamonds each

-- Theorem statement
theorem diamonds_in_G_15 : G 14 = 1849 := by
  sorry

end NUMINAMATH_CALUDE_diamonds_in_G_15_l845_84578


namespace NUMINAMATH_CALUDE_smallest_reciprocal_l845_84542

theorem smallest_reciprocal (a b c d e : ℚ) : 
  a = 1/4 → b = 3/7 → c = -2 → d = 10 → e = 2023 →
  (1/c < 1/a ∧ 1/c < 1/b ∧ 1/c < 1/d ∧ 1/c < 1/e) := by
  sorry

end NUMINAMATH_CALUDE_smallest_reciprocal_l845_84542


namespace NUMINAMATH_CALUDE_floor_minus_x_is_zero_l845_84513

theorem floor_minus_x_is_zero (x : ℝ) (h : ⌈x⌉ - ⌊x⌋ = 0) : ⌊x⌋ - x = 0 := by
  sorry

end NUMINAMATH_CALUDE_floor_minus_x_is_zero_l845_84513


namespace NUMINAMATH_CALUDE_college_entrance_exam_score_l845_84567

theorem college_entrance_exam_score
  (total_questions : ℕ)
  (answered_questions : ℕ)
  (raw_score : ℚ)
  (h1 : total_questions = 85)
  (h2 : answered_questions = 82)
  (h3 : raw_score = 67)
  (h4 : answered_questions ≤ total_questions) :
  ∃ (correct_answers : ℕ),
    correct_answers ≤ answered_questions ∧
    (correct_answers : ℚ) - 0.25 * ((answered_questions : ℚ) - (correct_answers : ℚ)) = raw_score ∧
    correct_answers = 69 :=
by sorry

end NUMINAMATH_CALUDE_college_entrance_exam_score_l845_84567


namespace NUMINAMATH_CALUDE_fred_has_ten_balloons_l845_84556

/-- The number of red balloons Fred has -/
def fred_balloons (total sam dan : ℕ) : ℕ := total - (sam + dan)

/-- Theorem stating that Fred has 10 red balloons -/
theorem fred_has_ten_balloons (total sam dan : ℕ) 
  (h_total : total = 72)
  (h_sam : sam = 46)
  (h_dan : dan = 16) :
  fred_balloons total sam dan = 10 := by
  sorry

end NUMINAMATH_CALUDE_fred_has_ten_balloons_l845_84556


namespace NUMINAMATH_CALUDE_vegetable_price_calculation_l845_84577

/-- The price of vegetables and the final cost after discount -/
theorem vegetable_price_calculation :
  let cucumber_price : ℝ := 5
  let tomato_price : ℝ := cucumber_price * 0.8
  let bell_pepper_price : ℝ := cucumber_price * 1.5
  let total_cost : ℝ := 2 * tomato_price + 3 * cucumber_price + 4 * bell_pepper_price
  let discount_rate : ℝ := 0.1
  let final_price : ℝ := total_cost * (1 - discount_rate)
  final_price = 47.7 := by
sorry


end NUMINAMATH_CALUDE_vegetable_price_calculation_l845_84577


namespace NUMINAMATH_CALUDE_average_height_theorem_l845_84540

/-- The height difference between Itzayana and Zora in inches -/
def height_diff_itzayana_zora : ℝ := 4

/-- The height difference between Brixton and Zora in inches -/
def height_diff_brixton_zora : ℝ := 8

/-- Zara's height in inches -/
def height_zara : ℝ := 64

/-- Jaxon's height in centimeters -/
def height_jaxon_cm : ℝ := 170

/-- Conversion factor from centimeters to inches -/
def cm_to_inch : ℝ := 2.54

/-- The number of people -/
def num_people : ℕ := 5

theorem average_height_theorem :
  let height_brixton : ℝ := height_zara
  let height_zora : ℝ := height_brixton - height_diff_brixton_zora
  let height_itzayana : ℝ := height_zora + height_diff_itzayana_zora
  let height_jaxon : ℝ := height_jaxon_cm / cm_to_inch
  (height_itzayana + height_zora + height_brixton + height_zara + height_jaxon) / num_people = 62.2 := by
  sorry

end NUMINAMATH_CALUDE_average_height_theorem_l845_84540


namespace NUMINAMATH_CALUDE_quadrilateral_area_l845_84545

-- Define the vertices of the quadrilateral
def v1 : ℝ × ℝ := (0, 0)
def v2 : ℝ × ℝ := (4, 3)
def v3 : ℝ × ℝ := (7, 0)
def v4 : ℝ × ℝ := (4, 4)

-- Define the quadrilateral as a list of vertices
def quadrilateral : List (ℝ × ℝ) := [v1, v2, v3, v4]

-- Function to calculate the area of a quadrilateral using its vertices
def quadrilateralArea (vertices : List (ℝ × ℝ)) : ℝ := sorry

-- Theorem stating that the area of the given quadrilateral is 3.5
theorem quadrilateral_area : quadrilateralArea quadrilateral = 3.5 := by sorry

end NUMINAMATH_CALUDE_quadrilateral_area_l845_84545


namespace NUMINAMATH_CALUDE_continuous_bounded_function_theorem_l845_84523

theorem continuous_bounded_function_theorem (f : ℝ → ℝ) 
  (hcont : Continuous f) 
  (hbound : ∃ M, ∀ x, |f x| ≤ M) 
  (heq : ∀ x y, (f x)^2 - (f y)^2 = f (x + y) * f (x - y)) :
  ∃ a b : ℝ, ∀ x, f x = b * Real.sin (π * x / (2 * a)) := by
sorry

end NUMINAMATH_CALUDE_continuous_bounded_function_theorem_l845_84523


namespace NUMINAMATH_CALUDE_largest_number_value_l845_84516

theorem largest_number_value (a b c : ℝ) (h1 : a < b) (h2 : b < c)
  (h3 : a + b + c = 100) (h4 : c = b + 10) (h5 : b = a + 5) : c = 125/3 := by
  sorry

end NUMINAMATH_CALUDE_largest_number_value_l845_84516


namespace NUMINAMATH_CALUDE_min_sum_bound_min_sum_achievable_l845_84530

theorem min_sum_bound (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  a / (3 * b) + b / (6 * c) + c / (9 * a) ≥ 3 / Real.rpow 162 (1/3) :=
by sorry

theorem min_sum_achievable :
  ∃ (a b c : ℝ), 0 < a ∧ 0 < b ∧ 0 < c ∧
    a / (3 * b) + b / (6 * c) + c / (9 * a) = 3 / Real.rpow 162 (1/3) :=
by sorry

end NUMINAMATH_CALUDE_min_sum_bound_min_sum_achievable_l845_84530


namespace NUMINAMATH_CALUDE_arctan_sum_not_standard_angle_l845_84575

theorem arctan_sum_not_standard_angle :
  let a : ℝ := 2/3
  let b : ℝ := (3 / (5/3)) - 1
  ¬(Real.arctan a + Real.arctan b = π/2 ∨
    Real.arctan a + Real.arctan b = π/3 ∨
    Real.arctan a + Real.arctan b = π/4 ∨
    Real.arctan a + Real.arctan b = π/5 ∨
    Real.arctan a + Real.arctan b = π/6) :=
by
  sorry

end NUMINAMATH_CALUDE_arctan_sum_not_standard_angle_l845_84575


namespace NUMINAMATH_CALUDE_fraction_sum_l845_84503

theorem fraction_sum (a b : ℕ+) (h1 : (a : ℚ) / b = 9 / 16) 
  (h2 : ∀ d : ℕ, d > 1 → d ∣ a → d ∣ b → False) : 
  (a : ℕ) + b = 25 := by
  sorry

end NUMINAMATH_CALUDE_fraction_sum_l845_84503


namespace NUMINAMATH_CALUDE_tax_calculation_l845_84537

/-- Calculates the tax paid given gross pay and net pay -/
def tax_paid (gross_pay : ℕ) (net_pay : ℕ) : ℕ :=
  gross_pay - net_pay

/-- Theorem stating that the tax paid is 135 dollars given the conditions -/
theorem tax_calculation (gross_pay net_pay : ℕ) 
  (h1 : gross_pay = 450)
  (h2 : net_pay = 315)
  (h3 : tax_paid gross_pay net_pay = gross_pay - net_pay) :
  tax_paid gross_pay net_pay = 135 := by
  sorry

end NUMINAMATH_CALUDE_tax_calculation_l845_84537


namespace NUMINAMATH_CALUDE_pure_imaginary_product_l845_84548

def complex (a b : ℝ) : ℂ := Complex.mk a b

theorem pure_imaginary_product (m : ℝ) : 
  let z₁ : ℂ := complex 3 2
  let z₂ : ℂ := complex 1 m
  (z₁ * z₂).re = 0 → m = 3/2 := by
  sorry

end NUMINAMATH_CALUDE_pure_imaginary_product_l845_84548


namespace NUMINAMATH_CALUDE_nine_numbers_system_solution_l845_84525

theorem nine_numbers_system_solution (n : ℕ) (S : Finset ℕ) 
  (h₁ : n ≥ 3)
  (h₂ : S ⊆ Finset.range (n^3 + 1))
  (h₃ : S.card = 3 * n^2) :
  ∃ (a₁ a₂ a₃ a₄ a₅ a₆ a₇ a₈ a₉ : ℕ) (x y z : ℤ),
    a₁ ∈ S ∧ a₂ ∈ S ∧ a₃ ∈ S ∧ a₄ ∈ S ∧ a₅ ∈ S ∧ a₆ ∈ S ∧ a₇ ∈ S ∧ a₈ ∈ S ∧ a₉ ∈ S ∧
    a₁ ≠ a₂ ∧ a₁ ≠ a₃ ∧ a₁ ≠ a₄ ∧ a₁ ≠ a₅ ∧ a₁ ≠ a₆ ∧ a₁ ≠ a₇ ∧ a₁ ≠ a₈ ∧ a₁ ≠ a₉ ∧
    a₂ ≠ a₃ ∧ a₂ ≠ a₄ ∧ a₂ ≠ a₅ ∧ a₂ ≠ a₆ ∧ a₂ ≠ a₇ ∧ a₂ ≠ a₈ ∧ a₂ ≠ a₉ ∧
    a₃ ≠ a₄ ∧ a₃ ≠ a₅ ∧ a₃ ≠ a₆ ∧ a₃ ≠ a₇ ∧ a₃ ≠ a₈ ∧ a₃ ≠ a₉ ∧
    a₄ ≠ a₅ ∧ a₄ ≠ a₆ ∧ a₄ ≠ a₇ ∧ a₄ ≠ a₈ ∧ a₄ ≠ a₉ ∧
    a₅ ≠ a₆ ∧ a₅ ≠ a₇ ∧ a₅ ≠ a₈ ∧ a₅ ≠ a₉ ∧
    a₆ ≠ a₇ ∧ a₆ ≠ a₈ ∧ a₆ ≠ a₉ ∧
    a₇ ≠ a₈ ∧ a₇ ≠ a₉ ∧
    a₈ ≠ a₉ ∧
    x ≠ 0 ∧ y ≠ 0 ∧ z ≠ 0 ∧
    (a₁ : ℤ) * x + (a₂ : ℤ) * y + (a₃ : ℤ) * z = 0 ∧
    (a₄ : ℤ) * x + (a₅ : ℤ) * y + (a₆ : ℤ) * z = 0 ∧
    (a₇ : ℤ) * x + (a₈ : ℤ) * y + (a₉ : ℤ) * z = 0 := by
  sorry

end NUMINAMATH_CALUDE_nine_numbers_system_solution_l845_84525


namespace NUMINAMATH_CALUDE_arithmetic_mean_of_scores_l845_84507

def scores : List ℝ := [93, 87, 90, 96, 88, 94]

theorem arithmetic_mean_of_scores :
  (scores.sum / scores.length : ℝ) = 91.333 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_mean_of_scores_l845_84507


namespace NUMINAMATH_CALUDE_f_positive_implies_a_greater_than_half_open_l845_84539

-- Define the function f(x)
def f (a : ℝ) (x : ℝ) : ℝ := a * x^2 - 2 * x + 2

-- State the theorem
theorem f_positive_implies_a_greater_than_half_open :
  ∀ a : ℝ, (∀ x : ℝ, x ∈ Set.Icc 1 4 → f a x > 0) → a > 1/2 := by
  sorry

end NUMINAMATH_CALUDE_f_positive_implies_a_greater_than_half_open_l845_84539


namespace NUMINAMATH_CALUDE_multiple_births_l845_84576

theorem multiple_births (total_babies : ℕ) (twins triplets quintuplets : ℕ) : 
  total_babies = 1200 →
  triplets = 2 * quintuplets →
  twins = 2 * triplets →
  2 * twins + 3 * triplets + 5 * quintuplets = total_babies →
  5 * quintuplets = 316 :=
by
  sorry

end NUMINAMATH_CALUDE_multiple_births_l845_84576


namespace NUMINAMATH_CALUDE_score_difference_l845_84518

theorem score_difference (chuck_team_score red_team_score : ℕ) 
  (h1 : chuck_team_score = 95) 
  (h2 : red_team_score = 76) : 
  chuck_team_score - red_team_score = 19 := by
sorry

end NUMINAMATH_CALUDE_score_difference_l845_84518


namespace NUMINAMATH_CALUDE_probability_all_white_balls_l845_84522

def total_balls : ℕ := 15
def white_balls : ℕ := 8
def black_balls : ℕ := 7
def drawn_balls : ℕ := 7

theorem probability_all_white_balls :
  (Nat.choose white_balls drawn_balls : ℚ) / (Nat.choose total_balls drawn_balls : ℚ) = 8 / 6435 :=
sorry

end NUMINAMATH_CALUDE_probability_all_white_balls_l845_84522


namespace NUMINAMATH_CALUDE_chocolate_box_problem_l845_84560

theorem chocolate_box_problem (B : ℕ) : 
  (B : ℚ) - ((1/4 : ℚ) * B - 5) - ((1/4 : ℚ) * B - 10) = 110 → B = 190 := by
  sorry

end NUMINAMATH_CALUDE_chocolate_box_problem_l845_84560


namespace NUMINAMATH_CALUDE_planting_schemes_count_l845_84531

def number_of_seeds : ℕ := 6
def number_of_plots : ℕ := 4
def number_of_first_plot_options : ℕ := 2

def planting_schemes : ℕ :=
  number_of_first_plot_options * (number_of_seeds - 1).factorial / (number_of_seeds - number_of_plots).factorial

theorem planting_schemes_count : planting_schemes = 120 := by
  sorry

end NUMINAMATH_CALUDE_planting_schemes_count_l845_84531


namespace NUMINAMATH_CALUDE_empty_carton_weight_l845_84582

/-- Given the weights of a half-full and full milk carton, calculate the weight of an empty carton -/
theorem empty_carton_weight (half_full_weight full_weight : ℝ) :
  half_full_weight = 5 →
  full_weight = 8 →
  full_weight - 2 * (full_weight - half_full_weight) = 2 := by
  sorry

end NUMINAMATH_CALUDE_empty_carton_weight_l845_84582


namespace NUMINAMATH_CALUDE_cappuccino_cost_l845_84553

theorem cappuccino_cost (cappuccino_cost : ℝ) : 
  (3 : ℝ) * cappuccino_cost + 2 * 3 + 2 * 1.5 + 2 * 1 = 20 - 3 → 
  cappuccino_cost = 2 := by
  sorry

end NUMINAMATH_CALUDE_cappuccino_cost_l845_84553


namespace NUMINAMATH_CALUDE_reciprocal_location_l845_84520

-- Define the complex number G
def G (a b : ℝ) : ℂ := -a + b * Complex.I

-- Theorem statement
theorem reciprocal_location (a b : ℝ) 
  (h1 : a > 0) 
  (h2 : b > 0) 
  (h3 : Complex.abs (G a b) ^ 2 = 5) :
  let z := (G a b)⁻¹
  Complex.abs z < 1 ∧ Complex.re z < 0 ∧ Complex.im z < 0 := by
  sorry


end NUMINAMATH_CALUDE_reciprocal_location_l845_84520


namespace NUMINAMATH_CALUDE_recipe_total_cups_l845_84535

/-- Represents the ratio of ingredients in the recipe -/
structure RecipeRatio where
  butter : ℕ
  flour : ℕ
  sugar : ℕ

/-- Calculates the total cups of ingredients given a recipe ratio and the amount of flour used -/
def totalCups (ratio : RecipeRatio) (flourUsed : ℕ) : ℕ :=
  let partSize := flourUsed / ratio.flour
  partSize * (ratio.butter + ratio.flour + ratio.sugar)

/-- Theorem stating that for the given recipe ratio and flour amount, the total cups is 20 -/
theorem recipe_total_cups :
  let ratio : RecipeRatio := ⟨2, 3, 5⟩
  let flourUsed : ℕ := 6
  totalCups ratio flourUsed = 20 := by
  sorry


end NUMINAMATH_CALUDE_recipe_total_cups_l845_84535


namespace NUMINAMATH_CALUDE_det_3A_eq_96_l845_84566

def A : Matrix (Fin 2) (Fin 2) ℝ := !![7, -2; -5, 6]

theorem det_3A_eq_96 : Matrix.det (3 • A) = 96 := by
  sorry

end NUMINAMATH_CALUDE_det_3A_eq_96_l845_84566


namespace NUMINAMATH_CALUDE_min_sum_a_b_is_six_l845_84543

/-- Given that the roots of x^2 + ax + 2b = 0 and x^2 + 2bx + a = 0 are both real,
    and a, b > 0, the minimum value of a + b is 6. -/
theorem min_sum_a_b_is_six (a b : ℝ) 
    (h1 : a > 0)
    (h2 : b > 0)
    (h3 : a^2 - 8*b ≥ 0)  -- Condition for real roots of x^2 + ax + 2b = 0
    (h4 : 4*b^2 - 4*a ≥ 0)  -- Condition for real roots of x^2 + 2bx + a = 0
    : ∀ a' b' : ℝ, a' > 0 → b' > 0 → a'^2 - 8*b' ≥ 0 → 4*b'^2 - 4*a' ≥ 0 → a + b ≤ a' + b' :=
by sorry

end NUMINAMATH_CALUDE_min_sum_a_b_is_six_l845_84543


namespace NUMINAMATH_CALUDE_intersection_and_perpendicular_line_equal_intercepts_lines_l845_84561

-- Define the lines
def line1 (x y : ℝ) : Prop := 2 * x - y - 4 = 0
def line2 (x y : ℝ) : Prop := x - 2 * y + 1 = 0
def line3 (x y : ℝ) : Prop := 3 * x + 4 * y - 15 = 0

-- Define the intersection point P
def P : ℝ × ℝ := (3, 2)

-- Define the perpendicular line l1
def l1 (x y : ℝ) : Prop := 4 * x - 3 * y - 6 = 0

-- Define the lines l2 with equal intercepts
def l2_1 (x y : ℝ) : Prop := 2 * x - 3 * y = 0
def l2_2 (x y : ℝ) : Prop := x + y - 5 = 0

theorem intersection_and_perpendicular_line :
  (∀ x y, line1 x y ∧ line2 x y → (x, y) = P) ∧
  (∀ x y, l1 x y → (4 : ℝ) * 3 + 3 * 4 = 0) ∧
  l1 P.1 P.2 :=
sorry

theorem equal_intercepts_lines :
  (∀ x y, line1 x y ∧ line2 x y → (x, y) = P) ∧
  (l2_1 P.1 P.2 ∨ l2_2 P.1 P.2) ∧
  (∃ a ≠ 0, ∀ x y, l2_1 x y → x / a + y / a = 1) ∧
  (∃ a ≠ 0, ∀ x y, l2_2 x y → x / a + y / a = 1) :=
sorry

end NUMINAMATH_CALUDE_intersection_and_perpendicular_line_equal_intercepts_lines_l845_84561


namespace NUMINAMATH_CALUDE_students_answering_yes_for_R_l845_84573

theorem students_answering_yes_for_R (total : ℕ) (only_M : ℕ) (neither : ℕ) (h1 : total = 800) (h2 : only_M = 150) (h3 : neither = 250) : 
  ∃ R : ℕ, R = 400 ∧ R = total - neither - only_M :=
by sorry

end NUMINAMATH_CALUDE_students_answering_yes_for_R_l845_84573


namespace NUMINAMATH_CALUDE_cos_alpha_cos_2alpha_distinct_digits_l845_84504

/-- Represents a repeating decimal of the form 0.aḃ -/
def repeating_decimal (a b : ℕ) : ℚ :=
  (10 * a + b) / 90

theorem cos_alpha_cos_2alpha_distinct_digits :
  ∃! (a b c d : ℕ),
    a < 10 ∧ b < 10 ∧ c < 10 ∧ d < 10 ∧
    a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d ∧
    repeating_decimal a b = Real.cos α ∧
    repeating_decimal c d = -Real.cos (2 * α) ∧
    a = 1 ∧ b = 6 ∧ c = 9 ∧ d = 4 :=
by
  sorry

end NUMINAMATH_CALUDE_cos_alpha_cos_2alpha_distinct_digits_l845_84504


namespace NUMINAMATH_CALUDE_total_students_l845_84509

theorem total_students (rank_from_right rank_from_left : ℕ) 
  (h1 : rank_from_right = 21) 
  (h2 : rank_from_left = 11) : 
  rank_from_right + rank_from_left - 1 = 31 := by
  sorry

end NUMINAMATH_CALUDE_total_students_l845_84509


namespace NUMINAMATH_CALUDE_distribute_four_books_to_three_people_l845_84593

/-- Represents the number of ways to distribute books to people. -/
def distribute_books (num_books : ℕ) (num_people : ℕ) : ℕ :=
  sorry

/-- Theorem stating that distributing 4 different books to 3 people,
    with each person getting at least one book, can be done in 36 ways. -/
theorem distribute_four_books_to_three_people :
  distribute_books 4 3 = 36 :=
sorry

end NUMINAMATH_CALUDE_distribute_four_books_to_three_people_l845_84593


namespace NUMINAMATH_CALUDE_min_value_theorem_l845_84538

theorem min_value_theorem (a b : ℝ) (h1 : a > b) (h2 : b > 0) :
  a^2 + 1 / (b * (a - b)) ≥ 4 := by
  sorry

end NUMINAMATH_CALUDE_min_value_theorem_l845_84538


namespace NUMINAMATH_CALUDE_complex_fraction_calculations_l845_84527

theorem complex_fraction_calculations :
  (1 / 60) / ((1 / 3) - (1 / 4) + (1 / 12)) = 1 / 10 ∧
  -(1 / 42) / ((3 / 7) - (5 / 14) + (2 / 3) - (1 / 6)) = -(1 / 24) := by
  sorry

end NUMINAMATH_CALUDE_complex_fraction_calculations_l845_84527


namespace NUMINAMATH_CALUDE_cos_sin_identity_l845_84599

theorem cos_sin_identity (a : Real) (h : Real.cos (π/6 - a) = Real.sqrt 3 / 3) :
  Real.cos (5*π/6 + a) - Real.sin (a - π/6)^2 = -(Real.sqrt 3 + 2) / 3 := by
  sorry

end NUMINAMATH_CALUDE_cos_sin_identity_l845_84599


namespace NUMINAMATH_CALUDE_age_ratio_in_four_years_l845_84526

/-- Represents the ages of Paul and Kim -/
structure Ages where
  paul : ℕ
  kim : ℕ

/-- The conditions of the problem -/
def age_conditions (a : Ages) : Prop :=
  (a.paul - 8 = 2 * (a.kim - 8)) ∧ 
  (a.paul - 14 = 3 * (a.kim - 14))

/-- The theorem to prove -/
theorem age_ratio_in_four_years (a : Ages) :
  age_conditions a →
  ∃ (x : ℕ), x = 4 ∧ 
    (a.paul + x) * 2 = (a.kim + x) * 3 :=
by sorry

end NUMINAMATH_CALUDE_age_ratio_in_four_years_l845_84526


namespace NUMINAMATH_CALUDE_mixture_ratio_correct_l845_84587

def initial_alcohol : ℚ := 4
def initial_water : ℚ := 4
def added_water : ℚ := 8/3

def final_alcohol : ℚ := initial_alcohol
def final_water : ℚ := initial_water + added_water
def final_total : ℚ := final_alcohol + final_water

def desired_alcohol_ratio : ℚ := 3/8
def desired_water_ratio : ℚ := 5/8

theorem mixture_ratio_correct :
  (final_alcohol / final_total = desired_alcohol_ratio) ∧
  (final_water / final_total = desired_water_ratio) :=
sorry

end NUMINAMATH_CALUDE_mixture_ratio_correct_l845_84587


namespace NUMINAMATH_CALUDE_orchard_difference_l845_84589

/-- Represents the number of trees of each type in an orchard -/
structure Orchard where
  orange : ℕ
  lemon : ℕ
  apple : ℕ
  apricot : ℕ

/-- Calculates the total number of trees in an orchard -/
def totalTrees (o : Orchard) : ℕ :=
  o.orange + o.lemon + o.apple + o.apricot

theorem orchard_difference : 
  let ahmed : Orchard := { orange := 8, lemon := 6, apple := 4, apricot := 0 }
  let hassan : Orchard := { orange := 2, lemon := 5, apple := 1, apricot := 3 }
  totalTrees ahmed - totalTrees hassan = 7 := by
  sorry

end NUMINAMATH_CALUDE_orchard_difference_l845_84589


namespace NUMINAMATH_CALUDE_truck_tunnel_height_l845_84532

theorem truck_tunnel_height (tunnel_radius : ℝ) (truck_width : ℝ) 
  (h_radius : tunnel_radius = 4.5)
  (h_width : truck_width = 2.7) :
  Real.sqrt (tunnel_radius^2 - (truck_width/2)^2) = 3.6 := by
sorry

end NUMINAMATH_CALUDE_truck_tunnel_height_l845_84532


namespace NUMINAMATH_CALUDE_circle_and_line_properties_l845_84558

-- Define the circle C
def circle_C (x y : ℝ) : Prop := x^2 + y^2 - 2*y - 1 = 0

-- Define the line l
def line_l (x y m : ℝ) : Prop := y = x + m

-- Define the center of a circle
def is_center (h k : ℝ) (C : ℝ → ℝ → Prop) : Prop :=
  ∃ r, ∀ x y, C x y ↔ (x - h)^2 + (y - k)^2 = r^2

-- Define tangency between a line and a circle
def is_tangent (l : ℝ → ℝ → ℝ → Prop) (C : ℝ → ℝ → Prop) (m : ℝ) : Prop :=
  ∃! p, (∃ x y, l x y m ∧ C x y ∧ p = (x, y))

-- Theorem statement
theorem circle_and_line_properties :
  (is_center 0 1 circle_C) ∧
  (∀ m, is_tangent line_l circle_C m ↔ (m = 3 ∨ m = -1)) :=
sorry

end NUMINAMATH_CALUDE_circle_and_line_properties_l845_84558


namespace NUMINAMATH_CALUDE_wire_length_proof_l845_84588

theorem wire_length_proof (piece1 piece2 : ℝ) : 
  piece1 = 14 → 
  piece2 = 16 → 
  piece2 = piece1 + 2 → 
  piece1 + piece2 = 30 :=
by
  sorry

end NUMINAMATH_CALUDE_wire_length_proof_l845_84588


namespace NUMINAMATH_CALUDE_lower_right_is_two_l845_84579

/-- Represents a 5x5 grid of integers -/
def Grid := Fin 5 → Fin 5 → Nat

/-- Check if all numbers in a list are distinct -/
def allDistinct (l : List Nat) : Prop := l.Nodup

/-- Check if a grid satisfies the row constraint -/
def validRows (g : Grid) : Prop :=
  ∀ i, allDistinct [g i 0, g i 1, g i 2, g i 3, g i 4]

/-- Check if a grid satisfies the column constraint -/
def validColumns (g : Grid) : Prop :=
  ∀ j, allDistinct [g 0 j, g 1 j, g 2 j, g 3 j, g 4 j]

/-- Check if all numbers in the grid are between 1 and 5 -/
def validNumbers (g : Grid) : Prop :=
  ∀ i j, 1 ≤ g i j ∧ g i j ≤ 5

/-- Check if the sum of the first row is 15 -/
def firstRowSum15 (g : Grid) : Prop :=
  g 0 0 + g 0 1 + g 0 2 + g 0 3 + g 0 4 = 15

/-- Check if the given numbers in the grid match the problem description -/
def matchesGivenNumbers (g : Grid) : Prop :=
  g 0 0 = 1 ∧ g 0 2 = 3 ∧ g 0 3 = 4 ∧
  g 1 0 = 5 ∧ g 1 2 = 1 ∧ g 1 4 = 3 ∧
  g 2 1 = 4 ∧ g 2 3 = 5 ∧
  g 3 0 = 4

theorem lower_right_is_two (g : Grid) 
  (hrows : validRows g)
  (hcols : validColumns g)
  (hnums : validNumbers g)
  (hsum : firstRowSum15 g)
  (hgiven : matchesGivenNumbers g) :
  g 4 4 = 2 := by
  sorry

end NUMINAMATH_CALUDE_lower_right_is_two_l845_84579


namespace NUMINAMATH_CALUDE_ryan_fraction_l845_84508

theorem ryan_fraction (total : ℚ) (leo_final : ℚ) (ryan_to_leo : ℚ) (leo_to_ryan : ℚ) :
  total = 48 →
  leo_final = 19 →
  ryan_to_leo = 10 →
  leo_to_ryan = 7 →
  ∃ (ryan_fraction : ℚ),
    ryan_fraction = 11 / 24 ∧
    ryan_fraction * total = total - (leo_final + leo_to_ryan - ryan_to_leo) :=
by sorry


end NUMINAMATH_CALUDE_ryan_fraction_l845_84508


namespace NUMINAMATH_CALUDE_ellipse_equation_l845_84550

/-- An ellipse with given properties -/
structure Ellipse where
  a : ℝ
  b : ℝ
  eccentricity : ℝ
  perimeter_triangle : ℝ
  h1 : a > b
  h2 : b > 0
  h3 : eccentricity = Real.sqrt 3 / 3
  h4 : perimeter_triangle = 4 * Real.sqrt 3

/-- The theorem stating that an ellipse with the given properties has the equation x²/3 + y²/2 = 1 -/
theorem ellipse_equation (C : Ellipse) : 
  C.a = Real.sqrt 3 ∧ C.b = Real.sqrt 2 :=
sorry

end NUMINAMATH_CALUDE_ellipse_equation_l845_84550


namespace NUMINAMATH_CALUDE_bowling_money_theorem_l845_84505

/-- The cost of renting bowling shoes for a day -/
def shoe_rental_cost : ℚ := 0.50

/-- The cost of bowling one game -/
def game_cost : ℚ := 1.75

/-- The maximum number of complete games the person can bowl -/
def max_games : ℕ := 7

/-- The total amount of money the person has -/
def total_money : ℚ := shoe_rental_cost + max_games * game_cost

theorem bowling_money_theorem :
  total_money = 12.75 := by sorry

end NUMINAMATH_CALUDE_bowling_money_theorem_l845_84505


namespace NUMINAMATH_CALUDE_axis_of_symmetry_l845_84511

/-- The quadratic function f(x) = 3(x-1)^2 + 2 -/
def f (x : ℝ) : ℝ := 3 * (x - 1)^2 + 2

/-- The axis of symmetry for f(x) is x = 1 -/
theorem axis_of_symmetry (x : ℝ) : f (1 + x) = f (1 - x) := by sorry

end NUMINAMATH_CALUDE_axis_of_symmetry_l845_84511


namespace NUMINAMATH_CALUDE_fib_150_mod_9_l845_84597

/-- Fibonacci sequence -/
def fib : ℕ → ℕ
  | 0 => 1
  | 1 => 1
  | (n + 2) => fib n + fib (n + 1)

/-- The period of the Fibonacci sequence modulo 9 -/
def fib_mod_9_period : ℕ := 24

theorem fib_150_mod_9 :
  fib 149 % 9 = 8 := by
  sorry

end NUMINAMATH_CALUDE_fib_150_mod_9_l845_84597


namespace NUMINAMATH_CALUDE_smith_family_laundry_l845_84596

/-- The number of bath towels Kylie uses in one month -/
def kylie_towels : ℕ := 3

/-- The number of bath towels Kylie's daughters use in one month -/
def daughters_towels : ℕ := 6

/-- The number of bath towels Kylie's husband uses in one month -/
def husband_towels : ℕ := 3

/-- The number of bath towels that fit in one load of laundry -/
def towels_per_load : ℕ := 4

/-- The total number of bath towels used by the Smith family in one month -/
def total_towels : ℕ := kylie_towels + daughters_towels + husband_towels

/-- The number of loads of laundry needed to clean all used towels -/
def loads_needed : ℕ := (total_towels + towels_per_load - 1) / towels_per_load

theorem smith_family_laundry : loads_needed = 3 := by
  sorry

end NUMINAMATH_CALUDE_smith_family_laundry_l845_84596


namespace NUMINAMATH_CALUDE_folded_rectangle_perimeter_l845_84541

/-- The perimeter of a rectangular sheet folded along its diagonal -/
theorem folded_rectangle_perimeter (length width : ℝ) (h1 : length = 20) (h2 : width = 12) :
  2 * (length + width) = 64 := by
  sorry

end NUMINAMATH_CALUDE_folded_rectangle_perimeter_l845_84541


namespace NUMINAMATH_CALUDE_triangle_coloring_theorem_l845_84519

-- Define the set of colors
inductive Color
| Blue
| Red
| Yellow

-- Define a point with a color
structure Point where
  color : Color

-- Define a triangle
structure Triangle where
  vertex1 : Point
  vertex2 : Point
  vertex3 : Point

-- Define the main theorem
theorem triangle_coloring_theorem 
  (K P S : Point)
  (A B C D E F : Point)
  (h1 : K.color = Color.Blue)
  (h2 : P.color = Color.Red)
  (h3 : S.color = Color.Yellow)
  (h4 : (A.color = K.color ∨ A.color = S.color) ∧
        (B.color = K.color ∨ B.color = S.color) ∧
        (C.color = K.color ∨ C.color = P.color) ∧
        (D.color = P.color ∨ D.color = S.color) ∧
        (E.color = P.color ∨ E.color = S.color) ∧
        (F.color = K.color ∨ F.color = P.color)) :
  ∃ (t : Triangle), t.vertex1.color ≠ t.vertex2.color ∧ 
                    t.vertex2.color ≠ t.vertex3.color ∧ 
                    t.vertex3.color ≠ t.vertex1.color :=
by sorry

end NUMINAMATH_CALUDE_triangle_coloring_theorem_l845_84519


namespace NUMINAMATH_CALUDE_f_less_than_4_iff_in_M_abs_sum_less_than_abs_product_plus_4_l845_84591

-- Define the function f
def f (x : ℝ) : ℝ := |x - 1| + |x + 1|

-- Define the set M
def M : Set ℝ := Set.Ioo (-2) 2

-- Statement 1
theorem f_less_than_4_iff_in_M : ∀ x : ℝ, f x < 4 ↔ x ∈ M := by sorry

-- Statement 2
theorem abs_sum_less_than_abs_product_plus_4 : 
  ∀ x y : ℝ, x ∈ M → y ∈ M → |x + y| < |x * y / 2 + 2| := by sorry

end NUMINAMATH_CALUDE_f_less_than_4_iff_in_M_abs_sum_less_than_abs_product_plus_4_l845_84591


namespace NUMINAMATH_CALUDE_equal_average_groups_product_l845_84595

theorem equal_average_groups_product (groups : Fin 3 → List ℕ) : 
  (∀ i : Fin 3, ∀ n ∈ groups i, 1 ≤ n ∧ n ≤ 99) →
  (groups 0).sum + (groups 1).sum + (groups 2).sum = List.sum (List.range 99) →
  (groups 0).length + (groups 1).length + (groups 2).length = 99 →
  (groups 0).sum / (groups 0).length = (groups 1).sum / (groups 1).length →
  (groups 1).sum / (groups 1).length = (groups 2).sum / (groups 2).length →
  ((groups 0).sum / (groups 0).length) * ((groups 1).sum / (groups 1).length) * ((groups 2).sum / (groups 2).length) = 125000 := by
sorry

end NUMINAMATH_CALUDE_equal_average_groups_product_l845_84595


namespace NUMINAMATH_CALUDE_min_value_of_expression_l845_84515

/-- Given a function f(x) = x² + 2√a x - b + 1, where a and b are positive real numbers,
    and f(x) has only one zero, the minimum value of 1/a + 2a/(b+1) is 5/2. -/
theorem min_value_of_expression (a b : ℝ) (ha : a > 0) (hb : b > 0) : 
  (∃! x, x^2 + 2 * Real.sqrt a * x - b + 1 = 0) → 
  (∀ a' b', a' > 0 → b' > 0 → 
    (∃! x, x^2 + 2 * Real.sqrt a' * x - b' + 1 = 0) → 
    1 / a + 2 * a / (b + 1) ≤ 1 / a' + 2 * a' / (b' + 1)) ∧
  (∃ a₀ b₀, a₀ > 0 ∧ b₀ > 0 ∧ 
    (∃! x, x^2 + 2 * Real.sqrt a₀ * x - b₀ + 1 = 0) ∧
    1 / a₀ + 2 * a₀ / (b₀ + 1) = 5 / 2) :=
by sorry

end NUMINAMATH_CALUDE_min_value_of_expression_l845_84515


namespace NUMINAMATH_CALUDE_line_equation_through_points_l845_84563

/-- Prove that the equation x + 2y - 2 = 0 represents the line passing through points A(0,1) and B(2,0). -/
theorem line_equation_through_points :
  ∀ (x y : ℝ), x + 2*y - 2 = 0 ↔ ∃ (t : ℝ), (x, y) = (1 - t, t) * 2 + (0, 1) :=
by sorry

end NUMINAMATH_CALUDE_line_equation_through_points_l845_84563


namespace NUMINAMATH_CALUDE_zhang_slower_than_li_l845_84557

theorem zhang_slower_than_li :
  let zhang_efficiency : ℚ := 5 / 8
  let li_efficiency : ℚ := 3 / 4
  zhang_efficiency < li_efficiency :=
by
  sorry

end NUMINAMATH_CALUDE_zhang_slower_than_li_l845_84557


namespace NUMINAMATH_CALUDE_trains_crossing_time_l845_84510

/-- Proves that two trains of equal length traveling in opposite directions will cross each other in 12 seconds -/
theorem trains_crossing_time (length : ℝ) (time1 time2 : ℝ) 
  (h1 : length = 120)
  (h2 : time1 = 10)
  (h3 : time2 = 15) : 
  (2 * length) / (length / time1 + length / time2) = 12 := by
  sorry


end NUMINAMATH_CALUDE_trains_crossing_time_l845_84510


namespace NUMINAMATH_CALUDE_weight_of_replaced_person_l845_84544

theorem weight_of_replaced_person (n : ℕ) (avg_increase : ℝ) (new_weight : ℝ) :
  n = 8 →
  avg_increase = 2.5 →
  new_weight = 75 →
  ∃ (old_weight : ℝ), old_weight = 55 ∧ n * avg_increase = new_weight - old_weight :=
by sorry

end NUMINAMATH_CALUDE_weight_of_replaced_person_l845_84544


namespace NUMINAMATH_CALUDE_solution_uniqueness_l845_84555

theorem solution_uniqueness (x y : ℝ) : x^2 - 2*x + y^2 + 6*y + 10 = 0 → x = 1 ∧ y = -3 := by
  sorry

end NUMINAMATH_CALUDE_solution_uniqueness_l845_84555


namespace NUMINAMATH_CALUDE_darryl_honeydew_price_l845_84528

/-- The price of a honeydew given Darryl's sales data -/
def honeydew_price (cantaloupe_price : ℚ) (initial_cantaloupes : ℕ) (initial_honeydews : ℕ)
  (dropped_cantaloupes : ℕ) (rotten_honeydews : ℕ) (final_cantaloupes : ℕ) (final_honeydews : ℕ)
  (total_revenue : ℚ) : ℚ :=
  let sold_cantaloupes := initial_cantaloupes - final_cantaloupes - dropped_cantaloupes
  let sold_honeydews := initial_honeydews - final_honeydews - rotten_honeydews
  let cantaloupe_revenue := cantaloupe_price * sold_cantaloupes
  let honeydew_revenue := total_revenue - cantaloupe_revenue
  honeydew_revenue / sold_honeydews

theorem darryl_honeydew_price :
  honeydew_price 2 30 27 2 3 8 9 85 = 3 := by
  sorry

#eval honeydew_price 2 30 27 2 3 8 9 85

end NUMINAMATH_CALUDE_darryl_honeydew_price_l845_84528


namespace NUMINAMATH_CALUDE_smallest_sum_of_reciprocals_l845_84580

theorem smallest_sum_of_reciprocals (x y : ℕ+) : 
  x ≠ y → 
  (1 : ℚ) / x + (1 : ℚ) / y = (1 : ℚ) / 10 → 
  ∀ a b : ℕ+, a ≠ b → (1 : ℚ) / a + (1 : ℚ) / b = (1 : ℚ) / 10 → 
  (x : ℕ) + (y : ℕ) ≤ (a : ℕ) + (b : ℕ) → 
  (x : ℕ) + (y : ℕ) = 45 :=
by sorry

end NUMINAMATH_CALUDE_smallest_sum_of_reciprocals_l845_84580


namespace NUMINAMATH_CALUDE_exponential_above_line_l845_84570

theorem exponential_above_line (k : ℝ) : 
  (∀ x : ℝ, x > 0 → Real.exp x > k * x + 1) → k ≤ 1 := by
  sorry

end NUMINAMATH_CALUDE_exponential_above_line_l845_84570


namespace NUMINAMATH_CALUDE_fourth_term_is_plus_minus_three_l845_84594

/-- A geometric sequence with a_3 = 9 and a_5 = 1 -/
def geometric_sequence (a : ℕ → ℝ) : Prop :=
  (∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n) ∧ 
  a 3 = 9 ∧ 
  a 5 = 1

/-- The fourth term of the geometric sequence is ±3 -/
theorem fourth_term_is_plus_minus_three 
  (a : ℕ → ℝ) 
  (h : geometric_sequence a) : 
  a 4 = 3 ∨ a 4 = -3 :=
sorry

end NUMINAMATH_CALUDE_fourth_term_is_plus_minus_three_l845_84594


namespace NUMINAMATH_CALUDE_least_addition_for_divisibility_l845_84581

def divisors : List Nat := [5, 7, 11, 13, 17, 19]

theorem least_addition_for_divisibility (x : Nat) : 
  (∀ d ∈ divisors, (5432 + x) % d = 0) ∧
  (∀ y < x, ∃ d ∈ divisors, (5432 + y) % d ≠ 0) →
  x = 1611183 := by sorry

end NUMINAMATH_CALUDE_least_addition_for_divisibility_l845_84581


namespace NUMINAMATH_CALUDE_scientific_notation_of_595_5_billion_yuan_l845_84546

def billion : ℝ := 1000000000

theorem scientific_notation_of_595_5_billion_yuan :
  ∃ (a : ℝ) (n : ℤ), 
    595.5 * billion = a * (10 : ℝ) ^ n ∧ 
    1 ≤ |a| ∧ 
    |a| < 10 ∧
    a = 5.955 ∧
    n = 11 := by
  sorry

end NUMINAMATH_CALUDE_scientific_notation_of_595_5_billion_yuan_l845_84546


namespace NUMINAMATH_CALUDE_corresponding_angles_not_always_equal_l845_84554

/-- A structure representing angles in a geometric context -/
structure Angle where
  measure : ℝ

/-- A predicate to determine if two angles are corresponding -/
def are_corresponding (a b : Angle) : Prop := sorry

/-- The theorem stating that the general claim "Corresponding angles are equal" is false -/
theorem corresponding_angles_not_always_equal :
  ¬ (∀ (a b : Angle), are_corresponding a b → a = b) := by sorry

end NUMINAMATH_CALUDE_corresponding_angles_not_always_equal_l845_84554


namespace NUMINAMATH_CALUDE_manuscript_typing_cost_is_1400_l845_84549

/-- Calculates the total cost of typing a manuscript with given parameters. -/
def manuscriptTypingCost (totalPages : ℕ) (firstTypeCost : ℕ) (revisionCost : ℕ) 
  (pagesRevisedOnce : ℕ) (pagesRevisedTwice : ℕ) : ℕ :=
  totalPages * firstTypeCost + 
  pagesRevisedOnce * revisionCost + 
  pagesRevisedTwice * revisionCost * 2

/-- Proves that the total cost of typing the manuscript is $1400. -/
theorem manuscript_typing_cost_is_1400 : 
  manuscriptTypingCost 100 10 5 20 30 = 1400 := by
  sorry

#eval manuscriptTypingCost 100 10 5 20 30

end NUMINAMATH_CALUDE_manuscript_typing_cost_is_1400_l845_84549


namespace NUMINAMATH_CALUDE_unknown_number_proof_l845_84559

theorem unknown_number_proof (x : ℝ) : 
  (10 + 30 + 50) / 3 = (20 + x + 6) / 3 + 8 → x = 40 := by
  sorry

end NUMINAMATH_CALUDE_unknown_number_proof_l845_84559


namespace NUMINAMATH_CALUDE_intersection_sum_l845_84506

-- Define the lines l₁ and l₂
def l₁ (a : ℝ) (x y : ℝ) : Prop := a * x + y + 1 = 0
def l₂ (b : ℝ) (x y : ℝ) : Prop := 2 * x - b * y - 1 = 0

-- Theorem statement
theorem intersection_sum (a b : ℝ) : 
  (l₁ a 1 1 ∧ l₂ b 1 1) → a + b = -1 := by
  sorry

end NUMINAMATH_CALUDE_intersection_sum_l845_84506


namespace NUMINAMATH_CALUDE_gas_price_calculation_l845_84586

/-- Proves that the actual cost of gas per gallon is $1.80 given the problem conditions -/
theorem gas_price_calculation (expected_price : ℝ) : 
  (12 * expected_price = 10 * (expected_price + 0.3)) → 
  (expected_price + 0.3 = 1.8) := by
  sorry

#check gas_price_calculation

end NUMINAMATH_CALUDE_gas_price_calculation_l845_84586


namespace NUMINAMATH_CALUDE_chord_length_l845_84533

/-- The length of the chord formed by the intersection of a line and a circle -/
theorem chord_length (t : ℝ) : 
  let line := {(x, y) : ℝ × ℝ | x = 1 + 2*t ∧ y = 2 + t}
  let circle := {(x, y) : ℝ × ℝ | x^2 + y^2 = 9}
  let chord := line ∩ circle
  ∃ (p q : ℝ × ℝ), p ∈ chord ∧ q ∈ chord ∧ p ≠ q ∧
  Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2) = 12 * Real.sqrt 5 / 5 :=
sorry

end NUMINAMATH_CALUDE_chord_length_l845_84533


namespace NUMINAMATH_CALUDE_min_value_inequality_l845_84551

theorem min_value_inequality (x y z : ℝ) (h : x + 2*y + 3*z = 1) : 
  x^2 + 2*y^2 + 3*z^2 ≥ 1/3 := by
  sorry

#check min_value_inequality

end NUMINAMATH_CALUDE_min_value_inequality_l845_84551
