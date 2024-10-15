import Mathlib

namespace NUMINAMATH_CALUDE_binomial_18_choose_6_l554_55454

theorem binomial_18_choose_6 : Nat.choose 18 6 = 4765 := by
  sorry

end NUMINAMATH_CALUDE_binomial_18_choose_6_l554_55454


namespace NUMINAMATH_CALUDE_third_day_sale_l554_55466

/-- Proves that given an average sale of 625 for 5 days, and sales of 435, 927, 230, and 562
    for 4 of those days, the sale on the remaining day must be 971. -/
theorem third_day_sale (average : ℕ) (day1 day2 day4 day5 : ℕ) :
  average = 625 →
  day1 = 435 →
  day2 = 927 →
  day4 = 230 →
  day5 = 562 →
  ∃ day3 : ℕ, day3 = 971 ∧ (day1 + day2 + day3 + day4 + day5) / 5 = average :=
by sorry

end NUMINAMATH_CALUDE_third_day_sale_l554_55466


namespace NUMINAMATH_CALUDE_circle_intersection_and_reflection_l554_55456

-- Define the circles C₁ and C₂
def C₁ (x y : ℝ) : Prop := (x + 3)^2 + (y - 1)^2 = 4
def C₂ (x y : ℝ) : Prop := (x - 4)^2 + (y - 5)^2 = 1

-- Define point A
def A : ℝ × ℝ := (4, 0)

-- Define the reflecting line
def reflecting_line (x y : ℝ) : Prop := x - y - 3 = 0

theorem circle_intersection_and_reflection :
  -- Part I: Equation of line l
  (∃ (k : ℝ), (∀ x y : ℝ, y = k * (x - 4) → 
    (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ C₁ x₁ (k * (x₁ - 4)) ∧ C₁ x₂ (k * (x₂ - 4)) ∧ 
    (x₁ - x₂)^2 + (k * (x₁ - 4) - k * (x₂ - 4))^2 = 12)) ↔
    (k = 0 ∨ 7 * x + 24 * y - 28 = 0)) ∧
  -- Part II: Range of slope of reflected line
  (∀ k : ℝ, (∃ x y : ℝ, C₂ x y ∧ k * x - y - 4 * k - 6 = 0) ↔ 
    (k ≤ -2 * Real.sqrt 30 ∨ k ≥ 2 * Real.sqrt 30)) :=
sorry

end NUMINAMATH_CALUDE_circle_intersection_and_reflection_l554_55456


namespace NUMINAMATH_CALUDE_max_value_z_l554_55460

/-- The maximum value of z = 2x + y given the specified constraints -/
theorem max_value_z (x y : ℝ) (h1 : y ≤ 2 * x) (h2 : x - 2 * y - 4 ≤ 0) (h3 : y ≤ 4 - x) :
  (∀ x' y' : ℝ, y' ≤ 2 * x' → x' - 2 * y' - 4 ≤ 0 → y' ≤ 4 - x' → 2 * x' + y' ≤ 2 * x + y) ∧
  2 * x + y = 8 :=
by sorry

end NUMINAMATH_CALUDE_max_value_z_l554_55460


namespace NUMINAMATH_CALUDE_largest_x_sqrt_3x_eq_5x_l554_55400

theorem largest_x_sqrt_3x_eq_5x : 
  (∃ (x : ℚ), x > 0 ∧ Real.sqrt (3 * x) = 5 * x) ∧ 
  (∀ (y : ℚ), y > 0 ∧ Real.sqrt (3 * y) = 5 * y → y ≤ 3/25) :=
by sorry

end NUMINAMATH_CALUDE_largest_x_sqrt_3x_eq_5x_l554_55400


namespace NUMINAMATH_CALUDE_solve_equation_and_evaluate_l554_55404

theorem solve_equation_and_evaluate (x : ℝ) : 
  (5 * x - 8 = 15 * x + 4) → (3 * (x + 10) = 26.4) := by sorry

end NUMINAMATH_CALUDE_solve_equation_and_evaluate_l554_55404


namespace NUMINAMATH_CALUDE_equation_solution_l554_55484

theorem equation_solution :
  ∃ x : ℚ, (2 / 7) * (1 / 4) * x = 8 ∧ x = 112 :=
by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l554_55484


namespace NUMINAMATH_CALUDE_total_children_l554_55491

theorem total_children (happy : ℕ) (sad : ℕ) (neutral : ℕ) 
  (boys : ℕ) (girls : ℕ) (happy_boys : ℕ) (sad_girls : ℕ) (neutral_boys : ℕ)
  (h1 : happy = 30)
  (h2 : sad = 10)
  (h3 : neutral = 20)
  (h4 : boys = 16)
  (h5 : girls = 44)
  (h6 : happy_boys = 6)
  (h7 : sad_girls = 4)
  (h8 : neutral_boys = 4) :
  boys + girls = 60 := by
  sorry

end NUMINAMATH_CALUDE_total_children_l554_55491


namespace NUMINAMATH_CALUDE_bottle_recycling_result_l554_55475

/-- Calculates the number of new bottles created through recycling -/
def recycleBottles (initialBottles : ℕ) : ℕ :=
  let firstRound := initialBottles / 5
  let secondRound := firstRound / 5
  let thirdRound := secondRound / 5
  firstRound + secondRound + thirdRound

/-- Represents the recycling process with initial conditions -/
def bottleRecyclingProcess (initialBottles : ℕ) : Prop :=
  recycleBottles initialBottles = 179

/-- Theorem stating the result of the bottle recycling process -/
theorem bottle_recycling_result :
  bottleRecyclingProcess 729 := by sorry

end NUMINAMATH_CALUDE_bottle_recycling_result_l554_55475


namespace NUMINAMATH_CALUDE_quadratic_polynomial_with_complex_root_l554_55477

theorem quadratic_polynomial_with_complex_root :
  ∃ (a b c : ℝ), 
    (a = 3) ∧ 
    (∀ (z : ℂ), z = (2 : ℝ) + (2 : ℝ) * I → a * z^2 + b * z + c = 0) ∧
    (a * X^2 + b * X + c = 3 * X^2 - 12 * X + 24) :=
sorry

end NUMINAMATH_CALUDE_quadratic_polynomial_with_complex_root_l554_55477


namespace NUMINAMATH_CALUDE_one_point_45_deg_equals_1_deg_27_min_l554_55443

/-- Conversion of degrees to minutes -/
def deg_to_min (d : ℝ) : ℝ := d * 60

/-- Theorem stating that 1.45° is equal to 1°27′ -/
theorem one_point_45_deg_equals_1_deg_27_min :
  ∃ (deg min : ℕ), deg = 1 ∧ min = 27 ∧ 1.45 = deg + (min : ℝ) / 60 :=
by
  sorry

end NUMINAMATH_CALUDE_one_point_45_deg_equals_1_deg_27_min_l554_55443


namespace NUMINAMATH_CALUDE_factor_expression_l554_55472

theorem factor_expression (x y : ℝ) : 
  5 * x * (x + 1) + 7 * (x + 1) - 2 * y * (x + 1) = (x + 1) * (5 * x + 7 - 2 * y) := by
  sorry

end NUMINAMATH_CALUDE_factor_expression_l554_55472


namespace NUMINAMATH_CALUDE_sin_960_degrees_l554_55464

theorem sin_960_degrees : Real.sin (960 * π / 180) = -Real.sqrt 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_sin_960_degrees_l554_55464


namespace NUMINAMATH_CALUDE_fifty_billion_scientific_notation_l554_55463

/-- Scientific notation representation -/
structure ScientificNotation where
  coefficient : ℝ
  exponent : ℤ
  h_coeff_range : 1 ≤ |coefficient| ∧ |coefficient| < 10

/-- Convert a real number to scientific notation -/
def toScientificNotation (x : ℝ) : ScientificNotation :=
  sorry

theorem fifty_billion_scientific_notation :
  toScientificNotation 50000000000 = ScientificNotation.mk 5 10 sorry :=
sorry

end NUMINAMATH_CALUDE_fifty_billion_scientific_notation_l554_55463


namespace NUMINAMATH_CALUDE_value_of_a_fourth_plus_reciprocal_l554_55440

theorem value_of_a_fourth_plus_reciprocal (a : ℝ) (h : (a + 1/a)^2 = 5) :
  a^4 + 1/a^4 = 7 := by
sorry

end NUMINAMATH_CALUDE_value_of_a_fourth_plus_reciprocal_l554_55440


namespace NUMINAMATH_CALUDE_total_pure_acid_l554_55426

theorem total_pure_acid (solution1_volume : Real) (solution1_concentration : Real)
                        (solution2_volume : Real) (solution2_concentration : Real)
                        (solution3_volume : Real) (solution3_concentration : Real) :
  solution1_volume = 6 →
  solution1_concentration = 0.40 →
  solution2_volume = 4 →
  solution2_concentration = 0.35 →
  solution3_volume = 3 →
  solution3_concentration = 0.55 →
  solution1_volume * solution1_concentration +
  solution2_volume * solution2_concentration +
  solution3_volume * solution3_concentration = 5.45 := by
sorry

end NUMINAMATH_CALUDE_total_pure_acid_l554_55426


namespace NUMINAMATH_CALUDE_num_divisors_3960_l554_55441

/-- The number of positive divisors of a natural number n -/
def num_positive_divisors (n : ℕ) : ℕ := sorry

/-- Theorem: The number of positive divisors of 3960 is 48 -/
theorem num_divisors_3960 : num_positive_divisors 3960 = 48 := by sorry

end NUMINAMATH_CALUDE_num_divisors_3960_l554_55441


namespace NUMINAMATH_CALUDE_smartphone_price_l554_55455

theorem smartphone_price (x : ℝ) : (0.90 * x - 100) = (0.80 * x - 20) → x = 800 := by
  sorry

end NUMINAMATH_CALUDE_smartphone_price_l554_55455


namespace NUMINAMATH_CALUDE_jumping_contest_total_distance_l554_55497

/-- Represents the distance jumped by an animal and the obstacle they cleared -/
structure JumpDistance where
  jump : ℕ
  obstacle : ℕ

/-- Calculates the total distance jumped including the obstacle -/
def totalDistance (jd : JumpDistance) : ℕ := jd.jump + jd.obstacle

theorem jumping_contest_total_distance 
  (grasshopper : JumpDistance)
  (frog : JumpDistance)
  (kangaroo : JumpDistance)
  (h1 : grasshopper.jump = 25 ∧ grasshopper.obstacle = 5)
  (h2 : frog.jump = grasshopper.jump + 15 ∧ frog.obstacle = 10)
  (h3 : kangaroo.jump = 2 * frog.jump ∧ kangaroo.obstacle = 15) :
  totalDistance grasshopper + totalDistance frog + totalDistance kangaroo = 175 := by
  sorry

#check jumping_contest_total_distance

end NUMINAMATH_CALUDE_jumping_contest_total_distance_l554_55497


namespace NUMINAMATH_CALUDE_jean_speed_is_45_over_46_l554_55453

/-- Represents the hiking scenario with Chantal and Jean --/
structure HikingScenario where
  speed_first_third : ℝ
  speed_uphill : ℝ
  break_time : ℝ
  speed_downhill : ℝ
  meeting_point : ℝ

/-- Calculates Jean's average speed given a hiking scenario --/
def jeanAverageSpeed (scenario : HikingScenario) : ℝ :=
  sorry

/-- Theorem stating that Jean's average speed is 45/46 miles per hour --/
theorem jean_speed_is_45_over_46 (scenario : HikingScenario) :
  scenario.speed_first_third = 5 ∧
  scenario.speed_uphill = 3 ∧
  scenario.break_time = 1/6 ∧
  scenario.speed_downhill = 4 ∧
  scenario.meeting_point = 3/2 →
  jeanAverageSpeed scenario = 45/46 :=
sorry

end NUMINAMATH_CALUDE_jean_speed_is_45_over_46_l554_55453


namespace NUMINAMATH_CALUDE_equation_solution_l554_55448

theorem equation_solution (x : ℝ) : 
  (∀ y : ℝ, 10 * x * y - 15 * y + 3 * x - 9 / 2 = 0) ↔ x = 3 / 2 :=
by sorry

end NUMINAMATH_CALUDE_equation_solution_l554_55448


namespace NUMINAMATH_CALUDE_vivienne_phone_count_l554_55414

theorem vivienne_phone_count : ∀ (v : ℕ), 
  (400 * v + 400 * (v + 10) = 36000) → v = 40 := by
  sorry

end NUMINAMATH_CALUDE_vivienne_phone_count_l554_55414


namespace NUMINAMATH_CALUDE_right_rectangular_prism_volume_specific_prism_volume_l554_55402

/-- Given a right rectangular prism with face areas a₁, a₂, and a₃,
    prove that its volume is the square root of the product of these areas. -/
theorem right_rectangular_prism_volume 
  (a₁ a₂ a₃ : ℝ) 
  (h₁ : a₁ > 0) (h₂ : a₂ > 0) (h₃ : a₃ > 0) : 
  ∃ (l w h : ℝ), l * w = a₁ ∧ w * h = a₂ ∧ l * h = a₃ ∧ 
  l * w * h = Real.sqrt (a₁ * a₂ * a₃) := by
  sorry

/-- The volume of a right rectangular prism with face areas 56, 63, and 72 
    square units is 504 cubic units. -/
theorem specific_prism_volume : 
  ∃ (l w h : ℝ), l * w = 56 ∧ w * h = 63 ∧ l * h = 72 ∧ 
  l * w * h = 504 := by
  sorry

end NUMINAMATH_CALUDE_right_rectangular_prism_volume_specific_prism_volume_l554_55402


namespace NUMINAMATH_CALUDE_greatest_number_in_set_l554_55496

/-- Given a set of 45 consecutive multiples of 5 starting from 55, 
    the greatest number in the set is 275. -/
theorem greatest_number_in_set (s : Set ℕ) 
  (h1 : ∀ n ∈ s, ∃ k, n = 5 * k) 
  (h2 : ∀ n ∈ s, 55 ≤ n ∧ n ≤ 275)
  (h3 : ∀ n, 55 ≤ n ∧ n ≤ 275 ∧ 5 ∣ n → n ∈ s)
  (h4 : 55 ∈ s)
  (h5 : Fintype s)
  (h6 : Fintype.card s = 45) : 
  275 ∈ s ∧ ∀ n ∈ s, n ≤ 275 := by
  sorry

end NUMINAMATH_CALUDE_greatest_number_in_set_l554_55496


namespace NUMINAMATH_CALUDE_fermat_little_theorem_extension_l554_55469

theorem fermat_little_theorem_extension (p : ℕ) (a b : ℤ) 
  (hp : Nat.Prime p) (hab : a ≡ b [ZMOD p]) : 
  a^p ≡ b^p [ZMOD p^2] := by
  sorry

end NUMINAMATH_CALUDE_fermat_little_theorem_extension_l554_55469


namespace NUMINAMATH_CALUDE_negation_of_universal_proposition_l554_55446

theorem negation_of_universal_proposition :
  (¬ ∀ x : ℝ, x^3 - x^2 + 1 ≤ 1) ↔ (∃ x : ℝ, x^3 - x^2 + 1 > 1) := by sorry

end NUMINAMATH_CALUDE_negation_of_universal_proposition_l554_55446


namespace NUMINAMATH_CALUDE_sine_tangent_comparison_l554_55490

open Real

theorem sine_tangent_comparison (α : ℝ) (h : 0 < α ∧ α < π / 2) : 
  sin α < tan α ∧ (deriv sin) α < (deriv tan) α := by sorry

end NUMINAMATH_CALUDE_sine_tangent_comparison_l554_55490


namespace NUMINAMATH_CALUDE_part_one_part_two_l554_55415

-- Define the function f
def f (x : ℝ) : ℝ := |2 * x - 1|

-- Part 1
theorem part_one (m : ℝ) (h_m : m > 0) 
  (h_set : Set.Icc (-3/2) (1/2) = {x | f (x + 1) ≤ 2 * m}) : 
  m = 1 := by sorry

-- Part 2
theorem part_two : 
  (∃ n : ℝ, (∀ x y : ℝ, f (x + 1) ≤ 2018^y + n / 2018^y + |2*x - 1|) ∧ 
  (∀ n' : ℝ, (∀ x y : ℝ, f (x + 1) ≤ 2018^y + n' / 2018^y + |2*x - 1|) → n ≤ n')) ∧
  (∀ n : ℝ, (∀ x y : ℝ, f (x + 1) ≤ 2018^y + n / 2018^y + |2*x - 1|) → n ≥ 1) := by sorry

end NUMINAMATH_CALUDE_part_one_part_two_l554_55415


namespace NUMINAMATH_CALUDE_total_nuts_eq_3200_l554_55445

/-- The number of busy squirrels -/
def busy_squirrels : ℕ := 2

/-- The number of nuts stockpiled by each busy squirrel per day -/
def busy_squirrel_nuts_per_day : ℕ := 30

/-- The number of sleepy squirrels -/
def sleepy_squirrels : ℕ := 1

/-- The number of nuts stockpiled by each sleepy squirrel per day -/
def sleepy_squirrel_nuts_per_day : ℕ := 20

/-- The number of days the squirrels have been stockpiling -/
def stockpiling_days : ℕ := 40

/-- The total number of nuts stockpiled by all squirrels -/
def total_nuts : ℕ := 
  (busy_squirrels * busy_squirrel_nuts_per_day + 
   sleepy_squirrels * sleepy_squirrel_nuts_per_day) * 
  stockpiling_days

theorem total_nuts_eq_3200 : total_nuts = 3200 :=
by sorry

end NUMINAMATH_CALUDE_total_nuts_eq_3200_l554_55445


namespace NUMINAMATH_CALUDE_messages_in_week_after_removal_l554_55403

/-- Calculates the total number of messages sent in a week by remaining members of a group after some members were removed. -/
def total_messages_in_week (initial_members : ℕ) (removed_members : ℕ) (messages_per_day : ℕ) (days_in_week : ℕ) : ℕ :=
  (initial_members - removed_members) * messages_per_day * days_in_week

/-- Proves that the total number of messages sent in a week by remaining members is 45500, given the specified conditions. -/
theorem messages_in_week_after_removal :
  total_messages_in_week 150 20 50 7 = 45500 := by
  sorry

end NUMINAMATH_CALUDE_messages_in_week_after_removal_l554_55403


namespace NUMINAMATH_CALUDE_conference_room_seating_l554_55439

/-- Represents the seating arrangement in a conference room. -/
structure ConferenceRoom where
  totalPeople : ℕ
  rowCapacities : List ℕ
  allSeatsFilled : totalPeople = rowCapacities.sum

/-- Checks if a conference room arrangement is valid. -/
def isValidArrangement (room : ConferenceRoom) : Prop :=
  ∀ capacity ∈ room.rowCapacities, capacity = 9 ∨ capacity = 10

/-- The main theorem about the conference room seating arrangement. -/
theorem conference_room_seating
  (room : ConferenceRoom)
  (validArrangement : isValidArrangement room)
  (h : room.totalPeople = 54) :
  (room.rowCapacities.filter (· = 10)).length = 0 := by
  sorry


end NUMINAMATH_CALUDE_conference_room_seating_l554_55439


namespace NUMINAMATH_CALUDE_shift_direct_proportion_l554_55411

/-- A linear function represented by its slope and y-intercept -/
structure LinearFunction where
  slope : ℝ
  intercept : ℝ

/-- Shift a linear function horizontally -/
def shift_right (f : LinearFunction) (units : ℝ) : LinearFunction :=
  { slope := f.slope, intercept := f.slope * units + f.intercept }

/-- The original direct proportion function y = -2x -/
def original_function : LinearFunction :=
  { slope := -2, intercept := 0 }

theorem shift_direct_proportion :
  shift_right original_function 3 = { slope := -2, intercept := 6 } := by
  sorry

end NUMINAMATH_CALUDE_shift_direct_proportion_l554_55411


namespace NUMINAMATH_CALUDE_complex_number_properties_l554_55406

/-- Complex number z as a function of real number m -/
def z (m : ℝ) : ℂ := m^2 - 2*m + m*Complex.I

/-- The line x - y + 2 = 0 -/
def line (z : ℂ) : Prop := z.re - z.im + 2 = 0

/-- z is in the second quadrant -/
def second_quadrant (z : ℂ) : Prop := z.re < 0 ∧ z.im > 0

theorem complex_number_properties :
  ∀ m : ℝ,
  (second_quadrant (z m) ∧ line (z m)) ↔ (m = 1 ∨ m = 2) ∧
  (m = 1 → Complex.abs (z m) = Real.sqrt 2) ∧
  (m = 2 → Complex.abs (z m) = 2) := by sorry

end NUMINAMATH_CALUDE_complex_number_properties_l554_55406


namespace NUMINAMATH_CALUDE_mod_equivalence_problem_l554_55447

theorem mod_equivalence_problem : ∃! n : ℤ, 0 ≤ n ∧ n < 17 ∧ 27483 % 17 = n := by
  sorry

end NUMINAMATH_CALUDE_mod_equivalence_problem_l554_55447


namespace NUMINAMATH_CALUDE_no_integer_solutions_for_3x_minus_12y_eq_7_l554_55420

theorem no_integer_solutions_for_3x_minus_12y_eq_7 :
  ¬ ∃ (x y : ℤ), 3 * x - 12 * y = 7 := by
  sorry

end NUMINAMATH_CALUDE_no_integer_solutions_for_3x_minus_12y_eq_7_l554_55420


namespace NUMINAMATH_CALUDE_solution_equivalence_l554_55405

theorem solution_equivalence (x : ℝ) : 
  (3/10 : ℝ) + |x - 7/20| < 4/15 ↔ x ∈ Set.Ioo (19/60 : ℝ) (23/60 : ℝ) :=
by sorry

end NUMINAMATH_CALUDE_solution_equivalence_l554_55405


namespace NUMINAMATH_CALUDE_jeans_and_shirts_cost_l554_55437

/-- The cost of one pair of jeans -/
def jean_cost : ℝ := 11

/-- The cost of one shirt -/
def shirt_cost : ℝ := 18

/-- The cost of 2 pairs of jeans and 3 shirts -/
def cost_2j_3s : ℝ := 76

/-- The cost of 3 pairs of jeans and 2 shirts -/
def cost_3j_2s : ℝ := 3 * jean_cost + 2 * shirt_cost

theorem jeans_and_shirts_cost : cost_3j_2s = 69 := by
  sorry

end NUMINAMATH_CALUDE_jeans_and_shirts_cost_l554_55437


namespace NUMINAMATH_CALUDE_cubic_roots_sum_l554_55432

theorem cubic_roots_sum (x y z : ℝ) : 
  (x^3 - 2*x^2 - 9*x - 1 = 0) →
  (y^3 - 2*y^2 - 9*y - 1 = 0) →
  (z^3 - 2*z^2 - 9*z - 1 = 0) →
  (y*z/x + x*z/y + x*y/z = 77) :=
by sorry

end NUMINAMATH_CALUDE_cubic_roots_sum_l554_55432


namespace NUMINAMATH_CALUDE_nancy_balloons_l554_55434

theorem nancy_balloons (nancy_balloons : ℕ) (mary_balloons : ℕ) : 
  mary_balloons = 28 → mary_balloons = 4 * nancy_balloons → nancy_balloons = 7 := by
  sorry

end NUMINAMATH_CALUDE_nancy_balloons_l554_55434


namespace NUMINAMATH_CALUDE_area_of_curve_l554_55462

theorem area_of_curve (x y : ℝ) : 
  (∃ A : ℝ, A = Real.pi * 16 ∧ 
   A = Real.pi * (Real.sqrt ((x - 2)^2 + (y + 3)^2))^2 ∧
   x^2 + y^2 - 4*x + 6*y - 3 = 0) := by
sorry

end NUMINAMATH_CALUDE_area_of_curve_l554_55462


namespace NUMINAMATH_CALUDE_equation_solutions_l554_55488

theorem equation_solutions :
  (∃ x₁ x₂ : ℝ, x₁ = 2 + Real.sqrt 5 ∧ x₂ = 2 - Real.sqrt 5 ∧
    x₁^2 - 4*x₁ - 1 = 0 ∧ x₂^2 - 4*x₂ - 1 = 0) ∧
  (∃ x₁ x₂ : ℝ, x₁ = 1 ∧ x₂ = 5/3 ∧
    3*(x₁-1)^2 = 2*(x₁-1) ∧ 3*(x₂-1)^2 = 2*(x₂-1)) :=
by sorry

end NUMINAMATH_CALUDE_equation_solutions_l554_55488


namespace NUMINAMATH_CALUDE_sundae_price_l554_55478

/-- Given the following conditions:
  * The caterer ordered 125 ice-cream bars
  * The caterer ordered 125 sundaes
  * The total price was $225.00
  * The price of each ice-cream bar was $0.60
Prove that the price of each sundae was $1.20 -/
theorem sundae_price 
  (num_ice_cream : ℕ) 
  (num_sundae : ℕ) 
  (total_price : ℚ) 
  (ice_cream_price : ℚ) 
  (h1 : num_ice_cream = 125)
  (h2 : num_sundae = 125)
  (h3 : total_price = 225)
  (h4 : ice_cream_price = 6/10) : 
  (total_price - num_ice_cream * ice_cream_price) / num_sundae = 12/10 := by
  sorry


end NUMINAMATH_CALUDE_sundae_price_l554_55478


namespace NUMINAMATH_CALUDE_newspaper_delivery_totals_l554_55452

/-- Represents the days of the week --/
inductive Day
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday
  | Saturday
  | Sunday

/-- Represents the different routes --/
inductive Route
  | Route1
  | Route2
  | Route3
  | Route4
  | Route5

/-- Represents the different newspapers --/
inductive Newspaper
  | A
  | B
  | C

/-- Defines the delivery schedule for each newspaper --/
def delivery_schedule (n : Newspaper) (d : Day) (r : Route) : Nat :=
  match n, d, r with
  | Newspaper.A, Day.Sunday, Route.Route1 => 90
  | Newspaper.A, Day.Sunday, Route.Route2 => 30
  | Newspaper.A, _, Route.Route1 => 100
  | Newspaper.B, Day.Tuesday, Route.Route3 => 80
  | Newspaper.B, Day.Thursday, Route.Route3 => 80
  | Newspaper.B, Day.Saturday, Route.Route3 => 50
  | Newspaper.B, Day.Saturday, Route.Route4 => 20
  | Newspaper.B, Day.Sunday, Route.Route3 => 50
  | Newspaper.B, Day.Sunday, Route.Route4 => 20
  | Newspaper.C, Day.Monday, Route.Route5 => 70
  | Newspaper.C, Day.Wednesday, Route.Route5 => 70
  | Newspaper.C, Day.Friday, Route.Route5 => 70
  | Newspaper.C, Day.Sunday, Route.Route5 => 15
  | Newspaper.C, Day.Sunday, Route.Route4 => 25
  | _, _, _ => 0

/-- Calculates the total newspapers delivered for a given newspaper in a week --/
def total_newspapers (n : Newspaper) : Nat :=
  List.sum (List.map (fun d => List.sum (List.map (fun r => delivery_schedule n d r) [Route.Route1, Route.Route2, Route.Route3, Route.Route4, Route.Route5]))
    [Day.Monday, Day.Tuesday, Day.Wednesday, Day.Thursday, Day.Friday, Day.Saturday, Day.Sunday])

/-- Theorem stating the correct total number of newspapers delivered for each type in a week --/
theorem newspaper_delivery_totals :
  (total_newspapers Newspaper.A = 720) ∧
  (total_newspapers Newspaper.B = 300) ∧
  (total_newspapers Newspaper.C = 250) := by
  sorry

end NUMINAMATH_CALUDE_newspaper_delivery_totals_l554_55452


namespace NUMINAMATH_CALUDE_regular_polygon_perimeter_l554_55433

/-- A regular polygon with side length 7 units and exterior angle 90 degrees has a perimeter of 28 units. -/
theorem regular_polygon_perimeter (n : ℕ) (sides : ℕ) (side_length : ℝ) (exterior_angle : ℝ) :
  n > 2 ∧
  sides = n ∧
  side_length = 7 ∧
  exterior_angle = 90 ∧
  (360 : ℝ) / n = exterior_angle →
  sides * side_length = 28 := by
  sorry

end NUMINAMATH_CALUDE_regular_polygon_perimeter_l554_55433


namespace NUMINAMATH_CALUDE_diophantine_equation_unique_solution_l554_55428

theorem diophantine_equation_unique_solution :
  ∀ a b c : ℤ, a^2 = 2*b^2 + 3*c^2 → a = 0 ∧ b = 0 ∧ c = 0 := by
  sorry

end NUMINAMATH_CALUDE_diophantine_equation_unique_solution_l554_55428


namespace NUMINAMATH_CALUDE_student_selection_problem_l554_55429

/-- The number of ways to select k items from n items -/
def choose (n k : ℕ) : ℕ := sorry

theorem student_selection_problem :
  let total_students : ℕ := 8
  let selected_students : ℕ := 4
  let students_except_AB : ℕ := 6
  
  -- Number of ways to select with exactly one of A or B
  let with_one_AB : ℕ := 2 * (choose students_except_AB (selected_students - 1))
  
  -- Number of ways to select without A and B
  let without_AB : ℕ := choose students_except_AB selected_students
  
  -- Total number of valid selections
  let total_selections : ℕ := with_one_AB + without_AB
  
  total_selections = 55 := by sorry

end NUMINAMATH_CALUDE_student_selection_problem_l554_55429


namespace NUMINAMATH_CALUDE_sum_of_ages_l554_55483

/-- 
Given that Tom is 15 years old now and in 3 years he will be twice Tim's age,
prove that the sum of their current ages is 21 years.
-/
theorem sum_of_ages : 
  ∀ (tim_age : ℕ), 
  (15 + 3 = 2 * (tim_age + 3)) →
  (15 + tim_age = 21) := by
sorry

end NUMINAMATH_CALUDE_sum_of_ages_l554_55483


namespace NUMINAMATH_CALUDE_count_satisfying_integers_l554_55461

def is_geometric_mean_integer (n : ℕ+) : Prop :=
  ∃ k : ℕ+, n = 2015 * k^2

def is_harmonic_mean_integer (n : ℕ+) : Prop :=
  ∃ m : ℕ+, 2 * 2015 * n = m * (2015 + n)

def satisfies_conditions (n : ℕ+) : Prop :=
  is_geometric_mean_integer n ∧ is_harmonic_mean_integer n

theorem count_satisfying_integers :
  (∃! (s : Finset ℕ+), s.card = 5 ∧ ∀ n, n ∈ s ↔ satisfies_conditions n) ∧
  2015 = 5 * 13 * 31 := by
  sorry

end NUMINAMATH_CALUDE_count_satisfying_integers_l554_55461


namespace NUMINAMATH_CALUDE_polynomial_equality_l554_55481

theorem polynomial_equality (a b A : ℝ) (h : A / (2 * a * b) = 1 - 4 * a^2) : 
  A = 2 * a * b - 8 * a^3 * b :=
by sorry

end NUMINAMATH_CALUDE_polynomial_equality_l554_55481


namespace NUMINAMATH_CALUDE_subtracted_number_l554_55425

theorem subtracted_number (x y : ℤ) (h1 : x = 127) (h2 : 2 * x - y = 102) : y = 152 := by
  sorry

end NUMINAMATH_CALUDE_subtracted_number_l554_55425


namespace NUMINAMATH_CALUDE_winter_uniform_count_l554_55435

/-- The number of packages of winter uniforms delivered -/
def num_packages : ℕ := 10

/-- The number of dozens per package -/
def dozens_per_package : ℕ := 10

/-- The number of sets per dozen -/
def sets_per_dozen : ℕ := 12

/-- The total number of winter uniform sets -/
def total_sets : ℕ := num_packages * dozens_per_package * sets_per_dozen

theorem winter_uniform_count : total_sets = 1200 := by
  sorry

end NUMINAMATH_CALUDE_winter_uniform_count_l554_55435


namespace NUMINAMATH_CALUDE_nine_hash_seven_l554_55408

-- Define the # operation
noncomputable def hash (r s : ℝ) : ℝ :=
  sorry

-- State the conditions
axiom hash_zero (r : ℝ) : hash r 0 = r
axiom hash_comm (r s : ℝ) : hash r s = hash s r
axiom hash_succ (r s : ℝ) : hash (r + 1) s = hash r s + s + 1

-- State the theorem to be proved
theorem nine_hash_seven : hash 9 7 = 79 := by
  sorry

end NUMINAMATH_CALUDE_nine_hash_seven_l554_55408


namespace NUMINAMATH_CALUDE_fractional_equation_simplification_l554_55431

theorem fractional_equation_simplification (x : ℝ) : 
  (x / (3 - x) - 4 = 6 / (x - 3)) ↔ (x - 4 * (3 - x) = -6) :=
by sorry

end NUMINAMATH_CALUDE_fractional_equation_simplification_l554_55431


namespace NUMINAMATH_CALUDE_incorrect_inequality_transformation_l554_55482

theorem incorrect_inequality_transformation (a b : ℝ) (h : a < b) :
  ¬(3 - a < 3 - b) := by
  sorry

end NUMINAMATH_CALUDE_incorrect_inequality_transformation_l554_55482


namespace NUMINAMATH_CALUDE_jimmy_has_more_sheets_l554_55468

/-- Given the initial number of sheets and additional sheets received,
    calculate the difference between Jimmy's and Tommy's final sheet counts. -/
def sheet_difference (jimmy_initial : ℕ) (tommy_more_initial : ℕ) 
  (jimmy_additional1 : ℕ) (jimmy_additional2 : ℕ)
  (tommy_additional1 : ℕ) (tommy_additional2 : ℕ) : ℕ :=
  let tommy_initial := jimmy_initial + tommy_more_initial
  let jimmy_final := jimmy_initial + jimmy_additional1 + jimmy_additional2
  let tommy_final := tommy_initial + tommy_additional1 + tommy_additional2
  jimmy_final - tommy_final

/-- Theorem stating that Jimmy will have 58 more sheets than Tommy
    after receiving additional sheets. -/
theorem jimmy_has_more_sheets :
  sheet_difference 58 25 85 47 30 19 = 58 := by
  sorry

end NUMINAMATH_CALUDE_jimmy_has_more_sheets_l554_55468


namespace NUMINAMATH_CALUDE_water_bucket_problem_l554_55495

theorem water_bucket_problem (initial_amount : ℝ) (added_amount : ℝ) :
  initial_amount = 3 →
  added_amount = 6.8 →
  initial_amount + added_amount = 9.8 :=
by
  sorry

end NUMINAMATH_CALUDE_water_bucket_problem_l554_55495


namespace NUMINAMATH_CALUDE_cistern_fill_time_l554_55489

/-- If a cistern can be emptied by a tap in 10 hours, and when both this tap and another tap
    are opened simultaneously the cistern gets filled in 20/3 hours, then the time it takes
    for the other tap alone to fill the cistern is 4 hours. -/
theorem cistern_fill_time (empty_rate : ℝ) (combined_fill_time : ℝ) (fill_time : ℝ) :
  empty_rate = 10 →
  combined_fill_time = 20 / 3 →
  1 / fill_time - 1 / empty_rate = 1 / combined_fill_time →
  fill_time = 4 := by
  sorry

end NUMINAMATH_CALUDE_cistern_fill_time_l554_55489


namespace NUMINAMATH_CALUDE_cube_roots_of_specific_numbers_l554_55493

theorem cube_roots_of_specific_numbers :
  (∃ x : ℕ, x^3 = 59319) ∧ (∃ y : ℕ, y^3 = 195112) :=
by
  have h1 : (10 : ℕ)^3 = 1000 := by norm_num
  have h2 : (100 : ℕ)^3 = 1000000 := by norm_num
  sorry

end NUMINAMATH_CALUDE_cube_roots_of_specific_numbers_l554_55493


namespace NUMINAMATH_CALUDE_cosine_value_in_triangle_l554_55499

/-- Triangle ABC with sides a, b, c opposite to angles A, B, C respectively -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- Vector in 2D space -/
structure Vector2D where
  x : ℝ
  y : ℝ

/-- Check if two vectors are parallel -/
def parallel (v w : Vector2D) : Prop :=
  v.x * w.y = v.y * w.x

theorem cosine_value_in_triangle (t : Triangle) 
  (hm : Vector2D := ⟨Real.sqrt 3 * t.b - t.c, Real.cos t.C⟩)
  (hn : Vector2D := ⟨t.a, Real.cos t.A⟩)
  (h_parallel : parallel hm hn) :
  Real.cos t.A = Real.sqrt 3 / 3 :=
sorry

end NUMINAMATH_CALUDE_cosine_value_in_triangle_l554_55499


namespace NUMINAMATH_CALUDE_complex_roots_nature_l554_55409

theorem complex_roots_nature (k : ℝ) (hk : k > 0) :
  ∃ (z₁ z₂ : ℂ), 
    (10 * z₁^2 + 5 * Complex.I * z₁ - k = 0) ∧
    (10 * z₂^2 + 5 * Complex.I * z₂ - k = 0) ∧
    (z₁.re ≠ 0 ∧ z₁.im ≠ 0) ∧
    (z₂.re ≠ 0 ∧ z₂.im ≠ 0) ∧
    (z₁ ≠ z₂) :=
by sorry

end NUMINAMATH_CALUDE_complex_roots_nature_l554_55409


namespace NUMINAMATH_CALUDE_pirate_rick_sand_ratio_l554_55470

/-- Pirate Rick's treasure digging problem -/
theorem pirate_rick_sand_ratio :
  let initial_sand : ℝ := 8
  let initial_time : ℝ := 4
  let tsunami_sand : ℝ := 2
  let final_time : ℝ := 3
  let digging_rate : ℝ := initial_sand / initial_time
  let final_sand : ℝ := final_time * digging_rate
  let storm_sand : ℝ := initial_sand + tsunami_sand - final_sand
  storm_sand / initial_sand = 1 / 2 := by
sorry

end NUMINAMATH_CALUDE_pirate_rick_sand_ratio_l554_55470


namespace NUMINAMATH_CALUDE_monomino_position_l554_55422

/-- Represents a position on an 8x8 board -/
def Position := Fin 8 × Fin 8

/-- Represents a tromino (3x1 rectangle) -/
def Tromino := List Position

/-- Represents a monomino (1x1 square) -/
def Monomino := Position

/-- Represents a coloring of the board -/
def Coloring := Position → Fin 3

/-- The first coloring pattern -/
def coloring1 : Coloring := sorry

/-- The second coloring pattern -/
def coloring2 : Coloring := sorry

/-- Checks if a tromino is valid (covers exactly one square of each color) -/
def isValidTromino (t : Tromino) (c : Coloring) : Prop := sorry

/-- Checks if a set of trominos and a monomino form a valid covering of the board -/
def isValidCovering (trominos : List Tromino) (monomino : Monomino) : Prop := sorry

theorem monomino_position (trominos : List Tromino) (monomino : Monomino) :
  isValidCovering trominos monomino →
  coloring1 monomino = 1 ∧ coloring2 monomino = 1 := by sorry

end NUMINAMATH_CALUDE_monomino_position_l554_55422


namespace NUMINAMATH_CALUDE_hannah_movie_remaining_time_l554_55465

/-- Calculates the remaining movie time given the total duration and watched duration. -/
def remaining_movie_time (total_duration watched_duration : ℕ) : ℕ :=
  total_duration - watched_duration

/-- Proves that for a 3-hour movie watched for 2 hours and 24 minutes, 36 minutes remain. -/
theorem hannah_movie_remaining_time :
  let total_duration : ℕ := 3 * 60  -- 3 hours in minutes
  let watched_duration : ℕ := 2 * 60 + 24  -- 2 hours and 24 minutes
  remaining_movie_time total_duration watched_duration = 36 := by
  sorry

#eval remaining_movie_time (3 * 60) (2 * 60 + 24)

end NUMINAMATH_CALUDE_hannah_movie_remaining_time_l554_55465


namespace NUMINAMATH_CALUDE_x_value_proof_l554_55458

theorem x_value_proof (x y : ℤ) (h1 : x + y = 20) (h2 : x - y = 10) : x = 15 := by
  sorry

end NUMINAMATH_CALUDE_x_value_proof_l554_55458


namespace NUMINAMATH_CALUDE_completing_square_l554_55407

theorem completing_square (x : ℝ) : x^2 + 2*x - 1 = 0 ↔ (x + 1)^2 = 2 := by
  sorry

end NUMINAMATH_CALUDE_completing_square_l554_55407


namespace NUMINAMATH_CALUDE_dog_cord_length_l554_55416

/-- The maximum radius of the semi-circular path -/
def max_radius : ℝ := 5

/-- The approximate arc length of the semi-circular path -/
def arc_length : ℝ := 30

/-- The length of the nylon cord -/
def cord_length : ℝ := max_radius

theorem dog_cord_length :
  cord_length = max_radius := by sorry

end NUMINAMATH_CALUDE_dog_cord_length_l554_55416


namespace NUMINAMATH_CALUDE_sin_cos_sum_eighty_forty_l554_55487

theorem sin_cos_sum_eighty_forty : 
  Real.sin (80 * π / 180) * Real.cos (40 * π / 180) + 
  Real.cos (80 * π / 180) * Real.sin (40 * π / 180) = 
  Real.sqrt 3 / 2 := by sorry

end NUMINAMATH_CALUDE_sin_cos_sum_eighty_forty_l554_55487


namespace NUMINAMATH_CALUDE_four_children_prob_l554_55451

def prob_boy_or_girl : ℚ := 1/2

def prob_at_least_one_boy_and_girl (n : ℕ) : ℚ :=
  1 - (prob_boy_or_girl ^ n + prob_boy_or_girl ^ n)

theorem four_children_prob :
  prob_at_least_one_boy_and_girl 4 = 7/8 :=
by sorry

end NUMINAMATH_CALUDE_four_children_prob_l554_55451


namespace NUMINAMATH_CALUDE_min_value_inequality_l554_55423

theorem min_value_inequality (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  (∀ x, |a - x| + |x + b| + c ≥ 1) →
  a^2 + b^2 + c^2 ≥ 1/3 := by
  sorry

end NUMINAMATH_CALUDE_min_value_inequality_l554_55423


namespace NUMINAMATH_CALUDE_goldfish_count_l554_55418

/-- Given that 25% of goldfish are at the surface and 75% are below the surface,
    with 45 goldfish below the surface, prove that there are 15 goldfish at the surface. -/
theorem goldfish_count (surface_percent : ℝ) (below_percent : ℝ) (below_count : ℕ) :
  surface_percent = 25 →
  below_percent = 75 →
  below_count = 45 →
  ↑below_count / below_percent * surface_percent = 15 :=
by sorry

end NUMINAMATH_CALUDE_goldfish_count_l554_55418


namespace NUMINAMATH_CALUDE_dihedral_angle_range_l554_55459

/-- The dihedral angle between two adjacent faces in a regular n-sided prism -/
def dihedral_angle (n : ℕ) (θ : ℝ) : Prop :=
  n ≥ 3 ∧ ((n - 2 : ℝ) / n * Real.pi < θ ∧ θ < Real.pi)

/-- Theorem: The dihedral angle in a regular n-sided prism is within the specified range -/
theorem dihedral_angle_range (n : ℕ) :
  ∃ θ : ℝ, dihedral_angle n θ :=
sorry

end NUMINAMATH_CALUDE_dihedral_angle_range_l554_55459


namespace NUMINAMATH_CALUDE_paint_cans_used_l554_55498

/-- Given:
  - Paul originally had enough paint for 50 rooms.
  - He lost 5 cans of paint.
  - After losing the paint, he had enough for 40 rooms.
Prove that the number of cans of paint used for 40 rooms is 20. -/
theorem paint_cans_used (original_rooms : ℕ) (lost_cans : ℕ) (remaining_rooms : ℕ)
  (h1 : original_rooms = 50)
  (h2 : lost_cans = 5)
  (h3 : remaining_rooms = 40) :
  (remaining_rooms : ℚ) / ((original_rooms - remaining_rooms : ℕ) / lost_cans : ℚ) = 20 := by
  sorry

end NUMINAMATH_CALUDE_paint_cans_used_l554_55498


namespace NUMINAMATH_CALUDE_derivative_periodicity_l554_55442

theorem derivative_periodicity (f : ℝ → ℝ) (T : ℝ) (h_diff : Differentiable ℝ f) (h_periodic : ∀ x, f (x + T) = f x) (h_pos : T > 0) :
  ∀ x, deriv f (x + T) = deriv f x :=
by sorry

end NUMINAMATH_CALUDE_derivative_periodicity_l554_55442


namespace NUMINAMATH_CALUDE_second_grade_sample_l554_55450

/-- Represents the number of students to be sampled from a stratum in stratified sampling -/
def stratifiedSample (totalSample : ℕ) (stratumWeight : ℕ) (totalWeight : ℕ) : ℕ :=
  (stratumWeight * totalSample) / totalWeight

/-- Theorem: In a school with grades in 3:3:4 ratio, stratified sampling of 50 students
    results in 15 students from the second grade -/
theorem second_grade_sample :
  let totalSample : ℕ := 50
  let firstGradeWeight : ℕ := 3
  let secondGradeWeight : ℕ := 3
  let thirdGradeWeight : ℕ := 4
  let totalWeight : ℕ := firstGradeWeight + secondGradeWeight + thirdGradeWeight
  stratifiedSample totalSample secondGradeWeight totalWeight = 15 := by
  sorry

#eval stratifiedSample 50 3 10  -- Expected output: 15

end NUMINAMATH_CALUDE_second_grade_sample_l554_55450


namespace NUMINAMATH_CALUDE_total_height_climbed_l554_55492

/-- The number of staircases John climbs -/
def num_staircases : ℕ := 3

/-- The number of steps in the first staircase -/
def first_staircase : ℕ := 20

/-- The number of steps in the second staircase -/
def second_staircase : ℕ := 2 * first_staircase

/-- The number of steps in the third staircase -/
def third_staircase : ℕ := second_staircase - 10

/-- The height of each step in feet -/
def step_height : ℚ := 1/2

/-- The total number of steps climbed -/
def total_steps : ℕ := first_staircase + second_staircase + third_staircase

/-- The total height climbed in feet -/
def total_feet : ℚ := (total_steps : ℚ) * step_height

theorem total_height_climbed : total_feet = 45 := by
  sorry

end NUMINAMATH_CALUDE_total_height_climbed_l554_55492


namespace NUMINAMATH_CALUDE_ab_equals_op_l554_55479

noncomputable section

/-- Line l with parametric equations x = -1/2 * t, y = a + (√3/2) * t -/
def line_l (a t : ℝ) : ℝ × ℝ := (-1/2 * t, a + (Real.sqrt 3 / 2) * t)

/-- Curve C with rectangular equation x² + y² - 4x = 0 -/
def curve_C (x y : ℝ) : Prop := x^2 + y^2 - 4*x = 0

/-- Length of AB, where A and B are intersection points of line l and curve C -/
def length_AB (a : ℝ) : ℝ := Real.sqrt (4 + 4 * Real.sqrt 3 * a - a^2)

/-- Theorem stating that |AB| = 2 if and only if a = 0 or a = 4√3 -/
theorem ab_equals_op (a : ℝ) : length_AB a = 2 ↔ a = 0 ∨ a = 4 * Real.sqrt 3 := by
  sorry

end

end NUMINAMATH_CALUDE_ab_equals_op_l554_55479


namespace NUMINAMATH_CALUDE_course_selection_problem_l554_55480

/-- The number of ways to choose k items from n items -/
def choose (n k : ℕ) : ℕ := sorry

/-- The total number of ways two people can choose 2 courses each from 4 courses -/
def totalWays : ℕ :=
  choose 4 2 * choose 4 2

/-- The number of ways two people can choose 2 courses each from 4 courses with at least one course in common -/
def waysWithCommon : ℕ :=
  totalWays - choose 4 2

theorem course_selection_problem :
  (totalWays = 36) ∧
  (waysWithCommon / totalWays = 5 / 6) := by sorry

end NUMINAMATH_CALUDE_course_selection_problem_l554_55480


namespace NUMINAMATH_CALUDE_complex_fraction_equality_l554_55449

theorem complex_fraction_equality : ∃ (i : ℂ), i^2 = -1 ∧ (1 + 2*i) / ((1 - i)^2) = 1 - (1/2)*i := by
  sorry

end NUMINAMATH_CALUDE_complex_fraction_equality_l554_55449


namespace NUMINAMATH_CALUDE_unique_c_for_quadratic_equation_l554_55436

theorem unique_c_for_quadratic_equation :
  ∃! (c : ℝ), c ≠ 0 ∧
  (∃! (b : ℝ), b > 0 ∧
    (∃! (x : ℝ), x^2 + (b + 1/b) * x + c = 0)) ∧
  c = 1 := by
sorry

end NUMINAMATH_CALUDE_unique_c_for_quadratic_equation_l554_55436


namespace NUMINAMATH_CALUDE_expression_simplification_l554_55471

theorem expression_simplification (w : ℝ) : 2*w + 4*w + 6*w + 8*w + 10*w + 12 = 30*w + 12 := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l554_55471


namespace NUMINAMATH_CALUDE_modified_pattern_cannot_form_polyhedron_l554_55419

/-- Represents a flat pattern of squares -/
structure FlatPattern where
  squares : ℕ
  foldingLines : ℕ

/-- Represents a modified flat pattern with an extra square and a removed folding line -/
def ModifiedPattern (fp : FlatPattern) : FlatPattern :=
  { squares := fp.squares + 1
  , foldingLines := fp.foldingLines - 1 }

/-- Represents whether a pattern can form a simple polyhedron -/
def CanFormPolyhedron (fp : FlatPattern) : Prop := sorry

/-- Theorem stating that a modified pattern cannot form a simple polyhedron -/
theorem modified_pattern_cannot_form_polyhedron (fp : FlatPattern) : 
  ¬(CanFormPolyhedron (ModifiedPattern fp)) := by
  sorry

end NUMINAMATH_CALUDE_modified_pattern_cannot_form_polyhedron_l554_55419


namespace NUMINAMATH_CALUDE_quadratic_root_existence_l554_55438

theorem quadratic_root_existence (a b c x₁ x₂ : ℝ) (ha : a ≠ 0)
  (h1 : a * x₁^2 + b * x₁ + c = 0)
  (h2 : -a * x₂^2 + b * x₂ + c = 0) :
  ∃ x₃ : ℝ, (a / 2) * x₃^2 + b * x₃ + c = 0 ∧
    ((x₁ ≤ x₃ ∧ x₃ ≤ x₂) ∨ (x₁ ≥ x₃ ∧ x₃ ≥ x₂)) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_root_existence_l554_55438


namespace NUMINAMATH_CALUDE_select_five_from_eight_l554_55413

theorem select_five_from_eight : Nat.choose 8 5 = 56 := by sorry

end NUMINAMATH_CALUDE_select_five_from_eight_l554_55413


namespace NUMINAMATH_CALUDE_art_collection_area_is_282_l554_55467

/-- Calculates the total area of Davonte's art collection -/
def art_collection_area : ℕ :=
  let square_painting_area := 3 * (6 * 6)
  let small_painting_area := 4 * (2 * 3)
  let large_painting_area := 10 * 15
  square_painting_area + small_painting_area + large_painting_area

/-- Proves that the total area of Davonte's art collection is 282 square feet -/
theorem art_collection_area_is_282 : art_collection_area = 282 := by
  sorry

end NUMINAMATH_CALUDE_art_collection_area_is_282_l554_55467


namespace NUMINAMATH_CALUDE_distinct_primes_dividing_sequence_l554_55485

theorem distinct_primes_dividing_sequence (n M : ℕ) (h : M > n^(n-1)) :
  ∃ (p : Fin n → ℕ), (∀ i : Fin n, Nat.Prime (p i)) ∧ 
  (∀ i j : Fin n, i ≠ j → p i ≠ p j) ∧
  (∀ i : Fin n, (p i) ∣ (M + i.val + 1)) :=
sorry

end NUMINAMATH_CALUDE_distinct_primes_dividing_sequence_l554_55485


namespace NUMINAMATH_CALUDE_solution_set_f_range_of_m_l554_55430

-- Define the functions f and g
def f (x : ℝ) := |x + 3| + |x - 1|
def g (m : ℝ) (x : ℝ) := -x^2 + 2*m*x

-- Statement for the solution set of f(x) > 4
theorem solution_set_f (x : ℝ) : f x > 4 ↔ x < -3 ∨ x > 1 := by sorry

-- Statement for the range of m
theorem range_of_m (m : ℝ) : 
  (∀ x₁ x₂ : ℝ, f x₁ ≥ g m x₂) → -2 < m ∧ m < 2 := by sorry

end NUMINAMATH_CALUDE_solution_set_f_range_of_m_l554_55430


namespace NUMINAMATH_CALUDE_gcd_12n_plus_5_7n_plus_3_l554_55486

theorem gcd_12n_plus_5_7n_plus_3 (n : ℕ+) : Nat.gcd (12 * n + 5) (7 * n + 3) = 1 := by
  sorry

end NUMINAMATH_CALUDE_gcd_12n_plus_5_7n_plus_3_l554_55486


namespace NUMINAMATH_CALUDE_equal_selection_probability_l554_55410

/-- Represents a sampling method -/
inductive SamplingMethod
  | SimpleRandom
  | Systematic
  | Stratified

/-- The probability of an individual being selected given a population size, sample size, and sampling method -/
noncomputable def selectionProbability (N n : ℕ) (method : SamplingMethod) : ℝ :=
  sorry

theorem equal_selection_probability (N n : ℕ) (h1 : N > 0) (h2 : n > 0) (h3 : n ≤ N) :
  selectionProbability N n SamplingMethod.SimpleRandom =
  selectionProbability N n SamplingMethod.Systematic ∧
  selectionProbability N n SamplingMethod.SimpleRandom =
  selectionProbability N n SamplingMethod.Stratified :=
by
  sorry

end NUMINAMATH_CALUDE_equal_selection_probability_l554_55410


namespace NUMINAMATH_CALUDE_x_less_than_one_implications_l554_55412

theorem x_less_than_one_implications (x : ℝ) (h : x < 1) : x^3 < 1 ∧ |x| < 1 := by
  sorry

end NUMINAMATH_CALUDE_x_less_than_one_implications_l554_55412


namespace NUMINAMATH_CALUDE_impossible_to_get_50_51_l554_55417

/-- Represents the operation of replacing consecutive numbers with their count -/
def replace_with_count (s : List ℕ) (start : ℕ) (len : ℕ) : List ℕ := sorry

/-- Checks if a list contains only the numbers 50 and 51 -/
def contains_only_50_51 (s : List ℕ) : Prop := sorry

/-- The initial sequence of numbers from 1 to 100 -/
def initial_sequence : List ℕ := List.range 100

/-- Represents the result of applying the operation multiple times -/
def apply_operations (s : List ℕ) : List ℕ := sorry

theorem impossible_to_get_50_51 :
  ¬∃ (result : List ℕ), (apply_operations initial_sequence = result) ∧ (contains_only_50_51 result) := by
  sorry

end NUMINAMATH_CALUDE_impossible_to_get_50_51_l554_55417


namespace NUMINAMATH_CALUDE_simplify_expression_1_simplify_expression_2_l554_55457

-- Part 1
theorem simplify_expression_1 (a b : ℝ) : 2*a - (-3*b - 3*(3*a - b)) = 11*a := by sorry

-- Part 2
theorem simplify_expression_2 (a b : ℝ) : 12*a*b^2 - (7*a^2*b - (a*b^2 - 3*a^2*b)) = 13*a*b^2 - 10*a^2*b := by sorry

end NUMINAMATH_CALUDE_simplify_expression_1_simplify_expression_2_l554_55457


namespace NUMINAMATH_CALUDE_circle_properties_l554_55424

theorem circle_properties (A : ℝ) (h : A = 4 * Real.pi) :
  ∃ (r : ℝ), r > 0 ∧ A = Real.pi * r^2 ∧ 2 * r = 4 ∧ 2 * Real.pi * r = 4 * Real.pi :=
by
  sorry

end NUMINAMATH_CALUDE_circle_properties_l554_55424


namespace NUMINAMATH_CALUDE_car_down_payment_l554_55427

/-- Given a total down payment to be split equally among a number of people,
    rounding up to the nearest dollar, calculate the amount each person must pay. -/
def splitPayment (total : ℕ) (people : ℕ) : ℕ :=
  (total + people - 1) / people

theorem car_down_payment :
  splitPayment 3500 3 = 1167 := by
  sorry

end NUMINAMATH_CALUDE_car_down_payment_l554_55427


namespace NUMINAMATH_CALUDE_exactly_two_lines_l554_55473

/-- Two lines in 3D space -/
structure Line3D where
  -- We'll represent a line by a point and a direction vector
  point : ℝ × ℝ × ℝ
  direction : ℝ × ℝ × ℝ

/-- Angle between two lines -/
def angle_between_lines (l1 l2 : Line3D) : ℝ := sorry

/-- Check if two lines are skew -/
def are_skew (l1 l2 : Line3D) : Prop := sorry

/-- A point in 3D space -/
def Point3D := ℝ × ℝ × ℝ

/-- Count lines through a point forming a specific angle with two given lines -/
def count_lines_with_angle (a b : Line3D) (P : Point3D) (θ : ℝ) : ℕ := sorry

theorem exactly_two_lines 
  (a b : Line3D) (P : Point3D) 
  (h_skew : are_skew a b) 
  (h_angle : angle_between_lines a b = 40 * π / 180) :
  count_lines_with_angle a b P (30 * π / 180) = 2 := by
  sorry

end NUMINAMATH_CALUDE_exactly_two_lines_l554_55473


namespace NUMINAMATH_CALUDE_mean_problem_l554_55444

theorem mean_problem (x : ℝ) : 
  (12 + x + 42 + 78 + 104) / 5 = 62 → 
  (128 + 255 + 511 + 1023 + x) / 5 = 413 := by
sorry

end NUMINAMATH_CALUDE_mean_problem_l554_55444


namespace NUMINAMATH_CALUDE_z_purely_imaginary_z_squared_over_z_plus_5_plus_2i_l554_55476

def z (m : ℝ) : ℂ := (2 * m^2 - 3 * m - 2 : ℝ) + (m^2 - 3 * m + 2 : ℝ) * Complex.I

theorem z_purely_imaginary : z (-1/2) = Complex.I * ((-1/4)^2 - 3 * (-1/4) + 2) := by sorry

theorem z_squared_over_z_plus_5_plus_2i :
  z 0 ^ 2 / (z 0 + 5 + 2 * Complex.I) = -32/25 - 24/25 * Complex.I := by sorry

end NUMINAMATH_CALUDE_z_purely_imaginary_z_squared_over_z_plus_5_plus_2i_l554_55476


namespace NUMINAMATH_CALUDE_trigonometric_system_solution_l554_55401

theorem trigonometric_system_solution (x y z : ℝ) : 
  Real.sin x + Real.sin y + Real.sin (x + y + z) = 0 ∧
  Real.sin x + 2 * Real.sin z = 0 ∧
  Real.sin y + 3 * Real.sin z = 0 →
  ∃ (k m n : ℤ), x = π * k ∧ y = π * m ∧ z = π * n := by
sorry

end NUMINAMATH_CALUDE_trigonometric_system_solution_l554_55401


namespace NUMINAMATH_CALUDE_inequality_proof_equality_condition_l554_55494

theorem inequality_proof (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) (h_prod : x * y * z = 1) :
  (x * y) / (x^5 + x * y + y^5) + (y * z) / (y^5 + y * z + z^5) + (z * x) / (z^5 + z * x + x^5) ≤ 1 :=
by sorry

theorem equality_condition (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) (h_prod : x * y * z = 1) :
  (x * y) / (x^5 + x * y + y^5) + (y * z) / (y^5 + y * z + z^5) + (z * x) / (z^5 + z * x + x^5) = 1 ↔
  x = 1 ∧ y = 1 ∧ z = 1 :=
by sorry

end NUMINAMATH_CALUDE_inequality_proof_equality_condition_l554_55494


namespace NUMINAMATH_CALUDE_parabola_intersection_theorem_parabola_equation_l554_55474

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a parabola of the form y² = 2px -/
structure Parabola where
  p : ℝ
  hp : p > 0

/-- The origin point (0, 0) -/
def origin : Point := ⟨0, 0⟩

/-- Check if two line segments are perpendicular -/
def perpendicular (a b c d : Point) : Prop :=
  (b.x - a.x) * (d.x - c.x) + (b.y - a.y) * (d.y - c.y) = 0

/-- Theorem: If a parabola y² = 2px intersects the line x = 2 at points D and E,
    and OD ⊥ OE where O is the origin, then p = 1 -/
theorem parabola_intersection_theorem (C : Parabola) (D E : Point) :
  D.x = 2 ∧ E.x = 2 ∧                        -- D and E are on the line x = 2
  D.y^2 = 2 * C.p * D.x ∧ E.y^2 = 2 * C.p * E.x ∧  -- D and E are on the parabola
  perpendicular origin D origin E            -- OD ⊥ OE
  → C.p = 1 := by sorry

/-- Corollary: Under the conditions of the theorem, the parabola's equation is y² = 2x -/
theorem parabola_equation (C : Parabola) (D E : Point) :
  D.x = 2 ∧ E.x = 2 ∧
  D.y^2 = 2 * C.p * D.x ∧ E.y^2 = 2 * C.p * E.x ∧
  perpendicular origin D origin E
  → ∀ x y : ℝ, y^2 = 2 * x ↔ y^2 = 2 * C.p * x := by sorry

end NUMINAMATH_CALUDE_parabola_intersection_theorem_parabola_equation_l554_55474


namespace NUMINAMATH_CALUDE_permutations_with_non_adjacency_l554_55421

theorem permutations_with_non_adjacency (n : ℕ) (h : n ≥ 4) :
  let total_permutations := n.factorial
  let adjacent_a1_a2 := 2 * (n - 1).factorial
  let adjacent_a3_a4 := 2 * (n - 1).factorial
  let both_adjacent := 4 * (n - 2).factorial
  total_permutations - adjacent_a1_a2 - adjacent_a3_a4 + both_adjacent = (n^2 - 5*n + 8) * (n - 2).factorial :=
by sorry

#check permutations_with_non_adjacency

end NUMINAMATH_CALUDE_permutations_with_non_adjacency_l554_55421
