import Mathlib

namespace NUMINAMATH_CALUDE_sales_discount_effect_l1355_135554

theorem sales_discount_effect (P N : ℝ) (h_positive : P > 0 ∧ N > 0) :
  let D : ℝ := 10  -- Discount percentage
  let new_price : ℝ := P * (1 - D / 100)
  let new_quantity : ℝ := N * 1.20
  let original_income : ℝ := P * N
  let new_income : ℝ := new_price * new_quantity
  (new_quantity = N * 1.20) ∧ (new_income = original_income * 1.08) :=
by sorry

end NUMINAMATH_CALUDE_sales_discount_effect_l1355_135554


namespace NUMINAMATH_CALUDE_problem_solution_l1355_135556

theorem problem_solution (f : ℝ → ℝ) (m : ℝ) (a b c : ℝ) : 
  (∀ x, f x = m - |x - 2|) →
  ({x | f (x + 2) ≥ 0} = Set.Icc (-1) 1) →
  (1/a + 1/(2*b) + 1/(3*c) = m) →
  (m = 1 ∧ a + 2*b + 3*c ≥ 9) := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l1355_135556


namespace NUMINAMATH_CALUDE_average_reading_time_emery_serena_l1355_135504

/-- The average reading time for two people, given one person's reading speed and time -/
def averageReadingTime (fasterReaderTime : ℕ) (speedRatio : ℕ) : ℚ :=
  (fasterReaderTime + fasterReaderTime * speedRatio) / 2

/-- Theorem: The average reading time for Emery and Serena is 60 days -/
theorem average_reading_time_emery_serena :
  averageReadingTime 20 5 = 60 := by
  sorry

end NUMINAMATH_CALUDE_average_reading_time_emery_serena_l1355_135504


namespace NUMINAMATH_CALUDE_hexagon_minus_triangle_area_l1355_135521

/-- The area of a hexagon with side length 2 and height 4, minus the area of an inscribed equilateral triangle with side length 4 -/
theorem hexagon_minus_triangle_area : 
  let hexagon_side : ℝ := 2
  let hexagon_height : ℝ := 4
  let triangle_side : ℝ := 4
  let hexagon_area : ℝ := 6 * (1/2 * hexagon_side * hexagon_height)
  let triangle_area : ℝ := (Real.sqrt 3 / 4) * triangle_side^2
  hexagon_area - triangle_area = 24 - 4 * Real.sqrt 3 :=
by sorry

end NUMINAMATH_CALUDE_hexagon_minus_triangle_area_l1355_135521


namespace NUMINAMATH_CALUDE_wolverine_workout_hours_l1355_135516

def workout_hours (rayman junior wolverine : ℕ) : Prop :=
  rayman = junior / 2 ∧ 
  wolverine = 2 * (rayman + junior) ∧ 
  rayman = 10

theorem wolverine_workout_hours :
  ∀ rayman junior wolverine : ℕ,
  workout_hours rayman junior wolverine →
  wolverine = 60 := by
sorry

end NUMINAMATH_CALUDE_wolverine_workout_hours_l1355_135516


namespace NUMINAMATH_CALUDE_necessary_not_sufficient_condition_l1355_135537

theorem necessary_not_sufficient_condition (a b : ℝ) : 
  (∀ a b : ℝ, a < b → a < b + 1) ∧ 
  (∃ a b : ℝ, a < b + 1 ∧ ¬(a < b)) := by
  sorry

end NUMINAMATH_CALUDE_necessary_not_sufficient_condition_l1355_135537


namespace NUMINAMATH_CALUDE_aunt_may_milk_problem_l1355_135562

/-- Aunt May's milk problem -/
theorem aunt_may_milk_problem (morning_milk : ℕ) (evening_milk : ℕ) (sold_milk : ℕ) (leftover_milk : ℕ)
  (h1 : morning_milk = 365)
  (h2 : evening_milk = 380)
  (h3 : sold_milk = 612)
  (h4 : leftover_milk = 15) :
  morning_milk + evening_milk + leftover_milk - sold_milk = 148 :=
by sorry

end NUMINAMATH_CALUDE_aunt_may_milk_problem_l1355_135562


namespace NUMINAMATH_CALUDE_complex_number_properties_l1355_135519

theorem complex_number_properties (z : ℂ) (h : z - 2*I = z*I + 4) : 
  Complex.abs z = Real.sqrt 10 ∧ ((z - 1) / 3) ^ 2023 = -I := by
  sorry

end NUMINAMATH_CALUDE_complex_number_properties_l1355_135519


namespace NUMINAMATH_CALUDE_erasers_bought_l1355_135524

theorem erasers_bought (initial_erasers final_erasers : ℝ) (h1 : initial_erasers = 95.0) (h2 : final_erasers = 137) : 
  final_erasers - initial_erasers = 42 := by
  sorry

end NUMINAMATH_CALUDE_erasers_bought_l1355_135524


namespace NUMINAMATH_CALUDE_eighth_group_selection_l1355_135545

/-- Systematic sampling from a population -/
def systematicSampling (totalPopulation : ℕ) (numGroups : ℕ) (firstGroupSelection : ℕ) (targetGroup : ℕ) : ℕ :=
  (targetGroup - 1) * (totalPopulation / numGroups) + firstGroupSelection

/-- Theorem: In a systematic sampling of 30 groups from 480 students, 
    if the selected number from the first group is 5, 
    then the selected number from the eighth group is 117. -/
theorem eighth_group_selection :
  systematicSampling 480 30 5 8 = 117 := by
  sorry

end NUMINAMATH_CALUDE_eighth_group_selection_l1355_135545


namespace NUMINAMATH_CALUDE_apples_per_person_is_two_l1355_135503

/-- Calculates the number of pounds of apples each person gets in a family -/
def applesPerPerson (originalPrice : ℚ) (priceIncrease : ℚ) (totalCost : ℚ) (familySize : ℕ) : ℚ :=
  let newPrice := originalPrice * (1 + priceIncrease)
  let costPerPerson := totalCost / familySize
  costPerPerson / newPrice

/-- Theorem stating that under the given conditions, each person gets 2 pounds of apples -/
theorem apples_per_person_is_two :
  applesPerPerson (8/5) (1/4) 16 4 = 2 := by
  sorry

end NUMINAMATH_CALUDE_apples_per_person_is_two_l1355_135503


namespace NUMINAMATH_CALUDE_quadratic_max_l1355_135568

/-- The function f(x) = -2x^2 + 8x - 6 achieves its maximum value when x = 2 -/
theorem quadratic_max (x : ℝ) : 
  ∀ y : ℝ, -2 * x^2 + 8 * x - 6 ≥ -2 * y^2 + 8 * y - 6 ↔ x = 2 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_max_l1355_135568


namespace NUMINAMATH_CALUDE_coffee_lasts_12_days_l1355_135597

-- Define the constants
def coffee_lbs : ℕ := 3
def cups_per_lb : ℕ := 40
def weekday_consumption : ℕ := 3 + 2 + 4
def weekend_consumption : ℕ := 2 + 3 + 5
def days_in_week : ℕ := 7
def weekdays_per_week : ℕ := 5
def weekend_days_per_week : ℕ := 2

-- Define the theorem
theorem coffee_lasts_12_days :
  let total_cups := coffee_lbs * cups_per_lb
  let weekly_consumption := weekday_consumption * weekdays_per_week + weekend_consumption * weekend_days_per_week
  let days_coffee_lasts := (total_cups * days_in_week) / weekly_consumption
  days_coffee_lasts = 12 :=
by sorry

end NUMINAMATH_CALUDE_coffee_lasts_12_days_l1355_135597


namespace NUMINAMATH_CALUDE_historical_fiction_new_releases_fraction_l1355_135595

/-- Represents a bookstore inventory -/
structure Bookstore where
  total : ℕ
  historical_fiction : ℕ
  historical_fiction_new : ℕ
  other_new : ℕ

/-- Conditions for Joel's bookstore -/
def joels_bookstore (b : Bookstore) : Prop :=
  b.historical_fiction = (2 * b.total) / 5 ∧
  b.historical_fiction_new = (2 * b.historical_fiction) / 5 ∧
  b.other_new = (2 * (b.total - b.historical_fiction)) / 5

/-- Theorem: In Joel's bookstore, 2/5 of all new releases are historical fiction -/
theorem historical_fiction_new_releases_fraction (b : Bookstore) 
  (h : joels_bookstore b) : 
  (b.historical_fiction_new : ℚ) / (b.historical_fiction_new + b.other_new) = 2 / 5 := by
  sorry

end NUMINAMATH_CALUDE_historical_fiction_new_releases_fraction_l1355_135595


namespace NUMINAMATH_CALUDE_fraction_multiplication_l1355_135509

theorem fraction_multiplication : ((1 / 4 : ℚ) * (1 / 8 : ℚ)) * 4 = 1 / 8 := by
  sorry

end NUMINAMATH_CALUDE_fraction_multiplication_l1355_135509


namespace NUMINAMATH_CALUDE_radical_simplification_l1355_135560

theorem radical_simplification (x : ℝ) (h : 4 < x ∧ x < 7) :
  (((x - 4) ^ 4) ^ (1/4 : ℝ)) + (((x - 7) ^ 4) ^ (1/4 : ℝ)) = 3 := by
  sorry

end NUMINAMATH_CALUDE_radical_simplification_l1355_135560


namespace NUMINAMATH_CALUDE_committee_age_difference_l1355_135531

theorem committee_age_difference (n : ℕ) (A : ℝ) (O N : ℝ) : 
  n = 20 → 
  n * A = n * A + O - N → 
  O - N = 160 :=
by
  sorry

end NUMINAMATH_CALUDE_committee_age_difference_l1355_135531


namespace NUMINAMATH_CALUDE_position_after_five_steps_l1355_135579

/-- A student's walk on a number line --/
structure StudentWalk where
  total_steps : ℕ
  total_distance : ℝ
  step_length : ℝ
  marking_distance : ℝ

/-- The position after a certain number of steps --/
def position_after_steps (walk : StudentWalk) (steps : ℕ) : ℝ :=
  walk.step_length * steps

/-- The theorem to prove --/
theorem position_after_five_steps (walk : StudentWalk) 
  (h1 : walk.total_steps = 8)
  (h2 : walk.total_distance = 48)
  (h3 : walk.marking_distance = 3)
  (h4 : walk.step_length = walk.total_distance / walk.total_steps) :
  position_after_steps walk 5 = 30 := by
  sorry

end NUMINAMATH_CALUDE_position_after_five_steps_l1355_135579


namespace NUMINAMATH_CALUDE_ceiling_sum_equals_56_l1355_135550

theorem ceiling_sum_equals_56 :
  ⌈Real.sqrt 3⌉ + ⌈Real.sqrt 30⌉^2 + ⌈Real.sqrt 300⌉ = 56 := by
  sorry

end NUMINAMATH_CALUDE_ceiling_sum_equals_56_l1355_135550


namespace NUMINAMATH_CALUDE_x_squared_y_not_less_than_x_cubed_plus_y_fifth_l1355_135591

theorem x_squared_y_not_less_than_x_cubed_plus_y_fifth 
  (x y : ℝ) (hx : 0 < x ∧ x < 1) (hy : 0 < y ∧ y < 1) : 
  x^2 * y ≥ x^3 + y^5 := by
  sorry

end NUMINAMATH_CALUDE_x_squared_y_not_less_than_x_cubed_plus_y_fifth_l1355_135591


namespace NUMINAMATH_CALUDE_square_area_ratio_l1355_135567

/-- The ratio of the areas of two squares, where one has a side length 5 times the other, is 1/25. -/
theorem square_area_ratio (y : ℝ) (h : y > 0) : 
  (y^2) / ((5*y)^2) = 1 / 25 := by sorry

end NUMINAMATH_CALUDE_square_area_ratio_l1355_135567


namespace NUMINAMATH_CALUDE_find_a_l1355_135559

-- Define the universal set U
def U (a : ℝ) : Set ℝ := {2, 4, 3 - a^2}

-- Define set P
def P (a : ℝ) : Set ℝ := {2, a^2 - a + 2}

-- Define the complement of P with respect to U
def complement_P (a : ℝ) : Set ℝ := {-1}

-- Theorem statement
theorem find_a : ∃ a : ℝ, (U a = P a ∪ complement_P a) ∧ (a = 2) := by
  sorry

end NUMINAMATH_CALUDE_find_a_l1355_135559


namespace NUMINAMATH_CALUDE_constant_term_in_expansion_l1355_135588

theorem constant_term_in_expansion :
  ∃ (k : ℕ), k > 0 ∧ k < 5 ∧ (2 * 5 = 5 * k) := by
  sorry

end NUMINAMATH_CALUDE_constant_term_in_expansion_l1355_135588


namespace NUMINAMATH_CALUDE_lost_card_number_l1355_135583

theorem lost_card_number (n : ℕ) (h1 : n > 0) (h2 : (n * (n + 1)) / 2 - 101 ≤ n) : 
  (n * (n + 1)) / 2 - 101 = 4 := by
  sorry

#check lost_card_number

end NUMINAMATH_CALUDE_lost_card_number_l1355_135583


namespace NUMINAMATH_CALUDE_farmer_tomatoes_l1355_135533

def initial_tomatoes (picked_yesterday : ℕ) (picked_today : ℕ) (remaining : ℕ) : ℕ :=
  picked_yesterday + picked_today + remaining

theorem farmer_tomatoes : initial_tomatoes 134 30 7 = 171 := by
  sorry

end NUMINAMATH_CALUDE_farmer_tomatoes_l1355_135533


namespace NUMINAMATH_CALUDE_equation_solution_l1355_135539

theorem equation_solution :
  ∀ x : ℚ, 7 * (4 * x + 3) - 3 = -3 * (2 - 5 * x) + 5 * x / 2 ↔ x = -16 / 7 :=
by sorry

end NUMINAMATH_CALUDE_equation_solution_l1355_135539


namespace NUMINAMATH_CALUDE_power_five_mod_seven_l1355_135543

theorem power_five_mod_seven : 5^207 % 7 = 6 := by
  sorry

end NUMINAMATH_CALUDE_power_five_mod_seven_l1355_135543


namespace NUMINAMATH_CALUDE_first_race_length_l1355_135511

/-- Represents the length of the first race in meters -/
def L : ℝ := sorry

/-- Theorem stating that the length of the first race is 100 meters -/
theorem first_race_length : L = 100 := by
  -- Define the relationships between runners based on the given conditions
  let A_finish := L
  let B_finish := L - 10
  let C_finish := L - 13
  
  -- Define the relationship in the second race
  let B_second_race := 180
  let C_second_race := 174  -- 180 - 6
  
  -- The ratio of B's performance to C's performance should be consistent across races
  have ratio_equality : (B_finish / C_finish) = (B_second_race / C_second_race) := by sorry
  
  -- Use the ratio equality to solve for L
  sorry

end NUMINAMATH_CALUDE_first_race_length_l1355_135511


namespace NUMINAMATH_CALUDE_fibonacci_pythagorean_hypotenuse_l1355_135561

def fibonacci : ℕ → ℕ
  | 0 => 0
  | 1 => 1
  | (n + 2) => fibonacci (n + 1) + fibonacci n

theorem fibonacci_pythagorean_hypotenuse (k : ℕ) (h : k ≥ 2) :
  fibonacci (2 * k + 1) = fibonacci k ^ 2 + fibonacci (k + 1) ^ 2 := by
  sorry

end NUMINAMATH_CALUDE_fibonacci_pythagorean_hypotenuse_l1355_135561


namespace NUMINAMATH_CALUDE_perfect_square_trinomial_condition_l1355_135500

/-- A trinomial ax^2 + bx + c is a perfect square if there exist real numbers p and q
    such that ax^2 + bx + c = (px + q)^2 for all x. -/
def is_perfect_square_trinomial (a b c : ℝ) : Prop :=
  ∃ p q : ℝ, ∀ x, a * x^2 + b * x + c = (p * x + q)^2

/-- If 4x^2 + 2kx + 9 is a perfect square trinomial, then k = ±6. -/
theorem perfect_square_trinomial_condition (k : ℝ) :
  is_perfect_square_trinomial 4 (2 * k) 9 → k = 6 ∨ k = -6 :=
by
  sorry


end NUMINAMATH_CALUDE_perfect_square_trinomial_condition_l1355_135500


namespace NUMINAMATH_CALUDE_rhombus_sides_not_equal_l1355_135517

theorem rhombus_sides_not_equal (d1 d2 p : ℝ) (h1 : d1 = 30) (h2 : d2 = 18) (h3 : p = 80) :
  ¬ (4 * Real.sqrt ((d1 / 2) ^ 2 + (d2 / 2) ^ 2) = p) :=
by sorry

end NUMINAMATH_CALUDE_rhombus_sides_not_equal_l1355_135517


namespace NUMINAMATH_CALUDE_inequality_proof_l1355_135512

theorem inequality_proof (x : ℝ) : (2*x - 1)/3 + 1 ≤ 0 → x ≤ -1 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l1355_135512


namespace NUMINAMATH_CALUDE_jane_ribbons_theorem_l1355_135544

/-- The number of dresses Jane sews per day in the first week -/
def dresses_per_day_week1 : ℕ := 2

/-- The number of days Jane sews in the first week -/
def days_week1 : ℕ := 7

/-- The number of dresses Jane sews per day in the second period -/
def dresses_per_day_week2 : ℕ := 3

/-- The number of days Jane sews in the second period -/
def days_week2 : ℕ := 2

/-- The number of ribbons Jane adds to each dress -/
def ribbons_per_dress : ℕ := 2

/-- The total number of ribbons Jane uses -/
def total_ribbons : ℕ := 40

theorem jane_ribbons_theorem : 
  (dresses_per_day_week1 * days_week1 + dresses_per_day_week2 * days_week2) * ribbons_per_dress = total_ribbons := by
  sorry

end NUMINAMATH_CALUDE_jane_ribbons_theorem_l1355_135544


namespace NUMINAMATH_CALUDE_trig_identity_l1355_135555

theorem trig_identity (α : ℝ) : 
  (Real.cos (π / 2 - α / 4) - Real.sin (π / 2 - α / 4) * Real.tan (α / 8)) / 
  (Real.sin (7 * π / 2 - α / 4) + Real.sin (α / 4 - 3 * π) * Real.tan (α / 8)) = 
  -Real.tan (α / 8) := by sorry

end NUMINAMATH_CALUDE_trig_identity_l1355_135555


namespace NUMINAMATH_CALUDE_integral_inequality_l1355_135549

open MeasureTheory

theorem integral_inequality 
  (f g : ℝ → ℝ) 
  (hf_pos : ∀ x, 0 ≤ f x) 
  (hg_pos : ∀ x, 0 ≤ g x)
  (hf_cont : Continuous f) 
  (hg_cont : Continuous g)
  (hf_incr : MonotoneOn f (Set.Icc 0 1))
  (hg_decr : AntitoneOn g (Set.Icc 0 1)) :
  ∫ x in (Set.Icc 0 1), f x * g x ≤ ∫ x in (Set.Icc 0 1), f x * g (1 - x) :=
sorry

end NUMINAMATH_CALUDE_integral_inequality_l1355_135549


namespace NUMINAMATH_CALUDE_combination_problem_l1355_135551

theorem combination_problem (m : ℕ) : 
  (1 : ℚ) / (Nat.choose 5 m) - (1 : ℚ) / (Nat.choose 6 m) = (7 : ℚ) / (10 * Nat.choose 7 m) →
  Nat.choose 21 m = 210 := by
  sorry

end NUMINAMATH_CALUDE_combination_problem_l1355_135551


namespace NUMINAMATH_CALUDE_bathroom_kitchen_bulbs_l1355_135514

theorem bathroom_kitchen_bulbs 
  (total_packs : ℕ) 
  (bulbs_per_pack : ℕ) 
  (bedroom_bulbs : ℕ) 
  (basement_bulbs : ℕ) 
  (h1 : total_packs = 6) 
  (h2 : bulbs_per_pack = 2) 
  (h3 : bedroom_bulbs = 2) 
  (h4 : basement_bulbs = 4) :
  total_packs * bulbs_per_pack - (bedroom_bulbs + basement_bulbs + basement_bulbs / 2) = 4 := by
  sorry

end NUMINAMATH_CALUDE_bathroom_kitchen_bulbs_l1355_135514


namespace NUMINAMATH_CALUDE_ages_solution_l1355_135571

/-- Represents the ages of Rahul and Deepak -/
structure Ages where
  rahul : ℕ
  deepak : ℕ

/-- The conditions of the problem -/
def satisfies_conditions (ages : Ages) : Prop :=
  -- The ratio between Rahul and Deepak's age is 4:3
  4 * ages.deepak = 3 * ages.rahul ∧
  -- In 6 years, Rahul will be 26 years old
  ages.rahul + 6 = 26 ∧
  -- In 6 years, Deepak's age will be equal to half the sum of Rahul's present and future ages
  ages.deepak + 6 = (ages.rahul + (ages.rahul + 6)) / 2 ∧
  -- Five years after that, the sum of their ages will be 59
  (ages.rahul + 11) + (ages.deepak + 11) = 59

/-- The theorem to prove -/
theorem ages_solution :
  ∃ (ages : Ages), satisfies_conditions ages ∧ ages.rahul = 20 ∧ ages.deepak = 17 := by
  sorry

end NUMINAMATH_CALUDE_ages_solution_l1355_135571


namespace NUMINAMATH_CALUDE_recurring_decimal_division_l1355_135526

def repeating_decimal_to_fraction (a b c : ℕ) : ℚ :=
  (a * 1000 + b * 100 + c * 10 + a * 1 + b * (1/10) + c * (1/100)) / 999

theorem recurring_decimal_division (a b c d e f : ℕ) :
  (repeating_decimal_to_fraction a b c) / (1 + repeating_decimal_to_fraction d e f) = 714 / 419 :=
by
  sorry

end NUMINAMATH_CALUDE_recurring_decimal_division_l1355_135526


namespace NUMINAMATH_CALUDE_complex_power_four_l1355_135558

-- Define the complex number i
def i : ℂ := Complex.I

-- State the theorem
theorem complex_power_four (h : i^2 = -1) : (1 - i)^4 = -4 := by
  sorry

end NUMINAMATH_CALUDE_complex_power_four_l1355_135558


namespace NUMINAMATH_CALUDE_abs_two_over_one_plus_i_l1355_135506

theorem abs_two_over_one_plus_i : Complex.abs (2 / (1 + Complex.I)) = Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_abs_two_over_one_plus_i_l1355_135506


namespace NUMINAMATH_CALUDE_circular_dome_larger_interior_angle_l1355_135584

/-- A circular dome structure constructed from congruent isosceles trapezoids. -/
structure CircularDome where
  /-- The number of trapezoids in the dome -/
  num_trapezoids : ℕ
  /-- The measure of the larger interior angle of each trapezoid in degrees -/
  larger_interior_angle : ℝ

/-- Theorem: In a circular dome constructed from 10 congruent isosceles trapezoids,
    where the non-parallel sides of the trapezoids extend to meet at the center of
    the circle formed by the base of the dome, the measure of the larger interior
    angle of each trapezoid is 81°. -/
theorem circular_dome_larger_interior_angle
  (dome : CircularDome)
  (h₁ : dome.num_trapezoids = 10)
  : dome.larger_interior_angle = 81 := by
  sorry

end NUMINAMATH_CALUDE_circular_dome_larger_interior_angle_l1355_135584


namespace NUMINAMATH_CALUDE_supplementary_angles_difference_l1355_135594

theorem supplementary_angles_difference (a b : ℝ) : 
  a + b = 180 →  -- supplementary angles
  a / b = 5 / 3 →  -- ratio of 5:3
  abs (a - b) = 45 :=  -- positive difference
by sorry

end NUMINAMATH_CALUDE_supplementary_angles_difference_l1355_135594


namespace NUMINAMATH_CALUDE_inequality_solution_set_l1355_135532

theorem inequality_solution_set (x : ℝ) :
  (3 - x) / (2 * x - 4) < 1 ↔ x > 2 := by sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l1355_135532


namespace NUMINAMATH_CALUDE_absolute_value_equation_l1355_135513

theorem absolute_value_equation (a : ℝ) : |a - 1| = 2 → a = 3 ∨ a = -1 := by
  sorry

end NUMINAMATH_CALUDE_absolute_value_equation_l1355_135513


namespace NUMINAMATH_CALUDE_parallelogram_roots_l1355_135510

/-- The polynomial equation with parameter b -/
def P (b : ℝ) (z : ℂ) : ℂ :=
  z^4 - 8*z^3 + 15*b*z^2 - 5*(3*b^2 + 4*b - 4)*z + 9

/-- Condition for roots to form a parallelogram -/
def forms_parallelogram (b : ℝ) : Prop :=
  ∃ (w₁ w₂ : ℂ), (P b w₁ = 0) ∧ (P b (-w₁) = 0) ∧ (P b w₂ = 0) ∧ (P b (-w₂) = 0)

/-- The main theorem stating the values of b for which the roots form a parallelogram -/
theorem parallelogram_roots :
  ∀ b : ℝ, forms_parallelogram b ↔ (b = 2/3 ∨ b = -2) :=
sorry

end NUMINAMATH_CALUDE_parallelogram_roots_l1355_135510


namespace NUMINAMATH_CALUDE_camillas_jelly_beans_l1355_135523

theorem camillas_jelly_beans (b c : ℕ) : 
  b = 2 * c →                     -- Initial condition: twice as many blueberry as cherry
  b - 10 = 3 * (c - 10) →         -- Condition after eating: three times as many blueberry as cherry
  b = 40                          -- Conclusion: original number of blueberry jelly beans
:= by sorry

end NUMINAMATH_CALUDE_camillas_jelly_beans_l1355_135523


namespace NUMINAMATH_CALUDE_ahmed_goats_l1355_135527

/-- Given information about goats owned by Adam, Andrew, and Ahmed -/
theorem ahmed_goats (adam : ℕ) (andrew : ℕ) (ahmed : ℕ) : 
  adam = 7 →
  andrew = 2 * adam + 5 →
  ahmed = andrew - 6 →
  ahmed = 13 := by sorry

end NUMINAMATH_CALUDE_ahmed_goats_l1355_135527


namespace NUMINAMATH_CALUDE_polyhedron_20_faces_l1355_135578

/-- A polyhedron with triangular faces -/
structure Polyhedron where
  faces : Nat
  is_triangular : Bool

/-- The number of edges in a polyhedron -/
def num_edges (p : Polyhedron) : Nat :=
  3 * p.faces / 2

/-- The number of vertices in a polyhedron -/
def num_vertices (p : Polyhedron) : Nat :=
  p.faces + 2 - num_edges p

/-- Theorem: A polyhedron with 20 triangular faces has 30 edges and 12 vertices -/
theorem polyhedron_20_faces (p : Polyhedron) 
  (h1 : p.faces = 20) 
  (h2 : p.is_triangular = true) : 
  num_edges p = 30 ∧ num_vertices p = 12 := by
  sorry


end NUMINAMATH_CALUDE_polyhedron_20_faces_l1355_135578


namespace NUMINAMATH_CALUDE_permutation_remainder_cardinality_l1355_135525

theorem permutation_remainder_cardinality 
  (a : Fin 100 → Fin 100) 
  (h_perm : Function.Bijective a) :
  let b : Fin 100 → ℕ := fun i => (Finset.range i.succ).sum (fun j => (a j).val + 1)
  let r : Fin 100 → Fin 100 := fun i => (b i) % 100
  Finset.card (Finset.image r (Finset.univ : Finset (Fin 100))) ≥ 11 :=
by
  sorry

end NUMINAMATH_CALUDE_permutation_remainder_cardinality_l1355_135525


namespace NUMINAMATH_CALUDE_union_of_M_and_N_l1355_135540

def M : Set ℝ := {x : ℝ | -3 < x ∧ x ≤ 5}
def N : Set ℝ := {x : ℝ | x > 3}

theorem union_of_M_and_N : M ∪ N = {x : ℝ | x > -3} := by sorry

end NUMINAMATH_CALUDE_union_of_M_and_N_l1355_135540


namespace NUMINAMATH_CALUDE_sequence_properties_arithmetic_sequence_l1355_135582

def a_n (n a : ℕ+) : ℚ := n / (n + a)

theorem sequence_properties (a : ℕ+) :
  (∃ r : ℚ, a_n 1 a * r = a_n 3 a ∧ a_n 3 a * r = a_n 15 a) →
  a = 9 :=
sorry

theorem arithmetic_sequence (a k : ℕ+) :
  k ≥ 3 →
  (a_n 1 a + a_n k a = 2 * a_n 2 a) →
  ((a = 1 ∧ k = 5) ∨ (a = 2 ∧ k = 4)) :=
sorry

end NUMINAMATH_CALUDE_sequence_properties_arithmetic_sequence_l1355_135582


namespace NUMINAMATH_CALUDE_square_diff_and_product_l1355_135529

theorem square_diff_and_product (a b : ℝ) 
  (sum_eq : a + b = 10) 
  (diff_eq : a - b = 4) 
  (sum_squares_eq : a^2 + b^2 = 58) : 
  a^2 - b^2 = 40 ∧ a * b = 21 := by
  sorry

end NUMINAMATH_CALUDE_square_diff_and_product_l1355_135529


namespace NUMINAMATH_CALUDE_answer_key_combinations_l1355_135596

/-- The number of answer choices for each multiple-choice question -/
def multiple_choice_options : ℕ := 4

/-- The number of true-false questions -/
def true_false_questions : ℕ := 5

/-- The number of multiple-choice questions -/
def multiple_choice_questions : ℕ := 2

/-- The total number of possible true-false answer combinations -/
def total_true_false_combinations : ℕ := 2^true_false_questions

/-- The number of true-false combinations where all answers are the same -/
def same_answer_combinations : ℕ := 2

/-- The number of valid true-false combinations (excluding all same answers) -/
def valid_true_false_combinations : ℕ := total_true_false_combinations - same_answer_combinations

/-- The total number of possible multiple-choice answer combinations -/
def multiple_choice_combinations : ℕ := multiple_choice_options^multiple_choice_questions

/-- The theorem stating the total number of ways to create the answer key -/
theorem answer_key_combinations : 
  valid_true_false_combinations * multiple_choice_combinations = 480 := by
  sorry

end NUMINAMATH_CALUDE_answer_key_combinations_l1355_135596


namespace NUMINAMATH_CALUDE_log_equality_l1355_135541

theorem log_equality (x k : ℝ) :
  (Real.log 3 / Real.log 4 = x) →
  (Real.log 9 / Real.log 2 = k * x) →
  k = 4 := by
sorry

end NUMINAMATH_CALUDE_log_equality_l1355_135541


namespace NUMINAMATH_CALUDE_riverside_denial_rate_l1355_135565

theorem riverside_denial_rate (total_kids : ℕ) (riverside_kids : ℕ) (westside_kids : ℕ) (mountaintop_kids : ℕ)
  (westside_denial_rate : ℚ) (mountaintop_denial_rate : ℚ) (kids_admitted : ℕ) :
  total_kids = riverside_kids + westside_kids + mountaintop_kids →
  total_kids = 260 →
  riverside_kids = 120 →
  westside_kids = 90 →
  mountaintop_kids = 50 →
  westside_denial_rate = 7/10 →
  mountaintop_denial_rate = 1/2 →
  kids_admitted = 148 →
  (riverside_kids - (total_kids - kids_admitted - (westside_denial_rate * westside_kids).num
    - (mountaintop_denial_rate * mountaintop_kids).num)) / riverside_kids = 4/5 := by
  sorry

end NUMINAMATH_CALUDE_riverside_denial_rate_l1355_135565


namespace NUMINAMATH_CALUDE_quadratic_equation_sum_l1355_135507

theorem quadratic_equation_sum (x q t : ℝ) : 
  (9 * x^2 - 36 * x - 81 = 0) →
  ((x + q)^2 = t) →
  (9 * (x + q)^2 = 9 * t) →
  (q + t = 11) := by
sorry

end NUMINAMATH_CALUDE_quadratic_equation_sum_l1355_135507


namespace NUMINAMATH_CALUDE_marble_problem_solution_l1355_135599

/-- Represents the number of marbles of each color in a box -/
structure MarbleBox where
  red : ℕ
  green : ℕ
  yellow : ℕ
  other : ℕ

/-- Calculates the total number of marbles in the box -/
def MarbleBox.total (box : MarbleBox) : ℕ :=
  box.red + box.green + box.yellow + box.other

/-- Represents the conditions of the marble problem -/
def marble_problem (box : MarbleBox) : Prop :=
  box.red = 20 ∧
  box.green = 3 * box.red ∧
  box.yellow = box.green / 5 ∧
  box.total = 4 * box.green

theorem marble_problem_solution (box : MarbleBox) :
  marble_problem box → box.other = 148 := by
  sorry

end NUMINAMATH_CALUDE_marble_problem_solution_l1355_135599


namespace NUMINAMATH_CALUDE_garden_length_is_140_l1355_135534

/-- Represents a rectangular garden -/
structure RectangularGarden where
  length : ℝ
  breadth : ℝ

/-- Calculates the perimeter of a rectangular garden -/
def perimeter (g : RectangularGarden) : ℝ :=
  2 * (g.length + g.breadth)

/-- Theorem: A rectangular garden with perimeter 480 m and breadth 100 m has length 140 m -/
theorem garden_length_is_140
  (g : RectangularGarden)
  (h1 : perimeter g = 480)
  (h2 : g.breadth = 100) :
  g.length = 140 := by
  sorry

end NUMINAMATH_CALUDE_garden_length_is_140_l1355_135534


namespace NUMINAMATH_CALUDE_quadratic_equation_roots_l1355_135518

theorem quadratic_equation_roots (m : ℝ) : 
  let f : ℝ → ℝ := λ x => x^2 - 6*m*x + 9*m^2 - 4
  ∃ x₁ x₂ : ℝ, x₁ > x₂ ∧ f x₁ = 0 ∧ f x₂ = 0 ∧ x₁ = 2*x₂ → m = 2 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_roots_l1355_135518


namespace NUMINAMATH_CALUDE_train_speed_l1355_135538

/-- Given a train of length 360 meters passing a bridge of length 140 meters in 36 seconds,
    prove that its speed is 50 km/h. -/
theorem train_speed (train_length : ℝ) (bridge_length : ℝ) (time : ℝ) :
  train_length = 360 →
  bridge_length = 140 →
  time = 36 →
  (train_length + bridge_length) / time * 3.6 = 50 := by
  sorry

end NUMINAMATH_CALUDE_train_speed_l1355_135538


namespace NUMINAMATH_CALUDE_subset_of_any_implies_zero_l1355_135522

theorem subset_of_any_implies_zero (a : ℝ) : 
  (∀ S : Set ℝ, {x : ℝ | a * x = 1} ⊆ S) → a = 0 :=
by sorry

end NUMINAMATH_CALUDE_subset_of_any_implies_zero_l1355_135522


namespace NUMINAMATH_CALUDE_x_squared_plus_y_squared_l1355_135546

theorem x_squared_plus_y_squared (x y : ℝ) : 
  x * y = 8 → x^2 * y + x * y^2 + x + y = 80 → x^2 + y^2 = 5104 / 81 := by
sorry

end NUMINAMATH_CALUDE_x_squared_plus_y_squared_l1355_135546


namespace NUMINAMATH_CALUDE_function_interval_theorem_l1355_135528

theorem function_interval_theorem (a b : Real) :
  let f := fun x => -1/2 * x^2 + 13/2
  (∀ x ∈ Set.Icc a b, f x ≥ 2*a) ∧
  (∀ x ∈ Set.Icc a b, f x ≤ 2*b) ∧
  (∃ x ∈ Set.Icc a b, f x = 2*a) ∧
  (∃ x ∈ Set.Icc a b, f x = 2*b) →
  ((a = 1 ∧ b = 3) ∨ (a = -2 - Real.sqrt 17 ∧ b = 13/4)) := by
  sorry

#check function_interval_theorem

end NUMINAMATH_CALUDE_function_interval_theorem_l1355_135528


namespace NUMINAMATH_CALUDE_rational_function_property_l1355_135536

theorem rational_function_property (f : ℚ → ℝ) 
  (h : ∀ r s : ℚ, ∃ n : ℤ, f (r + s) - f r - f s = n) :
  ∃ (q : ℕ+) (p : ℤ), |f (1 / q) - p| ≤ 1 / 2012 := by
  sorry

end NUMINAMATH_CALUDE_rational_function_property_l1355_135536


namespace NUMINAMATH_CALUDE_annual_cost_difference_is_5525_l1355_135586

def annual_cost_difference : ℕ → ℕ → ℕ → ℕ → ℕ → ℕ → ℕ → ℕ → ℕ → ℕ := 
  fun clarinet_rate clarinet_hours piano_rate piano_hours violin_rate violin_hours 
      singing_rate singing_hours weeks_per_year =>
    let weeks_with_lessons := weeks_per_year - 2
    let clarinet_cost := clarinet_rate * clarinet_hours * weeks_with_lessons
    let piano_cost := (piano_rate * piano_hours * weeks_with_lessons * 9) / 10
    let violin_cost := (violin_rate * violin_hours * weeks_with_lessons * 85) / 100
    let singing_cost := singing_rate * singing_hours * weeks_with_lessons
    piano_cost + violin_cost + singing_cost - clarinet_cost

theorem annual_cost_difference_is_5525 :
  annual_cost_difference 40 3 28 5 35 2 45 1 52 = 5525 := by
  sorry

end NUMINAMATH_CALUDE_annual_cost_difference_is_5525_l1355_135586


namespace NUMINAMATH_CALUDE_roy_pens_count_l1355_135580

/-- The total number of pens Roy has -/
def total_pens (blue : ℕ) (black : ℕ) (red : ℕ) : ℕ :=
  blue + black + red

/-- The number of blue pens Roy has -/
def blue_pens : ℕ := 2

/-- The number of black pens Roy has -/
def black_pens : ℕ := 2 * blue_pens

/-- The number of red pens Roy has -/
def red_pens : ℕ := 2 * black_pens - 2

theorem roy_pens_count :
  total_pens blue_pens black_pens red_pens = 12 := by
  sorry

end NUMINAMATH_CALUDE_roy_pens_count_l1355_135580


namespace NUMINAMATH_CALUDE_exists_x_less_than_zero_l1355_135530

theorem exists_x_less_than_zero : ∃ x : ℝ, x^2 - 4*x + 3 < 0 := by
  sorry

end NUMINAMATH_CALUDE_exists_x_less_than_zero_l1355_135530


namespace NUMINAMATH_CALUDE_minimum_correct_problems_l1355_135542

def total_problems : ℕ := 25
def attempted_problems : ℕ := 21
def unanswered_problems : ℕ := total_problems - attempted_problems
def correct_points : ℕ := 7
def incorrect_points : ℤ := -1
def unanswered_points : ℕ := 2
def minimum_score : ℕ := 120

def score (correct : ℕ) : ℤ :=
  (correct * correct_points : ℤ) + 
  ((attempted_problems - correct) * incorrect_points) + 
  (unanswered_problems * unanswered_points)

theorem minimum_correct_problems : 
  ∀ x : ℕ, x ≥ 17 ↔ score x ≥ minimum_score :=
by sorry

end NUMINAMATH_CALUDE_minimum_correct_problems_l1355_135542


namespace NUMINAMATH_CALUDE_least_number_for_divisibility_l1355_135505

theorem least_number_for_divisibility (x : ℕ) : 
  (∀ y : ℕ, y < x → ¬((5432 + y) % 5 = 0 ∧ (5432 + y) % 6 = 0 ∧ (5432 + y) % 7 = 0 ∧ (5432 + y) % 11 = 0 ∧ (5432 + y) % 13 = 0)) ∧
  ((5432 + x) % 5 = 0 ∧ (5432 + x) % 6 = 0 ∧ (5432 + x) % 7 = 0 ∧ (5432 + x) % 11 = 0 ∧ (5432 + x) % 13 = 0) →
  x = 24598 :=
by sorry

end NUMINAMATH_CALUDE_least_number_for_divisibility_l1355_135505


namespace NUMINAMATH_CALUDE_plane_contains_points_and_normalized_l1355_135547

def point1 : ℝ × ℝ × ℝ := (2, -1, 3)
def point2 : ℝ × ℝ × ℝ := (0, 3, 1)
def point3 : ℝ × ℝ × ℝ := (-1, 2, 4)

def plane_equation (x y z : ℝ) := 5*x + 2*y + 3*z - 17

theorem plane_contains_points_and_normalized :
  (plane_equation point1.1 point1.2.1 point1.2.2 = 0) ∧
  (plane_equation point2.1 point2.2.1 point2.2.2 = 0) ∧
  (plane_equation point3.1 point3.2.1 point3.2.2 = 0) ∧
  (5 > 0) ∧
  (Nat.gcd (Nat.gcd (Nat.gcd 5 2) 3) 17 = 1) := by
  sorry

end NUMINAMATH_CALUDE_plane_contains_points_and_normalized_l1355_135547


namespace NUMINAMATH_CALUDE_no_real_roots_quadratic_l1355_135564

theorem no_real_roots_quadratic (k : ℝ) : 
  (∀ x : ℝ, x^2 + 2*x + k ≠ 0) ↔ k > 1 := by
  sorry

end NUMINAMATH_CALUDE_no_real_roots_quadratic_l1355_135564


namespace NUMINAMATH_CALUDE_number_division_problem_l1355_135508

theorem number_division_problem (x : ℝ) : x / 5 = 100 + x / 6 → x = 3000 := by
  sorry

end NUMINAMATH_CALUDE_number_division_problem_l1355_135508


namespace NUMINAMATH_CALUDE_ratio_is_three_halves_l1355_135557

/-- Represents a rectangular parallelepiped with dimensions a, b, and c -/
structure RectangularParallelepiped (α : Type*) [LinearOrderedField α] where
  a : α
  b : α
  c : α

/-- The ratio of the sum of squares of sides of triangle KLM to the square of the parallelepiped's diagonal -/
def triangle_to_diagonal_ratio {α : Type*} [LinearOrderedField α] (p : RectangularParallelepiped α) : α :=
  (3 : α) / 2

/-- Theorem stating that the ratio is always 3/2 for any rectangular parallelepiped -/
theorem ratio_is_three_halves {α : Type*} [LinearOrderedField α] (p : RectangularParallelepiped α) :
  triangle_to_diagonal_ratio p = (3 : α) / 2 := by
  sorry

#check ratio_is_three_halves

end NUMINAMATH_CALUDE_ratio_is_three_halves_l1355_135557


namespace NUMINAMATH_CALUDE_point_in_fourth_quadrant_l1355_135575

def fourth_quadrant (p : ℝ × ℝ) : Prop :=
  p.1 > 0 ∧ p.2 < 0

theorem point_in_fourth_quadrant : 
  fourth_quadrant (2, -Real.sqrt 3) :=
by sorry

end NUMINAMATH_CALUDE_point_in_fourth_quadrant_l1355_135575


namespace NUMINAMATH_CALUDE_simplify_expression_l1355_135552

theorem simplify_expression : (27 * (10 ^ 12)) / (9 * (10 ^ 5)) = 30000000 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l1355_135552


namespace NUMINAMATH_CALUDE_geometric_sequence_condition_l1355_135569

/-- A sequence of real numbers -/
def Sequence := ℕ → ℝ

/-- Condition for three consecutive terms of a sequence to form a geometric sequence -/
def IsGeometric (a : Sequence) (n : ℕ) : Prop :=
  ∃ r : ℝ, a (n + 1) = a n * r ∧ a (n + 2) = a (n + 1) * r

/-- The condition a_{n+1}^2 = a_n * a_{n+2} -/
def SquareMiddleCondition (a : Sequence) (n : ℕ) : Prop :=
  a (n + 1) ^ 2 = a n * a (n + 2)

theorem geometric_sequence_condition (a : Sequence) :
  (∀ n : ℕ, IsGeometric a n → SquareMiddleCondition a n) ∧
  ¬(∀ n : ℕ, SquareMiddleCondition a n → IsGeometric a n) :=
sorry

end NUMINAMATH_CALUDE_geometric_sequence_condition_l1355_135569


namespace NUMINAMATH_CALUDE_math_score_proof_l1355_135576

def science : ℕ := 65
def social_studies : ℕ := 82
def english : ℕ := 47
def biology : ℕ := 85
def average : ℕ := 71
def total_subjects : ℕ := 5

theorem math_score_proof :
  ∃ (math : ℕ), 
    (science + social_studies + english + biology + math) / total_subjects = average ∧
    math = 76 := by
  sorry

end NUMINAMATH_CALUDE_math_score_proof_l1355_135576


namespace NUMINAMATH_CALUDE_linear_function_intersection_l1355_135570

theorem linear_function_intersection (k : ℝ) : 
  (∃ x : ℝ, k * x + 2 = 0 ∧ abs x = 4) → k = 1/2 ∨ k = -1/2 := by
  sorry

end NUMINAMATH_CALUDE_linear_function_intersection_l1355_135570


namespace NUMINAMATH_CALUDE_upgraded_sensor_fraction_l1355_135587

/-- Represents a satellite with modular units and sensors. -/
structure Satellite where
  units : ℕ
  non_upgraded_per_unit : ℕ
  total_upgraded : ℕ
  non_upgraded_ratio : non_upgraded_per_unit = total_upgraded / 4

/-- The fraction of upgraded sensors on the satellite is 1/7. -/
theorem upgraded_sensor_fraction (s : Satellite) (h : s.units = 24) :
  s.total_upgraded / (s.units * s.non_upgraded_per_unit + s.total_upgraded) = 1 / 7 := by
  sorry

end NUMINAMATH_CALUDE_upgraded_sensor_fraction_l1355_135587


namespace NUMINAMATH_CALUDE_no_three_digit_integers_with_five_units_divisible_by_ten_l1355_135572

theorem no_three_digit_integers_with_five_units_divisible_by_ten :
  ¬ ∃ n : ℕ, 
    100 ≤ n ∧ n < 1000 ∧  -- three-digit positive integer
    n % 10 = 5 ∧          -- 5 in the units place
    n % 10 = 0            -- divisible by 10
  := by sorry

end NUMINAMATH_CALUDE_no_three_digit_integers_with_five_units_divisible_by_ten_l1355_135572


namespace NUMINAMATH_CALUDE_square_diagonals_equal_l1355_135515

-- Define the basic shapes
class Rectangle where
  diagonals_equal : Bool

class Square extends Rectangle

-- Define the properties
axiom rectangle_diagonals_equal : ∀ (r : Rectangle), r.diagonals_equal = true

-- Theorem to prove
theorem square_diagonals_equal (s : Square) : s.diagonals_equal = true := by
  sorry

end NUMINAMATH_CALUDE_square_diagonals_equal_l1355_135515


namespace NUMINAMATH_CALUDE_mixture_problem_l1355_135581

/-- Proves that the percentage of the first solution is 30% given the conditions of the mixture problem. -/
theorem mixture_problem (total_volume : ℝ) (result_percentage : ℝ) (second_solution_percentage : ℝ)
  (first_solution_volume : ℝ) (second_solution_volume : ℝ)
  (h1 : total_volume = 40)
  (h2 : result_percentage = 45)
  (h3 : second_solution_percentage = 80)
  (h4 : first_solution_volume = 28)
  (h5 : second_solution_volume = 12)
  (h6 : total_volume = first_solution_volume + second_solution_volume)
  (h7 : result_percentage / 100 * total_volume =
        (first_solution_percentage / 100 * first_solution_volume) +
        (second_solution_percentage / 100 * second_solution_volume)) :
  first_solution_percentage = 30 :=
sorry

end NUMINAMATH_CALUDE_mixture_problem_l1355_135581


namespace NUMINAMATH_CALUDE_subtract_p_q_equals_five_twentyfourths_l1355_135589

theorem subtract_p_q_equals_five_twentyfourths 
  (p q : ℚ) 
  (hp : 3 / p = 8) 
  (hq : 3 / q = 18) : 
  p - q = 5 / 24 := by
  sorry

end NUMINAMATH_CALUDE_subtract_p_q_equals_five_twentyfourths_l1355_135589


namespace NUMINAMATH_CALUDE_xyz_sign_sum_l1355_135553

theorem xyz_sign_sum (x y z : ℝ) (h : x * y * z / |x * y * z| = 1) :
  |x| / x + y / |y| + |z| / z = -1 ∨ |x| / x + y / |y| + |z| / z = 3 :=
by sorry

end NUMINAMATH_CALUDE_xyz_sign_sum_l1355_135553


namespace NUMINAMATH_CALUDE_quadratic_real_solutions_l1355_135593

theorem quadratic_real_solutions (m : ℝ) :
  (∃ x : ℝ, m * x^2 - 4 * x + 1 = 0) → m ≤ 4 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_real_solutions_l1355_135593


namespace NUMINAMATH_CALUDE_vegetable_cost_l1355_135585

theorem vegetable_cost (beef_weight : ℝ) (vegetable_weight : ℝ) (total_cost : ℝ) :
  beef_weight = 4 →
  vegetable_weight = 6 →
  total_cost = 36 →
  ∃ (v : ℝ), v * vegetable_weight + 3 * v * beef_weight = total_cost ∧ v = 2 :=
by sorry

end NUMINAMATH_CALUDE_vegetable_cost_l1355_135585


namespace NUMINAMATH_CALUDE_circle_area_ratio_l1355_135573

theorem circle_area_ratio (diameter_R diameter_S area_R area_S : ℝ) :
  diameter_R = 0.6 * diameter_S →
  area_R = π * (diameter_R / 2)^2 →
  area_S = π * (diameter_S / 2)^2 →
  area_R / area_S = 0.36 :=
by
  sorry

end NUMINAMATH_CALUDE_circle_area_ratio_l1355_135573


namespace NUMINAMATH_CALUDE_multiple_of_numbers_l1355_135577

theorem multiple_of_numbers (s l k : ℤ) : 
  s = 18 →                  -- The smaller number is 18
  l = k * s - 3 →           -- One number is 3 less than a multiple of the other
  s + l = 51 →              -- The sum of the two numbers is 51
  k = 2 :=                  -- The multiple is 2
by sorry

end NUMINAMATH_CALUDE_multiple_of_numbers_l1355_135577


namespace NUMINAMATH_CALUDE_geometric_sum_problem_l1355_135520

/-- Sum of the first n terms of a geometric sequence -/
def geometric_sum (a : ℚ) (r : ℚ) (n : ℕ) : ℚ :=
  a * (1 - r^n) / (1 - r)

/-- The problem statement -/
theorem geometric_sum_problem :
  let a : ℚ := 1/3
  let r : ℚ := 1/4
  let n : ℕ := 7
  geometric_sum a r n = 16383/12288 := by
sorry

end NUMINAMATH_CALUDE_geometric_sum_problem_l1355_135520


namespace NUMINAMATH_CALUDE_circle_distance_to_line_l1355_135501

/-- Given a circle (x-a)^2 + (y-a)^2 = 8 and the shortest distance from any point on the circle
    to the line y = -x is √2, prove that a = ±3 -/
theorem circle_distance_to_line (a : ℝ) :
  (∀ x y : ℝ, (x - a)^2 + (y - a)^2 = 8 →
    (∃ d : ℝ, d = Real.sqrt 2 ∧
      ∀ d' : ℝ, d' ≥ 0 → (x + y) / Real.sqrt 2 ≤ d')) →
  a = 3 ∨ a = -3 := by
  sorry

end NUMINAMATH_CALUDE_circle_distance_to_line_l1355_135501


namespace NUMINAMATH_CALUDE_percentage_difference_l1355_135592

theorem percentage_difference (n : ℝ) (h : n = 140) : (4/5 * n) - (65/100 * n) = 21 := by
  sorry

end NUMINAMATH_CALUDE_percentage_difference_l1355_135592


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l1355_135563

-- Define the sets A and B
def A : Set (ℝ × ℝ) := {p : ℝ × ℝ | 2 * p.1 - p.2 = 0}
def B : Set (ℝ × ℝ) := {p : ℝ × ℝ | 3 * p.1 + p.2 = 0}

-- State the theorem
theorem intersection_of_A_and_B : A ∩ B = {(0, 0)} := by sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l1355_135563


namespace NUMINAMATH_CALUDE_highest_class_strength_l1355_135535

theorem highest_class_strength (total : ℕ) (g1 g2 g3 : ℕ) : 
  total = 333 →
  g1 + g2 + g3 = total →
  5 * g1 = 3 * g2 →
  11 * g2 = 7 * g3 →
  g1 ≤ g2 ∧ g2 ≤ g3 →
  g3 = 165 := by
sorry

end NUMINAMATH_CALUDE_highest_class_strength_l1355_135535


namespace NUMINAMATH_CALUDE_problem_solution_l1355_135574

/-- The surface area of an open box formed by removing square corners from a rectangular sheet. -/
def boxSurfaceArea (length width corner_size : ℕ) : ℕ :=
  length * width - 4 * (corner_size * corner_size)

/-- Theorem stating that the surface area of the box described in the problem is 500 square units. -/
theorem problem_solution :
  boxSurfaceArea 30 20 5 = 500 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l1355_135574


namespace NUMINAMATH_CALUDE_parallelepiped_dimensions_l1355_135566

theorem parallelepiped_dimensions (n : ℕ) (h1 : n > 6) : 
  (n - 2) * (n - 4) * (n - 6) = (2 * n * (n - 2) * (n - 4)) / 3 → n = 18 :=
by sorry

end NUMINAMATH_CALUDE_parallelepiped_dimensions_l1355_135566


namespace NUMINAMATH_CALUDE_solve_system_for_w_l1355_135598

theorem solve_system_for_w (x y z w : ℝ) 
  (eq1 : 2*x + y + z + w = 1)
  (eq2 : x + 3*y + z + w = 2)
  (eq3 : x + y + 4*z + w = 3)
  (eq4 : x + y + z + 5*w = 25) : 
  w = 11/2 := by sorry

end NUMINAMATH_CALUDE_solve_system_for_w_l1355_135598


namespace NUMINAMATH_CALUDE_unique_triangle_l1355_135548

/-- A triangle with integer side lengths a, b, c, where a ≤ b ≤ c -/
structure IntegerTriangle where
  a : ℕ+
  b : ℕ+
  c : ℕ+
  a_le_b : a ≤ b
  b_le_c : b ≤ c

/-- The set of all integer triangles with perimeter 10 satisfying the triangle inequality -/
def ValidTriangles : Set IntegerTriangle :=
  {t : IntegerTriangle | t.a + t.b + t.c = 10 ∧ t.a + t.b > t.c}

theorem unique_triangle : ∃! t : IntegerTriangle, t ∈ ValidTriangles :=
sorry

end NUMINAMATH_CALUDE_unique_triangle_l1355_135548


namespace NUMINAMATH_CALUDE_planes_distance_zero_l1355_135502

-- Define the planes
def plane1 (x y z : ℝ) : Prop := x + 2*y - z = 3
def plane2 (x y z : ℝ) : Prop := 2*x + 4*y - 2*z = 6

-- Define the distance function between two planes
noncomputable def distance_between_planes (p1 p2 : ℝ → ℝ → ℝ → Prop) : ℝ := sorry

-- Theorem statement
theorem planes_distance_zero :
  distance_between_planes plane1 plane2 = 0 := by sorry

end NUMINAMATH_CALUDE_planes_distance_zero_l1355_135502


namespace NUMINAMATH_CALUDE_max_vector_difference_l1355_135590

theorem max_vector_difference (x : ℝ) : 
  let m : ℝ × ℝ := (Real.cos (x / 2), Real.sin (x / 2))
  let n : ℝ × ℝ := (-Real.sqrt 3, 1)
  (∀ y : ℝ, ‖(m.1 - n.1, m.2 - n.2)‖ ≤ 3) ∧ 
  (∃ z : ℝ, ‖(Real.cos (z / 2) - (-Real.sqrt 3), Real.sin (z / 2) - 1)‖ = 3) := by
sorry

end NUMINAMATH_CALUDE_max_vector_difference_l1355_135590
