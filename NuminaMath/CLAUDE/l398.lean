import Mathlib

namespace NUMINAMATH_CALUDE_units_digit_sum_of_powers_l398_39876

theorem units_digit_sum_of_powers : (24^4 + 42^4) % 10 = 2 := by
  sorry

end NUMINAMATH_CALUDE_units_digit_sum_of_powers_l398_39876


namespace NUMINAMATH_CALUDE_square_not_always_positive_l398_39816

theorem square_not_always_positive : ¬ (∀ x : ℝ, x^2 > 0) := by
  sorry

end NUMINAMATH_CALUDE_square_not_always_positive_l398_39816


namespace NUMINAMATH_CALUDE_complex_number_location_l398_39857

theorem complex_number_location :
  let z : ℂ := (1 + Complex.I) / Complex.I
  (z.re > 0) ∧ (z.im < 0) := by sorry

end NUMINAMATH_CALUDE_complex_number_location_l398_39857


namespace NUMINAMATH_CALUDE_complex_modulus_problem_l398_39896

theorem complex_modulus_problem (z : ℂ) (h : 3 + z * Complex.I = z - 3 * Complex.I) : 
  Complex.abs z = 3 := by sorry

end NUMINAMATH_CALUDE_complex_modulus_problem_l398_39896


namespace NUMINAMATH_CALUDE_union_of_sets_l398_39821

theorem union_of_sets : 
  let A : Set Int := {-2, 0}
  let B : Set Int := {-2, 3}
  A ∪ B = {-2, 0, 3} := by
sorry

end NUMINAMATH_CALUDE_union_of_sets_l398_39821


namespace NUMINAMATH_CALUDE_grid_ball_probability_l398_39813

theorem grid_ball_probability
  (a b c r : ℝ)
  (h_a_pos : 0 < a)
  (h_b_pos : 0 < b)
  (h_a_gt_b : a > b)
  (h_r_lt_b_half : r < b / 2)
  (h_strip_width : 2 * c = Real.sqrt ((a + b)^2 / 4 + a * b) - (a + b) / 2)
  : (a - 2 * r) * (b - 2 * r) / ((a + 2 * c) * (b + 2 * c)) ≤ 1 / 2 :=
by sorry

end NUMINAMATH_CALUDE_grid_ball_probability_l398_39813


namespace NUMINAMATH_CALUDE_paint_calculation_l398_39832

theorem paint_calculation (initial_paint : ℚ) : 
  (1 / 4 * initial_paint + 1 / 6 * (3 / 4 * initial_paint) = 135) → 
  ⌈initial_paint⌉ = 463 := by
  sorry

end NUMINAMATH_CALUDE_paint_calculation_l398_39832


namespace NUMINAMATH_CALUDE_third_quadrant_condition_l398_39807

-- Define the complex number z
def z (a : ℝ) : ℂ := Complex.mk (a - 1) (a + 1)

-- Define the condition for a point to be in the third quadrant
def in_third_quadrant (w : ℂ) : Prop := w.re < 0 ∧ w.im < 0

-- Theorem statement
theorem third_quadrant_condition (a : ℝ) :
  in_third_quadrant (z a) ↔ a < -1 := by
  sorry

end NUMINAMATH_CALUDE_third_quadrant_condition_l398_39807


namespace NUMINAMATH_CALUDE_arithmetic_geometric_mean_difference_bound_l398_39891

theorem arithmetic_geometric_mean_difference_bound 
  (a b : ℝ) 
  (ha : 0 < a) 
  (hb : 0 < b) 
  (hab : a < b) : 
  (a + b) / 2 - Real.sqrt (a * b) < (b - a)^2 / (8 * a) := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_geometric_mean_difference_bound_l398_39891


namespace NUMINAMATH_CALUDE_additional_candles_l398_39833

/-- 
Given:
- initial_candles: The initial number of candles on Molly's birthday cake
- current_age: Molly's current age
Prove that the number of additional candles is equal to current_age - initial_candles
-/
theorem additional_candles (initial_candles current_age : ℕ) :
  initial_candles = 14 →
  current_age = 20 →
  current_age - initial_candles = 6 := by
  sorry

end NUMINAMATH_CALUDE_additional_candles_l398_39833


namespace NUMINAMATH_CALUDE_line_hyperbola_intersection_range_l398_39814

/-- The range of k for which the line y = kx intersects the hyperbola x^2/9 - y^2/4 = 1 -/
theorem line_hyperbola_intersection_range :
  ∀ k : ℝ, 
  (∃ x y : ℝ, y = k * x ∧ x^2 / 9 - y^2 / 4 = 1) ↔ 
  -2/3 < k ∧ k < 2/3 :=
by sorry

end NUMINAMATH_CALUDE_line_hyperbola_intersection_range_l398_39814


namespace NUMINAMATH_CALUDE_spaghetti_to_manicotti_ratio_l398_39804

/-- The ratio of students who preferred spaghetti to those who preferred manicotti -/
def pasta_preference_ratio (spaghetti_count manicotti_count : ℕ) : ℚ :=
  spaghetti_count / manicotti_count

/-- The total number of students surveyed -/
def total_students : ℕ := 650

/-- The theorem stating the ratio of spaghetti preference to manicotti preference -/
theorem spaghetti_to_manicotti_ratio : 
  pasta_preference_ratio 250 100 = 5/2 := by
  sorry

end NUMINAMATH_CALUDE_spaghetti_to_manicotti_ratio_l398_39804


namespace NUMINAMATH_CALUDE_quadratic_equation_solutions_l398_39844

theorem quadratic_equation_solutions :
  ∀ x : ℝ, x^2 - 3*x = 0 ↔ x = 0 ∨ x = 3 := by sorry

end NUMINAMATH_CALUDE_quadratic_equation_solutions_l398_39844


namespace NUMINAMATH_CALUDE_divisors_540_multiple_of_two_l398_39881

/-- The number of positive divisors of 540 that are multiples of 2 -/
def divisors_multiple_of_two (n : ℕ) : ℕ :=
  (Finset.filter (λ d => d > 0 ∧ n % d = 0 ∧ d % 2 = 0) (Finset.range (n + 1))).card

/-- Theorem stating that the number of positive divisors of 540 that are multiples of 2 is 16 -/
theorem divisors_540_multiple_of_two :
  divisors_multiple_of_two 540 = 16 := by
  sorry

end NUMINAMATH_CALUDE_divisors_540_multiple_of_two_l398_39881


namespace NUMINAMATH_CALUDE_number_division_l398_39873

theorem number_division (x : ℚ) : x / 4 = 12 → x / 3 = 16 := by
  sorry

end NUMINAMATH_CALUDE_number_division_l398_39873


namespace NUMINAMATH_CALUDE_correct_meal_probability_l398_39836

def number_of_people : ℕ := 12
def pasta_orders : ℕ := 5
def salad_orders : ℕ := 7

theorem correct_meal_probability : 
  let total_arrangements := Nat.factorial number_of_people
  let favorable_outcomes := 157410
  (favorable_outcomes : ℚ) / total_arrangements = 157410 / 479001600 := by
  sorry

end NUMINAMATH_CALUDE_correct_meal_probability_l398_39836


namespace NUMINAMATH_CALUDE_sixth_power_sum_l398_39861

theorem sixth_power_sum (x : ℝ) (h : x + 1/x = 5) : x^6 + 1/x^6 = 12098 := by
  sorry

end NUMINAMATH_CALUDE_sixth_power_sum_l398_39861


namespace NUMINAMATH_CALUDE_trig_function_relation_l398_39856

theorem trig_function_relation (x : ℝ) (n : ℤ) (f : ℝ → ℝ)
  (h : f (Real.sin x) = Real.sin ((4 * ↑n + 1) * x)) :
  f (Real.cos x) = Real.cos ((4 * ↑n + 1) * x) := by
  sorry

end NUMINAMATH_CALUDE_trig_function_relation_l398_39856


namespace NUMINAMATH_CALUDE_books_movies_difference_l398_39820

def books_read : ℕ := 17
def movies_watched : ℕ := 21

theorem books_movies_difference :
  (books_read : ℤ) - movies_watched = -4 :=
sorry

end NUMINAMATH_CALUDE_books_movies_difference_l398_39820


namespace NUMINAMATH_CALUDE_divisibility_equivalence_l398_39829

theorem divisibility_equivalence (m n : ℕ+) :
  (6 * m.val ∣ (2 * m.val + 3)^n.val + 1) ↔ (4 * m.val ∣ 3^n.val + 1) := by
  sorry

end NUMINAMATH_CALUDE_divisibility_equivalence_l398_39829


namespace NUMINAMATH_CALUDE_first_woman_work_time_l398_39882

/-- Represents the wall-building scenario with women joining at intervals -/
structure WallBuilding where
  /-- Total time to build the wall if all women worked together -/
  totalTime : ℝ
  /-- Number of women -/
  numWomen : ℕ
  /-- Time interval between each woman joining -/
  joinInterval : ℝ
  /-- Time all women work together -/
  allWorkTime : ℝ

/-- The first woman works 5 times as long as the last woman -/
def firstLastRatio (w : WallBuilding) : Prop :=
  w.joinInterval * (w.numWomen - 1) + w.allWorkTime = 5 * w.allWorkTime

/-- The total work done is equivalent to all women working for the total time -/
def totalWorkEquivalence (w : WallBuilding) : Prop :=
  (w.joinInterval * (w.numWomen - 1) / 2 + w.allWorkTime) * w.numWomen = w.totalTime * w.numWomen

/-- Main theorem: The first woman works for 75 hours -/
theorem first_woman_work_time (w : WallBuilding) 
    (h1 : w.totalTime = 45)
    (h2 : firstLastRatio w)
    (h3 : totalWorkEquivalence w) : 
  w.joinInterval * (w.numWomen - 1) + w.allWorkTime = 75 := by
  sorry

#check first_woman_work_time

end NUMINAMATH_CALUDE_first_woman_work_time_l398_39882


namespace NUMINAMATH_CALUDE_tank_dimension_l398_39889

/-- Given a rectangular tank with dimensions 4, x, and 2 feet, 
    if the total cost to cover its surface with insulation at $20 per square foot is $1520, 
    then x = 5 feet. -/
theorem tank_dimension (x : ℝ) : 
  x > 0 →  -- Ensuring positive dimension
  (12 * x + 16) * 20 = 1520 → 
  x = 5 := by
sorry

end NUMINAMATH_CALUDE_tank_dimension_l398_39889


namespace NUMINAMATH_CALUDE_proportional_function_expression_l398_39880

/-- Given a proportional function y = kx (k ≠ 0), if y = 6 when x = 4, 
    then the function can be expressed as y = (3/2)x -/
theorem proportional_function_expression (k : ℝ) (h1 : k ≠ 0) : 
  (∀ x y, y = k * x) → (6 : ℝ) = k * 4 → 
  ∀ x y, y = k * x ↔ y = (3/2) * x := by
  sorry

end NUMINAMATH_CALUDE_proportional_function_expression_l398_39880


namespace NUMINAMATH_CALUDE_prob_three_dice_sum_18_l398_39865

/-- The probability of rolling a specific number on a standard die -/
def prob_single_die : ℚ := 1 / 6

/-- The number of faces on a standard die -/
def dice_faces : ℕ := 6

/-- The sum we're looking for -/
def target_sum : ℕ := 18

/-- The number of dice rolled -/
def num_dice : ℕ := 3

theorem prob_three_dice_sum_18 : 
  (prob_single_die ^ num_dice : ℚ) = 1 / 216 := by
  sorry

end NUMINAMATH_CALUDE_prob_three_dice_sum_18_l398_39865


namespace NUMINAMATH_CALUDE_quotient_with_negative_remainder_l398_39834

theorem quotient_with_negative_remainder
  (dividend : ℤ)
  (divisor : ℤ)
  (remainder : ℤ)
  (h1 : dividend = 474232)
  (h2 : divisor = 800)
  (h3 : remainder = -968)
  (h4 : dividend = divisor * (dividend / divisor) + remainder) :
  dividend / divisor = 594 := by
  sorry

end NUMINAMATH_CALUDE_quotient_with_negative_remainder_l398_39834


namespace NUMINAMATH_CALUDE_select_cubes_eq_31_l398_39825

/-- The number of ways to select 10 cubes from a set of 7 red cubes, 3 blue cubes, and 9 green cubes -/
def select_cubes : ℕ :=
  let red_cubes := 7
  let blue_cubes := 3
  let green_cubes := 9
  let total_selected := 10
  (Finset.range (red_cubes + 1)).sum (λ r => 
    (Finset.range (blue_cubes + 1)).sum (λ b => 
      let g := total_selected - r - b
      if g ≥ 0 ∧ g ≤ green_cubes then 1 else 0
    )
  )

theorem select_cubes_eq_31 : select_cubes = 31 := by sorry

end NUMINAMATH_CALUDE_select_cubes_eq_31_l398_39825


namespace NUMINAMATH_CALUDE_equation_solution_l398_39828

theorem equation_solution : ∃ x : ℚ, (3*x + 5*x = 600 - (4*x + 6*x)) ∧ x = 100/3 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l398_39828


namespace NUMINAMATH_CALUDE_total_salaries_l398_39870

/-- The total amount of A and B's salaries given the specified conditions -/
theorem total_salaries (A_salary B_salary : ℝ) : 
  A_salary = 1500 →
  A_salary * 0.05 = B_salary * 0.15 →
  A_salary + B_salary = 2000 := by
  sorry

end NUMINAMATH_CALUDE_total_salaries_l398_39870


namespace NUMINAMATH_CALUDE_circus_tickets_cost_l398_39802

/-- Given the cost per ticket and the number of tickets bought, 
    calculate the total amount spent on tickets. -/
def total_spent (cost_per_ticket : ℕ) (num_tickets : ℕ) : ℕ :=
  cost_per_ticket * num_tickets

/-- Theorem: If each ticket costs 44 dollars and 7 tickets are bought,
    the total amount spent is 308 dollars. -/
theorem circus_tickets_cost :
  let cost_per_ticket : ℕ := 44
  let num_tickets : ℕ := 7
  total_spent cost_per_ticket num_tickets = 308 := by
  sorry

end NUMINAMATH_CALUDE_circus_tickets_cost_l398_39802


namespace NUMINAMATH_CALUDE_walking_speed_is_10_l398_39879

/-- Represents the walking speed of person A in km/h -/
def walking_speed : ℝ := 10

/-- Represents the cycling speed of person B in km/h -/
def cycling_speed : ℝ := 20

/-- Represents the time difference in hours between when A starts walking and B starts cycling -/
def time_difference : ℝ := 6

/-- Represents the distance in km at which B catches up with A -/
def catch_up_distance : ℝ := 120

theorem walking_speed_is_10 : 
  walking_speed = 10 ∧ 
  cycling_speed = 20 ∧
  time_difference = 6 ∧
  catch_up_distance = 120 ∧
  ∃ t : ℝ, t > time_difference ∧ 
        walking_speed * t = catch_up_distance ∧ 
        cycling_speed * (t - time_difference) = catch_up_distance :=
by sorry

end NUMINAMATH_CALUDE_walking_speed_is_10_l398_39879


namespace NUMINAMATH_CALUDE_db_length_determined_l398_39823

-- Define a triangle ABC
structure Triangle :=
  (A B C : ℝ × ℝ)

-- Define the altitude CD to AB
def altitudeCD (t : Triangle) (D : ℝ × ℝ) : Prop :=
  let (xA, yA) := t.A
  let (xB, yB) := t.B
  let (xC, yC) := t.C
  let (xD, yD) := D
  (xD - xA) * (xB - xA) + (yD - yA) * (yB - yA) = 0 ∧
  (xC - xD) * (xB - xA) + (yC - yD) * (yB - yA) = 0

-- Define the altitude AE to BC
def altitudeAE (t : Triangle) (E : ℝ × ℝ) : Prop :=
  let (xA, yA) := t.A
  let (xB, yB) := t.B
  let (xC, yC) := t.C
  let (xE, yE) := E
  (xE - xB) * (xC - xB) + (yE - yB) * (yC - yB) = 0 ∧
  (xA - xE) * (xC - xB) + (yA - yE) * (yC - yB) = 0

-- Define the lengths of AB, CD, and AE
def lengthAB (t : Triangle) : ℝ := sorry
def lengthCD (t : Triangle) (D : ℝ × ℝ) : ℝ := sorry
def lengthAE (t : Triangle) (E : ℝ × ℝ) : ℝ := sorry

-- Define the length of DB
def lengthDB (t : Triangle) (D : ℝ × ℝ) : ℝ := sorry

-- Theorem statement
theorem db_length_determined (t : Triangle) (D E : ℝ × ℝ) :
  altitudeCD t D → altitudeAE t E →
  ∃! db : ℝ, db = lengthDB t D := by sorry

end NUMINAMATH_CALUDE_db_length_determined_l398_39823


namespace NUMINAMATH_CALUDE_ellipse_sum_l398_39890

/-- For an ellipse with center (h, k), semi-major axis length a, and semi-minor axis length b,
    prove that h + k + a + b = 4 when the center is (3, -5), a = 4, and b = 2. -/
theorem ellipse_sum (h k a b : ℝ) : 
  h = 3 → k = -5 → a = 4 → b = 2 → h + k + a + b = 4 := by
  sorry

end NUMINAMATH_CALUDE_ellipse_sum_l398_39890


namespace NUMINAMATH_CALUDE_sandwich_combinations_l398_39831

theorem sandwich_combinations (meat_types : ℕ) (cheese_types : ℕ) (condiment_types : ℕ) :
  meat_types = 12 →
  cheese_types = 11 →
  condiment_types = 5 →
  (meat_types * Nat.choose cheese_types 2 * (condiment_types + 1)) = 3960 :=
by sorry

end NUMINAMATH_CALUDE_sandwich_combinations_l398_39831


namespace NUMINAMATH_CALUDE_jack_morning_emails_l398_39862

/-- The number of emails Jack received in the morning -/
def morning_emails : ℕ := sorry

/-- The number of emails Jack received in the evening -/
def evening_emails : ℕ := 8

/-- The total number of emails Jack received in the morning and evening -/
def total_morning_evening : ℕ := 11

/-- Theorem stating that Jack received 3 emails in the morning -/
theorem jack_morning_emails :
  morning_emails = 3 :=
by sorry

end NUMINAMATH_CALUDE_jack_morning_emails_l398_39862


namespace NUMINAMATH_CALUDE_min_value_expression_l398_39878

theorem min_value_expression (x y : ℝ) : 
  (15 - x) * (8 - x) * (15 + x) * (8 + x) + y^2 ≥ -208.25 := by
  sorry

end NUMINAMATH_CALUDE_min_value_expression_l398_39878


namespace NUMINAMATH_CALUDE_triangle_is_right_angled_l398_39824

/-- A triangle is right-angled if the square of its longest side equals the sum of squares of the other two sides. -/
def IsRightAngled (a b c : ℝ) : Prop :=
  (a ≥ b ∧ a ≥ c ∧ a^2 = b^2 + c^2) ∨
  (b ≥ a ∧ b ≥ c ∧ b^2 = a^2 + c^2) ∨
  (c ≥ a ∧ c ≥ b ∧ c^2 = a^2 + b^2)

/-- Given three real numbers a, b, and c that satisfy the equation
    a^2 + b^2 + c^2 - 12a - 16b - 20c + 200 = 0,
    prove that they form a right-angled triangle. -/
theorem triangle_is_right_angled (a b c : ℝ)
  (h : a^2 + b^2 + c^2 - 12*a - 16*b - 20*c + 200 = 0) :
  IsRightAngled a b c :=
sorry

end NUMINAMATH_CALUDE_triangle_is_right_angled_l398_39824


namespace NUMINAMATH_CALUDE_dantes_age_l398_39845

theorem dantes_age (cooper dante maria : ℕ) : 
  cooper + dante + maria = 31 →
  cooper = dante / 2 →
  maria = dante + 1 →
  dante = 12 := by
sorry

end NUMINAMATH_CALUDE_dantes_age_l398_39845


namespace NUMINAMATH_CALUDE_raine_initial_payment_l398_39871

/-- The price of a bracelet in dollars -/
def bracelet_price : ℕ := 15

/-- The price of a gold heart necklace in dollars -/
def necklace_price : ℕ := 10

/-- The price of a personalized coffee mug in dollars -/
def mug_price : ℕ := 20

/-- The number of bracelets Raine bought -/
def bracelets_bought : ℕ := 3

/-- The number of gold heart necklaces Raine bought -/
def necklaces_bought : ℕ := 2

/-- The number of personalized coffee mugs Raine bought -/
def mugs_bought : ℕ := 1

/-- The amount of change Raine received in dollars -/
def change_received : ℕ := 15

/-- The theorem stating the amount Raine initially gave -/
theorem raine_initial_payment : 
  bracelet_price * bracelets_bought + 
  necklace_price * necklaces_bought + 
  mug_price * mugs_bought + 
  change_received = 100 := by
  sorry

end NUMINAMATH_CALUDE_raine_initial_payment_l398_39871


namespace NUMINAMATH_CALUDE_streetlight_distance_l398_39895

/-- The distance between streetlights in meters -/
def interval : ℝ := 60

/-- The number of streetlights -/
def num_streetlights : ℕ := 45

/-- The distance from the first to the last streetlight in kilometers -/
def distance_km : ℝ := 2.64

theorem streetlight_distance :
  (interval * (num_streetlights - 1)) / 1000 = distance_km := by
  sorry

end NUMINAMATH_CALUDE_streetlight_distance_l398_39895


namespace NUMINAMATH_CALUDE_f_greater_than_f_prime_plus_three_halves_l398_39869

open Real

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * (x - log x) + (2 * x - 1) / x^2

theorem f_greater_than_f_prime_plus_three_halves (x : ℝ) (hx : x ∈ Set.Icc 1 2) :
  f 1 x > (deriv (f 1)) x + 3/2 := by
  sorry

end NUMINAMATH_CALUDE_f_greater_than_f_prime_plus_three_halves_l398_39869


namespace NUMINAMATH_CALUDE_mary_income_90_percent_of_juan_l398_39817

/-- Represents the income of an individual -/
structure Income where
  amount : ℝ
  amount_pos : amount > 0

/-- The relationship between incomes of Mary, Tim, Juan, Sophia, and Alex -/
structure IncomeRelationship where
  alex : Income
  sophia : Income
  juan : Income
  tim : Income
  mary : Income
  sophia_alex : sophia.amount = 1.25 * alex.amount
  juan_sophia : juan.amount = 0.7 * sophia.amount
  tim_juan : tim.amount = 0.6 * juan.amount
  mary_tim : mary.amount = 1.5 * tim.amount

/-- Theorem stating that Mary's income is 90% of Juan's income -/
theorem mary_income_90_percent_of_juan (r : IncomeRelationship) : 
  r.mary.amount = 0.9 * r.juan.amount := by sorry

end NUMINAMATH_CALUDE_mary_income_90_percent_of_juan_l398_39817


namespace NUMINAMATH_CALUDE_evaluate_expression_l398_39826

theorem evaluate_expression (x : ℤ) (h : x = -2) : 5 * x + 7 = -3 := by
  sorry

end NUMINAMATH_CALUDE_evaluate_expression_l398_39826


namespace NUMINAMATH_CALUDE_triangle_arithmetic_angle_sequence_side_relation_l398_39830

open Real

/-- Given a triangle ABC with sides a, b, c and angles A, B, C (in radians),
    where A, B, C form an arithmetic sequence, 
    prove that 1/(a+b) + 1/(b+c) = 3/(a+b+c) -/
theorem triangle_arithmetic_angle_sequence_side_relation 
  (a b c A B C : ℝ) 
  (h_triangle : 0 < a ∧ 0 < b ∧ 0 < c)
  (h_angles : 0 < A ∧ 0 < B ∧ 0 < C)
  (h_sum_angles : A + B + C = π)
  (h_arithmetic_seq : ∃ d : ℝ, B = A + d ∧ C = B + d) :
  1 / (a + b) + 1 / (b + c) = 3 / (a + b + c) := by
  sorry

end NUMINAMATH_CALUDE_triangle_arithmetic_angle_sequence_side_relation_l398_39830


namespace NUMINAMATH_CALUDE_extremum_conditions_another_extremum_l398_39846

/-- The function f with parameters a and b -/
def f (a b : ℝ) (x : ℝ) : ℝ := x^3 + a*x^2 + b*x + a^2

/-- The derivative of f with respect to x -/
def f' (a b : ℝ) (x : ℝ) : ℝ := 3*x^2 + 2*a*x + b

theorem extremum_conditions (a b : ℝ) : 
  (f a b (-1) = 8 ∧ f' a b (-1) = 0) → (a = -2 ∧ b = -7) :=
by sorry

theorem another_extremum : 
  f (-2) (-7) (7/3) = -284/27 ∧ 
  (∀ x : ℝ, x ≠ -1 ∧ x ≠ 7/3 → |f (-2) (-7) x| ≤ |f (-2) (-7) (7/3)|) :=
by sorry

end NUMINAMATH_CALUDE_extremum_conditions_another_extremum_l398_39846


namespace NUMINAMATH_CALUDE_roger_trips_l398_39884

def trays_per_trip : ℕ := 4
def total_trays : ℕ := 12

theorem roger_trips : (total_trays + trays_per_trip - 1) / trays_per_trip = 3 := by
  sorry

end NUMINAMATH_CALUDE_roger_trips_l398_39884


namespace NUMINAMATH_CALUDE_special_rectangle_side_lengths_l398_39838

/-- A rectangle with specific properties -/
structure SpecialRectangle where
  AB : ℝ  -- Length of side AB
  BC : ℝ  -- Length of side BC
  ratio_condition : AB / BC = 7 / 5
  square_area : ℝ  -- Area of the common square
  square_area_value : square_area = 72

/-- Theorem stating the side lengths of the special rectangle -/
theorem special_rectangle_side_lengths (rect : SpecialRectangle) : 
  rect.AB = 42 ∧ rect.BC = 30 := by
  sorry

end NUMINAMATH_CALUDE_special_rectangle_side_lengths_l398_39838


namespace NUMINAMATH_CALUDE_common_roots_product_l398_39853

/-- Given two cubic equations with two common roots, prove that the product of these common roots is 10 * ∛2 -/
theorem common_roots_product (C D : ℝ) : 
  ∃ (u v w t : ℝ), 
    (u^3 + C*u^2 + 20 = 0) ∧ 
    (v^3 + C*v^2 + 20 = 0) ∧ 
    (w^3 + C*w^2 + 20 = 0) ∧
    (u^3 + D*u + 100 = 0) ∧ 
    (v^3 + D*v + 100 = 0) ∧ 
    (t^3 + D*t + 100 = 0) ∧
    (u ≠ v) ∧ 
    (u * v = 10 * (2 : ℝ)^(1/3)) := by
  sorry

end NUMINAMATH_CALUDE_common_roots_product_l398_39853


namespace NUMINAMATH_CALUDE_women_half_of_total_l398_39886

/-- Represents the number of bones in different types of skeletons -/
structure BoneCount where
  woman : ℕ
  man : ℕ
  child : ℕ

/-- Represents the count of different types of skeletons -/
structure SkeletonCount where
  women : ℕ
  men : ℕ
  children : ℕ

theorem women_half_of_total (bc : BoneCount) (sc : SkeletonCount) : 
  bc.woman = 20 →
  bc.man = bc.woman + 5 →
  bc.child = bc.woman / 2 →
  sc.men = sc.children →
  sc.women + sc.men + sc.children = 20 →
  bc.woman * sc.women + bc.man * sc.men + bc.child * sc.children = 375 →
  2 * sc.women = sc.women + sc.men + sc.children := by
  sorry

#check women_half_of_total

end NUMINAMATH_CALUDE_women_half_of_total_l398_39886


namespace NUMINAMATH_CALUDE_five_number_difference_l398_39810

theorem five_number_difference (a b c d e : ℝ) 
  (h1 : (a + b + c + d) / 4 + e = 74)
  (h2 : (a + b + c + e) / 4 + d = 80)
  (h3 : (a + b + d + e) / 4 + c = 98)
  (h4 : (a + c + d + e) / 4 + b = 116)
  (h5 : (b + c + d + e) / 4 + a = 128) :
  max a (max b (max c (max d e))) - min a (min b (min c (min d e))) = 126 := by
  sorry

end NUMINAMATH_CALUDE_five_number_difference_l398_39810


namespace NUMINAMATH_CALUDE_first_number_10th_group_l398_39837

/-- Sequence term definition -/
def a (n : ℕ) : ℤ := 2 * n - 3

/-- Sum of first n natural numbers -/
def sum_first_n (n : ℕ) : ℕ := n * (n + 1) / 2

/-- The first number in the kth group -/
def first_in_group (k : ℕ) : ℕ := sum_first_n (k - 1) + 1

theorem first_number_10th_group :
  a (first_in_group 10) = 89 :=
sorry

end NUMINAMATH_CALUDE_first_number_10th_group_l398_39837


namespace NUMINAMATH_CALUDE_square_difference_division_problem_solution_l398_39835

theorem square_difference_division (a b : ℕ) (h : a > b) :
  (a^2 - b^2) / (a - b) = a + b :=
by sorry

theorem problem_solution : (144^2 - 121^2) / 23 = 265 :=
by sorry

end NUMINAMATH_CALUDE_square_difference_division_problem_solution_l398_39835


namespace NUMINAMATH_CALUDE_boat_round_trip_equation_l398_39806

/-- Represents the equation for a boat's round trip between two points -/
def boat_equation (distance : ℝ) (flow_speed : ℝ) (boat_speed : ℝ) (total_time : ℝ) : Prop :=
  (distance / (boat_speed + flow_speed)) + (distance / (boat_speed - flow_speed)) = total_time

/-- Theorem stating that the given equation correctly represents the boat's round trip -/
theorem boat_round_trip_equation : 
  ∀ (x : ℝ), x > 5 → boat_equation 60 5 x 8 :=
by sorry

end NUMINAMATH_CALUDE_boat_round_trip_equation_l398_39806


namespace NUMINAMATH_CALUDE_apps_remaining_proof_l398_39801

/-- Calculates the number of remaining apps after deletions -/
def remaining_apps (total : ℕ) (gaming : ℕ) (deleted_utility : ℕ) : ℕ :=
  total - gaming - deleted_utility

/-- Theorem: Given 12 total apps, 5 gaming apps, and deleting 3 utility apps,
    the number of remaining apps is 4 -/
theorem apps_remaining_proof :
  remaining_apps 12 5 3 = 4 := by
  sorry

end NUMINAMATH_CALUDE_apps_remaining_proof_l398_39801


namespace NUMINAMATH_CALUDE_coordinates_sum_of_A_l398_39818

/-- Given points B and C, and the condition that AC/AB = BC/AB = 1/3, 
    prove that the sum of coordinates of point A is -22 -/
theorem coordinates_sum_of_A (B C : ℝ × ℝ) (h : B = (2, -3) ∧ C = (-2, 6)) :
  let A : ℝ × ℝ := (3 * C.1 - 2 * B.1, 3 * C.2 - 2 * B.2)
  (A.1 + A.2 : ℝ) = -22 := by
  sorry

end NUMINAMATH_CALUDE_coordinates_sum_of_A_l398_39818


namespace NUMINAMATH_CALUDE_proportions_sum_l398_39848

theorem proportions_sum (x y : ℚ) :
  (4 : ℚ) / 7 = x / 63 ∧ (4 : ℚ) / 7 = 84 / y → x + y = 183 := by
  sorry

end NUMINAMATH_CALUDE_proportions_sum_l398_39848


namespace NUMINAMATH_CALUDE_circle_radius_l398_39819

-- Define the circle equation
def circle_equation (x y m : ℝ) : Prop :=
  x^2 + y^2 - 2*x + m*y - 4 = 0

-- Define the symmetry line
def symmetry_line (x y : ℝ) : Prop :=
  2*x + y = 0

-- Define the theorem
theorem circle_radius : 
  ∀ m : ℝ, 
  (∃ M N : ℝ × ℝ, 
    circle_equation M.1 M.2 m ∧ 
    circle_equation N.1 N.2 m ∧ 
    (∃ k : ℝ, symmetry_line ((M.1 + N.1)/2) ((M.2 + N.2)/2))) →
  (∃ center : ℝ × ℝ, ∀ x y : ℝ, 
    circle_equation x y m ↔ (x - center.1)^2 + (y - center.2)^2 = 3^2) :=
sorry

end NUMINAMATH_CALUDE_circle_radius_l398_39819


namespace NUMINAMATH_CALUDE_cost_difference_l398_39898

-- Define the monthly costs
def rental_cost : ℕ := 20
def new_car_cost : ℕ := 30

-- Define the number of months in a year
def months_in_year : ℕ := 12

-- Define the total costs for a year
def total_rental_cost : ℕ := rental_cost * months_in_year
def total_new_car_cost : ℕ := new_car_cost * months_in_year

-- Theorem statement
theorem cost_difference :
  total_new_car_cost - total_rental_cost = 120 :=
by sorry

end NUMINAMATH_CALUDE_cost_difference_l398_39898


namespace NUMINAMATH_CALUDE_incorrect_inequality_transformation_l398_39892

theorem incorrect_inequality_transformation (x y : ℝ) (h : x < y) : ¬(-2*x < -2*y) := by
  sorry

end NUMINAMATH_CALUDE_incorrect_inequality_transformation_l398_39892


namespace NUMINAMATH_CALUDE_expression_quadrupled_l398_39827

variables (x y : ℝ) (h : x ≠ y)

theorem expression_quadrupled :
  (2*x)^2 * (2*y) / (2*x - 2*y) = 4 * (x^2 * y / (x - y)) :=
sorry

end NUMINAMATH_CALUDE_expression_quadrupled_l398_39827


namespace NUMINAMATH_CALUDE_alloy_price_calculation_l398_39855

/-- Calculates the price of an alloy per kg given the prices of two metals and their mixing ratio -/
theorem alloy_price_calculation (price_a price_b : ℚ) (ratio : ℚ) :
  price_a = 68 →
  price_b = 96 →
  ratio = 3 →
  (ratio * price_a + price_b) / (ratio + 1) = 75 :=
by sorry

end NUMINAMATH_CALUDE_alloy_price_calculation_l398_39855


namespace NUMINAMATH_CALUDE_hemisphere_exposed_area_l398_39864

/-- Given a hemisphere of radius r, where half of it is submerged in liquid,
    the total exposed surface area (including the circular top) is 2πr². -/
theorem hemisphere_exposed_area (r : ℝ) (hr : r > 0) :
  let exposed_area := π * r^2 + (π * r^2)
  exposed_area = 2 * π * r^2 := by
  sorry

end NUMINAMATH_CALUDE_hemisphere_exposed_area_l398_39864


namespace NUMINAMATH_CALUDE_day_200_N_minus_1_is_wednesday_l398_39815

/-- Represents the days of the week -/
inductive DayOfWeek
  | Sunday
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday
  | Saturday

/-- Represents a specific day in a year -/
structure YearDay where
  year : Int
  day : Nat

/-- Function to determine the day of the week for a given YearDay -/
def dayOfWeek (yd : YearDay) : DayOfWeek := sorry

theorem day_200_N_minus_1_is_wednesday 
  (N : Int)
  (h1 : dayOfWeek ⟨N, 400⟩ = DayOfWeek.Wednesday)
  (h2 : dayOfWeek ⟨N + 2, 300⟩ = DayOfWeek.Wednesday) :
  dayOfWeek ⟨N - 1, 200⟩ = DayOfWeek.Wednesday := by
  sorry

end NUMINAMATH_CALUDE_day_200_N_minus_1_is_wednesday_l398_39815


namespace NUMINAMATH_CALUDE_locus_of_midpoint_of_tangent_l398_39841

/-- Given two circles with centers at (0, 0) and (a, 0), prove that the locus of the midpoint
    of their common outer tangent is part of a specific circle. -/
theorem locus_of_midpoint_of_tangent (a c : ℝ) (h₁ : a > c) (h₂ : c > 0) :
  ∃ (x y : ℝ), 
    4 * x^2 + 4 * y^2 - 4 * a * x + a^2 = c^2 ∧ 
    (a^2 - c^2) / (2 * a) ≤ x ∧ 
    x ≤ (a^2 + c^2) / (2 * a) ∧ 
    y > 0 :=
sorry

end NUMINAMATH_CALUDE_locus_of_midpoint_of_tangent_l398_39841


namespace NUMINAMATH_CALUDE_jackie_pushups_l398_39805

/-- Calculates the number of push-ups Jackie can do in one minute given her initial rate,
    rate of decrease, break times, and rate recovery during breaks. -/
def pushups_in_one_minute (initial_rate : ℕ) (decrease_rate : ℚ) 
                          (break_times : List ℕ) (recovery_rate : ℚ) : ℕ :=
  sorry

/-- Theorem stating that Jackie can do 15 push-ups in one minute under the given conditions. -/
theorem jackie_pushups : 
  pushups_in_one_minute 5 (1/5) [22, 38] (1/10) = 15 := by sorry

end NUMINAMATH_CALUDE_jackie_pushups_l398_39805


namespace NUMINAMATH_CALUDE_least_addition_for_divisibility_l398_39840

theorem least_addition_for_divisibility :
  ∃! x : ℕ, x < 23 ∧ (1053 + x) % 23 = 0 ∧ ∀ y : ℕ, y < x → (1053 + y) % 23 ≠ 0 :=
by sorry

end NUMINAMATH_CALUDE_least_addition_for_divisibility_l398_39840


namespace NUMINAMATH_CALUDE_gcd_problem_l398_39885

theorem gcd_problem (b : ℤ) (h : ∃ k : ℤ, b = 887 * (2 * k + 1)) :
  Int.gcd (3 * b^2 + 47 * b + 91) (b + 17) = 3 := by
  sorry

end NUMINAMATH_CALUDE_gcd_problem_l398_39885


namespace NUMINAMATH_CALUDE_books_added_to_bin_l398_39860

/-- Proves the number of books added to a bargain bin -/
theorem books_added_to_bin (initial books_sold final : ℕ) 
  (h1 : initial = 4)
  (h2 : books_sold = 3)
  (h3 : final = 11) :
  final - (initial - books_sold) = 10 := by
  sorry

end NUMINAMATH_CALUDE_books_added_to_bin_l398_39860


namespace NUMINAMATH_CALUDE_homework_completion_l398_39858

theorem homework_completion (total : ℕ) (math : ℕ) (korean : ℕ) 
  (h1 : total = 48) 
  (h2 : math = 37) 
  (h3 : korean = 42) 
  (h4 : math + korean - total ≥ 0) : 
  math + korean - total = 31 := by
  sorry

end NUMINAMATH_CALUDE_homework_completion_l398_39858


namespace NUMINAMATH_CALUDE_remaining_payment_l398_39852

def deposit_percentage : ℝ := 0.1
def deposit_amount : ℝ := 120

theorem remaining_payment (total : ℝ) (h1 : total * deposit_percentage = deposit_amount) :
  total - deposit_amount = 1080 := by sorry

end NUMINAMATH_CALUDE_remaining_payment_l398_39852


namespace NUMINAMATH_CALUDE_completing_square_result_l398_39897

theorem completing_square_result (x : ℝ) : 
  x^2 - 6*x + 7 = 0 ↔ (x - 3)^2 = 2 := by
  sorry

end NUMINAMATH_CALUDE_completing_square_result_l398_39897


namespace NUMINAMATH_CALUDE_pure_imaginary_complex_number_l398_39811

theorem pure_imaginary_complex_number (a : ℝ) :
  (Complex.I * (a - 2) : ℂ).re = 0 → a = 1 := by
  sorry

end NUMINAMATH_CALUDE_pure_imaginary_complex_number_l398_39811


namespace NUMINAMATH_CALUDE_paco_sweet_cookies_left_l398_39877

/-- The number of sweet cookies Paco has left -/
def sweet_cookies_left (initial_sweet : ℕ) (eaten_sweet : ℕ) : ℕ :=
  initial_sweet - eaten_sweet

/-- Theorem: Paco has 19 sweet cookies left -/
theorem paco_sweet_cookies_left : 
  sweet_cookies_left 34 15 = 19 := by
  sorry

end NUMINAMATH_CALUDE_paco_sweet_cookies_left_l398_39877


namespace NUMINAMATH_CALUDE_tan_22_5_decomposition_l398_39894

theorem tan_22_5_decomposition :
  ∃ (a b c : ℕ), 
    (a > 0) ∧ (b > 0) ∧ (c > 0) ∧
    (a ≥ b) ∧ (b ≥ c) ∧
    (Real.tan (22.5 * π / 180) = Real.sqrt a - Real.sqrt b - c) ∧
    (a + b + c = 4) := by
  sorry

end NUMINAMATH_CALUDE_tan_22_5_decomposition_l398_39894


namespace NUMINAMATH_CALUDE_weighing_problem_l398_39843

/-- Represents the masses of 8 items in descending order -/
def Masses := Fin 8 → ℕ

/-- The conditions for the weighing problem -/
def WeighingConditions (a : Masses) : Prop :=
  (∀ i j, i < j → a i > a j) ∧ 
  (∀ i, a i ≤ 15) ∧
  (a 0 + a 4 + a 5 + a 6 > a 1 + a 2 + a 3 + a 7) ∧
  (a 4 + a 5 > a 0 + a 6) ∧
  (a 4 > a 5)

theorem weighing_problem (a : Masses) (h : WeighingConditions a) : 
  a 4 = 11 ∧ a 6 = 5 := by
  sorry

end NUMINAMATH_CALUDE_weighing_problem_l398_39843


namespace NUMINAMATH_CALUDE_components_exceed_quarter_square_l398_39874

/-- Represents a square grid of size n × n -/
structure Grid (n : ℕ) where
  size : n > 8

/-- Represents a diagonal in a cell of the grid -/
inductive Diagonal
  | TopLeft
  | TopRight

/-- Represents the configuration of diagonals in the grid -/
def DiagonalConfig (n : ℕ) := Fin n → Fin n → Diagonal

/-- Represents a connected component in the grid -/
structure Component (n : ℕ) where
  cells : Set (Fin n × Fin n)
  is_connected : True  -- Simplified connectivity condition

/-- The number of connected components in a given diagonal configuration -/
def num_components (n : ℕ) (config : DiagonalConfig n) : ℕ := sorry

/-- Theorem stating that the number of components can exceed n²/4 for n > 8 -/
theorem components_exceed_quarter_square {n : ℕ} (grid : Grid n) :
  ∃ (config : DiagonalConfig n), num_components n config > n^2 / 4 := by sorry

end NUMINAMATH_CALUDE_components_exceed_quarter_square_l398_39874


namespace NUMINAMATH_CALUDE_simple_interest_rate_percent_l398_39822

/-- Simple interest calculation -/
theorem simple_interest_rate_percent 
  (principal : ℝ) 
  (interest : ℝ) 
  (time : ℝ) 
  (h1 : principal = 1000) 
  (h2 : interest = 400) 
  (h3 : time = 4) : 
  (interest * 100) / (principal * time) = 10 := by
sorry

end NUMINAMATH_CALUDE_simple_interest_rate_percent_l398_39822


namespace NUMINAMATH_CALUDE_smallest_k_sum_squares_multiple_180_sum_squares_360_multiple_180_smallest_k_is_360_l398_39863

theorem smallest_k_sum_squares_multiple_180 :
  ∀ k : ℕ+, (k.val * (k.val + 1) * (2 * k.val + 1)) % 1080 = 0 → k.val ≥ 360 :=
by sorry

theorem sum_squares_360_multiple_180 :
  (360 * 361 * 721) % 1080 = 0 :=
by sorry

theorem smallest_k_is_360 :
  ∃! k : ℕ+, k.val = 360 ∧
    (∀ m : ℕ+, (m.val * (m.val + 1) * (2 * m.val + 1)) % 1080 = 0 → k ≤ m) ∧
    (k.val * (k.val + 1) * (2 * k.val + 1)) % 1080 = 0 :=
by sorry

end NUMINAMATH_CALUDE_smallest_k_sum_squares_multiple_180_sum_squares_360_multiple_180_smallest_k_is_360_l398_39863


namespace NUMINAMATH_CALUDE_spongebob_earnings_l398_39850

/-- Represents the sales data for a single item --/
structure ItemSales where
  quantity : ℕ
  price : ℚ

/-- Calculates the total earnings for a single item --/
def itemEarnings (item : ItemSales) : ℚ :=
  item.quantity * item.price

/-- Represents all sales data for the day --/
structure DailySales where
  burgers : ItemSales
  largeFries : ItemSales
  smallFries : ItemSales
  sodas : ItemSales
  milkshakes : ItemSales
  softServeCones : ItemSales

/-- Calculates the total earnings for the day --/
def totalEarnings (sales : DailySales) : ℚ :=
  itemEarnings sales.burgers +
  itemEarnings sales.largeFries +
  itemEarnings sales.smallFries +
  itemEarnings sales.sodas +
  itemEarnings sales.milkshakes +
  itemEarnings sales.softServeCones

/-- Spongebob's sales data for the day --/
def spongebobSales : DailySales :=
  { burgers := { quantity := 30, price := 2.5 }
  , largeFries := { quantity := 12, price := 1.75 }
  , smallFries := { quantity := 20, price := 1.25 }
  , sodas := { quantity := 50, price := 1 }
  , milkshakes := { quantity := 18, price := 2.85 }
  , softServeCones := { quantity := 40, price := 1.3 }
  }

theorem spongebob_earnings :
  totalEarnings spongebobSales = 274.3 := by
  sorry

end NUMINAMATH_CALUDE_spongebob_earnings_l398_39850


namespace NUMINAMATH_CALUDE_bottle_filling_proportion_l398_39875

/-- Given two bottles with capacities of 4 and 8 cups, and a total of 8 cups of milk,
    prove that the proportion of capacity each bottle should be filled to is 2/3,
    when the 8-cup bottle contains 5.333333333333333 cups of milk. -/
theorem bottle_filling_proportion :
  let total_milk : ℚ := 8
  let bottle1_capacity : ℚ := 4
  let bottle2_capacity : ℚ := 8
  let milk_in_bottle2 : ℚ := 5.333333333333333
  let proportion : ℚ := milk_in_bottle2 / bottle2_capacity
  proportion = 2/3 ∧ 
  bottle1_capacity * proportion + bottle2_capacity * proportion = total_milk :=
by sorry

end NUMINAMATH_CALUDE_bottle_filling_proportion_l398_39875


namespace NUMINAMATH_CALUDE_z_in_fourth_quadrant_l398_39839

-- Define the complex number i
def i : ℂ := Complex.I

-- Define the equation z(1+i) = 2i
def equation (z : ℂ) : Prop := z * (1 + i) = 2 * i

-- Define the fourth quadrant
def fourth_quadrant (z : ℂ) : Prop :=
  z.re > 0 ∧ z.im < 0

-- Theorem statement
theorem z_in_fourth_quadrant :
  ∃ z : ℂ, equation z ∧ fourth_quadrant z :=
sorry

end NUMINAMATH_CALUDE_z_in_fourth_quadrant_l398_39839


namespace NUMINAMATH_CALUDE_arrangement_count_l398_39866

/-- Represents the number of people wearing each color -/
structure ColorCount where
  red : Nat
  yellow : Nat
  blue : Nat

/-- Represents the total number of people -/
def totalPeople (cc : ColorCount) : Nat :=
  cc.red + cc.yellow + cc.blue

/-- Calculates the number of valid arrangements -/
noncomputable def validArrangements (cc : ColorCount) : Nat :=
  sorry

/-- The main theorem to prove -/
theorem arrangement_count (cc : ColorCount) : 
  cc.red = 2 → cc.yellow = 2 → cc.blue = 1 → 
  totalPeople cc = 5 → validArrangements cc = 48 := by
  sorry

end NUMINAMATH_CALUDE_arrangement_count_l398_39866


namespace NUMINAMATH_CALUDE_gcd_50400_37800_l398_39800

theorem gcd_50400_37800 : Nat.gcd 50400 37800 = 12600 := by
  sorry

end NUMINAMATH_CALUDE_gcd_50400_37800_l398_39800


namespace NUMINAMATH_CALUDE_grid_removal_l398_39899

theorem grid_removal (n : ℕ) (h : n ≥ 10) :
  ∀ (grid : Fin n → Fin n → Bool),
  (∃ (rows : Finset (Fin n)),
    rows.card = n - 10 ∧
    ∀ (j : Fin n), ∃ (i : Fin n), i ∉ rows ∧ grid i j = true) ∨
  (∃ (cols : Finset (Fin n)),
    cols.card = n - 10 ∧
    ∀ (i : Fin n), ∃ (j : Fin n), j ∉ cols ∧ grid i j = false) :=
sorry

end NUMINAMATH_CALUDE_grid_removal_l398_39899


namespace NUMINAMATH_CALUDE_thursday_temperature_l398_39868

/-- Calculates the temperature on Thursday given the temperatures for the other days of the week and the average temperature. -/
def temperature_on_thursday (sunday monday tuesday wednesday friday saturday average : ℝ) : ℝ :=
  7 * average - (sunday + monday + tuesday + wednesday + friday + saturday)

/-- Theorem stating that the temperature on Thursday is 82° given the specified conditions. -/
theorem thursday_temperature :
  temperature_on_thursday 40 50 65 36 72 26 53 = 82 := by
  sorry

end NUMINAMATH_CALUDE_thursday_temperature_l398_39868


namespace NUMINAMATH_CALUDE_geometric_sequence_ratio_l398_39849

theorem geometric_sequence_ratio (a : ℕ → ℝ) (q : ℝ) (S : ℕ → ℝ) :
  (∀ n, a (n + 1) = q * a n) →  -- geometric sequence condition
  q ≠ 1 →  -- common ratio is not 1
  (∀ n, S n = (a 1) * (1 - q^n) / (1 - q)) →  -- sum formula for geometric sequence
  (2 * a 5 = (a 2 + 3 * a 8) / 2) →  -- arithmetic sequence condition
  (3 * S 3) / (S 6) = 9 / 4 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_ratio_l398_39849


namespace NUMINAMATH_CALUDE_floor_of_e_equals_two_l398_39859

-- Define e as the base of natural logarithms
noncomputable def e : ℝ := Real.exp 1

-- Theorem statement
theorem floor_of_e_equals_two : ⌊e⌋ = 2 := by
  sorry

end NUMINAMATH_CALUDE_floor_of_e_equals_two_l398_39859


namespace NUMINAMATH_CALUDE_smallest_rectangle_containing_circle_l398_39851

/-- The diameter of the circle -/
def circle_diameter : ℝ := 10

/-- The area of the smallest rectangle containing the circle -/
def smallest_rectangle_area : ℝ := 120

/-- Theorem stating that the area of the smallest rectangle containing a circle
    with diameter 10 units is 120 square units -/
theorem smallest_rectangle_containing_circle :
  smallest_rectangle_area = circle_diameter * (circle_diameter + 2) :=
by sorry

end NUMINAMATH_CALUDE_smallest_rectangle_containing_circle_l398_39851


namespace NUMINAMATH_CALUDE_first_five_valid_numbers_l398_39867

def is_valid (n : ℕ) : Bool :=
  n ≥ 0 ∧ n ≤ 499

def random_sequence : List ℕ :=
  [164, 785, 916, 955, 567, 199, 810, 507, 185, 128, 673, 580, 744, 395]

def first_five_valid (seq : List ℕ) : List ℕ :=
  seq.filter is_valid |> List.take 5

theorem first_five_valid_numbers :
  first_five_valid random_sequence = [164, 199, 185, 128, 395] := by
  sorry

end NUMINAMATH_CALUDE_first_five_valid_numbers_l398_39867


namespace NUMINAMATH_CALUDE_polynomial_division_remainder_l398_39854

theorem polynomial_division_remainder : ∃ q : Polynomial ℝ, 
  3 * X^4 + 14 * X^3 - 55 * X^2 - 73 * X + 65 = 
  (X^2 + 8 * X - 6) * q + (-477 * X + 323) := by
  sorry

end NUMINAMATH_CALUDE_polynomial_division_remainder_l398_39854


namespace NUMINAMATH_CALUDE_coin_toss_is_random_event_l398_39872

/-- Represents the outcome of a coin toss -/
inductive CoinOutcome
| Heads
| Tails

/-- Represents a random event -/
class RandomEvent (α : Type) where
  /-- The probability of the event occurring is between 0 and 1, exclusive -/
  prob_between_zero_and_one : ∃ (p : ℝ), 0 < p ∧ p < 1

/-- Definition of a coin toss -/
def coinToss : Set CoinOutcome := {CoinOutcome.Heads, CoinOutcome.Tails}

/-- Theorem: Tossing a coin is a random event -/
theorem coin_toss_is_random_event : RandomEvent coinToss := by
  sorry


end NUMINAMATH_CALUDE_coin_toss_is_random_event_l398_39872


namespace NUMINAMATH_CALUDE_completing_square_equivalence_l398_39809

theorem completing_square_equivalence :
  ∀ x : ℝ, x^2 - 2*x = 2 ↔ (x - 1)^2 = 3 :=
by sorry

end NUMINAMATH_CALUDE_completing_square_equivalence_l398_39809


namespace NUMINAMATH_CALUDE_intersection_point_l398_39803

theorem intersection_point (x y : ℚ) : 
  (8 * x - 5 * y = 10) ∧ (6 * x + 2 * y = 22) ↔ (x = 65/23 ∧ y = -137/23) :=
by sorry

end NUMINAMATH_CALUDE_intersection_point_l398_39803


namespace NUMINAMATH_CALUDE_product_of_smallest_primes_l398_39812

def smallest_two_digit_prime_1 : Nat := 11
def smallest_two_digit_prime_2 : Nat := 13
def smallest_three_digit_prime : Nat := 101

theorem product_of_smallest_primes :
  smallest_two_digit_prime_1 * smallest_two_digit_prime_2 * smallest_three_digit_prime = 14443 := by
  sorry

end NUMINAMATH_CALUDE_product_of_smallest_primes_l398_39812


namespace NUMINAMATH_CALUDE_correct_operation_is_subtraction_l398_39847

-- Define the possible operations
inductive Operation
  | Add
  | Multiply
  | Divide
  | Subtract

-- Function to apply the operation
def applyOperation (op : Operation) (a b : ℤ) : ℤ :=
  match op with
  | Operation.Add => a + b
  | Operation.Multiply => a * b
  | Operation.Divide => a / b
  | Operation.Subtract => a - b

-- Theorem statement
theorem correct_operation_is_subtraction :
  ∃! op : Operation, (applyOperation op 8 4) + 6 - (3 - 2) = 9 :=
by sorry

end NUMINAMATH_CALUDE_correct_operation_is_subtraction_l398_39847


namespace NUMINAMATH_CALUDE_number_of_possible_sums_l398_39808

/-- The set of chips in Bag A -/
def bagA : Finset ℕ := {1, 4, 5}

/-- The set of chips in Bag B -/
def bagB : Finset ℕ := {2, 4, 6}

/-- The set of all possible sums when drawing one chip from each bag -/
def possibleSums : Finset ℕ := (bagA.product bagB).image (fun p => p.1 + p.2)

theorem number_of_possible_sums : Finset.card possibleSums = 8 := by
  sorry

end NUMINAMATH_CALUDE_number_of_possible_sums_l398_39808


namespace NUMINAMATH_CALUDE_final_output_is_218_l398_39893

def machine_transform (a : ℕ) : ℕ :=
  if a % 2 = 1 then a + 3 else a + 5

def repeated_transform (a : ℕ) (n : ℕ) : ℕ :=
  match n with
  | 0 => a
  | n + 1 => machine_transform (repeated_transform a n)

theorem final_output_is_218 :
  repeated_transform 15 51 = 218 := by
  sorry

end NUMINAMATH_CALUDE_final_output_is_218_l398_39893


namespace NUMINAMATH_CALUDE_third_term_is_negative_eight_l398_39842

def sequence_sum (n : ℕ) : ℤ := 2 - 2^(n + 1)

def sequence_term (n : ℕ) : ℤ := -2^n

theorem third_term_is_negative_eight :
  sequence_term 3 = -8 ∧
  ∀ n : ℕ, n ≥ 1 → sequence_sum n - sequence_sum (n - 1) = sequence_term n :=
sorry

end NUMINAMATH_CALUDE_third_term_is_negative_eight_l398_39842


namespace NUMINAMATH_CALUDE_ellipse_properties_l398_39888

/-- Definition of the ellipse C -/
def ellipse_C (x y : ℝ) (a b : ℝ) : Prop :=
  x^2 / a^2 + y^2 / b^2 = 1 ∧ a > b ∧ b > 0

/-- Definition of the triangle AF₁F₂ -/
def triangle_AF1F2 (A F1 F2 : ℝ × ℝ) : Prop :=
  let d12 := Real.sqrt ((F1.1 - F2.1)^2 + (F1.2 - F2.2)^2)
  let d1A := Real.sqrt ((F1.1 - A.1)^2 + (F1.2 - A.2)^2)
  let d2A := Real.sqrt ((F2.1 - A.1)^2 + (F2.2 - A.2)^2)
  d12 = 2 * Real.sqrt 2 ∧ d1A = d2A ∧ d1A^2 + d2A^2 = d12^2

/-- Main theorem -/
theorem ellipse_properties
  (a b : ℝ)
  (A F1 F2 : ℝ × ℝ)
  (h_ellipse : ∀ x y, ellipse_C x y a b)
  (h_triangle : triangle_AF1F2 A F1 F2) :
  (∀ x y, x^2 / 4 + y^2 / 2 = 1 ↔ ellipse_C x y a b) ∧
  (∀ P Q : ℝ × ℝ, P.2 = P.1 + 1 → Q.2 = Q.1 + 1 →
    ellipse_C P.1 P.2 a b → ellipse_C Q.1 Q.2 a b →
    (P.1 - Q.1)^2 + (P.2 - Q.2)^2 = (4/3)^2 * 5) ∧
  (¬ ∃ m : ℝ, ∀ P Q : ℝ × ℝ,
    P.2 = P.1 + m → Q.2 = Q.1 + m →
    ellipse_C P.1 P.2 a b → ellipse_C Q.1 Q.2 a b →
    (1/2) * Real.sqrt ((P.1 - Q.1)^2 + (P.2 - Q.2)^2) * (|m| / Real.sqrt 2) = 4/3) :=
sorry

end NUMINAMATH_CALUDE_ellipse_properties_l398_39888


namespace NUMINAMATH_CALUDE_circles_tangent_line_slope_l398_39887

-- Define the circles and their properties
def Circle := ℝ × ℝ → Prop

-- Define the conditions
def intersect_at_4_9 (C₁ C₂ : Circle) : Prop := 
  C₁ (4, 9) ∧ C₂ (4, 9)

def product_of_radii_85 (C₁ C₂ : Circle) : Prop := 
  ∃ r₁ r₂ : ℝ, r₁ * r₂ = 85

def tangent_to_y_axis (C : Circle) : Prop := 
  ∃ x : ℝ, C (0, x)

def tangent_line (n : ℝ) (C : Circle) : Prop := 
  ∃ x y : ℝ, C (x, y) ∧ y = n * x

-- Main theorem
theorem circles_tangent_line_slope (C₁ C₂ : Circle) (n : ℝ) :
  intersect_at_4_9 C₁ C₂ →
  product_of_radii_85 C₁ C₂ →
  tangent_to_y_axis C₁ →
  tangent_to_y_axis C₂ →
  tangent_line n C₁ →
  tangent_line n C₂ →
  n > 0 →
  ∃ d e f : ℕ,
    d > 0 ∧ e > 0 ∧ f > 0 ∧
    (∀ (p : ℕ), Prime p → ¬(p^2 ∣ e)) ∧
    Nat.Coprime d f ∧
    n = (d : ℝ) * Real.sqrt e / f ∧
    d + e + f = 243 :=
sorry

end NUMINAMATH_CALUDE_circles_tangent_line_slope_l398_39887


namespace NUMINAMATH_CALUDE_box_tie_length_l398_39883

/-- Calculates the length of string used to tie a box given the initial length,
    the amount given away, and the fraction of the remainder used. -/
def string_used_for_box (initial_length : ℝ) (given_away : ℝ) (fraction_used : ℚ) : ℝ :=
  (initial_length - given_away) * (fraction_used : ℝ)

/-- Proves that given a string of 90 cm, after removing 30 cm, and using 8/15 of the remainder,
    the length used to tie the box is 32 cm. -/
theorem box_tie_length : 
  string_used_for_box 90 30 (8/15) = 32 := by sorry

end NUMINAMATH_CALUDE_box_tie_length_l398_39883
