import Mathlib

namespace NUMINAMATH_CALUDE_caterpillar_problem_solution_l313_31328

/-- Represents the caterpillar problem --/
structure CaterpillarProblem where
  initial_caterpillars : ℕ
  fallen_caterpillars : ℕ
  hatched_eggs : ℕ
  leaves_per_day : ℕ
  observation_days : ℕ
  cocooned_caterpillars : ℕ

/-- Calculates the number of caterpillars left on the tree and leaves eaten --/
def solve_caterpillar_problem (problem : CaterpillarProblem) : ℕ × ℕ :=
  let remaining_after_storm := problem.initial_caterpillars - problem.fallen_caterpillars
  let total_after_hatching := remaining_after_storm + problem.hatched_eggs
  let remaining_after_cocooning := total_after_hatching - problem.cocooned_caterpillars
  let final_caterpillars := remaining_after_cocooning / 2
  let leaves_eaten := problem.hatched_eggs * problem.leaves_per_day * problem.observation_days
  (final_caterpillars, leaves_eaten)

/-- Theorem stating the solution to the caterpillar problem --/
theorem caterpillar_problem_solution :
  let problem : CaterpillarProblem := {
    initial_caterpillars := 14,
    fallen_caterpillars := 3,
    hatched_eggs := 6,
    leaves_per_day := 2,
    observation_days := 7,
    cocooned_caterpillars := 9
  }
  solve_caterpillar_problem problem = (4, 84) := by
  sorry


end NUMINAMATH_CALUDE_caterpillar_problem_solution_l313_31328


namespace NUMINAMATH_CALUDE_dinner_cost_l313_31307

theorem dinner_cost (num_people : ℕ) (total_amount : ℚ) (ice_cream_cost : ℚ) 
  (h1 : num_people = 3)
  (h2 : total_amount = 45)
  (h3 : ice_cream_cost = 5) :
  (total_amount - num_people * ice_cream_cost) / num_people = 10 :=
by sorry

end NUMINAMATH_CALUDE_dinner_cost_l313_31307


namespace NUMINAMATH_CALUDE_shifted_data_invariants_l313_31377

variable {n : ℕ}
variable (X Y : Fin n → ℝ)
variable (c : ℝ)

def is_shifted (X Y : Fin n → ℝ) (c : ℝ) : Prop :=
  ∀ i, Y i = X i + c

def standard_deviation (X : Fin n → ℝ) : ℝ := sorry

def range (X : Fin n → ℝ) : ℝ := sorry

theorem shifted_data_invariants (h : is_shifted X Y c) (h_nonzero : c ≠ 0) :
  standard_deviation Y = standard_deviation X ∧ range Y = range X := by sorry

end NUMINAMATH_CALUDE_shifted_data_invariants_l313_31377


namespace NUMINAMATH_CALUDE_base_number_power_remainder_l313_31350

theorem base_number_power_remainder (base : ℕ) : base = 1 → base ^ 8 % 100 = 1 := by
  sorry

end NUMINAMATH_CALUDE_base_number_power_remainder_l313_31350


namespace NUMINAMATH_CALUDE_stating_kevin_vanessa_age_multiple_l313_31300

/-- Represents the age difference between Kevin and Vanessa -/
def age_difference : ℕ := 14

/-- Represents Kevin's initial age -/
def kevin_initial_age : ℕ := 16

/-- Represents Vanessa's initial age -/
def vanessa_initial_age : ℕ := 2

/-- 
Theorem stating that the first time Kevin's age becomes a multiple of Vanessa's age, 
Kevin will be 4.5 times older than Vanessa.
-/
theorem kevin_vanessa_age_multiple :
  ∃ (years : ℕ), 
    (kevin_initial_age + years) % (vanessa_initial_age + years) = 0 ∧
    (kevin_initial_age + years : ℚ) / (vanessa_initial_age + years : ℚ) = 4.5 ∧
    ∀ (y : ℕ), y < years → (kevin_initial_age + y) % (vanessa_initial_age + y) ≠ 0 :=
sorry

end NUMINAMATH_CALUDE_stating_kevin_vanessa_age_multiple_l313_31300


namespace NUMINAMATH_CALUDE_product_of_integers_l313_31386

theorem product_of_integers (p q r : ℤ) : 
  p ≠ 0 → q ≠ 0 → r ≠ 0 →
  p + q + r = 30 →
  1 / p + 1 / q + 1 / r + 450 / (p * q * r) = 1 →
  p * q * r = 1920 := by
sorry

end NUMINAMATH_CALUDE_product_of_integers_l313_31386


namespace NUMINAMATH_CALUDE_min_value_a_l313_31387

theorem min_value_a (a : ℝ) (h1 : a > 1) :
  (∀ x : ℝ, x ≥ 1/3 → (1/(3*x) - 2*x + Real.log (3*x) ≤ 1/(a*(Real.exp (2*x))) + Real.log a)) →
  a ≥ 3/(2*(Real.exp 1)) := by
  sorry

end NUMINAMATH_CALUDE_min_value_a_l313_31387


namespace NUMINAMATH_CALUDE_jean_price_satisfies_conditions_l313_31379

/-- The price of a jean that satisfies the given conditions -/
def jean_price : ℝ := 11

/-- The price of a tee -/
def tee_price : ℝ := 8

/-- The number of tees sold -/
def tees_sold : ℕ := 7

/-- The number of jeans sold -/
def jeans_sold : ℕ := 4

/-- The total revenue -/
def total_revenue : ℝ := 100

/-- Theorem stating that the jean price satisfies the given conditions -/
theorem jean_price_satisfies_conditions :
  tee_price * tees_sold + jean_price * jeans_sold = total_revenue := by
  sorry

#check jean_price_satisfies_conditions

end NUMINAMATH_CALUDE_jean_price_satisfies_conditions_l313_31379


namespace NUMINAMATH_CALUDE_cube_sum_equals_sum_l313_31371

theorem cube_sum_equals_sum (a b : ℝ) (ha : a > 0) (hb : b > 0)
  (h : a / (1 + b) + b / (1 + a) = 1) : a^3 + b^3 = a + b := by
  sorry

end NUMINAMATH_CALUDE_cube_sum_equals_sum_l313_31371


namespace NUMINAMATH_CALUDE_train_length_l313_31390

/-- The length of a train given specific conditions --/
theorem train_length (jogger_speed : ℝ) (train_speed : ℝ) (initial_distance : ℝ) (passing_time : ℝ) :
  jogger_speed = 9 * (1000 / 3600) →
  train_speed = 45 * (1000 / 3600) →
  initial_distance = 280 →
  passing_time = 40 →
  (train_speed - jogger_speed) * passing_time + initial_distance = 680 := by
  sorry

end NUMINAMATH_CALUDE_train_length_l313_31390


namespace NUMINAMATH_CALUDE_xyz_sum_l313_31313

theorem xyz_sum (x y z : ℕ+) 
  (h1 : x * y + z = 47)
  (h2 : y * z + x = 47)
  (h3 : z * x + y = 47) : 
  x + y + z = 48 := by
sorry

end NUMINAMATH_CALUDE_xyz_sum_l313_31313


namespace NUMINAMATH_CALUDE_composite_n_fourth_plus_64_l313_31383

theorem composite_n_fourth_plus_64 : ∃ (n : ℕ), ∃ (a b : ℕ), a > 1 ∧ b > 1 ∧ n^4 + 64 = a * b :=
sorry

end NUMINAMATH_CALUDE_composite_n_fourth_plus_64_l313_31383


namespace NUMINAMATH_CALUDE_pure_imaginary_modulus_l313_31366

theorem pure_imaginary_modulus (a : ℝ) : 
  let z : ℂ := a^2 - 1 + (a + 1) * Complex.I
  (z.re = 0 ∧ z.im ≠ 0) → Complex.abs z = 2 := by
  sorry

end NUMINAMATH_CALUDE_pure_imaginary_modulus_l313_31366


namespace NUMINAMATH_CALUDE_correct_average_after_adjustments_l313_31388

theorem correct_average_after_adjustments (n : ℕ) (initial_avg : ℚ) 
  (error1 : ℚ) (wrong_num : ℚ) (correct_num : ℚ) :
  n = 10 →
  initial_avg = 40.2 →
  error1 = 17 →
  wrong_num = 13 →
  correct_num = 31 →
  (n : ℚ) * initial_avg - error1 - wrong_num + correct_num = n * 40.3 := by
  sorry

end NUMINAMATH_CALUDE_correct_average_after_adjustments_l313_31388


namespace NUMINAMATH_CALUDE_point_coordinates_l313_31360

/-- A point in the second quadrant with given distances from axes -/
structure PointInSecondQuadrant where
  x : ℝ
  y : ℝ
  in_second_quadrant : x < 0 ∧ y > 0
  distance_from_x_axis : |y| = 2
  distance_from_y_axis : |x| = 5

/-- The coordinates of the point are (-5, 2) -/
theorem point_coordinates (P : PointInSecondQuadrant) : P.x = -5 ∧ P.y = 2 := by
  sorry

end NUMINAMATH_CALUDE_point_coordinates_l313_31360


namespace NUMINAMATH_CALUDE_figurine_cost_is_17_l313_31351

/-- The cost of each figurine in Annie's purchase --/
def figurine_cost : ℚ :=
  let brand_a_cost : ℚ := 65
  let brand_b_cost : ℚ := 75
  let brand_c_cost : ℚ := 85
  let brand_a_count : ℕ := 3
  let brand_b_count : ℕ := 2
  let brand_c_count : ℕ := 4
  let figurine_count : ℕ := 10
  let figurine_total_cost : ℚ := 2 * brand_c_cost
  figurine_total_cost / figurine_count

theorem figurine_cost_is_17 : figurine_cost = 17 := by
  sorry

end NUMINAMATH_CALUDE_figurine_cost_is_17_l313_31351


namespace NUMINAMATH_CALUDE_data_analytics_course_hours_l313_31310

/-- Calculates the total hours spent on a course given the course duration and weekly schedule. -/
def total_course_hours (weeks : ℕ) (three_hour_classes : ℕ) (four_hour_classes : ℕ) (homework_hours : ℕ) : ℕ :=
  weeks * (three_hour_classes * 3 + four_hour_classes * 4 + homework_hours)

/-- Proves that the total hours spent on the given course is 336. -/
theorem data_analytics_course_hours : 
  total_course_hours 24 2 1 4 = 336 := by
  sorry

end NUMINAMATH_CALUDE_data_analytics_course_hours_l313_31310


namespace NUMINAMATH_CALUDE_race_time_proof_l313_31391

/-- Represents the time taken by the first five runners to finish the race -/
def first_five_time : ℝ → ℝ := λ t => 5 * t

/-- Represents the time taken by the last three runners to finish the race -/
def last_three_time : ℝ → ℝ := λ t => 3 * (t + 2)

/-- Represents the total time taken by all runners to finish the race -/
def total_time : ℝ → ℝ := λ t => first_five_time t + last_three_time t

theorem race_time_proof :
  ∃ t : ℝ, total_time t = 70 ∧ first_five_time t = 40 :=
sorry

end NUMINAMATH_CALUDE_race_time_proof_l313_31391


namespace NUMINAMATH_CALUDE_ceiling_fraction_evaluation_l313_31319

theorem ceiling_fraction_evaluation : 
  (⌈⌈(23:ℝ)/9 - ⌈(35:ℝ)/21⌉⌉⌉ : ℝ) / ⌈⌈(36:ℝ)/9 + ⌈(9:ℝ)*23/36⌉⌉⌉ = 1/10 := by
  sorry

end NUMINAMATH_CALUDE_ceiling_fraction_evaluation_l313_31319


namespace NUMINAMATH_CALUDE_negation_of_universal_proposition_l313_31312

theorem negation_of_universal_proposition :
  (¬ (∀ x : ℝ, x^2 > 1)) ↔ (∃ x : ℝ, x^2 ≤ 1) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_universal_proposition_l313_31312


namespace NUMINAMATH_CALUDE_number_of_men_is_seven_l313_31397

/-- The number of men in a group, given certain conditions about age changes when replacing men with women. -/
def number_of_men : ℕ :=
  let initial_men_count : ℕ := 7
  let age_increase : ℝ := 3
  let replaced_man1_age : ℝ := 18
  let replaced_man2_age : ℝ := 22
  let women_avg_age : ℝ := 30.5
  initial_men_count

theorem number_of_men_is_seven :
  let M := number_of_men
  let A := Real
  (∀ A, (M * A - (replaced_man1_age + replaced_man2_age) + 2 * women_avg_age) / M = A + age_increase) →
  M = 7 := by
  sorry

#eval number_of_men

end NUMINAMATH_CALUDE_number_of_men_is_seven_l313_31397


namespace NUMINAMATH_CALUDE_crayon_factory_colors_crayon_factory_colors_proof_l313_31395

/-- A crayon factory problem -/
theorem crayon_factory_colors (crayons_per_color : ℕ) (boxes_per_hour : ℕ) 
  (total_crayons : ℕ) (total_hours : ℕ) (colors : ℕ) : Prop :=
  crayons_per_color = 2 →
  boxes_per_hour = 5 →
  total_crayons = 160 →
  total_hours = 4 →
  colors * crayons_per_color * boxes_per_hour * total_hours = total_crayons →
  colors = 4

/-- Proof of the crayon factory problem -/
theorem crayon_factory_colors_proof : 
  ∃ (crayons_per_color boxes_per_hour total_crayons total_hours colors : ℕ),
  crayon_factory_colors crayons_per_color boxes_per_hour total_crayons total_hours colors :=
by
  sorry

end NUMINAMATH_CALUDE_crayon_factory_colors_crayon_factory_colors_proof_l313_31395


namespace NUMINAMATH_CALUDE_total_hats_bought_l313_31368

theorem total_hats_bought (blue_hat_price green_hat_price total_price green_hats : ℕ)
  (h1 : blue_hat_price = 6)
  (h2 : green_hat_price = 7)
  (h3 : total_price = 540)
  (h4 : green_hats = 30) :
  ∃ (blue_hats : ℕ), blue_hats * blue_hat_price + green_hats * green_hat_price = total_price ∧
                     blue_hats + green_hats = 85 := by
  sorry

end NUMINAMATH_CALUDE_total_hats_bought_l313_31368


namespace NUMINAMATH_CALUDE_garbs_are_morks_and_plogs_l313_31353

-- Define the sets
variable (U : Type) -- Universe set
variable (Mork Plog Snark Garb : Set U)

-- Define the given conditions
variable (h1 : Mork ⊆ Plog)    -- All Morks are Plogs
variable (h2 : Snark ⊆ Mork)   -- All Snarks are Morks
variable (h3 : Garb ⊆ Plog)    -- All Garbs are Plogs
variable (h4 : Garb ⊆ Snark)   -- All Garbs are Snarks

-- Theorem to prove
theorem garbs_are_morks_and_plogs : Garb ⊆ Mork ∩ Plog :=
sorry

end NUMINAMATH_CALUDE_garbs_are_morks_and_plogs_l313_31353


namespace NUMINAMATH_CALUDE_sum_of_bases_equals_1657_l313_31369

/-- Converts a number from base 13 to base 10 -/
def base13ToBase10 (a b c : ℕ) : ℕ := a * 13^2 + b * 13 + c

/-- Converts a number from base 14 to base 10 -/
def base14ToBase10 (a b c : ℕ) : ℕ := a * 14^2 + b * 14 + c

/-- The value of digit C in base 14 -/
def C : ℕ := 12

theorem sum_of_bases_equals_1657 :
  base13ToBase10 4 2 0 + base14ToBase10 4 C 3 = 1657 := by sorry

end NUMINAMATH_CALUDE_sum_of_bases_equals_1657_l313_31369


namespace NUMINAMATH_CALUDE_height_weight_most_suitable_for_regression_l313_31315

-- Define the types of variables
inductive Variable
| CircleArea
| CircleRadius
| Height
| Weight
| ColorBlindness
| Gender
| AcademicPerformance

-- Define a pair of variables
structure VariablePair where
  var1 : Variable
  var2 : Variable

-- Define the types of relationships between variables
inductive Relationship
| Functional
| Correlated
| Unrelated

-- Function to determine the relationship between a pair of variables
def relationshipBetween (pair : VariablePair) : Relationship :=
  match pair with
  | ⟨Variable.CircleArea, Variable.CircleRadius⟩ => Relationship.Functional
  | ⟨Variable.ColorBlindness, Variable.Gender⟩ => Relationship.Unrelated
  | ⟨Variable.Height, Variable.AcademicPerformance⟩ => Relationship.Unrelated
  | ⟨Variable.Height, Variable.Weight⟩ => Relationship.Correlated
  | _ => Relationship.Unrelated  -- Default case

-- Function to determine if a pair is suitable for regression analysis
def suitableForRegression (pair : VariablePair) : Prop :=
  relationshipBetween pair = Relationship.Correlated

-- Theorem stating that height and weight is the most suitable pair for regression
theorem height_weight_most_suitable_for_regression :
  suitableForRegression ⟨Variable.Height, Variable.Weight⟩ ∧
  ¬suitableForRegression ⟨Variable.CircleArea, Variable.CircleRadius⟩ ∧
  ¬suitableForRegression ⟨Variable.ColorBlindness, Variable.Gender⟩ ∧
  ¬suitableForRegression ⟨Variable.Height, Variable.AcademicPerformance⟩ :=
by sorry

end NUMINAMATH_CALUDE_height_weight_most_suitable_for_regression_l313_31315


namespace NUMINAMATH_CALUDE_consecutive_even_sum_l313_31305

theorem consecutive_even_sum (n : ℤ) : 
  (∃ m : ℤ, m = n + 2 ∧ (m^2 - n^2 = 84)) → n + (n + 2) = 42 := by
  sorry

end NUMINAMATH_CALUDE_consecutive_even_sum_l313_31305


namespace NUMINAMATH_CALUDE_suresh_work_hours_l313_31317

/-- Proves that Suresh worked for 9 hours given the conditions of the problem -/
theorem suresh_work_hours 
  (suresh_rate : ℚ) 
  (ashutosh_rate : ℚ) 
  (ashutosh_remaining_hours : ℚ) 
  (h1 : suresh_rate = 1 / 15)
  (h2 : ashutosh_rate = 1 / 20)
  (h3 : ashutosh_remaining_hours = 8)
  : ∃ x : ℚ, x * suresh_rate + ashutosh_remaining_hours * ashutosh_rate = 1 ∧ x = 9 := by
  sorry

end NUMINAMATH_CALUDE_suresh_work_hours_l313_31317


namespace NUMINAMATH_CALUDE_golden_ratio_greater_than_three_fifths_l313_31302

theorem golden_ratio_greater_than_three_fifths : (Real.sqrt 5 - 1) / 2 > 3 / 5 := by
  sorry

end NUMINAMATH_CALUDE_golden_ratio_greater_than_three_fifths_l313_31302


namespace NUMINAMATH_CALUDE_mango_ratio_l313_31321

theorem mango_ratio (total_mangoes : ℕ) (unripe_fraction : ℚ) (kept_unripe : ℕ) 
  (mangoes_per_jar : ℕ) (jars_made : ℕ) : 
  total_mangoes = 54 →
  unripe_fraction = 2/3 →
  kept_unripe = 16 →
  mangoes_per_jar = 4 →
  jars_made = 5 →
  (total_mangoes * (1 - unripe_fraction) : ℚ) / total_mangoes = 1/3 :=
by sorry

end NUMINAMATH_CALUDE_mango_ratio_l313_31321


namespace NUMINAMATH_CALUDE_circle_a_properties_l313_31341

/-- Circle A with center (m, 2/m) passing through origin -/
def CircleA (m : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | (p.1 - m)^2 + (p.2 - 2/m)^2 = m^2 + 4/m^2}

/-- Line l: 2x + y - 4 = 0 -/
def LineL : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | 2 * p.1 + p.2 - 4 = 0}

theorem circle_a_properties (m : ℝ) (hm : m > 0) :
  /- When m = 2, the circle equation is (x-2)² + (y-1)² = 5 -/
  (∀ p : ℝ × ℝ, p ∈ CircleA 2 ↔ (p.1 - 2)^2 + (p.2 - 1)^2 = 5) ∧
  /- The area of triangle OBC is constant and equal to 4 -/
  (∃ B C : ℝ × ℝ, B ∈ CircleA m ∧ C ∈ CircleA m ∧ B.2 = 0 ∧ C.1 = 0 ∧
    abs (B.1 * C.2) / 2 = 4) ∧
  /- If line l intersects circle A at P and Q where |OP| = |OQ|, then |PQ| = 4√30/5 -/
  (∃ P Q : ℝ × ℝ, P ∈ CircleA 2 ∧ Q ∈ CircleA 2 ∧ P ∈ LineL ∧ Q ∈ LineL ∧
    P.1^2 + P.2^2 = Q.1^2 + Q.2^2 →
    (P.1 - Q.1)^2 + (P.2 - Q.2)^2 = (4 * Real.sqrt 30 / 5)^2) :=
by sorry


end NUMINAMATH_CALUDE_circle_a_properties_l313_31341


namespace NUMINAMATH_CALUDE_math_olympiad_scores_l313_31394

theorem math_olympiad_scores (n : ℕ) (scores : Fin n → ℕ) : 
  n = 20 →
  (∀ i j : Fin n, i ≠ j → scores i ≠ scores j) →
  (∀ i j k : Fin n, i ≠ j → j ≠ k → i ≠ k → scores i < scores j + scores k) →
  ∀ i : Fin n, scores i > 18 := by
  sorry

end NUMINAMATH_CALUDE_math_olympiad_scores_l313_31394


namespace NUMINAMATH_CALUDE_company_bonus_fund_l313_31322

theorem company_bonus_fund : ∃ (n : ℕ) (initial_fund : ℕ), 
  (60 * n - 10 = initial_fund) ∧ 
  (50 * n + 110 = initial_fund) ∧ 
  (initial_fund = 710) := by
  sorry

end NUMINAMATH_CALUDE_company_bonus_fund_l313_31322


namespace NUMINAMATH_CALUDE_unreserved_seat_cost_l313_31335

theorem unreserved_seat_cost (total_revenue : ℚ) (reserved_seat_cost : ℚ) 
  (reserved_tickets : ℕ) (unreserved_tickets : ℕ) :
  let unreserved_seat_cost := (total_revenue - reserved_seat_cost * reserved_tickets) / unreserved_tickets
  total_revenue = 26170 ∧ 
  reserved_seat_cost = 25 ∧ 
  reserved_tickets = 246 ∧ 
  unreserved_tickets = 246 → 
  unreserved_seat_cost = 81.3 := by
sorry

end NUMINAMATH_CALUDE_unreserved_seat_cost_l313_31335


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l313_31344

def A : Set ℝ := {1, 4}
def B : Set ℝ := {x | 1 ≤ x ∧ x ≤ 3}

theorem intersection_of_A_and_B : A ∩ B = {1} := by sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l313_31344


namespace NUMINAMATH_CALUDE_perfect_square_base7_b_values_l313_31343

/-- A number in base 7 of the form ac4b where a ≠ 0 and 0 ≤ b < 7 -/
structure Base7Number where
  a : ℕ
  c : ℕ
  b : ℕ
  a_nonzero : a ≠ 0
  b_range : b < 7

/-- Convert a Base7Number to its decimal representation -/
def to_decimal (n : Base7Number) : ℕ :=
  343 * n.a + 49 * n.c + 28 + n.b

/-- Predicate to check if a natural number is a perfect square -/
def is_perfect_square (n : ℕ) : Prop :=
  ∃ m : ℕ, n = m * m

theorem perfect_square_base7_b_values (n : Base7Number) :
  is_perfect_square (to_decimal n) → n.b = 0 ∨ n.b = 1 ∨ n.b = 4 :=
sorry

end NUMINAMATH_CALUDE_perfect_square_base7_b_values_l313_31343


namespace NUMINAMATH_CALUDE_green_bean_to_corn_ratio_l313_31393

/-- Represents the number of servings produced by each type of plant. -/
structure PlantServings where
  carrot : ℕ
  corn : ℕ
  greenBean : ℕ

/-- Represents the number of plants in each plot. -/
def plantsPerPlot : ℕ := 9

/-- Represents the total number of servings produced. -/
def totalServings : ℕ := 306

/-- The theorem stating the ratio of green bean to corn servings. -/
theorem green_bean_to_corn_ratio (s : PlantServings) :
  s.carrot = 4 →
  s.corn = 5 * s.carrot →
  s.greenBean * plantsPerPlot + s.carrot * plantsPerPlot + s.corn * plantsPerPlot = totalServings →
  s.greenBean * 2 = s.corn := by
  sorry

#check green_bean_to_corn_ratio

end NUMINAMATH_CALUDE_green_bean_to_corn_ratio_l313_31393


namespace NUMINAMATH_CALUDE_cost_of_45_roses_l313_31320

/-- The cost of a bouquet of roses with discount applied -/
def bouquet_cost (n : ℕ) : ℚ :=
  let base_cost := 30 * (n / 15 : ℚ)
  if n > 30 then base_cost * (1 - 1/10) else base_cost

/-- Theorem stating the cost of a bouquet with 45 roses -/
theorem cost_of_45_roses : bouquet_cost 45 = 81 := by
  sorry

end NUMINAMATH_CALUDE_cost_of_45_roses_l313_31320


namespace NUMINAMATH_CALUDE_factory_production_l313_31380

/-- The total number of cars made by a factory over two days, given the production on the first day and that the second day's production is twice the first day's. -/
def total_cars (first_day_production : ℕ) : ℕ :=
  first_day_production + 2 * first_day_production

/-- Theorem stating that the total number of cars made over two days is 180,
    given that 60 cars were made on the first day. -/
theorem factory_production : total_cars 60 = 180 := by
  sorry

end NUMINAMATH_CALUDE_factory_production_l313_31380


namespace NUMINAMATH_CALUDE_irrationality_of_lambda_l313_31396

theorem irrationality_of_lambda (n : ℕ) : Irrational (Real.sqrt (3 * n^2 + 2 * n + 2)) := by
  sorry

end NUMINAMATH_CALUDE_irrationality_of_lambda_l313_31396


namespace NUMINAMATH_CALUDE_robins_gum_increase_l313_31338

/-- Given Robin's initial and final gum counts, prove the number of pieces her brother gave her. -/
theorem robins_gum_increase (initial final brother_gave : ℕ) 
  (h1 : initial = 63)
  (h2 : final = 159)
  (h3 : final = initial + brother_gave) :
  brother_gave = 96 := by
  sorry

end NUMINAMATH_CALUDE_robins_gum_increase_l313_31338


namespace NUMINAMATH_CALUDE_probability_factor_less_than_10_l313_31361

def factors_of_120 : Finset ℕ := {1, 2, 3, 4, 5, 6, 8, 10, 12, 15, 20, 24, 30, 40, 60, 120}

def factors_less_than_10 : Finset ℕ := {1, 2, 3, 4, 5, 6, 8, 9}

theorem probability_factor_less_than_10 : 
  (factors_less_than_10.card : ℚ) / (factors_of_120.card : ℚ) = 1/2 := by
sorry

end NUMINAMATH_CALUDE_probability_factor_less_than_10_l313_31361


namespace NUMINAMATH_CALUDE_angle_equation_solutions_l313_31323

theorem angle_equation_solutions (θ : Real) : 
  0 ≤ θ ∧ θ ≤ π ∧ Real.sqrt 2 * (Real.cos (2 * θ)) = Real.cos θ + Real.sin θ → 
  θ = π / 12 ∨ θ = 3 * π / 4 := by
  sorry

end NUMINAMATH_CALUDE_angle_equation_solutions_l313_31323


namespace NUMINAMATH_CALUDE_complement_of_M_l313_31364

-- Define the universal set U
def U : Set ℕ := {1, 2, 3}

-- Define the set M
def M : Set ℕ := {x ∈ U | x^2 - 3*x + 2 = 0}

-- State the theorem
theorem complement_of_M (x : ℕ) : x ∈ (U \ M) ↔ x = 3 := by
  sorry

end NUMINAMATH_CALUDE_complement_of_M_l313_31364


namespace NUMINAMATH_CALUDE_dots_not_visible_is_81_l313_31326

/-- The number of faces on each die -/
def faces_per_die : ℕ := 6

/-- The number of dice -/
def num_dice : ℕ := 5

/-- The list of visible numbers on the dice -/
def visible_numbers : List ℕ := [1, 2, 3, 1, 4, 5, 6, 2]

/-- The total number of dots on all dice -/
def total_dots : ℕ := num_dice * (faces_per_die * (faces_per_die + 1) / 2)

/-- The sum of visible numbers -/
def sum_visible : ℕ := visible_numbers.sum

/-- Theorem: The number of dots not visible is 81 -/
theorem dots_not_visible_is_81 : total_dots - sum_visible = 81 := by
  sorry

end NUMINAMATH_CALUDE_dots_not_visible_is_81_l313_31326


namespace NUMINAMATH_CALUDE_apples_in_basket_l313_31333

/-- Calculates the number of apples remaining in a basket after removals. -/
def remaining_apples (initial : ℕ) (ricki_removal : ℕ) : ℕ :=
  initial - (ricki_removal + 2 * ricki_removal)

/-- Theorem stating that given the initial conditions, 32 apples remain. -/
theorem apples_in_basket : remaining_apples 74 14 = 32 := by
  sorry

end NUMINAMATH_CALUDE_apples_in_basket_l313_31333


namespace NUMINAMATH_CALUDE_eating_competition_time_l313_31378

/-- Represents the number of minutes it takes to eat everything -/
def total_time : ℕ := 48

/-- Represents the number of jars of honey eaten by Carlson -/
def carlson_honey : ℕ := 8

/-- Represents the number of jars of jam eaten by Carlson -/
def carlson_jam : ℕ := 4

/-- The time it takes Carlson to eat a jar of jam -/
def carlson_jam_time : ℕ := 2

/-- The time it takes Winnie the Pooh to eat a jar of jam -/
def pooh_jam_time : ℕ := 7

/-- The time it takes Winnie the Pooh to eat a pot of honey -/
def pooh_honey_time : ℕ := 3

/-- The time it takes Carlson to eat a pot of honey -/
def carlson_honey_time : ℕ := 5

/-- The total number of jars of jam and pots of honey -/
def total_jars : ℕ := 10

theorem eating_competition_time :
  carlson_honey * carlson_honey_time + carlson_jam * carlson_jam_time = total_time ∧
  (total_jars - carlson_honey) * pooh_honey_time + (total_jars - carlson_jam) * pooh_jam_time = total_time ∧
  carlson_honey + carlson_jam ≤ total_jars :=
by sorry

end NUMINAMATH_CALUDE_eating_competition_time_l313_31378


namespace NUMINAMATH_CALUDE_sum_of_x_solutions_l313_31336

theorem sum_of_x_solutions (y : ℝ) (x : ℝ → Prop) : 
  y = 5 → 
  (∀ x', x x' ↔ x'^2 + y^2 + 2*x' - 4*y = 80) → 
  (∃ a b, (x a ∧ x b) ∧ (∀ c, x c → (c = a ∨ c = b)) ∧ (a + b = -2)) :=
by sorry

end NUMINAMATH_CALUDE_sum_of_x_solutions_l313_31336


namespace NUMINAMATH_CALUDE_max_cherries_proof_l313_31357

/-- Represents the number of fruits Alice can buy -/
structure FruitPurchase where
  apples : ℕ
  bananas : ℕ
  cherries : ℕ

/-- Checks if a purchase satisfies all conditions -/
def isValidPurchase (p : FruitPurchase) : Prop :=
  p.apples ≥ 1 ∧ p.bananas ≥ 1 ∧ p.cherries ≥ 1 ∧
  2 * p.apples + 5 * p.bananas + 10 * p.cherries = 100

/-- The maximum number of cherries Alice can purchase -/
def maxCherries : ℕ := 8

theorem max_cherries_proof :
  (∃ p : FruitPurchase, isValidPurchase p ∧ p.cherries = maxCherries) ∧
  (∀ p : FruitPurchase, isValidPurchase p → p.cherries ≤ maxCherries) :=
sorry

end NUMINAMATH_CALUDE_max_cherries_proof_l313_31357


namespace NUMINAMATH_CALUDE_num_positive_divisors_180_l313_31337

/-- The number of positive divisors of a natural number -/
def numPositiveDivisors (n : ℕ) : ℕ := sorry

/-- The prime factorization of 180 -/
def primeFactorization180 : List (ℕ × ℕ) := [(2, 2), (3, 2), (5, 1)]

/-- Theorem: The number of positive divisors of 180 is 18 -/
theorem num_positive_divisors_180 : numPositiveDivisors 180 = 18 := by sorry

end NUMINAMATH_CALUDE_num_positive_divisors_180_l313_31337


namespace NUMINAMATH_CALUDE_inequality_solution_l313_31370

theorem inequality_solution (x : ℝ) : 
  1 / (x^2 + 1) > 3 / x + 17 / 10 ↔ -2 < x ∧ x < 0 :=
by sorry

end NUMINAMATH_CALUDE_inequality_solution_l313_31370


namespace NUMINAMATH_CALUDE_line_k_equation_l313_31330

/-- Given two lines in the xy-plane and conditions for a third line K, prove that
    the equation y = (4/15)x + (89/15) satisfies all conditions for line K. -/
theorem line_k_equation (x y : ℝ) : 
  let line1 : ℝ → ℝ := λ x => (4/5) * x + 3
  let line2 : ℝ → ℝ := λ x => (3/4) * x + 5
  let lineK : ℝ → ℝ := λ x => (4/15) * x + (89/15)
  (∀ x, lineK x = (1/3) * (line1 x - 3) + 3 * 3) ∧ 
  (lineK 4 = line2 4) ∧ 
  (lineK 4 = 7) := by
  sorry


end NUMINAMATH_CALUDE_line_k_equation_l313_31330


namespace NUMINAMATH_CALUDE_usual_time_calculation_l313_31327

theorem usual_time_calculation (T : ℝ) 
  (h1 : T > 0) 
  (h2 : (1 : ℝ) / 0.25 = (T + 24) / T) : T = 8 := by
  sorry

end NUMINAMATH_CALUDE_usual_time_calculation_l313_31327


namespace NUMINAMATH_CALUDE_sum_of_squares_of_roots_l313_31345

theorem sum_of_squares_of_roots (p q : ℝ) (a b : ℝ) 
  (h : ∀ x, 3 * x^2 - 2 * p * x + q = 0 ↔ x = a ∨ x = b) :
  a^2 + b^2 = 4 * p^2 - 6 * q := by
  sorry

end NUMINAMATH_CALUDE_sum_of_squares_of_roots_l313_31345


namespace NUMINAMATH_CALUDE_anna_candy_count_l313_31329

theorem anna_candy_count (initial_candies : ℕ) (received_candies : ℕ) :
  initial_candies = 5 →
  received_candies = 86 →
  initial_candies + received_candies = 91 := by
  sorry

end NUMINAMATH_CALUDE_anna_candy_count_l313_31329


namespace NUMINAMATH_CALUDE_grid_coverage_iff_divisible_by_four_l313_31398

/-- A T-tetromino is a set of four cells in the shape of a "T" -/
def TTetromino : Type := Unit

/-- Represents the property of an n × n grid being completely covered by T-tetrominoes without overlapping -/
def is_completely_covered (n : ℕ) : Prop := sorry

theorem grid_coverage_iff_divisible_by_four (n : ℕ) : 
  is_completely_covered n ↔ 4 ∣ n :=
sorry

end NUMINAMATH_CALUDE_grid_coverage_iff_divisible_by_four_l313_31398


namespace NUMINAMATH_CALUDE_solution_triples_l313_31359

def f (x : ℤ) : ℤ := x^3 - 4*x^2 - 16*x + 60

theorem solution_triples : 
  ∀ x y z : ℤ, 
    f x = y ∧ f y = z ∧ f z = x → 
      (x = 3 ∧ y = 3 ∧ z = 3) ∨ 
      (x = -4 ∧ y = -4 ∧ z = -4) ∨ 
      (x = 5 ∧ y = 5 ∧ z = 5) := by
  sorry

end NUMINAMATH_CALUDE_solution_triples_l313_31359


namespace NUMINAMATH_CALUDE_cards_left_calculation_l313_31367

def initial_cards : ℕ := 455
def cards_given_away : ℕ := 301

theorem cards_left_calculation : initial_cards - cards_given_away = 154 := by
  sorry

end NUMINAMATH_CALUDE_cards_left_calculation_l313_31367


namespace NUMINAMATH_CALUDE_binary_101101_equals_octal_55_l313_31325

/-- Converts a binary number represented as a list of bits to its decimal equivalent -/
def binary_to_decimal (bits : List Bool) : ℕ :=
  bits.foldl (fun acc b => 2 * acc + if b then 1 else 0) 0

/-- Converts a decimal number to its octal representation -/
def decimal_to_octal (n : ℕ) : List ℕ :=
  if n < 8 then [n]
  else (n % 8) :: decimal_to_octal (n / 8)

/-- The binary representation of 101101 -/
def binary_101101 : List Bool := [true, false, true, true, false, true]

theorem binary_101101_equals_octal_55 :
  decimal_to_octal (binary_to_decimal binary_101101) = [5, 5] := by
  sorry

end NUMINAMATH_CALUDE_binary_101101_equals_octal_55_l313_31325


namespace NUMINAMATH_CALUDE_sachin_age_l313_31340

theorem sachin_age (sachin_age rahul_age : ℕ) 
  (h1 : rahul_age = sachin_age + 7)
  (h2 : sachin_age * 12 = rahul_age * 5) : 
  sachin_age = 5 := by
  sorry

end NUMINAMATH_CALUDE_sachin_age_l313_31340


namespace NUMINAMATH_CALUDE_correct_propositions_l313_31318

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relationships between lines and planes
variable (subset : Line → Plane → Prop)
variable (perp : Line → Plane → Prop)
variable (parallel_lines : Line → Plane → Prop)
variable (parallel_planes : Plane → Plane → Prop)

-- State the theorem
theorem correct_propositions
  (m n : Line) (α β : Plane) 
  (h_distinct_lines : m ≠ n)
  (h_distinct_planes : α ≠ β) :
  -- Proposition 2
  (parallel_planes α β ∧ subset m α → parallel_lines m β) ∧
  -- Proposition 3
  (perp n α ∧ perp n β ∧ perp m α → perp m β) :=
by sorry

end NUMINAMATH_CALUDE_correct_propositions_l313_31318


namespace NUMINAMATH_CALUDE_f_symmetry_l313_31334

-- Define a convex polygon as a list of vectors
def ConvexPolygon := List (ℝ × ℝ)

-- Define the projection function
def projection (v : ℝ × ℝ) (line : ℝ × ℝ) : ℝ := sorry

-- Define the function f
def f (P Q : ConvexPolygon) : ℝ :=
  List.sum (List.map (λ p => 
    (norm p) * (List.sum (List.map (λ q => abs (projection q p)) Q))
  ) P)

-- State the theorem
theorem f_symmetry (P Q : ConvexPolygon) : f P Q = f Q P := by sorry

end NUMINAMATH_CALUDE_f_symmetry_l313_31334


namespace NUMINAMATH_CALUDE_problem_book_solution_l313_31376

/-- The number of problems solved by Taeyeon and Yura -/
def total_problems_solved (taeyeon_per_day : ℕ) (taeyeon_days : ℕ) (yura_per_day : ℕ) (yura_days : ℕ) : ℕ :=
  taeyeon_per_day * taeyeon_days + yura_per_day * yura_days

/-- Theorem stating that Taeyeon and Yura solved 262 problems in total -/
theorem problem_book_solution :
  total_problems_solved 16 7 25 6 = 262 := by
  sorry

end NUMINAMATH_CALUDE_problem_book_solution_l313_31376


namespace NUMINAMATH_CALUDE_nathaniel_ticket_distribution_l313_31355

/-- The number of tickets Nathaniel gives to each of his best friends -/
def tickets_per_friend (initial_tickets : ℕ) (remaining_tickets : ℕ) (num_friends : ℕ) : ℕ :=
  (initial_tickets - remaining_tickets) / num_friends

/-- Proof that Nathaniel gave 2 tickets to each of his best friends -/
theorem nathaniel_ticket_distribution :
  tickets_per_friend 11 3 4 = 2 := by
  sorry

end NUMINAMATH_CALUDE_nathaniel_ticket_distribution_l313_31355


namespace NUMINAMATH_CALUDE_exists_quadratic_function_with_conditions_l313_31311

/-- A quadratic function with coefficient a, b, and c -/
def QuadraticFunction (a b c : ℝ) : ℝ → ℝ := fun x ↦ a * x^2 + b * x + c

/-- The vertex of a quadratic function is on the negative half of the y-axis -/
def VertexOnNegativeYAxis (a b c : ℝ) : Prop :=
  b = 0 ∧ c < 0

/-- The part of the quadratic function to the left of its axis of symmetry is rising -/
def LeftPartRising (a b c : ℝ) : Prop :=
  a < 0

/-- Theorem stating the existence of a quadratic function satisfying the given conditions -/
theorem exists_quadratic_function_with_conditions : ∃ a b c : ℝ,
  VertexOnNegativeYAxis a b c ∧
  LeftPartRising a b c ∧
  QuadraticFunction a b c = QuadraticFunction (-1) 0 (-1) :=
sorry

end NUMINAMATH_CALUDE_exists_quadratic_function_with_conditions_l313_31311


namespace NUMINAMATH_CALUDE_total_matches_proof_l313_31399

def grade1_classes : ℕ := 5
def grade2_classes : ℕ := 7
def grade3_classes : ℕ := 4

def matches_in_tournament (n : ℕ) : ℕ := n * (n - 1) / 2

theorem total_matches_proof :
  matches_in_tournament grade1_classes +
  matches_in_tournament grade2_classes +
  matches_in_tournament grade3_classes = 37 := by
sorry

end NUMINAMATH_CALUDE_total_matches_proof_l313_31399


namespace NUMINAMATH_CALUDE_first_super_lucky_year_l313_31304

def is_valid_date (month day year : ℕ) : Prop :=
  1 ≤ month ∧ month ≤ 12 ∧ 1 ≤ day ∧ day ≤ 31 ∧ year > 2000

def is_super_lucky_date (month day year : ℕ) : Prop :=
  is_valid_date month day year ∧ month * day = year % 100

def has_two_super_lucky_dates (year : ℕ) : Prop :=
  ∃ (m1 d1 m2 d2 : ℕ), 
    is_super_lucky_date m1 d1 year ∧ 
    is_super_lucky_date m2 d2 year ∧ 
    (m1 ≠ m2 ∨ d1 ≠ d2)

theorem first_super_lucky_year : 
  (∀ y, 2000 < y ∧ y < 2004 → ¬ has_two_super_lucky_dates y) ∧ 
  has_two_super_lucky_dates 2004 :=
sorry

end NUMINAMATH_CALUDE_first_super_lucky_year_l313_31304


namespace NUMINAMATH_CALUDE_parabola_c_value_l313_31347

/-- A parabola with equation y = ax^2 + bx + c, vertex at (-3, -5), and passing through (-1, -4) -/
def Parabola (a b c : ℚ) : Prop :=
  ∀ x y : ℚ, y = a * x^2 + b * x + c →
  (∃ t : ℚ, y = a * (x + 3)^2 - 5) ∧  -- vertex form
  (-4 : ℚ) = a * (-1 + 3)^2 - 5       -- passes through (-1, -4)

/-- The value of c for the given parabola is -11/4 -/
theorem parabola_c_value (a b c : ℚ) (h : Parabola a b c) : c = -11/4 := by
  sorry

end NUMINAMATH_CALUDE_parabola_c_value_l313_31347


namespace NUMINAMATH_CALUDE_cos_135_degrees_l313_31306

theorem cos_135_degrees :
  Real.cos (135 * π / 180) = -Real.sqrt 2 / 2 := by
  sorry

end NUMINAMATH_CALUDE_cos_135_degrees_l313_31306


namespace NUMINAMATH_CALUDE_right_triangle_area_l313_31381

theorem right_triangle_area (a b c : ℝ) (h1 : a = 24) (h2 : c = 25) (h3 : a^2 + b^2 = c^2) :
  (1/2) * a * b = 84 := by
sorry

end NUMINAMATH_CALUDE_right_triangle_area_l313_31381


namespace NUMINAMATH_CALUDE_complex_number_simplification_l313_31346

theorem complex_number_simplification :
  ∀ (i : ℂ), i^2 = -1 →
  3*i * (2 - 5*i) - (4 - 7*i) = 11 + 13*i :=
by
  sorry

end NUMINAMATH_CALUDE_complex_number_simplification_l313_31346


namespace NUMINAMATH_CALUDE_first_three_digits_of_expression_l313_31316

-- Define the expression
def expression : ℝ := (10^2003 + 1)^(11/9)

-- Define a function to get the first three digits after the decimal point
def firstThreeDecimalDigits (x : ℝ) : ℕ × ℕ × ℕ := sorry

-- Theorem statement
theorem first_three_digits_of_expression :
  firstThreeDecimalDigits expression = (2, 2, 2) := by sorry

end NUMINAMATH_CALUDE_first_three_digits_of_expression_l313_31316


namespace NUMINAMATH_CALUDE_not_prime_for_all_positive_n_l313_31372

def f (n : ℕ+) : ℤ := (n : ℤ)^3 - 9*(n : ℤ)^2 + 23*(n : ℤ) - 17

theorem not_prime_for_all_positive_n : ∀ n : ℕ+, ¬(Nat.Prime (Int.natAbs (f n))) := by
  sorry

end NUMINAMATH_CALUDE_not_prime_for_all_positive_n_l313_31372


namespace NUMINAMATH_CALUDE_complex_sum_zero_l313_31324

theorem complex_sum_zero : 
  let x : ℂ := 2 * Complex.I / (1 - Complex.I)
  let n : ℕ := 2016
  (Finset.sum (Finset.range n) (fun k => Nat.choose n (k + 1) * x ^ (k + 1))) = 0 := by
  sorry

end NUMINAMATH_CALUDE_complex_sum_zero_l313_31324


namespace NUMINAMATH_CALUDE_mrs_hilt_remaining_money_l313_31358

/-- Calculates the remaining money after purchases -/
def remaining_money (initial : ℚ) (pencil notebook pens : ℚ) : ℚ :=
  initial - (pencil + notebook + pens)

/-- Proves that Mrs. Hilt's remaining money is $3.00 -/
theorem mrs_hilt_remaining_money :
  remaining_money 12.5 1.25 3.45 4.8 = 3 := by
  sorry

end NUMINAMATH_CALUDE_mrs_hilt_remaining_money_l313_31358


namespace NUMINAMATH_CALUDE_smallest_a_in_special_progression_l313_31339

theorem smallest_a_in_special_progression (a b c : ℤ) 
  (h1 : a < b) (h2 : b < c)
  (h3 : 2 * b = a + c)  -- arithmetic progression
  (h4 : a * a = c * b)  -- geometric progression
  : a ≥ 1 ∧ ∃ (a₀ b₀ c₀ : ℤ), a₀ = 1 ∧ 
    a₀ < b₀ ∧ b₀ < c₀ ∧ 
    2 * b₀ = a₀ + c₀ ∧
    a₀ * a₀ = c₀ * b₀ := by
  sorry

end NUMINAMATH_CALUDE_smallest_a_in_special_progression_l313_31339


namespace NUMINAMATH_CALUDE_equilateral_triangle_l313_31385

theorem equilateral_triangle (a b c : ℝ) 
  (h1 : a + b - c = 2) 
  (h2 : 2 * a * b - c^2 = 4) : 
  a = 2 ∧ b = 2 ∧ c = 2 := by
  sorry

end NUMINAMATH_CALUDE_equilateral_triangle_l313_31385


namespace NUMINAMATH_CALUDE_rhombus_diagonal_l313_31309

/-- Given a rhombus with area 432 sq m and one diagonal 24 m, prove the other diagonal is 36 m -/
theorem rhombus_diagonal (area : ℝ) (diagonal2 : ℝ) (diagonal1 : ℝ) : 
  area = 432 → diagonal2 = 24 → area = (diagonal1 * diagonal2) / 2 → diagonal1 = 36 :=
by sorry

end NUMINAMATH_CALUDE_rhombus_diagonal_l313_31309


namespace NUMINAMATH_CALUDE_not_equivalent_statement_and_converse_l313_31342

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relations
variable (subset : Line → Plane → Prop)
variable (perp : Line → Plane → Prop)
variable (perpPlanes : Plane → Plane → Prop)

-- Define the given lines and planes
variable (a b c : Line)
variable (α β : Plane)

-- State the theorem
theorem not_equivalent_statement_and_converse :
  (b ≠ a ∧ c ≠ a ∧ c ≠ b) →  -- three different lines
  (α ≠ β) →  -- two different planes
  (subset b α) →  -- b is a subset of α
  (¬ subset c α) →  -- c is not a subset of α
  ¬ (((perp b β → perpPlanes α β) ↔ (perpPlanes α β → perp b β))) :=
by sorry

end NUMINAMATH_CALUDE_not_equivalent_statement_and_converse_l313_31342


namespace NUMINAMATH_CALUDE_half_power_inequality_l313_31365

theorem half_power_inequality (a b : ℝ) (h : a > b) : (1/2 : ℝ)^b > (1/2 : ℝ)^a := by
  sorry

end NUMINAMATH_CALUDE_half_power_inequality_l313_31365


namespace NUMINAMATH_CALUDE_book_cost_in_rubles_l313_31389

/-- Represents the exchange rate between US dollars and Namibian dollars -/
def usd_to_namibian : ℚ := 10

/-- Represents the exchange rate between US dollars and Russian rubles -/
def usd_to_rubles : ℚ := 8

/-- Represents the cost of the book in Namibian dollars -/
def book_cost_namibian : ℚ := 200

/-- Theorem stating that the cost of the book in Russian rubles is 160 -/
theorem book_cost_in_rubles :
  (book_cost_namibian / usd_to_namibian) * usd_to_rubles = 160 := by
  sorry


end NUMINAMATH_CALUDE_book_cost_in_rubles_l313_31389


namespace NUMINAMATH_CALUDE_wickets_before_last_match_is_85_l313_31314

/-- Represents a bowler's statistics -/
structure BowlerStats where
  initialAverage : ℝ
  lastMatchWickets : ℕ
  lastMatchRuns : ℕ
  averageDecrease : ℝ

/-- Calculates the number of wickets taken before the last match -/
def wicketsBeforeLastMatch (stats : BowlerStats) : ℕ :=
  sorry

/-- Theorem stating that for the given conditions, the number of wickets before the last match is 85 -/
theorem wickets_before_last_match_is_85 (stats : BowlerStats) 
  (h1 : stats.initialAverage = 12.4)
  (h2 : stats.lastMatchWickets = 5)
  (h3 : stats.lastMatchRuns = 26)
  (h4 : stats.averageDecrease = 0.4) :
  wicketsBeforeLastMatch stats = 85 :=
by sorry

end NUMINAMATH_CALUDE_wickets_before_last_match_is_85_l313_31314


namespace NUMINAMATH_CALUDE_exam_maximum_marks_l313_31384

theorem exam_maximum_marks :
  let pass_percentage : ℚ := 45 / 100
  let fail_score : ℕ := 180
  let fail_margin : ℕ := 45
  let max_marks : ℕ := 500
  (pass_percentage * max_marks = fail_score + fail_margin) ∧
  (pass_percentage * max_marks = (fail_score + fail_margin : ℚ)) :=
by sorry

end NUMINAMATH_CALUDE_exam_maximum_marks_l313_31384


namespace NUMINAMATH_CALUDE_trigonometric_simplification_l313_31348

theorem trigonometric_simplification :
  let numerator := Real.sin (15 * π / 180) + Real.sin (25 * π / 180) + 
                   Real.sin (35 * π / 180) + Real.sin (45 * π / 180) + 
                   Real.sin (55 * π / 180) + Real.sin (65 * π / 180) + 
                   Real.sin (75 * π / 180) + Real.sin (85 * π / 180)
  let denominator := Real.cos (10 * π / 180) * Real.cos (15 * π / 180) * Real.cos (25 * π / 180)
  numerator / denominator = 4 * Real.sin (50 * π / 180) := by
  sorry

end NUMINAMATH_CALUDE_trigonometric_simplification_l313_31348


namespace NUMINAMATH_CALUDE_cake_distribution_l313_31308

theorem cake_distribution (total_cakes : ℕ) (num_children : ℕ) : 
  total_cakes = 18 → num_children = 3 → 
  ∃ (oldest middle youngest : ℕ),
    oldest = (2 * total_cakes / 5 : ℕ) ∧
    middle = total_cakes / 3 ∧
    youngest = total_cakes - (oldest + middle) ∧
    oldest = 7 ∧ middle = 6 ∧ youngest = 5 := by
  sorry

#check cake_distribution

end NUMINAMATH_CALUDE_cake_distribution_l313_31308


namespace NUMINAMATH_CALUDE_building_floors_upper_bound_l313_31332

theorem building_floors_upper_bound 
  (num_elevators : ℕ) 
  (floors_per_elevator : ℕ) 
  (h1 : num_elevators = 7)
  (h2 : floors_per_elevator = 6)
  (h3 : ∀ (f1 f2 : ℕ), f1 ≠ f2 → ∃ (e : ℕ), e ≤ num_elevators ∧ 
    (∃ (s : Finset ℕ), s.card = floors_per_elevator ∧ f1 ∈ s ∧ f2 ∈ s)) :
  ∃ (max_floors : ℕ), max_floors ≤ 14 ∧ 
    ∀ (n : ℕ), (∀ (f1 f2 : ℕ), f1 ≤ n ∧ f2 ≤ n ∧ f1 ≠ f2 → 
      ∃ (e : ℕ), e ≤ num_elevators ∧ 
        (∃ (s : Finset ℕ), s.card = floors_per_elevator ∧ f1 ∈ s ∧ f2 ∈ s)) → 
    n ≤ max_floors := by
  sorry

end NUMINAMATH_CALUDE_building_floors_upper_bound_l313_31332


namespace NUMINAMATH_CALUDE_sum_of_possible_sums_l313_31356

theorem sum_of_possible_sums (n : ℕ) (h : n = 9) : 
  (n * (n * (n + 1) / 2) - (n * (n + 1) / 2)) = 360 := by
  sorry

#check sum_of_possible_sums

end NUMINAMATH_CALUDE_sum_of_possible_sums_l313_31356


namespace NUMINAMATH_CALUDE_class_trip_cost_l313_31362

/-- Calculates the total cost of a class trip to a science museum --/
def total_cost (num_students : ℕ) (num_teachers : ℕ) (student_ticket_price : ℚ) 
  (teacher_ticket_price : ℚ) (discount_rate : ℚ) (min_group_size : ℕ) 
  (bus_fee : ℚ) (meal_price : ℚ) : ℚ :=
  let total_people := num_students + num_teachers
  let ticket_cost := num_students * student_ticket_price + num_teachers * teacher_ticket_price
  let discounted_ticket_cost := 
    if total_people ≥ min_group_size 
    then ticket_cost * (1 - discount_rate) 
    else ticket_cost
  let meal_cost := meal_price * total_people
  discounted_ticket_cost + bus_fee + meal_cost

/-- Theorem stating the total cost for the class trip --/
theorem class_trip_cost : 
  total_cost 30 4 8 12 0.2 25 150 10 = 720.4 := by
  sorry

end NUMINAMATH_CALUDE_class_trip_cost_l313_31362


namespace NUMINAMATH_CALUDE_smallest_multiplier_for_perfect_square_l313_31392

theorem smallest_multiplier_for_perfect_square : 
  ∃ (n : ℕ), n > 0 ∧ ∃ (m : ℕ), 1008 * n = m^2 ∧ ∀ (k : ℕ), k > 0 → k < n → ¬∃ (l : ℕ), 1008 * k = l^2 :=
by sorry

end NUMINAMATH_CALUDE_smallest_multiplier_for_perfect_square_l313_31392


namespace NUMINAMATH_CALUDE_parity_relation_l313_31354

theorem parity_relation (a b : ℤ) : 
  (Even (5*b + a) → Even (a - 3*b)) ∧ 
  (Odd (5*b + a) → Odd (a - 3*b)) := by sorry

end NUMINAMATH_CALUDE_parity_relation_l313_31354


namespace NUMINAMATH_CALUDE_point_on_graph_and_coordinate_sum_l313_31374

theorem point_on_graph_and_coordinate_sum 
  (f : ℝ → ℝ) 
  (h : f 6 = 10) : 
  ∃ (x y : ℝ), 
    x = 2 ∧ 
    y = 28.5 ∧ 
    2 * y = 5 * f (3 * x) + 7 ∧ 
    x + y = 30.5 := by
  sorry

end NUMINAMATH_CALUDE_point_on_graph_and_coordinate_sum_l313_31374


namespace NUMINAMATH_CALUDE_sufficient_condition_range_l313_31331

theorem sufficient_condition_range (a : ℝ) : 
  (∀ x : ℝ, |x - a| ≤ 1 → x^2 - 5*x + 4 ≤ 0) → 
  2 ≤ a ∧ a ≤ 3 := by
  sorry

end NUMINAMATH_CALUDE_sufficient_condition_range_l313_31331


namespace NUMINAMATH_CALUDE_polynomial_divisibility_implies_perfect_powers_l313_31375

/-- Given a polynomial ax³ + 3bx² + 3cx + d that is divisible by ax² + 2bx + c,
    prove that it's a perfect cube and the divisor is a perfect square. -/
theorem polynomial_divisibility_implies_perfect_powers
  (a b c d : ℝ) (h : a ≠ 0) :
  (∃ (q : ℝ → ℝ), ∀ x, a * x^3 + 3*b * x^2 + 3*c * x + d = (a * x^2 + 2*b * x + c) * q x) →
  (∃ y, a * x^3 + 3*b * x^2 + 3*c * x + d = (a * x + y)^3) ∧
  (∃ z, a * x^2 + 2*b * x + c = (a * x + z)^2) ∧
  c = 2 * b^2 / a ∧
  d = 2 * b^3 / a^2 :=
by sorry

end NUMINAMATH_CALUDE_polynomial_divisibility_implies_perfect_powers_l313_31375


namespace NUMINAMATH_CALUDE_modulus_of_complex_number_l313_31303

theorem modulus_of_complex_number : 
  let i : ℂ := Complex.I
  let z : ℂ := 2 / (1 + i) + (1 - i)^2
  Complex.abs z = Real.sqrt 10 := by
sorry

end NUMINAMATH_CALUDE_modulus_of_complex_number_l313_31303


namespace NUMINAMATH_CALUDE_sqrt_meaningful_condition_l313_31349

theorem sqrt_meaningful_condition (x : ℝ) : 
  (∃ y : ℝ, y ^ 2 = x - 4) ↔ x ≥ 4 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_meaningful_condition_l313_31349


namespace NUMINAMATH_CALUDE_five_letter_words_same_ends_l313_31301

/-- The number of letters in the alphabet --/
def alphabet_size : ℕ := 26

/-- The length of the words we're considering --/
def word_length : ℕ := 5

/-- The number of freely chosen letters in each word --/
def free_letters : ℕ := word_length - 2

/-- The number of five-letter words with the same first and last letter --/
def count_words : ℕ := alphabet_size ^ (free_letters + 1)

theorem five_letter_words_same_ends :
  count_words = 456976 := by sorry

end NUMINAMATH_CALUDE_five_letter_words_same_ends_l313_31301


namespace NUMINAMATH_CALUDE_sum_of_squared_roots_l313_31382

theorem sum_of_squared_roots (p q r : ℝ) : 
  (3 * p^3 - 2 * p^2 + 5 * p + 15 = 0) ∧
  (3 * q^3 - 2 * q^2 + 5 * q + 15 = 0) ∧
  (3 * r^3 - 2 * r^2 + 5 * r + 15 = 0) →
  p^2 + q^2 + r^2 = -26/9 := by
sorry

end NUMINAMATH_CALUDE_sum_of_squared_roots_l313_31382


namespace NUMINAMATH_CALUDE_parallel_vectors_m_value_l313_31352

/-- Given vectors a, b, and c in ℝ², prove that if a is parallel to m*b - c, then m = -3. -/
theorem parallel_vectors_m_value (a b c : ℝ × ℝ) (m : ℝ) 
    (ha : a = (2, -1))
    (hb : b = (1, 0))
    (hc : c = (1, -2))
    (h_parallel : ∃ (k : ℝ), k ≠ 0 ∧ a = k • (m • b - c)) :
  m = -3 := by
  sorry

end NUMINAMATH_CALUDE_parallel_vectors_m_value_l313_31352


namespace NUMINAMATH_CALUDE_infinite_solutions_cubic_equation_l313_31373

theorem infinite_solutions_cubic_equation :
  ∀ k : ℕ+, ∃ a b c : ℕ+,
    (a : ℤ)^3 + 1990 * (b : ℤ)^3 = (c : ℤ)^4 := by
  sorry

end NUMINAMATH_CALUDE_infinite_solutions_cubic_equation_l313_31373


namespace NUMINAMATH_CALUDE_package_weight_ratio_l313_31363

theorem package_weight_ratio (x y : ℝ) (h1 : x > y) (h2 : x > 0) (h3 : y > 0) 
  (h4 : x + y = 7 * (x - y)) : x / y = 4 / 3 := by
  sorry

end NUMINAMATH_CALUDE_package_weight_ratio_l313_31363
