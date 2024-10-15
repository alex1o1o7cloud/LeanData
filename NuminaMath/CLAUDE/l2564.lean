import Mathlib

namespace NUMINAMATH_CALUDE_odd_implies_derivative_even_exists_increasing_not_increasing_derivative_l2564_256488

-- Define a real-valued function on R
variable (f : ℝ → ℝ)

-- Define the property of being an odd function
def IsOdd (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

-- Define the property of being an even function
def IsEven (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = f x

-- Define the property of being strictly increasing
def StrictlyIncreasing (f : ℝ → ℝ) : Prop := ∀ x y, x < y → f x < f y

-- Proposition 1: If f is odd, then f' is even
theorem odd_implies_derivative_even (hf : IsOdd f) : IsEven (deriv f) := by sorry

-- Proposition 2: There exists a strictly increasing function whose derivative is not strictly increasing
theorem exists_increasing_not_increasing_derivative : 
  ∃ f : ℝ → ℝ, StrictlyIncreasing f ∧ ¬StrictlyIncreasing (deriv f) := by sorry

end NUMINAMATH_CALUDE_odd_implies_derivative_even_exists_increasing_not_increasing_derivative_l2564_256488


namespace NUMINAMATH_CALUDE_sum_of_evens_l2564_256474

theorem sum_of_evens (n : ℕ) (sum_first_n : ℕ) (first_term : ℕ) (last_term : ℕ) : 
  n = 50 → 
  sum_first_n = 2550 → 
  first_term = 102 → 
  last_term = 200 → 
  (n : ℕ) * (first_term + last_term) / 2 = 7550 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_evens_l2564_256474


namespace NUMINAMATH_CALUDE_loot_box_solution_l2564_256496

/-- Represents the loot box problem -/
def LootBoxProblem (cost_per_box : ℝ) (avg_value_per_box : ℝ) (total_spent : ℝ) : Prop :=
  let num_boxes : ℝ := total_spent / cost_per_box
  let total_avg_value : ℝ := avg_value_per_box * num_boxes
  let total_lost : ℝ := total_spent - total_avg_value
  let avg_lost_per_box : ℝ := total_lost / num_boxes
  avg_lost_per_box = 1.5

/-- Theorem stating the solution to the loot box problem -/
theorem loot_box_solution :
  LootBoxProblem 5 3.5 40 := by
  sorry

#check loot_box_solution

end NUMINAMATH_CALUDE_loot_box_solution_l2564_256496


namespace NUMINAMATH_CALUDE_apple_pear_ratio_l2564_256459

theorem apple_pear_ratio (apples oranges pears : ℕ) 
  (h1 : oranges = 3 * apples) 
  (h2 : pears = 4 * oranges) : 
  apples = (1 : ℚ) / 12 * pears :=
by sorry

end NUMINAMATH_CALUDE_apple_pear_ratio_l2564_256459


namespace NUMINAMATH_CALUDE_halfway_point_between_fractions_l2564_256486

theorem halfway_point_between_fractions :
  (1 / 12 + 1 / 15) / 2 = 3 / 40 := by
  sorry

end NUMINAMATH_CALUDE_halfway_point_between_fractions_l2564_256486


namespace NUMINAMATH_CALUDE_village_population_percentage_l2564_256472

theorem village_population_percentage (total : ℕ) (part : ℕ) (h1 : total = 9000) (h2 : part = 8100) :
  (part : ℚ) / total * 100 = 90 := by
  sorry

end NUMINAMATH_CALUDE_village_population_percentage_l2564_256472


namespace NUMINAMATH_CALUDE_pens_before_discount_is_75_l2564_256441

/-- The number of pens that can be bought before the discount -/
def pens_before_discount : ℕ := 75

/-- The discount rate -/
def discount_rate : ℚ := 1/4

/-- The number of additional pens that can be bought after the discount -/
def additional_pens : ℕ := 25

theorem pens_before_discount_is_75 :
  pens_before_discount = 75 ∧
  discount_rate = 1/4 ∧
  additional_pens = 25 ∧
  (pens_before_discount : ℚ) = (pens_before_discount + additional_pens) * (1 - discount_rate) :=
by sorry

end NUMINAMATH_CALUDE_pens_before_discount_is_75_l2564_256441


namespace NUMINAMATH_CALUDE_product_of_difference_and_sum_of_squares_l2564_256419

theorem product_of_difference_and_sum_of_squares (a b : ℝ) 
  (h1 : a - b = 6) 
  (h2 : a^2 + b^2 = 50) : 
  a * b = 7 := by
sorry

end NUMINAMATH_CALUDE_product_of_difference_and_sum_of_squares_l2564_256419


namespace NUMINAMATH_CALUDE_journey_speed_problem_l2564_256499

theorem journey_speed_problem (total_distance : ℝ) (total_time : ℝ) 
  (speed1 : ℝ) (speed2 : ℝ) (segment_time : ℝ) :
  total_distance = 150 →
  total_time = 2 →
  speed1 = 50 →
  speed2 = 70 →
  segment_time = 2/3 →
  ∃ (speed3 : ℝ),
    speed3 = 105 ∧
    total_distance = speed1 * segment_time + speed2 * segment_time + speed3 * segment_time :=
by sorry

end NUMINAMATH_CALUDE_journey_speed_problem_l2564_256499


namespace NUMINAMATH_CALUDE_special_sequence_sum_property_l2564_256439

/-- A sequence of pairwise distinct nonnegative integers satisfying the given conditions -/
def SpecialSequence (b : ℕ → ℕ) : Prop :=
  (∀ i j, i ≠ j → b i ≠ b j) ∧ 
  (b 0 = 0) ∧ 
  (∀ n > 0, b n < 2 * n)

/-- The main theorem -/
theorem special_sequence_sum_property (b : ℕ → ℕ) (h : SpecialSequence b) :
  ∀ m : ℕ, ∃ k ℓ : ℕ, b k + b ℓ = m := by
  sorry

end NUMINAMATH_CALUDE_special_sequence_sum_property_l2564_256439


namespace NUMINAMATH_CALUDE_gcf_of_104_and_156_l2564_256494

theorem gcf_of_104_and_156 : Nat.gcd 104 156 = 52 := by
  sorry

end NUMINAMATH_CALUDE_gcf_of_104_and_156_l2564_256494


namespace NUMINAMATH_CALUDE_second_race_length_is_600_l2564_256434

/-- Represents a race between three runners A, B, and C -/
structure Race where
  length : ℝ
  a_beats_b : ℝ
  a_beats_c : ℝ

/-- Calculates the length of a second race given the first race data -/
def second_race_length (first_race : Race) (b_beats_c : ℝ) : ℝ :=
  600

/-- Theorem stating that given the conditions of the first race and the fact that B beats C by 60m in the second race, the length of the second race is 600m -/
theorem second_race_length_is_600 (first_race : Race) (h1 : first_race.length = 200) 
    (h2 : first_race.a_beats_b = 20) (h3 : first_race.a_beats_c = 38) (h4 : b_beats_c = 60) : 
    second_race_length first_race b_beats_c = 600 := by
  sorry

end NUMINAMATH_CALUDE_second_race_length_is_600_l2564_256434


namespace NUMINAMATH_CALUDE_differential_equation_solution_l2564_256482

/-- The differential equation dy/dx = y^2 has a general solution y = a₀ / (1 - a₀x) -/
theorem differential_equation_solution (x : ℝ) (a₀ : ℝ) :
  let y : ℝ → ℝ := λ x => a₀ / (1 - a₀ * x)
  ∀ x, (deriv y) x = (y x)^2 :=
by sorry

end NUMINAMATH_CALUDE_differential_equation_solution_l2564_256482


namespace NUMINAMATH_CALUDE_five_digit_number_divisibility_l2564_256408

theorem five_digit_number_divisibility (U : ℕ) : 
  U < 10 →
  (2018 * 10 + U) % 9 = 0 →
  (2018 * 10 + U) % 8 = 3 :=
by sorry

end NUMINAMATH_CALUDE_five_digit_number_divisibility_l2564_256408


namespace NUMINAMATH_CALUDE_all_points_above_x_axis_l2564_256429

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a parallelogram defined by four vertices -/
structure Parallelogram where
  P : Point
  Q : Point
  R : Point
  S : Point

/-- Checks if a point is above or on the x-axis -/
def isAboveOrOnXAxis (p : Point) : Prop :=
  p.y ≥ 0

/-- Checks if a point is inside or on the boundary of a parallelogram -/
def isInsideOrOnParallelogram (para : Parallelogram) (p : Point) : Prop :=
  sorry  -- Definition of this function is omitted for brevity

/-- The main theorem to be proved -/
theorem all_points_above_x_axis (para : Parallelogram) 
    (h1 : para.P = ⟨-4, 4⟩) 
    (h2 : para.Q = ⟨4, 2⟩)
    (h3 : para.R = ⟨2, -2⟩)
    (h4 : para.S = ⟨-6, -4⟩) :
    ∀ p : Point, isInsideOrOnParallelogram para p → isAboveOrOnXAxis p :=
  sorry

#check all_points_above_x_axis

end NUMINAMATH_CALUDE_all_points_above_x_axis_l2564_256429


namespace NUMINAMATH_CALUDE_one_fifths_in_one_tenth_l2564_256461

theorem one_fifths_in_one_tenth : (1 / 10 : ℚ) / (1 / 5 : ℚ) = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_one_fifths_in_one_tenth_l2564_256461


namespace NUMINAMATH_CALUDE_floor_properties_l2564_256409

-- Define the floor function
noncomputable def floor (x : ℝ) : ℤ :=
  Int.floor x

-- Theorem statement
theorem floor_properties (x y : ℝ) :
  (x - 1 < floor x) ∧
  (floor x - floor y - 1 < x - y) ∧
  (x - y < floor x - floor y + 1) ∧
  (x^2 + 1/3 > floor x) :=
by sorry

end NUMINAMATH_CALUDE_floor_properties_l2564_256409


namespace NUMINAMATH_CALUDE_teacup_lid_arrangement_l2564_256415

def teacups : ℕ := 6
def lids : ℕ := 6
def matching_lids : ℕ := 2

theorem teacup_lid_arrangement :
  (teacups.choose matching_lids) * 
  ((teacups - matching_lids - 1) * (lids - matching_lids - 1)) = 135 := by
sorry

end NUMINAMATH_CALUDE_teacup_lid_arrangement_l2564_256415


namespace NUMINAMATH_CALUDE_min_students_with_all_characteristics_l2564_256433

theorem min_students_with_all_characteristics
  (total : ℕ)
  (brown_eyes : ℕ)
  (lunch_boxes : ℕ)
  (glasses : ℕ)
  (h_total : total = 35)
  (h_brown_eyes : brown_eyes = 15)
  (h_lunch_boxes : lunch_boxes = 25)
  (h_glasses : glasses = 10) :
  ∃ (n : ℕ), n ≥ 5 ∧
    n = (brown_eyes + lunch_boxes + glasses - total).max 0 :=
by sorry

end NUMINAMATH_CALUDE_min_students_with_all_characteristics_l2564_256433


namespace NUMINAMATH_CALUDE_negation_of_universal_proposition_l2564_256498

theorem negation_of_universal_proposition :
  (¬ ∀ x : ℝ, x^2 + 5*x = 4) ↔ (∃ x : ℝ, x^2 + 5*x ≠ 4) := by sorry

end NUMINAMATH_CALUDE_negation_of_universal_proposition_l2564_256498


namespace NUMINAMATH_CALUDE_equivalent_annual_rate_l2564_256404

/-- Given an annual interest rate of 8% compounded quarterly, 
    the equivalent constant annual compounding rate is approximately 8.24% -/
theorem equivalent_annual_rate (quarterly_rate : ℝ) (annual_rate : ℝ) (r : ℝ) : 
  quarterly_rate = 0.08 / 4 →
  annual_rate = 0.08 →
  (1 + quarterly_rate)^4 = 1 + r →
  ∀ ε > 0, |r - 0.0824| < ε :=
sorry

end NUMINAMATH_CALUDE_equivalent_annual_rate_l2564_256404


namespace NUMINAMATH_CALUDE_f_is_even_and_increasing_l2564_256412

-- Define the function f(x) = |x| + 1
def f (x : ℝ) : ℝ := |x| + 1

-- State the theorem
theorem f_is_even_and_increasing :
  (∀ x : ℝ, f x = f (-x)) ∧
  (∀ x y : ℝ, 0 < x → x < y → f x < f y) :=
by sorry

end NUMINAMATH_CALUDE_f_is_even_and_increasing_l2564_256412


namespace NUMINAMATH_CALUDE_coefficient_x_squared_in_expansion_l2564_256435

theorem coefficient_x_squared_in_expansion (x : ℝ) : 
  (Finset.range 7).sum (fun k => (Nat.choose 6 k) * ((-1)^k) * ((Real.sqrt 2)^(6-k)) * (x^k)) = 
  60 * x^2 + (Finset.range 7).sum (fun k => if k ≠ 2 then (Nat.choose 6 k) * ((-1)^k) * ((Real.sqrt 2)^(6-k)) * (x^k) else 0) :=
by sorry

end NUMINAMATH_CALUDE_coefficient_x_squared_in_expansion_l2564_256435


namespace NUMINAMATH_CALUDE_min_value_problem1_min_value_problem2_l2564_256487

theorem min_value_problem1 (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : 2/x + y = 1) :
  2*x + 1/(3*y) ≥ (13 + 4*Real.sqrt 3) / 3 :=
sorry

theorem min_value_problem2 (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : x + y = 1) :
  1/(2*x) + x/(y+1) ≥ 5/4 :=
sorry

end NUMINAMATH_CALUDE_min_value_problem1_min_value_problem2_l2564_256487


namespace NUMINAMATH_CALUDE_range_of_a_l2564_256458

-- Define the sets A, B, and C
def A : Set ℝ := {x : ℝ | 1 < x ∧ x < 4}
def B : Set ℝ := {x : ℝ | 3 * x - 1 < x + 5}
def C (a : ℝ) : Set ℝ := {x : ℝ | x ≤ a}

-- Define the complement of A with respect to ℝ
def complementA : Set ℝ := {x : ℝ | x ≤ 1 ∨ x ≥ 4}

-- State the theorem
theorem range_of_a (a : ℝ) : 
  (complementA ∩ C a = C a) → a ≤ 1 := by
  sorry

end NUMINAMATH_CALUDE_range_of_a_l2564_256458


namespace NUMINAMATH_CALUDE_tan_half_alpha_eq_two_implies_ratio_l2564_256418

theorem tan_half_alpha_eq_two_implies_ratio (α : Real) 
  (h : Real.tan (α / 2) = 2) : 
  (6 * Real.sin α + Real.cos α) / (3 * Real.sin α - 2 * Real.cos α) = 7 / 6 := by
  sorry

end NUMINAMATH_CALUDE_tan_half_alpha_eq_two_implies_ratio_l2564_256418


namespace NUMINAMATH_CALUDE_cupcakes_per_box_l2564_256423

theorem cupcakes_per_box 
  (total_baked : ℕ) 
  (left_at_home : ℕ) 
  (boxes_given : ℕ) 
  (h1 : total_baked = 53) 
  (h2 : left_at_home = 2) 
  (h3 : boxes_given = 17) :
  (total_baked - left_at_home) / boxes_given = 3 := by
sorry

end NUMINAMATH_CALUDE_cupcakes_per_box_l2564_256423


namespace NUMINAMATH_CALUDE_tangent_points_satisfy_locus_l2564_256407

/-- A conic section with focus at the origin and directrix x - d = 0 -/
structure ConicSection (d : ℝ) where
  -- Point on the conic section
  x : ℝ
  y : ℝ
  -- Eccentricity
  e : ℝ
  -- Conic section equation
  eq : x^2 + y^2 = e^2 * (x - d)^2

/-- A point of tangency on the conic section -/
structure TangentPoint (d : ℝ) extends ConicSection d where
  -- Tangent line has slope 1 (parallel to y = x)
  tangent_slope : (1 - e^2) * x + y + e^2 * d = 0

/-- The locus of points of tangency -/
def locus_equation (d : ℝ) (p : ℝ × ℝ) : Prop :=
  let (x, y) := p
  y^2 - x*y + d*(x + y) = 0

/-- The main theorem: points of tangency satisfy the locus equation -/
theorem tangent_points_satisfy_locus (d : ℝ) (p : TangentPoint d) :
  locus_equation d (p.x, p.y) := by
  sorry


end NUMINAMATH_CALUDE_tangent_points_satisfy_locus_l2564_256407


namespace NUMINAMATH_CALUDE_total_evening_sales_l2564_256427

/-- Calculates the total evening sales given the conditions of the problem -/
theorem total_evening_sales :
  let remy_bottles : ℕ := 55
  let nick_bottles : ℕ := remy_bottles - 6
  let price_per_bottle : ℚ := 1/2
  let morning_sales : ℚ := (remy_bottles + nick_bottles : ℚ) * price_per_bottle
  let evening_sales : ℚ := morning_sales + 3
  evening_sales = 55 := by sorry

end NUMINAMATH_CALUDE_total_evening_sales_l2564_256427


namespace NUMINAMATH_CALUDE_eighteenth_term_is_three_l2564_256421

/-- An equal sum sequence with public sum 5 and a₁ = 2 -/
def EqualSumSequence (a : ℕ → ℕ) : Prop :=
  (∀ n, a n + a (n + 1) = 5) ∧ a 1 = 2

/-- The 18th term of the equal sum sequence is 3 -/
theorem eighteenth_term_is_three (a : ℕ → ℕ) (h : EqualSumSequence a) : a 18 = 3 := by
  sorry

end NUMINAMATH_CALUDE_eighteenth_term_is_three_l2564_256421


namespace NUMINAMATH_CALUDE_stability_comparison_A_more_stable_than_B_l2564_256476

-- Define a structure for a student's test scores
structure StudentScores where
  average : ℝ
  variance : ℝ

-- Define the stability comparison function
def more_stable (a b : StudentScores) : Prop :=
  a.average = b.average ∧ a.variance < b.variance

-- Theorem statement
theorem stability_comparison (a b : StudentScores) 
  (h_avg : a.average = b.average) 
  (h_var : a.variance < b.variance) : 
  more_stable a b := by
  sorry

-- Define students A and B
def student_A : StudentScores := { average := 88, variance := 0.61 }
def student_B : StudentScores := { average := 88, variance := 0.72 }

-- Theorem application to students A and B
theorem A_more_stable_than_B : more_stable student_A student_B := by
  sorry

end NUMINAMATH_CALUDE_stability_comparison_A_more_stable_than_B_l2564_256476


namespace NUMINAMATH_CALUDE_repeating_decimal_equals_fraction_fraction_is_lowest_terms_sum_of_numerator_and_denominator_l2564_256438

/-- The repeating decimal 0.134134134... as a real number -/
def repeating_decimal : ℝ := 0.134134134

/-- The fraction representation of the repeating decimal -/
def fraction : ℚ := 134 / 999

theorem repeating_decimal_equals_fraction : 
  repeating_decimal = fraction := by sorry

theorem fraction_is_lowest_terms : 
  Nat.gcd 134 999 = 1 := by sorry

theorem sum_of_numerator_and_denominator : 
  134 + 999 = 1133 := by sorry

#eval 134 + 999  -- To verify the result

end NUMINAMATH_CALUDE_repeating_decimal_equals_fraction_fraction_is_lowest_terms_sum_of_numerator_and_denominator_l2564_256438


namespace NUMINAMATH_CALUDE_geometric_sum_n_terms_l2564_256465

def geometric_sum (a : ℚ) (r : ℚ) (n : ℕ) : ℚ :=
  a * (1 - r^n) / (1 - r)

theorem geometric_sum_n_terms (a r : ℚ) (n : ℕ) (h1 : a = 1/3) (h2 : r = 1/3) :
  geometric_sum a r n = 80/243 ↔ n = 5 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sum_n_terms_l2564_256465


namespace NUMINAMATH_CALUDE_sin_plus_sin_alpha_nonperiodic_l2564_256445

/-- A function f is periodic if there exists a non-zero real number T such that f(x + T) = f(x) for all x -/
def IsPeriodic (f : ℝ → ℝ) : Prop :=
  ∃ T : ℝ, T ≠ 0 ∧ ∀ x : ℝ, f (x + T) = f x

/-- The main theorem: for any positive irrational α, the function f(x) = sin x + sin(αx) is non-periodic -/
theorem sin_plus_sin_alpha_nonperiodic (α : ℝ) (h_pos : α > 0) (h_irr : Irrational α) :
  ¬IsPeriodic (fun x ↦ Real.sin x + Real.sin (α * x)) := by
  sorry


end NUMINAMATH_CALUDE_sin_plus_sin_alpha_nonperiodic_l2564_256445


namespace NUMINAMATH_CALUDE_expression1_eval_expression2_eval_l2564_256402

-- Part 1
def expression1 (x : ℝ) : ℝ := -3*x^2 + 5*x - 0.5*x^2 + x - 1

theorem expression1_eval : expression1 2 = -3 := by sorry

-- Part 2
def expression2 (a b : ℝ) : ℝ := (a^2*b + 3*a*b^2) - 3*(a^2*b + a*b^2 - 1)

theorem expression2_eval : expression2 (-2) 2 = -13 := by sorry

end NUMINAMATH_CALUDE_expression1_eval_expression2_eval_l2564_256402


namespace NUMINAMATH_CALUDE_females_chose_malt_cheerleader_malt_choice_l2564_256473

/-- Represents the group of cheerleaders -/
structure CheerleaderGroup where
  total : Nat
  males : Nat
  females : Nat
  malt_choosers : Nat
  coke_choosers : Nat
  male_malt_choosers : Nat

/-- The theorem to prove -/
theorem females_chose_malt (group : CheerleaderGroup) : Nat :=
  let female_malt_choosers := group.malt_choosers - group.male_malt_choosers
  female_malt_choosers

/-- The main theorem stating the conditions and the result to prove -/
theorem cheerleader_malt_choice : ∃ (group : CheerleaderGroup), 
  group.total = 26 ∧
  group.males = 10 ∧
  group.females = 16 ∧
  group.malt_choosers = 2 * group.coke_choosers ∧
  group.male_malt_choosers = 6 ∧
  females_chose_malt group = 10 := by
  sorry


end NUMINAMATH_CALUDE_females_chose_malt_cheerleader_malt_choice_l2564_256473


namespace NUMINAMATH_CALUDE_recliner_price_drop_l2564_256471

/-- Proves that a 80% increase in sales and a 44% increase in gross revenue
    results in a 20% price drop -/
theorem recliner_price_drop (P N : ℝ) (P' N' : ℝ) :
  N' = 1.8 * N →
  P' * N' = 1.44 * (P * N) →
  P' = 0.8 * P :=
by sorry

end NUMINAMATH_CALUDE_recliner_price_drop_l2564_256471


namespace NUMINAMATH_CALUDE_cubic_inequality_solution_l2564_256446

theorem cubic_inequality_solution (x : ℝ) : 
  -2 * x^3 + 5 * x^2 + 7 * x - 10 < 0 ↔ x < -1.35 ∨ (1.85 < x ∧ x < 2) :=
by sorry

end NUMINAMATH_CALUDE_cubic_inequality_solution_l2564_256446


namespace NUMINAMATH_CALUDE_min_distance_from_start_l2564_256426

/-- Represents a robot's movement on a 2D plane. -/
structure RobotMovement where
  /-- The distance the robot moves per minute. -/
  speed : ℝ
  /-- The total number of minutes the robot moves. -/
  total_time : ℕ
  /-- The number of minutes before the robot starts turning. -/
  initial_straight_time : ℕ

/-- Theorem stating the minimum distance from the starting point after the robot's movement. -/
theorem min_distance_from_start (r : RobotMovement) 
  (h1 : r.speed = 10)
  (h2 : r.total_time = 9)
  (h3 : r.initial_straight_time = 1) :
  ∃ (d : ℝ), d = 10 ∧ ∀ (final_pos : ℝ × ℝ), 
    (final_pos.1^2 + final_pos.2^2).sqrt ≥ d :=
sorry

end NUMINAMATH_CALUDE_min_distance_from_start_l2564_256426


namespace NUMINAMATH_CALUDE_green_hats_count_l2564_256424

theorem green_hats_count (total_hats : ℕ) (blue_cost green_cost total_cost : ℕ) 
  (h1 : total_hats = 85)
  (h2 : blue_cost = 6)
  (h3 : green_cost = 7)
  (h4 : total_cost = 548) :
  ∃ (blue_hats green_hats : ℕ),
    blue_hats + green_hats = total_hats ∧
    blue_cost * blue_hats + green_cost * green_hats = total_cost ∧
    green_hats = 38 := by
sorry

end NUMINAMATH_CALUDE_green_hats_count_l2564_256424


namespace NUMINAMATH_CALUDE_burger_cost_theorem_l2564_256411

/-- The cost of a single burger given the total spent, total burgers, double burger cost, and number of double burgers --/
def single_burger_cost (total_spent : ℚ) (total_burgers : ℕ) (double_burger_cost : ℚ) (double_burgers : ℕ) : ℚ :=
  let single_burgers := total_burgers - double_burgers
  let double_burgers_cost := double_burger_cost * double_burgers
  let single_burgers_total_cost := total_spent - double_burgers_cost
  single_burgers_total_cost / single_burgers

theorem burger_cost_theorem :
  single_burger_cost 68.50 50 1.50 37 = 1.00 := by
  sorry

end NUMINAMATH_CALUDE_burger_cost_theorem_l2564_256411


namespace NUMINAMATH_CALUDE_smallest_divisible_by_15_11_12_l2564_256466

theorem smallest_divisible_by_15_11_12 : ∃ n : ℕ+, (∀ m : ℕ+, m < n → ¬(15 ∣ m ∧ 11 ∣ m ∧ 12 ∣ m)) ∧ (15 ∣ n ∧ 11 ∣ n ∧ 12 ∣ n) := by
  sorry

end NUMINAMATH_CALUDE_smallest_divisible_by_15_11_12_l2564_256466


namespace NUMINAMATH_CALUDE_square_measurement_unit_l2564_256468

/-- Given a square with sides of length 5 units and an actual area of at least 20.25 square centimeters,
    prove that the length of one unit in this measurement system is 0.9 centimeters. -/
theorem square_measurement_unit (side_length : ℝ) (actual_area : ℝ) :
  side_length = 5 →
  actual_area ≥ 20.25 →
  actual_area = (side_length * 0.9) ^ 2 :=
by sorry

end NUMINAMATH_CALUDE_square_measurement_unit_l2564_256468


namespace NUMINAMATH_CALUDE_power_sum_equality_l2564_256463

theorem power_sum_equality (a b c d : ℝ) 
  (h1 : a + b = c + d) 
  (h2 : a^3 + b^3 = c^3 + d^3) : 
  (a^5 + b^5 = c^5 + d^5) ∧ 
  (∃ a b c d : ℝ, (a + b = c + d) ∧ (a^3 + b^3 = c^3 + d^3) ∧ (a^4 + b^4 ≠ c^4 + d^4)) :=
by sorry

end NUMINAMATH_CALUDE_power_sum_equality_l2564_256463


namespace NUMINAMATH_CALUDE_linear_function_composition_l2564_256414

/-- A linear function from ℝ to ℝ -/
def LinearFunction (f : ℝ → ℝ) : Prop :=
  ∃ k b : ℝ, k ≠ 0 ∧ ∀ x, f x = k * x + b

theorem linear_function_composition (f : ℝ → ℝ) :
  LinearFunction f → (∀ x, f (f x) = 4 * x + 9) →
  (∀ x, f x = 2 * x + 3) ∨ (∀ x, f x = -2 * x - 9) :=
by sorry

end NUMINAMATH_CALUDE_linear_function_composition_l2564_256414


namespace NUMINAMATH_CALUDE_restaurant_expenditure_l2564_256470

theorem restaurant_expenditure (total_people : Nat) (standard_spenders : Nat) (standard_amount : ℝ) (total_spent : ℝ) :
  total_people = 8 →
  standard_spenders = 7 →
  standard_amount = 10 →
  total_spent = 88 →
  (total_spent - (standard_spenders * standard_amount)) - (total_spent / total_people) = 7 := by
  sorry

end NUMINAMATH_CALUDE_restaurant_expenditure_l2564_256470


namespace NUMINAMATH_CALUDE_binomial_coefficient_sum_l2564_256444

theorem binomial_coefficient_sum (a₀ a₁ a₂ a₃ a₄ a₅ a₆ a₇ : ℝ) :
  (∀ x, (1 - 2*x)^7 = a₀ + a₁*x + a₂*x^2 + a₃*x^3 + a₄*x^4 + a₅*x^5 + a₆*x^6 + a₇*x^7) →
  |a₀| + |a₁| + |a₂| + |a₃| + |a₄| + |a₅| + |a₆| + |a₇| = 2187 := by
sorry

end NUMINAMATH_CALUDE_binomial_coefficient_sum_l2564_256444


namespace NUMINAMATH_CALUDE_probability_of_valid_roll_l2564_256481

/-- A standard six-sided die -/
def Die : Type := Fin 6

/-- The set of possible outcomes when rolling two dice -/
def TwoDiceRoll : Type := Die × Die

/-- The set of valid two-digit numbers between 40 and 50 (inclusive) -/
def ValidNumbers : Set ℕ := {n : ℕ | 40 ≤ n ∧ n ≤ 50}

/-- Function to convert a dice roll to a two-digit number -/
def rollToNumber (roll : TwoDiceRoll) : ℕ :=
  10 * (roll.1.val + 1) + (roll.2.val + 1)

/-- The set of favorable outcomes -/
def FavorableOutcomes : Set TwoDiceRoll :=
  {roll : TwoDiceRoll | rollToNumber roll ∈ ValidNumbers}

/-- Total number of possible outcomes when rolling two dice -/
def TotalOutcomes : ℕ := 36

/-- Number of favorable outcomes -/
def FavorableOutcomesCount : ℕ := 12

/-- Probability of rolling a number between 40 and 50 (inclusive) -/
theorem probability_of_valid_roll :
  (FavorableOutcomesCount : ℚ) / TotalOutcomes = 1 / 3 := by
  sorry

end NUMINAMATH_CALUDE_probability_of_valid_roll_l2564_256481


namespace NUMINAMATH_CALUDE_target_hitting_probability_l2564_256491

theorem target_hitting_probability : 
  let p_single_hit : ℚ := 1/2
  let total_shots : ℕ := 7
  let total_hits : ℕ := 3
  let consecutive_hits : ℕ := 2

  -- Probability of exactly 3 hits out of 7 shots
  let p_total_hits : ℚ := (Nat.choose total_shots total_hits : ℚ) * p_single_hit ^ total_shots

  -- Number of ways to arrange 2 consecutive hits out of 3 in 7 shots
  let arrangements : ℕ := Nat.descFactorial (total_shots - consecutive_hits) consecutive_hits

  -- Final probability
  (arrangements : ℚ) * p_single_hit ^ total_shots = 5/32 :=
by sorry

end NUMINAMATH_CALUDE_target_hitting_probability_l2564_256491


namespace NUMINAMATH_CALUDE_possible_values_of_a_l2564_256475

theorem possible_values_of_a (x y a : ℝ) 
  (h1 : x + y = a) 
  (h2 : x^3 + y^3 = a) 
  (h3 : x^5 + y^5 = a) : 
  a = -2 ∨ a = -1 ∨ a = 0 ∨ a = 1 ∨ a = 2 := by
  sorry

end NUMINAMATH_CALUDE_possible_values_of_a_l2564_256475


namespace NUMINAMATH_CALUDE_triangle_side_length_l2564_256425

theorem triangle_side_length (a b c : ℝ) (B : ℝ) : 
  a = 2 →
  b + c = 7 →
  Real.cos B = -(1/4 : ℝ) →
  b = 4 := by
sorry

end NUMINAMATH_CALUDE_triangle_side_length_l2564_256425


namespace NUMINAMATH_CALUDE_decagon_diagonal_intersection_probability_l2564_256452

/-- A regular decagon -/
structure RegularDecagon where
  -- Add necessary fields here

/-- The probability that two randomly chosen diagonals of a regular decagon intersect inside the decagon -/
def intersection_probability (d : RegularDecagon) : ℚ :=
  42 / 119

/-- Theorem stating that the probability of two randomly chosen diagonals 
    of a regular decagon intersecting inside the decagon is 42/119 -/
theorem decagon_diagonal_intersection_probability (d : RegularDecagon) :
  intersection_probability d = 42 / 119 := by
  sorry

#check decagon_diagonal_intersection_probability

end NUMINAMATH_CALUDE_decagon_diagonal_intersection_probability_l2564_256452


namespace NUMINAMATH_CALUDE_victor_sugar_usage_l2564_256490

theorem victor_sugar_usage (brown_sugar : ℝ) (difference : ℝ) (white_sugar : ℝ)
  (h1 : brown_sugar = 0.62)
  (h2 : brown_sugar = white_sugar + difference)
  (h3 : difference = 0.38) :
  white_sugar = 0.24 := by
  sorry

end NUMINAMATH_CALUDE_victor_sugar_usage_l2564_256490


namespace NUMINAMATH_CALUDE_ratio_nine_to_five_percent_l2564_256483

/-- The ratio 9 : 5 expressed as a percentage -/
def ratio_to_percent : ℚ := 9 / 5 * 100

/-- Theorem: The ratio 9 : 5 expressed as a percentage is equal to 180% -/
theorem ratio_nine_to_five_percent : ratio_to_percent = 180 := by
  sorry

end NUMINAMATH_CALUDE_ratio_nine_to_five_percent_l2564_256483


namespace NUMINAMATH_CALUDE_absolute_value_equation_solution_l2564_256401

theorem absolute_value_equation_solution :
  ∀ y : ℝ, (|y - 4| + 3 * y = 11) ↔ (y = 15/4 ∨ y = 7/2) := by sorry

end NUMINAMATH_CALUDE_absolute_value_equation_solution_l2564_256401


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_l2564_256449

theorem quadratic_inequality_solution : 
  {x : ℝ | x^2 + 2*x ≤ -1} = {-1} := by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_l2564_256449


namespace NUMINAMATH_CALUDE_extreme_value_at_zero_decreasing_on_interval_l2564_256492

-- Define the function f(x)
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (3 * x^2 + a * x) / Real.exp x

-- Theorem for the first part of the problem
theorem extreme_value_at_zero (a : ℝ) :
  (∃ (ε : ℝ), ε > 0 ∧ ∀ (x : ℝ), 0 < |x| ∧ |x| < ε → f a 0 ≥ f a x) ∨
  (∃ (ε : ℝ), ε > 0 ∧ ∀ (x : ℝ), 0 < |x| ∧ |x| < ε → f a 0 ≤ f a x) ↔
  a = 0 :=
sorry

-- Theorem for the second part of the problem
theorem decreasing_on_interval (a : ℝ) :
  (∀ (x y : ℝ), 3 ≤ x ∧ x < y → f a x > f a y) ↔
  a ≥ -9/2 :=
sorry

end NUMINAMATH_CALUDE_extreme_value_at_zero_decreasing_on_interval_l2564_256492


namespace NUMINAMATH_CALUDE_perfect_square_condition_l2564_256431

theorem perfect_square_condition (a b c d : ℕ+) : 
  (↑a + Real.rpow 2 (1/3 : ℝ) * ↑b + Real.rpow 2 (2/3 : ℝ) * ↑c)^2 = ↑d → 
  ∃ (n : ℕ), d = n^2 := by
  sorry

end NUMINAMATH_CALUDE_perfect_square_condition_l2564_256431


namespace NUMINAMATH_CALUDE_max_sides_of_special_polygon_existence_of_five_sided_polygon_l2564_256450

-- Define a convex polygon
def ConvexPolygon (n : ℕ) := Unit

-- Define a property that a polygon has at least one side of length 1
def HasSideOfLengthOne (p : ConvexPolygon n) : Prop := sorry

-- Define a property that all diagonals of a polygon have integer lengths
def AllDiagonalsInteger (p : ConvexPolygon n) : Prop := sorry

-- State the theorem
theorem max_sides_of_special_polygon :
  ∀ n : ℕ, n > 5 →
  ¬∃ (p : ConvexPolygon n), HasSideOfLengthOne p ∧ AllDiagonalsInteger p :=
sorry

theorem existence_of_five_sided_polygon :
  ∃ (p : ConvexPolygon 5), HasSideOfLengthOne p ∧ AllDiagonalsInteger p :=
sorry

end NUMINAMATH_CALUDE_max_sides_of_special_polygon_existence_of_five_sided_polygon_l2564_256450


namespace NUMINAMATH_CALUDE_simplify_part1_simplify_part2_l2564_256455

-- Part 1
theorem simplify_part1 (x : ℝ) (h : x ≠ -2) :
  x^2 / (x + 2) + (4*x + 4) / (x + 2) = x + 2 := by sorry

-- Part 2
theorem simplify_part2 (x : ℝ) (h : x ≠ 1) :
  x^2 / (x^2 - 2*x + 1) / ((1 - 2*x) / (x - 1) - x + 1) = -1 / (x - 1) := by sorry

end NUMINAMATH_CALUDE_simplify_part1_simplify_part2_l2564_256455


namespace NUMINAMATH_CALUDE_dog_paws_on_ground_l2564_256451

theorem dog_paws_on_ground (total_dogs : ℕ) (dogs_on_back_legs : ℕ) (dogs_on_all_legs : ℕ) : 
  total_dogs = 12 →
  dogs_on_back_legs = total_dogs / 2 →
  dogs_on_all_legs = total_dogs / 2 →
  dogs_on_back_legs * 2 + dogs_on_all_legs * 4 = 36 := by
  sorry

end NUMINAMATH_CALUDE_dog_paws_on_ground_l2564_256451


namespace NUMINAMATH_CALUDE_max_gcd_13n_plus_4_8n_plus_3_l2564_256410

theorem max_gcd_13n_plus_4_8n_plus_3 :
  (∀ n : ℕ+, Nat.gcd (13 * n + 4) (8 * n + 3) ≤ 7) ∧
  (∃ n : ℕ+, Nat.gcd (13 * n + 4) (8 * n + 3) = 7) :=
by sorry

end NUMINAMATH_CALUDE_max_gcd_13n_plus_4_8n_plus_3_l2564_256410


namespace NUMINAMATH_CALUDE_polynomial_factor_l2564_256403

theorem polynomial_factor (a b c : ℤ) (x : ℚ) : 
  a = 1 ∧ b = -1 ∧ c = -8 →
  ∃ k : ℚ, (x^2 - 2*x - 1) * (2*a*x + k) = 2*a*x^3 + b*x^2 + c*x - 3 :=
by sorry

end NUMINAMATH_CALUDE_polynomial_factor_l2564_256403


namespace NUMINAMATH_CALUDE_not_all_face_sums_distinct_not_all_face_sums_distinct_l2564_256416

-- Define a cube type
structure Cube where
  vertices : Fin 8 → ℤ
  vertex_values : ∀ v, vertices v = 0 ∨ vertices v = 1

-- Define a function to get the sum of a face
def face_sum (c : Cube) (face : Fin 6) : ℤ :=
  sorry

-- Theorem statement
theorem not_all_face_sums_distinct (c : Cube) :
  ¬ (∀ f₁ f₂ : Fin 6, f₁ ≠ f₂ → face_sum c f₁ ≠ face_sum c f₂) :=
sorry

-- For part b, we can define a similar structure and theorem
structure Cube' where
  vertices : Fin 8 → ℤ
  vertex_values : ∀ v, vertices v = 1 ∨ vertices v = -1

def face_sum' (c : Cube') (face : Fin 6) : ℤ :=
  sorry

theorem not_all_face_sums_distinct' (c : Cube') :
  ¬ (∀ f₁ f₂ : Fin 6, f₁ ≠ f₂ → face_sum' c f₁ ≠ face_sum' c f₂) :=
sorry

end NUMINAMATH_CALUDE_not_all_face_sums_distinct_not_all_face_sums_distinct_l2564_256416


namespace NUMINAMATH_CALUDE_parallel_vectors_x_value_l2564_256462

def vector_a : ℝ × ℝ := (1, 2)
def vector_b (x : ℝ) : ℝ × ℝ := (2*x, -3)

def parallel (v w : ℝ × ℝ) : Prop :=
  ∃ (k : ℝ), v.1 = k * w.1 ∧ v.2 = k * w.2

theorem parallel_vectors_x_value :
  parallel vector_a (vector_b x) → x = -3/4 :=
by
  sorry

end NUMINAMATH_CALUDE_parallel_vectors_x_value_l2564_256462


namespace NUMINAMATH_CALUDE_trig_fraction_value_l2564_256495

theorem trig_fraction_value (α : Real) (h : Real.tan α = 2) :
  (2 * Real.sin α - Real.cos α) / (2 * Real.cos α + Real.sin α) = 3/4 := by
  sorry

end NUMINAMATH_CALUDE_trig_fraction_value_l2564_256495


namespace NUMINAMATH_CALUDE_rachel_saturday_water_consumption_l2564_256448

def glassesToOunces (glasses : ℕ) : ℕ := glasses * 10

def waterConsumed (sun mon tue wed thu fri : ℕ) : ℕ :=
  glassesToOunces (sun + mon + tue + wed + thu + fri)

theorem rachel_saturday_water_consumption
  (h1 : waterConsumed 2 4 3 3 3 3 + glassesToOunces x = 220) :
  x = 4 := by
  sorry

end NUMINAMATH_CALUDE_rachel_saturday_water_consumption_l2564_256448


namespace NUMINAMATH_CALUDE_tangent_line_and_extrema_l2564_256453

noncomputable def f (x : ℝ) : ℝ := (x - 1) / x * Real.log x

theorem tangent_line_and_extrema :
  let a := (1 : ℝ) / 4
  let b := Real.exp 1
  ∃ (tl : ℝ → ℝ) (max_val min_val : ℝ),
    (∀ x, tl x = 2 * x - 2 + Real.log 2) ∧
    (∀ x ∈ Set.Icc a b, f x ≤ max_val) ∧
    (∃ x ∈ Set.Icc a b, f x = max_val) ∧
    (∀ x ∈ Set.Icc a b, min_val ≤ f x) ∧
    (∃ x ∈ Set.Icc a b, f x = min_val) ∧
    max_val = 0 ∧
    min_val = Real.log 4 - 3 ∧
    (HasDerivAt f 2 (1/2) ∧ f (1/2) = -1 + Real.log 2) :=
by sorry

end NUMINAMATH_CALUDE_tangent_line_and_extrema_l2564_256453


namespace NUMINAMATH_CALUDE_expression_simplification_l2564_256464

theorem expression_simplification (x y : ℝ) 
  (hx : x = Real.tan (60 * π / 180)^2 + 1)
  (hy : y = Real.tan (45 * π / 180) - 2 * Real.cos (30 * π / 180)) :
  (x - (2*x*y - y^2) / x) / ((x^2 - y^2) / (x^2 + x*y)) = 3 + Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l2564_256464


namespace NUMINAMATH_CALUDE_regular_polygon_sides_l2564_256477

theorem regular_polygon_sides (n : ℕ) (h : n > 2) : 
  (((n : ℝ) - 2) * 180 = n * 160) → n = 18 := by
  sorry

end NUMINAMATH_CALUDE_regular_polygon_sides_l2564_256477


namespace NUMINAMATH_CALUDE_nut_mixture_weight_l2564_256479

/-- Represents a mixture of nuts -/
structure NutMixture where
  almond_ratio : ℚ
  walnut_ratio : ℚ
  almond_weight : ℚ

/-- Calculates the total weight of a nut mixture -/
def total_weight (mix : NutMixture) : ℚ :=
  (mix.almond_weight / mix.almond_ratio) * (mix.almond_ratio + mix.walnut_ratio)

/-- Theorem: The total weight of the given nut mixture is 210 pounds -/
theorem nut_mixture_weight :
  let mix : NutMixture := {
    almond_ratio := 5,
    walnut_ratio := 2,
    almond_weight := 150
  }
  total_weight mix = 210 := by
  sorry

end NUMINAMATH_CALUDE_nut_mixture_weight_l2564_256479


namespace NUMINAMATH_CALUDE_millet_dominant_on_wednesday_l2564_256422

/-- Represents the state of the bird feeder on a given day -/
structure FeederState where
  day : ℕ
  millet : ℝ
  other : ℝ

/-- Calculates the next day's feeder state -/
def nextDay (state : FeederState) : FeederState :=
  { day := state.day + 1,
    millet := 0.8 * state.millet + 0.3,
    other := 0.5 * state.other + 0.7 }

/-- Checks if millet constitutes more than half of the seeds -/
def milletDominant (state : FeederState) : Prop :=
  state.millet > (state.millet + state.other) / 2

/-- Initial state of the feeder -/
def initialState : FeederState :=
  { day := 1, millet := 0.3, other := 0.7 }

/-- Theorem stating that millet becomes dominant on day 3 (Wednesday) -/
theorem millet_dominant_on_wednesday :
  let day3 := nextDay (nextDay initialState)
  milletDominant day3 ∧ ¬milletDominant (nextDay initialState) :=
sorry

end NUMINAMATH_CALUDE_millet_dominant_on_wednesday_l2564_256422


namespace NUMINAMATH_CALUDE_linda_notebooks_count_l2564_256469

/-- The number of notebooks Linda bought -/
def num_notebooks : ℕ := 3

/-- The cost of each notebook in dollars -/
def notebook_cost : ℚ := 6/5

/-- The cost of a box of pencils in dollars -/
def pencil_box_cost : ℚ := 3/2

/-- The cost of a box of pens in dollars -/
def pen_box_cost : ℚ := 17/10

/-- The total amount spent in dollars -/
def total_spent : ℚ := 68/10

theorem linda_notebooks_count :
  (num_notebooks : ℚ) * notebook_cost + pencil_box_cost + pen_box_cost = total_spent :=
sorry

end NUMINAMATH_CALUDE_linda_notebooks_count_l2564_256469


namespace NUMINAMATH_CALUDE_find_a_l2564_256442

-- Define the points A and B
def A (a : ℝ) : ℝ × ℝ := (-1, a)
def B (a : ℝ) : ℝ × ℝ := (a, 8)

-- Define the slope of the line 2x - y + 1 = 0
def slope_given_line : ℝ := 2

-- Define the theorem
theorem find_a : ∃ a : ℝ, 
  (B a).2 - (A a).2 = slope_given_line * ((B a).1 - (A a).1) :=
sorry

-- Note: (p.1) and (p.2) represent the x and y coordinates of a point p respectively

end NUMINAMATH_CALUDE_find_a_l2564_256442


namespace NUMINAMATH_CALUDE_technician_count_l2564_256443

theorem technician_count (total_workers : ℕ) (avg_salary : ℝ) (avg_tech_salary : ℝ) (avg_rest_salary : ℝ) :
  total_workers = 12 ∧ 
  avg_salary = 9000 ∧ 
  avg_tech_salary = 12000 ∧ 
  avg_rest_salary = 6000 →
  ∃ (tech_count : ℕ),
    tech_count = 6 ∧
    tech_count + (total_workers - tech_count) = total_workers ∧
    (avg_tech_salary * tech_count + avg_rest_salary * (total_workers - tech_count)) / total_workers = avg_salary :=
by
  sorry

end NUMINAMATH_CALUDE_technician_count_l2564_256443


namespace NUMINAMATH_CALUDE_gcd_840_1764_l2564_256432

theorem gcd_840_1764 : Nat.gcd 840 1764 = 84 := by
  sorry

end NUMINAMATH_CALUDE_gcd_840_1764_l2564_256432


namespace NUMINAMATH_CALUDE_grinder_price_correct_l2564_256480

/-- Represents the purchase and sale of two items with given profit/loss percentages --/
structure TwoItemSale where
  grinder_price : ℝ
  mobile_price : ℝ
  grinder_loss_percent : ℝ
  mobile_profit_percent : ℝ
  total_profit : ℝ

/-- The specific scenario described in the problem --/
def problem_scenario : TwoItemSale where
  grinder_price := 15000  -- This is what we want to prove
  mobile_price := 10000
  grinder_loss_percent := 4
  mobile_profit_percent := 10
  total_profit := 400

/-- Theorem stating that the given scenario satisfies the problem conditions --/
theorem grinder_price_correct (s : TwoItemSale) : 
  s.mobile_price = 10000 ∧
  s.grinder_loss_percent = 4 ∧
  s.mobile_profit_percent = 10 ∧
  s.total_profit = 400 →
  s.grinder_price = 15000 :=
by
  sorry

#check grinder_price_correct problem_scenario

end NUMINAMATH_CALUDE_grinder_price_correct_l2564_256480


namespace NUMINAMATH_CALUDE_angle_bisector_length_l2564_256485

-- Define the triangle ABC
def triangle_ABC (A B C : ℝ × ℝ) : Prop :=
  let AB := Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2)
  let BC := Real.sqrt ((B.1 - C.1)^2 + (B.2 - C.2)^2)
  let AC := Real.sqrt ((A.1 - C.1)^2 + (A.2 - C.2)^2)
  AB = 5 ∧ BC = 12 ∧ AC = 13

-- Define the angle bisector BE
def angle_bisector (A B C E : ℝ × ℝ) : Prop :=
  let AB := Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2)
  let BC := Real.sqrt ((B.1 - C.1)^2 + (B.2 - C.2)^2)
  let BE := Real.sqrt ((B.1 - E.1)^2 + (B.2 - E.2)^2)
  let AE := Real.sqrt ((A.1 - E.1)^2 + (A.2 - E.2)^2)
  let CE := Real.sqrt ((C.1 - E.1)^2 + (C.2 - E.2)^2)
  AE / AB = CE / BC

-- Theorem statement
theorem angle_bisector_length 
  (A B C E : ℝ × ℝ) 
  (h1 : triangle_ABC A B C) 
  (h2 : angle_bisector A B C E) :
  let BE := Real.sqrt ((B.1 - E.1)^2 + (B.2 - E.2)^2)
  ∃ m : ℝ, BE = m * Real.sqrt 2 ∧ m = Real.sqrt 138 / 4 := by
  sorry

end NUMINAMATH_CALUDE_angle_bisector_length_l2564_256485


namespace NUMINAMATH_CALUDE_diophantine_equation_solutions_l2564_256497

theorem diophantine_equation_solutions :
  (∃! n : ℕ, n = (Finset.filter (fun p : ℕ × ℕ => 
    4 * p.1 + 7 * p.2 = 1003 ∧ p.1 > 0 ∧ p.2 > 0) (Finset.product (Finset.range 1004) (Finset.range 1004))).card ∧ n = 36) :=
by sorry

end NUMINAMATH_CALUDE_diophantine_equation_solutions_l2564_256497


namespace NUMINAMATH_CALUDE_gvidon_descendants_l2564_256420

/-- The number of sons King Gvidon had -/
def kings_sons : ℕ := 5

/-- The number of descendants who had exactly 3 sons each -/
def descendants_with_sons : ℕ := 100

/-- The number of sons each fertile descendant had -/
def sons_per_descendant : ℕ := 3

/-- The total number of descendants of King Gvidon -/
def total_descendants : ℕ := kings_sons + descendants_with_sons * sons_per_descendant

theorem gvidon_descendants :
  total_descendants = 305 :=
sorry

end NUMINAMATH_CALUDE_gvidon_descendants_l2564_256420


namespace NUMINAMATH_CALUDE_quadratic_form_equivalence_l2564_256478

theorem quadratic_form_equivalence :
  ∀ x : ℝ, 2 * x^2 + 3 * x - 1 = 2 * (x + 3/4)^2 - 17/8 :=
by
  sorry

end NUMINAMATH_CALUDE_quadratic_form_equivalence_l2564_256478


namespace NUMINAMATH_CALUDE_y_coordinate_relationship_l2564_256400

/-- The quadratic function f(x) = -(x-3)^2 - 4 -/
def f (x : ℝ) : ℝ := -(x - 3)^2 - 4

/-- Theorem stating the relationship between y-coordinates of three points on the quadratic function -/
theorem y_coordinate_relationship :
  let y₁ := f (-1/2)
  let y₂ := f 1
  let y₃ := f 4
  y₁ < y₂ ∧ y₂ < y₃ := by sorry

end NUMINAMATH_CALUDE_y_coordinate_relationship_l2564_256400


namespace NUMINAMATH_CALUDE_circumscribable_with_special_area_is_inscribable_l2564_256457

-- Define a quadrilateral
structure Quadrilateral where
  a : ℝ
  b : ℝ
  c : ℝ
  d : ℝ
  area : ℝ

-- Define the properties of being circumscribable and inscribable
def is_circumscribable (q : Quadrilateral) : Prop := sorry
def is_inscribable (q : Quadrilateral) : Prop := sorry

-- State the theorem
theorem circumscribable_with_special_area_is_inscribable (q : Quadrilateral) :
  is_circumscribable q →
  q.area = Real.sqrt (q.a * q.b * q.c * q.d) →
  is_inscribable q := by sorry

end NUMINAMATH_CALUDE_circumscribable_with_special_area_is_inscribable_l2564_256457


namespace NUMINAMATH_CALUDE_f_properties_l2564_256428

noncomputable def f (x : ℝ) : ℝ := x^2 * Real.exp x

theorem f_properties :
  (∃ a b : ℝ, a = -2 ∧ b = 0 ∧ ∀ x ∈ Set.Ioo a b, StrictMonoOn f (Set.Ioo a b)) ∧
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ f x₁ = x₁ - 2012 ∧ f x₂ = x₂ - 2012) :=
by sorry

end NUMINAMATH_CALUDE_f_properties_l2564_256428


namespace NUMINAMATH_CALUDE_correct_sample_l2564_256436

/-- Represents a random number table --/
def RandomNumberTable := List (List Nat)

/-- Represents a sample of student numbers --/
def Sample := List Nat

/-- Checks if a number is a valid student number (between 1 and 50) --/
def isValidStudentNumber (n : Nat) : Bool :=
  1 ≤ n ∧ n ≤ 50

/-- Selects a sample of distinct student numbers from the random number table --/
def selectSample (table : RandomNumberTable) (startRow : Nat) (startCol : Nat) (sampleSize : Nat) : Sample :=
  sorry

/-- The specific random number table given in the problem --/
def givenTable : RandomNumberTable :=
  [[03, 47, 43, 73, 86, 36, 96, 47, 36, 61, 46, 98, 63, 71, 62, 33, 26, 16, 80],
   [45, 60, 11, 14, 10, 95, 97, 74, 24, 67, 62, 42, 81, 14, 57, 20, 42, 53],
   [32, 37, 32, 27, 07, 36, 07, 51, 24, 51, 79, 89, 73, 16, 76, 62, 27, 66],
   [56, 50, 26, 71, 07, 32, 90, 79, 78, 53, 13, 55, 38, 58, 59, 88, 97, 54],
   [14, 10, 12, 56, 85, 99, 26, 96, 96, 68, 27, 31, 05, 03, 72, 93, 15, 57],
   [12, 10, 14, 21, 88, 26, 49, 81, 76, 55, 59, 56, 35, 64, 38, 54, 82, 46],
   [22, 31, 62, 43, 09, 90, 06, 18, 44, 32, 53, 23, 83, 01, 30, 30]]

theorem correct_sample :
  selectSample givenTable 3 6 5 = [22, 2, 10, 29, 7] :=
sorry

end NUMINAMATH_CALUDE_correct_sample_l2564_256436


namespace NUMINAMATH_CALUDE_polynomial_division_remainder_l2564_256493

theorem polynomial_division_remainder (t : ℚ) :
  (∀ x, (6 * x^2 - 7 * x + 8) = (5 * x^2 + t * x + 12) * (4 * x^2 - 9 * x + 12)) →
  t = -7/12 := by
sorry

end NUMINAMATH_CALUDE_polynomial_division_remainder_l2564_256493


namespace NUMINAMATH_CALUDE_triangle_height_l2564_256405

theorem triangle_height (base : ℝ) (area : ℝ) (height : ℝ) : 
  base = 10 → area = 50 → area = (base * height) / 2 → height = 10 := by
  sorry

end NUMINAMATH_CALUDE_triangle_height_l2564_256405


namespace NUMINAMATH_CALUDE_smiths_bakery_pies_l2564_256440

/-- The number of pies sold by Mcgee's Bakery -/
def mcgees_pies : ℕ := 16

/-- The number of pies sold by Smith's Bakery -/
def smiths_pies : ℕ := 4 * mcgees_pies + 6

/-- Theorem stating that Smith's Bakery sold 70 pies -/
theorem smiths_bakery_pies : smiths_pies = 70 := by
  sorry

end NUMINAMATH_CALUDE_smiths_bakery_pies_l2564_256440


namespace NUMINAMATH_CALUDE_initial_fund_is_740_l2564_256456

/-- Represents the company fund problem --/
structure CompanyFund where
  intended_bonus : ℕ
  actual_bonus : ℕ
  remaining_amount : ℕ
  fixed_expense : ℕ
  shortage : ℕ

/-- Calculates the initial fund amount before bonuses and expenses --/
def initial_fund_amount (cf : CompanyFund) : ℕ :=
  sorry

/-- Theorem stating the initial fund amount is 740 given the problem conditions --/
theorem initial_fund_is_740 (cf : CompanyFund) 
  (h1 : cf.intended_bonus = 60)
  (h2 : cf.actual_bonus = 50)
  (h3 : cf.remaining_amount = 110)
  (h4 : cf.fixed_expense = 30)
  (h5 : cf.shortage = 10) :
  initial_fund_amount cf = 740 :=
sorry

end NUMINAMATH_CALUDE_initial_fund_is_740_l2564_256456


namespace NUMINAMATH_CALUDE_joe_paint_usage_l2564_256460

/-- The amount of paint Joe used in total -/
def total_paint_used (initial_paint : ℚ) (first_week_fraction : ℚ) (second_week_fraction : ℚ) : ℚ :=
  let first_week_usage := first_week_fraction * initial_paint
  let remaining_paint := initial_paint - first_week_usage
  let second_week_usage := second_week_fraction * remaining_paint
  first_week_usage + second_week_usage

/-- Theorem stating that Joe used 264 gallons of paint -/
theorem joe_paint_usage :
  total_paint_used 360 (2/3) (1/5) = 264 := by
  sorry

end NUMINAMATH_CALUDE_joe_paint_usage_l2564_256460


namespace NUMINAMATH_CALUDE_quadratic_solution_unique_positive_l2564_256454

theorem quadratic_solution_unique_positive (x : ℝ) :
  x > 0 ∧ 3 * x^2 + 8 * x - 35 = 0 ↔ x = 7/3 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_solution_unique_positive_l2564_256454


namespace NUMINAMATH_CALUDE_people_per_institution_l2564_256467

theorem people_per_institution 
  (total_institutions : ℕ) 
  (total_people : ℕ) 
  (h1 : total_institutions = 6) 
  (h2 : total_people = 480) : 
  total_people / total_institutions = 80 := by
  sorry

end NUMINAMATH_CALUDE_people_per_institution_l2564_256467


namespace NUMINAMATH_CALUDE_element_type_determined_by_protons_nuclide_type_determined_by_protons_and_neutrons_chemical_properties_determined_by_outermost_electrons_highest_valence_equals_main_group_number_l2564_256430

-- Define basic types
structure Element where
  protons : ℕ

structure Nuclide where
  protons : ℕ
  neutrons : ℕ

structure MainGroupElement where
  protons : ℕ
  outermostElectrons : ℕ

-- Define properties
def elementType (e : Element) : ℕ := e.protons

def nuclideType (n : Nuclide) : ℕ × ℕ := (n.protons, n.neutrons)

def mainChemicalProperties (e : MainGroupElement) : ℕ := e.outermostElectrons

def highestPositiveValence (e : MainGroupElement) : ℕ := e.outermostElectrons

-- Theorem statements
theorem element_type_determined_by_protons (e1 e2 : Element) :
  elementType e1 = elementType e2 ↔ e1.protons = e2.protons :=
sorry

theorem nuclide_type_determined_by_protons_and_neutrons (n1 n2 : Nuclide) :
  nuclideType n1 = nuclideType n2 ↔ n1.protons = n2.protons ∧ n1.neutrons = n2.neutrons :=
sorry

theorem chemical_properties_determined_by_outermost_electrons (e : MainGroupElement) :
  mainChemicalProperties e = e.outermostElectrons :=
sorry

theorem highest_valence_equals_main_group_number (e : MainGroupElement) :
  highestPositiveValence e = e.outermostElectrons :=
sorry

end NUMINAMATH_CALUDE_element_type_determined_by_protons_nuclide_type_determined_by_protons_and_neutrons_chemical_properties_determined_by_outermost_electrons_highest_valence_equals_main_group_number_l2564_256430


namespace NUMINAMATH_CALUDE_total_shaded_area_is_75_over_4_l2564_256413

/-- Represents a truncated square-based pyramid -/
structure TruncatedPyramid where
  base_side : ℝ
  top_side : ℝ
  height : ℝ

/-- Calculate the total shaded area of the truncated pyramid -/
def total_shaded_area (p : TruncatedPyramid) : ℝ :=
  sorry

/-- The main theorem stating that the total shaded area is 75/4 -/
theorem total_shaded_area_is_75_over_4 :
  ∀ (p : TruncatedPyramid),
  p.base_side = 7 ∧ p.top_side = 1 ∧ p.height = 4 →
  total_shaded_area p = 75 / 4 := by
  sorry

end NUMINAMATH_CALUDE_total_shaded_area_is_75_over_4_l2564_256413


namespace NUMINAMATH_CALUDE_cosine_rational_values_l2564_256406

theorem cosine_rational_values (α : ℚ) (h : ∃ (q : ℚ), q = Real.cos (α * Real.pi)) :
  Real.cos (α * Real.pi) = 0 ∨ 
  Real.cos (α * Real.pi) = (1/2 : ℝ) ∨ 
  Real.cos (α * Real.pi) = -(1/2 : ℝ) ∨ 
  Real.cos (α * Real.pi) = 1 ∨ 
  Real.cos (α * Real.pi) = -1 := by
sorry

end NUMINAMATH_CALUDE_cosine_rational_values_l2564_256406


namespace NUMINAMATH_CALUDE_problem_statement_l2564_256437

theorem problem_statement (a b c : ℝ) 
  (h_pos : a > 0 ∧ b > 0 ∧ c > 0) 
  (h_sum : a + b + c = 1) : 
  (∃ min_val : ℝ, min_val = 144/49 ∧ 
    ∀ x y z : ℝ, x > 0 → y > 0 → z > 0 → x + y + z = 1 → 
      (x + 1)^2 + 4*y^2 + 9*z^2 ≥ min_val) ∧
  (1 / (Real.sqrt a + Real.sqrt b) + 
   1 / (Real.sqrt b + Real.sqrt c) + 
   1 / (Real.sqrt c + Real.sqrt a) ≥ 3 * Real.sqrt 3 / 2) :=
by sorry

end NUMINAMATH_CALUDE_problem_statement_l2564_256437


namespace NUMINAMATH_CALUDE_price_per_diaper_l2564_256417

def boxes : ℕ := 30
def packs_per_box : ℕ := 40
def diapers_per_pack : ℕ := 160
def total_revenue : ℕ := 960000

def total_diapers : ℕ := boxes * packs_per_box * diapers_per_pack

theorem price_per_diaper :
  total_revenue / total_diapers = 5 :=
by sorry

end NUMINAMATH_CALUDE_price_per_diaper_l2564_256417


namespace NUMINAMATH_CALUDE_initial_mixture_volume_l2564_256484

/-- Proves that given a mixture with 20% water content, if adding 8.333333333333334 gallons
    of water increases the water percentage to 25%, then the initial volume of the mixture
    is 125 gallons. -/
theorem initial_mixture_volume
  (initial_water_percentage : ℝ)
  (added_water : ℝ)
  (final_water_percentage : ℝ)
  (h1 : initial_water_percentage = 0.20)
  (h2 : added_water = 8.333333333333334)
  (h3 : final_water_percentage = 0.25)
  (h4 : ∀ v : ℝ, final_water_percentage * (v + added_water) = initial_water_percentage * v + added_water) :
  ∃ v : ℝ, v = 125 := by
  sorry

end NUMINAMATH_CALUDE_initial_mixture_volume_l2564_256484


namespace NUMINAMATH_CALUDE_jerry_shelf_difference_l2564_256447

/-- Calculates the difference between books and action figures on Jerry's shelf -/
def shelf_difference (initial_figures : ℕ) (initial_books : ℕ) (added_figures : ℕ) : ℕ :=
  initial_books - (initial_figures + added_figures)

/-- Proves that the difference between books and action figures on Jerry's shelf is 4 -/
theorem jerry_shelf_difference :
  shelf_difference 2 10 4 = 4 := by
  sorry

end NUMINAMATH_CALUDE_jerry_shelf_difference_l2564_256447


namespace NUMINAMATH_CALUDE_camdens_dogs_legs_l2564_256489

theorem camdens_dogs_legs : 
  ∀ (justin_dogs rico_dogs camden_dogs : ℕ) (legs_per_dog : ℕ),
  justin_dogs = 14 →
  rico_dogs = justin_dogs + 10 →
  camden_dogs = rico_dogs * 3 / 4 →
  legs_per_dog = 4 →
  camden_dogs * legs_per_dog = 72 :=
by sorry

end NUMINAMATH_CALUDE_camdens_dogs_legs_l2564_256489
