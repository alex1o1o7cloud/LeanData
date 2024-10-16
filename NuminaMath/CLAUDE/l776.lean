import Mathlib

namespace NUMINAMATH_CALUDE_amanda_earnings_l776_77600

/-- Amanda's hourly rate in dollars -/
def hourly_rate : ℝ := 20

/-- Total hours worked on Monday -/
def monday_hours : ℝ := 5 * 1.5

/-- Total hours worked on Tuesday -/
def tuesday_hours : ℝ := 3

/-- Total hours worked on Thursday -/
def thursday_hours : ℝ := 2 * 2

/-- Total hours worked on Saturday -/
def saturday_hours : ℝ := 6

/-- Total hours worked in the week -/
def total_hours : ℝ := monday_hours + tuesday_hours + thursday_hours + saturday_hours

/-- Amanda's total earnings for the week -/
def total_earnings : ℝ := hourly_rate * total_hours

theorem amanda_earnings : total_earnings = 410 := by
  sorry

end NUMINAMATH_CALUDE_amanda_earnings_l776_77600


namespace NUMINAMATH_CALUDE_female_officers_count_l776_77620

/-- The total number of officers on duty -/
def total_on_duty : ℕ := 300

/-- The fraction of officers on duty who are female -/
def female_fraction : ℚ := 1/2

/-- The percentage of female officers who were on duty -/
def female_on_duty_percent : ℚ := 15/100

/-- The total number of female officers on the police force -/
def total_female_officers : ℕ := 1000

theorem female_officers_count :
  (total_on_duty : ℚ) * female_fraction / female_on_duty_percent = total_female_officers := by
  sorry

end NUMINAMATH_CALUDE_female_officers_count_l776_77620


namespace NUMINAMATH_CALUDE_circumscribed_circle_area_specific_trapezoid_l776_77619

/-- An isosceles trapezoid with given dimensions. -/
structure IsoscelesTrapezoid where
  height : ℝ
  base1 : ℝ
  base2 : ℝ

/-- The circumscribed circle of an isosceles trapezoid. -/
def circumscribedCircleArea (t : IsoscelesTrapezoid) : ℝ :=
  sorry

/-- Theorem stating the area of the circumscribed circle for a specific isosceles trapezoid. -/
theorem circumscribed_circle_area_specific_trapezoid :
  let t : IsoscelesTrapezoid := { height := 14, base1 := 16, base2 := 12 }
  circumscribedCircleArea t = 100 * Real.pi := by
  sorry

end NUMINAMATH_CALUDE_circumscribed_circle_area_specific_trapezoid_l776_77619


namespace NUMINAMATH_CALUDE_number_of_preferred_shares_l776_77675

/-- Represents the number of preferred shares -/
def preferred_shares : ℕ := sorry

/-- Represents the number of common shares -/
def common_shares : ℕ := 3000

/-- Represents the par value of each share in rupees -/
def par_value : ℚ := 50

/-- Represents the annual dividend rate for preferred shares -/
def preferred_dividend_rate : ℚ := 1 / 10

/-- Represents the annual dividend rate for common shares -/
def common_dividend_rate : ℚ := 7 / 100

/-- Represents the total annual dividend received in rupees -/
def total_annual_dividend : ℚ := 16500

/-- Theorem stating that the number of preferred shares is 1200 -/
theorem number_of_preferred_shares : 
  preferred_shares = 1200 :=
by sorry

end NUMINAMATH_CALUDE_number_of_preferred_shares_l776_77675


namespace NUMINAMATH_CALUDE_complement_intersection_theorem_l776_77614

-- Define the sets M and N
def M : Set ℝ := {x | ∃ y, y = Real.log (1 - x)}
def N : Set ℝ := {x | (2 : ℝ) ^ (x * (x - 2)) < 1}

-- Define the universal set U
def U : Set ℝ := Set.univ

-- State the theorem
theorem complement_intersection_theorem :
  (Set.compl M) ∩ N = {x : ℝ | 1 ≤ x ∧ x < 2} := by sorry

end NUMINAMATH_CALUDE_complement_intersection_theorem_l776_77614


namespace NUMINAMATH_CALUDE_roots_equal_magnitude_opposite_sign_l776_77644

theorem roots_equal_magnitude_opposite_sign (a b c m : ℝ) :
  (∃ x y : ℝ, x ≠ 0 ∧ y = -x ∧
    (x^2 - b*x) / (a*x - c) = (m - 1) / (m + 1) ∧
    (y^2 - b*y) / (a*y - c) = (m - 1) / (m + 1)) →
  m = (a - b) / (a + b) :=
by sorry

end NUMINAMATH_CALUDE_roots_equal_magnitude_opposite_sign_l776_77644


namespace NUMINAMATH_CALUDE_f_2019_equals_2016_l776_77638

def f : ℕ → ℕ
| x => if x ≤ 2015 then x + 2 else f (x - 5)

theorem f_2019_equals_2016 : f 2019 = 2016 := by
  sorry

end NUMINAMATH_CALUDE_f_2019_equals_2016_l776_77638


namespace NUMINAMATH_CALUDE_taxi_charge_calculation_l776_77623

/-- Calculates the total charge for a taxi trip -/
def totalCharge (initialFee : ℚ) (additionalChargePerIncrement : ℚ) (incrementDistance : ℚ) (tripDistance : ℚ) : ℚ :=
  initialFee + (tripDistance / incrementDistance).floor * additionalChargePerIncrement

/-- Theorem: The total charge for a 3.6-mile trip with given fees is $5.50 -/
theorem taxi_charge_calculation :
  let initialFee : ℚ := 235 / 100
  let additionalChargePerIncrement : ℚ := 35 / 100
  let incrementDistance : ℚ := 2 / 5
  let tripDistance : ℚ := 36 / 10
  totalCharge initialFee additionalChargePerIncrement incrementDistance tripDistance = 550 / 100 := by
  sorry

#eval totalCharge (235/100) (35/100) (2/5) (36/10)

end NUMINAMATH_CALUDE_taxi_charge_calculation_l776_77623


namespace NUMINAMATH_CALUDE_tiles_difference_l776_77659

/-- The side length of the nth square in the sequence -/
def side_length (n : ℕ) : ℕ := 2 * n - 1

/-- The number of tiles in the nth square -/
def tiles_in_square (n : ℕ) : ℕ := (side_length n) ^ 2

/-- The theorem stating the difference in tiles between the 10th and 9th squares -/
theorem tiles_difference : tiles_in_square 10 - tiles_in_square 9 = 72 := by
  sorry

end NUMINAMATH_CALUDE_tiles_difference_l776_77659


namespace NUMINAMATH_CALUDE_vacation_cost_division_l776_77605

theorem vacation_cost_division (total_cost : ℕ) (initial_people : ℕ) (cost_reduction : ℕ) (n : ℕ) : 
  total_cost = 360 →
  initial_people = 3 →
  (total_cost / initial_people) - (total_cost / n) = cost_reduction →
  cost_reduction = 30 →
  n = 4 :=
by sorry

end NUMINAMATH_CALUDE_vacation_cost_division_l776_77605


namespace NUMINAMATH_CALUDE_parabola_equation_l776_77674

/-- A parabola with equation y² = 2px where p > 0 -/
structure Parabola where
  p : ℝ
  h_p_pos : p > 0

/-- A line that passes through the focus of a parabola and intersects it at two points -/
structure IntersectingLine (P : Parabola) where
  A : ℝ × ℝ
  B : ℝ × ℝ
  h_on_parabola_A : A.2^2 = 2 * P.p * A.1
  h_on_parabola_B : B.2^2 = 2 * P.p * B.1
  h_through_focus : True  -- We don't need to specify this condition explicitly for the proof

/-- The theorem stating the conditions and the result to be proved -/
theorem parabola_equation (P : Parabola) (L : IntersectingLine P)
  (h_length : Real.sqrt ((L.A.1 - L.B.1)^2 + (L.A.2 - L.B.2)^2) = 8)
  (h_midpoint : (L.A.1 + L.B.1) / 2 = 2) :
  P.p = 4 ∧ ∀ (x y : ℝ), y^2 = 8*x ↔ y^2 = 2*P.p*x := by
  sorry

end NUMINAMATH_CALUDE_parabola_equation_l776_77674


namespace NUMINAMATH_CALUDE_wall_length_l776_77685

/-- Proves that the length of a wall is 250 meters given specific brick and wall dimensions -/
theorem wall_length (brick_length : ℝ) (brick_width : ℝ) (brick_height : ℝ)
                    (wall_height : ℝ) (wall_width : ℝ) (num_bricks : ℕ) :
  brick_length = 0.2 →
  brick_width = 0.1 →
  brick_height = 0.075 →
  wall_height = 2 →
  wall_width = 0.75 →
  num_bricks = 25000 →
  ∃ (wall_length : ℝ), wall_length = 250 :=
by
  sorry


end NUMINAMATH_CALUDE_wall_length_l776_77685


namespace NUMINAMATH_CALUDE_fifth_power_sum_l776_77680

theorem fifth_power_sum (a b x y : ℝ) 
  (h1 : a * x + b * y = 5)
  (h2 : a * x^2 + b * y^2 = 11)
  (h3 : a * x^3 + b * y^3 = 24)
  (h4 : a * x^4 + b * y^4 = 58) :
  a * x^5 + b * y^5 = -16 := by
sorry

end NUMINAMATH_CALUDE_fifth_power_sum_l776_77680


namespace NUMINAMATH_CALUDE_min_rectangles_for_problem_figure_l776_77629

/-- Represents a corner in the figure -/
inductive Corner
| Type1
| Type2

/-- Represents a set of three Type2 corners -/
structure CornerSet :=
  (corners : Fin 3 → Corner)
  (all_type2 : ∀ i, corners i = Corner.Type2)

/-- The figure with its corner structure -/
structure Figure :=
  (total_corners : Nat)
  (type1_corners : Nat)
  (type2_corners : Nat)
  (corner_sets : Nat)
  (valid_total : total_corners = type1_corners + type2_corners)
  (valid_type2 : type2_corners = 3 * corner_sets)

/-- The minimum number of rectangles needed to cover the figure -/
def min_rectangles (f : Figure) : Nat :=
  f.type1_corners + f.corner_sets

/-- The specific figure from the problem -/
def problem_figure : Figure :=
  { total_corners := 24
  , type1_corners := 12
  , type2_corners := 12
  , corner_sets := 4
  , valid_total := by rfl
  , valid_type2 := by rfl }

theorem min_rectangles_for_problem_figure :
  min_rectangles problem_figure = 12 := by sorry

end NUMINAMATH_CALUDE_min_rectangles_for_problem_figure_l776_77629


namespace NUMINAMATH_CALUDE_investment_plans_count_l776_77696

/-- The number of cities available for investment -/
def num_cities : ℕ := 5

/-- The number of projects to be invested -/
def num_projects : ℕ := 3

/-- The maximum number of projects that can be invested in a single city -/
def max_projects_per_city : ℕ := 2

/-- The function that calculates the number of investment plans -/
def num_investment_plans : ℕ :=
  -- The actual calculation would go here
  120

/-- Theorem stating that the number of investment plans is 120 -/
theorem investment_plans_count :
  num_investment_plans = 120 := by sorry

end NUMINAMATH_CALUDE_investment_plans_count_l776_77696


namespace NUMINAMATH_CALUDE_existence_of_n_with_k_prime_factors_l776_77678

theorem existence_of_n_with_k_prime_factors (k m : ℕ) (hk : k > 0) (hm : m > 0) (hm_odd : Odd m) :
  ∃ n : ℕ, n > 0 ∧ (∃ (S : Finset ℕ), S.card ≥ k ∧ ∀ p ∈ S, Prime p ∧ p ∣ (m^n + n^m)) :=
sorry

end NUMINAMATH_CALUDE_existence_of_n_with_k_prime_factors_l776_77678


namespace NUMINAMATH_CALUDE_rectangular_prism_parallel_edges_l776_77672

/-- A rectangular prism with different length, width, and height. -/
structure RectangularPrism where
  length : ℝ
  width : ℝ
  height : ℝ
  length_ne_width : length ≠ width
  length_ne_height : length ≠ height
  width_ne_height : width ≠ height

/-- The number of pairs of parallel edges in a rectangular prism. -/
def parallelEdgePairs (prism : RectangularPrism) : ℕ := 12

/-- Theorem stating that a rectangular prism with different dimensions has exactly 12 pairs of parallel edges. -/
theorem rectangular_prism_parallel_edges (prism : RectangularPrism) :
  parallelEdgePairs prism = 12 := by
  sorry

end NUMINAMATH_CALUDE_rectangular_prism_parallel_edges_l776_77672


namespace NUMINAMATH_CALUDE_subtraction_result_l776_77673

theorem subtraction_result (x : ℝ) (h : 96 / x = 6) : 34 - x = 18 := by
  sorry

end NUMINAMATH_CALUDE_subtraction_result_l776_77673


namespace NUMINAMATH_CALUDE_daughter_and_child_weight_l776_77689

/-- The combined weight of a daughter and her daughter (child) given specific family weight conditions -/
theorem daughter_and_child_weight (total_weight mother_weight daughter_weight child_weight : ℝ) :
  total_weight = mother_weight + daughter_weight + child_weight →
  child_weight = (1 / 5) * mother_weight →
  daughter_weight = 46 →
  total_weight = 130 →
  daughter_weight + child_weight = 60 := by
  sorry

end NUMINAMATH_CALUDE_daughter_and_child_weight_l776_77689


namespace NUMINAMATH_CALUDE_unique_solution_x_squared_minus_x_minus_one_l776_77615

theorem unique_solution_x_squared_minus_x_minus_one (x : ℝ) :
  x^2 - x - 1 = (x + 1)^0 ↔ x = 2 := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_x_squared_minus_x_minus_one_l776_77615


namespace NUMINAMATH_CALUDE_nonreal_cube_root_unity_sum_l776_77603

theorem nonreal_cube_root_unity_sum (ω : ℂ) : 
  ω^3 = 1 ∧ ω ≠ 1 → (1 - 2*ω + 2*ω^2)^6 + (1 + 2*ω - 2*ω^2)^6 = 0 :=
by sorry

end NUMINAMATH_CALUDE_nonreal_cube_root_unity_sum_l776_77603


namespace NUMINAMATH_CALUDE_greatest_integer_fraction_inequality_l776_77641

theorem greatest_integer_fraction_inequality : 
  (∀ y : ℤ, (8 : ℚ) / 11 > (y : ℚ) / 17 ↔ y ≤ 12) :=
by sorry

end NUMINAMATH_CALUDE_greatest_integer_fraction_inequality_l776_77641


namespace NUMINAMATH_CALUDE_number_less_than_l776_77650

theorem number_less_than : ∃ x : ℝ, x = 0.8 - 0.3 ∧ x = 0.5 := by
  sorry

end NUMINAMATH_CALUDE_number_less_than_l776_77650


namespace NUMINAMATH_CALUDE_initial_number_proof_l776_77683

theorem initial_number_proof (x : ℕ) : 7899665 - (3 * 2 * x) = 7899593 ↔ x = 12 := by
  sorry

end NUMINAMATH_CALUDE_initial_number_proof_l776_77683


namespace NUMINAMATH_CALUDE_tangent_line_y_intercept_l776_77633

/-- A circle in a 2D plane --/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- A line in a 2D plane, represented by its slope and y-intercept --/
structure Line where
  slope : ℝ
  yIntercept : ℝ

/-- Check if a line is tangent to a circle --/
def isTangent (l : Line) (c : Circle) : Prop :=
  let (x, y) := c.center
  (l.slope * x - y + l.yIntercept)^2 = c.radius^2 * (l.slope^2 + 1)

theorem tangent_line_y_intercept :
  ∃ (l : Line),
    isTangent l { center := (2, 0), radius := 2 } ∧
    isTangent l { center := (5, 0), radius := 1 } ∧
    l.yIntercept = 2 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_tangent_line_y_intercept_l776_77633


namespace NUMINAMATH_CALUDE_equations_solution_set_l776_77628

def solution_set : Set (ℕ × ℕ × ℕ × ℕ) :=
  {(0,0,0,0), (2,2,2,2), (1,5,2,3), (5,1,2,3), (1,5,3,2), (5,1,3,2),
   (2,3,1,5), (2,3,5,1), (3,2,1,5), (3,2,5,1)}

def satisfies_equations (x y z t : ℕ) : Prop :=
  x + y = z + t ∧ z + t = x * y

theorem equations_solution_set :
  ∀ x y z t : ℕ, satisfies_equations x y z t ↔ (x, y, z, t) ∈ solution_set := by
  sorry

end NUMINAMATH_CALUDE_equations_solution_set_l776_77628


namespace NUMINAMATH_CALUDE_seeds_per_row_in_top_bed_l776_77635

theorem seeds_per_row_in_top_bed (
  top_beds : Nat
  ) (bottom_beds : Nat)
  (rows_per_top_bed : Nat)
  (rows_per_bottom_bed : Nat)
  (seeds_per_row_bottom : Nat)
  (total_seeds : Nat)
  (h1 : top_beds = 2)
  (h2 : bottom_beds = 2)
  (h3 : rows_per_top_bed = 4)
  (h4 : rows_per_bottom_bed = 3)
  (h5 : seeds_per_row_bottom = 20)
  (h6 : total_seeds = 320) :
  (total_seeds - (bottom_beds * rows_per_bottom_bed * seeds_per_row_bottom)) / (top_beds * rows_per_top_bed) = 25 := by
  sorry

end NUMINAMATH_CALUDE_seeds_per_row_in_top_bed_l776_77635


namespace NUMINAMATH_CALUDE_certain_number_proof_l776_77608

theorem certain_number_proof : ∃ x : ℤ, (287^2 : ℤ) + x^2 - 2*287*x = 324 ∧ x = 269 := by
  sorry

end NUMINAMATH_CALUDE_certain_number_proof_l776_77608


namespace NUMINAMATH_CALUDE_correct_calculation_l776_77682

theorem correct_calculation : -1^4 * (-1)^3 = 1 := by
  sorry

end NUMINAMATH_CALUDE_correct_calculation_l776_77682


namespace NUMINAMATH_CALUDE_probability_two_sunny_days_l776_77616

/-- The probability of exactly k sunny days in n days, given the probability of a sunny day --/
def probability_k_sunny_days (n k : ℕ) (p : ℝ) : ℝ :=
  (n.choose k : ℝ) * p^k * (1 - p)^(n - k)

/-- The number of days in the holiday weekend --/
def num_days : ℕ := 5

/-- The probability of a sunny day --/
def prob_sunny : ℝ := 0.3

/-- The number of desired sunny days --/
def desired_sunny_days : ℕ := 2

theorem probability_two_sunny_days :
  probability_k_sunny_days num_days desired_sunny_days prob_sunny = 0.3087 := by
  sorry

end NUMINAMATH_CALUDE_probability_two_sunny_days_l776_77616


namespace NUMINAMATH_CALUDE_p_sufficient_not_necessary_for_q_r_necessary_not_sufficient_for_p_l776_77693

-- Define the sets A, B, and C
def A : Set ℝ := {x : ℝ | |3*x - 4| > 2}
def B : Set ℝ := {x : ℝ | 1 / (x^2 - x - 2) > 0}
def C (a : ℝ) : Set ℝ := {x : ℝ | (x - a) * (x - a - 1) ≥ 0}

-- Define the propositions p, q, and r
def p (x : ℝ) : Prop := x ∉ A
def q (x : ℝ) : Prop := x ∉ B
def r (a x : ℝ) : Prop := x ∈ C a

-- Theorem 1: p is a sufficient but not necessary condition for q
theorem p_sufficient_not_necessary_for_q :
  (∀ x : ℝ, (2/3 ≤ x ∧ x ≤ 2) → (-1 ≤ x ∧ x ≤ 2)) ∧
  (∃ x : ℝ, (-1 ≤ x ∧ x ≤ 2) ∧ ¬(2/3 ≤ x ∧ x ≤ 2)) :=
sorry

-- Theorem 2: r is a necessary but not sufficient condition for p
--            if and only if a ≥ 2 or a ≤ -1/3
theorem r_necessary_not_sufficient_for_p (a : ℝ) :
  ((∀ x : ℝ, p x → r a x) ∧ (∃ x : ℝ, r a x ∧ ¬(p x))) ↔ (a ≥ 2 ∨ a ≤ -1/3) :=
sorry

end NUMINAMATH_CALUDE_p_sufficient_not_necessary_for_q_r_necessary_not_sufficient_for_p_l776_77693


namespace NUMINAMATH_CALUDE_cherry_pie_degrees_l776_77632

/-- The number of students in Richelle's class -/
def total_students : ℕ := 36

/-- The number of students who prefer chocolate pie -/
def chocolate_preference : ℕ := 12

/-- The number of students who prefer apple pie -/
def apple_preference : ℕ := 8

/-- The number of students who prefer blueberry pie -/
def blueberry_preference : ℕ := 6

/-- The number of students who prefer cherry pie -/
def cherry_preference : ℕ := (total_students - (chocolate_preference + apple_preference + blueberry_preference)) / 2

theorem cherry_pie_degrees : 
  (cherry_preference : ℚ) / total_students * 360 = 50 := by
  sorry

end NUMINAMATH_CALUDE_cherry_pie_degrees_l776_77632


namespace NUMINAMATH_CALUDE_quadratic_function_property_l776_77677

/-- Given a quadratic function f(x) = x^2 - 2x + 3, if f(m) = f(n) where m ≠ n, 
    then f(m + n) = 3 -/
theorem quadratic_function_property (m n : ℝ) (h : m ≠ n) : 
  let f : ℝ → ℝ := λ x ↦ x^2 - 2*x + 3
  f m = f n → f (m + n) = 3 := by
sorry

end NUMINAMATH_CALUDE_quadratic_function_property_l776_77677


namespace NUMINAMATH_CALUDE_share_division_l776_77611

/-- Given a total sum of 427 to be divided among three people A, B, and C,
    where 3 times A's share equals 4 times B's share equals 7 times C's share,
    C's share is 84. -/
theorem share_division (a b c : ℚ) : 
  a + b + c = 427 →
  3 * a = 4 * b →
  4 * b = 7 * c →
  c = 84 := by
  sorry

end NUMINAMATH_CALUDE_share_division_l776_77611


namespace NUMINAMATH_CALUDE_function_has_zero_l776_77626

theorem function_has_zero (m : ℝ) : ∃ x : ℝ, x^3 + 5*m*x - 2 = 0 := by
  sorry

end NUMINAMATH_CALUDE_function_has_zero_l776_77626


namespace NUMINAMATH_CALUDE_train_crossing_time_l776_77602

/-- Proves that a train with given length and speed takes the calculated time to cross a pole -/
theorem train_crossing_time (train_length : ℝ) (train_speed_kmh : ℝ) (crossing_time : ℝ) : 
  train_length = 50 →
  train_speed_kmh = 60 →
  crossing_time = 3 →
  crossing_time = train_length / (train_speed_kmh * 1000 / 3600) := by
  sorry

#check train_crossing_time

end NUMINAMATH_CALUDE_train_crossing_time_l776_77602


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l776_77607

def A : Set ℝ := {-1, 1, 2, 4}
def B : Set ℝ := {x : ℝ | -1 < x ∧ x < 3}

theorem intersection_of_A_and_B : A ∩ B = {1, 2} := by sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l776_77607


namespace NUMINAMATH_CALUDE_water_balloon_ratio_l776_77690

/-- The number of water balloons each person has -/
structure WaterBalloons where
  cynthia : ℕ
  randy : ℕ
  janice : ℕ

/-- The conditions of the problem -/
def problem_conditions (wb : WaterBalloons) : Prop :=
  wb.cynthia = 12 ∧ wb.janice = 6 ∧ wb.randy = wb.janice / 2

/-- The theorem stating the ratio of Cynthia's to Randy's water balloons -/
theorem water_balloon_ratio (wb : WaterBalloons) 
  (h : problem_conditions wb) : wb.cynthia / wb.randy = 4 := by
  sorry

end NUMINAMATH_CALUDE_water_balloon_ratio_l776_77690


namespace NUMINAMATH_CALUDE_recruitment_plans_count_l776_77646

/-- Represents the daily installation capacity of workers -/
structure WorkerCapacity where
  skilled : ℕ
  new : ℕ

/-- Represents a recruitment plan -/
structure RecruitmentPlan where
  skilled : ℕ
  new : ℕ

/-- Checks if a recruitment plan is valid given the constraints -/
def isValidPlan (plan : RecruitmentPlan) : Prop :=
  1 < plan.skilled ∧ plan.skilled < 8 ∧ 0 < plan.new

/-- Checks if a recruitment plan can complete the task -/
def canCompleteTask (capacity : WorkerCapacity) (plan : RecruitmentPlan) : Prop :=
  15 * (capacity.skilled * plan.skilled + capacity.new * plan.new) = 360

/-- Main theorem to prove -/
theorem recruitment_plans_count 
  (capacity : WorkerCapacity)
  (h1 : 2 * capacity.skilled + capacity.new = 10)
  (h2 : 3 * capacity.skilled + 2 * capacity.new = 16) :
  ∃! (plans : Finset RecruitmentPlan), 
    plans.card = 4 ∧ 
    (∀ plan ∈ plans, isValidPlan plan ∧ canCompleteTask capacity plan) ∧
    (∀ plan, isValidPlan plan ∧ canCompleteTask capacity plan → plan ∈ plans) :=
  sorry

end NUMINAMATH_CALUDE_recruitment_plans_count_l776_77646


namespace NUMINAMATH_CALUDE_quadratic_real_roots_l776_77664

theorem quadratic_real_roots (p1 p2 q1 q2 : ℝ) 
  (h : p1 * p2 = 2 * (q1 + q2)) : 
  ∃ x : ℝ, (x^2 + p1*x + q1 = 0) ∨ (x^2 + p2*x + q2 = 0) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_real_roots_l776_77664


namespace NUMINAMATH_CALUDE_trig_expression_equality_l776_77684

theorem trig_expression_equality : 
  1 / Real.cos (70 * π / 180) - 2 / Real.sin (70 * π / 180) = 
  2 * (Real.sin (50 * π / 180) - 1) / Real.sin (40 * π / 180) := by sorry

end NUMINAMATH_CALUDE_trig_expression_equality_l776_77684


namespace NUMINAMATH_CALUDE_limit_special_function_l776_77636

/-- The limit of ((x+1)/(2x))^((ln(x+2))/(ln(2-x))) as x approaches 1 is √3 -/
theorem limit_special_function :
  ∀ ε > 0, ∃ δ > 0, ∀ x : ℝ, 0 < |x - 1| ∧ |x - 1| < δ →
    |((x + 1) / (2 * x))^((Real.log (x + 2)) / (Real.log (2 - x))) - Real.sqrt 3| < ε :=
sorry

end NUMINAMATH_CALUDE_limit_special_function_l776_77636


namespace NUMINAMATH_CALUDE_complex_equation_solution_l776_77679

theorem complex_equation_solution (x y : ℝ) :
  (x * Complex.I + 2 = y - Complex.I) → (x - y = -3) := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l776_77679


namespace NUMINAMATH_CALUDE_right_triangle_acute_angles_l776_77654

theorem right_triangle_acute_angles (θ : ℝ) : 
  θ = 27 → 
  90 + θ + (90 - θ) = 180 :=
by sorry

end NUMINAMATH_CALUDE_right_triangle_acute_angles_l776_77654


namespace NUMINAMATH_CALUDE_unique_solution_power_equation_l776_77642

theorem unique_solution_power_equation :
  ∀ x y : ℕ+, 3^(x:ℕ) + 7 = 2^(y:ℕ) → x = 2 ∧ y = 4 := by sorry

end NUMINAMATH_CALUDE_unique_solution_power_equation_l776_77642


namespace NUMINAMATH_CALUDE_jared_age_difference_l776_77640

/-- Given three friends Jared, Hakimi, and Molly, this theorem proves that Jared is 10 years older than Hakimi -/
theorem jared_age_difference (jared_age hakimi_age molly_age : ℕ) : 
  hakimi_age = 40 →
  molly_age = 30 →
  (jared_age + hakimi_age + molly_age) / 3 = 40 →
  jared_age - hakimi_age = 10 := by
sorry

end NUMINAMATH_CALUDE_jared_age_difference_l776_77640


namespace NUMINAMATH_CALUDE_piggy_bank_dimes_l776_77630

theorem piggy_bank_dimes (total_value : ℚ) (total_coins : ℕ) 
  (quarter_value : ℚ) (dime_value : ℚ) :
  total_value = 39.5 ∧ 
  total_coins = 200 ∧ 
  quarter_value = 0.25 ∧ 
  dime_value = 0.1 →
  ∃ (quarters dimes : ℕ), 
    quarters + dimes = total_coins ∧
    quarter_value * quarters + dime_value * dimes = total_value ∧
    dimes = 70 := by
  sorry

end NUMINAMATH_CALUDE_piggy_bank_dimes_l776_77630


namespace NUMINAMATH_CALUDE_existence_of_integers_l776_77631

theorem existence_of_integers : ∃ (m n p q : ℕ+), 
  m ≠ n ∧ m ≠ p ∧ m ≠ q ∧ n ≠ p ∧ n ≠ q ∧ p ≠ q ∧
  (m : ℝ) + n = p + q ∧
  Real.sqrt (m : ℝ) + (n : ℝ) ^ (1/4) = Real.sqrt (p : ℝ) + (q : ℝ) ^ (1/3) ∧
  Real.sqrt (m : ℝ) + (n : ℝ) ^ (1/4) > 2004 := by
  sorry

end NUMINAMATH_CALUDE_existence_of_integers_l776_77631


namespace NUMINAMATH_CALUDE_solution_set_f_gt_2_min_a_for_full_solution_set_l776_77634

-- Define the function f
def f (x : ℝ) : ℝ := |x - 1| - |2*x + 3|

-- Theorem for part (I)
theorem solution_set_f_gt_2 :
  {x : ℝ | f x > 2} = {x : ℝ | -2 < x ∧ x < -4/3} :=
sorry

-- Theorem for part (II)
theorem min_a_for_full_solution_set (a : ℝ) :
  (∀ x, f x ≤ (3/2)*a^2 - a) ↔ a ≥ 5/3 :=
sorry

end NUMINAMATH_CALUDE_solution_set_f_gt_2_min_a_for_full_solution_set_l776_77634


namespace NUMINAMATH_CALUDE_probability_of_opening_classroom_door_l776_77621

/-- Represents a keychain with a total number of keys and a number of keys that can open a specific door. -/
structure Keychain where
  total_keys : ℕ
  opening_keys : ℕ
  h_opening_keys_le_total : opening_keys ≤ total_keys

/-- Calculates the probability of randomly selecting a key that can open the door. -/
def probability_of_opening (k : Keychain) : ℚ :=
  k.opening_keys / k.total_keys

/-- The class monitor's keychain. -/
def class_monitor_keychain : Keychain :=
  { total_keys := 5
    opening_keys := 2
    h_opening_keys_le_total := by norm_num }

theorem probability_of_opening_classroom_door :
  probability_of_opening class_monitor_keychain = 2 / 5 := by
  sorry

end NUMINAMATH_CALUDE_probability_of_opening_classroom_door_l776_77621


namespace NUMINAMATH_CALUDE_total_amount_is_500_l776_77601

/-- Calculate the total amount received after applying simple interest -/
def total_amount_with_simple_interest (principal : ℚ) (rate : ℚ) (time : ℚ) : ℚ :=
  principal + (principal * rate * time / 100)

/-- Theorem stating that the total amount received is 500 given the specified conditions -/
theorem total_amount_is_500 :
  total_amount_with_simple_interest 468.75 4 (5/3) = 500 := by
  sorry

#eval total_amount_with_simple_interest 468.75 4 (5/3)

end NUMINAMATH_CALUDE_total_amount_is_500_l776_77601


namespace NUMINAMATH_CALUDE_cost_of_25_pencils_20_notebooks_l776_77652

/-- The cost of a pencil in dollars -/
def pencil_cost : ℝ := sorry

/-- The cost of a notebook in dollars -/
def notebook_cost : ℝ := sorry

/-- The pricing conditions for pencils and notebooks -/
axiom pricing_condition_1 : 9 * pencil_cost + 10 * notebook_cost = 5.45
axiom pricing_condition_2 : 7 * pencil_cost + 6 * notebook_cost = 3.67
axiom pricing_condition_3 : 20 * pencil_cost + 15 * notebook_cost = 10.00

/-- The theorem stating the cost of 25 pencils and 20 notebooks -/
theorem cost_of_25_pencils_20_notebooks : 
  25 * pencil_cost + 20 * notebook_cost = 12.89 := by sorry

end NUMINAMATH_CALUDE_cost_of_25_pencils_20_notebooks_l776_77652


namespace NUMINAMATH_CALUDE_erica_age_is_17_l776_77656

def casper_age : ℕ := 18

def ivy_age (casper_age : ℕ) : ℕ := casper_age + 4

def erica_age (ivy_age : ℕ) : ℕ := ivy_age - 5

theorem erica_age_is_17 :
  erica_age (ivy_age casper_age) = 17 := by
  sorry

end NUMINAMATH_CALUDE_erica_age_is_17_l776_77656


namespace NUMINAMATH_CALUDE_same_grade_percentage_l776_77667

/-- Represents the number of students in the classroom -/
def total_students : ℕ := 25

/-- Represents the number of students who scored 'A' on both exams -/
def students_a : ℕ := 3

/-- Represents the number of students who scored 'B' on both exams -/
def students_b : ℕ := 2

/-- Represents the number of students who scored 'C' on both exams -/
def students_c : ℕ := 1

/-- Represents the number of students who scored 'D' on both exams -/
def students_d : ℕ := 3

/-- Calculates the total number of students who received the same grade on both exams -/
def same_grade_students : ℕ := students_a + students_b + students_c + students_d

/-- Theorem: The percentage of students who received the same grade on both exams is 36% -/
theorem same_grade_percentage :
  (same_grade_students : ℚ) / total_students * 100 = 36 := by
  sorry

end NUMINAMATH_CALUDE_same_grade_percentage_l776_77667


namespace NUMINAMATH_CALUDE_jelly_bean_ratio_l776_77612

/-- Given 1200 total jelly beans divided between two jars X and Y, where jar X has 800 jelly beans,
    prove that the ratio of jelly beans in jar X to jar Y is 2:1. -/
theorem jelly_bean_ratio :
  let total_beans : ℕ := 1200
  let jar_x : ℕ := 800
  let jar_y : ℕ := total_beans - jar_x
  (jar_x : ℚ) / jar_y = 2 := by sorry

end NUMINAMATH_CALUDE_jelly_bean_ratio_l776_77612


namespace NUMINAMATH_CALUDE_quadratic_two_distinct_roots_l776_77653

theorem quadratic_two_distinct_roots (a : ℝ) :
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ x₁^2 - 2*x₁ + a = 0 ∧ x₂^2 - 2*x₂ + a = 0) ↔ a < 1 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_two_distinct_roots_l776_77653


namespace NUMINAMATH_CALUDE_one_cow_one_bag_days_l776_77647

/-- The number of days it takes for one cow to eat one bag of husk -/
def days_for_one_cow_one_bag (num_cows : ℕ) (num_bags : ℕ) (num_days : ℕ) : ℚ :=
  (num_days : ℚ) * (num_cows : ℚ) / (num_bags : ℚ)

/-- Theorem stating that one cow will eat one bag of husk in 36 days -/
theorem one_cow_one_bag_days :
  days_for_one_cow_one_bag 60 75 45 = 36 := by
  sorry

#eval days_for_one_cow_one_bag 60 75 45

end NUMINAMATH_CALUDE_one_cow_one_bag_days_l776_77647


namespace NUMINAMATH_CALUDE_coefficient_of_x_squared_l776_77648

theorem coefficient_of_x_squared (a₀ a₁ a₂ a₃ a₄ a₅ : ℝ) :
  (∀ x : ℝ, (x - 2)^5 = a₀ + a₁*x + a₂*x^2 + a₃*x^3 + a₄*x^4 + a₅*x^5) →
  a₂ = -80 := by
  sorry

end NUMINAMATH_CALUDE_coefficient_of_x_squared_l776_77648


namespace NUMINAMATH_CALUDE_modified_lucas_units_digit_l776_77617

/-- Modified Lucas sequence -/
def L' : ℕ → ℕ
  | 0 => 3
  | 1 => 2
  | n + 2 => 2 * L' (n + 1) + L' n

/-- Function to get the units digit of a natural number -/
def unitsDigit (n : ℕ) : ℕ := n % 10

/-- Theorem stating that the units digit of L'_{L'_{20}} is d -/
theorem modified_lucas_units_digit :
  ∃ d : ℕ, d < 10 ∧ unitsDigit (L' (L' 20)) = d :=
sorry

end NUMINAMATH_CALUDE_modified_lucas_units_digit_l776_77617


namespace NUMINAMATH_CALUDE_intersection_of_perpendicular_tangents_on_parabola_l776_77698

/-- Given a parabola y = 4x and two points on it with perpendicular tangents,
    prove that the x-coordinate of the intersection of these tangents is -1. -/
theorem intersection_of_perpendicular_tangents_on_parabola
  (x₁ y₁ x₂ y₂ : ℝ) 
  (h₁ : y₁ = 4 * x₁)
  (h₂ : y₂ = 4 * x₂)
  (h_perp : (4 / y₁) * (4 / y₂) = -1) :
  ∃ (x y : ℝ), x = -1 ∧ 
    y = (4 / y₁) * x + y₁ / 2 ∧
    y = (4 / y₂) * x + y₂ / 2 :=
sorry

end NUMINAMATH_CALUDE_intersection_of_perpendicular_tangents_on_parabola_l776_77698


namespace NUMINAMATH_CALUDE_square_of_sum_fifteen_three_l776_77643

theorem square_of_sum_fifteen_three : 15^2 + 2*(15*3) + 3^2 = 324 := by
  sorry

end NUMINAMATH_CALUDE_square_of_sum_fifteen_three_l776_77643


namespace NUMINAMATH_CALUDE_complex_modulus_squared_l776_77663

theorem complex_modulus_squared (z : ℂ) (h1 : z + Complex.abs z = 6 + 2*I) 
  (h2 : z.re ≥ 0) : Complex.abs z ^ 2 = 100 / 9 := by
  sorry

end NUMINAMATH_CALUDE_complex_modulus_squared_l776_77663


namespace NUMINAMATH_CALUDE_number_of_rolls_not_random_variable_l776_77637

/-- A type representing the outcome of a single die roll -/
inductive DieRoll
| one | two | three | four | five | six

/-- A type representing a random variable on die rolls -/
def RandomVariable := DieRoll → ℝ

/-- The number of times the die is rolled -/
def numberOfRolls : ℕ := 2

theorem number_of_rolls_not_random_variable :
  ¬ ∃ (f : RandomVariable), ∀ (r₁ r₂ : DieRoll), f r₁ = numberOfRolls ∧ f r₂ = numberOfRolls :=
sorry

end NUMINAMATH_CALUDE_number_of_rolls_not_random_variable_l776_77637


namespace NUMINAMATH_CALUDE_monomial_degree_implications_l776_77681

-- Define the condition
def is_monomial_of_degree_5 (a : ℝ) : Prop :=
  2 + (1 + a) = 5

-- Theorem statement
theorem monomial_degree_implications (a : ℝ) 
  (h : is_monomial_of_degree_5 a) : 
  a^3 + 1 = 9 ∧ 
  (a + 1) * (a^2 - a + 1) = 9 ∧ 
  a^3 + 1 = (a + 1) * (a^2 - a + 1) := by
  sorry

end NUMINAMATH_CALUDE_monomial_degree_implications_l776_77681


namespace NUMINAMATH_CALUDE_reciprocal_roots_imply_a_eq_neg_one_l776_77692

theorem reciprocal_roots_imply_a_eq_neg_one (a : ℝ) : 
  (∃ x y : ℝ, x ≠ 0 ∧ y ≠ 0 ∧ 
    x^2 + (a-1)*x + a^2 = 0 ∧ 
    y^2 + (a-1)*y + a^2 = 0 ∧ 
    x*y = 1) → 
  a = -1 :=
by sorry

end NUMINAMATH_CALUDE_reciprocal_roots_imply_a_eq_neg_one_l776_77692


namespace NUMINAMATH_CALUDE_eleanor_distance_between_meetings_l776_77618

/-- The distance of the circular track in meters -/
def track_length : ℝ := 720

/-- The time Eric takes to complete one circuit in minutes -/
def eric_time : ℝ := 4

/-- The time Eleanor takes to complete one circuit in minutes -/
def eleanor_time : ℝ := 5

/-- The theorem stating the distance Eleanor runs between consecutive meetings -/
theorem eleanor_distance_between_meetings :
  let eric_speed := track_length / eric_time
  let eleanor_speed := track_length / eleanor_time
  let relative_speed := eric_speed + eleanor_speed
  let time_between_meetings := track_length / relative_speed
  eleanor_speed * time_between_meetings = 320 := by
  sorry

end NUMINAMATH_CALUDE_eleanor_distance_between_meetings_l776_77618


namespace NUMINAMATH_CALUDE_carpet_reconstruction_l776_77609

theorem carpet_reconstruction (original_length original_width cut_length cut_width new_side : ℝ) 
  (h1 : original_length = 12)
  (h2 : original_width = 9)
  (h3 : cut_length = 8)
  (h4 : cut_width = 1)
  (h5 : new_side = 10) :
  original_length * original_width - cut_length * cut_width = new_side * new_side := by
sorry

end NUMINAMATH_CALUDE_carpet_reconstruction_l776_77609


namespace NUMINAMATH_CALUDE_can_guess_number_l776_77699

theorem can_guess_number (n : Nat) (q : Nat) : n ≤ 1000 → q = 10 → 2^q ≥ n → True := by
  sorry

end NUMINAMATH_CALUDE_can_guess_number_l776_77699


namespace NUMINAMATH_CALUDE_systematic_sampling_theorem_l776_77622

/-- Represents a systematic sampling of students. -/
structure SystematicSampling where
  total_students : ℕ
  num_groups : ℕ
  students_per_group : ℕ
  selected_number : ℕ
  selected_group : ℕ

/-- Calculates the number of the selected student in a given group. -/
def selected_number_in_group (s : SystematicSampling) (group : ℕ) : ℕ :=
  s.selected_number + s.students_per_group * (group - s.selected_group)

/-- Theorem stating that if student 12 is selected from group 3, 
    then student 37 will be selected from group 8 in a systematic sampling of 50 students. -/
theorem systematic_sampling_theorem (s : SystematicSampling) :
  s.total_students = 50 ∧ 
  s.num_groups = 10 ∧ 
  s.students_per_group = 5 ∧ 
  s.selected_number = 12 ∧ 
  s.selected_group = 3 →
  selected_number_in_group s 8 = 37 := by
  sorry

end NUMINAMATH_CALUDE_systematic_sampling_theorem_l776_77622


namespace NUMINAMATH_CALUDE_unique_cube_root_between_9_and_9_2_l776_77671

theorem unique_cube_root_between_9_and_9_2 :
  ∃! n : ℕ+, 27 ∣ n ∧ 9 < (n : ℝ)^(1/3) ∧ (n : ℝ)^(1/3) < 9.2 :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_unique_cube_root_between_9_and_9_2_l776_77671


namespace NUMINAMATH_CALUDE_non_chihuahua_male_dogs_l776_77610

theorem non_chihuahua_male_dogs (total_dogs : ℕ) (male_ratio : ℚ) (chihuahua_ratio : ℚ) :
  total_dogs = 32 →
  male_ratio = 5/8 →
  chihuahua_ratio = 3/4 →
  (total_dogs : ℚ) * male_ratio * (1 - chihuahua_ratio) = 5 := by
  sorry

end NUMINAMATH_CALUDE_non_chihuahua_male_dogs_l776_77610


namespace NUMINAMATH_CALUDE_jane_age_problem_l776_77613

def is_perfect_square (n : ℕ) : Prop := ∃ k : ℕ, n = k^2

def is_perfect_cube (n : ℕ) : Prop := ∃ k : ℕ, n = k^3

theorem jane_age_problem :
  ∃ j : ℕ, j > 0 ∧ is_perfect_square (j - 2) ∧ is_perfect_cube (j + 2) ∧
  ∀ k : ℕ, k > 0 → is_perfect_square (k - 2) → is_perfect_cube (k + 2) → j ≤ k :=
sorry

end NUMINAMATH_CALUDE_jane_age_problem_l776_77613


namespace NUMINAMATH_CALUDE_complement_intersection_theorem_l776_77627

def U : Set Nat := {1, 2, 3, 4, 5, 6}
def M : Set Nat := {2, 3}
def N : Set Nat := {1, 3}

theorem complement_intersection_theorem : 
  {4, 5, 6} = (U \ M) ∩ (U \ N) := by sorry

end NUMINAMATH_CALUDE_complement_intersection_theorem_l776_77627


namespace NUMINAMATH_CALUDE_base_conversion_2200_to_base9_l776_77676

-- Define a function to convert a base 9 number to base 10
def base9ToBase10 (digits : List Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * (9 ^ i)) 0

-- Theorem statement
theorem base_conversion_2200_to_base9 :
  base9ToBase10 [4, 1, 0, 3] = 2200 := by
  sorry

end NUMINAMATH_CALUDE_base_conversion_2200_to_base9_l776_77676


namespace NUMINAMATH_CALUDE_sally_grew_five_onions_l776_77658

/-- The number of onions grown by Sara -/
def sara_onions : ℕ := 4

/-- The number of onions grown by Fred -/
def fred_onions : ℕ := 9

/-- The total number of onions grown -/
def total_onions : ℕ := 18

/-- The number of onions grown by Sally -/
def sally_onions : ℕ := total_onions - (sara_onions + fred_onions)

theorem sally_grew_five_onions : sally_onions = 5 := by
  sorry

end NUMINAMATH_CALUDE_sally_grew_five_onions_l776_77658


namespace NUMINAMATH_CALUDE_power_equation_solution_l776_77665

theorem power_equation_solution (n : ℕ) : 2^(2*n) + 2^(2*n) + 2^(2*n) + 2^(2*n) = 4^28 → n = 27 := by
  sorry

end NUMINAMATH_CALUDE_power_equation_solution_l776_77665


namespace NUMINAMATH_CALUDE_sqrt_of_nine_l776_77688

theorem sqrt_of_nine (x : ℝ) : x = Real.sqrt 9 → x = 3 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_of_nine_l776_77688


namespace NUMINAMATH_CALUDE_cross_product_solution_l776_77624

theorem cross_product_solution :
  let v1 : ℝ × ℝ × ℝ := (128/15, -2, 7/5)
  let v2 : ℝ × ℝ × ℝ := (4, 5, 3)
  let result : ℝ × ℝ × ℝ := (-13, -20, 23)
  (v1.2.1 * v2.2.2 - v1.2.2 * v2.2.1,
   v1.2.2 * v2.1 - v1.1 * v2.2.2,
   v1.1 * v2.2.1 - v1.2.1 * v2.1) = result :=
by sorry

end NUMINAMATH_CALUDE_cross_product_solution_l776_77624


namespace NUMINAMATH_CALUDE_special_geometric_sequence_ratio_l776_77666

/-- An increasing geometric sequence with specific conditions -/
structure SpecialGeometricSequence where
  a : ℕ → ℝ
  increasing : ∀ n, a n < a (n + 1)
  geometric : ∃ q > 1, ∀ n, a (n + 1) = q * a n
  sum_condition : a 1 + a 4 = 9
  product_condition : a 2 * a 3 = 8

/-- The common ratio of the special geometric sequence is 2 -/
theorem special_geometric_sequence_ratio (seq : SpecialGeometricSequence) :
  ∃ q > 1, (∀ n, seq.a (n + 1) = q * seq.a n) ∧ q = 2 := by sorry

end NUMINAMATH_CALUDE_special_geometric_sequence_ratio_l776_77666


namespace NUMINAMATH_CALUDE_shared_edge_angle_l776_77651

-- Define the angle of a regular decagon
def decagon_angle : ℝ := 144

-- Define the angle of a square
def square_angle : ℝ := 90

-- Theorem statement
theorem shared_edge_angle (x : ℝ) : 
  x + 36 + (360 - decagon_angle) + square_angle = 360 → x = 18 := by
  sorry

end NUMINAMATH_CALUDE_shared_edge_angle_l776_77651


namespace NUMINAMATH_CALUDE_point_on_circle_l776_77625

def unit_circle (x y : ℝ) : Prop := x^2 + y^2 = 1

def arc_length (θ : ℝ) : ℝ := θ

theorem point_on_circle (start_x start_y end_x end_y θ : ℝ) : 
  unit_circle start_x start_y →
  unit_circle end_x end_y →
  arc_length θ = π/3 →
  start_x = 1 →
  start_y = 0 →
  end_x = 1/2 →
  end_y = Real.sqrt 3 / 2 :=
sorry

end NUMINAMATH_CALUDE_point_on_circle_l776_77625


namespace NUMINAMATH_CALUDE_installment_value_approximation_l776_77645

def tv_price : ℕ := 15000
def num_installments : ℕ := 20
def interest_rate : ℚ := 6 / 100
def last_installment : ℕ := 13000

def calculate_installment_value (price : ℕ) (num_inst : ℕ) (rate : ℚ) (last_inst : ℕ) : ℚ :=
  let avg_balance : ℚ := price / 2
  let interest : ℚ := avg_balance * rate
  let total_amount : ℚ := price + interest
  (total_amount - last_inst) / (num_inst - 1)

theorem installment_value_approximation :
  ∃ (ε : ℚ), ε > 0 ∧ ε < 1 ∧ 
  |calculate_installment_value tv_price num_installments interest_rate last_installment - 129| < ε :=
sorry

end NUMINAMATH_CALUDE_installment_value_approximation_l776_77645


namespace NUMINAMATH_CALUDE_value_of_M_l776_77661

theorem value_of_M : ∃ M : ℝ, (0.25 * M = 0.55 * 1500) ∧ (M = 3300) := by
  sorry

end NUMINAMATH_CALUDE_value_of_M_l776_77661


namespace NUMINAMATH_CALUDE_original_average_proof_l776_77695

theorem original_average_proof (n : ℕ) (original_avg new_avg : ℚ) :
  n > 0 →
  new_avg = 2 * original_avg →
  new_avg = 72 →
  original_avg = 36 := by
sorry

end NUMINAMATH_CALUDE_original_average_proof_l776_77695


namespace NUMINAMATH_CALUDE_travis_cereal_expenditure_l776_77655

theorem travis_cereal_expenditure 
  (boxes_per_week : ℕ) 
  (cost_per_box : ℚ) 
  (weeks_per_year : ℕ) 
  (h1 : boxes_per_week = 2) 
  (h2 : cost_per_box = 3) 
  (h3 : weeks_per_year = 52) : 
  (boxes_per_week * cost_per_box * weeks_per_year : ℚ) = 312 :=
by sorry

end NUMINAMATH_CALUDE_travis_cereal_expenditure_l776_77655


namespace NUMINAMATH_CALUDE_head_start_calculation_l776_77670

/-- Represents a runner in the race -/
inductive Runner : Type
  | A | B | C | D

/-- The head start (in meters) that Runner A can give to another runner -/
def headStart (r : Runner) : ℕ :=
  match r with
  | Runner.A => 0
  | Runner.B => 150
  | Runner.C => 310
  | Runner.D => 400

/-- The head start one runner can give to another -/
def headStartBetween (r1 r2 : Runner) : ℤ :=
  (headStart r2 : ℤ) - (headStart r1 : ℤ)

theorem head_start_calculation :
  (headStartBetween Runner.B Runner.C = 160) ∧
  (headStartBetween Runner.C Runner.D = 90) ∧
  (headStartBetween Runner.B Runner.D = 250) := by
  sorry

#check head_start_calculation

end NUMINAMATH_CALUDE_head_start_calculation_l776_77670


namespace NUMINAMATH_CALUDE_nickel_difference_is_zero_l776_77604

/-- Represents the number of coins of each type -/
structure CoinCollection where
  nickels : ℕ
  dimes : ℕ
  quarters : ℕ

/-- The total number of coins -/
def total_coins : ℕ := 150

/-- The total value of the coins in cents -/
def total_value : ℕ := 2000

/-- The value of a nickel in cents -/
def nickel_value : ℕ := 5

/-- The value of a dime in cents -/
def dime_value : ℕ := 10

/-- The value of a quarter in cents -/
def quarter_value : ℕ := 25

/-- Checks if a coin collection is valid -/
def is_valid_collection (c : CoinCollection) : Prop :=
  c.nickels + c.dimes + c.quarters = total_coins ∧
  c.nickels * nickel_value + c.dimes * dime_value + c.quarters * quarter_value = total_value ∧
  c.nickels > 0 ∧ c.dimes > 0 ∧ c.quarters > 0

/-- The theorem to be proved -/
theorem nickel_difference_is_zero :
  ∃ (min_nickels max_nickels : ℕ),
    (∀ c : CoinCollection, is_valid_collection c → c.nickels ≥ min_nickels) ∧
    (∀ c : CoinCollection, is_valid_collection c → c.nickels ≤ max_nickels) ∧
    max_nickels - min_nickels = 0 := by
  sorry

end NUMINAMATH_CALUDE_nickel_difference_is_zero_l776_77604


namespace NUMINAMATH_CALUDE_A_n_nonempty_finite_l776_77691

/-- The set A_n for a positive integer n -/
def A_n (n : ℕ+) : Set (ℕ × ℕ) :=
  {p : ℕ × ℕ | ∃ k : ℕ, (Real.sqrt (p.1^2 + p.2 + n) + Real.sqrt (p.2^2 + p.1 + n) : ℝ) = k}

/-- Theorem stating that A_n is non-empty and finite for any positive integer n -/
theorem A_n_nonempty_finite (n : ℕ+) : Set.Nonempty (A_n n) ∧ Set.Finite (A_n n) := by
  sorry

end NUMINAMATH_CALUDE_A_n_nonempty_finite_l776_77691


namespace NUMINAMATH_CALUDE_motel_pricing_l776_77697

/-- Represents the motel's pricing structure and guest stays. -/
structure MotelStay where
  flatFee : ℝ
  regularRate : ℝ
  discountRate : ℝ
  markStay : ℕ
  markCost : ℝ
  lucyStay : ℕ
  lucyCost : ℝ

/-- The motel's pricing satisfies the given conditions. -/
def validPricing (m : MotelStay) : Prop :=
  m.discountRate = 0.8 * m.regularRate ∧
  m.markStay = 5 ∧
  m.lucyStay = 7 ∧
  m.markCost = m.flatFee + 3 * m.regularRate + 2 * m.discountRate ∧
  m.lucyCost = m.flatFee + 3 * m.regularRate + 4 * m.discountRate ∧
  m.markCost = 310 ∧
  m.lucyCost = 410

/-- The theorem stating the correct flat fee and regular rate. -/
theorem motel_pricing (m : MotelStay) (h : validPricing m) :
  m.flatFee = 22.5 ∧ m.regularRate = 62.5 := by
  sorry

end NUMINAMATH_CALUDE_motel_pricing_l776_77697


namespace NUMINAMATH_CALUDE_hockey_championship_wins_l776_77694

theorem hockey_championship_wins (total_matches : ℕ) (total_points : ℤ) 
  (win_points loss_points : ℤ) (h_total_matches : total_matches = 38) 
  (h_total_points : total_points = 60) (h_win_points : win_points = 12) 
  (h_loss_points : loss_points = 5) : 
  ∃! wins : ℕ, ∃ losses draws : ℕ,
    wins + losses + draws = total_matches ∧ 
    wins * win_points - losses * loss_points = total_points ∧
    losses > 0 := by
  sorry

#check hockey_championship_wins

end NUMINAMATH_CALUDE_hockey_championship_wins_l776_77694


namespace NUMINAMATH_CALUDE_jane_played_eight_rounds_l776_77649

/-- Represents a card game where players earn points for winning rounds. -/
structure CardGame where
  /-- The number of points awarded for winning a round. -/
  points_per_round : ℕ
  /-- The final number of points a player has. -/
  final_points : ℕ
  /-- The number of points a player lost during the game. -/
  lost_points : ℕ

/-- Calculates the number of rounds played in a card game. -/
def rounds_played (game : CardGame) : ℕ :=
  (game.final_points + game.lost_points) / game.points_per_round

/-- Theorem stating that Jane played 8 rounds in the card game. -/
theorem jane_played_eight_rounds :
  let game := CardGame.mk 10 60 20
  rounds_played game = 8 := by
  sorry

end NUMINAMATH_CALUDE_jane_played_eight_rounds_l776_77649


namespace NUMINAMATH_CALUDE_smallest_next_divisor_after_437_l776_77662

theorem smallest_next_divisor_after_437 (m : ℕ) (h1 : 1000 ≤ m ∧ m ≤ 9999) 
  (h2 : Even m) (h3 : m % 437 = 0) : 
  (∀ d : ℕ, d ∣ m → 437 < d → d ≥ 874) ∧ 874 ∣ m :=
sorry

end NUMINAMATH_CALUDE_smallest_next_divisor_after_437_l776_77662


namespace NUMINAMATH_CALUDE_intersection_distance_product_l776_77686

/-- Given an ellipse and a hyperbola sharing the same foci, the product of distances
    from their intersection point to the foci is equal to the difference of their
    respective parameters. -/
theorem intersection_distance_product (a b m n : ℝ) (P F₁ F₂ : ℝ × ℝ) :
  a > b ∧ b > 0 ∧ m > 0 ∧ n > 0 →
  (P.1^2 / a + P.2^2 / b = 1) →
  (P.1^2 / m - P.2^2 / n = 1) →
  (∀ Q : ℝ × ℝ, Q.1^2 / a + Q.2^2 / b = 1 → dist Q F₁ + dist Q F₂ = 2 * Real.sqrt a) →
  (∀ R : ℝ × ℝ, R.1^2 / m - R.2^2 / n = 1 → |dist R F₁ - dist R F₂| = 2 * Real.sqrt m) →
  dist P F₁ * dist P F₂ = a - m :=
by sorry

end NUMINAMATH_CALUDE_intersection_distance_product_l776_77686


namespace NUMINAMATH_CALUDE_petya_winning_strategy_exists_l776_77657

/-- Represents a player in the coin game -/
inductive Player
| Vasya
| Petya

/-- Represents the state of the game -/
structure GameState where
  chests : Nat
  coins : Nat
  currentPlayer : Player

/-- Defines a strategy for Petya -/
def PetyaStrategy := GameState → Nat

/-- Checks if a game state is valid -/
def isValidGameState (state : GameState) : Prop :=
  state.chests > 0 ∧ state.coins ≥ state.chests

/-- Represents the initial game state -/
def initialState : GameState :=
  { chests := 1011, coins := 2022, currentPlayer := Player.Vasya }

/-- Theorem stating Petya's winning strategy exists -/
theorem petya_winning_strategy_exists :
  ∃ (strategy : PetyaStrategy),
    ∀ (game : GameState),
      isValidGameState game →
      game.coins = 2 →
      ∃ (chest : Nat),
        chest < game.chests ∧
        strategy game = chest :=
  sorry

end NUMINAMATH_CALUDE_petya_winning_strategy_exists_l776_77657


namespace NUMINAMATH_CALUDE_roots_properties_l776_77687

theorem roots_properties (r s t : ℝ) : 
  (∀ x : ℝ, x * (x - 2) * (3 * x - 7) = 2 ↔ x = r ∨ x = s ∨ x = t) →
  (r > 0 ∧ s > 0 ∧ t > 0) ∧
  (Real.arctan r + Real.arctan s + Real.arctan t = 3 * π / 4) := by
sorry

end NUMINAMATH_CALUDE_roots_properties_l776_77687


namespace NUMINAMATH_CALUDE_factory_profit_l776_77669

noncomputable section

-- Define the daily cost function
def C (x : ℝ) : ℝ := 3 + x

-- Define the daily sales revenue function
def S (x k : ℝ) : ℝ := 
  if 0 < x ∧ x < 6 then 3*x + k/(x-8) + 5
  else if x ≥ 6 then 14
  else 0  -- undefined for x ≤ 0

-- Define the daily profit function
def L (x k : ℝ) : ℝ := S x k - C x

-- State the theorem
theorem factory_profit (k : ℝ) :
  (L 2 k = 3) →  -- Condition: when x = 2, L = 3
  (k = 18 ∧ 
   ∀ x, 0 < x → L x k ≤ 6 ∧
   L 5 k = 6) := by
  sorry

end

end NUMINAMATH_CALUDE_factory_profit_l776_77669


namespace NUMINAMATH_CALUDE_hyperbola_to_ellipse_l776_77660

/-- Given a hyperbola with equation x²/4 - y²/12 = -1, 
    the ellipse with foci at the vertices of this hyperbola 
    has the equation x²/4 + y²/16 = 1 -/
theorem hyperbola_to_ellipse : 
  ∃ (h : Set (ℝ × ℝ)) (e : Set (ℝ × ℝ)),
    (h = {(x, y) | x^2/4 - y^2/12 = -1}) →
    (e = {(x, y) | x^2/4 + y^2/16 = 1}) →
    (∀ (fx fy : ℝ), (fx, fy) ∈ {v | v ∈ h ∧ (∀ (x y : ℝ), (x, y) ∈ h → x^2 + y^2 ≤ fx^2 + fy^2)} →
      (fx, fy) ∈ {f | f ∈ e ∧ (∀ (x y : ℝ), (x, y) ∈ e → (x - fx)^2 + (y - fy)^2 ≥ 
        (x + fx)^2 + (y + fy)^2)}) :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_to_ellipse_l776_77660


namespace NUMINAMATH_CALUDE_malfunctioning_mix_sum_l776_77668

/-- Represents the fractional composition of Papaya Splash ingredients -/
structure PapayaSplash :=
  (soda_water : ℚ)
  (lemon_juice : ℚ)
  (sugar : ℚ)
  (papaya_puree : ℚ)
  (secret_spice : ℚ)
  (lime_extract : ℚ)

/-- The standard formula for Papaya Splash -/
def standard_formula : PapayaSplash :=
  { soda_water := 8/21,
    lemon_juice := 4/21,
    sugar := 3/21,
    papaya_puree := 3/21,
    secret_spice := 2/21,
    lime_extract := 1/21 }

/-- The malfunctioning machine's mixing ratios -/
def malfunction_ratios : PapayaSplash :=
  { soda_water := 1/2,
    lemon_juice := 3,
    sugar := 2,
    papaya_puree := 1,
    secret_spice := 1/5,
    lime_extract := 1 }

/-- Applies the malfunction ratios to the standard formula -/
def apply_malfunction (formula : PapayaSplash) (ratios : PapayaSplash) : PapayaSplash :=
  { soda_water := formula.soda_water * ratios.soda_water,
    lemon_juice := formula.lemon_juice * ratios.lemon_juice,
    sugar := formula.sugar * ratios.sugar,
    papaya_puree := formula.papaya_puree * ratios.papaya_puree,
    secret_spice := formula.secret_spice * ratios.secret_spice,
    lime_extract := formula.lime_extract * ratios.lime_extract }

/-- Calculates the sum of soda water, sugar, and secret spice blend fractions -/
def sum_selected_ingredients (mix : PapayaSplash) : ℚ :=
  mix.soda_water + mix.sugar + mix.secret_spice

/-- Theorem stating that the sum of selected ingredients in the malfunctioning mix is 52/105 -/
theorem malfunctioning_mix_sum :
  sum_selected_ingredients (apply_malfunction standard_formula malfunction_ratios) = 52/105 :=
sorry

end NUMINAMATH_CALUDE_malfunctioning_mix_sum_l776_77668


namespace NUMINAMATH_CALUDE_sum_of_sequences_l776_77639

/-- Sum of arithmetic sequence with 5 terms -/
def arithmetic_sum (a₁ : ℕ) : ℕ := a₁ + (a₁ + 10) + (a₁ + 20) + (a₁ + 30) + (a₁ + 40)

/-- The sum of two specific arithmetic sequences equals 270 -/
theorem sum_of_sequences : arithmetic_sum 3 + arithmetic_sum 11 = 270 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_sequences_l776_77639


namespace NUMINAMATH_CALUDE_fraction_meaningful_l776_77606

theorem fraction_meaningful (x : ℝ) : 
  (∃ y : ℝ, y = x / (x - 1)) ↔ x ≠ 1 :=
sorry

end NUMINAMATH_CALUDE_fraction_meaningful_l776_77606
