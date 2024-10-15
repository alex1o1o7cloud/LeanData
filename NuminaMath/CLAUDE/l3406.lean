import Mathlib

namespace NUMINAMATH_CALUDE_personal_preference_invalid_l3406_340627

/-- Represents the principles of sample selection --/
structure SampleSelectionPrinciples where
  representativeness : Bool
  randomness : Bool
  adequateSize : Bool

/-- Represents a sample selection method --/
inductive SampleSelectionMethod
  | Random
  | Representative
  | LargeEnough
  | PersonalPreference

/-- Checks if a sample selection method adheres to the principles --/
def isValidMethod (principles : SampleSelectionPrinciples) (method : SampleSelectionMethod) : Prop :=
  match method with
  | .Random => principles.randomness
  | .Representative => principles.representativeness
  | .LargeEnough => principles.adequateSize
  | .PersonalPreference => False

/-- Theorem stating that personal preference is not a valid sample selection method --/
theorem personal_preference_invalid (principles : SampleSelectionPrinciples) :
  ¬(isValidMethod principles SampleSelectionMethod.PersonalPreference) := by
  sorry


end NUMINAMATH_CALUDE_personal_preference_invalid_l3406_340627


namespace NUMINAMATH_CALUDE_number_composition_l3406_340606

def number_from_parts (ten_thousands : ℕ) (ones : ℕ) : ℕ :=
  ten_thousands * 10000 + ones

theorem number_composition :
  number_from_parts 45 64 = 450064 := by
  sorry

end NUMINAMATH_CALUDE_number_composition_l3406_340606


namespace NUMINAMATH_CALUDE_ch4_formation_and_consumption_l3406_340634

/-- Represents a chemical compound with its coefficient in a reaction --/
structure Compound where
  name : String
  coefficient : ℚ

/-- Represents a chemical reaction --/
structure Reaction where
  reactants : List Compound
  products : List Compound

/-- Represents the initial conditions of the problem --/
structure InitialConditions where
  be2c : ℚ
  h2o : ℚ
  o2 : ℚ
  temperature : ℚ
  pressure : ℚ

/-- The first reaction: Be2C + 4H2O → 2Be(OH)2 + CH4 --/
def reaction1 : Reaction := {
  reactants := [⟨"Be2C", 1⟩, ⟨"H2O", 4⟩],
  products := [⟨"Be(OH)2", 2⟩, ⟨"CH4", 1⟩]
}

/-- The second reaction: CH4 + 2O2 → CO2 + 2H2O --/
def reaction2 : Reaction := {
  reactants := [⟨"CH4", 1⟩, ⟨"O2", 2⟩],
  products := [⟨"CO2", 1⟩, ⟨"H2O", 2⟩]
}

/-- The initial conditions of the problem --/
def initialConditions : InitialConditions := {
  be2c := 3,
  h2o := 15,
  o2 := 6,
  temperature := 350,
  pressure := 2
}

/-- Theorem stating the amount of CH4 formed and remaining --/
theorem ch4_formation_and_consumption 
  (r1 : Reaction)
  (r2 : Reaction)
  (ic : InitialConditions)
  (h1 : r1 = reaction1)
  (h2 : r2 = reaction2)
  (h3 : ic = initialConditions) :
  ∃ (ch4_formed : ℚ) (ch4_remaining : ℚ),
    ch4_formed = 3 ∧ ch4_remaining = 0 :=
  sorry


end NUMINAMATH_CALUDE_ch4_formation_and_consumption_l3406_340634


namespace NUMINAMATH_CALUDE_c_equals_square_l3406_340689

/-- The sequence of positive perfect squares -/
def perfect_squares : ℕ → ℕ := λ n => n^2

/-- The nth term of the sequence formed by arranging all positive perfect squares in ascending order -/
def c : ℕ → ℕ := perfect_squares

/-- Theorem: For all positive integers n, c(n) = n^2 -/
theorem c_equals_square (n : ℕ) : c n = n^2 := by sorry

end NUMINAMATH_CALUDE_c_equals_square_l3406_340689


namespace NUMINAMATH_CALUDE_jakes_allowance_l3406_340640

/-- 
Given:
- An amount x (in cents)
- One-quarter of x can buy 5 items
- Each item costs 20 cents

Prove that x = 400 cents ($4.00)
-/
theorem jakes_allowance (x : ℕ) 
  (h1 : x / 4 = 5 * 20) : 
  x = 400 := by
sorry

end NUMINAMATH_CALUDE_jakes_allowance_l3406_340640


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l3406_340660

def set_A : Set ℝ := {x | 2 * x < 2 + x}
def set_B : Set ℝ := {x | 5 - x > 8 - 4 * x}

theorem intersection_of_A_and_B : set_A ∩ set_B = {x : ℝ | 1 < x ∧ x < 2} := by sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l3406_340660


namespace NUMINAMATH_CALUDE_difference_of_sum_and_difference_of_squares_l3406_340617

theorem difference_of_sum_and_difference_of_squares (x y : ℝ) 
  (h1 : x + y = 8) (h2 : x^2 - y^2 = 16) : x - y = 2 := by
  sorry

end NUMINAMATH_CALUDE_difference_of_sum_and_difference_of_squares_l3406_340617


namespace NUMINAMATH_CALUDE_probability_white_and_red_l3406_340628

/-- The probability of drawing one white ball and one red ball from a box 
    containing 7 white balls, 8 black balls, and 1 red ball, 
    when two balls are drawn at random. -/
theorem probability_white_and_red (white : ℕ) (black : ℕ) (red : ℕ) : 
  white = 7 → black = 8 → red = 1 → 
  (white * red : ℚ) / (Nat.choose (white + black + red) 2) = 7 / 120 := by
  sorry

end NUMINAMATH_CALUDE_probability_white_and_red_l3406_340628


namespace NUMINAMATH_CALUDE_exponent_properties_l3406_340646

theorem exponent_properties (a b : ℝ) (n : ℕ) :
  (a * b) ^ n = a ^ n * b ^ n ∧
  2 ^ 5 * (-1/2) ^ 5 = -1 ∧
  (-0.125) ^ 2022 * 2 ^ 2021 * 4 ^ 2020 = 1/32 := by
  sorry

end NUMINAMATH_CALUDE_exponent_properties_l3406_340646


namespace NUMINAMATH_CALUDE_arithmetic_sequence_1005th_term_l3406_340650

/-- An arithmetic sequence with specific properties -/
structure ArithmeticSequence where
  p : ℝ
  r : ℝ
  first_term : ℝ := p
  second_term : ℝ := 11
  third_term : ℝ := 4*p - r
  fourth_term : ℝ := 4*p + r

/-- The nth term of the arithmetic sequence -/
def nth_term (seq : ArithmeticSequence) (n : ℕ) : ℝ :=
  seq.first_term + (n - 1) * (seq.second_term - seq.first_term)

/-- Theorem stating that the 1005th term of the sequence is 6029 -/
theorem arithmetic_sequence_1005th_term (seq : ArithmeticSequence) :
  nth_term seq 1005 = 6029 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_1005th_term_l3406_340650


namespace NUMINAMATH_CALUDE_min_value_of_expression_l3406_340601

theorem min_value_of_expression (x y : ℝ) (hx : x > 0) (hy : y > 0) 
  (h : 1/x + 1/y = 1) :
  ∃ (m : ℝ), m = 11 + 4 * Real.sqrt 6 ∧ 
  ∀ (a b : ℝ), a > 0 → b > 0 → 1/a + 1/b = 1 → 
  3*a/(a-1) + 8*b/(b-1) ≥ m :=
sorry

end NUMINAMATH_CALUDE_min_value_of_expression_l3406_340601


namespace NUMINAMATH_CALUDE_simplify_expression_l3406_340657

theorem simplify_expression (b : ℝ) : 3 * b * (3 * b^2 + 2 * b) - 2 * b^2 = 9 * b^3 + 4 * b^2 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l3406_340657


namespace NUMINAMATH_CALUDE_cone_slant_height_l3406_340693

/-- The slant height of a cone with base radius 1 and lateral surface that unfolds into a semicircle -/
def slant_height : ℝ := 2

/-- The base radius of the cone -/
def base_radius : ℝ := 1

/-- Theorem: The slant height of a cone with base radius 1 and lateral surface that unfolds into a semicircle is 2 -/
theorem cone_slant_height :
  let r := base_radius
  let s := slant_height
  r = 1 ∧ 2 * π * r = π * s → s = 2 :=
by sorry

end NUMINAMATH_CALUDE_cone_slant_height_l3406_340693


namespace NUMINAMATH_CALUDE_ghi_equilateral_same_circumcenter_l3406_340649

-- Define the points
variable (A B C D E F G H I G' H' I' : ℝ × ℝ)

-- Define the triangles
def triangle (P Q R : ℝ × ℝ) := Set.insert P (Set.insert Q (Set.singleton R))

-- Define equilateral triangle
def is_equilateral (t : Set (ℝ × ℝ)) : Prop :=
  ∃ P Q R, t = triangle P Q R ∧ 
    dist P Q = dist Q R ∧ dist Q R = dist R P

-- Define reflection
def reflect (P Q R : ℝ × ℝ) : ℝ × ℝ := sorry

-- Define circumcenter
def circumcenter (t : Set (ℝ × ℝ)) : ℝ × ℝ := sorry

-- Assumptions
variable (h1 : is_equilateral (triangle A B D))
variable (h2 : is_equilateral (triangle A C E))
variable (h3 : is_equilateral (triangle B C F))
variable (h4 : G = circumcenter (triangle A B D))
variable (h5 : H = circumcenter (triangle A C E))
variable (h6 : I = circumcenter (triangle B C F))
variable (h7 : G' = reflect B A G)
variable (h8 : H' = reflect C B H)
variable (h9 : I' = reflect A C I)

-- Theorem statements
theorem ghi_equilateral :
  is_equilateral (triangle G H I) := sorry

theorem same_circumcenter :
  circumcenter (triangle G H I) = circumcenter (triangle G' H' I') := sorry

end NUMINAMATH_CALUDE_ghi_equilateral_same_circumcenter_l3406_340649


namespace NUMINAMATH_CALUDE_alex_has_more_listens_l3406_340621

/-- Calculates total listens over 3 months given initial listens and monthly growth rate -/
def totalListens (initial : ℝ) (growthRate : ℝ) : ℝ :=
  initial + initial * growthRate + initial * growthRate^2

/-- Represents the streaming statistics for a song -/
structure SongStats where
  spotify : ℝ
  appleMusic : ℝ
  youtube : ℝ

/-- Calculates total listens across all platforms -/
def overallListens (initial : SongStats) (growth : SongStats) : ℝ :=
  totalListens initial.spotify growth.spotify +
  totalListens initial.appleMusic growth.appleMusic +
  totalListens initial.youtube growth.youtube

/-- Jordan's initial listens -/
def jordanInitial : SongStats := ⟨60000, 35000, 45000⟩

/-- Jordan's monthly growth rates -/
def jordanGrowth : SongStats := ⟨2, 1.5, 1.25⟩

/-- Alex's initial listens -/
def alexInitial : SongStats := ⟨75000, 50000, 65000⟩

/-- Alex's monthly growth rates -/
def alexGrowth : SongStats := ⟨1.5, 1.8, 1.1⟩

theorem alex_has_more_listens :
  overallListens alexInitial alexGrowth > overallListens jordanInitial jordanGrowth :=
by sorry

end NUMINAMATH_CALUDE_alex_has_more_listens_l3406_340621


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l3406_340661

/-- Given an arithmetic sequence {a_n} where a_3 + a_11 = 40, prove that a_6 + a_7 + a_8 = 60 -/
theorem arithmetic_sequence_sum (a : ℕ → ℝ) :
  (∀ n : ℕ, a (n + 1) - a n = a (n + 2) - a (n + 1)) →  -- arithmetic sequence condition
  a 3 + a 11 = 40 →                                     -- given condition
  a 6 + a 7 + a 8 = 60 :=                               -- conclusion to prove
by sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l3406_340661


namespace NUMINAMATH_CALUDE_laundry_dishes_multiple_l3406_340622

theorem laundry_dishes_multiple : ∃ m : ℝ, 46 = m * 20 + 6 :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_CALUDE_laundry_dishes_multiple_l3406_340622


namespace NUMINAMATH_CALUDE_sufficient_but_not_necessary_l3406_340651

theorem sufficient_but_not_necessary (a : ℝ) : 
  (∀ a, a > 2 → 1/a < 1/2) ∧ 
  (∃ a, 1/a < 1/2 ∧ ¬(a > 2)) :=
by sorry

end NUMINAMATH_CALUDE_sufficient_but_not_necessary_l3406_340651


namespace NUMINAMATH_CALUDE_f_even_and_decreasing_l3406_340615

def f (x : ℝ) := -x^2 + 1

theorem f_even_and_decreasing :
  (∀ x : ℝ, f (-x) = f x) ∧
  (∀ x y : ℝ, 0 < x → x < y → f y < f x) :=
by sorry

end NUMINAMATH_CALUDE_f_even_and_decreasing_l3406_340615


namespace NUMINAMATH_CALUDE_teachers_count_l3406_340609

/-- Represents the total number of faculty and students in the school -/
def total_population : ℕ := 2400

/-- Represents the total number of individuals in the sample -/
def sample_size : ℕ := 160

/-- Represents the number of students in the sample -/
def students_in_sample : ℕ := 150

/-- Calculates the number of teachers in the school -/
def number_of_teachers : ℕ :=
  total_population - (total_population * students_in_sample) / sample_size

theorem teachers_count : number_of_teachers = 150 := by
  sorry

end NUMINAMATH_CALUDE_teachers_count_l3406_340609


namespace NUMINAMATH_CALUDE_cube_root_of_negative_one_twenty_seventh_l3406_340633

theorem cube_root_of_negative_one_twenty_seventh :
  ((-1 / 3 : ℝ) : ℝ)^3 = -1 / 27 := by sorry

end NUMINAMATH_CALUDE_cube_root_of_negative_one_twenty_seventh_l3406_340633


namespace NUMINAMATH_CALUDE_class_size_proof_l3406_340632

/-- Proves that the number of students in a class is 27 given specific score distributions and averages -/
theorem class_size_proof (n : ℕ) : 
  (5 : ℝ) * 95 + (3 : ℝ) * 0 + ((n : ℝ) - 8) * 45 = (n : ℝ) * 49.25925925925926 → 
  n = 27 := by
  sorry

end NUMINAMATH_CALUDE_class_size_proof_l3406_340632


namespace NUMINAMATH_CALUDE_correct_sum_and_digit_sum_l3406_340612

def num1 : ℕ := 943587
def num2 : ℕ := 329430
def incorrect_sum : ℕ := 1412017

def change_digit (n : ℕ) (d e : ℕ) : ℕ := 
  sorry

theorem correct_sum_and_digit_sum :
  ∃ (d e : ℕ),
    (change_digit num1 d e + change_digit num2 d e ≠ incorrect_sum) ∧
    (change_digit num1 d e + change_digit num2 d e = num1 + change_digit num2 d e) ∧
    (d + e = 7) :=
  sorry

end NUMINAMATH_CALUDE_correct_sum_and_digit_sum_l3406_340612


namespace NUMINAMATH_CALUDE_x_plus_y_value_l3406_340683

theorem x_plus_y_value (x y : ℝ) 
  (h1 : |x| + x + y = 12) 
  (h2 : x + |y| - y = 10) : 
  x + y = 26/5 := by
sorry

end NUMINAMATH_CALUDE_x_plus_y_value_l3406_340683


namespace NUMINAMATH_CALUDE_system_one_solution_system_two_solution_l3406_340669

-- System (1)
theorem system_one_solution (x y : ℝ) : 
  x = 5 - y ∧ x - 3*y = 1 → x = 4 ∧ y = 1 := by sorry

-- System (2)
theorem system_two_solution (x y : ℝ) :
  x - 2*y = 6 ∧ 2*x + 3*y = -2 → x = 2 ∧ y = -2 := by sorry

end NUMINAMATH_CALUDE_system_one_solution_system_two_solution_l3406_340669


namespace NUMINAMATH_CALUDE_kenny_basketball_time_l3406_340611

/-- Represents Kenny's activities and their durations --/
structure KennyActivities where
  basketball : ℝ
  running : ℝ
  trumpet : ℝ
  swimming : ℝ
  studying : ℝ

/-- Theorem stating the duration of Kenny's basketball playing --/
theorem kenny_basketball_time (k : KennyActivities) 
  (h1 : k.running = 2 * k.basketball)
  (h2 : k.trumpet = 2 * k.running)
  (h3 : k.swimming = 2.5 * k.trumpet)
  (h4 : k.studying = 0.5 * k.swimming)
  (h5 : k.trumpet = 40) : 
  k.basketball = 10 := by
  sorry

end NUMINAMATH_CALUDE_kenny_basketball_time_l3406_340611


namespace NUMINAMATH_CALUDE_big_sale_commission_proof_l3406_340685

/-- Calculates the commission amount for a big sale given the following conditions:
  * new_average: Matt's new average commission after the big sale
  * total_sales: Total number of sales including the big sale
  * average_increase: The amount by which the big sale raised the average commission
-/
def big_sale_commission (new_average : ℚ) (total_sales : ℕ) (average_increase : ℚ) : ℚ :=
  new_average * total_sales - (new_average - average_increase) * (total_sales - 1)

/-- Theorem stating that given Matt's new average commission is $250, he has made 6 sales,
    and the big sale commission raises his average by $150, the commission amount for the
    big sale is $1000. -/
theorem big_sale_commission_proof :
  big_sale_commission 250 6 150 = 1000 := by
  sorry

end NUMINAMATH_CALUDE_big_sale_commission_proof_l3406_340685


namespace NUMINAMATH_CALUDE_complex_expression_equals_23_over_150_l3406_340637

theorem complex_expression_equals_23_over_150 : 
  let x := (27/8)^(2/3) - (49/9)^(1/2) + 0.008^(2/3) / 0.02^(1/2) * 0.32^(1/2)
  (x / 0.0625^0.25) = 23/150 := by
  sorry

end NUMINAMATH_CALUDE_complex_expression_equals_23_over_150_l3406_340637


namespace NUMINAMATH_CALUDE_halfway_fraction_reduced_l3406_340636

theorem halfway_fraction_reduced (a b c d e f : ℚ) : 
  a = 3/4 → 
  b = 5/6 → 
  c = (a + b) / 2 → 
  d = 1/12 → 
  e = c - d → 
  f = 17/24 → 
  e = f := by sorry

end NUMINAMATH_CALUDE_halfway_fraction_reduced_l3406_340636


namespace NUMINAMATH_CALUDE_train_braking_problem_l3406_340618

/-- The braking distance function for a train -/
def S (t : ℝ) : ℝ := 27 * t - 0.45 * t^2

/-- The derivative of the braking distance function -/
def S' (t : ℝ) : ℝ := 27 - 0.9 * t

theorem train_braking_problem :
  (∃ t : ℝ, S' t = 0 ∧ t = 30) ∧
  S 30 = 405 := by
  sorry

end NUMINAMATH_CALUDE_train_braking_problem_l3406_340618


namespace NUMINAMATH_CALUDE_coord_sum_of_point_B_l3406_340682

/-- Given points A(0, 0) and B(x, 3) where the slope of AB is 4/5,
    prove that the sum of B's coordinates is 6.75 -/
theorem coord_sum_of_point_B (x : ℝ) : 
  let A : ℝ × ℝ := (0, 0)
  let B : ℝ × ℝ := (x, 3)
  let slope : ℝ := (B.2 - A.2) / (B.1 - A.1)
  slope = 4/5 → x + 3 = 6.75 := by
sorry

end NUMINAMATH_CALUDE_coord_sum_of_point_B_l3406_340682


namespace NUMINAMATH_CALUDE_kelly_time_indeterminate_but_longest_l3406_340664

/-- Represents the breath-holding contest results -/
structure BreathHoldingContest where
  kelly_time : ℝ
  brittany_time : ℝ
  buffy_time : ℝ
  brittany_kelly_diff : kelly_time - brittany_time = 20
  buffy_time_exact : buffy_time = 120

/-- Kelly's time is indeterminate but greater than Buffy's if she won -/
theorem kelly_time_indeterminate_but_longest (contest : BreathHoldingContest) :
  (∀ t : ℝ, contest.kelly_time ≠ t) ∧
  (contest.kelly_time > contest.buffy_time) :=
sorry

end NUMINAMATH_CALUDE_kelly_time_indeterminate_but_longest_l3406_340664


namespace NUMINAMATH_CALUDE_petyas_fruits_l3406_340610

theorem petyas_fruits (total : ℕ) (apples tangerines oranges : ℕ) : 
  total = 20 →
  apples + tangerines + oranges = total →
  tangerines * 6 = apples →
  apples > oranges →
  oranges = 6 :=
by sorry

end NUMINAMATH_CALUDE_petyas_fruits_l3406_340610


namespace NUMINAMATH_CALUDE_quadratic_equation_real_roots_quadratic_equation_real_roots_for_m_1_l3406_340653

theorem quadratic_equation_real_roots (m : ℝ) : 
  (∃ x : ℝ, (x + 2)^2 = m + 2) ↔ m ≥ -2 :=
by sorry

-- Example for m = 1
theorem quadratic_equation_real_roots_for_m_1 : 
  ∃ x : ℝ, (x + 2)^2 = 1 + 2 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equation_real_roots_quadratic_equation_real_roots_for_m_1_l3406_340653


namespace NUMINAMATH_CALUDE_infinite_geometric_series_l3406_340666

/-- Given an infinite geometric series with first term a and sum S,
    prove the common ratio r and the second term -/
theorem infinite_geometric_series
  (a : ℝ) (S : ℝ) (h_a : a = 540) (h_S : S = 4500) :
  ∃ (r : ℝ),
    r = 0.88 ∧
    S = a / (1 - r) ∧
    abs r < 1 ∧
    a * r = 475.2 := by
  sorry

end NUMINAMATH_CALUDE_infinite_geometric_series_l3406_340666


namespace NUMINAMATH_CALUDE_no_solution_exists_l3406_340690

/-- Given a third-degree polynomial P(x) = x^3 - mx^2 + nx + 42,
    where (x + 6) and (x - a + bi) are factors,
    with a and b being real numbers and b ≠ 0,
    prove that there are no real values for m, n, a, and b
    that satisfy all conditions simultaneously. -/
theorem no_solution_exists (m n a b : ℝ) : b ≠ 0 →
  (∀ x, x^3 - m*x^2 + n*x + 42 = (x + 6) * (x - a + b*I) * (x - a - b*I)) →
  False := by sorry

end NUMINAMATH_CALUDE_no_solution_exists_l3406_340690


namespace NUMINAMATH_CALUDE_product_of_integers_l3406_340687

theorem product_of_integers (x y : ℤ) (h1 : x + y = 8) (h2 : x^2 + y^2 = 34) : x * y = 15 := by
  sorry

end NUMINAMATH_CALUDE_product_of_integers_l3406_340687


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_l3406_340676

theorem sufficient_not_necessary :
  (∀ a b c d : ℝ, a > b ∧ c > d → a + c > b + d) ∧
  (∃ a b c d : ℝ, a + c > b + d ∧ ¬(a > b ∧ c > d)) :=
by sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_l3406_340676


namespace NUMINAMATH_CALUDE_peppers_required_per_day_l3406_340667

/-- Represents the number of jalapeno pepper strips per sandwich -/
def strips_per_sandwich : ℕ := 4

/-- Represents the number of slices one jalapeno pepper can make -/
def slices_per_pepper : ℕ := 8

/-- Represents the time in minutes between serving each sandwich -/
def minutes_per_sandwich : ℕ := 5

/-- Represents the number of hours in a workday -/
def hours_per_day : ℕ := 8

/-- Represents the number of minutes in an hour -/
def minutes_per_hour : ℕ := 60

/-- Theorem stating the number of jalapeno peppers required for an 8-hour day -/
theorem peppers_required_per_day : 
  (hours_per_day * minutes_per_hour / minutes_per_sandwich) * 
  (strips_per_sandwich : ℚ) / slices_per_pepper = 48 := by
  sorry


end NUMINAMATH_CALUDE_peppers_required_per_day_l3406_340667


namespace NUMINAMATH_CALUDE_min_modulus_on_circle_l3406_340642

theorem min_modulus_on_circle (z : ℂ) (h : Complex.abs (z - (1 + Complex.I)) = 1) :
  ∃ (w : ℂ), Complex.abs w = Real.sqrt 2 - 1 ∧ 
  ∀ (v : ℂ), Complex.abs (v - (1 + Complex.I)) = 1 → Complex.abs v ≥ Complex.abs w :=
by sorry

end NUMINAMATH_CALUDE_min_modulus_on_circle_l3406_340642


namespace NUMINAMATH_CALUDE_min_sum_of_intercepts_equality_condition_l3406_340694

theorem min_sum_of_intercepts (a b : ℝ) : 
  a > 0 → b > 0 → (4 / a + 1 / b = 1) → a + b ≥ 9 := by
  sorry

theorem equality_condition (a b : ℝ) :
  a > 0 → b > 0 → (4 / a + 1 / b = 1) → (a + b = 9) → (a = 6 ∧ b = 3) := by
  sorry

end NUMINAMATH_CALUDE_min_sum_of_intercepts_equality_condition_l3406_340694


namespace NUMINAMATH_CALUDE_susan_age_l3406_340662

theorem susan_age (susan arthur tom bob : ℕ) 
  (h1 : arthur = susan + 2)
  (h2 : tom = bob - 3)
  (h3 : bob = 11)
  (h4 : susan + arthur + tom + bob = 51) :
  susan = 15 := by
sorry

end NUMINAMATH_CALUDE_susan_age_l3406_340662


namespace NUMINAMATH_CALUDE_expression_equals_two_l3406_340674

theorem expression_equals_two (a b c : ℝ) 
  (h_pos_a : a > 0) (h_pos_b : b > 0) (h_pos_c : c > 0) 
  (h_abc : a * b * c = 1) : 
  (1 + a) / (1 + a + a * b) + 
  (1 + b) / (1 + b + b * c) + 
  (1 + c) / (1 + c + c * a) = 2 := by
sorry

end NUMINAMATH_CALUDE_expression_equals_two_l3406_340674


namespace NUMINAMATH_CALUDE_six_people_eight_chairs_two_restricted_l3406_340639

/-- The number of ways to arrange n people in r chairs -/
def arrangements (n r : ℕ) : ℕ := n.factorial

/-- The number of ways to choose r chairs from n chairs -/
def chair_selections (n r : ℕ) : ℕ := n.choose r

/-- The number of ways to seat people in chairs with restrictions -/
def seating_arrangements (total_chairs people : ℕ) (restricted_pairs : ℕ) : ℕ :=
  (chair_selections total_chairs people - restricted_pairs) * arrangements people people

theorem six_people_eight_chairs_two_restricted : 
  seating_arrangements 8 6 30 = 18720 := by
  sorry

end NUMINAMATH_CALUDE_six_people_eight_chairs_two_restricted_l3406_340639


namespace NUMINAMATH_CALUDE_daily_profit_properties_l3406_340613

/-- Represents the daily sales profit function for a company -/
def daily_profit (x : ℝ) : ℝ := 10 * x^2 - 80 * x

/-- Theorem stating the properties of the daily sales profit function -/
theorem daily_profit_properties :
  -- The daily profit function is correct
  (∀ x, daily_profit x = 10 * x^2 - 80 * x) ∧
  -- When the selling price increases by 3 yuan, the daily profit is 350 yuan
  (daily_profit 3 = 350) ∧
  -- When the daily profit is 360 yuan, the selling price has increased by 4 yuan
  (daily_profit 4 = 360) := by
  sorry


end NUMINAMATH_CALUDE_daily_profit_properties_l3406_340613


namespace NUMINAMATH_CALUDE_glasses_cost_glasses_cost_proof_l3406_340608

/-- Calculate the total cost of glasses after discounts -/
theorem glasses_cost (frame_cost lens_cost : ℝ) 
  (insurance_coverage : ℝ) (frame_coupon : ℝ) : ℝ :=
  let discounted_lens_cost := lens_cost * (1 - insurance_coverage)
  let discounted_frame_cost := frame_cost - frame_coupon
  discounted_lens_cost + discounted_frame_cost

/-- Prove that the total cost of glasses after discounts is $250 -/
theorem glasses_cost_proof :
  glasses_cost 200 500 0.8 50 = 250 := by
  sorry

end NUMINAMATH_CALUDE_glasses_cost_glasses_cost_proof_l3406_340608


namespace NUMINAMATH_CALUDE_student_probability_problem_l3406_340677

theorem student_probability_problem (p q : ℝ) 
  (h_p_pos : 0 < p) (h_q_pos : 0 < q) (h_p_le_one : p ≤ 1) (h_q_le_one : q ≤ 1)
  (h_p_gt_q : p > q)
  (h_at_least_one : 1 - (1 - p) * (1 - q) = 5/6)
  (h_both_correct : p * q = 1/3)
  : p = 2/3 ∧ q = 1/2 ∧ 
    (1 - p)^2 * 2 * (1 - q) * q + (1 - p)^2 * q^2 + 2 * (1 - p) * p * q^2 = 7/36 := by
  sorry

end NUMINAMATH_CALUDE_student_probability_problem_l3406_340677


namespace NUMINAMATH_CALUDE_club_membership_theorem_l3406_340681

/-- Represents the number of students in various club combinations -/
structure ClubMembership where
  total : ℕ
  music : ℕ
  science : ℕ
  sports : ℕ
  none : ℕ
  onlyMusic : ℕ
  onlyScience : ℕ
  onlySports : ℕ
  musicScience : ℕ
  scienceSports : ℕ
  musicSports : ℕ
  allThree : ℕ

/-- Theorem stating that given the conditions, the number of students in all three clubs is 1 -/
theorem club_membership_theorem (c : ClubMembership) : 
  c.total = 40 ∧ 
  c.music = c.total / 4 ∧ 
  c.science = c.total / 5 ∧ 
  c.sports = 8 ∧ 
  c.none = 7 ∧ 
  c.onlyMusic = 6 ∧ 
  c.onlyScience = 5 ∧ 
  c.onlySports = 2 ∧ 
  c.music = c.onlyMusic + c.musicScience + c.musicSports + c.allThree ∧ 
  c.science = c.onlyScience + c.musicScience + c.scienceSports + c.allThree ∧ 
  c.sports = c.onlySports + c.scienceSports + c.musicSports + c.allThree ∧ 
  c.total = c.none + c.onlyMusic + c.onlyScience + c.onlySports + c.musicScience + c.scienceSports + c.musicSports + c.allThree →
  c.allThree = 1 := by
sorry


end NUMINAMATH_CALUDE_club_membership_theorem_l3406_340681


namespace NUMINAMATH_CALUDE_sum_of_x_and_y_l3406_340648

theorem sum_of_x_and_y (x y : ℝ) (some_number : ℝ) 
  (h1 : x + y = some_number) 
  (h2 : x - y = 5) 
  (h3 : x = 10) 
  (h4 : y = 5) : 
  x + y = 15 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_x_and_y_l3406_340648


namespace NUMINAMATH_CALUDE_square_sum_plus_sum_squares_l3406_340696

theorem square_sum_plus_sum_squares : (5 + 9)^2 + (5^2 + 9^2) = 302 := by sorry

end NUMINAMATH_CALUDE_square_sum_plus_sum_squares_l3406_340696


namespace NUMINAMATH_CALUDE_sector_central_angle_l3406_340630

/-- Given a sector with area 1 cm² and perimeter 4 cm, its central angle is 2 radians. -/
theorem sector_central_angle (r : ℝ) (θ : ℝ) 
  (h_area : (1/2) * θ * r^2 = 1)
  (h_perimeter : 2*r + θ*r = 4) :
  θ = 2 := by
  sorry

end NUMINAMATH_CALUDE_sector_central_angle_l3406_340630


namespace NUMINAMATH_CALUDE_cost_of_fencing_square_l3406_340623

/-- The cost of fencing a square -/
theorem cost_of_fencing_square (cost_per_side : ℕ) (h : cost_per_side = 79) : 
  4 * cost_per_side = 316 := by
  sorry

end NUMINAMATH_CALUDE_cost_of_fencing_square_l3406_340623


namespace NUMINAMATH_CALUDE_book_combinations_l3406_340631

theorem book_combinations (n m : ℕ) (h1 : n = 15) (h2 : m = 3) : Nat.choose n m = 455 := by
  sorry

end NUMINAMATH_CALUDE_book_combinations_l3406_340631


namespace NUMINAMATH_CALUDE_triangle_PPB_area_l3406_340686

/-- A square with side length 10 inches -/
def square_side : ℝ := 10

/-- Point P is a vertex of the square -/
def P : ℝ × ℝ := (0, 0)

/-- Point B is on the side of the square -/
def B : ℝ × ℝ := (square_side, 0)

/-- Point Q is inside the square and 8 inches above P -/
def Q : ℝ × ℝ := (0, 8)

/-- PQ is perpendicular to PB -/
axiom PQ_perp_PB : (Q.1 - P.1) * (B.1 - P.1) + (Q.2 - P.2) * (B.2 - P.2) = 0

/-- The area of triangle PPB -/
def triangle_area : ℝ := 0.5 * square_side * 8

/-- Theorem: The area of triangle PPB is 40 square inches -/
theorem triangle_PPB_area : triangle_area = 40 := by sorry

end NUMINAMATH_CALUDE_triangle_PPB_area_l3406_340686


namespace NUMINAMATH_CALUDE_binomial_expansion_coefficient_l3406_340644

theorem binomial_expansion_coefficient (x : ℝ) :
  (1 + 2*x)^3 = 1 + 6*x + 12*x^2 + 8*x^3 :=
by sorry

end NUMINAMATH_CALUDE_binomial_expansion_coefficient_l3406_340644


namespace NUMINAMATH_CALUDE_square_perimeter_problem_l3406_340692

theorem square_perimeter_problem (A B C : ℝ) : 
  (A > 0) → (B > 0) → (C > 0) →
  (4 * A = 16) → (4 * B = 32) → (C = A + B - 2) →
  (4 * C = 40) := by
sorry

end NUMINAMATH_CALUDE_square_perimeter_problem_l3406_340692


namespace NUMINAMATH_CALUDE_sum_and_reciprocal_squared_l3406_340607

theorem sum_and_reciprocal_squared (x N : ℝ) (h1 : x ≠ 0) (h2 : x + 1/x = N) (h3 : x^2 + 1/x^2 = 2) : N = 2 := by
  sorry

end NUMINAMATH_CALUDE_sum_and_reciprocal_squared_l3406_340607


namespace NUMINAMATH_CALUDE_stanley_lemonade_sales_l3406_340614

/-- The number of cups of lemonade Carl sells per hour -/
def carl_cups_per_hour : ℕ := 7

/-- The number of hours considered -/
def hours : ℕ := 3

/-- The difference in cups sold between Carl and Stanley over 3 hours -/
def difference_in_cups : ℕ := 9

/-- The number of cups of lemonade Stanley sells per hour -/
def stanley_cups_per_hour : ℕ := 4

theorem stanley_lemonade_sales :
  stanley_cups_per_hour * hours + difference_in_cups = carl_cups_per_hour * hours := by
  sorry

end NUMINAMATH_CALUDE_stanley_lemonade_sales_l3406_340614


namespace NUMINAMATH_CALUDE_handshake_count_l3406_340604

/-- The number of handshakes in a conference of 25 people -/
def conference_handshakes : ℕ := 300

/-- The number of attendees at the conference -/
def num_attendees : ℕ := 25

theorem handshake_count :
  (num_attendees.choose 2 : ℕ) = conference_handshakes :=
sorry

end NUMINAMATH_CALUDE_handshake_count_l3406_340604


namespace NUMINAMATH_CALUDE_square_difference_identity_l3406_340645

theorem square_difference_identity (a b : ℝ) : a^2 - b^2 = (a + b) * (a - b) := by
  sorry

end NUMINAMATH_CALUDE_square_difference_identity_l3406_340645


namespace NUMINAMATH_CALUDE_divisibility_implies_equality_l3406_340680

theorem divisibility_implies_equality (a b : ℕ) :
  (∃ S : Set ℕ, Set.Infinite S ∧ ∀ n ∈ S, (a^(n+1) + b^(n+1)) % (a^n + b^n) = 0) →
  a = b :=
sorry

end NUMINAMATH_CALUDE_divisibility_implies_equality_l3406_340680


namespace NUMINAMATH_CALUDE_similar_triangles_height_l3406_340659

theorem similar_triangles_height (h_small : ℝ) (area_ratio : ℝ) :
  h_small > 0 →
  area_ratio = 25 →
  ∃ h_large : ℝ,
    h_large = h_small * Real.sqrt area_ratio ∧
    h_large = 20 :=
by
  sorry

end NUMINAMATH_CALUDE_similar_triangles_height_l3406_340659


namespace NUMINAMATH_CALUDE_inverse_variation_l3406_340695

/-- Given that r and s vary inversely, and s = 0.35 when r = 1200, 
    prove that s = 0.175 when r = 2400 -/
theorem inverse_variation (r s : ℝ) (h : r * s = 1200 * 0.35) :
  r = 2400 → s = 0.175 := by
  sorry

end NUMINAMATH_CALUDE_inverse_variation_l3406_340695


namespace NUMINAMATH_CALUDE_main_theorem_l3406_340663

/-- A function satisfying the given functional equation -/
def FunctionalEquation (f : ℝ → ℝ) : Prop :=
  ∀ x y, f (x * y) = y * f x + x * f y

/-- The main theorem capturing the problem statements -/
theorem main_theorem (f : ℝ → ℝ) (hf : FunctionalEquation f)
    (a b c d : ℝ) (F : ℝ → ℝ) (hF : ∀ x, F x = a * f x + b * x^5 + c * x^3 + 2 * x^2 + d * x + 3)
    (hF_neg5 : F (-5) = 7) :
    f 0 = 0 ∧ (∀ x, f (-x) = -f x) ∧ F 5 = 99 := by
  sorry


end NUMINAMATH_CALUDE_main_theorem_l3406_340663


namespace NUMINAMATH_CALUDE_kishore_savings_l3406_340679

def total_expenses : ℕ := 16200
def savings_rate : ℚ := 1 / 10

theorem kishore_savings :
  ∀ (salary : ℕ),
  (salary : ℚ) = (total_expenses : ℚ) + savings_rate * (salary : ℚ) →
  savings_rate * (salary : ℚ) = 1800 := by
  sorry

end NUMINAMATH_CALUDE_kishore_savings_l3406_340679


namespace NUMINAMATH_CALUDE_problem_solution_l3406_340668

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x < 1 then 2^x + 1 else x^2 + a*x

theorem problem_solution (a : ℝ) : f a (f a 0) = 4*a → a = 2 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l3406_340668


namespace NUMINAMATH_CALUDE_book_pricing_problem_l3406_340603

/-- Proves that the cost price is approximately 64% of the marked price
    given the conditions of the book pricing problem. -/
theorem book_pricing_problem (MP CP : ℝ) : 
  MP > 0 → -- Marked price is positive
  CP > 0 → -- Cost price is positive
  MP * 0.88 = 1.375 * CP → -- Condition after applying discount and gain
  ∃ ε > 0, |CP / MP - 0.64| < ε := by
sorry


end NUMINAMATH_CALUDE_book_pricing_problem_l3406_340603


namespace NUMINAMATH_CALUDE_sum_of_coefficients_equals_one_l3406_340672

theorem sum_of_coefficients_equals_one (a : ℝ) (a₁ a₂ a₃ a₄ a₅ a₆ a₇ a₈ a₉ a₁₀ : ℝ) :
  (∀ x : ℝ, (2*x - 3)^10 = a + a₁*(x-1) + a₂*(x-1)^2 + a₃*(x-1)^3 + a₄*(x-1)^4 + 
                           a₅*(x-1)^5 + a₆*(x-1)^6 + a₇*(x-1)^7 + a₈*(x-1)^8 + 
                           a₉*(x-1)^9 + a₁₀*(x-1)^10) →
  a + a₁ + a₂ + a₃ + a₄ + a₅ + a₆ + a₇ + a₈ + a₉ + a₁₀ = 1 :=
by
  sorry

end NUMINAMATH_CALUDE_sum_of_coefficients_equals_one_l3406_340672


namespace NUMINAMATH_CALUDE_function_equation_implies_odd_l3406_340691

/-- A non-zero function satisfying the given functional equation is odd -/
theorem function_equation_implies_odd (f : ℝ → ℝ) 
  (h_nonzero : ∃ x, f x ≠ 0)
  (h_eq : ∀ a b : ℝ, f (a * b) = a * f b + b * f a) :
  ∀ x : ℝ, f (-x) = -f x := by
  sorry

end NUMINAMATH_CALUDE_function_equation_implies_odd_l3406_340691


namespace NUMINAMATH_CALUDE_probability_specific_marble_draw_l3406_340688

/-- Represents the number of marbles of each color in the jar -/
structure MarbleCount where
  red : ℕ
  green : ℕ
  white : ℕ

/-- Calculates the probability of drawing two red marbles followed by one green marble -/
def probability_two_red_one_green (mc : MarbleCount) : ℚ :=
  let total := mc.red + mc.green + mc.white
  (mc.red : ℚ) / total *
  ((mc.red - 1) : ℚ) / (total - 1) *
  (mc.green : ℚ) / (total - 2)

/-- The main theorem stating the probability for the given marble counts -/
theorem probability_specific_marble_draw :
  probability_two_red_one_green ⟨3, 4, 12⟩ = 12 / 2907 := by
  sorry

#eval probability_two_red_one_green ⟨3, 4, 12⟩

end NUMINAMATH_CALUDE_probability_specific_marble_draw_l3406_340688


namespace NUMINAMATH_CALUDE_max_sum_reciprocal_zeros_l3406_340678

noncomputable def f (k : ℝ) (x : ℝ) : ℝ :=
  if 0 < x ∧ x ≤ 1 then k * x^2 + 2 * x - 1
  else if x > 1 then k * x + 1
  else 0

theorem max_sum_reciprocal_zeros (k : ℝ) :
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ f k x₁ = 0 ∧ f k x₂ = 0) →
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ f k x₁ = 0 ∧ f k x₂ = 0 ∧ 1/x₁ + 1/x₂ ≤ 9/4) ∧
  (∃ k₀ : ℝ, ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ f k₀ x₁ = 0 ∧ f k₀ x₂ = 0 ∧ 1/x₁ + 1/x₂ = 9/4) :=
by sorry

end NUMINAMATH_CALUDE_max_sum_reciprocal_zeros_l3406_340678


namespace NUMINAMATH_CALUDE_larger_number_proof_l3406_340658

theorem larger_number_proof (a b : ℕ+) 
  (hcf_eq : Nat.gcd a b = 28)
  (lcm_eq : Nat.lcm a b = 28 * 12 * 15) :
  max a b = 180 := by
  sorry

end NUMINAMATH_CALUDE_larger_number_proof_l3406_340658


namespace NUMINAMATH_CALUDE_complex_fraction_simplification_l3406_340656

theorem complex_fraction_simplification :
  let z₁ : ℂ := Complex.mk 3 5
  let z₂ : ℂ := Complex.mk (-2) 3
  z₁ / z₂ = Complex.mk (-21/13) (-19/13) := by
  sorry

end NUMINAMATH_CALUDE_complex_fraction_simplification_l3406_340656


namespace NUMINAMATH_CALUDE_zoo_trip_buses_l3406_340665

/-- Given a school trip to the zoo with the following conditions:
  * There are 396 total students
  * 4 students traveled in cars
  * Each bus can hold 56 students
  * All buses were filled
  Prove that the number of buses required is 7. -/
theorem zoo_trip_buses (total_students : ℕ) (car_students : ℕ) (students_per_bus : ℕ) :
  total_students = 396 →
  car_students = 4 →
  students_per_bus = 56 →
  (total_students - car_students) % students_per_bus = 0 →
  (total_students - car_students) / students_per_bus = 7 :=
by sorry

end NUMINAMATH_CALUDE_zoo_trip_buses_l3406_340665


namespace NUMINAMATH_CALUDE_curve_intersection_tangent_l3406_340620

/-- The value of a for which the curves y = a√x and y = ln√x have a common point
    with the same tangent line. -/
theorem curve_intersection_tangent (a : ℝ) (h : a > 0) :
  (∃ x : ℝ, x > 0 ∧ a * Real.sqrt x = Real.log (Real.sqrt x) ∧
    a / (2 * Real.sqrt x) = 1 / (2 * x)) →
  a = Real.exp (-1) := by
sorry

end NUMINAMATH_CALUDE_curve_intersection_tangent_l3406_340620


namespace NUMINAMATH_CALUDE_arithmetic_sequence_property_l3406_340619

def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_property 
  (a : ℕ → ℝ) 
  (h_arithmetic : is_arithmetic_sequence a) 
  (h_sum : a 3 + a 4 + a 5 + a 6 + a 7 = 45) : 
  a 5 = 9 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_property_l3406_340619


namespace NUMINAMATH_CALUDE_value_of_expression_constant_difference_implies_b_value_l3406_340605

/-- Definition of A in terms of a and b -/
def A (a b : ℝ) : ℝ := 2*a^2 + 3*a*b - 2*a - 1

/-- Definition of B in terms of a and b -/
def B (a b : ℝ) : ℝ := a^2 + a*b - 1

/-- Theorem 1: The value of 4A - (3A - 2B) -/
theorem value_of_expression (a b : ℝ) :
  4 * A a b - (3 * A a b - 2 * B a b) = 4*a^2 + 5*a*b - 2*a - 3 := by sorry

/-- Theorem 2: When A - 2B is constant for all a, b must equal 2 -/
theorem constant_difference_implies_b_value (b : ℝ) :
  (∀ a : ℝ, ∃ k : ℝ, A a b - 2 * B a b = k) → b = 2 := by sorry

end NUMINAMATH_CALUDE_value_of_expression_constant_difference_implies_b_value_l3406_340605


namespace NUMINAMATH_CALUDE_sale_price_ratio_l3406_340699

theorem sale_price_ratio (c x y : ℝ) (hx : x = 0.8 * c) (hy : y = 1.25 * c) :
  y / x = 25 / 16 := by
  sorry

end NUMINAMATH_CALUDE_sale_price_ratio_l3406_340699


namespace NUMINAMATH_CALUDE_heidi_has_five_more_than_kim_l3406_340671

/-- The number of nail polishes each person has -/
structure NailPolishes where
  kim : ℕ
  heidi : ℕ
  karen : ℕ

/-- The conditions of the nail polish problem -/
def nail_polish_problem (np : NailPolishes) : Prop :=
  np.kim = 12 ∧
  np.heidi > np.kim ∧
  np.karen = np.kim - 4 ∧
  np.karen + np.heidi = 25

/-- The theorem stating that Heidi has 5 more nail polishes than Kim -/
theorem heidi_has_five_more_than_kim (np : NailPolishes) 
  (h : nail_polish_problem np) : np.heidi - np.kim = 5 := by
  sorry

end NUMINAMATH_CALUDE_heidi_has_five_more_than_kim_l3406_340671


namespace NUMINAMATH_CALUDE_brianna_marbles_l3406_340643

theorem brianna_marbles (x : ℕ) : 
  x - 4 - (2 * 4) - (4 / 2) = 10 → x = 24 := by
  sorry

end NUMINAMATH_CALUDE_brianna_marbles_l3406_340643


namespace NUMINAMATH_CALUDE_tangent_line_slope_intersecting_line_equation_l3406_340625

-- Define the circle C
def circle_C (x y : ℝ) : Prop := x^2 - 4*x + y^2 + 3 = 0

-- Define points P and Q
def P : ℝ × ℝ := (0, 1)
def Q : ℝ × ℝ := (0, -2)

-- Define the condition for the slopes of OA and OB
def slope_condition (k₁ k₂ : ℝ) : Prop := k₁ * k₂ = -1/7

-- Statement for part (1)
theorem tangent_line_slope :
  ∃ m : ℝ, (m = 0 ∨ m = -4/3) ∧
  ∀ x y : ℝ, y = m * x + P.2 →
  (∃! t : ℝ, circle_C t (m * t + P.2)) :=
sorry

-- Statement for part (2)
theorem intersecting_line_equation :
  ∃ k : ℝ, (k = 1 ∨ k = 5/3) ∧
  ∀ x y : ℝ, y = k * x + Q.2 →
  (∃ A B : ℝ × ℝ, 
    circle_C A.1 A.2 ∧ 
    circle_C B.1 B.2 ∧
    A.2 = k * A.1 + Q.2 ∧
    B.2 = k * B.1 + Q.2 ∧
    slope_condition (A.2 / A.1) (B.2 / B.1)) :=
sorry

end NUMINAMATH_CALUDE_tangent_line_slope_intersecting_line_equation_l3406_340625


namespace NUMINAMATH_CALUDE_square_and_sqrt_properties_l3406_340641

theorem square_and_sqrt_properties : 
  let a : ℕ := 10001
  let b : ℕ := 100010001
  let c : ℕ := 1000200030004000300020001
  (a^2 = 100020001) ∧ 
  (b^2 = 10002000300020001) ∧ 
  (c.sqrt = 1000100010001) := by
  sorry

end NUMINAMATH_CALUDE_square_and_sqrt_properties_l3406_340641


namespace NUMINAMATH_CALUDE_quadratic_polynomial_negative_root_l3406_340654

-- Define a quadratic polynomial type
def QuadraticPolynomial (α : Type*) [Ring α] := α → α

-- Define the property of having two distinct real roots
def HasTwoDistinctRealRoots (P : QuadraticPolynomial ℝ) : Prop :=
  ∃ (r₁ r₂ : ℝ), r₁ ≠ r₂ ∧ P r₁ = 0 ∧ P r₂ = 0

-- Define the inequality condition
def SatisfiesInequality (P : QuadraticPolynomial ℝ) : Prop :=
  ∀ (a b : ℝ), (abs a ≥ 2017 ∧ abs b ≥ 2017) → P (a^2 + b^2) ≥ P (2*a*b)

-- Define the property of having at least one negative root
def HasNegativeRoot (P : QuadraticPolynomial ℝ) : Prop :=
  ∃ (r : ℝ), r < 0 ∧ P r = 0

-- The main theorem
theorem quadratic_polynomial_negative_root 
  (P : QuadraticPolynomial ℝ) 
  (h1 : HasTwoDistinctRealRoots P) 
  (h2 : SatisfiesInequality P) : 
  HasNegativeRoot P :=
sorry

end NUMINAMATH_CALUDE_quadratic_polynomial_negative_root_l3406_340654


namespace NUMINAMATH_CALUDE_system_of_equations_sum_l3406_340600

theorem system_of_equations_sum (a b c x y z : ℝ) 
  (eq1 : 17 * x + b * y + c * z = 0)
  (eq2 : a * x + 29 * y + c * z = 0)
  (eq3 : a * x + b * y + 53 * z = 0)
  (ha : a ≠ 17)
  (hx : x ≠ 0) :
  a / (a - 17) + b / (b - 29) + c / (c - 53) = 1 := by
  sorry

end NUMINAMATH_CALUDE_system_of_equations_sum_l3406_340600


namespace NUMINAMATH_CALUDE_sort_table_in_99_moves_l3406_340635

/-- Represents a 10x10 table of integers -/
def Table := Fin 10 → Fin 10 → ℤ

/-- Checks if a table is sorted in ascending order both row-wise and column-wise -/
def is_sorted (t : Table) : Prop :=
  ∀ i j k, i < j → t i k < t j k ∧
  ∀ i j k, i < j → t k i < t k j

/-- Represents a rectangular rotation operation on the table -/
def rotate (t : Table) (i j k l : Fin 10) : Table :=
  sorry

/-- The main theorem stating that any table can be sorted in 99 or fewer moves -/
theorem sort_table_in_99_moves (t : Table) :
  (∀ i j k l, t i j ≠ t k l) →  -- All numbers are distinct
  ∃ (moves : List (Fin 10 × Fin 10 × Fin 10 × Fin 10)),
    moves.length ≤ 99 ∧
    is_sorted (moves.foldl (λ acc m => rotate acc m.1 m.2.1 m.2.2.1 m.2.2.2) t) :=
sorry

end NUMINAMATH_CALUDE_sort_table_in_99_moves_l3406_340635


namespace NUMINAMATH_CALUDE_nikolai_faster_l3406_340684

/-- Represents a mountain goat with a specific jump distance -/
structure Goat where
  name : String
  jump_distance : ℕ

/-- Calculates the number of jumps needed to cover a given distance -/
def jumps_needed (g : Goat) (distance : ℕ) : ℕ :=
  (distance + g.jump_distance - 1) / g.jump_distance

theorem nikolai_faster (nikolai gennady : Goat)
  (h1 : nikolai.jump_distance = 4)
  (h2 : gennady.jump_distance = 6)
  (h3 : jumps_needed nikolai 2000 * nikolai.jump_distance = 2000)
  (h4 : jumps_needed gennady 2000 * gennady.jump_distance = 2004) :
  jumps_needed nikolai 2000 < jumps_needed gennady 2000 := by
  sorry

#eval jumps_needed (Goat.mk "Nikolai" 4) 2000
#eval jumps_needed (Goat.mk "Gennady" 6) 2000

end NUMINAMATH_CALUDE_nikolai_faster_l3406_340684


namespace NUMINAMATH_CALUDE_function_inequality_l3406_340647

open Real

theorem function_inequality (f : ℝ → ℝ) (f' : ℝ → ℝ) (h : ∀ x > 0, x * (f' x) + x^2 < f x) :
  2 * f 1 > f 2 + 2 ∧ 3 * f 1 > f 3 + 3 := by
  sorry

end NUMINAMATH_CALUDE_function_inequality_l3406_340647


namespace NUMINAMATH_CALUDE_middle_circle_number_l3406_340629

def numbers : List ℕ := [1, 5, 6, 7, 13, 14, 17, 22, 26]

def middle_fixed : List ℕ := [13, 17]

def total_sum : ℕ := numbers.sum

def group_sum : ℕ := total_sum / 3

theorem middle_circle_number (x : ℕ) 
  (h1 : x ∈ numbers)
  (h2 : ∀ (a b c : ℕ), a ∈ numbers → b ∈ numbers → c ∈ numbers → 
       a ≠ b → b ≠ c → a ≠ c → 
       (a + b + c = group_sum) → 
       (a = 13 ∧ b = 17) ∨ (a = 13 ∧ c = 17) ∨ (b = 13 ∧ c = 17) → 
       x = c ∨ x = b) :
  x = 7 := by sorry

end NUMINAMATH_CALUDE_middle_circle_number_l3406_340629


namespace NUMINAMATH_CALUDE_unique_solution_for_equation_l3406_340638

theorem unique_solution_for_equation (n : ℕ+) (p : ℕ) : 
  Nat.Prime p → (n : ℕ)^8 - (n : ℕ)^2 = p^5 + p^2 → (n = 2 ∧ p = 3) :=
by sorry

end NUMINAMATH_CALUDE_unique_solution_for_equation_l3406_340638


namespace NUMINAMATH_CALUDE_minimum_value_quadratic_l3406_340697

theorem minimum_value_quadratic (x : ℝ) :
  (4 * x^2 + 8 * x + 3 = 5) → x ≥ -1 - Real.sqrt 6 / 2 :=
by sorry

end NUMINAMATH_CALUDE_minimum_value_quadratic_l3406_340697


namespace NUMINAMATH_CALUDE_expression_evaluation_l3406_340655

theorem expression_evaluation (a b c d : ℚ) : 
  a = 3 → 
  b = a + 3 → 
  c = b - 8 → 
  d = a + 5 → 
  a + 2 ≠ 0 → 
  b - 4 ≠ 0 → 
  c + 5 ≠ 0 → 
  d - 3 ≠ 0 → 
  (a + 3) / (a + 2) * (b - 2) / (b - 4) * (c + 9) / (c + 5) * (d + 1) / (d - 3) = 1512 / 75 := by
sorry

end NUMINAMATH_CALUDE_expression_evaluation_l3406_340655


namespace NUMINAMATH_CALUDE_sum_perpendiculars_equals_altitude_l3406_340602

/-- Represents a triangle in 2D space -/
structure Triangle :=
  (A B C : ℝ × ℝ)

/-- Represents a point in 2D space -/
def Point := ℝ × ℝ

/-- Checks if a triangle is isosceles with AB = AC -/
def Triangle.isIsosceles (t : Triangle) : Prop :=
  dist t.A t.B = dist t.A t.C

/-- Calculates the altitude of a triangle -/
noncomputable def Triangle.altitude (t : Triangle) : ℝ := sorry

/-- Calculates the perpendicular distance from a point to a line segment -/
noncomputable def perpendicularDistance (p : Point) (a b : Point) : ℝ := sorry

/-- Checks if a point is inside or on a triangle -/
def Triangle.containsPoint (t : Triangle) (p : Point) : Prop := sorry

/-- Theorem: Sum of perpendiculars equals altitude for isosceles triangle -/
theorem sum_perpendiculars_equals_altitude (t : Triangle) (p : Point) :
  t.isIsosceles →
  t.containsPoint p →
  perpendicularDistance p t.B t.C + 
  perpendicularDistance p t.C t.A + 
  perpendicularDistance p t.A t.B = 
  t.altitude := by sorry

end NUMINAMATH_CALUDE_sum_perpendiculars_equals_altitude_l3406_340602


namespace NUMINAMATH_CALUDE_weight_of_BaF2_l3406_340616

/-- The atomic weight of barium in g/mol -/
def atomic_weight_Ba : ℝ := 137.33

/-- The atomic weight of fluorine in g/mol -/
def atomic_weight_F : ℝ := 19.00

/-- The number of moles of BaF2 -/
def moles_BaF2 : ℝ := 6

/-- The molecular weight of BaF2 in g/mol -/
def molecular_weight_BaF2 : ℝ := atomic_weight_Ba + 2 * atomic_weight_F

/-- The weight of BaF2 in grams -/
def weight_BaF2 : ℝ := moles_BaF2 * molecular_weight_BaF2

theorem weight_of_BaF2 : weight_BaF2 = 1051.98 := by
  sorry

end NUMINAMATH_CALUDE_weight_of_BaF2_l3406_340616


namespace NUMINAMATH_CALUDE_smallest_integer_in_set_l3406_340624

theorem smallest_integer_in_set (n : ℤ) : 
  (n + 4 < 3 * ((n + (n + 1) + (n + 2) + (n + 3) + (n + 4)) / 5)) → n ≥ 0 :=
by sorry

end NUMINAMATH_CALUDE_smallest_integer_in_set_l3406_340624


namespace NUMINAMATH_CALUDE_function_properties_l3406_340652

-- Define the function f
variable (f : ℝ → ℝ)

-- Define the conditions
variable (h₁ : ∃ x, f x ≠ 0)
variable (h₂ : ∀ x, f (x + 3) = -f (3 - x))
variable (h₃ : ∀ x, f (x + 4) = -f (4 - x))

-- Theorem statement
theorem function_properties :
  (∀ x, f (-x) = -f x) ∧ (∃ p > 0, ∀ x, f (x + p) = f x) :=
sorry

end NUMINAMATH_CALUDE_function_properties_l3406_340652


namespace NUMINAMATH_CALUDE_find_n_l3406_340675

theorem find_n (a n : ℕ) (h1 : a^2 % n = 8) (h2 : a^3 % n = 25) : n = 113 := by
  sorry

end NUMINAMATH_CALUDE_find_n_l3406_340675


namespace NUMINAMATH_CALUDE_sum_of_squares_of_roots_l3406_340673

theorem sum_of_squares_of_roots (a b c : ℚ) (h : a = 6 ∧ b = 9 ∧ c = -21) :
  let x₁ := (-b + Real.sqrt (b^2 - 4*a*c)) / (2*a)
  let x₂ := (-b - Real.sqrt (b^2 - 4*a*c)) / (2*a)
  x₁^2 + x₂^2 = 37/4 := by sorry

end NUMINAMATH_CALUDE_sum_of_squares_of_roots_l3406_340673


namespace NUMINAMATH_CALUDE_number_ratio_l3406_340698

theorem number_ratio (x : ℝ) (h : 3 * (2 * x + 9) = 69) : 2 * x / x = 2 := by
  sorry

end NUMINAMATH_CALUDE_number_ratio_l3406_340698


namespace NUMINAMATH_CALUDE_quotient_problem_l3406_340626

theorem quotient_problem (dividend : ℕ) (divisor : ℕ) (remainder : ℕ) (quotient : ℕ) 
  (h1 : dividend = 158)
  (h2 : divisor = 17)
  (h3 : remainder = 5)
  (h4 : dividend = quotient * divisor + remainder) :
  quotient = 9 := by
  sorry

end NUMINAMATH_CALUDE_quotient_problem_l3406_340626


namespace NUMINAMATH_CALUDE_expansion_coefficients_theorem_l3406_340670

def binomial_expansion (x y : ℤ) (n : ℕ) := (x + y)^n

def max_coefficient (x y : ℤ) (n : ℕ) : ℕ := Nat.choose n (n / 2)

def second_largest_coefficient (x y : ℤ) (n : ℕ) : ℕ :=
  max (Nat.choose n ((n + 1) / 2)) (Nat.choose n ((n - 1) / 2))

theorem expansion_coefficients_theorem :
  let x : ℤ := 2
  let y : ℤ := 8
  let n : ℕ := 8
  max_coefficient x y n = 70 ∧
  second_largest_coefficient x y n = 1792 ∧
  (second_largest_coefficient x y n : ℚ) / (max_coefficient x y n : ℚ) = 128 / 5 := by
  sorry

#check expansion_coefficients_theorem

end NUMINAMATH_CALUDE_expansion_coefficients_theorem_l3406_340670
