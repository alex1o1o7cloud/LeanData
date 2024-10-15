import Mathlib

namespace NUMINAMATH_CALUDE_nonagon_angle_measure_l2637_263718

theorem nonagon_angle_measure :
  ∀ (small_angle large_angle : ℝ),
  (9 : ℝ) * small_angle + (9 : ℝ) * large_angle = (7 : ℝ) * 180 →
  6 * small_angle + 3 * large_angle = (7 : ℝ) * 180 →
  large_angle = 3 * small_angle →
  large_angle = 252 := by
  sorry

end NUMINAMATH_CALUDE_nonagon_angle_measure_l2637_263718


namespace NUMINAMATH_CALUDE_fourth_grade_students_l2637_263799

theorem fourth_grade_students (initial_students leaving_students new_students : ℕ) :
  initial_students = 35 →
  leaving_students = 10 →
  new_students = 10 →
  initial_students - leaving_students + new_students = 35 := by
sorry

end NUMINAMATH_CALUDE_fourth_grade_students_l2637_263799


namespace NUMINAMATH_CALUDE_simplify_logarithmic_expression_l2637_263747

-- Define the base-10 logarithm
noncomputable def lg (x : ℝ) : ℝ := Real.log x / Real.log 10

-- State the theorem
theorem simplify_logarithmic_expression :
  lg 5 * lg 20 - lg 2 * lg 50 - lg 25 = -lg 5 :=
by sorry

end NUMINAMATH_CALUDE_simplify_logarithmic_expression_l2637_263747


namespace NUMINAMATH_CALUDE_enrique_commission_l2637_263729

/-- Calculates the commission earned by Enrique based on his sales --/
theorem enrique_commission :
  let commission_rate : ℚ := 15 / 100
  let suit_price : ℚ := 700
  let suit_quantity : ℕ := 2
  let shirt_price : ℚ := 50
  let shirt_quantity : ℕ := 6
  let loafer_price : ℚ := 150
  let loafer_quantity : ℕ := 2
  let total_sales : ℚ := suit_price * suit_quantity + shirt_price * shirt_quantity + loafer_price * loafer_quantity
  let commission : ℚ := commission_rate * total_sales
  commission = 300
  := by sorry

end NUMINAMATH_CALUDE_enrique_commission_l2637_263729


namespace NUMINAMATH_CALUDE_recipe_reduction_l2637_263792

/-- Represents a mixed number as a pair of integers (whole, fractional) -/
def MixedNumber := ℤ × ℚ

/-- Converts a mixed number to a rational number -/
def mixedToRational (m : MixedNumber) : ℚ :=
  m.1 + m.2

/-- The amount of flour in the original recipe -/
def originalFlour : MixedNumber := (5, 3/4)

/-- The amount of sugar in the original recipe -/
def originalSugar : MixedNumber := (2, 1/2)

/-- The fraction of the recipe we want to make -/
def recipeFraction : ℚ := 1/3

theorem recipe_reduction :
  (mixedToRational originalFlour * recipeFraction = 23/12) ∧
  (mixedToRational originalSugar * recipeFraction = 5/6) :=
sorry

end NUMINAMATH_CALUDE_recipe_reduction_l2637_263792


namespace NUMINAMATH_CALUDE_house_cleaning_time_l2637_263714

/-- Proves that given John cleans the entire house in 6 hours and Nick takes 3 times as long as John to clean half the house, the time it takes for them to clean the entire house together is 3.6 hours. -/
theorem house_cleaning_time (john_time nick_time combined_time : ℝ) : 
  john_time = 6 → 
  nick_time = 3 * (john_time / 2) → 
  combined_time = 18 / 5 → 
  combined_time = 3.6 :=
by sorry

end NUMINAMATH_CALUDE_house_cleaning_time_l2637_263714


namespace NUMINAMATH_CALUDE_rattle_ownership_l2637_263711

structure Brother :=
  (id : ℕ)
  (claims_ownership : Bool)

def Alice := Unit

def determine_owner (b1 b2 : Brother) (a : Alice) : Brother :=
  sorry

theorem rattle_ownership (b1 b2 : Brother) (a : Alice) :
  b1.id = 1 →
  b2.id = 2 →
  b1.claims_ownership = true →
  (determine_owner b1 b2 a).id = 1 :=
sorry

end NUMINAMATH_CALUDE_rattle_ownership_l2637_263711


namespace NUMINAMATH_CALUDE_fifth_root_of_unity_l2637_263715

theorem fifth_root_of_unity (p q r s t m : ℂ) :
  p ≠ 0 →
  p * m^4 + q * m^3 + r * m^2 + s * m + t = 0 →
  q * m^4 + r * m^3 + s * m^2 + t * m + p = 0 →
  m^5 = 1 :=
by sorry

end NUMINAMATH_CALUDE_fifth_root_of_unity_l2637_263715


namespace NUMINAMATH_CALUDE_strawberry_price_difference_l2637_263794

theorem strawberry_price_difference (sale_price regular_price : ℚ) : 
  (54 * sale_price = 216) →
  (54 * regular_price = 216 + 108) →
  regular_price - sale_price = 2 := by
sorry

end NUMINAMATH_CALUDE_strawberry_price_difference_l2637_263794


namespace NUMINAMATH_CALUDE_min_value_of_f_l2637_263780

-- Define the function f
def f (x : ℝ) : ℝ := x^2 - 4*x + 4

-- State the theorem
theorem min_value_of_f :
  ∃ (x_min : ℝ), ∀ (x : ℝ), f x_min ≤ f x ∧ f x_min = 0 := by
  sorry

end NUMINAMATH_CALUDE_min_value_of_f_l2637_263780


namespace NUMINAMATH_CALUDE_pen_distribution_l2637_263766

theorem pen_distribution (num_pencils : ℕ) (num_students : ℕ) (num_pens : ℕ) : 
  num_pencils = 828 →
  num_students = 4 →
  num_pencils % num_students = 0 →
  num_pens % num_students = 0 →
  ∃ k : ℕ, num_pens = 4 * k :=
by sorry

end NUMINAMATH_CALUDE_pen_distribution_l2637_263766


namespace NUMINAMATH_CALUDE_simplify_expression_l2637_263702

theorem simplify_expression (w : ℝ) : -2*w + 3 - 4*w + 7 + 6*w - 5 - 8*w + 8 = -8*w + 13 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l2637_263702


namespace NUMINAMATH_CALUDE_spatial_diagonals_count_l2637_263743

/-- A convex polyhedron with specified properties -/
structure ConvexPolyhedron where
  vertices : ℕ
  edges : ℕ
  faces : ℕ
  triangular_faces : ℕ
  quadrilateral_faces : ℕ

/-- Calculate the number of spatial diagonals in a convex polyhedron -/
def spatial_diagonals (P : ConvexPolyhedron) : ℕ :=
  Nat.choose P.vertices 2 - P.edges - 2 * P.quadrilateral_faces

/-- Theorem stating the number of spatial diagonals in the given polyhedron -/
theorem spatial_diagonals_count (P : ConvexPolyhedron) 
  (h1 : P.vertices = 26)
  (h2 : P.edges = 60)
  (h3 : P.faces = 36)
  (h4 : P.triangular_faces = 24)
  (h5 : P.quadrilateral_faces = 12)
  (h6 : P.triangular_faces + P.quadrilateral_faces = P.faces) :
  spatial_diagonals P = 241 := by
  sorry

#eval spatial_diagonals ⟨26, 60, 36, 24, 12⟩

end NUMINAMATH_CALUDE_spatial_diagonals_count_l2637_263743


namespace NUMINAMATH_CALUDE_not_both_bidirectional_l2637_263791

-- Define the proof methods
inductive ProofMethod
| Synthetic
| Analytic

-- Define the reasoning directions
inductive ReasoningDirection
| CauseToEffect
| EffectToCause

-- Define the properties of the proof methods
def methodProperties (m : ProofMethod) : ReasoningDirection :=
  match m with
  | ProofMethod.Synthetic => ReasoningDirection.CauseToEffect
  | ProofMethod.Analytic => ReasoningDirection.EffectToCause

-- Theorem statement
theorem not_both_bidirectional : 
  ¬ (∀ (m : ProofMethod), 
      methodProperties m = ReasoningDirection.CauseToEffect ∧ 
      methodProperties m = ReasoningDirection.EffectToCause) :=
by sorry

end NUMINAMATH_CALUDE_not_both_bidirectional_l2637_263791


namespace NUMINAMATH_CALUDE_xiao_wang_processes_60_parts_l2637_263756

/-- Represents the number of parts processed by a worker in a given time period -/
def ProcessedParts (rate : ℕ) (workTime : ℕ) : ℕ := rate * workTime

/-- Represents the total time taken by Xiao Wang to process a given number of parts -/
def XiaoWangTotalTime (parts : ℕ) : ℚ :=
  let workHours := parts / 15
  let breaks := workHours / 2
  (workHours + breaks : ℚ)

/-- Represents the total time taken by Xiao Li to process a given number of parts -/
def XiaoLiTotalTime (parts : ℕ) : ℚ := parts / 12

/-- Theorem stating that Xiao Wang processes 60 parts when both finish at the same time -/
theorem xiao_wang_processes_60_parts :
  ∃ (parts : ℕ), parts = 60 ∧ XiaoWangTotalTime parts = XiaoLiTotalTime parts :=
sorry

end NUMINAMATH_CALUDE_xiao_wang_processes_60_parts_l2637_263756


namespace NUMINAMATH_CALUDE_star_not_associative_l2637_263706

-- Define the set T as non-zero real numbers
def T := {x : ℝ | x ≠ 0}

-- Define the binary operation ★
def star (x y : ℝ) : ℝ := 3 * x * y + x + y

-- Theorem stating that ★ is not associative over T
theorem star_not_associative :
  ∃ (x y z : T), star (star x y) z ≠ star x (star y z) := by
  sorry

end NUMINAMATH_CALUDE_star_not_associative_l2637_263706


namespace NUMINAMATH_CALUDE_nonagon_diagonal_intersection_probability_l2637_263734

/-- A regular nonagon is a 9-sided regular polygon -/
def RegularNonagon : Type := Unit

/-- The number of vertices in a regular nonagon -/
def num_vertices : ℕ := 9

/-- The number of diagonals in a regular nonagon -/
def num_diagonals (n : RegularNonagon) : ℕ := (num_vertices * (num_vertices - 3)) / 2

/-- The number of pairs of intersecting diagonals in a regular nonagon -/
def num_intersecting_diagonals (n : RegularNonagon) : ℕ := Nat.choose num_vertices 4

/-- The total number of pairs of diagonals in a regular nonagon -/
def total_diagonal_pairs (n : RegularNonagon) : ℕ := Nat.choose (num_diagonals n) 2

/-- The probability that two randomly chosen diagonals intersect inside the nonagon -/
def intersection_probability (n : RegularNonagon) : ℚ :=
  (num_intersecting_diagonals n : ℚ) / (total_diagonal_pairs n : ℚ)

theorem nonagon_diagonal_intersection_probability (n : RegularNonagon) :
  intersection_probability n = 14 / 39 := by
  sorry

end NUMINAMATH_CALUDE_nonagon_diagonal_intersection_probability_l2637_263734


namespace NUMINAMATH_CALUDE_ellipse_equation_l2637_263740

/-- The standard equation of an ellipse with given eccentricity and major axis length -/
theorem ellipse_equation (e : ℝ) (major_axis : ℝ) (h_e : e = 1 / 2) (h_major : major_axis = 4) :
  ∃ (a b : ℝ), a > 0 ∧ b > 0 ∧ a > b ∧
  (∀ (x y : ℝ), (x^2 / a^2 + y^2 / b^2 = 1) ↔ (x^2 / 4 + y^2 / 3 = 1)) :=
by sorry

end NUMINAMATH_CALUDE_ellipse_equation_l2637_263740


namespace NUMINAMATH_CALUDE_joans_balloons_l2637_263790

theorem joans_balloons (initial_balloons final_balloons : ℕ) 
  (h1 : initial_balloons = 8)
  (h2 : final_balloons = 10) :
  final_balloons - initial_balloons = 2 := by
  sorry

end NUMINAMATH_CALUDE_joans_balloons_l2637_263790


namespace NUMINAMATH_CALUDE_y_divisibility_l2637_263733

def y : ℕ := 96 + 144 + 200 + 320 + 480 + 512 + 4096

theorem y_divisibility :
  (∃ k : ℕ, y = 4 * k) ∧
  (∃ k : ℕ, y = 8 * k) ∧
  (∃ k : ℕ, y = 16 * k) ∧
  ¬(∃ k : ℕ, y = 32 * k) := by
  sorry

end NUMINAMATH_CALUDE_y_divisibility_l2637_263733


namespace NUMINAMATH_CALUDE_apples_given_to_larry_l2637_263730

/-- Given that Joyce starts with 75 apples and ends up with 23 apples,
    prove that she gave 52 apples to Larry. -/
theorem apples_given_to_larry (initial : ℕ) (final : ℕ) (given : ℕ) :
  initial = 75 →
  final = 23 →
  given = initial - final →
  given = 52 := by sorry

end NUMINAMATH_CALUDE_apples_given_to_larry_l2637_263730


namespace NUMINAMATH_CALUDE_cos_75_degrees_l2637_263778

theorem cos_75_degrees :
  Real.cos (75 * π / 180) = (Real.sqrt 6 - Real.sqrt 2) / 4 := by
  sorry

end NUMINAMATH_CALUDE_cos_75_degrees_l2637_263778


namespace NUMINAMATH_CALUDE_probability_first_greater_than_second_l2637_263707

def card_set : Finset ℕ := {1, 2, 3, 4, 5}

def favorable_outcomes : Finset (ℕ × ℕ) :=
  {(2, 1), (3, 1), (3, 2), (4, 1), (4, 2), (4, 3), (5, 1), (5, 2), (5, 3), (5, 4)}

theorem probability_first_greater_than_second :
  (Finset.card favorable_outcomes : ℚ) / (Finset.card card_set ^ 2 : ℚ) = 2 / 5 := by
  sorry

end NUMINAMATH_CALUDE_probability_first_greater_than_second_l2637_263707


namespace NUMINAMATH_CALUDE_selection_theorem_1_selection_theorem_2_selection_theorem_3_l2637_263732

/-- The number of female students -/
def num_female : ℕ := 5

/-- The number of male students -/
def num_male : ℕ := 4

/-- The number of students to be selected -/
def num_selected : ℕ := 4

/-- The number of ways to select exactly 2 male and 2 female students -/
def selection_method_1 : ℕ := 1440

/-- The number of ways to select at least 1 male and 1 female student -/
def selection_method_2 : ℕ := 2880

/-- The number of ways to select at least 1 male and 1 female student, 
    but male student A and female student B cannot be selected together -/
def selection_method_3 : ℕ := 2376

/-- Theorem for the first selection method -/
theorem selection_theorem_1 : 
  (Nat.choose num_male 2 * Nat.choose num_female 2) * (Nat.factorial num_selected) = selection_method_1 := by
  sorry

/-- Theorem for the second selection method -/
theorem selection_theorem_2 : 
  ((Nat.choose num_male 1 * Nat.choose num_female 3) + 
   (Nat.choose num_male 2 * Nat.choose num_female 2) + 
   (Nat.choose num_male 3 * Nat.choose num_female 1)) * 
  (Nat.factorial num_selected) = selection_method_2 := by
  sorry

/-- Theorem for the third selection method -/
theorem selection_theorem_3 : 
  selection_method_2 - 
  ((Nat.choose (num_male - 1) 2 + Nat.choose (num_female - 1) 1 * Nat.choose (num_male - 1) 1 + 
    Nat.choose (num_female - 1) 2) * Nat.factorial num_selected) = selection_method_3 := by
  sorry

end NUMINAMATH_CALUDE_selection_theorem_1_selection_theorem_2_selection_theorem_3_l2637_263732


namespace NUMINAMATH_CALUDE_spencer_jump_rope_session_length_l2637_263762

/-- Proves that Spencer's jump rope session length is 10 minutes -/
theorem spencer_jump_rope_session_length :
  ∀ (jumps_per_minute : ℕ) 
    (sessions_per_day : ℕ) 
    (total_jumps : ℕ) 
    (total_days : ℕ),
  jumps_per_minute = 4 →
  sessions_per_day = 2 →
  total_jumps = 400 →
  total_days = 5 →
  (total_jumps / total_days / sessions_per_day) / jumps_per_minute = 10 :=
by
  sorry

#check spencer_jump_rope_session_length

end NUMINAMATH_CALUDE_spencer_jump_rope_session_length_l2637_263762


namespace NUMINAMATH_CALUDE_rectangular_prism_volume_l2637_263738

theorem rectangular_prism_volume
  (face_area1 face_area2 face_area3 : ℝ)
  (h1 : face_area1 = 15)
  (h2 : face_area2 = 20)
  (h3 : face_area3 = 30)
  (h4 : ∃ l w h : ℝ, l * w = face_area1 ∧ w * h = face_area2 ∧ l * h = face_area3) :
  ∃ volume : ℝ, volume = 30 * Real.sqrt 10 ∧
    (∀ l w h : ℝ, l * w = face_area1 → w * h = face_area2 → l * h = face_area3 →
      volume = l * w * h) :=
by sorry

end NUMINAMATH_CALUDE_rectangular_prism_volume_l2637_263738


namespace NUMINAMATH_CALUDE_quadratic_inequality_min_value_l2637_263736

theorem quadratic_inequality_min_value (a b : ℝ) (h1 : a > b)
  (h2 : ∀ x : ℝ, a * x^2 + 2 * x + b ≥ 0)
  (h3 : ∃ x0 : ℝ, a * x0^2 + 2 * x0 + b = 0) :
  (∀ x : ℝ, (a^2 + b^2) / (a - b) ≥ 2 * Real.sqrt 2) ∧
  (∃ x : ℝ, (a^2 + b^2) / (a - b) = 2 * Real.sqrt 2) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_min_value_l2637_263736


namespace NUMINAMATH_CALUDE_cody_marbles_l2637_263709

/-- The number of marbles Cody gave to his brother -/
def marbles_given : ℕ := 5

/-- The number of marbles Cody has now -/
def marbles_now : ℕ := 7

/-- The initial number of marbles Cody had -/
def initial_marbles : ℕ := marbles_now + marbles_given

theorem cody_marbles : initial_marbles = 12 := by
  sorry

end NUMINAMATH_CALUDE_cody_marbles_l2637_263709


namespace NUMINAMATH_CALUDE_unrolled_value_is_four_fifty_l2637_263752

/-- The number of quarters -/
def total_quarters : ℕ := 100

/-- The number of dimes -/
def total_dimes : ℕ := 185

/-- The capacity of a roll of quarters -/
def quarters_per_roll : ℕ := 45

/-- The capacity of a roll of dimes -/
def dimes_per_roll : ℕ := 55

/-- The value of a quarter in dollars -/
def quarter_value : ℚ := 1/4

/-- The value of a dime in dollars -/
def dime_value : ℚ := 1/10

/-- The total dollar value of coins that cannot be rolled -/
def unrolled_value : ℚ :=
  (total_quarters % quarters_per_roll) * quarter_value +
  (total_dimes % dimes_per_roll) * dime_value

theorem unrolled_value_is_four_fifty :
  unrolled_value = 9/2 := by sorry

end NUMINAMATH_CALUDE_unrolled_value_is_four_fifty_l2637_263752


namespace NUMINAMATH_CALUDE_ratio_problem_l2637_263772

theorem ratio_problem (w x y z : ℚ) 
  (h1 : w / x = 4 / 3)
  (h2 : y / z = 3 / 2)
  (h3 : z / x = 1 / 6) :
  w / y = 16 / 3 := by
  sorry

end NUMINAMATH_CALUDE_ratio_problem_l2637_263772


namespace NUMINAMATH_CALUDE_min_distance_to_line_l2637_263749

/-- The curve function f(x) = x^2 - ln x -/
noncomputable def f (x : ℝ) : ℝ := x^2 - Real.log x

/-- The line function g(x) = x - 2 -/
def g (x : ℝ) : ℝ := x - 2

/-- A point P on the curve -/
structure PointOnCurve where
  x : ℝ
  y : ℝ
  h : y = f x

/-- Theorem: The minimum distance from any point on the curve to the line is 1 -/
theorem min_distance_to_line (P : PointOnCurve) : 
  ∃ (d : ℝ), d = 1 ∧ ∀ (Q : ℝ × ℝ), Q.2 = g Q.1 → Real.sqrt ((P.x - Q.1)^2 + (P.y - Q.2)^2) ≥ d :=
sorry

end NUMINAMATH_CALUDE_min_distance_to_line_l2637_263749


namespace NUMINAMATH_CALUDE_train_length_l2637_263796

/-- The length of a train given its speed, the speed of a man walking in the opposite direction, and the time it takes for the train to pass the man. -/
theorem train_length (train_speed : ℝ) (man_speed : ℝ) (crossing_time : ℝ) :
  train_speed = 174.98560115190784 →
  man_speed = 5 →
  crossing_time = 10 →
  ∃ (length : ℝ), abs (length - 499.96) < 0.01 ∧
    length = (train_speed + man_speed) * (1000 / 3600) * crossing_time :=
sorry

end NUMINAMATH_CALUDE_train_length_l2637_263796


namespace NUMINAMATH_CALUDE_sum_of_numbers_l2637_263717

theorem sum_of_numbers (x y : ℝ) (h1 : x > 0) (h2 : y > 0) (h3 : x * y = 16) 
  (h4 : 1 / x = 3 * (1 / y)) : x + y = (16 * Real.sqrt 3) / 3 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_numbers_l2637_263717


namespace NUMINAMATH_CALUDE_prob_three_odd_in_six_rolls_prob_three_odd_in_six_rolls_correct_l2637_263795

/-- The probability of getting exactly 3 odd numbers when rolling a fair 6-sided die 6 times -/
theorem prob_three_odd_in_six_rolls : ℚ :=
  5/16

/-- Proves that the probability of getting exactly 3 odd numbers when rolling a fair 6-sided die 6 times is 5/16 -/
theorem prob_three_odd_in_six_rolls_correct : prob_three_odd_in_six_rolls = 5/16 := by
  sorry

end NUMINAMATH_CALUDE_prob_three_odd_in_six_rolls_prob_three_odd_in_six_rolls_correct_l2637_263795


namespace NUMINAMATH_CALUDE_zoo_trip_money_left_l2637_263786

/-- The amount of money left for lunch and snacks after a zoo trip -/
def money_left_for_lunch_and_snacks (
  zoo_ticket_price : ℚ)
  (bus_fare_one_way : ℚ)
  (total_money : ℚ)
  (num_people : ℕ) : ℚ :=
  total_money - (num_people * zoo_ticket_price + 2 * num_people * bus_fare_one_way)

/-- Theorem: Noah and Ava have $24 left for lunch and snacks after their zoo trip -/
theorem zoo_trip_money_left :
  money_left_for_lunch_and_snacks 5 (3/2) 40 2 = 24 := by
  sorry

end NUMINAMATH_CALUDE_zoo_trip_money_left_l2637_263786


namespace NUMINAMATH_CALUDE_pentagon_angle_measure_l2637_263764

theorem pentagon_angle_measure (Q R S T U : ℝ) :
  R = 120 ∧ S = 94 ∧ T = 115 ∧ U = 101 →
  Q + R + S + T + U = 540 →
  Q = 110 := by
sorry

end NUMINAMATH_CALUDE_pentagon_angle_measure_l2637_263764


namespace NUMINAMATH_CALUDE_karen_cookies_to_grandparents_l2637_263739

/-- The number of cookies Karen gave to her grandparents -/
def cookies_to_grandparents (total_cookies class_size cookies_per_student cookies_for_self : ℕ) : ℕ :=
  total_cookies - (cookies_for_self + class_size * cookies_per_student)

/-- Theorem stating the number of cookies Karen gave to her grandparents -/
theorem karen_cookies_to_grandparents :
  cookies_to_grandparents 50 16 2 10 = 8 := by
  sorry

end NUMINAMATH_CALUDE_karen_cookies_to_grandparents_l2637_263739


namespace NUMINAMATH_CALUDE_line_bisects_circle_l2637_263765

/-- The equation of the circle -/
def circle_equation (x y : ℝ) : Prop :=
  x^2 + y^2 + 4*x - 4*y - 8 = 0

/-- The equation of the line -/
def line_equation (x y b : ℝ) : Prop :=
  y = x + b

/-- The center of the circle -/
def circle_center : ℝ × ℝ := (-2, 2)

/-- The line bisects the circumference of the circle if it passes through the center -/
def bisects_circle (b : ℝ) : Prop :=
  let (cx, cy) := circle_center
  line_equation cx cy b

theorem line_bisects_circle (b : ℝ) :
  bisects_circle b → b = 4 := by sorry

end NUMINAMATH_CALUDE_line_bisects_circle_l2637_263765


namespace NUMINAMATH_CALUDE_tangent_circle_radius_l2637_263771

/-- A circle tangent to coordinate axes and the hypotenuse of a 45-45-90 triangle --/
structure TangentCircle where
  O : ℝ × ℝ  -- Center of the circle
  r : ℝ      -- Radius of the circle
  h : r > 0  -- Radius is positive

/-- A 45-45-90 triangle with side length 2 --/
structure RightIsoscelesTriangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ
  h1 : B.1 - A.1 = 2  -- AB has length 2
  h2 : C.2 - A.2 = 2  -- AC has length 2 in y-direction
  h3 : B.2 = A.2      -- AB is horizontal

/-- The main theorem --/
theorem tangent_circle_radius
  (t : TangentCircle)
  (tri : RightIsoscelesTriangle)
  (h_tangent_x : t.O.2 = t.r)
  (h_tangent_y : t.O.1 = t.r)
  (h_tangent_hyp : t.O.2 + t.r = tri.C.2) :
  t.r = 2 := by
  sorry

end NUMINAMATH_CALUDE_tangent_circle_radius_l2637_263771


namespace NUMINAMATH_CALUDE_prism_volume_in_cubic_yards_l2637_263781

/-- Conversion factor from cubic feet to cubic yards -/
def cubic_feet_per_cubic_yard : ℝ := 27

/-- Volume of the rectangular prism in cubic feet -/
def prism_volume_cubic_feet : ℝ := 216

/-- Theorem stating that the volume of the prism in cubic yards is 8 -/
theorem prism_volume_in_cubic_yards :
  prism_volume_cubic_feet / cubic_feet_per_cubic_yard = 8 := by
  sorry

end NUMINAMATH_CALUDE_prism_volume_in_cubic_yards_l2637_263781


namespace NUMINAMATH_CALUDE_exam_score_problem_l2637_263704

theorem exam_score_problem (correct_score : ℕ) (wrong_score : ℕ) 
  (total_score : ℕ) (num_correct : ℕ) :
  correct_score = 3 →
  wrong_score = 1 →
  total_score = 180 →
  num_correct = 75 →
  ∃ (num_wrong : ℕ), 
    total_score = correct_score * num_correct - wrong_score * num_wrong ∧
    num_correct + num_wrong = 120 :=
by sorry

end NUMINAMATH_CALUDE_exam_score_problem_l2637_263704


namespace NUMINAMATH_CALUDE_jack_final_apples_l2637_263721

def initial_apples : ℕ := 150
def sold_to_jill_percent : ℚ := 30 / 100
def sold_to_june_percent : ℚ := 20 / 100
def apples_eaten : ℕ := 2
def apples_given_to_teacher : ℕ := 1

theorem jack_final_apples :
  let after_jill := initial_apples - (initial_apples * sold_to_jill_percent).floor
  let after_june := after_jill - (after_jill * sold_to_june_percent).floor
  let after_eating := after_june - apples_eaten
  let final_apples := after_eating - apples_given_to_teacher
  final_apples = 81 := by sorry

end NUMINAMATH_CALUDE_jack_final_apples_l2637_263721


namespace NUMINAMATH_CALUDE_sinusoidal_function_properties_l2637_263726

/-- Given a sinusoidal function y = a * sin(b * x + c) with a > 0 and b > 0,
    if the maximum occurs at x = π/6 and the amplitude is 3,
    then a = 3 and c = (3 - b) * π/6 -/
theorem sinusoidal_function_properties (a b c : ℝ) (h1 : a > 0) (h2 : b > 0)
  (h3 : ∀ x, a * Real.sin (b * x + c) ≤ a * Real.sin (b * (π/6) + c))
  (h4 : a = 3) :
  a = 3 ∧ c = (3 - b) * π/6 := by
sorry

end NUMINAMATH_CALUDE_sinusoidal_function_properties_l2637_263726


namespace NUMINAMATH_CALUDE_function_minimum_and_inequality_l2637_263744

-- Define the function f
def f (a b x : ℝ) : ℝ := |x + a| + |2*x - b|

-- State the theorem
theorem function_minimum_and_inequality (a b : ℝ) 
  (ha : a > 0) (hb : b > 0) 
  (hmin : ∀ x, f a b x ≥ 1) 
  (hequal : ∃ x, f a b x = 1) : 
  (2*a + b = 2) ∧ 
  (∀ t : ℝ, a + 2*b ≥ t*a*b → t ≤ 9/2) ∧
  (∃ t : ℝ, t = 9/2 ∧ a + 2*b = t*a*b) :=
by sorry

end NUMINAMATH_CALUDE_function_minimum_and_inequality_l2637_263744


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_set_l2637_263741

theorem quadratic_inequality_solution_set (a : ℝ) :
  let solution_set := {x : ℝ | a * x^2 - (2*a - 1) * x + (a - 1) < 0}
  if a > 0 then
    solution_set = {x : ℝ | (a - 1) / a < x ∧ x < 1}
  else if a = 0 then
    solution_set = {x : ℝ | x < 1}
  else
    solution_set = {x : ℝ | x > (a - 1) / a ∨ x < 1} :=
by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_set_l2637_263741


namespace NUMINAMATH_CALUDE_range_of_a_l2637_263750

-- Define propositions p and q
def p (a : ℝ) : Prop := -2 < a ∧ a ≤ 2
def q (a : ℝ) : Prop := 0 < a ∧ a < 1

-- Define the set of valid a values
def valid_a_set : Set ℝ := {a | (1 ≤ a ∧ a ≤ 2) ∨ (-2 < a ∧ a ≤ 0)}

-- Theorem statement
theorem range_of_a (a : ℝ) :
  (p a ∨ q a) ∧ ¬(p a ∧ q a) ↔ a ∈ valid_a_set :=
sorry

end NUMINAMATH_CALUDE_range_of_a_l2637_263750


namespace NUMINAMATH_CALUDE_car_cost_car_cost_proof_l2637_263769

/-- The cost of Alex's car, given his savings and earnings from grocery deliveries -/
theorem car_cost (initial_savings : ℝ) (trip_charge : ℝ) (grocery_percentage : ℝ) 
  (num_trips : ℕ) (grocery_value : ℝ) : ℝ :=
  let earnings_from_trips := num_trips * trip_charge
  let earnings_from_groceries := grocery_percentage * grocery_value
  let total_earnings := earnings_from_trips + earnings_from_groceries
  let total_savings := initial_savings + total_earnings
  total_savings

/-- Proof that the car costs $14,600 -/
theorem car_cost_proof : 
  car_cost 14500 1.5 0.05 40 800 = 14600 := by
  sorry

end NUMINAMATH_CALUDE_car_cost_car_cost_proof_l2637_263769


namespace NUMINAMATH_CALUDE_all_reals_satisfy_property_l2637_263753

theorem all_reals_satisfy_property :
  ∀ (α : ℝ), ∀ (n : ℕ), n > 0 → ∃ (m : ℤ), |α - (m : ℝ) / n| < 1 / (3 * n) :=
by sorry

end NUMINAMATH_CALUDE_all_reals_satisfy_property_l2637_263753


namespace NUMINAMATH_CALUDE_sum_of_roots_l2637_263713

/-- Given distinct real numbers p, q, r, s such that
    x^2 - 12px - 13q = 0 has roots r and s, and
    x^2 - 12rx - 13s = 0 has roots p and q,
    prove that p + q + r + s = 2028 -/
theorem sum_of_roots (p q r s : ℝ) 
  (h_distinct : p ≠ q ∧ p ≠ r ∧ p ≠ s ∧ q ≠ r ∧ q ≠ s ∧ r ≠ s)
  (h_roots1 : ∀ x, x^2 - 12*p*x - 13*q = 0 ↔ x = r ∨ x = s)
  (h_roots2 : ∀ x, x^2 - 12*r*x - 13*s = 0 ↔ x = p ∨ x = q) :
  p + q + r + s = 2028 := by
sorry

end NUMINAMATH_CALUDE_sum_of_roots_l2637_263713


namespace NUMINAMATH_CALUDE_locus_of_Q_l2637_263773

-- Define the circle
def Circle (O : ℝ × ℝ) (r : ℝ) := {P : ℝ × ℝ | (P.1 - O.1)^2 + (P.2 - O.2)^2 = r^2}

-- Define the points and their properties
def SymmetricalPoints (O A B : ℝ × ℝ) := A.1 + B.1 = 2 * O.1 ∧ A.2 + B.2 = 2 * O.2

-- Define the perpendicular chord
def PerpendicularChord (P P' A : ℝ × ℝ) := 
  (P'.1 - P.1) * (A.1 - P.1) + (P'.2 - P.2) * (A.2 - P.2) = 0

-- Define the symmetric point C
def SymmetricPoint (B C PP' : ℝ × ℝ) := 
  (C.1 - PP'.1) = (PP'.1 - B.1) ∧ (C.2 - PP'.2) = (PP'.2 - B.2)

-- Define the intersection point Q
def IntersectionPoint (Q PP' A C : ℝ × ℝ) := 
  (Q.1 - PP'.1) * (C.2 - A.2) = (Q.2 - PP'.2) * (C.1 - A.1) ∧
  (Q.1 - A.1) * (C.2 - A.2) = (Q.2 - A.2) * (C.1 - A.1)

-- Theorem statement
theorem locus_of_Q (O : ℝ × ℝ) (r : ℝ) (A B P P' C Q : ℝ × ℝ) :
  P ∈ Circle O r →
  SymmetricalPoints O A B →
  PerpendicularChord P P' A →
  SymmetricPoint B C P →
  IntersectionPoint Q P A C →
  ∃ (a b : ℝ), 
    (Q.1 / a)^2 + (Q.2 / b)^2 = 1 ∧
    a^2 - b^2 = (A.1 - B.1)^2 + (A.2 - B.2)^2 ∧
    a = r :=
by sorry

end NUMINAMATH_CALUDE_locus_of_Q_l2637_263773


namespace NUMINAMATH_CALUDE_teacher_number_game_l2637_263716

theorem teacher_number_game (x : ℝ) : 
  x = 5 → 3 * ((2 * (2 * x + 3)) + 2) = 84 := by
  sorry

end NUMINAMATH_CALUDE_teacher_number_game_l2637_263716


namespace NUMINAMATH_CALUDE_system_solution_l2637_263710

theorem system_solution : ∃! (x y z : ℝ), 
  x * y / (x + y) = 1 / 3 ∧
  y * z / (y + z) = 1 / 4 ∧
  z * x / (z + x) = 1 / 5 ∧
  x = 1 / 2 ∧ y = 1 ∧ z = 1 / 3 := by
sorry

end NUMINAMATH_CALUDE_system_solution_l2637_263710


namespace NUMINAMATH_CALUDE_factorial_base_700_a4_l2637_263755

/-- Factorial function -/
def factorial (n : ℕ) : ℕ :=
  match n with
  | 0 => 1
  | n + 1 => (n + 1) * factorial n

/-- Coefficient in factorial base representation -/
def factorial_base_coeff (n k : ℕ) : ℕ :=
  (n / factorial k) % (k + 1)

/-- Theorem: The coefficient a₄ in the factorial base representation of 700 is 4 -/
theorem factorial_base_700_a4 : factorial_base_coeff 700 4 = 4 := by
  sorry

end NUMINAMATH_CALUDE_factorial_base_700_a4_l2637_263755


namespace NUMINAMATH_CALUDE_amount_spent_on_books_l2637_263725

/-- Calculates the amount spent on books given the total allowance and percentages spent on other items --/
theorem amount_spent_on_books
  (total_allowance : ℚ)
  (games_percentage : ℚ)
  (clothes_percentage : ℚ)
  (snacks_percentage : ℚ)
  (h1 : total_allowance = 50)
  (h2 : games_percentage = 1/4)
  (h3 : clothes_percentage = 2/5)
  (h4 : snacks_percentage = 3/20) :
  total_allowance - (games_percentage + clothes_percentage + snacks_percentage) * total_allowance = 10 :=
by sorry

end NUMINAMATH_CALUDE_amount_spent_on_books_l2637_263725


namespace NUMINAMATH_CALUDE_sqrt_product_equals_six_l2637_263737

theorem sqrt_product_equals_six : Real.sqrt 8 * Real.sqrt (9/2) = 6 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_product_equals_six_l2637_263737


namespace NUMINAMATH_CALUDE_triangle_tan_A_l2637_263784

theorem triangle_tan_A (A B C : ℝ) (AB BC : ℝ) 
  (h_angle : A = π/3)
  (h_AB : AB = 20)
  (h_BC : BC = 21) : 
  Real.tan A = (21 * Real.sqrt 3) / (2 * Real.sqrt (421 - 1323/4)) := by
  sorry

end NUMINAMATH_CALUDE_triangle_tan_A_l2637_263784


namespace NUMINAMATH_CALUDE_triangle_with_small_angle_l2637_263720

-- Define a point in a plane
structure Point where
  x : ℝ
  y : ℝ

-- Define a set of n points
def PointSet (n : ℕ) := Fin n → Point

-- Define a function to calculate the angle between three points
noncomputable def angle (p1 p2 p3 : Point) : ℝ := sorry

-- Theorem statement
theorem triangle_with_small_angle (n : ℕ) (points : PointSet n) :
  ∃ (i j k : Fin n), i ≠ j ∧ j ≠ k ∧ i ≠ k ∧
    ∃ (θ : ℝ), θ ≤ 180 / n ∧
      (θ = angle (points i) (points j) (points k) ∨
       θ = angle (points j) (points k) (points i) ∨
       θ = angle (points k) (points i) (points j)) :=
sorry

end NUMINAMATH_CALUDE_triangle_with_small_angle_l2637_263720


namespace NUMINAMATH_CALUDE_tank_capacity_comparison_l2637_263761

theorem tank_capacity_comparison :
  let tank_a_height : ℝ := 10
  let tank_a_circumference : ℝ := 7
  let tank_b_height : ℝ := 7
  let tank_b_circumference : ℝ := 10
  let tank_a_volume := π * (tank_a_circumference / (2 * π))^2 * tank_a_height
  let tank_b_volume := π * (tank_b_circumference / (2 * π))^2 * tank_b_height
  (tank_a_volume / tank_b_volume) * 100 = 70 :=
by
  sorry

end NUMINAMATH_CALUDE_tank_capacity_comparison_l2637_263761


namespace NUMINAMATH_CALUDE_monica_savings_l2637_263779

/-- Calculates the total amount saved given the weekly savings, number of weeks, and number of repetitions. -/
def total_savings (weekly_savings : ℕ) (weeks : ℕ) (repetitions : ℕ) : ℕ :=
  weekly_savings * weeks * repetitions

/-- Proves that saving $15 per week for 60 weeks, repeated 5 times, results in a total savings of $4500. -/
theorem monica_savings : total_savings 15 60 5 = 4500 := by
  sorry

end NUMINAMATH_CALUDE_monica_savings_l2637_263779


namespace NUMINAMATH_CALUDE_all_N_composite_l2637_263770

def N (n : ℕ) : ℕ := 200 * 10^n + 88 * ((10^n - 1) / 9) + 21

theorem all_N_composite (n : ℕ) : ∃ (a b : ℕ), a > 1 ∧ b > 1 ∧ N n = a * b := by
  sorry

end NUMINAMATH_CALUDE_all_N_composite_l2637_263770


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_l2637_263767

def is_even_function (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f x = f (-x)

def phi_condition (φ : ℝ) : Prop :=
  ∃ k : ℤ, φ = 2 * k * Real.pi + Real.pi / 2

theorem sufficient_not_necessary :
  (∀ φ : ℝ, phi_condition φ → is_even_function (λ x => Real.sin (x + φ))) ∧
  (∃ φ : ℝ, is_even_function (λ x => Real.sin (x + φ)) ∧ ¬phi_condition φ) := by
  sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_l2637_263767


namespace NUMINAMATH_CALUDE_intersection_range_intersection_length_l2637_263731

-- Define the hyperbola and line
def hyperbola (x y : ℝ) : Prop := x^2 - y^2 = 1
def line (k : ℝ) (x y : ℝ) : Prop := y = k * x + 1

-- Define the intersection condition
def intersects_at_two_points (k : ℝ) : Prop :=
  ∃ x₁ y₁ x₂ y₂ : ℝ, x₁ ≠ x₂ ∧ hyperbola x₁ y₁ ∧ hyperbola x₂ y₂ ∧ line k x₁ y₁ ∧ line k x₂ y₂

-- Define the range of k
def k_range (k : ℝ) : Prop :=
  (k > -Real.sqrt 2 ∧ k < -1) ∨ (k > -1 ∧ k < 1) ∨ (k > 1 ∧ k < Real.sqrt 2)

-- Define the midpoint condition
def midpoint_condition (k : ℝ) : Prop :=
  ∃ x₁ y₁ x₂ y₂ : ℝ, hyperbola x₁ y₁ ∧ hyperbola x₂ y₂ ∧ line k x₁ y₁ ∧ line k x₂ y₂ ∧
    (x₁ + x₂) / 2 = Real.sqrt 2

-- Theorem 1: Range of k for two distinct intersections
theorem intersection_range :
  ∀ k : ℝ, intersects_at_two_points k ↔ k_range k := by sorry

-- Theorem 2: Length of AB when midpoint x-coordinate is √2
theorem intersection_length :
  ∀ k : ℝ, midpoint_condition k → 
    ∃ x₁ y₁ x₂ y₂ : ℝ, hyperbola x₁ y₁ ∧ hyperbola x₂ y₂ ∧ line k x₁ y₁ ∧ line k x₂ y₂ ∧
      Real.sqrt ((x₂ - x₁)^2 + (y₂ - y₁)^2) = 6 := by sorry

end NUMINAMATH_CALUDE_intersection_range_intersection_length_l2637_263731


namespace NUMINAMATH_CALUDE_exam_score_proof_l2637_263705

/-- Proves that the average score of students who took the exam on the assigned day is 60% -/
theorem exam_score_proof (total_students : ℕ) (assigned_day_percentage : ℝ) 
  (makeup_score : ℝ) (class_average : ℝ) : 
  total_students = 100 →
  assigned_day_percentage = 0.7 →
  makeup_score = 90 →
  class_average = 69 →
  let assigned_students := total_students * assigned_day_percentage
  let makeup_students := total_students - assigned_students
  let assigned_score := (class_average * total_students - makeup_score * makeup_students) / assigned_students
  assigned_score = 60 := by
sorry


end NUMINAMATH_CALUDE_exam_score_proof_l2637_263705


namespace NUMINAMATH_CALUDE_fraction_addition_and_simplification_l2637_263788

theorem fraction_addition_and_simplification :
  ∃ (n d : ℤ), (8 : ℚ) / 19 + (5 : ℚ) / 57 = n / d ∧ 
  n / d = (29 : ℚ) / 57 ∧
  (∀ k : ℤ, k ∣ n ∧ k ∣ d → k = 1 ∨ k = -1) :=
by sorry

end NUMINAMATH_CALUDE_fraction_addition_and_simplification_l2637_263788


namespace NUMINAMATH_CALUDE_michelangelo_painting_l2637_263748

theorem michelangelo_painting (total : ℕ) (left : ℕ) (this_week : ℕ) : 
  total = 28 → 
  left = 13 → 
  total - left = this_week + this_week / 4 →
  this_week = 12 := by
  sorry

end NUMINAMATH_CALUDE_michelangelo_painting_l2637_263748


namespace NUMINAMATH_CALUDE_evaluate_expression_l2637_263700

theorem evaluate_expression : (32 / (7 + 3 - 5)) * 8 = 51.2 := by
  sorry

end NUMINAMATH_CALUDE_evaluate_expression_l2637_263700


namespace NUMINAMATH_CALUDE_badge_exchange_l2637_263759

theorem badge_exchange (x : ℝ) : 
  (x + 5) - (24/100) * (x + 5) + (20/100) * x = x - (20/100) * x + (24/100) * (x + 5) - 1 → 
  x = 45 := by
sorry

end NUMINAMATH_CALUDE_badge_exchange_l2637_263759


namespace NUMINAMATH_CALUDE_sum_of_special_function_l2637_263723

theorem sum_of_special_function (f : ℝ → ℝ) 
  (h : ∀ x : ℝ, f (1/2 + x) + f (1/2 - x) = 2) : 
  f (1/8) + f (2/8) + f (3/8) + f (4/8) + f (5/8) + f (6/8) + f (7/8) = 7 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_special_function_l2637_263723


namespace NUMINAMATH_CALUDE_shirt_price_change_l2637_263754

theorem shirt_price_change (P : ℝ) (P_pos : P > 0) :
  P * (1 + 0.15) * (1 - 0.15) = P * 0.9775 := by
  sorry

#check shirt_price_change

end NUMINAMATH_CALUDE_shirt_price_change_l2637_263754


namespace NUMINAMATH_CALUDE_handshakes_count_l2637_263775

/-- Represents the social event with given conditions -/
structure SocialEvent where
  total_people : ℕ
  group_a_size : ℕ
  group_b_size : ℕ
  group_a_knows_all : group_a_size = 25
  group_b_knows_one : group_b_size = 15
  total_is_sum : total_people = group_a_size + group_b_size

/-- Calculates the number of handshakes in the social event -/
def count_handshakes (event : SocialEvent) : ℕ :=
  let group_b_internal_handshakes := (event.group_b_size * (event.group_b_size - 1)) / 2
  let group_a_b_handshakes := event.group_b_size * (event.group_a_size - 1)
  group_b_internal_handshakes + group_a_b_handshakes

/-- Theorem stating that the number of handshakes in the given social event is 465 -/
theorem handshakes_count (event : SocialEvent) : count_handshakes event = 465 := by
  sorry

end NUMINAMATH_CALUDE_handshakes_count_l2637_263775


namespace NUMINAMATH_CALUDE_problem1_l2637_263719

theorem problem1 (x : ℝ) : (12 * x^4 + 6 * x^2) / (3 * x) - (-2 * x)^2 * (x + 1) = 2 * x - 4 * x^2 := by
  sorry

end NUMINAMATH_CALUDE_problem1_l2637_263719


namespace NUMINAMATH_CALUDE_total_musicians_is_98_l2637_263768

/-- The total number of musicians in the orchestra, band, and choir -/
def total_musicians (orchestra_males orchestra_females band_multiplier choir_males choir_females : ℕ) : ℕ :=
  let orchestra_total := orchestra_males + orchestra_females
  let band_total := band_multiplier * orchestra_total
  let choir_total := choir_males + choir_females
  orchestra_total + band_total + choir_total

/-- Theorem stating that the total number of musicians is 98 given the specific conditions -/
theorem total_musicians_is_98 :
  total_musicians 11 12 2 12 17 = 98 := by
  sorry

end NUMINAMATH_CALUDE_total_musicians_is_98_l2637_263768


namespace NUMINAMATH_CALUDE_soccer_campers_count_l2637_263789

/-- The number of soccer campers at a summer sports camp -/
def soccer_campers (total : ℕ) (basketball : ℕ) (football : ℕ) : ℕ :=
  total - (basketball + football)

/-- Theorem stating the number of soccer campers given the conditions -/
theorem soccer_campers_count :
  soccer_campers 88 24 32 = 32 := by
  sorry

end NUMINAMATH_CALUDE_soccer_campers_count_l2637_263789


namespace NUMINAMATH_CALUDE_sequences_not_periodic_l2637_263774

/-- Sequence A constructed by writing slices of increasing lengths from 1,0,0,0,... -/
def sequence_A : ℕ → ℕ := sorry

/-- Sequence B constructed by writing slices of two, four, six, etc., elements from 1,2,3,1,2,3,... -/
def sequence_B : ℕ → ℕ := sorry

/-- Sequence C formed by adding the corresponding elements of A and B -/
def sequence_C (n : ℕ) : ℕ := sequence_A n + sequence_B n

/-- A sequence is periodic if there exists a positive integer k such that
    for all n ≥ some fixed N, a(n+k) = a(n) -/
def is_periodic (a : ℕ → ℕ) : Prop :=
  ∃ (k : ℕ) (h : k > 0), ∃ (N : ℕ), ∀ (n : ℕ), n ≥ N → a (n + k) = a n

theorem sequences_not_periodic :
  ¬(is_periodic sequence_A) ∧ ¬(is_periodic sequence_B) ∧ ¬(is_periodic sequence_C) := by sorry

end NUMINAMATH_CALUDE_sequences_not_periodic_l2637_263774


namespace NUMINAMATH_CALUDE_new_capacity_is_250_l2637_263797

/-- Calculates the new combined total lifting capacity after improvements -/
def new_total_capacity (initial_clean_jerk : ℝ) (initial_snatch : ℝ) : ℝ :=
  (2 * initial_clean_jerk) + (initial_snatch + 0.8 * initial_snatch)

/-- Theorem stating that given the initial capacities and improvements, 
    the new total capacity is 250 kg -/
theorem new_capacity_is_250 :
  new_total_capacity 80 50 = 250 := by
  sorry

end NUMINAMATH_CALUDE_new_capacity_is_250_l2637_263797


namespace NUMINAMATH_CALUDE_solution_set_inequality_l2637_263728

theorem solution_set_inequality (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  {x : ℝ | -b < 1/x ∧ 1/x < a} = {x : ℝ | x < -1/b ∨ x > 1/a} := by sorry

end NUMINAMATH_CALUDE_solution_set_inequality_l2637_263728


namespace NUMINAMATH_CALUDE_teacher_problem_l2637_263785

theorem teacher_problem (x : ℤ) : 4 * (3 * (x + 3) - 2) = 4 * (3 * x + 9 - 2) := by
  sorry

#check teacher_problem

end NUMINAMATH_CALUDE_teacher_problem_l2637_263785


namespace NUMINAMATH_CALUDE_point_D_coordinates_l2637_263712

/-- Given points A and B, and the relation between vectors AD and AB,
    prove that the coordinates of point D are (-7, 9) -/
theorem point_D_coordinates (A B D : ℝ × ℝ) : 
  A = (2, 3) → 
  B = (-1, 5) → 
  D - A = 3 • (B - A) → 
  D = (-7, 9) := by
sorry

end NUMINAMATH_CALUDE_point_D_coordinates_l2637_263712


namespace NUMINAMATH_CALUDE_simplify_radical_product_l2637_263782

theorem simplify_radical_product (x : ℝ) (h : x > 0) :
  Real.sqrt (48 * x) * Real.sqrt (3 * x) * (81 * x^2)^(1/3) = 36 * x * (3 * x^2)^(1/3) := by
  sorry

end NUMINAMATH_CALUDE_simplify_radical_product_l2637_263782


namespace NUMINAMATH_CALUDE_odd_m_triple_g_16_l2637_263703

def g (n : ℤ) : ℤ :=
  if n % 2 = 1 then n + 5
  else if n % 3 = 0 ∧ n % 2 ≠ 0 then n / 3
  else n / 2

theorem odd_m_triple_g_16 (m : ℤ) (h1 : m % 2 = 1) (h2 : g (g (g m)) = 16) :
  m = 59 ∨ m = 91 := by
  sorry

end NUMINAMATH_CALUDE_odd_m_triple_g_16_l2637_263703


namespace NUMINAMATH_CALUDE_existence_of_odd_fifth_powers_sum_l2637_263793

theorem existence_of_odd_fifth_powers_sum (m : ℤ) :
  ∃ (a b : ℤ) (k : ℕ+), 
    Odd a ∧ Odd b ∧ (2 * m = a^5 + b^5 + k * 2^100) := by
  sorry

end NUMINAMATH_CALUDE_existence_of_odd_fifth_powers_sum_l2637_263793


namespace NUMINAMATH_CALUDE_wheel_spinner_probability_wheel_spinner_probability_proof_l2637_263735

theorem wheel_spinner_probability : Real → Real → Real → Real → Prop :=
  fun prob_E prob_F prob_G prob_H =>
    prob_E = 1/2 →
    prob_F = 1/4 →
    prob_G = 2 * prob_H →
    prob_E + prob_F + prob_G + prob_H = 1 →
    prob_G = 1/6

-- The proof is omitted
theorem wheel_spinner_probability_proof : wheel_spinner_probability (1/2) (1/4) (1/6) (1/12) := by
  sorry

end NUMINAMATH_CALUDE_wheel_spinner_probability_wheel_spinner_probability_proof_l2637_263735


namespace NUMINAMATH_CALUDE_calculate_rates_l2637_263777

/-- Represents the rates and quantities in the problem -/
structure Rates where
  p : ℕ  -- number of pears
  b : ℕ  -- number of bananas
  d : ℕ  -- number of dishes
  tp : ℕ -- time spent picking pears (in hours)
  tb : ℕ -- time spent cooking bananas (in hours)
  tw : ℕ -- time spent washing dishes (in hours)
  rp : ℚ -- rate of picking pears (pears per hour)
  rb : ℚ -- rate of cooking bananas (bananas per hour)
  rw : ℚ -- rate of washing dishes (dishes per hour)

/-- The main theorem that proves the rates given the conditions -/
theorem calculate_rates (r : Rates) 
    (h1 : r.d = r.b + 10)
    (h2 : r.b = 3 * r.p)
    (h3 : r.p = 50)
    (h4 : r.tp = 4)
    (h5 : r.tb = 2)
    (h6 : r.tw = 5)
    : r.rp = 25/2 ∧ r.rb = 75 ∧ r.rw = 32 := by
  sorry


end NUMINAMATH_CALUDE_calculate_rates_l2637_263777


namespace NUMINAMATH_CALUDE_eraser_price_is_75_cents_l2637_263701

/-- The price of each eraser sold by the student council -/
def price_per_eraser (num_boxes : ℕ) (erasers_per_box : ℕ) (total_revenue : ℚ) : ℚ :=
  total_revenue / (num_boxes * erasers_per_box)

/-- Theorem: The price of each eraser is $0.75 -/
theorem eraser_price_is_75_cents :
  price_per_eraser 48 24 864 = 3/4 := by
  sorry

end NUMINAMATH_CALUDE_eraser_price_is_75_cents_l2637_263701


namespace NUMINAMATH_CALUDE_winter_olympics_volunteer_allocation_l2637_263776

theorem winter_olympics_volunteer_allocation :
  let n_volunteers : ℕ := 5
  let n_projects : ℕ := 4
  let allocation_schemes : ℕ := (n_volunteers.choose 2) * n_projects.factorial
  allocation_schemes = 240 :=
by sorry

end NUMINAMATH_CALUDE_winter_olympics_volunteer_allocation_l2637_263776


namespace NUMINAMATH_CALUDE_cube_volume_from_surface_area_l2637_263760

def cube_surface_area (s : ℝ) : ℝ := 6 * s^2

def cube_volume (s : ℝ) : ℝ := s^3

theorem cube_volume_from_surface_area (surface_area : ℝ) (h : surface_area = 150) :
  ∃ s : ℝ, cube_surface_area s = surface_area ∧ cube_volume s = 125 :=
by sorry

end NUMINAMATH_CALUDE_cube_volume_from_surface_area_l2637_263760


namespace NUMINAMATH_CALUDE_linda_furniture_fraction_l2637_263745

/-- Proves that the fraction of Linda's savings spent on furniture is 3/5 -/
theorem linda_furniture_fraction (original_savings : ℚ) (tv_cost : ℚ) : 
  original_savings = 1000 →
  tv_cost = 400 →
  (original_savings - tv_cost) / original_savings = 3/5 := by
sorry

end NUMINAMATH_CALUDE_linda_furniture_fraction_l2637_263745


namespace NUMINAMATH_CALUDE_exists_silver_division_l2637_263746

/-- Represents the relationship between the number of people and the amount of silver in the problem. -/
def silver_division (x y : ℕ) : Prop :=
  (6 * x - 6 = y) ∧ (5 * x + 5 = y)

/-- The theorem states that the silver_division relationship holds for some positive integers x and y. -/
theorem exists_silver_division : ∃ (x y : ℕ), x > 0 ∧ y > 0 ∧ silver_division x y := by
  sorry

end NUMINAMATH_CALUDE_exists_silver_division_l2637_263746


namespace NUMINAMATH_CALUDE_false_statement_l2637_263751

-- Define the types for planes and lines
variable {α β : Plane} {m n : Line}

-- Define the relations
def perpendicular (l : Line) (p : Plane) : Prop := sorry
def parallel (p q : Plane) : Prop := sorry
def parallel_line_plane (l : Line) (p : Plane) : Prop := sorry
def parallel_lines (l1 l2 : Line) : Prop := sorry
def contained_in (l : Line) (p : Plane) : Prop := sorry

-- State the theorem
theorem false_statement :
  ¬(∀ (α β : Plane) (m n : Line),
    (¬parallel_line_plane m α ∧ parallel α β ∧ contained_in n β) →
    parallel_lines m n) :=
sorry

end NUMINAMATH_CALUDE_false_statement_l2637_263751


namespace NUMINAMATH_CALUDE_runt_pig_revenue_l2637_263757

/-- Calculates the revenue from selling bacon from a pig -/
def bacon_revenue (average_yield : ℝ) (price_per_pound : ℝ) (size_ratio : ℝ) : ℝ :=
  average_yield * size_ratio * price_per_pound

/-- Proves that the farmer will make $60 from the runt pig's bacon -/
theorem runt_pig_revenue :
  let average_yield : ℝ := 20
  let price_per_pound : ℝ := 6
  let size_ratio : ℝ := 0.5
  bacon_revenue average_yield price_per_pound size_ratio = 60 := by
sorry

end NUMINAMATH_CALUDE_runt_pig_revenue_l2637_263757


namespace NUMINAMATH_CALUDE_solution_set_equivalence_l2637_263708

/-- Given that the solution set of (ax+1)/(x+b) > 1 is (-∞, -1) ∪ (3, +∞),
    prove that the solution set of x^2 + ax - 2b < 0 is (-3, -2) -/
theorem solution_set_equivalence (a b : ℝ) :
  ({x : ℝ | (a * x + 1) / (x + b) > 1} = {x : ℝ | x < -1 ∨ x > 3}) →
  {x : ℝ | x^2 + a*x - 2*b < 0} = {x : ℝ | -3 < x ∧ x < -2} :=
by sorry

end NUMINAMATH_CALUDE_solution_set_equivalence_l2637_263708


namespace NUMINAMATH_CALUDE_sum_product_equality_l2637_263783

theorem sum_product_equality (x y z : ℝ) 
  (hx : |x| ≠ 1/Real.sqrt 3) 
  (hy : |y| ≠ 1/Real.sqrt 3) 
  (hz : |z| ≠ 1/Real.sqrt 3) 
  (h : x + y + z = x * y * z) : 
  (3*x - x^3)/(1-3*x^2) + (3*y - y^3)/(1-3*y^2) + (3*z - z^3)/(1-3*z^2) = 
  (3*x - x^3)/(1-3*x^2) * (3*y - y^3)/(1-3*y^2) * (3*z - z^3)/(1-3*z^2) := by
  sorry

end NUMINAMATH_CALUDE_sum_product_equality_l2637_263783


namespace NUMINAMATH_CALUDE_correlation_coefficient_comparison_l2637_263727

def X : List Float := [10, 11.3, 11.8, 12.5, 13]
def Y : List Float := [1, 2, 3, 4, 5]
def U : List Float := [10, 11.3, 11.8, 12.5, 13]
def V : List Float := [5, 4, 3, 2, 1]

def linear_correlation_coefficient (x : List Float) (y : List Float) : Float :=
  sorry

def r₁ : Float := linear_correlation_coefficient X Y
def r₂ : Float := linear_correlation_coefficient U V

theorem correlation_coefficient_comparison : r₂ < r₁ := by
  sorry

end NUMINAMATH_CALUDE_correlation_coefficient_comparison_l2637_263727


namespace NUMINAMATH_CALUDE_reciprocal_root_property_l2637_263742

theorem reciprocal_root_property (c : ℝ) : 
  c^3 - c + 1 = 0 → (1/c)^5 + (1/c) + 1 = 0 := by
sorry

end NUMINAMATH_CALUDE_reciprocal_root_property_l2637_263742


namespace NUMINAMATH_CALUDE_probability_expired_20_2_l2637_263798

/-- The probability of selecting an expired bottle from a set of bottles -/
def probability_expired (total : ℕ) (expired : ℕ) : ℚ :=
  (expired : ℚ) / (total : ℚ)

/-- Theorem: The probability of selecting an expired bottle from 20 bottles, where 2 are expired, is 1/10 -/
theorem probability_expired_20_2 :
  probability_expired 20 2 = 1 / 10 := by
  sorry

end NUMINAMATH_CALUDE_probability_expired_20_2_l2637_263798


namespace NUMINAMATH_CALUDE_complex_expression_simplification_l2637_263758

theorem complex_expression_simplification :
  (-3 : ℂ) + 7 * Complex.I - 3 * (2 - 5 * Complex.I) + 4 * Complex.I = -9 + 26 * Complex.I :=
by sorry

end NUMINAMATH_CALUDE_complex_expression_simplification_l2637_263758


namespace NUMINAMATH_CALUDE_spade_example_l2637_263763

-- Define the spade operation
def spade (a b : ℝ) : ℝ := |a - b|

-- Theorem statement
theorem spade_example : spade 3 (spade 5 8) = 0 := by
  sorry

end NUMINAMATH_CALUDE_spade_example_l2637_263763


namespace NUMINAMATH_CALUDE_cubic_derivative_value_l2637_263722

def f (x : ℝ) := x^3

theorem cubic_derivative_value (x₀ : ℝ) :
  (deriv f) x₀ = 3 → x₀ = 1 ∨ x₀ = -1 := by sorry

end NUMINAMATH_CALUDE_cubic_derivative_value_l2637_263722


namespace NUMINAMATH_CALUDE_smallest_positive_angle_with_same_terminal_side_l2637_263724

theorem smallest_positive_angle_with_same_terminal_side (angle : Real) : 
  angle = -660 * Real.pi / 180 → 
  ∃ (k : ℤ), (angle + 2 * Real.pi * k) % (2 * Real.pi) = Real.pi / 3 ∧ 
  ∀ (x : Real), 0 < x ∧ x < Real.pi / 3 → 
  ¬∃ (m : ℤ), (angle + 2 * Real.pi * m) % (2 * Real.pi) = x :=
by sorry

end NUMINAMATH_CALUDE_smallest_positive_angle_with_same_terminal_side_l2637_263724


namespace NUMINAMATH_CALUDE_inequality_solution_l2637_263787

theorem inequality_solution (x : ℝ) : 1 - 1 / (3 * x + 4) < 3 ↔ x < -5/3 ∨ x > -4/3 := by
  sorry

end NUMINAMATH_CALUDE_inequality_solution_l2637_263787
