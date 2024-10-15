import Mathlib

namespace NUMINAMATH_CALUDE_math_club_members_l1308_130886

/-- 
Given a Math club where:
- There are two times as many males as females
- There are 6 female members
Prove that the total number of members in the Math club is 18.
-/
theorem math_club_members :
  ∀ (female_members male_members total_members : ℕ),
    female_members = 6 →
    male_members = 2 * female_members →
    total_members = female_members + male_members →
    total_members = 18 := by
  sorry

end NUMINAMATH_CALUDE_math_club_members_l1308_130886


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l1308_130811

theorem arithmetic_sequence_sum (a₁ a₂ a₃ a₆ : ℕ) (h₁ : a₁ = 5) (h₂ : a₂ = 12) (h₃ : a₃ = 19) (h₆ : a₆ = 40) :
  let d := a₂ - a₁
  let a₄ := a₃ + d
  let a₅ := a₄ + d
  a₄ + a₅ = 59 := by sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l1308_130811


namespace NUMINAMATH_CALUDE_grape_bowls_problem_l1308_130862

theorem grape_bowls_problem (n : ℕ) : 
  (8 * 12 = 6 * n) → n = 16 := by
  sorry

end NUMINAMATH_CALUDE_grape_bowls_problem_l1308_130862


namespace NUMINAMATH_CALUDE_geometric_sequence_ratio_sum_l1308_130890

theorem geometric_sequence_ratio_sum (m a₂ a₃ b₂ b₃ x y : ℝ) 
  (hm : m ≠ 0)
  (hx : x ≠ 1)
  (hy : y ≠ 1)
  (hxy : x ≠ y)
  (ha₂ : a₂ = m * x)
  (ha₃ : a₃ = m * x^2)
  (hb₂ : b₂ = m * y)
  (hb₃ : b₃ = m * y^2)
  (heq : a₃ - b₃ = 3 * (a₂ - b₂)) :
  x + y = 3 := by sorry

end NUMINAMATH_CALUDE_geometric_sequence_ratio_sum_l1308_130890


namespace NUMINAMATH_CALUDE_product_of_digits_not_divisible_by_4_l1308_130883

def numbers : List Nat := [4624, 4634, 4644, 4652, 4672]

def is_divisible_by_4 (n : Nat) : Bool :=
  n % 4 = 0

def units_digit (n : Nat) : Nat :=
  n % 10

def tens_digit (n : Nat) : Nat :=
  (n / 10) % 10

theorem product_of_digits_not_divisible_by_4 :
  ∃ n ∈ numbers, ¬is_divisible_by_4 n ∧ units_digit n * tens_digit n = 12 := by
  sorry

end NUMINAMATH_CALUDE_product_of_digits_not_divisible_by_4_l1308_130883


namespace NUMINAMATH_CALUDE_class_division_l1308_130844

theorem class_division (total_students : ℕ) (x : ℕ) : 
  (total_students = 8 * x + 2) ∧ (total_students = 9 * x - 4) → x = 6 := by
sorry

end NUMINAMATH_CALUDE_class_division_l1308_130844


namespace NUMINAMATH_CALUDE_classmate_reading_comprehensive_only_classmate_reading_comprehensive_l1308_130819

/-- Represents a survey activity -/
inductive SurveyActivity
| SocketLifespan
| TreePlantingSurvival
| ClassmateReading
| DocumentaryViewership

/-- Determines if a survey activity is suitable for a comprehensive survey -/
def isSuitableForComprehensiveSurvey (activity : SurveyActivity) : Prop :=
  match activity with
  | SurveyActivity.ClassmateReading => true
  | _ => false

/-- Theorem stating that the classmate reading survey is suitable for a comprehensive survey -/
theorem classmate_reading_comprehensive :
  isSuitableForComprehensiveSurvey SurveyActivity.ClassmateReading :=
by sorry

/-- Theorem stating that the classmate reading survey is the only suitable activity for a comprehensive survey -/
theorem only_classmate_reading_comprehensive (activity : SurveyActivity) :
  isSuitableForComprehensiveSurvey activity ↔ activity = SurveyActivity.ClassmateReading :=
by sorry

end NUMINAMATH_CALUDE_classmate_reading_comprehensive_only_classmate_reading_comprehensive_l1308_130819


namespace NUMINAMATH_CALUDE_expected_final_set_size_l1308_130857

/-- The set of elements Marisa is working with -/
def S : Finset Nat := Finset.range 8

/-- The initial number of subsets in Marisa's collection -/
def initial_subsets : Nat := 2^8 - 1

/-- The number of steps in Marisa's process -/
def num_steps : Nat := 2^8 - 2

/-- The probability of an element being in a randomly chosen subset -/
def prob_in_subset : ℚ := 128 / 255

/-- The expected size of the final set in Marisa's subset collection process -/
theorem expected_final_set_size :
  (S.card : ℚ) * prob_in_subset = 1024 / 255 := by sorry

end NUMINAMATH_CALUDE_expected_final_set_size_l1308_130857


namespace NUMINAMATH_CALUDE_composition_result_l1308_130882

-- Define the functions f and g
def f (c : ℝ) (x : ℝ) : ℝ := 4 * x + c
def g (c : ℝ) (x : ℝ) : ℝ := c * x + 2

-- State the theorem
theorem composition_result (c d : ℝ) :
  (∀ x, f c (g c x) = 12 * x + d) → d = 11 :=
by
  sorry

end NUMINAMATH_CALUDE_composition_result_l1308_130882


namespace NUMINAMATH_CALUDE_fractional_equation_solution_l1308_130807

theorem fractional_equation_solution (m : ℝ) : 
  (∀ x : ℝ, x ≠ 2 → ((x + m) / (x - 2) + 1 / (2 - x) = 3)) →
  ((2 + m) / (2 - 2) + 1 / (2 - 2) = 3) →
  m = -1 := by
sorry

end NUMINAMATH_CALUDE_fractional_equation_solution_l1308_130807


namespace NUMINAMATH_CALUDE_seans_fraction_of_fritz_money_l1308_130867

theorem seans_fraction_of_fritz_money (fritz_money sean_money rick_money : ℚ) 
  (x : ℚ) : 
  fritz_money = 40 →
  sean_money = x * fritz_money + 4 →
  rick_money = 3 * sean_money →
  rick_money + sean_money = 96 →
  x = 1/2 := by
sorry

end NUMINAMATH_CALUDE_seans_fraction_of_fritz_money_l1308_130867


namespace NUMINAMATH_CALUDE_geometric_sequence_sum_l1308_130815

/-- A geometric sequence -/
def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, ∀ n : ℕ, a (n + 1) = a n * q

/-- The theorem statement -/
theorem geometric_sequence_sum (a : ℕ → ℝ) :
  geometric_sequence a →
  a 4 * a 5 * a 6 = 8 →
  a 2 = 1 →
  a 2 + a 5 + a 8 + a 11 = 15 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_sum_l1308_130815


namespace NUMINAMATH_CALUDE_max_m_proof_l1308_130885

/-- The maximum value of m given the condition -/
def max_m : ℝ := -2

/-- The condition function -/
def condition (x : ℝ) : Prop := x^2 - 2*x - 8 > 0

/-- The main theorem -/
theorem max_m_proof :
  (∀ x : ℝ, x < max_m → condition x) ∧
  (∃ x : ℝ, x < max_m ∧ ¬condition x) ∧
  (∀ m : ℝ, m > max_m → ∃ x : ℝ, x < m ∧ ¬condition x) :=
sorry

end NUMINAMATH_CALUDE_max_m_proof_l1308_130885


namespace NUMINAMATH_CALUDE_student_count_l1308_130865

theorem student_count (initial_avg : ℚ) (wrong_mark : ℚ) (correct_mark : ℚ) (final_avg : ℚ) :
  initial_avg = 100 →
  wrong_mark = 70 →
  correct_mark = 10 →
  final_avg = 98 →
  ∃ n : ℕ, n > 0 ∧ n * final_avg = n * initial_avg - (wrong_mark - correct_mark) :=
by
  sorry

end NUMINAMATH_CALUDE_student_count_l1308_130865


namespace NUMINAMATH_CALUDE_f_increasing_when_x_gt_1_l1308_130896

-- Define the function
def f (x : ℝ) : ℝ := 2 * x^2

-- State the theorem
theorem f_increasing_when_x_gt_1 :
  ∀ x₁ x₂ : ℝ, 1 < x₁ → x₁ < x₂ → f x₁ < f x₂ :=
by
  sorry

end NUMINAMATH_CALUDE_f_increasing_when_x_gt_1_l1308_130896


namespace NUMINAMATH_CALUDE_circle_equations_l1308_130806

-- Define the parallel lines
def line1 (a : ℝ) (x y : ℝ) : Prop := (a - 2) * x + y + Real.sqrt 2 = 0
def line2 (a : ℝ) (x y : ℝ) : Prop := a * x + 3 * y + 2 * Real.sqrt 2 * a = 0

-- Define the circle N
def circleN (x y : ℝ) : Prop := (x - 3)^2 + (y - 4)^2 = 9

-- Define point B
def pointB : ℝ × ℝ := (3, -2)

-- Define the line of symmetry
def lineSymmetry (x : ℝ) : Prop := x = -1

-- Define point C
def pointC : ℝ × ℝ := (-5, -2)

-- Define the circle C
def circleC (x y : ℝ) : Prop := (x + 5)^2 + (y + 2)^2 = 49

-- Theorem statement
theorem circle_equations :
  ∃ (a : ℝ),
    (∀ x y, line1 a x y ↔ line2 a x y) →
    (∀ x y, circleN x y) →
    (pointC.1 = -pointB.1 - 2 ∧ pointC.2 = pointB.2) →
    (∀ x y, circleC x y) ∧
    (∃ x y, circleN x y ∧ circleC x y ∧
      (x - 3)^2 + (y - 4)^2 + ((x + 5)^2 + (y + 2)^2).sqrt = 10) :=
by sorry

end NUMINAMATH_CALUDE_circle_equations_l1308_130806


namespace NUMINAMATH_CALUDE_vacation_duration_l1308_130877

/-- Represents the vacation of a family -/
structure Vacation where
  total_days : ℕ
  rain_days : ℕ
  clear_afternoons : ℕ

/-- Theorem stating that given the conditions, the total number of days is 18 -/
theorem vacation_duration (v : Vacation) 
  (h1 : v.rain_days = 13)
  (h2 : v.clear_afternoons = 12)
  (h3 : v.rain_days + v.clear_afternoons ≤ v.total_days)
  (h4 : v.total_days ≤ v.rain_days + v.clear_afternoons + 1) :
  v.total_days = 18 := by
  sorry

#check vacation_duration

end NUMINAMATH_CALUDE_vacation_duration_l1308_130877


namespace NUMINAMATH_CALUDE_inequality_solution_range_l1308_130853

theorem inequality_solution_range (a : ℝ) : 
  (∀ x : ℝ, x < 1 ↔ 2*a*x + 3*x > 2*a + 3) → a < -3/2 :=
by sorry

end NUMINAMATH_CALUDE_inequality_solution_range_l1308_130853


namespace NUMINAMATH_CALUDE_smaller_paintings_count_l1308_130868

/-- Represents a museum with paintings and artifacts --/
structure Museum where
  total_wings : ℕ
  painting_wings : ℕ
  artifacts_per_wing : ℕ
  artifact_painting_ratio : ℕ

/-- The number of smaller paintings in each of the two wings --/
def smaller_paintings_per_wing (m : Museum) : ℕ :=
  ((m.artifacts_per_wing * (m.total_wings - m.painting_wings)) / m.artifact_painting_ratio - 1) / 2

/-- Theorem stating the number of smaller paintings per wing --/
theorem smaller_paintings_count (m : Museum) 
  (h1 : m.total_wings = 8)
  (h2 : m.painting_wings = 3)
  (h3 : m.artifacts_per_wing = 20)
  (h4 : m.artifact_painting_ratio = 4) :
  smaller_paintings_per_wing m = 12 := by
  sorry

end NUMINAMATH_CALUDE_smaller_paintings_count_l1308_130868


namespace NUMINAMATH_CALUDE_semicircle_bounded_rectangle_perimeter_l1308_130871

theorem semicircle_bounded_rectangle_perimeter :
  let rectangle_length : ℝ := 4 / π
  let rectangle_width : ℝ := 1 / π
  let long_side_arcs_perimeter : ℝ := 2 * π * rectangle_length / 2
  let short_side_arcs_perimeter : ℝ := π * rectangle_width
  long_side_arcs_perimeter + short_side_arcs_perimeter = 9 := by
  sorry

end NUMINAMATH_CALUDE_semicircle_bounded_rectangle_perimeter_l1308_130871


namespace NUMINAMATH_CALUDE_no_integer_solutions_l1308_130802

theorem no_integer_solutions (n : ℤ) (s : ℕ) (h_s : Odd s) :
  ¬ ∃ x : ℤ, x^2 - 16*n*x + 7^s = 0 :=
by sorry

end NUMINAMATH_CALUDE_no_integer_solutions_l1308_130802


namespace NUMINAMATH_CALUDE_closest_point_l1308_130889

def w (s : ℝ) : Fin 3 → ℝ := fun i =>
  match i with
  | 0 => 3 + 8*s
  | 1 => -2 + 6*s
  | 2 => -4 - 2*s

def b : Fin 3 → ℝ := fun i =>
  match i with
  | 0 => 5
  | 1 => 5
  | 2 => 6

def direction : Fin 3 → ℝ := fun i =>
  match i with
  | 0 => 8
  | 1 => 6
  | 2 => -2

theorem closest_point (s : ℝ) : 
  (s = 19/52) ↔ 
  (∀ t : ℝ, ‖w s - b‖ ≤ ‖w t - b‖) :=
sorry

end NUMINAMATH_CALUDE_closest_point_l1308_130889


namespace NUMINAMATH_CALUDE_quadratic_minimum_l1308_130824

/-- A quadratic function f(x) = x^2 + px + q -/
def f (p q x : ℝ) : ℝ := x^2 + p*x + q

theorem quadratic_minimum (p q : ℝ) :
  (∀ x, f p q x ≥ f p q q) ∧ 
  (f p q q = (p + q)^2) →
  ((p = 0 ∧ q = 0) ∨ (p = -1 ∧ q = 1/2)) := by
sorry

end NUMINAMATH_CALUDE_quadratic_minimum_l1308_130824


namespace NUMINAMATH_CALUDE_b_sixth_mod_n_l1308_130897

theorem b_sixth_mod_n (n : ℕ+) (b : ℤ) (h : b^3 ≡ 1 [ZMOD n]) :
  b^6 ≡ 1 [ZMOD n] := by
  sorry

end NUMINAMATH_CALUDE_b_sixth_mod_n_l1308_130897


namespace NUMINAMATH_CALUDE_fourth_rectangle_area_l1308_130852

/-- Represents a rectangle with given dimensions -/
structure Rectangle where
  length : ℝ
  width : ℝ

/-- Represents a large rectangle divided into four smaller rectangles -/
structure DividedRectangle where
  large : Rectangle
  small1 : Rectangle
  small2 : Rectangle
  small3 : Rectangle
  small4 : Rectangle

/-- The area of a rectangle -/
def area (r : Rectangle) : ℝ := r.length * r.width

/-- Theorem: If a rectangle is divided into four smaller rectangles, and three of them have
    areas 20, 12, and 16, then the fourth rectangle has an area of 16 -/
theorem fourth_rectangle_area
  (dr : DividedRectangle)
  (h1 : area dr.small1 = 20)
  (h2 : area dr.small2 = 12)
  (h3 : area dr.small3 = 16)
  (h_sum : area dr.large = area dr.small1 + area dr.small2 + area dr.small3 + area dr.small4)
  : area dr.small4 = 16 := by
  sorry

end NUMINAMATH_CALUDE_fourth_rectangle_area_l1308_130852


namespace NUMINAMATH_CALUDE_hyperbola_focal_length_l1308_130898

-- Define the hyperbola parameters
def hyperbola_equation (x y a : ℝ) : Prop := x^2 / a^2 - y^2 / 20 = 1

-- Define the asymptote equation
def asymptote_equation (x y : ℝ) : Prop := y = 2 * x

-- Define the focal length calculation
def focal_length (a : ℝ) : ℝ := 2 * a

-- Theorem statement
theorem hyperbola_focal_length (a : ℝ) (h1 : a > 0) 
  (h2 : ∀ x y : ℝ, hyperbola_equation x y a → asymptote_equation x y) :
  focal_length a = 10 := by
  sorry

end NUMINAMATH_CALUDE_hyperbola_focal_length_l1308_130898


namespace NUMINAMATH_CALUDE_square_tiles_count_l1308_130872

theorem square_tiles_count (total_tiles : ℕ) (total_edges : ℕ) 
  (h_total_tiles : total_tiles = 35)
  (h_total_edges : total_edges = 140) :
  ∃ (t s p : ℕ),
    t + s + p = total_tiles ∧
    3 * t + 4 * s + 5 * p = total_edges ∧
    s = 35 := by
  sorry

end NUMINAMATH_CALUDE_square_tiles_count_l1308_130872


namespace NUMINAMATH_CALUDE_point_order_l1308_130810

def f (x : ℝ) : ℝ := 3 * x^2 + 6 * x + 12

theorem point_order (y₁ y₂ y₃ : ℝ) 
  (h₁ : f (-1) = y₁)
  (h₂ : f (-3) = y₂)
  (h₃ : f 2 = y₃) :
  y₃ > y₂ ∧ y₂ > y₁ := by
  sorry

end NUMINAMATH_CALUDE_point_order_l1308_130810


namespace NUMINAMATH_CALUDE_product_comparison_l1308_130823

theorem product_comparison (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  (1.1 * a) * (1.13 * b) * (0.8 * c) < a * b * c := by
  sorry

end NUMINAMATH_CALUDE_product_comparison_l1308_130823


namespace NUMINAMATH_CALUDE_jake_and_kendra_weight_l1308_130832

/-- Calculates the combined weight of Jake and Kendra given Jake's current weight and the condition about their weight relation after Jake loses 8 pounds. -/
def combinedWeight (jakeWeight : ℕ) : ℕ :=
  let kendraWeight := (jakeWeight - 8) / 2
  jakeWeight + kendraWeight

/-- Theorem stating that given Jake's current weight of 196 pounds and the condition about their weight relation, the combined weight of Jake and Kendra is 290 pounds. -/
theorem jake_and_kendra_weight : combinedWeight 196 = 290 := by
  sorry

#eval combinedWeight 196

end NUMINAMATH_CALUDE_jake_and_kendra_weight_l1308_130832


namespace NUMINAMATH_CALUDE_marcus_cookies_count_l1308_130846

/-- The number of peanut butter cookies Marcus brought to the bake sale -/
def marcus_peanut_butter_cookies : ℕ := 30

/-- The number of peanut butter cookies Jenny brought to the bake sale -/
def jenny_peanut_butter_cookies : ℕ := 40

/-- The total number of non-peanut butter cookies at the bake sale -/
def total_non_peanut_butter_cookies : ℕ := 70

/-- The probability of picking a peanut butter cookie -/
def peanut_butter_probability : ℚ := 1/2

theorem marcus_cookies_count :
  marcus_peanut_butter_cookies = 30 ∧
  jenny_peanut_butter_cookies + marcus_peanut_butter_cookies = total_non_peanut_butter_cookies ∧
  (jenny_peanut_butter_cookies + marcus_peanut_butter_cookies : ℚ) /
    (jenny_peanut_butter_cookies + marcus_peanut_butter_cookies + total_non_peanut_butter_cookies) = peanut_butter_probability :=
by sorry

end NUMINAMATH_CALUDE_marcus_cookies_count_l1308_130846


namespace NUMINAMATH_CALUDE_team_average_goals_is_seven_l1308_130892

/-- The average number of goals scored by a soccer team per game -/
def team_average_goals (carter_goals shelby_goals judah_goals : ℝ) : ℝ :=
  carter_goals + shelby_goals + judah_goals

/-- Theorem: Given the conditions, the team's average goals per game is 7 -/
theorem team_average_goals_is_seven :
  ∀ (carter_goals shelby_goals judah_goals : ℝ),
    carter_goals = 4 →
    shelby_goals = carter_goals / 2 →
    judah_goals = 2 * shelby_goals - 3 →
    team_average_goals carter_goals shelby_goals judah_goals = 7 := by
  sorry

end NUMINAMATH_CALUDE_team_average_goals_is_seven_l1308_130892


namespace NUMINAMATH_CALUDE_square_area_from_diagonal_l1308_130820

theorem square_area_from_diagonal (diagonal : ℝ) (area : ℝ) : 
  diagonal = 12 → area = diagonal^2 / 2 → area = 72 := by
  sorry

#check square_area_from_diagonal

end NUMINAMATH_CALUDE_square_area_from_diagonal_l1308_130820


namespace NUMINAMATH_CALUDE_pear_sales_l1308_130812

theorem pear_sales (morning_sales afternoon_sales : ℕ) : 
  afternoon_sales = 2 * morning_sales →
  morning_sales + afternoon_sales = 360 →
  afternoon_sales = 240 := by
sorry

end NUMINAMATH_CALUDE_pear_sales_l1308_130812


namespace NUMINAMATH_CALUDE_curtain_price_is_30_l1308_130873

/-- The cost of Emily's order, given the number of curtain pairs, wall prints, their prices, and installation cost. -/
def order_cost (curtain_pairs : ℕ) (curtain_price : ℝ) (wall_prints : ℕ) (print_price : ℝ) (installation : ℝ) : ℝ :=
  curtain_pairs * curtain_price + wall_prints * print_price + installation

/-- Theorem stating that the cost of each pair of curtains is $30.00 -/
theorem curtain_price_is_30 :
  ∃ (curtain_price : ℝ),
    order_cost 2 curtain_price 9 15 50 = 245 ∧ curtain_price = 30 := by
  sorry

#check curtain_price_is_30

end NUMINAMATH_CALUDE_curtain_price_is_30_l1308_130873


namespace NUMINAMATH_CALUDE_propositions_truth_l1308_130828

theorem propositions_truth : 
  (∀ a b : ℝ, a > 1 → b > 1 → a * b > 1) ∧ 
  (∃ a b c : ℝ, b = Real.sqrt (a * c) ∧ ¬(∃ r : ℝ, b = a * r ∧ c = b * r)) ∧
  (∃ a b c : ℝ, (∃ r : ℝ, b = a * r ∧ c = b * r) ∧ b ≠ Real.sqrt (a * c)) :=
by sorry


end NUMINAMATH_CALUDE_propositions_truth_l1308_130828


namespace NUMINAMATH_CALUDE_emerald_woods_circuit_length_l1308_130851

/-- Proves that the total length of the Emerald Woods Circuit is 43 miles given the hiking conditions --/
theorem emerald_woods_circuit_length :
  ∀ (a b c d e : ℝ),
    a + b + c = 28 →
    c + d = 24 →
    c + d + e = 39 →
    a + d = 30 →
    a + b + c + d + e = 43 := by
  sorry

end NUMINAMATH_CALUDE_emerald_woods_circuit_length_l1308_130851


namespace NUMINAMATH_CALUDE_circle_radius_equality_l1308_130859

theorem circle_radius_equality (r₁ r₂ : ℝ) (h₁ : r₁ = 37) (h₂ : r₂ = 23) :
  ∃ r : ℝ, r^2 = (r₁^2 - r₂^2) ∧ r = 2 * Real.sqrt 210 := by
  sorry

end NUMINAMATH_CALUDE_circle_radius_equality_l1308_130859


namespace NUMINAMATH_CALUDE_transaction_yearly_loss_l1308_130887

/-- Calculates the simple interest for a given principal, rate, and time -/
def simpleInterest (principal : ℚ) (rate : ℚ) (time : ℚ) : ℚ :=
  (principal * rate * time) / 100

/-- Represents the financial transaction described in the problem -/
structure FinancialTransaction where
  borrowAmount : ℚ
  borrowRate : ℚ
  lendRate : ℚ
  timeInYears : ℚ

/-- Calculates the yearly loss in the given financial transaction -/
def yearlyLoss (transaction : FinancialTransaction) : ℚ :=
  let borrowInterest := simpleInterest transaction.borrowAmount transaction.borrowRate transaction.timeInYears
  let lendInterest := simpleInterest transaction.borrowAmount transaction.lendRate transaction.timeInYears
  (borrowInterest - lendInterest) / transaction.timeInYears

/-- Theorem stating that the yearly loss in the given transaction is 140 -/
theorem transaction_yearly_loss :
  let transaction : FinancialTransaction := {
    borrowAmount := 7000
    borrowRate := 4
    lendRate := 6
    timeInYears := 2
  }
  yearlyLoss transaction = 140 := by sorry

end NUMINAMATH_CALUDE_transaction_yearly_loss_l1308_130887


namespace NUMINAMATH_CALUDE_recruit_line_total_l1308_130874

/-- Represents the position of a person in the line of recruits -/
structure Position where
  front : Nat
  behind : Nat

/-- The line of recruits -/
structure RecruitLine where
  peter : Position
  nikolai : Position
  denis : Position
  total : Nat

/-- The conditions of the problem -/
def initial_conditions : RecruitLine := {
  peter := { front := 50, behind := 0 },
  nikolai := { front := 100, behind := 0 },
  denis := { front := 170, behind := 0 },
  total := 0
}

/-- The condition after turning around -/
def turn_around_condition (line : RecruitLine) : Prop :=
  (line.peter.behind = 50 ∧ line.nikolai.behind = 100 ∧ line.denis.behind = 170) ∧
  ((4 * line.peter.front = line.nikolai.front ∧ line.peter.behind = 4 * line.nikolai.behind) ∨
   (4 * line.nikolai.front = line.denis.front ∧ line.nikolai.behind = 4 * line.denis.behind) ∨
   (4 * line.peter.front = line.denis.front ∧ line.peter.behind = 4 * line.denis.behind))

/-- The theorem to prove -/
theorem recruit_line_total (line : RecruitLine) :
  turn_around_condition line →
  line.total = 211 :=
by sorry

end NUMINAMATH_CALUDE_recruit_line_total_l1308_130874


namespace NUMINAMATH_CALUDE_cauliflower_earnings_l1308_130803

/-- Earnings from farmers' market --/
structure MarketEarnings where
  total : ℕ
  broccoli : ℕ
  carrots : ℕ
  spinach : ℕ
  cauliflower : ℕ

/-- Conditions for the farmers' market earnings --/
def validMarketEarnings (e : MarketEarnings) : Prop :=
  e.total = 380 ∧
  e.broccoli = 57 ∧
  e.carrots = 2 * e.broccoli ∧
  e.spinach = (e.carrots / 2) + 16 ∧
  e.total = e.broccoli + e.carrots + e.spinach + e.cauliflower

theorem cauliflower_earnings (e : MarketEarnings) (h : validMarketEarnings e) :
  e.cauliflower = 136 := by
  sorry

end NUMINAMATH_CALUDE_cauliflower_earnings_l1308_130803


namespace NUMINAMATH_CALUDE_triangle_properties_l1308_130864

theorem triangle_properties (A B C : ℝ) (a b c : ℝ) :
  0 < A ∧ 0 < B ∧ 0 < C ∧
  A + B + C = π ∧
  0 < a ∧ 0 < b ∧ 0 < c ∧
  a / (Real.sin A) = b / (Real.sin B) ∧
  b / (Real.sin B) = c / (Real.sin C) ∧
  4 * a = Real.sqrt 5 * c ∧
  Real.cos C = 3 / 5 →
  Real.sin A = Real.sqrt 5 / 5 ∧
  (b = 11 → a = 5) ∧
  (b = 11 → Real.cos (2 * A + C) = -7 / 25) := by
sorry


end NUMINAMATH_CALUDE_triangle_properties_l1308_130864


namespace NUMINAMATH_CALUDE_union_of_M_and_N_l1308_130839

open Set

def M : Set ℝ := {x : ℝ | -1/2 < x ∧ x < 1/2}
def N : Set ℝ := {x : ℝ | x * (x - 1) ≤ 0}

theorem union_of_M_and_N : M ∪ N = Ioo (-1/2 : ℝ) 1 := by sorry

end NUMINAMATH_CALUDE_union_of_M_and_N_l1308_130839


namespace NUMINAMATH_CALUDE_problem_one_problem_two_l1308_130861

-- Problem 1
theorem problem_one : -9 + 5 - (-12) + (-3) = 5 := by
  sorry

-- Problem 2
theorem problem_two : -(1.5) - (-4.25) + 3.75 - 8.5 = -2 := by
  sorry

end NUMINAMATH_CALUDE_problem_one_problem_two_l1308_130861


namespace NUMINAMATH_CALUDE_downstream_distance_l1308_130855

/-- Calculates the distance traveled downstream by a boat -/
theorem downstream_distance
  (boat_speed : ℝ)
  (stream_speed : ℝ)
  (travel_time : ℝ)
  (h1 : boat_speed = 22)
  (h2 : stream_speed = 5)
  (h3 : travel_time = 8) :
  boat_speed + stream_speed * travel_time = 216 :=
by
  sorry

#check downstream_distance

end NUMINAMATH_CALUDE_downstream_distance_l1308_130855


namespace NUMINAMATH_CALUDE_linear_system_solution_l1308_130834

/-- Given a system of linear equations 2x + my = 5 and nx - 3y = 2,
    if the augmented matrix transforms to [[1, 0, 3], [0, 1, 1]],
    then m/n = -3/5 -/
theorem linear_system_solution (m n : ℚ) : 
  (∃ x y : ℚ, 2*x + m*y = 5 ∧ n*x - 3*y = 2) →
  (∃ x y : ℚ, x = 3 ∧ y = 1) →
  m/n = -3/5 := by
  sorry

end NUMINAMATH_CALUDE_linear_system_solution_l1308_130834


namespace NUMINAMATH_CALUDE_rational_a_condition_l1308_130822

theorem rational_a_condition (m n : ℤ) : 
  ∃ (a : ℚ), a = (m^4 + n^4 + m^2*n^2) / (4*m^2*n^2) :=
by sorry

end NUMINAMATH_CALUDE_rational_a_condition_l1308_130822


namespace NUMINAMATH_CALUDE_households_with_bike_only_l1308_130836

/-- Proves that the number of households with only a bike is 35 -/
theorem households_with_bike_only
  (total : ℕ)
  (neither : ℕ)
  (both : ℕ)
  (with_car : ℕ)
  (h_total : total = 90)
  (h_neither : neither = 11)
  (h_both : both = 16)
  (h_with_car : with_car = 44) :
  total - neither - (with_car - both) - both = 35 :=
by sorry

end NUMINAMATH_CALUDE_households_with_bike_only_l1308_130836


namespace NUMINAMATH_CALUDE_regular_triangular_pyramid_volume_l1308_130842

/-- A regular triangular pyramid -/
structure RegularTriangularPyramid where
  /-- The dihedral angle between two adjacent faces -/
  α : Real
  /-- The distance from the center of the base to an edge of the lateral face -/
  d : Real

/-- The volume of a regular triangular pyramid -/
noncomputable def volume (p : RegularTriangularPyramid) : Real :=
  (9 * Real.tan p.α ^ 3) / (4 * Real.sqrt (3 * Real.tan p.α ^ 2 - 1))

theorem regular_triangular_pyramid_volume 
  (p : RegularTriangularPyramid) 
  (h1 : p.d = 1) 
  : volume p = (9 * Real.tan p.α ^ 3) / (4 * Real.sqrt (3 * Real.tan p.α ^ 2 - 1)) := by
  sorry

end NUMINAMATH_CALUDE_regular_triangular_pyramid_volume_l1308_130842


namespace NUMINAMATH_CALUDE_tangent_to_both_circumcircles_l1308_130804

-- Define the basic structures
structure Point := (x y : ℝ)

structure Line := (a b : Point)

structure Circle := (center : Point) (radius : ℝ)

-- Define the parallelogram
def Parallelogram (A B C D : Point) : Prop := sorry

-- Define a point between two other points
def PointBetween (E B F : Point) : Prop := sorry

-- Define the intersection of two lines
def Intersect (l₁ l₂ : Line) (O : Point) : Prop := sorry

-- Define a line tangent to a circle
def Tangent (l : Line) (c : Circle) : Prop := sorry

-- Define the circumcircle of a triangle
def Circumcircle (A B C : Point) : Circle := sorry

-- Main theorem
theorem tangent_to_both_circumcircles 
  (A B C D E F O : Point) 
  (h1 : Parallelogram A B C D)
  (h2 : PointBetween E B F)
  (h3 : Intersect (Line.mk A C) (Line.mk B D) O)
  (h4 : Tangent (Line.mk A E) (Circumcircle A O D))
  (h5 : Tangent (Line.mk D F) (Circumcircle A O D)) :
  Tangent (Line.mk A E) (Circumcircle E O F) ∧ 
  Tangent (Line.mk D F) (Circumcircle E O F) := by sorry

end NUMINAMATH_CALUDE_tangent_to_both_circumcircles_l1308_130804


namespace NUMINAMATH_CALUDE_film_product_unique_l1308_130854

/-- Represents the alphabet-to-number mapping -/
def letter_value (c : Char) : Nat :=
  match c with
  | 'A' => 1 | 'B' => 2 | 'C' => 3 | 'D' => 4 | 'E' => 5 | 'F' => 6 | 'G' => 7
  | 'H' => 8 | 'I' => 9 | 'J' => 10 | 'K' => 11 | 'L' => 12 | 'M' => 13 | 'N' => 14
  | 'O' => 15 | 'P' => 16 | 'Q' => 17 | 'R' => 18 | 'S' => 19 | 'T' => 20
  | 'U' => 21 | 'V' => 22 | 'W' => 23 | 'X' => 24 | 'Y' => 25 | 'Z' => 26
  | _ => 0

/-- Calculates the product of letter values for a given string -/
def string_product (s : String) : Nat :=
  s.data.foldl (fun acc c => acc * letter_value c) 1

/-- Checks if a string is a valid four-letter combination (all uppercase letters) -/
def is_valid_combination (s : String) : Bool :=
  s.length = 4 && s.data.all (fun c => 'A' ≤ c && c ≤ 'Z')

/-- Theorem: The product of "FILM" is unique among all four-letter combinations -/
theorem film_product_unique :
  ∀ s : String, is_valid_combination s → s ≠ "FILM" →
  string_product s ≠ string_product "FILM" :=
sorry


end NUMINAMATH_CALUDE_film_product_unique_l1308_130854


namespace NUMINAMATH_CALUDE_player_A_performance_l1308_130837

/-- Represents a basketball player's shooting performance -/
structure Player where
  shotProbability : ℝ
  roundsTotal : ℕ
  shotsPerRound : ℕ

/-- Calculates the probability of passing a single round -/
def passProbability (p : Player) : ℝ :=
  1 - (1 - p.shotProbability) ^ p.shotsPerRound

/-- Calculates the expected number of passed rounds -/
def expectedPassedRounds (p : Player) : ℝ :=
  p.roundsTotal * passProbability p

/-- Theorem stating the probability of passing a round and expected passed rounds for player A -/
theorem player_A_performance : 
  let playerA : Player := { shotProbability := 0.6, roundsTotal := 5, shotsPerRound := 2 }
  passProbability playerA = 0.84 ∧ expectedPassedRounds playerA = 4.2 := by
  sorry


end NUMINAMATH_CALUDE_player_A_performance_l1308_130837


namespace NUMINAMATH_CALUDE_birds_in_tree_l1308_130838

theorem birds_in_tree (initial_birds new_birds : ℕ) 
  (h1 : initial_birds = 14) 
  (h2 : new_birds = 21) : 
  initial_birds + new_birds = 35 :=
by sorry

end NUMINAMATH_CALUDE_birds_in_tree_l1308_130838


namespace NUMINAMATH_CALUDE_right_triangle_consecutive_sides_l1308_130858

theorem right_triangle_consecutive_sides (a c : ℕ) (b : ℝ) : 
  c = a + 1 → -- c and a are consecutive integers
  a^2 + b^2 = c^2 → -- Pythagorean theorem
  b^2 = c + a := by
sorry

end NUMINAMATH_CALUDE_right_triangle_consecutive_sides_l1308_130858


namespace NUMINAMATH_CALUDE_joseph_cards_l1308_130833

/-- Calculates the total number of cards Joseph had initially -/
def total_cards (num_students : ℕ) (cards_per_student : ℕ) (cards_left : ℕ) : ℕ :=
  num_students * cards_per_student + cards_left

/-- Proves that Joseph had 357 cards initially -/
theorem joseph_cards : total_cards 15 23 12 = 357 := by
  sorry

end NUMINAMATH_CALUDE_joseph_cards_l1308_130833


namespace NUMINAMATH_CALUDE_bobs_family_adults_l1308_130895

/-- The number of adults in Bob's family -/
def num_adults (total_apples : ℕ) (num_children : ℕ) (apples_per_child : ℕ) (apples_per_adult : ℕ) : ℕ :=
  (total_apples - num_children * apples_per_child) / apples_per_adult

/-- Theorem stating that the number of adults in Bob's family is 40 -/
theorem bobs_family_adults :
  num_adults 450 33 10 3 = 40 := by
  sorry

#eval num_adults 450 33 10 3

end NUMINAMATH_CALUDE_bobs_family_adults_l1308_130895


namespace NUMINAMATH_CALUDE_f_sum_2016_2015_l1308_130808

def is_odd_function (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

def is_even_function (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = f x

theorem f_sum_2016_2015 (f : ℝ → ℝ) 
  (h_odd : is_odd_function f)
  (h_even : is_even_function (fun x ↦ f (x + 1)))
  (h_f_1 : f 1 = 1) :
  f 2016 + f 2015 = -1 := by
  sorry

end NUMINAMATH_CALUDE_f_sum_2016_2015_l1308_130808


namespace NUMINAMATH_CALUDE_rabbits_in_park_l1308_130843

theorem rabbits_in_park (cage_rabbits : ℕ) (park_rabbits : ℕ) : 
  cage_rabbits = 13 →
  cage_rabbits + 7 = park_rabbits / 3 →
  park_rabbits = 60 := by
  sorry

end NUMINAMATH_CALUDE_rabbits_in_park_l1308_130843


namespace NUMINAMATH_CALUDE_picture_frame_interior_edges_sum_l1308_130863

theorem picture_frame_interior_edges_sum 
  (frame_width : ℝ) 
  (frame_area : ℝ) 
  (outer_edge : ℝ) :
  frame_width = 2 →
  frame_area = 68 →
  outer_edge = 15 →
  ∃ (inner_width inner_height : ℝ),
    inner_width = outer_edge - 2 * frame_width ∧
    frame_area = outer_edge * (inner_height + 2 * frame_width) - inner_width * inner_height ∧
    2 * (inner_width + inner_height) = 26 :=
by sorry

end NUMINAMATH_CALUDE_picture_frame_interior_edges_sum_l1308_130863


namespace NUMINAMATH_CALUDE_carol_initial_cupcakes_l1308_130879

/-- Given that if Carol sold 9 cupcakes and made 28 more, she would have 49 cupcakes,
    prove that Carol initially made 30 cupcakes. -/
theorem carol_initial_cupcakes : 
  ∀ (initial : ℕ), 
  (initial - 9 + 28 = 49) → 
  initial = 30 := by
sorry

end NUMINAMATH_CALUDE_carol_initial_cupcakes_l1308_130879


namespace NUMINAMATH_CALUDE_divide_forty_five_by_point_zero_five_l1308_130827

theorem divide_forty_five_by_point_zero_five : 45 / 0.05 = 900 := by
  sorry

end NUMINAMATH_CALUDE_divide_forty_five_by_point_zero_five_l1308_130827


namespace NUMINAMATH_CALUDE_five_Z_three_equals_twelve_l1308_130801

-- Define the Z operation
def Z (a b : ℝ) : ℝ := 3 * (a - b)^2

-- Theorem statement
theorem five_Z_three_equals_twelve : Z 5 3 = 12 := by
  sorry

end NUMINAMATH_CALUDE_five_Z_three_equals_twelve_l1308_130801


namespace NUMINAMATH_CALUDE_system_solution_l1308_130825

-- Define the system of equations
def system (x : Fin 6 → ℚ) : Prop :=
  2 * x 0 + 2 * x 1 - x 2 + x 3 + 4 * x 5 = 0 ∧
  x 0 + 2 * x 1 + 2 * x 2 + 3 * x 4 + x 5 = -2 ∧
  x 0 - 2 * x 1 + x 3 + 2 * x 4 = 0

-- Define the solution
def solution (x : Fin 6 → ℚ) : Prop :=
  x 0 = -1/4 - 5/8 * x 3 - 9/8 * x 4 - 9/8 * x 5 ∧
  x 1 = -1/8 + 3/16 * x 3 - 7/16 * x 4 + 9/16 * x 5 ∧
  x 2 = -3/4 + 1/8 * x 3 - 11/8 * x 4 + 5/8 * x 5

-- Theorem statement
theorem system_solution :
  ∀ x : Fin 6 → ℚ, system x ↔ solution x :=
sorry

end NUMINAMATH_CALUDE_system_solution_l1308_130825


namespace NUMINAMATH_CALUDE_largest_c_for_3_in_range_l1308_130860

def f (c : ℝ) (x : ℝ) : ℝ := x^2 - 6*x + c

theorem largest_c_for_3_in_range : 
  (∃ (c : ℝ), ∀ (c' : ℝ), 
    (∃ (x : ℝ), f c' x = 3) → c' ≤ c ∧ 
    (∃ (x : ℝ), f c x = 3) ∧
    c = 12) := by sorry

end NUMINAMATH_CALUDE_largest_c_for_3_in_range_l1308_130860


namespace NUMINAMATH_CALUDE_time_for_b_alone_l1308_130814

/-- Given that:
  1. It takes 'a' hours for A and B to complete the work together.
  2. It takes 'b' hours for A to complete the work alone.
  Prove that the time it takes B alone to complete the work is ab / (b - a) hours. -/
theorem time_for_b_alone (a b : ℝ) (h1 : a > 0) (h2 : b > a) : 
  (1 / a + 1 / (a * b / (b - a)) = 1) := by
sorry

end NUMINAMATH_CALUDE_time_for_b_alone_l1308_130814


namespace NUMINAMATH_CALUDE_candy_left_l1308_130845

/-- Represents the number of candy pieces Debby has -/
def candy_count : ℕ := 12

/-- Represents the number of candy pieces Debby ate -/
def eaten_candy : ℕ := 9

/-- Theorem stating how many pieces of candy Debby has left -/
theorem candy_left : candy_count - eaten_candy = 3 := by sorry

end NUMINAMATH_CALUDE_candy_left_l1308_130845


namespace NUMINAMATH_CALUDE_fraction_sum_equality_l1308_130826

theorem fraction_sum_equality : (3 / 10 : ℚ) + (5 / 100 : ℚ) - (2 / 1000 : ℚ) = (348 / 1000 : ℚ) := by
  sorry

end NUMINAMATH_CALUDE_fraction_sum_equality_l1308_130826


namespace NUMINAMATH_CALUDE_initial_number_of_persons_l1308_130817

theorem initial_number_of_persons
  (average_weight_increase : ℝ)
  (weight_difference : ℝ)
  (h1 : average_weight_increase = 2.5)
  (h2 : weight_difference = 20)
  (h3 : average_weight_increase * (initial_persons : ℝ) = weight_difference) :
  initial_persons = 8 := by
sorry

end NUMINAMATH_CALUDE_initial_number_of_persons_l1308_130817


namespace NUMINAMATH_CALUDE_school_fair_revenue_l1308_130848

/-- Calculates the total revenue from sales at a school fair -/
theorem school_fair_revenue (chips_sold : ℕ) (chips_price : ℚ)
  (hot_dogs_sold : ℕ) (hot_dogs_price : ℚ)
  (drinks_sold : ℕ) (drinks_price : ℚ) :
  chips_sold = 27 →
  chips_price = 3/2 →
  hot_dogs_sold = chips_sold - 8 →
  hot_dogs_price = 3 →
  drinks_sold = hot_dogs_sold + 12 →
  drinks_price = 2 →
  chips_sold * chips_price + hot_dogs_sold * hot_dogs_price + drinks_sold * drinks_price = 159.5 := by
  sorry

#eval (27 : ℕ) * (3/2 : ℚ) + (27 - 8 : ℕ) * (3 : ℚ) + ((27 - 8 : ℕ) + 12) * (2 : ℚ)

end NUMINAMATH_CALUDE_school_fair_revenue_l1308_130848


namespace NUMINAMATH_CALUDE_speedster_convertible_fraction_l1308_130830

theorem speedster_convertible_fraction :
  ∀ (total_inventory : ℕ) (speedsters : ℕ) (speedster_convertibles : ℕ),
    speedsters = total_inventory / 3 →
    total_inventory - speedsters = 30 →
    speedster_convertibles = 12 →
    (speedster_convertibles : ℚ) / speedsters = 4 / 5 := by
  sorry

end NUMINAMATH_CALUDE_speedster_convertible_fraction_l1308_130830


namespace NUMINAMATH_CALUDE_ice_cream_cup_cost_l1308_130891

/-- Calculates the cost of each ice-cream cup given the order details and total amount paid -/
theorem ice_cream_cup_cost
  (chapati_count : ℕ)
  (chapati_cost : ℕ)
  (rice_count : ℕ)
  (rice_cost : ℕ)
  (vegetable_count : ℕ)
  (vegetable_cost : ℕ)
  (ice_cream_count : ℕ)
  (total_paid : ℕ)
  (h1 : chapati_count = 16)
  (h2 : chapati_cost = 6)
  (h3 : rice_count = 5)
  (h4 : rice_cost = 45)
  (h5 : vegetable_count = 7)
  (h6 : vegetable_cost = 70)
  (h7 : ice_cream_count = 6)
  (h8 : total_paid = 883)
  : (total_paid - (chapati_count * chapati_cost + rice_count * rice_cost + vegetable_count * vegetable_cost)) / ice_cream_count = 12 := by
  sorry

#check ice_cream_cup_cost

end NUMINAMATH_CALUDE_ice_cream_cup_cost_l1308_130891


namespace NUMINAMATH_CALUDE_mary_screw_ratio_l1308_130809

/-- The number of screws Mary initially has -/
def initial_screws : ℕ := 8

/-- The number of sections Mary needs to split the screws into -/
def num_sections : ℕ := 4

/-- The number of screws needed in each section -/
def screws_per_section : ℕ := 6

/-- The ratio of screws Mary needs to buy to the screws she initially has -/
def screw_ratio : ℚ := 2

theorem mary_screw_ratio : 
  (num_sections * screws_per_section - initial_screws) / initial_screws = screw_ratio := by
  sorry

end NUMINAMATH_CALUDE_mary_screw_ratio_l1308_130809


namespace NUMINAMATH_CALUDE_smallest_factor_difference_l1308_130849

theorem smallest_factor_difference (n : ℕ) (hn : n = 2310) :
  ∃ (a b : ℕ), a * b = n ∧ 
    (∀ (x y : ℕ), x * y = n → x ≤ y → y - x ≥ (b - a)) ∧
    b - a = 13 :=
  sorry

end NUMINAMATH_CALUDE_smallest_factor_difference_l1308_130849


namespace NUMINAMATH_CALUDE_pool_width_calculation_l1308_130841

/-- Represents a rectangular swimming pool with a surrounding deck -/
structure PoolWithDeck where
  poolLength : ℝ
  poolWidth : ℝ
  deckWidth : ℝ

/-- Calculates the total area of the pool and deck -/
def totalArea (p : PoolWithDeck) : ℝ :=
  (p.poolLength + 2 * p.deckWidth) * (p.poolWidth + 2 * p.deckWidth)

/-- Theorem stating the width of the pool given specific conditions -/
theorem pool_width_calculation (p : PoolWithDeck) 
    (h1 : p.poolLength = 20)
    (h2 : p.deckWidth = 3)
    (h3 : totalArea p = 728) :
    p.poolWidth = 572 / 46 := by
  sorry

end NUMINAMATH_CALUDE_pool_width_calculation_l1308_130841


namespace NUMINAMATH_CALUDE_barrys_age_l1308_130813

theorem barrys_age (sisters_average_age : ℕ) (total_average_age : ℕ) : 
  sisters_average_age = 27 → total_average_age = 28 → 
  (3 * sisters_average_age + 31) / 4 = total_average_age :=
by
  sorry

#check barrys_age

end NUMINAMATH_CALUDE_barrys_age_l1308_130813


namespace NUMINAMATH_CALUDE_geometric_sequence_12th_term_l1308_130847

/-- A geometric sequence is defined by its first term and common ratio -/
def GeometricSequence (a₁ : ℝ) (r : ℝ) := fun n : ℕ => a₁ * r ^ (n - 1)

/-- The nth term of a geometric sequence -/
def nthTerm (seq : ℕ → ℝ) (n : ℕ) : ℝ := seq n

theorem geometric_sequence_12th_term
  (seq : ℕ → ℝ)
  (h_geometric : ∃ a₁ r, seq = GeometricSequence a₁ r)
  (h_4th : nthTerm seq 4 = 4)
  (h_7th : nthTerm seq 7 = 32) :
  nthTerm seq 12 = 1024 := by
  sorry


end NUMINAMATH_CALUDE_geometric_sequence_12th_term_l1308_130847


namespace NUMINAMATH_CALUDE_nancys_hourly_wage_l1308_130880

/-- Proves that Nancy needs to make $10 per hour to pay the rest of her tuition --/
theorem nancys_hourly_wage (tuition : ℝ) (scholarship : ℝ) (work_hours : ℝ) :
  tuition = 22000 →
  scholarship = 3000 →
  work_hours = 200 →
  (tuition / 2 - scholarship - 2 * scholarship) / work_hours = 10 := by
  sorry

end NUMINAMATH_CALUDE_nancys_hourly_wage_l1308_130880


namespace NUMINAMATH_CALUDE_y_minus_x_equals_one_tenth_l1308_130816

-- Define the rounding function to the tenths place
def roundToTenths (x : ℚ) : ℚ := ⌊x * 10 + 1/2⌋ / 10

-- Define the given values
def a : ℚ := 545/100
def b : ℚ := 295/100
def c : ℚ := 374/100

-- Define x as the sum of a, b, and c rounded to tenths
def x : ℚ := roundToTenths (a + b + c)

-- Define y as the sum of a, b, and c individually rounded to tenths
def y : ℚ := roundToTenths a + roundToTenths b + roundToTenths c

-- State the theorem
theorem y_minus_x_equals_one_tenth : y - x = 1/10 := by sorry

end NUMINAMATH_CALUDE_y_minus_x_equals_one_tenth_l1308_130816


namespace NUMINAMATH_CALUDE_largest_among_four_l1308_130888

theorem largest_among_four (a b : ℝ) (h1 : b > a) (h2 : a > 0) (h3 : a + b = 1) :
  b > (1/2 : ℝ) ∧ b > 2*a*b ∧ b > a^2 + b^2 := by
  sorry

end NUMINAMATH_CALUDE_largest_among_four_l1308_130888


namespace NUMINAMATH_CALUDE_increasing_f_implies_a_range_l1308_130850

def f (a x : ℝ) : ℝ := x^2 + 2*(a-2)*x + 5

theorem increasing_f_implies_a_range (a : ℝ) :
  (∀ x₁ x₂ : ℝ, 4 ≤ x₁ ∧ x₁ < x₂ → f a x₁ < f a x₂) →
  a ∈ Set.Ici (-2) :=
by sorry

end NUMINAMATH_CALUDE_increasing_f_implies_a_range_l1308_130850


namespace NUMINAMATH_CALUDE_parabola_y_intercepts_l1308_130835

/-- The number of y-intercepts for the parabola x = 3y^2 - 6y + 3 -/
theorem parabola_y_intercepts : 
  let f : ℝ → ℝ := fun y => 3 * y^2 - 6 * y + 3
  ∃! y : ℝ, f y = 0 :=
by sorry

end NUMINAMATH_CALUDE_parabola_y_intercepts_l1308_130835


namespace NUMINAMATH_CALUDE_jills_water_volume_l1308_130899

/-- Represents the number of jars of each size -/
def jars_per_size : ℕ := 48 / 3

/-- Represents the volume of a quart in gallons -/
def quart_volume : ℚ := 1 / 4

/-- Represents the volume of a half-gallon in gallons -/
def half_gallon_volume : ℚ := 1 / 2

/-- Represents the volume of a gallon in gallons -/
def gallon_volume : ℚ := 1

/-- Calculates the total volume of water in gallons -/
def total_water_volume : ℚ :=
  jars_per_size * quart_volume +
  jars_per_size * half_gallon_volume +
  jars_per_size * gallon_volume

theorem jills_water_volume :
  total_water_volume = 28 := by
  sorry

end NUMINAMATH_CALUDE_jills_water_volume_l1308_130899


namespace NUMINAMATH_CALUDE_eight_sided_die_probability_l1308_130870

/-- Represents the number of sides on the die -/
def sides : ℕ := 8

/-- Represents the event where the first roll is greater than or equal to the second roll -/
def favorable_outcomes (s : ℕ) : ℕ := (s * (s + 1)) / 2

/-- The probability of the first roll being greater than or equal to the second roll -/
def probability (s : ℕ) : ℚ := (favorable_outcomes s) / (s^2 : ℚ)

/-- Theorem stating that for an 8-sided die, the probability of the first roll being 
    greater than or equal to the second roll is 9/16 -/
theorem eight_sided_die_probability : probability sides = 9/16 := by
  sorry

end NUMINAMATH_CALUDE_eight_sided_die_probability_l1308_130870


namespace NUMINAMATH_CALUDE_hakimi_age_l1308_130831

/-- Given three friends with an average age of 40, where Jared is ten years older than Hakimi
    and Molly is 30 years old, prove that Hakimi's age is 40. -/
theorem hakimi_age (average_age : ℝ) (molly_age : ℝ) (jared_hakimi_age_diff : ℝ) 
  (h1 : average_age = 40)
  (h2 : molly_age = 30)
  (h3 : jared_hakimi_age_diff = 10) : 
  ∃ (hakimi_age : ℝ), hakimi_age = 40 ∧ 
    (hakimi_age + (hakimi_age + jared_hakimi_age_diff) + molly_age) / 3 = average_age :=
by sorry

end NUMINAMATH_CALUDE_hakimi_age_l1308_130831


namespace NUMINAMATH_CALUDE_cyclist_catchup_time_l1308_130875

/-- Two cyclists A and B travel from station A to station B -/
structure Cyclist where
  speed : ℝ
  startTime : ℝ

/-- The problem setup -/
def cyclistProblem (A B : Cyclist) (distance : ℝ) : Prop :=
  A.speed * 30 = distance ∧  -- A takes 30 minutes to reach station B
  B.speed * 40 = distance ∧  -- B takes 40 minutes to reach station B
  B.startTime = A.startTime - 5  -- B starts 5 minutes earlier than A

/-- The theorem to prove -/
theorem cyclist_catchup_time (A B : Cyclist) (distance : ℝ) 
  (h : cyclistProblem A B distance) : 
  ∃ t : ℝ, t = 15 ∧ A.speed * t = B.speed * (t + 5) :=
sorry

end NUMINAMATH_CALUDE_cyclist_catchup_time_l1308_130875


namespace NUMINAMATH_CALUDE_exactly_one_correct_l1308_130878

-- Define the four propositions
def proposition1 : Prop := ∀ (p q : Prop), (p ∧ q → p ∨ q) ∧ ¬(p ∨ q → p ∧ q)

def proposition2 : Prop :=
  let p := ∃ x : ℝ, x^2 + 2*x ≤ 0
  ¬p ↔ ∀ x : ℝ, x^2 + 2*x > 0

def proposition3 : Prop :=
  ¬(∀ x : ℝ, x^2 - 2*x + 3 > 0) ↔ (∃ x : ℝ, x^2 - 2*x + 3 < 0)

def proposition4 : Prop :=
  ∀ (p q : Prop), (¬p → q) ↔ (p → ¬q)

-- Theorem stating that exactly one proposition is correct
theorem exactly_one_correct : 
  (proposition2 ∧ ¬proposition1 ∧ ¬proposition3 ∧ ¬proposition4) :=
sorry

end NUMINAMATH_CALUDE_exactly_one_correct_l1308_130878


namespace NUMINAMATH_CALUDE_intersection_A_complement_B_equals_A_l1308_130829

open Set

-- Define the universe U as the set of real numbers
def U : Set ℝ := univ

-- Define set A
def A : Set ℝ := {0, 1, 2}

-- Define set B
def B : Set ℝ := {x | x^2 - 2*x - 3 > 0}

-- Theorem statement
theorem intersection_A_complement_B_equals_A : A ∩ (U \ B) = A := by sorry

end NUMINAMATH_CALUDE_intersection_A_complement_B_equals_A_l1308_130829


namespace NUMINAMATH_CALUDE_f_range_on_interval_l1308_130893

/-- The function f(x) = 1 - 4x - 2x^2 -/
def f (x : ℝ) : ℝ := 1 - 4*x - 2*x^2

/-- The range of f(x) on the interval (1, +∞) is (-∞, -5) -/
theorem f_range_on_interval :
  Set.range (fun x => f x) ∩ Set.Ioi 1 = Set.Iio (-5) := by sorry

end NUMINAMATH_CALUDE_f_range_on_interval_l1308_130893


namespace NUMINAMATH_CALUDE_smaller_root_equation_l1308_130869

theorem smaller_root_equation (x : ℚ) : 
  let equation := (x - 3/4) * (x - 3/4) + (x - 3/4) * (x - 1/2) = 0
  let smaller_root := 5/8
  (equation ∧ x = smaller_root) ∨ 
  (equation ∧ x ≠ smaller_root ∧ x > smaller_root) :=
by sorry

end NUMINAMATH_CALUDE_smaller_root_equation_l1308_130869


namespace NUMINAMATH_CALUDE_tangerines_count_l1308_130881

/-- The number of tangerines in a fruit basket -/
def num_tangerines (total fruits bananas apples pears : ℕ) : ℕ :=
  total - (bananas + apples + pears)

/-- Theorem: There are 13 tangerines in the fruit basket -/
theorem tangerines_count :
  let total := 60
  let bananas := 32
  let apples := 10
  let pears := 5
  num_tangerines total bananas apples pears = 13 := by
  sorry

end NUMINAMATH_CALUDE_tangerines_count_l1308_130881


namespace NUMINAMATH_CALUDE_binomial_identity_l1308_130840

theorem binomial_identity (k n : ℕ) (hk : k > 1) (hn : n > 1) :
  k * Nat.choose n k = n * Nat.choose (n - 1) (k - 1) := by
  sorry

end NUMINAMATH_CALUDE_binomial_identity_l1308_130840


namespace NUMINAMATH_CALUDE_line_through_points_l1308_130876

/-- Given a line y = ax + b passing through points (3,7) and (7,19), prove that a - b = 5 -/
theorem line_through_points (a b : ℝ) : 
  (∀ x y : ℝ, y = a * x + b) → 
  (7 : ℝ) = a * 3 + b → 
  (19 : ℝ) = a * 7 + b → 
  a - b = 5 := by
  sorry

end NUMINAMATH_CALUDE_line_through_points_l1308_130876


namespace NUMINAMATH_CALUDE_radiator_problem_l1308_130818

/-- Represents the fraction of water remaining after a number of replacements -/
def water_fraction (initial_volume : ℚ) (replacement_volume : ℚ) (num_replacements : ℕ) : ℚ :=
  (1 - replacement_volume / initial_volume) ^ num_replacements

/-- The problem statement -/
theorem radiator_problem (initial_volume : ℚ) (replacement_volume : ℚ) (num_replacements : ℕ)
    (h1 : initial_volume = 20)
    (h2 : replacement_volume = 5)
    (h3 : num_replacements = 4) :
  water_fraction initial_volume replacement_volume num_replacements = 81 / 256 := by
  sorry

#eval water_fraction 20 5 4

end NUMINAMATH_CALUDE_radiator_problem_l1308_130818


namespace NUMINAMATH_CALUDE_magnitude_of_sum_l1308_130884

def a : ℝ × ℝ := (2, 3)
def b (m : ℝ) : ℝ × ℝ := (m, -6)

theorem magnitude_of_sum (m : ℝ) (h : a.1 * (b m).1 + a.2 * (b m).2 = 0) :
  Real.sqrt ((2 * a.1 + (b m).1)^2 + (2 * a.2 + (b m).2)^2) = 13 :=
by sorry

end NUMINAMATH_CALUDE_magnitude_of_sum_l1308_130884


namespace NUMINAMATH_CALUDE_nesbitts_inequality_l1308_130800

theorem nesbitts_inequality (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  a / (b + c) + b / (a + c) + c / (a + b) ≥ 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_nesbitts_inequality_l1308_130800


namespace NUMINAMATH_CALUDE_age_problem_l1308_130805

theorem age_problem (a b c : ℕ) 
  (h1 : a = b + 2) 
  (h2 : b = 2 * c) 
  (h3 : a + b + c = 72) : 
  b = 28 := by
  sorry

end NUMINAMATH_CALUDE_age_problem_l1308_130805


namespace NUMINAMATH_CALUDE_circle_symmetry_l1308_130856

-- Define the original circle
def original_circle (x y : ℝ) : Prop := (x - 2)^2 + y^2 = 4

-- Define the line of symmetry
def symmetry_line (x y : ℝ) : Prop := y = (Real.sqrt 3 / 3) * x

-- Define the symmetric circle
def symmetric_circle (x y : ℝ) : Prop := (x - 1)^2 + (y - Real.sqrt 3)^2 = 4

-- Theorem statement
theorem circle_symmetry :
  ∀ (x₁ y₁ x₂ y₂ : ℝ),
    original_circle x₁ y₁ →
    symmetric_circle x₂ y₂ →
    ∃ (x_mid y_mid : ℝ),
      symmetry_line x_mid y_mid ∧
      x_mid = (x₁ + x₂) / 2 ∧
      y_mid = (y₁ + y₂) / 2 :=
sorry

end NUMINAMATH_CALUDE_circle_symmetry_l1308_130856


namespace NUMINAMATH_CALUDE_rectangular_solid_surface_area_l1308_130866

/-- The total surface area of a rectangular solid -/
def totalSurfaceArea (length width depth : ℝ) : ℝ :=
  2 * (length * width + width * depth + length * depth)

/-- Theorem: The total surface area of a rectangular solid with length 9 meters, width 8 meters, and depth 5 meters is 314 square meters -/
theorem rectangular_solid_surface_area :
  totalSurfaceArea 9 8 5 = 314 := by
  sorry

end NUMINAMATH_CALUDE_rectangular_solid_surface_area_l1308_130866


namespace NUMINAMATH_CALUDE_min_value_theorem_l1308_130894

theorem min_value_theorem (x y : ℝ) : (x + y)^2 + (x - 2/y)^2 ≥ 4 := by
  sorry

end NUMINAMATH_CALUDE_min_value_theorem_l1308_130894


namespace NUMINAMATH_CALUDE_box_triples_count_l1308_130821

/-- The number of ordered triples (a, b, c) of positive integers satisfying the box conditions -/
def box_triples : Nat :=
  (Finset.filter (fun t : Nat × Nat × Nat =>
    let (a, b, c) := t
    1 ≤ a ∧ a ≤ b ∧ b ≤ c ∧ a * b * c = 4 * (a * b + a * c + b * c))
    (Finset.product (Finset.range 100) (Finset.product (Finset.range 100) (Finset.range 100)))).card

/-- Theorem stating that there are exactly 2 ordered triples satisfying the box conditions -/
theorem box_triples_count : box_triples = 2 := by
  sorry

end NUMINAMATH_CALUDE_box_triples_count_l1308_130821
