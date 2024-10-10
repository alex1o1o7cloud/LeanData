import Mathlib

namespace third_offense_sentence_extension_l1658_165842

theorem third_offense_sentence_extension (original_sentence total_time : ℕ) 
  (h1 : original_sentence = 27)
  (h2 : total_time = 36) :
  (total_time - original_sentence) / original_sentence = 1 / 3 := by
sorry

end third_offense_sentence_extension_l1658_165842


namespace ratio_problem_l1658_165852

theorem ratio_problem (a b c : ℚ) (h1 : b/a = 4) (h2 : c/b = 5) : 
  (a + 2*b) / (3*b + c) = 9/32 := by
  sorry

end ratio_problem_l1658_165852


namespace max_volume_rect_frame_l1658_165889

/-- Represents the dimensions of a rectangular frame. -/
structure RectFrame where
  length : ℝ
  width : ℝ
  height : ℝ

/-- Calculates the volume of a rectangular frame. -/
def volume (frame : RectFrame) : ℝ :=
  frame.length * frame.width * frame.height

/-- Calculates the perimeter of the base of a rectangular frame. -/
def basePerimeter (frame : RectFrame) : ℝ :=
  2 * (frame.length + frame.width)

/-- Calculates the total length of steel bar used for a rectangular frame. -/
def totalBarLength (frame : RectFrame) : ℝ :=
  basePerimeter frame + 4 * frame.height

/-- Theorem: The maximum volume of a rectangular frame enclosed by an 18m steel bar,
    where the ratio of length to width is 2:1, is equal to the correct maximum volume. -/
theorem max_volume_rect_frame :
  ∃ (frame : RectFrame),
    frame.length = 2 * frame.width ∧
    totalBarLength frame = 18 ∧
    ∀ (other : RectFrame),
      other.length = 2 * other.width →
      totalBarLength other = 18 →
      volume frame ≥ volume other :=
by sorry


end max_volume_rect_frame_l1658_165889


namespace largest_valid_marking_l1658_165845

/-- A marking function that assigns a boolean value to each cell in an n × n grid. -/
def Marking (n : ℕ) := Fin n → Fin n → Bool

/-- Predicate that checks if a rectangle contains a marked cell. -/
def ContainsMarkedCell (m : Marking n) (x y w h : Fin n) : Prop :=
  ∃ i j, i < w ∧ j < h ∧ m (x + i) (y + j) = true

/-- Predicate that checks if a marking satisfies the condition for all rectangles. -/
def ValidMarking (n : ℕ) (m : Marking n) : Prop :=
  ∀ x y w h : Fin n, w * h ≥ n → ContainsMarkedCell m x y w h

/-- The main theorem stating that 7 is the largest n for which a valid marking exists. -/
theorem largest_valid_marking :
  (∃ (m : Marking 7), ValidMarking 7 m) ∧
  (∀ n > 7, ¬∃ (m : Marking n), ValidMarking n m) :=
sorry

end largest_valid_marking_l1658_165845


namespace time_to_run_square_field_l1658_165849

/-- The time taken for a boy to run around a square field -/
theorem time_to_run_square_field (side_length : ℝ) (speed_kmh : ℝ) : 
  side_length = 35 → speed_kmh = 9 → 
  (4 * side_length) / (speed_kmh * 1000 / 3600) = 56 := by
  sorry

end time_to_run_square_field_l1658_165849


namespace cupcakes_remaining_l1658_165856

/-- The number of cupcakes in a dozen -/
def dozen : ℕ := 12

/-- The number of cupcakes Dani brings -/
def cupcakes_brought : ℕ := (10 * dozen) + (dozen / 2)

/-- The total number of students in the class -/
def total_students : ℕ := 48

/-- The number of teachers -/
def teachers : ℕ := 2

/-- The number of teacher's aids -/
def teacher_aids : ℕ := 2

/-- The number of absent students -/
def absent_students : ℕ := 6

/-- The number of students on a field trip -/
def field_trip_students : ℕ := 8

/-- The number of people present in the class -/
def people_present : ℕ := total_students - absent_students - field_trip_students + teachers + teacher_aids

/-- The number of cupcakes left after distribution -/
def cupcakes_left : ℕ := cupcakes_brought - people_present

theorem cupcakes_remaining :
  cupcakes_left = 85 :=
sorry

end cupcakes_remaining_l1658_165856


namespace sandbox_area_l1658_165836

theorem sandbox_area (length width : ℕ) (h1 : length = 312) (h2 : width = 146) :
  length * width = 45552 := by
  sorry

end sandbox_area_l1658_165836


namespace a_value_satisfies_condition_l1658_165819

-- Define the property that needs to be satisfied
def satisfies_condition (a : ℕ) : Prop :=
  ∀ K : ℤ, K ≠ 27 → (27 - K) ∣ (a - K^1964)

-- State the theorem
theorem a_value_satisfies_condition :
  satisfies_condition (3^5892) :=
sorry

end a_value_satisfies_condition_l1658_165819


namespace translation_of_quadratic_l1658_165805

/-- The original quadratic function -/
def g (x : ℝ) : ℝ := -2 * x^2

/-- The translated quadratic function -/
def f (x : ℝ) : ℝ := -2 * x^2 - 12 * x - 16

/-- The vertex of the original function g -/
def vertex_g : ℝ × ℝ := (0, 0)

/-- The vertex of the translated function f -/
def vertex_f : ℝ × ℝ := (-3, 2)

/-- Theorem stating that f is the translation of g -/
theorem translation_of_quadratic :
  ∀ x : ℝ, f x = g (x + 3) + 2 :=
sorry

end translation_of_quadratic_l1658_165805


namespace arithmetic_sequence_properties_l1658_165876

/-- An arithmetic sequence with sum S_n of first n terms -/
structure ArithmeticSequence where
  a : ℕ → ℝ  -- The sequence
  d : ℝ      -- Common difference
  S : ℕ → ℝ  -- Sum function

/-- Properties of the specific arithmetic sequence -/
def SpecificSequence (seq : ArithmeticSequence) : Prop :=
  seq.S 6 > seq.S 7 ∧ seq.S 7 > seq.S 5

theorem arithmetic_sequence_properties (seq : ArithmeticSequence) 
  (h : SpecificSequence seq) : 
  seq.d < 0 ∧ 
  seq.S 11 > 0 ∧ 
  seq.S 12 > 0 ∧ 
  seq.S 8 < seq.S 5 := by
  sorry


end arithmetic_sequence_properties_l1658_165876


namespace largest_odd_in_sum_not_exceeding_200_l1658_165806

/-- The sum of the first n odd numbers -/
def sumOddNumbers (n : ℕ) : ℕ := n^2

/-- The nth odd number -/
def nthOddNumber (n : ℕ) : ℕ := 2*n - 1

theorem largest_odd_in_sum_not_exceeding_200 :
  ∃ n : ℕ, sumOddNumbers n ≤ 200 ∧ 
           sumOddNumbers (n + 1) > 200 ∧ 
           nthOddNumber n = 27 :=
sorry

end largest_odd_in_sum_not_exceeding_200_l1658_165806


namespace absolute_sum_zero_implies_sum_l1658_165816

theorem absolute_sum_zero_implies_sum (x y : ℝ) :
  |x - 1| + |y + 3| = 0 → x + y = -2 := by
sorry

end absolute_sum_zero_implies_sum_l1658_165816


namespace range_of_m_l1658_165864

def p (m : ℝ) : Prop :=
  ∃ x y : ℝ, x < 0 ∧ y < 0 ∧ x ≠ y ∧ x^2 + m*x + 1 = 0 ∧ y^2 + m*y + 1 = 0

def q (m : ℝ) : Prop :=
  ∀ x : ℝ, 4*x^2 + 4*(m-2)*x + 1 > 0

theorem range_of_m (m : ℝ) :
  (p m ∨ q m) ∧ ¬(p m ∧ q m) →
  (m ≥ 3 ∨ (1 < m ∧ m ≤ 2)) :=
by sorry

end range_of_m_l1658_165864


namespace simplify_and_evaluate_expression_l1658_165854

theorem simplify_and_evaluate_expression (a : ℝ) (h : a = 2) :
  (1 + 4 / (a - 1)) / ((a^2 + 6*a + 9) / (a^2 - a)) = 2/5 := by
  sorry

end simplify_and_evaluate_expression_l1658_165854


namespace glass_bowls_problem_l1658_165810

/-- The number of glass bowls initially bought -/
def initial_bowls : ℕ := 316

/-- The cost price per bowl in rupees -/
def cost_price : ℚ := 12

/-- The selling price per bowl in rupees -/
def selling_price : ℚ := 15

/-- The number of bowls sold -/
def bowls_sold : ℕ := 102

/-- The percentage gain -/
def percentage_gain : ℚ := 8050847457627118 / 1000000000000000

theorem glass_bowls_problem :
  initial_bowls = 316 ∧
  (bowls_sold : ℚ) * (selling_price - cost_price) / (initial_bowls * cost_price) = percentage_gain / 100 := by
  sorry


end glass_bowls_problem_l1658_165810


namespace arithmetic_sequence_sum_l1658_165865

def arithmetic_sequence (a : ℕ → ℝ) := ∀ n, a (n + 1) - a n = a (n + 2) - a (n + 1)

theorem arithmetic_sequence_sum (a : ℕ → ℝ) :
  arithmetic_sequence a →
  (a 3 + a 4 + a 5 = 12) →
  (a 1 + a 2 + a 3 + a 4 + a 5 + a 6 + a 7 = 28) :=
by
  sorry

end arithmetic_sequence_sum_l1658_165865


namespace track_length_l1658_165858

/-- The length of a circular track given specific running conditions -/
theorem track_length (first_lap : ℝ) (other_laps : ℝ) (avg_speed : ℝ) : 
  first_lap = 70 →
  other_laps = 85 →
  avg_speed = 5 →
  (3 : ℝ) * (first_lap + 2 * other_laps) * avg_speed / 3 = 400 := by
  sorry

end track_length_l1658_165858


namespace quadratic_equations_solutions_l1658_165888

theorem quadratic_equations_solutions :
  (∀ x, x^2 - 7*x - 18 = 0 ↔ x = 9 ∨ x = -2) ∧
  (∀ x, 4*x^2 + 1 = 4*x ↔ x = 1/2) := by
  sorry

end quadratic_equations_solutions_l1658_165888


namespace equal_distribution_of_drawings_l1658_165862

theorem equal_distribution_of_drawings (total_drawings : ℕ) (num_neighbors : ℕ) 
  (h1 : total_drawings = 54) (h2 : num_neighbors = 6) :
  total_drawings / num_neighbors = 9 := by
  sorry

end equal_distribution_of_drawings_l1658_165862


namespace unique_prime_tower_l1658_165817

def tower_of_twos (p : ℕ) : ℕ :=
  match p with
  | 0 => 1
  | n + 1 => 2^(tower_of_twos n)

def is_prime_tower (p : ℕ) : Prop :=
  Nat.Prime (tower_of_twos p + 9)

theorem unique_prime_tower : ∀ p : ℕ, is_prime_tower p ↔ p = 1 :=
sorry

end unique_prime_tower_l1658_165817


namespace bill_meets_dexter_at_12_50_l1658_165821

/-- Represents a person or dog in the problem -/
structure Participant where
  speed : ℝ
  startTime : ℝ

/-- Calculates the time when Bill meets Dexter -/
def meetingTime (anna bill dexter : Participant) : ℝ :=
  sorry

/-- Theorem stating that Bill meets Dexter at 12:50 pm -/
theorem bill_meets_dexter_at_12_50 :
  let anna : Participant := { speed := 4, startTime := 0 }
  let bill : Participant := { speed := 3, startTime := 0 }
  let dexter : Participant := { speed := 6, startTime := 0.25 }
  meetingTime anna bill dexter = 0.8333333333 := by
  sorry

end bill_meets_dexter_at_12_50_l1658_165821


namespace unique_solution_for_equation_l1658_165853

theorem unique_solution_for_equation : ∃! p n k : ℕ+, 
  Nat.Prime p ∧ 
  k > 1 ∧ 
  (3 : ℕ)^(p : ℕ) + (4 : ℕ)^(p : ℕ) = (n : ℕ)^(k : ℕ) ∧ 
  p = 2 ∧ n = 5 ∧ k = 2 := by
  sorry

end unique_solution_for_equation_l1658_165853


namespace base_number_irrelevant_l1658_165843

def decimal_places (n : ℝ) : ℕ := sorry

theorem base_number_irrelevant (x : ℤ) :
  decimal_places ((x^4 * 3.456789)^14) = decimal_places (3.456789^14) := by sorry

end base_number_irrelevant_l1658_165843


namespace mans_age_ratio_l1658_165850

theorem mans_age_ratio (mans_age father_age : ℕ) : 
  father_age = 60 →
  mans_age + 12 = (father_age + 12) / 2 →
  mans_age * 5 = father_age * 2 := by
sorry

end mans_age_ratio_l1658_165850


namespace two_digit_product_less_than_five_digits_l1658_165898

theorem two_digit_product_less_than_five_digits : ∀ a b : ℕ, 
  10 ≤ a ∧ a ≤ 99 → 10 ≤ b ∧ b ≤ 99 → a * b < 10000 := by
  sorry

end two_digit_product_less_than_five_digits_l1658_165898


namespace phone_answer_probability_l1658_165873

theorem phone_answer_probability : 
  let p1 : ℚ := 1/10  -- Probability of answering on the first ring
  let p2 : ℚ := 3/10  -- Probability of answering on the second ring
  let p3 : ℚ := 2/5   -- Probability of answering on the third ring
  let p4 : ℚ := 1/10  -- Probability of answering on the fourth ring
  p1 + p2 + p3 + p4 = 9/10 := by
sorry

end phone_answer_probability_l1658_165873


namespace ratio_of_sequences_l1658_165812

def arithmetic_sum (a₁ : ℚ) (d : ℚ) (n : ℕ) : ℚ :=
  n * (2 * a₁ + (n - 1) * d) / 2

def numerator_sequence : ℚ := arithmetic_sum 2 2 17
def denominator_sequence : ℚ := arithmetic_sum 3 3 17

theorem ratio_of_sequences : numerator_sequence / denominator_sequence = 2 / 3 := by
  sorry

end ratio_of_sequences_l1658_165812


namespace square_division_theorem_l1658_165867

-- Define the square and point P
def Square (E F G H : ℝ × ℝ) : Prop :=
  let (x₁, y₁) := E
  let (x₂, y₂) := F
  let (x₃, y₃) := G
  let (x₄, y₄) := H
  (x₂ - x₁)^2 + (y₂ - y₁)^2 = 16 ∧
  (x₃ - x₂)^2 + (y₃ - y₂)^2 = 16 ∧
  (x₄ - x₃)^2 + (y₄ - y₃)^2 = 16 ∧
  (x₁ - x₄)^2 + (y₁ - y₄)^2 = 16

def PointOnSide (P : ℝ × ℝ) (E H : ℝ × ℝ) : Prop :=
  ∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ P = (t * E.1 + (1 - t) * H.1, t * E.2 + (1 - t) * H.2)

-- Define the area division property
def DivideAreaEqually (P : ℝ × ℝ) (E F G H : ℝ × ℝ) : Prop :=
  let area_EFP := abs ((F.1 - E.1) * (P.2 - E.2) - (P.1 - E.1) * (F.2 - E.2)) / 2
  let area_FGP := abs ((G.1 - F.1) * (P.2 - F.2) - (P.1 - F.1) * (G.2 - F.2)) / 2
  let area_GHP := abs ((H.1 - G.1) * (P.2 - G.2) - (P.1 - G.1) * (H.2 - G.2)) / 2
  let area_HEP := abs ((E.1 - H.1) * (P.2 - H.2) - (P.1 - H.1) * (E.2 - H.2)) / 2
  area_EFP = area_FGP ∧ area_FGP = area_GHP ∧ area_GHP = area_HEP

-- State the theorem
theorem square_division_theorem (E F G H P : ℝ × ℝ) :
  Square E F G H →
  PointOnSide P E H →
  DivideAreaEqually P E F G H →
  (F.1 - P.1)^2 + (F.2 - P.2)^2 = 20 :=
by sorry

end square_division_theorem_l1658_165867


namespace factorial_difference_l1658_165895

theorem factorial_difference : Nat.factorial 10 - Nat.factorial 9 / Nat.factorial 3 = 3568320 := by
  sorry

end factorial_difference_l1658_165895


namespace vertex_coordinates_l1658_165801

-- Define the parabola equation
def parabola (x : ℝ) : ℝ := -(x - 1)^2 - 2

-- State the theorem
theorem vertex_coordinates :
  ∃ (x y : ℝ), x = 1 ∧ y = -2 ∧ ∀ (t : ℝ), parabola t ≤ parabola x :=
sorry

end vertex_coordinates_l1658_165801


namespace production_line_b_units_l1658_165822

/-- 
Given a factory with three production lines A, B, and C, prove that production line B 
produced 1000 units under the following conditions:
1. The total number of units produced is 3000
2. The number of units sampled from each production line (a, b, c) form an arithmetic sequence
3. The sum of a, b, and c equals the total number of units produced
-/
theorem production_line_b_units (a b c : ℕ) : 
  (a + b + c = 3000) → 
  (2 * b = a + c) → 
  b = 1000 := by
sorry

end production_line_b_units_l1658_165822


namespace min_value_of_a_l1658_165833

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := 2 * Real.log x + a / x^2

theorem min_value_of_a (a : ℝ) (h1 : a > 0) :
  (∀ x > 0, f a x ≥ 2) → a ≥ Real.exp 1 := by
  sorry

end min_value_of_a_l1658_165833


namespace difference_of_trailing_zeros_l1658_165830

def trailingZeros (n : ℕ) : ℕ :=
  (n / 5) + (n / 25) + (n / 125)

theorem difference_of_trailing_zeros : trailingZeros 300 - trailingZeros 280 = 5 := by
  sorry

end difference_of_trailing_zeros_l1658_165830


namespace root_coincidence_problem_l1658_165870

theorem root_coincidence_problem (a b : ℤ) (h1 : a ≠ 0) (h2 : b ≠ 0) : 
  (∃ r s : ℤ, (∀ x : ℝ, x^3 + a*x^2 + b*x + 16*a = (x - r)^2 * (x - s)) ∧
   (2*r + s = -a) ∧ (r^2 + 2*r*s = b) ∧ (r^2 * s = -16*a)) →
  |a*b| = 2128 :=
sorry

end root_coincidence_problem_l1658_165870


namespace seven_mondays_in_45_days_l1658_165807

/-- The number of Mondays in the first 45 days of a year that starts on a Monday. -/
def mondaysIn45Days (yearStartsOnMonday : Bool) : ℕ :=
  if yearStartsOnMonday then 7 else 0

/-- Theorem stating that if a year starts on a Monday, there are 7 Mondays in the first 45 days. -/
theorem seven_mondays_in_45_days (yearStartsOnMonday : Bool) 
  (h : yearStartsOnMonday = true) : mondaysIn45Days yearStartsOnMonday = 7 := by
  sorry

end seven_mondays_in_45_days_l1658_165807


namespace base_85_problem_l1658_165824

/-- Represents a number in base 85 --/
def BaseEightyFive : Type := List Nat

/-- Converts a number in base 85 to its decimal representation --/
def to_decimal (n : BaseEightyFive) : Nat :=
  sorry

/-- The specific number 3568432 in base 85 --/
def number : BaseEightyFive :=
  [3, 5, 6, 8, 4, 3, 2]

theorem base_85_problem (b : Int) 
  (h1 : 0 ≤ b) (h2 : b ≤ 19) 
  (h3 : (to_decimal number - b) % 17 = 0) : 
  b = 3 := by
  sorry

end base_85_problem_l1658_165824


namespace crosswalk_wait_probability_l1658_165839

/-- Represents the duration of the red light in seconds -/
def red_light_duration : ℝ := 40

/-- Represents the minimum waiting time in seconds -/
def min_wait_time : ℝ := 15

/-- Theorem: The probability of waiting at least 15 seconds for a green light
    when encountering a red light that lasts 40 seconds is 5/8 -/
theorem crosswalk_wait_probability :
  (red_light_duration - min_wait_time) / red_light_duration = 5 / 8 := by
  sorry

end crosswalk_wait_probability_l1658_165839


namespace cars_without_ac_l1658_165875

theorem cars_without_ac (total : ℕ) (min_racing : ℕ) (max_ac_no_racing : ℕ) 
  (h_total : total = 100)
  (h_min_racing : min_racing = 41)
  (h_max_ac_no_racing : max_ac_no_racing = 59) :
  total - (max_ac_no_racing + 0) = 41 := by
  sorry

end cars_without_ac_l1658_165875


namespace usual_walking_time_l1658_165829

/-- Given a constant distance and the fact that walking at 40% of usual speed takes 24 minutes more, 
    the usual time to cover the distance is 16 minutes. -/
theorem usual_walking_time (distance : ℝ) (usual_speed : ℝ) (usual_time : ℝ) 
  (h1 : usual_speed > 0)
  (h2 : usual_time > 0)
  (h3 : distance = usual_speed * usual_time)
  (h4 : distance = (0.4 * usual_speed) * (usual_time + 24)) :
  usual_time = 16 := by
sorry

end usual_walking_time_l1658_165829


namespace kite_area_in_regular_hexagon_l1658_165815

/-- The area of a kite-shaped region in a regular hexagon -/
theorem kite_area_in_regular_hexagon (side_length : ℝ) (h : side_length = 8) :
  let radius := side_length
  let angle := 120 * π / 180
  let kite_area := (1 / 2) * radius * radius * Real.sin angle
  kite_area = 16 * Real.sqrt 3 := by
sorry

end kite_area_in_regular_hexagon_l1658_165815


namespace unique_intersection_point_l1658_165813

def f (x : ℝ) : ℝ := x^3 + 6*x^2 + 28*x + 24

theorem unique_intersection_point :
  ∃! p : ℝ × ℝ, p.1 = f p.2 ∧ p.2 = f p.1 ∧ p = (-3, -3) :=
sorry

end unique_intersection_point_l1658_165813


namespace shara_shells_after_vacation_l1658_165893

/-- Calculates the total number of shells after a vacation -/
def total_shells_after_vacation (initial_shells : ℕ) (shells_per_day : ℕ) (days : ℕ) (fourth_day_shells : ℕ) : ℕ :=
  initial_shells + shells_per_day * days + fourth_day_shells

/-- Proves that Shara has 41 shells after her vacation -/
theorem shara_shells_after_vacation :
  total_shells_after_vacation 20 5 3 6 = 41 := by
  sorry

end shara_shells_after_vacation_l1658_165893


namespace lisa_marbles_problem_l1658_165823

/-- The minimum number of additional marbles needed -/
def min_additional_marbles (num_friends : ℕ) (initial_marbles : ℕ) : ℕ :=
  (num_friends * (num_friends + 1)) / 2 - initial_marbles

/-- Theorem stating the minimum number of additional marbles needed for Lisa's problem -/
theorem lisa_marbles_problem :
  min_additional_marbles 12 50 = 28 := by
  sorry

end lisa_marbles_problem_l1658_165823


namespace marble_probability_l1658_165859

theorem marble_probability (red blue green : ℕ) 
  (h_red : red = 4) 
  (h_blue : blue = 3) 
  (h_green : green = 6) : 
  (red + blue : ℚ) / (red + blue + green) = 7 / 13 := by
sorry

end marble_probability_l1658_165859


namespace distance_apex_to_circumsphere_center_l1658_165811

/-- Represents a rectangular pyramid with a frustum -/
structure RectangularPyramidWithFrustum where
  /-- Length of the rectangle base -/
  baseLength : ℝ
  /-- Width of the rectangle base -/
  baseWidth : ℝ
  /-- Height of the pyramid -/
  pyramidHeight : ℝ
  /-- Ratio of the volume of the smaller pyramid to the whole pyramid -/
  volumeRatio : ℝ

/-- Theorem stating the distance between the apex and the center of the frustum's circumsphere -/
theorem distance_apex_to_circumsphere_center
  (p : RectangularPyramidWithFrustum)
  (h1 : p.baseLength = 15)
  (h2 : p.baseWidth = 20)
  (h3 : p.pyramidHeight = 30)
  (h4 : p.volumeRatio = 1/9) :
  let xt := p.pyramidHeight - (1 - p.volumeRatio^(1/3)) * p.pyramidHeight +
            (p.baseLength^2 + p.baseWidth^2) / (18 * p.pyramidHeight)
  xt = 425/9 := by
  sorry

#check distance_apex_to_circumsphere_center

end distance_apex_to_circumsphere_center_l1658_165811


namespace cubic_sum_l1658_165828

theorem cubic_sum (a b c : ℝ) 
  (sum_eq : a + b + c = 7)
  (sum_prod_eq : a * b + a * c + b * c = 11)
  (prod_eq : a * b * c = -6) :
  a^3 + b^3 + c^3 = 223 := by
  sorry

end cubic_sum_l1658_165828


namespace second_white_given_first_red_l1658_165835

structure Bag where
  white : ℕ
  red : ℕ

def bagA : Bag := ⟨3, 2⟩
def bagB : Bag := ⟨2, 4⟩

def probChooseBag : ℚ := 1/2

def probRedFirst (b : Bag) : ℚ := b.red / (b.white + b.red)

def probWhiteSecondGivenRedFirst (b : Bag) : ℚ := b.white / (b.white + b.red - 1)

def probRedFirstThenWhite (b : Bag) : ℚ := probRedFirst b * probWhiteSecondGivenRedFirst b

theorem second_white_given_first_red :
  (probChooseBag * probRedFirstThenWhite bagA + probChooseBag * probRedFirstThenWhite bagB) /
  (probChooseBag * probRedFirst bagA + probChooseBag * probRedFirst bagB) = 17/32 := by
  sorry

end second_white_given_first_red_l1658_165835


namespace gold_value_calculation_l1658_165841

/-- The total value of gold for Legacy and Aleena -/
def total_gold_value (legacy_bars : ℕ) (aleena_difference : ℕ) (bar_value : ℕ) : ℕ :=
  (legacy_bars + (legacy_bars - aleena_difference)) * bar_value

/-- Theorem stating the total value of gold for Legacy and Aleena -/
theorem gold_value_calculation :
  total_gold_value 12 4 3500 = 70000 := by
  sorry

end gold_value_calculation_l1658_165841


namespace linear_function_decreasing_implies_negative_slope_l1658_165880

/-- A linear function y = mx + b -/
structure LinearFunction where
  m : ℝ
  b : ℝ

/-- Property that y decreases as x increases -/
def decreasing (f : LinearFunction) : Prop :=
  ∀ x₁ x₂ : ℝ, x₁ < x₂ → f.m * x₁ + f.b > f.m * x₂ + f.b

theorem linear_function_decreasing_implies_negative_slope (f : LinearFunction) 
    (h : f.b = 5) (dec : decreasing f) : f.m < 0 := by
  sorry

end linear_function_decreasing_implies_negative_slope_l1658_165880


namespace complex_cube_plus_one_in_first_quadrant_l1658_165878

theorem complex_cube_plus_one_in_first_quadrant : 
  let z : ℂ := 1 / Complex.I
  (z^3 + 1).re > 0 ∧ (z^3 + 1).im > 0 := by
  sorry

end complex_cube_plus_one_in_first_quadrant_l1658_165878


namespace coin_stack_solution_l1658_165825

/-- Represents the different types of coins --/
inductive CoinType
  | A
  | B
  | C
  | D

/-- Returns the thickness of a given coin type in millimeters --/
def coinThickness (t : CoinType) : ℚ :=
  match t with
  | CoinType.A => 21/10
  | CoinType.B => 18/10
  | CoinType.C => 12/10
  | CoinType.D => 2

/-- Represents a stack of coins --/
structure CoinStack :=
  (a b c d : ℕ)

/-- Calculates the height of a coin stack in millimeters --/
def stackHeight (s : CoinStack) : ℚ :=
  s.a * coinThickness CoinType.A +
  s.b * coinThickness CoinType.B +
  s.c * coinThickness CoinType.C +
  s.d * coinThickness CoinType.D

/-- The target height of the stack in millimeters --/
def targetHeight : ℚ := 18

theorem coin_stack_solution :
  ∃ (s : CoinStack), stackHeight s = targetHeight ∧
  s.a = 0 ∧ s.b = 0 ∧ s.c = 0 ∧ s.d = 9 :=
sorry

end coin_stack_solution_l1658_165825


namespace root_sum_power_property_l1658_165855

theorem root_sum_power_property (x₁ x₂ : ℂ) (n : ℤ) : 
  x₁^2 - 6*x₁ + 1 = 0 → 
  x₂^2 - 6*x₂ + 1 = 0 → 
  (∃ m : ℤ, x₁^n + x₂^n = m) ∧ 
  ¬(∃ k : ℤ, x₁^n + x₂^n = 5*k) := by
  sorry

end root_sum_power_property_l1658_165855


namespace sum_f_negative_l1658_165820

noncomputable def f (x : ℝ) : ℝ := x + x^3 + x^5

theorem sum_f_negative (x₁ x₂ x₃ : ℝ) 
  (h₁ : x₁ + x₂ < 0) 
  (h₂ : x₂ + x₃ < 0) 
  (h₃ : x₃ + x₁ < 0) : 
  f x₁ + f x₂ + f x₃ < 0 := by
  sorry

end sum_f_negative_l1658_165820


namespace cakes_baked_lunch_is_eight_l1658_165857

/-- The number of cakes baked during lunch today -/
def cakes_baked_lunch : ℕ := sorry

/-- The number of cakes sold during dinner -/
def cakes_sold_dinner : ℕ := 6

/-- The number of cakes baked yesterday -/
def cakes_baked_yesterday : ℕ := 3

/-- The number of cakes left -/
def cakes_left : ℕ := 2

/-- Theorem stating that the number of cakes baked during lunch today is 8 -/
theorem cakes_baked_lunch_is_eight :
  cakes_baked_lunch = 8 :=
by sorry

end cakes_baked_lunch_is_eight_l1658_165857


namespace ring_element_equality_l1658_165866

variable {A : Type*} [Ring A] [Finite A]

theorem ring_element_equality (a b : A) (h : (a * b - 1) * b = 0) : 
  b * (a * b - 1) = 0 := by
  sorry

end ring_element_equality_l1658_165866


namespace circle_symmetry_line_l1658_165897

/-- A circle in the 2D plane -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- A line in the 2D plane represented by ax + by + c = 0 -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Two points are symmetric with respect to a line -/
def symmetric_points (P Q : ℝ × ℝ) (l : Line) : Prop := sorry

/-- A point is on a circle -/
def on_circle (p : ℝ × ℝ) (c : Circle) : Prop := sorry

theorem circle_symmetry_line (C : Circle) (P Q : ℝ × ℝ) (m : ℝ) :
  C.center = (-1, 3) →
  C.radius = 3 →
  on_circle P C →
  on_circle Q C →
  symmetric_points P Q (Line.mk 1 m 4) →
  m = -1 := by sorry

end circle_symmetry_line_l1658_165897


namespace total_spent_equals_sum_l1658_165838

/-- The total amount Mike spent on car parts -/
def total_spent : ℝ := 224.87

/-- The amount Mike spent on speakers -/
def speakers_cost : ℝ := 118.54

/-- The amount Mike spent on new tires -/
def tires_cost : ℝ := 106.33

/-- Theorem stating that the total amount spent is the sum of speakers and tires costs -/
theorem total_spent_equals_sum : total_spent = speakers_cost + tires_cost := by
  sorry

end total_spent_equals_sum_l1658_165838


namespace set_equality_l1658_165884

theorem set_equality (A B X : Set α) 
  (h1 : A ∪ B ∪ X = A ∪ B) 
  (h2 : A ∩ X = A ∩ B) 
  (h3 : B ∩ X = A ∩ B) : 
  X = A ∩ B := by
  sorry

end set_equality_l1658_165884


namespace pencils_at_meeting_pencils_at_meeting_proof_l1658_165818

/-- The number of pencils brought to a committee meeting -/
theorem pencils_at_meeting : ℕ :=
  let associate_prof : ℕ → ℕ := λ x ↦ x  -- Number of associate professors
  let assistant_prof : ℕ → ℕ := λ x ↦ x  -- Number of assistant professors
  let total_people : ℕ := 7  -- Total number of people at the meeting
  let total_charts : ℕ := 11  -- Total number of charts brought to the meeting
  let pencils_per_associate : ℕ := 2  -- Pencils brought by each associate professor
  let pencils_per_assistant : ℕ := 1  -- Pencils brought by each assistant professor
  let charts_per_associate : ℕ := 1  -- Charts brought by each associate professor
  let charts_per_assistant : ℕ := 2  -- Charts brought by each assistant professor

  10  -- The theorem states that the number of pencils is 10

theorem pencils_at_meeting_proof :
  ∀ (x y : ℕ),
  x + y = total_people →
  charts_per_associate * x + charts_per_assistant * y = total_charts →
  pencils_per_associate * x + pencils_per_assistant * y = pencils_at_meeting :=
by
  sorry

#check pencils_at_meeting
#check pencils_at_meeting_proof

end pencils_at_meeting_pencils_at_meeting_proof_l1658_165818


namespace function_equality_l1658_165808

theorem function_equality (f : ℝ → ℝ) 
  (h1 : ∀ x : ℝ, f x ≤ x) 
  (h2 : ∀ x y : ℝ, f (x + y) ≤ f x + f y) : 
  ∀ x : ℝ, f x = x := by
sorry

end function_equality_l1658_165808


namespace max_silver_tokens_l1658_165887

/-- Represents the number of tokens Alex has -/
structure Tokens where
  red : ℕ
  blue : ℕ
  silver : ℕ

/-- Represents the exchange rules for the booths -/
structure ExchangeRule where
  redCost : ℕ
  blueCost : ℕ
  redGain : ℕ
  blueGain : ℕ
  silverGain : ℕ

/-- Defines if an exchange is possible given the current tokens and an exchange rule -/
def canExchange (t : Tokens) (r : ExchangeRule) : Prop :=
  t.red ≥ r.redCost ∧ t.blue ≥ r.blueCost

/-- Applies an exchange rule to the current tokens -/
def applyExchange (t : Tokens) (r : ExchangeRule) : Tokens :=
  { red := t.red - r.redCost + r.redGain,
    blue := t.blue - r.blueCost + r.blueGain,
    silver := t.silver + r.silverGain }

/-- Theorem: The maximum number of silver tokens Alex can obtain is 23 -/
theorem max_silver_tokens :
  ∀ (initial : Tokens)
    (rule1 rule2 : ExchangeRule),
  initial.red = 60 ∧ initial.blue = 90 ∧ initial.silver = 0 →
  rule1 = { redCost := 3, blueCost := 0, redGain := 0, blueGain := 2, silverGain := 1 } →
  rule2 = { redCost := 0, blueCost := 4, redGain := 1, blueGain := 0, silverGain := 1 } →
  ∃ (final : Tokens),
    (∀ t, (canExchange t rule1 ∨ canExchange t rule2) → t.silver ≤ final.silver) ∧
    final.silver = 23 :=
by sorry

end max_silver_tokens_l1658_165887


namespace xy_inequality_l1658_165885

theorem xy_inequality (x y : ℝ) (h : (x + 1) * (y + 2) = 8) :
  (x * y - 10)^2 ≥ 64 ∧
  ((x * y - 10)^2 = 64 ↔ (x = 1 ∧ y = 2) ∨ (x = -3 ∧ y = -6)) :=
by sorry

end xy_inequality_l1658_165885


namespace complex_modulus_squared_l1658_165868

theorem complex_modulus_squared (w : ℂ) (h : w + Complex.abs w = 4 + 5*I) : 
  Complex.abs w ^ 2 = 1681 / 64 := by
  sorry

end complex_modulus_squared_l1658_165868


namespace polynomial_multiplication_simplification_l1658_165826

theorem polynomial_multiplication_simplification (y : ℝ) : 
  (3*y - 2) * (5*y^12 + 3*y^9 + 5*y^7 + 2*y^5) = 
  15*y^13 - 10*y^12 + 9*y^10 - 6*y^9 + 15*y^8 - 10*y^7 + 6*y^6 - 4*y^5 := by
sorry

end polynomial_multiplication_simplification_l1658_165826


namespace seating_arrangements_l1658_165809

def factorial (n : ℕ) : ℕ := (List.range n).foldl (· * ·) 1

def totalArrangements (n : ℕ) : ℕ := factorial n

def adjacentArrangements (n : ℕ) : ℕ := factorial (n - 1) * factorial 2

theorem seating_arrangements (n : ℕ) (h : n = 8) :
  totalArrangements n - adjacentArrangements n = 30240 := by sorry

end seating_arrangements_l1658_165809


namespace t_range_for_strictly_decreasing_function_l1658_165879

theorem t_range_for_strictly_decreasing_function 
  (f : ℝ → ℝ) (h_decreasing : ∀ x y, x < y → f y < f x) :
  ∀ t : ℝ, f (t^2) - f t < 0 → t < 0 ∨ t > 1 :=
by sorry

end t_range_for_strictly_decreasing_function_l1658_165879


namespace prime_quadratic_residue_equivalence_l1658_165851

theorem prime_quadratic_residue_equivalence (p : ℕ) (hp : Nat.Prime p) :
  (∃ α : ℕ+, p ∣ α * (α - 1) + 3) ↔ (∃ β : ℕ+, p ∣ β * (β - 1) + 25) := by
  sorry

end prime_quadratic_residue_equivalence_l1658_165851


namespace bicycle_wheel_revolutions_l1658_165846

theorem bicycle_wheel_revolutions 
  (front_radius : ℝ) 
  (back_radius : ℝ) 
  (front_revolutions : ℝ) 
  (h1 : front_radius = 3) 
  (h2 : back_radius = 6 / 12) 
  (h3 : front_revolutions = 150) :
  (2 * Real.pi * front_radius * front_revolutions) / (2 * Real.pi * back_radius) = 900 := by
  sorry

end bicycle_wheel_revolutions_l1658_165846


namespace hash_difference_l1658_165831

-- Define the # operation
def hash (x y : ℝ) : ℝ := x * y - 3 * x

-- State the theorem
theorem hash_difference : hash 8 3 - hash 3 8 = -15 := by sorry

end hash_difference_l1658_165831


namespace points_in_quadrants_I_and_II_l1658_165814

-- Define the set of points satisfying the inequalities
def S : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.2 > p.1 + 1 ∧ p.2 > 3 - 2*p.1}

-- Theorem stating that all points in S are in Quadrants I or II
theorem points_in_quadrants_I_and_II : 
  ∀ p ∈ S, (p.1 > 0 ∧ p.2 > 0) ∨ (p.1 < 0 ∧ p.2 > 0) := by
  sorry

-- Helper lemma: All points in S have positive y-coordinate
lemma points_have_positive_y : 
  ∀ p ∈ S, p.2 > 0 := by
  sorry

end points_in_quadrants_I_and_II_l1658_165814


namespace tyrones_pennies_l1658_165894

/-- The number of pennies Tyrone found -/
def pennies : ℕ := sorry

/-- The value of a penny in dollars -/
def penny_value : ℚ := 1 / 100

/-- The total value of Tyrone's money in dollars -/
def total_value : ℚ := 13

/-- The value of Tyrone's money excluding pennies -/
def value_without_pennies : ℚ :=
  2 * 1 + -- two $1 bills
  1 * 5 + -- one $5 bill
  13 * (1 / 4) + -- 13 quarters
  20 * (1 / 10) + -- 20 dimes
  8 * (1 / 20) -- 8 nickels

theorem tyrones_pennies :
  pennies * penny_value = total_value - value_without_pennies ∧
  pennies = 35 := by sorry

end tyrones_pennies_l1658_165894


namespace range_of_m_l1658_165848

theorem range_of_m (x y m : ℝ) 
  (hx : x > 0) 
  (hy : y > 0) 
  (h_eq : 1/x + 4/y = 1) 
  (h_ineq : ∀ m : ℝ, x + y > m^2 + 8*m) : 
  -9 < m ∧ m < 1 :=
sorry

end range_of_m_l1658_165848


namespace male_workers_percentage_l1658_165899

theorem male_workers_percentage (female_workers : ℝ) (male_workers : ℝ) :
  male_workers = 0.6 * female_workers →
  (female_workers - male_workers) / female_workers = 0.4 :=
by
  sorry

end male_workers_percentage_l1658_165899


namespace third_vertex_y_coord_value_l1658_165860

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- An equilateral triangle with two vertices given -/
structure EquilateralTriangle where
  v1 : Point
  v2 : Point
  v3 : Point
  is_equilateral : True  -- This is a placeholder for the equilateral property
  third_vertex_in_first_quadrant : v3.x > 0 ∧ v3.y > 0

/-- The y-coordinate of the third vertex of an equilateral triangle -/
def third_vertex_y_coord (t : EquilateralTriangle) : ℝ :=
  t.v3.y

/-- The theorem stating the y-coordinate of the third vertex -/
theorem third_vertex_y_coord_value (t : EquilateralTriangle) 
    (h1 : t.v1 = ⟨2, 3⟩) 
    (h2 : t.v2 = ⟨10, 3⟩) : 
  third_vertex_y_coord t = 3 + 4 * Real.sqrt 3 := by
  sorry

#check third_vertex_y_coord_value

end third_vertex_y_coord_value_l1658_165860


namespace intersection_distance_implies_a_bound_l1658_165871

/-- Given a line and a circle with parameter a, if the distance between
    their intersection points is at least 2√3, then a ≤ -4/3 -/
theorem intersection_distance_implies_a_bound
  (a : ℝ)
  (line : ℝ → ℝ → Prop)
  (circle : ℝ → ℝ → Prop)
  (M N : ℝ × ℝ)
  (h_line : ∀ x y, line x y ↔ a * x - y + 3 = 0)
  (h_circle : ∀ x y, circle x y ↔ (x - 2)^2 + (y - a)^2 = 4)
  (h_intersection : line M.1 M.2 ∧ circle M.1 M.2 ∧ line N.1 N.2 ∧ circle N.1 N.2)
  (h_distance : (M.1 - N.1)^2 + (M.2 - N.2)^2 ≥ 12) :
  a ≤ -4/3 :=
sorry

end intersection_distance_implies_a_bound_l1658_165871


namespace arithmetic_calculations_l1658_165863

theorem arithmetic_calculations :
  (-4 - 4 = -8) ∧ ((-32) / 4 = -8) ∧ (-(-2)^3 = 8) := by sorry

end arithmetic_calculations_l1658_165863


namespace farmers_market_sales_l1658_165874

/-- The farmers' market sales problem -/
theorem farmers_market_sales
  (total_earnings : ℕ)
  (broccoli_sales : ℕ)
  (carrot_sales : ℕ)
  (spinach_sales : ℕ)
  (cauliflower_sales : ℕ)
  (h1 : total_earnings = 380)
  (h2 : broccoli_sales = 57)
  (h3 : carrot_sales = 2 * broccoli_sales)
  (h4 : spinach_sales = carrot_sales / 2 + 16)
  (h5 : total_earnings = broccoli_sales + carrot_sales + spinach_sales + cauliflower_sales) :
  cauliflower_sales = 136 := by
  sorry


end farmers_market_sales_l1658_165874


namespace det_B_equals_four_l1658_165803

theorem det_B_equals_four (b c : ℝ) (B : Matrix (Fin 2) (Fin 2) ℝ) :
  B = ![![b, 2], ![-3, c]] →
  B + 2 * B⁻¹ = 0 →
  Matrix.det B = 4 := by
sorry

end det_B_equals_four_l1658_165803


namespace inequality_of_squares_and_sum_l1658_165804

theorem inequality_of_squares_and_sum (a b c : ℝ) :
  Real.sqrt (a^2 + b^2) + Real.sqrt (b^2 + c^2) + Real.sqrt (c^2 + a^2) ≥ Real.sqrt 2 * (a + b + c) := by
  sorry

end inequality_of_squares_and_sum_l1658_165804


namespace integral_reciprocal_x_xplus1_l1658_165840

theorem integral_reciprocal_x_xplus1 : 
  ∫ x in (1 : ℝ)..2, 1 / (x * (x + 1)) = Real.log (4 / 3) := by sorry

end integral_reciprocal_x_xplus1_l1658_165840


namespace quadrilateral_area_l1658_165892

-- Define the quadrilateral ABCD
structure Quadrilateral :=
  (A B C D : ℝ × ℝ)

-- Define the properties of the quadrilateral
def is_right_angled_at_B_and_D (q : Quadrilateral) : Prop := sorry
def diagonal_length (q : Quadrilateral) (p1 p2 : ℝ × ℝ) : ℝ := sorry
def side_length (q : Quadrilateral) (p1 p2 : ℝ × ℝ) : ℝ := sorry
def area (q : Quadrilateral) : ℝ := sorry

-- Theorem statement
theorem quadrilateral_area 
  (q : Quadrilateral)
  (h1 : is_right_angled_at_B_and_D q)
  (h2 : diagonal_length q q.A q.C = 5)
  (h3 : side_length q q.B q.C = 4)
  (h4 : side_length q q.A q.D = 3) :
  area q = 12 := by sorry

end quadrilateral_area_l1658_165892


namespace andy_position_after_2023_turns_l1658_165872

/-- Andy's position on the coordinate plane -/
structure Position where
  x : Int
  y : Int

/-- Direction Andy is facing -/
inductive Direction
  | North
  | East
  | South
  | West

/-- Andy's state, including position and direction -/
structure AndyState where
  pos : Position
  dir : Direction

/-- Function to update Andy's state after one move -/
def move (state : AndyState) (distance : Int) : AndyState :=
  sorry

/-- Function to turn Andy 90° right -/
def turnRight (dir : Direction) : Direction :=
  sorry

/-- Function to simulate Andy's movement for a given number of turns -/
def simulateAndy (initialState : AndyState) (turns : Nat) : Position :=
  sorry

theorem andy_position_after_2023_turns :
  let initialState : AndyState := { pos := { x := 10, y := -10 }, dir := Direction.North }
  let finalPosition := simulateAndy initialState 2023
  finalPosition = { x := 1022, y := 1 } := by
  sorry

end andy_position_after_2023_turns_l1658_165872


namespace triangle_count_theorem_l1658_165832

/-- Represents a rectangle divided into columns and rows with diagonal lines -/
structure DividedRectangle where
  columns : Nat
  rows : Nat

/-- Counts the number of triangles in a divided rectangle -/
def count_triangles (rect : DividedRectangle) : Nat :=
  let smallest_triangles := rect.columns * rect.rows * 2
  let small_isosceles := rect.columns + rect.rows * 2
  let medium_right := (rect.columns / 2) * rect.rows * 2
  let large_isosceles := rect.columns / 2
  smallest_triangles + small_isosceles + medium_right + large_isosceles

/-- The main theorem stating the number of triangles in the specific rectangle -/
theorem triangle_count_theorem (rect : DividedRectangle) 
    (h_columns : rect.columns = 8) 
    (h_rows : rect.rows = 2) : 
  count_triangles rect = 76 := by
  sorry

#eval count_triangles ⟨8, 2⟩

end triangle_count_theorem_l1658_165832


namespace range_of_m_l1658_165891

theorem range_of_m (x y m : ℝ) 
  (hx : x > 0) 
  (hy : y > 0) 
  (h_eq : 2/x + 1/y = 1) 
  (h_ineq : ∀ (x y : ℝ), x > 0 → y > 0 → 2/x + 1/y = 1 → x + 2*y > m^2 + 2*m) : 
  -4 < m ∧ m < 2 := by
sorry

end range_of_m_l1658_165891


namespace second_grade_students_selected_l1658_165800

/-- Given a school with 3300 students and a ratio of 12:10:11 for first, second, and third grades,
    prove that when 66 students are randomly selected using stratified sampling,
    the number of second-grade students selected is 20. -/
theorem second_grade_students_selected
  (total_students : ℕ)
  (first_grade_ratio second_grade_ratio third_grade_ratio : ℕ)
  (selected_students : ℕ)
  (h1 : total_students = 3300)
  (h2 : first_grade_ratio = 12)
  (h3 : second_grade_ratio = 10)
  (h4 : third_grade_ratio = 11)
  (h5 : selected_students = 66) :
  (second_grade_ratio : ℚ) / (first_grade_ratio + second_grade_ratio + third_grade_ratio) * selected_students = 20 := by
  sorry

end second_grade_students_selected_l1658_165800


namespace button_probability_l1658_165847

/-- Represents a jar containing buttons -/
structure Jar where
  red : ℕ
  blue : ℕ

/-- The probability of selecting a red button from a jar -/
def redProbability (j : Jar) : ℚ :=
  j.red / (j.red + j.blue)

theorem button_probability (initialJarA initialJarB finalJarA finalJarB : Jar) :
  initialJarA.red = 5 →
  initialJarA.blue = 10 →
  initialJarB.red = 0 →
  initialJarB.blue = 0 →
  finalJarA.red + finalJarB.red = initialJarA.red →
  finalJarA.blue + finalJarB.blue = initialJarA.blue →
  finalJarB.red = finalJarB.blue / 2 →
  finalJarA.red + finalJarA.blue = (3 * (initialJarA.red + initialJarA.blue)) / 5 →
  redProbability finalJarA = 1/3 ∧ redProbability finalJarB = 1/3 ∧
  redProbability finalJarA * redProbability finalJarB = 1/9 := by
  sorry


end button_probability_l1658_165847


namespace golden_section_proportion_l1658_165882

/-- Golden section point of a line segment -/
def is_golden_section_point (A B C : ℝ) : Prop :=
  (B - A) / (C - A) = (C - A) / (B - C)

theorem golden_section_proportion (A B C : ℝ) 
  (h1 : is_golden_section_point A B C) 
  (h2 : C - A > B - C) : 
  (B - A) / (C - A) = (C - A) / (B - C) := by
  sorry

end golden_section_proportion_l1658_165882


namespace sqrt_x_div_sqrt_y_l1658_165886

theorem sqrt_x_div_sqrt_y (x y : ℝ) :
  (1/3)^2 + (1/4)^2 = (17*x/60) * ((1/5)^2 + (1/6)^2) →
  Real.sqrt x / Real.sqrt y = (25/2) * Real.sqrt (60/1037) := by
sorry

end sqrt_x_div_sqrt_y_l1658_165886


namespace cosine_intersection_theorem_l1658_165881

theorem cosine_intersection_theorem (f : ℝ → ℝ) (θ : ℝ) : 
  (∀ x ≥ 0, f x = |Real.cos x|) →
  (∃ l : ℝ → ℝ, l 0 = 0 ∧ (∃ a b c d : ℝ, 0 ≤ a ∧ a < b ∧ b < c ∧ c < d ∧ d = θ ∧
    f a = l a ∧ f b = l b ∧ f c = l c ∧ f d = l d ∧
    ∀ x : ℝ, x ≥ 0 → x ≠ a → x ≠ b → x ≠ c → x ≠ d → f x ≠ l x)) →
  ((1 + θ^2) * Real.sin (2*θ)) / θ = -2 := by
sorry

end cosine_intersection_theorem_l1658_165881


namespace work_completion_time_l1658_165802

/-- Given a work that can be completed by two workers A and B, this theorem proves
    the number of days B needs to complete the work alone, given certain conditions. -/
theorem work_completion_time (W : ℝ) (h_pos : W > 0) : 
  (∃ (work_A work_B : ℝ),
    -- A can finish the work in 21 days
    21 * work_A = W ∧
    -- B worked for 10 days
    10 * work_B + 
    -- A finished the remaining work in 7 days
    7 * work_A = W) →
  -- B can finish the work in 15 days
  15 * work_B = W :=
by
  sorry


end work_completion_time_l1658_165802


namespace evening_milk_is_380_l1658_165844

/-- Represents the milk production and sales for Aunt May's farm --/
structure MilkProduction where
  morning : ℕ
  evening : ℕ
  leftover : ℕ
  sold : ℕ
  remaining : ℕ

/-- Calculates the evening milk production given the other parameters --/
def calculate_evening_milk (mp : MilkProduction) : ℕ :=
  mp.remaining + mp.sold - mp.morning - mp.leftover

/-- Theorem stating that the evening milk production is 380 gallons --/
theorem evening_milk_is_380 (mp : MilkProduction) 
  (h1 : mp.morning = 365)
  (h2 : mp.leftover = 15)
  (h3 : mp.sold = 612)
  (h4 : mp.remaining = 148) :
  calculate_evening_milk mp = 380 := by
  sorry

#eval calculate_evening_milk { morning := 365, evening := 0, leftover := 15, sold := 612, remaining := 148 }

end evening_milk_is_380_l1658_165844


namespace sin_thirteen_pi_thirds_l1658_165877

theorem sin_thirteen_pi_thirds : Real.sin (13 * π / 3) = Real.sqrt 3 / 2 := by
  sorry

end sin_thirteen_pi_thirds_l1658_165877


namespace reciprocal_of_negative_half_l1658_165861

theorem reciprocal_of_negative_half : ((-1/2)⁻¹ : ℚ) = -2 := by sorry

end reciprocal_of_negative_half_l1658_165861


namespace trigonometric_expression_equals_one_l1658_165827

theorem trigonometric_expression_equals_one : 
  (Real.sin (20 * π / 180) * Real.cos (10 * π / 180) + Real.cos (160 * π / 180) * Real.cos (110 * π / 180)) /
  (Real.sin (24 * π / 180) * Real.cos (6 * π / 180) + Real.cos (156 * π / 180) * Real.cos (106 * π / 180)) = 1 := by
  sorry

end trigonometric_expression_equals_one_l1658_165827


namespace max_sum_of_factors_l1658_165890

theorem max_sum_of_factors (A B C : ℕ+) : 
  A ≠ B → B ≠ C → A ≠ C → A * B * C = 3003 → 
  ∀ (X Y Z : ℕ+), X ≠ Y → Y ≠ Z → X ≠ Z → X * Y * Z = 3003 → 
  A + B + C ≤ 105 ∧ (∃ (P Q R : ℕ+), P ≠ Q ∧ Q ≠ R ∧ P ≠ R ∧ P * Q * R = 3003 ∧ P + Q + R = 105) :=
by sorry

end max_sum_of_factors_l1658_165890


namespace xy_equation_solution_l1658_165883

theorem xy_equation_solution (x y : ℕ+) (p q : ℕ) :
  x ≥ y →
  x * y - (x + y) = 2 * p + q →
  p = Nat.gcd x y →
  q = Nat.lcm x y →
  ((x = 9 ∧ y = 3) ∨ (x = 5 ∧ y = 5)) :=
by sorry

end xy_equation_solution_l1658_165883


namespace phone_plan_ratio_l1658_165837

/-- Given Mandy's phone data plan details, prove the ratio of promotional rate to normal rate -/
theorem phone_plan_ratio : 
  ∀ (normal_rate promotional_rate : ℚ),
  normal_rate = 30 →
  promotional_rate + 2 * normal_rate + (normal_rate + 15) + 2 * normal_rate = 175 →
  promotional_rate / normal_rate = 1 / 3 := by
sorry

end phone_plan_ratio_l1658_165837


namespace max_value_xyz_l1658_165869

theorem max_value_xyz (x y z : ℝ) (h_pos : x > 0 ∧ y > 0 ∧ z > 0) (h_sum : x + y + z = 1) :
  x * y^2 * z^3 ≤ 1 / 432 :=
sorry

end max_value_xyz_l1658_165869


namespace sum_mod_thirteen_l1658_165896

theorem sum_mod_thirteen : (1230 + 1231 + 1232 + 1233 + 1234) % 13 = 0 := by
  sorry

end sum_mod_thirteen_l1658_165896


namespace corner_sum_equals_164_l1658_165834

-- Define the size of the checkerboard
def boardSize : Nat := 9

-- Define a function to get the number at a specific position
def getNumber (row : Nat) (col : Nat) : Nat :=
  if row % 2 = 1 then
    (row - 1) * boardSize + col
  else
    row * boardSize - col + 1

-- Theorem statement
theorem corner_sum_equals_164 :
  getNumber 1 1 + getNumber 1 boardSize + getNumber boardSize 1 + getNumber boardSize boardSize = 164 := by
  sorry

end corner_sum_equals_164_l1658_165834
