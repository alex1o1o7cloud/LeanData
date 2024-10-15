import Mathlib

namespace NUMINAMATH_CALUDE_theme_parks_sum_l2119_211967

theorem theme_parks_sum (jamestown venice marina_del_ray : ℕ) : 
  jamestown = 20 →
  venice = jamestown + 25 →
  marina_del_ray = jamestown + 50 →
  jamestown + venice + marina_del_ray = 135 := by
  sorry

end NUMINAMATH_CALUDE_theme_parks_sum_l2119_211967


namespace NUMINAMATH_CALUDE_min_tangent_length_l2119_211991

/-- The minimum length of a tangent from a point on the line y = x + 2 to the circle (x - 4)² + (y + 2)² = 1 is √31. -/
theorem min_tangent_length (x y : ℝ) : 
  let line := {(x, y) | y = x + 2}
  let circle := {(x, y) | (x - 4)^2 + (y + 2)^2 = 1}
  let center := (4, -2)
  let dist_to_line (p : ℝ × ℝ) := |p.1 + p.2 + 2| / Real.sqrt 2
  let min_dist := dist_to_line center
  let tangent_length := Real.sqrt (min_dist^2 - 1)
  tangent_length = Real.sqrt 31 := by sorry

end NUMINAMATH_CALUDE_min_tangent_length_l2119_211991


namespace NUMINAMATH_CALUDE_a_range_l2119_211917

/-- The function f(x) as defined in the problem -/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x < 1 then a^x else x^2 + 4/x + a * Real.log x

/-- The theorem statement -/
theorem a_range (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) 
  (h3 : ∀ x y : ℝ, x < y → f a x < f a y) : 
  2 ≤ a ∧ a ≤ 5 :=
sorry

end NUMINAMATH_CALUDE_a_range_l2119_211917


namespace NUMINAMATH_CALUDE_circle_center_l2119_211950

/-- The equation of a circle in the form (x - h)^2 + (y - k)^2 = r^2,
    where (h, k) is the center and r is the radius. -/
def CircleEquation (h k r : ℝ) (x y : ℝ) : Prop :=
  (x - h)^2 + (y - k)^2 = r^2

/-- The given equation of the circle -/
def GivenCircleEquation (x y : ℝ) : Prop :=
  x^2 - 8*x + y^2 - 4*y = 16

theorem circle_center :
  ∃ r, ∀ x y, GivenCircleEquation x y ↔ CircleEquation 4 2 r x y :=
sorry

end NUMINAMATH_CALUDE_circle_center_l2119_211950


namespace NUMINAMATH_CALUDE_class_test_probabilities_l2119_211987

theorem class_test_probabilities (p_first p_second p_both : ℝ) 
  (h1 : p_first = 0.63)
  (h2 : p_second = 0.49)
  (h3 : p_both = 0.32) :
  1 - (p_first + p_second - p_both) = 0.20 := by
  sorry

end NUMINAMATH_CALUDE_class_test_probabilities_l2119_211987


namespace NUMINAMATH_CALUDE_initial_to_doubled_ratio_l2119_211933

theorem initial_to_doubled_ratio (x : ℝ) (h : 3 * (2 * x + 13) = 93) : 
  x / (2 * x) = 1 / 2 := by
sorry

end NUMINAMATH_CALUDE_initial_to_doubled_ratio_l2119_211933


namespace NUMINAMATH_CALUDE_mod_thirteen_equiv_l2119_211901

theorem mod_thirteen_equiv (n : ℤ) : 0 ≤ n ∧ n ≤ 12 ∧ n ≡ -2345 [ZMOD 13] → n = 8 := by
  sorry

end NUMINAMATH_CALUDE_mod_thirteen_equiv_l2119_211901


namespace NUMINAMATH_CALUDE_book_selection_theorem_l2119_211926

theorem book_selection_theorem (chinese_books math_books sports_books : ℕ) 
  (h1 : chinese_books = 4) 
  (h2 : math_books = 5) 
  (h3 : sports_books = 6) : 
  (chinese_books + math_books + sports_books = 15) ∧ 
  (chinese_books * math_books * sports_books = 120) := by
  sorry

end NUMINAMATH_CALUDE_book_selection_theorem_l2119_211926


namespace NUMINAMATH_CALUDE_intersection_is_empty_l2119_211973

def A : Set ℝ := {x | x > 1}
def B : Set ℝ := {x | -1 ≤ x ∧ x ≤ 1}

theorem intersection_is_empty : A ∩ B = ∅ := by
  sorry

end NUMINAMATH_CALUDE_intersection_is_empty_l2119_211973


namespace NUMINAMATH_CALUDE_inhabitable_earth_surface_l2119_211975

theorem inhabitable_earth_surface (total_surface area_land area_inhabitable : ℝ) :
  area_land = (1 / 3 : ℝ) * total_surface →
  area_inhabitable = (3 / 4 : ℝ) * area_land →
  area_inhabitable = (1 / 4 : ℝ) * total_surface :=
by
  sorry

end NUMINAMATH_CALUDE_inhabitable_earth_surface_l2119_211975


namespace NUMINAMATH_CALUDE_length_to_height_ratio_l2119_211962

/-- Represents the dimensions of a box -/
structure BoxDimensions where
  height : ℝ
  width : ℝ
  length : ℝ

/-- Calculates the volume of a box given its dimensions -/
def volume (b : BoxDimensions) : ℝ :=
  b.height * b.width * b.length

/-- Theorem stating the ratio of length to height for a specific box -/
theorem length_to_height_ratio (b : BoxDimensions) :
  b.height = 12 →
  b.length = 4 * b.width →
  volume b = 3888 →
  b.length / b.height = 3 := by
  sorry

end NUMINAMATH_CALUDE_length_to_height_ratio_l2119_211962


namespace NUMINAMATH_CALUDE_chess_tournament_win_loss_difference_l2119_211961

theorem chess_tournament_win_loss_difference 
  (total_games : ℕ) 
  (total_score : ℚ) 
  (wins : ℕ) 
  (losses : ℕ) 
  (draws : ℕ) :
  total_games = 42 →
  total_score = 30 →
  wins + losses + draws = total_games →
  (wins : ℚ) + (1/2 : ℚ) * draws = total_score →
  wins - losses = 18 :=
by sorry

end NUMINAMATH_CALUDE_chess_tournament_win_loss_difference_l2119_211961


namespace NUMINAMATH_CALUDE_red_card_events_l2119_211952

-- Define the set of colors
inductive Color : Type
| Red | Black | White | Blue

-- Define the set of individuals
inductive Person : Type
| A | B | C | D

-- Define a distribution as a function from Person to Color
def Distribution := Person → Color

-- Define the event "A receives the red card"
def A_gets_red (d : Distribution) : Prop := d Person.A = Color.Red

-- Define the event "B receives the red card"
def B_gets_red (d : Distribution) : Prop := d Person.B = Color.Red

-- Theorem: A_gets_red and B_gets_red are mutually exclusive but not complementary
theorem red_card_events (d : Distribution) :
  (¬ (A_gets_red d ∧ B_gets_red d)) ∧
  (∃ (d : Distribution), ¬ A_gets_red d ∧ ¬ B_gets_red d) :=
by sorry

end NUMINAMATH_CALUDE_red_card_events_l2119_211952


namespace NUMINAMATH_CALUDE_domain_of_f_l2119_211983

def f (x : ℝ) : ℝ := (2 * x - 3) ^ (1/3) + (9 - x) ^ (1/2)

theorem domain_of_f :
  {x : ℝ | ∃ y, f x = y} = Set.Iic 9 := by sorry

end NUMINAMATH_CALUDE_domain_of_f_l2119_211983


namespace NUMINAMATH_CALUDE_perfect_negative_correlation_l2119_211915

/-- Represents a pair of data points (x, y) -/
structure DataPoint where
  x : ℝ
  y : ℝ

/-- Calculates the sample correlation coefficient for a list of data points -/
def sampleCorrelationCoefficient (data : List DataPoint) : ℝ :=
  sorry

/-- Theorem: For any set of paired sample data that fall on a straight line with negative slope,
    the sample correlation coefficient is -1 -/
theorem perfect_negative_correlation 
  (data : List DataPoint) 
  (h_line : ∃ (m : ℝ) (b : ℝ), m < 0 ∧ ∀ (point : DataPoint), point ∈ data → point.y = m * point.x + b) :
  sampleCorrelationCoefficient data = -1 :=
sorry

end NUMINAMATH_CALUDE_perfect_negative_correlation_l2119_211915


namespace NUMINAMATH_CALUDE_sqrt_sum_simplification_l2119_211960

theorem sqrt_sum_simplification : ∃ (a b c : ℕ+),
  (Real.sqrt 8 + (Real.sqrt 8)⁻¹ + Real.sqrt 9 + (Real.sqrt 9)⁻¹ = (a * Real.sqrt 8 + b * Real.sqrt 9) / c) ∧
  (∀ (a' b' c' : ℕ+), 
    (Real.sqrt 8 + (Real.sqrt 8)⁻¹ + Real.sqrt 9 + (Real.sqrt 9)⁻¹ = (a' * Real.sqrt 8 + b' * Real.sqrt 9) / c') →
    c ≤ c') ∧
  (a + b + c = 158) :=
sorry

end NUMINAMATH_CALUDE_sqrt_sum_simplification_l2119_211960


namespace NUMINAMATH_CALUDE_valid_sequences_characterization_l2119_211902

/-- Represents the possible weather observations: Plus for no rain, Minus for rain -/
inductive WeatherObservation
| Plus : WeatherObservation
| Minus : WeatherObservation

/-- Represents a sequence of three weather observations -/
structure ObservationSequence :=
  (first : WeatherObservation)
  (second : WeatherObservation)
  (third : WeatherObservation)

/-- Determines if a sequence is valid based on the third student's rule -/
def isValidSequence (seq : ObservationSequence) : Prop :=
  match seq.third with
  | WeatherObservation.Minus => 
      (seq.first = WeatherObservation.Minus ∧ seq.second = WeatherObservation.Minus) ∨
      (seq.first = WeatherObservation.Minus ∧ seq.second = WeatherObservation.Plus) ∨
      (seq.first = WeatherObservation.Plus ∧ seq.second = WeatherObservation.Minus)
  | WeatherObservation.Plus =>
      (seq.first = WeatherObservation.Plus ∧ seq.second = WeatherObservation.Plus) ∨
      (seq.first = WeatherObservation.Minus ∧ seq.second = WeatherObservation.Plus)

/-- The set of all valid observation sequences -/
def validSequences : Set ObservationSequence :=
  { seq | isValidSequence seq }

theorem valid_sequences_characterization :
  validSequences = {
    ⟨WeatherObservation.Plus, WeatherObservation.Plus, WeatherObservation.Plus⟩,
    ⟨WeatherObservation.Minus, WeatherObservation.Plus, WeatherObservation.Plus⟩,
    ⟨WeatherObservation.Minus, WeatherObservation.Minus, WeatherObservation.Plus⟩,
    ⟨WeatherObservation.Minus, WeatherObservation.Minus, WeatherObservation.Minus⟩
  } := by
  sorry

#check valid_sequences_characterization

end NUMINAMATH_CALUDE_valid_sequences_characterization_l2119_211902


namespace NUMINAMATH_CALUDE_complement_of_union_sets_l2119_211910

open Set

theorem complement_of_union_sets (A B : Set ℝ) :
  A = {x : ℝ | x < 1} →
  B = {x : ℝ | x > 3} →
  (A ∪ B)ᶜ = {x : ℝ | 1 ≤ x ∧ x ≤ 3} := by
  sorry

end NUMINAMATH_CALUDE_complement_of_union_sets_l2119_211910


namespace NUMINAMATH_CALUDE_midpoint_after_translation_l2119_211943

/-- Given a triangle DJH with vertices D(2, 3), J(3, 7), and H(7, 3),
    prove that the midpoint of D'H' after translating the triangle
    3 units right and 1 unit down is (7.5, 2). -/
theorem midpoint_after_translation :
  let D : ℝ × ℝ := (2, 3)
  let J : ℝ × ℝ := (3, 7)
  let H : ℝ × ℝ := (7, 3)
  let translate (p : ℝ × ℝ) : ℝ × ℝ := (p.1 + 3, p.2 - 1)
  let D' := translate D
  let H' := translate H
  let midpoint (p q : ℝ × ℝ) : ℝ × ℝ := ((p.1 + q.1) / 2, (p.2 + q.2) / 2)
  midpoint D' H' = (7.5, 2) :=
by sorry

end NUMINAMATH_CALUDE_midpoint_after_translation_l2119_211943


namespace NUMINAMATH_CALUDE_complex_equation_solution_l2119_211992

theorem complex_equation_solution :
  ∃ (z : ℂ), (1 - Complex.I) * z = 2 * Complex.I ∧ z = -1 + Complex.I := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l2119_211992


namespace NUMINAMATH_CALUDE_sum_of_five_consecutive_squares_not_perfect_square_l2119_211900

theorem sum_of_five_consecutive_squares_not_perfect_square (x : ℤ) : 
  ¬∃ (k : ℤ), 5 * (x^2 + 2) = k^2 := by
sorry

end NUMINAMATH_CALUDE_sum_of_five_consecutive_squares_not_perfect_square_l2119_211900


namespace NUMINAMATH_CALUDE_inequality_solution_l2119_211966

theorem inequality_solution (a : ℝ) (h : |a + 1| < 3) :
  (∀ x, x - (a + 1) * (x + 1) > 0 ↔ 
    ((-4 < a ∧ a < -2 ∧ (x > -1 ∨ x < 1 + a)) ∨
     (a = -2 ∧ x ≠ -1) ∨
     (-2 < a ∧ a < 2 ∧ (x > 1 + a ∨ x < -1)))) :=
by sorry

end NUMINAMATH_CALUDE_inequality_solution_l2119_211966


namespace NUMINAMATH_CALUDE_valid_numbers_count_l2119_211953

/-- The number of ways to distribute n identical objects into k distinct boxes --/
def stars_and_bars (n k : ℕ) : ℕ := Nat.choose (n + k - 1) (k - 1)

/-- The number of 6-digit positive integers with digits from 1 to 6 in increasing order --/
def total_increasing_numbers : ℕ := stars_and_bars 6 6

/-- The number of 5-digit positive integers with digits from 1 to 5 in increasing order --/
def numbers_starting_with_6 : ℕ := stars_and_bars 5 5

/-- The number of 6-digit positive integers with digits from 1 to 6 in increasing order, not starting with 6 --/
def valid_numbers : ℕ := total_increasing_numbers - numbers_starting_with_6

theorem valid_numbers_count : valid_numbers = 336 := by
  sorry

end NUMINAMATH_CALUDE_valid_numbers_count_l2119_211953


namespace NUMINAMATH_CALUDE_complex_equation_solution_l2119_211944

theorem complex_equation_solution (Z : ℂ) (i : ℂ) : 
  i * i = -1 → Z = (2 - Z) * i → Z = 1 + i := by sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l2119_211944


namespace NUMINAMATH_CALUDE_unique_solution_l2119_211947

/-- A function satisfying the given functional equation -/
def FunctionalEquation (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, (f x * f y - f (x * y)) / 3 = x + y + 2

/-- The main theorem stating that the function f(x) = x + 3 is the unique solution -/
theorem unique_solution :
  ∀ f : ℝ → ℝ, FunctionalEquation f → ∀ x : ℝ, f x = x + 3 := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_l2119_211947


namespace NUMINAMATH_CALUDE_inequality_problem_l2119_211929

theorem inequality_problem (m n : ℝ) (h : ∀ x : ℝ, m * x^2 + n * x - 1/m < 0 ↔ x < -1/2 ∨ x > 2) :
  (m = -1 ∧ n = 3/2) ∧
  (∀ a : ℝ, 
    (a < 1 → ∀ x : ℝ, (2*a-1-x)*(x+m) > 0 ↔ 2*a-1 < x ∧ x < 1) ∧
    (a = 1 → ∀ x : ℝ, ¬((2*a-1-x)*(x+m) > 0)) ∧
    (a > 1 → ∀ x : ℝ, (2*a-1-x)*(x+m) > 0 ↔ 1 < x ∧ x < 2*a-1)) :=
by sorry

end NUMINAMATH_CALUDE_inequality_problem_l2119_211929


namespace NUMINAMATH_CALUDE_same_solution_implies_c_value_l2119_211936

theorem same_solution_implies_c_value (c : ℝ) :
  (∃ x : ℝ, 3 * x + 8 = 5 ∧ c * x + 15 = 3) → c = 12 := by
  sorry

end NUMINAMATH_CALUDE_same_solution_implies_c_value_l2119_211936


namespace NUMINAMATH_CALUDE_simplify_expression_l2119_211922

theorem simplify_expression (a b k : ℝ) (h1 : a + b = -k) (h2 : a * b = -3) :
  (a - 3) * (b - 3) = 6 + 3 * k := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l2119_211922


namespace NUMINAMATH_CALUDE_concat_reverse_divisible_by_99_l2119_211984

def is_valid_permutation (p : List Nat) : Prop :=
  p.length = 10 ∧ 
  p.head? ≠ some 0 ∧ 
  (∀ i, i ∈ p → i < 10) ∧
  (∀ i, i < 10 → i ∈ p)

def concat_with_reverse (p : List Nat) : List Nat :=
  p ++ p.reverse

def to_number (digits : List Nat) : Nat :=
  digits.foldl (fun acc d => acc * 10 + d) 0

theorem concat_reverse_divisible_by_99 (p : List Nat) 
  (h : is_valid_permutation p) : 
  99 ∣ to_number (concat_with_reverse p) := by
  sorry

end NUMINAMATH_CALUDE_concat_reverse_divisible_by_99_l2119_211984


namespace NUMINAMATH_CALUDE_complex_modulus_problem_l2119_211938

theorem complex_modulus_problem (z : ℂ) :
  (1 + Complex.I * Real.sqrt 3)^2 * z = 1 - Complex.I^3 →
  Complex.abs z = Real.sqrt 2 / 4 := by
sorry

end NUMINAMATH_CALUDE_complex_modulus_problem_l2119_211938


namespace NUMINAMATH_CALUDE_remainder_of_binary_division_l2119_211951

-- Define the binary number
def binary_number : ℕ := 101100110011

-- Define the divisor
def divisor : ℕ := 8

-- Theorem statement
theorem remainder_of_binary_division :
  binary_number % divisor = 3 := by
  sorry

end NUMINAMATH_CALUDE_remainder_of_binary_division_l2119_211951


namespace NUMINAMATH_CALUDE_machine_A_rate_l2119_211916

/-- Production rates of machines A, P, and Q -/
structure MachineRates where
  rateA : ℝ
  rateP : ℝ
  rateQ : ℝ

/-- Time taken by machines P and Q to produce 220 sprockets -/
structure MachineTimes where
  timeP : ℝ
  timeQ : ℝ

/-- Conditions of the sprocket manufacturing problem -/
def sprocketProblem (r : MachineRates) (t : MachineTimes) : Prop :=
  220 / t.timeP = r.rateP
  ∧ 220 / t.timeQ = r.rateQ
  ∧ t.timeP = t.timeQ + 10
  ∧ r.rateQ = 1.1 * r.rateA
  ∧ r.rateA > 0
  ∧ r.rateP > 0
  ∧ r.rateQ > 0
  ∧ t.timeP > 0
  ∧ t.timeQ > 0

/-- Theorem stating that machine A's production rate is 20/9 sprockets per hour -/
theorem machine_A_rate (r : MachineRates) (t : MachineTimes) 
  (h : sprocketProblem r t) : r.rateA = 20/9 := by
  sorry

end NUMINAMATH_CALUDE_machine_A_rate_l2119_211916


namespace NUMINAMATH_CALUDE_arithmetic_progression_special_case_l2119_211928

/-- 
Given an arithmetic progression (a_n) where a_k = l and a_l = k (k ≠ l),
prove that the general term a_n is equal to k + l - n.
-/
theorem arithmetic_progression_special_case 
  (a : ℕ → ℤ) (k l : ℕ) (h_neq : k ≠ l) 
  (h_arith : ∀ n m : ℕ, a (n + 1) - a n = a (m + 1) - a m) 
  (h_k : a k = l) (h_l : a l = k) :
  ∀ n : ℕ, a n = k + l - n :=
sorry

end NUMINAMATH_CALUDE_arithmetic_progression_special_case_l2119_211928


namespace NUMINAMATH_CALUDE_rectangle_area_in_isosceles_triangle_l2119_211957

/-- The area of a rectangle inscribed in an isosceles triangle -/
theorem rectangle_area_in_isosceles_triangle 
  (b h x : ℝ) 
  (hb : b > 0) 
  (hh : h > 0) 
  (hx : x > 0) 
  (hx_bound : x < h/2) : 
  let rectangle_area := x * (b/2 - b*x/h)
  ∃ (rectangle_base : ℝ), 
    rectangle_base > 0 ∧ 
    rectangle_base = b * (h/2 - x) / h ∧
    rectangle_area = x * rectangle_base :=
by sorry

end NUMINAMATH_CALUDE_rectangle_area_in_isosceles_triangle_l2119_211957


namespace NUMINAMATH_CALUDE_price_increase_ratio_l2119_211999

theorem price_increase_ratio (original_price : ℝ) 
  (h1 : original_price > 0)
  (h2 : original_price * 1.3 = 364) : 
  364 / original_price = 1.3 := by
sorry

end NUMINAMATH_CALUDE_price_increase_ratio_l2119_211999


namespace NUMINAMATH_CALUDE_only_set_C_forms_triangle_l2119_211924

/-- A function that checks if three numbers can form a triangle --/
def can_form_triangle (a b c : ℝ) : Prop :=
  a + b > c ∧ b + c > a ∧ c + a > b

/-- The sets of lengths given in the problem --/
def set_A : List ℝ := [3, 4, 8]
def set_B : List ℝ := [8, 7, 15]
def set_C : List ℝ := [13, 12, 20]
def set_D : List ℝ := [5, 5, 11]

/-- Theorem stating that only set C can form a triangle --/
theorem only_set_C_forms_triangle :
  ¬(can_form_triangle set_A[0] set_A[1] set_A[2]) ∧
  ¬(can_form_triangle set_B[0] set_B[1] set_B[2]) ∧
  can_form_triangle set_C[0] set_C[1] set_C[2] ∧
  ¬(can_form_triangle set_D[0] set_D[1] set_D[2]) :=
by sorry

end NUMINAMATH_CALUDE_only_set_C_forms_triangle_l2119_211924


namespace NUMINAMATH_CALUDE_fish_estimation_result_l2119_211919

/-- Represents the catch-release-recatch method for estimating fish population -/
structure FishEstimation where
  initial_catch : ℕ
  initial_marked : ℕ
  second_catch : ℕ
  second_marked : ℕ

/-- Calculates the estimated number of fish in the pond -/
def estimate_fish_population (fe : FishEstimation) : ℕ :=
  (fe.initial_marked * fe.second_catch) / fe.second_marked

/-- Theorem stating that the estimated number of fish in the pond is 2500 -/
theorem fish_estimation_result :
  let fe : FishEstimation := {
    initial_catch := 100,
    initial_marked := 100,
    second_catch := 200,
    second_marked := 8
  }
  estimate_fish_population fe = 2500 := by
  sorry


end NUMINAMATH_CALUDE_fish_estimation_result_l2119_211919


namespace NUMINAMATH_CALUDE_shaded_area_is_73_l2119_211925

/-- The total area of two overlapping rectangles minus their common area -/
def total_shaded_area (length1 width1 length2 width2 overlap_area : ℕ) : ℕ :=
  length1 * width1 + length2 * width2 - overlap_area

/-- Theorem stating that the total shaded area is 73 for the given dimensions -/
theorem shaded_area_is_73 :
  total_shaded_area 8 5 4 9 3 = 73 := by
  sorry

end NUMINAMATH_CALUDE_shaded_area_is_73_l2119_211925


namespace NUMINAMATH_CALUDE_converse_opposites_sum_zero_l2119_211930

theorem converse_opposites_sum_zero :
  ∀ x y : ℝ, (x = -y) → (x + y = 0) := by
  sorry

end NUMINAMATH_CALUDE_converse_opposites_sum_zero_l2119_211930


namespace NUMINAMATH_CALUDE_magnificent_monday_l2119_211997

-- Define a structure for a month
structure Month where
  days : Nat
  firstMonday : Nat

-- Define a function to calculate the date of the nth Monday
def nthMonday (m : Month) (n : Nat) : Nat :=
  m.firstMonday + 7 * (n - 1)

-- Theorem statement
theorem magnificent_monday (m : Month) 
  (h1 : m.days = 31)
  (h2 : m.firstMonday = 2) :
  nthMonday m 5 = 30 := by
  sorry

end NUMINAMATH_CALUDE_magnificent_monday_l2119_211997


namespace NUMINAMATH_CALUDE_more_heads_than_tails_probability_l2119_211941

/-- The probability of getting more heads than tails when tossing a fair coin 4 times -/
def probability_more_heads_than_tails : ℚ := 5/16

/-- A fair coin is tossed 4 times -/
def num_tosses : ℕ := 4

/-- The probability of getting heads on a single toss of a fair coin -/
def probability_heads : ℚ := 1/2

/-- The probability of getting tails on a single toss of a fair coin -/
def probability_tails : ℚ := 1/2

theorem more_heads_than_tails_probability :
  probability_more_heads_than_tails = 
    (Nat.choose num_tosses 3 : ℚ) * probability_heads^3 * probability_tails +
    (Nat.choose num_tosses 4 : ℚ) * probability_heads^4 :=
by sorry

end NUMINAMATH_CALUDE_more_heads_than_tails_probability_l2119_211941


namespace NUMINAMATH_CALUDE_initial_figures_correct_figure_50_l2119_211923

/-- The number of unit squares in the nth figure -/
def f (n : ℕ) : ℕ := 3 * n^2 + 3 * n + 1

/-- The first four figures match the given pattern -/
theorem initial_figures_correct :
  f 0 = 1 ∧ f 1 = 7 ∧ f 2 = 19 ∧ f 3 = 37 := by sorry

/-- The 50th figure contains 7651 unit squares -/
theorem figure_50 : f 50 = 7651 := by sorry

end NUMINAMATH_CALUDE_initial_figures_correct_figure_50_l2119_211923


namespace NUMINAMATH_CALUDE_gcd_of_35_91_840_l2119_211939

theorem gcd_of_35_91_840 : Nat.gcd 35 (Nat.gcd 91 840) = 7 := by
  sorry

end NUMINAMATH_CALUDE_gcd_of_35_91_840_l2119_211939


namespace NUMINAMATH_CALUDE_square_sum_value_l2119_211970

theorem square_sum_value (x y : ℝ) (h1 : (x + y)^2 = 49) (h2 : x * y = 10) : x^2 + y^2 = 29 := by
  sorry

end NUMINAMATH_CALUDE_square_sum_value_l2119_211970


namespace NUMINAMATH_CALUDE_optimal_speed_yihuang_expressway_l2119_211972

/-- The optimal speed problem for the Yihuang Expressway -/
theorem optimal_speed_yihuang_expressway 
  (total_length : ℝ) 
  (min_speed max_speed : ℝ) 
  (fixed_cost : ℝ) 
  (k : ℝ) 
  (max_total_cost : ℝ) :
  total_length = 350 →
  min_speed = 60 →
  max_speed = 120 →
  fixed_cost = 200 →
  k * max_speed^2 + fixed_cost = max_total_cost →
  max_total_cost = 488 →
  ∃ (optimal_speed : ℝ), 
    optimal_speed = 100 ∧
    ∀ (v : ℝ), min_speed ≤ v ∧ v ≤ max_speed →
      total_length * (fixed_cost / v + k * v) ≥ 
      total_length * (fixed_cost / optimal_speed + k * optimal_speed) :=
by sorry

end NUMINAMATH_CALUDE_optimal_speed_yihuang_expressway_l2119_211972


namespace NUMINAMATH_CALUDE_matthew_initial_crackers_l2119_211976

/-- The number of crackers Matthew gave to each friend -/
def crackers_per_friend : ℕ := 2

/-- The number of friends Matthew gave crackers to -/
def number_of_friends : ℕ := 4

/-- The total number of crackers Matthew gave away -/
def total_crackers_given : ℕ := crackers_per_friend * number_of_friends

/-- Theorem stating that Matthew had at least 8 crackers initially -/
theorem matthew_initial_crackers :
  ∃ (initial_crackers : ℕ), initial_crackers ≥ total_crackers_given ∧ initial_crackers ≥ 8 := by
  sorry

end NUMINAMATH_CALUDE_matthew_initial_crackers_l2119_211976


namespace NUMINAMATH_CALUDE_sin_240_degrees_l2119_211912

theorem sin_240_degrees : Real.sin (240 * π / 180) = -Real.sqrt 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_sin_240_degrees_l2119_211912


namespace NUMINAMATH_CALUDE_unique_covering_100x100_l2119_211982

/-- A frame is the border of a square in a grid. -/
structure Frame where
  side_length : ℕ

/-- A covering is a list of non-overlapping frames that completely cover a square grid. -/
structure Covering where
  frames : List Frame
  non_overlapping : ∀ (f1 f2 : Frame), f1 ∈ frames → f2 ∈ frames → f1 ≠ f2 → 
    f1.side_length ≠ f2.side_length
  complete : ∀ (n : ℕ), n ∈ List.range 50 → 
    ∃ (f : Frame), f ∈ frames ∧ f.side_length = 100 - 2 * n

/-- The theorem states that there is a unique covering of a 100×100 grid with 50 frames. -/
theorem unique_covering_100x100 : 
  ∃! (c : Covering), c.frames.length = 50 ∧ 
    (∀ (f : Frame), f ∈ c.frames → f.side_length ≤ 100 ∧ f.side_length % 2 = 0) :=
sorry

end NUMINAMATH_CALUDE_unique_covering_100x100_l2119_211982


namespace NUMINAMATH_CALUDE_stick_difference_l2119_211909

/-- 
Given:
- Dave picked up 14 sticks
- Amy picked up 9 sticks
- There were initially 50 sticks in the yard

Prove that the difference between the number of sticks picked up by Dave and Amy
and the number of sticks left in the yard is 4.
-/
theorem stick_difference (dave_sticks amy_sticks initial_sticks : ℕ) 
  (h1 : dave_sticks = 14)
  (h2 : amy_sticks = 9)
  (h3 : initial_sticks = 50) :
  let picked_up := dave_sticks + amy_sticks
  let left_in_yard := initial_sticks - picked_up
  picked_up - left_in_yard = 4 := by
  sorry

end NUMINAMATH_CALUDE_stick_difference_l2119_211909


namespace NUMINAMATH_CALUDE_cory_fruit_eating_orders_l2119_211904

/-- Represents the number of fruits Cory has of each type -/
structure FruitInventory where
  apples : Nat
  bananas : Nat
  mangoes : Nat

/-- Represents the constraints of Cory's fruit-eating schedule -/
structure EatingSchedule where
  days : Nat
  startsWithApple : Bool
  endsWithApple : Bool

/-- Calculates the number of ways Cory can eat his fruits given his inventory and schedule constraints -/
def countEatingOrders (inventory : FruitInventory) (schedule : EatingSchedule) : Nat :=
  sorry

/-- Theorem stating that given Cory's specific fruit inventory and eating schedule, 
    there are exactly 80 different orders in which he can eat his fruits -/
theorem cory_fruit_eating_orders :
  let inventory : FruitInventory := ⟨3, 3, 1⟩
  let schedule : EatingSchedule := ⟨7, true, true⟩
  countEatingOrders inventory schedule = 80 :=
by sorry

end NUMINAMATH_CALUDE_cory_fruit_eating_orders_l2119_211904


namespace NUMINAMATH_CALUDE_victor_trays_capacity_l2119_211971

/-- The number of trays Victor picked up from the first table -/
def trays_table1 : ℕ := 23

/-- The number of trays Victor picked up from the second table -/
def trays_table2 : ℕ := 5

/-- The total number of trips Victor made -/
def total_trips : ℕ := 4

/-- The number of trays Victor could carry at a time -/
def trays_per_trip : ℕ := (trays_table1 + trays_table2) / total_trips

theorem victor_trays_capacity : trays_per_trip = 7 := by
  sorry

end NUMINAMATH_CALUDE_victor_trays_capacity_l2119_211971


namespace NUMINAMATH_CALUDE_jessica_cut_four_orchids_l2119_211945

/-- The number of orchids Jessica cut from her garden -/
def orchids_cut (initial_orchids final_orchids : ℕ) : ℕ :=
  final_orchids - initial_orchids

/-- Theorem stating that Jessica cut 4 orchids -/
theorem jessica_cut_four_orchids :
  orchids_cut 3 7 = 4 := by
  sorry

end NUMINAMATH_CALUDE_jessica_cut_four_orchids_l2119_211945


namespace NUMINAMATH_CALUDE_complement_A_intersect_B_l2119_211964

-- Define set A
def A : Set ℝ := {x | |x| < 1}

-- Define set B
def B : Set ℝ := {y | ∃ x, y = 2^x + 1}

-- State the theorem
theorem complement_A_intersect_B : (Set.compl A) ∩ B = Set.Ioi 1 := by
  sorry

end NUMINAMATH_CALUDE_complement_A_intersect_B_l2119_211964


namespace NUMINAMATH_CALUDE_percentage_problem_l2119_211918

theorem percentage_problem (n : ℝ) (p : ℝ) : 
  n = 50 → 
  p / 100 * n = 30 / 100 * 10 + 27 → 
  p = 60 := by
sorry

end NUMINAMATH_CALUDE_percentage_problem_l2119_211918


namespace NUMINAMATH_CALUDE_triangle_mapping_l2119_211907

theorem triangle_mapping :
  ∃ (f : ℂ → ℂ), 
    (∀ z w, f z = w ↔ w = (1 + Complex.I) * (1 - z)) ∧
    f 0 = 1 + Complex.I ∧
    f 1 = 0 ∧
    f Complex.I = 2 :=
by sorry

end NUMINAMATH_CALUDE_triangle_mapping_l2119_211907


namespace NUMINAMATH_CALUDE_quadratic_expression_value_l2119_211937

theorem quadratic_expression_value (a b c : ℝ) 
  (h1 : a - b = 2 + Real.sqrt 3)
  (h2 : b - c = 2 - Real.sqrt 3) :
  a^2 + b^2 + c^2 - a*b - b*c - c*a = 15 := by
sorry

end NUMINAMATH_CALUDE_quadratic_expression_value_l2119_211937


namespace NUMINAMATH_CALUDE_simplified_expression_l2119_211974

theorem simplified_expression : 
  (2 * Real.sqrt 8) / (Real.sqrt 3 + Real.sqrt 4 + Real.sqrt 7) = 
  Real.sqrt 6 + 2 * Real.sqrt 2 - Real.sqrt 14 := by
  sorry

end NUMINAMATH_CALUDE_simplified_expression_l2119_211974


namespace NUMINAMATH_CALUDE_parallel_vectors_x_value_l2119_211908

def vector_a (x : ℝ) : Fin 2 → ℝ := ![2, x]
def vector_b (x : ℝ) : Fin 2 → ℝ := ![3, x + 1]

theorem parallel_vectors_x_value :
  ∀ x : ℝ, (∃ k : ℝ, k ≠ 0 ∧ vector_a x = k • vector_b x) → x = 2 := by
sorry

end NUMINAMATH_CALUDE_parallel_vectors_x_value_l2119_211908


namespace NUMINAMATH_CALUDE_consecutive_pair_divisible_by_five_l2119_211931

theorem consecutive_pair_divisible_by_five (a b : ℕ) : 
  a < 1500 → 
  b < 1500 → 
  b = a + 1 → 
  (a + b) % 5 = 0 → 
  a = 57 → 
  b = 58 := by
sorry

end NUMINAMATH_CALUDE_consecutive_pair_divisible_by_five_l2119_211931


namespace NUMINAMATH_CALUDE_expression_simplification_l2119_211911

/-- Proves that the simplification of 7y + 8 - 3y + 15 + 2x is equivalent to 4y + 2x + 23 -/
theorem expression_simplification (x y : ℝ) :
  7 * y + 8 - 3 * y + 15 + 2 * x = 4 * y + 2 * x + 23 := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l2119_211911


namespace NUMINAMATH_CALUDE_f_monotone_decreasing_min_a_value_l2119_211920

noncomputable section

def f (x : ℝ) := 2 * (Real.cos x)^2 + 2 * Real.sqrt 3 * Real.sin x * Real.cos x

def g (x : ℝ) := x * Real.exp (-x)

def is_monotone_decreasing (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x y, a ≤ x ∧ x < y ∧ y ≤ b → f y < f x

def monotone_decreasing_intervals (f : ℝ → ℝ) : Set (Set ℝ) :=
  {I | ∃ k : ℤ, I = Set.Icc (k * Real.pi + Real.pi / 6) (k * Real.pi + 2 * Real.pi / 3) ∧
    is_monotone_decreasing f (k * Real.pi + Real.pi / 6) (k * Real.pi + 2 * Real.pi / 3)}

theorem f_monotone_decreasing : 
  monotone_decreasing_intervals f = {I | ∃ k : ℤ, I = Set.Icc (k * Real.pi + Real.pi / 6) (k * Real.pi + 2 * Real.pi / 3)} :=
sorry

theorem min_a_value :
  (∃ a : ℝ, ∀ x₁ x₂ : ℝ, x₁ ∈ Set.Icc 1 3 → x₂ ∈ Set.Icc 0 (Real.pi / 2) → 
    g x₁ + a + 3 > f x₂) ∧
  (∀ a' : ℝ, a' < -3 / Real.exp 3 → 
    ∃ x₁ x₂ : ℝ, x₁ ∈ Set.Icc 1 3 ∧ x₂ ∈ Set.Icc 0 (Real.pi / 2) ∧ 
      g x₁ + a' + 3 ≤ f x₂) :=
sorry

end NUMINAMATH_CALUDE_f_monotone_decreasing_min_a_value_l2119_211920


namespace NUMINAMATH_CALUDE_roulette_wheel_probability_l2119_211914

/-- The probability of a roulette wheel landing on section F -/
def prob_F (prob_D prob_E prob_G : ℚ) : ℚ :=
  1 - (prob_D + prob_E + prob_G)

/-- Theorem: The probability of landing on section F is 1/4 -/
theorem roulette_wheel_probability :
  let prob_D : ℚ := 3/8
  let prob_E : ℚ := 1/4
  let prob_G : ℚ := 1/8
  prob_F prob_D prob_E prob_G = 1/4 := by
  sorry

end NUMINAMATH_CALUDE_roulette_wheel_probability_l2119_211914


namespace NUMINAMATH_CALUDE_ufo_convention_attendees_l2119_211954

theorem ufo_convention_attendees 
  (total_attendees : ℕ) 
  (total_presenters : ℕ) 
  (male_presenters female_presenters : ℕ) 
  (male_general female_general : ℕ) :
  total_attendees = 1000 →
  total_presenters = 420 →
  male_presenters = female_presenters + 20 →
  female_general = male_general + 56 →
  total_attendees = total_presenters + male_general + female_general →
  male_general = 262 := by
sorry

end NUMINAMATH_CALUDE_ufo_convention_attendees_l2119_211954


namespace NUMINAMATH_CALUDE_cheryl_material_usage_l2119_211969

theorem cheryl_material_usage 
  (material1 : ℚ) 
  (material2 : ℚ) 
  (leftover : ℚ) 
  (h1 : material1 = 5 / 11)
  (h2 : material2 = 2 / 3)
  (h3 : leftover = 25 / 55) :
  material1 + material2 - leftover = 22 / 33 := by
  sorry

end NUMINAMATH_CALUDE_cheryl_material_usage_l2119_211969


namespace NUMINAMATH_CALUDE_distance_range_l2119_211921

/-- Hyperbola C with equation x^2 - y^2/3 = 1 -/
def hyperbola_C (x y : ℝ) : Prop := x^2 - y^2/3 = 1

/-- Focal length of the hyperbola -/
def focal_length : ℝ := 4

/-- Right triangle ABD formed by intersection with perpendicular line through right focus -/
def right_triangle_ABD : Prop := sorry

/-- Slopes of lines AM and AN -/
def slope_product (k₁ k₂ : ℝ) : Prop := k₁ * k₂ = -2

/-- Distance from A to line MN -/
def distance_A_to_MN (d : ℝ) : Prop := sorry

/-- Theorem stating the range of distance d -/
theorem distance_range :
  ∀ (d : ℝ), hyperbola_C 1 0 →
  focal_length = 4 →
  right_triangle_ABD →
  (∃ k₁ k₂, slope_product k₁ k₂) →
  distance_A_to_MN d →
  3 * Real.sqrt 3 < d ∧ d ≤ 6 := by sorry

end NUMINAMATH_CALUDE_distance_range_l2119_211921


namespace NUMINAMATH_CALUDE_age_difference_l2119_211903

theorem age_difference (a b c : ℕ) (h : a + b = b + c + 10) : a = c + 10 := by
  sorry

end NUMINAMATH_CALUDE_age_difference_l2119_211903


namespace NUMINAMATH_CALUDE_point_division_vector_representation_l2119_211993

/-- Given a line segment CD and a point Q on CD such that CQ:QD = 3:5,
    prove that Q⃗ = (5/8)C⃗ + (3/8)D⃗ -/
theorem point_division_vector_representation 
  (C D Q : EuclideanSpace ℝ (Fin n)) 
  (h_on_segment : ∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ Q = (1 - t) • C + t • D) 
  (h_ratio : dist C Q / dist Q D = 3 / 5) :
  Q = (5/8) • C + (3/8) • D :=
by sorry

end NUMINAMATH_CALUDE_point_division_vector_representation_l2119_211993


namespace NUMINAMATH_CALUDE_cube_root_problem_l2119_211968

theorem cube_root_problem (a : ℕ) (h : a^3 = 21 * 49 * 45 * 25) : a = 105 := by
  sorry

end NUMINAMATH_CALUDE_cube_root_problem_l2119_211968


namespace NUMINAMATH_CALUDE_range_of_m_l2119_211963

/-- α is the condition that x ≤ -5 or x ≥ 1 -/
def α (x : ℝ) : Prop := x ≤ -5 ∨ x ≥ 1

/-- β is the condition that 2m-3 ≤ x ≤ 2m+1 -/
def β (m x : ℝ) : Prop := 2*m - 3 ≤ x ∧ x ≤ 2*m + 1

/-- α is a necessary condition for β -/
def α_necessary_for_β (m : ℝ) : Prop := ∀ x, β m x → α x

theorem range_of_m (m : ℝ) : α_necessary_for_β m → m ≥ 2 ∨ m ≤ -3 := by
  sorry

end NUMINAMATH_CALUDE_range_of_m_l2119_211963


namespace NUMINAMATH_CALUDE_simplify_sqrt_sum_l2119_211994

theorem simplify_sqrt_sum : 
  Real.sqrt (12 + 8 * Real.sqrt 3) + Real.sqrt (12 - 8 * Real.sqrt 3) = 2 * Real.sqrt 6 := by
  sorry

end NUMINAMATH_CALUDE_simplify_sqrt_sum_l2119_211994


namespace NUMINAMATH_CALUDE_gcd_factorial_8_and_cube_factorial_6_l2119_211978

theorem gcd_factorial_8_and_cube_factorial_6 :
  Nat.gcd (Nat.factorial 8) (Nat.factorial 6 ^ 3) = 11520 := by
  sorry

end NUMINAMATH_CALUDE_gcd_factorial_8_and_cube_factorial_6_l2119_211978


namespace NUMINAMATH_CALUDE_hyperbola_tangent_dot_product_l2119_211988

/-- The hyperbola equation -/
def hyperbola (x y : ℝ) : Prop := x^2 / 4 - y^2 = 1

/-- The asymptotes of the hyperbola -/
def asymptote (x y : ℝ) : Prop := y = x / 2 ∨ y = -x / 2

/-- A point is on the line l -/
def on_line_l (x y : ℝ) : Prop := sorry

/-- The line l is tangent to the hyperbola at point P -/
def is_tangent (P : ℝ × ℝ) : Prop := 
  hyperbola P.1 P.2 ∧ on_line_l P.1 P.2 ∧ 
  ∀ Q : ℝ × ℝ, Q ≠ P → on_line_l Q.1 Q.2 → ¬hyperbola Q.1 Q.2

theorem hyperbola_tangent_dot_product 
  (P M N : ℝ × ℝ) 
  (h_tangent : is_tangent P) 
  (h_M : on_line_l M.1 M.2 ∧ asymptote M.1 M.2) 
  (h_N : on_line_l N.1 N.2 ∧ asymptote N.1 N.2) :
  M.1 * N.1 + M.2 * N.2 = 3 := 
sorry

end NUMINAMATH_CALUDE_hyperbola_tangent_dot_product_l2119_211988


namespace NUMINAMATH_CALUDE_unique_solution_l2119_211932

/-- Define a sequence of 100 real numbers satisfying given conditions -/
def SequenceOfHundred (a : Fin 100 → ℝ) : Prop :=
  (∀ i : Fin 99, a i - 4 * a (i + 1) + 3 * a (i + 2) ≥ 0) ∧
  (a 99 - 4 * a 0 + 3 * a 1 ≥ 0) ∧
  (a 0 = 1)

/-- Theorem stating that the sequence of all 1's is the unique solution -/
theorem unique_solution (a : Fin 100 → ℝ) (h : SequenceOfHundred a) :
  ∀ i : Fin 100, a i = 1 := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_l2119_211932


namespace NUMINAMATH_CALUDE_tangent_range_l2119_211995

/-- The circle C in the Cartesian coordinate system -/
def Circle (x y : ℝ) : Prop := x^2 + y^2 - 4*x = 0

/-- The line equation -/
def Line (k x y : ℝ) : Prop := y = k*(x + 1)

/-- Point P is on the line -/
def PointOnLine (P : ℝ × ℝ) (k : ℝ) : Prop :=
  Line k P.1 P.2

/-- Two tangents from P to the circle are perpendicular -/
def PerpendicularTangents (P : ℝ × ℝ) : Prop :=
  ∃ A B : ℝ × ℝ, Circle A.1 A.2 ∧ Circle B.1 B.2 ∧
    (A.1 - P.1) * (B.1 - P.1) + (A.2 - P.2) * (B.2 - P.2) = 0

/-- The main theorem -/
theorem tangent_range (k : ℝ) :
  (∃ P : ℝ × ℝ, PointOnLine P k ∧ PerpendicularTangents P) →
  k ∈ Set.Icc (-2 * Real.sqrt 2) (2 * Real.sqrt 2) :=
sorry

end NUMINAMATH_CALUDE_tangent_range_l2119_211995


namespace NUMINAMATH_CALUDE_smallest_n_for_integer_sum_eighteen_makes_sum_integer_smallest_n_is_eighteen_l2119_211927

theorem smallest_n_for_integer_sum : 
  ∀ n : ℕ+, (1/3 + 1/4 + 1/9 + 1/n : ℚ).isInt → n ≥ 18 := by
  sorry

theorem eighteen_makes_sum_integer : 
  (1/3 + 1/4 + 1/9 + 1/18 : ℚ).isInt := by
  sorry

theorem smallest_n_is_eighteen : 
  ∃! n : ℕ+, (1/3 + 1/4 + 1/9 + 1/n : ℚ).isInt ∧ ∀ m : ℕ+, (1/3 + 1/4 + 1/9 + 1/m : ℚ).isInt → n ≤ m := by
  sorry

end NUMINAMATH_CALUDE_smallest_n_for_integer_sum_eighteen_makes_sum_integer_smallest_n_is_eighteen_l2119_211927


namespace NUMINAMATH_CALUDE_james_flowers_l2119_211906

/-- The number of flowers planted by James and his friends --/
def flower_planting 
  (james friend_a friend_b friend_c friend_d friend_e friend_f friend_g : ℝ) : Prop :=
  james = friend_a * 1.2
  ∧ friend_a = friend_b * 1.15
  ∧ friend_b = friend_c * 0.7
  ∧ friend_c = friend_d * 1.1
  ∧ friend_d = friend_e * 1.25
  ∧ friend_e = friend_f
  ∧ friend_g = friend_f * 0.7
  ∧ friend_b = 12

/-- The theorem stating James plants 16.56 flowers per day --/
theorem james_flowers 
  (james friend_a friend_b friend_c friend_d friend_e friend_f friend_g : ℝ) 
  (h : flower_planting james friend_a friend_b friend_c friend_d friend_e friend_f friend_g) : 
  james = 16.56 := by
  sorry

end NUMINAMATH_CALUDE_james_flowers_l2119_211906


namespace NUMINAMATH_CALUDE_solution_to_diophantine_equation_l2119_211959

theorem solution_to_diophantine_equation :
  ∀ x y z : ℕ+,
    x ≤ y ∧ y ≤ z →
    5 * (x * y + y * z + z * x) = 4 * x * y * z →
    ((x = 2 ∧ y = 5 ∧ z = 10) ∨ (x = 2 ∧ y = 4 ∧ z = 20)) :=
by sorry

end NUMINAMATH_CALUDE_solution_to_diophantine_equation_l2119_211959


namespace NUMINAMATH_CALUDE_even_function_implies_m_zero_l2119_211965

def f (m : ℝ) (x : ℝ) : ℝ := (m - 2) * x^2 + m * x + 4

theorem even_function_implies_m_zero (m : ℝ) :
  (∀ x : ℝ, f m x = f m (-x)) → m = 0 := by
  sorry

end NUMINAMATH_CALUDE_even_function_implies_m_zero_l2119_211965


namespace NUMINAMATH_CALUDE_line_through_points_l2119_211913

-- Define the points
def P₁ : ℝ × ℝ := (3, -1)
def P₂ : ℝ × ℝ := (-2, 1)

-- Define the slope-intercept form
def slope_intercept (m b : ℝ) (x y : ℝ) : Prop :=
  y = m * x + b

-- Theorem statement
theorem line_through_points : 
  ∃ (m b : ℝ), m = -2/5 ∧ b = 1/5 ∧ 
  (slope_intercept m b P₁.1 P₁.2 ∧ slope_intercept m b P₂.1 P₂.2) :=
sorry

end NUMINAMATH_CALUDE_line_through_points_l2119_211913


namespace NUMINAMATH_CALUDE_system_solution_l2119_211996

theorem system_solution (x y : ℤ) (h1 : 7 - x = 15) (h2 : y - 3 = 4 + x) :
  x = -8 ∧ y = -1 := by
  sorry

end NUMINAMATH_CALUDE_system_solution_l2119_211996


namespace NUMINAMATH_CALUDE_smith_family_puzzle_l2119_211935

def is_valid_license_plate (n : ℕ) : Prop :=
  (n ≥ 10000 ∧ n < 100000) ∧
  ∃ (a b c : ℕ) (d : ℕ),
    a ≠ b ∧ b ≠ c ∧ a ≠ c ∧
    (n.digits 10).count a ≥ 1 ∧
    (n.digits 10).count b ≥ 1 ∧
    (n.digits 10).count c ≥ 1 ∧
    (n.digits 10).count d = 3 ∧
    (n.digits 10).sum = 2 * (n % 100)

theorem smith_family_puzzle :
  ∀ (license_plate : ℕ) (children_ages : List ℕ),
    is_valid_license_plate license_plate →
    children_ages.length = 9 →
    children_ages.maximum = some 10 →
    (∀ age ∈ children_ages, age < 10) →
    (∀ age ∈ children_ages, license_plate % age = 0) →
    4 ∉ children_ages :=
by sorry

end NUMINAMATH_CALUDE_smith_family_puzzle_l2119_211935


namespace NUMINAMATH_CALUDE_hyperbola_eccentricity_l2119_211986

/-- The eccentricity of a hyperbola with given conditions -/
theorem hyperbola_eccentricity (a b : ℝ) (ha : a > 0) (hb : b > 0) 
  (h_asymptote : b / a = Real.sqrt 6 / 6) :
  let c := Real.sqrt (a^2 + b^2)
  c / a = Real.sqrt 42 / 6 := by sorry

end NUMINAMATH_CALUDE_hyperbola_eccentricity_l2119_211986


namespace NUMINAMATH_CALUDE_rectangular_prism_diagonal_l2119_211980

theorem rectangular_prism_diagonal (a b c : ℝ) (ha : a = 12) (hb : b = 15) (hc : c = 8) :
  Real.sqrt (a^2 + b^2 + c^2) = Real.sqrt 433 := by
  sorry

end NUMINAMATH_CALUDE_rectangular_prism_diagonal_l2119_211980


namespace NUMINAMATH_CALUDE_four_digit_divisible_by_9_l2119_211977

def is_divisible_by_9 (n : ℕ) : Prop := ∃ k : ℕ, n = 9 * k

def sum_of_digits (n : ℕ) : ℕ :=
  let digits := n.digits 10
  digits.sum

theorem four_digit_divisible_by_9 (B : ℕ) :
  B < 10 →
  is_divisible_by_9 (4000 + 100 * B + 10 * B + 2) →
  B = 6 := by
  sorry

#check four_digit_divisible_by_9

end NUMINAMATH_CALUDE_four_digit_divisible_by_9_l2119_211977


namespace NUMINAMATH_CALUDE_remaining_bonus_l2119_211985

def bonus : ℚ := 1496
def kitchen_fraction : ℚ := 1 / 22
def holiday_fraction : ℚ := 1 / 4
def christmas_fraction : ℚ := 1 / 8

theorem remaining_bonus : 
  bonus - (bonus * kitchen_fraction + bonus * holiday_fraction + bonus * christmas_fraction) = 867 := by
  sorry

end NUMINAMATH_CALUDE_remaining_bonus_l2119_211985


namespace NUMINAMATH_CALUDE_point_coordinates_wrt_origin_l2119_211955

/-- The coordinates of a point P with respect to the origin are the same as its given coordinates in a Cartesian coordinate system. -/
theorem point_coordinates_wrt_origin (x y : ℝ) : 
  let P : ℝ × ℝ := (x, y)
  P = (x, y) := by sorry

end NUMINAMATH_CALUDE_point_coordinates_wrt_origin_l2119_211955


namespace NUMINAMATH_CALUDE_union_of_sets_l2119_211956

def set_A : Set ℝ := {x | x^2 - x = 0}
def set_B : Set ℝ := {x | x^2 + x = 0}

theorem union_of_sets : set_A ∪ set_B = {-1, 0, 1} := by sorry

end NUMINAMATH_CALUDE_union_of_sets_l2119_211956


namespace NUMINAMATH_CALUDE_twins_age_problem_l2119_211998

theorem twins_age_problem (age : ℕ) : 
  (age + 1) * (age + 1) = age * age + 15 → age = 7 := by
  sorry

end NUMINAMATH_CALUDE_twins_age_problem_l2119_211998


namespace NUMINAMATH_CALUDE_area_triangle_ABC_in_special_cyclic_quadrilateral_l2119_211946

/-- Represents a point in 2D space -/
structure Point :=
  (x : ℝ)
  (y : ℝ)

/-- Represents a circle -/
structure Circle :=
  (center : Point)
  (radius : ℝ)

/-- Represents a quadrilateral -/
structure Quadrilateral :=
  (A B C D : Point)

/-- Calculates the area of a triangle given three points -/
def triangleArea (A B C : Point) : ℝ := sorry

/-- Checks if a quadrilateral is cyclic (inscribed in a circle) -/
def isCyclic (q : Quadrilateral) (c : Circle) : Prop := sorry

/-- Finds the intersection point of two line segments -/
def intersectionPoint (A B C D : Point) : Point := sorry

/-- Theorem: Area of triangle ABC in a special cyclic quadrilateral -/
theorem area_triangle_ABC_in_special_cyclic_quadrilateral 
  (A B C D E : Point) (c : Circle) :
  isCyclic ⟨A, B, C, D⟩ c →
  E = intersectionPoint A C B D →
  A.x = D.x ∧ A.y = D.y →
  (C.x - E.x) / (E.x - D.x) = 3 / 2 ∧ (C.y - E.y) / (E.y - D.y) = 3 / 2 →
  triangleArea A B E = 8 →
  triangleArea A B C = 18 := by sorry

end NUMINAMATH_CALUDE_area_triangle_ABC_in_special_cyclic_quadrilateral_l2119_211946


namespace NUMINAMATH_CALUDE_number_of_green_balls_l2119_211948

/-- Given a total of 40 balls with red, blue, and green colors, where there are 11 blue balls
and the number of red balls is twice the number of blue balls, prove that there are 7 green balls. -/
theorem number_of_green_balls (total : ℕ) (blue : ℕ) (red : ℕ) (green : ℕ) : 
  total = 40 →
  blue = 11 →
  red = 2 * blue →
  total = red + blue + green →
  green = 7 := by
sorry

end NUMINAMATH_CALUDE_number_of_green_balls_l2119_211948


namespace NUMINAMATH_CALUDE_largest_angle_in_triangle_l2119_211940

theorem largest_angle_in_triangle (X Y Z : Real) (h_scalene : X ≠ Y ∧ Y ≠ Z ∧ X ≠ Z) 
  (h_angleY : Y = 25) (h_angleZ : Z = 100) : 
  max X (max Y Z) = 100 :=
sorry

end NUMINAMATH_CALUDE_largest_angle_in_triangle_l2119_211940


namespace NUMINAMATH_CALUDE_percent_gain_is_588_l2119_211990

-- Define the number of sheep bought and sold
def total_sheep : ℕ := 900
def sold_first : ℕ := 850
def sold_second : ℕ := 50

-- Define the cost and revenue functions
def cost (price_per_sheep : ℚ) : ℚ := price_per_sheep * total_sheep
def revenue_first (price_per_sheep : ℚ) : ℚ := cost price_per_sheep
def revenue_second (price_per_sheep : ℚ) : ℚ := 
  (revenue_first price_per_sheep / sold_first) * sold_second

-- Define the total revenue and profit
def total_revenue (price_per_sheep : ℚ) : ℚ := 
  revenue_first price_per_sheep + revenue_second price_per_sheep
def profit (price_per_sheep : ℚ) : ℚ := 
  total_revenue price_per_sheep - cost price_per_sheep

-- Define the percent gain
def percent_gain (price_per_sheep : ℚ) : ℚ := 
  (profit price_per_sheep / cost price_per_sheep) * 100

-- Theorem statement
theorem percent_gain_is_588 (price_per_sheep : ℚ) :
  percent_gain price_per_sheep = 52.94 / 9 :=
by sorry

end NUMINAMATH_CALUDE_percent_gain_is_588_l2119_211990


namespace NUMINAMATH_CALUDE_polynomial_value_at_zero_l2119_211981

def is_valid_polynomial (p : ℝ → ℝ) : Prop :=
  ∃ (a₀ a₁ a₂ a₃ a₄ a₅ a₆ : ℝ), 
    ∀ x, p x = a₀ + a₁ * x + a₂ * x^2 + a₃ * x^3 + a₄ * x^4 + a₅ * x^5 + a₆ * x^6

theorem polynomial_value_at_zero 
  (p : ℝ → ℝ) 
  (h_valid : is_valid_polynomial p) 
  (h_values : ∀ n : ℕ, n ≤ 6 → p (3^n) = (1 : ℝ) / (3^n)) :
  p 0 = 29523 / 2187 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_value_at_zero_l2119_211981


namespace NUMINAMATH_CALUDE_complex_magnitude_l2119_211979

theorem complex_magnitude (w z : ℂ) 
  (h1 : w * z + 2 * w - 3 * z = 10 - 6 * Complex.I)
  (h2 : Complex.abs w = 2)
  (h3 : Complex.abs (w + 2) = 3) :
  Complex.abs z = (2 * Real.sqrt 34 - 4) / 5 := by
sorry

end NUMINAMATH_CALUDE_complex_magnitude_l2119_211979


namespace NUMINAMATH_CALUDE_shooting_competition_probabilities_l2119_211905

/-- Probability of A hitting the target in a single shot -/
def prob_A_hit : ℚ := 2/3

/-- Probability of B hitting the target in a single shot -/
def prob_B_hit : ℚ := 3/4

/-- Number of consecutive shots -/
def num_shots : ℕ := 3

theorem shooting_competition_probabilities :
  let prob_A_miss_at_least_once := 1 - prob_A_hit ^ num_shots
  let prob_A_hit_twice := (num_shots.choose 2 : ℚ) * prob_A_hit^2 * (1 - prob_A_hit)
  let prob_B_hit_once := (num_shots.choose 1 : ℚ) * prob_B_hit * (1 - prob_B_hit)^2
  prob_A_miss_at_least_once = 19/27 ∧
  prob_A_hit_twice * prob_B_hit_once = 1/16 := by
  sorry


end NUMINAMATH_CALUDE_shooting_competition_probabilities_l2119_211905


namespace NUMINAMATH_CALUDE_M_divisible_by_41_l2119_211958

def M : ℕ := sorry

theorem M_divisible_by_41 : 41 ∣ M := by sorry

end NUMINAMATH_CALUDE_M_divisible_by_41_l2119_211958


namespace NUMINAMATH_CALUDE_solve_for_A_l2119_211934

theorem solve_for_A : ∃ A : ℝ, 4 * A + 5 = 33 ∧ A = 7 := by
  sorry

end NUMINAMATH_CALUDE_solve_for_A_l2119_211934


namespace NUMINAMATH_CALUDE_fraction_product_square_l2119_211989

theorem fraction_product_square : (8 / 9 : ℚ)^2 * (1 / 3 : ℚ)^2 = 64 / 729 := by
  sorry

end NUMINAMATH_CALUDE_fraction_product_square_l2119_211989


namespace NUMINAMATH_CALUDE_power_fraction_equality_l2119_211942

theorem power_fraction_equality : (88888 ^ 5 : ℚ) / (22222 ^ 5) = 1024 := by
  sorry

end NUMINAMATH_CALUDE_power_fraction_equality_l2119_211942


namespace NUMINAMATH_CALUDE_function_inequality_implies_a_bound_l2119_211949

theorem function_inequality_implies_a_bound (a : ℝ) : 
  (∀ x₁ x₂ : ℝ, 0 ≤ x₁ ∧ x₁ ≤ 2 ∧ 0 ≤ x₂ ∧ x₂ ≤ 2 → 
    x₁^3 - 3*x₁ ≤ Real.exp x₂ - 2*a*x₂ + 2) → 
  a ≤ Real.exp 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_function_inequality_implies_a_bound_l2119_211949
