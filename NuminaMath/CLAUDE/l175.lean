import Mathlib

namespace NUMINAMATH_CALUDE_book_selection_theorem_l175_17502

/-- The number of ways to select 5 books from 10 books with specific conditions -/
def select_books (n : ℕ) (k : ℕ) (adjacent_pairs : ℕ) (remaining : ℕ) : ℕ :=
  adjacent_pairs * Nat.choose remaining (k - 2)

/-- Theorem stating the number of ways to select 5 books from 10 books 
    where order doesn't matter and two of the selected books must be adjacent -/
theorem book_selection_theorem :
  select_books 10 5 9 8 = 504 := by
  sorry

end NUMINAMATH_CALUDE_book_selection_theorem_l175_17502


namespace NUMINAMATH_CALUDE_expected_red_lights_value_l175_17596

/-- The number of traffic posts -/
def n : ℕ := 3

/-- The probability of encountering a red light at each post -/
def p : ℝ := 0.4

/-- The expected number of red lights encountered -/
def expected_red_lights : ℝ := n * p

/-- Theorem: The expected number of red lights encountered is 1.2 -/
theorem expected_red_lights_value : expected_red_lights = 1.2 := by
  sorry

end NUMINAMATH_CALUDE_expected_red_lights_value_l175_17596


namespace NUMINAMATH_CALUDE_correct_completion_for_two_viewers_l175_17592

/-- Represents the options for completing the sentence --/
inductive SentenceCompletion
  | NoneOfThem
  | BothOfThem
  | NoneOfWhom
  | NeitherOfWhom

/-- Represents a person who looked at the house --/
structure HouseViewer where
  wantsToBuy : Bool

/-- The correct sentence completion given two house viewers --/
def correctCompletion (viewer1 viewer2 : HouseViewer) : SentenceCompletion :=
  if !viewer1.wantsToBuy ∧ !viewer2.wantsToBuy then
    SentenceCompletion.NeitherOfWhom
  else
    SentenceCompletion.BothOfThem  -- This else case is not actually used in our theorem

theorem correct_completion_for_two_viewers (viewer1 viewer2 : HouseViewer) 
  (h1 : ¬viewer1.wantsToBuy) (h2 : ¬viewer2.wantsToBuy) :
  correctCompletion viewer1 viewer2 = SentenceCompletion.NeitherOfWhom :=
by sorry

end NUMINAMATH_CALUDE_correct_completion_for_two_viewers_l175_17592


namespace NUMINAMATH_CALUDE_not_right_angled_triangle_l175_17549

theorem not_right_angled_triangle : ∃ (a b c : ℝ),
  ((a = 30 ∧ b = 60 ∧ c = 90) → a^2 + b^2 ≠ c^2) ∧
  ((a = 3*Real.sqrt 2 ∧ b = 4*Real.sqrt 2 ∧ c = 5*Real.sqrt 2) → a^2 + b^2 = c^2) ∧
  ((a = 1 ∧ b = Real.sqrt 2 ∧ c = Real.sqrt 3) → a^2 + b^2 = c^2) ∧
  ((a = 5 ∧ b = 12 ∧ c = 13) → a^2 + b^2 = c^2) :=
by sorry

end NUMINAMATH_CALUDE_not_right_angled_triangle_l175_17549


namespace NUMINAMATH_CALUDE_quadratic_roots_property_l175_17539

theorem quadratic_roots_property (d e : ℝ) : 
  (3 * d^2 + 5 * d - 7 = 0) → 
  (3 * e^2 + 5 * e - 7 = 0) → 
  (d - 2) * (e - 2) = 5 := by
sorry

end NUMINAMATH_CALUDE_quadratic_roots_property_l175_17539


namespace NUMINAMATH_CALUDE_odd_k_triple_f_35_l175_17572

def f (n : ℤ) : ℤ :=
  if n % 2 = 1 then n + 5 else n - 2

theorem odd_k_triple_f_35 (k : ℤ) (h1 : k % 2 = 1) (h2 : f (f (f k)) = 35) : k = 29 := by
  sorry

end NUMINAMATH_CALUDE_odd_k_triple_f_35_l175_17572


namespace NUMINAMATH_CALUDE_cos_24_minus_cos_48_l175_17563

theorem cos_24_minus_cos_48 : Real.cos (24 * Real.pi / 180) - Real.cos (48 * Real.pi / 180) = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_cos_24_minus_cos_48_l175_17563


namespace NUMINAMATH_CALUDE_square_inequality_l175_17591

theorem square_inequality (a x y : ℝ) :
  (2 ≤ x ∧ x ≤ 3) ∧ (3 ≤ y ∧ y ≤ 4) →
  ((3 * x - 2 * y - a) * (3 * x - 2 * y - a^2) ≤ 0 ↔ a ≤ -4) :=
by sorry

end NUMINAMATH_CALUDE_square_inequality_l175_17591


namespace NUMINAMATH_CALUDE_family_tickets_count_l175_17535

theorem family_tickets_count :
  let adult_ticket_cost : ℕ := 19
  let child_ticket_cost : ℕ := 13
  let adult_count : ℕ := 2
  let child_count : ℕ := 3
  let total_cost : ℕ := 77
  adult_ticket_cost = child_ticket_cost + 6 ∧
  total_cost = adult_count * adult_ticket_cost + child_count * child_ticket_cost →
  adult_count + child_count = 5 :=
by
  sorry

end NUMINAMATH_CALUDE_family_tickets_count_l175_17535


namespace NUMINAMATH_CALUDE_sample_size_C_l175_17533

def total_students : ℕ := 150 + 150 + 400 + 300
def students_in_C : ℕ := 400
def total_survey_size : ℕ := 40

theorem sample_size_C : 
  (students_in_C * total_survey_size) / total_students = 16 :=
sorry

end NUMINAMATH_CALUDE_sample_size_C_l175_17533


namespace NUMINAMATH_CALUDE_existence_of_symmetric_axis_l175_17571

/-- Represents the color of a stone -/
inductive Color
| Black
| White

/-- Represents a regular 13-gon with colored stones at each vertex -/
def Regular13Gon := Fin 13 → Color

/-- Counts the number of symmetric pairs with the same color for a given axis -/
def symmetricPairsCount (polygon : Regular13Gon) (axis : Fin 13) : ℕ :=
  sorry

/-- Main theorem: There exists an axis with at least 4 symmetric pairs of the same color -/
theorem existence_of_symmetric_axis (polygon : Regular13Gon) :
  ∃ axis : Fin 13, symmetricPairsCount polygon axis ≥ 4 := by
  sorry

end NUMINAMATH_CALUDE_existence_of_symmetric_axis_l175_17571


namespace NUMINAMATH_CALUDE_water_tank_capacity_l175_17564

/-- Represents a cylindrical water tank --/
structure WaterTank where
  capacity : ℝ
  initialWater : ℝ
  finalWater : ℝ

/-- Proves that the water tank has a capacity of 75 liters --/
theorem water_tank_capacity (tank : WaterTank)
  (h1 : tank.initialWater / tank.capacity = 1 / 3)
  (h2 : (tank.initialWater + 5) / tank.capacity = 2 / 5) :
  tank.capacity = 75 := by
  sorry

end NUMINAMATH_CALUDE_water_tank_capacity_l175_17564


namespace NUMINAMATH_CALUDE_sum_of_even_integers_l175_17508

theorem sum_of_even_integers (n : ℕ) (sum_first_n : ℕ) (first : ℕ) (last : ℕ) :
  n = 50 →
  sum_first_n = 2550 →
  first = 102 →
  last = 200 →
  (n : ℕ) * (2 + 2 * n) = 2 * sum_first_n →
  (last - first) / 2 + 1 = n →
  n / 2 * (first + last) = 7550 :=
by sorry

end NUMINAMATH_CALUDE_sum_of_even_integers_l175_17508


namespace NUMINAMATH_CALUDE_triangle_angle_c_is_sixty_degrees_l175_17524

theorem triangle_angle_c_is_sixty_degrees 
  (A B C : ℝ) (a b c : ℝ) 
  (h_triangle : 0 < A ∧ 0 < B ∧ 0 < C ∧ A + B + C = π)
  (h_sides : 0 < a ∧ 0 < b ∧ 0 < c)
  (h_sin : Real.sin A = 2 * Real.sin B)
  (h_sum : a + b = Real.sqrt 3 * c) :
  C = π / 3 := by
sorry

end NUMINAMATH_CALUDE_triangle_angle_c_is_sixty_degrees_l175_17524


namespace NUMINAMATH_CALUDE_sum_negative_implies_one_negative_l175_17500

theorem sum_negative_implies_one_negative (a b : ℚ) : a + b < 0 → a < 0 ∨ b < 0 := by
  sorry

end NUMINAMATH_CALUDE_sum_negative_implies_one_negative_l175_17500


namespace NUMINAMATH_CALUDE_line_slope_is_one_l175_17559

/-- The slope of a line in the xy-plane with y-intercept -2 and passing through 
    the midpoint of the line segment with endpoints (2, 8) and (8, -2) is 1. -/
theorem line_slope_is_one : 
  ∀ (m : Set (ℝ × ℝ)), 
    (∀ (x y : ℝ), (x, y) ∈ m → y = x - 2) →  -- y-intercept is -2
    ((5 : ℝ), 3) ∈ m →  -- passes through midpoint (5, 3)
    (∃ (k : ℝ), ∀ (x y : ℝ), (x, y) ∈ m → y = k * x - 2) →  -- line equation
    (∀ (x y : ℝ), (x, y) ∈ m → y = x - 2) :=  -- slope is 1
by sorry

end NUMINAMATH_CALUDE_line_slope_is_one_l175_17559


namespace NUMINAMATH_CALUDE_order_of_logarithmic_fractions_l175_17561

theorem order_of_logarithmic_fractions :
  let a := (Real.log 2) / 2
  let b := (Real.log 3) / 3
  let c := 1 / Real.exp 1
  a < b ∧ b < c := by sorry

end NUMINAMATH_CALUDE_order_of_logarithmic_fractions_l175_17561


namespace NUMINAMATH_CALUDE_perpendicular_slope_l175_17565

/-- The slope of a line perpendicular to a line passing through two given points -/
theorem perpendicular_slope (x₁ y₁ x₂ y₂ : ℝ) (h : x₁ ≠ x₂) :
  let m := (y₂ - y₁) / (x₂ - x₁)
  (- 1 / m) = 4 / 3 →
  x₁ = 3 ∧ y₁ = -7 ∧ x₂ = -5 ∧ y₂ = -1 :=
by sorry

end NUMINAMATH_CALUDE_perpendicular_slope_l175_17565


namespace NUMINAMATH_CALUDE_consecutive_numbers_sum_l175_17505

theorem consecutive_numbers_sum (n : ℕ) : 
  (n + (n + 1) + (n + 2) + (n + 3) + (n + 4) + (n + 5) = 105) → 
  ((n + 5) - n = 5) := by
  sorry

end NUMINAMATH_CALUDE_consecutive_numbers_sum_l175_17505


namespace NUMINAMATH_CALUDE_sqrt_square_abs_l175_17583

theorem sqrt_square_abs (x : ℝ) : Real.sqrt (x ^ 2) = |x| := by
  sorry

end NUMINAMATH_CALUDE_sqrt_square_abs_l175_17583


namespace NUMINAMATH_CALUDE_inequality_range_l175_17523

theorem inequality_range (a : ℝ) : 
  (∀ x : ℝ, a * x^2 - |x + 1| + 3 * a ≥ 0) ↔ a ≥ 1/2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_range_l175_17523


namespace NUMINAMATH_CALUDE_infinitely_many_divisible_by_15_l175_17553

def v : ℕ → ℤ
  | 0 => 0
  | 1 => 1
  | (n + 2) => 8 * v (n + 1) - v n

theorem infinitely_many_divisible_by_15 :
  ∀ k : ℕ, ∃ n : ℕ, n > k ∧ 15 ∣ v n :=
sorry

end NUMINAMATH_CALUDE_infinitely_many_divisible_by_15_l175_17553


namespace NUMINAMATH_CALUDE_pairing_fraction_l175_17552

/-- Represents the number of students in each grade --/
structure Students where
  seventh : ℕ
  tenth : ℕ

/-- Represents the pairing between seventh and tenth graders --/
def Pairing (s : Students) :=
  (s.tenth / 4 : ℚ) = (s.seventh / 3 : ℚ)

/-- Calculates the fraction of students with partners --/
def fractionWithPartners (s : Students) : ℚ :=
  (s.tenth / 4 + s.seventh / 3) / (s.tenth + s.seventh)

theorem pairing_fraction (s : Students) (h : Pairing s) :
  fractionWithPartners s = 2 / 7 := by
  sorry


end NUMINAMATH_CALUDE_pairing_fraction_l175_17552


namespace NUMINAMATH_CALUDE_megacorp_mining_earnings_l175_17543

/-- MegaCorp's daily earnings from mining -/
def daily_mining_earnings : ℝ := 67111111.11

/-- MegaCorp's daily earnings from oil refining -/
def daily_oil_earnings : ℝ := 5000000

/-- MegaCorp's monthly expenses -/
def monthly_expenses : ℝ := 30000000

/-- MegaCorp's fine -/
def fine : ℝ := 25600000

/-- The fine percentage of annual profits -/
def fine_percentage : ℝ := 0.01

/-- Number of days in a month (approximation) -/
def days_in_month : ℝ := 30

/-- Number of months in a year -/
def months_in_year : ℝ := 12

theorem megacorp_mining_earnings :
  fine = fine_percentage * months_in_year * (days_in_month * (daily_mining_earnings + daily_oil_earnings) - monthly_expenses) :=
by sorry

end NUMINAMATH_CALUDE_megacorp_mining_earnings_l175_17543


namespace NUMINAMATH_CALUDE_smallest_sum_after_slice_l175_17597

-- Define the structure of a die
structure Die :=
  (faces : Fin 6 → Nat)
  (opposite_sum : ∀ i : Fin 6, faces i + faces (5 - i) = 7)

-- Define the cube structure
structure Cube :=
  (dice : Fin 27 → Die)

-- Define the function to calculate the sum of visible faces
def sum_visible_faces (c : Cube) : Nat :=
  -- Implementation details omitted
  sorry

-- Main theorem
theorem smallest_sum_after_slice (c : Cube) : sum_visible_faces c ≥ 98 :=
  sorry

end NUMINAMATH_CALUDE_smallest_sum_after_slice_l175_17597


namespace NUMINAMATH_CALUDE_sum_value_theorem_l175_17509

theorem sum_value_theorem (a b c : ℚ) (h1 : |a + 1| + (b - 2)^2 = 0) (h2 : |c| = 3) :
  a + b + 2*c = 7 ∨ a + b + 2*c = -5 := by
  sorry

end NUMINAMATH_CALUDE_sum_value_theorem_l175_17509


namespace NUMINAMATH_CALUDE_proper_subset_of_A_l175_17568

def A : Set ℝ := { x | x^2 < 5*x }

theorem proper_subset_of_A : Set.Subset (Set.Ioo 1 5) A ∧ (Set.Ioo 1 5) ≠ A := by sorry

end NUMINAMATH_CALUDE_proper_subset_of_A_l175_17568


namespace NUMINAMATH_CALUDE_circle_center_sum_l175_17531

/-- Given a circle with equation x^2 + y^2 - 6x + 8y - 24 = 0, 
    prove that the sum of the coordinates of its center is -1 -/
theorem circle_center_sum (h k : ℝ) : 
  (∀ x y : ℝ, x^2 + y^2 - 6*x + 8*y - 24 = 0 ↔ (x - h)^2 + (y - k)^2 = (h^2 + k^2 - 24 : ℝ)) →
  h + k = -1 := by
sorry

end NUMINAMATH_CALUDE_circle_center_sum_l175_17531


namespace NUMINAMATH_CALUDE_line_point_theorem_l175_17507

/-- The line equation y = -2/3x + 10 -/
def line_equation (x y : ℝ) : Prop := y = -2/3 * x + 10

/-- Point P is where the line crosses the x-axis -/
def point_P : ℝ × ℝ := (15, 0)

/-- Point Q is where the line crosses the y-axis -/
def point_Q : ℝ × ℝ := (0, 10)

/-- Point T is on the line segment PQ -/
def point_T_on_PQ (r s : ℝ) : Prop :=
  ∃ (t : ℝ), 0 ≤ t ∧ t ≤ 1 ∧ 
  r = t * point_P.1 + (1 - t) * point_Q.1 ∧
  s = t * point_P.2 + (1 - t) * point_Q.2

/-- The area of triangle POQ is four times the area of triangle TOP -/
def area_condition (r s : ℝ) : Prop :=
  abs (point_P.1 * point_Q.2 - point_Q.1 * point_P.2) / 2 = 
  4 * abs (r * point_P.2 - point_P.1 * s) / 2

/-- Main theorem -/
theorem line_point_theorem (r s : ℝ) :
  line_equation r s →
  point_T_on_PQ r s →
  area_condition r s →
  r + s = 13.75 := by sorry

end NUMINAMATH_CALUDE_line_point_theorem_l175_17507


namespace NUMINAMATH_CALUDE_dot_product_problem_l175_17569

theorem dot_product_problem (b : ℝ × ℝ) : 
  let a : ℝ × ℝ := (1, 2)
  (a.1 + b.1, a.2 + b.2) = (-1, 1) →
  a.1 * b.1 + a.2 * b.2 = -4 :=
by
  sorry

end NUMINAMATH_CALUDE_dot_product_problem_l175_17569


namespace NUMINAMATH_CALUDE_cereal_box_servings_l175_17593

/-- Calculates the number of servings in a cereal box -/
def servings_in_box (total_cups : ℕ) (cups_per_serving : ℕ) : ℕ :=
  total_cups / cups_per_serving

/-- Theorem: The number of servings in a cereal box that holds 18 cups,
    with each serving being 2 cups, is 9. -/
theorem cereal_box_servings :
  servings_in_box 18 2 = 9 := by
  sorry

end NUMINAMATH_CALUDE_cereal_box_servings_l175_17593


namespace NUMINAMATH_CALUDE_jeds_speed_jeds_speed_is_89_l175_17541

def speed_limit : ℕ := 50
def speeding_fine_per_mph : ℕ := 16
def red_light_fine : ℕ := 75
def cellphone_fine : ℕ := 120
def parking_fine : ℕ := 50
def total_fine : ℕ := 1046
def red_light_violations : ℕ := 2
def parking_violations : ℕ := 3

theorem jeds_speed : ℕ :=
  let non_speeding_fines := red_light_fine * red_light_violations + 
                            cellphone_fine + 
                            parking_fine * parking_violations
  let speeding_fine := total_fine - non_speeding_fines
  let mph_over_limit := speeding_fine / speeding_fine_per_mph
  speed_limit + mph_over_limit

#check jeds_speed

theorem jeds_speed_is_89 : jeds_speed = 89 := by
  sorry

end NUMINAMATH_CALUDE_jeds_speed_jeds_speed_is_89_l175_17541


namespace NUMINAMATH_CALUDE_water_pumped_30_minutes_l175_17588

/-- Represents a water pumping system -/
structure WaterPump where
  gallons_per_hour : ℝ

/-- Calculates the amount of water pumped in a given time -/
def water_pumped (pump : WaterPump) (hours : ℝ) : ℝ :=
  pump.gallons_per_hour * hours

theorem water_pumped_30_minutes (pump : WaterPump) 
  (h : pump.gallons_per_hour = 500) : 
  water_pumped pump (30 / 60) = 250 := by
  sorry

#check water_pumped_30_minutes

end NUMINAMATH_CALUDE_water_pumped_30_minutes_l175_17588


namespace NUMINAMATH_CALUDE_twelve_divisor_number_is_1989_l175_17577

/-- The type of natural numbers with exactly 12 positive divisors. -/
def TwelveDivisorNumber (N : ℕ) : Prop :=
  (∃ (d : Fin 12 → ℕ), 
    (∀ i j, i < j → d i < d j) ∧
    (∀ i, d i ∣ N) ∧
    (∀ m, m ∣ N → ∃ i, d i = m) ∧
    (d 0 = 1) ∧
    (d 11 = N))

/-- The property that the divisor with index d₄ - 1 is equal to (d₁ + d₂ + d₄) · d₈ -/
def SpecialDivisorProperty (N : ℕ) (d : Fin 12 → ℕ) : Prop :=
  d ((d 3 : ℕ) - 1) = (d 0 + d 1 + d 3) * d 7

theorem twelve_divisor_number_is_1989 :
  ∃ N : ℕ, TwelveDivisorNumber N ∧ 
    (∃ d : Fin 12 → ℕ, SpecialDivisorProperty N d) ∧
    N = 1989 := by
  sorry

end NUMINAMATH_CALUDE_twelve_divisor_number_is_1989_l175_17577


namespace NUMINAMATH_CALUDE_negative_of_negative_two_equals_two_l175_17506

theorem negative_of_negative_two_equals_two : -(-2) = 2 := by sorry

end NUMINAMATH_CALUDE_negative_of_negative_two_equals_two_l175_17506


namespace NUMINAMATH_CALUDE_parallel_line_correct_perpendicular_line_correct_l175_17522

-- Define the given line
def given_line (x y : ℝ) : Prop := 2 * x - y - 1 = 0

-- Define the point (1,0)
def point : ℝ × ℝ := (1, 0)

-- Define parallel line
def parallel_line (x y : ℝ) : Prop := 2 * x - y - 2 = 0

-- Define perpendicular line
def perpendicular_line (x y : ℝ) : Prop := x + 2 * y - 1 = 0

-- Theorem for parallel line
theorem parallel_line_correct :
  (∀ x y : ℝ, parallel_line x y ↔ (given_line x y ∧ parallel_line x y)) ∧
  parallel_line point.1 point.2 :=
sorry

-- Theorem for perpendicular line
theorem perpendicular_line_correct :
  (∀ x y : ℝ, perpendicular_line x y ↔ (given_line x y ∧ perpendicular_line x y)) ∧
  perpendicular_line point.1 point.2 :=
sorry

end NUMINAMATH_CALUDE_parallel_line_correct_perpendicular_line_correct_l175_17522


namespace NUMINAMATH_CALUDE_town_population_distribution_l175_17576

theorem town_population_distribution (total_population : ℕ) 
  (h1 : total_population = 600) 
  (h2 : ∃ (males females children : ℕ), 
    males + females + children = total_population ∧ 
    children = 2 * males ∧ 
    males + females + children = 4 * males) : 
  ∃ (males : ℕ), males = 150 := by
sorry

end NUMINAMATH_CALUDE_town_population_distribution_l175_17576


namespace NUMINAMATH_CALUDE_arithmetic_sequence_60th_term_l175_17515

/-- Given an arithmetic sequence with first term 7 and 21st term 47, prove that the 60th term is 125 -/
theorem arithmetic_sequence_60th_term : 
  ∀ (a : ℕ → ℝ), 
    (∀ n : ℕ, a (n + 1) - a n = a 1 - a 0) →  -- arithmetic sequence condition
    a 0 = 7 →                                -- first term
    a 20 = 47 →                              -- 21st term (index starts at 0)
    a 59 = 125 :=                            -- 60th term (index starts at 0)
by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_60th_term_l175_17515


namespace NUMINAMATH_CALUDE_mike_marbles_l175_17548

theorem mike_marbles (given_away : ℕ) (remaining : ℕ) : 
  given_away = 4 → remaining = 4 → given_away + remaining = 8 := by
  sorry

end NUMINAMATH_CALUDE_mike_marbles_l175_17548


namespace NUMINAMATH_CALUDE_relationship_proof_l175_17558

open Real

noncomputable def f (x : ℝ) := Real.exp x + x - 2
noncomputable def g (x : ℝ) := Real.log x + x^2 - 3

theorem relationship_proof (a b : ℝ) (ha : f a = 0) (hb : g b = 0) :
  g a < 0 ∧ 0 < f b := by sorry

end NUMINAMATH_CALUDE_relationship_proof_l175_17558


namespace NUMINAMATH_CALUDE_chord_length_limit_l175_17582

theorem chord_length_limit (r : ℝ) (chord_length : ℝ) :
  r = 6 →
  chord_length ≤ 2 * r →
  chord_length ≠ 14 :=
by sorry

end NUMINAMATH_CALUDE_chord_length_limit_l175_17582


namespace NUMINAMATH_CALUDE_unique_five_digit_number_l175_17567

def is_valid_number (n : ℕ) : Prop :=
  ∃ (x y : ℕ),
    n = 10 * x + y ∧
    0 ≤ y ∧ y ≤ 9 ∧
    10000 ≤ n ∧ n ≤ 99999 ∧
    1000 ≤ x ∧ x ≤ 9999 ∧
    n - x = 54321

theorem unique_five_digit_number : 
  ∃! (n : ℕ), is_valid_number n ∧ n = 60356 :=
sorry

end NUMINAMATH_CALUDE_unique_five_digit_number_l175_17567


namespace NUMINAMATH_CALUDE_circular_film_radius_l175_17526

/-- Given a cylindrical canister filled with a liquid that forms a circular film on water,
    this theorem proves that the radius of the resulting circular film is 25√2 cm. -/
theorem circular_film_radius
  (canister_radius : ℝ)
  (canister_height : ℝ)
  (film_thickness : ℝ)
  (h_canister_radius : canister_radius = 5)
  (h_canister_height : canister_height = 10)
  (h_film_thickness : film_thickness = 0.2) :
  let canister_volume := π * canister_radius^2 * canister_height
  let film_radius := Real.sqrt (canister_volume / (π * film_thickness))
  film_radius = 25 * Real.sqrt 2 := by
sorry

end NUMINAMATH_CALUDE_circular_film_radius_l175_17526


namespace NUMINAMATH_CALUDE_power_sum_fifth_l175_17540

theorem power_sum_fifth (a b x y : ℝ) 
  (h1 : a*x + b*y = 1)
  (h2 : a*x^2 + b*y^2 = 9)
  (h3 : a*x^3 + b*y^3 = 28)
  (h4 : a*x^4 + b*y^4 = 96) :
  a*x^5 + b*y^5 = 28616 := by
  sorry

end NUMINAMATH_CALUDE_power_sum_fifth_l175_17540


namespace NUMINAMATH_CALUDE_color_one_third_square_l175_17575

theorem color_one_third_square (n : ℕ) (k : ℕ) : n = 18 ∧ k = 6 → Nat.choose n k = 18564 := by
  sorry

end NUMINAMATH_CALUDE_color_one_third_square_l175_17575


namespace NUMINAMATH_CALUDE_polynomial_composition_pairs_l175_17520

theorem polynomial_composition_pairs :
  ∀ (a b : ℝ),
    (∃ (P : ℝ → ℝ),
      (∀ x, P (P x) = x^4 - 8*x^3 + a*x^2 + b*x + 40) ∧
      (∃ (c d : ℝ), ∀ x, P x = x^2 + c*x + d)) ↔
    ((a = 28 ∧ b = -48) ∨ (a = 2 ∧ b = 56)) :=
by sorry

end NUMINAMATH_CALUDE_polynomial_composition_pairs_l175_17520


namespace NUMINAMATH_CALUDE_binomial_expansion_sum_l175_17510

theorem binomial_expansion_sum (a : ℝ) (a₀ a₁ a₂ a₃ a₄ a₅ a₆ a₇ a₈ : ℝ) :
  (∀ x : ℝ, (a - x)^8 = a₀ + a₁*x + a₂*x^2 + a₃*x^3 + a₄*x^4 + a₅*x^5 + a₆*x^6 + a₇*x^7 + a₈*x^8) →
  a₅ = 56 →
  a₀ + a₁ + a₂ + a₃ + a₄ + a₅ + a₆ + a₇ + a₈ = 256 := by
sorry

end NUMINAMATH_CALUDE_binomial_expansion_sum_l175_17510


namespace NUMINAMATH_CALUDE_diagonals_150_sided_polygon_l175_17595

/-- Number of diagonals in a polygon with n sides -/
def num_diagonals (n : ℕ) : ℕ := n * (n - 3) / 2

/-- The number of diagonals in a 150-sided polygon is 11025 -/
theorem diagonals_150_sided_polygon :
  num_diagonals 150 = 11025 := by
  sorry

end NUMINAMATH_CALUDE_diagonals_150_sided_polygon_l175_17595


namespace NUMINAMATH_CALUDE_matrix_power_50_l175_17566

/-- Given a 2x2 matrix C, prove that its 50th power is equal to a specific matrix. -/
theorem matrix_power_50 (C : Matrix (Fin 2) (Fin 2) ℤ) : 
  C = !![5, 2; -16, -6] → C^50 = !![-299, -100; 800, 249] := by
  sorry

end NUMINAMATH_CALUDE_matrix_power_50_l175_17566


namespace NUMINAMATH_CALUDE_intersection_M_N_l175_17554

-- Define the sets M and N
def M : Set ℝ := {y | ∃ x, y = x^2 + 1}
def N : Set ℝ := {y | ∃ x, y = x + 1}

-- State the theorem
theorem intersection_M_N : M ∩ N = {y | y ≥ 1} := by sorry

end NUMINAMATH_CALUDE_intersection_M_N_l175_17554


namespace NUMINAMATH_CALUDE_tino_jellybean_count_l175_17574

/-- The number of jellybeans each person has -/
structure JellybeanCount where
  tino : ℕ
  lee : ℕ
  arnold : ℕ

/-- The conditions of the jellybean problem -/
def jellybean_conditions (j : JellybeanCount) : Prop :=
  j.tino = j.lee + 24 ∧
  j.arnold = j.lee / 2 ∧
  j.arnold = 5

/-- Theorem stating that under the given conditions, Tino has 34 jellybeans -/
theorem tino_jellybean_count (j : JellybeanCount) 
  (h : jellybean_conditions j) : j.tino = 34 := by
  sorry

end NUMINAMATH_CALUDE_tino_jellybean_count_l175_17574


namespace NUMINAMATH_CALUDE_min_value_theorem_l175_17530

/-- Given that the solution set of (x+2)/(x+1) < 0 is {x | a < x < b},
    and point A(a,b) lies on the line mx + ny + 1 = 0 where mn > 0,
    prove that the minimum value of 2/m + 1/n is 9. -/
theorem min_value_theorem (a b m n : ℝ) : 
  (∀ x, (x + 2) / (x + 1) < 0 ↔ a < x ∧ x < b) →
  m * a + n * b + 1 = 0 →
  m * n > 0 →
  (∀ m' n', m' * n' > 0 → 2 / m' + 1 / n' ≥ 2 / m + 1 / n) →
  2 / m + 1 / n = 9 :=
by sorry

end NUMINAMATH_CALUDE_min_value_theorem_l175_17530


namespace NUMINAMATH_CALUDE_min_product_of_three_l175_17504

def S : Set Int := {-10, -7, -3, 0, 4, 6, 9}

theorem min_product_of_three (a b c : Int) (ha : a ∈ S) (hb : b ∈ S) (hc : c ∈ S) 
  (hab : a ≠ b) (hbc : b ≠ c) (hac : a ≠ c) :
  ∃ (x y z : Int), x ∈ S ∧ y ∈ S ∧ z ∈ S ∧ x ≠ y ∧ y ≠ z ∧ x ≠ z ∧
  x * y * z = -540 ∧ 
  ∀ (p q r : Int), p ∈ S → q ∈ S → r ∈ S → p ≠ q → q ≠ r → p ≠ r → 
  p * q * r ≥ -540 :=
sorry

end NUMINAMATH_CALUDE_min_product_of_three_l175_17504


namespace NUMINAMATH_CALUDE_doll_collection_l175_17562

theorem doll_collection (jazmin_dolls geraldine_dolls : ℕ) 
  (h1 : jazmin_dolls = 1209) 
  (h2 : geraldine_dolls = 2186) : 
  jazmin_dolls + geraldine_dolls = 3395 := by
  sorry

end NUMINAMATH_CALUDE_doll_collection_l175_17562


namespace NUMINAMATH_CALUDE_largest_square_size_l175_17586

theorem largest_square_size (board_length board_width : ℕ) 
  (h1 : board_length = 77) (h2 : board_width = 93) :
  Nat.gcd board_length board_width = 1 := by
  sorry

end NUMINAMATH_CALUDE_largest_square_size_l175_17586


namespace NUMINAMATH_CALUDE_led_messages_count_l175_17525

/-- Represents the number of LEDs in the row -/
def n : ℕ := 7

/-- Represents the number of LEDs that are lit -/
def k : ℕ := 3

/-- Represents the number of color options for each lit LED -/
def colors : ℕ := 2

/-- Calculates the number of ways to choose k items from n items -/
def choose (n k : ℕ) : ℕ := Nat.choose n k

/-- Calculates the number of ways to arrange k non-adjacent items in n+1 slots -/
def nonAdjacentArrangements (n k : ℕ) : ℕ := choose (n + 1 - k) k

/-- Calculates the total number of different messages -/
def totalMessages : ℕ := nonAdjacentArrangements n k * colors^k

theorem led_messages_count : totalMessages = 80 := by
  sorry

end NUMINAMATH_CALUDE_led_messages_count_l175_17525


namespace NUMINAMATH_CALUDE_three_digit_number_proof_l175_17551

theorem three_digit_number_proof (a : Nat) (h1 : a < 10) : 
  (100 * a + 10 * a + 5) % 9 = 8 → 100 * a + 10 * a + 5 = 665 := by
  sorry

end NUMINAMATH_CALUDE_three_digit_number_proof_l175_17551


namespace NUMINAMATH_CALUDE_distance_between_points_l175_17527

/-- The distance between points A and B given specific square dimensions -/
theorem distance_between_points (small_perimeter : ℝ) (large_area : ℝ) : 
  small_perimeter = 8 → large_area = 25 → ∃ (dist : ℝ), dist^2 = 58 := by
  sorry

end NUMINAMATH_CALUDE_distance_between_points_l175_17527


namespace NUMINAMATH_CALUDE_range_of_a_l175_17518

/-- The range of values for a given the conditions -/
theorem range_of_a (f g : ℝ → ℝ) (a : ℝ) : 
  (∀ x > 0, f x = x * Real.log x) →
  (∀ x, g x = x^3 + a*x - x + 2) →
  (∀ x > 0, 2 * f x ≤ deriv g x + 2) →
  a ≥ -2 ∧ ∀ b ≥ -2, ∃ x > 0, 2 * f x ≤ deriv g x + 2 :=
by sorry


end NUMINAMATH_CALUDE_range_of_a_l175_17518


namespace NUMINAMATH_CALUDE_min_value_of_function_min_value_achieved_l175_17594

theorem min_value_of_function (x : ℝ) (h : x > 1) :
  (x^2 + x + 1) / (x - 1) ≥ 3 + 2 * Real.sqrt 3 :=
sorry

theorem min_value_achieved (x : ℝ) (h : x > 1) :
  ∃ x₀ > 1, (x₀^2 + x₀ + 1) / (x₀ - 1) = 3 + 2 * Real.sqrt 3 :=
sorry

end NUMINAMATH_CALUDE_min_value_of_function_min_value_achieved_l175_17594


namespace NUMINAMATH_CALUDE_triangle_side_length_l175_17501

theorem triangle_side_length (A B C a b c : ℝ) : 
  A + C = 2 * B → 
  a + c = 8 → 
  a * c = 15 → 
  b = Real.sqrt 19 := by
  sorry

end NUMINAMATH_CALUDE_triangle_side_length_l175_17501


namespace NUMINAMATH_CALUDE_committee_selection_l175_17511

theorem committee_selection (n : ℕ) (k : ℕ) (h1 : n = 12) (h2 : k = 5) : 
  Nat.choose n k = 792 :=
by sorry

end NUMINAMATH_CALUDE_committee_selection_l175_17511


namespace NUMINAMATH_CALUDE_no_intersection_intersection_count_is_zero_l175_17529

-- Define the two functions
def f (x : ℝ) : ℝ := |3 * x + 6|
def g (x : ℝ) : ℝ := -|4 * x - 1|

-- Theorem statement
theorem no_intersection :
  ∀ x : ℝ, f x ≠ g x :=
by
  sorry

-- Define the number of intersection points
def intersection_count : ℕ := 0

-- Theorem to prove the number of intersection points is 0
theorem intersection_count_is_zero :
  intersection_count = 0 :=
by
  sorry

end NUMINAMATH_CALUDE_no_intersection_intersection_count_is_zero_l175_17529


namespace NUMINAMATH_CALUDE_cube_difference_square_root_l175_17581

theorem cube_difference_square_root : ∃ (n : ℕ), n > 0 ∧ n^2 = 105^3 - 104^3 :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_cube_difference_square_root_l175_17581


namespace NUMINAMATH_CALUDE_three_conditions_theorem_l175_17599

def condition1 (a b : ℕ) : Prop := (a^2 + 6*a + 8) % b = 0

def condition2 (a b : ℕ) : Prop := a^2 + a*b - 6*b^2 - 15*b - 9 = 0

def condition3 (a b : ℕ) : Prop := (a + 2*b + 2) % 4 = 0

def condition4 (a b : ℕ) : Prop := Nat.Prime (a + 6*b + 2)

def satisfiesThreeConditions (a b : ℕ) : Prop :=
  (condition1 a b ∧ condition2 a b ∧ condition3 a b) ∨
  (condition1 a b ∧ condition2 a b ∧ condition4 a b) ∨
  (condition1 a b ∧ condition3 a b ∧ condition4 a b) ∨
  (condition2 a b ∧ condition3 a b ∧ condition4 a b)

theorem three_conditions_theorem :
  ∀ a b : ℕ, satisfiesThreeConditions a b ↔ ((a = 5 ∧ b = 1) ∨ (a = 17 ∧ b = 7)) :=
sorry

end NUMINAMATH_CALUDE_three_conditions_theorem_l175_17599


namespace NUMINAMATH_CALUDE_hyperbola_asymptote_l175_17519

/-- Given a hyperbola with equation x²/a² - y²/9 = 1 where a > 0,
    if its asymptotes are given by 2x ± 3y = 0, then a = 3 -/
theorem hyperbola_asymptote (a : ℝ) (h1 : a > 0) :
  (∀ x y : ℝ, x^2 / a^2 - y^2 / 9 = 1 ↔ (2*x + 3*y = 0 ∨ 2*x - 3*y = 0)) →
  a = 3 := by
sorry

end NUMINAMATH_CALUDE_hyperbola_asymptote_l175_17519


namespace NUMINAMATH_CALUDE_system_of_equations_l175_17580

theorem system_of_equations (a b : ℝ) 
  (eq1 : 2 * a - b = 12) 
  (eq2 : a + 2 * b = 8) : 
  3 * a + b = 20 := by
  sorry

end NUMINAMATH_CALUDE_system_of_equations_l175_17580


namespace NUMINAMATH_CALUDE_salt_solution_dilution_l175_17538

/-- Proves that the initial volume of a 20% salt solution is 90 liters,
    given that adding 30 liters of water dilutes it to a 15% salt solution. -/
theorem salt_solution_dilution (initial_volume : ℝ) : 
  (0.20 * initial_volume = 0.15 * (initial_volume + 30)) → 
  initial_volume = 90 := by
  sorry

end NUMINAMATH_CALUDE_salt_solution_dilution_l175_17538


namespace NUMINAMATH_CALUDE_modulus_z_l175_17573

theorem modulus_z (z : ℂ) (h : z * (1 + 2*I) = 4 + 3*I) : Complex.abs z = Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_modulus_z_l175_17573


namespace NUMINAMATH_CALUDE_min_value_2a_plus_b_l175_17521

theorem min_value_2a_plus_b (a b : ℝ) (ha : a > 0) (hb : b > 0)
  (h1 : ∃ x : ℝ, x^2 + 2*a*x + 3*b = 0)
  (h2 : ∃ x : ℝ, x^2 + 3*b*x + 2*a = 0) :
  2*a + b ≥ 2 * Real.sqrt (3 * Real.rpow (8/3) (1/3)) + Real.rpow (8/3) (1/3) :=
sorry

end NUMINAMATH_CALUDE_min_value_2a_plus_b_l175_17521


namespace NUMINAMATH_CALUDE_negation_of_forall_x_squared_gt_one_l175_17546

theorem negation_of_forall_x_squared_gt_one :
  (¬ ∀ x : ℝ, x^2 > 1) ↔ (∃ x₀ : ℝ, x₀^2 ≤ 1) := by
  sorry

end NUMINAMATH_CALUDE_negation_of_forall_x_squared_gt_one_l175_17546


namespace NUMINAMATH_CALUDE_second_largest_number_l175_17560

theorem second_largest_number (A B C D : ℕ) : 
  A = 3 * 3 →
  C = 4 * A →
  B = C - 15 →
  D = A + 19 →
  (C > D ∧ D > B ∧ B > A) :=
by sorry

end NUMINAMATH_CALUDE_second_largest_number_l175_17560


namespace NUMINAMATH_CALUDE_soccer_teams_count_l175_17536

theorem soccer_teams_count (n : ℕ) (k : ℕ) (h : n = 12 ∧ k = 6) :
  (Nat.choose n k : ℕ) = (Nat.choose n (k - 1) : ℕ) / k :=
by sorry

#check soccer_teams_count

end NUMINAMATH_CALUDE_soccer_teams_count_l175_17536


namespace NUMINAMATH_CALUDE_total_time_theorem_l175_17598

/-- The time Carlotta spends practicing for each minute of singing -/
def practice_time : ℕ := 3

/-- The time Carlotta spends throwing tantrums for each minute of singing -/
def tantrum_time : ℕ := 5

/-- The length of the final stage performance in minutes -/
def performance_length : ℕ := 6

/-- The total time spent per minute of singing -/
def total_time_per_minute : ℕ := 1 + practice_time + tantrum_time

/-- Theorem: The total combined amount of time Carlotta spends practicing, 
    throwing tantrums, and singing in the final stage performance is 54 minutes -/
theorem total_time_theorem : performance_length * total_time_per_minute = 54 := by
  sorry

end NUMINAMATH_CALUDE_total_time_theorem_l175_17598


namespace NUMINAMATH_CALUDE_estimate_boys_in_grade_l175_17528

theorem estimate_boys_in_grade (total_students : ℕ) (sample_size : ℕ) (girls_in_sample : ℕ) 
  (h1 : total_students = 1200)
  (h2 : sample_size = 20)
  (h3 : girls_in_sample = 8) :
  total_students - (girls_in_sample * total_students / sample_size) = 720 := by
  sorry

end NUMINAMATH_CALUDE_estimate_boys_in_grade_l175_17528


namespace NUMINAMATH_CALUDE_sum_of_100th_bracket_l175_17512

def sequence_start : ℕ := 3

def cycle_length : ℕ := 4

def numbers_per_cycle : ℕ := 10

def target_bracket : ℕ := 100

theorem sum_of_100th_bracket :
  let total_numbers := (target_bracket - 1) / cycle_length * numbers_per_cycle
  let last_number := sequence_start + 2 * (total_numbers - 1)
  let bracket_numbers := [last_number - 6, last_number - 4, last_number - 2, last_number]
  List.sum bracket_numbers = 1992 := by
sorry

end NUMINAMATH_CALUDE_sum_of_100th_bracket_l175_17512


namespace NUMINAMATH_CALUDE_smallest_power_congruence_l175_17579

theorem smallest_power_congruence (h : 2015 = 5 * 13 * 31) :
  (∃ n : ℕ, n > 0 ∧ 2^n ≡ 1 [ZMOD 2015]) ∧
  (∀ m : ℕ, m > 0 ∧ 2^m ≡ 1 [ZMOD 2015] → m ≥ 60) ∧
  2^60 ≡ 1 [ZMOD 2015] := by
  sorry

end NUMINAMATH_CALUDE_smallest_power_congruence_l175_17579


namespace NUMINAMATH_CALUDE_set_B_equals_l175_17578

def U : Set Nat := {1, 3, 5, 7, 9}

theorem set_B_equals (A B : Set Nat) 
  (h1 : A ⊆ U)
  (h2 : B ⊆ U)
  (h3 : A ∩ B = {1, 3})
  (h4 : (U \ A) ∩ B = {5}) :
  B = {1, 3, 5} := by
  sorry

end NUMINAMATH_CALUDE_set_B_equals_l175_17578


namespace NUMINAMATH_CALUDE_calculation_proof_l175_17513

theorem calculation_proof :
  ((-56 * (-3/8)) / (-1 - 2/5) = -15) ∧
  ((-12) / (-4) * (1/4) = 3/4) := by
sorry

end NUMINAMATH_CALUDE_calculation_proof_l175_17513


namespace NUMINAMATH_CALUDE_middle_number_is_eleven_l175_17555

theorem middle_number_is_eleven (x y z : ℕ) 
  (sum_xy : x + y = 18) 
  (sum_xz : x + z = 23) 
  (sum_yz : y + z = 27) : 
  y = 11 := by
sorry

end NUMINAMATH_CALUDE_middle_number_is_eleven_l175_17555


namespace NUMINAMATH_CALUDE_f_min_value_f_min_at_9_2_l175_17584

/-- The function f(x, y) defined in the problem -/
def f (x y : ℝ) : ℝ := x^2 + 6*y^2 - 2*x*y - 14*x - 6*y + 72

/-- Theorem stating that f(x, y) has a minimum value of 3 -/
theorem f_min_value (x y : ℝ) : f x y ≥ 3 := by sorry

/-- Theorem stating that f(9, 2) achieves the minimum value -/
theorem f_min_at_9_2 : f 9 2 = 3 := by sorry

end NUMINAMATH_CALUDE_f_min_value_f_min_at_9_2_l175_17584


namespace NUMINAMATH_CALUDE_hiking_trip_days_l175_17589

/-- Represents the hiking trip scenario -/
structure HikingTrip where
  rateUp : ℝ
  rateDown : ℝ
  distanceDown : ℝ
  days : ℝ

/-- The hiking trip satisfies the given conditions -/
def validHikingTrip (trip : HikingTrip) : Prop :=
  trip.rateUp = 6 ∧
  trip.rateDown = 1.5 * trip.rateUp ∧
  trip.distanceDown = 18 ∧
  trip.rateUp * trip.days = trip.rateDown * trip.days

/-- The number of days for the hiking trip is 2 -/
theorem hiking_trip_days (trip : HikingTrip) (h : validHikingTrip trip) : trip.days = 2 := by
  sorry


end NUMINAMATH_CALUDE_hiking_trip_days_l175_17589


namespace NUMINAMATH_CALUDE_circle_center_from_axis_intersections_l175_17503

/-- Given a circle that intersects the x-axis at (a, 0) and (b, 0),
    and the y-axis at (0, c) and (0, d), its center is at ((a+b)/2, (c+d)/2) -/
theorem circle_center_from_axis_intersections 
  (a b c d : ℝ) : 
  ∃ (center : ℝ × ℝ),
    (∃ (circle : Set (ℝ × ℝ)), 
      (a, 0) ∈ circle ∧ 
      (b, 0) ∈ circle ∧ 
      (0, c) ∈ circle ∧ 
      (0, d) ∈ circle ∧
      center = ((a + b) / 2, (c + d) / 2) ∧
      ∀ p ∈ circle, (p.1 - center.1)^2 + (p.2 - center.2)^2 = 
        (a - center.1)^2 + (0 - center.2)^2) :=
by
  sorry

end NUMINAMATH_CALUDE_circle_center_from_axis_intersections_l175_17503


namespace NUMINAMATH_CALUDE_james_total_earnings_l175_17570

def january_earnings : ℕ := 4000

def february_earnings (jan : ℕ) : ℕ := 2 * jan

def march_earnings (feb : ℕ) : ℕ := feb - 2000

def total_earnings (jan feb mar : ℕ) : ℕ := jan + feb + mar

theorem james_total_earnings :
  total_earnings january_earnings (february_earnings january_earnings) (march_earnings (february_earnings january_earnings)) = 18000 := by
  sorry

end NUMINAMATH_CALUDE_james_total_earnings_l175_17570


namespace NUMINAMATH_CALUDE_direction_vector_y_component_l175_17550

/-- Given a line passing through two points, prove that if its direction vector
    has a specific form, then the y-component of the direction vector is 4.5. -/
theorem direction_vector_y_component
  (p1 : ℝ × ℝ)
  (p2 : ℝ × ℝ)
  (h1 : p1 = (1, -1))
  (h2 : p2 = (5, 5))
  (direction_vector : ℝ × ℝ)
  (h3 : direction_vector.1 = 3)
  (h4 : ∃ (t : ℝ), t • (p2 - p1) = direction_vector) :
  direction_vector.2 = 4.5 := by
sorry

end NUMINAMATH_CALUDE_direction_vector_y_component_l175_17550


namespace NUMINAMATH_CALUDE_unique_prime_perfect_power_l175_17532

theorem unique_prime_perfect_power : 
  ∃! p : ℕ, p.Prime ∧ p ≤ 1000 ∧ ∃ m n : ℕ, n ≥ 2 ∧ 2 * p + 1 = m^n ∧ p = 13 := by
  sorry

end NUMINAMATH_CALUDE_unique_prime_perfect_power_l175_17532


namespace NUMINAMATH_CALUDE_bad_carrots_count_l175_17590

theorem bad_carrots_count (olivia_carrots : ℕ) (mom_carrots : ℕ) (good_carrots : ℕ) : 
  olivia_carrots = 20 → 
  mom_carrots = 14 → 
  good_carrots = 19 → 
  olivia_carrots + mom_carrots - good_carrots = 15 := by
sorry

end NUMINAMATH_CALUDE_bad_carrots_count_l175_17590


namespace NUMINAMATH_CALUDE_find_c_l175_17542

theorem find_c (a b c : ℝ) (x : ℝ) 
  (eq : (x + a) * (x + b) = x^2 + c*x + 12)
  (h1 : b = 4)
  (h2 : a + b = 6) : 
  c = 6 := by
sorry

end NUMINAMATH_CALUDE_find_c_l175_17542


namespace NUMINAMATH_CALUDE_triangle_proof_l175_17557

/-- Triangle ABC with sides a, b, c opposite to angles A, B, C respectively --/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- The given conditions for the triangle --/
def TriangleConditions (t : Triangle) : Prop :=
  (2 * t.a - t.c) * Real.cos t.B = t.b * Real.cos t.C ∧
  t.b = Real.sqrt 7 ∧
  t.a + t.c = 4

theorem triangle_proof (t : Triangle) (h : TriangleConditions t) :
  t.B = π / 4 ∧ 
  (1 / 2) * t.a * t.c * Real.sin t.B = (3 * Real.sqrt 2) / 4 :=
by sorry

end NUMINAMATH_CALUDE_triangle_proof_l175_17557


namespace NUMINAMATH_CALUDE_cube_coloring_theorem_l175_17544

/-- Represents a point in the cube --/
inductive CubePoint
| Center
| FaceCenter
| Vertex
| EdgeCenter

/-- Represents a color --/
inductive Color
| Blue
| Red

/-- Represents a straight line in the cube --/
structure Line where
  points : List CubePoint
  aligned : points.length = 3

/-- A coloring of the cube points --/
def Coloring := CubePoint → Color

/-- The set of all points in the cube --/
def cubePoints : List CubePoint := 
  [CubePoint.Center] ++ 
  List.replicate 6 CubePoint.FaceCenter ++
  List.replicate 8 CubePoint.Vertex ++
  List.replicate 12 CubePoint.EdgeCenter

/-- Theorem: For any coloring of the cube points, there exists a line with three points of the same color --/
theorem cube_coloring_theorem :
  ∀ (coloring : Coloring),
  ∃ (line : Line),
  ∀ (p : CubePoint),
  p ∈ line.points → coloring p = coloring (line.points.get ⟨0, by sorry⟩) :=
by sorry

end NUMINAMATH_CALUDE_cube_coloring_theorem_l175_17544


namespace NUMINAMATH_CALUDE_forty_percent_changed_ratings_l175_17534

/-- Represents the survey results for parents' ratings of online class experience -/
structure SurveyResults where
  total_parents : ℕ
  upgrade_percent : ℚ
  maintain_percent : ℚ
  downgrade_percent : ℚ

/-- Calculates the percentage of parents who changed their ratings -/
def changed_ratings_percentage (results : SurveyResults) : ℚ :=
  (results.upgrade_percent + results.downgrade_percent) * 100

/-- Theorem stating that given the survey conditions, 40% of parents changed their ratings -/
theorem forty_percent_changed_ratings (results : SurveyResults) 
  (h1 : results.total_parents = 120)
  (h2 : results.upgrade_percent = 30 / 100)
  (h3 : results.maintain_percent = 60 / 100)
  (h4 : results.downgrade_percent = 10 / 100)
  (h5 : results.upgrade_percent + results.maintain_percent + results.downgrade_percent = 1) :
  changed_ratings_percentage results = 40 := by
  sorry

end NUMINAMATH_CALUDE_forty_percent_changed_ratings_l175_17534


namespace NUMINAMATH_CALUDE_largest_prime_divisor_factorial_sum_l175_17514

theorem largest_prime_divisor_factorial_sum : ∃ p : ℕ, 
  Nat.Prime p ∧ 
  p ∣ (Nat.factorial 13 + Nat.factorial 14) ∧
  ∀ q : ℕ, Nat.Prime q → q ∣ (Nat.factorial 13 + Nat.factorial 14) → q ≤ p :=
by sorry

end NUMINAMATH_CALUDE_largest_prime_divisor_factorial_sum_l175_17514


namespace NUMINAMATH_CALUDE_particular_propositions_count_l175_17585

-- Define a proposition type
inductive Proposition
| ExistsDivisorImpossible
| PrismIsPolyhedron
| AllEquationsHaveRealSolutions
| SomeTrianglesAreAcute

-- Define a function to check if a proposition is particular
def isParticular (p : Proposition) : Bool :=
  match p with
  | Proposition.ExistsDivisorImpossible => true
  | Proposition.PrismIsPolyhedron => false
  | Proposition.AllEquationsHaveRealSolutions => false
  | Proposition.SomeTrianglesAreAcute => true

-- Define the list of all propositions
def allPropositions : List Proposition :=
  [Proposition.ExistsDivisorImpossible, Proposition.PrismIsPolyhedron,
   Proposition.AllEquationsHaveRealSolutions, Proposition.SomeTrianglesAreAcute]

-- Theorem: The number of particular propositions is 2
theorem particular_propositions_count :
  (allPropositions.filter isParticular).length = 2 := by
  sorry

end NUMINAMATH_CALUDE_particular_propositions_count_l175_17585


namespace NUMINAMATH_CALUDE_rectangular_field_area_l175_17545

/-- Proves that a rectangular field with width one-third of length and perimeter 72 meters has an area of 243 square meters. -/
theorem rectangular_field_area (width length : ℝ) (h1 : width = length / 3) (h2 : 2 * (width + length) = 72) :
  width * length = 243 := by
  sorry

end NUMINAMATH_CALUDE_rectangular_field_area_l175_17545


namespace NUMINAMATH_CALUDE_money_sharing_l175_17517

theorem money_sharing (ken_share tony_share total : ℕ) : 
  ken_share = 1750 →
  tony_share = 2 * ken_share →
  total = ken_share + tony_share →
  total = 5250 := by
sorry

end NUMINAMATH_CALUDE_money_sharing_l175_17517


namespace NUMINAMATH_CALUDE_f_neg_l175_17516

-- Define an odd function f on the real numbers
def f : ℝ → ℝ := sorry

-- Define the property of f being odd
axiom f_odd : ∀ x : ℝ, f (-x) = -f x

-- Define f for positive x
axiom f_pos : ∀ x : ℝ, x > 0 → f x = x^2 + 1

-- Theorem to prove
theorem f_neg : ∀ x : ℝ, x < 0 → f x = -x^2 - 1 := by sorry

end NUMINAMATH_CALUDE_f_neg_l175_17516


namespace NUMINAMATH_CALUDE_perpendicular_lines_from_perpendicular_planes_l175_17547

-- Define the basic types
variable (Point : Type) (Line : Type) (Plane : Type)

-- Define the relations
variable (distinct : Line → Line → Prop)
variable (distinct_plane : Plane → Plane → Prop)
variable (perpendicular_line_plane : Line → Plane → Prop)
variable (perpendicular_plane : Plane → Plane → Prop)
variable (perpendicular_line : Line → Line → Prop)

-- Theorem statement
theorem perpendicular_lines_from_perpendicular_planes
  (m n : Line) (α β : Plane)
  (h1 : distinct m n)
  (h2 : distinct_plane α β)
  (h3 : perpendicular_plane α β)
  (h4 : perpendicular_line_plane m α)
  (h5 : perpendicular_line_plane n β) :
  perpendicular_line m n :=
sorry

end NUMINAMATH_CALUDE_perpendicular_lines_from_perpendicular_planes_l175_17547


namespace NUMINAMATH_CALUDE_odd_sum_squared_plus_product_not_both_even_l175_17587

theorem odd_sum_squared_plus_product_not_both_even (p q : ℤ) 
  (h : Odd (p^2 + q^2 + p*q)) : ¬(Even p ∧ Even q) := by
  sorry

end NUMINAMATH_CALUDE_odd_sum_squared_plus_product_not_both_even_l175_17587


namespace NUMINAMATH_CALUDE_basketball_lineup_count_l175_17556

/-- The number of ways to choose a lineup from a basketball team --/
def choose_lineup (team_size : ℕ) (lineup_size : ℕ) : ℕ :=
  (team_size - lineup_size + 1).factorial / (team_size - lineup_size).factorial

/-- Theorem: The number of ways to choose a lineup of 6 players from a team of 15 is 3,603,600 --/
theorem basketball_lineup_count :
  choose_lineup 15 6 = 3603600 := by
  sorry

end NUMINAMATH_CALUDE_basketball_lineup_count_l175_17556


namespace NUMINAMATH_CALUDE_average_problem_l175_17537

theorem average_problem (y : ℝ) : (15 + 30 + 45 + y) / 4 = 35 → y = 50 := by
  sorry

end NUMINAMATH_CALUDE_average_problem_l175_17537
