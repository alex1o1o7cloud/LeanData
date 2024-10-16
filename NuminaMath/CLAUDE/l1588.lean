import Mathlib

namespace NUMINAMATH_CALUDE_export_volume_equation_l1588_158855

def export_volume_2023 : ℝ := 107
def export_volume_2013 : ℝ → ℝ := λ x => x

theorem export_volume_equation (x : ℝ) : 
  export_volume_2023 = 4 * (export_volume_2013 x) + 3 ↔ 4 * x + 3 = 107 :=
by sorry

end NUMINAMATH_CALUDE_export_volume_equation_l1588_158855


namespace NUMINAMATH_CALUDE_youngest_child_age_l1588_158864

/-- A function that checks if a number is prime -/
def isPrime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ d : ℕ, d > 1 → d < n → ¬(n % d = 0)

/-- The set of ages of the six children -/
def childrenAges (x : ℕ) : Finset ℕ :=
  {x, x + 2, x + 6, x + 8, x + 12, x + 14}

/-- Theorem stating that the youngest child's age is 5 -/
theorem youngest_child_age :
  ∃ (x : ℕ), x = 5 ∧ 
    (∀ y ∈ childrenAges x, isPrime y) ∧
    (childrenAges x).card = 6 :=
  sorry

end NUMINAMATH_CALUDE_youngest_child_age_l1588_158864


namespace NUMINAMATH_CALUDE_average_problem_l1588_158893

theorem average_problem (c d e : ℝ) : 
  (4 + 6 + 9 + c + d + e) / 6 = 20 → (c + d + e) / 3 = 101 / 3 := by
  sorry

end NUMINAMATH_CALUDE_average_problem_l1588_158893


namespace NUMINAMATH_CALUDE_prime_power_sum_l1588_158853

theorem prime_power_sum (w x y z : ℕ) : 
  2^w * 3^x * 5^y * 7^z = 13230 → 3*w + 2*x + 6*y + 4*z = 23 := by
  sorry

end NUMINAMATH_CALUDE_prime_power_sum_l1588_158853


namespace NUMINAMATH_CALUDE_quadratic_equation_roots_l1588_158810

theorem quadratic_equation_roots (k : ℝ) (x₁ x₂ : ℝ) : 
  (∀ x, x^2 - 3*x + k = 0 ↔ x = x₁ ∨ x = x₂) →
  (x₁ * x₂ + 2*x₁ + 2*x₂ = 1) →
  k = -5 := by sorry

end NUMINAMATH_CALUDE_quadratic_equation_roots_l1588_158810


namespace NUMINAMATH_CALUDE_binomial_coefficient_equality_l1588_158880

theorem binomial_coefficient_equality (a : ℝ) (ha : a ≠ 0) :
  (Nat.choose 5 4 : ℝ) * a^4 = (Nat.choose 5 3 : ℝ) * a^3 → a = 2 := by
  sorry

end NUMINAMATH_CALUDE_binomial_coefficient_equality_l1588_158880


namespace NUMINAMATH_CALUDE_unique_good_number_adjacent_to_power_of_two_l1588_158859

theorem unique_good_number_adjacent_to_power_of_two :
  ∃! n : ℕ, n > 0 ∧
  (∃ a b : ℕ, a ≥ 2 ∧ b ≥ 2 ∧ n = a^b) ∧
  (∃ t : ℕ, t > 0 ∧ (n = 2^t + 1 ∨ n = 2^t - 1)) ∧
  n = 9 := by
sorry

end NUMINAMATH_CALUDE_unique_good_number_adjacent_to_power_of_two_l1588_158859


namespace NUMINAMATH_CALUDE_square_root_of_four_l1588_158895

theorem square_root_of_four :
  {x : ℝ | x ^ 2 = 4} = {2, -2} := by sorry

end NUMINAMATH_CALUDE_square_root_of_four_l1588_158895


namespace NUMINAMATH_CALUDE_polynomial_division_remainder_l1588_158865

theorem polynomial_division_remainder (x : ℤ) : 
  x^1010 % ((x^2 - 1) * (x + 1)) = 1 := by sorry

end NUMINAMATH_CALUDE_polynomial_division_remainder_l1588_158865


namespace NUMINAMATH_CALUDE_fourth_student_in_sample_l1588_158861

/-- Represents a systematic sample from a class of students. -/
structure SystematicSample where
  class_size : ℕ
  sample_size : ℕ
  interval : ℕ
  first_student : ℕ

/-- Checks if a student number is part of the systematic sample. -/
def is_in_sample (s : SystematicSample) (student : ℕ) : Prop :=
  ∃ k : ℕ, student = s.first_student + k * s.interval

/-- The main theorem to be proved. -/
theorem fourth_student_in_sample
  (s : SystematicSample)
  (h_class_size : s.class_size = 48)
  (h_sample_size : s.sample_size = 4)
  (h_interval : s.interval = s.class_size / s.sample_size)
  (h_6_in_sample : is_in_sample s 6)
  (h_30_in_sample : is_in_sample s 30)
  (h_42_in_sample : is_in_sample s 42)
  : is_in_sample s 18 :=
sorry

end NUMINAMATH_CALUDE_fourth_student_in_sample_l1588_158861


namespace NUMINAMATH_CALUDE_parabola_properties_l1588_158891

/-- Given a parabola y = x^2 - 8x + 12, prove its properties -/
theorem parabola_properties :
  let f (x : ℝ) := x^2 - 8*x + 12
  ∃ (axis vertex_x vertex_y x1 x2 : ℝ),
    -- The axis of symmetry
    axis = 4 ∧
    -- The vertex coordinates
    f vertex_x = vertex_y ∧
    vertex_x = 4 ∧
    vertex_y = -4 ∧
    -- The x-axis intersection points
    f x1 = 0 ∧
    f x2 = 0 ∧
    x1 = 2 ∧
    x2 = 6 := by
  sorry

end NUMINAMATH_CALUDE_parabola_properties_l1588_158891


namespace NUMINAMATH_CALUDE_circle_center_l1588_158832

/-- Given a circle with equation x^2 + y^2 - 2x + 4y + 1 = 0, its center is (1, -2) -/
theorem circle_center (x y : ℝ) : 
  (x^2 + y^2 - 2*x + 4*y + 1 = 0) → (∃ r : ℝ, (x - 1)^2 + (y + 2)^2 = r^2) :=
by sorry

end NUMINAMATH_CALUDE_circle_center_l1588_158832


namespace NUMINAMATH_CALUDE_unique_solution_sum_l1588_158811

-- Define the equation
def satisfies_equation (x y : ℕ+) : Prop :=
  (x : ℝ)^2 + 84 * (x : ℝ) + 2008 = (y : ℝ)^2

-- State the theorem
theorem unique_solution_sum :
  ∃! (x y : ℕ+), satisfies_equation x y ∧ x + y = 80 := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_sum_l1588_158811


namespace NUMINAMATH_CALUDE_money_distribution_l1588_158833

theorem money_distribution (x : ℝ) (x_pos : x > 0) : 
  let total_money := 6*x + 5*x + 4*x + 3*x
  let ott_money := x + x + x + x
  ott_money / total_money = 2 / 9 := by
sorry


end NUMINAMATH_CALUDE_money_distribution_l1588_158833


namespace NUMINAMATH_CALUDE_least_four_digit_13_heavy_l1588_158868

theorem least_four_digit_13_heavy : ∀ n : ℕ,
  1000 ≤ n ∧ n < 10000 ∧ n % 13 > 8 → n ≥ 1004 :=
by sorry

end NUMINAMATH_CALUDE_least_four_digit_13_heavy_l1588_158868


namespace NUMINAMATH_CALUDE_solution_set_part1_range_of_a_part2_l1588_158836

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := |x + a| + |x - 1|

-- Part 1
theorem solution_set_part1 :
  {x : ℝ | f 3 x ≥ x + 9} = {x : ℝ | x < -11/3 ∨ x > 7} := by sorry

-- Part 2
theorem range_of_a_part2 :
  ∀ a : ℝ, (∀ x ∈ Set.Icc 0 1, f a x ≤ |x - 4|) → a ∈ Set.Icc (-3) 2 := by sorry

end NUMINAMATH_CALUDE_solution_set_part1_range_of_a_part2_l1588_158836


namespace NUMINAMATH_CALUDE_mixture_volume_proportion_l1588_158800

/-- Given two solutions P and Q, where P is 80% carbonated water and Q is 55% carbonated water,
    if a mixture of P and Q contains 67.5% carbonated water, then the volume of P in the mixture
    is 50% of the total volume. -/
theorem mixture_volume_proportion (x y : ℝ) (hx : x ≥ 0) (hy : y ≥ 0) :
  0.80 * x + 0.55 * y = 0.675 * (x + y) →
  x / (x + y) = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_mixture_volume_proportion_l1588_158800


namespace NUMINAMATH_CALUDE_quadratic_equation_range_l1588_158816

/-- The range of k for which the quadratic equation (k-1)x^2 - 2x + 1 = 0 has two distinct real roots -/
theorem quadratic_equation_range (k : ℝ) : 
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ 
   (k - 1) * x₁^2 - 2 * x₁ + 1 = 0 ∧ 
   (k - 1) * x₂^2 - 2 * x₂ + 1 = 0) ↔ 
  (k < 2 ∧ k ≠ 1) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equation_range_l1588_158816


namespace NUMINAMATH_CALUDE_constant_term_expansion_l1588_158813

/-- The constant term in the expansion of (x + 2/x)^6 -/
def constant_term : ℕ := 160

/-- The binomial coefficient (n choose k) -/
def binomial (n k : ℕ) : ℕ := sorry

theorem constant_term_expansion :
  constant_term = binomial 6 3 * 2^3 := by sorry

end NUMINAMATH_CALUDE_constant_term_expansion_l1588_158813


namespace NUMINAMATH_CALUDE_unique_a_for_cubic_property_l1588_158888

theorem unique_a_for_cubic_property (a : ℕ+) :
  (∀ n : ℕ+, ∃ k : ℤ, 4 * (a.val ^ n.val + 1) = k ^ 3) →
  a = 1 :=
by sorry

end NUMINAMATH_CALUDE_unique_a_for_cubic_property_l1588_158888


namespace NUMINAMATH_CALUDE_arithmetic_sequence_n_eq_16_l1588_158873

/-- An arithmetic sequence with specific properties -/
structure ArithmeticSequence where
  a : ℕ → ℝ
  is_arithmetic : ∀ n : ℕ, a (n + 2) - a (n + 1) = a (n + 1) - a n
  a_4_eq_7 : a 4 = 7
  a_3_plus_a_6_eq_16 : a 3 + a 6 = 16
  exists_n : ∃ n : ℕ, a n = 31

/-- The theorem stating that n = 16 for the given arithmetic sequence -/
theorem arithmetic_sequence_n_eq_16 (seq : ArithmeticSequence) :
  ∃ n : ℕ, seq.a n = 31 ∧ n = 16 := by
  sorry


end NUMINAMATH_CALUDE_arithmetic_sequence_n_eq_16_l1588_158873


namespace NUMINAMATH_CALUDE_day_one_sales_is_86_l1588_158802

/-- The number of cups sold on day one -/
def day_one_sales : ℕ := sorry

/-- The number of cups sold on each of the next 11 days -/
def daily_sales : ℕ := 50

/-- The total number of days -/
def total_days : ℕ := 12

/-- The average daily sales over the 12-day period -/
def average_sales : ℕ := 53

/-- Theorem: The number of cups sold on day one is 86 -/
theorem day_one_sales_is_86 :
  day_one_sales = 86 :=
by
  sorry

end NUMINAMATH_CALUDE_day_one_sales_is_86_l1588_158802


namespace NUMINAMATH_CALUDE_prob_three_red_prob_same_color_prob_not_same_color_l1588_158839

-- Define the probability of drawing a red ball
def prob_red : ℚ := 1 / 2

-- Define the probability of drawing a yellow ball
def prob_yellow : ℚ := 1 - prob_red

-- Define the number of draws
def num_draws : ℕ := 3

-- Theorem for the probability of drawing three red balls
theorem prob_three_red :
  prob_red ^ num_draws = 1 / 8 := by sorry

-- Theorem for the probability of drawing three balls of the same color
theorem prob_same_color :
  prob_red ^ num_draws + prob_yellow ^ num_draws = 1 / 4 := by sorry

-- Theorem for the probability of not drawing all three balls of the same color
theorem prob_not_same_color :
  1 - (prob_red ^ num_draws + prob_yellow ^ num_draws) = 3 / 4 := by sorry

end NUMINAMATH_CALUDE_prob_three_red_prob_same_color_prob_not_same_color_l1588_158839


namespace NUMINAMATH_CALUDE_inscribed_cube_volume_l1588_158801

/-- The volume of a cube inscribed in a sphere, which is itself inscribed in a larger cube -/
theorem inscribed_cube_volume (outer_cube_edge : ℝ) (h : outer_cube_edge = 12) :
  let sphere_diameter := outer_cube_edge
  let inner_cube_diagonal := sphere_diameter
  let inner_cube_edge := inner_cube_diagonal / Real.sqrt 3
  let inner_cube_volume := inner_cube_edge ^ 3
  inner_cube_volume = 192 * Real.sqrt 3 := by
  sorry

#check inscribed_cube_volume

end NUMINAMATH_CALUDE_inscribed_cube_volume_l1588_158801


namespace NUMINAMATH_CALUDE_carolyns_project_time_l1588_158849

/-- Represents the embroidering project with given parameters -/
structure EmbroideringProject where
  stitches_per_minute : ℕ
  flower_stitches : ℕ
  unicorn_stitches : ℕ
  godzilla_stitches : ℕ
  num_flowers : ℕ
  num_unicorns : ℕ
  num_godzillas : ℕ
  embroidering_time_before_break : ℕ
  break_duration : ℕ

/-- Calculates the total time needed for the embroidering project -/
def total_time (project : EmbroideringProject) : ℕ :=
  sorry

/-- Theorem stating that the total time for Carolyn's project is 1265 minutes -/
theorem carolyns_project_time :
  let project : EmbroideringProject := {
    stitches_per_minute := 4,
    flower_stitches := 60,
    unicorn_stitches := 180,
    godzilla_stitches := 800,
    num_flowers := 50,
    num_unicorns := 3,
    num_godzillas := 1,
    embroidering_time_before_break := 30,
    break_duration := 5
  }
  total_time project = 1265 := by sorry

end NUMINAMATH_CALUDE_carolyns_project_time_l1588_158849


namespace NUMINAMATH_CALUDE_total_complaints_over_five_days_l1588_158817

/-- Represents the different staff shortage scenarios -/
inductive StaffShortage
  | Normal
  | TwentyPercent
  | FortyPercent

/-- Represents the different self-checkout states -/
inductive SelfCheckout
  | Working
  | PartiallyBroken
  | CompletelyBroken

/-- Represents the different weather conditions -/
inductive Weather
  | Clear
  | Rainy
  | Snowstorm

/-- Represents the different special events -/
inductive SpecialEvent
  | Normal
  | Holiday
  | OngoingSale

/-- Represents the conditions for a single day -/
structure DayConditions where
  staffShortage : StaffShortage
  selfCheckout : SelfCheckout
  weather : Weather
  specialEvent : SpecialEvent

/-- Calculates the number of complaints for a given day based on its conditions -/
def calculateComplaints (baseComplaints : ℕ) (conditions : DayConditions) : ℕ :=
  sorry

/-- The base number of complaints per day -/
def baseComplaints : ℕ := 120

/-- The conditions for each of the five days -/
def dayConditions : List DayConditions := [
  { staffShortage := StaffShortage.TwentyPercent, selfCheckout := SelfCheckout.CompletelyBroken, weather := Weather.Rainy, specialEvent := SpecialEvent.OngoingSale },
  { staffShortage := StaffShortage.FortyPercent, selfCheckout := SelfCheckout.PartiallyBroken, weather := Weather.Clear, specialEvent := SpecialEvent.Holiday },
  { staffShortage := StaffShortage.FortyPercent, selfCheckout := SelfCheckout.CompletelyBroken, weather := Weather.Snowstorm, specialEvent := SpecialEvent.Normal },
  { staffShortage := StaffShortage.Normal, selfCheckout := SelfCheckout.Working, weather := Weather.Rainy, specialEvent := SpecialEvent.OngoingSale },
  { staffShortage := StaffShortage.TwentyPercent, selfCheckout := SelfCheckout.CompletelyBroken, weather := Weather.Clear, specialEvent := SpecialEvent.Holiday }
]

/-- Theorem stating that the total number of complaints over the five days is 1038 -/
theorem total_complaints_over_five_days :
  (dayConditions.map (calculateComplaints baseComplaints)).sum = 1038 := by
  sorry

end NUMINAMATH_CALUDE_total_complaints_over_five_days_l1588_158817


namespace NUMINAMATH_CALUDE_class_7_highest_prob_l1588_158842

/-- The number of classes -/
def num_classes : ℕ := 12

/-- The probability of getting a sum of n when throwing two dice -/
def prob_sum (n : ℕ) : ℚ :=
  match n with
  | 2 => 1 / 36
  | 3 => 1 / 18
  | 4 => 1 / 12
  | 5 => 1 / 9
  | 6 => 5 / 36
  | 7 => 1 / 6
  | 8 => 5 / 36
  | 9 => 1 / 9
  | 10 => 1 / 12
  | 11 => 1 / 18
  | 12 => 1 / 36
  | _ => 0

/-- Theorem: Class 7 has the highest probability of being selected -/
theorem class_7_highest_prob :
  ∀ n : ℕ, 2 ≤ n → n ≤ num_classes → prob_sum n ≤ prob_sum 7 :=
by sorry

end NUMINAMATH_CALUDE_class_7_highest_prob_l1588_158842


namespace NUMINAMATH_CALUDE_square_side_length_l1588_158819

theorem square_side_length (s : ℝ) : s^2 + s - 4*s = 4 → s = 4 := by
  sorry

end NUMINAMATH_CALUDE_square_side_length_l1588_158819


namespace NUMINAMATH_CALUDE_large_circle_radius_l1588_158804

/-- Given two circles A and B with radii 3 and 2 respectively, internally tangent to a larger circle
    at different points, and the distance between the centers of circles A and B is 6,
    the radius of the large circle is (5 + √33) / 2. -/
theorem large_circle_radius (r : ℝ) : r > 0 →
  (r - 3) ^ 2 + (r - 2) ^ 2 + 2 * (r - 3) * (r - 2) = 36 →
  r = (5 + Real.sqrt 33) / 2 := by
  sorry

end NUMINAMATH_CALUDE_large_circle_radius_l1588_158804


namespace NUMINAMATH_CALUDE_send_more_money_solution_l1588_158862

def is_valid_assignment (S E N D M O R Y : Nat) : Prop :=
  S ≠ 0 ∧ M ≠ 0 ∧
  S < 10 ∧ E < 10 ∧ N < 10 ∧ D < 10 ∧ M < 10 ∧ O < 10 ∧ R < 10 ∧ Y < 10 ∧
  S ≠ E ∧ S ≠ N ∧ S ≠ D ∧ S ≠ M ∧ S ≠ O ∧ S ≠ R ∧ S ≠ Y ∧
  E ≠ N ∧ E ≠ D ∧ E ≠ M ∧ E ≠ O ∧ E ≠ R ∧ E ≠ Y ∧
  N ≠ D ∧ N ≠ M ∧ N ≠ O ∧ N ≠ R ∧ N ≠ Y ∧
  D ≠ M ∧ D ≠ O ∧ D ≠ R ∧ D ≠ Y ∧
  M ≠ O ∧ M ≠ R ∧ M ≠ Y ∧
  O ≠ R ∧ O ≠ Y ∧
  R ≠ Y

theorem send_more_money_solution :
  ∃ (S E N D M O R Y : Nat),
    is_valid_assignment S E N D M O R Y ∧
    1000 * S + 100 * E + 10 * N + D + 1000 * M + 100 * O + 10 * R + E =
    10000 * M + 1000 * O + 100 * N + 10 * E + Y :=
by sorry

end NUMINAMATH_CALUDE_send_more_money_solution_l1588_158862


namespace NUMINAMATH_CALUDE_cosine_range_in_geometric_progression_triangle_l1588_158831

theorem cosine_range_in_geometric_progression_triangle (a b c : ℝ) (hpos : a > 0 ∧ b > 0 ∧ c > 0)
  (hacute : 0 < a ^ 2 + b ^ 2 - c ^ 2 ∧ 0 < b ^ 2 + c ^ 2 - a ^ 2 ∧ 0 < c ^ 2 + a ^ 2 - b ^ 2)
  (hgeo : b ^ 2 = a * c) : 1 / 2 ≤ (a ^ 2 + c ^ 2 - b ^ 2) / (2 * a * c) ∧ (a ^ 2 + c ^ 2 - b ^ 2) / (2 * a * c) < 1 := by
  sorry

end NUMINAMATH_CALUDE_cosine_range_in_geometric_progression_triangle_l1588_158831


namespace NUMINAMATH_CALUDE_population_after_three_years_l1588_158866

def population_growth (initial : ℕ) (rate : ℚ) (additional : ℕ) : ℕ :=
  ⌊(initial : ℚ) * (1 + rate) + additional⌋.toNat

def three_year_population (initial : ℕ) (rate1 rate2 rate3 : ℚ) (add1 add2 add3 : ℕ) : ℕ :=
  let year1 := population_growth initial rate1 add1
  let year2 := population_growth year1 rate2 add2
  population_growth year2 rate3 add3

theorem population_after_three_years :
  three_year_population 14000 (12/100) (8/100) (6/100) 150 100 500 = 18728 :=
by sorry

end NUMINAMATH_CALUDE_population_after_three_years_l1588_158866


namespace NUMINAMATH_CALUDE_computer_sticker_price_l1588_158826

theorem computer_sticker_price : 
  ∀ (sticker_price : ℝ),
  (sticker_price * 0.85 - 90 = sticker_price * 0.75 - 15) →
  sticker_price = 750 := by
sorry

end NUMINAMATH_CALUDE_computer_sticker_price_l1588_158826


namespace NUMINAMATH_CALUDE_debate_participants_l1588_158878

theorem debate_participants (third_school : ℕ) 
  (h1 : third_school + (third_school + 40) + 2 * (third_school + 40) = 920) : 
  third_school = 200 := by
sorry

end NUMINAMATH_CALUDE_debate_participants_l1588_158878


namespace NUMINAMATH_CALUDE_equilateral_triangle_segment_length_l1588_158887

-- Define the triangle and points
def Triangle (A B C : EuclideanSpace ℝ (Fin 2)) : Prop :=
  (dist A B = dist B C) ∧ (dist B C = dist C A)

def EquilateralTriangle (A B C : EuclideanSpace ℝ (Fin 2)) : Prop :=
  Triangle A B C ∧ dist A B = dist B C

def OnSegment (X P Q : EuclideanSpace ℝ (Fin 2)) : Prop :=
  ∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ X = (1 - t) • P + t • Q

-- State the theorem
theorem equilateral_triangle_segment_length 
  (A B C K L M : EuclideanSpace ℝ (Fin 2)) :
  EquilateralTriangle A B C →
  OnSegment K A B →
  OnSegment L B C →
  OnSegment M B C →
  OnSegment L B M →
  dist K L = dist K M →
  dist B L = 2 →
  dist A K = 3 →
  dist C M = 5 := by
sorry

end NUMINAMATH_CALUDE_equilateral_triangle_segment_length_l1588_158887


namespace NUMINAMATH_CALUDE_valid_sequence_probability_l1588_158830

/-- Recursive function to calculate the number of valid sequences of length n -/
def b : ℕ → ℕ
| 0 => 0
| 1 => 0
| 2 => 0
| 3 => 0
| 4 => 1
| 5 => 1
| 6 => 2
| 7 => 3
| n + 8 => b (n + 5) + b (n + 4)

/-- The probability of generating a valid sequence of length 12 -/
def prob : ℚ := 5 / 1024

theorem valid_sequence_probability :
  b 12 = 5 ∧ 2^10 = 1024 ∧ prob = (b 12 : ℚ) / 2^10 := by sorry

end NUMINAMATH_CALUDE_valid_sequence_probability_l1588_158830


namespace NUMINAMATH_CALUDE_c_profit_share_l1588_158803

theorem c_profit_share (total_investment : ℕ) (total_profit : ℕ) (c_investment : ℕ) : 
  total_investment = 180000 → 
  total_profit = 60000 → 
  c_investment = 72000 → 
  (c_investment : ℚ) / total_investment * total_profit = 24000 := by
  sorry

end NUMINAMATH_CALUDE_c_profit_share_l1588_158803


namespace NUMINAMATH_CALUDE_wife_account_percentage_l1588_158824

def income : ℝ := 800000

def children_percentage : ℝ := 0.2
def num_children : ℕ := 3
def orphan_donation_percentage : ℝ := 0.05
def final_amount : ℝ := 40000

theorem wife_account_percentage :
  let children_total := children_percentage * num_children * income
  let after_children := income - children_total
  let orphan_donation := orphan_donation_percentage * after_children
  let after_donation := after_children - orphan_donation
  let wife_deposit := after_donation - final_amount
  (wife_deposit / income) * 100 = 33 := by sorry

end NUMINAMATH_CALUDE_wife_account_percentage_l1588_158824


namespace NUMINAMATH_CALUDE_integral_cos_sin_l1588_158828

theorem integral_cos_sin : ∫ x in (0)..(π/2), (1 + Real.cos x) / (1 + Real.sin x + Real.cos x) = Real.log 2 + π/2 := by
  sorry

end NUMINAMATH_CALUDE_integral_cos_sin_l1588_158828


namespace NUMINAMATH_CALUDE_profit_reached_l1588_158808

/-- The number of disks in a buying pack -/
def buying_pack : ℕ := 5

/-- The cost of a buying pack in dollars -/
def buying_cost : ℚ := 8

/-- The number of disks in a selling pack -/
def selling_pack : ℕ := 4

/-- The price of a selling pack in dollars -/
def selling_price : ℚ := 10

/-- The target profit in dollars -/
def target_profit : ℚ := 120

/-- The minimum number of disks that must be sold to reach the target profit -/
def disks_to_sell : ℕ := 134

theorem profit_reached :
  let cost_per_disk : ℚ := buying_cost / buying_pack
  let price_per_disk : ℚ := selling_price / selling_pack
  let profit_per_disk : ℚ := price_per_disk - cost_per_disk
  (disks_to_sell : ℚ) * profit_per_disk ≥ target_profit ∧
  ∀ n : ℕ, (n : ℚ) * profit_per_disk ≥ target_profit → n ≥ disks_to_sell :=
by sorry

end NUMINAMATH_CALUDE_profit_reached_l1588_158808


namespace NUMINAMATH_CALUDE_class_average_age_problem_l1588_158851

theorem class_average_age_problem (original_students : ℕ) (new_students : ℕ) (new_average_age : ℕ) (average_decrease : ℕ) :
  original_students = 18 →
  new_students = 18 →
  new_average_age = 32 →
  average_decrease = 4 →
  ∃ (original_average : ℕ),
    (original_students * original_average + new_students * new_average_age) / (original_students + new_students) = original_average - average_decrease ∧
    original_average = 40 := by
  sorry

end NUMINAMATH_CALUDE_class_average_age_problem_l1588_158851


namespace NUMINAMATH_CALUDE_no_solution_implies_m_geq_two_l1588_158870

theorem no_solution_implies_m_geq_two (m : ℝ) :
  (∀ x : ℝ, ¬(2*x - 1 < 3 ∧ x > m)) → m ≥ 2 := by
  sorry

end NUMINAMATH_CALUDE_no_solution_implies_m_geq_two_l1588_158870


namespace NUMINAMATH_CALUDE_monomial_properties_l1588_158860

-- Define a monomial as a product of a coefficient and variables with non-negative integer exponents
structure Monomial (α : Type*) [CommRing α] where
  coeff : α
  vars : List (Nat × Nat)

-- Define the coefficient of a monomial
def coefficient {α : Type*} [CommRing α] (m : Monomial α) : α := m.coeff

-- Define the degree of a monomial
def degree {α : Type*} [CommRing α] (m : Monomial α) : Nat :=
  m.vars.foldl (fun acc (_, exp) => acc + exp) 0

-- The monomial -1/3 * x * y^2
def m : Monomial ℚ := ⟨-1/3, [(1, 1), (2, 2)]⟩

-- Theorem statement
theorem monomial_properties :
  coefficient m = -1/3 ∧ degree m = 3 := by sorry

end NUMINAMATH_CALUDE_monomial_properties_l1588_158860


namespace NUMINAMATH_CALUDE_negation_of_existence_negation_of_proposition_l1588_158821

theorem negation_of_existence (p : ℝ → Prop) :
  (¬∃ x, p x) ↔ (∀ x, ¬p x) := by sorry

theorem negation_of_proposition :
  (¬∃ x : ℝ, 2 * x + 1 ≤ 0) ↔ (∀ x : ℝ, 2 * x + 1 > 0) := by sorry

end NUMINAMATH_CALUDE_negation_of_existence_negation_of_proposition_l1588_158821


namespace NUMINAMATH_CALUDE_root_product_sum_l1588_158834

theorem root_product_sum (x₁ x₂ x₃ : ℝ) : 
  x₁ < x₂ ∧ x₂ < x₃ ∧ 
  (∀ x, Real.sqrt 2021 * x^3 - 4043 * x^2 + 3 = 0 ↔ x = x₁ ∨ x = x₂ ∨ x = x₃) →
  x₂ * (x₁ + x₃) = (9 * (Real.sqrt 2021)^2 - 3 * Real.sqrt 2021 * Real.sqrt (9 * (Real.sqrt 2021)^2 + 12)) / 2 :=
by sorry

end NUMINAMATH_CALUDE_root_product_sum_l1588_158834


namespace NUMINAMATH_CALUDE_smallest_x_for_simplified_fractions_l1588_158837

theorem smallest_x_for_simplified_fractions : ∃ (x : ℕ), x > 0 ∧
  (∀ (k : ℕ), k ≥ 1 ∧ k ≤ 40 → Nat.gcd (3*x + k) (k + 7) = 1) ∧
  (∀ (y : ℕ), y > 0 ∧ y < x → ∃ (k : ℕ), k ≥ 1 ∧ k ≤ 40 ∧ Nat.gcd (3*y + k) (k + 7) ≠ 1) ∧
  x = 5 :=
by sorry

end NUMINAMATH_CALUDE_smallest_x_for_simplified_fractions_l1588_158837


namespace NUMINAMATH_CALUDE_jar_price_calculation_l1588_158896

noncomputable def jar_price (d h p : ℝ) (d' h' : ℝ) : ℝ :=
  p * (d' / d)^2 * (h' / h)

theorem jar_price_calculation (d₁ h₁ p₁ d₂ h₂ : ℝ) 
  (hd₁ : d₁ = 2) (hh₁ : h₁ = 5) (hp₁ : p₁ = 0.75)
  (hd₂ : d₂ = 4) (hh₂ : h₂ = 8) :
  jar_price d₁ h₁ p₁ d₂ h₂ = 2.40 := by
  sorry

end NUMINAMATH_CALUDE_jar_price_calculation_l1588_158896


namespace NUMINAMATH_CALUDE_line_through_points_with_slope_one_l1588_158867

/-- Given a line passing through points M(-2, a) and N(a, 4) with a slope of 1, prove that a = 1 -/
theorem line_through_points_with_slope_one (a : ℝ) : 
  (let M := (-2, a)
   let N := (a, 4)
   (4 - a) / (a - (-2)) = 1) → 
  a = 1 := by
  sorry

end NUMINAMATH_CALUDE_line_through_points_with_slope_one_l1588_158867


namespace NUMINAMATH_CALUDE_minyoung_position_l1588_158881

/-- Given a line of people, calculates the position from the front given the position from the back -/
def positionFromFront (totalPeople : ℕ) (positionFromBack : ℕ) : ℕ :=
  totalPeople - positionFromBack + 1

/-- Proves that in a line of 27 people, the 13th person from the back is 15th from the front -/
theorem minyoung_position :
  positionFromFront 27 13 = 15 := by
sorry

end NUMINAMATH_CALUDE_minyoung_position_l1588_158881


namespace NUMINAMATH_CALUDE_class_size_l1588_158872

/-- Proves that in a class where the number of girls is 0.4 of the number of boys
    and there are 10 girls, the total number of students is 35. -/
theorem class_size (boys girls : ℕ) : 
  girls = 10 → 
  girls = (2 / 5 : ℚ) * boys → 
  boys + girls = 35 := by
sorry

end NUMINAMATH_CALUDE_class_size_l1588_158872


namespace NUMINAMATH_CALUDE_product_zero_implies_factor_zero_l1588_158883

theorem product_zero_implies_factor_zero (a b : ℝ) : a * b = 0 → (a = 0 ∨ b = 0) := by
  contrapose
  intro h
  push_neg
  simp
  sorry

end NUMINAMATH_CALUDE_product_zero_implies_factor_zero_l1588_158883


namespace NUMINAMATH_CALUDE_sin_plus_sqrt3_cos_l1588_158845

/-- Given an angle θ in the second quadrant such that tan(θ + π/3) = 1/2,
    prove that sin θ + √3 cos θ = -2√5/5 -/
theorem sin_plus_sqrt3_cos (θ : Real) 
  (h1 : π/2 < θ ∧ θ < π) -- θ is in the second quadrant
  (h2 : Real.tan (θ + π/3) = 1/2) : 
  Real.sin θ + Real.sqrt 3 * Real.cos θ = -2 * Real.sqrt 5 / 5 := by
  sorry


end NUMINAMATH_CALUDE_sin_plus_sqrt3_cos_l1588_158845


namespace NUMINAMATH_CALUDE_point_reflection_y_axis_l1588_158882

/-- Given a point Q with coordinates (-3,7) in the Cartesian coordinate system,
    its coordinates with respect to the y-axis are (3,7). -/
theorem point_reflection_y_axis :
  let Q : ℝ × ℝ := (-3, 7)
  let reflected_Q : ℝ × ℝ := (3, 7)
  reflected_Q = (- Q.1, Q.2) :=
by sorry

end NUMINAMATH_CALUDE_point_reflection_y_axis_l1588_158882


namespace NUMINAMATH_CALUDE_rectangular_field_area_l1588_158807

theorem rectangular_field_area (w l : ℝ) : 
  l = 2 * w + 35 →
  2 * (w + l) = 700 →
  w * l = 25725 := by
sorry

end NUMINAMATH_CALUDE_rectangular_field_area_l1588_158807


namespace NUMINAMATH_CALUDE_line_equivalence_l1588_158894

/-- Given a line in the form (2, -1) · ((x, y) - (1, -3)) = 0, prove it's equivalent to y = 2x - 5 --/
theorem line_equivalence :
  ∀ (x y : ℝ), (2 : ℝ) * (x - 1) + (-1 : ℝ) * (y + 3) = 0 ↔ y = 2 * x - 5 := by
  sorry

end NUMINAMATH_CALUDE_line_equivalence_l1588_158894


namespace NUMINAMATH_CALUDE_cubic_function_property_l1588_158812

/-- Given a cubic function f(x) = ax³ - bx + 5 where a and b are real numbers,
    if f(-3) = -1, then f(3) = 11. -/
theorem cubic_function_property (a b : ℝ) :
  let f : ℝ → ℝ := λ x ↦ a * x^3 - b * x + 5
  f (-3) = -1 → f 3 = 11 := by
sorry

end NUMINAMATH_CALUDE_cubic_function_property_l1588_158812


namespace NUMINAMATH_CALUDE_boat_speed_ratio_l1588_158857

/-- Proves that the ratio of average speed to still water speed is 42/65 for a boat traveling in a river --/
theorem boat_speed_ratio :
  let still_water_speed : ℝ := 20
  let current_speed : ℝ := 8
  let downstream_distance : ℝ := 10
  let upstream_distance : ℝ := 10
  let downstream_speed : ℝ := still_water_speed + current_speed
  let upstream_speed : ℝ := still_water_speed - current_speed
  let total_time : ℝ := downstream_distance / downstream_speed + upstream_distance / upstream_speed
  let total_distance : ℝ := downstream_distance + upstream_distance
  let average_speed : ℝ := total_distance / total_time
  average_speed / still_water_speed = 42 / 65 := by
  sorry


end NUMINAMATH_CALUDE_boat_speed_ratio_l1588_158857


namespace NUMINAMATH_CALUDE_ellipse_equation_l1588_158884

theorem ellipse_equation (a b : ℝ) (h1 : a > b) (h2 : b > 0) :
  (∃ c : ℝ, c = 2 ∧ c^2 = a^2 - b^2) →  -- Right focus coincides with parabola focus
  (a / 2 = c) →  -- Eccentricity is 1/2
  (∀ x y : ℝ, x^2 / a^2 + y^2 / b^2 = 1 ↔ x^2 / 16 + y^2 / 12 = 1) :=
by sorry

end NUMINAMATH_CALUDE_ellipse_equation_l1588_158884


namespace NUMINAMATH_CALUDE_inverse_variation_cube_and_sqrt_l1588_158818

theorem inverse_variation_cube_and_sqrt (k : ℝ) :
  (∀ x > 0, x^3 * Real.sqrt x = k) →
  (4^3 * Real.sqrt 4 = 2 * k) →
  (16^3 * Real.sqrt 16 = 128 * k) := by
sorry

end NUMINAMATH_CALUDE_inverse_variation_cube_and_sqrt_l1588_158818


namespace NUMINAMATH_CALUDE_area_difference_circle_square_l1588_158829

/-- The difference between the area of a circle with diameter 8 inches and 
    the area of a square with diagonal 8 inches is approximately 18.3 square inches. -/
theorem area_difference_circle_square : 
  let circle_diameter : ℝ := 8
  let square_diagonal : ℝ := 8
  let circle_area : ℝ := π * (circle_diameter / 2)^2
  let square_area : ℝ := (square_diagonal^2) / 2
  let area_difference : ℝ := circle_area - square_area
  ∃ ε > 0, abs (area_difference - 18.3) < ε ∧ ε < 0.1 :=
by sorry

end NUMINAMATH_CALUDE_area_difference_circle_square_l1588_158829


namespace NUMINAMATH_CALUDE_max_sum_given_sum_squares_and_product_l1588_158844

theorem max_sum_given_sum_squares_and_product (x y : ℝ) 
  (h1 : x^2 + y^2 = 130) (h2 : x * y = 45) : 
  x + y ≤ Real.sqrt 220 := by
  sorry

end NUMINAMATH_CALUDE_max_sum_given_sum_squares_and_product_l1588_158844


namespace NUMINAMATH_CALUDE_diamonds_sequence_property_diamonds_10th_figure_l1588_158890

/-- The number of diamonds in the nth figure of the sequence -/
def diamonds (n : ℕ) : ℕ :=
  if n = 1 then 1
  else if n = 2 then 5
  else 1 + 8 * (n - 1) * n

/-- The sequence satisfies the given conditions -/
theorem diamonds_sequence_property (n : ℕ) (h : n ≥ 3) :
  diamonds n = diamonds (n-1) + 8 * (n-1) :=
sorry

/-- The total number of diamonds in the 10th figure is 721 -/
theorem diamonds_10th_figure :
  diamonds 10 = 721 :=
sorry

end NUMINAMATH_CALUDE_diamonds_sequence_property_diamonds_10th_figure_l1588_158890


namespace NUMINAMATH_CALUDE_pool_cleaning_threshold_l1588_158871

/-- Represents the pool maintenance scenario -/
structure PoolMaintenance where
  capacity : ℕ  -- Pool capacity in milliliters
  splash_per_jump : ℕ  -- Amount of water splashed out per jump in milliliters
  num_jumps : ℕ  -- Number of jumps before cleaning

/-- Calculates the percentage of water remaining in the pool after jumps -/
def remaining_water_percentage (p : PoolMaintenance) : ℚ :=
  let remaining_water := p.capacity - p.splash_per_jump * p.num_jumps
  (remaining_water : ℚ) / (p.capacity : ℚ) * 100

/-- Theorem stating that the remaining water percentage is 80% for the given scenario -/
theorem pool_cleaning_threshold (p : PoolMaintenance) 
  (h1 : p.capacity = 2000000)
  (h2 : p.splash_per_jump = 400)
  (h3 : p.num_jumps = 1000) :
  remaining_water_percentage p = 80 := by
  sorry


end NUMINAMATH_CALUDE_pool_cleaning_threshold_l1588_158871


namespace NUMINAMATH_CALUDE_time_to_work_l1588_158877

-- Define the variables
def speed_to_work : ℝ := 80
def speed_to_home : ℝ := 120
def total_time : ℝ := 3

-- Define the theorem
theorem time_to_work : 
  ∃ (distance : ℝ),
    distance / speed_to_work + distance / speed_to_home = total_time ∧
    (distance / speed_to_work) * 60 = 108 := by
  sorry

end NUMINAMATH_CALUDE_time_to_work_l1588_158877


namespace NUMINAMATH_CALUDE_grocery_receipt_total_cost_l1588_158885

/-- The total cost of three items after applying a tax -/
def totalCostAfterTax (sponge shampoo soap taxRate : ℚ) : ℚ :=
  let preTaxTotal := sponge + shampoo + soap
  let taxAmount := preTaxTotal * taxRate
  preTaxTotal + taxAmount

/-- Theorem stating that the total cost after tax for the given items is $15.75 -/
theorem grocery_receipt_total_cost :
  totalCostAfterTax (420/100) (760/100) (320/100) (5/100) = 1575/100 := by
  sorry

end NUMINAMATH_CALUDE_grocery_receipt_total_cost_l1588_158885


namespace NUMINAMATH_CALUDE_expression_equality_l1588_158847

theorem expression_equality : 6 * 111 - 2 * 111 = 444 := by
  sorry

end NUMINAMATH_CALUDE_expression_equality_l1588_158847


namespace NUMINAMATH_CALUDE_joshua_share_is_30_l1588_158835

/-- Joshua and Justin shared money, with Joshua's share being thrice Justin's. -/
def shared_money (total : ℕ) (joshua_share : ℕ) (justin_share : ℕ) : Prop :=
  joshua_share + justin_share = total ∧ joshua_share = 3 * justin_share

/-- The theorem states that if Joshua and Justin shared $40 with Joshua's share being thrice Justin's,
    then Joshua's share is $30. -/
theorem joshua_share_is_30 :
  ∃ (justin_share : ℕ), shared_money 40 30 justin_share :=
by sorry

end NUMINAMATH_CALUDE_joshua_share_is_30_l1588_158835


namespace NUMINAMATH_CALUDE_probability_two_absent_one_present_l1588_158899

theorem probability_two_absent_one_present :
  let p_absent : ℚ := 1 / 30
  let p_present : ℚ := 1 - p_absent
  let p_two_absent_one_present : ℚ := 3 * p_absent * p_absent * p_present
  p_two_absent_one_present = 29 / 9000 := by sorry

end NUMINAMATH_CALUDE_probability_two_absent_one_present_l1588_158899


namespace NUMINAMATH_CALUDE_consecutive_negative_integers_sum_l1588_158886

theorem consecutive_negative_integers_sum (x : ℤ) : 
  x < 0 ∧ x * (x + 1) = 3080 → x + (x + 1) = -111 := by
  sorry

end NUMINAMATH_CALUDE_consecutive_negative_integers_sum_l1588_158886


namespace NUMINAMATH_CALUDE_smallest_n_for_non_simplest_fraction_l1588_158827

theorem smallest_n_for_non_simplest_fraction : ∃ (d : ℕ), d > 1 ∧ d ∣ (17 + 2) ∧ d ∣ (3 * 17^2 + 7) ∧
  ∀ (n : ℕ), n > 0 ∧ n < 17 → ∀ (k : ℕ), k > 1 → ¬(k ∣ (n + 2) ∧ k ∣ (3 * n^2 + 7)) :=
by sorry

#check smallest_n_for_non_simplest_fraction

end NUMINAMATH_CALUDE_smallest_n_for_non_simplest_fraction_l1588_158827


namespace NUMINAMATH_CALUDE_function_composition_equality_l1588_158846

theorem function_composition_equality (m n p q : ℝ) 
  (f : ℝ → ℝ) (g : ℝ → ℝ) 
  (hf : ∀ x, f x = m * x + n) 
  (hg : ∀ x, g x = p * x + q) : 
  (∃ x, f (g x) = g (f x)) ↔ n * (1 - p) = q * (1 - m) := by
sorry

end NUMINAMATH_CALUDE_function_composition_equality_l1588_158846


namespace NUMINAMATH_CALUDE_least_common_multiple_3_4_6_7_8_l1588_158875

theorem least_common_multiple_3_4_6_7_8 : ∃ (n : ℕ), n > 0 ∧ 
  (∀ (m : ℕ), m > 0 → (3 ∣ m) ∧ (4 ∣ m) ∧ (6 ∣ m) ∧ (7 ∣ m) ∧ (8 ∣ m) → n ≤ m) ∧
  (3 ∣ n) ∧ (4 ∣ n) ∧ (6 ∣ n) ∧ (7 ∣ n) ∧ (8 ∣ n) :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_least_common_multiple_3_4_6_7_8_l1588_158875


namespace NUMINAMATH_CALUDE_least_number_divisible_by_four_primes_l1588_158825

theorem least_number_divisible_by_four_primes : 
  ∃ (n : ℕ), n > 0 ∧ 
  (∃ (p₁ p₂ p₃ p₄ : ℕ), Prime p₁ ∧ Prime p₂ ∧ Prime p₃ ∧ Prime p₄ ∧ 
   p₁ ≠ p₂ ∧ p₁ ≠ p₃ ∧ p₁ ≠ p₄ ∧ p₂ ≠ p₃ ∧ p₂ ≠ p₄ ∧ p₃ ≠ p₄ ∧
   p₁ ∣ n ∧ p₂ ∣ n ∧ p₃ ∣ n ∧ p₄ ∣ n) ∧
  (∀ m : ℕ, m > 0 → 
    (∃ (q₁ q₂ q₃ q₄ : ℕ), Prime q₁ ∧ Prime q₂ ∧ Prime q₃ ∧ Prime q₄ ∧ 
     q₁ ≠ q₂ ∧ q₁ ≠ q₃ ∧ q₁ ≠ q₄ ∧ q₂ ≠ q₃ ∧ q₂ ≠ q₄ ∧ q₃ ≠ q₄ ∧
     q₁ ∣ m ∧ q₂ ∣ m ∧ q₃ ∣ m ∧ q₄ ∣ m) → 
    n ≤ m) ∧
  n = 210 :=
by sorry

end NUMINAMATH_CALUDE_least_number_divisible_by_four_primes_l1588_158825


namespace NUMINAMATH_CALUDE_son_age_proof_l1588_158848

theorem son_age_proof (son_age father_age : ℕ) : 
  father_age = son_age + 26 →
  father_age + 2 = 2 * (son_age + 2) →
  son_age = 24 :=
by
  sorry

end NUMINAMATH_CALUDE_son_age_proof_l1588_158848


namespace NUMINAMATH_CALUDE_don_max_bottles_l1588_158892

/-- The number of bottles Shop A sells to Don -/
def shop_a_bottles : ℕ := 150

/-- The number of bottles Shop B sells to Don -/
def shop_b_bottles : ℕ := 180

/-- The number of bottles Shop C sells to Don -/
def shop_c_bottles : ℕ := 220

/-- The maximum number of bottles Don can buy -/
def max_bottles : ℕ := shop_a_bottles + shop_b_bottles + shop_c_bottles

theorem don_max_bottles : max_bottles = 550 := by sorry

end NUMINAMATH_CALUDE_don_max_bottles_l1588_158892


namespace NUMINAMATH_CALUDE_used_car_lot_vehicles_l1588_158814

theorem used_car_lot_vehicles (total_vehicles : ℕ) : 
  (total_vehicles / 3 : ℚ) * 2 + -- tires from motorcycles
  (total_vehicles * 2 / 3 * 3 / 4 : ℚ) * 4 + -- tires from cars without spare
  (total_vehicles * 2 / 3 * 1 / 4 : ℚ) * 5 = 84 → -- tires from cars with spare
  total_vehicles = 24 := by
sorry

end NUMINAMATH_CALUDE_used_car_lot_vehicles_l1588_158814


namespace NUMINAMATH_CALUDE_julio_bonus_julio_bonus_is_50_l1588_158898

/-- Calculates Julio's bonus given his commission rate, customer numbers, salary, and total earnings -/
theorem julio_bonus 
  (commission_rate : ℕ) 
  (first_week_customers : ℕ) 
  (salary : ℕ) 
  (total_earnings : ℕ) : ℕ :=
  let second_week_customers := 2 * first_week_customers
  let third_week_customers := 3 * first_week_customers
  let total_commission := commission_rate * (first_week_customers + second_week_customers + third_week_customers)
  let bonus := total_earnings - salary - total_commission
  by
    -- Proof goes here
    sorry

/-- Proves that Julio's bonus is $50 given the specific conditions -/
theorem julio_bonus_is_50 : 
  julio_bonus 1 35 500 760 = 50 :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_CALUDE_julio_bonus_julio_bonus_is_50_l1588_158898


namespace NUMINAMATH_CALUDE_partition_exists_l1588_158897

-- Define the type for our partition function
def PartitionFunction := ℕ+ → Fin 100

-- Define the property that the partition satisfies the required condition
def SatisfiesCondition (f : PartitionFunction) : Prop :=
  ∀ a b c : ℕ+, a + 99 * b = c → f a = f b ∨ f a = f c ∨ f b = f c

-- State the theorem
theorem partition_exists : ∃ f : PartitionFunction, SatisfiesCondition f := by
  sorry

end NUMINAMATH_CALUDE_partition_exists_l1588_158897


namespace NUMINAMATH_CALUDE_min_length_line_segment_ellipse_l1588_158876

/-- The minimum length of a line segment AB on an ellipse -/
theorem min_length_line_segment_ellipse (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a > b) :
  let ellipse := {p : ℝ × ℝ | (p.1 ^ 2 / a ^ 2) + (p.2 ^ 2 / b ^ 2) = 1}
  ∃ (A B : ℝ × ℝ), A ∈ ellipse ∧ B ∈ ellipse ∧ 
    (A.1 * B.1 + A.2 * B.2 = 0) ∧  -- OA ⊥ OB
    ∀ (C D : ℝ × ℝ), C ∈ ellipse → D ∈ ellipse → (C.1 * D.1 + C.2 * D.2 = 0) →
      (A.1 - B.1) ^ 2 + (A.2 - B.2) ^ 2 ≤ (C.1 - D.1) ^ 2 + (C.2 - D.2) ^ 2 ∧
      (A.1 - B.1) ^ 2 + (A.2 - B.2) ^ 2 = (2 * a * b * Real.sqrt (a ^ 2 + b ^ 2) / (a ^ 2 + b ^ 2)) ^ 2 :=
by sorry

end NUMINAMATH_CALUDE_min_length_line_segment_ellipse_l1588_158876


namespace NUMINAMATH_CALUDE_dmv_waiting_time_solution_l1588_158815

/-- Represents the waiting time problem at the DMV. -/
def DMVWaitingTime (x : ℚ) : Prop :=
  let timeToTakeNumber : ℚ := 20
  let timeWaitingForCall : ℚ := 20 * x + 14
  let totalWaitingTime : ℚ := 114
  (timeToTakeNumber + timeWaitingForCall = totalWaitingTime) ∧
  (timeWaitingForCall / timeToTakeNumber = 47 / 10)

/-- The theorem stating that there exists a solution to the DMV waiting time problem. -/
theorem dmv_waiting_time_solution : ∃ x : ℚ, DMVWaitingTime x := by
  sorry

end NUMINAMATH_CALUDE_dmv_waiting_time_solution_l1588_158815


namespace NUMINAMATH_CALUDE_lilith_cap_collection_years_l1588_158852

/-- Represents the cap collection problem for Lilith --/
def cap_collection_problem (years : ℕ) : Prop :=
  let first_year_caps := 3 * 12
  let subsequent_year_caps := 5 * 12
  let christmas_caps := 40
  let lost_caps := 15
  let total_caps := 401
  
  first_year_caps +
  (years - 1) * subsequent_year_caps +
  years * christmas_caps -
  years * lost_caps = total_caps

/-- Theorem stating that Lilith has been collecting caps for 5 years --/
theorem lilith_cap_collection_years : 
  ∃ (years : ℕ), years > 0 ∧ cap_collection_problem years ∧ years = 5 := by
  sorry

end NUMINAMATH_CALUDE_lilith_cap_collection_years_l1588_158852


namespace NUMINAMATH_CALUDE_function_property_l1588_158805

/-- Given a function f(x) = ax^3 - bx^(3/5) + 1, if f(-1) = 3, then f(1) = 1 -/
theorem function_property (a b : ℝ) :
  let f : ℝ → ℝ := λ x ↦ a * x^3 - b * x^(3/5) + 1
  f (-1) = 3 → f 1 = 1 := by
  sorry

end NUMINAMATH_CALUDE_function_property_l1588_158805


namespace NUMINAMATH_CALUDE_box_surface_area_proof_l1588_158841

noncomputable def surface_area_of_box (a b c : ℝ) : ℝ :=
  2 * (a * b + b * c + c * a)

theorem box_surface_area_proof (a b c : ℝ) 
  (h1 : a ≤ b) 
  (h2 : b ≤ c) 
  (h3 : c = 2 * a) 
  (h4 : 4 * a + 4 * b + 4 * c = 180) 
  (h5 : Real.sqrt (a^2 + b^2 + c^2) = 25) :
  ∃ ε > 0, |surface_area_of_box a b c - 1051.540| < ε :=
sorry

end NUMINAMATH_CALUDE_box_surface_area_proof_l1588_158841


namespace NUMINAMATH_CALUDE_halfway_point_sixth_twelfth_l1588_158806

theorem halfway_point_sixth_twelfth (x y : ℚ) (hx : x = 1/6) (hy : y = 1/12) :
  (x + y) / 2 = 1/8 := by
  sorry

end NUMINAMATH_CALUDE_halfway_point_sixth_twelfth_l1588_158806


namespace NUMINAMATH_CALUDE_problem_solution_l1588_158820

theorem problem_solution : (3.242 * 14) / 100 = 0.45388 := by sorry

end NUMINAMATH_CALUDE_problem_solution_l1588_158820


namespace NUMINAMATH_CALUDE_least_value_f_1998_l1588_158838

/-- A function from positive integers to positive integers satisfying the given condition -/
def SpecialFunction (f : ℕ+ → ℕ+) : Prop :=
  ∀ s t : ℕ+, f (t^2 * f s) = s * (f t)^2

/-- The theorem stating the least possible value of f(1998) -/
theorem least_value_f_1998 :
  ∃ (f : ℕ+ → ℕ+), SpecialFunction f ∧
    (∀ g : ℕ+ → ℕ+, SpecialFunction g → f 1998 ≤ g 1998) ∧
    f 1998 = 120 :=
sorry

end NUMINAMATH_CALUDE_least_value_f_1998_l1588_158838


namespace NUMINAMATH_CALUDE_partial_fraction_decomposition_l1588_158889

theorem partial_fraction_decomposition :
  ∃ (C D : ℚ), 
    (∀ x : ℚ, x ≠ 9 ∧ x ≠ -4 →
      (6 * x + 5) / (x^2 - 5*x - 36) = C / (x - 9) + D / (x + 4)) ∧
    C = 59 / 13 ∧
    D = 19 / 13 := by
  sorry

end NUMINAMATH_CALUDE_partial_fraction_decomposition_l1588_158889


namespace NUMINAMATH_CALUDE_astrophysics_degrees_l1588_158869

def microphotonics : Real := 12
def home_electronics : Real := 24
def food_additives : Real := 15
def genetically_modified_microorganisms : Real := 29
def industrial_lubricants : Real := 8
def total_degrees : Real := 360

def other_sectors_total : Real :=
  microphotonics + home_electronics + food_additives + 
  genetically_modified_microorganisms + industrial_lubricants

def astrophysics_percentage : Real := 100 - other_sectors_total

theorem astrophysics_degrees : 
  (astrophysics_percentage / 100) * total_degrees = 43.2 := by
  sorry

end NUMINAMATH_CALUDE_astrophysics_degrees_l1588_158869


namespace NUMINAMATH_CALUDE_field_length_is_32_l1588_158858

/-- Proves that a rectangular field with specific properties has a length of 32 meters -/
theorem field_length_is_32 (l w : ℝ) (h1 : l = 2 * w) (h2 : (8 * 8 : ℝ) = (1 / 8) * (l * w)) : l = 32 :=
by sorry

end NUMINAMATH_CALUDE_field_length_is_32_l1588_158858


namespace NUMINAMATH_CALUDE_roots_in_interval_l1588_158856

theorem roots_in_interval (m : ℝ) :
  (∀ x, 4 * x^2 - (3 * m + 1) * x - m - 2 = 0 → -1 < x ∧ x < 2) ↔ -1 < m ∧ m < 12/7 := by
  sorry

end NUMINAMATH_CALUDE_roots_in_interval_l1588_158856


namespace NUMINAMATH_CALUDE_min_distance_between_graphs_l1588_158840

-- Define the two functions
def f (x : ℝ) : ℝ := |x|
def g (x : ℝ) : ℝ := -x^2 + 2*x + 3

-- Define the distance function between the two graphs
def distance (x : ℝ) : ℝ := |f x - g x|

-- Theorem statement
theorem min_distance_between_graphs :
  ∃ (min_dist : ℝ), min_dist = 3/4 ∧ ∀ (x : ℝ), distance x ≥ min_dist :=
sorry

end NUMINAMATH_CALUDE_min_distance_between_graphs_l1588_158840


namespace NUMINAMATH_CALUDE_stream_speed_l1588_158850

/-- Proves that the speed of a stream is 3 kmph given specific boat travel times -/
theorem stream_speed (boat_speed : ℝ) (downstream_time : ℝ) (upstream_time : ℝ) : 
  boat_speed = 15 →
  downstream_time = 1 →
  upstream_time = 1.5 →
  (boat_speed + 3) * downstream_time = (boat_speed - 3) * upstream_time :=
by sorry

end NUMINAMATH_CALUDE_stream_speed_l1588_158850


namespace NUMINAMATH_CALUDE_polynomial_simplification_l1588_158863

theorem polynomial_simplification (x : ℝ) : 
  (3*x + 2) * (3*x - 2) - (3*x - 1)^2 = 6*x - 5 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_simplification_l1588_158863


namespace NUMINAMATH_CALUDE_simple_interest_rate_for_doubling_l1588_158843

/-- Simple interest rate for a sum that doubles in 10 years -/
theorem simple_interest_rate_for_doubling (principal : ℝ) (h : principal > 0) :
  let years : ℝ := 10
  let final_amount : ℝ := 2 * principal
  let rate : ℝ := (final_amount - principal) / (principal * years) * 100
  rate = 10 := by
  sorry

end NUMINAMATH_CALUDE_simple_interest_rate_for_doubling_l1588_158843


namespace NUMINAMATH_CALUDE_carolyn_sum_is_24_l1588_158874

def game_list : List Nat := [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

def is_removable (n : Nat) (l : List Nat) : Bool :=
  ∃ m ∈ l, m ≠ n ∧ n % m = 0

def remove_divisors (n : Nat) (l : List Nat) : List Nat :=
  l.filter (fun m => m = n ∨ n % m ≠ 0)

def carolyn_moves (l : List Nat) : List Nat :=
  let after_first_move := l.filter (· ≠ 8)
  let after_paul_first := remove_divisors 8 after_first_move
  let second_move := after_paul_first.filter (· ≠ 10)
  let after_paul_second := remove_divisors 10 second_move
  let third_move := after_paul_second.filter (· ≠ 6)
  third_move

theorem carolyn_sum_is_24 :
  let carolyn_removed := [8, 10, 6]
  carolyn_removed.sum = 24 ∧
  (∀ n ∈ carolyn_moves game_list, ¬is_removable n (carolyn_moves game_list)) := by
  sorry

end NUMINAMATH_CALUDE_carolyn_sum_is_24_l1588_158874


namespace NUMINAMATH_CALUDE_inverse_of_A_l1588_158823

def A : Matrix (Fin 2) (Fin 2) ℚ := !![2, -1; 4, 3]

theorem inverse_of_A :
  let A_inv : Matrix (Fin 2) (Fin 2) ℚ := !![3/10, 1/10; -2/5, 1/5]
  A * A_inv = 1 ∧ A_inv * A = 1 := by sorry

end NUMINAMATH_CALUDE_inverse_of_A_l1588_158823


namespace NUMINAMATH_CALUDE_isosceles_triangle_perimeter_l1588_158809

/-- An isosceles triangle with side lengths 5 and 8 has a perimeter of either 18 or 21. -/
theorem isosceles_triangle_perimeter : 
  ∀ a b c : ℝ, 
  (a = 5 ∧ b = 8) ∨ (a = 8 ∧ b = 5) → 
  (a = b ∨ a = c ∨ b = c) → 
  (a + b + c = 18 ∨ a + b + c = 21) := by
sorry

end NUMINAMATH_CALUDE_isosceles_triangle_perimeter_l1588_158809


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l1588_158854

def A : Set Int := {-2, -1}
def B : Set Int := {-1, 2, 3}

theorem intersection_of_A_and_B : A ∩ B = {-1} := by sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l1588_158854


namespace NUMINAMATH_CALUDE_f_even_and_increasing_l1588_158879

def f (x : ℝ) := x^2

theorem f_even_and_increasing :
  (∀ x, f (-x) = f x) ∧
  (∀ a b, 0 ≤ a → a < b → f a ≤ f b) :=
by sorry

end NUMINAMATH_CALUDE_f_even_and_increasing_l1588_158879


namespace NUMINAMATH_CALUDE_board_number_equation_l1588_158822

theorem board_number_equation (n : ℤ) : 7 * n + 3 = (3 * n + 7) + 84 ↔ n = 22 := by
  sorry

end NUMINAMATH_CALUDE_board_number_equation_l1588_158822
