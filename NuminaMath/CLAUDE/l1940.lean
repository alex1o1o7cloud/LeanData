import Mathlib

namespace NUMINAMATH_CALUDE_total_jellybeans_proof_l1940_194034

/-- The number of jellybeans needed to fill a large drinking glass -/
def large_glass_beans : ℕ := 50

/-- The number of large drinking glasses -/
def num_large_glasses : ℕ := 5

/-- The number of small drinking glasses -/
def num_small_glasses : ℕ := 3

/-- The number of jellybeans needed to fill a small drinking glass -/
def small_glass_beans : ℕ := large_glass_beans / 2

/-- The total number of jellybeans needed to fill all glasses -/
def total_beans : ℕ := large_glass_beans * num_large_glasses + small_glass_beans * num_small_glasses

theorem total_jellybeans_proof : total_beans = 325 := by
  sorry

end NUMINAMATH_CALUDE_total_jellybeans_proof_l1940_194034


namespace NUMINAMATH_CALUDE_manuscript_cost_calculation_l1940_194038

/-- Calculates the total cost of typing a manuscript given the page counts and rates. -/
def manuscript_typing_cost (
  total_pages : ℕ) 
  (revised_once : ℕ) 
  (revised_twice : ℕ) 
  (revised_twice_sets : ℕ) 
  (revised_thrice : ℕ) 
  (revised_thrice_sets : ℕ) 
  (initial_rate : ℕ) 
  (revision_rate : ℕ) 
  (set_rate_thrice : ℕ) 
  (set_rate_twice : ℕ) : ℕ :=
  sorry

theorem manuscript_cost_calculation :
  manuscript_typing_cost 
    250  -- total pages
    80   -- pages revised once
    95   -- pages revised twice
    2    -- sets of 20 pages revised twice
    50   -- pages revised thrice
    3    -- sets of 10 pages revised thrice
    5    -- initial typing rate
    3    -- revision rate
    10   -- flat fee for set of 10 pages revised 3+ times
    15   -- flat fee for set of 20 pages revised 2 times
  = 1775 := by sorry

end NUMINAMATH_CALUDE_manuscript_cost_calculation_l1940_194038


namespace NUMINAMATH_CALUDE_boys_to_total_ratio_l1940_194052

theorem boys_to_total_ratio 
  (b g : ℕ) -- number of boys and girls
  (h1 : b > 0 ∧ g > 0) -- ensure non-empty class
  (h2 : (b : ℚ) / (b + g) = 4/5 * (g : ℚ) / (b + g)) -- probability condition
  : (b : ℚ) / (b + g) = 4/9 := by
  sorry

end NUMINAMATH_CALUDE_boys_to_total_ratio_l1940_194052


namespace NUMINAMATH_CALUDE_composition_of_even_is_even_l1940_194060

/-- A function f is even if f(-x) = f(x) for all x. -/
def IsEven (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = f x

/-- Given an even function f, prove that f(f(x)) is also even. -/
theorem composition_of_even_is_even (f : ℝ → ℝ) (hf : IsEven f) : IsEven (f ∘ f) := by
  sorry

end NUMINAMATH_CALUDE_composition_of_even_is_even_l1940_194060


namespace NUMINAMATH_CALUDE_girls_in_college_l1940_194003

theorem girls_in_college (total_students : ℕ) (boys_ratio girls_ratio : ℕ) : 
  total_students = 546 →
  boys_ratio = 8 →
  girls_ratio = 5 →
  ∃ (num_girls : ℕ), num_girls = 210 ∧ 
    boys_ratio * num_girls + girls_ratio * num_girls = girls_ratio * total_students :=
by
  sorry

end NUMINAMATH_CALUDE_girls_in_college_l1940_194003


namespace NUMINAMATH_CALUDE_existence_of_A_for_any_E_l1940_194015

/-- Property P: A sequence is a permutation of {1, 2, ..., n} -/
def has_property_P (A : List ℕ) : Prop :=
  A.length ≥ 2 ∧ A.Nodup ∧ ∀ i, i ∈ A → i ∈ Finset.range A.length

/-- T(A) sequence definition -/
def T (A : List ℕ) : List ℕ :=
  List.zipWith (fun a b => if a < b then 1 else 0) A A.tail

theorem existence_of_A_for_any_E (n : ℕ) (E : List ℕ) 
    (h_n : n ≥ 2) 
    (h_E_length : E.length = n - 1) 
    (h_E_elements : ∀ e ∈ E, e = 0 ∨ e = 1) :
    ∃ A : List ℕ, has_property_P A ∧ T A = E :=
  sorry

end NUMINAMATH_CALUDE_existence_of_A_for_any_E_l1940_194015


namespace NUMINAMATH_CALUDE_factorization_equality_l1940_194093

theorem factorization_equality (m x : ℝ) : m * x^2 - 4 * m = m * (x + 2) * (x - 2) := by
  sorry

end NUMINAMATH_CALUDE_factorization_equality_l1940_194093


namespace NUMINAMATH_CALUDE_rosie_lou_speed_ratio_l1940_194079

/-- The ratio of Rosie's speed to Lou's speed on a circular track -/
theorem rosie_lou_speed_ratio :
  let track_length : ℚ := 1/4  -- Length of the track in miles
  let lou_distance : ℚ := 3    -- Lou's total distance in miles
  let rosie_laps : ℕ := 24     -- Number of laps Rosie completes
  let rosie_distance : ℚ := rosie_laps * track_length  -- Rosie's total distance in miles
  ∀ (lou_speed rosie_speed : ℚ),
    lou_speed > 0 →  -- Lou's speed is positive
    rosie_speed > 0 →  -- Rosie's speed is positive
    lou_speed * lou_distance = rosie_speed * rosie_distance →  -- They run for the same duration
    rosie_speed / lou_speed = 2/1 :=
by sorry

end NUMINAMATH_CALUDE_rosie_lou_speed_ratio_l1940_194079


namespace NUMINAMATH_CALUDE_daves_trays_l1940_194028

/-- Given that Dave can carry 9 trays at a time, picked up 17 trays from one table,
    and made 8 trips in total, prove that he picked up 55 trays from the second table. -/
theorem daves_trays (trays_per_trip : ℕ) (trips : ℕ) (trays_first_table : ℕ)
    (h1 : trays_per_trip = 9)
    (h2 : trips = 8)
    (h3 : trays_first_table = 17) :
    trips * trays_per_trip - trays_first_table = 55 := by
  sorry

end NUMINAMATH_CALUDE_daves_trays_l1940_194028


namespace NUMINAMATH_CALUDE_abs_a_gt_abs_b_l1940_194021

theorem abs_a_gt_abs_b (a b : ℝ) (ha : a > 0) (hb : b < 0) (hab : a + b > 0) : |a| > |b| := by
  sorry

end NUMINAMATH_CALUDE_abs_a_gt_abs_b_l1940_194021


namespace NUMINAMATH_CALUDE_median_is_212_l1940_194027

/-- The sum of integers from 1 to n -/
def triangularSum (n : ℕ) : ℕ := n * (n + 1) / 2

/-- The total count of numbers in our special list up to n -/
def cumulativeCount (n : ℕ) : ℕ := triangularSum n

/-- The total length of our special list -/
def totalLength : ℕ := triangularSum 300

/-- The position of the lower median element -/
def lowerMedianPos : ℕ := totalLength / 2

/-- The position of the upper median element -/
def upperMedianPos : ℕ := lowerMedianPos + 1

theorem median_is_212 : 
  ∃ (n : ℕ), n = 212 ∧ 
  cumulativeCount (n - 1) < lowerMedianPos ∧
  cumulativeCount n ≥ upperMedianPos :=
sorry

end NUMINAMATH_CALUDE_median_is_212_l1940_194027


namespace NUMINAMATH_CALUDE_probability_4_club_2_is_1_663_l1940_194000

/-- Represents a standard deck of 52 playing cards -/
def StandardDeck : ℕ := 52

/-- Number of 4s in a standard deck -/
def NumberOf4s : ℕ := 4

/-- Number of clubs in a standard deck -/
def NumberOfClubs : ℕ := 13

/-- Number of 2s in a standard deck -/
def NumberOf2s : ℕ := 4

/-- Probability of drawing a 4 as the first card, a club as the second card, 
    and a 2 as the third card from a standard 52-card deck -/
def probability_4_club_2 : ℚ :=
  (NumberOf4s : ℚ) / StandardDeck *
  NumberOfClubs / (StandardDeck - 1) *
  NumberOf2s / (StandardDeck - 2)

theorem probability_4_club_2_is_1_663 : 
  probability_4_club_2 = 1 / 663 := by
  sorry

end NUMINAMATH_CALUDE_probability_4_club_2_is_1_663_l1940_194000


namespace NUMINAMATH_CALUDE_xiao_ming_score_l1940_194087

/-- Calculates the weighted score for a given component -/
def weightedScore (score : ℝ) (weight : ℝ) : ℝ := score * weight

/-- Calculates the total score based on individual scores and weights -/
def totalScore (regularScore midtermScore finalScore : ℝ) 
               (regularWeight midtermWeight finalWeight : ℝ) : ℝ :=
  weightedScore regularScore regularWeight + 
  weightedScore midtermScore midtermWeight + 
  weightedScore finalScore finalWeight

theorem xiao_ming_score : 
  let regularScore : ℝ := 70
  let midtermScore : ℝ := 80
  let finalScore : ℝ := 85
  let totalWeight : ℝ := 3 + 3 + 4
  let regularWeight : ℝ := 3 / totalWeight
  let midtermWeight : ℝ := 3 / totalWeight
  let finalWeight : ℝ := 4 / totalWeight
  totalScore regularScore midtermScore finalScore 
             regularWeight midtermWeight finalWeight = 79 := by
  sorry

end NUMINAMATH_CALUDE_xiao_ming_score_l1940_194087


namespace NUMINAMATH_CALUDE_complex_equation_solution_l1940_194080

theorem complex_equation_solution (x : ℝ) : 
  (1 - 2*Complex.I) * (x + Complex.I) = 4 - 3*Complex.I → x = 2 := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l1940_194080


namespace NUMINAMATH_CALUDE_intersection_max_value_l1940_194035

def P (a b : ℝ) (x : ℝ) : ℝ := x^6 - 8*x^5 + 24*x^4 - 37*x^3 + a*x^2 + b*x - 6

def L (d : ℝ) (x : ℝ) : ℝ := d*x + 2

theorem intersection_max_value (a b d : ℝ) :
  (∃ x y z : ℝ, x ≠ y ∧ 
    P a b x = L d x ∧
    P a b y = L d y ∧
    P a b z = L d z ∧
    (∀ t : ℝ, t ≠ z → (P a b t - L d t) / (t - z) ≠ 0)) →
  (∃ w : ℝ, P a b w = L d w ∧ ∀ v : ℝ, P a b v = L d v → v ≤ w ∧ w = 5) :=
by sorry

end NUMINAMATH_CALUDE_intersection_max_value_l1940_194035


namespace NUMINAMATH_CALUDE_company_results_l1940_194040

structure Company where
  team_a_success_prob : ℚ
  team_b_success_prob : ℚ
  profit_a_success : ℤ
  loss_a_failure : ℤ
  profit_b_success : ℤ
  loss_b_failure : ℤ

def company : Company := {
  team_a_success_prob := 3/4,
  team_b_success_prob := 3/5,
  profit_a_success := 120,
  loss_a_failure := 50,
  profit_b_success := 100,
  loss_b_failure := 40
}

def exactly_one_success_prob (c : Company) : ℚ :=
  (1 - c.team_a_success_prob) * c.team_b_success_prob +
  c.team_a_success_prob * (1 - c.team_b_success_prob)

def profit_distribution (c : Company) : List (ℤ × ℚ) :=
  [(-90, (1 - c.team_a_success_prob) * (1 - c.team_b_success_prob)),
   (50, (1 - c.team_a_success_prob) * c.team_b_success_prob),
   (80, c.team_a_success_prob * (1 - c.team_b_success_prob)),
   (220, c.team_a_success_prob * c.team_b_success_prob)]

theorem company_results :
  exactly_one_success_prob company = 9/20 ∧
  profit_distribution company = [(-90, 1/10), (50, 3/20), (80, 3/10), (220, 9/20)] := by
  sorry

end NUMINAMATH_CALUDE_company_results_l1940_194040


namespace NUMINAMATH_CALUDE_inequality_solution_set_l1940_194066

theorem inequality_solution_set (x : ℝ) : 
  |5*x - x^2| < 6 ↔ (-1 < x ∧ x < 2) ∨ (3 < x ∧ x < 6) := by
  sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l1940_194066


namespace NUMINAMATH_CALUDE_sqrt_meaningful_range_l1940_194094

theorem sqrt_meaningful_range (x : ℝ) :
  (∃ y : ℝ, y ^ 2 = x + 3) → x ≥ -3 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_meaningful_range_l1940_194094


namespace NUMINAMATH_CALUDE_unique_solution_l1940_194071

def equation (y : ℝ) : Prop :=
  y ≠ 0 ∧ y ≠ 3 ∧ (3 * y^2 - 15 * y) / (y^2 - 3 * y) = y + 1

theorem unique_solution :
  ∃! y : ℝ, equation y :=
sorry

end NUMINAMATH_CALUDE_unique_solution_l1940_194071


namespace NUMINAMATH_CALUDE_cost_for_23_days_l1940_194032

/-- Calculates the total cost of staying in a student youth hostel for a given number of days. -/
def hostelCost (days : ℕ) : ℚ :=
  let firstWeekRate : ℚ := 18
  let additionalWeekRate : ℚ := 14
  let firstWeekDays : ℕ := min days 7
  let additionalDays : ℕ := days - firstWeekDays
  firstWeekRate * firstWeekDays + additionalWeekRate * additionalDays

/-- Theorem stating that the cost for a 23-day stay is $350.00 -/
theorem cost_for_23_days :
  hostelCost 23 = 350 := by
  sorry

#eval hostelCost 23

end NUMINAMATH_CALUDE_cost_for_23_days_l1940_194032


namespace NUMINAMATH_CALUDE_consecutive_integers_sum_of_cubes_l1940_194078

theorem consecutive_integers_sum_of_cubes (a b c : ℕ) : 
  (a > 0) → 
  (b = a + 1) → 
  (c = b + 1) → 
  (a^2 + b^2 + c^2 = 2450) → 
  (a^3 + b^3 + c^3 = 73341) :=
by sorry

end NUMINAMATH_CALUDE_consecutive_integers_sum_of_cubes_l1940_194078


namespace NUMINAMATH_CALUDE_fluffy_spotted_cats_ratio_l1940_194058

theorem fluffy_spotted_cats_ratio (total_cats : ℕ) (fluffy_spotted_cats : ℕ) :
  total_cats = 120 →
  fluffy_spotted_cats = 10 →
  (total_cats / 3 : ℚ) = (total_cats / 3 : ℕ) →
  (fluffy_spotted_cats : ℚ) / (total_cats / 3 : ℚ) = 1 / 4 := by
  sorry

end NUMINAMATH_CALUDE_fluffy_spotted_cats_ratio_l1940_194058


namespace NUMINAMATH_CALUDE_exponent_multiplication_l1940_194019

theorem exponent_multiplication (a : ℝ) : a^3 * a^4 = a^7 := by
  sorry

end NUMINAMATH_CALUDE_exponent_multiplication_l1940_194019


namespace NUMINAMATH_CALUDE_total_volume_of_cubes_l1940_194023

def cube_volume (side_length : ℕ) : ℕ := side_length ^ 3

def total_volume (carl_cubes : ℕ) (kate_cubes : ℕ) (carl_side_length : ℕ) (kate_side_length : ℕ) : ℕ :=
  carl_cubes * cube_volume carl_side_length + kate_cubes * cube_volume kate_side_length

theorem total_volume_of_cubes : total_volume 4 3 3 4 = 300 := by
  sorry

end NUMINAMATH_CALUDE_total_volume_of_cubes_l1940_194023


namespace NUMINAMATH_CALUDE_second_girl_speed_l1940_194082

/-- Given two girls walking in opposite directions, prove that the second girl's speed is 3 km/hr -/
theorem second_girl_speed (girl1_speed : ℝ) (time : ℝ) (distance : ℝ) : 
  girl1_speed = 7 ∧ time = 12 ∧ distance = 120 →
  ∃ girl2_speed : ℝ, girl2_speed = 3 ∧ distance = (girl1_speed + girl2_speed) * time :=
by
  sorry

end NUMINAMATH_CALUDE_second_girl_speed_l1940_194082


namespace NUMINAMATH_CALUDE_selena_bashar_passes_l1940_194086

/-- Represents a runner on a circular track -/
structure Runner where
  speed : ℝ  -- speed in m/min
  radius : ℝ  -- radius of the lane in meters
  direction : ℤ  -- 1 for clockwise, -1 for counterclockwise

/-- Calculates the number of times two runners pass each other on a circular track -/
def number_of_passes (runner1 runner2 : Runner) (total_time : ℝ) (delay : ℝ) : ℕ :=
  sorry

theorem selena_bashar_passes : 
  let selena : Runner := ⟨200, 70, 1⟩
  let bashar : Runner := ⟨240, 80, -1⟩
  let total_time : ℝ := 35
  let delay : ℝ := 5
  number_of_passes selena bashar total_time delay = 21 := by
  sorry

end NUMINAMATH_CALUDE_selena_bashar_passes_l1940_194086


namespace NUMINAMATH_CALUDE_product_xyz_l1940_194055

theorem product_xyz (x y z : ℝ) 
  (h1 : x + 2/y = 2) 
  (h2 : y + 2/z = 2) 
  (h3 : y ≠ 0) 
  (h4 : z ≠ 0) : x * y * z = 2 := by
  sorry

end NUMINAMATH_CALUDE_product_xyz_l1940_194055


namespace NUMINAMATH_CALUDE_find_b_l1940_194016

def gcd_notation (x y : ℕ) : ℕ := x * y

theorem find_b : ∃ b : ℕ, gcd_notation (gcd_notation (16 * b) (18 * 24)) 2 = 2 ∧ b = 1 := by sorry

end NUMINAMATH_CALUDE_find_b_l1940_194016


namespace NUMINAMATH_CALUDE_probability_below_curve_probability_is_one_third_l1940_194076

/-- The probability that a randomly chosen point in the unit square falls below the curve y = x^2 is 1/3 -/
theorem probability_below_curve : Real → Prop := λ p =>
  let curve := λ x : Real => x^2
  let unit_square_area := 1
  let area_below_curve := ∫ x in (0 : Real)..1, curve x
  p = area_below_curve / unit_square_area ∧ p = 1/3

/-- The main theorem stating the probability is 1/3 -/
theorem probability_is_one_third : ∃ p : Real, probability_below_curve p := by
  sorry

end NUMINAMATH_CALUDE_probability_below_curve_probability_is_one_third_l1940_194076


namespace NUMINAMATH_CALUDE_early_finish_hours_l1940_194050

/-- Represents the number of workers -/
def num_workers : ℕ := 3

/-- Represents the normal working hours per day -/
def normal_hours : ℕ := 8

/-- Represents the number of customers served per hour by each worker -/
def customers_per_hour : ℕ := 7

/-- Represents the total number of customers served that day -/
def total_customers : ℕ := 154

/-- Theorem stating that the worker who finished early worked for 6 hours -/
theorem early_finish_hours :
  ∃ (h : ℕ),
    h < normal_hours ∧
    (2 * normal_hours * customers_per_hour + h * customers_per_hour = total_customers) ∧
    h = 6 :=
sorry

end NUMINAMATH_CALUDE_early_finish_hours_l1940_194050


namespace NUMINAMATH_CALUDE_correct_seating_arrangements_l1940_194062

-- Define the number of students and rows
def num_students : ℕ := 12
def num_rows : ℕ := 2
def students_per_row : ℕ := num_students / num_rows

-- Define the number of test versions
def num_versions : ℕ := 2

-- Define the function to calculate the number of seating arrangements
def seating_arrangements : ℕ := 2 * (Nat.factorial students_per_row)^2

-- Theorem statement
theorem correct_seating_arrangements :
  seating_arrangements = 1036800 :=
sorry

end NUMINAMATH_CALUDE_correct_seating_arrangements_l1940_194062


namespace NUMINAMATH_CALUDE_original_number_proof_l1940_194083

theorem original_number_proof : ∃! n : ℤ, n * 74 = 19732 := by
  sorry

end NUMINAMATH_CALUDE_original_number_proof_l1940_194083


namespace NUMINAMATH_CALUDE_f_is_even_l1940_194097

def f (x : ℝ) : ℝ := |x - 1| + |x + 1|

theorem f_is_even : ∀ x : ℝ, f (-x) = f x := by
  sorry

end NUMINAMATH_CALUDE_f_is_even_l1940_194097


namespace NUMINAMATH_CALUDE_imaginary_part_of_i_times_one_minus_i_l1940_194048

theorem imaginary_part_of_i_times_one_minus_i (i : ℂ) : 
  (Complex.I * (1 - Complex.I)).im = 1 := by sorry

end NUMINAMATH_CALUDE_imaginary_part_of_i_times_one_minus_i_l1940_194048


namespace NUMINAMATH_CALUDE_steak_weight_problem_l1940_194012

theorem steak_weight_problem (original_weight : ℝ) : 
  (0.8 * (0.5 * original_weight) = 12) → original_weight = 30 := by
  sorry

end NUMINAMATH_CALUDE_steak_weight_problem_l1940_194012


namespace NUMINAMATH_CALUDE_unique_number_l1940_194004

def is_two_digit (n : ℕ) : Prop := 10 ≤ n ∧ n ≤ 99

def is_odd (n : ℕ) : Prop := ∃ k, n = 2*k + 1

def is_multiple_of_13 (n : ℕ) : Prop := ∃ k, n = 13*k

def sum_of_digits (n : ℕ) : ℕ :=
  (n / 10) + (n % 10)

def is_perfect_square (n : ℕ) : Prop := ∃ k, n = k^2

theorem unique_number : 
  ∃! n : ℕ, 
    is_two_digit n ∧ 
    is_odd n ∧ 
    is_multiple_of_13 n ∧ 
    is_perfect_square (sum_of_digits n) ∧ 
    n = 13 := by sorry

end NUMINAMATH_CALUDE_unique_number_l1940_194004


namespace NUMINAMATH_CALUDE_curve_properties_l1940_194054

structure Curve where
  m : ℝ
  n : ℝ
  equation : ℝ → ℝ → Prop

def isEllipse (C : Curve) : Prop := sorry

def hasYAxisFoci (C : Curve) : Prop := sorry

def isHyperbola (C : Curve) : Prop := sorry

def hasAsymptotes (C : Curve) (f : ℝ → ℝ) : Prop := sorry

def isTwoLines (C : Curve) : Prop := sorry

theorem curve_properties (C : Curve) 
  (h_eq : C.equation = fun x y ↦ C.m * x^2 + C.n * y^2 = 1) :
  (C.m > C.n ∧ C.n > 0 → isEllipse C ∧ hasYAxisFoci C) ∧
  (C.m * C.n < 0 → isHyperbola C ∧ hasAsymptotes C (fun x ↦ Real.sqrt (-C.m / C.n) * x)) ∧
  (C.m = 0 ∧ C.n > 0 → isTwoLines C) := by
  sorry

end NUMINAMATH_CALUDE_curve_properties_l1940_194054


namespace NUMINAMATH_CALUDE_min_value_of_expression_min_value_attained_l1940_194057

theorem min_value_of_expression (x : ℝ) : 
  (14 - x) * (9 - x) * (14 + x) * (9 + x) ≥ -1156.25 :=
by sorry

theorem min_value_attained : 
  ∃ x : ℝ, (14 - x) * (9 - x) * (14 + x) * (9 + x) = -1156.25 :=
by sorry

end NUMINAMATH_CALUDE_min_value_of_expression_min_value_attained_l1940_194057


namespace NUMINAMATH_CALUDE_log_relationship_l1940_194001

theorem log_relationship : ∀ (a b : ℝ), 
  a = Real.log 135 / Real.log 4 → 
  b = Real.log 45 / Real.log 2 → 
  a = b / 2 := by
sorry

end NUMINAMATH_CALUDE_log_relationship_l1940_194001


namespace NUMINAMATH_CALUDE_distance_to_x_axis_reflection_triangle_DEF_reflection_distance_l1940_194046

/-- The distance between a point and its reflection over the x-axis --/
theorem distance_to_x_axis_reflection (x y : ℝ) : 
  Real.sqrt ((x - x)^2 + ((-y) - y)^2) = 2 * |y| := by
  sorry

/-- The specific case for the triangle DEF --/
theorem triangle_DEF_reflection_distance : 
  Real.sqrt ((2 - 2)^2 + ((-1) - 1)^2) = 2 := by
  sorry

end NUMINAMATH_CALUDE_distance_to_x_axis_reflection_triangle_DEF_reflection_distance_l1940_194046


namespace NUMINAMATH_CALUDE_rice_cost_l1940_194084

/-- Proves that the cost of each kilogram of rice is $2 given the conditions of Vicente's purchase --/
theorem rice_cost (rice_kg : ℕ) (meat_lb : ℕ) (meat_cost_per_lb : ℕ) (total_spent : ℕ) : 
  rice_kg = 5 → meat_lb = 3 → meat_cost_per_lb = 5 → total_spent = 25 →
  ∃ (rice_cost_per_kg : ℕ), rice_cost_per_kg = 2 ∧ rice_kg * rice_cost_per_kg + meat_lb * meat_cost_per_lb = total_spent :=
by sorry

end NUMINAMATH_CALUDE_rice_cost_l1940_194084


namespace NUMINAMATH_CALUDE_hare_wins_l1940_194018

/-- Race parameters --/
def race_duration : ℕ := 60
def hare_speed : ℕ := 10
def hare_run_time : ℕ := 30
def hare_nap_time : ℕ := 30
def tortoise_delay : ℕ := 10
def tortoise_speed : ℕ := 4

/-- Calculate distance covered by the hare --/
def hare_distance : ℕ := hare_speed * hare_run_time

/-- Calculate distance covered by the tortoise --/
def tortoise_distance : ℕ := tortoise_speed * (race_duration - tortoise_delay)

/-- Theorem stating that the hare wins the race --/
theorem hare_wins : hare_distance > tortoise_distance := by
  sorry

end NUMINAMATH_CALUDE_hare_wins_l1940_194018


namespace NUMINAMATH_CALUDE_fair_haired_women_percentage_l1940_194041

theorem fair_haired_women_percentage
  (total_employees : ℝ)
  (women_fair_hair_percentage : ℝ)
  (fair_hair_percentage : ℝ)
  (h1 : women_fair_hair_percentage = 28)
  (h2 : fair_hair_percentage = 70) :
  (women_fair_hair_percentage / fair_hair_percentage) * 100 = 40 := by
sorry

end NUMINAMATH_CALUDE_fair_haired_women_percentage_l1940_194041


namespace NUMINAMATH_CALUDE_sara_pumpkins_l1940_194045

/-- The number of pumpkins eaten by rabbits -/
def pumpkins_eaten (initial : ℕ) (remaining : ℕ) : ℕ :=
  initial - remaining

theorem sara_pumpkins : pumpkins_eaten 43 20 = 23 := by
  sorry

end NUMINAMATH_CALUDE_sara_pumpkins_l1940_194045


namespace NUMINAMATH_CALUDE_dozen_pens_cost_is_600_l1940_194074

/-- The cost of a pen in rupees -/
def pen_cost : ℚ := sorry

/-- The cost of a pencil in rupees -/
def pencil_cost : ℚ := sorry

/-- The cost ratio of a pen to a pencil -/
def cost_ratio : ℚ := 5

/-- The total cost of 3 pens and 5 pencils in rupees -/
def total_cost : ℚ := 200

/-- The cost of one dozen pens in rupees -/
def dozen_pens_cost : ℚ := 12 * pen_cost

theorem dozen_pens_cost_is_600 :
  pen_cost = 5 * pencil_cost ∧
  3 * pen_cost + 5 * pencil_cost = total_cost →
  dozen_pens_cost = 600 := by
  sorry

end NUMINAMATH_CALUDE_dozen_pens_cost_is_600_l1940_194074


namespace NUMINAMATH_CALUDE_parabola_focus_distance_l1940_194061

-- Define the parabola
def parabola (x y : ℝ) : Prop := y^2 = -32*x

-- Define the focus of the parabola
def focus : ℝ × ℝ := (-8, 0)

-- Define a point on the parabola
def point_on_parabola (x₀ : ℝ) : ℝ × ℝ := (x₀, 4)

-- State the theorem
theorem parabola_focus_distance (x₀ : ℝ) :
  parabola x₀ 4 →
  let P := point_on_parabola x₀
  let F := focus
  dist P F = 17/2 := by sorry

end NUMINAMATH_CALUDE_parabola_focus_distance_l1940_194061


namespace NUMINAMATH_CALUDE_problem_statement_l1940_194011

theorem problem_statement (x : ℝ) (h : x = Real.sqrt 2) : 
  (x + 2)^2 - 4*x*(x + 1) = -2 := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l1940_194011


namespace NUMINAMATH_CALUDE_tony_puzzle_solution_l1940_194036

/-- The number of puzzles Tony solved after the warm-up puzzle -/
def puzzles_after_warmup : ℕ := 2

/-- The time taken for the warm-up puzzle in minutes -/
def warmup_time : ℕ := 10

/-- The total time Tony spent solving puzzles in minutes -/
def total_time : ℕ := 70

/-- Each puzzle after the warm-up takes this many times longer than the warm-up -/
def puzzle_time_multiplier : ℕ := 3

theorem tony_puzzle_solution :
  warmup_time + puzzles_after_warmup * (puzzle_time_multiplier * warmup_time) = total_time :=
by sorry

end NUMINAMATH_CALUDE_tony_puzzle_solution_l1940_194036


namespace NUMINAMATH_CALUDE_tank_volume_l1940_194059

-- Define the rates of the pipes
def inlet_rate : ℝ := 3
def outlet_rate_1 : ℝ := 9
def outlet_rate_2 : ℝ := 6

-- Define the time it takes to empty the tank
def emptying_time : ℝ := 4320

-- Define the conversion factor from cubic inches to cubic feet
def cubic_inches_per_cubic_foot : ℝ := 1728

-- State the theorem
theorem tank_volume (net_rate : ℝ) (volume_cubic_inches : ℝ) (volume_cubic_feet : ℝ) 
  (h1 : net_rate = outlet_rate_1 + outlet_rate_2 - inlet_rate)
  (h2 : volume_cubic_inches = net_rate * emptying_time)
  (h3 : volume_cubic_feet = volume_cubic_inches / cubic_inches_per_cubic_foot) :
  volume_cubic_feet = 30 := by
  sorry

end NUMINAMATH_CALUDE_tank_volume_l1940_194059


namespace NUMINAMATH_CALUDE_stratified_sampling_possible_after_adjustment_l1940_194033

/-- Represents the population sizes of different age groups -/
structure Population where
  elderly : Nat
  middleAged : Nat
  young : Nat

/-- Represents the sampling parameters -/
structure SamplingParams where
  population : Population
  sampleSize : Nat

/-- Checks if stratified sampling is possible with equal sampling fractions -/
def canStratifySample (p : SamplingParams) : Prop :=
  ∃ (k : Nat), k > 0 ∧
    k ∣ p.sampleSize ∧
    k ∣ p.population.elderly ∧
    k ∣ p.population.middleAged ∧
    k ∣ p.population.young

/-- The given population and sample size -/
def givenParams : SamplingParams :=
  { population := { elderly := 28, middleAged := 54, young := 81 },
    sampleSize := 36 }

/-- The adjusted parameters after removing one elderly person -/
def adjustedParams : SamplingParams :=
  { population := { elderly := 27, middleAged := 54, young := 81 },
    sampleSize := 36 }

/-- Theorem stating that stratified sampling becomes possible after adjustment -/
theorem stratified_sampling_possible_after_adjustment :
  ¬canStratifySample givenParams ∧ canStratifySample adjustedParams :=
sorry


end NUMINAMATH_CALUDE_stratified_sampling_possible_after_adjustment_l1940_194033


namespace NUMINAMATH_CALUDE_second_smallest_odd_is_three_l1940_194008

def is_odd (n : ℕ) : Prop := ∃ k, n = 2*k + 1

def in_range (n : ℕ) : Prop := 1 ≤ n ∧ n ≤ 10

def second_smallest_odd : ℕ → Prop
| 3 => ∃ (x : ℕ), (is_odd x ∧ in_range x ∧ x < 3) ∧
                  ∀ (y : ℕ), (is_odd y ∧ in_range y ∧ y ≠ x ∧ y ≠ 3) → y > 3
| _ => False

theorem second_smallest_odd_is_three : second_smallest_odd 3 := by
  sorry

end NUMINAMATH_CALUDE_second_smallest_odd_is_three_l1940_194008


namespace NUMINAMATH_CALUDE_divisibility_condition_l1940_194039

theorem divisibility_condition (m n : ℕ+) :
  (∃ k : ℤ, 4 * (m.val * n.val + 1) = k * (m.val + n.val)^2) ↔ m = n :=
sorry

end NUMINAMATH_CALUDE_divisibility_condition_l1940_194039


namespace NUMINAMATH_CALUDE_moving_circle_trajectory_l1940_194067

/-- The trajectory equation of the center of a moving circle that is tangent to the x-axis
    and internally tangent to the semicircle x^2 + y^2 = 4 (0 ≤ y ≤ 2) -/
theorem moving_circle_trajectory (x y : ℝ) : 
  (0 < y ∧ y ≤ 1) →
  (∃ (r : ℝ), r > 0 ∧ 
    (∀ (x' y' : ℝ), x'^2 + y'^2 = 4 ∧ 0 ≤ y' ∧ y' ≤ 2 → 
      (x - x')^2 + (y - y')^2 = (2 - r)^2) ∧
    y = r) →
  x^2 = -4*(y - 1) := by
  sorry

end NUMINAMATH_CALUDE_moving_circle_trajectory_l1940_194067


namespace NUMINAMATH_CALUDE_min_value_of_f_l1940_194042

def f (x a : ℝ) : ℝ := 2 * x^2 - 4 * a * x + a^2 + 2 * a + 2

theorem min_value_of_f (a : ℝ) :
  (∀ x, -1 ≤ x ∧ x ≤ 2 → f x a ≥ 2) ∧
  (∃ x, -1 ≤ x ∧ x ≤ 2 ∧ f x a = 2) →
  a = -3 - Real.sqrt 7 ∨ a = 0 ∨ a = 2 ∨ a = 4 :=
sorry

end NUMINAMATH_CALUDE_min_value_of_f_l1940_194042


namespace NUMINAMATH_CALUDE_chemistry_lab_workstations_l1940_194091

theorem chemistry_lab_workstations (total_capacity : ℕ) (total_workstations : ℕ) 
  (three_student_stations : ℕ) (remaining_stations : ℕ) 
  (h1 : total_capacity = 38)
  (h2 : total_workstations = 16)
  (h3 : three_student_stations = 6)
  (h4 : remaining_stations = 10)
  (h5 : total_workstations = three_student_stations + remaining_stations) :
  ∃ (students_per_remaining : ℕ),
    students_per_remaining * remaining_stations + 3 * three_student_stations = total_capacity ∧
    students_per_remaining * remaining_stations = 20 :=
by sorry

end NUMINAMATH_CALUDE_chemistry_lab_workstations_l1940_194091


namespace NUMINAMATH_CALUDE_f_ln_2_equals_3_l1940_194047

-- Define a monotonically increasing function
def MonoIncreasing (f : ℝ → ℝ) : Prop :=
  ∀ x y, x < y → f x < f y

-- State the theorem
theorem f_ln_2_equals_3 
  (f : ℝ → ℝ)
  (h_mono : MonoIncreasing f)
  (h_prop : ∀ x : ℝ, f (f x - Real.exp x) = Real.exp 1 + 1) : 
  f (Real.log 2) = 3 := by
sorry


end NUMINAMATH_CALUDE_f_ln_2_equals_3_l1940_194047


namespace NUMINAMATH_CALUDE_complex_fraction_equality_l1940_194099

/-- Given that i is the imaginary unit, prove that (1+3i)/(1+i) = 2+i -/
theorem complex_fraction_equality : (1 + 3 * I) / (1 + I) = 2 + I := by
  sorry

end NUMINAMATH_CALUDE_complex_fraction_equality_l1940_194099


namespace NUMINAMATH_CALUDE_map_distance_conversion_l1940_194013

/-- Given a map scale where 312 inches represents 136 km,
    prove that 34 inches on the map corresponds to approximately 14.82 km in actual distance. -/
theorem map_distance_conversion (map_distance : ℝ) (actual_distance : ℝ) (ram_map_distance : ℝ)
  (h1 : map_distance = 312)
  (h2 : actual_distance = 136)
  (h3 : ram_map_distance = 34) :
  ∃ (ε : ℝ), ε > 0 ∧ abs ((actual_distance / map_distance) * ram_map_distance - 14.82) < ε :=
sorry

end NUMINAMATH_CALUDE_map_distance_conversion_l1940_194013


namespace NUMINAMATH_CALUDE_min_value_xy_expression_min_value_achievable_l1940_194065

theorem min_value_xy_expression (x y : ℝ) : (x * y + 1)^2 + (x - y)^2 ≥ 1 := by sorry

theorem min_value_achievable : ∃ x y : ℝ, (x * y + 1)^2 + (x - y)^2 = 1 := by sorry

end NUMINAMATH_CALUDE_min_value_xy_expression_min_value_achievable_l1940_194065


namespace NUMINAMATH_CALUDE_chess_tournament_success_ratio_l1940_194031

theorem chess_tournament_success_ratio (charlie_day1_score charlie_day1_attempted charlie_day2_score charlie_day2_attempted : ℕ) : 
  -- Total points for both players
  charlie_day1_attempted + charlie_day2_attempted = 600 →
  -- Charlie's scores are positive integers
  charlie_day1_score > 0 →
  charlie_day2_score > 0 →
  -- Charlie's daily success ratios are less than Alpha's
  charlie_day1_score * 360 < 180 * charlie_day1_attempted →
  charlie_day2_score * 240 < 120 * charlie_day2_attempted →
  -- Charlie did not attempt 360 points on day 1
  charlie_day1_attempted ≠ 360 →
  -- The maximum two-day success ratio for Charlie
  (charlie_day1_score + charlie_day2_score : ℚ) / 600 ≤ 299 / 600 :=
by sorry

end NUMINAMATH_CALUDE_chess_tournament_success_ratio_l1940_194031


namespace NUMINAMATH_CALUDE_compare_expressions_l1940_194002

theorem compare_expressions (a : ℝ) : (a + 3) * (a - 5) < (a + 2) * (a - 4) := by
  sorry

end NUMINAMATH_CALUDE_compare_expressions_l1940_194002


namespace NUMINAMATH_CALUDE_min_plus_arg_is_pi_third_l1940_194049

noncomputable def f (x : ℝ) : ℝ := 9 / (8 * Real.cos (2 * x) + 16) - Real.sin x ^ 2

def has_min (f : ℝ → ℝ) (m : ℝ) : Prop :=
  ∀ x, f x ≥ m

def is_smallest_positive_min (f : ℝ → ℝ) (m n : ℝ) : Prop :=
  has_min f m ∧ f n = m ∧ n > 0 ∧ ∀ x, 0 < x ∧ x < n → f x > m

theorem min_plus_arg_is_pi_third :
  ∃ (m n : ℝ), is_smallest_positive_min f m n ∧ m + n = Real.pi / 3 :=
sorry

end NUMINAMATH_CALUDE_min_plus_arg_is_pi_third_l1940_194049


namespace NUMINAMATH_CALUDE_multiples_of_four_l1940_194014

/-- Given a natural number n, if there are exactly 25 multiples of 4
    between n and 108 (inclusive), then n = 12. -/
theorem multiples_of_four (n : ℕ) : 
  (∃ (l : List ℕ), l.length = 25 ∧ 
    (∀ x ∈ l, x % 4 = 0 ∧ n ≤ x ∧ x ≤ 108) ∧
    (∀ y, n ≤ y ∧ y ≤ 108 ∧ y % 4 = 0 → y ∈ l)) →
  n = 12 := by
  sorry

end NUMINAMATH_CALUDE_multiples_of_four_l1940_194014


namespace NUMINAMATH_CALUDE_miles_reading_pages_l1940_194068

/-- Calculates the total number of pages read by Miles --/
def total_pages_read (hours_in_day : ℝ) (reading_fraction : ℝ) 
  (novel_pages_per_hour : ℝ) (graphic_novel_pages_per_hour : ℝ) (comic_book_pages_per_hour : ℝ) 
  (fraction_per_book_type : ℝ) : ℝ :=
  let total_reading_hours := hours_in_day * reading_fraction
  let hours_per_book_type := total_reading_hours * fraction_per_book_type
  let novel_pages := novel_pages_per_hour * hours_per_book_type
  let graphic_novel_pages := graphic_novel_pages_per_hour * hours_per_book_type
  let comic_book_pages := comic_book_pages_per_hour * hours_per_book_type
  novel_pages + graphic_novel_pages + comic_book_pages

/-- Theorem stating that Miles reads 128 pages given the problem conditions --/
theorem miles_reading_pages : 
  total_pages_read 24 (1/6) 21 30 45 (1/3) = 128 := by
  sorry

end NUMINAMATH_CALUDE_miles_reading_pages_l1940_194068


namespace NUMINAMATH_CALUDE_tom_dancing_hours_l1940_194053

/-- Calculates the total dancing hours over multiple years -/
def total_dancing_hours (sessions_per_week : ℕ) (hours_per_session : ℕ) (years : ℕ) (weeks_per_year : ℕ) : ℕ :=
  sessions_per_week * hours_per_session * years * weeks_per_year

/-- Proves that Tom's total dancing hours over 10 years is 4160 -/
theorem tom_dancing_hours : 
  total_dancing_hours 4 2 10 52 = 4160 := by
  sorry

#eval total_dancing_hours 4 2 10 52

end NUMINAMATH_CALUDE_tom_dancing_hours_l1940_194053


namespace NUMINAMATH_CALUDE_sum_of_cubes_l1940_194020

theorem sum_of_cubes (x y : ℝ) (h1 : x + y = 11) (h2 : x * y = 12) : x^3 + y^3 = 935 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_cubes_l1940_194020


namespace NUMINAMATH_CALUDE_pentagon_extension_l1940_194010

/-- Given a pentagon ABCDE with extended sides, prove the relation between A and A', B', C', D', E' -/
theorem pentagon_extension (A B C D E A' B' C' D' E' : ℝ × ℝ) 
  (h1 : B = (A + A') / 2)
  (h2 : C = (B + B') / 2)
  (h3 : D = (C + C') / 2)
  (h4 : E = (D + D') / 2)
  (h5 : A = (E + E') / 2) :
  A = (1/32 : ℝ) • A' + (1/16 : ℝ) • B' + (1/8 : ℝ) • C' + (1/4 : ℝ) • D' + (1/2 : ℝ) • E' :=
by sorry

end NUMINAMATH_CALUDE_pentagon_extension_l1940_194010


namespace NUMINAMATH_CALUDE_scooter_depreciation_l1940_194005

theorem scooter_depreciation (initial_value : ℝ) : 
  (initial_value * (3/4)^5 = 9492.1875) → initial_value = 40000 := by
  sorry

end NUMINAMATH_CALUDE_scooter_depreciation_l1940_194005


namespace NUMINAMATH_CALUDE_function_form_l1940_194090

def StrictlyIncreasing (f : ℕ → ℕ) : Prop :=
  ∀ x y, x < y → f x < f y

theorem function_form (f : ℕ → ℕ) 
  (h1 : StrictlyIncreasing f)
  (h2 : ∀ x y : ℕ, ∃ k : ℕ+, (f x + f y) / (1 + f (x + y)) = k) :
  ∃ a : ℕ+, ∀ x : ℕ, f x = a * x + 1 :=
sorry

end NUMINAMATH_CALUDE_function_form_l1940_194090


namespace NUMINAMATH_CALUDE_E_80_l1940_194073

/-- E(n) represents the number of ways to express n as a product of integers greater than 1, where order matters -/
def E (n : ℕ) : ℕ := sorry

/-- The prime factorization of 80 is 2^4 * 5 -/
axiom prime_factorization_80 : 80 = 2^4 * 5

/-- Theorem: The number of ways to express 80 as a product of integers greater than 1, where order matters, is 42 -/
theorem E_80 : E 80 = 42 := by sorry

end NUMINAMATH_CALUDE_E_80_l1940_194073


namespace NUMINAMATH_CALUDE_x_value_l1940_194072

theorem x_value : ∃ x : ℝ, (0.25 * x = 0.15 * 1600 - 15) ∧ (x = 900) := by
  sorry

end NUMINAMATH_CALUDE_x_value_l1940_194072


namespace NUMINAMATH_CALUDE_no_special_arrangement_exists_l1940_194022

theorem no_special_arrangement_exists : ¬ ∃ (p : Fin 20 → Fin 20), Function.Bijective p ∧
  ∀ (i j : Fin 20), i.val % 10 = j.val % 10 → i ≠ j →
    |p i - p j| - 1 = i.val % 10 := by
  sorry

end NUMINAMATH_CALUDE_no_special_arrangement_exists_l1940_194022


namespace NUMINAMATH_CALUDE_tobias_driveways_shoveled_l1940_194026

/-- Calculates the number of driveways Tobias shoveled given his earnings and expenses. -/
theorem tobias_driveways_shoveled (shoe_cost allowance_per_month lawn_mowing_charge driveway_shoveling_charge change_after_purchase : ℕ) (months_saved lawns_mowed : ℕ) : 
  shoe_cost = 95 →
  allowance_per_month = 5 →
  months_saved = 3 →
  lawn_mowing_charge = 15 →
  driveway_shoveling_charge = 7 →
  change_after_purchase = 15 →
  lawns_mowed = 4 →
  (shoe_cost + change_after_purchase - months_saved * allowance_per_month - lawns_mowed * lawn_mowing_charge) / driveway_shoveling_charge = 5 := by
sorry

end NUMINAMATH_CALUDE_tobias_driveways_shoveled_l1940_194026


namespace NUMINAMATH_CALUDE_positive_polynomial_fraction_representation_l1940_194098

/-- A polynomial with real coefficients -/
def RealPolynomial := Polynomial ℝ

/-- A polynomial with non-negative coefficients -/
def NonNegativePolynomial (p : RealPolynomial) : Prop :=
  ∀ i, (p.coeff i) ≥ 0

/-- The theorem statement -/
theorem positive_polynomial_fraction_representation
  (P : RealPolynomial) (h : ∀ x : ℝ, x > 0 → P.eval x > 0) :
  ∃ (Q R : RealPolynomial), NonNegativePolynomial Q ∧ NonNegativePolynomial R ∧
    ∀ x : ℝ, x ≠ 0 → P.eval x = (Q.eval x) / (R.eval x) := by
  sorry

end NUMINAMATH_CALUDE_positive_polynomial_fraction_representation_l1940_194098


namespace NUMINAMATH_CALUDE_pages_copied_for_fifty_dollars_l1940_194024

/-- Given that 4 pages can be copied for 10 cents, prove that $50 allows for copying 2000 pages. -/
theorem pages_copied_for_fifty_dollars (cost_per_four_pages : ℚ) (pages_per_fifty_dollars : ℕ) :
  cost_per_four_pages = 10 / 100 →
  pages_per_fifty_dollars = 2000 :=
by sorry

end NUMINAMATH_CALUDE_pages_copied_for_fifty_dollars_l1940_194024


namespace NUMINAMATH_CALUDE_geese_percentage_among_non_swans_l1940_194070

theorem geese_percentage_among_non_swans 
  (total_percentage : ℝ) 
  (geese_percentage : ℝ) 
  (swan_percentage : ℝ) 
  (h1 : total_percentage = 100) 
  (h2 : geese_percentage = 20) 
  (h3 : swan_percentage = 25) : 
  (geese_percentage / (total_percentage - swan_percentage)) * 100 = 26.67 := by
sorry

end NUMINAMATH_CALUDE_geese_percentage_among_non_swans_l1940_194070


namespace NUMINAMATH_CALUDE_abs_is_even_l1940_194064

def is_even_function (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f x = f (-x)

def abs_function (x : ℝ) : ℝ := |x|

theorem abs_is_even : is_even_function abs_function := by
  sorry

end NUMINAMATH_CALUDE_abs_is_even_l1940_194064


namespace NUMINAMATH_CALUDE_linear_function_not_in_first_quadrant_l1940_194085

/-- A linear function y = mx + b, where m is the slope and b is the y-intercept -/
structure LinearFunction where
  m : ℝ
  b : ℝ

/-- The first quadrant of the Cartesian plane -/
def FirstQuadrant : Set (ℝ × ℝ) :=
  {p | p.1 > 0 ∧ p.2 > 0}

/-- Theorem: The linear function y = -2x - 3 does not pass through the first quadrant -/
theorem linear_function_not_in_first_quadrant :
  let f : LinearFunction := ⟨-2, -3⟩
  ∀ x y : ℝ, y = f.m * x + f.b → (x, y) ∉ FirstQuadrant :=
by
  sorry


end NUMINAMATH_CALUDE_linear_function_not_in_first_quadrant_l1940_194085


namespace NUMINAMATH_CALUDE_max_prime_factors_b_l1940_194007

theorem max_prime_factors_b (a b : ℕ+) 
  (h_gcd : (Nat.gcd a b).factors.length = 5)
  (h_lcm : (Nat.lcm a b).factors.length = 20)
  (h_fewer : (b.val.factors.length : ℕ) < a.val.factors.length) :
  b.val.factors.length ≤ 12 := by
  sorry

end NUMINAMATH_CALUDE_max_prime_factors_b_l1940_194007


namespace NUMINAMATH_CALUDE_square_of_101_l1940_194030

theorem square_of_101 : 101 * 101 = 10201 := by
  sorry

end NUMINAMATH_CALUDE_square_of_101_l1940_194030


namespace NUMINAMATH_CALUDE_floor_fraction_equals_eight_l1940_194051

theorem floor_fraction_equals_eight (n : ℕ) (h : n = 2006) : 
  ⌊(8 * (n^2 + 1 : ℝ)) / (n^2 - 1 : ℝ)⌋ = 8 := by
  sorry

end NUMINAMATH_CALUDE_floor_fraction_equals_eight_l1940_194051


namespace NUMINAMATH_CALUDE_ellipse_point_Q_l1940_194017

-- Define the ellipse
def ellipse (x y : ℝ) : Prop := x^2 / 4 + y^2 / 2 = 1

-- Define points A and B
def A : ℝ × ℝ := (-2, 0)
def B : ℝ × ℝ := (2, 0)

-- Define the condition for point M
def M_condition (M : ℝ × ℝ) : Prop :=
  let (mx, my) := M
  (mx - 2) * (mx + 2) + my^2 = 0  -- MB ⊥ AB

-- Define the condition for point P
def P_condition (P : ℝ × ℝ) : Prop :=
  let (px, py) := P
  ellipse px py  -- P is on the ellipse

-- Define the condition for point Q
def Q_condition (Q : ℝ × ℝ) : Prop :=
  let (qx, qy) := Q
  qy = 0 ∧ qx ≠ -2 ∧ qx ≠ 2  -- Q is on x-axis and distinct from A and B

-- Define the circle condition
def circle_condition (M P Q : ℝ × ℝ) : Prop :=
  let (mx, my) := M
  let (px, py) := P
  let (qx, qy) := Q
  ∃ (I : ℝ × ℝ), 
    (I.1 - px) * (mx - px) + (I.2 - py) * (my - py) = 0 ∧  -- I is on BP
    (I.1 - mx) * (qx - mx) + (I.2 - my) * (qy - my) = 0 ∧  -- I is on MQ
    (I.1 - (mx + px) / 2)^2 + (I.2 - (my + py) / 2)^2 = ((mx - px)^2 + (my - py)^2) / 4  -- I is on the circle

theorem ellipse_point_Q : 
  ∀ (M P Q : ℝ × ℝ),
    M_condition M →
    P_condition P →
    Q_condition Q →
    circle_condition M P Q →
    Q = (0, 0) := by sorry

end NUMINAMATH_CALUDE_ellipse_point_Q_l1940_194017


namespace NUMINAMATH_CALUDE_father_twice_son_age_l1940_194077

/-- Represents the ages of a father and son --/
structure Ages where
  sonPast : ℕ
  fatherPast : ℕ
  sonNow : ℕ
  fatherNow : ℕ

/-- The conditions of the problem --/
def ageConditions (a : Ages) : Prop :=
  a.fatherPast = 3 * a.sonPast ∧
  a.sonNow = a.sonPast + 18 ∧
  a.fatherNow = a.fatherPast + 18 ∧
  a.sonNow + a.fatherNow = 108 ∧
  ∃ k : ℕ, a.fatherNow = k * a.sonNow

/-- The theorem to be proved --/
theorem father_twice_son_age (a : Ages) (h : ageConditions a) : a.fatherNow = 2 * a.sonNow := by
  sorry

end NUMINAMATH_CALUDE_father_twice_son_age_l1940_194077


namespace NUMINAMATH_CALUDE_dodecagon_diagonal_intersection_probability_l1940_194096

/-- A regular dodecagon is a 12-sided polygon with all sides equal and all angles equal. -/
def RegularDodecagon : Type := Unit

/-- A diagonal of a regular dodecagon is a line segment connecting two non-adjacent vertices. -/
def Diagonal (d : RegularDodecagon) : Type := Unit

/-- The probability that two randomly chosen diagonals of a regular dodecagon intersect inside the polygon. -/
def intersectionProbability (d : RegularDodecagon) : ℚ :=
  165 / 287

/-- Theorem: The probability that the intersection of two randomly chosen diagonals 
    of a regular dodecagon lies inside the polygon is 165/287. -/
theorem dodecagon_diagonal_intersection_probability (d : RegularDodecagon) :
  intersectionProbability d = 165 / 287 :=
by
  sorry

end NUMINAMATH_CALUDE_dodecagon_diagonal_intersection_probability_l1940_194096


namespace NUMINAMATH_CALUDE_closest_integer_to_cube_root_1728_l1940_194092

theorem closest_integer_to_cube_root_1728 : 
  ∃ n : ℤ, ∀ m : ℤ, |n - (1728 : ℝ)^(1/3)| ≤ |m - (1728 : ℝ)^(1/3)| ∧ n = 12 :=
by
  sorry

end NUMINAMATH_CALUDE_closest_integer_to_cube_root_1728_l1940_194092


namespace NUMINAMATH_CALUDE_sum_of_f_values_negative_l1940_194043

def is_monotonically_decreasing (f : ℝ → ℝ) : Prop :=
  ∀ x y, x < y → f x > f y

def is_odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

theorem sum_of_f_values_negative
  (f : ℝ → ℝ)
  (h_decreasing : is_monotonically_decreasing f)
  (h_odd : is_odd_function f)
  (x₁ x₂ x₃ : ℝ)
  (h₁₂ : x₁ + x₂ > 0)
  (h₂₃ : x₂ + x₃ > 0)
  (h₃₁ : x₃ + x₁ > 0) :
  f x₁ + f x₂ + f x₃ < 0 :=
sorry

end NUMINAMATH_CALUDE_sum_of_f_values_negative_l1940_194043


namespace NUMINAMATH_CALUDE_adult_ticket_cost_l1940_194006

theorem adult_ticket_cost (num_children num_adults : ℕ) (child_ticket_cost total_cost : ℚ) 
  (h1 : num_children = 6)
  (h2 : num_adults = 10)
  (h3 : child_ticket_cost = 10)
  (h4 : total_cost = 220)
  : (total_cost - num_children * child_ticket_cost) / num_adults = 16 := by
  sorry

end NUMINAMATH_CALUDE_adult_ticket_cost_l1940_194006


namespace NUMINAMATH_CALUDE_min_value_a1_plus_a7_l1940_194069

/-- A positive geometric sequence -/
def is_positive_geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, r > 0 ∧ ∀ n : ℕ, a n > 0 ∧ a (n + 1) = r * a n

/-- The theorem stating the minimum value of a₁ + a₇ in a positive geometric sequence where a₃ * a₅ = 64 -/
theorem min_value_a1_plus_a7 (a : ℕ → ℝ) 
    (h_geom : is_positive_geometric_sequence a) 
    (h_prod : a 3 * a 5 = 64) : 
  (∀ b : ℕ → ℝ, is_positive_geometric_sequence b → b 3 * b 5 = 64 → a 1 + a 7 ≤ b 1 + b 7) → 
  a 1 + a 7 = 16 := by
sorry

end NUMINAMATH_CALUDE_min_value_a1_plus_a7_l1940_194069


namespace NUMINAMATH_CALUDE_playground_students_l1940_194029

/-- The number of students initially on the playground -/
def initial_students : ℕ := 32

/-- The number of students who left the playground -/
def students_left : ℕ := 16

/-- The number of new students who came to the playground -/
def new_students : ℕ := 9

/-- The final number of students on the playground -/
def final_students : ℕ := 25

theorem playground_students :
  initial_students - students_left + new_students = final_students :=
by sorry

end NUMINAMATH_CALUDE_playground_students_l1940_194029


namespace NUMINAMATH_CALUDE_johns_raise_l1940_194025

/-- Proves that if an amount x is increased by 9.090909090909092% to reach $60, then x is equal to $55 -/
theorem johns_raise (x : ℝ) : 
  x * (1 + 0.09090909090909092) = 60 → x = 55 := by
  sorry

end NUMINAMATH_CALUDE_johns_raise_l1940_194025


namespace NUMINAMATH_CALUDE_diophantine_equation_solutions_l1940_194044

theorem diophantine_equation_solutions :
  ∀ a b c d : ℕ, 2^a * 3^b - 5^c * 7^d = 1 ↔
    (a = 1 ∧ b = 0 ∧ c = 0 ∧ d = 0) ∨
    (a = 3 ∧ b = 0 ∧ c = 0 ∧ d = 1) ∨
    (a = 1 ∧ b = 1 ∧ c = 1 ∧ d = 0) ∨
    (a = 2 ∧ b = 2 ∧ c = 1 ∧ d = 1) :=
by sorry


end NUMINAMATH_CALUDE_diophantine_equation_solutions_l1940_194044


namespace NUMINAMATH_CALUDE_six_bottle_caps_cost_l1940_194088

/-- The cost of a given number of bottle caps -/
def bottle_cap_cost (num_caps : ℕ) (cost_per_cap : ℕ) : ℕ :=
  num_caps * cost_per_cap

/-- Theorem: The cost of 6 bottle caps at $2 each is $12 -/
theorem six_bottle_caps_cost : bottle_cap_cost 6 2 = 12 := by
  sorry

end NUMINAMATH_CALUDE_six_bottle_caps_cost_l1940_194088


namespace NUMINAMATH_CALUDE_l_triomino_division_l1940_194056

/-- An L-triomino is a shape with 3 squares formed by removing one square from a 2x2 grid. -/
def L_triomino_area : ℕ := 3

/-- Theorem: A 1961 × 1963 grid rectangle cannot be exactly divided into L-triominoes,
    but a 1963 × 1965 rectangle can be exactly divided into L-triominoes. -/
theorem l_triomino_division :
  (¬ (1961 * 1963) % L_triomino_area = 0) ∧
  ((1963 * 1965) % L_triomino_area = 0) := by
  sorry

end NUMINAMATH_CALUDE_l_triomino_division_l1940_194056


namespace NUMINAMATH_CALUDE_fraction_sum_inequality_l1940_194009

theorem fraction_sum_inequality (a b c : ℝ) 
  (ha : 0 ≤ a ∧ a ≤ 1) 
  (hb : 0 ≤ b ∧ b ≤ 1) 
  (hc : 0 ≤ c ∧ c ≤ 1) : 
  (a / (b*c + 1)) + (b / (a*c + 1)) + (c / (a*b + 1)) ≤ 2 := by
  sorry

end NUMINAMATH_CALUDE_fraction_sum_inequality_l1940_194009


namespace NUMINAMATH_CALUDE_trig_identity_l1940_194081

theorem trig_identity (α : Real) (h : Real.tan α = 2) :
  Real.sin α ^ 2 + 2 * Real.sin α * Real.cos α + 3 * Real.cos α ^ 2 = 11 / 5 := by
  sorry

end NUMINAMATH_CALUDE_trig_identity_l1940_194081


namespace NUMINAMATH_CALUDE_cherry_weekly_earnings_l1940_194037

/-- Represents the delivery rates for different weight ranges -/
structure DeliveryRates :=
  (kg3to5 : ℝ)
  (kg6to8 : ℝ)
  (kg9to12 : ℝ)
  (kg13to15 : ℝ)

/-- Represents the daily deliveries -/
structure DailyDeliveries :=
  (kg5 : ℕ)
  (kg8 : ℕ)
  (kg10 : ℕ)
  (kg14 : ℕ)

def weekdayRates : DeliveryRates :=
  { kg3to5 := 2.5, kg6to8 := 4, kg9to12 := 6, kg13to15 := 8 }

def weekendRates : DeliveryRates :=
  { kg3to5 := 3, kg6to8 := 5, kg9to12 := 7.5, kg13to15 := 10 }

def weekdayDeliveries : DailyDeliveries :=
  { kg5 := 4, kg8 := 2, kg10 := 3, kg14 := 1 }

def weekendDeliveries : DailyDeliveries :=
  { kg5 := 2, kg8 := 3, kg10 := 0, kg14 := 2 }

def weekdaysInWeek : ℕ := 5
def weekendDaysInWeek : ℕ := 2

/-- Calculates the daily earnings based on rates and deliveries -/
def dailyEarnings (rates : DeliveryRates) (deliveries : DailyDeliveries) : ℝ :=
  rates.kg3to5 * deliveries.kg5 +
  rates.kg6to8 * deliveries.kg8 +
  rates.kg9to12 * deliveries.kg10 +
  rates.kg13to15 * deliveries.kg14

/-- Calculates the total weekly earnings -/
def weeklyEarnings : ℝ :=
  weekdaysInWeek * dailyEarnings weekdayRates weekdayDeliveries +
  weekendDaysInWeek * dailyEarnings weekendRates weekendDeliveries

theorem cherry_weekly_earnings :
  weeklyEarnings = 302 := by sorry

end NUMINAMATH_CALUDE_cherry_weekly_earnings_l1940_194037


namespace NUMINAMATH_CALUDE_divisor_problem_l1940_194075

theorem divisor_problem (n d k m : ℤ) : 
  n = k * d + 4 → 
  n + 15 = 5 * m + 4 → 
  d = 5 := by sorry

end NUMINAMATH_CALUDE_divisor_problem_l1940_194075


namespace NUMINAMATH_CALUDE_cos_sum_diff_product_leq_cos_sq_l1940_194095

theorem cos_sum_diff_product_leq_cos_sq (x y : ℝ) :
  Real.cos (x + y) * Real.cos (x - y) ≤ Real.cos x ^ 2 := by
  sorry

end NUMINAMATH_CALUDE_cos_sum_diff_product_leq_cos_sq_l1940_194095


namespace NUMINAMATH_CALUDE_evaluate_expression_l1940_194089

theorem evaluate_expression : 6 - 5 * (9 - 2^3) * 3 = -9 := by
  sorry

end NUMINAMATH_CALUDE_evaluate_expression_l1940_194089


namespace NUMINAMATH_CALUDE_vector_properties_l1940_194063

/-- Given two vectors in ℝ², prove dot product and parallelism properties -/
theorem vector_properties (a b : ℝ × ℝ) (h1 : a = (1, -2)) (h2 : b = (-3, 2)) :
  (a + b) • (a - b) = -8 ∧
  ∃ k : ℝ, k = (1 : ℝ) / 3 ∧ ∃ c : ℝ, c ≠ 0 ∧ k • a + b = c • (a - 3 • b) := by
  sorry

end NUMINAMATH_CALUDE_vector_properties_l1940_194063
