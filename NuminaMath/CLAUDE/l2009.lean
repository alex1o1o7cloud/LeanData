import Mathlib

namespace divisibility_property_l2009_200989

theorem divisibility_property (a n p : ℕ) : 
  a ≥ 2 → 
  n ≥ 1 → 
  Nat.Prime p → 
  p ∣ (a^(2^n) + 1) → 
  2^(n+1) ∣ (p-1) := by
  sorry

end divisibility_property_l2009_200989


namespace parabola_focus_specific_parabola_focus_l2009_200966

/-- The focus of a parabola with equation y^2 = ax has coordinates (a/4, 0) -/
theorem parabola_focus (a : ℝ) :
  let parabola := {(x, y) : ℝ × ℝ | y^2 = a * x}
  let focus := (a / 4, 0)
  focus ∈ parabola ∧ 
  ∀ (p : ℝ × ℝ), p ∈ parabola → dist p focus = dist p (0, -a/4) :=
sorry

/-- The focus of the parabola y^2 = 8x has coordinates (2, 0) -/
theorem specific_parabola_focus :
  let parabola := {(x, y) : ℝ × ℝ | y^2 = 8 * x}
  let focus := (2, 0)
  focus ∈ parabola ∧ 
  ∀ (p : ℝ × ℝ), p ∈ parabola → dist p focus = dist p (0, -2) :=
sorry

end parabola_focus_specific_parabola_focus_l2009_200966


namespace village_population_equality_l2009_200983

/-- The initial population of Village X -/
def initial_population_X : ℕ := 72000

/-- The yearly decrease in population of Village X -/
def decrease_rate_X : ℕ := 1200

/-- The initial population of Village Y -/
def initial_population_Y : ℕ := 42000

/-- The yearly increase in population of Village Y -/
def increase_rate_Y : ℕ := 800

/-- The number of years after which the populations are equal -/
def years : ℕ := 15

theorem village_population_equality :
  initial_population_X - (decrease_rate_X * years) =
  initial_population_Y + (increase_rate_Y * years) :=
by sorry

end village_population_equality_l2009_200983


namespace female_officers_count_l2009_200969

theorem female_officers_count (total_on_duty : ℕ) (female_ratio_on_duty : ℚ) (female_percentage : ℚ) :
  total_on_duty = 100 →
  female_ratio_on_duty = 1/2 →
  female_percentage = 1/5 →
  (female_ratio_on_duty * total_on_duty : ℚ) / female_percentage = 250 := by
  sorry

end female_officers_count_l2009_200969


namespace tickets_sold_is_525_l2009_200931

/-- Represents the total number of tickets sold given ticket prices, total money collected, and number of general admission tickets. -/
def total_tickets_sold (student_price general_price total_collected general_tickets : ℕ) : ℕ :=
  let student_tickets := (total_collected - general_price * general_tickets) / student_price
  student_tickets + general_tickets

/-- Theorem stating that given the specific conditions, the total number of tickets sold is 525. -/
theorem tickets_sold_is_525 :
  total_tickets_sold 4 6 2876 388 = 525 := by
  sorry

end tickets_sold_is_525_l2009_200931


namespace permutation_combination_equality_l2009_200942

theorem permutation_combination_equality (n : ℕ) : 
  n * (n - 1) * (n - 2) = 6 * (n * (n - 1) * (n - 2) * (n - 3)) / 24 → n = 7 := by
  sorry

end permutation_combination_equality_l2009_200942


namespace synthetic_analytic_direct_l2009_200921

-- Define proof methods
structure ProofMethod where
  name : String
  direction : String
  isDirect : Bool

-- Define synthetic and analytic methods
def synthetic : ProofMethod := {
  name := "Synthetic",
  direction := "cause to effect",
  isDirect := true
}

def analytic : ProofMethod := {
  name := "Analytic",
  direction := "effect to cause",
  isDirect := true
}

-- Theorem statement
theorem synthetic_analytic_direct :
  synthetic.isDirect ∧ analytic.isDirect :=
sorry

end synthetic_analytic_direct_l2009_200921


namespace school_population_l2009_200963

theorem school_population (b g t a : ℕ) : 
  b = 4 * g ∧ 
  g = 8 * t ∧ 
  t = 2 * a → 
  b + g + t + a = 83 * a :=
by sorry

end school_population_l2009_200963


namespace journey_feasibility_l2009_200930

/-- Proves that a journey can be completed in the given time at the given average speed -/
theorem journey_feasibility 
  (total_distance : ℝ) 
  (segment1 : ℝ) 
  (segment2 : ℝ) 
  (total_time : ℝ) 
  (average_speed : ℝ) 
  (h1 : total_distance = segment1 + segment2)
  (h2 : total_distance = 693)
  (h3 : segment1 = 420)
  (h4 : segment2 = 273)
  (h5 : total_time = 11)
  (h6 : average_speed = 63)
  : total_distance / average_speed = total_time :=
by sorry

#check journey_feasibility

end journey_feasibility_l2009_200930


namespace product_of_fractions_l2009_200957

theorem product_of_fractions : (2 : ℚ) / 3 * (3 : ℚ) / 8 = (1 : ℚ) / 4 := by
  sorry

end product_of_fractions_l2009_200957


namespace derivative_not_in_second_quadrant_l2009_200907

-- Define the quadratic function
def f (x : ℝ) (b c : ℝ) : ℝ := x^2 + b*x + c

-- Define the derivative of f
def f' (x : ℝ) (b : ℝ) : ℝ := 2*x + b

-- Theorem statement
theorem derivative_not_in_second_quadrant (b c : ℝ) :
  (∀ x, f x b c = f (-x + 4) b c) →  -- axis of symmetry is x = 2
  ∀ x y, f' x b = y → ¬(x < 0 ∧ y > 0) :=
by sorry

end derivative_not_in_second_quadrant_l2009_200907


namespace bisection_interval_valid_l2009_200994

/-- The function f(x) = x^3 + 5 -/
def f (x : ℝ) : ℝ := x^3 + 5

/-- Theorem stating that [-2, 1] is a valid initial interval for the bisection method -/
theorem bisection_interval_valid :
  f (-2) * f 1 < 0 := by sorry

end bisection_interval_valid_l2009_200994


namespace initial_mask_sets_l2009_200985

/-- The number of mask sets Alicia gave away -/
def given_away : ℕ := 51

/-- The number of mask sets Alicia had left -/
def left : ℕ := 39

/-- The initial number of mask sets in Alicia's collection -/
def initial : ℕ := given_away + left

/-- Theorem stating that the initial number of mask sets is 90 -/
theorem initial_mask_sets : initial = 90 := by
  sorry

end initial_mask_sets_l2009_200985


namespace socks_difference_l2009_200970

/-- The number of pairs of socks Laticia knitted in the first week -/
def first_week : ℕ := 12

/-- The number of pairs of socks Laticia knitted in the second week -/
def second_week : ℕ := sorry

/-- The number of pairs of socks Laticia knitted in the third week -/
def third_week : ℕ := (first_week + second_week) / 2

/-- The number of pairs of socks Laticia knitted in the fourth week -/
def fourth_week : ℕ := third_week - 3

/-- The total number of pairs of socks Laticia knitted -/
def total_socks : ℕ := 57

theorem socks_difference : 
  first_week + second_week + third_week + fourth_week = total_socks ∧ 
  second_week - first_week = 1 := by sorry

end socks_difference_l2009_200970


namespace caitlin_sara_weight_l2009_200986

/-- Given the weights of three people (Annette, Caitlin, and Sara), proves that
    Caitlin and Sara weigh 87 pounds together. -/
theorem caitlin_sara_weight 
  (annette caitlin sara : ℝ) 
  (h1 : annette + caitlin = 95)   -- Annette and Caitlin weigh 95 pounds together
  (h2 : annette = sara + 8) :     -- Annette weighs 8 pounds more than Sara
  caitlin + sara = 87 := by sorry

end caitlin_sara_weight_l2009_200986


namespace quadratic_inequality_range_l2009_200945

theorem quadratic_inequality_range :
  (∀ x : ℝ, ∀ a : ℝ, (a - 2) * x^2 - 2 * (a - 2) * x - 4 < 0) ↔
  (a ∈ Set.Ioc (-2 : ℝ) 2) :=
by sorry

end quadratic_inequality_range_l2009_200945


namespace min_value_abc_l2009_200976

theorem min_value_abc (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (h_prod : a * b * c = 27) :
  a + 3 * b + 9 * c ≥ 27 ∧ 
  (a + 3 * b + 9 * c = 27 ↔ a = 9 ∧ b = 3 ∧ c = 1) :=
by sorry

end min_value_abc_l2009_200976


namespace negative_inequality_l2009_200900

theorem negative_inequality (a b : ℝ) (h : a > b) : -b > -a := by
  sorry

end negative_inequality_l2009_200900


namespace sqrt_ratio_equation_l2009_200917

theorem sqrt_ratio_equation (x : ℝ) :
  (Real.sqrt (2 * x + 7) / Real.sqrt (4 * x + 7) = Real.sqrt 7 / 2) →
  x = -21 / 20 := by
sorry

end sqrt_ratio_equation_l2009_200917


namespace sqrt_sum_irrational_l2009_200997

theorem sqrt_sum_irrational (n : ℕ+) : Irrational (Real.sqrt (n + 1) + Real.sqrt n) := by
  sorry

end sqrt_sum_irrational_l2009_200997


namespace administrative_staff_sample_size_l2009_200906

/-- Represents the number of administrative staff to be drawn in a stratified sample -/
def administrative_staff_in_sample (total_population : ℕ) (sample_size : ℕ) (administrative_staff : ℕ) : ℕ :=
  (administrative_staff * sample_size) / total_population

/-- Theorem stating that the number of administrative staff to be drawn is 4 -/
theorem administrative_staff_sample_size :
  administrative_staff_in_sample 160 20 32 = 4 := by
  sorry

end administrative_staff_sample_size_l2009_200906


namespace paint_per_statue_l2009_200909

theorem paint_per_statue (total_paint : ℚ) (num_statues : ℕ) : 
  total_paint = 7/8 ∧ num_statues = 14 → 
  total_paint / num_statues = 7/112 := by
sorry

end paint_per_statue_l2009_200909


namespace root_implies_m_values_l2009_200952

theorem root_implies_m_values (m : ℝ) : 
  ((m + 2) * 1^2 - 2 * 1 + m^2 - 2 * m - 6 = 0) → (m = -2 ∨ m = 3) :=
by sorry

end root_implies_m_values_l2009_200952


namespace simplify_expression_l2009_200946

theorem simplify_expression (x y : ℝ) (hx : x ≥ 0) (hy : y ≥ 0) :
  (x + 2 * Real.sqrt (x * y) + y) / (Real.sqrt x + Real.sqrt y) - 
  (Real.sqrt (x * y) + Real.sqrt x) * Real.sqrt (1 / x) = Real.sqrt x - 1 := by
  sorry

end simplify_expression_l2009_200946


namespace gcd_g_x_l2009_200962

def g (x : ℤ) : ℤ := (5*x+3)*(8*x+2)*(12*x+7)*(3*x+10)

theorem gcd_g_x (x : ℤ) (h : 46800 ∣ x) : 
  Nat.gcd (Int.natAbs (g x)) (Int.natAbs x) = 60 := by
  sorry

end gcd_g_x_l2009_200962


namespace percentage_problem_l2009_200971

theorem percentage_problem (x : ℝ) (h : 0.4 * x = 160) : 0.6 * x = 240 := by
  sorry

end percentage_problem_l2009_200971


namespace count_negative_numbers_l2009_200936

theorem count_negative_numbers : ∃ (S : Finset ℝ), 
  S = {8, 0, |(-2)|, -5, -2/3, (-1)^2} ∧ 
  (S.filter (λ x => x < 0)).card = 2 := by
sorry

end count_negative_numbers_l2009_200936


namespace fuel_mixture_problem_l2009_200947

/-- Proves that given a 200-gallon tank filled with two types of fuel,
    where one contains 12% ethanol and the other 16% ethanol,
    if the full tank contains 30 gallons of ethanol,
    then the volume of the first fuel added is 50 gallons. -/
theorem fuel_mixture_problem (tank_capacity : ℝ) (ethanol_a : ℝ) (ethanol_b : ℝ) 
    (total_ethanol : ℝ) (fuel_a : ℝ) :
  tank_capacity = 200 →
  ethanol_a = 0.12 →
  ethanol_b = 0.16 →
  total_ethanol = 30 →
  fuel_a * ethanol_a + (tank_capacity - fuel_a) * ethanol_b = total_ethanol →
  fuel_a = 50 := by
sorry

end fuel_mixture_problem_l2009_200947


namespace expression_not_33_l2009_200903

theorem expression_not_33 (x y : ℤ) : 
  x^5 + 3*x^4*y - 5*x^3*y^2 - 15*x^2*y^3 + 4*x*y^4 + 12*y^5 ≠ 33 := by
  sorry

end expression_not_33_l2009_200903


namespace angle_measure_theorem_l2009_200944

theorem angle_measure_theorem (x : ℝ) : 
  (180 - x = 7 * (90 - x)) → x = 75 := by
  sorry

end angle_measure_theorem_l2009_200944


namespace min_rows_for_hockey_arena_l2009_200905

/-- Represents a seating arrangement in a hockey arena --/
structure ArenaSeating where
  total_students : ℕ
  seats_per_row : ℕ
  max_students_per_school : ℕ
  same_row_constraint : Bool

/-- Calculates the minimum number of rows required for the given seating arrangement --/
def min_rows_required (seating : ArenaSeating) : ℕ :=
  sorry

/-- Theorem stating the minimum number of rows required for the given problem --/
theorem min_rows_for_hockey_arena :
  let seating := ArenaSeating.mk 2016 168 40 true
  min_rows_required seating = 15 :=
sorry

end min_rows_for_hockey_arena_l2009_200905


namespace negation_equivalence_l2009_200934

theorem negation_equivalence :
  (¬ ∃ x : ℝ, x ≥ 1 ∨ x > 2) ↔ (∀ x : ℝ, x < 1) := by
  sorry

end negation_equivalence_l2009_200934


namespace water_fraction_after_replacements_l2009_200961

/-- Represents the fraction of water in the radiator mixture -/
def water_fraction (n : ℕ) : ℚ :=
  (3/4 : ℚ) ^ n

/-- The radiator capacity in quarts -/
def radiator_capacity : ℕ := 16

/-- The amount of mixture removed and replaced in each iteration -/
def replacement_amount : ℕ := 4

/-- The number of replacement iterations -/
def num_iterations : ℕ := 4

theorem water_fraction_after_replacements :
  water_fraction num_iterations = 81/256 := by
  sorry

end water_fraction_after_replacements_l2009_200961


namespace equation_roots_range_l2009_200993

theorem equation_roots_range (n : ℕ) (k : ℝ) :
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧
    2*n - 1 < x₁ ∧ x₁ ≤ 2*n + 1 ∧
    2*n - 1 < x₂ ∧ x₂ ≤ 2*n + 1 ∧
    |x₁ - 2*n| = k * Real.sqrt x₁ ∧
    |x₂ - 2*n| = k * Real.sqrt x₂) →
  (0 < k ∧ k ≤ 1 / Real.sqrt (2*n + 1)) :=
by sorry

end equation_roots_range_l2009_200993


namespace base_conversion_theorem_l2009_200964

def base_7_to_10 (n : Nat) : Nat :=
  5 * 7^5 + 4 * 7^4 + 3 * 7^3 + 2 * 7^2 + 1 * 7^1 + 0 * 7^0

def base_10_to_4 (n : Nat) : List Nat :=
  let rec convert (m : Nat) (acc : List Nat) : List Nat :=
    if m = 0 then acc
    else convert (m / 4) ((m % 4) :: acc)
  convert n []

theorem base_conversion_theorem :
  (base_7_to_10 543210 = 94773) ∧
  (base_10_to_4 94773 = [1, 1, 3, 2, 3, 0, 1, 1]) := by
  sorry

end base_conversion_theorem_l2009_200964


namespace red_paint_cans_l2009_200902

theorem red_paint_cans (total_cans : ℕ) (red_ratio white_ratio : ℕ) 
  (h1 : total_cans = 35)
  (h2 : red_ratio = 4)
  (h3 : white_ratio = 3) : 
  (red_ratio : ℚ) / (red_ratio + white_ratio : ℚ) * total_cans = 20 :=
by
  sorry

end red_paint_cans_l2009_200902


namespace committee_meeting_attendance_l2009_200990

/-- Given a committee meeting with only associate and assistant professors, where:
    - Each associate professor brings 2 pencils and 1 chart
    - Each assistant professor brings 1 pencil and 2 charts
    - A total of 10 pencils and 11 charts are brought to the meeting
    Prove that the total number of people present is 7. -/
theorem committee_meeting_attendance :
  ∀ (associate_profs assistant_profs : ℕ),
    2 * associate_profs + assistant_profs = 10 →
    associate_profs + 2 * assistant_profs = 11 →
    associate_profs + assistant_profs = 7 :=
by sorry

end committee_meeting_attendance_l2009_200990


namespace green_shirt_cost_l2009_200911

-- Define the number of students in each grade
def kindergartners : ℕ := 101
def first_graders : ℕ := 113
def second_graders : ℕ := 107
def third_graders : ℕ := 108

-- Define the cost of shirts for each grade (in cents to avoid floating-point issues)
def orange_shirt_cost : ℕ := 580
def yellow_shirt_cost : ℕ := 500
def blue_shirt_cost : ℕ := 560

-- Define the total amount spent on all shirts (in cents)
def total_spent : ℕ := 231700

-- Theorem to prove
theorem green_shirt_cost :
  (total_spent - 
   (kindergartners * orange_shirt_cost + 
    first_graders * yellow_shirt_cost + 
    second_graders * blue_shirt_cost)) / third_graders = 525 := by
  sorry

end green_shirt_cost_l2009_200911


namespace valid_number_difference_l2009_200955

def is_valid_number (n : ℕ) : Prop :=
  (n ≥ 1000000000) ∧ (n < 10000000000) ∧ (n % 11 = 0) ∧
  (∀ d : ℕ, d < 10 → (∃! i : ℕ, i < 10 ∧ (n / 10^i) % 10 = d))

def largest_valid_number : ℕ := 9876524130

def smallest_valid_number : ℕ := 1024375869

theorem valid_number_difference :
  is_valid_number largest_valid_number ∧
  is_valid_number smallest_valid_number ∧
  (∀ n : ℕ, is_valid_number n → n ≤ largest_valid_number) ∧
  (∀ n : ℕ, is_valid_number n → n ≥ smallest_valid_number) ∧
  (largest_valid_number - smallest_valid_number = 8852148261) :=
sorry

end valid_number_difference_l2009_200955


namespace parallelogram_base_l2009_200940

/-- The base of a parallelogram given its area and height -/
theorem parallelogram_base (area height base : ℝ) 
  (h_area : area = 384) 
  (h_height : height = 16) 
  (h_formula : area = base * height) : 
  base = 24 := by
sorry

end parallelogram_base_l2009_200940


namespace taqeeshas_grade_l2009_200987

theorem taqeeshas_grade (total_students : ℕ) (students_present : ℕ) (initial_average : ℕ) (final_average : ℕ) :
  total_students = 17 →
  students_present = 16 →
  initial_average = 77 →
  final_average = 78 →
  (students_present * initial_average + (total_students - students_present) * 94) / total_students = final_average :=
by sorry

end taqeeshas_grade_l2009_200987


namespace jar_marbles_l2009_200915

theorem jar_marbles (a b c : ℕ) : 
  b = a + 12 →
  c = 2 * b →
  a + b + c = 148 →
  a = 28 := by sorry

end jar_marbles_l2009_200915


namespace equation_proof_l2009_200926

theorem equation_proof : Real.sqrt (5 + Real.sqrt (3 + Real.sqrt 14)) = (2 + Real.sqrt 14) ^ (1/4) := by
  sorry

end equation_proof_l2009_200926


namespace largest_number_in_ratio_l2009_200949

theorem largest_number_in_ratio (a b c d : ℕ) : 
  a + b + c + d = 1344 →
  2 * b = 3 * a →
  4 * a = 2 * c →
  5 * a = 2 * d →
  d = 480 := by
  sorry

end largest_number_in_ratio_l2009_200949


namespace max_segments_theorem_l2009_200919

/-- A configuration of points on a plane. -/
structure PointConfiguration where
  n : ℕ  -- number of points
  m : ℕ  -- number of points on the convex hull
  no_collinear_triple : Bool  -- no three points are collinear
  m_le_n : m ≤ n  -- number of points on convex hull cannot exceed total points

/-- The maximum number of non-intersecting line segments for a given point configuration. -/
def max_segments (config : PointConfiguration) : ℕ :=
  3 * config.n - config.m - 3

/-- Theorem stating the maximum number of non-intersecting line segments. -/
theorem max_segments_theorem (config : PointConfiguration) :
  config.no_collinear_triple →
  max_segments config = 3 * config.n - config.m - 3 :=
sorry

end max_segments_theorem_l2009_200919


namespace nicki_total_miles_l2009_200965

/-- Represents the number of weeks in a year -/
def weeks_in_year : ℕ := 52

/-- Represents the number of miles Nicki ran per week in the first half of the year -/
def first_half_miles_per_week : ℕ := 20

/-- Represents the number of miles Nicki ran per week in the second half of the year -/
def second_half_miles_per_week : ℕ := 30

/-- Calculates the total miles Nicki ran for the year -/
def total_miles_run : ℕ := 
  (first_half_miles_per_week * (weeks_in_year / 2)) + 
  (second_half_miles_per_week * (weeks_in_year / 2))

/-- Theorem stating that Nicki ran 1300 miles in total for the year -/
theorem nicki_total_miles : total_miles_run = 1300 := by
  sorry

end nicki_total_miles_l2009_200965


namespace four_integer_sum_l2009_200925

theorem four_integer_sum (a b c d : ℕ) : 
  a > 1 ∧ b > 1 ∧ c > 1 ∧ d > 1 →
  a * b * c * d = 14400 →
  Nat.gcd a b = 1 ∧ Nat.gcd a c = 1 ∧ Nat.gcd a d = 1 ∧
  Nat.gcd b c = 1 ∧ Nat.gcd b d = 1 ∧ Nat.gcd c d = 1 →
  a + b + c + d = 98 := by
sorry

end four_integer_sum_l2009_200925


namespace field_length_width_ratio_l2009_200901

/-- Proves that the ratio of length to width of a rectangular field is 2:1 given specific conditions -/
theorem field_length_width_ratio :
  ∀ (w : ℝ),
  w > 0 →
  ∃ (k : ℕ), k > 0 ∧ 20 = k * w →
  25 = (1/8) * (20 * w) →
  (20 : ℝ) / w = 2 := by
sorry

end field_length_width_ratio_l2009_200901


namespace divisible_by_27_l2009_200973

/-- Sum of digits of a natural number -/
def sum_of_digits (n : ℕ) : ℕ := sorry

/-- A natural number is three times the sum of its digits -/
def is_three_times_sum_of_digits (n : ℕ) : Prop :=
  n = 3 * sum_of_digits n

theorem divisible_by_27 (n : ℕ) (h : is_three_times_sum_of_digits n) : 
  27 ∣ n := by sorry

end divisible_by_27_l2009_200973


namespace smallest_number_divisible_by_8_11_24_l2009_200975

theorem smallest_number_divisible_by_8_11_24 :
  ∃ (k : ℕ), 255 + k > 255 ∧ (255 + k) % 8 = 0 ∧ (255 + k) % 11 = 0 ∧ (255 + k) % 24 = 0 ∧
  ∀ (n : ℕ), n < 255 → ¬∃ (m : ℕ), m > 0 ∧ (n + m) % 8 = 0 ∧ (n + m) % 11 = 0 ∧ (n + m) % 24 = 0 :=
by sorry

end smallest_number_divisible_by_8_11_24_l2009_200975


namespace unique_three_digit_number_l2009_200929

/-- A function that returns the sum of digits of a natural number -/
def sumOfDigits (n : ℕ) : ℕ :=
  if n < 10 then n else (n % 10) + sumOfDigits (n / 10)

/-- A predicate that checks if a number is a three-digit number -/
def isThreeDigit (n : ℕ) : Prop :=
  100 ≤ n ∧ n < 1000

theorem unique_three_digit_number : 
  ∃! n : ℕ, isThreeDigit n ∧ n = 12 * sumOfDigits n :=
by
  -- The proof would go here
  sorry

end unique_three_digit_number_l2009_200929


namespace dress_design_combinations_l2009_200980

theorem dress_design_combinations (colors patterns : ℕ) (h1 : colors = 5) (h2 : patterns = 6) : colors * patterns = 30 := by
  sorry

end dress_design_combinations_l2009_200980


namespace unique_solution_3_and_7_equation_l2009_200968

theorem unique_solution_3_and_7_equation :
  ∀ a y : ℕ, a ≥ 1 → y ≥ 1 →
  (3 ^ (2 * a - 1) + 3 ^ a + 1 = 7 ^ y) →
  (a = 1 ∧ y = 1) :=
by sorry

end unique_solution_3_and_7_equation_l2009_200968


namespace water_needed_proof_l2009_200938

/-- The ratio of water to lemon juice in the lemonade recipe -/
def water_ratio : ℚ := 8 / 10

/-- The number of gallons of lemonade to make -/
def gallons_to_make : ℚ := 2

/-- The number of quarts in a gallon -/
def quarts_per_gallon : ℚ := 4

/-- The number of liters in a quart -/
def liters_per_quart : ℚ := 95 / 100

/-- The amount of water needed in liters -/
def water_needed : ℚ := 
  water_ratio * gallons_to_make * quarts_per_gallon * liters_per_quart

theorem water_needed_proof : water_needed = 608 / 100 := by
  sorry

end water_needed_proof_l2009_200938


namespace hare_hunt_probability_l2009_200927

theorem hare_hunt_probability (p1 p2 p3 : ℝ) 
  (h1 : p1 = 3/5) 
  (h2 : p2 = 3/10) 
  (h3 : p3 = 1/10) : 
  1 - (1 - p1) * (1 - p2) * (1 - p3) = 0.748 := by
  sorry

end hare_hunt_probability_l2009_200927


namespace a_range_l2009_200918

def A : Set ℝ := {x | x > 1}
def B (a : ℝ) : Set ℝ := {a + 2}

theorem a_range (a : ℝ) : A ∩ B a = ∅ → a ≤ -1 := by
  sorry

end a_range_l2009_200918


namespace quadratic_decreasing_interval_l2009_200979

def quadratic_function (a b c : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x + c

theorem quadratic_decreasing_interval 
  (a b c : ℝ) 
  (h1 : a > 0) 
  (h2 : quadratic_function a b c (-5) = 0) 
  (h3 : quadratic_function a b c 3 = 0) :
  ∀ x ∈ Set.Iic (-1), 
    ∀ y ∈ Set.Iic (-1), 
      x < y → quadratic_function a b c x > quadratic_function a b c y :=
by sorry

end quadratic_decreasing_interval_l2009_200979


namespace slope_of_right_triangle_l2009_200912

/-- Given a right triangle ABC in the x-y plane where ∠B = 90°, AC = 100, and AB = 80,
    the slope of AC is 4/3 -/
theorem slope_of_right_triangle (A B C : ℝ × ℝ) : 
  (B.2 - A.2) ^ 2 + (B.1 - A.1) ^ 2 = 80 ^ 2 →
  (C.2 - A.2) ^ 2 + (C.1 - A.1) ^ 2 = 100 ^ 2 →
  (C.2 - B.2) ^ 2 + (C.1 - B.1) ^ 2 = (C.2 - A.2) ^ 2 + (C.1 - A.1) ^ 2 - (B.2 - A.2) ^ 2 - (B.1 - A.1) ^ 2 →
  (C.2 - A.2) / (C.1 - A.1) = 4 / 3 :=
by sorry

end slope_of_right_triangle_l2009_200912


namespace perpendicular_vectors_l2009_200954

/-- Two-dimensional vector type -/
def Vector2D := ℝ × ℝ

/-- Dot product of two 2D vectors -/
def dot_product (v w : Vector2D) : ℝ :=
  v.1 * w.1 + v.2 * w.2

/-- Perpendicularity of two 2D vectors -/
def perpendicular (v w : Vector2D) : Prop :=
  dot_product v w = 0

theorem perpendicular_vectors (k : ℝ) :
  let a : Vector2D := (2, 1)
  let b : Vector2D := (-1, k)
  perpendicular a b → k = 2 := by
  sorry

end perpendicular_vectors_l2009_200954


namespace max_sphere_surface_area_from_cube_l2009_200948

/-- Given a cube with side length 2, the maximum surface area of a sphere carved from this cube is 4π. -/
theorem max_sphere_surface_area_from_cube (cube_side_length : ℝ) (sphere_surface_area : ℝ → ℝ) :
  cube_side_length = 2 →
  (∀ r : ℝ, r ≤ 1 → sphere_surface_area r ≤ sphere_surface_area 1) →
  sphere_surface_area 1 = 4 * Real.pi :=
by
  sorry


end max_sphere_surface_area_from_cube_l2009_200948


namespace parabola_focus_lines_range_l2009_200958

/-- A parabola with equation y^2 = 2px, where p > 0 -/
structure Parabola where
  p : ℝ
  h_p_pos : p > 0

/-- A line passing through the focus of the parabola -/
structure FocusLine (para : Parabola) where
  k : ℝ  -- slope of the line

/-- Intersection points of a focus line with the parabola -/
def intersection_points (para : Parabola) (line : FocusLine para) : ℝ × ℝ := sorry

/-- Distance between intersection points -/
def distance (para : Parabola) (line : FocusLine para) : ℝ := sorry

/-- Number of focus lines with a specific intersection distance -/
def num_lines_with_distance (para : Parabola) (d : ℝ) : ℕ := sorry

theorem parabola_focus_lines_range (para : Parabola) :
  (num_lines_with_distance para 4 = 2) → (0 < para.p ∧ para.p < 2) := by sorry

end parabola_focus_lines_range_l2009_200958


namespace morning_campers_l2009_200977

theorem morning_campers (total : ℕ) (afternoon : ℕ) (morning : ℕ) : 
  total = 62 → afternoon = 27 → morning = total - afternoon → morning = 35 := by
  sorry

end morning_campers_l2009_200977


namespace cost_price_calculation_article_cost_price_l2009_200933

theorem cost_price_calculation (selling_price : ℝ) (discount_rate : ℝ) (profit_rate : ℝ) : ℝ :=
  let discounted_price := selling_price * (1 - discount_rate)
  let cost_price := discounted_price / (1 + profit_rate)
  cost_price

theorem article_cost_price : 
  cost_price_calculation 15000 0.1 0.08 = 12500 := by
  sorry

end cost_price_calculation_article_cost_price_l2009_200933


namespace sarahs_job_men_degree_percentage_l2009_200910

/-- Calculates the percentage of men with a college degree -/
def percentage_men_with_degree (total_employees : ℕ) (women_percentage : ℚ) 
  (num_women : ℕ) (men_without_degree : ℕ) : ℚ :=
  let num_men := total_employees - num_women
  let men_with_degree := num_men - men_without_degree
  (men_with_degree : ℚ) / (num_men : ℚ) * 100

/-- The percentage of men with a college degree at Sarah's job is 75% -/
theorem sarahs_job_men_degree_percentage :
  ∃ (total_employees : ℕ),
    (48 : ℚ) / (total_employees : ℚ) = (60 : ℚ) / 100 ∧
    percentage_men_with_degree total_employees ((60 : ℚ) / 100) 48 8 = 75 := by
  sorry

end sarahs_job_men_degree_percentage_l2009_200910


namespace combined_teaching_years_l2009_200995

/-- The combined teaching years of Mr. Spencer and Mrs. Randall -/
theorem combined_teaching_years : 
  let spencer_fourth_grade : ℕ := 12
  let spencer_first_grade : ℕ := 5
  let randall_third_grade : ℕ := 18
  let randall_second_grade : ℕ := 8
  (spencer_fourth_grade + spencer_first_grade + randall_third_grade + randall_second_grade) = 43 := by
  sorry

end combined_teaching_years_l2009_200995


namespace peter_investment_duration_l2009_200996

/-- Calculates the final amount after simple interest -/
def simple_interest (principal : ℝ) (rate : ℝ) (time : ℝ) : ℝ :=
  principal + (principal * rate * time)

theorem peter_investment_duration :
  ∀ (rate : ℝ),
  rate > 0 →
  simple_interest 650 rate 3 = 815 →
  simple_interest 650 rate 4 = 870 →
  ∃ (t : ℝ), t = 3 ∧ simple_interest 650 rate t = 815 :=
by sorry

end peter_investment_duration_l2009_200996


namespace rectangle_area_l2009_200904

/-- Given a rectangle with width 4 inches and perimeter 30 inches, prove its area is 44 square inches -/
theorem rectangle_area (width : ℝ) (perimeter : ℝ) : 
  width = 4 → perimeter = 30 → width * ((perimeter / 2) - width) = 44 :=
by sorry

end rectangle_area_l2009_200904


namespace perpendicular_planes_l2009_200950

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the perpendicular relation between lines and between a line and a plane
variable (perp_line : Line → Line → Prop)
variable (perp_line_plane : Line → Plane → Prop)

-- Define the perpendicular relation between planes
variable (perp_plane : Plane → Plane → Prop)

-- Theorem statement
theorem perpendicular_planes 
  (a b : Line) 
  (α β : Plane) 
  (h_diff_lines : a ≠ b) 
  (h_diff_planes : α ≠ β) 
  (h1 : perp_line a b) 
  (h2 : perp_line_plane a α) 
  (h3 : perp_line_plane b β) : 
  perp_plane α β :=
sorry

end perpendicular_planes_l2009_200950


namespace yard_length_is_250_l2009_200998

/-- The length of a yard with trees planted at equal distances -/
def yard_length (num_trees : ℕ) (tree_distance : ℝ) : ℝ :=
  (num_trees - 1) * tree_distance

/-- Theorem: The length of the yard is 250 meters -/
theorem yard_length_is_250 :
  yard_length 51 5 = 250 := by
  sorry

end yard_length_is_250_l2009_200998


namespace perfect_square_polynomial_l2009_200937

theorem perfect_square_polynomial (k : ℝ) : 
  (∃ a b : ℝ, ∀ x : ℝ, 4 * x^2 + (k - 1) * x + 9 = (a * x + b)^2) → 
  (k = 13 ∨ k = -11) :=
by sorry

end perfect_square_polynomial_l2009_200937


namespace area_R_specific_rhombus_l2009_200972

/-- Represents a rhombus ABCD -/
structure Rhombus where
  side_length : ℝ
  angle_B : ℝ

/-- Represents the region R inside the rhombus -/
def region_R (r : Rhombus) : Set (ℝ × ℝ) :=
  sorry

/-- The area of region R in the rhombus -/
def area_R (r : Rhombus) : ℝ :=
  sorry

/-- Theorem stating the area of region R in the specific rhombus -/
theorem area_R_specific_rhombus :
  let r : Rhombus := { side_length := 3, angle_B := 150 * π / 180 }
  area_R r = 9 * (Real.sqrt 6 - Real.sqrt 2) / 8 := by
    sorry

end area_R_specific_rhombus_l2009_200972


namespace cube_root_2450_l2009_200956

theorem cube_root_2450 : ∃ (a b : ℕ), a > 0 ∧ b > 0 ∧ (2450 : ℝ)^(1/3) = a * b^(1/3) ∧ 
  (∀ (c d : ℕ), c > 0 → d > 0 → (2450 : ℝ)^(1/3) = c * d^(1/3) → d ≥ b) ∧
  a = 35 ∧ b = 2 := by
  sorry

end cube_root_2450_l2009_200956


namespace original_purchase_cups_l2009_200923

/-- The cost of a single paper plate -/
def plate_cost : ℝ := sorry

/-- The cost of a single paper cup -/
def cup_cost : ℝ := sorry

/-- The number of paper cups in the original purchase -/
def num_cups : ℕ := sorry

/-- The total cost of 100 paper plates and some paper cups is $6.00 -/
axiom total_cost : 100 * plate_cost + num_cups * cup_cost = 6

/-- The total cost of 20 plates and 40 cups is $1.20 -/
axiom partial_cost : 20 * plate_cost + 40 * cup_cost = 1.2

theorem original_purchase_cups : num_cups = 200 := by
  sorry

end original_purchase_cups_l2009_200923


namespace sin_x_squared_not_periodic_l2009_200978

/-- Definition of a periodic function -/
def IsPeriodic (f : ℝ → ℝ) : Prop :=
  ∃ T : ℝ, T > 0 ∧ ∀ x : ℝ, f (x + T) = f x

/-- Statement: sin(x^2) is not periodic -/
theorem sin_x_squared_not_periodic : ¬ IsPeriodic (fun x ↦ Real.sin (x^2)) := by
  sorry


end sin_x_squared_not_periodic_l2009_200978


namespace contractor_engagement_days_l2009_200967

/-- Proves that the contractor was engaged for 20 days given the problem conditions --/
theorem contractor_engagement_days : 
  ∀ (daily_wage : ℚ) (daily_fine : ℚ) (total_amount : ℚ) (absent_days : ℕ),
    daily_wage = 25 →
    daily_fine = (15/2) →
    total_amount = 425 →
    absent_days = 10 →
    ∃ (engaged_days : ℕ), 
      engaged_days * daily_wage - absent_days * daily_fine = total_amount ∧
      engaged_days = 20 := by
  sorry

end contractor_engagement_days_l2009_200967


namespace smallest_sum_of_digits_product_l2009_200908

/-- A function that returns the sum of digits of a natural number -/
def sum_of_digits (n : ℕ) : ℕ := sorry

/-- A function that checks if a natural number has unique digits -/
def has_unique_digits (n : ℕ) : Prop := sorry

/-- A function that checks if a natural number is a two-digit number -/
def is_two_digit (n : ℕ) : Prop := 10 ≤ n ∧ n ≤ 99

/-- The main theorem -/
theorem smallest_sum_of_digits_product :
  ∃ (x y : ℕ),
    is_two_digit x ∧
    is_two_digit y ∧
    has_unique_digits (x * 100 + y) ∧
    (x * y ≥ 1000) ∧
    (x * y < 10000) ∧
    sum_of_digits (x * y) = 12 ∧
    ∀ (a b : ℕ),
      is_two_digit a →
      is_two_digit b →
      has_unique_digits (a * 100 + b) →
      (a * b ≥ 1000) →
      (a * b < 10000) →
      sum_of_digits (a * b) ≥ 12 :=
by sorry

end smallest_sum_of_digits_product_l2009_200908


namespace min_colors_is_three_l2009_200999

/-- Represents a coloring of a 5x5 grid -/
def Coloring := Fin 5 → Fin 5 → ℕ

/-- Checks if three points are collinear -/
def collinear (p1 p2 p3 : Fin 5 × Fin 5) : Prop :=
  let (x1, y1) := p1
  let (x2, y2) := p2
  let (x3, y3) := p3
  (x3 - x1) * (y2 - y1) = (x2 - x1) * (y3 - y1)

/-- Checks if a coloring is valid (no three same-colored points are collinear) -/
def valid_coloring (c : Coloring) : Prop :=
  ∀ p1 p2 p3 : Fin 5 × Fin 5,
    collinear p1 p2 p3 →
    (c p1.1 p1.2 = c p2.1 p2.2 ∧ c p2.1 p2.2 = c p3.1 p3.2) →
    p1 = p2 ∨ p2 = p3 ∨ p3 = p1

/-- The main theorem: the minimum number of colors for a valid coloring is 3 -/
theorem min_colors_is_three :
  (∃ (c : Coloring), valid_coloring c ∧ (∀ i j, c i j < 3)) ∧
  (∀ (c : Coloring), valid_coloring c → ∃ i j, c i j ≥ 2) :=
sorry

end min_colors_is_three_l2009_200999


namespace fraction_product_minus_one_l2009_200939

theorem fraction_product_minus_one : 
  (2/3) * (3/4) * (4/5) * (5/6) * (6/7) * (7/8) * (8/9) - 1 = -7/9 := by
  sorry

end fraction_product_minus_one_l2009_200939


namespace intersection_of_P_and_complement_of_M_l2009_200984

-- Define the universal set U as integers
def U : Set Int := Set.univ

-- Define set M
def M : Set Int := {1, 2}

-- Define set P
def P : Set Int := {x : Int | |x| ≤ 2}

-- State the theorem
theorem intersection_of_P_and_complement_of_M :
  P ∩ (U \ M) = {-2, -1, 0} := by sorry

end intersection_of_P_and_complement_of_M_l2009_200984


namespace largest_squared_fraction_l2009_200951

theorem largest_squared_fraction : 
  let a := (8/9 : ℚ)^2
  let b := (2/3 : ℚ)^2
  let c := (3/4 : ℚ)^2
  let d := (5/8 : ℚ)^2
  let e := (7/12 : ℚ)^2
  (a > b) ∧ (a > c) ∧ (a > d) ∧ (a > e) :=
by sorry

end largest_squared_fraction_l2009_200951


namespace line_intercept_ratio_l2009_200974

theorem line_intercept_ratio (b : ℝ) (s t : ℝ) 
  (h_b : b ≠ 0)
  (h_s : 0 = 10 * s + b)
  (h_t : 0 = 6 * t + b) :
  s / t = 3 / 5 := by
sorry

end line_intercept_ratio_l2009_200974


namespace rectangle_perimeter_l2009_200920

theorem rectangle_perimeter (a b c w : ℝ) (h1 : a = 5) (h2 : b = 12) (h3 : c = 13) (h4 : w = 5) : 
  let triangle_area := (1/2) * a * b
  let rectangle_length := triangle_area / w
  2 * (rectangle_length + w) = 22 :=
by sorry

end rectangle_perimeter_l2009_200920


namespace friendly_function_fixed_point_l2009_200943

def FriendlyFunction (f : ℝ → ℝ) : Prop :=
  (∀ x ∈ Set.Icc 0 1, f x ≥ 0) ∧
  (f 1 = 1) ∧
  (∀ x₁ x₂ : ℝ, x₁ ≥ 0 → x₂ ≥ 0 → x₁ + x₂ ≤ 1 → f (x₁ + x₂) ≥ f x₁ + f x₂)

theorem friendly_function_fixed_point
  (f : ℝ → ℝ)
  (h_friendly : FriendlyFunction f)
  (x₀ : ℝ)
  (h_x₀_in_range : x₀ ∈ Set.Icc 0 1)
  (h_fx₀_in_range : f x₀ ∈ Set.Icc 0 1)
  (h_ffx₀_eq_x₀ : f (f x₀) = x₀) :
  f x₀ = x₀ :=
by sorry

end friendly_function_fixed_point_l2009_200943


namespace tan_inequality_l2009_200988

-- Define the constants and their properties
axiom α : Real
axiom β : Real
axiom k : Int

-- Define the conditions
axiom sin_inequality : Real.sin α > Real.sin β
axiom α_not_right_angle : ∀ k, α ≠ k * Real.pi + Real.pi / 2
axiom β_not_right_angle : ∀ k, β ≠ k * Real.pi + Real.pi / 2
axiom fourth_quadrant : 3 * Real.pi / 2 < α ∧ α < 2 * Real.pi ∧ 3 * Real.pi / 2 < β ∧ β < 2 * Real.pi

-- State the theorem
theorem tan_inequality : Real.tan α > Real.tan β := by
  sorry

end tan_inequality_l2009_200988


namespace decagon_diagonals_l2009_200928

/-- The number of diagonals in a regular polygon with n sides -/
def num_diagonals (n : ℕ) : ℕ := n * (n - 3) / 2

/-- A decagon has 10 sides -/
def decagon_sides : ℕ := 10

theorem decagon_diagonals : num_diagonals decagon_sides = 35 := by
  sorry

end decagon_diagonals_l2009_200928


namespace max_profit_at_180_l2009_200913

-- Define the selling price x and daily sales y
variable (x y : ℝ)

-- Define the cost price
def cost_price : ℝ := 80

-- Define the range of selling price
def selling_price_range (x : ℝ) : Prop := 120 ≤ x ∧ x ≤ 180

-- Define the relationship between y and x
def sales_function (x : ℝ) : ℝ := -0.5 * x + 160

-- Define the profit function
def profit_function (x : ℝ) : ℝ := (x - cost_price) * (sales_function x)

-- Theorem statement
theorem max_profit_at_180 :
  ∀ x, selling_price_range x →
    profit_function x ≤ profit_function 180 ∧
    profit_function 180 = 7000 :=
by sorry

end max_profit_at_180_l2009_200913


namespace students_needed_to_fill_buses_l2009_200922

theorem students_needed_to_fill_buses (total_students : ℕ) (bus_capacity : ℕ) : 
  total_students = 254 → bus_capacity = 30 → 
  (((total_students + 16) / bus_capacity : ℕ) * bus_capacity = total_students + 16) ∧
  (((total_students + 15) / bus_capacity : ℕ) * bus_capacity < total_students + 15) := by
  sorry


end students_needed_to_fill_buses_l2009_200922


namespace gcd_660_924_l2009_200982

theorem gcd_660_924 : Nat.gcd 660 924 = 132 := by
  sorry

end gcd_660_924_l2009_200982


namespace subtract_squares_l2009_200935

theorem subtract_squares (a : ℝ) : 3 * a^2 - a^2 = 2 * a^2 := by
  sorry

end subtract_squares_l2009_200935


namespace ratio_of_divisors_sums_l2009_200959

def N : ℕ := 68 * 68 * 125 * 135

def sum_of_odd_divisors (n : ℕ) : ℕ := sorry
def sum_of_even_divisors (n : ℕ) : ℕ := sorry

theorem ratio_of_divisors_sums :
  (sum_of_odd_divisors N) * 30 = sum_of_even_divisors N :=
sorry

end ratio_of_divisors_sums_l2009_200959


namespace smallest_integer_satisfying_inequality_l2009_200924

theorem smallest_integer_satisfying_inequality :
  ∀ y : ℤ, y < 3*y - 10 → y ≥ 6 ∧ 6 < 3*6 - 10 := by
  sorry

end smallest_integer_satisfying_inequality_l2009_200924


namespace infinite_solutions_l2009_200932

theorem infinite_solutions (p : Nat) (hp : p.Prime) (hp_gt_7 : p > 7) :
  ∃ f : Nat → Nat,
    Function.Injective f ∧
    ∀ k : Nat, 
      (f k ≡ 1 [MOD 2016]) ∧ 
      (p ∣ (2^(f k) + f k)) :=
sorry

end infinite_solutions_l2009_200932


namespace average_difference_l2009_200953

def num_students : ℕ := 120
def num_teachers : ℕ := 6
def class_sizes : List ℕ := [40, 35, 25, 10, 5, 5]

def t : ℚ := (List.sum class_sizes) / num_teachers

def s : ℚ := (List.sum (List.map (λ x => x * x) class_sizes)) / num_students

theorem average_difference : t - s = -10 := by sorry

end average_difference_l2009_200953


namespace speed_to_achieve_average_l2009_200914

/-- Given a person driving at two different speeds over two time periods, 
    this theorem proves the required speed for the second period to achieve a specific average speed. -/
theorem speed_to_achieve_average 
  (initial_speed : ℝ) 
  (initial_time : ℝ) 
  (additional_time : ℝ) 
  (average_speed : ℝ) 
  (h1 : initial_speed = 60) 
  (h2 : initial_time = 3) 
  (h3 : additional_time = 2) 
  (h4 : average_speed = 70) : 
  ∃ x : ℝ, 
    (initial_speed * initial_time + x * additional_time) / (initial_time + additional_time) = average_speed 
    ∧ x = 85 := by
  sorry

end speed_to_achieve_average_l2009_200914


namespace xiaoming_mother_height_l2009_200981

/-- Given Xiaoming's height, stool height, and the difference between Xiaoming on the stool and his mother's height, prove the height of Xiaoming's mother. -/
theorem xiaoming_mother_height 
  (xiaoming_height : ℝ) 
  (stool_height : ℝ) 
  (height_difference : ℝ) 
  (h1 : xiaoming_height = 1.30)
  (h2 : stool_height = 0.4)
  (h3 : height_difference = 0.08)
  (h4 : xiaoming_height + stool_height = height_difference + mother_height) :
  mother_height = 1.62 :=
by
  sorry

#check xiaoming_mother_height

end xiaoming_mother_height_l2009_200981


namespace largest_abab_divisible_by_14_l2009_200992

/-- Represents a four-digit number of the form abab -/
def IsAbabForm (n : ℕ) : Prop :=
  ∃ (a b : ℕ), a < 10 ∧ b < 10 ∧ n = 1000 * a + 100 * b + 10 * a + b

/-- Checks if a number is the product of a two-digit and a three-digit number -/
def IsProductOfTwoAndThreeDigit (n : ℕ) : Prop :=
  ∃ (x y : ℕ), 10 ≤ x ∧ x < 100 ∧ 100 ≤ y ∧ y < 1000 ∧ n = x * y

/-- The main theorem stating the largest four-digit number of the form abab
    that is divisible by 14 and a product of two-digit and three-digit numbers -/
theorem largest_abab_divisible_by_14 :
  ∀ A : ℕ,
  IsAbabForm A →
  IsProductOfTwoAndThreeDigit A →
  A % 14 = 0 →
  A ≤ 9898 :=
by sorry

end largest_abab_divisible_by_14_l2009_200992


namespace gcd_triple_existence_l2009_200941

theorem gcd_triple_existence (S : Set ℕ+) 
  (h_infinite : Set.Infinite S)
  (h_distinct_gcd : ∃ (a b c d : ℕ+), a ∈ S ∧ b ∈ S ∧ c ∈ S ∧ d ∈ S ∧
    a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d ∧
    Nat.gcd a b ≠ Nat.gcd c d) :
  ∃ (x y z : ℕ+), x ∈ S ∧ y ∈ S ∧ z ∈ S ∧
    x ≠ y ∧ y ≠ z ∧ x ≠ z ∧
    Nat.gcd x y = Nat.gcd y z ∧ Nat.gcd y z ≠ Nat.gcd z x :=
by
  sorry

end gcd_triple_existence_l2009_200941


namespace imaginary_part_of_z_l2009_200916

theorem imaginary_part_of_z (z : ℂ) : z = (1 - I) / (1 + 3*I) → z.im = -2/5 := by
  sorry

end imaginary_part_of_z_l2009_200916


namespace cubic_expansion_coefficient_l2009_200991

theorem cubic_expansion_coefficient (a a₁ a₂ a₃ : ℝ) :
  (∀ x : ℝ, x^3 = a + a₁*(x-2) + a₂*(x-2)^2 + a₃*(x-2)^3) →
  a₂ = 6 := by
sorry

end cubic_expansion_coefficient_l2009_200991


namespace initial_bench_weight_l2009_200960

/-- Represents the weightlifting scenario for John --/
structure WeightliftingScenario where
  initialSquat : ℝ
  initialDeadlift : ℝ
  squatLossPercentage : ℝ
  deadliftLoss : ℝ
  newTotal : ℝ

/-- Calculates the initial bench weight given the weightlifting scenario --/
def calculateInitialBench (scenario : WeightliftingScenario) : ℝ :=
  scenario.newTotal - 
  (scenario.initialSquat * (1 - scenario.squatLossPercentage)) - 
  (scenario.initialDeadlift - scenario.deadliftLoss)

/-- Theorem stating that the initial bench weight is 400 pounds --/
theorem initial_bench_weight (scenario : WeightliftingScenario) 
  (h1 : scenario.initialSquat = 700)
  (h2 : scenario.initialDeadlift = 800)
  (h3 : scenario.squatLossPercentage = 0.3)
  (h4 : scenario.deadliftLoss = 200)
  (h5 : scenario.newTotal = 1490) :
  calculateInitialBench scenario = 400 := by
  sorry


end initial_bench_weight_l2009_200960
