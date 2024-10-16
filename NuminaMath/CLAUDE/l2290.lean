import Mathlib

namespace NUMINAMATH_CALUDE_children_on_bus_after_stop_l2290_229050

theorem children_on_bus_after_stop (initial : ℕ) (got_on : ℕ) (got_off : ℕ) :
  initial = 22 → got_on = 40 → got_off = 60 →
  initial + got_on - got_off = 2 := by
  sorry

end NUMINAMATH_CALUDE_children_on_bus_after_stop_l2290_229050


namespace NUMINAMATH_CALUDE_largest_number_on_board_l2290_229084

def is_two_digit (n : ℕ) : Prop := 10 ≤ n ∧ n ≤ 99

def ends_in_four (n : ℕ) : Prop := n % 10 = 4

def set_of_numbers : Set ℕ := {n | is_two_digit n ∧ n % 6 = 0 ∧ ends_in_four n}

theorem largest_number_on_board : 
  ∃ (m : ℕ), m ∈ set_of_numbers ∧ ∀ (n : ℕ), n ∈ set_of_numbers → n ≤ m ∧ m = 84 :=
sorry

end NUMINAMATH_CALUDE_largest_number_on_board_l2290_229084


namespace NUMINAMATH_CALUDE_perpendicular_vector_scalar_l2290_229071

/-- Given plane vectors a and b, if m * a + b is perpendicular to a, then m = 1 -/
theorem perpendicular_vector_scalar (a b : ℝ × ℝ) (m : ℝ) 
  (h1 : a = (-1, 3)) 
  (h2 : b = (4, -2)) 
  (h3 : (m * a.1 + b.1) * a.1 + (m * a.2 + b.2) * a.2 = 0) : 
  m = 1 := by
  sorry

end NUMINAMATH_CALUDE_perpendicular_vector_scalar_l2290_229071


namespace NUMINAMATH_CALUDE_tens_digit_of_nine_power_2021_l2290_229034

theorem tens_digit_of_nine_power_2021 : ∃ n : ℕ, 9^2021 = 100 * n + 9 :=
sorry

end NUMINAMATH_CALUDE_tens_digit_of_nine_power_2021_l2290_229034


namespace NUMINAMATH_CALUDE_board_numbers_product_l2290_229095

theorem board_numbers_product (a b c d e : ℤ) : 
  ({a + b, a + c, a + d, a + e, b + c, b + d, b + e, c + d, c + e, d + e} : Finset ℤ) = 
    {2, 6, 10, 10, 12, 14, 16, 18, 20, 24} → 
  a * b * c * d * e = -3003 := by
sorry

end NUMINAMATH_CALUDE_board_numbers_product_l2290_229095


namespace NUMINAMATH_CALUDE_job_completion_time_l2290_229054

/-- The time taken by two workers to complete a job together, given their relative efficiencies and the time taken by one worker alone. -/
theorem job_completion_time 
  (efficiency_a : ℝ) 
  (efficiency_b : ℝ) 
  (time_a_alone : ℝ) 
  (h1 : efficiency_a = efficiency_b + 0.6 * efficiency_b) 
  (h2 : time_a_alone = 35) 
  : (1 / (1 / time_a_alone + efficiency_b / (efficiency_a * time_a_alone))) = 25 := by
  sorry

end NUMINAMATH_CALUDE_job_completion_time_l2290_229054


namespace NUMINAMATH_CALUDE_hexagonal_glass_side_length_l2290_229061

/-- A glass with regular hexagonal top and bottom, containing three identical spheres -/
structure HexagonalGlass where
  /-- Side length of the hexagonal bottom -/
  sideLength : ℝ
  /-- Volume of the glass -/
  volume : ℝ
  /-- The glass contains three identical spheres each touching every side -/
  spheresFit : True
  /-- The volume of the glass is 108 cm³ -/
  volumeIs108 : volume = 108

/-- The theorem stating the relationship between the glass volume and side length -/
theorem hexagonal_glass_side_length (g : HexagonalGlass) : 
  g.sideLength = 2 / Real.rpow 3 (1/3) :=
sorry

end NUMINAMATH_CALUDE_hexagonal_glass_side_length_l2290_229061


namespace NUMINAMATH_CALUDE_gcd_problem_l2290_229081

theorem gcd_problem (n : ℕ) : 
  80 ≤ n ∧ n ≤ 100 → Nat.gcd 36 n = 12 → n = 84 ∨ n = 96 := by
  sorry

end NUMINAMATH_CALUDE_gcd_problem_l2290_229081


namespace NUMINAMATH_CALUDE_inequality_proof_l2290_229027

theorem inequality_proof (a b c : ℝ) 
  (pos_a : a > 0) (pos_b : b > 0) (pos_c : c > 0)
  (sum_of_squares : a^2 + b^2 + c^2 = 1) :
  a*b/c + b*c/a + c*a/b ≥ Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l2290_229027


namespace NUMINAMATH_CALUDE_sum_of_coefficients_l2290_229013

theorem sum_of_coefficients (a a₁ a₂ a₃ a₄ a₅ a₆ a₇ a₈ a₉ a₁₀ a₁₁ : ℝ) :
  (∀ x : ℝ, (x^2 + 1) * (2*x + 1)^9 = a + a₁*(x+2) + a₂*(x+2)^2 + a₃*(x+2)^3 + 
    a₄*(x+2)^4 + a₅*(x+2)^5 + a₆*(x+2)^6 + a₇*(x+2)^7 + a₈*(x+2)^8 + 
    a₉*(x+2)^9 + a₁₀*(x+2)^10 + a₁₁*(x+2)^11) →
  a + a₁ + a₂ + a₃ + a₄ + a₅ + a₆ + a₇ + a₈ + a₉ + a₁₀ + a₁₁ = -2 :=
by sorry

end NUMINAMATH_CALUDE_sum_of_coefficients_l2290_229013


namespace NUMINAMATH_CALUDE_regression_line_not_necessarily_through_sample_point_l2290_229041

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a linear regression model -/
structure LinearRegression where
  a : ℝ  -- intercept
  b : ℝ  -- slope

/-- Calculates the y-value for a given x using the linear regression model -/
def predict (model : LinearRegression) (x : ℝ) : ℝ :=
  model.a + model.b * x

/-- Checks if a point lies on the regression line -/
def pointOnLine (model : LinearRegression) (p : Point) : Prop :=
  p.y = predict model p.x

/-- Theorem: The linear regression line does not necessarily pass through any sample point -/
theorem regression_line_not_necessarily_through_sample_point :
  ∃ (model : LinearRegression) (samples : List Point),
    samples.length > 0 ∧ ∀ p ∈ samples, ¬(pointOnLine model p) :=
by sorry

end NUMINAMATH_CALUDE_regression_line_not_necessarily_through_sample_point_l2290_229041


namespace NUMINAMATH_CALUDE_trigonometric_identities_l2290_229088

theorem trigonometric_identities (α : Real) 
  (h1 : 3 * Real.pi / 4 < α) 
  (h2 : α < Real.pi) 
  (h3 : Real.tan α + 1 / Real.tan α = -10 / 3) : 
  (Real.tan α = -1 / 3) ∧ 
  ((Real.sin α + Real.cos α) / (Real.sin α - Real.cos α) = -1 / 2) ∧ 
  (2 * Real.sin α ^ 2 - Real.sin α * Real.cos α - 3 * Real.cos α ^ 2 = -11 / 5) := by
  sorry

end NUMINAMATH_CALUDE_trigonometric_identities_l2290_229088


namespace NUMINAMATH_CALUDE_dislike_sector_angle_l2290_229017

-- Define the ratios for the four categories
def ratio_extremely_like : ℕ := 6
def ratio_like : ℕ := 9
def ratio_somewhat_like : ℕ := 2
def ratio_dislike : ℕ := 1

-- Define the total ratio
def total_ratio : ℕ := ratio_extremely_like + ratio_like + ratio_somewhat_like + ratio_dislike

-- Define the central angle of the dislike sector
def central_angle_dislike : ℚ := (ratio_dislike : ℚ) / (total_ratio : ℚ) * 360

-- Theorem statement
theorem dislike_sector_angle :
  central_angle_dislike = 20 := by sorry

end NUMINAMATH_CALUDE_dislike_sector_angle_l2290_229017


namespace NUMINAMATH_CALUDE_negation_of_all_squares_positive_l2290_229059

theorem negation_of_all_squares_positive :
  ¬(∀ n : ℕ, n^2 > 0) ↔ ∃ n : ℕ, ¬(n^2 > 0) := by sorry

end NUMINAMATH_CALUDE_negation_of_all_squares_positive_l2290_229059


namespace NUMINAMATH_CALUDE_unique_solution_for_complex_equation_l2290_229060

theorem unique_solution_for_complex_equation (x : ℝ) :
  x - 8 ≥ 0 →
  (7 / (Real.sqrt (x - 8) - 10) + 2 / (Real.sqrt (x - 8) - 4) +
   9 / (Real.sqrt (x - 8) + 4) + 14 / (Real.sqrt (x - 8) + 10) = 0) ↔
  x = 55 := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_for_complex_equation_l2290_229060


namespace NUMINAMATH_CALUDE_power_of_two_divisibility_l2290_229006

theorem power_of_two_divisibility (n : ℕ+) :
  (∃ k : ℕ, 2^n.val - 1 = 7 * k) ↔ (∃ m : ℕ, n.val = 3 * m) ∧
  ¬(∃ k : ℕ, 2^n.val + 1 = 7 * k) :=
by sorry

end NUMINAMATH_CALUDE_power_of_two_divisibility_l2290_229006


namespace NUMINAMATH_CALUDE_isosceles_triangle_angle_b_l2290_229093

-- Define an isosceles triangle
structure IsoscelesTriangle where
  -- We only need to define two angles, as the third can be derived
  angle_a : ℝ
  angle_b : ℝ
  is_isosceles : (angle_a = angle_b) ∨ (angle_a + 2 * angle_b = 180) ∨ (2 * angle_a + angle_b = 180)

-- Define the theorem
theorem isosceles_triangle_angle_b (t : IsoscelesTriangle) 
  (h : t.angle_a = 70) : 
  t.angle_b = 55 ∨ t.angle_b = 70 ∨ t.angle_b = 40 := by
  sorry


end NUMINAMATH_CALUDE_isosceles_triangle_angle_b_l2290_229093


namespace NUMINAMATH_CALUDE_ten_parabolas_regions_l2290_229007

/-- The number of regions a circle can be divided into by n parabolas -/
def circle_regions (n : ℕ) : ℕ := 2 * n^2 + 1

/-- Theorem stating that 10 parabolas divide a circle into 201 regions -/
theorem ten_parabolas_regions : circle_regions 10 = 201 := by
  sorry

end NUMINAMATH_CALUDE_ten_parabolas_regions_l2290_229007


namespace NUMINAMATH_CALUDE_luke_candy_purchase_l2290_229092

/-- The number of candy pieces Luke can buy given his tickets and candy cost -/
def candyPieces (whackAMoleTickets skeeBallTickets candyCost : ℕ) : ℕ :=
  (whackAMoleTickets + skeeBallTickets) / candyCost

/-- Proof that Luke can buy 5 pieces of candy -/
theorem luke_candy_purchase :
  candyPieces 2 13 3 = 5 := by
  sorry

end NUMINAMATH_CALUDE_luke_candy_purchase_l2290_229092


namespace NUMINAMATH_CALUDE_last_digit_of_seven_power_seven_power_l2290_229045

theorem last_digit_of_seven_power_seven_power (n : ℕ) : 7^(7^7) ≡ 3 [ZMOD 10] := by
  sorry

end NUMINAMATH_CALUDE_last_digit_of_seven_power_seven_power_l2290_229045


namespace NUMINAMATH_CALUDE_inscribed_circle_radius_is_five_halves_l2290_229067

/-- A trapezoid with an inscribed circle -/
structure InscribedCircleTrapezoid where
  /-- The length of the larger base -/
  a : ℕ
  /-- The length of the smaller base -/
  b : ℕ
  /-- The height of the trapezoid -/
  h : ℕ
  /-- The radius of the inscribed circle -/
  r : ℚ
  /-- The area of the upper part divided by the median -/
  upper_area : ℕ
  /-- The area of the lower part divided by the median -/
  lower_area : ℕ
  /-- Ensure the bases are different (it's a trapezoid) -/
  base_diff : a > b
  /-- The total area of the trapezoid -/
  total_area : (a + b) * h / 2 = upper_area + lower_area
  /-- The median divides the trapezoid into two parts -/
  median_division : upper_area = 15 ∧ lower_area = 30
  /-- The radius is half the height (property of inscribed circle in trapezoid) -/
  radius_height_relation : r = h / 2

/-- Theorem stating that the radius of the inscribed circle is 5/2 -/
theorem inscribed_circle_radius_is_five_halves (t : InscribedCircleTrapezoid) : t.r = 5 / 2 := by
  sorry


end NUMINAMATH_CALUDE_inscribed_circle_radius_is_five_halves_l2290_229067


namespace NUMINAMATH_CALUDE_family_divisors_characterization_l2290_229090

/-- Represents a six-digit number and its family -/
def SixDigitFamily :=
  {A : ℕ // A ≥ 100000 ∧ A < 1000000}

/-- Generates the k-th member of the family for a six-digit number -/
def family_member (A : SixDigitFamily) (k : Fin 6) : ℕ :=
  let B := A.val / (10^k.val)
  let C := A.val % (10^k.val)
  10^(6-k.val) * C + B

/-- The set of numbers that divide all members of a six-digit number's family -/
def family_divisors (A : SixDigitFamily) : Set ℕ :=
  {x : ℕ | ∀ k : Fin 6, (family_member A k) % x = 0}

/-- The set of numbers we're proving to be the family_divisors -/
def target_set : Set ℕ :=
  {x : ℕ | x ≥ 1000000 ∨ 
           (∃ h : Fin 9, x = 111111 * (h.val + 1)) ∨
           999999 % x = 0}

/-- The main theorem stating that family_divisors is a subset of target_set -/
theorem family_divisors_characterization (A : SixDigitFamily) :
  family_divisors A ⊆ target_set := by
  sorry


end NUMINAMATH_CALUDE_family_divisors_characterization_l2290_229090


namespace NUMINAMATH_CALUDE_doll_difference_proof_l2290_229053

-- Define the number of dolls for each person
def geraldine_dolls : ℝ := 2186.0
def jazmin_dolls : ℝ := 1209.0

-- Define the difference in dolls
def doll_difference : ℝ := geraldine_dolls - jazmin_dolls

-- Theorem statement
theorem doll_difference_proof : doll_difference = 977.0 := by
  sorry

end NUMINAMATH_CALUDE_doll_difference_proof_l2290_229053


namespace NUMINAMATH_CALUDE_prob_at_most_one_first_class_l2290_229021

/-- The probability of selecting at most one first-class product when randomly choosing 2 out of 5 products (3 first-class and 2 second-class) is 0.7 -/
theorem prob_at_most_one_first_class (total : ℕ) (first_class : ℕ) (second_class : ℕ) (selected : ℕ) :
  total = 5 →
  first_class = 3 →
  second_class = 2 →
  selected = 2 →
  (Nat.choose first_class 1 * Nat.choose second_class 1 + Nat.choose second_class 2) / Nat.choose total selected = 7 / 10 :=
by sorry

end NUMINAMATH_CALUDE_prob_at_most_one_first_class_l2290_229021


namespace NUMINAMATH_CALUDE_meadowood_58_impossible_l2290_229039

/-- Represents the village of Meadowood with its animal and people relationships -/
structure Meadowood where
  sheep : ℕ
  horses : ℕ
  ducks : ℕ := 5 * sheep
  cows : ℕ := 2 * horses
  people : ℕ := 4 * ducks

/-- The total population in Meadowood -/
def Meadowood.total (m : Meadowood) : ℕ :=
  m.people + m.horses + m.sheep + m.cows + m.ducks

/-- Theorem stating that 58 cannot be the total population in Meadowood -/
theorem meadowood_58_impossible : ¬∃ m : Meadowood, m.total = 58 := by
  sorry

end NUMINAMATH_CALUDE_meadowood_58_impossible_l2290_229039


namespace NUMINAMATH_CALUDE_josh_and_anna_marriage_problem_l2290_229028

/-- Josh and Anna's marriage problem -/
theorem josh_and_anna_marriage_problem 
  (josh_marriage_age : ℕ) 
  (marriage_duration : ℕ) 
  (combined_age_factor : ℕ) 
  (h1 : josh_marriage_age = 22)
  (h2 : marriage_duration = 30)
  (h3 : combined_age_factor = 5)
  (h4 : josh_marriage_age + marriage_duration + (josh_marriage_age + marriage_duration + anna_marriage_age) = combined_age_factor * josh_marriage_age) :
  anna_marriage_age = 28 :=
by sorry

end NUMINAMATH_CALUDE_josh_and_anna_marriage_problem_l2290_229028


namespace NUMINAMATH_CALUDE_least_subtrahend_for_divisibility_specific_case_l2290_229009

theorem least_subtrahend_for_divisibility (n : Nat) (d : Nat) (h : d > 0) :
  ∃ (k : Nat), k < d ∧ (n - k) % d = 0 ∧ ∀ (m : Nat), m < k → (n - m) % d ≠ 0 :=
by
  sorry

theorem specific_case : 
  ∃ (k : Nat), k < 47 ∧ (929 - k) % 47 = 0 ∧ ∀ (m : Nat), m < k → (929 - m) % 47 ≠ 0 ∧ k = 44 :=
by
  sorry

end NUMINAMATH_CALUDE_least_subtrahend_for_divisibility_specific_case_l2290_229009


namespace NUMINAMATH_CALUDE_inverse_N_expression_l2290_229000

def N : Matrix (Fin 2) (Fin 2) ℚ := !![3, 0; 2, -4]

theorem inverse_N_expression : 
  N⁻¹ = (1 / 12 : ℚ) • N + (1 / 12 : ℚ) • (1 : Matrix (Fin 2) (Fin 2) ℚ) := by
  sorry

end NUMINAMATH_CALUDE_inverse_N_expression_l2290_229000


namespace NUMINAMATH_CALUDE_strawberries_per_jar_solution_l2290_229091

/-- The number of strawberries used in one jar of jam -/
def strawberries_per_jar (betty_strawberries : ℕ) (matthew_extra : ℕ) (jar_price : ℕ) (total_revenue : ℕ) : ℕ :=
  let matthew_strawberries := betty_strawberries + matthew_extra
  let natalie_strawberries := matthew_strawberries / 2
  let total_strawberries := betty_strawberries + matthew_strawberries + natalie_strawberries
  let jars_sold := total_revenue / jar_price
  total_strawberries / jars_sold

theorem strawberries_per_jar_solution :
  strawberries_per_jar 16 20 4 40 = 7 := by
  sorry

end NUMINAMATH_CALUDE_strawberries_per_jar_solution_l2290_229091


namespace NUMINAMATH_CALUDE_max_sum_cubes_l2290_229044

theorem max_sum_cubes (e f g h i : ℝ) (h1 : e^4 + f^4 + g^4 + h^4 + i^4 = 5) :
  ∃ (M : ℝ), M = 5^(3/4) ∧ e^3 + f^3 + g^3 + h^3 + i^3 ≤ M ∧
  ∃ (e' f' g' h' i' : ℝ), e'^4 + f'^4 + g'^4 + h'^4 + i'^4 = 5 ∧
                          e'^3 + f'^3 + g'^3 + h'^3 + i'^3 = M :=
by sorry

end NUMINAMATH_CALUDE_max_sum_cubes_l2290_229044


namespace NUMINAMATH_CALUDE_september_first_was_wednesday_l2290_229075

/-- Represents the days of the week -/
inductive DayOfWeek
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday
  | Saturday
  | Sunday

/-- Calculates the number of lessons Vasya skips on a given day -/
def lessonsSkipped (day : DayOfWeek) : Nat :=
  match day with
  | DayOfWeek.Monday => 1
  | DayOfWeek.Tuesday => 2
  | DayOfWeek.Wednesday => 3
  | DayOfWeek.Thursday => 4
  | DayOfWeek.Friday => 5
  | _ => 0

/-- Calculates the day of the week for a given date in September -/
def dayOfWeekForDate (date : Nat) (sept1 : DayOfWeek) : DayOfWeek :=
  sorry

/-- Calculates the total number of lessons Vasya skipped in September -/
def totalLessonsSkipped (sept1 : DayOfWeek) : Nat :=
  sorry

theorem september_first_was_wednesday :
  totalLessonsSkipped DayOfWeek.Wednesday = 64 :=
by sorry

end NUMINAMATH_CALUDE_september_first_was_wednesday_l2290_229075


namespace NUMINAMATH_CALUDE_teachers_in_school_l2290_229056

/-- Calculates the number of teachers required in a school --/
def teachers_required (total_students : ℕ) (lessons_per_student : ℕ) (lessons_per_teacher : ℕ) (students_per_class : ℕ) : ℕ :=
  (total_students * lessons_per_student) / (students_per_class * lessons_per_teacher)

/-- Theorem stating that 50 teachers are required given the specific conditions --/
theorem teachers_in_school : 
  teachers_required 1200 5 4 30 = 50 := by
  sorry

#eval teachers_required 1200 5 4 30

end NUMINAMATH_CALUDE_teachers_in_school_l2290_229056


namespace NUMINAMATH_CALUDE_h_has_one_zero_and_inequality_l2290_229002

noncomputable def f (x : ℝ) := Real.log (x + 1)
noncomputable def g (x : ℝ) := Real.exp x - 1
noncomputable def h (x : ℝ) := f x - g x

theorem h_has_one_zero_and_inequality :
  (∃! x, h x = 0) ∧
  (g (Real.exp 2 - Real.log 2 - 1) > Real.log (Real.exp 2 - Real.log 2)) ∧
  (Real.log (Real.exp 2 - Real.log 2) > 2 - f (Real.log 2)) := by
  sorry

end NUMINAMATH_CALUDE_h_has_one_zero_and_inequality_l2290_229002


namespace NUMINAMATH_CALUDE_mans_swimming_speed_l2290_229046

/-- Proves that a man's swimming speed in still water is 1.5 km/h given the conditions -/
theorem mans_swimming_speed 
  (stream_speed : ℝ) 
  (upstream_time downstream_time : ℝ) 
  (h1 : stream_speed = 0.5)
  (h2 : upstream_time = 2 * downstream_time) : 
  ∃ (still_water_speed : ℝ), still_water_speed = 1.5 :=
by
  sorry

#check mans_swimming_speed

end NUMINAMATH_CALUDE_mans_swimming_speed_l2290_229046


namespace NUMINAMATH_CALUDE_shaded_fraction_of_semicircle_l2290_229011

/-- Given a larger semicircle with diameter 4 and a smaller semicircle removed from it,
    where the two semicircles touch at exactly three points, prove that the fraction
    of the larger semicircle that remains shaded is 1/2. -/
theorem shaded_fraction_of_semicircle (R : ℝ) (r : ℝ) : 
  R = 2 →  -- Radius of larger semicircle
  r^2 + r^2 = (R - r)^2 →  -- Condition for touching at three points
  (π * R^2 / 2 - π * r^2 / 2) / (π * R^2 / 2) = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_shaded_fraction_of_semicircle_l2290_229011


namespace NUMINAMATH_CALUDE_abs_x_less_than_2_sufficient_not_necessary_l2290_229008

theorem abs_x_less_than_2_sufficient_not_necessary :
  (∀ x : ℝ, (|x| < 2 ↔ -2 < x ∧ x < 2)) →
  (∀ x : ℝ, (x^2 - x - 6 < 0 ↔ -2 < x ∧ x < 3)) →
  (∀ x : ℝ, |x| < 2 → x^2 - x - 6 < 0) ∧
  ¬(∀ x : ℝ, x^2 - x - 6 < 0 → |x| < 2) :=
by sorry

end NUMINAMATH_CALUDE_abs_x_less_than_2_sufficient_not_necessary_l2290_229008


namespace NUMINAMATH_CALUDE_distribute_four_teachers_three_schools_l2290_229074

/-- Number of ways to distribute n distinct teachers among k distinct schools,
    with each school receiving at least one teacher -/
def distribute_teachers (n k : ℕ) : ℕ := sorry

/-- The number of ways to distribute 4 distinct teachers among 3 distinct schools,
    with each school receiving at least one teacher, is 36 -/
theorem distribute_four_teachers_three_schools :
  distribute_teachers 4 3 = 36 := by sorry

end NUMINAMATH_CALUDE_distribute_four_teachers_three_schools_l2290_229074


namespace NUMINAMATH_CALUDE_ab_multiplier_l2290_229036

theorem ab_multiplier (a b : ℚ) (h1 : 6 * a = 20) (h2 : 7 * b = 20) : ∃ n : ℚ, n * (a * b) = 800 ∧ n = 84 := by
  sorry

end NUMINAMATH_CALUDE_ab_multiplier_l2290_229036


namespace NUMINAMATH_CALUDE_isosceles_right_triangle_hypotenuse_l2290_229065

theorem isosceles_right_triangle_hypotenuse (square_side : ℝ) (triangle_leg : ℝ) : 
  square_side = 2 →
  4 * (1/2 * triangle_leg^2) = square_side^2 →
  triangle_leg^2 + triangle_leg^2 = 4 := by
  sorry

end NUMINAMATH_CALUDE_isosceles_right_triangle_hypotenuse_l2290_229065


namespace NUMINAMATH_CALUDE_inequality_proof_l2290_229025

theorem inequality_proof (a b c : ℝ) (ha : a ≥ 0) (hb : b ≥ 0) (hc : c ≥ 0) :
  a^4 + b^4 + c^4 - 2*(a^2*b^2 + a^2*c^2 + b^2*c^2) + a^2*b*c + b^2*a*c + c^2*a*b ≥ 0 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l2290_229025


namespace NUMINAMATH_CALUDE_x_plus_y_equals_ten_l2290_229010

theorem x_plus_y_equals_ten (x y : ℝ) 
  (hx : x + Real.log x / Real.log 10 = 10) 
  (hy : y + 10^y = 10) : 
  x + y = 10 := by
sorry

end NUMINAMATH_CALUDE_x_plus_y_equals_ten_l2290_229010


namespace NUMINAMATH_CALUDE_quartic_roots_equivalence_l2290_229003

theorem quartic_roots_equivalence (x : ℂ) : 
  (3 * x^4 + 2 * x^3 - 8 * x^2 + 2 * x + 3 = 0) ↔ 
  (x + 1/x = (-1 + Real.sqrt 43)/3 ∨ x + 1/x = (-1 - Real.sqrt 43)/3) :=
by sorry

end NUMINAMATH_CALUDE_quartic_roots_equivalence_l2290_229003


namespace NUMINAMATH_CALUDE_probability_need_change_is_six_sevenths_l2290_229005

/-- Represents the cost of a toy in cents -/
def ToyCost := Fin 8 → Nat

/-- The machine with 8 toys -/
structure ToyMachine where
  toys : Fin 8
  costs : ToyCost
  favorite_toy_cost : costs 3 = 175  -- $1.75 is the 4th most expensive toy (index 3)

/-- Sam's initial money in quarters -/
def initial_quarters : Nat := 8

/-- Probability of needing to get change -/
def probability_need_change (m : ToyMachine) : Rat :=
  1 - (1 : Rat) / 7

/-- Main theorem: The probability of needing change is 6/7 -/
theorem probability_need_change_is_six_sevenths (m : ToyMachine) :
  probability_need_change m = 6 / 7 := by
  sorry

/-- All costs are between 25 cents and 2 dollars, decreasing by 25 cents each time -/
axiom cost_constraint (m : ToyMachine) :
  ∀ i : Fin 8, m.costs i = 200 - 25 * i.val

/-- The machine randomly selects one of the remaining toys each time -/
axiom random_selection (m : ToyMachine) : True

/-- The machine only accepts quarters -/
axiom quarters_only (m : ToyMachine) : True

end NUMINAMATH_CALUDE_probability_need_change_is_six_sevenths_l2290_229005


namespace NUMINAMATH_CALUDE_cos_power_six_sum_of_squares_l2290_229024

theorem cos_power_six_sum_of_squares :
  ∃ (b₁ b₂ b₃ b₄ b₅ b₆ : ℝ),
    (∀ θ : ℝ, Real.cos θ ^ 6 = b₁ * Real.cos θ + b₂ * Real.cos (2 * θ) + b₃ * Real.cos (3 * θ) +
                              b₄ * Real.cos (4 * θ) + b₅ * Real.cos (5 * θ) + b₆ * Real.cos (6 * θ)) →
    b₁ ^ 2 + b₂ ^ 2 + b₃ ^ 2 + b₄ ^ 2 + b₅ ^ 2 + b₆ ^ 2 = 131 / 128 :=
by sorry

end NUMINAMATH_CALUDE_cos_power_six_sum_of_squares_l2290_229024


namespace NUMINAMATH_CALUDE_tank_dimension_proof_l2290_229029

/-- Proves that the second dimension of a rectangular tank is 5 feet -/
theorem tank_dimension_proof (w : ℝ) : 
  w > 0 → -- w is positive
  4 * w * 3 > 0 → -- tank volume is positive
  2 * (4 * w + 4 * 3 + w * 3) = 1880 / 20 → -- surface area equation
  w = 5 := by
  sorry

end NUMINAMATH_CALUDE_tank_dimension_proof_l2290_229029


namespace NUMINAMATH_CALUDE_factorial_solutions_l2290_229022

def factorial : ℕ → ℕ
  | 0 => 1
  | n + 1 => (n + 1) * factorial n

theorem factorial_solutions :
  ∀ x y : ℕ, factorial x + 2^y = factorial (x + 1) ↔ (x = 1 ∧ y = 0) ∨ (x = 2 ∧ y = 1) := by
  sorry

end NUMINAMATH_CALUDE_factorial_solutions_l2290_229022


namespace NUMINAMATH_CALUDE_number_thought_of_l2290_229085

theorem number_thought_of (x : ℝ) : (6 * x^2 - 10) / 3 + 15 = 95 → x = 5 * Real.sqrt 15 / 3 := by
  sorry

end NUMINAMATH_CALUDE_number_thought_of_l2290_229085


namespace NUMINAMATH_CALUDE_solution_set_when_a_is_neg_three_range_of_a_given_condition_l2290_229020

-- Define the function f
def f (a x : ℝ) : ℝ := |x + a| + |x - 2|

-- Theorem for part 1
theorem solution_set_when_a_is_neg_three :
  {x : ℝ | f (-3) x ≥ 3} = {x : ℝ | x ≤ 1 ∨ x ≥ 4} := by sorry

-- Theorem for part 2
theorem range_of_a_given_condition :
  ∀ a : ℝ, (∀ x ∈ Set.Icc 1 2, f a x ≤ |x - 4|) → a ∈ Set.Icc (-3) 0 := by sorry

end NUMINAMATH_CALUDE_solution_set_when_a_is_neg_three_range_of_a_given_condition_l2290_229020


namespace NUMINAMATH_CALUDE_quadratic_equation_solution_l2290_229057

theorem quadratic_equation_solution (x : ℝ) : x^2 - 5 = 0 ↔ x = Real.sqrt 5 ∨ x = -Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_solution_l2290_229057


namespace NUMINAMATH_CALUDE_right_triangle_from_equation_l2290_229055

theorem right_triangle_from_equation (a b c : ℝ) 
  (h : a^2 + b^2 + c^2 + 338 = 10*a + 24*b + 26*c) : 
  a^2 + b^2 = c^2 := by
sorry

end NUMINAMATH_CALUDE_right_triangle_from_equation_l2290_229055


namespace NUMINAMATH_CALUDE_kim_math_test_probability_l2290_229073

theorem kim_math_test_probability (p : ℚ) (h : p = 4/7) :
  1 - p = 3/7 := by
  sorry

end NUMINAMATH_CALUDE_kim_math_test_probability_l2290_229073


namespace NUMINAMATH_CALUDE_sum_power_mod_five_l2290_229048

theorem sum_power_mod_five (n : ℕ) :
  (1^n + 2^n + 3^n + 4^n) % 5 = 0 ↔ n % 4 ≠ 0 := by
  sorry

end NUMINAMATH_CALUDE_sum_power_mod_five_l2290_229048


namespace NUMINAMATH_CALUDE_circular_segment_probability_l2290_229012

/-- The ratio of circumference to diameter in ancient Chinese mathematics -/
def ancient_pi : ℚ := 3

/-- The area of a circular segment given chord length and height difference -/
def segment_area (a c : ℚ) : ℚ := (1/2) * a * (a + c)

/-- The probability of a point mass landing in a circular segment -/
theorem circular_segment_probability (c a : ℚ) (h1 : c = 6) (h2 : a = 1) :
  let r := (c^2 / 4 + a^2) / (2 * a)
  let circle_area := ancient_pi * r^2
  segment_area a c / circle_area = 7 / 150 := by
  sorry

end NUMINAMATH_CALUDE_circular_segment_probability_l2290_229012


namespace NUMINAMATH_CALUDE_bottle_production_l2290_229098

/-- Given that 5 identical machines produce 270 bottles per minute at a constant rate,
    prove that 10 such machines will produce 2160 bottles in 4 minutes. -/
theorem bottle_production
  (rate : ℕ → ℕ → ℕ) -- rate function: number of machines → time in minutes → number of bottles
  (h1 : rate 5 1 = 270) -- 5 machines produce 270 bottles in 1 minute
  (h2 : ∀ n t, rate n t = n * t * (rate 5 1 / 5)) -- linear scaling of production
  : rate 10 4 = 2160 := by
  sorry


end NUMINAMATH_CALUDE_bottle_production_l2290_229098


namespace NUMINAMATH_CALUDE_diagonal_contains_all_numbers_l2290_229079

theorem diagonal_contains_all_numbers (n : ℕ) (h_odd : Odd n) 
  (grid : Fin n → Fin n → Fin n)
  (h_row : ∀ i j k, i ≠ k → grid i j ≠ grid k j)
  (h_col : ∀ i j k, j ≠ k → grid i j ≠ grid i k)
  (h_sym : ∀ i j, grid i j = grid j i) :
  ∀ k : Fin n, ∃ i : Fin n, grid i i = k := by
sorry

end NUMINAMATH_CALUDE_diagonal_contains_all_numbers_l2290_229079


namespace NUMINAMATH_CALUDE_additional_grazing_area_l2290_229051

theorem additional_grazing_area (π : ℝ) (h : π > 0) : 
  π * 23^2 - π * 16^2 = 273 * π := by
  sorry

end NUMINAMATH_CALUDE_additional_grazing_area_l2290_229051


namespace NUMINAMATH_CALUDE_averageIs295_l2290_229038

/-- Calculates the average number of visitors per day in a 30-day month starting on a Sunday -/
def averageVisitorsPerDay (sundayVisitors : ℕ) (otherDayVisitors : ℕ) : ℚ :=
  let totalSundays : ℕ := 5
  let totalOtherDays : ℕ := 25
  let totalVisitors : ℕ := sundayVisitors * totalSundays + otherDayVisitors * totalOtherDays
  totalVisitors / 30

/-- Theorem stating that the average number of visitors per day is 295 -/
theorem averageIs295 (sundayVisitors : ℕ) (otherDayVisitors : ℕ) 
    (h1 : sundayVisitors = 570) (h2 : otherDayVisitors = 240) : 
    averageVisitorsPerDay sundayVisitors otherDayVisitors = 295 := by
  sorry

end NUMINAMATH_CALUDE_averageIs295_l2290_229038


namespace NUMINAMATH_CALUDE_family_ages_l2290_229066

theorem family_ages (man son daughter : ℕ) : 
  man = son + 46 →
  man + 2 = 2 * (son + 2) →
  daughter = son - 4 →
  son + daughter = 84 := by
sorry

end NUMINAMATH_CALUDE_family_ages_l2290_229066


namespace NUMINAMATH_CALUDE_xyz_sum_l2290_229086

theorem xyz_sum (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0)
  (hxy : x * y = 24) (hxz : x * z = 48) (hyz : y * z = 72) :
  x + y + z = 22 := by
  sorry

end NUMINAMATH_CALUDE_xyz_sum_l2290_229086


namespace NUMINAMATH_CALUDE_excess_donation_l2290_229064

/-- Trader's profit calculation -/
def trader_profit : ℝ := 1200

/-- Allocation percentage for next shipment -/
def allocation_percentage : ℝ := 0.60

/-- Family donation amount -/
def family_donation : ℝ := 250

/-- Friends donation calculation -/
def friends_donation : ℝ := family_donation * 1.20

/-- Local association donation calculation -/
def local_association_donation : ℝ := (family_donation + friends_donation) * 1.5

/-- Total donations received -/
def total_donations : ℝ := family_donation + friends_donation + local_association_donation

/-- Allocated amount for next shipment -/
def allocated_amount : ℝ := trader_profit * allocation_percentage

/-- Theorem: The difference between total donations and allocated amount is $655 -/
theorem excess_donation : total_donations - allocated_amount = 655 := by sorry

end NUMINAMATH_CALUDE_excess_donation_l2290_229064


namespace NUMINAMATH_CALUDE_remainder_55_power_55_plus_10_mod_8_l2290_229052

theorem remainder_55_power_55_plus_10_mod_8 : 55^55 + 10 ≡ 1 [ZMOD 8] := by
  sorry

end NUMINAMATH_CALUDE_remainder_55_power_55_plus_10_mod_8_l2290_229052


namespace NUMINAMATH_CALUDE_batman_game_cost_batman_game_cost_proof_l2290_229077

def total_spent : ℝ := 35.52
def football_cost : ℝ := 14.02
def strategy_cost : ℝ := 9.46

theorem batman_game_cost : ℝ := by
  sorry

theorem batman_game_cost_proof : batman_game_cost = 12.04 := by
  sorry

end NUMINAMATH_CALUDE_batman_game_cost_batman_game_cost_proof_l2290_229077


namespace NUMINAMATH_CALUDE_repeating_decimal_equals_fraction_l2290_229032

/-- The repeating decimal 0.137137137... -/
def repeating_decimal : ℚ := 0.137137137

/-- The fraction 137/999 -/
def fraction : ℚ := 137 / 999

/-- Theorem stating that the repeating decimal 0.137137137... is equal to the fraction 137/999 -/
theorem repeating_decimal_equals_fraction : repeating_decimal = fraction := by
  sorry

end NUMINAMATH_CALUDE_repeating_decimal_equals_fraction_l2290_229032


namespace NUMINAMATH_CALUDE_train_length_l2290_229043

/-- The length of a train given its speed and time to cross a pole -/
theorem train_length (speed_kmh : ℝ) (time_s : ℝ) : 
  speed_kmh = 48 → time_s = 9 → ∃ length_m : ℝ, abs (length_m - 119.97) < 0.01 := by
  sorry

end NUMINAMATH_CALUDE_train_length_l2290_229043


namespace NUMINAMATH_CALUDE_min_value_triangle_ratio_l2290_229018

/-- Given a triangle ABC with sides a, b, c opposite to angles A, B, C respectively,
    if c - a = 2a * cos B, then the minimum possible value of (3a + c) / b is 2√2. -/
theorem min_value_triangle_ratio (a b c : ℝ) (A B C : ℝ) :
  a > 0 → b > 0 → c > 0 →
  A > 0 → B > 0 → C > 0 →
  A + B + C = π →
  c - a = 2 * a * Real.cos B →
  ∃ (m : ℝ), m = 2 * Real.sqrt 2 ∧ ∀ (x : ℝ), (3 * a + c) / b ≥ m :=
by sorry

end NUMINAMATH_CALUDE_min_value_triangle_ratio_l2290_229018


namespace NUMINAMATH_CALUDE_vector_equality_properties_l2290_229068

variable {E : Type*} [NormedAddCommGroup E] [InnerProductSpace ℝ E]

def same_direction (a b : E) : Prop :=
  ∃ (k : ℝ), k > 0 ∧ a = k • b

theorem vector_equality_properties (a b : E) (ha : a ≠ 0) (hb : b ≠ 0) (heq : a = b) :
  same_direction a b ∧ ‖a‖ = ‖b‖ := by sorry

end NUMINAMATH_CALUDE_vector_equality_properties_l2290_229068


namespace NUMINAMATH_CALUDE_ways_to_go_home_via_library_l2290_229047

theorem ways_to_go_home_via_library (school_to_library : ℕ) (library_to_home : ℕ) : 
  school_to_library = 2 → library_to_home = 3 → school_to_library * library_to_home = 6 := by
  sorry

end NUMINAMATH_CALUDE_ways_to_go_home_via_library_l2290_229047


namespace NUMINAMATH_CALUDE_limit_S_over_a_squared_ln_a_nonzero_l2290_229087

/-- The area S(a) bounded by the curve y = (a-x)ln x and the x-axis for a > 1 -/
noncomputable def S (a : ℝ) : ℝ := ∫ x in (1)..(a), (a - x) * Real.log x

/-- The limit of S(a)/(a^2 ln a) as a approaches infinity is a non-zero real number -/
theorem limit_S_over_a_squared_ln_a_nonzero :
  ∃ (L : ℝ), L ≠ 0 ∧ Filter.Tendsto (fun a => S a / (a^2 * Real.log a)) Filter.atTop (nhds L) := by
  sorry

end NUMINAMATH_CALUDE_limit_S_over_a_squared_ln_a_nonzero_l2290_229087


namespace NUMINAMATH_CALUDE_work_completion_time_l2290_229058

theorem work_completion_time 
  (a_time b_time c_time : ℝ) 
  (ha : a_time = 8) 
  (hb : b_time = 12) 
  (hc : c_time = 24) : 
  1 / (1 / a_time + 1 / b_time + 1 / c_time) = 4 := by
  sorry

end NUMINAMATH_CALUDE_work_completion_time_l2290_229058


namespace NUMINAMATH_CALUDE_reciprocal_of_mixed_number_l2290_229096

def mixed_number_to_fraction (whole : ℤ) (numerator : ℤ) (denominator : ℤ) : ℚ :=
  (whole * denominator + numerator) / denominator

def reciprocal (x : ℚ) : ℚ := 1 / x

theorem reciprocal_of_mixed_number :
  let original : ℚ := mixed_number_to_fraction (-1) 2 3
  let recip : ℚ := -3 / 5
  (reciprocal original = recip) ∧ (original * recip = 1) := by sorry

end NUMINAMATH_CALUDE_reciprocal_of_mixed_number_l2290_229096


namespace NUMINAMATH_CALUDE_soup_per_bag_is_three_l2290_229083

-- Define the quantities
def milk_quarts : ℚ := 2
def vegetable_quarts : ℚ := 1
def num_bags : ℕ := 3

-- Define the relationship between milk and chicken stock
def chicken_stock_quarts : ℚ := 3 * milk_quarts

-- Calculate the total amount of soup
def total_soup : ℚ := milk_quarts + chicken_stock_quarts + vegetable_quarts

-- Define the amount of soup per bag
def soup_per_bag : ℚ := total_soup / num_bags

-- Theorem to prove
theorem soup_per_bag_is_three : soup_per_bag = 3 := by
  sorry

end NUMINAMATH_CALUDE_soup_per_bag_is_three_l2290_229083


namespace NUMINAMATH_CALUDE_right_triangle_with_given_sides_l2290_229072

theorem right_triangle_with_given_sides :
  ∃ (a b c : ℝ), a = 8 ∧ b = 15 ∧ c = Real.sqrt 161 ∧ a^2 + b^2 = c^2 :=
by sorry

end NUMINAMATH_CALUDE_right_triangle_with_given_sides_l2290_229072


namespace NUMINAMATH_CALUDE_rod_cutting_l2290_229015

theorem rod_cutting (rod_length : ℝ) (total_pieces : ℝ) (piece_length : ℝ) : 
  rod_length = 47.5 →
  total_pieces = 118.75 →
  piece_length = rod_length / total_pieces →
  piece_length = 0.4 := by
sorry

end NUMINAMATH_CALUDE_rod_cutting_l2290_229015


namespace NUMINAMATH_CALUDE_mango_rate_is_55_l2290_229033

-- Define the given quantities
def grapeQuantity : ℕ := 8
def grapeRate : ℕ := 70
def mangoQuantity : ℕ := 9
def totalPaid : ℕ := 1055

-- Define the function to calculate the mango rate
def mangoRate : ℕ := (totalPaid - grapeQuantity * grapeRate) / mangoQuantity

-- Theorem to prove
theorem mango_rate_is_55 : mangoRate = 55 := by
  sorry

end NUMINAMATH_CALUDE_mango_rate_is_55_l2290_229033


namespace NUMINAMATH_CALUDE_pencil_count_l2290_229049

/-- The number of pencils Mitchell and Antonio have together -/
def total_pencils (mitchell_pencils : ℕ) (difference : ℕ) : ℕ :=
  mitchell_pencils + (mitchell_pencils - difference)

/-- Theorem stating the total number of pencils Mitchell and Antonio have -/
theorem pencil_count (mitchell_pencils : ℕ) (difference : ℕ) 
  (h1 : mitchell_pencils = 30)
  (h2 : difference = 6) : 
  total_pencils mitchell_pencils difference = 54 := by
  sorry

end NUMINAMATH_CALUDE_pencil_count_l2290_229049


namespace NUMINAMATH_CALUDE_convention_handshakes_l2290_229040

/-- The number of companies at the convention -/
def num_companies : ℕ := 5

/-- The number of representatives from each company -/
def reps_per_company : ℕ := 5

/-- The total number of people at the convention -/
def total_people : ℕ := num_companies * reps_per_company

/-- The number of people each person shakes hands with -/
def handshakes_per_person : ℕ := total_people - reps_per_company - 1

/-- The total number of handshakes at the convention -/
def total_handshakes : ℕ := (total_people * handshakes_per_person) / 2

theorem convention_handshakes :
  total_handshakes = 250 :=
by sorry

end NUMINAMATH_CALUDE_convention_handshakes_l2290_229040


namespace NUMINAMATH_CALUDE_complex_equality_l2290_229070

theorem complex_equality (z : ℂ) : z = -1 + I ↔ Complex.abs (z - 2) = Complex.abs (z + 4) ∧ Complex.abs (z - 2) = Complex.abs (z - 2*I) := by
  sorry

end NUMINAMATH_CALUDE_complex_equality_l2290_229070


namespace NUMINAMATH_CALUDE_salary_calculation_l2290_229099

def net_monthly_salary (discretionary_income : ℝ) (remaining_amount : ℝ) : Prop :=
  let vacation_fund_percent : ℝ := 0.30
  let savings_percent : ℝ := 0.20
  let socializing_percent : ℝ := 0.35
  let total_allocated_percent : ℝ := vacation_fund_percent + savings_percent + socializing_percent
  let remaining_percent : ℝ := 1 - total_allocated_percent
  discretionary_income * remaining_percent = remaining_amount ∧
  discretionary_income * 5 = 3400

theorem salary_calculation (discretionary_income : ℝ) (remaining_amount : ℝ) :
  discretionary_income = remaining_amount / 0.15 →
  net_monthly_salary discretionary_income remaining_amount :=
by
  sorry

#check salary_calculation 680 102

end NUMINAMATH_CALUDE_salary_calculation_l2290_229099


namespace NUMINAMATH_CALUDE_unique_solution_l2290_229089

/-- Represents the number of vehicles of each type Jeff has -/
structure VehicleCounts where
  trucks : ℕ
  cars : ℕ
  motorcycles : ℕ
  buses : ℕ

/-- Checks if the given vehicle counts satisfy all the conditions -/
def satisfiesConditions (v : VehicleCounts) : Prop :=
  v.cars = 2 * v.trucks ∧
  v.motorcycles = 3 * v.cars ∧
  v.buses = v.trucks / 2 ∧
  v.trucks + v.cars + v.motorcycles + v.buses = 180

/-- The theorem stating that the given vehicle counts are the unique solution -/
theorem unique_solution : 
  ∃! v : VehicleCounts, satisfiesConditions v ∧ 
    v.trucks = 19 ∧ v.cars = 38 ∧ v.motorcycles = 114 ∧ v.buses = 9 :=
by sorry

end NUMINAMATH_CALUDE_unique_solution_l2290_229089


namespace NUMINAMATH_CALUDE_souvenir_sales_theorem_l2290_229080

/-- Represents the souvenir sales scenario -/
structure SouvenirSales where
  purchase_price : ℝ
  base_selling_price : ℝ
  base_daily_sales : ℝ
  price_sales_ratio : ℝ

/-- Calculates the daily sales quantity for a given selling price -/
def daily_sales (s : SouvenirSales) (selling_price : ℝ) : ℝ :=
  s.base_daily_sales - s.price_sales_ratio * (selling_price - s.base_selling_price)

/-- Calculates the daily profit for a given selling price -/
def daily_profit (s : SouvenirSales) (selling_price : ℝ) : ℝ :=
  (selling_price - s.purchase_price) * (daily_sales s selling_price)

/-- The main theorem about the souvenir sales scenario -/
theorem souvenir_sales_theorem (s : SouvenirSales) 
  (h1 : s.purchase_price = 40)
  (h2 : s.base_selling_price = 50)
  (h3 : s.base_daily_sales = 200)
  (h4 : s.price_sales_ratio = 10) :
  (daily_sales s 52 = 180) ∧ 
  (∃ x : ℝ, ∀ y : ℝ, daily_profit s x ≥ daily_profit s y) ∧
  (daily_profit s 55 = 2250) := by
  sorry

#check souvenir_sales_theorem

end NUMINAMATH_CALUDE_souvenir_sales_theorem_l2290_229080


namespace NUMINAMATH_CALUDE_unique_solution_for_x_l2290_229014

theorem unique_solution_for_x (x y z : ℤ) 
  (h1 : x > y ∧ y > z ∧ z > 0)
  (h2 : x + y + z + x*y + y*z + z*x = 104) : 
  x = 6 := by sorry

end NUMINAMATH_CALUDE_unique_solution_for_x_l2290_229014


namespace NUMINAMATH_CALUDE_average_side_lengths_of_squares_l2290_229037

theorem average_side_lengths_of_squares (a₁ a₂ a₃ : ℝ) 
  (h₁ : a₁ = 36) (h₂ : a₂ = 64) (h₃ : a₃ = 144) :
  (Real.sqrt a₁ + Real.sqrt a₂ + Real.sqrt a₃) / 3 = 26 / 3 := by
  sorry

end NUMINAMATH_CALUDE_average_side_lengths_of_squares_l2290_229037


namespace NUMINAMATH_CALUDE_unique_square_divisible_by_five_l2290_229030

theorem unique_square_divisible_by_five (y : ℕ) : 
  (∃ n : ℕ, y = n^2) ∧ 
  y % 5 = 0 ∧ 
  50 < y ∧ 
  y < 120 → 
  y = 100 := by
sorry

end NUMINAMATH_CALUDE_unique_square_divisible_by_five_l2290_229030


namespace NUMINAMATH_CALUDE_photo_arrangement_l2290_229035

/-- The number of ways to select and permute 3 people out of 8, keeping the rest in place -/
theorem photo_arrangement (n m : ℕ) (hn : n = 8) (hm : m = 3) : 
  (n.choose m) * (Nat.factorial m) = 336 := by
  sorry

end NUMINAMATH_CALUDE_photo_arrangement_l2290_229035


namespace NUMINAMATH_CALUDE_cos_330_deg_l2290_229069

/-- Cosine of 330 degrees is equal to √3/2 -/
theorem cos_330_deg : Real.cos (330 * π / 180) = Real.sqrt 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_cos_330_deg_l2290_229069


namespace NUMINAMATH_CALUDE_class_representative_count_l2290_229004

theorem class_representative_count (male_students female_students : ℕ) :
  male_students = 26 → female_students = 24 →
  male_students + female_students = 50 :=
by sorry

end NUMINAMATH_CALUDE_class_representative_count_l2290_229004


namespace NUMINAMATH_CALUDE_alternating_sum_equals_eight_l2290_229023

theorem alternating_sum_equals_eight :
  43 - 41 + 39 - 37 + 35 - 33 + 31 - 29 = 8 := by
  sorry

end NUMINAMATH_CALUDE_alternating_sum_equals_eight_l2290_229023


namespace NUMINAMATH_CALUDE_neither_necessary_nor_sufficient_l2290_229042

open Real

/-- A function f is increasing on (0,∞) -/
def IsIncreasing (f : ℝ → ℝ) : Prop :=
  ∀ x y, 0 < x ∧ x < y → f x < f y

theorem neither_necessary_nor_sufficient :
  ∃ (f₁ f₂ : ℝ → ℝ),
    (∀ x, x > 0 → f₁ x ≠ 0 ∧ f₂ x ≠ 0) ∧
    IsIncreasing f₁ ∧
    ¬IsIncreasing (fun x ↦ x * f₁ x) ∧
    ¬IsIncreasing f₂ ∧
    IsIncreasing (fun x ↦ x * f₂ x) :=
by sorry

end NUMINAMATH_CALUDE_neither_necessary_nor_sufficient_l2290_229042


namespace NUMINAMATH_CALUDE_quadratic_square_completion_l2290_229063

theorem quadratic_square_completion (p q : ℤ) : 
  (∀ x : ℝ, x^2 - 6*x + 3 = 0 ↔ (x + p)^2 = q) → p + q = 3 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_square_completion_l2290_229063


namespace NUMINAMATH_CALUDE_fraction_simplification_l2290_229001

theorem fraction_simplification (x y : ℝ) (h : x / y = 2 / 5) :
  (3 * y - 2 * x) / (3 * y + 2 * x) = 11 / 19 := by
  sorry

end NUMINAMATH_CALUDE_fraction_simplification_l2290_229001


namespace NUMINAMATH_CALUDE_horner_method_v3_l2290_229019

def horner_polynomial (x : ℤ) : ℤ := 12 + 35*x - 8*x^2 + 6*x^4 + 5*x^5 + 3*x^6

def horner_v0 : ℤ := 3
def horner_v1 (x : ℤ) : ℤ := horner_v0 * x + 5
def horner_v2 (x : ℤ) : ℤ := horner_v1 x * x + 6
def horner_v3 (x : ℤ) : ℤ := horner_v2 x * x + 0

theorem horner_method_v3 :
  horner_v3 (-4) = -57 :=
sorry

end NUMINAMATH_CALUDE_horner_method_v3_l2290_229019


namespace NUMINAMATH_CALUDE_intersection_line_l2290_229094

-- Define the two circles
def circle1 (x y : ℝ) : Prop := x^2 + y^2 + 6*x + 4*y = 0
def circle2 (x y : ℝ) : Prop := x^2 + y^2 + 4*x + 2*y - 4 = 0

-- Define the line
def line (x y : ℝ) : Prop := x + y - 2 = 0

-- Theorem statement
theorem intersection_line :
  ∀ x y : ℝ, circle1 x y ∧ circle2 x y → line x y :=
by sorry

end NUMINAMATH_CALUDE_intersection_line_l2290_229094


namespace NUMINAMATH_CALUDE_geometric_sequence_ratio_l2290_229062

-- Define a geometric sequence
def geometric_sequence (a : ℕ → ℝ) (q : ℝ) : Prop :=
  ∀ n : ℕ, a (n + 1) = a n * q

-- State the theorem
theorem geometric_sequence_ratio 
  (a : ℕ → ℝ) (q : ℝ) 
  (h_geom : geometric_sequence a q)
  (h_a1 : a 1 = 4)
  (h_a4 : a 4 = 1/2) :
  q = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_ratio_l2290_229062


namespace NUMINAMATH_CALUDE_sides_when_k_is_two_k_values_l2290_229016

/-- Represents a regular pyramid -/
structure RegularPyramid where
  n : ℕ  -- number of sides of the base
  α : ℝ  -- dihedral angle at the base
  β : ℝ  -- angle formed by lateral edges with the base plane
  k : ℝ  -- relationship constant between α and β
  h1 : α > 0
  h2 : β > 0
  h3 : k > 0
  h4 : n ≥ 3
  h5 : Real.tan α = k * Real.tan β
  h6 : k = 1 / Real.cos (π / n)

/-- The number of sides of the base is 3 when k = 2 -/
theorem sides_when_k_is_two (p : RegularPyramid) : p.k = 2 → p.n = 3 := by sorry

/-- The possible values of k are given by 1 / cos(π/n) where n ≥ 3 -/
theorem k_values (p : RegularPyramid) : 
  ∃ (n : ℕ), n ≥ 3 ∧ p.k = 1 / Real.cos (π / n) := by sorry

end NUMINAMATH_CALUDE_sides_when_k_is_two_k_values_l2290_229016


namespace NUMINAMATH_CALUDE_juice_cans_for_eight_litres_l2290_229082

/-- Given that three cans of juice fill 2/3 of a one-litre jug, 
    prove that 36 cans are needed to completely fill 8 one-litre jugs. -/
theorem juice_cans_for_eight_litres :
  let cans_per_two_thirds_litre : ℚ := 3
  let litres_to_fill : ℚ := 8
  let total_cans : ℚ := 36
  (cans_per_two_thirds_litre * (2/3) = 1) →
  (litres_to_fill * total_cans / cans_per_two_thirds_litre = litres_to_fill / (2/3)) :=
by sorry

end NUMINAMATH_CALUDE_juice_cans_for_eight_litres_l2290_229082


namespace NUMINAMATH_CALUDE_circle_center_and_radius_circle_properties_l2290_229031

theorem circle_center_and_radius 
  (x y : ℝ) : 
  x^2 + y^2 + 4*x - 6*y = 11 ↔ 
  (x + 2)^2 + (y - 3)^2 = 24 :=
by sorry

theorem circle_properties : 
  ∃ (center : ℝ × ℝ) (radius : ℝ), 
  center = (-2, 3) ∧ 
  radius = 2 * Real.sqrt 6 ∧
  ∀ (x y : ℝ), x^2 + y^2 + 4*x - 6*y = 11 ↔ 
  (x - center.1)^2 + (y - center.2)^2 = radius^2 :=
by sorry

end NUMINAMATH_CALUDE_circle_center_and_radius_circle_properties_l2290_229031


namespace NUMINAMATH_CALUDE_point_in_first_or_third_quadrant_l2290_229078

/-- A point is in the first or third quadrant if the product of its coordinates is positive -/
theorem point_in_first_or_third_quadrant (x y : ℝ) :
  x * y > 0 → (x > 0 ∧ y > 0) ∨ (x < 0 ∧ y < 0) :=
by sorry

end NUMINAMATH_CALUDE_point_in_first_or_third_quadrant_l2290_229078


namespace NUMINAMATH_CALUDE_unique_c_complex_magnitude_l2290_229076

theorem unique_c_complex_magnitude : ∃! c : ℝ, Complex.abs (1 - (c + 1) * Complex.I) = 1 := by
  sorry

end NUMINAMATH_CALUDE_unique_c_complex_magnitude_l2290_229076


namespace NUMINAMATH_CALUDE_solution_set_inequality_l2290_229097

theorem solution_set_inequality (x : ℝ) : 
  (x - 2) * (3 - x) > 0 ↔ x ∈ Set.Ioo 2 3 := by sorry

end NUMINAMATH_CALUDE_solution_set_inequality_l2290_229097


namespace NUMINAMATH_CALUDE_number_equation_solution_l2290_229026

theorem number_equation_solution : ∃ x : ℝ, 2 * x - 3 = 7 ∧ x = 5 := by
  sorry

end NUMINAMATH_CALUDE_number_equation_solution_l2290_229026
