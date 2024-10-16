import Mathlib

namespace NUMINAMATH_CALUDE_amusement_park_ride_orders_l1584_158447

theorem amusement_park_ride_orders : Nat.factorial 6 = 720 := by
  sorry

end NUMINAMATH_CALUDE_amusement_park_ride_orders_l1584_158447


namespace NUMINAMATH_CALUDE_num_valid_schedules_l1584_158486

/-- Represents the number of periods in a day -/
def num_periods : ℕ := 8

/-- Represents the number of courses to be scheduled -/
def num_courses : ℕ := 4

/-- 
Calculates the number of ways to schedule courses with exactly one consecutive pair
num_periods: The total number of periods in a day
num_courses: The number of courses to be scheduled
-/
def schedule_with_one_consecutive_pair (num_periods : ℕ) (num_courses : ℕ) : ℕ := sorry

/-- The main theorem stating the number of valid schedules -/
theorem num_valid_schedules : 
  schedule_with_one_consecutive_pair num_periods num_courses = 1680 := by sorry

end NUMINAMATH_CALUDE_num_valid_schedules_l1584_158486


namespace NUMINAMATH_CALUDE_boris_climbs_needed_l1584_158413

def hugo_elevation : ℕ := 10000
def boris_elevation : ℕ := hugo_elevation - 2500
def hugo_climbs : ℕ := 3

theorem boris_climbs_needed : 
  (hugo_elevation * hugo_climbs) / boris_elevation = 4 := by sorry

end NUMINAMATH_CALUDE_boris_climbs_needed_l1584_158413


namespace NUMINAMATH_CALUDE_mikes_seashells_l1584_158499

/-- Given that Joan initially found 79 seashells and has 142 seashells in total after Mike gave her some,
    prove that Mike gave Joan 63 seashells. -/
theorem mikes_seashells (joans_initial : ℕ) (joans_total : ℕ) (mikes_gift : ℕ)
    (h1 : joans_initial = 79)
    (h2 : joans_total = 142)
    (h3 : joans_total = joans_initial + mikes_gift) :
  mikes_gift = 63 := by
  sorry

end NUMINAMATH_CALUDE_mikes_seashells_l1584_158499


namespace NUMINAMATH_CALUDE_pizza_total_slices_l1584_158445

def pizza_problem (john_slices sam_slices remaining_slices : ℕ) : Prop :=
  john_slices = 3 ∧
  sam_slices = 2 * john_slices ∧
  remaining_slices = 3

theorem pizza_total_slices 
  (john_slices sam_slices remaining_slices : ℕ) 
  (h : pizza_problem john_slices sam_slices remaining_slices) : 
  john_slices + sam_slices + remaining_slices = 12 :=
by
  sorry

#check pizza_total_slices

end NUMINAMATH_CALUDE_pizza_total_slices_l1584_158445


namespace NUMINAMATH_CALUDE_rectangle_count_l1584_158420

/-- A point in a 2D plane -/
structure Point2D where
  x : ℝ
  y : ℝ

/-- A rectangle in a 2D plane -/
structure Rectangle where
  A : Point2D
  B : Point2D
  C : Point2D
  D : Point2D

/-- Check if a point lies on a line segment between two other points -/
def pointOnSegment (P Q R : Point2D) : Prop := sorry

/-- Check if two line segments are perpendicular -/
def perpendicular (P Q R S : Point2D) : Prop := sorry

/-- Check if a line segment forms a 30° angle with another line segment -/
def angle30Degrees (P Q R S : Point2D) : Prop := sorry

/-- Check if a rectangle satisfies the given conditions -/
def validRectangle (rect : Rectangle) (P1 P2 P3 : Point2D) : Prop :=
  (rect.A = P1 ∨ rect.A = P2 ∨ rect.A = P3) ∧
  (pointOnSegment P1 rect.A rect.B ∨ pointOnSegment P2 rect.A rect.B ∨ pointOnSegment P3 rect.A rect.B) ∧
  (pointOnSegment P1 rect.B rect.C ∨ pointOnSegment P2 rect.B rect.C ∨ pointOnSegment P3 rect.B rect.C) ∧
  (pointOnSegment P1 rect.C rect.D ∨ pointOnSegment P2 rect.C rect.D ∨ pointOnSegment P3 rect.C rect.D) ∧
  (pointOnSegment P1 rect.D rect.A ∨ pointOnSegment P2 rect.D rect.A ∨ pointOnSegment P3 rect.D rect.A) ∧
  perpendicular rect.A rect.B rect.B rect.C ∧
  perpendicular rect.B rect.C rect.C rect.D ∧
  (angle30Degrees rect.A rect.C rect.A rect.B ∨ angle30Degrees rect.A rect.C rect.C rect.D)

theorem rectangle_count (P1 P2 P3 : Point2D) 
  (h_distinct : P1 ≠ P2 ∧ P2 ≠ P3 ∧ P1 ≠ P3) :
  ∃! (s : Finset Rectangle), (∀ rect ∈ s, validRectangle rect P1 P2 P3) ∧ s.card = 60 :=
sorry

end NUMINAMATH_CALUDE_rectangle_count_l1584_158420


namespace NUMINAMATH_CALUDE_jonas_library_space_l1584_158474

theorem jonas_library_space (total_space : ℝ) (shelf_space : ℝ) (num_shelves : ℕ) 
  (h1 : total_space = 400)
  (h2 : shelf_space = 80)
  (h3 : num_shelves = 3) :
  total_space - (↑num_shelves * shelf_space) = 160 := by
sorry

end NUMINAMATH_CALUDE_jonas_library_space_l1584_158474


namespace NUMINAMATH_CALUDE_exact_arrival_speed_l1584_158437

theorem exact_arrival_speed 
  (d : ℝ) (t : ℝ) 
  (h1 : d = 30 * (t + 1/12)) 
  (h2 : d = 70 * (t - 1/12)) : 
  d / t = 42 := by
sorry

end NUMINAMATH_CALUDE_exact_arrival_speed_l1584_158437


namespace NUMINAMATH_CALUDE_area_between_concentric_circles_l1584_158435

theorem area_between_concentric_circles 
  (r : ℝ)  -- radius of inner circle
  (h1 : r > 0)  -- radius is positive
  (h2 : 3*r - r = 3)  -- width of gray region is 3
  : π * (3*r)^2 - π * r^2 = 18 * π := by
  sorry

end NUMINAMATH_CALUDE_area_between_concentric_circles_l1584_158435


namespace NUMINAMATH_CALUDE_product_of_divisors_1024_l1584_158468

/-- The product of divisors of a positive integer -/
def product_of_divisors (n : ℕ+) : ℕ := sorry

/-- Theorem: If the product of divisors of n is 1024, then n = 16 -/
theorem product_of_divisors_1024 (n : ℕ+) :
  product_of_divisors n = 1024 → n = 16 := by sorry

end NUMINAMATH_CALUDE_product_of_divisors_1024_l1584_158468


namespace NUMINAMATH_CALUDE_total_triangles_is_200_l1584_158444

/-- Represents an equilateral triangle -/
structure EquilateralTriangle where
  sideLength : ℕ

/-- Represents the large equilateral triangle -/
def largeTriangle : EquilateralTriangle :=
  { sideLength := 10 }

/-- Represents a small equilateral triangle -/
def smallTriangle : EquilateralTriangle :=
  { sideLength := 1 }

/-- The number of small triangles that fit in the large triangle -/
def numSmallTriangles : ℕ := 100

/-- Counts the number of equilateral triangles of a given side length -/
def countTriangles (sideLength : ℕ) : ℕ :=
  if sideLength = 1 then numSmallTriangles
  else if sideLength > largeTriangle.sideLength then 0
  else largeTriangle.sideLength - sideLength + 1

/-- The total number of equilateral triangles -/
def totalTriangles : ℕ :=
  (List.range largeTriangle.sideLength).map countTriangles |>.sum

theorem total_triangles_is_200 : totalTriangles = 200 := by
  sorry

end NUMINAMATH_CALUDE_total_triangles_is_200_l1584_158444


namespace NUMINAMATH_CALUDE_leila_time_allocation_l1584_158401

/-- Leila's utility function --/
def utility (juggling_hours coding_hours : ℝ) : ℝ := juggling_hours * coding_hours

/-- Leila's time allocation problem --/
theorem leila_time_allocation (s : ℝ) : 
  utility s (12 - s) = utility (6 - s) (s + 4) → s = 12 / 5 := by
  sorry

end NUMINAMATH_CALUDE_leila_time_allocation_l1584_158401


namespace NUMINAMATH_CALUDE_expression_evaluation_l1584_158481

theorem expression_evaluation : 3 * (-3)^4 + 2 * (-3)^3 + (-3)^2 + 3^2 + 2 * 3^3 + 3 * 3^4 = 504 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l1584_158481


namespace NUMINAMATH_CALUDE_inequality_solution_set_l1584_158448

theorem inequality_solution_set (a b : ℝ) : 
  (∀ x : ℝ, (x > 4 ∧ x < b) ↔ (Real.sqrt x > a * x + 3/2)) →
  (a = 1/8 ∧ b = 36) := by
sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l1584_158448


namespace NUMINAMATH_CALUDE_sequence_is_geometric_from_second_term_l1584_158416

def is_geometric_from_second_term (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, n ≥ 2 → a (n + 1) = r * a n

theorem sequence_is_geometric_from_second_term
  (a : ℕ → ℝ)
  (S : ℕ → ℝ)
  (h1 : S 1 = 1)
  (h2 : S 2 = 2)
  (h3 : ∀ n : ℕ, n ≥ 2 → S (n + 1) - 3 * S n + 2 * S (n - 1) = 0)
  (h4 : ∀ n : ℕ, S (n + 1) - S n = a (n + 1))
  : is_geometric_from_second_term a :=
by
  sorry

#check sequence_is_geometric_from_second_term

end NUMINAMATH_CALUDE_sequence_is_geometric_from_second_term_l1584_158416


namespace NUMINAMATH_CALUDE_sqrt_16_minus_pi_minus_3_pow_0_l1584_158407

theorem sqrt_16_minus_pi_minus_3_pow_0 : Real.sqrt 16 - (π - 3)^0 = 3 := by sorry

end NUMINAMATH_CALUDE_sqrt_16_minus_pi_minus_3_pow_0_l1584_158407


namespace NUMINAMATH_CALUDE_min_cone_cylinder_volume_ratio_l1584_158460

/-- The minimum ratio of the volume of a cone circumscribed around a sphere
    to the volume of a cylinder circumscribed around the same sphere -/
theorem min_cone_cylinder_volume_ratio (r : ℝ) (hr : r > 0) :
  ∃ (V₁ V₂ : ℝ),
    (∀ (Vc Vn : ℝ),
      (Vc = 2 * π * r^3) →  -- Volume of circumscribed cylinder
      (∃ (R m : ℝ), Vn = (1/3) * π * R^2 * m ∧ 
        R^2 / m^2 = r^2 / (m * (m - 2*r))) →  -- Volume and geometry of circumscribed cone
      V₂ / V₁ ≤ Vn / Vc) ∧
    V₂ / V₁ = 4/3 :=
sorry

end NUMINAMATH_CALUDE_min_cone_cylinder_volume_ratio_l1584_158460


namespace NUMINAMATH_CALUDE_x_lower_bound_l1584_158479

def x : ℕ → ℝ
  | 0 => 1
  | 1 => 1
  | 2 => 3
  | (n + 3) => 4 * x (n + 2) - 2 * x (n + 1) - 3 * x n

theorem x_lower_bound : ∀ n : ℕ, n ≥ 3 → x n > (3/2) * (1 + 3^(n-2)) := by
  sorry

end NUMINAMATH_CALUDE_x_lower_bound_l1584_158479


namespace NUMINAMATH_CALUDE_horner_method_f_2_l1584_158466

def f (x : ℝ) : ℝ := 4 * x^4 + 3 * x^3 + 2 * x^2 + x + 7

theorem horner_method_f_2 : f 2 = 105 := by
  sorry

end NUMINAMATH_CALUDE_horner_method_f_2_l1584_158466


namespace NUMINAMATH_CALUDE_boarding_students_count_l1584_158452

theorem boarding_students_count (x : ℕ) (students : ℕ) : 
  (students = 4 * x + 10) →  -- If each dormitory houses 4 people with 10 left over
  (6 * (x - 1) + 1 ≤ students) →  -- Lower bound when housing 6 per dormitory
  (students ≤ 6 * (x - 1) + 5) →  -- Upper bound when housing 6 per dormitory
  (students = 34 ∨ students = 38) :=
by sorry

end NUMINAMATH_CALUDE_boarding_students_count_l1584_158452


namespace NUMINAMATH_CALUDE_probability_at_least_one_correct_l1584_158469

theorem probability_at_least_one_correct (n : ℕ) (choices : ℕ) : 
  n = 6 → choices = 6 → 
  1 - (1 - 1 / choices : ℚ) ^ n = 31031 / 46656 := by
  sorry

end NUMINAMATH_CALUDE_probability_at_least_one_correct_l1584_158469


namespace NUMINAMATH_CALUDE_caitlin_age_caitlin_age_proof_l1584_158441

/-- Proves that Caitlin is 54 years old given the conditions in the problem -/
theorem caitlin_age : ℕ → ℕ → ℕ → Prop :=
  λ anna_age brianna_age caitlin_age =>
    anna_age = 48 ∧
    brianna_age = 2 * (anna_age - 18) ∧
    caitlin_age = brianna_age - 6 →
    caitlin_age = 54

/-- Proof of the theorem -/
theorem caitlin_age_proof : caitlin_age 48 60 54 := by
  sorry

end NUMINAMATH_CALUDE_caitlin_age_caitlin_age_proof_l1584_158441


namespace NUMINAMATH_CALUDE_sampling_appropriate_l1584_158498

/-- Represents methods of investigation -/
inductive InvestigationMethod
  | Sampling
  | Comprehensive
  | Other

/-- Represents the characteristics of an investigation -/
structure InvestigationCharacteristics where
  isElectronicProduct : Bool
  largeVolume : Bool
  needComprehensive : Bool

/-- Determines the appropriate investigation method based on given characteristics -/
def appropriateMethod (chars : InvestigationCharacteristics) : InvestigationMethod :=
  sorry

/-- Theorem stating that sampling investigation is appropriate for the given conditions -/
theorem sampling_appropriate (chars : InvestigationCharacteristics)
  (h1 : chars.isElectronicProduct = true)
  (h2 : chars.largeVolume = true)
  (h3 : chars.needComprehensive = false) :
  appropriateMethod chars = InvestigationMethod.Sampling :=
sorry

end NUMINAMATH_CALUDE_sampling_appropriate_l1584_158498


namespace NUMINAMATH_CALUDE_airplane_seats_l1584_158458

theorem airplane_seats (total_seats : ℕ) (first_class : ℕ) (coach_class : ℕ) : 
  total_seats = 387 →
  coach_class = 4 * first_class + 2 →
  first_class + coach_class = total_seats →
  coach_class = 310 := by
sorry

end NUMINAMATH_CALUDE_airplane_seats_l1584_158458


namespace NUMINAMATH_CALUDE_problem_statement_l1584_158403

theorem problem_statement (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  -- Statement 1
  (a^2 - b^2 = 1 → a - b < 1) ∧
  -- Statement 2
  (∃ (a b : ℝ), a > 0 ∧ b > 0 ∧ 1/b - 1/a = 1 ∧ a - b ≥ 1) ∧
  -- Statement 3
  (∃ (a b : ℝ), a > 0 ∧ b > 0 ∧ |Real.sqrt a - Real.sqrt b| = 1 ∧ |a - b| ≥ 1) ∧
  -- Statement 4
  (|a^3 - b^3| = 1 → |a - b| < 1) :=
by sorry

end NUMINAMATH_CALUDE_problem_statement_l1584_158403


namespace NUMINAMATH_CALUDE_smallest_integer_with_remainder_one_smallest_integer_is_2395_l1584_158478

theorem smallest_integer_with_remainder_one (k : ℕ) : 
  (k > 1) ∧ 
  (k % 19 = 1) ∧ 
  (k % 14 = 1) ∧ 
  (k % 9 = 1) → 
  k ≥ 2395 :=
by sorry

theorem smallest_integer_is_2395 : 
  (2395 > 1) ∧ 
  (2395 % 19 = 1) ∧ 
  (2395 % 14 = 1) ∧ 
  (2395 % 9 = 1) :=
by sorry

end NUMINAMATH_CALUDE_smallest_integer_with_remainder_one_smallest_integer_is_2395_l1584_158478


namespace NUMINAMATH_CALUDE_sum_of_fractions_l1584_158421

theorem sum_of_fractions : (9 : ℚ) / 10 + (5 : ℚ) / 6 = (26 : ℚ) / 15 := by sorry

end NUMINAMATH_CALUDE_sum_of_fractions_l1584_158421


namespace NUMINAMATH_CALUDE_length_AB_l1584_158489

/-- Given an angle α with vertex A and a point B at distances a and b from the sides of the angle,
    the length of AB is either (√(a² + b² + 2ab*cos(α))) / sin(α) or (√(a² + b² - 2ab*cos(α))) / sin(α) -/
theorem length_AB (α a b : ℝ) (hα : 0 < α ∧ α < π) (ha : a > 0) (hb : b > 0) :
  ∃ (AB : ℝ), AB > 0 ∧
  (AB = (Real.sqrt (a^2 + b^2 + 2*a*b*Real.cos α)) / Real.sin α ∨
   AB = (Real.sqrt (a^2 + b^2 - 2*a*b*Real.cos α)) / Real.sin α) :=
by sorry

end NUMINAMATH_CALUDE_length_AB_l1584_158489


namespace NUMINAMATH_CALUDE_leading_coefficient_of_specific_polynomial_l1584_158483

/-- A polynomial function from ℝ to ℝ -/
noncomputable def PolynomialFunction := ℝ → ℝ

/-- The leading coefficient of a polynomial function -/
noncomputable def leadingCoefficient (g : PolynomialFunction) : ℝ := sorry

theorem leading_coefficient_of_specific_polynomial 
  (g : PolynomialFunction)
  (h : ∀ x : ℝ, g (x + 1) - g x = 8 * x + 6) :
  leadingCoefficient g = 4 := by sorry

end NUMINAMATH_CALUDE_leading_coefficient_of_specific_polynomial_l1584_158483


namespace NUMINAMATH_CALUDE_no_solutions_abs_x_eq_3_abs_x_plus_2_l1584_158456

theorem no_solutions_abs_x_eq_3_abs_x_plus_2 :
  ∀ x : ℝ, ¬(|x| = 3 * (|x| + 2)) :=
by
  sorry

end NUMINAMATH_CALUDE_no_solutions_abs_x_eq_3_abs_x_plus_2_l1584_158456


namespace NUMINAMATH_CALUDE_cosine_sine_fraction_equals_negative_tangent_l1584_158462

theorem cosine_sine_fraction_equals_negative_tangent (α : ℝ) :
  (Real.cos α - Real.cos (3 * α) + Real.cos (5 * α) - Real.cos (7 * α)) / 
  (Real.sin α + Real.sin (3 * α) + Real.sin (5 * α) + Real.sin (7 * α)) = 
  -Real.tan α := by
  sorry

end NUMINAMATH_CALUDE_cosine_sine_fraction_equals_negative_tangent_l1584_158462


namespace NUMINAMATH_CALUDE_max_silver_tokens_l1584_158411

/-- Represents the number of tokens Eva has -/
structure TokenCount where
  red : ℕ
  blue : ℕ
  silver : ℕ

/-- Represents the exchange rates at the booths -/
structure ExchangeRates where
  red_to_silver : ℕ × ℕ × ℕ  -- (red input, silver output, blue output)
  blue_to_silver : ℕ × ℕ × ℕ  -- (blue input, silver output, red output)

/-- Determines if an exchange is possible given a TokenCount and ExchangeRates -/
def can_exchange (tokens : TokenCount) (rates : ExchangeRates) : Bool :=
  tokens.red ≥ rates.red_to_silver.1 ∨ tokens.blue ≥ rates.blue_to_silver.1

/-- Performs all possible exchanges and returns the final TokenCount -/
def exchange_all (initial : TokenCount) (rates : ExchangeRates) : TokenCount :=
  sorry

/-- Theorem: Given the initial conditions and exchange rates, 
    the maximum number of silver tokens Eva can obtain is 57 -/
theorem max_silver_tokens (initial : TokenCount) (rates : ExchangeRates) 
    (h_initial : initial.red = 60 ∧ initial.blue = 90 ∧ initial.silver = 0)
    (h_rates : rates.red_to_silver = (3, 2, 1) ∧ rates.blue_to_silver = (4, 3, 1)) :
    (exchange_all initial rates).silver = 57 := by
  sorry

end NUMINAMATH_CALUDE_max_silver_tokens_l1584_158411


namespace NUMINAMATH_CALUDE_seokjin_position_relative_to_jungkook_l1584_158485

/-- Given the positions of Jungkook, Yoojeong, and Seokjin on a staircase,
    prove that Seokjin stands 3 steps above Jungkook. -/
theorem seokjin_position_relative_to_jungkook 
  (jungkook_stair : ℕ) 
  (yoojeong_above_jungkook : ℕ) 
  (seokjin_below_yoojeong : ℕ) 
  (h1 : jungkook_stair = 19)
  (h2 : yoojeong_above_jungkook = 8)
  (h3 : seokjin_below_yoojeong = 5) :
  (jungkook_stair + yoojeong_above_jungkook - seokjin_below_yoojeong) - jungkook_stair = 3 :=
by sorry

end NUMINAMATH_CALUDE_seokjin_position_relative_to_jungkook_l1584_158485


namespace NUMINAMATH_CALUDE_abs_neg_two_equals_two_l1584_158480

theorem abs_neg_two_equals_two :
  abs (-2) = 2 := by
sorry

end NUMINAMATH_CALUDE_abs_neg_two_equals_two_l1584_158480


namespace NUMINAMATH_CALUDE_equation_solution_l1584_158436

theorem equation_solution : 
  ∃! x : ℚ, (x^2 + 3*x + 5) / (x + 6) = x + 7 ∧ x = -37/10 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l1584_158436


namespace NUMINAMATH_CALUDE_pencil_boxes_filled_l1584_158427

theorem pencil_boxes_filled (total_pencils : ℕ) (pencils_per_box : ℕ) (h1 : total_pencils = 7344) (h2 : pencils_per_box = 7) : 
  total_pencils / pencils_per_box = 1049 := by
  sorry

end NUMINAMATH_CALUDE_pencil_boxes_filled_l1584_158427


namespace NUMINAMATH_CALUDE_inequality_holds_iff_l1584_158405

theorem inequality_holds_iff (x : ℝ) : 
  0 ≤ x ∧ x ≤ 2 * π →
  (2 * Real.cos x ≤ |Real.sqrt (1 + Real.sin (2 * x)) - Real.sqrt (1 - Real.sin (2 * x))| ∧
   |Real.sqrt (1 + Real.sin (2 * x)) - Real.sqrt (1 - Real.sin (2 * x))| ≤ Real.sqrt 2) ↔
  (π / 4 ≤ x ∧ x ≤ 7 * π / 4) :=
by sorry

end NUMINAMATH_CALUDE_inequality_holds_iff_l1584_158405


namespace NUMINAMATH_CALUDE_blue_marble_probability_l1584_158490

theorem blue_marble_probability : 
  ∀ (total yellow green red blue : ℕ),
    total = 60 →
    yellow = 20 →
    green = yellow / 2 →
    red = blue →
    total = yellow + green + red + blue →
    (blue : ℚ) / total * 100 = 25 :=
by
  sorry

end NUMINAMATH_CALUDE_blue_marble_probability_l1584_158490


namespace NUMINAMATH_CALUDE_trig_identities_l1584_158491

theorem trig_identities (α : Real) (h : Real.sin α = 2 * Real.cos α) :
  (Real.sin α - 4 * Real.cos α) / (5 * Real.sin α + 2 * Real.cos α) = -1/6 ∧
  Real.sin α ^ 2 + 2 * Real.sin α * Real.cos α = 8/5 := by
  sorry

end NUMINAMATH_CALUDE_trig_identities_l1584_158491


namespace NUMINAMATH_CALUDE_initial_red_orchids_l1584_158409

/-- Represents the number of orchids in a vase -/
structure OrchidVase where
  initialRed : ℕ
  initialWhite : ℕ
  addedRed : ℕ
  finalRed : ℕ

/-- Theorem stating the initial number of red orchids in the vase -/
theorem initial_red_orchids (vase : OrchidVase)
  (h1 : vase.initialWhite = 3)
  (h2 : vase.addedRed = 6)
  (h3 : vase.finalRed = 15)
  : vase.initialRed = 9 := by
  sorry

end NUMINAMATH_CALUDE_initial_red_orchids_l1584_158409


namespace NUMINAMATH_CALUDE_action_figures_added_jerry_action_figures_l1584_158412

theorem action_figures_added (initial : ℕ) (final : ℕ) (removed : ℕ) : ℕ :=
  let added := final - (initial - removed)
  by
    sorry

theorem jerry_action_figures : action_figures_added 3 6 1 = 4 := by
  sorry

end NUMINAMATH_CALUDE_action_figures_added_jerry_action_figures_l1584_158412


namespace NUMINAMATH_CALUDE_max_equal_distribution_l1584_158424

theorem max_equal_distribution (bags : Nat) (eyeliners : Nat) : 
  bags = 2923 → eyeliners = 3239 → Nat.gcd bags eyeliners = 1 := by
  sorry

end NUMINAMATH_CALUDE_max_equal_distribution_l1584_158424


namespace NUMINAMATH_CALUDE_triangle_area_l1584_158400

/-- The area of a triangle with perimeter 32 and inradius 2.5 is 40 -/
theorem triangle_area (perimeter : ℝ) (inradius : ℝ) (area : ℝ) 
  (h1 : perimeter = 32) 
  (h2 : inradius = 2.5) 
  (h3 : area = inradius * (perimeter / 2)) : 
  area = 40 := by
sorry

end NUMINAMATH_CALUDE_triangle_area_l1584_158400


namespace NUMINAMATH_CALUDE_square_difference_thirteen_twelve_l1584_158482

theorem square_difference_thirteen_twelve : (13 + 12)^2 - (13 - 12)^2 = 624 := by
  sorry

end NUMINAMATH_CALUDE_square_difference_thirteen_twelve_l1584_158482


namespace NUMINAMATH_CALUDE_journey_average_speed_l1584_158430

/-- Prove that the average speed of a journey with four equal-length segments,
    traveled at speeds of 3, 2, 6, and 3 km/h respectively, is 3 km/h. -/
theorem journey_average_speed (x : ℝ) (hx : x > 0) : 
  let total_distance := 4 * x
  let total_time := x / 3 + x / 2 + x / 6 + x / 3
  total_distance / total_time = 3 := by
  sorry

end NUMINAMATH_CALUDE_journey_average_speed_l1584_158430


namespace NUMINAMATH_CALUDE_plywood_cut_perimeter_difference_l1584_158425

/-- Represents a rectangular piece of plywood -/
structure Plywood where
  length : ℝ
  width : ℝ

/-- Represents a way to cut the plywood into 8 congruent pieces -/
structure CutConfiguration where
  piece_length : ℝ
  piece_width : ℝ

/-- Calculates the perimeter of a rectangular piece -/
def perimeter (p : CutConfiguration) : ℝ :=
  2 * (p.piece_length + p.piece_width)

/-- Theorem: The difference between max and min perimeter of cut pieces is 11.5 feet -/
theorem plywood_cut_perimeter_difference :
  let original : Plywood := ⟨12, 6⟩
  let valid_cut (c : CutConfiguration) : Prop :=
    c.piece_length * c.piece_width * 8 = original.length * original.width
  ∃ (max_cut min_cut : CutConfiguration),
    valid_cut max_cut ∧ valid_cut min_cut ∧
    (∀ c, valid_cut c → perimeter c ≤ perimeter max_cut) ∧
    (∀ c, valid_cut c → perimeter c ≥ perimeter min_cut) ∧
    perimeter max_cut - perimeter min_cut = 11.5 :=
by
  sorry

end NUMINAMATH_CALUDE_plywood_cut_perimeter_difference_l1584_158425


namespace NUMINAMATH_CALUDE_tangent_difference_l1584_158459

theorem tangent_difference (θ : Real) 
  (h : 3 * Real.sin θ + Real.cos θ = Real.sqrt 10) : 
  Real.tan (θ + π/8) - 1 / Real.tan (θ + π/8) = -14 := by
  sorry

end NUMINAMATH_CALUDE_tangent_difference_l1584_158459


namespace NUMINAMATH_CALUDE_determinant_equals_t_minus_s_plus_r_l1584_158496

-- Define the polynomial
def polynomial (x r s t : ℝ) : ℝ := x^4 + r*x^2 + s*x + t

-- Define the matrix
def matrix (a b c d : ℝ) : Matrix (Fin 4) (Fin 4) ℝ :=
  ![![1+a, 1,   1,   1],
    ![1,   1+b, 1,   1],
    ![1,   1,   1+c, 1],
    ![1,   1,   1,   1+d]]

theorem determinant_equals_t_minus_s_plus_r 
  (r s t : ℝ) (a b c d : ℝ) 
  (h1 : polynomial a r s t = 0)
  (h2 : polynomial b r s t = 0)
  (h3 : polynomial c r s t = 0)
  (h4 : polynomial d r s t = 0) :
  Matrix.det (matrix a b c d) = t - s + r := by
  sorry

end NUMINAMATH_CALUDE_determinant_equals_t_minus_s_plus_r_l1584_158496


namespace NUMINAMATH_CALUDE_sequence_equality_l1584_158431

def A : ℕ → ℚ
  | 0 => 1
  | n + 1 => (A n + 2) / (A n + 1)

def B : ℕ → ℚ
  | 0 => 1
  | n + 1 => (B n ^ 2 + 2) / (2 * B n)

theorem sequence_equality (n : ℕ) : B (n + 1) = A (2 ^ n) := by
  sorry

end NUMINAMATH_CALUDE_sequence_equality_l1584_158431


namespace NUMINAMATH_CALUDE_clock_angle_at_7_l1584_158426

/-- The number of hours on a clock face -/
def clock_hours : ℕ := 12

/-- The time in hours -/
def time : ℕ := 7

/-- The angle between each hour mark on the clock -/
def angle_per_hour : ℚ := 360 / clock_hours

/-- The position of the hour hand in degrees -/
def hour_hand_position : ℚ := time * angle_per_hour

/-- The smaller angle between the hour and minute hands at the given time -/
def smaller_angle : ℚ := min hour_hand_position (360 - hour_hand_position)

/-- Theorem stating that the smaller angle between clock hands at 7 o'clock is 150 degrees -/
theorem clock_angle_at_7 : smaller_angle = 150 := by sorry

end NUMINAMATH_CALUDE_clock_angle_at_7_l1584_158426


namespace NUMINAMATH_CALUDE_expression_simplification_l1584_158467

theorem expression_simplification (x y : ℝ) :
  (x + 2*y) * (x - 2*y) - x * (x + 3*y) = -4*y^2 - 3*x*y ∧
  (x - 1 - 3/(x + 1)) / ((x^2 - 4*x + 4) / (x + 1)) = (x + 2) / (x - 2) :=
by sorry

end NUMINAMATH_CALUDE_expression_simplification_l1584_158467


namespace NUMINAMATH_CALUDE_waiter_customers_l1584_158415

theorem waiter_customers (non_tipping_customers : ℕ) (tip_amount : ℕ) (total_tips : ℕ) : 
  non_tipping_customers = 5 →
  tip_amount = 8 →
  total_tips = 32 →
  non_tipping_customers + (total_tips / tip_amount) = 9 :=
by sorry

end NUMINAMATH_CALUDE_waiter_customers_l1584_158415


namespace NUMINAMATH_CALUDE_DE_length_l1584_158487

-- Define the fixed points A and B
def A : ℝ × ℝ := (-2, 0)
def B : ℝ × ℝ := (1, 0)

-- Define the curve C
def C : Set (ℝ × ℝ) := {p : ℝ × ℝ | (p.1 - 2)^2 + p.2^2 = 4}

-- Define the condition for point P
def P_condition (P : ℝ × ℝ) : Prop :=
  (P.1 + 2)^2 + P.2^2 = 4 * ((P.1 - 1)^2 + P.2^2)

-- Define the line l
def l (k : ℝ) : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.2 = k * p.1 - 3}

-- Define the intersection points D and E
def intersection_points (k : ℝ) : Prop :=
  ∃ (D E : ℝ × ℝ), D ∈ C ∩ l k ∧ E ∈ C ∩ l k ∧ D ≠ E

-- Define the condition x₁x₂ + y₁y₂ = 3
def point_product_condition (D E : ℝ × ℝ) : Prop :=
  D.1 * E.1 + D.2 * E.2 = 3

-- Theorem statement
theorem DE_length :
  ∀ (k : ℝ) (D E : ℝ × ℝ),
  k > 5/12 →
  intersection_points k →
  point_product_condition D E →
  (D.1 - E.1)^2 + (D.2 - E.2)^2 = 14 := by
  sorry

end NUMINAMATH_CALUDE_DE_length_l1584_158487


namespace NUMINAMATH_CALUDE_departure_interval_is_six_l1584_158457

/-- Represents the tram system with a person riding along the route -/
structure TramSystem where
  tram_speed : ℝ
  person_speed : ℝ
  overtake_time : ℝ
  approach_time : ℝ

/-- The interval between tram departures from the station -/
def departure_interval (sys : TramSystem) : ℝ :=
  6

/-- Theorem stating that the departure interval is 6 minutes -/
theorem departure_interval_is_six (sys : TramSystem) 
  (h1 : sys.tram_speed > sys.person_speed) 
  (h2 : sys.overtake_time = 12)
  (h3 : sys.approach_time = 4) :
  departure_interval sys = 6 := by
  sorry

end NUMINAMATH_CALUDE_departure_interval_is_six_l1584_158457


namespace NUMINAMATH_CALUDE_two_axisymmetric_additions_l1584_158417

/-- Represents a position on a 4x4 grid --/
structure Position where
  x : Fin 4
  y : Fin 4

/-- Represents a configuration of shaded squares on a 4x4 grid --/
def Configuration := List Position

/-- Checks if a configuration is axisymmetric --/
def isAxisymmetric (config : Configuration) : Bool :=
  sorry

/-- Counts the number of ways to add one square to make the configuration axisymmetric --/
def countAxisymmetricAdditions (initialConfig : Configuration) : Nat :=
  sorry

/-- The initial configuration with 3 shaded squares --/
def initialConfig : Configuration :=
  sorry

theorem two_axisymmetric_additions :
  countAxisymmetricAdditions initialConfig = 2 :=
sorry

end NUMINAMATH_CALUDE_two_axisymmetric_additions_l1584_158417


namespace NUMINAMATH_CALUDE_amount_distributed_l1584_158406

theorem amount_distributed (A : ℚ) : 
  (∀ (x : ℚ), x = A / 30 - A / 40 → x = 135.50) →
  A = 16260 := by
sorry

end NUMINAMATH_CALUDE_amount_distributed_l1584_158406


namespace NUMINAMATH_CALUDE_min_value_squared_difference_l1584_158477

theorem min_value_squared_difference (f : ℝ → ℝ) :
  (∀ x, f x = (x - 1)^2) →
  ∃ m : ℝ, (∀ x, f x ≥ m) ∧ (∃ x₀, f x₀ = m) ∧ m = 0 :=
by sorry

end NUMINAMATH_CALUDE_min_value_squared_difference_l1584_158477


namespace NUMINAMATH_CALUDE_rectangle_width_decrease_l1584_158438

theorem rectangle_width_decrease (L W : ℝ) (L' W' : ℝ) (h1 : L' = 1.3 * L) (h2 : L * W = L' * W') : 
  (W - W') / W = 23.08 / 100 :=
sorry

end NUMINAMATH_CALUDE_rectangle_width_decrease_l1584_158438


namespace NUMINAMATH_CALUDE_only_2015_could_be_hexadecimal_l1584_158472

def is_hexadecimal_digit (d : Char) : Bool :=
  ('0' <= d && d <= '9') || ('A' <= d && d <= 'F')

def could_be_hexadecimal (n : Nat) : Bool :=
  n.repr.all is_hexadecimal_digit

theorem only_2015_could_be_hexadecimal :
  (could_be_hexadecimal 66 = false) ∧
  (could_be_hexadecimal 108 = false) ∧
  (could_be_hexadecimal 732 = false) ∧
  (could_be_hexadecimal 2015 = true) :=
by sorry

end NUMINAMATH_CALUDE_only_2015_could_be_hexadecimal_l1584_158472


namespace NUMINAMATH_CALUDE_min_perimeter_isosceles_triangles_l1584_158450

/-- Represents an isosceles triangle with integer side lengths -/
structure IsoscelesTriangle where
  side : ℕ
  base : ℕ

/-- Checks if two isosceles triangles are noncongruent -/
def noncongruent (t1 t2 : IsoscelesTriangle) : Prop :=
  t1.side ≠ t2.side ∨ t1.base ≠ t2.base

/-- Calculates the perimeter of an isosceles triangle -/
def perimeter (t : IsoscelesTriangle) : ℕ :=
  2 * t.side + t.base

/-- Calculates the area of an isosceles triangle -/
noncomputable def area (t : IsoscelesTriangle) : ℝ :=
  (t.base / 4 : ℝ) * Real.sqrt (4 * t.side^2 - t.base^2)

/-- Theorem: Minimum perimeter of two specific isosceles triangles -/
theorem min_perimeter_isosceles_triangles :
  ∃ (t1 t2 : IsoscelesTriangle),
    noncongruent t1 t2 ∧
    perimeter t1 = perimeter t2 ∧
    area t1 = area t2 ∧
    5 * t2.base = 4 * t1.base ∧
    ∀ (s1 s2 : IsoscelesTriangle),
      noncongruent s1 s2 →
      perimeter s1 = perimeter s2 →
      area s1 = area s2 →
      5 * s2.base = 4 * s1.base →
      perimeter t1 ≤ perimeter s1 ∧
    perimeter t1 = 1180 :=
  sorry

end NUMINAMATH_CALUDE_min_perimeter_isosceles_triangles_l1584_158450


namespace NUMINAMATH_CALUDE_initial_participants_count_l1584_158443

/-- The number of participants in the social event -/
def n : ℕ := 15

/-- The number of people who left early -/
def early_leavers : ℕ := 4

/-- The number of handshakes each early leaver performed -/
def handshakes_per_leaver : ℕ := 2

/-- The total number of handshakes that occurred -/
def total_handshakes : ℕ := 60

/-- Theorem stating that n is the correct number of initial participants -/
theorem initial_participants_count :
  Nat.choose n 2 - (early_leavers * handshakes_per_leaver - Nat.choose early_leavers 2) = total_handshakes :=
by sorry

end NUMINAMATH_CALUDE_initial_participants_count_l1584_158443


namespace NUMINAMATH_CALUDE_basket_weight_l1584_158446

theorem basket_weight (pear_weight : ℝ) (num_pears : ℕ) (total_weight : ℝ) 
  (h1 : pear_weight = 0.36)
  (h2 : num_pears = 30)
  (h3 : total_weight = 11.26) :
  total_weight - pear_weight * (num_pears : ℝ) = 0.46 := by
  sorry

end NUMINAMATH_CALUDE_basket_weight_l1584_158446


namespace NUMINAMATH_CALUDE_largest_certain_divisor_l1584_158464

-- Define the set of numbers on the die
def dieNumbers : Finset ℕ := Finset.range 8

-- Define the type for products of seven numbers from the die
def ProductOfSeven : Type :=
  {s : Finset ℕ // s ⊆ dieNumbers ∧ s.card = 7}

-- Define the product function
def product (s : ProductOfSeven) : ℕ :=
  s.val.prod id

-- Theorem statement
theorem largest_certain_divisor :
  (∀ s : ProductOfSeven, 192 ∣ product s) ∧
  (∀ n : ℕ, n > 192 → ∃ s : ProductOfSeven, ¬(n ∣ product s)) :=
sorry

end NUMINAMATH_CALUDE_largest_certain_divisor_l1584_158464


namespace NUMINAMATH_CALUDE_total_amount_is_correct_l1584_158451

def grapes_quantity : ℕ := 10
def grapes_rate : ℕ := 70
def mangoes_quantity : ℕ := 9
def mangoes_rate : ℕ := 55
def apples_quantity : ℕ := 12
def apples_rate : ℕ := 80
def papayas_quantity : ℕ := 7
def papayas_rate : ℕ := 45
def oranges_quantity : ℕ := 15
def oranges_rate : ℕ := 30
def bananas_quantity : ℕ := 5
def bananas_rate : ℕ := 25

def total_amount : ℕ := 
  grapes_quantity * grapes_rate +
  mangoes_quantity * mangoes_rate +
  apples_quantity * apples_rate +
  papayas_quantity * papayas_rate +
  oranges_quantity * oranges_rate +
  bananas_quantity * bananas_rate

theorem total_amount_is_correct : total_amount = 3045 := by
  sorry

end NUMINAMATH_CALUDE_total_amount_is_correct_l1584_158451


namespace NUMINAMATH_CALUDE_bob_daily_earnings_l1584_158402

/-- Proves that Bob makes $4 per day given the conditions of the problem -/
theorem bob_daily_earnings (sally_earnings : ℝ) (total_savings : ℝ) (days_in_year : ℕ) :
  sally_earnings = 6 →
  total_savings = 1825 →
  days_in_year = 365 →
  ∃ (bob_earnings : ℝ),
    bob_earnings = 4 ∧
    (sally_earnings / 2 + bob_earnings / 2) * days_in_year = total_savings :=
by sorry

end NUMINAMATH_CALUDE_bob_daily_earnings_l1584_158402


namespace NUMINAMATH_CALUDE_vaishalis_hats_l1584_158455

/-- The number of hats with three stripes each that Vaishali has -/
def hats_with_three_stripes : ℕ := sorry

/-- The number of hats with four stripes each that Vaishali has -/
def hats_with_four_stripes : ℕ := 3

/-- The number of hats with no stripes that Vaishali has -/
def hats_with_no_stripes : ℕ := 6

/-- The number of hats with five stripes each that Vaishali has -/
def hats_with_five_stripes : ℕ := 2

/-- The total number of stripes on all of Vaishali's hats -/
def total_stripes : ℕ := 34

/-- Theorem stating that the number of hats with three stripes is 4 -/
theorem vaishalis_hats : hats_with_three_stripes = 4 := by
  sorry

end NUMINAMATH_CALUDE_vaishalis_hats_l1584_158455


namespace NUMINAMATH_CALUDE_johns_trip_distance_l1584_158418

theorem johns_trip_distance : ∃ (total_distance : ℝ), 
  (total_distance / 2) + 40 + (total_distance / 4) = total_distance ∧ 
  total_distance = 160 := by
sorry

end NUMINAMATH_CALUDE_johns_trip_distance_l1584_158418


namespace NUMINAMATH_CALUDE_cinema_chairs_l1584_158439

/-- The total number of chairs in a cinema with a given number of rows and chairs per row. -/
def total_chairs (rows : ℕ) (chairs_per_row : ℕ) : ℕ := rows * chairs_per_row

/-- Theorem: The total number of chairs in a cinema with 4 rows and 8 chairs per row is 32. -/
theorem cinema_chairs : total_chairs 4 8 = 32 := by
  sorry

end NUMINAMATH_CALUDE_cinema_chairs_l1584_158439


namespace NUMINAMATH_CALUDE_complement_of_M_in_U_l1584_158449

-- Define the set M
def M : Set ℝ := {x : ℝ | x^2 - 2*x > 0}

-- Define the universe U as the set of real numbers
def U : Set ℝ := Set.univ

-- Theorem statement
theorem complement_of_M_in_U : 
  U \ M = Set.Icc 0 2 := by sorry

end NUMINAMATH_CALUDE_complement_of_M_in_U_l1584_158449


namespace NUMINAMATH_CALUDE_twin_prime_power_sum_divisibility_l1584_158470

theorem twin_prime_power_sum_divisibility (p q : ℕ) : 
  Nat.Prime p → Nat.Prime q → q = p + 2 → (p + q) ∣ (p^q + q^p) := by
  sorry

end NUMINAMATH_CALUDE_twin_prime_power_sum_divisibility_l1584_158470


namespace NUMINAMATH_CALUDE_binary_11011_equals_27_l1584_158423

def binary_to_decimal (b : List Bool) : Nat :=
  b.enum.foldr (fun (i, bit) acc => acc + if bit then 2^i else 0) 0

theorem binary_11011_equals_27 : 
  binary_to_decimal [true, true, false, true, true] = 27 := by
  sorry

end NUMINAMATH_CALUDE_binary_11011_equals_27_l1584_158423


namespace NUMINAMATH_CALUDE_meat_voters_count_l1584_158433

/-- The number of students who voted for veggies -/
def veggies_votes : ℕ := 337

/-- The total number of students who voted -/
def total_votes : ℕ := 672

/-- The number of students who voted for meat -/
def meat_votes : ℕ := total_votes - veggies_votes

theorem meat_voters_count : meat_votes = 335 := by
  sorry

end NUMINAMATH_CALUDE_meat_voters_count_l1584_158433


namespace NUMINAMATH_CALUDE_circles_have_three_common_tangents_l1584_158471

/-- Circle C₁ with equation x² + y² + 2x + 4y + 1 = 0 -/
def C₁ (x y : ℝ) : Prop := x^2 + y^2 + 2*x + 4*y + 1 = 0

/-- Circle C₂ with equation x² + y² - 4x - 4y - 1 = 0 -/
def C₂ (x y : ℝ) : Prop := x^2 + y^2 - 4*x - 4*y - 1 = 0

/-- The number of common tangents between C₁ and C₂ -/
def num_common_tangents : ℕ := 3

theorem circles_have_three_common_tangents :
  ∃ (n : ℕ), n = num_common_tangents ∧ 
  (∀ (x y : ℝ), C₁ x y ∨ C₂ x y → n = 3) :=
sorry

end NUMINAMATH_CALUDE_circles_have_three_common_tangents_l1584_158471


namespace NUMINAMATH_CALUDE_triangle_point_coordinates_l1584_158429

/-- Given a triangle ABC with median CM and angle bisector BL, prove that the coordinates of C are (14, 2) -/
theorem triangle_point_coordinates (A M L : ℝ × ℝ) :
  A = (2, 8) →
  M = (4, 11) →
  L = (6, 6) →
  ∃ (B C : ℝ × ℝ),
    (M = ((A.1 + B.1) / 2, (A.2 + B.2) / 2)) ∧  -- M is the midpoint of AB
    (∃ (t : ℝ), L = (1 - t) • B + t • C) ∧      -- L lies on BC
    (∃ (s : ℝ), C = (1 - s) • A + s • B) ∧     -- C lies on AB
    (C.1 = 14 ∧ C.2 = 2) :=
by sorry


end NUMINAMATH_CALUDE_triangle_point_coordinates_l1584_158429


namespace NUMINAMATH_CALUDE_sum_of_threes_place_values_l1584_158463

def number : ℕ := 63130

theorem sum_of_threes_place_values (hundreds_digit : ℕ) (tens_digit : ℕ) :
  (number / 100 % 10 = 3) →
  (number / 10 % 10 = 3) →
  (hundreds_digit = 3 * 100) →
  (tens_digit = 3 * 10) →
  hundreds_digit + tens_digit = 330 := by
sorry

end NUMINAMATH_CALUDE_sum_of_threes_place_values_l1584_158463


namespace NUMINAMATH_CALUDE_quadruple_base_triple_exponent_l1584_158475

theorem quadruple_base_triple_exponent (a b : ℝ) (x : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : x > 0) :
  (4 * a) ^ (3 * b) = a ^ b * x ^ b → x = 64 * a ^ 2 := by
  sorry

end NUMINAMATH_CALUDE_quadruple_base_triple_exponent_l1584_158475


namespace NUMINAMATH_CALUDE_sabertooth_tails_count_l1584_158476

/-- Represents the number of legs for Triassic Discoglossus tadpoles -/
def triassic_legs : ℕ := 5

/-- Represents the number of tails for Triassic Discoglossus tadpoles -/
def triassic_tails : ℕ := 1

/-- Represents the number of legs for Sabertooth Frog tadpoles -/
def sabertooth_legs : ℕ := 4

/-- Represents the total number of legs of all captured tadpoles -/
def total_legs : ℕ := 100

/-- Represents the total number of tails of all captured tadpoles -/
def total_tails : ℕ := 64

/-- Proves that the number of tails per Sabertooth Frog tadpole is 3 -/
theorem sabertooth_tails_count :
  ∃ (n k : ℕ),
    n * triassic_legs + k * sabertooth_legs = total_legs ∧
    n * triassic_tails + k * 3 = total_tails :=
by sorry

end NUMINAMATH_CALUDE_sabertooth_tails_count_l1584_158476


namespace NUMINAMATH_CALUDE_defective_smartphone_probability_l1584_158461

/-- The probability of selecting two defective smartphones from a shipment -/
theorem defective_smartphone_probability
  (total : ℕ) (defective : ℕ) (h1 : total = 220) (h2 : defective = 84) :
  (defective : ℚ) / total * ((defective - 1) : ℚ) / (total - 1) =
  (84 : ℚ) / 220 * (83 : ℚ) / 219 :=
by sorry

end NUMINAMATH_CALUDE_defective_smartphone_probability_l1584_158461


namespace NUMINAMATH_CALUDE_smallest_n_inequality_l1584_158497

theorem smallest_n_inequality (w x y z : ℝ) : 
  ∃ (n : ℕ), (w^2 + x^2 + y^2 + z^2)^2 ≤ n*(w^4 + x^4 + y^4 + z^4) ∧ 
  ∀ (m : ℕ), m < n → ∃ (a b c d : ℝ), (a^2 + b^2 + c^2 + d^2)^2 > m*(a^4 + b^4 + c^4 + d^4) :=
by
  sorry

end NUMINAMATH_CALUDE_smallest_n_inequality_l1584_158497


namespace NUMINAMATH_CALUDE_combined_girls_avg_is_84_l1584_158492

/-- Represents a high school with exam scores -/
structure School where
  boys_avg : ℝ
  girls_avg : ℝ
  combined_avg : ℝ

/-- Calculates the combined average score for girls across two schools -/
def combined_girls_avg (school1 school2 : School) (boys_combined_avg : ℝ) : ℝ :=
  -- Implementation not provided, as per instructions
  sorry

/-- Theorem stating that the combined average score for girls is 84 -/
theorem combined_girls_avg_is_84 
  (adams : School)
  (baker : School)
  (h_adams_boys : adams.boys_avg = 71)
  (h_adams_girls : adams.girls_avg = 76)
  (h_adams_combined : adams.combined_avg = 74)
  (h_baker_boys : baker.boys_avg = 81)
  (h_baker_girls : baker.girls_avg = 90)
  (h_baker_combined : baker.combined_avg = 84)
  (h_boys_combined : boys_combined_avg = 79) :
  combined_girls_avg adams baker boys_combined_avg = 84 := by
  sorry

end NUMINAMATH_CALUDE_combined_girls_avg_is_84_l1584_158492


namespace NUMINAMATH_CALUDE_factorization_equality_l1584_158495

theorem factorization_equality (m n : ℝ) : 2 * m * n^2 - 4 * m * n + 2 * m = 2 * m * (n - 1)^2 := by
  sorry

end NUMINAMATH_CALUDE_factorization_equality_l1584_158495


namespace NUMINAMATH_CALUDE_equation_solution_l1584_158473

theorem equation_solution :
  ∃ x : ℚ, (3 * x + 4 * x = 600 - (5 * x + 6 * x)) ∧ (x = 100 / 3) := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l1584_158473


namespace NUMINAMATH_CALUDE_cookies_left_after_ted_l1584_158408

/-- Calculates the number of cookies left after Frank's baking and consumption, and Ted's visit -/
def cookies_left (days : ℕ) (trays_per_day : ℕ) (cookies_per_tray : ℕ) 
                 (frank_daily_consumption : ℕ) (ted_consumption : ℕ) : ℕ :=
  days * trays_per_day * cookies_per_tray - days * frank_daily_consumption - ted_consumption

/-- Proves that 134 cookies are left after 6 days of Frank's baking and Ted's visit -/
theorem cookies_left_after_ted : cookies_left 6 2 12 1 4 = 134 := by
  sorry

end NUMINAMATH_CALUDE_cookies_left_after_ted_l1584_158408


namespace NUMINAMATH_CALUDE_speeds_satisfy_conditions_l1584_158453

/-- The speed of person A in km/h -/
def speed_A : ℝ := 3.6

/-- The speed of person B in km/h -/
def speed_B : ℝ := 6

/-- The total distance between the starting points of person A and person B in km -/
def total_distance : ℝ := 36

/-- Theorem stating that the given speeds satisfy the conditions of the problem -/
theorem speeds_satisfy_conditions :
  (5 * speed_A + 3 * speed_B = total_distance) ∧
  (2.5 * speed_A + 4.5 * speed_B = total_distance) :=
by sorry

end NUMINAMATH_CALUDE_speeds_satisfy_conditions_l1584_158453


namespace NUMINAMATH_CALUDE_partial_fraction_decomposition_l1584_158488

theorem partial_fraction_decomposition (x : ℝ) (h1 : x ≠ 3) (h2 : x ≠ 4) :
  6 * x / ((x - 4) * (x - 3)^2) = 
    24 / (x - 4) + (-162/7) / (x - 3) + (-18) / (x - 3)^2 :=
by sorry

end NUMINAMATH_CALUDE_partial_fraction_decomposition_l1584_158488


namespace NUMINAMATH_CALUDE_weight_gain_calculation_l1584_158493

/-- Calculates the new weight of a person after muscle and fat gain --/
def new_weight (initial_weight : ℝ) (muscle_gain_percent : ℝ) (fat_gain_ratio : ℝ) : ℝ :=
  let muscle_gain := initial_weight * muscle_gain_percent
  let fat_gain := muscle_gain * fat_gain_ratio
  initial_weight + muscle_gain + fat_gain

/-- Theorem stating that for the given conditions, the new weight is 150 kg --/
theorem weight_gain_calculation :
  new_weight 120 0.2 0.25 = 150 := by
  sorry

#eval new_weight 120 0.2 0.25

end NUMINAMATH_CALUDE_weight_gain_calculation_l1584_158493


namespace NUMINAMATH_CALUDE_arithmetic_sequence_ninth_term_l1584_158454

/-- An arithmetic sequence is a sequence where the difference between any two consecutive terms is constant. -/
def ArithmeticSequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

/-- The theorem states that for an arithmetic sequence where the 3rd term is 5 and the 6th term is 11, the 9th term is 17. -/
theorem arithmetic_sequence_ninth_term
  (a : ℕ → ℝ)
  (h_arithmetic : ArithmeticSequence a)
  (h_third : a 3 = 5)
  (h_sixth : a 6 = 11) :
  a 9 = 17 := by
  sorry


end NUMINAMATH_CALUDE_arithmetic_sequence_ninth_term_l1584_158454


namespace NUMINAMATH_CALUDE_negative_324_same_terminal_side_as_36_l1584_158434

/-- Two angles have the same terminal side if their difference is a multiple of 360° -/
def same_terminal_side (α β : ℝ) : Prop :=
  ∃ k : ℤ, β = α + k * 360

/-- The angle -324° has the same terminal side as 36° -/
theorem negative_324_same_terminal_side_as_36 :
  same_terminal_side 36 (-324) := by
  sorry

end NUMINAMATH_CALUDE_negative_324_same_terminal_side_as_36_l1584_158434


namespace NUMINAMATH_CALUDE_square_sum_theorem_l1584_158428

theorem square_sum_theorem (x y z : ℝ) 
  (eq1 : x^2 + 3*y = 8)
  (eq2 : y^2 + 5*z = -9)
  (eq3 : z^2 + 7*x = -16) :
  x^2 + y^2 + z^2 = 20.75 := by
sorry

end NUMINAMATH_CALUDE_square_sum_theorem_l1584_158428


namespace NUMINAMATH_CALUDE_cos_240_degrees_l1584_158494

theorem cos_240_degrees : Real.cos (240 * π / 180) = -1/2 := by
  sorry

end NUMINAMATH_CALUDE_cos_240_degrees_l1584_158494


namespace NUMINAMATH_CALUDE_b_share_is_1540_l1584_158432

/-- Represents the share of profits for a partner in a partnership. -/
structure PartnerShare where
  investment : ℕ
  share : ℕ

/-- Calculates the share of a partner given the total profit and the investment ratios. -/
def calculateShare (totalProfit : ℕ) (investmentRatios : List ℕ) (partnerRatio : ℕ) : ℕ :=
  (totalProfit * partnerRatio) / (investmentRatios.sum)

/-- Theorem stating that given the investments and a's share, b's share is $1540. -/
theorem b_share_is_1540 (a b c : PartnerShare) 
  (h1 : a.investment = 15000)
  (h2 : b.investment = 21000)
  (h3 : c.investment = 27000)
  (h4 : a.share = 1100) : 
  b.share = 1540 := by
  sorry


end NUMINAMATH_CALUDE_b_share_is_1540_l1584_158432


namespace NUMINAMATH_CALUDE_intersecting_lines_a_value_l1584_158442

/-- Three lines intersect at one point if and only if their equations are satisfied simultaneously -/
def intersect_at_one_point (a : ℝ) : Prop :=
  ∃ x y : ℝ, a * x + 2 * y + 8 = 0 ∧ 4 * x + 3 * y = 10 ∧ 2 * x - y = 10

/-- The theorem stating that if the three given lines intersect at one point, then a = -1 -/
theorem intersecting_lines_a_value :
  ∀ a : ℝ, intersect_at_one_point a → a = -1 :=
by
  sorry

#check intersecting_lines_a_value

end NUMINAMATH_CALUDE_intersecting_lines_a_value_l1584_158442


namespace NUMINAMATH_CALUDE_prime_squared_with_totient_42_l1584_158404

theorem prime_squared_with_totient_42 (p : ℕ) (N : ℕ) : 
  Prime p → N = p^2 → Nat.totient N = 42 → N = 49 := by
  sorry

end NUMINAMATH_CALUDE_prime_squared_with_totient_42_l1584_158404


namespace NUMINAMATH_CALUDE_root_ratio_sum_l1584_158414

theorem root_ratio_sum (k₁ k₂ : ℝ) : 
  (∃ a b : ℝ, 3 * a^2 - (3 - k₁) * a + 7 = 0 ∧ 
              3 * b^2 - (3 - k₁) * b + 7 = 0 ∧ 
              a / b + b / a = 9 / 7) ∧
  (∃ a b : ℝ, 3 * a^2 - (3 - k₂) * a + 7 = 0 ∧ 
              3 * b^2 - (3 - k₂) * b + 7 = 0 ∧ 
              a / b + b / a = 9 / 7) →
  k₁ / k₂ + k₂ / k₁ = -20 / 7 := by
sorry

end NUMINAMATH_CALUDE_root_ratio_sum_l1584_158414


namespace NUMINAMATH_CALUDE_triangle_angle_calculation_l1584_158465

theorem triangle_angle_calculation (A B C : ℝ) (a b c : ℝ) :
  B = π / 3 →  -- 60° in radians
  a = Real.sqrt 6 →
  b = 3 →
  A + B + C = π →  -- sum of angles in a triangle
  a / (Real.sin A) = b / (Real.sin B) →  -- law of sines
  A < B →  -- larger side opposite larger angle
  A = π / 4  -- 45° in radians
  := by sorry

end NUMINAMATH_CALUDE_triangle_angle_calculation_l1584_158465


namespace NUMINAMATH_CALUDE_log_216_equals_3_log_6_l1584_158422

theorem log_216_equals_3_log_6 : Real.log 216 = 3 * Real.log 6 := by sorry

end NUMINAMATH_CALUDE_log_216_equals_3_log_6_l1584_158422


namespace NUMINAMATH_CALUDE_quadratic_roots_l1584_158484

/-- The quadratic equation kx^2 - (2k-3)x + k-2 = 0 -/
def quadratic_equation (k : ℝ) (x : ℝ) : Prop :=
  k * x^2 - (2*k - 3) * x + (k - 2) = 0

/-- The discriminant of the quadratic equation -/
def discriminant (k : ℝ) : ℝ :=
  9 - 4*k

theorem quadratic_roots :
  (∃! x : ℝ, quadratic_equation 0 x) ∧
  (∀ k : ℝ, 0 < k → k ≤ 9/4 → ∃ x y : ℝ, x ≠ y ∧ quadratic_equation k x ∧ quadratic_equation k y) :=
sorry

end NUMINAMATH_CALUDE_quadratic_roots_l1584_158484


namespace NUMINAMATH_CALUDE_geometric_sequence_154th_term_l1584_158410

/-- Represents a geometric sequence with first term a₁ and common ratio r -/
def GeometricSequence (a₁ : ℝ) (r : ℝ) : ℕ → ℝ := fun n => a₁ * r ^ (n - 1)

/-- The 154th term of a geometric sequence with first term 4 and second term 12 -/
theorem geometric_sequence_154th_term :
  let seq := GeometricSequence 4 3
  seq 154 = 4 * 3^153 := by sorry

end NUMINAMATH_CALUDE_geometric_sequence_154th_term_l1584_158410


namespace NUMINAMATH_CALUDE_sqrt_3_5_7_not_arithmetic_sequence_l1584_158419

theorem sqrt_3_5_7_not_arithmetic_sequence : 
  ¬ ∃ (d : ℝ), Real.sqrt 5 - Real.sqrt 3 = d ∧ Real.sqrt 7 - Real.sqrt 5 = d :=
by
  sorry

end NUMINAMATH_CALUDE_sqrt_3_5_7_not_arithmetic_sequence_l1584_158419


namespace NUMINAMATH_CALUDE_quadratic_other_intercept_l1584_158440

/-- For a quadratic function f(x) = ax^2 + bx + c with vertex (2, 10) and one x-intercept at (1, 0),
    the x-coordinate of the other x-intercept is 3. -/
theorem quadratic_other_intercept 
  (f : ℝ → ℝ) 
  (a b c : ℝ) 
  (h1 : ∀ x, f x = a * x^2 + b * x + c) 
  (h2 : f 2 = 10 ∧ (∀ x, f x ≤ 10)) 
  (h3 : f 1 = 0) : 
  ∃ x, x ≠ 1 ∧ f x = 0 ∧ x = 3 :=
sorry

end NUMINAMATH_CALUDE_quadratic_other_intercept_l1584_158440
