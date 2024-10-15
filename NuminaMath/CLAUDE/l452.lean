import Mathlib

namespace NUMINAMATH_CALUDE_binomial_product_l452_45208

/-- The product of (2x² + 3y - 4) and (y + 6) is equal to 2x²y + 12x² + 3y² + 14y - 24 -/
theorem binomial_product (x y : ℝ) :
  (2 * x^2 + 3 * y - 4) * (y + 6) = 2 * x^2 * y + 12 * x^2 + 3 * y^2 + 14 * y - 24 := by
  sorry

end NUMINAMATH_CALUDE_binomial_product_l452_45208


namespace NUMINAMATH_CALUDE_wrapping_paper_area_l452_45228

/-- A rectangular box with dimensions l, w, and h, wrapped with a square sheet of paper -/
structure Box where
  l : ℝ  -- length
  w : ℝ  -- width
  h : ℝ  -- height
  l_gt_w : l > w

/-- The square sheet of wrapping paper -/
structure WrappingPaper where
  side : ℝ  -- side length of the square sheet

/-- The wrapping configuration -/
structure WrappingConfig (box : Box) (paper : WrappingPaper) where
  centered : Bool  -- box is centered on the paper
  vertices_on_midlines : Bool  -- vertices of longer side on paper midlines
  corners_meet_at_top : Bool  -- unoccupied corners meet at top center

theorem wrapping_paper_area (box : Box) (paper : WrappingPaper) 
    (config : WrappingConfig box paper) : paper.side^2 = 4 * box.l^2 := by
  sorry

#check wrapping_paper_area

end NUMINAMATH_CALUDE_wrapping_paper_area_l452_45228


namespace NUMINAMATH_CALUDE_smallest_angle_in_special_triangle_l452_45222

theorem smallest_angle_in_special_triangle :
  ∀ (a b c : ℝ),
    a > 0 ∧ b > 0 ∧ c > 0 →
    a + b + c = 180 →
    b = 3 * a →
    c = 5 * a →
    a = 20 := by
  sorry

end NUMINAMATH_CALUDE_smallest_angle_in_special_triangle_l452_45222


namespace NUMINAMATH_CALUDE_computation_proof_l452_45275

theorem computation_proof : 
  20 * (150 / 3 + 36 / 4 + 4 / 25 + 2) = 1223 + 1 / 5 := by
  sorry

end NUMINAMATH_CALUDE_computation_proof_l452_45275


namespace NUMINAMATH_CALUDE_car_speed_proof_l452_45299

/-- Proves that the speed of a car is 60 miles per hour given specific conditions -/
theorem car_speed_proof (
  fuel_efficiency : Real
) (
  tank_capacity : Real
) (
  travel_time : Real
) (
  fuel_used_ratio : Real
) (
  h1 : fuel_efficiency = 30 -- miles per gallon
) (
  h2 : tank_capacity = 12 -- gallons
) (
  h3 : travel_time = 5 -- hours
) (
  h4 : fuel_used_ratio = 0.8333333333333334 -- ratio of full tank
) : Real := by
  sorry

end NUMINAMATH_CALUDE_car_speed_proof_l452_45299


namespace NUMINAMATH_CALUDE_helga_usual_work_hours_l452_45205

/-- Helga's work schedule and article writing capacity -/
structure HelgaWork where
  articles_per_30min : ℕ
  days_per_week : ℕ
  extra_hours_thursday : ℕ
  extra_hours_friday : ℕ
  total_articles_this_week : ℕ

/-- Calculate Helga's usual daily work hours -/
def usual_daily_hours (hw : HelgaWork) : ℚ :=
  let articles_per_hour : ℚ := (hw.articles_per_30min : ℚ) * 2
  let total_hours_this_week : ℚ := (hw.total_articles_this_week : ℚ) / articles_per_hour
  let usual_hours_this_week : ℚ := total_hours_this_week - (hw.extra_hours_thursday + hw.extra_hours_friday)
  usual_hours_this_week / (hw.days_per_week : ℚ)

/-- Theorem: Helga usually works 4 hours each day -/
theorem helga_usual_work_hours (hw : HelgaWork)
  (h1 : hw.articles_per_30min = 5)
  (h2 : hw.days_per_week = 5)
  (h3 : hw.extra_hours_thursday = 2)
  (h4 : hw.extra_hours_friday = 3)
  (h5 : hw.total_articles_this_week = 250) :
  usual_daily_hours hw = 4 := by
  sorry

end NUMINAMATH_CALUDE_helga_usual_work_hours_l452_45205


namespace NUMINAMATH_CALUDE_triangle_side_value_l452_45253

/-- Given a triangle ABC with sides a, b, c opposite to angles A, B, C respectively,
    prove that if a = 3, c = 2√3, and bsinA = acos(B + π/6), then b = √3 -/
theorem triangle_side_value (A B C : ℝ) (a b c : ℝ) :
  a = 3 →
  c = 2 * Real.sqrt 3 →
  b * Real.sin A = a * Real.cos (B + π/6) →
  b = Real.sqrt 3 := by
sorry

end NUMINAMATH_CALUDE_triangle_side_value_l452_45253


namespace NUMINAMATH_CALUDE_triangle_perimeter_l452_45286

theorem triangle_perimeter (a b c : ℝ) (A B C : ℝ) : 
  A = π / 3 →  -- 60 degrees in radians
  (1 / 2) * b * c * Real.sin A = (15 * Real.sqrt 3) / 4 →  -- Area formula
  5 * Real.sin B = 3 * Real.sin C →
  a + b + c = 8 + Real.sqrt 19 := by
sorry

end NUMINAMATH_CALUDE_triangle_perimeter_l452_45286


namespace NUMINAMATH_CALUDE_triangle_abc_properties_l452_45284

theorem triangle_abc_properties (A B C : ℝ) (a b c : ℝ) :
  a = 2 →
  Real.sqrt 3 * Real.sin B - Real.cos B = 1 →
  b^2 = a * c →
  B = π / 3 ∧ (1/2) * a * c * Real.sin B = Real.sqrt 3 :=
by sorry

end NUMINAMATH_CALUDE_triangle_abc_properties_l452_45284


namespace NUMINAMATH_CALUDE_zoe_family_members_l452_45238

/-- Proves that Zoe is buying for 5 family members given the problem conditions -/
theorem zoe_family_members :
  let cost_per_person : ℚ := 3/2  -- $1.50
  let total_cost : ℚ := 9
  ∀ x : ℚ, (x + 1) * cost_per_person = total_cost → x = 5 :=
by
  sorry

end NUMINAMATH_CALUDE_zoe_family_members_l452_45238


namespace NUMINAMATH_CALUDE_remainder_sum_l452_45285

theorem remainder_sum (n : ℤ) (h : n % 21 = 13) : (n % 3 + n % 7 = 7) := by
  sorry

end NUMINAMATH_CALUDE_remainder_sum_l452_45285


namespace NUMINAMATH_CALUDE_min_sum_for_product_4410_l452_45202

theorem min_sum_for_product_4410 (a b c d : ℕ+) 
  (h : a * b * c * d = 4410) : 
  (∀ w x y z : ℕ+, w * x * y * z = 4410 → a + b + c + d ≤ w + x + y + z) ∧ 
  (∃ w x y z : ℕ+, w * x * y * z = 4410 ∧ w + x + y + z = 69) :=
by sorry

end NUMINAMATH_CALUDE_min_sum_for_product_4410_l452_45202


namespace NUMINAMATH_CALUDE_perimeter_of_C_l452_45261

-- Define squares A, B, and C
def square_A : Real → Real := λ s ↦ 4 * s
def square_B : Real → Real := λ s ↦ 4 * s
def square_C : Real → Real := λ s ↦ 4 * s

-- Define the conditions
def perimeter_A : Real := 20
def perimeter_B : Real := 40

-- Define the relationship between side lengths
def side_C (side_A side_B : Real) : Real := 2 * (side_A + side_B)

-- Theorem to prove
theorem perimeter_of_C (side_A side_B : Real) 
  (h1 : square_A side_A = perimeter_A)
  (h2 : square_B side_B = perimeter_B)
  : square_C (side_C side_A side_B) = 120 := by
  sorry


end NUMINAMATH_CALUDE_perimeter_of_C_l452_45261


namespace NUMINAMATH_CALUDE_expo_artworks_arrangements_l452_45210

/-- Represents the number of artworks of each type -/
structure ArtworkCounts where
  calligraphy : Nat
  paintings : Nat
  architectural : Nat

/-- Calculates the number of arrangements for the given artwork counts -/
def arrangeArtworks (counts : ArtworkCounts) : Nat :=
  sorry

/-- The specific artwork counts for the problem -/
def expoArtworks : ArtworkCounts :=
  { calligraphy := 2, paintings := 2, architectural := 1 }

/-- Theorem stating that the number of arrangements for the expo artworks is 36 -/
theorem expo_artworks_arrangements :
  arrangeArtworks expoArtworks = 36 := by
  sorry

end NUMINAMATH_CALUDE_expo_artworks_arrangements_l452_45210


namespace NUMINAMATH_CALUDE_angle_of_inclination_sqrt3_l452_45283

/-- The angle of inclination of a line with slope √3 is 60°. -/
theorem angle_of_inclination_sqrt3 :
  let slope : ℝ := Real.sqrt 3
  let angle : ℝ := 60 * π / 180  -- Convert 60° to radians
  Real.tan angle = slope := by sorry

end NUMINAMATH_CALUDE_angle_of_inclination_sqrt3_l452_45283


namespace NUMINAMATH_CALUDE_candy_probability_l452_45247

def total_candies : ℕ := 20
def red_candies : ℕ := 10
def blue_candies : ℕ := 10

def probability_same_combination : ℚ := 118 / 323

theorem candy_probability : 
  total_candies = red_candies + blue_candies →
  probability_same_combination = 
    (2 * (red_candies * (red_candies - 1) * (red_candies - 2) * (red_candies - 3) + 
          blue_candies * (blue_candies - 1) * (blue_candies - 2) * (blue_candies - 3)) + 
     6 * red_candies * (red_candies - 1) * blue_candies * (blue_candies - 1)) / 
    (total_candies * (total_candies - 1) * (total_candies - 2) * (total_candies - 3)) :=
by sorry

end NUMINAMATH_CALUDE_candy_probability_l452_45247


namespace NUMINAMATH_CALUDE_amy_soup_count_l452_45240

/-- The number of chicken soup cans Amy bought -/
def chicken_soup : ℕ := 6

/-- The number of tomato soup cans Amy bought -/
def tomato_soup : ℕ := 3

/-- The total number of soup cans Amy bought -/
def total_soup : ℕ := chicken_soup + tomato_soup

theorem amy_soup_count : total_soup = 9 := by
  sorry

end NUMINAMATH_CALUDE_amy_soup_count_l452_45240


namespace NUMINAMATH_CALUDE_abc_sum_sqrt_l452_45231

theorem abc_sum_sqrt (a b c : ℝ) 
  (h1 : b + c = 17)
  (h2 : c + a = 18)
  (h3 : a + b = 19) :
  Real.sqrt (a * b * c * (a + b + c)) = 36 * Real.sqrt 15 := by
  sorry

end NUMINAMATH_CALUDE_abc_sum_sqrt_l452_45231


namespace NUMINAMATH_CALUDE_round_85960_to_three_sig_figs_l452_45215

/-- Rounds a number to a specified number of significant figures using the round-half-up method -/
def roundToSigFigs (x : ℝ) (sigFigs : ℕ) : ℝ :=
  sorry

/-- Theorem: Rounding 85960 to three significant figures using the round-half-up method results in 8.60 × 10^4 -/
theorem round_85960_to_three_sig_figs :
  roundToSigFigs 85960 3 = 8.60 * (10 : ℝ)^4 :=
sorry

end NUMINAMATH_CALUDE_round_85960_to_three_sig_figs_l452_45215


namespace NUMINAMATH_CALUDE_sum_of_max_and_min_g_l452_45207

-- Define the function g(x)
def g (x : ℝ) : ℝ := |x - 3| + |x - 5| - |2*x - 8|

-- Define the interval [3, 10]
def I : Set ℝ := {x | 3 ≤ x ∧ x ≤ 10}

-- Theorem statement
theorem sum_of_max_and_min_g :
  ∃ (max_g min_g : ℝ),
    (∀ x ∈ I, g x ≤ max_g) ∧
    (∃ x ∈ I, g x = max_g) ∧
    (∀ x ∈ I, min_g ≤ g x) ∧
    (∃ x ∈ I, g x = min_g) ∧
    max_g + min_g = 2 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_max_and_min_g_l452_45207


namespace NUMINAMATH_CALUDE_max_M_value_l452_45269

def J (k : ℕ) : ℕ := 10^(k+2) + 100

def M (k : ℕ) : ℕ := (J k).factorization 2

theorem max_M_value :
  ∃ (k : ℕ), k > 0 ∧ M k = 4 ∧ ∀ (j : ℕ), j > 0 → M j ≤ 4 :=
sorry

end NUMINAMATH_CALUDE_max_M_value_l452_45269


namespace NUMINAMATH_CALUDE_log_product_simplification_l452_45237

theorem log_product_simplification (x y : ℝ) (hx : x > 0) (hy : y > 0) :
  (Real.log x / Real.log (y^6)) * (Real.log (y^2) / Real.log (x^5)) *
  (Real.log (x^3) / Real.log (y^4)) * (Real.log (y^4) / Real.log (x^3)) *
  (Real.log (x^5) / Real.log (y^2)) = (1/6) * (Real.log x / Real.log y) := by
  sorry

end NUMINAMATH_CALUDE_log_product_simplification_l452_45237


namespace NUMINAMATH_CALUDE_point_c_coordinates_l452_45297

/-- A point in 2D space -/
structure Point2D where
  x : ℝ
  y : ℝ

/-- The area of a triangle given three points -/
def triangleArea (a b c : Point2D) : ℝ := sorry

/-- Theorem: Given the conditions, point C has coordinates (0,4) or (0,-4) -/
theorem point_c_coordinates :
  let a : Point2D := ⟨-2, 0⟩
  let b : Point2D := ⟨3, 0⟩
  ∀ c : Point2D,
    c.x = 0 →  -- C lies on the y-axis
    triangleArea a b c = 10 →
    (c.y = 4 ∨ c.y = -4) :=
by sorry

end NUMINAMATH_CALUDE_point_c_coordinates_l452_45297


namespace NUMINAMATH_CALUDE_rectangle_frame_area_l452_45260

theorem rectangle_frame_area (a b : ℕ) (ha : a > 0) (hb : b > 0) : 
  a * b = ((a + 2) * (b + 2) - a * b) → 
  ((a = 3 ∧ b = 10) ∨ (a = 10 ∧ b = 3) ∨ (a = 4 ∧ b = 6) ∨ (a = 6 ∧ b = 4)) :=
by sorry

end NUMINAMATH_CALUDE_rectangle_frame_area_l452_45260


namespace NUMINAMATH_CALUDE_symmetry_axis_implies_a_equals_one_l452_45270

/-- The line equation -/
def line_equation (x y a : ℝ) : Prop := x - 2*a*y - 3 = 0

/-- The circle equation -/
def circle_equation (x y : ℝ) : Prop := x^2 + y^2 - 2*x + 2*y - 3 = 0

/-- The center of the circle -/
def circle_center : ℝ × ℝ := (1, -1)

/-- The theorem stating that if the line is a symmetry axis of the circle, then a = 1 -/
theorem symmetry_axis_implies_a_equals_one (a : ℝ) :
  (∀ x y : ℝ, line_equation x y a → circle_equation x y) →
  (line_equation (circle_center.1) (circle_center.2) a) →
  a = 1 :=
by sorry

end NUMINAMATH_CALUDE_symmetry_axis_implies_a_equals_one_l452_45270


namespace NUMINAMATH_CALUDE_quadratic_equation_roots_l452_45262

theorem quadratic_equation_roots (p q : ℤ) (h1 : p + q = 28) : 
  ∃ (x₁ x₂ : ℤ), x₁ > 0 ∧ x₂ > 0 ∧ x₁^2 + p*x₁ + q = 0 ∧ x₂^2 + p*x₂ + q = 0 ∧ 
  ((x₁ = 30 ∧ x₂ = 2) ∨ (x₁ = 2 ∧ x₂ = 30)) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_roots_l452_45262


namespace NUMINAMATH_CALUDE_dolphin_count_l452_45250

theorem dolphin_count (initial : ℕ) (joining_factor : ℕ) (h1 : initial = 65) (h2 : joining_factor = 3) :
  initial + joining_factor * initial = 260 := by
  sorry

end NUMINAMATH_CALUDE_dolphin_count_l452_45250


namespace NUMINAMATH_CALUDE_equal_fractions_imply_one_third_l452_45287

theorem equal_fractions_imply_one_third (x : ℝ) (h1 : x > 0) 
  (h2 : (2/3) * x = (16/216) * (1/x)) : x = 1/3 := by
  sorry

end NUMINAMATH_CALUDE_equal_fractions_imply_one_third_l452_45287


namespace NUMINAMATH_CALUDE_function_monotonicity_l452_45274

def is_monotonic (f : ℝ → ℝ) : Prop :=
  ∀ x y, x ≤ y → f x ≤ f y ∨ f y ≤ f x

theorem function_monotonicity (f : ℝ → ℝ) 
  (h : ∀ a b x, a < x ∧ x < b → min (f a) (f b) < f x ∧ f x < max (f a) (f b)) :
  is_monotonic f := by
  sorry

end NUMINAMATH_CALUDE_function_monotonicity_l452_45274


namespace NUMINAMATH_CALUDE_leahs_coins_value_l452_45214

theorem leahs_coins_value (d n : ℕ) : 
  d + n = 15 ∧ 
  d = 2 * (n + 3) → 
  10 * d + 5 * n = 135 :=
by sorry

end NUMINAMATH_CALUDE_leahs_coins_value_l452_45214


namespace NUMINAMATH_CALUDE_school_sections_l452_45227

theorem school_sections (boys girls : ℕ) (h1 : boys = 408) (h2 : girls = 192) :
  let gcd := Nat.gcd boys girls
  let boys_sections := boys / gcd
  let girls_sections := girls / gcd
  boys_sections + girls_sections = 25 := by
sorry

end NUMINAMATH_CALUDE_school_sections_l452_45227


namespace NUMINAMATH_CALUDE_smallest_positive_solution_sqrt_equation_l452_45232

theorem smallest_positive_solution_sqrt_equation :
  let f : ℝ → ℝ := λ x ↦ Real.sqrt (3 * x) - (5 * x - 1)
  ∃! x : ℝ, x > 0 ∧ f x = 0 ∧ ∀ y : ℝ, y > 0 ∧ f y = 0 → x ≤ y :=
by
  sorry

end NUMINAMATH_CALUDE_smallest_positive_solution_sqrt_equation_l452_45232


namespace NUMINAMATH_CALUDE_recurrence_2004_values_l452_45201

/-- A sequence of real numbers satisfying the given recurrence relation -/
def RecurrenceSequence (a : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, a (n + 1) = a n * (a n + 2)

/-- The set of possible values for the 2004th term of the sequence -/
def PossibleValues (a : ℕ → ℝ) : Set ℝ :=
  {x : ℝ | ∃ (seq : ℕ → ℝ), RecurrenceSequence seq ∧ seq 2004 = x}

/-- The theorem stating that the set of possible values for a₂₀₀₄ is [-1, ∞) -/
theorem recurrence_2004_values :
  ∀ a : ℕ → ℝ, RecurrenceSequence a →
  PossibleValues a = Set.Ici (-1) :=
sorry

end NUMINAMATH_CALUDE_recurrence_2004_values_l452_45201


namespace NUMINAMATH_CALUDE_meaningful_expression_l452_45264

theorem meaningful_expression (x : ℝ) : 
  (∃ y : ℝ, y = (Real.sqrt (x + 1)) / x) ↔ x ≥ -1 ∧ x ≠ 0 := by
  sorry

end NUMINAMATH_CALUDE_meaningful_expression_l452_45264


namespace NUMINAMATH_CALUDE_officer_assignment_count_l452_45221

def group_members : Nat := 4
def officer_positions : Nat := 3

theorem officer_assignment_count : 
  group_members ^ officer_positions = 64 := by
  sorry

end NUMINAMATH_CALUDE_officer_assignment_count_l452_45221


namespace NUMINAMATH_CALUDE_negation_equivalence_l452_45223

theorem negation_equivalence :
  (¬ ∃ x : ℝ, x^2 + 3*x + 2 < 0) ↔ (∀ x : ℝ, x^2 + 3*x + 2 ≥ 0) :=
by sorry

end NUMINAMATH_CALUDE_negation_equivalence_l452_45223


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_l452_45251

theorem quadratic_inequality_solution (x : ℝ) : x^2 - 7*x + 10 < 0 ↔ 2 < x ∧ x < 5 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_l452_45251


namespace NUMINAMATH_CALUDE_distance_P_to_xaxis_l452_45218

/-- The distance from a point to the x-axis in a Cartesian coordinate system -/
def distanceToXAxis (x y : ℝ) : ℝ := |y|

/-- The point P -/
def P : ℝ × ℝ := (2, -3)

/-- Theorem: The distance from point P(2, -3) to the x-axis is 3 -/
theorem distance_P_to_xaxis : distanceToXAxis P.1 P.2 = 3 := by
  sorry

end NUMINAMATH_CALUDE_distance_P_to_xaxis_l452_45218


namespace NUMINAMATH_CALUDE_largest_power_of_five_dividing_sum_l452_45268

def factorial (n : ℕ) : ℕ := (List.range n).foldl (·*·) 1

def sum_of_factorials : ℕ := factorial 50 + factorial 52 + factorial 54

theorem largest_power_of_five_dividing_sum : 
  (∃ (k : ℕ), sum_of_factorials = 5^12 * k ∧ ¬(∃ (m : ℕ), sum_of_factorials = 5^13 * m)) := by
  sorry

end NUMINAMATH_CALUDE_largest_power_of_five_dividing_sum_l452_45268


namespace NUMINAMATH_CALUDE_pages_left_to_read_l452_45255

/-- Calculates the number of pages left to read in a storybook -/
theorem pages_left_to_read (a b : ℕ) : a - 8 * b = a - 8 * b :=
by
  sorry

#check pages_left_to_read

end NUMINAMATH_CALUDE_pages_left_to_read_l452_45255


namespace NUMINAMATH_CALUDE_ratio_and_equation_solution_l452_45281

theorem ratio_and_equation_solution (a b : ℝ) : 
  b / a = 4 → b = 16 - 6 * a + a^2 → (a = -5 + Real.sqrt 41 ∨ a = -5 - Real.sqrt 41) :=
by sorry

end NUMINAMATH_CALUDE_ratio_and_equation_solution_l452_45281


namespace NUMINAMATH_CALUDE_elder_person_age_l452_45254

/-- Proves that given two persons whose ages differ by 16 years, and 6 years ago the elder one was 3 times as old as the younger one, the present age of the elder person is 30 years. -/
theorem elder_person_age (y e : ℕ) : 
  e = y + 16 → 
  e - 6 = 3 * (y - 6) → 
  e = 30 :=
by sorry

end NUMINAMATH_CALUDE_elder_person_age_l452_45254


namespace NUMINAMATH_CALUDE_sum_of_reciprocals_of_roots_l452_45239

theorem sum_of_reciprocals_of_roots (m n : ℝ) 
  (hm : m^2 + 3*m + 5 = 0) 
  (hn : n^2 + 3*n + 5 = 0) : 
  1/n + 1/m = -3/5 := by sorry

end NUMINAMATH_CALUDE_sum_of_reciprocals_of_roots_l452_45239


namespace NUMINAMATH_CALUDE_negative_quarter_to_11_times_negative_four_to_12_l452_45217

theorem negative_quarter_to_11_times_negative_four_to_12 :
  (-0.25)^11 * (-4)^12 = -4 := by
  sorry

end NUMINAMATH_CALUDE_negative_quarter_to_11_times_negative_four_to_12_l452_45217


namespace NUMINAMATH_CALUDE_cherry_tomato_jars_l452_45291

theorem cherry_tomato_jars (total_tomatoes : ℕ) (tomatoes_per_jar : ℕ) (h1 : total_tomatoes = 550) (h2 : tomatoes_per_jar = 14) :
  ∃ (jars : ℕ), jars = ((total_tomatoes + tomatoes_per_jar - 1) / tomatoes_per_jar) ∧ jars = 40 :=
by sorry

end NUMINAMATH_CALUDE_cherry_tomato_jars_l452_45291


namespace NUMINAMATH_CALUDE_line_perpendicular_to_plane_l452_45230

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the parallel and perpendicular relations
variable (parallel : Line → Line → Prop)
variable (perpendicular : Line → Plane → Prop)

-- State the theorem
theorem line_perpendicular_to_plane 
  (m n : Line) (α : Plane) 
  (h1 : parallel m n) 
  (h2 : perpendicular m α) : 
  perpendicular n α :=
sorry

end NUMINAMATH_CALUDE_line_perpendicular_to_plane_l452_45230


namespace NUMINAMATH_CALUDE_largest_divisor_of_P_l452_45246

def P (n : ℕ) : ℕ := (n + 1) * (n + 3) * (n + 5) * (n + 7) * (n + 9)

theorem largest_divisor_of_P (n : ℕ) (h : Even n) (k : ℕ) :
  (∀ m : ℕ, Even m → k ∣ P m) → k ≤ 15 :=
by sorry

end NUMINAMATH_CALUDE_largest_divisor_of_P_l452_45246


namespace NUMINAMATH_CALUDE_problem_1_problem_2_l452_45282

-- Problem 1
theorem problem_1 : Real.sqrt 27 / Real.sqrt 3 + Real.sqrt 12 * Real.sqrt (1/3) - Real.sqrt 5 = 5 - Real.sqrt 5 := by
  sorry

-- Problem 2
theorem problem_2 : (Real.sqrt 5 + 2) * (Real.sqrt 5 - 2) + (2 * Real.sqrt 3 + 1)^2 = 14 + 4 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_problem_1_problem_2_l452_45282


namespace NUMINAMATH_CALUDE_expression_evaluations_l452_45200

theorem expression_evaluations :
  -- Part 1
  (25 ^ (1/3) - 125 ^ (1/2)) / (5 ^ (1/4)) = 5 ^ (5/12) - 5 * (5 ^ (1/4)) ∧
  -- Part 2
  ∀ a : ℝ, a > 0 → a^2 / (a^(1/2) * a^(2/3)) = a^(5/6) := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluations_l452_45200


namespace NUMINAMATH_CALUDE_trains_catch_up_catch_up_at_ten_pm_l452_45236

/-- The time (in hours after 3:00 pm) when the second train catches the first train -/
def catch_up_time : ℝ := 7

/-- The speed of the first train in km/h -/
def speed_train1 : ℝ := 70

/-- The speed of the second train in km/h -/
def speed_train2 : ℝ := 80

/-- The time difference between the trains' departure times in hours -/
def time_difference : ℝ := 1

theorem trains_catch_up : 
  speed_train1 * (catch_up_time + time_difference) = speed_train2 * catch_up_time := by
  sorry

theorem catch_up_at_ten_pm : 
  catch_up_time = 7 := by
  sorry

end NUMINAMATH_CALUDE_trains_catch_up_catch_up_at_ten_pm_l452_45236


namespace NUMINAMATH_CALUDE_pat_calculation_error_l452_45220

theorem pat_calculation_error (x : ℝ) : 
  (x / 7 - 20 = 13) → (7 * x + 20 > 1100) := by
  sorry

end NUMINAMATH_CALUDE_pat_calculation_error_l452_45220


namespace NUMINAMATH_CALUDE_right_triangle_and_inverse_mod_l452_45213

theorem right_triangle_and_inverse_mod : 
  (60^2 + 144^2 = 156^2) ∧ 
  (∃ n : ℕ, n < 3751 ∧ (300 * n) % 3751 = 1) :=
by sorry

end NUMINAMATH_CALUDE_right_triangle_and_inverse_mod_l452_45213


namespace NUMINAMATH_CALUDE_students_count_l452_45296

/-- The total number of students in a rectangular arrangement -/
def total_students (left right front back : ℕ) : ℕ :=
  (left + right - 1) * (front + back - 1)

/-- Theorem stating that the total number of students is 399 -/
theorem students_count (left right front back : ℕ) 
  (h1 : left = 7)
  (h2 : right = 13)
  (h3 : front = 8)
  (h4 : back = 14) :
  total_students left right front back = 399 := by
  sorry

#eval total_students 7 13 8 14

end NUMINAMATH_CALUDE_students_count_l452_45296


namespace NUMINAMATH_CALUDE_point_transformation_l452_45249

/-- Rotation of a point (x, y) by 90° counterclockwise around (h, k) -/
def rotate90 (x y h k : ℝ) : ℝ × ℝ :=
  (k - (y - h) + h, h + (x - h) + k)

/-- Reflection of a point (x, y) about the line y = x -/
def reflectYEqualsX (x y : ℝ) : ℝ × ℝ :=
  (y, x)

theorem point_transformation (a b : ℝ) :
  let (x₁, y₁) := rotate90 a b 2 3
  let (x₂, y₂) := reflectYEqualsX x₁ y₁
  (x₂ = -3 ∧ y₂ = 1) → b - a = -6 := by
  sorry

end NUMINAMATH_CALUDE_point_transformation_l452_45249


namespace NUMINAMATH_CALUDE_factorization_problems_l452_45245

theorem factorization_problems :
  (∀ m : ℝ, m * (m - 3) + 3 * (3 - m) = (m - 3)^2) ∧
  (∀ x : ℝ, 4 * x^3 - 12 * x^2 + 9 * x = x * (2 * x - 3)^2) := by
  sorry

end NUMINAMATH_CALUDE_factorization_problems_l452_45245


namespace NUMINAMATH_CALUDE_min_bird_species_l452_45233

theorem min_bird_species (total_birds : ℕ) (h_total : total_birds = 2021) :
  let min_species := (total_birds + 1) / 2
  ∀ (num_species : ℕ),
    (∀ (i j : ℕ) (species : ℕ → ℕ),
      i < j ∧ j < total_birds ∧ species i = species j →
      ∃ (k : ℕ), k ∈ Finset.range (j - i - 1) ∧ species (i + k + 1) ≠ species i) →
    num_species ≥ min_species :=
by sorry

end NUMINAMATH_CALUDE_min_bird_species_l452_45233


namespace NUMINAMATH_CALUDE_circle_center_tangent_parabola_l452_45277

/-- A circle that passes through (1,0) and is tangent to y = x^2 at (1,1) has its center at (1,1) -/
theorem circle_center_tangent_parabola : 
  ∀ (center : ℝ × ℝ),
  (∀ (p : ℝ × ℝ), p.1^2 = p.2 → (center.1 - p.1)^2 + (center.2 - p.2)^2 = (center.1 - 1)^2 + center.2^2) →
  (center.1 - 1)^2 + (center.2 - 1)^2 = (center.1 - 1)^2 + center.2^2 →
  center = (1, 1) :=
by sorry

end NUMINAMATH_CALUDE_circle_center_tangent_parabola_l452_45277


namespace NUMINAMATH_CALUDE_sector_radius_l452_45209

theorem sector_radius (A : ℝ) (θ : ℝ) (r : ℝ) : 
  A = 6 * Real.pi → θ = (4 * Real.pi) / 3 → A = (1/2) * r^2 * θ → r = 3 := by
  sorry

end NUMINAMATH_CALUDE_sector_radius_l452_45209


namespace NUMINAMATH_CALUDE_amandas_quiz_average_l452_45298

theorem amandas_quiz_average :
  ∀ (num_quizzes : ℕ) (final_quiz_score : ℝ) (required_average : ℝ),
    num_quizzes = 4 →
    final_quiz_score = 97 →
    required_average = 93 →
    ∃ (current_average : ℝ),
      current_average = 92 ∧
      (num_quizzes : ℝ) * current_average + final_quiz_score = (num_quizzes + 1 : ℝ) * required_average :=
by
  sorry

end NUMINAMATH_CALUDE_amandas_quiz_average_l452_45298


namespace NUMINAMATH_CALUDE_total_product_weight_is_correct_l452_45289

/-- Represents a chemical element or compound -/
structure Chemical where
  formula : String
  molarMass : Float

/-- Represents a chemical reaction -/
structure Reaction where
  reactants : List (Chemical × Float)
  products : List (Chemical × Float)

def CaCO3 : Chemical := ⟨"CaCO3", 100.09⟩
def CaO : Chemical := ⟨"CaO", 56.08⟩
def CO2 : Chemical := ⟨"CO2", 44.01⟩
def HCl : Chemical := ⟨"HCl", 36.46⟩
def CaCl2 : Chemical := ⟨"CaCl2", 110.98⟩
def H2O : Chemical := ⟨"H2O", 18.02⟩

def reaction1 : Reaction := ⟨[(CaCO3, 1)], [(CaO, 1), (CO2, 1)]⟩
def reaction2 : Reaction := ⟨[(HCl, 2), (CaCO3, 1)], [(CaCl2, 1), (CO2, 1), (H2O, 1)]⟩

def initialCaCO3 : Float := 8
def initialHCl : Float := 12

/-- Calculates the total weight of products from both reactions -/
def totalProductWeight (r1 : Reaction) (r2 : Reaction) (initCaCO3 : Float) (initHCl : Float) : Float :=
  sorry

theorem total_product_weight_is_correct :
  totalProductWeight reaction1 reaction2 initialCaCO3 initialHCl = 800.72 := by sorry

end NUMINAMATH_CALUDE_total_product_weight_is_correct_l452_45289


namespace NUMINAMATH_CALUDE_johns_recycling_money_l452_45203

/-- The weight of a Monday-Saturday newspaper in ounces -/
def weekdayPaperWeight : ℕ := 8

/-- The weight of a Sunday newspaper in ounces -/
def sundayPaperWeight : ℕ := 2 * weekdayPaperWeight

/-- The number of papers John is supposed to deliver daily -/
def dailyPapers : ℕ := 250

/-- The number of weeks John steals the papers -/
def stolenWeeks : ℕ := 10

/-- The recycling value of one ton of paper in dollars -/
def recyclingValuePerTon : ℕ := 20

/-- The number of ounces in a ton -/
def ouncesPerTon : ℕ := 32000

/-- Calculate the total money John makes from recycling stolen newspapers -/
def johnsMoney : ℚ :=
  let totalWeekdayWeight := 6 * stolenWeeks * dailyPapers * weekdayPaperWeight
  let totalSundayWeight := stolenWeeks * dailyPapers * sundayPaperWeight
  let totalWeight := totalWeekdayWeight + totalSundayWeight
  let weightInTons := totalWeight / ouncesPerTon
  weightInTons * recyclingValuePerTon

/-- Theorem stating that John makes $100 from recycling the stolen newspapers -/
theorem johns_recycling_money : johnsMoney = 100 := by
  sorry

end NUMINAMATH_CALUDE_johns_recycling_money_l452_45203


namespace NUMINAMATH_CALUDE_sandbox_length_l452_45292

/-- The length of a rectangular sandbox given its width and area -/
theorem sandbox_length (width : ℝ) (area : ℝ) (h1 : width = 146) (h2 : area = 45552) :
  area / width = 312 := by
  sorry

end NUMINAMATH_CALUDE_sandbox_length_l452_45292


namespace NUMINAMATH_CALUDE_tangent_line_equation_chord_line_equation_l452_45211

-- Define the circle C
def C (x y : ℝ) : Prop := x^2 + y^2 - 8*y + 12 = 0

-- Define the point P
def P : ℝ × ℝ := (-2, 0)

-- Define a line passing through P
def line_through_P (k : ℝ) (x y : ℝ) : Prop := y = k * (x + 2)

-- Define tangent line condition
def is_tangent (k : ℝ) : Prop :=
  ∃ (x y : ℝ), C x y ∧ line_through_P k x y ∧
  ∀ (x' y' : ℝ), C x' y' ∧ line_through_P k x' y' → (x', y') = (x, y)

-- Define chord length condition
def has_chord_length (k : ℝ) (len : ℝ) : Prop :=
  ∃ (x₁ y₁ x₂ y₂ : ℝ), C x₁ y₁ ∧ C x₂ y₂ ∧
  line_through_P k x₁ y₁ ∧ line_through_P k x₂ y₂ ∧
  (x₁ - x₂)^2 + (y₁ - y₂)^2 = len^2

-- Theorem for part (1)
theorem tangent_line_equation :
  ∀ k : ℝ, is_tangent k ↔ (k = 0 ∨ 3 * k = 4) :=
sorry

-- Theorem for part (2)
theorem chord_line_equation :
  ∀ k : ℝ, has_chord_length k (2 * Real.sqrt 2) ↔ (k = 1 ∨ k = 7) :=
sorry

end NUMINAMATH_CALUDE_tangent_line_equation_chord_line_equation_l452_45211


namespace NUMINAMATH_CALUDE_sum_of_roots_greater_than_five_l452_45258

theorem sum_of_roots_greater_than_five (a b c : ℝ) 
  (ha : a > 0) (hb : b > 0) (hc : c > 0) 
  (sum_eq_one : a + b + c = 1) :
  Real.sqrt (4 * a + 1) + Real.sqrt (4 * b + 1) + Real.sqrt (4 * c + 1) > 5 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_roots_greater_than_five_l452_45258


namespace NUMINAMATH_CALUDE_complex_modulus_l452_45272

theorem complex_modulus (z : ℂ) (h : (1 + Complex.I) * z = 2 * Complex.I) : 
  Complex.abs z = Real.sqrt 2 := by sorry

end NUMINAMATH_CALUDE_complex_modulus_l452_45272


namespace NUMINAMATH_CALUDE_simplify_expression_l452_45248

theorem simplify_expression (a : ℝ) (ha : a > 0) :
  a^2 / (a * (a^3)^(1/2))^(1/3) = a^(7/6) := by sorry

end NUMINAMATH_CALUDE_simplify_expression_l452_45248


namespace NUMINAMATH_CALUDE_least_positive_angle_phi_l452_45295

theorem least_positive_angle_phi : 
  ∃ φ : ℝ, φ > 0 ∧ φ ≤ π/2 ∧ 
  (∀ ψ : ℝ, ψ > 0 → ψ < φ → Real.cos (10 * π/180) ≠ Real.sin (15 * π/180) + Real.sin ψ) ∧
  Real.cos (10 * π/180) = Real.sin (15 * π/180) + Real.sin φ ∧
  φ = 42.5 * π/180 :=
sorry

end NUMINAMATH_CALUDE_least_positive_angle_phi_l452_45295


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l452_45271

-- Define the sets A and B
def A : Set ℝ := {x : ℝ | -1 ≤ x ∧ x < 2}
def B : Set ℝ := {x : ℝ | x < 1}

-- State the theorem
theorem intersection_of_A_and_B :
  A ∩ B = {x : ℝ | -1 ≤ x ∧ x < 1} := by sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l452_45271


namespace NUMINAMATH_CALUDE_train_crossing_time_l452_45278

/-- The time taken for a train to cross a man walking in the same direction -/
theorem train_crossing_time (train_length : Real) (train_speed : Real) (man_speed : Real) :
  train_length = 500 ∧ 
  train_speed = 63 * 1000 / 3600 ∧ 
  man_speed = 3 * 1000 / 3600 →
  (train_length / (train_speed - man_speed)) = 30 := by
sorry

end NUMINAMATH_CALUDE_train_crossing_time_l452_45278


namespace NUMINAMATH_CALUDE_opposite_of_negative_2023_l452_45235

-- Define the concept of opposite for integers
def opposite (n : ℤ) : ℤ := -n

-- Theorem stating that the opposite of -2023 is 2023
theorem opposite_of_negative_2023 : opposite (-2023) = 2023 := by
  sorry

end NUMINAMATH_CALUDE_opposite_of_negative_2023_l452_45235


namespace NUMINAMATH_CALUDE_trigonometric_expression_equals_sqrt_three_l452_45273

theorem trigonometric_expression_equals_sqrt_three (α : Real) (h : α = -35 * Real.pi / 6) :
  (2 * Real.sin (Real.pi + α) * Real.cos (Real.pi - α) - Real.cos (Real.pi + α)) /
  (1 + Real.sin α ^ 2 + Real.sin (Real.pi - α) - Real.cos (Real.pi + α) ^ 2) = Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_trigonometric_expression_equals_sqrt_three_l452_45273


namespace NUMINAMATH_CALUDE_vanessa_missed_days_l452_45288

/-- Represents the number of days missed by each student -/
structure MissedDays where
  vanessa : ℕ
  mike : ℕ
  sarah : ℕ

/-- The conditions of the problem -/
def satisfiesConditions (d : MissedDays) : Prop :=
  d.vanessa + d.mike + d.sarah = 17 ∧
  d.vanessa + d.mike = 14 ∧
  d.mike + d.sarah = 12

/-- The theorem to prove -/
theorem vanessa_missed_days (d : MissedDays) (h : satisfiesConditions d) : d.vanessa = 5 := by
  sorry

end NUMINAMATH_CALUDE_vanessa_missed_days_l452_45288


namespace NUMINAMATH_CALUDE_min_x_plus_y_l452_45290

def is_median (x : ℝ) : Prop := 
  x ≥ 2 ∧ x ≤ 4

def average_condition (x y : ℝ) : Prop :=
  (-1 + 5 + (-1/x) + y) / 4 = 3

theorem min_x_plus_y (x y : ℝ) 
  (h1 : is_median x) 
  (h2 : average_condition x y) : 
  x + y ≥ 21/2 := by
  sorry

end NUMINAMATH_CALUDE_min_x_plus_y_l452_45290


namespace NUMINAMATH_CALUDE_sequence_gcd_property_l452_45224

/-- Given a sequence of natural numbers satisfying the GCD property, prove that a_i = i for all i. -/
theorem sequence_gcd_property (a : ℕ → ℕ) 
  (h : ∀ (i j : ℕ), i ≠ j → Nat.gcd (a i) (a j) = Nat.gcd i j) :
  ∀ (i : ℕ), a i = i :=
by sorry

end NUMINAMATH_CALUDE_sequence_gcd_property_l452_45224


namespace NUMINAMATH_CALUDE_school_purchase_options_l452_45242

theorem school_purchase_options : 
  let valid_purchase := λ (x y : ℕ) => x ≥ 8 ∧ y ≥ 2 ∧ 120 * x + 140 * y ≤ 1500
  ∃! (n : ℕ), ∃ (S : Finset (ℕ × ℕ)), 
    S.card = n ∧ 
    (∀ (p : ℕ × ℕ), p ∈ S ↔ valid_purchase p.1 p.2) ∧
    n = 5 :=
by sorry

end NUMINAMATH_CALUDE_school_purchase_options_l452_45242


namespace NUMINAMATH_CALUDE_popsicle_sticks_left_l452_45280

/-- Calculates the number of popsicle sticks Miss Davis has left after distribution -/
theorem popsicle_sticks_left (initial_sticks : ℕ) (sticks_per_group : ℕ) (num_groups : ℕ) : 
  initial_sticks = 170 → sticks_per_group = 15 → num_groups = 10 → 
  initial_sticks - (sticks_per_group * num_groups) = 20 := by
sorry

end NUMINAMATH_CALUDE_popsicle_sticks_left_l452_45280


namespace NUMINAMATH_CALUDE_greatest_non_expressible_as_sum_of_composites_l452_45212

-- Define what it means for a number to be composite
def IsComposite (n : ℕ) : Prop := n > 1 ∧ ¬(Nat.Prime n)

-- Define the property of being expressible as the sum of two composite numbers
def ExpressibleAsSumOfTwoComposites (n : ℕ) : Prop :=
  ∃ (a b : ℕ), IsComposite a ∧ IsComposite b ∧ n = a + b

-- State the theorem
theorem greatest_non_expressible_as_sum_of_composites :
  (∀ n > 11, ExpressibleAsSumOfTwoComposites n) ∧
  ¬(ExpressibleAsSumOfTwoComposites 11) := by sorry

end NUMINAMATH_CALUDE_greatest_non_expressible_as_sum_of_composites_l452_45212


namespace NUMINAMATH_CALUDE_remainder_thirteen_power_thirteen_plus_thirteen_mod_fourteen_l452_45259

theorem remainder_thirteen_power_thirteen_plus_thirteen_mod_fourteen :
  (13^13 + 13) % 14 = 12 := by
  sorry

end NUMINAMATH_CALUDE_remainder_thirteen_power_thirteen_plus_thirteen_mod_fourteen_l452_45259


namespace NUMINAMATH_CALUDE_foci_distance_of_hyperbola_l452_45252

def hyperbola_equation (x y : ℝ) : Prop :=
  9 * x^2 - 18 * x - 16 * y^2 - 32 * y = -144

theorem foci_distance_of_hyperbola :
  ∃ (h : ℝ → ℝ → ℝ), 
    (∀ x y, hyperbola_equation x y → 
      h x y = (let a := 4; let b := 3; Real.sqrt (a^2 + b^2) * 2)) ∧
    (∀ x y, hyperbola_equation x y → h x y = 10) :=
sorry

end NUMINAMATH_CALUDE_foci_distance_of_hyperbola_l452_45252


namespace NUMINAMATH_CALUDE_rectangles_on_grid_l452_45257

/-- The number of rectangles on a 4x4 grid with 5 points in each direction -/
def num_rectangles : ℕ := 100

/-- The number of points in each direction of the grid -/
def points_per_direction : ℕ := 5

/-- Theorem stating the number of rectangles on the grid -/
theorem rectangles_on_grid : 
  (Nat.choose points_per_direction 2) * (Nat.choose points_per_direction 2) = num_rectangles := by
  sorry

end NUMINAMATH_CALUDE_rectangles_on_grid_l452_45257


namespace NUMINAMATH_CALUDE_probability_of_one_in_first_20_rows_l452_45216

/-- Calculates the number of elements in the first n rows of Pascal's Triangle. -/
def elementsInRows (n : ℕ) : ℕ := n * (n + 1) / 2

/-- Calculates the number of ones in the first n rows of Pascal's Triangle. -/
def onesInRows (n : ℕ) : ℕ := if n = 0 then 1 else 2 * n - 1

/-- The probability of selecting a 1 from the first n rows of Pascal's Triangle. -/
def probabilityOfOne (n : ℕ) : ℚ :=
  (onesInRows n) / (elementsInRows n)

theorem probability_of_one_in_first_20_rows :
  probabilityOfOne 20 = 13 / 70 := by
  sorry

end NUMINAMATH_CALUDE_probability_of_one_in_first_20_rows_l452_45216


namespace NUMINAMATH_CALUDE_feeding_sequences_count_l452_45229

/-- Represents the number of animal pairs in the zoo -/
def num_pairs : Nat := 4

/-- Represents the constraint of alternating genders when feeding -/
def alternating_genders : Bool := true

/-- Represents the condition of starting with a specific male animal -/
def starts_with_male : Bool := true

/-- Calculates the number of possible feeding sequences -/
def feeding_sequences : Nat :=
  (num_pairs) * (num_pairs - 1) * (num_pairs - 1) * (num_pairs - 2) * (num_pairs - 2)

/-- Theorem stating that the number of possible feeding sequences is 144 -/
theorem feeding_sequences_count :
  alternating_genders ∧ starts_with_male → feeding_sequences = 144 := by
  sorry

end NUMINAMATH_CALUDE_feeding_sequences_count_l452_45229


namespace NUMINAMATH_CALUDE_initial_tickets_l452_45279

/-- 
Given that a person spends some tickets and has some tickets left,
this theorem proves the total number of tickets they initially had.
-/
theorem initial_tickets (spent : ℕ) (left : ℕ) : 
  spent = 3 → left = 8 → spent + left = 11 := by
  sorry

end NUMINAMATH_CALUDE_initial_tickets_l452_45279


namespace NUMINAMATH_CALUDE_circular_garden_area_l452_45256

-- Define the radius of the garden
def radius : ℝ := 16

-- Define the relationship between circumference and area
def fence_area_relation (circumference area : ℝ) : Prop :=
  circumference = (1/8) * area

-- Theorem statement
theorem circular_garden_area :
  let circumference := 2 * Real.pi * radius
  let area := Real.pi * radius^2
  fence_area_relation circumference area →
  area = 256 * Real.pi := by
sorry

end NUMINAMATH_CALUDE_circular_garden_area_l452_45256


namespace NUMINAMATH_CALUDE_dakota_medical_bill_l452_45225

def hospital_stay_days : ℕ := 3
def bed_cost_per_day : ℚ := 900
def specialist_cost_per_hour : ℚ := 250
def specialist_time_hours : ℚ := 1/4
def num_specialists : ℕ := 2
def ambulance_cost : ℚ := 1800

theorem dakota_medical_bill : 
  hospital_stay_days * bed_cost_per_day + 
  specialist_cost_per_hour * specialist_time_hours * num_specialists +
  ambulance_cost = 4750 := by
  sorry

end NUMINAMATH_CALUDE_dakota_medical_bill_l452_45225


namespace NUMINAMATH_CALUDE_roots_of_equation_l452_45206

theorem roots_of_equation : 
  ∀ x : ℝ, (x^3 - 2*x^2 - x + 2)*(x - 5) = 0 ↔ x = -1 ∨ x = 1 ∨ x = 2 ∨ x = 5 := by
  sorry

end NUMINAMATH_CALUDE_roots_of_equation_l452_45206


namespace NUMINAMATH_CALUDE_c_investment_is_half_l452_45244

/-- Represents the investment of a partner in a partnership --/
structure Investment where
  capital : ℚ  -- Fraction of total capital invested
  time : ℚ     -- Fraction of total time invested

/-- Represents a partnership with three investors --/
structure Partnership where
  a : Investment
  b : Investment
  c : Investment
  total_profit : ℚ
  a_share : ℚ

/-- The theorem stating that given the conditions of the problem, C's investment is 1/2 of the total capital --/
theorem c_investment_is_half (p : Partnership) : 
  p.a = ⟨1/6, 1/6⟩ → 
  p.b = ⟨1/3, 1/3⟩ → 
  p.c.time = 1 →
  p.total_profit = 2300 →
  p.a_share = 100 →
  p.c.capital = 1/2 := by
  sorry


end NUMINAMATH_CALUDE_c_investment_is_half_l452_45244


namespace NUMINAMATH_CALUDE_davids_chemistry_marks_l452_45226

theorem davids_chemistry_marks
  (english : ℕ)
  (mathematics : ℕ)
  (physics : ℕ)
  (biology : ℕ)
  (chemistry : ℕ)
  (average : ℕ)
  (h1 : english = 76)
  (h2 : mathematics = 65)
  (h3 : physics = 82)
  (h4 : biology = 85)
  (h5 : average = 75)
  (h6 : (english + mathematics + physics + biology + chemistry) / 5 = average) :
  chemistry = 67 := by
sorry

end NUMINAMATH_CALUDE_davids_chemistry_marks_l452_45226


namespace NUMINAMATH_CALUDE_no_solution_implies_m_equals_two_l452_45263

theorem no_solution_implies_m_equals_two :
  (∀ x : ℝ, (2 - m) / (1 - x) ≠ 1) → m = 2 := by
  sorry

end NUMINAMATH_CALUDE_no_solution_implies_m_equals_two_l452_45263


namespace NUMINAMATH_CALUDE_dislike_radio_and_music_l452_45241

theorem dislike_radio_and_music (total_people : ℕ) 
  (radio_dislike_percent : ℚ) (music_dislike_percent : ℚ) :
  total_people = 1500 →
  radio_dislike_percent = 25 / 100 →
  music_dislike_percent = 15 / 100 →
  ⌊(total_people : ℚ) * radio_dislike_percent * music_dislike_percent⌋ = 56 :=
by sorry

end NUMINAMATH_CALUDE_dislike_radio_and_music_l452_45241


namespace NUMINAMATH_CALUDE_crackers_distribution_l452_45204

theorem crackers_distribution (total_crackers : ℕ) (num_friends : ℕ) (crackers_per_friend : ℕ) :
  total_crackers = 8 →
  num_friends = 4 →
  total_crackers = num_friends * crackers_per_friend →
  crackers_per_friend = 2 := by
sorry

end NUMINAMATH_CALUDE_crackers_distribution_l452_45204


namespace NUMINAMATH_CALUDE_unique_root_between_zero_and_e_l452_45294

/-- The natural logarithm function -/
noncomputable def ln : ℝ → ℝ := Real.log

/-- The mathematical constant e -/
noncomputable def e : ℝ := Real.exp 1

theorem unique_root_between_zero_and_e (a : ℝ) (h1 : 0 < a) (h2 : a < e) :
  ∃! x : ℝ, x = ln (a * x) := by sorry

end NUMINAMATH_CALUDE_unique_root_between_zero_and_e_l452_45294


namespace NUMINAMATH_CALUDE_calculate_train_speed_goods_train_speed_l452_45266

/-- Calculates the speed of a train given the speed of another train traveling in the opposite direction, the length of the train, and the time it takes to pass. -/
theorem calculate_train_speed (speed_a : ℝ) (length_b : ℝ) (pass_time : ℝ) : ℝ :=
  let speed_a_ms := speed_a * 1000 / 3600
  let relative_speed := length_b / pass_time
  let speed_b_ms := relative_speed - speed_a_ms
  let speed_b_kmh := speed_b_ms * 3600 / 1000
  speed_b_kmh

/-- Proves that given a train A traveling at 50 km/h and a goods train B of length 280 m passing train A in the opposite direction in 9 seconds, the speed of train B is approximately 62 km/h. -/
theorem goods_train_speed : 
  ∃ (ε : ℝ), ε > 0 ∧ ε < 0.5 ∧ 
  |calculate_train_speed 50 280 9 - 62| < ε :=
sorry

end NUMINAMATH_CALUDE_calculate_train_speed_goods_train_speed_l452_45266


namespace NUMINAMATH_CALUDE_increasing_quadratic_implies_a_bound_l452_45276

/-- Given a quadratic function f(x) = 2x^2 - 4(1-a)x + 1, 
    if f is increasing on [3,+∞), then a ≥ -2 -/
theorem increasing_quadratic_implies_a_bound (a : ℝ) : 
  (∀ x ≥ 3, ∀ y ≥ x, (2*y^2 - 4*(1-a)*y + 1) ≥ (2*x^2 - 4*(1-a)*x + 1)) →
  a ≥ -2 :=
by sorry

end NUMINAMATH_CALUDE_increasing_quadratic_implies_a_bound_l452_45276


namespace NUMINAMATH_CALUDE_average_marks_math_chem_l452_45265

theorem average_marks_math_chem (math physics chem : ℕ) : 
  math + physics = 60 → 
  chem = physics + 10 → 
  (math + chem) / 2 = 35 := by
sorry

end NUMINAMATH_CALUDE_average_marks_math_chem_l452_45265


namespace NUMINAMATH_CALUDE_printer_equation_l452_45219

theorem printer_equation (y : ℝ) : y > 0 →
  (300 : ℝ) / 6 + 300 / y = 300 / 3 ↔ 1 / 6 + 1 / y = 1 / 3 := by
  sorry

end NUMINAMATH_CALUDE_printer_equation_l452_45219


namespace NUMINAMATH_CALUDE_lawn_length_is_four_l452_45293

-- Define the lawn's properties
def lawn_area : ℝ := 20
def lawn_width : ℝ := 5

-- Theorem statement
theorem lawn_length_is_four :
  ∃ (length : ℝ), length * lawn_width = lawn_area ∧ length = 4 :=
by
  sorry

end NUMINAMATH_CALUDE_lawn_length_is_four_l452_45293


namespace NUMINAMATH_CALUDE_debby_total_messages_l452_45243

/-- The total number of text messages Debby received -/
def total_messages (before_noon after_noon : ℕ) : ℕ := before_noon + after_noon

/-- Proof that Debby received 39 text messages in total -/
theorem debby_total_messages :
  total_messages 21 18 = 39 := by
  sorry

end NUMINAMATH_CALUDE_debby_total_messages_l452_45243


namespace NUMINAMATH_CALUDE_number_pair_problem_l452_45234

theorem number_pair_problem (a b : ℕ) : 
  a + b = 62 → 
  (a = b + 12 ∨ b = a + 12) → 
  (a = 25 ∨ b = 25) → 
  (a = 37 ∨ b = 37) :=
by sorry

end NUMINAMATH_CALUDE_number_pair_problem_l452_45234


namespace NUMINAMATH_CALUDE_inverse_of_three_mod_191_l452_45267

theorem inverse_of_three_mod_191 : ∃ x : ℕ, x < 191 ∧ (3 * x) % 191 = 1 ∧ x = 64 := by
  sorry

end NUMINAMATH_CALUDE_inverse_of_three_mod_191_l452_45267
