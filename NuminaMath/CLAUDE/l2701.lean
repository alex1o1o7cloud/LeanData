import Mathlib

namespace NUMINAMATH_CALUDE_congruence_solution_l2701_270149

theorem congruence_solution :
  ∃ n : ℕ, 0 ≤ n ∧ n < 53 ∧ (14 * n) % 53 = 9 % 53 ∧ n = 36 := by
  sorry

end NUMINAMATH_CALUDE_congruence_solution_l2701_270149


namespace NUMINAMATH_CALUDE_equation_solution_l2701_270146

theorem equation_solution :
  ∃! x : ℚ, x + 5/6 = 11/18 - 2/9 ∧ x = -4/9 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l2701_270146


namespace NUMINAMATH_CALUDE_total_fish_l2701_270116

theorem total_fish (lilly_fish rosy_fish : ℕ) 
  (h1 : lilly_fish = 10) 
  (h2 : rosy_fish = 14) : 
  lilly_fish + rosy_fish = 24 := by
sorry

end NUMINAMATH_CALUDE_total_fish_l2701_270116


namespace NUMINAMATH_CALUDE_team_discount_saving_l2701_270119

/-- Represents the prices for a brand's uniform items -/
structure BrandPrices where
  shirt : ℝ
  pants : ℝ
  socks : ℝ

/-- Represents the prices for customization -/
structure CustomizationPrices where
  name : ℝ
  number : ℝ

def teamSize : ℕ := 12

def brandA : BrandPrices := ⟨7.5, 15, 4.5⟩
def brandB : BrandPrices := ⟨10, 18, 6⟩

def discountedBrandA : BrandPrices := ⟨6.75, 13.5, 3.75⟩
def discountedBrandB : BrandPrices := ⟨9, 16.5, 5.5⟩

def customization : CustomizationPrices := ⟨5, 3⟩

def playersWithFullCustomization : ℕ := 11

theorem team_discount_saving :
  let regularCost := 
    teamSize * (brandA.shirt + customization.name + customization.number) +
    teamSize * brandB.pants +
    teamSize * brandA.socks
  let discountedCost := 
    playersWithFullCustomization * (discountedBrandA.shirt + customization.name + customization.number) +
    (discountedBrandA.shirt + customization.name) +
    teamSize * discountedBrandB.pants +
    teamSize * brandA.socks
  regularCost - discountedCost = 31 := by sorry

end NUMINAMATH_CALUDE_team_discount_saving_l2701_270119


namespace NUMINAMATH_CALUDE_cube_distance_to_plane_l2701_270159

theorem cube_distance_to_plane (cube_side : ℝ) (h1 h2 h3 : ℝ) :
  cube_side = 10 →
  h1 = 10 ∧ h2 = 11 ∧ h3 = 12 →
  ∃ (d : ℝ), d = (33 - Real.sqrt 294) / 3 ∧
    d = min h1 (min h2 h3) - (cube_side - Real.sqrt (cube_side^2 + (h2 - h1)^2 + (h3 - h1)^2)) :=
by sorry

end NUMINAMATH_CALUDE_cube_distance_to_plane_l2701_270159


namespace NUMINAMATH_CALUDE_tan_double_angle_special_case_l2701_270124

theorem tan_double_angle_special_case (x : ℝ) 
  (h : Real.sin x - 3 * Real.cos x = Real.sqrt 5) : 
  Real.tan (2 * x) = 4 / 3 := by
  sorry

end NUMINAMATH_CALUDE_tan_double_angle_special_case_l2701_270124


namespace NUMINAMATH_CALUDE_min_value_and_inequality_range_l2701_270179

def f (x : ℝ) : ℝ := |2*x - 1| + |x + 2|

theorem min_value_and_inequality_range :
  (∃ (x_min : ℝ), ∀ (x : ℝ), f x ≥ f x_min ∧ f x_min = 5/2) ∧
  (∀ (a b x : ℝ), a ≠ 0 → |2*b - a| + |b + 2*a| ≥ |a| * (|x + 1| + |x - 1|) → -5/4 ≤ x ∧ x ≤ 5/4) :=
sorry

end NUMINAMATH_CALUDE_min_value_and_inequality_range_l2701_270179


namespace NUMINAMATH_CALUDE_age_difference_is_32_l2701_270125

/-- The age difference between Mrs Bai and her daughter Jenni -/
def age_difference : ℕ :=
  let jenni_age : ℕ := 19
  let sum_of_ages : ℕ := 70
  sum_of_ages - 2 * jenni_age

/-- Theorem stating that the age difference between Mrs Bai and Jenni is 32 years -/
theorem age_difference_is_32 : age_difference = 32 := by
  sorry

end NUMINAMATH_CALUDE_age_difference_is_32_l2701_270125


namespace NUMINAMATH_CALUDE_crayons_per_box_l2701_270131

theorem crayons_per_box 
  (total_boxes : ℕ) 
  (total_crayons : ℕ) 
  (h1 : total_boxes = 7)
  (h2 : total_crayons = 35)
  (h3 : total_crayons = total_boxes * (total_crayons / total_boxes)) :
  total_crayons / total_boxes = 5 := by
sorry

end NUMINAMATH_CALUDE_crayons_per_box_l2701_270131


namespace NUMINAMATH_CALUDE_jill_phone_time_l2701_270181

def geometric_sum (a : ℕ) (r : ℕ) (n : ℕ) : ℕ :=
  a * (r^n - 1) / (r - 1)

theorem jill_phone_time : geometric_sum 5 2 5 = 155 := by
  sorry

end NUMINAMATH_CALUDE_jill_phone_time_l2701_270181


namespace NUMINAMATH_CALUDE_dagger_example_l2701_270196

def dagger (m n p q : ℚ) : ℚ := (m + n) * (p + q) * (q / n)

theorem dagger_example : dagger (5/9) (7/4) = 616/9 := by
  sorry

end NUMINAMATH_CALUDE_dagger_example_l2701_270196


namespace NUMINAMATH_CALUDE_quadratic_equation_distinct_roots_l2701_270144

theorem quadratic_equation_distinct_roots (k : ℝ) :
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧
    (k - 1) * x₁^2 - 2 * x₁ + 3 = 0 ∧
    (k - 1) * x₂^2 - 2 * x₂ + 3 = 0) ↔
  (k < 4/3 ∧ k ≠ 1) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equation_distinct_roots_l2701_270144


namespace NUMINAMATH_CALUDE_two_digit_reverse_difference_cube_l2701_270127

/-- A two-digit number -/
def TwoDigitNumber (n : ℕ) : Prop := 10 ≤ n ∧ n < 100

/-- The reversed digits of a two-digit number -/
def reverseDigits (n : ℕ) : ℕ :=
  (n % 10) * 10 + (n / 10)

/-- Predicate for a number being a positive perfect cube -/
def isPositivePerfectCube (n : ℕ) : Prop :=
  ∃ k : ℕ, k > 0 ∧ n = k^3

/-- The main theorem -/
theorem two_digit_reverse_difference_cube :
  ∃! (s : Finset ℕ),
    (∀ n ∈ s, TwoDigitNumber n ∧ isPositivePerfectCube (n - reverseDigits n)) ∧
    Finset.card s = 2 := by
  sorry

end NUMINAMATH_CALUDE_two_digit_reverse_difference_cube_l2701_270127


namespace NUMINAMATH_CALUDE_inequality_solution_set_l2701_270156

theorem inequality_solution_set (x : ℝ) : 
  (x^2 - 2*x - 3) * (x^2 - 4*x + 4) < 0 ↔ -1 < x ∧ x < 3 ∧ x ≠ 2 :=
by sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l2701_270156


namespace NUMINAMATH_CALUDE_negative_east_equals_positive_west_l2701_270195

-- Define the direction type
inductive Direction
| East
| West

-- Define a function to represent movement
def move (distance : Int) (direction : Direction) : Int :=
  match direction with
  | Direction.East => distance
  | Direction.West => -distance

-- Theorem statement
theorem negative_east_equals_positive_west :
  move (-8) Direction.East = move 8 Direction.West :=
by sorry

end NUMINAMATH_CALUDE_negative_east_equals_positive_west_l2701_270195


namespace NUMINAMATH_CALUDE_sum_of_extreme_prime_factors_of_1260_l2701_270169

def is_prime_factor (p n : ℕ) : Prop :=
  Nat.Prime p ∧ n % p = 0

theorem sum_of_extreme_prime_factors_of_1260 :
  ∃ (min max : ℕ),
    is_prime_factor min 1260 ∧
    is_prime_factor max 1260 ∧
    (∀ p, is_prime_factor p 1260 → min ≤ p) ∧
    (∀ p, is_prime_factor p 1260 → p ≤ max) ∧
    min + max = 9 :=
  sorry

end NUMINAMATH_CALUDE_sum_of_extreme_prime_factors_of_1260_l2701_270169


namespace NUMINAMATH_CALUDE_valid_outfit_choices_eq_239_l2701_270113

/-- Represents the number of valid outfit choices given the specified conditions -/
def valid_outfit_choices : ℕ := by
  -- Define the number of shirts, pants, and hats
  let num_shirts : ℕ := 6
  let num_pants : ℕ := 7
  let num_hats : ℕ := 6
  
  -- Define the number of colors
  let num_colors : ℕ := 6
  
  -- Calculate total number of outfits without restrictions
  let total_outfits : ℕ := num_shirts * num_pants * num_hats
  
  -- Calculate number of outfits with all items the same color
  let all_same_color : ℕ := num_colors
  
  -- Calculate number of outfits with shirt and pants the same color
  let shirt_pants_same : ℕ := num_colors + 1
  
  -- Calculate the number of valid outfits
  exact total_outfits - all_same_color - shirt_pants_same

/-- Theorem stating that the number of valid outfit choices is 239 -/
theorem valid_outfit_choices_eq_239 : valid_outfit_choices = 239 := by
  sorry

end NUMINAMATH_CALUDE_valid_outfit_choices_eq_239_l2701_270113


namespace NUMINAMATH_CALUDE_total_pencils_l2701_270197

/-- Given that each child has 4 pencils and there are 8 children, 
    prove that the total number of pencils is 32. -/
theorem total_pencils (pencils_per_child : ℕ) (num_children : ℕ) 
  (h1 : pencils_per_child = 4) (h2 : num_children = 8) : 
  pencils_per_child * num_children = 32 := by
  sorry

end NUMINAMATH_CALUDE_total_pencils_l2701_270197


namespace NUMINAMATH_CALUDE_angle_trigonometry_l2701_270138

open Real

theorem angle_trigonometry (x : ℝ) (h1 : π/2 < x) (h2 : x < π) 
  (h3 : cos x = tan x) (h4 : sin x ≠ cos x) : 
  sin x = (-1 + Real.sqrt 5) / 2 := by
sorry

end NUMINAMATH_CALUDE_angle_trigonometry_l2701_270138


namespace NUMINAMATH_CALUDE_expression_evaluation_l2701_270134

theorem expression_evaluation (c : ℕ) (h : c = 4) :
  (2 * c^c - (c + 1) * (c - 1)^c)^c = 131044201 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l2701_270134


namespace NUMINAMATH_CALUDE_forward_journey_time_l2701_270184

/-- Represents the journey of a car -/
structure Journey where
  distance : ℝ
  forwardTime : ℝ
  returnTime : ℝ
  speedIncrease : ℝ

/-- Theorem stating the conditions and the result to be proved -/
theorem forward_journey_time (j : Journey)
  (h1 : j.distance = 210)
  (h2 : j.returnTime = 5)
  (h3 : j.speedIncrease = 12)
  (h4 : j.distance = j.distance / j.forwardTime * j.returnTime + j.speedIncrease * j.returnTime) :
  j.forwardTime = 7 := by
  sorry

end NUMINAMATH_CALUDE_forward_journey_time_l2701_270184


namespace NUMINAMATH_CALUDE_smallest_positive_difference_l2701_270126

/-- Vovochka's sum method for two three-digit numbers -/
def vovochkaSum (a b c d e f : ℕ) : ℕ :=
  1000 * (a + d) + 100 * (b + e) + (c + f)

/-- Correct sum for two three-digit numbers -/
def correctSum (a b c d e f : ℕ) : ℕ :=
  100 * (a + d) + 10 * (b + e) + (c + f)

/-- The difference between Vovochka's sum and the correct sum -/
def sumDifference (a b c d e f : ℕ) : ℤ :=
  (vovochkaSum a b c d e f : ℤ) - (correctSum a b c d e f : ℤ)

theorem smallest_positive_difference :
  ∀ a b c d e f : ℕ,
    a < 10 → b < 10 → c < 10 → d < 10 → e < 10 → f < 10 →
    (∃ a' b' c' d' e' f' : ℕ,
      a' < 10 ∧ b' < 10 ∧ c' < 10 ∧ d' < 10 ∧ e' < 10 ∧ f' < 10 ∧
      sumDifference a' b' c' d' e' f' > 0 ∧
      sumDifference a' b' c' d' e' f' ≤ sumDifference a b c d e f) →
    (∃ a' b' c' d' e' f' : ℕ,
      a' < 10 ∧ b' < 10 ∧ c' < 10 ∧ d' < 10 ∧ e' < 10 ∧ f' < 10 ∧
      sumDifference a' b' c' d' e' f' = 1800 ∧
      sumDifference a' b' c' d' e' f' > 0 ∧
      ∀ x y z u v w : ℕ,
        x < 10 → y < 10 → z < 10 → u < 10 → v < 10 → w < 10 →
        sumDifference x y z u v w > 0 →
        sumDifference a' b' c' d' e' f' ≤ sumDifference x y z u v w) :=
by sorry

end NUMINAMATH_CALUDE_smallest_positive_difference_l2701_270126


namespace NUMINAMATH_CALUDE_cos_135_degrees_l2701_270194

theorem cos_135_degrees : Real.cos (135 * π / 180) = -Real.sqrt 2 / 2 := by
  sorry

end NUMINAMATH_CALUDE_cos_135_degrees_l2701_270194


namespace NUMINAMATH_CALUDE_max_ab_value_l2701_270191

/-- A function f with a parameter a and b -/
def f (a b x : ℝ) : ℝ := 4 * x^3 - a * x^2 - 2 * b * x + 2

/-- The derivative of f with respect to x -/
def f' (a b x : ℝ) : ℝ := 12 * x^2 - 2 * a * x - 2 * b

theorem max_ab_value (a b : ℝ) (ha : a > 0) (hb : b > 0) 
  (h_extremum : f' a b 1 = 0) : 
  ab ≤ 9 ∧ ∃ a b, a > 0 ∧ b > 0 ∧ f' a b 1 = 0 ∧ a * b = 9 := by
  sorry

end NUMINAMATH_CALUDE_max_ab_value_l2701_270191


namespace NUMINAMATH_CALUDE_pi_is_max_l2701_270118

theorem pi_is_max : ∀ (π : ℝ), π > 0 → (1 / 2023 : ℝ) > 0 → -2 * π < 0 →
  max (max (max 0 π) (1 / 2023)) (-2 * π) = π :=
by sorry

end NUMINAMATH_CALUDE_pi_is_max_l2701_270118


namespace NUMINAMATH_CALUDE_greatest_multiple_of_four_cubed_less_than_5000_l2701_270175

theorem greatest_multiple_of_four_cubed_less_than_5000 :
  ∀ x : ℕ, x > 0 → x % 4 = 0 → x^3 < 5000 → x ≤ 16 :=
by
  sorry

end NUMINAMATH_CALUDE_greatest_multiple_of_four_cubed_less_than_5000_l2701_270175


namespace NUMINAMATH_CALUDE_rectangle_area_constraint_l2701_270122

theorem rectangle_area_constraint (m : ℝ) : 
  (3 * m + 8) * (m - 3) = 70 → m = (1 + Real.sqrt 1129) / 6 :=
by
  sorry

end NUMINAMATH_CALUDE_rectangle_area_constraint_l2701_270122


namespace NUMINAMATH_CALUDE_x_eq_one_sufficient_not_necessary_for_x_squared_eq_one_l2701_270101

theorem x_eq_one_sufficient_not_necessary_for_x_squared_eq_one :
  (∀ x : ℝ, x = 1 → x^2 = 1) ∧
  ¬(∀ x : ℝ, x^2 = 1 → x = 1) :=
by sorry

end NUMINAMATH_CALUDE_x_eq_one_sufficient_not_necessary_for_x_squared_eq_one_l2701_270101


namespace NUMINAMATH_CALUDE_jerrys_age_l2701_270186

/-- Given that Mickey's age is 10 years more than 200% of Jerry's age,
    and Mickey is 22 years old, Jerry's age is 6 years. -/
theorem jerrys_age (mickey jerry : ℕ) 
  (h1 : mickey = 2 * jerry + 10)  -- Mickey's age relation to Jerry's
  (h2 : mickey = 22)              -- Mickey's age
  : jerry = 6 := by
  sorry

end NUMINAMATH_CALUDE_jerrys_age_l2701_270186


namespace NUMINAMATH_CALUDE_cube_face_perimeter_l2701_270139

theorem cube_face_perimeter (volume : ℝ) (perimeter : ℝ) : 
  volume = 125 → perimeter = 20 := by
  sorry

end NUMINAMATH_CALUDE_cube_face_perimeter_l2701_270139


namespace NUMINAMATH_CALUDE_even_expressions_l2701_270151

theorem even_expressions (m n : ℕ) (hm : m % 2 = 1) (hn : n % 2 = 1) :
  ∃ (k₁ k₂ k₃ : ℕ),
    (m - n)^2 = 2 * k₁ ∧
    (m - n - 4)^2 = 2 * k₂ ∧
    2 * m * n + 4 = 2 * k₃ := by
  sorry

end NUMINAMATH_CALUDE_even_expressions_l2701_270151


namespace NUMINAMATH_CALUDE_difference_divisible_by_nine_l2701_270165

def reverse (n : ℕ) : ℕ := sorry

theorem difference_divisible_by_nine (n : ℕ) : 
  ∃ k : ℤ, n - reverse n = 9 * k := by sorry

end NUMINAMATH_CALUDE_difference_divisible_by_nine_l2701_270165


namespace NUMINAMATH_CALUDE_prob_at_least_four_girls_l2701_270163

-- Define the number of children
def num_children : ℕ := 6

-- Define the probability of a child being a girl
def prob_girl : ℚ := 1/2

-- Define the function to calculate the probability of at least k girls out of n children
def prob_at_least_k_girls (n k : ℕ) : ℚ :=
  sorry

theorem prob_at_least_four_girls :
  prob_at_least_k_girls num_children 4 = 11/32 :=
sorry

end NUMINAMATH_CALUDE_prob_at_least_four_girls_l2701_270163


namespace NUMINAMATH_CALUDE_tickets_to_be_sold_l2701_270160

def total_tickets : ℕ := 100
def jude_sales : ℕ := 16

def andrea_sales (jude_sales : ℕ) : ℕ := 2 * jude_sales

def sandra_sales (jude_sales : ℕ) : ℕ := jude_sales / 2 + 4

def total_sold (jude_sales : ℕ) : ℕ :=
  jude_sales + andrea_sales jude_sales + sandra_sales jude_sales

theorem tickets_to_be_sold :
  total_tickets - total_sold jude_sales = 40 :=
by sorry

end NUMINAMATH_CALUDE_tickets_to_be_sold_l2701_270160


namespace NUMINAMATH_CALUDE_min_translation_value_l2701_270178

theorem min_translation_value (f g : ℝ → ℝ) (m : ℝ) :
  (∀ x, f x = Real.sin (2 * x) + Real.cos (2 * x)) →
  (∀ x, g x = f (x - m)) →
  (m > 0) →
  (∀ x, g (-π/3) ≤ g x) →
  ∃ k : ℤ, m = k * π + π/24 ∧ 
  (∀ m' : ℝ, m' > 0 → (∀ x, g (-π/3) ≤ g x) → m' ≥ π/24) :=
sorry

end NUMINAMATH_CALUDE_min_translation_value_l2701_270178


namespace NUMINAMATH_CALUDE_average_visitors_is_276_l2701_270150

/-- Calculates the average number of visitors per day in a 30-day month starting on a Sunday -/
def averageVisitors (sundayVisitors : ℕ) (otherDayVisitors : ℕ) : ℚ :=
  let numSundays := 4
  let numOtherDays := 26
  let totalVisitors := numSundays * sundayVisitors + numOtherDays * otherDayVisitors
  totalVisitors / 30

/-- Theorem stating that the average number of visitors is 276 given the specified conditions -/
theorem average_visitors_is_276 :
  averageVisitors 510 240 = 276 := by
  sorry

#eval averageVisitors 510 240

end NUMINAMATH_CALUDE_average_visitors_is_276_l2701_270150


namespace NUMINAMATH_CALUDE_probability_two_blue_balls_l2701_270135

/-- The probability of drawing two blue balls consecutively from an urn -/
theorem probability_two_blue_balls (total_balls : Nat) (blue_balls : Nat) (red_balls : Nat) :
  total_balls = blue_balls + red_balls →
  blue_balls = 6 →
  red_balls = 4 →
  (blue_balls : ℚ) / total_balls * (blue_balls - 1) / (total_balls - 1) = 1 / 3 := by
  sorry

end NUMINAMATH_CALUDE_probability_two_blue_balls_l2701_270135


namespace NUMINAMATH_CALUDE_largest_three_digit_multiple_of_8_with_digit_sum_24_l2701_270109

/-- A function that returns the sum of digits of a natural number -/
def digit_sum (n : ℕ) : ℕ :=
  sorry

/-- A function that checks if a number is a three-digit number -/
def is_three_digit (n : ℕ) : Prop :=
  100 ≤ n ∧ n ≤ 999

theorem largest_three_digit_multiple_of_8_with_digit_sum_24 :
  ∀ n : ℕ, is_three_digit n → n.mod 8 = 0 → digit_sum n = 24 → n ≤ 888 :=
sorry

end NUMINAMATH_CALUDE_largest_three_digit_multiple_of_8_with_digit_sum_24_l2701_270109


namespace NUMINAMATH_CALUDE_no_periodic_difference_with_periods_3_and_pi_l2701_270105

-- Define a periodic function
def IsPeriodic (f : ℝ → ℝ) :=
  (∃ x y, f x ≠ f y) ∧
  (∃ p : ℝ, p > 0 ∧ ∀ x, f (x + p) = f x)

-- Define the period of a function
def IsPeriodOf (p : ℝ) (f : ℝ → ℝ) :=
  p > 0 ∧ ∀ x, f (x + p) = f x

-- Theorem statement
theorem no_periodic_difference_with_periods_3_and_pi :
  ¬ ∃ (g h : ℝ → ℝ),
    IsPeriodic g ∧ IsPeriodic h ∧
    IsPeriodOf 3 g ∧ IsPeriodOf π h ∧
    IsPeriodic (g - h) :=
sorry

end NUMINAMATH_CALUDE_no_periodic_difference_with_periods_3_and_pi_l2701_270105


namespace NUMINAMATH_CALUDE_millet_exceeds_half_on_day_three_l2701_270198

/-- Represents the proportion of millet in the feeder on a given day -/
def milletProportion (day : ℕ) : ℝ :=
  0.4 * (1 - (0.5 ^ day))

/-- The day when millet first exceeds half of the seeds -/
def milletExceedsHalfDay : ℕ :=
  3

theorem millet_exceeds_half_on_day_three :
  milletProportion milletExceedsHalfDay > 0.5 ∧
  ∀ d : ℕ, d < milletExceedsHalfDay → milletProportion d ≤ 0.5 :=
by sorry

end NUMINAMATH_CALUDE_millet_exceeds_half_on_day_three_l2701_270198


namespace NUMINAMATH_CALUDE_landscaping_equation_l2701_270137

-- Define the variables and constants
def total_area : ℝ := 180
def original_workers : ℕ := 6
def additional_workers : ℕ := 2
def time_saved : ℝ := 3

-- Define the theorem
theorem landscaping_equation (x : ℝ) :
  (total_area / (original_workers * x)) - (total_area / ((original_workers + additional_workers) * x)) = time_saved :=
by sorry

end NUMINAMATH_CALUDE_landscaping_equation_l2701_270137


namespace NUMINAMATH_CALUDE_inequality_proof_l2701_270172

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  Real.sqrt (a^2 + a*b + b^2) + Real.sqrt (a^2 + a*c + c^2) ≥ 
  4 * Real.sqrt ((a*b / (a+b))^2 + (a*b / (a+b)) * (a*c / (a+c)) + (a*c / (a+c))^2) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l2701_270172


namespace NUMINAMATH_CALUDE_min_value_quadratic_l2701_270129

theorem min_value_quadratic :
  (∀ x : ℝ, x^2 + 6*x ≥ -9) ∧ (∃ x : ℝ, x^2 + 6*x = -9) := by
  sorry

end NUMINAMATH_CALUDE_min_value_quadratic_l2701_270129


namespace NUMINAMATH_CALUDE_students_behind_minyoung_l2701_270143

/-- Given a line of students with Minyoung, prove the number behind her. -/
theorem students_behind_minyoung 
  (total : ℕ) 
  (in_front : ℕ) 
  (h1 : total = 35) 
  (h2 : in_front = 27) : 
  total - (in_front + 1) = 7 := by
  sorry

end NUMINAMATH_CALUDE_students_behind_minyoung_l2701_270143


namespace NUMINAMATH_CALUDE_phone_rep_work_hours_l2701_270168

theorem phone_rep_work_hours 
  (num_reps : ℕ) 
  (num_days : ℕ) 
  (hourly_rate : ℚ) 
  (total_pay : ℚ) 
  (h1 : num_reps = 50)
  (h2 : num_days = 5)
  (h3 : hourly_rate = 14)
  (h4 : total_pay = 28000) :
  (total_pay / hourly_rate) / (num_reps * num_days) = 8 := by
sorry

end NUMINAMATH_CALUDE_phone_rep_work_hours_l2701_270168


namespace NUMINAMATH_CALUDE_square_root_two_plus_one_squared_minus_two_times_plus_two_equals_three_l2701_270155

theorem square_root_two_plus_one_squared_minus_two_times_plus_two_equals_three :
  let x : ℝ := Real.sqrt 2 + 1
  (x^2 - 2*x + 2 : ℝ) = 3 := by
sorry

end NUMINAMATH_CALUDE_square_root_two_plus_one_squared_minus_two_times_plus_two_equals_three_l2701_270155


namespace NUMINAMATH_CALUDE_optimal_rectangle_dimensions_l2701_270133

/-- The total length of the fence in meters -/
def fence_length : ℝ := 60

/-- The area of the rectangle as a function of its width -/
def area (x : ℝ) : ℝ := x * (fence_length - 2 * x)

/-- The width that maximizes the area -/
def optimal_width : ℝ := 15

/-- The length that maximizes the area -/
def optimal_length : ℝ := 30

theorem optimal_rectangle_dimensions :
  (∀ x : ℝ, 0 < x → x < fence_length / 2 → area x ≤ area optimal_width) ∧
  optimal_length = fence_length - 2 * optimal_width :=
sorry

end NUMINAMATH_CALUDE_optimal_rectangle_dimensions_l2701_270133


namespace NUMINAMATH_CALUDE_distance_to_y_axis_l2701_270117

/-- Given a point P(x, -9) where the distance from the x-axis to P is half the distance
    from the y-axis to P, prove that the distance from P to the y-axis is 18 units. -/
theorem distance_to_y_axis (x : ℝ) :
  let P : ℝ × ℝ := (x, -9)
  (abs (P.2) = (1/2 : ℝ) * abs P.1) →
  abs P.1 = 18 := by
sorry

end NUMINAMATH_CALUDE_distance_to_y_axis_l2701_270117


namespace NUMINAMATH_CALUDE_forest_gathering_handshakes_count_l2701_270100

/-- The number of handshakes at the Forest Gathering -/
def forest_gathering_handshakes : ℕ :=
  let total_gremlins : ℕ := 30
  let total_pixies : ℕ := 12
  let unfriendly_gremlins : ℕ := total_gremlins / 2
  let friendly_gremlins : ℕ := total_gremlins - unfriendly_gremlins
  
  -- Handshakes among friendly gremlins
  let friendly_gremlin_handshakes : ℕ := friendly_gremlins * (friendly_gremlins - 1) / 2
  
  -- Handshakes between friendly and unfriendly gremlins
  let mixed_gremlin_handshakes : ℕ := friendly_gremlins * unfriendly_gremlins
  
  -- Handshakes between all gremlins and pixies
  let gremlin_pixie_handshakes : ℕ := total_gremlins * total_pixies
  
  -- Total handshakes
  friendly_gremlin_handshakes + mixed_gremlin_handshakes + gremlin_pixie_handshakes

/-- Theorem stating that the number of handshakes at the Forest Gathering is 690 -/
theorem forest_gathering_handshakes_count : forest_gathering_handshakes = 690 := by
  sorry

end NUMINAMATH_CALUDE_forest_gathering_handshakes_count_l2701_270100


namespace NUMINAMATH_CALUDE_photo_arrangements_l2701_270199

/-- The number of male students -/
def num_male : ℕ := 5

/-- The number of female students -/
def num_female : ℕ := 2

/-- The total number of students -/
def total_students : ℕ := num_male + num_female

/-- Calculates the number of arrangements with female student A at one end -/
def arrangements_a_at_end : ℕ := sorry

/-- Calculates the number of arrangements with both female students not at the ends -/
def arrangements_females_not_at_ends : ℕ := sorry

/-- Calculates the number of arrangements with the two female students not adjacent -/
def arrangements_females_not_adjacent : ℕ := sorry

/-- Calculates the number of arrangements with female student A on the right side of female student B -/
def arrangements_a_right_of_b : ℕ := sorry

theorem photo_arrangements :
  arrangements_a_at_end = 1440 ∧
  arrangements_females_not_at_ends = 2400 ∧
  arrangements_females_not_adjacent = 3600 ∧
  arrangements_a_right_of_b = 2520 := by sorry

end NUMINAMATH_CALUDE_photo_arrangements_l2701_270199


namespace NUMINAMATH_CALUDE_simplify_expression_l2701_270153

theorem simplify_expression (x : ℝ) : 
  3*x + 5*x^2 + 12 - (6 - 3*x - 10*x^2) = 15*x^2 + 6*x + 6 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l2701_270153


namespace NUMINAMATH_CALUDE_peregrine_falcon_problem_l2701_270164

/-- The percentage of pigeons eaten by peregrines -/
def percentage_eaten (initial_pigeons : ℕ) (chicks_per_pigeon : ℕ) (remaining_pigeons : ℕ) : ℚ :=
  let total_pigeons := initial_pigeons + initial_pigeons * chicks_per_pigeon
  let eaten_pigeons := total_pigeons - remaining_pigeons
  (eaten_pigeons : ℚ) / (total_pigeons : ℚ) * 100

theorem peregrine_falcon_problem :
  percentage_eaten 40 6 196 = 30 := by
  sorry

end NUMINAMATH_CALUDE_peregrine_falcon_problem_l2701_270164


namespace NUMINAMATH_CALUDE_age_difference_l2701_270170

theorem age_difference (A B C : ℕ) : 
  (∃ k : ℕ, A = B + k) →  -- A is some years older than B
  B = 2 * C →             -- B is twice as old as C
  A + B + C = 27 →        -- Total of ages is 27
  B = 10 →                -- B is 10 years old
  A = B + 2 :=            -- A is 2 years older than B
by sorry

end NUMINAMATH_CALUDE_age_difference_l2701_270170


namespace NUMINAMATH_CALUDE_male_attendee_fraction_l2701_270142

theorem male_attendee_fraction :
  let male_fraction : ℝ → ℝ := λ x => x
  let female_fraction : ℝ → ℝ := λ x => 1 - x
  let male_on_time : ℝ → ℝ := λ x => (7/8) * x
  let female_on_time : ℝ → ℝ := λ x => (9/10) * (1 - x)
  let total_on_time : ℝ := 0.885
  ∀ x : ℝ, male_on_time x + female_on_time x = total_on_time → x = 0.6 :=
by
  sorry

end NUMINAMATH_CALUDE_male_attendee_fraction_l2701_270142


namespace NUMINAMATH_CALUDE_f_expression_sum_f_expression_l2701_270132

/-- A linear function f satisfying specific conditions -/
def f (x : ℝ) : ℝ := sorry

/-- The condition that f(8) = 15 -/
axiom f_8 : f 8 = 15

/-- The condition that f(2), f(5), f(4) form a geometric sequence -/
axiom f_geometric : ∃ (r : ℝ), f 5 = r * f 2 ∧ f 4 = r * f 5

/-- Theorem stating that f(x) = 4x - 17 -/
theorem f_expression : ∀ x, f x = 4 * x - 17 := by sorry

/-- Function to calculate the sum of f(2) + f(4) + ... + f(2n) -/
def sum_f (n : ℕ) : ℝ := sorry

/-- Theorem stating the sum of f(2) + f(4) + ... + f(2n) = 4n^2 - 13n -/
theorem sum_f_expression : ∀ n, sum_f n = 4 * n^2 - 13 * n := by sorry

end NUMINAMATH_CALUDE_f_expression_sum_f_expression_l2701_270132


namespace NUMINAMATH_CALUDE_tan_theta_two_implies_expression_l2701_270183

theorem tan_theta_two_implies_expression (θ : Real) (h : Real.tan θ = 2) :
  (Real.sin θ + Real.cos θ) * Real.cos (2 * θ) / Real.sin θ = -9/10 := by
  sorry

end NUMINAMATH_CALUDE_tan_theta_two_implies_expression_l2701_270183


namespace NUMINAMATH_CALUDE_negation_equivalence_l2701_270140

theorem negation_equivalence (x y : ℝ) : 
  ¬(x^2 + y^2 = 0 → x = 0 ∧ y = 0) ↔ (x^2 + y^2 ≠ 0 → ¬(x = 0 ∧ y = 0)) :=
sorry

end NUMINAMATH_CALUDE_negation_equivalence_l2701_270140


namespace NUMINAMATH_CALUDE_quadratic_roots_sum_of_squares_l2701_270162

theorem quadratic_roots_sum_of_squares (m : ℝ) (x₁ x₂ : ℝ) : 
  (∀ x, x^2 - m*x + (2*m - 1) = 0 ↔ x = x₁ ∨ x = x₂) →
  x₁^2 + x₂^2 = 7 →
  m = -1 := by
sorry

end NUMINAMATH_CALUDE_quadratic_roots_sum_of_squares_l2701_270162


namespace NUMINAMATH_CALUDE_solve_jewelry_problem_l2701_270111

/-- Represents the jewelry store inventory problem -/
def jewelry_problem (necklace_capacity : ℕ) (current_necklaces : ℕ) 
  (ring_capacity : ℕ) (current_rings : ℕ) (bracelet_capacity : ℕ) 
  (necklace_cost : ℕ) (ring_cost : ℕ) (bracelet_cost : ℕ) (total_cost : ℕ) : Prop :=
  ∃ (current_bracelets : ℕ),
    necklace_capacity = 12 ∧
    current_necklaces = 5 ∧
    ring_capacity = 30 ∧
    current_rings = 18 ∧
    bracelet_capacity = 15 ∧
    necklace_cost = 4 ∧
    ring_cost = 10 ∧
    bracelet_cost = 5 ∧
    total_cost = 183 ∧
    (necklace_capacity - current_necklaces) * necklace_cost + 
    (ring_capacity - current_rings) * ring_cost + 
    (bracelet_capacity - current_bracelets) * bracelet_cost = total_cost ∧
    current_bracelets = 8

theorem solve_jewelry_problem :
  jewelry_problem 12 5 30 18 15 4 10 5 183 :=
sorry

end NUMINAMATH_CALUDE_solve_jewelry_problem_l2701_270111


namespace NUMINAMATH_CALUDE_moe_has_least_money_l2701_270166

-- Define the set of people
inductive Person : Type
  | Bo | Coe | Flo | Jo | Moe | Zoe

-- Define the "has more money than" relation
def has_more_money (p1 p2 : Person) : Prop := sorry

-- Define the conditions
axiom different_amounts : ∀ (p1 p2 : Person), p1 ≠ p2 → has_more_money p1 p2 ∨ has_more_money p2 p1
axiom flo_bo_zoe : has_more_money Person.Flo Person.Bo ∧ has_more_money Person.Zoe Person.Flo
axiom zoe_coe : has_more_money Person.Zoe Person.Coe
axiom bo_coe_moe : has_more_money Person.Bo Person.Moe ∧ has_more_money Person.Coe Person.Moe
axiom jo_moe_zoe : has_more_money Person.Jo Person.Moe ∧ has_more_money Person.Zoe Person.Jo

-- Define the "has least money" property
def has_least_money (p : Person) : Prop :=
  ∀ (other : Person), other ≠ p → has_more_money other p

-- Theorem statement
theorem moe_has_least_money : has_least_money Person.Moe := by
  sorry

end NUMINAMATH_CALUDE_moe_has_least_money_l2701_270166


namespace NUMINAMATH_CALUDE_correct_calculation_l2701_270180

theorem correct_calculation (x : ℤ) : x + 19 = 50 → 16 * x = 496 := by
  sorry

end NUMINAMATH_CALUDE_correct_calculation_l2701_270180


namespace NUMINAMATH_CALUDE_geometric_sequence_product_l2701_270193

theorem geometric_sequence_product (a b c : ℝ) : 
  a > 0 ∧ b > 0 ∧ c > 0 → -- Ensuring a, b, c are positive
  (1 : ℝ) < a ∧ a < b ∧ b < c ∧ c < 256 → -- Ensuring the order of the sequence
  (b / a = a / 1) ∧ (c / b = b / a) ∧ (256 / c = c / b) → -- Geometric sequence condition
  1 * a * b * c * 256 = 2^20 := by
sorry


end NUMINAMATH_CALUDE_geometric_sequence_product_l2701_270193


namespace NUMINAMATH_CALUDE_bus_driver_overtime_rate_increase_bus_driver_overtime_rate_increase_approx_l2701_270154

/-- Calculates the percentage increase in overtime rate compared to regular rate for a bus driver --/
theorem bus_driver_overtime_rate_increase 
  (regular_rate : ℝ) 
  (regular_hours : ℝ) 
  (total_compensation : ℝ) 
  (total_hours : ℝ) : ℝ :=
  let overtime_hours := total_hours - regular_hours
  let regular_earnings := regular_rate * regular_hours
  let overtime_earnings := total_compensation - regular_earnings
  let overtime_rate := overtime_earnings / overtime_hours
  let percentage_increase := (overtime_rate - regular_rate) / regular_rate * 100
  percentage_increase

/-- The percentage increase in overtime rate is approximately 74.93% --/
theorem bus_driver_overtime_rate_increase_approx :
  ∃ ε > 0, abs (bus_driver_overtime_rate_increase 14 40 998 57.88 - 74.93) < ε :=
sorry

end NUMINAMATH_CALUDE_bus_driver_overtime_rate_increase_bus_driver_overtime_rate_increase_approx_l2701_270154


namespace NUMINAMATH_CALUDE_score_ordering_l2701_270176

structure Participant where
  score : ℕ

def Leonard : Participant := sorry
def Nina : Participant := sorry
def Oscar : Participant := sorry
def Paula : Participant := sorry

theorem score_ordering :
  (Oscar.score = Leonard.score) →
  (Nina.score < max Oscar.score Paula.score) →
  (Paula.score > Leonard.score) →
  (Oscar.score < Nina.score) ∧ (Nina.score < Paula.score) := by
  sorry

end NUMINAMATH_CALUDE_score_ordering_l2701_270176


namespace NUMINAMATH_CALUDE_no_primes_in_range_l2701_270167

theorem no_primes_in_range (n m : ℕ) (hn : n > 1) (hm : 1 ≤ m ∧ m ≤ n) :
  ∀ k, n! + m < k ∧ k < n! + n + m → ¬ Nat.Prime k := by
  sorry

end NUMINAMATH_CALUDE_no_primes_in_range_l2701_270167


namespace NUMINAMATH_CALUDE_gravitational_force_at_new_distance_l2701_270161

/-- Gravitational force calculation -/
theorem gravitational_force_at_new_distance
  (f1 : ℝ) (d1 : ℝ) (d2 : ℝ)
  (h1 : f1 = 480)
  (h2 : d1 = 5000)
  (h3 : d2 = 300000)
  (h4 : ∀ (f d : ℝ), f * d^2 = f1 * d1^2) :
  ∃ (f2 : ℝ), f2 = 1 / 75 ∧ f2 * d2^2 = f1 * d1^2 := by
  sorry

end NUMINAMATH_CALUDE_gravitational_force_at_new_distance_l2701_270161


namespace NUMINAMATH_CALUDE_store_sales_theorem_l2701_270187

/-- Represents the daily sales and profit calculations for a store. -/
structure StoreSales where
  initial_sales : ℕ
  initial_profit : ℕ
  sales_increase : ℕ
  min_profit : ℕ

/-- Calculates the new sales quantity after a price reduction. -/
def new_sales (s : StoreSales) (reduction : ℕ) : ℕ :=
  s.initial_sales + s.sales_increase * reduction

/-- Calculates the new profit per item after a price reduction. -/
def new_profit_per_item (s : StoreSales) (reduction : ℕ) : ℕ :=
  s.initial_profit - reduction

/-- Calculates the total daily profit after a price reduction. -/
def total_daily_profit (s : StoreSales) (reduction : ℕ) : ℕ :=
  (new_sales s reduction) * (new_profit_per_item s reduction)

/-- The main theorem stating the two parts of the problem. -/
theorem store_sales_theorem (s : StoreSales) 
    (h1 : s.initial_sales = 20)
    (h2 : s.initial_profit = 40)
    (h3 : s.sales_increase = 2)
    (h4 : s.min_profit = 25) : 
  (new_sales s 4 = 28) ∧ 
  (∃ (x : ℕ), x = 5 ∧ total_daily_profit s x = 1050 ∧ new_profit_per_item s x ≥ s.min_profit) := by
  sorry


end NUMINAMATH_CALUDE_store_sales_theorem_l2701_270187


namespace NUMINAMATH_CALUDE_room_length_proof_l2701_270112

/-- Given a room with dimensions L * 15 * 12 feet, prove that L = 25 feet
    based on the whitewashing cost and room features. -/
theorem room_length_proof (L : ℝ) : 
  L * 15 * 12 > 0 →  -- room has positive volume
  (3 : ℝ) * (2 * (L * 12 + 15 * 12) - (6 * 3 + 3 * 4 * 3)) = 2718 →
  L = 25 := by sorry

end NUMINAMATH_CALUDE_room_length_proof_l2701_270112


namespace NUMINAMATH_CALUDE_sasha_plucked_leaves_l2701_270158

/-- The number of leaves Sasha plucked -/
def leaves_plucked (apple_trees poplar_trees masha_last_apple sasha_start_apple unphotographed : ℕ) : ℕ :=
  (apple_trees + poplar_trees) - (sasha_start_apple - 1) - unphotographed

/-- Theorem stating the number of leaves Sasha plucked -/
theorem sasha_plucked_leaves :
  leaves_plucked 17 18 10 8 13 = 22 := by
  sorry

#eval leaves_plucked 17 18 10 8 13

end NUMINAMATH_CALUDE_sasha_plucked_leaves_l2701_270158


namespace NUMINAMATH_CALUDE_mowing_time_c_l2701_270128

-- Define the work rates
def work_rate (days : ℚ) : ℚ := 1 / days

-- Define the given conditions
def condition1 (a b : ℚ) : Prop := a + b = work_rate 28
def condition2 (a b c : ℚ) : Prop := a + b + c = work_rate 21

-- Theorem statement
theorem mowing_time_c (a b c : ℚ) 
  (h1 : condition1 a b) (h2 : condition2 a b c) : c = work_rate 84 := by
  sorry

end NUMINAMATH_CALUDE_mowing_time_c_l2701_270128


namespace NUMINAMATH_CALUDE_triangle_problem_l2701_270115

theorem triangle_problem (a b c : ℝ) (A B C : ℝ) :
  -- Triangle ABC exists
  0 < a ∧ 0 < b ∧ 0 < c ∧ 
  0 < A ∧ A < π ∧ 0 < B ∧ B < π ∧ 0 < C ∧ C < π ∧
  A + B + C = π →
  -- Given conditions
  a^2 + c^2 - b^2 = a * c →
  a = 8 * Real.sqrt 3 →
  Real.cos A = 3 / 5 →
  -- Conclusions
  B = π / 3 ∧ b = 15 := by
sorry

end NUMINAMATH_CALUDE_triangle_problem_l2701_270115


namespace NUMINAMATH_CALUDE_square_roots_problem_l2701_270185

theorem square_roots_problem (n : ℝ) (x : ℝ) (hn : n > 0) 
  (h1 : x + 1 = Real.sqrt n) (h2 : x - 5 = Real.sqrt n) : n = 9 := by
  sorry

end NUMINAMATH_CALUDE_square_roots_problem_l2701_270185


namespace NUMINAMATH_CALUDE_system_solution_l2701_270152

theorem system_solution (x y z : ℝ) 
  (h1 : x > 0) (h2 : y > 0) (h3 : z > 0)
  (eq1 : x * y = 4 * z)
  (eq2 : x / y = 81)
  (eq3 : x * z = 36) :
  x = 36 ∧ y = 4/9 ∧ z = 1 := by
  sorry

end NUMINAMATH_CALUDE_system_solution_l2701_270152


namespace NUMINAMATH_CALUDE_increase_by_percentage_l2701_270104

/-- Proves that increasing 90 by 50% results in 135 -/
theorem increase_by_percentage (initial : ℕ) (percentage : ℚ) (result : ℕ) : 
  initial = 90 → percentage = 50 / 100 → result = initial + (initial * percentage) → result = 135 :=
by sorry

end NUMINAMATH_CALUDE_increase_by_percentage_l2701_270104


namespace NUMINAMATH_CALUDE_student_distribution_l2701_270174

/-- The number of ways to distribute n students to k universities --/
def distribute (n : ℕ) (k : ℕ) : ℕ :=
  sorry

/-- The condition that each university admits at least one student --/
def at_least_one (n : ℕ) (k : ℕ) : Prop :=
  sorry

theorem student_distribution :
  ∃ (d : ℕ → ℕ → ℕ), ∃ (c : ℕ → ℕ → Prop),
    d 5 3 = 150 ∧ c 5 3 ∧
    ∀ (n k : ℕ), c n k → d n k = distribute n k :=
by sorry

end NUMINAMATH_CALUDE_student_distribution_l2701_270174


namespace NUMINAMATH_CALUDE_geometric_sequence_sum_l2701_270190

/-- Given a geometric sequence {a_n} with first term a₁ and common ratio q,
    prove that if a₂ * a₃ = 2a₁ and the arithmetic mean of (1/2)a₄ and a₇ is 5/8,
    then the sum of the first 4 terms (S₄) is 30. -/
theorem geometric_sequence_sum (a₁ q : ℝ) (h1 : a₁ * q * (a₁ * q^2) = 2 * a₁)
    (h2 : (1/2 * a₁ * q^3 + a₁ * q^6) / 2 = 5/8) :
  a₁ * (1 - q^4) / (1 - q) = 30 := by
  sorry


end NUMINAMATH_CALUDE_geometric_sequence_sum_l2701_270190


namespace NUMINAMATH_CALUDE_factorable_p_values_l2701_270148

def is_factorable (p : ℤ) : Prop :=
  ∃ (a b : ℤ), ∀ x, x^2 + p*x + 12 = (x + a) * (x + b)

theorem factorable_p_values (p : ℤ) :
  is_factorable p ↔ p ∈ ({-13, -8, -7, 7, 8, 13} : Set ℤ) := by
  sorry

end NUMINAMATH_CALUDE_factorable_p_values_l2701_270148


namespace NUMINAMATH_CALUDE_absolute_opposite_reciprocal_of_negative_three_halves_l2701_270188

theorem absolute_opposite_reciprocal_of_negative_three_halves :
  let x : ℚ := -3/2
  (abs x = 3/2) ∧ (-x = 3/2) ∧ (x⁻¹ = -2/3) := by
  sorry

end NUMINAMATH_CALUDE_absolute_opposite_reciprocal_of_negative_three_halves_l2701_270188


namespace NUMINAMATH_CALUDE_solution_in_first_quadrant_l2701_270102

theorem solution_in_first_quadrant (d : ℝ) :
  (∃ x y : ℝ, x - 2*y = 5 ∧ d*x + y = 6 ∧ x > 0 ∧ y > 0) ↔ -1/2 < d ∧ d < 6/5 := by
  sorry

end NUMINAMATH_CALUDE_solution_in_first_quadrant_l2701_270102


namespace NUMINAMATH_CALUDE_sqrt_of_sqrt_plus_one_over_two_is_integer_l2701_270192

theorem sqrt_of_sqrt_plus_one_over_two_is_integer (n : ℕ+) 
  (h : ∃ (x : ℕ+), x^2 = 12 * n^2 + 1) :
  ∃ (q : ℕ+), q^2 = (Nat.sqrt (12 * n^2 + 1) + 1) / 2 := by
sorry

end NUMINAMATH_CALUDE_sqrt_of_sqrt_plus_one_over_two_is_integer_l2701_270192


namespace NUMINAMATH_CALUDE_triangle_existence_l2701_270103

/-- Given a perimeter, inscribed circle radius, and an angle, 
    there exists a triangle with these properties -/
theorem triangle_existence (s ρ α : ℝ) (h1 : s > 0) (h2 : ρ > 0) (h3 : 0 < α ∧ α < π) :
  ∃ (a b c : ℝ), 
    0 < a ∧ 0 < b ∧ 0 < c ∧  -- Positive side lengths
    a + b + c = 2 * s ∧      -- Perimeter condition
    ρ = (a * b * c) / (4 * s) ∧  -- Inscribed circle radius formula
    α = Real.arccos ((b^2 + c^2 - a^2) / (2 * b * c)) :=  -- Cosine law for angle
by sorry

end NUMINAMATH_CALUDE_triangle_existence_l2701_270103


namespace NUMINAMATH_CALUDE_max_value_sqrt_sum_l2701_270177

theorem max_value_sqrt_sum (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) 
  (h_sum : a + b + 9 * c^2 = 1) : 
  Real.sqrt a + Real.sqrt b + Real.sqrt 3 * c ≤ Real.sqrt 21 / 3 ∧ 
  ∃ a₀ b₀ c₀ : ℝ, 0 < a₀ ∧ 0 < b₀ ∧ 0 < c₀ ∧ 
  a₀ + b₀ + 9 * c₀^2 = 1 ∧ 
  Real.sqrt a₀ + Real.sqrt b₀ + Real.sqrt 3 * c₀ = Real.sqrt 21 / 3 :=
sorry

end NUMINAMATH_CALUDE_max_value_sqrt_sum_l2701_270177


namespace NUMINAMATH_CALUDE_consecutive_cs_majors_probability_l2701_270157

/-- The number of people sitting at the round table -/
def total_people : ℕ := 12

/-- The number of computer science majors -/
def cs_majors : ℕ := 5

/-- The number of engineering majors -/
def eng_majors : ℕ := 4

/-- The number of art majors -/
def art_majors : ℕ := 3

/-- The probability of all computer science majors sitting consecutively -/
def consecutive_cs_prob : ℚ := 1 / 66

theorem consecutive_cs_majors_probability :
  (total_people = cs_majors + eng_majors + art_majors) →
  (consecutive_cs_prob = (total_people : ℚ) / (total_people.choose cs_majors)) := by
  sorry

end NUMINAMATH_CALUDE_consecutive_cs_majors_probability_l2701_270157


namespace NUMINAMATH_CALUDE_sum_of_extremes_in_third_row_l2701_270107

/-- Represents a position in the grid -/
structure Position :=
  (x : ℕ)
  (y : ℕ)

/-- Represents the spiral grid -/
def SpiralGrid :=
  Position → ℕ

/-- The size of the grid -/
def gridSize : ℕ := 17

/-- The total number of cells in the grid -/
def totalCells : ℕ := gridSize * gridSize

/-- The center position of the grid -/
def centerPosition : Position :=
  ⟨gridSize / 2, gridSize / 2⟩

/-- Creates a spiral grid with numbers from 1 to totalCells -/
def createSpiralGrid : SpiralGrid := sorry

/-- Gets the number at a specific position in the grid -/
def getNumber (grid : SpiralGrid) (pos : Position) : ℕ := sorry

/-- Finds the smallest number in the third row -/
def smallestInThirdRow (grid : SpiralGrid) : ℕ := sorry

/-- Finds the largest number in the third row -/
def largestInThirdRow (grid : SpiralGrid) : ℕ := sorry

/-- The main theorem to prove -/
theorem sum_of_extremes_in_third_row :
  let grid := createSpiralGrid
  smallestInThirdRow grid + largestInThirdRow grid = 544 := by sorry

end NUMINAMATH_CALUDE_sum_of_extremes_in_third_row_l2701_270107


namespace NUMINAMATH_CALUDE_strawberry_smoothies_l2701_270106

theorem strawberry_smoothies (initial_strawberries additional_strawberries strawberries_per_smoothie : ℚ)
  (h1 : initial_strawberries = 28)
  (h2 : additional_strawberries = 35)
  (h3 : strawberries_per_smoothie = 7.5) :
  ⌊(initial_strawberries + additional_strawberries) / strawberries_per_smoothie⌋ = 8 := by
  sorry

end NUMINAMATH_CALUDE_strawberry_smoothies_l2701_270106


namespace NUMINAMATH_CALUDE_pure_imaginary_complex_number_l2701_270182

theorem pure_imaginary_complex_number (m : ℝ) : 
  let i : ℂ := Complex.I
  let Z : ℂ := (1 + i) / (1 - i) + m * (1 - i)
  (Z.re = 0) → m = 0 := by
  sorry

end NUMINAMATH_CALUDE_pure_imaginary_complex_number_l2701_270182


namespace NUMINAMATH_CALUDE_mall_computer_sales_l2701_270121

theorem mall_computer_sales (planned_sales : ℕ) (golden_week_avg : ℕ) (increase_percent : ℕ) (remaining_days : ℕ) :
  planned_sales = 900 →
  golden_week_avg = 54 →
  increase_percent = 30 →
  remaining_days = 24 →
  (∃ x : ℕ, x ≥ 33 ∧ golden_week_avg * 7 + x * remaining_days ≥ planned_sales + planned_sales * increase_percent / 100) :=
by
  sorry

#check mall_computer_sales

end NUMINAMATH_CALUDE_mall_computer_sales_l2701_270121


namespace NUMINAMATH_CALUDE_f_equation_solution_l2701_270110

def f (x : ℝ) : ℝ := 3 * x - 5

theorem f_equation_solution :
  ∃ x : ℝ, 1 = f (x - 6) ∧ x = 8 := by
  sorry

end NUMINAMATH_CALUDE_f_equation_solution_l2701_270110


namespace NUMINAMATH_CALUDE_ezekiel_shoe_pairs_l2701_270130

/-- The number of shoes in a pair -/
def shoes_per_pair : ℕ := 2

/-- The total number of new shoes Ezekiel has -/
def total_shoes : ℕ := 6

/-- The number of pairs of shoes Ezekiel bought -/
def pairs_bought : ℕ := total_shoes / shoes_per_pair

theorem ezekiel_shoe_pairs : pairs_bought = 3 := by
  sorry

end NUMINAMATH_CALUDE_ezekiel_shoe_pairs_l2701_270130


namespace NUMINAMATH_CALUDE_coffee_shop_weekly_production_l2701_270171

/-- A coffee shop that brews a certain number of coffee cups per day -/
structure CoffeeShop where
  weekday_cups_per_hour : ℕ
  weekend_total_cups : ℕ
  hours_open_per_day : ℕ

/-- Calculate the total number of coffee cups brewed in one week -/
def weekly_coffee_cups (shop : CoffeeShop) : ℕ :=
  let weekday_cups := shop.weekday_cups_per_hour * shop.hours_open_per_day * 5
  let weekend_cups := shop.weekend_total_cups
  weekday_cups + weekend_cups

/-- Theorem stating that a coffee shop with given parameters brews 370 cups in a week -/
theorem coffee_shop_weekly_production :
  ∀ (shop : CoffeeShop),
    shop.weekday_cups_per_hour = 10 →
    shop.weekend_total_cups = 120 →
    shop.hours_open_per_day = 5 →
    weekly_coffee_cups shop = 370 :=
by
  sorry

end NUMINAMATH_CALUDE_coffee_shop_weekly_production_l2701_270171


namespace NUMINAMATH_CALUDE_solution_range_l2701_270189

def A : Set ℝ := {x | (x + 1) / (x - 3) ≤ 0 ∧ x ≠ 3}
def B (a : ℝ) : Set ℝ := {x | x^2 - 2*a*x - 3*a^2 ≤ 0}

theorem solution_range (a : ℝ) : 
  (∀ x, x ∈ B a → x ∈ A) ↔ -1/3 ≤ a ∧ a < 1 :=
sorry

end NUMINAMATH_CALUDE_solution_range_l2701_270189


namespace NUMINAMATH_CALUDE_air_conditioner_energy_savings_l2701_270136

/-- Represents the monthly energy savings in kWh for an air conditioner type -/
structure EnergySavings where
  savings : ℝ

/-- Represents the two types of air conditioners -/
inductive AirConditionerType
  | A
  | B

/-- The energy savings after raising temperature and cleaning for both air conditioner types -/
def energy_savings_after_measures (x y : EnergySavings) : ℝ :=
  x.savings + 1.1 * y.savings

/-- The theorem to be proved -/
theorem air_conditioner_energy_savings 
  (savings_A savings_B : EnergySavings) :
  savings_A.savings - savings_B.savings = 27 ∧
  energy_savings_after_measures savings_A savings_B = 405 →
  savings_A.savings = 207 ∧ savings_B.savings = 180 := by
  sorry

end NUMINAMATH_CALUDE_air_conditioner_energy_savings_l2701_270136


namespace NUMINAMATH_CALUDE_curve_symmetry_l2701_270123

/-- The curve represented by the equation xy(x+y)=1 is symmetric about the line y=x -/
theorem curve_symmetry (x y : ℝ) : x * y * (x + y) = 1 ↔ y * x * (y + x) = 1 := by
  sorry

end NUMINAMATH_CALUDE_curve_symmetry_l2701_270123


namespace NUMINAMATH_CALUDE_flower_city_theorem_l2701_270173

/-- A bipartite graph representing the relationship between short men and little girls -/
structure FlowerCityGraph where
  A : Type -- Set of short men
  B : Type -- Set of little girls
  edge : A → B → Prop -- Edge relation

/-- The property that each short man knows exactly 6 little girls -/
def each_man_knows_six_girls (G : FlowerCityGraph) : Prop :=
  ∀ a : G.A, (∃! (b1 b2 b3 b4 b5 b6 : G.B), 
    G.edge a b1 ∧ G.edge a b2 ∧ G.edge a b3 ∧ G.edge a b4 ∧ G.edge a b5 ∧ G.edge a b6 ∧
    (∀ b : G.B, G.edge a b → (b = b1 ∨ b = b2 ∨ b = b3 ∨ b = b4 ∨ b = b5 ∨ b = b6)))

/-- The property that each little girl knows exactly 6 short men -/
def each_girl_knows_six_men (G : FlowerCityGraph) : Prop :=
  ∀ b : G.B, (∃! (a1 a2 a3 a4 a5 a6 : G.A), 
    G.edge a1 b ∧ G.edge a2 b ∧ G.edge a3 b ∧ G.edge a4 b ∧ G.edge a5 b ∧ G.edge a6 b ∧
    (∀ a : G.A, G.edge a b → (a = a1 ∨ a = a2 ∨ a = a3 ∨ a = a4 ∨ a = a5 ∨ a = a6)))

/-- The theorem stating that the number of short men equals the number of little girls -/
theorem flower_city_theorem (G : FlowerCityGraph) 
  (h1 : each_man_knows_six_girls G) 
  (h2 : each_girl_knows_six_men G) : 
  Nonempty (Equiv G.A G.B) :=
sorry

end NUMINAMATH_CALUDE_flower_city_theorem_l2701_270173


namespace NUMINAMATH_CALUDE_factorization_of_quadratic_l2701_270141

theorem factorization_of_quadratic (a : ℝ) : a^2 - 2*a - 15 = (a + 3) * (a - 5) := by
  sorry

end NUMINAMATH_CALUDE_factorization_of_quadratic_l2701_270141


namespace NUMINAMATH_CALUDE_sequence_sum_l2701_270114

/-- An arithmetic sequence -/
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

/-- A geometric sequence with positive terms -/
def geometric_sequence (b : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, q > 0 ∧ ∀ n : ℕ, b (n + 1) = b n * q

theorem sequence_sum (a b : ℕ → ℝ) :
  arithmetic_sequence a →
  geometric_sequence b →
  b 1 = 1 →
  b 3 = b 2 + 2 →
  b 4 = a 3 + a 5 →
  b 5 = a 4 + 2 * a 6 →
  a 2018 + b 9 = 2274 := by
  sorry


end NUMINAMATH_CALUDE_sequence_sum_l2701_270114


namespace NUMINAMATH_CALUDE_circle_triangle_angle_measure_l2701_270147

-- Define the circle and triangle
def Circle : Type := Unit
def Point : Type := Unit
def Triangle : Type := Unit

-- Define the center of the circle
def center (c : Circle) : Point := sorry

-- Define the vertices of the triangle
def X (t : Triangle) : Point := sorry
def Y (t : Triangle) : Point := sorry
def Z (t : Triangle) : Point := sorry

-- Define the property of being circumscribed
def is_circumscribed (c : Circle) (t : Triangle) : Prop := sorry

-- Define the measure of an angle
def angle_measure (p q r : Point) : ℝ := sorry

-- State the theorem
theorem circle_triangle_angle_measure 
  (c : Circle) (t : Triangle) (h_circumscribed : is_circumscribed c t) 
  (h_XOY : angle_measure (X t) (center c) (Y t) = 120)
  (h_YOZ : angle_measure (Y t) (center c) (Z t) = 140) :
  angle_measure (X t) (Y t) (Z t) = 60 := by sorry

end NUMINAMATH_CALUDE_circle_triangle_angle_measure_l2701_270147


namespace NUMINAMATH_CALUDE_congruence_problem_l2701_270145

theorem congruence_problem (x : ℤ) : 
  x ≡ 1 [ZMOD 27] ∧ x ≡ 6 [ZMOD 37] → x ≡ 110 [ZMOD 999] := by
  sorry

end NUMINAMATH_CALUDE_congruence_problem_l2701_270145


namespace NUMINAMATH_CALUDE_negation_of_universal_proposition_l2701_270120

theorem negation_of_universal_proposition :
  (¬ ∀ x : ℝ, x^2 - 2*x + 2 ≤ 0) ↔ (∃ x : ℝ, x^2 - 2*x + 2 > 0) := by sorry

end NUMINAMATH_CALUDE_negation_of_universal_proposition_l2701_270120


namespace NUMINAMATH_CALUDE_log_product_equals_three_eighths_l2701_270108

theorem log_product_equals_three_eighths :
  (1/2) * (Real.log 3 / Real.log 2) * (1/2) * (Real.log 8 / Real.log 9) = 3/8 := by
  sorry

end NUMINAMATH_CALUDE_log_product_equals_three_eighths_l2701_270108
