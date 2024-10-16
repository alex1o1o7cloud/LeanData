import Mathlib

namespace NUMINAMATH_CALUDE_ice_cream_scoops_left_l790_79010

/-- Represents the flavors of ice cream --/
inductive Flavor
  | Chocolate
  | Strawberry
  | Vanilla

/-- Represents a person --/
inductive Person
  | Ethan
  | Lucas
  | Danny
  | Connor
  | Olivia
  | Shannon

/-- The number of scoops in each carton --/
def scoops_per_carton : ℕ := 10

/-- The initial number of scoops for each flavor --/
def initial_scoops (f : Flavor) : ℕ := scoops_per_carton

/-- The number of scoops a person wants for each flavor --/
def scoops_wanted (p : Person) (f : Flavor) : ℕ :=
  match p, f with
  | Person.Ethan, Flavor.Chocolate => 1
  | Person.Ethan, Flavor.Vanilla => 1
  | Person.Lucas, Flavor.Chocolate => 2
  | Person.Danny, Flavor.Chocolate => 2
  | Person.Connor, Flavor.Chocolate => 2
  | Person.Olivia, Flavor.Strawberry => 1
  | Person.Olivia, Flavor.Vanilla => 1
  | Person.Shannon, Flavor.Strawberry => 2
  | Person.Shannon, Flavor.Vanilla => 2
  | _, _ => 0

/-- The total number of scoops taken for each flavor --/
def total_scoops_taken (f : Flavor) : ℕ :=
  (scoops_wanted Person.Ethan f) +
  (scoops_wanted Person.Lucas f) +
  (scoops_wanted Person.Danny f) +
  (scoops_wanted Person.Connor f) +
  (scoops_wanted Person.Olivia f) +
  (scoops_wanted Person.Shannon f)

/-- The number of scoops left for each flavor --/
def scoops_left (f : Flavor) : ℕ :=
  initial_scoops f - total_scoops_taken f

/-- The total number of scoops left --/
def total_scoops_left : ℕ :=
  (scoops_left Flavor.Chocolate) +
  (scoops_left Flavor.Strawberry) +
  (scoops_left Flavor.Vanilla)

theorem ice_cream_scoops_left : total_scoops_left = 16 := by
  sorry

end NUMINAMATH_CALUDE_ice_cream_scoops_left_l790_79010


namespace NUMINAMATH_CALUDE_smallest_integer_y_smallest_integer_solution_l790_79008

theorem smallest_integer_y (y : ℤ) : (8 - 3 * y < 26) ↔ (-5 ≤ y) := by sorry

theorem smallest_integer_solution : ∃ (y : ℤ), (8 - 3 * y < 26) ∧ (∀ (z : ℤ), z < y → 8 - 3 * z ≥ 26) := by sorry

end NUMINAMATH_CALUDE_smallest_integer_y_smallest_integer_solution_l790_79008


namespace NUMINAMATH_CALUDE_middle_marble_radius_l790_79046

/-- Given a sequence of five marbles with radii forming a geometric sequence,
    where the smallest radius is 8 and the largest radius is 18,
    prove that the middle (third) marble has a radius of 12. -/
theorem middle_marble_radius 
  (r : Fin 5 → ℝ)  -- r is a function mapping the index of each marble to its radius
  (h_geom_seq : ∀ i j k, i < j → j < k → r j ^ 2 = r i * r k)  -- geometric sequence condition
  (h_smallest : r 0 = 8)  -- radius of the smallest marble
  (h_largest : r 4 = 18)  -- radius of the largest marble
  : r 2 = 12 := by  -- radius of the middle (third) marble
sorry


end NUMINAMATH_CALUDE_middle_marble_radius_l790_79046


namespace NUMINAMATH_CALUDE_park_length_l790_79019

/-- Given a rectangular park with width 9 km and perimeter 46 km, its length is 14 km. -/
theorem park_length (width : ℝ) (perimeter : ℝ) (length : ℝ) : 
  width = 9 → perimeter = 46 → perimeter = 2 * (length + width) → length = 14 := by
  sorry

end NUMINAMATH_CALUDE_park_length_l790_79019


namespace NUMINAMATH_CALUDE_mike_lego_bridge_l790_79024

/-- Calculates the number of bricks of other types Mike needs for his LEGO bridge. -/
def other_bricks (type_a : ℕ) (total : ℕ) : ℕ :=
  total - (type_a + type_a / 2)

/-- Theorem stating that Mike will use 90 bricks of other types for his LEGO bridge. -/
theorem mike_lego_bridge :
  ∀ (type_a : ℕ) (total : ℕ),
    type_a ≥ 40 →
    total = 150 →
    other_bricks type_a total = 90 := by
  sorry

end NUMINAMATH_CALUDE_mike_lego_bridge_l790_79024


namespace NUMINAMATH_CALUDE_ray_AB_bisects_PAQ_l790_79022

-- Define the circle C
def circle_C (x y : ℝ) : Prop :=
  (x - 2)^2 + (y - 5/2)^2 = 25/4

-- Define points A and B on the y-axis
def point_A : ℝ × ℝ := (0, 4)
def point_B : ℝ × ℝ := (0, 1)

-- Define the ellipse
def ellipse (x y : ℝ) : Prop :=
  x^2/8 + y^2/4 = 1

-- Define line l passing through B
def line_l (x y : ℝ) (k : ℝ) : Prop :=
  y = k * x + 1

-- Theorem statement
theorem ray_AB_bisects_PAQ :
  ∀ (P Q : ℝ × ℝ) (k : ℝ),
    circle_C 2 0 →  -- Circle C is tangent to x-axis at T(2,0)
    |point_A.2 - point_B.2| = 3 →  -- |AB| = 3
    line_l P.1 P.2 k →  -- P is on line l
    line_l Q.1 Q.2 k →  -- Q is on line l
    ellipse P.1 P.2 →  -- P is on the ellipse
    ellipse Q.1 Q.2 →  -- Q is on the ellipse
    -- Ray AB bisects angle PAQ
    (P.2 - point_A.2) / (P.1 - point_A.1) + (Q.2 - point_A.2) / (Q.1 - point_A.1) = 0 :=
by
  sorry


end NUMINAMATH_CALUDE_ray_AB_bisects_PAQ_l790_79022


namespace NUMINAMATH_CALUDE_cubic_equation_root_l790_79082

theorem cubic_equation_root (c d : ℚ) : 
  (3 + Real.sqrt 5 : ℂ) ^ 3 + c * (3 + Real.sqrt 5 : ℂ) ^ 2 + d * (3 + Real.sqrt 5 : ℂ) + 15 = 0 →
  d = -18.5 := by
sorry

end NUMINAMATH_CALUDE_cubic_equation_root_l790_79082


namespace NUMINAMATH_CALUDE_product_of_numbers_l790_79095

theorem product_of_numbers (x y : ℝ) (h1 : x - y = 11) (h2 : x^2 + y^2 = 205) : x * y = 42 := by
  sorry

end NUMINAMATH_CALUDE_product_of_numbers_l790_79095


namespace NUMINAMATH_CALUDE_quadratic_equation_solution_l790_79048

theorem quadratic_equation_solution (a k : ℝ) : 
  (∃ x : ℝ, a * x^2 + 3 * x - k = 0) → 
  (k = 44) → 
  (a * 4^2 + 3 * 4 - k = 0) → 
  a = 2 := by
sorry

end NUMINAMATH_CALUDE_quadratic_equation_solution_l790_79048


namespace NUMINAMATH_CALUDE_total_tuition_correct_l790_79012

/-- The total tuition fee that Bran needs to pay -/
def total_tuition : ℝ := 90

/-- Bran's monthly earnings from his part-time job -/
def monthly_earnings : ℝ := 15

/-- The percentage of tuition covered by Bran's scholarship -/
def scholarship_percentage : ℝ := 0.3

/-- The number of months Bran has to pay his tuition -/
def payment_period : ℕ := 3

/-- The amount Bran still needs to pay after scholarship and earnings -/
def remaining_payment : ℝ := 18

/-- Theorem stating that the total tuition is correct given the conditions -/
theorem total_tuition_correct :
  (1 - scholarship_percentage) * total_tuition - 
  (monthly_earnings * payment_period) = remaining_payment :=
by sorry

end NUMINAMATH_CALUDE_total_tuition_correct_l790_79012


namespace NUMINAMATH_CALUDE_mean_study_days_is_4_05_l790_79001

/-- Represents the study data for Ms. Rossi's class -/
structure StudyData where
  oneDay : Nat
  twoDays : Nat
  fourDays : Nat
  fiveDays : Nat
  sixDays : Nat

/-- Calculates the mean number of study days for the given data -/
def calculateMean (data : StudyData) : Float :=
  let totalDays := data.oneDay * 1 + data.twoDays * 2 + data.fourDays * 4 + data.fiveDays * 5 + data.sixDays * 6
  let totalStudents := data.oneDay + data.twoDays + data.fourDays + data.fiveDays + data.sixDays
  (totalDays.toFloat) / (totalStudents.toFloat)

/-- Theorem stating that the mean number of study days for Ms. Rossi's class is 4.05 -/
theorem mean_study_days_is_4_05 (data : StudyData) 
  (h1 : data.oneDay = 2)
  (h2 : data.twoDays = 4)
  (h3 : data.fourDays = 5)
  (h4 : data.fiveDays = 7)
  (h5 : data.sixDays = 4) :
  calculateMean data = 4.05 := by
  sorry

end NUMINAMATH_CALUDE_mean_study_days_is_4_05_l790_79001


namespace NUMINAMATH_CALUDE_cubic_equation_root_l790_79079

theorem cubic_equation_root : 
  ∃ (x : ℝ), x = -4/3 ∧ (x + 1)^(1/3) + (2*x + 3)^(1/3) + 3*x + 4 = 0 :=
sorry

end NUMINAMATH_CALUDE_cubic_equation_root_l790_79079


namespace NUMINAMATH_CALUDE_equation_solution_l790_79078

theorem equation_solution : ∃ x : ℝ, (x - 3)^2 = x^2 - 9 ∧ x = 3 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l790_79078


namespace NUMINAMATH_CALUDE_ice_cream_volume_specific_ice_cream_volume_l790_79044

/-- The volume of ice cream in a right circular cone with a hemisphere on top -/
theorem ice_cream_volume (h : ℝ) (r : ℝ) (h_pos : h > 0) (r_pos : r > 0) :
  let cone_volume := (1/3) * π * r^2 * h
  let hemisphere_volume := (2/3) * π * r^3
  cone_volume + hemisphere_volume = (320/3) * π :=
by
  sorry

/-- The specific case with h = 12 and r = 4 -/
theorem specific_ice_cream_volume :
  let h : ℝ := 12
  let r : ℝ := 4
  let cone_volume := (1/3) * π * r^2 * h
  let hemisphere_volume := (2/3) * π * r^3
  cone_volume + hemisphere_volume = (320/3) * π :=
by
  sorry

end NUMINAMATH_CALUDE_ice_cream_volume_specific_ice_cream_volume_l790_79044


namespace NUMINAMATH_CALUDE_no_valid_list_exists_l790_79091

theorem no_valid_list_exists : ¬ ∃ (list : List ℤ), 
  (list.length = 10) ∧ 
  (∀ i j k, i + 1 = j ∧ j + 1 = k → i < list.length ∧ k < list.length → 
    (list.get ⟨i, sorry⟩ * list.get ⟨j, sorry⟩ * list.get ⟨k, sorry⟩) % 6 = 0) ∧
  (∀ i j, i + 1 = j → j < list.length → 
    (list.get ⟨i, sorry⟩ * list.get ⟨j, sorry⟩) % 6 ≠ 0) :=
by sorry

end NUMINAMATH_CALUDE_no_valid_list_exists_l790_79091


namespace NUMINAMATH_CALUDE_no_real_roots_min_value_is_three_l790_79049

-- Define the quadratic function
def f (m x : ℝ) : ℝ := x^2 - 2*m*x + m^2 + 3

-- Theorem 1: The quadratic equation has no real solutions for any m
theorem no_real_roots (m : ℝ) : ∀ x : ℝ, f m x ≠ 0 := by sorry

-- Theorem 2: The minimum value of the function is 3 for all m
theorem min_value_is_three (m : ℝ) : ∀ x : ℝ, f m x ≥ 3 := by sorry

end NUMINAMATH_CALUDE_no_real_roots_min_value_is_three_l790_79049


namespace NUMINAMATH_CALUDE_negation_of_quadratic_inequality_l790_79050

theorem negation_of_quadratic_inequality :
  (¬ ∀ x : ℝ, x^2 - x + 1 ≥ 0) ↔ (∃ x : ℝ, x^2 - x + 1 < 0) := by sorry

end NUMINAMATH_CALUDE_negation_of_quadratic_inequality_l790_79050


namespace NUMINAMATH_CALUDE_stratified_sampling_sophomores_l790_79035

/-- Given a school with 2000 students, of which 700 are sophomores,
    and a stratified sample of 100 students, the number of sophomores
    in the sample should be 35. -/
theorem stratified_sampling_sophomores :
  ∀ (total_students sample_size num_sophomores : ℕ),
    total_students = 2000 →
    sample_size = 100 →
    num_sophomores = 700 →
    (num_sophomores * sample_size) / total_students = 35 :=
by
  sorry

#check stratified_sampling_sophomores

end NUMINAMATH_CALUDE_stratified_sampling_sophomores_l790_79035


namespace NUMINAMATH_CALUDE_name_tag_area_l790_79065

/-- The area of a square name tag with side length 11 cm is 121 cm² -/
theorem name_tag_area : 
  let side_length : ℝ := 11
  let area : ℝ := side_length * side_length
  area = 121 := by sorry

end NUMINAMATH_CALUDE_name_tag_area_l790_79065


namespace NUMINAMATH_CALUDE_shells_added_l790_79097

/-- Given that Jovana initially had 5 pounds of shells and now has 28 pounds,
    prove that she added 23 pounds of shells. -/
theorem shells_added (initial : ℕ) (final : ℕ) (h1 : initial = 5) (h2 : final = 28) :
  final - initial = 23 := by
  sorry

end NUMINAMATH_CALUDE_shells_added_l790_79097


namespace NUMINAMATH_CALUDE_common_tangent_range_l790_79013

/-- The range of a for which y = ln x and y = ax² have a common tangent line -/
theorem common_tangent_range (a : ℝ) : 
  (a > 0 ∧ ∃ (x₁ x₂ : ℝ), x₁ > 0 ∧ x₂ > 0 ∧ 
    (1 / x₁ = 2 * a * x₂) ∧ 
    (Real.log x₁ - 1 = -a * x₂^2)) ↔ 
  a ≥ 1 / (2 * Real.exp 1) :=
sorry

end NUMINAMATH_CALUDE_common_tangent_range_l790_79013


namespace NUMINAMATH_CALUDE_new_quadratic_from_roots_sum_product_l790_79088

theorem new_quadratic_from_roots_sum_product (a b c : ℝ) (ha : a ≠ 0) :
  let original_eq := fun x => a * x^2 + b * x + c
  let new_eq := fun x => a^2 * x^2 + (a*b - a*c) * x - b*c
  let sum_of_roots := -b / a
  let product_of_roots := c / a
  (∀ x, original_eq x = 0 ↔ x = sum_of_roots ∨ x = product_of_roots) →
  (∀ x, new_eq x = 0 ↔ x = sum_of_roots ∨ x = product_of_roots) :=
by sorry

end NUMINAMATH_CALUDE_new_quadratic_from_roots_sum_product_l790_79088


namespace NUMINAMATH_CALUDE_chores_repayment_l790_79029

/-- Calculates the amount earned for a given hour in the chore cycle -/
def hourly_rate (hour : ℕ) : ℕ :=
  match hour % 3 with
  | 1 => 2
  | 2 => 4
  | 0 => 6
  | _ => 0 -- This case should never occur, but Lean requires it for completeness

/-- Calculates the total amount earned for a given number of hours -/
def total_earned (hours : ℕ) : ℕ :=
  (List.range hours).map hourly_rate |>.sum

/-- The main theorem stating that 45 hours of chores results in $180 earned -/
theorem chores_repayment : total_earned 45 = 180 := by
  sorry

end NUMINAMATH_CALUDE_chores_repayment_l790_79029


namespace NUMINAMATH_CALUDE_specific_triangle_intercepted_segments_l790_79036

/-- Represents a right triangle with an inscribed circle -/
structure RightTriangleWithInscribedCircle where
  side1 : ℝ
  side2 : ℝ
  hypotenuse : ℝ
  (right_triangle : side1^2 + side2^2 = hypotenuse^2)

/-- Calculates the lengths of segments intercepted by lines drawn through the center of the inscribed circle parallel to the sides of the triangle -/
def intercepted_segments (triangle : RightTriangleWithInscribedCircle) : (ℝ × ℝ × ℝ) :=
  sorry

/-- Theorem statement for the specific right triangle with sides 6, 8, and 10 -/
theorem specific_triangle_intercepted_segments :
  let triangle : RightTriangleWithInscribedCircle := {
    side1 := 6,
    side2 := 8,
    hypotenuse := 10,
    right_triangle := by norm_num
  }
  intercepted_segments triangle = (3/2, 8/3, 25/6) := by sorry

end NUMINAMATH_CALUDE_specific_triangle_intercepted_segments_l790_79036


namespace NUMINAMATH_CALUDE_max_words_is_16056_l790_79040

/-- Represents a language with two letters and words of maximum length 13 -/
structure TwoLetterLanguage where
  max_word_length : ℕ
  max_word_length_eq : max_word_length = 13

/-- Calculates the maximum number of words in the language -/
def max_words (L : TwoLetterLanguage) : ℕ :=
  2^14 - 2^7

/-- States that no concatenation of two words forms another word -/
axiom no_concat_word (L : TwoLetterLanguage) :
  ∀ (w1 w2 : String), (w1.length ≤ L.max_word_length ∧ w2.length ≤ L.max_word_length) →
    (w1 ++ w2).length > L.max_word_length

/-- Theorem: The maximum number of words in the language is 16056 -/
theorem max_words_is_16056 (L : TwoLetterLanguage) :
  max_words L = 16056 := by
  sorry

end NUMINAMATH_CALUDE_max_words_is_16056_l790_79040


namespace NUMINAMATH_CALUDE_square_increasing_on_positive_reals_l790_79092

theorem square_increasing_on_positive_reals :
  ∀ x₁ x₂ : ℝ, 0 < x₁ → 0 < x₂ → x₁ < x₂ → x₁^2 < x₂^2 := by
  sorry

end NUMINAMATH_CALUDE_square_increasing_on_positive_reals_l790_79092


namespace NUMINAMATH_CALUDE_minimum_groups_l790_79090

theorem minimum_groups (n : Nat) (h : n = 29) : 
  Nat.ceil (n / 4 : ℚ) = 8 := by
  sorry

end NUMINAMATH_CALUDE_minimum_groups_l790_79090


namespace NUMINAMATH_CALUDE_y_is_odd_square_l790_79032

def x : ℕ → ℤ
  | 0 => 0
  | 1 => 1
  | (n + 2) => 3 * x (n + 1) - 2 * x n

def y (n : ℕ) : ℤ := x n ^ 2 + 2 ^ (n + 2)

theorem y_is_odd_square (n : ℕ) (h : n > 0) :
  ∃ k : ℤ, Odd k ∧ y n = k ^ 2 := by
  sorry

end NUMINAMATH_CALUDE_y_is_odd_square_l790_79032


namespace NUMINAMATH_CALUDE_sqrt_three_squared_equals_three_l790_79023

theorem sqrt_three_squared_equals_three : (Real.sqrt 3)^2 = 3 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_three_squared_equals_three_l790_79023


namespace NUMINAMATH_CALUDE_max_N_is_seven_l790_79063

def J (k : ℕ) : ℕ := 10^(k+3) + 128

def N (k : ℕ) : ℕ := (J k).factors.count 2

theorem max_N_is_seven : ∀ k : ℕ, k > 0 → N k ≤ 7 ∧ ∃ k₀ : ℕ, k₀ > 0 ∧ N k₀ = 7 :=
sorry

end NUMINAMATH_CALUDE_max_N_is_seven_l790_79063


namespace NUMINAMATH_CALUDE_min_value_M_l790_79084

theorem min_value_M (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  let M := (((a / (b + c)) ^ (1/4)) + ((b / (c + a)) ^ (1/4)) + ((c / (a + b)) ^ (1/4)) +
            ((b + c) / a) ^ (1/2) + ((a + c) / b) ^ (1/2) + ((a + b) / c) ^ (1/2))
  M ≥ 3 * Real.sqrt 2 + (3 * (8 ^ (1/4))) / 8 := by
  sorry

end NUMINAMATH_CALUDE_min_value_M_l790_79084


namespace NUMINAMATH_CALUDE_min_payment_amount_l790_79099

/-- Represents the number of bills of each denomination --/
structure BillCount where
  tens : Nat
  fives : Nat
  ones : Nat

/-- Calculates the total value of bills --/
def totalValue (bills : BillCount) : Nat :=
  10 * bills.tens + 5 * bills.fives + bills.ones

/-- Calculates the total count of bills --/
def totalCount (bills : BillCount) : Nat :=
  bills.tens + bills.fives + bills.ones

/-- Represents Tim's initial bill distribution --/
def timsBills : BillCount :=
  { tens := 13, fives := 11, ones := 17 }

/-- Theorem stating the minimum amount Tim can pay using at least 16 bills --/
theorem min_payment_amount (payment : BillCount) : 
  totalCount payment ≥ 16 → 
  totalCount payment ≤ totalCount timsBills → 
  totalValue payment ≥ 40 :=
by sorry

end NUMINAMATH_CALUDE_min_payment_amount_l790_79099


namespace NUMINAMATH_CALUDE_alexa_katerina_weight_l790_79057

/-- The combined weight of Alexa and Katerina is 92 pounds -/
theorem alexa_katerina_weight (total_weight : ℕ) (alexa_weight : ℕ) (michael_weight : ℕ)
  (h1 : total_weight = 154)
  (h2 : alexa_weight = 46)
  (h3 : michael_weight = 62) :
  total_weight - michael_weight = 92 :=
by sorry

end NUMINAMATH_CALUDE_alexa_katerina_weight_l790_79057


namespace NUMINAMATH_CALUDE_imaginary_root_cubic_equation_l790_79015

theorem imaginary_root_cubic_equation (a b q r : ℝ) :
  b ≠ 0 →
  (∃ (x : ℂ), x^3 + q*x + r = 0 ∧ x = a + b*Complex.I) →
  q = b^2 - 3*a^2 := by
  sorry

end NUMINAMATH_CALUDE_imaginary_root_cubic_equation_l790_79015


namespace NUMINAMATH_CALUDE_logarithm_equation_l790_79011

theorem logarithm_equation : 
  (((1 - Real.log 3 / Real.log 6) ^ 2 + 
    (Real.log 2 / Real.log 6) * (Real.log 18 / Real.log 6)) / 
   (Real.log 4 / Real.log 6)) = 1 := by
  sorry

end NUMINAMATH_CALUDE_logarithm_equation_l790_79011


namespace NUMINAMATH_CALUDE_chili_beans_cans_l790_79093

-- Define the ratio of tomato soup cans to chili beans cans
def soup_to_beans_ratio : ℚ := 1 / 2

-- Define the total number of cans
def total_cans : ℕ := 12

-- Theorem to prove
theorem chili_beans_cans (t c : ℕ) 
  (h1 : t + c = total_cans) 
  (h2 : c = 2 * t) : c = 8 := by
  sorry

end NUMINAMATH_CALUDE_chili_beans_cans_l790_79093


namespace NUMINAMATH_CALUDE_min_value_theorem_l790_79052

theorem min_value_theorem (x y : ℝ) (h : x * y > 0) :
  ∃ m : ℝ, m = 4 - 2 * Real.sqrt 2 ∧
    ∀ z : ℝ, z = y / (x + y) + 2 * x / (2 * x + y) → z ≥ m := by
  sorry

end NUMINAMATH_CALUDE_min_value_theorem_l790_79052


namespace NUMINAMATH_CALUDE_jan_2022_is_saturday_l790_79055

/-- Enumeration of days of the week -/
inductive DayOfWeek
  | Sunday
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday
  | Saturday

/-- Function to get the next day of the week -/
def nextDay (d : DayOfWeek) : DayOfWeek :=
  match d with
  | DayOfWeek.Sunday => DayOfWeek.Monday
  | DayOfWeek.Monday => DayOfWeek.Tuesday
  | DayOfWeek.Tuesday => DayOfWeek.Wednesday
  | DayOfWeek.Wednesday => DayOfWeek.Thursday
  | DayOfWeek.Thursday => DayOfWeek.Friday
  | DayOfWeek.Friday => DayOfWeek.Saturday
  | DayOfWeek.Saturday => DayOfWeek.Sunday

/-- Function to advance a day by n days -/
def advanceDay (d : DayOfWeek) (n : Nat) : DayOfWeek :=
  match n with
  | 0 => d
  | n + 1 => advanceDay (nextDay d) n

/-- Theorem: If January 2021 has exactly five Fridays, five Saturdays, and five Sundays,
    then January 1, 2022 falls on a Saturday -/
theorem jan_2022_is_saturday
  (h : ∃ (first_day : DayOfWeek),
       (advanceDay first_day 0 = DayOfWeek.Friday ∧
        advanceDay first_day 1 = DayOfWeek.Saturday ∧
        advanceDay first_day 2 = DayOfWeek.Sunday) ∧
       (∀ (n : Nat), n < 31 → 
        (advanceDay first_day n = DayOfWeek.Friday ∨
         advanceDay first_day n = DayOfWeek.Saturday ∨
         advanceDay first_day n = DayOfWeek.Sunday) →
        (advanceDay first_day (n + 7) = advanceDay first_day n))) :
  advanceDay DayOfWeek.Friday 365 = DayOfWeek.Saturday := by
  sorry


end NUMINAMATH_CALUDE_jan_2022_is_saturday_l790_79055


namespace NUMINAMATH_CALUDE_trapezoid_rhombus_properties_triangle_parallelogram_properties_rectangle_circle_symmetry_l790_79033

-- Define the geometric shapes
class ConvexPolygon
class Polygon extends ConvexPolygon
class Trapezoid extends ConvexPolygon
class Rhombus extends ConvexPolygon
class Triangle extends Polygon
class Parallelogram extends Polygon
class Rectangle extends Polygon
class Circle

-- Define properties
def hasExteriorAngleSum360 (shape : Type) : Prop := sorry
def lineIntersectsTwice (shape : Type) : Prop := sorry
def hasCentralSymmetry (shape : Type) : Prop := sorry

-- Theorem statements
theorem trapezoid_rhombus_properties :
  (hasExteriorAngleSum360 Trapezoid ∧ hasExteriorAngleSum360 Rhombus) ∧
  (lineIntersectsTwice Trapezoid ∧ lineIntersectsTwice Rhombus) := by sorry

theorem triangle_parallelogram_properties :
  (hasExteriorAngleSum360 Triangle ∧ hasExteriorAngleSum360 Parallelogram) ∧
  (lineIntersectsTwice Triangle ∧ lineIntersectsTwice Parallelogram) := by sorry

theorem rectangle_circle_symmetry :
  hasCentralSymmetry Rectangle ∧ hasCentralSymmetry Circle := by sorry

end NUMINAMATH_CALUDE_trapezoid_rhombus_properties_triangle_parallelogram_properties_rectangle_circle_symmetry_l790_79033


namespace NUMINAMATH_CALUDE_fib_F10_units_digit_l790_79062

/-- Fibonacci sequence -/
def fib : ℕ → ℕ
  | 0 => 0
  | 1 => 1
  | (n + 2) => fib (n + 1) + fib n

/-- Units digit of a natural number -/
def unitsDigit (n : ℕ) : ℕ := n % 10

theorem fib_F10_units_digit :
  unitsDigit (fib (fib 10)) = 5 := by sorry

end NUMINAMATH_CALUDE_fib_F10_units_digit_l790_79062


namespace NUMINAMATH_CALUDE_cowbell_coloring_l790_79016

theorem cowbell_coloring (n : ℕ) (hn : n ≥ 3) :
  ∃ (m : ℕ), m = n + 1 ∧
  (∀ (k : ℕ), k > m → 
    ∃ (f : ℕ → Fin n), 
      ∀ (i : ℕ), (∀ (c : Fin n), ∃ (j : ℕ), j < n + 1 ∧ f ((i + j) % k) = c)) ∧
  (¬ ∃ (f : ℕ → Fin n), 
    ∀ (i : ℕ), (∀ (c : Fin n), ∃ (j : ℕ), j < n + 1 ∧ f ((i + j) % m) = c)) :=
by sorry

end NUMINAMATH_CALUDE_cowbell_coloring_l790_79016


namespace NUMINAMATH_CALUDE_grid_coloring_theorem_l790_79058

theorem grid_coloring_theorem (n : ℕ) :
  (∀ (grid : Fin 25 → Fin n → Fin 8),
    ∃ (cols : Fin 4 → Fin n) (rows : Fin 4 → Fin 25),
      ∀ (i j : Fin 4), grid (rows i) (cols j) = grid (rows 0) (cols 0)) ↔
  n ≥ 303601 :=
by sorry

end NUMINAMATH_CALUDE_grid_coloring_theorem_l790_79058


namespace NUMINAMATH_CALUDE_tangent_line_cubic_curve_l790_79028

/-- Given a cubic function f(x) = x³ + ax + b and a line g(x) = kx + 1 tangent to f at x = 1,
    prove that 2a + b = 1 when f(1) = 3. -/
theorem tangent_line_cubic_curve (a b k : ℝ) : 
  (∀ x, (x^3 + a*x + b) = 3 * x^2 + a) →  -- Derivative condition
  (1^3 + a*1 + b = 3) →                   -- Point (1, 3) lies on the curve
  (k*1 + 1 = 3) →                         -- Point (1, 3) lies on the line
  (k = 3*1^2 + a) →                       -- Slope of tangent equals derivative at x = 1
  (2*a + b = 1) := by
sorry

end NUMINAMATH_CALUDE_tangent_line_cubic_curve_l790_79028


namespace NUMINAMATH_CALUDE_largest_three_digit_number_satisfying_condition_l790_79017

/-- Represents a three-digit number -/
structure ThreeDigitNumber where
  hundreds : Nat
  tens : Nat
  ones : Nat
  is_valid : hundreds ≥ 1 ∧ hundreds ≤ 9 ∧ tens ≤ 9 ∧ ones ≤ 9

/-- Converts a ThreeDigitNumber to its numerical value -/
def ThreeDigitNumber.toNat (n : ThreeDigitNumber) : Nat :=
  100 * n.hundreds + 10 * n.tens + n.ones

/-- Calculates the sum of digits of a ThreeDigitNumber -/
def ThreeDigitNumber.digitSum (n : ThreeDigitNumber) : Nat :=
  n.hundreds + n.tens + n.ones

/-- Checks if a ThreeDigitNumber satisfies the given condition -/
def ThreeDigitNumber.satisfiesCondition (n : ThreeDigitNumber) : Prop :=
  n.toNat = n.digitSum + (2 * n.digitSum)^2

theorem largest_three_digit_number_satisfying_condition :
  ∃ (n : ThreeDigitNumber), n.toNat = 915 ∧
    n.satisfiesCondition ∧
    ∀ (m : ThreeDigitNumber), m.satisfiesCondition → m.toNat ≤ n.toNat :=
  sorry

end NUMINAMATH_CALUDE_largest_three_digit_number_satisfying_condition_l790_79017


namespace NUMINAMATH_CALUDE_sequence_length_730_l790_79000

/-- Given a sequence of real numbers satisfying certain conditions, prove that the length of the sequence is 730. -/
theorem sequence_length_730 (n : ℕ+) (b : ℕ → ℝ) : 
  b 0 = 45 → 
  b 1 = 81 → 
  b n = 0 → 
  (∀ k : ℕ, 1 ≤ k ∧ k < n → b (k + 1) = b (k - 1) - 5 / b k) → 
  n = 730 := by
  sorry

end NUMINAMATH_CALUDE_sequence_length_730_l790_79000


namespace NUMINAMATH_CALUDE_second_car_rate_l790_79047

/-- Given two cars starting at the same point, with the first car traveling at 50 mph,
    and after 3 hours the distance between them is 30 miles,
    prove that the rate of the second car is 40 mph. -/
theorem second_car_rate (v : ℝ) : 
  v > 0 →  -- The rate of the second car is positive
  50 * 3 - v * 3 = 30 →  -- After 3 hours, the distance between the cars is 30 miles
  v = 40 := by
sorry

end NUMINAMATH_CALUDE_second_car_rate_l790_79047


namespace NUMINAMATH_CALUDE_problem_solution_l790_79043

theorem problem_solution (p q r : ℝ) (h_distinct : p ≠ q ∧ q ≠ r ∧ r ≠ p) 
  (h_equation : p / (q - r) + q / (r - p) + r / (p - q) = 3) :
  p / (q - r)^2 + q / (r - p)^2 + r / (p - q)^2 = 0 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l790_79043


namespace NUMINAMATH_CALUDE_function_problem_l790_79030

theorem function_problem (f : ℝ → ℝ) (a b c : ℝ) 
  (h_inv : Function.Injective f)
  (h1 : f a = b)
  (h2 : f b = 5)
  (h3 : f c = 3)
  (h4 : c = a + 1) :
  a - b = -2 := by
  sorry

end NUMINAMATH_CALUDE_function_problem_l790_79030


namespace NUMINAMATH_CALUDE_march_walking_distance_l790_79080

theorem march_walking_distance (days_in_month : Nat) (miles_per_day : Nat) (skipped_days : Nat) : 
  days_in_month = 31 → miles_per_day = 4 → skipped_days = 4 → 
  (days_in_month - skipped_days) * miles_per_day = 108 := by
  sorry

end NUMINAMATH_CALUDE_march_walking_distance_l790_79080


namespace NUMINAMATH_CALUDE_binomial_coefficient_sum_squared_difference_l790_79021

theorem binomial_coefficient_sum_squared_difference (a₀ a₁ a₂ a₃ : ℝ) : 
  (∀ x, (Real.sqrt 5 * x - 1)^3 = a₀ + a₁ * x + a₂ * x^2 + a₃ * x^3) →
  (a₀ + a₂)^2 - (a₁ + a₃)^2 = -64 := by
  sorry

end NUMINAMATH_CALUDE_binomial_coefficient_sum_squared_difference_l790_79021


namespace NUMINAMATH_CALUDE_six_digit_repeating_divisible_by_11_l790_79064

/-- A 6-digit integer where the first three digits and the last three digits
    form the same three-digit number in the same order is divisible by 11. -/
theorem six_digit_repeating_divisible_by_11 (N : ℕ) (a b c : ℕ) :
  N = 100000 * a + 10000 * b + 1000 * c + 100 * a + 10 * b + c →
  a < 10 → b < 10 → c < 10 →
  11 ∣ N :=
by sorry

end NUMINAMATH_CALUDE_six_digit_repeating_divisible_by_11_l790_79064


namespace NUMINAMATH_CALUDE_all_points_in_triangle_satisfy_condition_probability_a_minus_b_positive_is_zero_l790_79067

-- Define the triangle
def triangle : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.1 ≥ 0 ∧ p.1 ≤ 4 ∧ p.2 ≥ 0 ∧ 4 * p.2 ≤ 10 * p.1}

-- Theorem statement
theorem all_points_in_triangle_satisfy_condition :
  ∀ p : ℝ × ℝ, p ∈ triangle → p.1 - p.2 ≤ 0 :=
by
  sorry

-- Probability statement
theorem probability_a_minus_b_positive_is_zero :
  ∀ p : ℝ × ℝ, p ∈ triangle → (p.1 - p.2 > 0) = false :=
by
  sorry

end NUMINAMATH_CALUDE_all_points_in_triangle_satisfy_condition_probability_a_minus_b_positive_is_zero_l790_79067


namespace NUMINAMATH_CALUDE_largest_solution_of_equation_l790_79072

theorem largest_solution_of_equation (a : ℝ) : 
  (3 * a + 4) * (a - 2) = 8 * a → a ≤ 4 := by
  sorry

end NUMINAMATH_CALUDE_largest_solution_of_equation_l790_79072


namespace NUMINAMATH_CALUDE_magnitude_of_z_is_one_l790_79026

/-- Given a complex number z defined as z = (1-i)/(1+i) + 2i, prove that its magnitude |z| is equal to 1 -/
theorem magnitude_of_z_is_one : 
  let z : ℂ := (1 - Complex.I) / (1 + Complex.I) + 2 * Complex.I
  Complex.abs z = 1 := by
  sorry

end NUMINAMATH_CALUDE_magnitude_of_z_is_one_l790_79026


namespace NUMINAMATH_CALUDE_equal_savings_time_l790_79051

def sara_initial_savings : ℕ := 4100
def sara_weekly_savings : ℕ := 10
def jim_weekly_savings : ℕ := 15

theorem equal_savings_time : 
  ∃ w : ℕ, w = 820 ∧ 
  sara_initial_savings + sara_weekly_savings * w = jim_weekly_savings * w :=
by sorry

end NUMINAMATH_CALUDE_equal_savings_time_l790_79051


namespace NUMINAMATH_CALUDE_point_coordinates_sum_l790_79045

theorem point_coordinates_sum (X Y Z : ℝ × ℝ) : 
  (X.1 - Z.1) / (X.1 - Y.1) = 1/2 →
  (X.2 - Z.2) / (X.2 - Y.2) = 1/2 →
  (Z.1 - Y.1) / (X.1 - Y.1) = 1/2 →
  (Z.2 - Y.2) / (X.2 - Y.2) = 1/2 →
  Y = (2, 5) →
  Z = (1, -3) →
  X.1 + X.2 = -11 := by
sorry

end NUMINAMATH_CALUDE_point_coordinates_sum_l790_79045


namespace NUMINAMATH_CALUDE_unique_number_with_specific_divisors_l790_79096

theorem unique_number_with_specific_divisors : ∃! (N : ℕ),
  (5 ∣ N) ∧ (49 ∣ N) ∧ (Finset.card (Nat.divisors N) = 10) :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_unique_number_with_specific_divisors_l790_79096


namespace NUMINAMATH_CALUDE_perfect_square_count_l790_79039

theorem perfect_square_count : ∃ (S : Finset ℕ), 
  (∀ n ∈ S, n ≤ 1500 ∧ ∃ k : ℕ, 21 * n = k^2) ∧ 
  S.card = 8 := by
  sorry

end NUMINAMATH_CALUDE_perfect_square_count_l790_79039


namespace NUMINAMATH_CALUDE_thirtieth_term_of_sequence_l790_79070

def arithmetic_sequence (a₁ : ℚ) (d : ℚ) (n : ℕ) : ℚ := a₁ + (n - 1) * d

theorem thirtieth_term_of_sequence (a₁ a₁₃ : ℚ) (h₁ : a₁ = 10) (h₂ : a₁₃ = 50) :
  arithmetic_sequence a₁ ((a₁₃ - a₁) / 12) 30 = 100 := by
  sorry

end NUMINAMATH_CALUDE_thirtieth_term_of_sequence_l790_79070


namespace NUMINAMATH_CALUDE_truncated_pyramid_volume_l790_79002

/-- A regular truncated quadrilateral pyramid -/
structure TruncatedPyramid where
  upper_base : ℝ
  lower_base : ℝ

/-- The volume of a truncated pyramid -/
def volume (p : TruncatedPyramid) : ℝ := sorry

/-- A plane that divides the pyramid into two equal parts -/
structure DividingPlane where
  perpendicular_to_diagonal : Bool
  passes_through_upper_edge : Bool

theorem truncated_pyramid_volume
  (p : TruncatedPyramid)
  (d : DividingPlane)
  (h1 : p.upper_base = 1)
  (h2 : p.lower_base = 7)
  (h3 : d.perpendicular_to_diagonal = true)
  (h4 : d.passes_through_upper_edge = true)
  (h5 : ∃ (v : ℝ), volume { upper_base := p.upper_base, lower_base := v } = volume { upper_base := v, lower_base := p.lower_base }) :
  volume p = 38 / Real.sqrt 5 := by sorry

end NUMINAMATH_CALUDE_truncated_pyramid_volume_l790_79002


namespace NUMINAMATH_CALUDE_empty_solution_set_range_l790_79041

theorem empty_solution_set_range (a : ℝ) : 
  (∀ x : ℝ, |x - 1| + |x - 2| > a^2 + a + 1) → 
  -1 < a ∧ a < 0 := by
sorry

end NUMINAMATH_CALUDE_empty_solution_set_range_l790_79041


namespace NUMINAMATH_CALUDE_two_integers_sum_l790_79098

theorem two_integers_sum (a b : ℕ+) : 
  (a : ℤ) - (b : ℤ) = 3 → a * b = 63 → (a : ℤ) + (b : ℤ) = 17 := by
  sorry

end NUMINAMATH_CALUDE_two_integers_sum_l790_79098


namespace NUMINAMATH_CALUDE_fountain_water_after_25_days_l790_79014

def fountain_water_volume (initial_volume : ℝ) (evaporation_rate : ℝ) (rain_interval : ℕ) (rain_amount : ℝ) (days : ℕ) : ℝ :=
  let total_evaporation := evaporation_rate * days
  let rain_events := days / rain_interval
  let total_rain := rain_events * rain_amount
  initial_volume + total_rain - total_evaporation

theorem fountain_water_after_25_days :
  fountain_water_volume 120 0.8 5 5 25 = 125 := by sorry

end NUMINAMATH_CALUDE_fountain_water_after_25_days_l790_79014


namespace NUMINAMATH_CALUDE_jerrys_action_figures_l790_79020

theorem jerrys_action_figures (initial_figures : ℕ) : 
  initial_figures + 4 - 1 = 6 → initial_figures = 3 :=
by
  sorry

end NUMINAMATH_CALUDE_jerrys_action_figures_l790_79020


namespace NUMINAMATH_CALUDE_triangle_side_area_relation_l790_79031

/-- Given a triangle with altitudes m₁, m₂, m₃ to sides a, b, c respectively,
    prove the relation between sides and area. -/
theorem triangle_side_area_relation (m₁ m₂ m₃ a b c S : ℝ) 
  (h₁ : m₁ = 20)
  (h₂ : m₂ = 24)
  (h₃ : m₃ = 30)
  (ha : S = a * m₁ / 2)
  (hb : S = b * m₂ / 2)
  (hc : S = c * m₃ / 2) :
  (a / b = 6 / 5 ∧ b / c = 5 / 4) ∧ S = 10 * a ∧ S = 12 * b ∧ S = 15 * c := by
  sorry

end NUMINAMATH_CALUDE_triangle_side_area_relation_l790_79031


namespace NUMINAMATH_CALUDE_num_purchasing_methods_eq_seven_l790_79018

/-- The number of purchasing methods for equipment types A and B -/
def num_purchasing_methods : ℕ :=
  (Finset.filter (fun p : ℕ × ℕ =>
    let (x, y) := p
    600000 * x + 700000 * y ≤ 5000000 ∧
    x ≥ 3 ∧
    y ≥ 2
  ) (Finset.product (Finset.range 10) (Finset.range 10))).card

/-- Theorem stating that the number of purchasing methods is 7 -/
theorem num_purchasing_methods_eq_seven :
  num_purchasing_methods = 7 := by sorry

end NUMINAMATH_CALUDE_num_purchasing_methods_eq_seven_l790_79018


namespace NUMINAMATH_CALUDE_flour_needed_for_butter_l790_79087

/-- Given a recipe with a ratio of butter to flour, calculate the amount of flour needed for a given amount of butter -/
theorem flour_needed_for_butter 
  (original_butter : ℚ) 
  (original_flour : ℚ) 
  (used_butter : ℚ) 
  (h1 : original_butter > 0) 
  (h2 : original_flour > 0) 
  (h3 : used_butter > 0) : 
  (used_butter / original_butter) * original_flour = 30 := by
  sorry

#check flour_needed_for_butter 2 5 12

end NUMINAMATH_CALUDE_flour_needed_for_butter_l790_79087


namespace NUMINAMATH_CALUDE_symmetric_points_sum_power_l790_79061

/-- Given two points P₁ and P₂ symmetric with respect to the x-axis, 
    prove that (a+b)^2023 = -1 --/
theorem symmetric_points_sum_power (a b : ℝ) : 
  (a - 1 = 2) →  -- x-coordinates are equal
  (5 = -(b - 1)) →  -- y-coordinates are opposite
  (a + b)^2023 = -1 := by
  sorry

end NUMINAMATH_CALUDE_symmetric_points_sum_power_l790_79061


namespace NUMINAMATH_CALUDE_three_digit_cube_divisible_by_eight_l790_79059

theorem three_digit_cube_divisible_by_eight :
  ∃! n : ℕ, 100 ≤ n ∧ n ≤ 999 ∧ ∃ m : ℕ, n = m^3 ∧ 8 ∣ n := by sorry

end NUMINAMATH_CALUDE_three_digit_cube_divisible_by_eight_l790_79059


namespace NUMINAMATH_CALUDE_symmetric_function_max_value_l790_79094

/-- Given a function f(x) = (1-x^2)(x^2 + ax + b) that is symmetric about x = -2,
    prove that its maximum value is 16. -/
theorem symmetric_function_max_value
  (f : ℝ → ℝ)
  (a b : ℝ)
  (h_def : ∀ x, f x = (1 - x^2) * (x^2 + a*x + b))
  (h_sym : ∀ x, f (x + (-2)) = f ((-2) - x)) :
  ∃ x, f x = 16 ∧ ∀ y, f y ≤ 16 := by
  sorry

end NUMINAMATH_CALUDE_symmetric_function_max_value_l790_79094


namespace NUMINAMATH_CALUDE_division_remainder_problem_l790_79081

theorem division_remainder_problem (P D Q R D' Q' R' C : ℕ) 
  (h1 : P = Q * D + R)
  (h2 : Q = Q' * D' + R')
  : P % (D * D') = D * R' + R + C := by
  sorry

end NUMINAMATH_CALUDE_division_remainder_problem_l790_79081


namespace NUMINAMATH_CALUDE_M_subset_N_l790_79007

def M : Set ℚ := {x | ∃ k : ℤ, x = k / 2 + 1 / 4}
def N : Set ℚ := {x | ∃ k : ℤ, x = k / 4 + 1 / 2}

theorem M_subset_N : M ⊆ N := by sorry

end NUMINAMATH_CALUDE_M_subset_N_l790_79007


namespace NUMINAMATH_CALUDE_checkerboard_covering_l790_79077

/-- Represents an L-shaped piece that covers 3 squares -/
structure LPiece

/-- Represents a checkerboard -/
structure Checkerboard (n : ℕ) where
  size : n % 2 = 1  -- n is odd
  black_corners : Bool

/-- Defines a covering of a checkerboard with L-shaped pieces -/
def Covering (n : ℕ) := Checkerboard n → List LPiece

/-- Checks if a covering is valid (covers all black squares) -/
def is_valid_covering (n : ℕ) (c : Covering n) : Prop := sorry

theorem checkerboard_covering (n : ℕ) :
  n % 2 = 1 →  -- n is odd
  (∃ (c : Covering n), is_valid_covering n c) ↔ n ≥ 7 := by sorry

end NUMINAMATH_CALUDE_checkerboard_covering_l790_79077


namespace NUMINAMATH_CALUDE_travel_speed_l790_79068

/-- Given a distance of 195 km and a travel time of 3 hours, prove that the speed is 65 km/h -/
theorem travel_speed (distance : ℝ) (time : ℝ) (speed : ℝ) : 
  distance = 195 ∧ time = 3 ∧ speed = distance / time → speed = 65 := by
  sorry

end NUMINAMATH_CALUDE_travel_speed_l790_79068


namespace NUMINAMATH_CALUDE_bell_pepper_cost_l790_79089

/-- The cost of a single bell pepper given the total cost of ingredients for tacos -/
theorem bell_pepper_cost (taco_shells_cost meat_price_per_pound meat_pounds total_spent bell_pepper_count : ℚ) :
  taco_shells_cost = 5 →
  bell_pepper_count = 4 →
  meat_pounds = 2 →
  meat_price_per_pound = 3 →
  total_spent = 17 →
  (total_spent - (taco_shells_cost + meat_price_per_pound * meat_pounds)) / bell_pepper_count = 3/2 := by
sorry

end NUMINAMATH_CALUDE_bell_pepper_cost_l790_79089


namespace NUMINAMATH_CALUDE_right_triangle_increased_sides_is_acute_l790_79073

/-- 
Given a right-angled triangle with sides a, b, and c (where c is the hypotenuse),
and a positive real number d, prove that the triangle with sides (a+d), (b+d), and (c+d)
is an acute-angled triangle.
-/
theorem right_triangle_increased_sides_is_acute 
  (a b c d : ℝ) 
  (h_right : a^2 + b^2 = c^2)  -- Original triangle is right-angled
  (h_pos : d > 0)              -- Increase is positive
  : (a+d)^2 + (b+d)^2 > (c+d)^2 := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_increased_sides_is_acute_l790_79073


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l790_79075

/-- Given an arithmetic sequence {a_n} with a_1 = -2014 and S_n as the sum of first n terms,
    if S_{2012}/2012 - S_{10}/10 = 2002, then S_{2016} = 2016 -/
theorem arithmetic_sequence_sum (a : ℕ → ℤ) (S : ℕ → ℤ) :
  (∀ n, S n = n * a 1 + n * (n - 1) / 2 * (a 2 - a 1)) →  -- Definition of S_n
  (a 1 = -2014) →                                         -- First term condition
  (S 2012 / 2012 - S 10 / 10 = 2002) →                    -- Given condition
  (S 2016 = 2016) :=                                      -- Conclusion to prove
by sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l790_79075


namespace NUMINAMATH_CALUDE_bacteria_growth_proof_l790_79071

/-- The increase in bacteria population given initial and final counts -/
def bacteria_increase (initial final : ℕ) : ℕ :=
  final - initial

/-- Theorem stating the increase in bacteria population for the given scenario -/
theorem bacteria_growth_proof (initial final : ℕ) 
  (h1 : initial = 600) 
  (h2 : final = 8917) : 
  bacteria_increase initial final = 8317 := by
  sorry

end NUMINAMATH_CALUDE_bacteria_growth_proof_l790_79071


namespace NUMINAMATH_CALUDE_complement_of_union_l790_79025

def U : Set Nat := {1, 2, 3, 4, 5, 6}
def A : Set Nat := {4, 5}
def B : Set Nat := {3, 4}

theorem complement_of_union :
  (U \ (A ∪ B)) = {1, 2, 6} := by sorry

end NUMINAMATH_CALUDE_complement_of_union_l790_79025


namespace NUMINAMATH_CALUDE_congruence_problem_l790_79005

theorem congruence_problem (n : ℤ) : 
  3 ≤ n ∧ n ≤ 9 ∧ n ≡ 12345 [ZMOD 6] → n = 3 ∨ n = 9 := by
sorry

end NUMINAMATH_CALUDE_congruence_problem_l790_79005


namespace NUMINAMATH_CALUDE_polynomial_coefficients_sum_l790_79066

theorem polynomial_coefficients_sum (a₀ a₁ a₂ a₃ a₄ a₅ a₆ : ℝ) :
  (∀ x : ℝ, (2*x - 1)^6 = a₀ + a₁*x + a₂*x^2 + a₃*x^3 + a₄*x^4 + a₅*x^5 + a₆*x^6) →
  (a₀ + a₁ + a₂ + a₃ + a₄ + a₅ + a₆ = 1 ∧ a₀ + a₂ + a₄ + a₆ = 365) :=
by sorry

end NUMINAMATH_CALUDE_polynomial_coefficients_sum_l790_79066


namespace NUMINAMATH_CALUDE_garden_walkway_area_l790_79027

/-- Calculates the total area of walkways in a garden with specified dimensions and layout. -/
def walkway_area (rows : ℕ) (columns : ℕ) (bed_width : ℕ) (bed_height : ℕ) (walkway_width : ℕ) : ℕ :=
  let total_width := columns * bed_width + (columns + 1) * walkway_width
  let total_height := rows * bed_height + (rows + 1) * walkway_width
  let total_area := total_width * total_height
  let bed_area := rows * columns * bed_width * bed_height
  total_area - bed_area

theorem garden_walkway_area :
  walkway_area 4 3 8 3 2 = 416 := by
  sorry

end NUMINAMATH_CALUDE_garden_walkway_area_l790_79027


namespace NUMINAMATH_CALUDE_train_distance_l790_79003

theorem train_distance (v_ab v_ba : ℝ) (t_diff : ℝ) (h1 : v_ab = 160)
    (h2 : v_ba = 120) (h3 : t_diff = 1) : ∃ D : ℝ,
  D / v_ba = D / v_ab + t_diff ∧ D = 480 := by
  sorry

end NUMINAMATH_CALUDE_train_distance_l790_79003


namespace NUMINAMATH_CALUDE_vector_subtraction_l790_79083

def a : Fin 2 → ℝ := ![3, 2]
def b : Fin 2 → ℝ := ![0, -1]

theorem vector_subtraction :
  (3 • b - a) = ![(-3 : ℝ), -5] := by sorry

end NUMINAMATH_CALUDE_vector_subtraction_l790_79083


namespace NUMINAMATH_CALUDE_computer_multiplications_l790_79056

/-- Represents the number of multiplications a computer can perform per second -/
def multiplications_per_second : ℕ := 15000

/-- Represents the number of seconds in an hour -/
def seconds_per_hour : ℕ := 3600

/-- Represents the number of hours we're calculating for -/
def hours : ℕ := 2

/-- Theorem stating that the computer will perform 108 million multiplications in two hours -/
theorem computer_multiplications :
  multiplications_per_second * seconds_per_hour * hours = 108000000 := by
  sorry

#eval multiplications_per_second * seconds_per_hour * hours

end NUMINAMATH_CALUDE_computer_multiplications_l790_79056


namespace NUMINAMATH_CALUDE_number_properties_l790_79085

def number : ℕ := 52300600

-- Define a function to get the digit at a specific position
def digit_at_position (n : ℕ) (pos : ℕ) : ℕ :=
  (n / (10 ^ (pos - 1))) % 10

-- Define a function to get the value represented by a digit at a specific position
def value_at_position (n : ℕ) (pos : ℕ) : ℕ :=
  (digit_at_position n pos) * (10 ^ (pos - 1))

-- Define a function to convert a number to its word representation
def number_to_words (n : ℕ) : String :=
  sorry -- Implementation details omitted

theorem number_properties :
  (digit_at_position number 8 = 2) ∧
  (value_at_position number 8 = 20000000) ∧
  (digit_at_position number 9 = 5) ∧
  (value_at_position number 9 = 500000000) ∧
  (number_to_words number = "five hundred twenty-three million six hundred") := by
  sorry

end NUMINAMATH_CALUDE_number_properties_l790_79085


namespace NUMINAMATH_CALUDE_expression_evaluation_l790_79038

theorem expression_evaluation :
  let x : ℚ := 4 / 7
  let y : ℚ := 8 / 5
  (7 * x + 5 * y + 4) / (60 * x * y + 5) = 560 / 559 :=
by sorry

end NUMINAMATH_CALUDE_expression_evaluation_l790_79038


namespace NUMINAMATH_CALUDE_dutch_americans_window_seats_fraction_l790_79086

/-- The fraction of Dutch Americans who got window seats on William's bus -/
theorem dutch_americans_window_seats_fraction
  (total_people : ℕ)
  (dutch_fraction : ℚ)
  (dutch_american_fraction : ℚ)
  (dutch_americans_with_window_seats : ℕ)
  (h1 : total_people = 90)
  (h2 : dutch_fraction = 3 / 5)
  (h3 : dutch_american_fraction = 1 / 2)
  (h4 : dutch_americans_with_window_seats = 9) :
  (dutch_americans_with_window_seats : ℚ) / (dutch_fraction * dutch_american_fraction * total_people) = 1 / 3 := by
  sorry

#check dutch_americans_window_seats_fraction

end NUMINAMATH_CALUDE_dutch_americans_window_seats_fraction_l790_79086


namespace NUMINAMATH_CALUDE_unique_solution_l790_79076

/-- The factorial function -/
def factorial (n : ℕ) : ℕ := if n = 0 then 1 else n * factorial (n - 1)

/-- The equation that n must satisfy -/
def satisfies_equation (n : ℕ) : Prop :=
  factorial (n + 1) + factorial (n + 3) = factorial n * 1540

theorem unique_solution :
  ∃! n : ℕ, n > 0 ∧ satisfies_equation n ∧ n = 10 := by sorry

end NUMINAMATH_CALUDE_unique_solution_l790_79076


namespace NUMINAMATH_CALUDE_hyperbola_eccentricity_l790_79060

/-- Hyperbola with foci and a special point -/
structure Hyperbola where
  a : ℝ
  b : ℝ
  F₁ : ℝ × ℝ  -- Left focus
  F₂ : ℝ × ℝ  -- Right focus
  P : ℝ × ℝ   -- Special point on the right branch
  h₁ : a > b
  h₂ : b > 0
  h₃ : F₁.1 < 0 ∧ F₁.2 = 0  -- Left focus on negative x-axis
  h₄ : F₂.1 > 0 ∧ F₂.2 = 0  -- Right focus on positive x-axis
  h₅ : P.1 > 0  -- P is on the right branch
  h₆ : P.1^2 / a^2 - P.2^2 / b^2 = 1  -- P satisfies hyperbola equation
  h₇ : (P.1 + F₂.1) * (P.1 - F₂.1) + P.2 * P.2 = 0  -- Dot product condition
  h₈ : (P.1 - F₁.1)^2 + P.2^2 = 4 * ((P.1 - F₂.1)^2 + P.2^2)  -- Distance condition

/-- The eccentricity of a hyperbola with the given properties is √5 -/
theorem hyperbola_eccentricity (h : Hyperbola) : 
  Real.sqrt ((h.F₂.1 - h.F₁.1)^2 / (4 * h.a^2)) = Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_hyperbola_eccentricity_l790_79060


namespace NUMINAMATH_CALUDE_philips_farm_animals_l790_79034

/-- Represents the number of animals on Philip's farm -/
structure FarmAnimals where
  cows : ℕ
  ducks : ℕ
  horses : ℕ
  pigs : ℕ
  chickens : ℕ

/-- Calculates the total number of animals on the farm -/
def total_animals (farm : FarmAnimals) : ℕ :=
  farm.cows + farm.ducks + farm.horses + farm.pigs + farm.chickens

/-- Theorem stating the total number of animals on Philip's farm -/
theorem philips_farm_animals :
  ∃ (farm : FarmAnimals),
    farm.cows = 20 ∧
    farm.ducks = farm.cows + farm.cows / 2 ∧
    farm.horses = (farm.cows + farm.ducks) / 5 ∧
    farm.pigs = (farm.cows + farm.ducks + farm.horses) / 5 ∧
    farm.chickens = 3 * (farm.cows - farm.horses) ∧
    total_animals farm = 102 := by
  sorry


end NUMINAMATH_CALUDE_philips_farm_animals_l790_79034


namespace NUMINAMATH_CALUDE_multiplicative_inverse_modulo_l790_79006

def A : Nat := 111111
def B : Nat := 142857
def M : Nat := 1000000

theorem multiplicative_inverse_modulo :
  (63 * (A * B)) % M = 1 := by sorry

end NUMINAMATH_CALUDE_multiplicative_inverse_modulo_l790_79006


namespace NUMINAMATH_CALUDE_exists_valid_assignment_l790_79042

/-- Represents a rectangular parallelepiped with dimensions a, b, and c -/
structure Parallelepiped where
  a : ℕ
  b : ℕ
  c : ℕ

/-- Represents an assignment of numbers to the unit squares on the surface of a parallelepiped -/
def SurfaceAssignment (p : Parallelepiped) := (ℕ × ℕ × ℕ) → ℝ

/-- Calculates the sum of numbers in a 1-width band surrounding the parallelepiped -/
def bandSum (p : Parallelepiped) (assignment : SurfaceAssignment p) : ℝ := sorry

/-- Theorem stating the existence of a valid assignment for a 3 × 4 × 5 parallelepiped -/
theorem exists_valid_assignment :
  ∃ (assignment : SurfaceAssignment ⟨3, 4, 5⟩),
    bandSum ⟨3, 4, 5⟩ assignment = 120 := by sorry

end NUMINAMATH_CALUDE_exists_valid_assignment_l790_79042


namespace NUMINAMATH_CALUDE_log_ratio_squared_l790_79009

theorem log_ratio_squared (x y : ℝ) 
  (hx_pos : x > 0) (hy_pos : y > 0) 
  (hx_neq_one : x ≠ 1) (hy_neq_one : y ≠ 1) 
  (h_log : Real.log x / Real.log 3 = Real.log 81 / Real.log y) 
  (h_prod : x * y^2 = 243) : 
  (Real.log (x/y) / Real.log 3)^2 = 49/36 := by
sorry

end NUMINAMATH_CALUDE_log_ratio_squared_l790_79009


namespace NUMINAMATH_CALUDE_total_wage_calculation_l790_79054

/-- Represents the number of days it takes for a worker to complete the job alone -/
structure WorkerSpeed :=
  (days : ℕ)

/-- Calculates the daily work rate of a worker -/
def dailyRate (w : WorkerSpeed) : ℚ :=
  1 / w.days

/-- Represents the wage distribution between two workers -/
structure WageDistribution :=
  (worker_a : ℚ)
  (total : ℚ)

theorem total_wage_calculation 
  (speed_a : WorkerSpeed)
  (speed_b : WorkerSpeed)
  (wage_dist : WageDistribution)
  (h1 : speed_a.days = 10)
  (h2 : speed_b.days = 15)
  (h3 : wage_dist.worker_a = 1980)
  : wage_dist.total = 3300 :=
sorry

end NUMINAMATH_CALUDE_total_wage_calculation_l790_79054


namespace NUMINAMATH_CALUDE_fourth_day_pages_l790_79074

/-- Represents the number of pages read each day -/
structure DailyPages where
  day1 : ℕ
  day2 : ℕ
  day3 : ℕ
  day4 : ℕ

/-- Represents the conditions of the book reading problem -/
structure BookReading where
  totalPages : ℕ
  dailyPages : DailyPages
  day1Condition : dailyPages.day1 = 63
  day2Condition : dailyPages.day2 = 2 * dailyPages.day1
  day3Condition : dailyPages.day3 = dailyPages.day2 + 10
  totalCondition : totalPages = dailyPages.day1 + dailyPages.day2 + dailyPages.day3 + dailyPages.day4

/-- Theorem stating that given the conditions, the number of pages read on the fourth day is 29 -/
theorem fourth_day_pages (br : BookReading) (h : br.totalPages = 354) : br.dailyPages.day4 = 29 := by
  sorry

end NUMINAMATH_CALUDE_fourth_day_pages_l790_79074


namespace NUMINAMATH_CALUDE_ice_cream_theorem_l790_79069

def num_flavors : ℕ := 4
def num_scoops : ℕ := 4

def ice_cream_combinations (n m : ℕ) : ℕ :=
  Nat.choose (n + m - 1) (n - 1)

theorem ice_cream_theorem :
  ice_cream_combinations num_flavors num_scoops = 35 := by
  sorry

end NUMINAMATH_CALUDE_ice_cream_theorem_l790_79069


namespace NUMINAMATH_CALUDE_expansion_coefficients_l790_79037

def S (n : ℕ) : ℚ :=
  (n + 1) / (2 * Nat.factorial (n - 1))

def T (n : ℕ) : ℚ := sorry

theorem expansion_coefficients (n : ℕ) (h : n ≥ 2) :
  S n = (n + 1) / (2 * Nat.factorial (n - 1)) ∧
  T n / S n = (1/4 : ℚ) * n^2 - (1/12 : ℚ) * n - (1/6 : ℚ) :=
by sorry

end NUMINAMATH_CALUDE_expansion_coefficients_l790_79037


namespace NUMINAMATH_CALUDE_fraction_simplification_l790_79053

theorem fraction_simplification (a b : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hab : a ≠ b) :
  (a^2 - b^2) / (a * b) - (a * b - b^2) / (a * b - a^2) = a / b :=
by sorry

end NUMINAMATH_CALUDE_fraction_simplification_l790_79053


namespace NUMINAMATH_CALUDE_circle_center_l790_79004

def circle_equation (x y : ℝ) : Prop := (x - 1)^2 + (y - 2)^2 = 5

theorem circle_center : 
  ∃ (h k : ℝ), (∀ (x y : ℝ), circle_equation x y ↔ (x - h)^2 + (y - k)^2 = 5) ∧ h = 1 ∧ k = 2 :=
sorry

end NUMINAMATH_CALUDE_circle_center_l790_79004
