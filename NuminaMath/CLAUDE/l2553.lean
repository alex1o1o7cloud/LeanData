import Mathlib

namespace NUMINAMATH_CALUDE_floor_length_approx_l2553_255340

/-- Represents the dimensions and painting cost of a rectangular floor. -/
structure RectangularFloor where
  breadth : ℝ
  length : ℝ
  paintingCost : ℝ
  paintingRate : ℝ

/-- The length of the floor is 200% more than the breadth. -/
def lengthCondition (floor : RectangularFloor) : Prop :=
  floor.length = 3 * floor.breadth

/-- The total cost to paint the floor is 624 Rs. -/
def totalCostCondition (floor : RectangularFloor) : Prop :=
  floor.paintingCost = 624

/-- The painting rate is 4 Rs per square meter. -/
def paintingRateCondition (floor : RectangularFloor) : Prop :=
  floor.paintingRate = 4

/-- The theorem stating that the length of the floor is approximately 21.63 meters. -/
theorem floor_length_approx (floor : RectangularFloor) 
  (h1 : lengthCondition floor)
  (h2 : totalCostCondition floor)
  (h3 : paintingRateCondition floor) :
  ∃ ε > 0, |floor.length - 21.63| < ε :=
sorry

end NUMINAMATH_CALUDE_floor_length_approx_l2553_255340


namespace NUMINAMATH_CALUDE_horner_method_v₄_l2553_255386

-- Define the polynomial coefficients
def a₀ : ℝ := 12
def a₁ : ℝ := 35
def a₂ : ℝ := -8
def a₃ : ℝ := 79
def a₄ : ℝ := 6
def a₅ : ℝ := 5
def a₆ : ℝ := 3

-- Define x
def x : ℝ := -4

-- Define v₄ using Horner's method
def v₄ : ℝ := ((((a₆ * x + a₅) * x + a₄) * x + a₃) * x + a₂) * x + a₁

-- Theorem statement
theorem horner_method_v₄ : v₄ = 220 := by
  sorry

end NUMINAMATH_CALUDE_horner_method_v₄_l2553_255386


namespace NUMINAMATH_CALUDE_circle_symmetry_l2553_255350

-- Define the original circle C
def circle_C (x y : ℝ) : Prop := x^2 + y^2 - 2*x - 2*y + 1 = 0

-- Define the line l
def line_l (x y : ℝ) : Prop := x - y = 2

-- Define the symmetric circle
def symmetric_circle (x y : ℝ) : Prop := (x-3)^2 + (y+1)^2 = 1

-- Theorem statement
theorem circle_symmetry :
  ∀ x y : ℝ, 
  (∃ x₀ y₀ : ℝ, circle_C x₀ y₀ ∧ 
    (x + x₀)/2 - (y + y₀)/2 = 2 ∧ 
    (y - y₀)/(x - x₀) = -1) →
  symmetric_circle x y :=
sorry

end NUMINAMATH_CALUDE_circle_symmetry_l2553_255350


namespace NUMINAMATH_CALUDE_q_satisfies_conditions_l2553_255317

/-- A quadratic polynomial that satisfies specific conditions -/
def q (x : ℝ) : ℝ := x^2 + 1

/-- Theorem stating that q satisfies the given conditions -/
theorem q_satisfies_conditions :
  q (-2) = 5 ∧ q 1 = 2 ∧ q 3 = 10 := by
  sorry

#eval q (-2)
#eval q 1
#eval q 3

end NUMINAMATH_CALUDE_q_satisfies_conditions_l2553_255317


namespace NUMINAMATH_CALUDE_expression_evaluation_l2553_255392

theorem expression_evaluation :
  let x : ℤ := -2
  (2 * x + 1) * (x - 2) - (2 - x)^2 = -8 := by sorry

end NUMINAMATH_CALUDE_expression_evaluation_l2553_255392


namespace NUMINAMATH_CALUDE_arithmetic_expression_equality_l2553_255326

theorem arithmetic_expression_equality : 9 - 3 / (1 / 3) + 3 = 3 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_expression_equality_l2553_255326


namespace NUMINAMATH_CALUDE_tileB_smallest_unique_p_l2553_255364

/-- Represents a rectangular tile with four labeled sides -/
structure Tile where
  p : ℤ
  q : ℤ
  r : ℤ
  s : ℤ

/-- The set of all tiles -/
def tiles : Finset Tile := sorry

/-- Tile A -/
def tileA : Tile := { p := 5, q := 2, r := 8, s := 11 }

/-- Tile B -/
def tileB : Tile := { p := 2, q := 1, r := 4, s := 7 }

/-- Tile C -/
def tileC : Tile := { p := 4, q := 9, r := 6, s := 3 }

/-- Tile D -/
def tileD : Tile := { p := 10, q := 6, r := 5, s := 9 }

/-- Tile E -/
def tileE : Tile := { p := 11, q := 3, r := 7, s := 0 }

/-- Function to check if a value is unique among all tiles -/
def isUnique (t : Tile) (f : Tile → ℤ) : Prop :=
  ∀ t' ∈ tiles, t' ≠ t → f t' ≠ f t

/-- Theorem: Tile B has the smallest unique p value -/
theorem tileB_smallest_unique_p :
  isUnique tileB Tile.p ∧ 
  ∀ t ∈ tiles, isUnique t Tile.p → tileB.p ≤ t.p :=
sorry

end NUMINAMATH_CALUDE_tileB_smallest_unique_p_l2553_255364


namespace NUMINAMATH_CALUDE_james_adoption_payment_l2553_255316

/-- The total amount James pays for adopting a puppy and a kitten -/
def jamesPayment (puppyFee kittenFee : ℚ) (multiPetDiscount : ℚ) 
  (friendPuppyContribution friendKittenContribution : ℚ) : ℚ :=
  let totalFee := puppyFee + kittenFee
  let discountedFee := totalFee * (1 - multiPetDiscount)
  let friendContributions := puppyFee * friendPuppyContribution + kittenFee * friendKittenContribution
  discountedFee - friendContributions

/-- Theorem stating that James pays $242.50 for adopting a puppy and a kitten -/
theorem james_adoption_payment :
  jamesPayment 200 150 (1/10) (1/4) (3/20) = 485/2 :=
by sorry

end NUMINAMATH_CALUDE_james_adoption_payment_l2553_255316


namespace NUMINAMATH_CALUDE_max_ab_value_l2553_255359

theorem max_ab_value (a b c : ℝ) (h1 : a + b + c = 4) (h2 : 3*a + 2*b - c = 0) :
  ∀ x y : ℝ, x + y + c = 4 → 3*x + 2*y - c = 0 → x*y ≤ a*b ∧ a*b = 1/3 :=
sorry

end NUMINAMATH_CALUDE_max_ab_value_l2553_255359


namespace NUMINAMATH_CALUDE_latest_start_time_is_10am_l2553_255333

-- Define the number of turkeys
def num_turkeys : ℕ := 2

-- Define the weight of each turkey in pounds
def turkey_weight : ℕ := 16

-- Define the roasting time per pound in minutes
def roasting_time_per_pound : ℕ := 15

-- Define the dinner time (18:00 in 24-hour format)
def dinner_time : ℕ := 18 * 60

-- Define the function to calculate the total roasting time in minutes
def total_roasting_time : ℕ := num_turkeys * turkey_weight * roasting_time_per_pound

-- Define the function to calculate the latest start time in minutes after midnight
def latest_start_time : ℕ := dinner_time - total_roasting_time

-- Theorem stating that the latest start time is 10:00 am (600 minutes after midnight)
theorem latest_start_time_is_10am : latest_start_time = 600 := by
  sorry

end NUMINAMATH_CALUDE_latest_start_time_is_10am_l2553_255333


namespace NUMINAMATH_CALUDE_sum_of_roots_l2553_255338

theorem sum_of_roots (a b : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) 
  (h : ∀ x : ℝ, x^2 - a*x + 3*b = 0 ↔ x = a ∨ x = b) : 
  a + b = a :=
sorry

end NUMINAMATH_CALUDE_sum_of_roots_l2553_255338


namespace NUMINAMATH_CALUDE_hockey_team_size_l2553_255349

/-- Calculates the total number of players on a hockey team given specific conditions -/
theorem hockey_team_size 
  (percent_boys : ℚ)
  (num_junior_girls : ℕ)
  (h1 : percent_boys = 60 / 100)
  (h2 : num_junior_girls = 10) : 
  (2 * num_junior_girls : ℚ) / (1 - percent_boys) = 50 := by
  sorry

end NUMINAMATH_CALUDE_hockey_team_size_l2553_255349


namespace NUMINAMATH_CALUDE_gary_stickers_l2553_255384

theorem gary_stickers (initial_stickers : ℕ) : 
  (initial_stickers : ℚ) * (2/3) * (3/4) = 36 → initial_stickers = 72 := by
  sorry

end NUMINAMATH_CALUDE_gary_stickers_l2553_255384


namespace NUMINAMATH_CALUDE_society_coleaders_selection_l2553_255377

theorem society_coleaders_selection (n : ℕ) (k : ℕ) (h1 : n = 20) (h2 : k = 2) :
  Nat.choose n k = 190 := by
  sorry

end NUMINAMATH_CALUDE_society_coleaders_selection_l2553_255377


namespace NUMINAMATH_CALUDE_sixth_student_stickers_l2553_255354

def sticker_sequence (n : ℕ) : ℕ :=
  29 + 6 * (n - 1)

theorem sixth_student_stickers : sticker_sequence 6 = 59 := by
  sorry

end NUMINAMATH_CALUDE_sixth_student_stickers_l2553_255354


namespace NUMINAMATH_CALUDE_sector_max_area_l2553_255370

/-- Given a sector with circumference 30, its area is maximized when the radius is 15/2 and the central angle is 2. -/
theorem sector_max_area (R α : ℝ) : 
  R + R + (α * R) = 30 →  -- circumference condition
  (∀ R' α' : ℝ, R' + R' + (α' * R') = 30 → 
    (1/2) * α * R^2 ≥ (1/2) * α' * R'^2) →  -- area is maximized
  R = 15/2 ∧ α = 2 := by
sorry


end NUMINAMATH_CALUDE_sector_max_area_l2553_255370


namespace NUMINAMATH_CALUDE_distributions_five_balls_three_boxes_l2553_255395

/-- The number of ways to distribute n distinguishable balls into k distinguishable boxes -/
def total_distributions (n k : ℕ) : ℕ := k^n

/-- The number of ways to distribute n distinguishable balls into k distinguishable boxes
    such that at least one specific box is empty -/
def distributions_with_empty_box (n k : ℕ) : ℕ := (k - 1)^n

/-- The number of ways to distribute n distinguishable balls into k distinguishable boxes
    such that at least two specific boxes are empty -/
def distributions_with_two_empty_boxes (n k : ℕ) : ℕ := 1

/-- The number of ways to distribute n distinguishable balls into k distinguishable boxes
    such that no box remains empty -/
def distributions_no_empty_boxes (n k : ℕ) : ℕ :=
  total_distributions n k - 
  (k * distributions_with_empty_box n k) +
  (Nat.choose k 2 * distributions_with_two_empty_boxes n k)

theorem distributions_five_balls_three_boxes :
  distributions_no_empty_boxes 5 3 = 150 := by
  sorry

end NUMINAMATH_CALUDE_distributions_five_balls_three_boxes_l2553_255395


namespace NUMINAMATH_CALUDE_min_value_expression_l2553_255353

theorem min_value_expression (p q r : ℝ) (hp : p > 0) (hq : q > 0) (hr : r > 0) :
  (5 * r) / (3 * p + 2 * q) + (5 * p) / (2 * q + 3 * r) + (2 * q) / (p + r) ≥ 151 / 18 := by
  sorry

end NUMINAMATH_CALUDE_min_value_expression_l2553_255353


namespace NUMINAMATH_CALUDE_regular_dodecahedron_vertex_count_l2553_255302

/-- A regular dodecahedron has 20 vertices. -/
def regular_dodecahedron_vertices : ℕ := 20

/-- The number of vertices in a regular dodecahedron is 20. -/
theorem regular_dodecahedron_vertex_count : 
  regular_dodecahedron_vertices = 20 := by sorry

end NUMINAMATH_CALUDE_regular_dodecahedron_vertex_count_l2553_255302


namespace NUMINAMATH_CALUDE_proposition_equivalence_l2553_255345

theorem proposition_equivalence (A B : Set α) :
  (∀ x, x ∈ A → x ∈ B) ↔ (∀ x, x ∉ B → x ∉ A) := by
  sorry

end NUMINAMATH_CALUDE_proposition_equivalence_l2553_255345


namespace NUMINAMATH_CALUDE_journey_time_equation_l2553_255363

theorem journey_time_equation (x : ℝ) (h1 : x > 0) : 
  (240 / x - 240 / (1.5 * x) = 1) ↔ 
  (240 / x = 240 / (1.5 * x) + 1) := by
sorry

end NUMINAMATH_CALUDE_journey_time_equation_l2553_255363


namespace NUMINAMATH_CALUDE_vector_difference_magnitude_l2553_255300

theorem vector_difference_magnitude (a b : ℝ × ℝ) :
  ‖a‖ = 1 →
  ‖b‖ = 3 →
  a + b = (Real.sqrt 3, 1) →
  ‖a - b‖ = 4 := by
  sorry

end NUMINAMATH_CALUDE_vector_difference_magnitude_l2553_255300


namespace NUMINAMATH_CALUDE_total_students_count_l2553_255347

def third_grade : ℕ := 19

def fourth_grade : ℕ := 2 * third_grade

def second_grade_boys : ℕ := 10
def second_grade_girls : ℕ := 19

def total_students : ℕ := third_grade + fourth_grade + second_grade_boys + second_grade_girls

theorem total_students_count : total_students = 86 := by sorry

end NUMINAMATH_CALUDE_total_students_count_l2553_255347


namespace NUMINAMATH_CALUDE_prob_three_pass_min_students_scheme_A_l2553_255368

/-- Represents the two testing schemes -/
inductive Scheme
| A
| B

/-- Represents a student -/
structure Student where
  name : String
  scheme : Scheme

/-- Probability of passing for each scheme -/
def passProbability (s : Scheme) : ℚ :=
  match s with
  | Scheme.A => 2/3
  | Scheme.B => 1/2

/-- Group of students participating in the test -/
def testGroup : List Student := [
  ⟨"A", Scheme.A⟩, ⟨"B", Scheme.A⟩, ⟨"C", Scheme.A⟩,
  ⟨"D", Scheme.B⟩, ⟨"E", Scheme.B⟩
]

/-- Calculates the probability of exactly k students passing out of n students -/
def probExactlyKPass (n k : ℕ) (p : ℚ) : ℚ :=
  (Nat.choose n k : ℚ) * p^k * (1 - p)^(n - k)

/-- Theorem: The probability of exactly three students passing the test is 19/54 -/
theorem prob_three_pass :
  (probExactlyKPass 3 3 (passProbability Scheme.A) *
   probExactlyKPass 2 0 (passProbability Scheme.B)) +
  (probExactlyKPass 3 2 (passProbability Scheme.A) *
   probExactlyKPass 2 1 (passProbability Scheme.B)) +
  (probExactlyKPass 3 1 (passProbability Scheme.A) *
   probExactlyKPass 2 2 (passProbability Scheme.B)) = 19/54 := by
  sorry

/-- Expected number of passing students given n students choose scheme A -/
def expectedPass (n : ℕ) : ℚ := n * (passProbability Scheme.A) + (5 - n) * (passProbability Scheme.B)

/-- Theorem: The minimum number of students choosing scheme A for the expected number
    of passing students to be at least 3 is 3 -/
theorem min_students_scheme_A :
  (∀ m : ℕ, m < 3 → expectedPass m < 3) ∧
  expectedPass 3 ≥ 3 := by
  sorry

end NUMINAMATH_CALUDE_prob_three_pass_min_students_scheme_A_l2553_255368


namespace NUMINAMATH_CALUDE_rachels_homework_l2553_255382

/-- Rachel's homework problem -/
theorem rachels_homework (reading_homework : ℕ) (math_homework : ℕ) : 
  reading_homework = 2 → math_homework = reading_homework + 7 → math_homework = 9 :=
by
  sorry


end NUMINAMATH_CALUDE_rachels_homework_l2553_255382


namespace NUMINAMATH_CALUDE_expansion_has_four_nonzero_terms_l2553_255318

/-- The polynomial expression to be expanded -/
def polynomial_expression (x : ℝ) : ℝ :=
  (2*x + 5) * (3*x^2 + x + 6) - 4*(x^3 + 3*x^2 - 4*x + 1)

/-- The expanded form of the polynomial expression -/
def expanded_polynomial (x : ℝ) : ℝ :=
  2*x^3 + 5*x^2 + 33*x + 26

/-- Theorem stating that the expansion has exactly 4 nonzero terms -/
theorem expansion_has_four_nonzero_terms :
  ∃ (a b c d : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) (hd : d ≠ 0),
  ∀ x, polynomial_expression x = a*x^3 + b*x^2 + c*x + d :=
sorry

end NUMINAMATH_CALUDE_expansion_has_four_nonzero_terms_l2553_255318


namespace NUMINAMATH_CALUDE_unique_sums_count_l2553_255367

def X : Finset ℕ := {1, 4, 5, 7}
def Y : Finset ℕ := {3, 4, 6, 8}

theorem unique_sums_count : 
  Finset.card ((X.product Y).image (fun p => p.1 + p.2)) = 10 := by
  sorry

end NUMINAMATH_CALUDE_unique_sums_count_l2553_255367


namespace NUMINAMATH_CALUDE_laurie_kurt_difference_l2553_255394

/-- The number of marbles each person has -/
structure Marbles where
  dennis : ℕ
  kurt : ℕ
  laurie : ℕ

/-- The conditions of the problem -/
def marble_problem (m : Marbles) : Prop :=
  m.dennis = 70 ∧ 
  m.kurt = m.dennis - 45 ∧
  m.laurie = 37

/-- The theorem to prove -/
theorem laurie_kurt_difference (m : Marbles) 
  (h : marble_problem m) : m.laurie - m.kurt = 12 := by
  sorry

end NUMINAMATH_CALUDE_laurie_kurt_difference_l2553_255394


namespace NUMINAMATH_CALUDE_three_digit_palindrome_gcf_and_divisibility_l2553_255342

/-- Represents a three-digit palindrome -/
def ThreeDigitPalindrome : Type := { n : ℕ // ∃ (a b : ℕ), a ≠ 0 ∧ a < 10 ∧ b < 10 ∧ n = 102 * a + 10 * b }

/-- The set of all three-digit palindromes -/
def AllThreeDigitPalindromes : Set ℕ :=
  { n | ∃ (p : ThreeDigitPalindrome), n = p.val }

theorem three_digit_palindrome_gcf_and_divisibility :
  (∃ (g : ℕ), g > 0 ∧ 
    (∀ n ∈ AllThreeDigitPalindromes, g ∣ n) ∧
    (∀ d : ℕ, d > 0 → (∀ n ∈ AllThreeDigitPalindromes, d ∣ n) → d ∣ g)) ∧
  (∀ n ∈ AllThreeDigitPalindromes, 3 ∣ n) :=
by sorry

end NUMINAMATH_CALUDE_three_digit_palindrome_gcf_and_divisibility_l2553_255342


namespace NUMINAMATH_CALUDE_jordana_age_proof_l2553_255355

/-- Jennifer's age in ten years -/
def jennifer_future_age : ℕ := 30

/-- The number of years in the future we're considering -/
def years_ahead : ℕ := 10

/-- Jordana's age relative to Jennifer's in the future -/
def jordana_relative_age : ℕ := 3

/-- Jordana's current age -/
def jordana_current_age : ℕ := 80

theorem jordana_age_proof :
  jordana_current_age = jennifer_future_age * jordana_relative_age - years_ahead :=
by sorry

end NUMINAMATH_CALUDE_jordana_age_proof_l2553_255355


namespace NUMINAMATH_CALUDE_cross_section_distance_l2553_255391

/-- Represents a right hexagonal pyramid -/
structure RightHexagonalPyramid where
  /-- The distance from the apex to the base -/
  height : ℝ
  /-- The side length of the base hexagon -/
  baseSide : ℝ

/-- Represents a cross section of the pyramid -/
structure CrossSection where
  /-- The distance from the apex to the cross section -/
  distanceFromApex : ℝ
  /-- The area of the cross section -/
  area : ℝ

/-- The theorem to be proved -/
theorem cross_section_distance (pyramid : RightHexagonalPyramid) 
  (section1 section2 : CrossSection) :
  section1.area = 162 * Real.sqrt 3 →
  section2.area = 288 * Real.sqrt 3 →
  |section1.distanceFromApex - section2.distanceFromApex| = 6 →
  max section1.distanceFromApex section2.distanceFromApex = 24 :=
by sorry

end NUMINAMATH_CALUDE_cross_section_distance_l2553_255391


namespace NUMINAMATH_CALUDE_problem_stack_total_l2553_255311

/-- Represents a stack of logs -/
structure LogStack where
  topRow : ℕ
  bottomRow : ℕ

/-- Calculates the total number of logs in a stack -/
def totalLogs (stack : LogStack) : ℕ :=
  let n := stack.bottomRow - stack.topRow + 1
  n * (stack.topRow + stack.bottomRow) / 2

/-- The specific log stack described in the problem -/
def problemStack : LogStack := { topRow := 5, bottomRow := 15 }

/-- Theorem stating that the total number of logs in the problem stack is 110 -/
theorem problem_stack_total : totalLogs problemStack = 110 := by
  sorry

end NUMINAMATH_CALUDE_problem_stack_total_l2553_255311


namespace NUMINAMATH_CALUDE_participant_age_l2553_255308

/-- Represents the initial state of the lecture rooms -/
structure LectureRooms where
  room1_count : ℕ
  room1_avg_age : ℕ
  room2_count : ℕ
  room2_avg_age : ℕ

/-- Calculates the total age sum of all participants -/
def total_age_sum (rooms : LectureRooms) : ℕ :=
  rooms.room1_count * rooms.room1_avg_age + rooms.room2_count * rooms.room2_avg_age

/-- Calculates the total number of participants -/
def total_count (rooms : LectureRooms) : ℕ :=
  rooms.room1_count + rooms.room2_count

/-- Theorem stating the age of the participant who left -/
theorem participant_age (rooms : LectureRooms) 
  (h1 : rooms.room1_count = 8)
  (h2 : rooms.room1_avg_age = 20)
  (h3 : rooms.room2_count = 12)
  (h4 : rooms.room2_avg_age = 45)
  (h5 : (total_age_sum rooms - x) / (total_count rooms - 1) = (total_age_sum rooms) / (total_count rooms) + 1) :
  x = 16 :=
sorry


end NUMINAMATH_CALUDE_participant_age_l2553_255308


namespace NUMINAMATH_CALUDE_jonathan_took_45_oranges_l2553_255360

/-- The number of oranges Jonathan took -/
def oranges_taken (initial : ℕ) (remaining : ℕ) : ℕ :=
  initial - remaining

/-- Theorem stating that Jonathan took 45 oranges -/
theorem jonathan_took_45_oranges :
  oranges_taken 96 51 = 45 := by
  sorry

end NUMINAMATH_CALUDE_jonathan_took_45_oranges_l2553_255360


namespace NUMINAMATH_CALUDE_scale_division_l2553_255362

/-- Given a scale of length 80 inches divided into equal parts of 20 inches each,
    prove that the number of equal parts is 4. -/
theorem scale_division (scale_length : ℕ) (part_length : ℕ) (h1 : scale_length = 80) (h2 : part_length = 20) :
  scale_length / part_length = 4 := by
  sorry

end NUMINAMATH_CALUDE_scale_division_l2553_255362


namespace NUMINAMATH_CALUDE_pets_remaining_l2553_255307

theorem pets_remaining (initial_puppies initial_kittens sold_puppies sold_kittens : ℕ) 
  (h1 : initial_puppies = 7)
  (h2 : initial_kittens = 6)
  (h3 : sold_puppies = 2)
  (h4 : sold_kittens = 3) :
  initial_puppies + initial_kittens - (sold_puppies + sold_kittens) = 8 := by
sorry

end NUMINAMATH_CALUDE_pets_remaining_l2553_255307


namespace NUMINAMATH_CALUDE_sum_divisibility_l2553_255319

theorem sum_divisibility : 
  let y := 72 + 144 + 216 + 288 + 576 + 720 + 4608
  (∃ k : ℤ, y = 6 * k) ∧ 
  (∃ k : ℤ, y = 12 * k) ∧ 
  (∃ k : ℤ, y = 24 * k) ∧ 
  ¬(∃ k : ℤ, y = 48 * k) := by
  sorry

end NUMINAMATH_CALUDE_sum_divisibility_l2553_255319


namespace NUMINAMATH_CALUDE_triangle_t_range_l2553_255366

theorem triangle_t_range (a b c : ℝ) (A B C : ℝ) (t : ℝ) :
  a > 0 → b > 0 → c > 0 →
  A > 0 → B > 0 → C > 0 →
  A + B + C = Real.pi →
  a * c = (1/4) * b^2 →
  Real.sin A + Real.sin C = t * Real.sin B →
  0 < B → B < Real.pi/2 →
  ∃ (t_min t_max : ℝ), t_min = Real.sqrt 6 / 2 ∧ t_max = Real.sqrt 2 ∧ t_min < t ∧ t < t_max :=
by sorry

end NUMINAMATH_CALUDE_triangle_t_range_l2553_255366


namespace NUMINAMATH_CALUDE_balance_theorem_l2553_255371

-- Define the weights of balls in terms of blue balls
def green_weight : ℚ := 2
def yellow_weight : ℚ := 5/2
def white_weight : ℚ := 3/2

-- Define the balance conditions
axiom green_balance : 3 * green_weight = 6
axiom yellow_balance : 2 * yellow_weight = 5
axiom white_balance : 6 = 4 * white_weight

-- Theorem to prove
theorem balance_theorem : 
  4 * green_weight + 2 * yellow_weight + 2 * white_weight = 16 := by
  sorry


end NUMINAMATH_CALUDE_balance_theorem_l2553_255371


namespace NUMINAMATH_CALUDE_lowest_unique_score_l2553_255328

/-- Represents the scoring system for the national Mathematics Competition. -/
structure ScoringSystem where
  totalProblems : ℕ
  correctPoints : ℕ
  wrongPoints : ℕ
  baseScore : ℕ

/-- Calculates the score based on the number of correct and wrong answers. -/
def calculateScore (system : ScoringSystem) (correct : ℕ) (wrong : ℕ) : ℕ :=
  system.baseScore + system.correctPoints * correct - system.wrongPoints * wrong

/-- Checks if the number of correct answers can be uniquely determined from the score. -/
def isUniqueDetermination (system : ScoringSystem) (score : ℕ) : Prop :=
  ∃! correct : ℕ, ∃ wrong : ℕ,
    correct + wrong ≤ system.totalProblems ∧
    calculateScore system correct wrong = score

/-- The theorem stating that 105 is the lowest score above 100 for which
    the number of correctly solved problems can be uniquely determined. -/
theorem lowest_unique_score (system : ScoringSystem)
    (h1 : system.totalProblems = 40)
    (h2 : system.correctPoints = 5)
    (h3 : system.wrongPoints = 1)
    (h4 : system.baseScore = 40) :
    (∀ s, 100 < s → s < 105 → ¬ isUniqueDetermination system s) ∧
    isUniqueDetermination system 105 := by
  sorry

end NUMINAMATH_CALUDE_lowest_unique_score_l2553_255328


namespace NUMINAMATH_CALUDE_inverse_function_decreasing_l2553_255304

theorem inverse_function_decreasing :
  ∀ (x₁ x₂ : ℝ), x₁ > 0 → x₂ > 0 → x₁ > x₂ → (1 / x₁) < (1 / x₂) := by
  sorry

end NUMINAMATH_CALUDE_inverse_function_decreasing_l2553_255304


namespace NUMINAMATH_CALUDE_range_of_a_l2553_255374

theorem range_of_a (a : ℝ) : (∀ x : ℝ, x^2 + a*x + 1 ≥ 0) → -2 ≤ a ∧ a ≤ 2 := by
  sorry

end NUMINAMATH_CALUDE_range_of_a_l2553_255374


namespace NUMINAMATH_CALUDE_fifty_percent_relation_l2553_255390

theorem fifty_percent_relation (x y : ℝ) : 
  (0.5 * x = y + 20) → (x - 2 * y = 40) := by
  sorry

end NUMINAMATH_CALUDE_fifty_percent_relation_l2553_255390


namespace NUMINAMATH_CALUDE_planes_parallel_if_perpendicular_to_same_line_l2553_255378

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the perpendicular relation between a line and a plane
variable (perpendicular : Line → Plane → Prop)

-- Define the parallel relation between two planes
variable (parallel : Plane → Plane → Prop)

-- Define the non-overlapping property for planes
variable (non_overlapping : Plane → Plane → Prop)

-- Theorem statement
theorem planes_parallel_if_perpendicular_to_same_line 
  (m : Line) (α β : Plane) 
  (h1 : perpendicular m α) 
  (h2 : perpendicular m β) 
  (h3 : non_overlapping α β) : 
  parallel α β :=
sorry

end NUMINAMATH_CALUDE_planes_parallel_if_perpendicular_to_same_line_l2553_255378


namespace NUMINAMATH_CALUDE_simplify_square_roots_l2553_255301

theorem simplify_square_roots : 
  (Real.sqrt 800 / Real.sqrt 200) * ((Real.sqrt 180 / Real.sqrt 72) - (Real.sqrt 224 / Real.sqrt 56)) = Real.sqrt 10 - 4 := by
  sorry

end NUMINAMATH_CALUDE_simplify_square_roots_l2553_255301


namespace NUMINAMATH_CALUDE_no_integer_solutions_l2553_255331

theorem no_integer_solutions : ¬∃ (x y : ℤ), x^2 + 3*x*y - 2*y^2 = 122 := by
  sorry

end NUMINAMATH_CALUDE_no_integer_solutions_l2553_255331


namespace NUMINAMATH_CALUDE_washer_dryer_cost_l2553_255339

theorem washer_dryer_cost (dryer_cost washer_cost total_cost : ℕ) : 
  dryer_cost = 150 →
  washer_cost = 3 * dryer_cost →
  total_cost = washer_cost + dryer_cost →
  total_cost = 600 := by
sorry

end NUMINAMATH_CALUDE_washer_dryer_cost_l2553_255339


namespace NUMINAMATH_CALUDE_at_least_one_less_than_two_l2553_255348

theorem at_least_one_less_than_two (a b : ℝ) 
  (ha : a > 0) (hb : b > 0) (hab : a + b > 2) : 
  (1 + b) / a < 2 ∨ (1 + a) / b < 2 := by
  sorry

end NUMINAMATH_CALUDE_at_least_one_less_than_two_l2553_255348


namespace NUMINAMATH_CALUDE_max_area_rectangular_pen_l2553_255325

theorem max_area_rectangular_pen (fencing : ℝ) (h_fencing : fencing = 60) :
  let width := fencing / 6
  let length := 2 * width
  let area := width * length
  area = 200 := by sorry

end NUMINAMATH_CALUDE_max_area_rectangular_pen_l2553_255325


namespace NUMINAMATH_CALUDE_function_value_range_bounds_are_tight_l2553_255379

theorem function_value_range (x : ℝ) : 
  ∃ (y : ℝ), y = Real.sin x - Real.cos (x + π/6) ∧ 
  -Real.sqrt 3 ≤ y ∧ y ≤ Real.sqrt 3 :=
by sorry

theorem bounds_are_tight : 
  (∃ (x : ℝ), Real.sin x - Real.cos (x + π/6) = -Real.sqrt 3) ∧
  (∃ (x : ℝ), Real.sin x - Real.cos (x + π/6) = Real.sqrt 3) :=
by sorry

end NUMINAMATH_CALUDE_function_value_range_bounds_are_tight_l2553_255379


namespace NUMINAMATH_CALUDE_factorization_equality_l2553_255396

theorem factorization_equality (x y : ℝ) : 4 * x^2 * y - 12 * x * y = 4 * x * y * (x - 3) := by
  sorry

end NUMINAMATH_CALUDE_factorization_equality_l2553_255396


namespace NUMINAMATH_CALUDE_square_sum_given_product_and_sum_l2553_255330

theorem square_sum_given_product_and_sum (a b : ℝ) 
  (h1 : a * b = 16) 
  (h2 : a + b = 10) : 
  a^2 + b^2 = 68 := by
sorry

end NUMINAMATH_CALUDE_square_sum_given_product_and_sum_l2553_255330


namespace NUMINAMATH_CALUDE_gas_refill_amount_l2553_255389

/-- Calculates the amount of gas needed to refill a car's tank --/
theorem gas_refill_amount (initial_gas tank_capacity store_trip doctor_trip : ℝ) 
  (h1 : initial_gas = 10)
  (h2 : tank_capacity = 12)
  (h3 : store_trip = 6)
  (h4 : doctor_trip = 2) :
  tank_capacity - (initial_gas - store_trip - doctor_trip) = 10 := by
  sorry

end NUMINAMATH_CALUDE_gas_refill_amount_l2553_255389


namespace NUMINAMATH_CALUDE_x_value_equality_l2553_255315

def F (x y z : ℝ) : ℝ := x * y^3 + z^2

theorem x_value_equality : ∃ x : ℝ, F x 3 2 = F x 2 5 ∧ x = 21/19 := by
  sorry

end NUMINAMATH_CALUDE_x_value_equality_l2553_255315


namespace NUMINAMATH_CALUDE_coordinate_point_A_coordinate_point_B_l2553_255352

-- Definition of a coordinate point
def is_coordinate_point (x y : ℝ) : Prop := 2 * x - y = 1

-- Part 1
theorem coordinate_point_A : 
  ∀ a : ℝ, is_coordinate_point 3 a ↔ a = 5 :=
sorry

-- Part 2
theorem coordinate_point_B :
  ∀ b c : ℕ, (is_coordinate_point (b + c) (b + 5) ∧ b > 0 ∧ c > 0) ↔ 
  ((b = 2 ∧ c = 2) ∨ (b = 4 ∧ c = 1)) :=
sorry

end NUMINAMATH_CALUDE_coordinate_point_A_coordinate_point_B_l2553_255352


namespace NUMINAMATH_CALUDE_smallest_common_factor_l2553_255398

theorem smallest_common_factor (n : ℕ) : 
  (∀ m : ℕ, m > 0 ∧ m < n → ¬(∃ k : ℕ, k > 1 ∧ k ∣ (8*m - 3) ∧ k ∣ (6*m + 4))) ∧ 
  (∃ k : ℕ, k > 1 ∧ k ∣ (8*n - 3) ∧ k ∣ (6*n + 4)) → 
  n = 1 := by
sorry

end NUMINAMATH_CALUDE_smallest_common_factor_l2553_255398


namespace NUMINAMATH_CALUDE_shoe_cost_theorem_l2553_255397

/-- Calculates the final price after applying discount and tax -/
def calculate_price (original_price : ℝ) (discount_rate : ℝ) (tax_rate : ℝ) : ℝ :=
  let discounted_price := original_price * (1 - discount_rate)
  discounted_price * (1 + tax_rate)

/-- Calculates the total cost of four pairs of shoes -/
def total_cost (price1 price2 price3 : ℝ) : ℝ :=
  let pair1 := calculate_price price1 0.10 0.05
  let pair2 := calculate_price (price1 * 1.5) 0.15 0.07
  let pair3_4 := price3 * (1 + 0.12)
  pair1 + pair2 + pair3_4

theorem shoe_cost_theorem :
  total_cost 22 (22 * 1.5) 40 = 95.60 := by
  sorry

end NUMINAMATH_CALUDE_shoe_cost_theorem_l2553_255397


namespace NUMINAMATH_CALUDE_circle_intersection_m_range_l2553_255365

-- Define the circle equation
def circle_equation (x y m : ℝ) : Prop :=
  x^2 + y^2 - 4*x + 2*m*y + m + 6 = 0

-- Define the condition that intersections are on the same side of the origin
def intersections_same_side (m : ℝ) : Prop :=
  ∃ y₁ y₂ : ℝ, y₁ * y₂ > 0 ∧ circle_equation 0 y₁ m ∧ circle_equation 0 y₂ m

-- Theorem statement
theorem circle_intersection_m_range :
  ∀ m : ℝ, intersections_same_side m → (m > 2 ∨ (-6 < m ∧ m < -2)) :=
sorry

end NUMINAMATH_CALUDE_circle_intersection_m_range_l2553_255365


namespace NUMINAMATH_CALUDE_unique_positive_p_for_geometric_progression_l2553_255336

/-- A geometric progression is a sequence where each term after the first is found by multiplying the previous term by a fixed, non-zero number called the common ratio. -/
def IsGeometricProgression (a b c : ℝ) : Prop :=
  ∃ r : ℝ, r ≠ 0 ∧ b = a * r ∧ c = b * r

/-- The theorem states that 4 is the only positive real number p such that -p-12, 2√p, and p-5 form a geometric progression. -/
theorem unique_positive_p_for_geometric_progression :
  ∃! p : ℝ, p > 0 ∧ IsGeometricProgression (-p - 12) (2 * Real.sqrt p) (p - 5) :=
by
  sorry

#check unique_positive_p_for_geometric_progression

end NUMINAMATH_CALUDE_unique_positive_p_for_geometric_progression_l2553_255336


namespace NUMINAMATH_CALUDE_fixed_point_on_line_l2553_255385

theorem fixed_point_on_line (m : ℝ) : (2*m - 1)*2 + (m + 3)*(-3) - (m - 11) = 0 := by
  sorry

end NUMINAMATH_CALUDE_fixed_point_on_line_l2553_255385


namespace NUMINAMATH_CALUDE_triangle_side_length_l2553_255372

theorem triangle_side_length (a b c : ℝ) (A B C : ℝ) :
  -- Conditions
  (2 * b = a + c) →  -- a, b, c form an arithmetic sequence
  (B = π / 6) →  -- Angle B = 30° (in radians)
  (1 / 2 * a * c * Real.sin B = 3 / 2) →  -- Area of triangle ABC = 3/2
  -- Conclusion
  b = Real.sqrt 3 + 1 := by
  sorry

end NUMINAMATH_CALUDE_triangle_side_length_l2553_255372


namespace NUMINAMATH_CALUDE_solve_quadratic_equation_l2553_255373

theorem solve_quadratic_equation (x : ℝ) (h1 : 3 * x^2 - 9 * x = 0) (h2 : x ≠ 0) : x = 3 := by
  sorry

end NUMINAMATH_CALUDE_solve_quadratic_equation_l2553_255373


namespace NUMINAMATH_CALUDE_intersection_S_T_l2553_255332

def S : Set ℝ := {x | x^2 + 2*x = 0}
def T : Set ℝ := {x | x^2 - 2*x = 0}

theorem intersection_S_T : S ∩ T = {0} := by sorry

end NUMINAMATH_CALUDE_intersection_S_T_l2553_255332


namespace NUMINAMATH_CALUDE_polynomial_remainder_l2553_255343

/-- Given a polynomial Q(x) where Q(17) = 41 and Q(93) = 13, 
    the remainder when Q(x) is divided by (x - 17)(x - 93) is -7/19*x + 900/19 -/
theorem polynomial_remainder (Q : ℝ → ℝ) (h1 : Q 17 = 41) (h2 : Q 93 = 13) :
  ∃ (R : ℝ → ℝ), ∀ x, Q x = (x - 17) * (x - 93) * R x + (-7/19 * x + 900/19) :=
sorry

end NUMINAMATH_CALUDE_polynomial_remainder_l2553_255343


namespace NUMINAMATH_CALUDE_floor_sqrt_sum_l2553_255337

theorem floor_sqrt_sum (a b c : ℝ) : 
  |a| = 4 → b^2 = 9 → c^3 = -8 → a > b → b > c → 
  ⌊Real.sqrt (a + b + c)⌋ = 2 := by sorry

end NUMINAMATH_CALUDE_floor_sqrt_sum_l2553_255337


namespace NUMINAMATH_CALUDE_consecutive_integers_sum_equality_l2553_255329

theorem consecutive_integers_sum_equality (x : ℤ) (h : x = 25) :
  (x + (x + 1) + (x + 2) + (x + 3) + (x + 4) + (x + 5)) =
  ((x + 6) + (x + 7) + (x + 8) + (x + 9) + (x + 10)) := by
  sorry

#check consecutive_integers_sum_equality

end NUMINAMATH_CALUDE_consecutive_integers_sum_equality_l2553_255329


namespace NUMINAMATH_CALUDE_system_of_equations_solution_l2553_255393

theorem system_of_equations_solution (x y m : ℝ) : 
  x + 2*y = 5*m →
  x - 2*y = 9*m →
  3*x + 2*y = 19 →
  m = 1 := by
sorry

end NUMINAMATH_CALUDE_system_of_equations_solution_l2553_255393


namespace NUMINAMATH_CALUDE_dollar_evaluation_l2553_255303

def dollar (a b : ℝ) : ℝ := (a - b)^2

theorem dollar_evaluation (x y : ℝ) :
  dollar (2*x + 3*y) (3*x - 4*y) = x^2 - 14*x*y + 49*y^2 := by
  sorry

end NUMINAMATH_CALUDE_dollar_evaluation_l2553_255303


namespace NUMINAMATH_CALUDE_mailing_cost_formula_l2553_255387

/-- The cost function for mailing a package -/
noncomputable def mailing_cost (W : ℝ) : ℝ := 8 * ⌈W / 2⌉

/-- Theorem stating the correct formula for the mailing cost -/
theorem mailing_cost_formula (W : ℝ) : 
  mailing_cost W = 8 * ⌈W / 2⌉ := by
  sorry

#check mailing_cost_formula

end NUMINAMATH_CALUDE_mailing_cost_formula_l2553_255387


namespace NUMINAMATH_CALUDE_man_in_well_l2553_255313

/-- The number of days required for a man to climb out of a well -/
def daysToClimbOut (wellDepth : ℕ) (climbUp : ℕ) (slipDown : ℕ) : ℕ :=
  let netClimbPerDay := climbUp - slipDown
  (wellDepth - 1) / netClimbPerDay + 1

/-- Theorem: It takes 30 days for a man to climb out of a 30-meter deep well
    when he climbs 4 meters up and slips 3 meters down each day -/
theorem man_in_well : daysToClimbOut 30 4 3 = 30 := by
  sorry

end NUMINAMATH_CALUDE_man_in_well_l2553_255313


namespace NUMINAMATH_CALUDE_impossibleTable_l2553_255322

/-- Represents a digit from 1 to 9 -/
def Digit := Fin 9

/-- Represents a 10x10 table of digits -/
def Table := Fin 10 → Fin 10 → Digit

/-- Converts a sequence of 10 digits to a natural number -/
def toNumber (seq : Fin 10 → Digit) : ℕ := sorry

/-- The main theorem stating the impossibility of constructing the required table -/
theorem impossibleTable : ¬ ∃ (t : Table),
  (∀ i : Fin 10, toNumber (λ j => t i j) > toNumber (λ k => t k k)) ∧
  (∀ j : Fin 10, toNumber (λ k => t k k) > toNumber (λ i => t i j)) := by
  sorry


end NUMINAMATH_CALUDE_impossibleTable_l2553_255322


namespace NUMINAMATH_CALUDE_bobby_deadlift_difference_l2553_255356

/-- Given Bobby's initial deadlift and yearly increase, prove the difference between his deadlift at 18 and 250% of his deadlift at 13 -/
theorem bobby_deadlift_difference (initial_deadlift : ℕ) (yearly_increase : ℕ) (years : ℕ) : 
  initial_deadlift = 300 →
  yearly_increase = 110 →
  years = 5 →
  (initial_deadlift + years * yearly_increase) - (initial_deadlift * 250 / 100) = 100 := by
sorry

end NUMINAMATH_CALUDE_bobby_deadlift_difference_l2553_255356


namespace NUMINAMATH_CALUDE_square_difference_of_quadratic_solutions_l2553_255344

theorem square_difference_of_quadratic_solutions : 
  ∀ Φ φ : ℝ, Φ ≠ φ → Φ^2 = Φ + 2 → φ^2 = φ + 2 → (Φ - φ)^2 = 9 := by
  sorry

end NUMINAMATH_CALUDE_square_difference_of_quadratic_solutions_l2553_255344


namespace NUMINAMATH_CALUDE_dads_strawberries_weight_l2553_255312

/-- The weight of Marco's dad's strawberries -/
def dads_strawberries (marcos_weight total_weight : ℕ) : ℕ :=
  total_weight - marcos_weight

/-- Theorem: Marco's dad's strawberries weigh 22 pounds -/
theorem dads_strawberries_weight :
  dads_strawberries 15 37 = 22 := by
  sorry

end NUMINAMATH_CALUDE_dads_strawberries_weight_l2553_255312


namespace NUMINAMATH_CALUDE_parabola_c_value_l2553_255357

/-- Represents a parabola of the form x = ay² + by + c -/
structure Parabola where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Checks if a point (x, y) lies on the parabola -/
def Parabola.contains (p : Parabola) (x y : ℝ) : Prop :=
  x = p.a * y^2 + p.b * y + p.c

/-- The vertex of a parabola -/
def Parabola.vertex (p : Parabola) : ℝ × ℝ := sorry

theorem parabola_c_value :
  ∀ p : Parabola,
  p.vertex = (3, -5) →
  p.contains 0 6 →
  p.c = 288 / 121 := by
  sorry

end NUMINAMATH_CALUDE_parabola_c_value_l2553_255357


namespace NUMINAMATH_CALUDE_cone_volume_in_cylinder_with_spheres_l2553_255388

/-- The volume of a cone inscribed in a cylinder with specific properties --/
theorem cone_volume_in_cylinder_with_spheres (r h : ℝ) : 
  r = 1 → 
  h = 12 / (3 + 2 * Real.sqrt 3) →
  ∃ (cone_volume : ℝ), 
    cone_volume = (2/3) * Real.pi ∧
    cone_volume = (1/3) * Real.pi * r^2 * 1 ∧
    ∃ (sphere_radius : ℝ),
      sphere_radius = 2 * Real.sqrt 3 - 3 ∧
      sphere_radius > 0 ∧
      sphere_radius < r ∧
      h = 2 * sphere_radius :=
by sorry

end NUMINAMATH_CALUDE_cone_volume_in_cylinder_with_spheres_l2553_255388


namespace NUMINAMATH_CALUDE_repeating_decimal_equals_fraction_l2553_255381

/-- Represents the repeating decimal 0.35247̄ -/
def repeating_decimal : ℚ :=
  35247 / 100000 + (247 / 100000) / (1 - 1 / 1000)

/-- The fraction we want to prove equality with -/
def target_fraction : ℚ := 3518950 / 999900

/-- Theorem stating that the repeating decimal equals the target fraction -/
theorem repeating_decimal_equals_fraction :
  repeating_decimal = target_fraction := by sorry

end NUMINAMATH_CALUDE_repeating_decimal_equals_fraction_l2553_255381


namespace NUMINAMATH_CALUDE_arithmetic_calculations_l2553_255306

theorem arithmetic_calculations : 
  (12 - (-18) + (-7) - 15 = 8) ∧ 
  ((-1/2) * (-8) + (-6) / (-1/3)^2 = -50) := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_calculations_l2553_255306


namespace NUMINAMATH_CALUDE_apple_price_reduction_l2553_255358

-- Define the given conditions
def reduced_price_per_dozen : ℚ := 3
def total_money : ℚ := 40
def additional_apples : ℕ := 64

-- Define the function to calculate the percentage reduction
def calculate_percentage_reduction (original_price reduced_price : ℚ) : ℚ :=
  ((original_price - reduced_price) / original_price) * 100

-- State the theorem
theorem apple_price_reduction :
  let dozens_at_reduced_price := total_money / reduced_price_per_dozen
  let additional_dozens := additional_apples / 12
  let dozens_at_original_price := dozens_at_reduced_price - additional_dozens
  let original_price_per_dozen := total_money / dozens_at_original_price
  calculate_percentage_reduction original_price_per_dozen reduced_price_per_dozen = 40 := by
  sorry

end NUMINAMATH_CALUDE_apple_price_reduction_l2553_255358


namespace NUMINAMATH_CALUDE_unique_m_existence_l2553_255346

theorem unique_m_existence : ∃! m : ℤ,
  50 ≤ m ∧ m ≤ 180 ∧
  m % 9 = 0 ∧
  m % 10 = 7 ∧
  m % 7 = 5 ∧
  m = 117 := by
  sorry

end NUMINAMATH_CALUDE_unique_m_existence_l2553_255346


namespace NUMINAMATH_CALUDE_perfect_square_floor_equality_l2553_255323

theorem perfect_square_floor_equality (n : ℕ+) :
  ⌊2 * Real.sqrt n⌋ = ⌊Real.sqrt (n - 1) + Real.sqrt (n + 1)⌋ + 1 ↔ ∃ m : ℕ+, n = m^2 :=
by sorry

end NUMINAMATH_CALUDE_perfect_square_floor_equality_l2553_255323


namespace NUMINAMATH_CALUDE_geometric_sequence_sum_l2553_255310

/-- Given a geometric sequence of real numbers {a_n}, prove that if the sum of the first three terms is 2
    and the sum of the 4th, 5th, and 6th terms is 16, then the sum of the 7th, 8th, and 9th terms is 128. -/
theorem geometric_sequence_sum (a : ℕ → ℝ) (h_geometric : ∀ n : ℕ, ∃ q : ℝ, a (n + 1) = q * a n)
    (h_sum1 : a 1 + a 2 + a 3 = 2) (h_sum2 : a 4 + a 5 + a 6 = 16) : a 7 + a 8 + a 9 = 128 := by
  sorry


end NUMINAMATH_CALUDE_geometric_sequence_sum_l2553_255310


namespace NUMINAMATH_CALUDE_pie_eating_contest_l2553_255375

theorem pie_eating_contest (erik_pie frank_pie : ℚ) : 
  erik_pie = 0.6666666666666666 →
  erik_pie = frank_pie + 0.3333333333333333 →
  frank_pie = 0.3333333333333333 := by
  sorry

end NUMINAMATH_CALUDE_pie_eating_contest_l2553_255375


namespace NUMINAMATH_CALUDE_specific_quadrilateral_perimeter_l2553_255335

/-- A convex quadrilateral with an interior point -/
structure ConvexQuadrilateral where
  W : ℝ × ℝ
  X : ℝ × ℝ
  Y : ℝ × ℝ
  Z : ℝ × ℝ
  Q : ℝ × ℝ
  area : ℝ
  wq : ℝ
  xq : ℝ
  yq : ℝ
  zq : ℝ
  convex : Bool
  interior : Bool

/-- The perimeter of a quadrilateral -/
def perimeter (quad : ConvexQuadrilateral) : ℝ :=
  sorry

/-- Theorem: The perimeter of the specific quadrilateral is 230 + 10√41 -/
theorem specific_quadrilateral_perimeter (quad : ConvexQuadrilateral) 
  (h_area : quad.area = 2500)
  (h_wq : quad.wq = 30)
  (h_xq : quad.xq = 40)
  (h_yq : quad.yq = 50)
  (h_zq : quad.zq = 60)
  (h_convex : quad.convex = true)
  (h_interior : quad.interior = true) :
  perimeter quad = 230 + 10 * Real.sqrt 41 :=
sorry

end NUMINAMATH_CALUDE_specific_quadrilateral_perimeter_l2553_255335


namespace NUMINAMATH_CALUDE_age_ratio_l2553_255309

theorem age_ratio (sum_ages : ℕ) (your_age : ℕ) : 
  sum_ages = 40 → your_age = 10 → (sum_ages - your_age) / your_age = 3 := by
  sorry

end NUMINAMATH_CALUDE_age_ratio_l2553_255309


namespace NUMINAMATH_CALUDE_grid_square_triangle_count_l2553_255399

/-- Represents a square divided into a 4x4 grid with diagonals --/
structure GridSquare :=
  (size : ℕ)
  (has_diagonals : Bool)
  (has_small_square_diagonals : Bool)

/-- Counts the number of triangles in a GridSquare --/
def count_triangles (sq : GridSquare) : ℕ :=
  sorry

/-- The main theorem stating that a 4x4 GridSquare with all diagonals has 42 triangles --/
theorem grid_square_triangle_count :
  ∀ (sq : GridSquare), 
    sq.size = 4 ∧ 
    sq.has_diagonals = true ∧ 
    sq.has_small_square_diagonals = true → 
    count_triangles sq = 42 :=
  sorry

end NUMINAMATH_CALUDE_grid_square_triangle_count_l2553_255399


namespace NUMINAMATH_CALUDE_min_value_implies_a_l2553_255327

/-- The function f(x) defined in the problem -/
def f (a : ℝ) (x : ℝ) : ℝ := |x + 1| + 2 * |x - a|

/-- The theorem stating the relationship between the minimum value of f and the value of a -/
theorem min_value_implies_a (a : ℝ) : 
  (∀ x, f a x ≥ 5) ∧ (∃ x, f a x = 5) → a = -6 ∨ a = 4 :=
sorry

end NUMINAMATH_CALUDE_min_value_implies_a_l2553_255327


namespace NUMINAMATH_CALUDE_vector_problem_l2553_255324

/-- Given vectors in R^2 -/
def a : Fin 2 → ℝ := ![1, 2]
def b (x : ℝ) : Fin 2 → ℝ := ![x, 1]

/-- Dot product of two 2D vectors -/
def dot_product (v w : Fin 2 → ℝ) : ℝ := (v 0) * (w 0) + (v 1) * (w 1)

/-- Acute angle between vectors -/
def is_acute_angle (v w : Fin 2 → ℝ) : Prop := 
  dot_product v w > 0 ∧ ¬ ∃ (k : ℝ), v = fun i => k * (w i)

/-- Orthogonality of vectors -/
def is_orthogonal (v w : Fin 2 → ℝ) : Prop := dot_product v w = 0

theorem vector_problem (x : ℝ) :
  (is_acute_angle a (b x) ↔ x > -2 ∧ x ≠ 1/2) ∧
  (is_orthogonal (fun i => a i + 2 * (b x i)) (fun i => 2 * a i - b x i) ↔ x = 7/2) := by
  sorry

end NUMINAMATH_CALUDE_vector_problem_l2553_255324


namespace NUMINAMATH_CALUDE_work_completion_time_l2553_255320

/-- Given two people working together can complete a task in 10 days,
    and one person (Prakash) can complete the task in 30 days,
    prove that the other person can complete the task alone in 15 days. -/
theorem work_completion_time (prakash_time : ℕ) (joint_time : ℕ) (x : ℕ) :
  prakash_time = 30 →
  joint_time = 10 →
  (1 : ℚ) / x + (1 : ℚ) / prakash_time = (1 : ℚ) / joint_time →
  x = 15 := by
  sorry

end NUMINAMATH_CALUDE_work_completion_time_l2553_255320


namespace NUMINAMATH_CALUDE_parallel_vectors_l2553_255380

/-- Given two vectors AB and CD in R², if AB is parallel to CD and AB = (6,1) and CD = (x,-3), then x = -18 -/
theorem parallel_vectors (AB CD : ℝ × ℝ) (x : ℝ) 
  (h1 : AB = (6, 1)) 
  (h2 : CD = (x, -3)) 
  (h3 : ∃ (k : ℝ), k ≠ 0 ∧ AB = k • CD) : 
  x = -18 := by
  sorry

end NUMINAMATH_CALUDE_parallel_vectors_l2553_255380


namespace NUMINAMATH_CALUDE_cos_105_degrees_l2553_255376

theorem cos_105_degrees :
  Real.cos (105 * π / 180) = (Real.sqrt 2 - Real.sqrt 6) / 4 := by
  sorry

end NUMINAMATH_CALUDE_cos_105_degrees_l2553_255376


namespace NUMINAMATH_CALUDE_circle_area_difference_l2553_255334

/-- The difference between the areas of two circles with given circumferences -/
theorem circle_area_difference (c₁ c₂ : ℝ) (h₁ : c₁ = 660) (h₂ : c₂ = 704) :
  ∃ (diff : ℝ), abs (diff - ((c₂^2 - c₁^2) / (4 * Real.pi))) < 0.001 := by
  sorry

end NUMINAMATH_CALUDE_circle_area_difference_l2553_255334


namespace NUMINAMATH_CALUDE_total_raisins_l2553_255361

theorem total_raisins (yellow_raisins : ℝ) (black_raisins : ℝ) (red_raisins : ℝ)
  (h1 : yellow_raisins = 0.3)
  (h2 : black_raisins = 0.4)
  (h3 : red_raisins = 0.5) :
  yellow_raisins + black_raisins + red_raisins = 1.2 := by
  sorry

end NUMINAMATH_CALUDE_total_raisins_l2553_255361


namespace NUMINAMATH_CALUDE_kids_at_camp_l2553_255314

theorem kids_at_camp (total : ℕ) (stay_home : ℕ) (h1 : total = 898051) (h2 : stay_home = 268627) :
  total - stay_home = 629424 := by
  sorry

end NUMINAMATH_CALUDE_kids_at_camp_l2553_255314


namespace NUMINAMATH_CALUDE_linear_function_not_in_quadrant_III_l2553_255341

/-- A linear function defined by y = mx + b -/
structure LinearFunction where
  m : ℝ  -- slope
  b : ℝ  -- y-intercept

/-- Represents the four quadrants of a 2D coordinate system -/
inductive Quadrant
  | I
  | II
  | III
  | IV

/-- Checks if a point (x, y) is in Quadrant III -/
def isInQuadrantIII (x y : ℝ) : Prop :=
  x < 0 ∧ y < 0

/-- Theorem: The linear function y = -2x + 1 does not pass through Quadrant III -/
theorem linear_function_not_in_quadrant_III :
  ∀ x y : ℝ, y = -2 * x + 1 → ¬(isInQuadrantIII x y) := by
  sorry

end NUMINAMATH_CALUDE_linear_function_not_in_quadrant_III_l2553_255341


namespace NUMINAMATH_CALUDE_problem_solution_l2553_255383

theorem problem_solution (m n c : Int) (hm : m = -4) (hn : n = -5) (hc : c = -7) :
  m - n - c = 8 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l2553_255383


namespace NUMINAMATH_CALUDE_intersection_of_M_and_N_l2553_255369

-- Define the sets M and N
def M : Set ℝ := {y | ∃ x : ℝ, y = x^2}
def N : Set ℝ := {y | ∃ x : ℝ, y = x + 2}

-- State the theorem
theorem intersection_of_M_and_N : M ∩ N = Set.Ici 0 := by
  sorry

end NUMINAMATH_CALUDE_intersection_of_M_and_N_l2553_255369


namespace NUMINAMATH_CALUDE_lives_difference_l2553_255321

/-- The number of lives of animals in a fictional world --/
def lives_problem (cat_lives dog_lives mouse_lives : ℕ) : Prop :=
  cat_lives = 9 ∧
  dog_lives = cat_lives - 3 ∧
  mouse_lives = 13 ∧
  mouse_lives - dog_lives = 7

theorem lives_difference :
  ∃ (cat_lives dog_lives mouse_lives : ℕ),
    lives_problem cat_lives dog_lives mouse_lives :=
by
  sorry

end NUMINAMATH_CALUDE_lives_difference_l2553_255321


namespace NUMINAMATH_CALUDE_trisector_triangle_angles_l2553_255305

/-- Given a triangle ABC with angles α, β, and γ, if the triangle formed by the first angle trisectors
    has two angles of 45° and 55°, then the triangle formed by the second angle trisectors
    has angles of 40°, 65°, and 75°. -/
theorem trisector_triangle_angles 
  (α β γ : Real) 
  (h_sum : α + β + γ = 180)
  (h_positive : 0 < α ∧ 0 < β ∧ 0 < γ)
  (h_first_trisector : 
    ((β + 2*γ)/3 = 45 ∧ (γ + 2*α)/3 = 55) ∨ 
    ((β + 2*γ)/3 = 55 ∧ (γ + 2*α)/3 = 45) ∨
    ((γ + 2*α)/3 = 45 ∧ (α + 2*β)/3 = 55) ∨
    ((γ + 2*α)/3 = 55 ∧ (α + 2*β)/3 = 45) ∨
    ((α + 2*β)/3 = 45 ∧ (β + 2*γ)/3 = 55) ∨
    ((α + 2*β)/3 = 55 ∧ (β + 2*γ)/3 = 45)) :
  (2*β + γ)/3 = 65 ∧ (2*γ + α)/3 = 40 ∧ (2*α + β)/3 = 75 := by
  sorry


end NUMINAMATH_CALUDE_trisector_triangle_angles_l2553_255305


namespace NUMINAMATH_CALUDE_inequality_equivalence_l2553_255351

theorem inequality_equivalence (x : ℝ) : 
  (5 / 24 + |x - 11 / 48| < 5 / 16) ↔ (1 / 8 < x ∧ x < 1 / 3) := by
  sorry

end NUMINAMATH_CALUDE_inequality_equivalence_l2553_255351
