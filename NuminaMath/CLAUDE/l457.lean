import Mathlib

namespace circle_probability_theorem_l457_45763

theorem circle_probability_theorem (R : ℝ) (h : R = 4) :
  let outer_circle_area := π * R^2
  let inner_circle_radius := R - 3
  let inner_circle_area := π * inner_circle_radius^2
  (inner_circle_area / outer_circle_area) = 1/16 := by
sorry

end circle_probability_theorem_l457_45763


namespace all_rooms_on_same_hall_l457_45718

/-- A type representing a hall in the castle -/
def Hall : Type := ℕ

/-- A function that assigns a hall to each room number -/
def room_to_hall : ℕ → Hall := sorry

/-- The property that room n is on the same hall as rooms 2n+1 and 3n+1 -/
def hall_property (room_to_hall : ℕ → Hall) : Prop :=
  ∀ n : ℕ, (room_to_hall n = room_to_hall (2*n + 1)) ∧ (room_to_hall n = room_to_hall (3*n + 1))

/-- The theorem stating that all rooms must be on the same hall -/
theorem all_rooms_on_same_hall (room_to_hall : ℕ → Hall) 
  (h : hall_property room_to_hall) : 
  ∀ m n : ℕ, room_to_hall m = room_to_hall n :=
by sorry

end all_rooms_on_same_hall_l457_45718


namespace keith_turnips_l457_45762

theorem keith_turnips (total : ℕ) (alyssa : ℕ) (keith : ℕ)
  (h1 : total = 15)
  (h2 : alyssa = 9)
  (h3 : keith + alyssa = total) :
  keith = 15 - 9 :=
by sorry

end keith_turnips_l457_45762


namespace complex_fourth_power_equality_l457_45767

theorem complex_fourth_power_equality (a b : ℝ) (ha : a > 0) (hb : b > 0) 
  (h : (a + b * Complex.I) ^ 4 = (a - b * Complex.I) ^ 4) : b / a = 1 := by
  sorry

end complex_fourth_power_equality_l457_45767


namespace geometric_progression_ratio_l457_45775

theorem geometric_progression_ratio (x y z r : ℝ) : 
  x ≠ 0 → y ≠ 0 → z ≠ 0 → x ≠ y → y ≠ z → x ≠ z →
  (∃ a : ℝ, a ≠ 0 ∧ 
    x * (y - z) = a ∧ 
    y * (z - x) = a * r ∧ 
    z * (x - y) = a * r^2) →
  x * (y - z) * y * (z - x) * z * (x - y) = (y * (z - x))^2 →
  r = 1 := by
sorry

end geometric_progression_ratio_l457_45775


namespace log_equation_solution_l457_45719

theorem log_equation_solution (y : ℝ) (h : y > 0) : 
  Real.log y / Real.log 3 + Real.log y / Real.log 9 = 5 → y = 3^(10/3) := by
  sorry

end log_equation_solution_l457_45719


namespace sum_mod_twelve_l457_45766

theorem sum_mod_twelve : (2101 + 2103 + 2105 + 2107 + 2109) % 12 = 1 := by
  sorry

end sum_mod_twelve_l457_45766


namespace expression_evaluation_l457_45712

theorem expression_evaluation :
  let x : ℚ := -1/3
  let y : ℚ := -1
  ((x - 2*y)^2 - (2*x + y)*(x - 4*y) - (-x + 3*y)*(x + 3*y)) / (-y) = 0 := by
  sorry

end expression_evaluation_l457_45712


namespace smallest_dual_base_palindrome_l457_45776

/-- A function to check if a number is a palindrome in a given base -/
def isPalindromeInBase (n : ℕ) (base : ℕ) : Prop := sorry

/-- A function to convert a number from base 10 to another base -/
def toBase (n : ℕ) (base : ℕ) : List ℕ := sorry

theorem smallest_dual_base_palindrome : 
  ∀ k : ℕ, k > 15 → isPalindromeInBase k 3 → isPalindromeInBase k 5 → k ≥ 26 :=
sorry

end smallest_dual_base_palindrome_l457_45776


namespace tangent_lines_for_cubic_curve_l457_45717

def f (x : ℝ) := x^3 - x

theorem tangent_lines_for_cubic_curve :
  let C := f
  -- Tangent line at (2, f(2))
  ∃ (m b : ℝ), (∀ x y, y = m*x + b ↔ m*x - y + b = 0) ∧
               m = 3*2^2 - 1 ∧
               b = f 2 - m*2 ∧
               m*2 - f 2 + b = 0
  ∧
  -- Tangent lines parallel to y = 5x + 3
  ∃ (x₁ x₂ : ℝ), x₁ ≠ x₂ ∧
                 x₁^2 = 2 ∧ x₂^2 = 2 ∧
                 (∀ x y, y - f x₁ = 5*(x - x₁) ↔ 5*x - y - 4*Real.sqrt 2 = 0) ∧
                 (∀ x y, y - f x₂ = 5*(x - x₂) ↔ 5*x - y + 4*Real.sqrt 2 = 0) := by
  sorry

end tangent_lines_for_cubic_curve_l457_45717


namespace thirteen_fourth_mod_eight_l457_45729

theorem thirteen_fourth_mod_eight : 13^4 % 8 = 1 := by
  sorry

end thirteen_fourth_mod_eight_l457_45729


namespace power_of_eight_sum_equals_power_of_two_l457_45744

theorem power_of_eight_sum_equals_power_of_two : ∃ x : ℕ, 8^3 + 8^3 + 8^3 + 8^3 = 2^x ∧ x = 11 := by
  sorry

end power_of_eight_sum_equals_power_of_two_l457_45744


namespace license_plate_difference_l457_45746

/-- The number of letters in the alphabet -/
def num_letters : ℕ := 26

/-- The number of digits -/
def num_digits : ℕ := 10

/-- The number of possible Sunland license plates -/
def sunland_plates : ℕ := num_letters^4 * num_digits^2

/-- The number of possible Moonland license plates -/
def moonland_plates : ℕ := num_letters^3 * num_digits^3

/-- The difference in the number of possible license plates between Sunland and Moonland -/
theorem license_plate_difference :
  sunland_plates - moonland_plates = 7321600 :=
by sorry

end license_plate_difference_l457_45746


namespace sophie_total_spend_l457_45764

-- Define the quantities and prices
def cupcakes : ℕ := 5
def cupcake_price : ℚ := 2

def doughnuts : ℕ := 6
def doughnut_price : ℚ := 1

def apple_pie_slices : ℕ := 4
def apple_pie_price : ℚ := 2

def cookies : ℕ := 15
def cookie_price : ℚ := 0.6

-- Define the total cost function
def total_cost : ℚ :=
  cupcakes * cupcake_price +
  doughnuts * doughnut_price +
  apple_pie_slices * apple_pie_price +
  cookies * cookie_price

-- Theorem statement
theorem sophie_total_spend : total_cost = 33 := by
  sorry

end sophie_total_spend_l457_45764


namespace divisibility_condition_l457_45708

theorem divisibility_condition (n : ℕ) : (n + 1) ∣ (n^2 + 1) ↔ n = 0 ∨ n = 1 := by
  sorry

end divisibility_condition_l457_45708


namespace twelve_by_twelve_grid_intersection_points_l457_45723

/-- The number of interior intersection points in an n by n grid of squares -/
def interior_intersection_points (n : ℕ) : ℕ := (n - 1) * (n - 1)

/-- Theorem: The number of interior intersection points in a 12 by 12 grid of squares is 121 -/
theorem twelve_by_twelve_grid_intersection_points :
  interior_intersection_points 12 = 121 := by
  sorry

end twelve_by_twelve_grid_intersection_points_l457_45723


namespace steve_height_after_growth_l457_45706

/-- Converts feet and inches to total inches -/
def feet_inches_to_inches (feet : ℕ) (inches : ℕ) : ℕ :=
  feet * 12 + inches

/-- Represents a person's height in feet and inches -/
structure Height where
  feet : ℕ
  inches : ℕ

/-- Calculates the new height after growth -/
def new_height_after_growth (initial_height : Height) (growth_inches : ℕ) : ℕ :=
  feet_inches_to_inches initial_height.feet initial_height.inches + growth_inches

theorem steve_height_after_growth :
  let initial_height : Height := ⟨5, 6⟩
  let growth_inches : ℕ := 6
  new_height_after_growth initial_height growth_inches = 72 := by
  sorry

end steve_height_after_growth_l457_45706


namespace oplus_comm_l457_45778

def oplus (a b : ℕ+) : ℕ+ := a ^ b.val + b ^ a.val

theorem oplus_comm (a b : ℕ+) : oplus a b = oplus b a := by
  sorry

end oplus_comm_l457_45778


namespace lcm_12_18_30_l457_45721

theorem lcm_12_18_30 : Nat.lcm 12 (Nat.lcm 18 30) = 180 := by
  sorry

end lcm_12_18_30_l457_45721


namespace binomial_expansion_example_l457_45765

theorem binomial_expansion_example : 
  57^4 + 4*(57^3 * 2) + 6*(57^2 * 2^2) + 4*(57 * 2^3) + 2^4 = 12117361 := by
  sorry

end binomial_expansion_example_l457_45765


namespace a_range_l457_45773

-- Define the sets A and B
def A : Set ℝ := {x : ℝ | x^2 ≤ 1}
def B (a : ℝ) : Set ℝ := {x : ℝ | x ≤ a}

-- State the theorem
theorem a_range (a : ℝ) : A ∪ B a = B a → a ≥ 1 := by
  sorry

end a_range_l457_45773


namespace line_l1_parallel_line_l1_perpendicular_l457_45753

-- Define the line l passing through points A(4,0) and B(0,3)
def line_l (x y : ℝ) : Prop := x / 4 + y / 3 = 1

-- Define the two given lines
def line1 (x y : ℝ) : Prop := 3 * x + y = 0
def line2 (x y : ℝ) : Prop := x + y = 2

-- Define the intersection point of line1 and line2
def intersection_point (x y : ℝ) : Prop := line1 x y ∧ line2 x y

-- Define a parallel line to l
def parallel_line (m : ℝ) (x y : ℝ) : Prop := x / 4 + y / 3 = m

-- Define a perpendicular line to l
def perpendicular_line (n : ℝ) (x y : ℝ) : Prop := x / 3 - y / 4 = n

theorem line_l1_parallel :
  ∃ m : ℝ, (∀ x y : ℝ, intersection_point x y → parallel_line m x y) ∧
           (∀ x y : ℝ, parallel_line m x y ↔ 3 * x + 4 * y - 9 = 0) :=
sorry

theorem line_l1_perpendicular :
  ∃ n1 n2 : ℝ, n1 ≠ n2 ∧
    (∀ x y : ℝ, perpendicular_line n1 x y ↔ 4 * x - 3 * y - 12 = 0) ∧
    (∀ x y : ℝ, perpendicular_line n2 x y ↔ 4 * x - 3 * y + 12 = 0) ∧
    (∀ n : ℝ, n = n1 ∨ n = n2 →
      ∃ x1 y1 x2 y2 : ℝ,
        perpendicular_line n x1 0 ∧ perpendicular_line n 0 y2 ∧
        x1 * y2 / 2 = 6) :=
sorry

end line_l1_parallel_line_l1_perpendicular_l457_45753


namespace smallest_factor_of_32_with_sum_3_l457_45741

theorem smallest_factor_of_32_with_sum_3 (a b c : Int) : 
  a * b * c = 32 → a + b + c = 3 → min a (min b c) = -4 := by
  sorry

end smallest_factor_of_32_with_sum_3_l457_45741


namespace z1_div_z2_equals_one_minus_two_i_l457_45704

-- Define complex numbers z1 and z2
def z1 : ℂ := Complex.mk 2 1
def z2 : ℂ := Complex.mk 0 1

-- Theorem statement
theorem z1_div_z2_equals_one_minus_two_i :
  z1 / z2 = Complex.mk 1 (-2) :=
sorry

end z1_div_z2_equals_one_minus_two_i_l457_45704


namespace square_root_equality_implies_zero_product_l457_45703

theorem square_root_equality_implies_zero_product (x y z : ℝ) 
  (h : Real.sqrt (x - y + z) = Real.sqrt x - Real.sqrt y + Real.sqrt z) : 
  (x - y) * (y - z) * (z - x) = 0 := by
  sorry

end square_root_equality_implies_zero_product_l457_45703


namespace election_votes_l457_45710

theorem election_votes (total_votes : ℕ) (invalid_percent : ℚ) (excess_percent : ℚ) 
  (h_total : total_votes = 6720)
  (h_invalid : invalid_percent = 1/5)
  (h_excess : excess_percent = 3/20) :
  ∃ (votes_b : ℕ), votes_b = 2184 ∧ 
  (↑votes_b : ℚ) + (↑votes_b + excess_percent * total_votes) = (1 - invalid_percent) * total_votes :=
sorry

end election_votes_l457_45710


namespace matts_age_l457_45755

theorem matts_age (john_age matt_age : ℕ) 
  (h1 : matt_age = 4 * john_age - 3)
  (h2 : matt_age + john_age = 52) :
  matt_age = 41 := by
  sorry

end matts_age_l457_45755


namespace age_score_ratio_l457_45713

def almas_age : ℕ := 20
def melinas_age : ℕ := 60
def almas_score : ℕ := 40

theorem age_score_ratio :
  (almas_age + melinas_age) / almas_score = 2 ∧
  melinas_age = 3 * almas_age ∧
  melinas_age = 60 ∧
  almas_score = 40 := by
  sorry

end age_score_ratio_l457_45713


namespace triangle_similarity_equality_equivalence_l457_45769

/-- Two triangles are similar if their corresponding sides are proportional -/
def SimilarTriangles (a b c a₁ b₁ c₁ : ℝ) : Prop :=
  ∃ k : ℝ, k > 0 ∧ a = k * a₁ ∧ b = k * b₁ ∧ c = k * c₁

/-- The theorem stating the equivalence between triangle similarity and the given equation -/
theorem triangle_similarity_equality_equivalence
  (a b c a₁ b₁ c₁ : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0)
  (ha₁ : a₁ > 0) (hb₁ : b₁ > 0) (hc₁ : c₁ > 0) :
  SimilarTriangles a b c a₁ b₁ c₁ ↔
  Real.sqrt (a * a₁) + Real.sqrt (b * b₁) + Real.sqrt (c * c₁) =
  Real.sqrt ((a + b + c) * (a₁ + b₁ + c₁)) := by
    sorry

#check triangle_similarity_equality_equivalence

end triangle_similarity_equality_equivalence_l457_45769


namespace three_digit_divisibility_l457_45701

/-- Given a three-digit number ABC divisible by 37, prove that BCA + CAB is also divisible by 37 -/
theorem three_digit_divisibility (A B C : ℕ) (h : 37 ∣ (100 * A + 10 * B + C)) :
  37 ∣ ((100 * B + 10 * C + A) + (100 * C + 10 * A + B)) := by
  sorry

end three_digit_divisibility_l457_45701


namespace grilled_cheese_slices_l457_45716

/-- The number of ham sandwiches made -/
def num_ham_sandwiches : ℕ := 10

/-- The number of grilled cheese sandwiches made -/
def num_grilled_cheese : ℕ := 10

/-- The number of cheese slices used in one ham sandwich -/
def cheese_per_ham : ℕ := 2

/-- The total number of cheese slices used -/
def total_cheese : ℕ := 50

/-- The number of cheese slices in one grilled cheese sandwich -/
def cheese_per_grilled_cheese : ℕ := (total_cheese - num_ham_sandwiches * cheese_per_ham) / num_grilled_cheese

theorem grilled_cheese_slices :
  cheese_per_grilled_cheese = 3 :=
sorry

end grilled_cheese_slices_l457_45716


namespace n_sum_of_squares_l457_45795

theorem n_sum_of_squares (n : ℕ) (h1 : n > 2) 
  (h2 : ∃ (x : ℕ), n^2 = (x + 1)^3 - x^3) : 
  (∃ (a b : ℕ), n = a^2 + b^2) ∧ 
  (∃ (m : ℕ), m > 2 ∧ (∃ (y : ℕ), m^2 = (y + 1)^3 - y^3) ∧ (∃ (c d : ℕ), m = c^2 + d^2)) :=
by sorry

end n_sum_of_squares_l457_45795


namespace root_product_of_quartic_l457_45702

theorem root_product_of_quartic (p q r s : ℂ) : 
  (3 * p^4 - 8 * p^3 - 15 * p^2 + 10 * p - 2 = 0) →
  (3 * q^4 - 8 * q^3 - 15 * q^2 + 10 * q - 2 = 0) →
  (3 * r^4 - 8 * r^3 - 15 * r^2 + 10 * r - 2 = 0) →
  (3 * s^4 - 8 * s^3 - 15 * s^2 + 10 * s - 2 = 0) →
  p * q * r * s = 2/3 := by
sorry

end root_product_of_quartic_l457_45702


namespace evaluate_otimes_expression_l457_45759

-- Define the ⊗ operation
def otimes (a b : ℚ) : ℚ := (a^2 + b^2) / (a - b)

-- State the theorem
theorem evaluate_otimes_expression :
  (otimes (otimes 5 3) 2) = 293/15 := by sorry

end evaluate_otimes_expression_l457_45759


namespace compound_interest_rate_l457_45772

/-- Proves that given the compound interest conditions, the rate of interest is 5% -/
theorem compound_interest_rate (P R : ℝ) 
  (h1 : P * (1 + R / 100) ^ 2 = 17640)
  (h2 : P * (1 + R / 100) ^ 3 = 18522) : 
  R = 5 := by
  sorry

end compound_interest_rate_l457_45772


namespace five_topping_pizzas_l457_45747

theorem five_topping_pizzas (n k : ℕ) : n = 8 → k = 5 → Nat.choose n k = 56 := by
  sorry

end five_topping_pizzas_l457_45747


namespace sum_always_positive_l457_45726

/-- An increasing function on ℝ -/
def IncreasingFunction (f : ℝ → ℝ) : Prop :=
  ∀ x y, x < y → f x < f y

/-- An odd function -/
def OddFunction (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

/-- An arithmetic sequence -/
def ArithmeticSequence (a : ℕ → ℝ) : Prop :=
  ∃ d, ∀ n, a (n + 1) = a n + d

theorem sum_always_positive
  (f : ℝ → ℝ)
  (a : ℕ → ℝ)
  (h_incr : IncreasingFunction f)
  (h_odd : OddFunction f)
  (h_arith : ArithmeticSequence a)
  (h_a3_pos : a 3 > 0) :
  f (a 1) + f (a 3) + f (a 5) > 0 :=
sorry

end sum_always_positive_l457_45726


namespace degrees_120_45_equals_120_75_l457_45709

/-- Converts degrees and minutes to decimal degrees -/
def degreesMinutesToDecimal (degrees : ℕ) (minutes : ℕ) : ℚ :=
  degrees + (minutes : ℚ) / 60

/-- Theorem stating that 120°45' is equal to 120.75° -/
theorem degrees_120_45_equals_120_75 :
  degreesMinutesToDecimal 120 45 = 120.75 := by
  sorry

end degrees_120_45_equals_120_75_l457_45709


namespace larry_expression_equality_l457_45720

theorem larry_expression_equality (e : ℝ) : 
  let a : ℝ := 2
  let b : ℝ := 1
  let c : ℝ := 4
  let d : ℝ := 5
  (a + (b - (c + (d + e)))) = (a + b - c - d - e) :=
by sorry

end larry_expression_equality_l457_45720


namespace badminton_survey_k_squared_l457_45731

/-- Represents the contingency table for the badminton survey --/
structure ContingencyTable :=
  (male_like : ℕ)
  (male_dislike : ℕ)
  (female_like : ℕ)
  (female_dislike : ℕ)

/-- Calculates the K² statistic for a given contingency table --/
def calculate_k_squared (table : ContingencyTable) : ℚ :=
  let n := table.male_like + table.male_dislike + table.female_like + table.female_dislike
  let a := table.male_like
  let b := table.male_dislike
  let c := table.female_like
  let d := table.female_dislike
  (n * (a * d - b * c)^2 : ℚ) / ((a + b) * (c + d) * (a + c) * (b + d))

theorem badminton_survey_k_squared :
  ∀ (table : ContingencyTable),
    table.male_like + table.male_dislike = 100 →
    table.female_like + table.female_dislike = 100 →
    table.male_like = 40 →
    table.female_dislike = 90 →
    calculate_k_squared table = 24 := by
  sorry

end badminton_survey_k_squared_l457_45731


namespace magnitude_sum_perpendicular_vectors_l457_45711

/-- Given two vectors a and b in R², where a is perpendicular to b,
    prove that the magnitude of a + 2b is 2√10 -/
theorem magnitude_sum_perpendicular_vectors
  (a b : ℝ × ℝ)
  (h1 : a.1 = 4)
  (h2 : b = (1, -2))
  (h3 : a.1 * b.1 + a.2 * b.2 = 0) :
  ‖a + 2 • b‖ = 2 * Real.sqrt 10 := by
  sorry

end magnitude_sum_perpendicular_vectors_l457_45711


namespace arithmetic_sequence_length_l457_45790

theorem arithmetic_sequence_length (a₁ : ℚ) (aₙ : ℚ) (d : ℚ) (n : ℕ) 
  (h₁ : a₁ = 3.25)
  (h₂ : aₙ = 55.25)
  (h₃ : d = 4)
  (h₄ : aₙ = a₁ + (n - 1) * d) :
  n = 14 := by
sorry

end arithmetic_sequence_length_l457_45790


namespace english_teachers_count_l457_45734

def committee_size (E : ℕ) : ℕ := E + 4 + 2

def probability_two_math_teachers (E : ℕ) : ℚ :=
  6 / (committee_size E * (committee_size E - 1) / 2)

theorem english_teachers_count :
  ∃ E : ℕ, probability_two_math_teachers E = 1/12 ∧ E = 3 :=
sorry

end english_teachers_count_l457_45734


namespace voldemort_shopping_l457_45730

theorem voldemort_shopping (book_price : ℝ) (journal_price : ℝ) : 
  book_price = 8 ∧ 
  book_price = (1/8) * (book_price * 8) ∧ 
  journal_price = 2 * book_price →
  (book_price * 8 = 64) ∧ 
  (book_price + journal_price = 24) := by
sorry

end voldemort_shopping_l457_45730


namespace three_positions_from_eight_people_l457_45700

def number_of_people : ℕ := 8
def number_of_positions : ℕ := 3

theorem three_positions_from_eight_people :
  (number_of_people.factorial) / ((number_of_people - number_of_positions).factorial) = 336 :=
sorry

end three_positions_from_eight_people_l457_45700


namespace coin_stack_theorem_l457_45737

def fibonacci : ℕ → ℕ
| 0 => 0
| 1 => 1
| (n + 2) => fibonacci n + fibonacci (n + 1)

def coin_arrangements (n : ℕ) : ℕ := fibonacci (n + 2)

theorem coin_stack_theorem :
  coin_arrangements 10 = 233 :=
by sorry

end coin_stack_theorem_l457_45737


namespace circle_properties_l457_45782

/-- Given a circle with equation x^2 + y^2 = 10x - 8y + 4, prove its properties --/
theorem circle_properties :
  let equation := fun (x y : ℝ) => x^2 + y^2 = 10*x - 8*y + 4
  ∃ (center : ℝ × ℝ) (radius : ℝ),
    (center.1 = 5 ∧ center.2 = -4) ∧  -- Center is (5, -4)
    (radius = 3 * Real.sqrt 5) ∧     -- Radius is 3√5
    (center.1 + center.2 = 1) ∧      -- Sum of center coordinates is 1
    ∀ (x y : ℝ), equation x y ↔ ((x - center.1)^2 + (y - center.2)^2 = radius^2) :=
by
  sorry


end circle_properties_l457_45782


namespace symmetric_line_equation_l457_45739

-- Define the original line
def original_line (x y : ℝ) : Prop := 4 * x - 3 * y + 5 = 0

-- Define a point on the x-axis
def x_axis_point (x : ℝ) : ℝ × ℝ := (x, 0)

-- Define symmetry with respect to x-axis
def symmetric_wrt_x_axis (p q : ℝ × ℝ) : Prop :=
  p.1 = q.1 ∧ p.2 = -q.2

-- State the theorem
theorem symmetric_line_equation :
  ∀ (x y : ℝ),
  (∃ (x₀ : ℝ), original_line x₀ 0) →
  (∀ (p q : ℝ × ℝ),
    original_line p.1 p.2 →
    symmetric_wrt_x_axis p q →
    4 * q.1 + 3 * q.2 + 5 = 0) :=
sorry

end symmetric_line_equation_l457_45739


namespace min_value_on_interval_l457_45722

/-- A function f(x) = x^2 + x + a with a maximum value of 2 on [-1, 1] -/
def f (a : ℝ) : ℝ → ℝ := fun x ↦ x^2 + x + a

/-- The maximum value of f on [-1, 1] is 2 -/
axiom max_value_2 (a : ℝ) : ∃ x ∈ Set.Icc (-1) 1, ∀ y ∈ Set.Icc (-1) 1, f a x ≥ f a y ∧ f a x = 2

/-- The theorem to prove -/
theorem min_value_on_interval (a : ℝ) : 
  (∃ x ∈ Set.Icc (-1) 1, ∀ y ∈ Set.Icc (-1) 1, f a x ≥ f a y ∧ f a x = 2) →
  ∃ z ∈ Set.Icc (-1) 1, ∀ w ∈ Set.Icc (-1) 1, f a z ≤ f a w ∧ f a z = -1/4 :=
sorry

end min_value_on_interval_l457_45722


namespace abs_diff_plus_smaller_l457_45786

theorem abs_diff_plus_smaller (a b : ℝ) (h : a > b) : |a - b| + b = a := by
  sorry

end abs_diff_plus_smaller_l457_45786


namespace set_operations_l457_45758

def U : Set ℝ := {x | x ≤ 5}
def A : Set ℝ := {x | -2 < x ∧ x < 3}
def B : Set ℝ := {x | -4 < x ∧ x ≤ 2}

theorem set_operations :
  (A ∩ B = {x | -2 < x ∧ x ≤ 2}) ∧
  (A ∩ (U \ B) = {x | 2 < x ∧ x < 3}) ∧
  ((U \ A) ∪ B = {x | x ≤ 2 ∨ (3 ≤ x ∧ x ≤ 5)}) ∧
  ((U \ A) ∪ (U \ B) = {x | x ≤ -2 ∨ (2 < x ∧ x ≤ 5)}) :=
by sorry

end set_operations_l457_45758


namespace evaluate_expression_l457_45735

theorem evaluate_expression : 11 + Real.sqrt (-4 + 6 * 4 / 3) = 13 := by
  sorry

end evaluate_expression_l457_45735


namespace green_eyed_students_l457_45788

theorem green_eyed_students (total : ℕ) (both : ℕ) (neither : ℕ) :
  total = 50 →
  both = 10 →
  neither = 5 →
  ∃ (green : ℕ),
    green * 2 = (total - both - neither) - green ∧
    green = 15 := by
  sorry

end green_eyed_students_l457_45788


namespace barn_paint_area_l457_45797

/-- Calculates the total area to be painted in a rectangular barn -/
def total_paint_area (width length height : ℝ) : ℝ :=
  2 * (2 * width * height + 2 * length * height) + 2 * width * length

/-- Theorem stating the total area to be painted for a specific barn -/
theorem barn_paint_area :
  let width : ℝ := 15
  let length : ℝ := 20
  let height : ℝ := 8
  total_paint_area width length height = 1720 := by
  sorry

end barn_paint_area_l457_45797


namespace specific_cone_properties_l457_45784

/-- Represents a cone with given height and slant height -/
structure Cone where
  height : ℝ
  slant_height : ℝ

/-- The central angle (in degrees) of the unfolded lateral surface of a cone -/
def central_angle (c : Cone) : ℝ := sorry

/-- The lateral surface area of a cone -/
def lateral_surface_area (c : Cone) : ℝ := sorry

/-- Theorem stating the properties of a specific cone -/
theorem specific_cone_properties :
  let c := Cone.mk (2 * Real.sqrt 2) 3
  central_angle c = 120 ∧ lateral_surface_area c = 3 * Real.pi := by sorry

end specific_cone_properties_l457_45784


namespace intersection_locus_l457_45774

/-- The locus of intersection points of two lines passing through fixed points on the x-axis and intersecting a parabola at four concyclic points. -/
theorem intersection_locus (a b : ℝ) (h : 0 < a ∧ a < b) :
  ∀ (l m : ℝ → ℝ → Prop) (P : ℝ × ℝ),
    (∀ y, l a y ↔ y = 0) →  -- Line l passes through (a, 0)
    (∀ y, m b y ↔ y = 0) →  -- Line m passes through (b, 0)
    (∃ x₁ y₁ x₂ y₂ x₃ y₃ x₄ y₄,  -- Four distinct intersection points
      x₁ ≠ x₂ ∧ x₁ ≠ x₃ ∧ x₁ ≠ x₄ ∧ x₂ ≠ x₃ ∧ x₂ ≠ x₄ ∧ x₃ ≠ x₄ ∧
      l x₁ y₁ ∧ l x₂ y₂ ∧ m x₃ y₃ ∧ m x₄ y₄ ∧
      y₁^2 = x₁ ∧ y₂^2 = x₂ ∧ y₃^2 = x₃ ∧ y₄^2 = x₄ ∧
      ∃ (r : ℝ) (c : ℝ × ℝ), -- Points are concyclic
        (x₁ - c.1)^2 + (y₁ - c.2)^2 = r^2 ∧
        (x₂ - c.1)^2 + (y₂ - c.2)^2 = r^2 ∧
        (x₃ - c.1)^2 + (y₃ - c.2)^2 = r^2 ∧
        (x₄ - c.1)^2 + (y₄ - c.2)^2 = r^2) →
    (∀ x y, l x y ∧ m x y → P = (x, y)) →  -- P is the intersection of l and m
    P.1 = (a + b) / 2  -- The x-coordinate of P satisfies 2x - (a + b) = 0
  := by sorry

end intersection_locus_l457_45774


namespace solution_characterization_l457_45792

def f (p : ℕ × ℕ) : ℕ × ℕ :=
  (p.2, 5 * p.2 - p.1)

def h (p : ℕ × ℕ) : ℕ × ℕ :=
  (p.2, p.1)

def solution_set : Set (ℕ × ℕ) :=
  {(1, 2), (1, 3), (2, 1), (3, 1)} ∪
  {p | ∃ n : ℕ, p = Nat.iterate f n (1, 2) ∨ p = Nat.iterate f n (1, 3)} ∪
  {p | ∃ n : ℕ, p = h (Nat.iterate f n (1, 2)) ∨ p = h (Nat.iterate f n (1, 3))}

theorem solution_characterization :
  ∀ x y : ℕ, x > 0 ∧ y > 0 →
  (x^2 + y^2 - 5*x*y + 5 = 0 ↔ (x, y) ∈ solution_set) :=
by sorry

end solution_characterization_l457_45792


namespace board_tiling_divisibility_l457_45725

/-- Represents a square on the board -/
structure Square where
  row : Nat
  col : Nat

/-- Represents a domino placement -/
inductive Domino
  | horizontal : Square → Domino
  | vertical : Square → Domino

/-- Represents a tiling of the board -/
def Tiling := List Domino

/-- Represents an assignment of integers to squares -/
def Assignment := Square → Int

/-- Checks if a tiling is valid for a 2n × 2n board -/
def is_valid_tiling (n : Nat) (t : Tiling) : Prop := sorry

/-- Checks if an assignment satisfies the neighbor condition -/
def satisfies_neighbor_condition (n : Nat) (red_tiling blue_tiling : Tiling) (assignment : Assignment) : Prop := sorry

theorem board_tiling_divisibility (n : Nat) 
  (red_tiling blue_tiling : Tiling) 
  (h_red_valid : is_valid_tiling n red_tiling)
  (h_blue_valid : is_valid_tiling n blue_tiling)
  (assignment : Assignment)
  (h_nonzero : ∀ s, assignment s ≠ 0)
  (h_satisfies : satisfies_neighbor_condition n red_tiling blue_tiling assignment) :
  3 ∣ n := by
  sorry

end board_tiling_divisibility_l457_45725


namespace hyperbola_equation_l457_45752

/-- 
Given a hyperbola with equation (x²/a² - y²/b² = 1) that passes through the point (√2, √3) 
and has eccentricity 2, prove that its equation is x² - y²/3 = 1.
-/
theorem hyperbola_equation (a b : ℝ) (h1 : a > 0) (h2 : b > 0) : 
  (2 / a^2 - 3 / b^2 = 1) →  -- The hyperbola passes through (√2, √3)
  ((a^2 + b^2) / a^2 = 4) →  -- The eccentricity is 2
  (∀ x y : ℝ, x^2 / a^2 - y^2 / b^2 = 1 ↔ x^2 - y^2 / 3 = 1) := by
sorry

end hyperbola_equation_l457_45752


namespace opposite_roots_implies_n_eq_neg_two_l457_45754

/-- The equation has roots that are numerically equal but of opposite signs -/
def has_opposite_roots (k b d e n : ℝ) : Prop :=
  ∃ x : ℝ, (k * x^2 - b * x) / (d * x - e) = (n - 2) / (n + 2) ∧
            ∃ y : ℝ, y = -x ∧ (k * y^2 - b * y) / (d * y - e) = (n - 2) / (n + 2)

/-- Theorem stating that if the equation has roots that are numerically equal but of opposite signs, then n = -2 -/
theorem opposite_roots_implies_n_eq_neg_two (k b d e : ℝ) :
  ∀ n : ℝ, has_opposite_roots k b d e n → n = -2 :=
by sorry

end opposite_roots_implies_n_eq_neg_two_l457_45754


namespace simultaneous_colonies_count_l457_45738

/-- Represents the growth of bacteria colonies over time -/
def bacteriaGrowth (n : ℕ) (t : ℕ) : ℕ := n * 2^t

/-- The number of days it takes for a single colony to reach the habitat limit -/
def singleColonyLimit : ℕ := 25

/-- The number of days it takes for multiple colonies to reach the habitat limit -/
def multipleColoniesLimit : ℕ := 24

/-- Theorem stating that the number of simultaneously growing colonies is 2 -/
theorem simultaneous_colonies_count :
  ∃ (n : ℕ), n > 0 ∧ 
    bacteriaGrowth n multipleColoniesLimit = bacteriaGrowth 1 singleColonyLimit ∧ 
    n = 2 := by
  sorry

end simultaneous_colonies_count_l457_45738


namespace two_car_garage_count_l457_45785

theorem two_car_garage_count (total : ℕ) (pool : ℕ) (both : ℕ) (neither : ℕ) :
  total = 70 →
  pool = 40 →
  both = 35 →
  neither = 15 →
  ∃ garage : ℕ, garage = 50 ∧ garage + pool - both + neither = total :=
by sorry

end two_car_garage_count_l457_45785


namespace quarterback_no_throw_percentage_l457_45789

/-- Given a quarterback's statistics in a game, calculate the percentage of time he doesn't throw a pass. -/
theorem quarterback_no_throw_percentage 
  (total_attempts : ℕ) 
  (sacks : ℕ) 
  (h1 : total_attempts = 80) 
  (h2 : sacks = 12) 
  (h3 : 2 * sacks = total_attempts - (total_attempts - 2 * sacks)) : 
  (2 * sacks : ℚ) / total_attempts = 3 / 10 := by
  sorry

end quarterback_no_throw_percentage_l457_45789


namespace stating_tour_cost_is_correct_l457_45715

/-- Represents the cost of a tour at an aqua park -/
def tour_cost : ℝ := 6

/-- Represents the admission fee for the aqua park -/
def admission_fee : ℝ := 12

/-- Represents the total number of people in the first group -/
def group1_size : ℕ := 10

/-- Represents the total number of people in the second group -/
def group2_size : ℕ := 5

/-- Represents the total earnings of the aqua park -/
def total_earnings : ℝ := 240

/-- 
Theorem stating that the tour cost is correct given the problem conditions
-/
theorem tour_cost_is_correct : 
  group1_size * (admission_fee + tour_cost) + group2_size * admission_fee = total_earnings :=
by sorry

end stating_tour_cost_is_correct_l457_45715


namespace perpendicular_vectors_l457_45736

/-- The value of k for which vectors a and b are perpendicular --/
theorem perpendicular_vectors (i j a b : ℝ × ℝ) (k : ℝ) : 
  i = (1, 0) →
  j = (0, 1) →
  a = (2 * i.1 + 0 * i.2, 0 * j.1 + 3 * j.2) →
  b = (k * i.1 + 0 * i.2, 0 * j.1 + (-4) * j.2) →
  a.1 * b.1 + a.2 * b.2 = 0 →
  k = 6 := by
sorry

end perpendicular_vectors_l457_45736


namespace largest_integer_with_remainder_l457_45757

theorem largest_integer_with_remainder : ∃ n : ℕ, n < 100 ∧ n % 7 = 5 ∧ ∀ m : ℕ, m < 100 ∧ m % 7 = 5 → m ≤ n :=
  by sorry

end largest_integer_with_remainder_l457_45757


namespace right_triangle_count_l457_45780

/-- Represents a right triangle with integer vertices and right angle at the origin -/
structure RightTriangle where
  a : ℤ × ℤ
  b : ℤ × ℤ

/-- Checks if a point is the incenter of a right triangle -/
def is_incenter (t : RightTriangle) (m : ℚ × ℚ) : Prop :=
  sorry

/-- Counts the number of right triangles with given incenter -/
def count_triangles (p : ℕ) : ℕ :=
  sorry

theorem right_triangle_count (p : ℕ) (h : Nat.Prime p) :
  count_triangles p = 108 ∨ count_triangles p = 42 ∨ count_triangles p = 60 := by
  sorry

end right_triangle_count_l457_45780


namespace john_post_break_time_l457_45745

/-- The number of hours John danced before the break -/
def john_pre_break : ℝ := 3

/-- The number of hours John took for break -/
def john_break : ℝ := 1

/-- The total dancing time of both John and James (excluding John's break) -/
def total_dance_time : ℝ := 20

/-- The number of hours John danced after the break -/
def john_post_break : ℝ := 5

theorem john_post_break_time : 
  john_post_break = 
    (total_dance_time - john_pre_break - 
      (john_pre_break + john_break + john_post_break + 
        (1/3) * (john_pre_break + john_break + john_post_break))) / 
    (7/3) := by sorry

end john_post_break_time_l457_45745


namespace union_A_complement_B_l457_45748

def I : Set ℤ := {x | |x| < 3}
def A : Set ℤ := {1, 2}
def B : Set ℤ := {-2, -1, 2}

theorem union_A_complement_B : A ∪ (I \ B) = {0, 1, 2} := by sorry

end union_A_complement_B_l457_45748


namespace wang_gang_seat_location_l457_45798

/-- Represents a seat in a classroom -/
structure Seat where
  row : Nat
  column : Nat

/-- Represents a classroom -/
structure Classroom where
  rows : Nat
  columns : Nat

/-- Checks if a seat is valid for a given classroom -/
def is_valid_seat (c : Classroom) (s : Seat) : Prop :=
  s.row ≤ c.rows ∧ s.column ≤ c.columns

theorem wang_gang_seat_location (c : Classroom) (s : Seat) :
  c.rows = 7 ∧ c.columns = 8 ∧ s = Seat.mk 5 8 ∧ is_valid_seat c s →
  s.row = 5 ∧ s.column = 8 := by
  sorry

end wang_gang_seat_location_l457_45798


namespace line_translation_l457_45733

theorem line_translation (x : ℝ) : 
  let original_line := λ x : ℝ => x / 3
  let translated_line := λ x : ℝ => (x + 5) / 3
  translated_line x - original_line x = 5 / 3 := by
  sorry

end line_translation_l457_45733


namespace first_divisible_by_three_and_seven_l457_45724

theorem first_divisible_by_three_and_seven (lower_bound upper_bound : ℕ) 
  (h_lower : lower_bound = 100) (h_upper : upper_bound = 600) :
  ∃ (n : ℕ), 
    n ≥ lower_bound ∧ 
    n ≤ upper_bound ∧ 
    n % 3 = 0 ∧ 
    n % 7 = 0 ∧
    ∀ (m : ℕ), m ≥ lower_bound ∧ m < n → m % 3 ≠ 0 ∨ m % 7 ≠ 0 :=
by
  -- Proof goes here
  sorry

#eval (105 : ℕ)

end first_divisible_by_three_and_seven_l457_45724


namespace tangent_condition_l457_45743

/-- A line with equation kx - y - 3√2 = 0 is tangent to the circle x² + y² = 9 -/
def is_tangent (k : ℝ) : Prop :=
  ∃ (x y : ℝ), k*x - y - 3*Real.sqrt 2 = 0 ∧ x^2 + y^2 = 9 ∧
  ∀ (x' y' : ℝ), k*x' - y' - 3*Real.sqrt 2 = 0 → x'^2 + y'^2 ≥ 9

/-- k = 1 is a sufficient but not necessary condition for the line to be tangent -/
theorem tangent_condition : 
  (is_tangent 1) ∧ (∃ (k : ℝ), k ≠ 1 ∧ is_tangent k) :=
sorry

end tangent_condition_l457_45743


namespace principal_amount_satisfies_conditions_l457_45705

/-- The principal amount that satisfies the given conditions -/
def principal_amount : ℝ := 6400

/-- The annual interest rate -/
def interest_rate : ℝ := 0.05

/-- The time period in years -/
def time_period : ℝ := 2

/-- The difference between compound interest and simple interest -/
def interest_difference : ℝ := 16

/-- Theorem stating that the principal amount satisfies the given conditions -/
theorem principal_amount_satisfies_conditions :
  let compound_interest := principal_amount * (1 + interest_rate) ^ time_period - principal_amount
  let simple_interest := principal_amount * interest_rate * time_period
  compound_interest - simple_interest = interest_difference :=
by sorry

end principal_amount_satisfies_conditions_l457_45705


namespace sqrt_a_plus_one_real_l457_45771

theorem sqrt_a_plus_one_real (a : ℝ) : (∃ (x : ℝ), x ^ 2 = a + 1) ↔ a ≥ -1 := by sorry

end sqrt_a_plus_one_real_l457_45771


namespace isabellas_hair_growth_l457_45787

/-- Given Isabella's initial and final hair lengths, prove that her hair growth is 6 inches. -/
theorem isabellas_hair_growth 
  (initial_length : ℝ) 
  (final_length : ℝ) 
  (h1 : initial_length = 18) 
  (h2 : final_length = 24) : 
  final_length - initial_length = 6 := by
  sorry

end isabellas_hair_growth_l457_45787


namespace trip_average_speed_l457_45783

/-- Calculates the average speed given three segments of a trip -/
def average_speed (d1 d2 d3 t1 t2 t3 : ℚ) : ℚ :=
  (d1 + d2 + d3) / (t1 + t2 + t3)

/-- Theorem: The average speed for the given trip is 1200/18 miles per hour -/
theorem trip_average_speed :
  average_speed 420 480 300 6 7 5 = 1200 / 18 := by
  sorry

end trip_average_speed_l457_45783


namespace tom_balloons_l457_45727

theorem tom_balloons (initial_balloons : ℕ) (given_balloons : ℕ) : 
  initial_balloons = 30 → given_balloons = 16 → initial_balloons - given_balloons = 14 := by
  sorry

end tom_balloons_l457_45727


namespace gcd_n_cube_plus_25_and_n_plus_3_l457_45750

theorem gcd_n_cube_plus_25_and_n_plus_3 (n : ℕ) (h : n > 9) :
  Nat.gcd (n^3 + 25) (n + 3) = if n % 2 = 1 then 2 else 1 := by
  sorry

end gcd_n_cube_plus_25_and_n_plus_3_l457_45750


namespace ditch_length_greater_than_70_l457_45799

/-- Represents a square field with irrigation ditches -/
structure IrrigatedField where
  side_length : ℝ
  ditch_length : ℝ
  max_distance_to_ditch : ℝ

/-- Theorem stating that the total length of ditches in the irrigated field is greater than 70 units -/
theorem ditch_length_greater_than_70 (field : IrrigatedField) 
  (h1 : field.side_length = 12)
  (h2 : field.max_distance_to_ditch ≤ 1) :
  field.ditch_length > 70 := by
  sorry

end ditch_length_greater_than_70_l457_45799


namespace negation_of_universal_statement_l457_45777

theorem negation_of_universal_statement :
  ¬(∀ a : ℝ, ∃ x : ℝ, x > 0 ∧ a * x^2 - 3 * x - a = 0) ↔
  (∃ a : ℝ, ∀ x : ℝ, x > 0 → a * x^2 - 3 * x - a ≠ 0) :=
by sorry

end negation_of_universal_statement_l457_45777


namespace sin_plus_tan_10_deg_l457_45751

theorem sin_plus_tan_10_deg : 
  Real.sin (10 * π / 180) + (Real.sqrt 3 / 4) * Real.tan (10 * π / 180) = 1 / 4 := by
  sorry

end sin_plus_tan_10_deg_l457_45751


namespace fraction_sum_to_decimal_l457_45779

theorem fraction_sum_to_decimal : (9 : ℚ) / 10 + (8 : ℚ) / 100 = (98 : ℚ) / 100 := by sorry

end fraction_sum_to_decimal_l457_45779


namespace area_between_tangents_and_curve_l457_45707

noncomputable section

-- Define the curve
def C (x : ℝ) : ℝ := 1 / x

-- Define the points P and Q
def P (a : ℝ) : ℝ × ℝ := (a, C a)
def Q (a : ℝ) : ℝ × ℝ := (2*a, C (2*a))

-- Define the tangent lines at P and Q
def l (a : ℝ) (x : ℝ) : ℝ := -1/(a^2) * x + 2/a
def m (a : ℝ) (x : ℝ) : ℝ := -1/(4*a^2) * x + 1/a

-- Define the area function
def area (a : ℝ) : ℝ :=
  ∫ x in a..(2*a), (C x - l a x) + (C x - m a x)

-- State the theorem
theorem area_between_tangents_and_curve (a : ℝ) (h : a > 0) :
  area a = 2 * Real.log 2 - 9/8 :=
sorry

end

end area_between_tangents_and_curve_l457_45707


namespace smallest_b_value_l457_45728

theorem smallest_b_value (a b : ℝ) : 
  (2 < a ∧ a < b) →
  (2 + a ≤ b) →
  (1/a + 1/b ≤ 1/2) →
  b ≥ (7 + Real.sqrt 17) / 4 :=
by sorry

end smallest_b_value_l457_45728


namespace money_division_l457_45732

/-- The problem of dividing money among A, B, and C -/
theorem money_division (a b c : ℚ) : 
  a + b + c = 720 →  -- Total amount is $720
  a = (1/3) * (b + c) →  -- A gets 1/3 of what B and C get
  b = (2/7) * (a + c) →  -- B gets 2/7 of what A and C get
  a > b →  -- A receives more than B
  a - b = 20 :=  -- Prove that A receives $20 more than B
by sorry

end money_division_l457_45732


namespace complex_number_sum_of_parts_l457_45761

theorem complex_number_sum_of_parts (a : ℝ) :
  let z : ℂ := a / (2 + Complex.I) + (2 + Complex.I) / 5
  (z.re + z.im = 1) → a = 2 := by
  sorry

end complex_number_sum_of_parts_l457_45761


namespace sum_unchanged_l457_45794

theorem sum_unchanged (a b c : ℤ) (h : a + b + c = 1281) :
  (a - 329) + (b + 401) + (c - 72) = 1281 := by
sorry

end sum_unchanged_l457_45794


namespace square_sum_lower_bound_l457_45756

theorem square_sum_lower_bound (x y : ℝ) (h : |x - 2*y| = 5) : x^2 + y^2 ≥ 5 := by
  sorry

end square_sum_lower_bound_l457_45756


namespace notebooks_divisible_by_three_l457_45793

/-- A family is preparing backpacks with school supplies. -/
structure SchoolSupplies where
  pencils : ℕ
  notebooks : ℕ
  backpacks : ℕ

/-- The conditions of the problem -/
def problem_conditions (s : SchoolSupplies) : Prop :=
  s.pencils = 9 ∧
  s.backpacks = 3 ∧
  s.pencils % s.backpacks = 0 ∧
  s.notebooks % s.backpacks = 0

/-- Theorem stating that the number of notebooks must be divisible by 3 -/
theorem notebooks_divisible_by_three (s : SchoolSupplies) 
  (h : problem_conditions s) : 
  s.notebooks % 3 = 0 := by
  sorry

end notebooks_divisible_by_three_l457_45793


namespace minimal_force_to_submerge_cube_l457_45742

-- Define constants
def cube_volume : Real := 10e-6  -- 10 cm³ in m³
def cube_density : Real := 500   -- kg/m³
def water_density : Real := 1000 -- kg/m³
def gravity : Real := 10         -- m/s²

-- Define the minimal force function
def minimal_force (v : Real) (ρ_cube : Real) (ρ_water : Real) (g : Real) : Real :=
  (ρ_water - ρ_cube) * v * g

-- Theorem statement
theorem minimal_force_to_submerge_cube :
  minimal_force cube_volume cube_density water_density gravity = 0.05 := by
  sorry

end minimal_force_to_submerge_cube_l457_45742


namespace linear_equation_exponent_sum_l457_45768

theorem linear_equation_exponent_sum (a b : ℝ) : 
  (∀ x y : ℝ, ∃ k m : ℝ, 4*x^(a+b) - 3*y^(3*a+2*b-4) = k*x + m*y + 2) → 
  a + b = 1 := by
sorry

end linear_equation_exponent_sum_l457_45768


namespace triangle_area_l457_45770

/-- The area of a triangle with base 7 units and height 3 units is 10.5 square units. -/
theorem triangle_area : 
  let base : ℝ := 7
  let height : ℝ := 3
  let area : ℝ := (1/2) * base * height
  area = 10.5 := by sorry

end triangle_area_l457_45770


namespace equation_solutions_l457_45740

theorem equation_solutions :
  (∃ x : ℚ, x + 2 * (x - 3) = 3 * (1 - x) ∧ x = 3/2) ∧
  (∃ x : ℚ, 1 - (2*x - 1)/3 = (3 + x)/6 ∧ x = 1) := by
  sorry

end equation_solutions_l457_45740


namespace percentage_difference_l457_45749

theorem percentage_difference (x y : ℝ) (h : x = 7 * y) :
  (1 - y / x) * 100 = (1 - 1 / 7) * 100 := by
  sorry

end percentage_difference_l457_45749


namespace paint_for_similar_statues_l457_45796

theorem paint_for_similar_statues
  (original_height : ℝ)
  (original_paint : ℝ)
  (new_height : ℝ)
  (num_statues : ℝ)
  (h1 : original_height = 6)
  (h2 : original_paint = 1)
  (h3 : new_height = 1)
  (h4 : num_statues = 540)
  : (num_statues * new_height^2 * original_paint) / original_height^2 = 15 :=
by sorry

end paint_for_similar_statues_l457_45796


namespace complex_equation_solution_l457_45760

theorem complex_equation_solution :
  ∃ (z : ℂ), ∃ (a b : ℝ),
    z = Complex.mk a b ∧
    z * (z + Complex.I) * (z + 2 * Complex.I) = 1800 * Complex.I ∧
    a = 20.75 := by
  sorry

end complex_equation_solution_l457_45760


namespace division_remainder_proof_l457_45714

theorem division_remainder_proof (dividend : ℕ) (divisor : ℕ) (quotient : ℕ) (remainder : ℕ) :
  dividend = 265 →
  divisor = 22 →
  quotient = 12 →
  dividend = divisor * quotient + remainder →
  remainder = 1 := by
  sorry

end division_remainder_proof_l457_45714


namespace equation_solutions_l457_45781

theorem equation_solutions : 
  ∀ x : ℝ, x^4 + (3 - x)^4 = 146 ↔ 
  x = 1.5 + Real.sqrt 3.4175 ∨ x = 1.5 - Real.sqrt 3.4175 := by
sorry

end equation_solutions_l457_45781


namespace floor_counterexamples_l457_45791

-- Define the floor function
noncomputable def floor (x : ℝ) : ℤ :=
  Int.floor x

-- Statement of the theorem
theorem floor_counterexamples : ∃ (x y : ℝ),
  (floor (2^x) ≠ floor (2^(floor x))) ∧
  (floor (y^2) ≠ (floor y)^2) := by
  sorry

end floor_counterexamples_l457_45791
