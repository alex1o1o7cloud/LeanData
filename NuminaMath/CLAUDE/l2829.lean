import Mathlib

namespace new_students_count_l2829_282988

/-- The number of new students who joined Hendrix's class -/
def new_students : ℕ :=
  let initial_students : ℕ := 160
  let final_students : ℕ := 120
  let transfer_ratio : ℚ := 1/3
  let total_after_join : ℕ := final_students * 3 / 2
  total_after_join - initial_students

theorem new_students_count : new_students = 20 := by sorry

end new_students_count_l2829_282988


namespace combination_sum_l2829_282985

theorem combination_sum : Nat.choose 5 2 + Nat.choose 5 3 = 20 := by
  sorry

end combination_sum_l2829_282985


namespace inequality_solution_transformation_l2829_282935

theorem inequality_solution_transformation (a c : ℝ) :
  (∀ x : ℝ, ax^2 + 2*x + c > 0 ↔ -1/3 < x ∧ x < 1/2) →
  (∀ x : ℝ, -c*x^2 + 2*x - a > 0 ↔ -2 < x ∧ x < 3) :=
by sorry

end inequality_solution_transformation_l2829_282935


namespace school_students_count_l2829_282995

theorem school_students_count (total : ℕ) (difference : ℕ) (boys : ℕ) : 
  total = 650 →
  difference = 106 →
  boys + (boys + difference) = total →
  boys = 272 := by
sorry

end school_students_count_l2829_282995


namespace complex_fraction_equals_negative_two_l2829_282916

theorem complex_fraction_equals_negative_two :
  let z : ℂ := 1 + I
  (z^2) / (1 - z) = -2 := by sorry

end complex_fraction_equals_negative_two_l2829_282916


namespace expression_value_l2829_282990

theorem expression_value : 
  let a : ℝ := Real.sqrt 3 - Real.sqrt 2
  let b : ℝ := Real.sqrt 3 + Real.sqrt 2
  (a + 2*b)^2 + (a + 2*b)*(a - 2*b) + 2*a*(b - a) = 6 := by
  sorry

end expression_value_l2829_282990


namespace inequality_and_equality_condition_l2829_282982

theorem inequality_and_equality_condition (a b c d : ℝ) 
  (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0) :
  (a * b * c + a * b * d + a * c * d + b * c * d) / 4 ≤ 
    ((a * b + a * c + a * d + b * c + b * d + c * d) / 6) ^ (3/2) ∧
  ((a * b * c + a * b * d + a * c * d + b * c * d) / 4 = 
    ((a * b + a * c + a * d + b * c + b * d + c * d) / 6) ^ (3/2) ↔ 
      a = b ∧ b = c ∧ c = d) :=
by sorry

end inequality_and_equality_condition_l2829_282982


namespace power_of_two_start_with_any_digits_l2829_282977

theorem power_of_two_start_with_any_digits :
  ∀ A : ℕ, ∃ n m : ℕ+, (10 ^ m.val : ℝ) * A < (2 ^ n.val : ℝ) ∧ (2 ^ n.val : ℝ) < (10 ^ m.val : ℝ) * (A + 1) := by
  sorry

end power_of_two_start_with_any_digits_l2829_282977


namespace jay_and_paul_distance_l2829_282905

/-- The distance between two people walking in opposite directions -/
def distance_apart (jay_speed : ℚ) (paul_speed : ℚ) (time : ℚ) : ℚ :=
  jay_speed * time + paul_speed * time

/-- Theorem: Jay and Paul's distance apart after 2 hours -/
theorem jay_and_paul_distance : 
  let jay_speed : ℚ := 1 / 20 -- miles per minute
  let paul_speed : ℚ := 3 / 40 -- miles per minute
  let time : ℚ := 120 -- minutes (2 hours)
  distance_apart jay_speed paul_speed time = 15
  := by sorry

end jay_and_paul_distance_l2829_282905


namespace point_c_value_l2829_282979

/-- Represents a point on a number line --/
structure Point where
  value : ℝ

/-- The distance between two points on a number line --/
def distance (p q : Point) : ℝ := |p.value - q.value|

theorem point_c_value (a b c : Point) :
  a.value = -1 →
  distance a b = 11 →
  b.value > a.value →
  distance b c = 5 →
  c.value = 5 ∨ c.value = -5 := by
  sorry

end point_c_value_l2829_282979


namespace a_44_mod_45_l2829_282949

/-- Definition of a_n as the integer obtained by writing all integers from 1 to n from left to right -/
def a (n : ℕ) : ℕ := sorry

/-- Theorem stating that the remainder when a_44 is divided by 45 is 9 -/
theorem a_44_mod_45 : a 44 % 45 = 9 := by sorry

end a_44_mod_45_l2829_282949


namespace geometric_sequence_property_l2829_282993

/-- A geometric sequence -/
def GeometricSequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = a n * r

/-- Theorem: In a geometric sequence, if a₂ * a₈ = 16, then a₁ * a₉ = 16 -/
theorem geometric_sequence_property (a : ℕ → ℝ) (h : GeometricSequence a) 
    (h_prod : a 2 * a 8 = 16) : a 1 * a 9 = 16 := by
  sorry


end geometric_sequence_property_l2829_282993


namespace smallest_prime_divisor_of_sum_l2829_282931

theorem smallest_prime_divisor_of_sum : ∃ (p : ℕ), p.Prime ∧ p ∣ (3^19 + 11^23) ∧ ∀ (q : ℕ), q.Prime → q ∣ (3^19 + 11^23) → p ≤ q :=
by sorry

end smallest_prime_divisor_of_sum_l2829_282931


namespace checkered_board_division_l2829_282940

theorem checkered_board_division (n : ℕ) : 
  (∃ m : ℕ, n^2 = 9 + 7*m) ∧ 
  (∃ k : ℕ, n = 7*k + 3) ↔ 
  n % 7 = 3 :=
sorry

end checkered_board_division_l2829_282940


namespace min_value_theorem_l2829_282904

/-- The minimum value of a specific function given certain conditions -/
theorem min_value_theorem (a b c x y z : ℝ) 
  (h_pos : a > 0 ∧ b > 0 ∧ c > 0 ∧ x > 0 ∧ y > 0 ∧ z > 0) 
  (h1 : c * y + b * z = a) 
  (h2 : a * z + c * x = b) 
  (h3 : b * x + a * y = c) : 
  (∃ (m : ℝ), m = (x^2 / (1 + x) + y^2 / (1 + y) + z^2 / (1 + z)) ∧ 
   ∀ (x' y' z' : ℝ), x' > 0 → y' > 0 → z' > 0 → 
   x'^2 / (1 + x') + y'^2 / (1 + y') + z'^2 / (1 + z') ≥ m) ∧ 
  (x^2 / (1 + x) + y^2 / (1 + y) + z^2 / (1 + z) = 1/2) := by
sorry

end min_value_theorem_l2829_282904


namespace shelter_ratio_l2829_282983

/-- 
Given a shelter with dogs and cats, prove that if there are 75 dogs, 
and adding 20 cats would make the ratio of dogs to cats 15:11, 
then the initial ratio of dogs to cats is 15:7.
-/
theorem shelter_ratio (initial_cats : ℕ) : 
  (75 : ℚ) / (initial_cats + 20) = 15 / 11 → 
  75 / initial_cats = 15 / 7 := by
sorry

end shelter_ratio_l2829_282983


namespace min_a_correct_l2829_282956

/-- The number of cards in the deck -/
def n : ℕ := 52

/-- The probability that Alex and Dylan are on the same team given Alex's card number a -/
def p (a : ℕ) : ℚ :=
  let lower := (n - (a + 6) + 1).choose 2
  let higher := (a - 1).choose 2
  (lower + higher : ℚ) / (n - 2).choose 2

/-- The minimum value of a such that p(a) ≥ 1/2 -/
def min_a : ℕ := 14

theorem min_a_correct :
  (∀ a < min_a, p a < 1/2) ∧ p min_a ≥ 1/2 :=
sorry

#eval min_a

end min_a_correct_l2829_282956


namespace scalene_triangle_with_double_angle_and_36_degree_l2829_282999

def is_valid_triangle (a b c : ℝ) : Prop :=
  0 < a ∧ 0 < b ∧ 0 < c ∧ a + b + c = 180

def is_scalene (a b c : ℝ) : Prop :=
  a ≠ b ∧ b ≠ c ∧ a ≠ c

theorem scalene_triangle_with_double_angle_and_36_degree :
  ∀ a b c : ℝ,
  is_valid_triangle a b c →
  is_scalene a b c →
  ((a = 2 * b ∨ b = 2 * a ∨ a = 2 * c ∨ c = 2 * a ∨ b = 2 * c ∨ c = 2 * b) ∧
   (a = 36 ∨ b = 36 ∨ c = 36)) →
  ((a = 36 ∧ b = 48 ∧ c = 96) ∨ (a = 18 ∧ b = 36 ∧ c = 126) ∨
   (a = 48 ∧ b = 96 ∧ c = 36) ∨ (a = 36 ∧ b = 126 ∧ c = 18) ∨
   (a = 96 ∧ b = 36 ∧ c = 48) ∨ (a = 126 ∧ b = 18 ∧ c = 36)) :=
by sorry

end scalene_triangle_with_double_angle_and_36_degree_l2829_282999


namespace two_digit_average_decimal_l2829_282921

theorem two_digit_average_decimal (m n : ℕ) : 
  (10 ≤ m ∧ m < 100) →
  (10 ≤ n ∧ n < 100) →
  (m + n) / 2 = m + n / 10 →
  m = n :=
by sorry

end two_digit_average_decimal_l2829_282921


namespace smallest_k_for_divisibility_by_10_l2829_282978

def is_largest_prime_with_2005_digits (p : ℕ) : Prop :=
  Nat.Prime p ∧ 
  (10^2004 ≤ p) ∧ 
  (p < 10^2005) ∧ 
  ∀ q, Nat.Prime q → (10^2004 ≤ q) → (q < 10^2005) → q ≤ p

theorem smallest_k_for_divisibility_by_10 (p : ℕ) 
  (h : is_largest_prime_with_2005_digits p) : 
  (∃ k : ℕ, k > 0 ∧ (10 ∣ (p^2 - k))) ∧
  (∀ k : ℕ, k > 0 → (10 ∣ (p^2 - k)) → k ≥ 5) :=
sorry

end smallest_k_for_divisibility_by_10_l2829_282978


namespace internally_tangent_circles_distance_l2829_282974

theorem internally_tangent_circles_distance (r₁ r₂ d : ℝ) : 
  r₁ = 3 → r₂ = 6 → d = r₂ - r₁ → d = 3 := by sorry

end internally_tangent_circles_distance_l2829_282974


namespace chimney_bricks_total_bricks_correct_l2829_282971

-- Define the problem parameters
def brenda_time : ℝ := 8
def brandon_time : ℝ := 12
def combined_decrease : ℝ := 12
def combined_time : ℝ := 6

-- Define the theorem
theorem chimney_bricks : ∃ (h : ℝ),
  h > 0 ∧
  h / brenda_time + h / brandon_time - combined_decrease = h / combined_time :=
by
  -- The proof goes here
  sorry

-- Define the final answer
def total_bricks : ℕ := 288

-- Prove that the total_bricks satisfies the theorem
theorem total_bricks_correct : 
  ∃ (h : ℝ), h = total_bricks ∧
  h > 0 ∧
  h / brenda_time + h / brandon_time - combined_decrease = h / combined_time :=
by
  -- The proof goes here
  sorry

end chimney_bricks_total_bricks_correct_l2829_282971


namespace lunch_packing_days_l2829_282924

/-- Represents the number of school days for each school -/
structure SchoolDays where
  A : ℕ
  B : ℕ
  C : ℕ

/-- Represents the number of days a student packs lunch -/
structure LunchDays where
  A : ℕ
  B : ℕ
  C : ℕ
  D : ℕ

/-- Given the conditions of the problem, prove the correct expressions for lunch packing days -/
theorem lunch_packing_days (sd : SchoolDays) : 
  ∃ (ld : LunchDays), 
    ld.A = (3 * sd.A) / 5 ∧
    ld.B = (3 * sd.B) / 20 ∧
    ld.C = (3 * sd.C) / 10 ∧
    ld.D = sd.A / 3 := by
  sorry

end lunch_packing_days_l2829_282924


namespace max_ab_bisecting_line_l2829_282950

/-- A line that bisects the circumference of a circle --/
structure BisectingLine where
  a : ℝ
  b : ℝ
  bisects : ∀ (x y : ℝ), a * x + b * y - 1 = 0 → x^2 + y^2 - 4*x - 4*y - 8 = 0

/-- The maximum value of ab for a bisecting line --/
theorem max_ab_bisecting_line (l : BisectingLine) : 
  ∃ (max : ℝ), (∀ (l' : BisectingLine), l'.a * l'.b ≤ max) ∧ max = 1/16 := by
sorry

end max_ab_bisecting_line_l2829_282950


namespace quadratic_equation_solutions_l2829_282915

theorem quadratic_equation_solutions : 
  ∀ x : ℝ, x^2 = 6*x ↔ x = 0 ∨ x = 6 := by sorry

end quadratic_equation_solutions_l2829_282915


namespace determinant_equality_l2829_282930

theorem determinant_equality (p q r s : ℝ) : 
  p * s - q * r = 10 → (p + 2*r) * s - (q + 2*s) * r = 10 := by
  sorry

end determinant_equality_l2829_282930


namespace solve_system_l2829_282926

theorem solve_system (p q : ℚ) 
  (eq1 : 5 * p + 6 * q = 20) 
  (eq2 : 6 * p + 5 * q = 29) : 
  q = -25 / 11 := by
sorry

end solve_system_l2829_282926


namespace johns_former_apartment_cost_l2829_282960

/-- Proves that the cost per square foot of John's former apartment was $2 -/
theorem johns_former_apartment_cost (former_size : ℝ) (new_rent : ℝ) (savings : ℝ) : 
  former_size = 750 →
  new_rent = 2800 →
  savings = 1200 →
  ∃ (cost_per_sqft : ℝ), 
    cost_per_sqft = 2 ∧ 
    former_size * cost_per_sqft * 12 = (new_rent / 2) * 12 + savings :=
by sorry

end johns_former_apartment_cost_l2829_282960


namespace greatest_common_length_l2829_282963

theorem greatest_common_length (a b c d : ℕ) 
  (ha : a = 72) (hb : b = 48) (hc : c = 120) (hd : d = 96) : 
  Nat.gcd a (Nat.gcd b (Nat.gcd c d)) = 24 := by
  sorry

end greatest_common_length_l2829_282963


namespace total_students_count_l2829_282906

/-- The number of students who wish to go on a scavenger hunting trip -/
def scavenger_hunting_students : ℕ := 4000

/-- The number of students who wish to go on a skiing trip -/
def skiing_students : ℕ := 2 * scavenger_hunting_students

/-- The total number of students -/
def total_students : ℕ := scavenger_hunting_students + skiing_students

theorem total_students_count : total_students = 12000 := by
  sorry

end total_students_count_l2829_282906


namespace equivalence_of_statements_l2829_282969

theorem equivalence_of_statements (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (1/a + 1/b = Real.sqrt (a*b)) ↔ (∃ x : ℝ, x^2 + 1 ≤ 0) :=
sorry

end equivalence_of_statements_l2829_282969


namespace find_x_value_l2829_282992

theorem find_x_value (numbers : List ℕ) (x : ℕ) : 
  numbers = [54, 55, 57, 58, 59, 62, 62, 63, 65] →
  numbers.length = 9 →
  (numbers.sum + x) / 10 = 60 →
  x = 65 := by
sorry

end find_x_value_l2829_282992


namespace system_real_solutions_l2829_282966

theorem system_real_solutions (k : ℝ) : 
  (∃ x y : ℝ, x - k * y = 0 ∧ x^2 + y = -1) ↔ -1/2 ≤ k ∧ k ≤ 1/2 := by
  sorry

end system_real_solutions_l2829_282966


namespace trisha_walk_distance_l2829_282948

theorem trisha_walk_distance (total_distance : ℝ) (tshirt_to_hotel : ℝ) (hotel_to_postcard : ℝ) :
  total_distance = 0.89 →
  tshirt_to_hotel = 0.67 →
  total_distance = hotel_to_postcard + hotel_to_postcard + tshirt_to_hotel →
  hotel_to_postcard = 0.11 := by
sorry

end trisha_walk_distance_l2829_282948


namespace interval_sum_theorem_l2829_282954

open Real

/-- The floor function -/
noncomputable def floor (x : ℝ) : ℤ := Int.floor x

/-- The fractional part of a real number -/
noncomputable def frac (x : ℝ) : ℝ := x - (floor x : ℝ)

/-- The function g as defined in the problem -/
noncomputable def g (x : ℝ) : ℝ := (floor x : ℝ) * (2013^(frac x) - 2)

/-- The theorem statement -/
theorem interval_sum_theorem :
  ∃ (S : Set ℝ), S = {x : ℝ | 1 ≤ x ∧ x < 2013 ∧ g x ≤ 0} ∧
  (∫ x in S, 1) = 2012 * (log 2 / log 2013) := by sorry

end interval_sum_theorem_l2829_282954


namespace smallest_m_equals_n_l2829_282942

theorem smallest_m_equals_n (n : ℕ) (hn : n > 1) :
  ∃ (m : ℕ),
    (∀ (a b : ℕ) (ha : a ∈ Finset.range (2 * n)) (hb : b ∈ Finset.range (2 * n)) (hab : a ≠ b),
      ∃ (x y : ℕ) (hxy : x + y > 0),
        (2 * n ∣ a * x + b * y) ∧ (x + y ≤ m)) ∧
    (∀ (k : ℕ),
      (∀ (a b : ℕ) (ha : a ∈ Finset.range (2 * n)) (hb : b ∈ Finset.range (2 * n)) (hab : a ≠ b),
        ∃ (x y : ℕ) (hxy : x + y > 0),
          (2 * n ∣ a * x + b * y) ∧ (x + y ≤ k)) →
      k ≥ m) ∧
    m = n :=
by sorry

end smallest_m_equals_n_l2829_282942


namespace sufficient_not_necessary_condition_l2829_282929

theorem sufficient_not_necessary_condition (x : ℝ) :
  (∀ x, x > 0 → x ≠ 0) ∧ (∃ x, x ≠ 0 ∧ ¬(x > 0)) := by
  sorry

end sufficient_not_necessary_condition_l2829_282929


namespace fourth_red_ball_is_24_l2829_282958

/-- Represents a random number table --/
def RandomTable : List (List Nat) :=
  [[2, 9, 7, 63, 4, 1, 32, 8, 4, 14, 2, 4, 1],
   [8, 3, 0, 39, 8, 2, 25, 8, 8, 82, 4, 1, 0],
   [5, 5, 5, 68, 5, 2, 66, 1, 6, 68, 2, 3, 1]]

/-- Checks if a number is a valid red ball number --/
def isValidRedBall (n : Nat) : Bool :=
  1 ≤ n ∧ n ≤ 33

/-- Selects valid red ball numbers from a list --/
def selectValidNumbers (numbers : List Nat) : List Nat :=
  numbers.filter isValidRedBall

/-- Flattens the random table into a single list, starting from the specified position --/
def flattenTableFrom (table : List (List Nat)) (startRow startCol : Nat) : List Nat :=
  let rowsFromStart := table.drop startRow
  let firstRow := (rowsFromStart.head!).drop startCol
  firstRow ++ (rowsFromStart.tail!).join

/-- The main theorem to prove --/
theorem fourth_red_ball_is_24 :
  let flattenedTable := flattenTableFrom RandomTable 0 8
  let validNumbers := selectValidNumbers flattenedTable
  validNumbers[3] = 24 := by sorry

end fourth_red_ball_is_24_l2829_282958


namespace sheet_reduction_percentage_l2829_282947

def original_sheets : ℕ := 20
def original_lines_per_sheet : ℕ := 55
def original_chars_per_line : ℕ := 65

def new_lines_per_sheet : ℕ := 65
def new_chars_per_line : ℕ := 70

def total_chars : ℕ := original_sheets * original_lines_per_sheet * original_chars_per_line
def new_chars_per_sheet : ℕ := new_lines_per_sheet * new_chars_per_line
def new_sheets : ℕ := (total_chars + new_chars_per_sheet - 1) / new_chars_per_sheet

theorem sheet_reduction_percentage : 
  (original_sheets - new_sheets : ℚ) / original_sheets * 100 = 20 := by sorry

end sheet_reduction_percentage_l2829_282947


namespace no_five_two_digit_coprime_composites_l2829_282968

def is_two_digit (n : ℕ) : Prop := 10 ≤ n ∧ n ≤ 99

def is_composite (n : ℕ) : Prop := ∃ a b, 1 < a ∧ 1 < b ∧ n = a * b

def are_coprime (a b : ℕ) : Prop := Nat.gcd a b = 1

theorem no_five_two_digit_coprime_composites :
  ¬ ∃ a b c d e : ℕ,
    is_two_digit a ∧ is_two_digit b ∧ is_two_digit c ∧ is_two_digit d ∧ is_two_digit e ∧
    is_composite a ∧ is_composite b ∧ is_composite c ∧ is_composite d ∧ is_composite e ∧
    are_coprime a b ∧ are_coprime a c ∧ are_coprime a d ∧ are_coprime a e ∧
    are_coprime b c ∧ are_coprime b d ∧ are_coprime b e ∧
    are_coprime c d ∧ are_coprime c e ∧
    are_coprime d e :=
by
  sorry

end no_five_two_digit_coprime_composites_l2829_282968


namespace quadratic_polynomial_problem_l2829_282944

theorem quadratic_polynomial_problem :
  ∃ (p : ℝ → ℝ),
    (∀ x, p x = (20/9) * x^2 + (20/3) * x - 40) ∧
    p (-6) = 0 ∧
    p 3 = 0 ∧
    p (-3) = -40 := by
  sorry

end quadratic_polynomial_problem_l2829_282944


namespace parallel_angles_theorem_l2829_282920

theorem parallel_angles_theorem (angle1 angle2 : ℝ) : 
  (angle1 + angle2 = 180 ∨ angle1 = angle2) →  -- parallel sides condition
  angle2 = 3 * angle1 - 20 →                   -- angle relationship
  (angle1 = 50 ∧ angle2 = 130) :=              -- conclusion
by sorry

end parallel_angles_theorem_l2829_282920


namespace inequality_proof_l2829_282933

theorem inequality_proof (a b c d : ℝ) (h : a > b ∧ b > c ∧ c > d) :
  c < (c * d - a * b) / (c - a + d - b) ∧ (c * d - a * b) / (c - a + d - b) < b := by
  sorry

end inequality_proof_l2829_282933


namespace faster_train_speed_l2829_282902

theorem faster_train_speed
  (train_length : ℝ)
  (speed_difference : ℝ)
  (passing_time : ℝ)
  (h1 : train_length = 37.5)
  (h2 : speed_difference = 36)
  (h3 : passing_time = 27)
  : ∃ (faster_speed : ℝ),
    faster_speed = 46 ∧
    (faster_speed - speed_difference) * 1000 / 3600 * passing_time = 2 * train_length :=
by
  sorry

end faster_train_speed_l2829_282902


namespace midpoint_coordinate_sum_l2829_282962

/-- Given that P(1, -3) is the midpoint of line segment CD and C is located at (7, 5),
    prove that the sum of the coordinates of point D is -16. -/
theorem midpoint_coordinate_sum (C D : ℝ × ℝ) : 
  C = (7, 5) →
  (1, -3) = ((C.1 + D.1) / 2, (C.2 + D.2) / 2) →
  D.1 + D.2 = -16 := by
  sorry

end midpoint_coordinate_sum_l2829_282962


namespace factorial_fraction_simplification_l2829_282980

theorem factorial_fraction_simplification (N : ℕ) :
  (Nat.factorial (N - 1) * N * (N + 1)) / Nat.factorial (N + 2) = 1 / (N + 2) := by
  sorry

end factorial_fraction_simplification_l2829_282980


namespace tiffany_miles_per_day_l2829_282939

theorem tiffany_miles_per_day (T : ℚ) : 
  (7 : ℚ) = 3 * T + 3 * (1/3 : ℚ) + 0 → T = 2 := by
  sorry

end tiffany_miles_per_day_l2829_282939


namespace rhombus_longer_diagonal_l2829_282925

/-- Given a rhombus with side length 60 units and shorter diagonal 56 units,
    the longer diagonal has a length of 32√11 units. -/
theorem rhombus_longer_diagonal (side : ℝ) (shorter_diag : ℝ) (longer_diag : ℝ) 
    (h1 : side = 60) 
    (h2 : shorter_diag = 56) 
    (h3 : side^2 = (shorter_diag/2)^2 + (longer_diag/2)^2) : 
  longer_diag = 32 * Real.sqrt 11 := by
  sorry

end rhombus_longer_diagonal_l2829_282925


namespace nth_equation_pattern_l2829_282967

theorem nth_equation_pattern (n : ℕ) : 1 + 6 * n = (3 * n + 1)^2 - 9 * n^2 := by
  sorry

end nth_equation_pattern_l2829_282967


namespace solution_set_for_half_range_of_m_l2829_282994

-- Define the function f
def f (x m : ℝ) : ℝ := |x + m| - |2*x - 2*m|

-- Part 1
theorem solution_set_for_half (x : ℝ) :
  (f x (1/2) ≥ 1/2) ↔ (1/3 ≤ x ∧ x < 1) :=
sorry

-- Part 2
theorem range_of_m :
  ∀ m : ℝ, (m > 0 ∧ m < 7/2) ↔
    (∀ x : ℝ, ∃ t : ℝ, f x m + |t - 3| < |t + 4|) :=
sorry

end solution_set_for_half_range_of_m_l2829_282994


namespace largest_triangle_perimeter_l2829_282918

theorem largest_triangle_perimeter :
  ∀ x : ℕ,
  (7 + 9 > x) ∧ (7 + x > 9) ∧ (9 + x > 7) →
  (∀ y : ℕ, (7 + 9 > y) ∧ (7 + y > 9) ∧ (9 + y > 7) → x ≥ y) →
  7 + 9 + x = 31 :=
by sorry

end largest_triangle_perimeter_l2829_282918


namespace f_composition_equals_sqrt2_over_2_l2829_282908

noncomputable def f (x : ℝ) : ℝ :=
  if x ≤ 0 then 1 - 2^x else Real.sqrt x

theorem f_composition_equals_sqrt2_over_2 : f (f (-1)) = Real.sqrt 2 / 2 := by
  sorry

end f_composition_equals_sqrt2_over_2_l2829_282908


namespace wrong_mark_correction_l2829_282914

theorem wrong_mark_correction (n : ℕ) (initial_avg correct_avg : ℚ) (correct_mark : ℚ) (x : ℚ) 
  (h1 : n = 30)
  (h2 : initial_avg = 100)
  (h3 : correct_avg = 98)
  (h4 : correct_mark = 10) :
  (n : ℚ) * initial_avg - x + correct_mark = n * correct_avg → x = 70 := by
  sorry

end wrong_mark_correction_l2829_282914


namespace june_upload_total_l2829_282984

/-- Represents the upload schedule for a YouTuber in June --/
structure UploadSchedule where
  early_june : Nat  -- videos per day from June 1st to June 15th
  mid_june : Nat    -- videos per day from June 16th to June 23rd
  late_june : Nat   -- videos per day from June 24th to June 30th

/-- Calculates the total number of video hours uploaded in June --/
def total_video_hours (schedule : UploadSchedule) : Nat :=
  schedule.early_june * 15 + schedule.mid_june * 8 + schedule.late_june * 7

/-- Theorem stating that the given upload schedule results in 480 total video hours --/
theorem june_upload_total (schedule : UploadSchedule) 
  (h1 : schedule.early_june = 10)
  (h2 : schedule.mid_june = 15)
  (h3 : schedule.late_june = 30) : 
  total_video_hours schedule = 480 := by
  sorry

#eval total_video_hours { early_june := 10, mid_june := 15, late_june := 30 }

end june_upload_total_l2829_282984


namespace percentage_of_800_l2829_282953

theorem percentage_of_800 : (25 / 100) * 800 = 200 := by
  sorry

end percentage_of_800_l2829_282953


namespace inequality_preservation_l2829_282961

theorem inequality_preservation (a b c : ℝ) (h : a > b) :
  a / (c^2 + 1) > b / (c^2 + 1) := by
  sorry

end inequality_preservation_l2829_282961


namespace central_angle_A_B_l2829_282907

noncomputable def earthRadius : ℝ := 1 -- Normalized Earth radius

/-- Represents a point on the Earth's surface using latitude and longitude -/
structure EarthPoint where
  latitude : ℝ
  longitude : ℝ

/-- Calculates the angle at the Earth's center between two points on the surface -/
noncomputable def centralAngle (p1 p2 : EarthPoint) : ℝ := sorry

/-- Point A on Earth's surface -/
def pointA : EarthPoint := { latitude := 0, longitude := 90 }

/-- Point B on Earth's surface -/
def pointB : EarthPoint := { latitude := 30, longitude := -80 }

/-- Theorem stating that the central angle between points A and B is 140 degrees -/
theorem central_angle_A_B :
  centralAngle pointA pointB = 140 * (π / 180) := by sorry

end central_angle_A_B_l2829_282907


namespace mutually_exclusive_events_l2829_282965

/-- Represents the outcome of a single shot -/
inductive ShotOutcome
| Hit
| Miss

/-- Represents the outcome of two shots -/
def TwoShotOutcome := ShotOutcome × ShotOutcome

/-- The event of hitting the target at least once in two shots -/
def hitAtLeastOnce (outcome : TwoShotOutcome) : Prop :=
  outcome.1 = ShotOutcome.Hit ∨ outcome.2 = ShotOutcome.Hit

/-- The event of missing the target both times in two shots -/
def missBothTimes (outcome : TwoShotOutcome) : Prop :=
  outcome.1 = ShotOutcome.Miss ∧ outcome.2 = ShotOutcome.Miss

theorem mutually_exclusive_events :
  ∀ (outcome : TwoShotOutcome), ¬(hitAtLeastOnce outcome ∧ missBothTimes outcome) :=
sorry

end mutually_exclusive_events_l2829_282965


namespace dot_product_perpendiculars_l2829_282934

/-- Given a point P(x₀, y₀) on the curve y = x + 2/x for x > 0,
    and points A and B as the feet of perpendiculars from P to y = x and y-axis respectively,
    prove that the dot product of PA and PB is -1. -/
theorem dot_product_perpendiculars (x₀ : ℝ) (h₀ : x₀ > 0) : 
  let y₀ := x₀ + 2 / x₀
  let P := (x₀, y₀)
  let A := ((x₀ + y₀) / 2, (x₀ + y₀) / 2)  -- Foot of perpendicular to y = x
  let B := (0, y₀)  -- Foot of perpendicular to y-axis
  (P.1 - A.1) * (P.1 - B.1) + (P.2 - A.2) * (P.2 - B.2) = -1 :=
by sorry

end dot_product_perpendiculars_l2829_282934


namespace stride_sync_l2829_282900

/-- The least common multiple of Jack and Jill's stride lengths -/
def stride_lcm (jack_stride jill_stride : ℕ) : ℕ :=
  Nat.lcm jack_stride jill_stride

/-- Theorem stating that the LCM of Jack and Jill's strides is 448 cm -/
theorem stride_sync (jack_stride jill_stride : ℕ) 
  (h1 : jack_stride = 64) 
  (h2 : jill_stride = 56) : 
  stride_lcm jack_stride jill_stride = 448 := by
  sorry

end stride_sync_l2829_282900


namespace count_polygons_l2829_282946

/-- The number of distinct convex polygons with 3 or more sides
    that can be drawn from 12 points on a circle -/
def num_polygons : ℕ := 4017

/-- The number of points marked on the circle -/
def num_points : ℕ := 12

/-- Theorem stating that the number of distinct convex polygons
    with 3 or more sides drawn from 12 points on a circle is 4017 -/
theorem count_polygons :
  (2^num_points : ℕ) - (Nat.choose num_points 0) - (Nat.choose num_points 1) - (Nat.choose num_points 2) = num_polygons :=
by sorry

end count_polygons_l2829_282946


namespace rectangular_solid_diagonal_l2829_282909

theorem rectangular_solid_diagonal (a b c : ℝ) 
  (surface_area : 2 * (a * b + b * c + a * c) = 54)
  (edge_length : 4 * (a + b + c) = 40) :
  ∃ d : ℝ, d^2 = a^2 + b^2 + c^2 ∧ d = Real.sqrt 46 := by
  sorry

end rectangular_solid_diagonal_l2829_282909


namespace sequence_sum_problem_l2829_282996

theorem sequence_sum_problem (x₁ x₂ x₃ x₄ x₅ x₆ x₇ : ℝ) 
  (eq1 : x₁ + 4*x₂ + 9*x₃ + 16*x₄ + 25*x₅ + 36*x₆ + 49*x₇ = 1)
  (eq2 : 4*x₁ + 9*x₂ + 16*x₃ + 25*x₄ + 36*x₅ + 49*x₆ + 64*x₇ = 8)
  (eq3 : 9*x₁ + 16*x₂ + 25*x₃ + 36*x₄ + 49*x₅ + 64*x₆ + 81*x₇ = 81) :
  25*x₁ + 36*x₂ + 49*x₃ + 64*x₄ + 81*x₅ + 100*x₆ + 121*x₇ = 425 := by
  sorry

end sequence_sum_problem_l2829_282996


namespace p_and_q_true_l2829_282911

theorem p_and_q_true (h1 : p ∨ q) (h2 : p ∧ q) : p ∧ q := by
  sorry

end p_and_q_true_l2829_282911


namespace power_five_plus_five_mod_eight_l2829_282973

theorem power_five_plus_five_mod_eight : (5^123 + 5) % 8 = 2 := by
  sorry

end power_five_plus_five_mod_eight_l2829_282973


namespace arithmetic_simplification_l2829_282945

theorem arithmetic_simplification :
  (100 - 25 * 4 = 0) ∧
  (20 / 5 * 2 = 8) ∧
  (360 - 200 / 4 = 310) ∧
  (36 / 3 + 27 = 39) := by
  sorry

end arithmetic_simplification_l2829_282945


namespace derivative_f_l2829_282972

noncomputable def f (x : ℝ) : ℝ := (x + 1/x)^5

theorem derivative_f (x : ℝ) (hx : x ≠ 0) :
  deriv f x = 5 * (x + 1/x)^4 * (1 - 1/x^2) :=
by sorry

end derivative_f_l2829_282972


namespace linear_function_property_l2829_282922

/-- A linear function f where f(6) - f(2) = 12 -/
def LinearFunction (f : ℝ → ℝ) : Prop :=
  (∀ x y t : ℝ, f (t * x + (1 - t) * y) = t * f x + (1 - t) * f y) ∧ 
  (f 6 - f 2 = 12)

theorem linear_function_property (f : ℝ → ℝ) (h : LinearFunction f) : 
  f 12 - f 2 = 30 := by
  sorry

end linear_function_property_l2829_282922


namespace claudia_weekend_earnings_l2829_282923

-- Define the charge per class
def charge_per_class : ℝ := 10.00

-- Define the number of kids in Saturday's class
def saturday_attendance : ℕ := 20

-- Define the number of kids in Sunday's class
def sunday_attendance : ℕ := saturday_attendance / 2

-- Define the total attendance for the weekend
def total_attendance : ℕ := saturday_attendance + sunday_attendance

-- Theorem to prove
theorem claudia_weekend_earnings :
  (total_attendance : ℝ) * charge_per_class = 300.00 := by
  sorry

end claudia_weekend_earnings_l2829_282923


namespace fraction_sum_integer_implies_not_divisible_by_three_l2829_282989

theorem fraction_sum_integer_implies_not_divisible_by_three (n : ℕ+) 
  (h : ∃ (k : ℤ), (1 : ℚ) / 3 + (1 : ℚ) / 4 + (1 : ℚ) / 8 + (1 : ℚ) / n.val = k) : 
  ¬(3 ∣ n.val) := by
sorry

end fraction_sum_integer_implies_not_divisible_by_three_l2829_282989


namespace initial_trees_per_row_garden_problem_l2829_282943

theorem initial_trees_per_row : ℕ → ℕ → ℕ → ℕ → Prop :=
  fun initial_rows added_rows final_trees_per_row result =>
    let final_rows := initial_rows + added_rows
    (initial_rows * result = final_rows * final_trees_per_row) →
    (result = 42)

/-- Given the initial number of rows, added rows, and final trees per row,
    prove that the initial number of trees per row is 42. -/
theorem garden_problem (initial_rows added_rows final_trees_per_row : ℕ)
    (h1 : initial_rows = 24)
    (h2 : added_rows = 12)
    (h3 : final_trees_per_row = 28) :
    initial_trees_per_row initial_rows added_rows final_trees_per_row 42 := by
  sorry

end initial_trees_per_row_garden_problem_l2829_282943


namespace time_after_316h59m59s_l2829_282952

/-- Represents time on a 12-hour digital clock -/
structure Time where
  hours : Nat
  minutes : Nat
  seconds : Nat
  deriving Repr

def addTime (t : Time) (hours minutes seconds : Nat) : Time :=
  let totalSeconds := t.hours * 3600 + t.minutes * 60 + t.seconds + hours * 3600 + minutes * 60 + seconds
  let newHours := (totalSeconds / 3600) % 12
  let newMinutes := (totalSeconds % 3600) / 60
  let newSeconds := totalSeconds % 60
  { hours := if newHours = 0 then 12 else newHours, minutes := newMinutes, seconds := newSeconds }

def sumDigits (t : Time) : Nat :=
  t.hours + t.minutes + t.seconds

theorem time_after_316h59m59s (startTime : Time) :
  startTime.hours = 3 ∧ startTime.minutes = 0 ∧ startTime.seconds = 0 →
  sumDigits (addTime startTime 316 59 59) = 125 := by
  sorry

end time_after_316h59m59s_l2829_282952


namespace quadratic_min_values_l2829_282917

-- Define the quadratic function
def f (x a : ℝ) : ℝ := 2 * x^2 - 4 * a * x + a^2 + 2 * a + 2

-- State the theorem
theorem quadratic_min_values (a : ℝ) :
  (∀ x, -1 ≤ x ∧ x ≤ 2 → f x a ≥ 2) ∧
  (∃ x, -1 ≤ x ∧ x ≤ 2 ∧ f x a = 2) →
  a = 0 ∨ a = 2 ∨ a = -3 - Real.sqrt 7 ∨ a = 4 :=
by sorry

end quadratic_min_values_l2829_282917


namespace cyclists_meeting_time_l2829_282997

/-- Two cyclists on a circular track problem -/
theorem cyclists_meeting_time
  (circumference : ℝ)
  (speed1 : ℝ)
  (speed2 : ℝ)
  (h1 : circumference = 180)
  (h2 : speed1 = 7)
  (h3 : speed2 = 8) :
  circumference / (speed1 + speed2) = 12 := by
  sorry

end cyclists_meeting_time_l2829_282997


namespace bridget_apples_l2829_282998

theorem bridget_apples (x : ℕ) : 
  (x : ℚ) / 3 + 5 + 6 = x → x = 17 := by
  sorry

end bridget_apples_l2829_282998


namespace model_price_increase_l2829_282932

theorem model_price_increase (original_price : ℚ) (original_quantity : ℕ) (new_quantity : ℕ) 
  (h1 : original_price = 45 / 100)
  (h2 : original_quantity = 30)
  (h3 : new_quantity = 27) :
  let total_saved := original_price * original_quantity
  let new_price := total_saved / new_quantity
  new_price = 1 / 2 := by sorry

end model_price_increase_l2829_282932


namespace rachel_reading_homework_l2829_282927

theorem rachel_reading_homework (literature_pages : ℕ) (additional_reading_pages : ℕ) 
  (h1 : literature_pages = 10) 
  (h2 : additional_reading_pages = 6) : 
  literature_pages + additional_reading_pages = 16 := by
  sorry

end rachel_reading_homework_l2829_282927


namespace dividend_calculation_l2829_282919

theorem dividend_calculation (divisor quotient remainder : ℝ) 
  (h_divisor : divisor = 127.5)
  (h_quotient : quotient = 238)
  (h_remainder : remainder = 53.2) :
  divisor * quotient + remainder = 30398.2 := by
  sorry

end dividend_calculation_l2829_282919


namespace map_scale_conversion_l2829_282986

theorem map_scale_conversion (map_cm : ℝ) (real_km : ℝ) : 
  (20 : ℝ) * real_km = 100 * map_cm → 25 * real_km = 125 * map_cm := by
  sorry

end map_scale_conversion_l2829_282986


namespace sphere_volume_l2829_282937

theorem sphere_volume (r : ℝ) (h : 4 * Real.pi * r^2 = 36 * Real.pi) :
  (4 / 3) * Real.pi * r^3 = 36 * Real.pi := by sorry

end sphere_volume_l2829_282937


namespace exist_cubes_sum_100_power_100_l2829_282955

theorem exist_cubes_sum_100_power_100 : ∃ (a b c d : ℕ+), (a.val ^ 3 + b.val ^ 3 + c.val ^ 3 + d.val ^ 3 : ℕ) = 100 ^ 100 := by
  sorry

end exist_cubes_sum_100_power_100_l2829_282955


namespace fixed_point_of_exponential_function_l2829_282981

theorem fixed_point_of_exponential_function (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) :
  let f : ℝ → ℝ := λ x ↦ a^(x - 2015) + 2015
  f 2015 = 2016 ∧ ∃ (x : ℝ), f x = x := by
  sorry

end fixed_point_of_exponential_function_l2829_282981


namespace equation_solution_l2829_282957

theorem equation_solution :
  ∃ x : ℝ, 45 - (28 - (37 - (15 - x))) = 57 ∧ x = 18 := by
  sorry

end equation_solution_l2829_282957


namespace line_points_theorem_l2829_282987

-- Define the line L with slope 2 passing through (3, 5)
def L (x y : ℝ) : Prop := y - 5 = 2 * (x - 3)

-- Define the points
def P1 : ℝ × ℝ := (3, 5)
def P2 (x2 : ℝ) : ℝ × ℝ := (x2, 7)
def P3 (y3 : ℝ) : ℝ × ℝ := (-1, y3)

theorem line_points_theorem (x2 y3 : ℝ) :
  L P1.1 P1.2 ∧ L (P2 x2).1 (P2 x2).2 ∧ L (P3 y3).1 (P3 y3).2 →
  x2 = 4 ∧ y3 = -3 := by
  sorry

end line_points_theorem_l2829_282987


namespace length_of_CF_l2829_282936

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- A rectangle ABCD with given properties -/
structure Rectangle where
  A : Point
  B : Point
  C : Point
  D : Point
  ab_length : ℝ
  bc_length : ℝ
  cd_length : ℝ
  da_length : ℝ
  is_rectangle : ab_length = cd_length ∧ bc_length = da_length

/-- Triangle DEF with B as its centroid -/
structure TriangleDEF where
  D : Point
  E : Point
  F : Point
  B : Point
  is_centroid : B.x = (2 * D.x + E.x) / 3 ∧ B.y = (2 * D.y + E.y) / 3

/-- The main theorem -/
theorem length_of_CF (rect : Rectangle) (tri : TriangleDEF) :
  rect.A = tri.D ∧
  rect.B = tri.B ∧
  rect.C.x = tri.F.x ∧
  rect.da_length = 7 ∧
  rect.ab_length = 6 ∧
  rect.cd_length = 8 →
  Real.sqrt ((rect.C.x - tri.F.x)^2 + (rect.C.y - tri.F.y)^2) = 10.66 := by
  sorry


end length_of_CF_l2829_282936


namespace product_of_roots_l2829_282903

theorem product_of_roots (k m x₁ x₂ : ℝ) (h_distinct : x₁ ≠ x₂)
  (h₁ : 4 * x₁^2 - k * x₁ - m = 0) (h₂ : 4 * x₂^2 - k * x₂ - m = 0) :
  x₁ * x₂ = -m / 4 := by
  sorry

end product_of_roots_l2829_282903


namespace subtraction_of_decimals_l2829_282951

theorem subtraction_of_decimals : 3.75 - 0.48 = 3.27 := by
  sorry

end subtraction_of_decimals_l2829_282951


namespace eric_containers_l2829_282970

/-- The number of containers Eric has for his colored pencils. -/
def number_of_containers (initial_pencils : ℕ) (additional_pencils : ℕ) (pencils_per_container : ℕ) : ℕ :=
  (initial_pencils + additional_pencils) / pencils_per_container

theorem eric_containers :
  number_of_containers 150 30 36 = 5 := by
  sorry

end eric_containers_l2829_282970


namespace sqrt_twelve_equals_two_sqrt_three_l2829_282959

theorem sqrt_twelve_equals_two_sqrt_three : Real.sqrt 12 = 2 * Real.sqrt 3 := by
  sorry

end sqrt_twelve_equals_two_sqrt_three_l2829_282959


namespace parabola_c_value_l2829_282910

/-- A parabola with equation y = 2x^2 + bx + c passes through the points (1,5) and (3,17).
    This theorem proves that the value of c is 5. -/
theorem parabola_c_value :
  ∀ b c : ℝ,
  (5 : ℝ) = 2 * (1 : ℝ)^2 + b * (1 : ℝ) + c →
  (17 : ℝ) = 2 * (3 : ℝ)^2 + b * (3 : ℝ) + c →
  c = 5 := by
sorry

end parabola_c_value_l2829_282910


namespace arithmetic_sequence_unique_determination_l2829_282938

/-- Given an arithmetic sequence b₁, b₂, b₃, ..., we define:
    S'ₙ = b₁ + b₂ + b₃ + ... + bₙ
    T'ₙ = S'₁ + S'₂ + S'₃ + ... + S'ₙ
    This theorem states that if we know the value of S'₃₀₂₈, 
    then 4543 is the smallest positive integer n for which 
    T'ₙ can be uniquely determined. -/
theorem arithmetic_sequence_unique_determination (b₁ : ℚ) (d : ℚ) (S'₃₀₂₈ : ℚ) :
  let b : ℕ → ℚ := λ n => b₁ + (n - 1) * d
  let S' : ℕ → ℚ := λ n => (n : ℚ) * (2 * b₁ + (n - 1) * d) / 2
  let T' : ℕ → ℚ := λ n => (n * (n + 1) * (3 * b₁ + (n - 1) * d)) / 6
  ∃! (T'₄₅₄₃ : ℚ), S'₃₀₂₈ = S' 3028 ∧ T'₄₅₄₃ = T' 4543 ∧
    ∀ m : ℕ, m < 4543 → ¬∃! (T'ₘ : ℚ), S'₃₀₂₈ = S' 3028 ∧ T'ₘ = T' m :=
by
  sorry

end arithmetic_sequence_unique_determination_l2829_282938


namespace circle_equation_proof_l2829_282928

/-- The circle with center on y = -4x and tangent to x + y - 1 = 0 at (3, -2) -/
def special_circle (x y : ℝ) : Prop :=
  (x - 1)^2 + (y + 4)^2 = 8

/-- The line y = -4x -/
def center_line (x y : ℝ) : Prop :=
  y = -4 * x

/-- The line x + y - 1 = 0 -/
def tangent_line (x y : ℝ) : Prop :=
  x + y - 1 = 0

/-- The point P(3, -2) -/
def point_P : ℝ × ℝ :=
  (3, -2)

theorem circle_equation_proof :
  ∃ (c : ℝ × ℝ), 
    (center_line c.1 c.2) ∧ 
    (∀ (x y : ℝ), tangent_line x y → 
      ((x - c.1)^2 + (y - c.2)^2 = (c.1 - point_P.1)^2 + (c.2 - point_P.2)^2)) ↔
    (∀ (x y : ℝ), special_circle x y ↔ 
      ((x - 1)^2 + (y + 4)^2 = (1 - point_P.1)^2 + (-4 - point_P.2)^2)) :=
sorry

end circle_equation_proof_l2829_282928


namespace number_of_bags_l2829_282913

theorem number_of_bags (students : ℕ) (nuts_per_student : ℕ) (nuts_per_bag : ℕ) : 
  students = 13 → nuts_per_student = 75 → nuts_per_bag = 15 →
  (students * nuts_per_student) / nuts_per_bag = 65 := by
  sorry

end number_of_bags_l2829_282913


namespace simons_score_l2829_282901

theorem simons_score (n : ℕ) (avg_before avg_after simons_score : ℚ) : 
  n = 21 →
  avg_before = 86 →
  avg_after = 88 →
  n * avg_after = (n - 1) * avg_before + simons_score →
  simons_score = 128 :=
by sorry

end simons_score_l2829_282901


namespace right_pyramid_base_side_l2829_282991

-- Define the pyramid structure
structure RightPyramid :=
  (base_side : ℝ)
  (slant_height : ℝ)
  (lateral_face_area : ℝ)

-- Theorem statement
theorem right_pyramid_base_side 
  (p : RightPyramid) 
  (h1 : p.lateral_face_area = 120) 
  (h2 : p.slant_height = 40) : 
  p.base_side = 6 := by
  sorry


end right_pyramid_base_side_l2829_282991


namespace sqrt_88200_simplification_l2829_282975

theorem sqrt_88200_simplification : Real.sqrt 88200 = 70 * Real.sqrt 6 := by
  sorry

end sqrt_88200_simplification_l2829_282975


namespace students_not_in_biology_l2829_282964

theorem students_not_in_biology (total_students : ℕ) (biology_percentage : ℚ) 
  (h1 : total_students = 880) 
  (h2 : biology_percentage = 275 / 1000) : 
  total_students - (total_students * biology_percentage).floor = 638 := by
  sorry

end students_not_in_biology_l2829_282964


namespace investment_percentage_l2829_282976

/-- Given an investment scenario, prove that the unknown percentage is 7% -/
theorem investment_percentage (total_investment : ℝ) (known_rate : ℝ) (total_interest : ℝ) (amount_at_unknown_rate : ℝ) (unknown_rate : ℝ) :
  total_investment = 12000 ∧
  known_rate = 0.09 ∧
  total_interest = 970 ∧
  amount_at_unknown_rate = 5500 ∧
  amount_at_unknown_rate * unknown_rate + (total_investment - amount_at_unknown_rate) * known_rate = total_interest →
  unknown_rate = 0.07 := by
  sorry

end investment_percentage_l2829_282976


namespace train_speed_l2829_282941

/-- The speed of a train given its length and time to pass a stationary point -/
theorem train_speed (length : ℝ) (time : ℝ) (h1 : length = 160) (h2 : time = 4) :
  length / time = 40 := by
  sorry

#check train_speed

end train_speed_l2829_282941


namespace triangle_transformation_l2829_282912

theorem triangle_transformation (n : ℕ) (remaining_fraction : ℚ) :
  n = 3 ∧ 
  remaining_fraction = (8 / 9 : ℚ)^n → 
  remaining_fraction = 512 / 729 := by
sorry

end triangle_transformation_l2829_282912
