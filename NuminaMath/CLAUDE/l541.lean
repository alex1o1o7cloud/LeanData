import Mathlib

namespace NUMINAMATH_CALUDE_steps_to_eleventh_floor_l541_54151

/-- Given that there are 42 steps between the 3rd and 5th floors of a building,
    prove that there are 210 steps from the ground floor to the 11th floor. -/
theorem steps_to_eleventh_floor :
  let steps_between_3_and_5 : ℕ := 42
  let floor_xiao_dong_lives : ℕ := 11
  let ground_floor : ℕ := 1
  let steps_to_xiao_dong : ℕ := (floor_xiao_dong_lives - ground_floor) * 
    (steps_between_3_and_5 / (5 - 3))
  steps_to_xiao_dong = 210 := by
  sorry


end NUMINAMATH_CALUDE_steps_to_eleventh_floor_l541_54151


namespace NUMINAMATH_CALUDE_prob_odd_after_removal_is_11_21_l541_54105

/-- A standard die with faces numbered 1 to 6 -/
def standardDie : Finset ℕ := {1, 2, 3, 4, 5, 6}

/-- Total number of dots on a standard die -/
def totalDots : ℕ := standardDie.sum id

/-- Probability of removing a dot from a specific face -/
def probRemoveDot (face : ℕ) : ℚ := face / totalDots

/-- Probability of rolling an odd number after removing a dot -/
def probOddAfterRemoval : ℚ :=
  (1 / 6 * (probRemoveDot 2 + probRemoveDot 4 + probRemoveDot 6)) +
  (1 / 3 * (probRemoveDot 1 + probRemoveDot 3 + probRemoveDot 5))

theorem prob_odd_after_removal_is_11_21 : probOddAfterRemoval = 11 / 21 := by
  sorry

end NUMINAMATH_CALUDE_prob_odd_after_removal_is_11_21_l541_54105


namespace NUMINAMATH_CALUDE_table_permutation_exists_l541_54142

/-- Represents a 2 × n table of real numbers -/
def Table (n : ℕ) := Fin 2 → Fin n → ℝ

/-- Calculates the sum of a column in the table -/
def columnSum (t : Table n) (j : Fin n) : ℝ :=
  (t 0 j) + (t 1 j)

/-- Calculates the sum of a row in the table -/
def rowSum (t : Table n) (i : Fin 2) : ℝ :=
  Finset.sum (Finset.univ : Finset (Fin n)) (λ j => t i j)

/-- States that all column sums in a table are different -/
def distinctColumnSums (t : Table n) : Prop :=
  ∀ j k : Fin n, j ≠ k → columnSum t j ≠ columnSum t k

/-- Represents a permutation of table elements -/
def tablePermutation (n : ℕ) := Fin 2 → Fin n → Fin 2 × Fin n

/-- Applies a permutation to a table -/
def applyPermutation (t : Table n) (p : tablePermutation n) : Table n :=
  λ i j => let (i', j') := p i j; t i' j'

theorem table_permutation_exists (n : ℕ) (h : n > 2) (t : Table n) 
  (hd : distinctColumnSums t) :
  ∃ p : tablePermutation n, 
    distinctColumnSums (applyPermutation t p) ∧ 
    rowSum (applyPermutation t p) 0 ≠ rowSum (applyPermutation t p) 1 :=
  sorry

end NUMINAMATH_CALUDE_table_permutation_exists_l541_54142


namespace NUMINAMATH_CALUDE_linear_function_inequality_l541_54179

theorem linear_function_inequality (f : ℝ → ℝ) (a b : ℝ) :
  (∀ x, f x = a * x + b) →
  (∀ x, f (f x) ≥ x - 3) ↔
  ((a = -1 ∧ b ∈ Set.univ) ∨ (a = 1 ∧ b ≥ -3/2)) :=
by sorry

end NUMINAMATH_CALUDE_linear_function_inequality_l541_54179


namespace NUMINAMATH_CALUDE_cryptarithmetic_puzzle_l541_54122

theorem cryptarithmetic_puzzle (T W O F U R : ℕ) : 
  (T = 9) →
  (O % 2 = 1) →
  (T + T + W + W = F * 1000 + O * 100 + U * 10 + R) →
  (T ≠ W ∧ T ≠ O ∧ T ≠ F ∧ T ≠ U ∧ T ≠ R ∧
   W ≠ O ∧ W ≠ F ∧ W ≠ U ∧ W ≠ R ∧
   O ≠ F ∧ O ≠ U ∧ O ≠ R ∧
   F ≠ U ∧ F ≠ R ∧
   U ≠ R) →
  (T < 10 ∧ W < 10 ∧ O < 10 ∧ F < 10 ∧ U < 10 ∧ R < 10) →
  W = 1 := by
sorry

end NUMINAMATH_CALUDE_cryptarithmetic_puzzle_l541_54122


namespace NUMINAMATH_CALUDE_four_people_name_condition_l541_54185

/-- Represents a person with a first name, patronymic, and last name -/
structure Person where
  firstName : String
  patronymic : String
  lastName : String

/-- Checks if two people share any attribute -/
def shareAttribute (p1 p2 : Person) : Prop :=
  p1.firstName = p2.firstName ∨ p1.patronymic = p2.patronymic ∨ p1.lastName = p2.lastName

/-- Theorem stating the existence of 4 people satisfying the given conditions -/
theorem four_people_name_condition : ∃ (people : Finset Person),
  (Finset.card people = 4) ∧
  (∀ (attr : Person → String),
    ∀ (p1 p2 p3 : Person),
      p1 ∈ people → p2 ∈ people → p3 ∈ people →
      p1 ≠ p2 → p2 ≠ p3 → p1 ≠ p3 →
      ¬(attr p1 = attr p2 ∧ attr p2 = attr p3)) ∧
  (∀ (p1 p2 : Person),
    p1 ∈ people → p2 ∈ people → p1 ≠ p2 →
    shareAttribute p1 p2) :=
by sorry

end NUMINAMATH_CALUDE_four_people_name_condition_l541_54185


namespace NUMINAMATH_CALUDE_max_value_of_prime_sum_diff_l541_54139

theorem max_value_of_prime_sum_diff (a b c : ℕ) : 
  Nat.Prime a ∧ Nat.Prime b ∧ Nat.Prime c ∧  -- a, b, c are prime
  a ≠ b ∧ b ≠ c ∧ a ≠ c ∧                    -- a, b, c are distinct
  a + b * c = 37 →                           -- given equation
  ∀ x y z : ℕ, 
    Nat.Prime x ∧ Nat.Prime y ∧ Nat.Prime z ∧
    x ≠ y ∧ y ≠ z ∧ x ≠ z ∧
    x + y * z = 37 →
    x + y - z ≤ a + b - c ∧                  -- a + b - c is maximum
    a + b - c = 32                           -- the maximum value is 32
  := by sorry

end NUMINAMATH_CALUDE_max_value_of_prime_sum_diff_l541_54139


namespace NUMINAMATH_CALUDE_product_sum_fractions_l541_54180

theorem product_sum_fractions : (3 * 4 * 5) * (1/3 + 1/4 + 1/5) = 47 := by
  sorry

end NUMINAMATH_CALUDE_product_sum_fractions_l541_54180


namespace NUMINAMATH_CALUDE_jerry_candy_boxes_l541_54178

theorem jerry_candy_boxes (initial boxes_sold boxes_left : ℕ) :
  boxes_sold = 5 →
  boxes_left = 5 →
  initial = boxes_sold + boxes_left →
  initial = 10 :=
by sorry

end NUMINAMATH_CALUDE_jerry_candy_boxes_l541_54178


namespace NUMINAMATH_CALUDE_average_speed_ratio_l541_54145

/-- Represents the average speed ratio problem -/
theorem average_speed_ratio 
  (distance_eddy : ℝ) 
  (distance_freddy : ℝ) 
  (time_eddy : ℝ) 
  (time_freddy : ℝ) 
  (h1 : distance_eddy = 600) 
  (h2 : distance_freddy = 460) 
  (h3 : time_eddy = 3) 
  (h4 : time_freddy = 4) : 
  (distance_eddy / time_eddy) / (distance_freddy / time_freddy) = 200 / 115 := by
  sorry

end NUMINAMATH_CALUDE_average_speed_ratio_l541_54145


namespace NUMINAMATH_CALUDE_diagonal_triangle_area_l541_54199

/-- Represents a rectangular prism with given face areas -/
structure RectangularPrism where
  face_area_1 : ℝ
  face_area_2 : ℝ
  face_area_3 : ℝ

/-- Calculates the area of the triangle formed by the diagonals of the prism's faces -/
noncomputable def triangle_area (prism : RectangularPrism) : ℝ :=
  sorry

/-- Theorem stating that for a rectangular prism with face areas 24, 30, and 32,
    the triangle formed by the diagonals of these faces has an area of 25 -/
theorem diagonal_triangle_area :
  let prism : RectangularPrism := ⟨24, 30, 32⟩
  triangle_area prism = 25 := by
  sorry

end NUMINAMATH_CALUDE_diagonal_triangle_area_l541_54199


namespace NUMINAMATH_CALUDE_concentric_circles_area_ratio_l541_54125

theorem concentric_circles_area_ratio : 
  let small_diameter : ℝ := 2
  let large_diameter : ℝ := 4
  let small_radius : ℝ := small_diameter / 2
  let large_radius : ℝ := large_diameter / 2
  let small_area : ℝ := π * small_radius^2
  let large_area : ℝ := π * large_radius^2
  let area_between : ℝ := large_area - small_area
  area_between / small_area = 3 := by sorry

end NUMINAMATH_CALUDE_concentric_circles_area_ratio_l541_54125


namespace NUMINAMATH_CALUDE_positive_integer_pairs_l541_54165

theorem positive_integer_pairs (a b : ℕ+) :
  (∃ k : ℤ, (a.val^3 * b.val - 1) = k * (a.val + 1)) ∧
  (∃ m : ℤ, (b.val^3 * a.val + 1) = m * (b.val - 1)) →
  ((a = 2 ∧ b = 2) ∨ (a = 1 ∧ b = 3) ∨ (a = 3 ∧ b = 3)) :=
by sorry

end NUMINAMATH_CALUDE_positive_integer_pairs_l541_54165


namespace NUMINAMATH_CALUDE_trigonometric_identities_l541_54129

open Real

theorem trigonometric_identities (α : ℝ) (h : 3 * sin α - 2 * cos α = 0) :
  ((cos α - sin α) / (cos α + sin α) + (cos α + sin α) / (cos α - sin α) = 5) ∧
  (sin α ^ 2 - 2 * sin α * cos α + 4 * cos α ^ 2 = 28 / 13) := by
  sorry

end NUMINAMATH_CALUDE_trigonometric_identities_l541_54129


namespace NUMINAMATH_CALUDE_intersection_at_one_point_l541_54107

theorem intersection_at_one_point (c : ℝ) : 
  (∃! x : ℝ, c * x^2 - 5 * x + 3 = 2 * x + 5) ↔ c = -49/8 := by
sorry

end NUMINAMATH_CALUDE_intersection_at_one_point_l541_54107


namespace NUMINAMATH_CALUDE_diamond_equation_solution_l541_54149

-- Define the diamond operation
def diamond (a b : ℚ) : ℚ := a * b + 3 * b - 2 * a

-- State the theorem
theorem diamond_equation_solution :
  ∀ y : ℚ, diamond 4 y = 50 → y = 58 / 7 := by
  sorry

end NUMINAMATH_CALUDE_diamond_equation_solution_l541_54149


namespace NUMINAMATH_CALUDE_cube_volume_from_space_diagonal_l541_54169

/-- The volume of a cube with a space diagonal of 6√3 units is 216 cubic units. -/
theorem cube_volume_from_space_diagonal :
  ∀ (s : ℝ), s > 0 → s * Real.sqrt 3 = 6 * Real.sqrt 3 → s^3 = 216 := by
  sorry

end NUMINAMATH_CALUDE_cube_volume_from_space_diagonal_l541_54169


namespace NUMINAMATH_CALUDE_happy_family_cows_count_cow_ratio_l541_54126

/-- The number of cows We the People has -/
def we_the_people_cows : ℕ := 17

/-- The total number of cows when both groups are together -/
def total_cows : ℕ := 70

/-- The number of cows Happy Good Healthy Family has -/
def happy_family_cows : ℕ := total_cows - we_the_people_cows

theorem happy_family_cows_count : happy_family_cows = 53 := by
  sorry

theorem cow_ratio : 
  (happy_family_cows : ℚ) / (we_the_people_cows : ℚ) = 53 / 17 := by
  sorry

end NUMINAMATH_CALUDE_happy_family_cows_count_cow_ratio_l541_54126


namespace NUMINAMATH_CALUDE_equation_four_real_solutions_l541_54133

theorem equation_four_real_solutions :
  ∃! (s : Finset ℝ), (∀ x ∈ s, (x^2 - 3*x - 4)^2 = 9) ∧ s.card = 4 := by
sorry

end NUMINAMATH_CALUDE_equation_four_real_solutions_l541_54133


namespace NUMINAMATH_CALUDE_area_of_similar_rectangle_l541_54158

/-- Given a rectangle R1 with one side of 4 inches and an area of 32 square inches,
    and a similar rectangle R2 with a diagonal of 10 inches,
    the area of R2 is 40 square inches. -/
theorem area_of_similar_rectangle (side_R1 area_R1 diagonal_R2 : ℝ) :
  side_R1 = 4 →
  area_R1 = 32 →
  diagonal_R2 = 10 →
  ∃ (side_a_R2 side_b_R2 : ℝ),
    side_a_R2 * side_b_R2 = 40 ∧
    side_a_R2^2 + side_b_R2^2 = diagonal_R2^2 ∧
    side_b_R2 / side_a_R2 = area_R1 / side_R1^2 :=
by sorry


end NUMINAMATH_CALUDE_area_of_similar_rectangle_l541_54158


namespace NUMINAMATH_CALUDE_triangle_angle_inequality_l541_54173

theorem triangle_angle_inequality (f : ℝ → ℝ) (α β : ℝ) : 
  (∀ x y, x ∈ [-1, 1] → y ∈ [-1, 1] → x < y → f x > f y) →  -- f is decreasing on [-1,1]
  0 < α →                                                   -- α is positive
  0 < β →                                                   -- β is positive
  α < π / 2 →                                               -- α is less than π/2
  β < π / 2 →                                               -- β is less than π/2
  α + β > π / 2 →                                           -- sum of α and β is greater than π/2
  α ≠ β →                                                   -- α and β are distinct
  f (Real.cos α) > f (Real.sin β) :=                        -- prove this inequality
by sorry

end NUMINAMATH_CALUDE_triangle_angle_inequality_l541_54173


namespace NUMINAMATH_CALUDE_fraction_of_married_men_l541_54177

theorem fraction_of_married_men (total : ℕ) (h1 : total > 0) : 
  let women := (60 : ℚ) / 100 * total
  let men := total - women
  let married := (60 : ℚ) / 100 * total
  let single_men := (3 : ℚ) / 4 * men
  (men - single_men) / men = (1 : ℚ) / 4 :=
by sorry

end NUMINAMATH_CALUDE_fraction_of_married_men_l541_54177


namespace NUMINAMATH_CALUDE_secret_number_count_l541_54157

def is_two_digit (n : ℕ) : Prop := 10 ≤ n ∧ n ≤ 99

def tens_digit (n : ℕ) : ℕ := n / 10

def units_digit (n : ℕ) : ℕ := n % 10

def secret_number (n : ℕ) : Prop :=
  is_two_digit n ∧
  Odd (tens_digit n) ∧
  Even (units_digit n) ∧
  n > 75 ∧
  n % 3 = 0

theorem secret_number_count : 
  ∃! (s : Finset ℕ), s.card = 3 ∧ ∀ n, n ∈ s ↔ secret_number n :=
sorry

end NUMINAMATH_CALUDE_secret_number_count_l541_54157


namespace NUMINAMATH_CALUDE_divisible_count_theorem_l541_54150

def count_divisible (n : ℕ) : ℕ :=
  let div2 := n / 2
  let div3 := n / 3
  let div5 := n / 5
  let div6 := n / 6
  let div10 := n / 10
  let div15 := n / 15
  let div30 := n / 30
  (div2 + div3 + div5 - div6 - div10 - div15 + div30) - div6

theorem divisible_count_theorem :
  count_divisible 1000 = 568 := by sorry

end NUMINAMATH_CALUDE_divisible_count_theorem_l541_54150


namespace NUMINAMATH_CALUDE_ones_digit_of_34_power_power_4_cycle_seventeen_power_odd_main_theorem_l541_54134

theorem ones_digit_of_34_power (n : ℕ) : n > 0 → (34^n) % 10 = (4^n) % 10 := by sorry

theorem power_4_cycle : ∀ n : ℕ, n > 0 → (4^n) % 10 = if n % 2 = 1 then 4 else 6 := by sorry

theorem seventeen_power_odd : (17^17) % 2 = 1 := by sorry

theorem main_theorem : (34^(34*(17^17))) % 10 = 4 := by sorry

end NUMINAMATH_CALUDE_ones_digit_of_34_power_power_4_cycle_seventeen_power_odd_main_theorem_l541_54134


namespace NUMINAMATH_CALUDE_angle_C_is_120_max_area_condition_l541_54167

noncomputable section

-- Define the triangle ABC
variable (A B C : ℝ) -- Angles
variable (a b c : ℝ) -- Sides

-- Define the conditions
axiom triangle_condition : (2 * a + b) * Real.cos C + c * Real.cos B = 0
axiom positive_sides : a > 0 ∧ b > 0 ∧ c > 0

-- Part 1: Prove that angle C is 120°
theorem angle_C_is_120 : C = 2 * π / 3 := by sorry

-- Part 2: Prove that when c = 4, area is maximized when a = b = (4√3)/3
theorem max_area_condition (h : c = 4) :
  (∀ a' b', a' > 0 → b' > 0 → a' * b' * Real.sin C ≤ a * b * Real.sin C) →
  a = 4 * Real.sqrt 3 / 3 ∧ b = 4 * Real.sqrt 3 / 3 := by sorry

end NUMINAMATH_CALUDE_angle_C_is_120_max_area_condition_l541_54167


namespace NUMINAMATH_CALUDE_product_of_four_numbers_l541_54108

theorem product_of_four_numbers (E F G H : ℝ) :
  E > 0 → F > 0 → G > 0 → H > 0 →
  E + F + G + H = 50 →
  E - 3 = F + 3 ∧ E - 3 = G * 3 ∧ E - 3 = H / 3 →
  E * F * G * H = 7461.9140625 := by
sorry

end NUMINAMATH_CALUDE_product_of_four_numbers_l541_54108


namespace NUMINAMATH_CALUDE_winter_hamburger_sales_l541_54168

/-- Given the total annual sales and percentages for spring and summer,
    calculate the number of hamburgers sold in winter. -/
theorem winter_hamburger_sales
  (total_sales : ℝ)
  (spring_percent : ℝ)
  (summer_percent : ℝ)
  (h_total : total_sales = 20)
  (h_spring : spring_percent = 0.3)
  (h_summer : summer_percent = 0.35) :
  total_sales - (spring_percent * total_sales + summer_percent * total_sales + (1 - spring_percent - summer_percent) / 2 * total_sales) = 3.5 :=
sorry

end NUMINAMATH_CALUDE_winter_hamburger_sales_l541_54168


namespace NUMINAMATH_CALUDE_sum_of_60_digits_eq_180_l541_54138

/-- The sum of the first 60 digits after the decimal point in the decimal expansion of 1/1234 -/
def sum_of_60_digits : ℕ :=
  -- Define the sum here
  180

/-- Theorem stating that the sum of the first 60 digits after the decimal point
    in the decimal expansion of 1/1234 is equal to 180 -/
theorem sum_of_60_digits_eq_180 :
  sum_of_60_digits = 180 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_60_digits_eq_180_l541_54138


namespace NUMINAMATH_CALUDE_A_subseteq_C_l541_54132

-- Define the universe
def U : Set ℝ := {x | -1 ≤ x ∧ x ≤ 3}

-- Define set A
def A : Set ℝ := {x | -1 ≤ x ∧ x < 3}

-- Define set B
def B : Set ℝ := {x | x^2 - 2*x - 3 = 0}

-- Define set C
def C : Set ℝ := {x | -1 < x ∧ x < 3}

-- Theorem statement
theorem A_subseteq_C : C ⊆ A := by sorry

end NUMINAMATH_CALUDE_A_subseteq_C_l541_54132


namespace NUMINAMATH_CALUDE_range_of_a_l541_54130

theorem range_of_a (x a : ℝ) : 
  (∀ x, x^2 + 2*x - 3 ≤ 0 → x ≤ a) ∧ 
  (∃ x, x^2 + 2*x - 3 ≤ 0 ∧ x > a) →
  a ≥ 1 := by
sorry

end NUMINAMATH_CALUDE_range_of_a_l541_54130


namespace NUMINAMATH_CALUDE_sqrt_gt_3x_iff_l541_54166

theorem sqrt_gt_3x_iff (x : ℝ) (h : x > 0) : 
  Real.sqrt x > 3 * x ↔ 0 < x ∧ x < 1/9 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_gt_3x_iff_l541_54166


namespace NUMINAMATH_CALUDE_original_cat_count_l541_54127

theorem original_cat_count (first_relocation second_relocation final_count : ℕ) 
  (h1 : first_relocation = 600)
  (h2 : second_relocation = (original_count - first_relocation) / 2)
  (h3 : final_count = 600)
  (h4 : final_count = original_count - first_relocation - second_relocation) :
  original_count = 1800 :=
by sorry

#check original_cat_count

end NUMINAMATH_CALUDE_original_cat_count_l541_54127


namespace NUMINAMATH_CALUDE_train_passes_jogger_train_passes_jogger_time_l541_54159

/-- The time taken for a train to pass a jogger given their speeds and initial positions -/
theorem train_passes_jogger (jogger_speed : ℝ) (train_speed : ℝ) (train_length : ℝ) (initial_distance : ℝ) : ℝ :=
  let jogger_speed_ms := jogger_speed * (1000 / 3600)
  let train_speed_ms := train_speed * (1000 / 3600)
  let relative_speed := train_speed_ms - jogger_speed_ms
  let total_distance := initial_distance + train_length
  total_distance / relative_speed

/-- The time taken for the train to pass the jogger is 40 seconds -/
theorem train_passes_jogger_time : train_passes_jogger 9 45 200 200 = 40 := by
  sorry

end NUMINAMATH_CALUDE_train_passes_jogger_train_passes_jogger_time_l541_54159


namespace NUMINAMATH_CALUDE_literature_club_students_l541_54128

theorem literature_club_students (total : ℕ) (english : ℕ) (french : ℕ) (both : ℕ) 
  (h_total : total = 120)
  (h_english : english = 72)
  (h_french : french = 52)
  (h_both : both = 12) :
  total - (english + french - both) = 8 := by
  sorry

end NUMINAMATH_CALUDE_literature_club_students_l541_54128


namespace NUMINAMATH_CALUDE_functional_equation_implies_odd_l541_54146

/-- A function satisfying the given functional equation -/
def FunctionalEquation (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, f (x * f y) = y * f x

/-- Theorem stating that f(-x) = -f(x) for functions satisfying the functional equation -/
theorem functional_equation_implies_odd (f : ℝ → ℝ) (h : FunctionalEquation f) :
  ∀ x : ℝ, f (-x) = -f x := by
  sorry

end NUMINAMATH_CALUDE_functional_equation_implies_odd_l541_54146


namespace NUMINAMATH_CALUDE_afternoon_eggs_count_l541_54111

def initial_eggs : ℕ := 20
def morning_eggs : ℕ := 4
def remaining_eggs : ℕ := 13

theorem afternoon_eggs_count : initial_eggs - morning_eggs - remaining_eggs = 3 := by
  sorry

end NUMINAMATH_CALUDE_afternoon_eggs_count_l541_54111


namespace NUMINAMATH_CALUDE_students_playing_both_football_and_cricket_l541_54153

/-- The number of students playing both football and cricket -/
def students_playing_both (total students_football students_cricket students_neither : ℕ) : ℕ :=
  students_football + students_cricket - (total - students_neither)

/-- Proof that 140 students play both football and cricket -/
theorem students_playing_both_football_and_cricket :
  students_playing_both 410 325 175 50 = 140 := by
  sorry

end NUMINAMATH_CALUDE_students_playing_both_football_and_cricket_l541_54153


namespace NUMINAMATH_CALUDE_vector_simplification_l541_54172

/-- Given four points A, B, C, and D in a vector space, 
    prove that the vector AB minus DC minus CB equals AD -/
theorem vector_simplification (V : Type*) [AddCommGroup V] 
  (A B C D : V) : 
  (B - A) - (C - D) - (B - C) = D - A := by sorry

end NUMINAMATH_CALUDE_vector_simplification_l541_54172


namespace NUMINAMATH_CALUDE_instantaneous_speed_at_4_l541_54184

-- Define the motion equation
def s (t : ℝ) : ℝ := t^2 - 2*t + 5

-- Define the instantaneous speed (derivative of s)
def v (t : ℝ) : ℝ := 2*t - 2

-- Theorem: The instantaneous speed at t = 4 is 6 m/s
theorem instantaneous_speed_at_4 : v 4 = 6 := by
  sorry

end NUMINAMATH_CALUDE_instantaneous_speed_at_4_l541_54184


namespace NUMINAMATH_CALUDE_greatest_divisor_with_remainders_l541_54115

theorem greatest_divisor_with_remainders (d : ℕ) : d > 0 ∧ 
  (∃ q1 : ℤ, 4351 = d * q1 + 8) ∧ 
  (∃ r1 : ℤ, 5161 = d * r1 + 10) ∧ 
  (∀ n : ℕ, n > d → 
    (∃ q2 : ℤ, 4351 = n * q2 + 8) ∧ 
    (∃ r2 : ℤ, 5161 = n * r2 + 10) → n = d) → 
  d = 1 := by
sorry

end NUMINAMATH_CALUDE_greatest_divisor_with_remainders_l541_54115


namespace NUMINAMATH_CALUDE_louise_oranges_l541_54192

theorem louise_oranges (num_boxes : ℕ) (oranges_per_box : ℕ) 
  (h1 : num_boxes = 7) 
  (h2 : oranges_per_box = 6) : 
  num_boxes * oranges_per_box = 42 := by
  sorry

end NUMINAMATH_CALUDE_louise_oranges_l541_54192


namespace NUMINAMATH_CALUDE_girls_after_joining_l541_54190

def initial_girls : ℕ := 732
def new_girls : ℕ := 682

theorem girls_after_joining (initial_girls new_girls : ℕ) : 
  initial_girls + new_girls = 1414 :=
sorry

end NUMINAMATH_CALUDE_girls_after_joining_l541_54190


namespace NUMINAMATH_CALUDE_expression_simplification_and_evaluation_l541_54140

theorem expression_simplification_and_evaluation (a b : ℚ) 
  (ha : a = -2) (hb : b = 3/2) : 
  1/2 * a - 2 * (a - 1/2 * b^2) - (3/2 * a - 1/3 * b^2) = 9 := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_and_evaluation_l541_54140


namespace NUMINAMATH_CALUDE_ellipse_hyperbola_tangency_l541_54124

-- Define the ellipse equation
def ellipse (x y n : ℝ) : Prop := x^2 + n*(y-1)^2 = n

-- Define the hyperbola equation
def hyperbola (x y : ℝ) : Prop := x^2 - 4*(y+3)^2 = 4

-- Define the tangency condition (discriminant = 0)
def tangent_condition (n : ℝ) : Prop := (24-2*n)^2 - 4*(4+n)*40 = 0

-- Theorem statement
theorem ellipse_hyperbola_tangency :
  ∃ n₁ n₂ : ℝ, 
    (abs (n₁ - 62.20625) < 0.00001) ∧ 
    (abs (n₂ - 1.66875) < 0.00001) ∧
    (∀ x y : ℝ, ellipse x y n₁ ∧ hyperbola x y → tangent_condition n₁) ∧
    (∀ x y : ℝ, ellipse x y n₂ ∧ hyperbola x y → tangent_condition n₂) :=
sorry

end NUMINAMATH_CALUDE_ellipse_hyperbola_tangency_l541_54124


namespace NUMINAMATH_CALUDE_min_players_sum_divisible_by_10_l541_54188

/-- Represents a 3x9 grid of distinct non-negative integers -/
def Grid := Matrix (Fin 3) (Fin 9) ℕ

/-- Predicate to check if all elements in a grid are distinct -/
def all_distinct (g : Grid) : Prop :=
  ∀ i j i' j', (i ≠ i' ∨ j ≠ j') → g i j ≠ g i' j'

/-- Predicate to check if a sum is divisible by 10 -/
def sum_divisible_by_10 (a b : ℕ) : Prop :=
  (a + b) % 10 = 0

/-- Main theorem statement -/
theorem min_players_sum_divisible_by_10 (g : Grid) (h : all_distinct g) :
  ∃ i j i' j', sum_divisible_by_10 (g i j) (g i' j') :=
sorry

end NUMINAMATH_CALUDE_min_players_sum_divisible_by_10_l541_54188


namespace NUMINAMATH_CALUDE_stamp_cost_problem_l541_54191

theorem stamp_cost_problem (total_stamps : ℕ) (high_denom : ℕ) (total_cost : ℚ) (high_denom_count : ℕ) :
  total_stamps = 20 →
  high_denom = 37 →
  total_cost = 706/100 →
  high_denom_count = 18 →
  ∃ (low_denom : ℕ),
    low_denom * (total_stamps - high_denom_count) = (total_cost * 100 - high_denom * high_denom_count : ℚ) ∧
    low_denom = 20 :=
by sorry

end NUMINAMATH_CALUDE_stamp_cost_problem_l541_54191


namespace NUMINAMATH_CALUDE_eggs_per_plate_count_l541_54186

def breakfast_plate (num_customers : ℕ) (total_bacon : ℕ) : ℕ → Prop :=
  λ eggs_per_plate : ℕ =>
    eggs_per_plate > 0 ∧
    2 * eggs_per_plate * num_customers = total_bacon

theorem eggs_per_plate_count (num_customers : ℕ) (total_bacon : ℕ) 
    (h1 : num_customers = 14) (h2 : total_bacon = 56) :
    ∃ eggs_per_plate : ℕ, breakfast_plate num_customers total_bacon eggs_per_plate ∧ 
    eggs_per_plate = 2 := by
  sorry

end NUMINAMATH_CALUDE_eggs_per_plate_count_l541_54186


namespace NUMINAMATH_CALUDE_archaeopteryx_humerus_estimate_l541_54174

/-- Represents the linear regression equation for Archaeopteryx fossil specimens -/
def archaeopteryx_regression (x : ℝ) : ℝ := 1.197 * x - 3.660

/-- Theorem stating the estimated humerus length for a given femur length -/
theorem archaeopteryx_humerus_estimate :
  archaeopteryx_regression 50 = 56.19 := by
  sorry

end NUMINAMATH_CALUDE_archaeopteryx_humerus_estimate_l541_54174


namespace NUMINAMATH_CALUDE_intersection_equals_open_interval_l541_54135

-- Define the sets M and N
def M : Set ℝ := {x : ℝ | |x - 1| < 2}
def N : Set ℝ := {x : ℝ | x < 2}

-- Define the open interval (-1, 2)
def open_interval : Set ℝ := {x : ℝ | -1 < x ∧ x < 2}

-- Theorem statement
theorem intersection_equals_open_interval : M ∩ N = open_interval := by
  sorry

end NUMINAMATH_CALUDE_intersection_equals_open_interval_l541_54135


namespace NUMINAMATH_CALUDE_num_paths_upper_bound_l541_54143

/-- Represents a rectangular grid city -/
structure City where
  length : ℕ
  width : ℕ

/-- The number of possible paths from southwest to northeast corner -/
def num_paths (c : City) : ℕ := sorry

/-- The theorem to be proved -/
theorem num_paths_upper_bound (c : City) :
  num_paths c ≤ 2^(c.length * c.width) := by sorry

end NUMINAMATH_CALUDE_num_paths_upper_bound_l541_54143


namespace NUMINAMATH_CALUDE_tangent_line_equation_l541_54114

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := x^2 * (x - a)

-- Define the derivative of f
def f' (a : ℝ) (x : ℝ) : ℝ := 3 * x^2 - 2 * a * x

-- Theorem statement
theorem tangent_line_equation (a : ℝ) (h : f' a 1 = 3) :
  ∃ (m b : ℝ), m * 1 - b = f a 1 ∧ 
                ∀ x, m * x - b = 3 * x - 2 :=
sorry

end NUMINAMATH_CALUDE_tangent_line_equation_l541_54114


namespace NUMINAMATH_CALUDE_base_8_to_10_98765_l541_54104

-- Define the base-8 number as a list of digits
def base_8_number : List Nat := [9, 8, 7, 6, 5]

-- Define the function to convert a base-8 number to base-10
def base_8_to_10 (digits : List Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * (8 ^ (digits.length - 1 - i))) 0

-- Theorem statement
theorem base_8_to_10_98765 :
  base_8_to_10 base_8_number = 41461 := by
  sorry

end NUMINAMATH_CALUDE_base_8_to_10_98765_l541_54104


namespace NUMINAMATH_CALUDE_circle_area_through_points_l541_54171

/-- The area of a circle with center P(-5, 3) passing through Q(7, -2) is 169π -/
theorem circle_area_through_points :
  let P : ℝ × ℝ := (-5, 3)
  let Q : ℝ × ℝ := (7, -2)
  let r := Real.sqrt ((P.1 - Q.1)^2 + (P.2 - Q.2)^2)
  π * r^2 = 169 * π := by
  sorry

end NUMINAMATH_CALUDE_circle_area_through_points_l541_54171


namespace NUMINAMATH_CALUDE_approx_cube_root_2370_l541_54106

-- Define the approximation relation
def approx (x y : ℝ) := ∃ ε > 0, |x - y| < ε

-- Define the cube root function
noncomputable def cubeRoot (x : ℝ) : ℝ := Real.rpow x (1/3)

-- State the theorem
theorem approx_cube_root_2370 (h : approx (cubeRoot 2.37) 1.333) :
  approx (cubeRoot 2370) 13.33 := by
  sorry

end NUMINAMATH_CALUDE_approx_cube_root_2370_l541_54106


namespace NUMINAMATH_CALUDE_mings_estimate_smaller_l541_54182

theorem mings_estimate_smaller (x y δ : ℝ) (hx : x > y) (hy : y > 0) (hδ : δ > 0) :
  (x + δ) - (y + 2*δ) < x - y := by
  sorry

end NUMINAMATH_CALUDE_mings_estimate_smaller_l541_54182


namespace NUMINAMATH_CALUDE_root_shift_theorem_l541_54147

/-- Given a, b, and c are roots of x³ - 5x + 7 = 0, prove that a+3, b+3, and c+3 are roots of x³ - 9x² + 22x - 5 = 0 -/
theorem root_shift_theorem (a b c : ℝ) : 
  (a^3 - 5*a + 7 = 0) → 
  (b^3 - 5*b + 7 = 0) → 
  (c^3 - 5*c + 7 = 0) → 
  ((a+3)^3 - 9*(a+3)^2 + 22*(a+3) - 5 = 0) ∧
  ((b+3)^3 - 9*(b+3)^2 + 22*(b+3) - 5 = 0) ∧
  ((c+3)^3 - 9*(c+3)^2 + 22*(c+3) - 5 = 0) := by
  sorry


end NUMINAMATH_CALUDE_root_shift_theorem_l541_54147


namespace NUMINAMATH_CALUDE_concert_songs_theorem_l541_54181

/-- Represents the number of songs sung by each girl -/
structure SongCount where
  mary : ℕ
  alina : ℕ
  tina : ℕ
  hanna : ℕ
  lucy : ℕ

/-- The total number of songs sung by the trios -/
def total_songs (s : SongCount) : ℕ :=
  (s.mary + s.alina + s.tina + s.hanna + s.lucy) / 3

/-- The conditions given in the problem -/
def satisfies_conditions (s : SongCount) : Prop :=
  s.hanna = 9 ∧
  s.lucy = 5 ∧
  s.mary > s.lucy ∧ s.mary < s.hanna ∧
  s.alina > s.lucy ∧ s.alina < s.hanna ∧
  s.tina > s.lucy ∧ s.tina < s.hanna

theorem concert_songs_theorem (s : SongCount) :
  satisfies_conditions s → total_songs s = 11 := by
  sorry

end NUMINAMATH_CALUDE_concert_songs_theorem_l541_54181


namespace NUMINAMATH_CALUDE_imaginary_part_of_complex_fraction_l541_54120

theorem imaginary_part_of_complex_fraction (i : ℂ) :
  i * i = -1 →
  Complex.im ((1 + 2*i) / (i - 1)) = -3/2 := by
sorry

end NUMINAMATH_CALUDE_imaginary_part_of_complex_fraction_l541_54120


namespace NUMINAMATH_CALUDE_tom_completion_time_l541_54102

/-- The time it takes Tom to complete a wall on his own after working with Avery for one hour -/
theorem tom_completion_time (avery_rate tom_rate : ℚ) : 
  avery_rate = 1/2 →  -- Avery's rate in walls per hour
  tom_rate = 1/4 →    -- Tom's rate in walls per hour
  (avery_rate + tom_rate) * 1 = 3/4 →  -- Combined work in first hour
  (1 - (avery_rate + tom_rate) * 1) / tom_rate = 1 := by
  sorry

end NUMINAMATH_CALUDE_tom_completion_time_l541_54102


namespace NUMINAMATH_CALUDE_triangle_properties_l541_54100

-- Define the triangle ABC
structure Triangle where
  A : Real
  B : Real
  C : Real
  a : Real
  b : Real
  c : Real

-- Define the theorem
theorem triangle_properties (t : Triangle) 
  (h1 : 2 * Real.cos t.C * (t.a * Real.cos t.B + t.b * Real.cos t.A) = t.c)
  (h2 : t.c = Real.sqrt 7)
  (h3 : (1/2) * t.a * t.b * Real.sin t.C = (3 * Real.sqrt 3) / 2) :
  t.C = π/3 ∧ t.a + t.b + t.c = 5 + Real.sqrt 7 := by
  sorry


end NUMINAMATH_CALUDE_triangle_properties_l541_54100


namespace NUMINAMATH_CALUDE_short_trees_after_planting_verify_total_short_trees_l541_54119

/-- The number of short trees in the park after planting -/
def total_short_trees (current_short_trees new_short_trees : ℕ) : ℕ :=
  current_short_trees + new_short_trees

/-- Theorem: The total number of short trees after planting is the sum of current and new short trees -/
theorem short_trees_after_planting 
  (current_short_trees : ℕ) (new_short_trees : ℕ) :
  total_short_trees current_short_trees new_short_trees = current_short_trees + new_short_trees :=
by sorry

/-- The correct number of short trees after planting, given the problem conditions -/
def correct_total : ℕ := 98

/-- Theorem: The total number of short trees after planting, given the problem conditions, is 98 -/
theorem verify_total_short_trees :
  total_short_trees 41 57 = correct_total :=
by sorry

end NUMINAMATH_CALUDE_short_trees_after_planting_verify_total_short_trees_l541_54119


namespace NUMINAMATH_CALUDE_book_sale_fraction_l541_54109

/-- Given a book sale where some books were sold for $2 each, 36 books remained unsold,
    and the total amount received was $144, prove that 2/3 of the books were sold. -/
theorem book_sale_fraction (B : ℕ) (h1 : B > 36) : 
  2 * (B - 36) = 144 → (B - 36 : ℚ) / B = 2/3 := by
  sorry

end NUMINAMATH_CALUDE_book_sale_fraction_l541_54109


namespace NUMINAMATH_CALUDE_base_nine_representation_l541_54117

theorem base_nine_representation (b : ℕ) : 
  (777 : ℕ) = 1 * b^3 + 0 * b^2 + 5 * b^1 + 3 * b^0 ∧ 
  b > 1 ∧ 
  b^3 ≤ 777 ∧ 
  777 < b^4 ∧
  (∃ (A C : ℕ), A ≠ C ∧ A < b ∧ C < b ∧ 
    777 = A * b^3 + C * b^2 + A * b^1 + C * b^0) →
  b = 9 := by
sorry

end NUMINAMATH_CALUDE_base_nine_representation_l541_54117


namespace NUMINAMATH_CALUDE_circle_radius_l541_54160

theorem circle_radius (x y : ℝ) : 
  (2 * x^2 + 2 * y^2 - 4 * x + 6 * y = 3/2) → 
  ∃ (h k r : ℝ), r = 2 ∧ (x - h)^2 + (y - k)^2 = r^2 := by
sorry

end NUMINAMATH_CALUDE_circle_radius_l541_54160


namespace NUMINAMATH_CALUDE_quadratic_function_range_l541_54195

/-- Given a quadratic function f(x) = x^2 + ax + 5 that is symmetric about x = -2
    and has a range of [1, 5] on the interval [m, 0], prove that -4 ≤ m ≤ -2. -/
theorem quadratic_function_range (a : ℝ) (m : ℝ) (h_m : m < 0) :
  (∀ x, ((-2 + x)^2 + a*(-2 + x) + 5 = (-2 - x)^2 + a*(-2 - x) + 5)) →
  (∀ x ∈ Set.Icc m 0, 1 ≤ x^2 + a*x + 5 ∧ x^2 + a*x + 5 ≤ 5) →
  -4 ≤ m ∧ m ≤ -2 := by sorry

end NUMINAMATH_CALUDE_quadratic_function_range_l541_54195


namespace NUMINAMATH_CALUDE_min_distance_of_sine_extrema_l541_54187

open Real

theorem min_distance_of_sine_extrema :
  ∀ (f : ℝ → ℝ) (x₁ x₂ : ℝ),
  (∀ x, f x = sin (π * x)) →
  (∀ x, f x₁ ≤ f x ∧ f x ≤ f x₂) →
  (∃ (d : ℝ), d > 0 ∧ ∀ (y₁ y₂ : ℝ), (∀ x, f y₁ ≤ f x ∧ f x ≤ f y₂) → |y₁ - y₂| ≥ d) →
  (∀ (y₁ y₂ : ℝ), (∀ x, f y₁ ≤ f x ∧ f x ≤ f y₂) → |y₁ - y₂| ≥ 1) →
  |x₁ - x₂| = 1 := by
sorry

end NUMINAMATH_CALUDE_min_distance_of_sine_extrema_l541_54187


namespace NUMINAMATH_CALUDE_min_value_theorem_l541_54193

theorem min_value_theorem (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a > b) (heq : a * b = 1 / 2) :
  (4 * a^2 + b^2 + 1) / (2 * a - b) ≥ 2 * Real.sqrt 3 :=
by sorry

end NUMINAMATH_CALUDE_min_value_theorem_l541_54193


namespace NUMINAMATH_CALUDE_collinear_points_k_value_l541_54113

/-- Given three points A, B, and C in 2D space, this function checks if they are collinear --/
def are_collinear (A B C : ℝ × ℝ) : Prop :=
  let AB := (B.1 - A.1, B.2 - A.2)
  let BC := (C.1 - B.1, C.2 - B.2)
  AB.1 * BC.2 = AB.2 * BC.1

/-- Theorem stating that if A(k, 12), B(4, 5), and C(10, k) are collinear, then k = 11 or k = -2 --/
theorem collinear_points_k_value (k : ℝ) :
  are_collinear (k, 12) (4, 5) (10, k) → k = 11 ∨ k = -2 := by
  sorry


end NUMINAMATH_CALUDE_collinear_points_k_value_l541_54113


namespace NUMINAMATH_CALUDE_output_value_scientific_notation_l541_54198

/-- Represents a number in scientific notation -/
structure ScientificNotation where
  coefficient : ℝ
  exponent : ℤ
  h_coeff : 1 ≤ coefficient ∧ coefficient < 10

/-- Converts a real number to scientific notation -/
def toScientificNotation (x : ℝ) : ScientificNotation :=
  sorry

theorem output_value_scientific_notation :
  toScientificNotation 110000000000 = ScientificNotation.mk 1.1 10 (by norm_num) :=
sorry

end NUMINAMATH_CALUDE_output_value_scientific_notation_l541_54198


namespace NUMINAMATH_CALUDE_john_annual_profit_l541_54141

/-- Calculates John's annual profit from subletting his apartment -/
def annual_profit (tenant_a_rent tenant_b_rent tenant_c_rent john_rent utilities maintenance : ℕ) : ℕ :=
  let monthly_income := tenant_a_rent + tenant_b_rent + tenant_c_rent
  let monthly_expenses := john_rent + utilities + maintenance
  let monthly_profit := monthly_income - monthly_expenses
  12 * monthly_profit

/-- Theorem stating John's annual profit given his rental income and expenses -/
theorem john_annual_profit :
  annual_profit 350 400 450 900 100 50 = 1800 := by
  sorry

end NUMINAMATH_CALUDE_john_annual_profit_l541_54141


namespace NUMINAMATH_CALUDE_binary_to_decimal_conversion_l541_54183

/-- Converts a list of binary digits to a natural number. -/
def binaryToNat (digits : List Bool) : ℕ :=
  digits.foldr (fun b n => 2 * n + if b then 1 else 0) 0

/-- The binary representation of the number we want to convert. -/
def binaryNumber : List Bool :=
  [true, true, true, false, true, true, false, false, true, false, false, true]

theorem binary_to_decimal_conversion :
  binaryToNat binaryNumber = 3785 := by
  sorry

end NUMINAMATH_CALUDE_binary_to_decimal_conversion_l541_54183


namespace NUMINAMATH_CALUDE_unique_m_exists_l541_54137

theorem unique_m_exists : ∃! m : ℤ,
  30 ≤ m ∧ m ≤ 80 ∧
  ∃ k : ℤ, m = 6 * k ∧
  m % 8 = 2 ∧
  m % 5 = 2 ∧
  m = 42 := by sorry

end NUMINAMATH_CALUDE_unique_m_exists_l541_54137


namespace NUMINAMATH_CALUDE_function_translation_transformation_result_l541_54162

-- Define the original function
def f (x : ℝ) : ℝ := 2 * (x + 1)^2 - 3

-- Define the transformed function
def g (x : ℝ) : ℝ := 2 * x^2

-- Theorem stating that g is the result of translating f
theorem function_translation (x : ℝ) : 
  g x = f (x - 1) + 3 := by
  sorry

-- Prove that the transformation results in g
theorem transformation_result : 
  ∀ x, g x = 2 * x^2 := by
  sorry

end NUMINAMATH_CALUDE_function_translation_transformation_result_l541_54162


namespace NUMINAMATH_CALUDE_star_emilio_sum_difference_l541_54101

/-- The sum of numbers from 1 to 50 -/
def starSum : ℕ := (List.range 50).map (· + 1) |>.sum

/-- The sum of numbers from 1 to 50 with '3' replaced by '2' -/
def emilioSum : ℕ := (List.range 50).map (· + 1) |>.map (replaceThreeWithTwo) |>.sum
  where
    replaceThreeWithTwo (n : ℕ) : ℕ :=
      let tens := n / 10
      let ones := n % 10
      if tens = 3 then 20 + ones
      else if ones = 3 then 10 * tens + 2
      else n

/-- The difference between Star's sum and Emilio's sum is 105 -/
theorem star_emilio_sum_difference : starSum - emilioSum = 105 := by
  sorry

end NUMINAMATH_CALUDE_star_emilio_sum_difference_l541_54101


namespace NUMINAMATH_CALUDE_power_of_eight_division_l541_54144

theorem power_of_eight_division (n : ℕ) : 8^(n+1) / 8 = 8^n := by
  sorry

end NUMINAMATH_CALUDE_power_of_eight_division_l541_54144


namespace NUMINAMATH_CALUDE_brown_eyed_brunettes_l541_54163

/-- The total number of girls -/
def total_girls : ℕ := 60

/-- The number of green-eyed redheads -/
def green_eyed_redheads : ℕ := 20

/-- The number of brunettes -/
def brunettes : ℕ := 35

/-- The number of brown-eyed girls -/
def brown_eyed : ℕ := 25

/-- Theorem: The number of brown-eyed brunettes is 20 -/
theorem brown_eyed_brunettes : 
  total_girls - (green_eyed_redheads + (brunettes - (brown_eyed - (total_girls - brunettes - green_eyed_redheads)))) = 20 := by
  sorry

end NUMINAMATH_CALUDE_brown_eyed_brunettes_l541_54163


namespace NUMINAMATH_CALUDE_collinear_points_iff_k_eq_neg_ten_l541_54112

/-- Three points in R² are collinear if the slope between any two pairs of points is equal. -/
def collinear (p1 p2 p3 : ℝ × ℝ) : Prop :=
  (p2.2 - p1.2) * (p3.1 - p2.1) = (p3.2 - p2.2) * (p2.1 - p1.1)

/-- The theorem states that the points (1, -2), (3, k), and (6, 2k - 2) are collinear 
    if and only if k = -10. -/
theorem collinear_points_iff_k_eq_neg_ten :
  ∀ k : ℝ, collinear (1, -2) (3, k) (6, 2*k - 2) ↔ k = -10 := by
  sorry

end NUMINAMATH_CALUDE_collinear_points_iff_k_eq_neg_ten_l541_54112


namespace NUMINAMATH_CALUDE_max_value_of_quadratic_l541_54148

theorem max_value_of_quadratic (x : ℝ) (h1 : 0 < x) (h2 : x < 3/2) :
  x * (3 - 2*x) ≤ 9/8 :=
by sorry

end NUMINAMATH_CALUDE_max_value_of_quadratic_l541_54148


namespace NUMINAMATH_CALUDE_bird_cage_problem_l541_54131

theorem bird_cage_problem (initial_birds : ℕ) : 
  (1 / 3 : ℚ) * (3 / 5 : ℚ) * (1 / 3 : ℚ) * initial_birds = 8 →
  initial_birds = 60 := by
sorry

end NUMINAMATH_CALUDE_bird_cage_problem_l541_54131


namespace NUMINAMATH_CALUDE_certain_number_value_l541_54121

theorem certain_number_value (t b c : ℝ) :
  (t + b + c + 14 + 15) / 5 = 12 ∧ (t + b + c + 29) / 4 = 15 → 14 = 14 := by
  sorry

end NUMINAMATH_CALUDE_certain_number_value_l541_54121


namespace NUMINAMATH_CALUDE_geometric_sequence_ratio_sum_l541_54110

theorem geometric_sequence_ratio_sum (k p r : ℝ) (hk : k ≠ 0) (hp : p ≠ 1) (hr : r ≠ 1) (hpr : p ≠ r) :
  k * p^2 - k * r^2 = 5 * (k * p - k * r) → p + r = 5 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_ratio_sum_l541_54110


namespace NUMINAMATH_CALUDE_computer_table_cost_price_l541_54161

/-- The cost price of a computer table, given the selling price and markup percentage. -/
def cost_price (selling_price : ℚ) (markup_percentage : ℚ) : ℚ :=
  selling_price / (1 + markup_percentage)

/-- Theorem: The cost price of the computer table is 6500, given the conditions. -/
theorem computer_table_cost_price :
  let selling_price : ℚ := 8450
  let markup_percentage : ℚ := 0.30
  cost_price selling_price markup_percentage = 6500 := by
sorry

end NUMINAMATH_CALUDE_computer_table_cost_price_l541_54161


namespace NUMINAMATH_CALUDE_probability_one_from_each_set_l541_54189

theorem probability_one_from_each_set (n : ℕ) :
  let total := 2 * n
  let prob_first := n / total
  let prob_second := n / (total - 1)
  2 * (prob_first * prob_second) = n / (n + 1) :=
by
  sorry

#check probability_one_from_each_set 6

end NUMINAMATH_CALUDE_probability_one_from_each_set_l541_54189


namespace NUMINAMATH_CALUDE_brown_eggs_survived_l541_54196

/-- Given that Linda initially had three times as many white eggs as brown eggs,
    and after dropping her basket she ended up with a dozen eggs in total,
    prove that 3 brown eggs survived the fall. -/
theorem brown_eggs_survived (white_eggs brown_eggs : ℕ) : 
  white_eggs = 3 * brown_eggs →  -- Initial condition
  white_eggs + brown_eggs = 12 →  -- Total eggs after the fall
  brown_eggs > 0 →  -- Some brown eggs survived
  brown_eggs = 3 := by
  sorry

end NUMINAMATH_CALUDE_brown_eggs_survived_l541_54196


namespace NUMINAMATH_CALUDE_scientific_notation_260000_l541_54123

/-- Scientific notation representation of a number -/
structure ScientificNotation where
  coefficient : ℝ
  exponent : ℤ
  h1 : 1 ≤ |coefficient| ∧ |coefficient| < 10

/-- Check if a ScientificNotation represents a given real number -/
def represents (sn : ScientificNotation) (x : ℝ) : Prop :=
  x = sn.coefficient * (10 : ℝ) ^ sn.exponent

/-- The number 260000 in scientific notation -/
def n : ScientificNotation :=
  { coefficient := 2.6
    exponent := 5
    h1 := by sorry }

theorem scientific_notation_260000 :
  represents n 260000 := by sorry

end NUMINAMATH_CALUDE_scientific_notation_260000_l541_54123


namespace NUMINAMATH_CALUDE_f_decreasing_interval_a_upper_bound_l541_54176

-- Define the function f(x) = x ln x
noncomputable def f (x : ℝ) : ℝ := x * Real.log x

-- Theorem for the monotonically decreasing interval
theorem f_decreasing_interval :
  ∀ x ∈ Set.Ioo (0 : ℝ) (Real.exp (-1)),
  StrictMonoOn f (Set.Ioo 0 (Real.exp (-1))) :=
sorry

-- Theorem for the range of a
theorem a_upper_bound
  (h : ∀ x > 0, f x ≥ -x^2 + a*x - 6) :
  a ≤ 5 + Real.log 2 :=
sorry

end NUMINAMATH_CALUDE_f_decreasing_interval_a_upper_bound_l541_54176


namespace NUMINAMATH_CALUDE_integer_pair_gcd_equation_l541_54155

theorem integer_pair_gcd_equation :
  ∀ x y : ℕ+, 
    (x.val * y.val * Nat.gcd x.val y.val = x.val + y.val + (Nat.gcd x.val y.val)^2) ↔ 
    ((x, y) = (2, 2) ∨ (x, y) = (2, 3) ∨ (x, y) = (3, 2)) := by
  sorry

end NUMINAMATH_CALUDE_integer_pair_gcd_equation_l541_54155


namespace NUMINAMATH_CALUDE_north_village_conscripts_l541_54154

/-- The number of people to be conscripted from a village, given its population and the total population and conscription numbers. -/
def conscriptsFromVillage (villagePopulation totalPopulation totalConscripts : ℕ) : ℕ :=
  (villagePopulation * totalConscripts) / totalPopulation

/-- Theorem stating that given the specific village populations and total conscripts, 
    the number of conscripts from the north village is 108. -/
theorem north_village_conscripts :
  let northPopulation : ℕ := 8100
  let westPopulation : ℕ := 7488
  let southPopulation : ℕ := 6912
  let totalConscripts : ℕ := 300
  let totalPopulation : ℕ := northPopulation + westPopulation + southPopulation
  conscriptsFromVillage northPopulation totalPopulation totalConscripts = 108 := by
  sorry

end NUMINAMATH_CALUDE_north_village_conscripts_l541_54154


namespace NUMINAMATH_CALUDE_fraction_calculation_l541_54197

theorem fraction_calculation : 
  (2 / 7 + 5 / 8 * 1 / 3) / (3 / 4 - 2 / 9) = 15 / 16 := by
  sorry

end NUMINAMATH_CALUDE_fraction_calculation_l541_54197


namespace NUMINAMATH_CALUDE_linear_system_solution_l541_54194

-- Define the determinant function
def det2x2 (a b c d : ℝ) : ℝ := a * d - b * c

-- Define the system of linear equations
def system (x y : ℝ) : Prop := 2 * x + y = 1 ∧ 3 * x - 2 * y = 12

-- State the theorem
theorem linear_system_solution :
  let D := det2x2 2 1 3 (-2)
  let Dx := det2x2 1 1 12 (-2)
  let Dy := det2x2 2 1 3 12
  D = -7 ∧ Dx = -14 ∧ Dy = 21 ∧ system (Dx / D) (Dy / D) ∧ system 2 (-3) := by
  sorry


end NUMINAMATH_CALUDE_linear_system_solution_l541_54194


namespace NUMINAMATH_CALUDE_city_population_l541_54118

/-- If 96% of a city's population is 23040, then the total population is 24000. -/
theorem city_population (population : ℕ) : 
  (96 : ℚ) / 100 * population = 23040 → population = 24000 := by
  sorry

end NUMINAMATH_CALUDE_city_population_l541_54118


namespace NUMINAMATH_CALUDE_inscribed_circle_radius_l541_54164

/-- For a right triangle with legs a and b, hypotenuse c, and an inscribed circle of radius r -/
def RightTriangleWithInscribedCircle (a b c r : ℝ) : Prop :=
  a > 0 ∧ b > 0 ∧ c > 0 ∧ r > 0 ∧ a^2 + b^2 = c^2

/-- The radius of the inscribed circle in a right triangle is equal to (a + b - c) / 2 -/
theorem inscribed_circle_radius 
  (a b c r : ℝ) 
  (h : RightTriangleWithInscribedCircle a b c r) : 
  r = (a + b - c) / 2 := by
  sorry

end NUMINAMATH_CALUDE_inscribed_circle_radius_l541_54164


namespace NUMINAMATH_CALUDE_no_prime_sum_10003_l541_54170

/-- A function that returns the number of ways to write n as the sum of two primes -/
def countPrimeSumWays (n : ℕ) : ℕ :=
  (Finset.filter (fun p => Nat.Prime p ∧ Nat.Prime (n - p)) (Finset.range n)).card

/-- Theorem stating that 10003 cannot be written as the sum of two primes -/
theorem no_prime_sum_10003 : countPrimeSumWays 10003 = 0 := by
  sorry

end NUMINAMATH_CALUDE_no_prime_sum_10003_l541_54170


namespace NUMINAMATH_CALUDE_original_ratio_l541_54136

theorem original_ratio (x y : ℕ) (h1 : y = 16) (h2 : x + 12 = y) :
  ∃ (a b : ℕ), a = 1 ∧ b = 4 ∧ x * b = y * a :=
by sorry

end NUMINAMATH_CALUDE_original_ratio_l541_54136


namespace NUMINAMATH_CALUDE_system_solution_l541_54152

theorem system_solution (a b c k x y z : ℝ) 
  (h1 : a * x + b * y + c * z = k)
  (h2 : a^2 * x + b^2 * y + c^2 * z = k^2)
  (h3 : a^3 * x + b^3 * y + c^3 * z = k^3)
  (hab : a ≠ b) (hbc : b ≠ c) (hca : c ≠ a) :
  x = k * (k - c) * (k - b) / (a * (a - c) * (a - b)) ∧
  y = k * (k - c) * (k - a) / (b * (b - c) * (b - a)) ∧
  z = k * (k - a) * (k - b) / (c * (c - a) * (c - b)) := by
  sorry

end NUMINAMATH_CALUDE_system_solution_l541_54152


namespace NUMINAMATH_CALUDE_mean_of_points_l541_54103

def points : List ℝ := [81, 73, 83, 86, 73]

theorem mean_of_points : (points.sum / points.length : ℝ) = 79.2 := by
  sorry

end NUMINAMATH_CALUDE_mean_of_points_l541_54103


namespace NUMINAMATH_CALUDE_rope_triangle_probability_l541_54116

/-- The probability of forming a triangle from three rope segments --/
theorem rope_triangle_probability (L : ℝ) (h : L > 0) : 
  (∫ x in (0)..(L/2), if x > L/4 then 1 else 0) / (L/2) = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_rope_triangle_probability_l541_54116


namespace NUMINAMATH_CALUDE_clock_time_l541_54175

/-- Represents a clock with a specific ticking pattern -/
structure Clock where
  ticks_at_hour : ℕ
  time_between_first_and_last : ℝ
  time_at_12 : ℝ

/-- The number of ticks at 12 o'clock -/
def ticks_at_12 : ℕ := 12

theorem clock_time (c : Clock) (h1 : c.ticks_at_hour = 6) 
  (h2 : c.time_between_first_and_last = 25) 
  (h3 : c.time_at_12 = 55) : 
  c.ticks_at_hour = 6 := by sorry

end NUMINAMATH_CALUDE_clock_time_l541_54175


namespace NUMINAMATH_CALUDE_coat_duration_proof_l541_54156

/-- The duration (in years) for which the more expensive coat lasts -/
def duration_expensive_coat : ℕ := sorry

/-- The cost of the more expensive coat -/
def cost_expensive_coat : ℕ := 300

/-- The cost of the cheaper coat -/
def cost_cheaper_coat : ℕ := 120

/-- The duration (in years) for which the cheaper coat lasts -/
def duration_cheaper_coat : ℕ := 5

/-- The time period (in years) over which savings are calculated -/
def savings_period : ℕ := 30

/-- The amount saved over the savings period by choosing the more expensive coat -/
def savings_amount : ℕ := 120

theorem coat_duration_proof :
  duration_expensive_coat = 15 ∧
  cost_expensive_coat * savings_period / duration_expensive_coat +
    savings_amount =
  cost_cheaper_coat * savings_period / duration_cheaper_coat :=
by sorry

end NUMINAMATH_CALUDE_coat_duration_proof_l541_54156
