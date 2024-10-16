import Mathlib

namespace NUMINAMATH_CALUDE_art_department_probabilities_l1875_187533

/-- The number of male members in the student art department -/
def num_males : ℕ := 4

/-- The number of female members in the student art department -/
def num_females : ℕ := 3

/-- The total number of members in the student art department -/
def total_members : ℕ := num_males + num_females

/-- The number of members to be selected for the art performance event -/
def num_selected : ℕ := 2

/-- The probability of selecting exactly one female member -/
def prob_one_female : ℚ := 4 / 7

/-- The probability of selecting a specific female member given a specific male member is selected -/
def prob_female_given_male : ℚ := 1 / 6

theorem art_department_probabilities :
  (prob_one_female = 4 / 7) ∧
  (prob_female_given_male = 1 / 6) := by
  sorry

#check art_department_probabilities

end NUMINAMATH_CALUDE_art_department_probabilities_l1875_187533


namespace NUMINAMATH_CALUDE_crackers_per_person_is_76_l1875_187594

/-- The number of crackers each person receives when Darren and Calvin's crackers are shared equally among themselves and 3 friends. -/
def crackers_per_person : ℕ :=
  let darren_type_a_boxes := 4
  let darren_type_b_boxes := 2
  let crackers_per_type_a_box := 24
  let crackers_per_type_b_box := 30
  let calvin_type_a_boxes := 2 * darren_type_a_boxes - 1
  let calvin_type_b_boxes := darren_type_b_boxes
  let total_crackers := 
    (darren_type_a_boxes + calvin_type_a_boxes) * crackers_per_type_a_box +
    (darren_type_b_boxes + calvin_type_b_boxes) * crackers_per_type_b_box
  let number_of_people := 5
  total_crackers / number_of_people

theorem crackers_per_person_is_76 : crackers_per_person = 76 := by
  sorry

end NUMINAMATH_CALUDE_crackers_per_person_is_76_l1875_187594


namespace NUMINAMATH_CALUDE_largest_prime_check_l1875_187528

theorem largest_prime_check (n : ℕ) (h1 : 1000 ≤ n) (h2 : n ≤ 1100) :
  (∀ p : ℕ, p ≤ 31 → Nat.Prime p → ¬(p ∣ n)) → Nat.Prime n :=
sorry

end NUMINAMATH_CALUDE_largest_prime_check_l1875_187528


namespace NUMINAMATH_CALUDE_inequality_system_solution_l1875_187504

theorem inequality_system_solution :
  {x : ℝ | (5 * x + 3 > 3 * (x - 1)) ∧ ((8 * x + 2) / 9 > x)} = {x : ℝ | -3 < x ∧ x < 2} := by
  sorry

end NUMINAMATH_CALUDE_inequality_system_solution_l1875_187504


namespace NUMINAMATH_CALUDE_max_value_cube_roots_l1875_187530

theorem max_value_cube_roots (a b c : ℝ) 
  (ha : 0 ≤ a ∧ a ≤ 2) (hb : 0 ≤ b ∧ b ≤ 2) (hc : 0 ≤ c ∧ c ≤ 2) : 
  (a * b * c) ^ (1/3 : ℝ) + ((2 - a) * (2 - b) * (2 - c)) ^ (1/3 : ℝ) ≤ 2 ∧ 
  ∃ (x : ℝ), 0 ≤ x ∧ x ≤ 2 ∧ 
    (x * x * x) ^ (1/3 : ℝ) + ((2 - x) * (2 - x) * (2 - x)) ^ (1/3 : ℝ) = 2 :=
by sorry

end NUMINAMATH_CALUDE_max_value_cube_roots_l1875_187530


namespace NUMINAMATH_CALUDE_parallelogram_circumference_l1875_187532

/-- The circumference of a parallelogram with side lengths 18 and 12 is 60. -/
theorem parallelogram_circumference : ℝ → ℝ → ℝ → Prop :=
  fun a b c => (a = 18 ∧ b = 12) → c = 2 * (a + b) → c = 60

/-- Proof of the theorem -/
lemma prove_parallelogram_circumference : parallelogram_circumference 18 12 60 := by
  sorry

end NUMINAMATH_CALUDE_parallelogram_circumference_l1875_187532


namespace NUMINAMATH_CALUDE_quadratic_equation_properties_l1875_187564

/-- The quadratic equation x^2 - (2m+1)x + m^2 + m = 0 -/
def quadratic_equation (m x : ℝ) : ℝ := x^2 - (2*m+1)*x + m^2 + m

/-- The discriminant of the quadratic equation -/
def discriminant (m : ℝ) : ℝ := (2*m+1)^2 - 4*(m^2 + m)

/-- The sum of roots of the quadratic equation -/
def sum_of_roots (m : ℝ) : ℝ := 2*m + 1

/-- The product of roots of the quadratic equation -/
def product_of_roots (m : ℝ) : ℝ := m^2 + m

theorem quadratic_equation_properties (m : ℝ) :
  (discriminant m = 1) ∧
  (∃ a b : ℝ, quadratic_equation m a = 0 ∧ quadratic_equation m b = 0 ∧
    (2*a + b) * (a + 2*b) = 20 → (m = -2 ∨ m = 1)) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equation_properties_l1875_187564


namespace NUMINAMATH_CALUDE_cyclist_hiker_catch_up_l1875_187556

/-- Proves that the time the cyclist travels after passing the hiker before stopping
    is equal to the time it takes the hiker to catch up to the cyclist while waiting. -/
theorem cyclist_hiker_catch_up (hiker_speed cyclist_speed : ℝ) (wait_time : ℝ) :
  hiker_speed > 0 →
  cyclist_speed > hiker_speed →
  wait_time > 0 →
  cyclist_speed = 4 * hiker_speed →
  (cyclist_speed / hiker_speed - 1) * wait_time = wait_time :=
by
  sorry

#check cyclist_hiker_catch_up

end NUMINAMATH_CALUDE_cyclist_hiker_catch_up_l1875_187556


namespace NUMINAMATH_CALUDE_problem_1_l1875_187526

theorem problem_1 : (1 : ℝ) - 1^4 - 1/2 * (3 - (-3)^2) = 2 := by sorry

end NUMINAMATH_CALUDE_problem_1_l1875_187526


namespace NUMINAMATH_CALUDE_quadratic_inequality_range_l1875_187596

theorem quadratic_inequality_range (a : ℝ) :
  (∀ x : ℝ, x^2 - a*x + 2*a > 0) ↔ (0 < a ∧ a < 8) :=
sorry

end NUMINAMATH_CALUDE_quadratic_inequality_range_l1875_187596


namespace NUMINAMATH_CALUDE_complementary_angles_theorem_l1875_187503

theorem complementary_angles_theorem (x : ℝ) : 
  (2 * x + 3 * x = 90) → x = 18 := by
  sorry

end NUMINAMATH_CALUDE_complementary_angles_theorem_l1875_187503


namespace NUMINAMATH_CALUDE_expression_evaluation_l1875_187513

theorem expression_evaluation : (1 + 2 + 3) * (1 + 1/2 + 1/3) = 11 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l1875_187513


namespace NUMINAMATH_CALUDE_f_of_one_eq_zero_l1875_187549

theorem f_of_one_eq_zero (f : ℝ → ℝ) (h : ∀ x, f (x + 1) = x^2 - 2*x) : f 1 = 0 := by
  sorry

end NUMINAMATH_CALUDE_f_of_one_eq_zero_l1875_187549


namespace NUMINAMATH_CALUDE_interest_rate_calculation_l1875_187539

/-- Calculates the simple interest rate given principal, amount, and time -/
def calculate_interest_rate (principal amount : ℚ) (time : ℕ) : ℚ :=
  ((amount - principal) * 100) / (principal * time)

/-- Theorem stating that the interest rate is approximately 1.11% -/
theorem interest_rate_calculation (principal amount : ℚ) (time : ℕ) 
  (h_principal : principal = 900)
  (h_amount : amount = 950)
  (h_time : time = 5) :
  abs (calculate_interest_rate principal amount time - 1.11) < 0.01 := by
  sorry

#eval calculate_interest_rate 900 950 5

end NUMINAMATH_CALUDE_interest_rate_calculation_l1875_187539


namespace NUMINAMATH_CALUDE_range_of_m_l1875_187541

theorem range_of_m (m : ℝ) : 
  (∀ x : ℝ, (2*x + 5)/3 - 1 ≤ 2 - x → 3*(x - 1) + 5 > 5*x + 2*(m + x)) → 
  m < -3/5 := by
sorry

end NUMINAMATH_CALUDE_range_of_m_l1875_187541


namespace NUMINAMATH_CALUDE_sin_cos_sum_fifteen_seventyfive_l1875_187510

theorem sin_cos_sum_fifteen_seventyfive : 
  Real.sin (15 * π / 180) * Real.cos (75 * π / 180) + 
  Real.cos (15 * π / 180) * Real.sin (75 * π / 180) = 1 := by
  sorry

end NUMINAMATH_CALUDE_sin_cos_sum_fifteen_seventyfive_l1875_187510


namespace NUMINAMATH_CALUDE_average_after_17th_is_40_l1875_187508

/-- Represents a batsman's performance -/
structure Batsman where
  totalRunsBefore : ℕ  -- Total runs before the 17th inning
  inningsBefore : ℕ    -- Number of innings before the 17th inning (16)
  runsIn17th : ℕ       -- Runs scored in the 17th inning (88)
  averageIncrease : ℕ  -- Increase in average after 17th inning (3)

/-- Calculate the average score after the 17th inning -/
def averageAfter17th (b : Batsman) : ℚ :=
  (b.totalRunsBefore + b.runsIn17th) / (b.inningsBefore + 1)

/-- The main theorem to prove -/
theorem average_after_17th_is_40 (b : Batsman) 
    (h1 : b.inningsBefore = 16)
    (h2 : b.runsIn17th = 88) 
    (h3 : b.averageIncrease = 3)
    (h4 : averageAfter17th b = (b.totalRunsBefore / b.inningsBefore) + b.averageIncrease) :
  averageAfter17th b = 40 := by
  sorry


end NUMINAMATH_CALUDE_average_after_17th_is_40_l1875_187508


namespace NUMINAMATH_CALUDE_bijection_and_size_equivalence_l1875_187548

/-- Represents an integer grid -/
def IntegerGrid := ℤ → ℤ → ℤ

/-- Represents a plane partition -/
def PlanePartition := ℕ → ℕ → ℕ

/-- The size of a plane partition -/
def size (pp : PlanePartition) : ℕ := sorry

/-- The bijection between integer grids and plane partitions -/
def grid_to_partition (g : IntegerGrid) : PlanePartition := sorry

/-- The inverse bijection from plane partitions to integer grids -/
def partition_to_grid (pp : PlanePartition) : IntegerGrid := sorry

/-- The sum of integers in a grid, counting k times for k-th highest diagonal -/
def weighted_sum (g : IntegerGrid) : ℤ := sorry

theorem bijection_and_size_equivalence :
  ∃ (f : IntegerGrid → PlanePartition) (g : PlanePartition → IntegerGrid),
    (∀ grid, g (f grid) = grid) ∧
    (∀ partition, f (g partition) = partition) ∧
    (∀ grid, size (f grid) = weighted_sum grid) := by
  sorry

end NUMINAMATH_CALUDE_bijection_and_size_equivalence_l1875_187548


namespace NUMINAMATH_CALUDE_pirate_treasure_l1875_187515

theorem pirate_treasure (m : ℕ) : 
  (m / 3 + 1) + (m / 4 + 5) + (m / 5 + 20) = m → m = 120 :=
by
  sorry

end NUMINAMATH_CALUDE_pirate_treasure_l1875_187515


namespace NUMINAMATH_CALUDE_gus_total_eggs_l1875_187570

/-- The number of eggs Gus ate for breakfast -/
def breakfast_eggs : ℕ := 2

/-- The number of eggs Gus ate for lunch -/
def lunch_eggs : ℕ := 3

/-- The number of eggs Gus ate for dinner -/
def dinner_eggs : ℕ := 1

/-- The total number of eggs Gus ate -/
def total_eggs : ℕ := breakfast_eggs + lunch_eggs + dinner_eggs

theorem gus_total_eggs : total_eggs = 6 := by
  sorry

end NUMINAMATH_CALUDE_gus_total_eggs_l1875_187570


namespace NUMINAMATH_CALUDE_fraction_evaluation_l1875_187525

theorem fraction_evaluation : (3 : ℚ) / (1 - 2/5) = 5 := by sorry

end NUMINAMATH_CALUDE_fraction_evaluation_l1875_187525


namespace NUMINAMATH_CALUDE_number_of_boys_l1875_187545

theorem number_of_boys (total_students : ℕ) (boys_fraction : ℚ) (boys_count : ℕ) : 
  total_students = 12 →
  boys_fraction = 2/3 →
  boys_count = (total_students : ℚ) * boys_fraction →
  boys_count = 8 := by
sorry

end NUMINAMATH_CALUDE_number_of_boys_l1875_187545


namespace NUMINAMATH_CALUDE_isosceles_obtuse_triangle_smallest_angle_l1875_187550

/-- 
Proves that in an isosceles, obtuse triangle where one angle is 30% larger than a right angle, 
the measure of one of the two smallest angles is 31.5°.
-/
theorem isosceles_obtuse_triangle_smallest_angle : 
  ∀ (a b c : ℝ), 
  a + b + c = 180 →  -- Sum of angles in a triangle is 180°
  a = b →            -- Isosceles triangle condition
  c > 90 →           -- Obtuse triangle condition
  c = 1.3 * 90 →     -- One angle is 30% larger than a right angle
  a = 31.5 :=        -- The measure of one of the two smallest angles
by
  sorry

end NUMINAMATH_CALUDE_isosceles_obtuse_triangle_smallest_angle_l1875_187550


namespace NUMINAMATH_CALUDE_licorice_probability_l1875_187512

def n : ℕ := 7
def k : ℕ := 5
def p : ℚ := 3/5

theorem licorice_probability :
  Nat.choose n k * p^k * (1 - p)^(n - k) = 20412/78125 := by
  sorry

end NUMINAMATH_CALUDE_licorice_probability_l1875_187512


namespace NUMINAMATH_CALUDE_clothing_price_problem_l1875_187551

theorem clothing_price_problem (total_spent : ℕ) (num_pieces : ℕ) (tax_rate : ℚ)
  (untaxed_piece1 : ℕ) (untaxed_piece2 : ℕ) (h_total : total_spent = 610)
  (h_num : num_pieces = 7) (h_tax : tax_rate = 1/10) (h_untaxed1 : untaxed_piece1 = 49)
  (h_untaxed2 : untaxed_piece2 = 81) :
  ∃ (price : ℕ), price * 5 = (total_spent - untaxed_piece1 - untaxed_piece2) * 10 / 11 ∧
  price % 5 = 0 ∧ price = 87 := by
sorry

end NUMINAMATH_CALUDE_clothing_price_problem_l1875_187551


namespace NUMINAMATH_CALUDE_mixed_doubles_selection_count_l1875_187540

theorem mixed_doubles_selection_count :
  let male_count : ℕ := 5
  let female_count : ℕ := 4
  male_count * female_count = 20 :=
by sorry

end NUMINAMATH_CALUDE_mixed_doubles_selection_count_l1875_187540


namespace NUMINAMATH_CALUDE_min_value_expression_l1875_187557

theorem min_value_expression (x : ℝ) :
  (x + 1) * (x + 2) * (x + 3) * (x + 4) + 2021 ≥ 2020 ∧
  ∃ y : ℝ, (y + 1) * (y + 2) * (y + 3) * (y + 4) + 2021 = 2020 :=
by sorry

end NUMINAMATH_CALUDE_min_value_expression_l1875_187557


namespace NUMINAMATH_CALUDE_parabola_coefficient_l1875_187534

/-- Given a parabola y = ax^2 + bx + c with vertex at (h, h) and y-intercept at (0, -2h),
    where h ≠ 0, prove that b = 6 -/
theorem parabola_coefficient (a b c h : ℝ) : 
  h ≠ 0 →
  (∀ x, a * x^2 + b * x + c = a * (x - h)^2 + h) →
  a * h^2 + h = -2 * h →
  b = 6 := by
  sorry

end NUMINAMATH_CALUDE_parabola_coefficient_l1875_187534


namespace NUMINAMATH_CALUDE_real_number_groups_ratio_l1875_187538

theorem real_number_groups_ratio (k : ℝ) (hk : k > 0) : 
  ∃ (group : Set ℝ) (a b c : ℝ), 
    (group ∪ (Set.univ \ group) = Set.univ) ∧ 
    (group ∩ (Set.univ \ group) = ∅) ∧
    (a ∈ group ∧ b ∈ group ∧ c ∈ group) ∧
    (a < b ∧ b < c) ∧
    ((c - b) / (b - a) = k) :=
sorry

end NUMINAMATH_CALUDE_real_number_groups_ratio_l1875_187538


namespace NUMINAMATH_CALUDE_train_crossing_time_l1875_187559

/-- The time taken for a train to cross a man walking in the same direction --/
theorem train_crossing_time (train_length : ℝ) (train_speed : ℝ) (man_speed : ℝ) : 
  train_length = 500 →
  train_speed = 75 * 1000 / 3600 →
  man_speed = 3 * 1000 / 3600 →
  train_length / (train_speed - man_speed) = 25 := by
sorry

end NUMINAMATH_CALUDE_train_crossing_time_l1875_187559


namespace NUMINAMATH_CALUDE_enemies_left_undefeated_l1875_187577

theorem enemies_left_undefeated 
  (total_enemies : ℕ) 
  (points_per_enemy : ℕ) 
  (points_earned : ℕ) : 
  total_enemies = 6 → 
  points_per_enemy = 3 → 
  points_earned = 12 → 
  total_enemies - (points_earned / points_per_enemy) = 2 := by
sorry

end NUMINAMATH_CALUDE_enemies_left_undefeated_l1875_187577


namespace NUMINAMATH_CALUDE_perfect_cube_in_range_l1875_187565

theorem perfect_cube_in_range : 
  ∃! (K : ℤ), 
    K > 1 ∧ 
    ∃ (Z : ℤ), 3000 < Z ∧ Z < 4000 ∧ Z = K^4 ∧ 
    ∃ (n : ℤ), Z = n^3 ∧
    K = 7 := by
  sorry

end NUMINAMATH_CALUDE_perfect_cube_in_range_l1875_187565


namespace NUMINAMATH_CALUDE_det_roots_matrix_l1875_187509

-- Define the polynomial and its roots
def polynomial (m p q : ℝ) (x : ℝ) : ℝ := x^3 - m*x^2 + p*x + q

-- Define the roots a, b, c
def roots (m p q : ℝ) : ℝ × ℝ × ℝ := 
  let (a, b, c) := sorry
  (a, b, c)

-- Define the matrix
def matrix (m p q : ℝ) : Matrix (Fin 3) (Fin 3) ℝ :=
  let (a, b, c) := roots m p q
  !![a, b, c; b, c, a; c, a, b]

-- Theorem statement
theorem det_roots_matrix (m p q : ℝ) :
  let (a, b, c) := roots m p q
  polynomial m p q a = 0 ∧ 
  polynomial m p q b = 0 ∧ 
  polynomial m p q c = 0 →
  Matrix.det (matrix m p q) = -3*q - m^3 + 3*m*p := by sorry

end NUMINAMATH_CALUDE_det_roots_matrix_l1875_187509


namespace NUMINAMATH_CALUDE_min_product_of_three_numbers_l1875_187572

theorem min_product_of_three_numbers (a b c : ℝ) : 
  a > 0 → b > 0 → c > 0 → 
  a + b + c = 2 → 
  a ≤ 3*b ∧ a ≤ 3*c ∧ b ≤ 3*a ∧ b ≤ 3*c ∧ c ≤ 3*a ∧ c ≤ 3*b → 
  2/3 ≤ a * b * c := by
sorry

end NUMINAMATH_CALUDE_min_product_of_three_numbers_l1875_187572


namespace NUMINAMATH_CALUDE_zyx_syndrome_diagnosis_l1875_187582

/-- Represents the characteristics and diagnostic information for ZYX syndrome --/
structure ZYXSyndromeData where
  total_patients : ℕ
  female_ratio : ℚ
  female_syndrome_ratio : ℚ
  male_syndrome_ratio : ℚ
  female_diagnostic_accuracy : ℚ
  male_diagnostic_accuracy : ℚ
  female_false_negative_rate : ℚ
  male_false_negative_rate : ℚ

/-- Calculates the number of patients diagnosed with ZYX syndrome --/
def diagnosed_patients (data : ZYXSyndromeData) : ℕ :=
  sorry

/-- Theorem stating that given the specific conditions, 14 patients will be diagnosed with ZYX syndrome --/
theorem zyx_syndrome_diagnosis :
  let data : ZYXSyndromeData := {
    total_patients := 52,
    female_ratio := 3/5,
    female_syndrome_ratio := 1/5,
    male_syndrome_ratio := 3/10,
    female_diagnostic_accuracy := 7/10,
    male_diagnostic_accuracy := 4/5,
    female_false_negative_rate := 1/10,
    male_false_negative_rate := 3/20
  }
  diagnosed_patients data = 14 := by
  sorry


end NUMINAMATH_CALUDE_zyx_syndrome_diagnosis_l1875_187582


namespace NUMINAMATH_CALUDE_existence_of_special_quadratic_l1875_187500

theorem existence_of_special_quadratic (n : ℕ) (h_odd : Odd n) (h_gt_one : n > 1) :
  ∃ (a b : ℕ), a > 0 ∧ b > 0 ∧
    (Nat.gcd a n = 1) ∧
    (Nat.gcd b n = 1) ∧
    (n ∣ (a^2 + b)) ∧
    (∀ x : ℕ, x ≥ 1 → ∃ p : ℕ, Prime p ∧ p ∣ ((x + a)^2 + b) ∧ ¬(p ∣ n)) := by
  sorry

end NUMINAMATH_CALUDE_existence_of_special_quadratic_l1875_187500


namespace NUMINAMATH_CALUDE_record_storage_cost_l1875_187505

-- Define the box dimensions
def box_length : ℝ := 15
def box_width : ℝ := 12
def box_height : ℝ := 10

-- Define the total occupied space in cubic inches
def total_space : ℝ := 1080000

-- Define the storage cost per box per month
def cost_per_box : ℝ := 0.5

-- Theorem to prove
theorem record_storage_cost :
  let box_volume : ℝ := box_length * box_width * box_height
  let num_boxes : ℝ := total_space / box_volume
  let total_cost : ℝ := num_boxes * cost_per_box
  total_cost = 300 := by
sorry


end NUMINAMATH_CALUDE_record_storage_cost_l1875_187505


namespace NUMINAMATH_CALUDE_power_of_two_starts_with_1968_l1875_187579

-- Define the conditions
def m : ℕ := 3^2
def k : ℕ := 2^3

-- Define a function to check if a number starts with 1968
def starts_with_1968 (x : ℕ) : Prop :=
  ∃ y : ℕ, 1968 * 10^y ≤ x ∧ x < 1969 * 10^y

-- State the theorem
theorem power_of_two_starts_with_1968 :
  ∃ n : ℕ, n > 2^k ∧ starts_with_1968 (2^n) ∧
  ∀ m : ℕ, m < n → ¬starts_with_1968 (2^m) :=
sorry

end NUMINAMATH_CALUDE_power_of_two_starts_with_1968_l1875_187579


namespace NUMINAMATH_CALUDE_triangle_angle_inequality_l1875_187501

theorem triangle_angle_inequality (A B C : ℝ) (h_triangle : A + B + C = π) (h_positive : A > 0 ∧ B > 0 ∧ C > 0) :
  2 * (Real.sin A / A + Real.sin B / B + Real.sin C / C) ≤
  (1/B + 1/C) * Real.sin A + (1/C + 1/A) * Real.sin B + (1/A + 1/B) * Real.sin C :=
by sorry

end NUMINAMATH_CALUDE_triangle_angle_inequality_l1875_187501


namespace NUMINAMATH_CALUDE_trig_identity_l1875_187563

theorem trig_identity : 
  Real.sin (71 * π / 180) * Real.cos (26 * π / 180) - 
  Real.sin (19 * π / 180) * Real.sin (26 * π / 180) = 
  Real.sqrt 2 / 2 := by sorry

end NUMINAMATH_CALUDE_trig_identity_l1875_187563


namespace NUMINAMATH_CALUDE_quilt_shaded_area_is_40_percent_l1875_187554

/-- Represents a square quilt with shaded areas -/
structure Quilt where
  size : Nat
  fully_shaded : Nat
  half_shaded_single : Nat
  half_shaded_double : Nat

/-- Calculates the percentage of shaded area in the quilt -/
def shaded_percentage (q : Quilt) : Rat :=
  let total_squares := q.size * q.size
  let shaded_area := q.fully_shaded + (q.half_shaded_single / 2) + (q.half_shaded_double / 2)
  (shaded_area / total_squares) * 100

/-- Theorem stating that the specific quilt configuration has 40% shaded area -/
theorem quilt_shaded_area_is_40_percent :
  let q : Quilt := {
    size := 5,
    fully_shaded := 4,
    half_shaded_single := 8,
    half_shaded_double := 4
  }
  shaded_percentage q = 40 := by sorry

end NUMINAMATH_CALUDE_quilt_shaded_area_is_40_percent_l1875_187554


namespace NUMINAMATH_CALUDE_ratio_sum_to_base_l1875_187578

theorem ratio_sum_to_base (x y : ℚ) (h : y ≠ 0) (h1 : x / y = 2 / 3) :
  (x + y) / y = 5 / 3 := by
  sorry

end NUMINAMATH_CALUDE_ratio_sum_to_base_l1875_187578


namespace NUMINAMATH_CALUDE_total_scissors_l1875_187595

/-- The total number of scissors after adding more is equal to the sum of the initial number of scissors and the number of scissors added. -/
theorem total_scissors (initial : ℕ) (added : ℕ) : 
  initial + added = initial + added :=
by sorry

end NUMINAMATH_CALUDE_total_scissors_l1875_187595


namespace NUMINAMATH_CALUDE_chess_tournament_games_l1875_187589

/-- The number of games played in a chess tournament. -/
def num_games (n : ℕ) (games_per_pair : ℕ) : ℕ :=
  n * (n - 1) / 2 * games_per_pair

/-- Theorem: In a chess tournament with 25 players, where every player plays
    three times with each of their opponents, the total number of games is 900. -/
theorem chess_tournament_games :
  num_games 25 3 = 900 := by
  sorry

end NUMINAMATH_CALUDE_chess_tournament_games_l1875_187589


namespace NUMINAMATH_CALUDE_reciprocal_product_theorem_l1875_187547

theorem reciprocal_product_theorem (a b : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) 
  (h : a + b = 3 * a * b) : 
  (1 / a) * (1 / b) = 9 / 4 := by
sorry

end NUMINAMATH_CALUDE_reciprocal_product_theorem_l1875_187547


namespace NUMINAMATH_CALUDE_jenny_activities_alignment_l1875_187546

def dance_interval : ℕ := 6
def karate_interval : ℕ := 12
def swimming_interval : ℕ := 15
def painting_interval : ℕ := 20
def library_interval : ℕ := 18
def sick_days : ℕ := 7

def next_alignment_day : ℕ := 187

theorem jenny_activities_alignment :
  let intervals := [dance_interval, karate_interval, swimming_interval, painting_interval, library_interval]
  let lcm_intervals := Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm dance_interval karate_interval) swimming_interval) painting_interval) library_interval
  next_alignment_day = lcm_intervals + sick_days := by
  sorry

end NUMINAMATH_CALUDE_jenny_activities_alignment_l1875_187546


namespace NUMINAMATH_CALUDE_rotate_A_180_l1875_187599

def rotate_180_origin (p : ℝ × ℝ) : ℝ × ℝ :=
  (-p.1, -p.2)

theorem rotate_A_180 :
  let A : ℝ × ℝ := (-3, 2)
  rotate_180_origin A = (3, -2) := by
  sorry

end NUMINAMATH_CALUDE_rotate_A_180_l1875_187599


namespace NUMINAMATH_CALUDE_sqrt_equation_solutions_l1875_187542

theorem sqrt_equation_solutions (x : ℝ) : 
  (Real.sqrt (9 * x - 4) + 18 / Real.sqrt (9 * x - 4) = 10) ↔ (x = 85 / 9 ∨ x = 8 / 9) :=
by sorry

end NUMINAMATH_CALUDE_sqrt_equation_solutions_l1875_187542


namespace NUMINAMATH_CALUDE_divisibility_in_base_greater_than_six_l1875_187511

theorem divisibility_in_base_greater_than_six (a : ℕ) (h : a > 6) :
  ∃ k : ℕ, a^10 + 2*a^9 + 3*a^8 + 4*a^7 + 5*a^6 + 6*a^5 + 5*a^4 + 4*a^3 + 3*a^2 + 2*a + 1
         = k * (a^4 + 2*a^3 + 3*a^2 + 2*a + 1) :=
by sorry

end NUMINAMATH_CALUDE_divisibility_in_base_greater_than_six_l1875_187511


namespace NUMINAMATH_CALUDE_stationery_cost_l1875_187575

/-- The total cost of a pen, pencil, and eraser with given price relationships -/
theorem stationery_cost (pencil_cost : ℚ) : 
  pencil_cost = 8 →
  (pencil_cost + (1/2 * pencil_cost) + (2 * (1/2 * pencil_cost))) = 20 := by
  sorry

end NUMINAMATH_CALUDE_stationery_cost_l1875_187575


namespace NUMINAMATH_CALUDE_conclusion_one_conclusion_three_l1875_187527

-- Define the custom operation
def custom_op (a b : ℝ) : ℝ := a * (1 - b)

-- Theorem for the first correct conclusion
theorem conclusion_one : custom_op 2 (-2) = 6 := by sorry

-- Theorem for the third correct conclusion
theorem conclusion_three (a b : ℝ) (h : a + b = 0) :
  custom_op a a + custom_op b b = 2 * a * b := by sorry

end NUMINAMATH_CALUDE_conclusion_one_conclusion_three_l1875_187527


namespace NUMINAMATH_CALUDE_train_passing_bridge_l1875_187531

/-- Time taken for a train to pass a bridge -/
theorem train_passing_bridge
  (train_length : ℝ)
  (train_speed_kmh : ℝ)
  (bridge_length : ℝ)
  (h1 : train_length = 870)
  (h2 : train_speed_kmh = 90)
  (h3 : bridge_length = 370) :
  (train_length + bridge_length) / (train_speed_kmh * 1000 / 3600) = 49.6 :=
by sorry

end NUMINAMATH_CALUDE_train_passing_bridge_l1875_187531


namespace NUMINAMATH_CALUDE_quadratic_equation_properties_l1875_187583

/-- The discriminant of the quadratic equation x² - (m + 3)x + m + 1 = 0 -/
def discriminant (m : ℝ) : ℝ := (m + 1)^2 + 4

theorem quadratic_equation_properties :
  (∀ m : ℝ, discriminant m > 0) ∧
  ({m : ℝ | discriminant m = 5} = {0, -2}) := by
  sorry

#check quadratic_equation_properties

end NUMINAMATH_CALUDE_quadratic_equation_properties_l1875_187583


namespace NUMINAMATH_CALUDE_megacorp_fine_l1875_187544

/-- MegaCorp's fine calculation --/
theorem megacorp_fine :
  let daily_mining_profit : ℕ := 3000000
  let daily_oil_profit : ℕ := 5000000
  let monthly_expenses : ℕ := 30000000
  let days_per_year : ℕ := 365
  let months_per_year : ℕ := 12
  let fine_percentage : ℚ := 1 / 100

  let annual_revenue : ℕ := (daily_mining_profit + daily_oil_profit) * days_per_year
  let annual_expenses : ℕ := monthly_expenses * months_per_year
  let annual_profit : ℕ := annual_revenue - annual_expenses
  let fine : ℚ := (annual_profit : ℚ) * fine_percentage

  fine = 25600000 := by sorry

end NUMINAMATH_CALUDE_megacorp_fine_l1875_187544


namespace NUMINAMATH_CALUDE_inequality_for_negative_numbers_l1875_187516

theorem inequality_for_negative_numbers (a b : ℝ) (h : a < b ∧ b < 0) :
  a^2 > a*b ∧ a*b > b^2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_for_negative_numbers_l1875_187516


namespace NUMINAMATH_CALUDE_lily_siblings_count_l1875_187553

/-- The number of suitcases each sibling brings -/
def suitcases_per_sibling : ℕ := 2

/-- The number of suitcases parents bring -/
def suitcases_parents : ℕ := 6

/-- The total number of suitcases the family brings -/
def total_suitcases : ℕ := 14

/-- The number of Lily's siblings -/
def num_siblings : ℕ := (total_suitcases - suitcases_parents) / suitcases_per_sibling

theorem lily_siblings_count : num_siblings = 4 := by
  sorry

end NUMINAMATH_CALUDE_lily_siblings_count_l1875_187553


namespace NUMINAMATH_CALUDE_stratified_sampling_grade12_l1875_187597

theorem stratified_sampling_grade12 (total_students : ℕ) (grade12_students : ℕ) (sample_size : ℕ) 
  (h1 : total_students = 3600) 
  (h2 : grade12_students = 1500) 
  (h3 : sample_size = 720) :
  (grade12_students : ℚ) / (total_students : ℚ) * (sample_size : ℚ) = 300 := by
  sorry

end NUMINAMATH_CALUDE_stratified_sampling_grade12_l1875_187597


namespace NUMINAMATH_CALUDE_smallest_sum_is_11_l1875_187568

/-- B is a digit in base 4 -/
def is_base_4_digit (B : ℕ) : Prop := B < 4

/-- b is a base greater than 5 -/
def is_base_greater_than_5 (b : ℕ) : Prop := b > 5

/-- BBB₄ = 44ᵦ -/
def equality_condition (B b : ℕ) : Prop := 21 * B = 4 * (b + 1)

/-- The smallest possible sum of B and b is 11 -/
theorem smallest_sum_is_11 :
  ∃ (B b : ℕ), is_base_4_digit B ∧ is_base_greater_than_5 b ∧ equality_condition B b ∧
  B + b = 11 ∧
  ∀ (B' b' : ℕ), is_base_4_digit B' → is_base_greater_than_5 b' → equality_condition B' b' →
  B' + b' ≥ 11 :=
sorry

end NUMINAMATH_CALUDE_smallest_sum_is_11_l1875_187568


namespace NUMINAMATH_CALUDE_arithmetic_calculation_l1875_187569

theorem arithmetic_calculation : 5 * 7 + 9 * 4 - 30 / 3 + (5 - 3)^2 = 65 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_calculation_l1875_187569


namespace NUMINAMATH_CALUDE_pizza_toppings_combinations_l1875_187562

theorem pizza_toppings_combinations (n : ℕ) (h : n = 8) : 
  n + (n.choose 2) + (n.choose 3) = 92 := by
  sorry

end NUMINAMATH_CALUDE_pizza_toppings_combinations_l1875_187562


namespace NUMINAMATH_CALUDE_cauchy_schwarz_inequality_l1875_187566

theorem cauchy_schwarz_inequality (a b x y : ℝ) : (a^2 + b^2) * (x^2 + y^2) ≥ (a*x + b*y)^2 := by
  sorry

end NUMINAMATH_CALUDE_cauchy_schwarz_inequality_l1875_187566


namespace NUMINAMATH_CALUDE_decimal_number_problem_l1875_187580

theorem decimal_number_problem :
  ∃ x : ℝ, 
    0 ≤ x ∧ 
    x < 10 ∧ 
    (∃ n : ℤ, ⌊x⌋ = n) ∧
    ⌊x⌋ + 4 * x = 21.2 ∧
    x = 4.3 := by
  sorry

end NUMINAMATH_CALUDE_decimal_number_problem_l1875_187580


namespace NUMINAMATH_CALUDE_integral_inequality_l1875_187543

-- Define a non-decreasing function on [0,∞)
def NonDecreasing (f : ℝ → ℝ) : Prop :=
  ∀ x y, 0 ≤ x → x ≤ y → f x ≤ f y

-- State the theorem
theorem integral_inequality
  (f : ℝ → ℝ)
  (h_nondec : NonDecreasing f)
  {x y z : ℝ}
  (h_x : 0 ≤ x)
  (h_xy : x < y)
  (h_yz : y < z) :
  (z - x) * ∫ u in y..z, f u ≥ (z - y) * ∫ u in x..z, f u :=
sorry

end NUMINAMATH_CALUDE_integral_inequality_l1875_187543


namespace NUMINAMATH_CALUDE_x_equals_five_l1875_187506

theorem x_equals_five (x y : ℝ) (h_pos_x : x > 0) (h_pos_y : y > 0) 
  (h_eq : 5 * x^2 + 15 * x * y = x^3 + 2 * x^2 * y + 3 * x * y^2) : x = 5 := by
  sorry

end NUMINAMATH_CALUDE_x_equals_five_l1875_187506


namespace NUMINAMATH_CALUDE_equilateral_triangle_condition_l1875_187523

/-- A function that checks if a natural number n satisfies the conditions for forming an equilateral triangle with sticks of lengths 1 to n. -/
def can_form_equilateral_triangle (n : ℕ) : Prop :=
  n ≥ 5 ∧ (n % 6 = 0 ∨ n % 6 = 2 ∨ n % 6 = 3 ∨ n % 6 = 5)

/-- The sum of the first n natural numbers. -/
def sum_of_first_n (n : ℕ) : ℕ :=
  n * (n + 1) / 2

/-- Theorem stating the necessary and sufficient conditions for forming an equilateral triangle with sticks of lengths 1 to n. -/
theorem equilateral_triangle_condition (n : ℕ) :
  (sum_of_first_n n % 3 = 0) ↔ can_form_equilateral_triangle n :=
sorry

end NUMINAMATH_CALUDE_equilateral_triangle_condition_l1875_187523


namespace NUMINAMATH_CALUDE_insect_legs_l1875_187536

theorem insect_legs (num_insects : ℕ) (total_legs : ℕ) (h1 : num_insects = 8) (h2 : total_legs = 48) :
  total_legs / num_insects = 6 := by
  sorry

end NUMINAMATH_CALUDE_insect_legs_l1875_187536


namespace NUMINAMATH_CALUDE_salary_problem_l1875_187535

theorem salary_problem (a b : ℝ) 
  (h1 : a + b = 3000)
  (h2 : a * 0.05 = b * 0.15) : 
  a = 2250 := by
sorry

end NUMINAMATH_CALUDE_salary_problem_l1875_187535


namespace NUMINAMATH_CALUDE_tangent_slope_at_point_A_l1875_187552

-- Define the curve
def f (x : ℝ) : ℝ := 2 * x^2

-- State the theorem
theorem tangent_slope_at_point_A :
  let x₀ : ℝ := 2
  let y₀ : ℝ := f x₀
  (y₀ = 8) →  -- This ensures the point (2,8) is on the curve
  (deriv f x₀ = 8) :=
by sorry

end NUMINAMATH_CALUDE_tangent_slope_at_point_A_l1875_187552


namespace NUMINAMATH_CALUDE_angle_A_measure_l1875_187519

-- Define the triangle and its properties
structure Triangle :=
  (A B C : ℝ)
  (sum_of_angles : A + B + C = 180)

-- Define the configuration
def geometric_configuration (t : Triangle) (x y : ℝ) : Prop :=
  t.B = 120 ∧ 
  x = 50 ∧
  y = 130 ∧
  x + (180 - y) + t.C = 180

-- Theorem statement
theorem angle_A_measure (t : Triangle) (x y : ℝ) 
  (h : geometric_configuration t x y) : t.A = 120 :=
sorry

end NUMINAMATH_CALUDE_angle_A_measure_l1875_187519


namespace NUMINAMATH_CALUDE_no_negative_roots_l1875_187590

theorem no_negative_roots : ∀ x : ℝ, x < 0 → 4 * x^4 - 7 * x^3 - 20 * x^2 - 13 * x + 25 ≠ 0 := by
  sorry

end NUMINAMATH_CALUDE_no_negative_roots_l1875_187590


namespace NUMINAMATH_CALUDE_battle_station_staffing_l1875_187560

theorem battle_station_staffing (n : ℕ) (k : ℕ) (h1 : n = 15) (h2 : k = 5) :
  n * (n - 1) * (n - 2) * (n - 3) * (n - 4) = 360360 :=
by sorry

end NUMINAMATH_CALUDE_battle_station_staffing_l1875_187560


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l1875_187524

theorem arithmetic_sequence_sum : 
  ∀ (a l : ℤ) (d : ℤ) (n : ℕ),
    a = 162 →
    d = -6 →
    l = 48 →
    n > 0 →
    l = a + (n - 1) * d →
    (n : ℤ) * (a + l) / 2 = 2100 :=
by sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l1875_187524


namespace NUMINAMATH_CALUDE_solve_average_weight_l1875_187518

def average_weight_problem (num_boys num_girls : ℕ) (avg_weight_boys avg_weight_girls : ℚ) : Prop :=
  let total_children := num_boys + num_girls
  let total_weight := (num_boys : ℚ) * avg_weight_boys + (num_girls : ℚ) * avg_weight_girls
  let avg_weight_all := total_weight / total_children
  (↑(round avg_weight_all) : ℚ) = 141

theorem solve_average_weight :
  average_weight_problem 8 5 160 110 := by
  sorry

end NUMINAMATH_CALUDE_solve_average_weight_l1875_187518


namespace NUMINAMATH_CALUDE_faster_train_speed_calculation_l1875_187522

/-- The speed of the slower train in km/h -/
def slower_train_speed : ℝ := 32

/-- The time it takes for the faster train to pass in seconds -/
def passing_time : ℝ := 15

/-- The length of the faster train in meters -/
def faster_train_length : ℝ := 75.006

/-- The speed of the faster train in km/h -/
def faster_train_speed : ℝ := 50.00144

theorem faster_train_speed_calculation :
  let relative_speed := (faster_train_speed - slower_train_speed) * (5 / 18)
  relative_speed = faster_train_length / passing_time :=
by sorry

#check faster_train_speed_calculation

end NUMINAMATH_CALUDE_faster_train_speed_calculation_l1875_187522


namespace NUMINAMATH_CALUDE_special_trapezoid_ratio_l1875_187555

/-- Isosceles trapezoid with specific properties -/
structure SpecialTrapezoid where
  /-- Length of the shorter base -/
  a : ℝ
  /-- Length of the longer base -/
  long_base : ℝ
  /-- Length of the altitude -/
  altitude : ℝ
  /-- Length of a diagonal -/
  diagonal : ℝ
  /-- Longer base is square of shorter base -/
  long_base_eq : long_base = a^2
  /-- Shorter base equals altitude -/
  altitude_eq : altitude = a
  /-- Diagonal equals radius of circumscribed circle -/
  diagonal_eq : diagonal = 2

/-- The ratio of shorter base to longer base in the special trapezoid is 3/16 -/
theorem special_trapezoid_ratio (t : SpecialTrapezoid) : t.a / t.long_base = 3/16 := by
  sorry

end NUMINAMATH_CALUDE_special_trapezoid_ratio_l1875_187555


namespace NUMINAMATH_CALUDE_prism_faces_count_l1875_187537

/-- Represents a polygonal prism -/
structure Prism where
  base_sides : ℕ
  edges : ℕ := 3 * base_sides
  faces : ℕ := 2 + base_sides

/-- Represents a polygonal pyramid -/
structure Pyramid where
  base_sides : ℕ
  edges : ℕ := 2 * base_sides

/-- Theorem stating that a prism has 8 faces given the conditions -/
theorem prism_faces_count (p : Prism) (py : Pyramid) 
  (h1 : p.base_sides = py.base_sides) 
  (h2 : p.edges + py.edges = 30) : p.faces = 8 := by
  sorry

end NUMINAMATH_CALUDE_prism_faces_count_l1875_187537


namespace NUMINAMATH_CALUDE_lemonade_pitchers_sum_l1875_187593

theorem lemonade_pitchers_sum : 
  let first_intermission : ℝ := 0.25
  let second_intermission : ℝ := 0.42
  let third_intermission : ℝ := 0.25
  first_intermission + second_intermission + third_intermission = 0.92 := by
sorry

end NUMINAMATH_CALUDE_lemonade_pitchers_sum_l1875_187593


namespace NUMINAMATH_CALUDE_meaningful_expression_range_l1875_187514

theorem meaningful_expression_range (x : ℝ) :
  (∃ y : ℝ, y = (1 : ℝ) / Real.sqrt (x - 2)) ↔ x > 2 := by
  sorry

end NUMINAMATH_CALUDE_meaningful_expression_range_l1875_187514


namespace NUMINAMATH_CALUDE_karl_drove_420_miles_l1875_187521

/-- Represents Karl's car and trip details -/
structure KarlsTrip where
  miles_per_gallon : ℝ
  tank_capacity : ℝ
  initial_distance : ℝ
  gas_bought : ℝ
  final_tank_fraction : ℝ

/-- Calculates the total distance Karl drove given the trip details -/
def total_distance (trip : KarlsTrip) : ℝ :=
  sorry

/-- Theorem stating that Karl drove 420 miles given the specific conditions -/
theorem karl_drove_420_miles :
  let trip : KarlsTrip := {
    miles_per_gallon := 30,
    tank_capacity := 16,
    initial_distance := 360,
    gas_bought := 10,
    final_tank_fraction := 3/4
  }
  total_distance trip = 420 := by
  sorry

end NUMINAMATH_CALUDE_karl_drove_420_miles_l1875_187521


namespace NUMINAMATH_CALUDE_circle_equation_equivalence_l1875_187591

theorem circle_equation_equivalence :
  ∀ x y : ℝ, x^2 - 6*x + y^2 - 10*y + 18 = 0 ↔ (x-3)^2 + (y-5)^2 = 16 := by
  sorry

end NUMINAMATH_CALUDE_circle_equation_equivalence_l1875_187591


namespace NUMINAMATH_CALUDE_hemisphere_on_cone_surface_area_l1875_187561

/-- The total surface area of a solid figure formed by placing a hemisphere on top of a cone -/
theorem hemisphere_on_cone_surface_area
  (hemisphere_radius : ℝ)
  (cone_base_radius : ℝ)
  (cone_slant_height : ℝ)
  (hemisphere_radius_eq : hemisphere_radius = 5)
  (cone_base_radius_eq : cone_base_radius = 7)
  (cone_slant_height_eq : cone_slant_height = 14) :
  2 * π * hemisphere_radius^2 + π * hemisphere_radius^2 + π * cone_base_radius * cone_slant_height = 173 * π :=
by sorry

end NUMINAMATH_CALUDE_hemisphere_on_cone_surface_area_l1875_187561


namespace NUMINAMATH_CALUDE_binomial_p_value_l1875_187502

/-- A random variable following a binomial distribution -/
structure BinomialRV where
  n : ℕ
  p : ℝ
  mean : ℝ
  variance : ℝ
  mean_eq : mean = n * p
  variance_eq : variance = n * p * (1 - p)

/-- Theorem stating the value of p for a binomial random variable with given mean and variance -/
theorem binomial_p_value (ξ : BinomialRV) 
  (h_mean : ξ.mean = 300)
  (h_var : ξ.variance = 200) :
  ξ.p = 1/3 := by
  sorry

#check binomial_p_value

end NUMINAMATH_CALUDE_binomial_p_value_l1875_187502


namespace NUMINAMATH_CALUDE_binomial_square_constant_l1875_187517

theorem binomial_square_constant (c : ℝ) : 
  (∃ a b : ℝ, ∀ x : ℝ, 9 * x^2 - 24 * x + c = (a * x + b)^2) → c = 16 := by
  sorry

end NUMINAMATH_CALUDE_binomial_square_constant_l1875_187517


namespace NUMINAMATH_CALUDE_calculation_problem_linear_system_solution_l1875_187581

-- Problem 1
theorem calculation_problem : -Real.sqrt 3 + (-5/2)^0 + |1 - Real.sqrt 3| = 0 := by sorry

-- Problem 2
theorem linear_system_solution :
  ∃ (x y : ℝ), 4*x + 3*y = 10 ∧ 3*x + y = 5 ∧ x = 1 ∧ y = 2 := by sorry

end NUMINAMATH_CALUDE_calculation_problem_linear_system_solution_l1875_187581


namespace NUMINAMATH_CALUDE_ab_value_l1875_187529

theorem ab_value (a b : ℝ) (h : |3*a - 1| + b^2 = 0) : a^b = 1 := by
  sorry

end NUMINAMATH_CALUDE_ab_value_l1875_187529


namespace NUMINAMATH_CALUDE_bug_final_position_l1875_187584

def CirclePoints : Nat := 7

def jump (start : Nat) : Nat :=
  if start % 2 == 0 then
    (start + 2 - 1) % CirclePoints + 1
  else
    (start + 3 - 1) % CirclePoints + 1

def bug_position (start : Nat) (jumps : Nat) : Nat :=
  match jumps with
  | 0 => start
  | n + 1 => jump (bug_position start n)

theorem bug_final_position :
  bug_position 7 2023 = 1 := by sorry

end NUMINAMATH_CALUDE_bug_final_position_l1875_187584


namespace NUMINAMATH_CALUDE_ellipse_foci_distance_sum_l1875_187567

/-- An ellipse with semi-major axis 5 and semi-minor axis 4 -/
def Ellipse : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | (p.1^2 / 25) + (p.2^2 / 16) = 1}

/-- The foci of the ellipse -/
def Foci : (ℝ × ℝ) × (ℝ × ℝ) := sorry

/-- The distance between two points in ℝ² -/
def distance (p q : ℝ × ℝ) : ℝ := sorry

/-- Theorem: For any point on the ellipse, the sum of distances to the foci is 10 -/
theorem ellipse_foci_distance_sum (p : ℝ × ℝ) (h : p ∈ Ellipse) :
  distance p Foci.1 + distance p Foci.2 = 10 := by sorry

end NUMINAMATH_CALUDE_ellipse_foci_distance_sum_l1875_187567


namespace NUMINAMATH_CALUDE_line_through_points_l1875_187585

/-- Theorem: For a line y = ax + b passing through points (3, 4) and (10, 22), a - b = 6 2/7 -/
theorem line_through_points (a b : ℚ) : 
  (4 : ℚ) = a * 3 + b ∧ (22 : ℚ) = a * 10 + b → a - b = (44 : ℚ) / 7 := by
  sorry

end NUMINAMATH_CALUDE_line_through_points_l1875_187585


namespace NUMINAMATH_CALUDE_revenue_change_specific_l1875_187588

/-- Calculates the change in revenue given price increase, sales decrease, loyalty discount, and sales tax -/
def revenueChange (priceIncrease salesDecrease loyaltyDiscount salesTax : ℝ) : ℝ :=
  let newPrice := 1 + priceIncrease
  let newSales := 1 - salesDecrease
  let discountedPrice := newPrice * (1 - loyaltyDiscount)
  let finalPrice := discountedPrice * (1 + salesTax)
  finalPrice * newSales - 1

/-- The revenue change given specific conditions -/
theorem revenue_change_specific : 
  revenueChange 0.9 0.3 0.1 0.15 = 0.37655 := by sorry

end NUMINAMATH_CALUDE_revenue_change_specific_l1875_187588


namespace NUMINAMATH_CALUDE_prob_all_suits_in_five_draws_l1875_187558

/-- Represents a standard 52-card deck -/
def StandardDeck : ℕ := 52

/-- Number of suits in a standard deck -/
def NumberOfSuits : ℕ := 4

/-- Number of cards drawn -/
def NumberOfDraws : ℕ := 5

/-- Probability of drawing a card from a specific suit -/
def ProbSingleSuit : ℚ := 1 / NumberOfSuits

/-- Theorem: Probability of getting at least one card from each suit in 5 draws with replacement -/
theorem prob_all_suits_in_five_draws : 
  let prob_different_suit (n : ℕ) := (NumberOfSuits - n) / NumberOfSuits
  (prob_different_suit 1) * (prob_different_suit 2) * (prob_different_suit 3) = 3 / 32 := by
  sorry

end NUMINAMATH_CALUDE_prob_all_suits_in_five_draws_l1875_187558


namespace NUMINAMATH_CALUDE_triangle_BC_length_l1875_187573

/-- Triangle ABC with given properties --/
structure TriangleABC where
  AB : ℝ
  AC : ℝ
  BC : ℝ
  BX : ℕ
  CX : ℕ
  h_AB : AB = 75
  h_AC : AC = 85
  h_BC : BC = BX + CX
  h_circle : BX^2 + CX^2 = AB^2

/-- Theorem: BC = 89 in the given triangle --/
theorem triangle_BC_length (t : TriangleABC) : t.BC = 89 := by
  sorry

end NUMINAMATH_CALUDE_triangle_BC_length_l1875_187573


namespace NUMINAMATH_CALUDE_fraction_inequality_solution_l1875_187592

theorem fraction_inequality_solution (x : ℝ) : 
  x ≠ 3 → (x * (x + 1) / (x - 3)^2 ≥ 9 ↔ 
    (2.13696 ≤ x ∧ x < 3) ∨ (3 < x ∧ x ≤ 4.73804)) :=
by sorry

end NUMINAMATH_CALUDE_fraction_inequality_solution_l1875_187592


namespace NUMINAMATH_CALUDE_apple_cost_price_l1875_187576

/-- 
Given:
- The selling price of an apple is 18
- The seller loses 1/6th of the cost price
Prove that the cost price is 21.6
-/
theorem apple_cost_price (selling_price : ℝ) (loss_fraction : ℝ) (cost_price : ℝ) : 
  selling_price = 18 → 
  loss_fraction = 1/6 → 
  selling_price = cost_price * (1 - loss_fraction) →
  cost_price = 21.6 := by
sorry

end NUMINAMATH_CALUDE_apple_cost_price_l1875_187576


namespace NUMINAMATH_CALUDE_sticker_exchange_result_l1875_187520

def sticker_exchange (ryan : ℕ) : ℕ :=
  let steven := 3 * ryan
  let terry := steven + 20
  let emily := steven / 2
  let jasmine := terry + terry / 10
  let total_before := ryan + steven + terry + emily + jasmine
  let exchange_diff := 5 - 3
  total_before - 5 * exchange_diff

theorem sticker_exchange_result :
  sticker_exchange 30 = 386 := by
  sorry

end NUMINAMATH_CALUDE_sticker_exchange_result_l1875_187520


namespace NUMINAMATH_CALUDE_potato_chips_count_l1875_187507

/-- The number of potato chips one potato can make -/
def potato_chips_per_potato (total_potatoes wedge_potatoes wedges_per_potato : ℕ) 
  (chip_wedge_difference : ℕ) : ℕ :=
let remaining_potatoes := total_potatoes - wedge_potatoes
let chip_potatoes := remaining_potatoes / 2
let total_wedges := wedge_potatoes * wedges_per_potato
let total_chips := total_wedges + chip_wedge_difference
total_chips / chip_potatoes

/-- Theorem stating that one potato can make 20 potato chips under given conditions -/
theorem potato_chips_count : 
  potato_chips_per_potato 67 13 8 436 = 20 := by
  sorry

end NUMINAMATH_CALUDE_potato_chips_count_l1875_187507


namespace NUMINAMATH_CALUDE_complex_number_quadrant_l1875_187574

theorem complex_number_quadrant (z : ℂ) (h : (2 - Complex.I) * z = Complex.I) :
  Real.sign (z.re) = -1 ∧ Real.sign (z.im) = 1 := by
  sorry

end NUMINAMATH_CALUDE_complex_number_quadrant_l1875_187574


namespace NUMINAMATH_CALUDE_shirt_price_l1875_187586

theorem shirt_price (total_items : Nat) (dress_count : Nat) (shirt_count : Nat)
  (total_money : ℕ) (dress_price : ℕ) :
  total_items = dress_count + shirt_count →
  total_money = dress_count * dress_price + shirt_count * (total_money - dress_count * dress_price) / shirt_count →
  dress_count = 7 →
  shirt_count = 4 →
  total_money = 69 →
  dress_price = 7 →
  (total_money - dress_count * dress_price) / shirt_count = 5 := by
sorry

end NUMINAMATH_CALUDE_shirt_price_l1875_187586


namespace NUMINAMATH_CALUDE_opposite_of_sqrt_3_l1875_187571

-- Define the concept of opposite for real numbers
def opposite (x : ℝ) : ℝ := -x

-- State the theorem
theorem opposite_of_sqrt_3 : opposite (Real.sqrt 3) = -(Real.sqrt 3) := by
  sorry

end NUMINAMATH_CALUDE_opposite_of_sqrt_3_l1875_187571


namespace NUMINAMATH_CALUDE_slower_pump_fill_time_l1875_187587

/-- Represents a water pump with a constant fill rate -/
structure Pump where
  rate : ℝ
  rate_positive : rate > 0

/-- Represents a swimming pool -/
structure Pool where
  volume : ℝ
  volume_positive : volume > 0

/-- Theorem stating the time taken by the slower pump to fill the pool alone -/
theorem slower_pump_fill_time (pool : Pool) (pump1 pump2 : Pump)
    (h1 : pump2.rate = 1.5 * pump1.rate)
    (h2 : (pump1.rate + pump2.rate) * 5 = pool.volume) :
    pump1.rate * 12.5 = pool.volume := by
  sorry

end NUMINAMATH_CALUDE_slower_pump_fill_time_l1875_187587


namespace NUMINAMATH_CALUDE_jeds_speed_l1875_187598

def speed_limit : ℕ := 50
def speeding_fine_rate : ℕ := 16
def red_light_fine : ℕ := 75
def cellphone_fine : ℕ := 120
def total_fine : ℕ := 826

def non_speeding_fines : ℕ := 2 * red_light_fine + cellphone_fine

theorem jeds_speed :
  ∃ (speed : ℕ),
    speed = speed_limit + (total_fine - non_speeding_fines) / speeding_fine_rate ∧
    speed = 84 := by
  sorry

end NUMINAMATH_CALUDE_jeds_speed_l1875_187598
