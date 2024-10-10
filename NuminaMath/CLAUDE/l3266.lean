import Mathlib

namespace distinct_differences_sequence_length_l3266_326651

def is_valid_n (n : ℕ) : Prop :=
  ∃ k : ℕ+, (n = 4 * k ∨ n = 4 * k - 1)

theorem distinct_differences_sequence_length {n : ℕ} (h_n : n ≥ 3) :
  (∃ (a : ℕ → ℝ), (∀ i j : Fin n, i ≠ j → |a i - a (i + 1)| ≠ |a j - a (j + 1)|)) →
  is_valid_n n :=
sorry

end distinct_differences_sequence_length_l3266_326651


namespace square_root_sum_equality_l3266_326643

theorem square_root_sum_equality (n : ℕ) :
  (∃ (x : ℕ), (x : ℝ) * (2018 : ℝ)^2 = (2018 : ℝ)^20) ∧
  (Real.sqrt ((x : ℝ) * (2018 : ℝ)^2) = (2018 : ℝ)^10) →
  x = 2018^18 :=
by sorry

end square_root_sum_equality_l3266_326643


namespace geometric_sequence_properties_l3266_326603

/-- Geometric sequence with specific properties -/
def GeometricSequence (a : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, ∀ n : ℕ, a (n + 1) = a n * q

theorem geometric_sequence_properties
  (a : ℕ → ℝ)
  (h_geom : GeometricSequence a)
  (h_prop : a 2 * a 4 = a 5)
  (h_a4 : a 4 = 8) :
  ∃ (q : ℝ) (S : ℕ → ℝ),
    (q = 2) ∧
    (∀ n : ℕ, S n = 2^n - 1) ∧
    (∀ n : ℕ, S n = (a 1) * (1 - q^n) / (1 - q)) :=
sorry

end geometric_sequence_properties_l3266_326603


namespace sonya_falls_count_l3266_326629

/-- The number of times Sonya fell while ice skating --/
def sonya_falls (steven_falls stephanie_falls : ℕ) : ℕ :=
  (stephanie_falls / 2) - 2

/-- Proof that Sonya fell 6 times given the conditions --/
theorem sonya_falls_count :
  ∀ (steven_falls stephanie_falls : ℕ),
    steven_falls = 3 →
    stephanie_falls = steven_falls + 13 →
    sonya_falls steven_falls stephanie_falls = 6 := by
  sorry

end sonya_falls_count_l3266_326629


namespace apple_sale_percentage_l3266_326637

theorem apple_sale_percentage (total_apples : ℝ) (first_batch_percentage : ℝ) 
  (first_batch_profit : ℝ) (second_batch_profit : ℝ) (total_profit : ℝ) :
  first_batch_percentage > 0 ∧ first_batch_percentage < 100 →
  first_batch_profit = second_batch_profit →
  first_batch_profit = total_profit →
  (100 - first_batch_percentage) = (100 - first_batch_percentage) := by
sorry

end apple_sale_percentage_l3266_326637


namespace new_lamp_height_is_correct_l3266_326627

/-- The height of the old lamp in feet -/
def old_lamp_height : ℝ := 1

/-- The difference in height between the new and old lamp in feet -/
def height_difference : ℝ := 1.3333333333333333

/-- The height of the new lamp in feet -/
def new_lamp_height : ℝ := old_lamp_height + height_difference

theorem new_lamp_height_is_correct : new_lamp_height = 2.3333333333333333 := by
  sorry

end new_lamp_height_is_correct_l3266_326627


namespace faye_pencil_count_l3266_326609

/-- The number of rows of pencils and crayons --/
def num_rows : ℕ := 30

/-- The number of pencils in each row --/
def pencils_per_row : ℕ := 24

/-- The total number of pencils --/
def total_pencils : ℕ := num_rows * pencils_per_row

theorem faye_pencil_count : total_pencils = 720 := by
  sorry

end faye_pencil_count_l3266_326609


namespace largest_common_divisor_462_231_l3266_326670

theorem largest_common_divisor_462_231 : Nat.gcd 462 231 = 231 := by
  sorry

end largest_common_divisor_462_231_l3266_326670


namespace surface_area_of_combined_solid_l3266_326631

/-- Calculates the surface area of a rectangular solid -/
def surfaceAreaRect (l w h : ℝ) : ℝ := 2 * (l * w + l * h + w * h)

/-- Represents the combined solid formed by attaching a rectangular prism to a rectangular solid -/
structure CombinedSolid where
  mainLength : ℝ
  mainWidth : ℝ
  mainHeight : ℝ
  attachedLength : ℝ
  attachedWidth : ℝ
  attachedHeight : ℝ

/-- Calculates the total surface area of the combined solid -/
def totalSurfaceArea (s : CombinedSolid) : ℝ :=
  surfaceAreaRect s.mainLength s.mainWidth s.mainHeight +
  surfaceAreaRect s.attachedLength s.attachedWidth s.attachedHeight -
  2 * (s.attachedLength * s.attachedWidth)

/-- The specific combined solid from the problem -/
def problemSolid : CombinedSolid :=
  { mainLength := 4
    mainWidth := 3
    mainHeight := 2
    attachedLength := 2
    attachedWidth := 1
    attachedHeight := 1 }

theorem surface_area_of_combined_solid :
  totalSurfaceArea problemSolid = 58 := by
  sorry

end surface_area_of_combined_solid_l3266_326631


namespace total_guppies_per_day_l3266_326632

/-- The number of guppies eaten by a moray eel per day -/
def moray_eel_guppies : ℕ := 20

/-- The number of betta fish Jason has -/
def num_betta : ℕ := 5

/-- The number of guppies eaten by each betta fish per day -/
def betta_guppies : ℕ := 7

/-- The number of angelfish Jason has -/
def num_angelfish : ℕ := 3

/-- The number of guppies eaten by each angelfish per day -/
def angelfish_guppies : ℕ := 4

/-- The number of lionfish Jason has -/
def num_lionfish : ℕ := 2

/-- The number of guppies eaten by each lionfish per day -/
def lionfish_guppies : ℕ := 10

/-- Theorem stating the total number of guppies Jason needs to buy per day -/
theorem total_guppies_per_day :
  moray_eel_guppies +
  num_betta * betta_guppies +
  num_angelfish * angelfish_guppies +
  num_lionfish * lionfish_guppies = 87 := by
  sorry

end total_guppies_per_day_l3266_326632


namespace quadratic_inequality_l3266_326646

-- Define the quadratic function
def f (b c x : ℝ) := x^2 + b*x + c

-- Define the solution set condition
def solution_set (b c : ℝ) : Prop :=
  ∀ x, f b c x > 0 ↔ (x > 2 ∨ x < 1)

-- Theorem statement
theorem quadratic_inequality (b c : ℝ) (h : solution_set b c) :
  (b = -3 ∧ c = 2) ∧
  (∀ x, 2*x^2 - 3*x + 1 ≤ 0 ↔ 1/2 ≤ x ∧ x ≤ 1) :=
sorry

end quadratic_inequality_l3266_326646


namespace log_ratio_equals_two_thirds_l3266_326673

theorem log_ratio_equals_two_thirds :
  (Real.log 9 / Real.log 8) / (Real.log 3 / Real.log 2) = 2 / 3 := by
  sorry

end log_ratio_equals_two_thirds_l3266_326673


namespace class_A_student_count_l3266_326602

/-- The number of students who like social studies -/
def social_studies_count : ℕ := 25

/-- The number of students who like music -/
def music_count : ℕ := 32

/-- The number of students who like both social studies and music -/
def both_count : ℕ := 27

/-- The total number of students in class (A) -/
def total_students : ℕ := social_studies_count + music_count - both_count

theorem class_A_student_count :
  total_students = 30 :=
sorry

end class_A_student_count_l3266_326602


namespace correct_balanced_redox_reaction_l3266_326690

/-- Represents a chemical species in a redox reaction -/
structure ChemicalSpecies where
  formula : String
  charge : Int

/-- Represents a half-reaction in a redox reaction -/
structure HalfReaction where
  reactants : List ChemicalSpecies
  products : List ChemicalSpecies
  electrons : Int

/-- Represents a complete redox reaction -/
structure RedoxReaction where
  oxidation : HalfReaction
  reduction : HalfReaction

/-- Standard conditions in an acidic solution -/
def standardAcidicConditions : Prop := sorry

/-- Salicylic acid -/
def salicylicAcid : ChemicalSpecies := ⟨"C7H6O2", 0⟩

/-- Iron (III) ion -/
def ironIII : ChemicalSpecies := ⟨"Fe", 3⟩

/-- 2,3-dihydroxybenzoic acid -/
def dihydroxybenzoicAcid : ChemicalSpecies := ⟨"C7H6O4", 0⟩

/-- Hydrogen ion -/
def hydrogenIon : ChemicalSpecies := ⟨"H", 1⟩

/-- Iron (II) ion -/
def ironII : ChemicalSpecies := ⟨"Fe", 2⟩

/-- The balanced redox reaction between iron (III) nitrate and salicylic acid under standard acidic conditions -/
def balancedRedoxReaction (conditions : Prop) : RedoxReaction := sorry

/-- Theorem stating that the given redox reaction is the correct balanced reaction under standard acidic conditions -/
theorem correct_balanced_redox_reaction :
  standardAcidicConditions →
  balancedRedoxReaction standardAcidicConditions =
    RedoxReaction.mk
      (HalfReaction.mk [salicylicAcid] [dihydroxybenzoicAcid, hydrogenIon, hydrogenIon] 2)
      (HalfReaction.mk [ironIII, ironIII] [ironII, ironII] (-2)) :=
sorry

end correct_balanced_redox_reaction_l3266_326690


namespace average_speed_inequality_l3266_326662

theorem average_speed_inequality (a b v : ℝ) 
  (ha : 0 < a) (hb : 0 < b) (hab : a < b) 
  (hv : v = (2 * a * b) / (a + b)) : 
  a < v ∧ v < Real.sqrt (a * b) := by
  sorry

end average_speed_inequality_l3266_326662


namespace min_value_quadratic_expression_l3266_326615

theorem min_value_quadratic_expression :
  ∀ x y : ℝ, (x + 3)^2 + 2*(y - 2)^2 + 4*(x - 7)^2 + (y + 4)^2 ≥ 104 ∧
  ∃ x₀ y₀ : ℝ, (x₀ + 3)^2 + 2*(y₀ - 2)^2 + 4*(x₀ - 7)^2 + (y₀ + 4)^2 = 104 :=
by sorry

end min_value_quadratic_expression_l3266_326615


namespace unique_number_pair_l3266_326689

theorem unique_number_pair : ∃! (a b : ℕ), 
  a + b = 2015 ∧ 
  ∃ (c : ℕ), c ≤ 9 ∧ a = 10 * b + c ∧
  a = 1832 ∧ b = 183 := by
sorry

end unique_number_pair_l3266_326689


namespace marcos_strawberries_weight_l3266_326652

/-- Given the initial total weight of strawberries collected by Marco and his dad,
    the additional weight of strawberries found by dad, and dad's final weight of strawberries,
    prove that Marco's strawberries weigh 6 pounds. -/
theorem marcos_strawberries_weight
  (initial_total : ℕ)
  (dads_additional : ℕ)
  (dads_final : ℕ)
  (h1 : initial_total = 22)
  (h2 : dads_additional = 30)
  (h3 : dads_final = 16) :
  initial_total - dads_final = 6 :=
by sorry

end marcos_strawberries_weight_l3266_326652


namespace gibi_score_is_59_percent_l3266_326696

/-- Represents the exam scores of four students -/
structure ExamScores where
  max_score : ℕ
  jigi_percent : ℕ
  mike_percent : ℕ
  lizzy_percent : ℕ
  average_mark : ℕ

/-- Calculates Gibi's score percentage given the exam scores -/
def gibi_score_percent (scores : ExamScores) : ℕ :=
  let total_marks := 4 * scores.average_mark
  let other_scores := (scores.jigi_percent * scores.max_score / 100) +
                      (scores.mike_percent * scores.max_score / 100) +
                      (scores.lizzy_percent * scores.max_score / 100)
  let gibi_score := total_marks - other_scores
  (gibi_score * 100) / scores.max_score

/-- Theorem stating that Gibi's score percentage is 59% given the exam conditions -/
theorem gibi_score_is_59_percent (scores : ExamScores)
  (h1 : scores.max_score = 700)
  (h2 : scores.jigi_percent = 55)
  (h3 : scores.mike_percent = 99)
  (h4 : scores.lizzy_percent = 67)
  (h5 : scores.average_mark = 490) :
  gibi_score_percent scores = 59 := by
  sorry

end gibi_score_is_59_percent_l3266_326696


namespace greatest_prime_factor_of_4_pow_17_minus_2_pow_29_l3266_326618

theorem greatest_prime_factor_of_4_pow_17_minus_2_pow_29 :
  ∃ (p : ℕ), Prime p ∧ p ∣ (4^17 - 2^29) ∧ ∀ (q : ℕ), Prime q → q ∣ (4^17 - 2^29) → q ≤ p ∧ p = 31 :=
sorry

end greatest_prime_factor_of_4_pow_17_minus_2_pow_29_l3266_326618


namespace lines_perpendicular_iff_slope_product_neg_one_l3266_326671

/-- Two lines in the plane are perpendicular if and only if the product of their slopes is -1 -/
theorem lines_perpendicular_iff_slope_product_neg_one 
  (A₁ B₁ C₁ A₂ B₂ C₂ : ℝ) (hB₁ : B₁ ≠ 0) (hB₂ : B₂ ≠ 0) :
  (∀ x y : ℝ, A₁ * x + B₁ * y + C₁ = 0 → A₂ * x + B₂ * y + C₂ = 0 → 
    (A₁ * x + B₁ * y + C₁ = 0 ∧ A₂ * x + B₂ * y + C₂ = 0) → 
    (A₁ * A₂) / (B₁ * B₂) = -1) ↔
  (A₁ * A₂) / (B₁ * B₂) = -1 :=
by sorry

end lines_perpendicular_iff_slope_product_neg_one_l3266_326671


namespace negation_of_positive_quadratic_l3266_326650

theorem negation_of_positive_quadratic (x : ℝ) :
  (¬ ∀ x : ℝ, x^2 + x + 1 > 0) ↔ (∃ x₀ : ℝ, x₀^2 + x₀ + 1 ≤ 0) := by
  sorry

end negation_of_positive_quadratic_l3266_326650


namespace cone_generatrix_length_l3266_326607

/-- The length of the generatrix of a cone with lateral area 6π and base radius 2 is 3. -/
theorem cone_generatrix_length :
  ∀ (l : ℝ), 
    (l > 0) →
    (2 * Real.pi * l = 6 * Real.pi) →
    l = 3 := by
  sorry

end cone_generatrix_length_l3266_326607


namespace quotient_change_l3266_326695

theorem quotient_change (initial_quotient : ℝ) (dividend_multiplier : ℝ) (divisor_multiplier : ℝ) :
  initial_quotient = 0.78 →
  dividend_multiplier = 10 →
  divisor_multiplier = 0.1 →
  initial_quotient * dividend_multiplier / divisor_multiplier = 78 := by
  sorry

#check quotient_change

end quotient_change_l3266_326695


namespace expense_increase_percentage_l3266_326680

theorem expense_increase_percentage (monthly_salary : ℝ) (initial_savings_rate : ℝ) (new_savings : ℝ) :
  monthly_salary = 6500 →
  initial_savings_rate = 0.20 →
  new_savings = 260 →
  let initial_savings := monthly_salary * initial_savings_rate
  let initial_expenses := monthly_salary - initial_savings
  let expense_increase := initial_savings - new_savings
  expense_increase / initial_expenses = 0.20 := by sorry

end expense_increase_percentage_l3266_326680


namespace range_of_a_l3266_326697

-- Define sets A and B
def A : Set ℝ := {x | x^2 - x - 2 < 0}
def B (a : ℝ) : Set ℝ := {x | a < x ∧ x < a + 5}

-- Define the theorem
theorem range_of_a (a : ℝ) : A ⊆ B a ↔ -3 ≤ a ∧ a ≤ -1 := by sorry

end range_of_a_l3266_326697


namespace optimal_distribution_part1_optimal_distribution_part2_l3266_326630

/-- Represents the types of vegetables -/
inductive VegetableType
| A
| B
| C

/-- Properties of each vegetable type -/
def tons_per_truck (v : VegetableType) : ℚ :=
  match v with
  | .A => 2
  | .B => 1
  | .C => 2.5

def profit_per_ton (v : VegetableType) : ℚ :=
  match v with
  | .A => 5
  | .B => 7
  | .C => 4

/-- Theorem for part 1 -/
theorem optimal_distribution_part1 :
  ∃ (b c : ℕ),
    b + c = 14 ∧
    b * tons_per_truck VegetableType.B + c * tons_per_truck VegetableType.C = 17 ∧
    b = 12 ∧ c = 2 := by sorry

/-- Theorem for part 2 -/
theorem optimal_distribution_part2 :
  ∃ (a b c : ℕ) (max_profit : ℚ),
    a + b + c = 30 ∧
    1 ≤ a ∧ a ≤ 10 ∧
    a * tons_per_truck VegetableType.A + b * tons_per_truck VegetableType.B + c * tons_per_truck VegetableType.C = 48 ∧
    a = 9 ∧ b = 15 ∧ c = 6 ∧
    max_profit = 255 ∧
    (∀ (a' b' c' : ℕ),
      a' + b' + c' = 30 →
      1 ≤ a' ∧ a' ≤ 10 →
      a' * tons_per_truck VegetableType.A + b' * tons_per_truck VegetableType.B + c' * tons_per_truck VegetableType.C = 48 →
      a' * tons_per_truck VegetableType.A * profit_per_ton VegetableType.A +
      b' * tons_per_truck VegetableType.B * profit_per_ton VegetableType.B +
      c' * tons_per_truck VegetableType.C * profit_per_ton VegetableType.C ≤ max_profit) := by sorry

end optimal_distribution_part1_optimal_distribution_part2_l3266_326630


namespace union_equals_A_iff_m_in_range_l3266_326611

-- Define sets A and B
def A : Set ℝ := {x : ℝ | -1 ≤ x ∧ x ≤ 2}
def B (m : ℝ) : Set ℝ := {x : ℝ | x^2 - (2*m + 1)*x + 2*m < 0}

-- State the theorem
theorem union_equals_A_iff_m_in_range (m : ℝ) : 
  A ∪ B m = A ↔ -1/2 ≤ m ∧ m ≤ 1 := by
  sorry

end union_equals_A_iff_m_in_range_l3266_326611


namespace squares_sequence_correct_l3266_326605

/-- Represents the number of nonoverlapping unit squares in figure n -/
def squares (n : ℕ) : ℕ :=
  3 * n^2 + n + 1

theorem squares_sequence_correct : 
  squares 0 = 1 ∧ 
  squares 1 = 5 ∧ 
  squares 2 = 15 ∧ 
  squares 3 = 29 ∧ 
  squares 4 = 49 ∧ 
  squares 100 = 30101 :=
by sorry

end squares_sequence_correct_l3266_326605


namespace rahuls_share_l3266_326634

/-- Calculates the share of payment for a worker in a joint work scenario -/
def calculate_share (days_worker1 days_worker2 total_payment : ℚ) : ℚ :=
  let worker1_rate := 1 / days_worker1
  let worker2_rate := 1 / days_worker2
  let combined_rate := worker1_rate + worker2_rate
  let share_ratio := worker1_rate / combined_rate
  share_ratio * total_payment

/-- Theorem stating that Rahul's share of the payment is $68 -/
theorem rahuls_share :
  calculate_share 3 2 170 = 68 := by
  sorry

#eval calculate_share 3 2 170

end rahuls_share_l3266_326634


namespace problem_solution_l3266_326669

/-- An arithmetic sequence with positive terms -/
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d ∧ a n > 0

/-- A geometric sequence -/
def geometric_sequence (b : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, b (n + 1) = b n * r

theorem problem_solution (a b : ℕ → ℝ) 
    (h_arith : arithmetic_sequence a)
    (h_eq : 2 * a 3 - (a 7)^2 + 2 * a 11 = 0)
    (h_geom : geometric_sequence b)
    (h_equal : b 7 = a 7) :
  b 6 * b 8 = 16 := by
  sorry


end problem_solution_l3266_326669


namespace unique_triple_gcd_sum_l3266_326608

theorem unique_triple_gcd_sum (m n l : ℕ) : 
  (m + n = (Nat.gcd m n)^2) ∧ 
  (m + l = (Nat.gcd m l)^2) ∧ 
  (n + l = (Nat.gcd n l)^2) →
  m = 2 ∧ n = 2 ∧ l = 2 :=
by sorry

end unique_triple_gcd_sum_l3266_326608


namespace fraction_product_equals_twelve_l3266_326612

theorem fraction_product_equals_twelve :
  (1 / 3) * (9 / 2) * (1 / 27) * (54 / 1) * (1 / 81) * (162 / 1) * (1 / 243) * (486 / 1) = 12 := by
  sorry

end fraction_product_equals_twelve_l3266_326612


namespace factorization_of_x4_plus_64_l3266_326659

theorem factorization_of_x4_plus_64 (x : ℝ) : x^4 + 64 = (x^2 - 4*x + 8) * (x^2 + 4*x + 8) := by
  sorry

end factorization_of_x4_plus_64_l3266_326659


namespace income_record_l3266_326681

/-- Represents the recording of a financial transaction -/
def record (amount : ℤ) : ℤ := amount

/-- An expenditure is recorded as a negative number -/
axiom expenditure_record : record (-200) = -200

/-- Theorem: An income is recorded as a positive number -/
theorem income_record : record 60 = 60 := by sorry

end income_record_l3266_326681


namespace range_of_g_l3266_326649

theorem range_of_g (x : ℝ) : 3/4 ≤ Real.cos x ^ 4 + Real.sin x ^ 2 ∧ Real.cos x ^ 4 + Real.sin x ^ 2 ≤ 1 := by
  sorry

end range_of_g_l3266_326649


namespace equal_values_l3266_326676

-- Define the algebraic expression
def f (a : ℝ) : ℝ := a^4 - 2*a^2 + 3

-- State the theorem
theorem equal_values : f 2 = f (-2) := by sorry

end equal_values_l3266_326676


namespace fraction_to_decimal_l3266_326661

theorem fraction_to_decimal : (7 : ℚ) / 32 = 0.21875 := by sorry

end fraction_to_decimal_l3266_326661


namespace remainder_4672_div_34_l3266_326613

theorem remainder_4672_div_34 : 4672 % 34 = 14 := by
  sorry

end remainder_4672_div_34_l3266_326613


namespace function_problem_l3266_326622

/-- Given a function f(x) = x / (ax + b) where a ≠ 0, f(4) = 4/3, and f(x) = x has a unique solution,
    prove that f(x) = 2x / (x + 2) and f[f(-3)] = 3/2 -/
theorem function_problem (a b : ℝ) (ha : a ≠ 0) :
  let f : ℝ → ℝ := λ x ↦ x / (a * x + b)
  (f 4 = 4 / 3) →
  (∃! x, f x = x) →
  (∀ x, f x = 2 * x / (x + 2)) ∧
  (f (f (-3)) = 3 / 2) :=
by sorry

end function_problem_l3266_326622


namespace min_amount_lost_l3266_326624

/-- Represents the denomination of a bill -/
inductive Bill
  | ten  : Bill
  | fifty : Bill

/-- Calculates the value of a bill -/
def billValue (b : Bill) : Nat :=
  match b with
  | Bill.ten  => 10
  | Bill.fifty => 50

/-- Represents the cash transaction and usage -/
structure CashTransaction where
  totalCashed : Nat
  billsUsed : Nat
  tenBills : Nat
  fiftyBills : Nat

/-- Conditions of the problem -/
def transactionConditions (t : CashTransaction) : Prop :=
  t.totalCashed = 1270 ∧
  t.billsUsed = 15 ∧
  (t.tenBills = t.fiftyBills + 1 ∨ t.tenBills = t.fiftyBills - 1) ∧
  t.tenBills * billValue Bill.ten + t.fiftyBills * billValue Bill.fifty ≤ t.totalCashed

/-- Theorem stating the minimum amount lost -/
theorem min_amount_lost (t : CashTransaction) 
  (h : transactionConditions t) : 
  t.totalCashed - (t.tenBills * billValue Bill.ten + t.fiftyBills * billValue Bill.fifty) = 800 := by
  sorry

end min_amount_lost_l3266_326624


namespace correct_calculation_l3266_326633

theorem correct_calculation (a : ℝ) : 3 * a^2 + 2 * a^2 = 5 * a^2 := by
  sorry

end correct_calculation_l3266_326633


namespace imaginary_part_of_z_l3266_326657

theorem imaginary_part_of_z (i : ℂ) (h : i^2 = -1) : 
  Complex.im ((1 - 2*i) / i) = -1 := by sorry

end imaginary_part_of_z_l3266_326657


namespace expression_evaluation_l3266_326691

theorem expression_evaluation : 4^3 - 4 * 4^2 + 6 * 4 - 2 = 22 := by
  sorry

end expression_evaluation_l3266_326691


namespace multiply_subtract_equation_l3266_326600

theorem multiply_subtract_equation : ∃ x : ℝ, 12 * x - 3 = (12 - 7) * 9 ∧ x = 4 := by
  sorry

end multiply_subtract_equation_l3266_326600


namespace range_of_m_l3266_326601

-- Define the propositions p and q
def p (x : ℝ) : Prop := -2 ≤ 1 - (x - 1) / 3 ∧ 1 - (x - 1) / 3 ≤ 2

def q (x m : ℝ) : Prop := x^2 - 2*x + 1 - m^2 ≤ 0

-- Define the theorem
theorem range_of_m (m : ℝ) :
  (∀ x, ¬(p x) → ¬(q x m)) ∧  -- "not p" is sufficient for "not q"
  (∃ x, ¬(q x m) ∧ p x) ∧     -- "not p" is not necessary for "not q"
  (m > 0) →                   -- given condition m > 0
  m ≤ 3 :=                    -- prove that m ≤ 3
sorry

end range_of_m_l3266_326601


namespace remainder_2027_div_28_l3266_326674

theorem remainder_2027_div_28 : 2027 % 28 = 3 := by
  sorry

end remainder_2027_div_28_l3266_326674


namespace decreasing_even_shifted_function_property_l3266_326672

/-- A function that is decreasing on (8, +∞) and f(x+8) is even -/
def DecreasingEvenShiftedFunction (f : ℝ → ℝ) : Prop :=
  (∀ x y, x > 8 ∧ y > 8 ∧ x > y → f x < f y) ∧
  (∀ x, f (x + 8) = f (-x + 8))

theorem decreasing_even_shifted_function_property
  (f : ℝ → ℝ) (h : DecreasingEvenShiftedFunction f) :
  f 7 > f 10 := by
  sorry

end decreasing_even_shifted_function_property_l3266_326672


namespace min_sum_of_sides_l3266_326638

theorem min_sum_of_sides (a b c : ℝ) (A B C : ℝ) : 
  (a > 0) → (b > 0) → (c > 0) →
  (A > 0) → (B > 0) → (C > 0) →
  ((a + b)^2 - c^2 = 4) →
  (C = Real.pi / 3) →
  (∃ (x : ℝ), (a + b ≥ x) ∧ (∀ y, a + b ≥ y → x ≤ y) ∧ (x = 4 * Real.sqrt 3 / 3)) :=
by sorry

end min_sum_of_sides_l3266_326638


namespace rayman_workout_hours_l3266_326621

/-- Represents the workout hours of Rayman, Junior, and Wolverine in a week --/
structure WorkoutHours where
  rayman : ℝ
  junior : ℝ
  wolverine : ℝ

/-- Defines the relationship between Rayman's, Junior's, and Wolverine's workout hours --/
def valid_workout_hours (h : WorkoutHours) : Prop :=
  h.rayman = h.junior / 2 ∧
  h.wolverine = 2 * (h.rayman + h.junior) ∧
  h.wolverine = 60

/-- Theorem stating that Rayman works out for 10 hours in a week --/
theorem rayman_workout_hours (h : WorkoutHours) (hvalid : valid_workout_hours h) : 
  h.rayman = 10 := by
  sorry

#check rayman_workout_hours

end rayman_workout_hours_l3266_326621


namespace intersection_line_equation_l3266_326628

-- Define the two planes
def plane1 (x y z : ℝ) : Prop := 6*x - 7*y - 4*z - 2 = 0
def plane2 (x y z : ℝ) : Prop := x + 7*y - z - 5 = 0

-- Define the line equation
def line_equation (x y z : ℝ) : Prop :=
  (x - 1) / 35 = (y - 4/7) / 2 ∧ (y - 4/7) / 2 = z / 49

-- Theorem statement
theorem intersection_line_equation :
  ∀ x y z : ℝ, plane1 x y z ∧ plane2 x y z → line_equation x y z :=
by sorry

end intersection_line_equation_l3266_326628


namespace sqrt_three_addition_l3266_326687

theorem sqrt_three_addition : 2 * Real.sqrt 3 + Real.sqrt 3 = 3 * Real.sqrt 3 := by
  sorry

end sqrt_three_addition_l3266_326687


namespace product_75_360_trailing_zeros_l3266_326640

/-- The number of trailing zeros in a natural number -/
def trailingZeros (n : ℕ) : ℕ := sorry

/-- Theorem: The product of 75 and 360 has 3 trailing zeros -/
theorem product_75_360_trailing_zeros :
  trailingZeros (75 * 360) = 3 := by sorry

end product_75_360_trailing_zeros_l3266_326640


namespace subtraction_of_decimals_l3266_326677

theorem subtraction_of_decimals : 3.75 - 2.18 = 1.57 := by
  sorry

end subtraction_of_decimals_l3266_326677


namespace quadratic_minimum_l3266_326623

theorem quadratic_minimum (f : ℝ → ℝ) (h : f = λ x => (x - 1)^2 + 3) : 
  ∀ x, f x ≥ 3 ∧ ∃ x₀, f x₀ = 3 :=
sorry

end quadratic_minimum_l3266_326623


namespace Q_sufficient_not_necessary_l3266_326626

open Real

-- Define a differentiable function f on ℝ
variable (f : ℝ → ℝ)
variable (hf : Differentiable ℝ f)

-- Define proposition P
def P (f : ℝ → ℝ) : Prop :=
  ∀ x₁ x₂ : ℝ, x₁ ≠ x₂ → |(f x₁ - f x₂) / (x₁ - x₂)| < 2018

-- Define proposition Q
def Q (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, |deriv f x| < 2018

-- Theorem stating that Q is sufficient but not necessary for P
theorem Q_sufficient_not_necessary (f : ℝ → ℝ) (hf : Differentiable ℝ f) :
  (Q f → P f) ∧ ∃ g : ℝ → ℝ, Differentiable ℝ g ∧ P g ∧ ¬(Q g) := by
  sorry

end Q_sufficient_not_necessary_l3266_326626


namespace evaluate_expression_l3266_326648

theorem evaluate_expression : 3^(1^(2^8)) + ((3^1)^2)^4 = 6564 := by
  sorry

end evaluate_expression_l3266_326648


namespace factorial_base_312_b3_is_zero_l3266_326692

/-- Factorial function -/
def factorial (n : ℕ) : ℕ :=
  match n with
  | 0 => 1
  | n + 1 => (n + 1) * factorial n

/-- Checks if a list of coefficients is a valid factorial base representation -/
def isValidFactorialBase (coeffs : List ℕ) : Prop :=
  ∀ (i : ℕ), i < coeffs.length → coeffs[i]! ≤ i + 1

/-- Computes the value represented by a list of coefficients in factorial base -/
def valueFromFactorialBase (coeffs : List ℕ) : ℕ :=
  coeffs.enum.foldl (fun acc (i, b) => acc + b * factorial (i + 1)) 0

/-- Theorem: The factorial base representation of 312 has b₃ = 0 -/
theorem factorial_base_312_b3_is_zero :
  ∃ (coeffs : List ℕ),
    isValidFactorialBase coeffs ∧
    valueFromFactorialBase coeffs = 312 ∧
    coeffs.length > 3 ∧
    coeffs[2]! = 0 :=
by sorry

end factorial_base_312_b3_is_zero_l3266_326692


namespace f_strictly_increasing_l3266_326683

def f (x : ℝ) : ℝ := x^3 - x^2 - x

theorem f_strictly_increasing :
  (∀ x y, x < y ∧ x < -1/3 → f x < f y) ∧
  (∀ x y, x < y ∧ 1 < x → f x < f y) :=
sorry

end f_strictly_increasing_l3266_326683


namespace prime_power_divisibility_l3266_326658

theorem prime_power_divisibility (p : ℕ) (h_prime : Nat.Prime p) (h_p_gt_3 : p > 3) :
  let n := (2^(2*p) - 1) / 3
  n ∣ (2^n - 2) := by
  sorry

end prime_power_divisibility_l3266_326658


namespace susan_peaches_in_knapsack_l3266_326619

/-- The number of peaches Susan bought -/
def total_peaches : ℕ := 5 * 12

/-- The number of cloth bags Susan has -/
def num_cloth_bags : ℕ := 2

/-- Represents the relationship between peaches in cloth bags and knapsack -/
def knapsack_ratio : ℚ := 1 / 2

/-- The number of peaches in the knapsack -/
def peaches_in_knapsack : ℕ := 12

theorem susan_peaches_in_knapsack :
  ∃ (x : ℕ), 
    (x : ℚ) * num_cloth_bags + (x : ℚ) * knapsack_ratio = total_peaches ∧
    peaches_in_knapsack = (x : ℚ) * knapsack_ratio := by
  sorry

end susan_peaches_in_knapsack_l3266_326619


namespace winnie_keeps_remainder_l3266_326666

/-- The number of balloons Winnie keeps for herself -/
def balloons_kept (total_balloons : ℕ) (num_friends : ℕ) : ℕ :=
  total_balloons % num_friends

/-- The total number of balloons Winnie has -/
def total_balloons : ℕ := 17 + 33 + 65 + 83

/-- The number of friends Winnie has -/
def num_friends : ℕ := 10

theorem winnie_keeps_remainder :
  balloons_kept total_balloons num_friends = 8 :=
sorry

end winnie_keeps_remainder_l3266_326666


namespace f_extrema_and_inequality_l3266_326616

noncomputable def f (x : ℝ) : ℝ := x^2 + Real.log x

noncomputable def g (x : ℝ) : ℝ := (2/3) * x^3 + (1/2) * x^2

theorem f_extrema_and_inequality :
  (∀ x ∈ Set.Icc 1 (Real.exp 1), f x ≥ 1) ∧
  (∀ x ∈ Set.Icc 1 (Real.exp 1), f x ≤ 1 + (Real.exp 1)^2) ∧
  (∀ x ∈ Set.Ioi 1, f x < g x) :=
by sorry

end f_extrema_and_inequality_l3266_326616


namespace common_chord_intersection_l3266_326641

-- Define a type for points in a plane
variable (Point : Type)

-- Define a type for circles
variable (Circle : Type)

-- Function to check if a point is on a circle
variable (on_circle : Point → Circle → Prop)

-- Function to check if two circles intersect
variable (intersect : Circle → Circle → Prop)

-- Function to create a circle passing through two points
variable (circle_through : Point → Point → Circle)

-- Function to find the common chord of two circles
variable (common_chord : Circle → Circle → Set Point)

-- Theorem statement
theorem common_chord_intersection
  (A B C D : Point)
  (h : ∀ (c1 c2 : Circle), on_circle A c1 → on_circle B c1 → 
                           on_circle C c2 → on_circle D c2 → 
                           intersect c1 c2) :
  ∃ (P : Point), ∀ (c1 c2 : Circle),
    on_circle A c1 → on_circle B c1 →
    on_circle C c2 → on_circle D c2 →
    P ∈ common_chord c1 c2 :=
sorry

end common_chord_intersection_l3266_326641


namespace friends_money_sharing_l3266_326682

theorem friends_money_sharing (A : ℝ) (h_pos : A > 0) :
  let jorge_total := 5 * A
  let jose_total := 4 * A
  let janio_total := 3 * A
  let joao_received := 3 * A
  let group_total := jorge_total + jose_total + janio_total
  (joao_received / group_total) = (1 : ℝ) / 4 := by
sorry

end friends_money_sharing_l3266_326682


namespace cubic_equation_sum_l3266_326636

theorem cubic_equation_sum (p q r : ℝ) : 
  (p^3 - 6*p^2 + 11*p = 14) → 
  (q^3 - 6*q^2 + 11*q = 14) → 
  (r^3 - 6*r^2 + 11*r = 14) → 
  (p*q/r + q*r/p + r*p/q = -47/14) := by
  sorry

end cubic_equation_sum_l3266_326636


namespace alpha_value_l3266_326647

theorem alpha_value (α β : ℂ) 
  (h1 : (α + β).im = 0 ∧ (α + β).re > 0)
  (h2 : (Complex.I * (α - 3 * β)).im = 0 ∧ (Complex.I * (α - 3 * β)).re > 0)
  (h3 : β = 4 + 3 * Complex.I) :
  α = 12 - 3 * Complex.I := by sorry

end alpha_value_l3266_326647


namespace aunt_wang_lilies_l3266_326698

theorem aunt_wang_lilies (rose_cost lily_cost roses_bought total_spent : ℕ) : 
  rose_cost = 5 →
  lily_cost = 9 →
  roses_bought = 2 →
  total_spent = 55 →
  (total_spent - rose_cost * roses_bought) / lily_cost = 5 := by
  sorry

end aunt_wang_lilies_l3266_326698


namespace geometric_sequence_ratio_l3266_326664

/-- Geometric sequence with common ratio greater than 1 -/
def GeometricSequence (a : ℕ → ℝ) (q : ℝ) : Prop :=
  q > 1 ∧ ∀ n : ℕ, a (n + 1) = q * a n

theorem geometric_sequence_ratio (a : ℕ → ℝ) (q : ℝ) 
  (h_geo : GeometricSequence a q)
  (h_sum : a 1 + a 4 = 9)
  (h_prod : a 2 * a 3 = 8) :
  (a 2015 + a 2016) / (a 2013 + a 2014) = 4 := by
  sorry

end geometric_sequence_ratio_l3266_326664


namespace prime_count_in_range_l3266_326610

theorem prime_count_in_range (n : ℕ) (h : n > 2) :
  (n > 3 → ∀ p, Nat.Prime p → ¬((n - 1).factorial + 2 < p ∧ p < (n - 1).factorial + n)) ∧
  (n = 3 → ∃! p, Nat.Prime p ∧ ((n - 1).factorial + 2 < p ∧ p < (n - 1).factorial + n)) :=
sorry

end prime_count_in_range_l3266_326610


namespace faiths_weekly_earnings_l3266_326667

/-- Faith's weekly earnings calculation --/
theorem faiths_weekly_earnings
  (hourly_rate : ℝ)
  (regular_hours_per_day : ℕ)
  (working_days_per_week : ℕ)
  (overtime_hours_per_day : ℕ)
  (h1 : hourly_rate = 13.5)
  (h2 : regular_hours_per_day = 8)
  (h3 : working_days_per_week = 5)
  (h4 : overtime_hours_per_day = 2) :
  let regular_pay := hourly_rate * regular_hours_per_day * working_days_per_week
  let overtime_pay := hourly_rate * overtime_hours_per_day * working_days_per_week
  let total_earnings := regular_pay + overtime_pay
  total_earnings = 675 := by sorry

end faiths_weekly_earnings_l3266_326667


namespace six_quarters_around_nickel_l3266_326694

/-- Represents the arrangement of coins on a table -/
structure CoinArrangement where
  nickelDiameter : ℝ
  quarterDiameter : ℝ

/-- Calculates the maximum number of quarters that can be placed around a nickel -/
def maxQuarters (arrangement : CoinArrangement) : ℕ :=
  sorry

/-- Theorem stating that for the given coin sizes, 6 quarters can be placed around a nickel -/
theorem six_quarters_around_nickel :
  let arrangement : CoinArrangement := { nickelDiameter := 2, quarterDiameter := 2.4 }
  maxQuarters arrangement = 6 := by
  sorry

end six_quarters_around_nickel_l3266_326694


namespace cupboard_cost_price_l3266_326688

/-- The cost price of a cupboard satisfying certain conditions -/
def cost_price : ℝ := 5625

/-- The selling price of the cupboard -/
def selling_price : ℝ := 0.84 * cost_price

/-- The increased selling price that would result in a profit -/
def increased_selling_price : ℝ := 1.16 * cost_price

/-- Theorem stating that the cost price satisfies the given conditions -/
theorem cupboard_cost_price : 
  selling_price = 0.84 * cost_price ∧ 
  increased_selling_price = selling_price + 1800 :=
by sorry

end cupboard_cost_price_l3266_326688


namespace zeros_before_first_nonzero_digit_l3266_326660

def fraction : ℚ := 3 / (2^7 * 5^10)

theorem zeros_before_first_nonzero_digit : 
  (∃ (n : ℕ) (d : ℚ), fraction * 10^n = d ∧ d ≥ 1 ∧ d < 10 ∧ n = 8) :=
sorry

end zeros_before_first_nonzero_digit_l3266_326660


namespace adult_meal_cost_l3266_326668

/-- Proves that the cost of each adult meal is $6 given the conditions of the restaurant bill. -/
theorem adult_meal_cost (num_adults num_children : ℕ) (child_meal_cost soda_cost total_bill : ℚ) :
  num_adults = 6 →
  num_children = 2 →
  child_meal_cost = 4 →
  soda_cost = 2 →
  total_bill = 60 →
  ∃ (adult_meal_cost : ℚ),
    adult_meal_cost * num_adults + child_meal_cost * num_children + soda_cost * (num_adults + num_children) = total_bill ∧
    adult_meal_cost = 6 :=
by sorry

end adult_meal_cost_l3266_326668


namespace larger_solution_quadratic_equation_l3266_326655

theorem larger_solution_quadratic_equation :
  ∃ (x y : ℝ), x ≠ y ∧ 
  x^2 - 13*x + 36 = 0 ∧
  y^2 - 13*y + 36 = 0 ∧
  (∀ z : ℝ, z^2 - 13*z + 36 = 0 → z = x ∨ z = y) ∧
  max x y = 9 := by
sorry

end larger_solution_quadratic_equation_l3266_326655


namespace angle_A_measure_max_area_l3266_326604

-- Define a triangle ABC
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ
  -- Condition that sides are positive
  ha : a > 0
  hb : b > 0
  hc : c > 0
  -- Condition that angles are between 0 and π
  hA : 0 < A ∧ A < π
  hB : 0 < B ∧ B < π
  hC : 0 < C ∧ C < π
  -- Condition that angles sum to π
  hsum : A + B + C = π
  -- Law of cosines
  hlawA : a^2 = b^2 + c^2 - 2*b*c*Real.cos A
  hlawB : b^2 = a^2 + c^2 - 2*a*c*Real.cos B
  hlawC : c^2 = a^2 + b^2 - 2*a*b*Real.cos C

-- Part 1
theorem angle_A_measure (t : Triangle) (h : t.b^2 + t.c^2 - t.a^2 + t.b*t.c = 0) :
  t.A = 2*π/3 := by sorry

-- Part 2
theorem max_area (t : Triangle) (h1 : t.b^2 + t.c^2 - t.a^2 + t.b*t.c = 0) (h2 : t.a = Real.sqrt 3) :
  (t.b * t.c * Real.sin t.A / 2) ≤ Real.sqrt 3 / 4 := by sorry

end angle_A_measure_max_area_l3266_326604


namespace pyramid_volume_transformation_l3266_326699

/-- Represents a pyramid with a triangular base -/
structure Pyramid where
  volume : ℝ
  base_a : ℝ
  base_b : ℝ
  base_c : ℝ
  height : ℝ

/-- Transforms a pyramid according to the given conditions -/
def transform_pyramid (p : Pyramid) : Pyramid :=
  { volume := 0,  -- We'll prove this is 12 * p.volume
    base_a := 2 * p.base_a,
    base_b := 2 * p.base_b,
    base_c := 3 * p.base_c,
    height := 3 * p.height }

theorem pyramid_volume_transformation (p : Pyramid) :
  (transform_pyramid p).volume = 12 * p.volume := by
  sorry

#check pyramid_volume_transformation

end pyramid_volume_transformation_l3266_326699


namespace all_players_odd_sum_probability_l3266_326678

def number_of_tiles : ℕ := 15
def number_of_players : ℕ := 5
def tiles_per_player : ℕ := 3

def probability_all_odd_sum : ℚ :=
  480 / 19019

theorem all_players_odd_sum_probability :
  (number_of_tiles = 15) →
  (number_of_players = 5) →
  (tiles_per_player = 3) →
  probability_all_odd_sum = 480 / 19019 :=
by sorry

end all_players_odd_sum_probability_l3266_326678


namespace function_zero_nonpositive_l3266_326656

/-- A function satisfying the given inequality property -/
def SatisfiesInequality (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, f (x + y) ≤ y * f x + f (f x)

/-- The main theorem to prove -/
theorem function_zero_nonpositive (f : ℝ → ℝ) (h : SatisfiesInequality f) :
    ∀ x : ℝ, x ≤ 0 → f x = 0 := by
  sorry

end function_zero_nonpositive_l3266_326656


namespace laura_pants_purchase_l3266_326684

def pants_cost : ℕ := 54
def shirt_cost : ℕ := 33
def num_shirts : ℕ := 4
def money_given : ℕ := 250
def change_received : ℕ := 10

theorem laura_pants_purchase :
  (money_given - change_received - num_shirts * shirt_cost) / pants_cost = 2 := by
  sorry

end laura_pants_purchase_l3266_326684


namespace digits_of_multiples_of_3_l3266_326635

/-- The number of multiples of 3 from 1 to 100 -/
def multiplesOf3 : ℕ := 33

/-- The number of single-digit multiples of 3 from 1 to 100 -/
def singleDigitMultiples : ℕ := 3

/-- The number of two-digit multiples of 3 from 1 to 100 -/
def twoDigitMultiples : ℕ := multiplesOf3 - singleDigitMultiples

/-- The total number of digits written when listing all multiples of 3 from 1 to 100 -/
def totalDigits : ℕ := singleDigitMultiples * 1 + twoDigitMultiples * 2

theorem digits_of_multiples_of_3 : totalDigits = 63 := by
  sorry

end digits_of_multiples_of_3_l3266_326635


namespace seating_arrangements_count_l3266_326614

/-- Represents the number of people to be seated. -/
def num_people : ℕ := 4

/-- Represents the total number of chairs in a row. -/
def total_chairs : ℕ := 8

/-- Represents the number of consecutive empty seats required. -/
def consecutive_empty_seats : ℕ := 3

/-- Calculates the number of seating arrangements for the given conditions. -/
def seating_arrangements (p : ℕ) (c : ℕ) (e : ℕ) : ℕ :=
  (Nat.factorial (p + 1)) * (c - p - e + 1)

/-- Theorem stating the number of seating arrangements for the given conditions. -/
theorem seating_arrangements_count :
  seating_arrangements num_people total_chairs consecutive_empty_seats = 600 :=
by
  sorry


end seating_arrangements_count_l3266_326614


namespace blackboard_multiplication_l3266_326679

theorem blackboard_multiplication (a b : ℕ) (n : ℕ+) : 
  (100 ≤ a ∧ a ≤ 999) →
  (100 ≤ b ∧ b ≤ 999) →
  10000 * a + b = n * (a * b) →
  n = 73 := by sorry

end blackboard_multiplication_l3266_326679


namespace trapezoid_longest_diagonal_lower_bound_trapezoid_longest_diagonal_lower_bound_tight_l3266_326675

/-- A trapezoid with area 1 -/
structure Trapezoid :=
  (a b h : ℝ)  -- lengths of bases and height
  (d₁ d₂ : ℝ)  -- lengths of diagonals
  (area_eq : (a + b) * h / 2 = 1)
  (d₁_ge_d₂ : d₁ ≥ d₂)

/-- The longest diagonal of a trapezoid with area 1 is at least √2 -/
theorem trapezoid_longest_diagonal_lower_bound (T : Trapezoid) : 
  T.d₁ ≥ Real.sqrt 2 := by sorry

/-- There exists a trapezoid with area 1 whose longest diagonal is exactly √2 -/
theorem trapezoid_longest_diagonal_lower_bound_tight : 
  ∃ T : Trapezoid, T.d₁ = Real.sqrt 2 := by sorry

end trapezoid_longest_diagonal_lower_bound_trapezoid_longest_diagonal_lower_bound_tight_l3266_326675


namespace not_always_valid_solution_set_l3266_326665

theorem not_always_valid_solution_set (a b : ℝ) (h : b ≠ 0) :
  ¬ (∀ x, x ∈ Set.Ioi (b / a) ↔ a * x + b > 0) :=
sorry

end not_always_valid_solution_set_l3266_326665


namespace problem_1_problem_2_l3266_326654

variable (x y : ℝ)

theorem problem_1 : ((x * y + 2) * (x * y - 2) - 2 * x^2 * y^2 + 4) / (x * y) = -x * y :=
by sorry

theorem problem_2 : (2 * x + y)^2 - (2 * x + 3 * y) * (2 * x - 3 * y) = 4 * x * y + 10 * y^2 :=
by sorry

end problem_1_problem_2_l3266_326654


namespace rat_speed_l3266_326693

/-- Proves that under given conditions, the rat's speed is 36 kmph -/
theorem rat_speed (head_start : ℝ) (catch_up_time : ℝ) (cat_speed : ℝ)
  (h1 : head_start = 6)
  (h2 : catch_up_time = 4)
  (h3 : cat_speed = 90) :
  let rat_speed := (cat_speed * catch_up_time) / (head_start + catch_up_time)
  rat_speed = 36 := by
sorry

end rat_speed_l3266_326693


namespace f_composition_value_l3266_326639

noncomputable def f (z : ℂ) : ℂ :=
  if z.im = 0 then
    if z.re = 0 then 2 * z else -z^2 + 1
  else z^2 + 1

theorem f_composition_value : f (f (f (f (1 + I)))) = 378 + 336 * I := by
  sorry

end f_composition_value_l3266_326639


namespace rectangle_area_l3266_326663

theorem rectangle_area (x y : ℝ) 
  (h1 : (x + 3.5) * (y - 1.5) = x * y)
  (h2 : (x - 3.5) * (y + 2.5) = x * y)
  (h3 : 2 * (x + 3.5) + 2 * (y - 3.5) = 2 * x + 2 * y) :
  x * y = 196 := by sorry

end rectangle_area_l3266_326663


namespace other_root_of_quadratic_l3266_326620

theorem other_root_of_quadratic (m : ℝ) : 
  ((-4 : ℝ)^2 + m * (-4) - 20 = 0) → (5^2 + m * 5 - 20 = 0) := by
  sorry

end other_root_of_quadratic_l3266_326620


namespace parallelogram_count_l3266_326625

/-- Given a triangle ABC with each side divided into n equal segments and connected by parallel lines,
    f(n) represents the total number of parallelograms formed within the network. -/
def f (n : ℕ) : ℕ := 3 * (Nat.choose (n + 2) 4)

/-- Theorem stating that f(n) correctly counts the number of parallelograms in the described configuration. -/
theorem parallelogram_count (n : ℕ) : 
  f n = 3 * (Nat.choose (n + 2) 4) := by sorry

end parallelogram_count_l3266_326625


namespace jakes_test_average_l3266_326644

theorem jakes_test_average : 
  let first_test : ℕ := 80
  let second_test : ℕ := first_test + 10
  let third_test : ℕ := 65
  let fourth_test : ℕ := third_test
  let total_marks : ℕ := first_test + second_test + third_test + fourth_test
  let num_tests : ℕ := 4
  (total_marks : ℚ) / num_tests = 75 := by
  sorry

end jakes_test_average_l3266_326644


namespace intersection_A_B_l3266_326617

-- Define sets A and B
def A : Set ℝ := {x | (x + 2) * (x - 3) < 0}
def B : Set ℝ := {x | x > 2}

-- State the theorem
theorem intersection_A_B : A ∩ B = {x | 2 < x ∧ x < 3} := by
  sorry

end intersection_A_B_l3266_326617


namespace tony_graduate_degree_time_l3266_326653

/-- Time spent on graduate degree in physics -/
def graduate_degree_time (first_degree_time additional_degree_time number_of_additional_degrees total_school_time : ℕ) : ℕ :=
  total_school_time - (first_degree_time + additional_degree_time * number_of_additional_degrees)

/-- Theorem stating that Tony's graduate degree time is 2 years -/
theorem tony_graduate_degree_time :
  graduate_degree_time 4 4 2 14 = 2 := by
  sorry

end tony_graduate_degree_time_l3266_326653


namespace length_of_angle_bisector_l3266_326686

-- Define the triangle PQR
def Triangle (P Q R : ℝ × ℝ) : Prop :=
  let pq := Real.sqrt ((P.1 - Q.1)^2 + (P.2 - Q.2)^2)
  let qr := Real.sqrt ((Q.1 - R.1)^2 + (Q.2 - R.2)^2)
  let pr := Real.sqrt ((P.1 - R.1)^2 + (P.2 - R.2)^2)
  pq = 8 ∧ qr = 15 ∧ pr = 17

-- Define the angle bisector PS
def AngleBisector (P Q R S : ℝ × ℝ) : Prop :=
  let pq := Real.sqrt ((P.1 - Q.1)^2 + (P.2 - Q.2)^2)
  let qr := Real.sqrt ((Q.1 - R.1)^2 + (Q.2 - R.2)^2)
  let qs := Real.sqrt ((Q.1 - S.1)^2 + (Q.2 - S.2)^2)
  let rs := Real.sqrt ((R.1 - S.1)^2 + (R.2 - S.2)^2)
  qs / rs = pq / qr

-- Theorem statement
theorem length_of_angle_bisector 
  (P Q R S : ℝ × ℝ) 
  (h1 : Triangle P Q R) 
  (h2 : AngleBisector P Q R S) : 
  Real.sqrt ((P.1 - S.1)^2 + (P.2 - S.2)^2) = Real.sqrt 87.04 :=
by
  sorry

end length_of_angle_bisector_l3266_326686


namespace increase_by_percentage_increase_120_by_75_percent_l3266_326606

theorem increase_by_percentage (initial : ℝ) (percentage : ℝ) :
  initial * (1 + percentage / 100) = initial + initial * (percentage / 100) := by sorry

theorem increase_120_by_75_percent :
  120 * (1 + 75 / 100) = 210 := by sorry

end increase_by_percentage_increase_120_by_75_percent_l3266_326606


namespace smallest_lcm_with_gcd_5_l3266_326685

theorem smallest_lcm_with_gcd_5 (k l : ℕ) : 
  1000 ≤ k ∧ k < 10000 ∧ 
  1000 ≤ l ∧ l < 10000 ∧ 
  Nat.gcd k l = 5 →
  ∀ m n : ℕ, 1000 ≤ m ∧ m < 10000 ∧ 
             1000 ≤ n ∧ n < 10000 ∧ 
             Nat.gcd m n = 5 →
  Nat.lcm k l ≤ Nat.lcm m n ∧
  Nat.lcm k l = 203010 :=
by sorry

end smallest_lcm_with_gcd_5_l3266_326685


namespace original_production_was_125_l3266_326642

/-- Represents the clothing production problem --/
structure ClothingProduction where
  plannedDays : ℕ
  actualDailyProduction : ℕ
  daysAheadOfSchedule : ℕ

/-- Calculates the original planned daily production --/
def originalPlannedProduction (cp : ClothingProduction) : ℚ :=
  (cp.actualDailyProduction * (cp.plannedDays - cp.daysAheadOfSchedule)) / cp.plannedDays

/-- Theorem stating that the original planned production was 125 sets per day --/
theorem original_production_was_125 (cp : ClothingProduction) 
  (h1 : cp.plannedDays = 30)
  (h2 : cp.actualDailyProduction = 150)
  (h3 : cp.daysAheadOfSchedule = 5) :
  originalPlannedProduction cp = 125 := by
  sorry

#eval originalPlannedProduction ⟨30, 150, 5⟩

end original_production_was_125_l3266_326642


namespace difference_of_squares_601_599_l3266_326645

theorem difference_of_squares_601_599 : 601^2 - 599^2 = 2400 := by
  sorry

end difference_of_squares_601_599_l3266_326645
