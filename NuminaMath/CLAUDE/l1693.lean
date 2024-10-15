import Mathlib

namespace NUMINAMATH_CALUDE_max_trig_ratio_max_trig_ratio_equals_one_l1693_169362

theorem max_trig_ratio (x : ℝ) : 
  (Real.sin x)^4 + (Real.cos x)^4 + 2 ≤ (Real.sin x)^2 + (Real.cos x)^2 + 2 := by
  sorry

theorem max_trig_ratio_equals_one :
  ∃ x : ℝ, (Real.sin x)^4 + (Real.cos x)^4 + 2 = (Real.sin x)^2 + (Real.cos x)^2 + 2 := by
  sorry

end NUMINAMATH_CALUDE_max_trig_ratio_max_trig_ratio_equals_one_l1693_169362


namespace NUMINAMATH_CALUDE_program_arrangement_count_l1693_169304

def num_singing_programs : ℕ := 4
def num_skit_programs : ℕ := 2
def num_singing_between_skits : ℕ := 3

def arrange_programs : ℕ := sorry

theorem program_arrangement_count :
  arrange_programs = 96 := by sorry

end NUMINAMATH_CALUDE_program_arrangement_count_l1693_169304


namespace NUMINAMATH_CALUDE_regular_polygon_150_degrees_l1693_169369

/-- A regular polygon with interior angles measuring 150 degrees has 12 sides. -/
theorem regular_polygon_150_degrees : 
  ∀ n : ℕ, 
  n > 2 → 
  (180 * (n - 2) : ℝ) = 150 * n → 
  n = 12 :=
by sorry

end NUMINAMATH_CALUDE_regular_polygon_150_degrees_l1693_169369


namespace NUMINAMATH_CALUDE_leading_coefficient_is_negative_seven_l1693_169339

def polynomial (x : ℝ) : ℝ := -3 * (x^4 - 2*x^3 + 3*x) + 8 * (x^4 + 5) - 4 * (3*x^4 + x^3 + 1)

theorem leading_coefficient_is_negative_seven :
  ∃ (f : ℝ → ℝ) (a : ℝ), a ≠ 0 ∧ (∀ x, polynomial x = a * x^4 + f x) ∧ a = -7 := by
  sorry

end NUMINAMATH_CALUDE_leading_coefficient_is_negative_seven_l1693_169339


namespace NUMINAMATH_CALUDE_opposite_unit_vector_l1693_169392

def vector_a : Fin 2 → ℝ := ![4, 2]

theorem opposite_unit_vector :
  let magnitude := Real.sqrt (vector_a 0 ^ 2 + vector_a 1 ^ 2)
  let opposite_unit_vector := fun i => -vector_a i / magnitude
  opposite_unit_vector 0 = -2 * Real.sqrt 5 / 5 ∧
  opposite_unit_vector 1 = -Real.sqrt 5 / 5 := by
  sorry

end NUMINAMATH_CALUDE_opposite_unit_vector_l1693_169392


namespace NUMINAMATH_CALUDE_composite_sum_divisibility_l1693_169398

theorem composite_sum_divisibility (s : ℕ) (h1 : s ≥ 4) :
  (∃ (a b c d : ℕ+), (a + b + c + d : ℕ) = s ∧ s ∣ (a * b * c + a * b * d + a * c * d + b * c * d)) ↔
  ¬ Nat.Prime s :=
sorry

end NUMINAMATH_CALUDE_composite_sum_divisibility_l1693_169398


namespace NUMINAMATH_CALUDE_smallest_positive_d_l1693_169387

theorem smallest_positive_d : ∃ d : ℝ,
  d > 0 ∧
  (2 * Real.sqrt 7)^2 + (d + 5)^2 = (2 * d + 1)^2 ∧
  ∀ d' : ℝ, d' > 0 → (2 * Real.sqrt 7)^2 + (d' + 5)^2 = (2 * d' + 1)^2 → d ≤ d' ∧
  d = 1 + Real.sqrt 660 / 6 :=
by sorry

end NUMINAMATH_CALUDE_smallest_positive_d_l1693_169387


namespace NUMINAMATH_CALUDE_definite_integral_problem_l1693_169332

theorem definite_integral_problem (f : ℝ → ℝ) :
  (∫ x in (π/4)..(π/2), (x * Real.cos x + Real.sin x) / (x * Real.sin x)^2) = (4 * Real.sqrt 2 - 2) / π := by
  sorry

end NUMINAMATH_CALUDE_definite_integral_problem_l1693_169332


namespace NUMINAMATH_CALUDE_milford_future_age_l1693_169317

/-- Proves that Milford's age in 3 years will be 21, given the conditions about Eustace's age. -/
theorem milford_future_age :
  ∀ (eustace_age milford_age : ℕ),
  eustace_age = 2 * milford_age →
  eustace_age + 3 = 39 →
  milford_age + 3 = 21 := by
sorry

end NUMINAMATH_CALUDE_milford_future_age_l1693_169317


namespace NUMINAMATH_CALUDE_cos_2theta_plus_pi_l1693_169319

theorem cos_2theta_plus_pi (θ : Real) (h : Real.tan θ = 2) : 
  Real.cos (2 * θ + Real.pi) = 3 / 5 := by
  sorry

end NUMINAMATH_CALUDE_cos_2theta_plus_pi_l1693_169319


namespace NUMINAMATH_CALUDE_eggs_per_omelet_is_two_l1693_169354

/-- Represents the number of eggs per omelet for the Rotary Club's Omelet Breakfast. -/
def eggs_per_omelet : ℚ :=
  let small_children_tickets : ℕ := 53
  let older_children_tickets : ℕ := 35
  let adult_tickets : ℕ := 75
  let senior_tickets : ℕ := 37
  let small_children_omelets : ℚ := 0.5
  let older_children_omelets : ℚ := 1
  let adult_omelets : ℚ := 2
  let senior_omelets : ℚ := 1.5
  let extra_omelets : ℕ := 25
  let total_eggs : ℕ := 584
  let total_omelets : ℚ := small_children_tickets * small_children_omelets +
                           older_children_tickets * older_children_omelets +
                           adult_tickets * adult_omelets +
                           senior_tickets * senior_omelets +
                           extra_omelets
  total_eggs / total_omelets

/-- Theorem stating that the number of eggs per omelet is 2. -/
theorem eggs_per_omelet_is_two : eggs_per_omelet = 2 := by
  sorry

end NUMINAMATH_CALUDE_eggs_per_omelet_is_two_l1693_169354


namespace NUMINAMATH_CALUDE_change_ratio_for_quadratic_function_l1693_169338

/-- Given a function f(x) = 2x^2 - 4, prove that the ratio of change in y to change in x
    between the points (1, -2) and (1 + Δx, -2 + Δy) is equal to 4 + 2Δx -/
theorem change_ratio_for_quadratic_function (Δx Δy : ℝ) :
  let f : ℝ → ℝ := λ x ↦ 2 * x^2 - 4
  f 1 = -2 →
  f (1 + Δx) = -2 + Δy →
  Δy / Δx = 4 + 2 * Δx :=
by
  sorry

end NUMINAMATH_CALUDE_change_ratio_for_quadratic_function_l1693_169338


namespace NUMINAMATH_CALUDE_optimal_ticket_price_l1693_169311

/-- Represents the net income function for the cinema --/
def net_income (x : ℕ) : ℝ :=
  if x ≤ 10 then 100 * x - 575
  else -3 * x^2 + 130 * x - 575

/-- The domain of valid ticket prices --/
def valid_price (x : ℕ) : Prop :=
  6 ≤ x ∧ x ≤ 38

theorem optimal_ticket_price :
  ∀ x : ℕ, valid_price x → net_income x ≤ net_income 22 :=
sorry

end NUMINAMATH_CALUDE_optimal_ticket_price_l1693_169311


namespace NUMINAMATH_CALUDE_f_le_one_l1693_169383

noncomputable def f (x : ℝ) : ℝ := (1 + Real.log x) / x

theorem f_le_one (x : ℝ) (hx : x > 0) : f x ≤ 1 := by
  sorry

end NUMINAMATH_CALUDE_f_le_one_l1693_169383


namespace NUMINAMATH_CALUDE_min_value_arithmetic_sequence_l1693_169360

/-- Given an arithmetic sequence {a_n} with common difference d ≠ 0,
    where a₁ = 1 and a₁, a₃, a₁₃ form a geometric sequence,
    prove that the minimum value of (2S_n + 8) / (a_n + 3) for n ∈ ℕ* is 5/2,
    where S_n is the sum of the first n terms of {a_n}. -/
theorem min_value_arithmetic_sequence (d : ℝ) (h_d : d ≠ 0) :
  let a : ℕ → ℝ := λ n => 1 + (n - 1) * d
  let S : ℕ → ℝ := λ n => (n * (2 + (n - 1) * d)) / 2
  (a 1 = 1) ∧ 
  (a 1 * a 13 = (a 3)^2) →
  (∃ (n : ℕ), n > 0 ∧ (2 * S n + 8) / (a n + 3) = 5/2) ∧
  (∀ (n : ℕ), n > 0 → (2 * S n + 8) / (a n + 3) ≥ 5/2) :=
by sorry

end NUMINAMATH_CALUDE_min_value_arithmetic_sequence_l1693_169360


namespace NUMINAMATH_CALUDE_final_deficit_is_twelve_l1693_169380

/-- Calculates the final score difference for Liz's basketball game --/
def final_score_difference (initial_deficit : ℕ) 
  (liz_free_throws liz_threes liz_jumps liz_and_ones : ℕ)
  (taylor_threes taylor_jumps : ℕ)
  (opp1_threes : ℕ)
  (opp2_jumps opp2_free_throws : ℕ)
  (opp3_jumps opp3_threes : ℕ) : ℤ :=
  let liz_score := liz_free_throws + 3 * liz_threes + 2 * liz_jumps + 3 * liz_and_ones
  let taylor_score := 3 * taylor_threes + 2 * taylor_jumps
  let opp1_score := 3 * opp1_threes
  let opp2_score := 2 * opp2_jumps + opp2_free_throws
  let opp3_score := 2 * opp3_jumps + 3 * opp3_threes
  let team_score_diff := (liz_score + taylor_score) - (opp1_score + opp2_score + opp3_score)
  initial_deficit - team_score_diff

theorem final_deficit_is_twelve :
  final_score_difference 25 5 4 5 1 2 3 4 4 2 2 1 = 12 := by
  sorry

end NUMINAMATH_CALUDE_final_deficit_is_twelve_l1693_169380


namespace NUMINAMATH_CALUDE_max_chocolate_bars_correct_l1693_169337

/-- The maximum number of chocolate bars Henrique could buy -/
def max_chocolate_bars : ℕ :=
  7

/-- The cost of each chocolate bar in dollars -/
def cost_per_bar : ℚ :=
  135/100

/-- The amount Henrique paid in dollars -/
def amount_paid : ℚ :=
  10

/-- Theorem stating that max_chocolate_bars is the maximum number of bars Henrique could buy -/
theorem max_chocolate_bars_correct :
  (max_chocolate_bars : ℚ) * cost_per_bar < amount_paid ∧
  ((max_chocolate_bars + 1 : ℚ) * cost_per_bar > amount_paid ∨
   amount_paid - (max_chocolate_bars : ℚ) * cost_per_bar ≥ 1) :=
by sorry

end NUMINAMATH_CALUDE_max_chocolate_bars_correct_l1693_169337


namespace NUMINAMATH_CALUDE_cracked_seashells_l1693_169331

/-- The number of cracked seashells given the conditions from the problem -/
theorem cracked_seashells (mary_shells keith_shells total_shells : ℕ) 
  (h1 : mary_shells = 2)
  (h2 : keith_shells = 5)
  (h3 : total_shells = 7) :
  mary_shells + keith_shells - total_shells = 0 := by
  sorry

end NUMINAMATH_CALUDE_cracked_seashells_l1693_169331


namespace NUMINAMATH_CALUDE_square_adjustment_theorem_l1693_169364

theorem square_adjustment_theorem (a b : ℤ) (k : ℤ) : 
  (∃ (b : ℤ), b^2 = a^2 + 2*k ∨ b^2 = a^2 - 2*k) → 
  (∃ (c : ℤ), a^2 + k = c^2 + (b-a)^2 ∨ a^2 - k = c^2 + (b-a)^2) :=
sorry

end NUMINAMATH_CALUDE_square_adjustment_theorem_l1693_169364


namespace NUMINAMATH_CALUDE_project_estimated_hours_l1693_169390

/-- The number of extra hours Anie needs to work each day -/
def extra_hours : ℕ := 5

/-- The number of hours in Anie's normal work schedule each day -/
def normal_hours : ℕ := 10

/-- The number of days it would take Anie to finish the job -/
def days_to_finish : ℕ := 100

/-- The total number of hours Anie works each day -/
def total_hours_per_day : ℕ := normal_hours + extra_hours

/-- Theorem: The project is estimated to take 1500 hours -/
theorem project_estimated_hours : 
  days_to_finish * total_hours_per_day = 1500 := by
  sorry


end NUMINAMATH_CALUDE_project_estimated_hours_l1693_169390


namespace NUMINAMATH_CALUDE_three_chords_for_sixty_degrees_l1693_169335

/-- Represents a pair of concentric circles with chords drawn on the larger circle -/
structure ConcentricCirclesWithChords where
  /-- The measure of the angle formed by two adjacent chords at their intersection point -/
  chord_angle : ℝ
  /-- The number of chords needed to complete a full revolution -/
  num_chords : ℕ

/-- Theorem stating that for a 60° chord angle, 3 chords are needed to complete a revolution -/
theorem three_chords_for_sixty_degrees (circles : ConcentricCirclesWithChords) 
  (h : circles.chord_angle = 60) : circles.num_chords = 3 := by
  sorry

#check three_chords_for_sixty_degrees

end NUMINAMATH_CALUDE_three_chords_for_sixty_degrees_l1693_169335


namespace NUMINAMATH_CALUDE_min_value_of_function_l1693_169399

theorem min_value_of_function (x : ℝ) (h : x > 0) :
  (12 / x + 4 * x) ≥ 8 * Real.sqrt 3 ∧ ∃ y > 0, 12 / y + 4 * y = 8 * Real.sqrt 3 :=
by sorry

end NUMINAMATH_CALUDE_min_value_of_function_l1693_169399


namespace NUMINAMATH_CALUDE_arithmetic_equality_l1693_169393

theorem arithmetic_equality : 253 - 47 + 29 + 18 = 253 := by sorry

end NUMINAMATH_CALUDE_arithmetic_equality_l1693_169393


namespace NUMINAMATH_CALUDE_system_solution_difference_l1693_169378

theorem system_solution_difference (a b x y : ℝ) : 
  (2 * x + y = b) → 
  (x - b * y = a) → 
  (x = 1) → 
  (y = 0) → 
  (a - b = -1) := by
sorry

end NUMINAMATH_CALUDE_system_solution_difference_l1693_169378


namespace NUMINAMATH_CALUDE_function_properties_l1693_169386

def f (x : ℝ) := -5 * x + 1

theorem function_properties :
  (∃ (x₁ x₂ x₃ : ℝ), f x₁ > 0 ∧ x₁ > 0 ∧ f x₂ < 0 ∧ x₂ > 0 ∧ f x₃ < 0 ∧ x₃ < 0) ∧
  (∀ x : ℝ, x > 1 → f x < 0) :=
sorry

end NUMINAMATH_CALUDE_function_properties_l1693_169386


namespace NUMINAMATH_CALUDE_remainder_2024_3047_mod_800_l1693_169300

theorem remainder_2024_3047_mod_800 : (2024 * 3047) % 800 = 728 := by
  sorry

end NUMINAMATH_CALUDE_remainder_2024_3047_mod_800_l1693_169300


namespace NUMINAMATH_CALUDE_unique_number_l1693_169329

/-- A six-digit number with leftmost digit 7 -/
def SixDigitNumber (n : ℕ) : Prop :=
  100000 ≤ n ∧ n < 1000000 ∧ n / 100000 = 7

/-- Function to move the leftmost digit to the end -/
def moveLeftmostToEnd (n : ℕ) : ℕ :=
  (n % 100000) * 10 + (n / 100000)

/-- The main theorem -/
theorem unique_number : ∃! n : ℕ, SixDigitNumber n ∧ moveLeftmostToEnd n = n / 5 :=
  sorry

end NUMINAMATH_CALUDE_unique_number_l1693_169329


namespace NUMINAMATH_CALUDE_product_sum_bounds_l1693_169359

def X : Finset ℕ := Finset.range 11

theorem product_sum_bounds (A B : Finset ℕ) (hA : A ⊆ X) (hB : B ⊆ X) 
  (hAB : A ∪ B = X) (hAnonempty : A.Nonempty) (hBnonempty : B.Nonempty) :
  12636 ≤ (A.prod id) + (B.prod id) ∧ (A.prod id) + (B.prod id) ≤ 2 * Nat.factorial 11 := by
  sorry

end NUMINAMATH_CALUDE_product_sum_bounds_l1693_169359


namespace NUMINAMATH_CALUDE_cu2co3_2_weight_calculation_l1693_169365

-- Define the chemical equation coefficients
def cu_no3_2_coeff : ℚ := 2
def na2co3_coeff : ℚ := 3
def cu2co3_2_coeff : ℚ := 1

-- Define the available moles of reactants
def cu_no3_2_moles : ℚ := 1.85
def na2co3_moles : ℚ := 3.21

-- Define the molar mass of Cu2(CO3)2
def cu2co3_2_molar_mass : ℚ := 247.12

-- Define the function to calculate the limiting reactant
def limiting_reactant (cu_no3_2 : ℚ) (na2co3 : ℚ) : ℚ :=
  min (cu_no3_2 / cu_no3_2_coeff) (na2co3 / na2co3_coeff)

-- Define the function to calculate the moles of Cu2(CO3)2 produced
def cu2co3_2_produced (limiting : ℚ) : ℚ :=
  limiting * (cu2co3_2_coeff / cu_no3_2_coeff)

-- Define the function to calculate the weight of Cu2(CO3)2 produced
def cu2co3_2_weight (moles : ℚ) : ℚ :=
  moles * cu2co3_2_molar_mass

-- Theorem statement
theorem cu2co3_2_weight_calculation :
  cu2co3_2_weight (cu2co3_2_produced (limiting_reactant cu_no3_2_moles na2co3_moles)) = 228.586 := by
  sorry

end NUMINAMATH_CALUDE_cu2co3_2_weight_calculation_l1693_169365


namespace NUMINAMATH_CALUDE_kindergarten_tissue_problem_l1693_169305

theorem kindergarten_tissue_problem :
  ∀ (group1 : ℕ), 
    (group1 * 40 + 10 * 40 + 11 * 40 = 1200) → 
    group1 = 9 := by
  sorry

end NUMINAMATH_CALUDE_kindergarten_tissue_problem_l1693_169305


namespace NUMINAMATH_CALUDE_largest_office_number_l1693_169344

def house_number : List Nat := [9, 0, 2, 3, 4]

def office_number_sum : Nat := house_number.sum

def is_valid_office_number (n : List Nat) : Prop :=
  n.length = 10 ∧ n.sum = office_number_sum

def lexicographically_greater (a b : List Nat) : Prop :=
  ∃ i, (∀ j < i, a.get! j = b.get! j) ∧ a.get! i > b.get! i

theorem largest_office_number :
  ∃ (max : List Nat),
    is_valid_office_number max ∧
    (∀ n, is_valid_office_number n → lexicographically_greater max n ∨ max = n) ∧
    max = [9, 0, 5, 4, 0, 0, 0, 0, 0, 4] :=
sorry

end NUMINAMATH_CALUDE_largest_office_number_l1693_169344


namespace NUMINAMATH_CALUDE_device_records_720_instances_l1693_169346

/-- Represents the number of instances recorded by a device in one hour -/
def instances_recorded (seconds_per_record : ℕ) : ℕ :=
  (60 * 60) / seconds_per_record

/-- Theorem stating that a device recording every 5 seconds for one hour will record 720 instances -/
theorem device_records_720_instances :
  instances_recorded 5 = 720 := by
  sorry

end NUMINAMATH_CALUDE_device_records_720_instances_l1693_169346


namespace NUMINAMATH_CALUDE_contractor_fine_calculation_l1693_169333

/-- Calculates the fine per day of absence for a contractor --/
def calculate_fine (total_days : ℕ) (pay_rate : ℚ) (total_received : ℚ) (days_absent : ℕ) : ℚ :=
  let days_worked := total_days - days_absent
  let amount_earned := days_worked * pay_rate
  (amount_earned - total_received) / days_absent

/-- Proves that the fine per day of absence is 7.5 given the contract conditions --/
theorem contractor_fine_calculation :
  calculate_fine 30 25 685 2 = 7.5 := by
  sorry

end NUMINAMATH_CALUDE_contractor_fine_calculation_l1693_169333


namespace NUMINAMATH_CALUDE_sufficient_condition_not_necessary_condition_sufficient_but_not_necessary_l1693_169382

/-- Two lines are parallel if they have the same slope -/
def parallel (m1 n1 c1 m2 n2 c2 : ℝ) : Prop :=
  m1 * n2 = m2 * n1

/-- The condition that a = 1 is sufficient for the lines to be parallel -/
theorem sufficient_condition (a : ℝ) :
  a = 1 → parallel 1 a (-1) (2*a - 1) a (-2) := by sorry

/-- The condition that a = 1 is not necessary for the lines to be parallel -/
theorem not_necessary_condition :
  ∃ a : ℝ, a ≠ 1 ∧ parallel 1 a (-1) (2*a - 1) a (-2) := by sorry

/-- The main theorem stating that a = 1 is sufficient but not necessary -/
theorem sufficient_but_not_necessary :
  (∀ a : ℝ, a = 1 → parallel 1 a (-1) (2*a - 1) a (-2)) ∧
  (∃ a : ℝ, a ≠ 1 ∧ parallel 1 a (-1) (2*a - 1) a (-2)) := by sorry

end NUMINAMATH_CALUDE_sufficient_condition_not_necessary_condition_sufficient_but_not_necessary_l1693_169382


namespace NUMINAMATH_CALUDE_f_never_prime_l1693_169347

def f (n : ℕ+) : ℕ := n^4 + 100 * n^2 + 169

theorem f_never_prime : ∀ n : ℕ+, ¬ Nat.Prime (f n) := by
  sorry

end NUMINAMATH_CALUDE_f_never_prime_l1693_169347


namespace NUMINAMATH_CALUDE_polynomial_simplification_l1693_169367

theorem polynomial_simplification (x : ℝ) :
  (2 * x^2 + 5*x - 4) - (x^2 - 2*x + 1) + (3 * x^2 + 4*x - 7) = 4 * x^2 + 11*x - 12 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_simplification_l1693_169367


namespace NUMINAMATH_CALUDE_set_equality_implies_values_l1693_169363

def A (a b : ℝ) : Set ℝ := {2, a, b}
def B (b : ℝ) : Set ℝ := {0, 2, b^2 - 2}

theorem set_equality_implies_values (a b : ℝ) :
  A a b = B b → ((a = 0 ∧ b = -1) ∨ (a = -2 ∧ b = 0)) :=
by sorry

end NUMINAMATH_CALUDE_set_equality_implies_values_l1693_169363


namespace NUMINAMATH_CALUDE_complex_polynomial_solution_l1693_169373

theorem complex_polynomial_solution (c₀ c₁ c₂ c₃ c₄ a b : ℝ) :
  let z : ℂ := Complex.mk a b
  let i : ℂ := Complex.I
  let f : ℂ → ℂ := λ w => c₄ * w^4 + i * c₃ * w^3 + c₂ * w^2 + i * c₁ * w + c₀
  f z = 0 → f (Complex.mk (-a) b) = 0 := by sorry

end NUMINAMATH_CALUDE_complex_polynomial_solution_l1693_169373


namespace NUMINAMATH_CALUDE_brenda_spay_count_l1693_169351

/-- The number of cats Brenda needs to spay -/
def num_cats : ℕ := 7

/-- The number of dogs Brenda needs to spay -/
def num_dogs : ℕ := 2 * num_cats

/-- The total number of animals Brenda needs to spay -/
def total_animals : ℕ := num_cats + num_dogs

theorem brenda_spay_count : total_animals = 21 := by
  sorry

end NUMINAMATH_CALUDE_brenda_spay_count_l1693_169351


namespace NUMINAMATH_CALUDE_max_square_garden_area_l1693_169320

theorem max_square_garden_area (perimeter : ℕ) (side_length : ℕ) : 
  perimeter = 160 →
  4 * side_length = perimeter →
  ∀ s : ℕ, 4 * s ≤ perimeter → s ^ 2 ≤ side_length ^ 2 :=
by
  sorry

#check max_square_garden_area

end NUMINAMATH_CALUDE_max_square_garden_area_l1693_169320


namespace NUMINAMATH_CALUDE_fh_length_squared_value_l1693_169301

/-- Represents a parallelogram EFGH with specific properties -/
structure Parallelogram where
  /-- Area of the parallelogram -/
  area : ℝ
  /-- Length of JK, where J and K are projections of E and G onto FH -/
  jk_length : ℝ
  /-- Length of LM, where L and M are projections of F and H onto EG -/
  lm_length : ℝ
  /-- Assertion that EG is √2 times shorter than FH -/
  eg_fh_ratio : ℝ

/-- The square of the length of FH in the parallelogram -/
def fh_length_squared (p : Parallelogram) : ℝ := sorry

/-- Theorem stating the square of FH's length given specific conditions -/
theorem fh_length_squared_value (p : Parallelogram) 
  (h_area : p.area = 20)
  (h_jk : p.jk_length = 7)
  (h_lm : p.lm_length = 9)
  (h_ratio : p.eg_fh_ratio = Real.sqrt 2) :
  fh_length_squared p = 27.625 := by sorry

end NUMINAMATH_CALUDE_fh_length_squared_value_l1693_169301


namespace NUMINAMATH_CALUDE_general_formula_minimize_s_l1693_169348

-- Define the sequence and its sum
def s (n : ℕ) : ℤ := 2 * n^2 - 30 * n

-- Define the general term of the sequence
def a (n : ℕ) : ℤ := 4 * n - 32

-- Theorem 1: The general formula for a_n is 4n - 32
theorem general_formula : ∀ n : ℕ, a n = s n - s (n - 1) := by sorry

-- Theorem 2: s_n is minimized when n = 7 or n = 8
theorem minimize_s : ∃ n : ℕ, (n = 7 ∨ n = 8) ∧ ∀ m : ℕ, s n ≤ s m := by sorry

end NUMINAMATH_CALUDE_general_formula_minimize_s_l1693_169348


namespace NUMINAMATH_CALUDE_a_sufficient_not_necessary_l1693_169371

/-- The function f(x) = x³ + a -/
def f (a : ℝ) (x : ℝ) : ℝ := x^3 + a

/-- f is strictly increasing on ℝ -/
def strictly_increasing (f : ℝ → ℝ) : Prop :=
  ∀ x y, x < y → f x < f y

theorem a_sufficient_not_necessary :
  (∃ a : ℝ, a > 1 ∧ strictly_increasing (f a)) ∧
  (∃ a : ℝ, strictly_increasing (f a) ∧ ¬(a > 1)) :=
sorry

end NUMINAMATH_CALUDE_a_sufficient_not_necessary_l1693_169371


namespace NUMINAMATH_CALUDE_group_b_sample_size_l1693_169307

/-- Calculates the number of cities to be sampled from a group in stratified sampling -/
def stratified_sample_size (total_cities : ℕ) (group_cities : ℕ) (sample_size : ℕ) : ℕ :=
  (group_cities * sample_size) / total_cities

/-- Proves that the number of cities to be sampled from Group B is 4 -/
theorem group_b_sample_size :
  let total_cities : ℕ := 36
  let group_b_cities : ℕ := 12
  let sample_size : ℕ := 12
  stratified_sample_size total_cities group_b_cities sample_size = 4 := by
  sorry

#eval stratified_sample_size 36 12 12

end NUMINAMATH_CALUDE_group_b_sample_size_l1693_169307


namespace NUMINAMATH_CALUDE_committee_size_l1693_169391

theorem committee_size (n : ℕ) : 
  (n * (n - 1) = 42) → n = 7 := by
  sorry

#check committee_size

end NUMINAMATH_CALUDE_committee_size_l1693_169391


namespace NUMINAMATH_CALUDE_sum_of_expressions_l1693_169324

def replace_asterisks (n : ℕ) : ℕ := 2^(n-1)

theorem sum_of_expressions : 
  (replace_asterisks 6) = 32 :=
sorry

end NUMINAMATH_CALUDE_sum_of_expressions_l1693_169324


namespace NUMINAMATH_CALUDE_quartet_songs_theorem_l1693_169314

theorem quartet_songs_theorem (a b c d e : ℕ) 
  (h1 : (a + b + c + d + e) % 4 = 0)
  (h2 : e = 8)
  (h3 : a = 5)
  (h4 : b > 5 ∧ b < 8)
  (h5 : c > 5 ∧ c < 8)
  (h6 : d > 5 ∧ d < 8) :
  (a + b + c + d + e) / 4 = 8 := by
sorry

end NUMINAMATH_CALUDE_quartet_songs_theorem_l1693_169314


namespace NUMINAMATH_CALUDE_perfect_pairs_iff_even_l1693_169385

/-- A pair of integers (a, b) is perfect if ab + 1 is a perfect square. -/
def IsPerfectPair (a b : ℤ) : Prop :=
  ∃ k : ℤ, a * b + 1 = k ^ 2

/-- The set {1, ..., 2n} can be divided into n perfect pairs. -/
def CanDivideIntoPerfectPairs (n : ℕ) : Prop :=
  ∃ f : Fin n → Fin (2 * n) × Fin (2 * n),
    (∀ i : Fin n, IsPerfectPair (f i).1.val.succ (f i).2.val.succ) ∧
    (∀ i j : Fin n, i ≠ j → (f i).1 ≠ (f j).1 ∧ (f i).1 ≠ (f j).2 ∧ 
                            (f i).2 ≠ (f j).1 ∧ (f i).2 ≠ (f j).2)

/-- The main theorem: The set {1, ..., 2n} can be divided into n perfect pairs 
    if and only if n is even. -/
theorem perfect_pairs_iff_even (n : ℕ) :
  CanDivideIntoPerfectPairs n ↔ Even n :=
sorry

end NUMINAMATH_CALUDE_perfect_pairs_iff_even_l1693_169385


namespace NUMINAMATH_CALUDE_quadratic_roots_range_l1693_169358

/-- Given a quadratic equation x^2 - x + 1 - m = 0 with two real roots α and β 
    satisfying |α| + |β| ≤ 5, the range of m is [3/4, 7]. -/
theorem quadratic_roots_range (m : ℝ) (α β : ℝ) : 
  (∀ x, x^2 - x + 1 - m = 0 ↔ x = α ∨ x = β) →
  (|α| + |β| ≤ 5) →
  (3/4 ≤ m ∧ m ≤ 7) :=
sorry

end NUMINAMATH_CALUDE_quadratic_roots_range_l1693_169358


namespace NUMINAMATH_CALUDE_symmetry_of_point_l1693_169330

/-- A point in a 2D plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- Symmetry with respect to the origin -/
def symmetricToOrigin (p q : Point) : Prop :=
  q.x = -p.x ∧ q.y = -p.y

/-- The theorem statement -/
theorem symmetry_of_point :
  let A : Point := ⟨3, -2⟩
  let A' : Point := ⟨-3, 2⟩
  symmetricToOrigin A A' := by sorry

end NUMINAMATH_CALUDE_symmetry_of_point_l1693_169330


namespace NUMINAMATH_CALUDE_circle_equation_specific_l1693_169370

/-- The standard equation of a circle with center (h, k) and radius r -/
def circle_equation (x y h k r : ℝ) : Prop :=
  (x - h)^2 + (y - k)^2 = r^2

/-- Theorem: The standard equation of a circle with center (2, -1) and radius 3 -/
theorem circle_equation_specific : ∀ x y : ℝ,
  circle_equation x y 2 (-1) 3 ↔ (x - 2)^2 + (y + 1)^2 = 9 := by
  sorry

end NUMINAMATH_CALUDE_circle_equation_specific_l1693_169370


namespace NUMINAMATH_CALUDE_sum_of_x_and_y_l1693_169303

theorem sum_of_x_and_y (x y : ℝ) (h : x^2 + y^2 = 18*x - 10*y + 22) :
  x + y = 4 + 2 * Real.sqrt 42 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_x_and_y_l1693_169303


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_l1693_169361

def vector_a (x : ℝ) : ℝ × ℝ := (6, x)

theorem sufficient_not_necessary :
  ∃ (x : ℝ), (x ≠ 8 ∧ ‖vector_a x‖ = 10) ∧
  ∀ (x : ℝ), (x = 8 → ‖vector_a x‖ = 10) :=
sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_l1693_169361


namespace NUMINAMATH_CALUDE_increasing_condition_l1693_169389

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := x^2 + 2*(a-2)*x + 5

-- State the theorem
theorem increasing_condition (a : ℝ) :
  (∀ x₁ x₂ : ℝ, 4 < x₁ ∧ x₁ < x₂ → f a x₁ < f a x₂) ↔ a ≥ -2 :=
by sorry

end NUMINAMATH_CALUDE_increasing_condition_l1693_169389


namespace NUMINAMATH_CALUDE_ada_original_seat_l1693_169342

-- Define the seat numbers
inductive Seat
| one
| two
| three
| four
| five

-- Define the friends
inductive Friend
| Ada
| Bea
| Ceci
| Dee
| Edie

-- Define the seating arrangement as a function from Friend to Seat
def Seating := Friend → Seat

-- Define the movement function
def move (s : Seat) (n : Int) : Seat :=
  match s, n with
  | Seat.one, 1 => Seat.two
  | Seat.one, 2 => Seat.three
  | Seat.two, -1 => Seat.one
  | Seat.two, 1 => Seat.three
  | Seat.two, 2 => Seat.four
  | Seat.three, -1 => Seat.two
  | Seat.three, 1 => Seat.four
  | Seat.three, 2 => Seat.five
  | Seat.four, -1 => Seat.three
  | Seat.four, 1 => Seat.five
  | Seat.five, -1 => Seat.four
  | _, _ => s  -- Default case: no movement

-- Define the theorem
theorem ada_original_seat (initial_seating final_seating : Seating) :
  (∀ f : Friend, f ≠ Friend.Ada → 
    (f = Friend.Bea → move (initial_seating f) 2 = final_seating f) ∧
    (f = Friend.Ceci → move (initial_seating f) (-1) = final_seating f) ∧
    ((f = Friend.Dee ∨ f = Friend.Edie) → 
      (initial_seating Friend.Dee = final_seating Friend.Edie ∧
       initial_seating Friend.Edie = final_seating Friend.Dee))) →
  (final_seating Friend.Ada = Seat.one ∨ final_seating Friend.Ada = Seat.five) →
  initial_seating Friend.Ada = Seat.two :=
sorry

end NUMINAMATH_CALUDE_ada_original_seat_l1693_169342


namespace NUMINAMATH_CALUDE_semicircle_area_problem_l1693_169341

/-- The area of the shaded region in the semicircle problem -/
theorem semicircle_area_problem (A B C D E F G : ℝ) : 
  A < B ∧ B < C ∧ C < D ∧ D < E ∧ E < F ∧ F < G →
  B - A = 3 →
  C - B = 3 →
  D - C = 3 →
  E - D = 3 →
  F - E = 3 →
  G - F = 6 →
  let semicircle_area (d : ℝ) := π * d^2 / 8
  let total_small_area := semicircle_area (B - A) + semicircle_area (C - B) + 
                          semicircle_area (D - C) + semicircle_area (E - D) + 
                          semicircle_area (F - E) + semicircle_area (G - F)
  let large_semicircle_area := semicircle_area (G - A)
  large_semicircle_area - total_small_area = 225 * π / 8 := by
  sorry

end NUMINAMATH_CALUDE_semicircle_area_problem_l1693_169341


namespace NUMINAMATH_CALUDE_zero_function_theorem_l1693_169379

theorem zero_function_theorem (f : ℝ → ℝ) (hf : Differentiable ℝ f) 
  (h1 : ∀ x : ℤ, deriv f x = 0)
  (h2 : ∀ x : ℝ, deriv f x = 0 → f x = 0) :
  ∀ x : ℝ, f x = 0 := by
sorry

end NUMINAMATH_CALUDE_zero_function_theorem_l1693_169379


namespace NUMINAMATH_CALUDE_number_of_amateurs_l1693_169374

/-- The number of chess amateurs in the tournament -/
def n : ℕ := sorry

/-- The number of other amateurs each amateur plays with -/
def games_per_amateur : ℕ := 4

/-- The total number of possible games in the tournament -/
def total_games : ℕ := 10

/-- Theorem stating the number of chess amateurs in the tournament -/
theorem number_of_amateurs :
  n = 5 ∧
  games_per_amateur = 4 ∧
  total_games = 10 ∧
  n.choose 2 = total_games :=
sorry

end NUMINAMATH_CALUDE_number_of_amateurs_l1693_169374


namespace NUMINAMATH_CALUDE_splittable_point_range_l1693_169325

/-- A function f is splittable at x_0 if f(x_0 + 1) = f(x_0) + f(1) -/
def IsSplittable (f : ℝ → ℝ) (x_0 : ℝ) : Prop :=
  f (x_0 + 1) = f x_0 + f 1

/-- The logarithm function with base 5 -/
noncomputable def log5 (x : ℝ) : ℝ := Real.log x / Real.log 5

/-- The function f(x) = log_5(a / (2^x + 1)) -/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  log5 (a / (2^x + 1))

theorem splittable_point_range (a : ℝ) :
  (a > 0) → (∃ x_0 : ℝ, IsSplittable (f a) x_0) ↔ (3/2 < a ∧ a < 3) := by
  sorry


end NUMINAMATH_CALUDE_splittable_point_range_l1693_169325


namespace NUMINAMATH_CALUDE_abs_two_implies_plus_minus_two_l1693_169336

theorem abs_two_implies_plus_minus_two (a : ℝ) : |a| = 2 → a = 2 ∨ a = -2 := by
  sorry

end NUMINAMATH_CALUDE_abs_two_implies_plus_minus_two_l1693_169336


namespace NUMINAMATH_CALUDE_least_positive_t_for_geometric_progression_l1693_169353

open Real

theorem least_positive_t_for_geometric_progression : 
  ∃ (t : ℝ), t > 0 ∧ 
  (∀ (α : ℝ), 0 < α → α < π/2 → 
    (∃ (r : ℝ), r > 0 ∧
      arcsin (sin α) * r = arcsin (sin (2*α)) ∧
      arcsin (sin (2*α)) * r = arcsin (sin (5*α)) ∧
      arcsin (sin (5*α)) * r = arcsin (sin (t*α)))) ∧
  (∀ (t' : ℝ), 0 < t' → t' < t →
    ¬(∀ (α : ℝ), 0 < α → α < π/2 → 
      (∃ (r : ℝ), r > 0 ∧
        arcsin (sin α) * r = arcsin (sin (2*α)) ∧
        arcsin (sin (2*α)) * r = arcsin (sin (5*α)) ∧
        arcsin (sin (5*α)) * r = arcsin (sin (t'*α))))) ∧
  t = 8 :=
by sorry

end NUMINAMATH_CALUDE_least_positive_t_for_geometric_progression_l1693_169353


namespace NUMINAMATH_CALUDE_negation_of_forall_positive_negation_of_greater_than_zero_l1693_169375

theorem negation_of_forall_positive (P : ℝ → Prop) :
  (¬ ∀ x : ℝ, P x) ↔ (∃ x : ℝ, ¬ P x) :=
by sorry

theorem negation_of_greater_than_zero :
  (¬ ∀ x : ℝ, x^2 + x + 1 > 0) ↔ (∃ x : ℝ, x^2 + x + 1 ≤ 0) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_forall_positive_negation_of_greater_than_zero_l1693_169375


namespace NUMINAMATH_CALUDE_max_triangle_area_l1693_169310

-- Define the circle C
def C : Set (ℝ × ℝ) := {p | (p.1^2 + (p.2 - 2)^2 = 4)}

-- Define the line l
def l : Set (ℝ × ℝ) := {p | p.1 - Real.sqrt 3 * p.2 + Real.sqrt 3 = 0}

-- Define the intersection points A and B
def A : ℝ × ℝ := sorry
def B : ℝ × ℝ := sorry

-- Define a function to calculate the area of a triangle given three points
def triangleArea (p1 p2 p3 : ℝ × ℝ) : ℝ := sorry

-- Theorem statement
theorem max_triangle_area :
  ∀ P ∈ C, P ≠ A → P ≠ B →
  triangleArea P A B ≤ (4 * Real.sqrt 13 + Real.sqrt 39) / 4 :=
sorry

end NUMINAMATH_CALUDE_max_triangle_area_l1693_169310


namespace NUMINAMATH_CALUDE_angle_sum_around_point_l1693_169316

theorem angle_sum_around_point (y : ℝ) : 
  6 * y + 7 * y + 3 * y + 2 * y = 360 → y = 20 := by
  sorry

end NUMINAMATH_CALUDE_angle_sum_around_point_l1693_169316


namespace NUMINAMATH_CALUDE_omega_sum_simplification_l1693_169394

theorem omega_sum_simplification (ω : ℂ) (h1 : ω^8 = 1) (h2 : ω ≠ 1) :
  ω^17 + ω^21 + ω^25 + ω^29 + ω^33 + ω^37 + ω^41 + ω^45 + ω^49 + ω^53 + ω^57 + ω^61 + ω^65 = ω :=
by sorry

end NUMINAMATH_CALUDE_omega_sum_simplification_l1693_169394


namespace NUMINAMATH_CALUDE_sum_of_squares_on_sides_l1693_169312

/-- Given a triangle XYZ with side XZ = 12 units and perpendicular height from Y to XZ being 5 units,
    the sum of the areas of squares on sides XY and YZ is 122 square units. -/
theorem sum_of_squares_on_sides (X Y Z : ℝ × ℝ) : 
  let XZ : ℝ := Real.sqrt ((X.1 - Z.1)^2 + (X.2 - Z.2)^2)
  let height : ℝ := 5
  let XY : ℝ := Real.sqrt ((X.1 - Y.1)^2 + (X.2 - Y.2)^2)
  let YZ : ℝ := Real.sqrt ((Y.1 - Z.1)^2 + (Y.2 - Z.2)^2)
  XZ = 12 →
  (∃ D : ℝ × ℝ, (D.1 - X.1) * (Z.2 - X.2) = (Z.1 - X.1) * (D.2 - X.2) ∧ 
                (Y.1 - D.1) * (Z.1 - X.1) = (X.2 - D.2) * (Z.2 - X.2) ∧
                Real.sqrt ((Y.1 - D.1)^2 + (Y.2 - D.2)^2) = height) →
  XY^2 + YZ^2 = 122 :=
by sorry

end NUMINAMATH_CALUDE_sum_of_squares_on_sides_l1693_169312


namespace NUMINAMATH_CALUDE_work_completion_time_l1693_169315

theorem work_completion_time (a b c : ℝ) : 
  (a > 0) →
  (b > 0) →
  (c > 0) →
  (a = 1/6) →
  (a + b + c = 1/3) →
  (c = 1/8 * (a + b)) →
  (b = 1/28) :=
sorry

end NUMINAMATH_CALUDE_work_completion_time_l1693_169315


namespace NUMINAMATH_CALUDE_tangent_sphere_radius_l1693_169381

/-- A truncated cone with a sphere tangent to its surfaces -/
structure TruncatedConeWithSphere where
  bottom_radius : ℝ
  top_radius : ℝ
  slant_height : ℝ
  sphere_radius : ℝ

/-- The sphere is tangent to the top, bottom, and lateral surface of the truncated cone -/
def is_tangent_sphere (cone : TruncatedConeWithSphere) : Prop :=
  cone.sphere_radius > 0 ∧
  cone.sphere_radius ≤ cone.bottom_radius ∧
  cone.sphere_radius ≤ cone.top_radius ∧
  cone.sphere_radius ≤ cone.slant_height

/-- The theorem stating the radius of the tangent sphere -/
theorem tangent_sphere_radius (cone : TruncatedConeWithSphere) 
  (h1 : cone.bottom_radius = 20)
  (h2 : cone.top_radius = 5)
  (h3 : cone.slant_height = 25)
  (h4 : is_tangent_sphere cone) :
  cone.sphere_radius = 10 := by
  sorry

end NUMINAMATH_CALUDE_tangent_sphere_radius_l1693_169381


namespace NUMINAMATH_CALUDE_number_order_l1693_169349

/-- Convert a number from base b to decimal --/
def to_decimal (digits : List Nat) (b : Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * b^i) 0

/-- Definition of 85 in base 9 --/
def num_85_base9 : Nat := to_decimal [5, 8] 9

/-- Definition of 210 in base 6 --/
def num_210_base6 : Nat := to_decimal [0, 1, 2] 6

/-- Definition of 1000 in base 4 --/
def num_1000_base4 : Nat := to_decimal [0, 0, 0, 1] 4

/-- Definition of 111111 in base 2 --/
def num_111111_base2 : Nat := to_decimal [1, 1, 1, 1, 1, 1] 2

/-- Theorem stating the order of the numbers --/
theorem number_order :
  num_210_base6 > num_85_base9 ∧
  num_85_base9 > num_1000_base4 ∧
  num_1000_base4 > num_111111_base2 :=
by sorry

end NUMINAMATH_CALUDE_number_order_l1693_169349


namespace NUMINAMATH_CALUDE_comparison_sqrt_l1693_169343

theorem comparison_sqrt : 3 * Real.sqrt 2 > Real.sqrt 13 := by
  sorry

end NUMINAMATH_CALUDE_comparison_sqrt_l1693_169343


namespace NUMINAMATH_CALUDE_palindrome_expansion_existence_l1693_169350

theorem palindrome_expansion_existence (x y k : ℕ+) : 
  ∃ (N : ℕ+) (b : Fin (k + 1) → ℕ+),
    (∀ i : Fin (k + 1), ∃ (a c : ℕ), 
      N = a * (b i)^2 + c * (b i) + a ∧ 
      a < b i ∧ 
      c < b i) ∧
    (∃ (B : ℕ+), 
      N = x * (B^2 + 1) + y * B ∧ 
      b 0 = B) :=
sorry

end NUMINAMATH_CALUDE_palindrome_expansion_existence_l1693_169350


namespace NUMINAMATH_CALUDE_grape_rate_calculation_l1693_169340

/-- The rate per kg for grapes -/
def grape_rate : ℝ := 70

/-- The amount of grapes purchased in kg -/
def grape_amount : ℝ := 3

/-- The rate per kg for mangoes -/
def mango_rate : ℝ := 55

/-- The amount of mangoes purchased in kg -/
def mango_amount : ℝ := 9

/-- The total amount paid -/
def total_paid : ℝ := 705

theorem grape_rate_calculation :
  grape_rate * grape_amount + mango_rate * mango_amount = total_paid :=
by sorry

end NUMINAMATH_CALUDE_grape_rate_calculation_l1693_169340


namespace NUMINAMATH_CALUDE_square_product_extension_l1693_169345

theorem square_product_extension (a b : ℕ) 
  (h1 : ∃ x : ℕ, a * b = x ^ 2)
  (h2 : ∃ y : ℕ, (a + 1) * (b + 1) = y ^ 2) :
  ∃ n : ℕ, n > 1 ∧ ∃ z : ℕ, (a + n) * (b + n) = z ^ 2 := by
  sorry

end NUMINAMATH_CALUDE_square_product_extension_l1693_169345


namespace NUMINAMATH_CALUDE_min_balls_correct_l1693_169384

/-- The minimum number of balls that satisfies the given conditions -/
def min_balls : ℕ := 24

/-- The number of white balls -/
def white_balls : ℕ := min_balls / 3

/-- The number of black balls -/
def black_balls : ℕ := 2 * white_balls

/-- The number of pairs of different colors -/
def different_color_pairs : ℕ := min_balls / 4

/-- The number of pairs of the same color -/
def same_color_pairs : ℕ := 3 * different_color_pairs

theorem min_balls_correct :
  (black_balls = 2 * white_balls) ∧
  (black_balls + white_balls = min_balls) ∧
  (same_color_pairs = 3 * different_color_pairs) ∧
  (same_color_pairs + different_color_pairs = min_balls) ∧
  (∀ n : ℕ, n < min_balls → ¬(
    (2 * (n / 3) = n - (n / 3)) ∧
    (3 * (n / 4) = n - (n / 4))
  )) := by
  sorry

#eval min_balls

end NUMINAMATH_CALUDE_min_balls_correct_l1693_169384


namespace NUMINAMATH_CALUDE_coordinates_wrt_origin_l1693_169323

/-- The coordinates of a point with respect to the origin are the same as its given coordinates. -/
theorem coordinates_wrt_origin (x y : ℝ) : 
  let A : ℝ × ℝ := (x, y)
  A = A :=
by sorry

end NUMINAMATH_CALUDE_coordinates_wrt_origin_l1693_169323


namespace NUMINAMATH_CALUDE_paradise_park_ferris_wheel_seats_l1693_169334

/-- The number of seats on a Ferris wheel -/
def ferris_wheel_seats (total_people : ℕ) (people_per_seat : ℕ) : ℕ :=
  total_people / people_per_seat

/-- Theorem: The Ferris wheel in paradise park has 4 seats -/
theorem paradise_park_ferris_wheel_seats :
  ferris_wheel_seats 20 5 = 4 := by
  sorry

end NUMINAMATH_CALUDE_paradise_park_ferris_wheel_seats_l1693_169334


namespace NUMINAMATH_CALUDE_base_10_to_9_conversion_l1693_169318

-- Define a custom type for base-9 digits
inductive Base9Digit
| D0 | D1 | D2 | D3 | D4 | D5 | D6 | D7 | D8 | H

def base9ToNat : List Base9Digit → Nat
  | [] => 0
  | d::ds => 
    match d with
    | Base9Digit.D0 => 0 + 9 * base9ToNat ds
    | Base9Digit.D1 => 1 + 9 * base9ToNat ds
    | Base9Digit.D2 => 2 + 9 * base9ToNat ds
    | Base9Digit.D3 => 3 + 9 * base9ToNat ds
    | Base9Digit.D4 => 4 + 9 * base9ToNat ds
    | Base9Digit.D5 => 5 + 9 * base9ToNat ds
    | Base9Digit.D6 => 6 + 9 * base9ToNat ds
    | Base9Digit.D7 => 7 + 9 * base9ToNat ds
    | Base9Digit.D8 => 8 + 9 * base9ToNat ds
    | Base9Digit.H => 8 + 9 * base9ToNat ds

theorem base_10_to_9_conversion :
  base9ToNat [Base9Digit.D3, Base9Digit.D1, Base9Digit.D4] = 256 := by
  sorry

end NUMINAMATH_CALUDE_base_10_to_9_conversion_l1693_169318


namespace NUMINAMATH_CALUDE_barney_average_speed_l1693_169357

def initial_reading : ℕ := 2332
def final_reading : ℕ := 2772
def total_time : ℕ := 12

def distance : ℕ := final_reading - initial_reading

def average_speed : ℚ := distance / total_time

theorem barney_average_speed : 
  initial_reading = 2332 → 
  final_reading = 2772 → 
  total_time = 12 → 
  ⌊average_speed⌋ = 36 := by sorry

end NUMINAMATH_CALUDE_barney_average_speed_l1693_169357


namespace NUMINAMATH_CALUDE_total_hockey_games_l1693_169327

theorem total_hockey_games (attended : ℕ) (missed : ℕ) 
  (h1 : attended = 13) (h2 : missed = 18) : 
  attended + missed = 31 := by
  sorry

end NUMINAMATH_CALUDE_total_hockey_games_l1693_169327


namespace NUMINAMATH_CALUDE_f_decreasing_interval_l1693_169368

/-- A quadratic function with specific properties -/
def f : ℝ → ℝ := sorry

/-- The properties of the quadratic function -/
axiom f_prop1 : f (-1) = 0
axiom f_prop2 : f 4 = 0
axiom f_prop3 : f 0 = 4

/-- The decreasing interval of f -/
def decreasing_interval : Set ℝ := {x | x ≥ 3/2}

/-- Theorem stating that the given set is the decreasing interval of f -/
theorem f_decreasing_interval : 
  ∀ x ∈ decreasing_interval, ∀ y ∈ decreasing_interval, x < y → f x > f y :=
sorry

end NUMINAMATH_CALUDE_f_decreasing_interval_l1693_169368


namespace NUMINAMATH_CALUDE_nonCongruentTrianglesCount_l1693_169306

-- Define the grid
def Grid := Fin 3 → Fin 3 → ℝ × ℝ

-- Define the grid with 0.5 unit spacing
def standardGrid : Grid :=
  λ i j => (0.5 * i.val, 0.5 * j.val)

-- Define a triangle as a tuple of three points
def Triangle := (ℝ × ℝ) × (ℝ × ℝ) × (ℝ × ℝ)

-- Define congruence for triangles
def areCongruent (t1 t2 : Triangle) : Prop := sorry

-- Define a function to generate all possible triangles from the grid
def allTriangles (g : Grid) : List Triangle := sorry

-- Define a function to count non-congruent triangles
def countNonCongruentTriangles (triangles : List Triangle) : Nat := sorry

-- The main theorem
theorem nonCongruentTrianglesCount :
  countNonCongruentTriangles (allTriangles standardGrid) = 9 := by sorry

end NUMINAMATH_CALUDE_nonCongruentTrianglesCount_l1693_169306


namespace NUMINAMATH_CALUDE_parabola_equations_l1693_169302

/-- Parabola with x-axis symmetry -/
def parabola_x_axis (m : ℝ) (x y : ℝ) : Prop :=
  y^2 = m * x

/-- Parabola with y-axis symmetry -/
def parabola_y_axis (p : ℝ) (x y : ℝ) : Prop :=
  x^2 = 4 * p * y

theorem parabola_equations :
  (∃ m : ℝ, m ≠ 0 ∧ parabola_x_axis m 6 (-3)) ∧
  (∃ p : ℝ, p > 0 ∧ parabola_y_axis p x y ↔ x^2 = 12 * y ∨ x^2 = -12 * y) :=
by sorry

end NUMINAMATH_CALUDE_parabola_equations_l1693_169302


namespace NUMINAMATH_CALUDE_expression_simplification_l1693_169356

theorem expression_simplification (a b c : ℕ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  let num := a^2 * (1/b - 1/c) + b^2 * (1/c - 1/a) + c^2 * (1/a - 1/b)
  let den := a * (1/b - 1/c) + b * (1/c - 1/a) + c * (1/a - 1/b)
  num / den = a + b + c := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l1693_169356


namespace NUMINAMATH_CALUDE_cube_root_of_y_fourth_root_of_y_to_six_l1693_169355

theorem cube_root_of_y_fourth_root_of_y_to_six (y : ℝ) :
  (y * (y^6)^(1/4))^(1/3) = 5 → y = 5^(6/5) := by
  sorry

end NUMINAMATH_CALUDE_cube_root_of_y_fourth_root_of_y_to_six_l1693_169355


namespace NUMINAMATH_CALUDE_inequality_equivalence_l1693_169322

theorem inequality_equivalence (x : ℝ) : (3 + x) * (2 - x) < 0 ↔ x > 2 ∨ x < -3 := by sorry

end NUMINAMATH_CALUDE_inequality_equivalence_l1693_169322


namespace NUMINAMATH_CALUDE_remainder_of_1234567_divided_by_257_l1693_169328

theorem remainder_of_1234567_divided_by_257 : 
  1234567 % 257 = 774 := by sorry

end NUMINAMATH_CALUDE_remainder_of_1234567_divided_by_257_l1693_169328


namespace NUMINAMATH_CALUDE_prime_iff_binomial_congruence_l1693_169377

theorem prime_iff_binomial_congruence (n : ℕ) (hn : n > 0) :
  Nat.Prime n ↔ ∀ k : ℕ, k < n → (Nat.choose (n - 1) k) % n = ((-1 : ℤ) ^ k).toNat % n :=
sorry

end NUMINAMATH_CALUDE_prime_iff_binomial_congruence_l1693_169377


namespace NUMINAMATH_CALUDE_three_digit_divisible_by_17_l1693_169395

theorem three_digit_divisible_by_17 : 
  (Finset.filter (fun k => 100 ≤ 17 * k ∧ 17 * k ≤ 999) (Finset.range 1000)).card = 53 := by
  sorry

end NUMINAMATH_CALUDE_three_digit_divisible_by_17_l1693_169395


namespace NUMINAMATH_CALUDE_triangle_side_length_l1693_169352

theorem triangle_side_length 
  (A B C : Real) -- Angles of the triangle
  (a b c : Real) -- Side lengths of the triangle
  (h1 : 0 < a ∧ 0 < b ∧ 0 < c) -- Side lengths are positive
  (h2 : B = Real.pi / 3) -- B = 60°
  (h3 : (1/2) * a * c * Real.sin B = Real.sqrt 3) -- Area of the triangle is √3
  (h4 : a^2 + c^2 = 3*a*c) -- Given equation
  : b = 2 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_triangle_side_length_l1693_169352


namespace NUMINAMATH_CALUDE_bisection_solves_x_squared_minus_two_program_flowchart_l1693_169321

/-- Represents different types of flowcharts -/
inductive FlowchartType
  | Process
  | Program
  | KnowledgeStructure
  | OrganizationalStructure

/-- Represents a method for solving equations -/
inductive SolvingMethod
  | Bisection
  | Newton
  | Secant

/-- Represents an equation to be solved -/
structure Equation where
  f : ℝ → ℝ

/-- Determines the type of flowchart used to solve an equation using a specific method -/
def flowchartTypeForSolving (eq : Equation) (method : SolvingMethod) : FlowchartType :=
  sorry

/-- The theorem stating that solving x^2 - 2 = 0 using the bisection method results in a program flowchart -/
theorem bisection_solves_x_squared_minus_two_program_flowchart :
  flowchartTypeForSolving { f := fun x => x^2 - 2 } SolvingMethod.Bisection = FlowchartType.Program :=
  sorry

end NUMINAMATH_CALUDE_bisection_solves_x_squared_minus_two_program_flowchart_l1693_169321


namespace NUMINAMATH_CALUDE_area_between_tangent_circles_l1693_169366

/-- The area of the region between two tangent circles -/
theorem area_between_tangent_circles (r₁ r₂ : ℝ) (h₁ : r₁ > 0) (h₂ : r₂ > 0) (h₃ : r₂ = 3 * r₁) :
  π * r₂^2 - π * r₁^2 = 32 * π * r₁^2 :=
sorry

end NUMINAMATH_CALUDE_area_between_tangent_circles_l1693_169366


namespace NUMINAMATH_CALUDE_aaron_sheep_count_l1693_169313

theorem aaron_sheep_count (beth_sheep : ℕ) (aaron_sheep : ℕ) : 
  aaron_sheep = 7 * beth_sheep →
  aaron_sheep + beth_sheep = 608 →
  aaron_sheep = 532 := by
sorry

end NUMINAMATH_CALUDE_aaron_sheep_count_l1693_169313


namespace NUMINAMATH_CALUDE_rectangle_dimension_change_l1693_169372

theorem rectangle_dimension_change (L B : ℝ) (L' B' : ℝ) (h_positive : L > 0 ∧ B > 0) :
  B' = 1.15 * B →
  L' * B' = 1.2075 * (L * B) →
  L' = 1.05 * L := by
sorry

end NUMINAMATH_CALUDE_rectangle_dimension_change_l1693_169372


namespace NUMINAMATH_CALUDE_bob_dogs_count_l1693_169309

/-- Represents the number of cats Bob has -/
def num_cats : ℕ := 4

/-- Represents the portion of the food bag a single cat receives -/
def cat_portion : ℚ := 125 / 4000

/-- Represents the portion of the food bag a single dog receives -/
def dog_portion : ℚ := num_cats * cat_portion

/-- The number of dogs Bob has -/
def num_dogs : ℕ := 7

theorem bob_dogs_count :
  (num_dogs : ℚ) * dog_portion + (num_cats : ℚ) * cat_portion = 1 :=
sorry

#check bob_dogs_count

end NUMINAMATH_CALUDE_bob_dogs_count_l1693_169309


namespace NUMINAMATH_CALUDE_sine_shift_and_stretch_l1693_169397

/-- Given a function f(x) = sin(x), prove that shifting it right by π/10 units
    and then stretching the x-coordinates by a factor of 2 results in
    the function g(x) = sin(1/2x - π/10) -/
theorem sine_shift_and_stretch (x : ℝ) :
  let f : ℝ → ℝ := λ t ↦ Real.sin t
  let shift : ℝ → ℝ := λ t ↦ t - π / 10
  let stretch : ℝ → ℝ := λ t ↦ t / 2
  let g : ℝ → ℝ := λ t ↦ Real.sin (1/2 * t - π / 10)
  f (stretch (shift x)) = g x := by
  sorry

end NUMINAMATH_CALUDE_sine_shift_and_stretch_l1693_169397


namespace NUMINAMATH_CALUDE_expected_hearts_in_modified_deck_l1693_169326

/-- A circular arrangement of cards -/
structure CircularDeck :=
  (total : ℕ)
  (hearts : ℕ)
  (h_total : total ≥ hearts)

/-- Expected number of adjacent heart pairs in a circular arrangement -/
def expected_adjacent_hearts (deck : CircularDeck) : ℚ :=
  (deck.hearts : ℚ) * (deck.hearts - 1) / (deck.total - 1)

theorem expected_hearts_in_modified_deck :
  let deck := CircularDeck.mk 40 10 (by norm_num)
  expected_adjacent_hearts deck = 30 / 13 := by
sorry

end NUMINAMATH_CALUDE_expected_hearts_in_modified_deck_l1693_169326


namespace NUMINAMATH_CALUDE_parabola_equation_l1693_169308

/-- A parabola with focus on the x-axis, vertex at the origin, and opening to the right -/
structure RightParabola where
  p : ℝ
  eq : ℝ → ℝ → Prop
  h1 : ∀ x y, eq x y ↔ y^2 = 2*p*x
  h2 : p > 0

/-- The point (1, 2) lies on the parabola -/
def PassesThroughPoint (par : RightParabola) : Prop :=
  par.eq 1 2

theorem parabola_equation (par : RightParabola) (h : PassesThroughPoint par) :
  par.p = 2 ∧ ∀ x y, par.eq x y ↔ y^2 = 4*x := by
  sorry

end NUMINAMATH_CALUDE_parabola_equation_l1693_169308


namespace NUMINAMATH_CALUDE_cookies_left_for_birthday_l1693_169396

theorem cookies_left_for_birthday 
  (pans : ℕ) 
  (cookies_per_pan : ℕ) 
  (eaten_cookies : ℕ) 
  (burnt_cookies : ℕ) 
  (h1 : pans = 12)
  (h2 : cookies_per_pan = 15)
  (h3 : eaten_cookies = 9)
  (h4 : burnt_cookies = 6) :
  (pans * cookies_per_pan) - (eaten_cookies + burnt_cookies) = 165 := by
  sorry

end NUMINAMATH_CALUDE_cookies_left_for_birthday_l1693_169396


namespace NUMINAMATH_CALUDE_smallest_integer_l1693_169388

theorem smallest_integer (a b : ℕ) (ha : a = 60) (h_lcm_gcd : Nat.lcm a b / Nat.gcd a b = 60) : 
  b ≥ 60 ∧ ∀ c : ℕ, c < 60 → Nat.lcm a c / Nat.gcd a c ≠ 60 := by
  sorry

end NUMINAMATH_CALUDE_smallest_integer_l1693_169388


namespace NUMINAMATH_CALUDE_simplify_and_evaluate_evaluate_at_two_l1693_169376

theorem simplify_and_evaluate (x : ℝ) (h1 : x ≠ -1) (h2 : x ≠ 0) :
  (1 - 1 / (x + 1)) / (x / (x^2 + 2*x + 1)) = x + 1 := by
  sorry

-- Evaluation for x = 2
theorem evaluate_at_two :
  (1 - 1 / (2 + 1)) / (2 / (2^2 + 2*2 + 1)) = 3 := by
  sorry

end NUMINAMATH_CALUDE_simplify_and_evaluate_evaluate_at_two_l1693_169376
