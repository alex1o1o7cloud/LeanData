import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_two_longest_altitudes_eq_21_l1324_132440

/-- A triangle with sides 9, 12, and 15 -/
structure RightTriangle where
  a : ℝ
  b : ℝ
  c : ℝ
  h_a : a = 9
  h_b : b = 12
  h_c : c = 15
  h_right : a^2 + b^2 = c^2

/-- The sum of the lengths of the two longest altitudes in the right triangle -/
noncomputable def sum_two_longest_altitudes (t : RightTriangle) : ℝ :=
  max t.a (max t.b (t.a * t.b / t.c)) + max t.b (max t.a (t.a * t.b / t.c))

/-- Theorem: The sum of the lengths of the two longest altitudes in the right triangle is 21 -/
theorem sum_two_longest_altitudes_eq_21 (t : RightTriangle) :
  sum_two_longest_altitudes t = 21 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_two_longest_altitudes_eq_21_l1324_132440


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_product_of_roots_l1324_132425

theorem product_of_roots : Real.sqrt 150 * Real.sqrt 48 * Real.sqrt 12 * (27 ^ (1/3 : ℝ)) = 360 * Real.sqrt 6 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_product_of_roots_l1324_132425


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_ratio_one_third_l1324_132478

noncomputable section

-- Define a triangle ABC in a plane
variable (A B C M : ℝ × ℝ)

-- Define the vector addition condition
def vector_sum_zero (A B C M : ℝ × ℝ) : Prop :=
  (M.1 - A.1, M.2 - A.2) + (M.1 - B.1, M.2 - B.2) + (M.1 - C.1, M.2 - C.2) = (0, 0)

-- Define the area of a triangle
noncomputable def triangle_area (P Q R : ℝ × ℝ) : ℝ :=
  abs ((Q.1 - P.1) * (R.2 - P.2) - (R.1 - P.1) * (Q.2 - P.2)) / 2

-- State the theorem
theorem area_ratio_one_third (h : vector_sum_zero A B C M) :
  triangle_area A B M / triangle_area A B C = 1/3 := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_ratio_one_third_l1324_132478


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_asymptote_sum_l1324_132476

/-- Given a function g(x) = (x+5) / (x^2 + cx + d) with vertical asymptotes at x = -1 and x = 3,
    the sum of constants c and d is -5. -/
theorem asymptote_sum (c d : ℝ) (g : ℝ → ℝ) : 
  (∀ x : ℝ, g x = (x + 5) / (x^2 + c*x + d)) →
  (∀ x : ℝ, x^2 + c*x + d = 0 ↔ x = -1 ∨ x = 3) →
  c + d = -5 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_asymptote_sum_l1324_132476


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_whales_fishing_not_sustainable_l1324_132403

/-- Represents sustainable development activities --/
inductive SustainableActivity
  | monitorSoilWaterLoss
  | analyzeWaterPollution
  | discoverWhalesFish
  | developEcoAgriculture

/-- Checks if an activity is sustainable --/
def isSustainable (activity : SustainableActivity) : Bool :=
  match activity with
  | .monitorSoilWaterLoss => true
  | .analyzeWaterPollution => true
  | .discoverWhalesFish => false
  | .developEcoAgriculture => true

/-- Theorem: Discovering whales and fishing them is not sustainable --/
theorem whales_fishing_not_sustainable :
    isSustainable SustainableActivity.discoverWhalesFish = false := by
  rfl

#eval isSustainable SustainableActivity.discoverWhalesFish

end NUMINAMATH_CALUDE_ERRORFEEDBACK_whales_fishing_not_sustainable_l1324_132403


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_concert_stay_probability_l1324_132417

/-- The probability that at least 7 out of 8 people stay for an entire concert, 
    given that 4 people have a 1/3 chance of staying and 4 are certain to stay. -/
theorem concert_stay_probability : ℝ := by
  -- Define the total number of people
  let total_people : ℕ := 8
  -- Define the number of people who are certain to stay
  let certain_people : ℕ := 4
  -- Define the number of people who are uncertain
  let uncertain_people : ℕ := 4
  -- Define the probability of an uncertain person staying
  let stay_prob : ℝ := 1/3

  -- The probability we want to prove
  have h : (1 : ℝ) / 9 = 1/9 := by sorry

  -- Return the result
  exact 1/9

end NUMINAMATH_CALUDE_ERRORFEEDBACK_concert_stay_probability_l1324_132417


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_partner_a_share_l1324_132498

/-- Calculates the share of profit for a partner in a business partnership --/
noncomputable def calculateShare (investments : Fin 3 → ℝ) (totalProfit : ℝ) (partnerIndex : Fin 3) : ℝ :=
  (investments partnerIndex / (investments 0 + investments 1 + investments 2)) * totalProfit

theorem partner_a_share 
  (investments : Fin 3 → ℝ)
  (h_invest_a : investments 0 = 7000)
  (h_invest_b : investments 1 = 11000)
  (h_invest_c : investments 2 = 18000)
  (h_b_share : calculateShare investments ((11 / 36) * 2200) 1 = 2200) :
  calculateShare investments ((11 / 36) * 2200) 0 = 1400 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_partner_a_share_l1324_132498


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_constant_term_expansion_l1324_132412

theorem constant_term_expansion : 
  ∃ (coeffs : List ℝ), ∀ x : ℝ, x > 0 → 
  (Real.sqrt x - 2 / Real.sqrt x)^6 = (coeffs.map (λ c => c * x^(3:ℕ))).sum + (-160 : ℝ) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_constant_term_expansion_l1324_132412


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_of_f_l1324_132428

noncomputable def f (x : ℝ) : ℝ := x - 4 + 9 / (x + 1)

theorem min_value_of_f :
  ∃ (a : ℝ), a > -1 ∧ f a = 1 ∧ ∀ x > -1, f x ≥ 1 :=
by
  -- We'll use 2 as the value of a
  use 2
  
  -- Split the goal into three parts
  apply And.intro
  · -- Prove 2 > -1
    norm_num
  
  apply And.intro
  · -- Prove f 2 = 1
    unfold f
    norm_num
  
  · -- Prove ∀ x > -1, f x ≥ 1
    intro x hx
    unfold f
    -- Here we should prove the inequality, but we'll use sorry for now
    sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_of_f_l1324_132428


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_k_between_f_values_l1324_132487

noncomputable def f (x : ℝ) : ℝ := 2 * (Real.cos x)^2 - (Real.sin x + Real.cos x)^2 - 2

theorem exists_k_between_f_values :
  ∃ (k : ℤ) (x₁ x₂ : ℝ),
    x₁ ∈ Set.Icc (0 : ℝ) (3 * π / 4) ∧
    x₂ ∈ Set.Icc (0 : ℝ) (3 * π / 4) ∧
    f x₁ < (k : ℝ) ∧ (k : ℝ) < f x₂ ∧
    k = -3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_k_between_f_values_l1324_132487


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_specific_triangle_l1324_132445

/-- Represents a right isosceles triangle -/
structure RightIsoscelesTriangle where
  /-- The length of the hypotenuse -/
  hypotenuse : ℝ
  /-- Condition that the hypotenuse is positive -/
  hypotenuse_pos : hypotenuse > 0

/-- Calculate the area of a right isosceles triangle -/
noncomputable def area (t : RightIsoscelesTriangle) : ℝ :=
  (t.hypotenuse^2) / 4

/-- Theorem stating that a right isosceles triangle with hypotenuse 6.000000000000001
    has an area of approximately 9.000000000000002 -/
theorem area_of_specific_triangle :
  let t : RightIsoscelesTriangle := ⟨6.000000000000001, by norm_num⟩
  abs (area t - 9.000000000000002) < 1e-10 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_specific_triangle_l1324_132445


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_parameterization_l1324_132455

/-- The line y = 2x - 8 parameterized by (x, y) = (r, -2) + t(3, m) has r = 3 and m = 6 -/
theorem line_parameterization (r m : ℝ) : 
  (∀ x y : ℝ, y = 2*x - 8 ↔ ∃ t : ℝ, (x, y) = (r + 3*t, -2 + m*t)) → 
  r = 3 ∧ m = 6 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_parameterization_l1324_132455


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complete_set_is_reals_l1324_132429

/-- A set of real numbers is complete if for any two real numbers a and b,
    if a + b is in the set, then ab is also in the set. -/
def IsCompleteSet (A : Set ℝ) : Prop :=
  ∀ a b : ℝ, (a + b) ∈ A → a * b ∈ A

theorem complete_set_is_reals (A : Set ℝ) (h_nonempty : A.Nonempty) (h_complete : IsCompleteSet A) :
  A = Set.univ := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_complete_set_is_reals_l1324_132429


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_no_valid_arrangement_l1324_132473

/-- A rectangular parallelepiped in 3D space -/
structure Parallelepiped where
  x_min : ℝ
  x_max : ℝ
  y_min : ℝ
  y_max : ℝ
  z_min : ℝ
  z_max : ℝ
  h_x : x_min < x_max
  h_y : y_min < y_max
  h_z : z_min < z_max

/-- Two parallelepipeds intersect if their projections on all axes overlap -/
def intersects (p q : Parallelepiped) : Prop :=
  (p.x_min ≤ q.x_max ∧ q.x_min ≤ p.x_max) ∧
  (p.y_min ≤ q.y_max ∧ q.y_min ≤ p.y_max) ∧
  (p.z_min ≤ q.z_max ∧ q.z_min ≤ p.z_max)

/-- The intersection conditions for the 12 parallelepipeds -/
def valid_arrangement (p : Fin 12 → Parallelepiped) : Prop :=
  ∀ i j : Fin 12, i ≠ j →
    intersects (p i) (p j) ↔ (j ≠ i.succ ∧ j ≠ (if i = 0 then 11 else i - 1))

/-- The main theorem stating that no valid arrangement exists -/
theorem no_valid_arrangement : ¬∃ p : Fin 12 → Parallelepiped, valid_arrangement p := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_no_valid_arrangement_l1324_132473


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_progression_first_term_l1324_132491

/-- Represents an infinite geometric progression -/
structure GeometricProgression where
  a : ℝ  -- first term
  r : ℝ  -- common ratio

/-- The sum to infinity of the geometric progression -/
noncomputable def sumToInfinity (gp : GeometricProgression) : ℝ := gp.a / (1 - gp.r)

/-- The sum of the first two terms of the geometric progression -/
def sumFirstTwo (gp : GeometricProgression) : ℝ := gp.a + gp.a * gp.r

theorem geometric_progression_first_term 
  (gp : GeometricProgression) 
  (h1 : sumToInfinity gp = 10)
  (h2 : sumFirstTwo gp = 7) :
  gp.a = 10 * (1 - Real.sqrt (3/10)) ∨ gp.a = 10 * (1 + Real.sqrt (3/10)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_progression_first_term_l1324_132491


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_total_rent_is_5400_l1324_132409

/-- Given the rent shares of three people, calculate the total rent -/
noncomputable def total_rent (purity_share : ℝ) : ℝ :=
  let sheila_share := 5 * purity_share
  let rose_share := 3 * purity_share
  sheila_share + purity_share + rose_share

/-- Theorem stating that the total rent is $5,400 given the conditions -/
theorem total_rent_is_5400 :
  ∃ (purity_share : ℝ),
    3 * purity_share = 1800 ∧
    total_rent purity_share = 5400 :=
by
  use 600
  constructor
  · norm_num
  · unfold total_rent
    norm_num

-- This line is commented out as it's not necessary for building
-- #eval total_rent (1800 / 3)

end NUMINAMATH_CALUDE_ERRORFEEDBACK_total_rent_is_5400_l1324_132409


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rainy_days_count_l1324_132402

/-- Represents the number of rainy days in a week. -/
def rainy_days : ℕ := sorry

/-- Represents the number of non-rainy days in a week. -/
def non_rainy_days : ℕ := sorry

/-- Represents the number of cups of hot chocolate Mo drinks on a rainy day. -/
def hot_chocolate_cups : ℕ := sorry

/-- Axiom: The number of cups of hot chocolate is even. -/
axiom hot_chocolate_even : Even hot_chocolate_cups

/-- Axiom: The total number of days in a week is 7. -/
axiom total_days : rainy_days + non_rainy_days = 7

/-- Axiom: The total number of cups drunk in a week is 36. -/
axiom total_cups : hot_chocolate_cups * rainy_days + 3 * non_rainy_days = 36

/-- Axiom: Mo drank 12 more tea cups than hot chocolate cups. -/
axiom tea_chocolate_diff : 3 * non_rainy_days - hot_chocolate_cups * rainy_days = 12

/-- Axiom: The number of rainy days is odd. -/
axiom rainy_days_odd : Odd rainy_days

/-- Theorem: The number of rainy days last week was 3. -/
theorem rainy_days_count : rainy_days = 3 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rainy_days_count_l1324_132402


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_double_angle_plus_pi_12_l1324_132446

theorem cos_double_angle_plus_pi_12 (α : ℝ) (h1 : 0 < α ∧ α < π/2) (h2 : Real.sin (α + π/6) = 3/5) :
  Real.cos (2*α + π/12) = 31*Real.sqrt 2 / 50 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_double_angle_plus_pi_12_l1324_132446


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_l1324_132490

noncomputable section

variables (f : ℝ → ℝ)

axiom f_derivative (x : ℝ) : deriv f x > 1 - f x
axiom f_initial : f 0 = 6

theorem solution_set :
  (∃ y > 0, ∀ z, z > 0 → z < y → Real.exp z * f z > Real.exp z + 5) ∧
  (∀ y ≤ 0, Real.exp y * f y ≤ Real.exp y + 5) :=
sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_l1324_132490


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_successive_odd_terms_l1324_132497

/-- Represents a sequence of positive integers where each term is obtained
    by adding the largest digit of the previous term to itself. -/
def DigitAddSequence : Type := ℕ+ → ℕ+

/-- Returns the largest digit of a positive integer. -/
def largestDigit (n : ℕ+) : Fin 10 :=
  sorry

/-- Generates the next term in the sequence. -/
def nextTerm (n : ℕ+) : ℕ+ :=
  ⟨n + (largestDigit n).val, by
    have h : 0 < n + (largestDigit n).val := by
      apply Nat.add_pos_left
      exact n.property
    exact h⟩

/-- Checks if a positive integer is odd. -/
def isOdd (n : ℕ+) : Prop :=
  ∃ k : ℕ, n.val = 2 * k + 1

/-- Counts the number of successive odd terms starting from a given index. -/
def countSuccessiveOddTerms (seq : DigitAddSequence) (start : ℕ) : ℕ :=
  sorry

/-- The main theorem stating that the maximal number of successive odd terms is 5. -/
theorem max_successive_odd_terms (seq : DigitAddSequence) :
  (∀ start : ℕ, countSuccessiveOddTerms seq start ≤ 5) ∧
  (∃ seq : DigitAddSequence, ∃ start : ℕ, countSuccessiveOddTerms seq start = 5) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_successive_odd_terms_l1324_132497


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_consecutive_integer_sets_no_overlap_l1324_132418

/-- Given two sets of 7 consecutive positive integers where the sum of the integers
    in the set with greater numbers is 42 more than the sum of the integers in the other set,
    prove that the two sets have 0 integers in common. -/
theorem consecutive_integer_sets_no_overlap (a b : ℕ) : 
  (∀ i ∈ Finset.range 7, (a + i : ℕ) > 0) →
  (∀ i ∈ Finset.range 7, (b + i : ℕ) > 0) →
  (Finset.sum (Finset.range 7) (λ i => b + i) = 
   Finset.sum (Finset.range 7) (λ i => a + i) + 42) →
  Finset.card (Finset.range 7 ∩ Finset.image (λ i => i + (b - a)) (Finset.range 7)) = 0 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_consecutive_integer_sets_no_overlap_l1324_132418


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_state_A_selection_percentage_l1324_132436

theorem state_A_selection_percentage
  (total_candidates : ℕ)
  (state_B_percentage : ℚ)
  (state_A_percentage : ℚ)
  (difference : ℕ)
  (h1 : total_candidates = 8200)
  (h2 : state_B_percentage = 7 / 100)
  (h3 : (state_B_percentage * total_candidates).floor - 
        (state_A_percentage * total_candidates).floor = difference)
  (h4 : difference = 82)
  : state_A_percentage = 6 / 100 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_state_A_selection_percentage_l1324_132436


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_base8_digits_of_1024_l1324_132463

/-- The number of digits in the base-8 representation of a positive integer -/
noncomputable def num_digits_base8 (n : ℕ+) : ℕ :=
  Nat.floor (Real.log n / Real.log 8) + 1

/-- Theorem: The number of digits in the base-8 representation of 1024 is 4 -/
theorem base8_digits_of_1024 : num_digits_base8 1024 = 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_base8_digits_of_1024_l1324_132463


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_sum_product_relation_l1324_132444

theorem tangent_sum_product_relation (α β γ : Real) :
  Real.tan α + Real.tan β + Real.tan γ = Real.tan α * Real.tan β * Real.tan γ →
  ∃ k : Int, α + β + γ = k * Real.pi := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_sum_product_relation_l1324_132444


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_exact_four_twos_l1324_132424

/-- The number of sides on each die -/
def numSides : ℕ := 8

/-- The number of dice rolled -/
def numDice : ℕ := 12

/-- The number of dice showing a 2 -/
def numSuccess : ℕ := 4

/-- The probability of rolling exactly 4 twos when rolling 12 eight-sided dice -/
noncomputable def probabilityExactFourTwos : ℝ :=
  (Nat.choose numDice numSuccess : ℝ) * (1 / numSides : ℝ) ^ numSuccess * ((numSides - 1) / numSides : ℝ) ^ (numDice - numSuccess)

theorem probability_exact_four_twos :
  abs (probabilityExactFourTwos - 0.091) < 0.0005 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_exact_four_twos_l1324_132424


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sophia_ate_one_sixth_l1324_132499

/-- The fraction of pie Sophia ate -/
noncomputable def fraction_eaten (pie_left : ℝ) (pie_eaten : ℝ) : ℝ :=
  pie_eaten / (pie_left + pie_eaten)

/-- Theorem: Given the weight of pie left and eaten, prove the fraction eaten is 1/6 -/
theorem sophia_ate_one_sixth (pie_left : ℝ) (pie_eaten : ℝ) 
  (h1 : pie_left = 1200) 
  (h2 : pie_eaten = 240) : 
  fraction_eaten pie_left pie_eaten = 1/6 := by
  sorry

-- Remove the #eval statement as it's not computable
-- #eval fraction_eaten 1200 240

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sophia_ate_one_sixth_l1324_132499


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_and_function_properties_l1324_132486

-- Define the triangle ABC
def triangle_ABC (A B C : ℝ) : Prop :=
  let BC := 2 * Real.sqrt 2
  let AC := 2
  Real.cos (A + B) = -Real.sqrt 2 / 2

-- Define the function f(x)
noncomputable def f (x C : ℝ) : ℝ := Real.sin (2 * x + C)

-- Theorem statement
theorem triangle_and_function_properties :
  ∀ (A B C : ℝ),
  triangle_ABC A B C →
  ∃ (AB : ℝ),
    AB = 2 ∧
    ∀ (C : ℝ),
      ∃ (min_distance : ℝ),
        min_distance = Real.pi / 6 ∧
        (∀ (x₁ x₂ : ℝ),
          f x₁ C = Real.sqrt 3 / 2 →
          f x₂ C = Real.sqrt 3 / 2 →
          x₁ ≠ x₂ →
          |x₁ - x₂| ≥ min_distance) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_and_function_properties_l1324_132486


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_calculation_proof_l1324_132483

theorem calculation_proof :
  (- (-1)^4 - (1 - 1/2) * (1 / 3) * (2 - 3^2) = 1 / 6) ∧
  (5 / 13 * (-13/4) - 1/2 / |(-3 - 1)| = -11 / 8) := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_calculation_proof_l1324_132483


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sufficient_but_not_necessary_l1324_132485

noncomputable def f (m : ℤ) (x : ℝ) : ℝ := x ^ (m - 1)

theorem sufficient_but_not_necessary (m : ℤ) :
  (∀ x : ℝ, f m (-x) = -(f m x)) →  -- f is an odd function
  (∀ x y : ℝ, 0 < x → x < y → f 4 x ≤ f 4 y) ∧ 
  (∃ n : ℤ, n ≠ 4 ∧ (∀ x y : ℝ, 0 < x → x < y → f n x ≤ f n y)) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sufficient_but_not_necessary_l1324_132485


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_derivative_of_u_l1324_132448

/-- Given a function u(x) defined as u = e^(z - 2y), where z = sin x and y = x^2,
    prove that its derivative with respect to x is e^(sin x - 2x^2) * (cos x - 4x) -/
theorem derivative_of_u (x : ℝ) : 
  let z := Real.sin x
  let y := x^2
  let u := fun x => Real.exp (Real.sin x - 2*x^2)
  (deriv u) x = Real.exp (Real.sin x - 2*x^2) * (Real.cos x - 4*x) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_derivative_of_u_l1324_132448


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parallelogram_z_value_l1324_132469

-- Define the complex numbers
def A : ℂ := -3 - 2*Complex.I
def B : ℂ := -4 + 5*Complex.I
def C : ℂ := 2 + Complex.I

-- Define the parallelogram property
def is_parallelogram (A B C D : ℂ) : Prop := B - A = D - C

-- Theorem statement
theorem parallelogram_z_value (z : ℂ) 
  (h : is_parallelogram A B C z) : z = 1 + 8*Complex.I := by
  sorry

#check parallelogram_z_value

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parallelogram_z_value_l1324_132469


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_list_price_is_35_l1324_132443

/-- The list price of the item -/
def list_price : ℝ := 35

/-- Alice's selling price -/
def alice_price : ℝ := list_price - 15

/-- Bob's selling price -/
def bob_price : ℝ := list_price - 20

/-- Alice's commission rate -/
def alice_commission_rate : ℝ := 0.15

/-- Bob's commission rate -/
def bob_commission_rate : ℝ := 0.20

/-- Alice's commission -/
def alice_commission : ℝ := alice_commission_rate * alice_price

/-- Bob's commission -/
def bob_commission : ℝ := bob_commission_rate * bob_price

theorem list_price_is_35 : 
  alice_commission = bob_commission → list_price = 35 := by
  intro h
  -- Here we would normally prove that list_price = 35
  -- For now, we'll use sorry to skip the proof
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_list_price_is_35_l1324_132443


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l1324_132464

open Real

/-- The function f(x) as defined in the problem -/
noncomputable def f (ω : ℝ) (x : ℝ) : ℝ := 
  sin (ω * x) ^ 2 + sqrt 3 * cos (ω * x) * cos (Real.pi / 2 - ω * x)

/-- The theorem stating the properties of the function f -/
theorem function_properties (ω : ℝ) (h_ω : ω > 0) :
  (∃ (T : ℝ), T > 0 ∧ ∀ (x : ℝ), f ω (x + T / 2) = f ω x) →
  (∀ (x : ℝ), f ω (x + Real.pi / 2) = f ω x) →
  (f ω (Real.pi / 6) = 1) ∧
  (∀ (k : ℝ), 0 < k ∧ k ≤ 3 / 4 ↔
    (∀ (x y : ℝ), -Real.pi / 6 ≤ x ∧ x < y ∧ y ≤ Real.pi / 3 →
      f ω (k * x + Real.pi / 12) < f ω (k * y + Real.pi / 12))) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l1324_132464


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_count_semisymmetric_codes_l1324_132427

/-- A semisymmetric scanning code is a 7x7 grid that appears the same when rotated 180° or reflected across horizontal or vertical center lines. -/
def SemisymmetricScanningCode : Type := Matrix (Fin 7) (Fin 7) Bool

/-- A scanning code is valid if it contains at least one square of each color. -/
def is_valid_code (code : SemisymmetricScanningCode) : Prop :=
  (∃ i j, code i j = true) ∧ (∃ i j, code i j = false)

/-- The number of independently variable cells in a semisymmetric 7x7 grid. -/
def independent_cells : Nat := 10

/-- The total number of possible semisymmetric scanning codes. -/
def total_semisymmetric_codes : Nat := 2^independent_cells

/-- The number of invalid codes (all black or all white). -/
def invalid_codes : Nat := 2

/-- The count of valid semisymmetric scanning codes. -/
def count_valid_codes : Nat := total_semisymmetric_codes - invalid_codes

theorem count_semisymmetric_codes :
  count_valid_codes = 1022 :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_count_semisymmetric_codes_l1324_132427


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_in_fourth_quadrant_l1324_132437

theorem angle_in_fourth_quadrant (α : Real) 
  (h1 : Real.sin (2 * α) < 0) 
  (h2 : Real.tan α * Real.cos α < 0) : 
  π < α ∧ α < (3 * π) / 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_in_fourth_quadrant_l1324_132437


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_at_origin_l1324_132461

/-- The function f(x) defined as x² + a * ln(x+1) -/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := x^2 + a * Real.log (x + 1)

/-- The derivative of f(x) -/
noncomputable def f_derivative (a : ℝ) (x : ℝ) : ℝ := 2 * x + a / (x + 1)

theorem tangent_line_at_origin (a : ℝ) :
  (∀ x, f_derivative a 0 * x = -x) → a = -1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_at_origin_l1324_132461


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_division_remainder_l1324_132492

theorem polynomial_division_remainder : 
  ∃ q : Polynomial ℝ, X^2023 = q * ((X^2 + 1) * (X - 1)) + X^3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_division_remainder_l1324_132492


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_not_trisectable_120_div_n_l1324_132475

-- Define a type for constructible angles
def ConstructibleAngle : Type := ℝ

-- Define a predicate for trisectable angles
def IsTrisectable (α : ConstructibleAngle) : Prop := sorry

-- Define a function to convert degrees to radians
noncomputable def degToRad (deg : ℝ) : ℝ := (deg * Real.pi) / 180

-- Axiom: 60° angle is not trisectable
axiom not_trisectable_60 : ¬ IsTrisectable (degToRad 60)

-- Theorem to prove
theorem not_trisectable_120_div_n (n : ℕ) (hn : n > 0) :
  ¬ IsTrisectable (degToRad (120 / n)) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_not_trisectable_120_div_n_l1324_132475


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_region_R_l1324_132431

/-- A square with side length 10 -/
structure Square :=
  (side_length : ℝ)
  (is_ten : side_length = 10)

/-- The region R in the square -/
def region_R (s : Square) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.1 ≥ 0 ∧ p.2 ≥ 0 ∧ p.1 + p.2 ≤ 10}

/-- The theorem stating that the area of region R is 25 -/
theorem area_of_region_R (s : Square) : MeasureTheory.volume (region_R s) = 25 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_region_R_l1324_132431


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_inequality_l1324_132421

open Set
open Function

noncomputable section

def f : ℝ → ℝ := sorry

theorem function_inequality (hf : Differentiable ℝ f) 
  (h1 : ∀ x > 0, f x < -x * (deriv f x)) :
  ∀ x > 2, f (x + 1) > (x - 1) * f (x^2 - 1) := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_inequality_l1324_132421


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sphere_volume_circumscribing_unit_cube_l1324_132432

theorem sphere_volume_circumscribing_unit_cube : 
  (4 / 3 : ℝ) * Real.pi * ((Real.sqrt 3 / 2) ^ 3) = (Real.sqrt 3 * Real.pi) / 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sphere_volume_circumscribing_unit_cube_l1324_132432


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_log_sum_equality_l1324_132472

noncomputable def log_base (b x : ℝ) := Real.log x / Real.log b

noncomputable def f₁ (x : ℝ) := log_base (x^2) (x^2 - 10*x + 21)
noncomputable def f₂ (x : ℝ) := log_base (x^2) (x^2 / (x - 7))
noncomputable def f₃ (x : ℝ) := log_base (x^2) (x^2 / (x - 3))

theorem log_sum_equality (x : ℝ) :
  (x ≠ 0 ∧ x ≠ 3 ∧ x ≠ 7) →
  (∃ (i j k : Fin 3), i ≠ j ∧ j ≠ k ∧ i ≠ k ∧
    [f₁ x, f₂ x, f₃ x].get i = [f₁ x, f₂ x, f₃ x].get j + [f₁ x, f₂ x, f₃ x].get k) →
  x = 8 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_log_sum_equality_l1324_132472


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_between_floor_values_distance_between_floor_values_alt_l1324_132458

noncomputable def floor (x : ℝ) : ℤ := ⌊x⌋

theorem distance_between_floor_values : 
  |floor 3.7 - floor (-6.5)| = 10 := by
  -- Convert the real numbers to their floor values
  have h1 : floor 3.7 = 3 := by sorry
  have h2 : floor (-6.5) = -7 := by sorry

  -- Calculate the absolute difference
  calc
    |floor 3.7 - floor (-6.5)| = |3 - (-7)| := by rw [h1, h2]
    _ = |3 + 7| := by ring
    _ = |10| := by ring
    _ = 10 := by norm_num

theorem distance_between_floor_values_alt : 
  |floor 3.7 - floor (-6.5)| = 10 := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_between_floor_values_distance_between_floor_values_alt_l1324_132458


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_savings_calculation_l1324_132489

/-- Calculates the total savings given income, income-expenditure ratio, tax rate, and investment rate. -/
def totalSavings (income : ℚ) (incomeRatio expenditureRatio taxRate investmentRate : ℚ) : ℚ :=
  let expenditure := (expenditureRatio / incomeRatio) * income
  let taxes := taxRate * income
  let investments := investmentRate * income
  income - (expenditure + taxes + investments)

/-- Theorem stating that given the specific conditions, the total savings is 900. -/
theorem savings_calculation (income : ℚ) (incomeRatio expenditureRatio taxRate investmentRate : ℚ)
  (h1 : income = 17000)
  (h2 : incomeRatio = 5)
  (h3 : expenditureRatio = 4)
  (h4 : taxRate = 15/100)
  (h5 : investmentRate = 10/100) :
  totalSavings income incomeRatio expenditureRatio taxRate investmentRate = 900 := by
  sorry

#eval totalSavings 17000 5 4 (15/100) (10/100)

end NUMINAMATH_CALUDE_ERRORFEEDBACK_savings_calculation_l1324_132489


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tank_volume_ratio_l1324_132481

/-- The volume of a right circular cylinder -/
noncomputable def cylinderVolume (height : ℝ) (circumference : ℝ) : ℝ :=
  (height * circumference^2) / (4 * Real.pi)

/-- The theorem stating that the volume of tank A is 80% of the volume of tank B -/
theorem tank_volume_ratio :
  let volumeA := cylinderVolume 10 8
  let volumeB := cylinderVolume 8 10
  volumeA / volumeB = 4/5 := by
  sorry

#check tank_volume_ratio

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tank_volume_ratio_l1324_132481


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_number_of_valid_pet_choices_l1324_132447

/-- Represents the types of pets available in the store -/
inductive PetType
  | Puppy
  | Kitten
  | Hamster
  | Bird
deriving Fintype, Repr

/-- Represents the customers -/
inductive Customer
  | Alice
  | Bob
  | Charlie
  | David
deriving Fintype, Repr

/-- The number of each type of pet available -/
def petCount (pt : PetType) : Nat :=
  match pt with
  | PetType.Puppy => 20
  | PetType.Kitten => 10
  | PetType.Hamster => 8
  | PetType.Bird => 5

/-- Predicate that determines if a customer can choose a given pet type -/
def canChoose (c : Customer) (pt : PetType) : Prop :=
  match c, pt with
  | Customer.Alice, PetType.Bird => False
  | Customer.Bob, PetType.Hamster => False
  | Customer.Charlie, PetType.Kitten => False
  | Customer.David, PetType.Puppy => False
  | _, _ => True

/-- A valid pet choice is a function from Customer to PetType satisfying the constraints -/
def ValidPetChoice : Type :=
  { f : Customer → PetType // 
    (∀ c, canChoose c (f c)) ∧ 
    (∀ c1 c2, c1 ≠ c2 → f c1 ≠ f c2) }

instance : Fintype ValidPetChoice := by
  sorry  -- This instance is required for Fintype.card to work

/-- The main theorem stating the number of valid pet choices -/
theorem number_of_valid_pet_choices : 
  Fintype.card ValidPetChoice = 791440 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_number_of_valid_pet_choices_l1324_132447


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_beta_value_l1324_132442

theorem cos_beta_value (α β m : ℝ) 
  (h1 : Real.sin (α - β) * Real.cos α - Real.cos (α - β) * Real.sin α = m)
  (h2 : π < β ∧ β < 3 * π / 2) : 
  Real.cos β = -Real.sqrt (1 - m^2) := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_beta_value_l1324_132442


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_perimeter_l1324_132411

-- Define the triangle ABC
structure Triangle :=
  (A B C : ℝ × ℝ)

-- Define the properties of the triangle
noncomputable def Triangle.area (t : Triangle) : ℝ := 4 * Real.sqrt 3 / 3

def Triangle.sideAC (t : Triangle) : ℝ := 3

noncomputable def Triangle.angleB (t : Triangle) : ℝ := 60 * Real.pi / 180

-- State the theorem
theorem triangle_perimeter (t : Triangle) :
  t.area = 4 * Real.sqrt 3 / 3 →
  t.sideAC = 3 →
  t.angleB = 60 * Real.pi / 180 →
  (let a := Real.sqrt ((t.A.1 - t.B.1)^2 + (t.A.2 - t.B.2)^2)
   let b := Real.sqrt ((t.B.1 - t.C.1)^2 + (t.B.2 - t.C.2)^2)
   let c := t.sideAC
   a + b + c = 8) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_perimeter_l1324_132411


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_derivative_at_neg_pi_fourth_l1324_132467

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := Real.cos x

-- State the theorem
theorem cos_derivative_at_neg_pi_fourth :
  deriv f (-π/4) = Real.sqrt 2 / 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_derivative_at_neg_pi_fourth_l1324_132467


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangle_angle_congruence_l1324_132433

-- Define the point type
variable {Point : Type*}

-- Define the property of being a rectangle
def is_rectangle (A B C D : Point) : Prop := sorry

-- Define the midpoint of a line segment
def is_midpoint (M P Q : Point) : Prop := sorry

-- Define the intersection of two lines
def is_intersection (G P Q R S : Point) : Prop := sorry

-- Define angle congruence
def angle_congruent (P Q R S T U : Point) : Prop := sorry

-- State the theorem
theorem rectangle_angle_congruence 
  {A B C D E F G : Point}
  (h_rectangle : is_rectangle A B C D)
  (h_midpoint_E : is_midpoint E A D)
  (h_midpoint_F : is_midpoint F D C)
  (h_intersection : is_intersection G A F E C) :
  angle_congruent C G F F B E := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangle_angle_congruence_l1324_132433


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_circle_no_intersection_l1324_132462

/-- The line equation is x - y - 4 = 0 -/
def line (x y : ℝ) : Prop := x - y - 4 = 0

/-- The circle equation is x^2 + y^2 - 2x - 2y - 2 = 0 -/
def circle_eq (x y : ℝ) : Prop := x^2 + y^2 - 2*x - 2*y - 2 = 0

/-- The distance between a point (x, y) and the line ax + by + c = 0 -/
noncomputable def distance_point_to_line (x y a b c : ℝ) : ℝ :=
  abs (a*x + b*y + c) / Real.sqrt (a^2 + b^2)

theorem line_circle_no_intersection :
  ∀ x y : ℝ, ¬(line x y ∧ circle_eq x y) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_circle_no_intersection_l1324_132462


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_zero_of_f_when_a_is_neg_sqrt2_range_of_a_when_f_has_no_zero_l1324_132456

/-- Piecewise function definition -/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x < 2 then 2^x + a else a - x

/-- Theorem for the zero of f when a = -√2 -/
theorem zero_of_f_when_a_is_neg_sqrt2 :
  ∃! x : ℝ, f (-Real.sqrt 2) x = 0 ∧ x = 1/2 := by sorry

/-- Theorem for the range of a when f has no zero -/
theorem range_of_a_when_f_has_no_zero :
  ∀ a : ℝ, (∀ x : ℝ, f a x ≠ 0) ↔ a ∈ Set.Iic (-4) ∪ Set.Ioc 0 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_zero_of_f_when_a_is_neg_sqrt2_range_of_a_when_f_has_no_zero_l1324_132456


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_wise_men_strategy_exists_l1324_132466

/-- Represents a hat color -/
def HatColor := Fin 1000

/-- Represents a card color (white or black) -/
inductive CardColor
| white
| black

/-- Represents the configuration of hats on the wise men -/
def HatConfiguration := Fin 11 → HatColor

/-- Represents the cards shown by the wise men -/
def CardConfiguration := Fin 11 → CardColor

/-- A strategy for a wise man to choose a card color based on what they see -/
def CardStrategy := (Fin 11 → Option HatColor) → CardColor

/-- A strategy for a wise man to guess their hat color based on what they see and all shown cards -/
def GuessStrategy := (Fin 11 → Option HatColor) → CardConfiguration → HatColor

/-- The theorem stating that a successful strategy exists -/
theorem wise_men_strategy_exists :
  ∃ (card_strategy : Fin 11 → CardStrategy) (guess_strategy : Fin 11 → GuessStrategy),
    ∀ (config : HatConfiguration),
      let card_config : CardConfiguration := λ i ↦
        card_strategy i (λ j ↦ if i = j then none else some (config j))
      ∀ i : Fin 11, guess_strategy i (λ j ↦ if i = j then none else some (config j)) card_config = config i :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_wise_men_strategy_exists_l1324_132466


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_area_increase_l1324_132459

theorem circle_area_increase (r : ℝ) (h : r > 0) : 
  (π * (1.5 * r)^2 - π * r^2) / (π * r^2) = 1.25 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_area_increase_l1324_132459


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_michael_record_score_l1324_132438

/-- Represents a basketball game with Michael and his teammates --/
structure BasketballGame where
  totalScore : ℕ
  otherPlayers : ℕ
  otherPlayersAverage : ℚ

/-- Calculates Michael's score in the basketball game --/
def michaelScore (game : BasketballGame) : ℕ :=
  game.totalScore - (game.otherPlayers * game.otherPlayersAverage).floor.toNat

/-- Theorem stating that Michael scored 36 points in the record-setting game --/
theorem michael_record_score (game : BasketballGame) 
    (h1 : game.totalScore = 72)
    (h2 : game.otherPlayers = 8)
    (h3 : game.otherPlayersAverage = 4.5) :
    michaelScore game = 36 := by
  sorry

/-- The actual record-setting game --/
def recordGame : BasketballGame :=
  { totalScore := 72
    otherPlayers := 8
    otherPlayersAverage := 4.5 }

#eval michaelScore recordGame

end NUMINAMATH_CALUDE_ERRORFEEDBACK_michael_record_score_l1324_132438


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_A_rolls_correct_l1324_132414

/-- The probability that player A rolls on the nth turn in a dice game with the following rules:
  1. A and B take turns rolling a die, with A going first.
  2. If A rolls a 1, A continues to roll; otherwise, it's B's turn.
  3. If B rolls a 3, B continues to roll; otherwise, it's A's turn. -/
noncomputable def prob_A_rolls (n : ℕ) : ℝ :=
  1/2 - 1/3 * (-2/3)^(n-2)

/-- The probability that A rolls a 1 and continues -/
noncomputable def prob_A_continues : ℝ := 1/6

/-- The probability that B rolls a 3 and continues -/
noncomputable def prob_B_continues : ℝ := 1/6

/-- Theorem stating that the probability function is correct -/
theorem prob_A_rolls_correct (n : ℕ) :
  prob_A_rolls n = 1/2 - 1/3 * (-2/3)^(n-2) :=
by
  -- The proof is omitted for now
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_A_rolls_correct_l1324_132414


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_nested_function_evaluation_l1324_132471

-- Define the functions N and O
noncomputable def N (x : ℝ) : ℝ := 3 * Real.sqrt x
def O (x : ℝ) : ℝ := x^2 + 1

-- State the theorem
theorem nested_function_evaluation :
  N (O (N (O (N (O 2))))) = 3 * Real.sqrt 415 := by
  -- Expand the definition of O(2)
  have h1 : O 2 = 5 := by
    rw [O]
    norm_num
  
  -- Apply N to O(2)
  have h2 : N (O 2) = 3 * Real.sqrt 5 := by
    rw [h1, N]
  
  -- Apply O to N(O(2))
  have h3 : O (N (O 2)) = 46 := by
    rw [h2, O]
    ring_nf
    norm_num
  
  -- Apply N to O(N(O(2)))
  have h4 : N (O (N (O 2))) = 3 * Real.sqrt 46 := by
    rw [h3, N]
  
  -- Apply O to N(O(N(O(2))))
  have h5 : O (N (O (N (O 2)))) = 415 := by
    rw [h4, O]
    ring_nf
    norm_num
  
  -- Final step: apply N to O(N(O(N(O(2)))))
  rw [h5, N]


end NUMINAMATH_CALUDE_ERRORFEEDBACK_nested_function_evaluation_l1324_132471


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_investment_duration_approximation_l1324_132474

/-- Calculates the duration of an investment given the principal, interest rate, and final amount. -/
noncomputable def calculate_duration (principal : ℝ) (rate : ℝ) (final_amount : ℝ) : ℝ :=
  (Real.log (final_amount / principal)) / (Real.log (1 + rate))

/-- Theorem stating that the calculated duration for the given investment parameters is approximately 2.237 years. -/
theorem investment_duration_approximation :
  let principal : ℝ := 886.0759493670886
  let rate : ℝ := 0.11
  let final_amount : ℝ := 1120
  let duration := calculate_duration principal rate final_amount
  ∃ ε > 0, |duration - 2.237| < ε ∧ ε < 0.001 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_investment_duration_approximation_l1324_132474


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_statement_l1324_132423

/-- The function f(x) for part 1 -/
noncomputable def f₁ (c : ℝ) (x : ℝ) : ℝ := x + c^2 / x

/-- The function f(x) for part 2 -/
noncomputable def f₂ (a c : ℝ) (x : ℝ) : ℝ := a^2 * x + c^2 / (x - 1)

theorem problem_statement :
  (∀ c : ℝ, c > 0 → ∀ x : ℝ, x ≠ 0 → |f₁ c x| ≥ 2 * c) ∧
  (∀ a c : ℝ, a > 0 → c > 0 → (∀ x : ℝ, x > 1 → f₂ a c x > a) → a + 2 * c > 1) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_statement_l1324_132423


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_has_unique_root_in_interval_l1324_132420

-- Define the function f(x)
noncomputable def f (x : ℝ) : ℝ := Real.exp x + (1/2) * x - 2

-- State the theorem
theorem f_has_unique_root_in_interval :
  ∃! x, x ∈ Set.Ioo (1/2 : ℝ) 1 ∧ f x = 0 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_has_unique_root_in_interval_l1324_132420


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_perp_line_plane_iff_perp_line_l1324_132439

-- Define the types for our geometric objects
variable (Point Line Plane : Type)

-- Define the relations we need
variable (on_plane : Point → Plane → Prop)
variable (on_line : Point → Line → Prop)
variable (perp_plane : Plane → Plane → Prop)
variable (perp_line_plane : Point → Point → Plane → Prop)
variable (perp_line : Point → Point → Line → Prop)
variable (intersect : Plane → Plane → Line → Prop)

-- State the theorem
theorem perp_line_plane_iff_perp_line 
  (α β : Plane) (l : Line) (P Q : Point) :
  perp_plane α β →
  intersect α β l →
  on_plane P α →
  on_line Q l →
  (perp_line_plane P Q β ↔ perp_line P Q l) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_perp_line_plane_iff_perp_line_l1324_132439


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_chord_ratio_l1324_132452

-- Define the parabola
def parabola (p : ℝ) (x y : ℝ) : Prop := y^2 = 2*p*x ∧ p > 0

-- Define the focus of the parabola
noncomputable def focus (p : ℝ) : ℝ × ℝ := (p/2, 0)

-- Define a point on the parabola
def pointOnParabola (p : ℝ) (x y : ℝ) : Prop := parabola p x y

-- Define a line with 60° inclination passing through the focus
def lineThrough60Deg (p : ℝ) (x y : ℝ) : Prop := 
  y = Real.sqrt 3 * (x - p/2)

-- Define the chord AB
def chordAB (p : ℝ) (x₁ y₁ x₂ y₂ : ℝ) : Prop :=
  pointOnParabola p x₁ y₁ ∧ 
  pointOnParabola p x₂ y₂ ∧
  lineThrough60Deg p x₁ y₁ ∧
  lineThrough60Deg p x₂ y₂

-- Define the distances |AF| and |BF|
noncomputable def distAF (p : ℝ) (x₁ : ℝ) : ℝ := x₁ + p/2
noncomputable def distBF (p : ℝ) (x₂ : ℝ) : ℝ := x₂ + p/2

-- Theorem statement
theorem parabola_chord_ratio (p : ℝ) (x₁ y₁ x₂ y₂ : ℝ) :
  parabola p x₁ y₁ →
  parabola p x₂ y₂ →
  chordAB p x₁ y₁ x₂ y₂ →
  distAF p x₁ > distBF p x₂ →
  distAF p x₁ / distBF p x₂ = 3 := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_chord_ratio_l1324_132452


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_m_for_f_zero_l1324_132413

-- Define the function f
noncomputable def f (x m : ℝ) : ℝ := (Real.sin x + Real.cos x)^2 - 2 * (Real.cos x)^2 - m

-- State the theorem
theorem range_of_m_for_f_zero :
  ∃ (m_min m_max : ℝ), m_min = -1 ∧ m_max = Real.sqrt 2 ∧
  (∀ m : ℝ, (∃ x : ℝ, x ∈ Set.Icc 0 (Real.pi / 2) ∧ f x m = 0) ↔ m ∈ Set.Icc m_min m_max) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_m_for_f_zero_l1324_132413


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_fraction_equality_l1324_132451

theorem complex_fraction_equality : (6 : ℂ) + 7 * Complex.I / (1 + 2 * Complex.I) = 4 - Complex.I := by
  -- The proof is omitted
  sorry

#check complex_fraction_equality

end NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_fraction_equality_l1324_132451


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_clownfish_in_display_tank_l1324_132406

theorem clownfish_in_display_tank (total_fish : ℕ) 
  (blowfish_own_tank : ℕ) (clownfish_return_fraction : ℚ) 
  (angelfish_display_ratio : ℚ) (angelfish_own_ratio : ℚ) :
  total_fish = 180 ∧ 
  blowfish_own_tank = 26 ∧ 
  clownfish_return_fraction = 1/3 ∧
  angelfish_display_ratio = 3/2 ∧
  angelfish_own_ratio = 1/2 →
  ∃ (clownfish blowfish angelfish : ℕ),
    clownfish + blowfish + angelfish = total_fish ∧
    clownfish = blowfish ∧
    angelfish = 2 * blowfish ∧
    ∃ (clownfish_display : ℕ),
      clownfish_display = clownfish - Int.floor (clownfish_return_fraction * (blowfish - blowfish_own_tank : ℚ)) ∧
      clownfish_display = 13 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_clownfish_in_display_tank_l1324_132406


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_first_10_terms_l1324_132415

def sequenceA (n : ℕ) : ℤ := (-1)^n * (3*n - 2)

theorem sum_of_first_10_terms : 
  (Finset.range 10).sum (λ i => sequenceA (i + 1)) = 15 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_first_10_terms_l1324_132415


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_truck_distance_calculation_l1324_132494

/-- The direct distance from the starting point to the final position of a truck -/
noncomputable def truck_distance (north1 : ℝ) (east : ℝ) (north2 : ℝ) : ℝ :=
  Real.sqrt ((north1 + north2)^2 + east^2)

theorem truck_distance_calculation :
  truck_distance 20 30 20 = 50 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_truck_distance_calculation_l1324_132494


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_interval_implies_a_range_l1324_132479

/-- The function f(x) = -1/3 * x^3 + x --/
noncomputable def f (x : ℝ) : ℝ := -1/3 * x^3 + x

/-- The theorem stating that if f(x) has a maximum value on the interval (a, 10 - a^2),
    then a is in the range [-2, 1) --/
theorem max_value_interval_implies_a_range (a : ℝ) :
  (∃ (x : ℝ), a < x ∧ x < 10 - a^2 ∧ ∀ (y : ℝ), a < y ∧ y < 10 - a^2 → f y ≤ f x) →
  -2 ≤ a ∧ a < 1 := by
  sorry

#check max_value_interval_implies_a_range

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_interval_implies_a_range_l1324_132479


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_triangle_area_l1324_132405

/-- Parabola with focus at (0,1) and equation x^2 = 4y -/
def Parabola : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.1^2 = 4 * p.2}

/-- The focus of the parabola -/
def F : ℝ × ℝ := (0, 1)

/-- Vector from F to a point P -/
def FP (P : ℝ × ℝ) : ℝ × ℝ := (P.1 - F.1, P.2 - F.2)

/-- Condition that A, B, C satisfy FA + FB + FC = 0 -/
def SumZero (A B C : ℝ × ℝ) : Prop :=
  FP A + FP B + FP C = (0, 0)

/-- Area of triangle ABC -/
noncomputable def TriangleArea (A B C : ℝ × ℝ) : ℝ :=
  abs ((B.1 - A.1) * (C.2 - A.2) - (C.1 - A.1) * (B.2 - A.2)) / 2

/-- Theorem stating the maximum area of triangle ABC -/
theorem max_triangle_area :
  ∃ (max_area : ℝ),
    max_area = (3 * Real.sqrt 6) / 2 ∧
    ∀ (A B C : ℝ × ℝ),
      A ∈ Parabola → B ∈ Parabola → C ∈ Parabola →
      SumZero A B C →
      TriangleArea A B C ≤ max_area :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_triangle_area_l1324_132405


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complement_A_intersect_B_when_a_is_3_A_disjoint_B_iff_a_in_range_l1324_132407

-- Define sets A and B
def A (a : ℝ) : Set ℝ := {x | a ≤ x ∧ x ≤ a + 3}
def B : Set ℝ := {x | x < -1 ∨ x > 5}

-- Part I
theorem complement_A_intersect_B_when_a_is_3 : 
  (Set.univ \ A 3) ∩ B = {x | x < -1 ∨ x > 6} := by sorry

-- Part II
theorem A_disjoint_B_iff_a_in_range : 
  ∀ a : ℝ, (A a ∩ B = ∅) ↔ -1 ≤ a ∧ a ≤ 5 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_complement_A_intersect_B_when_a_is_3_A_disjoint_B_iff_a_in_range_l1324_132407


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_karen_hike_food_weight_l1324_132484

/-- The initial weight of food packed by Karen for her hike --/
noncomputable def initial_food_weight : ℝ := 10

/-- The total weight Karen is carrying after 6 hours of hiking --/
noncomputable def weight_after_6_hours : ℝ := 34

/-- The amount of water Karen packs initially --/
noncomputable def initial_water_weight : ℝ := 20

/-- The amount of gear Karen packs --/
noncomputable def gear_weight : ℝ := 20

/-- The rate at which Karen drinks water per hour --/
noncomputable def water_consumption_rate : ℝ := 2

/-- The rate at which Karen eats food relative to water consumption --/
noncomputable def food_consumption_rate : ℝ := 1/3

/-- The number of hours Karen hikes --/
noncomputable def hiking_duration : ℝ := 6

theorem karen_hike_food_weight :
  initial_food_weight = 10 ∧
  weight_after_6_hours = 
    (initial_water_weight - water_consumption_rate * hiking_duration) +
    (initial_food_weight - food_consumption_rate * water_consumption_rate * hiking_duration) +
    gear_weight :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_karen_hike_food_weight_l1324_132484


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_limit_sqrt_minus_one_plus_x_over_x_l1324_132400

/-- The limit of (√(1-2x+x^2)-(1+x))/x as x approaches 0 is -2 -/
theorem limit_sqrt_minus_one_plus_x_over_x :
  ∀ ε > 0, ∃ δ > 0, ∀ x : ℝ, 0 < |x| ∧ |x| < δ →
    |((Real.sqrt (1 - 2*x + x^2) - (1 + x)) / x) + 2| < ε := by
  sorry

#check limit_sqrt_minus_one_plus_x_over_x

end NUMINAMATH_CALUDE_ERRORFEEDBACK_limit_sqrt_minus_one_plus_x_over_x_l1324_132400


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_grasshoppers_l1324_132488

/-- Represents the color of a cell on the checkerboard -/
inductive CellColor
| White
| Black

/-- Represents a position on the board -/
structure Position where
  row : Nat
  col : Nat

/-- Represents the board -/
def Board := Nat → Nat → CellColor

/-- Defines the checkerboard pattern -/
def checkerboardPattern : Board :=
  fun r c => if (r + c) % 2 = 0 then CellColor.White else CellColor.Black

/-- Defines the attack rule for grasshoppers -/
def isAttacking (board : Board) (p1 p2 : Position) : Prop :=
  (p1.row = p2.row ∧ board p1.row p1.col = board p2.row p2.col) ∨
  (p1.col = p2.col ∧ board p1.row p1.col ≠ board p2.row p2.col)

/-- Defines a valid placement of grasshoppers -/
def isValidPlacement (board : Board) (placement : List Position) : Prop :=
  ∀ p1 p2, p1 ∈ placement → p2 ∈ placement → p1 ≠ p2 → ¬isAttacking board p1 p2

/-- The main theorem stating the maximum number of non-attacking grasshoppers -/
theorem max_grasshoppers :
  ∃ (placement : List Position),
    placement.length = 4034 ∧
    isValidPlacement checkerboardPattern placement ∧
    ∀ (other : List Position),
      isValidPlacement checkerboardPattern other →
      other.length ≤ 4034 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_grasshoppers_l1324_132488


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_log_base_ratio_l1324_132480

theorem log_base_ratio (a b y : ℝ) (ha : a > 0 ∧ a ≠ 1) (hb : b > 0 ∧ b ≠ 1) (hy : y > 0)
  (h : 4 * (Real.log y / Real.log a)^2 + 5 * (Real.log y / Real.log b)^2 = 10 * (Real.log y)^2) :
  Real.log a / Real.log b = Real.sqrt (5/6) ∨ Real.log a / Real.log b = -Real.sqrt (5/6) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_log_base_ratio_l1324_132480


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_identity_function_theorem_l1324_132453

theorem identity_function_theorem :
  ∀ f : ℕ → ℕ, (∀ n : ℕ, f (f n) < f (n + 1)) → (∀ n : ℕ, f n = n) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_identity_function_theorem_l1324_132453


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l1324_132460

/-- The function f(x) = x^2 - ax ln x - 1 -/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := x^2 - a * x * Real.log x - 1

/-- Theorem stating the properties of function f -/
theorem f_properties (a : ℝ) (x₁ x₂ x₃ : ℝ) :
  (∀ x > 0, f a x + x^2 * f a (1/x) = 0) ∧
  (x₁ < x₂ ∧ x₂ < x₃ ∧ f a x₁ = 0 ∧ f a x₂ = 0 ∧ f a x₃ = 0) →
  (a > 2 ∧ x₁ + x₃ > 2*a - 2) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l1324_132460


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_a_value_x_minus_one_f_nonnegative_l1324_132457

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := (x + 1) * Real.log x - x + 1

-- Define the derivative of f
noncomputable def f_deriv (x : ℝ) : ℝ := Real.log x + 1 / x

-- Theorem for part I
theorem min_a_value (a : ℝ) : 
  (∀ x > 0, x * (f_deriv x) ≤ x^2 + a*x + 1) ↔ a ≥ -1 :=
sorry

-- Theorem for part II
theorem x_minus_one_f_nonnegative :
  ∀ x > 0, (x - 1) * f x ≥ 0 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_a_value_x_minus_one_f_nonnegative_l1324_132457


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_abc_value_l1324_132422

/-- Represents a number formed by digits a₁, a₂, ..., aₙ --/
def digit_number (digits : List Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * (10^(digits.length - 1 - i))) 0

/-- Given conditions for the problem --/
structure ProblemConditions where
  a : Nat
  b : Nat
  c : Nat
  a_in_range : a ∈ Finset.range 9
  b_in_range : b ∈ Finset.range 10
  c_in_range : c ∈ Finset.range 10
  ab_condition : digit_number [a, b] = b^2
  acbc_condition : digit_number [a, c, b, c] = (digit_number [b, a])^2

/-- The main theorem to prove --/
theorem abc_value (conditions : ProblemConditions) : 
  digit_number [conditions.a, conditions.b, conditions.c] = 369 := by
  sorry

#eval digit_number [3, 6, 9]

end NUMINAMATH_CALUDE_ERRORFEEDBACK_abc_value_l1324_132422


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_intersection_range_l1324_132493

/-- Parabola represented by the equation x^2 = 4y -/
def Parabola : Set (Real × Real) :=
  {p | p.1^2 = 4 * p.2}

/-- Focus of the parabola -/
def Focus : Real × Real := (0, 1)

/-- Point on the x-axis -/
def P (a : Real) : Real × Real := (a, 0)

/-- Line through two points -/
def Line (p q : Real × Real) : Set (Real × Real) :=
  {r | ∃ t : Real, r = p + t • (q - p)}

/-- Angle of inclination of a line through two points -/
noncomputable def AngleOfInclination (p q : Real × Real) : Real :=
  Real.arctan ((q.2 - p.2) / (q.1 - p.1))

/-- Two angles are complementary -/
def Complementary (α β : Real) : Prop :=
  α + β = Real.pi / 2

theorem parabola_intersection_range (a : Real) :
  (∃ A B : Real × Real, A ∈ Parabola ∧ B ∈ Parabola ∧
    A ∈ Line Focus B ∧
    Complementary (AngleOfInclination (P a) A) (AngleOfInclination (P a) B)) →
  -Real.sqrt 2 / 2 ≤ a ∧ a ≤ Real.sqrt 2 / 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_intersection_range_l1324_132493


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_buffered_decreasing_interval_l1324_132470

noncomputable section

-- Define the function f
def f (x : ℝ) : ℝ := (1/2) * x^2 - 2*x + 1

-- Define the buffered decreasing property
def is_buffered_decreasing (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  (∀ x y, a ≤ x ∧ x < y ∧ y ≤ b → f x > f y) ∧
  (∀ x y, a ≤ x ∧ x < y ∧ y ≤ b → (f x / x) < (f y / y))

-- State the theorem
theorem buffered_decreasing_interval :
  is_buffered_decreasing f (Real.sqrt 2) 2 ∧
  ∀ a b : ℝ, a < Real.sqrt 2 ∧ b > 2 →
    ¬(is_buffered_decreasing f a b) := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_buffered_decreasing_interval_l1324_132470


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_no_extreme_points_iff_a_in_range_l1324_132482

-- Define the function f(x) = (x - a)e^x
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (x - a) * Real.exp x

-- Define the property of having no extreme points in an interval
def no_extreme_points (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x ∈ Set.Ioo a b, (∀ y ∈ Set.Ioo a b, f y ≤ f x) ∨ (∀ y ∈ Set.Ioo a b, f y ≥ f x)

-- State the theorem
theorem no_extreme_points_iff_a_in_range (a : ℝ) :
  no_extreme_points (f a) 2 3 ↔ a ∈ Set.Iic 3 ∪ Set.Ici 4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_no_extreme_points_iff_a_in_range_l1324_132482


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_round_trip_speed_l1324_132454

/-- Calculates the return speed given the parameters of a round trip --/
noncomputable def return_speed (distance : ℝ) (outbound_speed : ℝ) (average_speed : ℝ) : ℝ :=
  (2 * distance * outbound_speed) / (4 * distance - outbound_speed * (2 * distance / average_speed))

/-- Theorem stating that for the given conditions, the return speed is 37.5 mph --/
theorem round_trip_speed : 
  let distance := (150 : ℝ)
  let outbound_speed := (75 : ℝ)
  let average_speed := (50 : ℝ)
  return_speed distance outbound_speed average_speed = 37.5 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_round_trip_speed_l1324_132454


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fourth_term_of_sequence_l1324_132449

def is_arithmetic_sequence (a : ℕ → ℤ) : Prop :=
  ∃ d : ℤ, ∀ n : ℕ, a (n + 1) - a n = d

theorem fourth_term_of_sequence :
  ∃ a : ℕ → ℤ, is_arithmetic_sequence a ∧
    (∀ n : ℕ, n < 3 → (a n)^2 - 2*(a n) - 3 < 0) ∧
    (∀ y : ℤ, y^2 - 2*y - 3 < 0 → ∃ n : ℕ, n < 3 ∧ y = a n) →
  ∃ a : ℕ → ℤ, is_arithmetic_sequence a ∧
    (∀ n : ℕ, n < 3 → (a n)^2 - 2*(a n) - 3 < 0) ∧
    (∀ y : ℤ, y^2 - 2*y - 3 < 0 → ∃ n : ℕ, n < 3 ∧ y = a n) ∧
    (a 3 = 3 ∨ a 3 = -1) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_fourth_term_of_sequence_l1324_132449


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_first_order_functions_l1324_132404

-- Define the concept of a first-order function
noncomputable def is_first_order (f : ℝ → ℝ) : Prop :=
  ∃ (k b : ℝ), k ≠ 0 ∧ (∀ x, f x = k * x + b)

-- Define the given functions
def f1 : ℝ → ℝ := λ x => -x
def f2 : ℝ → ℝ := λ x => 2 * x + 1
def f3 : ℝ → ℝ := λ x => x^2 + x - 1
noncomputable def f4 : ℝ → ℝ := λ x => 1 / (x + 2)

-- State the theorem
theorem first_order_functions :
  is_first_order f1 ∧ 
  is_first_order f2 ∧ 
  ¬is_first_order f3 ∧ 
  ¬is_first_order f4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_first_order_functions_l1324_132404


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotone_iff_a_le_half_l1324_132410

/-- The function f(x) defined as e^x - 1 - x - ax^2 -/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.exp x - 1 - x - a * x^2

/-- f is monotonically increasing on [0, ∞) -/
def is_monotone_increasing (a : ℝ) : Prop :=
  ∀ x y, 0 ≤ x → x ≤ y → f a x ≤ f a y

theorem f_monotone_iff_a_le_half :
  ∀ a : ℝ, is_monotone_increasing a ↔ a ≤ (1/2 : ℝ) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotone_iff_a_le_half_l1324_132410


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_faster_train_speed_l1324_132495

/-- Proves that the speed of the faster train is approximately 11.1 m/s given the problem conditions -/
theorem faster_train_speed (train_length : ℝ) (crossing_time : ℝ) (speed_ratio : ℝ) :
  train_length = 100 →
  crossing_time = 12 →
  speed_ratio = 2 →
  ∃ (faster_speed : ℝ),
    (faster_speed ≥ 11.0 ∧ faster_speed ≤ 11.2) ∧
    faster_speed = (2 * train_length) / (crossing_time * (1 + 1/speed_ratio)) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_faster_train_speed_l1324_132495


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_of_f_l1324_132434

open Real

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := sin x - cos x + x + 1

-- State the theorem
theorem max_value_of_f :
  ∃ (x : ℝ), x ∈ Set.Icc (3 * π / 4) (7 * π / 4) ∧
  f x = π + 2 ∧
  ∀ (y : ℝ), y ∈ Set.Icc (3 * π / 4) (7 * π / 4) → f y ≤ f x :=
by
  -- Proof goes here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_of_f_l1324_132434


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_cos_power_six_l1324_132419

theorem sin_cos_power_six (θ : Real) (h : Real.sin (2 * θ) = 1/4) : 
  (Real.sin θ) ^ 6 + (Real.cos θ) ^ 6 = 61/64 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_cos_power_six_l1324_132419


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_class_a_student_count_l1324_132401

theorem class_a_student_count :
  let art_students : Finset ℕ := Finset.range 35
  let music_students : Finset ℕ := Finset.range 32
  let both_classes : Finset ℕ := Finset.range 19
  let class_a : Finset ℕ := art_students ∪ music_students
  Finset.card class_a = 48 :=
by
  sorry

#check class_a_student_count

end NUMINAMATH_CALUDE_ERRORFEEDBACK_class_a_student_count_l1324_132401


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sector_half_triangle_area_l1324_132416

theorem sector_half_triangle_area (θ : Real) (h1 : 0 < θ) (h2 : θ < π/3) :
  (θ/2 = (Real.tan θ)/4) ↔ (Real.tan θ = 2*θ) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sector_half_triangle_area_l1324_132416


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l1324_132468

noncomputable def f (m : ℝ) (x : ℝ) : ℝ := (9 : ℝ)^x - 2 * (3 : ℝ)^(x + m)

noncomputable def g (m : ℝ) (x : ℝ) : ℝ := f m x + f m (-x)

theorem problem_solution :
  -- Part 1
  (∀ x : ℝ, f 1 x ≤ 27 ↔ x ≤ 2) ∧
  -- Part 2
  (∀ m : ℝ, m > 0 → ∀ x₁ x₂ : ℝ, x₂ > x₁ ∧ x₁ > 0 ∧ x₁ * x₂ = m^2 → f m x₂ > f m x₁) ∧
  -- Part 3
  (∃ m : ℝ, (∀ x : ℝ, g m x ≥ -11) ∧ (∃ x : ℝ, g m x = -11) → m = 1) := by
  sorry

#check problem_solution

end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l1324_132468


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_B_fourth_power_decomposition_l1324_132477

def B : Matrix (Fin 2) (Fin 2) ℝ := !![0, 1; 4, 5]

theorem B_fourth_power_decomposition :
  B^4 = 165 • B + 116 • (1 : Matrix (Fin 2) (Fin 2) ℝ) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_B_fourth_power_decomposition_l1324_132477


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_length_AE_l1324_132465

-- Define the points
variable (A B C D E : EuclideanSpace ℝ (Fin 2))

-- Define the distances
variable (hCA : ‖C - A‖ = 12)
variable (hAB : ‖A - B‖ = 8)
variable (hBC : ‖B - C‖ = 4)
variable (hCD : ‖C - D‖ = 5)
variable (hDB : ‖D - B‖ = 3)
variable (hBE : ‖B - E‖ = 6)
variable (hED : ‖E - D‖ = 3)

-- Theorem statement
theorem length_AE : ‖A - E‖ = Real.sqrt 113 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_length_AE_l1324_132465


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_midpoint_slope_zero_l1324_132496

-- Define the midpoint of a segment
def my_midpoint (x₁ y₁ x₂ y₂ : ℚ) : ℚ × ℚ :=
  ((x₁ + x₂) / 2, (y₁ + y₂) / 2)

-- Define the slope between two points
def my_slope (x₁ y₁ x₂ y₂ : ℚ) : ℚ :=
  if x₂ ≠ x₁ then (y₂ - y₁) / (x₂ - x₁) else 0

-- Theorem statement
theorem midpoint_slope_zero :
  let m₁ := my_midpoint 0 0 3 4
  let m₂ := my_midpoint 6 0 7 4
  my_slope m₁.1 m₁.2 m₂.1 m₂.2 = 0 := by
  sorry

#eval my_slope (my_midpoint 0 0 3 4).1 (my_midpoint 0 0 3 4).2 (my_midpoint 6 0 7 4).1 (my_midpoint 6 0 7 4).2

end NUMINAMATH_CALUDE_ERRORFEEDBACK_midpoint_slope_zero_l1324_132496


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_count_theorem_l1324_132408

/-- A regular polygon inscribed in a circle -/
structure InscribedPolygon where
  sides : ℕ
  vertices : Fin sides → ℝ × ℝ

/-- The set of inscribed polygons in the problem -/
def problem_polygons : Finset InscribedPolygon := sorry

/-- Two polygons do not share a vertex -/
def no_shared_vertices (p1 p2 : InscribedPolygon) : Prop := sorry

/-- No three sides intersect at a common point -/
def no_triple_intersections (ps : Finset InscribedPolygon) : Prop := sorry

/-- The number of intersection points between two polygons -/
def intersection_count (p1 p2 : InscribedPolygon) : ℕ := sorry

/-- The total number of intersection points for all polygon pairs -/
def total_intersections (ps : Finset InscribedPolygon) : ℕ := sorry

theorem intersection_count_theorem :
  ∀ ps : Finset InscribedPolygon,
  ps = problem_polygons →
  (∀ p1 p2, p1 ∈ ps → p2 ∈ ps → p1 ≠ p2 → no_shared_vertices p1 p2) →
  no_triple_intersections ps →
  total_intersections ps = 80 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_count_theorem_l1324_132408


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_isosceles_triangle_same_color_l1324_132435

/-- A color type representing red and blue --/
inductive Color
| Red
| Blue

/-- A vertex of a regular polygon --/
structure Vertex where
  index : Fin 13
  color : Color

/-- A regular 13-sided polygon --/
def RegularPolygon := Fin 13

/-- A coloring of the vertices of a regular 13-sided polygon --/
def Coloring := RegularPolygon → Color

/-- Three vertices form an isosceles triangle in a regular 13-sided polygon --/
def IsIsoscelesTriangle (v1 v2 v3 : RegularPolygon) : Prop :=
  ∃ (d1 d2 : Nat), d1 ≠ d2 ∧
    ((v2.val - v1.val + 13) % 13 = d1 ∧ (v3.val - v2.val + 13) % 13 = d2) ∨
    ((v3.val - v2.val + 13) % 13 = d1 ∧ (v1.val - v3.val + 13) % 13 = d2) ∨
    ((v1.val - v3.val + 13) % 13 = d1 ∧ (v2.val - v1.val + 13) % 13 = d2)

/-- Main theorem: In any coloring of a regular 13-sided polygon,
    there exist 3 vertices of the same color forming an isosceles triangle --/
theorem isosceles_triangle_same_color (c : Coloring) :
  ∃ (v1 v2 v3 : RegularPolygon),
    v1 ≠ v2 ∧ v2 ≠ v3 ∧ v1 ≠ v3 ∧
    c v1 = c v2 ∧ c v2 = c v3 ∧
    IsIsoscelesTriangle v1 v2 v3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_isosceles_triangle_same_color_l1324_132435


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_average_of_sequence_l1324_132441

def is_divisible_by_7 (n : ℕ) : Prop := n % 7 = 0

def sequence_sum (start : ℕ) (end_ : ℕ) : ℕ :=
  (end_ - start) / 7 + 1

theorem average_of_sequence (start end_ : ℕ) (h1 : start = 7) (h2 : end_ = 21) :
  (sequence_sum start end_ * (start + end_)) / (2 * sequence_sum start end_) = 15 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_average_of_sequence_l1324_132441


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_matrix_equation_solution_l1324_132426

def B : Matrix (Fin 3) (Fin 3) ℚ :=
  ![![1, 0, 3],
    ![0, 1, 0],
    ![3, 0, 1]]

theorem matrix_equation_solution :
  ∃ (p q r : ℚ), 
    B^3 + p • B^2 + q • B + r • (1 : Matrix (Fin 3) (Fin 3) ℚ) = 0 ∧ 
    p = -3 ∧ q = -6 ∧ r = 8 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_matrix_equation_solution_l1324_132426


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_non_coplanar_set_with_parallel_property_l1324_132430

/-- A point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Check if four points are coplanar -/
def coplanar (p q r s : Point3D) : Prop := sorry

/-- Check if two lines defined by four points are parallel -/
def parallel_lines (p q r s : Point3D) : Prop := sorry

/-- The main theorem stating the existence of the required set of points -/
theorem exists_non_coplanar_set_with_parallel_property : 
  ∃ (S : Set Point3D), 
    (S.Finite) ∧ 
    (∃ p q r s, p ∈ S ∧ q ∈ S ∧ r ∈ S ∧ s ∈ S ∧ ¬coplanar p q r s) ∧
    (∀ A B, A ∈ S → B ∈ S → A ≠ B → 
      ∃ C D, C ∈ S ∧ D ∈ S ∧ C ≠ D ∧ A ≠ C ∧ A ≠ D ∧ B ≠ C ∧ B ≠ D ∧ parallel_lines A B C D) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_non_coplanar_set_with_parallel_property_l1324_132430


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_z_coordinate_at_x_7_l1324_132450

/-- A line passing through two points in 3D space -/
structure Line3D where
  point1 : ℝ × ℝ × ℝ
  point2 : ℝ × ℝ × ℝ

/-- The z-coordinate of a point on the line given its x-coordinate -/
noncomputable def zCoordinate (l : Line3D) (x : ℝ) : ℝ :=
  let (x1, y1, z1) := l.point1
  let (x2, y2, z2) := l.point2
  let t := (x - x1) / (x2 - x1)
  z1 + t * (z2 - z1)

theorem z_coordinate_at_x_7 (l : Line3D) :
  l.point1 = (3, 3, 2) →
  l.point2 = (8, 2, -3) →
  zCoordinate l 7 = -2 := by
  sorry

#check z_coordinate_at_x_7

end NUMINAMATH_CALUDE_ERRORFEEDBACK_z_coordinate_at_x_7_l1324_132450
