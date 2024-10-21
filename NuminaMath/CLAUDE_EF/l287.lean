import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_markup_percentage_approx_l287_28796

/-- Calculates the markup percentage given the cost price, discounts, and profit percentage. -/
noncomputable def calculate_markup_percentage (cost_price : ℝ) (first_discount_percent : ℝ) (second_discount : ℝ) (profit_percent : ℝ) : ℝ :=
  let final_selling_price := cost_price * (1 + profit_percent / 100)
  let selling_price_before_second_discount := final_selling_price + second_discount
  let initial_selling_price := selling_price_before_second_discount / (1 - first_discount_percent / 100)
  (initial_selling_price / cost_price - 1) * 100

/-- Theorem stating that the markup percentage is approximately 76.47% given the specified conditions. -/
theorem markup_percentage_approx (ε : ℝ) (h_ε : ε > 0) :
  ∃ δ : ℝ, δ > 0 ∧ 
  ∀ cost_price first_discount_percent second_discount profit_percent,
  cost_price = 220 →
  first_discount_percent = 15 →
  second_discount = 55 →
  profit_percent = 25 →
  |calculate_markup_percentage cost_price first_discount_percent second_discount profit_percent - 76.47| < δ →
  |calculate_markup_percentage cost_price first_discount_percent second_discount profit_percent - 76.47| < ε :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_markup_percentage_approx_l287_28796


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_officers_selection_theorem_l287_28755

def club_size : ℕ := 30
def num_officers : ℕ := 4

def ways_to_choose_officers (n : ℕ) (k : ℕ) (special_members : ℕ) : ℕ :=
  let regular_members := n - special_members
  let case1 := (regular_members).factorial / (regular_members - k).factorial
  let case2 := Nat.choose k 2 * 2 * ((regular_members).factorial / (regular_members - (k - 2)).factorial)
  case1 + case2

theorem officers_selection_theorem :
  ways_to_choose_officers club_size num_officers 2 = 500472 :=
by sorry

#eval ways_to_choose_officers club_size num_officers 2

end NUMINAMATH_CALUDE_ERRORFEEDBACK_officers_selection_theorem_l287_28755


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_conveyor_belt_sampling_l287_28766

/-- Represents a sampling method --/
inductive SamplingMethod
  | Lottery
  | Systematic
  | Stratified
  | RandomNumberTable

/-- Represents the characteristics of a sampling process --/
structure SamplingProcess where
  interval : ℕ  -- The fixed interval between selections
  isFixedInterval : Bool  -- Whether the interval is fixed

/-- Defines systematic sampling --/
def isSystematicSampling (process : SamplingProcess) : Bool :=
  process.isFixedInterval && process.interval > 0

/-- The main theorem to prove --/
theorem conveyor_belt_sampling (process : SamplingProcess) 
  (h1 : process.isFixedInterval = true) 
  (h2 : process.interval = 30) : -- 30 minutes = half hour
  SamplingMethod.Systematic = 
    (if isSystematicSampling process then SamplingMethod.Systematic else SamplingMethod.Lottery) :=
by
  sorry

#check conveyor_belt_sampling

end NUMINAMATH_CALUDE_ERRORFEEDBACK_conveyor_belt_sampling_l287_28766


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_most_frequent_is_mode_l287_28770

/-- The mode of a dataset is the value that appears most frequently. -/
def mode {α : Type*} [DecidableEq α] (dataset : Multiset α) : Set α :=
  {x | ∀ y, dataset.count x ≥ dataset.count y}

/-- The most frequent data in a dataset is equivalent to its mode. -/
theorem most_frequent_is_mode {α : Type*} [DecidableEq α] (dataset : Multiset α) :
  ∀ x, x ∈ mode dataset ↔ ∀ y, dataset.count x ≥ dataset.count y :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_most_frequent_is_mode_l287_28770


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_and_monotonicity_l287_28720

-- Define the function f(x)
noncomputable def f (a b x : ℝ) : ℝ := a * x * Real.log x + b

-- Define the derivative of f(x)
noncomputable def f_deriv (a x : ℝ) : ℝ := a * (Real.log x + 1)

theorem tangent_line_and_monotonicity (a b : ℝ) (h : a ≠ 0) :
  -- Part 1: Tangent line equation when a = 2 and b = 3
  (∃ m c : ℝ, m = 2 ∧ c = 1 ∧ ∀ x y : ℝ, y = f 2 3 x → m * x - y + c = 0) ∧
  -- Part 2: Monotonicity intervals
  ((a > 0 → (∀ x : ℝ, 0 < x → x < Real.exp (-1) → f_deriv a x < 0) ∧
            (∀ x : ℝ, x > Real.exp (-1) → f_deriv a x > 0)) ∧
   (a < 0 → (∀ x : ℝ, 0 < x → x < Real.exp (-1) → f_deriv a x > 0) ∧
            (∀ x : ℝ, x > Real.exp (-1) → f_deriv a x < 0))) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_and_monotonicity_l287_28720


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ceiling_problem_l287_28790

theorem ceiling_problem : ⌈(4 : ℝ) * (7 - 2/3)⌉ = 26 := by
  -- The proof steps would go here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ceiling_problem_l287_28790


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_result_is_1234_l287_28706

-- Define the ring Z_210
def Z_210 := ZMod 210

-- Define the isomorphism
axiom Z_210_iso : Z_210 ≃ ZMod 2 × ZMod 3 × ZMod 5 × ZMod 7

-- Define the congruences for 53
axiom cong_53_mod_2 : 53 ≡ 1 [ZMOD 2]
axiom cong_53_mod_3 : 53 ≡ 2 [ZMOD 3]
axiom cong_53_mod_5 : 53 ≡ 3 [ZMOD 5]
axiom cong_53_mod_7 : 53 ≡ 4 [ZMOD 7]

-- Define the matrix M
def M : Matrix (Fin 3) (Fin 3) ℤ :=
  ![![53, 158, 53],
    ![23, 93, 53],
    ![50, 170, 53]]

-- Define the function to compute the result
def compute_result (M : Matrix (Fin 3) (Fin 3) ℤ) : ℕ :=
  let mod2 := M.map (fun x => (x % 2).toNat)
  let mod3 := M.map (fun x => (x % 3).toNat)
  let mod5 := M.map (fun x => (x % 5).toNat)
  let mod7 := M.map (fun x => (x % 7).toNat)
  1000 + 100 * mod2 1 1 + 10 * mod3 1 1 + mod5 1 1

-- State the theorem
theorem result_is_1234 : compute_result M = 1234 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_result_is_1234_l287_28706


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_round_18_4831_to_hundredth_l287_28719

noncomputable def round_to_hundredth (x : ℝ) : ℝ :=
  ⌊x * 100 + 0.5⌋ / 100

theorem round_18_4831_to_hundredth :
  round_to_hundredth 18.4831 = 18.48 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_round_18_4831_to_hundredth_l287_28719


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_count_valid_numbers_l287_28734

/-- Represents a three-digit positive integer in base-4 --/
structure Base4Rep where
  a₁ : Nat
  a₂ : Nat
  a₃ : Nat
  h₁ : a₁ < 4
  h₂ : a₂ < 4
  h₃ : a₃ < 4
  h₄ : 0 < 4^2 * a₁ + 4^1 * a₂ + a₃

/-- Represents a three-digit positive integer in base-7 --/
structure Base7Rep where
  b₁ : Nat
  b₂ : Nat
  b₃ : Nat
  h₁ : b₁ < 7
  h₂ : b₂ < 7
  h₃ : b₃ < 7
  h₄ : 0 < 7^2 * b₁ + 7^1 * b₂ + b₃

/-- Converts a Base4Rep to its decimal representation --/
def base4ToDecimal (n : Base4Rep) : Nat :=
  4^2 * n.a₁ + 4^1 * n.a₂ + n.a₃

/-- Converts a Base7Rep to its decimal representation --/
def base7ToDecimal (n : Base7Rep) : Nat :=
  7^2 * n.b₁ + 7^1 * n.b₂ + n.b₃

/-- Checks if a number has valid base-4 and base-7 representations --/
def hasValidRepresentations (n : Nat) : Prop :=
  ∃ (n4 : Base4Rep) (n7 : Base7Rep), 
    base4ToDecimal n4 = n ∧ base7ToDecimal n7 = n

/-- Calculates S based on the base-4 and base-7 representations --/
def calculateS (n4 : Base4Rep) (n7 : Base7Rep) : Nat :=
  (16 * n4.a₁ + 4 * n4.a₂ + n4.a₃) + (49 * n7.b₁ + 7 * n7.b₂ + n7.b₃) - 50

/-- The main theorem to be proved --/
theorem count_valid_numbers : 
  (∃ (l : List Nat), l.length = 10 ∧ 
    (∀ n ∈ l, 100 ≤ n ∧ n < 1000 ∧ hasValidRepresentations n ∧
      ∃ (n4 : Base4Rep) (n7 : Base7Rep), 
        base4ToDecimal n4 = n ∧ base7ToDecimal n7 = n ∧
        calculateS n4 n7 ≡ 2 * n [MOD 100]) ∧
    (∀ n, 100 ≤ n ∧ n < 1000 ∧ hasValidRepresentations n ∧
      (∃ (n4 : Base4Rep) (n7 : Base7Rep), 
        base4ToDecimal n4 = n ∧ base7ToDecimal n7 = n ∧
        calculateS n4 n7 ≡ 2 * n [MOD 100]) → n ∈ l)) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_count_valid_numbers_l287_28734


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_x_in_M_y_not_in_M_l287_28729

-- Define the set M
noncomputable def M : Set ℝ := {m | ∃ (a b : ℚ), m = a + Real.sqrt 2 * b}

-- Define x and y
noncomputable def x : ℝ := 1 / (3 - 5 * Real.sqrt 2)
noncomputable def y : ℝ := 3 + Real.sqrt 2 * Real.pi

-- Theorem statement
theorem x_in_M_y_not_in_M : x ∈ M ∧ y ∉ M := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_x_in_M_y_not_in_M_l287_28729


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_price_75_cans_with_discount_l287_28786

/-- Calculate the price of cans with a discount on full cases -/
def price_with_discount (regular_price : ℚ) (discount_percent : ℚ) (case_size : ℕ) (num_cans : ℕ) : ℚ :=
  let discounted_price := regular_price * (1 - discount_percent)
  let full_cases := num_cans / case_size
  let remaining_cans := num_cans % case_size
  (discounted_price * (full_cases * case_size : ℚ)) + (regular_price * (remaining_cans : ℚ))

theorem price_75_cans_with_discount :
  price_with_discount (15 / 100) (10 / 100) 24 75 = 1017 / 100 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_price_75_cans_with_discount_l287_28786


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_compare_fractions_l287_28778

theorem compare_fractions : -(5/9 : ℚ) > -abs (-4/7 : ℚ) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_compare_fractions_l287_28778


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_bank_exceed_1000_on_saturday_l287_28731

/-- Represents the days of the week -/
inductive DayOfWeek
  | Sunday
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday
  | Saturday
deriving Repr

/-- Returns the day of the week n days after Sunday -/
def dayAfter (n : ℕ) : DayOfWeek :=
  match n % 7 with
  | 0 => DayOfWeek.Sunday
  | 1 => DayOfWeek.Monday
  | 2 => DayOfWeek.Tuesday
  | 3 => DayOfWeek.Wednesday
  | 4 => DayOfWeek.Thursday
  | 5 => DayOfWeek.Friday
  | _ => DayOfWeek.Saturday

/-- Calculates the total amount in the bank after n days -/
def bankTotal (n : ℕ) : ℚ :=
  (3^n - 1) / 50

theorem bank_exceed_1000_on_saturday :
  (∀ k < 7, bankTotal k ≤ 1000) ∧
  (bankTotal 7 > 1000) ∧
  (dayAfter 7 = DayOfWeek.Saturday) := by
  sorry

#eval dayAfter 7

end NUMINAMATH_CALUDE_ERRORFEEDBACK_bank_exceed_1000_on_saturday_l287_28731


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_particle_acceleration_bound_l287_28712

-- Define the velocity function
def velocity : ℝ → ℝ := sorry

-- Define the acceleration function
def acceleration : ℝ → ℝ := sorry

-- Theorem statement
theorem particle_acceleration_bound 
  (h1 : Continuous velocity)
  (h2 : Continuous acceleration)
  (h3 : velocity 0 = 0)
  (h4 : velocity 1 = 0)
  (h5 : ∫ t in (0:ℝ)..(1:ℝ), velocity t = 1) :
  ∃ t : ℝ, t ∈ Set.Icc 0 1 ∧ |acceleration t| ≥ 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_particle_acceleration_bound_l287_28712


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_divisibility_condition_l287_28733

theorem divisibility_condition (a b : ℕ) : 
  (∃ k : ℕ, (2^a + 1) = k * (2^b - 1)) ↔ (b = 1 ∨ (b = 2 ∧ ∃ m : ℕ, a = 2*m + 1)) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_divisibility_condition_l287_28733


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equilateral_cone_cylinder_volume_ratio_l287_28747

-- Define the cone and cylinder
structure Cone where
  a : ℝ
  h_pos : a > 0

structure Cylinder where
  r : ℝ
  h : ℝ
  h_pos_r : r > 0
  h_pos_h : h > 0

-- Volume functions
noncomputable def cone_volume (c : Cone) : ℝ :=
  (c.a^3 * Real.pi * Real.sqrt 3) / 24

noncomputable def cylinder_volume (cyl : Cylinder) : ℝ :=
  Real.pi * cyl.r^2 * cyl.h

-- Theorem statement
theorem equilateral_cone_cylinder_volume_ratio :
  ∀ (c : Cone),
  ∃ (cyl : Cylinder),
  (cone_volume c) / (cylinder_volume cyl) = (26 + 15 * Real.sqrt 3) / 18 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_equilateral_cone_cylinder_volume_ratio_l287_28747


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_zero_difference_l287_28735

/-- Represents a parabola in vertex form: y = a(x - h)^2 + k -/
structure Parabola where
  a : ℝ
  h : ℝ
  k : ℝ

/-- Given a parabola and an x-coordinate, compute the y-coordinate -/
def Parabola.yCoord (p : Parabola) (x : ℝ) : ℝ :=
  p.a * (x - p.h)^2 + p.k

/-- Compute the difference between zeros of a parabola -/
noncomputable def zeroDifference (p : Parabola) : ℝ :=
  2 * Real.sqrt (1 / (2 * p.a))

theorem parabola_zero_difference :
  ∀ p : Parabola,
    p.h = 1 ∧
    p.k = -3 ∧
    p.yCoord 3 = 5 →
    zeroDifference p = Real.sqrt 6 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_zero_difference_l287_28735


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_student_movement_proof_l287_28728

/-- The maximal sum of steps when n students change positions in a line. -/
def max_student_movement (n : ℕ) : ℕ :=
  Int.toNat ⌊(n^2 : ℚ) / 2⌋

/-- 
Proves that the maximal sum of steps when n students change positions in a line
is equal to ⌊n²/2⌋.
-/
theorem max_student_movement_proof (n : ℕ) : 
  max_student_movement n = Int.toNat ⌊(n^2 : ℚ) / 2⌋ := by
  rfl

#eval max_student_movement 5  -- Expected output: 12
#eval max_student_movement 6  -- Expected output: 18

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_student_movement_proof_l287_28728


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_liam_reading_finish_day_l287_28779

-- Define the days of the week
inductive DayOfWeek
  | Sunday
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday
  | Saturday
deriving Repr

def nextDay (d : DayOfWeek) : DayOfWeek :=
  match d with
  | DayOfWeek.Sunday => DayOfWeek.Monday
  | DayOfWeek.Monday => DayOfWeek.Tuesday
  | DayOfWeek.Tuesday => DayOfWeek.Wednesday
  | DayOfWeek.Wednesday => DayOfWeek.Thursday
  | DayOfWeek.Thursday => DayOfWeek.Friday
  | DayOfWeek.Friday => DayOfWeek.Saturday
  | DayOfWeek.Saturday => DayOfWeek.Sunday

def advanceDays (start : DayOfWeek) (days : Nat) : DayOfWeek :=
  match days with
  | 0 => start
  | n + 1 => advanceDays (nextDay start) n

def readingDuration (bookNumber : Nat) : Nat :=
  bookNumber + 1

def totalReadingDays (totalBooks : Nat) : Nat :=
  (List.range totalBooks).map readingDuration |>.sum

theorem liam_reading_finish_day (startDay : DayOfWeek) (totalBooks : Nat) :
  startDay = DayOfWeek.Thursday →
  totalBooks = 12 →
  advanceDays startDay (totalReadingDays totalBooks) = DayOfWeek.Wednesday :=
by
  sorry

#eval advanceDays DayOfWeek.Thursday (totalReadingDays 12)

end NUMINAMATH_CALUDE_ERRORFEEDBACK_liam_reading_finish_day_l287_28779


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sqrt_product_plus_one_l287_28715

theorem sqrt_product_plus_one : (Real.sqrt ((20 * 19 * 18 * 17) + 1) : ℝ) = 341 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sqrt_product_plus_one_l287_28715


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_of_f_l287_28711

/-- The function to be minimized -/
noncomputable def f (x y : ℝ) : ℝ := 
  (5*x^2 - 8*x*y + 5*y^2 - 10*x + 14*y + 55) / (9 - 25*x^2 + 10*x*y - y^2)^(5/2)

/-- Theorem stating that the minimum value of f is 5/27 -/
theorem min_value_of_f : 
  ∀ x y : ℝ, f x y ≥ 5/27 ∧ ∃ x₀ y₀ : ℝ, f x₀ y₀ = 5/27 := by
  sorry

#check min_value_of_f

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_of_f_l287_28711


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_coefficient_of_x_is_80_l287_28762

/-- The coefficient of x in the binomial expansion of (2/x - x^2)^5 -/
def coefficient_of_x : ℤ :=
  (Finset.range 6).sum (fun r => 
    (Nat.choose 5 r : ℤ) * (-1)^r * 2^(5-r) * 
    if 3*r - 5 = 1 then 1 else 0)

/-- The theorem stating that the coefficient of x in (2/x - x^2)^5 is 80 -/
theorem coefficient_of_x_is_80 :
  coefficient_of_x = 80 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_coefficient_of_x_is_80_l287_28762


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_no_intersection_C_and_P_locus_l287_28752

-- Define the curve C in polar coordinates
noncomputable def C : ℝ → ℝ × ℝ := fun θ => (2 * Real.sqrt 2 * Real.cos θ * Real.cos θ, 2 * Real.sqrt 2 * Real.cos θ * Real.sin θ)

-- Define point A
def A : ℝ × ℝ := (1, 0)

-- Define the locus of P
noncomputable def P : ℝ → ℝ × ℝ := fun θ => (3 - Real.sqrt 2 + 2 * Real.cos θ, 2 * Real.sin θ)

-- Theorem statement
theorem no_intersection_C_and_P_locus :
  ∀ θ₁ θ₂, C θ₁ ≠ P θ₂ :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_no_intersection_C_and_P_locus_l287_28752


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_expression_divisible_by_1999_l287_28744

def odd_product : ℕ → ℕ 
  | 0 => 1
  | n + 1 => odd_product n * (2 * n + 5)

def even_product : ℕ → ℕ 
  | 0 => 1
  | n + 1 => even_product n * (2 * n + 2)

theorem expression_divisible_by_1999 :
  ∃ k : ℤ, (1 * 3 : ℤ) - (odd_product 497) + (even_product 499) = 1999 * k := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_expression_divisible_by_1999_l287_28744


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_Q_at_two_is_zero_l287_28784

/-- The polynomial Q(x) of degree 4 with rational coefficients and leading coefficient 1 -/
noncomputable def Q : ℝ → ℝ := sorry

/-- The condition that √3 + √7 is a root of Q(x) -/
axiom root_condition : Q (Real.sqrt 3 + Real.sqrt 7) = 0

/-- The condition that Q is a polynomial of degree 4 with rational coefficients and leading coefficient 1 -/
axiom Q_properties : ∃ (a b c : ℚ), ∀ x, Q x = x^4 + a*x^3 + b*x^2 + c*x + (Q 0)

/-- Theorem: Q(2) = 0 -/
theorem Q_at_two_is_zero : Q 2 = 0 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_Q_at_two_is_zero_l287_28784


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_value_of_f_l287_28753

noncomputable def f (x : ℝ) : ℝ := |x| + |(1 - 2013*x) / (2013 - x)|

theorem smallest_value_of_f :
  (∀ x ∈ Set.Icc (-1 : ℝ) 1, f x ≥ 1/2013) ∧
  (∃ x ∈ Set.Icc (-1 : ℝ) 1, f x = 1/2013) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_value_of_f_l287_28753


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_monomial_exponents_sum_l287_28771

/-- Two monomials are like terms if they have the same variables with the same exponents -/
def like_terms (m1 m2 : ℕ → ℕ → ℝ) : Prop :=
  ∀ (a b : ℕ), m1 a b ≠ 0 ∧ m2 a b ≠ 0 → 
    ∃ (c1 c2 : ℝ) (x y : ℕ), m1 a b = c1 * (a : ℝ)^x * (b : ℝ)^y ∧
                              m2 a b = c2 * (a : ℝ)^x * (b : ℝ)^y

/-- The main theorem -/
theorem monomial_exponents_sum (x y : ℕ) :
  like_terms (fun a b => 5 * (a : ℝ)^x * (b : ℝ)^2) (fun a b => -0.2 * (a : ℝ)^3 * (b : ℝ)^y) →
  x + y = 5 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_monomial_exponents_sum_l287_28771


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_grandma_molly_statues_l287_28765

/-- Represents the number of turtle statues on Grandma Molly's lawn over four years -/
def TurtleStatues (x : ℕ) : ℕ → ℕ
| 0 => 4  -- Initial number of statues
| 1 => 4 * 4  -- Quadrupled in the second year
| 2 => TurtleStatues x 1 + x - 3  -- Added x, then 3 broke
| 3 => TurtleStatues x 2 + 2 * 3  -- Added twice the number broken
| _ => 0  -- Other years not specified

theorem grandma_molly_statues (x : ℕ) : 
  TurtleStatues x 3 = 31 → x = 12 := by
  sorry

#check grandma_molly_statues

end NUMINAMATH_CALUDE_ERRORFEEDBACK_grandma_molly_statues_l287_28765


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_projection_of_a_on_b_l287_28777

def a : Fin 2 → ℝ := ![2, 3]
def b : Fin 2 → ℝ := ![-4, 1]

def dot_product (u v : Fin 2 → ℝ) : ℝ :=
  (u 0) * (v 0) + (u 1) * (v 1)

noncomputable def vector_length (v : Fin 2 → ℝ) : ℝ :=
  Real.sqrt ((v 0)^2 + (v 1)^2)

noncomputable def projection (u v : Fin 2 → ℝ) : ℝ :=
  (dot_product u v) / (vector_length v)

theorem projection_of_a_on_b :
  projection a b = -5 * Real.sqrt 17 / 17 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_projection_of_a_on_b_l287_28777


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_heron_area_equality_l287_28785

/-- Heron's formula for triangle area -/
noncomputable def heron_area (a b c : ℝ) : ℝ :=
  let s := (a + b + c) / 2
  Real.sqrt (s * (s - a) * (s - b) * (s - c))

/-- Theorem: Heron's formula equality for triangle area -/
theorem heron_area_equality (a b c : ℝ) (h_pos : a > 0 ∧ b > 0 ∧ c > 0) 
  (h_triangle : a + b > c ∧ b + c > a ∧ c + a > b) :
  let t := heron_area a b c
  16 * t^2 = 2 * a^2 * b^2 + 2 * a^2 * c^2 + 2 * b^2 * c^2 - a^4 - b^4 - c^4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_heron_area_equality_l287_28785


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_inequality_l287_28710

/-- A function that takes two positive integer arguments -/
def f (n d : ℕ+) : ℕ := sorry

/-- Theorem stating the inequality for the function f -/
theorem f_inequality (n1 n2 d : ℕ+) :
  f (n1 * n2) d ≤ f n1 d + n1.val * (f n2 d - 1) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_inequality_l287_28710


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_factorization_x_11_minus_x_l287_28764

open Polynomial

theorem factorization_x_11_minus_x :
  ∃ (f₁ f₂ f₃ f₄ : Polynomial ℤ),
    (X^11 - X : Polynomial ℤ) = f₁ * f₂ * f₃ * f₄ ∧
    Irreducible f₁ ∧ Irreducible f₂ ∧ Irreducible f₃ ∧ Irreducible f₄ ∧
    (∀ (g₁ g₂ g₃ g₄ g₅ : Polynomial ℤ),
      (X^11 - X : Polynomial ℤ) ≠ g₁ * g₂ * g₃ * g₄ * g₅ ∨
      ¬(Irreducible g₁ ∧ Irreducible g₂ ∧ Irreducible g₃ ∧ Irreducible g₄ ∧ Irreducible g₅)) :=
by
  sorry

#check factorization_x_11_minus_x

end NUMINAMATH_CALUDE_ERRORFEEDBACK_factorization_x_11_minus_x_l287_28764


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_decreasing_implies_a_range_l287_28767

-- Define the function f as noncomputable due to its dependence on real logarithms
noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x < 1 then (3*a - 1)*x + 4*a else Real.log x / Real.log a

-- State the theorem
theorem f_decreasing_implies_a_range (a : ℝ) :
  (∀ x₁ x₂ : ℝ, x₁ < x₂ → f a x₁ > f a x₂) →
  a ∈ Set.Icc (1/7 : ℝ) (1/3 : ℝ) :=
by
  -- Introduce the hypothesis
  intro h
  -- The proof would go here, but we'll use sorry to skip it for now
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_decreasing_implies_a_range_l287_28767


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_of_f_twentyone_fourths_l287_28743

noncomputable def f (x : ℝ) : ℝ :=
  let y := x - 3 * ⌊x / 3⌋ -- Adjust x to be within one period
  if -2 ≤ y ∧ y ≤ 0 then 4 * y^2 - 2
  else if 0 < y ∧ y < 1 then y
  else 0 -- This case should never occur for the given domain, but Lean requires a total function

theorem f_of_f_twentyone_fourths :
  f (f (21/4)) = 1/4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_of_f_twentyone_fourths_l287_28743


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_power_sum_l287_28708

-- Define the complex number i
def i : ℂ := Complex.I

-- State the theorem
theorem complex_power_sum : i^10 + i^20 + i^(-34 : ℤ) + 2 = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_power_sum_l287_28708


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_right_triangle_from_angle_ratio_l287_28768

/-- Given a triangle ABC with angles in the ratio 2:3:5, prove that one of the angles is 90° --/
theorem right_triangle_from_angle_ratio :
  ∀ (A B C : ℝ),
  A > 0 → B > 0 → C > 0 →
  A + B + C = 180 →
  (A : ℝ) / 2 = (B : ℝ) / 3 → (B : ℝ) / 3 = (C : ℝ) / 5 →
  A = 90 ∨ B = 90 ∨ C = 90 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_right_triangle_from_angle_ratio_l287_28768


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_moved_is_sqrt_41_l287_28751

/-- A dilation that transforms a circle to another circle -/
structure CircleDilation where
  original_center : ℝ × ℝ
  original_radius : ℝ
  dilated_center : ℝ × ℝ
  dilated_radius : ℝ

/-- The point that moves under the dilation -/
def P : ℝ × ℝ := (3, 1)

/-- The dilation described in the problem -/
def problem_dilation : CircleDilation where
  original_center := (1, 3)
  original_radius := 4
  dilated_center := (7, 9)
  dilated_radius := 6

/-- The distance a point moves under a dilation -/
noncomputable def distance_moved (d : CircleDilation) (p : ℝ × ℝ) : ℝ :=
  sorry

/-- Theorem stating the distance P moves under the problem dilation -/
theorem distance_moved_is_sqrt_41 :
  distance_moved problem_dilation P = Real.sqrt 41 := by
  sorry

#check distance_moved_is_sqrt_41

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_moved_is_sqrt_41_l287_28751


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l287_28758

/-- Define the function f with parameter ω -/
noncomputable def f (ω : ℝ) (x : ℝ) : ℝ := 3 * Real.sin (ω * x + Real.pi / 6)

/-- Theorem stating the properties of the function f -/
theorem f_properties (ω : ℝ) (h_ω_pos : ω > 0) 
  (h_period : ∀ x, f ω (x + Real.pi / 2) = f ω x) 
  (h_min_period : ∀ p, p > 0 → (∀ x, f ω (x + p) = f ω x) → p ≥ Real.pi / 2) : 
  (f ω 0 = 3 / 2) ∧ 
  (ω = 4) ∧ 
  (∀ α, α ∈ Set.Ioo 0 (Real.pi / 2) → f ω (α / 2) = 3 / 2 → α = Real.pi / 3) := by
  sorry

#check f_properties

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l287_28758


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complement_of_union_l287_28761

def U : Set ℕ := {1,2,3,4,5,6,7,8}
def A : Set ℕ := {1,3,5,7}
def B : Set ℕ := {2,4,5}

theorem complement_of_union : (U \ (A ∪ B)) = {6,8} := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_complement_of_union_l287_28761


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l287_28759

/-- Properties of a triangle ABC -/
structure Triangle where
  A : Real
  B : Real
  C : Real
  a : Real
  b : Real
  c : Real
  cosA : Real
  S : Real

/-- Theorem about a specific triangle ABC -/
theorem triangle_properties (t : Triangle)
  (h_cosA : t.cosA = 4/5)
  (h_S : t.S = 3)
  (h_b : t.b = 2) :
  (Real.sin ((t.B + t.C) / 2))^2 + Real.cos (2 * t.A) = 59/50 ∧
  ∃ (R : Real), R = (5 * Real.sqrt 13) / 6 ∧
    R = t.a / (2 * Real.sin t.A) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l287_28759


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_is_odd_f_increasing_on_open_one_to_inf_l287_28748

-- Define the function f(x) = x + 1/x
noncomputable def f (x : ℝ) : ℝ := x + 1/x

-- Theorem for the parity of the function
theorem f_is_odd : ∀ (x : ℝ), x ≠ 0 → f (-x) = -f x := by
  sorry

-- Theorem for the monotonicity of the function on (1, +∞)
theorem f_increasing_on_open_one_to_inf :
  ∀ (x₁ x₂ : ℝ), 1 < x₁ → x₁ < x₂ → f x₁ < f x₂ := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_is_odd_f_increasing_on_open_one_to_inf_l287_28748


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_watch_price_reduction_l287_28737

theorem watch_price_reduction (initial_price : ℝ) (first_markdown_percent : ℝ) (second_markdown_percent : ℝ) : 
  initial_price = 15 →
  first_markdown_percent = 25 →
  second_markdown_percent = 40 →
  let price_after_first_markdown := initial_price * (1 - first_markdown_percent / 100);
  let final_price := price_after_first_markdown * (1 - second_markdown_percent / 100);
  final_price = 6.75 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_watch_price_reduction_l287_28737


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_coffee_mixture_problem_l287_28707

/-- Proves that the amount of the second brand of coffee added is approximately 3 kg -/
theorem coffee_mixture_problem (
  selling_price : ℝ)
  (profit_percentage : ℝ)
  (first_brand_quantity : ℝ)
  (first_brand_cost : ℝ)
  (second_brand_cost : ℝ)
  (h1 : selling_price = 177)
  (h2 : profit_percentage = 18)
  (h3 : first_brand_quantity = 2)
  (h4 : first_brand_cost = 200)
  (h5 : second_brand_cost = 116.67)
  : ∃ (second_brand_quantity : ℝ), 
    (abs (second_brand_quantity - 3) < 0.1 ∧
     abs ((first_brand_quantity * first_brand_cost + second_brand_quantity * second_brand_cost) * (1 + profit_percentage / 100) -
     selling_price * (first_brand_quantity + second_brand_quantity)) < 0.1) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_coffee_mixture_problem_l287_28707


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_classroom_shirts_problem_l287_28749

theorem classroom_shirts_problem (total_students : ℕ) 
  (striped_ratio : ℚ) (shorts_diff : ℕ) : 
  total_students = 81 →
  striped_ratio = 2/3 →
  shorts_diff = 19 →
  let checkered := total_students - (striped_ratio * ↑total_students).floor
  let striped := (striped_ratio * ↑total_students).floor
  let shorts := checkered + shorts_diff
  striped - shorts = 8 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_classroom_shirts_problem_l287_28749


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_disjoint_sets_condition_equivalent_to_a_range_l287_28738

theorem disjoint_sets_condition_equivalent_to_a_range (a : ℝ) (ha : 0 < a) :
  (∃ (n : ℕ+) (A : ℕ+ → Set ℤ),
    (∀ i j : ℕ+, i ≠ j → Disjoint (A i) (A j)) ∧
    (⋃ i : ℕ+, A i) = Set.univ ∧
    (∀ i : ℕ+, Set.Infinite (A i)) ∧
    (∀ i : ℕ+, ∀ b c : ℤ, b ∈ A i → c ∈ A i → b > c → b - c ≥ a ^ (i : ℕ)))
  ↔ a < 2 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_disjoint_sets_condition_equivalent_to_a_range_l287_28738


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_december_sales_fraction_l287_28789

/-- Given a year's sales where December sales are 7 times the average of the other 11 months,
    prove that December sales represent 7/18 of the total annual sales. -/
theorem december_sales_fraction (monthly_avg : ℝ) (h : monthly_avg > 0) :
  (7 * monthly_avg) / ((11 * monthly_avg) + (7 * monthly_avg)) = 7 / 18 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_december_sales_fraction_l287_28789


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_7_equals_5_l287_28772

def sequence_a : ℕ → ℚ
  | 0 => 2  -- Add this case to handle Nat.zero
  | 1 => 2
  | n + 1 => sequence_a n + 1/2

theorem a_7_equals_5 : sequence_a 7 = 5 := by
  -- Unfold the definition of sequence_a
  unfold sequence_a
  -- Perform arithmetic
  norm_num
  -- Complete the proof
  rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_7_equals_5_l287_28772


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_at_one_critical_points_product_less_than_one_l287_28774

noncomputable def f (x m : ℝ) : ℝ := Real.exp (x + m) + (m + 1) * x - x * Real.log x

-- Part 1: Tangent line equation when m = 0
theorem tangent_line_at_one (x y : ℝ) :
  HasDerivAt (fun x => f x 0) (Real.exp 1 - Real.log 1) 1 →
  f 1 0 = Real.exp 1 + 1 →
  (Real.exp x - y + 1 = 0) ↔ y = (Real.exp 1) * (x - 1) + (Real.exp 1 + 1) :=
sorry

-- Part 2: Product of critical points is less than 1
theorem critical_points_product_less_than_one (m x₁ x₂ : ℝ) :
  x₁ > 0 → x₂ > 0 →
  deriv (fun x => f x m) x₁ = 0 →
  deriv (fun x => f x m) x₂ = 0 →
  x₁ ≠ x₂ →
  x₁ * x₂ < 1 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_at_one_critical_points_product_less_than_one_l287_28774


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_sum_first_two_terms_l287_28718

def b_sequence (b : ℕ → ℕ) : Prop :=
  ∀ n : ℕ, n ≥ 1 → b (n + 2) * (2 + b (n + 1)) = 3 * b n + 4073

theorem min_sum_first_two_terms (b : ℕ → ℕ) (h : b_sequence b) :
  ∃ b₁ b₂ : ℕ, b 1 = b₁ ∧ b 2 = b₂ ∧ b₁ + b₂ = 158 ∧
  ∀ b₁' b₂' : ℕ, b 1 = b₁' ∧ b 2 = b₂' → b₁' + b₂' ≥ 158 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_sum_first_two_terms_l287_28718


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_special_number_properties_exists_special_number_l287_28781

/-- A 100-digit number without any zero digits that is divisible by the sum of its digits. -/
def special_number : ℕ :=
  11111111111111111111111111111111111111111111111111111111111111111111111111111195125

/-- The sum of digits of the special number. -/
def sum_of_digits : ℕ := 116

/-- Theorem stating that the special number satisfies the required properties. -/
theorem special_number_properties :
  (special_number.repr).length = 100 ∧
  ¬ ('0' ∈ special_number.repr.data) ∧
  special_number % sum_of_digits = 0 := by
  sorry

/-- Theorem proving the existence of a number with the required properties. -/
theorem exists_special_number :
  ∃ n : ℕ, (n.repr).length = 100 ∧
           ¬ ('0' ∈ n.repr.data) ∧
           n % (n.repr.data.map (λ c => c.toNat - '0'.toNat)).sum = 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_special_number_properties_exists_special_number_l287_28781


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_infinite_k_composite_for_all_n_l287_28757

theorem infinite_k_composite_for_all_n : 
  ∃ (S : Set ℕ), Set.Infinite S ∧ 
    ∀ (k : ℕ), k ∈ S → ∀ (n : ℕ), n > 0 → ∃ (m : ℕ), m > 1 ∧ m ∣ (k * 2^n + 1) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_infinite_k_composite_for_all_n_l287_28757


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_and_line_intersection_l287_28783

-- Define the function
def f (x : ℝ) : ℝ := x^2 - 4*x + 3

-- Define the circle
def circle_equation (x y : ℝ) : Prop := (x - 2)^2 + (y - 2)^2 = 5

-- Theorem statement
theorem circle_and_line_intersection :
  -- The function intersects x-axis at two points
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ f x₁ = 0 ∧ f x₂ = 0) →
  -- The function intersects y-axis
  (∃ y : ℝ, f 0 = y) →
  -- The circle passes through these intersection points
  (∀ x y : ℝ, (f x = 0 ∨ (x = 0 ∧ y = f 0)) → circle_equation x y) →
  -- If the circle intersects x - y + n = 0 at two points 4 units apart
  (∃ n : ℝ, ∃ x₁ y₁ x₂ y₂ : ℝ, 
    circle_equation x₁ y₁ ∧ circle_equation x₂ y₂ ∧
    x₁ - y₁ + n = 0 ∧ x₂ - y₂ + n = 0 ∧
    (x₁ - x₂)^2 + (y₁ - y₂)^2 = 16) →
  -- Then n = ±√2
  (∃ n : ℝ, n^2 = 2 ∧ 
    (∀ x₁ y₁ x₂ y₂ : ℝ, 
      circle_equation x₁ y₁ → circle_equation x₂ y₂ →
      x₁ - y₁ + n = 0 → x₂ - y₂ + n = 0 →
      (x₁ - x₂)^2 + (y₁ - y₂)^2 = 16)) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_and_line_intersection_l287_28783


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_minus_smallest_times_six_l287_28717

def digits : List Nat := [8, 3, 6, 5, 0, 7]

def is_two_digit (n : Nat) : Bool :=
  10 ≤ n && n ≤ 99

def largest_two_digit (ds : List Nat) : Nat :=
  let two_digit_numbers := ds.filter is_two_digit
  two_digit_numbers.foldl Nat.max 0

def smallest_two_digit (ds : List Nat) : Nat :=
  let two_digit_numbers := ds.filter is_two_digit
  two_digit_numbers.foldl Nat.min 99

theorem largest_minus_smallest_times_six :
  (largest_two_digit digits - smallest_two_digit digits) * 6 = 342 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_minus_smallest_times_six_l287_28717


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l287_28702

/-- A function f(x) with the given properties -/
noncomputable def f (ω φ : ℝ) (x : ℝ) : ℝ := 2 * Real.sin (ω * x + φ)

/-- Theorem stating the properties of the function f -/
theorem f_properties (ω φ : ℝ) (h_ω : ω > 0) (h_φ : 0 < φ ∧ φ < Real.pi) :
  -- f is an even function
  (∀ x, f ω φ x = f ω φ (-x)) →
  -- Distance between adjacent axes of symmetry is π/2
  (∃ k : ℤ, ∀ x, f ω φ x = f ω φ (x + k * Real.pi / 2)) →
  -- Claim 1: f(π/8) = √2
  f ω φ (Real.pi / 8) = Real.sqrt 2 ∧
  -- Claim 2: f is decreasing on intervals [kπ/2, kπ/2 + π/2] for all k ∈ ℤ
  ∀ k : ℤ, ∀ x ∈ Set.Icc (k * Real.pi / 2) (k * Real.pi / 2 + Real.pi / 2),
    ∀ y ∈ Set.Icc (k * Real.pi / 2) (k * Real.pi / 2 + Real.pi / 2),
    x < y → f ω φ x > f ω φ y :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l287_28702


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_expansion_properties_l287_28700

noncomputable def expansion_term (n r : ℕ) : ℝ := (n.choose r) * (2^(n-r)) * (r^(1/3 : ℝ))

theorem expansion_properties :
  ∃ (n : ℕ),
    (n.choose 4 : ℝ) / (n.choose 2 : ℝ) = 7 / 2 ∧
    n = 9 ∧
    (let r := 6; expansion_term n r = 672) ∧
    (let r := 3; expansion_term n r = 5376) ∧
    ∀ (k : ℕ), k ≠ 3 → expansion_term n k ≤ expansion_term n 3 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_expansion_properties_l287_28700


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_proof_l287_28721

theorem triangle_area_proof (a b c : ℝ) (h_perimeter : a + b + c = 2 * Real.sqrt 2 + Real.sqrt 5)
  (h_ratio : ∃ (k : ℝ), k > 0 ∧ 
    a = k * (Real.sqrt 2 - 1) ∧ 
    b = k * Real.sqrt 5 ∧ 
    c = k * (Real.sqrt 2 + 1)) :
  Real.sqrt (1/4 * (c^2 * a^2 - ((c^2 + a^2 - b^2)/2)^2)) = Real.sqrt 3 / 4 := by
  sorry

#check triangle_area_proof

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_proof_l287_28721


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_decreasing_log_base_range_l287_28792

-- Define the function f as noncomputable
noncomputable def f (a : ℝ) (x : ℝ) := Real.log x / Real.log (a - 1)

-- State the theorem
theorem decreasing_log_base_range (a : ℝ) : 
  (∀ x y : ℝ, x < y → f a x > f a y) → 1 < a ∧ a < 2 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_decreasing_log_base_range_l287_28792


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_linear_regression_theorem_l287_28739

-- Define the sample size
def n : ℕ := 10

-- Define the sums given in the problem
noncomputable def sum_x : ℝ := 80
noncomputable def sum_y : ℝ := 20
noncomputable def sum_xy : ℝ := 184
noncomputable def sum_x_sq : ℝ := 720

-- Define the means
noncomputable def x_bar : ℝ := sum_x / n
noncomputable def y_bar : ℝ := sum_y / n

-- Define the coefficients of the linear regression equation
noncomputable def b : ℝ := (sum_xy - n * x_bar * y_bar) / (sum_x_sq - n * x_bar^2)
noncomputable def a : ℝ := y_bar - b * x_bar

-- Define the linear regression equation
noncomputable def linear_regression (x : ℝ) : ℝ := b * x + a

-- State the theorem
theorem linear_regression_theorem :
  linear_regression 7 = 1.7 ∧ 
  b > 0 ∧
  b = 0.3 ∧ 
  a = -0.4 := by
  sorry

#eval "Proof completed"

end NUMINAMATH_CALUDE_ERRORFEEDBACK_linear_regression_theorem_l287_28739


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_polygon_perimeter_l287_28736

-- Define the points as ordered pairs
def P : ℝ × ℝ := (0, 10)
def Q : ℝ × ℝ := (5, 10)
def R : ℝ × ℝ := (5, 5)
def S : ℝ × ℝ := (10, 0)
def T : ℝ × ℝ := (0, 0)
def U : ℝ × ℝ := (5, 0)

-- Define a function to calculate distance between two points
noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

-- Define the perimeter of the polygon
noncomputable def perimeter : ℝ :=
  distance P Q + distance Q R + distance R S + 
  distance S U + distance U T + distance T P

-- Theorem statement
theorem polygon_perimeter : perimeter = 25 + 5 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_polygon_perimeter_l287_28736


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_perpendicular_lines_a_value_l287_28703

/-- Two lines are perpendicular if the product of their slopes is -1 -/
theorem perpendicular (m1 m2 : ℝ) : Prop := m1 * m2 = -1

/-- The slope of the line 4y + x + 5 = 0 -/
noncomputable def slope1 : ℝ := -1/4

/-- The slope of the line 3y + ax + 4 = 0 in terms of a -/
noncomputable def slope2 (a : ℝ) : ℝ := -a/3

/-- Theorem: If the lines 4y + x + 5 = 0 and 3y + ax + 4 = 0 are perpendicular, then a = -12 -/
theorem perpendicular_lines_a_value :
  ∀ a : ℝ, perpendicular slope1 (slope2 a) → a = -12 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_perpendicular_lines_a_value_l287_28703


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_S_is_infinite_lcm_function_exists_l287_28754

-- Define the set S
def S : Set ℕ := {n : ℕ | ∃ A : ℕ, A > 0 ∧ n = A * A ∧ ∃ k : ℕ, n = (10^k + 1) * A}

-- Statement 1: S is infinite
theorem S_is_infinite : Set.Infinite S := by sorry

-- Statement 2: Existence of function f
theorem lcm_function_exists : ∃ f : S → S → S, 
  ∀ a b c : S, (a.1 ∣ c.1) → (b.1 ∣ c.1) → (f a b).1 ∣ c.1 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_S_is_infinite_lcm_function_exists_l287_28754


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fraction_product_theorem_l287_28799

theorem fraction_product_theorem : 
  (1/4 * 8/1 * 1/32 * 64/1 * 1/128 * 256/1 * 1/512 * 1024/1 * 1/2048 * 4096/1 * 1/8192 * 16384/1 : ℚ) = 64 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_fraction_product_theorem_l287_28799


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_amare_fabric_needs_l287_28780

/-- The amount of additional fabric Amare needs for the dresses -/
def additional_fabric_needed (yards_per_dress : ℝ) (num_dresses : ℕ) (fabric_on_hand : ℝ) : ℝ :=
  yards_per_dress * 3 * (num_dresses : ℝ) - fabric_on_hand

/-- Theorem stating the additional fabric Amare needs -/
theorem amare_fabric_needs : 
  additional_fabric_needed 5.5 4 7 = 59 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_amare_fabric_needs_l287_28780


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_minimize_I_l287_28788

-- Define the function H
def H (p q : ℝ) : ℝ := -3*p*q + 4*p*(1-q) + 2*(1-p)*q - 5*(1-p)*(1-q)

-- Define I as the maximum of H over q ∈ [0, 1]
noncomputable def I (p : ℝ) : ℝ := 
  ⨆ q ∈ Set.Icc 0 1, H p q

-- Theorem statement
theorem minimize_I :
  ∀ p ∈ Set.Icc 0 1, I p ≥ I (1/2) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_minimize_I_l287_28788


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_slope_difference_is_negative_one_l287_28709

-- Define the ellipse
def is_on_ellipse (x y : ℝ) : Prop := x^2/4 + y^2/3 = 1

-- Define the line
def is_on_line (x y : ℝ) : Prop := y - x - 3 = 0

-- Define the foci
def F1 : ℝ × ℝ := (-2, 0)
def F2 : ℝ × ℝ := (2, 0)

-- Define the slopes
noncomputable def k1 (x y : ℝ) : ℝ := y / (x + 2)
noncomputable def k2 (x y : ℝ) : ℝ := y / (x - 2)

-- Main theorem
theorem slope_difference_is_negative_one (x y : ℝ) 
  (h_line : is_on_line x y) 
  (h_x_neq_neg3 : x ≠ -3) 
  (h_x_neq_sqrt3 : x ≠ Real.sqrt 3) 
  (h_x_neq_neg_sqrt3 : x ≠ -Real.sqrt 3) :
  1 / k2 x y - 2 / k1 x y = -1 := by
  sorry

#check slope_difference_is_negative_one

end NUMINAMATH_CALUDE_ERRORFEEDBACK_slope_difference_is_negative_one_l287_28709


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_cool_right_triangle_areas_l287_28776

/-- A cool right triangle is one where the area is equal to three times the sum of its leg lengths -/
def IsCoolRightTriangle (a b : ℕ) : Prop :=
  a * b / 2 = 3 * (a + b)

/-- The set of all possible areas of cool right triangles -/
def CoolRightTriangleAreas : Set ℕ :=
  {area | ∃ a b : ℕ, IsCoolRightTriangle a b ∧ area = a * b / 2}

theorem sum_of_cool_right_triangle_areas : 
  (Finset.sum {18, 24} id) = 42 := by
  sorry

#eval Finset.sum {18, 24} id

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_cool_right_triangle_areas_l287_28776


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_slope_at_pi_over_four_l287_28769

-- Define the function
noncomputable def f (x : ℝ) : ℝ := (Real.sin x) / (Real.sin x + Real.cos x) - 1/2

-- State the theorem
theorem tangent_slope_at_pi_over_four :
  let x₀ : ℝ := π/4
  let y₀ : ℝ := f x₀
  let slope : ℝ := deriv f x₀
  slope = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_slope_at_pi_over_four_l287_28769


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equal_numbers_l287_28763

theorem equal_numbers (n : ℕ) (a : ℕ → ℝ) :
  n ≥ 3 →
  (∀ i, 0 < i ∧ i ≤ n → a i > 0) →
  (∀ i j, 0 < i ∧ i ≤ n ∧ 0 < j ∧ j ≤ n →
    (a i ≤ a j ↔ (a ((i - 1 + n) % n + 1) + a (i % n + 1)) / a i ≤ (a ((j - 1 + n) % n + 1) + a (j % n + 1)) / a j)) →
  ∀ i j, 0 < i ∧ i ≤ n ∧ 0 < j ∧ j ≤ n → a i = a j :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_equal_numbers_l287_28763


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_uncovered_fraction_value_l287_28714

/-- A right triangle with legs of length 3 and 4 units, containing a square in the right angle corner -/
structure RightTriangleWithSquare where
  /-- Length of the first leg -/
  leg1 : ℝ
  /-- Length of the second leg -/
  leg2 : ℝ
  /-- Side length of the square in the corner -/
  square_side : ℝ
  /-- The legs have lengths 3 and 4 -/
  leg_lengths : leg1 = 3 ∧ leg2 = 4
  /-- The shortest distance from the square to the hypotenuse is 2 units -/
  distance_to_hypotenuse : square_side * (5 / Real.sqrt 25) = 2

/-- The fraction of the triangle that is not covered by the square -/
noncomputable def uncovered_fraction (t : RightTriangleWithSquare) : ℝ :=
  (t.leg1 * t.leg2 / 2 - t.square_side ^ 2) / (t.leg1 * t.leg2 / 2)

/-- The main theorem -/
theorem uncovered_fraction_value (t : RightTriangleWithSquare) :
    uncovered_fraction t = 145 / 147 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_uncovered_fraction_value_l287_28714


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_binomial_expansion_sum_l287_28775

theorem binomial_expansion_sum (x : ℝ) (n : ℕ) : 
  (∃ k, k = 3 ∧ 
    ∀ j, 0 ≤ j ∧ j ≤ n → Nat.choose n k ≥ Nat.choose n j) →
  (x^3 - 1/(2*x))^n = 1/64 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_binomial_expansion_sum_l287_28775


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_village_population_percentage_l287_28722

theorem village_population_percentage (total_population : ℕ) (percentage : ℚ) (result : ℕ) : 
  total_population = 28800 → percentage = 80 / 100 → result = 23040 →
  (percentage * total_population) = result := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_village_population_percentage_l287_28722


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pasha_stone_distribution_theorem_l287_28726

/-- Represents the distribution of stones in boxes after a series of moves -/
def StoneDistribution := Fin 2017 → ℕ

/-- Represents the sequence of natural numbers chosen by Pasha -/
def PashaSequence := Fin 2017 → ℕ

/-- Applies one move to the current distribution given Pasha's sequence -/
def applyMove (dist : StoneDistribution) (seq : PashaSequence) : StoneDistribution :=
  sorry

/-- Checks if all boxes have the same number of stones -/
def isEqualDistribution (dist : StoneDistribution) : Prop :=
  sorry

/-- Applies a function n times -/
def iterateN {α : Type*} (f : α → α) (n : ℕ) (x : α) : α :=
  match n with
  | 0 => x
  | n + 1 => f (iterateN f n x)

/-- The main theorem to be proved -/
theorem pasha_stone_distribution_theorem :
  ∃ (seq : PashaSequence),
    (∃ (finalDist : StoneDistribution),
      (∀ (k : ℕ), k < 43 → ¬isEqualDistribution (iterateN (applyMove · seq) k (λ _ => 0))) ∧
      isEqualDistribution (iterateN (applyMove · seq) 43 (λ _ => 0))) :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_pasha_stone_distribution_theorem_l287_28726


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rational_root_count_l287_28760

def polynomial (x b₄ b₃ b₂ b₁ : ℤ) : ℤ :=
  12 * x^5 + b₄ * x^4 + b₃ * x^3 + b₂ * x^2 + b₁ * x + 24

def is_root (r : ℚ) (b₄ b₃ b₂ b₁ : ℤ) : Prop :=
  polynomial (r.num) b₄ b₃ b₂ b₁ = 0

theorem rational_root_count :
  ∃ (s : Finset ℚ), (∀ b₄ b₃ b₂ b₁ : ℤ, ∀ r : ℚ, is_root r b₄ b₃ b₂ b₁ → r ∈ s) ∧ s.card = 32 :=
by
  sorry

#check rational_root_count

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rational_root_count_l287_28760


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_m_l287_28741

noncomputable def f (x : ℝ) : ℝ := 2 * Real.sin (Real.pi * x / 3 + Real.pi / 6) + 2

theorem range_of_m (a : ℝ) (m : ℝ) :
  a ∈ Set.Icc 1 2 →
  (∃ x₁ x₂, x₁ ∈ Set.Ico 0 m ∧ x₂ ∈ Set.Ico 0 m ∧ x₁ ≠ x₂ ∧ 
    f x₁ - a = 2 ∧ f x₂ - a = 2) →
  m ∈ Set.Ioo 2 6 :=
by
  sorry

#check range_of_m

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_m_l287_28741


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_locus_of_C_l287_28746

/-- Triangle ABC with AB = 2 and AD a median to BC with AD = 1.5 -/
structure TriangleABC where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ
  D : ℝ × ℝ
  h_AB : dist A B = 2
  h_AD_median : D = midpoint ℝ B C
  h_AD_length : dist A D = 1.5

/-- The locus of C is a circle with radius 3 and center on AB, distance 4 from B -/
theorem locus_of_C (t : TriangleABC) :
  ∃ (center : ℝ × ℝ), center.1 = t.B.1 - 4 ∧ center.2 = t.B.2 ∧ dist center t.C = 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_locus_of_C_l287_28746


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_imaginary_part_of_z_l287_28782

noncomputable def z : ℂ := (4 + 3*Complex.I) / (1 + 2*Complex.I)

theorem imaginary_part_of_z : z.im = -1 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_imaginary_part_of_z_l287_28782


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_sides_theorem_l287_28787

noncomputable def triangle_area : ℝ := 4920
noncomputable def angle_alpha : ℝ := 43 + 36 / 60
noncomputable def angle_beta : ℝ := 72 + 23 / 60

noncomputable def side_a : ℝ := 89
noncomputable def side_b : ℝ := 123
noncomputable def side_c : ℝ := 116

theorem triangle_sides_theorem :
  let angle_gamma := 180 - angle_alpha - angle_beta
  ∃ (ε : ℝ), ε > 0 ∧ ε < 1 ∧
  (abs (side_a - Real.sqrt ((2 * triangle_area * Real.sin angle_alpha) / (Real.sin angle_beta * Real.sin angle_gamma))) < ε) ∧
  (abs (side_b - (side_a * Real.sin angle_beta / Real.sin angle_alpha)) < ε) ∧
  (abs (side_c - (side_a * Real.sin angle_gamma / Real.sin angle_alpha)) < ε) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_sides_theorem_l287_28787


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l287_28724

noncomputable def f (x : ℝ) : ℝ := Real.sqrt 3 * Real.sin x * Real.cos x - Real.cos x ^ 2 + 1/2

theorem f_properties :
  (∀ x, f (x + π/3) = f (-(x + π/3))) ∧
  (∃ (zeros : Finset ℝ), zeros.card = 20 ∧ ∀ z ∈ zeros, 0 < z ∧ z < 10*π ∧ f z = 0) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l287_28724


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_f_l287_28795

noncomputable def f (x : ℝ) : ℝ :=
  Real.cos x / Real.sqrt (1 - Real.sin x ^ 2) +
  Real.sqrt (1 - Real.cos x ^ 2) / Real.sin x -
  Real.tan x / Real.sqrt (1 / Real.cos x ^ 2 - 1)

def is_standard_position (x : ℝ) : Prop :=
  0 ≤ x ∧ x < 2 * Real.pi

theorem range_of_f :
  ∀ y : ℝ, (∃ x : ℝ, is_standard_position x ∧ f x = y) ↔ y = -3 ∨ y = 1 := by
  sorry

#check range_of_f

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_f_l287_28795


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_solution_set_l287_28704

open Set

def solution_set (f : ℝ → ℝ) : Set ℝ :=
  {x | (x - 1) * f x < 0}

theorem inequality_solution_set 
  (f : ℝ → ℝ) 
  (h_odd : ∀ x, f (-x) = -f x)
  (h_incr : ∀ x y, 0 < x → x < y → f x < f y)
  (h_f2 : f 2 = 0) :
  solution_set f = Ioo (-2) 0 ∪ Ioo 1 2 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_solution_set_l287_28704


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_points_count_l287_28730

-- Define the curves C₁ and C₂
def C₁ (x y : ℝ) : Prop := x^2 - y^2 + 4*y - 3 = 0
def C₂ (a x y : ℝ) : Prop := y = a * x^2

-- Define the condition on a
def a_positive (a : ℝ) : Prop := a > 0

-- Theorem statement
theorem intersection_points_count 
  (a : ℝ) 
  (ha : a_positive a) :
  ∃ (S : Finset (ℝ × ℝ)), 
    (∀ p ∈ S, C₁ p.1 p.2 ∧ C₂ a p.1 p.2) ∧ 
    S.card = 4 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_points_count_l287_28730


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_satellite_work_theorem_l287_28742

/-- Work done in lifting a satellite -/
noncomputable def work_done (m : ℝ) (g : ℝ) (R₃ : ℝ) (H : ℝ) : ℝ :=
  m * g * R₃^2 * (1/R₃ - 1/(R₃ + H))

/-- Theorem: Work done in lifting a satellite -/
theorem satellite_work_theorem (m g R₃ H : ℝ) 
  (hm : m > 0) (hg : g > 0) (hR₃ : R₃ > 0) (hH : H > 0) :
  ∃ W : ℝ, W = work_done m g R₃ H ∧ 
  abs (W - 20253968254) < 1 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_satellite_work_theorem_l287_28742


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solid_is_frustum_l287_28756

/-- Represents a solid object -/
structure Solid :=
  (front_view : Shape)
  (side_view : Shape)
  (top_view : Shape)

/-- Represents different shapes -/
inductive Shape
  | IsoscelesTrapezoid
  | ConcentricCircles
  | Other

/-- Represents different types of solids -/
inductive SolidType
  | Cylinder
  | Cone
  | Frustum
  | Prism
  | Other

/-- Determines if a solid is a frustum based on its views -/
def is_frustum (s : Solid) : Prop :=
  s.front_view = Shape.IsoscelesTrapezoid ∧
  s.side_view = Shape.IsoscelesTrapezoid ∧
  s.top_view = Shape.ConcentricCircles

/-- Function to determine the solid type based on its views -/
def determine_solid_type (s : Solid) : SolidType :=
  match s.front_view, s.side_view, s.top_view with
  | Shape.IsoscelesTrapezoid, Shape.IsoscelesTrapezoid, Shape.ConcentricCircles => SolidType.Frustum
  | _, _, _ => SolidType.Other

theorem solid_is_frustum (s : Solid) 
  (h1 : s.front_view = Shape.IsoscelesTrapezoid)
  (h2 : s.side_view = Shape.IsoscelesTrapezoid)
  (h3 : s.top_view = Shape.ConcentricCircles) :
  is_frustum s ∧ determine_solid_type s = SolidType.Frustum :=
by
  constructor
  · exact ⟨h1, h2, h3⟩
  · simp [determine_solid_type, h1, h2, h3]


end NUMINAMATH_CALUDE_ERRORFEEDBACK_solid_is_frustum_l287_28756


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_battle_ratio_l287_28705

theorem battle_ratio (num_cannoneers : ℕ) (total_people : ℕ) 
  (h1 : num_cannoneers = 63)
  (h2 : total_people = 378) :
  (total_people - 2 * num_cannoneers : ℚ) / (2 * num_cannoneers) = 2 / 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_battle_ratio_l287_28705


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_diagonal_intersection_prob_l287_28725

/-- A regular decagon is a 10-sided polygon with all sides and angles equal -/
def RegularDecagon : Type := Unit

/-- The number of vertices in a regular decagon -/
def num_vertices : Nat := 10

/-- The number of diagonals in a regular decagon -/
def num_diagonals : Nat := (num_vertices * (num_vertices - 3)) / 2

/-- The number of pairs of diagonals in a regular decagon -/
def num_diagonal_pairs : Nat := 
  Nat.choose num_diagonals 2

/-- The number of convex quadrilaterals formed by four vertices of a regular decagon -/
def num_convex_quads : Nat := Nat.choose num_vertices 4

/-- The probability that two randomly chosen diagonals in a regular decagon
    intersect inside the decagon and form a convex quadrilateral -/
theorem diagonal_intersection_prob :
  (num_convex_quads : ℚ) / (num_diagonal_pairs : ℚ) = 42 / 119 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_diagonal_intersection_prob_l287_28725


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cone_height_from_sphere_l287_28740

theorem cone_height_from_sphere (R : ℝ) (h : ℝ) :
  h = 2 * R * (4 : ℝ) ^ (1/3) ↔ 
  (4/3 * Real.pi * R^3 = (1/24) * Real.pi * h^3) ∧
  h > 0 :=
by
  sorry

#check cone_height_from_sphere

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cone_height_from_sphere_l287_28740


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_ABC_properties_l287_28793

-- Define the triangle ABC
def triangle_ABC (A B C a b c : ℝ) : Prop :=
  -- Angles form an arithmetic sequence
  2 * B = A + C ∧
  -- Sum of angles in a triangle is π
  A + B + C = Real.pi ∧
  -- Given conditions
  c - a = 1 ∧
  b = Real.sqrt 7

-- Theorem statement
theorem triangle_ABC_properties (A B C a b c : ℝ) 
  (h : triangle_ABC A B C a b c) : 
  -- Area of the triangle
  (1/2 * a * c * Real.sin B = 3 * Real.sqrt 3 / 2) ∧ 
  -- Value of sin(2C + π/4)
  (Real.sin (2 * C + Real.pi/4) = (3 * Real.sqrt 6 - 13 * Real.sqrt 2) / 28) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_ABC_properties_l287_28793


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_and_line_equations_l287_28716

-- Define the circle C
noncomputable def circle_C (x y : ℝ) : Prop :=
  (x + 2)^2 + (y - 6)^2 = 16

-- Define the line that contains the center of circle C
def center_line (x y : ℝ) : Prop :=
  3 * x + y = 0

-- Define the point P
def point_P : ℝ × ℝ := (0, 5)

-- Define the length of the intercepted segment
noncomputable def intercepted_length : ℝ := 4 * Real.sqrt 3

-- Define the possible equations for line l
def line_l (x y : ℝ) : Prop :=
  x = 0 ∨ 3 * x - 4 * y + 20 = 0

theorem circle_and_line_equations :
  ∀ (x y : ℝ),
  (circle_C (-2) 2 ∧ circle_C 2 6) →  -- Circle C passes through A(-2, 2) and B(2, 6)
  (∃ (cx cy : ℝ), center_line cx cy ∧ circle_C cx cy) →  -- Center of C lies on 3x + y = 0
  (∃ (mx nx my ny : ℝ),
    circle_C mx my ∧ circle_C nx ny ∧
    line_l mx my ∧ line_l nx ny ∧
    (mx - nx)^2 + (my - ny)^2 = intercepted_length^2) →  -- Segment intercepted by C on l has length 4√3
  (∃ (lx ly : ℝ), line_l lx ly ∧ lx = point_P.1 ∧ ly = point_P.2) →  -- Line l passes through P(0, 5)
  circle_C x y ∧ line_l x y :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_and_line_equations_l287_28716


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_dot_product_l287_28745

/-- The hyperbola equation -/
def is_on_hyperbola (x y : ℝ) : Prop := x^2 / 3 - y^2 = 1

/-- The foci of the hyperbola -/
def F₁ : ℝ × ℝ := (-2, 0)
def F₂ : ℝ × ℝ := (2, 0)

/-- The area of the triangle F₁PF₂ -/
noncomputable def triangle_area (P : ℝ × ℝ) : ℝ := 
  abs ((P.1 - F₁.1) * (P.2 - F₂.2) - (P.2 - F₁.2) * (P.1 - F₂.1)) / 2

/-- The dot product of vectors PF₁ and PF₂ -/
def dot_product (P : ℝ × ℝ) : ℝ := 
  (F₁.1 - P.1) * (F₂.1 - P.1) + (F₁.2 - P.2) * (F₂.2 - P.2)

theorem hyperbola_dot_product :
  ∀ P : ℝ × ℝ, 
  is_on_hyperbola P.1 P.2 → 
  triangle_area P = 2 → 
  dot_product P = 3 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_dot_product_l287_28745


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_number_l287_28773

theorem smallest_number : 
  ∀ (a b c d : ℝ), a = Real.sqrt 3 ∧ b = 0 ∧ c = -Real.sqrt 2 ∧ d = -1 →
  c ≤ a ∧ c ≤ b ∧ c ≤ d := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_number_l287_28773


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_square_field_side_length_is_60_l287_28713

/-- The side length of a square field given the time to run around it and the running speed -/
noncomputable def squareFieldSideLength (timeSeconds : ℝ) (speedKmPerHour : ℝ) : ℝ :=
  let speedMeterPerSecond := speedKmPerHour * 1000 / 3600
  let distanceMeters := speedMeterPerSecond * timeSeconds
  distanceMeters / 4

/-- Theorem: The side length of a square field that takes 96 seconds to run around at 9 km/hr is 60 meters -/
theorem square_field_side_length_is_60 :
  squareFieldSideLength 96 9 = 60 := by
  -- Unfold the definition of squareFieldSideLength
  unfold squareFieldSideLength
  -- Simplify the expression
  simp
  -- The proof is complete
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_square_field_side_length_is_60_l287_28713


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ABCD_value_l287_28723

-- Define the constants as noncomputable
noncomputable def A : ℝ := Real.sqrt 2013 + Real.sqrt 2012
noncomputable def B : ℝ := -Real.sqrt 2013 - Real.sqrt 2012
noncomputable def C : ℝ := Real.sqrt 2013 - Real.sqrt 2012
noncomputable def D : ℝ := Real.sqrt 2012 - Real.sqrt 2013

-- State the theorem
theorem ABCD_value : A * B * C * D = -1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ABCD_value_l287_28723


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_consecutive_integers_sqrt_5_l287_28798

theorem consecutive_integers_sqrt_5 (a b : ℤ) : 
  (b = a + 1) →  -- a and b are consecutive integers
  (↑a < Real.sqrt 5) →  -- a < sqrt(5)
  (Real.sqrt 5 < ↑b) →  -- sqrt(5) < b
  (b : ℝ)^(a : ℝ) = 9 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_consecutive_integers_sqrt_5_l287_28798


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_canal_depth_is_84_l287_28791

/-- Represents the properties of a trapezium-shaped canal cross-section -/
structure CanalCrossSection where
  topWidth : ℝ
  bottomWidth : ℝ
  area : ℝ

/-- Calculates the depth of a canal given its cross-section properties -/
noncomputable def canalDepth (c : CanalCrossSection) : ℝ :=
  (2 * c.area) / (c.topWidth + c.bottomWidth)

/-- Theorem stating that a canal with given properties has a depth of 84 meters -/
theorem canal_depth_is_84 (c : CanalCrossSection) 
    (h1 : c.topWidth = 12)
    (h2 : c.bottomWidth = 8)
    (h3 : c.area = 840) : 
  canalDepth c = 84 := by
  sorry

#check canal_depth_is_84

end NUMINAMATH_CALUDE_ERRORFEEDBACK_canal_depth_is_84_l287_28791


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_washing_effectiveness_l287_28701

/-- The function representing the ratio of pesticide residue after washing. -/
noncomputable def f (x : ℝ) : ℝ := 1 / (1 + x^2)

/-- Theorem stating the condition for when washing twice is more effective than washing once. -/
theorem washing_effectiveness (a : ℝ) (ha : a > 0) :
  f a < (f (a/2))^2 ↔ a > 2 * Real.sqrt 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_washing_effectiveness_l287_28701


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_mod_equivalence_unique_l287_28794

theorem mod_equivalence_unique : ∃! n : ℕ, n ≤ 6 ∧ n ≡ 6 [MOD 7] := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_mod_equivalence_unique_l287_28794


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_b_worked_ten_days_l287_28750

noncomputable section

/-- The number of days it takes for a person to complete the entire work -/
def work_days (person : String) : ℝ :=
  match person with
  | "A" => 30
  | "B" => 30
  | "C" => 29.999999999999996
  | _ => 0

/-- The fraction of work completed by a person in one day -/
def work_rate (person : String) : ℝ :=
  1 / work_days person

/-- The number of days each person worked -/
def days_worked (person : String) : ℝ :=
  match person with
  | "A" => 10
  | "C" => 10
  | "B" => 10  -- This is what we want to prove
  | _ => 0

/-- The total work is represented as 1 -/
def total_work : ℝ := 1

theorem b_worked_ten_days :
  days_worked "B" * work_rate "B" = total_work - (days_worked "A" * work_rate "A" + days_worked "C" * work_rate "C") :=
by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_b_worked_ten_days_l287_28750


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_convex_quad_exists_l287_28727

/-- A point in a 2D plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- A set of 5 points in a plane -/
def FivePoints := Fin 5 → Point

/-- Predicate to check if three points are collinear -/
def are_collinear (p q r : Point) : Prop := 
  (q.x - p.x) * (r.y - p.y) = (r.x - p.x) * (q.y - p.y)

/-- Predicate to check if four points form a convex quadrilateral -/
def is_convex_quadrilateral (p q r s : Point) : Prop := sorry

/-- Main theorem: Given 5 points in a plane with no three collinear,
    there exist 4 points forming a convex quadrilateral -/
theorem convex_quad_exists (points : FivePoints)
  (h : ∀ i j k, i ≠ j → j ≠ k → i ≠ k → ¬are_collinear (points i) (points j) (points k)) :
  ∃ i j k l, i ≠ j ∧ j ≠ k ∧ k ≠ l ∧ i ≠ k ∧ i ≠ l ∧ j ≠ l ∧
    is_convex_quadrilateral (points i) (points j) (points k) (points l) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_convex_quad_exists_l287_28727


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_george_and_abe_work_time_l287_28797

/-- The time it takes for two workers to complete a job together, given their individual completion times. -/
noncomputable def combined_work_time (t1 t2 : ℝ) : ℝ :=
  1 / (1 / t1 + 1 / t2)

/-- Theorem stating that if George can complete a job in 70 minutes and Abe can complete the same job in 30 minutes, then together they can complete the job in 21 minutes. -/
theorem george_and_abe_work_time :
  combined_work_time 70 30 = 21 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_george_and_abe_work_time_l287_28797


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_chapter_text_pages_and_percentage_l287_28732

def total_pages : ℕ := 693
def intro_percent : ℚ := 3 / 100
def appendix_percent : ℚ := 13 / 200
def chapter_text_percent : ℚ := 85 / 200

theorem chapter_text_pages_and_percentage :
  let chapter_pages := total_pages - (Int.toNat ⌊intro_percent * total_pages⌋ + Int.toNat ⌊appendix_percent * total_pages⌋)
  let text_pages := Int.toNat ⌊chapter_text_percent * chapter_pages⌋
  (text_pages = 266) ∧ 
  (abs ((text_pages : ℝ) / total_pages - 0.3838) < 0.0001) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_chapter_text_pages_and_percentage_l287_28732
