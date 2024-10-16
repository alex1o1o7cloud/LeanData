import Mathlib

namespace NUMINAMATH_CALUDE_inequality_system_solution_l406_40668

theorem inequality_system_solution (x : ℝ) :
  (x + 5 < 4) ∧ ((3 * x + 1) / 2 ≥ 2 * x - 1) → x < -1 := by
  sorry

end NUMINAMATH_CALUDE_inequality_system_solution_l406_40668


namespace NUMINAMATH_CALUDE_odd_function_domain_sum_l406_40640

def OddFunction (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

theorem odd_function_domain_sum (f : ℝ → ℝ) (a b : ℝ) 
  (h1 : OddFunction f) 
  (h2 : Set.range f = {-1, 2, a, b}) : 
  a + b = -1 := by
  sorry

end NUMINAMATH_CALUDE_odd_function_domain_sum_l406_40640


namespace NUMINAMATH_CALUDE_existence_of_common_source_l406_40659

/-- Represents the process of obtaining one number from another through digit manipulation -/
def Obtainable (m n : ℕ) : Prop := sorry

/-- Checks if a natural number contains the digit 5 in its decimal representation -/
def ContainsDigitFive (n : ℕ) : Prop := sorry

theorem existence_of_common_source (S : Finset ℕ) 
  (h1 : S.Nonempty) 
  (h2 : ∀ s ∈ S, ¬ContainsDigitFive s) : 
  ∃ N : ℕ, ∀ s ∈ S, Obtainable s N := by sorry

end NUMINAMATH_CALUDE_existence_of_common_source_l406_40659


namespace NUMINAMATH_CALUDE_selection_problem_l406_40614

theorem selection_problem (n_boys : ℕ) (n_girls : ℕ) : 
  n_boys = 4 → n_girls = 3 → 
  (Nat.choose n_boys 2 * Nat.choose n_girls 1) + 
  (Nat.choose n_girls 2 * Nat.choose n_boys 1) = 30 := by
sorry

end NUMINAMATH_CALUDE_selection_problem_l406_40614


namespace NUMINAMATH_CALUDE_probability_red_or_white_l406_40665

def total_marbles : ℕ := 50
def blue_marbles : ℕ := 5
def red_marbles : ℕ := 9
def white_marbles : ℕ := total_marbles - (blue_marbles + red_marbles)

theorem probability_red_or_white :
  (red_marbles + white_marbles : ℚ) / total_marbles = 9 / 10 := by
  sorry

end NUMINAMATH_CALUDE_probability_red_or_white_l406_40665


namespace NUMINAMATH_CALUDE_circle_and_line_properties_l406_40644

-- Define the circle C
def circle_C (x y : ℝ) : Prop := (x - 3)^2 + (y - 4)^2 = 4

-- Define the line l₁ passing through A(1,0)
def line_l1 (x y : ℝ) : Prop := ∃ k : ℝ, y = k * (x - 1)

-- Define tangent line condition
def is_tangent (line : (ℝ → ℝ → Prop)) (circle : (ℝ → ℝ → Prop)) : Prop :=
  ∃ x y : ℝ, line x y ∧ circle x y ∧
  ∀ x' y' : ℝ, line x' y' → circle x' y' → (x', y') = (x, y)

-- Define the slope angle of π/4
def slope_angle_pi_4 (line : (ℝ → ℝ → Prop)) : Prop :=
  ∃ k : ℝ, (∀ x y : ℝ, line x y ↔ y = k * (x - 1)) ∧ k = 1

-- Main theorem
theorem circle_and_line_properties :
  (is_tangent line_l1 circle_C →
    (∀ x y : ℝ, line_l1 x y ↔ (x = 1 ∨ 3*x - 4*y - 3 = 0))) ∧
  (slope_angle_pi_4 line_l1 →
    ∃ P Q : ℝ × ℝ,
      circle_C P.1 P.2 ∧ circle_C Q.1 Q.2 ∧
      line_l1 P.1 P.2 ∧ line_l1 Q.1 Q.2 ∧
      ((P.1 + Q.1) / 2, (P.2 + Q.2) / 2) = (4, 3)) :=
sorry

end NUMINAMATH_CALUDE_circle_and_line_properties_l406_40644


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l406_40654

-- Define the sets A and B
def A : Set ℝ := {x | x - 1 > 1}
def B : Set ℝ := {x | x < 3}

-- State the theorem
theorem intersection_of_A_and_B :
  A ∩ B = {x : ℝ | 2 < x ∧ x < 3} := by sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l406_40654


namespace NUMINAMATH_CALUDE_exists_initial_points_for_82_l406_40671

/-- The function that calculates the number of points after one application of the procedure -/
def points_after_one_step (n : ℕ) : ℕ := 3 * n - 2

/-- The function that calculates the number of points after two applications of the procedure -/
def points_after_two_steps (n : ℕ) : ℕ := 9 * n - 8

/-- Theorem stating that there exists an initial number of points that results in 82 points after two steps -/
theorem exists_initial_points_for_82 : ∃ n : ℕ, n > 0 ∧ points_after_two_steps n = 82 := by
  sorry

end NUMINAMATH_CALUDE_exists_initial_points_for_82_l406_40671


namespace NUMINAMATH_CALUDE_ruby_candies_l406_40672

/-- The number of friends Ruby shares her candies with -/
def num_friends : ℕ := 9

/-- The number of candies each friend receives -/
def candies_per_friend : ℕ := 4

/-- The initial number of candies Ruby has -/
def initial_candies : ℕ := num_friends * candies_per_friend

theorem ruby_candies : initial_candies = 36 := by
  sorry

end NUMINAMATH_CALUDE_ruby_candies_l406_40672


namespace NUMINAMATH_CALUDE_sin_difference_of_complex_exponentials_l406_40628

theorem sin_difference_of_complex_exponentials (α β : ℝ) :
  Complex.exp (α * I) = 4/5 + 3/5 * I →
  Complex.exp (β * I) = 12/13 + 5/13 * I →
  Real.sin (α - β) = -16/65 := by
  sorry

end NUMINAMATH_CALUDE_sin_difference_of_complex_exponentials_l406_40628


namespace NUMINAMATH_CALUDE_integer_fraction_equality_l406_40642

theorem integer_fraction_equality (d : ℤ) : ∃ m n : ℤ, d * (m^2 - n) = n - 2*m + 1 := by
  sorry

end NUMINAMATH_CALUDE_integer_fraction_equality_l406_40642


namespace NUMINAMATH_CALUDE_impossible_arrangement_l406_40678

theorem impossible_arrangement : ¬ ∃ (seq : Fin 3972 → Fin 1986), 
  (∀ k : Fin 1986, (∃! i j : Fin 3972, seq i = k ∧ seq j = k ∧ i ≠ j)) ∧
  (∀ k : Fin 1986, ∀ i j : Fin 3972, 
    seq i = k → seq j = k → i ≠ j → 
    (j.val > i.val → j.val - i.val = k.val + 1)) :=
by sorry

end NUMINAMATH_CALUDE_impossible_arrangement_l406_40678


namespace NUMINAMATH_CALUDE_peanut_butter_servings_l406_40690

/-- Represents the amount of peanut butter in tablespoons -/
def peanut_butter : ℚ := 37 + 2/3

/-- Represents the serving size in tablespoons -/
def serving_size : ℚ := 2 + 1/2

/-- Calculates the number of servings in the jar -/
def number_of_servings : ℚ := peanut_butter / serving_size

/-- Proves that the number of servings in the jar is equal to 15 1/15 -/
theorem peanut_butter_servings : number_of_servings = 15 + 1/15 := by
  sorry

end NUMINAMATH_CALUDE_peanut_butter_servings_l406_40690


namespace NUMINAMATH_CALUDE_store_discount_difference_l406_40632

theorem store_discount_difference :
  let initial_discount : ℝ := 0.25
  let additional_discount : ℝ := 0.10
  let claimed_discount : ℝ := 0.35
  let price_after_initial := 1 - initial_discount
  let price_after_both := price_after_initial * (1 - additional_discount)
  let true_discount := 1 - price_after_both
  claimed_discount - true_discount = 0.025 := by
sorry

end NUMINAMATH_CALUDE_store_discount_difference_l406_40632


namespace NUMINAMATH_CALUDE_consecutive_negative_integers_sum_l406_40666

theorem consecutive_negative_integers_sum (n : ℤ) : 
  n < 0 ∧ n + 1 < 0 ∧ n * (n + 1) = 2550 → n + (n + 1) = -101 := by
  sorry

end NUMINAMATH_CALUDE_consecutive_negative_integers_sum_l406_40666


namespace NUMINAMATH_CALUDE_attendance_difference_l406_40606

/-- Proves that given the initial ratio of boys to girls to adults as 9.5:6.25:4.75,
    with 30% of attendees being girls, and after 15% of girls and 20% of adults leave,
    the percentage difference between boys and the combined number of remaining girls
    and adults is approximately 2.304%. -/
theorem attendance_difference (boys girls adults : ℝ) 
    (h_ratio : boys = 9.5 ∧ girls = 6.25 ∧ adults = 4.75)
    (h_girls_percent : girls / (boys + girls + adults) = 0.3)
    (h_girls_left : ℝ) (h_adults_left : ℝ)
    (h_girls_left_percent : h_girls_left = 0.15)
    (h_adults_left_percent : h_adults_left = 0.2) :
    let total := boys + girls + adults
    let boys_percent := boys / total
    let girls_adjusted := girls * (1 - h_girls_left)
    let adults_adjusted := adults * (1 - h_adults_left)
    let girls_adults_adjusted_percent := (girls_adjusted + adults_adjusted) / total
    abs (boys_percent - girls_adults_adjusted_percent - 0.02304) < 0.00001 := by
  sorry

end NUMINAMATH_CALUDE_attendance_difference_l406_40606


namespace NUMINAMATH_CALUDE_timothy_speed_l406_40680

theorem timothy_speed (mother_speed : ℝ) (distance : ℝ) (head_start : ℝ) :
  mother_speed = 36 →
  distance = 1.8 →
  head_start = 0.25 →
  let mother_time : ℝ := distance / mother_speed
  let total_time : ℝ := mother_time + head_start
  let timothy_speed : ℝ := distance / total_time
  timothy_speed = 6 := by sorry

end NUMINAMATH_CALUDE_timothy_speed_l406_40680


namespace NUMINAMATH_CALUDE_complex_equation_sum_l406_40677

theorem complex_equation_sum (x y : ℝ) :
  (↑x + (↑y - 2) * I : ℂ) = 2 / (1 + I) →
  x + y = 2 := by
sorry

end NUMINAMATH_CALUDE_complex_equation_sum_l406_40677


namespace NUMINAMATH_CALUDE_complex_equation_solution_l406_40600

theorem complex_equation_solution (a : ℝ) (i : ℂ) (h1 : i * i = -1) :
  (Complex.im ((1 - i^2023) / (a * i)) = 3) → a = -1/3 := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l406_40600


namespace NUMINAMATH_CALUDE_five_fridays_in_august_l406_40679

/-- Represents days of the week -/
inductive DayOfWeek
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday
  | Saturday
  | Sunday

/-- Represents a calendar month -/
structure Month where
  days : Nat
  firstDay : DayOfWeek

/-- Function to get the next day of the week -/
def nextDay (d : DayOfWeek) : DayOfWeek :=
  match d with
  | DayOfWeek.Monday => DayOfWeek.Tuesday
  | DayOfWeek.Tuesday => DayOfWeek.Wednesday
  | DayOfWeek.Wednesday => DayOfWeek.Thursday
  | DayOfWeek.Thursday => DayOfWeek.Friday
  | DayOfWeek.Friday => DayOfWeek.Saturday
  | DayOfWeek.Saturday => DayOfWeek.Sunday
  | DayOfWeek.Sunday => DayOfWeek.Monday

/-- Function to count occurrences of a specific day in a month -/
def countDayOccurrences (m : Month) (d : DayOfWeek) : Nat :=
  sorry

/-- Theorem stating that if July has five Tuesdays, August will have five Fridays -/
theorem five_fridays_in_august 
  (july : Month) 
  (august : Month) 
  (h1 : july.days = 31) 
  (h2 : august.days = 31) 
  (h3 : countDayOccurrences july DayOfWeek.Tuesday = 5) :
  countDayOccurrences august DayOfWeek.Friday = 5 :=
sorry

end NUMINAMATH_CALUDE_five_fridays_in_august_l406_40679


namespace NUMINAMATH_CALUDE_half_angle_quadrant_l406_40685

-- Define the concept of quadrant
def in_first_quadrant (θ : ℝ) : Prop := 0 < θ ∧ θ < Real.pi / 2
def in_third_quadrant (θ : ℝ) : Prop := Real.pi < θ ∧ θ < 3 * Real.pi / 2

theorem half_angle_quadrant (α : ℝ) :
  in_first_quadrant α → in_first_quadrant (α / 2) ∨ in_third_quadrant (α / 2) := by
  sorry

end NUMINAMATH_CALUDE_half_angle_quadrant_l406_40685


namespace NUMINAMATH_CALUDE_vector_addition_l406_40651

variable {V : Type*} [AddCommGroup V]

theorem vector_addition (A B C : V) (a b : V) 
  (h1 : B - A = a) (h2 : C - B = b) : C - A = a + b := by
  sorry

end NUMINAMATH_CALUDE_vector_addition_l406_40651


namespace NUMINAMATH_CALUDE_octal_minus_quinary_in_decimal_l406_40627

def base_to_decimal (digits : List Nat) (base : Nat) : Nat :=
  digits.enum.foldr (fun (i, d) acc => acc + d * base ^ i) 0

def octal_54321 : List Nat := [1, 2, 3, 4, 5]
def quinary_4321 : List Nat := [1, 2, 3, 4]

theorem octal_minus_quinary_in_decimal : 
  base_to_decimal octal_54321 8 - base_to_decimal quinary_4321 5 = 22151 := by
  sorry

end NUMINAMATH_CALUDE_octal_minus_quinary_in_decimal_l406_40627


namespace NUMINAMATH_CALUDE_tangent_points_parallel_to_line_tangent_points_on_curve_unique_tangent_points_main_theorem_l406_40689

-- Define the function f(x) = x^3 + x - 2
def f (x : ℝ) : ℝ := x^3 + x - 2

-- Define the derivative of f(x)
def f_derivative (x : ℝ) : ℝ := 3 * x^2 + 1

theorem tangent_points_parallel_to_line (x : ℝ) :
  (f_derivative x = 4) ↔ (x = 1 ∨ x = -1) := by sorry

theorem tangent_points_on_curve :
  f 1 = 0 ∧ f (-1) = -4 := by sorry

theorem unique_tangent_points :
  ∀ x : ℝ, f_derivative x = 4 → (x = 1 ∨ x = -1) := by sorry

-- Main theorem
theorem main_theorem :
  ∃! s : Set (ℝ × ℝ), s = {(1, 0), (-1, -4)} ∧
  (∀ (x y : ℝ), (x, y) ∈ s ↔ f x = y ∧ f_derivative x = 4) := by sorry

end NUMINAMATH_CALUDE_tangent_points_parallel_to_line_tangent_points_on_curve_unique_tangent_points_main_theorem_l406_40689


namespace NUMINAMATH_CALUDE_quadratic_equation_result_l406_40626

theorem quadratic_equation_result (a : ℝ) (h : a^2 - 2*a + 1 = 0) : 
  4*a - 2*a^2 + 2 = 4 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_result_l406_40626


namespace NUMINAMATH_CALUDE_complex_roots_theorem_l406_40613

theorem complex_roots_theorem (a b : ℝ) : 
  (Complex.I : ℂ) ^ 2 = -1 →
  (a + 4 * Complex.I) * (a + 4 * Complex.I) - (10 + 9 * Complex.I) * (a + 4 * Complex.I) + (4 + 46 * Complex.I) = 0 →
  (b + 5 * Complex.I) * (b + 5 * Complex.I) - (10 + 9 * Complex.I) * (b + 5 * Complex.I) + (4 + 46 * Complex.I) = 0 →
  a = 6 ∧ b = 4 := by
sorry

end NUMINAMATH_CALUDE_complex_roots_theorem_l406_40613


namespace NUMINAMATH_CALUDE_triangle_nature_l406_40602

theorem triangle_nature (a b c : ℝ) (h_ratio : a / b = 3 / 4 ∧ b / c = 4 / 5)
  (h_perimeter : a + b + c = 36) : a^2 + b^2 = c^2 := by
  sorry

end NUMINAMATH_CALUDE_triangle_nature_l406_40602


namespace NUMINAMATH_CALUDE_theo_cookie_days_l406_40695

/-- The number of days Theo eats cookies each month -/
def days_per_month (cookies_per_time cookies_per_day total_cookies months : ℕ) : ℕ :=
  (total_cookies / months) / (cookies_per_time * cookies_per_day)

/-- Theorem stating that Theo eats cookies for 20 days each month -/
theorem theo_cookie_days : days_per_month 13 3 2340 3 = 20 := by
  sorry

end NUMINAMATH_CALUDE_theo_cookie_days_l406_40695


namespace NUMINAMATH_CALUDE_expression_simplification_and_evaluation_l406_40698

theorem expression_simplification_and_evaluation :
  let x : ℝ := Real.sqrt 3 + 1
  (x / (x^2 - 1)) / (1 - 1 / (x + 1)) = Real.sqrt 3 / 3 := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_and_evaluation_l406_40698


namespace NUMINAMATH_CALUDE_third_score_calculation_l406_40667

/-- Given four scores where three are known and their average with an unknown fourth score is 76.6,
    prove that the unknown score must be 79.4. -/
theorem third_score_calculation (score1 score2 score4 : ℝ) (average : ℝ) :
  score1 = 65 →
  score2 = 67 →
  score4 = 95 →
  average = 76.6 →
  ∃ score3 : ℝ, score3 = 79.4 ∧ (score1 + score2 + score3 + score4) / 4 = average :=
by sorry

end NUMINAMATH_CALUDE_third_score_calculation_l406_40667


namespace NUMINAMATH_CALUDE_probability_of_sum_5_is_one_thirty_sixth_l406_40692

/-- A fair 6-sided die with distinct numbers 1 through 6 -/
def FairDie : Type := Fin 6

/-- The number of dice rolled -/
def numDice : ℕ := 3

/-- The target sum we're aiming for -/
def targetSum : ℕ := 5

/-- The set of all possible outcomes when rolling three fair 6-sided dice -/
def allOutcomes : Finset (FairDie × FairDie × FairDie) := sorry

/-- The set of favorable outcomes (those that sum to targetSum) -/
def favorableOutcomes : Finset (FairDie × FairDie × FairDie) := sorry

/-- The probability of rolling a total of 5 with three fair 6-sided dice -/
def probabilityOfSum5 : ℚ :=
  (Finset.card favorableOutcomes : ℚ) / (Finset.card allOutcomes : ℚ)

/-- Theorem stating that the probability of rolling a sum of 5 with three fair 6-sided dice is 1/36 -/
theorem probability_of_sum_5_is_one_thirty_sixth :
  probabilityOfSum5 = 1 / 36 := by sorry

end NUMINAMATH_CALUDE_probability_of_sum_5_is_one_thirty_sixth_l406_40692


namespace NUMINAMATH_CALUDE_donation_problem_solution_l406_40621

/-- Represents a transportation plan with type A and B trucks -/
structure TransportPlan where
  typeA : Nat
  typeB : Nat

/-- Represents the problem setup -/
structure DonationProblem where
  totalItems : Nat
  waterExcess : Nat
  typeAWaterCapacity : Nat
  typeAVegCapacity : Nat
  typeBWaterCapacity : Nat
  typeBVegCapacity : Nat
  totalTrucks : Nat
  typeACost : Nat
  typeBCost : Nat

def isValidPlan (p : DonationProblem) (plan : TransportPlan) : Prop :=
  plan.typeA + plan.typeB = p.totalTrucks ∧
  plan.typeA * p.typeAWaterCapacity + plan.typeB * p.typeBWaterCapacity ≥ (p.totalItems + p.waterExcess) / 2 ∧
  plan.typeA * p.typeAVegCapacity + plan.typeB * p.typeBVegCapacity ≥ (p.totalItems - p.waterExcess) / 2

def planCost (p : DonationProblem) (plan : TransportPlan) : Nat :=
  plan.typeA * p.typeACost + plan.typeB * p.typeBCost

theorem donation_problem_solution (p : DonationProblem)
  (h_total : p.totalItems = 320)
  (h_excess : p.waterExcess = 80)
  (h_typeA : p.typeAWaterCapacity = 40 ∧ p.typeAVegCapacity = 10)
  (h_typeB : p.typeBWaterCapacity = 20 ∧ p.typeBVegCapacity = 20)
  (h_trucks : p.totalTrucks = 8)
  (h_costs : p.typeACost = 400 ∧ p.typeBCost = 360) :
  -- 1. Number of water and vegetable pieces
  (p.totalItems + p.waterExcess) / 2 = 200 ∧ (p.totalItems - p.waterExcess) / 2 = 120 ∧
  -- 2. Valid transportation plans
  (∀ plan, isValidPlan p plan ↔ 
    (plan = ⟨2, 6⟩ ∨ plan = ⟨3, 5⟩ ∨ plan = ⟨4, 4⟩)) ∧
  -- 3. Minimum cost plan
  (∀ plan, isValidPlan p plan → planCost p ⟨2, 6⟩ ≤ planCost p plan) ∧
  planCost p ⟨2, 6⟩ = 2960 :=
sorry

end NUMINAMATH_CALUDE_donation_problem_solution_l406_40621


namespace NUMINAMATH_CALUDE_tangent_circles_F_value_l406_40649

/-- Circle C₁ with equation x² + y² = 1 -/
def C₁ : Set (ℝ × ℝ) := {p | p.1^2 + p.2^2 = 1}

/-- Circle C₂ with equation x² + y² - 6x - 8y + F = 0 -/
def C₂ (F : ℝ) : Set (ℝ × ℝ) := {p | p.1^2 + p.2^2 - 6*p.1 - 8*p.2 + F = 0}

/-- Two circles are internally tangent if the distance between their centers
    is equal to the absolute difference of their radii -/
def internally_tangent (S T : Set (ℝ × ℝ)) : Prop :=
  ∃ (c₁ c₂ : ℝ × ℝ) (r₁ r₂ : ℝ),
    (∀ p, p ∈ S ↔ (p.1 - c₁.1)^2 + (p.2 - c₁.2)^2 = r₁^2) ∧
    (∀ p, p ∈ T ↔ (p.1 - c₂.1)^2 + (p.2 - c₂.2)^2 = r₂^2) ∧
    (c₂.1 - c₁.1)^2 + (c₂.2 - c₁.2)^2 = (r₂ - r₁)^2

/-- If C₁ and C₂ are internally tangent, then F = -11 -/
theorem tangent_circles_F_value :
  internally_tangent C₁ (C₂ F) → F = -11 := by sorry

end NUMINAMATH_CALUDE_tangent_circles_F_value_l406_40649


namespace NUMINAMATH_CALUDE_fraction_product_theorem_l406_40650

/-- The type of fractions in the sequence -/
def Fraction (n : ℕ) := { k : ℕ // 2 ≤ k ∧ k ≤ n }

/-- The sequence of fractions -/
def fractionSequence (n : ℕ) : List (Fraction n) :=
  List.range (n - 1) |>.map (fun i => ⟨i + 2, by sorry⟩)

/-- The product of the original sequence of fractions -/
def originalProduct (n : ℕ) : ℚ :=
  (fractionSequence n).foldl (fun acc f => acc * (f.val : ℚ) / ((f.val - 1) : ℚ)) 1

/-- A function that determines whether a fraction should be reciprocated -/
def reciprocate (n : ℕ) : Fraction n → Bool := sorry

/-- The product after reciprocating some fractions -/
def modifiedProduct (n : ℕ) : ℚ :=
  (fractionSequence n).foldl
    (fun acc f => 
      if reciprocate n f
      then acc * ((f.val - 1) : ℚ) / (f.val : ℚ)
      else acc * (f.val : ℚ) / ((f.val - 1) : ℚ))
    1

/-- The main theorem -/
theorem fraction_product_theorem (n : ℕ) (h : n > 2) :
  (∃ (reciprocate : Fraction n → Bool), modifiedProduct n = 1) ↔ ∃ (a : ℕ), n = a^2 ∧ a > 1 := by
  sorry

end NUMINAMATH_CALUDE_fraction_product_theorem_l406_40650


namespace NUMINAMATH_CALUDE_hyperbola_asymptote_perpendicular_line_l406_40660

/-- Represents a hyperbola in the Cartesian plane -/
structure Hyperbola where
  a : ℝ
  equation : ℝ → ℝ → Prop
  asymptote : ℝ → ℝ → Prop

/-- Represents a line in the Cartesian plane -/
structure Line where
  equation : ℝ → ℝ → Prop

/-- Two lines are perpendicular if the product of their slopes is -1 -/
def Perpendicular (l1 l2 : Line) : Prop :=
  ∃ m1 m2 : ℝ, (∀ x y, l1.equation x y ↔ y = m1 * x + (0 : ℝ)) ∧ 
              (∀ x y, l2.equation x y ↔ y = m2 * x + (0 : ℝ)) ∧ 
              m1 * m2 = -1

/-- The main theorem -/
theorem hyperbola_asymptote_perpendicular_line (h : Hyperbola) (l : Line) : 
  h.a > 0 ∧ 
  (∀ x y, h.equation x y ↔ x^2 / h.a^2 - y^2 = 1) ∧
  (∀ x y, l.equation x y ↔ 2*x - y + 1 = 0) ∧
  (∃ la : Line, h.asymptote = la.equation ∧ Perpendicular la l) →
  h.a = 2 := by
  sorry

end NUMINAMATH_CALUDE_hyperbola_asymptote_perpendicular_line_l406_40660


namespace NUMINAMATH_CALUDE_prime_divisors_and_totient_l406_40694

theorem prime_divisors_and_totient (a b c t q : ℕ) (k n : ℕ) 
  (hk : k = c^t) 
  (hn : n = a^k - b^k) 
  (hq : ∃ (p : List ℕ), (∀ x ∈ p, Nat.Prime x) ∧ (List.length p ≥ q) ∧ (∀ x ∈ p, x ∣ k)) :
  (∃ (p : List ℕ), (∀ x ∈ p, Nat.Prime x) ∧ (List.length p ≥ q * t) ∧ (∀ x ∈ p, x ∣ n)) ∧
  (∃ m : ℕ, Nat.totient n = m * 2^(t/2)) := by
  sorry

end NUMINAMATH_CALUDE_prime_divisors_and_totient_l406_40694


namespace NUMINAMATH_CALUDE_total_spent_is_638_l406_40676

/-- The total amount spent by Elizabeth, Emma, and Elsa -/
def total_spent (emma_spent : ℕ) : ℕ :=
  let elsa_spent := 2 * emma_spent
  let elizabeth_spent := 4 * elsa_spent
  emma_spent + elsa_spent + elizabeth_spent

/-- Theorem stating that the total amount spent is 638 -/
theorem total_spent_is_638 : total_spent 58 = 638 := by
  sorry

end NUMINAMATH_CALUDE_total_spent_is_638_l406_40676


namespace NUMINAMATH_CALUDE_initial_amount_simple_interest_l406_40635

/-- Simple interest calculation --/
def simple_interest (principal : ℝ) (rate : ℝ) (time : ℝ) : ℝ :=
  principal * rate * time

/-- Total amount after applying simple interest --/
def total_amount (principal : ℝ) (rate : ℝ) (time : ℝ) : ℝ :=
  principal + simple_interest principal rate time

/-- Theorem: Initial amount in a simple interest scenario --/
theorem initial_amount_simple_interest :
  ∃ (principal : ℝ),
    total_amount principal 0.10 5 = 1125 ∧
    principal = 750 := by
  sorry

end NUMINAMATH_CALUDE_initial_amount_simple_interest_l406_40635


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_times_three_l406_40604

theorem arithmetic_sequence_sum_times_three (a₁ l n : ℕ) (h1 : n = 11) (h2 : a₁ = 101) (h3 : l = 121) :
  3 * (a₁ + (a₁ + 2) + (a₁ + 4) + (a₁ + 6) + (a₁ + 8) + (a₁ + 10) + (a₁ + 12) + (a₁ + 14) + (a₁ + 16) + (a₁ + 18) + l) = 3663 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_times_three_l406_40604


namespace NUMINAMATH_CALUDE_exist_integers_satisfying_equation_l406_40629

theorem exist_integers_satisfying_equation : ∃ (a b : ℤ), a * b * (2 * a + b) = 2015 ∧ a = 13 ∧ b = 5 := by
  sorry

end NUMINAMATH_CALUDE_exist_integers_satisfying_equation_l406_40629


namespace NUMINAMATH_CALUDE_problem_statement_l406_40633

theorem problem_statement :
  -- Part I
  (∀ a b c : ℝ, a^2 + b^2 + c^2 = 1 → |a + b + c| ≤ Real.sqrt 3) ∧
  -- Part II
  {x : ℝ | ∀ a b c : ℝ, a^2 + b^2 + c^2 = 1 → |x - 1| + |x + 1| ≥ (a + b + c)^2} = 
    {x : ℝ | x ≤ -3/2 ∨ x ≥ 3/2} :=
by sorry

end NUMINAMATH_CALUDE_problem_statement_l406_40633


namespace NUMINAMATH_CALUDE_binomial_expansion_sum_l406_40610

theorem binomial_expansion_sum (a₀ a₁ a₂ a₃ a₄ a₅ : ℝ) :
  (∀ x, (1 - 2*x)^5 = a₀ + a₁*(1+x) + a₂*(1+x)^2 + a₃*(1+x)^3 + a₄*(1+x)^4 + a₅*(1+x)^5) →
  a₃ + a₄ = -480 := by
sorry

end NUMINAMATH_CALUDE_binomial_expansion_sum_l406_40610


namespace NUMINAMATH_CALUDE_circular_table_dice_probability_l406_40670

/-- The number of people seated around the table -/
def num_people : ℕ := 5

/-- The number of sides on the die -/
def die_sides : ℕ := 8

/-- The probability of adjacent people not rolling the same number -/
def prob_not_same : ℚ := 7 / 8

/-- The probability that no two adjacent people roll the same number -/
def prob_no_adjacent_same : ℚ := (prob_not_same ^ (num_people - 1))

theorem circular_table_dice_probability :
  prob_no_adjacent_same = 2401 / 4096 := by sorry

end NUMINAMATH_CALUDE_circular_table_dice_probability_l406_40670


namespace NUMINAMATH_CALUDE_operations_to_equality_l406_40638

theorem operations_to_equality (a b : ℕ) (h : a = 515 ∧ b = 53) : 
  ∃ n : ℕ, n = 21 ∧ a - 11 * n = b + 11 * n :=
by sorry

end NUMINAMATH_CALUDE_operations_to_equality_l406_40638


namespace NUMINAMATH_CALUDE_find_adult_ticket_cost_l406_40625

def adult_ticket_cost (total_cost children_cost : ℕ) : Prop :=
  ∃ (adult_cost : ℕ), adult_cost + 6 * children_cost = total_cost

theorem find_adult_ticket_cost :
  adult_ticket_cost 155 20 → ∃ (adult_cost : ℕ), adult_cost = 35 :=
by
  sorry

end NUMINAMATH_CALUDE_find_adult_ticket_cost_l406_40625


namespace NUMINAMATH_CALUDE_complex_magnitude_bounds_l406_40697

/-- Given a complex number z satisfying 2|z-3-3i| = |z|, prove that the maximum value of |z| is 6√2 and the minimum value of |z| is 2√2. -/
theorem complex_magnitude_bounds (z : ℂ) (h : 2 * Complex.abs (z - (3 + 3*I)) = Complex.abs z) :
  (∃ (w : ℂ), 2 * Complex.abs (w - (3 + 3*I)) = Complex.abs w ∧ Complex.abs w = 6 * Real.sqrt 2) ∧
  (∃ (v : ℂ), 2 * Complex.abs (v - (3 + 3*I)) = Complex.abs v ∧ Complex.abs v = 2 * Real.sqrt 2) ∧
  (∀ (u : ℂ), 2 * Complex.abs (u - (3 + 3*I)) = Complex.abs u → 
    2 * Real.sqrt 2 ≤ Complex.abs u ∧ Complex.abs u ≤ 6 * Real.sqrt 2) :=
by sorry

end NUMINAMATH_CALUDE_complex_magnitude_bounds_l406_40697


namespace NUMINAMATH_CALUDE_arithmetic_mean_reciprocals_first_four_primes_l406_40664

theorem arithmetic_mean_reciprocals_first_four_primes : 
  let first_four_primes := [2, 3, 5, 7]
  ((first_four_primes.map (λ x => 1 / x)).sum) / first_four_primes.length = 247 / 840 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_mean_reciprocals_first_four_primes_l406_40664


namespace NUMINAMATH_CALUDE_factorization_x8_minus_81_l406_40693

theorem factorization_x8_minus_81 (x : ℝ) : 
  x^8 - 81 = (x^2 - 3) * (x^2 + 3) * (x^4 + 9) := by
  sorry

end NUMINAMATH_CALUDE_factorization_x8_minus_81_l406_40693


namespace NUMINAMATH_CALUDE_pascals_triangle_row20_l406_40643

def binomial (n k : ℕ) : ℕ := (Nat.factorial n) / (Nat.factorial k * Nat.factorial (n - k))

theorem pascals_triangle_row20 : 
  (binomial 20 6 = 38760) ∧ 
  (binomial 20 6 / binomial 20 2 = 204) := by
sorry

end NUMINAMATH_CALUDE_pascals_triangle_row20_l406_40643


namespace NUMINAMATH_CALUDE_classroom_count_l406_40663

/-- Given a classroom with a 1:2 ratio of girls to boys and 20 boys, prove the total number of students is 30. -/
theorem classroom_count (num_boys : ℕ) (ratio_girls_to_boys : ℚ) : 
  num_boys = 20 → ratio_girls_to_boys = 1/2 → num_boys + (ratio_girls_to_boys * num_boys) = 30 := by
  sorry

end NUMINAMATH_CALUDE_classroom_count_l406_40663


namespace NUMINAMATH_CALUDE_intersection_A_complement_B_A_necessary_not_sufficient_for_B_l406_40683

-- Define sets A and B
def A : Set ℝ := {x | x^2 - 2*x - 8 ≤ 0}
def B (m : ℝ) : Set ℝ := {x | x^2 - 2*x + 1 - m^2 ≤ 0}

-- Part 1
theorem intersection_A_complement_B : 
  (A ∩ (Set.univ \ B 2)) = {x | -2 ≤ x ∧ x < -1 ∨ 3 < x ∧ x ≤ 4} := by sorry

-- Part 2
theorem A_necessary_not_sufficient_for_B :
  (∀ x, x ∈ A → x ∈ B m) ∧ (∃ x, x ∈ B m ∧ x ∉ A) ↔ 0 < m ∧ m < 3 := by sorry

end NUMINAMATH_CALUDE_intersection_A_complement_B_A_necessary_not_sufficient_for_B_l406_40683


namespace NUMINAMATH_CALUDE_intersection_M_N_l406_40652

def M : Set ℝ := {-1, 0, 1}
def N : Set ℝ := {x | x^2 ≤ x}

theorem intersection_M_N : M ∩ N = {0, 1} := by sorry

end NUMINAMATH_CALUDE_intersection_M_N_l406_40652


namespace NUMINAMATH_CALUDE_selection_theorem_l406_40624

def boys : ℕ := 4
def girls : ℕ := 5
def total_selection : ℕ := 5

theorem selection_theorem :
  -- Condition 1
  (Nat.choose boys 2 * Nat.choose (girls - 1) 2 = 36) ∧
  -- Condition 2
  (Nat.choose (boys - 1 + girls - 1) (total_selection - 1) +
   Nat.choose (boys - 1 + girls) total_selection = 91) := by
  sorry

end NUMINAMATH_CALUDE_selection_theorem_l406_40624


namespace NUMINAMATH_CALUDE_max_consecutive_indivisible_l406_40687

def is_five_digit (n : ℕ) : Prop := 10000 ≤ n ∧ n ≤ 99999

def is_indivisible (n : ℕ) : Prop :=
  ∀ a b : ℕ, (100 ≤ a ∧ a ≤ 999) → (100 ≤ b ∧ b ≤ 999) → n ≠ a * b

theorem max_consecutive_indivisible :
  ∀ start : ℕ, is_five_digit start →
    ∃ k : ℕ, k ≤ 99 ∧ ¬(is_indivisible (start + k + 1)) :=
by sorry

end NUMINAMATH_CALUDE_max_consecutive_indivisible_l406_40687


namespace NUMINAMATH_CALUDE_unique_congruence_l406_40648

theorem unique_congruence (n : ℤ) : 3 ≤ n ∧ n ≤ 11 ∧ n ≡ 2023 [ZMOD 7] → n = 7 := by
  sorry

end NUMINAMATH_CALUDE_unique_congruence_l406_40648


namespace NUMINAMATH_CALUDE_soccer_stars_points_l406_40634

/-- Calculates the total points for a soccer team given their game results -/
def total_points (total_games win_games loss_games : ℕ) 
  (win_points draw_points loss_points : ℕ) : ℕ :=
  let draw_games := total_games - win_games - loss_games
  win_games * win_points + draw_games * draw_points + loss_games * loss_points

/-- Theorem stating that Team Soccer Stars earned 46 points at the end of the season -/
theorem soccer_stars_points : 
  total_points 20 14 2 3 1 0 = 46 := by
  sorry

#eval total_points 20 14 2 3 1 0

end NUMINAMATH_CALUDE_soccer_stars_points_l406_40634


namespace NUMINAMATH_CALUDE_proposition_and_variations_l406_40682

theorem proposition_and_variations (x : ℝ) :
  ((x = 3 ∨ x = 7) → (x - 3) * (x - 7) = 0) ∧
  ((x - 3) * (x - 7) = 0 → (x = 3 ∨ x = 7)) ∧
  ((x ≠ 3 ∧ x ≠ 7) → (x - 3) * (x - 7) ≠ 0) ∧
  ((x - 3) * (x - 7) ≠ 0 → (x ≠ 3 ∧ x ≠ 7)) :=
by sorry

end NUMINAMATH_CALUDE_proposition_and_variations_l406_40682


namespace NUMINAMATH_CALUDE_valid_param_iff_l406_40674

/-- A vector parameterization of a line --/
structure VectorParam where
  x₀ : ℝ
  y₀ : ℝ
  dx : ℝ
  dy : ℝ

/-- The line y = 3x - 4 --/
def line (x : ℝ) : ℝ := 3 * x - 4

/-- Predicate for a valid vector parameterization --/
def is_valid_param (p : VectorParam) : Prop :=
  p.y₀ = line p.x₀ ∧ p.dy = 3 * p.dx

/-- Theorem: A vector parameterization is valid iff it satisfies the conditions --/
theorem valid_param_iff (p : VectorParam) :
  is_valid_param p ↔ ∀ t : ℝ, line (p.x₀ + t * p.dx) = p.y₀ + t * p.dy :=
sorry

end NUMINAMATH_CALUDE_valid_param_iff_l406_40674


namespace NUMINAMATH_CALUDE_N_subset_M_l406_40603

-- Define the sets M and N
def M : Set ℝ := {x | |x| ≤ 1}
def N : Set ℝ := {y | ∃ x, y = 2^x ∧ x ≤ 0}

-- State the theorem
theorem N_subset_M : N ⊆ M := by
  sorry

end NUMINAMATH_CALUDE_N_subset_M_l406_40603


namespace NUMINAMATH_CALUDE_chessboard_cut_theorem_l406_40623

/-- Represents a 2×1 domino on a chessboard -/
structure Domino :=
  (x : ℕ) (y : ℕ)

/-- Represents a chessboard configuration -/
structure Chessboard :=
  (n : ℕ)
  (dominoes : List Domino)

/-- Checks if a given line cuts through any domino -/
def line_cuts_domino (board : Chessboard) (line : ℕ) (is_vertical : Bool) : Prop :=
  sorry

/-- Checks if there exists a line that doesn't cut any domino -/
def exists_uncut_line (board : Chessboard) : Prop :=
  sorry

/-- The main theorem -/
theorem chessboard_cut_theorem (board : Chessboard) :
  (board.dominoes.length = 2 * board.n^2) →
  (exists_uncut_line board ↔ board.n = 1 ∨ board.n = 2) :=
sorry

end NUMINAMATH_CALUDE_chessboard_cut_theorem_l406_40623


namespace NUMINAMATH_CALUDE_train_speed_theorem_l406_40607

theorem train_speed_theorem (passing_pole_time passing_train_time stationary_train_length : ℝ) 
  (h1 : passing_pole_time = 12)
  (h2 : passing_train_time = 27)
  (h3 : stationary_train_length = 300) :
  let train_length := (passing_train_time * stationary_train_length) / (passing_train_time - passing_pole_time)
  let train_speed := train_length / passing_pole_time
  train_speed = 20 := by sorry

end NUMINAMATH_CALUDE_train_speed_theorem_l406_40607


namespace NUMINAMATH_CALUDE_parabola_focus_directrix_distance_l406_40636

/-- For a parabola with equation y^2 = ax, if the distance from its focus to its directrix is 2, then a = 4. -/
theorem parabola_focus_directrix_distance (a : ℝ) : 
  (∃ y : ℝ → ℝ, ∀ x, (y x)^2 = a * x) →  -- Parabola equation
  (∃ f d : ℝ, abs (f - d) = 2) →        -- Distance between focus and directrix
  a = 4 := by
sorry

end NUMINAMATH_CALUDE_parabola_focus_directrix_distance_l406_40636


namespace NUMINAMATH_CALUDE_tony_future_age_l406_40617

def jacob_age : ℕ := 24
def tony_age : ℕ := jacob_age / 2
def years_passed : ℕ := 6

theorem tony_future_age :
  tony_age + years_passed = 18 := by
  sorry

end NUMINAMATH_CALUDE_tony_future_age_l406_40617


namespace NUMINAMATH_CALUDE_six_rounds_maximize_configurations_optimal_rounds_is_six_l406_40647

/-- The number of cities and days in the championship --/
def n : ℕ := 8

/-- The number of possible configurations for k rounds --/
def N (k : ℕ) : ℚ :=
  (Nat.factorial n * Nat.factorial n) / (Nat.factorial k * (Nat.factorial (n - k))^2)

/-- The theorem stating that 6 rounds maximizes the number of configurations --/
theorem six_rounds_maximize_configurations :
  ∀ k : ℕ, k ≠ 6 → k ≤ n → N k ≤ N 6 := by
  sorry

/-- The main theorem proving that 6 is the optimal number of rounds --/
theorem optimal_rounds_is_six :
  ∃ k : ℕ, k ≤ n ∧ (∀ j : ℕ, j ≤ n → N j ≤ N k) ∧ k = 6 := by
  sorry

end NUMINAMATH_CALUDE_six_rounds_maximize_configurations_optimal_rounds_is_six_l406_40647


namespace NUMINAMATH_CALUDE_composite_function_value_l406_40639

-- Define the functions f and g
def f (c : ℝ) (x : ℝ) : ℝ := 5 * x + c
def g (c : ℝ) (x : ℝ) : ℝ := c * x + 3

-- State the theorem
theorem composite_function_value (c d : ℝ) :
  (∀ x, f c (g c x) = 15 * x + d) → d = 18 := by
  sorry

end NUMINAMATH_CALUDE_composite_function_value_l406_40639


namespace NUMINAMATH_CALUDE_cubic_root_sum_l406_40657

theorem cubic_root_sum (u v w : ℝ) : 
  (u^3 - 6*u^2 + 11*u - 6 = 0) →
  (v^3 - 6*v^2 + 11*v - 6 = 0) →
  (w^3 - 6*w^2 + 11*w - 6 = 0) →
  u*v/w + v*w/u + w*u/v = 49/6 := by
sorry

end NUMINAMATH_CALUDE_cubic_root_sum_l406_40657


namespace NUMINAMATH_CALUDE_scout_saturday_customers_l406_40611

/-- Scout's delivery earnings over a weekend --/
def scout_earnings (base_pay hourly_rate tip_rate : ℚ) 
                   (saturday_hours sunday_hours : ℚ) 
                   (sunday_customers : ℕ) 
                   (total_earnings : ℚ) : Prop :=
  let saturday_base := base_pay * saturday_hours
  let sunday_base := base_pay * sunday_hours
  let sunday_tips := tip_rate * sunday_customers
  ∃ saturday_customers : ℕ,
    saturday_base + sunday_base + sunday_tips + (tip_rate * saturday_customers) = total_earnings

theorem scout_saturday_customers :
  scout_earnings 10 10 5 4 5 8 155 →
  ∃ saturday_customers : ℕ, saturday_customers = 5 :=
by sorry

end NUMINAMATH_CALUDE_scout_saturday_customers_l406_40611


namespace NUMINAMATH_CALUDE_number_of_boys_in_class_number_of_boys_is_correct_l406_40618

/-- The number of boys in a class given certain weight conditions -/
theorem number_of_boys_in_class : ℕ :=
  let initial_average : ℚ := 58.4
  let misread_weight : ℕ := 56
  let correct_weight : ℕ := 68
  let correct_average : ℚ := 59
  20

theorem number_of_boys_is_correct (n : ℕ) :
  (n : ℚ) * 58.4 + (68 - 56) = n * 59 → n = number_of_boys_in_class :=
by sorry

end NUMINAMATH_CALUDE_number_of_boys_in_class_number_of_boys_is_correct_l406_40618


namespace NUMINAMATH_CALUDE_stratified_sampling_problem_l406_40699

theorem stratified_sampling_problem (total_sample : ℕ) (school_A : ℕ) (school_B : ℕ) (school_C : ℕ) 
  (h1 : total_sample = 60)
  (h2 : school_A = 180)
  (h3 : school_B = 140)
  (h4 : school_C = 160) :
  (total_sample * school_C) / (school_A + school_B + school_C) = 20 :=
by sorry

end NUMINAMATH_CALUDE_stratified_sampling_problem_l406_40699


namespace NUMINAMATH_CALUDE_peter_pictures_l406_40645

theorem peter_pictures (peter_pictures : ℕ) (quincy_pictures : ℕ) (randy_pictures : ℕ)
  (h1 : quincy_pictures = peter_pictures + 20)
  (h2 : randy_pictures + peter_pictures + quincy_pictures = 41)
  (h3 : randy_pictures = 5) :
  peter_pictures = 8 := by
sorry

end NUMINAMATH_CALUDE_peter_pictures_l406_40645


namespace NUMINAMATH_CALUDE_problem_statement_l406_40691

theorem problem_statement (a b m n c : ℝ) 
  (h1 : a + b = 0) 
  (h2 : m * n = 1) 
  (h3 : |c| = 3) : 
  a + b + m * n - |c| = -2 := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l406_40691


namespace NUMINAMATH_CALUDE_min_specialists_needed_l406_40631

/-- Represents the number of specialists in energy efficiency -/
def energy_efficiency : ℕ := 95

/-- Represents the number of specialists in waste management -/
def waste_management : ℕ := 80

/-- Represents the number of specialists in water conservation -/
def water_conservation : ℕ := 110

/-- Represents the number of specialists in both energy efficiency and waste management -/
def energy_waste : ℕ := 30

/-- Represents the number of specialists in both waste management and water conservation -/
def waste_water : ℕ := 35

/-- Represents the number of specialists in both energy efficiency and water conservation -/
def energy_water : ℕ := 25

/-- Represents the number of specialists in all three areas -/
def all_three : ℕ := 15

/-- Theorem stating the minimum number of specialists needed -/
theorem min_specialists_needed : 
  energy_efficiency + waste_management + water_conservation - 
  energy_waste - waste_water - energy_water + all_three = 210 := by
  sorry

end NUMINAMATH_CALUDE_min_specialists_needed_l406_40631


namespace NUMINAMATH_CALUDE_trig_identity_l406_40658

theorem trig_identity (x : ℝ) (h : Real.sin (π / 6 - x) = 1 / 2) :
  Real.sin (19 * π / 6 - x) + (Real.sin (-2 * π / 3 + x))^2 = 1 / 4 := by
  sorry

end NUMINAMATH_CALUDE_trig_identity_l406_40658


namespace NUMINAMATH_CALUDE_log_343_equation_solution_l406_40605

theorem log_343_equation_solution (x : ℝ) : 
  (Real.log 343 / Real.log (3 * x) = x) → 
  (∃ (a b : ℤ), x = a / b ∧ b ≠ 0 ∧ ¬∃ (n : ℤ), x = n ∧ ¬∃ (m : ℚ), x = m ^ 2 ∧ ¬∃ (k : ℚ), x = k ^ 3) := by
  sorry

end NUMINAMATH_CALUDE_log_343_equation_solution_l406_40605


namespace NUMINAMATH_CALUDE_clay_capacity_scaling_l406_40684

/-- Represents the dimensions of a rectangular box -/
structure BoxDimensions where
  depth : ℝ
  width : ℝ
  length : ℝ

/-- Calculates the volume of a box given its dimensions -/
def boxVolume (d : BoxDimensions) : ℝ := d.depth * d.width * d.length

/-- Theorem: Given a box with dimensions 3x4x6 cm holding 60g of clay,
    a box with dimensions 9x16x6 cm will hold 720g of clay -/
theorem clay_capacity_scaling (clayMass₁ : ℝ) :
  let box₁ : BoxDimensions := ⟨3, 4, 6⟩
  let box₂ : BoxDimensions := ⟨9, 16, 6⟩
  clayMass₁ = 60 →
  (boxVolume box₂ / boxVolume box₁) * clayMass₁ = 720 := by
  sorry

end NUMINAMATH_CALUDE_clay_capacity_scaling_l406_40684


namespace NUMINAMATH_CALUDE_distance_A_proof_l406_40619

/-- The distance that runner A can run, given the conditions of the problem -/
def distance_A : ℝ := 224

theorem distance_A_proof (time_A time_B beat_distance : ℝ) 
  (h1 : time_A = 28)
  (h2 : time_B = 32)
  (h3 : beat_distance = 32)
  (h4 : distance_A / time_A * time_B = distance_A + beat_distance) : 
  distance_A = 224 := by sorry

end NUMINAMATH_CALUDE_distance_A_proof_l406_40619


namespace NUMINAMATH_CALUDE_coin_count_l406_40630

theorem coin_count (total_value : ℕ) (two_dollar_coins : ℕ) : 
  total_value = 402 → two_dollar_coins = 148 → 
  ∃ (one_dollar_coins : ℕ), 
    total_value = 2 * two_dollar_coins + one_dollar_coins ∧
    one_dollar_coins + two_dollar_coins = 254 :=
by
  sorry

end NUMINAMATH_CALUDE_coin_count_l406_40630


namespace NUMINAMATH_CALUDE_optimal_triangle_sides_l406_40686

noncomputable def minTriangleSides (S : ℝ) (x : ℝ) : ℝ × ℝ × ℝ :=
  let BC := 2 * Real.sqrt (S * Real.tan (x / 2))
  let AB := Real.sqrt (S / (Real.sin (x / 2) * Real.cos (x / 2)))
  (BC, AB, AB)

theorem optimal_triangle_sides (S : ℝ) (x : ℝ) (h1 : 0 < S) (h2 : 0 < x) (h3 : x < π) :
  let (BC, AB, AC) := minTriangleSides S x
  BC = 2 * Real.sqrt (S * Real.tan (x / 2)) ∧
  AB = AC ∧
  AB = Real.sqrt (S / (Real.sin (x / 2) * Real.cos (x / 2))) ∧
  ∀ (BC' AB' AC' : ℝ), 
    (BC' * AB' * Real.sin x) / 2 = S → 
    BC' ≥ BC :=
by sorry

end NUMINAMATH_CALUDE_optimal_triangle_sides_l406_40686


namespace NUMINAMATH_CALUDE_average_difference_l406_40622

/-- The total number of students in the school -/
def total_students : ℕ := 120

/-- The class sizes -/
def class_sizes : List ℕ := [60, 30, 20, 5, 5]

/-- The number of teachers -/
def total_teachers : ℕ := 6

/-- The number of teaching teachers -/
def teaching_teachers : ℕ := 5

/-- Average class size from teaching teachers' perspective -/
def t : ℚ := (List.sum class_sizes) / teaching_teachers

/-- Average class size from students' perspective -/
def s : ℚ := (List.sum (List.map (λ x => x * x) class_sizes)) / total_students

theorem average_difference : t - s = -17.25 := by sorry

end NUMINAMATH_CALUDE_average_difference_l406_40622


namespace NUMINAMATH_CALUDE_compound_interest_problem_l406_40601

/-- Compound interest calculation --/
def compound_interest (principal : ℝ) (rate : ℝ) (time : ℕ) : ℝ :=
  principal * ((1 + rate) ^ time - 1)

/-- Theorem statement --/
theorem compound_interest_problem :
  let principal : ℝ := 500
  let rate : ℝ := 0.05
  let time : ℕ := 5
  let interest : ℝ := compound_interest principal rate time
  ∃ ε > 0, |interest - 138.14| < ε :=
by sorry

end NUMINAMATH_CALUDE_compound_interest_problem_l406_40601


namespace NUMINAMATH_CALUDE_units_digit_of_S_is_3_l406_40655

def factorial (n : ℕ) : ℕ := (List.range n).foldl (· * ·) 1

def S : ℕ := (List.range 12).map (λ i => factorial (i + 1)) |>.sum

theorem units_digit_of_S_is_3 : S % 10 = 3 := by
  sorry

end NUMINAMATH_CALUDE_units_digit_of_S_is_3_l406_40655


namespace NUMINAMATH_CALUDE_count_integers_satisfying_inequality_l406_40612

theorem count_integers_satisfying_inequality :
  ∃! (S : Finset ℤ), (∀ n ∈ S, Real.sqrt (3 * n) ≤ Real.sqrt (5 * n - 8) ∧
                                Real.sqrt (5 * n - 8) < Real.sqrt (3 * n + 7)) ∧
                     S.card = 4 := by
  sorry

end NUMINAMATH_CALUDE_count_integers_satisfying_inequality_l406_40612


namespace NUMINAMATH_CALUDE_factory_earnings_l406_40609

/-- Represents a factory with machines producing material --/
structure Factory where
  original_machines : ℕ
  original_hours : ℕ
  new_machines : ℕ
  new_hours : ℕ
  production_rate : ℕ
  price_per_kg : ℕ

/-- Calculates the daily earnings of the factory --/
def daily_earnings (f : Factory) : ℕ :=
  ((f.original_machines * f.original_hours + f.new_machines * f.new_hours) * f.production_rate) * f.price_per_kg

/-- Theorem stating that the factory's daily earnings are $8100 --/
theorem factory_earnings :
  ∃ (f : Factory), 
    f.original_machines = 3 ∧
    f.original_hours = 23 ∧
    f.new_machines = 1 ∧
    f.new_hours = 12 ∧
    f.production_rate = 2 ∧
    f.price_per_kg = 50 ∧
    daily_earnings f = 8100 := by
  sorry


end NUMINAMATH_CALUDE_factory_earnings_l406_40609


namespace NUMINAMATH_CALUDE_cubic_fraction_factorization_l406_40653

theorem cubic_fraction_factorization (a b c : ℝ) :
  ((a^3 - b^3)^3 + (b^3 - c^3)^3 + (c^3 - a^3)^3) / ((a - b)^3 + (b - c)^3 + (c - a)^3) 
  = (a^2 + a*b + b^2) * (b^2 + b*c + c^2) * (c^2 + c*a + a^2) := by
  sorry

end NUMINAMATH_CALUDE_cubic_fraction_factorization_l406_40653


namespace NUMINAMATH_CALUDE_display_rows_for_225_cans_l406_40661

/-- Represents a pyramidal display of cans -/
structure CanDisplay where
  rows : ℕ

/-- The number of cans in the nth row of the display -/
def cans_in_row (n : ℕ) : ℕ := 4 * n - 3

/-- The total number of cans in the display -/
def total_cans (d : CanDisplay) : ℕ :=
  (d.rows * (cans_in_row 1 + cans_in_row d.rows)) / 2

/-- The theorem stating that a display with 225 cans has 11 rows -/
theorem display_rows_for_225_cans :
  ∃ (d : CanDisplay), total_cans d = 225 ∧ d.rows = 11 :=
by sorry

end NUMINAMATH_CALUDE_display_rows_for_225_cans_l406_40661


namespace NUMINAMATH_CALUDE_carls_paintable_area_l406_40616

/-- Calculates the total paintable area in square feet for a given number of bedrooms --/
def total_paintable_area (num_bedrooms : ℕ) (length width height : ℝ) (unpaintable_area : ℝ) : ℝ :=
  let wall_area := 2 * (length * height + width * height)
  let paintable_area_per_room := wall_area - unpaintable_area
  num_bedrooms * paintable_area_per_room

/-- The total paintable area for Carl's bedrooms is 1552 square feet --/
theorem carls_paintable_area :
  total_paintable_area 4 15 11 9 80 = 1552 := by
sorry

end NUMINAMATH_CALUDE_carls_paintable_area_l406_40616


namespace NUMINAMATH_CALUDE_fruit_drink_composition_l406_40615

/-- Represents a fruit drink mixture -/
structure FruitDrink where
  total_volume : ℝ
  grape_volume : ℝ
  orange_percent : ℝ
  watermelon_percent : ℝ
  grape_percent : ℝ

/-- The theorem statement -/
theorem fruit_drink_composition (drink : FruitDrink)
  (h1 : drink.total_volume = 150)
  (h2 : drink.grape_volume = 45)
  (h3 : drink.orange_percent = drink.watermelon_percent)
  (h4 : drink.orange_percent + drink.watermelon_percent + drink.grape_percent = 100)
  (h5 : drink.grape_volume / drink.total_volume * 100 = drink.grape_percent) :
  drink.orange_percent = 35 ∧ drink.watermelon_percent = 35 := by
  sorry

end NUMINAMATH_CALUDE_fruit_drink_composition_l406_40615


namespace NUMINAMATH_CALUDE_consecutive_integers_product_l406_40681

theorem consecutive_integers_product (a b c d e : ℕ) : 
  a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0 ∧ e > 0 →
  b = a + 1 →
  c = b + 1 →
  d = c + 1 →
  e = d + 1 →
  a * b * c * d * e = 15120 →
  e = 9 := by
sorry

end NUMINAMATH_CALUDE_consecutive_integers_product_l406_40681


namespace NUMINAMATH_CALUDE_f_max_value_f_no_real_roots_l406_40608

-- Define the function
def f (x : ℝ) : ℝ := -2 * x^2 + 8 * x - 10

-- Theorem for the maximum value
theorem f_max_value :
  ∃ (x_max : ℝ), f x_max = -2 ∧ ∀ (x : ℝ), f x ≤ f x_max ∧ x_max = 2 :=
sorry

-- Theorem for no real roots
theorem f_no_real_roots :
  ∀ (x : ℝ), f x ≠ 0 :=
sorry

end NUMINAMATH_CALUDE_f_max_value_f_no_real_roots_l406_40608


namespace NUMINAMATH_CALUDE_functional_equation_solution_l406_40673

-- Define the property that a function f must satisfy
def SatisfiesEquation (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, f x * f y + f (x + y) = x * y

-- State the theorem
theorem functional_equation_solution :
  ∀ f : ℝ → ℝ, SatisfiesEquation f ↔ (∀ x, f x = x - 1) ∨ (∀ x, f x = -x - 1) :=
by sorry

end NUMINAMATH_CALUDE_functional_equation_solution_l406_40673


namespace NUMINAMATH_CALUDE_cindy_envelopes_left_l406_40662

/-- Calculates the number of envelopes Cindy has left after giving some to her friends -/
def envelopes_left (initial : ℕ) (friends : ℕ) (per_friend : ℕ) : ℕ :=
  initial - friends * per_friend

/-- Proves that Cindy has 22 envelopes left -/
theorem cindy_envelopes_left : 
  envelopes_left 37 5 3 = 22 := by
  sorry

end NUMINAMATH_CALUDE_cindy_envelopes_left_l406_40662


namespace NUMINAMATH_CALUDE_christine_wandering_l406_40620

/-- The distance traveled given a constant speed and time -/
def distance (speed : ℝ) (time : ℝ) : ℝ := speed * time

/-- Christine's wandering problem -/
theorem christine_wandering (christine_speed : ℝ) (christine_time : ℝ) 
  (h1 : christine_speed = 4)
  (h2 : christine_time = 5) :
  distance christine_speed christine_time = 20 := by
  sorry

end NUMINAMATH_CALUDE_christine_wandering_l406_40620


namespace NUMINAMATH_CALUDE_beatrice_tv_ratio_l406_40696

/-- Proves that the ratio of TVs Beatrice looked at in the online store to the first store is 3:1 -/
theorem beatrice_tv_ratio : 
  ∀ (first_store online_store auction_site total : ℕ),
  first_store = 8 →
  auction_site = 10 →
  total = 42 →
  first_store + online_store + auction_site = total →
  online_store / first_store = 3 := by
sorry

end NUMINAMATH_CALUDE_beatrice_tv_ratio_l406_40696


namespace NUMINAMATH_CALUDE_purple_flowers_killed_is_40_l406_40637

/-- Represents the florist's bouquet problem -/
structure BouquetProblem where
  flowers_per_bouquet : ℕ
  initial_seeds_per_color : ℕ
  num_colors : ℕ
  red_killed : ℕ
  yellow_killed : ℕ
  orange_killed : ℕ
  bouquets_made : ℕ

/-- Calculates the number of purple flowers killed by the fungus -/
def purple_flowers_killed (problem : BouquetProblem) : ℕ :=
  let total_initial := problem.initial_seeds_per_color * problem.num_colors
  let red_left := problem.initial_seeds_per_color - problem.red_killed
  let yellow_left := problem.initial_seeds_per_color - problem.yellow_killed
  let orange_left := problem.initial_seeds_per_color - problem.orange_killed
  let total_needed := problem.flowers_per_bouquet * problem.bouquets_made
  let non_purple_left := red_left + yellow_left + orange_left
  problem.initial_seeds_per_color - (total_needed - non_purple_left)

/-- Theorem stating that the number of purple flowers killed is 40 -/
theorem purple_flowers_killed_is_40 (problem : BouquetProblem) 
    (h1 : problem.flowers_per_bouquet = 9)
    (h2 : problem.initial_seeds_per_color = 125)
    (h3 : problem.num_colors = 4)
    (h4 : problem.red_killed = 45)
    (h5 : problem.yellow_killed = 61)
    (h6 : problem.orange_killed = 30)
    (h7 : problem.bouquets_made = 36) :
    purple_flowers_killed problem = 40 := by
  sorry

end NUMINAMATH_CALUDE_purple_flowers_killed_is_40_l406_40637


namespace NUMINAMATH_CALUDE_average_speed_two_hours_l406_40656

/-- Calculates the average speed of a car given its speeds in two consecutive hours -/
theorem average_speed_two_hours (speed1 speed2 : ℝ) (h1 : speed1 = 90) (h2 : speed2 = 80) :
  (speed1 + speed2) / 2 = 85 := by
  sorry

end NUMINAMATH_CALUDE_average_speed_two_hours_l406_40656


namespace NUMINAMATH_CALUDE_box_volume_l406_40641

/-- The volume of a rectangular box with given dimensions -/
theorem box_volume (height length width : ℝ) 
  (h_height : height = 12)
  (h_length : length = 3 * height)
  (h_width : width = length / 4) :
  height * length * width = 3888 :=
by sorry

end NUMINAMATH_CALUDE_box_volume_l406_40641


namespace NUMINAMATH_CALUDE_one_third_percent_of_180_l406_40675

theorem one_third_percent_of_180 : (1 / 3) * (1 / 100) * 180 = 0.6 := by
  sorry

end NUMINAMATH_CALUDE_one_third_percent_of_180_l406_40675


namespace NUMINAMATH_CALUDE_brown_mice_count_l406_40646

theorem brown_mice_count (total : ℕ) (white : ℕ) : 
  (2 : ℚ) / 3 * total = white → white = 14 → total - white = 7 := by
  sorry

end NUMINAMATH_CALUDE_brown_mice_count_l406_40646


namespace NUMINAMATH_CALUDE_expression_factorization_l406_40669

theorem expression_factorization (a b c : ℝ) : 
  a^4 * (b^3 - c^3) + b^4 * (c^3 - a^3) + c^4 * (a^3 - b^3) = 
  (a - b) * (b - c) * (c - a) * (-(a + b + c) * (a^2 + b^2 + c^2 + a*b + b*c + a*c)) := by
  sorry

end NUMINAMATH_CALUDE_expression_factorization_l406_40669


namespace NUMINAMATH_CALUDE_oranges_sum_l406_40688

/-- The number of oranges Janet has -/
def janet_oranges : ℕ := 9

/-- The number of oranges Sharon has -/
def sharon_oranges : ℕ := 7

/-- The total number of oranges Janet and Sharon have together -/
def total_oranges : ℕ := janet_oranges + sharon_oranges

theorem oranges_sum : total_oranges = 16 := by
  sorry

end NUMINAMATH_CALUDE_oranges_sum_l406_40688
