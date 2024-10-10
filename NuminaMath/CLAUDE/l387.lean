import Mathlib

namespace alex_overall_score_l387_38750

def quiz_problems : ℕ := 30
def test_problems : ℕ := 50
def exam_problems : ℕ := 20

def quiz_score : ℚ := 75 / 100
def test_score : ℚ := 85 / 100
def exam_score : ℚ := 80 / 100

def total_problems : ℕ := quiz_problems + test_problems + exam_problems

def correct_problems : ℚ := 
  quiz_score * quiz_problems + test_score * test_problems + exam_score * exam_problems

theorem alex_overall_score : correct_problems / total_problems = 81 / 100 := by
  sorry

end alex_overall_score_l387_38750


namespace samples_left_over_proof_l387_38715

/-- Calculates the number of samples left over given the number of samples per box,
    number of boxes opened, and number of customers who tried a sample. -/
def samples_left_over (samples_per_box : ℕ) (boxes_opened : ℕ) (customers : ℕ) : ℕ :=
  samples_per_box * boxes_opened - customers

/-- Proves that given 20 samples per box, 12 boxes opened, and 235 customers,
    the number of samples left over is 5. -/
theorem samples_left_over_proof :
  samples_left_over 20 12 235 = 5 := by
  sorry

end samples_left_over_proof_l387_38715


namespace parabola_axis_of_symmetry_l387_38746

/-- The axis of symmetry of the parabola y = 2x² is the line x = 0 -/
theorem parabola_axis_of_symmetry :
  let f : ℝ → ℝ := fun x ↦ 2 * x^2
  ∀ x y : ℝ, f (x) = f (-x) → x = 0 :=
by sorry

end parabola_axis_of_symmetry_l387_38746


namespace tan_product_30_60_l387_38780

theorem tan_product_30_60 : 
  (1 + Real.tan (30 * π / 180)) * (1 + Real.tan (60 * π / 180)) = 2 + 4 * Real.sqrt 3 / 3 := by
  sorry

end tan_product_30_60_l387_38780


namespace triangle_properties_l387_38703

/-- Given a triangle ABC with specific properties, prove its angle A and area. -/
theorem triangle_properties (a b c : ℝ) (A B C : ℝ) : 
  -- Triangle ABC exists
  0 < a ∧ 0 < b ∧ 0 < c ∧ 0 < A ∧ 0 < B ∧ 0 < C →
  -- Sum of angles is π
  A + B + C = π →
  -- Side lengths satisfy triangle inequality
  a + b > c ∧ b + c > a ∧ c + a > b →
  -- Given conditions
  Real.sin A + Real.sqrt 3 * Real.cos A = 2 →
  a = 2 →
  B = π / 4 →
  -- Prove angle A and area
  A = π / 6 ∧ 
  (1/2 : ℝ) * a * b * Real.sin C = Real.sqrt 3 + 1 := by
sorry

end triangle_properties_l387_38703


namespace annual_growth_rate_l387_38702

theorem annual_growth_rate (initial_amount final_amount : ℝ) (h : initial_amount * (1 + 0.125)^2 = final_amount) :
  ∃ (rate : ℝ), initial_amount * (1 + rate)^2 = final_amount ∧ rate = 0.125 := by
  sorry

end annual_growth_rate_l387_38702


namespace constants_are_like_terms_different_variables_not_like_terms_different_exponents_not_like_terms_like_terms_classification_l387_38720

/-- Represents an algebraic term --/
inductive Term
  | Constant (n : ℕ)
  | Variable (name : String)
  | Product (terms : List Term)

/-- Defines when two terms are like terms --/
def areLikeTerms (t1 t2 : Term) : Prop :=
  match t1, t2 with
  | Term.Constant _, Term.Constant _ => True
  | Term.Variable x, Term.Variable y => x = y
  | Term.Product l1, Term.Product l2 => l1 = l2
  | _, _ => False

/-- Theorem stating that constants are like terms --/
theorem constants_are_like_terms (a b : ℕ) :
  areLikeTerms (Term.Constant a) (Term.Constant b) := by sorry

/-- Theorem stating that terms with different variables are not like terms --/
theorem different_variables_not_like_terms (x y : String) (h : x ≠ y) :
  ¬ areLikeTerms (Term.Variable x) (Term.Variable y) := by sorry

/-- Theorem stating that terms with different exponents are not like terms --/
theorem different_exponents_not_like_terms (x : String) (a b : ℕ) (h : a ≠ b) :
  ¬ areLikeTerms 
    (Term.Product [Term.Variable x, Term.Constant a]) 
    (Term.Product [Term.Variable x, Term.Constant b]) := by sorry

/-- Main theorem combining the results for the given problem --/
theorem like_terms_classification 
  (a b : ℕ) 
  (x y z : String) 
  (h1 : x ≠ y) 
  (h2 : y ≠ z) 
  (h3 : x ≠ z) :
  areLikeTerms (Term.Constant a) (Term.Constant b) ∧
  ¬ areLikeTerms 
    (Term.Product [Term.Variable x, Term.Variable x, Term.Variable y])
    (Term.Product [Term.Variable y, Term.Variable y, Term.Variable x]) ∧
  ¬ areLikeTerms 
    (Term.Product [Term.Variable x, Term.Variable y])
    (Term.Product [Term.Variable y, Term.Variable z]) ∧
  ¬ areLikeTerms 
    (Term.Product [Term.Variable x, Term.Variable y])
    (Term.Product [Term.Variable x, Term.Variable y, Term.Variable z]) := by sorry

end constants_are_like_terms_different_variables_not_like_terms_different_exponents_not_like_terms_like_terms_classification_l387_38720


namespace catch_up_time_meeting_distance_l387_38790

def distance_AB : ℝ := 46
def speed_A : ℝ := 15
def speed_B : ℝ := 40
def time_difference : ℝ := 1

-- Time for Person B to catch up with Person A
theorem catch_up_time : 
  ∃ t : ℝ, speed_B * t = speed_A * (t + time_difference) ∧ t = 3/5 := by sorry

-- Distance from point B where they meet on Person B's return journey
theorem meeting_distance : 
  ∃ y : ℝ, 
    (distance_AB - y) / speed_A - (distance_AB + y) / speed_B = time_difference ∧ 
    y = 10 := by sorry

end catch_up_time_meeting_distance_l387_38790


namespace polynomial_factorization_l387_38788

theorem polynomial_factorization (x y : ℝ) :
  x^8 - x^7*y + x^6*y^2 - x^5*y^3 + x^4*y^4 - x^3*y^5 + x^2*y^6 - x*y^7 + y^8 =
  (x^2 - x*y + y^2) * (x^6 - x^3*y^3 + y^6) := by
  sorry

end polynomial_factorization_l387_38788


namespace intersection_A_B_l387_38727

-- Define the function f
def f (x : ℝ) : ℝ := x^2 - 2*x

-- Define set A
def A : Set ℝ := {x | f x < 0}

-- Define set B
def B : Set ℝ := {x | (deriv f) x > 0}

-- State the theorem
theorem intersection_A_B : A ∩ B = {x : ℝ | 1 < x ∧ x < 2} := by sorry

end intersection_A_B_l387_38727


namespace coefficient_x_cubed_in_expansion_l387_38777

theorem coefficient_x_cubed_in_expansion :
  let n : ℕ := 5
  let a : ℤ := 3
  let r : ℕ := 3
  let coeff : ℤ := (n.choose r) * a^(n-r) * (-1)^r
  coeff = -90 := by sorry

end coefficient_x_cubed_in_expansion_l387_38777


namespace family_divisors_characterization_l387_38741

/-- Represents a six-digit number and its family -/
def SixDigitFamily :=
  {A : ℕ // A ≥ 100000 ∧ A < 1000000}

/-- Generates the k-th member of the family for a six-digit number -/
def family_member (A : SixDigitFamily) (k : Fin 6) : ℕ :=
  let B := A.val / (10^k.val)
  let C := A.val % (10^k.val)
  10^(6-k.val) * C + B

/-- The set of numbers that divide all members of a six-digit number's family -/
def family_divisors (A : SixDigitFamily) : Set ℕ :=
  {x : ℕ | ∀ k : Fin 6, (family_member A k) % x = 0}

/-- The set of numbers we're proving to be the family_divisors -/
def target_set : Set ℕ :=
  {x : ℕ | x ≥ 1000000 ∨ 
           (∃ h : Fin 9, x = 111111 * (h.val + 1)) ∨
           999999 % x = 0}

/-- The main theorem stating that family_divisors is a subset of target_set -/
theorem family_divisors_characterization (A : SixDigitFamily) :
  family_divisors A ⊆ target_set := by
  sorry


end family_divisors_characterization_l387_38741


namespace train_speeds_l387_38716

-- Define the problem parameters
def distance : ℝ := 450
def time : ℝ := 5
def speed_difference : ℝ := 6

-- Define the theorem
theorem train_speeds (slower_speed faster_speed : ℝ) : 
  slower_speed > 0 ∧ 
  faster_speed = slower_speed + speed_difference ∧
  distance = (slower_speed + faster_speed) * time →
  slower_speed = 42 ∧ faster_speed = 48 := by
sorry

end train_speeds_l387_38716


namespace goldbach_refutation_l387_38763

theorem goldbach_refutation (n : ℕ) : 
  (∃ n : ℕ, n > 2 ∧ Even n ∧ ¬∃ p q : ℕ, Prime p ∧ Prime q ∧ n = p + q) → 
  ¬(∀ n : ℕ, n > 2 → Even n → ∃ p q : ℕ, Prime p ∧ Prime q ∧ n = p + q) :=
by sorry

end goldbach_refutation_l387_38763


namespace triangle_angle_c_is_right_angle_l387_38734

/-- Given a triangle ABC, if |sin A - 1/2| and (tan B - √3)² are opposite in sign, 
    then angle C is 90°. -/
theorem triangle_angle_c_is_right_angle 
  (A B C : ℝ) -- Angles of the triangle
  (h_triangle : A + B + C = PI) -- Sum of angles in a triangle is π radians (180°)
  (h_opposite_sign : (|Real.sin A - 1/2| * (Real.tan B - Real.sqrt 3)^2 < 0)) -- Opposite sign condition
  : C = PI / 2 := by -- C is π/2 radians (90°)
  sorry

end triangle_angle_c_is_right_angle_l387_38734


namespace least_k_for_convergence_l387_38736

def u : ℕ → ℚ
  | 0 => 1/8
  | n + 1 => 3 * u n - 5 * (u n)^2

def L : ℚ := 1/5

theorem least_k_for_convergence :
  ∀ k < 7, |u k - L| > 1/2^100 ∧ |u 7 - L| ≤ 1/2^100 := by sorry

end least_k_for_convergence_l387_38736


namespace largest_c_value_l387_38731

theorem largest_c_value : ∃ (c_max : ℚ), 
  (∀ c : ℚ, (3 * c + 4) * (c - 2) = 9 * c → c ≤ c_max) ∧ 
  ((3 * c_max + 4) * (c_max - 2) = 9 * c_max) ∧
  c_max = 4 := by
  sorry

end largest_c_value_l387_38731


namespace root_equation_implies_expression_value_l387_38724

theorem root_equation_implies_expression_value (x₀ : ℝ) (h : x₀ > 0) :
  x₀^3 * Real.exp (x₀ - 4) + 2 * Real.log x₀ - 4 = 0 →
  Real.exp ((4 - x₀) / 2) + 2 * Real.log x₀ = 4 := by
  sorry

end root_equation_implies_expression_value_l387_38724


namespace point_comparison_l387_38700

/-- Given points in a 2D coordinate system, prove that a > c -/
theorem point_comparison (a b c d e f : ℝ) : 
  b > 0 →  -- (a, b) is above x-axis
  d > 0 →  -- (c, d) is above x-axis
  f < 0 →  -- (e, f) is below x-axis
  a > 0 →  -- (a, b) is to the right of y-axis
  c > 0 →  -- (c, d) is to the right of y-axis
  e < 0 →  -- (e, f) is to the left of y-axis
  a > c →  -- (a, b) is horizontally farther from y-axis than (c, d)
  b > d →  -- (a, b) is vertically farther from x-axis than (c, d)
  a > c :=
by sorry

end point_comparison_l387_38700


namespace shoe_company_earnings_l387_38751

/-- Proves that the current monthly earnings of a shoe company are $4000,
    given their annual goal and required monthly increase. -/
theorem shoe_company_earnings (annual_goal : ℕ) (monthly_increase : ℕ) (months_per_year : ℕ) :
  annual_goal = 60000 →
  monthly_increase = 1000 →
  months_per_year = 12 →
  (annual_goal / months_per_year - monthly_increase : ℕ) = 4000 := by
  sorry

end shoe_company_earnings_l387_38751


namespace quadratic_inequality_equivalence_l387_38771

theorem quadratic_inequality_equivalence (a : ℝ) : 
  (∀ x : ℝ, x^2 + a*x - 2*a ≥ 0) ↔ (-8 ≤ a ∧ a ≤ 0) := by sorry

end quadratic_inequality_equivalence_l387_38771


namespace log_xy_value_l387_38713

theorem log_xy_value (x y : ℝ) (hxy3 : Real.log (x * y^3) = 1) (hx2y : Real.log (x^2 * y) = 1) :
  Real.log (x * y) = 3/5 := by
  sorry

end log_xy_value_l387_38713


namespace divisibility_condition_l387_38783

/-- Represents a four-digit number MCUD -/
structure FourDigitNumber where
  M : Nat
  C : Nat
  D : Nat
  U : Nat
  h_M : M < 10
  h_C : C < 10
  h_D : D < 10
  h_U : U < 10

/-- Calculates the value of a four-digit number -/
def FourDigitNumber.value (n : FourDigitNumber) : Nat :=
  1000 * n.M + 100 * n.C + 10 * n.D + n.U

/-- Calculates the remainders r₁, r₂, and r₃ for a given divisor -/
def calculateRemainders (A : Nat) : Nat × Nat × Nat :=
  let r₁ := 10 % A
  let r₂ := (10 * r₁) % A
  let r₃ := (10 * r₂) % A
  (r₁, r₂, r₃)

/-- The main theorem stating the divisibility condition -/
theorem divisibility_condition (n : FourDigitNumber) (A : Nat) (hA : A > 0) :
  A ∣ n.value ↔ A ∣ (n.U + n.D * (calculateRemainders A).1 + n.C * (calculateRemainders A).2.1 + n.M * (calculateRemainders A).2.2) :=
sorry

end divisibility_condition_l387_38783


namespace expression_evaluation_l387_38744

theorem expression_evaluation (c d : ℤ) (hc : c = 2) (hd : d = 3) :
  (c^3 + d^2)^2 - (c^3 - d^2)^2 = 288 := by
  sorry

end expression_evaluation_l387_38744


namespace chip_consumption_theorem_l387_38722

/-- Calculates the total number of bags of chips consumed in a week -/
def weekly_chip_consumption (breakfast_bags : ℕ) (lunch_bags : ℕ) (days_in_week : ℕ) : ℕ :=
  let dinner_bags := 2 * lunch_bags
  let daily_consumption := breakfast_bags + lunch_bags + dinner_bags
  daily_consumption * days_in_week

/-- Theorem stating that consuming 1 bag for breakfast, 2 for lunch, and doubling lunch for dinner
    every day for a week results in 49 bags consumed -/
theorem chip_consumption_theorem :
  weekly_chip_consumption 1 2 7 = 49 := by
  sorry

#eval weekly_chip_consumption 1 2 7

end chip_consumption_theorem_l387_38722


namespace exists_abs_less_than_one_l387_38726

def sequence_property (a : ℕ → ℝ) : Prop :=
  (a 1 * a 2 < 0) ∧
  (∀ n > 2, ∃ i j, 1 ≤ i ∧ i < j ∧ j < n ∧
    a n = a i + a j ∧
    ∀ k l, 1 ≤ k ∧ k < l ∧ l < n → |a i + a j| ≤ |a k + a l|)

theorem exists_abs_less_than_one (a : ℕ → ℝ) (h : sequence_property a) :
  ∃ i : ℕ, |a i| < 1 := by sorry

end exists_abs_less_than_one_l387_38726


namespace josh_marbles_count_l387_38759

def final_marbles (initial found traded broken : ℕ) : ℕ :=
  initial + found - traded - broken

theorem josh_marbles_count : final_marbles 357 146 32 10 = 461 := by
  sorry

end josh_marbles_count_l387_38759


namespace equilateral_triangle_height_equals_rectangle_width_l387_38725

theorem equilateral_triangle_height_equals_rectangle_width (w : ℝ) :
  let rectangle_area := 2 * w^2
  let triangle_side := (2 * w^2 * 4 / Real.sqrt 3).sqrt
  let triangle_height := triangle_side * Real.sqrt 3 / 2
  triangle_height = w * Real.sqrt 6 := by sorry

end equilateral_triangle_height_equals_rectangle_width_l387_38725


namespace exam_outcomes_count_l387_38778

/-- The number of possible outcomes for n people in a qualification exam -/
def exam_outcomes (n : ℕ) : ℕ := 2^n

/-- Theorem: The number of possible outcomes for n people in a qualification exam is 2^n -/
theorem exam_outcomes_count (n : ℕ) : exam_outcomes n = 2^n := by
  sorry

end exam_outcomes_count_l387_38778


namespace angle_properties_l387_38714

/-- Given that the terminal side of angle α passes through point P(5a, -12a) where a < 0,
    prove that tan α = -12/5 and sin α + cos α = 7/13 -/
theorem angle_properties (a : ℝ) (α : ℝ) (h : a < 0) :
  let x := 5 * a
  let y := -12 * a
  let r := Real.sqrt (x^2 + y^2)
  (Real.tan α = -12/5) ∧ (Real.sin α + Real.cos α = 7/13) := by
sorry

end angle_properties_l387_38714


namespace factorial_nine_mod_eleven_l387_38728

theorem factorial_nine_mod_eleven : Nat.factorial 9 % 11 = 1 := by
  sorry

end factorial_nine_mod_eleven_l387_38728


namespace fraction_inequality_l387_38794

theorem fraction_inequality (x : ℝ) : x / (x + 3) ≥ 0 ↔ x ∈ Set.Ici 0 ∪ Set.Iic (-3) :=
sorry

end fraction_inequality_l387_38794


namespace a_equals_b_l387_38787

theorem a_equals_b (a b : ℝ) : 
  a = Real.sqrt 5 + Real.sqrt 6 → 
  b = 1 / (Real.sqrt 6 - Real.sqrt 5) → 
  a = b := by sorry

end a_equals_b_l387_38787


namespace complex_number_equality_l387_38756

theorem complex_number_equality (a : ℝ) : 
  (Complex.re ((2 * Complex.I - a) / Complex.I) = Complex.im ((2 * Complex.I - a) / Complex.I)) → a = 2 := by
  sorry

end complex_number_equality_l387_38756


namespace periodic_even_function_theorem_l387_38774

def is_periodic (f : ℝ → ℝ) (p : ℝ) : Prop :=
  ∀ x, f (x + p) = f x

def is_even (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = f x

theorem periodic_even_function_theorem (f : ℝ → ℝ) 
  (h_periodic : is_periodic f 2)
  (h_even : is_even f)
  (h_defined : ∀ x ∈ Set.Icc 2 3, f x = x) :
  ∀ x ∈ Set.Icc (-2) 0, f x = 3 - |x + 1| := by
  sorry

end periodic_even_function_theorem_l387_38774


namespace unknown_bill_value_l387_38796

/-- Represents the contents of Ali's wallet -/
structure Wallet where
  five_dollar_bills : ℕ
  unknown_bill : ℕ
  total_amount : ℕ

/-- Theorem stating that given the conditions of Ali's wallet, the unknown bill is $10 -/
theorem unknown_bill_value (w : Wallet) 
  (h1 : w.five_dollar_bills = 7)
  (h2 : w.total_amount = 45) :
  w.unknown_bill = 10 := by
  sorry

#check unknown_bill_value

end unknown_bill_value_l387_38796


namespace exists_divisible_by_sum_of_digits_l387_38775

/-- Sum of digits of a three-digit number -/
def sumOfDigits (n : ℕ) : ℕ :=
  (n / 100) + ((n / 10) % 10) + (n % 10)

/-- Theorem: Among 18 consecutive three-digit numbers, there is at least one divisible by its sum of digits -/
theorem exists_divisible_by_sum_of_digits (n : ℕ) (h : 100 ≤ n ∧ n ≤ 982) :
  ∃ k : ℕ, n ≤ k ∧ k ≤ n + 17 ∧ k % sumOfDigits k = 0 :=
sorry

end exists_divisible_by_sum_of_digits_l387_38775


namespace max_value_of_f_l387_38782

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * Real.sin x * Real.cos x - Real.sin x ^ 2 + 1 / 2

theorem max_value_of_f (a : ℝ) :
  (∀ x : ℝ, f a x = f a (π / 3 - x)) →
  (∃ m : ℝ, ∀ x : ℝ, f a x ≤ m ∧ ∃ x₀ : ℝ, f a x₀ = m) →
  (∃ x₀ : ℝ, f a x₀ = 1) ∧ (∀ x : ℝ, f a x ≤ 1) :=
by sorry

end max_value_of_f_l387_38782


namespace fish_filets_count_l387_38733

/-- The number of fish filets Ben and his family will have after their fishing trip -/
def fish_filets : ℕ :=
  let ben_fish := 4
  let judy_fish := 1
  let billy_fish := 3
  let jim_fish := 2
  let susie_fish := 5
  let small_fish := 3
  let filets_per_fish := 2
  let total_caught := ben_fish + judy_fish + billy_fish + jim_fish + susie_fish
  let kept_fish := total_caught - small_fish
  kept_fish * filets_per_fish

/-- Theorem stating that the number of fish filets Ben and his family will have is 24 -/
theorem fish_filets_count : fish_filets = 24 := by
  sorry

end fish_filets_count_l387_38733


namespace circumscribed_polygon_similarity_l387_38786

/-- A circumscribed n-gon (n > 3) divided by non-intersecting diagonals into triangles -/
structure CircumscribedPolygon (n : ℕ) :=
  (n_gt_three : n > 3)
  (divided_into_triangles : Bool)
  (non_intersecting_diagonals : Bool)

/-- Predicate to check if all triangles are similar to at least one other triangle -/
def all_triangles_similar (p : CircumscribedPolygon n) : Prop := sorry

/-- The set of possible n values for which the described situation is possible -/
def possible_n_values : Set ℕ := {n | n = 4 ∨ n > 5}

/-- Theorem stating the possible values of n for which the described situation is possible -/
theorem circumscribed_polygon_similarity (n : ℕ) (p : CircumscribedPolygon n) :
  all_triangles_similar p ↔ n ∈ possible_n_values :=
sorry

end circumscribed_polygon_similarity_l387_38786


namespace coloring_theorem_l387_38723

/-- A coloring of natural numbers using k colors -/
def Coloring (k : ℕ) := ℕ → Fin k

/-- Proposition: For any coloring of natural numbers using k colors,
    there exist four distinct natural numbers a, b, c, d of the same color
    satisfying the required properties. -/
theorem coloring_theorem (k : ℕ) (coloring : Coloring k) :
  ∃ (a b c d : ℕ) (color : Fin k),
    a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d ∧
    coloring a = color ∧ coloring b = color ∧ coloring c = color ∧ coloring d = color ∧
    a * d = b * c ∧
    (∃ m : ℕ, b = a * 2^m) ∧
    (∃ n : ℕ, c = a * 3^n) :=
  sorry

end coloring_theorem_l387_38723


namespace chocolate_bar_cost_l387_38706

theorem chocolate_bar_cost (total_bars : ℕ) (unsold_bars : ℕ) (total_amount : ℚ) :
  total_bars = 8 →
  unsold_bars = 3 →
  total_amount = 20 →
  (total_bars - unsold_bars : ℚ) * (total_amount / (total_bars - unsold_bars : ℚ)) = 4 := by
  sorry

end chocolate_bar_cost_l387_38706


namespace masking_tape_for_room_l387_38757

/-- Calculates the amount of masking tape needed for a room with given dimensions --/
def masking_tape_needed (wall_width1 : ℝ) (wall_width2 : ℝ) (window_width : ℝ) (door_width : ℝ) : ℝ :=
  2 * (wall_width1 + wall_width2) - (2 * window_width + door_width)

/-- Theorem stating that the amount of masking tape needed for the given room is 15 meters --/
theorem masking_tape_for_room : masking_tape_needed 4 6 1.5 2 = 15 := by
  sorry

#check masking_tape_for_room

end masking_tape_for_room_l387_38757


namespace number_ratio_l387_38760

theorem number_ratio (x y z : ℝ) (k : ℝ) : 
  y = 2 * x →
  z = k * y →
  (x + y + z) / 3 = 165 →
  x = 45 →
  z / y = 4 := by
sorry

end number_ratio_l387_38760


namespace coefficient_of_x_l387_38749

theorem coefficient_of_x (x : ℝ) : 
  let expansion := (1 + x) * (x - 2/x)^3
  ∃ (a b c d e : ℝ), expansion = a*x^4 + b*x^3 + c*x^2 + (-6)*x + e
  := by sorry

end coefficient_of_x_l387_38749


namespace vector_sum_theorem_l387_38721

def vector_a : ℝ × ℝ × ℝ := (2, -3, 4)
def vector_b : ℝ × ℝ × ℝ := (-5, 1, 6)
def vector_c : ℝ × ℝ × ℝ := (3, 0, -2)

theorem vector_sum_theorem :
  vector_a.1 + vector_b.1 + vector_c.1 = 0 ∧
  vector_a.2.1 + vector_b.2.1 + vector_c.2.1 = -2 ∧
  vector_a.2.2 + vector_b.2.2 + vector_c.2.2 = 8 :=
by sorry

end vector_sum_theorem_l387_38721


namespace heathers_oranges_l387_38705

theorem heathers_oranges (initial remaining taken : ℕ) : 
  remaining = initial - taken → 
  taken = 35 → 
  remaining = 25 → 
  initial = 60 := by sorry

end heathers_oranges_l387_38705


namespace probability_two_red_two_blue_eq_l387_38755

def total_marbles : ℕ := 27
def red_marbles : ℕ := 15
def blue_marbles : ℕ := 12
def marbles_selected : ℕ := 4

def probability_two_red_two_blue : ℚ :=
  6 * (red_marbles.choose 2 * blue_marbles.choose 2) / total_marbles.choose marbles_selected

theorem probability_two_red_two_blue_eq :
  probability_two_red_two_blue = 154 / 225 := by
  sorry

end probability_two_red_two_blue_eq_l387_38755


namespace sally_buttons_theorem_l387_38779

/-- The number of buttons needed for Sally's shirts -/
def buttons_needed (monday_shirts tuesday_shirts wednesday_shirts buttons_per_shirt : ℕ) : ℕ :=
  (monday_shirts + tuesday_shirts + wednesday_shirts) * buttons_per_shirt

/-- Theorem: Sally needs 45 buttons for all her shirts -/
theorem sally_buttons_theorem :
  buttons_needed 4 3 2 5 = 45 := by
  sorry

end sally_buttons_theorem_l387_38779


namespace complement_of_union_l387_38719

def U : Set Nat := {0, 1, 2, 3, 4}
def A : Set Nat := {0, 1, 3}
def B : Set Nat := {2, 3}

theorem complement_of_union (U A B : Set Nat) 
  (hU : U = {0, 1, 2, 3, 4})
  (hA : A = {0, 1, 3})
  (hB : B = {2, 3}) :
  (U \ (A ∪ B)) = {4} := by
  sorry

end complement_of_union_l387_38719


namespace perpendicular_planes_l387_38732

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the perpendicular relation between lines
variable (perp_line : Line → Line → Prop)

-- Define the perpendicular relation between a line and a plane
variable (perp_line_plane : Line → Plane → Prop)

-- Define the perpendicular relation between planes
variable (perp_plane : Plane → Plane → Prop)

-- Theorem statement
theorem perpendicular_planes 
  (a b : Line) 
  (α β : Plane) 
  (hab : a ≠ b) 
  (hαβ : α ≠ β) 
  (hab_perp : perp_line a b) 
  (haα_perp : perp_line_plane a α) 
  (hbβ_perp : perp_line_plane b β) : 
  perp_plane α β :=
sorry

end perpendicular_planes_l387_38732


namespace smallest_k_for_900_digit_sum_l387_38738

def digit_sum (n : ℕ) : ℕ := sorry

def repeated_7 (k : ℕ) : ℕ := (10^k - 1) / 9

theorem smallest_k_for_900_digit_sum : 
  ∀ k : ℕ, k > 0 → 
  (∀ j : ℕ, 0 < j ∧ j < k → digit_sum (9 * repeated_7 j) ≠ 900) ∧ 
  digit_sum (9 * repeated_7 k) = 900 → 
  k = 100 := by sorry

end smallest_k_for_900_digit_sum_l387_38738


namespace triangle_height_equals_twice_rectangle_width_l387_38729

/-- Given a rectangle with dimensions a and b, and an isosceles triangle with base a and height h',
    if they have the same area, then the height of the triangle is 2b. -/
theorem triangle_height_equals_twice_rectangle_width
  (a b h' : ℝ) 
  (ha : a > 0)
  (hb : b > 0)
  (hh' : h' > 0)
  (h_area_eq : (1/2) * a * h' = a * b) :
  h' = 2 * b :=
by sorry

end triangle_height_equals_twice_rectangle_width_l387_38729


namespace jane_calculation_l387_38791

theorem jane_calculation (x y z : ℝ) 
  (h1 : x - (y - z) = 15) 
  (h2 : x - y - z = 7) : 
  x - y = 11 := by sorry

end jane_calculation_l387_38791


namespace triplet_sum_not_two_l387_38792

theorem triplet_sum_not_two : ∃! (a b c : ℚ), 
  ((a, b, c) = (3/4, 1/2, 3/4) ∨ 
   (a, b, c) = (6/5, 1/5, 2/5) ∨ 
   (a, b, c) = (3/5, 7/10, 7/10) ∨ 
   (a, b, c) = (33/10, -8/5, 3/10) ∨ 
   (a, b, c) = (6/5, 1/5, 2/5)) ∧ 
  a + b + c ≠ 2 := by
  sorry

end triplet_sum_not_two_l387_38792


namespace remainder_987654_div_6_l387_38711

theorem remainder_987654_div_6 : 987654 % 6 = 0 := by
  sorry

end remainder_987654_div_6_l387_38711


namespace trig_expression_equality_l387_38785

theorem trig_expression_equality : 
  (1 - Real.cos (10 * π / 180)^2) / 
  (Real.cos (800 * π / 180) * Real.sqrt (1 - Real.cos (20 * π / 180))) = 
  Real.sqrt 2 / 2 := by sorry

end trig_expression_equality_l387_38785


namespace candy_cost_l387_38747

theorem candy_cost (tickets_game1 tickets_game2 candies : ℕ) 
  (h1 : tickets_game1 = 33)
  (h2 : tickets_game2 = 9)
  (h3 : candies = 7) :
  (tickets_game1 + tickets_game2) / candies = 6 := by
  sorry

end candy_cost_l387_38747


namespace chord_inclination_range_l387_38765

/-- The range of inclination angles for a chord through the focus of a parabola -/
theorem chord_inclination_range (x y : ℝ) (α : ℝ) : 
  (y^2 = 4*x) →                             -- Parabola equation
  (3*x^2 + 2*y^2 = 2) →                     -- Ellipse equation
  (∃ (x₁ x₂ : ℝ), x₁ ≠ x₂ ∧ 
    y = (x - 1)*Real.tan α ∧               -- Chord passes through focus (1, 0)
    y^2 = 4*x ∧                            -- Chord intersects parabola
    (x₂ - x₁)^2 + ((x₂ - 1)*Real.tan α - (x₁ - 1)*Real.tan α)^2 ≤ 64) → -- Chord length ≤ 8
  (α ∈ Set.Icc (Real.pi/4) (Real.pi/3) ∪ Set.Icc (2*Real.pi/3) (3*Real.pi/4)) :=
by sorry

end chord_inclination_range_l387_38765


namespace constant_function_proof_l387_38708

theorem constant_function_proof (f : ℝ → ℝ) 
  (h_continuous : Continuous f) 
  (h_condition : ∀ (x : ℝ) (t : ℝ), t ≥ 0 → f x = f (Real.exp t * x)) : 
  ∃ (c : ℝ), ∀ (x : ℝ), f x = c := by
  sorry

end constant_function_proof_l387_38708


namespace polynomial_coefficients_l387_38789

theorem polynomial_coefficients 
  (a₁ a₂ a₃ a₄ : ℝ) 
  (h : ∀ x, (x - 1)^3 + (x + 1)^4 = x^4 + a₁*x^3 + a₂*x^2 + a₃*x + a₄) : 
  a₁ = 5 ∧ a₂ + a₃ + a₄ = 10 := by
sorry

end polynomial_coefficients_l387_38789


namespace alloy_composition_l387_38717

/-- Proves that the amount of the first alloy used is 15 kg given the specified conditions -/
theorem alloy_composition (x : ℝ) : 
  (0.12 * x + 0.10 * 35 = 0.106 * (x + 35)) → x = 15 :=
by sorry

end alloy_composition_l387_38717


namespace f_is_mapping_from_A_to_B_l387_38776

def A : Set ℕ := {0, 1, 2, 4}
def B : Set ℚ := {1/2, 0, 1, 2, 6, 8}

def f (x : ℕ) : ℚ := 2^(x - 1)

theorem f_is_mapping_from_A_to_B : ∀ x ∈ A, f x ∈ B := by
  sorry

end f_is_mapping_from_A_to_B_l387_38776


namespace unique_solution_l387_38740

/-- Represents the number of vehicles of each type Jeff has -/
structure VehicleCounts where
  trucks : ℕ
  cars : ℕ
  motorcycles : ℕ
  buses : ℕ

/-- Checks if the given vehicle counts satisfy all the conditions -/
def satisfiesConditions (v : VehicleCounts) : Prop :=
  v.cars = 2 * v.trucks ∧
  v.motorcycles = 3 * v.cars ∧
  v.buses = v.trucks / 2 ∧
  v.trucks + v.cars + v.motorcycles + v.buses = 180

/-- The theorem stating that the given vehicle counts are the unique solution -/
theorem unique_solution : 
  ∃! v : VehicleCounts, satisfiesConditions v ∧ 
    v.trucks = 19 ∧ v.cars = 38 ∧ v.motorcycles = 114 ∧ v.buses = 9 :=
by sorry

end unique_solution_l387_38740


namespace rachels_homework_difference_l387_38795

/-- Rachel's homework problem -/
theorem rachels_homework_difference (math_pages reading_pages : ℕ) 
  (h1 : math_pages = 3) 
  (h2 : reading_pages = 4) : 
  reading_pages - math_pages = 1 := by
  sorry

end rachels_homework_difference_l387_38795


namespace root_sum_inverse_squares_l387_38752

theorem root_sum_inverse_squares (a b c : ℝ) : 
  a^3 - 12*a^2 + 20*a - 3 = 0 →
  b^3 - 12*b^2 + 20*b - 3 = 0 →
  c^3 - 12*c^2 + 20*c - 3 = 0 →
  a ≠ b → b ≠ c → a ≠ c →
  1/a^2 + 1/b^2 + 1/c^2 = 328/9 := by
  sorry

end root_sum_inverse_squares_l387_38752


namespace nesbitt_inequality_l387_38793

theorem nesbitt_inequality (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  a / (b + c) + b / (c + a) + c / (a + b) ≥ 3 / 2 := by
  sorry

end nesbitt_inequality_l387_38793


namespace monkey_bird_problem_l387_38742

theorem monkey_bird_problem (initial_monkeys initial_birds : ℕ) 
  (eating_monkeys : ℕ) (percentage_monkeys : ℚ) :
  initial_monkeys = 6 →
  initial_birds = 6 →
  eating_monkeys = 2 →
  percentage_monkeys = 60 / 100 →
  ∃ (birds_eaten : ℕ),
    birds_eaten * eating_monkeys = initial_birds - (initial_monkeys / percentage_monkeys - initial_monkeys) ∧
    birds_eaten = 1 :=
by sorry

end monkey_bird_problem_l387_38742


namespace saree_price_l387_38781

theorem saree_price (final_price : ℝ) (discount1 : ℝ) (discount2 : ℝ) : 
  final_price = 144 ∧ discount1 = 0.20 ∧ discount2 = 0.10 →
  ∃ original_price : ℝ, 
    original_price = 200 ∧ 
    final_price = original_price * (1 - discount1) * (1 - discount2) :=
by
  sorry

end saree_price_l387_38781


namespace square_side_length_l387_38754

theorem square_side_length (x : ℝ) (h : x > 0) : 4 * x = 2 * (x ^ 2) → x = 2 := by
  sorry

end square_side_length_l387_38754


namespace truck_meeting_distance_difference_l387_38739

theorem truck_meeting_distance_difference 
  (initial_distance : ℝ) 
  (speed_a : ℝ) 
  (speed_b : ℝ) 
  (head_start : ℝ) :
  initial_distance = 855 →
  speed_a = 90 →
  speed_b = 80 →
  head_start = 1 →
  let relative_speed := speed_a + speed_b
  let meeting_time := (initial_distance - speed_a * head_start) / relative_speed
  let distance_a := speed_a * (meeting_time + head_start)
  let distance_b := speed_b * meeting_time
  distance_a - distance_b = 135 := by sorry

end truck_meeting_distance_difference_l387_38739


namespace unequal_gender_probability_l387_38764

theorem unequal_gender_probability :
  let n : ℕ := 8  -- number of children
  let p : ℚ := 1/2  -- probability of each gender
  let total_outcomes : ℕ := 2^n
  let equal_outcomes : ℕ := n.choose (n/2)
  let unequal_outcomes : ℕ := total_outcomes - equal_outcomes
  (unequal_outcomes : ℚ) / total_outcomes = 93/128 := by
sorry

end unequal_gender_probability_l387_38764


namespace overlapping_squares_area_l387_38718

theorem overlapping_squares_area (side_length : ℝ) (rotation_angle : ℝ) : 
  side_length = 12 →
  rotation_angle = 30 * π / 180 →
  ∃ (common_area : ℝ), common_area = 48 * Real.sqrt 3 ∧
    common_area = 2 * (1/2 * side_length * (side_length / Real.sqrt 3)) :=
by sorry

end overlapping_squares_area_l387_38718


namespace intersection_count_7_intersection_count_21_l387_38770

-- Define the line equation
def line_equation (k : ℝ) (x y : ℝ) : Prop := k * x + y + k^3 = 0

-- Define the set of k values for the first case
def k_values_7 : Set ℝ := {0, 0.3, -0.3, 0.6, -0.6, 0.9, -0.9}

-- Define the set of k values for the second case
def k_values_21 : Set ℝ := {x : ℝ | ∃ n : ℤ, -10 ≤ n ∧ n ≤ 10 ∧ x = n / 10}

-- Define the function to count intersection points
noncomputable def count_intersections (k_values : Set ℝ) : ℕ := sorry

-- Theorem for the first case
theorem intersection_count_7 : 
  count_intersections k_values_7 = 11 := by sorry

-- Theorem for the second case
theorem intersection_count_21 :
  count_intersections k_values_21 = 110 := by sorry

end intersection_count_7_intersection_count_21_l387_38770


namespace power_of_power_l387_38704

theorem power_of_power : (2^3)^3 = 512 := by
  sorry

end power_of_power_l387_38704


namespace visitors_count_l387_38798

/-- Represents the cost per person based on the number of visitors -/
def cost_per_person (n : ℕ) : ℚ :=
  if n ≤ 30 then 100
  else max 72 (100 - 2 * (n - 30))

/-- The total cost for n visitors -/
def total_cost (n : ℕ) : ℚ := n * cost_per_person n

/-- Theorem stating that 35 is the number of visitors given the conditions -/
theorem visitors_count : ∃ (n : ℕ), n > 30 ∧ total_cost n = 3150 ∧ n = 35 := by
  sorry


end visitors_count_l387_38798


namespace rational_expression_evaluation_l387_38797

theorem rational_expression_evaluation : 
  let x : ℝ := 8
  (x^4 - 18*x^2 + 81) / (x^2 - 9) = 55 := by
  sorry

end rational_expression_evaluation_l387_38797


namespace certain_number_value_l387_38709

-- Define the operation #
def hash (a b : ℝ) : ℝ := a * b - b + b^2

-- Theorem statement
theorem certain_number_value :
  ∀ x : ℝ, hash x 6 = 48 → x = 3 := by
  sorry

end certain_number_value_l387_38709


namespace largest_element_l387_38735

def S (a : ℝ) : Set ℝ := {-3*a, 2*a, 18/a, a^2, 1}

theorem largest_element (a : ℝ) (h : a = 3) : ∀ x ∈ S a, x ≤ a^2 := by
  sorry

end largest_element_l387_38735


namespace a_3_value_l387_38745

/-- Given a sequence {a_n} where S_n is the sum of the first n terms -/
def S (n : ℕ) : ℚ := (n + 1 : ℚ) / (n + 2 : ℚ)

/-- Definition of a_n in terms of S_n -/
def a (n : ℕ) : ℚ :=
  if n = 1 then S 1
  else S n - S (n - 1)

theorem a_3_value : a 3 = 1 / 20 := by sorry

end a_3_value_l387_38745


namespace decimal_to_fraction_l387_38799

theorem decimal_to_fraction : (2.35 : ℚ) = 47 / 20 := by
  sorry

end decimal_to_fraction_l387_38799


namespace construct_segment_a_construct_segment_b_l387_38761

-- Part a
theorem construct_segment_a (a : ℝ) (h : a = Real.sqrt 5) : ∃ b : ℝ, b = 1 := by
  sorry

-- Part b
theorem construct_segment_b (a : ℝ) (h : a = 7) : ∃ b : ℝ, b = Real.sqrt 7 := by
  sorry

end construct_segment_a_construct_segment_b_l387_38761


namespace two_number_difference_l387_38769

theorem two_number_difference (x y : ℝ) : 
  x + y = 40 → 3 * y - 4 * x = 20 → |y - x| = 11.42 := by
  sorry

end two_number_difference_l387_38769


namespace linear_function_and_inequality_l387_38773

-- Define the linear function f
def f : ℝ → ℝ := fun x ↦ x + 2

-- Define the function g
def g (a : ℝ) : ℝ → ℝ := fun x ↦ (1 - a) * x^2 - x

theorem linear_function_and_inequality (a : ℝ) :
  (∀ x, f (f x) = x + 4) →
  (∀ x₁ ∈ Set.Icc (1/4 : ℝ) 4, ∃ x₂ ∈ Set.Icc (-3 : ℝ) (1/3 : ℝ), g a x₁ ≥ f x₂) →
  (∀ x, f x = x + 2) ∧ a ∈ Set.Iic (3/4 : ℝ) :=
by sorry

end linear_function_and_inequality_l387_38773


namespace max_child_age_fraction_is_five_eighths_l387_38758

/-- The maximum fraction of Jane's age that a child she babysat could be -/
def max_child_age_fraction : ℚ :=
  let jane_current_age : ℕ := 34
  let years_since_stopped : ℕ := 10
  let jane_age_when_stopped : ℕ := jane_current_age - years_since_stopped
  let oldest_child_current_age : ℕ := 25
  let oldest_child_age_when_jane_stopped : ℕ := oldest_child_current_age - years_since_stopped
  (oldest_child_age_when_jane_stopped : ℚ) / jane_age_when_stopped

/-- Theorem stating that the maximum fraction of Jane's age that a child she babysat could be is 5/8 -/
theorem max_child_age_fraction_is_five_eighths :
  max_child_age_fraction = 5 / 8 := by
  sorry

end max_child_age_fraction_is_five_eighths_l387_38758


namespace two_digit_sum_theorem_l387_38707

def is_valid_set (a b c : ℕ) : Prop :=
  a ≠ b ∧ b ≠ c ∧ a ≠ c ∧
  a < 10 ∧ b < 10 ∧ c < 10 ∧
  (10 * a + b) + (10 * a + c) + (10 * b + a) + (10 * b + c) + (10 * c + a) + (10 * c + b) = 484

def valid_sets : List (Fin 10 × Fin 10 × Fin 10) :=
  [(9, 4, 9), (9, 5, 8), (9, 6, 7), (8, 6, 8), (8, 7, 7)]

theorem two_digit_sum_theorem (a b c : ℕ) :
  is_valid_set a b c →
  (a, b, c) ∈ valid_sets.map (fun (x, y, z) => (x.val, y.val, z.val)) :=
by
  sorry

end two_digit_sum_theorem_l387_38707


namespace exactly_one_positive_integer_solution_l387_38772

theorem exactly_one_positive_integer_solution : 
  ∃! (n : ℕ), n > 0 ∧ 25 - 5 * n > 15 :=
sorry

end exactly_one_positive_integer_solution_l387_38772


namespace central_academy_olympiad_l387_38710

theorem central_academy_olympiad (j s : ℕ) (hj : j > 0) (hs : s > 0) : 
  (3 * j : ℚ) / 7 = (6 * s : ℚ) / 7 → j = 2 * s := by
  sorry

end central_academy_olympiad_l387_38710


namespace actual_weight_loss_percentage_l387_38768

-- Define the weight loss challenge scenario
def weight_loss_challenge (W : ℝ) (actual_loss_percent : ℝ) (clothes_add_percent : ℝ) (measured_loss_percent : ℝ) : Prop :=
  let final_weight := W * (1 - actual_loss_percent / 100 + clothes_add_percent / 100)
  final_weight = W * (1 - measured_loss_percent / 100)

-- Theorem statement
theorem actual_weight_loss_percentage 
  (W : ℝ) (actual_loss_percent : ℝ) (clothes_add_percent : ℝ) (measured_loss_percent : ℝ)
  (h1 : W > 0)
  (h2 : clothes_add_percent = 2)
  (h3 : measured_loss_percent = 8.2)
  (h4 : weight_loss_challenge W actual_loss_percent clothes_add_percent measured_loss_percent) :
  actual_loss_percent = 10.2 := by
sorry


end actual_weight_loss_percentage_l387_38768


namespace water_to_pool_volume_l387_38701

/-- Proves that one gallon of water fills 1 cubic foot of Jerry's pool --/
theorem water_to_pool_volume 
  (total_water : ℝ) 
  (drinking_cooking : ℝ) 
  (shower_water : ℝ) 
  (pool_length pool_width pool_height : ℝ) 
  (num_showers : ℕ) 
  (h1 : total_water = 1000) 
  (h2 : drinking_cooking = 100) 
  (h3 : shower_water = 20) 
  (h4 : pool_length = 10 ∧ pool_width = 10 ∧ pool_height = 6) 
  (h5 : num_showers = 15) : 
  (total_water - drinking_cooking - num_showers * shower_water) / (pool_length * pool_width * pool_height) = 1 := by
  sorry

end water_to_pool_volume_l387_38701


namespace complex_square_roots_l387_38784

theorem complex_square_roots (z : ℂ) : 
  z ^ 2 = -91 - 49 * I ↔ z = (7 * Real.sqrt 2) / 2 - 7 * Real.sqrt 2 * I ∨ 
                         z = -(7 * Real.sqrt 2) / 2 + 7 * Real.sqrt 2 * I := by
  sorry

end complex_square_roots_l387_38784


namespace triangle_theorem_l387_38737

/-- Triangle ABC with side lengths a, b, c and angles A, B, C -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- The main theorem about the triangle -/
theorem triangle_theorem (t : Triangle) 
  (h1 : Real.sqrt 2 * t.b * t.c = t.b^2 + t.c^2 - t.a^2) :
  t.A = π / 4 ∧ 
  (t.a = 2 * Real.sqrt 2 ∧ t.B = π / 3 → t.b = 2 * Real.sqrt 3) := by
  sorry

end triangle_theorem_l387_38737


namespace hat_price_calculation_l387_38743

theorem hat_price_calculation (total_hats green_hats : ℕ) (blue_price green_price : ℚ) 
  (h1 : total_hats = 85)
  (h2 : green_hats = 38)
  (h3 : blue_price = 6)
  (h4 : green_price = 7) :
  let blue_hats := total_hats - green_hats
  (blue_hats * blue_price + green_hats * green_price : ℚ) = 548 := by
  sorry

end hat_price_calculation_l387_38743


namespace min_value_of_exponential_sum_l387_38730

theorem min_value_of_exponential_sum (x y : ℝ) (h : x + 2 * y = 3) :
  ∃ (min : ℝ), min = 4 * Real.sqrt 2 ∧ ∀ (a b : ℝ), a + 2 * b = 3 → 2^a + 4^b ≥ min :=
sorry

end min_value_of_exponential_sum_l387_38730


namespace ratio_a_to_c_l387_38766

theorem ratio_a_to_c (a b c d : ℚ) 
  (hab : a / b = 5 / 4)
  (hcd : c / d = 4 / 1)
  (hdb : d / b = 1 / 5) :
  a / c = 25 / 16 := by
  sorry

end ratio_a_to_c_l387_38766


namespace range_of_m_given_p_q_l387_38767

/-- The range of m given the conditions of p and q -/
theorem range_of_m_given_p_q :
  ∀ (m : ℝ),
  (∀ x : ℝ, x^2 - 8*x - 20 > 0 → (x - (1 - m)) * (x - (1 + m)) > 0) ∧
  (∃ x : ℝ, (x - (1 - m)) * (x - (1 + m)) > 0 ∧ x^2 - 8*x - 20 ≤ 0) ∧
  m > 0 →
  0 < m ∧ m ≤ 3 :=
by sorry

end range_of_m_given_p_q_l387_38767


namespace probability_three_common_books_l387_38762

theorem probability_three_common_books (total_books : ℕ) (books_to_select : ℕ) (common_books : ℕ) :
  total_books = 12 →
  books_to_select = 7 →
  common_books = 3 →
  (Nat.choose total_books common_books * Nat.choose (total_books - common_books) (books_to_select - common_books) * Nat.choose (total_books - common_books) (books_to_select - common_books)) /
  (Nat.choose total_books books_to_select * Nat.choose total_books books_to_select) =
  3502800 / 627264 :=
by sorry

end probability_three_common_books_l387_38762


namespace expressions_equality_l387_38712

/-- 
Theorem: The expressions 2a+3bc and (a+b)(2a+c) are equal if and only if a+b+c = 2.
-/
theorem expressions_equality (a b c : ℝ) : 2*a + 3*b*c = (a+b)*(2*a+c) ↔ a + b + c = 2 := by
  sorry

end expressions_equality_l387_38712


namespace driver_speed_driver_speed_proof_l387_38753

/-- The actual average speed of a driver, given that increasing the speed by 12 miles per hour
would have reduced the travel time by 1/3. -/
theorem driver_speed : ℝ → Prop :=
  fun v : ℝ =>
    ∀ t d : ℝ,
      t > 0 → d > 0 →
      d = v * t →
      d = (v + 12) * (2/3 * t) →
      v = 24

-- The proof is omitted
theorem driver_speed_proof : driver_speed 24 := by sorry

end driver_speed_driver_speed_proof_l387_38753


namespace lego_problem_solution_l387_38748

def lego_problem (initial_pieces : ℕ) : ℕ :=
  let castle_pieces := initial_pieces / 4
  let after_castle := initial_pieces - castle_pieces
  let spaceship_pieces := (after_castle * 2) / 5
  let after_spaceship := after_castle - spaceship_pieces
  let lost_after_building := (after_spaceship * 15) / 100
  let after_loss := after_spaceship - lost_after_building
  let town_pieces := after_loss / 2
  let after_town := after_loss - town_pieces
  let final_loss := (after_town * 10) / 100
  after_town - final_loss

theorem lego_problem_solution :
  lego_problem 500 = 85 := by sorry

end lego_problem_solution_l387_38748
