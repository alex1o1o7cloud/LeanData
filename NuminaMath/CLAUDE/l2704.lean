import Mathlib

namespace NUMINAMATH_CALUDE_cos_one_sufficient_not_necessary_l2704_270423

theorem cos_one_sufficient_not_necessary (x : ℝ) : 
  (∀ x, Real.cos x = 1 → Real.sin x = 0) ∧ 
  (∃ x, Real.sin x = 0 ∧ Real.cos x ≠ 1) := by
  sorry

end NUMINAMATH_CALUDE_cos_one_sufficient_not_necessary_l2704_270423


namespace NUMINAMATH_CALUDE_encyclopedia_interest_percentage_l2704_270426

/-- Calculates the interest paid as a percentage of the amount borrowed for a set of encyclopedias. -/
theorem encyclopedia_interest_percentage (cost : ℝ) (down_payment : ℝ) (monthly_payment : ℝ) (num_months : ℕ) (final_payment : ℝ) :
  cost = 1200 →
  down_payment = 500 →
  monthly_payment = 70 →
  num_months = 12 →
  final_payment = 45 →
  let total_paid := down_payment + (monthly_payment * num_months) + final_payment
  let amount_borrowed := cost - down_payment
  let interest_paid := total_paid - cost
  let interest_percentage := (interest_paid / amount_borrowed) * 100
  ∃ ε > 0, |interest_percentage - 26.43| < ε :=
by sorry

end NUMINAMATH_CALUDE_encyclopedia_interest_percentage_l2704_270426


namespace NUMINAMATH_CALUDE_largest_root_of_cubic_l2704_270489

theorem largest_root_of_cubic (p q r : ℝ) : 
  p + q + r = 3 → 
  p * q + p * r + q * r = -6 → 
  p * q * r = -8 → 
  ∃ (largest : ℝ), largest = (1 + Real.sqrt 17) / 2 ∧ 
    largest ≥ p ∧ largest ≥ q ∧ largest ≥ r ∧
    largest^3 - 3 * largest^2 - 6 * largest + 8 = 0 := by
  sorry

end NUMINAMATH_CALUDE_largest_root_of_cubic_l2704_270489


namespace NUMINAMATH_CALUDE_onions_remaining_l2704_270487

theorem onions_remaining (initial : Nat) (sold : Nat) (h1 : initial = 98) (h2 : sold = 65) :
  initial - sold = 33 := by
  sorry

end NUMINAMATH_CALUDE_onions_remaining_l2704_270487


namespace NUMINAMATH_CALUDE_gcd_of_72_120_168_l2704_270499

theorem gcd_of_72_120_168 : Nat.gcd 72 (Nat.gcd 120 168) = 24 := by
  sorry

end NUMINAMATH_CALUDE_gcd_of_72_120_168_l2704_270499


namespace NUMINAMATH_CALUDE_ellipse_line_intersection_slope_l2704_270493

/-- Theorem: For an ellipse and a line intersecting it under specific conditions, the slope of the line is ±1/2. -/
theorem ellipse_line_intersection_slope (k : ℝ) : 
  (∀ x y, x^2/4 + y^2/3 = 1 → y = k*x + 1 → 
    ∃ x₁ x₂ y₁ y₂, 
      x₁^2/4 + y₁^2/3 = 1 ∧ 
      x₂^2/4 + y₂^2/3 = 1 ∧
      y₁ = k*x₁ + 1 ∧ 
      y₂ = k*x₂ + 1 ∧ 
      x₁ = -x₂/2) →
  k = 1/2 ∨ k = -1/2 := by
sorry

end NUMINAMATH_CALUDE_ellipse_line_intersection_slope_l2704_270493


namespace NUMINAMATH_CALUDE_isosceles_when_negative_one_is_root_roots_of_equilateral_triangle_l2704_270484

/-- Represents a triangle with side lengths a, b, and c -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ

/-- The quadratic equation associated with the triangle -/
def quadratic (t : Triangle) (x : ℝ) : ℝ :=
  (t.a + t.c) * x^2 + 2 * t.b * x + (t.b - t.c)

theorem isosceles_when_negative_one_is_root (t : Triangle) :
  quadratic t (-1) = 0 → t.a = t.b :=
sorry

theorem roots_of_equilateral_triangle (t : Triangle) :
  t.a = t.b ∧ t.b = t.c →
  (quadratic t 0 = 0 ∧ quadratic t (-1) = 0) :=
sorry

end NUMINAMATH_CALUDE_isosceles_when_negative_one_is_root_roots_of_equilateral_triangle_l2704_270484


namespace NUMINAMATH_CALUDE_residue_of_11_pow_2010_mod_19_l2704_270453

theorem residue_of_11_pow_2010_mod_19 : (11 : ℤ) ^ 2010 ≡ 3 [ZMOD 19] := by
  sorry

end NUMINAMATH_CALUDE_residue_of_11_pow_2010_mod_19_l2704_270453


namespace NUMINAMATH_CALUDE_necklace_beads_l2704_270442

theorem necklace_beads (total : ℕ) (blue : ℕ) (h1 : total = 40) (h2 : blue = 5) : 
  let red := 2 * blue
  let white := blue + red
  let silver := total - (blue + red + white)
  silver = 10 := by
  sorry

end NUMINAMATH_CALUDE_necklace_beads_l2704_270442


namespace NUMINAMATH_CALUDE_rod_length_proof_l2704_270463

/-- Given that a 6-meter rod weighs 14.04 kg, prove that a rod weighing 23.4 kg is 10 meters long. -/
theorem rod_length_proof (weight_per_meter : ℝ) (h1 : weight_per_meter = 14.04 / 6) :
  23.4 / weight_per_meter = 10 := by
sorry

end NUMINAMATH_CALUDE_rod_length_proof_l2704_270463


namespace NUMINAMATH_CALUDE_winning_percentage_approx_l2704_270469

def total_votes : ℕ := 2500 + 5000 + 15000

def winning_votes : ℕ := 15000

def winning_percentage : ℚ := (winning_votes : ℚ) / (total_votes : ℚ) * 100

theorem winning_percentage_approx : 
  ∃ (ε : ℚ), ε > 0 ∧ ε < 0.01 ∧ |winning_percentage - 200/3| < ε :=
sorry

end NUMINAMATH_CALUDE_winning_percentage_approx_l2704_270469


namespace NUMINAMATH_CALUDE_antifreeze_solution_l2704_270411

def antifreeze_problem (x : ℝ) : Prop :=
  let solution1_percent : ℝ := 10
  let total_volume : ℝ := 20
  let target_percent : ℝ := 15
  let volume_each : ℝ := 7.5
  (solution1_percent * volume_each + x * volume_each) / total_volume = target_percent

theorem antifreeze_solution : 
  ∃ x : ℝ, antifreeze_problem x ∧ x = 30 := by
sorry

end NUMINAMATH_CALUDE_antifreeze_solution_l2704_270411


namespace NUMINAMATH_CALUDE_rational_root_of_polynomial_l2704_270460

def p (x : ℚ) : ℚ := 3 * x^5 - 4 * x^3 - 7 * x^2 + 2 * x + 1

theorem rational_root_of_polynomial :
  (p (-1/3) = 0) ∧ (∀ q : ℚ, p q = 0 → q = -1/3) :=
by sorry

end NUMINAMATH_CALUDE_rational_root_of_polynomial_l2704_270460


namespace NUMINAMATH_CALUDE_divisible_by_48_l2704_270433

theorem divisible_by_48 (n : ℕ) (h : Even n) : ∃ k : ℤ, (n^3 : ℤ) + 20*n = 48*k := by
  sorry

end NUMINAMATH_CALUDE_divisible_by_48_l2704_270433


namespace NUMINAMATH_CALUDE_f_monotonic_and_odd_l2704_270450

def f (x : ℝ) : ℝ := x^3

theorem f_monotonic_and_odd : 
  (∀ x y, x < y → f x < f y) ∧ (∀ x, f (-x) = -f x) := by sorry

end NUMINAMATH_CALUDE_f_monotonic_and_odd_l2704_270450


namespace NUMINAMATH_CALUDE_investment_profit_comparison_l2704_270456

/-- Profit calculation for selling at the beginning of the month -/
def profit_beginning (x : ℝ) : ℝ := 0.265 * x

/-- Profit calculation for selling at the end of the month -/
def profit_end (x : ℝ) : ℝ := 0.3 * x - 700

theorem investment_profit_comparison :
  /- The investment amount where profits are equal is 20,000 yuan -/
  (∃ x : ℝ, x = 20000 ∧ profit_beginning x = profit_end x) ∧
  /- For a 50,000 yuan investment, profit from selling at the end is greater -/
  (profit_end 50000 > profit_beginning 50000) :=
by sorry

end NUMINAMATH_CALUDE_investment_profit_comparison_l2704_270456


namespace NUMINAMATH_CALUDE_valid_solution_l2704_270404

/-- A number is a perfect square if it's the square of an integer. -/
def IsPerfectSquare (x : ℕ) : Prop := ∃ y : ℕ, x = y^2

/-- A function to check if a number is prime. -/
def IsPrime (p : ℕ) : Prop := p > 1 ∧ ∀ d : ℕ, d ∣ p → d = 1 ∨ d = p

/-- The theorem stating that 900 is a valid solution for n. -/
theorem valid_solution :
  ∃ m : ℕ, IsPerfectSquare m ∧ IsPerfectSquare 900 ∧ IsPrime (m - 900) :=
by sorry

end NUMINAMATH_CALUDE_valid_solution_l2704_270404


namespace NUMINAMATH_CALUDE_problem_solution_l2704_270491

-- Define the sets A and B
def A : Set ℝ := {x | x^2 - 5*x + 6 = 0}
def B (a : ℝ) : Set ℝ := {a, 2, 2*a - 1}

-- State the theorem
theorem problem_solution :
  ∃ (a : ℝ), A ⊆ B a ∧ A = {2, 3} ∧ a = 3 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l2704_270491


namespace NUMINAMATH_CALUDE_cory_candy_packs_l2704_270445

/-- The number of packs of candies Cory wants to buy -/
def num_packs : ℕ := sorry

/-- The amount of money Cory has -/
def cory_money : ℚ := 20

/-- The cost of each pack of candies -/
def pack_cost : ℚ := 49

/-- The additional amount Cory needs -/
def additional_money : ℚ := 78

theorem cory_candy_packs : num_packs = 2 := by sorry

end NUMINAMATH_CALUDE_cory_candy_packs_l2704_270445


namespace NUMINAMATH_CALUDE_final_output_is_four_l2704_270408

def program_output (initial : ℕ) (increment1 : ℕ) (increment2 : ℕ) : ℕ :=
  initial + increment1 + increment2

theorem final_output_is_four :
  program_output 1 1 2 = 4 := by
  sorry

end NUMINAMATH_CALUDE_final_output_is_four_l2704_270408


namespace NUMINAMATH_CALUDE_supplement_of_complement_of_35_l2704_270466

def complement (α : ℝ) : ℝ := 90 - α

def supplement (α : ℝ) : ℝ := 180 - α

theorem supplement_of_complement_of_35 : 
  supplement (complement 35) = 125 := by sorry

end NUMINAMATH_CALUDE_supplement_of_complement_of_35_l2704_270466


namespace NUMINAMATH_CALUDE_reflection_sum_coordinates_l2704_270424

/-- Given a point C with coordinates (x, 8), when reflected over the y-axis to point D,
    the sum of all coordinate values of C and D is 16. -/
theorem reflection_sum_coordinates (x : ℝ) : 
  let C : ℝ × ℝ := (x, 8)
  let D : ℝ × ℝ := (-x, 8)
  x + 8 + (-x) + 8 = 16 := by sorry

end NUMINAMATH_CALUDE_reflection_sum_coordinates_l2704_270424


namespace NUMINAMATH_CALUDE_quadratic_equation_solution_l2704_270449

theorem quadratic_equation_solution : ∃ x₁ x₂ : ℝ, 
  (x₁ + 10)^2 = (4*x₁ + 6)*(x₁ + 8) ∧ 
  (x₂ + 10)^2 = (4*x₂ + 6)*(x₂ + 8) ∧ 
  (abs (x₁ - 2.131) < 0.001) ∧ 
  (abs (x₂ + 8.131) < 0.001) := by
sorry

end NUMINAMATH_CALUDE_quadratic_equation_solution_l2704_270449


namespace NUMINAMATH_CALUDE_stratified_sampling_middle_schools_l2704_270414

theorem stratified_sampling_middle_schools 
  (total_schools : ℕ) 
  (middle_schools : ℕ) 
  (sample_size : ℕ) 
  (h1 : total_schools = 700) 
  (h2 : middle_schools = 200) 
  (h3 : sample_size = 70) :
  (sample_size : ℚ) * (middle_schools : ℚ) / (total_schools : ℚ) = 20 := by
sorry

end NUMINAMATH_CALUDE_stratified_sampling_middle_schools_l2704_270414


namespace NUMINAMATH_CALUDE_paco_cookie_difference_l2704_270477

/-- Calculates the difference between eaten cookies and the sum of given away and bought cookies -/
def cookieDifference (initial bought eaten givenAway : ℕ) : ℤ :=
  (eaten : ℤ) - ((givenAway : ℤ) + (bought : ℤ))

/-- Theorem stating the cookie difference for Paco's scenario -/
theorem paco_cookie_difference :
  cookieDifference 25 3 5 4 = -2 := by
  sorry

end NUMINAMATH_CALUDE_paco_cookie_difference_l2704_270477


namespace NUMINAMATH_CALUDE_triangle_problem_l2704_270474

open Real

noncomputable def f (x : ℝ) := 2 * sin x * cos (x - π/3) - sqrt 3 / 2

theorem triangle_problem (A B C : ℝ) (a b c R : ℝ) :
  (0 < A ∧ A < π/2) →
  (0 < B ∧ B < π/2) →
  (0 < C ∧ C < π/2) →
  A + B + C = π →
  a * cos B - b * cos A = R →
  f A = 1 →
  a = 2 * R * sin A →
  b = 2 * R * sin B →
  c = 2 * R * sin C →
  (B = π/4 ∧ -1 < (R - c) / b ∧ (R - c) / b < 0) :=
by sorry

end NUMINAMATH_CALUDE_triangle_problem_l2704_270474


namespace NUMINAMATH_CALUDE_product_four_consecutive_integers_l2704_270481

theorem product_four_consecutive_integers (a : ℤ) : 
  a^2 = 1000 * 1001 * 1002 * 1003 + 1 → a = 1002001 := by
  sorry

end NUMINAMATH_CALUDE_product_four_consecutive_integers_l2704_270481


namespace NUMINAMATH_CALUDE_inequality_proof_l2704_270402

theorem inequality_proof (a b c : ℝ) (h1 : a > b) (h2 : b > c) : 
  1 / (b - c) > 1 / (a - c) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l2704_270402


namespace NUMINAMATH_CALUDE_opposite_solutions_value_of_m_l2704_270409

theorem opposite_solutions_value_of_m :
  ∀ (x y m : ℝ),
  (3 * x + 4 * y = 7) →
  (5 * x - 4 * y = m) →
  (x + y = 0) →
  m = -63 := by
sorry

end NUMINAMATH_CALUDE_opposite_solutions_value_of_m_l2704_270409


namespace NUMINAMATH_CALUDE_two_problems_without_conditional_statements_l2704_270497

/-- Represents a mathematical problem that may or may not require conditional statements in its algorithm. -/
inductive Problem
| CommonLogarithm
| SquarePerimeter
| MaximumOfThree
| PiecewiseFunction

/-- Determines if a problem requires conditional statements in its algorithm. -/
def requiresConditionalStatements (p : Problem) : Bool :=
  match p with
  | Problem.CommonLogarithm => false
  | Problem.SquarePerimeter => false
  | Problem.MaximumOfThree => true
  | Problem.PiecewiseFunction => true

/-- The list of all problems given in the question. -/
def allProblems : List Problem :=
  [Problem.CommonLogarithm, Problem.SquarePerimeter, Problem.MaximumOfThree, Problem.PiecewiseFunction]

/-- Theorem stating that the number of problems not requiring conditional statements is 2. -/
theorem two_problems_without_conditional_statements :
  (allProblems.filter (fun p => ¬requiresConditionalStatements p)).length = 2 := by
  sorry

end NUMINAMATH_CALUDE_two_problems_without_conditional_statements_l2704_270497


namespace NUMINAMATH_CALUDE_correct_num_classes_l2704_270492

/-- The number of classes in a single round-robin basketball tournament -/
def num_classes : ℕ := 10

/-- The total number of games played in the tournament -/
def total_games : ℕ := 45

/-- Theorem stating that the number of classes is correct given the total number of games played -/
theorem correct_num_classes : 
  (num_classes * (num_classes - 1)) / 2 = total_games :=
by sorry

end NUMINAMATH_CALUDE_correct_num_classes_l2704_270492


namespace NUMINAMATH_CALUDE_teaching_position_allocation_l2704_270410

theorem teaching_position_allocation :
  let total_positions : ℕ := 8
  let num_schools : ℕ := 3
  let min_positions_per_school : ℕ := 1
  let min_positions_school_a : ℕ := 2
  let remaining_positions : ℕ := total_positions - (min_positions_school_a + min_positions_per_school * (num_schools - 1))
  (remaining_positions.choose (num_schools - 1)) = 6 :=
by sorry

end NUMINAMATH_CALUDE_teaching_position_allocation_l2704_270410


namespace NUMINAMATH_CALUDE_rectangular_views_imply_prism_or_cylinder_l2704_270436

/-- A solid object in 3D space -/
structure Solid where
  -- Add necessary fields here
  mk :: -- Constructor

/-- Front view of a solid -/
def frontView (s : Solid) : Set (ℝ × ℝ) :=
  sorry

/-- Side view of a solid -/
def sideView (s : Solid) : Set (ℝ × ℝ) :=
  sorry

/-- Predicate to check if a set is a rectangle -/
def isRectangle (s : Set (ℝ × ℝ)) : Prop :=
  sorry

/-- Predicate to check if a solid is a rectangular prism -/
def isRectangularPrism (s : Solid) : Prop :=
  sorry

/-- Predicate to check if a solid is a cylinder -/
def isCylinder (s : Solid) : Prop :=
  sorry

/-- Theorem: If a solid has rectangular front and side views, it can be either a rectangular prism or a cylinder -/
theorem rectangular_views_imply_prism_or_cylinder (s : Solid) :
  isRectangle (frontView s) → isRectangle (sideView s) →
  isRectangularPrism s ∨ isCylinder s :=
by
  sorry

end NUMINAMATH_CALUDE_rectangular_views_imply_prism_or_cylinder_l2704_270436


namespace NUMINAMATH_CALUDE_brendas_weight_multiple_l2704_270425

theorem brendas_weight_multiple (brenda_weight mel_weight : ℕ) (multiple : ℚ) : 
  brenda_weight = 220 →
  mel_weight = 70 →
  brenda_weight = mel_weight * multiple + 10 →
  multiple = 3 := by
  sorry

end NUMINAMATH_CALUDE_brendas_weight_multiple_l2704_270425


namespace NUMINAMATH_CALUDE_intersection_A_B_union_complement_B_P_intersection_AB_complement_P_l2704_270464

-- Define the sets A, B, and P
def A : Set ℝ := {x | -4 ≤ x ∧ x < 2}
def B : Set ℝ := {x | -1 < x ∧ x ≤ 3}
def P : Set ℝ := {x | x ≤ 0 ∨ x ≥ 5/2}

-- Theorem for A ∩ B
theorem intersection_A_B : A ∩ B = {x : ℝ | -1 < x ∧ x < 2} := by sorry

-- Theorem for (ᶜB) ∪ P
theorem union_complement_B_P : (Bᶜ : Set ℝ) ∪ P = {x : ℝ | x ≤ 0 ∨ x ≥ 5/2} := by sorry

-- Theorem for (A ∩ B) ∩ (ᶜP)
theorem intersection_AB_complement_P : (A ∩ B) ∩ (Pᶜ : Set ℝ) = {x : ℝ | 0 < x ∧ x < 2} := by sorry

end NUMINAMATH_CALUDE_intersection_A_B_union_complement_B_P_intersection_AB_complement_P_l2704_270464


namespace NUMINAMATH_CALUDE_total_cost_of_pens_and_pencils_l2704_270485

/-- The cost of buying multiple items given their individual prices -/
theorem total_cost_of_pens_and_pencils (x y : ℝ) : 
  5 * x + 3 * y = 5 * x + 3 * y := by sorry

end NUMINAMATH_CALUDE_total_cost_of_pens_and_pencils_l2704_270485


namespace NUMINAMATH_CALUDE_metal_price_calculation_l2704_270470

/-- Given two metals mixed in a 3:1 ratio, prove the price of the first metal -/
theorem metal_price_calculation (price_second : ℚ) (price_alloy : ℚ) :
  price_second = 96 →
  price_alloy = 75 →
  ∃ (price_first : ℚ),
    price_first = 68 ∧
    (3 * price_first + 1 * price_second) / 4 = price_alloy :=
by sorry

end NUMINAMATH_CALUDE_metal_price_calculation_l2704_270470


namespace NUMINAMATH_CALUDE_complex_power_modulus_l2704_270441

theorem complex_power_modulus : 
  Complex.abs ((1 / 2 : ℂ) + (Complex.I * Real.sqrt 3 / 2)) ^ 12 = 1 := by sorry

end NUMINAMATH_CALUDE_complex_power_modulus_l2704_270441


namespace NUMINAMATH_CALUDE_reduced_rate_fraction_l2704_270448

def hours_in_week : ℕ := 7 * 24

def weekday_reduced_hours : ℕ := 5 * 12

def weekend_reduced_hours : ℕ := 2 * 24

def total_reduced_hours : ℕ := weekday_reduced_hours + weekend_reduced_hours

theorem reduced_rate_fraction :
  (total_reduced_hours : ℚ) / hours_in_week = 9 / 14 := by
  sorry

end NUMINAMATH_CALUDE_reduced_rate_fraction_l2704_270448


namespace NUMINAMATH_CALUDE_mikes_net_spending_l2704_270431

/-- The net amount Mike spent at the music store -/
def net_amount (trumpet_cost song_book_price : ℚ) : ℚ :=
  trumpet_cost - song_book_price

/-- Theorem stating the net amount Mike spent -/
theorem mikes_net_spending :
  let trumpet_cost : ℚ := 145.16
  let song_book_price : ℚ := 5.84
  net_amount trumpet_cost song_book_price = 139.32 := by
  sorry

end NUMINAMATH_CALUDE_mikes_net_spending_l2704_270431


namespace NUMINAMATH_CALUDE_two_lines_theorem_l2704_270415

/-- Two lines in a 2D plane -/
structure TwoLines where
  l₁ : ℝ → ℝ → Prop
  l₂ : ℝ → ℝ → ℝ → Prop

/-- The given lines -/
def given_lines : TwoLines where
  l₁ := fun x y ↦ 2 * x + y + 4 = 0
  l₂ := fun a x y ↦ a * x + 4 * y + 1 = 0

/-- Perpendicularity condition -/
def perpendicular (lines : TwoLines) (a : ℝ) : Prop :=
  ∃ x y, lines.l₁ x y ∧ lines.l₂ a x y

/-- Parallelism condition -/
def parallel (lines : TwoLines) (a : ℝ) : Prop :=
  ∀ x y, lines.l₁ x y ↔ ∃ k, lines.l₂ a (x + k) (y + k)

/-- Main theorem -/
theorem two_lines_theorem (lines : TwoLines) :
  (∃ a, perpendicular lines a → 
    ∃ x y, lines.l₁ x y ∧ lines.l₂ a x y ∧ x = -3/2 ∧ y = -1) ∧
  (∃ a, parallel lines a → 
    ∃ d, d = (3 * Real.sqrt 5) / 4 ∧ 
      ∀ x₁ y₁ x₂ y₂, lines.l₁ x₁ y₁ → lines.l₂ a x₂ y₂ → 
        ((x₂ - x₁)^2 + (y₂ - y₁)^2 : ℝ) ≥ d^2) :=
by sorry

end NUMINAMATH_CALUDE_two_lines_theorem_l2704_270415


namespace NUMINAMATH_CALUDE_arithmetic_progression_non_prime_existence_l2704_270429

theorem arithmetic_progression_non_prime_existence 
  (a d : ℕ+) : 
  ∃ K : ℕ+, ∀ n : ℕ, ∃ i : Fin K, ¬ Nat.Prime (a + (n + i) * d) :=
sorry

end NUMINAMATH_CALUDE_arithmetic_progression_non_prime_existence_l2704_270429


namespace NUMINAMATH_CALUDE_base_10_to_base_3_172_l2704_270475

/-- Converts a natural number to its base 3 representation as a list of digits -/
def toBase3 (n : ℕ) : List ℕ :=
  if n = 0 then [0]
  else
    let rec aux (m : ℕ) (acc : List ℕ) : List ℕ :=
      if m = 0 then acc
      else aux (m / 3) ((m % 3) :: acc)
    aux n []

theorem base_10_to_base_3_172 :
  toBase3 172 = [2, 0, 1, 0, 1] := by
  sorry

#eval toBase3 172  -- This line is for verification purposes

end NUMINAMATH_CALUDE_base_10_to_base_3_172_l2704_270475


namespace NUMINAMATH_CALUDE_polyhedron_edge_intersection_l2704_270417

/-- A polyhedron with a specified number of edges. -/
structure Polyhedron where
  edges : ℕ

/-- A plane that can intersect edges of a polyhedron. -/
structure IntersectingPlane where
  intersected_edges : ℕ

/-- Theorem about the maximum number of intersected edges for different types of polyhedra. -/
theorem polyhedron_edge_intersection
  (p : Polyhedron)
  (h : p.edges = 100) :
  ∃ (convex_max non_convex_max : ℕ),
    -- For a convex polyhedron, the maximum number of intersected edges is 66
    (∀ (plane : IntersectingPlane), plane.intersected_edges ≤ convex_max) ∧
    convex_max = 66 ∧
    -- For a non-convex polyhedron, there exists a configuration where 96 edges can be intersected
    (∃ (plane : IntersectingPlane), plane.intersected_edges = non_convex_max) ∧
    non_convex_max = 96 ∧
    -- For any polyhedron, it's impossible to intersect all 100 edges
    (∀ (plane : IntersectingPlane), plane.intersected_edges < p.edges) :=
by
  sorry

end NUMINAMATH_CALUDE_polyhedron_edge_intersection_l2704_270417


namespace NUMINAMATH_CALUDE_vanessa_missed_days_l2704_270421

theorem vanessa_missed_days (total : ℕ) (vanessa_mike : ℕ) (mike_sarah : ℕ)
  (h1 : total = 17)
  (h2 : vanessa_mike = 14)
  (h3 : mike_sarah = 12) :
  ∃ (vanessa mike sarah : ℕ),
    vanessa + mike + sarah = total ∧
    vanessa + mike = vanessa_mike ∧
    mike + sarah = mike_sarah ∧
    vanessa = 5 := by
  sorry

end NUMINAMATH_CALUDE_vanessa_missed_days_l2704_270421


namespace NUMINAMATH_CALUDE_monotonicity_and_inequality_min_m_value_l2704_270496

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := x + a / x

theorem monotonicity_and_inequality (a : ℝ) (h_a : a > 0) :
  (∀ x₁ x₂ : ℝ, 0 < x₁ ∧ x₁ < x₂ ∧ x₂ ≤ 3 → f a x₁ ≥ f a x₂) ∧
  (∀ x₁ x₂ : ℝ, 3 ≤ x₁ ∧ x₁ < x₂ → f a x₁ ≤ f a x₂) ↔
  a = 9 :=
sorry

theorem min_m_value :
  (∀ x : ℝ, x ≥ 1 ∧ x ≤ 4 → x + 9 / x - 6.25 ≤ 0) ∧
  ∀ m : ℝ, m < 6.25 →
    ∃ x : ℝ, x ≥ 1 ∧ x ≤ 4 ∧ x + 9 / x - m > 0 :=
sorry

end NUMINAMATH_CALUDE_monotonicity_and_inequality_min_m_value_l2704_270496


namespace NUMINAMATH_CALUDE_eighth_term_of_geometric_sequence_l2704_270476

-- Define a geometric sequence
def geometric_sequence (a : ℝ) (r : ℝ) (n : ℕ) : ℝ := a * r^(n - 1)

-- Theorem statement
theorem eighth_term_of_geometric_sequence :
  let a := 4
  let a2 := 16
  let r := a2 / a
  geometric_sequence a r 8 = 65536 := by
sorry

end NUMINAMATH_CALUDE_eighth_term_of_geometric_sequence_l2704_270476


namespace NUMINAMATH_CALUDE_function_identity_l2704_270480

theorem function_identity (f : ℕ → ℕ) (h : ∀ n, f (n + 1) > f (f n)) : ∀ n, f n = n := by
  sorry

end NUMINAMATH_CALUDE_function_identity_l2704_270480


namespace NUMINAMATH_CALUDE_dot_product_equals_negative_102_l2704_270457

def vector1 : Fin 4 → ℤ := ![4, -5, 6, -3]
def vector2 : Fin 4 → ℤ := ![-2, 8, -7, 4]

theorem dot_product_equals_negative_102 :
  (Finset.univ.sum fun i => (vector1 i) * (vector2 i)) = -102 := by
  sorry

end NUMINAMATH_CALUDE_dot_product_equals_negative_102_l2704_270457


namespace NUMINAMATH_CALUDE_zane_picked_62_pounds_l2704_270444

/-- The amount of garbage picked up by Daliah in pounds -/
def daliah_garbage : ℝ := 17.5

/-- The amount of garbage picked up by Dewei in pounds -/
def dewei_garbage : ℝ := daliah_garbage - 2

/-- The amount of garbage picked up by Zane in pounds -/
def zane_garbage : ℝ := 4 * dewei_garbage

/-- Theorem stating that Zane picked up 62 pounds of garbage -/
theorem zane_picked_62_pounds : zane_garbage = 62 := by sorry

end NUMINAMATH_CALUDE_zane_picked_62_pounds_l2704_270444


namespace NUMINAMATH_CALUDE_solve_equation_l2704_270434

theorem solve_equation : ∃ x : ℝ, 35 - (23 - (15 - x)) = 12 * 2 / (1 / 2) ∧ x = -21 := by
  sorry

end NUMINAMATH_CALUDE_solve_equation_l2704_270434


namespace NUMINAMATH_CALUDE_rectangle_dimension_change_l2704_270467

/-- If a rectangle's width is halved and its area increases by 30.000000000000004%,
    then the length of the rectangle increases by 160%. -/
theorem rectangle_dimension_change (L W : ℝ) (L' W' : ℝ) (h1 : W' = W / 2) 
  (h2 : L' * W' = 1.30000000000000004 * L * W) : L' = 2.6 * L := by
  sorry

end NUMINAMATH_CALUDE_rectangle_dimension_change_l2704_270467


namespace NUMINAMATH_CALUDE_inequality_proof_l2704_270416

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  (b + c) / (2 * a) + (c + a) / (2 * b) + (a + b) / (2 * c) ≥
  2 * a / (b + c) + 2 * b / (c + a) + 2 * c / (a + b) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l2704_270416


namespace NUMINAMATH_CALUDE_symmetric_points_difference_l2704_270418

/-- Two points are symmetric with respect to the origin if their coordinates are negations of each other -/
def symmetric_wrt_origin (A B : ℝ × ℝ) : Prop :=
  A.1 = -B.1 ∧ A.2 = -B.2

theorem symmetric_points_difference (a b : ℝ) :
  let A : ℝ × ℝ := (-2, b)
  let B : ℝ × ℝ := (a, 3)
  symmetric_wrt_origin A B → a - b = 5 := by
  sorry

end NUMINAMATH_CALUDE_symmetric_points_difference_l2704_270418


namespace NUMINAMATH_CALUDE_inner_cube_surface_area_l2704_270486

/-- Given a cube of volume 64 cubic meters with an inscribed sphere, which in turn has an inscribed cube, 
    the surface area of the inner cube is 32 square meters. -/
theorem inner_cube_surface_area (outer_cube : Real → Real → Real → Bool) 
  (outer_sphere : Real → Real → Real → Bool) (inner_cube : Real → Real → Real → Bool) :
  (∀ x y z, outer_cube x y z ↔ (0 ≤ x ∧ x ≤ 4) ∧ (0 ≤ y ∧ y ≤ 4) ∧ (0 ≤ z ∧ z ≤ 4)) →
  (∀ x y z, outer_sphere x y z ↔ (x - 2)^2 + (y - 2)^2 + (z - 2)^2 ≤ 4) →
  (∀ x y z, inner_cube x y z → outer_sphere x y z) →
  (∃! l : Real, ∀ x y z, inner_cube x y z ↔ 
    (0 ≤ x ∧ x ≤ l) ∧ (0 ≤ y ∧ y ≤ l) ∧ (0 ≤ z ∧ z ≤ l) ∧ l^2 + l^2 + l^2 = 16) →
  (∃ sa : Real, sa = 6 * (4 * Real.sqrt 3 / 3)^2 ∧ sa = 32) :=
by sorry

end NUMINAMATH_CALUDE_inner_cube_surface_area_l2704_270486


namespace NUMINAMATH_CALUDE_line_intersects_circle_l2704_270412

/-- 
Given a > 0, prove that the line x + a²y - a = 0 intersects 
the circle (x - a)² + (y - 1/a)² = 1
-/
theorem line_intersects_circle (a : ℝ) (h : a > 0) : 
  ∃ (x y : ℝ), (x + a^2 * y - a = 0) ∧ ((x - a)^2 + (y - 1/a)^2 = 1) :=
by sorry

end NUMINAMATH_CALUDE_line_intersects_circle_l2704_270412


namespace NUMINAMATH_CALUDE_jimmy_stair_climbing_time_l2704_270454

/-- Represents the time taken to climb stairs with an increasing time for each flight -/
def stair_climbing_time (initial_time : ℕ) (time_increase : ℕ) (num_flights : ℕ) : ℕ :=
  (num_flights * (2 * initial_time + (num_flights - 1) * time_increase)) / 2

/-- Theorem stating the total time Jimmy takes to climb eight flights of stairs -/
theorem jimmy_stair_climbing_time :
  stair_climbing_time 30 10 8 = 520 := by
  sorry

end NUMINAMATH_CALUDE_jimmy_stair_climbing_time_l2704_270454


namespace NUMINAMATH_CALUDE_field_length_is_96_l2704_270406

/-- Proves that the length of a rectangular field is 96 meters given the specified conditions. -/
theorem field_length_is_96 (l w : ℝ) (h1 : l = 2 * w) (h2 : 64 = (1 / 72) * (l * w)) : l = 96 := by
  sorry

end NUMINAMATH_CALUDE_field_length_is_96_l2704_270406


namespace NUMINAMATH_CALUDE_meat_pie_cost_l2704_270458

/-- The cost of a meat pie given Gerald's initial farthings and remaining pfennigs -/
theorem meat_pie_cost
  (initial_farthings : ℕ)
  (farthings_per_pfennig : ℕ)
  (remaining_pfennigs : ℕ)
  (h1 : initial_farthings = 54)
  (h2 : farthings_per_pfennig = 6)
  (h3 : remaining_pfennigs = 7)
  : (initial_farthings / farthings_per_pfennig) - remaining_pfennigs = 2 := by
  sorry

#check meat_pie_cost

end NUMINAMATH_CALUDE_meat_pie_cost_l2704_270458


namespace NUMINAMATH_CALUDE_total_salaries_l2704_270455

/-- The problem of calculating total salaries given specific conditions -/
theorem total_salaries (A_salary B_salary : ℝ) : 
  A_salary = 2250 →
  0.05 * A_salary = 0.15 * B_salary →
  A_salary + B_salary = 3000 := by
  sorry

#check total_salaries

end NUMINAMATH_CALUDE_total_salaries_l2704_270455


namespace NUMINAMATH_CALUDE_proctoring_arrangements_l2704_270428

theorem proctoring_arrangements (n : ℕ) (k : ℕ) (h : n = 12 ∧ k = 8) :
  (Nat.choose n k) * ((n - k) * (n - k - 1)) = 4455 :=
sorry

end NUMINAMATH_CALUDE_proctoring_arrangements_l2704_270428


namespace NUMINAMATH_CALUDE_arccos_sin_three_l2704_270462

theorem arccos_sin_three (x : ℝ) : x = Real.arccos (Real.sin 3) → x = 3 - π / 2 := by
  sorry

end NUMINAMATH_CALUDE_arccos_sin_three_l2704_270462


namespace NUMINAMATH_CALUDE_m_range_l2704_270459

theorem m_range (m : ℝ) : (∀ x > 0, x + 1/x - m > 0) → m < 2 := by
  sorry

end NUMINAMATH_CALUDE_m_range_l2704_270459


namespace NUMINAMATH_CALUDE_existence_of_m_l2704_270495

def M : Set ℕ := {n : ℕ | n ≤ 2007}

def arithmetic_progression (n : ℕ) : Set ℕ :=
  {k : ℕ | ∃ i : ℕ, k = n + i * (n + 1)}

theorem existence_of_m :
  (∀ n ∈ M, arithmetic_progression n ⊆ M) →
  ∃ m : ℕ, ∀ k > m, k ∈ M :=
sorry

end NUMINAMATH_CALUDE_existence_of_m_l2704_270495


namespace NUMINAMATH_CALUDE_fraction_simplification_l2704_270435

theorem fraction_simplification (x : ℝ) (h1 : x ≠ -2) (h2 : x ≠ 3) :
  (3*x^2 - 2*x - 4) / ((x+2)*(x-3)) - (5+x) / ((x+2)*(x-3)) = 3*(x^2-x-3) / ((x+2)*(x-3)) :=
by sorry

end NUMINAMATH_CALUDE_fraction_simplification_l2704_270435


namespace NUMINAMATH_CALUDE_train_journey_time_l2704_270473

theorem train_journey_time (X : ℝ) (h1 : 0 < X) (h2 : X < 60) : 
  (X * 6 - X * 0.5 = 360 - X) → X = 360 / 7 := by
  sorry

end NUMINAMATH_CALUDE_train_journey_time_l2704_270473


namespace NUMINAMATH_CALUDE_B_2_1_equals_12_l2704_270422

-- Define the function B using the given recurrence relation
def B : ℕ → ℕ → ℕ
| 0, n => n + 2
| m + 1, 0 => B m 2
| m + 1, n + 1 => B m (B (m + 1) n)

-- Theorem statement
theorem B_2_1_equals_12 : B 2 1 = 12 := by
  sorry

end NUMINAMATH_CALUDE_B_2_1_equals_12_l2704_270422


namespace NUMINAMATH_CALUDE_soccer_penalty_kicks_l2704_270498

theorem soccer_penalty_kicks (total_players : ℕ) (goalkeepers : ℕ) 
  (h1 : total_players = 24) 
  (h2 : goalkeepers = 4) 
  (h3 : goalkeepers ≤ total_players) : 
  (total_players - 1) * goalkeepers = 92 := by
  sorry

end NUMINAMATH_CALUDE_soccer_penalty_kicks_l2704_270498


namespace NUMINAMATH_CALUDE_square_plus_reciprocal_square_l2704_270420

theorem square_plus_reciprocal_square (x : ℝ) (h : x + 1/x = 5) : x^2 + 1/x^2 = 23 := by
  sorry

end NUMINAMATH_CALUDE_square_plus_reciprocal_square_l2704_270420


namespace NUMINAMATH_CALUDE_angle_A_triangle_area_l2704_270482

-- Define the triangle ABC
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

-- Define the conditions
def triangle_conditions (t : Triangle) : Prop :=
  t.a = Real.sqrt 7 ∧
  t.b = 2 ∧
  Real.sqrt 3 * t.b * Real.cos t.A = t.a * Real.sin t.B

-- Theorem for angle A
theorem angle_A (t : Triangle) (h : triangle_conditions t) : t.A = π / 3 :=
sorry

-- Theorem for area of triangle ABC
theorem triangle_area (t : Triangle) (h : triangle_conditions t) : 
  (1 / 2) * t.b * t.c * Real.sin t.A = (3 * Real.sqrt 3) / 2 :=
sorry

end NUMINAMATH_CALUDE_angle_A_triangle_area_l2704_270482


namespace NUMINAMATH_CALUDE_reflect_point_coordinates_l2704_270439

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Reflects a point across the y-axis -/
def reflectAcrossYAxis (p : Point) : Point :=
  { x := -p.x, y := p.y }

theorem reflect_point_coordinates :
  let P : Point := { x := 4, y := -1 }
  reflectAcrossYAxis P = { x := -4, y := -1 } := by
  sorry

end NUMINAMATH_CALUDE_reflect_point_coordinates_l2704_270439


namespace NUMINAMATH_CALUDE_chord_cosine_theorem_l2704_270468

theorem chord_cosine_theorem (r : ℝ) (φ ψ θ : ℝ) 
  (h1 : φ + ψ + θ < π)
  (h2 : 3^2 = 2*r^2 - 2*r^2*Real.cos φ)
  (h3 : 4^2 = 2*r^2 - 2*r^2*Real.cos ψ)
  (h4 : 5^2 = 2*r^2 - 2*r^2*Real.cos θ)
  (h5 : 12^2 = 2*r^2 - 2*r^2*Real.cos (φ + ψ + θ))
  (h6 : r > 0) :
  Real.cos φ = 7/8 := by
  sorry

end NUMINAMATH_CALUDE_chord_cosine_theorem_l2704_270468


namespace NUMINAMATH_CALUDE_china_gdp_scientific_notation_l2704_270407

def trillion : ℝ := 10^12

theorem china_gdp_scientific_notation :
  11.69 * trillion = 1.169 * 10^14 := by sorry

end NUMINAMATH_CALUDE_china_gdp_scientific_notation_l2704_270407


namespace NUMINAMATH_CALUDE_correct_staffing_arrangements_l2704_270437

def total_members : ℕ := 6
def positions_to_fill : ℕ := 4
def restricted_members : ℕ := 2
def restricted_positions : ℕ := 1

def staffing_arrangements (n m k r : ℕ) : ℕ :=
  (n.factorial / (n - m).factorial) - k * ((n - 1).factorial / (n - m).factorial)

theorem correct_staffing_arrangements :
  staffing_arrangements total_members positions_to_fill restricted_members restricted_positions = 240 := by
  sorry

end NUMINAMATH_CALUDE_correct_staffing_arrangements_l2704_270437


namespace NUMINAMATH_CALUDE_woodburning_profit_l2704_270446

/-- Calculate the profit from selling woodburnings -/
theorem woodburning_profit (num_sold : ℕ) (price_per_item : ℕ) (wood_cost : ℕ) :
  num_sold = 20 →
  price_per_item = 15 →
  wood_cost = 100 →
  num_sold * price_per_item - wood_cost = 200 := by
sorry

end NUMINAMATH_CALUDE_woodburning_profit_l2704_270446


namespace NUMINAMATH_CALUDE_arithmetic_sequence_common_difference_l2704_270443

theorem arithmetic_sequence_common_difference
  (a : ℕ → ℝ)  -- a is a sequence of real numbers indexed by natural numbers
  (h_arithmetic : ∀ n, a (n + 1) - a n = a (n + 2) - a (n + 1))  -- a is an arithmetic sequence
  (h_sum1 : a 2 + a 6 = 8)  -- given condition
  (h_sum2 : a 3 + a 4 = 3)  -- given condition
  : ∃ d, ∀ n, a (n + 1) - a n = d ∧ d = 5 :=
by sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_common_difference_l2704_270443


namespace NUMINAMATH_CALUDE_moss_pollen_diameter_scientific_notation_l2704_270483

/-- Expresses a given decimal number in scientific notation -/
def scientificNotation (x : ℝ) : ℝ × ℤ :=
  sorry

theorem moss_pollen_diameter_scientific_notation :
  scientificNotation 0.0000084 = (8.4, -6) := by sorry

end NUMINAMATH_CALUDE_moss_pollen_diameter_scientific_notation_l2704_270483


namespace NUMINAMATH_CALUDE_sqrt_expression_equality_l2704_270478

theorem sqrt_expression_equality : 
  Real.sqrt 18 / Real.sqrt 6 - Real.sqrt 12 + Real.sqrt 48 * Real.sqrt (1/3) = -Real.sqrt 3 + 4 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_expression_equality_l2704_270478


namespace NUMINAMATH_CALUDE_derivative_f_at_one_l2704_270479

noncomputable def f (x : ℝ) : ℝ := Real.exp x / x

theorem derivative_f_at_one : 
  deriv f 1 = 0 := by sorry

end NUMINAMATH_CALUDE_derivative_f_at_one_l2704_270479


namespace NUMINAMATH_CALUDE_divisibility_by_three_l2704_270465

theorem divisibility_by_three (u v : ℤ) : 
  (9 ∣ u^2 + u*v + v^2) → (3 ∣ u) ∧ (3 ∣ v) := by
  sorry

end NUMINAMATH_CALUDE_divisibility_by_three_l2704_270465


namespace NUMINAMATH_CALUDE_quality_difference_confidence_l2704_270488

/-- Production data for two machines -/
structure ProductionData :=
  (machine_a_first : ℕ)
  (machine_a_second : ℕ)
  (machine_b_first : ℕ)
  (machine_b_second : ℕ)

/-- Calculate K^2 statistic -/
def calculate_k_squared (data : ProductionData) : ℚ :=
  let n := data.machine_a_first + data.machine_a_second + data.machine_b_first + data.machine_b_second
  let a := data.machine_a_first
  let b := data.machine_a_second
  let c := data.machine_b_first
  let d := data.machine_b_second
  n * (a * d - b * c)^2 / ((a + b) * (c + d) * (a + c) * (b + d))

/-- Critical values for K^2 test -/
def critical_value_99_percent : ℚ := 6635 / 1000
def critical_value_999_percent : ℚ := 10828 / 1000

/-- Theorem stating the confidence level for the difference in quality -/
theorem quality_difference_confidence (data : ProductionData) 
  (h1 : data.machine_a_first = 150)
  (h2 : data.machine_a_second = 50)
  (h3 : data.machine_b_first = 120)
  (h4 : data.machine_b_second = 80) :
  critical_value_99_percent < calculate_k_squared data ∧ 
  calculate_k_squared data < critical_value_999_percent :=
sorry

end NUMINAMATH_CALUDE_quality_difference_confidence_l2704_270488


namespace NUMINAMATH_CALUDE_particle_position_after_1991_minutes_l2704_270494

-- Define the particle's position type
def Position := ℤ × ℤ

-- Define the starting position
def start_position : Position := (0, 1)

-- Define the movement pattern for a single rectangle
def rectangle_movement (n : ℕ) : Position := 
  if n % 2 = 1 then (n, n + 1) else (-(n + 1), -n)

-- Define the time taken for a single rectangle
def rectangle_time (n : ℕ) : ℕ := 2 * n + 1

-- Define the total time for n rectangles
def total_time (n : ℕ) : ℕ := (n + 1)^2 - 1

-- Define the function to calculate the position after n rectangles
def position_after_rectangles (n : ℕ) : Position :=
  sorry

-- Define the function to calculate the final position
def final_position (time : ℕ) : Position :=
  sorry

-- The theorem to prove
theorem particle_position_after_1991_minutes :
  final_position 1991 = (-45, -32) :=
sorry

end NUMINAMATH_CALUDE_particle_position_after_1991_minutes_l2704_270494


namespace NUMINAMATH_CALUDE_fish_pond_population_l2704_270471

theorem fish_pond_population (initial_tagged : ℕ) (second_catch : ℕ) (tagged_in_second : ℕ) :
  initial_tagged = 30 →
  second_catch = 50 →
  tagged_in_second = 2 →
  (tagged_in_second : ℚ) / second_catch = initial_tagged / (initial_tagged + 750) :=
by sorry

end NUMINAMATH_CALUDE_fish_pond_population_l2704_270471


namespace NUMINAMATH_CALUDE_lisas_teaspoons_l2704_270413

theorem lisas_teaspoons (num_children : ℕ) (baby_spoons_per_child : ℕ) (decorative_spoons : ℕ) 
  (large_spoons : ℕ) (total_spoons : ℕ) :
  num_children = 4 →
  baby_spoons_per_child = 3 →
  decorative_spoons = 2 →
  large_spoons = 10 →
  total_spoons = 39 →
  total_spoons - (num_children * baby_spoons_per_child + decorative_spoons + large_spoons) = 15 := by
  sorry

end NUMINAMATH_CALUDE_lisas_teaspoons_l2704_270413


namespace NUMINAMATH_CALUDE_initial_phone_count_prove_initial_phone_count_l2704_270451

theorem initial_phone_count : ℕ → Prop :=
  fun initial_count =>
    let defective_count : ℕ := 5
    let customer_a_bought : ℕ := 3
    let customer_b_bought : ℕ := 5
    let customer_c_bought : ℕ := 7
    let total_sold := customer_a_bought + customer_b_bought + customer_c_bought
    initial_count - defective_count = total_sold ∧ initial_count = 20

theorem prove_initial_phone_count :
  ∃ (x : ℕ), initial_phone_count x :=
sorry

end NUMINAMATH_CALUDE_initial_phone_count_prove_initial_phone_count_l2704_270451


namespace NUMINAMATH_CALUDE_gasoline_price_growth_rate_l2704_270452

theorem gasoline_price_growth_rate (initial_price final_price : ℝ) (months : ℕ) (x : ℝ) 
  (h1 : initial_price = 6.2)
  (h2 : final_price = 8.9)
  (h3 : months = 2)
  (h4 : x > 0)
  : initial_price * (1 + x)^months = final_price := by
  sorry

end NUMINAMATH_CALUDE_gasoline_price_growth_rate_l2704_270452


namespace NUMINAMATH_CALUDE_min_sum_with_constraint_l2704_270401

theorem min_sum_with_constraint (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : 4 * a + b = a * b) :
  a + b ≥ 9 ∧ ∃ (a₀ b₀ : ℝ), a₀ > 0 ∧ b₀ > 0 ∧ 4 * a₀ + b₀ = a₀ * b₀ ∧ a₀ + b₀ = 9 := by
  sorry

end NUMINAMATH_CALUDE_min_sum_with_constraint_l2704_270401


namespace NUMINAMATH_CALUDE_age_difference_l2704_270472

theorem age_difference (patrick michael monica nathan : ℝ) : 
  patrick / michael = 3 / 5 →
  michael / monica = 3 / 4 →
  monica / nathan = 5 / 7 →
  nathan / patrick = 4 / 9 →
  patrick + michael + monica + nathan = 252 →
  nathan - patrick = 66.5 := by
sorry

end NUMINAMATH_CALUDE_age_difference_l2704_270472


namespace NUMINAMATH_CALUDE_normal_distribution_probability_l2704_270430

def normal_distribution (μ σ : ℝ) (X : ℝ → ℝ) : Prop :=
  ∃ f : ℝ → ℝ, ∀ x, f x = (1 / (σ * Real.sqrt (2 * Real.pi))) * Real.exp (-(1/2) * ((x - μ) / σ)^2)

theorem normal_distribution_probability (X : ℝ → ℝ) (μ σ : ℝ) :
  normal_distribution μ σ X →
  (∫ x in Set.Ioo (μ - 2*σ) (μ + 2*σ), X x) = 0.9544 →
  (∫ x in Set.Ioo (μ - σ) (μ + σ), X x) = 0.6826 →
  μ = 4 →
  σ = 1 →
  (∫ x in Set.Ioo 5 6, X x) = 0.1359 :=
by sorry

end NUMINAMATH_CALUDE_normal_distribution_probability_l2704_270430


namespace NUMINAMATH_CALUDE_same_sign_range_l2704_270461

theorem same_sign_range (m : ℝ) : 
  ((2 - m) * (|m| - 3) > 0) → (m ∈ Set.Ioo 2 3 ∪ Set.Iio (-3)) := by
  sorry

end NUMINAMATH_CALUDE_same_sign_range_l2704_270461


namespace NUMINAMATH_CALUDE_ascending_order_abc_l2704_270432

theorem ascending_order_abc : 
  let a := (Real.sqrt 2 / 2) * (Real.sin (17 * π / 180) + Real.cos (17 * π / 180))
  let b := 2 * (Real.cos (13 * π / 180))^2 - 1
  let c := Real.sqrt 3 / 2
  c < a ∧ a < b := by sorry

end NUMINAMATH_CALUDE_ascending_order_abc_l2704_270432


namespace NUMINAMATH_CALUDE_frog_jump_probability_l2704_270427

-- Define the square
def Square := {(x, y) : ℝ × ℝ | 0 ≤ x ∧ x ≤ 6 ∧ 0 ≤ y ∧ y ≤ 6}

-- Define the vertical sides of the square
def VerticalSides := {(x, y) : ℝ × ℝ | (x = 0 ∨ x = 6) ∧ 0 ≤ y ∧ y ≤ 6}

-- Define the possible jump directions
inductive Direction
| Up
| Down
| Left
| Right

-- Define a function to represent a single jump
def jump (pos : ℝ × ℝ) (dir : Direction) : ℝ × ℝ :=
  match dir with
  | Direction.Up => (pos.1, pos.2 + 2)
  | Direction.Down => (pos.1, pos.2 - 2)
  | Direction.Left => (pos.1 - 2, pos.2)
  | Direction.Right => (pos.1 + 2, pos.2)

-- Define the probability function
noncomputable def P (pos : ℝ × ℝ) : ℝ :=
  sorry  -- The actual implementation would go here

-- State the theorem
theorem frog_jump_probability :
  P (1, 3) = 2/3 :=
sorry

end NUMINAMATH_CALUDE_frog_jump_probability_l2704_270427


namespace NUMINAMATH_CALUDE_arithmetic_sequence_iff_constant_difference_l2704_270405

/-- A sequence is arithmetic if and only if the difference between consecutive terms is constant -/
theorem arithmetic_sequence_iff_constant_difference (a : ℕ → ℝ) :
  (∃ d : ℝ, ∀ n : ℕ, a (n + 1) - a n = d) ↔ 
  (∃ a₀ d : ℝ, ∀ n : ℕ, a n = a₀ + n • d) :=
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_iff_constant_difference_l2704_270405


namespace NUMINAMATH_CALUDE_remainder_theorem_l2704_270403

theorem remainder_theorem : ∃ q : ℕ, 3^303 + 303 = (3^151 + 3^76 + 1) * q + 303 := by
  sorry

end NUMINAMATH_CALUDE_remainder_theorem_l2704_270403


namespace NUMINAMATH_CALUDE_geometric_sequence_value_l2704_270440

/-- A geometric sequence with sum of first n terms S_n = a · 2^n + a - 2 -/
def GeometricSequence (a : ℝ) : ℕ → ℝ := fun n ↦ 
  if n = 0 then 0 else (a * 2^n + a - 2) - (a * 2^(n-1) + a - 2)

/-- The sum of the first n terms of the geometric sequence -/
def SumFirstNTerms (a : ℝ) : ℕ → ℝ := fun n ↦ a * 2^n + a - 2

theorem geometric_sequence_value (a : ℝ) : 
  (∀ n : ℕ, n ≥ 1 → GeometricSequence a (n+1) / GeometricSequence a n = GeometricSequence a 2 / GeometricSequence a 1) →
  a = 1 := by
  sorry


end NUMINAMATH_CALUDE_geometric_sequence_value_l2704_270440


namespace NUMINAMATH_CALUDE_import_tax_percentage_l2704_270419

theorem import_tax_percentage 
  (total_value : ℝ) 
  (non_taxed_portion : ℝ) 
  (import_tax_amount : ℝ) 
  (h1 : total_value = 2580) 
  (h2 : non_taxed_portion = 1000) 
  (h3 : import_tax_amount = 110.60) : 
  (import_tax_amount / (total_value - non_taxed_portion)) * 100 = 7 := by
  sorry

end NUMINAMATH_CALUDE_import_tax_percentage_l2704_270419


namespace NUMINAMATH_CALUDE_trumpet_trombone_difference_l2704_270438

/-- Represents the number of players for each instrument in the school band --/
structure BandComposition where
  flute : Nat
  trumpet : Nat
  trombone : Nat
  drummer : Nat
  clarinet : Nat
  french_horn : Nat

/-- Theorem stating the difference between trumpet and trombone players --/
theorem trumpet_trombone_difference (band : BandComposition) : 
  band.flute = 5 →
  band.trumpet = 3 * band.flute →
  band.trumpet > band.trombone →
  band.drummer = band.trombone + 11 →
  band.clarinet = 2 * band.flute →
  band.french_horn = band.trombone + 3 →
  band.flute + band.trumpet + band.trombone + band.drummer + band.clarinet + band.french_horn = 65 →
  band.trumpet - band.trombone = 8 := by
  sorry


end NUMINAMATH_CALUDE_trumpet_trombone_difference_l2704_270438


namespace NUMINAMATH_CALUDE_point_order_on_line_l2704_270490

theorem point_order_on_line (m n b : ℝ) : 
  (2 * (-1/2) + b = m) → (2 * 2 + b = n) → m < n := by sorry

end NUMINAMATH_CALUDE_point_order_on_line_l2704_270490


namespace NUMINAMATH_CALUDE_distribution_schemes_eq_60_l2704_270447

/-- Represents the number of girls in the group. -/
def num_girls : ℕ := 5

/-- Represents the number of boys in the group. -/
def num_boys : ℕ := 2

/-- Represents the number of places for volunteer activities. -/
def num_places : ℕ := 2

/-- Calculates the number of ways to distribute girls and boys to two places. -/
def distribution_schemes : ℕ := sorry

/-- Theorem stating that the number of distribution schemes is 60. -/
theorem distribution_schemes_eq_60 : distribution_schemes = 60 := by sorry

end NUMINAMATH_CALUDE_distribution_schemes_eq_60_l2704_270447


namespace NUMINAMATH_CALUDE_rectangle_center_line_slope_l2704_270400

/-- The slope of a line passing through the origin and the center of a rectangle
    with vertices (1, 0), (5, 0), (1, 2), and (5, 2) is 1/3. -/
theorem rectangle_center_line_slope :
  let vertices : List (ℝ × ℝ) := [(1, 0), (5, 0), (1, 2), (5, 2)]
  let center : ℝ × ℝ := (
    (vertices.map Prod.fst).sum / vertices.length,
    (vertices.map Prod.snd).sum / vertices.length
  )
  let slope : ℝ := (center.2 - 0) / (center.1 - 0)
  slope = 1 / 3 := by
sorry


end NUMINAMATH_CALUDE_rectangle_center_line_slope_l2704_270400
