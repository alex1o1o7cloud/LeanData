import Mathlib

namespace theorem_A_theorem_B_theorem_C_theorem_D_l140_14029

-- Define the types for planes and lines
variable (Plane Line : Type)

-- Define the relations between planes and lines
variable (parallel_planes : Plane → Plane → Prop)
variable (perpendicular_plane_line : Plane → Line → Prop)
variable (parallel_line_plane : Line → Plane → Prop)
variable (perpendicular_planes : Plane → Plane → Prop)
variable (line_in_plane : Line → Plane → Prop)
variable (perpendicular_lines : Line → Line → Prop)
variable (parallel_lines : Line → Line → Prop)
variable (intersection : Plane → Plane → Line)

-- Define the planes and lines
variable (α β : Plane)
variable (m n : Line)

-- Axioms for the relations
axiom different_planes : α ≠ β
axiom different_lines : m ≠ n

-- Theorem A
theorem theorem_A : 
  parallel_planes α β → perpendicular_plane_line α m → perpendicular_plane_line β m :=
sorry

-- Theorem B
theorem theorem_B :
  perpendicular_plane_line α m → perpendicular_plane_line α n → parallel_lines m n :=
sorry

-- Theorem C
theorem theorem_C :
  perpendicular_planes α β → intersection α β = n → ¬parallel_line_plane m α → 
  perpendicular_lines m n → perpendicular_plane_line β m :=
sorry

-- Theorem D (which should be false)
theorem theorem_D :
  parallel_line_plane m α → parallel_line_plane n α → 
  parallel_line_plane m β → parallel_line_plane n β → 
  ¬(parallel_planes α β) :=
sorry

end theorem_A_theorem_B_theorem_C_theorem_D_l140_14029


namespace smallest_undefined_inverse_l140_14010

theorem smallest_undefined_inverse (b : ℕ) : b = 6 ↔ 
  (b > 0) ∧ 
  (∀ x : ℕ, x * b % 30 ≠ 1) ∧ 
  (∀ y : ℕ, y * b % 42 ≠ 1) ∧ 
  (∀ c < b, c > 0 → (∃ x : ℕ, x * c % 30 = 1) ∨ (∃ y : ℕ, y * c % 42 = 1)) :=
by sorry

end smallest_undefined_inverse_l140_14010


namespace article_original_price_l140_14034

/-- Calculates the original price of an article given its selling price and loss percentage. -/
def originalPrice (sellingPrice : ℚ) (lossPercent : ℚ) : ℚ :=
  sellingPrice / (1 - lossPercent / 100)

/-- Theorem stating that an article sold for 450 with a 25% loss had an original price of 600. -/
theorem article_original_price :
  originalPrice 450 25 = 600 := by
  sorry

end article_original_price_l140_14034


namespace monotonic_decreasing_interval_l140_14092

def f (x : ℝ) := x^3 - 15*x^2 - 33*x + 6

theorem monotonic_decreasing_interval :
  {x : ℝ | ∀ y, x < y → f x > f y} = {x | -1 < x ∧ x < 11} := by sorry

end monotonic_decreasing_interval_l140_14092


namespace like_terms_exponents_l140_14036

theorem like_terms_exponents (a b : ℝ) (x y : ℝ) : 
  (∃ k : ℝ, 2 * a^(2*x) * b^(3*y) = k * (-3 * a^2 * b^(2-x))) → 
  x = 1 ∧ y = 1/3 :=
sorry

end like_terms_exponents_l140_14036


namespace fourth_power_sum_l140_14088

theorem fourth_power_sum (a : ℝ) (h : a^2 - 3*a + 1 = 0) : a^4 + 1/a^4 = 47 := by
  sorry

end fourth_power_sum_l140_14088


namespace alphabet_value_problem_l140_14076

theorem alphabet_value_problem (T M A H E : ℤ) : 
  T = 15 →
  M + A + T + H = 47 →
  T + E + A + M = 58 →
  M + E + E + T = 45 →
  M = 8 := by
sorry

end alphabet_value_problem_l140_14076


namespace cubic_roots_sum_cubes_l140_14043

theorem cubic_roots_sum_cubes (r s t : ℝ) : 
  (6 * r^3 + 4 * r^2 + 1500 * r + 3000 = 0) →
  (6 * s^3 + 4 * s^2 + 1500 * s + 3000 = 0) →
  (6 * t^3 + 4 * t^2 + 1500 * t + 3000 = 0) →
  (r + s)^3 + (s + t)^3 + (t + r)^3 = -5992/27 := by
sorry

end cubic_roots_sum_cubes_l140_14043


namespace billy_sodas_l140_14012

/-- The number of sodas in Billy's pack -/
def sodas_in_pack (sisters : ℕ) (brothers : ℕ) (sodas_per_sibling : ℕ) : ℕ :=
  (sisters + brothers) * sodas_per_sibling

/-- Theorem: The number of sodas in Billy's pack is 12 -/
theorem billy_sodas :
  ∀ (sisters brothers sodas_per_sibling : ℕ),
    brothers = 2 * sisters →
    sisters = 2 →
    sodas_per_sibling = 2 →
    sodas_in_pack sisters brothers sodas_per_sibling = 12 := by
  sorry

end billy_sodas_l140_14012


namespace tshirt_price_proof_l140_14024

/-- The regular price of a T-shirt -/
def regular_price : ℝ := 14.50

/-- The cost of a discounted T-shirt -/
def discount_price : ℝ := 1

/-- The total number of T-shirts bought -/
def total_shirts : ℕ := 12

/-- The total cost of all T-shirts -/
def total_cost : ℝ := 120

/-- The number of T-shirts in a "lot" (2 regular + 1 discounted) -/
def lot_size : ℕ := 3

theorem tshirt_price_proof :
  regular_price * (2 * (total_shirts / lot_size)) + 
  discount_price * (total_shirts / lot_size) = total_cost :=
sorry

end tshirt_price_proof_l140_14024


namespace no_double_application_function_l140_14021

theorem no_double_application_function :
  ¬∃ (f : ℕ → ℕ), ∀ (x : ℕ), f (f x) = x + 1 := by
  sorry

end no_double_application_function_l140_14021


namespace digit_sum_problem_l140_14032

theorem digit_sum_problem (A B C D E : ℕ) : 
  (A < 10) → (B < 10) → (C < 10) → (D < 10) → (E < 10) →
  (10 * E + A) + (10 * E + C) = 10 * D + A →
  (10 * E + A) - (10 * E + C) = A →
  D = 8 := by
sorry

end digit_sum_problem_l140_14032


namespace remaining_debt_percentage_l140_14025

def original_debt : ℝ := 500
def initial_payment : ℝ := 125

theorem remaining_debt_percentage :
  (original_debt - initial_payment) / original_debt * 100 = 75 := by
sorry

end remaining_debt_percentage_l140_14025


namespace quadratic_inequality_solution_set_l140_14017

theorem quadratic_inequality_solution_set :
  {x : ℝ | x^2 + x - 6 > 0} = {x : ℝ | x < -3 ∨ x > 2} := by
  sorry

end quadratic_inequality_solution_set_l140_14017


namespace set_A_range_l140_14087

theorem set_A_range (a : ℝ) : 
  let A := {x : ℝ | a * x^2 - 3 * x - 4 = 0}
  (∀ x y : ℝ, x ∈ A → y ∈ A → x = y) → 
  (a ≤ -9/16 ∨ a = 0) := by
sorry

end set_A_range_l140_14087


namespace chantel_bracelets_l140_14000

def bracelet_problem (
  days1 : ℕ) (bracelets_per_day1 : ℕ) (give_away1 : ℕ)
  (days2 : ℕ) (bracelets_per_day2 : ℕ) (give_away2 : ℕ)
  (days3 : ℕ) (bracelets_per_day3 : ℕ)
  (days4 : ℕ) (bracelets_per_day4 : ℕ) (give_away3 : ℕ) : ℕ :=
  (days1 * bracelets_per_day1 - give_away1 +
   days2 * bracelets_per_day2 - give_away2 +
   days3 * bracelets_per_day3 +
   days4 * bracelets_per_day4 - give_away3)

theorem chantel_bracelets :
  bracelet_problem 7 4 8 10 5 12 4 6 2 3 10 = 78 := by
  sorry

end chantel_bracelets_l140_14000


namespace tangent_line_sum_l140_14063

/-- Given a function f: ℝ → ℝ with a tangent line 2x - y - 3 = 0 at x = 2,
    prove that f(2) + f'(2) = 3 -/
theorem tangent_line_sum (f : ℝ → ℝ) (hf : Differentiable ℝ f) 
    (h_tangent : ∀ x y, y = f 2 → 2*x - y - 3 = 0 ↔ y = f x) : 
    f 2 + deriv f 2 = 3 := by
  sorry

end tangent_line_sum_l140_14063


namespace range_of_m_l140_14002

def p (x : ℝ) : Prop := |x - 4| ≤ 6

def q (x m : ℝ) : Prop := x^2 - 2*x + 1 - m^2 ≤ 0

theorem range_of_m :
  (∀ x m : ℝ, (¬(p x) → ¬(q x m)) ∧ ∃ x, ¬(p x) ∧ (q x m)) →
  ∀ m : ℝ, -3 ≤ m ∧ m ≤ 3 :=
sorry

end range_of_m_l140_14002


namespace last_row_value_l140_14033

/-- Represents a triangular table with the given properties -/
def TriangularTable (n : ℕ) : Type :=
  Fin n → Fin n → ℕ

/-- The first row of the table contains the first n positive integers -/
def FirstRowProperty (t : TriangularTable 100) : Prop :=
  ∀ i : Fin 100, t 0 i = i.val + 1

/-- Each element (from the second row onwards) is the sum of the two elements directly above it -/
def SumProperty (t : TriangularTable 100) : Prop :=
  ∀ i j : Fin 100, i > 0 → j < i → t i j = t (i-1) j + t (i-1) (j+1)

/-- The last row contains only one element -/
def LastRowProperty (t : TriangularTable 100) : Prop :=
  t 99 0 = t 99 0  -- This is always true, but it ensures the element exists

/-- The main theorem: the value in the last row is 101 × 2^98 -/
theorem last_row_value (t : TriangularTable 100) 
  (h1 : FirstRowProperty t) 
  (h2 : SumProperty t) 
  (h3 : LastRowProperty t) : 
  t 99 0 = 101 * 2^98 := by
  sorry

end last_row_value_l140_14033


namespace painting_sale_difference_l140_14016

def previous_painting_sale : ℕ := 9000
def recent_painting_sale : ℕ := 44000

theorem painting_sale_difference : 
  (5 * previous_painting_sale + previous_painting_sale) - recent_painting_sale = 10000 := by
  sorry

end painting_sale_difference_l140_14016


namespace inequality_proof_l140_14064

theorem inequality_proof (a b c d : ℝ) 
  (h_order : a ≥ b ∧ b ≥ c ∧ c ≥ d)
  (h_product : (a-b)*(b-c)*(c-d)*(d-a) = -3) :
  (a + b + c + d = 6 → d < 0.36) ∧
  (a^2 + b^2 + c^2 + d^2 = 14 → (a+c)*(b+d) ≤ 8) :=
by sorry

end inequality_proof_l140_14064


namespace unique_triple_l140_14065

theorem unique_triple : 
  ∃! (A B C : ℕ), A^2 + B - C = 100 ∧ A + B^2 - C = 124 :=
by
  -- The proof would go here
  sorry

end unique_triple_l140_14065


namespace trumpet_players_count_l140_14096

def orchestra_size : ℕ := 21
def drummer_count : ℕ := 1
def trombone_count : ℕ := 4
def french_horn_count : ℕ := 1
def violinist_count : ℕ := 3
def cellist_count : ℕ := 1
def contrabassist_count : ℕ := 1
def clarinet_count : ℕ := 3
def flute_count : ℕ := 4
def maestro_count : ℕ := 1

theorem trumpet_players_count :
  orchestra_size - (drummer_count + trombone_count + french_horn_count + 
    violinist_count + cellist_count + contrabassist_count + 
    clarinet_count + flute_count + maestro_count) = 2 := by
  sorry

end trumpet_players_count_l140_14096


namespace turtle_conservation_count_l140_14073

theorem turtle_conservation_count :
  let green_turtles : ℕ := 800
  let hawksbill_turtles : ℕ := 2 * green_turtles
  let total_turtles : ℕ := green_turtles + hawksbill_turtles
  total_turtles = 2400 :=
by sorry

end turtle_conservation_count_l140_14073


namespace chef_potato_problem_l140_14086

/-- The number of potatoes already cooked -/
def potatoes_cooked : ℕ := 7

/-- The time it takes to cook one potato (in minutes) -/
def cooking_time_per_potato : ℕ := 5

/-- The time it takes to cook the remaining potatoes (in minutes) -/
def remaining_cooking_time : ℕ := 45

/-- The total number of potatoes the chef needs to cook -/
def total_potatoes : ℕ := 16

theorem chef_potato_problem :
  total_potatoes = potatoes_cooked + remaining_cooking_time / cooking_time_per_potato :=
by sorry

end chef_potato_problem_l140_14086


namespace similar_triangle_area_reduction_l140_14059

/-- Given a right-angled triangle with area A and hypotenuse H, if a smaller similar triangle
    is formed by cutting parallel to the hypotenuse such that the new hypotenuse H' = 0.65H,
    then the area A' of the smaller triangle is equal to A * (0.65)^2. -/
theorem similar_triangle_area_reduction (A H H' A' : ℝ) 
    (h1 : A > 0) 
    (h2 : H > 0) 
    (h3 : H' = 0.65 * H) 
    (h4 : A' / A = (H' / H)^2) : 
  A' = A * (0.65)^2 := by
  sorry

#check similar_triangle_area_reduction

end similar_triangle_area_reduction_l140_14059


namespace modulus_of_z_l140_14070

theorem modulus_of_z (z : ℂ) (h : z * (2 - 3*I) = 6 + 4*I) : Complex.abs z = 2 := by
  sorry

end modulus_of_z_l140_14070


namespace balog_theorem_l140_14001

theorem balog_theorem (q : ℕ+) (A : Finset ℤ) :
  ∃ (C_q : ℕ), (A.card + q * A.card : ℤ) ≥ ((q + 1) * A.card : ℤ) - C_q :=
by sorry

end balog_theorem_l140_14001


namespace soap_brands_survey_l140_14074

theorem soap_brands_survey (total : Nat) (neither : Nat) (only_a : Nat) (both : Nat) : 
  total = 300 →
  neither = 80 →
  only_a = 60 →
  total = neither + only_a + 3 * both + both →
  both = 40 := by
sorry

end soap_brands_survey_l140_14074


namespace inequality_solution_set_range_of_m_l140_14062

-- Define the function f(x) = |x+1|
def f (x : ℝ) : ℝ := |x + 1|

-- Theorem for the solution set of the inequality
theorem inequality_solution_set :
  {x : ℝ | f x ≥ 2 * x + 1} = {x : ℝ | x ≤ 0} :=
sorry

-- Theorem for the range of m
theorem range_of_m :
  ∀ m : ℝ, (∃ x : ℝ, f (x - 2) - f (x + 6) < m) ↔ m > -8 :=
sorry

end inequality_solution_set_range_of_m_l140_14062


namespace smallest_prime_with_digit_sum_17_gt_200_l140_14080

def digit_sum (n : ℕ) : ℕ :=
  if n < 10 then n else n % 10 + digit_sum (n / 10)

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ d : ℕ, d > 1 → d < n → ¬(n % d = 0)

theorem smallest_prime_with_digit_sum_17_gt_200 :
  ∃ p : ℕ,
    is_prime p ∧
    digit_sum p = 17 ∧
    p > 200 ∧
    (∀ q : ℕ, is_prime q → digit_sum q = 17 → q > 200 → p ≤ q) ∧
    p = 197 :=
sorry

end smallest_prime_with_digit_sum_17_gt_200_l140_14080


namespace compound_interest_principal_l140_14030

/-- Given a principal amount and an annual compound interest rate,
    prove that the principal amount is approximately 5967.79 if it grows
    to 8000 after 2 years and 9261 after 3 years under compound interest. -/
theorem compound_interest_principal (P r : ℝ) 
  (h1 : P * (1 + r)^2 = 8000)
  (h2 : P * (1 + r)^3 = 9261) :
  ∃ ε > 0, |P - 5967.79| < ε :=
sorry

end compound_interest_principal_l140_14030


namespace mitch_weekend_to_weekday_ratio_l140_14018

/-- Represents Mitch's work schedule and earnings --/
structure MitchSchedule where
  weekdayHours : ℕ  -- Hours worked per weekday
  weekendHours : ℕ  -- Hours worked per weekend day
  weekdayRate : ℚ   -- Hourly rate for weekdays
  totalEarnings : ℚ -- Total weekly earnings

/-- Calculates the ratio of weekend rate to weekday rate --/
def weekendToWeekdayRatio (schedule : MitchSchedule) : ℚ :=
  let totalWeekdayHours := schedule.weekdayHours * 5
  let totalWeekendHours := schedule.weekendHours * 2
  let weekdayEarnings := schedule.weekdayRate * totalWeekdayHours
  let weekendEarnings := schedule.totalEarnings - weekdayEarnings
  let weekendRate := weekendEarnings / totalWeekendHours
  weekendRate / schedule.weekdayRate

/-- Theorem stating that Mitch's weekend to weekday rate ratio is 2:1 --/
theorem mitch_weekend_to_weekday_ratio :
  let schedule : MitchSchedule := {
    weekdayHours := 5,
    weekendHours := 3,
    weekdayRate := 3,
    totalEarnings := 111
  }
  weekendToWeekdayRatio schedule = 2 := by
  sorry

end mitch_weekend_to_weekday_ratio_l140_14018


namespace imaginary_part_of_z_l140_14091

-- Define the complex number z
variable (z : ℂ)

-- Define the condition
def condition (z : ℂ) : Prop := 1 + z = 2 + 3 * Complex.I

-- Theorem statement
theorem imaginary_part_of_z (h : condition z) : z.im = 3 := by
  sorry

end imaginary_part_of_z_l140_14091


namespace garden_bed_area_l140_14057

/-- Represents the dimensions of a rectangular garden bed -/
structure GardenBed where
  length : ℝ
  width : ℝ

/-- Calculates the area of a garden bed -/
def area (bed : GardenBed) : ℝ := bed.length * bed.width

/-- Theorem: Given the conditions, prove that the area of each unknown garden bed is 9 sq ft -/
theorem garden_bed_area 
  (known_bed : GardenBed)
  (unknown_bed : GardenBed)
  (h1 : known_bed.length = 4)
  (h2 : known_bed.width = 3)
  (h3 : area known_bed + area known_bed + area unknown_bed + area unknown_bed = 42) :
  area unknown_bed = 9 := by
  sorry

end garden_bed_area_l140_14057


namespace trajectory_equation_l140_14049

theorem trajectory_equation (x y : ℝ) (h1 : x > 0) :
  (((x - 1/2)^2 + y^2)^(1/2) = x + 1/2) → y^2 = 2*x := by
  sorry

end trajectory_equation_l140_14049


namespace puzzle_spells_bach_l140_14079

/-- Represents a musical symbol --/
inductive MusicalSymbol
  | DoubleFlatSolKey
  | ATenorClef
  | CAltoClef
  | BNaturalSolKey

/-- Represents the interpretation rules --/
def interpretSymbol (s : MusicalSymbol) : Char :=
  match s with
  | MusicalSymbol.DoubleFlatSolKey => 'B'
  | MusicalSymbol.ATenorClef => 'A'
  | MusicalSymbol.CAltoClef => 'C'
  | MusicalSymbol.BNaturalSolKey => 'H'

/-- The sequence of symbols in the puzzle --/
def puzzleSequence : List MusicalSymbol := [
  MusicalSymbol.DoubleFlatSolKey,
  MusicalSymbol.ATenorClef,
  MusicalSymbol.CAltoClef,
  MusicalSymbol.BNaturalSolKey
]

/-- The theorem stating that the puzzle sequence spells "BACH" --/
theorem puzzle_spells_bach :
  puzzleSequence.map interpretSymbol = ['B', 'A', 'C', 'H'] := by
  sorry


end puzzle_spells_bach_l140_14079


namespace square_of_integer_ending_in_five_l140_14053

theorem square_of_integer_ending_in_five (a : ℤ) : (10 * a + 5)^2 = 100 * a * (a + 1) + 25 := by
  sorry

end square_of_integer_ending_in_five_l140_14053


namespace line_l_equation_no_symmetric_points_l140_14066

-- Define the lines l₁ and l₂
def l₁ (x y : ℝ) : Prop := 2*x + y + 1 = 0
def l₂ (x y : ℝ) : Prop := x - 2*y - 3 = 0

-- Define line l
def l (x y : ℝ) : Prop := x + y = 0

-- Define the parabola
def parabola (a x y : ℝ) : Prop := y = a*x^2 - 1

-- Theorem 1: Prove that l is the correct line given the midpoint condition
theorem line_l_equation : 
  (∃ (x₁ y₁ x₂ y₂ : ℝ), l₁ x₁ y₁ ∧ l₂ x₂ y₂ ∧ l x₁ y₁ ∧ l x₂ y₂ ∧ (x₁ + x₂)/2 = 0 ∧ (y₁ + y₂)/2 = 0) →
  (∀ x y : ℝ, l x y ↔ x + y = 0) :=
sorry

-- Theorem 2: Prove the condition for non-existence of symmetric points
theorem no_symmetric_points (a : ℝ) :
  (a ≠ 0) →
  (∀ x₁ y₁ x₂ y₂ : ℝ, 
    (parabola a x₁ y₁ ∧ parabola a x₂ y₂ ∧ 
     (x₁ + x₂)/2 + (y₁ + y₂)/2 = 0) → x₁ = x₂ ∧ y₁ = y₂) ↔ 
  (a ≤ 3/4) :=
sorry

end line_l_equation_no_symmetric_points_l140_14066


namespace sum_of_even_and_multiples_of_five_l140_14094

/-- The number of four-digit even numbers -/
def C : ℕ := 4500

/-- The number of four-digit multiples of 5 -/
def B : ℕ := 1800

/-- The sum of four-digit even numbers and four-digit multiples of 5 is 6300 -/
theorem sum_of_even_and_multiples_of_five : C + B = 6300 := by
  sorry

end sum_of_even_and_multiples_of_five_l140_14094


namespace a_not_square_l140_14045

/-- Sequence definition -/
def a : ℕ → ℚ
  | 0 => 2016
  | n + 1 => a n + 2 / (a n)

/-- Theorem statement -/
theorem a_not_square : ∀ n : ℕ, ¬ ∃ q : ℚ, a n = q ^ 2 := by
  sorry

end a_not_square_l140_14045


namespace triangle_perimeter_is_seven_l140_14035

-- Define the triangle sides
variable (a b c : ℝ)

-- Define the condition equation
def condition (a b c : ℝ) : Prop :=
  a^2 + b^2 + c^2 - 4*a - 4*b - 6*c + 17 = 0

-- Define what it means for a, b, c to form a triangle
def is_triangle (a b c : ℝ) : Prop :=
  a > 0 ∧ b > 0 ∧ c > 0 ∧ a + b > c ∧ b + c > a ∧ c + a > b

-- Define the perimeter
def perimeter (a b c : ℝ) : ℝ := a + b + c

-- State the theorem
theorem triangle_perimeter_is_seven 
  (h1 : is_triangle a b c) 
  (h2 : condition a b c) : 
  perimeter a b c = 7 :=
sorry

end triangle_perimeter_is_seven_l140_14035


namespace wedge_volume_l140_14048

/-- The volume of a wedge cut from a cylindrical log -/
theorem wedge_volume (d : ℝ) (α : ℝ) (V : ℝ) : 
  d = 10 → -- diameter of the log
  α = 60 → -- angle between the two cuts in degrees
  V = (125/18) * Real.pi → -- volume of the wedge
  ∃ (r h : ℝ),
    r = d/2 ∧ -- radius of the log
    h = r ∧ -- height of the cone (equal to radius due to 60° angle)
    V = (1/6) * ((1/3) * Real.pi * r^2 * h) -- volume formula
  :=
by sorry

end wedge_volume_l140_14048


namespace sqrt_equation_solution_l140_14082

theorem sqrt_equation_solution : 
  ∀ x : ℝ, (Real.sqrt (5 * x - 4) + 12 / Real.sqrt (5 * x - 4) = 9) ↔ (x = 13/5 ∨ x = 4) :=
by sorry

end sqrt_equation_solution_l140_14082


namespace johnson_yield_l140_14077

/-- Represents the yield of corn per hectare every two months -/
structure CornYield where
  amount : ℕ

/-- Represents a cornfield -/
structure Cornfield where
  hectares : ℕ
  yield : CornYield

def total_yield (field : Cornfield) (periods : ℕ) : ℕ :=
  field.hectares * field.yield.amount * periods

theorem johnson_yield (johnson : Cornfield) (neighbor : Cornfield) 
    (h1 : johnson.hectares = 1)
    (h2 : neighbor.hectares = 2)
    (h3 : neighbor.yield.amount = 2 * johnson.yield.amount)
    (h4 : total_yield johnson 3 + total_yield neighbor 3 = 1200) :
  johnson.yield.amount = 80 := by
  sorry

#check johnson_yield

end johnson_yield_l140_14077


namespace max_cables_used_l140_14060

/-- Represents the number of brand A computers -/
def brand_A_count : Nat := 25

/-- Represents the number of brand B computers -/
def brand_B_count : Nat := 15

/-- Represents the total number of employees -/
def total_employees : Nat := brand_A_count + brand_B_count

/-- Theorem stating the maximum number of cables that can be used -/
theorem max_cables_used : 
  ∀ (cables : Nat), 
    (cables ≤ brand_A_count * brand_B_count) → 
    (∀ (b : Nat), b < brand_B_count → ∃ (a : Nat), a < brand_A_count ∧ True) → 
    cables ≤ 375 :=
by sorry

end max_cables_used_l140_14060


namespace smallest_m_with_divisible_digit_sum_l140_14046

/-- Represents the sum of digits of a natural number -/
def digitSum (n : ℕ) : ℕ := sorry

/-- Checks if a number's digit sum is divisible by 6 -/
def hasSumDivisibleBy6 (n : ℕ) : Prop :=
  (digitSum n) % 6 = 0

/-- Main theorem: 9 is the smallest m satisfying the condition -/
theorem smallest_m_with_divisible_digit_sum : 
  ∀ (start : ℕ), ∃ (i : ℕ), i < 9 ∧ hasSumDivisibleBy6 (start + i) ∧
  ∀ (m : ℕ), m < 9 → ∃ (start' : ℕ), ∀ (j : ℕ), j < m → ¬hasSumDivisibleBy6 (start' + j) :=
sorry

end smallest_m_with_divisible_digit_sum_l140_14046


namespace work_completion_men_count_l140_14008

theorem work_completion_men_count :
  ∀ (M : ℕ),
  (∃ (W : ℕ), W = M * 9) →  -- Original work amount
  (∃ (W : ℕ), W = (M + 10) * 6) →  -- Same work amount after adding 10 men
  M = 20 :=
by sorry

end work_completion_men_count_l140_14008


namespace solve_exponential_equation_l140_14081

theorem solve_exponential_equation :
  ∃ n : ℕ, 4^n * 4^n * 4^n = 16^3 ∧ n = 2 :=
by sorry

end solve_exponential_equation_l140_14081


namespace shorter_leg_of_second_triangle_l140_14095

/-- Represents a 30-60-90 triangle -/
structure Triangle30_60_90 where
  hypotenuse : ℝ
  shorterLeg : ℝ
  longerLeg : ℝ
  hypotenuse_eq : hypotenuse = 2 * shorterLeg
  longerLeg_eq : longerLeg = shorterLeg * Real.sqrt 3

/-- A sequence of two 30-60-90 triangles where the hypotenuse of the first is the longer leg of the second -/
def TwoTriangles (t1 t2 : Triangle30_60_90) :=
  t1.hypotenuse = 12 ∧ t1.longerLeg = t2.hypotenuse

theorem shorter_leg_of_second_triangle (t1 t2 : Triangle30_60_90) 
  (h : TwoTriangles t1 t2) : t2.shorterLeg = 3 * Real.sqrt 3 := by
  sorry

end shorter_leg_of_second_triangle_l140_14095


namespace monotonic_cubic_function_a_range_l140_14052

/-- The function f(x) = -x^3 + ax^2 - x - 1 is monotonic on ℝ if and only if a ∈ [-√3, √3] -/
theorem monotonic_cubic_function_a_range (a : ℝ) :
  (∀ x : ℝ, Monotone (fun x => -x^3 + a*x^2 - x - 1)) ↔ a ∈ Set.Icc (-Real.sqrt 3) (Real.sqrt 3) :=
sorry

end monotonic_cubic_function_a_range_l140_14052


namespace max_abc_value_l140_14015

theorem max_abc_value (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c)
  (h1 : a * b + c + a * b = (a + c) * (b + c))
  (h2 : a + b + c = 2) :
  a * b * c ≤ 8 / 27 :=
sorry

end max_abc_value_l140_14015


namespace largest_prime_factor_l140_14078

theorem largest_prime_factor : 
  (∃ (p : ℕ), Nat.Prime p ∧ p ∣ (16^4 + 2 * 16^2 + 1 - 15^4) ∧ 
    ∀ (q : ℕ), Nat.Prime q → q ∣ (16^4 + 2 * 16^2 + 1 - 15^4) → q ≤ p) ∧
  (Nat.Prime 241 ∧ 241 ∣ (16^4 + 2 * 16^2 + 1 - 15^4)) := by
  sorry

end largest_prime_factor_l140_14078


namespace olivia_correct_answers_l140_14007

theorem olivia_correct_answers 
  (total_problems : ℕ) 
  (correct_points : ℤ) 
  (incorrect_points : ℤ) 
  (total_score : ℤ) 
  (h1 : total_problems = 15)
  (h2 : correct_points = 4)
  (h3 : incorrect_points = -3)
  (h4 : total_score = 25) : 
  ∃ (correct_answers : ℕ), 
    correct_answers = 10 ∧ 
    correct_answers ≤ total_problems ∧
    (correct_points * correct_answers + incorrect_points * (total_problems - correct_answers) = total_score) := by
  sorry

end olivia_correct_answers_l140_14007


namespace defective_units_shipped_l140_14019

theorem defective_units_shipped (total_units : ℝ) (defective_rate : ℝ) (shipped_rate : ℝ) :
  defective_rate = 0.04 →
  shipped_rate = 0.04 →
  (defective_rate * shipped_rate * 100) = 0.16 := by
sorry

end defective_units_shipped_l140_14019


namespace zero_to_positive_power_l140_14050

theorem zero_to_positive_power (n : ℕ+) : 0 ^ (n : ℕ) = 0 := by
  sorry

end zero_to_positive_power_l140_14050


namespace no_f_iteration_to_one_l140_14039

def f (n : ℤ) : ℤ :=
  if n % 2 = 1 then n^2 + 1 else n / 2 + 3

def iterateF (n : ℤ) (k : ℕ) : ℤ :=
  match k with
  | 0 => n
  | k + 1 => f (iterateF n k)

theorem no_f_iteration_to_one :
  ∀ n : ℤ, 1 ≤ n ∧ n ≤ 100 → ∀ k : ℕ, iterateF n k ≠ 1 :=
sorry

end no_f_iteration_to_one_l140_14039


namespace ellipse_foci_distance_l140_14047

/-- The distance between the foci of an ellipse with semi-major axis 9 and semi-minor axis 3 -/
theorem ellipse_foci_distance (a b : ℝ) (ha : a = 9) (hb : b = 3) :
  2 * Real.sqrt (a^2 - b^2) = 12 * Real.sqrt 2 := by
  sorry

end ellipse_foci_distance_l140_14047


namespace vector_magnitude_l140_14040

def a : ℝ × ℝ := (2, 1)
def c : ℝ × ℝ := (2, -4)

theorem vector_magnitude (x : ℝ) (b : ℝ × ℝ) 
  (h1 : b = (-1, x))
  (h2 : ∃ k : ℝ, b.1 = k * c.1 ∧ b.2 = k * c.2) :
  ‖a + b‖ = Real.sqrt 10 := by
  sorry

end vector_magnitude_l140_14040


namespace rectangle_area_l140_14004

/-- The area of a rectangle with given vertices in a rectangular coordinate system -/
theorem rectangle_area (v1 v2 v3 v4 : ℝ × ℝ) : 
  v1 = (-8, 1) → v2 = (1, 1) → v3 = (1, -7) → v4 = (-8, -7) →
  let width := |v2.1 - v1.1|
  let height := |v2.2 - v3.2|
  width * height = 72 := by sorry

end rectangle_area_l140_14004


namespace shaded_area_octagon_semicircles_l140_14020

/-- The area of the shaded region inside a regular octagon but outside eight semicircles -/
theorem shaded_area_octagon_semicircles : 
  let s : ℝ := 4  -- side length of the octagon
  let octagon_area : ℝ := 2 * (1 + Real.sqrt 2) * s^2
  let semicircle_area : ℝ := π * (s/2)^2 / 2
  let total_semicircle_area : ℝ := 8 * semicircle_area
  octagon_area - total_semicircle_area = 32 * (1 + Real.sqrt 2) - 16 * π :=
by sorry

end shaded_area_octagon_semicircles_l140_14020


namespace tan_double_angle_l140_14084

theorem tan_double_angle (α : Real) 
  (h : (Real.sin α - Real.cos α) / (Real.sin α + Real.cos α) = 1/2) : 
  Real.tan (2 * α) = -3/4 := by
  sorry

end tan_double_angle_l140_14084


namespace sin_plus_cos_special_angle_l140_14093

/-- Given a point P(-3,4) on the terminal side of angle α, prove that sin α + cos α = 1/5 -/
theorem sin_plus_cos_special_angle (α : Real) :
  let P : ℝ × ℝ := (-3, 4)
  (∃ t : ℝ, t > 0 ∧ P.1 = t * Real.cos α ∧ P.2 = t * Real.sin α) →
  Real.sin α + Real.cos α = 1/5 := by
sorry

end sin_plus_cos_special_angle_l140_14093


namespace student_cannot_enter_finals_l140_14023

/-- Represents the competition structure and student's performance -/
structure Competition where
  total_rounds : ℕ
  required_specified : ℕ
  required_creative : ℕ
  min_selected_for_award : ℕ
  rounds_for_finals : ℕ
  specified_selected : ℕ
  total_specified : ℕ
  creative_selected : ℕ
  total_creative : ℕ
  prob_increase : ℚ

/-- Calculates the probability of winning the "Skillful Hands Award" in one round -/
def prob_win_award (c : Competition) : ℚ :=
  sorry

/-- Calculates the expected number of times winning the award in all rounds after intensive training -/
def expected_wins_after_training (c : Competition) : ℚ :=
  sorry

/-- Main theorem: The student cannot enter the finals -/
theorem student_cannot_enter_finals (c : Competition) 
  (h1 : c.total_rounds = 5)
  (h2 : c.required_specified = 2)
  (h3 : c.required_creative = 2)
  (h4 : c.min_selected_for_award = 3)
  (h5 : c.rounds_for_finals = 4)
  (h6 : c.specified_selected = 4)
  (h7 : c.total_specified = 5)
  (h8 : c.creative_selected = 3)
  (h9 : c.total_creative = 5)
  (h10 : c.prob_increase = 1/10) :
  prob_win_award c = 33/50 ∧ expected_wins_after_training c < 4 :=
sorry

end student_cannot_enter_finals_l140_14023


namespace percent_defective_units_l140_14006

/-- Given that 4% of defective units are shipped for sale and 0.32% of all units
    produced are defective units that are shipped for sale, prove that 8% of
    all units produced are defective. -/
theorem percent_defective_units (shipped_defective_ratio : Real)
                                 (total_shipped_defective_ratio : Real)
                                 (h1 : shipped_defective_ratio = 0.04)
                                 (h2 : total_shipped_defective_ratio = 0.0032) :
  shipped_defective_ratio * (total_shipped_defective_ratio / shipped_defective_ratio) = 0.08 := by
  sorry

end percent_defective_units_l140_14006


namespace isosceles_triangle_removal_l140_14022

/-- Given a square with isosceles right triangles removed from each corner to form a rectangle,
    if the diagonal of the resulting rectangle is 15 units,
    then the combined area of the four removed triangles is 112.5 square units. -/
theorem isosceles_triangle_removal (r s : ℝ) : 
  r > 0 → s > 0 →  -- r and s are positive real numbers
  (r + s)^2 + (r - s)^2 = 15^2 →  -- diagonal of resulting rectangle is 15
  2 * r * s = 112.5  -- combined area of four removed triangles
  := by sorry

end isosceles_triangle_removal_l140_14022


namespace inequality_proof_l140_14061

def M : Set ℝ := {x : ℝ | |x + 1| + |x - 1| ≤ 2}

theorem inequality_proof (x y z : ℝ) (hx : x ∈ M) (hy : |y| ≤ 1/6) (hz : |z| ≤ 1/9) :
  |x + 2*y - 3*z| ≤ 5/3 := by
  sorry

end inequality_proof_l140_14061


namespace special_function_property_l140_14044

/-- A function f: ℝ → ℝ satisfying specific properties -/
def special_function (f : ℝ → ℝ) : Prop :=
  (∀ x, f (x - 1) = -f (-x - 1)) ∧  -- f(x-1) is odd
  (∀ x, f (x + 1) = f (-x + 1)) ∧  -- f(x+1) is even
  (∀ x, x > -1 ∧ x < 1 → f x = -Real.exp x)  -- f(x) = -e^x for x ∈ (-1,1)

/-- Theorem stating the property of the special function -/
theorem special_function_property (f : ℝ → ℝ) (h : special_function f) :
  ∀ x, f (2 * x) = f (2 * x + 8) :=
by sorry

end special_function_property_l140_14044


namespace max_envelopes_proof_l140_14031

def number_of_bus_tickets : ℕ := 18
def number_of_subway_tickets : ℕ := 12

def max_envelopes : ℕ := Nat.gcd number_of_bus_tickets number_of_subway_tickets

theorem max_envelopes_proof :
  (∀ k : ℕ, k ∣ number_of_bus_tickets ∧ k ∣ number_of_subway_tickets → k ≤ max_envelopes) ∧
  (max_envelopes ∣ number_of_bus_tickets) ∧
  (max_envelopes ∣ number_of_subway_tickets) :=
sorry

end max_envelopes_proof_l140_14031


namespace torn_page_numbers_l140_14041

theorem torn_page_numbers (n : ℕ) (k : ℕ) : 
  n > 0 ∧ k > 1 ∧ k < n ∧ (n * (n + 1)) / 2 - (2 * k - 1) = 15000 → k = 113 := by
  sorry

end torn_page_numbers_l140_14041


namespace coefficient_m5n5_in_expansion_l140_14054

theorem coefficient_m5n5_in_expansion : (Nat.choose 10 5) = 252 := by
  sorry

end coefficient_m5n5_in_expansion_l140_14054


namespace six_disks_common_point_implies_center_inside_l140_14027

-- Define a disk in 2D space
structure Disk :=
  (center : ℝ × ℝ)
  (radius : ℝ)

-- Define what it means for a point to be inside a disk
def isInside (p : ℝ × ℝ) (d : Disk) : Prop :=
  let (x, y) := p
  let (cx, cy) := d.center
  (x - cx)^2 + (y - cy)^2 < d.radius^2

-- Define a set of six disks
def SixDisks := Fin 6 → Disk

-- The theorem statement
theorem six_disks_common_point_implies_center_inside
  (disks : SixDisks)
  (common_point : ℝ × ℝ)
  (h : ∀ i : Fin 6, isInside common_point (disks i)) :
  ∃ i j : Fin 6, i ≠ j ∧ isInside (disks j).center (disks i) :=
sorry

end six_disks_common_point_implies_center_inside_l140_14027


namespace sqrt_twelve_simplification_l140_14055

theorem sqrt_twelve_simplification : Real.sqrt 12 = 2 * Real.sqrt 3 := by
  sorry

end sqrt_twelve_simplification_l140_14055


namespace infinitely_many_composite_generating_numbers_l140_14085

theorem infinitely_many_composite_generating_numbers :
  ∃ f : ℕ → ℕ, Infinite {k | ∀ n : ℕ, ∃ x y : ℕ, x > 1 ∧ y > 1 ∧ n^4 + f k = x * y} :=
sorry

end infinitely_many_composite_generating_numbers_l140_14085


namespace tenth_term_of_arithmetic_sequence_l140_14058

def arithmetic_sequence (a : ℝ) (d : ℝ) (n : ℕ) : ℝ := a + (n - 1) * d

theorem tenth_term_of_arithmetic_sequence
  (a : ℝ)
  (d : ℝ)
  (h1 : arithmetic_sequence a d 3 = 23)
  (h2 : arithmetic_sequence a d 7 = 35) :
  arithmetic_sequence a d 10 = 44 :=
by
  sorry

end tenth_term_of_arithmetic_sequence_l140_14058


namespace root_between_roots_l140_14072

theorem root_between_roots (a b : ℝ) (α β : ℝ) 
  (hα : α^2 + a*α + b = 0) 
  (hβ : β^2 - a*β - b = 0) : 
  ∃ x, x ∈ Set.Icc α β ∧ x^2 - 2*a*x - 2*b = 0 := by
  sorry

end root_between_roots_l140_14072


namespace second_term_of_geometric_series_l140_14098

theorem second_term_of_geometric_series (a : ℝ) (r : ℝ) (S : ℝ) : 
  r = (1 : ℝ) / 4 →
  S = 16 →
  S = a / (1 - r) →
  a * r = 3 :=
by sorry

end second_term_of_geometric_series_l140_14098


namespace annabelle_savings_l140_14013

/-- Calculates the amount saved from a weekly allowance after spending on junk food and sweets -/
def calculate_savings (weekly_allowance : ℚ) (junk_food_fraction : ℚ) (sweets_cost : ℚ) : ℚ :=
  weekly_allowance - (weekly_allowance * junk_food_fraction + sweets_cost)

/-- Proves that given a weekly allowance of $30, spending 1/3 of it on junk food and an additional $8 on sweets, the remaining amount saved is $12 -/
theorem annabelle_savings :
  calculate_savings 30 (1/3) 8 = 12 := by
  sorry

end annabelle_savings_l140_14013


namespace symmetry_shift_l140_14014

noncomputable def smallest_shift_for_symmetry : ℝ := 7 * Real.pi / 6

def is_symmetric_about_y_axis (f : ℝ → ℝ) : Prop :=
  ∀ x, f x = f (-x)

theorem symmetry_shift :
  let f (x m : ℝ) := Real.cos (x + m) - Real.sqrt 3 * Real.sin (x + m)
  ∀ m : ℝ, m > 0 → (
    (is_symmetric_about_y_axis (f · m)) ↔ 
    m ≥ smallest_shift_for_symmetry
  ) :=
sorry

end symmetry_shift_l140_14014


namespace terms_difference_l140_14038

theorem terms_difference (k : ℕ) (h : k > 1) :
  (Finset.range (2^(k+1) - 1)).card - (Finset.range (2^k - 1)).card = 2^k :=
sorry

end terms_difference_l140_14038


namespace min_chips_for_adjacency_l140_14056

/-- Represents a color of a chip -/
def Color : Type := Fin 6

/-- Represents a row of chips -/
def ChipRow := List Color

/-- Checks if two colors are adjacent in a row -/
def areAdjacent (c1 c2 : Color) (row : ChipRow) : Prop :=
  ∃ i, (row.get? i = some c1 ∧ row.get? (i+1) = some c2) ∨
       (row.get? i = some c2 ∧ row.get? (i+1) = some c1)

/-- Checks if all pairs of colors are adjacent in a row -/
def allPairsAdjacent (row : ChipRow) : Prop :=
  ∀ c1 c2 : Color, c1 ≠ c2 → areAdjacent c1 c2 row

/-- The main theorem stating the minimum number of chips required -/
theorem min_chips_for_adjacency :
  ∃ (row : ChipRow), allPairsAdjacent row ∧ row.length = 18 ∧
  (∀ (row' : ChipRow), allPairsAdjacent row' → row'.length ≥ 18) :=
sorry

end min_chips_for_adjacency_l140_14056


namespace ellipse_line_intersection_ratio_l140_14089

/-- An ellipse intersecting a line with a specific midpoint property -/
structure EllipseLineIntersection where
  m : ℝ
  n : ℝ
  -- Ellipse equation: mx^2 + ny^2 = 1
  -- Line equation: x + y - 1 = 0
  -- Intersection points exist (implicit)
  -- Line through midpoint and origin has slope √2/2 (implicit)

/-- The ratio m/n equals √2/2 for the given ellipse-line intersection -/
theorem ellipse_line_intersection_ratio (e : EllipseLineIntersection) : e.m / e.n = Real.sqrt 2 / 2 := by
  sorry

end ellipse_line_intersection_ratio_l140_14089


namespace square_perimeter_ratio_l140_14068

theorem square_perimeter_ratio (s₁ s₂ : ℝ) (h : s₁ > 0) (h' : s₂ > 0) :
  s₂ * Real.sqrt 2 = 1.5 * (s₁ * Real.sqrt 2) →
  (4 * s₂) / (4 * s₁) = 3 / 2 := by
sorry

end square_perimeter_ratio_l140_14068


namespace k_value_is_four_thirds_l140_14069

/-- The function f(x) = x + 1 -/
def f (x : ℝ) : ℝ := x + 1

/-- The function g(x) = kx^2 - x - (k+1) -/
def g (k : ℝ) (x : ℝ) : ℝ := k * x^2 - x - (k + 1)

/-- The theorem stating that k = 4/3 given the conditions -/
theorem k_value_is_four_thirds (k : ℝ) (h1 : k > 1) :
  (∀ x₁ ∈ Set.Icc 2 4, ∃ x₂ ∈ Set.Icc 2 4, f x₁ / g k x₁ = g k x₂ / f x₂) →
  k = 4/3 := by sorry

end k_value_is_four_thirds_l140_14069


namespace root_equation_problem_l140_14097

/-- Given two constants p and q, if the specified equations have the given number of distinct roots
    and q = 8, then 50p - 10q = 20 -/
theorem root_equation_problem (p q : ℝ) : 
  (∃! x y z, x ≠ y ∧ y ≠ z ∧ x ≠ z ∧ 
    (x + p) * (x + q) * (x - 8) / (x + 4)^2 = 0) →
  (∃! x y, x ≠ y ∧ 
    (x + 4*p) * (x - 4) * (x - 10) / ((x + q) * (x - 8)) = 0) →
  q = 8 →
  50 * p - 10 * q = 20 := by
sorry

end root_equation_problem_l140_14097


namespace seed_germination_percentage_l140_14009

theorem seed_germination_percentage (seeds_plot1 seeds_plot2 : ℕ)
  (germination_rate_plot1 total_germination_rate : ℚ) :
  seeds_plot1 = 300 →
  seeds_plot2 = 200 →
  germination_rate_plot1 = 30 / 100 →
  total_germination_rate = 32 / 100 →
  (germination_rate_plot1 * seeds_plot1 + (35 / 100) * seeds_plot2) / (seeds_plot1 + seeds_plot2) = total_germination_rate :=
by sorry

end seed_germination_percentage_l140_14009


namespace self_employed_tax_calculation_l140_14005

/-- Calculates the tax amount for a self-employed citizen --/
def calculate_tax_amount (gross_income : ℝ) (tax_rate : ℝ) : ℝ :=
  gross_income * tax_rate

/-- Theorem: The tax amount for a self-employed citizen with a gross income of 350,000.00 rubles and a tax rate of 6% is 21,000.00 rubles --/
theorem self_employed_tax_calculation :
  let gross_income : ℝ := 350000.00
  let tax_rate : ℝ := 0.06
  calculate_tax_amount gross_income tax_rate = 21000.00 := by
  sorry

#eval calculate_tax_amount 350000.00 0.06

end self_employed_tax_calculation_l140_14005


namespace gloria_pine_tree_price_l140_14026

/-- Proves that the price per pine tree is $200 given the conditions of Gloria's cabin purchase --/
theorem gloria_pine_tree_price :
  let cabin_price : ℕ := 129000
  let initial_cash : ℕ := 150
  let cypress_trees : ℕ := 20
  let pine_trees : ℕ := 600
  let maple_trees : ℕ := 24
  let cypress_price : ℕ := 100
  let maple_price : ℕ := 300
  let remaining_cash : ℕ := 350
  let pine_price : ℕ := (cabin_price - initial_cash + remaining_cash - 
    (cypress_trees * cypress_price + maple_trees * maple_price)) / pine_trees
  pine_price = 200 := by sorry

end gloria_pine_tree_price_l140_14026


namespace number_problem_l140_14037

theorem number_problem : 
  let x : ℝ := 25
  80 / 100 * 60 - (4 / 5 * x) = 28 := by sorry

end number_problem_l140_14037


namespace plum_picking_l140_14051

/-- The number of plums picked by Melanie -/
def melanie_plums : ℕ := 4

/-- The number of plums picked by Dan -/
def dan_plums : ℕ := 9

/-- The number of plums picked by Sally -/
def sally_plums : ℕ := 3

/-- The total number of plums picked -/
def total_plums : ℕ := melanie_plums + dan_plums + sally_plums

theorem plum_picking :
  total_plums = 16 := by sorry

end plum_picking_l140_14051


namespace function_property_l140_14028

theorem function_property (f : ℤ → ℤ) 
  (h1 : ∀ x y : ℤ, f (x + y) = f x + f y + 2 * x * y + 1) 
  (h2 : f (-2) = 1) :
  ∀ n : ℕ+, f (2 * n) = 4 * n^2 + 2 * n - 1 :=
by sorry

end function_property_l140_14028


namespace perpendicular_vectors_l140_14071

/-- Given vectors a and b in ℝ², prove that if a + b is perpendicular to b, then the second component of a is 8. -/
theorem perpendicular_vectors (a b : ℝ × ℝ) (h : a.1 = 1 ∧ b = (3, -2)) :
  (a + b) • b = 0 → a.2 = 8 := by
sorry

end perpendicular_vectors_l140_14071


namespace triangle_area_theorem_l140_14042

-- Define the triangle ABC
def Triangle (A B C : ℝ × ℝ) := True

-- Define that ABC is a right-angled triangle
def IsRightAngled (A B C : ℝ × ℝ) := True

-- Define AD as an angle bisector
def IsAngleBisector (A B C D : ℝ × ℝ) := True

-- Define the lengths of the sides
def SideLength (A B : ℝ × ℝ) : ℝ := sorry

-- Define the area of a triangle
def TriangleArea (A B C : ℝ × ℝ) : ℝ := sorry

theorem triangle_area_theorem (A B C D : ℝ × ℝ) (x : ℝ) :
  Triangle A B C →
  IsRightAngled A B C →
  IsAngleBisector A B C D →
  SideLength A B = 100 →
  SideLength B C = x →
  SideLength A C = x + 10 →
  Int.floor (TriangleArea A D C + 0.5) = 20907 := by
  sorry

end triangle_area_theorem_l140_14042


namespace polygon_diagonals_l140_14083

/-- 
For an n-sided polygon, if 6 diagonals can be drawn from a single vertex, then n = 9.
-/
theorem polygon_diagonals (n : ℕ) : (n - 3 = 6) → n = 9 := by
  sorry

end polygon_diagonals_l140_14083


namespace least_positive_integer_with_given_remainders_l140_14075

theorem least_positive_integer_with_given_remainders : 
  ∃ (d : ℕ), d > 0 ∧ d % 7 = 1 ∧ d % 5 = 2 ∧ d % 3 = 2 ∧ 
  ∀ (n : ℕ), n > 0 ∧ n % 7 = 1 ∧ n % 5 = 2 ∧ n % 3 = 2 → d ≤ n :=
by
  use 92
  sorry

end least_positive_integer_with_given_remainders_l140_14075


namespace map_scale_theorem_l140_14099

/-- Represents the scale of a map in inches per foot -/
def scale : ℚ := 2 / 500

/-- Represents the length of a line segment on the map in inches -/
def map_length : ℚ := 29 / 4

/-- Calculates the actual length in feet represented by a length on the map -/
def actual_length (map_len : ℚ) : ℚ := map_len / scale

theorem map_scale_theorem :
  actual_length map_length = 1812.5 := by sorry

end map_scale_theorem_l140_14099


namespace bookstore_new_releases_fraction_l140_14003

/-- Represents a bookstore inventory --/
structure Bookstore where
  total : ℕ
  historicalFiction : ℕ
  historicalFictionNewReleases : ℕ
  otherNewReleases : ℕ

/-- Calculates the fraction of new releases that are historical fiction --/
def newReleasesFraction (store : Bookstore) : ℚ :=
  store.historicalFictionNewReleases / (store.historicalFictionNewReleases + store.otherNewReleases)

theorem bookstore_new_releases_fraction :
  ∀ (store : Bookstore),
    store.total > 0 →
    store.historicalFiction = (2 * store.total) / 5 →
    store.historicalFictionNewReleases = (2 * store.historicalFiction) / 5 →
    store.otherNewReleases = (7 * (store.total - store.historicalFiction)) / 10 →
    newReleasesFraction store = 8 / 29 := by
  sorry

end bookstore_new_releases_fraction_l140_14003


namespace joe_money_left_l140_14011

def initial_amount : ℕ := 56
def notebooks_bought : ℕ := 7
def books_bought : ℕ := 2
def notebook_cost : ℕ := 4
def book_cost : ℕ := 7

theorem joe_money_left : 
  initial_amount - (notebooks_bought * notebook_cost + books_bought * book_cost) = 14 := by
  sorry

end joe_money_left_l140_14011


namespace angle_PSQ_measure_l140_14090

-- Define the points
variable (K L M N P Q S : Point) (ω : Circle)

-- Define the trapezoid
def is_trapezoid (K L M N : Point) : Prop := sorry

-- Define the circle passing through L and M
def circle_through (ω : Circle) (L M : Point) : Prop := sorry

-- Define the circle intersecting KL at P and MN at Q
def circle_intersects (ω : Circle) (K L M N P Q : Point) : Prop := sorry

-- Define the circle tangent to KN at S
def circle_tangent_at (ω : Circle) (K N S : Point) : Prop := sorry

-- Define the angle measure
def angle_measure (A B C : Point) : ℝ := sorry

-- State the theorem
theorem angle_PSQ_measure 
  (h_trapezoid : is_trapezoid K L M N)
  (h_circle_through : circle_through ω L M)
  (h_circle_intersects : circle_intersects ω K L M N P Q)
  (h_circle_tangent : circle_tangent_at ω K N S)
  (h_angle_LSM : angle_measure L S M = 50)
  (h_angle_equal : angle_measure K L S = angle_measure S N M) :
  angle_measure P S Q = 65 := by sorry

end angle_PSQ_measure_l140_14090


namespace cake_recipe_salt_l140_14067

theorem cake_recipe_salt (sugar_total : ℕ) (salt : ℕ) : 
  sugar_total = 8 → 
  sugar_total = salt + 1 → 
  salt = 7 := by
sorry

end cake_recipe_salt_l140_14067
