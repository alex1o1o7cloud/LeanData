import Mathlib

namespace order_of_equations_l1387_138729

def order_of_diff_eq (eq : String) : ℕ :=
  match eq with
  | "y' + 2x = 0" => 1
  | "y'' + 3y' - 4 = 0" => 2
  | "2dy - 3x dx = 0" => 1
  | "y'' = cos x" => 2
  | _ => 0

theorem order_of_equations :
  (order_of_diff_eq "y' + 2x = 0" = 1) ∧
  (order_of_diff_eq "y'' + 3y' - 4 = 0" = 2) ∧
  (order_of_diff_eq "2dy - 3x dx = 0" = 1) ∧
  (order_of_diff_eq "y'' = cos x" = 2) := by
  sorry

end order_of_equations_l1387_138729


namespace circular_pond_area_l1387_138775

theorem circular_pond_area (AB CD : ℝ) (h1 : AB = 20) (h2 : CD = 12) : 
  let R := CD
  let A := π * R^2
  A = 244 * π := by sorry

end circular_pond_area_l1387_138775


namespace alcohol_mixture_proof_l1387_138719

/-- Proves that mixing 300 mL of 10% alcohol solution with 200 mL of 30% alcohol solution results in 18% alcohol solution -/
theorem alcohol_mixture_proof :
  let x_volume : ℝ := 300
  let y_volume : ℝ := 200
  let x_concentration : ℝ := 0.10
  let y_concentration : ℝ := 0.30
  let final_concentration : ℝ := 0.18
  (x_volume * x_concentration + y_volume * y_concentration) / (x_volume + y_volume) = final_concentration := by
  sorry

end alcohol_mixture_proof_l1387_138719


namespace square_operation_l1387_138738

theorem square_operation (x y : ℝ) (h1 : y = 68.70953354520753) (h2 : y^2 - x^2 = 4321) :
  ∃ (z : ℝ), z^2 = x^2 ∧ z = x :=
sorry

end square_operation_l1387_138738


namespace ratio_DO_OP_l1387_138757

/-- Parallelogram ABCD with points P on AB and Q on BC -/
structure Parallelogram (V : Type*) [AddCommGroup V] [Module ℝ V] :=
  (A B C D P Q O : V)
  (is_parallelogram : (C - B) = (D - A))
  (P_on_AB : ∃ t : ℝ, P = A + t • (B - A) ∧ 0 ≤ t ∧ t ≤ 1)
  (Q_on_BC : ∃ s : ℝ, Q = B + s • (C - B) ∧ 0 ≤ s ∧ s ≤ 1)
  (prop_AB_BP : 3 • (B - A) = 7 • (P - B))
  (prop_BC_BQ : 3 • (C - B) = 4 • (Q - B))
  (O_intersect : ∃ r t : ℝ, O = A + r • (Q - A) ∧ O = D + t • (P - D))

/-- The ratio DO : OP is 7 : 3 -/
theorem ratio_DO_OP (V : Type*) [AddCommGroup V] [Module ℝ V] (para : Parallelogram V) :
  ∃ k : ℝ, para.D - para.O = (7 * k) • (para.O - para.P) ∧ k ≠ 0 := by sorry

end ratio_DO_OP_l1387_138757


namespace haleigh_dogs_count_l1387_138792

/-- The number of cats Haleigh has -/
def num_cats : ℕ := 3

/-- The total number of leggings needed -/
def total_leggings : ℕ := 14

/-- The number of leggings each animal needs -/
def leggings_per_animal : ℕ := 1

/-- The number of dogs Haleigh has -/
def num_dogs : ℕ := total_leggings - (num_cats * leggings_per_animal)

theorem haleigh_dogs_count : num_dogs = 11 := by sorry

end haleigh_dogs_count_l1387_138792


namespace championship_outcomes_l1387_138766

/-- The number of possible outcomes for awarding n championship titles to m students. -/
def numberOfOutcomes (m n : ℕ) : ℕ := m^n

/-- Theorem: Given 8 students competing for 3 championship titles, 
    the number of possible outcomes for the champions is equal to 8^3. -/
theorem championship_outcomes : numberOfOutcomes 8 3 = 512 := by
  sorry

end championship_outcomes_l1387_138766


namespace total_wall_area_l1387_138701

/-- Represents the properties of tiles and the wall they cover -/
structure TileWall where
  regularTileArea : ℝ
  regularTileCount : ℝ
  jumboTileCount : ℝ
  jumboTileLengthRatio : ℝ

/-- The theorem stating the total wall area given the tile properties -/
theorem total_wall_area (w : TileWall)
  (h1 : w.regularTileArea * w.regularTileCount = 60)
  (h2 : w.jumboTileCount = w.regularTileCount / 3)
  (h3 : w.jumboTileLengthRatio = 3) :
  w.regularTileArea * w.regularTileCount + 
  (w.jumboTileLengthRatio * w.regularTileArea) * w.jumboTileCount = 120 := by
  sorry


end total_wall_area_l1387_138701


namespace x_convergence_l1387_138711

def x : ℕ → ℚ
  | 0 => 7
  | n + 1 => (x n ^ 2 + 8 * x n + 9) / (x n + 7)

theorem x_convergence :
  ∃ m : ℕ, m ≥ 81 ∧ m ≤ 242 ∧ 
    x m ≤ 5 + 1 / 2^15 ∧ 
    ∀ k : ℕ, k > 0 ∧ k < m → x k > 5 + 1 / 2^15 :=
by sorry

end x_convergence_l1387_138711


namespace total_shaded_area_is_one_third_l1387_138772

/-- Represents the fractional area shaded at each step of the square division process -/
def shadedAreaSequence : ℕ → ℚ
  | 0 => 5 / 16
  | n + 1 => shadedAreaSequence n + 5 / 16^(n + 2)

/-- The theorem stating that the total shaded area is 1/3 of the square -/
theorem total_shaded_area_is_one_third :
  (∑' n, shadedAreaSequence n) = 1 / 3 := by sorry

end total_shaded_area_is_one_third_l1387_138772


namespace domain_of_g_l1387_138798

-- Define the function f with domain [0,2]
def f : Set ℝ := Set.Icc 0 2

-- Define the function g(x) = f(x²)
def g (x : ℝ) : Prop := x^2 ∈ f

-- Theorem statement
theorem domain_of_g :
  {x : ℝ | g x} = Set.Icc (-Real.sqrt 2) (Real.sqrt 2) :=
sorry

end domain_of_g_l1387_138798


namespace fraction_inequality_counterexample_l1387_138726

theorem fraction_inequality_counterexample : 
  ∃ (a₁ a₂ b₁ b₂ c₁ c₂ d₁ d₂ : ℕ), 
    a₁ > 0 ∧ a₂ > 0 ∧ b₁ > 0 ∧ b₂ > 0 ∧ c₁ > 0 ∧ c₂ > 0 ∧ d₁ > 0 ∧ d₂ > 0 ∧
    (a₁ : ℚ) / b₁ < a₂ / b₂ ∧
    (c₁ : ℚ) / d₁ < c₂ / d₂ ∧
    (a₁ + c₁ : ℚ) / (b₁ + d₁) ≥ (a₂ + c₂) / (b₂ + d₂) := by
  sorry

end fraction_inequality_counterexample_l1387_138726


namespace rival_awards_l1387_138735

/-- Given Scott won 4 awards, Jessie won 3 times as many awards as Scott,
    and the rival won twice as many awards as Jessie,
    prove that the rival won 24 awards. -/
theorem rival_awards (scott_awards : ℕ) (jessie_awards : ℕ) (rival_awards : ℕ)
    (h1 : scott_awards = 4)
    (h2 : jessie_awards = 3 * scott_awards)
    (h3 : rival_awards = 2 * jessie_awards) :
  rival_awards = 24 := by
  sorry

end rival_awards_l1387_138735


namespace whiteboard_count_per_class_l1387_138794

/-- Given:
  - There are 5 classes in a building block at Oakland High.
  - Each whiteboard needs 20ml of ink for a day's use.
  - Ink costs 50 cents per ml.
  - It costs $100 to use the boards for one day.
Prove that each class uses 10 whiteboards. -/
theorem whiteboard_count_per_class (
  num_classes : ℕ)
  (ink_per_board : ℝ)
  (ink_cost_per_ml : ℝ)
  (total_daily_cost : ℝ)
  (h1 : num_classes = 5)
  (h2 : ink_per_board = 20)
  (h3 : ink_cost_per_ml = 0.5)
  (h4 : total_daily_cost = 100) :
  (total_daily_cost * num_classes) / (ink_per_board * ink_cost_per_ml) / num_classes = 10 := by
  sorry

end whiteboard_count_per_class_l1387_138794


namespace bankers_gain_calculation_l1387_138784

/-- Banker's gain calculation -/
theorem bankers_gain_calculation (P TD : ℚ) (h1 : P = 576) (h2 : TD = 96) :
  TD^2 / P = 16 := by sorry

end bankers_gain_calculation_l1387_138784


namespace great_dane_weight_l1387_138756

theorem great_dane_weight (chihuahua pitbull great_dane : ℕ) : 
  chihuahua + pitbull + great_dane = 439 →
  pitbull = 3 * chihuahua →
  great_dane = 10 + 3 * pitbull →
  great_dane = 307 := by
  sorry

end great_dane_weight_l1387_138756


namespace probability_one_correct_l1387_138764

/-- The number of options for each multiple-choice question -/
def num_options : ℕ := 4

/-- The number of questions -/
def num_questions : ℕ := 2

/-- The number of correct answers needed -/
def correct_answers : ℕ := 1

/-- The probability of getting exactly one answer correct out of two multiple-choice questions,
    each with 4 options and only one correct answer, when answers are randomly selected -/
theorem probability_one_correct :
  (num_options - 1) * num_questions / (num_options ^ num_questions) = 3 / 8 := by
  sorry

end probability_one_correct_l1387_138764


namespace specific_pentagon_area_l1387_138716

/-- Pentagon with specified side lengths and right angles -/
structure Pentagon where
  PQ : ℝ
  QR : ℝ
  RS : ℝ
  ST : ℝ
  TP : ℝ
  angle_TPQ : ℝ
  angle_PQR : ℝ

/-- The area of a pentagon with the given properties -/
def pentagon_area (p : Pentagon) : ℝ := sorry

/-- Theorem stating the area of the specific pentagon is 100 -/
theorem specific_pentagon_area :
  let p : Pentagon := {
    PQ := 8,
    QR := 2,
    RS := 13,
    ST := 13,
    TP := 8,
    angle_TPQ := 90,
    angle_PQR := 90
  }
  pentagon_area p = 100 := by sorry

end specific_pentagon_area_l1387_138716


namespace problem_solution_l1387_138796

-- Define the base 10 logarithm
noncomputable def log10 (x : ℝ) := Real.log x / Real.log 10

-- Define the problem conditions
def problem_conditions (a b : ℝ) : Prop :=
  a > 0 ∧ b > 0 ∧
  ∃ (p q r : ℕ), 
    Real.sqrt (log10 a) = p ∧
    Real.sqrt (log10 b) = q ∧
    log10 (Real.sqrt (a * b^2)) = r ∧
    p + q + r = 150

-- State the theorem
theorem problem_solution (a b : ℝ) : 
  problem_conditions a b → a^2 * b^3 = 10^443 := by
  sorry

end problem_solution_l1387_138796


namespace smallest_solution_floor_square_diff_l1387_138781

theorem smallest_solution_floor_square_diff (x : ℝ) :
  (∀ y : ℝ, y < x → ⌊y^2⌋ - ⌊y⌋^2 ≠ 19) ∧ ⌊x^2⌋ - ⌊x⌋^2 = 19 ↔ x = Real.sqrt 104 := by
  sorry

end smallest_solution_floor_square_diff_l1387_138781


namespace right_triangle_inequality_right_triangle_inequality_equality_l1387_138728

theorem right_triangle_inequality (a b c : ℝ) (h : a^2 + b^2 = c^2) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  3*a + 4*b ≤ 5*c :=
by sorry

theorem right_triangle_inequality_equality (a b c : ℝ) (h : a^2 + b^2 = c^2) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  3*a + 4*b = 5*c ↔ ∃ (k : ℝ), k > 0 ∧ a = 3*k ∧ b = 4*k ∧ c = 5*k :=
by sorry

end right_triangle_inequality_right_triangle_inequality_equality_l1387_138728


namespace distance_A_l1387_138723

-- Define the points
def A : ℝ × ℝ := (0, 11)
def B : ℝ × ℝ := (0, 15)
def C : ℝ × ℝ := (3, 9)

-- Define the line y = x
def line_y_eq_x (p : ℝ × ℝ) : Prop := p.2 = p.1

-- Define the condition that AA' and BB' intersect at C
def intersect_at_C (A' B' : ℝ × ℝ) : Prop :=
  ∃ t₁ t₂ : ℝ, 
    C = (t₁ * A'.1 + (1 - t₁) * A.1, t₁ * A'.2 + (1 - t₁) * A.2) ∧
    C = (t₂ * B'.1 + (1 - t₂) * B.1, t₂ * B'.2 + (1 - t₂) * B.2)

-- Main theorem
theorem distance_A'B'_is_2_26 :
  ∃ A' B' : ℝ × ℝ, 
    line_y_eq_x A' ∧ 
    line_y_eq_x B' ∧ 
    intersect_at_C A' B' ∧ 
    Real.sqrt ((A'.1 - B'.1)^2 + (A'.2 - B'.2)^2) = 2.26 := by
  sorry

end distance_A_l1387_138723


namespace expression_value_l1387_138752

theorem expression_value : 
  let a := 2015
  let b := 2016
  (a^3 - 3*a^2*b + 5*a*b^2 - b^3 + 4) / (a*b) = 4032 := by
sorry

end expression_value_l1387_138752


namespace alex_jane_pen_difference_l1387_138730

/-- The number of pens Alex has after n weeks, given she starts with 4 pens and triples her collection each week -/
def alexPens (n : ℕ) : ℕ := 4 * 3^(n - 1)

/-- The number of pens Jane has after a month -/
def janePens : ℕ := 50

/-- The number of weeks in a month -/
def weeksInMonth : ℕ := 4

theorem alex_jane_pen_difference :
  alexPens weeksInMonth - janePens = 58 := by sorry

end alex_jane_pen_difference_l1387_138730


namespace base_conversion_sum_l1387_138708

/-- Converts a number from given base to base 10 -/
def to_base_10 (digits : List Nat) (base : Nat) : Nat :=
  digits.foldr (fun d acc => d + base * acc) 0

/-- The main theorem -/
theorem base_conversion_sum :
  let a := to_base_10 [2, 1, 3] 8
  let b := to_base_10 [1, 2] 3
  let c := to_base_10 [2, 3, 4] 5
  let d := to_base_10 [3, 2] 4
  (a / b : Rat) + (c / d : Rat) = 31 + 9 / 14 := by
  sorry

end base_conversion_sum_l1387_138708


namespace total_price_is_1797_80_l1387_138789

/-- The price of Marion's bike in dollars -/
def marion_bike_price : ℝ := 356

/-- The price of Stephanie's bike before discount in dollars -/
def stephanie_bike_price_before_discount : ℝ := 2 * marion_bike_price

/-- The discount percentage Stephanie received -/
def stephanie_discount_percent : ℝ := 0.1

/-- The price of Patrick's bike before promotion in dollars -/
def patrick_bike_price_before_promotion : ℝ := 3 * marion_bike_price

/-- The percentage of the original price Patrick pays -/
def patrick_payment_percent : ℝ := 0.75

/-- The total price paid for all three bikes -/
def total_price : ℝ := 
  marion_bike_price + 
  stephanie_bike_price_before_discount * (1 - stephanie_discount_percent) + 
  patrick_bike_price_before_promotion * patrick_payment_percent

/-- Theorem stating that the total price paid for the three bikes is $1797.80 -/
theorem total_price_is_1797_80 : total_price = 1797.80 := by
  sorry

end total_price_is_1797_80_l1387_138789


namespace flour_to_add_l1387_138739

/-- Given a recipe requiring a total amount of flour and an amount already added,
    calculate the remaining amount of flour to be added. -/
def remaining_flour (total : ℕ) (added : ℕ) : ℕ :=
  total - added

theorem flour_to_add : remaining_flour 10 6 = 4 := by
  sorry

end flour_to_add_l1387_138739


namespace unique_sequence_sum_property_l1387_138744

-- Define the sequence type
def UniqueIntegerSequence := ℕ+ → ℕ+

-- Define the property that every positive integer occurs exactly once
def IsUniqueSequence (a : UniqueIntegerSequence) : Prop :=
  ∀ n : ℕ+, ∃! k : ℕ+, a k = n

-- State the theorem
theorem unique_sequence_sum_property (a : UniqueIntegerSequence) 
    (h : IsUniqueSequence a) : 
    ∃ ℓ m : ℕ+, 1 < ℓ ∧ ℓ < m ∧ a 1 + a m = 2 * a ℓ := by
  sorry

end unique_sequence_sum_property_l1387_138744


namespace hash_six_two_l1387_138710

-- Define the # operation
def hash (a b : ℚ) : ℚ := a + a / b

-- Theorem statement
theorem hash_six_two : hash 6 2 = 9 := by
  sorry

end hash_six_two_l1387_138710


namespace inscribed_circle_sector_ratio_l1387_138771

theorem inscribed_circle_sector_ratio :
  ∀ (R r : ℝ),
  R > 0 → r > 0 →
  R = (2 * Real.sqrt 3 + 3) * r / 3 →
  (π * r^2) / ((π * R^2) / 6) = 2 / 3 :=
by sorry

end inscribed_circle_sector_ratio_l1387_138771


namespace fayes_age_l1387_138783

/-- Represents the ages of Chad, Diana, Eduardo, and Faye --/
structure Ages where
  chad : ℕ
  diana : ℕ
  eduardo : ℕ
  faye : ℕ

/-- The age relationships between Chad, Diana, Eduardo, and Faye --/
def age_relationships (ages : Ages) : Prop :=
  ages.diana = ages.eduardo - 4 ∧
  ages.eduardo = ages.chad + 5 ∧
  ages.faye = ages.chad + 2 ∧
  ages.diana = 16

/-- Theorem stating that given the age relationships, Faye's age is 17 --/
theorem fayes_age (ages : Ages) : age_relationships ages → ages.faye = 17 := by
  sorry

end fayes_age_l1387_138783


namespace remainder_of_2685976_div_8_l1387_138741

theorem remainder_of_2685976_div_8 : 2685976 % 8 = 0 := by
  sorry

end remainder_of_2685976_div_8_l1387_138741


namespace sequence_equivalence_l1387_138745

theorem sequence_equivalence (n : ℕ+) : (2*n - 1)^2 - 1 = 4*n*(n + 1) := by
  sorry

end sequence_equivalence_l1387_138745


namespace parabola_greatest_a_l1387_138714

/-- The greatest possible value of a for a parabola with given conditions -/
theorem parabola_greatest_a (a b c : ℝ) : 
  (∃ (x y : ℝ), y = a * x^2 + b * x + c ∧ x = 3/5 ∧ y = -1/5) → -- vertex condition
  a < 0 → -- a is negative
  (∃ (k : ℤ), b + 2*c = k) → -- b + 2c is an integer
  (∀ (a' : ℝ), (∃ (b' c' : ℝ), 
    (∃ (x y : ℝ), y = a' * x^2 + b' * x + c' ∧ x = 3/5 ∧ y = -1/5) ∧
    a' < 0 ∧
    (∃ (k : ℤ), b' + 2*c' = k)) →
    a' ≤ a) →
  a = -5/6 := by
sorry

end parabola_greatest_a_l1387_138714


namespace characterize_positive_product_set_l1387_138754

def positive_product_set : Set ℤ :=
  {a : ℤ | (5 + a) * (3 - a) > 0}

theorem characterize_positive_product_set :
  positive_product_set = {-4, -3, -2, -1, 0, 1, 2} := by
  sorry

end characterize_positive_product_set_l1387_138754


namespace cube_difference_l1387_138703

theorem cube_difference (x y : ℝ) (h1 : x + y = 8) (h2 : 3 * x + y = 14) :
  x^3 - y^3 = -98 := by
  sorry

end cube_difference_l1387_138703


namespace sin_315_degrees_l1387_138767

theorem sin_315_degrees : 
  Real.sin (315 * π / 180) = -Real.sqrt 2 / 2 := by sorry

end sin_315_degrees_l1387_138767


namespace largest_power_of_five_dividing_factorial_sum_l1387_138773

def factorial (n : ℕ) : ℕ := Nat.factorial n

def divides_exactly (x n y : ℕ) : Prop :=
  (x^n ∣ y) ∧ ¬(x^(n+1) ∣ y)

theorem largest_power_of_five_dividing_factorial_sum :
  ∃ (n : ℕ), n = 26 ∧ divides_exactly 5 n (factorial 98 + factorial 99 + factorial 100) ∧
  ∀ (m : ℕ), m > n → ¬(divides_exactly 5 m (factorial 98 + factorial 99 + factorial 100)) :=
sorry

end largest_power_of_five_dividing_factorial_sum_l1387_138773


namespace henry_payment_l1387_138720

/-- The payment Henry receives for painting a bike -/
def paint_payment : ℕ := 5

/-- The additional payment Henry receives for selling a bike compared to painting it -/
def sell_additional_payment : ℕ := 8

/-- The number of bikes Henry sells and paints -/
def num_bikes : ℕ := 8

/-- The total payment Henry receives for selling and painting the given number of bikes -/
def total_payment (paint : ℕ) (sell_additional : ℕ) (bikes : ℕ) : ℕ :=
  bikes * (paint + sell_additional + paint)

theorem henry_payment :
  total_payment paint_payment sell_additional_payment num_bikes = 144 := by
sorry

end henry_payment_l1387_138720


namespace polynomial_real_root_l1387_138727

/-- The polynomial in question -/
def P (a x : ℝ) : ℝ := x^4 + a*x^3 - 2*x^2 + a*x + 2

/-- The theorem stating the condition for the polynomial to have at least one real root -/
theorem polynomial_real_root (a : ℝ) :
  (∃ x : ℝ, P a x = 0) ↔ a ≤ 0 := by sorry

end polynomial_real_root_l1387_138727


namespace complex_number_problem_l1387_138718

theorem complex_number_problem (a : ℝ) : 
  (∃ (z₁ : ℂ), z₁ = a + (2 / (1 - Complex.I)) ∧ z₁.re < 0 ∧ z₁.im > 0) ∧ 
  Complex.abs (a - Complex.I) = 2 → 
  a = -Real.sqrt 3 := by
sorry

end complex_number_problem_l1387_138718


namespace isosceles_trapezoid_area_theorem_l1387_138724

/-- An isosceles trapezoid with given midline length and height -/
structure IsoscelesTrapezoid where
  midline : ℝ
  height : ℝ

/-- The area of an isosceles trapezoid -/
def area (t : IsoscelesTrapezoid) : ℝ := t.midline * t.height

/-- Theorem: The area of an isosceles trapezoid with midline 15 and height 3 is 45 -/
theorem isosceles_trapezoid_area_theorem :
  ∀ t : IsoscelesTrapezoid, t.midline = 15 ∧ t.height = 3 → area t = 45 := by
  sorry

end isosceles_trapezoid_area_theorem_l1387_138724


namespace root_transformation_l1387_138743

theorem root_transformation (r₁ r₂ r₃ : ℂ) : 
  (r₁^3 - 5*r₁^2 + 10 = 0) ∧ 
  (r₂^3 - 5*r₂^2 + 10 = 0) ∧ 
  (r₃^3 - 5*r₃^2 + 10 = 0) → 
  ∀ x : ℂ, x^3 - 15*x^2 + 270 = (x - 3*r₁) * (x - 3*r₂) * (x - 3*r₃) := by
sorry

end root_transformation_l1387_138743


namespace product_103_97_l1387_138777

theorem product_103_97 : 103 * 97 = 9991 := by
  sorry

end product_103_97_l1387_138777


namespace lindseys_remaining_money_l1387_138740

/-- Calculates Lindsey's remaining money after saving and spending --/
theorem lindseys_remaining_money (september : ℝ) (october : ℝ) (november : ℝ) 
  (h_sep : september = 50)
  (h_oct : october = 37)
  (h_nov : november = 11)
  (h_dec : december = november * 1.1)
  (h_mom_bonus : total_savings > 75 → mom_bonus = total_savings * 0.2)
  (h_spending : spending = (total_savings + mom_bonus) * 0.75)
  : remaining_money = 33.03 :=
by
  sorry

where
  december : ℝ := november * 1.1
  total_savings : ℝ := september + october + november + december
  mom_bonus : ℝ := if total_savings > 75 then total_savings * 0.2 else 0
  spending : ℝ := (total_savings + mom_bonus) * 0.75
  remaining_money : ℝ := total_savings + mom_bonus - spending

end lindseys_remaining_money_l1387_138740


namespace complex_number_in_third_quadrant_l1387_138790

def complex_number : ℂ := Complex.I * (1 + Complex.I)

theorem complex_number_in_third_quadrant :
  Real.sign (complex_number.re) = -1 ∧ Real.sign (complex_number.im) = -1 :=
by
  sorry

end complex_number_in_third_quadrant_l1387_138790


namespace exists_composite_in_sequence_l1387_138725

-- Define the sequence type
def RecurrenceSequence := ℕ → ℕ

-- Define the recurrence relation
def SatisfiesRecurrence (a : RecurrenceSequence) : Prop :=
  ∀ n : ℕ, (a (n + 1) = 2 * a n + 1) ∨ (a (n + 1) = 2 * a n - 1)

-- Define a non-constant sequence
def NonConstant (a : RecurrenceSequence) : Prop :=
  ∃ m n : ℕ, a m ≠ a n

-- Define a positive sequence
def Positive (a : RecurrenceSequence) : Prop :=
  ∀ n : ℕ, a n > 0

-- Define a composite number
def Composite (n : ℕ) : Prop :=
  n > 1 ∧ ∃ k : ℕ, 1 < k ∧ k < n ∧ n % k = 0

-- The main theorem
theorem exists_composite_in_sequence (a : RecurrenceSequence)
  (h1 : SatisfiesRecurrence a)
  (h2 : NonConstant a)
  (h3 : Positive a) :
  ∃ n : ℕ, Composite (a n) :=
  sorry

end exists_composite_in_sequence_l1387_138725


namespace cuboid_volume_l1387_138709

/-- 
Theorem: Volume of a specific cuboid

Given a cuboid with the following properties:
1. The length and width are equal.
2. The length is 2 cm more than the height.
3. When the height is increased by 2 cm (making it equal to the length and width),
   the surface area increases by 56 square centimeters.

This theorem proves that the volume of the original cuboid is 245 cubic centimeters.
-/
theorem cuboid_volume (l w h : ℝ) : 
  l = w → -- length equals width
  l = h + 2 → -- length is 2 more than height
  6 * l^2 - 2 * (l^2 - (l-2)^2) = 56 → -- surface area increase condition
  l * w * h = 245 :=
by sorry

end cuboid_volume_l1387_138709


namespace find_m_value_l1387_138797

theorem find_m_value (m : ℚ) : 
  (∃ (x y : ℚ), m * x - y = 4 ∧ x = 4 ∧ y = 3) → m = 7/4 := by
  sorry

end find_m_value_l1387_138797


namespace min_additional_coins_l1387_138768

/-- The number of friends Alex has -/
def num_friends : ℕ := 15

/-- The initial number of coins Alex has -/
def initial_coins : ℕ := 85

/-- The sum of the first n natural numbers -/
def sum_first_n (n : ℕ) : ℕ := n * (n + 1) / 2

/-- The theorem stating the minimum number of additional coins needed -/
theorem min_additional_coins : 
  sum_first_n num_friends - initial_coins = 35 := by sorry

end min_additional_coins_l1387_138768


namespace regression_line_intercept_l1387_138702

theorem regression_line_intercept (x_bar m : ℝ) (y_bar : ℝ) :
  x_bar = m → y_bar = 6 → y_bar = 2 * x_bar + m → m = 2 := by sorry

end regression_line_intercept_l1387_138702


namespace some_number_value_l1387_138746

theorem some_number_value (some_number : ℝ) : 
  (3.242 * some_number) / 100 = 0.032420000000000004 → some_number = 1 := by
  sorry

end some_number_value_l1387_138746


namespace equilateral_triangle_to_square_l1387_138734

/-- Given an equilateral triangle with area 121√3 cm², prove that decreasing each side by 6 cm
    and transforming it into a square results in a square with area 256 cm². -/
theorem equilateral_triangle_to_square (s : ℝ) : 
  (s^2 * Real.sqrt 3 / 4 = 121 * Real.sqrt 3) →
  ((s - 6)^2 = 256) := by
  sorry

end equilateral_triangle_to_square_l1387_138734


namespace larger_number_is_23_l1387_138759

theorem larger_number_is_23 (x y : ℝ) (h1 : x - y = 6) (h2 : x + y = 40) :
  x = 23 := by
  sorry

end larger_number_is_23_l1387_138759


namespace sum_of_coefficients_l1387_138782

theorem sum_of_coefficients (b₅ b₄ b₃ b₂ b₁ b₀ : ℝ) :
  (∀ x : ℝ, (2*x + 3)^5 = b₅*x^5 + b₄*x^4 + b₃*x^3 + b₂*x^2 + b₁*x + b₀) →
  b₅ + b₄ + b₃ + b₂ + b₁ + b₀ = 3125 := by
sorry

end sum_of_coefficients_l1387_138782


namespace odd_function_unique_m_l1387_138705

/-- A function f is odd if f(-x) = -f(x) for all x in its domain -/
def IsOddFunction (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

/-- The given function f(x) parameterized by m -/
def f (m : ℝ) (x : ℝ) : ℝ :=
  (m^2 - 1) * x^2 + (m - 2) * x + (m^2 - 7*m + 6)

/-- Theorem stating that m = 6 is the only value that makes f an odd function -/
theorem odd_function_unique_m :
  ∃! m : ℝ, IsOddFunction (f m) ∧ m = 6 :=
sorry

end odd_function_unique_m_l1387_138705


namespace arithmetic_sequence_constant_ratio_l1387_138755

/-- An arithmetic sequence -/
def ArithmeticSequence (a : ℕ → ℝ) : Prop :=
  ∃ (a₁ d : ℝ), ∀ n, a n = a₁ + (n - 1) * d

/-- The ratio of a_n to a_(2n) is constant -/
def ConstantRatio (a : ℕ → ℝ) : Prop :=
  ∃ c, ∀ n, a n ≠ 0 → a (2*n) ≠ 0 → a n / a (2*n) = c

theorem arithmetic_sequence_constant_ratio (a : ℕ → ℝ) 
    (h1 : ArithmeticSequence a) (h2 : ConstantRatio a) :
    ∃ c, (c = 1 ∨ c = 1/2) ∧ ∀ n, a n ≠ 0 → a (2*n) ≠ 0 → a n / a (2*n) = c :=
  sorry

end arithmetic_sequence_constant_ratio_l1387_138755


namespace geometric_series_common_ratio_l1387_138706

theorem geometric_series_common_ratio : 
  let a₁ : ℚ := 5/6
  let a₂ : ℚ := -4/9
  let a₃ : ℚ := 32/135
  let r : ℚ := a₂ / a₁
  (∀ n : ℕ, n ≥ 2 → (a₁ * r^(n-1) : ℚ) = if n = 2 then a₂ else if n = 3 then a₃ else 0) →
  r = -8/15 :=
by sorry

end geometric_series_common_ratio_l1387_138706


namespace eulers_formula_l1387_138732

/-- A convex polyhedron with vertices, edges, and faces. -/
structure ConvexPolyhedron where
  V : ℕ  -- number of vertices
  E : ℕ  -- number of edges
  F : ℕ  -- number of faces

/-- Euler's formula for convex polyhedra -/
theorem eulers_formula (P : ConvexPolyhedron) : P.V - P.E + P.F = 2 := by
  sorry

end eulers_formula_l1387_138732


namespace school_children_count_l1387_138722

/-- Proves that the number of children in the school is 320 given the banana distribution conditions --/
theorem school_children_count : ∃ (C : ℕ) (B : ℕ), 
  B = 2 * C ∧                   -- Total bananas if each child gets 2
  B = 4 * (C - 160) ∧           -- Total bananas distributed to present children
  C = 320                       -- The number we want to prove
  := by sorry

end school_children_count_l1387_138722


namespace remainder_property_l1387_138762

theorem remainder_property (x : ℕ) (h : x > 0) :
  (100 % x = 4) → ((100 + x) % x = 4) := by
  sorry

end remainder_property_l1387_138762


namespace incorrect_calculation_l1387_138713

theorem incorrect_calculation : 
  ((-11) + (-17) = -28) ∧ 
  ((-3/4 : ℚ) + (1/2 : ℚ) = -1/4) ∧ 
  ((-9) + 9 = 0) ∧ 
  ((5/8 : ℚ) + (-7/12 : ℚ) ≠ -1/24) := by
  sorry

end incorrect_calculation_l1387_138713


namespace sum_F_equals_250_l1387_138753

-- Define the function F
def F (n : ℕ) : ℕ := sorry

-- Define the sum of F from 1 to 50
def sum_F : ℕ := (List.range 50).map (fun i => F (i + 1)) |>.sum

-- Theorem statement
theorem sum_F_equals_250 : sum_F = 250 := by sorry

end sum_F_equals_250_l1387_138753


namespace quadratic_vertex_l1387_138721

/-- The quadratic function f(x) = 2x^2 - 4x + 5 has its vertex at (1, 3) -/
theorem quadratic_vertex (x : ℝ) :
  let f : ℝ → ℝ := λ x ↦ 2 * x^2 - 4 * x + 5
  (∀ x, f x ≥ f 1) ∧ f 1 = 3 :=
by sorry

end quadratic_vertex_l1387_138721


namespace total_rainfall_sum_l1387_138736

/-- The rainfall recorded on Monday in centimeters -/
def monday_rainfall : ℚ := 0.16666666666666666

/-- The rainfall recorded on Tuesday in centimeters -/
def tuesday_rainfall : ℚ := 0.4166666666666667

/-- The rainfall recorded on Wednesday in centimeters -/
def wednesday_rainfall : ℚ := 0.08333333333333333

/-- The total rainfall recorded over the three days -/
def total_rainfall : ℚ := monday_rainfall + tuesday_rainfall + wednesday_rainfall

/-- Theorem stating that the total rainfall equals 0.6666666666666667 cm -/
theorem total_rainfall_sum :
  total_rainfall = 0.6666666666666667 := by sorry

end total_rainfall_sum_l1387_138736


namespace sum_difference_equals_product_l1387_138700

-- Define the sequence
def seq : ℕ → ℕ
  | 0 => 0
  | n + 1 => n / 2 + 1

-- Define f(n) as the sum of the first n terms of the sequence
def f (n : ℕ) : ℕ := (List.range n).map seq |>.sum

-- Theorem statement
theorem sum_difference_equals_product {s t : ℕ} (hs : s > 0) (ht : t > 0) (hst : s > t) :
  f (s + t) - f (s - t) = s * t := by
  sorry

end sum_difference_equals_product_l1387_138700


namespace modulus_of_z_l1387_138704

-- Define the complex number z
def z : ℂ := (3 - Complex.I)^2 * Complex.I

-- State the theorem
theorem modulus_of_z : Complex.abs z = 10 := by sorry

end modulus_of_z_l1387_138704


namespace sum_of_four_consecutive_integers_divisible_by_two_l1387_138750

theorem sum_of_four_consecutive_integers_divisible_by_two (n : ℤ) :
  ∃ k : ℤ, (n - 1) + n + (n + 1) + (n + 2) = 2 * k := by
  sorry

end sum_of_four_consecutive_integers_divisible_by_two_l1387_138750


namespace derivative_of_one_minus_cosine_l1387_138717

theorem derivative_of_one_minus_cosine (α : ℝ) : 
  let f : ℝ → ℝ := λ x ↦ 1 - Real.cos x
  (deriv f) α = Real.sin α := by
sorry

end derivative_of_one_minus_cosine_l1387_138717


namespace special_triangle_bisecting_lines_angle_l1387_138795

/-- Triangle with specific side lengths -/
structure SpecialTriangle where
  a : ℝ
  b : ℝ
  c : ℝ
  a_eq : a = 13
  b_eq : b = 14
  c_eq : c = 15

/-- A line that bisects both perimeter and area of the triangle -/
structure BisectingLine (t : SpecialTriangle) where
  bisects_perimeter : Bool
  bisects_area : Bool

/-- The acute angle between two bisecting lines -/
def acute_angle (t : SpecialTriangle) (l1 l2 : BisectingLine t) : ℝ := sorry

theorem special_triangle_bisecting_lines_angle 
  (t : SpecialTriangle) 
  (l1 l2 : BisectingLine t) 
  (h_unique : ∀ (l : BisectingLine t), l = l1 ∨ l = l2) :
  Real.tan (acute_angle t l1 l2) = Real.sqrt 6 / 12 := by sorry

end special_triangle_bisecting_lines_angle_l1387_138795


namespace tangent_fraction_equals_one_l1387_138731

theorem tangent_fraction_equals_one (θ : Real) (h : Real.tan θ = -2 * Real.sqrt 2) :
  (2 * (Real.cos (θ / 2))^2 - Real.sin θ - 1) / (Real.sqrt 2 * Real.sin (θ + π/4)) = 1 := by
  sorry

end tangent_fraction_equals_one_l1387_138731


namespace quiz_probability_l1387_138787

theorem quiz_probability : 
  let n_questions : ℕ := 5
  let n_choices : ℕ := 6
  let p_correct : ℚ := 1 / n_choices
  let p_incorrect : ℚ := 1 - p_correct
  1 - p_incorrect ^ n_questions = 4651 / 7776 :=
by sorry

end quiz_probability_l1387_138787


namespace range_of_a_l1387_138779

-- Define the sets A and B
def A : Set ℝ := {x | x^2 - 4*x + 3 < 0}
def B (a : ℝ) : Set ℝ := {x | 2*x + a ≤ 0 ∧ x^2 - 2*(a + 7)*x + 5 ≤ 0}

-- State the theorem
theorem range_of_a (a : ℝ) : A ⊆ B a ↔ -4 ≤ a ∧ a ≤ -2 := by
  sorry

end range_of_a_l1387_138779


namespace symmetric_point_l1387_138799

/-- Given two points A and B in a plane, find the symmetric point A' of A with respect to B. -/
theorem symmetric_point (A B : ℝ × ℝ) (A' : ℝ × ℝ) : 
  A = (2, 1) → B = (-3, 7) → 
  (B.1 = (A.1 + A'.1) / 2 ∧ B.2 = (A.2 + A'.2) / 2) →
  A' = (-8, 13) := by sorry

end symmetric_point_l1387_138799


namespace quadratic_less_than_sqrt_l1387_138758

theorem quadratic_less_than_sqrt (x : ℝ) :
  x^2 - 3*x + 2 < Real.sqrt (x + 4) ↔ 1 < x ∧ x < 2 :=
by sorry

end quadratic_less_than_sqrt_l1387_138758


namespace choose_two_from_five_l1387_138748

theorem choose_two_from_five (n : ℕ) (k : ℕ) : n = 5 ∧ k = 2 → Nat.choose n k = 10 := by
  sorry

end choose_two_from_five_l1387_138748


namespace intersection_of_A_and_B_l1387_138770

def A : Set ℝ := {x | x > 0}
def B : Set ℝ := {-1, 0, 1, 2, 3}

theorem intersection_of_A_and_B : A ∩ B = {1, 2, 3} := by
  sorry

end intersection_of_A_and_B_l1387_138770


namespace solve_linear_equation_l1387_138774

theorem solve_linear_equation :
  ∀ x : ℝ, 7 - 2 * x = 15 → x = -4 := by
sorry

end solve_linear_equation_l1387_138774


namespace complement_A_union_B_equals_target_l1387_138791

-- Define the universal set U as ℝ
def U : Set ℝ := Set.univ

-- Define set A
def A : Set ℝ := {x : ℝ | 0 ≤ x ∧ x ≤ 2}

-- Define set B
def B : Set ℝ := {y : ℝ | 1 ≤ y ∧ y ≤ 3}

-- State the theorem
theorem complement_A_union_B_equals_target :
  (Set.compl A) ∪ B = {x : ℝ | x < 0 ∨ x ≥ 1} := by sorry

end complement_A_union_B_equals_target_l1387_138791


namespace sum_and_ratio_problem_l1387_138778

theorem sum_and_ratio_problem (x y : ℝ) 
  (sum_eq : x + y = 500)
  (ratio_eq : x / y = 4 / 5) :
  y - x = 500 / 9 := by
  sorry

end sum_and_ratio_problem_l1387_138778


namespace no_solution_gcd_lcm_sum_l1387_138760

theorem no_solution_gcd_lcm_sum (x y : ℕ) : 
  Nat.gcd x y + Nat.lcm x y + x + y ≠ 2019 := by
  sorry

end no_solution_gcd_lcm_sum_l1387_138760


namespace simplify_expression_l1387_138765

theorem simplify_expression (x : ℝ) : (3 * x + 20) + (50 * x + 25) = 53 * x + 45 := by
  sorry

end simplify_expression_l1387_138765


namespace x_fourth_coefficient_l1387_138763

def binomial_coefficient (n k : ℕ) : ℤ :=
  if k > n then 0
  else (Nat.factorial n) / (Nat.factorial k * Nat.factorial (n - k))

def expansion_coefficient (n r : ℕ) : ℤ :=
  (-1)^r * binomial_coefficient n r

theorem x_fourth_coefficient :
  expansion_coefficient 8 3 = -56 :=
by sorry

end x_fourth_coefficient_l1387_138763


namespace quadratic_two_distinct_roots_l1387_138749

/-- The quadratic equation x^2 - 2x - 6 = 0 has two distinct real roots -/
theorem quadratic_two_distinct_roots :
  ∃ (x₁ x₂ : ℝ), x₁ ≠ x₂ ∧ 
  (x₁^2 - 2*x₁ - 6 = 0) ∧ 
  (x₂^2 - 2*x₂ - 6 = 0) :=
by
  sorry


end quadratic_two_distinct_roots_l1387_138749


namespace edward_booth_tickets_l1387_138769

/-- The number of tickets Edward spent at the 'dunk a clown' booth -/
def tickets_spent_at_booth (total_tickets : ℕ) (cost_per_ride : ℕ) (possible_rides : ℕ) : ℕ :=
  total_tickets - (cost_per_ride * possible_rides)

/-- Proof that Edward spent 23 tickets at the 'dunk a clown' booth -/
theorem edward_booth_tickets : 
  tickets_spent_at_booth 79 7 8 = 23 := by
  sorry

end edward_booth_tickets_l1387_138769


namespace binomial_26_6_l1387_138733

theorem binomial_26_6 (h1 : Nat.choose 24 5 = 42504) 
                      (h2 : Nat.choose 24 6 = 134596) 
                      (h3 : Nat.choose 23 5 = 33649) : 
  Nat.choose 26 6 = 230230 := by
  sorry

end binomial_26_6_l1387_138733


namespace equal_selection_probability_l1387_138761

/-- Represents a sampling method -/
structure SamplingMethod where
  -- Add necessary fields here
  reasonable : Bool

/-- Represents a sample from a population -/
structure Sample where
  size : ℕ
  method : SamplingMethod

/-- Represents the probability of an individual being selected in a sample -/
def selectionProbability (s : Sample) (individual : ℕ) : ℝ :=
  -- Definition would go here
  sorry

theorem equal_selection_probability 
  (s1 s2 : Sample) 
  (h1 : s1.size = s2.size) 
  (h2 : s1.method.reasonable) 
  (h3 : s2.method.reasonable) 
  (individual : ℕ) : 
  selectionProbability s1 individual = selectionProbability s2 individual :=
sorry

end equal_selection_probability_l1387_138761


namespace x_y_power_product_l1387_138742

theorem x_y_power_product (x y : ℚ) (hx : x = 5/6) (hy : y = 6/5) :
  (1/3) * x^8 * y^9 = 2/5 := by
  sorry

end x_y_power_product_l1387_138742


namespace prob_other_side_red_is_two_thirds_l1387_138785

/-- Represents a card with two sides --/
structure Card where
  side1 : Bool  -- True for red, False for black
  side2 : Bool

/-- The set of all cards in the box --/
def box : Finset Card := sorry

/-- The total number of cards in the box --/
def total_cards : Nat := 8

/-- The number of cards that are black on both sides --/
def black_both_sides : Nat := 4

/-- The number of cards that are black on one side and red on the other --/
def black_red : Nat := 2

/-- The number of cards that are red on both sides --/
def red_both_sides : Nat := 2

/-- Axiom: The box contains the correct number of each type of card --/
axiom box_composition :
  (box.filter (fun c => !c.side1 ∧ !c.side2)).card = black_both_sides ∧
  (box.filter (fun c => (c.side1 ∧ !c.side2) ∨ (!c.side1 ∧ c.side2))).card = black_red ∧
  (box.filter (fun c => c.side1 ∧ c.side2)).card = red_both_sides

/-- Axiom: The total number of cards is correct --/
axiom total_cards_correct : box.card = total_cards

/-- The probability of selecting a card with a red side, given that one side is observed to be red --/
def prob_other_side_red (observed_red : Bool) : ℚ := sorry

/-- Theorem: The probability that the other side is red, given that the observed side is red, is 2/3 --/
theorem prob_other_side_red_is_two_thirds (observed_red : Bool) :
  observed_red → prob_other_side_red observed_red = 2/3 := by sorry

end prob_other_side_red_is_two_thirds_l1387_138785


namespace last_four_digits_5_pow_2011_l1387_138751

def last_four_digits (n : ℕ) : ℕ := n % 10000

theorem last_four_digits_5_pow_2011 :
  last_four_digits (5^2011) = 8125 :=
by
  sorry

end last_four_digits_5_pow_2011_l1387_138751


namespace runner_lead_l1387_138793

/-- Represents the relative speeds of runners in a race. -/
structure RunnerSpeeds where
  a : ℝ
  b : ℝ
  c : ℝ

/-- The race setup with given conditions. -/
def raceSetup (s : RunnerSpeeds) : Prop :=
  s.b = (5/6) * s.a ∧ s.c = (3/4) * s.a

/-- The theorem statement. -/
theorem runner_lead (s : RunnerSpeeds) (h : raceSetup s) :
  150 - (s.c * (150 / s.a)) = 37.5 := by
  sorry

#check runner_lead

end runner_lead_l1387_138793


namespace problem_solution_l1387_138747

theorem problem_solution (x y : Real) 
  (h1 : x + Real.cos y = 3009)
  (h2 : x + 3009 * Real.sin y = 3010)
  (h3 : 0 ≤ y ∧ y ≤ Real.pi) :
  x + y = 3009 + Real.pi / 2 := by
  sorry

end problem_solution_l1387_138747


namespace bowling_ball_volume_l1387_138712

/-- The volume of a sphere with diameter 24 cm minus the volume of three cylindrical holes
    (with depths 6 cm and diameters 1.5 cm, 2.5 cm, and 3 cm respectively) is equal to 2239.5π cubic cm. -/
theorem bowling_ball_volume :
  let sphere_diameter : ℝ := 24
  let hole_depth : ℝ := 6
  let hole_diameter1 : ℝ := 1.5
  let hole_diameter2 : ℝ := 2.5
  let hole_diameter3 : ℝ := 3
  let sphere_volume := (4 / 3) * π * (sphere_diameter / 2) ^ 3
  let hole_volume1 := π * (hole_diameter1 / 2) ^ 2 * hole_depth
  let hole_volume2 := π * (hole_diameter2 / 2) ^ 2 * hole_depth
  let hole_volume3 := π * (hole_diameter3 / 2) ^ 2 * hole_depth
  let remaining_volume := sphere_volume - (hole_volume1 + hole_volume2 + hole_volume3)
  remaining_volume = 2239.5 * π :=
by sorry

end bowling_ball_volume_l1387_138712


namespace A_inter_B_eq_open_interval_l1387_138788

-- Define the sets A and B
def A : Set ℝ := {x | x^2 - 2*x < 0}
def B : Set ℝ := {x | -Real.sqrt 3 < x ∧ x < Real.sqrt 3}

-- State the theorem
theorem A_inter_B_eq_open_interval : A ∩ B = {x | 0 < x ∧ x < Real.sqrt 3} := by sorry

end A_inter_B_eq_open_interval_l1387_138788


namespace f_inequality_l1387_138707

-- Define the function f
variable (f : ℝ → ℝ)

-- State the condition f'(x) > f(x) for all x ∈ ℝ
variable (h : ∀ x : ℝ, (deriv f) x > f x)

-- Theorem statement
theorem f_inequality : f 2 > Real.exp 2 * f 0 := by
  sorry

end f_inequality_l1387_138707


namespace ab_value_l1387_138715

theorem ab_value (a b : ℤ) (h1 : |a| = 7) (h2 : b = 5) (h3 : a + b < 0) : a * b = -35 := by
  sorry

end ab_value_l1387_138715


namespace smallest_n_satisfying_inequality_l1387_138786

theorem smallest_n_satisfying_inequality : 
  ∀ n : ℕ, n > 0 → (1 / n - 1 / (n + 1) < 1 / 15) ↔ n ≥ 4 :=
by sorry

end smallest_n_satisfying_inequality_l1387_138786


namespace red_balls_estimate_l1387_138737

/-- Represents a bag of balls -/
structure Bag where
  total : ℕ
  redProb : ℝ

/-- Calculates the expected number of red balls in the bag -/
def expectedRedBalls (b : Bag) : ℝ :=
  b.total * b.redProb

theorem red_balls_estimate (b : Bag) 
  (h1 : b.total = 20)
  (h2 : b.redProb = 0.25) : 
  expectedRedBalls b = 5 := by
  sorry

end red_balls_estimate_l1387_138737


namespace sixDigitPermutationsCount_l1387_138776

/-- The number of ways to choose k items from n items -/
def choose (n k : ℕ) : ℕ := sorry

/-- The factorial of a natural number -/
def factorial (n : ℕ) : ℕ := sorry

/-- The number of permutations of n elements -/
def permutations (n : ℕ) : ℕ := factorial n

/-- The number of ways to arrange k objects in n positions -/
def arrangements (n k : ℕ) : ℕ := (permutations n) / (factorial (n - k))

/-- The number of 6-digit permutations using x, y, and z with given conditions -/
def sixDigitPermutations : ℕ :=
  let xTwice := choose 6 2 * arrangements 4 2  -- x appears twice
  let xThrice := choose 6 3 * arrangements 3 1 -- x appears thrice
  let yOnce := choose 4 1                      -- y appears once
  let yThrice := choose 4 3                    -- y appears thrice
  let zTwice := 1                              -- z appears twice (only one way)
  (xTwice + xThrice) * (yOnce + yThrice) * zTwice

theorem sixDigitPermutationsCount : sixDigitPermutations = 60 := by
  sorry

end sixDigitPermutationsCount_l1387_138776


namespace pet_shop_dogs_l1387_138780

theorem pet_shop_dogs (total_dogs_bunnies : ℕ) (ratio_dogs : ℕ) (ratio_cats : ℕ) (ratio_bunnies : ℕ) 
  (h1 : total_dogs_bunnies = 330)
  (h2 : ratio_dogs = 7)
  (h3 : ratio_cats = 7)
  (h4 : ratio_bunnies = 8) :
  (ratio_dogs * total_dogs_bunnies) / (ratio_dogs + ratio_bunnies) = 154 := by
  sorry

#check pet_shop_dogs

end pet_shop_dogs_l1387_138780
