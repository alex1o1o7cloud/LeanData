import Mathlib

namespace equation_solution_l1684_168468

theorem equation_solution : 
  ∃! x : ℚ, (x - 30) / 3 = (3 * x + 10) / 8 - 2 ∧ x = -222 := by sorry

end equation_solution_l1684_168468


namespace geometric_sequence_sum_l1684_168458

/-- A geometric sequence with first term 3 and specific arithmetic property -/
def GeometricSequence (a : ℕ → ℝ) : Prop :=
  (∃ r : ℝ, ∀ n : ℕ, a (n + 1) = a n * r) ∧
  a 1 = 3 ∧
  ∃ d : ℝ, 2 * a 2 = 4 * a 1 + d ∧ a 3 = 2 * a 2 + d

theorem geometric_sequence_sum (a : ℕ → ℝ) (h : GeometricSequence a) :
  a 3 + a 5 = 60 := by
  sorry

end geometric_sequence_sum_l1684_168458


namespace williams_children_probability_l1684_168461

theorem williams_children_probability :
  let n : ℕ := 8  -- number of children
  let p : ℚ := 1/2  -- probability of each child being a boy (or girl)
  let total_outcomes : ℕ := 2^n  -- total number of possible gender combinations
  let balanced_outcomes : ℕ := n.choose (n/2)  -- number of combinations with equal boys and girls
  
  (total_outcomes - balanced_outcomes : ℚ) / total_outcomes = 93/128 :=
by sorry

end williams_children_probability_l1684_168461


namespace domain_of_f_l1684_168494

noncomputable def f (x : ℝ) : ℝ := Real.log (Real.tan x - 1) + Real.sqrt (9 - x^2)

theorem domain_of_f :
  {x : ℝ | ∃ y, f x = y} = {x | -3*π/4 < x ∧ x < -π/2} ∪ {x | π/4 < x ∧ x < π/2} :=
by sorry

end domain_of_f_l1684_168494


namespace eagles_per_section_l1684_168474

theorem eagles_per_section 
  (total_eagles : ℕ) 
  (total_sections : ℕ) 
  (h1 : total_eagles = 18) 
  (h2 : total_sections = 3) 
  (h3 : total_eagles % total_sections = 0) : 
  total_eagles / total_sections = 6 := by
  sorry

end eagles_per_section_l1684_168474


namespace vector_inequality_l1684_168484

/-- Given vectors u, v, and w in ℝ², prove that w ≠ u - 3v -/
theorem vector_inequality (u v w : ℝ × ℝ) 
  (hu : u = (3, -6)) 
  (hv : v = (4, 2)) 
  (hw : w = (-12, -6)) : 
  w ≠ u - 3 • v := by sorry

end vector_inequality_l1684_168484


namespace largest_prime_divisor_of_factorial_sum_l1684_168488

theorem largest_prime_divisor_of_factorial_sum : 
  ∃ p : ℕ, Nat.Prime p ∧ p ∣ (Nat.factorial 13 + Nat.factorial 14) ∧ 
  ∀ q : ℕ, Nat.Prime q → q ∣ (Nat.factorial 13 + Nat.factorial 14) → q ≤ p :=
by sorry

end largest_prime_divisor_of_factorial_sum_l1684_168488


namespace cars_with_both_features_l1684_168457

/-- Represents the car lot scenario -/
structure CarLot where
  total : Nat
  with_airbag : Nat
  with_power_windows : Nat
  with_neither : Nat

/-- Theorem stating the number of cars with both air-bag and power windows -/
theorem cars_with_both_features (lot : CarLot) 
  (h1 : lot.total = 65)
  (h2 : lot.with_airbag = 45)
  (h3 : lot.with_power_windows = 30)
  (h4 : lot.with_neither = 2) :
  lot.with_airbag + lot.with_power_windows - (lot.total - lot.with_neither) = 12 := by
  sorry

#check cars_with_both_features

end cars_with_both_features_l1684_168457


namespace investment_problem_l1684_168412

/-- Compound interest calculation -/
def compound_interest (principal : ℝ) (rate : ℝ) (time : ℕ) : ℝ :=
  principal * (1 + rate) ^ time

/-- The problem statement -/
theorem investment_problem :
  let principal : ℝ := 3000
  let rate : ℝ := 0.1
  let time : ℕ := 2
  compound_interest principal rate time = 3630.0000000000005 := by
  sorry

end investment_problem_l1684_168412


namespace equal_height_locus_is_circle_l1684_168439

/-- Two flagpoles in a plane -/
structure Flagpoles where
  h : ℝ  -- height of first flagpole
  k : ℝ  -- height of second flagpole
  a : ℝ  -- half the distance between flagpoles

/-- A point in the plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- The locus of points from which the flagpoles appear equally tall -/
def equalHeightLocus (f : Flagpoles) : Set Point := {p : Point | ∃ (t : ℝ), 
  p.x^2 + p.y^2 = t^2 ∧ 
  (p.x + f.a)^2 + p.y^2 = (t * f.k / f.h)^2 ∧
  (p.x - f.a)^2 + p.y^2 = (t * f.k / f.h)^2}

/-- The circle with diameter AB -/
def circleAB (f : Flagpoles) : Set Point := {p : Point | 
  (p.x + f.a * (f.k - f.h) / (f.k + f.h))^2 + p.y^2 = 
  (2 * f.a * f.h * f.k / (f.k + f.h))^2}

theorem equal_height_locus_is_circle (f : Flagpoles) : 
  equalHeightLocus f = circleAB f := by sorry

end equal_height_locus_is_circle_l1684_168439


namespace even_odd_sum_relation_l1684_168482

theorem even_odd_sum_relation (n : ℕ) : 
  (n * (n + 1) = 4970) → (n^2 = 4900) := by
  sorry

end even_odd_sum_relation_l1684_168482


namespace equation_solution_l1684_168416

theorem equation_solution (x : ℝ) : 
  x ≠ (1 / 3) → x ≠ -3 → 
  ((3 * x + 2) / (3 * x^2 + 8 * x - 3) = (3 * x) / (3 * x - 1)) ↔ 
  (x = -1 + Real.sqrt 15 / 3 ∨ x = -1 - Real.sqrt 15 / 3) :=
by sorry

end equation_solution_l1684_168416


namespace product_of_roots_l1684_168495

theorem product_of_roots (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (hdistinct : x ≠ y)
  (h : x + 4 / x = y + 4 / y) : x * y = 4 := by
  sorry

end product_of_roots_l1684_168495


namespace three_flower_purchase_options_l1684_168441

/-- Represents a flower purchase option -/
structure FlowerPurchase where
  carnations : Nat
  lilies : Nat

/-- The cost of a single carnation in yuan -/
def carnationCost : Nat := 2

/-- The cost of a single lily in yuan -/
def lilyCost : Nat := 3

/-- The total amount Xiaoming has to spend in yuan -/
def totalSpend : Nat := 20

/-- Predicate to check if a flower purchase is valid -/
def isValidPurchase (purchase : FlowerPurchase) : Prop :=
  carnationCost * purchase.carnations + lilyCost * purchase.lilies = totalSpend

/-- The theorem stating that there are exactly 3 valid flower purchase options -/
theorem three_flower_purchase_options :
  ∃ (options : List FlowerPurchase),
    (options.length = 3) ∧
    (∀ purchase ∈ options, isValidPurchase purchase) ∧
    (∀ purchase, isValidPurchase purchase → purchase ∈ options) :=
sorry

end three_flower_purchase_options_l1684_168441


namespace factory_production_l1684_168492

/-- The number of computers produced per day by a factory -/
def computers_per_day : ℕ := 1500

/-- The selling price of each computer in dollars -/
def price_per_computer : ℕ := 150

/-- The revenue from one week's production in dollars -/
def weekly_revenue : ℕ := 1575000

/-- The number of days in a week -/
def days_in_week : ℕ := 7

theorem factory_production :
  computers_per_day * price_per_computer * days_in_week = weekly_revenue :=
by sorry

end factory_production_l1684_168492


namespace unique_number_base_conversion_l1684_168402

def is_valid_base_8_digit (d : ℕ) : Prop := d < 8
def is_valid_base_6_digit (d : ℕ) : Prop := d < 6

def base_8_to_decimal (a b : ℕ) : ℕ := 8 * a + b
def base_6_to_decimal (b a : ℕ) : ℕ := 6 * b + a

theorem unique_number_base_conversion : ∃! n : ℕ, 
  ∃ (a b : ℕ), 
    is_valid_base_8_digit a ∧
    is_valid_base_6_digit b ∧
    n = base_8_to_decimal a b ∧
    n = base_6_to_decimal b a ∧
    n = 45 := by sorry

end unique_number_base_conversion_l1684_168402


namespace salary_restoration_l1684_168487

theorem salary_restoration (original_salary : ℝ) (original_salary_positive : original_salary > 0) :
  let reduced_salary := original_salary * (1 - 0.2)
  reduced_salary * (1 + 0.25) = original_salary :=
by sorry

end salary_restoration_l1684_168487


namespace fibonacci_determinant_identity_l1684_168478

def fibonacci : ℕ → ℤ
  | 0 => 0
  | 1 => 1
  | n + 2 => fibonacci (n + 1) + fibonacci n

def fibonacci_matrix (n : ℕ) : Matrix (Fin 2) (Fin 2) ℤ :=
  !![fibonacci (n + 1), fibonacci n; fibonacci n, fibonacci (n - 1)]

theorem fibonacci_determinant_identity (n : ℕ) :
  fibonacci (n + 1) * fibonacci (n - 1) - fibonacci n ^ 2 = (-1) ^ n :=
sorry

end fibonacci_determinant_identity_l1684_168478


namespace tan_function_property_l1684_168473

/-- Given a function f(x) = a * tan(b * x) where a and b are positive constants,
    if f has vertical asymptotes at x = ±π/4 and passes through (π/8, 3),
    then a * b = 6 -/
theorem tan_function_property (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (∀ x, x ≠ π/4 ∧ x ≠ -π/4 → ∃ y, y = a * Real.tan (b * x)) →
  a * Real.tan (b * π/8) = 3 →
  a * b = 6 := by
  sorry

end tan_function_property_l1684_168473


namespace unique_solution_abc_l1684_168452

theorem unique_solution_abc : ∃! (a b c : ℝ),
  a > 2 ∧ b > 2 ∧ c > 2 ∧
  ((a + 1)^2) / (b + c - 1) + ((b + 3)^2) / (c + a - 3) + ((c + 5)^2) / (a + b - 5) = 27 ∧
  a = 9 ∧ b = 7 ∧ c = 2 := by
  sorry

#check unique_solution_abc

end unique_solution_abc_l1684_168452


namespace sleeper_probability_l1684_168406

def total_delegates : ℕ := 9
def mexico_delegates : ℕ := 2
def canada_delegates : ℕ := 3
def us_delegates : ℕ := 4
def sleepers : ℕ := 3

theorem sleeper_probability :
  let total_outcomes := Nat.choose total_delegates sleepers
  let favorable_outcomes := 
    Nat.choose mexico_delegates 2 * Nat.choose canada_delegates 1 +
    Nat.choose mexico_delegates 2 * Nat.choose us_delegates 1 +
    Nat.choose canada_delegates 2 * Nat.choose mexico_delegates 1 +
    Nat.choose canada_delegates 2 * Nat.choose us_delegates 1 +
    Nat.choose us_delegates 2 * Nat.choose mexico_delegates 1 +
    Nat.choose us_delegates 2 * Nat.choose canada_delegates 1
  (favorable_outcomes : ℚ) / total_outcomes = 55 / 84 := by
  sorry

end sleeper_probability_l1684_168406


namespace abs_sum_diff_inequality_l1684_168422

theorem abs_sum_diff_inequality (x y : ℝ) :
  (abs x < 1 ∧ abs y < 1) ↔ abs (x + y) + abs (x - y) < 2 := by
  sorry

end abs_sum_diff_inequality_l1684_168422


namespace transform_1220_to_2012_not_transform_1220_to_2021_l1684_168469

/-- Represents a 4-digit number -/
structure FourDigitNumber where
  digits : Fin 4 → Fin 10

/-- Defines the allowed transformations on a 4-digit number -/
def transform (n : FourDigitNumber) (i : Fin 3) : Option FourDigitNumber :=
  if n.digits i ≠ 0 ∧ n.digits (i + 1) ≠ 0 then
    some ⟨fun j => if j = i ∨ j = i + 1 then n.digits j - 1 else n.digits j⟩
  else if n.digits i ≠ 9 ∧ n.digits (i + 1) ≠ 9 then
    some ⟨fun j => if j = i ∨ j = i + 1 then n.digits j + 1 else n.digits j⟩
  else
    none

/-- Defines the reachability of one number from another through transformations -/
def reachable (start finish : FourDigitNumber) : Prop :=
  ∃ (seq : List (Fin 3)), finish = seq.foldl (fun n i => (transform n i).getD n) start

/-- The initial number 1220 -/
def initial : FourDigitNumber := ⟨fun i => match i with | 0 => 1 | 1 => 2 | 2 => 2 | 3 => 0⟩

/-- The target number 2012 -/
def target1 : FourDigitNumber := ⟨fun i => match i with | 0 => 2 | 1 => 0 | 2 => 1 | 3 => 2⟩

/-- The target number 2021 -/
def target2 : FourDigitNumber := ⟨fun i => match i with | 0 => 2 | 1 => 0 | 2 => 2 | 3 => 1⟩

theorem transform_1220_to_2012 : reachable initial target1 := by sorry

theorem not_transform_1220_to_2021 : ¬reachable initial target2 := by sorry

end transform_1220_to_2012_not_transform_1220_to_2021_l1684_168469


namespace manufacturer_cost_effectiveness_l1684_168490

/-- Represents the cost calculation for manufacturers A and B -/
def cost_calculation (x : ℝ) : Prop :=
  let desk_price : ℝ := 200
  let chair_price : ℝ := 50
  let desk_quantity : ℝ := 60
  let discount_rate : ℝ := 0.9
  let cost_A : ℝ := desk_price * desk_quantity + chair_price * (x - desk_quantity)
  let cost_B : ℝ := (desk_price * desk_quantity + chair_price * x) * discount_rate
  (x ≥ desk_quantity) ∧
  (x < 360 → cost_A < cost_B) ∧
  (x > 360 → cost_B < cost_A)

/-- Theorem stating the conditions for cost-effectiveness of manufacturers A and B -/
theorem manufacturer_cost_effectiveness :
  ∀ x : ℝ, cost_calculation x :=
sorry

end manufacturer_cost_effectiveness_l1684_168490


namespace fraction_meaningful_l1684_168436

theorem fraction_meaningful (x : ℝ) : 
  (∃ y : ℝ, y = 3 / (x - 2)) ↔ x ≠ 2 :=
by sorry

end fraction_meaningful_l1684_168436


namespace identity_holds_iff_k_equals_negative_one_l1684_168499

theorem identity_holds_iff_k_equals_negative_one :
  ∀ k : ℝ, (∀ a b c : ℝ, (a + b) * (b + c) * (c + a) = (a + b + c) * (a * b + b * c + c * a) + k * a * b * c) ↔ k = -1 := by
  sorry

end identity_holds_iff_k_equals_negative_one_l1684_168499


namespace complex_sixth_power_real_count_l1684_168435

theorem complex_sixth_power_real_count : 
  ∃! (n : ℤ), (Complex.I + n : ℂ)^6 ∈ Set.range (Complex.ofReal : ℝ → ℂ) := by
  sorry

end complex_sixth_power_real_count_l1684_168435


namespace impossible_to_guarantee_same_state_l1684_168489

/-- Represents the state of a usamon (has an electron or not) -/
inductive UsamonState
| HasElectron
| NoElectron

/-- Represents a usamon with its current state -/
structure Usamon :=
  (state : UsamonState)

/-- Represents the action of connecting a diode between two usamons -/
def connectDiode (a b : Usamon) : Usamon × Usamon :=
  match a.state, b.state with
  | UsamonState.HasElectron, UsamonState.NoElectron => 
      ({ state := UsamonState.NoElectron }, { state := UsamonState.HasElectron })
  | _, _ => (a, b)

/-- The main theorem stating that it's impossible to guarantee two usamons are in the same state -/
theorem impossible_to_guarantee_same_state (usamons : Fin 2015 → Usamon) :
  ∀ (sequence : List (Fin 2015 × Fin 2015)),
  ¬∃ (i j : Fin 2015), i ≠ j ∧ (usamons i).state = (usamons j).state := by
  sorry


end impossible_to_guarantee_same_state_l1684_168489


namespace combined_blanket_thickness_l1684_168401

/-- The combined thickness of 5 blankets, each with an initial thickness of 3 inches
    and folded according to their color code (1 to 5), is equal to 186 inches. -/
theorem combined_blanket_thickness :
  let initial_thickness : ℝ := 3
  let color_codes : List ℕ := [1, 2, 3, 4, 5]
  let folded_thickness (c : ℕ) : ℝ := initial_thickness * (2 ^ c)
  List.sum (List.map folded_thickness color_codes) = 186 := by
  sorry


end combined_blanket_thickness_l1684_168401


namespace percentage_increase_problem_l1684_168417

theorem percentage_increase_problem (initial : ℝ) (increase_percent : ℝ) (decrease_percent : ℝ) (final : ℝ) :
  initial = 1500 →
  increase_percent = 20 →
  decrease_percent = 40 →
  final = 1080 →
  final = initial * (1 + increase_percent / 100) * (1 - decrease_percent / 100) :=
by sorry

end percentage_increase_problem_l1684_168417


namespace correct_proposition_l1684_168485

-- Define proposition p
def p : Prop := ∀ x : ℝ, x > 1 → x > 2

-- Define proposition q
def q : Prop := ∀ x y : ℝ, x + y ≠ 2 → x ≠ 1 ∨ y ≠ 1

-- Theorem to prove
theorem correct_proposition : (¬p) ∧ q := by
  sorry

end correct_proposition_l1684_168485


namespace triangle_area_three_lines_l1684_168403

/-- The area of the triangle formed by the intersection of three lines -/
theorem triangle_area_three_lines : 
  let line1 : ℝ → ℝ := λ x => 3 * x - 4
  let line2 : ℝ → ℝ := λ x => -2 * x + 16
  let y_axis : ℝ → ℝ := λ x => 0
  let intersection_x : ℝ := (16 + 4) / (3 + 2)
  let intersection_y : ℝ := line1 intersection_x
  let y_intercept1 : ℝ := line1 0
  let y_intercept2 : ℝ := line2 0
  let base : ℝ := y_intercept2 - y_intercept1
  let height : ℝ := intersection_x
  let area : ℝ := (1/2) * base * height
  area = 40 := by
sorry


end triangle_area_three_lines_l1684_168403


namespace rectangle_area_is_30_l1684_168440

/-- Represents a rectangle with length and width -/
structure Rectangle where
  length : ℝ
  width : ℝ

/-- Calculates the area of a rectangle -/
def Rectangle.area (r : Rectangle) : ℝ := r.length * r.width

/-- The original rectangle -/
def original : Rectangle := { length := 0, width := 0 }

/-- The rectangle with increased length -/
def increased_length : Rectangle := { length := original.length + 2, width := original.width }

/-- The rectangle with decreased width -/
def decreased_width : Rectangle := { length := original.length, width := original.width - 3 }

theorem rectangle_area_is_30 :
  increased_length.area - original.area = 10 →
  original.area - decreased_width.area = 18 →
  original.area = 30 := by sorry

end rectangle_area_is_30_l1684_168440


namespace sum_of_divisors_77_and_not_perfect_l1684_168472

def sum_of_divisors (n : ℕ) : ℕ := sorry

def is_perfect_number (n : ℕ) : Prop :=
  sum_of_divisors n = 2 * n

theorem sum_of_divisors_77_and_not_perfect :
  sum_of_divisors 77 = 96 ∧ ¬(is_perfect_number 77) := by sorry

end sum_of_divisors_77_and_not_perfect_l1684_168472


namespace therapy_pricing_theorem_l1684_168451

/-- Represents the pricing structure of a psychologist's therapy sessions. -/
structure TherapyPricing where
  firstHourCharge : ℕ
  additionalHourCharge : ℕ
  firstHourPremium : ℕ
  fiveHourTotal : ℕ

/-- Calculates the total charge for a given number of therapy hours. -/
def totalCharge (pricing : TherapyPricing) (hours : ℕ) : ℕ :=
  if hours = 0 then 0
  else pricing.firstHourCharge + (hours - 1) * pricing.additionalHourCharge

/-- Theorem stating the conditions and the result to be proved. -/
theorem therapy_pricing_theorem (pricing : TherapyPricing) 
  (h1 : pricing.firstHourCharge = pricing.additionalHourCharge + pricing.firstHourPremium)
  (h2 : pricing.firstHourPremium = 35)
  (h3 : pricing.fiveHourTotal = 350)
  (h4 : totalCharge pricing 5 = pricing.fiveHourTotal) :
  totalCharge pricing 2 = 161 := by
  sorry


end therapy_pricing_theorem_l1684_168451


namespace vector_c_solution_l1684_168459

theorem vector_c_solution (a b c : ℝ × ℝ) : 
  a = (1, 2) → 
  b = (2, -3) → 
  (∃ k : ℝ, c + a = k • b) →
  c • (a + b) = 0 →
  c = (-7/9, -7/3) := by sorry

end vector_c_solution_l1684_168459


namespace english_marks_proof_l1684_168460

def average (numbers : List ℕ) : ℚ :=
  (numbers.sum : ℚ) / numbers.length

theorem english_marks_proof (marks : List ℕ) (h1 : marks.length = 5) 
  (h2 : average marks = 76) 
  (h3 : 69 ∈ marks) (h4 : 92 ∈ marks) (h5 : 64 ∈ marks) (h6 : 82 ∈ marks) : 
  73 ∈ marks := by
  sorry

#check english_marks_proof

end english_marks_proof_l1684_168460


namespace right_triangle_condition_l1684_168437

theorem right_triangle_condition (A B C : ℝ) (h_triangle : A + B + C = Real.pi) 
  (h_condition : Real.sin A * Real.cos B = 1 - Real.cos A * Real.sin B) : C = Real.pi / 2 := by
  sorry

end right_triangle_condition_l1684_168437


namespace stating_policeman_speed_is_10_l1684_168480

/-- Represents the chase scenario between a policeman and a thief -/
structure ChaseScenario where
  initial_distance : ℝ  -- Initial distance in meters
  thief_speed : ℝ       -- Thief's speed in km/hr
  thief_distance : ℝ    -- Distance thief runs before being caught in meters
  policeman_speed : ℝ   -- Policeman's speed in km/hr

/-- 
Theorem stating that given the specific conditions of the chase,
the policeman's speed must be 10 km/hr
-/
theorem policeman_speed_is_10 (chase : ChaseScenario) 
  (h1 : chase.initial_distance = 100)
  (h2 : chase.thief_speed = 8)
  (h3 : chase.thief_distance = 400) :
  chase.policeman_speed = 10 := by
  sorry

#check policeman_speed_is_10

end stating_policeman_speed_is_10_l1684_168480


namespace triangle_side_a_value_l1684_168438

noncomputable def triangle_side_a (A B C : ℝ) (a b c : ℝ) : Prop :=
  -- Define the triangle ABC
  (0 < A ∧ 0 < B ∧ 0 < C) ∧
  (A + B + C = Real.pi) ∧
  -- Relate sides to angles using sine law
  (a / (Real.sin A) = b / (Real.sin B)) ∧
  (b / (Real.sin B) = c / (Real.sin C)) ∧
  -- Given conditions
  (Real.sin B = 3/5) ∧
  (b = 5) ∧
  (A = 2 * B) ∧
  -- Conclusion
  (a = 8)

theorem triangle_side_a_value :
  ∀ (A B C : ℝ) (a b c : ℝ),
  triangle_side_a A B C a b c :=
sorry

end triangle_side_a_value_l1684_168438


namespace valid_arrangements_l1684_168455

/-- Represents the number of plates of each color -/
structure PlateCount where
  yellow : Nat
  blue : Nat
  red : Nat
  purple : Nat

/-- Calculates the total number of plates -/
def totalPlates (count : PlateCount) : Nat :=
  count.yellow + count.blue + count.red + count.purple

/-- Calculates the number of circular arrangements -/
def circularArrangements (count : PlateCount) : Nat :=
  sorry

/-- Calculates the number of circular arrangements with red plates adjacent -/
def redAdjacentArrangements (count : PlateCount) : Nat :=
  sorry

/-- The main theorem stating the number of valid arrangements -/
theorem valid_arrangements (count : PlateCount) 
  (h1 : count.yellow = 4)
  (h2 : count.blue = 3)
  (h3 : count.red = 2)
  (h4 : count.purple = 1) :
  circularArrangements count - redAdjacentArrangements count = 980 := by
  sorry

end valid_arrangements_l1684_168455


namespace equilateral_triangle_area_perimeter_ratio_l1684_168428

theorem equilateral_triangle_area_perimeter_ratio :
  let side_length : ℝ := 12
  let height : ℝ := side_length * (Real.sqrt 3 / 2)
  let area : ℝ := (1 / 2) * side_length * height
  let perimeter : ℝ := 3 * side_length
  area / perimeter = Real.sqrt 3 := by sorry

end equilateral_triangle_area_perimeter_ratio_l1684_168428


namespace percentage_to_number_l1684_168405

theorem percentage_to_number (x : ℝ) (h : x = 209) :
  x / 100 * 100 = 209 := by
  sorry

end percentage_to_number_l1684_168405


namespace scatter_plot_for_linear_relationships_l1684_168483

-- Define the concept of a data visualization method
def DataVisualizationMethod : Type := String

-- Define scatter plot as a data visualization method
def scatter_plot : DataVisualizationMethod := "Scatter plot"

-- Define the property of showing relationships between data points
def shows_point_relationships (method : DataVisualizationMethod) : Prop := 
  method = scatter_plot

-- Define the property of being appropriate for determining linear relationships
def appropriate_for_linear_relationships (method : DataVisualizationMethod) : Prop :=
  shows_point_relationships method

-- Theorem stating that scatter plot is appropriate for determining linear relationships
theorem scatter_plot_for_linear_relationships :
  appropriate_for_linear_relationships scatter_plot :=
by
  sorry


end scatter_plot_for_linear_relationships_l1684_168483


namespace expression_simplification_l1684_168429

theorem expression_simplification (a : ℝ) (h : a = Real.sqrt 2 + 2) :
  (1 / (a - 2) - 2 / (a^2 - 4)) / ((a^2 - 2*a) / (a^2 - 4)) = Real.sqrt 2 / 2 := by
  sorry

end expression_simplification_l1684_168429


namespace coefficient_x_cubed_in_expansion_l1684_168445

/-- The coefficient of x^3 in the expansion of (1-2x^2)(1+x)^4 is -4 -/
theorem coefficient_x_cubed_in_expansion : ∃ (p : Polynomial ℤ), 
  p = (1 - 2 * X^2) * (1 + X)^4 ∧ p.coeff 3 = -4 := by sorry

end coefficient_x_cubed_in_expansion_l1684_168445


namespace gcd_bound_from_lcm_l1684_168410

theorem gcd_bound_from_lcm (a b : ℕ) : 
  10000 ≤ a ∧ a < 100000 ∧ 
  10000 ≤ b ∧ b < 100000 ∧ 
  100000000 ≤ Nat.lcm a b ∧ Nat.lcm a b < 1000000000 → 
  Nat.gcd a b < 100 := by
sorry

end gcd_bound_from_lcm_l1684_168410


namespace octahedron_cube_volume_ratio_l1684_168497

/-- The ratio of the volume of an octahedron formed by the centers of the faces of a cube
    to the volume of the cube itself, given that the cube has side length 2. -/
theorem octahedron_cube_volume_ratio :
  let cube_side_length : ℝ := 2
  let cube_volume : ℝ := cube_side_length ^ 3
  let octahedron_volume : ℝ := 4 / 3
  octahedron_volume / cube_volume = 1 / 6 := by sorry

end octahedron_cube_volume_ratio_l1684_168497


namespace sum_integers_11_to_24_l1684_168414

def sum_integers (a b : ℕ) : ℕ := (b - a + 1) * (a + b) / 2

theorem sum_integers_11_to_24 : sum_integers 11 24 = 245 := by sorry

end sum_integers_11_to_24_l1684_168414


namespace sum_of_series_equals_three_fourths_l1684_168479

/-- The sum of the infinite series ∑(k=1 to ∞) k/3^k is equal to 3/4 -/
theorem sum_of_series_equals_three_fourths :
  (∑' k : ℕ+, (k : ℝ) / (3 : ℝ) ^ (k : ℕ)) = 3 / 4 := by
  sorry

end sum_of_series_equals_three_fourths_l1684_168479


namespace sum_remainder_mod_16_l1684_168424

theorem sum_remainder_mod_16 : (List.sum [75, 76, 77, 78, 79, 80, 81, 82]) % 16 = 4 := by
  sorry

end sum_remainder_mod_16_l1684_168424


namespace quadratic_factorization_sum_l1684_168498

theorem quadratic_factorization_sum (h b c d : ℤ) : 
  (∀ x : ℝ, 6 * x^2 + x - 12 = (h * x + b) * (c * x + d)) → 
  |h| + |b| + |c| + |d| = 12 := by
sorry

end quadratic_factorization_sum_l1684_168498


namespace leo_weight_proof_l1684_168467

/-- Leo's current weight -/
def leo_weight : ℝ := 103.6

/-- Kendra's weight -/
def kendra_weight : ℝ := 68

/-- Jake's weight -/
def jake_weight : ℝ := kendra_weight + 30

theorem leo_weight_proof :
  -- Condition 1: If Leo gains 12 pounds, he will weigh 70% more than Kendra
  (leo_weight + 12 = 1.7 * kendra_weight) ∧
  -- Condition 2: The combined weight of Leo, Kendra, and Jake is 270 pounds
  (leo_weight + kendra_weight + jake_weight = 270) ∧
  -- Condition 3: Jake weighs 30 pounds more than Kendra
  (jake_weight = kendra_weight + 30) →
  -- Conclusion: Leo's current weight is 103.6 pounds
  leo_weight = 103.6 := by
sorry

end leo_weight_proof_l1684_168467


namespace selectPeopleCount_l1684_168433

/-- The number of ways to select k items from n items -/
def choose (n k : ℕ) : ℕ := Nat.choose n k

/-- The number of ways to select 4 people from 4 boys and 3 girls, 
    ensuring both boys and girls are included -/
def selectPeople : ℕ :=
  choose 4 3 * choose 3 1 + 
  choose 4 2 * choose 3 2 + 
  choose 4 1 * choose 3 3

theorem selectPeopleCount : selectPeople = 34 := by sorry

end selectPeopleCount_l1684_168433


namespace megawheel_capacity_l1684_168444

/-- The Megawheel problem -/
theorem megawheel_capacity (total_seats : ℕ) (total_people : ℕ) (people_per_seat : ℕ) 
  (h1 : total_seats = 15)
  (h2 : total_people = 75)
  (h3 : people_per_seat * total_seats = total_people) :
  people_per_seat = 5 := by
  sorry

end megawheel_capacity_l1684_168444


namespace fish_pond_estimation_l1684_168462

theorem fish_pond_estimation (x : ℕ) 
  (h1 : x > 0)  -- Ensure the pond has fish
  (h2 : 30 ≤ x) -- Ensure we can catch 30 fish initially
  : (2 : ℚ) / 30 = 30 / x → x = 450 := by
  sorry

#check fish_pond_estimation

end fish_pond_estimation_l1684_168462


namespace correct_average_after_error_correction_l1684_168481

/-- Given 12 numbers with an initial average of 22, where three numbers were incorrectly read
    (52 as 32, 47 as 27, and 68 as 45), the correct average is 27.25. -/
theorem correct_average_after_error_correction (total_numbers : ℕ) (initial_average : ℚ)
  (incorrect_num1 incorrect_num2 incorrect_num3 : ℚ)
  (correct_num1 correct_num2 correct_num3 : ℚ) :
  total_numbers = 12 →
  initial_average = 22 →
  incorrect_num1 = 32 →
  incorrect_num2 = 27 →
  incorrect_num3 = 45 →
  correct_num1 = 52 →
  correct_num2 = 47 →
  correct_num3 = 68 →
  ((total_numbers : ℚ) * initial_average - incorrect_num1 - incorrect_num2 - incorrect_num3 +
    correct_num1 + correct_num2 + correct_num3) / total_numbers = 27.25 :=
by sorry

end correct_average_after_error_correction_l1684_168481


namespace sentences_started_today_l1684_168466

/-- Calculates the number of sentences Janice started with today given her typing speed and work schedule. -/
theorem sentences_started_today (
  typing_speed : ℕ)  -- Sentences typed per minute
  (initial_typing_time : ℕ)  -- Minutes typed before break
  (extra_typing_time : ℕ)  -- Additional minutes typed after break
  (erased_sentences : ℕ)  -- Number of sentences erased due to errors
  (final_typing_time : ℕ)  -- Minutes typed after meeting
  (total_sentences : ℕ)  -- Total sentences in the paper by end of day
  (h1 : typing_speed = 6)
  (h2 : initial_typing_time = 20)
  (h3 : extra_typing_time = 15)
  (h4 : erased_sentences = 40)
  (h5 : final_typing_time = 18)
  (h6 : total_sentences = 536)
  : ℕ := by
  sorry

end sentences_started_today_l1684_168466


namespace b_completion_time_l1684_168423

-- Define the work rates and time worked by A
def a_rate : ℚ := 1 / 20
def b_rate : ℚ := 1 / 30
def a_time_worked : ℕ := 10

-- Define the total work as 1 (representing 100%)
def total_work : ℚ := 1

-- Theorem statement
theorem b_completion_time :
  let work_done_by_a : ℚ := a_rate * a_time_worked
  let remaining_work : ℚ := total_work - work_done_by_a
  remaining_work / b_rate = 15 := by
  sorry

end b_completion_time_l1684_168423


namespace complex_modulus_problem_l1684_168447

theorem complex_modulus_problem (i z : ℂ) (h1 : i^2 = -1) (h2 : i * z = (1 - 2*i)^2) : 
  Complex.abs z = 5 := by sorry

end complex_modulus_problem_l1684_168447


namespace rudolph_trip_signs_per_mile_l1684_168453

/-- Rudolph's car trip across town -/
def rudolph_trip (miles_base : ℕ) (miles_extra : ℕ) (signs_base : ℕ) (signs_less : ℕ) : ℚ :=
  let total_miles : ℕ := miles_base + miles_extra
  let total_signs : ℕ := signs_base - signs_less
  (total_signs : ℚ) / (total_miles : ℚ)

/-- Theorem stating the number of stop signs per mile Rudolph encountered -/
theorem rudolph_trip_signs_per_mile :
  rudolph_trip 5 2 17 3 = 2 := by
  sorry

end rudolph_trip_signs_per_mile_l1684_168453


namespace prob_sum_7_twice_l1684_168471

/-- The number of sides on each die -/
def sides : ℕ := 6

/-- The set of possible outcomes for a single die roll -/
def outcomes : Set ℕ := {1, 2, 3, 4, 5, 6}

/-- The probability of rolling a sum of 7 with two dice -/
def prob_sum_7 : ℚ := 6 / 36

/-- The probability of rolling a sum of 7 twice in a row with two dice -/
theorem prob_sum_7_twice (h : sides = 6) : prob_sum_7 * prob_sum_7 = 1 / 36 := by
  sorry

end prob_sum_7_twice_l1684_168471


namespace prob_at_least_one_spade_or_ace_value_l1684_168442

/-- The number of cards in the deck -/
def deck_size : ℕ := 54

/-- The number of cards that are either spades or aces -/
def spade_or_ace_count : ℕ := 16

/-- The probability of drawing at least one spade or ace in two independent draws with replacement -/
def prob_at_least_one_spade_or_ace : ℚ :=
  1 - (1 - spade_or_ace_count / deck_size) ^ 2

theorem prob_at_least_one_spade_or_ace_value :
  prob_at_least_one_spade_or_ace = 368 / 729 := by
  sorry

end prob_at_least_one_spade_or_ace_value_l1684_168442


namespace hyperbola_eccentricity_l1684_168496

/-- Given a hyperbola C with equation x²/m - y² = 1 and one focus at (2, 0),
    prove that its eccentricity is 2√3/3 -/
theorem hyperbola_eccentricity (m : ℝ) (h1 : m > 0) :
  let C := {(x, y) : ℝ × ℝ | x^2 / m - y^2 = 1}
  let focus : ℝ × ℝ := (2, 0)
  focus ∈ {f | ∃ (x y : ℝ), (x, y) ∈ C ∧ (x - f.1)^2 + (y - f.2)^2 = (x + f.1)^2 + (y - f.2)^2} →
  let e := Real.sqrt ((2 : ℝ)^2 / m)
  e = 2 * Real.sqrt 3 / 3 :=
by sorry

end hyperbola_eccentricity_l1684_168496


namespace percentage_of_160_to_50_l1684_168419

theorem percentage_of_160_to_50 : ∀ x : ℝ, (160 / 50) * 100 = x → x = 320 := by
  sorry

end percentage_of_160_to_50_l1684_168419


namespace unique_integer_l1684_168443

def is_valid_integer (n : ℕ) : Prop :=
  ∃ (a b c d : ℕ),
    n = a * 1000 + b * 100 + c * 10 + d ∧
    0 < a ∧ a < 10 ∧ 0 ≤ b ∧ b < 10 ∧ 0 ≤ c ∧ c < 10 ∧ 0 ≤ d ∧ d < 10 ∧
    a + b + c + d = 14 ∧
    b + c = 9 ∧
    a - d = 1 ∧
    n % 11 = 0

theorem unique_integer : ∃! n : ℕ, is_valid_integer n ∧ n = 3542 := by
  sorry

end unique_integer_l1684_168443


namespace orange_calculation_l1684_168491

/-- Calculates the total number and weight of oranges given the number of children,
    oranges per child, and average weight per orange. -/
theorem orange_calculation (num_children : ℕ) (oranges_per_child : ℕ) (avg_weight : ℚ) :
  num_children = 4 →
  oranges_per_child = 3 →
  avg_weight = 3/10 →
  (num_children * oranges_per_child = 12 ∧
   (num_children * oranges_per_child : ℚ) * avg_weight = 18/5) :=
by sorry

end orange_calculation_l1684_168491


namespace kids_to_adult_meals_ratio_l1684_168448

theorem kids_to_adult_meals_ratio 
  (kids_meals : ℕ) 
  (total_meals : ℕ) 
  (h1 : kids_meals = 8) 
  (h2 : total_meals = 12) : 
  (kids_meals : ℚ) / ((total_meals - kids_meals) : ℚ) = 2 / 1 := by
sorry

end kids_to_adult_meals_ratio_l1684_168448


namespace actual_tax_raise_expectation_l1684_168449

-- Define the population
def Population := ℝ

-- Define the fraction of liars and economists
def fraction_liars : ℝ := 0.1
def fraction_economists : ℝ := 0.9

-- Define the affirmative answer percentages
def taxes_raised : ℝ := 0.4
def money_supply_increased : ℝ := 0.3
def bonds_issued : ℝ := 0.5
def reserves_spent : ℝ := 0

-- Define the theorem
theorem actual_tax_raise_expectation :
  let total_affirmative := taxes_raised + money_supply_increased + bonds_issued + reserves_spent
  fraction_liars * 3 + fraction_economists = total_affirmative →
  taxes_raised - fraction_liars = 0.3 :=
by sorry

end actual_tax_raise_expectation_l1684_168449


namespace compute_fraction_power_l1684_168430

theorem compute_fraction_power : 8 * (2 / 7)^4 = 128 / 2401 := by
  sorry

end compute_fraction_power_l1684_168430


namespace expression_value_l1684_168418

theorem expression_value (x y : ℤ) (hx : x = 3) (hy : y = 2) :
  3 * x - 4 * y = 1 := by sorry

end expression_value_l1684_168418


namespace tournament_rankings_l1684_168456

/-- Represents a team in the volleyball tournament -/
inductive Team : Type
| E | F | G | H | I | J

/-- Represents a match between two teams -/
structure Match where
  team1 : Team
  team2 : Team

/-- Represents the tournament structure -/
structure Tournament where
  saturday_matches : Vector Match 3
  no_ties : Bool

/-- Calculates the number of possible ranking sequences -/
def possible_rankings (t : Tournament) : Nat :=
  6 * 6

/-- Theorem: The number of possible six-team ranking sequences is 36 -/
theorem tournament_rankings (t : Tournament) :
  t.no_ties → possible_rankings t = 36 := by
  sorry

end tournament_rankings_l1684_168456


namespace debra_accusation_l1684_168464

/-- Represents the number of cookies in various states -/
structure CookieCount where
  initial : ℕ
  louSeniorEaten : ℕ
  louieJuniorTaken : ℕ
  remaining : ℕ

/-- The cookie scenario as described in the problem -/
def cookieScenario : CookieCount where
  initial := 22
  louSeniorEaten := 4
  louieJuniorTaken := 7
  remaining := 11

/-- Theorem stating the portion of cookies Debra accuses Lou Senior of eating -/
theorem debra_accusation (c : CookieCount) (h1 : c = cookieScenario) :
  c.louSeniorEaten = 4 ∧ c.initial = 22 := by sorry

end debra_accusation_l1684_168464


namespace quadratic_root_difference_l1684_168463

theorem quadratic_root_difference (p q : ℝ) (hp : p > 0) (hq : q > 0) :
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧
    x₁^2 + p*x₁ + q = 0 ∧
    x₂^2 + p*x₂ + q = 0 ∧
    |x₁ - x₂| = 1) →
  p = Real.sqrt (4*q + 1) :=
by sorry

end quadratic_root_difference_l1684_168463


namespace f_is_quadratic_l1684_168404

/-- Definition of a quadratic equation -/
def is_quadratic_equation (f : ℝ → ℝ) : Prop :=
  ∃ (a b c : ℝ), a ≠ 0 ∧ ∀ x, f x = a * x^2 + b * x + c

/-- The specific equation we want to prove is quadratic -/
def f (x : ℝ) : ℝ := x^2 + 2*x - 1

/-- Theorem stating that f is a quadratic equation -/
theorem f_is_quadratic : is_quadratic_equation f := by
  sorry

end f_is_quadratic_l1684_168404


namespace exists_1990_edge_polyhedron_no_triangles_l1684_168434

/-- A convex polyhedron. -/
structure ConvexPolyhedron where
  -- Define the necessary properties of a convex polyhedron
  isConvex : Bool
  edges : Nat
  faces : List Nat

/-- Checks if a polyhedron has no triangular faces. -/
def hasNoTriangularFaces (p : ConvexPolyhedron) : Bool :=
  p.faces.all (· > 3)

/-- Theorem stating the existence of a convex polyhedron with 1990 edges and no triangular faces. -/
theorem exists_1990_edge_polyhedron_no_triangles : 
  ∃ p : ConvexPolyhedron, p.isConvex ∧ p.edges = 1990 ∧ hasNoTriangularFaces p :=
sorry

end exists_1990_edge_polyhedron_no_triangles_l1684_168434


namespace average_sales_is_16_l1684_168427

def january_sales : ℕ := 15
def february_sales : ℕ := 16
def march_sales : ℕ := 17

def total_months : ℕ := 3

def average_sales : ℚ := (january_sales + february_sales + march_sales : ℚ) / total_months

theorem average_sales_is_16 : average_sales = 16 := by
  sorry

end average_sales_is_16_l1684_168427


namespace truck_speed_l1684_168415

/-- Proves that a truck traveling 600 meters in 20 seconds has a speed of 108 kilometers per hour. -/
theorem truck_speed (distance : ℝ) (time : ℝ) (speed_ms : ℝ) (speed_kmh : ℝ) : 
  distance = 600 →
  time = 20 →
  speed_ms = distance / time →
  speed_kmh = speed_ms * 3.6 →
  speed_kmh = 108 := by
sorry

end truck_speed_l1684_168415


namespace oldest_child_age_l1684_168407

theorem oldest_child_age (a b c : ℕ) (h1 : a = 6) (h2 : b = 8) 
  (h3 : (a + b + c) / 3 = 9) (h4 : c ≥ b) (h5 : b ≥ a) : c = 13 := by
  sorry

end oldest_child_age_l1684_168407


namespace modulo_eleven_residue_l1684_168426

theorem modulo_eleven_residue : (312 + 6 * 47 + 8 * 154 + 5 * 22) % 11 = 0 := by
  sorry

end modulo_eleven_residue_l1684_168426


namespace max_value_of_operation_l1684_168454

theorem max_value_of_operation : ∃ (m : ℕ), 
  (∀ (n : ℕ), 10 ≤ n ∧ n ≤ 99 → 3 * (300 - n) ≤ m) ∧ 
  (∃ (n : ℕ), 10 ≤ n ∧ n ≤ 99 ∧ 3 * (300 - n) = m) ∧ 
  m = 870 := by
sorry

end max_value_of_operation_l1684_168454


namespace selling_price_ratio_l1684_168493

theorem selling_price_ratio (C : ℝ) (S1 S2 : ℝ) 
  (h1 : S1 = C + 0.60 * C) 
  (h2 : S2 = C + 3.20 * C) : 
  S2 / S1 = 21 / 8 := by
  sorry

end selling_price_ratio_l1684_168493


namespace problem_solution_l1684_168486

theorem problem_solution :
  -- Part 1
  let n : ℕ := Finset.sum (Finset.range 16) (λ i => 2 * i + 1)
  let m : ℕ := Finset.sum (Finset.range 16) (λ i => 2 * (i + 1))
  m - n = 16 ∧
  -- Part 2
  let trapezium_area (a b h : ℝ) := (a + b) * h / 2
  trapezium_area 4 16 16 = 160 ∧
  -- Part 3
  let isosceles_triangle (side angle : ℝ) := side > 0 ∧ 0 < angle ∧ angle < π
  ∀ side angle, isosceles_triangle side angle → angle = π / 3 → 3 = 3 ∧
  -- Part 4
  let f (x : ℝ) := 3 * x^(2/3) - 8 * x^(1/3) + 4
  ∃ x : ℝ, x > 0 ∧ f x = 0 ∧ x = 8/27 ∧ ∀ y, y > 0 → f y = 0 → x ≤ y :=
by
  sorry

end problem_solution_l1684_168486


namespace ellipse_hyperbola_same_foci_l1684_168476

/-- Given an ellipse and a hyperbola with the same foci, prove that the parameter n is √6 -/
theorem ellipse_hyperbola_same_foci (n : ℝ) :
  n > 0 →
  (∀ x y : ℝ, x^2 / 16 + y^2 / n^2 = 1 ↔ x^2 / n^2 - y^2 / 4 = 1) →
  n = Real.sqrt 6 :=
by sorry

end ellipse_hyperbola_same_foci_l1684_168476


namespace relation_between_exponents_l1684_168477

-- Define variables
variable (a d c e : ℝ)
variable (u v w r : ℝ)

-- State the theorem
theorem relation_between_exponents 
  (h1 : a^u = d^r) 
  (h2 : d^r = c)
  (h3 : d^v = a^w)
  (h4 : a^w = e) :
  r * w = v * u := by
  sorry

end relation_between_exponents_l1684_168477


namespace unique_prime_six_digit_number_l1684_168446

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ d : ℕ, d > 1 → d < n → ¬(n % d = 0)

def six_digit_number (B : ℕ) : ℕ :=
  303100 + B

theorem unique_prime_six_digit_number :
  ∃! B : ℕ, B < 10 ∧ is_prime (six_digit_number B) ∧ six_digit_number B = 303101 :=
sorry

end unique_prime_six_digit_number_l1684_168446


namespace min_value_theorem_l1684_168413

theorem min_value_theorem (x y z : ℝ) (h : x + y + z = x*y + y*z + z*x) :
  (x / (x^2 + 1)) + (y / (y^2 + 1)) + (z / (z^2 + 1)) ≥ -1/2 := by
  sorry

end min_value_theorem_l1684_168413


namespace ellipse_equation_1_ellipse_equation_2_l1684_168400

-- Define the type for points in 2D space
structure Point2D where
  x : ℝ
  y : ℝ

-- Define the type for ellipses
structure Ellipse where
  a : ℝ
  b : ℝ

def is_on_ellipse (e : Ellipse) (p : Point2D) : Prop :=
  (p.x^2 / e.a^2) + (p.y^2 / e.b^2) = 1

def has_common_focus (e1 e2 : Ellipse) : Prop :=
  ∃ (f : Point2D), (f.x^2 = e1.a^2 - e1.b^2) ∧ (f.x^2 = e2.a^2 - e2.b^2)

def has_foci_on_axes (e : Ellipse) : Prop :=
  ∃ (f : ℝ), (f^2 = e.a^2 - e.b^2) ∧ (f ≠ 0)

theorem ellipse_equation_1 :
  ∃ (e : Ellipse),
    has_common_focus e (Ellipse.mk 3 2) ∧
    is_on_ellipse e (Point2D.mk 3 (-2)) ∧
    has_foci_on_axes e ∧
    e.a^2 = 15 ∧ e.b^2 = 10 := by sorry

theorem ellipse_equation_2 :
  ∃ (e : Ellipse),
    has_foci_on_axes e ∧
    is_on_ellipse e (Point2D.mk (Real.sqrt 3) (-2)) ∧
    is_on_ellipse e (Point2D.mk (-2 * Real.sqrt 3) 1) ∧
    e.a^2 = 15 ∧ e.b^2 = 5 := by sorry

end ellipse_equation_1_ellipse_equation_2_l1684_168400


namespace intersection_product_is_three_l1684_168408

-- Define the two circle equations
def circle1 (x y : ℝ) : Prop := x^2 + 2*x + y^2 + 4*y + 4 = 0
def circle2 (x y : ℝ) : Prop := x^2 + 4*x + y^2 + 4*y + 7 = 0

-- Define the intersection point
def intersection_point : ℝ × ℝ := (-1.5, -2)

-- Theorem statement
theorem intersection_product_is_three :
  let (x, y) := intersection_point
  circle1 x y ∧ circle2 x y ∧ x * y = 3 :=
by sorry

end intersection_product_is_three_l1684_168408


namespace equidistant_point_is_perpendicular_bisector_intersection_l1684_168420

-- Define a triangle in a 2D plane
structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ

-- Define a point in a 2D plane
def Point := ℝ × ℝ

-- Define the distance between two points
def distance (p1 p2 : Point) : ℝ := sorry

-- Define a perpendicular bisector of a line segment
def perpendicularBisector (p1 p2 : Point) : Set Point := sorry

-- Define the intersection of three sets
def intersectionOfThree (s1 s2 s3 : Set Point) : Set Point := sorry

-- Theorem statement
theorem equidistant_point_is_perpendicular_bisector_intersection (t : Triangle) :
  ∃ (p : Point),
    (distance p t.A = distance p t.B ∧ distance p t.B = distance p t.C) ↔
    p ∈ intersectionOfThree
      (perpendicularBisector t.A t.B)
      (perpendicularBisector t.B t.C)
      (perpendicularBisector t.C t.A) :=
sorry

end equidistant_point_is_perpendicular_bisector_intersection_l1684_168420


namespace factors_of_product_l1684_168425

/-- A natural number with exactly three factors is the square of a prime. -/
def is_prime_square (n : ℕ) : Prop :=
  ∃ p : ℕ, Nat.Prime p ∧ n = p^2

/-- The number of factors of n^k where n is a prime square. -/
def num_factors_prime_square_pow (n k : ℕ) : ℕ :=
  2 * k + 1

/-- The main theorem -/
theorem factors_of_product (a b c : ℕ) 
  (ha : is_prime_square a) 
  (hb : is_prime_square b)
  (hc : is_prime_square c)
  (hab : a ≠ b) (hbc : b ≠ c) (hac : a ≠ c) :
  (num_factors_prime_square_pow a 3) * 
  (num_factors_prime_square_pow b 4) * 
  (num_factors_prime_square_pow c 5) = 693 := by
  sorry

end factors_of_product_l1684_168425


namespace smallest_positive_time_for_104_degrees_l1684_168421

def temperature (t : ℝ) : ℝ := -t^2 + 16*t + 40

theorem smallest_positive_time_for_104_degrees :
  let t := 8 + 8 * Real.sqrt 2
  (∀ s, s > 0 ∧ temperature s = 104 → s ≥ t) ∧ temperature t = 104 ∧ t > 0 := by
sorry

end smallest_positive_time_for_104_degrees_l1684_168421


namespace scientific_notation_21600_l1684_168409

/-- Scientific notation representation of a real number -/
structure ScientificNotation where
  coefficient : ℝ
  exponent : ℤ
  h1 : 1 ≤ |coefficient| ∧ |coefficient| < 10

/-- Function to convert a real number to scientific notation -/
def toScientificNotation (x : ℝ) : ScientificNotation :=
  sorry

theorem scientific_notation_21600 :
  toScientificNotation 21600 = ScientificNotation.mk 2.16 4 sorry := by sorry

end scientific_notation_21600_l1684_168409


namespace product_abcd_l1684_168470

theorem product_abcd (a b c d : ℚ) : 
  3*a + 4*b + 6*c + 8*d = 42 →
  4*(d+c) = b →
  4*b + 2*c = a →
  c - 2 = d →
  a * b * c * d = (367/37) * (76/37) * (93/74) * (-55/74) := by
sorry

end product_abcd_l1684_168470


namespace joan_toy_cars_cost_l1684_168431

theorem joan_toy_cars_cost (total_toys cost_skateboard cost_trucks : ℚ)
  (h1 : total_toys = 25.62)
  (h2 : cost_skateboard = 4.88)
  (h3 : cost_trucks = 5.86) :
  total_toys - cost_skateboard - cost_trucks = 14.88 := by
  sorry

end joan_toy_cars_cost_l1684_168431


namespace quadratic_function_minimum_l1684_168450

-- Define the quadratic function
def f (a b : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x + 1

-- State the theorem
theorem quadratic_function_minimum (a b : ℝ) :
  (∀ x : ℝ, f a b x ≥ f a b (-1)) ∧ (f a b (-1) = 0) →
  ∀ x : ℝ, f a b x = x^2 + 2*x + 1 := by
  sorry

end quadratic_function_minimum_l1684_168450


namespace inequality_of_squares_existence_of_positive_l1684_168465

theorem inequality_of_squares (a b c : ℝ) : a^2 + b^2 + c^2 ≥ a*b + b*c + c*a := by
  sorry

theorem existence_of_positive (x y z : ℝ) :
  let a := x^2 - 2*y + Real.pi/2
  let b := y^2 - 2*z + Real.pi/3
  let c := z^2 - 2*x + Real.pi/6
  (a > 0) ∨ (b > 0) ∨ (c > 0) := by
  sorry

end inequality_of_squares_existence_of_positive_l1684_168465


namespace max_three_digit_gp_length_l1684_168411

/-- A geometric progression of 3-digit natural numbers -/
def ThreeDigitGP (a : ℕ → ℕ) (n : ℕ) : Prop :=
  ∃ (q : ℚ), q > 1 ∧
  (∀ i ≤ n, 100 ≤ a i ∧ a i < 1000) ∧
  (∀ i < n, a (i + 1) = (a i : ℚ) * q)

/-- The maximum length of a 3-digit geometric progression -/
def MaxGPLength : ℕ := 6

/-- Theorem stating that 6 is the maximum length of a 3-digit geometric progression -/
theorem max_three_digit_gp_length :
  (∃ a : ℕ → ℕ, ThreeDigitGP a MaxGPLength) ∧
  (∀ n > MaxGPLength, ∀ a : ℕ → ℕ, ¬ ThreeDigitGP a n) :=
sorry

end max_three_digit_gp_length_l1684_168411


namespace pigeon_percentage_among_non_swans_l1684_168432

def bird_distribution (total : ℝ) : ℝ × ℝ × ℝ × ℝ × ℝ := 
  (0.20 * total, 0.30 * total, 0.15 * total, 0.25 * total, 0.10 * total)

theorem pigeon_percentage_among_non_swans (total : ℝ) (h : total > 0) :
  let (geese, swans, herons, ducks, pigeons) := bird_distribution total
  let non_swans := total - swans
  (pigeons / non_swans) * 100 = 14 := by
  sorry

end pigeon_percentage_among_non_swans_l1684_168432


namespace rectangular_plot_length_l1684_168475

/-- Proves that the length of a rectangular plot is 70 meters given the specified conditions. -/
theorem rectangular_plot_length (breadth : ℝ) (length : ℝ) (perimeter : ℝ) : 
  length = breadth + 40 →
  perimeter = 2 * (length + breadth) →
  26.50 * perimeter = 5300 →
  length = 70 := by
  sorry

end rectangular_plot_length_l1684_168475
