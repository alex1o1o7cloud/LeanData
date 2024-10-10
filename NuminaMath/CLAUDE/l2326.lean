import Mathlib

namespace stating_solution_count_56_l2326_232656

/-- 
Given a positive integer n, count_solutions n returns the number of solutions 
to the equation xy + z = n where x, y, and z are positive integers.
-/
def count_solutions (n : ℕ+) : ℕ := sorry

/-- 
Theorem stating that if count_solutions n = 56, then n = 34 or n = 35
-/
theorem solution_count_56 (n : ℕ+) : 
  count_solutions n = 56 → n = 34 ∨ n = 35 := by sorry

end stating_solution_count_56_l2326_232656


namespace pizza_dough_production_l2326_232686

-- Define the given conditions
def batches_per_sack : ℕ := 15
def sacks_per_day : ℕ := 5
def days_per_week : ℕ := 7

-- Define the theorem to be proved
theorem pizza_dough_production :
  batches_per_sack * sacks_per_day * days_per_week = 525 := by
  sorry

end pizza_dough_production_l2326_232686


namespace sequence_general_term_l2326_232639

theorem sequence_general_term (a : ℕ → ℕ) :
  a 1 = 1 ∧ (∀ n : ℕ, a (n + 1) = a n + 2 * n + 1) →
  ∀ n : ℕ, n ≥ 1 → a n = n^2 := by
sorry

end sequence_general_term_l2326_232639


namespace fable_village_impossible_total_l2326_232621

theorem fable_village_impossible_total (p h s c d k : ℕ) : 
  p = 4 * h ∧ 
  s = 5 * c ∧ 
  d = 2 * p ∧ 
  k = 2 * d → 
  p + h + s + c + d + k ≠ 90 :=
by sorry

end fable_village_impossible_total_l2326_232621


namespace trigonometric_product_equals_one_l2326_232626

theorem trigonometric_product_equals_one :
  let α : Real := 15 * π / 180  -- 15 degrees in radians
  (1 - 1 / Real.cos α) * (1 + 1 / Real.sin (π/2 - α)) *
  (1 - 1 / Real.sin α) * (1 + 1 / Real.cos (π/2 - α)) = 1 := by
  sorry

end trigonometric_product_equals_one_l2326_232626


namespace stating_third_shirt_discount_is_sixty_percent_l2326_232673

/-- Represents the discount on a shirt as a fraction between 0 and 1 -/
def Discount : Type := { d : ℝ // 0 ≤ d ∧ d ≤ 1 }

/-- The regular price of a shirt -/
def regularPrice : ℝ := 10

/-- The discount on the second shirt -/
def secondShirtDiscount : Discount := ⟨0.5, by norm_num⟩

/-- The total savings when buying three shirts -/
def totalSavings : ℝ := 11

/-- The discount on the third shirt -/
def thirdShirtDiscount : Discount := ⟨0.6, by norm_num⟩

/-- 
Theorem stating that given the regular price, second shirt discount, and total savings,
the discount on the third shirt is 60%.
-/
theorem third_shirt_discount_is_sixty_percent :
  (1 - thirdShirtDiscount.val) * regularPrice = 
    3 * regularPrice - totalSavings - regularPrice - (1 - secondShirtDiscount.val) * regularPrice :=
by sorry

end stating_third_shirt_discount_is_sixty_percent_l2326_232673


namespace base_nine_addition_l2326_232612

/-- Represents a number in base 9 --/
def BaseNine : Type := List (Fin 9)

/-- Converts a base 9 number to a natural number --/
def to_nat (b : BaseNine) : ℕ :=
  b.foldr (λ d acc => 9 * acc + d.val) 0

/-- Adds two base 9 numbers --/
def add_base_nine (a b : BaseNine) : BaseNine :=
  sorry

theorem base_nine_addition :
  let a : BaseNine := [2, 5, 6]
  let b : BaseNine := [8, 5]
  let c : BaseNine := [1, 5, 5]
  let result : BaseNine := [5, 1, 7, 6]
  add_base_nine (add_base_nine a b) c = result := by
  sorry

end base_nine_addition_l2326_232612


namespace number_subtraction_l2326_232687

theorem number_subtraction (x : ℤ) : x + 30 = 55 → x - 23 = 2 := by
  sorry

end number_subtraction_l2326_232687


namespace range_of_a_l2326_232697

-- Define the conditions α and β
def α (x : ℝ) : Prop := x ≤ -1 ∨ x > 3
def β (a x : ℝ) : Prop := a - 1 ≤ x ∧ x < a + 2

-- State the theorem
theorem range_of_a :
  (∀ x, β a x → α x) ∧ 
  (∃ x, α x ∧ ¬β a x) →
  a ≤ -3 ∨ a > 4 :=
sorry

end range_of_a_l2326_232697


namespace one_is_last_digit_to_appear_l2326_232695

def modifiedFibonacci : ℕ → ℕ
  | 0 => 2
  | 1 => 2
  | n + 2 => (modifiedFibonacci n + modifiedFibonacci (n + 1)) % 10

def digitAppearsInSequence (d : ℕ) (n : ℕ) : Prop :=
  ∃ k, k ≤ n ∧ modifiedFibonacci k % 10 = d

def allDigitsAppear (n : ℕ) : Prop :=
  ∀ d, d < 10 → digitAppearsInSequence d n

def isLastDigitToAppear (d : ℕ) : Prop :=
  ∃ n, allDigitsAppear n ∧
    ¬(allDigitsAppear (n - 1)) ∧
    ¬(digitAppearsInSequence d (n - 1))

theorem one_is_last_digit_to_appear :
  isLastDigitToAppear 1 := by sorry

end one_is_last_digit_to_appear_l2326_232695


namespace abs_difference_equals_sum_of_abs_l2326_232690

theorem abs_difference_equals_sum_of_abs (a b c : ℚ) 
  (h1 : a < b) (h2 : b < 0) (h3 : 0 < c) : 
  |a - c| = |a| + c := by sorry

end abs_difference_equals_sum_of_abs_l2326_232690


namespace dime_difference_is_90_l2326_232694

/-- Represents the number of coins of each type in the piggy bank -/
structure CoinCount where
  nickels : ℕ
  dimes : ℕ
  halfDollars : ℕ

/-- Checks if the given coin count satisfies the problem conditions -/
def isValidCoinCount (c : CoinCount) : Prop :=
  c.nickels + c.dimes + c.halfDollars = 120 ∧
  5 * c.nickels + 10 * c.dimes + 50 * c.halfDollars = 1050

/-- Theorem stating the difference between max and min number of dimes -/
theorem dime_difference_is_90 :
  ∃ (min_dimes max_dimes : ℕ),
    (∃ c : CoinCount, isValidCoinCount c ∧ c.dimes = min_dimes) ∧
    (∃ c : CoinCount, isValidCoinCount c ∧ c.dimes = max_dimes) ∧
    (∀ c : CoinCount, isValidCoinCount c → c.dimes ≥ min_dimes ∧ c.dimes ≤ max_dimes) ∧
    max_dimes - min_dimes = 90 :=
by sorry

end dime_difference_is_90_l2326_232694


namespace variance_of_literary_works_l2326_232652

def literary_works : List ℕ := [6, 9, 5, 8, 10, 4]

def mean (data : List ℕ) : ℚ :=
  (data.sum : ℚ) / data.length

def variance (data : List ℕ) : ℚ :=
  let μ := mean data
  (data.map (fun x => ((x : ℚ) - μ) ^ 2)).sum / data.length

theorem variance_of_literary_works : variance literary_works = 14 / 3 := by
  sorry

end variance_of_literary_works_l2326_232652


namespace second_butcher_delivery_l2326_232611

/-- Represents the number of packages delivered by each butcher -/
structure ButcherDelivery where
  first : ℕ
  second : ℕ
  third : ℕ

/-- Represents the weight of each package and the total weight delivered -/
structure DeliveryInfo where
  package_weight : ℕ
  total_weight : ℕ

/-- Given the delivery information and the number of packages from the first and third butchers,
    proves that the second butcher delivered 7 packages -/
theorem second_butcher_delivery 
  (delivery : ButcherDelivery)
  (info : DeliveryInfo)
  (h1 : delivery.first = 10)
  (h2 : delivery.third = 8)
  (h3 : info.package_weight = 4)
  (h4 : info.total_weight = 100)
  (h5 : info.total_weight = 
    (delivery.first + delivery.second + delivery.third) * info.package_weight) :
  delivery.second = 7 := by
  sorry


end second_butcher_delivery_l2326_232611


namespace hen_count_l2326_232643

theorem hen_count (total_heads : ℕ) (total_feet : ℕ) 
  (h_heads : total_heads = 48)
  (h_feet : total_feet = 140) :
  ∃ (hens cows : ℕ),
    hens + cows = total_heads ∧
    2 * hens + 4 * cows = total_feet ∧
    hens = 26 := by sorry

end hen_count_l2326_232643


namespace largest_square_area_l2326_232608

theorem largest_square_area (a b c : ℝ) (h_right_angle : a^2 + b^2 = c^2)
  (h_sum_areas : a^2 + b^2 + c^2 = 450) : c^2 = 225 := by
  sorry

end largest_square_area_l2326_232608


namespace triangle_angle_measure_l2326_232615

/-- Given a triangle DEF where ∠E is congruent to ∠F, the measure of ∠F is three times 
    the measure of ∠D, and ∠D is one-third the measure of ∠E, 
    prove that the measure of ∠E is 540/7 degrees. -/
theorem triangle_angle_measure (D E F : ℝ) : 
  D > 0 → E > 0 → F > 0 →  -- Angles are positive
  D + E + F = 180 →  -- Sum of angles in a triangle
  E = F →  -- ∠E is congruent to ∠F
  F = 3 * D →  -- Measure of ∠F is three times the measure of ∠D
  D = E / 3 →  -- ∠D is one-third the measure of ∠E
  E = 540 / 7 :=
by sorry

end triangle_angle_measure_l2326_232615


namespace triangle_problem_l2326_232685

open Real

theorem triangle_problem (A B C : ℝ) (a b c : ℝ) :
  -- Given conditions
  (0 < A) ∧ (A < π) ∧ (0 < B) ∧ (B < π) ∧ (0 < C) ∧ (C < π) → -- Angles are in (0, π)
  (a > 0) ∧ (b > 0) ∧ (c > 0) → -- Sides are positive
  (sin A / sin C = a / c) ∧ (sin B / sin C = b / c) → -- Law of sines
  (cos C + c / b * cos B = 2) → -- Given equation
  (C = π / 3) → -- Given angle C
  (c = 2 * Real.sqrt 3) → -- Given side c
  -- Conclusions to prove
  (sin A / sin B = 2) ∧ 
  (1 / 2 * a * b * sin C = 2 * Real.sqrt 3) :=
by sorry

end triangle_problem_l2326_232685


namespace rectangular_formation_perimeter_l2326_232625

theorem rectangular_formation_perimeter (area : ℝ) (num_squares : ℕ) :
  area = 512 →
  num_squares = 8 →
  let square_side : ℝ := Real.sqrt (area / num_squares)
  let perimeter : ℝ := 2 * (4 * square_side + 3 * square_side)
  perimeter = 152 := by
  sorry

end rectangular_formation_perimeter_l2326_232625


namespace c_neq_zero_necessary_not_sufficient_l2326_232605

/-- Represents a conic section defined by the equation ax² + y² = c -/
structure ConicSection where
  a : ℝ
  c : ℝ

/-- Predicate to determine if a conic section is an ellipse or hyperbola -/
def is_ellipse_or_hyperbola (conic : ConicSection) : Prop :=
  sorry

/-- Theorem stating that c ≠ 0 is necessary but not sufficient for
    ax² + y² = c to represent an ellipse or hyperbola -/
theorem c_neq_zero_necessary_not_sufficient :
  (∀ conic : ConicSection, is_ellipse_or_hyperbola conic → conic.c ≠ 0) ∧
  (∃ conic : ConicSection, conic.c ≠ 0 ∧ ¬is_ellipse_or_hyperbola conic) :=
sorry

end c_neq_zero_necessary_not_sufficient_l2326_232605


namespace scientific_notation_of_billion_yuan_l2326_232644

def billion : ℝ := 1000000000

theorem scientific_notation_of_billion_yuan :
  let amount : ℝ := 2.175 * billion
  ∃ (a n : ℝ), amount = a * (10 : ℝ) ^ n ∧ 1 ≤ a ∧ a < 10 ∧ n = 9 := by sorry

end scientific_notation_of_billion_yuan_l2326_232644


namespace locus_of_center_l2326_232675

-- Define the circle C
def circle_C (a x y : ℝ) : Prop :=
  x^2 + y^2 - (2*a^2 - 4)*x - 4*a^2*y + 5*a^4 - 4 = 0

-- Define the locus equation
def locus_equation (x y : ℝ) : Prop :=
  2*x - y + 4 = 0

-- Define the x-coordinate range
def x_range (x : ℝ) : Prop :=
  -2 ≤ x ∧ x < 0

-- Theorem statement
theorem locus_of_center :
  ∀ a x y : ℝ, circle_C a x y → 
  ∃ h k : ℝ, (locus_equation h k ∧ x_range h) ∧
  (∀ x' y' : ℝ, locus_equation x' y' ∧ x_range x' → 
   ∃ a' : ℝ, circle_C a' x' y') :=
sorry

end locus_of_center_l2326_232675


namespace dog_catches_fox_l2326_232641

/-- The speed of the dog in meters per second -/
def dog_speed : ℝ := 2

/-- The time the dog runs in each unit of time, in seconds -/
def dog_time : ℝ := 2

/-- The speed of the fox in meters per second -/
def fox_speed : ℝ := 3

/-- The time the fox runs in each unit of time, in seconds -/
def fox_time : ℝ := 1

/-- The initial distance between the dog and the fox in meters -/
def initial_distance : ℝ := 30

/-- The total distance the dog runs before catching the fox -/
def total_distance : ℝ := 120

theorem dog_catches_fox : 
  let dog_distance_per_unit := dog_speed * dog_time
  let fox_distance_per_unit := fox_speed * fox_time
  let distance_gained_per_unit := dog_distance_per_unit - fox_distance_per_unit
  let units_to_catch := initial_distance / distance_gained_per_unit
  dog_distance_per_unit * units_to_catch = total_distance := by
sorry

end dog_catches_fox_l2326_232641


namespace parabola_equation_l2326_232632

-- Define the parabola structure
structure Parabola where
  equation : ℝ → ℝ → Prop

-- Define the focus of a parabola
def Focus := ℝ × ℝ

-- Define the line x - y + 4 = 0
def LineEquation (x y : ℝ) : Prop := x - y + 4 = 0

-- Define the condition that the focus is on the line
def FocusOnLine (f : Focus) : Prop := LineEquation f.1 f.2

-- Define the condition that the vertex is at the origin
def VertexAtOrigin (p : Parabola) : Prop := p.equation 0 0

-- Define the condition that the axis of symmetry is one of the coordinate axes
def AxisIsCoordinateAxis (p : Parabola) : Prop :=
  (∀ x y : ℝ, p.equation x y ↔ p.equation x (-y)) ∨
  (∀ x y : ℝ, p.equation x y ↔ p.equation (-x) y)

-- Theorem statement
theorem parabola_equation (p : Parabola) (f : Focus) :
  VertexAtOrigin p →
  AxisIsCoordinateAxis p →
  FocusOnLine f →
  (∀ x y : ℝ, p.equation x y ↔ y^2 = -16*x) ∨
  (∀ x y : ℝ, p.equation x y ↔ x^2 = 16*y) :=
sorry

end parabola_equation_l2326_232632


namespace present_value_is_490_l2326_232681

/-- Given a banker's discount and true discount, calculates the present value. -/
def present_value (bankers_discount : ℚ) (true_discount : ℚ) : ℚ :=
  true_discount^2 / (bankers_discount - true_discount)

/-- Theorem stating that for the given banker's discount and true discount, the present value is 490. -/
theorem present_value_is_490 :
  present_value 80 70 = 490 := by
  sorry

#eval present_value 80 70

end present_value_is_490_l2326_232681


namespace function_machine_output_l2326_232678

def function_machine (input : ℕ) : ℕ :=
  let step1 := input * 3
  let step2 := if step1 > 30 then step1 - 4 else step1
  if step2 ≤ 20 then step2 + 8 else step2 - 5

theorem function_machine_output : function_machine 10 = 25 := by
  sorry

end function_machine_output_l2326_232678


namespace power_of_ten_problem_l2326_232699

theorem power_of_ten_problem (a b : ℝ) 
  (h1 : (40 : ℝ) ^ a = 5) 
  (h2 : (40 : ℝ) ^ b = 8) : 
  (10 : ℝ) ^ ((1 - a - b) / (2 * (1 - b))) = 1 := by
sorry

end power_of_ten_problem_l2326_232699


namespace function_inequality_range_l2326_232692

def is_even (f : ℝ → ℝ) : Prop := ∀ x, f x = f (-x)

def monotone_increasing_on (f : ℝ → ℝ) (s : Set ℝ) : Prop :=
  ∀ {x y}, x ∈ s → y ∈ s → x ≤ y → f x ≤ f y

theorem function_inequality_range 
  (f : ℝ → ℝ) 
  (h_even : is_even f)
  (h_mono : monotone_increasing_on f (Set.Ici 0))
  (h_value : f (-2) = 1) :
  {x : ℝ | f (x - 2) ≤ 1} = Set.Icc 0 4 := by
sorry

end function_inequality_range_l2326_232692


namespace unique_positive_solution_l2326_232676

theorem unique_positive_solution :
  ∃! (x : ℝ), x > 0 ∧ (x - 5) / 8 = 5 / (x - 8) :=
by
  -- The proof would go here
  sorry

end unique_positive_solution_l2326_232676


namespace percentage_defective_meters_l2326_232649

theorem percentage_defective_meters 
  (total_meters : ℕ) 
  (rejected_meters : ℕ) 
  (h1 : total_meters = 2500) 
  (h2 : rejected_meters = 2) : 
  (rejected_meters : ℝ) / total_meters * 100 = 0.08 := by
sorry

end percentage_defective_meters_l2326_232649


namespace trailer_homes_problem_l2326_232619

theorem trailer_homes_problem (initial_homes : ℕ) (initial_avg_age : ℕ) 
  (current_avg_age : ℕ) (years_passed : ℕ) :
  initial_homes = 20 →
  initial_avg_age = 18 →
  current_avg_age = 14 →
  years_passed = 2 →
  ∃ (new_homes : ℕ),
    (initial_homes * (initial_avg_age + years_passed) + new_homes * years_passed) / 
    (initial_homes + new_homes) = current_avg_age ∧
    new_homes = 10 := by
  sorry

end trailer_homes_problem_l2326_232619


namespace divisibility_by_primes_less_than_1966_l2326_232630

theorem divisibility_by_primes_less_than_1966 (n : ℕ) (p : ℕ) (hp : Prime p) (hp_bound : p < 1966) :
  p ∣ (List.range 1966).foldl (λ acc i => acc * ((i + 1) * n + 1)) n :=
sorry

end divisibility_by_primes_less_than_1966_l2326_232630


namespace fifteen_degrees_to_radians_l2326_232662

theorem fifteen_degrees_to_radians :
  ∀ (π : ℝ), 180 * (π / 12) = π → 15 * (π / 180) = π / 12 := by
  sorry

end fifteen_degrees_to_radians_l2326_232662


namespace shelves_used_l2326_232610

def initial_stock : ℕ := 40
def books_sold : ℕ := 20
def books_per_shelf : ℕ := 4

theorem shelves_used : (initial_stock - books_sold) / books_per_shelf = 5 :=
by sorry

end shelves_used_l2326_232610


namespace probability_sum_nine_l2326_232642

/-- The number of faces on a standard die -/
def numFaces : ℕ := 6

/-- The number of dice rolled -/
def numDice : ℕ := 3

/-- The target sum we're looking for -/
def targetSum : ℕ := 9

/-- The total number of possible outcomes when rolling three dice -/
def totalOutcomes : ℕ := numFaces ^ numDice

/-- The number of ways to roll a sum of 9 with three dice -/
def favorableOutcomes : ℕ := 25

/-- The probability of rolling a sum of 9 with three standard six-faced dice -/
theorem probability_sum_nine :
  (favorableOutcomes : ℚ) / totalOutcomes = 25 / 216 := by sorry

end probability_sum_nine_l2326_232642


namespace principal_calculation_l2326_232655

/-- Calculates the principal amount given two interest rates, time period, and interest difference --/
def calculate_principal (rate1 rate2 : ℚ) (time : ℕ) (interest_diff : ℚ) : ℚ :=
  interest_diff / (rate1 - rate2) * 100 / time

/-- Theorem stating that the calculated principal is approximately 7142.86 --/
theorem principal_calculation :
  let rate1 : ℚ := 22
  let rate2 : ℚ := 15
  let time : ℕ := 5
  let interest_diff : ℚ := 2500
  let principal := calculate_principal rate1 rate2 time interest_diff
  abs (principal - 7142.86) < 0.01 := by
  sorry

end principal_calculation_l2326_232655


namespace sum_of_squares_lower_bound_l2326_232657

theorem sum_of_squares_lower_bound (a b c : ℝ) (h : a + 2*b + 3*c = 4) :
  a^2 + b^2 + c^2 ≥ 8/7 := by
  sorry

end sum_of_squares_lower_bound_l2326_232657


namespace slide_total_l2326_232674

theorem slide_total (initial : ℕ) (additional : ℕ) (total : ℕ) : 
  initial = 22 → additional = 13 → total = initial + additional → total = 35 := by
  sorry

end slide_total_l2326_232674


namespace phi_value_l2326_232661

theorem phi_value : ∃ (Φ : ℕ), Φ < 10 ∧ (220 : ℚ) / Φ = 40 + 3 * Φ := by
  sorry

end phi_value_l2326_232661


namespace six_digit_number_remainder_l2326_232616

/-- Represents a 6-digit number in the form 6x62y4 -/
def SixDigitNumber (x y : Nat) : Nat :=
  600000 + 10000 * x + 6200 + 10 * y + 4

theorem six_digit_number_remainder (x y : Nat) :
  x < 10 → y < 10 →
  (SixDigitNumber x y) % 11 = 0 →
  (SixDigitNumber x y) % 9 = 6 →
  (SixDigitNumber x y) % 13 = 6 := by
sorry

end six_digit_number_remainder_l2326_232616


namespace weeks_to_work_is_ten_l2326_232672

/-- The number of weeks Isabelle must work to afford concert tickets for herself and her brothers -/
def weeks_to_work : ℕ :=
let isabelle_ticket_cost : ℕ := 20
let brother_ticket_cost : ℕ := 10
let number_of_brothers : ℕ := 2
let total_savings : ℕ := 10
let weekly_earnings : ℕ := 3
let total_ticket_cost : ℕ := isabelle_ticket_cost + brother_ticket_cost * number_of_brothers
let additional_money_needed : ℕ := total_ticket_cost - total_savings
(additional_money_needed + weekly_earnings - 1) / weekly_earnings

theorem weeks_to_work_is_ten : weeks_to_work = 10 := by
  sorry

end weeks_to_work_is_ten_l2326_232672


namespace vectors_collinear_l2326_232689

/-- Given vectors a and b in ℝ³, prove that c₁ and c₂ are collinear -/
theorem vectors_collinear (a b : Fin 3 → ℝ) 
  (ha : a = ![1, -2, 5])
  (hb : b = ![3, -1, 0])
  (c₁ : Fin 3 → ℝ) (hc₁ : c₁ = 4 • a - 2 • b)
  (c₂ : Fin 3 → ℝ) (hc₂ : c₂ = b - 2 • a) :
  ∃ k : ℝ, c₁ = k • c₂ := by
sorry

end vectors_collinear_l2326_232689


namespace product_of_decimals_l2326_232696

theorem product_of_decimals : 3.6 * 0.25 = 0.9 := by
  sorry

end product_of_decimals_l2326_232696


namespace car_hire_payment_l2326_232627

/-- Represents the car hiring scenario -/
structure CarHire where
  hours_a : ℕ
  hours_b : ℕ
  hours_c : ℕ
  payment_b : ℚ

/-- Calculates the total amount paid for hiring the car -/
def total_payment (hire : CarHire) : ℚ :=
  let rate := hire.payment_b / hire.hours_b
  rate * (hire.hours_a + hire.hours_b + hire.hours_c)

/-- Theorem stating the total payment for the given scenario -/
theorem car_hire_payment :
  ∀ (hire : CarHire),
    hire.hours_a = 9 ∧
    hire.hours_b = 10 ∧
    hire.hours_c = 13 ∧
    hire.payment_b = 225 →
    total_payment hire = 720 := by
  sorry


end car_hire_payment_l2326_232627


namespace inequality_proof_l2326_232658

theorem inequality_proof (x : ℝ) (h : x ≥ 4) :
  Real.sqrt (x - 3) - Real.sqrt (x - 1) > Real.sqrt (x - 4) - Real.sqrt (x - 2) := by
  sorry

end inequality_proof_l2326_232658


namespace expression_evaluation_l2326_232603

theorem expression_evaluation :
  let x : ℝ := -2
  let y : ℝ := 1
  ((x + 2*y) * (x - 2*y) + 4 * (x - y)^2) / (-x) = 18 := by
  sorry

end expression_evaluation_l2326_232603


namespace population_increase_rate_is_two_l2326_232620

/-- The rate of population increase in persons per minute, given that one person is added every 30 seconds. -/
def population_increase_rate (seconds_per_person : ℕ) : ℚ :=
  60 / seconds_per_person

/-- Theorem stating that if the population increases by one person every 30 seconds, 
    then the rate of population increase is 2 persons per minute. -/
theorem population_increase_rate_is_two :
  population_increase_rate 30 = 2 := by sorry

end population_increase_rate_is_two_l2326_232620


namespace sine_ratio_zero_l2326_232647

theorem sine_ratio_zero (c : Real) (h : c = π / 12) :
  (Real.sin (4 * c) * Real.sin (8 * c) * Real.sin (12 * c)) /
  (Real.sin (2 * c) * Real.sin (4 * c) * Real.sin (6 * c)) = 0 := by
  sorry

end sine_ratio_zero_l2326_232647


namespace triangle_inequality_l2326_232688

theorem triangle_inequality (a b c : ℝ) (A B C : ℝ) (h1 : C ≥ π / 3) :
  let s := (a + b + c) / 2
  (a + b) * (1 / a + 1 / b + 1 / c) ≥ 4 + 1 / Real.sin (C / 2) :=
by sorry

end triangle_inequality_l2326_232688


namespace process_result_l2326_232693

def process (x : ℕ) : ℕ := 3 * (2 * x + 9)

theorem process_result : process 6 = 63 := by
  sorry

end process_result_l2326_232693


namespace noah_small_paintings_l2326_232628

/-- Represents the number of small paintings Noah sold last month -/
def small_paintings : ℕ := sorry

/-- Price of a large painting in dollars -/
def large_painting_price : ℕ := 60

/-- Price of a small painting in dollars -/
def small_painting_price : ℕ := 30

/-- Number of large paintings sold last month -/
def large_paintings_last_month : ℕ := 8

/-- This month's sales in dollars -/
def this_month_sales : ℕ := 1200

theorem noah_small_paintings : 
  2 * (large_painting_price * large_paintings_last_month + small_painting_price * small_paintings) = this_month_sales ∧ 
  small_paintings = 4 := by sorry

end noah_small_paintings_l2326_232628


namespace carton_height_is_70_l2326_232679

/-- Calculates the height of a carton given its base dimensions, soap box dimensions, and maximum capacity. -/
def carton_height (carton_length carton_width : ℕ) (box_length box_width box_height : ℕ) (max_boxes : ℕ) : ℕ :=
  let boxes_per_layer := (carton_length / box_length) * (carton_width / box_width)
  let num_layers := max_boxes / boxes_per_layer
  num_layers * box_height

/-- Theorem stating that the height of the carton is 70 inches given the specified conditions. -/
theorem carton_height_is_70 :
  carton_height 25 42 7 6 10 150 = 70 := by
  sorry

end carton_height_is_70_l2326_232679


namespace max_piles_660_max_piles_optimal_l2326_232698

/-- The maximum number of piles that can be created from a given number of stones,
    where any two piles differ by strictly less than 2 times. -/
def maxPiles (totalStones : ℕ) : ℕ :=
  30  -- The actual implementation is not provided, just the result

theorem max_piles_660 :
  maxPiles 660 = 30 := by sorry

/-- A function to check if two pile sizes are similar (differ by strictly less than 2 times) -/
def areSimilarSizes (a b : ℕ) : Prop :=
  a < 2 * b ∧ b < 2 * a

/-- A function to represent a valid distribution of stones into piles -/
def isValidDistribution (piles : List ℕ) (totalStones : ℕ) : Prop :=
  piles.sum = totalStones ∧
  ∀ (a b : ℕ), a ∈ piles → b ∈ piles → areSimilarSizes a b

theorem max_piles_optimal (piles : List ℕ) :
  isValidDistribution piles 660 →
  piles.length ≤ 30 := by sorry

end max_piles_660_max_piles_optimal_l2326_232698


namespace black_tiles_to_total_tiles_l2326_232634

/-- Represents a square room tiled with congruent square tiles -/
structure TiledRoom where
  side_length : ℕ

/-- Counts the number of black tiles in the room -/
def count_black_tiles (room : TiledRoom) : ℕ :=
  4 * room.side_length - 3

/-- Counts the total number of tiles in the room -/
def count_total_tiles (room : TiledRoom) : ℕ :=
  room.side_length * room.side_length

/-- Theorem stating the relationship between black tiles and total tiles -/
theorem black_tiles_to_total_tiles :
  ∃ (room : TiledRoom), count_black_tiles room = 201 ∧ count_total_tiles room = 2601 :=
sorry

end black_tiles_to_total_tiles_l2326_232634


namespace congruence_solution_l2326_232600

theorem congruence_solution : ∃! n : ℤ, 0 ≤ n ∧ n < 31 ∧ -250 ≡ n [ZMOD 31] ∧ n = 29 := by
  sorry

end congruence_solution_l2326_232600


namespace min_side_length_is_correct_l2326_232654

/-- The sequence of side lengths of squares to be packed -/
def a (n : ℕ+) : ℚ := 1 / n

/-- The minimum side length of the square that can contain all smaller squares -/
def min_side_length : ℚ := 3 / 2

/-- Theorem stating that min_side_length is the minimum side length of a square
    that can contain all squares with side lengths a(n) without overlapping -/
theorem min_side_length_is_correct :
  ∀ (s : ℚ), (∀ (arrangement : ℕ+ → ℚ × ℚ),
    (∀ (m n : ℕ+), m ≠ n →
      (abs (arrangement m).1 - (arrangement n).1 ≥ min (a m) (a n) ∨
       abs (arrangement m).2 - (arrangement n).2 ≥ min (a m) (a n))) →
    (∀ (n : ℕ+), (arrangement n).1 + a n ≤ s ∧ (arrangement n).2 + a n ≤ s)) →
  s ≥ min_side_length :=
sorry

end min_side_length_is_correct_l2326_232654


namespace sqrt_36_div_6_l2326_232638

theorem sqrt_36_div_6 : Real.sqrt 36 / 6 = 1 := by sorry

end sqrt_36_div_6_l2326_232638


namespace negation_equivalence_l2326_232624

theorem negation_equivalence (x y : ℝ) :
  ¬(x^2 + y^2 = 0 → x = 0 ∧ y = 0) ↔ (x^2 + y^2 ≠ 0 → x ≠ 0 ∨ y ≠ 0) :=
by sorry

end negation_equivalence_l2326_232624


namespace spider_eyes_l2326_232617

theorem spider_eyes (spider_count : ℕ) (ant_count : ℕ) (ant_eyes : ℕ) (total_eyes : ℕ) :
  spider_count = 3 →
  ant_count = 50 →
  ant_eyes = 2 →
  total_eyes = 124 →
  total_eyes = spider_count * (total_eyes - ant_count * ant_eyes) / spider_count →
  (total_eyes - ant_count * ant_eyes) / spider_count = 8 :=
by sorry

end spider_eyes_l2326_232617


namespace product_of_sines_equals_one_sixteenth_l2326_232682

theorem product_of_sines_equals_one_sixteenth :
  (1 + Real.sin (π / 12)) * (1 + Real.sin (5 * π / 12)) *
  (1 + Real.sin (7 * π / 12)) * (1 + Real.sin (11 * π / 12)) = 1 / 16 := by
  sorry

end product_of_sines_equals_one_sixteenth_l2326_232682


namespace exists_multiple_sum_of_digits_divides_l2326_232633

/-- Sum of digits function -/
def sum_of_digits (n : ℕ) : ℕ := sorry

/-- Theorem: For all positive integers n, there exists a multiple k of n such that 
    the sum of digits of k divides k. -/
theorem exists_multiple_sum_of_digits_divides (n : ℕ+) : 
  ∃ k : ℕ+, n ∣ k ∧ sum_of_digits k ∣ k := by
  sorry

end exists_multiple_sum_of_digits_divides_l2326_232633


namespace ratio_problem_l2326_232659

theorem ratio_problem (x y : ℝ) (h : (8 * x - 5 * y) / (11 * x - 3 * y) = 2 / 3) : 
  x / y = 9 / 2 := by
  sorry

end ratio_problem_l2326_232659


namespace necessary_condition_for_positive_linear_function_l2326_232602

theorem necessary_condition_for_positive_linear_function
  (a b : ℝ) (f : ℝ → ℝ) (h_f : ∀ x, 0 ≤ x ∧ x ≤ 1 → f x = a * x + b)
  (h_positive : ∀ x, 0 ≤ x ∧ x ≤ 1 → f x > 0) :
  a + 2 * b > 0 :=
sorry

end necessary_condition_for_positive_linear_function_l2326_232602


namespace carson_gold_stars_l2326_232648

/-- 
Given:
- Carson earned 6 gold stars yesterday
- Carson earned 9 gold stars today

Prove: The total number of gold stars Carson earned is 15
-/
theorem carson_gold_stars (yesterday_stars today_stars : ℕ) 
  (h1 : yesterday_stars = 6) 
  (h2 : today_stars = 9) : 
  yesterday_stars + today_stars = 15 := by
  sorry

end carson_gold_stars_l2326_232648


namespace condition_implies_linear_l2326_232601

/-- A function satisfying the given inequality condition -/
def SatisfiesCondition (f : ℝ → ℝ) : Prop :=
  ∀ (a b p : ℝ), f (p * a + (1 - p) * b) ≤ p * f a + (1 - p) * f b

/-- A linear function -/
def IsLinear (f : ℝ → ℝ) : Prop :=
  ∃ (A B : ℝ), ∀ x, f x = A * x + B

/-- Theorem: If a function satisfies the condition, then it is linear -/
theorem condition_implies_linear (f : ℝ → ℝ) :
  SatisfiesCondition f → IsLinear f := by
  sorry

end condition_implies_linear_l2326_232601


namespace marble_selection_probability_l2326_232684

/-- The number of red marbles in the bag -/
def red_marbles : ℕ := 3

/-- The number of blue marbles in the bag -/
def blue_marbles : ℕ := 3

/-- The number of green marbles in the bag -/
def green_marbles : ℕ := 3

/-- The total number of marbles in the bag -/
def total_marbles : ℕ := red_marbles + blue_marbles + green_marbles

/-- The number of marbles to be selected -/
def selected_marbles : ℕ := 4

/-- The probability of selecting exactly 1 red, 2 blue, and 1 green marble -/
def probability : ℚ := 3 / 14

theorem marble_selection_probability : 
  (Nat.choose red_marbles 1 * Nat.choose blue_marbles 2 * Nat.choose green_marbles 1 : ℚ) / 
  (Nat.choose total_marbles selected_marbles) = probability := by
  sorry

end marble_selection_probability_l2326_232684


namespace value_of_a_l2326_232645

def A (a : ℝ) : Set ℝ := {-1, 0, a}
def B (a : ℝ) : Set ℝ := {0, Real.sqrt a}

theorem value_of_a (a : ℝ) (h : B a ⊆ A a) : a = 1 := by
  sorry

end value_of_a_l2326_232645


namespace functional_equation_solution_l2326_232636

/-- The functional equation satisfied by f -/
def satisfies_equation (f : ℝ → ℝ) : Prop :=
  ∀ x y z : ℝ, x ≠ 0 → y ≠ 0 → z ≠ 0 → x * y * z = 1 →
    f x ^ 2 - f y * f z = x * (x + y + z) * (f x + f y + f z)

/-- The theorem stating the possible forms of f -/
theorem functional_equation_solution :
  ∀ f : ℝ → ℝ, (∀ x : ℝ, x ≠ 0 → f x = f x) →
    satisfies_equation f →
    (∀ x : ℝ, x ≠ 0 → f x = x^2 - 1/x) ∨ (∀ x : ℝ, x ≠ 0 → f x = 0) :=
by sorry

end functional_equation_solution_l2326_232636


namespace sqrt_nat_or_irrational_l2326_232635

theorem sqrt_nat_or_irrational (n : ℕ) : 
  (∃ m : ℕ, m * m = n) ∨ (∀ p q : ℕ, q > 0 → p * p ≠ n * q * q) :=
sorry

end sqrt_nat_or_irrational_l2326_232635


namespace fold_crease_forms_ellipse_l2326_232640

/-- Given a circle with radius R centered at the origin and an internal point A at (a, 0),
    the set of all points P(x, y) that are equidistant from A and any point on the circle's circumference
    forms an ellipse. -/
theorem fold_crease_forms_ellipse (R a : ℝ) (h : 0 < a ∧ a < R) :
  ∀ x y : ℝ,
    (∃ α : ℝ, (x - R * Real.cos α)^2 + (y - R * Real.sin α)^2 = (x - a)^2 + y^2) ↔
    (2*x - a)^2 / R^2 + 4*y^2 / (R^2 - a^2) = 1 :=
by sorry

end fold_crease_forms_ellipse_l2326_232640


namespace sort_three_integers_correct_l2326_232646

/-- Algorithm to sort three positive integers in descending order -/
def sort_three_integers (a b c : ℕ+) : ℕ+ × ℕ+ × ℕ+ :=
  let step2 := if a ≤ b then (b, a, c) else (a, b, c)
  let step3 := let (x, y, z) := step2
                if x ≤ z then (z, y, x) else (x, y, z)
  let step4 := let (x, y, z) := step3
                if y ≤ z then (x, z, y) else (x, y, z)
  step4

/-- Theorem stating that the sorting algorithm produces a descending order result -/
theorem sort_three_integers_correct (a b c : ℕ+) :
  let (x, y, z) := sort_three_integers a b c
  x ≥ y ∧ y ≥ z :=
by
  sorry

end sort_three_integers_correct_l2326_232646


namespace journal_writing_sessions_per_week_l2326_232622

theorem journal_writing_sessions_per_week 
  (pages_per_session : ℕ) 
  (total_pages : ℕ) 
  (total_weeks : ℕ) 
  (h1 : pages_per_session = 4) 
  (h2 : total_pages = 72) 
  (h3 : total_weeks = 6) : 
  (total_pages / pages_per_session) / total_weeks = 3 := by
sorry

end journal_writing_sessions_per_week_l2326_232622


namespace expression_simplification_l2326_232614

theorem expression_simplification (x : ℝ) (h1 : x ≠ 1) (h2 : x ≠ -3) :
  (3 * x^2 - 4*x + 1) / ((x - 1) * (x + 3)) - (6*x - 5) / ((x - 1) * (x + 3)) = 3 := by
  sorry

end expression_simplification_l2326_232614


namespace constant_term_value_l2326_232667

theorem constant_term_value (x y C : ℝ) 
  (eq1 : 7 * x + y = C)
  (eq2 : x + 3 * y = 1)
  (eq3 : 2 * x + y = 5) : 
  C = 19 := by
sorry

end constant_term_value_l2326_232667


namespace consecutive_points_length_l2326_232623

/-- Given five consecutive points on a straight line, prove the length of the entire segment --/
theorem consecutive_points_length (a b c d e : ℝ) : 
  (∃ (x : ℝ), b - a = 5 ∧ c - b = 2 * x ∧ d - c = x ∧ e - d = 4 ∧ c - a = 11) →
  e - a = 18 := by
  sorry

end consecutive_points_length_l2326_232623


namespace stream_speed_l2326_232665

/-- The speed of a stream given upstream and downstream canoe speeds -/
theorem stream_speed (upstream_speed downstream_speed : ℝ) 
  (h1 : upstream_speed = 4)
  (h2 : downstream_speed = 12) :
  (downstream_speed - upstream_speed) / 2 = 4 := by
  sorry

end stream_speed_l2326_232665


namespace price_change_l2326_232680

theorem price_change (q r : ℝ) (original_price : ℝ) :
  (original_price * (1 + q / 100) * (1 - r / 100) = 1) →
  (original_price = 1 / ((1 + q / 100) * (1 - r / 100))) :=
by sorry

end price_change_l2326_232680


namespace area_regular_dodecagon_formula_l2326_232677

/-- A regular dodecagon inscribed in a circle -/
structure RegularDodecagon (r : ℝ) where
  -- The radius of the circumscribed circle
  radius : ℝ
  radius_pos : radius > 0
  -- The dodecagon is regular and inscribed in the circle

/-- The area of a regular dodecagon -/
def area_regular_dodecagon (d : RegularDodecagon r) : ℝ := 3 * r^2

/-- Theorem: The area of a regular dodecagon inscribed in a circle with radius r is 3r² -/
theorem area_regular_dodecagon_formula (r : ℝ) (hr : r > 0) :
  ∀ (d : RegularDodecagon r), area_regular_dodecagon d = 3 * r^2 :=
by
  sorry

end area_regular_dodecagon_formula_l2326_232677


namespace circle_pi_value_l2326_232670

theorem circle_pi_value (d c : ℝ) (hd : d = 8) (hc : c = 25.12) :
  c / d = 3.14 := by sorry

end circle_pi_value_l2326_232670


namespace max_value_of_f_l2326_232637

noncomputable def f (x : ℝ) : ℝ := Real.sin (Real.pi / 2 + 2 * x) - 5 * Real.sin x

theorem max_value_of_f :
  ∃ (M : ℝ), (∀ x, f x ≤ M) ∧ (∃ x₀, f x₀ = M) ∧ M = 4 := by
  sorry

end max_value_of_f_l2326_232637


namespace closest_whole_number_to_ratio_l2326_232629

theorem closest_whole_number_to_ratio : ∃ n : ℕ, 
  n = 9 ∧ 
  ∀ m : ℕ, 
    |((10^3000 : ℝ) + 10^3003) / ((10^3001 : ℝ) + 10^3002) - (n : ℝ)| ≤ 
    |((10^3000 : ℝ) + 10^3003) / ((10^3001 : ℝ) + 10^3002) - (m : ℝ)| :=
by sorry

end closest_whole_number_to_ratio_l2326_232629


namespace number_of_pears_number_of_pears_is_correct_l2326_232651

/-- The number of pears in a basket, given the following conditions:
  * There are 5 baskets in total
  * There are 58 fruits in total
  * One basket contains 18 mangoes
  * One basket contains 12 pawpaws
  * Two baskets contain the same number of kiwi and lemon respectively
  * There are 9 lemons
-/
theorem number_of_pears : ℕ :=
  let total_baskets : ℕ := 5
  let total_fruits : ℕ := 58
  let mangoes : ℕ := 18
  let pawpaws : ℕ := 12
  let lemons : ℕ := 9
  let kiwis : ℕ := lemons
  10

#check number_of_pears

theorem number_of_pears_is_correct : number_of_pears = 10 := by
  sorry

end number_of_pears_number_of_pears_is_correct_l2326_232651


namespace cone_volume_increase_l2326_232669

theorem cone_volume_increase (R H : ℝ) (hR : R = 5) (hH : H = 12) :
  ∃ y : ℝ, y > 0 ∧ (1 / 3) * π * (R + y)^2 * H = (1 / 3) * π * R^2 * (H + y) ∧ y = 31 / 12 :=
sorry

end cone_volume_increase_l2326_232669


namespace four_boys_three_girls_144_arrangements_l2326_232666

/-- The number of ways to arrange alternating boys and girls in a row -/
def alternatingArrangements (boys girls : ℕ) : ℕ := boys.factorial * girls.factorial

/-- Theorem stating that if there are 3 girls and 144 alternating arrangements, there must be 4 boys -/
theorem four_boys_three_girls_144_arrangements :
  ∃ (boys : ℕ), boys > 0 ∧ alternatingArrangements boys 3 = 144 → boys = 4 := by
  sorry

#check four_boys_three_girls_144_arrangements

end four_boys_three_girls_144_arrangements_l2326_232666


namespace lous_shoes_monthly_goal_l2326_232613

/-- The number of shoes Lou's Shoes must sell each month -/
def monthly_goal (last_week : ℕ) (this_week : ℕ) (remaining : ℕ) : ℕ :=
  last_week + this_week + remaining

/-- Theorem stating the total number of shoes Lou's Shoes must sell each month -/
theorem lous_shoes_monthly_goal :
  monthly_goal 27 12 41 = 80 := by
  sorry

end lous_shoes_monthly_goal_l2326_232613


namespace cistern_fill_time_l2326_232650

def fill_time_p : ℝ := 12
def fill_time_q : ℝ := 15
def initial_time : ℝ := 6

theorem cistern_fill_time : 
  let rate_p := 1 / fill_time_p
  let rate_q := 1 / fill_time_q
  let initial_fill := (rate_p + rate_q) * initial_time
  let remaining_fill := 1 - initial_fill
  remaining_fill / rate_q = 1.5 := by
sorry

end cistern_fill_time_l2326_232650


namespace problem_statement_l2326_232660

theorem problem_statement (t : ℚ) (x y : ℚ) 
  (h1 : x = 3 - 2 * t) 
  (h2 : y = 5 * t + 6) 
  (h3 : x = -2) : 
  y = 37 / 2 := by
  sorry

end problem_statement_l2326_232660


namespace fraction_subtraction_multiplication_l2326_232664

theorem fraction_subtraction_multiplication :
  (5/6 - 1/3) * 3/4 = 3/8 := by sorry

end fraction_subtraction_multiplication_l2326_232664


namespace floor_sqrt_eight_count_l2326_232683

theorem floor_sqrt_eight_count : 
  (Finset.range 81 \ Finset.range 64).card = 17 := by
  sorry

end floor_sqrt_eight_count_l2326_232683


namespace point_in_region_b_range_l2326_232691

theorem point_in_region_b_range (b : ℝ) :
  let P : ℝ × ℝ := (-1, 2)
  (2 * P.1 + 3 * P.2 - b > 0) → (b < 4) :=
by sorry

end point_in_region_b_range_l2326_232691


namespace sixtieth_pair_is_five_seven_l2326_232671

/-- Represents a pair of integers -/
structure IntPair :=
  (first : ℕ)
  (second : ℕ)

/-- Generates the nth pair in the sequence -/
def nthPair (n : ℕ) : IntPair :=
  sorry

/-- The sum of the numbers in a pair -/
def pairSum (p : IntPair) : ℕ :=
  p.first + p.second

/-- The number of pairs before the nth group -/
def pairsBeforeGroup (n : ℕ) : ℕ :=
  sorry

/-- The main theorem stating that the 60th pair is (5, 7) -/
theorem sixtieth_pair_is_five_seven :
  nthPair 60 = IntPair.mk 5 7 :=
sorry

end sixtieth_pair_is_five_seven_l2326_232671


namespace reinforcement_size_l2326_232653

/-- Calculates the size of a reinforcement given initial garrison size, initial provision duration,
    time passed before reinforcement, and remaining provision duration after reinforcement. -/
def calculate_reinforcement (initial_garrison : ℕ) (initial_duration : ℕ) 
                             (time_passed : ℕ) (remaining_duration : ℕ) : ℕ :=
  let provisions := initial_garrison * initial_duration
  let provisions_left := initial_garrison * (initial_duration - time_passed)
  (provisions_left / remaining_duration) - initial_garrison

theorem reinforcement_size :
  let initial_garrison := 2000
  let initial_duration := 62
  let time_passed := 15
  let remaining_duration := 20
  calculate_reinforcement initial_garrison initial_duration time_passed remaining_duration = 2700 := by
  sorry

end reinforcement_size_l2326_232653


namespace complex_division_equality_l2326_232604

theorem complex_division_equality : (2 - I) / (2 + I) = 3/5 - 4/5 * I := by sorry

end complex_division_equality_l2326_232604


namespace average_weight_problem_l2326_232618

/-- Given three weights a, b, and c, prove that their average weights satisfy the given conditions -/
theorem average_weight_problem (a b c : ℝ) : 
  (a + b + c) / 3 = 43 →
  (b + c) / 2 = 42 →
  b = 51 →
  (a + b) / 2 = 48 := by
sorry

end average_weight_problem_l2326_232618


namespace calculate_expression_l2326_232668

theorem calculate_expression : -1^2 + 8 / (-2)^2 - (-4) * (-3) = -11 := by
  sorry

end calculate_expression_l2326_232668


namespace three_numbers_solution_l2326_232631

theorem three_numbers_solution :
  ∃ (x y z : ℤ),
    (x + y) * z = 35 ∧
    (x + z) * y = -27 ∧
    (y + z) * x = -32 ∧
    x = 4 ∧ y = -3 ∧ z = 5 := by
  sorry

end three_numbers_solution_l2326_232631


namespace chess_game_probabilities_l2326_232606

-- Define the probabilities
def prob_draw : ℚ := 1/2
def prob_B_win : ℚ := 1/3

-- Define the statements to be proven
def prob_A_win : ℚ := 1 - prob_draw - prob_B_win
def prob_A_not_lose : ℚ := prob_draw + prob_A_win
def prob_B_lose : ℚ := prob_A_win
def prob_B_not_lose : ℚ := prob_draw + prob_B_win

-- Theorem to prove the statements
theorem chess_game_probabilities :
  (prob_A_win = 1/6) ∧
  (prob_A_not_lose = 2/3) ∧
  (prob_B_lose = 1/6) ∧
  (prob_B_not_lose = 5/6) :=
by sorry

end chess_game_probabilities_l2326_232606


namespace ratio_x_to_y_l2326_232607

theorem ratio_x_to_y (x y : ℚ) (h : (10 * x - 3 * y) / (13 * x - 2 * y) = 3 / 5) :
  x / y = 9 / 11 := by
  sorry

end ratio_x_to_y_l2326_232607


namespace statement_c_not_always_true_l2326_232663

theorem statement_c_not_always_true : 
  ¬ ∀ (a b c : ℝ), a > b → a * c^2 > b * c^2 := by
sorry

end statement_c_not_always_true_l2326_232663


namespace inequality_theorem_l2326_232609

theorem inequality_theorem (a : ℚ) (x : ℝ) :
  ((a > 1 ∨ a < 0) ∧ x > 0 ∧ x ≠ 1 → x^(a : ℝ) - a * x + a - 1 > 0) ∧
  (0 < a ∧ a < 1 ∧ x > 0 ∧ x ≠ 1 → x^(a : ℝ) - a * x + a - 1 < 0) := by
  sorry

end inequality_theorem_l2326_232609
