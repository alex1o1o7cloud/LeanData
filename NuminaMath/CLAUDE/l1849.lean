import Mathlib

namespace NUMINAMATH_CALUDE_sequence_to_zero_l1849_184933

/-- A transformation that applies |x - α| to each element of a sequence -/
def transform (s : List ℝ) (α : ℝ) : List ℝ :=
  s.map (fun x => |x - α|)

/-- Predicate to check if all elements in a list are zero -/
def all_zero (s : List ℝ) : Prop :=
  s.all (fun x => x = 0)

theorem sequence_to_zero (n : ℕ) :
  ∀ (s : List ℝ), s.length = n →
  (∃ (transformations : List ℝ),
    transformations.length = n ∧
    all_zero (transformations.foldl transform s)) ∧
  (∀ (transformations : List ℝ),
    transformations.length < n →
    ¬ all_zero (transformations.foldl transform s)) :=
sorry

end NUMINAMATH_CALUDE_sequence_to_zero_l1849_184933


namespace NUMINAMATH_CALUDE_max_profit_plan_l1849_184923

/-- Represents the production plan for cars -/
structure CarProduction where
  a : ℕ  -- number of A type cars
  b : ℕ  -- number of B type cars

/-- Calculates the total cost of production -/
def total_cost (p : CarProduction) : ℕ := 30 * p.a + 40 * p.b

/-- Calculates the total revenue from sales -/
def total_revenue (p : CarProduction) : ℕ := 35 * p.a + 50 * p.b

/-- Calculates the profit from a production plan -/
def profit (p : CarProduction) : ℤ := total_revenue p - total_cost p

/-- Theorem stating that the maximum profit is achieved with 5 A type cars and 35 B type cars -/
theorem max_profit_plan :
  ∀ p : CarProduction,
    p.a + p.b = 40 →
    total_cost p ≤ 1550 →
    profit p ≥ 365 →
    profit p ≤ profit { a := 5, b := 35 } :=
by sorry

end NUMINAMATH_CALUDE_max_profit_plan_l1849_184923


namespace NUMINAMATH_CALUDE_complex_expression_equality_l1849_184919

theorem complex_expression_equality : ∀ (i : ℂ), i^2 = -1 →
  (2 + i) / (1 - i) - (1 - i) = -1/2 + 5/2 * i := by
  sorry

end NUMINAMATH_CALUDE_complex_expression_equality_l1849_184919


namespace NUMINAMATH_CALUDE_pure_imaginary_solutions_l1849_184952

theorem pure_imaginary_solutions :
  let f : ℂ → ℂ := λ x => x^4 - 5*x^3 + 10*x^2 - 50*x - 75
  ∀ x : ℂ, (∃ k : ℝ, x = k * I) → (f x = 0 ↔ x = Complex.I * Real.sqrt 10 ∨ x = -Complex.I * Real.sqrt 10) :=
by sorry

end NUMINAMATH_CALUDE_pure_imaginary_solutions_l1849_184952


namespace NUMINAMATH_CALUDE_largest_d_inequality_d_satisfies_inequality_d_is_largest_l1849_184969

theorem largest_d_inequality (d : ℝ) : 
  (d > 0 ∧ 
   ∀ (x y : ℝ), x ≥ 0 → y ≥ 0 → 
   Real.sqrt (x^2 + y^2) + d * |x - y| ≤ Real.sqrt (2 * (x + y))) → 
  d ≤ 1 / Real.sqrt 2 :=
by sorry

theorem d_satisfies_inequality : 
  ∀ (x y : ℝ), x ≥ 0 → y ≥ 0 → 
  Real.sqrt (x^2 + y^2) + (1 / Real.sqrt 2) * |x - y| ≤ Real.sqrt (2 * (x + y)) :=
by sorry

theorem d_is_largest : 
  ∀ (d : ℝ), d > 1 / Real.sqrt 2 → 
  ∃ (x y : ℝ), x ≥ 0 ∧ y ≥ 0 ∧ 
  Real.sqrt (x^2 + y^2) + d * |x - y| > Real.sqrt (2 * (x + y)) :=
by sorry

end NUMINAMATH_CALUDE_largest_d_inequality_d_satisfies_inequality_d_is_largest_l1849_184969


namespace NUMINAMATH_CALUDE_ultra_savings_interest_theorem_l1849_184951

/-- Represents the Ultra Savings Account investment scenario -/
structure UltraSavingsAccount where
  principal : ℝ
  rate : ℝ
  years : ℕ

/-- Calculates the final balance after compound interest -/
def finalBalance (account : UltraSavingsAccount) : ℝ :=
  account.principal * (1 + account.rate) ^ account.years

/-- Calculates the interest earned -/
def interestEarned (account : UltraSavingsAccount) : ℝ :=
  finalBalance account - account.principal

/-- Theorem stating that the interest earned is approximately $328.49 -/
theorem ultra_savings_interest_theorem (account : UltraSavingsAccount) 
  (h1 : account.principal = 1500)
  (h2 : account.rate = 0.02)
  (h3 : account.years = 10) : 
  ∃ ε > 0, |interestEarned account - 328.49| < ε :=
sorry

end NUMINAMATH_CALUDE_ultra_savings_interest_theorem_l1849_184951


namespace NUMINAMATH_CALUDE_four_correct_statements_l1849_184979

theorem four_correct_statements : 
  (∀ x : ℝ, Irrational x → ¬ (∃ q : ℚ, x = ↑q)) ∧ 
  ({x : ℝ | x^2 = 4} = {2, -2}) ∧
  ({x : ℝ | x^3 = x} = {-1, 0, 1}) ∧
  (∀ x : ℝ, ∃! p : ℝ, p = x) := by
  sorry

end NUMINAMATH_CALUDE_four_correct_statements_l1849_184979


namespace NUMINAMATH_CALUDE_accountant_total_amount_l1849_184905

/-- Calculates the total amount given to the accountant for festival allowance --/
def festival_allowance_total (staff_count : ℕ) (daily_rate : ℕ) (days : ℕ) (petty_cash : ℕ) : ℕ :=
  staff_count * daily_rate * days + petty_cash

/-- Theorem stating the total amount given to the accountant --/
theorem accountant_total_amount :
  festival_allowance_total 20 100 30 1000 = 61000 := by
  sorry

end NUMINAMATH_CALUDE_accountant_total_amount_l1849_184905


namespace NUMINAMATH_CALUDE_population_growth_problem_l1849_184946

theorem population_growth_problem (x y z : ℕ) : 
  (3/2)^x * (128/225)^y * (5/6)^z = 2 ↔ x = 4 ∧ y = 1 ∧ z = 2 := by
  sorry

end NUMINAMATH_CALUDE_population_growth_problem_l1849_184946


namespace NUMINAMATH_CALUDE_five_digit_divisible_by_nine_l1849_184953

theorem five_digit_divisible_by_nine :
  ∀ x : ℕ, x < 10 →
  (738 * 10 + x) * 10 + 5 ≡ 0 [MOD 9] ↔ x = 4 := by
  sorry

end NUMINAMATH_CALUDE_five_digit_divisible_by_nine_l1849_184953


namespace NUMINAMATH_CALUDE_yellow_or_blue_consecutive_rolls_l1849_184974

/-- A die with 12 sides and specific color distribution -/
structure Die :=
  (sides : Nat)
  (red : Nat)
  (yellow : Nat)
  (blue : Nat)
  (green : Nat)
  (total_eq : sides = red + yellow + blue + green)

/-- The probability of an event occurring -/
def probability (favorable : Nat) (total : Nat) : ℚ :=
  ↑favorable / ↑total

/-- The probability of two independent events both occurring -/
def probability_both (p1 : ℚ) (p2 : ℚ) : ℚ := p1 * p2

theorem yellow_or_blue_consecutive_rolls (d : Die) 
  (h : d.sides = 12 ∧ d.red = 5 ∧ d.yellow = 4 ∧ d.blue = 2 ∧ d.green = 1) : 
  probability_both 
    (probability (d.yellow + d.blue) d.sides) 
    (probability (d.yellow + d.blue) d.sides) = 1/4 := by
  sorry

end NUMINAMATH_CALUDE_yellow_or_blue_consecutive_rolls_l1849_184974


namespace NUMINAMATH_CALUDE_even_function_sum_l1849_184900

def is_even_function (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (-x) = f x

theorem even_function_sum (f : ℝ → ℝ) (h_even : is_even_function f) (h_value : f (-1) = 2) :
  f (-1) + f 1 = 4 := by
  sorry

end NUMINAMATH_CALUDE_even_function_sum_l1849_184900


namespace NUMINAMATH_CALUDE_restaurant_tax_calculation_l1849_184970

/-- Proves that the tax amount is $3 given the initial money, order costs, and change received --/
theorem restaurant_tax_calculation (lee_money : ℕ) (friend_money : ℕ) 
  (wings_cost : ℕ) (salad_cost : ℕ) (soda_cost : ℕ) (change : ℕ) : ℕ :=
by
  -- Define the given conditions
  have h1 : lee_money = 10 := by sorry
  have h2 : friend_money = 8 := by sorry
  have h3 : wings_cost = 6 := by sorry
  have h4 : salad_cost = 4 := by sorry
  have h5 : soda_cost = 1 := by sorry
  have h6 : change = 3 := by sorry

  -- Calculate the total initial money
  let total_money := lee_money + friend_money

  -- Calculate the cost before tax
  let cost_before_tax := wings_cost + salad_cost + 2 * soda_cost

  -- Calculate the total spent including tax
  let total_spent := total_money - change

  -- Calculate the tax
  let tax := total_spent - cost_before_tax

  -- Prove that the tax is 3
  exact 3

end NUMINAMATH_CALUDE_restaurant_tax_calculation_l1849_184970


namespace NUMINAMATH_CALUDE_distance_between_mum_and_turbo_l1849_184960

/-- The distance between Usain's mum and Turbo when Usain has run 100 meters -/
theorem distance_between_mum_and_turbo (usain_speed mum_speed turbo_speed : ℝ) : 
  usain_speed = 2 * mum_speed →
  mum_speed = 5 * turbo_speed →
  usain_speed > 0 →
  (100 / usain_speed) * mum_speed - (100 / usain_speed) * turbo_speed = 40 := by
  sorry

end NUMINAMATH_CALUDE_distance_between_mum_and_turbo_l1849_184960


namespace NUMINAMATH_CALUDE_inverse_f_58_l1849_184939

def f (x : ℝ) : ℝ := 2 * x^3 + 4

theorem inverse_f_58 : f⁻¹ 58 = 3 := by sorry

end NUMINAMATH_CALUDE_inverse_f_58_l1849_184939


namespace NUMINAMATH_CALUDE_smallest_divisor_l1849_184954

theorem smallest_divisor (N D : ℕ) (q1 q2 k : ℕ) : 
  N = D * q1 + 75 →
  N = 37 * q2 + 1 →
  D > 75 →
  D = 37 * k + 38 →
  112 ≤ D :=
by sorry

end NUMINAMATH_CALUDE_smallest_divisor_l1849_184954


namespace NUMINAMATH_CALUDE_product_218_5_base9_l1849_184976

/-- Convert a base-9 number to base-10 --/
def base9ToBase10 (n : ℕ) : ℕ := sorry

/-- Convert a base-10 number to base-9 --/
def base10ToBase9 (n : ℕ) : ℕ := sorry

/-- Multiply two base-9 numbers and return the result in base-9 --/
def multiplyBase9 (a b : ℕ) : ℕ :=
  base10ToBase9 (base9ToBase10 a * base9ToBase10 b)

theorem product_218_5_base9 :
  multiplyBase9 218 5 = 1204 := by sorry

end NUMINAMATH_CALUDE_product_218_5_base9_l1849_184976


namespace NUMINAMATH_CALUDE_quadratic_inequality_implications_l1849_184942

theorem quadratic_inequality_implications 
  (a b c t : ℝ) 
  (h1 : t > 1) 
  (h2 : ∀ x : ℝ, ax^2 + b*x + c > 0 ↔ 1 < x ∧ x < t) :
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ a*x₁^2 + (a-b)*x₁ - c = 0 ∧ a*x₂^2 + (a-b)*x₂ - c = 0) ∧
  (∀ x₁ x₂ : ℝ, a*x₁^2 + (a-b)*x₁ - c = 0 → a*x₂^2 + (a-b)*x₂ - c = 0 → |x₂ - x₁| > Real.sqrt 13) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_implications_l1849_184942


namespace NUMINAMATH_CALUDE_sequence_problem_l1849_184983

def arithmetic_sequence (a b : ℕ) (n : ℕ) : ℕ := a + b * (n - 1)

def geometric_sequence (b a : ℕ) (n : ℕ) : ℕ := b * a^(n - 1)

theorem sequence_problem (a b : ℕ) 
  (h1 : a > 1) 
  (h2 : b > 1) 
  (h3 : a < b) 
  (h4 : b * a < arithmetic_sequence a b 3) 
  (h5 : ∀ n : ℕ, n > 0 → ∃ m : ℕ, m > 0 ∧ geometric_sequence b a n = arithmetic_sequence a b m + 3) :
  a = 2 ∧ ∀ n : ℕ, arithmetic_sequence a b n = 5 * n - 3 :=
sorry

end NUMINAMATH_CALUDE_sequence_problem_l1849_184983


namespace NUMINAMATH_CALUDE_chloe_boxes_l1849_184916

/-- The number of scarves in each box -/
def scarves_per_box : ℕ := 2

/-- The number of mittens in each box -/
def mittens_per_box : ℕ := 6

/-- The total number of winter clothing pieces Chloe has -/
def total_clothing : ℕ := 32

/-- The number of boxes Chloe found -/
def boxes : ℕ := total_clothing / (scarves_per_box + mittens_per_box)

theorem chloe_boxes : boxes = 4 := by
  sorry

end NUMINAMATH_CALUDE_chloe_boxes_l1849_184916


namespace NUMINAMATH_CALUDE_frank_bakes_two_trays_per_day_l1849_184941

/-- The number of days Frank bakes cookies -/
def days : ℕ := 6

/-- The number of cookies Frank eats per day -/
def frankEatsPerDay : ℕ := 1

/-- The number of cookies Ted eats on the sixth day -/
def tedEats : ℕ := 4

/-- The number of cookies each tray makes -/
def cookiesPerTray : ℕ := 12

/-- The number of cookies left when Ted leaves -/
def cookiesLeft : ℕ := 134

/-- The number of trays Frank bakes per day -/
def traysPerDay : ℕ := 2

theorem frank_bakes_two_trays_per_day :
  traysPerDay * cookiesPerTray * days - 
  (frankEatsPerDay * days + tedEats) = cookiesLeft := by
  sorry

end NUMINAMATH_CALUDE_frank_bakes_two_trays_per_day_l1849_184941


namespace NUMINAMATH_CALUDE_only_sqrt_6_is_quadratic_radical_l1849_184995

-- Define what it means for an expression to be a quadratic radical
def is_quadratic_radical (x : ℝ) : Prop :=
  ∃ y : ℝ, y ≥ 0 ∧ x = Real.sqrt y

-- Theorem statement
theorem only_sqrt_6_is_quadratic_radical :
  is_quadratic_radical (Real.sqrt 6) ∧
  ¬is_quadratic_radical (Real.sqrt (-5)) ∧
  ¬is_quadratic_radical (8 ^ (1/3 : ℝ)) ∧
  ¬∀ a : ℝ, is_quadratic_radical (Real.sqrt a) :=
by sorry

end NUMINAMATH_CALUDE_only_sqrt_6_is_quadratic_radical_l1849_184995


namespace NUMINAMATH_CALUDE_room_breadth_is_five_meters_l1849_184990

/-- Given a building with 5 equal-area rooms, prove that the breadth of each room is 5 meters. -/
theorem room_breadth_is_five_meters 
  (num_rooms : ℕ) 
  (room_length : ℝ) 
  (room_height : ℝ) 
  (bricks_per_sqm : ℕ) 
  (bricks_for_floor : ℕ) :
  num_rooms = 5 →
  room_length = 4 →
  room_height = 2 →
  bricks_per_sqm = 17 →
  bricks_for_floor = 340 →
  ∃ (room_breadth : ℝ), room_breadth = 5 :=
by sorry

end NUMINAMATH_CALUDE_room_breadth_is_five_meters_l1849_184990


namespace NUMINAMATH_CALUDE_f_properties_l1849_184949

noncomputable def f (x : ℝ) : ℝ := Real.sin x ^ 4 + 2 * Real.sin x * Real.cos x - Real.cos x ^ 4

theorem f_properties :
  (∃ (p : ℝ), p > 0 ∧ ∀ (x : ℝ), f (x + p) = f x ∧ ∀ (q : ℝ), q > 0 ∧ (∀ (x : ℝ), f (x + q) = f x) → p ≤ q) ∧
  (∀ (x : ℝ), f x ≥ -2) ∧
  (∀ (x : ℝ), x ∈ Set.Icc 0 (Real.pi / 2) → ∀ (y : ℝ), y ∈ Set.Icc 0 (Real.pi / 2) → x < y → f x < f y) ∧
  (∀ (x : ℝ), x ∈ Set.Ioc (Real.pi / 2) Real.pi → ∀ (y : ℝ), y ∈ Set.Ioc (Real.pi / 2) Real.pi → x < y → f x > f y) :=
by sorry

end NUMINAMATH_CALUDE_f_properties_l1849_184949


namespace NUMINAMATH_CALUDE_translation_equivalence_l1849_184908

noncomputable def original_function (x : ℝ) : ℝ :=
  Real.sin (2 * x + Real.pi / 6)

noncomputable def translated_function (x : ℝ) : ℝ :=
  original_function (x + Real.pi / 6)

theorem translation_equivalence :
  ∀ x : ℝ, translated_function x = Real.cos (2 * x) := by
  sorry

end NUMINAMATH_CALUDE_translation_equivalence_l1849_184908


namespace NUMINAMATH_CALUDE_lucy_additional_distance_l1849_184972

/-- The length of the field in kilometers -/
def field_length : ℚ := 24

/-- The fraction of the field that Mary ran -/
def mary_fraction : ℚ := 3/8

/-- The fraction of Mary's distance that Edna ran -/
def edna_fraction : ℚ := 2/3

/-- The fraction of Edna's distance that Lucy ran -/
def lucy_fraction : ℚ := 5/6

/-- Mary's running distance in kilometers -/
def mary_distance : ℚ := field_length * mary_fraction

/-- Edna's running distance in kilometers -/
def edna_distance : ℚ := mary_distance * edna_fraction

/-- Lucy's running distance in kilometers -/
def lucy_distance : ℚ := edna_distance * lucy_fraction

theorem lucy_additional_distance :
  mary_distance - lucy_distance = 4 := by
  sorry

end NUMINAMATH_CALUDE_lucy_additional_distance_l1849_184972


namespace NUMINAMATH_CALUDE_geometric_sequence_sum_l1849_184922

/-- Represents the sum of the first n terms of a geometric sequence -/
def S (n : ℕ) : ℝ := sorry

/-- The statement to prove -/
theorem geometric_sequence_sum :
  (S 4 = 4) → (S 8 = 12) → (S 16 = 60) := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_sum_l1849_184922


namespace NUMINAMATH_CALUDE_badminton_racket_cost_proof_l1849_184988

/-- The cost price of a badminton racket satisfying the given conditions -/
def badminton_racket_cost : ℝ := 125

/-- The markup percentage applied to the cost price -/
def markup_percentage : ℝ := 0.4

/-- The discount percentage applied to the marked price -/
def discount_percentage : ℝ := 0.2

/-- The profit made on the sale -/
def profit : ℝ := 15

theorem badminton_racket_cost_proof :
  (badminton_racket_cost * (1 + markup_percentage) * (1 - discount_percentage) =
   badminton_racket_cost + profit) := by
  sorry

end NUMINAMATH_CALUDE_badminton_racket_cost_proof_l1849_184988


namespace NUMINAMATH_CALUDE_triangle_height_inequality_l1849_184965

/-- For a triangle with side lengths a ≤ b ≤ c, heights h_a, h_b, h_c, 
    circumradius R, and semiperimeter p, the following inequality holds. -/
theorem triangle_height_inequality 
  (a b c : ℝ) (h_a h_b h_c R p : ℝ) 
  (h_order : a ≤ b ∧ b ≤ c) 
  (h_positive : a > 0 ∧ b > 0 ∧ c > 0 ∧ R > 0 ∧ p > 0) 
  (h_heights : h_a = 2 * (p - a) * (p - b) * (p - c) / (a * b * c) ∧ 
               h_b = 2 * (p - a) * (p - b) * (p - c) / (a * b * c) ∧ 
               h_c = 2 * (p - a) * (p - b) * (p - c) / (a * b * c))
  (h_semiperimeter : p = (a + b + c) / 2)
  (h_circumradius : R = a * b * c / (4 * (p - a) * (p - b) * (p - c))) :
  h_a + h_b + h_c ≤ 3 * b * (a^2 + a*c + c^2) / (4 * p * R) :=
sorry

end NUMINAMATH_CALUDE_triangle_height_inequality_l1849_184965


namespace NUMINAMATH_CALUDE_c_rent_share_l1849_184971

/-- Represents the usage of the pasture by a person -/
structure Usage where
  oxen : ℕ
  months : ℕ

/-- Calculates the ox-months for a given usage -/
def oxMonths (u : Usage) : ℕ := u.oxen * u.months

/-- Represents the rental situation of the pasture -/
structure PastureRental where
  a : Usage
  b : Usage
  c : Usage
  totalRent : ℕ

/-- Calculates the total ox-months for all users -/
def totalOxMonths (r : PastureRental) : ℕ :=
  oxMonths r.a + oxMonths r.b + oxMonths r.c

/-- Calculates the share of rent for a given usage -/
def rentShare (r : PastureRental) (u : Usage) : ℚ :=
  (oxMonths u : ℚ) / (totalOxMonths r : ℚ) * (r.totalRent : ℚ)

theorem c_rent_share (r : PastureRental) : 
  r.a = { oxen := 10, months := 7 } →
  r.b = { oxen := 12, months := 5 } →
  r.c = { oxen := 15, months := 3 } →
  r.totalRent = 245 →
  rentShare r r.c = 63 := by
  sorry

end NUMINAMATH_CALUDE_c_rent_share_l1849_184971


namespace NUMINAMATH_CALUDE_max_gaming_average_l1849_184967

theorem max_gaming_average (wednesday_hours : ℝ) (thursday_hours : ℝ) (tom_hours : ℝ) (fred_hours : ℝ) (additional_time : ℝ) :
  wednesday_hours = 2 →
  thursday_hours = 2 →
  tom_hours = 4 →
  fred_hours = 6 →
  additional_time = 0.5 →
  let total_hours := wednesday_hours + thursday_hours + max tom_hours fred_hours + additional_time
  let days := 3
  let average_hours := total_hours / days
  average_hours = 3.5 := by
sorry

end NUMINAMATH_CALUDE_max_gaming_average_l1849_184967


namespace NUMINAMATH_CALUDE_multiplication_sum_l1849_184943

theorem multiplication_sum (a b : ℕ) : 
  a ≤ 9 → b ≤ 9 → (30 * a + a) * (10 * b + 4) = 126 → a + b = 7 := by
  sorry

end NUMINAMATH_CALUDE_multiplication_sum_l1849_184943


namespace NUMINAMATH_CALUDE_age_difference_l1849_184911

theorem age_difference (A B C : ℕ) (h1 : C = A - 18) : A + B - (B + C) = 18 := by
  sorry

end NUMINAMATH_CALUDE_age_difference_l1849_184911


namespace NUMINAMATH_CALUDE_smallest_prime_divisor_of_sum_l1849_184973

theorem smallest_prime_divisor_of_sum : 
  ∃ (p : ℕ), p.Prime ∧ p ∣ (3^19 + 11^13) ∧ ∀ (q : ℕ), q.Prime → q ∣ (3^19 + 11^13) → p ≤ q :=
by
  sorry

end NUMINAMATH_CALUDE_smallest_prime_divisor_of_sum_l1849_184973


namespace NUMINAMATH_CALUDE_div_sqrt_three_equals_sqrt_three_l1849_184997

theorem div_sqrt_three_equals_sqrt_three : 3 / Real.sqrt 3 = Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_div_sqrt_three_equals_sqrt_three_l1849_184997


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l1849_184947

def arithmetic_sequence (a : ℕ → ℤ) : Prop :=
  ∃ d : ℤ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_sum (a : ℕ → ℤ) :
  arithmetic_sequence a →
  a 5 = 3 →
  a 6 = -2 →
  a 3 + a 4 + a 5 + a 6 + a 7 + a 8 = 0 :=
by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l1849_184947


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_range_l1849_184985

-- Define the quadratic function
def f (b : ℝ) (x : ℝ) : ℝ := x^2 - b*x + 1

-- State the theorem
theorem quadratic_inequality_solution_range (b : ℝ) (x₁ x₂ : ℝ) 
  (h1 : ∀ x, f b x > 0 ↔ x < x₁ ∨ x > x₂)
  (h2 : x₁ < 1)
  (h3 : x₂ > 1) :
  b > 2 ∧ b ∈ Set.Ioi 2 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_range_l1849_184985


namespace NUMINAMATH_CALUDE_ellipse_equation_l1849_184977

/-- An ellipse with center at the origin, foci on the x-axis, and point P(2, √3) on the ellipse. 
    The distances |PF₁|, |F₁F₂|, and |PF₂| form an arithmetic sequence. -/
structure Ellipse where
  a : ℝ
  b : ℝ
  c : ℝ
  h_positive : 0 < b ∧ b < a
  h_foci : c^2 = a^2 - b^2
  h_point_on_ellipse : 4 / a^2 + 3 / b^2 = 1
  h_arithmetic_sequence : ∃ (d : ℝ), |2 - c| + d = 2*c ∧ 2*c + d = |2 + c|

/-- The equation of the ellipse is x²/8 + y²/6 = 1 -/
theorem ellipse_equation (e : Ellipse) : e.a^2 = 8 ∧ e.b^2 = 6 := by
  sorry

end NUMINAMATH_CALUDE_ellipse_equation_l1849_184977


namespace NUMINAMATH_CALUDE_gcd_of_squares_plus_one_l1849_184935

theorem gcd_of_squares_plus_one (n : ℕ+) : 
  Nat.gcd (n.val^2 + 1) ((n.val + 1)^2 + 1) = 1 :=
by sorry

end NUMINAMATH_CALUDE_gcd_of_squares_plus_one_l1849_184935


namespace NUMINAMATH_CALUDE_figure_to_square_l1849_184940

/-- A figure that can be cut into three parts -/
structure Figure where
  area : ℕ

/-- Proves that a figure with an area of 57 unit squares can be assembled into a square -/
theorem figure_to_square (f : Figure) (h : f.area = 57) : 
  ∃ (s : ℝ), s^2 = f.area := by
  sorry

end NUMINAMATH_CALUDE_figure_to_square_l1849_184940


namespace NUMINAMATH_CALUDE_no_three_digit_odd_sum_30_l1849_184913

def is_three_digit (n : ℕ) : Prop := 100 ≤ n ∧ n ≤ 999

def digit_sum (n : ℕ) : ℕ :=
  (n / 100) + ((n / 10) % 10) + (n % 10)

theorem no_three_digit_odd_sum_30 :
  ¬ ∃ n : ℕ, is_three_digit n ∧ digit_sum n = 30 ∧ n % 2 = 1 :=
sorry

end NUMINAMATH_CALUDE_no_three_digit_odd_sum_30_l1849_184913


namespace NUMINAMATH_CALUDE_pure_imaginary_complex_l1849_184958

/-- Given that z = (a^2 - 4) + (a + 2)i is a pure imaginary number,
    prove that (a + i^2015) / (1 + 2i) = -i -/
theorem pure_imaginary_complex (a : ℝ) :
  (Complex.I : ℂ)^2015 = -Complex.I →
  (a^2 - 4 : ℂ) = 0 →
  (a + 2 : ℂ) ≠ 0 →
  (a + Complex.I^2015) / (1 + 2 * Complex.I) = -Complex.I :=
by sorry

end NUMINAMATH_CALUDE_pure_imaginary_complex_l1849_184958


namespace NUMINAMATH_CALUDE_smallest_debt_theorem_l1849_184955

/-- The value of a pig in dollars -/
def pig_value : ℕ := 250

/-- The value of a goat in dollars -/
def goat_value : ℕ := 175

/-- The value of a sheep in dollars -/
def sheep_value : ℕ := 125

/-- The smallest positive debt that can be resolved -/
def smallest_resolvable_debt : ℕ := 25

theorem smallest_debt_theorem :
  (∃ (p g s : ℤ), smallest_resolvable_debt = pig_value * p + goat_value * g + sheep_value * s) ∧
  (∀ (d : ℕ), d > 0 ∧ d < smallest_resolvable_debt →
    ¬∃ (p g s : ℤ), d = pig_value * p + goat_value * g + sheep_value * s) :=
by sorry

end NUMINAMATH_CALUDE_smallest_debt_theorem_l1849_184955


namespace NUMINAMATH_CALUDE_quadratic_equation_with_absolute_roots_l1849_184963

theorem quadratic_equation_with_absolute_roots 
  (x₁ x₂ m : ℝ) 
  (h₁ : x₁ > 0) 
  (h₂ : x₂ < 0) 
  (h₃ : ∃ (original_eq : ℝ → Prop), original_eq x₁ ∧ original_eq x₂) :
  ∃ (new_eq : ℝ → Prop), 
    new_eq (|x₁|) ∧ 
    new_eq (|x₂|) ∧ 
    ∀ x, new_eq x ↔ x^2 - (1 - 4*m)/x + 2 = 0 :=
sorry

end NUMINAMATH_CALUDE_quadratic_equation_with_absolute_roots_l1849_184963


namespace NUMINAMATH_CALUDE_problem_statement_l1849_184931

theorem problem_statement : (3150 - 3030)^2 / 144 = 100 := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l1849_184931


namespace NUMINAMATH_CALUDE_candy_remainder_l1849_184950

theorem candy_remainder : (31254389 : ℕ) % 6 = 5 := by sorry

end NUMINAMATH_CALUDE_candy_remainder_l1849_184950


namespace NUMINAMATH_CALUDE_number_of_arrangements_l1849_184929

/-- The number of applicants --/
def num_applicants : ℕ := 5

/-- The number of positions to be filled --/
def num_positions : ℕ := 3

/-- The number of students to be selected --/
def num_selected : ℕ := 3

/-- Function to calculate the number of arrangements --/
def calculate_arrangements (n : ℕ) (r : ℕ) : ℕ :=
  if r > n then 0
  else Nat.choose n r

/-- Theorem stating the number of arrangements --/
theorem number_of_arrangements :
  calculate_arrangements num_applicants num_selected +
  calculate_arrangements (num_applicants - 1) num_selected = 16 :=
sorry

end NUMINAMATH_CALUDE_number_of_arrangements_l1849_184929


namespace NUMINAMATH_CALUDE_assignment_count_theorem_l1849_184937

/-- The number of ways to assign 4 distinct objects to 3 distinct groups,
    where each group must contain at least one object. -/
def assignment_count : ℕ := 36

/-- The number of distinct objects to be assigned. -/
def num_objects : ℕ := 4

/-- The number of distinct groups to which objects are assigned. -/
def num_groups : ℕ := 3

theorem assignment_count_theorem :
  (∀ assignment : Fin num_objects → Fin num_groups,
    (∀ g : Fin num_groups, ∃ o : Fin num_objects, assignment o = g) →
    ∃! c : ℕ, c = assignment_count) :=
sorry

end NUMINAMATH_CALUDE_assignment_count_theorem_l1849_184937


namespace NUMINAMATH_CALUDE_passengers_in_first_class_l1849_184956

theorem passengers_in_first_class (total_passengers : ℕ) 
  (women_percentage : ℚ) (men_percentage : ℚ)
  (women_first_class_percentage : ℚ) (men_first_class_percentage : ℚ)
  (h1 : total_passengers = 300)
  (h2 : women_percentage = 1/2)
  (h3 : men_percentage = 1/2)
  (h4 : women_first_class_percentage = 1/5)
  (h5 : men_first_class_percentage = 3/20) :
  ⌈(total_passengers : ℚ) * women_percentage * women_first_class_percentage + 
   (total_passengers : ℚ) * men_percentage * men_first_class_percentage⌉ = 53 :=
by sorry

end NUMINAMATH_CALUDE_passengers_in_first_class_l1849_184956


namespace NUMINAMATH_CALUDE_triangle_third_side_exists_l1849_184959

theorem triangle_third_side_exists : ∃ x : ℕ, 
  3 ≤ x ∧ x ≤ 7 ∧ 
  (x + 3 > 5) ∧ (x + 5 > 3) ∧ (3 + 5 > x) ∧
  (x > 5 - 3) ∧ (x < 5 + 3) := by
  sorry

end NUMINAMATH_CALUDE_triangle_third_side_exists_l1849_184959


namespace NUMINAMATH_CALUDE_acid_dilution_l1849_184904

/-- Proves that adding 30 ounces of pure water to 50 ounces of a 40% acid solution results in a 25% acid solution -/
theorem acid_dilution (initial_volume : ℝ) (initial_concentration : ℝ) 
  (water_added : ℝ) (final_concentration : ℝ) : 
  initial_volume = 50 →
  initial_concentration = 0.40 →
  water_added = 30 →
  final_concentration = 0.25 →
  (initial_volume * initial_concentration) / (initial_volume + water_added) = final_concentration :=
by sorry

end NUMINAMATH_CALUDE_acid_dilution_l1849_184904


namespace NUMINAMATH_CALUDE_triangle_inequalities_l1849_184966

-- Define the points and lengths
variables (P Q R S : ℝ × ℝ) (a b c : ℝ)

-- Define the conditions
def collinear (P Q R S : ℝ × ℝ) : Prop :=
  ∃ t₁ t₂ t₃ : ℝ, 0 < t₁ ∧ t₁ < t₂ ∧ t₂ < t₃ ∧
  Q = P + t₁ • (S - P) ∧
  R = P + t₂ • (S - P) ∧
  S = P + t₃ • (S - P)

def segment_lengths (P Q R S : ℝ × ℝ) (a b c : ℝ) : Prop :=
  dist P Q = a ∧ dist P R = b ∧ dist P S = c

def can_form_triangle (a b c : ℝ) : Prop :=
  a + (b - a) > c - b ∧
  (b - a) + (c - b) > a ∧
  a + (c - b) > b - a

-- State the theorem
theorem triangle_inequalities
  (h_collinear : collinear P Q R S)
  (h_lengths : segment_lengths P Q R S a b c)
  (h_triangle : can_form_triangle a b c) :
  a < c / 3 ∧ b < 2 * a + c :=
sorry

end NUMINAMATH_CALUDE_triangle_inequalities_l1849_184966


namespace NUMINAMATH_CALUDE_parallel_lines_a_value_l1849_184964

/-- Two lines in the plane are parallel if and only if they have the same slope -/
axiom parallel_lines_same_slope {m₁ m₂ b₁ b₂ : ℝ} :
  (∀ x y, y = m₁ * x + b₁ ↔ y = m₂ * x + b₂) ↔ m₁ = m₂

/-- The problem statement -/
theorem parallel_lines_a_value :
  (∀ x y, y = x ↔ 2 * x + a * y = 1) → a = -2 :=
by sorry

end NUMINAMATH_CALUDE_parallel_lines_a_value_l1849_184964


namespace NUMINAMATH_CALUDE_sum_of_squares_lower_bound_l1849_184936

theorem sum_of_squares_lower_bound (x y z m : ℝ) (h : x + y + z = m) :
  x^2 + y^2 + z^2 ≥ m^2 / 3 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_squares_lower_bound_l1849_184936


namespace NUMINAMATH_CALUDE_eldest_child_age_l1849_184925

-- Define the number of children
def num_children : ℕ := 8

-- Define the age difference between consecutive children
def age_difference : ℕ := 3

-- Define the total sum of ages
def total_age_sum : ℕ := 100

-- Theorem statement
theorem eldest_child_age :
  ∃ (youngest_age : ℕ),
    (youngest_age + (num_children - 1) * age_difference) +
    (youngest_age + (num_children - 2) * age_difference) +
    (youngest_age + (num_children - 3) * age_difference) +
    (youngest_age + (num_children - 4) * age_difference) +
    (youngest_age + (num_children - 5) * age_difference) +
    (youngest_age + (num_children - 6) * age_difference) +
    (youngest_age + (num_children - 7) * age_difference) +
    youngest_age = total_age_sum →
    youngest_age + (num_children - 1) * age_difference = 23 :=
by
  sorry

end NUMINAMATH_CALUDE_eldest_child_age_l1849_184925


namespace NUMINAMATH_CALUDE_smallest_next_divisor_after_221_l1849_184910

theorem smallest_next_divisor_after_221 (m : ℕ) (h1 : m ≥ 1000 ∧ m < 10000) 
  (h2 : m % 2 = 0) (h3 : m % 221 = 0) : 
  ∃ (d : ℕ), d ∣ m ∧ d > 221 ∧ d ≤ 442 ∧ ∀ (x : ℕ), x ∣ m → x > 221 → x ≥ d :=
by sorry

end NUMINAMATH_CALUDE_smallest_next_divisor_after_221_l1849_184910


namespace NUMINAMATH_CALUDE_min_slope_and_sum_reciprocals_l1849_184957

noncomputable section

def f (x : ℝ) := x^3 - x^2 + (2 * Real.sqrt 2 - 3) * x + 3 - 2 * Real.sqrt 2

def f' (x : ℝ) := 3 * x^2 - 2 * x + 2 * Real.sqrt 2 - 3

theorem min_slope_and_sum_reciprocals :
  (∃ (x_min : ℝ), ∀ (x : ℝ), f' x_min ≤ f' x ∧ f' x_min = 2 * Real.sqrt 2 - 10 / 3) ∧
  (∃ (x₁ x₂ x₃ : ℝ), f x₁ = 0 ∧ f x₂ = 0 ∧ f x₃ = 0 ∧ 
    1 / f' x₁ + 1 / f' x₂ + 1 / f' x₃ = 0) := by
  sorry

end

end NUMINAMATH_CALUDE_min_slope_and_sum_reciprocals_l1849_184957


namespace NUMINAMATH_CALUDE_arithmetic_contains_geometric_l1849_184906

/-- An arithmetic sequence of positive real numbers -/
def arithmetic_sequence (a₀ d : ℝ) (n : ℕ) : ℝ := a₀ + n • d

/-- A geometric sequence of real numbers -/
def geometric_sequence (b₀ q : ℝ) (n : ℕ) : ℝ := b₀ * q^n

/-- Theorem: If an infinite arithmetic sequence of positive real numbers contains two different
    powers of an integer greater than 1, then it contains an infinite geometric sequence -/
theorem arithmetic_contains_geometric
  (a₀ d : ℝ) (a : ℕ) (h_a : a > 1) 
  (h_pos : ∀ n, arithmetic_sequence a₀ d n > 0)
  (m n : ℕ) (h_mn : m ≠ n)
  (h_power_m : ∃ k₁, arithmetic_sequence a₀ d k₁ = a^m)
  (h_power_n : ∃ k₂, arithmetic_sequence a₀ d k₂ = a^n) :
  ∃ b₀ q : ℝ, ∀ k, ∃ l, arithmetic_sequence a₀ d l = geometric_sequence b₀ q k :=
sorry

end NUMINAMATH_CALUDE_arithmetic_contains_geometric_l1849_184906


namespace NUMINAMATH_CALUDE_probability_point_between_C_and_E_l1849_184986

/-- Given a line segment AB with points C, D, and E, where AB = 4AD, AB = 5BC, 
    and E is the midpoint of CD, the probability that a randomly selected point 
    on AB falls between C and E is 1/4. -/
theorem probability_point_between_C_and_E 
  (A B C D E : ℝ) 
  (h1 : A < C) (h2 : C < D) (h3 : D < B)
  (h4 : B - A = 4 * (D - A))
  (h5 : B - A = 5 * (C - B))
  (h6 : E = (C + D) / 2) :
  (E - C) / (B - A) = 1 / 4 := by
  sorry

#check probability_point_between_C_and_E

end NUMINAMATH_CALUDE_probability_point_between_C_and_E_l1849_184986


namespace NUMINAMATH_CALUDE_two_inequalities_for_real_numbers_l1849_184907

theorem two_inequalities_for_real_numbers (a b c : ℝ) : 
  (a * b / c^2 + b * c / a^2 + a * c / b^2 ≥ a / c + b / a + c / b) ∧
  (a^2 / b^2 + b^2 / c^2 + c^2 / a^2 ≥ a / b + b / c + c / a) := by
  sorry

end NUMINAMATH_CALUDE_two_inequalities_for_real_numbers_l1849_184907


namespace NUMINAMATH_CALUDE_picks_theorem_irregular_polygon_area_l1849_184912

/-- Pick's Theorem for a polygon on a lattice -/
theorem picks_theorem (B I : ℕ) (A : ℚ) : A = I + B / 2 - 1 →
  B = 10 → I = 12 → A = 16 := by
  sorry

/-- The area of the irregular polygon -/
theorem irregular_polygon_area : ∃ A : ℚ, A = 16 := by
  sorry

end NUMINAMATH_CALUDE_picks_theorem_irregular_polygon_area_l1849_184912


namespace NUMINAMATH_CALUDE_exactly_one_pass_probability_l1849_184932

theorem exactly_one_pass_probability (p : ℝ) (hp : p = 1 / 2) :
  let q := 1 - p
  p * q + q * p = 1 / 2 := by sorry

end NUMINAMATH_CALUDE_exactly_one_pass_probability_l1849_184932


namespace NUMINAMATH_CALUDE_three_lines_intersection_l1849_184996

/-- Three lines intersecting at a point -/
structure ThreeLines where
  a : ℝ
  l₁ : ℝ → ℝ → Prop
  l₂ : ℝ → ℝ → Prop
  l₃ : ℝ → ℝ → Prop
  h₁ : l₁ x y ↔ a * x + 2 * y + 6 = 0
  h₂ : l₂ x y ↔ x + y - 4 = 0
  h₃ : l₃ x y ↔ 2 * x - y + 1 = 0
  intersection : ∃ (x y : ℝ), l₁ x y ∧ l₂ x y ∧ l₃ x y

/-- The value of a when three lines intersect at a point -/
theorem three_lines_intersection (t : ThreeLines) : t.a = -12 := by
  sorry

end NUMINAMATH_CALUDE_three_lines_intersection_l1849_184996


namespace NUMINAMATH_CALUDE_men_in_room_l1849_184917

theorem men_in_room (initial_men : ℕ) (initial_women : ℕ) : 
  (initial_men : ℚ) / initial_women = 4 / 5 →
  2 * (initial_women - 3) = 24 →
  initial_men + 2 = 14 := by
sorry

end NUMINAMATH_CALUDE_men_in_room_l1849_184917


namespace NUMINAMATH_CALUDE_divisor_problem_l1849_184987

theorem divisor_problem (N D : ℕ) (h1 : N % D = 255) (h2 : (2 * N) % D = 112) : D = 398 := by
  sorry

end NUMINAMATH_CALUDE_divisor_problem_l1849_184987


namespace NUMINAMATH_CALUDE_sphere_volume_from_cylinder_and_cone_l1849_184968

/-- The volume of a sphere given specific conditions involving a cylinder and cone -/
theorem sphere_volume_from_cylinder_and_cone 
  (r : ℝ) -- radius of the sphere
  (h : ℝ) -- height of the cylinder and cone
  (M : ℝ) -- volume of the cylinder
  (h_eq : h = 2 * r) -- height is twice the radius
  (M_eq : M = π * r^2 * h) -- volume formula for cylinder
  (V_cone : ℝ := (1/3) * π * r^2 * h) -- volume of the cone
  (C : ℝ) -- volume of the sphere
  (vol_eq : M - V_cone = C) -- combined volume equals sphere volume
  : C = (8/3) * π * r^3 := by
  sorry

end NUMINAMATH_CALUDE_sphere_volume_from_cylinder_and_cone_l1849_184968


namespace NUMINAMATH_CALUDE_shortest_path_general_drinking_horse_l1849_184914

-- Define the points and the line
def A : ℝ × ℝ := (1, 1)
def B : ℝ × ℝ := (4, 4)
def l (x y : ℝ) : Prop := x - y + 1 = 0

-- State the theorem
theorem shortest_path_general_drinking_horse :
  ∃ (P : ℝ × ℝ), l P.1 P.2 ∧
    Real.sqrt ((A.1 - P.1)^2 + (A.2 - P.2)^2) +
    Real.sqrt ((B.1 - P.1)^2 + (B.2 - P.2)^2) =
    2 * Real.sqrt 5 :=
sorry

end NUMINAMATH_CALUDE_shortest_path_general_drinking_horse_l1849_184914


namespace NUMINAMATH_CALUDE_distance_between_shores_is_600_l1849_184961

/-- Represents the distance between two shores --/
def distance_between_shores : ℝ := sorry

/-- Represents the distance of the first meeting point from shore A --/
def first_meeting_distance : ℝ := 500

/-- Represents the distance of the second meeting point from shore B --/
def second_meeting_distance : ℝ := 300

/-- Theorem stating that the distance between shores A and B is 600 yards --/
theorem distance_between_shores_is_600 :
  distance_between_shores = 600 :=
sorry

end NUMINAMATH_CALUDE_distance_between_shores_is_600_l1849_184961


namespace NUMINAMATH_CALUDE_edge_bound_l1849_184984

/-- A simple graph with no 4-cycles -/
structure NoCycleFourGraph where
  -- The vertex set
  V : Type
  -- The edge relation
  E : V → V → Prop
  -- Symmetry of edges
  symm : ∀ u v, E u v → E v u
  -- No self-loops
  irrefl : ∀ v, ¬E v v
  -- No 4-cycles
  no_four_cycle : ∀ a b c d, E a b → E b c → E c d → E d a → (a = c ∨ b = d)

/-- The number of vertices in a graph -/
def num_vertices (G : NoCycleFourGraph) : ℕ := sorry

/-- The number of edges in a graph -/
def num_edges (G : NoCycleFourGraph) : ℕ := sorry

/-- The main theorem -/
theorem edge_bound (G : NoCycleFourGraph) :
  let n := num_vertices G
  let m := num_edges G
  m ≤ (n / 4) * (1 + Real.sqrt (4 * n - 3)) := by sorry

end NUMINAMATH_CALUDE_edge_bound_l1849_184984


namespace NUMINAMATH_CALUDE_jokes_increase_factor_l1849_184989

/-- The factor by which Jessy and Alan increased their jokes -/
def increase_factor (first_saturday_jokes : ℕ) (total_jokes : ℕ) : ℚ :=
  (total_jokes - first_saturday_jokes : ℚ) / first_saturday_jokes

/-- Theorem stating that the increase factor is 2 -/
theorem jokes_increase_factor : increase_factor 18 54 = 2 := by
  sorry

#eval increase_factor 18 54

end NUMINAMATH_CALUDE_jokes_increase_factor_l1849_184989


namespace NUMINAMATH_CALUDE_no_integer_pairs_l1849_184920

theorem no_integer_pairs : ¬∃ (x y : ℤ), 0 < x ∧ x < y ∧ Real.sqrt 2500 = Real.sqrt x + 2 * Real.sqrt y := by
  sorry

end NUMINAMATH_CALUDE_no_integer_pairs_l1849_184920


namespace NUMINAMATH_CALUDE_factorization_1_factorization_2_factorization_3_l1849_184993

-- Problem 1
theorem factorization_1 (a : ℝ) : 3*a^3 - 6*a^2 + 3*a = 3*a*(a - 1)^2 := by sorry

-- Problem 2
theorem factorization_2 (a b x y : ℝ) : a^2*(x - y) + b^2*(y - x) = (x - y)*(a - b)*(a + b) := by sorry

-- Problem 3
theorem factorization_3 (a b : ℝ) : 16*(a + b)^2 - 9*(a - b)^2 = (a + 7*b)*(7*a + b) := by sorry

end NUMINAMATH_CALUDE_factorization_1_factorization_2_factorization_3_l1849_184993


namespace NUMINAMATH_CALUDE_consecutive_points_distance_l1849_184982

/-- Given five consecutive points on a straight line, if certain distance conditions are met,
    then the distance between the last two points is 4. -/
theorem consecutive_points_distance (a b c d e : ℝ) : 
  (b - a) + (c - b) + (d - c) + (e - d) = (e - a)  -- Points are consecutive on a line
  → (c - b) = 2 * (d - c)  -- bc = 2 cd
  → (b - a) = 5  -- ab = 5
  → (c - a) = 11  -- ac = 11
  → (e - a) = 18  -- ae = 18
  → (e - d) = 4  -- de = 4
:= by sorry

end NUMINAMATH_CALUDE_consecutive_points_distance_l1849_184982


namespace NUMINAMATH_CALUDE_percentage_increase_is_20_percent_l1849_184938

/-- Represents the number of units in each building --/
structure BuildingUnits where
  first : ℕ
  second : ℕ
  third : ℕ

/-- Calculates the percentage increase from the second to the third building --/
def percentageIncrease (units : BuildingUnits) : ℚ :=
  (units.third - units.second : ℚ) / units.second * 100

/-- The main theorem stating the percentage increase is 20% --/
theorem percentage_increase_is_20_percent 
  (total : ℕ) 
  (h1 : total = 7520) 
  (units : BuildingUnits) 
  (h2 : units.first = 4000) 
  (h3 : units.second = 2 * units.first / 5) 
  (h4 : total = units.first + units.second + units.third) : 
  percentageIncrease units = 20 := by
  sorry

end NUMINAMATH_CALUDE_percentage_increase_is_20_percent_l1849_184938


namespace NUMINAMATH_CALUDE_mortgage_payment_l1849_184944

theorem mortgage_payment (P : ℝ) : 
  (P * (1 - 3^10) / (1 - 3) = 2952400) → P = 100 := by
  sorry

end NUMINAMATH_CALUDE_mortgage_payment_l1849_184944


namespace NUMINAMATH_CALUDE_ralphs_socks_l1849_184975

theorem ralphs_socks (x y z : ℕ) : 
  x + y + z = 12 →  -- Total pairs of socks
  x + 3*y + 4*z = 24 →  -- Total cost
  x ≥ 1 →  -- At least one pair of $1 socks
  y ≥ 1 →  -- At least one pair of $3 socks
  z ≥ 1 →  -- At least one pair of $4 socks
  x = 7  -- Number of $1 socks Ralph bought
  := by sorry

end NUMINAMATH_CALUDE_ralphs_socks_l1849_184975


namespace NUMINAMATH_CALUDE_union_of_sets_l1849_184930

def A (a : ℝ) : Set ℝ := {-1, a}
def B (a b : ℝ) : Set ℝ := {2^a, b}

theorem union_of_sets (a b : ℝ) :
  (A a) ∩ (B a b) = {1} → (A a) ∪ (B a b) = {-1, 1, 2} := by
  sorry

end NUMINAMATH_CALUDE_union_of_sets_l1849_184930


namespace NUMINAMATH_CALUDE_x_plus_y_values_l1849_184999

theorem x_plus_y_values (x y : ℝ) 
  (eq1 : x^2 + x*y + 2*y = 10) 
  (eq2 : y^2 + x*y + 2*x = 14) : 
  x + y = 4 ∨ x + y = -6 := by
sorry

end NUMINAMATH_CALUDE_x_plus_y_values_l1849_184999


namespace NUMINAMATH_CALUDE_work_completion_time_l1849_184915

/-- Proves that if A completes a work in 10 days, and A and B together complete the work in 
    2.3076923076923075 days, then B completes the work alone in 3 days. -/
theorem work_completion_time (a_time b_time combined_time : ℝ) 
    (ha : a_time = 10)
    (hc : combined_time = 2.3076923076923075)
    (h_combined : 1 / a_time + 1 / b_time = 1 / combined_time) : 
  b_time = 3 := by
  sorry

end NUMINAMATH_CALUDE_work_completion_time_l1849_184915


namespace NUMINAMATH_CALUDE_rational_operations_closure_l1849_184924

theorem rational_operations_closure (a b : ℚ) (h : b ≠ 0) :
  (∃ (x : ℚ), x = a + b) ∧
  (∃ (y : ℚ), y = a - b) ∧
  (∃ (z : ℚ), z = a * b) ∧
  (∃ (w : ℚ), w = a / b) :=
by sorry

end NUMINAMATH_CALUDE_rational_operations_closure_l1849_184924


namespace NUMINAMATH_CALUDE_parabola_shift_l1849_184909

/-- Represents a parabola in the form y = ax^2 + bx + c -/
structure Parabola where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Shifts a parabola horizontally -/
def shift_horizontal (p : Parabola) (h : ℝ) : Parabola :=
  { a := p.a,
    b := p.b - 2 * p.a * h,
    c := p.c + p.a * h^2 + p.b * h }

/-- Shifts a parabola vertically -/
def shift_vertical (p : Parabola) (v : ℝ) : Parabola :=
  { a := p.a,
    b := p.b,
    c := p.c + v }

/-- The original parabola y = x^2 + 2 -/
def original : Parabola :=
  { a := 1,
    b := 0,
    c := 2 }

theorem parabola_shift :
  let p1 := shift_horizontal original 1
  let p2 := shift_vertical p1 (-1)
  p2 = { a := 1, b := 2, c := 1 } :=
by sorry

end NUMINAMATH_CALUDE_parabola_shift_l1849_184909


namespace NUMINAMATH_CALUDE_bisection_arbitrary_precision_l1849_184994

/-- Represents a continuous function on a closed interval -/
def ContinuousFunction (a b : ℝ) := ℝ → ℝ

/-- Represents the bisection method applied to a function -/
def BisectionMethod (f : ContinuousFunction a b) (ε : ℝ) : ℝ := sorry

/-- Theorem stating that the bisection method can achieve arbitrary precision -/
theorem bisection_arbitrary_precision 
  (f : ContinuousFunction a b) 
  (h₁ : a < b) 
  (h₂ : f a * f b ≤ 0) 
  (ε : ℝ) 
  (h₃ : ε > 0) :
  ∃ x : ℝ, |f x| < ε ∧ x ∈ Set.Icc a b :=
sorry

end NUMINAMATH_CALUDE_bisection_arbitrary_precision_l1849_184994


namespace NUMINAMATH_CALUDE_read_book_in_six_days_book_structure_l1849_184981

/-- The number of days required to read a book -/
def days_to_read (total_pages : ℕ) (pages_per_day : ℕ) : ℕ :=
  total_pages / pages_per_day

/-- Theorem: It takes 6 days to read a 612-page book at 102 pages per day -/
theorem read_book_in_six_days :
  days_to_read 612 102 = 6 := by
  sorry

/-- The book has 24 chapters with pages equally distributed -/
def pages_per_chapter (total_pages : ℕ) (num_chapters : ℕ) : ℕ :=
  total_pages / num_chapters

/-- The book has 612 pages and 24 chapters -/
theorem book_structure :
  pages_per_chapter 612 24 = 612 / 24 := by
  sorry

end NUMINAMATH_CALUDE_read_book_in_six_days_book_structure_l1849_184981


namespace NUMINAMATH_CALUDE_max_point_of_product_l1849_184921

/-- Linear function f(x) -/
def f (x : ℝ) : ℝ := 2 * x + 2

/-- Linear function g(x) -/
def g (x : ℝ) : ℝ := -x - 3

/-- Product function h(x) = f(x) * g(x) -/
def h (x : ℝ) : ℝ := f x * g x

theorem max_point_of_product (x : ℝ) :
  f (-1) = 0 ∧ f 0 = 2 ∧ g 3 = 0 ∧ g 0 = -3 →
  ∃ (max_x : ℝ), max_x = -2 ∧ ∀ y, h y ≤ h max_x :=
sorry

end NUMINAMATH_CALUDE_max_point_of_product_l1849_184921


namespace NUMINAMATH_CALUDE_ackermann_3_2_l1849_184992

def A : ℕ → ℕ → ℕ
  | 0, n => n + 1
  | m + 1, 0 => A m 1
  | m + 1, n + 1 => A m (A (m + 1) n)

theorem ackermann_3_2 : A 3 2 = 29 := by sorry

end NUMINAMATH_CALUDE_ackermann_3_2_l1849_184992


namespace NUMINAMATH_CALUDE_line_passes_through_fixed_point_l1849_184945

/-- The line (a+1)x - y - 2a + 1 = 0 passes through the point (2,3) for all real a -/
theorem line_passes_through_fixed_point :
  ∀ (a : ℝ), (a + 1) * 2 - 3 - 2 * a + 1 = 0 := by
  sorry

end NUMINAMATH_CALUDE_line_passes_through_fixed_point_l1849_184945


namespace NUMINAMATH_CALUDE_arithmetic_calculation_l1849_184901

theorem arithmetic_calculation : 5 * (9 / 3) + 7 * 4 - 36 / 4 = 34 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_calculation_l1849_184901


namespace NUMINAMATH_CALUDE_sufficient_but_not_necessary_l1849_184928

theorem sufficient_but_not_necessary (p q : Prop) :
  -- Part 1: Sufficient condition
  ((p ∧ q) → ¬(¬p)) ∧
  -- Part 2: Not necessary condition
  ∃ (r : Prop), (¬(¬r) ∧ ¬(r ∧ q)) :=
by sorry

end NUMINAMATH_CALUDE_sufficient_but_not_necessary_l1849_184928


namespace NUMINAMATH_CALUDE_doubled_b_cost_percentage_l1849_184998

-- Define the cost function
def cost (t : ℝ) (b : ℝ) : ℝ := t * b^4

-- Theorem statement
theorem doubled_b_cost_percentage (t : ℝ) (b : ℝ) (h : t > 0) (h' : b > 0) :
  cost t (2*b) = 16 * cost t b := by
  sorry

end NUMINAMATH_CALUDE_doubled_b_cost_percentage_l1849_184998


namespace NUMINAMATH_CALUDE_two_boxes_in_case_l1849_184934

/-- The number of boxes in a case, given the total number of blocks and blocks per box -/
def boxes_in_case (total_blocks : ℕ) (blocks_per_box : ℕ) : ℕ :=
  total_blocks / blocks_per_box

/-- Theorem: There are 2 boxes in a case when there are 12 blocks in total and 6 blocks per box -/
theorem two_boxes_in_case :
  boxes_in_case 12 6 = 2 := by
  sorry

end NUMINAMATH_CALUDE_two_boxes_in_case_l1849_184934


namespace NUMINAMATH_CALUDE_alcohol_dilution_l1849_184980

/-- Proves that mixing 50 ml of 30% alcohol after-shave lotion with 30 ml of pure water
    results in a solution with 18.75% alcohol content. -/
theorem alcohol_dilution (initial_volume : ℝ) (initial_percentage : ℝ) (water_volume : ℝ)
  (h1 : initial_volume = 50)
  (h2 : initial_percentage = 30)
  (h3 : water_volume = 30) :
  let alcohol_volume : ℝ := initial_volume * (initial_percentage / 100)
  let total_volume : ℝ := initial_volume + water_volume
  let new_percentage : ℝ := (alcohol_volume / total_volume) * 100
  new_percentage = 18.75 := by
sorry

end NUMINAMATH_CALUDE_alcohol_dilution_l1849_184980


namespace NUMINAMATH_CALUDE_hyperbola_equation_l1849_184918

-- Define the eccentricity
def e : ℝ := 2

-- Define the ellipse parameters
def a_ellipse : ℝ := 4
def b_ellipse : ℝ := 3

-- Define the hyperbola equations
def horizontal_hyperbola (x y : ℝ) : Prop := x^2 / 16 - y^2 / 48 = 1
def vertical_hyperbola (x y : ℝ) : Prop := y^2 / 9 - x^2 / 27 = 1

-- Theorem statement
theorem hyperbola_equation :
  ∀ x y : ℝ,
  (x^2 / a_ellipse^2 + y^2 / b_ellipse^2 = 1) →
  (∃ a b : ℝ, (a = a_ellipse ∧ b^2 = a^2 * (e^2 - 1)) ∨ (a = b_ellipse ∧ b^2 = a^2 * (e^2 - 1))) →
  (horizontal_hyperbola x y ∨ vertical_hyperbola x y) :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_equation_l1849_184918


namespace NUMINAMATH_CALUDE_x_value_proof_l1849_184926

theorem x_value_proof (x y : ℤ) (h1 : x + y = 4) (h2 : x - y = 36) : x = 20 := by
  sorry

end NUMINAMATH_CALUDE_x_value_proof_l1849_184926


namespace NUMINAMATH_CALUDE_inequality_solution_l1849_184991

theorem inequality_solution (x : ℝ) : (2 - x) / 3 + 2 > x - (x - 2) / 2 → x < 2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_solution_l1849_184991


namespace NUMINAMATH_CALUDE_some_number_value_l1849_184978

theorem some_number_value (x : ℝ) :
  1 / 2 + ((2 / 3 * x) + 4) - 8 / 16 = 4.25 → x = 0.375 := by
  sorry

end NUMINAMATH_CALUDE_some_number_value_l1849_184978


namespace NUMINAMATH_CALUDE_necessary_but_not_sufficient_l1849_184948

theorem necessary_but_not_sufficient (a : ℝ) :
  (a^2 < 2*a → a < 2) ∧ ¬(∀ a, a < 2 → a^2 < 2*a) :=
sorry

end NUMINAMATH_CALUDE_necessary_but_not_sufficient_l1849_184948


namespace NUMINAMATH_CALUDE_count_multiples_of_five_l1849_184927

def d₁ (a : ℕ) : ℕ := a^2 + 2^a + a * 2^((a + 1)/2)
def d₂ (a : ℕ) : ℕ := a^2 + 2^a - a * 2^((a + 1)/2)

theorem count_multiples_of_five :
  ∃ (S : Finset ℕ), S.card = 101 ∧
    (∀ a ∈ S, 1 ≤ a ∧ a ≤ 251 ∧ (d₁ a * d₂ a) % 5 = 0) ∧
    (∀ a : ℕ, 1 ≤ a ∧ a ≤ 251 ∧ (d₁ a * d₂ a) % 5 = 0 → a ∈ S) :=
by sorry

end NUMINAMATH_CALUDE_count_multiples_of_five_l1849_184927


namespace NUMINAMATH_CALUDE_integer_solution_problem_l1849_184903

theorem integer_solution_problem :
  ∀ a b c : ℤ,
  1 < a ∧ a < b ∧ b < c →
  (∃ k : ℤ, k * ((a - 1) * (b - 1) * (c - 1)) = a * b * c - 1) →
  ((a = 3 ∧ b = 5 ∧ c = 15) ∨ (a = 2 ∧ b = 4 ∧ c = 8)) :=
by sorry

end NUMINAMATH_CALUDE_integer_solution_problem_l1849_184903


namespace NUMINAMATH_CALUDE_family_can_purchase_in_fourth_month_l1849_184962

/-- Represents the family's financial situation and purchase plan -/
structure Family where
  monthly_income : ℕ
  monthly_expenses : ℕ
  initial_savings : ℕ
  furniture_cost : ℕ

/-- Calculates the month when the family can make the purchase -/
def purchase_month (f : Family) : ℕ :=
  let monthly_savings := f.monthly_income - f.monthly_expenses
  let additional_required := f.furniture_cost - f.initial_savings
  (additional_required + monthly_savings - 1) / monthly_savings + 1

/-- Theorem stating that the family can make the purchase in the 4th month -/
theorem family_can_purchase_in_fourth_month (f : Family) 
  (h1 : f.monthly_income = 150000)
  (h2 : f.monthly_expenses = 115000)
  (h3 : f.initial_savings = 45000)
  (h4 : f.furniture_cost = 127000) :
  purchase_month f = 4 := by
  sorry

#eval purchase_month { 
  monthly_income := 150000, 
  monthly_expenses := 115000, 
  initial_savings := 45000, 
  furniture_cost := 127000 
}

end NUMINAMATH_CALUDE_family_can_purchase_in_fourth_month_l1849_184962


namespace NUMINAMATH_CALUDE_digital_city_activities_l1849_184902

-- Define the concept of a digital city
structure DigitalCity where
  is_part_of_digital_earth : Bool

-- Define possible activities in a digital city
inductive DigitalCityActivity
  | DistanceEducation
  | OnlineShopping
  | OnlineMedicalAdvice

-- Define a function that checks if an activity is enabled in a digital city
def is_enabled (city : DigitalCity) (activity : DigitalCityActivity) : Prop :=
  city.is_part_of_digital_earth

-- Theorem stating that digital cities enable specific activities
theorem digital_city_activities (city : DigitalCity) 
  (h : city.is_part_of_digital_earth = true) : 
  (is_enabled city DigitalCityActivity.DistanceEducation) ∧
  (is_enabled city DigitalCityActivity.OnlineShopping) ∧
  (is_enabled city DigitalCityActivity.OnlineMedicalAdvice) :=
by
  sorry


end NUMINAMATH_CALUDE_digital_city_activities_l1849_184902
