import Mathlib

namespace NUMINAMATH_CALUDE_simplify_polynomial_l1264_126426

theorem simplify_polynomial (x : ℝ) : 
  2*x*(4*x^3 - 3*x + 1) - 7*(x^3 - x^2 + 3*x - 4) = 8*x^4 - 7*x^3 + x^2 - 19*x + 28 := by
  sorry

end NUMINAMATH_CALUDE_simplify_polynomial_l1264_126426


namespace NUMINAMATH_CALUDE_inequality_condition_l1264_126459

theorem inequality_condition (a b : ℝ) :
  a * Real.sqrt a + b * Real.sqrt b > a * Real.sqrt b + b * Real.sqrt a →
  a ≥ 0 ∧ b ≥ 0 ∧ a ≠ b :=
by sorry

end NUMINAMATH_CALUDE_inequality_condition_l1264_126459


namespace NUMINAMATH_CALUDE_binary_to_base5_l1264_126444

-- Define the binary number
def binary_num : List Bool := [true, true, false, true, false, true, true]

-- Function to convert binary to decimal
def binary_to_decimal (bin : List Bool) : ℕ :=
  bin.enum.foldl (fun acc (i, b) => acc + if b then 2^i else 0) 0

-- Function to convert decimal to base 5
def decimal_to_base5 (n : ℕ) : List ℕ :=
  if n = 0 then [0] else
    let rec aux (m : ℕ) (acc : List ℕ) : List ℕ :=
      if m = 0 then acc else aux (m / 5) ((m % 5) :: acc)
    aux n []

-- Theorem statement
theorem binary_to_base5 :
  decimal_to_base5 (binary_to_decimal binary_num) = [4, 1, 2] :=
sorry

end NUMINAMATH_CALUDE_binary_to_base5_l1264_126444


namespace NUMINAMATH_CALUDE_plus_sign_square_has_90_degree_symmetry_l1264_126437

/-- Represents a square with markings -/
structure MarkedSquare where
  markings : Set (ℝ × ℝ)

/-- Defines 90-degree rotational symmetry for a marked square -/
def has_90_degree_rotational_symmetry (s : MarkedSquare) : Prop :=
  ∀ (x y : ℝ), (x, y) ∈ s.markings ↔ (-y, x) ∈ s.markings

/-- Represents a square with vertical and horizontal midlines crossed (plus sign) -/
def plus_sign_square : MarkedSquare :=
  { markings := {(x, y) | x = 0 ∨ y = 0} }

/-- Theorem: A square with both vertical and horizontal midlines crossed (plus sign) has 90-degree rotational symmetry -/
theorem plus_sign_square_has_90_degree_symmetry :
  has_90_degree_rotational_symmetry plus_sign_square :=
sorry

end NUMINAMATH_CALUDE_plus_sign_square_has_90_degree_symmetry_l1264_126437


namespace NUMINAMATH_CALUDE_zephyr_in_top_three_l1264_126405

-- Define the propositions
variable (X : Prop) -- Xenon is in the top three
variable (Y : Prop) -- Yenofa is in the top three
variable (Z : Prop) -- Zephyr is in the top three

-- Define the conditions
axiom condition1 : Z → X
axiom condition2 : (X ∨ Y) → ¬Z
axiom condition3 : ¬((X ∨ Y) → ¬Z)

-- Theorem to prove
theorem zephyr_in_top_three : Z ∧ ¬X ∧ ¬Y := by
  sorry

end NUMINAMATH_CALUDE_zephyr_in_top_three_l1264_126405


namespace NUMINAMATH_CALUDE_divisible_by_56_l1264_126432

theorem divisible_by_56 (n : ℕ) 
  (h1 : ∃ k : ℕ, 3 * n + 1 = k ^ 2) 
  (h2 : ∃ m : ℕ, 4 * n + 1 = m ^ 2) : 
  56 ∣ n := by
sorry

end NUMINAMATH_CALUDE_divisible_by_56_l1264_126432


namespace NUMINAMATH_CALUDE_garrison_provision_theorem_l1264_126423

/-- Calculates the initial number of days provisions were supposed to last for a garrison --/
def initial_provision_days (initial_garrison : ℕ) (reinforcement : ℕ) (days_before_reinforcement : ℕ) (days_after_reinforcement : ℕ) : ℕ :=
  (initial_garrison + reinforcement) * days_after_reinforcement / initial_garrison + days_before_reinforcement

theorem garrison_provision_theorem (initial_garrison : ℕ) (reinforcement : ℕ) (days_before_reinforcement : ℕ) (days_after_reinforcement : ℕ) :
  initial_garrison = 2000 →
  reinforcement = 600 →
  days_before_reinforcement = 15 →
  days_after_reinforcement = 30 →
  initial_provision_days initial_garrison reinforcement days_before_reinforcement days_after_reinforcement = 39 :=
by
  sorry

#eval initial_provision_days 2000 600 15 30

end NUMINAMATH_CALUDE_garrison_provision_theorem_l1264_126423


namespace NUMINAMATH_CALUDE_cube_sum_given_sum_and_product_l1264_126477

theorem cube_sum_given_sum_and_product (a b : ℝ) 
  (h1 : a + b = 11) (h2 : a * b = 20) : 
  a^3 + b^3 = 671 := by sorry

end NUMINAMATH_CALUDE_cube_sum_given_sum_and_product_l1264_126477


namespace NUMINAMATH_CALUDE_divided_number_problem_l1264_126447

theorem divided_number_problem (x y : ℝ) : 
  x > y ∧ y = 11 ∧ 7 * x + 5 * y = 146 → x + y = 24 := by
  sorry

end NUMINAMATH_CALUDE_divided_number_problem_l1264_126447


namespace NUMINAMATH_CALUDE_cubic_root_sum_l1264_126479

theorem cubic_root_sum (a b c d : ℝ) (h1 : a ≠ 0) 
  (h2 : a * 4^3 + b * 4^2 + c * 4 + d = 0)
  (h3 : a * 1^3 + b * 1^2 + c * 1 + d = 0)
  (h4 : a * (-3)^3 + b * (-3)^2 + c * (-3) + d = 0) :
  (b + c) / a = 49 / 3 := by
  sorry

end NUMINAMATH_CALUDE_cubic_root_sum_l1264_126479


namespace NUMINAMATH_CALUDE_small_cheese_slices_l1264_126491

/-- The number of slices in a pizza order --/
structure PizzaOrder where
  small_cheese : ℕ
  large_pepperoni : ℕ
  eaten_per_person : ℕ
  left_per_person : ℕ

/-- Theorem: Given the conditions, the small cheese pizza has 8 slices --/
theorem small_cheese_slices (order : PizzaOrder)
  (h1 : order.large_pepperoni = 14)
  (h2 : order.eaten_per_person = 9)
  (h3 : order.left_per_person = 2)
  : order.small_cheese = 8 := by
  sorry

end NUMINAMATH_CALUDE_small_cheese_slices_l1264_126491


namespace NUMINAMATH_CALUDE_c_to_a_ratio_l1264_126460

/-- Represents the share of money for each person in Rupees -/
structure Shares where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Represents the conditions of the problem -/
def ProblemConditions (s : Shares) : Prop :=
  s.c = 56 ∧ 
  s.a + s.b + s.c = 287 ∧ 
  s.b = 0.65 * s.a

/-- Theorem stating the ratio of C's share to A's share in paisa -/
theorem c_to_a_ratio (s : Shares) 
  (h : ProblemConditions s) : (s.c * 100) / (s.a * 100) = 0.4 := by
  sorry

#check c_to_a_ratio

end NUMINAMATH_CALUDE_c_to_a_ratio_l1264_126460


namespace NUMINAMATH_CALUDE_river_width_proof_l1264_126494

theorem river_width_proof (total_distance : ℝ) (prob_find : ℝ) (x : ℝ) : 
  total_distance = 500 →
  prob_find = 4/5 →
  x / total_distance = 1 - prob_find →
  x = 100 := by
sorry

end NUMINAMATH_CALUDE_river_width_proof_l1264_126494


namespace NUMINAMATH_CALUDE_tea_mixture_price_l1264_126470

theorem tea_mixture_price (price1 price2 : ℝ) (ratio : ℝ) (mixture_price : ℝ) : 
  price1 = 64 →
  price2 = 74 →
  ratio = 1 →
  mixture_price = (price1 + price2) / (2 * ratio) →
  mixture_price = 69 := by
sorry

end NUMINAMATH_CALUDE_tea_mixture_price_l1264_126470


namespace NUMINAMATH_CALUDE_sarah_pencil_multiple_l1264_126440

/-- The number of pencils Sarah bought on Monday -/
def monday_pencils : ℕ := 20

/-- The number of pencils Sarah bought on Tuesday -/
def tuesday_pencils : ℕ := 18

/-- The total number of pencils Sarah has -/
def total_pencils : ℕ := 92

/-- The multiple of pencils bought on Wednesday compared to Tuesday -/
def wednesday_multiple : ℕ := (total_pencils - monday_pencils - tuesday_pencils) / tuesday_pencils

theorem sarah_pencil_multiple : wednesday_multiple = 3 := by
  sorry

end NUMINAMATH_CALUDE_sarah_pencil_multiple_l1264_126440


namespace NUMINAMATH_CALUDE_product_mod_twenty_l1264_126473

theorem product_mod_twenty : 58 * 73 * 84 ≡ 16 [MOD 20] := by sorry

end NUMINAMATH_CALUDE_product_mod_twenty_l1264_126473


namespace NUMINAMATH_CALUDE_reciprocal_sum_of_roots_l1264_126411

theorem reciprocal_sum_of_roots (α β : ℝ) : 
  (∃ x y : ℝ, 7 * x^2 - 6 * x + 8 = 0 ∧ 
               7 * y^2 - 6 * y + 8 = 0 ∧ 
               x ≠ y ∧ 
               α = 1 / x ∧ 
               β = 1 / y) → 
  α + β = 3/4 := by
sorry


end NUMINAMATH_CALUDE_reciprocal_sum_of_roots_l1264_126411


namespace NUMINAMATH_CALUDE_floor_neg_sqrt_64_over_9_l1264_126462

theorem floor_neg_sqrt_64_over_9 : ⌊-Real.sqrt (64/9)⌋ = -3 := by
  sorry

end NUMINAMATH_CALUDE_floor_neg_sqrt_64_over_9_l1264_126462


namespace NUMINAMATH_CALUDE_factorization_equality_l1264_126493

theorem factorization_equality (m n : ℝ) : m^2*n + 2*m*n^2 + n^3 = n*(m+n)^2 := by
  sorry

end NUMINAMATH_CALUDE_factorization_equality_l1264_126493


namespace NUMINAMATH_CALUDE_consecutive_odd_numbers_l1264_126449

theorem consecutive_odd_numbers (x : ℤ) : 
  (∃ (y z : ℤ), y = x + 2 ∧ z = x + 4 ∧ 
   Odd x ∧ Odd y ∧ Odd z ∧
   11 * x = 3 * (x + 4) + 16 + 4 * (x + 2)) → 
  x = 9 := by
sorry

end NUMINAMATH_CALUDE_consecutive_odd_numbers_l1264_126449


namespace NUMINAMATH_CALUDE_min_value_implies_a_l1264_126407

def f (x a : ℝ) : ℝ := |x + 1| + 2 * |x - a|

theorem min_value_implies_a (a : ℝ) :
  (∃ m : ℝ, m = 5 ∧ ∀ x : ℝ, f x a ≥ m) → a = -6 ∨ a = 4 := by sorry

end NUMINAMATH_CALUDE_min_value_implies_a_l1264_126407


namespace NUMINAMATH_CALUDE_problem_solution_l1264_126482

/-- The number of positive factors of n -/
def num_factors (n : ℕ) : ℕ := sorry

theorem problem_solution (y : ℕ) 
  (h1 : num_factors y = 18) 
  (h2 : 14 ∣ y) 
  (h3 : 18 ∣ y) : 
  y = 252 := by sorry

end NUMINAMATH_CALUDE_problem_solution_l1264_126482


namespace NUMINAMATH_CALUDE_intersection_M_N_l1264_126450

def M : Set ℝ := { x | x^2 ≥ 1 }
def N : Set ℝ := { y | ∃ x, y = 3*x^2 + 1 }

theorem intersection_M_N : M ∩ N = { x | x ≥ 1 ∨ x ≤ -1 } := by sorry

end NUMINAMATH_CALUDE_intersection_M_N_l1264_126450


namespace NUMINAMATH_CALUDE_stephanie_oranges_l1264_126496

/-- The number of times Stephanie went to the store last month -/
def store_visits : ℕ := 8

/-- The total number of oranges Stephanie bought last month -/
def total_oranges : ℕ := 16

/-- The number of oranges Stephanie buys each time she goes to the store -/
def oranges_per_visit : ℕ := total_oranges / store_visits

theorem stephanie_oranges : oranges_per_visit = 2 := by
  sorry

end NUMINAMATH_CALUDE_stephanie_oranges_l1264_126496


namespace NUMINAMATH_CALUDE_area_ratio_constant_l1264_126435

-- Define the ellipse C
def ellipse_C (x y : ℝ) : Prop := x^2 / 4 + y^2 = 1

-- Define points A, B, O, and T
def A : ℝ × ℝ := (-2, 0)
def B : ℝ × ℝ := (2, 0)
def O : ℝ × ℝ := (0, 0)
def T : ℝ × ℝ := (4, 0)

-- Define a line l passing through T
def line_l (m : ℝ) (x y : ℝ) : Prop := x = m * y + 4

-- Define the intersection points M and N
def M (m : ℝ) : ℝ × ℝ := sorry
def N (m : ℝ) : ℝ × ℝ := sorry

-- Define point P as the intersection of BM and x=1
def P (m : ℝ) : ℝ × ℝ := sorry

-- Define point Q as the intersection of AN and y-axis
def Q (m : ℝ) : ℝ × ℝ := sorry

-- Define the area of a triangle given three points
def triangle_area (p1 p2 p3 : ℝ × ℝ) : ℝ := sorry

-- Theorem statement
theorem area_ratio_constant (m : ℝ) : 
  triangle_area O A (Q m) / triangle_area O T (P m) = 1/3 := by sorry

end NUMINAMATH_CALUDE_area_ratio_constant_l1264_126435


namespace NUMINAMATH_CALUDE_ben_pea_picking_l1264_126474

/-- Given that Ben can pick 56 sugar snap peas in 7 minutes,
    prove that it will take him 9 minutes to pick 72 sugar snap peas. -/
theorem ben_pea_picking (rate : ℝ) (h : rate * 7 = 56) : rate * 9 = 72 := by
  sorry

end NUMINAMATH_CALUDE_ben_pea_picking_l1264_126474


namespace NUMINAMATH_CALUDE_least_subtraction_for_divisibility_l1264_126416

theorem least_subtraction_for_divisibility : 
  ∃ (n : ℕ), n = 1415 ∧ 
  (2500000 - n) % 1423 = 0 ∧ 
  ∀ (m : ℕ), m < n → (2500000 - m) % 1423 ≠ 0 := by
  sorry

end NUMINAMATH_CALUDE_least_subtraction_for_divisibility_l1264_126416


namespace NUMINAMATH_CALUDE_radius_C₁_is_sqrt_30_l1264_126487

/-- Two circles C₁ and C₂ with the following properties:
    1. The center O of C₁ lies on C₂
    2. C₁ and C₂ intersect at points X and Y
    3. There exists a point Z on C₂ exterior to C₁
    4. XZ = 13, OZ = 11, YZ = 7 -/
structure TwoCircles where
  O : ℝ × ℝ  -- Center of C₁
  X : ℝ × ℝ  -- Intersection point
  Y : ℝ × ℝ  -- Intersection point
  Z : ℝ × ℝ  -- Point on C₂ exterior to C₁
  C₁ : Set (ℝ × ℝ)  -- Circle C₁
  C₂ : Set (ℝ × ℝ)  -- Circle C₂
  O_on_C₂ : O ∈ C₂
  X_on_both : X ∈ C₁ ∧ X ∈ C₂
  Y_on_both : Y ∈ C₁ ∧ Y ∈ C₂
  Z_on_C₂ : Z ∈ C₂
  Z_exterior_C₁ : Z ∉ C₁
  XZ_length : dist X Z = 13
  OZ_length : dist O Z = 11
  YZ_length : dist Y Z = 7

/-- The radius of C₁ is √30 -/
theorem radius_C₁_is_sqrt_30 (tc : TwoCircles) : 
  ∃ (center : ℝ × ℝ) (r : ℝ), tc.C₁ = {p : ℝ × ℝ | dist p center = r} ∧ r = Real.sqrt 30 :=
sorry

end NUMINAMATH_CALUDE_radius_C₁_is_sqrt_30_l1264_126487


namespace NUMINAMATH_CALUDE_therapy_charge_theorem_l1264_126429

/-- Represents the pricing structure of a psychologist's therapy sessions. -/
structure TherapyPricing where
  first_hour : ℕ
  additional_hour : ℕ
  first_hour_premium : first_hour = additional_hour + 20

/-- Calculates the total charge for a given number of therapy hours. -/
def total_charge (pricing : TherapyPricing) (hours : ℕ) : ℕ :=
  pricing.first_hour + (hours - 1) * pricing.additional_hour

/-- Theorem stating the total charge for 3 hours of therapy given the conditions. -/
theorem therapy_charge_theorem (pricing : TherapyPricing) 
  (h1 : total_charge pricing 5 = 300) : 
  total_charge pricing 3 = 188 := by
  sorry

end NUMINAMATH_CALUDE_therapy_charge_theorem_l1264_126429


namespace NUMINAMATH_CALUDE_jiaki_calculation_final_result_l1264_126403

-- Define A as a function of x
def A (x : ℤ) : ℤ := 3 * x^2 - x + 1

-- Define B as a function of x
def B (x : ℤ) : ℤ := -x^2 - 2*x - 3

-- State the theorem
theorem jiaki_calculation (x : ℤ) :
  A x - B x = 2 * x^2 - 3*x - 2 ∧
  (x = -1 → A x - B x = 3) :=
by sorry

-- Define the largest negative integer
def largest_negative_integer : ℤ := -1

-- State the final result
theorem final_result :
  A largest_negative_integer - B largest_negative_integer = 3 :=
by sorry

end NUMINAMATH_CALUDE_jiaki_calculation_final_result_l1264_126403


namespace NUMINAMATH_CALUDE_statements_are_equivalent_l1264_126495

-- Define propositions
variable (R : Prop) -- R represents "It rains"
variable (G : Prop) -- G represents "I go outside"

-- Define the original statement
def original_statement : Prop := ¬R → ¬G

-- Define the equivalent statement
def equivalent_statement : Prop := G → R

-- Theorem stating the logical equivalence
theorem statements_are_equivalent : original_statement R G ↔ equivalent_statement R G := by
  sorry

end NUMINAMATH_CALUDE_statements_are_equivalent_l1264_126495


namespace NUMINAMATH_CALUDE_same_terminal_side_angle_l1264_126442

theorem same_terminal_side_angle : ∃ k : ℤ, k * 360 - 70 = 290 := by
  sorry

end NUMINAMATH_CALUDE_same_terminal_side_angle_l1264_126442


namespace NUMINAMATH_CALUDE_work_completion_time_l1264_126451

theorem work_completion_time 
  (john_time rose_time dave_time : ℝ) 
  (h1 : john_time = 8) 
  (h2 : rose_time = 16) 
  (h3 : dave_time = 12) :
  (1 / (1 / john_time + 1 / rose_time + 1 / dave_time)) = 48 / 13 := by
  sorry

end NUMINAMATH_CALUDE_work_completion_time_l1264_126451


namespace NUMINAMATH_CALUDE_diplomats_conference_l1264_126410

/-- The number of diplomats who attended the conference -/
def D : ℕ := 120

/-- The number of diplomats who spoke Japanese -/
def J : ℕ := 20

/-- The number of diplomats who did not speak Russian -/
def not_R : ℕ := 32

/-- The percentage of diplomats who spoke neither Japanese nor Russian -/
def neither_percent : ℚ := 20 / 100

/-- The percentage of diplomats who spoke both Japanese and Russian -/
def both_percent : ℚ := 10 / 100

theorem diplomats_conference :
  D = 120 ∧
  J = 20 ∧
  not_R = 32 ∧
  neither_percent = 20 / 100 ∧
  both_percent = 10 / 100 ∧
  (D : ℚ) * neither_percent = (D - (J + (D - not_R) - (D : ℚ) * both_percent) : ℚ) :=
by sorry

end NUMINAMATH_CALUDE_diplomats_conference_l1264_126410


namespace NUMINAMATH_CALUDE_f_not_prime_l1264_126464

def f (n : ℕ+) : ℤ := n.val^4 - 400 * n.val^2 + 600

theorem f_not_prime : ∀ n : ℕ+, ¬ Nat.Prime (Int.natAbs (f n)) := by
  sorry

end NUMINAMATH_CALUDE_f_not_prime_l1264_126464


namespace NUMINAMATH_CALUDE_girls_in_school_after_joining_l1264_126480

/-- The number of girls in a school after new students join -/
def girls_after_joining (initial_girls new_girls : ℕ) : ℕ :=
  initial_girls + new_girls

/-- Theorem: The number of girls in the school after new students joined is 1414 -/
theorem girls_in_school_after_joining :
  girls_after_joining 732 682 = 1414 := by
  sorry

end NUMINAMATH_CALUDE_girls_in_school_after_joining_l1264_126480


namespace NUMINAMATH_CALUDE_debt_amount_l1264_126445

/-- Represents the savings of the three girls and the debt amount -/
structure Savings where
  lulu : ℕ
  nora : ℕ
  tamara : ℕ
  debt : ℕ

/-- Theorem stating the debt amount given the conditions -/
theorem debt_amount (s : Savings) :
  s.lulu = 6 ∧
  s.nora = 5 * s.lulu ∧
  s.nora = 3 * s.tamara ∧
  s.lulu + s.nora + s.tamara = s.debt + 6 →
  s.debt = 40 := by
  sorry


end NUMINAMATH_CALUDE_debt_amount_l1264_126445


namespace NUMINAMATH_CALUDE_marla_horse_purchase_l1264_126401

/-- The number of bottle caps equivalent to one lizard -/
def bottlecaps_per_lizard : ℕ := 8

/-- The number of lizards equivalent to 5 gallons of water -/
def lizards_per_five_gallons : ℕ := 3

/-- The number of gallons of water equivalent to one horse -/
def gallons_per_horse : ℕ := 80

/-- The number of bottle caps Marla can scavenge per day -/
def daily_scavenge : ℕ := 20

/-- The number of bottle caps Marla pays per night for food and shelter -/
def daily_expense : ℕ := 4

/-- The number of days it takes Marla to collect enough bottle caps to buy a horse -/
def days_to_buy_horse : ℕ := 24

theorem marla_horse_purchase :
  days_to_buy_horse * (daily_scavenge - daily_expense) =
  (gallons_per_horse * lizards_per_five_gallons * bottlecaps_per_lizard) / 5 :=
by sorry

end NUMINAMATH_CALUDE_marla_horse_purchase_l1264_126401


namespace NUMINAMATH_CALUDE_danys_farm_bushels_l1264_126406

/-- Represents the farm animals and their food consumption -/
structure Farm where
  cows : Nat
  sheep : Nat
  chickens : Nat
  cow_sheep_consumption : Nat
  chicken_consumption : Nat

/-- Calculates the total bushels needed for a day -/
def total_bushels (farm : Farm) : Nat :=
  (farm.cows + farm.sheep) * farm.cow_sheep_consumption + 
  farm.chickens * farm.chicken_consumption

/-- Dany's farm -/
def danys_farm : Farm := {
  cows := 4,
  sheep := 3,
  chickens := 7,
  cow_sheep_consumption := 2,
  chicken_consumption := 3
}

/-- Theorem stating that Dany needs 35 bushels for a day -/
theorem danys_farm_bushels : total_bushels danys_farm = 35 := by
  sorry

end NUMINAMATH_CALUDE_danys_farm_bushels_l1264_126406


namespace NUMINAMATH_CALUDE_pasta_for_reunion_l1264_126434

/-- Calculates the amount of pasta needed for a given number of people, 
    based on a recipe that uses 2 pounds for 7 people. -/
def pasta_needed (people : ℕ) : ℚ :=
  2 * (people / 7 : ℚ)

/-- Proves that 10 pounds of pasta are needed for 35 people. -/
theorem pasta_for_reunion : pasta_needed 35 = 10 := by
  sorry

end NUMINAMATH_CALUDE_pasta_for_reunion_l1264_126434


namespace NUMINAMATH_CALUDE_square_sum_of_three_reals_l1264_126439

theorem square_sum_of_three_reals (x y z : ℝ) 
  (h1 : (x + y + z)^2 = 25)
  (h2 : x*y + x*z + y*z = 8) :
  x^2 + y^2 + z^2 = 9 := by
sorry

end NUMINAMATH_CALUDE_square_sum_of_three_reals_l1264_126439


namespace NUMINAMATH_CALUDE_find_divisor_l1264_126433

theorem find_divisor (n d : ℕ) (h1 : n % d = 255) (h2 : (2 * n) % d = 112) : d = 398 := by
  sorry

end NUMINAMATH_CALUDE_find_divisor_l1264_126433


namespace NUMINAMATH_CALUDE_percentage_failed_english_l1264_126418

theorem percentage_failed_english (failed_hindi : ℝ) (failed_both : ℝ) (passed_both : ℝ)
  (h1 : failed_hindi = 34)
  (h2 : failed_both = 22)
  (h3 : passed_both = 44) :
  ∃ failed_english : ℝ,
    failed_english = 44 ∧
    failed_hindi + failed_english - failed_both = 100 - passed_both :=
by sorry

end NUMINAMATH_CALUDE_percentage_failed_english_l1264_126418


namespace NUMINAMATH_CALUDE_max_value_f_l1264_126419

theorem max_value_f (x : ℝ) (h : x < 3) : 
  (x^2 - 3*x + 4) / (x - 3) ≤ -1 := by sorry

end NUMINAMATH_CALUDE_max_value_f_l1264_126419


namespace NUMINAMATH_CALUDE_right_triangle_sine_l1264_126443

theorem right_triangle_sine (X Y Z : ℝ) : 
  -- XYZ is a right triangle with Y as the right angle
  (X + Y + Z = π) →
  (Y = π / 2) →
  -- sin X = 8/17
  (Real.sin X = 8 / 17) →
  -- Conclusion: sin Z = 15/17
  (Real.sin Z = 15 / 17) := by
sorry

end NUMINAMATH_CALUDE_right_triangle_sine_l1264_126443


namespace NUMINAMATH_CALUDE_line_not_in_fourth_quadrant_l1264_126415

/-- Given a line ax + by + c = 0 where ac > 0 and bc < 0, 
    the line does not pass through the fourth quadrant -/
theorem line_not_in_fourth_quadrant 
  (a b c : ℝ) 
  (h1 : a * c > 0) 
  (h2 : b * c < 0) : 
  ∃ (x y : ℝ), x > 0 ∧ y > 0 ∧ a * x + b * y + c ≠ 0 := by
sorry

end NUMINAMATH_CALUDE_line_not_in_fourth_quadrant_l1264_126415


namespace NUMINAMATH_CALUDE_prohibited_items_most_suitable_for_census_l1264_126478

/-- Represents a survey type -/
inductive SurveyType
  | CrashResistance
  | ProhibitedItems
  | AppleSweetness
  | WetlandSpecies

/-- Determines if a survey type is suitable for a census -/
def isSuitableForCensus (survey : SurveyType) : Prop :=
  match survey with
  | .ProhibitedItems => true
  | _ => false

/-- Theorem: The survey about prohibited items on high-speed trains is the most suitable for a census -/
theorem prohibited_items_most_suitable_for_census :
  ∀ (survey : SurveyType), isSuitableForCensus survey → survey = SurveyType.ProhibitedItems :=
by
  sorry

#check prohibited_items_most_suitable_for_census

end NUMINAMATH_CALUDE_prohibited_items_most_suitable_for_census_l1264_126478


namespace NUMINAMATH_CALUDE_max_min_values_of_f_l1264_126431

-- Define the function f(x)
def f (x : ℝ) : ℝ := 2 * x^3 + 9 * x^2 - 2

-- Define the interval
def interval : Set ℝ := {x | -4 ≤ x ∧ x ≤ 2}

-- State the theorem
theorem max_min_values_of_f :
  (∃ x ∈ interval, f x = 50) ∧
  (∀ y ∈ interval, f y ≤ 50) ∧
  (∃ x ∈ interval, f x = -2) ∧
  (∀ y ∈ interval, f y ≥ -2) := by
  sorry

end NUMINAMATH_CALUDE_max_min_values_of_f_l1264_126431


namespace NUMINAMATH_CALUDE_katherines_apples_katherines_apples_proof_l1264_126497

theorem katherines_apples : ℕ → Prop :=
  fun a : ℕ =>
    let p := 3 * a  -- number of pears
    let b := 5  -- number of bananas
    (a + p + b = 21) →  -- total number of fruits
    (a = 4)

-- Proof
theorem katherines_apples_proof : ∃ a : ℕ, katherines_apples a :=
  sorry

end NUMINAMATH_CALUDE_katherines_apples_katherines_apples_proof_l1264_126497


namespace NUMINAMATH_CALUDE_arithmetic_calculations_l1264_126457

theorem arithmetic_calculations : 
  ((62 + 38) / 4 = 25) ∧ 
  ((34 + 19) * 7 = 371) ∧ 
  (1500 - 125 * 8 = 500) := by
sorry

end NUMINAMATH_CALUDE_arithmetic_calculations_l1264_126457


namespace NUMINAMATH_CALUDE_asymptotic_stability_l1264_126417

noncomputable section

/-- The system of differential equations -/
def system (x y : ℝ) : ℝ × ℝ :=
  (y - x/2 - x*y^3/2, -y - 2*x + x^2*y^2)

/-- The Lyapunov function candidate -/
def V (x y : ℝ) : ℝ :=
  2*x^2 + y^2

/-- The time derivative of V along the system trajectories -/
def dVdt (x y : ℝ) : ℝ :=
  let (dx, dy) := system x y
  4*x*dx + 2*y*dy

theorem asymptotic_stability :
  ∃ δ > 0, ∀ x y : ℝ, x^2 + y^2 < δ^2 →
    (∀ t : ℝ, t ≥ 0 → 
      let (xt, yt) := system x y
      V xt yt ≤ V x y ∧ (x ≠ 0 ∨ y ≠ 0 → V xt yt < V x y)) ∧
    (∀ ε > 0, ∃ T : ℝ, T > 0 → 
      let (xT, yT) := system x y
      xT^2 + yT^2 < ε^2) :=
sorry

end

end NUMINAMATH_CALUDE_asymptotic_stability_l1264_126417


namespace NUMINAMATH_CALUDE_inequality_solution_l1264_126486

theorem inequality_solution (x : ℝ) :
  x > 2 →
  (((x - 2) ^ (x^2 - 6*x + 8)) > 1) ↔ (x > 2 ∧ x < 3) ∨ x > 4 :=
by sorry

end NUMINAMATH_CALUDE_inequality_solution_l1264_126486


namespace NUMINAMATH_CALUDE_max_sum_of_squares_l1264_126455

theorem max_sum_of_squares (x y z : ℝ) 
  (h1 : x + y = z - 1) 
  (h2 : x * y = z^2 - 7*z + 14) : 
  (∃ (max : ℝ), ∀ (x' y' z' : ℝ), 
    x' + y' = z' - 1 → 
    x' * y' = z'^2 - 7*z' + 14 → 
    x'^2 + y'^2 ≤ max ∧ 
    (x'^2 + y'^2 = max ↔ z' = 3) ∧ 
    max = 2) :=
sorry

end NUMINAMATH_CALUDE_max_sum_of_squares_l1264_126455


namespace NUMINAMATH_CALUDE_triangle_properties_l1264_126483

theorem triangle_properties (A B C : ℝ) (a b c : ℝ) :
  A = π / 3 →
  (b + c) / (Real.sin B + Real.sin C) = 2 →
  a = Real.sqrt 3 ∧
  (a * b * Real.sin C / 2 = Real.sqrt 3 / 2 → a + b + c = 3 + Real.sqrt 3) :=
by sorry

end NUMINAMATH_CALUDE_triangle_properties_l1264_126483


namespace NUMINAMATH_CALUDE_necessary_but_not_sufficient_condition_l1264_126412

theorem necessary_but_not_sufficient_condition (x : ℝ) :
  (x > Real.exp 1 → x > 1) ∧ ∃ x, x > 1 ∧ x ≤ Real.exp 1 := by sorry

end NUMINAMATH_CALUDE_necessary_but_not_sufficient_condition_l1264_126412


namespace NUMINAMATH_CALUDE_croissants_leftover_l1264_126472

theorem croissants_leftover (total : Nat) (neighbors : Nat) (h1 : total = 59) (h2 : neighbors = 8) :
  total % neighbors = 3 := by
  sorry

end NUMINAMATH_CALUDE_croissants_leftover_l1264_126472


namespace NUMINAMATH_CALUDE_expression_value_l1264_126422

theorem expression_value (x y : ℤ) (hx : x = 3) (hy : y = 2) : 3 * x - 4 * y + 2 = 3 := by
  sorry

end NUMINAMATH_CALUDE_expression_value_l1264_126422


namespace NUMINAMATH_CALUDE_word_arrangements_count_l1264_126446

def word_length : ℕ := 12
def repeated_letter_1_count : ℕ := 3
def repeated_letter_2_count : ℕ := 2
def repeated_letter_3_count : ℕ := 2
def unique_letters_count : ℕ := 5

def arrangements_count : ℕ := 19958400

theorem word_arrangements_count :
  (word_length.factorial) / 
  (repeated_letter_1_count.factorial * 
   repeated_letter_2_count.factorial * 
   repeated_letter_3_count.factorial) = arrangements_count := by
  sorry

end NUMINAMATH_CALUDE_word_arrangements_count_l1264_126446


namespace NUMINAMATH_CALUDE_product_div_3_probability_l1264_126492

/-- The probability of rolling a number not divisible by 3 on a standard 6-sided die -/
def prob_not_div_3 : ℚ := 2/3

/-- The number of dice rolled -/
def num_dice : ℕ := 5

/-- The probability that the product of the numbers rolled on 'num_dice' standard 6-sided dice is divisible by 3 -/
def prob_product_div_3 : ℚ := 1 - prob_not_div_3 ^ num_dice

theorem product_div_3_probability :
  prob_product_div_3 = 211/243 :=
sorry

end NUMINAMATH_CALUDE_product_div_3_probability_l1264_126492


namespace NUMINAMATH_CALUDE_book_cost_is_15_l1264_126467

def total_books : ℕ := 10
def num_magazines : ℕ := 10
def magazine_cost : ℚ := 2
def total_spent : ℚ := 170

theorem book_cost_is_15 :
  ∃ (book_cost : ℚ),
    book_cost * total_books + magazine_cost * num_magazines = total_spent ∧
    book_cost = 15 :=
by sorry

end NUMINAMATH_CALUDE_book_cost_is_15_l1264_126467


namespace NUMINAMATH_CALUDE_smallest_integer_above_sqrt3_plus_sqrt2_to_8th_l1264_126490

theorem smallest_integer_above_sqrt3_plus_sqrt2_to_8th (x : ℝ) : 
  x = (Real.sqrt 3 + Real.sqrt 2)^8 → 
  ∀ n : ℤ, (n : ℝ) > x → n ≥ 5360 ∧ 5360 > x :=
by sorry

end NUMINAMATH_CALUDE_smallest_integer_above_sqrt3_plus_sqrt2_to_8th_l1264_126490


namespace NUMINAMATH_CALUDE_smallest_n_with_6474_l1264_126438

def concatenate (a b c : ℕ) : List ℕ :=
  (a.digits 10) ++ (b.digits 10) ++ (c.digits 10)

def contains_subseq (l : List ℕ) (s : List ℕ) : Prop :=
  ∃ i, l.drop i = s ++ l.drop (i + s.length)

theorem smallest_n_with_6474 :
  ∀ n : ℕ, n < 46 →
    ¬ (contains_subseq (concatenate n (n + 1) (n + 2)) [6, 4, 7, 4]) ∧
  contains_subseq (concatenate 46 47 48) [6, 4, 7, 4] :=
sorry

end NUMINAMATH_CALUDE_smallest_n_with_6474_l1264_126438


namespace NUMINAMATH_CALUDE_power_of_prime_exponent_l1264_126469

theorem power_of_prime_exponent (x y p n k : ℕ) 
  (h_n_gt_1 : n > 1)
  (h_n_odd : Odd n)
  (h_p_prime : Nat.Prime p)
  (h_p_odd : Odd p)
  (h_eq : x^n + y^n = p^k) :
  ∃ m : ℕ, n = p^m :=
sorry

end NUMINAMATH_CALUDE_power_of_prime_exponent_l1264_126469


namespace NUMINAMATH_CALUDE_red_grapes_count_l1264_126430

/-- Represents the number of fruits in a fruit salad. -/
structure FruitSalad where
  green_grapes : ℕ
  red_grapes : ℕ
  raspberries : ℕ

/-- Defines the conditions for a valid fruit salad. -/
def is_valid_fruit_salad (fs : FruitSalad) : Prop :=
  fs.red_grapes = 3 * fs.green_grapes + 7 ∧
  fs.raspberries = fs.green_grapes - 5 ∧
  fs.green_grapes + fs.red_grapes + fs.raspberries = 102

/-- Theorem stating that in a valid fruit salad, there are 67 red grapes. -/
theorem red_grapes_count (fs : FruitSalad) 
  (h : is_valid_fruit_salad fs) : fs.red_grapes = 67 := by
  sorry


end NUMINAMATH_CALUDE_red_grapes_count_l1264_126430


namespace NUMINAMATH_CALUDE_vector_operation_l1264_126436

def vector_a : ℝ × ℝ := (2, -1)
def vector_b : ℝ × ℝ := (1, 3)

theorem vector_operation :
  (-2 : ℝ) • vector_a + (3 : ℝ) • vector_b = (-1, 11) := by sorry

end NUMINAMATH_CALUDE_vector_operation_l1264_126436


namespace NUMINAMATH_CALUDE_apple_basket_count_l1264_126481

/-- The number of apples in a basket that is divided equally among apple lovers -/
def num_apples : ℕ → ℕ
| x => x * 22

/-- The theorem stating the number of apples in the basket -/
theorem apple_basket_count :
  ∃ (x : ℕ), 
    (num_apples x = (x + 45) * 13) ∧
    (num_apples x = 1430) :=
by sorry

end NUMINAMATH_CALUDE_apple_basket_count_l1264_126481


namespace NUMINAMATH_CALUDE_percentage_loss_l1264_126458

theorem percentage_loss (cost_price selling_price : ℚ) : 
  cost_price = 2300 →
  selling_price = 1610 →
  (cost_price - selling_price) / cost_price * 100 = 30 := by
sorry

end NUMINAMATH_CALUDE_percentage_loss_l1264_126458


namespace NUMINAMATH_CALUDE_sqrt_pattern_l1264_126456

theorem sqrt_pattern (n : ℕ) (h : n ≥ 1) : 
  Real.sqrt (n + 1 / (n + 2)) = (n + 1) * Real.sqrt (1 / (n + 2)) := by
  sorry

end NUMINAMATH_CALUDE_sqrt_pattern_l1264_126456


namespace NUMINAMATH_CALUDE_min_value_expression_l1264_126453

theorem min_value_expression (c d : ℝ) (hc : c ≠ 0) (hd : d ≠ 0) :
  c^2 + d^2 + 4/c^2 + 2*d/c ≥ 2 * Real.sqrt 3 ∧
  ∃ (c₀ d₀ : ℝ), c₀ ≠ 0 ∧ d₀ ≠ 0 ∧ c₀^2 + d₀^2 + 4/c₀^2 + 2*d₀/c₀ = 2 * Real.sqrt 3 :=
sorry

end NUMINAMATH_CALUDE_min_value_expression_l1264_126453


namespace NUMINAMATH_CALUDE_no_matching_product_and_sum_l1264_126461

theorem no_matching_product_and_sum : 
  ¬ ∃ (a b : ℕ), 1 ≤ a ∧ a < b ∧ b ≤ 15 ∧ 
  a * b = (List.range 16).sum - a - b :=
by sorry

end NUMINAMATH_CALUDE_no_matching_product_and_sum_l1264_126461


namespace NUMINAMATH_CALUDE_sixth_term_is_64_l1264_126441

/-- A geometric sequence with the given properties -/
structure GeometricSequence where
  a : ℕ → ℝ
  is_geometric : ∀ n, a (n + 1) / a n = a 2 / a 1
  sum_2_4 : a 2 + a 4 = 20
  sum_3_5 : a 3 + a 5 = 40

/-- The sixth term of the geometric sequence is 64 -/
theorem sixth_term_is_64 (seq : GeometricSequence) : seq.a 6 = 64 := by
  sorry

end NUMINAMATH_CALUDE_sixth_term_is_64_l1264_126441


namespace NUMINAMATH_CALUDE_boys_in_class_l1264_126421

theorem boys_in_class (total : ℕ) (girls_ratio : ℕ) (boys_ratio : ℕ) (boys : ℕ) : 
  total = 35 →
  girls_ratio = 4 →
  boys_ratio = 3 →
  girls_ratio * boys = boys_ratio * (total - boys) →
  boys = 15 := by
sorry

end NUMINAMATH_CALUDE_boys_in_class_l1264_126421


namespace NUMINAMATH_CALUDE_bills_toddler_count_l1264_126413

/-- The number of toddlers Bill thinks he counted -/
def billsCount (actualCount doubleCount missedCount : ℕ) : ℕ :=
  actualCount + doubleCount - missedCount

/-- Theorem stating that Bill thinks he counted 26 toddlers -/
theorem bills_toddler_count :
  let actualCount : ℕ := 21
  let doubleCount : ℕ := 8
  let missedCount : ℕ := 3
  billsCount actualCount doubleCount missedCount = 26 := by
  sorry

end NUMINAMATH_CALUDE_bills_toddler_count_l1264_126413


namespace NUMINAMATH_CALUDE_event_A_sufficient_not_necessary_for_event_B_l1264_126448

/- Define the number of balls for each color -/
def num_red_balls : ℕ := 5
def num_yellow_balls : ℕ := 3
def num_white_balls : ℕ := 2

/- Define the total number of balls -/
def total_balls : ℕ := num_red_balls + num_yellow_balls + num_white_balls

/- Define Event A: Selecting 1 red ball and 1 yellow ball -/
def event_A : Prop := ∃ (r : Fin num_red_balls) (y : Fin num_yellow_balls), True

/- Define Event B: Selecting any 2 balls from all available balls -/
def event_B : Prop := ∃ (b1 b2 : Fin total_balls), b1 ≠ b2

/- Theorem: Event A is sufficient but not necessary for Event B -/
theorem event_A_sufficient_not_necessary_for_event_B :
  (event_A → event_B) ∧ ¬(event_B → event_A) := by
  sorry


end NUMINAMATH_CALUDE_event_A_sufficient_not_necessary_for_event_B_l1264_126448


namespace NUMINAMATH_CALUDE_special_function_unique_l1264_126485

/-- A function f: ℤ × ℤ → ℝ satisfying specific conditions -/
def special_function (f : ℤ × ℤ → ℝ) : Prop :=
  (∀ x y z : ℤ, f (x, y) * f (y, z) * f (z, x) = 1) ∧
  (∀ x : ℤ, f (x + 1, x) = 2)

/-- Theorem stating that any function satisfying the special_function conditions 
    must be of the form f(x,y) = 2^(x-y) -/
theorem special_function_unique (f : ℤ × ℤ → ℝ) 
  (hf : special_function f) : 
  ∀ x y : ℤ, f (x, y) = 2^(x - y) := by
  sorry

end NUMINAMATH_CALUDE_special_function_unique_l1264_126485


namespace NUMINAMATH_CALUDE_max_min_product_l1264_126427

theorem max_min_product (a b : ℕ+) (h : a + b = 100) :
  (∀ x y : ℕ+, x + y = 100 → x * y ≤ a * b) → a * b = 2500 ∧
  (∀ x y : ℕ+, x + y = 100 → a * b ≤ x * y) → a * b = 99 :=
by sorry

end NUMINAMATH_CALUDE_max_min_product_l1264_126427


namespace NUMINAMATH_CALUDE_factorization_equality_l1264_126465

theorem factorization_equality (a : ℝ) : 2 * a^2 - 8 = 2 * (a + 2) * (a - 2) := by
  sorry

end NUMINAMATH_CALUDE_factorization_equality_l1264_126465


namespace NUMINAMATH_CALUDE_smallest_undefined_inverse_l1264_126499

theorem smallest_undefined_inverse (a : ℕ) : 
  (∀ b : ℕ, b < a → (Nat.gcd b 72 = 1 ∨ Nat.gcd b 45 = 1)) ∧ 
  Nat.gcd a 72 > 1 ∧ 
  Nat.gcd a 45 > 1 → 
  a = 3 := by
sorry

end NUMINAMATH_CALUDE_smallest_undefined_inverse_l1264_126499


namespace NUMINAMATH_CALUDE_rajesh_savings_amount_l1264_126468

/-- Calculates Rajesh's monthly savings based on his salary and spending habits -/
def rajesh_savings (monthly_salary : ℕ) : ℕ :=
  let food_expense := (40 * monthly_salary) / 100
  let medicine_expense := (20 * monthly_salary) / 100
  let remaining := monthly_salary - (food_expense + medicine_expense)
  (60 * remaining) / 100

/-- Theorem stating that Rajesh's monthly savings are 3600 given his salary and spending habits -/
theorem rajesh_savings_amount :
  rajesh_savings 15000 = 3600 := by
  sorry

end NUMINAMATH_CALUDE_rajesh_savings_amount_l1264_126468


namespace NUMINAMATH_CALUDE_min_distance_sum_l1264_126476

noncomputable def circle_C (x y : ℝ) : Prop := (x - 4)^2 + y^2 = 8

noncomputable def circle_D (x y : ℝ) : Prop := x^2 + (y - 4)^2 = 5

def origin : ℝ × ℝ := (0, 0)

theorem min_distance_sum (P Q : ℝ × ℝ) (hP : circle_C P.1 P.2) (hQ : circle_D Q.1 Q.2) :
  ∃ (min_val : ℝ), min_val = Real.sqrt 5 ∧
  ∀ (P' Q' : ℝ × ℝ), circle_C P'.1 P'.2 → circle_D Q'.1 Q'.2 →
    Real.sqrt ((P'.1 - Q'.1)^2 + (P'.2 - Q'.2)^2) + 
    (Real.sqrt 2 / 2) * Real.sqrt (P'.1^2 + P'.2^2) ≥ min_val :=
by sorry

end NUMINAMATH_CALUDE_min_distance_sum_l1264_126476


namespace NUMINAMATH_CALUDE_work_rate_problem_l1264_126463

theorem work_rate_problem (x y k : ℝ) : 
  x = k * y → 
  y = 1 / 80 → 
  x + y = 1 / 20 → 
  k = 3 := by sorry

end NUMINAMATH_CALUDE_work_rate_problem_l1264_126463


namespace NUMINAMATH_CALUDE_harmonic_progression_logarithm_equality_l1264_126475

/-- Given x, y, z form a harmonic progression, 
    prove that lg (x+z) + lg (x-2y+z) = 2 lg (x-z) -/
theorem harmonic_progression_logarithm_equality 
  (x y z : ℝ) 
  (h : (1/x + 1/z)/2 = 1/y) : 
  Real.log (x+z) + Real.log (x-2*y+z) = 2 * Real.log (x-z) := by
  sorry

end NUMINAMATH_CALUDE_harmonic_progression_logarithm_equality_l1264_126475


namespace NUMINAMATH_CALUDE_calculate_expression_l1264_126428

theorem calculate_expression : 
  (0.125 : ℝ)^8 * (-8 : ℝ)^7 = -0.125 := by
  sorry

end NUMINAMATH_CALUDE_calculate_expression_l1264_126428


namespace NUMINAMATH_CALUDE_apple_bag_price_apple_bag_price_is_8_l1264_126424

/-- Calculates the selling price of one bag of apples given the harvest and sales information. -/
theorem apple_bag_price (total_harvest : ℕ) (juice_amount : ℕ) (restaurant_amount : ℕ) 
  (bag_size : ℕ) (total_revenue : ℕ) : ℕ :=
  let remaining := total_harvest - juice_amount - restaurant_amount
  let num_bags := remaining / bag_size
  total_revenue / num_bags

/-- Proves that the selling price of one bag of apples is $8 given the specific harvest and sales information. -/
theorem apple_bag_price_is_8 :
  apple_bag_price 405 90 60 5 408 = 8 := by
  sorry

end NUMINAMATH_CALUDE_apple_bag_price_apple_bag_price_is_8_l1264_126424


namespace NUMINAMATH_CALUDE_problem_solution_l1264_126409

theorem problem_solution (m n : ℕ+) 
  (h1 : m.val + 5 < n.val)
  (h2 : (m.val + (m.val + 3) + (m.val + 5) + n.val + (n.val + 1) + (2 * n.val - 1)) / 6 = n.val)
  (h3 : (m.val + 5 + n.val) / 2 = n.val) : 
  m.val + n.val = 11 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l1264_126409


namespace NUMINAMATH_CALUDE_g_comp_three_roots_l1264_126466

/-- A quadratic function g(x) = x^2 + 4x + d where d is a real parameter -/
def g (d : ℝ) (x : ℝ) : ℝ := x^2 + 4*x + d

/-- The composition of g with itself -/
def g_comp (d : ℝ) (x : ℝ) : ℝ := g d (g d x)

/-- Theorem stating that g(g(x)) has exactly 3 distinct real roots iff d = 0 -/
theorem g_comp_three_roots (d : ℝ) : 
  (∃! (s : Finset ℝ), s.card = 3 ∧ ∀ x ∈ s, g_comp d x = 0) ↔ d = 0 := by
  sorry

end NUMINAMATH_CALUDE_g_comp_three_roots_l1264_126466


namespace NUMINAMATH_CALUDE_production_volume_proof_l1264_126452

/-- Represents the production volume equation over three years -/
def production_equation (x : ℝ) : Prop :=
  200 + 200 * (1 + x) + 200 * (1 + x)^2 = 1400

/-- 
  Proves that the given equation correctly represents the total production volume
  over three years, given an initial production of 200 units and a constant
  percentage increase x for two consecutive years, resulting in a total of 1400 units.
-/
theorem production_volume_proof (x : ℝ) : production_equation x := by
  sorry

end NUMINAMATH_CALUDE_production_volume_proof_l1264_126452


namespace NUMINAMATH_CALUDE_cone_height_ratio_l1264_126420

/-- Theorem about the ratio of heights in a cone with reduced height --/
theorem cone_height_ratio (original_height : ℝ) (base_circumference : ℝ) (new_volume : ℝ) :
  original_height = 20 →
  base_circumference = 18 * Real.pi →
  new_volume = 270 * Real.pi →
  ∃ (new_height : ℝ),
    (1 / 3 : ℝ) * Real.pi * (base_circumference / (2 * Real.pi))^2 * new_height = new_volume ∧
    new_height / original_height = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_cone_height_ratio_l1264_126420


namespace NUMINAMATH_CALUDE_sin_theta_plus_pi_fourth_l1264_126404

theorem sin_theta_plus_pi_fourth (θ : Real) 
  (h1 : θ > π/2 ∧ θ < π) 
  (h2 : Real.tan (θ - π/4) = -4/3) : 
  Real.sin (θ + π/4) = -3/5 := by
  sorry

end NUMINAMATH_CALUDE_sin_theta_plus_pi_fourth_l1264_126404


namespace NUMINAMATH_CALUDE_change_is_three_l1264_126425

/-- Calculates the change received after a restaurant visit -/
def calculate_change (lee_amount : ℕ) (friend_amount : ℕ) (wings_cost : ℕ) (salad_cost : ℕ) (soda_cost : ℕ) (soda_quantity : ℕ) (tax : ℕ) : ℕ :=
  let total_amount := lee_amount + friend_amount
  let food_cost := wings_cost + salad_cost + soda_cost * soda_quantity
  let total_cost := food_cost + tax
  total_amount - total_cost

/-- Proves that the change received is $3 given the specific conditions -/
theorem change_is_three :
  calculate_change 10 8 6 4 1 2 3 = 3 := by
  sorry

end NUMINAMATH_CALUDE_change_is_three_l1264_126425


namespace NUMINAMATH_CALUDE_square_rotation_cylinder_volume_l1264_126414

/-- The volume of a cylinder formed by rotating a square about its horizontal line of symmetry -/
theorem square_rotation_cylinder_volume (side_length : ℝ) (volume : ℝ) :
  side_length = 10 →
  volume = Real.pi * (side_length / 2)^2 * side_length →
  volume = 250 * Real.pi :=
by
  sorry

#check square_rotation_cylinder_volume

end NUMINAMATH_CALUDE_square_rotation_cylinder_volume_l1264_126414


namespace NUMINAMATH_CALUDE_volumes_not_equal_sufficient_not_necessary_for_areas_not_always_equal_l1264_126400

/-- Represents a geometric body -/
structure GeometricBody where
  volume : ℝ
  crossSectionArea : ℝ → ℝ  -- Function mapping height to cross-sectional area

/-- The Gougu Principle -/
axiom gougu_principle {A B : GeometricBody} (h : ∀ (height : ℝ), A.crossSectionArea height = B.crossSectionArea height) :
  A.volume = B.volume

/-- Two geometric bodies have the same height -/
def same_height (A B : GeometricBody) : Prop := true

theorem volumes_not_equal_sufficient_not_necessary_for_areas_not_always_equal
  (A B : GeometricBody) (h : same_height A B) :
  (∃ (height : ℝ), A.crossSectionArea height ≠ B.crossSectionArea height) ↔ 
  (A.volume ≠ B.volume ∨ (A.volume = B.volume ∧ ∃ (height : ℝ), A.crossSectionArea height ≠ B.crossSectionArea height)) :=
sorry

end NUMINAMATH_CALUDE_volumes_not_equal_sufficient_not_necessary_for_areas_not_always_equal_l1264_126400


namespace NUMINAMATH_CALUDE_parallel_vectors_x_value_l1264_126471

/-- Two vectors are parallel if one is a scalar multiple of the other -/
def parallel (a b : ℝ × ℝ) : Prop :=
  ∃ k : ℝ, a = (k * b.1, k * b.2)

theorem parallel_vectors_x_value :
  ∀ x : ℝ,
  let a : ℝ × ℝ := (x, 2)
  let b : ℝ × ℝ := (2, 1)
  parallel a b → x = 4 := by
sorry

end NUMINAMATH_CALUDE_parallel_vectors_x_value_l1264_126471


namespace NUMINAMATH_CALUDE_hyperbola_equation_l1264_126488

-- Define the circle C
def circle_C (x y : ℝ) : Prop := x^2 + y^2 - 6*x - 4*y + 8 = 0

-- Define the hyperbola with focus at (2, 0) and vertex at (4, 0)
def hyperbola_focus : ℝ × ℝ := (2, 0)
def hyperbola_vertex : ℝ × ℝ := (4, 0)

-- Theorem statement
theorem hyperbola_equation : 
  ∀ x y : ℝ, 
  circle_C x y →
  (hyperbola_focus = (2, 0) ∧ hyperbola_vertex = (4, 0)) →
  x^2/4 - y^2/12 = 1 :=
sorry

end NUMINAMATH_CALUDE_hyperbola_equation_l1264_126488


namespace NUMINAMATH_CALUDE_final_position_l1264_126408

-- Define the ant's position type
def Position := ℤ × ℤ

-- Define the direction type
inductive Direction
| East
| North
| West
| South

-- Define the function to get the next direction
def nextDirection (d : Direction) : Direction :=
  match d with
  | Direction.East => Direction.North
  | Direction.North => Direction.West
  | Direction.West => Direction.South
  | Direction.South => Direction.East

-- Define the function to move in a direction
def move (p : Position) (d : Direction) : Position :=
  match d with
  | Direction.East => (p.1 + 1, p.2)
  | Direction.North => (p.1, p.2 + 1)
  | Direction.West => (p.1 - 1, p.2)
  | Direction.South => (p.1, p.2 - 1)

-- Define the function to calculate the position after n steps
def positionAfterSteps (n : ℕ) : Position :=
  sorry -- Proof implementation goes here

-- The main theorem
theorem final_position : positionAfterSteps 2015 = (13, -22) := by
  sorry -- Proof implementation goes here

end NUMINAMATH_CALUDE_final_position_l1264_126408


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_24_l1264_126484

/-- An arithmetic sequence where a_n = 2n - 3 -/
def a (n : ℕ) : ℤ := 2 * n - 3

/-- Sum of the first n terms of the arithmetic sequence -/
def S (n : ℕ) : ℤ := (n : ℤ) * (2 * a 1 + (n - 1) * (a 2 - a 1)) / 2

theorem arithmetic_sequence_sum_24 :
  ∃ m : ℕ, m > 0 ∧ S m = 24 ∧ ∀ k : ℕ, k > 0 ∧ S k = 24 → k = m :=
by sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_24_l1264_126484


namespace NUMINAMATH_CALUDE_total_fishes_in_aquatic_reserve_l1264_126489

theorem total_fishes_in_aquatic_reserve (bodies_of_water : ℕ) (fishes_per_body : ℕ) 
  (h1 : bodies_of_water = 6) 
  (h2 : fishes_per_body = 175) : 
  bodies_of_water * fishes_per_body = 1050 := by
  sorry

end NUMINAMATH_CALUDE_total_fishes_in_aquatic_reserve_l1264_126489


namespace NUMINAMATH_CALUDE_beta_function_integral_l1264_126402

theorem beta_function_integral (p q : ℕ) :
  ∫ x in (0:ℝ)..1, x^p * (1-x)^q = (p.factorial * q.factorial) / (p+q+1).factorial :=
sorry

end NUMINAMATH_CALUDE_beta_function_integral_l1264_126402


namespace NUMINAMATH_CALUDE_cubic_function_constraint_l1264_126454

def f (a b c x : ℝ) : ℝ := 4 * x^3 + a * x^2 + b * x + c

theorem cubic_function_constraint (a b c : ℝ) :
  (∀ x : ℝ, -1 ≤ x ∧ x ≤ 1 → -1 ≤ f a b c x ∧ f a b c x ≤ 1) →
  a = 0 ∧ b = -3 ∧ c = 0 := by
  sorry

end NUMINAMATH_CALUDE_cubic_function_constraint_l1264_126454


namespace NUMINAMATH_CALUDE_point_on_line_l1264_126498

theorem point_on_line (m n : ℝ) :
  let line := fun (x y : ℝ) => x = y / 2 - 2 / 5
  let some_value := 2
  (line m n ∧ line (m + some_value) (n + 4)) →
  some_value = 2 := by
  sorry

end NUMINAMATH_CALUDE_point_on_line_l1264_126498
