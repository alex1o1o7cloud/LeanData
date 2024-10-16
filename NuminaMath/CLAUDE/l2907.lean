import Mathlib

namespace NUMINAMATH_CALUDE_angle_A_is_pi_over_four_angle_C_is_5pi_over_12_l2907_290761

-- Define a triangle
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ
  S : ℝ
  -- Ensure all sides and angles are positive
  ha : 0 < a
  hb : 0 < b
  hc : 0 < c
  hA : 0 < A ∧ A < π
  hB : 0 < B ∧ B < π
  hC : 0 < C ∧ C < π
  -- Ensure the sum of angles is π
  angle_sum : A + B + C = π
  -- Area formula
  area_formula : S = (1/2) * b * c * Real.sin A

-- Theorem 1
theorem angle_A_is_pi_over_four (t : Triangle) (h : t.a^2 + 4*t.S = t.b^2 + t.c^2) :
  t.A = π/4 := by sorry

-- Theorem 2
theorem angle_C_is_5pi_over_12 (t : Triangle) 
  (h1 : t.a^2 + 4*t.S = t.b^2 + t.c^2) (h2 : t.a = Real.sqrt 2) (h3 : t.b = Real.sqrt 3) :
  t.C = 5*π/12 := by sorry

end NUMINAMATH_CALUDE_angle_A_is_pi_over_four_angle_C_is_5pi_over_12_l2907_290761


namespace NUMINAMATH_CALUDE_triangle_properties_l2907_290736

-- Define the triangle ABC
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

-- Define the given conditions
def triangle_conditions (t : Triangle) : Prop :=
  t.b * Real.cos t.C + t.c * Real.cos t.B = 2 * (Real.sqrt 3 / 3) * t.a * Real.cos t.B ∧
  t.b = Real.sqrt 7 ∧
  t.c = 2 * Real.sqrt 3 ∧
  t.a > t.b

-- Theorem statement
theorem triangle_properties (t : Triangle) (h : triangle_conditions t) :
  t.B = π/6 ∧ (1/2 * t.b * t.c * Real.sin t.A = 3 * Real.sqrt 7 / 2) := by
  sorry

end NUMINAMATH_CALUDE_triangle_properties_l2907_290736


namespace NUMINAMATH_CALUDE_suitcase_weight_problem_l2907_290711

/-- Proves that given the initial ratio of books : clothes : electronics as 5 : 4 : 2, 
    and after removing 9 pounds of clothing, which doubles the ratio of books to clothes, 
    the weight of electronics is 9 pounds. -/
theorem suitcase_weight_problem (B C E : ℝ) : 
  B / C = 5 / 4 →  -- Initial ratio of books to clothes
  B / E = 5 / 2 →  -- Initial ratio of books to electronics
  B / (C - 9) = 10 / 4 →  -- New ratio after removing 9 pounds of clothes
  E = 9 := by
  sorry


end NUMINAMATH_CALUDE_suitcase_weight_problem_l2907_290711


namespace NUMINAMATH_CALUDE_route_distance_l2907_290778

theorem route_distance (time_Q : ℝ) (time_Y : ℝ) (speed_ratio : ℝ) :
  time_Q = 2 →
  time_Y = 4/3 →
  speed_ratio = 3/2 →
  ∃ (distance : ℝ) (speed_Q : ℝ),
    distance = speed_Q * time_Q ∧
    distance = (speed_ratio * speed_Q) * time_Y ∧
    distance = 3/2 := by
  sorry

end NUMINAMATH_CALUDE_route_distance_l2907_290778


namespace NUMINAMATH_CALUDE_gcd_triple_characterization_l2907_290756

theorem gcd_triple_characterization (a b c : ℕ+) :
  (Nat.gcd a.val 20 = b.val ∧ Nat.gcd b.val 15 = c.val ∧ Nat.gcd a.val c.val = 5) ↔
  (∃ t : ℕ+, (a = 20 * t ∧ b = 20 ∧ c = 5) ∨
             (a = 20 * t - 10 ∧ b = 10 ∧ c = 5) ∨
             (a = 10 * t - 5 ∧ b = 5 ∧ c = 5)) :=
by sorry


end NUMINAMATH_CALUDE_gcd_triple_characterization_l2907_290756


namespace NUMINAMATH_CALUDE_multiplicative_inverse_152_mod_367_l2907_290753

theorem multiplicative_inverse_152_mod_367 :
  ∃ a : ℕ, a < 367 ∧ (152 * a) % 367 = 1 ∧ a = 248 := by
  sorry

end NUMINAMATH_CALUDE_multiplicative_inverse_152_mod_367_l2907_290753


namespace NUMINAMATH_CALUDE_f_2_equals_neg_26_l2907_290774

-- Define the function f
def f (a b : ℝ) (x : ℝ) : ℝ := x^5 + a*x^3 + b*x - 8

-- State the theorem
theorem f_2_equals_neg_26 (a b : ℝ) :
  f a b (-2) = 10 → f a b 2 = -26 := by
  sorry

end NUMINAMATH_CALUDE_f_2_equals_neg_26_l2907_290774


namespace NUMINAMATH_CALUDE_m_range_l2907_290739

theorem m_range (m : ℝ) : 
  ¬((m + 1 ≤ 0) ∧ (∀ x : ℝ, x^2 + m*x + 1 > 0)) → 
  m ≤ -2 ∨ m > -1 := by
sorry

end NUMINAMATH_CALUDE_m_range_l2907_290739


namespace NUMINAMATH_CALUDE_integral_tangent_fraction_l2907_290740

theorem integral_tangent_fraction :
  ∫ x in -Real.arccos (1 / Real.sqrt 5)..0, (11 - 3 * Real.tan x) / (Real.tan x + 3) = Real.log 45 - 3 * Real.arctan 2 := by
  sorry

end NUMINAMATH_CALUDE_integral_tangent_fraction_l2907_290740


namespace NUMINAMATH_CALUDE_last_locker_opened_is_511_l2907_290741

/-- Represents the state of a locker (open or closed) -/
inductive LockerState
| Open
| Closed

/-- Represents the direction of the student's movement -/
inductive Direction
| Forward
| Backward

/-- Represents the student's position and movement direction -/
structure StudentPosition :=
  (position : Nat)
  (direction : Direction)

/-- Represents the state of all lockers -/
def LockerSystem := Fin 512 → LockerState

/-- The process of opening lockers according to the described pattern -/
def openLockers : LockerSystem → StudentPosition → LockerSystem
  | lockers, _ => sorry  -- Implementation details omitted

/-- Checks if all lockers are open -/
def allLockersOpen : LockerSystem → Bool
  | _ => sorry  -- Implementation details omitted

/-- Finds the number of the last closed locker -/
def lastClosedLocker : LockerSystem → Option Nat
  | _ => sorry  -- Implementation details omitted

/-- The main theorem stating that the last locker opened is 511 -/
theorem last_locker_opened_is_511 :
  let initial_lockers : LockerSystem := fun _ => LockerState.Closed
  let initial_position : StudentPosition := ⟨0, Direction.Forward⟩
  let final_lockers := openLockers initial_lockers initial_position
  allLockersOpen final_lockers ∧ lastClosedLocker final_lockers = some 511 :=
sorry


end NUMINAMATH_CALUDE_last_locker_opened_is_511_l2907_290741


namespace NUMINAMATH_CALUDE_cube_surface_area_from_prism_l2907_290751

/-- The surface area of a cube with the same volume as a rectangular prism -/
theorem cube_surface_area_from_prism (a b c : ℝ) (h1 : a = 5) (h2 : b = 4) (h3 : c = 40) :
  6 * (((a * b * c) ^ (1/3 : ℝ)) ^ 2) = 600 := by
  sorry

end NUMINAMATH_CALUDE_cube_surface_area_from_prism_l2907_290751


namespace NUMINAMATH_CALUDE_polynomial_has_real_root_l2907_290737

theorem polynomial_has_real_root (b : ℝ) : 
  ∃ x : ℝ, x^4 + b*x^3 + 2*x^2 + b*x - 2 = 0 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_has_real_root_l2907_290737


namespace NUMINAMATH_CALUDE_x_over_y_equals_two_l2907_290733

theorem x_over_y_equals_two 
  (x y z : ℝ) 
  (h_positive : x > 0 ∧ y > 0 ∧ z > 0) 
  (h_distinct : x ≠ y ∧ x ≠ z ∧ y ≠ z)
  (h_eq : y / (x - z) = 2 * (x + y) / z ∧ y / (x - z) = x / (2 * y)) :
  x / y = 2 := by
sorry

end NUMINAMATH_CALUDE_x_over_y_equals_two_l2907_290733


namespace NUMINAMATH_CALUDE_roberts_and_marias_ages_l2907_290797

theorem roberts_and_marias_ages (robert maria : ℕ) : 
  robert = maria + 8 →
  robert + 5 = 3 * (maria - 3) →
  robert + maria = 30 :=
by sorry

end NUMINAMATH_CALUDE_roberts_and_marias_ages_l2907_290797


namespace NUMINAMATH_CALUDE_alyosha_age_claim_possible_l2907_290765

-- Define a structure for dates
structure Date :=
  (year : ℕ)
  (month : ℕ)
  (day : ℕ)

-- Define a structure for a person's age and birthday
structure Person :=
  (birthday : Date)
  (current_date : Date)

def age (p : Person) : ℕ :=
  p.current_date.year - p.birthday.year

def is_birthday (p : Person) : Prop :=
  p.birthday.month = p.current_date.month ∧ p.birthday.day = p.current_date.day

-- Define Alyosha
def alyosha (birthday : Date) : Person :=
  { birthday := birthday,
    current_date := ⟨2024, 1, 1⟩ }  -- Assuming current year is 2024

-- Theorem statement
theorem alyosha_age_claim_possible :
  ∃ (birthday : Date),
    age (alyosha birthday) = 11 ∧
    age { birthday := birthday, current_date := ⟨2023, 12, 30⟩ } = 9 ∧
    age { birthday := birthday, current_date := ⟨2025, 1, 1⟩ } = 12 ↔
    birthday = ⟨2013, 12, 31⟩ :=
sorry

end NUMINAMATH_CALUDE_alyosha_age_claim_possible_l2907_290765


namespace NUMINAMATH_CALUDE_radish_carrot_ratio_l2907_290708

theorem radish_carrot_ratio :
  let cucumbers : ℕ := 15
  let radishes : ℕ := 3 * cucumbers
  let carrots : ℕ := 9
  radishes / carrots = 5 := by
sorry

end NUMINAMATH_CALUDE_radish_carrot_ratio_l2907_290708


namespace NUMINAMATH_CALUDE_problem_solution_l2907_290738

theorem problem_solution : 
  (∀ π : ℝ, (π - 2)^0 + (-1)^3 = 0) ∧ 
  (∀ m n : ℝ, (3*m + n) * (m - 2*n) = 3*m^2 - 5*m*n - 2*n^2) := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l2907_290738


namespace NUMINAMATH_CALUDE_discount_percentage_l2907_290787

theorem discount_percentage (original_price : ℝ) (discount_rate : ℝ) 
  (h1 : discount_rate = 0.8) : 
  original_price * (1 - discount_rate) = original_price * 0.8 := by
  sorry

#check discount_percentage

end NUMINAMATH_CALUDE_discount_percentage_l2907_290787


namespace NUMINAMATH_CALUDE_cricket_collection_l2907_290724

theorem cricket_collection (initial_crickets : ℕ) (additional_crickets : ℕ) : 
  initial_crickets = 7 → additional_crickets = 4 → initial_crickets + additional_crickets = 11 :=
by
  sorry

end NUMINAMATH_CALUDE_cricket_collection_l2907_290724


namespace NUMINAMATH_CALUDE_tank_emptying_time_l2907_290758

/-- Proves the time to empty a tank with given conditions -/
theorem tank_emptying_time 
  (tank_capacity : ℝ) 
  (leak_empty_time : ℝ) 
  (inlet_rate_per_minute : ℝ) 
  (h1 : tank_capacity = 4320)
  (h2 : leak_empty_time = 6)
  (h3 : inlet_rate_per_minute = 3) : 
  (tank_capacity / (tank_capacity / leak_empty_time - inlet_rate_per_minute * 60)) = 8 := by
  sorry

#check tank_emptying_time

end NUMINAMATH_CALUDE_tank_emptying_time_l2907_290758


namespace NUMINAMATH_CALUDE_intersection_angle_cosine_l2907_290749

/-- The ellipse C₁ -/
def C₁ (x y : ℝ) : Prop := x^2/6 + y^2/2 = 1

/-- The hyperbola C₂ -/
def C₂ (x y : ℝ) : Prop := x^2/3 - y^2 = 1

/-- The foci of both curves -/
def F₁ : ℝ × ℝ := (-2, 0)
def F₂ : ℝ × ℝ := (2, 0)

/-- The cosine of the angle F₁PF₂ -/
noncomputable def cos_angle (P : ℝ × ℝ) : ℝ :=
  let d₁ := Real.sqrt ((P.1 - F₁.1)^2 + (P.2 - F₁.2)^2)
  let d₂ := Real.sqrt ((P.2 - F₂.1)^2 + (P.2 - F₂.2)^2)
  let d := Real.sqrt ((F₂.1 - F₁.1)^2 + (F₂.2 - F₁.2)^2)
  (d₁^2 + d₂^2 - d^2) / (2 * d₁ * d₂)

theorem intersection_angle_cosine :
  ∀ (x y : ℝ), C₁ x y → C₂ x y → cos_angle (x, y) = 1/3 := by sorry

end NUMINAMATH_CALUDE_intersection_angle_cosine_l2907_290749


namespace NUMINAMATH_CALUDE_average_side_length_of_squares_l2907_290760

theorem average_side_length_of_squares (a₁ a₂ a₃ : ℝ) (h₁ : a₁ = 25) (h₂ : a₂ = 64) (h₃ : a₃ = 144) :
  (Real.sqrt a₁ + Real.sqrt a₂ + Real.sqrt a₃) / 3 = 25 / 3 := by
  sorry

end NUMINAMATH_CALUDE_average_side_length_of_squares_l2907_290760


namespace NUMINAMATH_CALUDE_sock_pair_combinations_l2907_290720

/-- The number of ways to choose a pair of socks of different colors -/
def different_color_sock_pairs (white brown blue red : ℕ) : ℕ :=
  white * brown + white * blue + white * red +
  brown * blue + brown * red +
  blue * red

/-- Theorem stating the number of ways to choose a pair of socks of different colors
    given the specific quantities of each color -/
theorem sock_pair_combinations :
  different_color_sock_pairs 5 3 3 1 = 50 := by
  sorry

end NUMINAMATH_CALUDE_sock_pair_combinations_l2907_290720


namespace NUMINAMATH_CALUDE_equation_solution_l2907_290782

theorem equation_solution :
  ∃! x : ℚ, 7 * (4 * x + 3) - 5 = -3 * (2 - 5 * x) ∧ x = -22 / 13 := by
sorry

end NUMINAMATH_CALUDE_equation_solution_l2907_290782


namespace NUMINAMATH_CALUDE_product_evaluation_l2907_290783

theorem product_evaluation : (3 + 1) * (3^2 + 1^2) * (3^4 + 1^4) = 3280 := by
  sorry

end NUMINAMATH_CALUDE_product_evaluation_l2907_290783


namespace NUMINAMATH_CALUDE_six_women_four_men_arrangements_l2907_290763

/-- The number of ways to arrange n indistinguishable objects of one type
    and m indistinguishable objects of another type in a row,
    such that no two objects of the same type are adjacent -/
def alternating_arrangements (n m : ℕ) : ℕ := sorry

/-- Theorem stating that there are 6 ways to arrange 6 women and 4 men
    alternately in a row -/
theorem six_women_four_men_arrangements :
  alternating_arrangements 6 4 = 6 := by sorry

end NUMINAMATH_CALUDE_six_women_four_men_arrangements_l2907_290763


namespace NUMINAMATH_CALUDE_mark_bread_making_time_l2907_290762

/-- The time it takes Mark to finish making bread -/
def bread_making_time (rise_time : ℕ) (rise_count : ℕ) (knead_time : ℕ) (bake_time : ℕ) : ℕ :=
  rise_time * rise_count + knead_time + bake_time

/-- Theorem stating the total time Mark takes to finish making the bread -/
theorem mark_bread_making_time :
  bread_making_time 120 2 10 30 = 280 := by
  sorry

end NUMINAMATH_CALUDE_mark_bread_making_time_l2907_290762


namespace NUMINAMATH_CALUDE_power_inequality_l2907_290726

theorem power_inequality (a b c d : ℝ) (h1 : 0 ≤ a) (h2 : a ≤ b) (h3 : b ≤ c) (h4 : c ≤ d) :
  a^b * b^c * c^d * d^a ≥ b^a * c^b * d^c * a^d := by
  sorry

end NUMINAMATH_CALUDE_power_inequality_l2907_290726


namespace NUMINAMATH_CALUDE_inequality_proof_l2907_290770

theorem inequality_proof (a b : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hab : a < b) :
  1 / (a * b^2) < 1 / (a^2 * b) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l2907_290770


namespace NUMINAMATH_CALUDE_tribe_leadership_proof_l2907_290735

def tribe_leadership_arrangements (n : ℕ) : ℕ :=
  n * (n - 1).choose 2 * (n - 3).choose 2 * (n - 5).choose 2

theorem tribe_leadership_proof (n : ℕ) (h : n = 11) :
  tribe_leadership_arrangements n = 207900 := by
  sorry

end NUMINAMATH_CALUDE_tribe_leadership_proof_l2907_290735


namespace NUMINAMATH_CALUDE_number_difference_proof_l2907_290759

theorem number_difference_proof (s l : ℕ) : 
  (∃ x : ℕ, l = 2 * s - x) →  -- One number is some less than twice another
  s + l = 39 →               -- Their sum is 39
  s = 14 →                   -- The smaller number is 14
  2 * s - l = 3 :=           -- The difference between twice the smaller number and the larger number is 3
by
  sorry

end NUMINAMATH_CALUDE_number_difference_proof_l2907_290759


namespace NUMINAMATH_CALUDE_partner_A_money_received_l2907_290725

/-- Calculates the money received by partner A in a business partnership --/
def money_received_by_A (total_profit : ℝ) : ℝ :=
  let management_share := 0.12 * total_profit
  let remaining_profit := total_profit - management_share
  let A_share_of_remaining := 0.35 * remaining_profit
  management_share + A_share_of_remaining

/-- Theorem stating that partner A receives Rs. 7062 given the problem conditions --/
theorem partner_A_money_received :
  money_received_by_A 16500 = 7062 := by
  sorry

#eval money_received_by_A 16500

end NUMINAMATH_CALUDE_partner_A_money_received_l2907_290725


namespace NUMINAMATH_CALUDE_largest_c_for_one_in_range_l2907_290700

-- Define the function f(x)
def f (c : ℝ) (x : ℝ) : ℝ := x^2 - 6*x + c

-- State the theorem
theorem largest_c_for_one_in_range : 
  (∃ (c : ℝ), ∀ (d : ℝ), (∃ (x : ℝ), f d x = 1) → d ≤ c) ∧ 
  (∃ (x : ℝ), f 10 x = 1) :=
by sorry

end NUMINAMATH_CALUDE_largest_c_for_one_in_range_l2907_290700


namespace NUMINAMATH_CALUDE_system_solution_l2907_290775

theorem system_solution :
  ∃! (x y : ℝ), 
    (x + 2*y = (7 - x) + (7 - 2*y)) ∧ 
    (3*x - 2*y = (x + 2) - (2*y + 2)) ∧
    x = 0 ∧ y = 7/2 := by
  sorry

end NUMINAMATH_CALUDE_system_solution_l2907_290775


namespace NUMINAMATH_CALUDE_pascal_triangle_51_numbers_l2907_290796

theorem pascal_triangle_51_numbers (n : ℕ) : 
  (n + 1 = 51) → Nat.choose n 2 = 1225 := by
  sorry

end NUMINAMATH_CALUDE_pascal_triangle_51_numbers_l2907_290796


namespace NUMINAMATH_CALUDE_value_of_a_l2907_290717

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := a * x + 2

-- State the theorem
theorem value_of_a (a : ℝ) :
  (∀ x, (deriv (f a)) x = a) →  -- The derivative of f is constant and equal to a
  (deriv (f a)) 1 = 2 →         -- The derivative of f at x = 1 is 2
  a = 2 :=                      -- Then a must be equal to 2
by
  sorry

end NUMINAMATH_CALUDE_value_of_a_l2907_290717


namespace NUMINAMATH_CALUDE_fraction_decomposition_l2907_290794

theorem fraction_decomposition :
  ∀ (x : ℝ) (C D : ℚ),
    (C / (x - 2) + D / (3 * x + 7) = (3 * x^2 + 7 * x - 20) / (3 * x^2 - x - 14)) →
    (C = -14/13 ∧ D = 81/13) := by
  sorry

end NUMINAMATH_CALUDE_fraction_decomposition_l2907_290794


namespace NUMINAMATH_CALUDE_proposition_count_l2907_290781

theorem proposition_count : 
  let prop1 := ∀ x : ℝ, x^2 + x + 1 ≥ 0
  let prop2 := ∀ x y : ℝ, (x + y ≠ 3) → (x ≠ 2 ∨ y ≠ 1)
  let prop3 := 
    let slope : ℝ := 1.23
    let center : ℝ × ℝ := (4, 5)
    (5 : ℝ) = slope * (4 : ℝ) + 0.08
  let prop4 := 
    ∀ m : ℝ, (m = 3 ↔ 
      ∀ x y : ℝ, ((m + 3) * x + m * y - 2 = 0 → m * x - 6 * y + 5 = 0) ∧
                 (m * x - 6 * y + 5 = 0 → (m + 3) * x + m * y - 2 = 0))
  (prop1 ∧ ¬prop2 ∧ ¬prop3 ∧ ¬prop4) ∨
  (¬prop1 ∧ prop2 ∧ ¬prop3 ∧ ¬prop4) ∨
  (¬prop1 ∧ ¬prop2 ∧ prop3 ∧ ¬prop4) ∨
  (¬prop1 ∧ ¬prop2 ∧ ¬prop3 ∧ prop4) ∨
  (prop1 ∧ prop2 ∧ ¬prop3 ∧ ¬prop4) ∨
  (prop1 ∧ ¬prop2 ∧ prop3 ∧ ¬prop4) ∨
  (prop1 ∧ ¬prop2 ∧ ¬prop3 ∧ prop4) ∨
  (¬prop1 ∧ prop2 ∧ prop3 ∧ ¬prop4) ∨
  (¬prop1 ∧ prop2 ∧ ¬prop3 ∧ prop4) ∨
  (¬prop1 ∧ ¬prop2 ∧ prop3 ∧ prop4) := by
  sorry

end NUMINAMATH_CALUDE_proposition_count_l2907_290781


namespace NUMINAMATH_CALUDE_age_difference_is_fifty_l2907_290799

/-- Represents the ages of family members in the year 2000 -/
structure FamilyAges where
  daughter : ℕ
  son : ℕ
  mother : ℕ
  father : ℕ

/-- Conditions given in the problem -/
def familyConditions (ages : FamilyAges) : Prop :=
  ages.mother = 4 * ages.daughter ∧
  ages.father = 6 * ages.son ∧
  ages.son = (3 * ages.daughter) / 2 ∧
  ages.father + 10 = 2 * (ages.mother + 10)

/-- The theorem to be proved -/
theorem age_difference_is_fifty (ages : FamilyAges) 
  (h : familyConditions ages) : ages.father - ages.mother = 50 := by
  sorry

#check age_difference_is_fifty

end NUMINAMATH_CALUDE_age_difference_is_fifty_l2907_290799


namespace NUMINAMATH_CALUDE_different_suit_card_combinations_l2907_290727

/-- The number of cards in a standard deck -/
def standard_deck_size : ℕ := 52

/-- The number of suits in a standard deck -/
def number_of_suits : ℕ := 4

/-- The number of cards per suit in a standard deck -/
def cards_per_suit : ℕ := standard_deck_size / number_of_suits

/-- The number of cards to be chosen -/
def cards_to_choose : ℕ := 4

-- Theorem statement
theorem different_suit_card_combinations :
  (number_of_suits.choose cards_to_choose) * (cards_per_suit ^ cards_to_choose) = 28561 := by
  sorry

end NUMINAMATH_CALUDE_different_suit_card_combinations_l2907_290727


namespace NUMINAMATH_CALUDE_midpoint_sum_invariant_l2907_290779

/-- Represents a polygon with n vertices in the Cartesian plane -/
structure Polygon (n : ℕ) where
  vertices : Fin n → ℝ × ℝ

/-- Creates a new polygon from the midpoints of the sides of the given polygon -/
def midpointPolygon {n : ℕ} (p : Polygon n) : Polygon n := sorry

/-- Sums the x-coordinates of a polygon's vertices -/
def sumXCoordinates {n : ℕ} (p : Polygon n) : ℝ := sorry

theorem midpoint_sum_invariant (p₁ : Polygon 200) 
  (h : sumXCoordinates p₁ = 4018) :
  let p₂ := midpointPolygon p₁
  let p₃ := midpointPolygon p₂
  let p₄ := midpointPolygon p₃
  sumXCoordinates p₄ = 4018 := by sorry

end NUMINAMATH_CALUDE_midpoint_sum_invariant_l2907_290779


namespace NUMINAMATH_CALUDE_investment_ratio_l2907_290757

/-- Prove that given the conditions, the ratio of B's investment to C's investment is 2:3 -/
theorem investment_ratio (a_invest b_invest c_invest : ℚ) 
  (h1 : a_invest = 3 * b_invest)
  (h2 : ∃ f : ℚ, b_invest = f * c_invest)
  (h3 : b_invest / (a_invest + b_invest + c_invest) * 7700 = 1400) :
  b_invest / c_invest = 2 / 3 := by
  sorry

end NUMINAMATH_CALUDE_investment_ratio_l2907_290757


namespace NUMINAMATH_CALUDE_tetrahedron_PQRS_volume_l2907_290784

def tetrahedron_volume (PQ PR PS QR QS RS : ℝ) : ℝ := sorry

theorem tetrahedron_PQRS_volume :
  let PQ : ℝ := 3
  let PR : ℝ := Real.sqrt 10
  let PS : ℝ := Real.sqrt 17
  let QR : ℝ := 5
  let QS : ℝ := 3 * Real.sqrt 2
  let RS : ℝ := 6
  let z : ℝ := Real.sqrt (17 - (4/3)^2 - (1/(2*Real.sqrt 10))^2)
  tetrahedron_volume PQ PR PS QR QS RS = (Real.sqrt 10 / 2) * z := by sorry

end NUMINAMATH_CALUDE_tetrahedron_PQRS_volume_l2907_290784


namespace NUMINAMATH_CALUDE_constant_term_is_60_l2907_290745

/-- The constant term in the binomial expansion of (2x^2 - 1/x)^6 -/
def constant_term : ℤ :=
  (Finset.range 7).sum (fun r => 
    (-1)^r * (Nat.choose 6 r) * 2^(6-r) * 
    if 12 - 3*r = 0 then 1 else 0)

/-- The constant term in the binomial expansion of (2x^2 - 1/x)^6 is 60 -/
theorem constant_term_is_60 : constant_term = 60 := by
  sorry

end NUMINAMATH_CALUDE_constant_term_is_60_l2907_290745


namespace NUMINAMATH_CALUDE_ngo_employees_proof_l2907_290793

/-- The number of literate employees in an NGO -/
def num_literate_employees : ℕ := 10

theorem ngo_employees_proof :
  let total_employees := num_literate_employees + 20
  let illiterate_wage_decrease := 300
  let total_wage_decrease := total_employees * 10
  illiterate_wage_decrease = total_wage_decrease →
  num_literate_employees = 10 := by
sorry

end NUMINAMATH_CALUDE_ngo_employees_proof_l2907_290793


namespace NUMINAMATH_CALUDE_volcano_eruption_percentage_l2907_290750

/-- Proves that the percentage of remaining volcanoes that exploded at the end of the year is 50% -/
theorem volcano_eruption_percentage (total : ℕ) (first_eruption : ℚ) (second_eruption : ℚ) (intact : ℕ) : 
  total = 200 →
  first_eruption = 1/5 →
  second_eruption = 2/5 →
  intact = 48 →
  let remaining_after_first := total - (first_eruption * total).num
  let remaining_after_second := remaining_after_first - (second_eruption * remaining_after_first).num
  let final_eruption := remaining_after_second - intact
  (final_eruption : ℚ) / remaining_after_second = 1/2 := by sorry

end NUMINAMATH_CALUDE_volcano_eruption_percentage_l2907_290750


namespace NUMINAMATH_CALUDE_expression_equality_l2907_290752

theorem expression_equality : 
  Real.sqrt 4 * 4^(1/2 : ℝ) + 16 / 4 * 2 - Real.sqrt 8 = 12 - 2 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_expression_equality_l2907_290752


namespace NUMINAMATH_CALUDE_mathematicians_ages_l2907_290713

/-- Represents a mathematician --/
inductive Mathematician
| A
| B
| C

/-- Calculates the age of mathematician A or C given the base and smallest number --/
def calculate_age_A_C (base : ℕ) (smallest : ℕ) : ℕ :=
  smallest * base + (smallest + 2)

/-- Calculates the age of mathematician B given the base and smallest number --/
def calculate_age_B (base : ℕ) (smallest : ℕ) : ℕ :=
  smallest * base + (smallest + 1)

/-- Checks if the calculated age matches the product of the two largest numbers --/
def check_age_A_C (age : ℕ) (smallest : ℕ) : Prop :=
  age = (smallest + 4) * (smallest + 6)

/-- Checks if the calculated age matches the product of the next two consecutive numbers --/
def check_age_B (age : ℕ) (smallest : ℕ) : Prop :=
  age = (smallest + 2) * (smallest + 3)

theorem mathematicians_ages :
  ∃ (age_A age_B age_C : ℕ) (base_A base_B : ℕ) (smallest_A smallest_B smallest_C : ℕ),
    calculate_age_A_C base_A smallest_A = age_A ∧
    calculate_age_B base_B smallest_B = age_B ∧
    calculate_age_A_C base_A smallest_C = age_C ∧
    check_age_A_C age_A smallest_A ∧
    check_age_B age_B smallest_B ∧
    check_age_A_C age_C smallest_C ∧
    age_C < age_A ∧
    age_C < age_B ∧
    age_A = 48 ∧
    age_B = 56 ∧
    age_C = 35 ∧
    base_B = 10 :=
  by sorry

/-- Identifies the absent-minded mathematician --/
def absent_minded : Mathematician := Mathematician.B

end NUMINAMATH_CALUDE_mathematicians_ages_l2907_290713


namespace NUMINAMATH_CALUDE_polynomial_sum_l2907_290798

def f (x : ℝ) : ℝ := -4 * x^2 + 2 * x - 5
def g (x : ℝ) : ℝ := -6 * x^2 + 4 * x - 9
def h (x : ℝ) : ℝ := 6 * x^2 + 6 * x + 2

theorem polynomial_sum (x : ℝ) : f x + g x + h x = -4 * x^2 + 12 * x - 12 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_sum_l2907_290798


namespace NUMINAMATH_CALUDE_swimmer_distance_l2907_290780

/-- Calculates the distance swum against a current given the swimmer's speed in still water,
    the speed of the current, and the time taken. -/
def distance_against_current (swimmer_speed : ℝ) (current_speed : ℝ) (time : ℝ) : ℝ :=
  (swimmer_speed - current_speed) * time

/-- Proves that given the specified conditions, the swimmer travels 6 km against the current. -/
theorem swimmer_distance :
  let swimmer_speed := 4
  let current_speed := 2
  let time := 3
  distance_against_current swimmer_speed current_speed time = 6 := by
sorry

end NUMINAMATH_CALUDE_swimmer_distance_l2907_290780


namespace NUMINAMATH_CALUDE_teachers_per_grade_l2907_290772

theorem teachers_per_grade (fifth_graders sixth_graders seventh_graders : ℕ)
  (parents_per_grade : ℕ) (num_buses seat_per_bus : ℕ) (num_grades : ℕ) :
  fifth_graders = 109 →
  sixth_graders = 115 →
  seventh_graders = 118 →
  parents_per_grade = 2 →
  num_buses = 5 →
  seat_per_bus = 72 →
  num_grades = 3 →
  (num_buses * seat_per_bus - (fifth_graders + sixth_graders + seventh_graders + parents_per_grade * num_grades)) / num_grades = 4 := by
  sorry

end NUMINAMATH_CALUDE_teachers_per_grade_l2907_290772


namespace NUMINAMATH_CALUDE_smallest_number_divisibility_l2907_290730

theorem smallest_number_divisibility (x : ℕ) : x = 1621432330 ↔ 
  (∀ y : ℕ, y < x → ¬(29 ∣ 5*(y+11) ∧ 53 ∣ 5*(y+11) ∧ 37 ∣ 5*(y+11) ∧ 
                     41 ∣ 5*(y+11) ∧ 47 ∣ 5*(y+11) ∧ 61 ∣ 5*(y+11))) ∧
  (29 ∣ 5*(x+11) ∧ 53 ∣ 5*(x+11) ∧ 37 ∣ 5*(x+11) ∧ 
   41 ∣ 5*(x+11) ∧ 47 ∣ 5*(x+11) ∧ 61 ∣ 5*(x+11)) := by
  sorry

end NUMINAMATH_CALUDE_smallest_number_divisibility_l2907_290730


namespace NUMINAMATH_CALUDE_circle_intersection_slope_range_l2907_290702

-- Define the circle C
def circle_C (x y : ℝ) : Prop :=
  (x - 1)^2 + y^2 = 25

-- Define the line that contains the center of C
def center_line (x y : ℝ) : Prop :=
  2*x - y - 2 = 0

-- Define the line l
def line_l (k : ℝ) (x y : ℝ) : Prop :=
  y - 5 = k*(x + 2)

-- Main theorem
theorem circle_intersection_slope_range :
  ∀ k : ℝ,
  (∃ x₁ y₁ x₂ y₂ : ℝ,
    -- C passes through M(-3,3) and N(1,-5)
    circle_C (-3) 3 ∧ circle_C 1 (-5) ∧
    -- Center of C lies on the given line
    ∃ xc yc : ℝ, circle_C xc yc ∧ center_line xc yc ∧
    -- l intersects C at two distinct points
    x₁ ≠ x₂ ∧ circle_C x₁ y₁ ∧ circle_C x₂ y₂ ∧
    line_l k x₁ y₁ ∧ line_l k x₂ y₂ ∧
    -- l passes through (-2,5)
    line_l k (-2) 5 ∧
    -- k > 0
    k > 0) →
  k > 15/8 :=
by sorry


end NUMINAMATH_CALUDE_circle_intersection_slope_range_l2907_290702


namespace NUMINAMATH_CALUDE_ellipse_major_axis_length_l2907_290776

/-- The length of the major axis of an ellipse formed by the intersection of a plane and a right circular cylinder -/
def majorAxisLength (cylinderRadius : ℝ) (majorAxisRatio : ℝ) : ℝ :=
  2 * cylinderRadius * (1 + majorAxisRatio)

/-- Theorem: The length of the major axis is 7 for the given conditions -/
theorem ellipse_major_axis_length :
  majorAxisLength 2 0.75 = 7 := by
  sorry

end NUMINAMATH_CALUDE_ellipse_major_axis_length_l2907_290776


namespace NUMINAMATH_CALUDE_scientific_notation_of_11930000_l2907_290719

/-- Proves that 11,930,000 is equal to 1.193 × 10^7 in scientific notation -/
theorem scientific_notation_of_11930000 : 
  11930000 = 1.193 * (10 : ℝ)^7 := by sorry

end NUMINAMATH_CALUDE_scientific_notation_of_11930000_l2907_290719


namespace NUMINAMATH_CALUDE_simplify_expression_l2907_290705

theorem simplify_expression (x : ℝ) : x + 3 - 4*x - 5 + 6*x + 7 - 8*x - 9 = -5*x - 4 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l2907_290705


namespace NUMINAMATH_CALUDE_zacks_marbles_l2907_290742

theorem zacks_marbles (initial_marbles : ℕ) : 
  (initial_marbles - 5) % 3 = 0 → 
  initial_marbles = 3 * 20 + 5 → 
  initial_marbles = 65 := by
sorry

end NUMINAMATH_CALUDE_zacks_marbles_l2907_290742


namespace NUMINAMATH_CALUDE_hyperbola_asymptotes_l2907_290791

/-- Given a hyperbola with equation x²/9 - y²/b² = 1 and foci at (-5,0) and (5,0),
    prove that its asymptotes have the equation 4x ± 3y = 0 -/
theorem hyperbola_asymptotes (b : ℝ) (h1 : b > 0) (h2 : 9 + b^2 = 25) :
  ∃ (k : ℝ), k > 0 ∧ 
  (∀ (x y : ℝ), (x^2 / 9 - y^2 / b^2 = 1) → 
   ((4*x + 3*y = 0) ∨ (4*x - 3*y = 0))) :=
sorry

end NUMINAMATH_CALUDE_hyperbola_asymptotes_l2907_290791


namespace NUMINAMATH_CALUDE_smallest_n_for_integer_T_l2907_290704

-- Define T_n as a function of n
def T (n : ℕ) : ℚ := sorry

-- Define the property of being the smallest positive integer n for which T_n is an integer
def is_smallest_integer_T (n : ℕ) : Prop :=
  (T n).isInt ∧ ∀ m : ℕ, m < n → ¬(T m).isInt

-- Theorem statement
theorem smallest_n_for_integer_T :
  is_smallest_integer_T 504 := by sorry

end NUMINAMATH_CALUDE_smallest_n_for_integer_T_l2907_290704


namespace NUMINAMATH_CALUDE_assign_roles_specific_scenario_l2907_290732

/-- Represents the number of ways to assign roles in a play. -/
def assignRoles (maleRoles femaleRoles eitherRoles maleActors femaleActors : ℕ) : ℕ :=
  (maleActors.descFactorial maleRoles) *
  (femaleActors.descFactorial femaleRoles) *
  ((maleActors + femaleActors - maleRoles - femaleRoles).descFactorial eitherRoles)

/-- Theorem stating the number of ways to assign roles in the specific scenario. -/
theorem assign_roles_specific_scenario :
  assignRoles 2 2 3 4 5 = 14400 := by
  sorry

end NUMINAMATH_CALUDE_assign_roles_specific_scenario_l2907_290732


namespace NUMINAMATH_CALUDE_system_solution_proof_l2907_290744

theorem system_solution_proof :
  ∃ (x y z : ℝ),
    x = 48 ∧ y = 16 ∧ z = 12 ∧
    (x * y) / (5 * x + 4 * y) = 6 ∧
    (x * z) / (3 * x + 2 * z) = 8 ∧
    (y * z) / (3 * y + 5 * z) = 6 := by
  sorry

end NUMINAMATH_CALUDE_system_solution_proof_l2907_290744


namespace NUMINAMATH_CALUDE_ellipse_major_axis_length_l2907_290769

/-- Given an ellipse with equation 16x^2 + 9y^2 = 144, its major axis length is 8 -/
theorem ellipse_major_axis_length :
  ∀ (x y : ℝ), 16 * x^2 + 9 * y^2 = 144 → ∃ (a b : ℝ), 
    x^2 / a^2 + y^2 / b^2 = 1 ∧ 
    ((a ≥ b ∧ 2 * a = 8) ∨ (b > a ∧ 2 * b = 8)) :=
by sorry

end NUMINAMATH_CALUDE_ellipse_major_axis_length_l2907_290769


namespace NUMINAMATH_CALUDE_mango_crates_sold_l2907_290706

-- Define the types of fruit
inductive Fruit
  | Grapes
  | Mangoes
  | PassionFruits

-- Define the total number of crates sold
def total_crates : ℕ := 50

-- Define the number of grape crates sold
def grape_crates : ℕ := 13

-- Define the number of passion fruit crates sold
def passion_fruit_crates : ℕ := 17

-- Define the function to calculate the number of mango crates
def mango_crates : ℕ := total_crates - (grape_crates + passion_fruit_crates)

-- Theorem statement
theorem mango_crates_sold : mango_crates = 20 := by
  sorry

end NUMINAMATH_CALUDE_mango_crates_sold_l2907_290706


namespace NUMINAMATH_CALUDE_A_empty_iff_a_in_range_l2907_290723

/-- The set of solutions to the quadratic equation ax^2 - ax + 1 = 0 -/
def A (a : ℝ) : Set ℝ := {x : ℝ | a * x^2 - a * x + 1 = 0}

/-- The theorem stating that A is empty if and only if a is in [0, 4) -/
theorem A_empty_iff_a_in_range : 
  ∀ a : ℝ, A a = ∅ ↔ 0 ≤ a ∧ a < 4 := by sorry

end NUMINAMATH_CALUDE_A_empty_iff_a_in_range_l2907_290723


namespace NUMINAMATH_CALUDE_negation_of_universal_proposition_l2907_290709

theorem negation_of_universal_proposition :
  (¬ ∀ x : ℝ, x > Real.sin x) ↔ (∃ x : ℝ, x ≤ Real.sin x) := by sorry

end NUMINAMATH_CALUDE_negation_of_universal_proposition_l2907_290709


namespace NUMINAMATH_CALUDE_solution_existence_implies_a_bound_l2907_290773

theorem solution_existence_implies_a_bound (a : ℝ) :
  (∃ x : ℝ, |x + a| + |x - 2| + a < 2010) → a < 1006 := by
  sorry

end NUMINAMATH_CALUDE_solution_existence_implies_a_bound_l2907_290773


namespace NUMINAMATH_CALUDE_investment_rate_problem_l2907_290747

theorem investment_rate_problem (total_investment : ℝ) (amount_at_eight_percent : ℝ) (income_difference : ℝ) (R : ℝ) :
  total_investment = 2000 →
  amount_at_eight_percent = 600 →
  income_difference = 92 →
  (total_investment - amount_at_eight_percent) * R - amount_at_eight_percent * 0.08 = income_difference →
  R = 0.1 := by
  sorry

end NUMINAMATH_CALUDE_investment_rate_problem_l2907_290747


namespace NUMINAMATH_CALUDE_unique_divisor_perfect_square_l2907_290755

theorem unique_divisor_perfect_square (p n : ℕ) (hp : Prime p) (hp2 : p > 2) :
  ∃! d : ℕ, d ∣ (p * n^2) ∧ ∃ m : ℕ, n^2 + d = m^2 :=
sorry

end NUMINAMATH_CALUDE_unique_divisor_perfect_square_l2907_290755


namespace NUMINAMATH_CALUDE_furniture_purchase_price_l2907_290703

theorem furniture_purchase_price :
  let marked_price : ℝ := 132
  let discount_rate : ℝ := 0.1
  let profit_rate : ℝ := 0.1
  let selling_price : ℝ := marked_price * (1 - discount_rate)
  ∃ (purchase_price : ℝ),
    selling_price - purchase_price = profit_rate * purchase_price ∧
    purchase_price = 108 :=
by sorry

end NUMINAMATH_CALUDE_furniture_purchase_price_l2907_290703


namespace NUMINAMATH_CALUDE_mikes_shopping_l2907_290771

/-- Mike's shopping problem -/
theorem mikes_shopping (food wallet shirt : ℝ) 
  (h1 : shirt = wallet / 3)
  (h2 : wallet = food + 60)
  (h3 : shirt + wallet + food = 150) :
  food = 52.5 := by
  sorry

end NUMINAMATH_CALUDE_mikes_shopping_l2907_290771


namespace NUMINAMATH_CALUDE_square_difference_times_three_l2907_290746

theorem square_difference_times_three : (538^2 - 462^2) * 3 = 228000 := by
  sorry

end NUMINAMATH_CALUDE_square_difference_times_three_l2907_290746


namespace NUMINAMATH_CALUDE_smallest_reducible_n_is_correct_l2907_290766

/-- The smallest positive integer n for which (n-17)/(6n+8) is non-zero and reducible -/
def smallest_reducible_n : ℕ := 127

/-- A fraction is reducible if the GCD of its numerator and denominator is greater than 1 -/
def is_reducible (n : ℕ) : Prop :=
  Nat.gcd (n - 17) (6 * n + 8) > 1

theorem smallest_reducible_n_is_correct :
  (∀ k : ℕ, k > 0 ∧ k < smallest_reducible_n → ¬(is_reducible k)) ∧
  (smallest_reducible_n > 0) ∧
  (is_reducible smallest_reducible_n) :=
sorry

end NUMINAMATH_CALUDE_smallest_reducible_n_is_correct_l2907_290766


namespace NUMINAMATH_CALUDE_fair_ride_cost_l2907_290731

theorem fair_ride_cost (total_tickets : ℕ) (spent_tickets : ℕ) (num_rides : ℕ) 
  (h1 : total_tickets = 79)
  (h2 : spent_tickets = 23)
  (h3 : num_rides = 8)
  (h4 : total_tickets ≥ spent_tickets) :
  (total_tickets - spent_tickets) / num_rides = 7 := by
sorry

end NUMINAMATH_CALUDE_fair_ride_cost_l2907_290731


namespace NUMINAMATH_CALUDE_gingers_size_l2907_290701

theorem gingers_size (anna_size becky_size ginger_size : ℕ) : 
  anna_size = 2 →
  becky_size = 3 * anna_size →
  ginger_size = 2 * becky_size - 4 →
  ginger_size = 8 := by
sorry

end NUMINAMATH_CALUDE_gingers_size_l2907_290701


namespace NUMINAMATH_CALUDE_max_product_of_three_integers_l2907_290722

/-- 
Given three integers where two are equal and their sum is 2000,
prove that their maximum product is 8000000000/27.
-/
theorem max_product_of_three_integers (x y z : ℤ) : 
  x = y ∧ x + y + z = 2000 → 
  x * y * z ≤ 8000000000 / 27 := by
sorry

end NUMINAMATH_CALUDE_max_product_of_three_integers_l2907_290722


namespace NUMINAMATH_CALUDE_rocket_max_height_l2907_290788

/-- The height function of the rocket -/
def h (t : ℝ) : ℝ := -20 * t^2 + 80 * t + 50

/-- The maximum height reached by the rocket -/
theorem rocket_max_height :
  ∃ (t_max : ℝ), ∀ (t : ℝ), h t ≤ h t_max ∧ h t_max = 130 :=
sorry

end NUMINAMATH_CALUDE_rocket_max_height_l2907_290788


namespace NUMINAMATH_CALUDE_shaded_area_rectangle_triangle_l2907_290728

/-- Given a rectangle with width 8 and height 12, and a right triangle with base 6 and height 8,
    prove that the area of the shaded region formed by a segment connecting the top-left vertex
    of the rectangle to the farthest vertex of the triangle is 120 square units. -/
theorem shaded_area_rectangle_triangle (rectangle_width : ℝ) (rectangle_height : ℝ)
    (triangle_base : ℝ) (triangle_height : ℝ) :
  rectangle_width = 8 →
  rectangle_height = 12 →
  triangle_base = 6 →
  triangle_height = 8 →
  let rectangle_area := rectangle_width * rectangle_height
  let triangle_area := (1 / 2) * triangle_base * triangle_height
  let shaded_area := rectangle_area + triangle_area
  shaded_area = 120 := by
  sorry

end NUMINAMATH_CALUDE_shaded_area_rectangle_triangle_l2907_290728


namespace NUMINAMATH_CALUDE_triangle_side_ratio_l2907_290734

theorem triangle_side_ratio 
  (z₁ z₂ z₃ : ℂ) 
  (h_distinct : z₁ ≠ z₂ ∧ z₂ ≠ z₃ ∧ z₃ ≠ z₁) 
  (h_eq : 4 * z₁^2 + 5 * z₂^2 + 5 * z₃^2 = 4 * z₁ * z₂ + 6 * z₂ * z₃ + 4 * z₃ * z₁) :
  ∃ (a b c : ℝ), 
    0 < a ∧ a ≤ b ∧ b ≤ c ∧
    Complex.abs (z₂ - z₁) = a ∧
    Complex.abs (z₃ - z₂) = b ∧
    Complex.abs (z₁ - z₃) = c ∧
    ∃ (k : ℝ), k > 0 ∧ a = 2 * k ∧ b = Real.sqrt 5 * k ∧ c = Real.sqrt 5 * k :=
by sorry

end NUMINAMATH_CALUDE_triangle_side_ratio_l2907_290734


namespace NUMINAMATH_CALUDE_division_problem_l2907_290729

theorem division_problem (x : ℝ) (h : (120 / x) - 15 = 5) : x = 6 := by
  sorry

end NUMINAMATH_CALUDE_division_problem_l2907_290729


namespace NUMINAMATH_CALUDE_shooter_stability_l2907_290714

/-- A shooter's score set -/
structure ScoreSet where
  scores : Finset ℝ
  card_eq : scores.card = 10

/-- Standard deviation of a score set -/
def standardDeviation (s : ScoreSet) : ℝ := sorry

/-- Dispersion of a score set -/
def dispersion (s : ScoreSet) : ℝ := sorry

/-- Larger standard deviation implies greater dispersion -/
axiom std_dev_dispersion_relation (s₁ s₂ : ScoreSet) :
  standardDeviation s₁ > standardDeviation s₂ → dispersion s₁ > dispersion s₂

theorem shooter_stability (A B : ScoreSet) :
  standardDeviation A > standardDeviation B →
  dispersion A > dispersion B :=
by sorry

end NUMINAMATH_CALUDE_shooter_stability_l2907_290714


namespace NUMINAMATH_CALUDE_sqrt_inequality_l2907_290743

theorem sqrt_inequality (p q x₁ x₂ : ℝ) (hp : p > 0) (hq : q > 0) (hpq : p + q = 1) :
  p * Real.sqrt x₁ + q * Real.sqrt x₂ ≤ Real.sqrt (p * x₁ + q * x₂) := by
  sorry

end NUMINAMATH_CALUDE_sqrt_inequality_l2907_290743


namespace NUMINAMATH_CALUDE_amicable_iff_ge_seven_l2907_290785

/-- An integer n ≥ 2 is amicable if there exist subsets A₁, A₂, ..., Aₙ of {1, 2, ..., n} satisfying:
    (i) i ∉ Aᵢ for any i = 1, 2, ..., n
    (ii) i ∈ Aⱼ for any j ∉ Aᵢ, for any i ≠ j
    (iii) Aᵢ ∩ Aⱼ ≠ ∅ for any i, j ∈ {1, 2, ..., n} -/
def IsAmicable (n : ℕ) : Prop :=
  n ≥ 2 ∧
  ∃ A : Fin n → Set (Fin n),
    (∀ i, i ∉ A i) ∧
    (∀ i j, i ≠ j → (j ∉ A i ↔ i ∈ A j)) ∧
    (∀ i j, (A i ∩ A j).Nonempty)

/-- For any integer n ≥ 2, n is amicable if and only if n ≥ 7 -/
theorem amicable_iff_ge_seven (n : ℕ) : IsAmicable n ↔ n ≥ 7 := by
  sorry

end NUMINAMATH_CALUDE_amicable_iff_ge_seven_l2907_290785


namespace NUMINAMATH_CALUDE_problem_solution_l2907_290768

theorem problem_solution (x : ℝ) (h : x + 1/x = 3) :
  (x - 1)^2 + 16/(x - 1)^2 = 7 + 3 * Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l2907_290768


namespace NUMINAMATH_CALUDE_third_shot_combination_l2907_290767

/-- Represents the scores of a shooter -/
structure ShooterScores where
  first_three : Fin 3 → Nat
  last_three : Fin 3 → Nat

/-- The set of all possible scores on the target -/
def target_scores : Finset Nat := {10, 9, 9, 8, 8, 5, 4, 4, 3, 2}

/-- Checks if the given scores are valid according to the target -/
def valid_scores (scores : ShooterScores) : Prop :=
  (∀ i, scores.first_three i ∈ target_scores) ∧
  (∀ i, scores.last_three i ∈ target_scores)

/-- The sum of the first three shots -/
def sum_first_three (scores : ShooterScores) : Nat :=
  (scores.first_three 0) + (scores.first_three 1) + (scores.first_three 2)

/-- The sum of the last three shots -/
def sum_last_three (scores : ShooterScores) : Nat :=
  (scores.last_three 0) + (scores.last_three 1) + (scores.last_three 2)

/-- Theorem stating the only possible combination for Petya and Vasya's third shots -/
theorem third_shot_combination (petya vasya : ShooterScores) 
  (h1 : valid_scores petya)
  (h2 : valid_scores vasya)
  (h3 : sum_first_three petya = sum_first_three vasya)
  (h4 : sum_last_three petya = 3 * sum_last_three vasya)
  (h5 : ∀ s ∈ target_scores, (petya.first_three 2 = s ∧ vasya.first_three 2 = s) → 
        s = 10 ∨ s = 2) :
  petya.first_three 2 = 10 ∧ vasya.first_three 2 = 2 := by
  sorry

end NUMINAMATH_CALUDE_third_shot_combination_l2907_290767


namespace NUMINAMATH_CALUDE_line_perp_plane_implies_perp_line_l2907_290777

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the perpendicular relation between a line and a plane
variable (perpendicular_line_plane : Line → Plane → Prop)

-- Define the perpendicular relation between two lines
variable (perpendicular_lines : Line → Line → Prop)

-- Define the "contained in" relation for a line in a plane
variable (contained_in : Line → Plane → Prop)

theorem line_perp_plane_implies_perp_line 
  (l m : Line) (α : Plane) :
  perpendicular_line_plane l α → contained_in m α → perpendicular_lines l m :=
sorry

end NUMINAMATH_CALUDE_line_perp_plane_implies_perp_line_l2907_290777


namespace NUMINAMATH_CALUDE_select_two_from_six_l2907_290718

theorem select_two_from_six (n : ℕ) (k : ℕ) : n = 6 → k = 2 → Nat.choose n k = 15 := by
  sorry

end NUMINAMATH_CALUDE_select_two_from_six_l2907_290718


namespace NUMINAMATH_CALUDE_lottery_first_prize_probability_l2907_290716

/-- The probability of winning a first prize in a lottery -/
theorem lottery_first_prize_probability
  (total_tickets : ℕ)
  (first_prizes : ℕ)
  (h_total : total_tickets = 150)
  (h_first : first_prizes = 5) :
  (first_prizes : ℚ) / total_tickets = 1 / 30 := by
  sorry

end NUMINAMATH_CALUDE_lottery_first_prize_probability_l2907_290716


namespace NUMINAMATH_CALUDE_no_nonsquare_triple_divisors_l2907_290710

theorem no_nonsquare_triple_divisors : 
  ¬ ∃ (N : ℕ+), (¬ ∃ (m : ℕ+), N = m * m) ∧ 
  (∃ (t : ℕ+), ∀ d : ℕ+, d ∣ N → ∃ (a b : ℕ+), (a ∣ N) ∧ (b ∣ N) ∧ (d * a * b = t)) :=
by sorry

end NUMINAMATH_CALUDE_no_nonsquare_triple_divisors_l2907_290710


namespace NUMINAMATH_CALUDE_system_solution_value_l2907_290792

theorem system_solution_value (a b x y : ℝ) : 
  x = 2 ∧ 
  y = 1 ∧ 
  a * x + b * y = 5 ∧ 
  b * x + a * y = 1 → 
  3 - a - b = 1 := by
sorry

end NUMINAMATH_CALUDE_system_solution_value_l2907_290792


namespace NUMINAMATH_CALUDE_molecular_weight_K2Cr2O7_is_296_l2907_290712

/-- The molecular weight of K2Cr2O7 in g/mole -/
def molecular_weight_K2Cr2O7 : ℝ := 296

/-- The number of moles given in the problem -/
def given_moles : ℝ := 4

/-- The total weight of the given moles in grams -/
def total_weight : ℝ := 1184

/-- Theorem stating that the molecular weight of K2Cr2O7 is 296 g/mole -/
theorem molecular_weight_K2Cr2O7_is_296 :
  molecular_weight_K2Cr2O7 = total_weight / given_moles :=
by sorry

end NUMINAMATH_CALUDE_molecular_weight_K2Cr2O7_is_296_l2907_290712


namespace NUMINAMATH_CALUDE_johnny_pencil_packs_l2907_290707

theorem johnny_pencil_packs :
  ∀ (total_red_pencils : ℕ) (extra_red_packs : ℕ) (extra_red_per_pack : ℕ),
    total_red_pencils = 21 →
    extra_red_packs = 3 →
    extra_red_per_pack = 2 →
    ∃ (total_packs : ℕ),
      total_packs = (total_red_pencils - extra_red_packs * extra_red_per_pack) + extra_red_packs ∧
      total_packs = 18 :=
by sorry

end NUMINAMATH_CALUDE_johnny_pencil_packs_l2907_290707


namespace NUMINAMATH_CALUDE_crates_in_load_l2907_290795

/-- Represents the weight of vegetables in a delivery truck load --/
structure VegetableLoad where
  crateWeight : ℕ     -- Weight of one crate in kilograms
  cartonWeight : ℕ    -- Weight of one carton in kilograms
  numCartons : ℕ      -- Number of cartons in the load
  totalWeight : ℕ     -- Total weight of the load in kilograms

/-- Calculates the number of crates in a vegetable load --/
def numCrates (load : VegetableLoad) : ℕ :=
  (load.totalWeight - load.cartonWeight * load.numCartons) / load.crateWeight

/-- Theorem stating that for the given conditions, the number of crates is 12 --/
theorem crates_in_load :
  ∀ (load : VegetableLoad),
    load.crateWeight = 4 →
    load.cartonWeight = 3 →
    load.numCartons = 16 →
    load.totalWeight = 96 →
    numCrates load = 12 := by
  sorry

end NUMINAMATH_CALUDE_crates_in_load_l2907_290795


namespace NUMINAMATH_CALUDE_percentage_problem_l2907_290715

theorem percentage_problem (x : ℝ) (h : (30/100) * (15/100) * x = 18) :
  (15/100) * (30/100) * x = 18 := by sorry

end NUMINAMATH_CALUDE_percentage_problem_l2907_290715


namespace NUMINAMATH_CALUDE_parallelepiped_construction_impossible_l2907_290721

/-- Represents the five shapes of blocks -/
inductive BlockShape
  | I
  | L
  | T
  | Plus
  | J

/-- Represents a parallelepiped -/
structure Parallelepiped where
  length : ℕ
  width : ℕ
  height : ℕ
  volume : ℕ

/-- Represents the construction requirements -/
structure ConstructionRequirements where
  total_blocks : ℕ
  shapes : List BlockShape
  volume : ℕ

/-- Checks if a parallelepiped satisfies the edge conditions -/
def valid_edges (p : Parallelepiped) : Prop :=
  p.length > 1 ∧ p.width > 1 ∧ p.height > 1

/-- Checks if a parallelepiped can be constructed with given requirements -/
def can_construct (p : Parallelepiped) (req : ConstructionRequirements) : Prop :=
  p.volume = req.volume ∧ valid_edges p

/-- Main theorem: Impossibility of constructing the required parallelepiped -/
theorem parallelepiped_construction_impossible (req : ConstructionRequirements) :
  req.total_blocks = 48 ∧ 
  req.shapes = [BlockShape.I, BlockShape.L, BlockShape.T, BlockShape.Plus, BlockShape.J] ∧
  req.volume = 1990 →
  ¬∃ (p : Parallelepiped), can_construct p req :=
sorry

end NUMINAMATH_CALUDE_parallelepiped_construction_impossible_l2907_290721


namespace NUMINAMATH_CALUDE_mean_median_difference_l2907_290754

/-- Represents the frequency of students for each number of days missed -/
def frequency : List (ℕ × ℕ) := [(0, 4), (1, 2), (2, 5), (3, 3), (4, 2), (5, 4)]

/-- Total number of students -/
def total_students : ℕ := 20

/-- Calculates the median of the dataset -/
def median (freq : List (ℕ × ℕ)) (total : ℕ) : ℚ := sorry

/-- Calculates the mean of the dataset -/
def mean (freq : List (ℕ × ℕ)) (total : ℕ) : ℚ := sorry

/-- The main theorem stating the difference between mean and median -/
theorem mean_median_difference :
  mean frequency total_students - median frequency total_students = 9 / 20 := by sorry

end NUMINAMATH_CALUDE_mean_median_difference_l2907_290754


namespace NUMINAMATH_CALUDE_smallest_solution_of_quadratic_l2907_290764

theorem smallest_solution_of_quadratic (y : ℝ) : 
  (3 * y^2 + 15 * y - 90 = y * (y + 20)) → y ≥ -6 :=
by
  sorry

end NUMINAMATH_CALUDE_smallest_solution_of_quadratic_l2907_290764


namespace NUMINAMATH_CALUDE_square_area_ratio_l2907_290748

theorem square_area_ratio (s : ℝ) (h : s > 0) : 
  let d := s * Real.sqrt 2
  let side_larger := 2 * d
  (side_larger ^ 2) / (s ^ 2) = 8 := by sorry

end NUMINAMATH_CALUDE_square_area_ratio_l2907_290748


namespace NUMINAMATH_CALUDE_joan_wednesday_spending_l2907_290786

/-- The number of half-dollars Joan spent on Wednesday -/
def wednesday_half_dollars : ℕ := 18 - 14

/-- The total amount Joan spent in half-dollars -/
def total_half_dollars : ℕ := 18

/-- The number of half-dollars Joan spent on Thursday -/
def thursday_half_dollars : ℕ := 14

theorem joan_wednesday_spending :
  wednesday_half_dollars = 4 :=
by sorry

end NUMINAMATH_CALUDE_joan_wednesday_spending_l2907_290786


namespace NUMINAMATH_CALUDE_simplify_fraction_l2907_290789

theorem simplify_fraction : (3^4 + 3^2) / (3^3 - 3) = 15 / 4 := by
  sorry

end NUMINAMATH_CALUDE_simplify_fraction_l2907_290789


namespace NUMINAMATH_CALUDE_initial_amounts_given_final_state_l2907_290790

/-- Represents the state of the game after each round -/
structure GameState where
  player1 : ℤ
  player2 : ℤ
  player3 : ℤ

/-- Simulates one round of the game where the specified player loses -/
def playRound (state : GameState) (loser : Fin 3) : GameState :=
  match loser with
  | 0 => ⟨state.player1 - (state.player2 + state.player3), 
          state.player2 + state.player1, 
          state.player3 + state.player1⟩
  | 1 => ⟨state.player1 + state.player2, 
          state.player2 - (state.player1 + state.player3), 
          state.player3 + state.player2⟩
  | 2 => ⟨state.player1 + state.player3, 
          state.player2 + state.player3, 
          state.player3 - (state.player1 + state.player2)⟩

/-- Theorem stating the initial amounts given the final state -/
theorem initial_amounts_given_final_state 
  (x y z : ℤ) 
  (h1 : playRound (playRound (playRound ⟨x, y, z⟩ 0) 1) 2 = ⟨104, 104, 104⟩) :
  x = 169 ∧ y = 91 ∧ z = 52 := by
  sorry


end NUMINAMATH_CALUDE_initial_amounts_given_final_state_l2907_290790
